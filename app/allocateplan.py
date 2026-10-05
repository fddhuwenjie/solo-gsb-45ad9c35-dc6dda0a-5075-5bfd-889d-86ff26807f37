"""One-shot address provisioning plus a confirmable BGP announcement plan.

``POST /pools/{id}/allocation-bgp-plan`` combines what otherwise needs the
standalone allocation endpoint plus the plan-creation endpoint: the caller
submits a ``request_id``, ``prefix_length``, ``plan_id``, the AS
``relationships`` and the ``announcements`` covering the pool's *current*
allocations, together with a ``new_announcement`` carrying only an ``id`` and
an ``as_path``.  The service chooses the subnet itself and completes the new
announcement's prefix with the chosen CIDR.

Everything -- idempotency checks, free-space selection, the path audit, the
coverage/origin checks and every insert -- happens inside one
``BEGIN IMMEDIATE`` write transaction:

* the lowest-address aligned free subnet outside exclusions and existing
  allocations is picked with the same rule as a plain allocation; lack of
  space fails with ``409 pool_exhausted``;
* the new announcement is completed with the chosen CIDR, then the *whole*
  changed allocation set (current allocations plus the candidate one) is
  audited with the same stateless path audit and checked against the exact
  plan coverage rules (inside pool, no exclusion overlap, every allocation
  covered by a ``valid`` announcement, one origin per allocation);
* only when every check passes are the ordinary allocation row and the
  immutable ``planned`` BGP plan -- pinned to the post-commit pool-state
  summary -- written together.

Insufficient space, an announcement that fails the path audit or a coverage
conflict rolls the transaction back: neither an allocation nor a plan is left
behind.  The stored plan is an ordinary ``bgp_plans`` row, so activation still
goes through ``/pools/{id}/bgp-plans/{plan_id}/activate`` and re-verifies the
latest pool state.

Both idempotency keys are owned by this one operation while it is fresh:
retrying with the same ``plan_id`` and the same request parameters replays the
original allocation and plan, changing any parameter conflicts, and a
``request_id`` or ``plan_id`` already occupied by another operation (a plain
allocation, a reservation, or a standalone plan) cannot be taken over.
"""

from __future__ import annotations

import json
import sqlite3

from .bgpaudit import BgpAuditError, audit
from .bgpplan import (
    _canonical_params,
    _response,
    _state_digest,
    _validate_coverage,
    _pool_state,
)
from .cidrmath import compute_free, lowest_aligned, parse_network
from .db import write_transaction
from .schemas import BgpAnnouncement
from .service import (
    ApiError,
    _pool_allocations,
    _pool_exclusions,
    _row_pool,
    utc_now,
)


def _request_fingerprint(pool_id: int, body) -> str:
    """Canonical fingerprint of the client-supplied request parameters.

    The new announcement's selected CIDR is derived server-side, not a request
    parameter, so it never participates in conflict detection.  Relationships
    and the current announcements use the same order-insensitive canonical
    form as standalone plan creation; the new announcement's submitted path
    is kept verbatim (with prepending) just like every other announcement.
    """

    current = _canonical_params(body.relationships, body.announcements)
    params = {
        "pool_id": pool_id,
        "request_id": body.request_id,
        "prefix_length": body.prefix_length,
        "plan_id": body.plan_id,
        "relationships": current["relationships"],
        "announcements": current["announcements"],
        "new_announcement": [
            body.new_announcement.id,
            list(body.new_announcement.as_path),
        ],
    }
    return json.dumps(params, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def allocate_with_bgp_plan(pool_id: int, body) -> dict:
    """Allocate one subnet and record its planned BGP announcements atomically."""

    request_id = body.request_id
    plan_id = body.plan_id

    # Parse current announcement prefixes up-front so the canonical fingerprint
    # below cannot fail on malformed input (the full audit with the completed
    # announcement runs inside the transaction).
    for ann in body.announcements:
        try:
            parse_network(ann.prefix)
        except ValueError as exc:
            raise ApiError(
                422,
                f"announcement {ann.id!r} has invalid prefix {ann.prefix!r}: {exc}",
                code="invalid_prefix",
            )

    params_json = _request_fingerprint(pool_id, body)

    with write_transaction() as conn:
        _row_pool(conn, pool_id)

        # --- Idempotency / ownership checks under the write lock ----------
        own_record = conn.execute(
            "SELECT * FROM allocation_bgp_plan_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        plan_record = conn.execute(
            "SELECT * FROM allocation_bgp_plan_requests WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()

        if own_record is not None:
            same_plan = own_record["plan_id"] == plan_id
            same_params = own_record["params"] == params_json
            plan_row = conn.execute(
                "SELECT * FROM bgp_plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if same_plan and same_params:
                if plan_row is None:
                    # Defensive: the allocation_bgp_plan_requests row has no
                    # cascade to bgp_plans and both are written together, so
                    # this can only happen after direct database tampering.
                    raise ApiError(
                        409,
                        f"stored plan {plan_id!r} for request_id {request_id!r} "
                        "is missing; the idempotency record cannot be replayed",
                        code="bgp_plan_id_conflict",
                    )
                result = json.loads(own_record["result"])
                return {
                    "pool_id": pool_id,
                    "allocation": {
                        "id": result["allocation_id"],
                        "pool_id": pool_id,
                        "cidr": result["cidr"],
                        "request_id": request_id,
                        "prefix_length": result["prefix_length"],
                        "created_at": result["created_at"],
                    },
                    # The plan keeps living its normal lifecycle after the
                    # one-shot commit, so a replay reports its current status.
                    "plan": _response(plan_row, replayed=True),
                    "replayed": True,
                }
            if not same_plan:
                raise ApiError(
                    409,
                    f"request_id {request_id!r} was already used by an "
                    "allocation-with-plan submission carrying plan_id "
                    f"{own_record['plan_id']!r}, not {plan_id!r}",
                    code="request_id_conflict",
                )
            raise ApiError(
                409,
                f"plan_id {plan_id!r} and request_id {request_id!r} were already "
                "used with different parameters",
                code="bgp_plan_id_conflict",
            )

        # The request_id keys another one-shot operation under a different plan.
        if plan_record is not None:
            raise ApiError(
                409,
                f"plan_id {plan_id!r} was already used by an "
                "allocation-with-plan submission for request_id "
                f"{plan_record['request_id']!r}",
                code="bgp_plan_id_conflict",
            )

        # Neither key may be taken over from a different operation: a plan
        # created through the standalone endpoint, or an allocation made via
        # plain allocation, a reservation batch or a cross-pool migration.
        standalone_plan = conn.execute(
            "SELECT pool_id FROM bgp_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if standalone_plan is not None:
            raise ApiError(
                409,
                f"plan_id {plan_id!r} already exists"
                + (
                    f" in pool {standalone_plan['pool_id']}"
                    if standalone_plan["pool_id"] != pool_id
                    else ""
                )
                + "; it cannot be reused by an allocation-with-plan submission",
                code="bgp_plan_id_conflict",
            )
        standalone_allocation = conn.execute(
            "SELECT pool_id FROM allocations WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if standalone_allocation is not None:
            raise ApiError(
                409,
                f"request_id {request_id!r} is already used by an allocation"
                f" in pool {standalone_allocation['pool_id']}; it cannot be "
                "reused by an allocation-with-plan submission",
                code="request_id_conflict",
            )

        # --- Selection, completion, audit and coverage --------------------
        pool = _row_pool(conn, pool_id)
        pool_net = parse_network(pool["cidr"])
        exclusions = [parse_network(r["cidr"]) for r in _pool_exclusions(conn, pool_id)]
        allocation_rows = _pool_allocations(conn, pool_id)
        allocations = [parse_network(r["cidr"]) for r in allocation_rows]

        free = compute_free(pool_net, exclusions + allocations)
        chosen = lowest_aligned(free, body.prefix_length)
        if chosen is None:
            raise ApiError(
                409,
                f"no contiguous free /{body.prefix_length} subnet in pool {pool_net}",
                code="pool_exhausted",
            )

        completed = BgpAnnouncement(
            id=body.new_announcement.id,
            prefix=str(chosen),
            as_path=list(body.new_announcement.as_path),
        )
        all_announcements = [*body.announcements, completed]

        # Existing path audit over relationships + the completed announcement
        # set; relation conflicts, unknown edges, loops, bad prefixes, empty
        # paths and duplicate ids surface as 422 exactly like the audit API.
        try:
            results = audit(body.relationships, all_announcements)["results"]
        except BgpAuditError as exc:
            raise ApiError(422, exc.detail, code=exc.code)

        # Existing plan coverage rules against the *changed* allocation set:
        # current allocations plus the candidate subnet, identified by its
        # request_id in any error message.  A dict supports the row["cidr"] /
        # row["request_id"] access the validator performs.
        changed_rows = [
            *allocation_rows,
            {"cidr": str(chosen), "request_id": request_id},
        ]
        _validate_coverage(results, pool_net, exclusions, changed_rows)

        # --- All checks passed: write both, pinned to post-commit state ----
        now = utc_now()
        try:
            cursor = conn.execute(
                "INSERT INTO allocations "
                "(pool_id, cidr, request_id, prefix_length, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (pool_id, str(chosen), request_id, body.prefix_length, now),
            )
        except sqlite3.IntegrityError:
            raise ApiError(
                409,
                f"request_id {request_id!r} conflicts with an existing allocation",
                code="request_id_conflict",
            )
        allocation_id = cursor.lastrowid

        # Re-read inside the same transaction so the pinned summary describes
        # exactly the committed rows (the new allocation carries created_at).
        pool_state = _pool_state(conn, pool_id)
        state_digest = _state_digest(pool_state)
        plan_params = json.dumps(_canonical_params(body.relationships, all_announcements))

        try:
            plan_cursor = conn.execute(
                "INSERT INTO bgp_plans "
                "(plan_id, pool_id, params, audit_results, pool_state, state_digest, "
                "status, created_at, activated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, NULL)",
                (
                    plan_id,
                    pool_id,
                    plan_params,
                    json.dumps(results),
                    json.dumps(pool_state),
                    state_digest,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            raise ApiError(
                409,
                f"plan_id {plan_id!r} conflicts with an existing plan",
                code="bgp_plan_id_conflict",
            )

        conn.execute(
            "INSERT INTO allocation_bgp_plan_requests "
            "(request_id, plan_id, pool_id, params, result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                request_id,
                plan_id,
                pool_id,
                params_json,
                json.dumps(
                    {
                        "allocation_id": allocation_id,
                        "cidr": str(chosen),
                        "prefix_length": body.prefix_length,
                        "created_at": now,
                    }
                ),
                now,
            ),
        )

        plan_row = conn.execute(
            "SELECT * FROM bgp_plans WHERE rowid = ?",
            (plan_cursor.lastrowid,),
        ).fetchone()

    return {
        "pool_id": pool_id,
        "allocation": {
            "id": allocation_id,
            "pool_id": pool_id,
            "cidr": str(chosen),
            "request_id": request_id,
            "prefix_length": body.prefix_length,
            "created_at": now,
        },
        "plan": _response(plan_row, replayed=False),
        "replayed": False,
    }
