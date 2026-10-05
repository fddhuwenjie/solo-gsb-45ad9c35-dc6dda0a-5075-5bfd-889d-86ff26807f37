"""API integration tests for the one-shot allocation + BGP plan submission."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient


RELS = [
    {"type": "customer_provider", "customer": 10, "provider": 100},
    {"type": "peer", "a": 100, "b": 50},
    {"type": "customer_provider", "customer": 60, "provider": 50},
]


def _create_pool(client, cidr="203.0.113.0/24", exclusions=None):
    resp = client.post(
        "/pools", json={"cidr": cidr, "exclusions": exclusions or []}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _allocate_plain(client, pool_id, request_id, prefix_length):
    resp = client.post(
        f"/pools/{pool_id}/allocations",
        json={"request_id": request_id, "prefix_length": prefix_length},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["cidr"]


def _body(
    request_id="order-1",
    prefix_length=25,
    plan_id="plan-1",
    relationships=None,
    announcements=None,
    new_announcement=None,
):
    return {
        "request_id": request_id,
        "prefix_length": prefix_length,
        "plan_id": plan_id,
        "relationships": RELS if relationships is None else relationships,
        "announcements": [] if announcements is None else announcements,
        "new_announcement": (
            {"id": "new", "as_path": [10, 100]}
            if new_announcement is None
            else new_announcement
        ),
    }


def _submit(client, pool_id, **overrides):
    return client.post(
        f"/pools/{pool_id}/allocation-bgp-plan", json=_body(**overrides)
    )


def _activate(client, pool_id, plan_id):
    return client.post(f"/pools/{pool_id}/bgp-plans/{plan_id}/activate")


def test_fresh_submission_allocates_and_plans_together(client):
    pid = _create_pool(client)
    resp = _submit(client, pid)
    assert resp.status_code == 201, resp.text
    data = resp.json()

    assert data["replayed"] is False
    assert data["allocation"]["cidr"] == "203.0.113.0/25"
    assert data["allocation"]["request_id"] == "order-1"
    assert data["allocation"]["prefix_length"] == 25
    assert data["plan"]["plan_id"] == "plan-1"
    assert data["plan"]["status"] == "planned"
    assert data["plan"]["replayed"] is False
    assert data["plan"]["pool_id"] == pid

    # The service completed the new announcement with the chosen CIDR.
    results = {r["id"]: r for r in data["plan"]["results"]}
    assert results["new"]["prefix"] == "203.0.113.0/25"
    assert results["new"]["as_path"] == [10, 100]
    assert results["new"]["status"] == "valid"

    # The plan is pinned to the post-commit pool state: the new allocation is
    # already part of the summary ...
    assert [a["cidr"] for a in data["plan"]["pool_state"]["allocations"]] == [
        "203.0.113.0/25"
    ]
    # ... which is why activation against the latest state succeeds.
    activated = _activate(client, pid, "plan-1")
    assert activated.status_code == 200, activated.text
    assert activated.json()["status"] == "active"
    assert activated.json()["state_digest"] == data["plan"]["state_digest"]

    # The allocation is an ordinary allocation visible to other endpoints.
    status = client.get(f"/pools/{pid}").json()
    assert [a["cidr"] for a in status["allocated"]] == ["203.0.113.0/25"]


def test_lowest_aligned_selection_with_current_allocations(client):
    pid = _create_pool(client)
    first = _allocate_plain(client, pid, "old-1", 26)  # 203.0.113.0/26
    assert first == "203.0.113.0/26"

    current = [
        {"id": "a0", "prefix": "203.0.113.0/26", "as_path": [10, 100]},
    ]
    resp = _submit(
        client,
        pid,
        prefix_length=26,
        request_id="order-1",
        announcements=current,
        new_announcement={"id": "a64", "as_path": [10, 100]},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    # Lowest aligned free /26 after .0/26 is .64/26.
    assert data["allocation"]["cidr"] == "203.0.113.64/26"
    assert data["plan"]["results"] and all(
        r["status"] == "valid" for r in data["plan"]["results"]
    )
    # Both current and new allocations appear in the pinned post-commit state.
    assert [a["cidr"] for a in data["plan"]["pool_state"]["allocations"]] == [
        "203.0.113.0/26",
        "203.0.113.64/26",
    ]


def test_replay_same_params_returns_original(client):
    pid = _create_pool(client)
    first = _submit(client, pid)
    assert first.status_code == 201
    first_data = first.json()

    second = _submit(client, pid)
    assert second.status_code == 200, second.text
    replay = second.json()
    assert replay["replayed"] is True
    assert replay["allocation"]["id"] == first_data["allocation"]["id"]
    assert replay["allocation"]["cidr"] == first_data["allocation"]["cidr"]
    assert replay["plan"]["plan_id"] == "plan-1"
    assert replay["plan"]["replayed"] is True
    assert replay["plan"]["created_at"] == first_data["plan"]["created_at"]

    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1


def test_replay_reflects_later_plan_lifecycle(client):
    pid = _create_pool(client)
    assert _submit(client, pid).status_code == 201
    assert _activate(client, pid, "plan-1").json()["status"] == "active"

    replay = _submit(client, pid)
    assert replay.status_code == 200
    assert replay.json()["plan"]["status"] == "active"
    assert replay.json()["plan"]["replayed"] is True


def test_replay_is_order_insensitive(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 26)  # 203.0.113.0/26
    current = [
        {"id": "a0", "prefix": "203.0.113.0/26", "as_path": [10, 100]},
    ]
    first = _submit(
        client,
        pid,
        prefix_length=26,
        announcements=current,
        new_announcement={"id": "a64", "as_path": [10, 100]},
    )
    assert first.status_code == 201

    # Reversed relationship order (deduped/sorted canonically) replays.
    replay = _submit(
        client,
        pid,
        prefix_length=26,
        relationships=list(reversed(RELS)),
        announcements=list(reversed(current)),
        new_announcement={"id": "a64", "as_path": [10, 100]},
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["replayed"] is True


def test_changed_new_announcement_conflicts(client):
    pid = _create_pool(client)
    assert _submit(client, pid).status_code == 201

    changed = _submit(client, pid, new_announcement={"id": "new", "as_path": [100, 50, 60]})
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "bgp_plan_id_conflict"


def test_changed_prefix_length_conflicts(client):
    pid = _create_pool(client)
    assert _submit(client, pid).status_code == 201

    changed = _submit(client, pid, prefix_length=26)
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "bgp_plan_id_conflict"
    # No second allocation slipped in.
    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1


def test_changed_plan_id_with_owned_request_id_conflicts(client):
    pid = _create_pool(client)
    assert _submit(client, pid).status_code == 201

    changed = _submit(client, pid, plan_id="plan-2")
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "request_id_conflict"


def test_request_id_taken_by_plain_allocation_cannot_be_taken_over(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "order-1", 25)

    resp = _submit(client, pid)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "request_id_conflict"
    # No plan was created.
    assert _activate(client, pid, "plan-1").status_code == 404


def test_plan_id_taken_by_standalone_plan_cannot_be_taken_over(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 25)
    created = client.post(
        f"/pools/{pid}/bgp-plans",
        json={
            "plan_id": "plan-1",
            "relationships": [
                {"type": "customer_provider", "customer": 10, "provider": 100}
            ],
            "announcements": [
                {"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]}
            ],
        },
    )
    assert created.status_code == 201

    resp = _submit(client, pid, request_id="order-1")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "bgp_plan_id_conflict"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["old-1"]


def test_plan_id_taken_by_other_one_shot_conflicts(client):
    pid = _create_pool(client)
    assert _submit(client, pid, request_id="order-1", plan_id="plan-1").status_code == 201

    hijack = _submit(client, pid, request_id="order-2", plan_id="plan-1")
    assert hijack.status_code == 409
    assert hijack.json()["error"]["code"] == "bgp_plan_id_conflict"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["order-1"]


def test_pool_exhausted_leaves_nothing(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 24)  # whole pool

    resp = _submit(client, pid)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "pool_exhausted"
    assert _activate(client, pid, "plan-1").status_code == 404
    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1


def test_invalid_path_leaves_nothing(client):
    pid = _create_pool(client)
    # No declared relationship between 10 and 999.
    resp = _submit(
        client,
        pid,
        relationships=[
            {"type": "customer_provider", "customer": 10, "provider": 100}
        ],
        new_announcement={"id": "new", "as_path": [10, 999]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "as_relation_unknown"
    assert _activate(client, pid, "plan-1").status_code == 404
    assert client.get(f"/pools/{pid}").json()["allocated"] == []


def test_duplicate_new_announcement_id_leaves_nothing(client):
    current = [
        {"id": "dup", "prefix": "203.0.113.0/25", "as_path": [10, 100]},
    ]
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 25)
    resp = _submit(
        client,
        pid,
        announcements=current,
        new_announcement={"id": "dup", "as_path": [100, 50, 60]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "duplicate_announcement_id"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["old-1"]


def test_current_allocation_uncovered_leaves_nothing(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 25)  # 203.0.113.0/25

    # No current announcement covers the existing /25; the new announcement
    # only covers the second /25 the service is about to allocate.
    resp = _submit(
        client,
        pid,
        announcements=[],
        new_announcement={"id": "a128", "as_path": [100, 50, 60]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_allocation_uncovered"
    assert _activate(client, pid, "plan-1").status_code == 404
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["old-1"]


def test_origin_conflict_on_changed_allocations_leaves_nothing(client):
    pid = _create_pool(client)
    _allocate_plain(client, pid, "old-1", 25)  # 203.0.113.0/25, origin 100
    current = [
        # A coarse /24 originating 50 covering the existing allocation ...
        {"id": "whole", "prefix": "203.0.113.0/24", "as_path": [50]},
        # ... and the exact existing /25 originating 100: already a conflict
        # before the new allocation is even considered.
    ]
    resp = _submit(
        client,
        pid,
        announcements=current
        + [{"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]}],
        new_announcement={"id": "a128", "as_path": [100, 50, 60]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_allocation_origin_conflict"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["old-1"]


def test_no_room_for_larger_prefix_leaves_nothing(client):
    pid = _create_pool(client, exclusions=["203.0.113.0/26"])
    _allocate_plain(client, pid, "old-1", 25)  # 203.0.113.128/25
    resp = _submit(
        client,
        pid,
        prefix_length=24,
        announcements=[
            {"id": "a128", "prefix": "203.0.113.128/25", "as_path": [100, 50, 60]}
        ],
        new_announcement={"id": "whole", "as_path": [10, 100]},
    )
    # No free /24 exists once a /25 is allocated from a /24 pool.
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "pool_exhausted"
    assert _activate(client, pid, "plan-1").status_code == 404


def test_current_announcement_outside_pool_leaves_nothing(client):
    # The chosen new subnet itself is always pool-scoped, but the supplied
    # current announcements are still validated against the pool.
    pid = _create_pool(client)
    resp = _submit(
        client,
        pid,
        announcements=[
            {"id": "foreign", "prefix": "198.51.100.0/24", "as_path": [50]}
        ],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_announcement_outside_pool"
    assert _activate(client, pid, "plan-1").status_code == 404
    assert client.get(f"/pools/{pid}").json()["allocated"] == []


def test_activation_conflicts_when_pool_changes_after_submission(client):
    pid = _create_pool(client)
    assert _submit(client, pid).status_code == 201

    # Another allocation appears outside the one-shot, unpinned by the plan:
    # the existing /25 announcement does not cover it.
    _allocate_plain(client, pid, "later-1", 25)
    resp = _activate(client, pid, "plan-1")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "bgp_plan_state_conflict"


def test_standalone_endpoints_still_work_alongside(client):
    pid = _create_pool(client)
    # Plain allocation.
    cidr = _allocate_plain(client, pid, "plain-1", 26)
    assert cidr == "203.0.113.0/26"
    # Standalone plan covering it.
    plan = client.post(
        f"/pools/{pid}/bgp-plans",
        json={
            "plan_id": "standalone-1",
            "relationships": [
                {"type": "customer_provider", "customer": 10, "provider": 100}
            ],
            "announcements": [
                {"id": "a0", "prefix": "203.0.113.0/26", "as_path": [10, 100]}
            ],
        },
    )
    assert plan.status_code == 201, plan.text
    assert _activate(client, pid, "standalone-1").status_code == 200


def test_missing_pool_404(client):
    resp = _submit(client, 999)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "pool_not_found"


def test_one_shot_plan_can_be_handed_over(client):
    # First one-shot plan for the first /25; activate it.
    pid = _create_pool(client)
    assert _submit(
        client, pid, request_id="order-1", plan_id="plan-1",
        new_announcement={"id": "a0", "as_path": [10, 100]},
    ).status_code == 201
    assert _activate(client, pid, "plan-1").status_code == 200

    # Second one-shot submission provisions the second /25 and a successor
    # plan; the request must name the announcement covering the old /25 too.
    resp = _submit(
        client,
        pid,
        request_id="order-2",
        plan_id="plan-2",
        announcements=[{"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]}],
        new_announcement={"id": "a128", "as_path": [100, 50, 60]},
    )
    assert resp.status_code == 201, resp.text

    handover = client.post(
        f"/pools/{pid}/bgp-plans/handover",
        json={"old_plan_ids": ["plan-1"], "new_plan_id": "plan-2"},
    )
    assert handover.status_code == 200, handover.text
    statuses = {p["plan_id"]: p["status"] for p in handover.json()["plans"]}
    assert statuses == {"plan-1": "retired", "plan-2": "active"}

    # Replaying the second one-shot reports the successor's current status.
    replay = _submit(
        client,
        pid,
        request_id="order-2",
        plan_id="plan-2",
        announcements=[{"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]}],
        new_announcement={"id": "a128", "as_path": [100, 50, 60]},
    )
    assert replay.status_code == 200
    assert replay.json()["plan"]["status"] == "active"


def test_concurrent_same_keys_provision_once(client):
    pid = _create_pool(client)
    barrier = threading.Barrier(8)
    statuses: list[int] = []
    lock = threading.Lock()

    from app.main import app

    body = _body()

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = local.post(f"/pools/{pid}/allocation-bgp-plan", json=body)
        with lock:
            statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses.count(201) == 1
    assert statuses.count(200) == 7
    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1
