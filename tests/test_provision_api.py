"""API integration tests for one-shot address provisioning with a BGP plan."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

RELATIONSHIPS = [
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


def _allocate(client, pool_id, request_id, prefix_length):
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
    new_id="new",
    new_path=(10, 100),
):
    return {
        "request_id": request_id,
        "prefix_length": prefix_length,
        "plan_id": plan_id,
        "relationships": RELATIONSHIPS if relationships is None else relationships,
        "announcements": [] if announcements is None else announcements,
        "new_announcement": {"id": new_id, "as_path": list(new_path)},
    }


def _provision(client, pool_id, body):
    return client.post(f"/pools/{pool_id}/provisions", json=body)


def test_provision_allocates_and_plans_in_one_call(client):
    pid = _create_pool(client)
    resp = _provision(client, pid, _body())
    assert resp.status_code == 201, resp.text
    data = resp.json()

    assert data["replayed"] is False
    alloc = data["allocation"]
    assert alloc["cidr"] == "203.0.113.0/25"
    assert alloc["request_id"] == "order-1"
    assert alloc["prefix_length"] == 25
    assert alloc["replayed"] is False

    plan = data["plan"]
    assert plan["plan_id"] == "plan-1"
    assert plan["pool_id"] == pid
    assert plan["status"] == "planned"
    assert plan["activated_at"] is None
    assert plan["replayed"] is False
    assert [a["cidr"] for a in plan["pool_state"]["allocations"]] == [
        "203.0.113.0/25"
    ]
    assert len(plan["state_digest"]) == 64

    new_ann = data["new_announcement"]
    assert new_ann["id"] == "new"
    assert new_ann["prefix"] == "203.0.113.0/25"
    assert new_ann["as_path"] == [10, 100]
    assert new_ann["origin"] == 100
    assert new_ann["status"] == "valid"

    # The allocation is a regular allocation, visible to the status endpoint.
    status = client.get(f"/pools/{pid}").json()
    assert [a["cidr"] for a in status["allocated"]] == ["203.0.113.0/25"]
    assert status["free_cidrs"] == ["203.0.113.128/25"]


def test_provision_covers_existing_allocations(client):
    pid = _create_pool(client)
    _allocate(client, pid, "existing", 25)  # 203.0.113.0/25
    body = _body(
        request_id="order-2",
        announcements=[
            {"id": "cur", "prefix": "203.0.113.0/25", "as_path": [10, 100]}
        ],
        new_path=(100, 50, 60),
    )
    resp = _provision(client, pid, body)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["allocation"]["cidr"] == "203.0.113.128/25"
    assert data["new_announcement"]["prefix"] == "203.0.113.128/25"
    assert data["new_announcement"]["origin"] == 60
    # The stored plan pins both allocations.
    assert [
        a["cidr"] for a in data["plan"]["pool_state"]["allocations"]
    ] == ["203.0.113.0/25", "203.0.113.128/25"]
    assert {r["id"] for r in data["plan"]["results"]} == {"cur", "new"}


def test_provision_selects_lowest_aligned_free(client):
    pid = _create_pool(client)
    # Occupy 203.0.113.0/26: the lowest aligned /25 then starts at .128
    # (the .64/26 remainder is not /25-aligned).
    _allocate(client, pid, "a", 26)
    body = _body(
        request_id="b",
        prefix_length=25,
        announcements=[
            {"id": "cur", "prefix": "203.0.113.0/26", "as_path": [10, 100]}
        ],
        new_path=(100, 50, 60),
    )
    resp = _provision(client, pid, body)
    assert resp.status_code == 201, resp.text
    assert resp.json()["allocation"]["cidr"] == "203.0.113.128/25"


def test_provision_replay_same_params_returns_original(client):
    pid = _create_pool(client)
    first = _provision(client, pid, _body())
    assert first.status_code == 201
    second = _provision(client, pid, _body())
    assert second.status_code == 200, second.text
    replay = second.json()
    assert replay["replayed"] is True
    assert replay["allocation"]["replayed"] is True
    assert replay["plan"]["replayed"] is True
    assert replay["allocation"]["id"] == first.json()["allocation"]["id"]
    assert replay["allocation"]["cidr"] == first.json()["allocation"]["cidr"]
    assert replay["plan"]["created_at"] == first.json()["plan"]["created_at"]

    # Only one allocation and one plan exist.
    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1


def test_provision_replay_after_activation_reflects_active(client):
    pid = _create_pool(client)
    _provision(client, pid, _body())
    activated = client.post(f"/pools/{pid}/bgp-plans/plan-1/activate")
    assert activated.status_code == 200, activated.text
    replay = _provision(client, pid, _body())
    assert replay.status_code == 200
    assert replay.json()["plan"]["status"] == "active"
    assert replay.json()["plan"]["activated_at"] is not None


def test_provision_replay_changed_allocation_params_conflicts(client):
    pid = _create_pool(client)
    assert _provision(client, pid, _body()).status_code == 201
    changed = _body(prefix_length=26)
    resp = _provision(client, pid, changed)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "provision_id_conflict"


def test_provision_replay_changed_plan_params_conflicts(client):
    pid = _create_pool(client)
    assert _provision(client, pid, _body()).status_code == 201
    changed = _body(new_path=(100, 50, 60))
    resp = _provision(client, pid, changed)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "provision_id_conflict"


def test_provision_replay_changed_request_id_conflicts(client):
    pid = _create_pool(client)
    assert _provision(client, pid, _body()).status_code == 201
    changed = _body(request_id="other-order")
    resp = _provision(client, pid, changed)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "provision_id_conflict"


def test_request_id_taken_by_standalone_allocation_cannot_be_taken_over(client):
    pid = _create_pool(client)
    _allocate(client, pid, "order-1", 25)
    resp = _provision(client, pid, _body())
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "request_id_conflict"
    # No plan was left behind.
    plans = client.get(f"/pools/{pid}").json()
    assert len(plans["allocated"]) == 1


def test_plan_id_taken_by_standalone_plan_cannot_be_taken_over(client):
    pid = _create_pool(client)
    _allocate(client, pid, "existing", 25)
    plan_body = {
        "plan_id": "plan-1",
        "relationships": [
            {"type": "customer_provider", "customer": 10, "provider": 100}
        ],
        "announcements": [
            {"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]}
        ],
    }
    assert client.post(f"/pools/{pid}/bgp-plans", json=plan_body).status_code == 201

    resp = _provision(
        client,
        pid,
        _body(
            request_id="order-1",
            relationships=[{"type": "customer_provider", "customer": 10, "provider": 100}],
            new_path=(10, 100),
        ),
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "bgp_plan_id_conflict"
    # The allocation was rolled back too.
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["existing"]


def test_provision_pool_exhausted_leaves_nothing(client):
    pid = _create_pool(client, cidr="203.0.113.0/25")
    _allocate(client, pid, "full", 25)
    resp = _provision(client, pid, _body(request_id="order-1"))
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "pool_exhausted"
    # Neither the allocation nor the plan exists; keys are reusable on a
    # fresh pool.
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["full"]
    other = _create_pool(client, cidr="198.51.100.0/25")
    assert _provision(client, other, _body(request_id="order-1")).status_code == 201


def test_provision_invalid_new_announcement_leaves_nothing(client):
    pid = _create_pool(client)
    # Empty AS path -> request validation 422.
    body = _body()
    body["new_announcement"] = {"id": "new", "as_path": []}
    resp = _provision(client, pid, body)
    assert resp.status_code == 422

    # Unknown relationship edge in the completed path -> 422, no partial state.
    resp = _provision(client, pid, _body(new_path=(10, 200)))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "as_relation_unknown"
    status = client.get(f"/pools/{pid}").json()
    assert status["allocated"] == []
    # Keys are not occupied.
    assert _provision(client, pid, _body()).status_code == 201


def test_provision_new_announcement_id_duplicate_is_422(client):
    pid = _create_pool(client)
    body = _body(
        announcements=[
            {"id": "dup", "prefix": "203.0.113.128/25", "as_path": [100, 50, 60]}
        ],
        new_id="dup",
        new_path=(10, 100),
    )
    resp = _provision(client, pid, body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "duplicate_announcement_id"
    assert client.get(f"/pools/{pid}").json()["allocated"] == []


def test_provision_origin_conflict_rolls_back(client):
    pid = _create_pool(client)
    _allocate(client, pid, "existing", 25)  # 203.0.113.0/25
    # Existing announcement origin 100 covers .0/25; the new /25 announcement
    # is valid for .128/25 -- both allocations covered by distinct origins is
    # fine.  Force a conflict by also covering the new block with origin 100
    # while the new announcement has origin 60: the new allocation then has
    # two covering announcements with different origins.
    body = _body(
        request_id="order-2",
        announcements=[
            {"id": "cur0", "prefix": "203.0.113.0/25", "as_path": [10, 100]},
            {"id": "whole", "prefix": "203.0.113.0/24", "as_path": [10, 100]},
        ],
        new_path=(100, 50, 60),
    )
    resp = _provision(client, pid, body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_allocation_origin_conflict"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["existing"]


def test_provision_uncovered_existing_allocation_rolls_back(client):
    pid = _create_pool(client)
    _allocate(client, pid, "existing", 25)  # 203.0.113.0/25
    # No current announcement covers .0/25; the new announcement only covers
    # the newly allocated .128/25.
    resp = _provision(client, pid, _body(request_id="order-2", new_path=(100, 50, 60)))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_allocation_uncovered"
    status = client.get(f"/pools/{pid}").json()
    assert [a["request_id"] for a in status["allocated"]] == ["existing"]


def test_provision_announcement_overlapping_exclusion_rolls_back(client):
    pid = _create_pool(client, exclusions=["203.0.113.0/26"])
    body = _body(
        announcements=[
            {"id": "bad", "prefix": "203.0.113.0/26", "as_path": [10, 100]}
        ],
        new_path=(10, 100),
    )
    resp = _provision(client, pid, body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bgp_announcement_overlaps_exclusion"
    assert client.get(f"/pools/{pid}").json()["allocated"] == []


def test_provision_created_plan_activates_with_fresh_state(client):
    pid = _create_pool(client)
    _provision(client, pid, _body())
    resp = client.post(f"/pools/{pid}/bgp-plans/plan-1/activate")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "active"


def test_provision_created_plan_blocks_conflicting_later_allocation_activation(client):
    pid = _create_pool(client)
    _provision(client, pid, _body())
    # Pool drifts after provisioning: the pinned plan must fail activation.
    _allocate(client, pid, "later", 25)
    resp = client.post(f"/pools/{pid}/bgp-plans/plan-1/activate")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "bgp_plan_state_conflict"


def test_provision_standalone_endpoints_still_work_alongside(client):
    pid = _create_pool(client)
    _provision(client, pid, _body())

    # The provisioned plan activates normally while the pool is still fresh.
    assert client.post(
        f"/pools/{pid}/bgp-plans/plan-1/activate"
    ).status_code == 200

    # Standalone allocation into the remaining /25.
    cidr = _allocate(client, pid, "standalone", 25)
    assert cidr == "203.0.113.128/25"

    # A standalone plan covering both allocations can be created (different
    # plan_id) and handed over from the active provisioned plan.
    plan_body = {
        "plan_id": "plan-2",
        "relationships": [
            {"type": "customer_provider", "customer": 10, "provider": 100},
            {"type": "peer", "a": 100, "b": 50},
            {"type": "customer_provider", "customer": 60, "provider": 50},
        ],
        "announcements": [
            {"id": "a0", "prefix": "203.0.113.0/25", "as_path": [10, 100]},
            {"id": "a128", "prefix": "203.0.113.128/25", "as_path": [100, 50, 60]},
        ],
    }
    assert client.post(f"/pools/{pid}/bgp-plans", json=plan_body).status_code == 201
    handover = client.post(
        f"/pools/{pid}/bgp-plans/handover",
        json={"old_plan_ids": ["plan-1"], "new_plan_id": "plan-2"},
    )
    assert handover.status_code == 200, handover.text
    statuses = {p["plan_id"]: p["status"] for p in handover.json()["plans"]}
    assert statuses == {"plan-1": "retired", "plan-2": "active"}

    # The provision replay now surfaces the retired status.
    replay = _provision(client, pid, _body())
    assert replay.status_code == 200
    assert replay.json()["plan"]["status"] == "retired"


def test_provision_unknown_pool_is_404(client):
    resp = _provision(client, 999, _body())
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "pool_not_found"


def test_concurrent_provisions_are_serialised_and_non_overlapping(client):
    pid = _create_pool(client)
    cidrs: list[str] = []
    errors: list[int] = []
    barrier = threading.Barrier(16)

    from app.main import app

    def worker(i: int):
        local = TestClient(app)
        barrier.wait()  # maximise contention
        # Every announcement originates the whole pool, so a provision
        # committing after another one still covers every allocation.
        resp = local.post(
            f"/pools/{pid}/provisions",
            json={
                "request_id": f"order-{i}",
                "prefix_length": 28,
                "plan_id": f"plan-{i}",
                "relationships": [
                    {"type": "customer_provider", "customer": 10, "provider": 100}
                ],
                "announcements": [
                    {"id": "pool", "prefix": "203.0.113.0/24", "as_path": [10, 100]}
                ],
                "new_announcement": {"id": f"new-{i}", "as_path": [10, 100]},
            },
        )
        if resp.status_code == 201:
            cidrs.append(resp.json()["allocation"]["cidr"])
        else:
            errors.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # A /24 holds exactly 16 aligned /28s, and each full-pool announcement
    # keeps coverage valid, so every provision commits.
    assert errors == []
    assert len(cidrs) == 16
    assert len(set(cidrs)) == 16


def test_concurrent_same_plan_id_provisions_commit_once(client):
    pid = _create_pool(client)
    statuses: list[int] = []
    cidrs: list[str] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    from app.main import app

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = local.post(
            f"/pools/{pid}/provisions",
            json={
                "request_id": "order-1",
                "prefix_length": 25,
                "plan_id": "plan-1",
                "relationships": [
                    {"type": "customer_provider", "customer": 10, "provider": 100}
                ],
                "announcements": [],
                "new_announcement": {"id": "new", "as_path": [10, 100]},
            },
        )
        with lock:
            statuses.append(resp.status_code)
            if resp.status_code in (200, 201):
                cidrs.append(resp.json()["allocation"]["cidr"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one real commit; all other writers replay it (200).
    assert statuses.count(201) == 1
    assert sorted(set(statuses)) == [200, 201]
    assert set(cidrs) == {"203.0.113.0/25"}
    status = client.get(f"/pools/{pid}").json()
    assert len(status["allocated"]) == 1
