"""CAM asks the operator for a capability; the operator's answer returns as an event (real environment + policy)."""
import json
import os

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService
from cam_live.harness.service import build_environment
from cam_live.harness import operator_requests as oreq


def _stack(tmp_path):
    R = str(tmp_path / "run")
    svc = PolicyService(R, fsync=False)
    env = build_environment(R, svc, secrets=svc.secrets_handle(), browser=False)
    task = C.TaskSpec(objective_id="req", objective="exercise operator requests", mode="staging", interfaces=["internal"],
                      limits=C.Limits(duration_s=3600, money=0, max_decisions=100, max_external_actions=10))
    return R, svc, env, task


def _go(env, svc, task, args):
    a = C.ActionSpec(action_id=C.new_id("a"), operation="internal.request_operator", modality=C.Modality.INTERNAL,
                     arguments=args)
    assert env.validate_action(a) == []
    return a, env.execute(a, svc.authorize(a, task, {}))


def test_request_is_recorded_deduplicated_and_answered(tmp_path):
    R, svc, env, task = _stack(tmp_path)
    assert "internal.request_operator" in {t.name for t in env.discover_tools()}
    args = {"kind": "permission", "title": "Permission to create events", "description": "scheduled events are refused",
            "justification": "three attempts failed with a missing-permission error", "blocking": False}
    a, res = _go(env, svc, task, args)
    assert res.status == C.ResultStatus.OK, res.failure_detail
    d = oreq.find_file(R)
    rows = oreq.OperatorRequests(d).open_requests()
    assert len(rows) == 1 and rows[0]["kind"] == "permission" and rows[0]["action_id"] == a.action_id
    _, res2 = _go(env, svc, task, dict(args, title="permission to CREATE events!"))        # same request, reworded case
    assert res2.status == C.ResultStatus.OK and len(oreq.OperatorRequests(d).open_requests()) == 1
    assert "already_open" in res2.observations[0].text
    events, cursor = env.poll_events("")
    assert not [e for e in events if e.data.get("event") == "operator.answer"]
    rid = rows[0]["request_id"]
    assert oreq.main(["--run-dir", R, "grant", rid, "--note", "role updated in the server settings"]) == 0
    assert oreq.main(["--run-dir", R, "grant", rid]) == 2                                   # already resolved
    events, cursor = env.poll_events(cursor)
    ans = [e for e in events if e.data.get("event") == "operator.answer"]
    assert len(ans) == 1 and ans[0].provenance.trust == C.Trust.OPERATOR
    assert "granted" in ans[0].text and "role updated" in ans[0].text and ans[0].data["request_id"] == rid
    events, cursor = env.poll_events(cursor)                                                # delivered exactly once
    assert not [e for e in events if e.data.get("event") == "operator.answer"]
    assert oreq.OperatorRequests(d).open_requests() == []


def test_invalid_kind_and_request_flood_are_refused(tmp_path):
    R, svc, env, task = _stack(tmp_path)
    bad = C.ActionSpec(action_id=C.new_id("a"), operation="internal.request_operator", modality=C.Modality.INTERNAL,
                       arguments={"kind": "money_printer", "title": "abc", "description": "abc"})
    assert env.validate_action(bad)
    reqs = oreq.OperatorRequests(str(tmp_path / "x"), max_open=3)
    for i in range(3):
        reqs.request("resource", f"thing {i}", "d")
    with pytest.raises(PermissionError):
        reqs.request("resource", "thing 99", "d")
    with pytest.raises(ValueError):
        reqs.request("bogus", "t", "d")
    r = reqs.resolve(reqs.all()[0]["request_id"], "denied", "not available")
    assert r["status"] == "denied" and len(reqs.open_requests()) == 2
    assert [x["status"] for x in reqs.undelivered_answers()] == ["denied"] and reqs.undelivered_answers() == []
    lines = [json.loads(x) for x in open(os.path.join(str(tmp_path / "x"), oreq.FILE))]
    assert len(lines) == 5                                                                  # append-only revisions
