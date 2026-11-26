"""The independent membership verifier, its signatures, its log, and the verifier service."""
import json
import os
import stat

import pytest

from cam_live.contracts import GoalStatus, OutcomeQuery, OutcomeStatus, SimClock, VerifiedOutcome
from cam_live.runtime import tasks as T
from cam_live.runtime.verifier import (DAY, HashChainLog, JsonlMembershipSource, JsonMetricSource, MemberRecord,
                                       MembershipVerifier, binom_quantile, build_verifier, load_or_create_key)
from cam_live.runtime.verifier_service import RemoteVerifier, serve

TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tasks")


class Source:
    name, live = "test-members", False

    def __init__(self, members, participation=None, feedback=None):
        self.members, self._p, self._f = members, participation or {}, feedback or []

    def list_members(self):
        return list(self.members)

    def participation(self):
        return dict(self._p)

    def feedback(self):
        return list(self._f)


def _task(target=100, **excl):
    raw = T.load_task_raw(os.path.join(TASKS, "discord_100.yaml"))
    t = T.parse_task(raw)
    t.success["target"] = target
    t.success["exclusions"]["test_account_ids"] = ["op-1", "op-2"]
    t.success["exclusions"].update(excl)
    t.limits.duration_s = 30 * DAY
    return t


def _humans(n, t0, prefix="h"):
    return [MemberRecord(id=f"{prefix}{i}", joined_at=t0 + i) for i in range(n)]


def test_exclusions_are_applied_and_counted(tmp_path):
    clock = SimClock()
    t0 = clock.now()
    members = _humans(40, t0) + [
        MemberRecord("bot1", is_bot=True, joined_at=t0), MemberRecord("bot2", is_bot=True, joined_at=t0),
        MemberRecord("op-1", joined_at=t0), MemberRecord("op-2", joined_at=t0),          # operator-declared test accounts
        MemberRecord("h3", joined_at=t0 + 500),                                         # the same account listed twice
        MemberRecord("alt-a", joined_at=t0 + 1, link_key="person-9"),
        MemberRecord("alt-b", joined_at=t0 + 2, link_key="person-9"),                   # identifiable second account
        MemberRecord("bought", joined_at=t0, flags=["purchased"]),
        MemberRecord("fake", joined_at=t0, flags=["artificial"]),
        MemberRecord("gone", joined_at=t0, left_at=t0 + 10),
    ]
    clock.advance(100)
    v = MembershipVerifier(Source(members), clock, str(tmp_path / "v"))
    o = v.verify(_task(target=41))
    assert o.value == 41 and o.status == GoalStatus.ACHIEVED          # 40 humans + the first of the linked pair
    assert o.excluded == {"artificial": 1, "bots": 2, "departed": 1, "duplicate_accounts": 1, "duplicates": 1,
                          "purchased": 1, "test_accounts": 2}
    assert o.uncertainty == 0 and v.check_signature(o)
    assert any(r.startswith("source:test-members") for r in o.evidence_refs)
    # member identifiers never appear in the outcome or in the verifier log
    blob = o.to_json() + open(v.log.path).read()
    assert "alt-a" not in blob and "bought" not in blob and "h17" not in blob


def test_uncertain_authenticity_is_reported_and_success_is_not_claimed(tmp_path):
    clock = SimClock()
    t0 = clock.now()
    members = (_humans(90, t0) + [MemberRecord(f"p{i}", joined_at=t0, pending=True) for i in range(8)]
               + [MemberRecord(f"s{i}", joined_at=t0, flags=["automation_suspected"]) for i in range(5)]
               + [MemberRecord(f"i{i}", joined_at=t0, invite_code="bulk-1") for i in range(4)])
    clock.advance(10)
    v = MembershipVerifier(Source(members), clock, str(tmp_path / "v"))
    task = _task(target=100, suspect_invite_codes=["bulk-1"])
    o = v.verify(task)
    # 107 accounts are listed, but only 90 are established: the verifier reports uncertainty, not success
    assert o.status == GoalStatus.PENDING and o.value == 90 and o.uncertainty == pytest.approx(17 / 2)
    assert "not established with the required confidence" in o.notes and "success is not claimed" in o.notes
    assert "pending_screening 8" in o.notes and "automation_suspected 5" in o.notes and "purchase_indicator 4" in o.notes
    assert v.check_signature(o)
    # even a favourable prior about uncertain accounts is not enough unless the lower bound reaches the target
    o2 = v.verify(_task(target=100, suspect_invite_codes=["bulk-1"], uncertain_p_range=[0.5, 1.0]))
    assert o2.status == GoalStatus.PENDING and 90 + binom_quantile(0.05, 17, 0.5) < 100
    # once enough accounts are established it is achieved
    members += _humans(10, t0, "late")
    assert v.verify(task).status == GoalStatus.ACHIEVED
    # after the window: failed only if even the upper bound is below the target, otherwise undetermined
    clock.advance(31 * DAY)
    v2 = MembershipVerifier(Source(members[:95]), clock, str(tmp_path / "v2"))
    task2 = _task(target=100)
    task2.measurement["deadline_ts"] = clock.now() - 1
    o3 = v2.verify(task2)                                            # 90 confirmed + 5 pending: upper bound 95 < 100
    assert o3.status == GoalStatus.FAILED
    o4 = MembershipVerifier(Source(members[:103]), clock, str(tmp_path / "v3")).verify(task2)
    assert o4.status == GoalStatus.PENDING and "operator review is required" in o4.notes


def test_secondary_metrics_are_separate_quantities(tmp_path):
    clock = SimClock()
    t0 = clock.now()
    members = [MemberRecord("a", joined_at=t0), MemberRecord("b", joined_at=t0), MemberRecord("c", joined_at=t0),
               MemberRecord("d", joined_at=t0, left_at=t0 + 2 * DAY),           # left inside the retention window
               MemberRecord("new", joined_at=t0 + 9 * DAY), MemberRecord("bot", is_bot=True, joined_at=t0)]
    src = Source(members, participation={"a": 5, "b": 0, "bot": 99, "new": 1},
                 feedback=[{"member_id": "a", "consented": True}, {"member_id": "b", "consented": False},
                           {"member_id": "bot", "consented": True}])
    clock.advance(10 * DAY)
    v = MembershipVerifier(src, clock, str(tmp_path / "v"), cost_source=lambda: {"money": 12.5, "currency": "USD"})
    o = v.verify(_task(target=100))
    sec = {q.name: q for q in o.secondary}
    assert sec["secondary.retention_7d"].value == pytest.approx(3 / 4) and sec["secondary.retention_7d"].exposure == 4
    assert sec["secondary.participation.members"].value == 2          # a and new; the bot's activity is not counted
    assert sec["secondary.participation.interactions"].value == 6
    assert sec["secondary.feedback.consented"].value == 1             # only consented feedback of counted members
    assert sec["secondary.operating_cost"].value == 12.5 and sec["secondary.operating_cost"].unit == "USD"
    assert sec["secondary.time_elapsed_s"].value >= 0
    # the primary value is the confirmed member count and nothing else
    assert o.value == 4 and o.status == GoalStatus.PENDING
    # verified measurements of matured queries
    v2 = build_verifier(_task(target=100), str(tmp_path / "v2"), clock, src)
    now = clock.now()
    qs = [OutcomeQuery("q1", "membership.confirmed", issued_at=now - 10, horizon_s=5, goal_relevant=True),
          OutcomeQuery("q2", "secondary.retention_7d", issued_at=now - 10, horizon_s=5),
          OutcomeQuery("q3", "membership.confirmed", issued_at=now, horizon_s=3600),
          OutcomeQuery("q4", "something.else", issued_at=now - 10, horizon_s=5)]
    got = {x.query_id: x for x in v2.measure(qs)}
    assert got["q1"].status == OutcomeStatus.OBSERVED and got["q1"].value == 4 and got["q1"].verified
    assert got["q2"].value == pytest.approx(0.75) and got["q3"].status == OutcomeStatus.PENDING
    assert got["q4"].status == OutcomeStatus.UNOBSERVABLE and not got["q4"].verified


def test_signature_key_file_and_hash_chained_log(tmp_path):
    clock = SimClock()
    v = MembershipVerifier(Source(_humans(3, clock.now())), clock, str(tmp_path / "v"))
    key = os.path.join(str(tmp_path / "v"), "verifier.key")
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600 and stat.S_IMODE(os.stat(str(tmp_path / "v")).st_mode) == 0o700
    task = _task(target=3)
    o = v.verify(task)
    assert o.status == GoalStatus.ACHIEVED and v.check_signature(o)
    # any change to a signed outcome breaks the signature; so does a missing or foreign signature
    for change in ({"value": 4.0}, {"status": GoalStatus.FAILED}, {"target": 1.0}, {"excluded": {"bots": 9}},
                   {"signature": ""}, {"signature": "0" * 64}):
        d = o.to_dict()
        d.update(change)
        assert not v.check_signature(VerifiedOutcome.from_dict(d)), change
    other = MembershipVerifier(Source(_humans(3, clock.now())), clock, str(tmp_path / "other"))
    assert not v.check_signature(other.verify(task))                  # signed with a different key
    # a key file readable by others is refused
    os.chmod(key, 0o644)
    with pytest.raises(PermissionError):
        load_or_create_key(key)
    os.chmod(key, 0o600)
    # every evaluation is in the log, and the chain detects edits, removals and replacement
    for _ in range(3):
        clock.advance(60)
        v.verify(task)
    head = json.load(open(os.path.join(str(tmp_path / "v"), "head.json")))
    ok, n, why = HashChainLog.verify_chain(v.log.path, head)
    assert ok and n == 5 and why == ""                               # first_seen + 4 evaluations
    lines = open(v.log.path).read().splitlines()
    edited = json.loads(lines[2])
    edited["payload"]["outcome"]["value"] = 999
    tampered = str(tmp_path / "tampered.jsonl")
    open(tampered, "w").write("\n".join(lines[:2] + [json.dumps(edited)] + lines[3:]) + "\n")
    assert HashChainLog.verify_chain(tampered)[0] is False
    open(tampered, "w").write("\n".join(lines[:2] + lines[3:]) + "\n")
    assert HashChainLog.verify_chain(tampered)[0] is False
    open(tampered, "w").write("\n".join(lines[:3]) + "\n")           # truncated but internally consistent
    assert HashChainLog.verify_chain(tampered)[0] is True and HashChainLog.verify_chain(tampered, head)[0] is False
    # a verifier refuses to continue on a log that no longer matches its recorded head
    open(v.log.path, "w").write("\n".join(lines[:3]) + "\n")
    with pytest.raises(RuntimeError):
        MembershipVerifier(Source([]), clock, str(tmp_path / "v"))


def test_verifier_service_over_rpc_uses_the_operators_task(tmp_path):
    clock = SimClock()
    run_dir = str(tmp_path / "run")
    members = str(tmp_path / "members.jsonl")
    with open(members, "w") as f:
        for m in _humans(5, clock.now()):
            f.write(json.dumps({"id": m.id, "joined_at": m.joined_at}) + "\n")
        f.write(json.dumps({"id": "b", "is_bot": True}) + "\n")
    task = _task(target=5)
    sock = os.path.join(run_dir, "sock", "verify.sock")
    server = serve(run_dir, sock, task, JsonlMembershipSource(members), clock=clock, allowed_uids={os.getuid()})
    try:
        rv = RemoteVerifier(sock)
        o = rv.verify(task)
        assert isinstance(o, VerifiedOutcome) and o.status == GoalStatus.ACHIEVED and o.value == 5
        assert o.excluded == {"bots": 1} and rv.check_signature(o)
        o.value = 50.0
        assert rv.check_signature(o) is False
        # the caller cannot lower the bar: the service judges the operator's pinned task, not the one it is sent
        easy = _task(target=1)
        easy.success["exclusions"]["exclude_bots"] = False
        assert rv.verify(easy).target == 5 and rv.verify(easy).excluded == {"bots": 1}
        other = _task(target=5)
        other.objective_id = "something-else"
        with pytest.raises(Exception, match="serves objective"):
            rv.verify(other)
        info = rv.info()
        assert info["source_live"] is False and info["objective_id"] == task.objective_id
        with pytest.raises(Exception, match="not exposed"):
            rv._rpc.call("_signature", o)
        assert os.path.exists(os.path.join(run_dir, "verifier.jsonl"))
        assert stat.S_IMODE(os.stat(os.path.join(run_dir, "verifier", "verifier.key")).st_mode) == 0o600
        rv.close()
    finally:
        server.stop()


def test_metric_threshold_verifier_for_the_second_objective(tmp_path):
    clock = SimClock()
    raw = T.load_task_raw(os.path.join(TASKS, "utility_staging.yaml"))
    task = T.parse_task(raw)
    metrics = str(tmp_path / "metrics.json")
    json.dump({"utility.distinct_successful_sessions": 7, "utility.requests_total": 31}, open(metrics, "w"))
    v = build_verifier(task, str(tmp_path / "v"), clock, JsonMetricSource(metrics))
    o = v.verify(task)
    assert o.status == GoalStatus.PENDING and o.value == 7 and o.target == 20 and v.check_signature(o)
    assert {q.name for q in o.secondary} >= {"utility.requests_total", "secondary.time_elapsed_s"}
    json.dump({"utility.distinct_successful_sessions": 22}, open(metrics, "w"))
    assert v.verify(task).status == GoalStatus.ACHIEVED
    json.dump({}, open(metrics, "w"))
    o = v.verify(task)
    assert o.status == GoalStatus.PENDING and "nothing is claimed" in o.notes
    clock.advance(task.limits.duration_s + 1)
    json.dump({"utility.distinct_successful_sessions": 3}, open(metrics, "w"))
    assert v.verify(task).status == GoalStatus.FAILED
