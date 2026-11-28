"""INT3B round 2: defects found by the independent verifier G2-runtime (docs/verification/G2_runtime.md) at a5ab772.
Each test fails on a5ab772 and passes with the fix; the verifier's probes in repro/g2runtime/ report HOLDS."""
from __future__ import annotations

import copy
import os

import pytest
import yaml

from cam_live.runtime import tasks as T

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISCORD = os.path.join(PROJECT, "tasks", "discord_100.yaml")
PLANNING = os.path.join(PROJECT, "tasks", "discord_100_planning.yaml")
ZW = "​"


# ---------------------------------------------------------------------------------------------------------- G2R-1
def _filled():
    raw = T.load_task_raw(DISCORD)
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"].update(discord_guild_id="123456789012345678", discord_bot_credential_name="bot",
                            owned_site="https://example.invalid", operator_authorization_record="rec-1",
                            authorized_tools=["discord.*", "http.*"], authorized_destinations=["example.invalid"],
                            currency="USD")
    raw["resources"]["publishing"].update(authorized=True, destinations=["example.invalid"])
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    raw["data_permissions"]["collect_feedback_with_consent"] = False
    return raw


def _refused(raw, tmp_path) -> bool:
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        T.load_task(str(p), for_run=True)
    except T.TaskValidationError:
        return True
    return False


# neutral placeholders for content words (the test file itself carries no community strategy): the rules look for
# kinds of words (platforms, cadence, strategy vocabulary, ordered / imperative steps), not for these examples
DISGUISES = {
    "underscore-joined metric": lambda r: r["secondary_metrics"].append("secondary.reddit_widget_strategy_with_daily_thing"),
    "dash-joined family": lambda r: r.update(family="community/widget-thing-then-invite-friends"),
    "tool pattern": lambda r: r["resources"].update(authorized_tools=["discord.*", "step_one_make_x.step_two_invite_y"]),
    "interfaces item": lambda r: r["interfaces"].append("widget_on_reddit.then_invite_people.*"),
    "destination query": lambda r: r["resources"].update(authorized_destinations=[
        "example.invalid", "https://example.invalid/?q=widget_then_share_daily"]),
    "channel labels": lambda r: r["resources"].update(discord={"guild_id": "123456789012345678", "channels": {
        "invite-some-friends": "223456789012345678", "start-a-contest": "323456789012345678"}}),
    "objective_id": lambda r: r.update(objective_id="discord_100_via_reddit_marketing"),
    "camelCase key": lambda r: r["resources"].update(discord={"guild_id": "123456789012345678",
                                                              "channels": {"growthPlan": "223456789012345678"}}),
    "full-width platform word": lambda r: r["secondary_metrics"].append("secondary.ｒｅｄｄｉｔ_x"),
    "zero-width objective": lambda r: r.update(objective="Create and operate a genuinely useful Discord community that "
                                                         "attracts 100 members by posting on r" + ZW + "eddit e" + ZW +
                                                         "very day, within the authorized experimental constraints"),
    "zero-width in an identifier": lambda r: r["resources"].update(authorized_tools=["disc" + ZW + "ord.*"]),
}


@pytest.mark.parametrize("name", sorted(DISGUISES))
def test_disguised_strategy_content_is_refused(tmp_path, name):
    base = _filled()
    assert not _refused(copy.deepcopy(base), tmp_path)             # the filled task itself loads
    raw = copy.deepcopy(base)
    DISGUISES[name](raw)
    assert _refused(raw, tmp_path), name


def test_normalisation_reads_like_a_reader_and_shipped_tasks_stay_clean():
    assert T.normalise_for_scan("secondary.a_b-c:d/e") == "secondary a b c d e"
    assert T.normalise_for_scan("growthPlan") == "growth Plan"
    assert T.normalise_for_scan("ｒｅｄ" + ZW + "dit") == "reddit"
    assert T.invisible_characters("a‮b" + ZW) == ["U+200B", "U+202E"]
    for f in ("discord_100.yaml", "discord_100_planning.yaml", "lab_staging.yaml", "utility_staging.yaml",
              os.path.join("staging", "discord_100_planning_staging.yaml")):
        raw = T.load_task_raw(os.path.join(PROJECT, "tasks", f))
        assert T.strategy_findings(raw) == [] and T.structural_problems(raw) == [], f
    # the operator's confirmed objective still loads verbatim
    t = T.load_task(PLANNING, for_run=False)
    assert T.learner_view(t).objective == T.load_task_raw(PLANNING)["objective"]


# ---------------------------------------------------------------------------------------------------------- G2R-2 / 3
import datetime as _dt
import json

from cam_live.contracts import GoalStatus, Limits, Quantity, RunFiles, SimClock, TaskSpec
from cam_live.harness.discord_membership import DiscordMembershipSource
from cam_live.runtime.verifier import build_verifier

DAY = 86400.0
T0 = 1_800_000_000.0
GUILD, OWNER = "900000000000000001", "900000000000000002"


class FakeDiscord:
    """Only CURRENT members are listed (like the REST API); channels / messages only when ``channels`` is set."""

    def __init__(self, channels=None):
        self.members, self.channels, self.requests_sent = {}, channels, 0

    def _request(self, method, path, *, route, params=None, **_kw):
        self.requests_sent += 1
        if path == f"/guilds/{GUILD}":
            return {"id": GUILD, "owner_id": OWNER}
        if path == f"/guilds/{GUILD}/members":
            after = int((params or {}).get("after", "0"))
            return [{"user": {"id": u, "bot": False}, "pending": False, "flags": 0,
                     "joined_at": _dt.datetime.fromtimestamp(j, _dt.timezone.utc).isoformat()}
                    for u, j in sorted(self.members.items(), key=lambda kv: int(kv[0])) if int(u) > after]
        if self.channels is not None and path == f"/guilds/{GUILD}/channels":
            return [{"id": c, "type": 0} for c in self.channels]
        if self.channels is not None and path.startswith("/channels/"):
            cid = path.split("/")[2]
            return [{"author": {"id": a, "bot": False}, "content": "x"} for a in self.channels[cid]]
        raise AssertionError(path)


def _uid(i):
    return str(1_100_000_000_000_000_000 + i)


def _task(**measurement):
    return TaskSpec(objective_id="o_members", objective="members", family="community/membership", mode="staging",
                    success={"verifier": "membership", "metric": "membership.confirmed", "target": 100,
                             "confidence": 0.95, "exclusions": {}},
                    limits=Limits(duration_s=60 * DAY, max_decisions=10),
                    measurement={"window_start_ts": T0, **measurement}, resources={"currency": "USD"})


def _verifier(tmp_path, fake, clock, name="v", **kw):
    src = DiscordMembershipSource(GUILD, lambda: "unused", client=fake)
    t = _task()
    return t, build_verifier(t, str(tmp_path / name / "verifier"), clock, src,
                             log_path=str(tmp_path / name / "v.jsonl"), **kw)


def test_members_who_joined_before_the_experiment_are_not_counted(tmp_path):
    fake = FakeDiscord()
    for i in range(100):
        fake.members[_uid(i)] = T0 - 30 * DAY
    for i in range(100, 103):
        fake.members[_uid(i)] = T0 + 3600.0
    t, v = _verifier(tmp_path, fake, SimClock(T0 + DAY))
    o = v.verify(t)
    assert o.value == 3 and o.status == GoalStatus.PENDING                 # a5ab772: 103, achieved
    assert o.excluded.get("joined_before_start") == 100 and "joined before the experiment" in o.notes
    log = [json.loads(x) for x in open(tmp_path / "v" / "v.jsonl")]
    base = [e for e in log if e["kind"] == "baseline"]
    assert len(base) == 1 and base[0]["payload"]["members_before_start"] == 100
    assert not any(_uid(0) in json.dumps(e) for e in log)                 # baseline by count only


def test_retention_counts_members_who_left_with_a_current_members_only_source(tmp_path):
    fake = FakeDiscord()
    for i in range(10):
        fake.members[_uid(i)] = T0 + 0.5 * DAY
    clock = SimClock(T0 + 2 * DAY)
    t, v = _verifier(tmp_path, fake, clock)
    v.verify(t)                                                            # the verifier saw all ten
    for i in range(5):
        del fake.members[_uid(i)]
    clock.advance(8 * DAY)
    sec = {q.name: q.value for q in v.verify(t).secondary}
    assert sec["secondary.retention_7d"] == pytest.approx(0.5)            # a5ab772: 1.0


def test_participation_from_aggregate_signals_cost_and_resource_use_are_cams(tmp_path):
    from cam_live.runtime.verifier import authorization_cost_source
    fake = FakeDiscord(channels={"300000000000000001": [_uid(0), _uid(0), _uid(1), "999"]})
    for i in range(3):
        fake.members[_uid(i)] = T0 + 3600.0

    class Ledger:
        def limits_state(self):
            return {"tasks": {"o_members": {"used": {"money": 2.5, "external_actions": 7.0, "decisions": 9.0,
                                                     "compute_s": 30.0}}}}
    t, v = _verifier(tmp_path, fake, SimClock(T0 + DAY),
                     cost_source=authorization_cost_source(Ledger(), lambda _o: "USD"))
    o = v.verify(t)
    sec = {q.name: q for q in o.secondary}
    assert sec["secondary.participation.members"].value == 2 and sec["secondary.participation.interactions"].value == 3
    assert sec["secondary.operating_cost"].value == 2.5 and sec["secondary.operating_cost"].unit == "USD"
    assert sec["secondary.resource_use.external_actions"].value == 7.0
    assert "secondary.resource_use.discord.verifier.requests" not in sec   # a5ab772: the verifier's own requests
    # nothing measurable: said so, no placeholder value
    t2, v2 = _verifier(tmp_path, FakeDiscord(), SimClock(T0 + DAY), name="w")
    o2 = v2.verify(t2)
    names = {q.name for q in o2.secondary}
    assert not any(n.startswith(("secondary.participation", "secondary.feedback", "secondary.operating_cost"))
                   for n in names)
    assert "Not measured:" in o2.notes and "participation" in o2.notes and "feedback" in o2.notes \
        and "operating_cost" in o2.notes


def test_the_run_services_connect_the_cost_accounting(tmp_path):
    import inspect
    from cam_live.runtime import loop_service, verifier_service
    assert "authorization_cost_source(policy" in inspect.getsource(loop_service.build_inproc)
    assert "authorization_cost_source(" in inspect.getsource(verifier_service.serve)


# ---------------------------------------------------------------------------------------------------------- G2R-4
def test_an_operator_assigned_objective_is_verified_and_a_forged_one_is_refused(tmp_path):
    from cam_live.diagnostics.world import LabConfig
    from cam_live.runtime.control import ControlChannel, OperatorControl, OperatorKey
    from cam_live.runtime.logs import RunReader
    from rt_helpers import make_lab

    class WorldMetric:
        name, live = "lab-world-metric", False

        def __init__(self):
            self.world = None

        def read(self):
            return [Quantity("lab.adopters.total", float(self.world.total()))]

    def edit(task):
        task.success["verifier"] = "metric_threshold"
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=3, budget=1000.0), task_edit=edit)
    OperatorKey(lab.run_dir).ensure()
    src = WorldMetric()
    src.world = lab.world
    v = build_verifier(lab.task, os.path.join(lab.run_dir, "verifier"), lab.clock, src,
                       log_path=os.path.join(lab.run_dir, "verifier.jsonl"), pin=True)   # operator_dir = run dir
    lab.rt.verifier = v
    assert lab.rt.run(max_iterations=2000) == "completed"
    raw = yaml.safe_load(open(os.path.join(PROJECT, "tasks", "lab_staging.yaml")))
    target_b = int(lab.world.total()) + 2
    raw.update(objective_id="lab-second", objective=f"Reach {target_b} adopters in the lab world before time runs out.")
    raw["success"].update(verifier="metric_threshold", metric="lab.adopters.total", target=target_b)
    path_b = str(tmp_path / "b.yaml")
    yaml.safe_dump(raw, open(path_b, "w"))
    # a forged command (not signed by the operator) is refused by the verifier itself
    forged = {"id": "cmd_x", "cmd": "assign_objective", "by": "operator", "issued_at": 0.0,
              "args": {"path": path_b, "sha256": ""}, "sig": "AAAA"}
    with pytest.raises(PermissionError):
        v.assign_objective(forged)
    cid = OperatorControl(lab.run_dir).assign_objective(path_b)
    for _ in range(400):
        lab.rt.step()
        if lab.rt.status != "running":
            break
    st = RunReader(lab.run_dir).state()
    assert st["control_results"][cid]["ok"] is True
    ev = RunReader(lab.run_dir).records(RunFiles.EVENTS)
    assert [e for e in ev if e["type"] == "verified_outcome" and e["data"]["outcome"]["objective_id"] == "lab-second"]
    assert not [e for e in ev if e["type"] == "error" and e["data"].get("what") == "verify"]   # a5ab772: 5, paused
    _ = ControlChannel
    lab.rt.close()


# ---------------------------------------------------------------------------------------------------------- G2R-5
def test_program_steps_count_toward_the_external_action_limit(tmp_path):
    from cam_live.contracts import ActionResult, ActionSpec, Cost, Level, Modality, ResultStatus, ToolDescriptor, new_id
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    from rt_lab_agent import LabAgent
    clock = SimClock()
    world = LabWorld(LabConfig(seed=3, target=100000, budget=1000.0), clock)

    class ProgramEnv:
        def __getattr__(self, name):
            return getattr(world, name)

        def discover_tools(self):
            return list(world.discover_tools()) + [ToolDescriptor(name="program", modality=Modality.INTERNAL,
                                                                  description="run steps in order",
                                                                  level=Level.PROCEDURE, side_effects="none")]

        def validate_action(self, action):
            return [e for s in action.steps for e in world.validate_action(s)] if action.steps else \
                world.validate_action(action)

        def execute(self, action, auth):
            if action.operation != "program":
                return world.execute(action, auth)
            for s in action.steps:
                world.execute(s, auth)
            return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=clock.now(),
                                ended=clock.now(), cost=Cost())

    class ProgramAgent(LabAgent):
        def propose(self, tools, k):
            act = {t.name: t for t in tools}["lab.act"]
            steps = [ActionSpec(action_id=new_id("act"), operation="lab.act", modality=act.modality, level=act.level,
                                arguments={"channel": i % 3, "effort": 1}) for i in range(3)]
            return [ActionSpec(action_id=new_id("act"), operation="program", modality=Modality.INTERNAL,
                               level=Level.PROCEDURE, steps=steps)]
    task = lab_task(world, "obj_prog")
    task.limits.max_external_actions = 4
    task.limits.max_decisions = 6
    rt = LiveRuntime(ProgramAgent(clock), ProgramEnv(), AllowAllPolicy(), LabVerifier(world), task,
                     str(tmp_path / "run"), clock, RuntimeConfig(fsync=False, settle_after_finish_s=0.0))
    rt.run(max_iterations=200)
    assert rt.counters["external_actions"] == world._n_exec > 0            # a5ab772: counter 0
    assert world._n_exec <= 4 + 3 and "max_external_actions" in rt.status_reason
    rt.close()


# ---------------------------------------------------------------------------------------------------------- G2R-8
def test_finishing_an_objective_again_after_a_crash_records_it_once(tmp_path):
    from cam_live.diagnostics.world import LabConfig
    from cam_live.runtime.logs import RunReader
    from rt_helpers import make_lab
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=3))
    assert lab.rt.run(max_iterations=2000) == "completed"
    lab.rt._finish("completed", "again, as a runtime restarted from the commit before the record would")
    fin = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS) if e["type"] == "objective_finished"]
    assert len(fin) == 1                                                   # a5ab772: 2
    lab.rt.close()


# ---------------------------------------------------------------------------------------------------------- G2I-14 / 15 / 8 / 9
def test_a_limit_refusal_is_a_resource_limit_not_a_missing_permission(tmp_path):
    from cam_live.contracts import AuthorizationDecision, AuthVerdict, FailureClass
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig
    from cam_live.runtime.logs import RunReader
    from cam_live.runtime.loop import refusal_failure_class
    from rt_helpers import make_lab
    limit = AuthorizationDecision(AuthVerdict.DENY, "h", ["resource: max_external_actions exhausted (used 2, limit 2)"],
                                  policy_ids=["L-EXTERNAL-ACTIONS"])
    assert refusal_failure_class(limit) == FailureClass.RESOURCE_LIMIT
    rule = AuthorizationDecision(AuthVerdict.DENY, "h", ["not an allowed destination"], policy_ids=["T-DEST"])
    assert refusal_failure_class(rule) == FailureClass.MISSING_PERMISSION

    class LimitPolicy(AllowAllPolicy):
        def authorize(self, action, task, context=None):
            if action.operation == "lab.act":
                return AuthorizationDecision(AuthVerdict.DENY, action.content_hash(), list(limit.reasons),
                                             policy_ids=list(limit.policy_ids))
            return super().authorize(action, task, context)
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000), policy=LimitPolicy(),
                   agent_kw={"only_ops": ["lab.act"]}, task_edit=lambda t: setattr(t.limits, "max_decisions", 2),
                   rt_cfg=__import__("cam_live.runtime.loop", fromlist=["RuntimeConfig"]).RuntimeConfig(
                       fsync=False, settle_after_finish_s=0.0))
    lab.rt.run(max_iterations=50)
    r = RunReader(lab.run_dir)
    res = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"]) if e["data"]["kind"] == "denied"]
    assert res and all(d["failure_class"] == FailureClass.RESOURCE_LIMIT for d in res)    # a5ab772: missing_permission
    obs = [o["data"]["obstacle"] for o in r.obstacles()]
    assert obs and obs[0]["category"] == "resource"
    lab.rt.close()


def test_an_approved_action_refused_at_re_authorization_is_not_reported_as_rejected(tmp_path):
    from cam_live.contracts import AuthorizationDecision, AuthVerdict, FailureClass
    from cam_live.diagnostics.world import LabConfig
    from cam_live.runtime.logs import RunReader
    from cam_live.runtime.loop import RuntimeConfig
    from rt_helpers import ApprovalPolicy, make_lab

    class ReservingPolicy(ApprovalPolicy):
        """Approved at the service by the operator's own tool; the identical re-authorization is then refused by a
        limit (as when the approval's own grant reserved the last unit of the budget)."""
        def authorize(self, action, task, context=None):
            for e in self.queue.values():
                if e["action_hash"] == action.content_hash() and e["status"] == "approved":
                    return AuthorizationDecision(AuthVerdict.DENY, e["action_hash"],
                                                 ["resource: max_external_actions exhausted (used 2 incl. reserved, "
                                                  "requested 1, limit 2)"], policy_ids=["L-EXTERNAL-ACTIONS"])
            return super().authorize(action, task, context)
    policy = ReservingPolicy(("lab.unlock",))
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000), policy=policy,
                   agent_kw={"only_ops": ["lab.unlock", "lab.probe"]}, rt_cfg=RuntimeConfig(fsync=False))
    for _ in range(40):
        lab.rt.step()
        if policy.queue:
            break
    (aid,) = list(policy.queue)
    lab.agent.only_ops = {"lab.probe"}
    policy.queue[aid]["status"] = "approved"                  # the operator approved it at the service
    for _ in range(10):
        lab.rt.step()
    r = RunReader(lab.run_dir)
    res = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"]) if e["data"].get("approval_id") == aid
           and e["data"]["kind"] != "queued"]
    assert res, "the approval was reported"
    d = res[-1]
    assert "rejected" not in d["failure_detail"] and "approved by the operator" in d["failure_detail"]   # a5ab772
    assert "L-EXTERNAL-ACTIONS" in d["failure_detail"] and d["failure_class"] == FailureClass.RESOURCE_LIMIT
    lab.rt.close()


def test_obstacles_unknown_class_is_informational_and_learning_links_need_use():
    from cam_live.contracts import ActionResult, ActionSpec, FailureClass, LearningUpdateRecord, ResultStatus, SimClock
    from cam_live.runtime.logs import RunLog
    from cam_live.runtime.obstacles import CATEGORY_BY_FAILURE, ObstacleTracker
    import tempfile
    assert CATEGORY_BY_FAILURE[FailureClass.UNKNOWN] == "informational"                     # a5ab772: ""
    log = RunLog(tempfile.mkdtemp(), SimClock(), durable=False)
    tr = ObstacleTracker(log, SimClock())
    a = ActionSpec(action_id="act_1", operation="x.y")
    rec = tr.open(a, ActionResult(action_id="act_1", status=ResultStatus.FAILED), decision_id="dec_000001")
    other = LearningUpdateRecord(update_id="u_other", ts=0.0, trigger="t", timescale="frequent", exp_ids=["exp_999"])
    used = LearningUpdateRecord(update_id="u_used", ts=0.0, trigger="t", timescale="frequent", exp_ids=["exp_000001"])
    tr.on_learning([other, used])
    assert rec.learning_update_ids == ["u_used"]                                            # a5ab772: both
    log.close()
