"""The 17-step loop on LabWorld: success, failure, ordering of predictions, waiting, delayed outcomes."""
import os

from cam_live.contracts import GoalStatus, RunFiles
from cam_live.diagnostics.world import DAY, LabConfig
from cam_live.runtime.logs import RunReader, read_json
from cam_live.runtime.loop import COMPLETED, EXHAUSTED, FAILED, RuntimeConfig
from rt_helpers import make_lab


def test_full_loop_reaches_verified_success(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=5))
    status = lab.rt.run(max_iterations=2000)
    assert status == COMPLETED
    r = RunReader(lab.run_dir)
    st = r.state()
    vo = st["last_verified"]["outcome"]
    assert vo["status"] == GoalStatus.ACHIEVED and vo["value"] >= 5 and lab.verifier.check_signature(
        type(lab.verifier.verify(lab.task)).from_dict(vo))
    assert lab.world.total() >= 5
    assert st["status"] == COMPLETED and st["awaiting_operator"] is True
    # the complete learner state was checkpointed at the end and can be loaded
    ckpt = st["committed"]["ckpt"]
    assert os.path.exists(os.path.join(ckpt["path"], "agent.json"))
    # every step of the loop was exercised on the learner
    for name in ("set_objective", "ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select",
                 "record_result", "internal_rewards", "learn", "meta_learn", "save"):
        assert lab.agent.calls.get(name, 0) >= 1, name
    # the verified outcome reached the learner, and only through reconcile
    assert (GoalStatus.ACHIEVED, vo["value"]) in lab.agent.verified_seen
    # every family of record exists
    for name in (RunFiles.EVENTS, RunFiles.DECISIONS, RunFiles.PREDICTIONS, RunFiles.EXPERIMENTS, RunFiles.LEARNING,
                 RunFiles.META):
        assert r.records(name), name
    assert os.path.exists(os.path.join(lab.run_dir, RunFiles.JOURNAL))
    # decisions on disk == decisions the agent made == decisions counted
    n_dec = len(r.records(RunFiles.DECISIONS))
    assert n_dec == lab.agent.calls["select"] == st["counters"]["decisions"]
    # no reward ever came from the environment or the runtime
    assert not any("reward" in k for rec in r.records(RunFiles.EVENTS, types=["action_result"]) for k in rec["data"])


def test_full_loop_reaches_verified_failure(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=4, target=100000, horizon_s=3 * DAY))
    status = lab.rt.run(max_iterations=5000)
    assert status == FAILED
    st = RunReader(lab.run_dir).state()
    assert st["last_verified"]["outcome"]["status"] == GoalStatus.FAILED
    assert (GoalStatus.FAILED, st["last_verified"]["outcome"]["value"]) in lab.agent.verified_seen
    assert os.path.exists(os.path.join(st["committed"]["ckpt"]["path"], "agent.json"))
    # nothing was executed after the verified failure
    events = RunReader(lab.run_dir).records(RunFiles.EVENTS)
    fin = max(e["seq"] for e in events if e["type"] == "objective_finished")
    assert not [e for e in events if e["type"] == "action_start" and e["seq"] > fin]


def test_predictions_are_on_disk_before_execution(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000, horizon_s=5 * DAY),
                   agent_kw={"only_ops": ["lab.act", "wait"], "wait_seconds": DAY})
    lab.rt.run(max_iterations=400)
    r = RunReader(lab.run_dir)
    preds = [p for p in r.records(RunFiles.PREDICTIONS, types=["prediction"])]
    starts = {e["data"]["action_id"]: e for e in r.records(RunFiles.EVENTS, types=["action_start"])}
    results = {e["data"]["action_id"]: e for e in r.records(RunFiles.EVENTS, types=["action_result"])}
    assert starts and preds
    checked = 0
    for p in preds:
        aid = p["data"]["prediction"]["action_id"]
        if p["data"]["for_chosen_action"]:
            # registered (global sequence number and clock) strictly before the executor was called
            assert p["seq"] < starts[aid]["seq"] < results[aid]["seq"]
            assert p["ts"] <= starts[aid]["ts"]
            assert p["data"]["prediction"]["resolved_status"] == "pending"
            checked += 1
    assert checked == len(starts)
    # predictions for candidates that were not chosen are registered too, and flagged as such
    assert any(not p["data"]["for_chosen_action"] for p in preds)
    # resolutions are appended later and never rewrite the registered line
    res = r.records(RunFiles.PREDICTIONS, types=["resolution"])
    by_id = {p["data"]["prediction"]["prediction_id"]: p for p in preds}
    assert res
    for x in res:
        reg = by_id[x["data"]["prediction_id"]]
        assert reg["seq"] < x["seq"] and x["data"]["registered_rid"] == reg["rid"]
        assert x["data"]["outcome"]["value"] is not None


def test_waiting_consumes_no_decisions_and_no_environment_actions(tmp_path):
    # one intervention, then the agent only ever chooses to wait three days; the outcome matures after two
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000),
                   agent_kw={"only_ops": ["lab.act"], "wait_seconds": 3 * DAY})
    rt, agent, world = lab.rt, lab.agent, lab.world
    while world._n_exec == 0:
        rt.step()
    agent.only_ops = {"wait"}
    executed = world._n_exec
    spent = world._spent
    t0 = lab.clock.now()
    rep = rt.step()                         # the wait decision; the runtime sleeps instead of acting
    assert rep["slept"] == "time"
    selects, proposes = agent.calls["select"], agent.calls["propose"]
    slept_until = lab.clock.now()
    # it slept until the pending outcome matured (two days), not for the three days the wait asked for,
    # and not in small polling increments
    assert DAY < slept_until - t0 <= 2 * DAY + 3600 + 1
    sleeps = RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["sleep"])
    assert sleeps[-1]["data"]["reason"] == "outcome" and len(sleeps) == 1
    # while waiting: no decision, no proposal, no environment action, no spending
    assert (agent.calls["select"], agent.calls["propose"]) == (selects, proposes)
    assert world._n_exec == executed and world._spent == spent
    st = read_json(os.path.join(lab.run_dir, RunFiles.STATE))
    assert st["counters"]["waits"] == 1 and st["counters"]["external_actions"] == executed
    # the wait itself never reached the executor
    waits = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["action_start"])
             if e["data"]["operation"] == "wait"]
    assert waits == []


def test_evidence_polling_wakes_without_decisions(tmp_path):
    # with a poll interval configured, poll wakes that find nothing run no decision cycle at all
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=11, target=10000, mean_delay_s=(40 * DAY, 41 * DAY)),
                   agent_kw={"only_ops": ["lab.act"], "wait_seconds": 3 * DAY},
                   rt_cfg=RuntimeConfig(evidence_poll_s=3600.0, fsync=False))
    rt, agent = lab.rt, lab.agent
    while lab.world._n_exec == 0:
        rt.step()
    agent.only_ops = {"wait"}
    rt.step()
    base = dict(agent.calls)
    cycles = rt.cycles
    for _ in range(20):                      # twenty hourly poll wakes, no arrivals (delays are ~40 days)
        rep = rt.step()
        assert rep["cycle"] is False
    assert rt.cycles == cycles
    for k in ("propose", "evaluate", "select", "learn", "ingest"):
        assert agent.calls.get(k, 0) == base.get(k, 0), k
    assert lab.world._n_exec == 1


def test_delayed_outcomes_reconcile_exactly_once(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=5, target=10000, horizon_s=6 * DAY),
                   agent_kw={"only_ops": ["lab.act", "wait"], "wait_seconds": DAY})
    lab.rt.run(max_iterations=400)
    r = RunReader(lab.run_dir)
    outcomes = r.records(RunFiles.EVENTS, types=["outcome"])
    qids = [o["data"]["query"]["query_id"] for o in outcomes]
    assert qids and len(qids) == len(set(qids))
    # the learner saw each matured outcome once
    assert sorted(lab.agent.reconciled) == sorted(qids)
    # each outcome was delayed: observed at or after its due time, well after the decision that caused it
    acts = {e["data"]["action_id"]: e for e in r.records(RunFiles.EVENTS, types=["action_start"])}
    for o in outcomes:
        q = o["data"]["query"]
        assert o["ts"] >= q["issued_at"] + q["horizon_s"]
        assert o["ts"] - acts[o["data"]["action_id"]]["ts"] >= q["horizon_s"]
        assert o["data"]["observation"]["verified"] is True          # goal-relevant: measured by the verifier
    # one resolution per registered prediction of an executed intervention whose outcome matured
    res = r.records(RunFiles.PREDICTIONS, types=["resolution"])
    assert len(res) == len(qids)
    # the values are the world's ground truth for that action
    for o in outcomes:
        aid = o["data"]["action_id"]
        truth = sum(1 for t, a, _ in lab.world._arrivals if a == aid and t <= o["data"]["query"]["issued_at"]
                    + o["data"]["query"]["horizon_s"])
        assert o["data"]["observation"]["value"] == truth
    # experiments: registered before the outcome, completed afterwards, history preserved
    exps = r.records(RunFiles.EXPERIMENTS, types=["experiment"])
    first = {}
    for e in exps:
        first.setdefault(e["data"]["experiment_id"], e)
    matured = [e for e in exps if e["data"]["status"] == "matured"]
    assert matured
    for e in matured:
        reg = first[e["data"]["experiment_id"]]
        assert reg["data"]["status"] == "registered" and reg["seq"] < e["seq"] and reg["data"]["actual"] == []
        assert e["data"]["actual"] and e["data"]["supported"] in (True, False)
    assert r.records(RunFiles.EXPERIMENTS, types=["experiment_outcome"])


def test_limits_exhaustion_checkpoints_and_pauses(tmp_path):
    def edit(task):
        task.limits.max_decisions = 7
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000), task_edit=edit)
    status = lab.rt.run(max_iterations=500)
    assert status == EXHAUSTED
    st = RunReader(lab.run_dir).state()
    assert st["counters"]["decisions"] == 7 == lab.agent.calls["select"]
    assert st["status_reason"] == "limit max_decisions exhausted" and st["awaiting_operator"]
    assert st["limits"]["max_decisions"] == {"used": 7.0, "limit": 7.0}
    ckpt = st["committed"]["ckpt"]
    assert ckpt["id"] == lab.agent.checkpoint_id and os.path.exists(os.path.join(ckpt["path"], "agent.json"))
    # a final verification was taken at exhaustion and handed to the learner before the checkpoint
    assert lab.agent.verified_seen and lab.agent.verified_seen[-1][0] == GoalStatus.PENDING
    # the loop does nothing further once exhausted
    n = lab.agent.calls["select"]
    lab.rt.run(max_iterations=5)
    assert lab.agent.calls["select"] == n


def test_money_limit_is_enforced_before_execution(tmp_path):
    def edit(task):
        task.limits.money = 9.0
        task.limits.max_decisions = 40
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000),
                   agent_kw={"only_ops": ["lab.act"]}, task_edit=edit)
    status = lab.rt.run(max_iterations=500)
    st = RunReader(lab.run_dir).state()
    assert lab.world._spent <= 9.0 + 1e-9 and st["counters"]["money"] == lab.world._spent
    refused = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["action_result"])
               if e["data"]["kind"] == "over_limit"]
    # refused actions flowed back to the learner as failed results and were written as obstacles
    assert refused and all(e["data"]["failure_class"] == "resource_limit" for e in refused)
    assert any(s == "failed" for _, s in lab.agent.results)
    obs = RunReader(lab.run_dir).obstacles()
    assert obs and all(o["data"]["obstacle"]["category"] == "resource" for o in obs)
    assert status == EXHAUSTED and st["status_reason"] == "limit max_decisions exhausted"
