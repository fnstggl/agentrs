"""Runtime wiring of the integrated agent (INT2; change requests M-int-1, -2, -3, -6 and F-11).

* the runtime only *looks* at a grant before ``env.execute``; the computer environment consumes it (M-int-6, also in
  in-flight recovery and the approval path — all three use ``runtime.loop.look_grant``);
* ``runtime.loop_service`` passes the verifier to the agent factory (M-int-1);
* verifier measurements used for supervision are signed by the verifier and checked by the agent; a forged one is
  refused (M-int-3 / F-11 b);
* ``cam-live validate --mode staging`` passes ``--debug/--real-model``, ``--decisions``, ``--out``, ``--seed`` (M-int-2);
* the agent works on the ``learner_view`` the runtime hands it (F-11 a).
"""
from __future__ import annotations

import os
import signal
import sys

import pytest

from cam_live.contracts import (ActionSpec, AuthVerdict, Limits, OutcomeObservation, OutcomeQuery, OutcomeStatus,
                                ResultStatus, SimClock, TaskSpec, Trust, WallClock, new_id)
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig, look_grant
from int_helpers import Recorder, agent_config, lab_config, run_lifetime

HERE = os.path.dirname(os.path.abspath(__file__))


def _env_and_task(tmp_path):
    """The computer environment (internal + http executors, no browser) with the real authorization service, on a
    local staging site."""
    from cam_live.harness.auth import PolicyService
    from cam_live.harness.env import ComputerEnvironment
    from cam_live.harness.factories import all_executors
    from cam_live.harness.staging import StagingSites
    sites = StagingSites().start()
    host = sites.urls["structured"].replace("http://", "")
    task = TaskSpec(objective_id="obj_int2_grant", objective="Read the records application.",
                    success={"verifier": "metric_threshold", "metric": "staging.items.count", "target": 99},
                    resources={"owned": [host, sites.urls["structured"] + "/"], "read": [host]},
                    interfaces=["api", "internal"], mode="staging",
                    limits=Limits(duration_s=3600.0, money=1.0, max_external_actions=50, max_decisions=8),
                    data_permissions={"aggregate_metrics": True})
    clock = WallClock()
    policy = PolicyService(str(tmp_path / "auth"), clock=clock)
    policy.register_task(task, "operator")
    env = ComputerEnvironment(str(tmp_path / "exec"), policy, clock=clock, secrets=policy.secrets_handle(),
                              provenance="staging")
    for ex in all_executors(str(tmp_path / "exec"), clock, env=env, policy=policy, task=task, browser_enabled=False,
                            exclude=("visual", "code", "skill")):
        env.register(ex)
    return sites, env, policy, task, clock


def test_look_grant_leaves_the_grant_to_the_executor(tmp_path):
    sites, env, policy, task, _clock = _env_and_task(tmp_path)
    try:
        tool = next(t for t in env.discover_tools() if t.name == "http.get")
        def action():
            return ActionSpec(action_id=new_id("act"), operation=tool.name, modality=tool.modality,
                              arguments={"url": sites.urls["structured"] + "/"}, auth_scope=tool.auth_scope)
        a = action()
        auth = policy.authorize(a, task, {})
        assert auth.verdict == AuthVerdict.ALLOW and auth.grant
        assert look_grant(policy, a, auth) and look_grant(policy, a, auth)      # looking twice uses nothing up
        res = env.execute(a, auth)
        assert res.status == ResultStatus.OK, res.failure_detail
        assert not policy.check_grant(a, auth, consume=True)                    # the environment consumed it
        # the defect this replaces: a consuming check before execute makes the environment refuse the action
        b = action()
        auth_b = policy.authorize(b, task, {})
        assert policy.check_grant(b, auth_b)                                     # consume=True (the old runtime call)
        res_b = env.execute(b, auth_b)
        assert res_b.status == ResultStatus.DENIED and "grant does not verify" in res_b.failure_detail
    finally:
        env.close()
        sites.stop()


def test_look_grant_with_a_two_argument_policy():
    from cam_live.diagnostics.world import AllowAllPolicy
    p = AllowAllPolicy()
    a = ActionSpec(action_id="act_x", operation="wait", modality="internal", arguments={"seconds": 1.0})
    auth = p.authorize(a, TaskSpec(objective_id="o", objective="x", mode="diagnostic"))
    assert look_grant(p, a, auth)


def test_runtime_executes_in_the_computer_environment_without_grant_refusals(tmp_path):
    """The real runtime + integrated agent + ComputerEnvironment + PolicyService (no LookOnlyGrants wrapper)."""
    from cam_live.agent import CamLiveAgent
    sites, env, policy, task, clock = _env_and_task(tmp_path)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "agent"), clock=clock)
    try:
        rt = LiveRuntime(agent, env, policy, _NoVerifier(), task, str(tmp_path / "rt"), clock,
                         RuntimeConfig(fsync=False, candidates_k=8, keep_checkpoints=1))
        rt.start()
        for _ in range(16):
            if rt.status != "running":
                break
            rt.step()
        rt.close()
        recs = [r for r in agent.store.all() if r.result is not None]
        assert len(recs) >= 5
        details = [r.result.failure_detail for r in recs]
        assert not any("grant does not verify" in d for d in details), details
        ran = [r for r in recs if r.result.status == ResultStatus.OK and r.decision.chosen.modality == "api"]
        assert ran, [(r.decision.chosen.operation, r.result.status, r.result.failure_detail[:80]) for r in recs]
    finally:
        agent.close()
        env.close()
        sites.stop()


class _NoVerifier:
    """A verifier that never confirms anything (this test is about grants, not outcomes)."""

    def verify(self, task):
        raise RuntimeError("not used")

    def check_signature(self, outcome):
        return False

    def measure(self, queries):
        return []


def test_loop_service_passes_the_verifier_to_the_agent(tmp_path):
    import int2_factory
    from cam_live.diagnostics.world import LabVerifier
    from cam_live.runtime.loop_service import run_service
    from cam_live.diagnostics.world import LabWorld, lab_task
    world = LabWorld(lab_config(5), SimClock())
    task = lab_task(world, "obj_int2_service")
    task.success = {**task.success, "verifier": "labworld"}
    task.limits.max_decisions = 3
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    try:
        status = run_service(str(tmp_path / "run"), task, agent_config(), agent_factory="int2_factory:build",
                             inproc=True, max_iterations=6, block=False)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    agent = int2_factory.SEEN["agent"]
    try:
        assert isinstance(int2_factory.SEEN["verifier"], LabVerifier), int2_factory.SEEN["verifier"]
        assert agent.verifier is int2_factory.SEEN["verifier"] and agent.status()["verifier_attached"]
        assert agent.counters["verified_refused"] == 0
        assert any(e.kind == "objective" for e in agent.meta.ledger.entries()), status    # signed outcome admitted
    finally:
        agent.close()


def test_signed_measurements_supervise_and_forged_ones_do_not(tmp_path):
    from cam_live.agent import CamLiveAgent
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "agent"), clock=clock)
    try:
        lt = run_lifetime(Recorder(agent), clock, str(tmp_path), 21, max_steps=24, max_decisions=10)
        recs = [r for r in agent.store.all() if r.result is not None]
        agent.meta.admit_measurements(recs)                # what the meta-learner does when it builds a window
        meas = [e for e in agent.meta.ledger.entries() if e.kind == "measurement"]
        assert meas, "no verifier measurement reached the outcome ledger"
        signed = [o for r in recs for o in r.outcomes if o.verified and o.signature]
        assert signed and all(lt.verifier.check_measurement(o) for o in signed)
        assert all(agent._check_measurement(o) for o in signed)
        trusted = [i for i in agent.store.items([f"{o.query_id}:out" for o in signed]) if i is not None]
        assert trusted and all(i.trust == Trust.VERIFIER for i in trusted)
        # the agent's goal-metric question is asked of the verifier and comes back signed
        goal = lt.task.success["metric"]
        assert any(q.metric == goal and q.goal_relevant for r in recs for q in r.pending_queries), "no goal query"
        # a measurement relabelled as verified without the verifier's signature is not trusted
        r0 = recs[0]
        q = OutcomeQuery(query_id=new_id("q"), metric=goal, kind="count", goal_relevant=True, issued_at=clock.now())
        forged = OutcomeObservation(query_id=q.query_id, status=OutcomeStatus.OBSERVED, value=999.0,
                                    observed_at=clock.now() + 1, verified=True, source="verifier:labworld",
                                    signature="0" * 64)
        assert not agent._check_measurement(forged)
        n_ledger = len(agent.meta.ledger)
        r0.pending_queries.append(q)                     # a query of that decision (adding queries is allowed)
        agent.store.update(r0)
        agent.query_owner[q.query_id] = {"exp_id": r0.exp_id, "action_id": r0.decision.chosen.action_id,
                                         "experiment_id": "", "arm": ""}
        agent.reconcile([forged])
        item = agent.store.items([f"{q.query_id}:out"])[0]
        assert item is not None and item.trust == Trust.UNTRUSTED
        agent.meta.admit_measurements([agent.store.get(r0.exp_id)])
        assert len(agent.meta.ledger) == n_ledger            # never admitted to the meta objective
    finally:
        agent.close()


def test_agent_receives_the_learner_view(tmp_path):
    from cam_live.agent import CamLiveAgent
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "agent"), clock=clock)
    try:
        lt = run_lifetime(Recorder(agent), clock, str(tmp_path), 22, max_steps=8, max_decisions=4)
        t = agent.task
        assert t.objective_id == lt.task.objective_id and t.measurement == {}
        assert set(t.success) <= {"metric", "target"} and t.success["metric"] == lt.task.success["metric"]
        assert lt.task.measurement                       # the full task had a measurement protocol; it was withheld
        assert agent.counters["results"] >= 3
    finally:
        agent.close()


def test_cli_validate_passes_its_options(tmp_path, monkeypatch):
    from cam_live.runtime import cli
    seen = {}

    def fake(cfg=None, *, out_dir=None, real_model=True, n_decisions=40, seed=0, evidence_dir=None):
        seen.update(out_dir=out_dir, real_model=real_model, n_decisions=n_decisions, seed=seed,
                    evidence_dir=evidence_dir)
        return {"ok": True, "passed": True}
    monkeypatch.setattr(cli, "load_hook", lambda spec, what: fake)
    out = str(tmp_path / "v")
    assert cli.main(["validate", "--mode", "staging", "--debug", "--decisions", "6", "--out", out, "--seed", "3"]) == 0
    assert seen == {"out_dir": os.path.abspath(out), "real_model": False, "n_decisions": 6, "seed": 3,
                    "evidence_dir": None}
    assert os.path.exists(os.path.join(out, "staging_report.json"))
    seen.clear()
    assert cli.main(["validate", "--mode", "staging", "--real-model", "--out", out]) == 0
    assert seen["real_model"] is True and seen["n_decisions"] == 40 and seen["seed"] == 0
    assert cli.main(["validate", "--mode", "live"]) == 2
