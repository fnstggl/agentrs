"""The integrated agent driven by the real runtime: LiveRuntime + CamLiveAgent + LabWorld + PolicyService +
the signing LabVerifier, over several verified lifetimes (CPU, tiny backbone, debug core).

What is checked (requirement ids of the completion manifest in brackets):
* every step of the 17-step loop calls the agent (IN-01, LX-15);
* predictions are registered before outcomes exist, and resolved only afterwards (PI-02);
* delayed outcomes are linked to the decisions that produced them (RC-01);
* every executed decision carries an internal reward (MR-01);
* learning updates change the experiential and the adapter weights (non-zero delta norms) (CL-02, CL-03, NA-08/09);
* a meta-update runs with a non-zero meta-gradient on the reward parameters (MR-03, MR-05, MR-06);
* behaviour changes: the policy distribution on a held-out context differs after learning, restoring the old
  trainable weights restores it exactly, and the experiential weights alone carry a measured part (NA-06, CU-22);
* a new objective keeps weights, reward and priors and resets fast beliefs and the meta lifetime (CL-09).
LabWorld results are diagnostic evidence about plumbing and learning mechanics, not about any real task.
"""
from __future__ import annotations

import collections
import contextlib
import json
import math
import os

import pytest
import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import (GoalStatus, OutcomeStatus, RunFiles, SimClock, VerifiedOutcome)
from cam_live.runtime.logs import iter_jsonl
from cam_live.validation import HeldOutProbe, weight_swap
from int_helpers import AGENT_METHODS, Recorder, agent_config, count, run_lifetime

# five independent lifetimes: the meta-update needs >= 3 independent verified outcome groups outside the promotion
# hold-out, and every decision of a lifetime with a signed verdict belongs to that lifetime's group (seed 15's
# objective group would be held out for promotion evidence)
SEEDS = (11, 12, 13, 14, 16)
# the module's shared session (five lifetimes) runs inside its first test: measured 200 s on a quiet machine and
# 450 s at load average ~40; the suite's 900 s per-test guard was exceeded at load ~55 (whole-suite run of
# 2026-11-27), so this module's guard is 5x the slowest measurement
pytestmark = pytest.mark.timeout(2400)
TRAINABLE = ("experiential", "semantic_adapters", "heads", "reward")


def _tv(p, q):
    return float(0.5 * (p - q).abs().sum())


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("integrated"))
    clock = SimClock()
    # policy updates every 6 rewarded decisions: each update's query window then holds ~6 decisions, enough for the
    # importance-weighted meta estimator's effective-sample-size floor per lifetime (with 3 it was below the floor
    # in every attempt that had enough independent groups: measured, docs/INTEGRATION.md §9)
    agent = CamLiveAgent(agent_config(policy_update_every=6), os.path.join(root, "agent"), clock=clock)
    rec = Recorder(agent)
    probe = HeldOutProbe(agent)
    p0, s0 = probe.distribution(), agent.model.trainable_state()
    # attribution windows of 30 min (LabWorld default in this family: 6 h): the interventions' outcomes mature
    # inside a test-sized lifetime whatever the agent chooses, instead of only when it happens to wait
    lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)}
    lifetimes = [run_lifetime(rec, clock, root, seed, max_steps=40, max_decisions=22, lab_kw=lab) for seed in SEEDS]
    status = lifetimes[-1].report.get("agent", {})
    yield {"root": root, "agent": agent, "rec": rec, "probe": probe, "p0": p0, "s0": s0, "lifetimes": lifetimes,
           "status": status, "clock": clock}
    agent.close()


def _logs(session, name):
    return [e for lt in session["lifetimes"] for e in iter_jsonl(os.path.join(lt.run_dir, name))]


def test_every_runtime_step_calls_the_agent(session):
    calls = count(session["rec"].calls)
    for m in ("set_objective", "ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select",
              "record_result", "internal_rewards", "learn", "meta_learn", "save", "status", "drain_experiments"):
        assert calls.get(m, 0) > 0, (m, calls)
    assert calls["set_objective"] == len(SEEDS)
    # in every cycle that acted, the neural steps came in the order of spec 15.1
    seq = session["rec"].calls
    order = ["update_beliefs", "propose", "evaluate", "select", "record_result", "internal_rewards", "learn",
             "meta_learn"]
    starts = [i for i, c in enumerate(seq) if c == "propose"]
    assert len(starts) >= 15
    for i in starts:
        nxt = [c for c in seq[i:i + 40] if c in order]
        assert nxt[:3] == ["propose", "evaluate", "select"], nxt[:6]
        tail = [c for c in nxt[3:] if c != "record_result"]      # a wait's result arrives when the wait ends
        assert tail.index("internal_rewards") < tail.index("learn") < tail.index("meta_learn")
    # every selected decision's result was handed back (a wait still running at the end excepted)
    assert calls["record_result"] >= calls["select"] - len(SEEDS)
    statuses = [lt.status for lt in session["lifetimes"]]
    assert all(s in ("completed", "failed", "exhausted", "running") for s in statuses), statuses
    assert sum(s in ("completed", "failed", "exhausted") for s in statuses) >= 2, statuses


def test_an_idle_environment_asks_the_agent_when_to_wake(tmp_path):
    """Step 17: with nothing to do (no tool available) the runtime asks the agent for its next wake-up and sleeps."""
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
    from cam_live.harness.auth import PolicyService
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    from int_helpers import lab_config

    class NoTools(LabWorld):
        def discover_tools(self):
            return []

    clock = SimClock()
    agent = CamLiveAgent(agent_config(seed=8), str(tmp_path / "agent"), clock=clock)
    try:
        rec = Recorder(agent)
        world = NoTools(lab_config(81), clock)
        task = lab_task(world, "obj_idle")
        policy = PolicyService(str(tmp_path / "auth"), clock=clock)
        policy.register_task(task, "operator")
        rt = LiveRuntime(rec, world, policy, LabVerifier(world), task, str(tmp_path / "run"), clock,
                         RuntimeConfig(fsync=False))
        rep = rt.step()
        assert rep["cycle"] and not rep["acted"]
        assert "propose" in rec.calls and "next_wake" in rec.calls and "evaluate" not in rec.calls
        assert agent.next_wake() is None
        rt.close()
    finally:
        agent.close()


def test_predictions_precede_outcomes(session):
    agent = session["agent"]
    resolved = [p for p in agent.predictions.records() if p.resolved_status != OutcomeStatus.PENDING]
    assert len(resolved) >= 20
    for p in resolved:
        assert p.resolved_at is not None and p.resolved_at > agent.predictions.registered_at(p.prediction_id)
    # the runtime logged every chosen action's predictions before the action started
    for lt in session["lifetimes"]:
        ev = list(iter_jsonl(os.path.join(lt.run_dir, RunFiles.EVENTS)))
        start_seq = {e["data"]["action_id"]: e["seq"] for e in ev if e["type"] == "action_start"}
        preds = [e for e in iter_jsonl(os.path.join(lt.run_dir, RunFiles.PREDICTIONS)) if e["type"] != "resolution"]
        reg_seq = {}
        for e in preds:
            d = e["data"]
            aid = d.get("action_id") or d.get("prediction", {}).get("action_id", "")
            reg_seq.setdefault(aid, e["seq"])
        common = [a for a in start_seq if a in reg_seq]
        assert common, "no logged prediction belongs to an executed action"
        assert all(reg_seq[a] < start_seq[a] for a in common)
    # every resolution in the runtime's log refers to a prediction registered earlier in that log
    res = _logs(session, RunFiles.PREDICTIONS)
    by_rid = {e["rid"]: e["seq"] for e in res}
    lines = [e for e in res if e["type"] == "resolution"]
    assert lines
    for e in lines:
        reg = e["data"].get("registered_rid", "")
        if reg:
            assert by_rid[reg] < e["seq"]


def test_delayed_outcomes_link_to_their_decisions(session):
    st = session["agent"].store
    acts = [r for r in st.all() if r.decision.chosen.operation == "lab.act" and r.result and r.result.status == "ok"]
    matured = [r for r in acts if any(o.query_id in {q.query_id for q in r.result.expected_outcomes}
                                      and o.status == OutcomeStatus.OBSERVED for o in r.outcomes)]
    assert len(matured) >= 5, (len(acts), len(matured))
    for r in matured:
        links = st.links(exp_id=r.exp_id)
        assert links, r.exp_id
        assert all(l["decision_id"] == r.decision.decision_id for l in links)
        o = next(o for o in r.outcomes if o.query_id in {q.query_id for q in r.result.expected_outcomes})
        assert o.observed_at > r.decision.ts and o.verified           # measured later by the verifier
        assert session["agent"]._check_measurement(o) or o.signature == ""
    # arrival events (no declared query) are linked through the environment's attribution data
    event_links = [l for r in acts for l in st.links(exp_id=r.exp_id) if l.get("evidence", "") != "declared_query"]
    assert event_links


def test_internal_rewards_are_stored_per_record(session):
    recs = [r for r in session["agent"].store.all() if r.result is not None]
    with_r = [r for r in recs if r.intrinsic_reward is not None]
    assert len(with_r) >= len(recs) - 2 and len(with_r) >= 15
    assert all(r.bootstrap_reward is not None and r.reward_version for r in with_r)
    assert all(math.isfinite(r.intrinsic_reward) and abs(r.intrinsic_reward) <= 1.0 + 1e-6 for r in with_r)
    assert len({round(r.intrinsic_reward, 6) for r in with_r}) > 3     # not a constant


def test_learning_changes_experiential_and_adapter_weights(session):
    ups = [e["data"] for e in _logs(session, RunFiles.LEARNING)]
    by = collections.defaultdict(list)
    for u in ups:
        by[u["timescale"]].append(u)
    freq = [u for u in by["frequent"] if u.get("weight_delta", {}).get("experiential", 0.0) > 0]
    assert len(freq) >= 5
    learner_freq = [u for u in freq if "experiential" in (u.get("hash_before") or {})]
    assert learner_freq and all(u["hash_before"]["experiential"] != u["hash_after"]["experiential"]
                                for u in learner_freq)
    slow = [u for u in by["slow_adapters"] if u.get("weight_delta", {}).get("semantic_adapters", 0.0) > 0]
    assert slow, [u.get("notes", "")[:200] for u in by["slow_adapters"]]
    assert all("reward" not in u["param_groups"] for u in by["frequent"] + by["slow_adapters"])
    assert by["fast"], "no fast belief update was recorded"
    policy = [u for u in by["frequent"] if u["trigger"].endswith("newly rewarded decisions")]
    assert policy and all(u["weight_delta"]["heads"] > 0 for u in policy)


def test_meta_update_runs_with_a_nonzero_meta_gradient(session):
    metas = [e["data"] for e in _logs(session, RunFiles.META)]
    applied = [m for m in metas if m["applied"]]
    skipped = collections.Counter(m["skipped_reason"].split(":")[0] for m in metas if not m["applied"])
    assert applied, f"no meta-update was applied; skipped reasons: {dict(skipped)}"
    nonzero = [m for m in applied if m["meta_grad_norm"] > 0]
    assert nonzero, [m["meta_grad_norm"] for m in applied]
    for m in nonzero:
        assert m["reward_version_before"] != m["reward_version_after"]
        assert m["diagnostics"]["psi_tensors_with_gradient"] > 0
        assert m["n_independent_outcomes"] >= 3
    assert all(m["n_independent_outcomes"] >= 3 for m in applied)
    assert all(m["skipped_reason"] for m in metas if not m["applied"])   # every skip says why


def _at_initial_chain_state(agent, probe):
    """The probe's distribution at the persistent core-state chain's initial state (nothing carried; ``nn.persist``,
    PS): the chain is experience-dependent state like the fast beliefs, so what the weights carry is compared at equal
    chain state. ``session["p0"]`` was taken before any decision, i.e. at that state too. Without a chain (the default)
    this is ``probe.distribution()``."""
    chain = getattr(agent, "persist", None)
    with (chain.initial() if chain is not None else contextlib.nullcontext()):
        return probe.distribution()


def test_behaviour_changes_and_lives_in_the_weights(session):
    agent, probe, p0, s0 = session["agent"], session["probe"], session["p0"], session["s0"]
    dist = lambda: _at_initial_chain_state(agent, probe)          # noqa: E731
    p1 = dist()
    s1 = agent.model.trainable_state()
    assert _tv(p0, p1) > 1e-4, (p0, p1)
    try:
        weight_swap(agent.model, s0, TRAINABLE)                     # store, fast state, registries untouched
        assert _tv(dist(), p0) < 1e-9
        weight_swap(agent.model, s1, TRAINABLE)
        weight_swap(agent.model, s0, ("experiential",))             # only the experiential group goes back
        partial = dist()
        assert _tv(partial, p1) > 1e-6                               # the experiential weights carry part of it
        weight_swap(agent.model, s1, ("experiential",))
        assert _tv(dist(), p1) < 1e-9
        # in the other direction: the learned experiential weights alone change the old model's decisions, and
        # restoring only them restores the old distribution exactly
        weight_swap(agent.model, s0, TRAINABLE)
        weight_swap(agent.model, s1, ("experiential",))
        assert _tv(dist(), p0) > 1e-6
        weight_swap(agent.model, s0, ("experiential",))
        assert _tv(dist(), p0) < 1e-9
    finally:
        weight_swap(agent.model, s1, TRAINABLE)


def test_new_objective_keeps_weights_and_resets_fast_beliefs(tmp_path):
    clock = SimClock()
    agent = CamLiveAgent(agent_config(seed=3), str(tmp_path / "agent"), clock=clock)
    try:
        lt = run_lifetime(Recorder(agent), clock, str(tmp_path), 31, max_steps=10)
        assert agent.learner.counters["frequent"] > 0
        fast_before = agent.learner.fast_state.state_dict()
        h, rv, life = agent.model.group_hashes(), agent.meta.reward.version, agent.meta._life
        prior = agent.model.new_fast_state().state_dict()
        assert not _same_state(fast_before, prior)                   # the beliefs had moved away from the prior
        agent.set_objective(type(lt.task).from_dict({**lt.task.to_dict(), "objective_id": "obj_next",
                                                     "objective": "Reach 5 adopters in the lab world."}))
        assert agent.model.group_hashes() == h and agent.meta.reward.version == rv
        assert agent.meta._life == life + 1 and agent.lifetime == 1
        assert _same_state(agent.learner.fast_state.state_dict(), prior)
        assert agent.predictor.fast is agent.learner.fast_state
        assert agent.ctx == [agent.goal_ref]
    finally:
        agent.close()


def _same_state(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_state(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return a == b


def test_verified_outcome_supervises_only_after_the_signature_checks(tmp_path):
    clock = SimClock()
    agent = CamLiveAgent(agent_config(seed=4), str(tmp_path / "agent"), clock=clock)
    try:
        from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
        from int_helpers import lab_config
        world = LabWorld(lab_config(41), clock)
        task = lab_task(world, "obj_sig")
        agent.set_objective(task)
        clock.advance(10 * 86400.0)
        v = LabVerifier(world).verify(task)
        assert v.status == GoalStatus.FAILED and v.signature
        out = agent.reconcile([], v)                                 # no verifier attached: refused
        assert out["verified"]["accepted"] is False and agent.store.objective_outcome("obj_sig") is None
        agent.attach_verifier(LabVerifier(world, secret=b"another key"))
        assert agent.reconcile([], v)["verified"]["accepted"] is False
        forged = VerifiedOutcome.from_dict({**v.to_dict(), "status": GoalStatus.ACHIEVED})
        agent.attach_verifier(LabVerifier(world))
        assert agent.reconcile([], forged)["verified"]["accepted"] is False
        ok = agent.reconcile([], v)["verified"]
        assert ok["accepted"] and ok["closed"] and ok["evaluator"]
        assert agent.store.objective_outcome("obj_sig")["verified"] is True
    finally:
        agent.close()


def test_goal_free_mode_chooses_and_resolves_its_own_learning_goals(tmp_path):
    """No assigned objective: GoalFreeMode proposes budgeted learning goals from the fast posterior, the policy
    chooses among them, their outcomes resolve them; nothing of it reaches the meta ledger (AE-04)."""
    from cam_live.contracts import Limits, TaskSpec, Trust
    from cam_live.diagnostics.world import LabVerifier, LabWorld
    from int_helpers import DriverState, drive, lab_config
    clock = SimClock()
    world = LabWorld(lab_config(71), clock)
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(seed=9), str(tmp_path / "agent"), clock=clock, verifier=verifier)
    try:
        agent.set_objective(TaskSpec(objective_id="free_71", objective="", success={}, family="lab/integration",
                                     limits=Limits(duration_s=2 * 86400.0, money=40.0, max_decisions=40),
                                     mode="diagnostic", origin=Trust.SELF))
        assert agent._goal_free_active()
        out = drive(agent, world, clock, verifier, DriverState(), 10)
        goals = agent.goal_free.goals
        assert goals, "no learning goal was proposed"
        assert any("goal_free" in r.decision.chosen.provenance for r in agent.store.all())
        assert any(g.status != GoalStatus.PENDING for g in goals)
        assert all(d["rule"] in ("voi", "randomized") for d in out)
        assert len(agent.meta.ledger) == 0                         # self-chosen goals never supervise psi
    finally:
        agent.close()
