"""W1 (plan 2026-11-26 §2): the wrappers over the canonical loop reproduce the pre-v4 loops they replaced.

The reference implementations are the loop bodies at tag seam-0 (runtime.run_steps, envs/harness.run_lifetime,
experiments/mechrun.run_batch), loaded from git into throwaway modules of the same package, and run side by side
with the wrappers on identical seeds:

* run_steps: CamUnifiedAgent (CPU doubles) on ToyShop, 3 episodes with resets and 3 updates - identical step rows,
  returns, goal scores, episode seeds, archived actions / observations / rewards / dt, and bit-identical weights.
* envs.harness.run_lifetime: the ExampleInductor (PolicyAgent) on a real ow_tiny OpenWorld - identical decisions,
  results, rewards, durations, success day, final verified result and counts. The one intended difference: each
  record's observations now also hold the previous action's tool results (the old harness dropped them), and
  nothing else.
* mechrun.run_batch: NIGToolAgent on real Crucible mechanism worlds (plain, predecessor notes, condition D) -
  identical EpisodeResults and per-round observations / rewards.
"""
import copy
import json
import os
import subprocess
import sys
import types

import pytest
import torch

from conftest import Stack
from superlearner.cam_unified.interfaces import ObsKind

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SEAM = "seam-0"


def seam_module(rel: str, pkg: str):
    """The file `rel` (relative to cam_unified/) as it was at tag seam-0, executed as a module of package `pkg`."""
    try:
        src = subprocess.run(["git", "show", f"{SEAM}:superlearner/src/superlearner/cam_unified/{rel}"], cwd=REPO,
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"git / tag {SEAM} unavailable: {e}")
    name = f"{pkg}._seam0_{rel.replace('/', '_')[:-3]}"
    mod = types.ModuleType(name)
    mod.__package__ = pkg
    mod.__file__ = f"{SEAM}:{rel}"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


V4_EXTRA = ("versions", "schema", "reward_mode")


def _rec_view(r, obs=True):
    a = r.action
    ex = {k: v for k, v in r.extra.items() if k not in V4_EXTRA and k != "decision_id"}
    return {"env": r.env_id, "step": r.step, "t": r.t_sim, "tool": a.tool if a else None,
            "args": json.dumps(a.args, sort_keys=True, default=str) if a else None, "reward": r.reward,
            "ok": r.result.ok if r.result else None, "dt": ex.get("dt", ex.get("dt_days")),
            "obs": [o.to_json() for o in r.observations] if obs else len(r.observations),
            "logp": r.behavior_logp, "extra": json.dumps({k: v for k, v in ex.items()
                                                          if k not in ("retrieved",)}, sort_keys=True, default=str)}


# ============================================================================ run_steps

def _steps_run(run_steps):
    st = Stack(seed=3)
    st.env.horizon = 0.25                     # several episodes in 12 rounds
    rep = run_steps(st.agent, st.env, 12, evaluator=st.evaluator, update_every=4, seeds=[11, 12])
    return st, rep


def test_run_steps_wrapper_reproduces_the_seam0_loop():
    old = seam_module("runtime.py", "superlearner.cam_unified")
    from superlearner.cam_unified.runtime import run_steps
    so, ro = _steps_run(old.run_steps)
    sn, rn = _steps_run(run_steps)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "update"} | {"has_update": "update" in r} for r in rows]
    assert strip(rn.steps) == strip(ro.steps)
    assert rn.returns == ro.returns and rn.goal_scores == ro.goal_scores and rn.episode_seeds == ro.episode_seeds
    assert len(ro.episode_seeds) >= 2                                    # resets happened
    assert [u.get("updated") for u in rn.updates] == [u.get("updated") for u in ro.updates] == [True] * 3
    assert rn.compute.decisions == ro.compute.decisions and rn.compute.episodes == ro.compute.episodes
    assert rn.compute.sim_days == pytest.approx(ro.compute.sim_days)
    assert [_rec_view(r) for r in sn.archive.rows] == [_rec_view(r) for r in so.archive.rows]
    po, pn = dict(so.model.named_parameters()), dict(sn.model.named_parameters())
    assert all(torch.equal(po[k], pn[k]) for k in po)
    assert sn.agent.version == so.agent.version == 3


# ============================================================================ harness (real OpenWorld)

def _harness_run(run_lifetime, tmp, name):
    import crucible
    crucible.load_all_modules()
    from crucible.research import llm_eval as L
    from crucible.usefulness.agents import ExampleInductor
    from superlearner.cam_unified.envs.agent_protocol import PolicyAgent
    from superlearner.cam_unified.envs.openworld import open_world
    from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator
    goal = L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=8,
                               window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2,
                               endowment=3_000_000)
    ad, op = open_world([goal], seed=0)
    ev = OpenWorldEvaluator(op)
    path = os.path.join(tmp, f"{name}.jsonl")
    seen = []
    try:
        res = run_lifetime(PolicyAgent(lambda b: ExampleInductor()), ad, ev, think_s=600, transcript=path,
                           update_every=24, on_step=lambda k, tr: seen.append(tr))
    finally:
        ad.close()
    from superlearner.cam_unified.interfaces import EventRecord
    recs = [EventRecord.from_dict(json.loads(x)) for x in open(path).read().splitlines()[1:]]
    return res, recs, seen


def test_harness_wrapper_reproduces_the_seam0_harness_plus_tool_results(tmp_path):
    old = seam_module("envs/harness.py", "superlearner.cam_unified.envs")
    from superlearner.cam_unified.envs.harness import run_lifetime
    ro, reco, seeno = _harness_run(old.run_lifetime, str(tmp_path), "old")
    rn, recn, seenn = _harness_run(run_lifetime, str(tmp_path), "new")
    assert rn.steps == ro.steps and rn.returns == ro.returns and rn.success_day == ro.success_day
    assert rn.elapsed_days == ro.elapsed_days and rn.t_days == ro.t_days
    assert rn.final[0]["achieved"] == ro.final[0]["achieved"] is True
    assert rn.final[0]["score"] == ro.final[0]["score"]
    for k in ("decisions", "actions", "waits", "rejected", "skipped", "by_tool", "errors"):
        assert rn.accounting[k] == ro.accounting[k], k
    assert [sum(t.rewards) for t in seenn] == [sum(t.rewards) for t in seeno]
    view = lambda r: _rec_view(r, obs=False) | {"obs": None, "env": None}   # env ids are random per world
    assert [view(r) for r in recn] == [view(r) for r in reco]
    ids_o = {r.action.action_id: f"action#{i}" for i, r in enumerate(reco) if r.action is not None}
    ids_n = {r.action.action_id: f"action#{i}" for i, r in enumerate(recn) if r.action is not None}

    def norm(x, ids):                                       # action ids are random per process
        d = x.to_dict()
        d["caused_by"] = ids.get(d.get("caused_by"), d.get("caused_by"))
        return json.dumps(d, sort_keys=True, default=str)
    extra_total = 0
    for i, (n, o) in enumerate(zip(recn, reco)):
        old_obs = [norm(x, ids_o) for x in o.observations]
        new_obs = [norm(x, ids_n) for x in n.observations]
        rest = list(new_obs)
        for x in old_obs:                                   # every old observation is still delivered, in order
            assert x in rest
            rest.remove(x)
        prev = recn[i - 1] if i > 0 else None
        want = [norm(x, ids_n) for x in prev.result.observations] if prev is not None and prev.result else []
        assert sorted(rest) == sorted(want)                 # the additions are exactly the previous tool results
        extra_total += len(rest)
    assert extra_total > 0
    assert all(r.extra["schema"] == "cam-event/4" and r.extra["reward_mode"] == "legacy" for r in recn)


# ============================================================================ mechrun (real mechanism worlds)

@pytest.fixture(scope="module")
def mech_tasks():
    from superlearner.cam_unified.experiments import mechrun as M
    return M.eval_task_set(["scm_additive", "cw1_scm_delay_stock"], 1, offset=100_000)


def _mech_run(run_batch, tasks, cond):
    import random
    from superlearner.cam_unified.experiments import mechrun as M
    from superlearner.cam_unified.experiments.nig_agent import NIGToolAgent
    w = M.make_world(tasks, seed=5)
    kw = {}
    if cond == "notes":
        notes, nrng = M.PredecessorNotes(), random.Random(9)
        kw["notes"] = lambda world: [notes.notes(world, b, 2, nrng) for b in range(world.B)]
    elif cond == "delayed":
        kw["delayed"] = M.DelayedPartial(w, delay=3, hide_frac=0.5, seed=4)
    trace, ids = [], {}

    def norm(o):                                            # action ids are random per process
        d = o.to_dict()
        d["caused_by"] = ids.get(d.get("caused_by"), d.get("caused_by"))
        return json.dumps(d, sort_keys=True, default=str)

    def rec(r, acts, res, rew, nob):
        for b, a in enumerate(acts or []):
            if a is not None:
                ids[a.action_id] = f"a#{r}:{b}"
        trace.append((r, None if acts is None else [a.to_dict()["args"] if a else None for a in acts],
                      None if rew is None else list(rew),
                      [[norm(o) for o in obs] for obs in nob.observations], list(nob.t), list(nob.done)))
    out = run_batch(NIGToolAgent("kg", seed=2), w, learn=False, max_rounds=40, on_round=rec, **kw)
    return [r.to_dict() for r in out], trace


@pytest.mark.parametrize("cond", ["plain", "notes", "delayed"])
def test_mechrun_wrapper_reproduces_the_seam0_driver(mech_tasks, cond):
    old = seam_module("experiments/mechrun.py", "superlearner.cam_unified.experiments")
    from superlearner.cam_unified.experiments import mechrun as M
    ro, to = _mech_run(old.run_batch, copy.deepcopy(mech_tasks), cond)
    rn, tn = _mech_run(M.run_batch, copy.deepcopy(mech_tasks), cond)
    assert rn == ro
    assert len(tn) == len(to) == 41 and tn == to
