"""Planner mechanics: matched transition budgets, choice = best imagined return, CAM_PLANNER provenance on the
chosen CAM candidate, DecisionTrace.imagined entries, and the real state left untouched."""
import copy
import json

import pytest
import torch

from superlearner.cam_unified.interfaces import ActionSource, DecisionTrace
from superlearner.cam_unified.testing import DoubleCore, tiny_config
from superlearner.cam_unified.world.dynamics import WorldModel
from superlearner.cam_unified.world.imagination import Imagination
from superlearner.cam_unified.world.planner import (PLANNERS, BeamPlanner, PolicyRolloutPlanner, SamplingPlanner,
                                                    compare_planners)

from world_synth import HypothesisBelief, ProbeWorld, probe_candidates, probe_hooks, real_start


def _setup(seed=0):
    torch.manual_seed(seed)
    cfg = tiny_config()
    w = ProbeWorld(cfg.d_model)
    bel = HypothesisBelief(cfg, w)
    wm = WorldModel(cfg, DoubleCore(cfg)).eval()
    with torch.no_grad():                                   # make imagined rewards differ between candidates
        wm.rew_head.l2.weight.normal_(0, 0.05)
    Z, fast = real_start(wm, w, bel, 3, torch.device("cpu"))
    return cfg, w, bel, wm, Z, fast


@pytest.mark.parametrize("kind", ["policy_rollout", "beam", "cem"])
def test_planner_budget_choice_and_trace(kind):
    cfg, w, bel, wm, Z, fast = _setup()
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=3, outcome_sampling="marginal")
    budget = 96
    Z0, f0 = Z.clone(), copy.deepcopy(fast)
    res = PLANNERS[kind](im, budget).plan(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    B, n = 3, 4
    # matched compute: never above the budget, and uses a substantial part of it
    assert res.n_model_evals <= budget * B and res.n_model_evals >= budget * B // 4, res.n_model_evals
    # the choice is the candidate with the best imagined return under this planner's estimate
    assert res.scores.shape == (B, n)
    assert torch.equal(res.choice, res.scores.argmax(1))
    for b in range(B):
        a = res.actions[b]
        i = int(res.choice[b])
        assert a.source == ActionSource.CAM_PLANNER and a.tool == cand.actions[b][i].tool
        assert cand.actions[b][i].source == ActionSource.CAM_DECODER          # the candidate itself is not mutated
        assert a.proposal["chosen"] == i and a.proposal["n"] == n and a.proposal["method"] == res.method
        assert abs(sum(a.proposal["probs"]) - 1.0) < 1e-5
        entries = res.imagined[b]
        assert sorted(e["candidate"] for e in entries) == list(range(n))
        assert all(e["provenance"] == "imagined" for e in entries)
        tr = DecisionTrace(decision_id="d", event_id="e", env_id=f"env{b}", policy_version=0, t_sim=0.0,
                           objective="probe", candidates=[{"action": c.to_dict()} for c in cand.actions[b]],
                           imagined=entries, selected=a.to_dict(), source=a.source.value)
        json.loads(tr.to_json())
    assert torch.equal(Z, Z0) and torch.equal(fast.logw, f0.logw) and torch.equal(fast.particles, f0.particles)


def test_planner_sizes_follow_the_budget():
    cfg, w, bel, wm, Z, fast = _setup(1)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    res = compare_planners(im, 64, Z, fast, None, cand, hooks, horizon=2)
    assert set(res) == {"policy_rollout", "beam", "cem"}
    assert res["policy_rollout"].info["n_samples"] == 64 // (4 * 2)
    assert res["beam"].info["n_samples"] == 64 // (4 + 1 * 2 * 4)
    assert res["cem"].info["plans_per_iter"] == 64 // (3 * 2 * 2)
    evals = [r.n_model_evals for r in res.values()]
    assert max(evals) <= 64 * 3 and min(evals) >= 64 * 3 // 3, evals
    big = PolicyRolloutPlanner(im, 256, horizon=2).plan(Z, fast, None, cand, hooks)
    assert big.info["n_samples"] == 32 and big.n_model_evals == 32 * 4 * 2 * 3


def test_beam_expands_only_the_best_nodes():
    cfg, w, bel, wm, Z, fast = _setup(2)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=3, outcome_sampling="marginal")
    res = BeamPlanner(im, 200, beam_width=2).plan(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    first = res.info["first_step_scores"]
    for b in range(3):
        kept = set(first[b].topk(2).indices.tolist())
        finite = set(torch.nonzero(torch.isfinite(res.scores[b])).flatten().tolist())
        assert finite <= kept and finite        # roots pruned after depth 1 have no final score


def test_cem_concentrates_on_the_best_first_action():
    cfg, w, bel, wm, Z, fast = _setup(3)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    res = SamplingPlanner(im, 240, iterations=4, n_samples=2).plan(Z, fast, None, cand, hooks,
                                                                   generator=torch.Generator().manual_seed(0))
    P = res.info["final_P"]
    assert torch.allclose(P.sum(-1), torch.ones_like(P.sum(-1)), atol=1e-5)
    # the final first-action distribution puts its largest mass on the chosen candidate's neighbourhood of value
    top = P[:, 0].argmax(1)
    sc = res.scores
    assert all(float(sc[b, top[b]]) >= float(sc[b][torch.isfinite(sc[b])].median()) for b in range(3))
