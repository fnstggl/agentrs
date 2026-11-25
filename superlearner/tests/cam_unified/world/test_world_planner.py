"""Planner mechanics: matched transition budgets (refused when unmeetable), choice = best imagined return (checked
against a deterministic oracle with a known best multi-step plan), CAM_PLANNER provenance with an unknown behaviour
probability, DecisionTrace.imagined entries, and the real state left untouched."""
import copy
import json
import math

import pytest
import torch

from superlearner.cam_unified.interfaces import Action, ActionSource, DecisionTrace, FastState, OutcomeBatch
from superlearner.cam_unified.testing import DoubleCore, tiny_config, zero_fast_state
from superlearner.cam_unified.world.dynamics import WorldModel
from superlearner.cam_unified.world.imagination import Candidates, Imagination, ImaginationHooks, StepContext
from superlearner.cam_unified.world.planner import (PLANNERS, BeamPlanner, PolicyRolloutPlanner, SamplingPlanner,
                                                    compare_planners)

from world_synth import HypothesisBelief, InvestWorld, ProbeWorld, probe_candidates, probe_hooks, real_start


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
    assert res.scores.shape == (B, n)
    for b in range(B):
        a = res.actions[b]
        i = int(res.choice[b])
        assert a.source == ActionSource.CAM_PLANNER and a.tool == cand.actions[b][i].tool
        assert cand.actions[b][i].source == ActionSource.CAM_DECODER          # the candidate itself is not mutated
        # an argmax over sampled candidates has no known behaviour probability: no logp; decoder value kept apart
        assert a.logp is None and a.proposal["behavior"] == "planner_argmax"
        assert a.logp_parts["decoder"] == pytest.approx(cand.actions[b][i].logp)
        assert cand.actions[b][i].logp == pytest.approx(-math.log(n))
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


# ------------------------------------------------------------------------------------------------ budget refusal

@pytest.mark.parametrize("kind", ["policy_rollout", "beam", "cem"])
def test_planner_refuses_a_budget_below_its_minimum_cost(kind):
    cfg, w, bel, wm, Z, fast = _setup(4)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=4, outcome_sampling="marginal")
    n, h = 4, 4
    need = {"policy_rollout": n * h, "beam": n + (h - 1) * 2 * n, "cem": 3 * 2 * h}[kind]
    assert PLANNERS[kind](im, 1).min_budget(n) == need
    for budget in (4, 8, need - 1):
        with pytest.raises(ValueError):
            PLANNERS[kind](im, budget).plan(Z, fast, None, cand, hooks)
    res = PLANNERS[kind](im, need).plan(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    assert 0 < res.n_model_evals <= need * 3


def test_compare_planners_refuses_a_budget_any_planner_cannot_meet():
    cfg, w, bel, wm, Z, fast = _setup(5)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 3, Z.device)
    im = Imagination(wm, horizon=4, outcome_sampling="marginal")
    with pytest.raises(ValueError):                 # enough for policy rollouts (16) and CEM (24), not beam (28)
        compare_planners(im, 24, Z, fast, None, cand, hooks, horizon=4)
    res = compare_planners(im, 28, Z, fast, None, cand, hooks, horizon=4)
    assert all(r.n_model_evals <= 28 * 3 for r in res.values())


# ------------------------------------------------------------------------------------------------ oracle choice

class _InvestOracle:
    """Deterministic oracle on top of an (untrained) world model: invest costs 1 and builds capacity, sell pays 5
    with capacity (else -0.5), idle 0. The capacity flag lives in each imagined row's cloned FastState
    (surprise[:, 0]); the best 2-step plan is invest -> sell (+4) although invest is the worst first step."""

    def __init__(self, w: InvestWorld):
        self.w = w
        self.emb = torch.stack([w.E("act:" + a) for a in w.actions])          # [3, D]

    def action_ids(self, act_tokens) -> torch.Tensor:
        return torch.cdist(act_tokens.x[:, 0].float(), self.emb.to(act_tokens.x.device)).argmin(-1)

    def update(self, fast, Zp, act_tokens, ob):
        a = self.action_ids(act_tokens)
        built = fast.surprise[:, 0]
        built = torch.where(a == 1, torch.ones_like(built), torch.where(a == 2, torch.zeros_like(built), built))
        sur = fast.surprise.clone()
        sur[:, 0] = built
        return FastState(fast.particles.clone(), fast.logw.clone(), fast.pscale.clone(), fast.bayes_mu.clone(),
                         fast.bayes_P.clone(), fast.bayes_noise.clone(), fast.bayes_n.clone(), sur,
                         fast.steps + 1, [dict(c) for c in fast.channels])

    def reward(self, ctx: StepContext):
        assert isinstance(ctx, StepContext) and ctx.Z_next.shape == ctx.Z_prev.shape and len(ctx.actions) == \
            ctx.Z_next.shape[0] and ctx.fast is not None and ctx.dt.shape == (ctx.Z_next.shape[0],)
        built = ctx.fast_prev.surprise[:, 0]
        r = [(-1.0 if a.tool == "invest" else (5.0 * float(bt) - 0.5 * (1 - float(bt)) if a.tool == "sell" else 0.0))
             for a, bt in zip(ctx.actions, built.tolist())]
        return torch.tensor(r, device=ctx.Z_next.device)

    def hooks(self):
        names = self.w.actions

        def policy(ctx):                       # sell when the imagined capacity exists, otherwise idle
            ids = [2 if float(b) > 0.5 else 0 for b in ctx.fast.surprise[:, 0].tolist()]
            return [Action(names[i]) for i in ids], self.w.act_tokens(ids)

        def queries(Zp, actions, toks, k):     # one query, never observable: only the action changes the state
            q = self.w.queries(len(actions))
            return OutcomeBatch(q.q, q.lik, q.n_classes, q.y, q.censored, torch.zeros_like(q.y_mask), q.channel,
                                q.keys)

        return ImaginationHooks(policy_fn=policy, outcome_queries_fn=queries, belief_update_fn=self.update,
                                reward_fn=self.reward)


def test_planners_choose_the_oracle_best_multistep_plan():
    torch.manual_seed(0)
    cfg = tiny_config()
    w = InvestWorld(cfg.d_model)
    wm = WorldModel(cfg, DoubleCore(cfg)).eval()
    B, n = 2, 3
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    fast = zero_fast_state(B, cfg)
    cand = Candidates([[Action(a, logp=-math.log(n)) for a in w.actions] for _ in range(B)],
                      w.act_tokens(list(range(n)) * B))
    oracle = _InvestOracle(w)
    hk = oracle.hooks()
    im = Imagination(wm, horizon=2, kappa=0.0, use_continue=False)          # undiscounted: exact oracle values
    res = compare_planners(im, 120, Z, fast, None, cand, hk, horizon=2)
    res["beam_full_width"] = BeamPlanner(im, 120, horizon=2, beam_width=n).plan(
        Z, fast, None, cand, hk, generator=torch.Generator().manual_seed(0))
    truth = torch.tensor([0.0, 4.0, -0.5])                                  # idle..., invest->sell, sell->idle
    for k in ("policy_rollout", "beam_full_width"):
        torch.testing.assert_close(res[k].scores, truth.expand(B, -1), atol=1e-5, rtol=0)
    for k in ("policy_rollout", "beam_full_width", "cem"):
        assert res[k].choice.tolist() == [1] * B, (k, res[k].scores)
        assert all(a.tool == "invest" for a in res[k].actions)
    # CEM's best plan per first candidate is the true best continuation
    assert all(e["plan"] == [1, 2] for b in range(B) for e in res["cem"].imagined[b] if e["candidate"] == 1)
    # a width-2 beam keeps idle (0) and sell (-0.5) after depth 1 and prunes the investment
    assert not bool(torch.isfinite(res["beam"].scores[:, 1]).any())
    assert res["beam"].choice.tolist() == [0] * B
    assert torch.equal(fast.surprise, torch.zeros_like(fast.surprise))       # real FastState untouched
