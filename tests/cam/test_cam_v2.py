"""CAM-v2 tests: reward contract and audits, semi-Markov time, structured off-policy learning,
gradient boundaries, imagination learning, replay provenance, archive burn-in, checkpoints."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from agentrs.cam.audit import audit, rollout
from agentrs.cam.contract import RewardContract, RewardSpec
from agentrs.cam.envs import ENVS
from agentrs.cam.v2.collector import Collector
from agentrs.cam.v2.data import Archive, Replay
from agentrs.cam.v2.dream import Dreamer
from agentrs.cam.v2.learner import OffPolicy, decision_logprob, enumerate_actions, observe, _grammar
from agentrs.cam.v2.model import CAM2, CAM2Config

TINY = dict(d=48, deter=64, stoch=4, classes=8, hidden=64, layers=1, heads=2, n_options=3, ens=3)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return CAM2(CAM2Config(**TINY))


def _eps(m, fams, n=24, seed=1, archive=None, frac=0.0, difficulty=None):
    col = Collector(m, 8, fams, "train", gen_seed=seed, archive=archive, archive_frac=frac, difficulty=difficulty or {})
    return col.collect(min_episodes=n)


# ---------------------------------------------------------------------------- reward contract
def test_audit_flags_the_original_reward_and_passes_the_contract():
    bad = audit(RewardContract(RewardSpec(mode="legacy")), ("product", "negotiation"), n=8)
    assert not bad["passed"] and any("terminating early" in v for v in bad["violations"])
    good = audit(RewardContract(RewardSpec(mode="contract")), ("product", "negotiation", "bizdev"), n=8, lam=0.5)
    assert good["passed"], good["violations"]
    p = good["families"]["product"]["policies"]
    assert p["early_terminate"]["J"] < p["preserve"]["J"] < p["succeed"]["J"]  # bankruptcy is strictly worse with lam > 0


def test_semi_markov_discounting_makes_wait_granularity_irrelevant():
    def run(policy, mode):
        rc = RewardContract(RewardSpec(mode=mode, shaping=0.5))
        return rollout("product", policy, rc, seed=7)["J"]

    def w4(e, st):
        return ["WAIT", "W4"] if st.setdefault("n", 0) == 0 and not st.update(n=1) else ["WAIT", "W1"]

    def w1(e, st):
        return ["WAIT", "W1"]

    assert abs(run(w4, "contract") - run(w1, "contract")) < 1e-9   # same simulated time, same outcome
    assert abs(run(w4, "legacy") - run(w1, "legacy")) > 1e-4       # per-action discount: depends on action count


def test_every_failure_has_the_same_time_cost_regardless_of_when_it_ends():
    rc = RewardContract(RewardSpec(mode="contract", shaping=0.0))
    early = rollout("negotiation", lambda e, s: ["WALK"], rc, seed=3)
    late = rollout("negotiation", lambda e, s: ["ASK", "price"], rc, seed=3)
    assert not early["success"] and not late["success"] and early["clock"] < late["clock"]
    assert abs(early["J"] - late["J"]) < 1e-9


# ---------------------------------------------------------------------------- structured actions
def test_enumerated_actions_form_an_exact_masked_distribution(model):
    acts = enumerate_actions(_grammar("product"))
    assert len(acts) == 33
    op = OffPolicy(model, {"actor": "sac"})
    f = torch.randn(3, model.F)
    g = torch.randn(3, model.cfg.d)
    lp = op._policy_all("product", f, g, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-4)  # probabilities over every valid action sum to 1


def test_recorded_decision_logprob_is_exact(model):
    gr = [_grammar("product")] * 4
    f, g = torch.randn(4, model.F), torch.randn(4, model.cfg.d)
    opt = torch.tensor([0, 1, 2, 0])
    with torch.no_grad():
        toks, ids, lps = model.generate_action(f, g, opt, gr)
        al = [[model.allowed_ids(gr[i], toks[i][:k]) for k in range(len(toks[i]))] for i in range(4)]
        lp, _ = model.action_logprob(f, g, opt, ids, al)
    assert torch.allclose(lp, torch.tensor([sum(x) for x in lps]), atol=1e-5)


# ---------------------------------------------------------------------------- learning + boundaries
def test_offpolicy_update_respects_gradient_boundaries(model):
    eps = _eps(model, {"product": 0.5, "negotiation": 0.5})
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    op = OffPolicy(model, {"actor": "awr"})
    groups = model.groups()
    snap = {k: [p.detach().clone() for p in v] for k, v in groups.items()}
    st = op.update(rp.sample(8, random.Random(0), her_p=0.5), {}, random.Random(0), train_wm=False)
    changed = {k: any((p.detach() - s).abs().max() > 0 for p, s in zip(groups[k], snap[k])) for k in groups}
    assert changed == {"wm": False, "actor": True, "critic": True}, changed   # no actor/critic gradient reaches the WM
    assert st["decisions"] > 0 and np.isfinite(st["q_loss"])
    st = op.update(rp.sample(8, random.Random(1)), {}, random.Random(1), train_wm=True)
    assert np.isfinite(st["wm_kl_dyn"]) and np.isfinite(st["wm_rew"])


def test_sac_actor_update_runs_on_enumerable_families(model):
    eps = _eps(model, {"product": 0.5, "bizdev": 0.5}, n=16)
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    op = OffPolicy(model, {"actor": "sac"})
    st = op.update(rp.sample(6, random.Random(0)), {}, random.Random(0))
    assert any(k.startswith("sac_alpha_") for k in st)


def test_imagination_changes_actor_weights_and_decisions_but_not_the_world_model(model):
    eps = _eps(model, {"product": 1.0}, n=12)
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    op = OffPolicy(model, {"actor": "awr"})
    views = rp.sample(6, random.Random(0))
    with torch.no_grad():
        ob = observe(model, views)
    cand = [(i, t) for i, v in enumerate(views) for t in range(len(v.steps))][:24]
    k = torch.tensor([ob["pos"][x] for x in cand])
    f = ob["feat"][k]
    gi = torch.tensor([i for i, _ in cand])
    steps = [views[i].steps[t] for i, t in cand]
    args = (f[:, :model.cfg.deter], f[:, model.cfg.deter:], ob["goal"][gi], ["product"] * len(cand),
            torch.tensor([s.prev_opt for s in steps]), torch.tensor([s.first for s in steps]), {})
    with torch.no_grad():
        lp0, _ = decision_logprob(model, f, ob["goal"][gi], steps, [_grammar("product")] * len(cand))
    groups = model.groups()
    snap = {kk: [p.detach().clone() for p in v] for kk, v in groups.items()}
    torch.manual_seed(0)
    d0 = Dreamer(model, {"mode": "fixed", "horizon": 3, "kappa": 0.0})
    d0.update(op.opt["actor"], op.opt["critic"], *args)
    assert all((p.detach() - s).abs().max() == 0 for p, s in zip(groups["actor"], snap["actor"]))  # kappa 0: no actor change
    d = Dreamer(model, {"mode": "fixed", "horizon": 3, "kappa": 1.0})
    out = d.update(op.opt["actor"], op.opt["critic"], *args)
    assert out["imagined_transitions"] == 3 * len(cand)
    assert any((p.detach() - s).abs().max() > 0 for p, s in zip(groups["actor"], snap["actor"]))
    assert all((p.detach() - s).abs().max() == 0 for p, s in zip(groups["wm"], snap["wm"]))  # world model untouched
    with torch.no_grad():
        lp1, _ = decision_logprob(model, f, ob["goal"][gi], steps, [_grammar("product")] * len(cand))
    assert (lp1 - lp0).abs().max() > 1e-6   # imagined experience changed the policy's decisions


def test_adaptive_imagination_is_off_for_an_unreliable_model(model):
    d = Dreamer(model, {"mode": "adaptive", "horizon": 8})
    d.set_reliability({"ret_abs_err_k1": float("nan"), "viol_ece_k1": 0.5, "u_err_spearman": float("nan")})
    assert d.reliability == 0.0 and d.u_thr == 0.0
    d.set_reliability({"ret_abs_err_k1": 0.01, "ret_abs_err_k2": 0.02, "ret_abs_err_k4": 0.2, "viol_ece_k1": 0.01,
                       "u_err_spearman": 0.5, "u_q80": 0.3, "u_q50": 0.1})
    assert d.reliability > 0.8 and d.max_h == 2 and d.u_thr == 0.3


# ---------------------------------------------------------------------------- replay, hindsight, archive
def test_hindsight_views_never_rewrite_recorded_experience(model):
    eps = _eps(model, {"bizdev": 0.5, "negotiation": 0.5}, n=40, seed=3)
    rp = Replay(100_000, 1_000)
    rp.add(eps, persist=False)
    with_h = [e for e in eps if e.hindsight]
    if not with_h:
        pytest.skip("no partial achievements in this small sample")
    before = [(list(e.goal), e.success, len(e.steps)) for e in with_h]
    views = [v for v in rp.sample(200, random.Random(0), her_p=1.0) if v.relabeled]
    assert views
    for v in views:
        r = v.rewards(0.0)
        assert r[-1] >= 1.0 - 0.1 and v.dones()[-1] and not v.ep.success  # hindsight success is only a view
        assert v.goal != v.ep.goal
    assert [(list(e.goal), e.success, len(e.steps)) for e in with_h] == before
    assert all(e.origin == "real" for e in list(rp.main) + list(rp.prot))


def test_archive_starts_replay_history_as_burn_in_only(model):
    ar = Archive(50, seed=0)
    _eps(model, {"product": 1.0}, n=16, archive=ar, frac=0.0)
    assert ar.size().get("product", 0) > 0
    eps = _eps(model, {"product": 1.0}, n=16, archive=ar, frac=1.0, seed=4)
    arch = [e for e in eps if e.start == "archive"]
    assert arch
    for e in arch:
        assert e.n_prefix > 0 and all(not s.learn for s in e.steps[:e.n_prefix]) and all(s.learn for s in e.steps[e.n_prefix:])


def test_trainer_checkpoint_resume(tmp_path):
    from agentrs.cam.v2.train import Trainer2, TrainConfig2

    cfg = TrainConfig2(run_dir=str(tmp_path / "r"), model=TINY, workers=8, steps_per_iter=120, max_iters=1, updates_per_iter=2,
                       batch_episodes=4, learn_start_steps=50, eval_every=100, eval_episodes=2, test_episodes=2, cf_states=1,
                       audit_episodes=20, archive={"frac": 0.2, "cells": 20}, lagrangian={"kappa": 0.05},
                       dream={"mode": "fixed", "horizon": 2, "updates_per_iter": 1, "starts": 8, "batch_episodes": 4, "start_iter": 1})
    t = Trainer2(cfg)
    t.train_iter()
    w = {k: v.clone() for k, v in t.model.state_dict().items()}
    t2 = Trainer2(cfg)
    assert t2.state["iter"] == 1 and t2.state["restarts"] == 1 and len(t2.replay) == len(t.replay)
    assert all(torch.equal(w[k], v) for k, v in t2.model.state_dict().items())


def test_hindsight_shaping_is_potential_based_relative_to_the_achieved_level(model):
    from agentrs.cam.v2.data import View

    eps = _eps(model, {"product": 1.0}, n=24, seed=5)
    hs = [e for e in eps if e.hindsight]
    if not hs:
        pytest.skip("no solvent deadline episodes in this sample")
    e = hs[0]
    v0 = View(e, list(e.hindsight[0][0]), len(e.steps), relabeled=True, shaping=0.0)
    v1 = View(e, list(e.hindsight[0][0]), len(e.steps), relabeled=True, shaping=0.5)
    r0, r1 = v0.rewards(0.0), v1.rewards(0.0)
    # discounted shaping telescopes to -phi'(s0) (= 0 at the start): returns agree up to that constant
    D, J0, J1 = 1.0, 0.0, 0.0
    for s, a, b in zip(e.steps, r0, r1):
        J0 += D * a
        J1 += D * b
        D *= s.disc
    phi0 = min(1.0, e.steps[0].prog_before / max(1e-6, e.steps[-1].prog_after))
    assert abs((J1 - J0) - (-0.5 * phi0)) < 1e-6
