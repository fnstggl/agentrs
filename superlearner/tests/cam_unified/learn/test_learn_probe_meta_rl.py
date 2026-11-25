"""Diagnostic (a), informative-probe bandit: a small recurrent policy trained with the meta-RL machinery
(TrialRunner -> hindsight learning progress -> MetaRLObjective / ActorCriticObjective -> CombinedTrainer, ValueHead,
BudgetEncoder) learns to pay for the probe first, exploits its report, conditions on the BUDGET token (no probe when
one decision is left) and beats the greedy / one-step value-of-information baselines.

Recorded (200 iterations x 128 episodes, CPU, ~1 min; held-out seeds; deterministic evaluation):
  learned (LP-driven)  H=1..6: 14-22 / 70 / 170 / 270 / 370 / 470 cents  == Bayes-optimal for H >= 2 (8/8 seeds)
  greedy / KG / EIG    H=1..6: 16 / 49.5 / 96 / 167 / 249 / 349.5       (elimination, never probes)
  extrinsic-only PPO   converges to elimination in 9/9 runs (local optimum; see the slow comparison test)
"""
import json
import os

import pytest
import torch

from learn_doubles import ProbeAgent, evaluate, train_probe_agent
from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
from superlearner.cam_unified.learn.math_assist import DiscreteToolPolicy
from superlearner.cam_unified.learn.meta import TrialRunner

BAYES = {h: ProbeBanditAdapter(1).bayes_optimal_value(h) for h in range(1, 7)}


def _report(name, d):
    print(f"\n[{name}] " + json.dumps(d, default=float))
    out = os.environ.get("CAM_D_RESULTS")
    if out:
        with open(out, "a") as fh:
            fh.write(json.dumps({"test": name, **d}, default=float) + "\n")


def test_probe_bandit_meta_rl_learns_to_probe_first_and_beats_greedy():
    torch.set_num_threads(2)
    agent = ProbeAgent(6, D=64, seed=0)
    out = train_probe_agent(agent, iters=200, B=128, horizon=(1, 6), lp_coef=200.0, seed0=0)
    learned = evaluate(agent)
    greedy = evaluate(DiscreteToolPolicy("greedy", prior_mean={"pull": 100 / 6}), greedy=False)
    _report("probe_meta_rl", {"train_s": out["seconds"], "learned": learned, "greedy": greedy, "bayes": BAYES})
    for H in range(2, 7):
        assert learned[H]["probe_first"] >= 0.95, (H, learned[H])
        assert learned[H]["return"] >= BAYES[H] - 15.0, (H, learned[H], BAYES[H])
    assert learned[1]["probe_first"] <= 0.05            # budget-conditioned: no experiment without a future
    gain = sum(learned[H]["return"] - greedy[H]["return"] for H in range(2, 7)) / 5
    assert gain >= 50.0, (learned, greedy)
    # the trainer measured every objective's gradient into the shared recurrent backbone
    probes = [u.backbone for u in out["trainer"].history if u.backbone is not None]
    assert len(probes) >= 8 and all(set(p["norm"]) == {"actor", "value"} for p in probes)
    assert any(p["norm"]["value"] > 0 for p in probes) and all(p["norm"]["actor"] > 0 for p in probes)
    assert all(-1.0 <= p["cosine"]["actor|value"] <= 1.0 for p in probes)


@pytest.mark.slow
def test_exploration_baseline_comparison_on_the_probe_bandit():
    """Spec 8.5: random, greedy, Thompson, CAM 7.0 KG, EIG, learned (extrinsic only), learned (learning progress),
    learned + math assist (CAM 7.0 scores as policy inputs, declared condition)."""
    torch.set_num_threads(2)
    res = {}
    for kind in ("random", "greedy", "thompson", "kg", "eig"):
        res[kind] = evaluate(DiscreteToolPolicy(kind, prior_mean={"pull": 100 / 6}, seed=1), greedy=False)
    a_ext = ProbeAgent(6, D=64, seed=0)
    t_ext = train_probe_agent(a_ext, iters=200, lp_coef=0.0, seed0=0)
    res["learned_extrinsic"] = evaluate(a_ext)
    a_lp = ProbeAgent(6, D=64, seed=0)
    t_lp = train_probe_agent(a_lp, iters=200, lp_coef=200.0, seed0=0)
    res["learned_lp"] = evaluate(a_lp)
    a_ma = ProbeAgent(6, D=64, seed=0, math_assist=True)
    t_ma = train_probe_agent(a_ma, iters=200, lp_coef=200.0, seed0=0)
    res["learned_lp_math_assist"] = evaluate(a_ma)
    summary = {k: {"mean_return_H2_6": sum(v[H]["return"] for H in range(2, 7)) / 5,
                   "probe_first_H3_6": sum(v[H]["probe_first"] for H in range(3, 7)) / 4} for k, v in res.items()}
    summary["bayes_optimal"] = {"mean_return_H2_6": sum(BAYES[H] for H in range(2, 7)) / 5}
    _report("exploration_baselines", {"summary": summary, "per_horizon": res,
                                      "train_s": [t_ext["seconds"], t_lp["seconds"], t_ma["seconds"]]})
    best_programmed = max(summary[k]["mean_return_H2_6"] for k in ("random", "greedy", "thompson", "kg", "eig"))
    assert summary["learned_lp"]["mean_return_H2_6"] > best_programmed + 50
    assert summary["learned_lp"]["probe_first_H3_6"] > 0.95
    for k in ("random", "greedy", "thompson", "kg", "eig"):
        assert summary[k]["probe_first_H3_6"] < 0.3       # one-step VOI cannot value the information-only probe


@pytest.mark.slow
def test_j_meta_early_trial_exploration_pays_off_in_the_later_trial():
    """Two trials of the same hidden task (2 decisions each); only trial 2 counts (trial weights [0, 1]). The agent
    must spend trial 1 on the probe and exploit in trial 2: credit flows across the (non-terminal) trial boundary."""
    torch.set_num_threads(2)
    agent = ProbeAgent(6, D=64, seed=0)
    out = train_probe_agent(agent, iters=200, B=128, horizon=2, n_trials=2, trial_weights=[0.0, 1.0],
                            lp_coef=200.0, seed0=0)
    ad = ProbeBanditAdapter(400, horizon=2)
    r = TrialRunner(ad, ad.evaluator).run(agent, list(range(5_000_000, 5_000_400)), n_trials=2, greedy=True)
    per_trial = r.per_trial_return.mean(0).tolist()
    p1 = sum(a[:2].count("probe") for a in r.actions) / 400
    p2 = sum(a[2:].count("probe") for a in r.actions) / 400
    _report("j_meta_two_trials", {"per_trial_return": per_trial, "probes_trial1": p1, "probes_trial2": p2,
                                  "train_s": out["seconds"]})
    assert per_trial[1] >= 180.0 and p1 >= 0.9 and p2 <= 0.1


@pytest.mark.slow
def test_lp_trained_agent_ignores_a_noisy_tv():
    torch.set_num_threads(2)
    agent = ProbeAgent(6, D=64, seed=0, noise_tv=True)
    out = train_probe_agent(agent, iters=200, B=128, horizon=(1, 6), lp_coef=200.0, seed0=0, noise_tv=True)
    ev = evaluate(agent, noise_tv=True)
    watch = []
    for H in (3, 6):
        ad = ProbeBanditAdapter(200, horizon=H, noise_tv=True)
        rr = TrialRunner(ad, ad.evaluator).run(agent, list(range(7_000_000, 7_000_200)), greedy=True)
        watch.append(sum(a.count("watch") for a in rr.actions) / 200)
    _report("noisy_tv", {"eval": ev, "watch_per_episode": watch, "train_s": out["seconds"]})
    assert max(watch) < 0.05 and ev[6]["probe_first"] > 0.9
