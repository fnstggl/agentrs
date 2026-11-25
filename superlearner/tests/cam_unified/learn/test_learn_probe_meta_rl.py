"""Diagnostic (a), informative-probe bandit: a small recurrent policy trained with the meta-RL machinery
(TrialRunner -> hindsight learning progress -> MetaRLObjective (key "meta", stage explore_meta, LP gated by
cfg.use_learned_exploration and the stage's lp_reward flag, scaled by cfg.lp_coef) -> CombinedTrainer, ValueHead,
BudgetEncoder) learns to pay for the probe first, exploits its report, conditions on the BUDGET token (no probe when
one decision is left) and beats the greedy / independent-candidate one-step value-of-information baselines.

Seed sweeps behind the numbers quoted in the claims are reproducible with
``superlearner/tests/cam_unified/learn/sweeps/probe_sweep.py``; its committed output is
``superlearner/tests/cam_unified/learn/sweeps/probe_sweep_results.jsonl`` (one JSON row per training run).
"""
import json
import os

import pytest
import torch

from learn_doubles import ProbeAgent, ReactiveArmBelief, competence_curve, evaluate, train_probe_agent
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
    greedy_pol = DiscreteToolPolicy("greedy", prior_mean={"pull": 100 / 6})
    greedy = evaluate(greedy_pol, greedy=False)
    comp = {"learned": competence_curve(agent), "greedy": competence_curve(greedy_pol, greedy=False)}
    _report("probe_meta_rl", {"train_s": out["seconds"], "learned": learned, "greedy": greedy, "bayes": BAYES,
                              "competence_after_k": comp})
    assert all(h["lp_in_reward"] == 1 for h in out["hist"])          # stage explore_meta: the LP term is on
    for H in range(2, 7):
        assert learned[H]["probe_first"] >= 0.95, (H, learned[H])
        assert learned[H]["return"] >= BAYES[H] - 15.0, (H, learned[H], BAYES[H])
    assert learned[1]["probe_first"] <= 0.05            # budget-conditioned: no experiment without a future
    gain = sum(learned[H]["return"] - greedy[H]["return"] for H in range(2, 7)) / 5
    assert gain >= 50.0, (learned, greedy)
    # competence after k support decisions (policy level): one paid experiment makes the learned agent competent
    assert comp["learned"][0] == pytest.approx(-30.0) and comp["learned"][1] >= 95.0
    assert comp["greedy"][1] < 40.0
    # the trainer measured every objective's gradient into the shared recurrent backbone
    probes = [u.backbone for u in out["trainer"].history if u.backbone is not None]
    assert len(probes) >= 8 and all(set(p["norm"]) == {"meta", "value"} for p in probes)
    assert any(p["norm"]["value"] > 0 for p in probes) and all(p["norm"]["meta"] > 0 for p in probes)
    assert all(-1.0 <= p["cosine"]["meta|value"] <= 1.0 for p in probes)


@pytest.mark.slow
def test_exploration_baseline_comparison_on_the_probe_bandit():
    """Spec 8.5: random, greedy, Thompson, CAM 7.0 KG, EIG (independent-candidate model), the structured-prior
    baselines (correlated hypotheses: greedy / Thompson / KG), learned (extrinsic only), learned (learning
    progress), learned + math assist (CAM 7.0 scores as policy inputs, declared condition)."""
    torch.set_num_threads(2)
    res = {}
    for kind in ("random", "greedy", "thompson", "kg", "eig"):
        res[kind] = evaluate(DiscreteToolPolicy(kind, prior_mean={"pull": 100 / 6}, seed=1), greedy=False)
    sp = ProbeBanditAdapter(1).structured_prior()
    for kind in ("greedy_structured", "thompson_structured", "kg_structured"):
        res[kind] = evaluate(DiscreteToolPolicy(kind, seed=1, structured=sp), greedy=False)
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
    independent = ("random", "greedy", "thompson", "kg", "eig")
    best_independent = max(summary[k]["mean_return_H2_6"] for k in independent)
    assert summary["learned_lp"]["mean_return_H2_6"] > best_independent + 50
    assert summary["learned_lp"]["probe_first_H3_6"] > 0.95
    for k in independent:
        assert summary[k]["probe_first_H3_6"] < 0.3       # independent-candidate VOI cannot value the probe
    # the structured-prior KG (given the hand-specified hypothesis model) values the probe and is Bayes-optimal
    # here; the learned agent (no hypothesis model, no KG input) matches it
    assert summary["kg_structured"]["probe_first_H3_6"] > 0.95
    assert summary["kg_structured"]["mean_return_H2_6"] == pytest.approx(summary["bayes_optimal"]["mean_return_H2_6"])
    assert summary["learned_lp"]["mean_return_H2_6"] >= summary["kg_structured"]["mean_return_H2_6"] - 15


@pytest.mark.slow
def test_j_meta_early_trial_exploration_pays_off_in_the_later_trial():
    """Two trials of the same hidden task; only trial 2 counts (trial weights [0, 1]). With 2 decisions per trial
    the agent must spend trial 1 on the probe and exploit in trial 2; with ONE decision per trial the probe's report
    exists only because the harness delivers the last outcome of trial 1 at the start of trial 2."""
    torch.set_num_threads(2)
    out = {}
    for H in (2, 1):
        agent = ProbeAgent(6, D=64, seed=0)
        tr = train_probe_agent(agent, iters=200, B=128, horizon=H, n_trials=2, trial_weights=[0.0, 1.0],
                               lp_coef=200.0, seed0=0)
        ad = ProbeBanditAdapter(400, horizon=H)
        r = TrialRunner(ad, ad.evaluator).run(agent, list(range(5_000_000, 5_000_400)), n_trials=2, greedy=True)
        out[H] = {"per_trial_return": r.per_trial_return.mean(0).tolist(),
                  "probes_trial1": sum(a[:H].count("probe") for a in r.actions) / 400,
                  "probes_trial2": sum(a[H:].count("probe") for a in r.actions) / 400, "train_s": tr["seconds"]}
    _report("j_meta_two_trials", out)
    assert out[2]["per_trial_return"][1] >= 180.0 and out[2]["probes_trial1"] >= 0.9 and out[2]["probes_trial2"] <= 0.1
    assert out[1]["per_trial_return"][1] >= 90.0 and out[1]["probes_trial1"] >= 0.9 and out[1]["probes_trial2"] <= 0.1


@pytest.mark.slow
def test_noise_correction_is_what_keeps_the_agent_away_from_a_noisy_tv():
    """The LP belief reacts to the TV (a miscalibrated filter whose uncertainty expands on surprising readings), so
    the raw LP of watching is positive. Trained on raw LP the agent learns to watch; with the epistemic signal
    fraction (0 for a reading every hypothesis predicts alike) it ignores the TV and stays Bayes-optimal."""
    torch.set_num_threads(2)
    out = {}
    for corrected in (True, False):
        agent = ProbeAgent(6, D=64, seed=0, noise_tv=True)
        tr = train_probe_agent(agent, iters=200, B=128, horizon=(1, 6), lp_coef=200.0, seed0=0, noise_tv=True,
                               belief=ReactiveArmBelief(6), corrected=corrected)
        ev = evaluate(agent, noise_tv=True)
        watch = {}
        for H in (3, 6):
            ad = ProbeBanditAdapter(200, horizon=H, noise_tv=True)
            rr = TrialRunner(ad, ad.evaluator).run(agent, list(range(7_000_000, 7_000_200)), greedy=True)
            watch[H] = sum(a.count("watch") for a in rr.actions) / 200
        out["corrected" if corrected else "raw_lp"] = {"eval": ev, "watch_per_episode": watch,
                                                       "train_s": tr["seconds"]}
    _report("noisy_tv_reactive_belief", out)
    c, raw = out["corrected"], out["raw_lp"]
    assert max(c["watch_per_episode"].values()) < 0.05 and c["eval"][6]["probe_first"] > 0.9
    assert c["eval"][6]["return"] >= BAYES[6] - 15
    assert raw["watch_per_episode"][6] > 0.5                        # the failure the correction prevents
    assert raw["eval"][6]["return"] < c["eval"][6]["return"] - 50
