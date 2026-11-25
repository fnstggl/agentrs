"""Diagnostic (a), informative-probe bandit: a small recurrent policy trained with the meta-RL machinery
(TrialRunner -> hindsight learning progress -> MetaRLObjective (key "meta", stage explore_meta, LP gated by
cfg.use_learned_exploration and the stage's lp_reward flag, scaled by cfg.lp_coef) -> CombinedTrainer, ValueHead,
BudgetEncoder) learns to pay for the probe first, exploits its report, conditions on the BUDGET token (no probe when
one decision is left) and beats the greedy / independent-candidate one-step value-of-information baselines.

Recorded (200 iterations x 128 episodes per run, CPU, ~1 min each; held-out seeds; deterministic evaluation):
  learned (LP, agent seed 0)  H=1..6: 14 / 70 / 170 / 270 / 370 / 470 cents == Bayes-optimal for H >= 2
  seed sweep, LP              6/8 runs Bayes-optimal for every H in 2..6 (mean return H2-6 267.5 of 270); the other two
                              miss one horizon (seed 2 does not probe at H=2: 49 vs 70; seed 5: 390 vs 470 at H=6)
  seed sweep, extrinsic only  0/8 runs ever probe (elimination; mean return H2-6 170.3): a local optimum
  greedy / KG / EIG           H=1..6 about 12-16 / 45-50 / 96 / 160-167 / 249-255 / 343-350 (never probe)
  structured-prior KG         Bayes-optimal (it is given the hypothesis model the learned agent never sees)

The seed sweep is reproducible with ``superlearner/tests/cam_unified/learn/sweeps/probe_sweep.py``; its committed
output is ``sweeps/probe_sweep_results.jsonl`` (one JSON row per training run). The outputs of the slow tests of this
module (and two exploratory runs) are in ``sweeps/slow_test_results.jsonl``.
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
def test_lp_trained_agent_ignores_a_noisy_tv_that_its_belief_reacts_to():
    """The LP belief reacts to the TV (`ReactiveArmBelief`: miscalibrated prior, uncertainty expansion on every
    reading), so the raw hindsight LP of watching is positive; the LP reward the agent trains on relies on the
    epistemic signal fraction (exactly 0 for a reading every hypothesis predicts alike). The agent ignores the TV and
    stays Bayes-optimal. (The single-decision contrast — a raw-LP experimenter learns to watch — is the default test
    `test_learn_explore::test_with_a_belief_that_reacts_to_noise_only_the_epistemic_correction_rejects_the_noisy_tv`;
    the recurrent agent trained on raw LP also converged to probe-first without watching at bias 3 and 6, because
    the TV only pays off in raw LP *before* the probe and PPO commits to probe-first first.)"""
    from learn_doubles import arm_learning_progress
    torch.set_num_threads(2)
    belief = ReactiveArmBelief(6)
    # the training signal: watching earns raw LP but no corrected LP (random initial policy, 512 episodes)
    probe_agent = ProbeAgent(6, D=64, seed=0, noise_tv=True)
    ad = ProbeBanditAdapter(512, horizon=6, noise_tv=True)
    res = TrialRunner(ad, ad.evaluator).run(probe_agent, list(range(512)))
    raw = arm_learning_progress(res, belief, 8, 1.0, corrected=False)
    cor = arm_learning_progress(res, belief, 8, 1.0, corrected=True)
    w = raw["watched"] & res.traj.mask
    w0 = raw["watched"][:, 0]                       # watching first, while the miscalibrated prior is untouched
    raw_watch, cor_watch = float(raw["intrinsic"][:, 0][w0].mean()), float(cor["intrinsic"][w].abs().max())
    agent = ProbeAgent(6, D=64, seed=0, noise_tv=True)
    tr = train_probe_agent(agent, iters=200, B=128, horizon=(1, 6), lp_coef=200.0, seed0=0, noise_tv=True,
                           belief=belief, corrected=True)
    ev = evaluate(agent, noise_tv=True)
    watch = {}
    for H in (3, 6):
        ad = ProbeBanditAdapter(200, horizon=H, noise_tv=True)
        rr = TrialRunner(ad, ad.evaluator).run(agent, list(range(7_000_000, 7_000_200)), greedy=True)
        watch[H] = sum(a.count("watch") for a in rr.actions) / 200
    _report("noisy_tv_reactive_belief", {"raw_lp_of_watching_nats": raw_watch, "corrected_lp_of_watching": cor_watch,
                                         "eval": ev, "watch_per_episode": watch, "train_s": tr["seconds"]})
    assert raw_watch > 0.5 and cor_watch == 0.0
    assert max(watch.values()) < 0.05 and ev[6]["probe_first"] > 0.9
    assert all(ev[H]["return"] >= BAYES[H] - 15 for H in range(2, 7))
