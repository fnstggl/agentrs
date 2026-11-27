# Change requests of the learning-mechanism engineer (LRN; owner of `predict/`, `reward/`)

Context: change request S-5 (`docs/change_requests/S.md`, `docs/S/DIAGNOSIS.md`). The mechanism fixes are in LRN's
own files; the agent needs only configuration-level wiring. Every request keeps public APIs source-compatible.

## LRN-1 (agent, `agent.py`, owner INT2): make `thompson_policy` the default selection rule

**What.** `AgentSettings.selection_rule = "thompson_policy"` (it is in `predict.policy.RULES`, so
`AgentSettings.from_cfg` already accepts it; `agent.selection_rule: thompson_policy` in a configuration works today
without any code change). `_with_value_prior` stays as it is: it only applies under rule `policy`, which remains
selectable (and is the ablation "policy head alone").

**Why.** The rule is linear Thompson sampling over the world model's joint posterior samples of each candidate's
goal value (standardised within the decision) and the trainable policy's bounded preference, the two weighted by a
Bayesian calibration on verified outcomes (`predict.policy.SelectionCalibration`, fed by `MetaLearner` from the
agent's signed per-decision `reward_signals.verified_progress_delta`). Exact propensities, epsilon floor, every
input recorded in `decision.diagnostics["selection"]`. The fixed `value_weight` prior of rule `policy` (S-5
workaround) gives the world model a constant, unvalidated weight and no uncertainty; this rule explores by posterior
probability of being best and lets the policy's weight grow only as its preferences predict verified progress.
Measured: `docs/evidence/B/choice_diagnostic_*.json` (see `docs/status/B.json`).

**Diff.**
```diff
-    selection_rule: str = "policy"           # predict.policy.RULES except "randomized"
+    selection_rule: str = "thompson_policy"  # predict.policy.RULES except "randomized"
```

**Workaround meanwhile.** Evaluation and diagnostics pass `agent.selection_rule=thompson_policy`.

## LRN-2 (agent, owner INT2): nothing else to wire — for review only

* The calibration object is shared through the policy head (`predict.policy.selection_calibration(policy)`, the
  same pattern as `WorldHeads._real_scaler`); `DecisionMaker` and `MetaLearner` find the same instance whatever
  their construction order. It is persisted in `MetaLearner.state_dict()["selection"]`, which the agent already
  checkpoints (`aux_state["agent"]["meta"]`). Older checkpoints load with an empty calibration.
* `MetaLearner.policy_update` now sizes the live step by a trust region (`reward.meta.target_kl`, default 0.05 nats;
  INT2-7) and replays up to `reward.meta.replay_matured` (16) earlier decisions whose verified progress is final
  (INT2-1 consequence: annotations arrive cycles after the decision). The step chosen is recorded in the update
  event (`inner_lr`) and in `LearningUpdateRecord.notes` / `.diagnostics["step_search"]`; the meta-gradient
  differentiates exactly that update. The agent's `policy_update_every` / `policy_update_batch` are unchanged.
* The bootstrap mix now moves (`reward.meta.mixer_step` 0.25, `mixer_min_independent` 4, non-inferiority margin
  `mixer_rel_margin` 0.05 x the RMS of the held-out verified utilities; reverts on shown harm, holds while
  inconclusive). `status()["bootstrap_mix"]` reports it; every decision is in `MetaUpdateRecord.diagnostics`
  (`promotion*`, `mix_after_promotion`) and `meta.mixer.decisions`.

## LRN-3 (agent, owner INT2): record the two bootstrap inputs that are still never written

**What.** `reward_signals` annotations for `epistemic_after` (the chosen candidate's goal-value epistemic std after
the fast-layer update with the decision's own observed outcomes, e.g. `Predictor` re-evaluating the chosen action
after `update_beliefs`) and `learning_progress` (`reward.bootstrap.LearningProgressTracker`: world-model loss on the
decision's matured outcomes before minus after the next frequent update).

**Why.** Spec 7.6 lists self-supervised prediction and uncertainty-aware exploration as bootstrap signals; without
these two inputs both bootstrap terms are masked (0) in every runtime run (measured: all `learning_progress` /
`exploration` components 0.0 in `docs/evidence/validation/diagnosis/debug_f1_20261127/`). Uncertainty-aware
exploration is carried meanwhile by the selection rule (Thompson sampling over the epistemic posterior), and the
bootstrap's grounded part by delta V (now V(b') - V(b)) and the verified term. Not blocking.

## LRN-4 (agent, owner INT2): no level query where outcomes are attributed — already done (dcb9592), noted

The world model's `q_goal` of a candidate sums its goal-relevant predictions of the longest horizon. A candidate
whose only goal-relevant question is the objective *level* (e.g. `lab.adopters.total` for a wait) gets a predicted
level, which is not comparable with another candidate's predicted *attributed increment*. INT2's dcb9592 removes the
level query once the environment is known to attribute outcomes; before that (the first decisions of the first
objective) the mismatch remains. The calibration standardises within each decision and learns from verified
outcomes, which limits but does not remove the effect. A query flag for "level vs increment" in the contract would
remove it (owner M, not requested now).
