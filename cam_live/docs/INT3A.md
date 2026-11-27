# INT3A — final decision-side integration fixes (2026-11-27)

Owner of `agent.py`, `predict/`, `reward/`, `eval/proposal_probe.py` for this round. Every number below is from a
file under `docs/evidence/INT3A/` or a test named here; "before" is the code at b663d5e (start of this round) unless
stated otherwise. Real configuration = Qwen3-4B (frozen, bf16) + LoRA + latent bridge + 50m core, under the shared GPU
lock. Change requests: `docs/change_requests/INT3A.md`.

<!-- SECTION S6B -->

## 3. LRN-3: the bootstrap's learning-progress and exploration inputs are written

Before, `epistemic_after` and `learning_progress` were never produced, so the bootstrap terms `learning_progress`
and `exploration` were 0 for every decision of every runtime run. Now both are decision annotations, written once as
their own `decision.diagnostics` key `reward_signals.<name>` (the store permits adding a key, never changing one; the
`reward_signals` dict itself is written once by the verified-progress annotation and cannot be extended later):

* `learning_progress` (`CamLiveAgent._lp_before` / `_lp_after`): for up to 8 newest executed records awaiting it,
  the world model's mean NLL of the record's matured outcomes (`predict.providers.record_world_nll`, eval mode, no
  scaler / normaliser update) right before `ContinualLearner.learn`, and on the same record objects (same targets)
  right after it when a frequent update happened; `before - after` (`reward.bootstrap.LearningProgressTracker`).
* `epistemic_after` (`Predictor.goal_epistemic_after`, `CamLiveAgent._annotate_epistemic_after`): the Predictor keeps,
  per decision, the decision-time head view of the executed candidate's goal-relevant predictions; once the fast layer
  has absorbed the decision's own outcome for *every* one of them (`update_beliefs`), the goal-value epistemic std is
  recomputed on that view with the current fast state — the same quantity as the decision-time
  `CandidateEvaluation.uncertainty` (`epistemic_before`). No network pass, no new prediction is registered.
* `reward.assemble.stored_signals` reads the `reward_signals` dict and every `reward_signals.<name>` key (the dict wins
  where both name a signal); the inner update's bootstrap uses them.

Test `tests/integration/test_int3a_bootstrap_inputs.py` (LabWorld, debug scale, 14 decisions + one maturation cycle):
learning progress annotated for at least half of the executed records (the value equals the independently measured
NLL change on the same targets), `epistemic_after` for the decisions whose goal-relevant outcomes all matured, and the
inner update's `boot_components["learning_progress"]` and `["exploration"]` are non-zero. Fails on b663d5e (the
annotation helpers do not exist; every component is 0). Measured property worth stating: `epistemic_after` can exceed
`epistemic_before` (a count's epistemic variance scales with the fast layer's mean multiplier, which moves with the
observation); the bootstrap rewards only a drop.

## 4. INT2-9: a parsed program has the program tool's modality

`predict/candidates.py::_build` gave a composition its first step's modality; the authorization rule T-PROGRAM admits
`program` only as `internal` (the harness declares the `program` tool internal), so `program [http.get, ...]` was
refused before any step was judged. Now the composition takes the modality the environment declares for its
`program` tool (internal when none is described); every step keeps, and is authorised by, its own modality, and the
steps are bounded by the tools given to the parser (the agent gives the task's permitted tools; `_feasible` still
drops a program with any step outside the task's interfaces). The agent's workaround in `_feasible` is kept (a no-op
for parser-built programs; behaviour identical). Tests `tests/predict/test_int3a_program_modality.py` (4; fail on the
old parser) and the changed expectation in `tests/predict/test_candidates.py::test_multi_step_program`. The INT2 test
that pinned the defect needs a one-line update (INT3A-1).

## 5. LRN-6 reconciled

Confirmed in the code (INT2, 07be9ba): `CamLiveAgent.promote` keeps R_psi's whole module state and the meta optimizer
across a rollback, and the policy head unless `decision_agreement` is a reason of the refusal
(`promotions[-1]["kept"]`). INT2's own test did not change psi or the meta optimizer after the promoted checkpoint,
so it would pass with the old rollback too. `tests/integration/test_int3a_lrn6.py` changes psi and the Adam state of
the meta optimizer (a real optimizer step), and the policy head (by 1e-6, decision agreement unchanged) after the
promotion, then damages the world model: the refused promotion rolls the world model back and keeps psi, the meta
optimizer state and the policy head bit-exactly (`kept == ["reward", "policy_head"]`); with a strongly damaged policy
the policy head goes back when decision agreement is among the reasons, psi still stays. On 78ad939 (before LRN-6)
the test fails at the psi assertion. Status: done (`docs/change_requests/INT3A.md`).

## A. KF-E01: stored credit now trains (spec 9.2)

`ExperienceRecord.credit` (simple / RUDDER / COCOA, written by `credit.providers.CreditAssigner`) was read by no
loss. The eligible update is the policy-improvement update (`reward/inner.py`, the live policy step and the
meta-differentiated one): its goal term now takes the stored credit of the evidence-selected estimator
(`CreditAssigner.default`, chosen by `credit.select` on held-out randomised experiments; `simple` until RUDDER or
COCOA agree with them) through `PolicyImprovementLoss.credit_fn`, which the agent sets (`CamLiveAgent._credit_for`):

* **return-like credit** — RUDDER's learning signal (the redistributed return-to-go, architecture section 5: "the
  redistributed return-to-go is used for learning") and COCOA's credit of the action for its own and later outcomes:
  it *replaces* the row's discounted goal return (`InnerBatch.g_credit`, recursion `G_i = c_g C_i / rms(C)` at a
  credited row, `g_i + gamma_i G_{i+1}` elsewhere), so redistributed credit is never added on top of the verified
  progress it redistributes;
* **per-decision credit** — `simple` (the decision's own evidence-linked outcome amounts): a goal reward of the row,
  used when no final verified annotation exists and only where the environment attributes outcomes to decisions. In
  other environments the linked answer of a decision's goal question is a *reading of the objective level* (e.g. the
  pass fraction after the action), not a contribution of that decision; crediting it would pay every action after a
  success (the defect INT2's round 2 removed for the verified term).

Rows carry the source (`goal_source = "credit:<estimator>"`), the batch snapshot includes the credited values (the
differentiated update is the update applied; checkpointed with the batch), and without stored credit the update is
bit-identical to before. Test `tests/integration/test_int3a_audit.py::test_stored_credit_changes_the_policy_update_
target_and_gradient` (a change of the stored simple credit, and of RUDDER credit once that estimator is selected,
changes the returns and the inner gradient) and `::test_without_credit_the_update_is_unchanged`; both fail on
b663d5e. Not shown: that credit makes the policy better (RUDDER / COCOA have not been promoted by the selector in any
run; at the default, credit enters only rows without a final verified annotation in attributing environments).

## B. KF-E02: the reward signal `info_gain` is produced

The assembler read `CandidateEvaluation.extras["info_gain"]`, which nothing writes; the predictor writes the expected
fast-posterior KL of the candidate's own outcomes as `info_gain_nats` (`Predictor._voi`). The assembler now reads
`info_gain` and falls back to `info_gain_nats` (signal description updated: expected belief change, estimated before
acting). Test `test_int3a_audit.py::test_info_gain_reaches_the_reward_signals` (masked for every decision before).

## C. KF-E05: R_psi's lifetime context is no longer reset at an objective change

The architecture (section 4) defines `life_ctx` as a recurrent summary of the lifetime "not reset at objective
boundaries", and `LifetimeContext.new_lifetime` says the same; `CamLiveAgent.set_objective` called
`MetaLearner.new_lifetime()`, which zeroed it. Now `set_objective` calls the new `MetaLearner.new_objective()`: the
objective is a new segment of the meta-learner's bookkeeping (returns, the return-to-go chain and the discount do not
run across objectives; `UpdateEvent.lifetime` names the segment) and R_psi's context carries on — the first step of
the new objective sees the summary of everything before it (`life_states` of that record equals the state at the
switch). `new_lifetime()` is kept for diagnostics whose lifetimes are separate lives. Test
`test_int3a_audit.py::test_the_lifetime_context_of_r_psi_is_not_reset_at_an_objective_change` (zeroed before).
`docs/MODEL_ARCHITECTURE.md` section 10.2 (not INT3A's file) still describes the old behaviour.

<!-- SECTION COST -->

## 6. The bootstrap mix stays at 1 on the real model: can the promotion test become conclusive?

How the test works (`reward/bootstrap.RewardMixer.update`, `reward/metalearner.MetaLearner._promotion`): after an
*applied* meta-update, every held-out independent group (25 % of groups, hash-reserved) among that update's query
rows contributes one number — the clipped importance-weighted verified utility of the policy updated with the
learned reward minus that of the policy updated with the bootstrap reward, from the same snapshot on the same support.
Once 4 numbers are buffered the mixer decides by a one-sided 95 % Student-t bound: lower the mix by 0.25 if the lower
bound is >= -0.05 x RMS(held-out utilities) (non-inferiority), raise it if the upper bound is < 0, else hold; the
buffer is then cleared, whatever the decision.

Checked for a defect: the unit is right (one number per independent group, rows of a group averaged; the group of a
row is its objective evaluation interval or, without a signed verdict, the decision), the held-out rows of every
episode of an applied update reach the buffer (traces exist for all of them), and the buffer is checkpointed
(`state_dict["promo"]`). No defect found that keeps data from the test; nothing was changed.

What it needs (measured spread): the one promotion decision of the real-model pathway run
(`docs/evidence/C/real_pathway_20261127T200546Z.json`) had 5 groups, mean difference 8.57, bounds [-15.0, 32.2],
margin 1.06 (so RMS(U) = 21.1); the implied standard deviation of the per-group difference is 24.8 = 1.17 RMS(U).
With that spread the probability that one decision on n groups lowers the mix (`docs/evidence/INT3A/
promotion_power.json`, non-central t):

| n groups per decision | true effect 0 | 0.1 RMS | 0.4 RMS | 1.0 RMS |
|---|---|---|---|---|
| 4 (what each decision gets now) | 0.057 | 0.075 | 0.150 | 0.401 |
| 20 | 0.072 | 0.137 | 0.505 | 0.987 |
| 80 | 0.103 | 0.306 | 0.961 | 1.000 |
| 400 | 0.214 | 0.820 | 1.000 | 1.000 |

Groups needed for 80 % power: about 3,400 (no effect: the learned reward exactly as good as the bootstrap), 380 (effect 0.1
RMS), 140 (0.2 RMS), 44 (0.4 RMS). Evidence available: the planning phase (120 decisions in one objective) and an
evaluation seed (20 decisions x 4-8 objectives = 80-160 decisions) give at most 30 and 20-40 held-out groups even if
every decision were its own group and every held-out row fell into an applied meta-update's episodes (in the real
pathway run 1 of 63 meta-update calls was applied, with 5 held-out groups). And each decision sees only the ~4-6
groups buffered since the previous one, because the buffer is cleared after a hold.

So: at realistic evidence rates the test is conclusive only for a learned reward that is far better than the
bootstrap (an effect of the order of the utility's RMS); a learned reward that is about as good stays at mix 1 by
design (inconclusive -> hold), which is what every real-model run showed. Pooling held-out evidence across decisions
(not clearing on hold) would raise the power at n = 20 from 0.15 to 0.50 for a 0.4 RMS effect, but repeated looks at a
growing buffer need a sequential bound (alpha spending or a confidence sequence) to keep the 5 % error; that is a
design change of the promotion rule, not a defect fix, and was not made. The mix was not forced.
