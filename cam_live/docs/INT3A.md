# INT3A — final decision-side integration fixes (2026-11-27)

Owner of `agent.py`, `predict/`, `reward/`, `eval/proposal_probe.py` for this round. Every number below is from a
file under `docs/evidence/INT3A/` or a test named here; "before" is the code at b663d5e (start of this round) unless
stated otherwise. Real configuration = Qwen3-4B (frozen, bf16) + LoRA + latent bridge + 50m core, under the shared GPU
lock. Change requests: `docs/change_requests/INT3A.md`.

## 1. S-6b: the backbone now attempts the software objective

**Measured before** (`python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4`, code of b663d5e, real
configuration; `docs/evidence/INT3A/probe_before_*.json`; every prompt sent is in the files):

| run | prompt: tools listed / tokens | rounds with a valid write of the named file (SW / MECH) | rounds running code | validity (valid / written) SW, MECH | what it wrote |
|---|---|---|---|---|---|
| b663d5e, run root below a 0700 directory (as S-6 ran) | 5 of 14 (SW), 7 of 16 (MECH) / 907, 1102 | 0 / 4, 0 / 4 | 0 | 16/16, 3/15 | notes restating the task, `program`s of `internal.inspect`, and in every SW round `internal.request_operator` for "the capability to write solution.py"; byte-identical to S-6's probe (same seeds) |
| b663d5e, traversable root | 12 of 14, 12 of 16 / 1269-1459 | 0 / 4, 2 / 4 | 0 | 1/15, 7/16 | full `solve` programs in `sandbox.write_file` / `sandbox.python` — refused: `path*`, `content*`, `code*`, `what*` as argument names |
| this round (snapshot of 8cebdb0's agent), traversable root | 14 of 14, 16 of 16 / 1690, 1932 | **4 / 4, 4 / 4** | 0 | **16/16, 16/16** | SW: `sandbox.write_file solution.py` with a complete `solve` every round (each of the 4 files passes 60 / 60 of the verifier's hidden tests when scored offline; the agent did not execute them in the probe), `sandbox.list_files`, `internal.inspect`, an operator request asking to *confirm* the solution; MECH: `http.get` of a measurement, a placeholder `model.py` (`return 0.0`), `sandbox.list_files` |

Why the model "believed it could not write files" — three integration causes, found in the prompt text and the raw
outputs:

1. **The sandbox tools were unavailable** in S-6's runs. The sandbox runs code as uid 996 and is available only if that
   uid can traverse to its workspace; S-6 ran from a frozen copy under the session scratch directory (`/tmp/claude-0`,
   mode 0700), so every `sandbox.*` tool was `available=False` ("workspace ... is below a directory the sandbox uid 996
   cannot traverse") and the agent (correctly) listed only the 5 internal tools. The model's request for the
   capability to write `solution.py` was the true state of that computer. Not an agent defect; nothing reported it.
   Now every decision records the permitted tools the environment reports unavailable, with the reason, and the probe
   stops unless `--allow-unavailable`; change request INT3A-2 asks the open-ended runner to refuse such a run.
2. **The required-argument marker was copied into argument names.** The compact tool form wrote required arguments as
   `path*`; the backbone wrote `{"path*": "solution.py", "content*": "def solve..."}` and the normaliser stripped the
   unknown names (the tool schema has `additionalProperties: false`), leaving `arguments: {}` — every file write and
   program run refused as "missing required property". Now names are listed exactly with `required` in their
   description, and a trailing `*` on a name the schema has is undone (logged `unmarked:<tool>.<name>`).
3. **Nested objects were shown as "object"**: the backbone invented `limits: {wall, cpu, memory}` for
   `{cpu_s, mem_mb, ...}` and the action was refused. Now one level of properties is shown.

Also changed (brief items a-c): the prompt lists the complete authorized action space every decision (every
available tool whose modality the task permits, internal tools always; the policy's top-ranked operations first,
then the environment's order; no random rotation; `proposer_tools = 0` by default, a positive value is an ablation);
descriptions are shortened to whole sentences (200 characters); the objective is verbatim, the task's success
definition is the first line of the digest (`[task] success definition: {...}`), and every executed action adds a
`[result] <operation> -> <status> (<failure class>): <detail>` line; the prompt is fitted to the backbone's window
(`nn.max_prompt_tokens`, 2048: the backbone keeps a longer prompt's *end*) by shortening the digest and then the tool
descriptions, never the tool list (INT3A-4: INT3B measured 4112 tokens in the planning task); a raw line break inside
a JSON string (multi-line program text) is joined back exactly. Every decision records the listed tools in prompt
order, the prompt's sha256, characters, tokens, the fit and the unavailable permitted tools
(`decision.diagnostics["generation"]`). Prompt size grew from 907 / 1102 to 1690 / 1932 tokens (SW / MECH), still
inside the 2048 window. Tests: `tests/integration/test_int3a_prompt.py` (8; the first two fail on b663d5e).

What is not shown by the probe: it calls `propose` without executing anything, on the objective's first context.
Whether the agent *chooses* and executes the write is a property of the selection rule on the real model — measured in
the runtime runs of section 2 (SW:rle, 4 decisions).

## 2. Decision cost on the real configuration

`docs/evidence/INT3A/profile_decision.py`: the agent built as the open-ended runner builds it (condition E,
substantive settings), driven by `LiveRuntime` with the runner's `RuntimeConfig`; every agent method, the proposer,
`model.generate` / `encode_*` / `backbone.embed`, the predictor (`evaluate_candidates`, `_voi`, `observe_outcomes`,
`belief_value`), decider, planner, learner, meta-learner, providers' losses, environment, verifier and runtime methods
are wrapped by timers (inclusive wall time, CUDA synchronised before and after). Machine shared (load 24-30, other GPU
processes resident). Mean seconds per runtime step that decided and acted:

| component (inclusive) | SW:rle, this round, old VOI (2 acted steps) | MECH:7, before (7) | MECH:7, after (8) |
|---|---|---|---|
| whole runtime step | 16.7 | 18.5 | 17.1 |
| propose (of which `model.generate`) | 12.5 (12.2) | 11.4 (11.2) | 10.8 (10.5) |
| evaluate (of which VOI) | 2.1 (1.5) | 2.1 (1.7) | 2.0 (1.5) |
| `backbone.embed` (candidate and context texts) | 1.1 | 2.1 | 2.1 |
| learn (of which `ContinualLearner.learn`) | 0.9 (0.8) | 3.9 (3.4) | 3.3 (2.7) |
| internal rewards / update beliefs / save | 0.3 / 0.2 / 0.5 | 0.3 / 0.3 / 0.2 | 0.3 / 0.2 / 0.2 |
| learning-progress measurement (LRN-3, new) | 0.07 | 0.2 | 0.3 |

* **110-150 s per decision is not reproduced**: a decision of the real configuration takes ~17-19 s here, 60-73 % of
  it the backbone writing the k = 4 alternatives (up to 512 new tokens; the generation itself is `nn/`'s). The
  b663d5e agent (the code the S-6 runs measured) profiled the same way on MECH:7 in the same session: 17.8 s per
  acted step (propose 11.8, generation 11.6, learn 3.0, evaluate 2.0, VOI 1.7, embed 1.5;
  `profile_mech_b663d5e.json`). The S-6 sequence run's 110-150 s were measured on a machine at load ~55 with another
  tenant on the GPU; no component of the agent, then or now, accounts for them.
* **Waste removed (same semantics)**: the VOI computed the decision's predictive means three times
  (`evaluate_candidates` already had them) and ran its `voi_max_candidates x voi_samples` (16 x 8) sampled-outcome
  re-evaluations on the GPU head view while the fast layer computes on the CPU (every re-evaluation copied every
  candidate's features to the CPU and back). Now the evaluation passes its means and the loop runs on a CPU copy of
  the view. Same numbers: on the CPU bit-identical (`tests/predict/test_int3a_cost.py`); on the real configuration the
  first VOI call of the SW run, captured with its inputs (fast state, generator state) and recomputed with the new code,
  differs by at most 4.4e-16 in VOI (max 0.26) and 7.1e-15 in information gain (max 16.4) — rounding of the same
  float64 operations on another device. VOI 1.67 -> 1.45 s per decision (MECH); the difference of whole steps
  (18.5 -> 17.1 s) is within the run-to-run spread (different actions are chosen, the generated text differs).
* Not changed: generation (11 s), embedding of new candidate texts (2 s: backbone proposals are new texts every
  decision; the embedding cache already serves repeats), the continual learner (`credit/`, frozen for verification).
  The runtime's own work is below 0.5 s per step.

What the runtime runs also show (S-6b in the loop, not only in the probe): on SW:rle the agent chose
`internal.request_operator` (asking the operator to confirm a solution; recorded, nothing sent) and then the
backbone-written `sandbox.write_file solution.py`; the verifier's next reading was pass fraction 1.0 and the objective
was verified achieved after 2 decisions (`runtime_sw_rle_profile.json`). On MECH:7 within 8 decisions the agent
measured the service with backbone-written `http.get`s at chosen points, wrote a placeholder `model.py` and inspected
its records; the verifier's R^2 stayed 0 (no fitted model was written in 8 decisions).

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
  `CandidateEvaluation.uncertainty` (`epistemic_before`). No network pass, no new prediction is registered. The
  views of executed decisions are checkpointed with the predictor state (a restored agent writes the same value) and
  dropped at a new objective (the fast state returns to the prior).
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
b663d5e. The independent verifier confirmed the connection (`docs/verification/G2_learn.md` G2L-7: the hook returns
stored credit for 20 / 20 and 34 / 34 records in integrated runs). Not shown: that credit makes the policy better (RUDDER / COCOA have not been promoted by the selector in any
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

## G2L-8: the first promotion of a short objective found no held-out evidence

Verifier report (`docs/verification/G2_learn.md` G2L-8): `test_int2_checkpoints.py::test_promotion_needs_independent_
evidence_and_a_refusal_rolls_back` passed 3 / 3 at 5562e65 and 1 / 4 at 3e29595 ("no independent validation metric
supplied"). Measured with the test's own setup (`val_probe.py` in the INT3A scratch: lifetime 31, then the first
promotion), per run: records of the first objective, records the learner holds out (`is_validation`), records still
carrying a `val:` flag at the promotion:

* 5562e65 (3 runs): 14 / 12 / 14 records; `val:` flags only for `bridge_grounding`, `rudder`, `cocoa`
  (`val:world_model`, `val:fast_prior`, `val:evaluator` 0 in every run); promoted 2 of 3.
* 3e29595 (3 runs): 12 / 12 / 12 records (the policy spends the money budget faster); `val:` flags only for
  `bridge_grounding`; promoted 2 of 3.

Cause (a defect in `predict/providers.py`, present in both): `_Base._permitted` read `record.eligible[<provider>]`,
which `ContinualLearner.refresh_eligibility` itself writes from `provider.eligible(record)`. A held-out record has its
train flag False by construction, so at the next refresh it was no longer eligible and its `val:` flag was cleared:
the world model's, fast prior's and evaluator's held-out sets emptied after one refresh, and promotion depended on the
few providers whose eligibility does not pass through `_permitted` and on how the objective ended. The same feedback
kept a record checked before its outcomes / verdict existed ineligible for ever: the evaluator (eligible only once
the objective has a signed verdict) trained on none of an objective's records (0 of 14 measured). Fix: `_permitted`
no longer reads the learner's flags. After (3 runs): 9 / 14 / 14 records, every held-out record keeps `val:world_model`,
`val:fast_prior` (and `val:evaluator` after the verdict), the evaluator's train flags are set for 12 of 14 records,
promoted 3 of 3; the INT2 test passed 8 of 8 consecutive runs (it also passed 4 of 4 here before the fix: the failure is
probabilistic). Regression tests `tests/predict/test_int3a_eligibility.py` (2; both fail on the old code).
Residual: with a 15 % record split an objective of N records has no held-out record with probability 0.85^N
(0.10-0.23 for N = 9-14), and then its promotion is refused (fail-closed). That is the split's property
(`credit/learner.py`, frozen): change request INT3A-5. Behaviour change to note: the evaluator and fast-prior now train
on records they were silently excluded from.

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

## Remaining limitations

* RUDDER / COCOA credit reaches training only once `credit.select` promotes one of them, and nothing in the agent
  feeds that selection (G2L-6 / RC-08: the agent's randomised experiments are not in the credit store); DX works on
  the credit side and may send a change request for `agent.py` — not started here.
* S-6b: shown for the first decisions of SW:rle (solved in 2 decisions through the runtime) and MECH:7 (measurements and
  a placeholder model in 8 decisions, R^2 0). No program execution (`sandbox.python`) was proposed in the probe rounds;
  whether the agent fits a model for MECH within an evaluation budget, and how it behaves on the planning task, is not
  measured here. Prompts are within the 2048-token window for these families; the planning task needs INT3A-4.
* Decision cost: ~17-19 s per decision measured (generation 60-73 %); the 110-150 s of the S-6 runs were not
  reproduced with either the old or the new agent; the VOI change saves ~0.2 s per decision.
* LRN-3 values are written; whether the exploration / learning-progress terms improve behaviour is not measured.
* The promotion test of the bootstrap mix is structurally unable to become conclusive at realistic evidence rates
  unless the learned reward is far better than the bootstrap (section 6); not changed.
* G2L-8 residual: an objective with no held-out record cannot be promoted (probability 0.85^N; INT3A-5).

