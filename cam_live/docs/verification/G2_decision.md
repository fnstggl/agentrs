# G2-decision — independent re-verification of prediction / decisions (B), the learned reward and meta-learning (C) and the agent-side credit items (D) at the code freeze

Verifier: G2-decision (lens `g2decision`). Code under test: **freeze commit `f54d8cc056722f2ba373ba70627bfc52e2500332`**,
exported read-only with `git archive f54d8cc | tar -x -C $SNAP` (scratch) and run with `PYTHONPATH` / `G2D_SRC`
pointing at `$SNAP/cam_live/src` (HEAD moved to HX commits during the run; `git diff f54d8cc HEAD` touches no file of
`predict/`, `reward/`, `credit/`, `nn/`, `agent.py`). Requirements: B — PI-01..PI-11, SE-03..SE-06, AE-01, CU-12;
C — MR-01..MR-11, AE-04; D (agent side) — RC-01, RC-02, RC-08. Verdicts: `verdicts_g2decision.json`. Probes:
`repro/g2decision/*.py` (each prints `[PASS]` when the property holds, `[FAIL]` when refuted; exit 1 on any refutation).
All commands from `/workspace/camlive`; `G2D_SRC=$SNAP/cam_live/src` selects the freeze snapshot. I edited no source
file and no other agent's test.

## 1. What was executed

| What | Command (on the snapshot) | Result |
|---|---|---|
| B default suite | `pytest cam_live/tests/predict -m "not gpu and not slow"` | **164 passed** (204 s) |
| C suite | `pytest cam_live/tests/reward` | **143 passed, 3 skipped** (150 s) |
| D suite | `pytest cam_live/tests/credit` | **135 passed, 8 skipped** (194 s) |
| integration | `pytest cam_live/tests/integration` | **64 passed, 1 skipped** (761 s) |
| slow / GPU suites | see section 6 | see section 6 |
| B measured numbers | default suite with `CAM_LIVE_B_MEASURE=<scratch>` vs `docs/evidence/B/measured.json` | 83 / 86 equal; 3 differ (`labworld_experiment_*`: estimate 2.72 vs 2.598, CI) — evidence older than the code |
| G-pr probes re-run unchanged | `G_PR_SRC=$SNAP/cam_live/src .venv/bin/python cam_live/docs/verification/repro/pr/<p>.py` | metagrad_small 40/40, integrity 21/21, predict_numerics 31/31, decisions 28/28, extras 4/4; metalearner_path 9/11 and metagrad_real 17/22 — **stale probes** (they encode the pre-LRN inner loss; superseded by `probe_meta_path.py`, section 3) |
| MetaLearner path with the current loss, 3 seeds | `G2D_SRC=... .venv/bin/python cam_live/docs/verification/repro/g2decision/probe_meta_path.py [--seed 6/7]` | **23/23, 11/11, 15/15** |
| Selection rule `thompson_policy` | `.../g2decision/probe_selection.py` | **10/10** |
| Proposer prompt / normalisation, CPU part + real-model output | `.../g2decision/probe_prompt.py --real <probe.json>` | **13/13** on the real-model output |
| PI-11, D1 re-check, promotion rule | `.../g2decision/probe_misc.py` | 2/3 — **G2D-1** |
| G2-learn's integrated-agent credit probe, snapshot copy | `G2_WORK=<scratch> .venv/bin/python $SNAP/cam_live/docs/verification/repro/g2learn/probe_agent_credit.py` | **13/13** (`repro/g2decision/out/g2learn_probe_agent_credit_f54d8cc.txt`): RC-01 links, credit hook 26/27 records, 0 selection rows |
| LRN choice diagnostic, condition E, 3 seeds x 8 lifetimes (debug scale) | `PYTHONPATH=$SNAP/cam_live/src .venv/bin/python $SNAP/cam_live/docs/evidence/B/choice_diagnostic.py --condition E --seed {0,1,2}` | expected choice quality 0.381 / 0.298 / 0.331 (mean 0.337; LRN: 0.375 [0.321, 0.428]); uniform 0.203, greedy-on-own-prediction 0.311, policy head alone 0.269 (LRN 0.216 / 0.392 / 0.308) (`repro/g2decision/out/choice_diagnostic_E_f54d8cc.json`) |
| Real Qwen3-4B + 50m proposer probe (GPU lock) | `PYTHONPATH=$SNAP/cam_live/src python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4 --root <traversable run root> --out P.json` | 16/16 + 16/16 valid, see section 4 |

The probe run roots live under `repro/g2decision/runs/` (git-ignored, deleted after the run) because the sandbox uid
must traverse to its workspace (INT3A's root cause of S-6b); my scratch directory is below a 0700 directory.

## 2. Earlier defects (G_pr.md) — re-checked at f54d8cc

| id | status | probe result at f54d8cc |
|---|---|---|
| D1 return-to-go baseline contains the row's own outcome | **fixed** | `probe_misc.py` check 2: baseline independent of the row's own measured outcome in 200 random chains (max |db| 3.6e-15); the return-to-go chain is one baseline group (`metalearner.py` `_episode`, `outer_baselines`); my recomputation of the applied gradient uses the same rule and agrees (section 3) |
| D2 caller-chosen utility admitted | **fixed** | `probe_integrity.py` PASS: "a validly signed outcome cannot be admitted with a utility the verifier never stated" |
| D3 self-generated goal supervises by default | **fixed** | `probe_integrity.py` PASS (refused, reason `objective_…`); the agent registers its task's origin |
| D4 goal-free budget ignores pending goals | **fixed** | `probe_extras.py` 4/4: `select(k=5)` under a 2-decision budget returns 2; repeated selection stays at 2 |
| D5 calibration slope CI ignores clustering | **fixed** | `probe_predict_numerics.py` PASS: clustered null excludes 0 in 21/300 = 7 % (was 61 %) |
| D6 reworded self-goals not discounted | **fixed** | weights 1, 0.5, 0.25, 0.125 for the four rewordings |
| D7 shadow update switches to on-policy | **fixed** | path probe: cold-start estimator `importance_weighted`; on-policy only when the live update replays bit-identically with the current psi and mix (my probe confirms the condition) |
| D8 truncated ledger keeps a valid chain | **fixed** | truncated ledger now fails `verify_chain` (`(False, 1)`), anchored by the head file |
| D9 bridge is a no-op at initialisation | unchanged, documented | structural; the bridge is trained only by D's BridgeGroundingProvider (not re-measured) |
| D10 NLL intervals not clustered | **fixed** | `score_predictions` clusters by experiment / decision / prediction |
| D11 float32 targets | **fixed** | `build_targets` keeps values and exposures in float64 |
| D12 'learned' label at mix 0 before any meta-update | **fixed** | label `bootstrap` until a meta-update was applied and promoted; MetaUpdateRecord reports the mix |

## 3. MetaLearner, inner update, trust region, credit (KF-E01) — attacks that failed

`probe_meta_path.py` runs the real debug `CamLiveModel` in LabWorld through `MetaLearner` with every part of the current
inner loss switched on (trust-region live step `target_kl` 0.05, matured replay 16, goal term, all-candidate model term)
and the KF-E01 credit hook returning return-like (RUDDER-like) credit for one third and per-decision (`simple`) credit
for another third of the decisions. LabCollector records carry no predictions or verified annotations, so the probe adds
them as seeded data (random `q_goal` per candidate, `reward_signals.verified_progress_delta` on half of the decisions);
the outer utilities still come only from the ledger's signed LabWorld verdicts. Everything below the data is mine.

* **The applied live update is the documented loss.** 11-15 live updates per seed (3 seeds, 38 in all) equal MY K-step SGD (my
  component-normalised mix, goal term c_g u / rms(u) with verified / credit:simple / calibrated-prediction sources,
  credited rows taking c_g C / rms(C) as their goal return, standardised returns, value baseline, entropy, all-candidate
  model term weighted by beta_q^+, per-prefix clip) to <= 4.3e-6 relative to the update; my recomputed goal sources,
  standardised candidate values, beta_q^+ and credited returns equal the batch's; the engine replays every live update
  bit-identically (`theta1_hash`), so what is differentiated is what was applied.
* **The applied meta-gradient is right.** In 5 applied meta-updates (3 seeds) the gradient `meta_update` applies equals my
  recomputation (own unroll in which the value baseline of step k is a sibling graph node of theta_k — fixed for the
  inner derivative, still carrying psi — own on-policy / importance decision from the documented condition, own ESS rule,
  own leave-group-out baselines, weight 1/N) to 2.4e-7 .. 7.2e-7 relative, including episodes estimated by
  importance weighting next to on-policy ones. The directional derivative of my surrogate equals central finite
  differences in float64 to <= 1e-6 relative, and the code's float32 gradient along the same random direction equals
  the FD value to <= 2e-5 relative. **K sweep** on the real adaptable subset (policy head + decision trunk of the debug-preset CamLiveModel) with the current loss: the code's engine gradient equals mine to
  <= 1.9e-15 and central FD to ~5e-10 for **K = 1, 2, 4, 8** (float64).
* **Trust region.** Every live step reaches KL in [target/1.5, 1.5 target] or sits at a step bound. Observation: the
  bounds bind often — in some runs steps sit at `max_inner_lr` = 2 with KL far below 0.05 (the in-graph gradient clip
  limits the update), in others at the lower bound `inner_lr` with KL up to 0.15 (3 x target); the agent's step range
  is [1e-2, 2]. My pathway runs at f54d8cc (section 6: debug scale, and the real Qwen3-4B + 50m configuration) show KL
  0.035-0.074 (debug) and 0.042-0.069 (real) per live update.
* **KF-E01 holds as documented**: return-like credit replaces the discounted goal return of the credited row, per-decision
  credit enters only rows without a final verified annotation, both inside the live and the differentiated update. In the
  integrated agent the estimator is `CreditAssigner.default`, which stays `simple` (RC-08), and `simple` credit is used
  only for attributing environments (`agent.py:2026`); G2-learn's agent probe re-run at f54d8cc: the hook returns stored
  credit for 26 of 27 records (all `simple`, per decision), 0 estimator-selection log rows, 0 of the agent's 6
  experiments in the experience store.
* Probe bug I hit and fixed (not a code defect): recomputing the goal inputs at meta-update time with the *current*
  selection calibration differs from the inputs the live update used; the code freezes them in the cached batch, which is
  the correct behaviour ("the differentiated update is the update applied").

Approximation stated (not a defect): the trust-region step is a constant of the differentiated function
(`metalearner.py:429-447`, used at `:702`); where the step is interior and the live mix < 1 the step depends on psi and
that derivative is dropped. At the agent's default (live mix 1) R_psi has weight 0 in the live update, the step does not
depend on psi and the statement is exact.

## 4. Selection rule and candidate generation

**thompson_policy** (`probe_selection.py`, 10/10): `DecisionMaker.select` draws the index from exactly the vector it
records (generator replay); over 20,000 decisions with overlapping posteriors the chosen-index frequencies equal the mean
recorded propensities (max |z| 0.9); infeasible = 0, eps floor exact; the distribution is invariant to q -> 37 q - 1234.5
and to logits + 3.7 z (the policy counts only beyond the world model's ranking); a uniform policy gives exactly the
value-only Thompson distribution; my own reference (own z, projection c, lambda from the same coefficient draws)
reproduces the vector to 0.0; the calibration posterior equals my weighted Bayesian linear regression, recovers a known
beta_pi / beta_q ratio (1.69 vs 1.75), and a preference unrelated to verified progress ends with mean lambda 0.003.
**Observation G2D-2**: with no verified outcome yet, the prior beta_pi ~ N(0, 0.5^2) already gives the policy weight
(mean lambda 0.36, P(lambda > 1) = 0.08); a peaked preference (logit 6 vs 0) moves a 6-candidate decision by TV 0.17
before any evidence. LRN_S5's "the policy's influence grows only as its preferences prove predictive" therefore holds
after evidence, not for the first decisions (an untrained policy has c = 0 and no influence, so this matters only for a
policy trained before its calibration has data).

**LRN choice diagnostic re-run at the freeze** (condition E, debug scale, 3 seeds x 8 lifetimes, same script):
expected choice quality 0.337 (seed means 0.381 / 0.298 / 0.331), inside LRN's interval [0.321, 0.428] around 0.375;
it is well above uniform (0.203) and here at or above greedy-on-the-agent's-own-predictions (0.311; LRN 0.392); the
policy head alone 0.269 (LRN 0.308). The direction of LRN's claims reproduces at f54d8cc; the numbers are an
engineering diagnostic at debug scale (one mix promotion, to 0.75, in seed 0; mix 1 in seeds 1-2).

**Backbone proposer, S-6b** (real Qwen3-4B + LoRA + bridge + 50m, my run of `cam_live.eval.proposal_probe`, traversable
root, sandbox available): every prompt listed every permitted available tool (14 of 14 SW, 16 of 16 MECH; none dropped,
none unavailable); prompt 1690 / 1932 tokens, inside the window (`max_prompt_tokens` 6144 in `configs/cam_live.yaml`; the
code default and INT3A.md say 2048 — both fit); the instruction block is byte-identical in all 8 rounds of both
objectives and names no tool, file or task word (checked against every tool name and the task vocabulary); the digest
adds only the task's own success definition (`{"metric", "target", "verifier"}`, nothing of the hidden tests) and
action results. Argument names are exact (no marker), required ones say so, nested object properties are shown; the
compact form drops only `minLength` / `maxLength` bounds of a few string arguments (listed in the probe output).
`normalise_proposal` undoes a `name*` marker only for a real property, removes unknown names only under
`additionalProperties: false`, coerces number-like text only for numeric schema types and never touches strings.
Results: 16/16 (SW) and 16/16 (MECH) backbone items valid; SW: a `sandbox.write_file solution.py` in 4/4 rounds — the 4
files pass **60/60** hidden tests when I score them offline (never shown to the agent); MECH: `model.py` written in 4/4
rounds, `sandbox.python` in 1/4 and one backbone multi-step program `program[sandbox.write_file, http.get, http.get]`.
INT3A's probe numbers (16/16, 1690 / 1932 tokens) reproduce. The only generic steering in the prompt is "Include actions
that change the environment as well as actions that only gather information" — generic diversity, not task-specific.

## 5. Defects and observations (new)

| id | sev. | where | finding | repro |
|---|---|---|---|---|
| G2D-1 | minor (PI-11) | `predict/predictor.py:694-697`, `predict/errors.py:51` | The Predictor's per-metric `ChangeDetector` is never reset (no `reset()` call in `predict/` or `agent.py`; the agent only checkpoints it); `detected_at` latches, so after the first detected shift every later confident miss of that metric is labelled `changed_conditions` and never `model_wrong` again, even after 2000 well-calibrated residuals. The labels are diagnostics only (no learning signal) | `G2D_SRC=$SNAP/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2decision/probe_misc.py` (check 1) |
| G2D-2 | observation (PI-08 wording) | `predict/policy.py:287-289` (prior), `:240-248` (`policy_weight`) | prior-only policy weight (mean lambda 0.36; TV 0.17 for a peaked preference) before any verified outcome | `probe_selection.py` (INFO lines) |
| G2D-3 | observation (MR-05) | `reward/metalearner.py:429-447, 702` | trust-region step held constant in the meta-gradient (exact at live mix 1) | code reading; section 3 |
| G2D-4 | observation (RC-08) | `agent.py:717` | DX-1 not applied at the freeze: no `attach_experiments`, so estimator selection never runs in the integrated agent | `grep -n attach_experiments cam_live/src/cam_live/agent.py` (no match) |
| G2D-5 | documentation | `docs/status/C.json` AE-04, `docs/INT3A.md` | C.json says the agent does not abandon unexecuted goal-free goals (C-11) — it does (`agent.py:1933-1938`); INT3A.md says a 2048-token window — the configuration has 6144; `docs/evidence/B/measured.json` 3 / 86 values not reproduced (`labworld_experiment_*`) | `probe_extras.py`; `configs/cam_live.yaml:26` |
| G2D-6 | verification | `repro/pr/probe_metalearner_path.py`, `repro/pr/probe_metagrad_real.py` | these G-pr probes encode the pre-LRN inner loss and fail on the current code for that reason only | superseded by `probe_meta_path.py` |

No defect was found in the meta-gradient, the importance-weighting / ESS rule, the trust-region live update, the
promotion test (300/300 decisions equal my one-sided t rule), the credit hook, the selection rule's propensities or the
proposer's prompt and normalisation.

## 6. Slow and real-model runs

| run | command (snapshot of f54d8cc) | result |
|---|---|---|
| B slow | `pytest cam_live/tests/predict/test_learning_evidence.py -m slow` | **7 passed** (380 s); the measured values differ from `docs/evidence/B/measured_slow.json` in 6 of 8 (e.g. world gain vs marginal at 2 d 0.56 vs 0.019; fast-adaptation NLL gain 0.068 vs 0.179; trained Brier 0.240 vs 0.262): the stored evidence predates the code; the tests' properties hold |
| C slow | `pytest cam_live/tests/reward/test_reward_experiment.py -m slow` | **1 passed** (33 s) |
| D slow | `pytest cam_live/tests/credit -m slow` | **6 passed** (545 s), incl. DX's integrated-agent test that applies the DX-1 line itself |
| real proposer | `python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4` (lock obtained 02:46-03:01) | section 4: 16/16 + 16/16 valid, all permitted tools listed, SW files 60/60 offline |
| pathway, debug scale | `cam_live/docs/evidence/C/real_pathway.py --debug --lifetimes 4 --meta '{"mix": 0.5, "mixer_step": 0.0}' --agent '{"selection_rule": "thompson_policy", "use_backbone": false, "promotion": false}'` (stopped by my 1700 s timeout in lifetime 4 on a loaded machine) | 10 live updates, KL 0.035-0.074, each depending on R_psi's shape (TV 0.006-0.148); 1 applied meta-update of 55 calls (importance-weighted, gradient 1.62, psi delta 0.047, 14 psi tensors); R_psi on the support changed by 0.0048 (std 0.030); that event's live update recomputed with psi after vs before: TV 0.0114. The *next* live update after the step was not reached before the timeout |
| pathway, REAL Qwen3-4B + LoRA + bridge + 50m | same, without `--debug`, `--lifetimes 3 --budget-s 780` (lock obtained at the 3rd attempt, 04:40 UTC) | 2 lifetimes (the script stopped before the 3rd: lock budget 780 s: 526 s used, the longest lifetime took 338 s, not starting lifetime 2); 8 live updates, KL 0.042-0.069, each depending on R_psi's shape at mix 0.5 (TV 0.0010-0.0100); 40 meta-update calls, **0 applied** ({'insufficient_independent_outcomes': 35, 'no_policy_update_recorded': 2, 'no_supported_episode': 3}); `pathway_complete` False: the outer step and its effect on the next live update were NOT reached in the two lifetimes the hold allowed; lock 526 s, peak 10.2 GB |
| B real-configuration / proposer GPU tests | `pytest cam_live/tests/predict/test_gpu_real_config.py cam_live/tests/predict/test_proposer.py -m gpu` | **not run**: both errored with pytest-timeout (900 s) while waiting for the GPU lock (03:01-03:31); a retry after the pathway attempts did not get the lock either. The proposer claim is covered by my own real-model probe run above |
| C 50m cost test | `pytest cam_live/tests/reward/test_reward_experiment.py -m gpu` | not run (GPU lock contention; cost measurement only) |

## 7. Per requirement (basis; full notes in `verdicts_g2decision.json`)

| id | verdict | basis |
|---|---|---|
| PI-01 | partial | likelihoods == scipy, action / horizon-conditioned predictions (real configuration: G-pr at 9ba1e45; B's GPU test could not get the lock at the freeze); FAILS on held-out new mechanisms (no gain over the action-independent baseline, B's own scientific failure); debug-scale learning only |
| PI-02 | verified | registry refusals and registration-before-return re-checked; suites pass |
| PI-03 | verified | targets from matured observations only (float64); providers now train under the integrated ContinualLearner (G2L-8 fixed); debug scale |
| PI-04 | verified | two horizons per candidate (tests at f54d8cc; real configuration: G-pr); continuous horizon input |
| PI-05 | verified | epistemic + aleatoric reported; shrinkage tests pass |
| PI-06 | verified | masking, never filled; zero loss for pending |
| PI-07 | partial | as B claims: horizon never > 1, transition worse than the mean, no decision comparison |
| PI-08 | verified | all rules incl. thompson_policy: recorded vector is the sampling distribution, frequencies match, invariances, own reference equal (G2D-2 wording only) |
| PI-09 | verified | real-model proposer: 32/32 valid, provenance, a backbone program; probabilities exact; bridge no-op at init stays a stated limitation |
| PI-10 | verified | imagined objects refused everywhere checked |
| PI-11 | partial | unclaimed; classifier and test exist; G2D-1: change detector latches forever (never model_wrong after one detected shift) |
| SE-03 | verified | NB / hazard / log-normal likelihoods == references |
| SE-04 | partial | D5 / D10 fixed; calibration measured only in debug LabWorld diagnostics, no report on the runtime / evaluation path |
| SE-05 | verified | quality / luck decomposition exact |
| SE-06 | partial | VOI correct; learned downstream-usefulness estimator untrained and unscored (B's own) |
| AE-01 | verified | registration before assignment; assignment frequencies |
| CU-12 | verified | on the harness's discover_tools() output with the real backbone: every permitted tool listed exactly, programs proposed |
| MR-01 | verified | rewards written per decision; LRN-3 / KF-E02 inputs produced |
| MR-02 | verified | psi changes only by applied meta-updates |
| MR-03 | verified | live update == my documented loss with R_psi (<= 4.3e-6); real Qwen3-4B + 50m at f54d8cc: all 8 live updates depend on R_psi's shape at mix 0.5 (TV 0.001-0.010); weight 0 at the default mix 1 until promotion |
| MR-04 | verified | engine == mine (1e-15) == FD (5e-10), K = 1, 2, 4, 8, real subset, current loss |
| MR-05 | verified | applied meta-gradient == my recomputation (<= 7.2e-7), FD agreement; D1 / D7 fixed; an applied outer step on the real configuration was not reached at the freeze (0 / 40, insufficient independent outcomes in 2 lifetimes) |
| MR-06 | verified | D2 / D3 fixed |
| MR-07 | verified | small-reference probe 40/40 and real-subset K sweep |
| MR-08 | verified | bootstrap labelled, shadow update importance-weighted, promotion == one-sided t rule |
| MR-09 | verified | 143 reward tests; D2 / D3 / D6 / D8 attacks fail now |
| MR-10 | verified | skip with reason; D1 fixed |
| MR-11 | verified | psi untouched by policy updates; ledger anchored |
| AE-04 | partial | D4 fixed, C-11 applied; transferable competence is a constant 1.0 |
| RC-01 | verified | credit + integration suites incl. the delayed-linking acceptance test pass at f54d8cc; G2-learn's agent probe 13/13 on the snapshot |
| RC-02 | verified | KF-E01 credit path re-verified in the live and differentiated update; hook returns stored credit for 26 / 27 agent records |
| RC-08 | partial | selection never runs in the integrated agent (DX-1 not applied, agent.py:717); lab selection held out correctly |

## 8. Not checked

* Only condition E of LRN's choice diagnostic was re-run (D, E_mix0 and the 'before' arm not); C's LabWorld
  experiment was not re-run; INT3A's runtime SW:rle run (verified achieved after 2 decisions) and its decision-cost
  profile were not re-run.
* No learning on the real configuration was attempted by anyone (B, C state it); everything learned is debug scale.
* PI-07 / SE-06 usefulness (planning, learned downstream-usefulness estimator) not re-measured — both are claimed
  partial by B.
* D9 (bridge conditioning) not re-measured.
* GPU tests that could not get the shared lock at the freeze (evaluation batch and other verifiers' real-model runs
  held it): `tests/predict/test_gpu_real_config.py`, `tests/predict/test_proposer.py -m gpu`,
  `tests/reward/test_reward_experiment.py -m gpu` (50m meta-update cost). An applied outer step on the real
  configuration was not reached in the one hold I got (2 lifetimes); C's evidence for it is from b6006a2.
