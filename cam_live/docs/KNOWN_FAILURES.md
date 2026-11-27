# CAM Live 1.0 — known failures, negative results and deployment blockers (IN-06)

This is an append-only register. New entries go at the end under a dated heading, and an entry whose state changes
gets a new line in a later section rather than an edit. Each entry gives:

* what was measured or found;
* its evidence: a file under `cam_live/docs/` unless an absolute path is given;
* its state at the commit named in the section heading. The states are:
  * **open**: still true at that commit.
  * **fixed**: the fix is recorded with a probe or test, named.
  * **stale verdict**: a failed verdict was issued before a recorded fix and has not been re-issued.
  * **not re-checked**: taken from the record; DOC did not confirm it against the code.

Synthetic environments (LabWorld, the tiny random backbone, the synthetic credit / fast-layer worlds) are
**engineering diagnostics** (operator redirection 2026-11-27, `docs/S/PROTOCOLS.md` v3). Their negative results are
listed so that no one reads them as successes. They are not evidence about the real configuration or the Discord
objective.

Sources read: `docs/status/*.json` (`scientific_failures`, `not_done`), `docs/S/DIAGNOSIS.md`, `docs/INTEGRATION.md`
§7–§9, `docs/change_requests/*.md`, `docs/verification/verdicts*.json`, `docs/COMPLETION.md`, the ledger, and DOC's
own documents (`docs/MODEL_ARCHITECTURE.md` §10, `docs/MATH_NOTES.md` §13–§14, `docs/REPRODUCE.md` §13,
`docs/OPERATOR_RESOURCES.md`).

---

## 2026-11-27 — initial register (DOC, code at `204e9d4`)

### 1. Deployment blockers: live execution

| Id | Blocker | Evidence | State |
|---|---|---|---|
| KF-D01 | **No durable state.** The doctor check `persistent_storage` fails (`workspace_is_volume=False`). It is `needed_for="live"`, so the preflight's live report always contains it, and the supervisor of `cam-live run` refuses to start the learner (`runtime/cli.py:cmd_supervise`). The code has no override. Run data and checkpoints have no off-box copy. The rental ends 2026-11-30 03:00 UTC. | `docs/OPERATOR_RESOURCES.md` §2.1, §6; `docs/evidence/DOC/command_check.txt` (doctor 48/51 ok, `blocking_live = [persistent_storage]`); ledger | open |
| KF-D02 | **No live authorization record, and no preflight against running live services recorded.** Status level 3 (ready for authorized live deployment) is not reached: "no preflight evidence (docs/evidence/preflight*.json) recorded". | `docs/COMPLETION.md` (generated at `799f032`) | open: needs the operator's signature (`tools/launch_live.sh`) |
| KF-D03 | **Execution task incomplete.** `tasks/discord_100.yaml` has 16 operator values unset (limits, tools, destinations, publishing, currency, guild, credential name, owned site, record id, test accounts, feedback permission). The preflight reports 25 live items. | `docs/OPERATOR_RESOURCES.md` §3, §6 | open: operator values |
| KF-D04 | **No live service was contacted by the agent.** The read-only live Discord observation run is prepared but was not executed: the session's permission system refused its launch, and it needs the operator. The planning phase is prepared, not run. | `docs/INTEGRATION.md` §8; `docs/status/M-int.json` not_done; `docs/evidence/validation/discord_live/run_live_observe.py` | open |
| KF-D05 | **Engineering completion not reached.** At `799f032`: 198 requirements, 5 complete, 110 implemented but needing re-verification, 75 partial, 8 missing, 0 failed. Status levels 1–5 are all false. | `docs/COMPLETION.md` | open |
| KF-D06 | **No kernel-level per-destination network filtering** (no `CAP_NET_ADMIN`). Destinations are enforced only by the authorization layer. There is no seccomp on the authorization service and no append-only flag on the audit log (no `CAP_LINUX_IMMUTABLE`). A VM is recommended for live operation. | doctor `kernel_network_filtering`; `docs/status/EA.json`, `EC.json` not_done | open (host limitation; preflight warning, not a block) |
| KF-D07 | **The runtime does not take the shared GPU lock.** A full-configuration run occupies the shared GPU for its whole duration. Measured peaks: 10.4–18.6 GB. | `docs/REPRODUCE.md` §13; `docs/INTEGRATION.md` §7.3, §7.6 | open (operator coordination) |
| KF-D08 | **Discord platform rules not checked against current published policy.** The shipped rules are EA's reading, written as editable data. Spec 12.5 asks for the check before activation. | `docs/status/EA.json` not_done | open |
| KF-D09 | **Record identity is checked loosely.** `resources.operator_authorization_record` is checked as a slug only, not against the `record_id` that `authorize_live.py` writes. `configs/cam_live.yaml` has no `launcher.operator_pubkey`, so a bare `cam-live run --authorization` fails at registration ("operator public key not installed") unless the key is installed first, as `launch_live.sh` does. One operator key; no rotation. | `docs/REPRODUCE.md` §13; `docs/OPERATOR_RESOURCES.md` §2.2; `docs/status/EA.json` | open |
| KF-D10 | **Approval from the dashboard needs root.** `resolve_approval` is root-only by peer uid, and no approval proxy exists, so the operator approves with `opctl` or `cam-live approve`. | `docs/status/EA.json` not_done | open |
| KF-D11 | **Discord user-account automation refused.** A login of the dedicated account stopped at hCaptcha and was not circumvented. Policy rule `PL-DISCORD-NO-USER-AUTOMATION` keeps it out of CAM's action space. | ledger session 1 | by design |

### 2. Real configuration (Qwen3-4B + 50m core): negative and unfinished results

| Id | Result | Evidence | State |
|---|---|---|---|
| KF-R01 | **S-6, the blocker for substantive evaluation.** Open-ended generation does not attempt the software objective. Snapshot `737996e`: 0 backbone-written candidates in 15 decisions on SW:rle, verifier pass fraction 0.0. After the INT2 proposer work (snapshot `d55986d`): the probe gave 16/16 valid backbone actions in 4 rounds, but none attempts the objective; every round asks the operator for a file-writing capability the task already grants. Through the runtime: SW, 7 decisions, `solution.py` never written; MECH, 5 of 6 executed, measurements at points the model chose. | `docs/change_requests/S.md` S-6; `docs/evidence/S/v3/`; `docs/evidence/validation/s6/probe_sw_20261127.json`, `oe_seq_20261127.json`; `docs/INTEGRATION.md` §7.5 | open (INT3A working on it) |
| KF-R02 | **Real-configuration staging validation executed, not passed.** Failed checks: `approvals_exercised`, `resource_limit_exercised` (the decision budget ran out first), `meta_update_applied` and `meta_step_changes_policy` (0 applied: 39 skips for insufficient independent outcomes, 2 unsupported). | `docs/evidence/validation/staging_20261127T223510Z.json`; `docs/INTEGRATION.md` §7.6 | approvals: fixed in `1039a6f` (`tests/integration/test_int2_queries.py::test_a_result_after_approval_is_recorded_without_rewriting_history`), not re-run on the real configuration; the rest open |
| KF-R03 | **Meta-updates are rare and small.** 63 calls, 4 applied (gradient norms 0.05–2.89, psi deltas 0.06–0.10, policy change TV ≤ 0.027). Whether the changed reward improves decisions is not measured. With the default bootstrap mix (stays 1.0) the live policy update is not changed by R_psi. It changed only with the mix fixed at 0.5 and rollback off (TV 0.0090 / 0.0042). | `docs/evidence/validation/meta_real/meta_real_20261127T184014Z.json`; `docs/evidence/C/real_pathway_20261127T200546Z.json`, `real_pathway_check_20261127T223812Z.json`; `docs/status/C.json` not_done (LRN) | open |
| KF-R04 | **Learning quality not established.** No repeated-task learning run of the real configuration was made (GPU time). There is no substantive result for SW / MECH / CU (Protocol v3 X0–X5 not run on a frozen snapshot). | `docs/INTEGRATION.md` §8; `docs/status/S.json` | open |
| KF-R05 | **On Qwen3-4B the world-NLL gradient that reaches LoRA is tiny** next to the bridge-grounding loss (adapter gradient norms 0.00101 vs 5.547), so the slow update is in practice driven by grounding. Adapter learning that improves prediction or generation is not shown. | `docs/evidence/D/real_update.json`; `docs/status/D.json`, `A.json` | open |
| KF-R06 | **`generation_logprob` on the bf16 trunk** differs from float32 by up to 0.36 nats on a one-token completion. The KV-cache sampler and `generation_logprob` differ by up to 0.073 nats per token. | `docs/status/A.json` scientific_failures; `docs/verification/G_nn.md` D3 | open (partly fixed: float32 head) |
| KF-R07 | **Features phi are ill-conditioned at initialisation** on real inputs: Gram max/min eigenvalue 2.3e4, 4.5e5 for one query, participation ratio 2.0. | `docs/status/A.json`; `docs/verification/G_nn.md` D6 | open |
| KF-R08 | **Policy dependence on the belief is small at initialisation:** KL 7.4e-05 on the real configuration (4.10 after imitation training). | `docs/status/B.json` | open (property of the untrained head) |
| KF-R09 | **Qwen3-8B and the 150m / 500m / 1b core presets were never trained or run on GPU.** No experiment shows whether more experiential capacity helps (spec 16.10). | `docs/status/A.json` not_done | open |
| KF-R10 | **Update times depend on machine load:** frequent update 8.6 s and slow update 9.7 s under load, against 0.73 s and 2.6 s at light load. Real-model decisions took 110–150 s each in the S-6 runtime sequence. | `docs/evidence/D/real_update.json`; `docs/INTEGRATION.md` §7.5 | recorded |

### 3. Engineering diagnostics: negative results (debug scale, synthetic worlds)

| Id | Result | Evidence | State |
|---|---|---|---|
| KF-X01 | **LabWorld choices near uniform** (v2, `00cb5a4`): no condition beats uniform by a detectable margin, and no condition improves over lifetimes (every slope's CI contains 0). Selection ignored the world model: greedy on the agent's own predictions would have given 5.05 expected adopters per decision against the behaviour's 1.42. 70–73 % of positive internal reward went to zero-value operations. The internal return of a lifetime is negatively correlated with verified progress (−0.17 [−0.32, −0.05]). | `docs/S/DIAGNOSIS.md`; `docs/evidence/S/diagnostics/v2_main_cpu_*.json`; `docs/change_requests/S.md` S-5 | partly addressed by LRN (KF-X02) |
| KF-X02 | **After the LRN fixes** (thompson_policy, standardised mixing, ΔV fix): expected choice quality 0.311 → 0.375, policy head 0.196 → 0.308, but **no learning curve over lifetimes** at this scale. | `docs/C/LRN_S5.md`; `docs/evidence/B/choice_summary.json`; `docs/status/B.json` | open (no lifetime-to-lifetime improvement) |
| KF-X03 | **Repeated staging task, debug scale:** success fell with repetition (learn_primitive 0.42 → 0.25, change CI [−0.29, −0.04]). No skill was registered (9 induce attempts failed validation), so skills against primitives is not evaluated. | `docs/evidence/validation/learning/debug_store_entry_20261127/`; `docs/INTEGRATION.md` §7.2 | open |
| KF-X04 | **No positive transfer** in the CPU transfer diagnostic: S_q 0.86 [0.67, 0.98] (T1), 0.97 [0.68, 1.47] (T2). | `docs/evidence/S/diagnostics/v2_transfer_cpu_summary.json`; `docs/status/S.json` | open |
| KF-X05 | **The learned reward from scratch hurt** (mix 0, 10 seeds × 320 lifetimes): 0.136 against 0.486 for bootstrap-only and 0.422 for verified-only; paired differences −0.350 [−0.430, −0.270] and −0.286 [−0.371, −0.202]. | `docs/status/C.json` scientific_failures; `docs/evidence/C/diagnostics.json` | open |
| KF-X06 | **Cold start against bootstrap-only:** no detectable difference (−0.038 [−0.186, +0.111]). The final mix is 1.0 on every seed. Bootstrap-only against verified-only: also no difference (+0.064 [−0.012, +0.140]). | `docs/status/C.json` | open |
| KF-X07 | **No learner discovered the key-to-door dependency:** best arm 0.486, against 0.90 for the scripted reference. | `docs/status/C.json` | open |
| KF-X08 | **In E the learned reward changes psi but not the live inner updates:** 9.1 applied meta-updates per run, 10/10 runs, bootstrap mix 1.0 throughout. Only E_mix0 lets psi act. | `docs/status/S.json` | open (same mechanism as KF-R03) |
| KF-X09 | **Stage-1 meta-gradient approximations.** First-order approximation error 13 / 30 / 46 % at K = 2 / 4 / 8. Adjoint checkpointing saves little or costs memory at the 50m preset. Truncated windows are biased in magnitude (relative error 0.75 / 0.55 / 0.24 for K = 1 / 2 / 4 of 8 updates). Small inner batches change the objective (gap 9–22 % at n ≤ 2). The meta-learned reward does not beat an oracle-scaled extrinsic reward under exact updates. | `docs/status/C.json`; `docs/evidence/C/exact_meta_gradient.json`, `meta_cost*.json` | recorded (exact unroll is the default) |
| KF-X10 | **World model on new mechanisms is no better than an action-independent baseline:** held-out NLL 1.666 vs 1.645 (6 h) and 2.211 vs 2.163 (2 d). The latent transition is worse than predicting the mean belief (normalised error 1.46), so imagination never extends beyond one step. | `docs/status/B.json`; `docs/evidence/B/diagnostics.json` | open |
| KF-X11 | **Fast adaptation lowers NLL (+0.119) but not the error of the mean** (−0.208 [−0.532, 0.115]). | `docs/status/B.json` | open |
| KF-X12 | **The evaluator does not beat the base rate** (Brier 0.240, equal to the base rate). It is close to a constant (ECE 0.021), and its calibration slope is weakly identified (cluster-robust CI −27.95..635.90). | `docs/status/B.json` | open |
| KF-X13 | **VOI.** The probe's VOI is not higher than the noisy gauge's: both 0 on the fast path, and the slow path shows spurious value from the gauge. VOI-aware selection did not need fewer interventions than greedy (−1.67 [−4.02, 0.68]). The plain estimator E[max Q(updated)] − max Q is biased upwards, which made VOI selection worse; the Jensen gap is used now. The learned downstream estimator did not beat computed VOI (Spearman 0.617 vs 0.671). | `docs/status/B.json`; `docs/evidence/B/diagnostics_value_of_information_plain_estimator.json` | open (slow path off; downstream head unused, KF-E05) |
| KF-X14 | **Fast Bayesian layer against baselines.** The learned prior is worse than the GRU at k < 3 on Gaussian held-out families (k = 0: NLL 1.700 vs 1.419). Fixed learned noise fails (1.993 vs 0.891). Counts are only level (−0.014 ± 0.008). The GRU has the lower RMSE. The training-family lead reflects the recorded GRU's size. | `docs/status/A.json`; `docs/evidence/A/fast_adaptation.json`, `fast_adaptation_noise_mode.json`; `docs/verification/G_nn.md` §4 | open |
| KF-X15 | **Credit estimators fail the held-out gate:** RUDDER mean z² 6.61, COCOA 13.84, gate 3.0, so the default stays `simple`. An earlier "rudder promoted" came from in-sample validation and is withdrawn. Learned COCOA is the less accurate estimator (max error 0.208 vs 0.036). Return-conditioned hindsight is biased (0.061; 16/81). The redistributed return-to-go gives no variance reduction at the key decision (8.09 for both). | `docs/status/D.json`; `docs/evidence/D/diagnostics.json` | open |
| KF-X16 | **Held-out prediction after learning:** mean NLL improved (2.981 → 2.291), median did not (1.902 → 2.106); 0.50 of records improved. | `docs/status/D.json` | open |
| KF-X17 | **Visual grounding.** MAI-UI-2B is worse than GoClick-Base on dense forms (0.50 vs 0.90). GoClick's confidence is ≈ 1.0 for wrong answers, so it cannot gate escalation. The best grounder (Vocaela-2-500M, 0.988) has a non-commercial licence. Single run, no confidence intervals. | `docs/status/EC.json`; `docs/evidence/EC/grounding_bench.json` | recorded |
| KF-X18 | **Skills did not reduce harness latency** (0.0050 vs 0.0029 s per task, ratio 1.71). The only gain is decisions per task (3 → 1). | `docs/status/ES.json`; `docs/evidence/ES/skills_vs_primitives.json` | recorded |
| KF-X19 | **Integrity battery on the current code:** 4 of 7 attacks detected in 5/5 seeds. `imagined_outcome`, `inflated_signed_measurement` and `relabelled_measurement` were not applicable in all 5 seeds, so they were not exercised. | `docs/evidence/S/diagnostics/v3_current_code/integrity.json` | open (coverage gap) |

### 4. Engineering defects and gaps open at `204e9d4`

| Id | Defect / gap | Evidence | State |
|---|---|---|---|
| KF-E01 | **Credit trains nothing.** Simple, RUDDER and COCOA credit is computed and stored (`ContinualLearner` calls `credit.assign`), but no loss reads `ExperienceRecord.credit`. Spec 9.2 ("use redistributed credit during eligible training updates") is not met. | `docs/MATH_NOTES.md` §14; `docs/status/D.json` not_done ("credited returns are exposed but not consumed"); grep of `reward/`, `predict/`, `agent.py` | open |
| KF-E02 | **The reward signal `info_gain` is never produced.** `reward/assemble.py` reads `CandidateEvaluation.extras["info_gain"]`; the predictor writes `info_gain_nats` (`predict/predictor.py`). The signal is masked in every runtime run. | `docs/MATH_NOTES.md` §13; confirmed by grep at `204e9d4` | open (found by DOC) |
| KF-E03 | **LRN-3: the bootstrap inputs `epistemic_after` and `learning_progress` are not written.** Together with KF-E02, the bootstrap reward is in practice ΔV + the verified-progress term. | `docs/change_requests/LRN.md` LRN-3; `docs/INTEGRATION.md` §8 | open (INT3A scheduled) |
| KF-E04 | **The experiential core's slots are not carried between calls:** every caller passes `state=None`. The state between decisions is the re-encoded context (≤ 256 tokens) plus `FastBeliefState`. | `docs/MODEL_ARCHITECTURE.md` §10.1 | open (design difference) |
| KF-E05 | **R_psi's lifetime context is reset at every objective change** (`CamLiveAgent.set_objective` → `MetaLearner.new_lifetime`). The architecture (§4) says it is not, and so does `reward/intrinsic.py:LifetimeContext.new_lifetime`'s docstring ("never at an objective boundary"). | `docs/MODEL_ARCHITECTURE.md` §10.2; `src/cam_live/agent.py:set_objective` | open (design difference) |
| KF-E06 | **The downstream-usefulness VOI head is attached and counted but not trained and not used.** VOI cost and time weights are 0, so VOI is gross of cost. Selection uses neither VOI nor imagined values. | `docs/MATH_NOTES.md` §14; `docs/status/B.json` not_done | open |
| KF-E07 | **Outcomes still pending when an objective completes are not measured.** The drain covers only an exhausted decision budget. | `docs/INTEGRATION.md` §8; `docs/status/M-int.json` | open (INT3B scheduled) |
| KF-E08 | **Meta-update evidence is scarce by design.** One lifetime is one independent group; ESS ≥ 2 is needed per episode, and 8 groups at the default (`meta_min_independent_outcomes`). In the CPU test 1 of 39 calls was applied; in the real staging run, 0. | `docs/INTEGRATION.md` §8, §9 e; `docs/evidence/validation/diagnosis/meta_evidence_20261127.json` | open |
| KF-E09 | **INT2-9: a parsed program gets its first step's modality.** Worked around in `agent.py:_feasible`; the proper fix in `predict/candidates.py` is pending. | `docs/change_requests/INT2.md` INT2-9 | workaround; open |
| KF-E10 | **INT2-5: the whole suite cannot be collected in one pytest process** (1459 collected, 1 error from duplicate module names). `cam-live test` without `--suite` hits the same error. The full suite is run one process per directory. | `docs/change_requests/INT2.md` INT2-5; `docs/REPRODUCE.md` §13 | open (INT3B scheduled) |
| KF-E11 | **`cam-live validate --mode staging --debug` is not a CPU run.** The CLI passes the full default configuration, and `validation.py:staging_config` keeps it, so it builds Qwen3-4B + 50m on CUDA without the GPU lock. | `docs/REPRODUCE.md` §13 (found while checking commands) | open (not re-checked by a probe) |
| KF-E12 | **Documented commands that do not exist:** `cam-live plan-report` (ledger), `eval.oe_runner fresh` (actual: `sequence --from-export`), `eval.cli batch --plan baselines_cpu` (choices `main_cpu`, `pilot`), and `eval.cli continual / shadow / skills`. | `docs/REPRODUCE.md` §13; `docs/evidence/DOC/command_check.txt` | open (`plan-report` scheduled for INT3B) |
| KF-E13 | **B-CR-10:** the binary fast layer is combined with the head by a posterior-shift approximation. Fast posteriors are per outcome kind, so a real-valued reading cannot inform a count posterior. Categorical outcomes are head-only. Time-to-event is constant-hazard. | `docs/change_requests/B.md` B-CR-10; `docs/status/A.json`, `B.json` | open |
| KF-E14 | **Promotion tolerances are set by hand, not calibrated:** margin 0.05 × utility RMS, step 0.25, 4-group minimum, held-out fraction 0.25. Checkpoint promotion is a gate with fixed absolute tolerances. The statistical non-inferiority rule applies to the bootstrap mix only. | `docs/status/C.json` not_done; `docs/MATH_NOTES.md` §10, §14 | open |
| KF-E15 | **Lifetime value beyond the truncation window** (Zheng et al. Eq. 5–6) is not implemented: intrinsic returns stop at the end of the support batch. The inner-data score term is inactive with `unroll_events = 1`. | `docs/status/C.json`; `docs/MATH_NOTES.md` §9, §14 | open |
| KF-E16 | **Unbounded bookkeeping and storage:** MetaLearner maps grow with the lifetime. The experience store has no blob garbage collection or retention policy. Logs, journal and event streams are append-only without rotation. | `docs/status/C.json`, `D.json`, `EB.json` not_done | open |
| KF-E17 | **The Predictor's decision-time feature cache is in memory:** after a restart, outcomes of earlier predictions resolve their records but no longer update the fast belief. | `docs/status/B.json` not_done | not re-checked |
| KF-E18 | **Harness gaps.** TLS tunnels are judged by host:port only. Structured observations do not enter iframes or closed shadow roots. `http_api.py` does not pin resolved addresses. Detection in opaque payloads is substring / URL based. Content detectors are lexical. The tool table denies operations it does not match. | `docs/status/EB.json`, `EA.json` not_done | open |
| KF-E19 | **The sandbox in the separated deployment** was recorded as unavailable while the executor runs as `cambrowser` (EA change request 3). The INT2 separated run lists a sandbox uid (996). | `docs/status/EC.json`, `EB.json` not_done; `docs/INTEGRATION.md` §7.1 | not re-checked |
| KF-E20 | **Discord adapter gaps.** No webhooks, threads, reactions, guild icon / banner, or editing of scheduled events. No persistent Gateway connection: each window is one Identify. `missing_capabilities()` is not wired into the operator-request file. The staging imitation's reading of error codes, invite metadata, Gateway timing and the 2FA requirement is unverified against the real service. | `docs/status/ES.json` not_done | open |
| KF-E21 | **Goal-free mode.** Expected transferable competence is a constant. On the Qwen3-4B embedding no threshold separates reworded self-goals from different ones (only the lexical fold is on). | `docs/status/C.json` not_done | open |
| KF-E22 | **Unclaimed requirements:** the 40 requirement ids added by MAN have no owner claims, and the checker (IN-04) and the standalone check (IN-09) have no independent verification. | `docs/change_requests/MAN.md` MAN-1; `docs/status/MAN.json` | open |

### 5. Verification verdicts recorded as failed or partial

| Id | Verdict | Evidence | State |
|---|---|---|---|
| KF-V01 | AE-04 **failed** (G-pr D4): the goal-free budget was not enforced for selected but unresolved goals | `docs/verification/verdicts.json`, `G_pr.md`; fix in `docs/C/FIXES.md` D4 (`tests/reward/test_reward_fixes.py::test_goal_free_budget_is_enforced_including_pending_goals`) | stale verdict (fixed, not re-verified) |
| KF-V02 | LX-10 **failed** (G-rt D1): the dashboard's number formatter let learner-written records run script in the operator's session, and that script approved an approval. `/api/health` leaked unredacted errors. | `docs/verification/verdicts.json`, `G_rt.md`; fix in `docs/F/FIXES.md` D1 | stale verdict (fixed, not re-verified) |
| KF-V03 | LX-14 **failed** (G-rt): approved actions were executed in a batch without limit checks, NaN / infinite limits passed validation, and `authorization_revoked` had no implementation. | `docs/verification/verdicts.json`, `G_rt.md`; fixes in `docs/F/FIXES.md` (D13 and others) | stale verdict (fixed, not re-verified) |
| KF-V04 | **65 partial verdicts.** Per lens: cl 8, env 21, nn 3, pr 11, rt 18, sec 4. Most were issued before the fix rounds. | `docs/verification/verdicts*.json` | open (G2 re-verification running for A + D; B/C, F/M and harness after the freeze) |
| KF-V05 | **rt probes still reporting BROKEN** for checks that encode the old design (p08 6, 8; p05 1a; p09 1b; p11 14). p10 aborts at 4a because the staging run exceeds the probe's timeout. | `docs/status/F.json`; `docs/change_requests/F.md` F-8 | open (probe update requested from G) |

### 6. Recorded failures fixed since (kept for the record)

| Failure | Fixed by | Evidence |
|---|---|---|
| The runtime consumed the single-use grant before `env.execute` (every action refused by `ComputerEnvironment`) | INT2 (`runtime/loop.py:look_grant`) | `docs/INTEGRATION.md` §2; `tests/integration/test_int2_wiring.py` |
| Candidate generation offered 4 of 12 interventions (verified progress 0.31 → 0.65 once spaces are enumerated) | INT2 (`proposer_per_tool = 12`) | `docs/INTEGRATION.md` §9 a; `docs/evidence/validation/diagnosis/debug_f1_20261127/` |
| The verified term had no input in runtime runs (INT2-1); the first fix paid waits and probes | INT2 / LRN | `docs/INTEGRATION.md` §9 b |
| Unanswerable duplicate questions: 0 of 24 interventions resolved for the meta-learner | INT2 (`d55986d`) | `docs/evidence/validation/diagnosis/meta_evidence_20261127.json`; `tests/integration/test_int2_queries.py` |
| Restored meta events rebuilt different support batches (INT2-8) | LRN (`5ad3777`) | `docs/change_requests/INT2.md`; `docs/evidence/validation/diagnosis/inner_batch_rebuild_20261127.txt` |
| A refused promotion rolled back psi and the policy head (LRN-6) | INT2 (`07be9ba`) | `docs/change_requests/LRN.md` LRN-6; `docs/INTEGRATION.md` §6 |
| The verified bootstrap term saturated at the bound (INT2-6); the live policy step was too small (INT2-7) | LRN (bound 4 and an unbounded goal term; trust region, target KL 0.05) | `docs/MATH_NOTES.md` §13; not re-measured on the real configuration |
| G_nn D1 / D2 / D4 / D5, G_pr D1 / D2 / D5 / D7 / D11, G_cl M3 / M4 / m4, C-9 | owners' fix rounds | `docs/MATH_NOTES.md` §13 (table with code locations); `docs/A/DESIGN.md`, `docs/C/FIXES.md`, `docs/D/FIXES.md` |
| 25 G-rt runtime defects (1 critical, 11 major, 13 minor) | F | `docs/F/FIXES.md`; `docs/status/F.json` |
| ES test `test_es_publishing.py` asserted 16 tools after 18 Discord tools were added | fixed (line 229 now counts the Discord tools) | `tests/harness/skills/test_es_publishing.py`; whole suite at `1039a6f`: 1468 tests, 0 failed (`docs/evidence/test_results/INT2_full.xml`) |
| The disk filled on 2026-11-27 (~10:16 UTC) from evaluation checkpoints with optimizer state | `cam_live.storage` (DiskGuard, retention) + `tools/prune_runs.py` | ledger session 3; `docs/INTEGRATION.md` §5a |
| A token-shaped test fixture blocked the push (GitHub push protection) | history rewrite `d9f8dde..ebf1678` | `docs/evidence/commit_map.tsv`; ledger session 2 |

---

## 2026-11-27 23:45 UTC — state changes after INT3B `5ae3b59` (DOC)

| Id | Change | Evidence | State |
|---|---|---|---|
| KF-D07 | The runtime process now holds the shared GPU lock (`runtime.gpu_lock: auto`) from before the agent is built until it exits. `runtime.gpu_hold_s` can bound a hold: checkpoint, pause and exit, then `cam-live resume`. The default config sets neither key, so a live run holds the lock for its whole duration and the other tenant's heavy jobs wait. Not re-checked: whether the learner's uid (`camlearn`) can open the lock file in separated mode. | `src/cam_live/runtime/loop_service.py` module docstring, `needs_gpu_lock`; `docs/REPRODUCE.md` §14 | changed: open as an operator coordination item |
| KF-E07 | Outcomes pending when an objective ends are now settled: measured for a bounded window, with nothing proposed or executed and never past the deadline. Otherwise they are carried in the persisted queue. | commit `5ae3b59`; `tests/runtime/test_int3b_settle.py` | fixed per INT3B (DOC did not run the tests) |
| KF-E12 | `cam-live plan-report` exists (`--help` exit 0). The other three missing commands are unchanged. | `docs/REPRODUCE.md` §14 | partly fixed |
