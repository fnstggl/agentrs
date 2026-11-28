# G2-integration — independent re-verification of the integrated agent at the code freeze

Verifier: G2, lens `g2integration`. I wrote none of the code under test, edited no source file and no other
agent's test. **Code verified: commit `a5ab7720f169c1bc23cc442be96a31310dfa29eb`** (the freeze). HEAD moved several
times while I worked (G2 commits and other agents' uncommitted source edits), so every run used a read-only export
`git archive a5ab772 | tar -x -C <scratch>/src` with `PYTHONPATH=<scratch>/src/cam_live/src` (the venv's editable
install is shadowed); `<scratch>` = `/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2INTEGRATION`.
`CAM_LIVE_HOME` was always inside `<scratch>`. Nothing contacted an external service; no live run was started; nothing
was pushed (the push script was exercised only as a modified copy against throwaway local repositories). The machine
was heavily loaded throughout (load average 20-62, other tenants' GPU jobs queued on the same lock).

Scope (28 requirements): NA-06 CL-09 AE-03 AE-05 AE-08 OB-01 OB-02 OB-03 OB-04 CU-22 CU-30 LX-11 SV-11 IN-01 IN-04
IN-07 IN-09 OP-01 OP-02 OP-03 OP-05 OP-06 OP-08 DL-02 DL-04 DL-20 IN-05 IN-06. Claims read: `docs/status/M-int.json`
(14 of them), `docs/status/MAN.json` (IN-04, IN-09, OP-08), `docs/status/DOC.json` (DL-02, DL-04, DL-20, IN-05,
IN-06); OB-01..04, OP-03 and OP-06 are claimed by no status file and are judged on code and tests. Earlier reports
re-checked: `G_rt.md` (D1-D25) and `G_pr.md` (D1-D12) by re-running their probes on the freeze.

Verdicts: `verdicts_g2integration.json`. Probes: `repro/g2integration/` (each prints `[HOLDS ]` / `[BROKEN]`, exit 1
on any broken property). The documentation checks (DL-02, DL-04, DL-20, IN-05, IN-06) were carried out by a helper
agent of this lens under my brief (probes `repro/g2integration/docs/d01..d05`); I re-ran `d04` and spot-checked four
of its findings against the code myself (all four confirmed).

**Result.** 9 verified, 18 partial, 1 failed (LX-11), 0 unverified. The end-to-end pathway holds on the real configuration in staging (observation -> latent -> prediction -> action -> consequence -> R_psi internal reward -> neural update -> changed behaviour on a held-out context, fully carried by the weights), but no reward meta-update applied in my two real-configuration runs; the staging validation does not pass (10/16 and 13/16 here; `approvals_exercised` does not hold with the validation's own defaults); 15 defects, 5 of them major: the push secret scan misses credentials that were added and removed inside the pushed range and anything in binary / compressed files (G2I-1, G2I-2), `approvals_exercised` is practically unreachable with the validation's defaults (G2I-3), the runtime records resource-limit refusals as missing permissions (G2I-14), and an operator-approved action whose re-authorization is refused is reported to the learner as rejected by the operator (G2I-15); the documentation deliverables are accurate in their numbers but stale at the freeze in many statements.

## 1. What was executed

All commands from the export `<scratch>/src` (or from `/workspace/camlive` for probes, with `PYTHONPATH` pointing at
the export). `P=cam_live/docs/verification/repro/g2integration`.
Recorded outputs: `$P/out/runA_staging_20261128T030744Z.json`, `$P/out/runB_staging_20261128T033952Z.json` (the two
validation reports), `$P/out/p08_approval_trace_runB.txt`, `$P/out/probe_results_a5ab772.txt` (final run of p01-p07).

| What | Command | Result |
|---|---|---|
| Acceptance tests of the scope | `pytest cam_live/tests/integration cam_live/tests/storage cam_live/tests/runtime/test_rt_logs_journal.py cam_live/tests/runtime/test_rt_dashboard.py cam_live/tests/runtime/test_rt_doctor_cli.py cam_live/tests/runtime/test_int2_storage_drain.py cam_live/tests/harness/skills/test_es_failures.py cam_live/tests/harness/skills/test_es_modality.py -q --rootdir=cam_live` | **138 passed, 1 skipped** (browser), 1403 s |
| The skipped browser test (CU-30) | `pytest cam_live/tests/integration/test_int2_grounding.py -m browser -q --rootdir=cam_live` | 1 passed, 108 s |
| Completion checker tests (IN-04, IN-09, OP-08) | `pytest cam_live/tests/test_verify_completion.py -q --rootdir=cam_live` | 27 passed, 24 s |
| Completion checker itself (read-only, worktree) | `cd cam_live && ../.venv/bin/python -m cam_live.verify_completion --json <scratch>/logs/completion.json --quiet` | runs; 198 requirements, 59 complete, 0 failed; levels 1-5 not reached |
| **Staging validation, run A** (the operator command, defaults) | `cam-live validate --mode staging --real-model --out <scratch>/staging/run1` (lock taken through the queued lease) | **10 of 16 checks hold, not passed** (section 3); lock 1095 s |
| **Staging validation, run B** (INT3B run-1 flags) | `cam-live validate --mode staging --real-model --decisions 40 --budget-s 840 --meta-min-independent 3 --computer-share 0.2 --out <scratch>/staging/run2` | **13 of 16 checks hold, not passed** (section 3); lock 754 s, peak 18.8 GB |
| Push secret scan (OP-08) | `.venv/bin/python $P/p01_push_scan.py --script <scratch>/src/cam_live/tools/push_milestone.sh` | **4 broken of 6** (G2I-1, G2I-2) |
| Per-commit scan of the real history 3968583..a5ab772 with the script's own pattern, outside its allow list; pattern scan of the two committed `.json.gz` files | shell loop over `git show` (no push, no secrets read) | 0 hits; 0 hits |
| Operator requests (OP-06) | `PYTHONPATH=<copy> .venv/bin/python $P/p02_operator_requests.py` | 14 of 14 hold |
| Storage guard / retention / monitoring (OP-01..03) | `PYTHONPATH=<copy> .venv/bin/python $P/p03_storage.py` | **5 broken of 11** (G2I-4..G2I-7) |
| Obstacles (OB-01..04, AE-08) | `PYTHONPATH=<copy> .venv/bin/python $P/p04_obstacles.py` | **3 broken of 12** (G2I-8, G2I-9) |
| What became of the approvals of run B (LX-11) | `.venv/bin/python -I $P/p08_approval_trace.py <scratch>/staging/run2` | **2 broken** (G2I-15) |
| Failure class of a limit refusal recorded by the runtime (OB-01) | `PYTHONPATH=<copy> .venv/bin/python $P/p07_limit_refusal_class.py` | **2 broken of 4** (G2I-14) |
| Standalone at run time (IN-09) | `PYTHONPATH=<copy> .venv/bin/python $P/p05_standalone.py` | **2 broken of 5** (G2I-10) |
| Discord observations into the pipeline, staging (OP-05) | `PYTHONPATH=<copy>/cam_live/src:<copy>/cam_live/tests/harness/skills .venv/bin/python $P/p06_discord_observations.py` | 6 of 6 hold |
| Documentation (DL-02, DL-04, DL-20, IN-05, IN-06) | `$P/docs/d01_citations.py`, `d02_operator_resources.py`, `d03_reproduce_commands.py`, `d04_known_failures.py`, `d05_model_math_claims.py` (usage line in each) | 187 citations resolve; parameter counts and `math_check.py` reproduce exactly; 66/66 `--help` exit 0; 5 of 21 executed claims wrong; register 18 of 19 checks broken (section 5) |
| G-rt's probes on the freeze | `G_RT_SCRATCH=<scratch>/rt .venv/bin/python <copy>/cam_live/docs/verification/repro/rt/<p>.py` for p01-p10, p13-p15, p02 | 745 hold; 4 broken, all outdated expectations (section 6); p10 aborted at its own 240 s timeout |
| G-pr's CPU probes on the freeze | `CUDA_VISIBLE_DEVICES= .venv/bin/python <copy>/cam_live/docs/verification/repro/pr/<p>.py` for integrity, metalearner_path, extras, predict_numerics, decisions, metagrad_small | integrity 21/21, metalearner_path 9/11 (2 FAIL, section 6), extras 4/4, predict_numerics 31/31, decisions 28/28, metagrad_small 40/40 |

Not run: G-rt `p11_separated.py` / `p12_numbers.py` (need a run directory reachable by the service uids, resp. fsync
timing on the repository disk; separation is G2-harness's scope), G-pr `probe_real_model.py`,
`probe_metagrad_real.py`, `probe_labworld_reduced.py` (GPU / long), any `-m gpu` / `-m slow` / `-m privileged` test.

## 2. End-to-end pathway on the real configuration (IN-07, NA-06, CU-22, CL-09)

Run A (`staging_20261128T030744Z.json` in `<scratch>/staging/run1`; Qwen/Qwen3-4B frozen bf16 + LoRA + bridge + 50m
core, 74,672,079 trainable / 4,022,468,096 frozen parameters, exactly the counts in `docs/evidence/DOC/param_counts.json`)
shows every link of the pathway on the staging services, at a very small number of decisions (1 computer + 3
LabWorld, because of load, section 3):

* **observation → latent update**: frequent updates record `prediction_change.latent_b` 0.62 / 0.42 / 0.25 and
  `latent_z` 0.60 / 0.52 / 0.31;
* **prediction before consequence**: 21 predictions resolved, all after registration; 3 delayed outcomes linked to
  their decisions;
* **action → consequence**: 4 executed actions (browser.screenshot on the staging store site; lab.act x2, lab.probe),
  0 grant refusals; 15 backbone-written, 4 grounded, 21 schema-walker candidates offered;
* **internal reward**: 4 of 4 records rewarded by R_psi (values 1.1e-3, 1.7e-3, -1.9e-4, -4.6e-4; bootstrap mix 1.0);
* **credit**: the default estimator is `simple` (`agent.py:2016`); the evidence-based estimator selection is never
  attached in the agent (`agent.py:716` builds an `EstimatorSelector`, nothing calls `attach_experiments` /
  `evaluate_experiments`; G2L-6 / DX-1), so credit is RUDDER/COCOA-free by construction;
* **neural update**: 4 frequent updates with non-zero deltas (experiential 1.05 / 1.47 / 0.33 / 1.27, heads 0.23 /
  0.38 / 0.36 / 0.35), one of them the R_psi-driven policy-improvement update (trigger "3 newly rewarded decisions",
  policy KL 0.062, largest probability change 0.11); the one slow-adapter step was deferred ("no provider has
  held-out validation records"), so **adapter weights did not change** in this run; no reward meta-update applied
  (2 skipped for insufficient independent outcomes, 1 no policy update yet);
* **changed behaviour, carried by the weights**: on the held-out LabWorld context (seed 9999, never acted in) the
  policy distribution moved by TV 0.128; restoring all trainable groups (frozen backbone, store, fast state and
  registries untouched) gives TV 0.0; restoring only the experiential group leaves TV 0.035 from before / 0.100 from
  after.

The same chain is pinned at debug scale by `tests/integration/test_agent_runtime.py` (all 11 pass).

Run B (`staging_20261128T033952Z.json` in `<scratch>/staging/run2`; 5 computer + 28 LabWorld decisions over four
lifetimes, lock 754 s at load average ~20) shows the same chain at a larger size: 172 predictions resolved after
registration, 62 delayed-outcome links, 33 of 33 records with internal rewards, 68 frequent updates all with non-zero
experiential deltas, 4 R_psi policy-improvement updates (policy KL 0.057 / 0.057 / 0.041 / 0.045), 18 slow-adapter
steps of which 5 changed the adapters (group delta norm 0.71), held-out TV 0.276 -> 0.0 after restoring the weights
(experiential only: 0.056 from before / 0.239 from after), three of four lifetimes achieved by the verifier's signed
verdict, interruption + restart from disk in lifetime 2 resumed. **In neither run did a reward meta-update apply**
(run B: 38 skipped for fewer than 3 independent verified groups, 7 no supported episode, 2 no policy update yet), so
the outer loop (verified outcomes -> psi) is shown only by INT3B's run 2 (2 applied, six lifetimes) and the tests.

## 3. Staging validation (LX-11) — judged check by check

Every check of `validation.checks` in my two runs and INT3B's two (evidence files of the commit):

| check | run A (mine, defaults) | run B (mine, INT3B run-1 flags) | INT3B run 1 | INT3B run 2 (six lifetimes, not via the CLI) |
|---|---|---|---|---|
| adapter_weights_changed | **fails** | holds | holds | holds |
| approvals_exercised | **fails** | **fails** | **fails** | **fails** |
| behaviour_changed_on_held_out_context | holds | holds | holds | holds |
| computer_environment_decisions | holds | holds | holds | holds |
| delayed_outcomes_linked | holds | holds | holds | holds |
| experiential_weights_changed | holds | holds | holds | holds |
| internal_rewards | holds | holds | holds | holds |
| lab_lifetimes_verified | **fails** | holds | holds | holds |
| logging | holds | holds | holds | holds |
| meta_step_changes_policy | **fails** | **fails** | **fails** | holds |
| meta_update_applied | **fails** | **fails** | **fails** | holds |
| no_grant_refusals | holds | holds | holds | holds |
| predictions_before_outcomes | holds | holds | holds | holds |
| recovery_exercised | **fails** | holds | holds | holds |
| resource_limit_exercised | holds | holds | holds | holds |
| weights_explain_the_change | holds | holds | holds | holds |
| **checks holding** | **10 / 16** | **13 / 16** | **13 / 16** | **15 / 16** |
| GPU lock held (s) / peak GB | 1095 / 9.5 | 754 / 18.8 | 546 / 18.5 | 809 / 18.7 |
| decisions computer + LabWorld | 1 + 3 | 5 + 28 | 4 + 24 | 10 + 32 |


What fails, and why:

* **approvals_exercised fails by construction of the validation's own parameters** (G2I-3). The check needs an
  approved action executed *and* a rejected one (`validation.py:854-858`); the validation rejects only every third
  approval request (`phase_computer(reject_every=3)`, `validation.py:398`), while the computer phase is ended by a
  compute limit of 8 s per decision of its budget (`validation.py:696-700`, 80 s for 10 decisions in run A, 64 s for
  8 in run B) that one to five decisions already use up on the real configuration (run A: 131 s for the first
  decision; run B: 69.6 s after five). Three approval requests plus an executed approval do not fit in practice: no
  run made more than one approval request (INT3B run 1: 0, run 2: 1; mine: 0 and 1), and the check failed in all
  four. In run B the one approval request was approved by the validation's operator and then
  reported to the learner as *rejected by the operator* without being executed (G2I-15, `p08_approval_trace.py`);
  one of the task's two external actions had been spent by the validation's own `reset_browser` navigation.
  The two checks `resource_limit_exercised` (needs the limit to end the phase) and
  `approvals_exercised` (needs several approvals before that) are in tension in the shipped configuration.
* **The GPU hold is not bounded by `budget_s`** (G2I-11). Run A held the shared lock **1095 s** with `budget_s` 780
  (lab phase budget 574 s, spent 845 s on three decisions): the time budget is checked only between runtime steps
  (`validation.py:479`, `:500`) and one LabWorld step took ~280 s at load average ~50. That exceeds the 15-minute
  hold rule of the shared GPU; INT3B measured an over-run of the same kind (1234 s) in the planning rehearsal.
* `resource_limit_exercised` holds in both runs through the compute limit; the external-action limit refusal of run B
  (`browser.click`) is not counted by the check because the runtime records it as `missing_permission` (G2I-14).
* Under load the default command reaches neither a second LabWorld lifetime (so no restart: `recovery_exercised`
  fails; no verified lifetime: `lab_lifetimes_verified` fails) nor a meta-update. INT3B run 2's 15/16 needed six
  lifetimes and `meta_min_independent` 3, a configuration **not reachable through `cam-live validate`** (it has no
  option for `lab_seeds` / `lab_target` / `compute_limit_s`; `runtime/cli.py:800-815`).
* The M-int / INT3B numbers of runs 1 and 2 reproduce from their evidence files (15/16 and 13/16; meta-gradient norms
  40.7 / 15.4; psi deltas 0.097 / 0.062; meta-step TV 0.221 / 0.011, max 0.242 / 0.019). One label is wrong: the "5
  ESS too small" skips of both runs are recorded as `no_supported_episode: 5` in the evidence.

## 4. Defects

Severity as in the earlier reports (major: a stated property does not hold in a way that matters; minor: narrow).
Every reproduction runs from `/workspace/camlive` with `PYTHONPATH=<copy>/cam_live/src` where Python is involved.

| id | sev. | where (at a5ab772) | defect | reproduction |
|---|---|---|---|---|
| G2I-1 | major | `tools/push_milestone.sh:2-3` (claim), `:19`, `:25` | The secret scan reads only the net diff `git diff BASE..SHA`, but the push uploads every commit: a credential added in one unpushed commit and removed in a later one is pushed in the history (both the exact check against this machine's credentials and the pattern check). | `p01_push_scan.py` checks B, E |
| G2I-2 | major | `tools/push_milestone.sh:19`, `:25`, `:28-29` | Binary / compressed files are not scanned (`git diff` prints "Binary files differ"; added binaries are only listed). A credential inside a `.json.gz` evidence file is pushed. The two committed `.json.gz` plan reports are also on the allow list (`tools/secret_scan_allow.txt`). Latent: my per-commit scan of 3968583..a5ab772 and of the decompressed `.gz` files found nothing. | `p01_push_scan.py` checks C, F |
| G2I-3 | major | `validation.py:398` (`reject_every=3`), `:696-700` (compute limit 8 s per decision), `:854-858` | `approvals_exercised` does not hold with the validation's own defaults on the real configuration: it needs a rejection, which comes only at the third approval request, and an executed approval, all before a compute limit of 8 s per budgeted decision ends the computer phase (1-5 decisions); no run made more than one approval request; failed in all four real-configuration runs since INT3B. | run A / run B evidence, `phase_computer.approvals_resolved` |
| G2I-14 | major | `runtime/loop.py:1294-1298` | Every refusal by the authorization service becomes `FailureClass.MISSING_PERMISSION` (`_synthetic(..., DENIED, MISSING_PERMISSION, detail)`); the harness's `refusal_class` (`harness/failures.py:392-400`), which makes a refusal whose reasons are all `resource: ...` a `RESOURCE_LIMIT`, is not used. A budget / rate-window refusal is therefore recorded, learned from and categorised as a *permission* obstacle (a hard class), not a resource one. Seen on the real configuration in run B (`browser.click` refused with `resource: max_external_actions exhausted (used 2 incl. reserved, requested 1, limit 2)` -> `missing_permission`, obstacle category `permission`); it also hides such refusals from the validation's `resource_limit_refusals` count. | `p07_limit_refusal_class.py` checks 2, 3 |
| G2I-15 | major | `runtime/loop.py:975-994`, `:1033-1036`; `harness/auth/limits.py:71-77` | An approval the operator resolves outside the runtime (operator tools; the validation) is re-authorized by the runtime; when that re-authorization is refused, the runtime marks the approval *rejected* and tells the learner "the operator rejected approval <id>" — although the operator approved it. In run B the refusal was caused by the approval's own grant: `resolve_approval` had issued a grant reserving the one remaining external action, and the re-authorization counted that reservation plus a new one ("used 2 incl. reserved, requested 1, limit 2"); the approved action was never executed and the reserved grant stayed `issued`. (The other external action of the budget of 2 was spent by the validation's own `reset_browser` navigation under the task, `validation.py:212-222`.) | `p08_approval_trace.py <scratch>/staging/run2` (2 broken) |
| G2I-4 | minor | `agent.py:2290-2298` (`export_weights`), `runtime/cli.py:680-708` (`cmd_export_checkpoint`) | Weights-only exports (running run and stopped run) write below the free-space minimum; no `DiskGuard.check`. | `p03_storage.py` checks 4, 5 |
| G2I-5 | minor | `tools/prune_runs.py:84-87`, `:106`, `:111` | A symbolic link to a checkpoint-shaped directory inside the pruned tree is taken as a checkpoint; its `optim/` is deleted *outside* the tree (rmtree through the link). Root running the operator tool over a learner-writable run tree deletes `optim/` directories anywhere. | `p03_storage.py` check 9 |
| G2I-6 | minor | `credit/checkpoint.py:980-995`, `agent.py:813` | Learner promotions keep full optimizer state in every kept checkpoint (keep_last 2 + promoted 1), not only in the newest (by design for rollback, but contrary to the OP-02 wording; M-int's limitation states the counts only). | `p03_storage.py` check 10 |
| G2I-7 | minor | `runtime/doctor.py:100`, `runtime/dashboard.py:342-345` | The doctor's `disk` check uses a fixed 20 GB, not `storage.min_free_gb`: with a configured minimum above the free space it reports ok while the runtime refuses every checkpoint. The dashboard reports free GB without the guard's level. | `p03_storage.py` check 11 |
| G2I-8 | minor | `runtime/obstacles.py:23` vs `harness/failures.py:40` | `FailureClass.UNKNOWN` gets category `""` in obstacle records (harness taxonomy: "informational"): an obstacle of unknown class is not classified into one of the five categories. | `p04_obstacles.py` checks 1, 2 |
| G2I-9 | minor | `runtime/obstacles.py:85-98` | `on_learning` appends every learning update after an obstacle to every open obstacle; "resulting neural learning updates" is a temporal association, not evidence that the update used the obstacle's experience (an unrelated update is linked). | `p04_obstacles.py` check 7 |
| G2I-10 | minor | `.venv/lib/python3.12/site-packages/_shared_base.pth`, `config.py:26` | At run time CAM Live is not standalone of AgentRS / CAM Unified's directories: 1126 of 1233 third-party modules load from `/workspace/agentrs/.venv`, and the default GPU lock is `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock`. No CAM Unified / Crucible / AgentRS module is imported (176 cam_live modules imported). | `p05_standalone.py` checks 3, 4 |
| G2I-11 | minor | `validation.py:479`, `:500` | `budget_s` bounds the GPU hold only between runtime steps; run A held the lock 1095 s (> 900 s rule) with `budget_s` 780. | run A `lock_seconds` |
| G2I-12 | minor | `docs/COMPLETION.md` "Durability" (at a5ab772) | The committed durability record is from 799f032 (25 unpushed commits then); the local remote-tracking ref `origin/feat/cam-live` has since moved (87e9cf1, later 702054b) and contains a5ab772. Regenerated only by `--merge-status`. | `git -C /workspace/camlive rev-parse origin/feat/cam-live` |
| G2I-13 | minor | docs (section 5) | Documentation deliverables contain statements false at a5ab772 (stale since INT3A / INT3B / HX, and four wrong at their own commit). | `$P/docs/d0*.py` |

Notes that are not counted as defects: obstacle explanations are a Dirichlet(1) posterior over the 11 failure-class
labels counted per operation (`agent.py:2541-2552`), independent of the failure's content: a refusal by the
authorization service gets p = 0.167 for `missing_permission` (2/12). Learner promotion at the end of the first
objective is refused for "no retention metric supplied" (run A), so promotion never succeeds on a first objective.

## 5. Documentation deliverables (DL-02, DL-04, DL-20, IN-05, IN-06)

All five documents state the commit they describe (d3ce519 / 55a0f4c for model and mathematics, 204e9d4 / 5ae3b59
for resources, reproduction, failures). What holds: parameter counts re-measured on CPU at a5ab772 equal
`param_counts.json` in every sub-count (and equal run A's `parameters`); 187 `file:Symbol` citations exist; seven
sampled equations match the code; `math_check.py` output is identical to `math_check.txt`; the offline preflight
reproduces exactly (planning task exit 3 with 9 live items, `discord_100.yaml` exit 3 with 25); 66 of 66 documented
`--help` commands exit 0; dependencies match `pyproject.toml` and the installed versions; the register is
append-only. What is false at a5ab772 (each confirmed by a probe of `$P/docs/`; four spot-checked by me):

* OPERATOR_RESOURCES.md:32-35, :70 and KNOWN_FAILURES.md:32 (KF-D01): "the code has no override" for
  `persistent_storage` — `runtime/preflight.py:300` `ACCEPTABLE_RISKS` and `--accept-ephemeral-storage`
  (`runtime/cli.py`) exist; REPRODUCE.md §9.1 (:312-313) gives the planning launch without that flag, which blocks
  on this machine.
* OPERATOR_RESOURCES.md:156-158, REPRODUCE.md:345-347, :445-450, KNOWN_FAILURES.md:38, :143: no GPU-hold keys in the
  default configuration — `configs/cam_live.yaml` sets `gpu_lock: auto`, `gpu_hold_s: 840`, `gpu_yield_s: 15`.
* OPERATOR_RESOURCES.md:103-108 / REPRODUCE.md:318-321: the by-hand launch sequence fails at step 2 (`install ...
  R/auth/keys/operator.pub`: directory absent; `tools/launch_live.sh:46` creates it first).
* REPRODUCE.md:456-457: the staging planning task's offline preflight "exit 3" — it exits 0 (also at 5ae3b59).
* REPRODUCE.md:156-159, :171-172, :428, KF-E10: the suite cannot be collected in one process — 1593 collected,
  exit 0 (`pyproject.toml` `--import-mode=importlib`); KF-E11 `validate --debug` builds the real model — fixed
  (`validation.py:336-342`); result values at :154 (runtime tests 245, not 215), :188 (doctor 52 checks), :411.
* MODEL_ARCHITECTURE.md:284, :302, :312-314: `set_objective` calls `MetaLearner.new_lifetime` / resets R_psi's
  context — it calls `meta.new_objective()` (`agent.py:913`, KF-E05 fix); §7 omits that the policy update consumes
  stored credit (`agent.py:741`). MODEL_ARCHITECTURE.md:249-250, :329-331 and MATH_NOTES.md:609-610 describe the
  estimator chosen from randomised experiments, which the agent never wires (G2L-6).
* MATH_NOTES.md :359-360, :666, :669, :693-694 (info_gain / epistemic_after / learning_progress have no producer — they
  do now), :413-421 (L_inner without the credited goal return, `reward/inner.py:285`), :611-612, :677-679 ("no
  training loss reads the stored credit"), :623-626 (slow guard now per provider and deferring), :116-121 (Poisson
  mode search, G2L-1).
* KNOWN_FAILURES.md: nine rows still "open" are fixed (KF-D01, D07, E01, E02, E03, E05, E10, E11, E12); KF-R02 says
  the validation was "not re-run" while two later evidence files exist (13/16, 15/16); missing: the staging
  validation not passing (M-int not_done), INT3B-4 (prompt over the window in 12/12), INT3B-5, CAM wrote no notes
  in the rehearsal, the GPU-hold over-run, G2L-6 / DX-1, INT3A-5, G2-harness R2-C1 / R3-M1 / R3-m1 / R3-m2.

The spec's items for each deliverable are present (architecture, equations with code locations, operator list,
commands, register); the defects are of currency and accuracy, hence partial for all five.

## 6. Earlier reports: are their defects fixed? (G_rt.md, G_pr.md)

**G_rt D1-D25** — re-run of its probes on the freeze: p01 36/36, p02 477/477 (60 kill -9 scenarios), p03 15/15,
p04 11/11, p06 38/38, p07 62/62, p13 8/8, p14 11/11, p15 3/3 hold. Broken checks are outdated expectations, not
defects: p05 1a and p09 1b count 17 REQUIRED items, the operator has since filled one (16 unset; p09 lists every one
of them: `not_listed` empty); p08 6 flags `<script src="/app.js">` (same origin) and p08 8 posts an approval without
the CSRF token and operator credential the hardened dashboard now requires (refused, as it should be); p08's D1
checks 12 and 13 hold (no planted markup executed, nothing approved). p10 stopped at check 4a because `validate`
now really runs (the probe's 240 s timeout); its checks 0a-3c hold (9). So D1-D18 and D22-D24 hold as fixed on the
probes' evidence; D19 and D21 (CLI) were not re-checked past p10's abort, D20 (separated resume) not run, D25 (two
comments in `tasks/discord_100.yaml`) not re-checked.
"Only confirmable with the real agent" (G_rt): `drain_experiments` and `explain_obstacle` exist and are called by the
runtime; `init`, weights-only `export-checkpoint`, `validate` run (export: G2I-4); decision / learning-update records
carry rationale, losses, deltas and checkpoint ids (run A).

**G_pr D1-D12** — D2, D3 and D6 (integrity 21/21), D4 (extras 4/4: never more goals than the 2-decision budget)
and D5 (predict_numerics 31/31: the clustered null now excludes 0 in 7 % of runs) hold as fixed; decisions 28/28 and
metagrad_small 40/40 unchanged. `probe_metalearner_path` 9/11: its D1 check found no episode with measured rows
(inconclusive — D1 not re-checked); "applied meta-gradient == my independent recomputation" fails with relative
error 0.23 — the probe re-implements the earlier group-leave-out baseline and the D1 fix changed the baseline groups
(`reward/metalearner.py:615`, `:812`), so this is most likely the probe; not judged a defect here (MR-* are outside
my scope) but left for the reward lens; "R_psi detached gives gradient 0" is vacuous (no applied update in that
variant). D7-D12 not re-checked. G_pr's integration remark "Neither package is called by the runtime for rewards/meta-learning" no longer holds:
run A computes internal rewards through `MetaLearner.internal_rewards` and calls `policy_update` / `meta_update`
from the runtime loop.

## 7. Per requirement: verdict and basis

Counts: **9 verified, 18 partial, 1 failed, 0 unverified** (28).

* **NA-06 — verified.** test_behaviour_changes_and_lives_in_the_weights passes (debug). Real configuration (run A): the policy distribution on a held-out LabWorld context moved TV 0.128 after 4 decisions of learning; restoring all trainable groups gives TV 0.0 (store, fast state, registries untouched); restoring only the experiential group leaves TV 0.035 / 0.100. The chosen actions come from the trainable policy (rule thompson_policy, 3 of 4) and world predictions of the experiential core. Run B: held-out TV 0.276 after 33 decisions, 0.0 after restoring the trainable groups; experiential only 0.056 / 0.239.
* **CL-09 — verified.** set_objective (agent.py:893-933): weights, reward version and learned prior kept, fast state reset to the learned prior, meta.new_objective (lifetime context kept, KF-E05). test_new_objective_keeps_weights_and_resets_fast_beliefs passes (group hashes and reward version equal, fast state == prior). Run A: one agent, staging computer objective then LabWorld objective, learning continued. Transfer not measured (not claimed).
* **AE-03 — partial.** Self-directed randomised experiments are registered and run in the loop (run A: 1 of 4 decisions by rule 'randomized'; experiments.jsonl written); fails: no evidence that self-directed learning improves anything - usefulness not evaluated (as claimed).
* **AE-05 — verified.** One agent lifetime spans several objectives (tests; run A: staging computer task then LabWorld, learning continued on the second). Supports = plumbing; performance on an unfamiliar objective is not established (not claimed).
* **AE-08 — partial.** Obstacle records are written and linked by the runtime (runtime/obstacles.py) with explain_obstacle explanations (p04 3-6 hold). Fails: explanations are only a Dirichlet(1) over failure-class labels counted per operation (a policy refusal gets p=0.167); strategy changes are a flag, not measured; resource-limit refusals are recorded as permission obstacles (G2I-14, p07); learning updates are linked temporally to every open obstacle (G2I-9, p04 check 7).
* **OB-01 — partial.** Failure observed and characterised (class, detail) in an ObstacleRecord with probabilistic explanations (p04 3, 4 hold; test_obstacle_records_link... and test_every_failure_class_is_reachable_and_categorised pass). Fails: (1) the runtime records every authorization-service refusal as missing_permission (runtime/loop.py:1294-1298, harness refusal_class unused), so a resource-limit refusal ('resource: max_external_actions exhausted ...') becomes a permission obstacle - seen on the real configuration in run B and reproduced by p07 checks 2, 3 (G2I-14); (2) FailureClass.UNKNOWN gets category '' in obstacle records (runtime/obstacles.py:23; harness/failures.py:40 says informational) - not one of the five categories (G2I-8, p04 checks 1, 2). Explanations do not use the failure's content.
* **OB-02 — verified.** The next decision's candidates (with their predictions and costs in the decision record) and the chosen alternative are recorded on the obstacle, the alternative is authorized and executed by the ordinary loop, its result recorded (p04 checks 5, 6; test_obstacle_records_link_failure_alternatives_and_learning passes). No recovery logic: the learner decides.
* **OB-03 — partial.** A failed action is an experience record like any other: fast beliefs, R_psi internal rewards and frequent updates use it (run A: records rewarded and learned from). Fails: the obstacle record's learning_update_ids are every later update (runtime/obstacles.py:85-98), not updates shown to use the obstacle's experience (G2I-9, p04 check 7).
* **OB-04 — verified.** No failure-conditioned recovery branch in the decision path (p04 check 10: the only failure-class branch in agent/runtime/predict/credit/reward abandons a self-generated goal); RetryPolicy retries only three transient classes, bounded (<=5), and refuses any rule for missing_permission / platform_restriction (p04 8, 9); a refusal never switches modality (test_a_refusal_never_causes_a_modality_switch x4, test_hard_classes_are_never_retried_and_cannot_be_configured pass).
* **CU-22 — partial.** Neural learning from interaction experience happens on the real configuration: run A 4 frequent updates with non-zero experiential deltas (1.05/1.47/0.33/1.27), one of them the R_psi policy-improvement update. Adapter weights: unchanged in run A (its one slow step was deferred: no held-out validation records), changed in run B by 5 of 18 slow steps. Fails: no improvement with repetition shown at the real configuration; no comparison with the skill library (as claimed). Run B (INT3B run-1 flags, 5 computer + 28 LabWorld decisions, 4 lifetimes): 13 of 16 hold (approvals_exercised, meta_update_applied, meta_step_changes_policy fail); held-out TV 0.276 -> 0.0 after restoring the weights; 68 frequent updates, 4 policy-improvement updates (KL 0.041-0.057), 5 of 18 slow-adapter steps changed the adapters; no reward meta-update applied (38 skips: fewer than 3 independent verified groups).
* **CU-30 — partial.** Grounded actions execute on the staging sites (test_grounded_actions_execute_on_the_staging_sites -m browser: 1 passed, 108 s; run A executed browser.screenshot on the staging store site, 80 tools from every executor). Fails: task success on an unfamiliar interface is not established (run A computer objective: 1 decision, verified status pending; as claimed).
* **LX-11 — failed.** Staging validation does not pass. Run A (default command): 10 of 16 hold; fail approvals_exercised, adapter_weights_changed, lab_lifetimes_verified, meta_update_applied, meta_step_changes_policy, recovery_exercised (under load only 1 computer + 3 LabWorld decisions). Run B (INT3B run-1 flags, 5 computer + 28 LabWorld decisions, 4 lifetimes): 13 of 16 hold (approvals_exercised, meta_update_applied, meta_step_changes_policy fail); held-out TV 0.276 -> 0.0 after restoring the weights; 68 frequent updates, 4 policy-improvement updates (KL 0.041-0.057), 5 of 18 slow-adapter steps changed the adapters; no reward meta-update applied (38 skips: fewer than 3 independent verified groups). In run B the one approval was approved by the validation operator, then reported to the learner as rejected by the operator and never executed (G2I-15, p08); an external-action limit refusal was recorded as missing_permission (G2I-14). approvals_exercised does not hold with the validation's own defaults (G2I-3: a rejection comes only at the 3rd approval request, the compute limit of 8 s per budgeted decision ends the phase after 1-5 real-configuration decisions; no run made more than one approval request); the lock was held 1095 s with budget_s 780 (G2I-11). INT3B run 2's 15/16 configuration (six lifetimes) is not reachable through cam-live validate. M-int's numbers of its runs reproduce from the evidence files.
* **SV-11 — verified.** No live or real-world result is reported anywhere in M-int.json / INT3B.md; the staging numbers I checked reproduce from the cited evidence files (15/16, 13/16, meta-gradient norms 40.7/15.4, psi deltas 0.097/0.062, TV 0.221/0.011). One label is wrong: the '5 ESS too small' skips are 'no_supported_episode' in the evidence.
* **IN-01 — partial.** The runtime calls the integrated agent at every step (test_every_runtime_step_calls_the_agent passes); the staging computer environment registers every executor (all_executors: 80 tools), cam-live run composes 'all'; promotion is called (run A: refused, 'no retention metric supplied'). Fails: the credit estimator selection is not wired (agent.py:716 EstimatorSelector, never attach_experiments / evaluate_experiments: default stays 'simple', G2L-6); M-int's limitation text (sandbox/visual/skills not registered, promotion not called) is outdated.
* **IN-04 — verified.** tests/test_verify_completion.py 27 passed on the export; the checker runs read-only on the worktree (198 requirements; groups and levels computed; stale evidence reported, e.g. IN-07 evidence predating changes). Limits as claimed: reachability static, tests not run by the checker.
* **IN-07 — verified.** Pathway on the real configuration in staging (run A): observation -> latent change (latent_b 0.62/0.42/0.25) -> prediction registered before outcome (21 resolved) -> action -> consequence (3 delayed outcomes linked) -> R_psi internal reward (4/4 records) -> neural update (experiential + heads; R_psi policy-improvement update, policy KL 0.062) -> changed behaviour on a held-out context (TV 0.128, 0.0 after restoring the weights). Reward meta-update (outer loop) not applied in run A. Run B (INT3B run-1 flags, 5 computer + 28 LabWorld decisions, 4 lifetimes): 13 of 16 hold (approvals_exercised, meta_update_applied, meta_step_changes_policy fail); held-out TV 0.276 -> 0.0 after restoring the weights; 68 frequent updates, 4 policy-improvement updates (KL 0.041-0.057), 5 of 18 slow-adapter steps changed the adapters; no reward meta-update applied (38 skips: fewer than 3 independent verified groups). Staging only; no live service.
* **IN-09 — partial.** Code is standalone: importing all 176 cam_live modules loads no module of CAM Unified / Crucible / AgentRS and no file of their source trees; tools/ and tests/ import none (p05 1, 2, 5); test_the_cam_live_source_is_standalone passes. Fails at run time: 1126 of 1233 third-party modules load from /workspace/agentrs/.venv (layered .pth), and config.GPU_LOCK_PATH defaults to /workspace/agentrs/runs/cam_unified/gpu_heavy.lock (G2I-10, p05 3, 4).
* **OP-01 — partial.** DiskGuard refuses before writing; agent.save and agent.promote are refused below the minimum and write nothing (p03 1-3); the runtime pauses instead of checkpointing (test_runtime_pauses_instead_of_checkpointing_when_disk_is_low, test_agent_promotion_checks_the_guard_and_the_runtime_pauses pass). Fails: weights-only exports (agent.export_weights, cam-live export-checkpoint of a stopped run) write below the minimum (G2I-4, p03 4, 5).
* **OP-02 — partial.** Runtime checkpoints: newest 2 kept, optimizer state only in the newest (test passes); prune_runs/apply_retention keep the newest, protected checkpoints and records (p03 6-8). Fails: a symlinked checkpoint makes retention delete optim/ outside the tree (G2I-5, p03 9); learner promotions keep optimizer state in every kept checkpoint (G2I-6, p03 10).
* **OP-03 — partial.** Doctor disk check, dashboard health disk_free_gb, runtime guard with storage_warning / storage_low events exist and are tested (test_read_endpoints_show_the_run, test_doctor_reports_what_is_on_this_machine pass). Fails: the doctor's disk check uses a fixed 20 GB, not storage.min_free_gb (reports ok while the runtime would refuse every checkpoint), and the dashboard shows no guard level (G2I-7, p03 11).
* **OP-05 — partial.** Staging: real DiscordExecutor against the local staging Discord, integrated agent choosing read-only Discord tools: 6 of 6 observations become store items, 5 of 6 are in the context of later decisions (model input), every Discord action is an experience record pointing at its observations, frequent updates with non-zero deltas ran on that store (p06 6/6); INT3B rehearsal evidence agrees. Fails / not shown: no observation from the live Discord (not permitted to test).
* **OP-06 — verified.** internal.request_operator is offered, records one deduplicated request, the operator CLI grants it, the answer returns once as an event with operator trust and the learner's own text separately under self trust, the authorization service's state is byte-identical before / after the grant and the same action stays denied, the answer is a store item in the next decision's context (p02 14/14; both acceptance tests pass).
* **OP-08 — partial.** Source is pushed (the local remote-tracking ref equals the branch head; no network check) and the checker records unpushed / uncommitted / ignored paths (test_durability_reports_unpushed_commits_and_uncommitted_files passes). Fails: tools/push_milestone.sh scans only the net diff, so a credential added and removed inside the unpushed commits is pushed (G2I-1, p01 B, E), and binary / compressed files are not scanned (G2I-2, p01 C, F); the committed durability record in docs/COMPLETION.md is from 799f032 (G2I-12). No leak found in the real history.
* **DL-02 — partial.** Holds: parameter counts re-measured at a5ab772 equal param_counts.json (74,672,079 trainable, 4,022,468,096 frozen; also equal to run A), 187 file:Symbol citations exist. Fails: MODEL_ARCHITECTURE.md:284/:302/:312-314 say set_objective calls MetaLearner.new_lifetime (it calls meta.new_objective, agent.py:913); the policy update's use of stored credit (agent.py:741) is not described; :249-250/:329-331 describe estimator selection that the agent never wires (G2L-6).
* **DL-04 — partial.** Holds: math_check.py output identical to math_check.txt; 7 sampled equations match the code. Fails: stale at a5ab772 - MATH_NOTES.md:359-360/:666/:669/:693-694 (info_gain, epistemic_after, learning_progress now produced), :413-421 (credited goal return missing from L_inner), :611-612/:677-679 ('no training loss reads the stored credit'), :623-626 (slow guard per provider), :116-121 (Poisson mode search); :609-610 estimator selection not wired.
* **DL-20 — partial.** Holds: offline preflight reproduces exactly (planning exit 3 / 9 items, discord_100 exit 3 / 25 items, same names); scopes, credential bindings, limits, service users correct. Fails: OPERATOR_RESOURCES.md:32-35/:70 'the code has no override' for persistent_storage (preflight.py:300 ACCEPTABLE_RISKS, --accept-ephemeral-storage); :156-158 GPU-hold keys absent (configs/cam_live.yaml sets them); :103-108 the by-hand launch fails at step 2 (auth/keys missing); :29 GPU item imprecise.
* **IN-05 — partial.** Holds: 66/66 documented --help commands exit 0; dependencies match. Fails: 5 of 21 executed claims wrong (REPRODUCE.md:456-457 staging preflight exits 0 not 3; :156-159/:171-172/:428 suite is collectable in one process (1593); :207-211/:425-427 validate --debug fixed; :154/:188/:411 result values); §9.1 launch lacks --accept-ephemeral-storage; :345-347/:445-450 GPU lock text stale; new validate options undocumented.
* **IN-06 — partial.** Holds: KNOWN_FAILURES.md is append-only and its rows were true at 204e9d4; evidence paths exist. Fails at a5ab772: nine rows still open are fixed (KF-D01, D07, E01, E02, E03, E05, E10, E11, E12); KF-R02 cites no later validation; missing: staging validation not passing (13/16, 15/16), INT3B-4, INT3B-5, no notes in the rehearsal, GPU-hold over-run, G2L-6/DX-1, INT3A-5, G2-harness R2-C1/R3-M1/R3-m1/R3-m2 (d04: 1 hold, 18 broken).

## 8. What I could not check

* No live service, no live Discord observation, no live operator request (by rule). OP-05, CU-30 and IN-07 are
  judged on the staging services only.
* Separated-uid operation (G-rt p11) and anything privileged; GPU and slow test marks; G-pr's real-model probes.
* Whether a pushed commit is really on GitHub: only the local remote-tracking ref was read (no network).
* The staging validation on a quiet machine: run A ran at load average 30-60 (other tenants' jobs and, during run A,
  some of my own CPU jobs), run B at about 20; load shortens the phases (run A reached 4 decisions); G2I-3 and
  G2I-15 do not depend on load.
* Learning quality (improvement with repetition, transfer, usefulness of self-directed experiments): not measured
  by anyone at the real configuration; not claimed.

