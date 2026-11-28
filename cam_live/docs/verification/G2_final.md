# G2-final — independent re-verification at the final code fdab7d3

Verifier: G2-final (workstream G). I did not write any of the code under test. Code verified:
**`fdab7d3033e98082bd213015063cfca6c35d8a12`** (2026-11-28 07:10 UTC), as an exact `git archive fdab7d3` extract in
my scratch directory (`<scratch>/G2FINAL/src`, `PYTHONPATH` / `G2F_SRC` pointing at its `cam_live/src`). Checks that
need the service uids to reach the code ran from the worktree, whose `cam_live/src` is identical to fdab7d3
(`git status --short cam_live/src` empty, `git diff --stat fdab7d3 -- cam_live/src cam_live/tests` empty). Every
process that could take the GPU lock had `CAM_LIVE_GPU_LOCK=/workspace/agentrs/runs/cam_unified/gpu_heavy.lock`.
Nothing live was started, no external service was contacted.

Scope: the 46 requirements whose verdicts the checker marks stale (their files changed after the verdict's commit):
NA-01 PI-02 SE-01 SE-02 MR-01 MR-05 RC-01 CU-06 CU-09 CU-24 CU-34 CU-37 CU-38 LX-01 LX-02 LX-07 LX-08 LX-10 LX-12
LX-13 LX-14 LX-15 LX-16 LX-17 LX-19 LX-20 CM-01 CM-03 CM-05 CM-06 CM-07 CM-08 CM-09 CM-11 CM-12 OP-01 OP-02 OP-07
AE-04 OB-02 LX-03 CM-02 CM-04 CM-10 IN-01 OP-03. Verdicts: `verdicts_g2final.json` (commit fdab7d3, supersedes all
earlier verdicts of these ids). New probes: `repro/g2final/`.

**Result: 41 verified, 5 partial (LX-02, LX-03, LX-14, LX-17, AE-04), 0 failed, 0 unverified** of 46. Fixed and confirmed by probes: G2R-1b (its 18 variants), G2R-2b, G2R-5, G2R-7b, G2R-15 (honest order), G2R-16, G2R-17 / G2I-18 (fail closed), G2R-18, the G2I-7 remainder. New: **G2F-1** (LX-02, readable disguises still pass), **G2F-2** (LX-03 / LX-14, minor: the verifier puts a superseded signed amendment back in force when the learner's process delivers it after a newer one), G2F-3 (observation: dashboard storage thresholds come from the learner-written state.json), G2F-4 (observation: GPU tests waiting for the lock error at pytest-timeout). Still partial for unchanged reasons: LX-17 (usefulness of community resources disclosed as not measured, not tracked), AE-04 (transferable competence constant).

## 1. What changed since each verdict

The file list of each requirement is the manifest's plus its status-file claims (`verify_completion.merge_status`);
`git diff <verdict commit> fdab7d3 -- <files>`:

| Change (commit) | Requirements whose files it touches |
|---|---|
| `nn/gpu.py` `open_lock_file`: the lock file is never created, read-only fallback for service uids, `GpuUnavailable` naming `CAM_LIVE_GPU_LOCK` (59807fc, 7265843); the same call in `runtime/gpu_cycle.py`, `runtime/cli.py`, `harness/visual.py`, `harness/grounding/bench.py`; doctor's `gpu_lock` check needs the file, level live | NA-01, CU-06, CU-09, CU-34, CU-38, LX-12, LX-16, CM-01 … CM-12 (cli.py) |
| `agent.py`: one line, the storage guard before a weights export (INT3B-8, in 257267c) | PI-02, MR-01, MR-05, RC-01, AE-04 (since ef0f3ff); RC-01's G2-learn verdict at cf46eb1 also predates G2L-10 (a3844c8) |
| `predict/providers.py` eligibility without the learner's own flags (G2L-8, 914b10a); `predict/predictor.py` goal views / VOI reuse (d807a40, 41d9386, 4476c27) | SE-01, SE-02 (G-pr verdict of 2026-11-27 05:02) |
| `runtime/tasks.py`: confusable skeleton, leetspeak, spelled letters, concatenation split, listing / bump sites, outreach moves; `read_task_file` / `load_task_bytes` (929434c, e091b41) | LX-01, LX-02, LX-14, LX-17 |
| `runtime/verifier.py`: signed amendment of the objective in force (G2R-15), one read + own 0600 copy + replay refusal (G2R-18), keyed baseline of members present at the start (G2R-2b), usefulness reported as not measured (G2R-3b) (e091b41) | LX-03, LX-17 |
| `runtime/loop.py`: `_assign` reads the file once and always asks the verifier; programs whose external steps do not fit are not started; `storage_guard` in state.json (e091b41, 2247328) | CU-24, LX-07, LX-08, LX-13, LX-14, LX-15, LX-19, CM-08, CM-09, CM-11, OP-01, OP-02, OB-02, IN-01 |
| `runtime/logs.py`: a 17-20 digit number is an account id only if its snowflake time is 2015-03 … 2100 (e091b41) | LX-07, LX-08, LX-13, LX-15, LX-20 (and CU-37 since 8f6c615) |
| `runtime/plan_report.py`: every text through the run logs' redactor, counts per section (e091b41); CU-37 / OP-07 also see INT3B rounds 1-2 since 8f6c615 | LX-07, CU-37, OP-07 |
| `runtime/dashboard.py`: storage level ok / warning / low in health (2247328) | LX-10, CM-07, OP-03 |

## 2. What I ran

| What (fdab7d3) | Result |
|---|---|
| The named acceptance tests of the 46 requirements (252 tests), extract, `-m "(slow or browser or not browser) and not privileged and not gpu"` | **304 passed, 1 failed**, 791 s: `test_meta_update_runs_with_a_nonzero_meta_gradient` (MR-05, IN-01) — no meta-update applied after the credit / harness tests in the same process (skips: 36 insufficient independent outcomes, 7 no supported episode, 6 no policy update); **passes alone** (next row): order-dependent, as INT3B reported |
| The privileged named tests (+ `test_m_gpu_lock_service_uid.py`), worktree, `-m "privileged or browser"` | **18 passed**, 44 s |
| NA-01's GPU tests (`tests/nn/test_gpu_qwen.py`, 3), extract, `-m gpu` | **not run**: the shared lock was held by another tenant's evaluation batches throughout (two queued blocking holders); the 3 tests ERRORED at pytest-timeout's 900 s while their fixture waits up to 1800 s (a test-configuration mismatch: a waiting GPU test errors instead of skipping). Instead: the real Qwen3-4B on the CPU (float32, same code path, extract): 4,022,468,096 frozen backbone parameters, 16,713,856 trainable adapter parameters; embeddings cos 0.676 (apple / crimson fruit) vs 0.078 (apple / tax filing); G2-learn's `probe_real_cpu_grads.py` 6/6 PASS (193 s) |
| MR-05's `test_meta_update_runs_with_a_nonzero_meta_gradient` alone, extract | **1 passed**, 265 s |
| `cam-live validate --mode staging --debug` (CPU), extract | ran to the end: exit 1, 443 s within its 840 s box, GPU peak 0.0 GB; **13 of 16** hold (fail: approvals_exercised — one request, approved and executed with its grant, none to reject; meta_update_applied / meta_step_changes_policy — 0 applied, 50 skips for insufficient independent outcomes) |
| `cam-live doctor --json --strict`, extract, without / with `CAM_LIVE_GPU_LOCK` | exit 0 both; without: `gpu_lock` fails, `needed_for: live`, blocks live only ("… is missing: set CAM_LIVE_GPU_LOCK …"); with: holds; 52 checks |
| `cam-live evaluate-transfer` on a debug weights-only export of `build_agent` (T1, seed 0, 1 lifetime, 2 decisions), extract | exit 0, 44 s, `{"T1": {"S_q": 1.0, …}}` |
| G2-runtime probes, extract | tasks_disguise 0/23, verifier_live_source 0/10, next_objective 0/7, limits_program 1/5 (check 3, see §4), plan_report 0/10, redaction 0/13, dashboard 0/19, control_signed 0/9, preflight 0/7, verifier_rules_after_start 0/9, settle_crash 0/15, cli_p10copy 1/35 (0c: other agents' files in the worktree's `runs/`), gpu_lease 1/3 (check 1, disclosed; run with its scratch lock created first, §4) |
| G2-runtime root probes, worktree (separated ones from a temporary copy next to my own probes, removed after) | gpu_lock_path 1/3 (check 2, see §4), separated_resume 0/8, separated_p11copy 1/39 (check 14, design, as in round 2) |
| G-rt p01 … p15, extract | p01 0/36, p02 0/477 (kill -9), p03 0/15, p04 0/11, p05 1/33 (1a: 16 REQUIRED, design), p06 14/38 (by construction of the G2R-2 fix: its members join before the first reading), p07 0/62, p09 1/28 (1b = 1a), p12 4/5 (documentation numbers, not code), p13 0/8, p14 0/11, p15 0/3 |
| G2-integration p03 / p04 / p05 / p07, extract | p03 0/11, p04 0/12, p05 1/5 (venv layering, IN-09, not in scope; check 4 holds without `CAM_LIVE_GPU_LOCK`), p07 0/4 |
| G2-harness g15 / g16 / g18, worktree, root | 2/0, 3/0, 7/0 (holds / broken) |
| G2-decision probe_misc, G2-learn probe_agent_credit, extract | 4/4 PASS; 13/13 PASS |
| **New** `repro/g2final/probe_tasks_disguise_r3.py` | **9 of 9 new variants accepted and delivered to the learner** (G2F-1); shipped task files clean (5/5) |
| **New** `repro/g2final/probe_assign_order.py` | **2 broken of 6** (G2F-2): an older signed amendment delivered after a newer one is put in force and decides the signed outcome |
| **New** `repro/g2final/probe_baseline_rejoin.py` | 0/4 broken (G2R-2b fixed); 1 note (late first reading) |
| **New** `repro/g2final/probe_gpu_lock_final.py` (root) | 0/12 broken: every lock path fails closed in a copy and creates nothing; camlearn / cambrowser wait on root's flock and get the lock after it |
| **New** `repro/g2final/probe_dashboard_storage.py` | 3 of 8 broken (G2F-3, observation): malformed thresholds in state.json give 500 or "ok" |

Outputs: `<scratch>/G2FINAL/{run1b,run2,run1g,validate}.log`, `…/probes_out/`, `…/rt_out/`, `…/lens_out/`, JUnit
`…/run1b_named_archive.xml`, `…/run2_privileged_worktree.xml`, `…/run1g_gpu_archive.xml` (scratch only: verifiers
write no JUnit into the evidence directory).

## 3. New defects

**G2F-1 (LX-02; same class and severity as G2R-1b) — readable disguises of the covered words still pass the
loader.** `runtime/tasks.py` `_scan_variants` / `skeleton` / `_content_hits`: the skeleton maps the listed
Cyrillic / Greek / Armenian / Cherokee letters, folds `0 1 3 4 5 7 @ $ ! | €`, joins letters separated by space, `.`
or `-`, and segments identifiers; the rules then match whole ASCII words. Each of these, in an otherwise fully filled
copy of `tasks/discord_100.yaml`, is accepted by `load_task(for_run=True)` with no finding and reaches `learner_view`,
while the same sentence written plainly is refused (control per variant):
Latin letters with diacritics (`réddit`, `gíveaways` — NFKC keeps them, they are Latin so no mixed-script flag),
combining marks (`reḍdit`), Latin small capitals (`ʀᴇᴅᴅɪᴛ`), letters joined by `*` / `_` in the objective
(`r*e*d*d*i*t`, `g_i_v_e_a_w_a_y_s`), a word split once (`red dit`, `give aways`), percent-encoded letters in an
`authorized_destinations` URL (`%72%65%64%64%69%74`), leetspeak outside the folded set (`+ik+ok`, `9iveaways`),
platforms of the covered category that are not listed (imgur, 9gag, lemmy, vk), synonyms of listed words (raffles,
prize draws, flyers). `G2F_SRC=<extract>/cam_live/src .venv/bin/python
cam_live/docs/verification/repro/g2final/probe_tasks_disguise_r3.py` (9 broken; G2-runtime's 23 variants all hold).
A deny-list over free text that CAM receives verbatim cannot be completed one spelling at a time; the objective text
is the channel that matters (labels, metric names and tool patterns are ASCII by schema; URLs are any non-space text). Suggestion: treat the objective text as an
allow-list item (always require `objective_confirmation` of its hash, the scan becoming a warning), and percent-decode
URLs before the scan.

**G2F-2 (LX-03, LX-14; minor, new with G2R-15) — the verifier can be made to judge a superseded signed amendment.**
`runtime/verifier.py` `assign_objective` (lines 360-391) accepts any operator-signed `assign_objective` whose file
hash matches and whose command id was not applied before, and makes it the objective in force — it never compares the
command's signed `issued_at` with the amendment already in force. The verifier service exposes the method to the
learner's process over its socket (`verifier_service.py:26, 50`; the threat model of G2-runtime's U1 check). The
operator signs X (target 40, 1 day) and then Y (target 100000, 10 days) for the running objective; handed to the
verifier in the order Y, X (what the learner's process can do with the commands it reads from the control file),
both are accepted and X is in force: on day 2 the verifier signs FAILED ("value 9 against target 40") although the
operator's latest amendment runs until day 10 (check 1b); in the same way an earlier, easier signed target comes back
(X's 40 instead of Y's 100000). Honest order through the runtime keeps Y (check 2 holds); a replayed applied command
is refused, also after a restart (check 3); the 0600 copy restores an edited file (4a, 4b hold).
`… repro/g2final/probe_assign_order.py` (checks 1, 1b). Fix: refuse an assignment of an objective whose in-force
command was issued later (its signed `issued_at`), or a signed per-objective sequence number.

**G2F-3 (OP-03 / LX-10; observation) — the dashboard's storage level is the learner side's claim.** `runtime/
dashboard.py:343-357` takes `min_free_gb` / `warn_free_gb` from `state.json`, which the runtime writes (in a separated
run: the learner uid), and converts them with `float()` without a guard: a string or null answers `/api/health` with
HTTP 500 (with CSP / nosniff / DENY, checked — the same accepted behaviour as a malformed state file on `/api/status`), NaN
gives level "ok" whatever the free space (`… repro/g2final/probe_dashboard_storage.py`, checks 5). The runtime's own
guard is not affected and an honest runtime writes floats; but the operator's view would be sounder computed from
the operator's configuration (`storage_thresholds`) than from the run's record.

**G2F-4 (tests; observation) — a GPU test waiting for the shared lock errors instead of skipping.**
`pyproject.toml:31` sets pytest-timeout 900 s; `tests/nn/test_gpu_qwen.py:25` (and the `gpu_heavy` fixture,
`tests/conftest.py`) wait up to 1800 s for the lock and then skip. While another tenant holds the lock for longer
than 900 s, the GPU tests end as ERROR ("Timeout (>900.0s) from pytest-timeout"), which reads like a failure of the
code. (In my 15-minute wait their non-blocking 5-s poll never got the lock while the other tenant's two processes
handed it to each other with blocking waits; the runtime's `GpuLease` waits blocking, as a queued waiter.)

## 4. Earlier defects and probe conflicts, judged at fdab7d3

| Item | At fdab7d3 | Basis |
|---|---|---|
| G2R-1b disguises (LX-02) | **fixed for the 9 round-2 variants** (and the 9 round-1 ones); new variants pass (G2F-1) | probe_tasks_disguise 0/23; probe_tasks_disguise_r3 9/9 accepted |
| G2R-2b rejoin (LX-03) | **fixed**: present at the start, leave, rejoin → value 0, also after a verifier restart; baseline file 0600, log holds counts only; newcomers still count | probe_verifier_live_source A2; probe_baseline_rejoin 1, 2, 3, 5. Inherent: a member who left before the verifier's first reading is unknown to it (note 4) |
| G2R-3b usefulness (LX-17) | **disclosed, not measured**: `not_measured.usefulness_of_community_resources` in every verdict's notes | probe_verifier_live_source C3 |
| G2R-5 program overshoot (LX-14) | **fixed**: a program whose external steps do not fit what is left is refused before it starts | probe_limits_program 4, 5 hold |
| probe_limits_program 3 vs 4 (implementers' conflict) | **check 3 is obsolete, not a defect**: its scenario (limit 4, programs of 3 steps) now stops after one program with 1 action left, which no program fits; the objective ends by `max_decisions` because the limit is not exhausted. Ending it "exhausted by max_external_actions" with an action left would be the wrong report. Check 4 (the property) holds | probe output; `loop.py:1411-1417` |
| G2R-7b plan report exports member data (LX-20) | **fixed**: name#tag, e-mail, Discord id in a CAM note are hashed / redacted with the run's salt; plausible snowflakes (also 17-digit 2015-05 ids) hashed, a 17-digit count kept | probe_plan_report R2d; spot check of `Redactor.text` |
| G2R-15 duration extension ignored by the verifier (LX-14) | **fixed** in the honest order (E1 holds); new ordering gap G2F-2 | probe_next_objective E1; probe_assign_order 1, 2 |
| G2R-16 separated learner could not take the lock (LX-16) | **fixed**: camlearn's `GpuLease` / `nn.gpu.gpu_lock` and cambrowser's visual lock wait on root's flock of a root:root 0664 file and get it after the release; file untouched | probe_gpu_lock_final B1-B4; probe_gpu_lock_path 3; `test_m_gpu_lock_service_uid.py` passes |
| G2R-17 / G2I-18 private lock in a copy (LX-16) | **fixed (fails closed)**: in an extract every lock path (nn.gpu, GpuLease, cli, visual, grounding bench) raises `GpuUnavailable` naming `CAM_LIVE_GPU_LOCK` and creates no file; doctor blocks live. probe_gpu_lock_path check 2 still prints BROKEN because the copy's default path is not the machine's lock — the hazard it describes (a silently created private lock) is gone. `runtime/cli.py:155` still makes the lock's directory (empty `runs/` left in the copy) | probe_gpu_lock_final A0-A, doctor run |
| probe_gpu_lease (implementers' conflict) | **the probe needs its scratch lock file to exist** since 7265843 (fail closed is the intended behaviour); with the file created first (`repro/g2final/run_gpu_lease_prelock.py`) the result equals round 2: 2, 3 hold, 1 broken (a single long call is not pre-empted, disclosed) | runner output |
| G2R-18 assigned file read twice | **fixed**: runtime and verifier each read once and hash what they parse; the verifier keeps a 0600 copy (0700 dir) and restores from it after the file was edited; replays refused | code reading `loop.py:953-958`, `verifier.py:319-349`; probe_assign_order 3, 4a, 4b |
| G2I-7 remainder: dashboard without guard level (OP-03) | **fixed** for honest runs (levels warning / low / ok and defaults marked through the served API); observation G2F-3 | probe_dashboard_storage 1-4 |
| probe_next_objective U4 | still refused, now as "already applied (replayed command)" before the file check | probe output |
| G-rt p06 14/38 | by construction of G2R-2 (members listed at the first reading are pre-existing), as in round 2 | p06 output |

## 5. Per requirement

| Requirement | Verdict | Basis (details in verdicts_g2final.json) |
|---|---|---|
| NA-01 | **verified** | only the lock open changed; real Qwen3-4B on the CPU (float32): 4,022,468,096 frozen parameters, embeddings, generation, gradients 6/6; the 3 GPU tests could not get the lock (G2F-4) |
| PI-02 | **verified** | registration before outcomes; suites pass |
| SE-01 | **verified** | goal-conditioned success prediction; tests pass |
| SE-02 | **verified** | prefix training corrects earlier predictions; tests pass |
| MR-01 | **verified** | internal rewards per record; tests pass |
| MR-05 | **verified** | meta-gradient mechanism; applies only with >= 3 independent outcomes (fragile in short runs) |
| RC-01 | **verified** | delayed outcomes linked with uncertainty |
| CU-06 | **verified** | screenshot observations; browser tests pass |
| CU-09 | **verified** | validated mouse / keyboard; browser tests pass |
| CU-24 | **verified** | no environment calls while waiting; outcomes once |
| CU-34 | **verified** | benchmark code unchanged except the lock open |
| CU-37 | **verified** | separated uids; privileged tests and root probes pass |
| CU-38 | **verified** | grounder output only a proposal |
| LX-01 | **verified** | Discord objective as external task configuration |
| LX-02 | **partial** | G2R-1b fixed for its 18 variants; G2F-1: 9 new readable variants pass |
| LX-07 | **verified** | experiment records; learner actuals apart |
| LX-08 | **verified** | learning / meta records logged |
| LX-10 | **verified** | credential / CSRF / origin / headers; storage level shown |
| LX-12 | **verified** | preflight names what is missing |
| LX-13 | **verified** | a record per consequential decision |
| LX-14 | **partial** | G2R-5 and G2R-15 fixed; G2F-2: a superseded signed amendment can be put back in force |
| LX-15 | **verified** | 17 steps; crash recovery |
| LX-16 | **verified** | detached and separated; GPU lock fixed (G2R-16, G2R-17) |
| LX-17 | **partial** | usefulness of community resources disclosed as not measured, not tracked |
| LX-19 | **verified** | stop with checkpoint; signed next objective verified |
| LX-20 | **verified** | redaction in logs and plan report |
| CM-01 | **verified** | cam-live doctor |
| CM-03 | **verified** | cam-live test |
| CM-05 | **verified** | cam-live inspect-harness |
| CM-06 | **verified** | cam-live run |
| CM-07 | **verified** | cam-live status |
| CM-08 | **verified** | cam-live pause |
| CM-09 | **verified** | cam-live resume |
| CM-11 | **verified** | cam-live export-checkpoint |
| CM-12 | **verified** | cam-live verify-completion |
| OP-01 | **verified** | storage guard refuses writes, run pauses |
| OP-02 | **verified** | retention |
| OP-07 | **verified** | operator credential entry isolated |
| AE-04 | **partial** | goal-free mode budgeted; transferable competence constant |
| OB-02 | **verified** | alternatives considered / chosen, result recorded |
| LX-03 | **partial** | G2R-2b fixed; G2F-2: the learner's process can make the verifier judge a superseded signed amendment |
| CM-02 | **verified** | cam-live init |
| CM-04 | **verified** | cam-live validate --mode staging |
| CM-10 | **verified** | cam-live evaluate-transfer |
| IN-01 | **verified** | the runtime calls the integrated agent at every step |
| OP-03 | **verified** | doctor, dashboard level, runtime guard |

## 6. Not checked / limits

* Live Discord, any live run. No GPU (bf16) run of the real configuration: NA-01's three GPU tests could not get the
  shared lock (G2F-4); a CPU float32 run of the same model replaced them (§2).
* The real-configuration staging validation (LX-11, not in scope) and the reward meta-update on the real
  configuration (MR-05: mechanism only, as G2-decision).
* CU-34's benchmark numbers were not re-run (only `bench.py`'s lock call changed; its executed evidence predates it).
* Documentation requirements (IN-06, DL-02, DL-04, DL-20, IN-05, CM-13) are left for after the documentation refresh.
* The machine was loaded (load average 15-26, other tenants' evaluation batches holding the GPU lock); no timing
  assertion was made.
