# CAM Live — substantive evaluation results (Protocol v3)

**Status (2026-11-28 ~09:30 UTC): pre-amendment baseline on snapshot 257267c complete for X0 and X1; X5 ran;
X2-X4 not run on 257267c (coordinator re-scope: 257267c is the PRE-AMENDMENT BASELINE, the GPU goes to CAM's live
planning run and the core-intelligence amendment agents; the comparison run on the amended snapshot follows with the
same seeds and budgets).** Binding plan: `docs/S/PROTOCOLS.md` Protocol v3 (v3.2 configuration, v3.3 families,
v3.4 experiments, v3.5 statistics). Per Protocol v3.5 **no comparison here can reach p < 0.05** (3 paired seeds:
smallest two-sided p = 0.25; 2 seeds: 0.5); everything below is an estimate with its range across seeds or a
count. Nothing here is a significance claim.

## 1. Configuration of the baseline (every substantive run)

* **Code: one fixed snapshot** `runs/cam_live/eval/snapshots/257267c1c315` = `git archive` of `cam_live/` at commit
  **257267c1c31568b693adf0cbcd0e9f9b547e62e1** (contains the DX round 2 slow-step guard fix, DX-1, G2R-5 and the
  evaluation code of the previous S agent). Every X0 / X1 / X2 result records it (`code.snapshot_commit` per run,
  `eval_code` per objective; `code_check.all_match: true` in `docs/evidence/S/v3/summary_x0.json` /
  `summary_x1.json`). X5 needs a goal-free driver that 257267c does not contain: it ran on the overlay
  `257267c1c315+evalcd34fd10f503` (agent / learner / runtime code of 257267c byte-identical; only `cam_live/eval`
  from cd34fd1; recorded as `257267c...+eval:cd34fd10...`). X6 meta checks ran on the same overlay.
* **Agent** (`eval.oe_runner.describe`, recorded per run): `CamLiveAgent`, backbone Qwen/Qwen3-4B revision
  1cfa9a7208912126459214e8b04321603b3df60c, frozen, bf16, on cuda; experiential core scale 50m. Parameters:
  frozen backbone 4,022,468,096; experiential core 50,953,765; semantic adapters 16,713,856 (LoRA 11,796,480 +
  latent bridge 4,917,376); reward 949,121; heads 6,055,337 (world 1,022,954, transition 1,977,600, evaluator
  783,192, policy 702,339, downstream 84,195, RUDDER 559,361, COCOA 925,696); persistent trainable total 74,672,079;
  fast-state numbers per kind 16,512. Loss providers: world_model, fast_prior, evaluator, world_model_adapters,
  evaluator_adapters, rudder, cocoa, retention, bridge_grounding; credit assigner on.
* **Open-ended generation on**: `use_backbone=True`, `proposer_max_new_tokens=512`, `proposer_k=4` (backbone
  proposals merged with the schema walker and skills); selection rule `thompson_policy`; `experiment_rate=0.1`;
  adapters, bridge grounding, retention on; learner settings `eval.oe_runner.SUBSTANTIVE_LEARNER` (adapter step
  every 3 frequent updates, slow batch 3); meta-update minimum 4 independent verified outcome groups; bootstrap
  mix at its default cold start (1.0 in every objective).
* **Computer**: `harness.env.ComputerEnvironment` with the harness executors (internal, code sandbox, HTTP),
  authorization by the real `harness.auth.PolicyService` in staging mode. No permitted tool was unavailable in
  any objective.
* **Scheduling**: `python -m cam_live.eval.oe_batch --plan v3 --concurrency 3 --max-gpu-gb 36`: one parent holds
  the shared GPU lock for at most 810 s per lease (child budget 690 s, children killed at 790 s; longest measured
  objective hold 670 s, none killed), one objective per sequence per lease, learner state carried between leases
  by the sequence's own checkpoint. GPU lock: the machine's shared lock file
  `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock` (`CAM_LIVE_GPU_LOCK`, change request S2-1; holder checked
  with `fuser` and `/proc/locks` during the first lease, `scratchpad/S2/oe_batch.log`). GPU peak per child 13.6 GB.
* **Budget per objective**: at most 20 decisions and the wall budget of the lease (an objective still running at
  ~620 s is stopped: "wall_budget"); achieved = verified value >= 0.9 (SW: pass fraction of 60 hidden tests; MECH:
  held-out R^2 on 80 points). Decisions to achievement are censored at 20 (also when the wall budget stopped the
  objective earlier).

Evidence: raw results `docs/evidence/S/v3/final_257267c/` (one JSON per sequence, child logs, `batch_log.jsonl`
of both batches), summaries `docs/evidence/S/v3/summary_<exp>.json`, per-objective tables
`docs/evidence/S/v3/tables.md` (`CAM_LIVE_EVAL_HOME=runs/cam_live/eval/v5 python -m cam_live.eval.v3_report
--out-dir cam_live/docs/evidence/S/v3 --expect-commit 257267c1c31568b693adf0cbcd0e9f9b547e62e1 --markdown
cam_live/docs/evidence/S/v3/tables.md`), meta checks `docs/evidence/S/v3/x6/`.

## 2. What ran on the baseline 257267c

| experiment | plan | status |
|---|---|---|
| X0 current capability | E, seeds 0-2, `SW:rle, SW:rpn, MECH:7` | complete, 9 of 9 objectives (05:52-07:08 UTC) |
| X1 learning within a family | E, seeds 0-2, `SW:rle, rpn, merge, topk`, exports pre / post | complete, 12 of 12 (06:57-09:09) |
| X5 goal-free (11.4) | E, seeds 0-1: 20 goal-free decisions, then X1's plan | complete, 10 of 10 (07:12-08:50; run because X1 showed achievement) |
| X2 transfer (16.8) | initial vs experienced X1 exports, fresh processes | **not run** (re-scope): only `initial_s0` 2 of 3 objectives ran before the stop (SW:unique achieved in 6, SW:base in 9); no initial / experienced pair exists, so no S_q |
| X3 baselines A, B, D, C | seeds 0-1, X1 plan | **not run** (re-scope) |
| X4 CU-33 skills vs primitive | E, seeds 0-1, CU staging | **not run** (re-scope) |
| X6 meta checks (16.5) on X1 roots | `python -m cam_live.eval.meta_checks` | see §6 |
| X6 integrity battery + trivial world | CPU debug configuration, seeds 0-4 | complete (engineering diagnostic, §7) |

The batch was stopped by SIGTERM between two leases (no child running) at 08:50 UTC on the re-scope and restarted
with `--only x0,x1` to finish X1 seed 2; no objective was interrupted.

## 3. X0 — current capability (E, seeds 0-2)

| seed | SW:rle | SW:rpn | MECH:7 |
|---|---|---|---|
| 0 | achieved in 16 decisions | achieved in 5 | not reached in 20 (R^2 0.00) |
| 1 | achieved in 7 | achieved in 4 | not reached in 20 (R^2 0.00) |
| 2 | not reached in 20 (pass 0.00) | achieved in 2 | not reached in 20 (R^2 0.00) |

* Achieved 5 of 9 (never reached 4, all at the 20-decision budget; interrupted 0). By problem: rle 2/3, rpn 3/3,
  MECH:7 0/3 (the mechanism family was never solved: held-out R^2 0 in every seed).
* Decisions to achievement (censored at 20), mean per seed 13.7, 10.3, 14.0 (mean 12.7, range 10.3-14.0).
* Executable fraction of chosen actions 0.91 (0.90-0.92); backbone-written fraction of chosen actions 0.43
  (0.35-0.48) and of candidates 0.35 (0.32-0.38); repeats of failed actions per objective 0.44 (0-0.67); distinct
  strategies per objective 7.7 (7.3-8.0); mean prediction NLL 0.10 (-1.04-1.23).
* Learning: semantic adapters changed in 9 of 9 objectives; **slow (adapter) steps 37: 34 kept, 2 reverted by the
  guard, 1 deferred**. Meta-updates: 123 records, 4 applied (meta-gradient norms 7.29, 6.21, 0.48, 16.79; reward
  group changed in 3 objectives); skipped: insufficient independent outcomes 85, zero advantage 16, no supported
  episode 12, no policy update recorded 6.
* Cost: wall time per objective 390 s (per-seed means 328-429 s).

## 4. X1 — learning within a family (E, seeds 0-2)

| seed | #0 SW:rle | #1 SW:rpn | #2 SW:merge | #3 SW:topk |
|---|---|---|---|---|
| 0 | not reached in 20 | achieved in 16 | achieved in 6 | achieved in 1 |
| 1 | achieved in 6 | achieved in 3 | achieved in 12 | achieved in 10 |
| 2 | achieved in 9 | achieved in 9 | achieved in 4 | achieved in 3 |

* Achieved 11 of 12 (never reached 1, at the decision budget; interrupted 0). By problem: rle 2/3, rpn, merge,
  topk 3/3.
* Primary (decisions to verified achievement, censored at 20, N_q with q = pass >= 0.9): mean per seed 10.75,
  7.75, 6.25 (mean 8.25, range 6.25-10.75). By position in the sequence (mean over seeds): 11.7, 9.3, 7.3, 4.7.
  The positions are different problems, so this sequence does not separate experience from problem difficulty;
  it is reported as observed, not as a learning curve.
* Executable fraction 0.96 (0.93-1.00); backbone-written fraction of chosen actions 0.49 (0.43-0.59), of
  candidates 0.34 (0.30-0.37); repeats of failed actions 0.08 (0-0.25); distinct strategies 5.4 (4.5-6.25); mean
  prediction NLL 1.55 (-1.33-3.67).
* Learning: adapters changed in 10 of 12 objectives; **slow steps 34: 25 kept, 8 reverted, 1 deferred**.
  Meta-updates: 110 records, 4 applied (norms 0.10, 3.10, 0.026, 4e-5; reward group changed in 4 objectives);
  skipped: insufficient independent outcomes 65, zero advantage 28, no supported episode 7, no policy update 6.
* Cost: wall time per objective 354 s (284-427 s per seed).
* **Run-to-run variability at equal seed.** X0 and X1 begin with the same objective instance (SW:rle, same world
  seed, same agent seed, fresh agent): seed 0 achieved in 16 (X0) vs not reached (X1); seed 1 7 vs 6; seed 2 not
  reached (X0) vs 9 (X1). The real-model runs are not reproducible from the seed (sampled generation and shared,
  non-deterministic GPU execution), so a single seed's outcome on one objective carries little information.

## 5. X5 — goal-free period before X1's plan (E, seeds 0-1)

| seed | goal-free (20 decisions) | SW:rle | SW:rpn | SW:merge | SW:topk |
|---|---|---|---|---|---|
| 0 | 20 decisions, nothing verified | achieved in 13 | not reached, **stopped by the wall budget at 15** | achieved in 4 | achieved in 11 |
| 1 | 20 decisions, nothing verified | not reached, **stopped by the wall budget at 18** | not reached, **stopped by the wall budget at 16** | achieved in 8 | achieved in 11 |

* The goal-free period ran with the agent's goal-free mode active (every chosen action carries provenance
  `+goal_free`); seed 0 chose `wait` 5x, `internal.request_operator` 4x, `internal.inspect` 4x, sandbox file /
  job operations and one backbone-written program. Its objectives are the same instances as X1's (world seeds by
  position among assigned objectives).
* Assigned objectives achieved 5 of 8 vs X1 seeds 0-1: 7 of 8. Decisions to achievement (censored at 20), mean
  per seed 12.0, 14.75 vs X1 10.75, 7.75: difference goal-free minus X1 +4.1 (bootstrap range over the 2 seeds
  1.25-7.0; exact sign-flip p = 0.5, the smallest attainable). **Confounded**: 3 of the 8 X5 objectives were
  stopped by the wall budget of the lease after 15-18 decisions (counted as 20), none of X1's; X5's decisions took
  longer (wall per objective 462 s vs 354 s). No benefit of the goal-free period is visible in this run; this
  is not evidence of harm either.
* Slow steps 38: 35 kept, 2 reverted, 1 deferred; meta-updates 134 records, 6 applied.

## 6. X6 — meta-learning checks (16.5) on the X1 run roots

`python -m cam_live.eval.meta_checks --root <X1 root> --result <X1 result>` (eval overlay cd34fd1 over the
257267c agent code; the experienced agent is rebuilt from `ckpt_post` with the real backbone under the shared GPU
lock, ~60 s per root; nothing written back). Results `docs/evidence/S/v3/x6/meta_x1_E_s<seed>.json`. Seeds 0 and
1 done; **seed 2 is still waiting for the shared GPU lock** (the first attempt was stopped by its own 1 h timeout
while the lock was held by other runs).

| check (16.5) | seed 0 | seed 1 |
|---|---|---|
| eligible records (support / held-out batches from the X1 store) | 43 | 31 |
| 1. reward parameters affect the inner update at the **live** mix 1.0: \|theta_K(psi_post) - theta_K(psi_pre)\| | **0.0** | **0.0** |
| 1. same at mix 0 (the differentiated update), against update norm | 0.021 / 0.028 | 0.0 / 0.028 |
| 1. noise floor (same psi repeated) | 0.0 | 0.0 |
| 2. inner update changes the policy: mean total variation on held-out rows | 0.029 (max 0.039) | 0.0065 (max 0.013) |
| 3. applied meta-updates with fewer than 4 independent verified outcome groups | 0 of 1 | 0 of 0 |
| 4. applied meta-updates with non-zero gradient and changed reward version | 1 (norm 0.10) | 0 |
| 4. reward group changed between the pre- and post-experience exports | yes | no |
| 6. learned reward pre vs post, Spearman over held-out rows | 0.51 | 1.0 (unchanged) |
| 6. pure magnitude change (rewards x 3) moves the update by | 1.4e-7 | 1.5e-7 |
| shadow evaluation of the checkpoints (11.2) | n = 0 (no per-candidate ground truth on SW) | n = 0 |

**Finding.** The meta-learning pathway is mechanically intact (deterministic inner update, the policy update
changes the policy, no meta-update applied without enough independent verified outcomes, a pure magnitude change
of the reward does not change the update), but **at the live bootstrap mix of 1.0 the learned reward psi has no
influence on the inner policy update** (difference 0.0 in both roots): the policy update uses the bootstrap
reward only, so whatever the meta-updates change in psi (1 applied update in seed 0, none in seed 1) does not
reach behaviour in these runs. Only at mix 0 does psi act (seed 0: 0.021 against an update norm of 0.028). The
same was seen in the v2 debug-scale diagnostics; the mix never left its cold-start value in any baseline
objective. Check 5 (pathway removed in D / E_unverified) needs X3's D runs, which did not run on 257267c.

## 7. X6 — integrity battery and trivial-reward world (engineering diagnostic, CPU, debug configuration)

`docs/evidence/S/diagnostics/v5_257267c/` (README). On 257267c the battery's trigger (a LabWorld outcome query
pending or due after >= 3 decisions) occurred in 1 of 5 seeds; there every delivered attack (forged objective
outcome, duplicate valid verdict x5, stale prediction) was refused and the independent verdict was unchanged; in
4 seeds nothing was delivered, so nothing was tested (the aggregate's "not detected" for those seeds is an
artefact, fixed in `eval.integrity.verdict_unchanged`). Trivial-reward world: `lab.tick` / noise chosen in 0 % of
decisions, verified progress 0.52 (0.37-0.67). Debug-scale diagnostic only.

## 8. Pre-fix comparison (snapshot f54d8cc + eval b06a6af) — reported separately, never pooled

`docs/evidence/S/v3/pre_fix_f54d8cc/` (README: the slow-step guard measured running-statistics changes and
reverted most adapter steps; DX-1; G2R-5). That batch was stopped between leases at 05:50 UTC.

| | pre-fix f54d8cc | baseline 257267c |
|---|---|---|
| X0 objectives achieved | 6 / 9 (rle 3/3, rpn 3/3, MECH 0/3) | 5 / 9 (rle 2/3, rpn 3/3, MECH 0/3) |
| X0 slow steps kept / reverted / deferred | 8 / 17 / 3 of 28 | 34 / 2 / 1 of 37 |
| X0 objectives stopped by the wall budget | 2 | 0 |
| X1 objectives achieved | 11 / 12 | 11 / 12 |
| X1 decisions to achievement, mean per seed | 10.25, 3.0, 7.25 | 10.75, 7.75, 6.25 |
| X1 slow steps kept / reverted / deferred | 4 / 17 / 3 of 24 | 25 / 8 / 1 of 34 |
| X1 meta-updates applied / records | 2 / 81 | 4 / 110 |
| X2 | 1 pair (seed 0, SW:unique: initial 2, experienced 7 decisions) | no pair (re-scope) |

The fix shows where it was aimed: slow adapter steps are now mostly kept (59 of 71 over X0 + X1 vs 12 of 52).
Competence on these objectives is about the same on both snapshots (X0 6 vs 5 of 9, X1 11 vs 11 of 12); with 3
seeds and the run-to-run variability of §4 no difference in either direction can be read from this.

## 9. Limitations of the baseline

* 2-3 seeds per arm; no p < 0.05 attainable (v3.5); single-seed outcomes are not reproducible (§4).
* X1's sequence confounds position with problem; a within-family learning effect is not established by it.
* MECH (mechanism discovery) was never achieved (0 of 3); X2 (parametric transfer), X3 (baselines A-D) and X4
  (primitive vs skills) did not run on this snapshot, so SV-01..04, SV-10, SV-13, CL-07 and CU-33 have no
  substantive result from 257267c.
* The wall budget of a lease can stop an objective before 20 decisions (3 objectives in X5).
* Bootstrap mix stayed 1.0 (intrinsic reward only for the policy update) throughout; 8 of 233 meta-update records
  over X0 + X1 were applied.
