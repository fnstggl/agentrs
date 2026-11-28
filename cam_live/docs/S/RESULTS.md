# CAM Live — substantive evaluation results (Protocol v3, final snapshot 257267c)

**Status: in progress** (interim commit; sections marked *running* or *not yet run* are updated as the batch
proceeds). Binding plan: `docs/S/PROTOCOLS.md` Protocol v3 (v3.2 configuration, v3.3 families, v3.4 experiments,
v3.5 statistics). Per Protocol v3.5 **no comparison here can reach p < 0.05** (3 paired seeds: smallest two-sided
p = 0.25; 2 seeds: 0.5); everything below is an estimate with its range across seeds or a count. Nothing here is a
significance claim.

## 1. Configuration (every substantive run)

* **Code: one fixed snapshot** `runs/cam_live/eval/snapshots/257267c1c315` = `git archive` of `cam_live/` at commit
  **257267c1c31568b693adf0cbcd0e9f9b547e62e1** (contains the DX round 2 slow-step guard fix, DX-1, G2R-5 and the
  evaluation code of the previous S agent). Every result file records it (`code.snapshot_commit` per run and
  `eval_code` per objective; `code_check.all_match` in each `docs/evidence/S/v3/summary_*.json`).
  X5 (goal-free) needs a driver that 257267c does not contain: it runs on the overlay
  `257267c1c315+evalcd34fd10f503` (agent / learner / runtime code of 257267c, only `cam_live/eval` from cd34fd1).
* **Agent** (`eval.oe_runner.describe`, recorded per run): `CamLiveAgent`, backbone Qwen/Qwen3-4B revision
  1cfa9a7208912126459214e8b04321603b3df60c, frozen, bf16, on cuda; experiential core scale 50m. Parameters:
  frozen backbone 4,022,468,096; experiential core 50,953,765; semantic adapters 16,713,856 (LoRA 11,796,480 +
  latent bridge 4,917,376); reward 949,121; heads 6,055,337 (world 1,022,954, transition 1,977,600, evaluator
  783,192, policy 702,339, downstream 84,195, RUDDER 559,361, COCOA 925,696); persistent trainable total 74,672,079;
  fast-state numbers per kind 16,512. Loss providers: world_model, fast_prior, evaluator, world_model_adapters,
  evaluator_adapters, rudder, cocoa, retention, bridge_grounding; credit assigner on.
* **Open-ended generation on**: `use_backbone=True`, `proposer_max_new_tokens=512`, `proposer_k=4` (backbone
  proposals merged with the schema walker and skills); selection rule `thompson_policy`; `experiment_rate=0.1`;
  adapters, bridge grounding, retention on. Learner settings of `eval.oe_runner.SUBSTANTIVE_LEARNER`
  (min_independent_outcomes 2, warmup 2, batch 8, slow batch 3, adapter step every 3 frequent updates);
  meta-update minimum of 4 independent verified outcome groups; bootstrap mix at its default cold start (1.0 in
  every objective so far).
* **Computer**: `harness.env.ComputerEnvironment` with the harness executors (internal, code sandbox, HTTP),
  authorization by the real `harness.auth.PolicyService` in staging mode; CU: `validation.StagingComputer`
  (+ structured browser and skills). Unavailable permitted tools are recorded per objective (none so far).
* **Scheduling**: `python -m cam_live.eval.oe_batch --plan v3 --concurrency 3 --max-gpu-gb 36`, one parent
  holding the shared GPU lock for at most 810 s per lease (child objective budget 690 s, children killed at
  790 s; measured longest objective hold so far 598 s), up to 3 sequences per lease, one objective per sequence
  per lease, learner state carried between leases by the sequence's own checkpoint. GPU lock: the machine's
  shared lock file `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock` (`CAM_LIVE_GPU_LOCK`, see change request
  S2-1; checked with `fuser` / `/proc/locks` during the first lease, `scratchpad/S2/oe_batch.log`).
* **Budget per objective**: at most 20 decisions; achieved = verified value >= 0.9 (SW: hidden-test pass fraction
  of 60 tests; MECH: held-out R^2 on 80 points). Decisions to achievement are censored at 20.

Evidence: raw results `docs/evidence/S/v3/final_257267c/` (one JSON per sequence, child logs, `batch_log.jsonl`),
summaries `docs/evidence/S/v3/summary_<exp>.json`, per-objective tables `docs/evidence/S/v3/tables.md`
(`python -m cam_live.eval.v3_report --out-dir cam_live/docs/evidence/S/v3 --expect-commit 257267c... --markdown
...` with `CAM_LIVE_EVAL_HOME=runs/cam_live/eval/v5`).

## 2. What ran

| experiment | plan | status |
|---|---|---|
| X0 current capability | E, seeds 0-2, `SW:rle, SW:rpn, MECH:7` | complete (9 of 9 objectives) |
| X1 learning within a family | E, seeds 0-2, `SW:rle, rpn, merge, topk` | *running* |
| X2 transfer (16.8) | initial vs experienced exports of X1, fresh processes, `SW:unique, SW:base, MECH:11` | *not yet run* (needs X1) |
| X3 baselines A, B, D, C | seeds 0-1, X1 plan | *not yet run* |
| X4 CU-33 skills vs primitive | E, seeds 0-1, 6 x `CU:store_entry` + 2 x `CU:record_item` | *not yet run* |
| X5 goal-free (11.4) | E, seeds 0-1, goal-free period of 20 decisions, then X1's plan | *conditional on X1 achievement* |
| X6 meta checks (16.5) on X1 roots | `eval.meta_checks` | *needs X1* |
| X6 integrity battery + trivial world (CPU diagnostic) | seeds 0-4 | complete (engineering diagnostic, §8) |

## 3. X0 — current capability (E, seeds 0-2)

| seed | SW:rle | SW:rpn | MECH:7 |
|---|---|---|---|
| 0 | achieved in 16 decisions | achieved in 5 | not reached in 20 (R^2 0.00) |
| 1 | achieved in 7 | achieved in 4 | not reached in 20 (R^2 0.00) |
| 2 | not reached in 20 (pass 0.00) | achieved in 2 | not reached in 20 (R^2 0.00) |

* Objectives achieved 5 of 9 (never reached 4, interrupted 0; every miss ended at the 20-decision budget).
  By problem: rle 2/3, rpn 3/3, MECH:7 0/3.
* Decisions to achievement censored at 20, mean per seed: 13.7, 10.3, 14.0 (mean 12.7, range 10.3-14.0).
* Executable fraction of chosen actions 0.91 (range 0.90-0.92); backbone-written fraction of chosen actions 0.43
  (0.35-0.48) and of candidates 0.35 (0.32-0.38); repeats of failed actions per objective 0.44 (0-0.67); distinct
  strategies per objective 7.7 (7.3-8.0); mean prediction NLL 0.10 (-1.04-1.23).
* Learning: semantic adapters changed in 9 of 9 objectives; **slow (adapter) steps 37: 34 kept, 2 reverted by the
  guard, 1 deferred**. Meta-updates: 123 records, 4 applied (gradient norms 7.29, 6.21, 0.48, 16.79; reward group
  changed in 3 objectives); skipped: insufficient independent outcomes 85, zero advantage 16, no supported episode
  12, no policy update recorded 6.
* Cost: wall time per objective 390 s (328-429 s per seed); GPU peak per child 12.5 GB.

## 8. X6 integrity battery and trivial-reward world (engineering diagnostic, CPU, debug configuration)

`docs/evidence/S/diagnostics/v5_257267c/` (README). On 257267c the battery's trigger (a LabWorld outcome query
pending or due after >= 3 decisions) occurred in 1 of 5 seeds; there every delivered attack (forged objective
outcome, duplicate valid verdict x5, stale prediction) was refused and the independent verdict was unchanged; in
4 seeds nothing was delivered, so nothing was tested (the aggregate's "not detected" for those seeds is an
artefact, fixed in `eval.integrity.verdict_unchanged`). Trivial-reward world: `lab.tick` / noise chosen in 0 % of
decisions, verified progress 0.52 (0.37-0.67).

## 9. Pre-fix comparison (snapshot f54d8cc + eval b06a6af) — reported separately, never pooled

`docs/evidence/S/v3/pre_fix_f54d8cc/` (README: what was wrong — the slow-step guard measured running-statistics
changes and reverted most adapter steps, DX-1, G2R-5). The batch was stopped between leases at 05:50 UTC.

| | pre-fix f54d8cc | final 257267c |
|---|---|---|
| X0 objectives achieved | 6 / 9 (rle 3/3, rpn 3/3, MECH 0/3) | 5 / 9 (rle 2/3, rpn 3/3, MECH 0/3) |
| X0 slow steps kept / reverted / deferred | 8 / 17 / 3 of 28 | 34 / 2 / 1 of 37 |
| X0 ended by the wall budget of the hold | 2 | 0 |
| X1 objectives achieved | 11 / 12 | *running* |
| X1 slow steps kept / reverted / deferred | 4 / 17 / 3 of 24 | *running* |
| X2 | 1 pair (seed 0, SW:unique: initial 2, experienced 7 decisions) | *not yet run* |
