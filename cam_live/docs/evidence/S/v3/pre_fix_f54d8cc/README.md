# Pre-fix comparison only: Protocol v3 batch on snapshot f54d8cc (+ eval overlay b06a6af)

These are the results of the Protocol v3 batch (`python -m cam_live.eval.oe_batch --plan v3`) that ran on
2026-11-28 from about 02:47 to 05:50 UTC on the snapshot
`runs/cam_live/eval/snapshots/f54d8cc05672+evalb06a6af7dc9c` (learner, runtime and agent code of commit
f54d8cc; measurement code `cam_live/eval` of b06a6af; `SNAPSHOT_COMMIT`). The first objective of `x0/E_s0` ran
on the earlier overlay `+evala372a09ae123` (same agent code, older measurement code: the batch was switched to
b06a6af after it, see `x0/E_s0.json` `code`).

**They are reported only as the pre-fix comparison and are never pooled with the final runs** (snapshot 257267c,
`../` and `cam_live/docs/S/RESULTS.md`).

## What was wrong in this snapshot

1. **Slow (adapter) learning was effectively frozen by a measurement defect of the slow-step guard.** The
   guard compares held-out losses before and after an adapter step; the loss evaluation itself updated running
   statistics (the world model's `KindNormaliser`, the loss providers' target scaler), so the guard measured
   world-model changes of 1-8 % that the adapters had not caused and reverted most steps. Fixed by DX round 2
   (5a59e17: the slow step restores those statistics; COCOA's held-out loss without bootstrap weights), verified
   by G2-learn round 3 at cf46eb1: on the real configuration 11 of 12 slow steps kept (before: 1 of 13).
   In this archive (from the run roots' learning logs, `summary/summary_x0.json` / `summary_x1.json`
   `slow_steps`): X0 28 slow steps - 8 kept, 17 reverted, 3 deferred; X1 24 - 4 kept, 17 reverted, 3 deferred.
2. **DX-1: the agent did not hand its experiments to the credit store** (applied in cf46eb1; experiments now
   reach the credit store and the estimator selection is evaluated).
3. **G2R-5: the runtime miscounted program steps against the limits** (b741142: every external step of a program
   counts toward `max_external_actions`).

## What ran

The batch was stopped by the S2 agent with SIGTERM between two leases (no child running), so no objective was
interrupted. Completed: X0 (E, seeds 0-2, `SW:rle, SW:rpn, MECH:7`), X1 (E, seeds 0-2,
`SW:rle, SW:rpn, SW:merge, SW:topk`, with exports), X2 seed 0: `initial` 3 of 3 objectives, `experienced` 1 of 3.
Not run on this snapshot: X2 seeds 1-2 (and 2 objectives of `experienced_s0`), X3, X4.

Summaries (`python -m cam_live.eval.v3_report --results-dir <this dir> --out-dir <this dir>/summary`):
X0 6 of 9 objectives achieved (SW:rle 3/3, SW:rpn 3/3, MECH:7 0/3: two stopped by the wall budget of the lock
hold, one by the decision budget); X1 11 of 12 (rle 2/3 - one stopped by the wall budget after 8 decisions -,
rpn, merge, topk 3/3); X2 one complete pair (seed 0, `SW:unique`: initial 2 decisions, experienced 7). Per
objective: `summary/tables.md`.

## Files

`x0/`, `x1/`, `x2/` result JSON (as written by `eval.oe_runner sequence`) and child logs; `batch_log.jsonl` (one
row per GPU lease); `summary/` the report of this archive. The run roots (checkpoints, exports, learning logs)
stayed under `runs/cam_live/eval/v4/oe/v3/` (not committed).

`../old_snapshot_48600c4/` holds results of two still older snapshots (737996e: X0 seed 0 before change request
S-6 was fixed; 48600c4: X4 seeds 0-1); they are superseded like this archive and not used in any result.
