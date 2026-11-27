# G3/G4 at the 1B dom preset — partial (gate agent cut off by the 5-hour limit, 2026-11-27 ~05:00 UTC)

Measured by the wave3c gate1b agent (code at 847559a + the uncommitted WIP committed with this file):

| run | config | decide B=16,K=4 warm | update of 128 decisions | peak |
|---|---|---|---|---|
| g3_run1 | microbatch 0 (all rows) | 2.03 s | OOM after 72.5 s | 46.9 GB |
| g3_run3 | microbatch 1, SplitOptimizer | 1.99 s | 107.1 s, core updated | 42.1 GB |
| g3_run5 | after "rescore only the selected action" (ba74bdd), microbatch 4 | **1.11 s** | OOM after 15 s | 46.2 GB |

Core 1,004,124,162 trainable parameters (32 x 1536 x 24). Against the G3 criteria (step peak <= 40 GB, decide <= 1.5 s,
update <= 90 s): decide PASSES (1.11 s); the update has not yet met both the memory and the time budget in one
configuration (mb1: 42.1 GB / 107 s). The WIP selects variant "adafactor+mb2+ols2" (Adafactor satellites,
microbatch 2, 2 open-loop starts) - not yet measured. G4 (1B end-to-end on 16 rapid + 2 ow_tiny) not run.
