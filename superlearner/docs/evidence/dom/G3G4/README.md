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

## g3_run6 (orchestrator, 2026-11-27 07:20-08:00 UTC, code e71d62e; GPU shared with another tenant using 2-12 GB)

Real `build_agent("dom")`, 16 rapid worlds, 8 rounds (128 decisions, K=4 candidates on every action), then two
`IntegratedLearner.step` updates per variant over the 128 decisions (2nd = optimiser moments resident):

| variant | update 1 | update 2 | peak | G3 (<= 40 GB and <= 90 s) |
|---|---|---|---|---|
| adafactor (AdamW core) + mb1 + ols2 | 240.9 s, 34.3 GB | OOM (44.2 GB) | 44.2 GB | FAIL |
| adafactor (AdamW core) + mb2 + ols2 | 79.6 s, 36.8 GB | 76.5 s, 43.9 GB | 43.9 GB | time PASS, memory FAIL |
| adafactor all + mb1 + ols2 | 105.9 s, 36.0 GB | 125.9 s, 35.4 GB | 36.0 GB | memory PASS, time FAIL |
| adafactor all + mb2 + ols1 | 127.0 s, 35.9 GB | 113.2 s, 34.8 GB | 35.9 GB | memory PASS, time FAIL |

Core hash changed in every successful update; core gradient shares world 0.44, adapt 0.17-0.45, actor 0.11-0.40.
Warm decide (B=16, K=4) 2.49 s in this run (1.11 s in g3_run5 at ba74bdd) - the difference coincides with another
tenant's GPU jobs; decide is not a clean measurement here. Verdict: G3 NOT MET in one configuration; the memory
budget is met with Adafactor everywhere, the time budget with an AdamW core, not both at once.

## G4 - 1B end-to-end through runtime.run_lifetime (orchestrator, 2026-11-27 07:23-07:28 UTC, code c3b71b8)

`stage1.dom_e2e --preset dom --rapid 16 --ow 2 --rounds 4 --update-every 2 --require-cam-eval on` with the G3
memory configuration (Adafactor all groups, microbatch 2, 2 open-loop starts, gradient checkpointing). Evidence
`g4_dom.json`. 1,573,522,304 trainable CAM parameters (core 1,004,124,162) + frozen Qwen3-4B.

| G4 criterion | result |
|---|---|
| >= 64 decisions, observation-only | 72 decisions over 18 worlds; Transition.rewards None every round; record rewards 0 - PASS |
| 2 updates, core changes | 2 updates (149.9 s, 43.3 s; peak 29.6 / 34.3 GB); core hash 9312812b -> 000a1e5e - PASS |
| every action cam_eval stamped and authorised | 72/72 stamped, 72/72 HMAC-verified, authorisation required - PASS |
| DOM-08 gradient table after a step | 13/14 groups nonzero after update 2 (blocks 0/mid/last, sem_proj, Mech C gate, reasoner, beliefs, world model, intent, decoder xattn, exploration, policy, V_int); predictors zero (no label resolved in 4 rounds) - PASS except predictors |
| KL(pi || pi with I permuted) >> 1e-9 | mean 1.7e-3 (min 1.1e-4, max 1.0e-2) at initialisation - PASS |
| integrity canary scan | 32 canaries, 90 records, 108 trace lines: clean - PASS |

Not met / caveats: warm decide 8.3 s at B=18 with 2 OpenWorld worlds (string decoding dominates: generate 6.5-10 s;
the 1.5 s G3 decide budget was for 16 numeric-tool rapid worlds); actions mostly wait/noop (untrained policy);
predictors and psi untrained (no labels within 4 rounds; rapid episodes are 60 ticks).
