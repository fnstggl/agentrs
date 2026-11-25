# CAM 7.0 Unified — continuation ledger

Read at the start of every session, after `CAM_UNIFIED_SPEC.md` (the user's specification, verbatim) and
`CAM_UNIFIED_COMPLETION.json` (requirement statuses; `superlearner/tests/cam_unified/test_manifest.py` rejects claims
that lack files/tests/verification). Append; do not rewrite history.

## Locations

| What | Where |
|---|---|
| Branch | `feat/cam-7.0-unified` (from CAM 7.0 `f3cee12`; Crucible 4.0 `c07098b` is an ancestor) |
| Package | `superlearner/src/superlearner/cam_unified/` |
| Tests | `superlearner/tests/cam_unified/` |
| Runs (git-ignored, ephemeral disk) | `/workspace/agentrs/runs/cam_unified/` |
| Event archive (lifetime memory) | `/workspace/agentrs/runs/cam_unified/archive/` |
| Interpreter | `/workspace/agentrs/.venv/bin/python` (torch 2.15.0+cu132, transformers 5.24.0, peft 0.22.0, pyarrow 26) |
| Pretrained semantic models | `/workspace/.hf_home/hub` — Qwen3-0.6B (1.5 GB), Qwen3-1.7B (3.8 GB), Qwen3-4B (7.5 GB), downloaded 2026-11-25 |
| Baselines kept reproducible | CAM 6.1 `feat/cam-6.1` (2ebe8f3), CAM 7.0 `feat/cam-7.0` (f3cee12), Superlearner 4.0 `feat/superlearner-4` |

## Hardware and limits (2026-11-25)

- 1x RTX A6000 48 GB, 48 CPUs, 125 GB RAM, 300 GB overlay disk (261 GB free). No persistent volume.
- Vast rental ends **2026-11-30 03:00 UTC**. Everything not pushed or copied off-box is lost then.
- GitHub is the only authorized durable destination so far; each push needs the user's go-ahead. Checkpoints and run
  outputs are not in git.

## Session log

### 2026-11-25 — Stage 0 (recovery and architecture contract)

- Local `feat/cam-7.0` == remote `f3cee12`; no later work found on the remote (branches listed with `git ls-remote`).
- Old-GPU backup remains incomplete (2,461 of 23,308 files; resume script `/workspace/.agentrs-sync/resume-from-mac.sh`).
- Created `feat/cam-7.0-unified`, saved the spec, wrote the 71-item completion manifest and its checker.
- Downloaded Qwen3 0.6B/1.7B/4B (only a 0.6B config had been present).
- Launched a read-only mapping workflow over Crucible OpenWorld, goals/worldgen, research envs, LLM integration,
  superlearner reuse, actors/self-play (results summarised in `CAM_UNIFIED_ARCHITECTURE.md` §0).

- Mapping workflow finished (6 readers + critic; `cam_unified_stage0_maps.json`, `cam_unified_stage0_critic.json`).
  Critic found contract/Crucible mismatches (seconds vs days, seq cursors, cents, 1k-token code args, clock owner);
  fixed in contract v2 (`b68cc0e`) with decisions 5–11 in the architecture doc.
- Confirmed facts that shape the science: no learned or frozen-LLM agent has ever earned independent revenue in an
  OpenWorld economy; the privileged Builder reaches OW-1 in 25–35 simulated days; the non-privileged ExampleInductor
  solves the 8-day ow_tiny open-economy goal on day 4 (day 3 on ow_market); ow_works offers a dense numeric rig
  return inside the full world. CAM 7.0's C-4 negative transfer is confounded by restored replay + AdamW state.

## Active jobs

| Started (UTC) | Job | Where |
|---|---|---|
| 2026-11-25 07:3x | Wave-2 build workflow (S Stage-1 real OpenWorld path, A2 decoder + demo-init, Y integrated capability tests, X foundational experiments, F2 distributed CAM), each verified by H then fixed; worktrees `/workspace/wt2/<id>` on `ws2/<id>` | workflow script `/tmp/claude-0/-workspace/8a8c4e56-2384-4271-b874-3a6441f3bc23/scratchpad/wave2.js` (persisted copy under the session's workflows/scripts/) |

Resume after an interruption: `Workflow({scriptPath: <persisted script>, resumeFromRunId: <run id in the table below>})`; worktrees
and their commits persist on disk. Completed: wave 1 `wf_c9e0d81b-aaa` (merged).

## Decisions

(see `CAM_UNIFIED_ARCHITECTURE.md` for interface decisions and their reasons)

### 2026-11-25 — Wave 1 merged (03:15–07:05 UTC)

- Workflow `wf_c9e0d81b-aaa`: 24 agents (8 build, 8 independent verify, 8 fix), 9.5 M subagent tokens, 4.0 h. Every
  workstream's tests passed under its verifier; verifiers reported 36 major and 84 minor issues, which the fix passes
  addressed. Branches `ws/A`..`ws/I` merged into `feat/cam-7.0-unified` (no overlapping files, no conflicts; merge
  commits `373916c`..`88d14cf`).
- Scale of the merged code: ~35k lines of production code in `cam_unified/` plus tests.
- Measured: main backbone 114.6 M params (workstream-B total 150.4 M), large 306 M; main fixed-J=4 train step 155 ms
  (B=8, ctx 377, bf16, 5.3 GB); Qwen3-1.7B encode ~24–30k tok/s, decode ~20 tok/s at B=1 and ~150 tok/s at B=8;
  archive ingest 8.4k events/s over 1 M events (389 B/event); index search over 1 M keys 10–22 ms; toy actors-only
  1.9k / 5.9k / 9.5k / 11k decisions/s for 1/4/8/16 actors; Builder reaches OW-1 on day 27 (ow_market seed 1).
- Biggest integration gaps (from the fix reports): no CAM decision has run in a real OpenWorld yet; the untrained
  decoder emits up to 1,536-token degenerate strings (533 s per 4 rounds); capabilities A03/A05/A06/A08/A09/E03 are
  shown only with doubles; distributed learning only with a toy GRU policy.
- Contract v3 (`6a67351`, architecture §13): the 50 wave-1 change requests folded in additively. Manifest statuses
  folded conservatively (claims a verifier did not support stay IN_PROGRESS; `verified_by` empty until the
  acceptance-level H pass).
- Merged suite (merged code + contract v3, 7 parallel groups): 735 passed, 3 failed, 13 skipped. Failures, all
  assigned to wave 2: `semantic/test_objectives.py::test_code_execution_matching_generalises_to_held_out_programs`
  (statistical threshold on 8 held-out programs; passed on 2 re-runs; A2),
  `world/test_world_learning.py::test_learns_delayed_versus_immediate_effects` (delayed effect under-learned in 2 of
  4 runs: 0.43 vs 1.0; Y to diagnose), `integration/test_i_model.py::test_count_parameters_dedupes_shared_modules_and_separates_semantic`
  (doubles test now auto-loads the merged real Qwen3-1.7B; S). Logs: `runs/cam_unified/logs/suite_v3/`.
- Lesson: never run the suite under `flock gpu_heavy.lock` — tests take that lock themselves (self-deadlock).

## Next steps

1. Wave 2 (running): Stage-1 real OpenWorld path (S), decoder length control + demo-init (A2), integrated capability
   tests (Y), Part 17 experiments 1–3 (X), distributed CAM on OpenWorld (F2).
2. Merge wave 2; acceptance-level H pass over A01–A22 (`superlearner/tests/cam_unified/acceptance/`).
3. Stage 3 main-model training; Stage 5 OpenWorld lifetimes (OW-1..OW-7, Exp 4) with baselines (Part 19).
4. Before 2026-11-30 03:00 UTC: push all code; decide an off-box destination for checkpoints/results with the user.

