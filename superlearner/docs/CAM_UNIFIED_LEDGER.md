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

## Active jobs

(none)

## Decisions

(see `CAM_UNIFIED_ARCHITECTURE.md` for interface decisions and their reasons)

## Next steps

1. Write the shared interface contract (`cam_unified/interfaces.py`) and architecture document.
2. Launch workstreams A–G in isolated worktrees with explicit file ownership; H verifies.
3. Stage 1 vertical path on the real OpenWorld.
