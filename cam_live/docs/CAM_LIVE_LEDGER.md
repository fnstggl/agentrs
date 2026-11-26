# CAM Live 1.0 — continuation ledger

Read at the start of every session, after `CAM_LIVE_SPEC.md` (operator specification, verbatim),
`CAM_LIVE_ARCHITECTURE.md` (binding design and workstream contract) and `CAM_LIVE_COMPLETION_MANIFEST.json`
(requirement statuses). Append; do not rewrite history.

## Locations

| What | Where |
|---|---|
| Branch | `feat/cam-live`, created 2026-11-26 from `feat/cam-7.0-unified-complete` `3968583` |
| Worktree | `/workspace/camlive` (the main checkout `/workspace/agentrs` belongs to the CAM Unified session: do not touch) |
| Project | `cam_live/` (own `pyproject.toml`; package `cam_live`; nothing imports superlearner / crucible / agentrs) |
| Interpreter | `/workspace/camlive/.venv/bin/python` (layered over `/workspace/agentrs/.venv` through a `.pth`; extra packages only here) |
| Backbone | `Qwen/Qwen3-4B` snapshot `1cfa9a72` under `/workspace/.hf_home` (Apache-2.0, 4.02 B parameters) |
| Chromium | Playwright 1.65, chromium 159 in `~/.cache/ms-playwright` |
| Run state | `<worktree>/runs/cam_live/` (git-ignored, ephemeral disk) |
| Shared GPU lock | `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock` (never wrap pytest in it) |

## Hardware and limits (2026-11-26)

- 1x RTX A6000 48 GB shared with the CAM Unified session (its 1B core uses 17–26 GB at times), 48 CPUs, 125 GB RAM,
  300 GB overlay disk. **No persistent volume**; the Vast rental ends 2026-11-30 03:00 UTC.
- Unprivileged container: no user namespaces, no CAP_NET_ADMIN / CAP_SYS_ADMIN. Available: setuid/setgid, Landlock,
  seccomp. Process isolation therefore uses separate uids; per-destination network filtering is policy-level only.
- Nothing is pushed. Pushing `feat/cam-live` needs the operator's authorization.

## Decisions

1. Standalone project directory `cam_live/` with its own package and tests; CAM Unified / Crucible code is copied
   (with provenance) where it was already verified, never imported.
2. Backbone Qwen3-4B (fits beside the other tenant; Qwen3-8B is a config change). Experiential core preset `50m`.
3. One shared worktree for all build agents with disjoint path ownership (no merges); master-owned contracts:
   `contracts.py`, `nn/api.py`, `config.py`, `learning_api.py`, `harness/__init__.py`, `harness/rpc.py`,
   `diagnostics/`.
4. Part 16 (scientific evaluation) gets its own build workstream `S` after integration; `G` verifies independently.
5. Process isolation by uid (`camlearn`, `cambrowser`, `camsbx`, `camauth`, `camverify`) + Landlock + seccomp.

## Session log

### 2026-11-26 — session 1

- Phase 0: inspected branches / models / hardware; read CAM 7.0 and CAM Unified ledgers and the 2026-11-26 audit
  (45 recorded failures and pitfalls, folded into the architecture document as design constraints).
- Wrote the contracts, tensor API, config, LabWorld diagnostic environment, RPC layer, learning API, collector,
  architecture document and the 158-requirement manifest skeleton (`3c4442e`, `070f096`).
- Launched build wave 1 (12 agents): A model; B1/C1/D1 tensor-level parts; EB/EC/EA/ES harness; F runtime; then
  B2/C2/D2 on the real model.

## Active jobs

| Started (UTC) | Job | Resume |
|---|---|---|
| 2026-11-26 ~21:40 | Build wave 1, run `wf_76b5e7c9-09a` | `Workflow({scriptPath: "<session>/workflows/scripts/cam-live-build-wave1-wf_76b5e7c9-09a.js", resumeFromRunId: "wf_76b5e7c9-09a"})`; script copy `docs/evidence/workflows/wave1.js`; agents commit to `feat/cam-live` as they go and write `docs/status/<Id>.json` |

## Next steps

1. When wave 1 returns: read `docs/status/*.json` and `docs/change_requests/*.md`; write `cam_live/agent.py`
   (integrated `AgentAPI`), `validation.py` (staging), `verify_completion.py`; run the whole suite.
2. Wave 2: `S` (baselines, experiments, transfer) and `G` (independent verification, three lenses), then fixes.
3. Real-backbone runs under the GPU lock; staging validation; preflight; research report; operator list.
