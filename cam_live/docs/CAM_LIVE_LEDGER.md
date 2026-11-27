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
- **Push authorization (2026-11-26, operator: "Yes, push milestones"):** `feat/cam-live` is pushed to `origin` at each
  committed milestone with `cam_live/tools/push_milestone.sh` (secret scan of every unpushed commit, explicit sha,
  never force). First push: `d9f8dde`. Run data and checkpoints are not in git and still have no off-box copy.

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

### 2026-11-27 — session 1 (continued after context compaction)

- Wave 1 stage 1 finished 01:27 UTC: A, B1, C1, D1, EA, EB, EC, ES, F committed with status files and change
  requests (`docs/status/`, `docs/change_requests/`). Stage 2 (B2, C2, D2 on the real model) started 01:28 UTC.
- **Operator supplied Discord resources (2026-11-27):** application "CAM Live" (id 1575681780462129232), server
  "CAM's lab" (id 1575680989060010034), a bot token and a dedicated account. Operator instructions, verbatim scope:
  run the authorized integration tests now; do **not** launch the public-facing experiment until readiness is
  verified and the operator explicitly authorizes it; credentials stay isolated from the learner, training data and
  logs; operate only in the designated server with scoped permissions; do not choose niche / branding / strategy.
- Credentials are held root-only in `/root/.config/camlive/` (`discord_bot_token`, `discord_account.json`,
  `discord_ids.json`, operator signing key `operator_ed25519{,.pub}`); never in the repository, logs or agent prompts.
  Entry path for a run: `python -m cam_live.harness.credentials set discord_bot_token --run-dir R --from-file ...`
  (vault copy for executors + verifier's own copy).
- Read-only checks against the real API: the token authenticates as the bot user; application flags 565248 = the
  limited presence / guild-members / message-content intents are enabled in the portal; `bot_public` is true;
  **the bot was not a member of the server** (`/users/@me/guilds` = `[]`). Operator was given the OAuth2 URL
  (`scope=bot`, permissions 2269727275740401, guild pre-selected). A background watcher polls once a minute.
- Direct login with the dedicated account (operator asked to try): one attempt from Playwright Chromium stopped at
  Discord's hCaptcha challenge. Not circumvented, not retried. User-account automation stays out of CAM's action
  space (Discord Community Guidelines "no self-bots or user-bots"; spec 12.5; policy rule
  `PL-DISCORD-NO-USER-AUTOMATION`).
- Master additions: `harness/credentials.py`, `harness/discord_membership.py` (verifier-side member list),
  `harness/operator_requests.py` + tool `internal.request_operator` (CAM asks the operator for capabilities; the
  answer returns as an event), tests in `tests/integration/`.
- A separate agent is extending the Discord adapter (server settings, channels, roles, permission report + operator
  scope, membership events by snapshot diff, bounded Gateway window, staging gateway, `tools/discord_live_check.py`).
- **Push blocked by GitHub push protection**: the fake Discord-token-shaped fixture in
  `tests/runtime/test_rt_logs_journal.py` (commit `ed56d46`, local only) is flagged. Fix at the next quiescent point
  (no agent committing): build the fixture at run time and rewrite the unpushed commits `ed56d46..HEAD`, record the
  old→new commit map in `docs/evidence/commit_map.tsv`, then push. Last pushed commit: `d9f8dde`.

## Active jobs

| Started (UTC) | Job | Resume |
|---|---|---|
| 2026-11-26 ~21:40 | Build wave 1, run `wf_76b5e7c9-09a` | `Workflow({scriptPath: "<session>/workflows/scripts/cam-live-build-wave1-wf_76b5e7c9-09a.js", resumeFromRunId: "wf_76b5e7c9-09a"})`; script copy `docs/evidence/workflows/wave1.js`; agents commit to `feat/cam-live` as they go and write `docs/status/<Id>.json` |

## Next steps

0. Discord: when the bot is in the server, store the token in an integration run directory, sign and register a
   live authorization for `tasks/discord_integration.yaml`, run `tools/discord_live_check.py`, save evidence under
   `docs/evidence/discord/`.
1. When wave 1 returns: apply the additive change requests to the master files; write `harness/factories.py`
   (all executors incl. Discord for the exec service), `cam_live/agent.py` (integrated `AgentAPI`),
   `validation.py` (staging), `configs/authorization.yaml`; run the whole suite.
2. Wave 2: `S` (baselines, experiments, transfer) and `G` (independent verification, three lenses), then fixes.
3. Real-backbone runs under the GPU lock; staging validation; preflight; research report; operator list.
