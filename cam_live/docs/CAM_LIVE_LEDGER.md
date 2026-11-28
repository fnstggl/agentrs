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

### 2026-11-27 — session 2 (Opus 5.5 takes over as master, 04:25 UTC)

- History rewrite d9f8dde..ebf1678 to remove a token-shaped test fixture (map: `docs/evidence/commit_map.tsv`); pushes
  now use the CAM Live GitHub credential (`/root/.config/camlive/gh_token`, helper `gh_cred_helper.sh`, root-only).
- Independent verification lenses G-sec, G-env, G-rt, G-nn, G-pr, G-cl (`docs/verification/G_*.md`) found critical
  and major defects in every workstream; fix agents EA, E, F, A, BC, D fixed them (`docs/<ws>/FIXES.md`).
- Master decision: per-member Discord events (`get_member_events`, `listen_events`) need `data_permissions.member_list`.
- **Discord live integration check on "CAM's lab": 50/50 steps ok, server restored** (`docs/evidence/discord/`).
- Integrated agent `cam_live.agent` (M-int); staging validation on Qwen3-4B + 50m: pathway works, no real-model
  meta-update yet. Round 2 (INT2) running: grant consumption (M-int-6), verifier wiring, register_objective (C-10),
  grounded candidates, checkpoint promotion/export, real-model meta-update, separated-mode staging run.
- Evaluation workstream S running (`cam_live/eval/`: baselines A–E, 16.2–16.10, transfer, scaling).
- Honest negative results so far: learned reward from scratch hurts in LabWorld (−0.35); RUDDER/COCOA fail the
  held-out selection gate (default credit = simple); fast-layer count advantage over GRU not significant.

### 2026-11-27 — session 3 (17:45 UTC resume after the disk filled)

- ~10:16 UTC the 300 GB disk filled (evaluation v2 runs: per-lifetime checkpoints with Adam state, 150m scaling);
  every agent stopped. Recovered: 300 → 134 GB used. Results archived (`docs/evidence/S/raw/`), then
  `tools/prune_runs.py` removed 132 GiB of intermediate checkpoints/optimizer states and ~33 GB of finished agents'
  scratch; new `cam_live.storage` (DiskGuard + retention) to be wired into runtime / learner / eval runner.
- Audit at resume (checker): 158 reqs — 32 complete, 23 verified with checker gaps, 65 partial (mostly pre-fix
  verdicts), 3 failed (pre-fix), 35 unverified. Discord bot re-checked read-only: in "CAM's lab", no other guild.
- **Operator evaluation redirection (verbatim in the session; binding):** synthetic envs are diagnostics only;
  substantive evaluation = full-size agent with open-ended backbone action generation, open-ended objective classes,
  intelligence-acquisition and transfer metrics, baselines A–E at full size. When ready, CAM receives the operator's
  Discord objective verbatim (`tasks/discord_100_planning.yaml`) and produces its own research, plan and operator
  requests; the operator then supplies resources. Claude Code must not make the plan.
- Running: INT2 (storage wiring, real-model meta-update pathway, near-random diagnosis, open-ended generation,
  grounded computer use, separated run, read-only live Discord observation run, planning-phase task file),
  S (protocol v3 + open-ended evaluation families on the full-size agent), MAN (manifest reconciliation + checker).

### 2026-11-27 — session 4 (23:00 UTC, after INT2 / LRN / MAN)

- MAN: manifest 198 requirements (40 added: amendments, evaluation redirection, OP-01..11), strict checker. At
  b663d5e: 6 complete, 113 implemented-needs-reverification (verdicts older than the code), 72 partial, 7 missing.
- INT2 (45a8a9b..66b4fce): covered-question fix (meta-updates were skipped: 27/29), program modality (INT2-9
  workaround), post-approval result recorded, LRN-1/LRN-6 applied; S-6 probe 16/16 valid backbone actions but the
  software objective is never attempted (the model asks the operator for a file-writing capability it has);
  real-config staging validation executed, not passed (approvals / resource limit / meta-update); whole suite
  1468 tests: 1378 passed, 90 skipped, 0 failed (per directory). Read-only live Discord observation run not
  executed (launch refused by the session permission system; needs the operator).
- LRN (8acc55f..b663d5e): selection rule thompson_policy (default), standardised reward mixing, Delta V fix,
  verified-progress fixes, promotion as non-inferiority test, INT2-8 restore fix. Debug scale: choice quality
  0.311 -> 0.375; no lifetime-to-lifetime improvement. Real model: the live update changes through R_psi only with
  mix fixed at 0.5 (default mix stays 1).
- Running from ~23:15 UTC: INT3A (S-6b complete tool list in the prompt + why the model does not write code,
  decision cost, LRN-3, INT2-9 proper, mix conclusiveness), INT3B (pending outcomes at completion, staging
  rehearsal of the planning phase, `cam-live plan-report`, real-config staging validation, one-process suite),
  HX (unclaimed harness requirements IH-13..17, CU-19/25/26/35..38, OP-04/07/09), DOC (DL-02, DL-04, DL-20,
  IN-05/CM-13, IN-06), G2-learn (re-verification of A + D at b663d5e+, nn/ and credit/ frozen).
  GPU queue gpu_queue3 (X4 CU-33 skills vs primitive, seeds 0-1) on the pre-fix snapshot 48600c4: diagnostic only.
- Plan: code freeze after INT3A/INT3B/HX (~05:00 UTC 2026-11-28); then G2 lenses for B/C (decision), F/M
  (runtime + integration), harness; Protocol v3 X0-X5 on the frozen snapshot (S); live planning phase: fill the
  operator values of tasks/discord_100_planning.yaml, sign the planning authorization (no publication
  pre-authorized), `cam-live run`, export CAM's own plan unedited with `cam-live plan-report` for the operator.

### 2026-11-28 — session 4 (continued, ~02:40 UTC): freeze for decision / learning / evaluation

- Done: INT3A (S-6b fixed: sandbox tools were unavailable in the probe root, required-marker and nested-schema
  causes; SW:rle achieved through the runtime in 2 decisions; KF-E01 credit now feeds training; info_gain key;
  LRN-3; INT2-9; G2L-8 cause), INT3B (settling of pending outcomes, GPU lease cycling 840 s, operator flag
  --accept-ephemeral-storage, plan-report, HX-5, float redaction, planning rehearsal in staging: 30 decisions,
  5 operator requests, 3 Discord writes queued none executed, 0 notes written by CAM; staging validation 15/16),
  HX rounds 1-3 (OP-09 hard constraints, IH-15 data use, X-PLATFORM-ADAPTER, root sandbox host, isolated service
  boot), DX (G2L-1..5, CL-10, INT3A-5; DX-1 agent hook not applied: flaky wiring test 5/28, selection would still
  not run), DOC (DL-02, DL-04, DL-20, IN-05, IN-06), G2-learn round 1 (24 verified / 2 partial), G2-harness rounds
  1-3 (now 44 verified / 5 partial / 0 failed; R3-M1 root write symlink race open).
- Freeze for decision, learning and evaluation: **f54d8cc**. Running: S (Protocol v3 X0-X6 on f54d8cc, results by
  2026-11-29 16:00 UTC), G2-decision (B, C, RC-01/02/08), G2-learn round 2 (DX changes), HX round 4 (R3-M1 root
  write race, audit anchor across restarts, OP-09 false refusals; may touch runtime/logs.py atomic writes,
  supervisor writes in runtime/cli.py, harness/credentials.py).
- After HX round 4: G2-harness round 4, then G2-runtime (F + CL-06) and G2-integration (M + docs) at the harness
  freeze; then the operator launches the planning phase (`tools/launch_live.sh --accept-ephemeral-storage
  tasks/discord_100_planning.yaml 8 --read 'https://*' --read 'http://*'`; signing is the operator's act — the
  permission system refused agent-signed records).
- Default config: nn.max_prompt_tokens 6144 (a7b8fa4). History rewrite of unpushed c0b408b..8172929 (fake canary
  literal, push protection); map in docs/evidence/commit_map.tsv.

### 2026-11-28 — session 4 (continued, ~05:45 UTC): FINAL FREEZE 257267c

- Fixed since f54d8cc: DX round 2 (slow-step guard measured in different units: running statistics restored after
  the slow step's loss evaluation, deterministic COCOA validation, noise-based tolerance; real configuration 11/12
  slow steps kept — was 1/13), DX-1 / DX-2 applied (experiments reach the credit store; auth hash in checkpoints),
  G2L-10 and G2D-1 (master), HX rounds 4-5 (safefs for every root write / read, audit anchor, export-checkpoint and
  plan-report as root, CAPTCHA false refusals), HX-6 (master), INT3B round 2 (membership counted from the
  experiment start, program steps counted against limits, secondary metrics, next-objective verification, task
  disguises, plan-report byte-exact + escaped, validation defaults, approval grant reuse, limit refusal class),
  INT3B-8 (master), push secret scan over every blob of every commit (master, G2I-1/2), retention symlink fix.
- Verification state: G2-learn 27/27 verified (cf46eb1); G2-harness 47 verified / 2 partial, GO for the planning
  launch (8f6c615); G2-decision 25 / 7 partial (f54d8cc; round 2 at ef0f3ff running); G2-runtime 20 / 11 partial
  (a5ab772; round 2 at 257267c running); G2-integration 9 / 18 / 1 failed LX-11 (a5ab772; round 2 running).
- Evaluation: S2 restarts Protocol v3 on 257267c (results runs/cam_live/eval/v5; the f54d8cc runs are archived as
  the pre-fix comparison); deadline 2026-11-29 16:00 UTC. Whole default suite on an archive of 257267c running.
- Planning launch (operator): `bash cam_live/tools/launch_live.sh --accept-ephemeral-storage
  cam_live/tasks/discord_100_planning.yaml 8 --read 'https://*' --read 'http://*'`; read the report with
  `cam-live plan-report --run-dir R --out <root-only dir>/plan.md`.

### 2026-11-28 — session 4 (continued, ~08:30-09:00 UTC): LIVE planning run; core-intelligence amendment

- Operator: "I GIVE FULL APPROVAL TO BEGIN ANY LIVE RUN ... CAM HAS MY FULL AUTHORIZATION TO DO ANYTHING IT NEEDS".
  Signed on that instruction: record for tasks/discord_100_planning.yaml (window 08:34-16:34 UTC; read https/http;
  writes to the operator's own server pre-authorized; everything else queued; hard constraints unchanged).
  First launch blocked by the preflight on publish.* tools the task does not authorize (fixed d1d06dc); relaunch:
  LIVE: READY, bot identity / membership / intents / permissions confirmed live; run dir
  runs/cam_live/live/discord-100-planning-20261128T083454Z (learner camlearn, shared GPU lease).
- Amendments (verbatim: docs/amendments/2026-11-28_autonomy_and_core_intelligence.md): operational autonomy
  (standing authorization for routine reversible operations, capability audit, infrastructure list) and seven core
  capabilities (learned rewards, imagination / multistep planning, persistent latent state, retrospective credit,
  strategic invention, skills, goal-free learning) with four evidence levels each.
- Agents from ~08:55 UTC (report by ~16:30): RW, IMP, PS, CR, ST (agent.py owner + integrator), SG, AU. S2 finishes X0 / X1
  on 257267c as the pre-amendment baseline, then waits for the amended snapshot.

### 2026-11-28 — live planning run, attempts 1 and 2 (09:00-10:50 UTC)

- Attempt 1 (run dir runs/cam_live/live/discord-100-planning-20261128T083454Z, code = worktree at d1d06dc): 36
  decisions, 8 external actions. ST's read-only analysis: 33 of 35 chosen actions from the schema walker, 14
  `http.request` refused (not in authorized_tools, offered anyway), 4 navigations to the pattern "https://*/" refused;
  the backbone's answer truncated in 34 of 35 decisions (256 tokens for 8 alternatives) so no note was ever proposed.
  CAM made 2 operator requests ("Access to a usable URL destination"; "Manage roles") — answered / granted with facts
  only. 3 x discord.modify_guild (default_message_notifications) executed under the operator's pre-authorization.
  Paused at 09:31:50 (ckpt_00000038), then the supervisor was stopped (SIGTERM) because a queued stop was not applied
  while paused; on restart the runtime applied that stale stop and the run became final (a stopped run is never
  restarted, by design; same-objective re-assignment is applied only by a running runtime).
- ST hotfix a0f3411 (only authorized tools offered, no wildcard values, 4 samples x 2 alternatives x 384 tokens).
- Attempt 2 (run dir runs/cam_live/live/discord-100-planning-20261128T104208Z) started ~10:50 UTC from a clean
  snapshot of a0f3411 (runs/cam_live/live/code_a0f3411), same signed record (bound to the task file; window to
  16:34 UTC). The worktree's uncommitted agent WIP never reaches it.
- Known runtime issues (to KNOWN_FAILURES): commands are applied only at step start (a pause took ~9 min under GPU
  contention); a paused runtime did not apply a queued stop within 10 min; no warm start of a new run from a stopped
  run's learner checkpoint (needed to carry learning into post-amendment re-runs).
- Load reached ~95 on 48 cores (agent diagnostics) and starved every GPU holder; CPU rule issued to all agents
  (OMP_NUM_THREADS<=4, nice 10, one heavy job each); the container does not permit negative nice values.

### 2026-11-28 — redirection (~18:00 UTC): fundamental intelligence and autonomy

- Live planning attempts 1-3 (276 decisions): 0 websites visited, 0 notes / plans, repeated discord.modify_guild, many
  operator requests (docs/evidence/live/planning_20261128/). Amendment results: RW (promotion now needs to beat 20
  untrained controls; no learned reward earns it), IMP (imagination changes decisions when properly paired, share_tv
  0.056, no outcome gain), PS (persistent state: no benefit, default off), CR (forced RUDDER result confounded; being
  wrapped up), ST (hotfix; information value ~0 on the real model), SG (skills fixed; goal-free paired -5.3 [-14.7, 5.0]),
  AU (standing profile; G3A-2 bypasses fixed, re-check pending). G3-act: same-seed runs were not paired (os.urandom
  agent id; fixed 831cc85 / b74b312); the full-size predictor ignores its context (3.3e-4 relative belief change).
- Operator redirection (docs/amendments/2026-11-28b_fundamental_intelligence_and_autonomy.md): critical path, zero
  routine intervention, competent baseline, anti-repetition, simplified process.
- Reorganised: evaluation batch (S2) stopped and written up; CR wraps up (CPU only); PS = primary research problem
  (context insensitivity), GPU priority; new CB = competent baseline (frozen backbone vs CAM vs CAM after learning),
  residual learning over a competent prior, operator requests out of the loop, generic anti-repetition, validation on
  the full-size model; G3-act re-checks AU's fix on CPU only. Few concurrent GPU users (PS, CB).

## Active jobs

| Started (UTC) | Job | Resume |
|---|---|---|
| 2026-11-26 ~21:40 | Build wave 1, run `wf_76b5e7c9-09a` | `Workflow({scriptPath: "<session>/workflows/scripts/cam-live-build-wave1-wf_76b5e7c9-09a.js", resumeFromRunId: "wf_76b5e7c9-09a"})`; script copy `docs/evidence/workflows/wave1.js`; agents commit to `feat/cam-live` as they go and write `docs/status/<Id>.json` |

## Next steps (session 2)

1. Collect INT2 and S; re-run independent verification (new G lenses on integration + evaluation + re-check of
   fixed defects); merge verdicts; `verify_completion --merge-status`.
2. Rename duplicate test basenames (`tests/predict/test_providers.py` vs `tests/credit/test_providers.py`) so the
   whole suite collects; full-suite JUnit.
3. Preflight for `tasks/discord_100.yaml` against live services; research report; operator list.

## Next steps

0. Discord: when the bot is in the server, store the token in an integration run directory, sign and register a
   live authorization for `tasks/discord_integration.yaml`, run `tools/discord_live_check.py`, save evidence under
   `docs/evidence/discord/`.
1. When wave 1 returns: apply the additive change requests to the master files; write `harness/factories.py`
   (all executors incl. Discord for the exec service), `cam_live/agent.py` (integrated `AgentAPI`),
   `validation.py` (staging), `configs/authorization.yaml`; run the whole suite.
2. Wave 2: `S` (baselines, experiments, transfer) and `G` (independent verification, three lenses), then fixes.
3. Real-backbone runs under the GPU lock; staging validation; preflight; research report; operator list.
