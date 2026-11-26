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


### 2026-11-26 — Wave 2 interrupted (usage limit), work recovered

- Wave-2 workflow `wf_2a13836b-ccd` stopped at the weekly usage limit; no build agent returned, so no verify/fix pass
  ran. All worktree work (S, A2, X, Y, F2) was committed as-is (uncommitted files in RECOVERY commits), pushed as
  backup branches `ws2/*`, and merged into `feat/cam-7.0-unified` without conflicts. NONE of it is independently
  verified; manifest statuses are unchanged from wave 1. Full test suite not re-run after the merge.
- Small run summaries: `superlearner/docs/evidence/wave2_recovery/`. Large run data (~17 GB checkpoints under
  `runs/cam_unified/`, plus CAM 7.0 / 6.1 runs) is NOT in git; file list + sha256 are in that evidence dir.
- Resume: re-run wave 2 from the merged state (S: the 200-decision real ow_tiny run is not done; only 6-decision smoke
  runs exist), then verify/fix, then A01–A22 acceptance.

### 2026-11-26 — Session 3: recovery, independent audit, complete implementation (branch `feat/cam-7.0-unified-complete`)

Governing documents added (verbatim): `CAM_UNIFIED_SPEC_RECOVERY_2026-11-26.md` (recovery/audit directive),
`CAM_UNIFIED_SPEC_META_REWARDS.md` (meta-learned rewards, credit, autonomy — mandatory), `CAM_UNIFIED_SPEC_DOM.md`
(1B core, CAM strategic dominance). Manifest v2: 147 requirements (+76: OB, UR, SE, IR, CR, KV, AU, CD, EX-A..K,
RPT-01, VER-01, DOM-01..13). Executable checker: `python -m superlearner.verify_completion [--run-tests --only ids]`
(claims vs files, resolvable tests, recorded JUnit passes in `docs/evidence/test_results/`, integration entry,
verified_by, experiment verdict files; engineering vs scientific reported separately). At creation: 0/147
engineering-complete — the earlier IMPLEMENTED claims had never had test results recorded.

- Branch created from `feat/cam-7.0-unified` 72c97c8 (preserved untouched) + merged Crucible 5.0
  `origin/feat/crucible-5-openworld` db70810 (an interrupted multi-agent build pushed at 08:49 UTC by another session).
  Push credential: root-only helper (see memory); every milestone pushed.
- Audit workflow `wf_038852ca-d04` (11 independent auditors + synthesis critic) launched 09:02 UTC.
- Implemented while it ran (all new files, tests recorded):
  * `credit/exact.py`, `credit/estimators.py`: exact tabular MDPs; REINFORCE/all-actions/COCOA/HCA/RUDDER verified
    unbiased against enumerated gradients; COCOA variance 1.57 vs REINFORCE 19.75 (key-to-door d6).
  * `credit/meta_exact.py`: Zheng-style meta-gradient through K inner updates; exact == finite differences;
    detaching the inner reward zeroes it; sampled truncated estimator cosine 0.9997; learned reward values the costly
    key pickup (+1.26) and beats extrinsic-reward training (J 1.53 vs 1.12) — Exp J's essential test, tabular.
  * `selfeval/evaluator.py` (verified-outcome success/hazard/magnitude/resource heads, calibration report),
    `selfeval/intrinsic.py` (R_psi, lifetime value), `credit/heads.py` (outcome encoder, hindsight model, RUDDER
    ReturnDecomposer with discounted conservation).
  * `envs/experience.py`: observation-only Crucible 5.0 adapter, three channels (actor / released labels /
    private), oracle-normalised verified utility for rapid families (none defines success upstream).
  * `core/semantic_xattn.py` (DOM-05) + `superlearner` scale: 1,081,244,162 core params.
  * `semantic/decoder_xattn.py` (DOM-03): hooked multilayer gated latent cross-attention into Qwen decoder layers.
  * `autonomy/goals.py`, `autonomy/curriculum.py`: observation-grounded self-goals, operator-side verifier, C_chi.
- Crucible 5.0 findings (audit + own probes; NOT fixed upstream, worked around CAM-side): mech `rig.episode` payload
  leaks `family:seed` (25/25 families incl. test split); rapid `query{target:'spec'}` leaks family/seed/split/engine/
  category/difficulty (36/73); six multi-agent rapid families broken (a0's result returned to all agents, private
  messages visible); kernel verifiers give false early success after one long advance (adapter now steps <=1 h);
  Tier 2/3 absent; no released-label channel upstream (CAM operator implements one).

#### 2026-11-26 (later) — audit complete, seam, wave 3 launched

- Audit `wf_038852ca-d04` done (12 agents, 4.1M tokens): `docs/cam_unified_audit_2026-11-26.json`; binding plan
  `docs/CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md` (9 workstreams W1..W9, gates G0..G5, experiments E0..E6).
- Seam (W1) `wf_f15eec65-2dd`: commits a358e54, ad769dd, b419e78, tag `seam-0` (pushed). Dev decide + 3 updates
  bit-identical before/after (1,779 tensors). Pre-seam CPU suite: 623 passed, 1 failed (semantic/test_actions
  replay-refusal), JUnit `docs/evidence/test_results/suite_cpu_20261126.xml`.
- Usage: 5-hour window hit 93% before the seam finished; extra usage showed $70.67 spent and then disabled (another
  session on the account may be drawing usage). Wave 3 launched at the 13:52 UTC reset.

## Active jobs (2026-11-26 13:53 UTC)

| Job | Run id | Resume |
|---|---|---|
| Wave 3 build/merge/verify/fix | `wf_477a68f9-804` | `Workflow({scriptPath: "<session>/workflows/scripts/...wave3...js" or /tmp/claude-0/-workspace/20daf834-869f-4bc1-be4b-4d3076bc13b9/scratchpad/wave3.js, resumeFromRunId: "wf_477a68f9-804"})`; a copy of the script is `docs/evidence/wave3/wave3.js`. Worktree branches of W2..W9 persist on disk (`git worktree list`). |
