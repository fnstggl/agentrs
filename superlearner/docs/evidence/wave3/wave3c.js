export const meta = {
  name: 'cam-unified-wave3c',
  description: 'Wave 3 integration: ordered merge of W3,W4,W6,W2,W5,W7,W8,W9 + dev gates -> exclusive-GPU 1B gate -> 5 independent verifiers -> fix',
  phases: [
    { title: 'Merge', detail: 'ordered merge W3,W4,W6,W2,W5,W7,W8,W9 + gates G1/G2 at dom_dev' },
    { title: 'Gate1B', detail: 'G3/G4 at the 1B dom preset with the GPU to itself' },
    { title: 'Verify', detail: 'independent H verifiers per area (read-only)' },
    { title: 'Fix', detail: 'one fix agent applies verified findings' },
  ],
}
const JOURNAL = '/root/.claude/projects/-workspace/20daf834-869f-4bc1-be4b-4d3076bc13b9/subagents/workflows/wf_477a68f9-804/journal.jsonl'
phase('Merge')
const merged = await agent(`You are the merge integrator for CAM Unified wave 3, working in the main checkout /workspace/agentrs on branch
feat/cam-7.0-unified-complete (W1 already committed there). Interpreter /workspace/agentrs/.venv/bin/python, run from
superlearner/ with PYTHONPATH including /workspace/agentrs/crucible/src. Read superlearner/docs/CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md §3-4.
Workstream reports: W1 (main branch, already committed there), W3 (branch worktree-wf_477a68f9-804-3) and W8 (branch
worktree-wf_477a68f9-804-8) are the "type":"result" lines of ${JOURNAL} (JSON); W1's notes on 1B measurements and change
requests are also in /tmp/claude-0/-workspace/20daf834-869f-4bc1-be4b-4d3076bc13b9/scratchpad/wave3_notes/W1_duplicate_handback.md.
Reports of the six continued workstreams (W2, W4, W5, W6, W7, W9; each names its branch and final commit) are in
superlearner/docs/evidence/wave3/finisher_reports.json (W6's was written by the orchestrator from W6's commits because
its finisher was cut off; W6 = branch cam-unified/W6-pred at 152b7d1). Read fields branch/final_commit/change_requests/
not_done/checks with a small python snippet; do not print the whole file.
Known cross-cutting items to apply during the merge (change requests nobody owned): EventRecord.env_id / goal_id and
DecisionTrace.context_refs must not carry "rapid:<family>#<seed>" (opaque ids; operator keeps the map); pass sem= (W4
semantic_context) to core/reasoner in encode_step/decide/policy_evaluate and through world-model calls; entity type tokens
in encode_step and agent.tokenise_sequences; core.state_gate/sem_proj/xattn/cond_proj in the learner's core groups and the
DOM-08 table; RewardModel built when reward_mode is observation_only and its version stamped; label_episode stamp;
dist/actor uses runtime.merge_by_seq; remaining stray act/execute loops moved onto run_lifetime where cheap.
RESUME SAFETY (a previous run of this step may have been cut off by a usage limit): first run git status, check for
MERGE_HEAD / conflict markers, and git log --oneline 19a7a4c..HEAD for existing "[merge]" commits; continue from where
it stopped instead of redoing merges. After each merge commit, append one line (branch, merge commit, tests run, result)
to superlearner/docs/evidence/wave3/merge_progress.md and commit it, so a later attempt can resume. If a workstream
report is missing from the list above (its finisher failed), still merge that branch at its current HEAD (it holds
committed, coherent work; worktree path .claude/worktrees/wf_477a68f9-804-<n>, branch worktree-wf_477a68f9-804-<n>),
run its tests, and say so. W2 found that run_lifetime delivers every direct result twice (adapter-delivered copy plus
the copy merged from ExecutionResult.observations); fix that during the merge (it is W1's code) with a test.
Merge the workstream branches into feat/cam-7.0-unified-complete in this order: W3, W4, W6, W2, W5, W7, W8, W9
(git merge --no-ff <branch>; message "cam unified complete [merge]: <ws> ..." ending with
"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"). Resolve conflicts preserving BOTH sides' intent; apply the
change_requests the reports list when they are small and clearly correct (note which). After each merge run the
directly affected test directories; after all merges run the full suite (pytest -m "not slow" tests/cam_unified, GPU
tests included, NOT under flock) and record JUnit XML to superlearner/docs/evidence/test_results/wave3_merged.xml
(-o junit_family=xunit1). Then run gates G1 and G2 of plan §4 at dom_dev on the real stack (dom_dev end-to-end through
run_lifetime on 4 rapid training environments: 200 decisions, 20 updates; report DOM-08 gradient table, actor core share,
observation-only integrity scan, action stamps). Fix integration breakage you find (you may edit any file now; keep
commits small and described). Never push. Report: merge commits, conflicts and how resolved, test counts (pass/fail/skip
with failing test ids), gate results with numbers, open problems.
COST DISCIPLINE (usage limits interrupted this wave three times): your whole context is re-sent on every turn and the
prompt cache expires after 5 minutes, so every wake-up after a long wait is expensive. Batch commands; read files with
targeted grep/sed ranges; run long test suites/jobs in the background and do useful work meanwhile instead of repeated
short wait loops; prefer the smallest run that answers the question. Do not babysit diagnostics.`, { label: 'merge', phase: 'Merge' })

phase('Gate1B')
const gate1b = await agent(`You run gates G3 and G4 (plan §4) for CAM Unified at the 1B "dom" preset (32x1536x24 core + Qwen3-4B) in the
main checkout /workspace/agentrs (branch feat/cam-7.0-unified-complete, just merged). Interpreter
/workspace/agentrs/.venv/bin/python from superlearner/ with PYTHONPATH incl. /workspace/agentrs/crucible/src. You have the
GPU to yourself now (no other workstream is running); still take /workspace/agentrs/runs/cam_unified/gpu_heavy.lock for
each heavy job and check nvidia-smi is nearly empty first (kill nothing that is not yours; report other tenants).
Use core/bench.py --full, tests integration/test_memory_budget_1b.py and test_dom_e2e_real.py, stage1/dom_e2e.py (W1).
G3: one full optimiser step peak <= 40 GB (try in order: base; no EMA target; Adafactor for satellites; microbatch 4;
d_satellite 512 — report every variant's peak and time), decide B=16,K=4 <= 1.5 s with candidates sharing the encode,
update of 128 decisions <= 90 s. G4: the 1B system end-to-end through runtime.run_lifetime on 16 rapid training-family
worlds + 2 ow_tiny worlds, observation-only, >= 64 decisions and 2 updates: core hash change, every action cam_eval
stamped, DOM-08 gradient table after one step (blocks 0/16/31, sem projections, Mech C gates, reasoner, beliefs, world
model, intent, decoder xattn, policy, V_int, predictors), KL(pi || pi with I permuted) >> 1e-9, integrity canary scan
clean. Fix what blocks the gates (small described commits ending "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>";
never push). Write evidence to superlearner/docs/evidence/dom/G3G4/*.json and commit it as you go. RESUME SAFETY: a previous attempt
may have been cut off; first check that directory and git log for G3/G4 commits and continue from there. Kill any
leftover GPU job of a previous attempt only if it is clearly one of this gate's (command line under G3G4 or bench.py
--full). Report every number, pass/fail per criterion.
MERGE REPORT:
${merged}
COST DISCIPLINE (usage limits interrupted this wave three times): your whole context is re-sent on every turn and the
prompt cache expires after 5 minutes, so every wake-up after a long wait is expensive. Batch commands; read files with
targeted grep/sed ranges; run long test suites/jobs in the background and do useful work meanwhile instead of repeated
short wait loops; prefer the smallest run that answers the question. Do not babysit diagnostics.`, { label: 'gate1b', phase: 'Gate1B' })

phase('Verify')
const AREAS = [
  { k: 'dom', p: `DOM amendment: DOM-01..DOM-09 and A01/A03. Verify parameter counts of the ASSEMBLED dom model (core vs satellites vs semantic), that Mechanism A (decoder latent cross-attention), B (intent bottleneck + concrete-action evaluation + execution authorisation) and C (gated semantic cross-attention) are wired in the real decision path, that gradients reach early/middle/late core blocks, semantic projections, reasoning, beliefs, world model, intent, decoder conditioning, exploration and policy/value (run the DOM-08 table yourself at dom_dev), Condition G latent swap, and that reasoning iterations change decisions.` },
  { k: 'integrity', p: `Observation-only operation and reward integrity: OB-01..OB-08, IR-01..IR-09, SE-01..SE-06, AU-01..AU-07. Trace every path from environment and evaluator to actor inputs, beliefs, world-model targets, actor losses, archive and traces; plant canaries; confirm labels reach only the trainer; confirm R_psi has no actor-gradient path except the inner update; confirm the meta-gradient pathway exists end to end on the real model.` },
  { k: 'pred', p: `World model, beliefs, imagination and credit: A05..A09, C01..C05, CR-01..CR-09, KV-01..KV-03. Run the causal tests (6.6) and the adaptation tests on the real dom_dev stack with several seeds, check open-loop multistep training, belief fusion calibration, VOI from the world model, and the CAM-scale R-HCA/COCOA candidate estimator against the exact references.` },
  { k: 'learn', p: `Integrated learning and runtime: A13, A16, A17, A20, A21, A22, D05..D08, UR-03, UR-04, CD-01..CD-03. Run a dom_dev lifetime through runtime.run_lifetime on rapid families (observation-only), confirm core weights change from real experience and the next decisions change, PPO/V-trace routing, per-objective gradient shares, checkpoint save/load (weights-only in a fresh process), versions on every record, single-loop property.` },
  { k: 'evid', p: `Experiments and governance: X01..X08, EX-A..EX-K, DOM-11, DOM-12, VER-01, RPT-01, G10. Check that experiment drivers run the REAL model only through run_lifetime in observation_only with provenance; pre-registrations exist and predate any confirm run; statistics are correct; verify_completion rejects unsupported claims (try to fool it).` },
]
const verdicts = await parallel(AREAS.map(a => () => agent(`You are an INDEPENDENT VERIFIER (workstream H) for CAM 7.0 Unified. Read-only on tracked files (scratch dir
/tmp/claude-0/verify_${a.k}/; runs/ is writable); no commits. Repository /workspace/agentrs, branch
feat/cam-7.0-unified-complete (just merged). Interpreter /workspace/agentrs/.venv/bin/python from superlearner/ with
PYTHONPATH incl. /workspace/agentrs/crucible/src. GPU shared: flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock for
ad-hoc jobs >4 GB (never pytest under it). Plan: superlearner/docs/CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md; specs:
superlearner/docs/CAM_UNIFIED_SPEC*.md; manifest superlearner/docs/CAM_UNIFIED_COMPLETION.json.
Do not trust reports or test names: read code, trace runtime paths, execute checks on the real model.
AREA: ${a.p}
Return findings ranked by severity: for each, file:line, what is wrong, the concrete failing check you ran, and the fix;
plus, for every manifest id in your area, the status the evidence supports (NOT_STARTED/IN_PROGRESS/IMPLEMENTED/
INTEGRATION_TESTED/EMPIRICALLY_DEMONSTRATED/FAILED_EMPIRICALLY/BLOCKED) with the test ids / evidence files that support it.
COST DISCIPLINE (usage limits interrupted this wave three times): your whole context is re-sent on every turn and the
prompt cache expires after 5 minutes, so every wake-up after a long wait is expensive. Batch commands; read files with
targeted grep/sed ranges; run long test suites/jobs in the background and do useful work meanwhile instead of repeated
short wait loops; prefer the smallest run that answers the question. Do not babysit diagnostics.`,
  { label: `verify:${a.k}`, phase: 'Verify' })))

phase('Fix')
const fixed = await agent(`You are the fix integrator for CAM Unified wave 3 in the main checkout /workspace/agentrs (branch
feat/cam-7.0-unified-complete). Independent verifiers reported the findings below. Fix every finding that is a real
defect (verify each yourself first; skip with a reason if it is not real), highest severity first, with small
described commits ending with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Do not weaken tests. Never
push. RESUME SAFETY: a previous attempt may have been cut off; check git log for recent fix commits first. Re-run the affected tests and record JUnit XML to superlearner/docs/evidence/test_results/wave3_fixed.xml
(-o junit_family=xunit1). Report each finding as fixed / not-real / not-fixed (why), test counts, and the per-requirement
status table the verifiers proposed, adjusted for your fixes.
MERGE REPORT:
${merged}
1B GATE REPORT:
${gate1b}
VERIFIER FINDINGS:
${verdicts.map((v, i) => v ? `--- ${AREAS[i].k} ---\n${v}` : `--- ${AREAS[i].k} --- (verifier failed; no findings)`).join('\n')}
COST DISCIPLINE (usage limits interrupted this wave three times): your whole context is re-sent on every turn and the
prompt cache expires after 5 minutes, so every wake-up after a long wait is expensive. Batch commands; read files with
targeted grep/sed ranges; run long test suites/jobs in the background and do useful work meanwhile instead of repeated
short wait loops; prefer the smallest run that answers the question. Do not babysit diagnostics.`, { label: 'fix', phase: 'Fix' })

return { merged, gate1b, verdicts, fixed }
