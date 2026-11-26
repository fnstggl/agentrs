export const meta = {
  name: 'cam-unified-wave3',
  description: 'Wave 3: 9 workstreams (W1 in main checkout, W2-W9 in worktrees) -> ordered merge + gates -> independent verification -> fix',
  phases: [
    { title: 'Build', detail: 'W1 main checkout + W2..W9 isolated worktrees, per CAM_UNIFIED_REDESIGN_PLAN §3' },
    { title: 'Merge', detail: 'ordered merge W3,W4,W6,W2,W5,W7,W8,W9 + gates G1/G2 at dom_dev' },
    { title: 'Verify', detail: 'independent H verifiers per area (read-only)' },
    { title: 'Fix', detail: 'one fix agent applies verified findings' },
  ],
}

const COMMON = `
You are an implementation workstream of the CAM 7.0 Unified rebuild (master orchestrator coordinates you).
Repository root /workspace/agentrs (branch feat/cam-7.0-unified-complete; tag seam-0 = the seam commit you start from).
Interpreter /workspace/agentrs/.venv/bin/python; run from <repo>/superlearner with PYTHONPATH including <repo>/crucible/src.
BINDING PLAN: superlearner/docs/CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md — read §2 (canonical loop, observation-only
reward, loss/parameter table, dom preset) and YOUR workstream's section in §3 (owned files, exposed interfaces, real
end-to-end check). Full audit findings for your subsystem: superlearner/docs/cam_unified_audit_2026-11-26.json (key
names: core='h-core', semantic, world, beliefs, learning, archive (memory), distributed, envs, crucible5, h
(experiments), research). Research note for credit/meta-reward math: superlearner/docs/CAM_UNIFIED_REWARD_CREDIT_RESEARCH.md.
Already implemented and tested by the orchestrator (build on these, do not duplicate): credit/{exact,estimators,
meta_exact,lifetime_exact,longdelay,heads}.py, selfeval/{evaluator,intrinsic,metagrad,voi,reward_model}.py,
envs/{experience,labels,integrity}.py, autonomy/{goals,curriculum,hindsight}.py, core/semantic_xattn.py (needs the
audit's fix: explicit sem argument, checkpoint-safe), semantic/decoder_xattn.py, verify_completion.py.

RULES
* Edit ONLY files your workstream owns (plan §3) plus new files inside your owned packages and your own new tests.
  If you need a change in someone else's file, do not make it: write it under "change_requests" in your report.
* Mechanisms must be REAL and exercised by the real model (dom_dev preset: 8x384 core, Qwen3-0.6B). Test doubles are
  allowed only in unit tests and never count as evidence. Do not weaken existing tests; if you change behaviour an old
  test encoded, update the test with a one-line reason in the commit message.
* Observation-only is the default for every new CAM path: no environment reward reaches actor inputs, beliefs, world
  model targets or actor losses; verified labels reach the trainer only through the LabelStore.
* GPU: one A6000 shared by ~9 agents. Use dom_dev (Qwen3-0.6B). Wrap any ad-hoc GPU command using >4 GB in
  \`flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock <cmd>\` and hold it <= 15 min; never run pytest under the lock.
  Only W1 runs 1B ("dom") jobs. Keep CPU threads modest (OMP_NUM_THREADS<=8).
* Commit coherent milestones frequently in YOUR checkout (messages start "cam unified complete [<WS>]:" and end with
  the line "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"). Never push, never force, never rewrite history,
  never switch the main checkout's branch. Do not edit superlearner/docs/CAM_UNIFIED_COMPLETION.json or
  CAM_UNIFIED_LEDGER.md. Write evidence JSON to superlearner/docs/evidence/dom/<WS>/*.json (numbers you measured,
  commands, commit).
* Before finishing: run your tests and your plan §3 "Real end-to-end check"; commit everything (no uncommitted work);
  report branch name (git rev-parse --abbrev-ref HEAD), final commit, what passed/failed with numbers, what you could
  not do and why, and change requests for other owners. Be honest: a failed check is reported as failed.
`

const REPORT = {
  type: 'object',
  properties: {
    workstream: { type: 'string' },
    branch: { type: 'string' },
    worktree_path: { type: 'string' },
    final_commit: { type: 'string' },
    summary: { type: 'string' },
    checks: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, passed: { type: 'boolean' }, evidence: { type: 'string' } },
      required: ['name', 'passed', 'evidence'] } },
    tests_added: { type: 'array', items: { type: 'string' } },
    requirements_advanced: { type: 'array', items: { type: 'string', description: 'manifest id + what is now true' } },
    not_done: { type: 'array', items: { type: 'string' } },
    change_requests: { type: 'array', items: { type: 'string', description: 'owner file - requested change - why' } },
  },
  required: ['workstream', 'branch', 'final_commit', 'summary', 'checks', 'tests_added', 'requirements_advanced', 'not_done', 'change_requests'],
}

const WS = [
  { id: 'W2', name: 'ENV: environments, reward channels, labels', extra: `Also: make ExperienceOperator and OpenWorld operators implement LabelSource.take_labels() -> [LabelRecord]; add envs/factory.py make_env(kind, specs|tasks, n_envs, reward_mode, split) with a split guard refusing val/test families in training streams; result observations delivered with caused_by into the next observation batch; reward_mode='observation_only' for OpenWorld operators (RewardGate); integrity canary tests.` },
  { id: 'W3', name: 'CORE: F_theta persistence, Mechanism C, reasoner, entities', extra: `Fix GatedUnifiedCore per the audit (explicit sem argument through forward/run_blocks/Reasoner, checkpoint-safe, works on imagination batches); learned state gate (W=0,b=-2) and state embeddings std 0.02 under cfg.state_gate / cfg.state_emb_std; core.transition_blocks(k); fixed-J reasoning option; delayed recall test at delay 16 >= 95%.` },
  { id: 'W4', name: 'SEM: tokenizer value channel, semantic stream, decoder conditioning, strings', extra: `Fix the JSON string-closure token-boundary bug (constraints.py:93) and the leaking prefix init; numeric VALUE channel; ObservationTokenizer.semantic_context() producing SemanticContext for Mechanism C; wire semantic/decoder_xattn.LatentDecoderConditioning into QwenSemantic/HoleFiller decoding via a cond= argument (alpha init 0 => bit-identical logits at init, measurable dependence after training).` },
  { id: 'W5', name: 'DEC: decision bottleneck (Mechanism B, DOM-04/09)', extra: `IntentHead I_t, ActionGenerator reading I_t and ctx_out only (no tool_prior, goal only via Z/R/I), K candidates in one batched decode, decision/evaluator.score (Q of the CONCRETE candidate actions), decision/select with recorded gen_logp/sel_logp/stamp, decision/fast.fast_params for the meta-gradient; Condition G latent-swap test; execution refuses unstamped actions (assert in runtime is W1's; you provide the stamp). Fallback 'bypass-lite' if blocked.` },
  { id: 'W6', name: 'PRED: world model and beliefs', extra: `No EMA target; free bits per group; consistency loss; prior on R through core.transition_blocks(6) (stub against W3 signature); beliefs use feat from the WM prior; gated fusion of particle mixture and Kalman predictive (RBPF deferred but design the API so per-particle Kalman means can be added); observation-only outcome spec (reward unobserved); belief_runtime.predictive_log_prob, belief_change(fast_prev, fast_new) differentiable (kl_weights, kalman_ig, d_entropy), separation-correct adapt_loss over all k; world/voi.decision_voi using selfeval/voi.py; multi-seed causal tests.` },
  { id: 'W7', name: 'LEARN: integrated learner, optimisation, gradient routing', extra: `learn/integrated.py IntegratedLearner.step(batch)->UpdateInfo with the plan §2 loss/parameter table, per-group clipping, loss auto-balancing so world share of core gradient <= 0.45 and actor share >= 0.1, PPO on lag-0 / V-trace on lag>0 (no PPO ratios on stale data), named-group gradient table logged (DOM-08), reward from the RewardModel protocol (stub ok), observation-only: no env reward anywhere. Exp D reward conditions selectable (meta-learned intrinsic default; sparse verified terminal and legacy shaped as labelled baselines).` },
  { id: 'W8', name: 'REWARD: R_psi, verified-label predictors, meta-gradient, credit, autonomy', extra: `selfeval/reward_model.py RewardModel implementing plan §2's default actor reward (potential from ProspectEvaluator, IG, cost, rho_psi); selfeval/label_join.py joining LabelStore labels to archived decisions (all prefixes, pending excluded); MetaRewardTrainer over decision.fast.fast_params (stub against W5's API: a dict of named params) using selfeval/metagrad.py incl. the inner-score term; CAM-scale R-HCA/COCOA restricted candidate estimator over recorded candidate sets (CR-06/CR-07) with a test against credit/exact on a tabular wrapper; C_chi curriculum training on measured learning gain; reward-integrity tests on envs/integrity.py worlds (EX-F harness).` },
  { id: 'W9', name: 'MEM+EVID: checkpoints, memory hygiene, experiment drivers, statistics, governance', extra: `Checkpoint migration (load_weights(allow_new=...), arch_hash, code commit), weights-only reset semantics; memory featurizer leak fix (memory stays off in DOM runs, declared); experiment drivers experiments/dom_suite.py and meta_suite.py that run Conditions A-I (DOM-11) and the Exp D/E/F/I/J conditions ONLY through runtime.run_lifetime (stub against W1's signature) in observation_only with provenance rows; pre-registrations docs/experiments/U-DOM-1.md, U-MR-1.md, U-OW-1.md, U-SC-1.md committed before any confirm evaluation; stats (studentised t, paired seeds); extend verify_completion per plan §3 W9 (commit-bound JUnit records, double-rejecting plugin, prereg-before-confirm).` },
]

phase('Build')
const w1 = agent(COMMON + `
YOUR WORKSTREAM: W1 INT (integration, canonical loop, contracts, budget). You work DIRECTLY in the main checkout
/workspace/agentrs (no worktree). Implement plan §3 W1: runtime.run_lifetime (the ONE act->execute->advance loop;
envs/harness.run_lifetime, runtime.run_steps, experiments/mechrun.run_batch, capabilities/loop.train_agent, stage1
runner, experiments/train.py become thin wrappers), build.py build_agent(preset, env_kind, ...), LearnerHook wiring,
merge_by_seq, versions stamping (cam-event/4), action-authorisation assertion (refuse actions without a cam_eval stamp
when the dom decision path is on), test_single_loop.py, test_loop_parity.py. Use stubs/feature flags for parts owned by
other workstreams (their branches merge after you). Then the 1B budget work: core/bench.py full 1B mode and
test_memory_budget_1b.py (marked slow+gpu): build 'dom' (1B core + Qwen3-4B), one full optimiser step peak <= 40 GB,
decide at B=16,K=4 timing; report measured numbers even if over budget, with the fixes you tried (microbatching,
Adafactor for satellites, d_satellite 512). You are the only workstream allowed to run 1B jobs.`, { label: 'W1', phase: 'Build', schema: REPORT })

const others = WS.map(w => agent(COMMON + `
YOUR WORKSTREAM: ${w.id} ${w.name}. You run in an isolated git worktree (your current directory); the repository
paths above refer to the same files inside your worktree. ${w.extra}`, { label: w.id, phase: 'Build', schema: REPORT, isolation: 'worktree' }))

const built = (await Promise.all([w1, ...others])).filter(Boolean)
log(`built: ${built.map(b => b.workstream + '@' + (b.branch || '?')).join(', ')}`)

phase('Merge')
const merged = await agent(`You are the merge integrator for CAM Unified wave 3, working in the main checkout /workspace/agentrs on branch
feat/cam-7.0-unified-complete (W1 already committed there). Interpreter /workspace/agentrs/.venv/bin/python, run from
superlearner/ with PYTHONPATH including /workspace/agentrs/crucible/src. Read superlearner/docs/CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md §3-4.
Workstream reports (JSON; each names its branch and final commit):
${JSON.stringify(built, null, 1)}
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
with failing test ids), gate results with numbers, open problems.`, { label: 'merge', phase: 'Merge' })

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
INTEGRATION_TESTED/EMPIRICALLY_DEMONSTRATED/FAILED_EMPIRICALLY/BLOCKED) with the test ids / evidence files that support it.`,
  { label: `verify:${a.k}`, phase: 'Verify' })))

phase('Fix')
const fixed = await agent(`You are the fix integrator for CAM Unified wave 3 in the main checkout /workspace/agentrs (branch
feat/cam-7.0-unified-complete). Independent verifiers reported the findings below. Fix every finding that is a real
defect (verify each yourself first; skip with a reason if it is not real), highest severity first, with small
described commits ending with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Do not weaken tests. Never
push. Re-run the affected tests and record JUnit XML to superlearner/docs/evidence/test_results/wave3_fixed.xml
(-o junit_family=xunit1). Report each finding as fixed / not-real / not-fixed (why), test counts, and the per-requirement
status table the verifiers proposed, adjusted for your fixes.
MERGE REPORT:
${merged}
VERIFIER FINDINGS:
${verdicts.filter(Boolean).map((v, i) => `--- ${AREAS[i].k} ---\n${v}`).join('\n')}`, { label: 'fix', phase: 'Fix' })

return { built, merged, verdicts, fixed }
