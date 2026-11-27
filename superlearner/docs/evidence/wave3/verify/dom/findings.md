# DOM verifier (H) findings — running log

## F-params (confirmed) assembled `dom` model parameter counts (build_agent('dom', device='cpu'), script /tmp/claude-0/verify_dom/scripts/params_dom.py, output /tmp/claude-0/verify_dom/params_dom.json)
- core 1,004,124,162 trainable (32 blocks 906,072,064; Mech C xattn 81,840,128 + core sem_proj 3,936,256 + cond_proj 2,360,832; state gate 4,720,128; slot_codes 4,718,594; embeddings ~0.47M). All in the optimizer (trainable_not_in_optimizer = 0).
- satellites (trainer 'heads' group) 569,391,998 incl. decoder conditioning (Mech A) 50,368,518, intent head 56.7M, evaluator 14.2M, action generator 191.6M, world model 130.6M, beliefs 49.8M. Semantic group 6,144 (bridge). Qwen3-4B 4,022,468,096 frozen, 0 trainable.
- CAM trainable total 1,573,522,304.
- Discrepancy: manifest DOM-01 evidence quotes 1,081,244,162 / sem xattn 163.7M, from tests/cam_unified/dom/test_dom_semantic_xattn.py:257 which uses make_config("superlearner", sem_xattn_every=2) on meta, NOT the dom preset (sem_xattn_every=4). Fix: evidence should cite the assembled preset count 1,004,124,162 (still inside 0.9-1.1B). Satellites 0.52B (excl. dec xattn) exceed the plan §2 memory-budget assumption (<= 0.3B).

## F-rm (code-confirmed; runtime check pending in mech.json) reward-model potential Phi never reaches the candidate evaluator
- src/superlearner/cam_unified/model.py:1602 `attach_reward_model` is never called anywhere (grep: only its definition); model.__dict__["reward_model"] stays None (model.py:932). `_score_row` (model.py:~1672) and `_dom_terms` read it -> potential=None -> Phi = 0 in Q_k = r^ + Gamma(w_phi*Phi + w_v*V_int). The evaluator's only explicitly goal-conditioned term (Phi(Z^,G), ProspectEvaluator) is dead in acting and learning; stamp versions["reward_model"] = None.
- Fix: in build.build_agent after RewardModel(cfg) (and in IntegratedLearner.__init__/from_system) call model.attach_reward_model(reward_model); add an e2e assertion that stamp versions.reward_model is not None and phi varies.

## F-condG-R (from W5's own evidence, re-check pending) Condition G holds for I_t only; the latent reasoning state R_t has zero influence on the action
- docs/evidence/dom/W5/condition_g_dom_dev.json trained.condition_g.swap_R_before_intent_follow_donor = 0.0 (swap_I_only 1.0). The intent head takes the goal from ctx_out (core outputs at the goal token positions), not from R_t/Z_t. tests/cam_unified/dom/test_dom_condition_g.py does not assert on the R-before-intent swap and asserts R-after-intent has no effect.

## F-auth (confirmed) execution authorisation is fail-open and lives in the loop, not the execution layer
- src/superlearner/cam_unified/runtime.py:642 `require = dom_decision_path(cfg)` where cfg = agent_config(agent) (runtime.py:271); an agent object without `.config` (or a wrapper not exposing `.agent.config`) gets require=False and its unstamped actions are executed. Check: /tmp/claude-0/verify_dom/scripts/auth_failopen.py -> "no-cfg EXECUTED unstamped actions ... authorisation {'required': False, 'checked': 0}"; with the dom_dev cfg the same action is REFUSED. Environment adapters (ExperienceAdapter/OpenWorldAdapter/GuardedEnv runtime.py:494) never check stamps, and run_lifetime(require_cam_eval=False) disables it.
- Fix: default require=True unless the caller explicitly declares a legacy/baseline path; or enforce verify_stamp inside the env adapters' execute (GuardedEnv / make_env factory) for dom presets.

## F-dom08 (confirmed) independent DOM-08 table at dom_dev, canonical build_agent + run_lifetime (stage1.dom_e2e, 4 rapid + 1 ow_tiny, 24 rounds, 3 IntegratedLearner updates, default dom_dev config)
- script /tmp/claude-0/verify_dom/scripts/dom08_run.py -> /tmp/claude-0/verify_dom/dom08.json. 120/120 actions stamped+verified, authorisation required, core hash changed, actor core share 0.35/0.10/0.11.
- update 1: 12/14 nonzero (dec_xattn ZERO: no string-hole action in the window; predictors ZERO). updates 2-3: 13/14 nonzero (predictors ZERO - no verified labels in 24 rounds). All 11 spec-listed DOM-08 items (early/mid/late core, sem projections, reasoning, beliefs, WM, intent, decoder conditioning, exploration, policy/value) are nonzero by update 2.
- Group-level "nonzero" hides dead sub-modules (post-accumulate hooks on every tensor, union over 3 updates): 58 tensors / 3.07M of 67.6M CAM params never got gradient: beliefs.act_cond.* 1.80M (BeliefModule.condition_on_action, core/beliefs.py:152/297 - beliefs group only 42-46% of its numel trained), world_model.rew_pool/rew_head/rew_src 1.09M (reward head dead in observation_only), core.cond_proj 147,840 (Mech C gate conditioning, see F-mechC-cond), bayes.* 26.6K, reasoner.halt_* (fixed J), action_generator.bridge.cond_src (n_prefix=0), unused typed-head digit/union/bool/sign outputs.
- tests/cam_unified/dom/test_dom08_gradients.py is marked slow and was excluded from the merged "not slow" suite (c6149ce), and it only fails on "zero" groups that are not in its hard list; predictors are "zero" in my run, so its first assertion would fail on this driver unless labels arrive.

## F-gen-credit (code-confirmed) generator gradient of the K-candidate policy is a biased surrogate
- decision/select.py:6-9 + model.py:1912 + learn/integrated.py:195-199: the actor's PPO ratio and score use logp = log p_gen(a_k | I) + log softmax(Q/tau)_k of the CHOSEN candidate only. The process samples all K candidates from p_gen, so the score of the executed (C, k) is sum_j grad log p_gen(c_j) + grad log sigma(k|C); dropping the j != k terms biases the generator gradient (toy K=2, perfect selector: surrogate E = 2-p vs true 2-2p). credit/candidates.py is exact only for the selector conditional on C. Non-chosen candidates' gen_logp in the stamp are unrescored incremental (bf16) values (ba74bdd).
- Fix: add sum_{j != k} log p_gen(c_j) (same advantage, or a leave-one-out baseline over the K candidates) to the generator term, or train the generator with its own estimator (e.g. per-candidate Q-advantage) and report the two separately; manifest DOM-09 should say which estimator is used.

## F-mechC-cond (confirmed) Mechanism C gate never receives B_t / G_t; core.cond_proj is dead weight inside the "1B core"
- semantic/tokenizer.py:429 docstring: "`cond` is left None (the model fills it from the belief and goal summaries)" - nothing fills it (grep: no writer of SemanticContext.cond; model.semantic_context model.py:1216 returns the tokenizer's object unchanged). backbone.py:269 -> cond None -> semantic_xattn.py:139 uses zeros, so g_l = sigmoid(Gate(Z_l, C_l, 0)) instead of Gate(Z_l, C_l, B_t, G_t) (spec 3C). core.cond_proj (dom_dev 147,840; dom 2,360,832 params) never got gradient in 3 real updates (dom08.json never_grad). Runtime spy of project_semantic in mech.json (pending).
- Fix: in CamUnifiedModel.encode_step build sem.cond from pooled belief + goal tokens (e.g. mean of Bt.x and G.x / ctx_out at those positions) before calling the core/reasoner; or delete cond_proj and the claim.

## F-dead (confirmed at dom_dev, sized at dom) dead satellite parameters counted in CAM totals
- At dom (scripts/dead_1b.py): beliefs.act_cond 28,319,360 (57% of beliefs 49.65M; BeliefModule.condition_on_action core/beliefs.py:297 only used when `feat` is None, never in the real path), world_model.rew_pool/rew_head/rew_src 14,959,359 (WM reward head, unused in observation_only), core.cond_proj 2,360,832, bayes.* 101,960, reasoner.halt_* 3,073, bridge 6,144: 45.75M of 1.57B CAM trainable never receive gradient. Core count is only inflated by cond_proj (0.24%).

## F-A03 / F-condG-R (confirmed on build_agent('dom_dev'), scripts/mech_checks.py -> /tmp/claude-0/verify_dom/mech_run1.json)
- Condition G after the 30-step imitation (cg.imitate): swap(R,I) -> donor 1.00, swap I only -> donor 1.00, KL(pi||pi with I permuted) 24.5 nats (before training 0.001), I=0 -> goal gap 0.0, stamps verified/refused correctly. BUT swap R_t BEFORE the intent head -> follows donor 0.00 (sampled and greedy): the latent reasoning state has no influence on the executed action; the intent reads the goal from ctx_out.
- Reasoning iterations (model.reason_iters J in 1,2,4,8,16 on fixed inputs, greedy executed action): trained model: 0% of tools/actions change vs J=2, KL(tool) <= 4.6e-5, although I moves 13-34% (relative norm). Untrained model: tools flip 12-100% but only because the tool distribution is near-uniform (KL <= 0.013). So in the real DOM decision path reasoning depth does not change decisions; A03's evidence (J=2 vs J=1 1.000/0.401) comes from test-local heads (tests/cam_unified/core/test_core_reasoning.py:474), not the real policy.
- Fix: make R_t carry the decision (e.g. intent reads R only, or ctx_out through R; or dropout of ctx_out in the intent head), and add acceptance tests on the real model: swap-R-before-intent follows donor >= 0.9 and forced J=1 vs J=4 changes >= 20% of decisions on a 2-hop task (plan E0).

## F-mechC-effect (confirmed) Mechanism C has negligible functional effect at dom_dev
- Removing S_t (semantic_context -> None) changes the tool distribution by KL 1.6e-6 (init) / 1.7e-8 (after imitation); I changes 1.1% / 0.3%. Gates 0.0474 at both blocks (sigmoid(-3)) and did not move in 30 imitation steps. Spy: the real decide passes a SemanticContext with cond=None (confirms F-mechC-cond). Wired, but no evidence yet that CAM uses semantic content.

## F-rm runtime-confirmed (mech2.json): build_agent('dom_dev'): model reward_model attached = False while system/learner hold a RewardModel; every stamp has versions.reward_model = None and phi = [0,0,0,0].

## F-mechA (confirmed, mech2.json) Mechanism A wired; gradient passes the frozen decoder; no trained dependence yet
- Real decide over the ow_tiny catalogue: dec_cond hooks fired 404 times (layers 5/11/17/23 of Qwen3-0.6B), candidate with string holes (market.confirm) found.
- alpha=0 (init): string-token logp identical under an altered I; I / intent head / core block0 get exactly 0 gradient from string tokens (only alpha gets gradient) - by design (tanh gate at 0).
- alpha=0.3 (set by hand): logp changes with I (-7.68 vs -7.52); gradient reaches the earliest conditioned layer's kv (1.1), I (0.06), intent head (5.7), core block 0 (2.2); frozen Qwen params get none. So the frozen semantic model does not detach CAM inputs (transformers 5.24 + HF checkpointing).
- Not shown anywhere: alpha moving away from 0 in real training, i.e. "decoder output measurably depends on Z_t" in a trained system. Decoder cross-attends to I_t (16 intent tokens), not to Z_t/R_t as spec 3A writes; at 1B (G3 runs, rapid tools only, ow_envs 0) string holes never occur, so Mechanism A was never exercised at 1B.

## F-1B-auth (confirmed) the 1B acting evidence ran with the authorisation gate off
- src/superlearner/cam_unified/core/bench.py:505 run_lifetime(..., require_cam_eval=False) for the G3 1B runs; docs/evidence/dom/G3G4/g3_run6_dom.json acting.authorisation = {"required": false, "source": "explicit"}, ow_envs 0, tools_per_env [1,1,1,1]. Not declared as a deviation in G3G4/README.md. Fix: drop require_cam_eval=False (or record it as a deviation).

## F-manifest (confirmed) stale ids / evidence
- DOM-05 acceptance id test_dom_semantic_xattn.py::test_without_semantic_input_it_is_exactly_the_unified_core does not exist (the test is ..._plain_backbone, line 69).
- tests/cam_unified/dom/test_dom08_gradients.py (slow) is absent from docs/evidence/test_results/wave3_merged.xml; no 1B DOM-08 table exists.
- CPU DOM tests: 46 passed, 11 skipped (GPU) (/tmp/claude-0/verify_dom/pytest_cpu.txt).
