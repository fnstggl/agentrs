## Verifier H: CAM 7.0 Unified, DOM amendment (DOM-01..DOM-09, A01, A03). Read-only, nothing committed.

I ran every check myself on the real model. The full running log is /tmp/claude-0/verify_dom/findings.md and the scripts are in /tmp/claude-0/verify_dom/scripts/. Raw outputs:
- params_dom.json: the assembled `dom` model, built on CPU.
- dead_1b.json: sizes of the parameters that never receive gradient, measured at `dom`.
- dom08.json: my DOM-08 run at dom_dev. I used the canonical `build_agent` + `run_lifetime` path (stage1.dom_e2e) on 4 rapid worlds plus 1 ow_tiny, 24 rounds, 3 IntegratedLearner updates and the default config. A post-accumulate-grad hook sat on every trainable tensor.
- mech_run1.json and mech2.json: mechanism checks on `build_agent('dom_dev')`.
- auth_failopen.py: the execution-gate check.
- pytest_cpu.txt: the DOM CPU tests, 46 passed and 11 skipped (the GPU tests).

### Parameter counts of the assembled `dom` model (from build_agent)
- **Core: 1,004,124,162 trainable.**
  - 32 blocks: 906,072,064.
  - Mechanism C: cross-attention 81,840,128, core `sem_proj` 3,936,256, `cond_proj` 2,360,832.
  - State gate 4,720,128, slot codes 4,718,594.
- **Satellites (the trainer's "heads" group): 569,391,998.** Of this, decoder conditioning (Mechanism A) is 50,368,518, the intent head 56.7M, the evaluator 14.2M, the action generator 191.6M and the world model 130.6M.
- **Semantic group: 6,144** (the bridge only).
- **Qwen3-4B: 4,022,468,096**, all frozen.
- CAM trainable total is 1,573,522,304, and every trainable tensor is in the optimiser.
- At dom_dev: core 16.9M, heads 50.6M, total 67.6M.

### Findings, most severe first

**1. HIGH: the latent reasoning state and the reasoning iterations do not change decisions in the trained real model (A03, Condition G).**
- Where: src/superlearner/cam_unified/decision/intent.py:59-75 and model.py:1725-1733. The intent head reads R_t together with ctx_out, the core's outputs at the context positions. After training it takes the goal from ctx_out only.
- Check (mech_run1.json): after the 30-step Condition G imitation on `build_agent('dom_dev')`:
  - Swapping R and I, or I alone, makes 1.00 of actions follow the donor. KL under intent permutation is 24.5 nats. With I = 0 the goal gap is 0.
  - Swapping R_t *before* the intent head makes 0.00 of actions follow the donor, both sampled and greedy. W5's own evidence (docs/evidence/dom/W5/condition_g_dom_dev.json) shows the same 0.0.
  - Forcing J = 1, 4, 8 or 16 instead of 2 on fixed inputs changes 0% of tools and actions (KL ≤ 4.6e-5), even though I moves by 13-34%.
  - At init J does flip 12-100% of greedy tools, but only because the tool distribution is near-uniform there (KL ≤ 0.013).
  - A03's evidence comes from test-local heads (tests/cam_unified/core/test_core_reasoning.py:474), and test_dom_condition_g.py never asserts on the R-before-intent swap.
- Fix: make R_t carry the decision, for example intent reading R only or ctx_out dropout in the intent head. Add real-model acceptance tests: the R-before-intent swap follows the donor ≥ 0.9, and forced J=1 vs J=4 changes ≥ 20% of decisions on a 2-hop task (plan E0).

**2. HIGH/MEDIUM: the reward model's potential Phi never reaches the candidate evaluator.**
- Where: model.py:1602. `attach_reward_model` is never called anywhere in the source, so `model.__dict__["reward_model"]` stays None (model.py:932). `_score_row` (around model.py:1672) and `_dom_terms` (around model.py:1957) therefore pass `potential=None`.
- Effect: Phi = 0 in Q_k = r̂ + Γ(w_phi·Phi + w_v·V_int). This dead term is the evaluator's only explicitly goal-conditioned one, and it is dead in both acting and learning.
- Check (mech2.json): attached = False, while the System and the learner each hold a RewardModel. Every stamp has versions.reward_model = None and phi = [0, 0, 0, 0].
- Fix: call `model.attach_reward_model(reward_model)` in build.build_agent (around build.py:349) and in IntegratedLearner.from_system. Add an end-to-end assertion that the stamp's reward_model version is set and that phi varies.

**3. MEDIUM: the generator's gradient in the K-candidate policy is biased (DOM-09).**
- Where: decision/select.py:6-9, model.py:1912 and learn/integrated.py:195-199. The PPO ratio and score use log p_gen(a_k) + log softmax(Q/τ)_k for the chosen candidate only.
- Why this is wrong: all K candidates are sampled, so the correct score is Σ_j ∇log p_gen(c_j) + ∇log σ(k|C). In a toy case (K=2, perfect selector) the surrogate's expected gradient is 2−p where the true one is 2−2p.
- credit/candidates.py is exact only for the selector, conditional on the candidate set C. The stored gen_logp of non-chosen candidates are unrescored bf16 sampling values (since ba74bdd).
- Fix: add the j≠k generator terms (or a leave-one-out baseline), or train the generator with its own estimator and report generator and selector learning separately.

**4. MEDIUM: the execution gate is fail-open and lives in the loop, not in the execution layer (DOM-04).**
- runtime.py:642 sets `require = dom_decision_path(agent_config(agent))`. An agent without `.config` gets require = False.
- Check (auth_failopen.py): with no config, unstamped actions were EXECUTED, authorisation {'required': False, 'checked': 0}. With the dom_dev config the same action was REFUSED.
- The environment adapters, including GuardedEnv (runtime.py:494), never check stamps.
- The 1B benchmark turns the gate off: core/bench.py:505 passes `require_cam_eval=False`. In g3_run6_dom.json, acting.authorisation is {"required": false, "source": "explicit"}, and G3G4/README.md does not declare this as a deviation.
- Fix: require stamps by default unless a legacy or baseline path is explicitly declared, enforce the stamp check in the environment wrapper, and remove the override in bench.py:505.

**5. MEDIUM: the Mechanism C gate never receives B_t or G_t, cond_proj is dead weight inside the counted core, and Mechanism C has almost no effect.**
- Nothing fills `SemanticContext.cond`. The tokenizer leaves it None (semantic/tokenizer.py:429, whose docstring wrongly says the model fills it), and model.semantic_context (model.py:1216) passes it through unchanged. backbone.py:269 then gets None, and semantic_xattn.py:139 feeds zeros to the gate.
- Effect: the gate computes g = σ(Gate(Z_l, C_l, 0)) instead of the spec's Gate(Z_l, C_l, B_t, G_t). `core.cond_proj` (147,840 parameters at dom_dev, 2,360,832 at dom) never received gradient. A spy on the real decide confirmed cond_none = True.
- Effect size: removing S_t changes the tool distribution by KL 1.6e-6 at init and 1.7e-8 after imitation. Both gates sit at 0.0474 (sigmoid(−3)) and did not move during the imitation.
- Fix: build cond from pooled belief and goal tokens in encode_step, or delete cond_proj and the claim.

**6. LOW/MEDIUM: parameters that never get gradient are counted in the CAM totals, and DOM-08's group-level "nonzero" hides them.**
- At dom_dev, 58 tensors (3.07M of 67.6M) received no gradient in any of the 3 updates.
- Sized at `dom` (dead_1b.json):
  - `beliefs.act_cond`: 28,319,360, which is 57% of the beliefs component (core/beliefs.py:297, used only when `feat` is None).
  - World-model reward head (rew_pool, rew_head, rew_src): 14,959,359, unused in observation_only mode.
  - `core.cond_proj`: 2,360,832.
  - bayes.*, reasoner.halt_* and the bridge: about 0.1M.
  - Total: about 45.75M of the 1.57B.
- learn/grad_probe.py:189-193 marks a group "nonzero" if any objective sends it any gradient. Report the trained fraction of each group instead.

**7. LOW: Mechanism A is wired and passes gradient through the frozen decoder, but no trained dependence is shown.**
- In the real decide over the ow_tiny catalogue, the conditioning hooks fired 404 times (Qwen3-0.6B layers 5, 11, 17 and 23).
- With alpha = 0 (init), the string log-probability ignores I, and only alpha receives gradient.
- With alpha set to 0.3 by hand, the log-probability changes with I (−7.68 vs −7.52). Gradient then reaches the earliest conditioned layer's kv (1.1), I (0.06), the intent head (5.7) and core block 0 (2.2). Frozen Qwen parameters get none, so the frozen decoder does not detach CAM inputs (transformers 5.24).
- Gaps:
  - No run shows alpha moving away from 0 in training.
  - The decoder reads the 16 intent tokens I_t, not Z_t as spec 3A writes.
  - Mechanism A was never exercised at 1B, because the G3 runs had ow_envs 0 and only rapid tools, so no string arguments.
  - In my run dec_xattn was zero in update 1 and nonzero only in updates 2-3.

**8. LOW: the manifest and evidence have stale or missing entries.**
- DOM-01's evidence (1,081,244,162) comes from `make_config("superlearner", sem_xattn_every=2)` at test_dom_semantic_xattn.py:257, not from the preset. The preset's real core is 1,004,124,162.
- DOM-05's acceptance id `…exactly_the_unified_core` does not exist; the test is `…exactly_the_plain_backbone` at line 69.
- test_dom08_gradients.py is marked slow and is absent from wave3_merged.xml. No DOM-08 table exists at 1B.
- "predictors" was zero in all 3 of my updates (no labels arrived).
- Satellites are 0.52B excluding decoder conditioning, against the plan §2 budget assumption of ≤ 0.3B. This feeds the G3 memory failure.

### DOM-08 table I ran at dom_dev (dom08.json)
- 120 of 120 actions were stamped and verified, authorisation was required, the core hash changed, and the actor's core share was 0.35, 0.10 and 0.11 across the three updates.
- Update 1: 12 of 14 groups nonzero; dec_xattn and predictors were zero.
- Updates 2-3: 13 of 14 nonzero; predictors still zero.
- All 11 groups the spec lists are nonzero by update 2: early, middle and late core blocks, semantic projections, reasoning, beliefs, world model, intent, decoder conditioning, exploration, and policy/value.

### Status each manifest id's evidence supports
| id | status | basis |
|---|---|---|
| A01 | IMPLEMENTED | Per-component counts reported. test_core_backbone dev test passes (pytest_cpu.txt). Assembled counts above (params_dom.json). |
| A03 | IMPLEMENTED (unit only) | Unit tests in test_core_reasoning.py pass. On the real path, reasoning depth and R swaps change 0% of decisions (mech_run1.json a03_trained, cond_g_after). Not integration-tested; FAILED_EMPIRICALLY if the 2-hop E0 check gives the same result. |
| DOM-01 | INTEGRATION_TESTED | Assembled preset core 1,004,124,162, all trainable and in the optimiser (params_dom.json). The core hash changes in 1B learner updates (G3G4/g3_run6_dom.json). Caveat: cond_proj is dead (2.36M); correct the evidence number. |
| DOM-02 | IN_PROGRESS | 1B builds, acts and trains on rapid only (G3G4). OpenWorld not exercised (ow_envs 0), authorisation off, G3 budgets not met in one config, G4 not run. |
| DOM-03 | INTEGRATION_TESTED | Wired in the real decide; dec_xattn gets gradient in the real loop (dom08.json); gradient passes the frozen decoder (mech2.json); test_dom_decoder_xattn. Not demonstrated empirically (alpha stays 0, no trained dependence). |
| DOM-04 | INTEGRATION_TESTED (dom_dev), with open defects | intent → generation → Q of concrete candidates → selection → stamp → `authorise` in run_lifetime (dom08.json, test_dom_bottleneck, test_selection_replay, Condition G on I in mech_run1.json). Open: findings 2 and 4. |
| DOM-05 | INTEGRATION_TESTED (wiring and gradients), spec deviation open | calls_with_sem increments in decide, gate gets gradient. Gate lacks B and G (finding 5), effect negligible, acceptance id stale. |
| DOM-06 | INTEGRATION_TESTED | World model and V_int are trained in the real loop; no LLM is used as a value (dom08.json). Phi is dead (finding 2). |
| DOM-07 | IN_PROGRESS | The exploration head gets actor gradient (dom08.json). Multistep belief-aware planning is off in every dom preset (config.py DOM_SWITCHES use_imagination=False; `_decide_dom` has no planner). |
| DOM-08 | INTEGRATION_TESTED (dom_dev only) | My table above. No 1B table; the test is not in JUnit; dead sub-modules are hidden (finding 6). |
| DOM-09 | IN_PROGRESS | gen_logp and sel_logp are recorded and replay exactly (test_selection_replay, condition_g replay). Generator and selector learning are not correctly separated (finding 3). |