# CAM Unified redesign plan (synthesis critic of audit wf_038852ca-d04, 2026-11-26)

# CAM 7.0 Unified: master redesign and implementation plan

Branch `feat/cam-7.0-unified-complete` at HEAD b0861a2. Path shorthand: `cu/` = `superlearner/src/superlearner/cam_unified/`, `T/` = `superlearner/tests/cam_unified/`.

## 0. Spot-checks I ran myself (read-only, HEAD b0861a2)

Every claim I checked holds.

1. **Decision bypass.** `cu/semantic/actions.py:300` builds `decision_context(R, goal, belief, …)` from raw tokenizer goal tokens. `:326` adds `self.tool_prior(tools2.x)` to the tool logits, and that prior does not depend on R. Confirmed.
2. **String constraint.** `cu/semantic/constraints.py:93` has `return (k == last), True, NORMAL, n`, so a closing quote is valid only as a token's last character. The Qwen tokens `",` and `"}` are therefore forbidden. Confirmed.
3. **Ungated state.** `cu/core/backbone.py:80-81` initialises `kind_emb` and `state_pos` with std 0.5 (emb_std). `embed_state` re-adds both every step, and `forward` returns `Z = h[:, :S]` after `out_norm` with no gate. Confirmed.
4. **Tool results lost; evaluator reward leaks.** In `cu/envs/harness.py`, `batch = nxt` takes only the advance observations, so `ExecutionResult.observations` are dropped. `rewards = evaluator.take_rewards()` goes into `EventRecord.reward` and `Transition`. Confirmed.
5. **Value-blind numerics.** `cu/semantic/tokenizer.py` `_numeric` computes `num_mlp(numf(v)) + unit_proj(...) + desc_proj(...)` and then a single `content_norm`, so the value term is swamped. Confirmed.
6. **KL free nats.** `cu/world/dynamics.py:573-574` clamps `dyn` and `rep` at `kl_free_nats` (1.0, `config.py:87`). Measured KL is about 0.3, so the KL gradient is zero. Confirmed.
7. **New modules are not wired in.** `selfeval/{metagrad,intrinsic,evaluator,voi}.py`, `credit/*`, `autonomy/*`, `envs/{experience,labels,integrity}.py` were all committed after the audits. A grep finds no import of any of them from `agent.py`, `model.py`, `runtime.py`, `stage1/*`, `dist/*`, `learn/*` or `envs/harness.py`. Their manifest statuses (c3d0b0c) rest on library unit tests only and must not count as integrated.

Not re-verified: the meta_exact inner-score bias is reported fixed in cc9dc94. Its regression test is part of W8.

GPU right now: 1.6 GB of 49 GB in use and 0% utilisation, so the card is available for exclusive 1B runs.

## 1. Verdicts per subsystem

### Core
| Component | Verdict | Reason |
|---|---|---|
| UnifiedCore F_θ | REPAIR | Add a gated state update. Set state_pos/kind_emb std to 0.02. Add an explicit `sem` argument. Unify the context budget in fit_budget. |
| GatedUnifiedCore (Mech C) | REPAIR | Replace module-state `set_semantic` with an explicit argument (it is checkpoint- and batch-unsafe). Wire it in at every 4th layer. |
| Reasoner | REPAIR | Fix J=2 until halting is trained; max 32 at the 1B scale. The world prior consumes R_t, so reasoning gets world-model gradient. |
| Decision pathway (encode/decide/AG conditioning) | REDESIGN | Close the bypass by construction: everything goes through the intent I_t. Pointers read CAM-processed `ctx_out`. |
| build_model / registry | RETAIN | Real classes only. Add a build_report assertion that no doubles are present. |
| Config / scales | REPAIR | Add the `dom`/`dom_dev`/`dom_150m` presets, d_satellite, and removal of the EMA target and the retriever for budget. |
| EntityEncoder | EXTEND | d_satellite width and type tokens. The EVENT encoder is deferred. |
| Invariance (W02) | EXTEND, deferred | Not on the critical path; it would be a trainer term after 24 h. |
| core/bench.py | EXTEND | Add a full-model 1B benchmark mode. |
| interfaces.py | EXTEND | Contract v4: INTENT kind with a migration hook, SemanticContext, Decision.intent/evaluation/candidates. |

### Beliefs
| Component | Verdict | Reason |
|---|---|---|
| BayesLastLayer | EXTEND | Per-key forgetting λ_c, per-key prior/noise, alarm-coupled inflation, a Kalman summary token into the belief tokens. |
| Particle+Kalman composition | REDESIGN, interim | A learned gated fusion replaces variance replacement. Add a mixture-only auxiliary loss and disable Kalman in 50% of adapt calls. The full RBPF is deferred past the rental. |
| BeliefModule | REPAIR | Use `feat = outcome_head.features(Z_hat)` (the world-model likelihood). film_init 1/√d_p. Compute surprise on the predictive actually used. |
| Runtime belief path | REPAIR | The reward key is unobserved in observation_only. LP comes from the full predictive. |
| adapt_loss | REPAIR | Support-end world-model rollout (`learn/meta.SupportQueryLoss`), all k in one pass, reset at done. |
| Channel binding | RETAIN | |
| Belief harness | EXTEND | NLL/coverage/ECE versus an oracle. |
| belief_change / IG features | IMPLEMENT | Needed by R_ψ and VOI. |

### Semantic
| Component | Verdict | Reason |
|---|---|---|
| QwenSemantic | EXTEND | `cond=` on `step`/`decoder_hidden`, attach the conditioning, Qwen3-4B for `dom`, merged-LoRA acting off. |
| ObservationTokenizer | EXTEND | The numeric VALUE channel is critical. Add a `semantic_context()` producer. |
| LatentBridge | REDESIGN | Zero-initialised residual prefix, no init phrase, no RMS renorm. Cross-attention becomes the primary path. |
| decoder_xattn (Mech A) | REPAIR | alpha 0 / zero-initialised `o`, plain residual. Wire into HoleFiller sample and score. |
| ActionGenerator | REDESIGN | Every head reads I_t. Remove `tool_prior`. Pointer keys come from `ctx_out`. |
| Grammar/constraints | REPAIR | Boundary-aware closing with a consumed-chars decode record v4. untyped_depth 4. |
| Semantic objectives | EXTEND, deferred | |
| demo_init | REPAIR, optional | Shuffle batches, clip per group, BPTT. Warm start only. |
| semserve | DEFER | The in-process batched decoder replaces it for this rental. |
| Sandbox / provenance | RETAIN | Provenance gains intent, candidates and selection probabilities. |
| Intent + pre-execution evaluation | IMPLEMENT | |

### World
| Component | Verdict | Reason |
|---|---|---|
| Dynamics | REDESIGN | Transition = last 6 core blocks shared by reference plus small adapters (52 GFLOP/row instead of 277). Single filter: world_loss uses `filt.Zs`. Consistency loss. Remove the EMA target (it is a frozen random net in the runtime anyway). |
| KL | REPAIR | Free bits per group at 0.02 nat, a mode-latent open-loop term, multi-seed tests. |
| EMA target | REDESIGN (delete) | Use the detached posterior Z as the latent target. Saves 3.6 GB and one core pass. |
| Outcomes | EXTEND | Drop 'reward' in observation_only. Goal-conditioned success/time/resource predictions move to `selfeval/evaluator.ProspectEvaluator`. |
| Runtime planner | REDESIGN | No Qwen decoding at imagined steps. Default is the h=1 evaluator. The optional h≤4 planner runs on intent/action tokens with utility = R_ψ + Φ + V. |
| world/planner.py, imagine_train | EXTEND, deferred | Only after the consistency gap is below 0.1. |
| Causal ledger / InterventionContrastive | REPAIR/EXTEND, deferred | |
| Baselines / capability diagnostics | RETAIN | Use as acceptance harnesses at dom_dev, not capability_config. |
| VOI | IMPLEMENT | Wrap `selfeval/voi.py` with cloned belief updates. |
| RUDDER/COCOA services | IMPLEMENT | `decision_features` (W8 consumes them). |

### Learning
| Component | Verdict | Reason |
|---|---|---|
| returns.py | RETAIN | The single source for advantages. Delete `agent.gae_semi_markov`. |
| ppo.py | EXTEND | `advantage_fn`, intrinsic critic, tool-level entropy. |
| agent.objective_policy | REBUILD | Becomes an adapter onto ActorCriticObjective (PPO at lag 0, V-trace at lag>0). |
| trainer.py | EXTEND | Role optimisers, per-group clipping, probe-calibrated loss balancing, named-group gradient probe that is actually logged. |
| meta.py | REDESIGN | SupportQueryLoss becomes the adapt objective. MetaRLObjective is kept only as a baseline. |
| Meta-gradient engine | EXTEND/wire | `selfeval/metagrad.py` already exists. |
| explore.py | REDESIGN | Its outputs become features of R_ψ, never the reward itself. |
| Exploration harness | REPAIR | 3-seed gate. |
| heads.py | EXTEND | V on (R, I). Predictors stay in selfeval. |
| curriculum | EXTEND | C_chi in `autonomy/curriculum.py`, wired after 24 h. |
| math_assist / trajectory | RETAIN / EXTEND | Trajectory gains reward_int and versions. |
| experiments/train.py | REBUILD | Thin wrapper over the canonical loop. |

### Memory
| Component | Verdict | Reason |
|---|---|---|
| Archive | EXTEND | Links table (attributed_to/caused_by); provenance at segment level only. |
| Artifacts / index | RETAIN | |
| Retriever | REDESIGN, deferred | `use_memory=False` in all DOM runs, declared as a deviation. A gradient-carrying `memory_tokens` plus a usefulness objective after the rental. |
| featurize | REBUILD | ActorView only. It reads the reward today (L2 4.79). |
| Consolidator | REDESIGN, deferred | Budgeted migration with a feature cache. |
| Replay | EXTEND | Burn-in slices and tag producers. |
| Checkpoint | REPAIR, critical for experiments | `allow_new` migration plus arch_hash. load_weights_only must reset fresh/trainer/trackers/RNG. |
| Plasticity | REPAIR | Fused w13 support. Monitor only. |
| Delayed-outcome links + label separation | IMPLEMENT | Join via `envs/labels.LabelStore`. |
| Dist memory | DEFER | |

### Distributed
| Component | Verdict | Reason |
|---|---|---|
| Queue, launcher, toy proofs | RETAIN | |
| Publish, central learner, cam_learner, cam_policy, inference server, cam_runs, sinks | DEFER | DOM runs are single-process batched acting plus learning, with one Qwen copy and B=16 environments. That is the degenerate form of central inference. Making 1B multi-process is a 40 h+ job that buys nothing on one GPU. No distributed claims at 1B. |
| Lag validation | REPAIR, only if dist is used | |
| Actor | REPAIR (obs-only), deferred | |
| cam_env | REBUILD | Replaced by `envs/experience.py` and the observation-only OpenWorld adapter. |
| Versioned records (CD-01) | IMPLEMENT in W1 | `extra['versions']`: policy, ψ, reward_mode, schema cam-event/4. |

### Environments and Crucible 5.0
| Component | Verdict | Reason |
|---|---|---|
| OpenWorld adapter | REPAIR | Deliver result observations; add `reward_mode`; strip family/meta. |
| OpenWorld evaluator | REDESIGN | RewardGate plus `take_labels`, implementing the LabelSource protocol. |
| harness | REPAIR | Becomes the canonical loop. |
| Parallel loops | REBUILD | Become wrappers. |
| Agent reward path | REDESIGN | |
| stage1 | EXTEND | |
| Protocols / goals | EXTEND, deferred | OW-2 recalibration and others; no CAM protocol claims this rental. |
| worldgen / heterogeneous agents / doubles | RETAIN | |
| `envs/experience.py` | RETAIN + test | It already scrubs the task-id and spec leaks, refuses broken multi-agent rapid families, chunks kernel advances, and computes oracle-relative labels. |
| labels.py / integrity.py | RETAIN + wire | |
| Crucible registry / rapid / kernel | REPAIR upstream, optional | The CAM adapter compensates. |
| Tier 2/3 | BLOCKED | |

### Experiments and evidence
| Component | Verdict | Reason |
|---|---|---|
| exp1/2/3 | REPAIR | Provenance and an observation-only channel; re-run at dom_dev. |
| Exp4 | IMPLEMENT, minimal | A single-lifetime OpenWorld demonstration only. |
| stats.py | REPAIR | Studentised t intervals; seed-independent paired comparison. |
| Capability tests | REPAIR | Correct the assertions and xfails. |
| Traces, dashboard, accounting | EXTEND/REPAIR | Add intent/evaluation/r_ψ fields to traces; GPU-busy seconds; token counts. |
| Register | REPAIR | |
| test_manifest | REBUILD | |
| verify_completion | REPAIR | Exact verdict enum, a guard against doubles, commit-bound JUnit records. |

## 2. One canonical runtime and the integrated training algorithm

**Canonical loop:** `cu/runtime.py::run_lifetime(agent, env, *, labels: LabelSource|None, label_store: LabelStore|None, reward_mode="observation_only", learner: LearnerHook|None, days, max_decisions, update_every=8, think_s, checkpoint, on_step) -> LifetimeResult`.

These become 10-line wrappers or are deleted: `envs/harness.run_lifetime`, `runtime.run_steps`, `experiments/mechrun.run_batch`, `capabilities/loop.train_agent`, the stage1 runner and `experiments/train.py`. `dist/actor` is declared non-DOM. CI test `T/integration/test_single_loop.py` greps that only `runtime.py` contains an act→execute→advance loop.

**Canonical factory:** `cu/build.py::build_agent(scale, *, env_kind, sem_model, device, run_dir, overrides) -> System`. It is used by every runner, by experiments and by capabilities.

**Canonical update:** `cu/learn/integrated.py::IntegratedLearner.step(batch) -> UpdateInfo`, called through the LearnerHook.

**Each round (B batched environments):**
1. `tools = env.available_tools()`.
2. `actions = agent.act(obs, tools)`. This calls `CamUnifiedModel.decide`:
   - encode: tokenizer, including the VALUE channel and S_t = `semantic_context()`;
   - Z_t = GatedCore(Z_{t-1}, ctx, sem);
   - R_t = Reasoner(Z_t, J=2, sem);
   - I_t = IntentHead(R_t, ctx_out) (n_intent=16, deterministic);
   - K=4 candidates from ActionGenerator(I_t). The tool pointer is a bilinear map of I_t against `ctx_out` at TOOL positions; evidence pointer over `ctx_out`; typed heads; strings through Qwen with I_t/R_t latent cross-attention every 6 layers. All K×B rows go through one batched decode.
   - `decision/evaluator.score`: Q_k = r̂_ψ + Φ(Ẑ_k, G) + γ V_int(Ẑ_k), with Ẑ_k = WM.prior(R_t, A_tok,k, B_t, dt) over the last 6 core blocks, plus a veto when Q_k < veto_q.
   - `decision/select`: k ~ softmax(Q/τ). The record stamps cam_eval, `gen_logp_k`, `sel_logp`, the candidate set, and the policy and ψ versions.
3. `res = env.execute(actions)`. The runtime asserts that every non-None action carries a cam_eval stamp.
4. `nxt = env.advance_time(dt)` with SMDP dt = duration + think.
5. `next_obs = merge_by_seq(res.observations, nxt)` (the function from dist/actor.py, moved here).
6. `agent.observe(Transition(records, next_obs, rewards=None, dt))`. In observation_only, `rewards is None` is asserted. Beliefs update on observations only (`reward_in_beliefs=False`).
7. `labels.take_labels()` → `label_store.append`. Learner-only; the agent object holds no reference to it.
8. Every `update_every` rounds: `learner.step()`. Every `meta_every=16` updates: `learner.meta_step()`.
9. Checkpoint every N updates: weights plus the full trainer and ψ state.

**Default observation-only actor reward** (`cu/selfeval/reward_model.py::RewardModel.reward`):

r_t = [Γ_t·Φ(R_{t+1},G) − Φ(R_t,G)] + λ_ig·IG_t − λ_c·ĉ_t + ρ_ψ(b_t, e_I, b_{t+1}, e_G, signals)

- Φ = ProspectEvaluator P̂(success by horizon | R, G). It is trained only on LabelStore verified labels; in r it is stop-gradient. This is potential shaping by CAM's own learned evaluator.
- Γ_t = exp(−κ·dt).
- IG_t = kalman_ig + KL(w_t‖w_{t−1}) from `belief_change`, computed on actor-visible observations.
- ĉ_t = observed resource spend (actor-visible cents and time), normalised.
- ρ_ψ = `selfeval/intrinsic.IntrinsicReward`, output layer zero-initialised. Its normalisation statistics are frozen inside each meta window.
- λ_ig = 0.05 and λ_c = 0.01 initially; both are members of ψ.

At initialisation Φ is about constant, so r ≈ λ_ig·IG − λ_c·ĉ. That is well defined and leak-free. r is detached in every actor loss.

Goal-free mode: G is a CAM-generated goal (`autonomy/goals.py`), with labels verified on the operator side.

A verified label never enters the actor reward. "Sparse verified terminal reward" (β_ver·U_ver as the actor reward) and the legacy shaped evaluator reward are explicit, labelled baseline conditions.

**Parameter groups and losses:**

| Loss | Updates | Gradient boundary |
|---|---|---|
| L_world: prior/posterior KL with free bits 0.02/group, obs+outcome NLL, open-loop h∈{1,2,4,8}, L_cons = ‖(Ẑ_{t+1} − sg(Z_post))/sd‖² | θ_wm (heads, transition adapters, outcome head), θ_core (filter states, last 6 shared blocks), θ_sem_in, θ_bel tokens | Weight auto-balanced so world share of core gradient ≤ 0.45. No EMA target. |
| L_adapt: SupportQueryLoss, support-end rollout, k∈{0,1,2,4,8,16,32,64} | θ_bel, Bayes layer, outcome head, θ_core via support-end Z only | Query outcomes never enter query states. |
| L_policy: ActorCriticObjective, PPO lag 0 / V-trace lag>0, 2 epochs, target_kl 0.03; logπ = log p_gen(a_k\|I) + log softmax(Q/τ)_k over recorded candidates; tool-entropy bonus | θ_dec_heads (intent, pointer, typed, length), θ_dec_cond (decoder xattn, bridge), θ_core (through I/R/Z), θ_sem_in, θ_eval-selection | Candidate-conditional surrogate (documented). Advantage from R_ψ returns, V_int bootstrap. |
| L_vint | Value head on (R, I) | Core share scaled. |
| L_eval: Q(R, A_exec) toward n-step r_ψ + Γ·V_int(next) | Evaluator head, θ_wm adapters | |
| L_pred: ProspectEvaluator success Bernoulli + survival time + resource, V_L on verified remaining utility | θ_pred, θ_core (share ≥ 0.1) | Labels only as targets. |
| L_meta (outer): −logπ_{φK}(a)·(U_ver − V_L) on fresh query windows | ψ only, separate AdamW | Test: ∂L_actor/∂ψ ≡ 0. |
| Inner (functional, `selfeval/metagrad.py`): K=2 steps on φ | φ = intent head + pointer + typed heads + selection/eval head + V_int (≤ 50M), on detached `policy_evaluate` features | Core, Qwen, WM and beliefs are never inner-updated. Bump weights_epoch while acting with φ_K, then consolidate φ_K into the slow weights. |

Frozen: Qwen3-4B base weights. Gradients flow through its activations into the decoder cross-attention adapters. Encoder states are cached and carry no gradient. LoRA is off in the `dom` preset (DOM-10 stage 1).

**Optimisers:**
- θ_core: AdamW fused, lr 3e-5, 500-update warmup.
- Heads, satellites and conditioning: lr 1e-4.
- ψ: lr 1e-3.
- Clipping is per group (median×5 running), not one global norm.
- Loss weights are recomputed every 25 updates from the named-group probe: w_i ← w_i·clip((target_i/share_i)^0.5, 0.5, 2).

**BPTT:**
- World and adapt replay: burn-in 16 steps without gradient, then 8 graded steps.
- Policy: one step from the stored Z_{t−1}. This is acceptable now that the state is gated.

**`dom` preset** (pre-declared in the seam commit; defaults keep current behaviour):
- Dimensions: 32×1536×24, S=128, d_ff 4096, Qwen3-4B, max_ctx_tokens 512.
- state_gate=True, state_emb_std=0.02.
- sem_xattn_every=4, max 256 semantic tokens.
- dec_xattn_every=6, dec_alpha_init=0.
- n_intent=16, n_candidates=4, select_tau=1.0.
- value_channel=True, d_satellite=768.
- wm_transition_blocks=6, wm_ema_target=False, kl_free_bits_group=0.02, wm_consistency=1.0, burn_in=16.
- reason_iters_fixed=2, max_reason_iters=32.
- use_imagination=False (the h=1 evaluator replaces it), use_memory=False.
- reward_mode="observation_only", reward_in_beliefs=False, meta_every=16.

Companion presets: `dom_dev` (8×384, Qwen3-0.6B for tests) and `dom_150m` (16×768).

**Memory budget (acceptance: ≤ 40 GB peak for one full optimiser step):**

| Item | Memory |
|---|---|
| Qwen3-4B bf16 | 7.5 GB |
| Core ≈ 1.0B incl. Mech C ×8 layers: fp32 weights + grads + AdamW | 16 GB |
| Satellites ≤ 0.3B | 4.8 GB |
| Decoder xattn ≈ 0.13B | 2 GB |
| Activations (checkpointed, microbatch 4) | ≈ 8 GB |
| **Total** | **≈ 38 GB** |

Removing the EMA target saves 3.6 GB and dropping the retriever saves 1.5 GB. Fallback if over budget: Adafactor for the satellites.

## 3. Workstreams (9). File ownership does not overlap.

**Seam commit (W1, T0 to T+2, before any parallel work):**
- Move `LatentImaginationPlanner` to `cu/world/planner_runtime.py`.
- Move `filter_sequence` and `world_loss` to `cu/world/train_seq.py`.
- Move `observe_outcomes`, `predict_outcomes`, `base_moments`, the `kalman_*` and `mixture_*` functions and `adapt_loss` to `cu/core/belief_runtime.py`.
- Move `retrieve`, `memory_tokens` and `stored_memory_tokens` to `cu/memory/runtime.py`.
- Move the agent objectives to `cu/learn/integrated.py`.
- Leave one-line delegations in `model.py` and `agent.py`.
- Add the interfaces v4 stubs, all config fields and presets, and the `RewardModel` protocol stub (returns 0).

Acceptance: existing tests pass, and a dev-scale `decide` is bit-identical before and after. Tag `seam-0`.

After the seam, every file has exactly one owner. `CAM_UNIFIED_COMPLETION.json` and `CAM_UNIFIED_LEDGER.md` belong to the orchestrator only. Workstreams write evidence to `docs/evidence/dom/<ws>/*.json`.

**GPU protocol:** `flock gpu_heavy.lock` with holds of 15 minutes or less. Workstreams run at `dom_dev` scale (Qwen3-0.6B). Only W1 runs 1B jobs until T+20. After that the GPU belongs to the experiment queue (W9 scripts, W1 operates).

---

### W1 INT: integration, canonical loop, contracts, budget
- **Owns:** `cu/runtime.py`, `cu/build.py` (new), `cu/agent.py`, `cu/config.py`, `cu/interfaces.py`, `cu/model.py` (seam and registry lines only; after the seam it is handed to W5), `cu/envs/harness.py` (shim), `cu/stage1/{system,runner,__main__,harness_agent}.py`, `cu/experiments/{stack,mechrun,train}.py`, `cu/capabilities/{stack,loop}.py`, `cu/core/bench.py` (full 1B mode).
- **New tests:** `T/integration/{test_single_loop.py, test_loop_parity.py, test_memory_budget_1b.py, test_dom_e2e_real.py}`.
- **Exposes:** `run_lifetime`, `build_agent`, `LearnerHook.on_step/step`, `merge_by_seq`, `EventRecord.extra['versions']` (cam-event/4), and an action-authorisation assertion.
- **Depends on:** everyone, at merge time.
- **Real end-to-end check:** `build_agent('dom')` (1B + Qwen3-4B) on 16 rapid training environments plus 2 ow_tiny environments through `run_lifetime`, 64 decisions and 2 updates. It must show: peak ≤ 40 GB; decide at B=16, K=4 in ≤ 1.5 s (numeric tools); an update of 128 decisions in ≤ 90 s; a core hash change; every action stamped; integrity canary scan clean.

### W2 ENV: environments, reward channels, labels
- **Owns:** `cu/envs/{openworld,openworld_eval,ow_convert,experience,labels,integrity,factory(new)}.py`, `cu/envs/protocols/*`, `cu/capabilities/envs.py` (payouts become observable NUMERIC observations), optional upstream fixes to `crucible/src/crucible/experience/{tier1,kernel}.py`.
- **New tests:** `T/envs/{test_reward_integrity.py, test_result_delivery.py, test_label_source.py}`.
- **Exposes:** `make_env(kind, specs|tasks, n_envs, reward_mode, split) -> (adapter, operator, LabelSource)`; `LabelSource.take_labels() -> [LabelRecord]`, implemented by both OpenWorldEvaluator (wrapping RewardGate) and ExperienceOperator; result observations delivered with `caused_by`; a split guard that refuses val/test families in training.
- **Depends on:** seam only.
- **Real end-to-end check:** the real `dom_dev` CamUnifiedAgent on ow_tiny for 12 decisions. The decision after `world.hosts` contains a TOOL_RESULT whose `caused_by` is the previous `action_id`. In observation_only, `Transition.rewards is None` and FastState is bit-identical when the evaluator's payout is altered. Canaries planted in task params, the evaluator state and the labels never appear in actor tensors, the archive or traces.

### W3 CORE: F_θ, persistence, Mechanism C, reasoner, entities
- **Owns:** `cu/core/{backbone,blocks,semantic_xattn,reasoning,entities}.py`.
- **Tests:** `T/core/test_core_state_persistence.py` (new), `T/dom/test_dom_semantic_xattn.py`, `T/core/test_core_reasoning.py`.
- **Exposes:**
  - `UnifiedCore.forward(Z_prev, ctx, code_rows=None, sem: SemanticContext|None=None) -> (Z, ctx_out)` with the gate u = σ(W[Z_prev; Z_new] + b), W=0, b=−2;
  - `run_blocks(h, key_mask, blocks=None, attn_bias=None, checkpoint_blocks=None, sem=None, block_ids=None)`;
  - `core.transition_blocks(k)`, the last k blocks;
  - `Reasoner.forward(Z, ctx, n_iters, code_rows, sem=None)`, with fixed J when `reason_iters_fixed` is set;
  - EntityEncoder at `d_satellite` width, consuming type tokens.
- **Depends on:** seam. Mech C real wiring needs W4's `semantic_context`; until then, test against synthetic token states.
- **Real end-to-end check:** real `build_model('dom_dev')`:
  - delayed recall at delay 16 ≥ 95% through the real core;
  - at init, cos(Z, const) < 0.9 and observation dependence > 10%;
  - decide + policy_evaluate + world_loss with `sem` gives a nonzero gradient on every cross-attention parameter;
  - a batch of B·K evaluator rows runs;
  - checkpointed Reasoner gradient equals the non-checkpointed gradient at rtol 1e-4.

### W4 SEM: tokenizer value channel, semantic stream, decoder conditioning, strings
- **Owns:** `cu/semantic/{tokenizer,layers,encoder,decoder_xattn,constraints,grammar,bridge,demo_init,objectives}.py`.
- **New tests:** `T/semantic/{test_value_channel.py, test_string_closure.py, test_decoder_conditioning.py}`.
- **Exposes:**
  - NUMERIC tokens = concat[norm(value features); norm(desc+unit)] → proj;
  - `ObservationTokenizer.semantic_context(obs_lists, tools, goals, max_tokens) -> SemanticContext`, newest first, cached, with refs;
  - `QwenSemantic.step/decoder_hidden(..., cond=(Zc, mask))`;
  - `HoleFiller.sample/score(..., cond=)`, decode record v4 (consumed forced chars);
  - DECODE_VERSION bump; old records are refused.
- **Depends on:** seam.
- **Real end-to-end check** (real Qwen3-0.6B + `dom_dev` tokenizer):
  - cos(+1, −1) < 0.5 for numeric tokens at init;
  - ≥ 90% of 50 ow_tiny string slots close by the LM's own choice, with 0 trailing commas;
  - rescored logp equals recomputed logp bit for bit with conditioning on;
  - with gates at init the logits are bit-identical; after 100 BC steps, ablating Z changes logp by > 1 nat per action;
  - code.run demo arguments are representable.

### W5 DEC: decision bottleneck (Mechanism B)
- **Owns:** `cu/model.py` (encode_step, decide, policy_evaluate, policy_probe, behavior_of, Decision assembly), `cu/semantic/actions.py`, `cu/learn/heads.py`, and the new package `cu/decision/{__init__,intent,evaluator,select,fast}.py`.
- **New tests:** `T/dom/{test_dom_bottleneck.py, test_dom_condition_g.py, test_selection_replay.py}`.
- **Exposes:**
  - `IntentHead(cfg)(R, ctx_out, mask) -> I [B,16,D]`;
  - `ActionGenerator.sample/log_prob(I, tools_ctx_out, obs_ctx_out, ...)`; no `tool_prior`; goal reaches the decision only via Z/R/I;
  - `decision.evaluator.score(R, B, G, cand_tokens, dt) -> Q [B,K], preds`;
  - `decision.select(Q, gen_logp, tau, veto) -> (idx, sel_logp, stamp)`;
  - `decision.fast.fast_params(model) -> dict`, the φ set for W8;
  - `Decision.intent/candidates/evaluation`.
- **Depends on:** W3 (sem argument; stubs are fine), W6 (`prior(R, act, ...)` signature, fixed in the seam), W4 (`cond=` API).
- **Fallback if late at T+13:** "bypass-lite", meaning remove `tool_prior` and make pointer and query read only R + `ctx_out`. About 2 h.
- **Real end-to-end check** (real `dom_dev` on a two-goal mech/rapid task):
  - zeroing I_t makes the tool distribution independent of the goal text;
  - Condition G: after imitation, swapping I/R rows makes ≥ 90% of executed actions follow the donor;
  - KL(π ‖ π with I permuted) > 0.1 nats after training (today 4e-9);
  - behaviour logp, including selection, replays exactly;
  - no execution happens without a stamp.

### W6 PRED: world model and beliefs
- **Owns:** `cu/world/{dynamics,outcomes,imagination,planner,planner_runtime,train_seq,voi(new),causal,imagine_train}.py`, `cu/core/{beliefs,bayes,belief_runtime}.py`, `cu/outcome_spec.py`, `cu/learn/meta.py`, `cu/capabilities/{causal,imagination,beliefs}.py`.
- **Tests:** `T/world/*` (made multi-seed), `T/core/test_core_adapt_learning.py` (port of exp_adapt), `T/world/test_consistency.py`.
- **Exposes:**
  - `WorldModel.prior(R, act, belief, dt, sample, z)` through `core.transition_blocks(6)` and adapters;
  - `train_seq.filter_sequence(model, seq, aux, burn_in)`;
  - `train_seq.world_loss(model, seq, filt)`, using `filt.Zs`, with no EMA target and with free bits, mode-latent and consistency terms;
  - `belief_runtime.predictive_log_prob`, `belief_change(fast_prev, fast_new)` (returns kl_weights, kalman_ig, d_entropy), `adapt_loss` (separation-correct, all k);
  - outcome_spec makes the reward unobserved in observation_only;
  - `world.voi.decision_voi(model, state, info_actions, n_y)`.
- **Depends on:** W3 `transition_blocks`.
- **Real end-to-end check** (real `dom_dev` stack, `capabilities/causal.run_causal` with `dom_dev`, 800 steps with LR decay, 3 seeds):
  - t1 > 0.5, t2 max error < 0.3, |t3 contrast − 2| < 0.4, t4 switch correct;
  - consistency MSE/Var < 0.1;
  - with the Kalman layer on, Gaussian mechanism NLL(k=2) ≤ oracle + 0.05 and z-sd in [0.8, 1.2];
  - integrated LP at the first informative step > 0.1 nats;
  - ProbeEnv h=1 evaluator margin between test and safe actions is ≥ 0 once trained.

### W7 LEARN: integrated learner, optimisation, gradient routing
- **Owns:** `cu/learn/{integrated,trainer,ppo,trajectory,returns,explore}.py`.
- **New tests:** `T/learn/test_integrated_learner.py`, `T/dom/test_dom08_gradients.py` (dev in CI; W1 runs the 1B variant).
- **Exposes:**
  - `IntegratedLearner(model, reward_model, label_store, cfg).step(batch) -> UpdateInfo`, which includes a nested named-group gradient table logged as JSON;
  - `meta_step()` delegating to W8;
  - `CombinedTrainer(groups, roles, balance_targets, per_group_clip)`;
  - `ActorCriticObjective(advantage_fn=, reward_int=)` plus a tool-entropy term.
- **Depends on:** W5 (recompute fields), W8 RewardModel (stub until merged), W6 losses.
- **Real end-to-end check** (real `dom_dev`, 4 rapid environments, 20 updates):
  - actor share of core gradient ≥ 0.1 and world share ≤ 0.7;
  - DOM-08 table nonzero after one optimiser step for core blocks 0 / L/2 / L−1, sem projections, Mech C gates, reasoner, beliefs, world model, intent, decoder xattn, exploration/IG path, policy, V_int, predictors;
  - n_stale > 0 routed to V-trace with no StalePolicyError;
  - clip rate < 0.5 after warmup.

### W8 REWARD: R_ψ, verified-label predictors, meta-gradient, credit, autonomy
- **Owns:** `cu/selfeval/{intrinsic,metagrad,evaluator,voi,reward_model(new),label_join(new)}.py`, `cu/credit/*`, `cu/autonomy/*`, `cu/learn/curriculum.py`.
- **New tests:** `T/selfeval/{test_reward_model.py, test_label_join.py, test_metagrad_cam.py}`, `T/credit/test_meta_unbiased_regression.py` (exact versus 2000-repetition check, max|z| < 4, no cosine-only checks).
- **Exposes:**
  - `RewardModel.reward(...)` (detached), `.potential(R, G)`, `.losses(label_batch)` covering ProspectEvaluator and V_L;
  - `label_join.join(label_store, archive_events) -> LabelBatch`, matching by (env_id, episode) and leaving pending labels out;
  - `MetaRewardTrainer(metagrad.MetaRL over decision.fast.fast_params)`, with `.inner_update/.outer_step` and a ψ version;
  - after T+24: credit `decision_features` consumers, RUDDER redistribution of late labels into V_L targets, COCOA candidate estimator, C_chi curriculum.
- **Depends on:** W2 labels, W5 fast params, W6 `belief_change`.
- **Real end-to-end check** (real `dom_dev` on integrity worlds and ProbeEnv through `run_lifetime`):
  - the reward is recomputed under the current ψ after a bump;
  - ∂L_actor/∂ψ ≡ 0;
  - one meta step changes ψ;
  - label canaries never reach actor tensors;
  - ProspectEvaluator ECE < 0.05 on held-out joined labels;
  - the 1B fast-set meta step peaks at ≤ 40 GB (run in W1's slot).

### W9 MEM+EVID: checkpoints, memory hygiene, experiment drivers, statistics, governance
- **Owns:** `cu/memory/{checkpoint,featurize,archive,plasticity,replay,runtime}.py`, `cu/experiments/{exp1,exp2,exp3,predict,stats,summary,baselines,nig_agent,dom_suite(new),meta_suite(new)}.py`, `cu/observe/*`, `superlearner/src/superlearner/verify_completion.py`, `T/test_manifest.py`, `docs/CAM_UNIFIED_EXPERIMENTS.md`, `docs/experiments/U-DOM-1.md, U-MR-1.md, U-OW-1.md, U-SC-1.md` (new).
- **Checkpoints:** `load_weights(allow_new=...)`, arch_hash and code_commit in the metadata; load_weights_only resets fresh/trainer/trackers/RNG.
- **Memory:** ActorView featurizer (memory stays off but leak-free); archive links table.
- **Statistics:** studentised t; paired comparisons with independent seeds.
- **verify_completion:** exact verdict enum, plugin that rejects doubles in integration tests, JUnit records bound to a commit, prereg-before-confirm check.
- **Drivers:** run Conditions A–I and the meta-reward conditions through `run_lifetime` only, with provenance rows (ckpt sha, cfg hash, reward_mode, decisions, GPU-busy seconds, Qwen tokens) and TraceRecorder on.
- **Depends on:** W1 for the canonical loop.
- **Real end-to-end check:** real `dom_dev` trained for 200 updates, saved, reloaded weights-only in a fresh process. The next update consumes 0 source decisions; two loaded processes with the same seed act identically. `dom_suite --condition G --smoke` writes provenance rows that the dashboard renders. A coverage simulation gives ≥ 0.93 at S=3. The synthetic manifests from the audit probe are all rejected.

## 4. Critical path to genuine end-to-end learning with the 1B model (~24 h)

| Time | Gate | Content |
|---|---|---|
| T+2 | G0 | Seam merged and tagged `seam-0`. Dev decide bit-identical; existing suites pass. |
| T+2 to T+10 | Parallel work | W3 gate + state embeddings + sem argument; W4 value channel + string fix + conditioning API; W6 EMA removal, free bits, consistency, prior on R via 6 blocks, belief feat=Ẑ + fusion + observation-only outcome spec; W2 result delivery, reward_mode, LabelSource; W7 learner, balancing, probe; W8 RewardModel default + label join; W9 checkpoint migration + stats; W1 build/budget and the loop with stubs. |
| T+10 | G1 | Merge in order W3 → W4 → W6 → W2. `dom_dev` end-to-end through `run_lifetime` on the legacy decision path. |
| T+13 | G2 | Merge W5 (or bypass-lite), W7, W8. `dom_dev` full path: 4 rapid environments, 200 decisions, 20 updates. DOM-08 dev table nonzero; actor core share ≥ 0.1. |
| T+16 | G3 | 1B `dom` build: full optimiser step ≤ 40 GB; decide B=16, K=4 ≤ 1.5 s; update ≤ 90 s per 128 decisions. Fixes allowed: Adafactor for satellites, microbatching, d_satellite 512. |
| T+18 | G4 | 1B end-to-end on real rapid/mech + ow_tiny: 64 decisions, 2 updates. DOM-08 1B table after one step. Before training, KL(π ‖ π with I permuted) is well above 1e-9. Integrity scan clean. |
| T+20 | G5 | Launch the E1 1B run in the background with resumable checkpoints every 2 h. |
| T+24 | Learning evidence | Rising verified oracle-relative utility on rapid training families against the first-window baseline; logged actor core share ≥ 0.1; at least one ψ update; at least one label join with ProspectEvaluator NLL falling. |

## 5. Feasible experiments and how to run them honestly

Budget: about 85 h remain (the rental ends 2026-11-30 03:00 UTC). Build takes ~24 h, about 45 GPU-hours are left for experiments, and 10 h are reserved for reporting and slip.

**Protocol (applies to everything):**
- Commit U-DOM-1, U-MR-1, U-OW-1 and U-SC-1 before any confirm evaluation. Each states the primary metric, conditions, seeds, decision budget, splits and analysis.
- Tuning and development use rapid *train* families and mech train families. The confirm sets are the rapid *val* split (19 structure-fingerprint held-out families) and mech val/test families with fresh seeds (≥ 500000). Each (learner, seed) is evaluated on confirm exactly once.
- Primary metric: verified oracle-relative utility U = (v − v_null)/(v_oracle − v_null), computed on the operator side from `envs/experience.py` labels. Success means U ≥ 0.5.
- Every CAM condition runs under reward_mode observation_only, asserted by the integrity scan of each run directory.
- Budgets are matched on environment decisions. GPU-busy seconds and Qwen tokens are reported, and any ratio above 2× is flagged.
- Development primaries use ≥ 5 seeds with studentised t intervals. 1B runs are a single seed, labelled "demonstration, no inferential claim".
- Negative results are recorded as FAILED_EMPIRICALLY. No capability_config (2×64) evidence.

**Experiments:**
- **E0, DOM mechanism verification at 1B (2 h).** DOM-08 gradient table after one step. Condition G latent swap (≥ 90% follow the donor after a short imitation phase). Condition F (decoder conditioning off) and Condition H (frozen core): Δlogp and action change on fixed probe contexts. A03 acceptance at dom_150m: forced J=1 against J=4 changes ≥ 20% of decisions on 2-hop ChainEnv.
- **E1, main 1B continual learning run (12 h; G5 onward).** Rapid train families plus mech train families, 16 environments, about 200–300k decisions. Weights-only and continual-on evaluations on confirm families at checkpoints spaced 4 h apart.
  - Baselines at equal decisions: untrained CAM (Condition B); Condition H frozen core with only heads and adapters trained (dom_150m, so it fits concurrently); null and oracle references (free from the operator); Qwen3-4B direct control (`envs/baselines.FrozenSemanticAgent`, Condition A) on a 2-family subset, token-matched.
- **E2, development ablations (dom_dev, Qwen3-0.6B, 5 seeds × 20k decisions each, about 40 min per run, 2 concurrent; ≈ 8 h).**
  - Conditions: full; no state gate; bypass restored (`tool_prior` plus raw goal, n_intent=0); no loss balancing; fixed-Φ shaping with no meta step; sparse verified terminal reward (β_ver, non-compliant baseline D2); legacy shaped evaluator reward (non-compliant reference D1).
  - Primary: confirm-family U after 20k decisions, full against each ablation.
- **E3, meta-reward suite (dom_dev, 5 seeds; ≈ 3 h).** Exp J on ProbeEnv (test-first rate, return against optimum 10; the uninformative variant must stay ≤ 0.1) and Exp F integrity worlds (`envs/integrity.py`: an adversarial agent cannot inflate R_ψ). Conditions: R_ψ meta-learned; Φ only; LP-only (old lp_coef); extrinsic.
- **E4, OpenWorld demonstration (1B from the E1 checkpoint, ≈ 6 h).** ow_tiny open_economy, observation-only, 2 parallel training-range seeds, one 8-sim-day lifetime each (~1,100 decisions). Report verified success, executed-ok rate, goal-directed action classes, and how often the evaluator vetoed or regenerated. References: ExampleInductor (scripted, cheap), FrozenSemanticAgent Qwen3-4B with token matching, untrained CAM. A single lifetime is labelled as a demonstration only (A18 partial).
- **E5, scaling (if time; ≈ 8 h).** dom_dev (14.5M core) × 3 seeds, dom_150m × 2 seeds and E1's 1B curve on the same rapid suite at matched decisions: decisions-to-U ≥ 0.5 and confirm U at equal decisions and at equal GPU-seconds. The 500M point is dropped.
- **E6, optional cheap rows (≈ 2 h, CPU-heavy).** Exp1 support/query prediction (dom_dev trained against nig_meta) and Exp2 (CAM against KG) on the existing dev sets, using the repaired harness with provenance.

**Not feasible this rental (recorded as NOT RUN with reasons):**
- Exp3 full A/B/C/D/F transfer at 1B (only a dev-scale B-vs-A pair if E5 is cut).
- Exp4 10/30/100 lifetimes.
- OW-1..OW-7 CAM protocol runs.
- 8/16-actor distributed runs at 1B.
- DOM-10 progressive unfreezing (at most a LoRA on/off retention probe).
- Memory/retrieval learning (E03).
- Tier 2/3 Crucible (BLOCKED upstream).
- 500M scaling.

## 6. Decisions where I override the audits

- **Distributed central inference server (40 h):** deferred. Single-process batched acting is the DOM path on one GPU.
- **World transition:** reuses the last 6 core blocks by reference instead of a new 150–250M transformer. This is declared as a deviation from the spec 6.1 full-core transition. It keeps F_θ shared and makes evaluator planning cheap.
- **Particle filter:** the RBPF is deferred and gated fusion is used now.
- **Retriever:** the redesign (60 h) is deferred. Memory is off in all DOM runs, declared, with the featurizer leak fixed anyway.
- **Intent:** deterministic, and the policy is a candidate-conditional surrogate over recorded candidates. This is exact and recomputable; it is documented as conditional on C_t, and COCOA-candidate credit matches it.
- **Labels and the actor reward:** verified labels never enter the actor reward. They enter only through Φ/V_L targets and the ψ outer loss.

**Remaining risks:**
- 1B checkpoints (2–4 GB in bf16) cannot be stored durably on GitHub. Push configs, seeds, commits and evidence JSON, and state that the weights are ephemeral.
- Closing the bypass will first lower measured task scores. Judge it by Condition G and confirm U, not by the old raw-path numbers.
- If Φ stays near constant because labels are sparse, the actor signal reduces to information gain minus cost. E3 tests whether the meta step adds value. Report honestly if it does not.
