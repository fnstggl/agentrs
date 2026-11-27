# CAM Live — model architecture as built (DL-02)

Describes the code at commit `d3ce519` (read 2026-11-27 23:12–23:25 UTC; `src/` and `configs/` are identical at
`55a0f4c`). Other workstreams were editing `agent.py` and `runtime/loop.py` while this was written: the working tree
then held uncommitted changes to the backbone prompt (`AgentSettings.proposer_tools` 12 → 0, i.e. every permitted
tool listed, and prompt fitting). This document describes the committed version and says where that matters.

Written from the code. Where `CAM_LIVE_ARCHITECTURE.md` (the design contract), `docs/A/DESIGN.md` or a status file
says something else, the code is described and the difference is listed in §10. Paths are relative to
`cam_live/src/cam_live/`; a component is cited as `file:Class.method` or `file:function`.

Evidence of every count below: `docs/evidence/DOC/param_counts.json` (commands in it and in §1.4).

## 0. Components at a glance

| component | code | group |
|---|---|---|
| frozen backbone Qwen/Qwen3-4B | `nn/backbone.py:SemanticBackbone` (built by `nn/backbone.py:build_backbone`) | `backbone_frozen` |
| LoRA adapters (q/v, all 36 layers, r 32) | `nn/backbone.py:LoRALinear`, installed in `_Backbone._finish_init` | `semantic_adapters` |
| latent bridge (belief → 8 virtual tokens read by cross-attention in 6 decoder layers) | `nn/backbone.py:LatentBridge`, hooks `_Backbone._layer_hook` | `semantic_adapters` |
| experience tokenizer | `nn/tokens.py:ExperienceTokenizer` (records → items: `nn/tokens.py:to_items`) | `experiential` |
| experiential core (50m preset) | `nn/core.py:ExperientialCore` | `experiential` |
| action / goal / query encoders | `nn/decision.py:ActionEncoder`, `GoalEncoder`, `QueryEncoder` | `experiential` |
| decision trunk z and feature map phi | `nn/decision.py:DecisionTrunk`, `FeatureMap` | `experiential` |
| learned prior of the fast layer | `nn/fast.py:FastPrior` | `experiential` |
| fast Bayesian beliefs (state, not parameters) | `nn/fast.py:FastBeliefState` (owned by `credit/learner.py:ContinualLearner.fast_state`) | — |
| assembled model | `nn/model.py:CamLiveModel` (`build_model`), parameter groups `nn/api.py:GROUPS` | — |
| world model, latent transition, goal evaluator, candidate policy, downstream-VOI | `predict/heads.py:attach_heads` → `predict/world.py:WorldHeads`, `LatentTransition`, `predict/evaluator.py:GoalEvaluator`, `predict/policy.py:CandidatePolicy`, `predict/voi.py:DownstreamValueEstimator` | `heads` |
| RUDDER return decomposer, COCOA hindsight model | `credit/rudder.py:ReturnDecomposer` (via `credit/providers.py:ReturnDecompositionProvider`), `credit/cocoa.py:HindsightModel` (via `HindsightProvider`) | `heads` |
| intrinsic reward R_psi with lifetime context | `reward/intrinsic.py:IntrinsicReward`, `LifetimeContext` (registered by `reward/metalearner.py:MetaLearner.__init__`) | `reward` |
| meta-learner (inner policy update, outer update of psi) | `reward/metalearner.py:MetaLearner`, `reward/inner.py:PolicyImprovementLoss`, `reward/meta.py:MetaGradient` | owns the `reward` optimizer |
| continual learner (fast / frequent / slow) | `credit/learner.py:ContinualLearner` | owns the `experiential`, `heads`, `semantic_adapters` optimizers |
| checkpoints, promotion, rollback, export | `credit/checkpoint.py:CheckpointManager`, `promote_learner`, `export_weights_only`, `load_exported_model` | — |
| composition | `agent.py:CamLiveAgent.__init__`, factory `agent.py:build_agent`, heads for an export `agent.py:register_agent_heads` | — |

## 1. Parameter counts (measured)

### 1.1 Default configuration (`configs/cam_live.yaml`: Qwen3-4B + LoRA + bridge + 50m core), integrated agent

Measured by constructing `CamLiveAgent(cfg, run_dir, model=model)` (as `build_agent` does, so every head the agent
registers exists) and calling `nn/model.py:CamLiveModel.count_parameters`.

| group (`nn.api.GROUPS`) | parameters | of which |
|---|---:|---|
| `backbone_frozen` | **4,022,468,096** | Qwen3-4B weights, `requires_grad=False` (0 trainable) |
| `semantic_adapters` | **16,713,856** | LoRA 11,796,480 (72 modules = 36 layers × {q_proj, v_proj}, r 32: 36 × 32 × [(2560+4096)+(2560+1024)]) · latent bridge 4,917,376 |
| `experiential` | **50,953,765** | tokenizer 4,263,746 · core 36,615,648 · encoders 1,977,600 · decision trunk 4,720,128 · feature map 3,343,104 · fast prior 33,539 |
| `reward` | **949,121** | `reward` = `IntrinsicReward` (d_hidden 256, d_life 64, incl. the GRU of the lifetime context) |
| `heads` | **6,055,337** | `predict_world` 1,022,954 · `predict_transition` 1,977,600 · `predict_evaluator` 783,192 · `predict_policy` 702,339 · `predict_downstream` 84,195 · `credit_rudder` 559,361 · `credit_cocoa` 925,696 |
| **total persistent trainable** | **74,672,079** | experiential + adapters + reward + heads; every one has `requires_grad=True` |

* The model alone (before the agent registers heads) has 67,667,621 trainable parameters, identical to
  workstream A's GPU measurement in `docs/evidence/A/params.json` (which counted no heads).
* The adaptable subset theta of the inner (meta-differentiated) policy update = `predict_policy` + decision trunk =
  702,339 + 4,720,128 = **5,422,467** (`reward/inner.py:AdaptableSubset`).
* Not parameters, not in the counts: the frozen randomised prior networks of the world-model ensemble are buffers,
  530,410 numbers (`predict/world.py:EnsembleLinear(trainable=False)`); running normalisation statistics of R_psi;
  calibration statistics of the backbone (`cal_mu`, `cal_sd`, 2 × 7680) and the bridge (`h_mu`, `h_sd`).
* Fast state: per outcome kind a mean of F = 128 and a full F × F covariance, i.e. 16,512 numbers per kind
  (`count_parameters()["fast_state_numbers_per_kind"]`), 4 kinds (+ the normal-inverse-gamma a, b of the real kind).

### 1.2 Debug configuration (`CamLiveConfig.debug()`: tiny random LM + debug core; CPU tests only)

frozen 131,392 · adapters 42,304 (LoRA 1,024, bridge 41,280) · experiential 202,421 · reward 227,521 · heads 709,269
(world 47,996, transition 107,424, evaluator 27,218, policy 29,315, downstream 3,491, rudder 289,025,
cocoa 204,800) · total trainable 1,181,515. The tiny LM is never reported as the backbone.

### 1.3 Other scale presets (experiential group only)

Measured by `python -m cam_live.nn.bench` (workstream A, `docs/evidence/A/params.json`): 5m 5,044,741 · 20m
20,087,333 · 50m 50,953,765 · 150m 151,022,853 · 500m 502,708,549 · 1b 1,000,547,333. Only `debug` and `50m` have been
run as part of the agent; 150m–1b were instantiated and counted on CPU only.

### 1.4 How the counts were produced

```
CAM_LIVE_HOME=<scratch>/DOC/arch/home /workspace/camlive/.venv/bin/python -I <scratch>/DOC/arch/count_params.py \
    <scratch>/DOC/arch/param_counts_raw.json          # builds debug and default agents on CPU, ~92 s for the default
/workspace/camlive/.venv/bin/python -I <scratch>/DOC/arch/extra_counts.py   # frozen prior buffers of the world heads
```

(`<scratch>` = `/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad`; both scripts are kept in
`docs/evidence/DOC/scripts/count_params.py` and `extra_counts.py`.) The script calls only
`CamLiveConfig.load()/debug()`, `nn.model.build_model`, `agent.CamLiveAgent(..., model=model)`,
`CamLiveModel.count_parameters()` and `param_groups()`. On CPU the backbone loads in float32
(`SemanticBackbone.__init__`); counts do not depend on dtype. No GPU was used.

## 2. Backbone and semantic adapters

**Backbone** (`nn/backbone.py:SemanticBackbone`). `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
revision="1cfa9a7208912126459214e8b04321603b3df60c", local_files_only=True)` from `HF_HOME` (default
`/workspace/.hf_home`), offline. dtype: `bfloat16` on CUDA, float32 on CPU. 36 layers, hidden 2560, vocabulary
151,936. Every LM parameter gets `requires_grad=False` and the LM stays in eval mode (`_Backbone._finish_init`,
`_Backbone.train`). The final RMSNorm and the LM head are recomputed in float32 (`_norm32`, `_HeadLogits`,
`_TokenLogprob`; checked against the module by `_check_final_norm`). `name="tiny"` builds `TinyBackbone` instead.

**How embeddings are used** (`_Backbone.embed`, `_pooled`, `_cached`, `_standardise`). A text is tokenised
(at most `max_text_tokens` = 384: head 3/4 + tail 1/4, truncation counted in `stats["truncated_tokens"]`), the hidden
states of the layers at fractional depths (0.5, 0.75, 1.0) — layer ids 17, 26, 35 — are mask-mean-pooled and
concatenated (d_sem = 3 × 2560 = 7680), then standardised per dimension with statistics of a fixed neutral
calibration corpus (`CALIBRATION_TEXTS`, winsorised at ±20). Fixed forward shapes per length bucket make a text's
embedding independent of the other texts of a call. Results are cached by text hash (float16, up to 200,000 items).
`with_grad=True` returns the same (cached) value plus a graph into the LoRA weights (`canon + (g − g.detach())`).
`_Backbone.sync_caches` compares the adapter tensors by content at every call and drops the caches when they changed.
Everything downstream of the backbone is float32 (`nn/api.py` contract).

**How generation is used** (`_Backbone.generate`, `_generate_rows`): sampling with a KV cache, per-call
`torch.Generator`, temperature (0.7 for the agent's proposer, `AgentSettings.proposer_temperature`), optional top-p,
stop on the end-of-turn ids, conditioned on `latent` through the bridge. The agent's proposer
(`agent.py:AgentProposer.propose`) builds one prompt with the Qwen chat template, thinking disabled
(`SemanticBackbone.chat_prompt(..., thinking=False)`), and asks for up to `proposer_k` = 8 JSON-lines actions with at
most 256 new tokens. `generation_logprob` (differentiable w.r.t. LoRA, bridge and latent) is used by the slow
`bridge_grounding` loss (§7) and by nothing in the decision path.

**LoRA** (`nn/backbone.py:LoRALinear`): `base(x) + (alpha/r)·B(A(x))` on `q_proj` and `v_proj` of all 36 attention
blocks, r = 32, alpha = 32; A uniform(±1/√in), B zeros (exact no-op at init); adapter weights float32, cast in the
forward; an all-zero B is skipped at inference.

**Latent bridge** (`nn/backbone.py:LatentBridge`): `memory(b)` = RMSNorm(Linear(768 → 8 × 128)(RMSNorm(b)) + pos):
8 virtual tokens of width 128. Six decoder layers (ids 0, 7, 14, 21, 28, 35) get a forward hook that adds
`h_scale · CrossAttention(standardise(h), memory)` (4 heads) to the residual stream. The cross-attention output is
zero-initialised: the bridge is an exact no-op at initialisation (measured bit-identical logits,
`docs/evidence/A/params.json` `bridge_noop_check`) yet receives gradient. The virtual tokens are not placed in the
token sequence. Used whenever `generate`/`generation_logprob` gets `latent=b`.

## 3. Experience encoding and the experiential core

**Tokens** (`nn/tokens.py:ExperienceTokenizer.encode`): every contract record becomes `ExperienceItem`s
(`nn/tokens.py:to_items`, `action_item`, `outcome_item`, `time_item`, `resource_items`). One token per item is the sum of
(i) content: the per-tap RMS-normalised semantic embedding through a low-rank projection 7680 → 384 → 768
(`sem_rank` 384 for 50m, `nn/presets.py:NN_PRESETS`), RMS-normalised; a learned vector for items without text;
(ii) a separate value channel for up to `max_values` = 4 exact quantities: odd features (sign, tanh of asinh at four
scales, sin of asinh·freq) and even features, bound multiplicatively to a hashed per-metric-name embedding (two
2048-bucket tables) and a unit embedding, then normalised on its own (`value_channel`); raw float64 values ride
along in `TokenBatch.values`; (iii) item-kind and trust embeddings; (iv) features of the gap to the previous token.

**Core** (`nn/core.py:ExperientialCore.forward`): d_model 768, 6 pre-norm blocks, 12 heads, SwiGLU d_ff 1024
(`nn/core.py:Block`); causal self-attention with rotary positions over the event index plus a learned per-head bias of
the elapsed time `t_i − t_j` (`elapsed_features`: 16 radial + 12 Fourier features of log1p seconds); every second
block (blocks 0, 2, 4) cross-attends to 8 persistent slots conditioned on the time since the previous call; after the
blocks a gated slot write (`gate_cand` bias −2 at init); the belief is
`b = RMSNorm(h_last + direct(RMSNorm(x_last)) + state_read(mean(slots)))` — the direct path from the newest event.
`CoreOutput.state` holds the updated slots, but **no caller carries it**: every call site passes `state=None`
(`agent.py:CamLiveAgent._belief`, `predict/predictor.py:Predictor.evaluate_candidates`, `learning_api.py:encode_decisions`,
`credit/learner.py:encode_with_adapter_grad`, `reward/inner.py:PolicyImprovementLoss._encode_next`,
`predict/planner.py`), so each belief is computed from the learned initial slots and the re-encoded context window
(§10, item 1). The context window (`agent.py:CamLiveAgent._window`) is the goal item, the tool-list item and the
newest items of the current objective, at most `max_context` = 256 tokens; it starts empty at every new objective
(`set_objective`).

**Encoders, trunk, features** (`nn/decision.py`): `ActionEncoder` (rendered action text + cost quantities through
the tokenizer, argument values through the value channel (`CamLiveModel.encode_actions`), modality / level
embeddings, timeout and step count); `GoalEncoder`; `QueryEncoder` (metric text, outcome kind, horizon, exposure,
goal-relevance, hashed metric name shared with the value channel). `DecisionTrunk(b, e_a, e_g) → z` [B, N, 768]:
each input RMS-normalised on its own, additive + multiplicative action × situation interaction, SwiGLU residual.
`FeatureMap(z, e_q) → phi` [B, N, Q, 128] with ‖phi‖₂ = 1 and a constant last coordinate 1/√128.

## 4. Fast Bayesian beliefs (state m_t, no gradient)

`nn/fast.py:FastBeliefState`: for each outcome kind (real, count, binary, time-to-event; categorical is not modelled)
one Bayesian last layer `w ~ N(mu, Sigma)` over phi, float64. Real: normal-inverse-gamma (noise inferred, Student-t
predictive; `NNConfig.fast_noise_mode="nig"`). Count (Poisson with exposure, tempered by a learned dispersion),
binary (binomial) and time-to-event (exponential, right-censoring via the Poisson-exposure identity): Laplace updates
at the mode found by a safeguarded Newton search, applied in Joseph form (`glm_update`, `gaussian_update`,
`nig_update`). Stabilised forgetting towards the prior (`forget`, gamma 0.999 per update) and a CUSUM change detector
on the standardised surprise (h = 12, `FastBeliefState.update`). The prior mean and Cholesky-parameterised covariance
per kind, the noise, its inverse-gamma shape and the dispersion are the learned `FastPrior` (experiential group),
trained ALPaCA-style through `nn/fast.py:online_nll` by the `fast_prior` loss provider
(`predict/providers.py:FastPriorLoss`). One posterior per kind is shared by all metrics of that kind; it is reset to the
learned prior at every new objective (`agent.py:CamLiveAgent.set_objective` → `ContinualLearner.reset_fast`). The
mathematics is in `docs/MATH_NOTES.md`.

## 5. Heads on the shared representation

Built by `predict/heads.py:attach_heads` from `PredictConfig` (hidden 192, seed-forked RNG) and registered in group
`heads`:

* **World model** `predict/world.py:WorldHeads.forward`: `p(y | z, q, horizon)` for real (Student-t / normal), count
  (negative binomial with exposure), binary (Bernoulli), time-to-event (12-bin discrete hazard, right-censored) and
  categorical (≤ 16) outcomes, the query and the horizon as inputs (no fixed target list). A bootstrapped ensemble of
  5 members, each with its own frozen randomised prior network; inputs z, e_q, horizon features, kind, phi. Combined
  with the fast layer in `predict/predictor.py:Predictor._means` / `_samples` as stated in `predict/heads.py:fast_inputs`
  (count: fast layer models log(actual/head rate); real: the normalised residual; time-to-event: log hazard ratio;
  binary: posterior shift of the fast logit added to the head logit). Epistemic = ensemble spread + fast-posterior
  variance, aleatoric = members' mean predictive variance, reported separately.
* **Latent transition** `predict/world.py:LatentTransition`: 3-member ensemble predicting the next belief `b'` from
  `(z, dt, b)` with diagonal variance; used only by the latent planner `predict/planner.py:LatentPlanner`
  (imagination; outputs flagged imagined).
* **Goal evaluator** `predict/evaluator.py:GoalEvaluator`: `forward(b, e_g, horizon)` (V) and
  `forward_candidates(z, e_g, horizon)` (Q): success logit, magnitude (negative binomial / Student-t), time to
  achievement (12-bin hazard, censored), 3 resource heads. Supervised only by signature-checked verified outcomes
  (`predict/providers.py:EvaluatorLoss`).
* **Candidate policy** `predict/policy.py:CandidatePolicy.forward(z, a_mask, b)`: logits over a variable candidate set,
  with a pooled summary of the alternatives and a multiplicative z × b path (so the distribution depends on the
  belief), plus V and Q read-outs. `functional_forward` evaluates it with explicit parameters (`torch.func`).
* **Downstream-VOI estimator** `predict/voi.py:DownstreamValueEstimator`: Student-t prediction of realised downstream
  usefulness from z and five signals (`predict/heads.py:VOI_SIGNALS`); scored against realised values by
  `score_downstream`.

**Selection** (`predict/policy.py:DecisionMaker.select`, rules `predict/policy.py:RULES`): policy, thompson, ucb,
greedy, voi, thompson_policy, randomized; every rule builds the full probability vector first and samples from it,
with an epsilon floor (0.02), so recorded propensities are exact. Agent default `thompson_policy`
(`agent.py:AgentSettings.selection_rule`): Thompson samples of each candidate's goal value (standardised within the
decision) plus the policy's centred log-probability beyond the world model's ranking, weighted by
λ = β_π⁺/β_q drawn from `predict/policy.py:SelectionCalibration` (Bayesian linear regression of verified progress on
those two numbers, forgetting 0.99; state kept on the policy head). With `experiment_rate` 0.1 the agent instead
registers a randomised comparison (`DecisionMaker.propose_experiment` / `select_experiment`) when the order of two
candidates is uncertain.

**VOI** (`predict/predictor.py:Predictor._voi`, `predict/voi.py`): `E_y[max_a' Q(b_updated(a, y), a')] − max_a' Q(b, a')`
with `y` sampled from the fast posterior and the fast update applied (8 samples, ≤ 16 candidates); the slow path that
re-encodes an imagined result into the context is off (`PredictConfig.voi_slow_path=False`).

## 6. Intrinsic reward and meta-learning

`reward/intrinsic.py:IntrinsicReward.forward(b_t, e_a, b_next, e_g, signals, life_ctx)`: running-normalised inputs
(statistics frozen inside a meta window, `meta_window`), `latent_in(4·768 → 256)` over [b, e_a, b_next − b, e_g],
`signal_in` over the normalised signal vector and its mask (`reward/signals.py:SIGNAL_NAMES`, schema 1.0), plus
`life_in(LifetimeContext.state)` (a GRUCell 256 → 64), SiLU MLP, output bounded to (−1, 1)
(`reward/integrity.py:bound_reward`). `version` is a hash of parameters and statistics.

`reward/metalearner.py:MetaLearner` owns the only optimizer of the reward group (Adam, lr 1e-4, clip 0.5).
* `internal_rewards(records)` (runtime step 13) assembles the signals (`reward/assemble.py:SignalAssembler`), the
  labelled bootstrap reward (`reward/bootstrap.py:BootstrapReward`) and R_psi for each new decision.
* `policy_update(records)` (every `policy_update_every` = 4 rewarded decisions, from `agent.py:CamLiveAgent.learn`)
  applies `theta ← theta − alpha ∇ L_inner(theta; psi, D_support)` to the live adaptable subset (policy head +
  decision trunk) with `reward/inner.py:PolicyImprovementLoss` (K = 2 steps, step size chosen by a trust region:
  mean KL ≈ `target_kl` 0.05, alpha in [0.01, 2.0]) and records an update event (snapshot of theta, support batch,
  hyper-parameters). The live reward is the mix `bootstrap_mix · n(r_bootstrap) + (1 − mix) · n(R_psi)` plus the
  verified/predicted goal term; the mix starts at 1.0 (`MetaConfig.mix`).
* `meta_update()` (step 15) needs ≥ `meta_min_independent_outcomes` = 8 independent verified groups; it recomputes
  theta_K(psi) functionally on the cached latents (`reward/meta.py:MetaGradient`, `torch.func`), scores the later query
  decisions under pi_{theta_K} (on-policy only if the logged policy is the recomputed one, else clipped importance
  weights with ESS ≥ 2), back-propagates to psi and steps the reward optimizer, or returns a skip record with the
  reason and leaves psi bit-identical. The differentiated update uses `meta_mix` = 0 (pure learned reward).
* Utilities come only from `reward/integrity.py:OutcomeLedger` (signed objective outcomes, authenticated verifier
  measurements). The mixer moves the bootstrap mix by 0.25 per non-inferiority decision on held-out groups
  (`reward/bootstrap.py:RewardMixer`).

## 7. Credit and continual learning

**Store and linking**: `credit/replay.py:SqliteExperienceStore` (`<run_dir>/learner/experience.sqlite`, WAL),
`credit/links.py:OutcomeLinker` (attribution, de-duplication by `dedup_key`, re-training queue).
**Credit**: `credit/providers.py:CreditAssigner.assign` (called by `ContinualLearner.learn` for the triggering
records) writes per-decision credit of three estimators into `ExperienceRecord.credit`: `simple`
(`credit/simple.py:credit_from_links`), RUDDER (`credit/rudder.py:ReturnDecomposer`: 2-layer causal transformer,
width 128, over [z, b] of a closed objective's decisions; redistributed return-to-go) and COCOA
(`credit/cocoa.py:HindsightModel`: 4-member ensemble, policy-logit prior, elapsed-time input; refuses unsupported
cases). `credit/select.py:EstimatorSelector` keeps `simple` as the training default until another estimator agrees with
≥ 5 randomised-experiment differences.

**Learner** `credit/learner.py:ContinualLearner`: one AdamW per learner group (experiential lr 3e-4, heads 3e-4,
adapters 5e-5; weight decay 0.01 on matrices; 8 warm-up steps; per-group norm clip 1.0 / 1.0 / 0.5), gradients by
`torch.autograd.grad` per provider, one provider's share of the experiential gradient capped at 0.9; reward
parameters are refused (`_sync_optimizers`).
* fast: `Predictor.observe_outcomes` → `FastBeliefState.update` (step 5, `agent.py:CamLiveAgent.update_beliefs`).
* frequent (`poll` → `learn` → `_frequent_update`): triggered by ≥ 4 matured independent outcome units; providers
  registered by the agent: `world_model`, `fast_prior`, `evaluator` (`predict/providers.py:default_providers`),
  `rudder`, `cocoa`, `retention` (`credit/retention.py:RetentionProvider`: penalised change of b and z on anchors of
  earlier objectives).
* slow (`_slow_update`, every `adapter_every` = 8 frequent updates): `world_model_adapters`, `evaluator_adapters`,
  `bridge_grounding` (`credit/providers.py:BridgeGroundingProvider`: −log p_LM(observed consequence text | action,
  belief via bridge)); a step is kept only if a held-out loss did not rise beyond a noise-derived tolerance, else undone
  exactly.
* every update writes a `LearningUpdateRecord` (hashes before/after, per-group deltas, behavioural probe).
The policy head has no loss among these providers; it changes only through `MetaLearner.policy_update`. The decision
trunk changes through both the learner (experiential group) and `MetaLearner.policy_update`.

**Checkpoints** (`credit/checkpoint.py`): atomic directories (tmp → fsync → rename) with weights per group as
safetensors, optimizer state by parameter name, fast state, lifetime context, RNG, `state.json`, a manifest with
hashes and `CamLiveModel.arch_signature`; tolerant loading of added heads (`load_weights`, reported). Promotion
(`agent.py:CamLiveAgent.promote` → `promote_learner` → `CheckpointManager.promote`, every 24 frequent updates and at
the end of an objective) is gated on held-out provider validation and retention metrics; a refusal rolls back to the
last promoted checkpoint except R_psi, its optimizer and (unless policy retention is a reason) the policy head.
Weights-only export: `agent.py:CamLiveAgent.export_weights` → `weights.safetensors` + `config.json`, rebuilt by
`load_exported_model(dest, register_heads=agent.register_agent_heads)`.

## 8. Data flow of one decision

`runtime/loop.py:LiveRuntime.step` calls the agent in the order of spec 15.1.

| step | call | what happens (functions) |
|---|---|---|
| 1 objective | `CamLiveAgent.set_objective(learner_view(task))` | new objective: promotion attempt for the old one, `MetaLearner.new_lifetime`, `ContinualLearner.reset_fast`, goal item into the store |
| 2–3 observation | `CamLiveAgent.ingest(observations)` | `_note_obs` (items via `_put`, interactive surface), `OutcomeLinker.reconcile(events=...)` |
| 4 outcomes | `CamLiveAgent.reconcile(outcomes, verified)` | `OutcomeLinker.reconcile` (dedup, links), `Predictor.resolve` (scores registered predictions), outcome items, matured outcomes buffered; `_annotate_verified_delta`; `_verified` → signature check → evaluator supervision, `MetaLearner.admit_objective` |
| 5 belief | `CamLiveAgent.update_beliefs()` | `Predictor.observe_outcomes` → `FastBeliefState.update` (fast `LearningUpdateRecord`); `Predictor.belief_value` → b and the evaluator's success probability |
| 6 candidates | `CamLiveAgent.propose(tools, k)` | b = `core_forward(tokenize(window))`; `predict/candidates.py:schema_candidates`, `agent.py:grounded_candidates`, `_skill_candidates` → `_feasible` → `policy_distribution` ranks the pool → `AgentProposer.propose` (`model.generate(prompt, latent=b)` through the bridge, `normalise_proposal`, `predict/candidates.py:parse_actions`) → `_offer` (C_t ≤ 16, inclusion probabilities) |
| 7–8 prediction | `CamLiveAgent.evaluate(C_t)` | `Predictor.evaluate_candidates`: tokenize → `core_forward` → b; `encode_actions` → e_a; `encode_goals` → e_g; `decision_latent` → z; `GoalEvaluator`; `CandidatePolicy`; `_queries_for` → `world_batch` (`encode_queries`, `features` → phi, `WorldHeads`) → `_means` / `_samples` with the fast state → `_voi`; every prediction registered in the hash-chained `PredictionRegistry` before execution; `LatentPlanner.plan` (imagined values recorded only) |
| 9 selection | `CamLiveAgent.select(evals)` | randomised experiment (`DecisionMaker.propose_experiment`) or `DecisionMaker.select(mode="thompson_policy")` → `Decision` with every candidate's probability |
| 10–11 execution | runtime | `env.validate_action`, `PolicyEngine.authorize`, executor (or approval queue) — no model code |
| 12 outcome | `CamLiveAgent.record_result(decision, result)` | action item, `Predictor.adopt_queries` / `commit`, `SqliteExperienceStore.append(ExperienceRecord)`, `mark_for_retraining`, immediate outcomes resolved |
| 13 reward | `CamLiveAgent.internal_rewards()` | `MetaLearner.internal_rewards` (signals, bootstrap reward, R_psi) written to the records |
| 14 credit + update | `CamLiveAgent.learn(budget)` | `ContinualLearner.poll` → `learn` (`CreditAssigner.assign`, frequent / slow updates); `MetaLearner.policy_update` every 4 rewarded decisions; `_measure_planner`; `promote` |
| 15 meta | `CamLiveAgent.meta_learn(budget)` | `MetaLearner.meta_update` (applied or skipped with reason) |
| 16 save | `CamLiveAgent.save(path)` | storage guard, store flush, learner checkpoint |

## 9. What is shared, what is per objective

Kept across objectives: all trainable weights (backbone adapters, experiential core, heads, R_psi), optimizer states,
the learned fast prior, the selection calibration, the experience store, registries. Reset at a new objective: the
fast posteriors (to the learned prior), the context window, R_psi's lifetime context (`MetaLearner.new_lifetime`),
per-objective bookkeeping. The experiential core carries no recurrent state between calls (§3).

## 10. Not implemented, or different from the design (`CAM_LIVE_ARCHITECTURE.md` §2–5)

1. **Persistent core slots are not carried.** The design's `m_t` = core slots + fast state, with a gated slot update
   "carrying state between calls". The update is computed (`ExperientialCore.forward` returns `CoreOutput.state`), but
   every caller passes `state=None` and discards the returned state (call sites in §3). In practice the state between
   decisions is the re-encoded context window (≤ 256 tokens of the current objective) plus `FastBeliefState`; across
   objectives only weights and the learned prior carry anything.
2. **R_psi's lifetime context is reset at every objective change.** Design §4: `life_ctx` is "not reset at objective
   boundaries". `CamLiveAgent.set_objective` calls `MetaLearner.new_lifetime()`, which zeroes
   `LifetimeContext.state`: the agent treats each objective as one lifetime.
3. **Latent bridge = cross-attention only.** Virtual prefix tokens are read by cross-attention in 6 layers; they are
   never inserted into the token sequence (change request A-2). Design allowed both; only the cross-attention form
   exists.
4. **Fast layer scope.** One posterior per outcome kind, shared by all metrics of that kind (not per metric or
   query); categorical outcomes are head-only; time-to-event is constant-hazard; binary outcomes are combined with the
   head by a posterior-shift approximation. In A's comparison the learned-prior layer is worse than a GRU baseline
   at support < 3 for Gaussian targets and only level with it for counts (`docs/status/A.json` scientific_failures).
5. **The live policy step is applied by the meta-learner, not by the continual learner**, with a trust-region step
   size and the live bootstrap mix (1.0 at start, i.e. bootstrap reward only), while the differentiated update uses
   the pure learned reward (`meta_mix` 0) from the same snapshot and is evaluated by importance weighting. The design
   says "the same update is what the continual learner actually applies to the live policy"; it is the same function
   (`PolicyImprovementLoss`), with a different mix and step size unless the mix reaches 0.
6. **Selection.** The default rule is `thompson_policy` (added by LRN; not in the design's list, which it combines).
   Imagined planner values are recorded in `CandidateEvaluation.extras` and used by no selection rule.
7. **Credit default is `simple`.** RUDDER and COCOA are trained as frequent providers and their credit is written, but
   neither has been promoted to the training default (`EstimatorSelector` needs ≥ 5 randomised experiments;
   `docs/INTEGRATION.md` §1: neither passed D's held-out gate).
8. **VOI** is computed through the fast-layer update on phi; the slow path (imagined result entering the context)
   exists but is off by default.
9. `CamLiveModel` does not override `LatentModelAPI.decision_trunk()` (it returns `None`);
   `reward/inner.py:AdaptableSubset` reaches the trunk through `model.exp.trunk` instead.
10. **Not run / not established**: Qwen3-8B (config change only, never loaded); presets 150m–1b never trained; adapter
    learning from real experience and the usefulness of the bridge after training are not evaluated
    (`docs/status/A.json` not_done); `generation_logprob` on the bf16 trunk differs from float32 by up to 0.36 nats on
    a one-token completion; features phi are ill-conditioned at initialisation on real inputs (Gram eigenvalue ratio
    2.3e4). Whether learning improves decisions is not established (`docs/INTEGRATION.md` §8).
11. Matches the design: frozen bf16 backbone with LoRA on q/v (11.8 M), bridge an exact no-op at init, multi-depth
    cached embedding invalidated on adapter change, separate value channel, causal core with time-aware attention and
    direct current-event path, 50m preset ≈ 50 M (50.95 M), shared trunk and feature map, exact/ADF fast updates in
    float64 with forgetting and change detection, learned ALPaCA prior, per-group clipping, ensembles with
    separately reported epistemic/aleatoric uncertainty, three learning timescales with held-out slow guard,
    atomic checkpoints with promotion, rollback and weights-only export.
