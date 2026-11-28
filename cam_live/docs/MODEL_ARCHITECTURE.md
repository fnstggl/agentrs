# CAM Live — model architecture as built (DL-02)

Describes the code at commit `fdab7d3` (read 2026-11-28 07:16–07:30 UTC). HEAD was `fdab7d3` while this was written;
`git log fdab7d3..HEAD -- cam_live/src cam_live/tools cam_live/configs cam_live/tasks cam_live/pyproject.toml` is
empty. This version replaces the one that described `d3ce519`.

Written from the code. Where `CAM_LIVE_ARCHITECTURE.md` (the design contract), `docs/A/DESIGN.md` or a status file
says something else, the code is described and the difference is listed in §11. Paths are relative to
`cam_live/src/cam_live/`; a component is cited as `file:Class.method` or `file:function`.

What changed in the model code between `d3ce519` and `fdab7d3` and is described below (commit log
`5562e65..fdab7d3`; each item read in the code):

* R_psi's lifetime context is no longer reset at a new objective (KF-E05): `set_objective` calls
  `MetaLearner.new_objective`, not `new_lifetime` (§6, §9, §10). The earlier version said the opposite.
* The live policy update consumes stored retrospective credit (KF-E01): the agent sets the inner loss's credit
  function (§6, §7). The earlier version omitted this.
* The agent's randomised experiments reach the experience store and the credit-estimator selection is run on them
  (DX-1, RC-08 / G2L-6) (§7). As configured it finds no held-out units, so the default stays `simple` (§11).
* Slow-step guard and adapter step (DX rounds 1-2, G2L-4/5/9), stratified record split (INT3A-5), provider
  eligibility without the learner's own flags (G2L-8) (§7).
* Checkpoints: authorization-configuration hash, the authorization in force wins on load (DX-2, G2L-10); optimizer
  state only in the newest kept checkpoint (§8).
* Fast layer: Poisson mode search fixed (G2L-1); prediction-error change detector re-arms (G2D-1) (§4, §5).
* Backbone prompt window `nn.max_prompt_tokens` 6144 (was the preset's 2048); the proposer prompt lists every
  permitted tool and is fitted to the window (§2).

Evidence of every count: `docs/evidence/DOC/param_counts.json` (re-measured at `fdab7d3`, §1.4).

## 0. Components at a glance

| component | code | group |
|---|---|---|
| frozen backbone Qwen/Qwen3-4B | `nn/backbone.py:SemanticBackbone` (built by `nn/backbone.py:build_backbone`) | `backbone_frozen` |
| LoRA adapters (q/v, all 36 layers, r 32) | `nn/backbone.py:LoRALinear`, installed in `nn/backbone.py:_Backbone._finish_init` | `semantic_adapters` |
| latent bridge (belief → 8 virtual tokens read by cross-attention in 6 decoder layers) | `nn/backbone.py:LatentBridge`, hooks `nn/backbone.py:_Backbone._layer_hook` | `semantic_adapters` |
| experience tokenizer | `nn/tokens.py:ExperienceTokenizer` (records → items: `nn/tokens.py:to_items`) | `experiential` |
| experiential core (50m preset) | `nn/core.py:ExperientialCore` | `experiential` |
| action / goal / query encoders | `nn/decision.py:ActionEncoder`, `nn/decision.py:GoalEncoder`, `nn/decision.py:QueryEncoder` | `experiential` |
| decision trunk z and feature map phi | `nn/decision.py:DecisionTrunk`, `nn/decision.py:FeatureMap` | `experiential` |
| learned prior of the fast layer | `nn/fast.py:FastPrior` | `experiential` |
| fast Bayesian beliefs (state, not parameters) | `nn/fast.py:FastBeliefState` (created in `credit/learner.py:ContinualLearner.__init__`, reset by `credit/learner.py:ContinualLearner.reset_fast`) | — |
| assembled model | `nn/model.py:CamLiveModel` (`nn/model.py:build_model`), parameter groups `nn/api.py:GROUPS` | — |
| world model, latent transition, goal evaluator, candidate policy, downstream-VOI | `predict/heads.py:attach_heads` → `predict/world.py:WorldHeads`, `predict/world.py:LatentTransition`, `predict/evaluator.py:GoalEvaluator`, `predict/policy.py:CandidatePolicy`, `predict/voi.py:DownstreamValueEstimator` | `heads` |
| prediction-error attribution (per metric) | `predict/errors.py:ChangeDetector`, `predict/errors.py:classify_error` (used in `predict/predictor.py:Predictor.observe_outcomes`) | — |
| RUDDER return decomposer, COCOA hindsight model | `credit/rudder.py:ReturnDecomposer` (via `credit/providers.py:ReturnDecompositionProvider`), `credit/cocoa.py:HindsightModel` (via `credit/providers.py:HindsightProvider`) | `heads` |
| credit assignment and estimator selection | `credit/providers.py:CreditAssigner`, `credit/select.py:EstimatorSelector` | — |
| intrinsic reward R_psi with lifetime context | `reward/intrinsic.py:IntrinsicReward`, `reward/intrinsic.py:LifetimeContext` (registered by `reward/metalearner.py:MetaLearner.__init__`) | `reward` |
| meta-learner (inner policy update, outer update of psi) | `reward/metalearner.py:MetaLearner`, `reward/inner.py:PolicyImprovementLoss`, `reward/meta.py:MetaGradient` | owns the `reward` optimizer |
| continual learner (fast / frequent / slow) | `credit/learner.py:ContinualLearner` | owns the `experiential`, `heads`, `semantic_adapters` optimizers |
| checkpoints, promotion, rollback, export | `credit/checkpoint.py:CheckpointManager`, `credit/checkpoint.py:promote_learner`, `credit/checkpoint.py:export_weights_only`, `credit/checkpoint.py:load_exported_model` | — |
| composition | `agent.py:CamLiveAgent.__init__`, factory `agent.py:build_agent`, heads for an export `agent.py:register_agent_heads` | — |

## 1. Parameter counts (measured)

### 1.1 Default configuration (`configs/cam_live.yaml`: Qwen3-4B + LoRA + bridge + 50m core), integrated agent

Measured by constructing `CamLiveAgent(cfg, run_dir, model=model)` (as `agent.py:build_agent` does, so every head the
agent registers exists) and calling `nn/model.py:CamLiveModel.count_parameters`. Re-measured at `fdab7d3`: every
sub-count is identical to the measurement at `d3ce519` (§1.4).

| group (`nn/api.py:GROUPS`) | parameters | of which |
|---|---:|---|
| `backbone_frozen` | **4,022,468,096** | Qwen3-4B weights, `requires_grad=False` (0 trainable) |
| `semantic_adapters` | **16,713,856** | LoRA 11,796,480 (72 modules = 36 layers × {q_proj, v_proj}, r 32: 36 × 32 × [(2560+4096)+(2560+1024)]) · latent bridge 4,917,376 |
| `experiential` | **50,953,765** | tokenizer 4,263,746 · core 36,615,648 · encoders 1,977,600 · decision trunk 4,720,128 · feature map 3,343,104 · fast prior 33,539 |
| `reward` | **949,121** | `reward` = `IntrinsicReward` (d_hidden 256, d_life 64, incl. the GRU of the lifetime context) |
| `heads` | **6,055,337** | `predict_world` 1,022,954 · `predict_transition` 1,977,600 · `predict_evaluator` 783,192 · `predict_policy` 702,339 · `predict_downstream` 84,195 · `credit_rudder` 559,361 · `credit_cocoa` 925,696 |
| **total persistent trainable** | **74,672,079** | experiential + adapters + reward + heads; every one has `requires_grad=True` |

* The model alone (before the agent registers heads) has 67,667,621 trainable parameters, identical to
  workstream A's GPU measurement in `docs/evidence/A/params.json` (which counted no heads; not re-checked). The
  substantive evaluation records the same group counts for its runs on CUDA (`docs/S/RESULTS.md` §1; not re-checked).
* The adaptable subset theta of the inner (meta-differentiated) policy update = `predict_policy` + decision trunk =
  702,339 + 4,720,128 = **5,422,467** (`reward/inner.py:AdaptableSubset`).
* Not parameters, not in the counts: the frozen randomised prior networks of the world-model ensemble are buffers,
  530,410 numbers (`predict/world.py:EnsembleLinear` with `trainable=False`; re-measured with `extra_counts.py`,
  identical); running normalisation statistics of R_psi and of the world model (`predict/world.py:KindNormaliser`);
  calibration statistics of the backbone (`cal_mu`, `cal_sd`, 2 × 7680) and the bridge (`h_mu`, `h_sd`).
* Fast state: per outcome kind a mean of F = 128 and a full F × F covariance, i.e. 16,512 numbers per kind
  (`count_parameters()["fast_state_numbers_per_kind"]`), 4 kinds (+ the normal-inverse-gamma a, b of the real kind).

### 1.2 Debug configuration (`CamLiveConfig.debug()`: tiny random LM + debug core; CPU tests only)

frozen 131,392 · adapters 42,304 (LoRA 1,024, bridge 41,280) · experiential 202,421 · reward 227,521 · heads 709,269
(world 47,996, transition 107,424, evaluator 27,218, policy 29,315, downstream 3,491, rudder 289,025,
cocoa 204,800) · total trainable 1,181,515. Re-measured at `fdab7d3`, identical. The tiny LM is never reported as the
backbone.

### 1.3 Other scale presets (experiential group only)

Measured by `python -m cam_live.nn.bench` (workstream A, `docs/evidence/A/params.json`; not re-checked; `nn/` is
unchanged since): 5m 5,044,741 · 20m 20,087,333 · 50m 50,953,765 · 150m 151,022,853 · 500m 502,708,549 · 1b
1,000,547,333. Only `debug` and `50m` have been run as part of the agent; 150m–1b were instantiated and counted on CPU
only.

### 1.4 How the counts were produced (re-measured at `fdab7d3`)

```
cd <scratch>/DOC/r3_ma && CUDA_VISIBLE_DEVICES= CAM_LIVE_HOME=<scratch>/DOC/r3_ma/home HF_HUB_OFFLINE=1 \
    HF_HOME=/workspace/.hf_home /workspace/camlive/.venv/bin/python \
    /workspace/camlive/cam_live/docs/evidence/DOC/scripts/count_params.py <scratch>/DOC/r3_ma/pc.json
cd <scratch>/DOC/r3_ma && CUDA_VISIBLE_DEVICES= CAM_LIVE_HOME=<scratch>/DOC/r3_ma/home \
    /workspace/camlive/.venv/bin/python /workspace/camlive/cam_live/docs/evidence/DOC/scripts/extra_counts.py
```

(`<scratch>` = `/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad`.) Run 2026-11-28
07:16:59–07:18:24 UTC on CPU, no GPU; the script records commit `fdab7d3033e98082bd213015063cfca6c35d8a12`. The
script only calls `CamLiveConfig.load()/debug()`, `nn.model.build_model`, `agent.CamLiveAgent(..., model=model)`,
`CamLiveModel.count_parameters()` and `param_groups()`. On CPU the backbone loads in float32
(`nn/backbone.py:SemanticBackbone`); counts do not depend on dtype. Compared key by key with the committed
`param_counts.json` (measured at `d3ce519`): the counts (`count_parameters`, `requires_grad_per_group`, `heads`,
`counts_model_only_before_agent`) and the frozen-prior buffer counts are identical; the only recorded value that
differs is `backbone_info.max_prompt_tokens`, 2048 → 6144 (configuration change `a7b8fa4`, §2), besides build time.
`param_counts.json` now records `remeasured_at_commit` and that field.

## 2. Backbone and semantic adapters

**Backbone** (`nn/backbone.py:SemanticBackbone`). `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
revision="1cfa9a7208912126459214e8b04321603b3df60c", local_files_only=True)` from `HF_HOME` (default
`/workspace/.hf_home`), offline. dtype: `bfloat16` on CUDA, float32 on CPU. 36 layers, hidden 2560, vocabulary
151,936. Every LM parameter gets `requires_grad=False` and the LM stays in eval mode (`nn/backbone.py:_Backbone._finish_init`,
`nn/backbone.py:_Backbone.train`). The final RMSNorm and the LM head are recomputed in float32
(`nn/backbone.py:SemanticBackbone._norm32`, `nn/backbone.py:_HeadLogits`, `nn/backbone.py:_TokenLogprob`; checked
against the module by `nn/backbone.py:SemanticBackbone._check_final_norm`). `name="tiny"` builds
`nn/backbone.py:TinyBackbone` instead.

**How embeddings are used** (`nn/backbone.py:_Backbone.embed`, `nn/backbone.py:_Backbone._pooled`,
`nn/backbone.py:_Backbone._cached`, `nn/backbone.py:_Backbone._standardise`). A text is tokenised (at most
`max_text_tokens` = 384: head 3/4 + tail 1/4, truncation counted in `stats["truncated_tokens"]`), the hidden states of
the layers at fractional depths (0.5, 0.75, 1.0) — layer ids 17, 26, 35 — are mask-mean-pooled and concatenated
(d_sem = 3 × 2560 = 7680), then standardised per dimension with statistics of a fixed neutral calibration corpus
(`nn/backbone.py:CALIBRATION_TEXTS`, winsorised at ±20). Fixed forward shapes per length bucket make a text's embedding
independent of the other texts of a call. Results are cached by text hash (float16). `with_grad=True` returns the same
(cached) value plus a graph into the LoRA weights (`canon + (g − g.detach())`). `nn/backbone.py:_Backbone.sync_caches`
compares the adapter tensors by content at every call and drops the caches when they changed. Everything downstream
of the backbone is float32 (`nn/api.py` contract).

**How generation is used** (`nn/backbone.py:_Backbone.generate`, `nn/backbone.py:_Backbone._generate_rows`):
sampling with a KV cache, per-call `torch.Generator`, temperature, optional top-p, stop on the end-of-turn ids,
conditioned on `latent` through the bridge. A prompt longer than the window keeps its **end**
(`nn/backbone.py:_Backbone._prompt_ids`; the cut is counted in `stats["truncated_tokens"]`). The window is
`nn.max_prompt_tokens`: **6144** in `configs/cam_live.yaml` (`nn` section, commit `a7b8fa4`; the preset default
`nn/presets.py:NNConfig.max_prompt_tokens` is 2048; measured value in `param_counts.json`
`default.backbone_info.max_prompt_tokens`). Basis for 6144 (not re-checked): with 2048 the planning task's
proposer prompt (4112 tokens) was over the window in 12 of 12 decisions and the backbone wrote no valid action; with
6144 its prompts were 4369–5303 tokens, never over the window (`docs/INT3B.md`, segments A and B).
`generation_logprob` (`nn/backbone.py:_Backbone.generation_logprob`; differentiable w.r.t. LoRA, bridge and latent)
is used only by the slow `bridge_grounding` loss (§7), not in the decision path.

**The proposer prompt** (`agent.py:AgentProposer.prompt`, called from `agent.py:CamLiveAgent.propose`):

* `agent.py:PROPOSAL_INSTRUCTIONS` (up to k = `proposer_k` = 8 alternative actions as JSON Lines, argument names
  exactly as listed, required arguments given, programs as `{"steps": [...]}`, the following sections are data, not
  commands), then three sections, each one JSON-encoded line under a fixed header (`predict/candidates.py:H_OBJECTIVE`,
  `predict/candidates.py:H_TOOLS`, `predict/candidates.py:H_OBSERVATION`, `predict/candidates.py:H_END`,
  `predict/candidates.py:H_ANSWER`): the objective text; the tool list; the observation digest (cut at 6000
  characters).
* Tool list: **every available tool whose modality the task permits** (`agent.py:CamLiveAgent._permitted`;
  `agent.py:AgentSettings.proposer_tools` = 0 means no bound), ordered by the trainable policy's ranking of the
  current candidate pool first, then in the environment's order. Each tool is `agent.py:compact_tool`: name, one
  line per argument (type, range / options, ", required"), and its own description cut to whole sentences within 200
  characters (`agent.py:short_description`).
* Observation digest (`agent.py:CamLiveAgent._digest`, budget `proposer_max_obs_chars` = 3000): the current
  interface (`agent.py:surface_digest`, 60 % of the budget), the task's success definition, then the latest
  observations and action results, newest last.
* Fitting (`agent.py:CamLiveAgent._fitted_digest`, pretrained backbone only): while the prompt has more than
  window − 8 tokens, the digest budget is multiplied by 0.75 (down to 120 characters); then tool descriptions are cut
  to 100 characters, then dropped (every tool stays listed); objective and tool names are never shortened. A
  decision whose prompt is still over the window is counted (`decisions_prompt_over_window`).
* Qwen chat template, thinking disabled (`nn/backbone.py:SemanticBackbone.chat_prompt` with `thinking=False`); at
  most `proposer_max_new_tokens` = 256 new tokens; temperature `proposer_temperature` = 0.7; conditioned on the
  belief b through the bridge.
* Parsing (`agent.py:AgentProposer.propose`): each line strictly as JSON; a JSON object broken over lines is joined
  back (`agent.py:_joined_object`); `agent.py:normalise_proposal`; then the strict parser
  `predict/candidates.py:parse_actions`. Each decision records the listed tools, the prompt's sha256, its token count,
  the fitting and the rejection reasons (`decision.diagnostics["generation"]`, written in `agent.py:CamLiveAgent.select`).

**LoRA** (`nn/backbone.py:LoRALinear`): `base(x) + (alpha/r)·B(A(x))` on `q_proj` and `v_proj` of all 36 attention
blocks, r = 32, alpha = 32; A uniform(±1/√in), B zeros (exact no-op at init); adapter weights float32, cast in the
forward; an all-zero B is skipped at inference.

**Latent bridge** (`nn/backbone.py:LatentBridge`): `memory(b)` = RMSNorm(Linear(768 → 8 × 128)(RMSNorm(b)) + pos):
8 virtual tokens of width 128. Six decoder layers (ids 0, 7, 14, 21, 28, 35) get a forward hook that adds
`h_scale · CrossAttention(standardise(h), memory)` (4 heads) to the residual stream. The cross-attention output is
zero-initialised: the bridge is an exact no-op at initialisation (bit-identical logits, `docs/evidence/A/params.json`
`bridge_noop_check`; not re-checked) yet receives gradient. The virtual tokens are not placed in the token sequence.
Used whenever `generate` / `generation_logprob` gets `latent=b`.

## 3. Experience encoding and the experiential core

`nn/` (backbone, tokens, core, decision, model) is unchanged between `d3ce519` and `fdab7d3`
(`git diff --stat d3ce519 fdab7d3 -- cam_live/src/cam_live/nn` lists only `fast.py` and `gpu.py`).

**Tokens** (`nn/tokens.py:ExperienceTokenizer.encode`): every contract record becomes `ExperienceItem`s
(`nn/tokens.py:to_items`, `nn/tokens.py:time_item`, `nn/tokens.py:resource_items`; the agent's own items:
`agent.py:observation_item`, `agent.py:action_item`, `agent.py:outcome_item`, `agent.py:tools_item`). One token per
item is the sum of (i) content: the per-tap RMS-normalised semantic embedding through a low-rank projection
7680 → 384 → 768 (`sem_rank` 384 for 50m, `nn/presets.py:NN_PRESETS`), RMS-normalised; a learned vector for items
without text; (ii) a separate value channel for up to `max_values` = 4 exact quantities: odd features (sign, tanh of
asinh at four scales, sin of asinh·freq) and even features, bound multiplicatively to a hashed per-metric-name
embedding (two 2048-bucket tables) and a unit embedding, then normalised on its own
(`nn/tokens.py:ExperienceTokenizer.value_channel`); raw float64 values ride along in `TokenBatch.values`; (iii)
item-kind and trust embeddings; (iv) features of the gap to the previous token.

**Core** (`nn/core.py:ExperientialCore.forward`): d_model 768, 6 pre-norm blocks, 12 heads, SwiGLU d_ff 1024
(`nn/core.py:Block`); causal self-attention with rotary positions over the event index plus a learned per-head bias of
the elapsed time `t_i − t_j` (`nn/core.py:elapsed_features`: 16 radial + 12 Fourier features of log1p seconds); every
second block (blocks 0, 2, 4) cross-attends to 8 persistent slots conditioned on the time since the previous call;
after the blocks a gated slot write (`gate_cand` bias −2 at init); the belief is
`b = RMSNorm(h_last + direct(RMSNorm(x_last)) + state_read(mean(slots)))` — the direct path from the newest event.
`CoreOutput.state` holds the updated slots, but **no caller carries it**. At `fdab7d3` every call of
`CamLiveModel.core_forward` outside the benchmark passes `None` (grep of `core_forward(`): `agent.py:CamLiveAgent._belief`,
`predict/predictor.py:Predictor.evaluate_candidates`, `predict/predictor.py:Predictor._slow_views`,
`predict/predictor.py:Predictor.belief_value`, `predict/planner.py:LatentPlanner.plan`,
`predict/providers.py:encode_with_adapters`, `predict/providers.py:_next_belief`, `learning_api.py:encode_decisions`,
`credit/learner.py:encode_with_adapter_grad`, `reward/inner.py:PolicyImprovementLoss._encode_next`, and the two
diagnostics modules. Each belief is therefore computed from the learned initial slots and the re-encoded context
window (§11, item 1). The context window (`agent.py:CamLiveAgent._window`) is the goal item, the tool-list item and
the newest items of the current objective, at most `max_context` = 256 tokens; it starts empty at every new
objective (`agent.py:CamLiveAgent.set_objective`).

**Encoders, trunk, features** (`nn/decision.py`): `ActionEncoder` (rendered action text + cost quantities through
the tokenizer, argument values through the value channel (`nn/model.py:CamLiveModel.encode_actions`), modality /
level embeddings, timeout and step count); `GoalEncoder`; `QueryEncoder` (metric text, outcome kind, horizon,
exposure, goal-relevance, hashed metric name shared with the value channel). `DecisionTrunk(b, e_a, e_g) → z`
[B, N, 768]: each input RMS-normalised on its own, additive + multiplicative action × situation interaction, SwiGLU
residual. `FeatureMap(z, e_q) → phi` [B, N, Q, 128] with ‖phi‖₂ = 1 and a constant last coordinate 1/√128.

## 4. Fast Bayesian beliefs (state m_t, no gradient)

`nn/fast.py:FastBeliefState`: for each outcome kind (real, count, binary, time-to-event; categorical is not modelled)
one Bayesian last layer `w ~ N(mu, Sigma)` over phi, float64. Real: normal-inverse-gamma (noise inferred, Student-t
predictive; `NNConfig.fast_noise_mode="nig"`, `nn/fast.py:nig_update`). Count (Poisson with exposure, tempered by a
learned dispersion), binary (binomial) and time-to-event (exponential, right-censoring via the Poisson-exposure
identity): Laplace updates at the mode, applied in Joseph form (`nn/fast.py:glm_update`, `nn/fast.py:gaussian_update`).
The count / time-to-event mode is found by `nn/fast.py:_mode_poisson`: on either side of the root the smaller of the
Newton steps on r(x) and on the log form G(x) is taken (G2L-1; before, the G-step was used only from the right), and
convergence is reported only where the root condition holds at the returned point (|r| ≤ 1e-8 · scale + float64
rounding, or |G| ≤ 1e-8), not only the relative step rule. Stabilised forgetting towards the prior
(`nn/fast.py:forget`, `fast_forgetting` 0.999 per update) and a CUSUM change detector on the squared standardised
surprise (`c ← max(0, c + min(z², 9) − 1 − 1)`, change at c > h = 12; on a change the posterior is flattened towards
the prior keeping 10 % of the evidence, `nn/fast.py:FastBeliefState._flatten`; all in `nn/fast.py:FastBeliefState.update`).
The prior mean and Cholesky-parameterised covariance per kind, the noise, its inverse-gamma shape and the dispersion
are the learned `nn/fast.py:FastPrior` (experiential group), trained ALPaCA-style through `nn/fast.py:online_nll` by
the `fast_prior` loss provider (`predict/providers.py:FastPriorLoss`). One posterior per kind is shared by all metrics
of that kind; it is reset to the learned prior at every new objective (`agent.py:CamLiveAgent.set_objective` →
`credit/learner.py:ContinualLearner.reset_fast`). The mathematics is in `docs/MATH_NOTES.md`.

## 5. Heads on the shared representation

`predict/` heads (world, evaluator, policy, VOI, planner) are unchanged since `d3ce519`. Built by
`predict/heads.py:attach_heads` from `PredictConfig` (hidden 192, seed-forked RNG) and registered in group `heads`:

* **World model** `predict/world.py:WorldHeads`: `p(y | z, q, horizon)` for real (Student-t / normal), count (negative
  binomial with exposure), binary (Bernoulli), time-to-event (12-bin discrete hazard, right-censored) and categorical
  (≤ 16) outcomes, the query and the horizon as inputs (no fixed target list). A bootstrapped ensemble of 5 members,
  each with its own frozen randomised prior network; inputs z, e_q, horizon features, kind, phi. Combined with the
  fast layer in `predict/predictor.py:Predictor._means` / `predict/predictor.py:Predictor._samples` as stated in
  `predict/heads.py:fast_inputs` (count: fast layer models log(actual / head rate); real: the normalised residual;
  time-to-event: log hazard ratio; binary: posterior shift of the fast logit added to the head logit). Epistemic =
  ensemble spread + fast-posterior variance, aleatoric = members' mean predictive variance, reported separately.
* **Latent transition** `predict/world.py:LatentTransition`: 3-member ensemble predicting the next belief `b'` from
  `(z, dt, b)` with diagonal variance; used only by the latent planner `predict/planner.py:LatentPlanner`
  (imagination; outputs flagged imagined).
* **Goal evaluator** `predict/evaluator.py:GoalEvaluator`: `forward(b, e_g, horizon)` (V) and
  `predict/evaluator.py:GoalEvaluator.forward_candidates` (Q): success logit, magnitude (negative binomial /
  Student-t), time to achievement (12-bin hazard, censored), 3 resource heads. Supervised only by signature-checked
  verified outcomes (`predict/providers.py:EvaluatorLoss`).
* **Candidate policy** `predict/policy.py:CandidatePolicy`: `forward(z, a_mask, b)` gives logits over a variable
  candidate set, with a pooled summary of the alternatives and a multiplicative z × b path (so the distribution
  depends on the belief), plus V and Q read-outs. `predict/policy.py:CandidatePolicy.functional_forward` evaluates it
  with explicit parameters (`torch.func`).
* **Downstream-VOI estimator** `predict/voi.py:DownstreamValueEstimator`: Student-t prediction of realised downstream
  usefulness from z and five signals (`predict/heads.py:VOI_SIGNALS`); scored against realised values by
  `predict/voi.py:score_downstream`.

**Prediction-error attribution** (`predict/predictor.py:Predictor.observe_outcomes`): every absorbed matured outcome
updates a per-metric `predict/errors.py:ChangeDetector` (two-sided CUSUM of the standardised residual, drift 0.5,
threshold 8) and is labelled by `predict/errors.py:classify_error` (`missing_observation`, `within_noise`,
`model_uncertain`, `changed_conditions`, `model_wrong`). Since G2D-1 the detector re-arms once both statistics fall
below `rearm` · threshold (0.25 · 8), so a later confident miss can be labelled `model_wrong` again (`predict/errors.py:ChangeDetector.update`). The label is recorded with the outcome; it
creates no training signal and does not change the fast update.

**Selection** (`predict/policy.py:DecisionMaker.select`, rules `predict/policy.py:RULES`): policy, thompson, ucb,
greedy, voi, thompson_policy, randomized; every rule builds the full probability vector first and samples from it,
with an epsilon floor (0.02), so recorded propensities are exact. Agent default `thompson_policy`
(`agent.py:AgentSettings.selection_rule`): Thompson samples of each candidate's goal value (standardised within the
decision) plus the policy's centred log-probability beyond the world model's ranking, weighted by λ = β_π⁺/β_q drawn
from `predict/policy.py:SelectionCalibration` (Bayesian linear regression of verified progress on those two numbers,
forgetting 0.99; state kept on the policy head). With `experiment_rate` 0.1 the agent instead registers a randomised
comparison (`predict/policy.py:DecisionMaker.propose_experiment` / `predict/policy.py:DecisionMaker.select_experiment`,
via `agent.py:CamLiveAgent._experiment`) when the expected flip probability of two candidates' order is ≥ 0.2. When the
task has no objective text or no success definition (goal-free mode), the rule is `goal_free_rule` = voi.

**VOI** (`predict/predictor.py:Predictor._voi`, `predict/voi.py`): `E_y[max_a' Q(b_updated(a, y), a')] − max_a' Q(b, a')`
with `y` sampled from the fast posterior and the fast update applied (8 samples, ≤ 16 candidates); it reuses the
evaluation's predictive means and runs its sample loop on the fast state's device. The slow path that re-encodes an
imagined result into the context is off (`PredictConfig.voi_slow_path=False`).

## 6. Intrinsic reward and meta-learning

`reward/intrinsic.py:IntrinsicReward` (`forward(b_t, e_a, b_next, e_g, signals, life_ctx)`): running-normalised
inputs (statistics frozen inside a meta window), `latent_in(4·768 → 256)` over [b, e_a, b_next − b, e_g], `signal_in`
over the normalised signal vector and its mask (`reward/signals.py:SIGNAL_NAMES`), plus `life_in(LifetimeContext.state)`
(a GRUCell 256 → 64), SiLU MLP, output bounded to (−1, 1) (`reward/integrity.py:bound_reward`). `version` is a hash of
parameters and statistics.

**Lifetime context** (`reward/intrinsic.py:LifetimeContext`): advanced once per rewarded decision in
`reward/inner.py:PolicyImprovementLoss.reward_step` (the state each decision saw is remembered for the inner loss).
It is **kept across objectives**: `agent.py:CamLiveAgent.set_objective` calls
`reward/metalearner.py:MetaLearner.new_objective`, which only starts a new segment of the meta-learner's bookkeeping
(returns, the return-to-go chain and the discount do not run across objectives); the reset
`reward/metalearner.py:MetaLearner.new_lifetime` is called only by `reward/diagnose.py`, never by the agent (KF-E05).
The lifetime context is saved in every checkpoint (`lifetime_context`, §8).

`reward/metalearner.py:MetaLearner` owns the only optimizer of the reward group (Adam, lr 1e-4, clip 0.5).
* `reward/metalearner.py:MetaLearner.internal_rewards` (runtime step 13) assembles the signals
  (`reward/assemble.py:SignalAssembler`), the labelled bootstrap reward (`reward/bootstrap.py:BootstrapReward`) and
  R_psi for each new decision. Two bootstrap inputs are written by the agent as annotations on the stored decision
  (LRN-3): `epistemic_after` (`agent.py:CamLiveAgent._annotate_epistemic_after`, after the fast layer absorbed the
  decision's goal-relevant outcomes) and `learning_progress` (`agent.py:CamLiveAgent._lp_after`: world-model loss on
  the same targets before minus after a frequent update).
* `reward/metalearner.py:MetaLearner.policy_update` (every `policy_update_every` = 4 rewarded decisions, on the newest
  16 rewarded decisions plus up to 16 earlier ones with final verified outcomes, from `agent.py:CamLiveAgent.learn`)
  applies `theta ← theta − alpha ∇ L_inner(theta; psi, D_support)` to the live adaptable subset (policy head +
  decision trunk) with `reward/inner.py:PolicyImprovementLoss` (K = 2 steps). The step size is chosen by a trust
  region: mean KL between the policy before and after ≈ `target_kl` 0.05 (accepted within a factor 1.5), alpha in
  [0.01, 2.0], up to 6 step-size adjustments. It records an update event (snapshot of theta, support batch, chosen
  step). The live reward is the mix `bootstrap_mix · n(r_bootstrap) + (1 − mix) · n(R_psi)` plus the goal term; the
  mix starts at 1.0 (`MetaConfig.mix`).
* **The goal term uses stored credit** (KF-E01). The agent sets the inner loss's credit function to
  `agent.py:CamLiveAgent._credit_for` (in `agent.py:CamLiveAgent.__init__`); the inner loss reads it in
  `reward/inner.py:PolicyImprovementLoss._credit`. Per row, in this order: a final verified per-decision annotation;
  else, if the evidence-selected estimator (`credit/providers.py:CreditAssigner.default`) is `simple`, the decision's
  own linked outcome amount as a per-decision goal reward — only for objectives whose environment attributes
  outcomes to decisions (`CamLiveAgent.attributing`); else the calibrated predicted goal value. If the selected
  estimator is RUDDER or COCOA, its stored credit is *return-like* and replaces the row's discounted goal return
  (`reward/inner.py:PolicyImprovementLoss._credit_returns`, `reward/inner.py:PolicyImprovementLoss.credited_goal_return`:
  G_i = c_g · C_i / rms(C) at a credited row, g_i + γ_i G_{i+1} elsewhere). Rows are labelled `verified`,
  `credit:<estimator>`, `predicted` or unknown.
* `reward/metalearner.py:MetaLearner.meta_update` (step 15) needs ≥ `meta_min_independent_outcomes` = 8 independent
  verified groups; it recomputes theta_K(psi) functionally on the cached latents (`reward/meta.py:MetaGradient`,
  `torch.func`), scores the later query decisions under pi_{theta_K} (on-policy only if the logged policy is the
  recomputed one, else clipped importance weights with ESS ≥ 2), back-propagates to psi and steps the reward
  optimizer, or returns a skip record with the reason and leaves psi bit-identical. The differentiated update uses
  `meta_mix` = 0 (pure learned reward).
* Utilities come only from `reward/integrity.py:OutcomeLedger` (signed objective outcomes, authenticated verifier
  measurements). The mixer moves the bootstrap mix by 0.25 per non-inferiority decision on held-out groups
  (`reward/bootstrap.py:RewardMixer`).

## 7. Credit and continual learning

**Store and linking**: `credit/replay.py:SqliteExperienceStore` (`<run_dir>/learner/experience.sqlite`, WAL;
`contracts.py:RunFiles.STORE`); it refuses matured outcomes dated before their query or decision
(`credit/replay.py:OutcomeTimingError`, G2L-2). `credit/links.py:OutcomeLinker` (attribution, de-duplication by
`dedup_key`, re-training queue; overflowing attribution amounts become shares of the largest,
`credit/links.py:attribution_amounts`, G2L-3).

**Credit** (`credit/providers.py:CreditAssigner.assign`, called by `credit/learner.py:ContinualLearner.learn` for the
records of the re-training queue) writes per-decision credit of three estimators into `ExperienceRecord.credit`:
`simple` (`credit/simple.py:credit_from_links`), RUDDER (`credit/rudder.py:ReturnDecomposer`: 2-layer causal
transformer, width 128, over [z, b] of a closed objective's decisions; redistributed return-to-go) and COCOA
(`credit/cocoa.py:HindsightModel`: 4-member ensemble, policy-logit prior, elapsed-time input; refuses unsupported
cases). Which estimator's credit is used for learning is `credit/providers.py:CreditAssigner.default`, chosen by
`credit/select.py:EstimatorSelector.select`: `simple` until a learned estimator's estimates of arm differences agree
with the measured differences of ≥ 5 scored randomised experiments, held out by lifetime. The stored credit is
consumed by the policy update's goal term (§6).

**The agent's experiments in the credit store** (DX-1, RC-08): `agent.py:CamLiveAgent.__init__` calls
`credit/providers.py:CreditAssigner.attach_experiments` with the agent's `ExperimentRegistry`. Every
`credit/learner.py:ContinualLearner.learn` then copies the experiment records into the store before the held-out split
is refreshed (`credit/providers.py:CreditAssigner.record_experiments`) and afterwards runs
`credit/providers.py:CreditAssigner.evaluate_experiments`: every randomised experiment whose arms have ≥ 2 matured
units in lifetimes that no estimator and no learner batch trained on is compared
(`credit/providers.py:CreditAssigner.compare_with_experiment`) and the selection is re-run when that held-out evidence
changed; otherwise the reason is logged (`credit_selection.jsonl`, `ContinualLearner.status()["credit_selection"]`).
In the agent as configured the evaluation runs but finds no held-out units — each experiment has one unit and the
agent trains on the objective it experiments in (`holdout_experiments=False`, below) — so the default stays `simple`
(`docs/DX.md`, "RC-08 in the integrated agent"; not re-checked).

**Learner** `credit/learner.py:ContinualLearner`: one AdamW per learner group (experiential lr 3e-4, heads 3e-4,
adapters 5e-5; weight decay 0.01 on matrices; 8 warm-up steps; per-group norm clip 1.0 / 1.0 / 0.5,
`config.py:OptimConfig`), gradients by `torch.autograd.grad` per provider; reward parameters are refused
(`credit/learner.py:ContinualLearner._sync_optimizers`).

* Held-out split as the agent configures it (`agent.py:CamLiveAgent.__init__` overrides two
  `credit/learner.py:LearnerSettings` defaults unless the configuration sets them): `val_split="record"` — stratified
  within each objective: the k-th stored record of an objective is held out iff ⌊u + (k+1)f⌋ > ⌊u + kf⌋, f =
  `val_fraction` 0.15, u a seeded hash of the objective, so exactly every 1/f-th record is held out and an objective
  with ≥ 7 records always has one (`credit/learner.py:ContinualLearner._record_held_out`, INT3A-5;
  `credit/replay.py:SqliteExperienceStore.position_in_objective`); and `holdout_experiments=False` (experiment
  lifetimes are trained on). Per provider the store keeps a training flag and a `val:<provider>` flag
  (`credit/learner.py:ContinualLearner.refresh_eligibility`); a provider's own eligibility is judged without the flags
  the learner wrote (`credit/learner.py:ContinualLearner._provider_eligible`, G2L-8).
* fast: `predict/predictor.py:Predictor.observe_outcomes` → `nn/fast.py:FastBeliefState.update` (step 5,
  `agent.py:CamLiveAgent.update_beliefs`).
* frequent (`credit/learner.py:ContinualLearner.poll` → `credit/learner.py:ContinualLearner.learn` →
  `credit/learner.py:ContinualLearner._frequent_update`): triggered by ≥ 4 matured informative independent outcome
  units (one unit per top-level decision), up to 8 steps per trigger; batch 16 with a quarter reserved for the
  triggering experience. Providers registered by the agent: `world_model`, `fast_prior`, `evaluator`
  (`predict/providers.py:default_providers`), `rudder`, `cocoa`, `retention` (`credit/retention.py:RetentionProvider`:
  penalised change of b and z on anchors of earlier objectives). One provider's share of the experiential gradient
  norm is capped at 0.9.
* slow (`credit/learner.py:ContinualLearner._slow_update`, every `adapter_every` = 8 applied frequent updates): the
  semantic adapters only. Providers in the step: those that use the batch and are slow or reach the adapters
  (RUDDER, COCOA and retention declare `adapter_grad = False`): `world_model`, `fast_prior`, `evaluator`,
  `world_model_adapters`, `evaluator_adapters`, `bridge_grounding` (`credit/providers.py:BridgeGroundingProvider`:
  −log p_LM(observed consequence text | action, belief via bridge)). This list was printed at `fdab7d3` by the same
  predicate on the debug agent (CPU): `CUDA_VISIBLE_DEVICES= CAM_LIVE_HOME=<dir> .venv/bin/python -c "import
  tempfile,os; from cam_live.config import CamLiveConfig; from cam_live.agent import CamLiveAgent;
  a=CamLiveAgent(CamLiveConfig.debug(), tempfile.mkdtemp(dir=os.environ['CAM_LIVE_HOME'])); print([p.name for p in
  a.learner.providers if getattr(p,'uses_batch',True) and (p.timescale=='slow_adapters' or
  getattr(p,'adapter_grad',True))])"`.
  Batch of 4 records encoded with a graph into the adapters for the newest 8 context items
  (`credit/learner.py:encode_with_adapter_grad`). The step, in order:
  1. Held-out losses before the step on each provider's fixed validation sample (32 records, seeded), split into 8
     chunks (`credit/learner.py:ContinualLearner._validation_cells`). No provider with held-out records: the step is
     **deferred** (not taken).
  2. Losses and adapter gradients; every model buffer and every provider's target scaler are snapshotted before and
     restored after this evaluation, so the slow step changes the adapters only
     (`credit/learner.py:ContinualLearner._running_state`, `credit/learner.py:ContinualLearner._restore_running_state`;
     DX round 2). COCOA's held-out loss is computed without bootstrap weights (deterministic). A loss without held-out
     records of its own is excluded; if none is left the step is deferred.
  3. Combination: PCGrad between conflicting adapter gradients plus the 0.9 share cap
     (`credit/learner.py:ContinualLearner._balance_adapter_grads`). COCOA, which reads the representation but does not
     train the adapters, adds a sensitivity gradient on its own training records
     (`credit/providers.py:HindsightProvider.adapter_sensitivity_loss`, via
     `credit/learner.py:ContinualLearner._reader_sensitivity`).
  4. Clip 0.5, AdamW step with lr 5e-5 × warm-up × trust-region scale (`credit/learner.py:ContinualLearner._set_lr`);
     then the applied update is projected so that its first-order change of every protected provider's loss is
     ≤ 0 (`credit/learner.py:ContinualLearner._project_update`, up to 4 passes).
  5. Verdict per provider (`credit/learner.py:ContinualLearner._guard_verdict`,
     `credit/learner.py:ContinualLearner._provider_verdict`): mean paired change d over the chunks; tolerance =
     min(0.01, 0.05·|L|, max(2·SE(d), floor)), never below the floor min(floor, 0.05·|L|), floor = max(1e-4·|L|,
     1e-7·max(1, |L|)) (G2L-9); the provider's summed change over accepted steps since the reference (last
     promoted checkpoint) must stay ≤ max(0.02, 0.02·|L|); a loss that cannot be computed after the step rejects
     it. The step is kept only if every judged provider accepts.
  6. Rejected: undone exactly (adapter weights, AdamW moments, learning rates, step count) and the next adapter step
     is halved (down to 1/16); accepted: the scale grows by 1.5 (up to 1). The scale and the drift are checkpointed
     with the learner state and survive promotions (`credit/learner.py:ContinualLearner.set_slow_reference`).
* every update writes a `LearningUpdateRecord` (hashes before/after, per-group deltas, behavioural probe, losses,
  guard verdict).

The policy head has no loss among these providers; it changes only through `MetaLearner.policy_update`. The decision
trunk changes through both the learner (experiential group) and `MetaLearner.policy_update`.

## 8. Checkpoints, promotion, rollback, export

* Format (`credit/checkpoint.py:capture_state`, `credit/checkpoint.py:save_checkpoint_at`): atomic directories
  (temp → fsync → rename) with weights per group as safetensors, optimizer state by parameter name, fast state,
  lifetime context, RNG, `state.json`, a manifest with hashes and `nn/model.py:CamLiveModel.arch_signature`; also the
  store's experiment records (written back on restore if the store lacks them) and pending outcome queries (checked
  against the store and reported; `credit/checkpoint.py:_restore_store_records`). Tolerant loading of added heads
  (`credit/checkpoint.py:load_weights`, reported). The agent's checkpoint (`agent.py:CamLiveAgent.save`, called by the runtime's
  `runtime/loop.py:LiveRuntime._commit`) adds the meta-learner, predictor, goal-free and RNG state.
* **Authorization configuration** (DX-2, CL-10): `agent.py:CamLiveAgent.save` stores
  `credit/checkpoint.py:auth_config_hash` of the current task (sha256 over `credit/checkpoint.py:AUTH_FIELDS`:
  resources, interfaces, limits, data permissions, stop conditions, mode, origin). `agent.py:CamLiveAgent.load` passes
  the hash of the task in force to `credit/checkpoint.py:restore_learner`; when it differs, the agent counts
  `auth_config_changed_on_load`, records it in `restore_report` and **keeps the task in force** — the checkpoint's old
  task and limits are not restored (G2L-10). An agent restored before it has a task compares when the runtime sets one
  (`agent.py:CamLiveAgent.set_objective`).
* Promotion (`agent.py:CamLiveAgent.promote` → `credit/checkpoint.py:promote_learner` →
  `credit/checkpoint.py:CheckpointManager.promote`, every 24 frequent updates and at the end of an objective) is gated
  on held-out provider validation and retention metrics; without any held-out loss the refusal says so ("no held-out
  evidence", `PromotionDecision.held_out`). A promotion resets the slow-step drift reference. A refusal rolls back to
  the last promoted checkpoint except R_psi, its optimizer and (unless decision agreement on earlier objectives is a
  reason) the policy head.
* Retention on disk: the agent's `CheckpointManager` keeps the newest 2 and the newest promoted checkpoint; optimizer
  state is kept only in the newest kept checkpoint (`keep_optim` = 1, `credit/checkpoint.py:CheckpointManager.prune`
  → `credit/checkpoint.py:strip_optimizer_state`). A rollback to a checkpoint without optimizer state starts the
  learner's optimizers fresh (`optimizer_state_reset` in the report).
* Weights-only export: `agent.py:CamLiveAgent.export_weights` (refused below the storage minimum) →
  `credit/checkpoint.py:export_learner_weights` → `weights.safetensors` + `config.json`, rebuilt by
  `credit/checkpoint.py:load_exported_model` with `register_heads=agent.register_agent_heads`.

## 9. Data flow of one decision

`runtime/loop.py:LiveRuntime.step` calls the agent in the order of spec 15.1 (steps 6–12 in
`runtime/loop.py:LiveRuntime._decide_and_act`). Function names at `fdab7d3`.

| step | call | what happens (functions) |
|---|---|---|
| 1 objective | `CamLiveAgent.set_objective(learner_view(task))` (`runtime/tasks.py:learner_view`) | authorization-hash comparison after a restore; for a new objective: `_close_objective_bookkeeping` (retention anchors, open experiments concluded), promotion attempt for the old one, `MetaLearner.new_objective` (lifetime context kept), `ContinualLearner.reset_fast`, `Predictor.forget_goal_views`, ledger registration, empty context, goal item into the store |
| 2–3 observation | `CamLiveAgent.ingest(observations)` (`runtime/loop.py:LiveRuntime._ingest`) | `_note_obs` (items via `_put`, interactive surface), `_resolve_observed`, `OutcomeLinker.reconcile(events=...)` |
| 4 outcomes | `CamLiveAgent.reconcile(outcomes, verified)` (`runtime/loop.py:LiveRuntime._measure`, `runtime/loop.py:LiveRuntime._verify`) | `OutcomeLinker.reconcile` (dedup, links, re-training queue), `Predictor.resolve` (scores registered predictions), outcome items, matured outcomes buffered, `_experiment_outcome`, `_annotate_verified_delta`; `_verified` → signature check → `MetaLearner.admit_objective`, evaluator supervision, objective outcome stored and its trajectory queued for re-training |
| 5 belief | `CamLiveAgent.update_beliefs()` | `Predictor.observe_outcomes` → `FastBeliefState.update` (fast `LearningUpdateRecord`, error labels via `ChangeDetector`); `_annotate_epistemic_after`; `Predictor.belief_value` → b and the evaluator's success probability |
| 6 candidates | `CamLiveAgent.propose(tools, k)` | tools item; b = `_belief` (`core_forward(tokenize(window), None)`); `predict/candidates.py:schema_candidates`, `agent.py:grounded_candidates`, `_skill_candidates` → `_feasible` → `policy_distribution` ranks the pool → permitted tools ordered by that ranking → `_fitted_digest` → `AgentProposer.propose` (`model.generate(prompt, latent=b)` through the bridge, `normalise_proposal`, `parse_actions`) → `_feasible` → `_offer` (C_t ≤ 16, inclusion probabilities) |
| 7–8 prediction | `CamLiveAgent.evaluate(C_t)` | `Predictor.evaluate_candidates`: tokenize → `core_forward` → b; `encode_actions` → e_a; `encode_goals` → e_g; `decision_latent` → z; `GoalEvaluator`; `CandidatePolicy`; queries `_queries_for` → `world_batch` (`encode_queries`, `features` → phi, `WorldHeads`) → `_means` / `_samples` with the fast state → `_voi`; every prediction registered in the hash-chained `PredictionRegistry` (and written to disk by the runtime) before execution; `_keep_goal_view`; `LatentPlanner.plan` where allowed (imagined values recorded only) |
| 9 selection | `CamLiveAgent.select(evals)` | with probability 0.1 (≥ 2 candidates, expected flip probability ≥ 0.2): `DecisionMaker.propose_experiment` → `_experiment` → `DecisionMaker.select_experiment`; else `DecisionMaker.select(mode="thompson_policy")` → `Decision` with every candidate's probability, inclusion probabilities and generation diagnostics |
| 10–11 execution | runtime | `env.validate_action`, `PolicyEngine.authorize`, `runtime/loop.py:LiveRuntime._execute` (or the approval queue) — no model code |
| 12 outcome | `CamLiveAgent.record_result(decision, result)` (`runtime/loop.py:LiveRuntime._absorb`) | action item, `Predictor.adopt_queries` / `commit`, `SqliteExperienceStore.append(ExperienceRecord)`, `mark_for_retraining`, immediate outcomes resolved and buffered |
| 13 reward | `CamLiveAgent.internal_rewards()` | `MetaLearner.internal_rewards` → `PolicyImprovementLoss.reward_step` (signals, bootstrap reward, R_psi, lifetime context advanced) written to the records |
| 14 credit + update | `CamLiveAgent.learn(budget)` | `ContinualLearner.poll` → `_lp_before` → `ContinualLearner.learn` (`CreditAssigner.record_experiments`, `CreditAssigner.assign`, `refresh_eligibility`, frequent updates with a slow update every 8th, `CreditAssigner.evaluate_experiments`) → `_lp_after`; `MetaLearner.policy_update` (goal term with `_credit_for`) every 4 rewarded decisions; `_measure_planner`; `promote` every 24 frequent updates |
| 15 meta | `CamLiveAgent.meta_learn(budget)` | `MetaLearner.meta_update` (applied or skipped with reason) |
| 16 save | `CamLiveAgent.save(path)` (`runtime/loop.py:LiveRuntime._commit`) | storage guard, store flush, learner checkpoint with authorization hash |

## 10. What is shared, what is per objective

Kept across objectives: all trainable weights (backbone adapters, experiential core, heads, R_psi), optimizer states,
the learned fast prior, R_psi's lifetime context (KF-E05), the selection calibration, the slow-step trust-region
scale and drift, the experience store, registries. Reset at a new objective: the fast posteriors (to the learned
prior), the context window, pending goal views of the predictor, per-objective bookkeeping (the meta-learner's
return segment, `CamLiveAgent.lifetime` counter, active goals, objective operations). The experiential core carries
no recurrent state between calls (§3).

## 11. Not implemented, or different from the design (`CAM_LIVE_ARCHITECTURE.md` §2–5)

1. **Persistent core slots are not carried.** The design's `m_t` = core slots + fast state, with a gated slot update
   "carrying state between calls". The update is computed (`nn/core.py:ExperientialCore.forward` returns
   `CoreOutput.state`), but every caller passes `state=None` and discards the returned state (call sites in §3). In
   practice the state between decisions is the re-encoded context window (≤ 256 tokens of the current objective)
   plus `FastBeliefState`; across objectives only weights, the learned prior and R_psi's lifetime context carry
   anything.
2. **Latent bridge = cross-attention only.** Virtual prefix tokens are read by cross-attention in 6 layers; they are
   never inserted into the token sequence (change request A-2). Design allowed both; only the cross-attention form
   exists.
3. **Fast layer scope.** One posterior per outcome kind, shared by all metrics of that kind (not per metric or
   query); categorical outcomes are head-only; time-to-event is constant-hazard; binary outcomes are combined with the
   head by a posterior-shift approximation. In A's comparison the learned-prior layer is worse than a GRU baseline at
   support < 3 for Gaussian targets and only level with it for counts (`docs/status/A.json` scientific_failures; not
   re-checked).
4. **The live policy step is applied by the meta-learner, not by the continual learner**, with a trust-region step
   size and the live bootstrap mix (1.0 at start, i.e. bootstrap reward only), while the differentiated update uses
   the pure learned reward (`meta_mix` 0) from the same snapshot and is evaluated by importance weighting. The design
   says "the same update is what the continual learner actually applies to the live policy"; it is the same function
   (`reward/inner.py:PolicyImprovementLoss`), with a different mix and step size unless the mix reaches 0.
5. **Selection.** The default rule is `thompson_policy` (added by LRN; not in the design's list, which it combines).
   Imagined planner values are recorded in `CandidateEvaluation.extras` and used by no selection rule.
6. **Credit default stays `simple` in the agent.** Estimator selection is wired (DX-1) and runs after every learning
   update, but as configured it never has held-out units (one unit per experiment, experiment lifetimes trained on:
   `holdout_experiments=False`; `docs/DX.md`). So RUDDER's and COCOA's credit is computed, stored and trained, but does
   not enter the policy update; `simple` credit enters it only for objectives whose environment attributes outcomes
   to decisions (§6). The record-level split also lets RUDDER / COCOA train on whole trajectories that contain
   held-out records (`docs/DX.md`, "Still open (round 2)").
7. **VOI** is computed through the fast-layer update on phi; the slow path (imagined result entering the context)
   exists but is off by default.
8. `nn/model.py:CamLiveModel` does not override `nn/api.py:LatentModelAPI.decision_trunk` (it returns `None`);
   `reward/inner.py:AdaptableSubset` reaches the trunk through `model.exp.trunk` instead.
9. **Not run / not established**: Qwen3-8B (configuration change only, never loaded); presets 150m–1b never trained
   (`docs/status/A.json` not_done). Adapter learning on the real configuration now takes place — in the substantive
   evaluation's X0 (evaluation learner settings, not the defaults) 34 of 37 slow steps were kept, 2 reverted, 1
   deferred (`docs/S/RESULTS.md` §3; not re-checked) — but whether it, or the bridge, improves decisions is not
   established; spec 12A.10's comparison with the skill library is not shown (`docs/verification/G2_integration.md`
   R2.5, CU-22). `generation_logprob` on the bf16 trunk differs from float32 by up to 0.36 nats on a one-token
   completion; features phi are ill-conditioned at initialisation on real inputs (Gram eigenvalue ratio 2.3e4)
   (`docs/status/A.json`; not re-checked). The reward meta-update applied 4 of 123 times in X0 (`docs/S/RESULTS.md`
   §3); whether learning improves decisions is not established (`docs/INTEGRATION.md` §8).
10. Matches the design: frozen bf16 backbone with LoRA on q/v (11.8 M), bridge an exact no-op at init, multi-depth
    cached embedding invalidated on adapter change, separate value channel, causal core with time-aware attention and
    direct current-event path, 50m preset ≈ 50 M (50.95 M), shared trunk and feature map, exact / Laplace fast updates
    in float64 with forgetting and change detection, learned ALPaCA prior, per-group clipping, ensembles with
    separately reported epistemic / aleatoric uncertainty, three learning timescales with a held-out slow guard that
    fails closed, R_psi's lifetime context not reset at objective boundaries (since KF-E05; the version describing
    `d3ce519` listed this as a difference), stored retrospective credit used by the policy update (since KF-E01),
    atomic checkpoints with promotion, rollback, authorization hash and weights-only export.
