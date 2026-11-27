# Workstream A — neural architecture: what exists and why

Code: `src/cam_live/nn/` (everything except `api.py`). Tests: `tests/nn/`. Evidence: `docs/evidence/A/`.

| File | Contents |
|---|---|
| `backbone.py` | `SemanticBackbone` (HF causal LM, default Qwen/Qwen3-4B, frozen, bf16), `TinyBackbone` (`name="tiny"`, random LM for CPU tests), `LoRALinear`, `LatentBridge`, embedding cache, `generate`, `generation_logprob` |
| `tokens.py` | `ExperienceTokenizer`, `to_items(record, **kw)`, `relation_item`, `time_item`, `resource_items`, `value_features`, `gap_features`, `stable_hash` |
| `core.py` | `ExperientialCore`, `RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `CrossAttention`, `elapsed_features` |
| `decision.py` | `ActionEncoder`, `GoalEncoder`, `QueryEncoder`, `DecisionTrunk`, `FeatureMap` |
| `fast.py` | `FastPrior` (learned prior, `experiential` group), `FastBeliefState`, functional updates (`gaussian_update`, `nig_update`, `glm_update`, `gaussian_fit`, `nig_fit`, `forget`, `online_nll`, predictive / NLL helpers) |
| `model.py` | `CamLiveModel(LatentModelAPI)`, `build_model(cfg)`, `ExperientialNet`, `build_experiential` |
| `presets.py` | `NN_PRESETS` (measured), `NNConfig` (`cfg.section("nn")`), `resolve_core` |
| `gpu.py` | `gpu_lock()` for scripts (same `flock` as `tests/conftest.py`) |
| `bench.py` | `python -m cam_live.nn.bench` → `docs/evidence/A/params.json` |
| `fast_compare.py` | `python -m cam_live.nn.fast_compare` → `docs/evidence/A/fast_adaptation.json` |

## Design decisions tied to earlier failures

| Earlier failure | What is built | Test |
|---|---|---|
| Text norm swamped numbers (cos(+1, −1) = 0.98) | Separate value channel: sign + four asinh scales + Fourier terms of asinh(v); odd features bound multiplicatively to a hashed per-metric embedding; content and value each RMS-normalised to unit scale before the sum. Raw float64 values in `TokenBatch.values`. | `test_tokens.py::test_value_channel_separates_plus_one_from_minus_one`, `::test_exact_values_survive_in_token_batch` |
| Ungated state 99.5 % constant, decaying | Slots updated by a convex gate between the old slot and an RMS-normalised candidate attended from the tokens; gate bias −2, weight std 0.01. Bounded, input-dependent, no forced decay. | `test_core.py::test_state_is_gated_input_dependent_and_carried`, `::test_persistent_state_is_learnable_eight_step_recall` |
| Core could not read the current event | `b = norm(h_last + W_direct · norm(x_last) + W_state · mean(slots'))`, `W_direct` initialised at 0.5 · I. | `test_core.py::test_belief_reads_the_current_event_directly` |
| Text-initialised soft prefix leaked into generations | No text anywhere in the bridge; virtual prefix tokens are read by zero-initialised cross-attention (exact no-op at init). See change request A-2 for why a literal zero prefix is not a no-op. | `test_backbone.py::test_bridge_is_an_exact_noop_at_init_and_conditions_after_one_step`, `test_gpu_qwen.py::test_bridge_is_an_exact_noop_at_init_on_the_real_backbone` |
| Policy ignored CAM's latent (KL 4e-9) | Every trunk input RMS-normalised on its own; belief × action multiplicative term; situation vector re-normalised. | `test_api.py::test_permuted_belief_changes_the_decision_latent` |
| Semantic-side gradients 100–1000× the core | Groups are exactly `nn.api.GROUPS`; `grad_norms()` / `clip_gradients({group: max})`; LoRA weights float32; bridge output scaled by the typical (median) per-dimension spread of the hooked layer, not its outlier-dominated RMS. Measured norms in `params.json`. | `test_api.py::test_group_hashes_change_after_an_optimizer_step` |
| Ill-conditioned features — **not fixed, measured** | `phi` = RMS-normalised features scaled to unit L2 norm + constant coordinate `1/sqrt(F)`: bounded, but on realistic inputs at initialisation **not** well-conditioned (an earlier test fed white noise: max / min eigenvalue 10). Real configuration, 2304 realistic (context, action, goal, query) combinations: max / min eigenvalue of the Gram 2.3e4, 90 % of the mass in 23 of 128 directions (5.4e4 / 15 before the argument channel); one query alone 4.5e5, participation ratio 2.0. What is tested is that the float64 Joseph updates stay exact on such features. | `test_api.py::test_features_on_realistic_inputs_conditioning_and_fast_layer_numerics`, `test_gpu_qwen.py::test_features_on_realistic_inputs_real_configuration` |
| Candidates differing in one argument looked identical (cos(z) 0.9997 for `{"ref": "e3"}` vs `{"ref": "e4"}`) | Target and argument leaves go through the exact value channel as well as the rendered text: numbers as exact quantities (`arg.<path>`) plus a hashed identity of every leaf (`arg.<path>=<value>`), no new parameters. Qwen3-4B: cos(z) 0.747 for e3 / e4, 0.584 for two different operations (were 0.9997 / 0.82). | `test_api.py::test_candidates_differing_in_one_argument_are_distinguishable`, `test_gpu_qwen.py::test_one_argument_changes_the_decision_latent_on_the_real_backbone` |
| An embedding depended on the batch it was first computed in (~1 %, bf16 reduction order), the cache kept whichever came first | Every text runs in a forward of a fixed shape for its bucket length (`embed_bucket`, `embed_rows`; unfilled rows repeat a text of the call): bit-exact independence of the other texts. `with_grad=True` returns the canonical (cached) value bit-exactly, with the gradient of a separate forward over the call's texts alone (dynamic padding, no filler rows, so no extra activation memory); that gradient is checked against finite differences of the returned value. | `test_backbone.py::test_embedding_is_independent_of_the_other_texts_in_the_call`, `test_gpu_qwen.py::test_embeddings_and_beliefs_do_not_depend_on_batch_composition` |
| bf16 LM head: log-probabilities off by up to 0.14 nats per token on the same hidden states | Final RMSNorm recomputed in float32 from its input, head in float32 (the bf16 weight used as stored, two-term split of the float32 input, float32 accumulation, chunked over the vocabulary), for `generation_logprob`, the sampler and `next_token_logits`. | `test_backbone.py::test_float32_head_matches_a_float64_reference_with_its_gradient`, `test_gpu_qwen.py::test_generation_logprob_has_a_float32_output_layer`, `test_gpu_qwen_fp32.py` |
| Caches stale after writes that bump no version counter (`.data`) | Every embedding / generation / log-probability call compares the content of every tensor the caches depend on (LoRA weights, calibration statistics) with a snapshot (`sync_caches`, ~1 ms) and drops the embedding cache and the LoRA inference casts when it changed. | `test_backbone.py::test_caches_are_invalidated_by_any_change_of_what_they_depend_on` |
| bf16 master weights rounding optimizer steps away | LoRA A/B and bridge parameters are float32 and cast inside the forward. | `test_gpu_qwen.py::test_parameter_counts_of_the_real_configuration` |
| LM hidden states dominated by a few huge shared dimensions | Pooled embeddings standardised per dimension with a fixed neutral calibration corpus (stored as buffers in the trainable state); bridge queries standardised per dimension. | `test_gpu_qwen.py::test_semantic_embeddings_are_meaningful_and_cached` |

## Causality and time

`h[:, i]` depends on items `≤ i` only (bit-identical under any change to later items). Token `i` encodes the
log-scaled gap `t_i − t_{i−1}`; attention from `i` to `j ≤ i` gets a learned per-head function of `t_i − t_j`
(radial + Fourier basis over `log1p(seconds)`), so for the newest position this is exactly the age of every earlier
event relative to the newest one. The persistent slots are read with the time elapsed since the previous call.

## Fast state

Per outcome kind (real / count / binary / time-to-event) one posterior `N(mu, Sigma)` over `phi`, float64.
Real: exact conjugate update in Joseph form, noise either inferred (normal-inverse-gamma, Student-t predictive;
default; `predict()` reports the variance of that Student-t, `(b / (a - 1)) (1 + phi^T V phi)`, split into epistemic
and noise parts) or a learned constant. Count / binary / time-to-event: the Laplace approximation of a single
observation is the mode of a strictly concave scalar problem on the linear predictor, found by a monotone safeguarded
search (Poisson family: Newton on the stationarity condition and on its log form, both stay on one side of the root;
binomial: bracketed Newton with the two tail forms) until `|step| <= 1e-11 (1 + |eta|)`, at most 100 iterations
(measured: <= 7 for the Poisson family and <= 16 for 1e9 binomial trials, also for a censoring window of a year in
seconds or 3 events in 1e7 impressions); iterations and convergence are returned. The gradient is the implicit-function
gradient at the mode. Applied as a Joseph update; right-censoring via the Poisson-exposure identity. The learned
dispersion (`exp(disp_raw)`, 1 at initialisation) tempers count likelihoods only. Impossible observations (negative
count or time, successes outside [0, trials], ...) raise `ValueError` before anything is absorbed. `update` returns
surprise (predictive NLL; adaptive Gauss–Hermite centred on the integrand's mode for GLMs) and the closed-form
belief-change KL. Stabilised forgetting flattens the posterior towards the learned prior (bounded in
every direction); a CUSUM on standardised surprise detects abrupt mechanism changes. Categorical outcomes are not
modelled by the fast layer.

## Fixes after the independent verification (`docs/verification/G_nn.md`)

Probes: `docs/verification/repro/nn/`; numbers below are from running them on the final code.

| Defect | What changed | Probe now | Still open |
|---|---|---|---|
| D1 GLM update not converged far from the prior predictor (rate off x1.9-x339) | Monotone safeguarded mode search to `|step| <= 1e-11 (1 + |eta|)`, cap 100, iterations and convergence returned; implicit-function gradient; `online_nll` and `fast_compare` use it | `probe_fast.py`: all 11 "Laplace mode reached" rows hold (error 0.000 nats; 30 days / one year of censoring in seconds, 3 in 1e7, 0 at 1e9, 1e6 of 1e6 trials) | — |
| D2 count dispersion tempered binary / time-to-event | Dispersion applied to counts only; `exp(disp_raw)`, exactly 1 at init (was 1.018); the Cholesky jitter is a floor, so the default prior is `prior_var * I` | stateful binary / time-to-event == untempered Laplace (1.3e-10); 50 updates at dispersion 7.4: eta 2.527 == exact | — |
| D3 bf16 LM head | float32 final norm + float32 head (see the table above) | head error on the same hidden states 0.143 -> 0.0000 max per token | vs the float32 **model** the bf16 trunk still gives -0.16 / -0.36 / +0.12 / -0.05 / -0.00 nats for completions of 1 / 1 / 11 / 1 / 2 tokens (was -0.21 / -0.31 / +0.13 / -0.02 / +0.01): the head was not the main error. KV-cache sampler vs `generation_logprob` on the same tokens: max 0.073 per token (was 0.10). The probe checks "within 0.05 nats per token of the float32 model" and "sampler agrees to 0.05" still fail. |
| D4 NIG `predict()` variance | epistemic + noise = `(b / (a - 1)) (1 + q)` (Student-t variance), `linear()` likewise | ratio 1.000 (was 0.5 at the default prior) | the probe's Student-t calibration check standardises by `predict()` and compares with Student-t quantiles, which apply to the *scale* `(b / a)(1 + q)`: it now reads 0.924 instead of 0.899 — that check and the D4 check cannot both hold for a < infinity |
| D5 impossible observations absorbed | `check_observations` in `update` and `online_nll` (masked rows ignored); state untouched on rejection | holds | — |
| D6 conditioning tested on white noise | test on realistic pipeline inputs; numbers reported (table above) | probe check (< 1e4) still fails: 2.3e4 (was 5.4e4) | not fixed: an initialisation property; the fast layer stays exact on such features |
| D7 one-argument candidates indistinguishable | argument value channel | cos(z) e3 / e4 0.747 (was 0.9997), two operations 0.584 | the recorded init diagnostics `z_cos_between_two_actions` / `action_embedding_cos_between_two_actions` changed to 0.586 / 0.405 by design |
| D8 batch-composition dependence | fixed-shape embedding forwards | embedding alone vs in a batch 0.0 (was 0.011); b 0.0 (was 0.014); `with_grad` vs cache 0.0 (was 1.6e-4) | — |
| D9 `arch_signature` incomplete | backbone settings incl. `max_text_tokens`, tiny-LM seed, dropout, encoding version | `probe_saveload_cpu.py` 8 / 8 | — |
| D10 stale caches after `.data` writes | content check per call; `mark_adapters_updated` clears the LoRA casts | `probe_stale_cache.py` 3 / 3 | — |
| D11 weak tests | causality with a live time bias (and shown to catch a leaking time basis); real-backbone gradients checked by finite differences on the float32 backbone (`test_gpu_qwen_fp32.py`); GLM updates against scipy references far from the prior; save -> fresh process -> load | — | `test_fast_compare.py` still checks the recorded JSON's structure only |

## Not in this workstream

Losses, optimizers and schedules (D), prediction / policy / VOI heads (B), the reward generator and meta-gradient (C)
build on the tensors above and register their modules with `CamLiveModel.register_head`.

## Commands

```
cd /workspace/camlive
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live              # default: CPU, seeded (56 s at load average 21 on 48 cores; the verifier measured 372 / 626 s on a more loaded machine)
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live -m gpu       # real Qwen3-4B in bf16, then in float32 (each takes the GPU lock)
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live -m slow      # instantiates every preset on CPU
cd cam_live
PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.bench                         # -> docs/evidence/A/params.json
PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.fast_compare --steps 2000 --seeds 0 1   # -> fast_adaptation.json
```

One model instance must not run embedding / generation from several threads at once (the active bridge memory and
the tap capture are instance state).


## Measured: fast adaptation (`docs/evidence/A/fast_adaptation.json`, `fast_adaptation_noise_mode.json`)

Mean prequential NLL over support sizes 1..31 (nats; lower is better; 2 seeds x 600 tasks per split):

| Target / split | learned-prior Bayes | BLR on raw inputs | GRU | running statistics |
|---|---:|---:|---:|---:|
| Gaussian, held-out families | 0.8909 | 1.1149 | 1.1196 | 1.4483 |
| Gaussian, training families | 0.2132 | 0.7996 | 0.5753 | 1.4447 |
| Count, held-out families | 2.6201 | 3.3037 | 2.6381 | 5.7539 |
| Count, training families | 2.275 | 3.2697 | 2.3903 | 6.2771 |

Where the learned prior does **not** win:

* Gaussian targets before about three observations: held-out NLL at k=0 is 1.6995 against 1.4188 (GRU) and 1.467 (running statistics); at k=1 1.5156 against 1.3854. It leads from k=3 (1.3269 vs 1.3325). The conjugate normal-inverse-gamma prior ties weight uncertainty to the unknown noise scale, so its prior predictive is heavy-tailed.
* With a fixed learned noise variance instead (ablation) the k=0 prediction is calibrated (1.4005) but the layer becomes over-confident on held-out families: mean NLL 1.9933 against 0.8909. Inferred noise is therefore the default (`nn.fast_noise_mode = "nig"`).
* Counts on held-out families: 2.6201 against 2.6381 for the GRU with a negative-binomial head — a margin of -0.018 nats (z = -2.31), and the GRU is ahead at k=31. Not a robust win.

Consequence for the world model (Workstream B): at very small support the head's own prediction should carry the
forecast and the fast layer should act as the correction that takes over as evidence accumulates.

## Measured: real configuration (`docs/evidence/A/params.json`)

Frozen 4,022,468,096 · experiential 50,953,765 ·
adapters 16,713,856 (LoRA 11,796,480 + bridge 4,917,376) ·
total persistent trainable 67,667,621. Load 6.22 s,
7.804 GB after load, peak 13.141 GB over the whole benchmark.
