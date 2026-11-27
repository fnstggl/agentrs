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
| Ill-conditioned features | `phi` = RMS-normalised features scaled to unit L2 norm + constant coordinate `1/sqrt(F)`. | `test_api.py::test_feature_gram_is_well_conditioned` |
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
default) or a learned constant. Count / binary / time-to-event: the Laplace approximation of a single observation is
found exactly on the scalar linear predictor (safeguarded Newton) and applied as a Joseph update; right-censoring via
the Poisson-exposure identity. `update` returns surprise (predictive NLL; adaptive Gauss–Hermite for GLMs) and the
closed-form belief-change KL. Stabilised forgetting flattens the posterior towards the learned prior (bounded in
every direction); a CUSUM on standardised surprise detects abrupt mechanism changes. Categorical outcomes are not
modelled by the fast layer.

## Not in this workstream

Losses, optimizers and schedules (D), prediction / policy / VOI heads (B), the reward generator and meta-gradient (C)
build on the tensors above and register their modules with `CamLiveModel.register_head`.

## Commands

```
cd /workspace/camlive
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live              # default: CPU, seeded, ~35 s
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live -m gpu       # real Qwen3-4B (takes the GPU lock)
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live -m slow      # instantiates every preset on CPU
cd cam_live
PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.bench                         # -> docs/evidence/A/params.json
PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.fast_compare --steps 2000 --seeds 0 1   # -> fast_adaptation.json
```

One model instance must not run embedding / generation from several threads at once (the active bridge memory and
the tap capture are instance state).
