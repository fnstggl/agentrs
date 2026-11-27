# Change requests from Workstream A (neural architecture)

All of these are worked around inside `src/cam_live/nn/`; nothing is blocked.

## A-1. `config.SCALE_PRESETS`: `d_ff` overshoots every target

**What.** The master presets size only the transformer blocks. With the tokenizer projections, slot
cross-attention, gated slot write, decision trunk and feature map, they measure 5m → 7.67 M, 20m → 29.6 M,
50m → 67.8 M, 150m → 173.8 M, 500m → 520.4 M, 1b → 1018.6 M.

**Workaround.** `nn/presets.py::NN_PRESETS` holds corrected values; `resolve_core()` substitutes them when
`cfg.core` is an untouched master preset (a hand-edited `cfg.core` is respected). Three small presets also use a
low-rank semantic projection (`nn` section key `sem_rank`), because a full 7680 → d_model projection alone is 40 % of
a 5 M budget.

**Measured with the corrected values** (`python -m cam_live.nn.bench`, `docs/evidence/A/params.json`):
5m 5,044,741 · 20m 20,087,333 · 50m 50,953,765 · 150m 151,022,853 · 500m 502,708,549 · 1b 1,000,547,333 ·
debug 202,421.

**Proposed diff** (`src/cam_live/config.py`):

```diff
-    "debug": dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, n_slots=4, n_features=16, target_params=0.2e6),
-    "5m": dict(d_model=256, n_layers=4, n_heads=4, d_ff=640, n_slots=8, n_features=64, target_params=5e6),
-    "20m": dict(d_model=512, n_layers=5, n_heads=8, d_ff=1280, n_slots=8, n_features=96, target_params=20e6),
-    "50m": dict(d_model=768, n_layers=6, n_heads=12, d_ff=1792, n_slots=8, n_features=128, target_params=50e6),
-    "150m": dict(d_model=1024, n_layers=10, n_heads=16, d_ff=2560, n_slots=16, n_features=192, target_params=150e6),
-    "500m": dict(d_model=1536, n_layers=14, n_heads=24, d_ff=4096, n_slots=16, n_features=256, target_params=500e6),
-    "1b": dict(d_model=2048, n_layers=16, n_heads=32, d_ff=5376, n_slots=16, n_features=256, target_params=1.0e9),
+    "debug": dict(d_model=64, n_layers=2, n_heads=4, d_ff=64, n_slots=4, n_features=16, target_params=0.2e6),
+    "5m": dict(d_model=256, n_layers=4, n_heads=4, d_ff=320, n_slots=8, n_features=64, target_params=5e6),
+    "20m": dict(d_model=512, n_layers=5, n_heads=8, d_ff=576, n_slots=8, n_features=96, target_params=20e6),
+    "50m": dict(d_model=768, n_layers=6, n_heads=12, d_ff=1024, n_slots=8, n_features=128, target_params=50e6),
+    "150m": dict(d_model=1024, n_layers=10, n_heads=16, d_ff=2048, n_slots=16, n_features=192, target_params=150e6),
+    "500m": dict(d_model=1536, n_layers=14, n_heads=24, d_ff=4096, n_slots=16, n_features=256, target_params=500e6),
+    "1b": dict(d_model=2048, n_layers=16, n_heads=32, d_ff=5504, n_slots=16, n_features=256, target_params=1.0e9),
```

and `CoreConfig.d_ff: int = 1024`. If this is applied, keep `NN_PRESETS` (it still carries the `sem_rank`
overrides: 5m → 96, 20m → 192, 50m → 384). After the change `resolve_core()` becomes an identity on presets.

## A-2. Architecture §2: a zero-initialised literal soft prefix is not a no-op

**What.** §2 asks for "`bridge_tokens` virtual prefix embeddings … zero-initialised output" *and* an exact no-op at
initialisation. Tokens placed literally in the sequence cannot satisfy both: even an all-zero embedding receives
softmax attention mass (its key is 0, so its logit is 0) and shifts positions. Measured on Qwen3-4B: 8 all-zero
prefix embeddings change the next-token logits by up to 1.05 (`params.json` →
`real_configuration.bridge_noop_check`). Masking them out would make the gradient exactly zero, so they could never
start learning.

**Implemented instead.** `b` is mapped to `bridge_tokens` virtual prefix tokens (width `bridge_dim`) that every
sequence position reads through cross-attention added to the residual stream of `bridge_layers` evenly spaced decoder
layers. Only the cross-attention **output projection** is zero-initialised: logits are bit-identical with and
without a latent at init, and that projection receives a non-zero gradient from the first step (tested on the tiny
LM and on Qwen3-4B). No text ever initialises the bridge.

**Proposed diff** (`docs/CAM_LIVE_ARCHITECTURE.md` §2, "Semantic adapters"):

```diff
-  from CAM's belief `b` to `bridge_tokens` virtual prefix embeddings (and optionally gated cross-attention into
-  decoder layers) that conditions generation. The bridge must start as an exact no-op (zero-initialised output,
-  not a text-initialised prefix: text-initialised prefixes leaked into generated strings before).
+  from CAM's belief `b` to `bridge_tokens` virtual prefix tokens that every position reads through cross-attention
+  added to the residual stream of several decoder layers. The bridge must start as an exact no-op: the
+  cross-attention output projection is zero-initialised (a literal zero soft prefix is not a no-op — it takes
+  attention mass — and text-initialised prefixes leaked into generated strings before).
```

## A-3. `nn/api.py`: optional `grad_last` on `tokenize`, third key from `load_trainable_state`

**What / why.** `tokenize(contexts, with_grad=True)` re-encodes every item text of every context through the
backbone with a graph (up to `batch × max_context` texts): too much memory for the slow adapter schedule.
`CamLiveModel.tokenize` therefore accepts `grad_last: Optional[int] = None`: only the newest `k` items of each
context get an adapter graph, older ones use cached detached embeddings. `load_trainable_state` additionally
returns `"mismatched"` (keys whose shapes differ; never loaded, an error under `strict=True`).

**Proposed diff** (`src/cam_live/nn/api.py`):

```diff
-    def tokenize(self, contexts: Sequence[Sequence[ExperienceItem]], *, with_grad: bool = False) -> TokenBatch: ...
+    def tokenize(self, contexts: Sequence[Sequence[ExperienceItem]], *, with_grad: bool = False,
+                 grad_last: Optional[int] = None) -> TokenBatch:
+        """``grad_last=k``: with ``with_grad=True`` only the newest k items of each context get a graph."""
-        """Returns {"missing": [...], "unexpected": [...]}; ``strict=False`` tolerates newly added heads."""
+        """Returns {"missing": [...], "unexpected": [...], "mismatched": [...]}; ``strict=False`` tolerates newly
+        added heads (shape-mismatched tensors are reported and never loaded)."""
```

and in `learning_api.encode_decisions` pass `grad_last` through when `with_grad_semantic=True`
(suggested default 2).

## A-4. Architecture §2: where the time difference to the newest token lives

**What.** The tokenizer does not put "age relative to the newest token" into the token embedding: that would make
`h[:, i]` depend on a later item's timestamp and break the causality contract of `nn/api.py` (and leak the time to
the end of the context into every prefix latent that the evaluator and RUDDER read). A token encodes the log-scaled
gap to its predecessor; the core's attention receives a learned per-head function of `t_i − t_j` (`j ≤ i`), which
for the newest position is exactly the age relative to the newest token. `TokenBatch.time` keeps absolute float64
seconds. No code change requested — a wording note for §2 ("Experience tokens … continuous-time encoding").

## A-5. Categorical outcomes are not modelled by the fast layer

`FastBeliefState.supports(OutcomeKind.CATEGORICAL)` is `False` and `update` raises `ValueError` for it: a categorical
outcome is not a scalar `phi^T w`. Workstream B's categorical head must be used alone for that kind. No contract
change requested; recorded so the limitation is visible.
