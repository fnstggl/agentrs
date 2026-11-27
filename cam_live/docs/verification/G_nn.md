# G-nn — independent verification of the neural architecture (workstream A)

Verifier: G, lens `nn`. Scope: requirements NA-01 NA-02 NA-03 NA-04 NA-05 NA-07 NA-09 NA-10 NA-11 NA-12 NA-13.
Code under test: `src/cam_live/nn/` at branch `feat/cam-live` (A's status commit `40b55ab`, last change under `nn/` `b9e671e`; nothing under `nn/`
changed afterwards). Environment: torch 2.15.0+cu132, transformers 5.24.0, RTX A6000 (shared), Qwen/Qwen3-4B at
revision `1cfa9a7208912126459214e8b04321603b3df60c` (local, offline). No source file and no existing test was
modified. Verdicts: `verdicts_nn.json`. All commands run from `/workspace/camlive`.

## 1. What was executed

| What | Command | Result |
|---|---|---|
| A's default tests | `.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live` | 61 passed, 7 skipped (6 gpu + 1 slow) |
| A's GPU tests (real Qwen3-4B) | `... -m gpu` | 6 passed, 13.5 s |
| A's slow test | `... -m slow` | 1 passed, 32 s |
| Invariants, debug and 50M core | `.venv/bin/python cam_live/docs/verification/repro/nn/probe_invariants.py [--scale 50m]` | 23 / 23 checks hold at both scales |
| Bayesian fast layer vs float64 references | `.venv/bin/python cam_live/docs/verification/repro/nn/probe_fast.py` | 44 checks, 12 refuted (D1, D2, D4, D5) |
| Presets, real instantiation | `.venv/bin/python cam_live/docs/verification/repro/nn/probe_presets.py` | 12 / 12 |
| Save → fresh process → load (tiny, CPU) | `.../probe_saveload_cpu.py save DIR` then `... load DIR` | bit-exact; D9 |
| Adapter cache staleness | `.../probe_stale_cache.py` | D10 |
| Real backbone: parameters | `.../probe_real_backbone.py --part params` | 16 / 16 |
| Real backbone: gradient paths (bf16) | `... --part grads` | 25 / 25 (bf16 finite differences are printed as information only: logit quantisation at small steps, curvature at large ones) |
| Real backbone in **float32**: finite differences of every gradient path; bf16 vs float32 gradients | `... --part gradsave --dir DIR` then `... --part fd32 --dir DIR` | all gradient checks hold; D3 |
| Real backbone: latent bridge | `... --part bridge` | 9 / 9 |
| Real backbone: value channel, init diagnostics | `... --part values` | 13 / 13 |
| Real backbone: save → fresh process → load | `... --part save --dir DIR` then `... --part load --dir DIR` | bit-exact, 9 / 9 |
| Real backbone: throughput / memory | `... --part throughput`, `... --part misc` | claims reproduce (section 5) |
| Real backbone: decision-path separation, feature conditioning | `... --part phi`, `... --part cond` | D6, D7, D8 |
| Fast-adaptation evidence, seed 0, full length | `cd cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.fast_compare --steps 2000 --seeds 0 --threads 8 --out /tmp/x.json` | recorded per-seed numbers reproduce to 4 decimals (section 4) |
| Noise-mode ablation, seed 0, full length | `... -m cam_live.nn.fast_compare --noise-ablation --steps 2000 --seeds 0 --threads 8 --out /tmp/y.json` | see section 4 |
| Stronger recurrent baseline | `.../probe_fast_compare_gru.py --hidden 256 --steps 16000 --kind real` | see section 4 |

Every probe prints `[PASS]` / `[FAIL]` per check and exits non-zero when a check is refuted. GPU probes take the
shared lock through `cam_live.nn.gpu.gpu_lock` for one part at a time (longest hold: 156 s, the float32 part, peak
16.7 GB; every other part below 13.2 GB).

## 2. Claims that survived the attack

**Parameter accounting (real configuration).** Independent count over the module tree: frozen
4,022,468,096 (all `requires_grad=False`, bf16); LoRA 11,796,480 = 36 layers × rank 32 × ((2560+4096) + (2560+1024)),
float32; bridge 4,917,376; experiential 50,953,765 (tokenizer 4,263,746 · core 36,615,648 · encoders 1,977,600 ·
trunk 4,720,128 · feature map 3,343,104 · fast prior 33,539); total persistent trainable 67,667,621. `param_groups()`
keys are exactly `nn.api.GROUPS`, groups are disjoint, their union is `model.parameters()`, and the set of
`requires_grad` tensors equals the union of the trainable groups. `trainable_state()` holds 315 tensors / 0.252 GB,
no frozen tensor, every trainable tensor, plus five calibration buffers. The only optimizers built over the model
(`credit/learner.py:236-250`, `predict/labdata.py:95`) take `experiential` / `heads` / `semantic_adapters` only, and
`credit/checkpoint.py:160-170` drops the frozen group.

**Not a frozen-LLM wrapper — gradient paths, with magnitudes.** With the backbone loaded in float32 (same code path,
`cfg.backbone.dtype="float32"`), central finite differences along the analytic gradient agree with its norm to
better than 1 % for: embedding loss → LoRA-B (1.000 / 0.999); generation log-probability → LoRA-B (0.998 / 0.999);
generation log-probability → the zero-initialised bridge output projections (1.002 / 0.999); with non-zero adapters:
the full adapter gradient (0.994 / 1.002), d/d latent (1.003 / 1.001), generation log-probability → bridge → `b` →
all experiential weights (0.996 / 0.995), and a loss on `z` through `tokenize / encode_actions /
encode_goals(with_grad=True)` → LoRA (1.000 / 1.000). Gradient checkpointing on vs off: relative difference 0.
The bf16 gradients used in normal operation agree with the float32 ones: cosine 0.995 / 0.999 / 0.999, norm ratio
0.975 / 0.997 / 0.999. All 72 LoRA modules receive gradient through the embedding path and through the generation
path; frozen tensors never get a `.grad`; `with_grad=False` builds no adapter graph. Nothing on these paths is
detached: the only `.detach()` / `no_grad` uses in `nn/` are the cache (`with_grad=False`), `generate`, calibration,
hashing and diagnostics. By design, at initialisation the generation loss gives exactly zero gradient to the
experiential weights and to every bridge tensor except `xattn[*].out.weight` (the output projection is zero); it
flows as soon as that projection has moved.

**Latent bridge.** At initialisation: next-token logits bit-identical with and without a latent (latent norms 1e-3
to 1e3, four prompts), `generation_logprob` bit-identical, greedy and seeded sampled generations identical — so no
text can come from the bridge; its only corpus-derived content is three statistics buffers. Gradient on the
zero-initialised projections at step 0: norm 45.9. After **one** Adam step (lr 1e-3) on the adapters the greedy text
changes (`'Red.'` → `"Sure! One color is **blue**…"`), TV(latent, −latent) = 0.42. Training the bridge alone for 40
steps makes two random latents produce `Green…` / `Purple…` for the same prompt while the unconditioned output stays
`'Red.'` (log-probability of the wrong pairing −26 / −30.5). Changing only experiential weights changes generation
log-probabilities through `latent=b`.

**Causality and invariances** (experiential network, attention time bias randomised so that path is live; debug and
50M scale): `h[:, :k]` is bit-identical under any change of items ≥ k (text, values, kind, trust, timestamps), also
when only later timestamps move; `d h[i] / d x[j>i]` is exactly 0; truncation changes the prefix by 1.9e-6 (float32);
padding 1.9e-6; batch order / other rows 0; candidate `n`'s `e_a` and `z` do not depend on other candidates (2.3e-6)
and the candidate axis is permutation-equivariant; `phi[b, n, q]` does not depend on other queries; an absolute
shift of all timestamps by 1.7e9 s changes nothing (0, and 6.6e-7 with fractional timestamps); irregular gaps change
`b` by 18 % relative. `z` at padded candidate slots is non-zero — callers must use `a_mask`.

**Persistent state.** Streaming is *not* equivalent to one-shot processing (relative difference of `b` 0.12 for
6 + 4 items vs 10) and the design does not claim it; the second call's belief depends on the first call's content
(0.19), on the time elapsed since it (0.008), `h` of the later call depends on the state too (0.10); save → load →
continue is bit-exact; the state passed in is not mutated; 300 calls keep the slots at RMS 0.86.

**Numeric values (real backbone).** A's init diagnostics reproduce exactly (max deviation 0.0000). Independently:
cos(+1, −1) ≤ 0.187 for texts of 1 to 380 tokens; cos(+1e15, −1e15) 0.11; cos(3, 4) 0.65; cos(100, 103) 0.975;
cos(0, missing) 0.72; cos(1e12, 1e15) 0.73; content and value channels both have RMS 1.0. A held-out linear probe
over 2400 random (text, value) pairs recovers asinh(value) with R² 0.993 from the token and 0.997 from the belief
`b`, sign accuracy 1.0 (control with the number only in the text: R² 0.83). `TokenBatch.values` carries exact
float64 (1e15+1, a denormal, the largest double). Non-finite quantities are dropped and counted; quantities beyond
four per item are continued in further items by `to_items` and dropped-but-counted if passed to `tokenize` directly.

**Save / load.** Real configuration, two optimizer steps on every trainable group, `trainable_state()` → file
(0.254 GB) → fresh process → `load_trainable_state(strict=True)`: embeddings, tokens, `h`, `b`, slots, `e_a`, `e_g`,
`e_q`, `z`, `phi`, a registered head, bridged logits, generation log-probabilities and generated text are all
bit-identical; group hashes equal. Wrong shapes and missing tensors are rejected under `strict=True` without a
partial load.

**Scale presets.** All six presets instantiated with real weights on CPU: 5,044,741 / 20,087,333 / 50,953,765 /
151,022,853 / 502,708,549 / 1,000,547,333 — the claimed counts exactly, all within 1 % of target; forward finite for
all, backward for ≤ 150m.

**Fast layer, the parts that hold.** Known-noise Gaussian and normal-inverse-gamma recursions equal independent
closed forms to 1e-15 (mean, covariance, a, b); summed one-step surprises equal the exact log marginal likelihood
(Gaussian: 3e-14; Student-t: 0); `kl_from_prior` equals the closed-form KL; order invariance 8e-16; 50,000
near-collinear updates stay symmetric positive definite (2.9e-11 from the closed form); epistemic variance is
non-increasing in the number of observations; GLM updates with moderate data equal a reference Laplace solution to
4e-15 for count, binary and time-to-event with censoring; the adaptive Gauss–Hermite predictive NLL equals
brute-force quadrature to 2.6e-7; predictive moments match Monte-Carlo; forgetting equals the stated
precision interpolation and keeps a variance floor; forgetting / CUSUM track an abrupt change (error 1.77 / 1.09
against 3.71 static); calibration on data from the model: 90 % interval coverage 0.8997, E z² 0.992 (known noise),
Student-t coverage 0.899 (NIG); state round trip through `torch.save` is bit-exact.

## 3. Defects

Severity: critical = a core claim is false in default operation; major = wrong numbers under realistic inputs;
minor = limited effect, weak test or reporting gap. No critical defect was found.

### D1 — major — GLM (count / time-to-event) updates are not the Laplace update when the observation is far from the prior predictor
`src/cam_live/nn/fast.py:248-258` (`_laplace_1d`: a fixed number of Newton steps from `eta0`; for Poisson /
exponential likelihoods a downward step is ≈ −1 per iteration whatever the distance), defaults `fast.py:490`
(`FastBeliefState`, 12 steps), `fast.py:435` (`online_nll`, 6 steps — the training path used by
`predict/providers.py:152,251`), `fast.py:291` (8).
Measured with the default prior (reference: Brent root on the same scalar problem, float64):

| Observation | reference mode | stateful (12 steps) | rate error | reported surprise vs brute force | `online_nll` default (6 steps) |
|---|---:|---:|---:|---:|---:|
| time-to-event censored at 30 days, unit 1 s | −12.25 | −11.63 | ×1.86 | 88.76 / 88.76 | −6.00 |
| time-to-event censored at 1 year, unit 1 s | −14.57 | −11.97 | ×13.5 | 192.3 / 122.3 | −6.00 |
| count 3 with exposure 1e7 | −13.31 | −11.87 | ×4.2 | 103.2 / 99.9 | −6.00 |
| count 0 with exposure 1e9 | −17.83 | −12.00 | ×339 | 5597 / 178 | −6.00 |
| time-to-event censored at 1 day, unit 1 s | −9.14 | −9.14 | ×1 | 52.19 / 52.19 | −5.99 (error +3.15) |

Upward moves converge (count 1e9 with exposure 1: exact). `DESIGN.md` ("found exactly on the scalar linear predictor
(safeguarded Newton)") and the docstring `fast.py:14-17, 295-299` are therefore not true in general.
`predict/predictor.py:596-601` rescales exposure / time by the head's own prediction before calling the fast layer,
which bounds the effect in the running system once the head is roughly right; the layer's own API (`exposure` =
"the time unit `y` is expressed in") does not.
Repro: `.venv/bin/python cam_live/docs/verification/repro/nn/probe_fast.py | grep "Laplace mode"`.
`tests/nn/test_fast.py:178` cannot see this (functional form, 30 steps, y = 9, exposure 2).

### D2 — minor now, grows with training — the count dispersion tempers binary and time-to-event updates
`src/cam_live/nn/fast.py:676` (`_apply`) and `fast.py:464` (`online_nll`) pass `prior["dispersion"]` for every GLM
kind; `FastPrior` documents it as "the count dispersion" and `predict()` applies it to counts only. At
initialisation (1.018) a stateful binary / time-to-event update differs from the exact Laplace update of its stated
likelihood by 2.6e-3 / 3.7e-3 in the weights. With the dispersion at 3.13 (what training on over-dispersed counts
produces), 50 identical observations give eta 1.85 instead of 2.53 (binary) and a posterior variance 1.79× too
large; time-to-event −2.15 instead of −2.81, variance ×1.59. Training on binary data also moves `disp_raw`.
Repro: `.venv/bin/python cam_live/docs/verification/repro/nn/probe_fast.py | grep -i dispersion`.

### D3 — minor — `generation_logprob` (and the sampler) are only accurate to 0.1–0.3 nats per token
`src/cam_live/nn/backbone.py:584` and `:725`: the LM head runs in bf16, so logits of magnitude 16–64 are multiples of
1/8 or 1/4 (max logit 54.75). Per-token |log p(bf16 head) − log p(float32 head on the same hidden states)|: mean
0.024, max 0.143. Against the same model in float32: −0.21, −0.31, +0.13, −0.02, +0.01 for completions of 1, 1, 11,
1, 2 tokens. KV-cache sampler vs `generation_logprob` on the same 16 greedy tokens: up to 0.10 per token. Gradients
are unaffected (cos ≥ 0.995 against float32). Matters wherever the value is used as a propensity or compared across
calls; computing the head for the selected positions in float32 removes the head part.
Repro: `... probe_real_backbone.py --part gradsave --dir DIR`, `--part fd32 --dir DIR`, `--part misc`.

### D4 — minor — NIG `predict()` under-states the predictive variance
`src/cam_live/nn/fast.py:608-611`: epistemic + noise = (b/a)(1+q); the variance of the Student-t it stands for is
(b/(a−1))(1+q). Ratio a/(a−1): 2.0 at the default prior (a0 = 2), 4.4 at the prior learned in A's recorded
experiment (a0 = 1.298). On data drawn from the model's own prior a Gaussian 90 % interval built from `predict()`
covers 0.84 / 0.87 / 0.86 / 0.87 at support sizes 0 / 1 / 5 / 24 (the Student-t interval covers 0.899). The docstring
says "plug-in"; `predict/predictor.py:180` consumes the pair as mean and variance.
Repro: `... probe_fast.py | grep NIG`.

### D5 — minor — impossible observations are absorbed
`src/cam_live/nn/fast.py:696` checks finiteness only. A negative count (returns surprise = inf, moves the mean by
2.2), successes > trials, and a negative time (surprise −0.5) all update the posterior.
Repro: `... probe_fast.py | grep impossible`.

### D6 — minor — "well-conditioned features" is tested on white noise only
`tests/nn/test_api.py:74-81` feeds i.i.d. Gaussian `z` and `e_q` into the debug feature map (max / min eigenvalue of
the Gram: 10 on the real feature map with that construction). On 2304 real (context, action, goal, query)
combinations of the real configuration the same statistic is 5.4e4 (the test's bound is 1e4), 15 of 128 directions
hold 90 % of the mass; for a single query it is 1.9e6 with a participation ratio of 1.6. The float64 Joseph updates
tolerate this numerically; the `DESIGN.md` row "Ill-conditioned features → fixed, tested" overstates what is shown.
Repro: `... probe_real_backbone.py --part cond`.

### D7 — minor — candidates that differ only in an argument are almost identical in `z` at initialisation
`src/cam_live/nn/model.py:159-177`: an action is the mean-pooled embedding of `render()` plus modality / level /
timeout / step count; `tokens.py:451 action_meta` (structured arguments) is never used. cos(z) for
`browser.click {"ref": "e3"}` vs `{"ref": "e4"}` = 0.9997 (two unrelated operations: 0.82). The recorded diagnostic
`z_cos_between_two_actions` 0.8256 only covers the easy case.
Repro: `... probe_real_backbone.py --part phi`.

### D8 — minor — text → belief depends on batch composition at the 1 % level on the bf16 backbone
The experiential network is exactly batch-invariant, but a text's semantic embedding differs by 1.1 % (relative)
between being embedded alone and inside a batch with longer texts, and the cache (`backbone.py:469-495`) keeps
whichever version was computed first; `b` of the same context differs by 1.4 %. `tokenize(with_grad=True)` vs the
cached default differs by 1.6e-4 (float16 storage).
Repro: `... probe_real_backbone.py --part values`, `--part phi`.

### D9 — minor — `arch_signature()` does not record `max_text_tokens`
`src/cam_live/nn/backbone.py:601-608`, `model.py:331-338`. A state loads with `strict=True` into a model with another
text truncation length, the signatures are equal, the outputs differ. (`lora_alpha`, tap layer ids, bridge layout
and the `nn` section are recorded.)
Repro: `... probe_saveload_cpu.py save DIR && ... probe_saveload_cpu.py load DIR`.

### D10 — minor — adapter caches go stale after a write that does not bump the tensor version
`src/cam_live/nn/backbone.py:105` (LoRA cast key) and `:323` (fingerprint) rely on `_version`. After
`lora_B.data.copy_(...)` the cached embedding is unchanged (off by 22) and no-grad logits differ from grad-mode
logits by 5.9 (tiny LM); `mark_adapters_updated()` does not clear the LoRA cast cache, so the no-grad path stays
stale. No code in the repository writes adapters through `.data` today.
Repro: `.venv/bin/python cam_live/docs/verification/repro/nn/probe_stale_cache.py`.

### D11 — minor — tests that cannot fail where it matters
* `tests/nn/test_core.py:10` (causality) runs with the attention time bias at its zero initialisation, so the
  time-conditioned attention path is not exercised (it does hold when randomised — section 2).
* `tests/nn/test_gpu_qwen.py:149-197`: gradients on the real backbone are only asserted to be non-zero.
* `tests/nn/test_fast.py:178`: see D1; the stateful path, binary and time-to-event are never compared with an exact
  solution (D2).
* `tests/nn/test_api.py:145`: the round trip is within one process.
* `tests/nn/test_fast_compare.py:46`: checks the structure of the recorded JSON only.

## 4. Fast-adaptation evidence (NA-13)

**Reproduction.** A full-length rerun of seed 0 (`--steps 2000 --seeds 0`, 41 min on the loaded machine) reproduces
every per-seed number recorded in `docs/evidence/A/fast_adaptation.json` — mean NLL, and NLL and RMSE at every
support size, for four methods × two targets × two splits — with zero difference at the four recorded decimals:

| Target, seed 0 | learned-prior Bayes | BLR on raw inputs | GRU | running statistics |
|---|---:|---:|---:|---:|
| Gaussian, held-out families | 0.8907 | 1.1123 | 1.1238 | 1.4472 |
| Gaussian, training families | 0.2206 | 0.8028 | 0.5762 | 1.4458 |
| Count, held-out families | 2.6132 | 3.2242 | 2.6317 | 5.7402 |
| Count, training families | 2.2577 | 3.1415 | 2.3684 | 5.5037 |

The stateful and functional code paths agree to 2e-14 in the recorded run. On seed 0 alone the count margin over the
GRU is −0.0186 ± 0.0105 (z = −1.76): A's description "not a robust win" is accurate.

**Honesty of the reported failures.** The three `scientific_failures` are what the evidence file contains (k = 0
NLL 1.6995 against 1.4188 for the GRU; fixed-noise variant 1.9933 on held-out families; count margin −0.018,
z = −2.31, GRU ahead at k = 31). One thing is in the evidence but not in the status file or `DESIGN.md`: on held-out
**Gaussian** families the GRU has the lower RMSE (0.8185 against 0.8629, mean over support sizes ≥ 1) — the learned
prior wins there on calibrated uncertainty (NLL), not on point accuracy.

**Noise-mode ablation.** A full-length rerun of seed 0 reproduces `fast_adaptation_noise_mode.json` exactly: inferred
noise 0.8907 held-out / 0.2206 in-distribution; fixed learned noise 1.9935 / 0.3033 (k = 31 on held-out families:
1.9256, i.e. over-confident as evidence accumulates), learned noise variance 0.0938 / 0.0768, learned shape a0 1.2977.
The fixed-noise failure is real and is reported as a failure.

**Stronger recurrent baseline.** (run in progress when this revision was committed; see the next revision.)


## 5. Status-file numbers spot-checked on the GPU

| `docs/status/A.json` `measured` | claimed | measured now |
|---|---:|---:|
| `gpu_memory_after_load_gb` | 7.804 | 7.804 |
| `embedding_tokens_per_s` 128 tokens | 10,625 | 10,226 |
| `generation_tokens_per_s` batch 1 / batch 8, no bridge | 24.1 / 192.7 | 24.1 / 179.8 (14.3 / 154.7 while a CPU job of mine ran) |
| `experiential_core_step_b16_t256` seconds / extra GB | 0.088 / 1.556 | 0.090 / 1.563 |
| `adapter_backward` generation, checkpointing: seconds / extra GB | 0.378 / 1.057 | 0.378 / 1.056 |
| same without checkpointing: extra GB | 5.266 | 5.281 |
| `joint_loss_grad_norms` experiential / adapters | 597.79 / 3.581 | 597.79 / 3.581 |
| `literal_zero_prefix_max_abs_logit_diff` | 1.0469 | 1.0469 |
| "about 15 % slower with trained adapters" (`not_done`, no number in the evidence file) | 15 % | 16.2 % (batch 1), 10.0 % (batch 8) |
| `init_diagnostics_real_configuration` (12 numbers) | — | identical |

## 6. Status-file statements that are not true as written

1. `DESIGN.md` "Fast state": the Laplace approximation "is found exactly" — not for observations whose mode is more
   than about 11 (stateful) / 5 (`online_nll`) nats below the prior predictor (D1).
2. `A.json` NA-07 limitations: "Count over-dispersion is a single learned tempering factor" — it also tempers binary
   and time-to-event updates (D2).
3. `DESIGN.md` "Ill-conditioned features … `test_feature_gram_is_well_conditioned`" — holds for white-noise inputs,
   not for real inputs (D6).
4. `backbone.py` docstring / `mark_adapters_updated`: the cache "is invalidated automatically when any adapter
   tensor changes" — only for writes that bump the version counter (D10).
5. `DESIGN.md` "default: CPU, seeded, ~35 s": the default suite took 372 s here, on a machine loaded by my own
   concurrent jobs; not re-timed on an idle machine.
6. `A.json` NA-13 / `scientific_failures` name the count RMSE but not that the GRU also has the lower RMSE on
   held-out **Gaussian** families (0.8185 against 0.8629, mean over support sizes ≥ 1; in the evidence file).

Everything else checked in `A.json` (`measured`, `not_done`, `scientific_failures`) matched what was reproduced.

## 7. Not checked

Qwen3-8B (not loaded, as A states); presets above 50m on the GPU; usefulness of trained representations or of a
trained bridge for candidate quality (needs the learner: D / S); adapter updates driven by real experience (D's
slow schedule — `credit/providers.py:633` passes `latent=b.detach()` to `generation_logprob`, so in that provider
the generation loss does not reach the experiential weights); seed 1 of the fast-adaptation runs.
