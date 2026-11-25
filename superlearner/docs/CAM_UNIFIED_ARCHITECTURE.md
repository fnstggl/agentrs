# CAM 7.0 Unified — architecture (binding design)

Contract code: `superlearner/src/superlearner/cam_unified/interfaces.py` (records, adapters, tensor protocols) and
`config.py` (scales and switches). This document explains the mathematics and assigns ownership. Workstreams code
against the contract and send change requests to the master; they do not edit either file.

## 0. Ground truth about the codebase

See §11 (filled from the Stage-0 mapping workflow).

## 1. One trainable intelligence

Per environment b and decision t the agent keeps

| Symbol | Type | Shape (main) | Owner |
|---|---|---|---|
| O_t | `TokenSet` of OBS_SEM / OBS_NUM / ENTITY / RELATION / EVENT tokens | [B, ≤384, 768] | A (tokens), B (entities) |
| Z_t | persistent latent state tokens | [B, 64, 768] | B |
| B_t | `FastState` (K=16 particles × 256, log-weights, diag scale, Kalman stats) → BELIEF tokens | [B, 17, 768] | B |
| M_t | retrieved MEMORY tokens | [B, ≤64, 768] | E |
| G_t | GOAL tokens | [B, ≤8, 768] | A |
| A_t | ACTION tokens (EncAction_θ) | [B, ≤4, 768] | A |

`Z_t = F_θ(Z_{t-1}, [O_t; A_{t-1}; B_t; M_t; G_t])`: the backbone (B) runs joint self-attention over the S state
tokens concatenated with the context tokens (masks + `TokenKind` embeddings + continuous-time encoding of
`TokenSet.time` + episode-local entity-slot codes); the outputs at the state positions are Z_t. The context budget is a
per-step working-memory budget, never a lifetime limit — the lifetime lives in the archive (E) and in the weights.

Scales (`config.SCALES`): dev 8×384×6 / 32 state tokens, main 16×768×12 / 64, large 24×1024×16 / 128. Blocks:
pre-norm RMSNorm → SDPA multi-head attention (flash kernels via `torch.nn.functional.scaled_dot_product_attention`)
→ RMSNorm → SwiGLU (d_ff = 64·round(8/3·d/64)); residuals; bf16 autocast; optional activation checkpointing.
Target: main ≈ 120–180 M core parameters, large ≈ 320–450 M; reported per component by `count_parameters()`;
the pretrained semantic model is reported separately.

Text is an input/output modality. The internal computational state is numerical (Z, R, B, M).

## 2. Semantic grounding and actions (workstream A) — `cam_unified/semantic/`

* `encoder.py` — `QwenSemantic` (Qwen3-1.7B by default, Qwen3-0.6B in tests): token-level hidden states
  (`output_hidden_states`, chosen layer), pooled embeddings, LoRA (peft) on attention projections, `version` string that
  changes with any trainable semantic update; a content-hash cache of hidden states keyed by (sha256, version) —
  never served across versions.
* `tokenizer.py` — `ObservationTokenizer(nn.Module)`: semantic observations → `sem_tokens_per_obs` learned
  attention-pooling queries over token hidden states → projection to d_model; numeric observations →
  [sign·log1p|v| Fourier features, unit embedding (semantic embedding of the unit string), entity slot, time]; tools →
  one TOOL token per tool from the encoded name+description+JSON schema (so unseen tools need no new head); goals →
  GOAL tokens (text + numeric targets); actions → ACTION tokens (tool token + encoded args/code). Exact content is never
  only in an embedding: every token keeps `refs` to the observation / artifact.
* `bridge.py` — `LatentBridge`: P_t = W_θ(R_t, G_t, B_t) → `n_prefix` soft prefix embeddings in the semantic model's
  input space (cross-attention from learned prefix queries to [R; G; B]).
* `actions.py` — `ActionGenerator` implements `ActionGeneratorP`. Factorisation
  π(a|R,G,B) = π(tool | R, tools) · π(evidence | R, obs) · Π_args π(arg | tool, R, ...):
  tool = pointer attention from R to TOOL tokens (any discovered tool, masked to the available set);
  evidence = pointer over observation tokens choosing ≤ `max_evidence` observations shown verbatim to the decoder
  (CAM, not the language model, decides what evidence the decoder sees); numeric/enum/bool arguments = typed heads
  conditioned on R and the argument's schema embedding; string/code arguments = the Qwen decoder conditioned on the
  bridge prefix, the tool schema and the selected evidence, JSON-schema-validated (invalid → logged failure, not a
  silent repair). `log_prob` recomputes the exact behaviour log-probability (tool + evidence + every argument token),
  differentiable into the backbone, bridge, typed heads and LoRA — this is what PPO/V-trace train. Every action carries
  `ActionSource`, `logp`, `logp_parts` and, when candidates were proposed, the proposal distribution.
* `sandbox.py` — local code-execution sandbox for code tasks outside OpenWorld (subprocess, rlimits, timeouts, temp
  cwd, no inherited credentials, network-disabling environment); OpenWorld code runs in Crucible's own sandbox.

## 3. Entities, relations, events (workstream B) — `cam_unified/core/entities.py`

e_i = E_θ(type, attributes, semantic, relationships, history): scatter-mean of the observation tokens bound to slot i
plus attention over them and a type embedding (semantic embedding of the free-text type, so categories are
extensible). r_ij = R_θ(e_i, e_j, relation observation, time) for observed relation tokens. Entity identity is a
random per-episode slot code (`EntityRegistry`), used for binding only.

## 4. Latent reasoning (B) — `core/reasoning.py`

R_0 = Z_t; R_{j+1} = Reason_θ(R_j, [Z_t; B_t; M_t; G_t]) reusing the last `reason_layers` backbone blocks (shared
weights) with an iteration embedding; J ∈ {1,2,4,8,16}. Adaptive computation: halting head p_halt(R_j) (ACT/PonderNet
style) with compute penalty `reason_compute_cost`·E[iters]. R conditions the policy, the decoder bridge, the value and the
exploration budget. Tests must intervene on R (and on J) and show decisions change.

## 5. Beliefs and fast adaptation (B) — `core/beliefs.py`, `core/bayes.py`

Particles B_t = {(m^k, w^k, Σ^k)}_{k=1..K}: log w_t^k = log w_{t-1}^k + log p_θ(o_t | Z_{t-1}, a_{t-1}, m^k) using the
world model's outcome head conditioned on each particle (`OutcomeHeadP.dist(feat, outcomes, cond=particles)`);
normalise; learned proposal m^k ← m^k + g_θ(m^k, event token) + Σ^k ε; ESS = 1/Σ w² < `ess_frac`·K →
systematic resampling + rejuvenation noise. Misspecification: running z-score of the mixture's predictive
log-likelihood against its own predicted entropy over `misspec_window`; sustained z > `misspec_z` → inject fresh
particles from the proposal conditioned on recent events and inflate Σ (uncertainty expansion) — logged.
Kalman/NIG last layer (retained from CAM 7.0, generalised): per outcome channel c (an outcome key bound in
`FastState.channels`) and features φ = Φ_θ(outcome features), residual y − head_mean ≈ φᵀμ_c with
K = Σφ/(σ²+φᵀΣφ), μ ← μ + K(y − φᵀμ), Σ ← Σ − KφᵀΣ (Joseph form, float64 for the update). It supplements the
nonlinear dynamics model; it is not the world model.

## 6. World model (C) — `cam_unified/world/`

* `dynamics.py` — Ẑ_{t+1} = F_θ(Z_t, [EncAction(a_t); B_t; EncTime(Δt); NOISE(z_t)]) using the **shared backbone**
  (same weights as the filter; no observations in imagination mode); stochastic latent z_t ~ Cat(32×32) from
  logits(Z_t, a_t) with straight-through gradients; posterior Z_{t+1} from the filter (with O_{t+1}); KL balancing with
  free nats. Latent target Z_target = stopgrad(EMA-encoder(O_{≤t+1})); variance/covariance regulariser against
  collapse. `loss(SequenceBatch)` = λ_z latent + λ_o observation (InfoNCE of predicted vs actual semantic observation
  tokens) + λ_y outcome NLL + λ_r reward + λ_d duration, plus open-loop multistep terms for h ∈ {1,2,4,8,16,32}
  rolled from Z_t with the recorded actions and no future observations; per-horizon errors returned.
* `outcomes.py` — `OutcomeHead`: features(Z, queries) by cross-attention of QUERY tokens onto Z; `dist(feat, cond)`
  with type-conditioned heads: Gaussian / Student-t (continuous), Bernoulli, categorical, negative binomial (counts),
  discrete-hazard survival with censoring. Optional particle conditioning (FiLM by m^k) → batch [B,K,Q].
* `imagination.py` — `Imagination.rollout`: from (Z_t, FastState, G) for n_branches contemplated first actions ×
  n_samples stochastic futures: sample z, predict Ẑ, sample outcomes, **update an independent clone of the
  FastState with the imagined outcomes**, re-run the reasoner, let the policy choose the next action from the
  imagined belief, repeat to horizon h; bootstrap with V at the horizon; uncertainty penalty from particle disagreement
  and outcome variance. Imagined steps are `Provenance.IMAGINED`; the archive rejects them.
* `planner.py` — beam / sampling / policy-rollout planners at matched compute; returns the chosen candidate and the
  traces for `DecisionTrace.imagined`. Candidates come from CAM's own ActionGenerator (`ActionSource.CAM_PLANNER`).
* `causal.py` — intervention provenance features (assignment probability, simultaneous interventions, delays,
  missingness) attached to EVENT tokens; intervention-sensitive contrastive objective (W03).

## 7. Learning and decisions (D) — `cam_unified/learn/`

* `heads.py` — value V(R,G), option policy (16 learned option embeddings, Gaussian/categorical), termination β(R,o),
  exploration budget token, auxiliary heads.
* `ppo.py`, `vtrace.py` — PPO-clip with GAE over decisions for fresh on-policy data; V-trace for asynchronous actor
  data (policy lag from `model_version`); semi-Markov discount γ_Δ = exp(−κ Δt) for options and long actions;
  return normalisation.
* `explore.py` — learning-progress reward LP = log p(Q | B after support) − log p(Q | B before support) with query
  observations disjoint from the support (later real observations scored in hindsight), noise-corrected (epistemic
  part: particle disagreement); meta-exploration: episodes of several trials where early experiments change later
  value; optional math-assist inputs (CAM 7.0 KG/EIG) as a separate condition.
* `meta.py` — support/query adaptation loss over k ∈ {0,1,2,4,8,16,32,64}; J_meta across tasks.
* `trainer.py` — staged combined objective L_total with per-objective gradient-norm contributions into the shared
  backbone logged every update; separate optimiser groups (core / heads / semantic LoRA+bridge).
* `curriculum.py` — learning-progress curriculum over training families × difficulty with diversity floor; never reads
  held-out families.

## 8. Memory and continual learning (E) — `cam_unified/memory/`

* `archive.py` — append-only Parquet chunks (`archive_chunk_rows`) + SQLite index (event id, env, actor, episode, t,
  entity links, chunk, row, model version) + `artifacts.py` content-addressed store; streaming scans; explicit
  `StorageLimitExceeded` at `archive_max_bytes` (never silent dropping); restart-safe (fsync + manifest).
* `index.py` / `retrieval.py` — keys from a learned segment encoder (`mem_segment` events → `mem_segment_tokens`
  latent memory tokens + key); brute-force GPU top-k over memory-mapped key shards per embedding version, with
  temporal/entity filters; query q = Q_θ(Z,G,B); retrieval trained from downstream usefulness (InfoNCE with positives
  = segments whose inclusion lowers world-model loss on the current query) since selection is not differentiable.
  Embedding versions: stale keys are re-encoded in the background or recomputed from raw events; never compared
  across versions.
* `consolidate.py` — hierarchy raw events → segments → longer representations → artifacts → weights, with source
  links.
* `replay.py` — stratified sampler over the archive (recent, uniform historical, domain-balanced, failures,
  innovations, rare/surprising, skills, underrepresented) with a bounded active cache.
* `checkpoint.py` — the seven operations of `CheckpointOpsP`. Weights-only files contain parameters + config only
  (verified by a test that inspects the file).
* `plasticity.py` — dormant units, effective rank, gradient norms/conflicts, update magnitudes; optional mechanisms
  (continual backprop, selective reinit, …) behind switches, each tested independently.

## 9. Distributed learning (F) — `cam_unified/dist/`

Actor processes (independent env, goal, latent state, fast state, seed, versioned weights copy) → durable spool queue
(segment files written then atomically renamed; event-id dedup; backpressure by spool size; recovery of half-written
segments; delayed outcomes as later records with `attributed_to`) → central learner (only it optimises) → atomic
checkpoint publication (`write tmp → fsync → rename`, monotone version) → actors refresh at trajectory boundaries and
rebuild latent state from version-compatible memory. Shared semantic-model serving process for Qwen. Policy lag
tracked per sample; V-trace corrections or discard. `trace.py` / `dashboard.py`: DecisionTrace recorder and the HTML
learning dashboard (Part 18). Compute accounting for every run.

## 10. Environments (G) — `cam_unified/envs/`

* `openworld.py` — `OpenWorldAdapter(EnvAdapter)` and `OpenWorldEvaluator(GoalEvaluator)` over the actual Crucible
  OpenWorld (§11). No mocks.
* `mechworld.py` — `MechWorldAdapter`: Crucible `MechVecEnv` exposed as entities (levers/gauges become entities with
  numeric observations) and tools (`set_<lever>` with a numeric argument, `advance_time`) — numerical worlds are one
  input format, not the architecture.
* `worldgen.py` — diverse procedurally generated mechanism worlds (delays, nonlinearities, feedback, constraints,
  partial observability, hidden state, competitors, regime changes, network effects, failures, dependencies) with
  disjoint structural holdouts and surface randomisation (names, units, ordering, wording).
* `protocols/` — OW-1 … OW-7 task protocols with independent evaluation and feasibility calibration.

## 11. Codebase map (Stage 0)

(filled after the mapping workflow)

## 12. Verification (H)

Acceptance tests `superlearner/tests/cam_unified/acceptance/test_aNN_*.py` for A01–A22, written and run by H against
the merged integration branch; H alone sets `verified_by` in the manifest.
