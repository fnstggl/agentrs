# CAM 2.0 — architecture

Branch `feat/cam-2`, cut from `origin/feat/superlearner` (which contains `feat/cam-hybrid`, `feat/cam-v2`,
`feat/cam`, and the original platform). Package: `src/agentrs/cam2/`. Nothing under `src/agentrs/sl/`,
`src/agentrs/cam/` or any earlier run directory is modified; the pure-CAM superlearner stays the B0 baseline.

Part 1 is the audit written **before** implementation (2026-11-20). Part 2 describes what was built.

---------------------------------------------------------------------------------------------------------

## Part 1 — audit of the existing system (before CAM 2.0)

### 1.1 What exists

| Line of work | Branch / package | Architecture | Trainable params (default config, measured) | LLM use |
|---|---|---|---|---|
| Original platform | `src/agentrs/models`, `training`, `planning` | HF causal LM policy (Pythia / Qwen) with **LoRA**, world model, critic, MPC planner | LoRA adapters + heads | LLM is the policy, fine-tuned with LoRA |
| CAM v1 | `feat/cam`, `src/agentrs/cam/{model,learner,train}.py` | token transformers + GRU belief + MLP dynamics + options + GRU action decoder; PPO+GAE | **12.82 M** | none |
| CAM-v2 / v2.1 / v2.2 | `feat/cam-v2`, `src/agentrs/cam/v2/` | RSSM world model (GRU + 16×16 categorical latents), twin-Q/expectile V, IQL/AWR/SAC, Go-Explore archive, dream learner | **25.02 M** | none |
| CAM-hybrid | `feat/cam-hybrid`, `src/agentrs/cam/hybrid/` | CAM-v2 + frozen Qwen3 (0.6B/4B) mean-pooled hidden states → trainable bridge | CAM-v2 + bridge | frozen encoder only (mechanism 1). CAM never conditioned LLM *generation* ("mechanism 2 not built") |
| Superlearner (pure CAM) | `feat/superlearner`, `src/agentrs/sl/` | hashed-symbol set-transformer perception, GRU belief, per-trial episodic KV memory, 8 skills, pointer action head over hashed labels, ensemble world model; recurrent PPO (RL²-style segments) | **25.52 M** | none |

### 1.2 Observation and action interfaces

* **CAM v1/v2**: environment-specific token grammars (`cam/envs/*`): the agent emits complete actions
  token by token under a grammar mask (`ASK price`, `PROPOSE N3 N5 ...`). The vocabulary is the union of the
  three simulators' grammars; a new environment needs new tokens.
* **Superlearner**: `sl/world.py` `World` interface (`observe`, `available_actions`, `execute`,
  `evaluate_goal`, `save/restore_world_state`). Observations are hashed symbols / named numbers / entity sets /
  images; actions are labelled discrete choices or a slot grammar, scored by a pointer head — the most general
  interface in the repository, but still a closed set of labels enumerated by each world.
* **Hybrid**: natural-language renderings of the CAM simulators' tokens (information-equivalent), encoded
  by the frozen LM; actions still CAM grammar tokens.
* **Original platform**: `environments/base.py` typed `Action` schema (click/type/select/…) over MiniWeb,
  a Playwright web environment and a BrowserGym adapter — composable primitives, but driven by the LoRA-tuned
  LM, not by CAM.

### 1.3 Environments available before CAM 2.0

Negotiation, product and bizdev simulators (hand-built, hidden parameters); Vending-Bench; Melting Pot
(separate TF venv); YC-Bench driver; MiniWeb and SocialWebLab (own simulators); BrowserGym/MiniWoB++ adapter
(original platform only). No AppWorld, WebArena, TheAgentCompany, OSWorld or Gym-Anything integration.
CAM v1/v2/SL act only in the business simulators and Melting Pot — i.e. **"choose among a fixed set of
strategy tokens"** worlds.

### 1.4 Training limitations found

1. **No CAM-controlled language generation.** The hybrid's frozen LM only produced features; the LoRA line
   adapted the LM's own weights. Nothing trained a separate network to steer a frozen LM's generation.
2. **Closed action vocabularies.** CAM grammars are fixed per simulator; adding an environment means new
   tokens and a new output layer (the SL pointer head fixed the output layer but still needs enumerated labels).
3. **Environment realism.** All CAM learning results are on hand-built simulators whose success says nothing
   about real software. The docs say so explicitly (`docs/cam.md` "Honest limitations").
4. **Reward-contract fragility.** CAM v1, v2 and SL each had at least one degenerate optimum (early
   termination, stalling, holding progress costs) found after runs started (`docs/superlearner.md`
   amendments 1–3). Lesson carried into CAM 2.0: use the benchmarks' own state-based verifiers as reward,
   avoid hand shaping.
5. **Planning was evaluation-only** in CAM v1; CAM-v2's dream learner existed but did not beat AWR at matched
   budget; no learned deliberation-allocation decision anywhere.
6. **Continual learning** across families exists in SL (lifetimes, retention monitoring, anchors) — the
   best-developed part of the codebase; replay + KL anchoring (v18) are reusable ideas.

### 1.5 Architecture search

**None.** `grep` for population/mutation/NAS finds only unrelated uses. Every architecture change in the
repository's history was a hand-written protocol amendment. There is no learning-rule search and no
meta-controller over experiments.

### 1.6 Reuse decision

| Reused (idea or code) | From | Why |
|---|---|---|
| Frozen-LM loading, hook-based layer capture, mask-mean pooling, weight fingerprint | `cam/hybrid/semantic.py` | sound; re-implemented inside `cam2/backbone.py` with differentiable prefix conditioning added |
| Pointer-over-candidates action head | `sl/model.py` | the right way to get open action sets without output layers; generalised to typed tool schemas |
| Recurrent PPO with stored segment-start states and exact log-prob replay check | `sl/learner.py` | correct; re-implemented for the new action factorisation (structured candidate + LLM text tokens) |
| Ensemble-disagreement world-model uncertainty, latent next-state targets | `sl/model.py`, `cam/v2` | reused as the CAM 2.0 world-model design |
| Network guard for BrowserGym (block non-`file://` requests) | `environments/browsergym_adapter.py` | safety: MiniWoB pages must not reach the internet |
| Replaced | — | fixed grammars, hand-built simulators as primary environments, LoRA LM policy |

The B0 baseline is the superlearner code as it stands on this branch (`python -m pytest tests/test_superlearner.py`).

---------------------------------------------------------------------------------------------------------

## Part 2 — what was built

Status labels used throughout the CAM 2.0 docs: **IMPLEMENTED** (code exists), **TESTED** (unit/integration test
passes), **EXPERIMENTAL** (ran in a real experiment, evidence weak or mixed), **NOT IMPLEMENTED**.

### 2.1 Module map (`src/agentrs/cam2/`)

| Concern | Module | Status |
|---|---|---|
| Frozen backbone (Qwen3, hidden-state features, CAM-controlled generation, exact log-probs) | `backbone.py` | TESTED (incl. real Qwen3-0.6B gradient test, real Qwen3-1.7B runs) |
| CAM network (integration transformer, recurrent core, memory, policy / value / Q / deliberation heads, LLM control vectors, world-model ensemble) | `model.py` | TESTED |
| Observation → features → batch | `featurize.py` | TESTED |
| Tool / action schemas, candidate enumeration, schema validation, external-effect gate | `tools.py` | TESTED |
| Acting (structured candidate + frozen-LLM text under CAM control) | `agent.py` | TESTED |
| Model-based planning (imagined rollouts, uncertainty fallback) | `planner.py` | TESTED (plumbing); EXPERIMENTAL (effect) |
| Inner learner: recurrent PPO with pluggable learning rule, world model, replay anchor, imagination | `learner.py` | TESTED |
| Rollout workers, training loop, evaluation without updates, checkpoints | `envs/pool.py`, `trainer.py` | TESTED |
| Environment adapters | `envs/{base,miniwob,appworld,appworld_worker,unavailable,fixture}.py` | MiniWoB + AppWorld TESTED end to end |
| Task splits, locked test, protocol hash | `tasks.py` | TESTED |
| Learning-rule expressions, mutation, admissibility checks, meta-target network | `search/rules.py` | TESTED |
| Typed candidate specs, admission checks, mutation operators, weight inheritance | `search/spec.py` | TESTED |
| Population, lineage, budgets, ASHA | `search/population.py` | TESTED |
| Search controllers (uniform / PBT / adaptive bandit) | `search/controller.py` | TESTED; EXPERIMENTAL (comparison) |
| LLM-guided proposals | `search/proposer.py` | TESTED (bounds); EXPERIMENTAL |
| Outer learner driver | `search/outer.py` | TESTED (fixture end to end); EXPERIMENTAL (MiniWoB) |
| B1 LLM-only baseline with identical tool access | `baselines.py` | TESTED |

Scripts: `scripts/cam2/train_eval.py` (train + reload verification + evaluation), `scripts/cam2/evaluate.py`
(checkpoint / B1 / planning / locked-test evaluation), `scripts/cam2/retention.py`, `scripts/cam2/report.py`.

### 2.2 Frozen LLM

* Default **Qwen/Qwen3-1.7B** (snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, 1,720,574,976 parameters with
  tied embeddings, bf16 = 3.4 GB, 28 layers, hidden 2048), chosen for the A6000 48 GB with room for a 65 M fp32 CAM, 16 browser workers and a
  second concurrent learner process. Qwen3-0.6B (tests) and Qwen3-4B are also cached; `llm:` selects any of them.
* Every parameter has `requires_grad=False`, the module is in eval mode, and the SHA-256 of all weights
  (`fingerprint()`) is recorded at the start and checked at the end of every run (asserted unchanged in every run
  reported). No LoRA / adapters / prompt tuning of the LLM's weights exist anywhere in `cam2/`.
* **Mechanism A — semantic features.** Forward hooks capture layers 14 and 28; texts are mask-mean pooled
  (4096-d), goals keep per-token states of layer 14. Cached by exact text (the frozen LLM makes cached features
  exact). Optional goal-conditioned element encoding (`el_goal_cond`: each element encoded after the goal, pooled
  over the element's own tokens) — a searchable architecture choice.
* **Mechanism B — CAM-controlled generation.** CAM emits `P` (default 8) continuous control vectors in the LLM's
  input-embedding space from its state and the chosen action candidate (`CAM.prefix`, a low-rank map initialised
  near real token embeddings). They are spliced between the context and the answer stub
  (`[ctx][P CAM vectors][<think></think>][generated tokens]`). The text part of an action is sampled from the
  frozen LLM's temperature-1 softmax.
* **Gradient pathway (explicit).** `logprob_with_prefix` recomputes `log p(tokens | ctx, prefix)` with autograd
  enabled: the gradient flows through the frozen LLM's computation into the control vectors and from there into
  CAM's parameters; the LLM's own parameters receive none. The text tokens are therefore an ordinary stochastic
  policy output and enter PPO's likelihood ratio together with the structured-candidate log-probability:
  `log π(a|s) = log π_delib + log π_cand(k | f, c) + Σ_t log p_LLM(y_t | ctx, prefix(f, c_k), y_<t)`.
  Log-probabilities are computed row by row without padding, because in bf16 a padded batched forward depends on
  its batch-mates (measured: up to 0.22 nats); row-wise evaluation is batch-invariant (measured difference 0.0),
  so the behaviour log-probability stored at collection is reproduced by the learner (logged as
  `replay_mismatch`). Measured in the main MiniWoB run: candidate part ≤ 3e-6; text part mean 8e-4 nats with rare
  outliers up to 0.37 nats, caused by fp32 noise in the *recomputed* control vectors flipping bf16 roundings.
  `LearnConfig.prefix_replay_correction` removes this (stores the bf16 control vectors at collection and shifts the
  learner's recomputation onto them; verified exact on real Qwen3-0.6B), but it was added after the reported runs
  started and is off in all of them.
* The text-log-prob gradient is computed with a memory-bounded exact surrogate (`Learner.text_surrogate`): the loss
  is formed with each row's text log-prob as a leaf, `∂loss/∂logp_j` is taken, and the LLM forward/backward is
  recomputed in chunks of `text_chunk` rows to obtain `∂loss/∂prefix_j`; tested equal to full backpropagation.
  (Holding one graph through the LLM for every text row of a minibatch ran out of memory on AppWorld, where every
  step writes JSON.)
* Tests (`tests/cam2/test_gradients.py`): LLM parameters get no gradient; CAM's policy, integration and control
  parameters get non-zero gradients through the LLM; an optimiser step changes CAM parameters, raises the
  probability of the reinforced candidate and raises the LLM's probability of the reinforced CAM-conditioned text;
  the control vectors change greedy generation; sampled log-probs equal the recomputation. Repeated on the real
  pretrained Qwen3-0.6B on GPU (fingerprint unchanged).

### 2.3 CAM network (default configuration, measured 65,242,894 trainable fp32 parameters)

| Component | Parameters | Design |
|---|---|---|
| Semantic integration | 20.3 M | projections of goal tokens, observation elements (+ role/depth/position features), feedback text, previous action + reward/step scalars, retrieved memories, recurrent state tokens; 4-layer pre-norm transformer, d = 512, 8 heads |
| Recurrent core | 8.4 M | GRUCell (or LSTM / feed-forward, searchable), 1024-d persistent state, reset per episode (optionally carried across a stream), truncated BPTT over whole episodes from stored segment-start states |
| Episodic memory | 1.1 M | persistent ring buffer (50k rows) of (core state, return-to-go, success, time); learned query/key retrieval of top-k = 8 rows per decision; retrieved rows stored with the transition for exact replay; checkpointed |
| Policy / value / Q / deliberation | 8.2 M | pointer network over schema-enumerated (tool, element, enum) candidates; state value V(f); action value Q(f, c); optional act-vs-plan deliberation head |
| LLM control | 3.7 M | 8 control vectors (rank 256) for the frozen LLM's generation |
| World model | 23.6 M | 5-member ensemble MLP: next core features, next integration summary, reward, termination; ensemble disagreement = epistemic uncertainty |

Variants searched by the outer learner range from ~35 M to the 120 M cap (`search/spec.py`). Measured peak GPU
memory (`torch.cuda.max_memory_allocated`) of the main MiniWoB training process — Qwen3-1.7B bf16 (3.4 GB) + 65 M-parameter CAM in fp32 + AdamW state + activations, including backpropagation through the frozen
LLM for text log-probs: **15.7 GB**; the AppWorld learner (457 candidates per step, JSON text at every step)
peaked at **21 GB** allocated. Inference-only evaluation needs ≈ 6 GB.

### 2.4 Action system

Environments publish tool schemas at run time; CAM chooses among the candidates the schemas and observation admit
(the action mask), the frozen LLM writes free-text / JSON arguments under CAM's control vectors, and every call is
validated (JSON-schema subset: object, required keys, known properties, primitive types, enums) before it reaches
the environment. Multi-step composition, information gathering (`read` tools, AppWorld `show_*` / `api_docs.*`),
termination (`supervisor.complete_task`, or the episode limit) and deliberation (`deliberation` head deciding
whether to invoke the planner, charged a small cost) are CAM decisions. No domain strategy is encoded.

### 2.5 Inner learner (`learner.py`)

Recurrent PPO over whole episodes (segments = episodes, BPTT from the stored initial state), with:
GAE(λ) with terminal (no bootstrap) vs truncation (bootstrap with V of the final observation) handling; advantage
normalisation; value and Q regression; entropy bonus; gradient clipping; linear LR warm-up; early stop on approx.
KL; exact behaviour-log-prob replay check; `params_changed` / `update_rel_norm` measured every update and an
assertion that optimiser steps change CAM parameters. The policy loss and value target are a **learning-rule
expression** (`search/rules.py`; PPO-clip by default) so the outer learner can search them. Replay (reservoir over
all past episodes, travels with checkpoints) is used only for world-model training and an optional CLEAR-style
behaviour-KL / value anchor on old states — never for an unweighted off-policy policy gradient. Imagination
training (actor-critic on world-model rollouts) is gated by the world model's measured held-out reliability.
Evaluation runs with `train=False`: no buffer or memory writes, no optimiser, CAM checksum asserted unchanged.

### 2.6 World model and planning

Targets are the next step's (stop-gradient) core features and integration summary, reward and termination. Every
update first measures the world model on the not-yet-trained-on batch: R² of next features and reward,
termination accuracy, calibration of ensemble disagreement against actual one-step return error, and the
largest disagreement level whose error stays below 0.25 (`u_max`). The planner imagines H-step rollouts for the
top-K candidates per ensemble member, turns mean imagined return minus β·disagreement into a logit bias, stores
the bias with the transition (exact PPO ratios), and falls back to the direct policy when measured reliability is
below `min_reliability` or disagreement exceeds `u_max`. Imagined rewards are never logged as outcomes.

### 2.7 Outer learner

See `docs/CAM2_OUTER_LEARNER.md`.

### 2.8 Not implemented

* Temporal abstraction / options / sub-goal policies (CAM 2.0 is flat; long-horizon structure relies on the
  recurrent core and episodic memory).
* Cross-attention from the frozen LLM into CAM, or CAM-written KV-cache entries (conditioning is prefix vectors
  only).
* V-trace / IMPALA off-policy correction (replay is not used for policy gradients; recurrent PPO on fresh episodes).
* Distilled long-term memory (the episodic store is non-parametric; consolidation into weights happens only through
  ordinary RL updates and the replay anchor).
* A neural meta-policy for the search controller (the adaptive controller is a Bayesian contextual bandit).
* Concurrent candidate training inside one search (`max_concurrent_gpu_jobs` is recorded but jobs run sequentially).
* Adapters for WebArena, TheAgentCompany, OSWorld, Gym-Anything, WebGym (blocked here; each raises
  `DependencyUnavailable` with the setup procedure — `docs/CAM2_ENVIRONMENTS.md`).
* A deployment-time online-adaptation benchmark.
