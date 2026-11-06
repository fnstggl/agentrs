# Architecture

This document describes what is implemented, how the pieces connect, and why the main design
choices were made. Everything described here exists in `src/agentrs/` and is exercised by tests.

## 1. Learning loop at a glance

```
             demonstrations (teacher / consenting humans / IDM-labelled recordings)
                                   │
                                   ▼
  ┌─────────────┐  real transitions  ┌──────────────────┐   gradient updates
  │ Environment │ ─────────────────▶ │ Experience store │ ─────────────────┐
  │ (MiniWeb,   │  (Parquet, real /  │ + replay buffer  │                  │
  │  LocalWeb,  │   imagined /       └──────────────────┘                  ▼
  │  BrowserGym)│   private parts)                       ┌───────────────────────────────┐
  └─────▲───────┘                                        │ policy  π_θ(a | belief, goal) │
        │ typed, allowlisted                             │ world model P_φ(Δs, e, r, d | │
        │ actions                                        │              belief, a)       │
  ┌─────┴──────────────────────┐   proposals / values    │ critic  V_ψ(belief, goal)     │
  │ GuardedEnvironment         │◀── planner (MPC) ◀──────│ inverse dynamics P(a | o, o') │
  │ (permissions, budgets,     │                         └───────────────────────────────┘
  │  timeouts, crash capture)  │
  └────────────────────────────┘
        ▲                                            imagined rollouts (IMAGINED source,
        │ verified outcome (server-side)             never verified, separate partition)
        └── env.verify_outcome()  ← the only source of task success
```

## 2. Environments (`environments/`)

`Environment` (`environments/base.py`) is the only interface agents use:
`reset(task) / observe() / available_actions() / step(action) / verify_outcome() / get_task_metadata() / close()`.
A Gymnasium view is not required by the agent; the interface is browser-independent.

| implementation | what it is | used for |
|---|---|---|
| `MiniWebEnv` (`miniweb/`) | pure-Python simulated web apps, ~µs per step | learning experiments on CPU |
| `LocalWebEnv` (`web/`) | the same apps rendered as real HTML by a local HTTP server, driven by Playwright/Chromium | real-browser validation; parity with MiniWeb verified (0 mismatches over 541 random steps) |
| `PlaywrightEnv` (`web/`) | generic DOM→observation extractor + allowlisted dispatcher for any page | extending to other sites |
| `BrowserGymEnv` | adapter for BrowserGym MiniWoB++ tasks | external benchmark |

**MiniWeb task families** (14). Train families: `click_button, navigate_link, fill_field, login,
checkboxes, dropdown, menu, search_lookup, form_validation, checkout`. Held-out families (only in
split `test_family`): `dialog_confirm, scroll_find, toggle_settings, lookup_then_fill`. Shared,
learnable dynamics: links navigate (new page, new element ids), typing sets values, incomplete
forms raise validation alerts, `form_validation` has a *stochastic* transient "server busy" error,
menus expand, dialogs appear, search results load asynchronously (`wait`), long lists require
scrolling, ids regenerate on navigation (stale references fail).

**Verification.** Each app keeps server-side state written only by its own handlers. The verifier
reads that state and the private task spec. `finish("I succeeded")` does nothing except record an
answer; the agent cannot write verifier state. In `LocalWebEnv`, verification is behind a control
endpoint with a random token that never appears in pages, and the page cannot reach it (the
allowlist blocks it even with the right token).

**Splits / leakage.** `make_task` derives every instance from a split-specific seed range.
Held out for test: instances, the `distractors` layout, the last goal template of each family,
a disjoint label/name vocabulary, and four entire families. The replay buffer and every training
CLI refuse non-`train` transitions (`LeakageError`); episodic memory never stores held-out episodes.

## 3. Actions and security (`actions/`, `security/`)

* `Action` is a typed Pydantic model with a canonical string grammar (`click(e12)`,
  `type(e3, "x")`, …) parsed by a small recursive-descent parser. Model output is never evaluated.
* `GuardedEnvironment` wraps every environment: allowed action types, URL allowlist
  (scheme/host/path), per-episode action and wall-clock budgets, per-step timeout (thread),
  crash capture (exceptions → observable `environment error`), optional rate limit, and a
  human-approval hook for consequential actions (default deny).
* `PermissionPolicy` is a frozen object built from configuration; no code path parses
  observations into permissions (tested with an injected "SYSTEM OVERRIDE" page).
* Skills are expanded into primitive actions that still pass through `env.step`, so they cannot
  bypass the guard (tested).
* Private human demonstrations go to a separate storage partition; `PrivacyGuard` refuses export.
  No code uploads data anywhere.

## 4. Belief state (`agent/belief.py`)

`BeliefState` keeps **observed facts** (current observation, bounded action→outcome history) and
**inferred state** (salient values seen on earlier pages, retrieved memory hints, optional latent)
separately; the prompt marks inferred content explicitly (`INFERRED (not currently visible)`).
Element ids are preserved verbatim so actions remain executable; stale ids are visible in the
history together with their errors.

## 5. Neural components (`models/`)

All components share `LMBackbone`: any Hugging Face causal LM (Pythia, SmolLM2, Qwen3, …) or a
tiny randomly initialised GPT-NeoX for offline tests, with optional LoRA (PEFT). Each component
is a *separate* model instance with its own parameters and optimiser.

### Policy  π_θ(a | belief, goal)
A causal LM restricted to the validated candidate set C(s) (the action mask from
`available_actions`, extended with values from inferred state and optionally with LM-generated
free-form proposals that parse and validate):

    π(a|s) = softmax_{a∈C(s)} ( log p_LM(" " + a + "\n" | prompt(s)) / T )

With T = 1 this is exactly the LM's distribution conditioned on a ∈ C(s), so log-probabilities,
entropy and KL are exact and differentiable — what BC and policy gradients need.

Engineering decisions, each verified by tests:
* **Prefix-shared KV cache** — each prompt is encoded once and its cache replicated per candidate
  (`repeat_interleave`); scores and gradients match the naive implementation (relative gradient
  error ~1e-9 in float64). ~4.5× faster for training.
* **Candidate-vocabulary normalisation** (`vocab_restrict`, default on CPU) — each token
  distribution is normalised over the tokens occurring in that state's candidate set instead of
  the 50k vocabulary. This is a deterministic function of the state (not of the batch), so it
  defines a consistent policy family; it removes the full-vocabulary softmax that dominated CPU
  cost. On GPU (`research` profile) the exact full-vocabulary policy is used.
* **Length normalisation** (`length_norm`, default off) — summed sequence log-likelihood makes an
  *untrained* model put ~99% of its mass on the shortest action (`finish()`), which kills
  exploration for RL from scratch (found by a failing test). Mean-token scoring is available;
  experiments start from BC instead, which removes the bias.

### World model  P_φ(ΔState, events, r, done | belief, action)
Input: goal, last 3 actions/outcomes, current observation, action. Outputs:
1. **Canonical state difference** (`data/diff.py`), generated token by token and terminated by
   `END`; trained with token NLL (a proper likelihood over structured outcomes). `StateDiff.apply`
   turns a sampled diff into an imagined next observation (new elements get imagined ids).
   Sampling multiple diffs represents multiple possible outcomes of one action.
2. **Ensemble heads**: `n_heads` MLPs on the pooled state predict 12 observable event Bernoullis
   (`error, url_changed, reload, elements_added, elements_removed, value_changed, toggled,
   expanded_changed, alert_appeared, dialog_appeared, loading, no_change`) plus P(verified
   reward > 0) and P(terminal). Each head trains on a fixed bootstrap mask; head disagreement
   is the epistemic uncertainty used by the planner and imagination filter.
One backbone forward pass serves both losses. Targets are observable next states and
environment-reported reward/termination — never hidden simulator state.

### Critic  V_ψ(belief, goal)
Sigmoid head on the pooled belief prompt, trained with BCE on λ-returns from real trajectories
(truncated episodes bootstrap with V of the final belief). The critic predicts; the environment
verifies.

### Inverse dynamics  P(a | o, o')
Scores generic candidates (`actions/candidates.py`) given the before-observation and the observed
diff; returns a confidence and abstains below a threshold. Labels carry `Source.INFERRED` and BC
weights them by confidence.

## 6. Planner (`planning/mpc.py`)

Receding-horizon MPC at every real decision: top-K policy candidates → world-model heads (batched)
and sampled next-state diffs (batched generation) → optional deeper branching with policy
proposals on imagined states (`horizon`, `branch_candidates`) → leaves scored by the critic →

    score(a) = p_r + γ (1 − p_term) · E_b[V(ŝ'_b)]  −  λ_u · U(a)  −  cost  +  β · log π(a|s)

with U = ensemble std of reward/terminal + mean event std (+ optional diff NLL). Only the first
action is executed in the real environment; the agent observes the real outcome and replans.
Configurable: enabled, horizon, candidates, branches, uncertainty penalty, action cost, compute
budget (`max_wm_calls`), selection (argmax/softmax), ε. The planner records both
log π(a|s) and the log-probability of its own selection rule (`behavior_logprob`).

## 7. Training (`training/`)

| trainer | objective | notes |
|---|---|---|
| `bc.py` | −w·log π(a_demo\|s) | outcome-weighted (success 1, unknown 0.3, failure 0), invalid actions excluded, inferred labels × confidence |
| `world_model.py` | diff token NLL + bootstrap-masked BCE | trained on *all* real transitions (demos, failures, own experience) |
| `critic.py` | BCE vs λ-returns | truncation bootstrapped |
| `rl.py` | REINFORCE+baseline, PPO (clip, GAE), GRPO | stored prompts/candidates → recomputed log-probs; advantages are constants; truncated IS weights for off-policy data; invalid actions never positively reinforced; entropy bonus; optional KL to a reference policy over candidate sets; PPO KL early stop; critic trained alongside |
| `imagination.py` | PG on short imagined rollouts | starts from real states, horizon ≤ 2–3, cut when epistemic std > threshold, `Source.IMAGINED`, down-weighted, exploitation monitor disables it when imagined success exceeds real success by a margin |
| `idm.py` | CE over candidates | evaluation reports selective accuracy and abstention |
| `common.py` | — | AdamW + warmup/cosine, grad clipping, NaN/Inf skip + divergence error, checkpoint (safetensors weights + optimizer + scheduler + RNG), exact resume (tested) |

**Planner data is off-policy.** When the planner changes the selection distribution, its actions
are not treated as samples from π. Two valid options are implemented: truncated importance
weights against the recorded behaviour probability, and search distillation (cross-entropy
towards planner choices). In the main experiments, RL data is collected on-policy and the planner
is used at decision time; a first pilot showed PPO ratios against a sharp planner distribution
are far from 1 (approx. KL ≈ 7.6, immediate early stop), confirming the need for this separation.

## 8. Continual learning (`training/continual.py`, `checkpoints/registry.py`)

* **Actor** process: loads the *promoted* policy checkpoint (fixed version id recorded in every
  transition), collects curriculum tasks, writes atomic Parquet shards to `inbox/`.
* **Learner**: ingests unseen shards (state persisted), historical replay for the world model,
  PPO with importance ratios against the actor's recorded behaviour log-probs, writes candidate
  checkpoints to a SQLite registry.
* **Evaluator** process: evaluates candidate and production on held-out validation tasks plus a
  regression suite of mastered families; `PromotionGate` (no aggregate drop, no per-family
  regression beyond a tolerance, bounded invalid-action rate) decides promote/reject; learner
  resets to production on rejection; `rollback` restores the previous production checkpoint.
* **Curriculum**: stages by difficulty; advances only when verified success on the frontier stage
  exceeds a threshold over a window, with replay of mastered stages.
* Forgetting is tracked per family across cycles.

## 9. Memory and skills (`memory/`)

* `EpisodicMemory`: SQLite + hashed bag-of-words embedding of the goal *template* (quoted values
  removed); stores id-free action summaries and errors; retrieval returns hints that enter the
  belief's inferred section. It is context, not learned capability.
* `Skill` / `SkillLibrary` / `SkillExecutor`: options with preconditions, routine, termination,
  success criteria, environment compatibility and a Beta-posterior reliability updated from
  verified outcomes; hierarchical choice of `skill(name)` as a macro-action.
  `discover_skills` mines frequent id-free action patterns from successful trajectories.

## 10. Experience data (`data/`)

`Transition` (schema v2) records every field required by the brief: ids, env/version, goal,
observation, belief prompt, action + candidates + chosen index, policy and behaviour log-probs,
selection rule, world-model prediction made *before* acting, next observation, canonical diff,
events, validity/error, reward, verified progress, termination vs truncation and reason,
episode success (filled from the verifier), checkpoint ids, timestamp, provenance (source,
consent, license, privacy), verification evidence, inferred-label confidence. A validator forbids
imagined transitions from carrying verified outcomes and human transitions without consent.
Storage is Parquet with flat filter columns + the full JSON record (lossless), atomic shard
writes, partitions `real/ imagined/ private/`, content-addressed blobs, and a migration path
(v1→v2 tested).

## 11. Relation to prior work

Nothing here is a new algorithmic idea in isolation:
action-conditioned world models + planning (MuZero, PlaNet/Dreamer/DreamerV3), learning from
imagined rollouts (Dyna, MBPO's short-rollout trick, Dreamer's actor-critic in imagination),
uncertainty via bootstrap ensembles (PETS, MOPO-style penalties), imitation + inverse dynamics
for unlabelled demonstrations (BCO, VPT), options/skills (Sutton et al.), and world models for web
agents (WebDreamer, WebEvolver, DynaWeb, WMA). The standard parts are BC, PPO/GRPO/REINFORCE,
GAE, bootstrap ensembles, MPC. The experimental parts are the specific combination for
text-structured web observations: canonical state diffs as the generative world-model target,
candidate-restricted LM policies with exact log-probs on CPU, and the controlled comparison
protocol. Updating weights from experience is continual learning; this system does not improve
its own learning algorithm.
