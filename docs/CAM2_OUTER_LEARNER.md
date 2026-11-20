# CAM 2.0 — outer learner (architecture, learning-rule and search-process discovery)

Code: `src/agentrs/cam2/search/` (`spec.py`, `rules.py`, `population.py`, `controller.py`, `proposer.py`,
`outer.py`). Tests: `tests/cam2/test_search.py`, `tests/cam2/test_protocol.py`,
`tests/cam2/test_learning.py::test_outer_search_end_to_end_with_resume`. Measured results: section 7 below and
`docs/CAM2_EXPERIMENTS.md`.

## 1. What the outer learner is (and is not)

It is an independent loop that **creates candidate CAMs, trains each one with the real inner learner on real
environment interaction, evaluates each on a fixed outer-validation task list with parameter updates disabled,
and selects / promotes / reproduces based on those measurements**. It does not ask an LLM whether an architecture
looks promising: the optional LLM proposer only suggests a spec edit, which then has to earn its place through
the same training and evaluation as every other candidate.

## 2. Candidate = typed, bounded data (`spec.py`)

`{"arch": {...}, "learn": {...}, "plan": {...}}` restricted to `SPACE`:

* **architecture** — integration width `d` (256–768), integration depth (1–6), heads, FF multiplier, recurrent core
  (`gru` / `lstm` / `none`), core width (512–1536), number of state tokens fed back (0–8), episodic-memory top-k
  (0/4/8/16), world-model ensemble size (1–7), width and depth, number and rank of LLM control vectors (0–16 / 64–256),
  head width, Q head, deliberation head, goal-conditioned element encoding;
* **learning algorithm** — learning rate, γ, λ, epochs, entropy, value/Q/world-model/imagination coefficients,
  imagination horizon, retention-anchor coefficient, replay batch, target KL, gradient clip, sampling temperature,
  the **learning-rule expression** (`rules.py`) and the parameters of a **meta-target network**;
* **planning** — enabled, top-K, horizon, κ, β, minimum reliability.

The protocol-fixed settings (episodes per update, minibatch size, episode step limit, evaluation tasks) are not in
the space. Candidates cannot name code, files, tasks or evaluators: the key set is closed and every value is
range-checked.

**Admission checks** (`validate`, run before any budget is spent; failures are recorded with the reason):
schema (keys, types, ranges, cross-field constraints) → learning-rule admissibility → graph construction on the
meta device and parameter-budget check (≤ 120 M) → a 3-step recurrent forward through every head on a synthetic
batch with the real tensor shapes, backward, finite loss, non-zero finite gradient into every parameter group,
measured peak memory (≤ 8 GB) → training itself runs in the normal sandboxed environments.

**Mutation operators** (named, so the controller can learn which pay off): `arch.width`, `arch.depth`,
`arch.recurrence`, `arch.memory`, `arch.world_model`, `arch.llm_interface`, `arch.heads`, `learn.optim`,
`learn.return`, `learn.aux`, `learn.retention`, `learn.entropy`, `rule.expr`, `rule.crossover`,
`rule.meta_target`, `plan`, and (optional) `llm.propose`. Cross-field constraints are repaired deterministically.

**Weight inheritance** (`origin: inherit`): tensors whose name and shape match are copied from the parent's
latest checkpoint (`inherit_weights` reports the inherited fraction); **optimizer state is never inherited**
(Adam moments belong to the parent's shapes and loss landscape; the child gets fresh moments and a fresh LR
warm-up). Inherited and scratch candidates are never mixed within one comparison: all reported searches use
`origin: scratch`, and the champion is re-trained from scratch with independent seeds before any claim.

## 3. Learning-rule discovery (`rules.py`) — what level this is

| Level (task statement §9) | Here |
|---|---|
| A. hyper-parameter search over known algorithms | `learn.*` operators |
| B. compositions of known loss primitives | `rule.expr`, `rule.crossover` over typed expression graphs |
| C. genuinely new update-target functions | the same expression system can express value targets and policy losses that are not PPO / A2C re-parameterisations (`equivalent_to_ppo_family` flags those that are); none has been shown to be better |
| D. a meta-network proposing learning targets | `MetaTargetNet` (a 5→16→1 network mapping per-sample statistics (advantage, TD error, entropy, KL, ratio−1) to a bounded re-weighting of the per-sample policy loss); its parameters are searched by evolution-strategies steps (`rule.meta_target`), not by inner gradients |

Expression leaves: ratio, logratio, logp, logp_old, normalised and raw advantage, λ-return, old value, TD error,
entropy, KL, Q(s,a), reward. Operators: add, sub, mul, guarded div, min, max, neg, abs, square, sqrt, tanh,
sigmoid, relu, guarded exp, signed log1p, stop-gradient, batch standardisation, clip with constant bounds,
constants. **Admissibility** (`check_rule`): size/depth bounds; a gradient path to the policy's log-probability;
finite loss and value target; finite, non-exploding, non-vanishing gradient on three synthetic batches; and a
descent direction positively correlated with the advantage (otherwise it is not a policy-improvement rule).

This is search at levels A–B with the machinery for C–D. **It is not DiscoRL**: DiscoRL meta-learns a large
target-producing network over populations of agents in many environments with far more compute; nothing here
reproduces its setting or claims its results.

## 4. Population, budgets, ASHA (`population.py`)

Each `Candidate` has an immutable id (hash of spec, parent, seed and creation index; never reused), the spec,
parent, the operator and a human-readable description of the change, the full lineage history from the seed,
its training seed, origin, per-rung results (budget, trained environment steps, checkpoint path, the complete
outer-validation result including every episode, compute: wall time, GPU-hours, environment steps, LLM tokens,
parameter count, peak GPU memory, and the training curve), status and failure reason. `population.json` holds all
candidates (successful and failed), the controller's hypothesis history, events and the budget ledger; it is
rewritten atomically after every change. **Resume**: a job interrupted mid-rung is re-run from the candidate's
own last checkpoint; an interrupted promotion is re-armed; the protocol hash and the validation task list are
compared and the search refuses to resume if either changed.

**Multi-fidelity allocation**: asynchronous successive halving. Rung r trains to `b0·ηʳ` environment steps
(continuing the same candidate's checkpoint); a candidate is promotable once it ranks in the top `1/η` of the
candidates completed at its rung (no waiting for full generations). A job that the remaining budget cannot finish
is not started (a truncated rung would be compared unfairly).

**Hard budget** (`Budget`): GPU hours, environment steps (training + evaluation), maximum candidates, maximum
parameters, wall-clock hours, maximum concurrent GPU jobs (1 on this machine). The search stops cleanly at any
limit with every result and checkpoint saved.

## 5. Discovery of discovery: controllers (`controller.py`)

All controllers make the same decisions — explore (new child) vs. promote, which parent, which operator, which
promotable candidate first — so they can be compared at matched budgets.

* **uniform**: uniform parent among evaluated candidates, uniform operator over all operators, ASHA's default
  allocation.
* **pbt** (fixed PBT baseline): truncation selection (parent from the top quartile), hyper-parameter perturbation
  only (`learn.optim`, `learn.entropy`, `learn.return`), fixed operator probabilities. No weight copying, so it
  follows the same scratch protocol.
* **adaptive**: a contextual Thompson-sampling bandit over operators — Bayesian linear regression per operator on
  parent features (rung-0 validation score, log parameter count, lineage depth, budget fraction), rewarded by the
  **measured** outer-validation improvement of the child over its parent at the same budget, divided by the
  child's compute cost relative to the median job (failed admissions get a small negative reward); a 15 %
  exploration floor; diversity-aware Boltzmann parent selection (families that dominate the population and
  parents with many children are down-weighted); and a **learned budget allocation**: once there is evidence of
  both kinds, it promotes only when the empirical gain of the population's best score per GPU-hour from past
  promotions exceeds that of past explorations. Its state is a deterministic function of the hypothesis history,
  so resume reconstructs it exactly.

Every proposal is appended to `population.hypotheses` with its features, operator, description, outcome
(evaluated / failed and why), measured child and parent scores, paired per-task difference with a bootstrap CI,
and cost — the retained history of failed and successful hypotheses.

## 6. Independence of evaluation

* INNER TRAINING tasks (MiniWoB TRAIN families, train seeds) update candidates; OUTER VALIDATION tasks (held-out
  VAL families + val seeds of TRAIN families; 3 per family = 66 tasks in the reported searches) rank them; the
  LOCKED TEST (MiniWoB TEST families, test seeds; AppWorld `test_normal`) is reachable only through
  `tasks.locked_test_tasks` with an acknowledgement string, every access is logged, and
  `tests/cam2/test_protocol.py` asserts by AST inspection that no module of `cam2/search/` references it.
* Evaluation code, success criteria and task lists live outside the search package; the protocol hash pins them.
* Promotion criteria for a champion (`OuterLearner.select_champion`), against the hand-designed seed candidate at
  the same rung: paired improvement on the same validation tasks; no drop > 0.10 on validation instances of the
  training families (catastrophic-forgetting guard); compute ≤ 1.5× unless the improvement's 95 % CI excludes 0;
  integrity (invalid-action rate not more than 0.05 above the seed's, no extra environment failures). A champion
  must then be re-trained from scratch with independent seeds before any superiority claim.

## 7. Results (summary; details and tables in `docs/CAM2_EXPERIMENTS.md` §4)

* Three arms (uniform, PBT, adaptive), 1.0–1.2 GPU-hours and 20–24k environment steps each, 23 candidates evaluated
  in total, every one trained from scratch on MiniWoB and evaluated on the same 66 validation tasks; every
  admission check passed; ASHA promoted 9 candidates to rung 1; all three searches persisted and resumed.
* **No searched candidate beat the hand-designed seed** at matched budget; the seed remained champion in every arm.
  The scratch reproduction with independent seeds (§4.4 there) is the test of the top candidates.
* Measured noise: five independent trainings of the identical seed configuration scored 0.121–0.288 at rung 0
  (SD ≈ 0.06), larger than every unresolved candidate difference.
* Learning-rule search: three non-trivial rules evaluated; one (`tanh(ratio)` in PPO's unclipped branch) clearly
  harmful (Δ −0.21 [−0.32, −0.12]); none better than PPO-clip.
* The adaptive controller did not outperform uniform or PBT (too few proposals for its posterior to matter; its
  parent-relative reward is biased by regression to the mean — see LIMITATIONS §4).
* LLM proposals: never chosen by the bandit; offline they parse after scalar coercion but only copy the best
  previously observed change.
