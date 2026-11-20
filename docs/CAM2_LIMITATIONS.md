# CAM 2.0 — limitations, threats to validity, and what is not done

Read with `docs/CAM2_EXPERIMENTS.md`. Everything below was either measured or is a direct consequence of how the
code works; nothing here is speculative about results that were not run.

## 1. Scale of the evidence

* **Compute.** One RTX A6000 (48 GB), one session of a few hours. Main CAM runs train for 12,000 environment
  steps (~25 min); search candidates for 1,200–3,600 steps each. These are smoke-to-pilot budgets, far below what
  RL on web tasks normally needs; learning curves had not plateaued.
* **Seeds.** The main run, the ablations and each search arm use one seed. Only the champion reproduction uses
  independent seeds (two). Differences of a few points between single-seed runs are within noise.
* **Evaluation size.** Outer validation uses 66–88 greedy episodes; 95 % bootstrap CIs are about ±0.10 wide.
  Paired per-task comparisons (same tasks, same seeds) are reported where they matter, but most candidate-vs-
  candidate differences in the searches are not statistically resolved.
* **Locked test.** Run once at the end on 6 held-out MiniWoB families (and never fed back). It is small.

* **Headline negative results.** After 12k environment steps the trained CAM (0.432) is not better than the frozen
  LLM prompted with the same tools on validation (0.500, difference unresolved) and is worse on the locked test of
  six unseen families (0.083 vs 0.208, Δ −0.125 [−0.25, −0.02]); the world-model auxiliary loss did not help;
  decision-time planning changed no decision; no searched architecture or learning rule beat the hand-designed
  configuration; the adaptive search controller did not beat uniform mutation or PBT; LLM-guided proposals only
  imitated the best previous change. See `docs/CAM2_EXPERIMENTS.md`.
* **Evaluation noise is larger than most effects.** Five independent 1.2k-step trainings of the same configuration
  scored 0.121–0.288 (SD ≈ 0.06) on the 66-task validation set; greedy evaluation of short-trained policies can move
  by 0.25 between rungs while the sampled policy improves. Single-seed comparisons below ~0.15 are not meaningful.
* **Features depend on batch composition.** The frozen LLM's bf16 features differ slightly with the composition of
  the encoding batch, which depends on the number of workers; the untrained CAM scored 0.068 with 16 workers and
  0.057 with 14 (one task of 88). Evaluations are exactly reproducible only at a fixed worker count.

## 2. Environments

* Only **MiniWoB++** (via BrowserGym, real Chromium) carries the learning results. Its tasks are short (≤ 10
  actions) single-page web interactions — open-ended in action composition (generic primitives over arbitrary
  page elements, free text from the LLM) but not long-horizon.
* **AppWorld** is integrated end to end with its official evaluator and was trained at smoke scale only; success
  stayed at 0 (see experiments). A 1.7 B frozen LLM writing JSON arguments for one of 457 APIs per step, chosen by a
  pointer network that starts uniform, is far from the regime where AppWorld tasks get solved.
* **WebArena, TheAgentCompany, OSWorld, Gym-Anything** are blocked in this container (no Docker daemon, no KVM);
  **WebGym** is excluded by the safety policy (live web) and has a non-deterministic LLM-judge evaluator. Each
  raises `DependencyUnavailable` with the setup procedure. Transfer to "unseen task families" is therefore measured
  only within MiniWoB (6 held-out validation families, 6 locked test families).
* Throughput is limited by the container's `pids.max = 2816` (≈ 78 threads per browser worker): at most one
  16-worker MiniWoB job, or two 10–14-worker jobs, at a time.

## 3. Model and learning

* **Frozen LLM size.** Qwen3-1.7B was chosen for throughput; Qwen3-4B is available but was not used in experiments.
* **Text generation under CAM control** matters only for actions with free-text / JSON arguments (MiniWoB `fill`,
  every AppWorld API call). On MiniWoB most decisions are structured (pointer over candidates), so most of the
  learned behaviour lives in CAM's pointer/recurrent network, not in the LLM-control vectors.
* **Exactness of text log-probabilities.** Candidate log-probabilities replay to ≤ 3e-6. Text log-probabilities
  were made batch-invariant (row-wise evaluation), but the learner's *recomputed* control vectors differ from the
  collection-time ones by fp32 noise that occasionally flips a bf16 rounding; in the reported MiniWoB runs this left
  a mean replay mismatch of ~8e-4 nats with rare outliers up to ~0.37 nats, and ~0.015 nats mean / 0.82 max on
  AppWorld's long JSON arguments. The PPO clip absorbs this, but it is a real deviation from exact importance
  ratios. A fix (`LearnConfig.prefix_replay_correction`: store the bf16 control vectors, shift the recomputation
  onto them) is implemented and verified exact on real Qwen3-0.6B, but was implemented after the reported runs
  started and is off in all of them (kept off so the controller-comparison arms ran identical code).
* **World model.** Measured one-step reliability stayed low (next-feature R² negative, reward R² near 0 on MiniWoB's
  sparse terminal rewards). Consequently the planner's reliability gate falls back to the direct policy and
  imagination training never activates in normal runs; forced planning is evaluated separately as an ablation. The
  world model helps (or not) only as an auxiliary representation loss — see the B3 vs B4 ablation.
* **Temporal abstraction / sub-goals** (options, hierarchical policies) are not implemented in CAM 2.0; long-horizon
  structure relies on the recurrent core and episodic memory only.
* **Episodic memory** retrieval is a brute-force top-k over up to 50k rows per decision; fine at this scale, not at
  millions of rows. Its contribution was not ablated separately.
* **Continual learning** is demonstrated within training processes and across restarts (checkpoint = weights +
  optimiser + replay + episodic memory + RNG + tool registry; resume verified exact). There is no deployment-time
  online-adaptation benchmark.

## 4. Outer learner

* **Search space.** Bounded and hand-designed (typed spec over existing modules, learning-rule expressions over a
  fixed primitive set, a tiny meta-target network). It cannot invent new module types; this is deliberate (no
  execution of generated code) and limits "architecture discovery" to configuration and loss-structure search.
* **Learning-rule discovery** is at levels A–B (hyper-parameters, compositions of known primitives) with the
  mechanism for C–D (expressions that are not PPO re-parameterisations; a meta-network re-weighting targets). No
  discovered rule has been shown to beat PPO-clip; nothing here reproduces or approaches DiscoRL's setting.
* **Search budgets.** 8 candidates per arm, 1,200 steps at rung 0 and 3,600 at rung 1. Rung-0 rankings at this
  budget are noisy; ASHA can promote lucky candidates and the scratch reproduction is the safeguard.
* **Controller reward is biased.** The adaptive controller is rewarded by child-minus-parent validation success.
  With single noisy parent measurements this rewards regression to the mean (children of an unlucky 0.061 parent
  showed "significant" +0.14…+0.21 gains with ordinary absolute scores). A sound reward would compare against
  repeated measurements of a reference (e.g. the seed re-evaluated per rung) or use absolute scores with a noise
  model. Not fixed.
* **Controller comparison.** One search per controller (one seed of the search itself), so "adaptive vs uniform vs
  PBT" is a single paired observation per arm, not a statistically powered comparison. With 7–8 proposals the bandit
  had no evidence to exploit: it tried 7 different operators once each. The adaptive arm also had the
  `llm.propose` operator available and the others did not, which confounds the comparison; per-operator outcomes are
  reported so the LLM proposals can be judged separately.
* **Weight inheritance** is implemented and tested but not used in reported searches (all scratch).
* **Concurrency.** `max_concurrent_gpu_jobs` is recorded but the driver runs candidates sequentially; three
  searches ran as separate processes on one GPU, which caused one candidate to crash with CUDA OOM (caused by a
  concurrent AppWorld learner; recorded as an infrastructure failure and excluded from statistics).
* **Repairs during the experiments** (all recorded as `repair` events in the populations): two adaptive-arm
  promotions failed on a checkpoint-resume bug (CPU RNG state restored from a CUDA tensor; fixed in
  `Trainer.load`, GPU resume test added) and were re-armed; the uniform arm's OOM-crashed candidate got its lost
  2.3 minutes back but the re-run did not fit the remaining budget. In the adaptive arm the seed was therefore never
  evaluated at rung 1; the driver now always promotes the seed as a reference.

## 5. Baselines

* **B0** (pure-CAM superlearner) is preserved unchanged and its test suite passes, but it acts in its own `World`
  interface (business simulators, Melting Pot) and cannot be run on MiniWoB/AppWorld without a new adapter; no
  B0-vs-CAM 2.0 number on a shared environment exists.
* **B1** (frozen LLM alone with identical tools and observations) uses one prompt format and greedy decoding; a
  stronger prompting/agent scaffold (few-shot, ReAct, retries) could do better. Even this simple B1 matches or beats
  the trained CAM on MiniWoB (validation and locked test), at ~40–50× the LLM tokens per evaluation. On AppWorld it
  fails like CAM (0/24, 96 % invalid calls).
* **B2** is the untrained CAM (random initialisation) with the frozen LLM — a floor, not a competitive baseline.

## 6. Reproducibility

Seeds, configs, LLM revision, package versions and git commit are written to every run's `config.json`;
checkpoints reload exactly. Runs are still not bit-reproducible: browser timing changes the order in which
episodes complete (and therefore batch composition), and bf16 kernels are batch-dependent. Greedy evaluations on
fixed seeds are deterministic for a fixed checkpoint (the untrained-CAM evaluation reproduced exactly across two
launches).

## 7. Next highest-value bottleneck

See the end of `docs/CAM2_EXPERIMENTS.md`.
