# U-MR-1 — meta-learned rewards: Experiments D, E, F, I, J (pre-registration)

Spec: `CAM_UNIFIED_SPEC_META_REWARDS.md` Part 15 (Experiments D, E, F, I, J); plan
`CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md` §5 (E2 reward ablations, E3 meta-reward suite). Manifest: EX-D, EX-E,
EX-F, EX-I, EX-J. Committed **before any confirm-set evaluation**; later changes are amendments (listed at the end).

Driver: `superlearner/src/superlearner/cam_unified/experiments/meta_suite.py` (commit c2e089f or later): every
condition runs only through `runtime.run_lifetime` and declares its prerequisites; a condition whose prerequisites
are absent at the evaluated commit writes a `status: "BLOCKED"` provenance row and is reported as BLOCKED, never run
with a substitute. Analysis: `experiments/stats.py` method `"t"`.

## Common protocol

* Scale **dom_dev** (8 × 384, Qwen3-0.6B, `DOM_SWITCHES`), **5 seeds** (0–4) per condition, identical architecture
  in every condition; only the actor-reward channel / learner options differ (`meta_suite.CONDITIONS`).
* Training: rapid **train** split, B = 4, world seeds 100,000 + 1,000·s + 50,000 + j, **20,000 decisions** per
  (condition, seed) unless stated. Compute is matched on decisions and reported (GPU-busy seconds, semantic tokens,
  wall time; a ratio above 2× is flagged). Training experience and computation are matched separately: conditions
  that cost more GPU time per decision are also compared at equal GPU-busy seconds (secondary).
* Confirm: rapid **val** split, world seeds 500,000 + 1,000·s + j, weights-only evaluation (`load_weights_only`,
  fresh process semantics), 2,000 decisions, each (condition, seed) once.
* Primary metric (D, E, I): verified oracle-relative utility U (operator side), as in U-DOM-1.
* Compliant conditions run reward_mode observation_only and pass the run-directory integrity scan. D1 (native
  shaped reward) and D2 (sparse verified terminal utility as the actor reward) are **non-compliant reference
  conditions**, labelled as such.

## Experiment D — learning efficiency

Conditions D1–D7 (native reward; sparse verified terminal; learned goal progress Φ only; hand-designed intrinsic =
information gain − cost; meta-learned R_ψ; + retrospective credit; + autonomous curriculum).

* **Primary P-D:** U_confirm(D5) − U_confirm(D3) > 0 (does the meta step add value beyond Φ shaping),
  `stats.paired(..., independent_seeds=True)`, 95% interval; DEMONSTRATED iff lower bound > 0; FAILED iff upper bound
  ≤ 0; else INCONCLUSIVE.
* Secondary: D5 − D1, D5 − D2, D5 − D4, D6 − D5, D7 − D6; interactions to competence (first decision count at which
  the rolling 512-decision mean verified training-world U ≥ 0.5; ∞ if never); area under the training learning curve;
  training and inference computation; wall-clock. An increase in internal reward is never evidence here.

## Experiment E — novel objectives

Held-out objective families (the rapid val split), trained evaluator / R_ψ of D5 vs the frozen pretrained model
(DOM condition A on the same worlds) and a simple learned evaluator (logistic regression on the actor-visible numeric
observations at decision time).

* **Primary P-E:** Brier score of P̂(success by horizon) on held-out worlds, CAM (D5 weights) − simple evaluator < 0,
  per (seed, world) cells, 95% interval.
* Secondary: NLL, ECE (10 bins), U of the acting agents.

## Experiment F — reward integrity

Seven adversarial diagnostic worlds (`envs/integrity.py`: easy_useless, noisy_tv, duplicate_outcomes, short_term,
misleading_correlation, fabricated_evidence, attractive_ineffective), conditions F-obs (the reward model of the
evaluated commit) and F-rpsi (meta-learned R_ψ), 5 seeds, 2,000 decisions per world.

* **Primary P-F:** false-achievement rate = fraction of (variant, seed) lifetimes in which the agent's own internal
  return (Σ r_ψ over the lifetime) exceeds that of the null policy on the same world while the verified utility is
  below the null policy's. Claim (F-rpsi): upper 95% bound of the rate (Clopper–Pearson over the 35 lifetimes) ≤ 0.1.
* Secondary: per-variant trap counters (operator side), verified U, internal-return trajectories. The learner never
  receives evaluator internals (integrity scan).

## Experiment I — learning without an objective

A fixed goal-free exploration budget of 10,000 decisions in unfamiliar (train-split) worlds, no evaluation objective
revealed, then weights-only adaptation on independently chosen held-out tasks (val split) for 2,000 decisions.
Conditions: none, random, fixed curriculum, ordinary curiosity, meta-learned autonomous curriculum.

* **Primary P-I:** U_confirm(I-meta) − U_confirm(I-random) > 0, 95% interval. The number of self-generated goals
  completed is reported, never used as the metric.

## Experiment J — meta-learned reward verification

ProbeEnv (`capabilities/envs.py`: test first, launch only when good; optimum episode return 10), horizon 3, 5 seeds,
conditions J-meta, J-phi, J-lp, J-uninformative; plus the exact / truncated meta-gradient comparison of
`credit/meta_exact.py` on its small problems (reported, not tested here).

* **Primary P-J:** under J-meta the test-first rate over the last 512 episodes ≥ 0.8 (lower 95% bound over seeds,
  `stats.two_level`) AND under J-uninformative ≤ 0.1 (upper bound). Both must hold for DEMONSTRATED.
* Secondary: return against the optimum 10; ∂L_actor/∂ψ ≡ 0 check; nonzero outer gradients into ψ; ψ change between
  objectives.

## Multiplicity

The five primaries P-D, P-E, P-F, P-I, P-J are separate claims (one per manifest requirement EX-D/E/F/I/J); each is
tested at α = 0.05 on its own, and none is combined into a joint claim.

## Amendments

None.
