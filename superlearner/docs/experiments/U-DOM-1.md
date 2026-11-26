# U-DOM-1 — CAM strategic dominance: DOM-11 conditions A–I (pre-registration)

Spec: `CAM_UNIFIED_SPEC_DOM.md` §6 (conditions A–I), §8 DOM-11; plan `CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md` §5
(E0, E1, E2). Manifest: DOM-11 (and DOM-08 / DOM-04 evidence rows from E0). Written and committed **before any
confirm-set evaluation** of any CAM Unified DOM learner. Nothing in this file may change after the first confirm row
is written; a change made later is an amendment, listed at the end with its commit, and the analysis then reports
both the pre-registered and the amended result.

Driver: `superlearner/src/superlearner/cam_unified/experiments/dom_suite.py` (code commit c2e089f or later;
conditions run only through `runtime.run_lifetime`, observation-only, one provenance row per (condition, seed) in
`<run>/eval/provenance.jsonl`). Analysis: `experiments/stats.py` (commit 1131895 or later): `paired` /
`two_level`, method `"t"` (studentised random-effects interval, calibrated degrees of freedom; coverage 0.945 at 3
seeds, 0.965 at 5 seeds in its own simulation).

## System under test

The DOM architecture of the redesign plan §2 at two scales:

* **dom_dev** (inferential): `config.make_preset("dom_dev")` — 8 layers × 384, 32 state tokens, Qwen3-0.6B frozen
  semantic model, every DOM switch of `DOM_SWITCHES` (state gate, semantic cross-attention, decoder conditioning,
  intent bottleneck K=4 candidates, value channel, world-model transition through the last core blocks, no EMA
  target, memory OFF, reward_mode observation_only, meta step every 16 updates).
* **dom** (demonstration only, single seed): `make_preset("dom")` — 32 × 1536 × 24, ~1.08 B core parameters,
  Qwen3-4B. No inferential claim is made from a single seed.

The actor reward is R_ψ (`selfeval/reward_model.py`); verified labels reach the learner only through the
LabelStore. Memory is off in every DOM run (declared deviation of the plan §6), so condition E differs from C only
by the source lifetime's archive/replay being absent.

## Training (source lifetimes)

Per seed s ∈ {0, 1, 2, 3, 4} (dom_dev): one lifetime on the **rapid train split** (`crucible.rapid.families("train")`
minus `BROKEN_MULTIAGENT_RAPID`), B = 4 environments, world seeds 100,000 + 1,000·s + j, families cycled in sorted
order, **20,000 environment decisions**, update every 8 rounds. Initial weights (before the first update) and final
weights are saved with `CheckpointOps.save_weights` (arch_hash and code_commit in the file). Conditions H and I
have their own source lifetimes with the same seeds, worlds and budget (H: core learning rate 0; I: only the
semantic group learns).

## Conditions (evaluation)

Evaluated on the **confirm set**: the rapid **val** split (19 structure-fingerprint held-out families), one world
per family per seed, world seeds 500,000 + 1,000·s + j, **2,000 decisions** per (condition, seed) lifetime, B = 4.
Each (condition, seed) is evaluated on the confirm set exactly once (a crashed evaluation is reported as missing
cells, not re-run).

| Condition | Weights | Learning during the evaluation lifetime |
|---|---|---|
| A | Qwen3-0.6B (dom_dev) as `FrozenSemanticAgent("tools")`, same tools; semantic tokens recorded | — |
| B | initial weights of seed s | off |
| C | final weights of seed s (weights-only load) | on (continual) |
| D | final weights, strategic core (`STRATEGIC_PREFIXES`) reloaded from the initial weights | off |
| E | final weights, no source archive / replay / memory (fresh archive) | off |
| F | final weights, decoder conditioning removed (latent inputs → 0) | off |
| G | final weights, latent-swap probes on every decision (greedy acting) | off |
| H | final weights of the frozen-core source lifetime | on, core lr 0 |
| I | final weights of the semantic-only source lifetime | on, semantic group only |

References computed on the operator side for free: null policy and oracle (they define U).

## Primary metric

Verified oracle-relative utility per world, **U = (v − v_null) / (v_oracle − v_null)**, computed on the operator side
from `envs/experience.py` (private references; never shown to the agent) at the end of the evaluation lifetime.
Success := U ≥ 0.5. Worlds whose oracle does not exceed the null policy (U undefined) are excluded for every
learner alike and listed.

## Primary hypotheses (fixed-sequence testing, family-wise α = 0.05)

* **P1 (acquisition).** U(C) − U(B) > 0: `stats.paired(U_C[s, w], U_B[s, w])` (same seeds; per-cell pairing),
  95% interval; DEMONSTRATED iff the lower bound > 0, FAILED iff the upper bound ≤ 0, else INCONCLUSIVE.
* **P2 (dominance; tested only if P1 is DEMONSTRATED).** U(C) − U(D) > 0 with the same rule, AND the point estimate
  of the retained fraction ρ_D = (Ū_D − Ū_B) / (Ū_C − Ū_B) ≤ 0.5 (removing CAM's experiential learning removes at
  least half of the acquired competence). If P1 is not demonstrated, P2 is INCONCLUSIVE by design.

## Secondary (descriptive, 95% intervals, no multiplicity claim)

U(C) − U(E), U(C) − U(F), U(C) − U(H), U(C) − U(I), U(C) − U(A); success rates; ρ for each of D, E, F, H, I; Condition G:
action-change rate, tool-change rate, donor-tool agreement and its excess over the own–donor baseline agreement;
the DOM-08 gradient table and KL(π ‖ π with I permuted) before training (E0); compute per condition (decisions,
GPU-busy seconds, semantic tokens, wall time), with every ratio above 2× relative to C flagged.

## 1B demonstration rows (E0 / E1; single seed, no inferential claim)

E0: DOM-08 gradient table after one step; G latent swap (target ≥ 90% donor-following after a short imitation
phase); F and H Δlog-prob and action change on fixed probe contexts. E1: the 1B continual-learning run on rapid +
mech train families, weights-only and continual-on evaluations on the confirm families at checkpoints 4 h apart,
with B, H (dom_150m), null / oracle and A (Qwen3-4B, 2-family subset, token-matched) at equal decisions. Reported as
"demonstration, no inferential claim".

## Integrity and exclusions

Every CAM run directory must pass `dom_suite.integrity_scan` (reward_mode observation_only, every archived actor
record reward 0, no label keys in actor-visible fields); a run that fails it is excluded and reported. Development
and tuning use the train split only; nothing tuned on val. The rapid val split and the mech val / test families with
seeds ≥ 500,000 are touched only by the confirm evaluation described here.

## Amendments

None.
