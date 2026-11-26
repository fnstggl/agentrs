# H - Tier 1 rapid scientific environments, wave 3 (`crucible.rapid`)

Continuation of workstream H (H0 foundation, wave-1 admission redesign) on the integrated tree. This guide records
what wave 3 changed, how the counts are measured and what they are. The H0 guide (`H0.md`) still describes the
module layout; the authoritative behaviour is the code and its tests.

## Admission (Part 7.4), as measured

Eleven checks per family (`admission.CHECKS`): consistency, conservation, controllability, achievable, informative,
identifiability, no_dominant_strategy, difficulty, no_leakage, distinct, mechanism. Statistical checks pass on two of
three seeds, the others on every seed. A family counts for Gate M only when it is admitted on seeds 0-2 **and** on
the held-out seeds 10-12 (`report.robust_report`) and its fingerprint is not a duplicate.

`no_dominant_strategy` (decision H-D1, wave 1, unchanged): black-box search with a fixed budget of 68 episodes per
seed on copies of the scored world (common random numbers, so the best constant is the hindsight-best for that
instance) over constant settings of every world-changing action kind (grid, random, coordinate search), threshold
rules on readings, win-stay/lose-shift and explore-then-commit; a family passes when the best simple policy's gain
over waiting is at most 0.8 of the family's adaptive public reference's gain. This is strict: a world whose optimum
is a constant once the mechanism is known is rejected however hard the mechanism is to learn, because a hindsight
constant reaches it. Most lever families of the primitive-graph engine fail it (best constant >= 0.9 of the
evaluator's upper reference), which is the audit's static-tuning finding measured honestly.

Mechanism claims (M.8): every family's controlled-intervention claims pass on seeds 0-2 and 10-12 (88 of 88 at
commit 4cae65d, 0 failures). Wave 3 repaired 14 claims that failed on some seed, mostly by measuring the claimed
mechanism directly (on the serviced machines, on adoption timing, on the incumbents' decisions, at the conversion
midpoint, with noise replicates) and, where the world itself did not do what it claimed on some instances, by
repairing the builder (profile attribute weights, distribution demand ladder, demand-shift capacity range,
topology-shift rewiring).

Public references (difficulty and dominance anchor): wave 3 removed reads of hidden configuration from references
that are presented as public: lever roles are found by probing or from public defaults, contest prizes and route
damages are estimated from observed income and costs, the dig cost from the first dig, payoff scales from observed
payoffs. References of program families still read their own config for sandbox plumbing (source templates).

## New in wave 3

| module | what | side |
|---|---|---|
| `categories.py` | the Part 8.2-8.10 behavioural requirements as measurements (held-out conditions outside the settable range, action forms, observed-income separation of discovery from random novelty, opponent populations, independent learners vs coordination, early-best constants failing later) | evaluator |
| `vec.py` | `RapidVecEnv` (learner-facing batched worlds, observation_only, reward 0.0), `BatchedSearchLearner`, throughput target H-D3 | world |
| `distinct.py` | pair protocols for pairs the generic response signature cannot separate (law driven past the settable range; overstock expiry; latency profile under an even split) | evaluator |
| `generator.compose` | context dimensions: adaptive rival participants, installed-base network, shared supply, output-quota institution, initial conditions | world |
| `report.assemble` | registry report from sharded admission reports | evaluator |
| `engines/base.observation_channels` | engines declare observation channels for fingerprints (aggregated routing) | world |

Gate tests added: `tests/v5/test_gate_SP_0_9.py`, `SP_7_0`, `SP_7_2`, `SP_7_5`, `SP_8_0`, `SP_8_0b`, `SP_8_12b`.

## Decisions

* H-D1 (wave 1): dominance fraction 0.8 of the adaptive public reference's gain, 68-episode budget per seed.
* H-D2 (wave 3): D-9 recreation is checked against 4.0 validation families as well as sealed test families; the three
  v5 families whose mechanism motif set equals a 4.0 held-out family's (`ch_sign_reversal`, `ch_rule_change` =
  scm_regime; `ch_wandering_taste` = scm_invu) moved to the evaluation partition.
* H-D3 (wave 3, fixed before measuring): a learner rolling out 256 worlds through `RapidVecEnv` must reach
  >= 20,000 world-steps per CPU second and >= 10x the single-world path.

## Limitations (measured, not assumed)

* Gate M counts are far below 60/20 under H-D1 (see the workstream report for the numbers at each commit).
* Program-execution families pay for compute by parent-measured CPU time (D-20). Their admission values depend on
  machine load: `pe_stream_processor`'s reference earned 857 in one run and 703 in another on the same seed and code.
* `lab_drift_blanks` scores one held-out dose (2.5) inside the settable range (0-4), so the 8.2 requirement
  "evaluated on unobserved conditions" is not met by every admitted 8.2 family.
* `net_topology_shift`'s claim fails on seeds 3, 5 and 9 (outside the admission seed sets).
* Families still declare a default objective (type and text); the measured independence is that every default
  objective is scored by its type's generic rule and the generator serves every family from its public face.
