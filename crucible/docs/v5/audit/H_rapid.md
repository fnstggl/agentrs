# Audit H_rapid: Tier 1 rapid scientific environments (Parts 7 and 8, Gate M, SP-7.1/7.3/7.4/8.11/8.12)

Auditor: independent audit subagent, area H_rapid (workstream H). Tree: `feat/crucible-5-complete`. Work was assigned at
a881d18. HEAD moved to dd745b9 during the audit, but that commit only touched docs and the manifest:
`git diff a881d18 HEAD -- src/crucible/rapid src/crucible/mechanisms tests` is empty. The rapid source digest is
`12ed0ba902248d274d5653a1d7a0e690`.

Every number below comes from runs I made myself. The probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/H_rapid/`. I re-ran admission from
scratch with no cached results (`p02_admit_all.py`, seeds 0-2 and 10-12). I then ran the gate tests against a cache
built from those runs. The admission computation is the same; the cache only spares a second run.

## 1. Summary

The Tier 1 suite is far beyond a stub, and a good part of it is solid:

* 74 registered families on 9 independent v5 engines. The 4.0 scm/abm/techsys engines are untouched.
* A real structured action interface with ten action kinds.
* 28 versioned primitives.
* An admission pipeline that rejects broken control worlds.

It does **not** meet Gate M, and several of its safeguards against inflation and shortcuts are weaker than they look.

* **Counts.** On seeds 0-2, 69 families are admitted: 51 train and 18 eval, against 60 and 20 required. On the held-out
  seeds 10-12, 48 train and 17 eval are admitted. Six of the 60 enumerated Part 8 families do not exist. Four more
  exist but none is admitted.
* **Fingerprints count parameter variants.** Doubling the horizon or adding distractors gives a nonzero fingerprint
  distance in 116 of 178 axis variants. Distances run 0.025-0.06; the closest pair of real families is at 0.063.
  The anti-inflation test M.n1 crashes.
* **Trivial strategies pass admission.** The dominance check tries only levers at min, mid and max. In 17 of 31
  admitted lever families, a single constant lever setting found on a 7-point grid gets more than 75% of the oracle's
  gain. In 9 it gets 98% or more, and in 3 it beats the "oracle".
* **Mechanisms are not verified at admission.** 13 families fail their own controlled-intervention tests, and 9 of
  them are admitted. Gate M.8 fails.
* **Shortcuts for the learner.**
  * The participant-visible `spec` query returns the family name, seed, category and split (`val` for evaluation
    families).
  * The Tier 1 learner adapter defaults to `legacy_shaped_reward`.
  * Rapid has no objective-free mode.
  * The evaluator outcome can be reached from the participant's own object.
* **The known failures.** Three are stale tests. One is a real bug in `SelectionEffect` (per-instance parameter
  broadcast); a one-line fix, checked in memory, gives 0 mismatches. The SP-7.3 gate test is also stale: it records
  4 of 8 engine kinds although all 8 have admitted families.

## 2. What I ran

| # | Command / probe | Result |
|---|---|---|
| T1 | `pytest tests/test_v5_rapid_engines.py tests/test_v5_rapid_families.py -k "five_engines or share_only_the_contract or bit_for_bit or contract_fields"` | 4 failed, 30 passed: the four known failures reproduced |
| T2 | `pytest tests/test_v5_rapid_{primitives,actions,engines,families}.py` | 199 passed, 4 failed |
| T3 | `crucible.scale.slots --cpu 1 -- python p02_admit_all.py 0,1,2` (every family, exceptions captured) | 74 families, 69 admitted (51 train / 18 eval), 0 errors |
| T4 | the same on seeds 10,11,12 | 65 admitted (48 train / 17 eval) |
| T5 | `pytest tests/v5/test_gate_SP_7_1.py ..._SP_7_3.py ..._SP_8_11.py ..._SP_8_12.py` (cache from T3) | 4 passed. Recorded: SP-7.1 = 28, SP-7.3 = 4 (stale), SP-8.11 = 10, SP-8.12 axes = 9 with 0 mismatches |
| T6 | `slots -- pytest tests/v5/test_gate_M.py tests/v5/test_gate_SP_7_4.py` (cache from T3/T4) | 9 passed, **2 failed** (M8, Mn1) |
| T7 | `pytest tests/test_mech_{scm,abm,techsys,rig,evaluator}.py` | 326 passed |
| p03 | every primitive re-run with per-instance (B,) parameters | `selection_effect` crashes; 25 conform |
| p13 | in-memory patch of `SelectionEffect.apply`, then batch vs single | unpatched: broadcast error. Patched: 0 mismatching ticks |
| p06 | fingerprint distance of difficulty-axis variants; closest real pairs | 116/178 variants have distance > 0; closest real pair 0.063 |
| p07 | best constant lever setting (grid) vs oracle, admitted lever families | 17/31 above 0.75 of the oracle's gain |
| p14 | every family's intervention claims, seeds 0-2 | 13 families fail, 9 of them admitted |
| p08 | coverage of the 60 enumerated items | 54 registered, 50 with an admitted family |
| p04 | participant `spec` query | returns family, seed, category, split |
| p09 | reward-like keys and words in everything participants receive | no reward key in 3,676 step results |
| p10 | observation-only default through `crucible.experience` | default is `legacy_shaped_reward` |
| p11 | 4.0 val/test motifs vs v5 training motifs | vocabularies do not overlap; distances 0.58-0.68 |
| p12 | in-process isolation | PublicView can be bypassed; `outcome()` and `verify()` are reachable |
| p15 | hidden values the leak scan never checks | 294 values unscanned; no verbatim string leak found |

## 3. Counts (independent admission)

| | seeds 0-2 | seeds 10-12 | required |
|---|---|---|---|
| registered families | 74 (55 train / 19 eval) | | |
| admitted train | **51** | **48** | 60 (M.1) |
| admitted eval (partition v5-eval-1) | **18** | **17** | 20 (M.2) |
| feasibility-validated | 73 | 72 | 80 (M.7) |
| families with all intervention claims passing | 61 (29 failures) | | 80, 0 failures (M.8) |
| engines with admitted families | 9 v5 + 3 legacy | 9 + 3 | 6 (M.4) / 8 new kinds (SP-7.3) |
| categories | 10 | 10 | 10 (M.3) |
| action kinds offered by admitted families | 10 | | 5 (M.6) / 10 (SP-8.11) |

**Rejected on seeds 0-2:**

* `net_cascade`: difficulty
* `pe_cascade_guard`: no_dominant_strategy
* `pe_contention_sched`: identifiability
* `pe_rollout_targeting`: difficulty
* `pe_stream_processor`: achievable, identifiability, no_dominant_strategy and difficulty

**Also rejected on seeds 10-12:** `des_bottleneck`, `dl_short_term_damage`, `pe_router` and `pe_rule_induction`.

The executable-program engine is the least robust. Only 5 of its 9 families are admitted on seeds 0-2, and 4 on
seeds 10-12.

**Coverage of the 60 enumerated families of Part 8** (`p08_items.log`):

* Missing entirely:
  * 8.2.2 competing hypotheses
  * 8.2.3 hidden conservation
  * 8.5.1 hidden opportunities with costly investigation
  * 8.5.4 uncertain investment
  * 8.6.5 irreversible commitments
  * 8.7.5 distributed resource coordination
* No admitted family: 8.3.4, 8.3.5, 8.6.3 and 8.7.4.
* Covered only by an evaluation family, so never trained on: 8.2.6, 8.4.4, 8.5.6, 8.6.4, 8.7.4, 8.7.6, 8.8.3, 8.8.6
  and 8.9.4.

**Action-space diversity among the admitted families:**

* 24 of 69 are lever-only.
* `graph_edit` is offered by 1 admitted family (net_repair) and `coordinate` by 4.
* Only 6 admitted families have several participants.
* Objectives: 69 of 74 families are `maximize_net`.

## 4. The known failing Tier 1 tests and the cxs_selected_layers bug

1. `tests/test_v5_rapid_engines.py:22`, `test_five_engines_registered_and_four_spec_kinds_covered`. Stale: it asserts
   exactly the H0 engine set.
2. `tests/test_v5_rapid_engines.py:34`, `test_engine_modules_share_only_the_contract`. Stale boundary: `program_exec`
   legitimately imports `crucible.sandbox.runner` (`engines/program_exec.py:43`).
3. `tests/test_v5_rapid_families.py:26`, `test_registry_entries_carry_the_contract_fields`. Stale: it asserts 5
   engines, and there are 9.
4. `test_batch_path_equals_single_worlds_bit_for_bit[cxs_selected_layers]`. A **real bug**:
   * Where: `src/crucible/rapid/primitives.py:539`, `y = x["x"][:, None] + p["sd"] * z[:, :int(p["n"])]`.
   * Why it fails: in the batch path, parameters are stacked per instance, so `p["sd"]` has shape (B,). Multiplying it
     by a (B, n) array cannot broadcast, which is the reported `(6,) (6,16)` error.
   * Why it was missed: the single-world path uses scalar parameters, so admission (single worlds) admitted the family
     and the error only shows on the batch path. The module docstring promises that parameters may be (B,), but the
     SP-7.1 gate checks that only for `saturation` (`tests/v5/test_gate_SP_7_1.py:37-43`).
   * Fix: `sd = np.asarray(p["sd"]); sd = sd[:, None] if sd.ndim == 1 else sd`. Patched in memory (p13), this gives
     0 mismatching ticks over 25 ticks × 6 seeds.
   * Scope: p03 found no other primitive with this defect.

**New gate-test failures I found:**

* `tests/v5/test_gate_M.py:170` (M8): 29 claim failures (section 6).
* `tests/v5/test_gate_M.py:190` (Mn1): `TypeError: can't multiply sequence by non-int of type 'float'`. The parameter
  variant builder does not handle `team_assembly`'s nested `unit_cost` list, so M.n1 cannot be measured at all.
* `tests/v5/test_gate_SP_7_3.py:23,77`: `_transition_markers` knows only the four H0 kinds and the test asserts
  `len(implemented) == 4`. It passes while recording `new_engine_kinds=4`, although `program_exec`, `strategic`,
  `experiment_lab` and `planning` all have admitted families and their own transition code. It can never record 8.
* `tests/v5/test_gate_M.py:62` records `legacy_4_0_val_test_families_untouched=0`. The 4.0 engines are not loaded
  there (`ensure_engines()` is never called). In fact there are 7 val and 11 test families.

## 5. Structural distinctness (fingerprints, Part 7.5, M.5, M.n1)

`src/crucible/rapid/fingerprints.py` does not measure structure in a way that rules out parameter variants:

* **Horizon.** The horizon enters as a bucket (`<=40`, `<=80`, `>80`, line 104), so a world run for longer gets a new
  fingerprint.
* **Network size.** Size enters as a power-of-two bucket of the node count (line 127), and the number of readings is
  bucketed the same way.
* **Labels instead of measurement for non-graph engines:**
  * topology and motif edges are the scenario label string (`scenario_structure`, line 130);
  * motifs are lists hand-declared per scenario (for example `strategic.motifs`);
  * the `participants` and `objective` components are the family's own declarations (lines 151-154).
* **No threshold.** `counted()` (line 261) treats a family as a duplicate only at distance exactly 0.0, and the
  admission `distinct` check is `distance > 0` (`admission.py:165`).

Probe p06 raised one difficulty axis at a time and compared each variant with its own family: 178 variants, 116 at
nonzero distance.

| Axis | Variants at nonzero distance |
|---|---|
| horizon | 73 |
| n_variables | 19 |
| graph_size | 15 |
| action_complexity | 6 |
| hidden_state_complexity | 3 |

Typical distances are 0.025 (horizon) and 0.042-0.058 (+2 distractors). The closest real pair, `mk_elastic~mk_signal`,
is at 0.063. Any distractor or horizon variant of a family would therefore count as a new structural family. The M.n1
test only tries renamed copies and costs scaled by 1.37, which are the variants the fingerprint already ignores.

The registry's "why distinct" entry is the measured nearest family plus which components differ. Nothing checks
that difference with a controlled intervention, except the four look-alike markets
(`claims.lookalike_matrix`, which passes). Part 7.5's "use controlled interventions to verify claimed structural
differences" is therefore unmet for 70 of the 74 families.

**Comparison with the 4.0 families (p11).** 4.0 motifs use a different vocabulary: `confounder`, `regime`,
`local_network_effects`, `shared_pool`, and so on. Every 4.0 val/test family sits at 0.58-0.68 from every v5 training
family, mostly because the vocabularies and the engines differ. The "distinct from 4.0" check is therefore vacuous.
It cannot see that v5 training now contains close analogues of 4.0 sealed test mechanisms:

| 4.0 sealed test family | v5 training analogues |
|---|---|
| `scm_regime` (sign/strength switch at a hidden tick) | `ch_sign_reversal`, `ch_rule_change` |
| `scm_feedback` (lagged feedback with hidden confounders; its docstring says "both motifs are absent from training") | `cx_reactive_history`, `cx_balancing_loop` |
| `tech_contention` (shared pool) | `pe_contention_sched` (resource classes interfere) |

D-9 is met literally, since no 4.0 family object is reused. The 4.0 sealed claims that a motif is novel are no longer
safe for a learner trained on v5-train.

## 6. Admission (Part 7.4, SP-7.4)

**What works (reproduced):**

* All ten checks run on every family, and each one rejects its deliberately broken control world.
* World-side modules import no evaluator module (checked in a fresh interpreter).
* 2,081 hidden numbers were scanned and none leaked.
* The kernel facility's tools leak nothing.

**Defects:**

* **Dominance.** `no_dominant_strategy` (`admission.py:383,406`) tries only the canonical constants from
  `policies.canonical_actions`: levers at min/mid/max, always one option of a decision, an even split or everything on
  one target.
  * The H3 recipes price levers so that the optimum sits at a hidden interior setting
    (`recipes.price_levers_at`). That defeats the check without removing constant-setting solutions.
  * Probe p07 searched a 7-point grid per lever on copies of the same world. In 17 of 31 admitted lever families, one
    constant setting reached more than 0.75 of the oracle's gain, the very threshold admission uses:

    | Family | Best constant (share of oracle gain) |
    |---|---|
    | hc_confounded | 1.06 |
    | net_routing | 1.06 |
    | net_topology_shift | 1.03 |
    | cx_overdose | 0.996 |
    | cx_competing_causes | 0.991 |
    | hc_conditional | 0.990 |
    | cxs_selected_layers | 0.983 |
    | cx_balancing_loop | 0.982 |
    | game_adaptive_rival | 0.980 |
    | ch_demand_shift | 0.92 |
    | ch_tech_transition | 0.92 |
    | ch_sign_reversal | 0.80 |

  * So a "hidden causal" world can be solved without causal inference, a "changing environment" without revising
    beliefs, and an "adaptive rival" world (8.8.1) almost optimally with one fixed price.
  * The canonical `levers_mid` already scores 0.6-0.97 in several admitted families (for example mk_signal 0.71, 0.70
    and 0.97), which passes only because 2 of 3 seeds suffice.
* **Intervention claims are not part of admission** (probe p14):
  * Admitted families failing their own claims:
    * `cx_balancing_loop`: the claimed "settles a third lower" is not seen; immediate effect 5.86, lasting 5.51.
    * `plan_distribution`: shipping to high-demand or low-demand sites sells the same 31 units.
    * `team_assembly`
    * `strat_bargain`
    * `strat_blotto`
    * `strat_entry`
    * `strat_price_war`
    * `pe_deadtime_control`
    * `pe_fleet_policy`
  * `pe_contention_sched` fails on float equality (ratio 2.9896439 vs expected 2.9896479), which is a tolerance bug.
* **Leak scan blind spots.** The leak scan matches only non-integer floats whose repr is at least 7 characters long
  (`admission.py:91`). 42 string, 4 integer and 248 short or integral hidden values are never scanned. I found no
  actual string leak (p15).
* **Identity leak.** `spec` exposure of family, seed and split is not covered by any check (section 8).
* **Difficulty axes.** The axis check passes when the config digest changes (`admission.py:355`). It does not show
  that the world becomes harder.
* **The oracle is not an upper bound.**
  * A constant beats it in 3 families.
  * The reference exceeds it (share > 1) in des_maintenance, pe_fleet_policy, net_routing and pe_router.
  * The generic reference (`references.practice_constant`, line 489) replays deep copies of the *same* world, with the
    same noise, before playing the scored episode. It is a hindsight constant, not a public-information learner.

## 7. Engines (Part 7.3, M.4, SP-7.3) and 4.0 preservation

**Engine independence.** Every v5 engine module has its own transition code and imports only `base`, `actions`,
`primitives` and, for program_exec, the sandbox. Examples:

* `discrete_event` uses an event heap;
* `strategic` has separate `_adv_<scenario>` functions with learning opponents (propensity, fictitious play, regret
  matching, Boulware and tit-for-tat negotiators);
* `program_exec` runs agent-written Python inside the dynamics through `crucible.sandbox`.

These are not thin wrappers. However:

* Each non-graph family is a single hand-written scenario. There are 44 distinct (engine, scenario) pairs for 74
  families; 29 families share `primitive_graph`, 2 share the discrete-event `inventory` scenario (perishability is a
  `shelf_life` parameter toggle) and 2 share routing.
* The four H1/H2 engines and their 45 families have **no unit tests**. `families_h3.py:14` cites a test file that does
  not exist.
* The 4.0 engines are preserved: `src/crucible/mechanisms` is unchanged since c453d27 (4.0 checkpoint 3), 326 4.0 tests
  pass, and the 25 families (7 train, 7 val, 11 test) are intact.

## 8. Observation-only amendment and shortcuts (Tier 1)

**What is right:**

* `RapidEnv.step` returns only `observation`, `result` and `events`. No reward key appeared in 3,676 step results
  across all 74 families (p09).
* Observations carry real consequences: cash, revenue and cost per tick, readings, and the results of the agent's own
  actions.
* Experiments and journal predictions are not rewarded.

**What violates it, or offers shortcuts:**

* **Identity in the spec.** The `spec` query (`env.py:68`, `:235`) returns `family`, `seed`, `engine`, `category` and
  `split`, for example `strat_learning_rival` → `split: "val"`. Family names describe the mechanism (`ch_sign_reversal`,
  `cx_overdose`, `strat_learning_rival`), so the name is both a memorisation key and an evaluation-partition leak.
  `kernel_world` also puts `family_split` in the WorldSpec params (`mechview.py:181`). Briefs are fixed per family, so
  the family can be recognised even without the query, except for the deliberate look-alikes.
* **Default reward mode.** `crucible.experience.make` (`registry.py:99`) and `RapidExperienceEnv` (`tier1.py:472`)
  default to `legacy_shaped_reward`. The 4.0 lever view returns nonzero shaped reward by default (p10). (Owned by
  workstream L, but it decides how Tier 1 is consumed.)
* **No objective-free operation.** Every rapid world carries objective text, and `make()` has no `objective` parameter.
* **Evaluator reachable in-process.** The evaluator outcome is reachable from the participant's object:
  * `PublicView` can be bypassed with `view._PublicView__env`;
  * `RapidEnv.outcome()` is a public method;
  * `RapidExperienceEnv.verify()` (`tier1.py:616`) returns the outcome even in observation_only mode (p12).

**Would a learner here acquire transferable capabilities?** In the program-execution, strategic, planning,
experiment-lab, search and state-machine families, plausibly yes: they need experiment design, model fitting,
controller construction, negotiation against held-out counterpart families, and coordination with private
information. In many lever families, the dominant learnable skill is black-box tuning of one or two constants. The
spec identity and fixed briefs also make per-family memorisation easy.

## 9. Fast path and kernel consistency (Part 8.12, SP-8.12)

**What the gate shows (reproduced):** 9 axes; 0 mismatches over 320 ticks between the fast simulator and kernel-hosted
`rig` twins for 4 lever families; 0 batch-vs-single mismatches; about 20,300 batch world-steps/s under shared load.

**Limits:**

* The twin hosts the same engine code through `mechview` and `rig`. It proves that hosting, event sourcing and ledger
  money are consistent, not that two independent representations agree.
* Twins exist only for the 26 lever-view families. The batch path covers 31 of 74 families, fails for
  `cxs_selected_layers`, and is used only by tests: no learner rollout uses it.
* The difficulty-axis check measures that a quantity increases (sizes, noise, cost), not that difficulty increases.

## 10. Classification

| Subsystem | Verdict | Effort |
|---|---|---|
| Primitive registry | REPAIR | S |
| Structural generator and recipes | EXTEND | L |
| H0 engines (pg, DES, graph, POSM, search) | RETAIN | S |
| H1/H2 engines (program_exec, experiment_lab, planning, strategic) | REPAIR | L |
| Family registry and families | EXTEND | L |
| Admission and evaluator side | REPAIR | M |
| Fingerprints and distinctness registry | REDESIGN | M |
| Structured action interface | RETAIN | S |
| RapidEnv and observations (observation-only) | REPAIR | M |
| Fast batch path and kernel twin | EXTEND | M |
| 4.0 engines and sealed families | RETAIN | S |
| Tier 1 unit and gate tests | REPAIR | M |

| Requirement | Verdict | Criteria weak | Why |
|---|---|---|---|
| M.1 | EXTEND | yes | 51 (48 robust) of 60; inherits fingerprint and dominance weaknesses |
| M.2 | EXTEND | yes | 18 (17 robust) of 20; split visible to participants |
| M.3 | RETAIN | yes | 10/10 categories; the 60 items are not checked |
| M.4 | RETAIN | no | 12 engines (9 v5 + 3 legacy), independent code |
| M.5 | REDESIGN | yes | fingerprints exist but count axis variants |
| M.6 | RETAIN | no | 10 modalities |
| M.7 | EXTEND | no | 73 of 80 |
| M.8 | REPAIR | yes | 61 families pass; 29 failures; no cross-family discrimination |
| M.n1 | REDESIGN | yes | test crashes; distance > 0 meaningless |
| SP-7.1 | REPAIR | yes | selection_effect contract bug; per-instance check covers one primitive |
| SP-7.3 | REPAIR | no | all 8 kinds present; gate stale at 4 |
| SP-7.4 | REPAIR | yes | dominance, claims, leak scan, 4.0 distinctness, spec identity |
| SP-8.11 | RETAIN | no | 10 types with verified effects |
| SP-8.12 | EXTEND | yes | narrow twin and batch coverage; axis ≠ difficulty |

**Spec requirements with no manifest entry:**

* SP-7.2: structural composition.
* SP-7.5: intervention-verified distinctness registry.
* Independent goal generation.
* Each of the 60 enumerated families, individually.
* 8.2: evaluation on unobserved conditions.
* 8.4: discrete, continuous and programmatic actions in resource allocation.
* 8.5: discovery distinguishable from random novelty without reward.
* 8.8: policy populations and held-out opponents.
* 8.9: genuine interdependence.
* 8.10: belief revision.
* 8.12: throughput and learner use of the fast path.
* Mechanism-level non-contamination of the 4.0 sealed families.
* Tier 1 observation-only specifics: no identity in the spec, objective-free mode, adapter default, unreachable
  evaluator.
* Robustness of admission on held-out seeds.

## 11. Work needed, in priority order

1. **Fix the defects.**
   * Fix the `SelectionEffect` broadcast and add a per-instance conformance test over all 28 primitives.
   * Update the three stale unit tests and the SP-7.3 gate: markers for the four new kinds and `>= 8`.
   * Make the M.n1 variant builder recursive and add axis variants; call `ensure_engines()` in M.2.
2. **Harden admission.**
   * Add a best-constant baseline (grid or random search over constant actions) and a simple adaptive baseline to
     `no_dominant_strategy`.
   * Make intervention claims a required check.
   * Scan string, integer and rounded hidden values.
   * Require each difficulty axis to move the reference share in the expected direction.
   * Make oracles true upper references.
3. **Redesign fingerprints.**
   * Make them invariant to axes and coefficients.
   * Use behavioural mechanism signatures from intervention responses.
   * Use a calibrated distance threshold validated against axis and parameter variants.
   * Verify the difference to the nearest family with an intervention that separates them.
   * Map the 4.0 vocabulary so that contamination of the sealed sets can be measured.
4. **Close observation-only gaps.**
   * Remove family, seed, category and split from participant-visible data.
   * Default `reward_mode="observation_only"` in the Tier 1 adapter.
   * Add objective-free worlds.
   * Put `outcome()` and `verify()` behind an evaluator handle the learner never receives.
5. **Extend the families.**
   * Implement the 6 missing items and get the 4 unadmitted ones admitted.
   * Retune the constant-dominated families so that state-dependent policies are needed.
   * Reach 60 training and 20 evaluation families admitted on held-out seeds as well.
   * Add unit tests for every H1-H3 engine and family.
   * Broaden the use of graph edits and coordination.
6. **Generation.** Generate objectives independently of worlds and extend composition beyond `primitive_graph`.

## Skeptic review

Reviewer: skeptic subagent for H_rapid (workstream H). I worked on the same tree (rapid sources as at a881d18; HEAD
dd745b9 changes only docs and the manifest). Every probe script and log is in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/H_rapid_skeptic/`. I tried to refute
every RETAIN and every verdict that might be too generous, and I checked whether the harsh verdicts hold up. Nothing
in the source tree was modified.

### S.1 What I ran

| # | Probe | Result |
|---|---|---|
| s02/s03 | **Mutation testing.** I wrote six mutants (`mut/mutants.py`), each breaking one H0 mechanism while keeping the world runnable. Each was run against (a) the 165 rapid unit tests in `tests/test_v5_rapid_{engines,families,actions}.py`, with the 4 known failures deselected and the mutants loaded by a pytest plugin (`mut/mutplug.py`); (b) the family's own intervention claims on seeds 0-2; (c) full admission on seeds 0-2; (d) the SP-8.11 effect functions. The unmutated baseline passes all four. | See the table below. `test_v5_rapid_engines.py` caught **0 of 6** mutants. |
| s05/s10 | Multi-agent and other action kinds through `crucible.experience` (the learner-facing path) | `coordinate` is never agreed. Agent a1 receives a0's result. Single-agent kinds work. |
| s06 | Inventory of every key participants receive in results, entities, resources and events | `posm_unlock` returns a per-step progress `signal` |
| s07/s07b | Default reward on each learner path | `rapid:` uses mode `legacy_shaped_reward`, but the scalar is 0. The 4.0 lever view of v5 families (`MechVecEnv`, the path the PPO test trains on) pays nonzero shaped reward by default (0.82, 0.63 per step). |
| s08/s08b | Sealed 4.0 test families through `crucible.experience` | 11 of 11 test-split families can be instantiated with any seed (`X.make(1, "mech:scm_regime", 7)`), with `legacy_shaped_reward` by default |
| s09/s14 | Counts with the spec's conditions applied together: admitted on both seed sets, own claims pass (auditor p14), not solved by one constant (auditor p07), not a scenario-parameter variant | **30 train / 11 eval** (not 51/18). All 10 categories keep at least 2 families at RapidEnv level. |
| s11 | Part 7 generator admission rate on random recipes | `rapid.generator.compose`: 1/16 admitted. `research.v5.compose` (calibrated): 1/14 admitted, and that one is solved by a single constant at 0.955 of the oracle's gain. 0/14 generated families have any intervention claim. |
| s12 | Does the `strat_learning_rival` opponent learn? | Yes. Regret matching converges to the best response against each constant play (counts 59/60). |
| s13 | Do difficulty axes make worlds harder? Reference share of the oracle gain at the default vs a harder axis value, 14 families, seeds 0-1 | **19 of 35** axis settings are no harder. Doubling `observation_noise` changes the share by exactly 0.000 in hc_confounded, cx_competing_causes, net_routing and mk_elastic (and by -0.003 in sci_functional_law). |
| s15 | What anchors the difficulty check? | For **27 of 69** admitted families, the "public-information reference" is `practice_constant`: one constant picked by replaying deep copies of the same world. A deep copy replays identical noise (`True`). |
| s01 | Are evaluation families parameter variants of training scenarios? | `des_perishable` is `des_inventory` plus `shelf_life`. `net_topology_shift` is `net_routing` plus `rewire_at`/`assign2`. Each shares the scenario code path. |
| s04 | Side check | One mutant was rejected only by the leak scan: the served demand legitimately equalled a hidden pair demand. That is a false positive on a genuine observable consequence. |

**Mutation results** (s02, s03):

| Mutant (what it breaks) | Target family | 165 unit tests | Own claims | Admission | SP-8.11 effect check |
|---|---|---|---|---|---|
| `pg_nolag`: every lagged edge reads the current value | eng_delayed_controller (8.3.3) | all pass | FAIL 3/3 seeds | **ADMITTED** | still "verified" |
| `des_servers_ignored`: server allocation has no effect | des_bottleneck | 1 fail (the claims test) | FAIL | rejected | `allocate` still "verified" |
| `net_meanfield`: diffusion over a complete graph | net_diffusion | 1 fail (the claims test) | FAIL | rejected (no_dominant_strategy only) | n/a |
| `net_built_edges_ignored`: `graph_edit add_edge` is charged and listed, but the new link never carries traffic | net_repair | **all pass** | **pass** | rejected only by an incidental leak-scan hit on seed 2 | "graph modifications" still **verified** |
| `posm_tests_uninformative`: diagnostic tests are independent of the fault | posm_fault_diagnosis | 1 fail (the identification test) | pass | rejected (identifiability) | n/a |
| `des_maint_memoryless`: failure hazard does not depend on age | des_maintenance | **all pass** | fail on 2/3 seeds | rejected | "discrete" still verified |

(`pg_nolag` also changed dl_delayed_returns by +80% under random play. Its claim still passes, correctly, because
that family's claimed delay lives in a `delay` primitive rather than in an edge lag. I do not count it.)

### S.2 Verdicts I overturn

1. **H0 engines (pg, DES, graph, POSM, search): RETAIN → REPAIR (S).** The auditor's evidence was that unit tests
   pass and families are admitted. Both stay green on broken engines:
   * The engine test file catches 0 of 6 mechanism mutants, and the whole unit suite catches only 3.
   * Admission admits eng_delayed_controller with every delay removed.
   * The `graph_edit` mechanism, links an agent builds, is verified nowhere. Neither `repair_restores`
     (`claims.py:423`) nor the SP-8.11 `eff_graph_edit` (which only checks that an entity `new0` is listed) looks
     at whether a built link carries traffic.

   The engines themselves look correct: the baseline claims pass and each mutant changes outcomes. What is missing
   is engine-level mechanism regression tests. Also, `cx_balancing_loop` (a primitive_graph family) fails its own
   claim.
2. **Structured action interface: RETAIN → REPAIR (S-M).** In the integrated learner path, the interface does not
   work as specified:
   * **Coordination cannot happen.** `RapidExperienceEnv.execute` steps the world once with only the calling
     agent's action (`experience/tier1.py:573`). `coordinate` takes effect only when every party submits the same
     plan in the same tick (`rapid/env.py:238`). Through `crucible.experience`, a0 and a1 submitting the correct
     plan gives `agreed: False` and 0 hits, while the direct `env.step` gives `agreed: True, hit: True` (s05). This
     disables coop_split_info, team_discovery, team_profile and team_schedule.
   * **Results go to the wrong agent.** A non-first agent receives a0's result: `tier1.py:577` reads `out["result"]`,
     which `env.py:208` always fills with a0's result. a1's invalid message comes back `ok: True, {"waited": true}`
     (s10).
   * **Tool discovery is credited vacuously.** `spec_coverage` counts any `tools` query target as tool discovery
     (`actions.py:237`). In 38 of 38 such families, the answer is byte-identical to the tool list the reset
     observation already holds (`env.py:229-233`). Only `srch_tools`' catalogue is real discovery.
   * **Graph modifications are thin.** They are offered by one admitted family with a single op (`add_edge`), and
     their effect on dynamics is untested (mutation above).
3. **4.0 engines and sealed families: RETAIN → REPAIR (S).** The 4.0 engine code is untouched, which I accept. But
   "preserve the sealed evaluation families" is about their integrity as held-out evaluation, and that is not
   preserved in 5.0:
   * All 11 test-split families can be instantiated by any learner through `crucible.experience.make(1,
     "mech:<test family>", seed)`, with shaped reward by default (s08, s08b). Sealing covers seeds only, not
     mechanisms.
   * v5-train now contains analogues of the sealed mechanisms. scm_feedback's own doc says "Both motifs are absent
     from training"; that is false for cx_reactive_history and cx_balancing_loop. scm_regime corresponds to
     ch_sign_reversal, and abm_network_fx to net_diffusion and cxs_lockin.
   * The admission distinctness check against 4.0 is vacuous, as the auditor showed.
4. **Structural generator: EXTEND → REDESIGN (L).** Part 7 calls the generator a central requirement, and the
   current architecture cannot meet it:
   * It composes only `primitive_graph` worlds.
   * It has no objective generation: 69/74 families are `maximize_net`.
   * Its random recipes are admitted at 1/16 (s11; the module's own docstring says 1/40,
     `research/v5/compose.py:26`).
   * The calibrated composer gets past admission by pricing levers so that the best constant is interior
     (`compose.py:19-24`). That defeats the min/mid/max dominance test while keeping a constant optimal: its one
     admitted world is solved by a constant at 0.955.
   * Generated families carry no intervention claims (`compose.py:591`), and their Part 8 category comes from a
     lookup table (`compose.py:53`).
5. **Admission (Part 7.4): REPAIR → REDESIGN (M), scoped to the oracle/reference layer.** `achievable`,
   `no_dominant_strategy` and `difficulty` all share one anchor:
   * For 27/69 admitted families (all six 8.10 `ch_*` families among them), the reference is a constant chosen by
     replaying the scored episode's own noise (`references.py:489,523`). The difficulty check *requires* that
     constant to reach 5-98% of the oracle's gain (`admission.py:48,411`).
   * The dominance check tests only canonical constants (`admission.py:406`).
   * The oracle is an open-loop four-segment schedule that a grid constant can beat.

   Together with the generator calibration, this pipeline selects for static-tuning worlds by construction.
   Bolting on a best-constant baseline does not fix the incoherence. The reference hierarchy needs to be rebuilt:
   a closed-loop privileged upper bound, a best open-loop baseline, an adaptive public-information baseline, and a
   requirement that the adaptive policy beats the best constant. Consistency, conservation, controllability,
   informativeness, leakage and distinctness remain REPAIR. The leak scan gives false positives on legitimate
   consequences as well as missing non-float secrets (s04).
6. **SP-7.4: REPAIR → REDESIGN** (same reason; criteria weak).
7. **SP-8.11: RETAIN → REPAIR; criteria_weak false → true.** The gate counts schema presence plus one plumbing check
   on one host family per type, using direct `env.step` with both agents in one tick. Through the learner-facing
   path, multi-agent coordination fails. The tool-discovery and graph-edit credits are as described in item 2.
8. **SP-8.12: EXTEND → REPAIR.** The axes exist, but in 19 of 35 probed settings turning an axis up does not lower
   the reference's share. The noise axis is invisible to the difficulty measure, because the hindsight reference
   replays the realised noise. Admission's `_axes_effective` (`admission.py:355`) only checks that the config
   digest changes.
9. **M.3: RETAIN → REPAIR (S).** At the RapidEnv level, the count of 10 survives even under the strict filter
   (s09: every category keeps at least 2 families). But category membership is a declared label that nothing
   verifies.
   * In the learner-facing path, 8.9 cooperation collapses: four of its families need `coordinate` (unplayable via
     the adapter), `team_formation` has a single agent, and `team_assembly` fails its claims.
   * 8.8 keeps only 3 of 7 families that pass their own claims.

### S.3 Verdicts I checked and uphold

* **M.4 RETAIN.** The engine modules import no other engine (gate test), and every mutant changes outcomes, so the
  engines carry their own mechanisms. The strictly filtered family set (s14) still spans all 9 v5 engines plus 3
  legacy ones (at least 6).
* **M.6 RETAIN** (criteria weak: it counts schema presence). Through the adapter, levers, discrete, allocate, query,
  graph_edit and message all work for a single agent (s07, s10): 6 or more.
* **H1/H2 engines REPAIR.** The strategic opponents genuinely learn (s12), so REDESIGN is not warranted. The missing
  unit tests and failing claims stand.
* **Fingerprints REDESIGN, M.5 REDESIGN, M.n1 REDESIGN.** Confirmed in source:
  * horizon bucket at `fingerprints.py:104`;
  * scenario label as the topology for non-graph engines at `:130`;
  * duplicates only at distance exactly 0.0 at `:261`.

  These are fingerprints declared from labels, not measured, so REDESIGN is not harsh.
* **Fast batch path EXTEND.** The batch is genuinely vectorised (`batch.py:50-70`). The kernel twin hosts the same
  engine code, so it is consistent by construction. No learner rollout uses the batch, and Tier 3 (civ) is a stub,
  so the twin is not integrated into any world.
* **M.1 and M.2 EXTEND, with corrected numbers.** The effective counts are 30 train and 11 eval (s14), effort L, and
  both depend on the admission redesign. M.2 criteria are weak: 2 evaluation families are one-parameter variants of
  training scenarios (s01).
* **M.7 EXTEND, M.8 REPAIR.** For M.8, the mutation run adds that the claims miss the graph_edit and diagnostic-test
  mechanisms.
* **RapidEnv REPAIR.** Two defects should be added: the multi-agent adapter stepping, and `posm_unlock`'s per-step
  progress `signal` (`state_machine.py:152`), a privileged goal-progress cue under the observation-only amendment.
* **Families EXTEND, primitive registry REPAIR, SP-7.1 REPAIR, SP-7.3 REPAIR and Tier 1 tests REPAIR** (the last is
  strengthened by the mutation results).

### S.4 Revised classification

| Subsystem | Auditor | Skeptic |
|---|---|---|
| Causal primitive registry | REPAIR | REPAIR |
| Structural generator and recipes | EXTEND | **REDESIGN** (L) |
| H0 engines | RETAIN | **REPAIR** (S) |
| H1/H2 engines | REPAIR | REPAIR |
| Family registry and families | EXTEND | EXTEND (effective 30/11) |
| Admission and evaluator side | REPAIR | **REDESIGN** (M; oracle/reference/dominance/difficulty layer) |
| Fingerprints and distinctness registry | REDESIGN | REDESIGN |
| Structured action interface | RETAIN | **REPAIR** (S-M) |
| RapidEnv and observations | REPAIR | REPAIR (+ multi-agent adapter, progress signal) |
| Fast batch path and kernel twin | EXTEND | EXTEND |
| 4.0 engines and sealed families | RETAIN | **REPAIR** (S) |
| Tier 1 unit and gate tests | REPAIR | REPAIR |

Requirements changed: M.3 RETAIN→REPAIR; SP-7.4 REPAIR→REDESIGN; SP-8.11 RETAIN→REPAIR (criteria_weak true);
SP-8.12 EXTEND→REPAIR. All others unchanged. Missing requirements to add:
* the learner-facing multi-agent stepping contract (simultaneous actions);
* engine-level mechanism regression tests (mutation-sensitive);
* a reference that is not a hindsight constant;
* sealed-mechanism access control and contamination;
* the generator's admission yield.

### S.5 Transfer and shortcuts (skeptic's addendum)

The auditor's caution understates the problem. Admission (a constant reference required to be decent, dominance
tested only at min/mid/max) and the generator's lever-price calibration together *select for* worlds whose best
policy is an interior constant. The dominant learnable skill in those worlds is black-box static tuning, which does
not transfer to the open internet.

Cooperation and coordination, the clearest transferable skills in the suite, cannot be practised through the learner
interface. The only demonstration of a neural learner on Tier 1 (PPO through the lever view) trains on legacy
shaped reward. In the strategic, program, search and planning families, the auditor's positive judgement stands.
