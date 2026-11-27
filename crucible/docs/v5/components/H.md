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


## Wave 3b (workstream H, continuation on the integrated tree 55433e7)

Admission (`admission.CHECKS`, H-D1) and the gate tests are unchanged. On the integrated tree the baseline was
22 training and 8 evaluation families robustly admitted (seeds 0-2 and 10-12); `pe_stream_processor` had fallen out
because the D-20 line metering priced its reference program out of its margin. Every family below was re-measured
with the full admission on both seed sets after its change (probes, not evidence).

What made families pass, by kind of fix:

* **Model-based public references** (D-64): `strat_blotto` (opponent values from first-order conditions, averaging
  weight by prediction error), `strat_interdiction` (catch rate by maximum likelihood, the attacker's softmax
  learning fitted to its observed attempts), `strat_bargain` (counterpart weights inferred from offers, aspiration
  read off the offer, type probed by a rejected concession), `pe_contention_sched` (hidden classes by model selection
  over class splits on job speeds under a two-job cap), `pe_stream_processor` (a lean estimator program).
* **Worlds that were static-tuning worlds made adaptive**, versions bumped: `des_bottleneck` v2 (the bottleneck
  moves), `ch_rule_change` and `cxc_competing_rule` v2 (the old lever is penalised after the rule change; reference
  `refs_adapt.drop_switch_tracker`), `cx_latched_pathway` (wide hysteresis band; reference
  `refs_adapt.latch_open_and_hold` finds the latch publicly), `team_assembly` (holding costs), `pe_deadtime_control`
  (faster setpoint schedule, 40 ticks), `strat_interdiction` and `net_repair` (larger defaults).
* **Interventional prediction** (D-63): `hc_conditional`, `cx_competing_causes`, `cx_balancing_loop` v2 score
  predictions of the (settled) response at held-out settings; references `refs_adapt.interleaved_law_predict`
  (cross-validated law library on an interleaved design), `arx_settled_predict` (Hammerstein-ARX steady state);
  `hc_causal_prediction` (interleaved additive fit), `sci_slow_response` (dead time and rate from a step test; the
  science oracle now predicts where slow processes settle - it scored the transient before, -50),
  `sci_selection_bias` (the cut sits inside the settable range; 256 units surveyed so readings are never missing).
* **New families for the four Part 8 items that had none** (D-65): `srch_leads` (8.5.1), `srch_ventures` (8.5.4),
  `plan_commit` (8.6.5), `net_balance` (8.7.5, three operators).

New modules: `refs_adapt.py` (adaptive public references), `claims_w3b.py` (claims, identification, oracles and
references of the redesigned and new families). Engine additions: search scenarios `leads` and `ventures`, planning
scenario `commit`, graph scenario `balance` (multi-participant, team objective), DES `mu_at` (shifting bottleneck),
planning `site_rate` (site demand regimes) and `chain_mu`, router retooling (engine support only).

Tried and not admitted in wave 3b (measured, left as they are or with a better reference): `plan_distribution`
(site demand regimes, low-margin terms: the reference reaches 0.5-0.9 of the hindsight constant's ratio threshold on
some seeds only), `plan_supply_chain` (second demand shift, holding costs), `plan_perishable`, `plan_portfolio`
(reference now splits staff; passes seeds 0-2 only), `pe_router` (retooled lines: the law-aware router is not an
upper bound under higher loads), `ch_drifting_ideal` (narrow drifting peak: threshold rules on a reading track it),
`ch_sign_reversal`, `cxc_latched_reversal`, `cxs_lockin`, `strat_price_war`.

## Wave 3b continuation (after the usage-limit stop)

Admission (`admission.CHECKS`, H-D1) and the gate tests are unchanged, except that the D-9 recreation check now
ignores journal filings in prediction worlds (`fingerprints.legacy_recreation`: a lever world with a journal still
recreates a lever-interfaced 4.0 mechanism). Every family below was re-measured with the full admission on seeds 0-2
and 10-12 after its change (single-CPU probes, not evidence). Probe counts: 54 training and 21 evaluation families,
54 of the 60 named Part 8 items.

Two patterns made families pass:

* **Hidden events per component** (D-64 amendment), for families whose every hidden quantity was fixed, so the
  hindsight-best constant was the oracle: `strat_entry` v2 (incumbent takeovers; planned-entry oracle by dynamic
  programming; reference that reads price levels, the price of fights and moved levels), `team_commons` v2 (bad
  spells; escapement reference), `team_discovery` v2 (ingredient depletion; rotation by expected current yield),
  `team_formation` v2 (specialists and a pay budget; the reference never read claimed skills - fixed),
  `team_schedule` v2 (job bursts; the shared plan used one participant's speed estimates - fixed),
  `plan_portfolio` v2 (projects opening at hidden times; probe-and-audit reference), `pe_cascade_guard` v2 (capacity
  factors that differ between units; per-unit safe exploration), `net_diffusion` v2 (communities coming online;
  incremental seeding).
* **Prediction of the interventional or end-of-episode response** (D-63 amendment), for lever worlds where a
  constant or a two-setting threshold rule tracked a one-off change: `hc_confounded`, `hc_feedback_hidden`,
  `ch_constraint_tightens`, `ch_demand_shift`, `ch_sign_reversal`, `cxc_balancing_constraint`,
  `cxc_latched_reversal`, `ch_drifting_ideal`, `cx_overdose`, `cx_reactive_history`, `game_adaptive_rival` and the four
  look-alike markets (one interface for all four, all admitted). References in
  `refs_adapt`: `fir_shape_predict` (randomised design, one shape per setting by cross-validation, exogenous
  readings as covariates), `fir_shape_predict_recent` (change point), `market_predict` (log-additive law against a
  logit demand curve, by cross-validation), `drifting_bump_predict`, `held_dose_predict` and `held_hub_predict` (hold
  until the delayed effect arrives; peaked dose law or one saturating channel, saturating output link),
  `rival_model_predict` (a rival that follows your remembered average price, logit demand).

Not admitted, redesigns reverted: `ch_tech_transition` (its prediction version passed admission but its best
constant does not change after obsolescence, failing the 8.10 requirement of SP-8.0b), `plan_supply_chain` (demand regimes), `net_routing` (capacity incidents with an AIMD
reference), `net_cascade` (zone heat waves), `strat_price_war` (two learning rivals), `des_inventory` (strong
seasonality), first attempts at `ch_drifting_ideal`. Untouched and not admitted: `dl_short_term_damage`,
`eng_delayed_controller`, `pe_rollout_targeting`, `plan_distribution`, `plan_perishable`, `des_perishable`,
`net_topology_shift`, `pe_router`, `cxs_lockin`, `cxs_selected_layers`, `ch_wandering_taste`.

## Wave 3b second continuation (2026-11-27)

Admission and the gate tests are unchanged. Every change below was re-measured with the full eleven-check admission on
seeds 0-2 and 10-12 (single-CPU probes, not evidence); the registry-wide run is the slot job `H: rapid final 2`.
Probe counts: 63 training and 22 evaluation families, all 60 named Part 8 items; the six SP-8.0b category checks pass
on the probe-admitted families.

* **Incompetent references fixed (worlds unchanged):** `plan_distribution` (reference and oracle bought stock that
  could not be sold before the end) and `des_inventory` (an order every tick against a fixed order cost; now an
  (s, Q) policy - demand = sales + customers turned away, lead time from deliveries, economic order quantity; the
  oracle adds (s, Q) policies with the true terms). Regression tests: `tests/test_v5_rapid_inventory_refs.py`.
* **Category requirements:** `plan_distribution` accepts a replenishment/shipping program (8.4 needs programmatic
  actions); `lab_drift_blanks` scores only doses above its settable range (8.2), with a reference that measures more
  at the top of the range.
* **Prediction tasks (D-63, version bumps):** `ch_tech_transition` v3 (the new process loses money while it is
  learned, so the early-best constant fails after obsolescence: 8.10 check passes), `eng_delayed_controller` v2
  (settled temperature of a delayed first-order plant above the settable range; control of 8.3.3 stays with
  `pe_deadtime_control`), `dl_short_term_damage` v2 (output after forty ticks of constant drive; wear visible on a
  noisy vibration reading). References `refs_adapt.learning_transition_predict`, `delayed_plant_predict`
  (`step_hold_design`, `first_order_fit`) and `wear_latch_predict`.
* **New compact primitive-graph families for items no family covered (D-65 amendment):** `dl_delayed_adoption`
  (8.6.3), `plan_spoilage` (8.4.2), `plan_serial_stages` (8.4.3), `net_congestion_routes` (8.7.1), all training, and
  `net_interdependent` (8.7.4, evaluation). Each has an intervention claim and an identification question in
  `claims_w3b`, a model-class reference in `refs_adapt` and a fidelity record. Stock-based ones bound their
  saturating outflows so that no stock ever gives up more than it holds (tested). Two pair protocols in `distinct`
  verify differences to same-interface nearest families. `cascading_failure` now broadcasts per-instance parameters on
  the batch path.
* **Tried and not adopted** (the worlds are unchanged and not admitted): demand regimes and holding costs for
  `plan_supply_chain`, demand regimes for `plan_perishable`, a dynamic-pricing seats world (`plan_seats`, simple
  pacing rules came within 0.9 of the oracle), readiness-aware targeting references for `pe_rollout_targeting`, a
  model-predictive controller for the control version of `eng_delayed_controller`, a lot tracker for `des_perishable`.

## Wave 3b repair pass (2026-11-27, after the verifier's review of c09eb24a)

What the verifier found and what changed (each with a regression test in `tests/test_v5_rapid_repair.py` or an
updated test; decisions in the D-63 and D-64 amendments of the repair pass):

* **Prediction dominance was too weak.** `dominance._naive_predictors` now also runs *hold and fit*: the corners and
  middle of the settable box (`hold_settings`) held for equal blocks, the scored reading averaged over the last third
  of each block, a line, an additive quadratic and their leave-one-out choice extrapolated to the public held-out
  list (labels `naive_prediction:hold_linear|hold_quadratic|hold_cv`). SP-7.4 gained a prediction negative control
  (`ctl_dominant_prediction`, a delayed straight line: random designs miss it, hold and fit answers it, so it is
  rejected by `no_dominant_strategy` with a hold arm as the best simple policy). Families that this dominated were
  redesigned (held-out conditions where the mechanism decides the answer; better references), not the check:
  `plan_spoilage`, `net_congestion_routes` (capacities inside the settable range; reference designs at heavy flows,
  BPR exponent by nonnegative least squares), `cx_balancing_loop` (reference `held_hub_predict`), `ch_tech_transition`
  (conditions weighted to the obsolete process), the four look-alike markets (one shared list at price and promotion
  extremes). `mk_drift` and `cxc_balancing_constraint` stay dominated (not admitted).
* **Program worlds were not deterministic.** Compute is billed per call in whole quanta of the parent's measurement
  (`program_exec.billed_units`: 100 units per call, plus 10,000 per whole 10,000 measured); references measure at most
  ~2,600 units per call, so outcomes repeat exactly in fresh processes (`CRUCIBLE_METER_MEMO=off`, tested).
* **8.10 on the scored channel.** `categories.beliefs_fail_after_change` (used by `constants_fail_after_change` for
  primitive-graph prediction families): the truth for the world as it stood at the end of the first fifth
  (`true_response(..., t_frozen, exogenous_ticks)`: timed regimes frozen, exogenous stocks held) filed as predictions
  must fall short of perfect by at least a quarter. A lever cost no longer decides it (mutation test).
* **Revenue bypassed observation noise.** Prediction worlds that book revenue book realised quantities
  (`families._realised_revenue`, relative sd 0.1 x observation_noise; one extra normal draw per tick after all others,
  none in the warm-up; the scalar twin follows the same contract).
* **Smaller items.** `claims_w3b.claim_spoilage` requires the late rise to be at most 0.8 of the early one (a linear or
  barely concave sales law fails); the inventory worlds state their trading terms in the brief and
  `references.inventory` reads them there (`public_inventory_terms`).
