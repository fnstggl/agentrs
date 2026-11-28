# RW — genuinely learned rewards (operator amendment of 2026-11-28, item 1)

Owner of `src/cam_live/reward/**` (except `selfgoals.py`) and `tests/reward/**` for this round; start d1d06dc; everything
was measured on isolated copies of the tree (HEAD dc1753c or 07303fe + RW's `reward/` and `tests/reward/` files;
other agents' uncommitted work not included). Commit of this code: 2dfa6db. Every
number below is from a file under `docs/evidence/RW/`, a test named here, or a read-only reading of the live planning
run named as such. Real configuration = Qwen3-4B (frozen, bf16) + LoRA + latent bridge + 50m core, backbone proposals
on, under the shared GPU lock. Debug = tiny random backbone + debug core on CPU: engineering diagnostics only, never
reported as the real model. Change requests for `agent.py`: `docs/change_requests/RW.md` (two small reporting hooks;
the mechanism itself needs none).

## 0. Summary

Item 1 asks that R_psi can be learned and *used* inside one long objective, on an empirically grounded criterion,
without forcing the mix. What changed and what it showed:

| | before (d1d06dc) | now |
|---|---|---|
| when R_psi gets weight | held-out value comparison of two single policy steps, only at applied meta-updates, 4 groups per decision, buffer cleared on hold, hand-set non-inferiority margin: ~3,400 groups for 80 % power at equal quality | prospective paired agreement of each reward with the decisions' later verified progress, every matured decision is evidence, pooled over the objective, anytime-valid e-processes (alpha 0.05), no margin; median 18-52 pairs when R_psi orders 100-80 % of informative pairs correctly |
| what the outer gradient uses | the 3-4 decisions between two live updates (per-episode ESS floor; 25 % of groups held out) | `auto`: every matured decision of the objective outside the update's support (first-order in the step, exact through it); query windows as fallback; nothing held out |
| outer utility | return-to-go (dominated by later decisions' outcomes) | `direct` (the decision's own signed outcomes) |
| level metrics (member count, sessions) | the reading's level was the decision's utility (defect) | the change since the latest earlier reading |

Evidence (details §3-§6): (a) 168 tests pass in `tests/reward` (21 new), the pooled gradient equals finite
differences, the e-processes hold their error level in simulation. (b) The integrated agent runs it by default;
the live run's newest checkpoint loads (validity state reported fresh). (c)/(d), debug scale, one LabWorld objective
per seed, controlled configuration (live update learns only from the mixed reward), same seeds against a frozen
R_psi: meta-learning raised R_psi's prospective agreement with verified outcomes above the frozen R_psi's in 17 of
18 seed-configurations; in 600-decision objectives the evidence handed it weight in 3 of 3 seeds at the reference
reward step 3e-3 and 2 of 6 at the agent's 1e-4 (after 281-568 decisions; frozen arms never, 12 objectives); the
decisions changed 7-31 decisions later. Afterwards the chosen actions' true value exceeded the bootstrap-only
counterfactual's clearly in one objective (+2.14 adopters per decision [1.63, 2.64], 319 decisions) and was
inconclusive in the four with 32-69 decisions left (-2.89 .. +0.87, every interval including 0): improvement is not
established. In the integrated agent's own configuration (debug, 120 decisions) the evidence did not move the mix.
Real configuration (Qwen3-4B + 50m, backbone proposals on, one long staging objective, one GPU hold): 35 decisions, 4 of 36
meta-updates applied to R_psi from verified outcomes, 17 validity pairs, e-value 1.0 — the mix stayed 1 and R_psi had
no influence on any of the 11 live updates (TV 0.0); the pathway is shown up to the evidence, not beyond it.
Diagnosed limits (§4, §6): the outer utility (fixed: `direct`) and the inner-loss return half-life (1 day against
decisions an hour apart; not changed — it governs every live update) blur the credit R_psi receives; the anytime test
needs tens of informative pairs, i.e. ~100+ matured decisions, more than one GPU hold provides; the planning phase of
the live run has no verified contrast at all yet.

## 1. Audit — why the bootstrap mix stayed at 1

The reward network received meta-updates, but no learned reward ever shaped an ordinary live update under the default
configuration. Five causes, measured (old code = d1d06dc):

| # | cause | measurement |
|---|---|---|
| A1 | **The promotion test compared the value of two single policy steps.** After an applied meta-update, each held-out group contributed the clipped importance-weighted verified utility of the learned-reward step minus the bootstrap step, from the same snapshot. A trust-region step changes the policy by KL ~0.05, so the true difference is small against the noise of one importance-weighted outcome: the one real-model decision had per-group SD 1.17 x RMS(U); ~3,400 groups give 80 % power at no effect, 44 at 0.4 RMS (`docs/INT3A.md` §6). | INT3A §6 / `docs/evidence/INT3A/promotion_power.json` |
| A2 | **Promotion evidence existed only at applied meta-updates** (the scarcest event), 4 held-out groups per decision, buffer cleared on every hold, a non-inferiority margin set by hand (0.05 RMS). | debug long objective (below): 6 held-out groups accumulated, 0 promotion decisions in 48 meta-update calls |
| A3 | **The outer estimator used only the decisions between two consecutive live updates** (3-4 query rows per episode); the per-episode effective-sample-size floor then failed whenever one importance weight dominated, and the rows were consumed. 25 % of all groups were withheld for A1. | debug, one 40-decision LabWorld objective through the runtime (`docs/evidence/RW/probe_before.json`): 48 calls, 2 applied, 41 `insufficient_independent_outcomes`, 3 `effective_sample_size_too_small` — every time 4 groups existed, 3 of 5 attempts failed the ESS floor |
| A4 | **The unit of independence was not the bottleneck.** Inside one objective the verifier attributes outcomes per decision: in that run 40 of 40 decisions had a final verified progress (31 non-zero). The evidence existed; A1-A3 discarded or could not use it. | same file |
| A5 | **Defect: readings of the objective's level metric supervised psi with the level itself.** In environments that do not attribute outcomes (Discord member count, the utility objective's session count) every decision carries the question "objective metric after the action"; the ledger stored the measured *level* as that decision's utility (and the return-to-go summed levels), so a decision late in an objective was credited with all progress before it — an unbiased but very high-variance signal. | code reading `reward/integrity.py:measurement_utility`, `metalearner._default_measurement_utility`; regression test below |

The live planning run (read-only, `runs/cam_live/live/discord-100-planning-20261128T083454Z`, 09:28 UTC): 36 decisions,
35 meta-update calls, 33 skipped `insufficient_independent_outcomes:5<8`, 2 before the first policy update; the
ledger holds 11 verifier readings of the member count, all 0, and one objective outcome of utility 0. In the planning
phase nothing verified varies yet — no reward can be learned from it, and with the new code nothing is (no
informative pair, zero advantage). That is the correct behaviour, not a defect.

## 2. What changed (all in `src/cam_live/reward/`)

### 2.1 When R_psi is used: prospective, anytime-valid validity evidence (`validity.py`, `bootstrap.py`, `metalearner.py`)

Why this quantity. A policy-gradient step with a standardised reward r moves the policy along
E[grad log pi(a|s) (r(s,a) - b(s))]; to first order the verified value changes by alpha <grad J, step>, which for a
softmax policy is a *within-state covariance between r and the verified advantage*. The reward that improves verified
value more is, to first order, the one whose ranking of decisions *taken in the same situation* agrees better with
their verified consequences. Two decisions taken one after the other share most of their state.

* **Unit**: one independent outcome group (`windows.independence_key`: experiment, parent decision, or the decision),
  once every decision of it has its final verified progress (`inner.verified_progress`: written by the agent from
  signed verifier readings only). A group counts once.
* **Pair**: the next two units that mature in the same objective segment *and were rewarded by the same psi* (no applied meta-update between them; the input statistics of R_psi move every
  decision and are not counted)
  (disjoint, in maturity order; the pairing never looks at outcomes; a unit whose partner-to-be was rewarded by another
  version is left unscored: `validity_same_version`, default on). Disjointness matters: overlapping pairs share a unit,
  and the shared unit biases both signs. The null is exact when the two units of a pair are exchangeable given the
  past; what can break that is a common drift of R_psi's output and of the verified outcomes over the few decisions
  between them — a meta-update in between is the large one (it shifts R_psi's output while the policy, and with it the
  outcomes, improve), hence the version rule; the slow drift of R_psi's lifetime context remains. Frozen R_psi (no
  version change, same drift of the context, improving policy) never showed positive agreement (-0.22 .. 0.00).
* **Score** for a bootstrap weight m: c_m = sign(Delta r_m) sign(Delta y), r_m = m r_b / s_b + (1 - m) r_l / s_l (s:
  running scale of each reward), so the score is invariant to R_psi -> a R_psi + b (a > 0), like the inner update.
* **Prospective**: r_b and r_l are the values both rewards gave the decision when it was rewarded — before any of its
  outcomes existed, with the psi of that moment; never re-scored by a later psi. Evaluation precedes training
  (prequential), so no group has to be withheld from the meta-update (`heldout_fraction` now defaults to 0 under this
  rule; 0.25 under the old one).
* **Tests**: at every grid point m_k of {1, 0.75, 0.5, 0.25, 0} two increments, x+ = (c_0 - c_mk)/2 ("R_psi alone
  agrees better than the mixture in force") and x- = (c_1 - c_mk)/2 ("the bootstrap alone agrees better"), each with
  a one-sided sequential test of H0: E[x | past] <= 0 — a test supermartingale with predictable aGRAPA bets
  (`BettingTest`); Ville's inequality makes it valid at every time simultaneously, so the evidence pools over the
  whole objective, is looked at after every decision and needs no fixed sample size.
* **Decision** at mix m_k: *raise* if the bootstrap alone is shown to agree better (wealth >= 2(G-1)/alpha), *lower*
  if R_psi alone is shown to agree better than the mixture (wealth >= 2/alpha), else *hold*. The mix can stand at
  m_{k+1} only after the x+ tests at m_0..m_k were rejected (fixed-sequence procedure): P(any lowering although R_psi
  alone does not agree better) <= alpha/2; raises are Bonferroni-corrected: P(any wrong raise) <= alpha/2. alpha = 0.05.
  No margin, threshold or mix value is set by hand (constants: alpha, the grid = the old mixer step 0.25, bet cap
  0.75). A reward merely as good as the bootstrap is not promoted; an uninformative one (random init, frozen psi) is
  not promoted; a harmful one is reverted. Reported interval per comparison: predictable plug-in empirical-Bernstein
  confidence sequence (`ConfidenceSequence`).
* **Where**: fed in `MetaLearner.internal_rewards` (both raw rewards of every new decision) and
  `refresh_validity()` (called from `internal_rewards` and `meta_update`: every decision whose verified progress
  became final); the decision moves the mix through `RewardMixer.apply` (logged like the old rule; "lower" after an
  applied meta-update counts as a promotion, label `mixed` / `learned`). `MetaConfig.promotion_rule = "validity"`
  (default) | `"value"` (the LRN rule, kept). A fixed mix (`mixer_step` 0, condition E_mix0) never moves.
* **Persistence**: `MetaLearner.state_dict()["validity"]`; a checkpoint without it (every checkpoint written before
  RW, e.g. the live run's) loads, starts the evidence empty and says so in `MetaLearner.restored_fresh`.

### 2.2 How R_psi learns inside one objective: the pooled first-order outer estimator (`metalearner.py`)

`MetaConfig.estimator = "pooled_first_order"` (and `"auto"`, the new default: pooled for the newest live update event,
the old query-window estimator when the pool has no evidence — one record either way):

    J(psi) ~= J(theta_0) + < grad_theta J(theta_0), theta_K(psi) - theta_0 >,   grad_psi J ~= (d theta_K / d psi)^T g
    g = grad_theta sum_j w_j log pi_theta_0(a_j) (U_j - b_j)

theta_K(psi) is the exact differentiable K-step inner update of the newest live update event (its snapshot, support
batch, step, K, at `meta_mix`); g is the importance-weighted policy gradient of verified utility at the snapshot
(self-normalised, clipped weights, logged propensities, ESS floor, leave-group-out baselines) from **every matured
decision of the objective outside the event's support and outside every outcome group of the support** (the
`pool_size` = 64 most recent; utilities from the hash-chained ledger exactly as before). No outcome is in both the
inner update and the outer estimate. The only approximation is linearising J at theta_0 instead of evaluating it at
theta_K (the live step is a KL ~0.05 trust region). Each event is stepped once. It does not wait for the few decisions
between two updates, for another lifetime or for an objective to end. The old estimator stays (`"query"`).

**Outer utility default `"direct"`** (was `"return_to_go"`): a pool or query decision is credited with its own signed
outcomes (attributed measurements, its level-change reading, objective outcomes credited to it). In a long objective
the return-to-go of a decision is dominated by the outcomes of the decisions after it (section 4 measures this); its
bias is that delayed consequences that the verifier does not attribute to the decision are not credited to it in the
outer objective (they still reach the live policy through the goal term's return).

### 2.3 Level readings are changes (`integrity.py`, `metalearner.py`) — defect A5 fixed

`OutcomeLedger.admit_measurement(..., level_key=)` derives the utility of a reading of an objective's level metric as
its change since the latest earlier admitted reading of that metric (`LEVEL_CHANGE`, raw reading kept in the hashed
entry note): first reading anchors (0), a reading observed no later than the latest one lies in an interval already
credited (0), so an objective's changes sum to last minus first reading and no progress is counted twice.
`MetaLearner.register_objective` remembers the task's success metric; `admit_measurements` admits its readings this
way (other measurements unchanged). The level state is rebuilt from the ledger file on restart.

### 2.4 ST-3 applied: realised information value as a bootstrap input (`assemble.py`, `inner.py`, `bootstrap.py`)

ST's change request (`docs/change_requests/ST.md` ST-3), reviewed and applied as proposed: the agent's annotation
`reward_signals.information_value` (the world model's loss on the next two decisions' matured outcomes with the
decision's context items removed minus with them; same weights, same targets; nats) enters the bootstrap reward as
`w * tanh(IV)`, `w = 1`, through the bootstrap-only extras path. Review:

* **Sound as a learning signal.** It is a measured, signed realised quantity, not a bonus for any kind of action: it is
  positive only when what an action brought into the context made later outcomes more predictable, negative when it
  misled. Irreducible randomness in a result cannot make *other* outcomes predictable, so it earns ~0 in expectation
  (no noisy-TV loop; the gate is not needed). Limits: it credits information about the next two decisions' outcomes
  only (myopic); it is 0 while the world model ignores context; a result that copies a later measurement counts as
  informative (genuinely, but it can favour echoing). The verified outcomes, the goal term and the validity evidence
  are unaffected by it.
* **Weight 1.** The bootstrap is, by design (spec 7.6), a fixed labelled stabiliser with hand-set component weights;
  IV is on the scale of learning progress (tanh of a loss difference in nats), so weight 1 puts the two on one scale,
  and the inner update standardises the bootstrap total, so only the relative weights inside the bootstrap matter.
  Learning how much it deserves is R_psi's job: IV is *not* an R_psi input here (signal schema and R_psi's shapes
  unchanged, so checkpoints stay compatible). Exposing it to R_psi — a new signal column whose input weights start at
  zero when an older checkpoint is loaded — is the principled next step; not done in this round.
* **Interaction with RW's changes.** IV is measured only after the next decisions' outcomes matured, so it is absent
  from the bootstrap reward *at reward time* — the reward the validity evidence scores prospectively — and present in
  the live updates of matured decisions (support and replay). The promotion criterion therefore neither gains nor
  loses from it; at mix 1 an information-only action can now be reinforced by the live update.
* **Test**: `tests/reward/test_reward_rw.py::test_realised_information_value_is_a_bounded_bootstrap_input_and_not_an_
  input_of_r_psi` (component = tanh, bounded, reaches the inner update's bootstrap reward for annotated rows and 0
  elsewhere, R_psi's signal vector and parameter shapes unchanged).

## 3. Evidence, by capability and level

Levels (operator item 8): (a) implementation correctness, (b) integration into the actual runtime, (c) causal
influence on behaviour (ablation, same seeds), (d) measured improvement — or a negative result with its diagnosis.

### 3.1 Validity evidence and the mix decision (`reward/validity.py`, `RewardMixer.apply`, `MetaLearner.refresh_validity`)

(a) `tests/reward/test_reward_rw.py`: the confidence sequence covers the mean at all times (300 runs x 150 steps, 3
means, ever-excluded <= alpha + MC error); the betting test's type-I error at the null boundary <= alpha (400 runs x
200 steps) and it finds a mean of 0.4 (median 60 observations, measured); pair scores ignore ties and the scale of
either reward; an informative R_psi is lowered step by step and never raised, with identical decisions under
R_psi -> 7 R_psi - 3; an uninformative one is never promoted in 150 pairs; a harmful one is reverted; groups count
once, pairs never cross segments, scores use the values noted at reward time, a decision whose outcome was known at
reward time is not a unit; state round trip continues bit-identically; a mix grid mismatch is refused.
(b) the agent's `MetaLearner` is built with these defaults (`promotion_rule="validity"`); the rewards of every
decision are noted in step 13 and resolved in steps 13/15 (`docs/evidence/RW/real_*.json`, `dbg_*.json`: `validity`).
(c)/(d) section 3.4.

### 3.2 Pooled first-order outer estimator (`MetaLearner._pooled_meta_update`)

(a) `test_pooled_first_order_meta_update_moves_only_psi_along_the_first_order_outer_gradient`: the applied gradient
equals central finite differences of the first-order surrogate <g, theta_K(psi)> along a random direction (5 %),
a step along it raises the surrogate, theta is untouched, no pool row is in the support, each event is stepped once;
`test_pooled_estimator_refuses_without_enough_independent_verified_groups`: refusal leaves psi bit-identical.
(b)-(d): section 3.4 and the diagnostics below.

### 3.3 Level readings as changes (`OutcomeLedger.admit_measurement(level_key=)`)

(a) `test_readings_of_the_objective_level_are_admitted_as_changes_never_twice` (anchor 0, +2, flat 0, late reading 0,
+4; sum = last - first; a caller-stated level is refused; the level state is rebuilt from the file),
`test_meta_learner_admits_goal_metric_readings_as_level_changes` (readings 4, 9, 9 of the task's success metric ->
utilities 0, 5, 0). On the old code the first test fails (no `level_key`; the stored utility was the level). (b) the
agent registers every assigned task with the meta-learner, so its success metric is the level metric. (c)/(d) not
measured: no objective with a moving level metric was run (the live planning run's readings are all 0).

### 3.3b The integrated agent through the runtime (debug scale)

`docs/evidence/RW/rw_pathway.py --debug --arm learned --decisions 120` (`CamLiveAgent` + `LiveRuntime`, tiny backbone
with backbone proposals on, `thompson_policy`, the agent's own configuration: trust-region live step, matured replay,
goal term, reward step 1e-4; one long LabWorld staging objective; `dbg_agent_learned_120.json`): 120 decisions, 143
meta-update calls, **9 applied** (3 pooled first-order, 6 query-window importance-weighted through the `auto`
fallback; 111 `insufficient_independent_outcomes`, 21 ESS below the floor — the pool is empty while the support
covers the newest ~28 decisions, §6), first applied at decision 33; **47 scored validity pairs**; R_psi's agreement
0.02 [-0.48, 0.53], the bootstrap's -0.32 [-0.80, 0.17]; e-value of "R_psi alone agrees better than mix 1" 0.37
(threshold 40): the mix stayed 1, so R_psi had weight 0 in all 37 live updates (recomputing each at mix 1 changes the
policy by TV 0.0). Every link up to the evidence runs in the integrated agent; the weight never moved because the
evidence did not support it.

### 3.4 The pathway inside one long objective, with the ablation (debug scale)

`docs/evidence/RW/rw_meta_diag.py` (debug model; `MetaLearner` acting with its own live policy in ONE LabWorld
objective of 300 or 600 decisions — target never reached, budget and deadline beyond the run; a live policy update
every 6 decisions; a meta-update attempt and the verified-progress annotation after every decision; the controlled
configuration of `reward.diagnose.make_stack`, in which the live update learns only from the mixed reward — fixed
inner step, no goal or model term — and the reward step is the stage-1 reference's 3e-3 unless stated; the agent's
is 1e-4). Arms, same seeds and worlds: `frozen` (reward step 0: R_psi stays at its initialisation), `pooled`
(estimator `pooled_first_order`) with `direct` or `return_to_go` outer utility, one arm at lr 1e-4, one pair
(pooled + frozen) with a 600 s inner-loss half-life. Summaries `docs/evidence/RW/diag_summary_300.json`,
`diag_summary_300_halflife600.json`, `diag_summary_600.json` (`summarise_diag.py`, `table_diag.py`); truth = expected
adopters of the chosen action (hidden rate x effort, never visible to the learner); "agreement" = mean pair score
c_0 of R_psi with its 95 % confidence sequence; "gain" = agreement minus the frozen R_psi's on the same decisions.

| arm (outer utility, inner half-life, reward lr) | seed | decisions | meta-updates applied | learned agreement, thirds of pairs | agreement overall [95 % CS] | gain over the frozen R_psi (same pairs) | first mix change (decision) | mix at end | first decision unlike frozen | truth of chosen actions after the change, minus frozen [95 % bootstrap over decisions] |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled (direct, half-life 86400 s, lr 0.003) | 0 | 300 | 49 | 0.06 / 0.29 / 0.53 | 0.30 [0.04, 0.48] | 0.41 | 252 | 0.75 | 269 | 0.93 [0.23, 1.83] |
| pooled (direct, half-life 86400 s, lr 0.003) | 1 | 300 | 48 | 0.14 / 0.24 / -0.10 | 0.09 [-0.11, 0.35] | 0.09 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.003) | 2 | 300 | 49 | 0.06 / -0.08 / 0.16 | 0.05 [-0.18, 0.24] | 0.27 | - | 1.00 | - | - |
| pooled (return_to_go, half-life 86400 s, lr 0.003) | 0 | 300 | 49 | -0.02 / 0.08 / 0.31 | 0.13 [-0.14, 0.32] | 0.24 | - | 1.00 | - | - |
| pooled (return_to_go, half-life 86400 s, lr 0.003) | 1 | 300 | 49 | 0.02 / -0.16 / 0.02 | -0.04 [-0.28, 0.20] | -0.04 | - | 1.00 | - | - |
| pooled (return_to_go, half-life 86400 s, lr 0.003) | 2 | 300 | 49 | 0.06 / 0.00 / 0.08 | 0.05 [-0.15, 0.26] | 0.27 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 0 | 300 | 49 | -0.27 / 0.04 / 0.20 | -0.01 [-0.27, 0.18] | 0.11 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 1 | 300 | 48 | 0.14 / 0.20 / -0.02 | 0.11 [-0.11, 0.35] | 0.11 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 2 | 300 | 49 | -0.22 / -0.20 / 0.16 | -0.09 [-0.32, 0.08] | 0.13 | - | 1.00 | - | - |
| query (direct, half-life 86400 s, lr 0.003) | 0 | 300 | 22 | -0.18 / -0.08 / 0.00 | -0.09 [-0.33, 0.12] | 0.03 | - | 1.00 | - | - |
| query (direct, half-life 86400 s, lr 0.003) | 1 | 300 | 28 | -0.10 / 0.16 / -0.06 | 0.00 [-0.24, 0.24] | 0.00 | - | 1.00 | - | - |
| query (direct, half-life 86400 s, lr 0.003) | 2 | 300 | 23 | -0.06 / 0.32 / 0.12 | 0.13 [-0.09, 0.32] | 0.35 | - | 1.00 | - | - |
| pooled (direct, half-life 600 s, lr 0.003) | 0 | 300 | 49 | 0.06 / 0.39 / 0.00 | 0.15 [-0.07, 0.41] | 0.11 | 221 | 0.75 | 229 | -0.05 [-0.79, 0.75] |
| pooled (direct, half-life 600 s, lr 0.003) | 1 | 300 | 48 | -0.10 / 0.18 / 0.45 | 0.18 [-0.13, 0.36] | 0.20 | - | 1.00 | - | - |
| pooled (direct, half-life 600 s, lr 0.003) | 2 | 300 | 49 | 0.10 / 0.14 / 0.24 | 0.16 [-0.07, 0.36] | 0.36 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.003) | 0 | 600 | 99 | 0.21 / 0.66 / 0.42 | 0.39 [0.17, 0.49] | 0.45 | 281 | 0.75 | 301 | 2.14 [1.63, 2.64] |
| pooled (direct, half-life 86400 s, lr 0.003) | 1 | 600 | 98 | 0.17 / 0.17 / 0.24 | 0.19 [-0.01, 0.35] | 0.15 | 537 | 0.75 | 547 | -0.14 [-1.56, 1.31] |
| pooled (direct, half-life 86400 s, lr 0.003) | 2 | 600 | 99 | -0.06 / 0.15 / 0.35 | 0.15 [-0.05, 0.25] | 0.35 | 531 | 0.75 | 538 | -2.89 [-8.07, 2.12] |
| pooled (direct, half-life 86400 s, lr 0.0001) | 0 | 600 | 99 | -0.08 / 0.32 / 0.28 | 0.17 [-0.07, 0.27] | 0.24 | 533 | 0.75 | 564 | 0.70 [-0.07, 1.55] |
| pooled (direct, half-life 86400 s, lr 0.0001) | 1 | 600 | 98 | 0.05 / 0.16 / 0.34 | 0.18 [-0.04, 0.32] | 0.14 | 568 | 0.75 | 578 | 0.87 [-0.04, 2.50] |
| pooled (direct, half-life 86400 s, lr 0.0001) | 2 | 600 | 96 | -0.14 / 0.04 / 0.19 | 0.03 [-0.16, 0.14] | 0.23 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 3 | 600 | 99 | 0.08 / 0.01 / 0.07 | 0.05 [-0.13, 0.24] | 0.11 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 4 | 600 | 98 | -0.14 / -0.13 / 0.12 | -0.05 [-0.25, 0.07] | 0.16 | - | 1.00 | - | - |
| pooled (direct, half-life 86400 s, lr 0.0001) | 5 | 600 | 99 | -0.13 / -0.25 / -0.19 | -0.19 [-0.36, 0.01] | -0.04 | - | 1.00 | - | - |

The 300-decision rows (and the credit probe) ran with the first version of the pairing, before the same-psi rule
(§2.1) existed, on an isolated copy of dc1753c; they compare the components (estimator, utility, half-life, reward
step). The 600-decision rows run the final rule on a copy of 07303fe, each against a frozen arm of the same code and
seeds (same decisions up to the first mix change: checked). Without the same-psi rule the same 600-decision
lr-3e-3 objectives promoted in 2 of 3 seeds, at decisions 252 and 573 (`diag_long600_pooled_direct_no_version_rule.json`).

Frozen arm (R_psi at its initialisation, same seeds): prospective agreement -0.11 / 0.00 / -0.22 (300 decisions) and
no mix change in any of 12 frozen objectives (3 seeds x 300 at both half-lives, 6 seeds x 600; 600-decision frozen agreement -0.21 .. +0.04). `query / direct` (the old
estimator with the new utility), 3 seeds: 22-28 applied steps per 300 decisions (pooled: 48-49), agreement gain over the
frozen R_psi +0.03 / 0.00 / +0.35 (pooled: +0.41 / +0.09 / +0.27), no mix change — the reason `auto` tries the pooled
estimator first.

Reading, per level:

* **(b) the real code path.** Every decision's two rewards were noted when it was rewarded and scored once its
  verified progress was final (149-150 pairs per 300 decisions, 83-108 informative); every live update event got one
  pooled outer step (48-49 applied per 300 decisions, 96-99 per 600); nothing was held out of the meta-update.
* **(c) learning changes what R_psi rewards, and only evidence lets it act.** On the same decisions, the meta-learned
  R_psi agreed with the verified outcomes better than the frozen one in 17 of 18 seed-configurations with `direct`
  (300 decisions: 3 seeds each at lr 3e-3, lr 1e-4, half-life 600 s; 600 decisions: 3 seeds at lr 3e-3, 6 at lr 1e-4;
  gains -0.04 .. +0.46) and in 2 of 3 with `return_to_go`. While the mix is 1, R_psi has weight 0 in the live update
  and every learned arm took exactly the same decisions as its frozen arm. Where the evidence lowered the mix, the
  decisions stayed identical for 7-31 more decisions — until a live update that used R_psi had changed the policy —
  and then differed. No frozen arm ever moved the mix (12 objectives).
* **(d) improvement — not established.** 600-decision objectives with the final rule: the evidence promoted R_psi in 3
  of 3 seeds at the reference reward step 3e-3 (decisions 281, 537, 531) and in 2 of 6 at the agent's 1e-4 (533,
  568). After the promotion, the true expected adopters of the chosen actions compared with the frozen arm on the same
  decisions: +2.14 per decision [1.63, 2.64] (lr 3e-3 seed 0, 319 decisions: 5.89 vs 3.74; R_psi had learned to reward
  the interventions, the policy chose them more), -0.14 [-1.56, 1.31] (63 decisions), -2.89 [-8.07, 2.12] (69),
  +0.70 [-0.07, 1.55] (lr 1e-4, 67), +0.87 [-0.04, 2.50] (32). Only the objective with a long post-promotion period
  shows a clear gain; the others are short and inconclusive, one points the other way. Intervals are bootstraps over
  the decisions of one seed (correlated decisions). Within 300 decisions (first pairing) the promotion came in 1 of 3
  seeds. Caveat on scope: in this controlled configuration the live update has no goal term (it learns only from the
  mixed reward), so a learned reward that tracks verified outcomes has more to add than in the agent, whose goal term
  already reinforces verified progress (§3.3b: the agent itself, 120 decisions: no promotion).

## 4. A finding about the learning signal: the return horizon smears the reward's credit

Why did R_psi not learn to rank decisions like their verified consequences even when meta-updates were applied?
`docs/evidence/RW/rw_credit_probe.py` differentiates the first-order outer surrogate with respect to an offset added
to R_psi's output on each support row (debug model, one LabWorld objective, the newest live update events; files
`credit_probe*.json`):

| outer utility | inner return half-life | gamma between support rows | corr(credit, own alignment c) | corr(credit, accumulated C) | corr(credit, verified progress y) | corr(c, y) | events (3 seeds) |
|---|---|---|---|---|---|---|---|
| return_to_go | 86400 s | 0.98 | 0.57 | 0.80 | -0.11 | 0.34 | 12 |
| return_to_go | 600 s | 0.21 | 0.89 | 0.57 | 0.65 | 0.71 | 12 |
| direct | 86400 s | 0.98 | 0.45 | 0.68 | 0.41 | 0.58 | 12 |
| direct | 600 s | 0.21 | 0.94 | 0.44 | 0.41 | 0.55 | 12 |

(medians over the newest 4 update events of each of 3 seeds; 60-decision objectives, updates every 6)

Reading. (1) In the inner loss the reward enters through the discounted return G_i = r_i + gamma G_{i+1}; with the
configured return half-life of one day and decisions about an hour apart (gamma 0.98 between support rows), the reward
at row k is credited to every earlier row, so the meta-gradient asks R_psi to reward row k for the accumulated
alignment of the *earlier* support actions (credit correlates 0.68-0.80 with C_k) more than for its own (0.45-0.57
with c_k); with a 600 s half-life (gamma 0.21) the credit follows the row's own alignment (0.89-0.94). (2) With
`utility="return_to_go"` the outer utility of a pool decision is dominated by the outcomes of the decisions after it,
and the credit psi receives for a row is uncorrelated with that row's verified progress (-0.11); with `direct` (the
decision's own signed outcomes) it is 0.41. Both make "reward what the verifier later confirms" a weak, noisy target
for psi in a long objective. Neither is a coding defect: (2) is a choice of the outer utility, changed here
(`direct` is now the default, §3.4 measures it); (1) is the inner-loss half-life, which also governs the bootstrap
and goal returns of every live policy update and was therefore not changed (§3.4 measures it as a diagnostic arm).

## 5. Full-size agent: one long staging objective on the real configuration

`docs/evidence/RW/rw_pathway.py --arm learned --decisions 80 --budget-s 800` on the real configuration (Qwen3-4B
frozen bf16 + LoRA + bridge + 50m core on the GPU under the shared lock; backbone proposals on, `thompson_policy`,
the agent's configuration with the RW defaults `auto` / `direct` / `validity`, reward step 1e-4); one long LabWorld
staging objective (target never reached; budget and deadline beyond the run); code = isolated copy of dc1753c + RW's
`reward/` of 2dfa6db (ST-3 and the same-psi pairing rule not yet included; at e-value 1.0 neither could have moved the
mix) + the fixed `nn/gpu.py` of 9656770. The first attempt waited 30 min and
timed out (`real_learned_try1_lock_timeout.log`; the old lock poller); the second got the lock at 12:24 UTC after
~16 min in the queue. Output `docs/evidence/RW/real_learned_20261128.json` / `.log`.

| quantity | measured |
|---|---|
| decisions in the hold | 35 (stopped by the hold budget: 616 s used, longest step 35 s; lock held 657 s; GPU peak 10.5 GB) |
| meta-update calls / applied | 36 / **4** (all query-window importance-weighted through the `auto` fallback, at decisions 16, 28, 32, 35, 4-9 groups, gradient norms 0.05-2.9); the pooled estimator never had a pool (27 `insufficient_independent_outcomes`, 2 ESS, 1 zero advantage, 2 before the first update) |
| validity evidence | 17 scored pairs (11 informative); R_psi's agreement -0.18 [-1.00, 0.82], the bootstrap's -0.06 [-1.00, 0.95]; e-value of "R_psi alone agrees better than mix 1": 1.0 (threshold 40) |
| bootstrap mix / R_psi's weight in the live updates | 1.0 throughout; recomputing each of the 11 live updates at mix 1 changes the policy by TV **0.0** (R_psi had no say) |
| behaviour | 27 `lab.act`, 4 `lab.probe`, 2 `wait`, 1 `lab.unlock`, 1 `lab.noise`; 259 verified adopters; the policy head's share of the selection up to TV 0.25 per decision |

Reading: (b) every link of the pathway up to the evidence ran on the full-size agent — rewards noted and scored
prospectively, meta-updates applied to R_psi from verified outcomes inside the objective, the mix decided by the
evidence; (c)/(d) not reached: in 35 decisions the evidence could not support giving R_psi weight (the expectation
stated before the run: an R_psi of the quality the debug runs reached needs 50-130 informative pairs, the hold gave
11), so R_psi did not influence a single live update and the frozen arm — identical by construction while the mix is
1 — was not run (it would only have measured GPU nondeterminism). No further GPU hold was used: one hold supplies
~35 decisions, the debug-scale promotions came after 221-568.

## 6. Negative results and their diagnosis

* **R_psi learns to reward what the verifier confirms only slowly.** Debug scale: the evidence lowered the mix in 1 of
  3 seeds within 300 decisions (first pairing), in 3 of 3 within 600 at reward step 3e-3 and in 2 of 6 at the agent's
  1e-4 (final rule), in 0 of 3 with `return_to_go` within 300. Diagnosis — an empirical limitation of the learning
  signal at these evidence rates, with two identified, measured contributors (section 4): (i) the outer utility:
  with the configured `return_to_go`, the verified policy gradient was dominated by later decisions' outcomes (credit
  to R_psi uncorrelated with the row's verified progress, median -0.11); `direct` fixes that part (0.41) and is now
  the default; (ii) the inner return horizon (half-life 1 day against decisions an hour apart, gamma 0.98) smears the
  credit for the reward at a row over all earlier support rows (credit follows the accumulated alignment C, 0.68-0.80,
  more than the row's own, 0.45-0.57). (ii) is not changed here: the half-life governs every live policy update.
* **The validity test is conservative by construction.** Anytime validity, the fixed-sequence lowering at alpha/2 and
  sign-only scores cost evidence. Simulation with `BettingTest` itself (bootstrap uninformative — random order on every
  pair, as the bootstrap is at reward time in LabWorld; 10 % tied outcomes; 400 runs): a learned reward that orders 70 /
  80 / 90 / 100 % of the informative pairs like the verifier is promoted after a median of 129 / 52 / 24 / 18 pairs
  (about twice as many matured decisions); at 50 % (no information) 1.8 % of runs are ever promoted within 400 pairs
  (nominal 2.5 %). A real-model hold of ~40 decisions supplies at most ~20 pairs, so inside one hold only a nearly
  perfect learned reward could be promoted.
* **In the integrated agent's configuration the pooled estimator starts late.** A live update's support is the
  12 newest rewarded decisions plus up to 16 replayed matured ones, and no pool decision may share an outcome group
  with the support, so the newest event's pool holds only decisions older than the newest ~28. In the debug agent run
  through the runtime (`docs/evidence/RW/dbg_agent_learned_partial.log` and its meta-update log, stopped at decision
  32 by the CPU rule): 0 applied — 32 `insufficient_independent_outcomes` (2-3 groups of 4), 3 ESS 1.28-1.31 with
  6-7 pool rows; the query-window fallback had no evidence either. This is the price of never using an outcome both
  in the inner update and in the outer estimate; it binds in the first few dozen decisions of an objective only.
* **A lead beyond item 1: the inner-loss return half-life also limits the live policy's own learning** (debug, the
  controlled configuration only). The frozen arms are bootstrap-only learners; with the configured 1-day half-life the
  true value of their chosen actions in the last third of a 300-decision objective was 2.91 / 3.73 / 5.63 adopters per
  decision (seeds 0-2), with a 600 s half-life 7.18 / 8.09 / 11.49 on the same worlds (first thirds 1.92-5.27 vs
  1.99-5.72). In the agent configuration (goal term, trust region, replay) this was not measured; it is reported, not
  acted on (`diag_long300hl_frozen_direct.json` vs `diag_long300_frozen_direct.json`).
* **The live planning run has no verified contrast yet** (every member-count reading 0): neither the meta-gradient
  nor the validity evidence can learn anything from it, and the mix correctly stays 1.

## 7. Compatibility with the live run's state

* `MetaLearner.load_state_dict` of a checkpoint written by the pre-RW code: everything loads; the validity evidence
  starts empty and is reported in `MetaLearner.restored_fresh`. Checked on the live planning run's newest checkpoint
  (`ckpt_00000039`, read-only; ledger from a copy): 9 update events, 35 meta-update records, mix 1.0, 12 ledger
  entries restored, `restored_fresh == ["validity (not in the checkpoint)"]`
  (`docs/evidence/RW/live_checkpoint_compat.py` / `.json`; the live run itself runs from a frozen snapshot and is not
  affected by this code) (`test_checkpoint_without_validity_state_loads_and_
  reports_it_fresh`); `pooled_done` / `goal_metric` start empty (the goal metric is re-registered when the runtime
  assigns the task). No new neural parameter was added (R_psi, the policy and psi's optimizer are unchanged), so
  every weight of an existing checkpoint is restored as before.
* The ledger file of an existing run keeps its entries (hash chain unchanged); readings admitted before RW keep their
  stored utility (the level); new readings of the level metric are admitted as changes. A mixed ledger is documented,
  not rewritten.
* Agent hooks (`docs/change_requests/RW.md`): RW-1 restore report `meta_restored_fresh`, RW-2 `status()["reward_validity"]`.


## 8. Tests and commands

* `tests/reward/test_reward_rw.py` (new, 21 tests): sequential statistics, validity evidence and mix decisions, the
  meta-learner wiring, checkpoint compatibility, the pooled estimator (finite differences), the `auto` fallback,
  level readings, the configuration's defaults. `tests/reward/test_reward_metalearner.py::test_promotion_moves_the_
  mix_only_on_held_out_evidence` now names `promotion_rule="value"` (the LRN rule it tests); `reward.diagnose.
  make_stack` keeps the stage-1/2 references' estimator (`query`) and utility (`return_to_go`) unless told otherwise.
* Whole directory: `cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/reward -q --rootdir=cam_live
  --junitxml=cam_live/docs/evidence/test_results/RW_fix.xml` — 168 passed, 4 skipped in 135 s (run on an isolated copy of HEAD 07303fe + RW's files, ST-3 and the same-psi rule included; `docs/evidence/test_results/RW_fix.xml`).
* Not mine but touched by the new defaults: `tests/integration/test_agent_runtime.py` 11 passed (693 s; incl. `test_meta_update_runs_with_a_nonzero_meta_gradient` under the new `auto` / `direct` defaults), `tests/integration/test_agent_recovery.py` + `tests/eval/test_meta_integrity.py` 4 passed, 1 skipped (same copy; logs `docs/evidence/RW/other_tests_*.log`). A run of the integration file in the shared worktree (with other agents' uncommitted changes) errored in its module fixture before it could report why; on the isolated copy it passes. G2-decision's `probe_meta_path.py` on RW's code: 21 of 23 checks (the two failures are the probe's own analytic-vs-FD comparison at derivative magnitudes 1.3e-8 and 1.6e-6, below FD precision; the code's gradient equals the probe's analytic one to 3e-15 in both); on HEAD 23 of 23; the probe is not run-to-run reproducible (HEAD twice: first applied gradient norm 0.258 / 0.249), so its trajectories differ between runs (`docs/evidence/RW/g2d_probe_meta_path_*.log`).

## 9. Not done / limitations

* No reliable improvement from the learned reward is shown. The pathway is complete at debug scale (5 promoted
  objectives of 9 at 600 decisions); a clear gain after promotion in one of them (319 decisions left), the four short
  post-promotion periods inconclusive; one seed per objective, correlated decisions; controlled configuration without
  the agent's goal term.
* The inner-loss return half-life (diagnosis (ii)) was not changed; it governs the live policy updates and needs its
  own evaluation of policy learning, which this round did not do.
* The validity score uses signs only and disjoint pairs; magnitude-aware or within-batch scores could need fewer
  decisions but need their own validity argument (overlapping pairs bias the null, §2.1).
* A learned reward only as good as the bootstrap is not promoted (no evidence that it helps); the old rule promoted
  non-inferior rewards. This is a deliberate change.
* Level readings: the first reading of an objective anchors (utility 0) — the level before it is unknown to the
  ledger; a reading that arrives out of order is credited 0 (its interval was already credited to the decision
  after which the later reading was taken).
* The pooled estimator evaluates the updated policy on the matured decisions *before* the update (states of the past,
  importance-weighted actions): it assumes the situation changes slowly within an objective. The query estimator
  (decisions after the update) is still used when the pool lacks evidence (`auto`).
* Real-model evidence is limited by the GPU budget: one hold of ~14 minutes is ~40 decisions on the real configuration, while the debug-scale promotions came after 221-568 decisions.
