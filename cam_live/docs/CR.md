# CR — retrospective credit assignment (amendment of 2026-11-28, item 4)

Scope (briefs/CR.md): why RUDDER / COCOA failed their held-out selection; retrospective credit that revisits stored
experience when new outcomes arrive, revises earlier credit and updates weights; scientifically justified estimator
selection whose evidence is obtainable inside one long objective; whether better credit improves the integrated
agent's learning from delayed, uncertain consequences. Owned paths: `src/cam_live/credit/**`, `tests/credit/**`, this
file, `docs/status/CR.json`, `docs/change_requests/CR.md` (evidence summaries in `docs/evidence/CR/`). agent.py hooks
are change requests CR-1 / CR-2 / CR-3 for ST (`docs/change_requests/CR.md`).

Debug scale = `CamLiveConfig.debug()` (tiny random backbone, debug core, CPU, LabWorld). Real = Qwen3-4B + 50m core.
Every number below comes from a command named next to it; raw outputs are listed in section 7.

## Summary

* **Why the selection failed** (section 1): not one cause. (i) In D's lab it depends on the seed and on the training
  budget: at D's 400 updates seed 0 (weakly heterogeneous mechanisms) fails because both estimators under-resolve the
  mechanisms (slopes 0.36 / 0.19), seed 2 because of RUDDER's bias (14.0), and in seed 1 both *pass* the old gate
  (RUDDER 1.42, COCOA 2.73). RUDDER was under-trained: with 4x the updates on the same data it reaches the ideal
  learner's level in both seeds tried (seed 0: 6.72 -> 3.53, ideal 3.10; seed 2: 14.0 -> 2.28, passing both rules);
  COCOA is unstable with training length (seed 0: 10.9 -> 32.0; seed 2: 3.43 -> 2.81).
  (ii) The gate rejects an ideal learner at realistic sizes (it ignores the estimator's own training error; ideal
  learner with as much training data as experiment data passes 79 %, a perfectly calibrated estimator at 2 units per
  arm 60 %). (iii) In the integrated agent the comparison was impossible by construction (closed lifetimes only,
  lifetime units, one-unit experiments, every lifetime trained on). D's claim that the first-decision latent is
  identical across mechanisms is refuted (identical within a mechanism, distinct across).
* **What changed** (section 2, credit/ only): credit units inside an open objective (complete segments of K = 16
  decisions) so RUDDER / COCOA credit exists and is revised during one long objective and re-credited at closure; a
  blocked held-out split by segment; an estimand-matched pooled selection rule over held-out randomised units of any
  experiment size (with per-experiment calibration kept where testable); a revision queue so revised credit can
  reach the policy weights (agent hook CR-3: measured 3 of 96 policy-update rows carried RUDDER credit without it).
* **Does better credit improve learning?** Controlled diagnostic (section 3.2): learning a consequence that no
  evidence links to the decision is impossible with the simple attribution (P stays 0.5) and works with every
  return-based signal (P(unlock) 0.93-0.94 in the last quarter, 6 informative mechanisms); RUDDER's learning signal
  is no better than plain REINFORCE there. Integrated agent (section 3.1, one open objective, 64 decisions, 9 paired
  seeds): forcing RUDDER as training default *lowered* competence (-66.8 +- 26.0 expected adopters per run, p ~ 0.03);
  the evidence-based selection keeps `simple`, which tracks the true value of the agent's own actions best at this
  amount of experience. Negative result for the learned estimators in the agent, with its diagnosis in section 3.
* **Real configuration** (section 5, one GPU hold, Qwen3-4B + 50m, backbone candidates on): RUDDER credit on 28 of 39
  records inside the open objective and 41 CR-3-replayed credited rows in the real policy update; COCOA credits
  nothing there (open-ended backbone candidates without tracked proposal probabilities are refused, as spec 9.3
  requires); no randomised experiment occurred, so no selection evidence. In the paired `simple` run CR-3 replayed 2
  rows, and simple credit tracked the true action value better (Pearson 0.64) than RUDDER's (0.51); one pair, so
  no competence comparison is claimed.

## 1. Why RUDDER and COCOA failed the held-out selection (KF-X15, KF-R14)

Three candidate causes were separated by scoring *reference* estimators through the selector's own per-experiment
gate (paired z, mean z^2 <= 3 over >= 5 experiments) on the same held-out units of D's lab
(`python -m cam_live.credit.cr_diagnose estimand`; six randomised experiments, one hidden LabWorld key-door mechanism
each, 48 lifetimes per experiment, estimators trained on independent training lifetimes of the same mechanisms):
`truth` (the mechanism's true effect from 200 forced-arm lifetimes per arm), `train_ideal` (the effect an ideal
learner can know from the actual training data: the training lifetimes' arm difference per mechanism), `marginal`
(one effect for every mechanism) and `no_effect`.

| run | old gate mean z^2: rudder / cocoa / truth / train_ideal / marginal | slope of estimate on truth: rudder / cocoa / train_ideal |
|---|---|---|
| seed 0, 48 training lifetimes per mechanism (D's setting) | 6.72 / 10.89 / 2.37 / **3.10** / 7.73 | 0.36 / 0.19 / 1.24 |
| seed 0, 12 per mechanism | 4.28 / 7.71 / 2.37 / 0.41 / 7.73 | 0.48 / 0.15 / 1.77 |
| seed 1, 48 per mechanism | **1.42 / 2.73** / 0.56 / 0.38 / 161 | 0.89 / 1.40 / 1.11 |
| seed 0, 144 per mechanism | 6.00 / 5.17 / 2.37 / 2.35 / 7.73 | 0.59 / 0.48 / 1.26 |
| seed 0, 48 per mechanism, **1600 updates** (D: 400) | **3.53** / 32.0 / 2.37 / 3.10 / 7.73 | **1.25** / -0.16 / 1.24 |
| seed 2, 48 per mechanism, 400 updates | 14.0 / 3.43 / 1.22 / 2.22 / 29.8 | 1.52 / 1.08 / 1.60 |
| seed 2, 48 per mechanism, **1600 updates** | **2.28** / 2.81 / 1.22 / 2.22 / 29.8 | 1.67 / 1.48 / 1.60 |

Findings (each measured, raw JSON in section 7):

1. **Not an estimand mismatch in D's lab.** The first-decision latents are identical within a mechanism and arm
   (within-mechanism spread 0, between-mechanism spread > 0: `first_decision_latent` ratio = inf for both arms), so
   the information state at the randomised decision identifies the mechanism and the per-experiment estimand is
   attainable. D's diagnosis "the latent is the same in every lifetime of every mechanism" is wrong for the current
   code (the channels' features are observed). The estimand problem is real elsewhere: when experiments differ by a
   mechanism the decision cannot see, even the best estimate given the information state fails the per-experiment
   gate (`tests/credit/test_cr.py::test_the_per_experiment_gate_rejects_the_best_estimate_at_the_information_state_the_pooled_rule_does_not`).
2. **At 400 updates the failure is seed-specific (low effect heterogeneity relative to noise in seed 0).** In seed 0 the true effects span
   3.9-8.7 (between-mechanism sd 1.8) while each experiment's measured difference has a standard error of 1.0-2.6;
   RUDDER and COCOA shrink strongly toward the average effect (slopes 0.36 and 0.19, correlation 0.44 and 0.40) and
   fail. In seed 1 the effects span 3.8-43 and **both pass the old gate** (RUDDER 1.42, COCOA 2.73; slopes 0.89 /
   1.40): under the old rule RUDDER would have been promoted there. D's "nothing is promoted" rested on one seed.
   More training data improves the resolution in seed 0 but not enough to pass: 12 / 48 / 144 training lifetimes
   per mechanism give RUDDER slopes 0.48 / 0.36 / 0.59 (mean z^2 4.28 / 6.72 / 6.00) and COCOA 0.15 / 0.19 / 0.48
   (7.71 / 10.89 / 5.17), while the ideal learner on the same training data scores 0.41 / 3.10 / 2.35.
3. **The gate rejects an ideal learner at realistic sizes.** The paired standard error contains the experiment's
   noise but not the estimator's own training error, which is as large when training data and experiment are of the
   same size. In seed 0 the ideal learner on the actual training data fails (3.10; chi^2_6 18.6 > 16.8). Monte Carlo
   with the lab's effects and spread (`cr_diagnose power`, 2000 repetitions, 6 experiments):

   | units per arm | exactly calibrated: pass old gate | ideal learner, training = experiment size | ideal learner, 4x training data | pooled rule (exactly calibrated) |
   |---|---|---|---|---|
   | 2 | 0.60 | 0.36 | 0.54 | 0.00 |
   | 3 | 0.84 | 0.55 | 0.76 | 0.07 |
   | 5 | 0.93 | 0.68 | 0.90 | 0.48 |
   | 10 | 0.97 | 0.77 | 0.94 | 0.89 |
   | 24 | 0.99 | 0.79 | 0.96 | 1.00 |

   At the agent's minimum of 2 matured held-out units per arm, a perfectly calibrated estimator fails 40 % of the
   time (the paired SE rests on 1 degree of freedom per arm). Validation at 0.5 sd precision needs ~100+ units.
4. **RUDDER was under-trained; COCOA is unstable with training length.** Same training lifetimes, 1600 instead of
   D's 400 updates: RUDDER improves in both seeds tried - seed 0 from 6.72 to 3.53 (slope 1.25, correlation 0.88: the
   ideal learner's 1.24 / 0.85 and 3.10, so the remaining failure is the gate's, finding 3), seed 2 from 14.0 (a
   pooled bias of +4.87) to **2.28, which passes both rules** (chi^2_6 13.7 < 16.8; ideal learner 2.22). COCOA goes
   from 10.89 to 32.0 in seed 0 (its estimates turn negative for five of six positive effects; pooled bias -6.15,
   refused by the pooled rule too) but from 3.43 to 2.81 in seed 2 (chi^2 16.9, just above 16.8): the learned
   hindsight model can overfit with training length. So D's failure was largely an estimator *training budget*
   problem for RUDDER (not a design defect), and a stability problem for COCOA; neither is a reason to change the
   gate, both are reasons why a fixed budget cannot validate an estimator once and for all (the selection re-runs
   whenever new held-out evidence arrives and falls back when an estimator stops passing).
5. **In the integrated agent the comparison could not run at all, by construction (KF-R14):** learned credit needed a
   *closed* lifetime, validation units were whole lifetimes, the agent's experiments have one unit each and every
   lifetime is trained on. Inside one long objective - CAM Live's real setting - this meant no learned credit before
   the objective ends and no held-out unit ever. And (newly found, section 3) revised credit did not reach the policy
   weights: with RUDDER forced, 3 of 96 policy-update rows carried RUDDER credit.

## 2. What changed (credit/ only; agent hooks are change requests)

* **Credit units inside an open objective** (`credit/segments.py`, `credit/providers.py`). A closed objective is one
  unit, exactly as before. An open objective's complete blocks of K = 16 top-level decisions (`CreditAssigner(...,
  segment_decisions=16)`) are units: a segment's return = the observed answers to the goal-relevant queries its own
  decisions declared, final when the block is complete and nothing of it is pending. RUDDER and COCOA train on
  matured segments, credit their decisions, conservation holds per segment, and every decision is re-credited over
  the whole lifetime when the objective closes (credit history keeps every version). Truncation: an outcome measured
  by a later segment's query is not part of an earlier segment's return until the lifetime closes.
* **Blocked held-out split** (`credit/learner.py`, `val_split="segment"`): every 1/f-th segment of each objective is
  held out as a whole (stratified as INT3A-5's record split; blocked cross-validation for temporally dependent data,
  Roberts et al. 2017), so randomised decisions of held-out segments are units nothing trained on. Records stored
  before the switch keep the record split and their units count as trained (live-run compatibility).
  `trained_units` records segment-level training (a segment also marks its objective).
* **Estimand-matched pooled selection** (`credit/select.py`): every held-out randomised unit (any experiment size,
  one unit included) carries its arms, logged propensity, measured return-to-go and each estimator's value; the
  paired inverse-propensity term d_i = (e_i - y_i - c_i) w_i (per-decision credit) or x_i - (y_i - c_i) w_i
  (per-unit contrast, COCOA) has expectation equal to the estimator's contrast bias at the unit's information state
  (c_i a leave-one-out baseline). An estimator is validated when (a) the pooled average bias is equivalent to zero
  within 0.5 sd of the measured return (two one-sided tests, 90 % interval, >= 8 units) and (b) on every experiment
  with >= 5 held-out units per arm its paired z^2 are not jointly rejected (sum z^2 <= chi^2_K 0.99 quantile). The
  validated estimator with the tightest bias bound becomes the training default; a learned default that stops
  passing falls back to `simple`. Without unit evidence the old rule is used unchanged. In D's lab the new rule
  decides as the old one: seed 0 nothing (calibration rejected: rudder 40.3, cocoa 65.3 > 16.8), seed 1 RUDDER.
* **Revision replay** (`CreditAssigner.pop_revised`, CR-3): records whose learning credit changed are queued
  (checkpointed) for the policy update.
* Diagnosis and evaluation programs: `credit/cr_diagnose.py`, `credit/cr_agent_eval.py`.

## 3. Does better credit improve learning from delayed consequences?

### 3.1 The integrated agent: one long open objective, simple vs selected vs forced estimators

`python -m cam_live.credit.cr_agent_eval --condition {simple,rudder,cocoa} [--no-replay] --seed S --decisions 64
--segment 8` on a snapshot of the code (`git archive` of dc1753c + CR's credit/ as of f937ce6 + CR-1 and CR-3 applied to the
copy of agent.py; `PYTHONPATH` at the copy): the real `CamLiveAgent` and `LiveRuntime` through ONE open LabWorld
objective of the key-door class (T1 mechanisms; target out of reach, so it never closes), debug scale on the CPU,
schema-walker candidates, same learner split (segment, val_fraction 0.15) in every condition. `simple` is the
evidence-selected default; `rudder` / `cocoa` force that estimator's credit as training default (ablation only).
Ground truth per executed action from `eval.envs.TruthTap` (expected adopters = hidden rate x effort).

| condition | runs | expected adopters, whole run (64 decisions) | per executed action, 2nd half | waits chosen | unlock found | credited policy-update rows | corr(credit, truth): simple / rudder / cocoa |
|---|---|---|---|---|---|---|---|
| simple | 9 | 175.28 +- 28.07 (n=9) | 4.01 +- 0.80 (n=9) | 3.78 +- 1.35 (n=9) | 5 / 9 | 139.33 +- 3.40 (n=9) | 0.39 / 0.28 / 0.25 |
| rudder | 9 | 108.52 +- 15.14 (n=9) | 2.42 +- 0.55 (n=9) | 4.78 +- 2.71 (n=9) | 1 / 9 | 118.22 +- 9.41 (n=9) | 0.20 / 0.13 / 0.03 |
| cocoa | 6 | 141.58 +- 67.09 (n=6) | 2.42 +- 1.21 (n=6) | 2.33 +- 1.12 (n=6) | 1 / 6 | 110.83 +- 7.36 (n=6) | 0.34 / 0.19 / 0.20 |
| rudder_norep | 3 | 130.65 +- 72.49 (n=3) | 3.20 +- 2.21 (n=3) | 1.00 +- 0.58 (n=3) | 1 / 3 | 32.67 +- 2.33 (n=3) | 0.28 / 0.27 / 0.18 |

Paired by seed against `simple` (expected adopters of the whole 64-decision run; per executed action in the second half):

* cocoa: whole run -38.20 +- 47.11 (n=6); per action, second half -1.68 +- 0.87 (n=6); seeds [0, 1, 2, 3, 4, 5]
* rudder: whole run -66.76 +- 26.00 (n=9); per action, second half -1.59 +- 0.64 (n=9); seeds [0, 1, 2, 3, 4, 5, 6, 7, 8]
* rudder_norep: whole run 4.31 +- 101.40 (n=3); per action, second half 0.45 +- 3.11 (n=3); seeds [0, 1, 2]

Credit quality on the agent's own decisions (`cr_agent_eval.credit_quality`, last column): Pearson correlation of each stored credit with the ground-truth value (expected adopters) of the executed `lab.act`, averaged over the runs of the condition (each run stores every estimator's credit, whatever the training default).

**Result.** Forcing RUDDER's credit as the training default *lowered* competence: paired by seed (n = 9), expected adopters over the run -66.8 +- 26.0 (t = 2.57, 8 df, p ~ 0.03) and per executed action in the second half -1.59 +- 0.64; the door was found in 1 of 9 RUDDER runs against 5 of 9 `simple` runs (Fisher p ~ 0.13). COCOA forced: -38.2 +- 47.1 over the run and -1.68 +- 0.87 per action (n = 6, not significant). With CR-3 the forced estimator reaches the policy update about 4x as often (credited rows 112-134 vs 28-35 without the line, seeds 0-2); without the line the forced estimator barely acts (rudder_norep - simple +4.3 +- 101, n = 3). The evidence-based selection kept `simple` in every `simple` run (0-2 held-out units per run; nothing can be promoted on fewer than 8).

**Diagnosis.** In this world most of a decision's value is its own adopters, which its own declared query measures exactly: the simple attribution is the right per-decision credit for them (correlation with the true value 0.39 in `simple` runs). RUDDER's return-like credit is the segment's measured return-to-go minus the decomposer's prefix value: it attributes the adopters of *later* decisions to every earlier decision of the segment (waits and probes included) and its decomposer has seen only a few matured segments, so it tracks the true value worse (0.13-0.28; the contribution, -0.02-0.13). This is an empirical limitation of the amount of experience (and of a credit that is unbiased but high-variance), not a defect found in the code: the same estimators learn a delayed-only consequence in the controlled diagnostic (3.2). It is exactly the case spec 9.4 guards against ('do not let speculative counterfactual predictions dominate learning before they are validated'): the selection rule refused to promote them, and the measured harm of forcing them is the evidence that the refusal was right. Process replicates of seed 0 (ids carry os.urandom bytes, so runs of one seed differ): simple 75.5 vs 70.8, rudder 83.8 vs 86.4 expected adopters.

### 3.2 Controlled diagnostic: learning a delayed-only consequence with each credit signal

`python -m cam_live.credit.cr_diagnose policy --signal SIG --seed S --lifetimes 80` (debug scale, D's lab): one
LabWorld key-door mechanism per seed; the first decision of every lifetime is drawn from a learned two-way softmax
{unlock, wait} (logged propensities), later decisions from D's behaviour policy; 24 randomised warm-up lifetimes
train the estimators, then 80 learning lifetimes in which, after each lifetime closed and the estimators trained, the
logit moves by lr x (credit / sd(G)) x (a - p) with the first decision's credit under SIG (same lifetimes, noise
seeds and step scale for every signal). `unlock` has no immediate effect (no evidence link) and multiplies the later
yield of one channel. True effect of unlock per mechanism from forced-arm simulations (150 lifetimes per arm).

| signal | P(unlock) last quarter, informative mechanisms | regret per lifetime (effect x P(wait)), informative | seeds |
|---|---|---|---|
| simple | 0.50 +- 0.00 (n=2) | 2.08 +- 0.07 (n=2) | [0, 1, 2] |
| return | 0.93 +- 0.01 (n=6) | 1.88 +- 0.57 (n=6) | [0, 1, 2, 3, 4, 5, 6] |
| rudder | 0.93 +- 0.01 (n=6) | 1.87 +- 0.60 (n=6) | [0, 1, 2, 3, 4, 5, 6] |
| rudder_contribution | 0.93 +- 0.03 (n=6) | 1.25 +- 0.27 (n=6) | [0, 1, 2, 3, 4, 5, 6] |
| cocoa | 0.94 +- 0.02 (n=6) | 2.05 +- 0.44 (n=6) | [0, 1, 2, 3, 4, 5, 6] |

(`simple` was run on seeds 0-2 only: it gives both arms no credit, so P(unlock) = 0.5 throughout by construction and its regret is 0.5 x effect, i.e. 16.7 / 5.5 / 6.6 / 8.4 on mechanisms 4000-7000.)

True effect of unlock per mechanism seed (forced-arm simulation, 150 lifetimes per arm): 1000: 4.31 +- 0.34, 2000: 4.02 +- 0.27, 3000: 0.39 +- 0.41, 4000: 33.33 +- 1.80, 5000: 10.99 +- 0.76, 6000: 13.25 +- 0.79, 7000: 16.72 +- 0.91. Informative = |effect| > 2 SE (mechanism 3000, seed 2, has no detectable effect: every signal's choice there costs < 0.4 per lifetime).

* `simple` (evidence-linked outcomes of the decision itself) gives the unlock decision no credit at all and learns
  nothing (P(unlock) stays 0.5 by construction): learning from a consequence that no evidence links to the decision
  needs return-based credit. In the integrated agent `simple` is not used alone: its per-decision goal rewards enter
  a discounted return over the policy-update batch (the `return` row is the analogue, truncated to the batch).
* With a real effect (6 mechanisms) every return-based signal ends near P(unlock) = 0.93-0.94; regret differences between them are within about one standard error (n = 6). RUDDER's learning signal (the agent's RUDDER key, unbiased for any decomposer) behaves like REINFORCE, consistent with D's measurement that it gives no variance reduction at the key decision. RUDDER's contribution, the quantity the selection validates, had the lowest regret in 5 of 6 informative mechanisms (low variance, biased only if the decomposer is) - a candidate learning signal once validated, which the agent cannot use today (`_credit_for` treats every learned estimator as return-like); not proposed as a change.
* On the null mechanism (3000) the biased signals committed to `wait` (P = 0.03 / 0.15) and RUDDER's signal to `unlock` (0.93) while REINFORCE stayed at 0.71 AUC - all near-zero regret; it shows how quickly a biased signal commits without evidence.

## 4. Evidence per capability (amendment item 8: correctness, runtime integration, causal influence, improvement)

### 4.1 Learned retrospective credit inside one long open objective (revisit, revise)
* (a) Correctness: `tests/credit/test_cr.py::test_learned_credit_appears_for_matured_segments_of_an_open_objective_and_is_revised_at_closure`
  (open 20-decision LabWorld objective, K = 6: the three complete matured segments get RUDDER credit, the incomplete
  one none; conservation error < 1e-6 per segment; RUDDER trains on segments before closure; at closure every
  decision is re-credited over the whole lifetime, the credit history keeps both versions); K = 0 reproduces the old
  lifetime-only behaviour; D's tests unchanged, DX's selection tests adapted (tests/credit 155 passed, section 6).
* (b) Runtime: in the integrated agent (`cr_agent_eval`, all 26 debug-scale runs of 64 decisions) 56-64 records carried
  RUDDER credit and 48-61 COCOA credit while the objective was still open (before CR: none until the objective
  closes), and stored credit was revised as segments matured and the estimators retrained (the most revised record
  of a run has 117-222 credit-history entries over all estimator keys); the slow test
  `tests/credit/test_cr_agent.py` pins it on the real agent / runtime. Real configuration: section 5.
* (c) Causal influence: only through the training default (`CamLiveAgent._credit_for`); see 4.3.
* (d) Improvement: none in the integrated agent (section 3.1: forcing the learned credit lowered competence); in the
  controlled diagnostic return-based credit learns a delayed-only consequence that the simple attribution cannot
  (section 3.2).

### 4.2 Estimand-matched estimator selection whose evidence exists inside one objective
* (a) `test_cr.py`: the per-experiment gate rejects the information-state optimum when mechanisms are hidden while
  the pooled rule accepts it and refuses a biased estimator; orientation invariance (swapped arm labels flip only the
  sign); one-unit experiments pooled, no promotion below 8 units; per-experiment calibration still rejects an
  estimator that ignores identifiable heterogeneity; checkpoints with / without units. DX's selection tests adapted
  (one-unit experiments now contribute units; the selection logs "only 3 held-out randomised units").
* (b) Runtime: with the segment split the agent holds out whole segments (8-16 records per 64-decision run) and
  every learning update evaluates the pooled units (0-2 units per run, logged with the reason); the selection never
  had the 8 units it needs to promote anything in 64 decisions (default stayed `simple` in all 10 `simple` runs).
  (These runs used the first version of the split, in which segment 0 could also be held out; e8dee3f keeps segment 0
  on the record split.)
* (c)/(d) The rule decides the training default; in 64-decision runs it never changed it. Units accrue at about
  experiment_rate x val_fraction = 1.5 per 100 decisions: validating an estimator at 0.5 sd precision (~100 units,
  section 1) needs thousands of decisions of one objective at the agent's current experimentation rate - an
  empirical limitation of the data, now stated by the selection log instead of being impossible by construction.

### 4.3 Revised credit reaches the policy weights (CR-3)
* (a) `test_cr.py::test_revised_credit_is_queued_for_replay_and_survives_a_checkpoint`.
* (b)/(c) Integrated agent, RUDDER forced, same seeds, with vs without the CR-3 line (patched copy): RUDDER-credited
  rows consumed by the real policy update with the line 112 / 132 / 134 (seeds 0-2; 84-99 of them replayed earlier decisions), without it 28 / 35 / 35 of 234 rows; the line changes the input of every policy update after the first matured segment (causal influence on the update shown; on behaviour see 4.4).
* (d) It amplifies whatever estimator is the training default: with a forced, unvalidated RUDDER the competence
  dropped (section 3.1); with the evidence-selected `simple` it replays almost nothing (0-2 rows per run: simple
  credit is final while a decision is still in the recent window). Its value is conditional on validated credit.

### 4.4 Does better credit improve learning from delayed, uncertain consequences? (simple vs selected / forced)
* Controlled diagnostic (section 3.2, 7 mechanisms, 6 with a detectable effect): the simple attribution cannot learn a consequence no evidence links to the decision (P(unlock) stays 0.5 by construction; regret 0.5 x effect per lifetime). Every return-based signal learns it (P(unlock) 0.93-0.94 in the last quarter). On the two mechanisms where simple was run (1000 / 2000) regret per lifetime was simple 2.16 / 2.01, REINFORCE 0.60 / 0.50, RUDDER signal 0.60 / 0.36, COCOA 2.02 / 0.75, RUDDER contribution 1.50 / 0.26; on the four others simple's regret is 0.5 x effect = 16.7 / 5.5 / 6.6 / 8.4 (by construction, not run) against 0.92-4.19 for the return-based signals. Between the return-based signals the differences are within about one standard error (n = 6): RUDDER signal 1.87 +- 0.61, REINFORCE 1.88 +- 0.57, COCOA 2.05 +- 0.44, RUDDER contribution 1.25 +- 0.27.
* Integrated agent (section 3.1): forcing RUDDER lowered competence (-66.8 +- 26.0 expected adopters per 64-decision run, n = 9 paired); COCOA -38.2 +- 47.1 (n = 6, n.s.). Diagnosis: an empirical limitation of the experience in 64 decisions of one objective (few matured segments to train the decomposer / hindsight model; the delayed-only consequence rarely experienced; in this world the simple attribution measures most of a decision's value exactly), not a defect found in the credit code - the same estimators learn the delayed consequence in the controlled diagnostic. The architectural defect that was found (revised credit not reaching the weights) is CR-3.

## 5. Real configuration (Qwen3-4B + 50m core, backbone-written candidates on)

`cr_agent_eval --condition {rudder,simple} --real --seed 0 --decisions 40 --segment 4 --max-wall-s 720` inside the shared GPU lock (two holds, 12:57-13:10 and 14:15-14:27 UTC, no other CPU job of CR running): Qwen3-4B (frozen, bf16) + 50m core on CUDA, backbone-written candidates on, the same patched snapshot as section 3.1 (credit/ learner.py and segments.py of e8dee3f, providers.py of f937ce6; CR-1 and CR-3 applied; nn/gpu.py of 9656770), one open LabWorld key-door objective; `rudder` forces RUDDER as training default, `simple` is the evidence-selected default.

* (b) Integration on the full-size agent (rudder run): 39 decisions in 720 s (35 executed actions, 4 waits); RUDDER credit on 28 of 39 records - the 7 matured segments of 4 decisions - while the objective was open; credit revised up to 67 credit-history entries per record. **No COCOA credit at all**: the backbone-written candidate sets are open-ended and their proposal probabilities are not tracked, so COCOA refuses them (spec 9.3: no counterfactual claim without support) - on the real configuration COCOA is not a usable estimator for open-ended actions.
* (c) The real policy update consumed RUDDER credit on 80 of 179 rows over 13 updates; 41 of them were earlier decisions replayed by CR-3 after their segment matured (updates 7-13: 5-6 replayed rows each).
* Credit quality of the rudder run on its own `lab.act` decisions (Pearson / Spearman with the true expected adopters): simple 0.03 / 0.42 (n = 26), RUDDER signal 0.22 / 0.38, RUDDER contribution -0.23 / 0.02 (n = 22) - no estimator tracks the true value well after 39 decisions.
* The run contained no randomised experiment ('no experiment is recorded in the experience store'), so the selection had no evidence at all on the real configuration; 5 records were held out by the segment split. All 6 slow steps were refused by the held-out guard.
* Paired `simple` run (same seed, 40 decisions): RUDDER credit was computed for 32 of 40 records but not used; the policy update consumed simple credit on 74 of 140 rows, 2 of them replays (CR-3 replays almost nothing under the evidence-selected default, as at debug scale). Credit quality on its `lab.act` decisions (Pearson / Spearman): simple 0.64 / 0.80 (n = 32), RUDDER signal 0.51 / 0.55, RUDDER contribution 0.15 / 0.11 (n = 26) - the same ordering as at debug scale.
* Behaviour (one run per condition, not a test): the RUDDER run unlocked the door at decision 20 and reached 150.9 expected adopters (7.47 per action in the second half), the `simple` run never unlocked it (40.6; 1.41). At debug scale unlock discovery was the dominant source of variance (5 of 9 `simple` vs 1 of 9 RUDDER runs found it), so a single real-configuration pair says nothing about which credit is better; the n = 9 paired debug-scale comparison is the evidence (section 3.1).

## 6. Tests and runs

* `cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/credit -q --rootdir=cam_live
  --junitxml=cam_live/docs/evidence/test_results/CR_fix.xml`: **155 passed, 9 skipped, 0 failed, 213 s** (at 96f7272's code; earlier runs during the work: 153 / 154 passed, 0 failed, 10-16 min on the loaded machine). CR's own tests: `tests/credit/test_cr.py` 11, `test_cr_agent.py` 1 (slow); DX's `test_dx_selection.py` adapted (2 assertions: one-unit experiments now contribute pooled units).
* `-m slow tests/credit/test_cr_agent.py` (real agent + runtime, one open objective): 1 passed (125 s): the real CamLiveAgent + LiveRuntime through one open 20-decision LabWorld objective with segments of 4 - RUDDER credit on >= 8 records and COCOA credit before closure, revised credit, selection log rows with reasons.
* Patched copy (CR-1 + CR-3 applied to agent.py at dc1753c): `tests/integration/test_int3a_audit.py`,
  `test_int2_wiring.py`, `test_agent_runtime.py`: test_int3a_audit 4/4 + tests/credit/test_cr.py 10/10 (59 s), test_int2_wiring + test_agent_runtime + test_int2_checkpoints 20/20 (364 s); an earlier run of the same files on the copy, stopped under the master's CPU rule, had one failure in test_int3a_audit that passed alone and in the full file afterwards (order / load dependent, KF-E24 pattern) (the copy's credit/learner.py and segments.py as of e8dee3f, providers.py as of f937ce6)
* Live-run compatibility (read-only): the newest checkpoint of the live planning run
  (`runs/cam_live/live/discord-100-planning-20261128T083454Z/checkpoints/ckpt_00000038`, learner schedules and credit
  assigner state) loads into CR's learner and assigner with both splits: default `simple`, no unit evidence, empty
  revision queue, the live objective counted as trained (`trained_units` contains `trained_objectives`); the new keys
  (`trained_units`, `segment_split_from_seq`, `revised`, `segment_decisions`) are added on the next save. No
  parameter was added or removed (no weights change shape). Script and output: section 7.

## 7. Raw outputs

Scratch directory `/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/CR/`: `diag/estimand_*.json`
(lab diagnosis), `diag/power.json`, `diag/policy/*.json` (policy-learning diagnostic), `agent/abl1/*.json`,
`agent/abl1rep/*.json`, `agent/abl2/*.json` (integrated ablation), `agent/real/*.json` (real configuration),
`compat/live_state_compat.json`, `snap/` (the patched snapshot the agent evidence ran on). Copies of the summaries are
in `docs/evidence/CR/`.

## 8. Not done / limitations

* agent.py hooks CR-1 (segment split) and CR-3 (revision replay) are change requests; until ST applies them the agent
  has segment credit (CreditAssigner default K = 16) but no held-out segments and revised credit mostly does not reach
  the policy (measured 3 of 96 rows with RUDDER forced).
* Segment returns truncate: an outcome measured by a later segment's query is not credited to an earlier segment's
  decisions until the objective closes (K bounds the horizon; K is not learned).
* Selection evidence accrues at ~1.5 held-out units per 100 decisions at the agent's experimentation rate; validation
  needs ~100 units at the 0.5 sd margin. The margin and unit minimum are choices, documented with their power.
* The pooled bias test is a necessary condition; with one-unit experiments per-context calibration cannot be tested
  (only where experiments reach 5 units per arm).
* RUDDER's decomposer is trained by one head step per frequent update in the agent (~40-50 steps in a 64-decision
  run, on a handful of segments); in D's lab it needed ~1600 steps on 288 lifetimes to resolve the mechanisms. A
  dedicated, guarded training schedule for the credit heads is the next step suggested by the data; not built.
* Segment credit is on by default (`CreditAssigner(segment_decisions=16)`): from the next restart the agent's RUDDER /
  COCOA losses also train on matured segments of the open objective. Its extra compute was not measured separately
  (real configuration: 413 s training vs 301 s inference for 39 decisions with it); `segment_decisions=0` (CR-2)
  restores the lifetime-only behaviour exactly.
* The integrated ablation is debug scale (tiny random backbone, schema-walker candidates) and LabWorld only; the
  real-configuration runs are in section 5.
* Under the master's CPU rule (10:50 UTC) the last 12 ablation runs were paused (SIGSTOP) and resumed one at a time;
  the runtime uses a simulated clock, so only wall-clock learning budgets could be affected. The 1600-update lab runs
  (section 1, finding 4) are two seeds.
* The live run's estimator selection is untouched until restart; no live-run directory was written.
