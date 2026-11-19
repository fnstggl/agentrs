# AGENTRS superlearner: architecture, environments and pre-registered protocol

Branch `feat/superlearner`. Package `src/agentrs/sl/`. Experiments run from the frozen worktree
`/workspace/agentrs-sl-exp` (tags `sl-exp-v*`) through the supervisor program `agentrs_sl_queue`
(`scripts/sl_queue.py`). Results go to `runs/sl/<job>`.

The central question: **with the same architecture and the same limited experience in an unfamiliar world, does a core
that has learned across other worlds learn faster than a newly initialised one, and does that advantage grow with the
breadth of its past experience?**

## Architecture (`sl/model.py`, `sl/runner.py`, `sl/learner.py`, `sl/train.py`)

| Component | Implementation |
|---|---|
| Unified world interface | `sl/world.py`: `initialize_world`, `observe`, `available_actions`, `execute`, `advance_time`, `observe_consequences`, `evaluate_goal`, `save/restore_world_state`, `reset_world`. Agent observations, simulator state and verifier outputs are separate. |
| Perception (shared) | Every symbol, field name and entity type is hashed twice: once qualified by world family, once bare (shared meaning). There is no hand-made shared vocabulary. Named numbers get symlog/Fourier value features. Entity sets and egocentric image cells feed a set transformer, with attention pooling. |
| Belief | A GRU over (observation, goal, previous action, previous reward, episode/time flags, last memory read). |
| Fast episodic memory | A per-trial key/value ring buffer, written every step with a learned gate and read by learned attention. It persists across the episodes of a trial and is cleared between trials, so no gradient step is needed to use a new fact. |
| Skills | 8 learned skill embeddings. A high-level policy picks a skill when a learned termination fires (always at episode start), and the low-level policy is conditioned on the active skill. |
| Actions | A pointer head scores candidate action labels from any world (hashed), slot by slot for compositional actions. New worlds need no new output layer. |
| Value | Time-aware: a semi-Markov discount γ^(dt/dt_ref) per family, plus time features. |
| World model | An ensemble predicting the next observation embedding, reward and termination from (belief, memory read, action). It is trained on real transitions as an auxiliary loss; disagreement is the uncertainty signal. |
| Learner | Recurrent PPO over trial segments (RL²-style objective). It replays exactly: collection log-probs are reproduced to below 1e-4 in every adaptation mode (`tests/test_superlearner.py`). The learner is interchangeable (`learner.py`). |
| Adaptation modes | `full` (belief and memory across the episodes of a trial), `gru` (belief only, no external memory), `episode` (cleared every episode), `none` (cleared every decision: no adaptation beyond the current observation). |

## Environments

| Family | Source | Status |
|---|---|---|
| `cam/negotiation`, `cam/product`, `cam/bizdev` | AGENTRS simulators. A task = a simulator seed (hidden counterparty / market / partners). | In-process |
| `ext/vending` | ProsusAI/vending-bench (Apache-2.0): a seeded, stdlib-only 30-day vending business, no LLM. | In-process; adapter fixed before evaluation (`sl/worlds/vending.py` documents its assumptions) |
| Melting Pot | google-deepmind/meltingpot 2.4.1 (Apache-2.0): 49 substrates, 312 held-out scenarios with background bots. | Runs end-to-end in `/workspace/ext/venv-mp` (CPU TensorFlow). Worker adapter: Phase 2. |
| YC-Bench | collinear-ai/yc-bench: a deterministic discrete-event company simulation, 1 simulated year, about 130 decision epochs, no LLM in the world. | Runs end-to-end (driver `/workspace/ext/drivers/yc_bench_driver.py`), about 35 s per episode. Planned as a held-out business target (Phase 2). |
| Neural MMO 2.1.2 | MIT. | Runs in `/workspace/ext/venv-nmmo` (Python 3.10). Not yet adapted. |
| CEO-Bench | Runs without LLMs, but has no license and costs 0.06–3.5 s per simulated week. | Not used. |
| Magentic Marketplace | Both sides are LLM agents, with no world dynamics. | Not used. |

## Phase 1 protocol (fixed 2026-11-18 before any Phase 1 result)

Families: N = negotiation, P = product, B = bizdev, V = Vending-Bench. Held-out targets: **B** and **V**.

**Pretraining (lifetimes).** 2M real interactions each. Trials of K = 3 episodes. Lifetime checkpoints at
250k / 500k / 1M / 2M.
- `P_N_s1`: {N}
- `P_NP_s1`: {N, P}
- `P_NPV_s1` and `P_NPV_s2`: {N, P, V}, target B
- `P_NPB_s1` and `P_NPB_s2`: {N, P, B}, target V
- Ablations of N+P+V: `A_NPV_gru` (no external memory), `A_NPV_episode` (no cross-episode memory), `A_NPV_noskill`
  (no skills), `A_NPV_nowm` (no world-model loss)

**Transfer.**
- Budget: 300k interactions on the target family alone.
- Inits: scratch, or `life_2000000` of a lifetime. The checkpoint rule is fixed, never "best".
- Seeds 1–3, with identical configuration for scratch and experienced inits.
- Evaluation at 0 / 10k / 25k / 50k / 100k / 200k / 300k on 48 validation tasks (one 3-episode trial each), in modes
  `full` and `episode`.
- Source families are monitored for retention.
- Sealed test: 96 test tasks, final only.

**Measures.**
- **Fast adaptation (B):** with frozen weights, episode-3 minus episode-1 success/score within trials, in `full`
  vs `episode` vs `none` (difference in differences).
- **Transfer (C):** ΔAUC = mean over seeds of [AUC of validation success (B) or score (V) over the transfer
  milestones] for an experienced init minus scratch. Also the interactions needed to reach success ≥ 0.10 (B) or
  score ≥ 1.5 × start (V).
- **Breadth:** the trend of ΔAUC across N → NP → NPV (target B) and N → NP → NPB (target V).
- **Retention:** source-family validation during transfer.
- **Components:** transfer from each ablation lifetime vs the full lifetime.

**Success criteria.**
- **Positive:** ΔAUC > 0 with a bootstrap 90% CI (over seeds × tasks) excluding 0, on both targets.
- **Stronger:** ΔAUC increases with breadth on at least one target.
- **Negative (for this setup):** every ΔAUC CI includes or lies below 0 after the full Phase 1 budget.
- **Plateau (persistent runs):** validation AUC gains < 0.01 over 3 consecutive milestones.
- **Failure:** a NaN or diverging loss, or a crash loop. The queue stops the job with `CRASH_REPORT.md`.

**Budget.**
- Phase 1: 10 lifetimes × 2M + 28 transfer runs × 300k = 28.4M real interactions, about 21 GPU-process-hours.
- Hard cap: 40 GPU-process-hours on the existing A6000 instance. No additional paid compute or services.
- Checkpoints: latest every 10 min plus at phase changes; light (weights-only) milestone and lifetime checkpoints;
  GitHub Releases every 3 h and at the end.
- Final tests are never used for selection or tuning.

**Not comparable** (reported separately): the original PPO CAM and the hybrid runs used different reward contracts,
observation interfaces and task sampling. Frozen-LLM-direct numbers (hybrid arm D) come from 60 episodes per family
under a different interface.

### Protocol amendment, 2026-11-18 22:15 UTC (before any valid Phase 1 result)
The first four Phase 1 pretraining runs (sl-exp-v1/v2) were stopped after 4–20 minutes and set aside in
`runs/sl/_invalid_v2/`.
- **Cause:** the CAM-world adapter lacked the CAM-v2 collector's per-episode decision cap. Invalid actions do not
  advance the CAM simulators' clocks, so the policies learned to stall indefinitely and avoid every terminal outcome.
- **Fix (sl-exp-v3):**
  - a cap of 2 × `max_steps` decisions, with timeout counted as failure;
  - −0.02 per decision, and −0.02 more per invalid action;
  - an automatic reward-contract audit at the start of every run (`strict_audit`). Scripted random and stalling
    policies must terminate within the cap, and inaction must never earn a positive return.
- **Unchanged:** the protocol, budgets and criteria above.

### Protocol amendment 2, 2026-11-18 22:30 UTC (before any valid result)
- **Cause:** family weights are now shares of *interactions*, not of trials. Under per-trial sampling, Vending-Bench
  trials (about 1,800 decisions) held about 90% of the N+P+V lifetime's experience.
- **Fix (sl-exp-v5):** the next trial goes to the family furthest below its target share of agent-steps.
- **Superseded runs:** the three runs started under v3 were stopped after 2–13 minutes (`runs/sl/_invalid_v3/`).

### Protocol amendment 3, 2026-11-19 00:10 UTC (first results showed a degenerate optimum)
**The v5 Phase 1 runs learned to end episodes as early as possible.**
- The negotiation-only lifetime's policy entropy fell from 4.75 to 0.05 within 40k interactions, with one decision
  per episode. Bizdev ran idle to its limit, and every CAM family scored 0–12% validation success.
- **Cause:** the CAM adapter charged a cost per decision but not the unused decision budget on failure, so
  quitting at once was the cheapest failure. The CAM-v2 contract had a failure-horizon cost.
- Two further defects came to light while fixing it:
  - some simulators change `max_steps` mid-episode, so the decision budget is now fixed at reset;
  - CAM world snapshots omitted the adapter's own state.
- The reward audit now also runs an early-termination policy, which uses simulator snapshots to end episodes
  whenever any action can. Ending early must never pay more than failing at the deadline. v9 passes on all four
  families (negotiation: −0.56 either way).
- **Bizdev metric:** its primary metric becomes verified progress (fraction of required partners committed),
  with threshold 0.5. Every learner so far (original PPO CAM, CAM-v2 PPO arms, superlearner) has 0% bizdev
  success within these budgets, so a success-based AUC cannot distinguish arms. Success is still reported.
- **Learning restored:** before restarting, a v9 negotiation-only diagnostic (200k interactions) reached 8–10%
  validation success, with verified progress rising from 0.29 to 0.53. The original PPO CAM was at 8% at 200k.
- All v5 Phase 1 runs (CAM families) are superseded and set aside. Phase 1 restarts on sl-exp-v9 with the same
  protocol. Melting Pot (native rewards, no per-decision costs) is unaffected.

### Protocol amendment 4, 2026-11-19 01:30 UTC (before any v9/v10 Phase 1 result)
**Matched-budget diagnostics on negotiation** (150k interactions, 48 validation tasks):

| Configuration | Validation success | Verified progress |
|---|---|---|
| 3-episode trials, trial-level returns, skills (v9 base) | 8.3% | 0.537 |
| same, no skills | 10.4% | 0.571 |
| 3-episode trials, per-episode advantages, skills | 12.5% | 0.629 |
| same, no skills | 23.6% | 0.663 |
| single-episode trials, no skills (seeds 1 and 2) | 41.7% and 14.6% | 0.734 and 0.609 |

- **Trial-level returns:** with memory kept across a trial's episodes, the return summed over the whole trial
  slowed mastery (−0.09 progress in both pairs).
- **Single-episode trials:** their apparent advantage mostly did not replicate across seeds.
- **New base learner:** `episode_returns: true`. Advantages are computed per episode, while belief and memory still
  persist across the trial.
- **Kept under test:** the trial-level (meta-RL) objective remains an ablation (`A_NPV_rl2_s1` and its bizdev
  transfer). Skills stay in the base architecture; their mild deficit is within noise and is tested by
  `A_NPV_noskill_s1`.

### Protocol amendment 5, 2026-11-19 04:30 UTC (before any experienced-transfer result)
**Problem: the 2M transfer checkpoint is not reachable in time.** Each lifetime has an 8 h wall-clock cap (Phase 2:
10 h). With 8 Phase 1 and 2 Phase 2 jobs sharing the GPU, training throughput is about 75–83 interactions/s per
lifetime. Projections at 04:00 UTC:

| Lifetime | Interactions | Hours needed to reach 2M | Hours left |
|---|---|---|---|
| P_NPV_s1 | 504k | 6.4 | 5.2 |
| P_NP_s1 | 459k | 6.5 | 5.6 |
| P_N_s1 | 303k | 6.8 | 5.8 |
| P_NPB_s1 | 582k | 5.0 | 5.4 |

- **Phase 1:** `life_2000000`, the checkpoint the transfer protocol required, would not exist for most sources.
- **Phase 2:** the first milestone evaluation alone (10 families) takes about 78 min per mode, so `P_MP6_s1` cannot
  reach `life_4000000` within its cap.

The running lifetimes are not changed. The changes below apply only to transfer jobs that have not started:
- **Primary experience level: `life_1000000`** for every source, in both phases. Every lifetime reaches 1M within its
  cap, so all breadth arms are compared at equal lifetime experience.
- **Earlier start:** an experienced-transfer job starts as soon as its source checkpoint exists (queue `wait_for`),
  instead of after the source lifetime finishes.
- **Lifetime-level curve:** single-seed transfers from `life_250000` and `life_500000` of the full-breadth source for
  each target (`T_B_fromNPV{250k,500k}_s1` and `T_V_fromNPB{250k,500k}_s1`).
- **2M level where reached:** single-seed transfers from `life_2000000` (`T_*_from*2M_s1`, Phase 2
  `T_*_fromMP6at2M_s1`) run only if the source reaches it. Otherwise the queue marks them `skipped`.

Unchanged: transfer budgets, seeds, evaluation schedule, metrics, success criteria and all code; only the queue
configuration changed. The queue orchestration moved to its own snapshot (`sl-ops-v11`, worktree
`/workspace/agentrs-sl-ops`), while jobs still run the frozen experiment code (`sl-exp-v10` for Phase 1, `sl-exp-v8`
for Phase 2).

### Protocol amendment 6, 2026-11-19 05:00 UTC: reward-contract defect (before any experienced bizdev result)
**Symptom.** In all three v10 bizdev scratch runs, validation progress (the fraction of required partners committed)
fell from 0.26 at initialisation to 0.00 by 300k interactions: 0.257 → 0.163 (25k) → 0.004 (100k) → 0.000. The
trained policy's return (−0.80) is higher than the initial policy's (−1.17), so the collapse is reward-optimal.
Bizdev progress also fell in every lifetime that trains on bizdev (`P_NPB_s1`: 0.26 → 0.00 at 250k). No lifetime
has any product success (product was also at 0% for every earlier CAM learner).

**Cause.** The contract's progress shaping, `F = shaping × (γ^dt · Φ(s') − Φ(s))`, used raw elapsed hours as `dt`. The
learner discounts each step by `γ^(dt / mean dt)` (≈ 0.99). The two discounts must be the same, or the shaping no
longer telescopes, and holding verified progress `p` for one step then costs `shaping × (0.99 − γ^dt) × p`.
Measured mean step durations and the resulting cost:

| World | Mean step | Cost per step of holding progress `p` |
|---|---|---|
| negotiation | 3–7 h | about 0.01–0.02 × p |
| bizdev | 38 h | 0.15 × p |
| product | about 400 h | 0.49 × p (γ^dt ≈ 0) |

Vending-Bench and Melting Pot use no shaping and are unaffected. The reward audit (stall, early-termination and
random policies) could not see this: their returns were all consistent with the defect.

**Fix (`sl-exp-v11`).**
- The contract takes the learner's per-step discount explicitly (`RewardContract.components(…, disc)`, with
  `runner.learner_discount` used by both the runner and the audit).
- New audit invariant: on every completed episode of every scripted policy, the shaping telescopes under the
  learner's discount (`Σ_t D_t · F_t = −shaping · Φ(s_0)`).
- Regression tests: telescoping holds in the production runner for product, bizdev and negotiation, and the audit
  flags the v10 formula.

**Consequences for the protocol.**
- **Valid as measured and kept:**
  - every running v10 job;
  - the Vending-target transfer series, whose target reward has no shaping;
  - the negotiation lifetimes, where the bias is small;
  - the v10 bizdev scratch runs, as the record of the defect.
- **Bizdev target moves to v11.** All bizdev-target runs (scratch × 3 and every experienced init) run on
  `sl-exp-v11` as `T_B11_*`. Experienced inits are the v10 lifetimes' `life_*` checkpoints, so both arms learn the
  target under identical code. The never-started v10 `T_B_from*` jobs are dropped.
- **Disprovable test (pre-registered).**
  - **H6:** the shaping-discount mismatch causes the bizdev collapse.
  - **Prediction:** with v11, every bizdev scratch seed (s1–s3) keeps validation progress ≥ 0.20 at 100k and 300k, against
    0.004 and 0.000 for v10.
  - **Disproof:** v11 bizdev progress at 300k < 0.10 in at least 2 of 3 seeds.
  - **Product (secondary):** `D_P11_s1` vs `D_P10_s1` (product only, 300k, identical configuration). Prediction:
    v11 validation progress at 300k exceeds v10's by ≥ 0.10.
- **Lifetimes stay on v10.** `A_NPV_*` and `P_*_s2` stay on v10 so they remain comparable to `P_*_s1`. Lifetimes
  under the fixed contract are a separate later series.

### Protocol amendment 7, 2026-11-19 06:10 UTC: learned inaction in product and Vending-Bench (before any curriculum result)
**Finding.** Where a world needs an up-front cost and a coordinated sequence of actions before any payoff, every learner
converges to the reward audit's do-nothing return. The report now checks this mechanically (section 0b).

| World | Learned validation return | Do-nothing return | Simple scripted policy, same action grammar, no hidden information |
|---|---|---|---|
| product (v10 and v11, every lifetime) | −1.04 | −1.04 | research all segments, build the 3 most-requested features, price P2, ad campaigns at B3: **63% success**, 7% bankrupt (100 val tasks); 70 of 217 variants succeed sometimes |
| Vending-Bench (`T_V_scratch_s1..3`, `P_NPV_s1`) | −3.60 (score 0.759) | −3.60 (score 0.760 = 30 days × €12 fee, no business) | order q160 of 5 cheap products, restock, price ×3: **score 1.03** mean, max 1.22 (12 val tasks) |
| bizdev v10 | −0.80 | −0.80 | (fixed by v11: v11 learns, −0.08) |
| negotiation | +0.25 to +0.31 | −0.56 | (learned) |

The scripted policies are test fixtures only (never shown to or imitated by the learner). Product training saw 2
successes in 17,362 episodes (`D_P11_s1`), so success never reaches the gradient and the learner minimises bankruptcy
risk by doing nothing. In Vending, orders below a supplier's minimum are refused (single-product orders need q80–q160),
paid orders leave net worth until delivery 2–12 days later, and selling needs order → delivery → restock into a
matching slot → price; random exploration never completes the chain (training score never exceeded 0.76 after the
first iterations).

**Consequences.**
- **Interpretation of the pre-registered Vending series (kept, valid as measured).** Vending-target learning curves
  and transfer ΔAUC measure how quickly an init stops losing money (reaches inaction), not business competence.
  Experienced inits that already do nothing start near 0.72–0.75. Every Vending result is reported with this caveat.
- **Product in lifetimes** contributes inaction only. NP/NPB/NPV breadth comparisons therefore add worlds the agent
  never learns, which must be read as such.

**Fix under test (`sl-exp-v12`): a training-only goal curriculum.**
- Training tasks of a curriculum family carry a difficulty `d`. Product scales its user target by `d` (floor 50 users);
  the mechanism is CAM-v2's, which the superlearner had not used.
- After every 256 training episodes at the current `d`, `d` rises ×1.25 if training success is above 50% and falls ÷1.25
  if below 20% (range [0.02, 1], start 0.2).
- Validation and test tasks never carry a difficulty, so every reported number is at the full goal. With an empty
  `curriculum` (the default) v12 behaves exactly like v11.

**Disprovable test (pre-registered).**
- **H7:** product fails because training almost never succeeds, and the curriculum fixes it.
- **Runs:** `D_P12c_s1`, `D_P12c_s2` (v12, product only, 300k), configured exactly like `D_P11_s1` (v11, the control)
  plus the curriculum.
- **Confirmed if** both seeds reach validation progress ≥ 0.20 and success > 0 at 300k.
- **Disproved if** both seeds have progress < 0.05 at 300k (still the inaction basin), or no seed's training
  difficulty ever rises (success not learnable even at the easiest goals).
- **Next:** if confirmed, the curriculum becomes part of the next lifetime series. Vending needs a different mechanism
  (it has no scalable goal; its obstacle is the action chain and order accounting): a separate hypothesis, tested
  after H7.

### Results of the amendment 6 and 7 tests (2026-11-19 07:40 UTC, mechanical verdicts from `scripts/sl_report.py`)
- **H6: CONFIRMED.** Bizdev scratch validation progress at 100k / 300k:
  - v11 (fixed contract): 0.486 / 0.704, 0.424 / 0.771, 0.454 / 0.738;
  - v10: 0.009 / 0.000, 0.000 / 0.000, 0.002 / 0.000.
  - Caveat: on bizdev's harder test distribution (more dependencies and rivalry, tighter cap) v11 reaches only 0.46
    progress and 0.7% success (`T_B11_scratch_s1`). In-distribution learning does not yet generalise to the shifted
    distribution.
- **Product secondary (v11 vs v10): NOT CONFIRMED** (Δ = +0.024). The shaping fix alone does not fix product
  (amendment 7).
- **H7: neither confirmed nor disproved (mixed).** Validation at full targets, 300k:

  | Run | Progress | Success | Note |
  |---|---|---|---|
  | `D_P11_s1` (control) | 0.024 | 0 | |
  | `D_P12c_s1` | 0.285 | 0 | final validation at the same budget: 0.323 / 0.014 |
  | `D_P12c_s2` | 0.439 | 0.035 | |

  - The disproof conditions clearly fail: the curriculum lifts product out of the inaction basin. Training difficulty
    rose 9–10 times to 0.25–0.39 of the full target, and validation bankruptcy is 13–19% (the agent now takes risks).
  - The success condition holds in one seed, so full-target success needs a longer budget than 300k.
  - Test split (B2B-like markets): 0.091 vs 0.040 progress.

### Protocol amendment 8, 2026-11-19 06:35 UTC: Vending-Bench adapter reward-timing defects (before any v13 result)
**Cause of Vending's learned inaction.** The discounted objective the learner optimises (γ = 0.99 per step) prefers doing
nothing. Discounted returns from the start of an episode, 12 validation tasks (`scripts/sl_vending_discount.py`):

| Policy | Score | Discounted return, adapter v1 (sl-exp ≤ v12) | Discounted return, v13 (accrual + world time) |
|---|---|---|---|
| wait to the next day, 30 calls | 0.760 | −3.12 | −0.66 |
| stall: ~20 no-op calls a day (what the trained agents do) | 0.760 | **−0.54** (best) | −0.62 |
| scripted business (order q160, restock, price ×3) | 1.031 | −2.18 | **−0.41** (best) |

The two defects, both in our adapter (the benchmark is unchanged):
- **Cash-timed orders.** The benchmark's net worth counts card money in transit but not paid goods in transit, so an
  order is a loss until delivery 2–12 days (40–240 calls) later.
- **Per-call discounting.** `dt` was 1 per call, so spreading the €12 daily fee over many cheap calls hid most of it.

**Fix (`sl-exp-v13`, `sl/worlds/vending.py`).**
- **Accrual accounting:** the per-call reward counts paid, not-yet-due orders at cost. Overdue undelivered orders are
  written off when due. The terminal call settles at the benchmark's own net worth, so every episode's undiscounted
  return is still exactly (final − initial net worth) / 100 (tested).
- **World time:** `dt` = simulated minutes per call (semi-Markov discount by world time, as for the CAM worlds).
- Success, score, observations and actions are unchanged. v13 also contains v12's curriculum, which is unused here.

**Consequences.**
- The v10 Vending series stays as the record. Its results are valid as measured: they measure learning to do nothing.
  Its pre-registered transfer jobs are moved to the end of the queue: kept, but behind every bizdev, curriculum and
  ablation job.
- **Disprovable test (pre-registered).**
  - **H8:** these two defects cause Vending's learned inaction.
  - **Runs:** `T_V13_scratch_s1`, `T_V13_scratch_s2`, configured exactly like `T_V_scratch_s*` on v13.
  - **Confirmed if** both seeds reach validation score ≥ 0.85 at 300k, with a learned return different from the
    do-nothing return.
  - **Disproved if** both seeds are ≤ 0.78 at 300k (still inaction, although the objective now favours the business).
- **If confirmed,** a Vending transfer series on v13 (`T_V13_from*`, same sources) replaces the v10 one as the Vending
  measurement.

**H8 result (2026-11-19 09:45 UTC): DISPROVED.** Validation score at 300k: `T_V13_scratch_s1` 0.774, `T_V13_scratch_s2`
0.758 (v1 adapter: 0.758–0.759). Both are below the 0.78 threshold, so the timing defects are not a sufficient cause.
- **What changed.** The v13 agents no longer stall. Their returns (−3.68, −3.73) differ from the do-nothing return, and
  they search, order, restock, price and clear.
- **Why they still fail** (`best` checkpoints, 8 validation tasks):
  - 89–92% of their orders are refused;
  - order quantities are spread almost uniformly over q10–q160;
  - single-product orders below about q80 fall under every supplier's minimum order value.
- **Information gap.** The benchmark shows that minimum in its `check_offers` reply, but the adapter dropped it from the
  observation. A refused order costs only simulated time.

### Protocol amendment 9, 2026-11-19 10:00 UTC: Vending order validity (before any v14 result)
**Fix (`sl-exp-v14`, `sl/worlds/vending.py`; v13's accrual accounting and world-time `dt` are kept).**
- The observation now carries each quoted supplier's minimum order value, from the benchmark's own `check_offers` reply.
- The order grammar offers only (supplier, product, quantity) choices that the agent's own quote shows to be valid:
  quote × quantity ≥ 1.1 × minimum and ≤ balance / 1.1. The margin covers the benchmark's small volume discounts.
- Nothing hidden is used.
- **Effect on a uniformly random policy** (8 validation tasks): order acceptance 26% → 98%, mean score 0.627 → 0.720.
  But 6 of 8 episodes now end with the operator terminated for unpaid fees, because valid orders spend the cash. Cash
  management stays the agent's problem; the grammar removes only orders the benchmark would refuse.

**Disprovable test (pre-registered).**
- **H9:** Vending fails because most orders are refused for want of information the benchmark gives the operator.
- **Runs:** `T_V14_scratch_s1`, `T_V14_scratch_s2` (v14, otherwise exactly like `T_V13_scratch_s*`, 300k).
- **Confirmed if** both seeds reach validation score ≥ 0.85 at 300k with a return different from the do-nothing
  return.
- **Disproved if** both are ≤ 0.78.

### Protocol amendment 10, 2026-11-19 11:40 UTC: Vending closure was cheaper than paying (before any v14/v15 result)
**What happened.** Neither `T_V14_scratch` run trained. Both production smokes stopped at the reward-contract audit
(12 scripted episodes per policy), which requires that ending an episode early never pays more than failing at the
deadline. With orders now accepted, the audit's `terminate` policy spent its cash on stock, stopped paying the EUR 12
daily spot fee and was closed after 10 unpaid days. The benchmark stops charging fees after closure and keeps the stock in
net worth at cost, so closure cost less than holding out to day 30: −3.54 vs −3.60, even after the −1 violation cost.
The unit test ran the same audit with 3 episodes and passed; it now uses the production 12.

**Cause.** An adapter hole, not benchmark behaviour: the benchmark's own score gives a closed run 0
(`reward = balance if completed else 0`). It also forgives unpaid fees once a later fee is paid.

**Fix (`sl-exp-v15`, `sl/worlds/vending.py`; v14's order validity and v13's accrual and world time are kept).**
- The operator stays liable for the whole 30-day contract. Fees owed = EUR 12 × elapsed days − fees paid, and once the
  run is over (finished or closed), EUR 12 × 30 − fees paid.
- Reward, score, progress and success use the settled value: benchmark net worth − fees owed. A run that pays every fee
  scores exactly as under v13/v14, so the do-nothing reference (0.760) and amendment 8's reachability evidence (a scripted
  business that pays its fees) are unchanged.
- The observation adds `fees_owed`, computed from the operator's own payment history.
- **Audit (12 episodes, production settings):** stall −3.60 (unchanged), terminate −3.54 → −5.54, random −4.89 → −6.93;
  passed.

**Test.** H9 is unchanged in substance and is now tested with v15, because v14 never produced a result. Runs:
`T_V15_scratch_s1`, `T_V15_scratch_s2` (v15, otherwise exactly like `T_V13_scratch_s*`, 300k); same thresholds.

**Risk recorded in advance.** Random play now loses more than doing nothing (−6.93 vs −3.60), which strengthens the
inaction attractor seen in v10–v13. If both v15 seeds again settle on the do-nothing return, H9 is disproved as stated:
valid orders alone do not make Vending learnable for this learner. Cash management, the remaining obstacle, would then
need its own hypothesis, not a reward change.

## Phase 2 (Melting Pot), `configs/sl/queue_phase2_mp.yaml`
**Lifetime** (`P_MP6_s1`, `P_MP6_s2`):
- Self-play with parameter sharing (every player is a learner stream with its own belief and memory) on 6
  substrates: coins, collaborative_cooking__asymmetric, commons_harvest__open, clean_up,
  prisoners_dilemma_in_the_matrix__repeated, stag_hunt_in_the_matrix__repeated.
- 4M learner agent-steps; trials of 2 episodes.
- Zero-shot monitors: the held-out substrates commons_harvest__closed and bach_or_stravinsky_in_the_matrix__repeated,
  and the held-out scenarios prisoners_dilemma_in_the_matrix__repeated_0 and coins_0 (the benchmark's
  background bots as unfamiliar co-players).

**Transfer:** 1M agent-steps on each held-out substrate, scratch vs `life_4000000`, seeds 1–2.
Amendment 5 changes this to `life_1000000`.

**Measure:** score is the agent's own episode return. Melting Pot defines no binary success.

**Not yet implemented:** archived-opponent populations (Milestone 6). Opponents are concurrent copies of the learner.

## Crash safety (section 21)
- `agentrs/runtime_safety.py`:
  - static main-guard check (the original failure);
  - production-command smoke runs;
  - crash signatures and a crash-loop ledger that writes `CRASH_REPORT.md`;
  - disk guard.
- **Trainers:**
  - sealed tests are written before any optional diagnostic;
  - diagnostics are persisted one by one and never fatal: an exception is recorded, and an interrupted attempt is
    retried once, then abandoned;
  - a finished run is never re-entered.
- **Queue:** before its first launch, each job's exact production command runs start to finish with budget-only
  overrides. A failing smoke run means the job never starts.
- **Tests:** `tests/test_runtime_safety.py` and `tests/test_superlearner.py`.
- **Earlier runs:** the end-of-run diagnostics that failed in completed runs were not rerun (per instruction).
