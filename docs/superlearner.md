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

**H9 result (2026-11-19 15:30 UTC): DISPROVED.** Validation score at 300k: `T_V15_scratch_s1` 0.734, `T_V15_scratch_s2`
0.684 (threshold for confirmation 0.85; for disproval ≤ 0.78). The agents did not settle on inaction: they keep ordering,
and over 300k the share of validation episodes closed for unpaid fees falls from 0.78 to 0.17 (s1) and 0.41 (s2),
while the mean return rises from −6.2 to −4.2 and −5.2. That is still below the do-nothing return (−3.60), and neither
seed has flattened. Valid orders plus a fee contract that cannot be escaped make Vending a cash-management problem the
learner is working on but has not solved in 300k. The next Vending hypothesis has to target cash management (or a
longer budget), not order validity.

### Protocol amendment 11, 2026-11-19 12:10 UTC: negative transfer to bizdev (before any v16 result)
**Observation (finished runs only; `scripts/sl_report.py` section 2).** Experienced inits learn bizdev more slowly than
scratch. Learning AUC of validation score over 0-300k (scratch, 3 seeds: 0.518). Table updated 15:30 UTC with every
seed complete (at 12:10 only fromN s1 and fromNP s1-s2 had finished: −0.133 and −0.210); no v16 run had started.

| Init | Seeds | ΔAUC vs scratch (90% CI) | Score at 0k (scratch 0.258) | Score at 300k (scratch 0.738) | Final test (scratch 0.440) |
|---|---|---|---|---|---|
| fromN (P_N_s1 @ 1M) | 3 | −0.071 (−0.166 to +0.024) | 0.020 | 0.758 | 0.436 |
| fromNP (P_NP_s1 @ 1M) | 3 | −0.156 (−0.247 to −0.066) | 0.015 | 0.701 | 0.417 |
| fromNPV (P_NPV_s1 @ 1M) | 3 | +0.008 (−0.041 to +0.056) | 0.043 | 0.728 | 0.424 |
| fromNPV250k | 1 | +0.069 | 0.113 | 0.775 | 0.455 |
| fromNPV500k | 1 | −0.214 | 0.102 | 0.549 | 0.311 |

Every experienced init starts far below scratch (0.02-0.11 against 0.26). Only fromNP's deficit is significant; all
inits end near the same final test score.

The report had averaged learning curves over whatever range each run had covered, so runs still in progress counted with
lower values. It now counts only runs that reached the budget, paired by seed.

**Diagnosis (offline, frozen weights, 48 validation tasks × 3 episodes; `scripts/sl_probe_actions.py`,
`scripts/sl_probe_temperature.py`).**
- **Overconfidence on never-trained actions.** At transfer start the fromN policy chooses WAIT 78% of the time, against
  about 25% for a random init. Its normalised policy entropy is 0.69 (random init: 1.00) and its score 0.02 (random 0.21).
  Negotiation has no WAIT action, so this is not transferred meaning: the bizdev labels' embedding rows were never
  trained, and the grown policy query turns their random values into confident, arbitrary preferences.
- **Mis-transferred shared labels.** Bizdev's PITCH takes the N0-N10 argument labels that negotiation's PROPOSE uses.
  With every never-trained label row zeroed (an exact retrofit: plain Adam leaves unused rows bit-identical to their
  initial values; P_N trained 75 of 65,536 rows), fromN's action types are uniform, but it still scores 0.07. Its
  negotiation-trained preferences over N levels are wrong in bizdev.
- **v10 learned inaction carried by labels.** For fromNP, zeroing changes nothing (entropy 0.51 → 0.50). Bizdev's WAIT
  shares its label and word rows with product's WAIT, which P_NP learned to prefer under the v10 shaping defect
  (amendments 6 and 7).
- **Softening the preferences recovers random-level play.** Dividing fromN's logits by a temperature T: T = 3 gives
  entropy 0.96, score 0.15; T = 10 gives 1.00, 0.21-0.24 (random level); T = 30 gives 1.00, 0.285. The representation
  is not harmful; its confident action preferences are.

**Architecture change (`sl-exp-v16`): a familiarity gate.**
- Each model counts its training interactions per world, n_f (a checkpointed buffer).
- Before each action choice, the logits are shrunk toward their mean over the valid options by g = n_f / (n_f + n0).
  A world never trained in starts with a uniform choice, and preferences strengthen as experience there accumulates.
- The representation (perception, belief, memory, world model, label embeddings, policy query) is untouched and is
  free to transfer.
- The collector records g; the learner's replay reuses it, so PPO log-probs stay exact (tested).
- **n0 = 4,096:** one training batch (128 worlds × 32 steps). g = 0.5 after the first batch in a world, 0.92 at 50k.
- **Source lifetimes trained before v16 carry no counts.** Their n_f is reconstructed as interactions × family weight
  (e.g. P_NP @ 1M: 0.5M each for negotiation and product), so retention monitors on source families keep g ≈ 1.
- With `fam_gate_n0 = 0` (the default), v16 is identical to v15.

**Disprovable test (pre-registered): H10.**
- **Claim:** the negative transfer is caused by confident action preferences not earned in the new world.
- **Runs:** `T_B16_scratch_s1/s2`, `T_B16_fromN_s1/s2`, `T_B16_fromNP_s1/s2`, identical to the `T_B11_*` runs except for
  the v16 code and `model.fam_gate_n0 = 4096`. The bizdev reward contract is v11's (v12-v15 change only product
  curriculum and Vending).
- **Primary measure:** learning AUC of validation score, 0-300k, paired by seed against `T_B16_scratch`.
- **Confirmed if** both gated transfer groups have mean ΔAUC ≥ −0.02. **Disproved if** gated fromNP, the only
  significant ungated deficit (−0.156), stays ≤ −0.10. Anything else is mixed. (Revised 15:30 UTC, before any v16 run
  started: the 12:10 rule needed both groups ≤ −0.10, which the complete fromN deficit, −0.071, cannot reach even
  without a gate.)
- **Mechanism check:** gated transfers start at score ≥ 0.20 at 0k.
- **Also reported:** the gate's cost on scratch (T_B16_scratch vs T_B11_scratch) and each gated transfer vs its ungated
  twin (same init and seed).
- **What the gate cannot fix:** if the experienced representation still learns more slowly once preferences are neutral,
  the cause is in the representation (e.g. lost plasticity), not the action head. H10 is then disproved, and the next
  hypothesis targets plasticity.

### Protocol amendment 12, 2026-11-19 15:41 UTC: forgetting during transfer; third Melting Pot seeds (before any result of these runs)
**Observation: forgetting (T_B11, every experienced transfer).** Transfer runs train on bizdev alone and monitor their
source worlds with frozen-split validation. Negotiation success (validation, 0k → 300k):

| Init | s1 | s2 | s3 |
|---|---|---|---|
| fromN | 0.660 → 0.188 | 0.674 → 0.049 | 0.667 → 0.215 |
| fromNP | 0.632 → 0.111 | 0.611 → 0.333 | 0.618 → 0.146 |
| fromNPV | 0.361 → 0.160 | 0.312 → 0.097 | 0.368 → 0.194 |

300k interactions of bizdev erase most of what 1M interactions built in negotiation. Scratch runs hold 0.00-0.04. A
continual learner must keep old worlds while it learns new ones; this one does not.

**Disprovable test (pre-registered): H11.**
- **Claim:** forgetting comes from training on the new world alone, not from a capacity conflict between the worlds.
  Rehearsing the source world on a small share of interactions keeps it, at little cost to learning the new world.
- **Runs:** `T_B16_fromNrh_s1/s2`: exactly `T_B16_fromN_s1/s2` (v16, gate on, P_N_s1 @ 1M), except the training mix is
  bizdev 0.9 and negotiation 0.1. The budget stays 300k interactions in total, so bizdev gets 270k; the learning
  curve's x-axis counts all interactions (rehearsal is paid for).
- **Measures:** negotiation validation success at 300k; bizdev learning AUC paired by seed against `T_B16_fromN`.
- **Confirmed if** negotiation success at 300k ≥ 0.50 in both seeds and the mean bizdev ΔAUC ≥ −0.03. **Disproved if**
  negotiation success at 300k ≤ 0.30 in both seeds: then 10% rehearsal does not protect the old world, and the conflict is
  in shared parameters (the next hypothesis would separate them). Anything else is mixed.
- Rehearsal uses the train split of negotiation tasks; validation uses the validation split, as for every family.

**Third Melting Pot seeds.** The Phase 2 transfers to commons_harvest__closed finished with n = 2 per arm. Both seeds of
the experienced arm start ahead (0k: 27.7 and 30.6 against 26.3 and 25.6) and then fall below scratch. Validation
evaluations are noisy (28 episodes; one run moves 32 → 19 → 29 between milestones), so two seeds cannot say whether the
later deficit is real. Added `T_closed_scratch_s3`, `T_closed_fromMP6_s3`, `T_bos_scratch_s3`, `T_bos_fromMP6_s3`. The
experienced arm of seed 3 starts from the second lifetime, `P_MP6_s2` @ 1M, so the estimate covers lifetime-seed variance
too; seeds 1-2 started from `P_MP6_s1`. Thresholds: none pre-registered for Melting Pot (amendment 5); reported as ΔAUC
with its 90% CI.

**Report.** The training-side memory-ablation diagnostic stores success AUC only. For score-metric families (Melting
Pot, Vending, bizdev) its zeros carry no information, and the report now says so instead of printing them as results.

### H10 result, 2026-11-19 19:05 UTC: DISPROVED (mechanical verdict, `scripts/sl_report.py` section 0)
All six v16 runs complete (300k each). Learning AUC of bizdev validation score, paired by seed against gated scratch:

| Init | s1 AUC (Δ) | s2 AUC (Δ) | Mean Δ | 300k validation | Final test |
|---|---|---|---|---|---|
| scratch | 0.508 | 0.576 | – | 0.752 | 0.454 |
| fromN | 0.449 (−0.058) | 0.360 (−0.216) | −0.137 | 0.791 | 0.448 |
| fromNP | 0.323 (−0.184) | 0.340 (−0.237) | −0.211 | 0.694 | 0.437 |

- The mechanism check passed: every gated run starts at 0.255 (s1) and 0.257 (s2), the scratch level.
- Gated vs ungated twin (same init and seed): fromN +0.067 and −0.048, fromNP +0.035 and +0.001, scratch −0.007 and
  +0.043. The gate changed nothing beyond seed noise.
- **Why it failed (training logs, `iters.jsonl`).** The gate hides the source's preferences instead of correcting them.
  - The gradient that reaches the raw logits is scaled by g, so while g is small the wrong preferences barely change.
  - g reaches 0.5 after one batch, and the preferences return. fromNP's training score falls from 0.24 (iteration 1) to
    0.02 by iteration 4 (16k), the ungated starting level.
  - The skill selector and termination are not gated: fromNP's iteration-1 KL is 0.14-0.16 while the pointer choice is
    uniform (scratch: 0.013).
- **What did reproduce in every run: exploration collapse.** Mean policy entropy and advantage std over iterations 3-40
  (8k-165k interactions):

| Run | Entropy | Advantage std | Learning AUC |
|---|---|---|---|
| T_B16_scratch_s1 / s2 | 4.02 / 4.13 | 0.13 / 0.20 | 0.508 / 0.576 |
| T_B16_fromN_s1 / s2 | 3.00 / 2.70 | 0.058 / 0.056 | 0.449 / 0.360 |
| T_B16_fromNP_s1 / s2 | 2.12 / 2.19 | 0.062 / 0.069 | 0.323 / 0.340 |

  The experienced agents act with low entropy, so their returns hardly vary (advantage std about 0.06) and the policy
  gradient has little to push on. Once one stumbles on reward (fromN_s1 near 200k, when advantage std jumps to 0.32), it
  learns faster than scratch: fromN's 300k validation (0.791) beats scratch's (0.752).
- **Status of H10.** The pre-registered test is disproved: the gate does not remove the negative transfer. Its causal
  clause (preferences, not representation) is not settled, because the gate never kept the preferences neutral after the
  first batch. Amendment 13 tests that clause directly.

### Protocol amendment 13, 2026-11-19 19:15 UTC: where the negative transfer lives (before any result of these runs)
**Pre-run check (`scripts/sl_probe_head_reset.py`; offline, frozen weights, 48 bizdev validation tasks, gate disabled
so raw preferences show).**
`init_reset` is a v16 training option, never used before: parameter-name prefixes listed there are not loaded from the
source, so those tensors keep the run's fresh initialisation, the same initialisation as scratch.

| Weights | Pointer entropy / maximum | Score |
|---|---|---|
| scratch init (seed 1) | 1.000 | 0.278 |
| P_N_s1 @ 1M | 0.670 | 0.023 |
| P_N_s1 @ 1M, action heads reset | 1.000 | 0.280 |
| P_NP_s1 @ 1M | 0.607 | 0.019 |
| P_NP_s1 @ 1M, action heads reset | 1.000 | 0.252 |

**Disprovable test (pre-registered): H12.**
- **Claim:** the negative transfer to bizdev lives in the transferred action-selection heads, whose source preferences
  collapse exploration in the new world. The heads are the pointer query (`query.`), slot decoder (`slot.`), skill
  selector (`hi.`) and termination (`term.`). Everything else transfers without harm: perception, belief, episodic
  memory, label and skill embeddings, value and world model.
- **Runs:** `T_B16_fromNhr_s1/s2` (P_N_s1 @ 1M) and `T_B16_fromNPhr_s1/s2` (P_NP_s1 @ 1M).
  - Exactly `T_B16_fromN` and `T_B16_fromNP` (frozen v16 code, gate on, same seeds and sources), except
    `init_reset: ['query.', 'slot.', 'hi.', 'term.']`.
  - No code change; the control is `T_B16_scratch` (same code and gate).
- **Primary measure:** learning AUC of bizdev validation score, 0-300k, paired by seed against `T_B16_scratch`.
- **Confirmed if** both head-reset groups have mean ΔAUC ≥ −0.02.
- **Disproved if** fromNPhr's mean ΔAUC ≤ −0.10. The deficit would then survive fresh heads, so it lives in the shared
  representation, value or world model, and the next hypothesis targets plasticity there.
- Anything else is mixed.
- **Mechanism check:** each head-reset run's mean entropy over iterations 3-40 is ≥ 0.9 × the same seed's scratch.
- **Also reported:** each head-reset run against its source with heads kept (`T_B16_fromN` and `T_B16_fromNP`, same
  seed). If a head-reset group's mean ΔAUC exceeds +0.02, the shared representation transfers positively. That reading
  is exploratory (n = 2), not a verdict.
- **What a confirmation would change.** Preferences in a new world would have to be re-learned there, not inherited, while
  the representation stays shared. That is an architecture change for multi-world lifetimes (heads that start fresh on
  entering a world), to be tested on Melting Pot as well as CAM before it is adopted.

**Melting Pot observations (n = 2 per arm; seed 3 running; no thresholds pre-registered, amendment 5).**
- **Zero-shot transfer is real in commons_harvest__closed.**
  - During the P_MP6 lifetimes, which never train on closed, closed validation rises from 24.1 to 30.3 (s1) and from 24.9
    to 33.3 (s2).
  - Scratch trained on closed itself reaches 30.0 at 1M.
- **Training on closed alone then erodes that lead.**
  - The experienced arm starts at 31.0 and ends at 26.5 at 1M; scratch rises from 24.1 to 30.0. ΔAUC −2.63 (90% CI −3.34
    to −1.91).
  - This is not exploration collapse: entropy is 1.6-1.9 against scratch's 1.7-2.1, and KL stays small.
- **bach_or_stravinsky does not train in either arm.** From about 40k on, entropy sits at its maximum (ln 8 = 2.08) and
  advantage std is about 0. Its gaps reflect starting weights and evaluation noise (ΔAUC +0.29, 90% CI −0.23 to +0.81;
  zero-shot during the lifetimes 1.24 → 1.70 and 1.03 → 2.52).

### Vending diagnosis, 2026-11-19 19:30 UTC (H9 follow-up; evidence only, no runs launched)
**Where the v15 agents lose money.** Frozen final weights, 16 validation tasks × 3 episodes. Probe:
`scripts/sl_probe_vending_ledger.py` (run with the frozen v15 `src` on `PYTHONPATH`) decomposes each episode's settled value.

| Per episode | T_V15_scratch_s1 | T_V15_scratch_s2 | s1 initial weights |
|---|---|---|---|
| Spent on delivered stock | 1,410 | 1,405 | 1,236 |
| Gross revenue (units) | 232 (107) | 74 (32) | 20 (9) |
| Stock left at cost | 1,169 | 1,248 | 1,174 |
| Cash balance at end (lowest) | 30 (0.7) | 10 (2.6) | 3 (0.4) |
| Fees paid / owed | 189 / 171 | 106 / 254 | 144 / 216 |
| Spoiled / marketing / paid but undelivered | 55 / 24 / 77 | 69 / 1 / 51 | 25 / 51 / 86 |
| Closed for unpaid fees | 9 of 48 | 23 of 48 | 36 of 48 |

- **Overstocking.** The agents turn about 94% of their cash into stock that mostly never sells.
- **Losses beyond the €360 of fees** come from spoilage, orders still in transit at the end, and marketing.
- **Calls go to the wrong actions.** About a third are `offers` (s1); 10% (s1) and 26% (s2) are `clear`, which returns a
  slot's stock to storage. That is neutral for net worth, but the slot then sells nothing.

**The adapter's reward and the benchmark's objective differ.** The benchmark tells the operator to "maximise your final
bank balance… Unfinished or bankrupt runs score zero… Inventory is not part of your money score". The adapter scores net
worth, which counts stock at cost. Under net worth, buying stock costs nothing until it spoils, and in v15 an unpaid fee
costs the same as a paid one.

**Reachability under both measures (`scripts/sl_reach_vending_money.py`).** A scripted operator uses only the orders
the agent's own grammar offers. It
restocks every slot, prices at a fixed multiple of unit cost, and stops ordering by a set day. Results over 6
validation seeds:

| Policy | Money score (benchmark) | Net worth (adapter) |
|---|---|---|
| Do nothing | 1,140 | 1,140 |
| x3 markup, 30% of cash, 3 products, last order day 10 | 1,138 (1,023-1,344) | 1,496 |
| x3 markup, 70% of cash, 8 products, last order day 10 | 840 | 1,575 |
| x2 markup, 30% of cash, 3 products, last order day 10 | 1,091 | 1,383 |
| v15 learners (s1 / s2) | – | 1,200 / 1,258 (settled 1,029 / 1,004) |

- **Under the adapter's measure there is a large reachable target.** Score 0.92-1.05, against 0.76 for doing nothing and
  0.67-0.69 for the learners. The v15 learners fail at selling: revenue 74-232 against 400-660 for the scripted operators.
- **Under the benchmark's own measure, no operator tried beats doing nothing on average.** Profit sits in stock valued at
  cost. Switching the reward to the money score would recreate a learned-inaction optimum (amendment 7) unless a
  profitable policy is shown first.
- **Consequence.** Vending results stay reported as the adapter's settled score, a different quantity from the
  benchmark's headline number. Business-transfer claims from Vending must say so. No new Vending hypothesis is
  pre-registered until one has a disproving test with a reachable target under the measure it uses.
- Earlier reachability scripts (`scripts/sl_reach_vending.py`) predate v14's order validity rules. Their operators now
  spend about €55 per episode because most orders are refused, so their figures no longer describe the current adapter.

### H11 and H12 results, 2026-11-19 22:20 UTC (mechanical verdicts, `scripts/sl_report.py` section 0)
**H11 (rehearsal keeps the source world without costing the new one): mixed.**

| Seed | Negotiation, rehearsal (start → 300k) | Negotiation, no rehearsal | Bizdev AUC, rehearsal | Bizdev AUC, no rehearsal | ΔAUC |
|---|---|---|---|---|---|
| 1 | 0.667 → 0.562 | 0.667 → 0.097 | 0.260 | 0.449 | −0.189 |
| 2 | 0.674 → 0.479 | 0.674 → 0.153 | 0.228 | 0.360 | −0.132 |

- **Rehearsal keeps negotiation.** It holds at 0.56 and 0.48, against 0.10 and 0.15 without rehearsal. Seed 2 misses the
  0.50 bar.
- **Bizdev pays for it.**
  - Mean ΔAUC is −0.160, against the −0.03 required.
  - Both rehearsal runs end at the do-nothing return in bizdev (−0.82 and −0.81, against −0.80), with final test 0.136.
- **Exploration is collapsed.** Entropy over iterations 3-40 is 2.80 and 2.88, against scratch's 4.02 and 4.13.
  - Rehearsal keeps negotiation's preferences alive in the shared action heads.
  - Bizdev uses the same action labels, so it inherits them (see H12).

**H12 (the negative transfer lives in the transferred action heads): mixed. fromNPhr meets the confirmation bar; fromNhr
misses it on seed 2. The mechanism check passed.**

| Init | Seed | AUC | Δ vs scratch | Δ vs same source, heads kept | Entropy / scratch |
|---|---|---|---|---|---|
| fromNhr | 1 | 0.512 | +0.004 | +0.063 | 0.96 |
| fromNhr | 2 | 0.484 | −0.092 | +0.124 | 0.96 |
| fromNPhr | 1 | 0.498 | −0.010 | +0.175 | 0.98 |
| fromNPhr | 2 | 0.562 | −0.014 | +0.222 | 0.96 |

- **Resetting the four action heads removes most of the deficit.**
  - Mean ΔAUC is −0.044 (fromNhr) and −0.012 (fromNPhr), against −0.137 and −0.211 with the heads kept.
  - Each of the four head-reset runs beats its heads-kept twin, by +0.06 to +0.22.
- **Why the verdict is mixed, not confirmed.** fromNhr's mean (−0.044) is below the −0.02 bar, because of seed 2 alone.
  The two scratch seeds differ from each other by 0.068, so n = 2 cannot separate −0.044 from zero.
- **No positive transfer over the full window.** No head-reset group beats scratch on 0-300k AUC.
- **The reset costs the source skill.** Negotiation success at the start of the head-reset runs is 0.014-0.035, against
  0.667-0.674 with the heads kept. The source world's policy lives in those heads.
- **Exploratory observation (not pre-registered; it generated H14 below).** In all four pairs the head-reset runs lead
  scratch over the first 50k interactions and trail it over 50k-300k. The two-world source leads the one-world source
  early in both seeds (+0.031, +0.029).

| Seed | Score AUC 0-50k / 50k-300k: scratch | fromNhr | fromNPhr |
|---|---|---|---|
| 1 | 0.257 / 0.558 | 0.299 / 0.554 | 0.330 / 0.531 |
| 2 | 0.279 / 0.636 | 0.324 / 0.516 | 0.353 / 0.603 |

**Melting Pot with seed 3 (n = 3 per arm; the third experienced run starts from the second lifetime, P_MP6_s2).**

| Substrate | Seed | Scratch AUC | Experienced AUC | Δ |
|---|---|---|---|---|
| closed | 1 | 32.00 | 28.94 | −3.06 |
| closed | 2 | 27.70 | 25.51 | −2.19 |
| closed | 3 | 26.31 | 27.35 | +1.04 |
| bos | 1 | 1.76 | 1.74 | −0.02 |
| bos | 2 | 0.52 | 1.12 | +0.60 |
| bos | 3 | 1.42 | 1.28 | −0.14 |

- **closed: the deficit is no longer significant.**
  - ΔAUC is −1.41 (90% CI −3.46 to +0.65), against −2.63 at n = 2.
  - The seed from the second lifetime gains.
  - Final test: 25.4 experienced, 24.3 scratch.
- **bos:** ΔAUC +0.15 (90% CI −0.23 to +0.53); final test 1.36 against 1.02.
- **Both experienced arms start far ahead.** At 0k, closed is 31.0 against 24.1, and bos 2.24 against 1.08. On these
  substrates the transferred heads carry useful zero-shot behaviour, unlike bizdev.
- **Exploration is only mildly reduced on closed.** Entropy over iterations 3-40 is 1.58-1.78 against scratch's
  1.96-2.04, a ratio of 0.80-0.89. Advantage std is not lower (0.66-0.72 against 0.63-0.66).
- **Evaluation noise is large.** Single milestones of one run move by up to 10 points (closed, experienced s2: 32.3 →
  19.1 → 29.4).

### Protocol amendment 14, 2026-11-19 22:35 UTC: per-world action heads (v17) and a replication of early transfer (before any result of these runs)
**Code: the sl-exp-v17 head bank.**
- **What it adds.** `model.head_bank` lists worlds that get their own action-selection heads: `hi`, `term`, `query`
  and `slot`, shaped exactly like the shared ones. Every other world uses the shared heads.
- **Routing.** Each decision row takes its world's head set. The collector records the choice and the learner replays it,
  as with the v16 gate.
- **No change elsewhere.** The bank is created last.
  - With an empty bank, v17 is v16: the same tensor names and identical initial values for a fixed seed.
  - With a bank, every other tensor still gets the same initial values.
  - Rows of a bank world send gradients only to that world's heads; all other rows only to the shared heads.
  - All three properties are checked.
- **Effect on transfer.** A transfer source has no bank tensors, so a bank world's heads start fresh, as with
  `init_reset`, while the other worlds keep the transferred heads.
- **Assumption.** The agent knows which world it is in: the world's name, not any hidden parameter. The familiarity gate
  makes the same assumption.

**Disprovable test (pre-registered): H13.**
- **Claim:** the forgetting of negotiation while learning bizdev, and the bizdev failure under negotiation rehearsal
  (H11), both come from the two worlds sharing action-selection heads. Giving bizdev its own heads, while negotiation
  keeps the transferred ones, removes both.
- **Runs:** v17, gate on, P_N_s1 @ 1M, seeds 1-2, `head_bank: ['cam/bizdev']`.
  - `T_B17_fromNbank`: bizdev only, with negotiation held out. Negotiation's heads then get no gradient, so any
    negotiation loss comes from the shared trunk.
  - `T_B17_fromNbankrh`: bizdev 0.9 and negotiation rehearsal 0.1, as in `T_B16_fromNrh`.
- **H13a, where forgetting lives.**
  - **Confirmed if** negotiation validation success at 300k is ≥ 0.40 in both `T_B17_fromNbank` seeds. With shared
    heads it was 0.097 and 0.153, from starts of 0.667 and 0.674.
  - **Disproved if** it is ≤ 0.20 in both seeds. Forgetting would then live in the shared trunk (perception, belief,
    memory, label and skill embeddings, value), which per-world heads cannot protect.
- **H13b, where the rehearsal conflict lives.**
  - **Confirmed if** `T_B17_fromNbankrh` keeps negotiation ≥ 0.50 at 300k in both seeds, and its mean bizdev ΔAUC
    (0-300k, paired by seed against `T_B16_scratch`) is ≥ −0.05.
  - **Disproved if** the mean bizdev ΔAUC is ≤ −0.15 (it was −0.298 with shared heads). The conflict would then live in
    the shared trunk or value head.
- Anything else is mixed.
- **Mechanism check:** `T_B17_fromNbank`'s mean entropy over iterations 3-40 is ≥ 0.9 × the same seed's scratch.
- **What a confirmation would change.**
  - Multi-world lifetimes get per-world heads over a shared trunk, and rehearsal becomes a usable defence against
    forgetting.
  - Melting Pot would be next. There the shared heads carry a zero-shot benefit that fresh per-world heads would give
    up, so the design would also need a rule for when a world inherits heads.

**Disprovable test (pre-registered): H14.** It replicates the exploratory observation above on fresh seeds only.
- **Claim:** with fresh action heads, experience in other CAM worlds speeds early learning of bizdev, and broader
  experience speeds it more.
- **Runs:** frozen v16, exactly the seed 1-2 configurations, on new seeds 3 and 4: `T_B16_scratch`, `T_B16_fromNhr`
  and `T_B16_fromNPhr`.
- **Measure:** early learning AUC, the normalised trapezoid area under bizdev validation score over 0-50k interactions
  (milestones 0, 10k, 25k and 50k). Seeds 1-2 are excluded because they produced the hypothesis.
- **H14a, experience.**
  - **Confirmed if** fromNPhr's early AUC exceeds scratch's in both new seeds, with a mean difference ≥ +0.03.
  - **Disproved if** the mean difference is ≤ 0.
- **H14b, breadth.**
  - **Confirmed if** fromNPhr's early AUC exceeds fromNhr's in both new seeds.
  - **Disproved if** the mean difference is ≤ 0.
- Anything else is mixed.
- **Also reported, not tested:** 0-300k AUC and 50k-300k AUC, to check the late deficit.

**Queue order.** The amendment-14 runs go ahead of the remaining A_NPV ablation transfers and the pending lifetimes and
transfers (P_NPV_s2, P_NPB_s2, the original T_V_* Vending transfers). Their configurations are unchanged, and no running
job is touched.

### Protocol amendment 15, 2026-11-19 22:40 UTC: breadth against the nearest single world, Melting Pot (before any result of these runs)
**Why.** Whether experience across more worlds transfers better has so far been measured only in CAM. There, with the
heads kept, the two-world source transferred worse than the one-world source (amendment 11). With the heads reset, it
led early (H14 tests this). Melting Pot has no single-world arm yet, and its queue is idle. No code change: frozen
sl-exp-mp, the same configuration as P_MP6.

**Disprovable test (pre-registered): H15.**
- **Claim:** at equal total experience (1M agent-steps), experience spread over six Melting Pot substrates transfers to
  the held-out commons_harvest__closed at least as well as experience on its nearest training substrate,
  commons_harvest__open, alone.
- **Runs:**
  - Lifetimes `P_MP1open_s1/s2`: P_MP6's configuration and held-out monitors, but trained on commons_harvest__open
    only.
  - Transfers `T_closed_fromMP1open_s1/s2/s3`: transfer seeds 1-2 start from lifetime s1 @ 1M and seed 3 from lifetime
    s2 @ 1M, exactly as for `T_closed_fromMP6`.
- **Measures:** paired by transfer seed (1-3), fromMP6 − fromMP1open.
  - Primary: learning AUC of closed validation score, 0-1M.
  - Secondary: zero-shot, the 0k validation score in the transfer runs.
- **Confirmed if** the mean AUC difference is ≥ 0 and the mean zero-shot difference is ≥ −1.0.
- **Disproved if** both the mean AUC difference and the mean zero-shot difference are ≤ −1.0. The nearest single world
  would then transfer better than breadth.
- Anything else is mixed.
- **Also reported:** each experienced arm against `T_closed_scratch`, and the zero-shot closed and bos monitors of both
  lifetimes at 0 and 1M.
- **Power.** Single milestones move by up to 10 points, and the closed score falls late in training in both existing
  arms. With n = 3, a mixed result is likely unless the effect is large.

### H13 result, 2026-11-20 00:10 UTC (mechanical verdicts, `scripts/sl_report.py` section 0)
All four v17 runs finished at 300k (23:43 to 00:00 UTC). Negotiation is validation success; bizdev is the
0-300k learning AUC of validation score, paired by seed against `T_B16_scratch`.

| run | heads | negotiation 0k → 300k | bizdev Δ vs scratch | entropy / scratch |
|---|---|---|---|---|
| T_B17_fromNbank s1 / s2 | bizdev own | 0.667 → 0.188 / 0.674 → 0.354 | −0.036 / −0.018 | 0.994 / 0.991 |
| T_B16_fromN s1 / s2 (reference) | shared | → 0.097 / → 0.153 | −0.058 / −0.216 | 0.745 / 0.652 |
| T_B17_fromNbankrh s1 / s2 | bizdev own | → 0.639 / → 0.618 | −0.052 / −0.048 | 0.877 / 0.919 |
| T_B16_fromNrh s1 / s2 (reference) | shared | → 0.562 / → 0.479 | −0.248 / −0.348 | 0.695 / 0.696 |

- **H13a (own heads alone stop forgetting): neither confirmed nor disproved (mixed).** Negotiation ends at 0.188 and
  0.354, between the ≤ 0.20 and ≥ 0.40 bars. The mechanism check passed (entropy ≥ 0.99 of scratch).
  - **Where the remaining forgetting lives.** In `T_B17_fromNbank_s1`, the shared heads that negotiation uses are
    bit-for-bit unchanged after 300k (relative parameter drift from the source 0.000 for `hi.`, `term.`, `query.`,
    `slot.`). Negotiation still fell from 0.67 to 0.10 by 50k. The forgetting without rehearsal therefore lives in the
    shared core whose output those heads read. Relative drift: belief GRU 0.17, memory projections 0.18–0.37,
    action-embedding MLP 0.16, value 0.16.
- **H13b (own heads + 10% rehearsal keep negotiation without costing bizdev): CONFIRMED, exactly at the bar.**
  - Mean bizdev ΔAUC is −0.0499 against the ≥ −0.05 criterion, a margin of 0.0001. Negotiation is kept at 0.639 and
    0.618 (source 0.667).
  - With shared heads, the same rehearsal cost −0.298 on average and kept negotiation at 0.56 and 0.48.
  - Read as: within 0.05 of scratch on the new world while keeping the old one, not "equal to scratch". The two scratch
    seeds differ by 0.068.
- **What this settles.** Per-world action heads with 10% rehearsal are the first configuration here that learns a new
  CAM world at about scratch speed and keeps the old one.
  - H11's failure (rehearsal stalls bizdev at inaction) came from the shared heads.
  - Forgetting without rehearsal comes from the shared core.
  - There is still no positive transfer over the full 0-300k window.

**Exploratory, not pre-registered (from training logs).**
- **Early lead and late deficit.** 0-50k early AUC minus scratch:
  - fromNbank: +0.071 / +0.027;
  - fromNbankrh: +0.028 / −0.016.
  - Every fresh-head transfer so far (fromNhr, fromNPhr, fromNbank; six pairs) trails scratch over 50k-300k, by
    0.004 to 0.12. H14's seeds 3-4 report this window.
- **Raw world-model loss falls faster after a product-containing source.** Mean world-model loss over the first 12
  iterations on bizdev:
  - scratch: 0.35–0.55;
  - sources that include product (fromNP, fromNPhr, fromNPV at 250k, 500k or 1M): 0.03–0.25;
  - negotiation-only sources: 0.37–0.56, and 2.5–4.9 at iteration 1 against scratch's 1.2.
  - The variance-normalised check below shows this is calibration, not knowledge of bizdev dynamics.

### Diagnostics of the transferred core, 2026-11-20 00:30 UTC (evidence only, no runs launched)
All three checks run on identical bizdev validation transitions: two segments of 16 worlds × 32 steps, collected by a
fresh-init policy, evaluated by every checkpoint with the frozen v17 code. They are read-only.

**1. Does the world model transfer knowledge of bizdev dynamics? No.**

| checkpoint | next-embedding R² | R² of "next = current" | reward MSE | termination BCE |
|---|---|---|---|---|
| fresh init | −43.8 | +0.70 | 0.040 | 0.66 |
| P_N_s1 @ 1M | −15.6 | +0.62 | 0.073 | 10.9 |
| P_NP_s1 @ 250k | +0.15 | +0.88 | 0.004 | 0.08 |
| P_NP_s1 @ 1M | +0.66 | +0.91 | 0.016 | 0.21 |
| P_NPV_s1 @ 1M | −2.1 | +0.62 | 0.209 | 7.4 |
| T_B16_scratch_s1 @ 300k (trained on bizdev) | +0.95 | +0.80 | 0.002 | 0.014 |

- The product-containing source predicts the next state worse than simply copying the current state. Its lower early
  loss comes from calibrated reward and termination predictions: near-zero reward, no termination.
- The negotiation source and the full NPV source predict termination confidently and wrongly (BCE 7–11).
- No source carries a bizdev dynamics model a planner could exploit. That argues against expecting a policy that uses
  the world model to turn transferred experience into speed. The world model stays auxiliary.

**2. Is the late deficit a loss of plasticity? No sign of it.**
- On bizdev data, the source cores' context has the same or higher effective rank than a fresh network:
  - entropy rank 36–41 vs fresh 33–37;
  - 99%-energy rank 213–261 vs 233–248.
- Dormant context units: 0.1–1.0% (fresh 0.3–0.6%).
- Core weight norms are within 1% of fresh init.
- A bizdev-trained network has a much lower rank (18), so transferred cores are not collapsed relative to what bizdev
  needs.
- Shrink-and-perturb, the standard fix for warm-start plasticity loss, therefore has no supporting evidence here and is
  not tested.

**3. Do the learned skills change behaviour? Barely.**
- In P_N_s1 (negotiation) and T_B16_scratch_s1 (bizdev), the high-level policy uses skills unevenly: entropy 1.5–1.7
  of 2.08, switch rate 0.02–0.12 per step.
- The first action slot is near-deterministic (max probability 0.98–0.99; logit spread across candidates about 20).
  Changing the skill moves those logits by about 0.24, so the mean pairwise Jensen–Shannon divergence between skills'
  first-slot distributions is 0.0000–0.0001.
- The skill hierarchy is close to inert in the decision that dominates behaviour. Any "skill transfer" would need a
  different design. The `noskill` ablation transfer (queued) measures what removing skills costs.

**Statistical caveat on the late deficit.** The six fresh-head pairs reuse the same two scratch runs (seeds 1-2) as
the reference, so they are effectively n = 2. H14's seeds 3-4 bring new scratch runs.

### Protocol amendment 16, 2026-11-20 00:35 UTC: replicate H13b and test breadth under per-world heads (before any result of these runs)
**Why.**
- H13b (per-world heads + 10% rehearsal) is the only configuration so far that keeps the source world and learns the new
  one within 0.05 of scratch. It passed by 0.0001 on two seeds, so it needs new seeds before it becomes the default
  transfer protocol.
- Breadth so far was tested only with heads that carry the source policy (two-world source worse) or are deleted
  (two-world source better early: H14b reference seeds +0.03, and 0-300k −0.012 vs −0.044).
- The diagnostics above found one thing the product-containing source carries to bizdev that the negotiation source
  does not: calibrated reward and termination predictions.
- No code change: frozen sl-exp-v17.

**Runs (seeds 1-4 share scratch references `T_B16_scratch_s1-s4`, v16 = v17 with an empty bank).**
- `T_B17_fromNbankrh_s3/s4`: exactly `T_B17_fromNbankrh_s1`'s configuration with new seeds.
- `T_B17_fromNPbankrh_s1-s4`:
  - source `P_NP_s1` @ 1M;
  - families bizdev 0.9, negotiation 0.05, product 0.05;
  - head bank `['cam/bizdev']` (negotiation and product keep the transferred shared heads);
  - nothing held out.
  - Bizdev gets the same 90% of interactions as in fromNbankrh. Rehearsal is split between the two source worlds.

**Disprovable tests (pre-registered).**
- **H13c (replication of H13b on seeds 3-4).**
  - **Replicates if** fromNbankrh keeps negotiation ≥ 0.50 at 300k in both new seeds and its mean bizdev ΔAUC (0-300k,
    paired against `T_B16_scratch_s3/s4`) is ≥ −0.05.
  - **Fails to replicate if** the mean ΔAUC is ≤ −0.15, or negotiation is < 0.50 at 300k in both new seeds.
  - Anything else is mixed.
- **H16 (breadth with per-world heads).**
  - **Claim:** with the new world on its own heads and rehearsal of the source worlds, a two-world source
    (negotiation + product) makes bizdev learning at least as fast as a negotiation-only source.
  - **Measure:** paired by seed (1-4), bizdev learning AUC 0-300k of fromNPbankrh − fromNbankrh.
  - **Confirmed if** the mean difference is ≥ +0.02 and positive in ≥ 3 of 4 seeds.
  - **Disproved if** the mean difference is ≤ 0.
  - Anything else is mixed.
  - **Also reported, not tested:**
    - both arms against scratch (n = 4), which is "does experience help?" under the best architecture so far;
    - early (0-50k) and late (50k-300k) windows;
    - negotiation retention of fromNPbankrh (it gets half the negotiation rehearsal).

**Queue order.** After H14's last run (`T_B16_fromNhr_s4`):
1. `T_B17_fromNbankrh_s3/s4`;
2. `T_B17_fromNPbankrh_s1-s4`;
3. the queued A_NPV ablation transfers (noskill, nowm, rl2);
4. then P_NPV_s2, P_NPB_s2 and the remaining jobs, unchanged.
Corrected 00:19 UTC, before any of these jobs started: the ablation transfers were first placed ahead of fromNPbankrh.
Each of their pre-start smoke tests takes about 37 min (Vending monitors) and blocks the queue loop, so slots would sit
idle. Only the order changed, not any configuration. No running job is touched.

### H14 result, 2026-11-20 01:45 UTC (mechanical verdicts, `scripts/sl_report.py` section 0)
All six v16 runs finished at 300k (00:59 to 01:38 UTC). Bizdev validation score, normalised AUC per window; all arms
have fresh action heads (`init_reset` of `query.`, `slot.`, `hi.`, `term.`) and the same familiarity gate.

| seed | scratch 0-50k / 50-300k / 0-300k | fromNhr 0-50k / 50-300k / 0-300k | fromNPhr 0-50k / 50-300k / 0-300k |
|---|---|---|---|
| 3 | 0.277 / 0.592 / 0.540 | 0.302 / 0.585 / 0.538 | 0.326 / 0.557 / 0.518 |
| 4 | 0.276 / 0.596 / 0.542 | 0.342 / 0.558 / 0.522 | 0.328 / 0.583 / 0.541 |

- **H14a (experience speeds early learning): CONFIRMED.** fromNPhr − scratch over 0-50k is +0.049 and +0.052 on the
  new seeds (mean +0.050; bar ≥ +0.03, positive in both). Seeds 1-2, which suggested the hypothesis, gave +0.073 and
  +0.074. Concretely: the experienced agent reaches at 25k the validation score scratch reaches at 50k (0.34-0.37).
- **H14b (broader experience speeds it more): neither confirmed nor disproved (mixed).** fromNPhr − fromNhr over 0-50k
  is +0.025 and −0.015 (mean +0.005). On the new seeds the two-world source is not reliably better early than the
  negotiation-only source.

**The late deficit replicates against independent scratch seeds** (exploratory window, reported by the pre-registered
test but not itself a criterion). 50k-300k minus same-seed scratch:
- fromNhr: −0.003, −0.120, −0.007, −0.038 (seeds 1-4);
- fromNPhr: −0.026, −0.032, −0.035, −0.013.
- All eight fresh-head pairs are negative, including the four against the new scratch runs, so the late deficit was not
  an artifact of reusing two scratch runs. It is small (fromNPhr mean −0.027) and scratch overtakes by about 100k.
- Net over 0-300k: fromNPhr −0.010, −0.014, −0.021, −0.002 (mean −0.012); fromNhr +0.004, −0.092, −0.002, −0.021
  (mean −0.028).

**What this settles for "does experience help?"** On bizdev, with fresh heads, experience in other CAM worlds is a
reliable head start worth about 25k interactions in the first 50k, and no gain over a 300k budget: scratch catches up
by about 100k and ends slightly ahead. Breadth (two source worlds instead of one) adds nothing measurable early.

**Exploratory, from training logs (seeds 1-4, transferred arm / scratch).** In 50k-300k the transferred runs' PPO
updates move the policy further than scratch's at a similar entropy: per-update KL 1.16-1.49× and clip fraction
1.22-1.48×, entropy 0.92-1.02×, value loss 0.82-1.11×. Larger steps that are clipped more often are one candidate
for the late deficit; the offline test on identical data is in amendment 17.

**Memory ablation transfers (n = 1 each, seed 1; not a pre-registered test).** Bizdev 0-300k AUC from the N+P+V
ablation lifetimes, against `T_B11_scratch_s1` (0.515):
- full memory (`T_B11_fromNPV_s1`): 0.541 (+0.026);
- no cross-episode memory (`fromNPVepisode`): 0.530 (+0.015), the highest score at 300k (0.806);
- no external memory, GRU only (`fromNPVgru`): 0.312 (−0.203).
- The GRU run never found the reward-producing behaviour the others found at about 150k (its value loss stayed at
  0.001-0.004 while the episode run's rose to 0.14-0.19). It peaked at 0.507 at 200k and drifted back to 0.113 at
  300k; its final bizdev validation return (−0.806) equals the do-nothing policy's (−0.800).
- One seed per arm cannot separate the memory variants from seed noise; the GRU run's failure is one run.

### H13c result, 2026-11-20 02:45 UTC (mechanical verdict, `scripts/sl_report.py` section 0)
`T_B17_fromNbankrh_s3/s4` (per-world heads + 10% negotiation rehearsal, new seeds) finished at 300k.

| seed | negotiation at 300k | bizdev AUC 0-300k (Δ vs same-seed scratch) | 0-50k / 50k-300k (scratch) |
|---|---|---|---|
| 3 | 0.604 | 0.512 (−0.028) | 0.287 / 0.557 (0.277 / 0.592) |
| 4 | 0.000 | 0.522 (−0.020) | 0.318 / 0.563 (0.276 / 0.596) |

- **H13c (replication of H13b): neither (mixed).** Mean bizdev ΔAUC −0.024 meets the ≥ −0.05 bar, but negotiation is
  kept in only one of the two new seeds; seed 4 collapsed to 0.000 (all-ASK, amendment 17).
- **Across all four seeds** of per-world heads + rehearsal: bizdev ΔAUC −0.052, −0.048, −0.028, −0.020 (mean −0.037);
  negotiation kept (0.60-0.64) in three, lost in one. The configuration is the best found for learning a new world
  while keeping the old one, but rehearsal by on-policy RL alone is not reliable; H17 tests the anchor.
- The early-lead / late-deficit pattern again: 0-50k above scratch in three of four seeds (seed 2: 0.263 vs 0.279),
  50k-300k below it in all four (−0.033 to −0.069).

### Protocol amendment 17, 2026-11-20 02:30 UTC: rehearsal collapse and the transferred core's step size (before any result of these runs)
**Evidence 1: on-policy rehearsal can collapse.** `T_B17_fromNbankrh_s4` (H13c replication, per-world heads + 10%
negotiation rehearsal) kept negotiation at 0.646 at 25k, fell to 0.27 at 50k and to exactly 0.000 at 100k and 200k.
Its sibling seeds kept 0.62-0.65. A read-only profile on 96 negotiation validation trials (3,072 sampled decisions,
dev worktree code, weights copied; `neg_actions.py`, `anc_grad.py`, `anc_scale.py` in the session scratchpad):

| checkpoint | success | ASK / PROPOSE / ACCEPT | invalid | decisions per episode | policy entropy per decision |
|---|---|---|---|---|---|
| source `P_N_s1` @ 1M | 0.76 | 0.83 / 0.10 / 0.08 | 0.00 | 10.4 | 1.27 |
| `T_B17_fromNbankrh_s1` final | 0.70 | 0.83 / 0.10 / 0.07 | 0.00 | 10.4 | 1.07 |
| `T_B17_fromNbankrh_s3` @ 237k | 0.67 | 0.18 / 0.11 / 0.71 | 0.63 | 9.5 | 1.03 |
| `T_B17_fromNbankrh_s4` @ 242k | 0.00 | 1.00 / 0.00 / 0.00 | 0.00 | 16.0 | 1.35 |

- **Seed 4 asks on every decision.** It never proposes or accepts, so every negotiation fails with zero progress.
- **On-policy rehearsal cannot get out of that state.** A failure costs the whole unused decision budget however it
  ends (by design, so that stalling is never cheaper), so all of seed 4's negotiation episodes have the same return.
  On mixed training segments its mean |advantage| on negotiation decisions is 0.033, against 0.246 in seed 1 and
  0.23-0.31 on bizdev decisions.
- **The entropy bonus does not see it.** The deterministic choice is the action type; the entropy sits in the
  argument slots (1.35 per decision, above the source's 1.27).
- **Distance from the source policy does not separate healthy from collapsed runs.** KL from the source policy
  (k3 estimate on each run's own decisions): seed 1 1.16, seed 3 6.5 (an ACCEPT-heavy strategy with 63% invalid
  actions), seed 4 2.3.
- An anchor toward the source still acts on the collapsed policy: with π(ASK) = 1 and π_src(ASK) = 0.83, the
  forward-KL gradient lowers ASK and raises the source's PROPOSE/ACCEPT, whatever the returns.

**Evidence 2: transferred cores take larger policy steps.** One production PPO update (lr 3e-4, 2 epochs, 32-stream
minibatches, the run's own Adam state) on identical bizdev data: 64 worlds × 32 steps collected by the same-seed
`T_B16_scratch` policy at 300k, each model's own log-probabilities (`sens.py`). True KL(old ‖ new) over all streams:

| seed | scratch | fromNhr (Δ own advantages / common advantages) | fromNPhr (own / common) | clip fraction scratch / fromNhr / fromNPhr |
|---|---|---|---|---|
| 3 | 0.0074 | 0.0229 (3.1×) / 0.0174 (2.4×) | 0.0235 (3.2×) / 0.0198 (2.7×) | 0.02 / 0.08 / 0.06 |
| 4 | 0.0088 | 0.0220 (2.5×) / 0.0130 (1.5×) | 0.0184 (2.1×) / 0.0129 (1.5×) | 0.04 / 0.16 / 0.05 |

- "Common advantages" gives every model the scratch model's value estimates, so the advantage signal is identical.
  The transferred policies still move 1.5-2.7× further: a property of the network, not of the signal.
- The policy gradient reaching the shared core is 1.3-1.9× scratch's (0.43-0.89 vs 0.33-0.59). Adam's step size is
  about lr per parameter whatever the gradient scale, so a more sensitive core turns the same step into a larger
  policy change.
- At initialisation it is different: transferred models' world-model and value losses make their total gradient
  2.6-7.4× scratch's (9-26 vs 3.4-3.6), gradient clipping at 1.0 shrinks every step, and the one-update KL is not
  larger (0.003-0.026 vs 0.004-0.020). The larger steps appear later in training, where the deficit is.
- Training logs agree in direction (per-update KL 1.16-1.49×, clip fraction 1.22-1.48× in 50k-300k), with a smaller
  ratio because the logged KL is measured on each minibatch before its own step.

**v18 (tag sl-exp-v18, worktree `agentrs-sl-exp-v18`) = v17 + two opt-in learner options, both off by default.**
- `learner.anchor` (β) with `anchor_from` (default `init_from`): β × KL penalty toward a frozen copy of the source
  policy on the decisions made with the shared heads (the rehearsed source worlds when the new world has its own
  heads; requires `model.head_bank`). k3 estimator on the sampled decisions, d = log π_src − log π clamped to ±10. The
  source is replayed from its own recurrent state, carried across segments (a trial start clears it, as at
  collection; after a restart it starts empty for one segment). Tested: with β = 0 updates are bit-identical to v17;
  an anchor equal to the policy gives KL < 1e-6 across segments (5e-5 without the carried state).
- `learner.lr_scale` / `lr_scale_rest`: learning-rate multipliers per parameter-name prefix (default: one Adam group
  at lr, as before).
- β = 0.05: at β = 1 the anchor's gradient norm is 10-15× the policy gradient's (5.4 vs 0.36 in seed 1, 3.0 vs 0.26
  in seed 4); 0.05 makes it 0.6-1× the policy gradient, so it can pull without dominating bizdev's updates.

**Disprovable tests (pre-registered).**
- **H17 (anchor).** Claim: on-policy rehearsal cannot recover a degenerate source-world policy because its returns
  carry no signal; a KL anchor to the frozen source policy supplies one, restores a collapsed policy and prevents
  the collapse.
  - **H17a (rescue).** `T_B18_rescue_{anc,ctl}_s1/s2` continue `T_B17_fromNbankrh_s4`'s final weights for 100k
    interactions (bizdev 0.9 / negotiation 0.1, same heads), with β = 0.05 (anc) or 0 (ctl, same v18 code).
    - Valid only if `T_B17_fromNbankrh_s4` ends with negotiation ≤ 0.20 at 300k; otherwise "not applicable".
    - **Confirmed if** the anchor arm reaches negotiation ≥ 0.40 at 100k in both seeds and the control stays ≤ 0.20 in
      both. **Disproved if** the anchor arm is ≤ 0.20 in both seeds. Anything else is mixed.
  - **H17b (prevention).** `T_B18_fromNbankanc_s1-s4` = `T_B17_fromNbankrh` with β = 0.05.
    - **Confirmed if** negotiation ≥ 0.50 at 300k in all four seeds and the mean bizdev ΔAUC (0-300k, paired against
      `T_B16_scratch`) ≥ −0.05. **Disproved if** negotiation ≤ 0.20 at 300k in any seed or the mean ΔAUC ≤ −0.15.
      Anything else is mixed.
    - Caveat: with one collapse in four unanchored seeds, four anchored seeds without a collapse are weak evidence
      by themselves (probability 0.32 under no effect); H17a is the sharp test of the mechanism.
- **H18 (core step size).** Claim: the late (50k-300k) deficit of fresh-head transfers comes from oversized
  per-update policy changes of the transferred core at scratch's learning rate; halving the learning rate of
  everything except the fresh action heads (`query.`, `slot.`, `hi.`, `term.`) removes it.
  - `T_B18_fromNPhrcore_s1-s4` = `T_B16_fromNPhr` with `lr_scale_rest` 0.5, against `T_B16_scratch_s1-s4` (paired by
    seed; late AUC = bizdev validation score over 50k-300k).
  - Mechanism check: mean logged per-update KL over 50k-300k, fromNPhrcore / same-seed scratch ≤ 1.1 in every seed.
  - **Confirmed if** the mean late Δ ≥ −0.005 and the check passes. **Disproved if** the mean late Δ ≤ −0.027 (no
    better than `T_B16_fromNPhr`'s −0.027) while the check passes. Anything else is mixed.
  - Control, reported: `T_B18_scratchcore_s1-s4` (scratch with the same learning-rate scale) shows whether a smaller
    core step helps any learner, in which case a confirmed H18 would not be specific to transfer.
- Verdicts are computed by `scripts/sl_report.py` (`_anchor_test`, `_core_lr_test`), written before any of these
  runs started.

**Queue (sl-ops-v23).** H17's eight runs go right after `T_B17_fromNPbankrh_s4`, ahead of the remaining A_NPV
ablation transfers: their pre-start smoke tests take about 5 minutes instead of 37, so they fill free slots while
the long smokes run later. H18's eight runs go after `P_NPV_s2` and `P_NPB_s2`, so the second lifetimes start first.
No running job is touched; new launches were held with `runs/sl/QUEUE_STOP` from 02:15 UTC until the restart so that
no smoke test was in flight.

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
