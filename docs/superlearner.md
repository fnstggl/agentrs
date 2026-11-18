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

## Phase 2 (Melting Pot), `configs/sl/queue_phase2_mp.yaml`
**Lifetime** (`P_MP6_s1`, `P_MP6_s2`):
- Self-play with parameter sharing (every player is a learner stream with its own belief and memory) on 6
  substrates: coins, collaborative_cooking__asymmetric, commons_harvest__open, clean_up,
  prisoners_dilemma_in_the_matrix__repeated, stag_hunt_in_the_matrix__repeated.
- 4M learner agent-steps; trials of 2 episodes.
- Zero-shot monitors: the held-out substrates commons_harvest__closed and bach_or_stravinsky_in_the_matrix__repeated,
  and the held-out scenarios prisoners_dilemma_in_the_matrix__repeated_0, coins_0 and clean_up_2 (the benchmark's
  background bots as unfamiliar co-players).

**Transfer:** 1M agent-steps on each held-out substrate, scratch vs `life_4000000`, seeds 1–2.

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
