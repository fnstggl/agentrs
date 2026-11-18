# CAM-v2: contract-based objectives, off-policy replay learning and imagination

Branch `feat/cam-v2` (worktree `/workspace/agentrs-cam-v2`). It builds on `feat/cam` (the original CAM,
PPO) and `feat/cam-product-fix` (the first product reward and curriculum fix). Every original run,
checkpoint and GitHub backup is untouched. PPO remains the benchmark: the `runs/cam`, `runs/cam_nowm`
and `runs/cam_pfix` runs.

## Why: diagnosis of the product failure

The original CAM learned to go bankrupt within weeks in 100% of product episodes. Three things combined:

- With Φ(terminal) = 0, every failure had the same shaped return.
- The per-action step cost therefore rewarded ending early.
- Exploration almost never reached the full target (0.2% for random play).

`python -m agentrs.cam.audit --mode legacy` now detects this automatically. On the original reward,
terminating early and wasting resources both score better than preserving resources (product and
negotiation).

## Components (`src/agentrs/cam/`)

| Area | What it does |
|---|---|
| Task contract (`envs/base.py`, per-environment `contract()` / `task_status()`) | Environment-independent and externally verified: clock unit and deadline, verified resources (fractions), hard constraints whose violation is an unacceptable terminal state, milestones (never counted as success), failure causes, and hindsight goals of the same schema. |
| Reward contract (`contract.py`) | Semi-Markov: discount γ^dt per action, using the clock units the action consumed. The time cost accrues over the action's duration, and a failure pays the rest of the horizon, so ending early is never cheaper. Potential shaping uses the same discount. Success comes only from the environment's verifier. A separate constraint-cost channel feeds Lagrange multipliers. |
| Audit (`audit.py`) | Adversarial scripted policies: early termination, waste, preserve, milestone, oracle success. These are fixtures, never shown to CAM. It checks that success scores best, early termination never beats preserving or milestones, waste never beats preserving, and constraint violation is strictly worse when λ > 0. It runs at the start of every v2 run, and the run refuses to train if the audit fails. |
| CAM2 model (`v2/model.py`) | An RSSM world model: a deterministic GRU state plus 16×16 categorical latents, a prior for imagination and a posterior for real data. Heads predict features, status, contract reward, continuation, action duration, events, violation, success and eventual outcome (viability), plus a next-embedding ensemble that measures uncertainty. The actor has its own embedding, options and a grammar-masked decoder. Critics: twin Q, an expectile V and V_img, with Polyak targets. There are three optimizers, and actor and critics read stop-gradient world-model features. |
| Replay (`v2/data.py`) | Complete real episodes with provenance (policy and environment version, archive start, reward components). Shards persist under `<run>/replay`. A protected reservoir holds successes and advanced attempts. Hindsight goals are sampling-time views only, and imagined data is never stored. |
| Off-policy learners (`v2/learner.py`) | IQL critics with semi-Markov n-step targets, cut where the current policy's probability of a recorded decision drops below 0.3× the behaviour probability. `awr`: advantage-weighted regression on the full structured decision. `sac`: discrete SAC with exact masked enumeration (product 33 actions, bizdev 231; negotiation falls back to AWR) and auto-tuned temperature. |
| Exploration | Go-Explore archive of promising viable states. Restored states get fresh noise, and their recorded history is replayed as belief burn-in that is excluded from learning. Reverse curriculum on start times. Training-only product-goal curriculum. Rare-success reservoir. |
| Dream learner (`v2/dream.py`) | Imagination from real replayed posterior states with CAM's own actor. Score-function actor gradients on imagined semi-Markov λ-returns. The imagination critic is also anchored to real n-step targets. `fixed` mode uses horizon H. `adaptive` mode stops each trajectory at the first step whose ensemble disagreement exceeds a threshold calibrated on held-out real data, and scales the actor's imagination loss by a reliability score from real open-loop errors (0 until calibrated). If imagination-trained policies lose more than 0.1 validation score, its weight is halved. |
| World-model validation (`v2/wm_eval.py`) | Open-loop errors by horizon (1/2/4/8), violation calibration by horizon, viability calibration, disagreement-vs-error correlation, errors on rare actions and terminal outcomes. Counterfactual action ranking against real simulator outcomes from identical snapshots, with the simulator oracle labelled. A corrupted-model control. |

## Experiments (`configs/cam2/queue.yaml`, run by `scripts/cam2_queue.py` as `agentrs_cam2_queue`)

All arms share the same config, the same budget of 600k real interactions and the same evaluation
protocol. Validation and test tasks are bit-identical to the PPO runs (verified by hash). Arms:

- `awr`: off-policy, no dreaming.
- `dreamH1`, `dreamH4`: fixed imagination horizons.
- `dreamD`: adaptive, up to 8 steps.
- `sac`.
- Two seeds each.
- Ablations of `awr` (no hindsight, no archive, no Lagrangian).
- bizdev from scratch vs initialised from `awr`'s best checkpoint, with negotiation and product
  monitored for forgetting.

Jobs start only as GPU capacity frees: at most 3 CAM GPU processes while the Qwen run is active, 6
afterwards. Each job has its own GitHub checkpoint watcher. Results are compiled by
`scripts/cam2_report.py` into `runs/cam2/report.md`. To restore a checkpoint from GitHub:
`scripts/verify_cam2_restore.py --run <run dir>`.

## Not done yet / limitations

- **Learned meta-controller:** there is none for ACT / REPLAY / IMAGINE / INVESTIGATE. The current rule
  is fixed real collection per iteration, fixed replay updates, imagination scaled by measured
  reliability, and archive investigation weighting set to 0.
- **Hierarchical (milestone-level) strategic imagination:** not implemented.
- **Imagination horizons:** H = 2 and H = 8 are not separate fixed arms; adaptive mode covers up to 8.
- **SAC on negotiation:** not run, because its action space (about 88k actions) can't be enumerated.
  The SAC arm uses AWR for negotiation.
- **Viability head:** learned and evaluated, but not used for action selection beyond the
  Lagrangian-penalised critics.
