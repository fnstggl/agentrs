# agentrs — an action-learning agent research platform

A model-based RL system that learns to act in (web) environments from **observation**
(demonstrations), **interaction** (real, independently verified outcomes), and **prediction of
action consequences** (a learned, action-conditioned world model used for planning and, optionally,
imagined training). All learning is gradient-based updates of real neural network weights.

The platform exists to test one hypothesis rigorously:

> an agent with an explicitly learned action-conditioned world model, observational learning,
> planning and continual RL may complete unfamiliar tasks better than an equivalent agent trained
> with imitation or model-free RL alone.

It does **not** assume the hypothesis is true. See [`docs/experiments.md`](docs/experiments.md) for
the measured results — including the negative ones.

* Architecture and design rationale: [`docs/architecture.md`](docs/architecture.md)
* Parameterized semantic action learning (content as part of the action): [`docs/content_learning.md`](docs/content_learning.md)
* Experiments, results, ablations, compute: [`docs/experiments.md`](docs/experiments.md)
* Phase-by-phase status (implemented / executed / tested / unverified): [`docs/status.md`](docs/status.md)
* Hardware profiles and resource estimates: [`docs/hardware.md`](docs/hardware.md)

## What is in the box

| component | where | status |
|---|---|---|
| Environment interface (+ allowlisted, budgeted guard) | `environments/base.py`, `security/` | tested |
| MiniWeb: 14 simulated web-app task families with server-side verifiers | `environments/miniweb/` | tested |
| Real-browser env (same apps as HTML, Playwright/Chromium) + generic page env | `environments/web/` | tested (25 browser tests) |
| BrowserGym / MiniWoB++ adapter | `environments/browsergym_adapter.py` | tested |
| Typed action grammar, candidate generation | `actions/` | tested |
| Belief state (observed facts vs inferred state) | `agent/belief.py` | tested |
| Experience schema, Parquet store, replay, leakage guards | `data/` | tested |
| LM backbone (HF causal LMs, LoRA, prefix-shared scoring) | `models/backbone.py` | tested |
| Policy, world model, critic, inverse dynamics | `models/` | tested |
| BC, world-model, critic, RL (REINFORCE / PPO / GRPO), imagination, IDM trainers | `training/` | tested |
| MPC planner with uncertainty penalty | `planning/mpc.py` | tested |
| Continual learning: actor / learner / evaluator processes, registry, promotion, rollback, curriculum | `training/continual.py`, `checkpoints/` | tested |
| Episodic memory, skills (options), skill discovery | `memory/` | tested |
| Consent-gated human demonstration recorders (terminal and browser) | `agent/runtime.py`, `demonstrations/` | tested |
| Evaluation, baselines A–D, ablations, reports, plots | `evaluation/`, `training/pipeline.py`, `scripts/` | executed |
| Semantic content actions: content-generating policy, content-conditioned world model, content planner, content RL, verifiable content environment | `content/`, `environments/content/` | tested (15 tests); A–E executed |

## Quick start (CPU)

```bash
bash scripts/setup.sh
```

```bash
source .venv/bin/activate
```

```bash
python -m pytest -q
```

The full suite (119 tests: unit, learning, safety, integration, browser) takes a few minutes on 4
CPU cores. Browser tests skip with a reason if Chromium cannot be launched; set
`AGENTRS_CHROMIUM=/path/to/chrome` or run `playwright install chromium`. Learning tests use a tiny
randomly initialised model and need no downloads; experiment configs download Pythia from the
Hugging Face hub on first use.

## Command-line workflows

All commands are `python -m agentrs <command>` (or `agentrs <command>` after installation).
`--model` accepts a preset (`smoke`, `dev` = Pythia-14M, `dev31m`, `research` = Qwen3-4B/0.6B on
GPU) or a YAML path. Every command below is executed (at smoke scale) by
`bash scripts/verify_cli_workflows.sh`.

**1. Set up and validate the environment**

```bash
python -m agentrs setup-check
```

**2. Run an agent episode** (actors: `teacher`, `random`, `policy`, `planner`, `human`)

```bash
python -m agentrs run-episode --family form_validation --actor teacher -v
```

**3. Record a demonstration** (explicit consent required; stored in the private partition by default)

```bash
python -m agentrs record-demo --family fill_field --consent --demonstrator-id alice --out runs/demos
```

In a real browser (needs a display): add `--env localweb`.

**4. Generate trajectory datasets**

```bash
python -m agentrs gen-data --out runs/data/train --split train --n-per-family 20
```

```bash
python -m agentrs gen-data --out runs/data/train --split train --n-per-family 10 --actor random
```

```bash
python -m agentrs gen-data --out runs/data/val --split val --n-per-family 4
```

**5. Train the world model**

```bash
python -m agentrs train-wm --data runs/data/train --val-data runs/data/val --model dev --steps 600 --out runs/ckpt/wm
```

**6. Evaluate world-model predictions** (vs marginal / action-type / persistence baselines; calibration; drift)

```bash
python -m agentrs eval-wm --world-model runs/ckpt/wm --train-data runs/data/train --val-data runs/data/val --out runs/wm_eval.json
```

**7. Behaviour-clone the policy**

```bash
python -m agentrs train-bc --data runs/data/train --val-data runs/data/val --model dev --steps 600 --out runs/ckpt/policy_bc
```

**8. Train a critic**

```bash
python -m agentrs train-critic --data runs/data/train --val-data runs/data/val --model dev --steps 300 --out runs/ckpt/critic
```

**9. Real-environment policy RL** (`--algo reinforce|ppo|grpo`)

```bash
python -m agentrs train-rl --policy runs/ckpt/policy_bc --critic runs/ckpt/critic --algo ppo --iterations 10 --episodes-per-iter 32 --out runs/rl
```

**10. Run the agent with model-based planning**

```bash
python -m agentrs run-episode --family login --actor planner --policy runs/rl/policy --world-model runs/ckpt/wm --critic runs/rl/critic -v
```

**11. Run continual-learning iterations** (separate actor/evaluator processes, gated promotion)

```bash
python -m agentrs continual --config configs/continual/dev.yaml
```

```bash
python -m agentrs rollback --registry runs/continual_dev/registry.sqlite --component policy
```

**12. Run all required baselines** (A policy-only, B imitation, C model-free RL, D model-based)

```bash
python -m agentrs baselines --config configs/experiments/v2_base.yaml --variants A,B,C,D --seeds 0,1 --prefix v2
```

Or in parallel, one process per variant and seed: `SEEDS="0 1" bash scripts/run_v2_suite.sh`.

**13. Held-out evaluation** (instances/layouts/templates/vocabulary: `test`; unseen families: `test_family`)

```bash
python -m agentrs evaluate --policy runs/rl/policy --sets val,test,test_family --n-per-family 4
```

```bash
python -m agentrs evaluate --actor planner --policy runs/rl/policy --world-model runs/ckpt/wm --critic runs/rl/critic --sets val,test,test_family
```

**14. Resume training from a checkpoint** (weights, optimizer, scheduler and RNG are restored)

```bash
python -m agentrs train-bc --data runs/data/train --val-data runs/data/val --steps 900 --out runs/ckpt/policy_bc --resume runs/ckpt/policy_bc
```

```bash
python -m agentrs train-rl --policy runs/ckpt/policy_bc --critic runs/ckpt/critic --iterations 20 --out runs/rl --resume runs/rl
```

**15. Compare two checkpoints** (paired, with bootstrap CI of the difference)

```bash
python -m agentrs compare --a runs/ckpt/policy_bc --b runs/rl/policy
```

**16. Visualise prediction errors and trajectories**

```bash
python -m agentrs visualize --run runs/v2/v2_D_s0
```

```bash
python -m agentrs visualize --store runs/rl/experience --world-model runs/ckpt/wm
```

**Content actions** (generate content as part of the action, predict its effect, learn from verified outcomes):

```bash
python -m agentrs content-episode --family content_code --split test --index 4
```

```bash
python -m agentrs content-experiment configs/content/base.yaml --set variant=E name=content_E_s0
```

Experiment reports and evaluation-only planner ablations:

```bash
python -m agentrs report --runs runs/v2/v2_A_s0 runs/v2/v2_B_s0 runs/v2/v2_C_s0 runs/v2/v2_D_s0 --out runs/v2/report.md
```

```bash
python scripts/planner_sweep.py runs/v2/v2_D_s0 --grid horizon=1,2 uncertainty_penalty=0,1,3
```

## Results (CPU development profile, Pythia-14M everywhere)

Measured on 4 CPU cores, 2 seeds, 32–40 verified evaluation episodes per split
(full tables, intervals, compute and ablations in [`docs/experiments.md`](docs/experiments.md);
raw files in `results/`):

| configuration | val | held-out instances (test) | unseen families |
|---|---|---|---|
| A pretrained policy, no updates | 0.025 | 0.025 | 0.031 |
| B behaviour cloning | 0.325 | 0.250 | 0.000 |
| C B + PPO from verified rewards (800 episodes) | 0.288 | 0.288 | 0.000 |
| D C + learned world model + MPC planning | 0.313 | 0.275 | 0.000 |
| D + imagination-based training (planner) | 0.300 | 0.250 | 0.016 |

* Imitation is what works; RL, planning and imagination produced **no measurable gain** at this
  scale, and planning cost ≈9× more inference compute per success. Nothing transferred to unseen
  task families. The hypothesis is **not supported** by these measurements — which are far too
  small (tiny models, ±12–15-point intervals) to refute it.
* The world model genuinely learns action consequences (held-out event AUROC 0.99 vs 0.81 for a
  frequency baseline) and partially transfers to unseen families (reward AUROC 0.65–0.84 vs 0.50)
  but is overconfident there; a 31M world model transfers better than a 14M one.
* A trained policy behaves identically in real Chromium and in the simulator (20/20 episodes).

No claim of superiority over frontier models is made or implied.

## Safety defaults

Learning and exploration happen only in sandboxed simulators or allowlisted local apps. Every
action passes an immutable permission policy (action types, URL allowlist, budgets, timeouts);
page content is untrusted and cannot alter permissions; consequential actions require an approval
callback that denies by default; task verification is computed environment-side; human
demonstrations require recorded consent and default to a private, non-exportable partition.
Nothing is uploaded anywhere.

## Repository layout

```
configs/        model profiles, experiment and continual-learning configs (typed YAML)
src/agentrs/    agent/ actions/ environments/ data/ models/ planning/ training/
                memory/ evaluation/ checkpoints/ monitoring/ security/ demonstrations/
scripts/        setup, experiment launchers, planner sweeps, CLI verification
tests/          unit/ learning/ safety/ integration/ (incl. browser)
docs/           architecture, experiments, status, hardware
```
