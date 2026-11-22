# CAM Superlearner 4.0

A reinforcement-learning research system built on the Crucible 3.0 world simulator (`../crucible`). Two connected
learners:

* **the inner learner (CAM)** — a frozen language model (Qwen3-14B, served locally) plus a trainable recurrent
  controller (19.7M parameters, all trained) that acts in Crucible worlds through grounded, open-ended action
  programs and learns from verified outcomes (behaviour cloning for the warm start, recurrent PPO with GAE and
  self-imitation replay for reinforcement learning);
* **the outer learner (Meta-Researcher)** — proposes changes to the inner learner's learning algorithm as code
  (written by the frozen LLM, composed from a program space of learning rules, or as parameters of a learned update
  rule), runs each candidate in its own confined source tree against a trusted environment hub under a fixed
  experience budget, compares learning curves with uncertainty, and updates its own trainable selection model.

Nothing in Crucible is replaced; `superlearner` uses it as a library.
Results, including negative ones, are in [docs/RESULTS.md](docs/RESULTS.md); what was implemented versus what was
experimentally demonstrated is in [docs/STATUS.md](docs/STATUS.md).

## Quick start

```bash
cd superlearner
export PYTHONPATH=src            # or: pip install -e .   (then `superlearner ...` instead of `python -m superlearner ...`)

python -m superlearner doctor                                     # hardware, models, isolation, end-to-end CPU check
python -m superlearner train --config configs/smoke_cpu.yaml      # 1-minute CPU smoke run, no language model
python -m pytest -q tests                                         # 41 tests (needs root for the confinement test)

python -m superlearner llm-serve &                                # frozen Qwen3-14B server (owns the GPU model)
python -m superlearner train --config configs/exp_b_fewdemo_sil.yaml
python -m superlearner report --experiment exp_b_fewdemo_sil
python -m superlearner evaluate --checkpoint runs/exp_b_fewdemo_sil/ckpt/latest.pt --split select
python -m superlearner resume --experiment exp_b_fewdemo_sil --iters 5
python -m superlearner inspect-trajectory --trajectory runs/exp_b_fewdemo_sil/traj/eval_iter0030.json.gz

python -m superlearner meta-search --config configs/meta_smoke.yaml     # smallest end-to-end Meta-Researcher round
python -m superlearner meta-search --config configs/meta_search.yaml --full-feedback --resume   # the real search (hours, CPU)
python -m superlearner experiment baselines --checkpoint runs/exp_b_fewdemo_sil/ckpt/latest.pt

python -m superlearner jobs configs/queue_experiments.yaml        # every reported experiment as one restartable queue
```

One GPU is shared by the language-model server and the training runs; `superlearner jobs` admits jobs under GPU and
CPU limits and continues after an interruption ([docs/REPRODUCE.md](docs/REPRODUCE.md)).

## Layout

| Path | What | Who may change it |
|---|---|---|
| `src/superlearner/world/`, `evalkit/` | tasks, splits, the trusted session core and env hub, evaluator metrics | protected |
| `src/superlearner/grounding/`, `actions/`, `llm/` | capability discovery, validation, grounded execution, agent-made tools; action programs; frozen-LLM service | protected |
| `src/superlearner/cam/` | controller network, features, decision procedure, memory, world model | candidates |
| `src/superlearner/algo/` | learning rule (`rule.py`), PPO, self-imitation, behaviour cloning | candidates |
| `src/superlearner/train/` | rollout workers, trainer, curriculum (candidates may change `curriculum.py`), demonstrations, reports | protected |
| `src/superlearner/meta/` | Meta-Researcher: workspaces and confinement, proposals, surrogate, comparison, ledger | protected |
| `src/superlearner/selfplay/` | several learners in one persistent world | protected |
| `src/superlearner/experiments.py`, `jobs.py` | the numbered experiments (resumable); the job queue | protected |
| `configs/` | every run's configuration | |
| `docs/` | architecture, experiments, results, security, limitations | |
| `runs/` | run directories (not committed) | |

Documentation: [ARCHITECTURE](docs/ARCHITECTURE.md) · [ACTIONS](docs/ACTIONS.md) · [TRAINING](docs/TRAINING.md) ·
[META](docs/META_RESEARCHER.md) · [SECURITY](docs/SECURITY.md) · [RESULTS](docs/RESULTS.md) ·
[LIMITATIONS](docs/LIMITATIONS.md) · [STATUS](docs/STATUS.md)
