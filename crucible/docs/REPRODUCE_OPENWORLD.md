# Reproducing Crucible 4.0

Every number in [OPENWORLD_REPORT.md](OPENWORLD_REPORT.md) comes from one of the commands below. Commands run from the
`crucible/` directory unless stated otherwise.

## 1. Requirements

| need | used for | required? |
|---|---|---|
| Python 3.12 (3.11+ supported), `numpy`, `pyyaml` | the world, mechanisms, simulators | yes |
| `pytest`, `pytest-timeout`, `hypothesis` (`[dev]`) | tests | for tests |
| `torch`, `gymnasium`, `pettingzoo` (`[rl]`) | trainable learners, self-play, Gym adapters | for learners |
| `fastapi`, `uvicorn`, `jinja2`, `httpx`, `python-multipart` (`[web]`) | browser mode, dashboard | optional |
| `playwright` + Chromium (`[browser]`) | real-browser tests | optional |
| `psycopg` and PostgreSQL server binaries (`[postgres]`) | PostgreSQL store tests | optional |
| root, Landlock and seccomp (Linux) | **running agent-written code** in the sandbox | for anything that executes agent code |
| GPU with about 8 GB free, Qwen3-4B / 1.7B in the HF cache | frozen-LLM and CAM baselines only | optional |

Exact versions used: `requirements.lock.txt`. Reference machine: 48 logical CPUs behind an **11.5-core cgroup quota**,
shared with other jobs; 188 GB RAM; one RTX A6000 (46 GB), also shared. Wall-clock numbers are pessimistic for that reason.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,rl,web]"
```

A clean-checkout install of the core and dev extras is what acceptance test N verifies.

## 2. Tests

```bash
python -m pytest -q --ignore=tests/test_acceptance.py    # the regression suite: 1,057 tests
python examples/run_acceptance.py                        # acceptance tests A-P, recorded in docs/ACCEPTANCE.md
python -m pytest -q 'tests/test_acceptance.py::test_G_invention_and_second_order_effects'   # one test
```

Run these outside any tool sandbox: tests that execute agent code need the Crucible sandbox. On the reference machine
the regression suite takes about 45 minutes in three parallel groups. Test N clones the repository's `HEAD`, so commit first.

## 3. Experiments

Set `CRUCIBLE_CACHE=<dir>` so validated tasks are built once. Every long script is resumable.

| result | command |
|---|---|
| validity of generated mechanisms | `python examples/ow_validity_acceptance.py --n 12 --workers 3` |
| learner training (one run) | `python examples/ow_train_learners.py train --learner ppo_gru --seed 0 --steps 180000 --eval-every 30000` |
| | `python examples/ow_train_learners.py train --learner model_based --seed 0 --steps 100000 --eval-every 20000` |
| learner tables | `python examples/ow_train_learners.py curves`, then `table`, then `heldout` |
| equal-budget comparison | `python examples/ow_benchmark.py compare --learners ppo_gru,model_based,probe_exploit --reference probe_exploit --seeds 0,1,2 --budget 120000 --eval-every 30000 --n-eval 6 --workers 3 --tag final` |
| three timescales | `python examples/ow_benchmark.py timescales --learners ppo_gru,model_based --seeds 0,1 --budget 60000 --n-eval 4 --workers 2 --tag final` |
| parametric internalisation | `python examples/ow_benchmark.py internalisation --learners ppo_gru,model_based --seeds 0,1,2 --budget 60000 --train-families scm_additive,scm_gate,scm_delay --second-families abm_utility,abm_wom --second-budget 30000 --n-eval 4 --workers 2 --tag final` |
| transfer | `python examples/ow_benchmark.py transfer --learners ppo_gru,model_based --seeds 0,1 --budget 60000 --adapt-budget 20000 --n-eval 4 --workers 2 --tag final` |
| self-play | `python examples/ow_selfplay.py --work <dir> --workers 3` |
| frozen LLM | `python examples/ow_llm_baseline.py llm ablation open short open_v3 sens summary` |
| existing CAM controller | `PYTHONPATH=../src:src python examples/ow_cam_baseline.py --ckpt-dir ../runs/cam2/main_s0/ckpt --run-dir <dir>` (needs that checkpoint, which is not in git) |
| long runs | `python examples/long_run4.py --family ow_market --seeds 1,2,3 --days 365 --arms full --shock-day 120 --prefix ow_longrun365 --jobs 3` |
| | `python examples/long_run4.py --family ow_frontier --seeds 1,2 --days 180 --arms full --shock-day 90 --prefix ow_longrun180 --jobs 2` |
| | `python examples/long_run4.py --family ow_heldout --seeds 1,2 --days 90 --arms full --shock-day 45 --prefix ow_longrun90 --jobs 2` |
| invention trials | `python examples/ow_invention_suite.py --classes capacity --seeds 1,2,3 --days 30 --removal-day 14 --out docs/experiments/ow_invention_capacity.json` (likewise `pricing`, `frontier`) |
| red team | `python examples/redteam_report.py` |
| fidelity register | `python examples/ow_fidelity_report.py` |
| performance | `python examples/bench_openworld.py --quick` (full table: omit `--quick`) |

## 4. The sealed test

The final test set `ow-final` has been opened **once** and its budget is spent. Its seeds derive from a salt in
`~/.crucible/sealed_salt`, which is outside the repository and is lost if this machine is destroyed. To evaluate a new
learner, declare a **new** set rather than reopening this one:

```bash
python examples/ow_benchmark.py sealed --learners <names> --seeds 0,1,2 --sealed-set <new-name> --sealed-n 8 --sealed-budget 1 \
  --purpose "<why>" --ack "I am running the final sealed evaluation and will not use it for tuning" --tag <tag>
```

## 5. Adding a learner

Implement `crucible.research.learner.Learner` (`reset_context`, `act`, `observe`, `update`, `freeze`, `save`, `load`) and
register it in `crucible.research.learners`. Every protocol above then accepts it by name. Observation and action layouts
are documented in `research/learner.py`.

## 6. Determinism guarantees

* **Replay** of recorded events reproduces the state exactly, always.
* **Regeneration** re-executes recorded commands and reproduces every event hash, because outcomes the kernel cannot
  recompute are recorded.
* **Fork equals parent** and **cross-process equality** hold under `params.cost_model = "deterministic"` (all `ow_*` and
  `mech_*_tools` families). Under the measured cost model two fresh executions differ.
* A handler that keeps state in process memory can still cost more on a fresh sandbox worker.
* Frozen-LLM runs are reproducible from their recorded transcripts, not from fresh generation on different hardware.
