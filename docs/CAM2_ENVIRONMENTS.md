# CAM 2.0 — environments

Status as of 2026-11-20 on this machine (unprivileged Docker container, 1× RTX A6000 48 GB, 48 CPU cores,
188 GB RAM, **no Docker daemon, no KVM**, cgroup `pids.max = 2816`). Measured results are in
`docs/CAM2_EXPERIMENTS.md`; this document covers what is integrated, how, and what is blocked.

| Priority | Environment | Status | Evaluator used | Notes |
|---|---|---|---|---|
| A | **AppWorld** 0.1.3 | **Integrated, runs end to end; trained in a smoke-scale run** | AppWorld's own `world.evaluate()` (state-based unit tests) | 9 apps, 457 documented APIs, splits train 90 / dev 57 / test_normal 168 tasks |
| B | **BrowserGym MiniWoB++** (browsergym 0.14.3, Playwright 1.44, Chromium) | **Integrated, main training environment** | MiniWoB's in-page reward via BrowserGym `MiniwobTask.validate` | 28 task families used (16 train / 6 val / 6 test) |
| B | BrowserGym WebArena | Blocked (explicit `DependencyUnavailable`) | — | needs 6 self-hosted web apps from Docker images |
| C | TheAgentCompany | Blocked | — | Docker Compose services + 175 per-task Docker images |
| D | OSWorld | Blocked | — | desktop VMs (VMware / VirtualBox / Docker+KVM) or hosted cloud providers |
| E | Gym-Anything | Blocked | — | Docker, QEMU+Apptainer (Slurm) or Apple Virtualization |
| F | WebGym | Excluded (safety policy + non-deterministic evaluator) | — | ~292k tasks on 127,645 real websites, rubric-based LLM judges |

Selecting an unavailable environment (`--set env=webarena` etc.) raises `DependencyUnavailable` with the blocker
and the setup procedure (`src/agentrs/cam2/envs/unavailable.py`); nothing silently substitutes another
environment. The only non-benchmark environment in the package is `envs/fixture.py`, labelled TEST FIXTURE ONLY,
used by unit tests for speed and never for reported results.

## Unified interface

`src/agentrs/cam2/envs/base.py` — `EnvironmentAdapter`:

| method | MiniWoB adapter | AppWorld adapter |
|---|---|---|
| `reset(task)` | `gym.make("browsergym/miniwob.<family>")`, `reset(seed)`; browser rebuilt once if it crashed | `AppWorld(task_id=...)` in a worker process |
| `observe()` | BrowserGym accessibility tree (`flatten_axtree_to_str`, `tags_to_mark="all"`) → `Element`s (no bids in the text) | last 3 calls + flattened JSON output of the last call as non-targetable elements, plus raw output text |
| `available_tools()` | 10 generic primitives (`click`, `dblclick`, `hover`, `focus`, `fill`, `clear`, `press`, `select_option`, `scroll`, `wait`) | every API of the task's apps from AppWorld's own API docs (457 tools), each with one JSON argument validated against a schema derived from the documented parameters |
| `step(call)` | schema validation → BrowserGym high-level `bid` action string | schema validation → `apis.<app>.<api>(**args)` executed in AppWorld's sandboxed interpreter |
| `evaluate()` | MiniWoB reward > 0 | `world.evaluate()`: success = all tests pass; score = passed / total |
| `snapshot()/restore()` | (task, seed, action history) + replay; refused for families flagged non-deterministic | AppWorld native `save_state` / `load_state` |
| `close()` | closes the browser | closes the world and the worker |

Tools are published at run time as typed schemas (`tools.py`: `ToolSpec`, `ArgSpec` with types `element`,
`enum`, `text`, `json`, and an `effect` of `read` / `sandboxed` / `external`). CAM never sees a fixed action
vocabulary: the candidate set at every step is enumerated from the schemas and the current observation, and CAM
scores candidates with a pointer network over their frozen-LLM embeddings (`featurize.py`). Every call is
validated against its schema before execution; invalid calls never reach the environment, cost a step and are
counted (`invalid_rate`).

Asynchronous workers: `envs/pool.py` runs one adapter per spawned subprocess with per-command deadlines; a hung
worker is killed together with its browser process group and replaced. Thread pools in workers are capped at one
thread (otherwise every worker started 48 unused BLAS/OpenMP threads, which exhausted the container's pid limit).

## Safety

* MiniWoB pages are local `file://` HTML; the adapter routes every browser request and aborts anything that is not
  `file://`, `data:` or `about:` (counted as `blocked_requests`).
* AppWorld executes API calls inside its own simulated databases and sandboxed interpreter, in a separate
  virtualenv and process (`appworld_worker.py`, started with `python -I`); only agent-visible information (task
  instruction, supervisor profile, API docs, outputs of the agent's own calls) leaves the worker. Ground truth is
  used only by AppWorld's evaluator inside the worker.
* `ToolGate` refuses tools with `effect="external"` unless explicitly approved; no training configuration
  approves any. No environment used here can send messages to real people, make payments or touch credentials
  outside its simulation.

## MiniWoB++ via BrowserGym

* Families (fixed before any CAM 2.0 result; `envs/miniwob.py`):
  TRAIN (16): click-button, click-link, click-test-2, click-dialog, click-checkboxes, click-tab, click-option,
  focus-text, enter-text, enter-password, login-user, click-collapsible, choose-list, click-widget,
  enter-text-dynamic, click-button-sequence.
  VAL (held-out families, 6): click-dialog-2, focus-text-2, enter-text-2, click-tab-2-easy,
  click-checkboxes-transfer, multi-layouts.
  TEST (locked, 6): unicode-test, login-user-popup, click-collapsible-2, click-checkboxes-soft, use-autocomplete,
  navigate-tree.
* Seeds: train 0–999,999; val 1,000,000–1,099,999; test 2,000,000–2,099,999.
* Outer validation = held-out VAL families + val-range seeds of TRAIN families (instance-level generalisation),
  so per-family results separate "new instances of trained families" from "unseen families".
* Episode limit 10 actions; truncation at the limit is bootstrapped, not treated as terminal.
* Throughput on this machine: ~10–11 environment steps/s with 16 browser workers (one worker ≈ 78 threads, so
  ≤ ~24 browser workers fit under the pid limit together with the learner processes).

## AppWorld

* Installed in `/workspace/ext/venv-appworld` (AppWorld pins its own dependencies; `appworld install` +
  `appworld download data`), data in `/workspace/ext/appworld_root`.
* Splits: inner training = `train`, outer validation = `dev`, locked test = `test_normal`.
* Action: CAM's pointer network chooses one of the 457 APIs; the frozen LLM, conditioned on CAM's control vectors,
  writes the JSON arguments; the arguments are validated against the API's documented parameter schema.
* Episode end: the agent calls `supervisor.complete_task` (AppWorld's own completion API) or reaches 15 calls.
* Reward: 1 on success; otherwise `0.5 × (tests passing at the end − tests passing on the untouched state) /
  num_tests` at episode end (credit only for tests the agent's actions made pass; "no collateral damage" tests that
  pass on the initial state give nothing). Success and score in reports are AppWorld's official ones.

## Blocked integrations — what would be needed

Sources read on 2026-11-20: the projects' READMEs and setup docs (github.com/web-arena-x/webarena,
github.com/TheAgentCompany/TheAgentCompany + docs/SETUP.md, github.com/xlang-ai/OSWorld,
github.com/cmu-l3/gym-anything + cmu-l3.github.io/gym-anything/docs/{installation,benchmarks/tasks-verifiers},
webgym-website.github.io + github.com/microsoft/webgym).

* **WebArena** (BrowserGym): six self-hosted web applications (shopping, shopping-admin, GitLab, Reddit,
  Wikipedia, map) from the WebArena Docker images. On a Docker host: start them per
  `webarena/environment_docker`, `uv pip install browsergym-webarena`, set the `WA_*` URLs, and add an adapter
  mirroring `MiniWoBAdapter` (BrowserGym ids `browsergym/webarena.<n>`; the evaluator is WebArena's own).
* **TheAgentCompany**: GitLab, Plane, ownCloud and RocketChat via Docker Compose (30+ GB, mounted docker socket);
  each of the 175 tasks is a Docker image with an encrypted evaluator (`/utils/eval.py`); some graders use an LLM.
* **OSWorld**: desktop VMs; local providers VMware / VirtualBox / Docker (recommended with KVM). Hosted providers
  (AWS, Modal, Daytona) avoid local virtualization but need external accounts, which autonomous training here
  does not use.
* **Gym-Anything** (MIT): environments launched through Docker (general), QEMU + Apptainer on rootless Slurm, or
  Apple Virtualization on macOS. Tasks are verified mostly by programmatic `verifier.py`; some use a VLM checklist.
  It would be the most promising way to scale environment diversity on a Docker host.
* **WebGym** (code MIT): ~292k tasks over 127,645 real websites; success judged by rubric-based LLM evaluators.
  Excluded: training would interact with the live web (no sandboxed copy is described), which this project's
  autonomous-training policy forbids, and its evaluator is not a deterministic state-based verifier. If approved,
  it should run against an allowlist of read-only sites with LLM-judge scores reported separately.
