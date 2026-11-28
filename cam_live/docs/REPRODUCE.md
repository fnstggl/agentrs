# CAM Live 1.0 — dependencies and exact commands (IN-05, CM-13)

Describes the code at commit `fdab7d3` (read 2026-11-28 07:16 UTC). No commit after `fdab7d3` touches `cam_live/src`,
`tools`, `configs`, `tasks` or `pyproject.toml` (`git log fdab7d3..HEAD -- …` empty at the time of reading).

One place for everything needed to set up, test, validate, evaluate, authorize, launch, operate, back up and verify
CAM Live. Every command below was checked on this machine on 2026-11-28 between 07:20 and 07:48 UTC, from the
worktree `/workspace/camlive` (branch `feat/cam-live`; HEAD `fdab7d3`, moved to `90274bb` during the checks by
documentation and probe commits only — none touches the paths above). The record of every check (exit code, duration,
first output line, result line) is `docs/evidence/DOC/command_check.txt`; it was produced by
`docs/evidence/DOC/scripts/check_help.sh` (every `--help`) and `docs/evidence/DOC/scripts/check_exec.sh` (every
offline execution), both re-runnable.

Each command carries a mark:

* **[help]** — `--help` ran and exited 0 (argparse handles `--help` before doing any work);
* **[exec: …]** — executed here, offline, with `CAM_LIVE_HOME` in a scratch directory; the result is stated;
* **[not run: …]** — not executed, with the reason (network, live service, services started as root, the operator's
  own act, or a write into files another workstream owns).

Conventions: all commands run from `/workspace/camlive` unless stated otherwise; `P=.venv/bin/python`; `cam-live` is
`.venv/bin/cam-live`, the only console script (`pyproject.toml` `[project.scripts]` → `runtime/cli.py:main`; also
`$P -m cam_live.runtime.cli`). Exit codes of `cam-live` (`runtime/cli.py` module docstring): 0 ok, 1 the command
ran and reports a failure, 2 usage, 3 refused (preflight / authorization / task validation), 4 a needed component
is not installed, 5 the runtime did not start.

---

## 1. Dependencies

### 1.1 Hardware and host (`cam-live doctor` [exec: exit 0, 49/52 checks ok], `nvidia-smi`)

| item | this machine (2026-11-28, 07:20–07:48 UTC) | needed for |
|---|---|---|
| GPU | 1x NVIDIA RTX A6000, 47.4 GB (25.5–26.0 GB free during the checks); **shared** with another tenant, coordinated by one lock file (§1.4) | real backbone (Qwen3-4B bf16 + 50m core): peak 18.0 GB in the last real-configuration staging validation (`docs/evidence/validation/staging_20261128T051702Z.json`, `gpu_peak_gb`; not re-checked) |
| CPU / RAM | 48 logical CPUs, 126 GB RAM (85–88 GB available) | CPU test suite, debug-scale runs |
| disk | 300 GB overlay (the state root, `/workspace` and `/tmp` are on it), **32–35 GB free** during the checks (doctor: above `storage.min_free_gb` 20, below `warn_free_gb` 50) | runs and checkpoints; below 20 GB nothing is checkpointed and a run pauses (§11) |
| persistent volume | **none** (`workspace_is_volume=False`; doctor FAIL `persistent_storage`, the only doctor item blocking live mode; accepted for a live run only with `--accept-ephemeral-storage`, §9.1) | durable state, or an off-box copy (§12) |
| isolation | root, setuid/setgid, Landlock ABI 1, seccomp; **no** CAP_NET_ADMIN, **no** user namespaces (doctor FAIL `kernel_network_filtering`, `user_namespaces`; not blocking) | separated services under five uids; per-destination network filtering is enforced by the authorization layer only (a VM is recommended for live operation) |
| service users | `camlearn` 999, `cambrowser` 997, `camsbx` 996, `camauth` 995, `camverify` 994 (doctor ok) | separated mode; created by the launcher (`harness/auth/launcher.py:_ensure_users`) on the first separated `cam-live run` as root |

Doctor [exec]: `49/52 checks ok; failed: persistent_storage, kernel_network_filtering, user_namespaces`; "blocking for
staging: nothing", "blocking for live: persistent_storage"; `--strict` exit 0 (it fails only when something blocks
staging). The check list is `runtime/doctor.py:run_doctor`.

### 1.2 Python packages

`cam_live/pyproject.toml`: `requires-python >= 3.11`; runtime dependencies torch>=2.4, transformers>=4.51,
safetensors, numpy, pyyaml>=6, httpx>=0.27, fastapi>=0.115, uvicorn>=0.30, playwright>=1.48, pillow>=10,
cryptography>=42, beautifulsoup4>=4.12, pypdf>=4, websockets>=12; extra `dev` = pytest>=8, pytest-timeout>=2, scipy.
Installed versions [exec: `importlib.metadata` in `.venv/bin/python -I`; "site" = where the distribution is loaded from]:

| package | installed | site | package | installed | site |
|---|---|---|---|---|---|
| Python | 3.12.15 | /venv/main (base interpreter) | fastapi | 0.165.0 | agentrs venv |
| torch | 2.15.0+cu132 (CUDA 13.2, cuDNN 9.26) | agentrs venv | uvicorn | 0.54.0 | agentrs venv |
| transformers | 5.24.0 | agentrs venv | playwright | 1.65.2 (Chromium build 1255) | agentrs venv |
| safetensors | 0.8.0 | agentrs venv | pillow | 13.0.0 | agentrs venv |
| huggingface_hub | 2.4.0 | agentrs venv | cryptography | 50.0.2 | agentrs venv |
| tokenizers | 0.23.3 | agentrs venv | beautifulsoup4 | 4.15.0 | agentrs venv |
| numpy | 2.5.5 | camlive venv | websockets | 17.4 | agentrs venv |
| scipy | 1.18.1 | camlive venv | httpx | 0.28.1 | agentrs venv |
| pytest | 9.2.0 | camlive venv | pyyaml | 6.0.4 | agentrs venv |
| pytest-timeout | 2.4.0 | camlive venv | pypdf | 7.2.0 | camlive venv |
| cam-live | 1.0.0 (editable, `cam_live/src`) | camlive venv | | | |

"agentrs venv" = `/workspace/agentrs/.venv/lib/python3.12/site-packages`, put on the path by
`.venv/lib/python3.12/site-packages/_shared_base.pth` (§2.1). This layering is an open finding (G2-integration IN-09
partial, `docs/verification/G2_integration.md` R2.5): nothing under `cam_live/` imports `superlearner`, `crucible` or
`agentrs`, but the packages above are loaded from that directory. `uv` 0.13.4 is at `/usr/local/bin/uv`.

### 1.3 Models (pinned)

| model | repository @ revision | licence | size | where |
|---|---|---|---|---|
| backbone (frozen) | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` (`config.py:BackboneConfig`, `configs/cam_live.yaml`) | Apache-2.0 | 7.5 GB (3 safetensors shards, doctor) | `$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a72…` |
| visual grounder, lightweight | `HongxinLi/GoClick-Base` @ `05633a495f9d2ce823372d0c7fd9b0b4bb67c6fa` | MIT | — | `$HF_HOME/hub` |
| visual grounder, advanced | `Tongyi-MAI/MAI-UI-2B` @ `503050934809558c8dfd2ddedaf9621fa74ac2de` | Apache-2.0 | — | `$HF_HOME/hub` |
| visual grounder (baseline only) | `vocaela/Vocaela-2-500M-1024R2` @ `b0280f2ce8e2f126a1c3c446c70e63ee7bbeaed2` | CC BY-NC-SA 4.0 (non-commercial) | — | `$HF_HOME/hub` |

`HF_HOME` defaults to `/workspace/.hf_home` (set by `nn/backbone.py:HFBackbone.__init__` and `tests/conftest.py` when
unset; the launcher passes `HF_HOME` through to the services, `harness/auth/launcher.py:ENV_PASSTHROUGH`). The
backbone loads with `local_files_only: true` and `HF_HUB_OFFLINE=1`: nothing is downloaded at run time. Doctor
[exec]: `backbone: Qwen/Qwen3-4B@1cfa9a720891: 3 weight file(s), 7.5 GB`, all three grounders `available: true`.

### 1.4 The shared GPU lock

The machine's GPU is shared with another tenant. Every heavy GPU job takes one `flock` on one lock file and holds it
for at most about 15 minutes.

* **Path**: `config.py:GPU_LOCK_PATH` = `$CAM_LIVE_GPU_LOCK`, else `<repo>/runs/gpu_heavy.lock`. On **this machine only**
  `/workspace/camlive/runs/gpu_heavy.lock` is a symbolic link (untracked; `runs/` is git-ignored) to the tenant's
  lock `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock` [exec: `ls -la runs/`]. In any other checkout or archive
  copy that path does not exist.
* **Never created**: `nn/gpu.py:open_lock_file` opens the existing file only; if it is missing it raises
  `GpuUnavailable` naming `CAM_LIVE_GPU_LOCK` (a private lock would coordinate with nobody). A service uid that may
  not write the file (the machine's lock is root:root 0664 and `camlearn` is neither, per the function's docstring)
  opens it read-only — `flock` works on any open file. Every lock path of the code uses it, among them
  `nn/gpu.py:gpu_lock` (scripts), `runtime/gpu_cycle.py:GpuLease` (the runtime), `runtime/cli.py:gpu_lock`
  (`cam-live init`), `validation.py:shared_gpu` and the test fixture (`tests/conftest.py`).
* **Doctor**: check `gpu_lock` (live-blocking when the file is absent) [exec with `CAM_LIVE_GPU_LOCK=<absent file>`:
  exit 0, "blocking for live: persistent_storage, gpu_lock"].
* **Runtime lease** (`runtime/gpu_cycle.py:GpuLease`, started by `runtime/loop_service.py:run_service` when
  `runtime/loop_service.py:needs_gpu_lock` is true): `configs/cam_live.yaml` sets `runtime.gpu_lock: auto` (= a
  pretrained backbone on CUDA; `on` / `off` force it), `runtime.gpu_hold_s: 840`, `runtime.gpu_yield_s: 15`;
  optional `runtime.gpu_lock_path` overrides the path for the runtime alone. The lock is taken before the agent is
  built; at the decision boundary where a hold would exceed `gpu_hold_s` the runtime commits its checkpoint,
  releases, yields `gpu_yield_s`, re-acquires (queued, blocking) and continues in the same process — no exit, no
  manual resume; it releases whenever the runtime sleeps; waiting is shown as `waiting_for_gpu_lock` and is not
  charged to compute (module docstring of `runtime/gpu_cycle.py`; code reading).
* **Separated services**: the launcher passes `CAM_LIVE_GPU_LOCK` to every service (`harness/auth/launcher.py:ENV_PASSTHROUGH`).
* **On another machine**: set `CAM_LIVE_GPU_LOCK` to that machine's shared lock file. With no other GPU user, create one
  lock file outside any checkout, readable by the service uids, and point `CAM_LIVE_GPU_LOCK` at it; otherwise every
  real-model command fails closed. [not run: follows from
  `nn/gpu.py:open_lock_file`.]
* Never wrap pytest or a script in `flock` on the lock file: the code takes the lock itself and would deadlock.

---

## 2. Environment setup

### 2.1 As built on this machine (layered venv)

Reconstructed from `.venv/pyvenv.cfg` (`command = /workspace/agentrs/.venv/bin/python -m venv --without-pip
/workspace/camlive/.venv`) and the two `.pth` files present; the creation itself was not re-executed:

```bash
cd /workspace/camlive
/workspace/agentrs/.venv/bin/python -m venv --without-pip /workspace/camlive/.venv
echo /workspace/agentrs/.venv/lib/python3.12/site-packages \
    > .venv/lib/python3.12/site-packages/_shared_base.pth          # base packages (torch, transformers, ...)
uv pip install --python .venv/bin/python -e 'cam_live[dev]'        # editable cam_live (+ numpy, scipy, pytest, pypdf)
.venv/bin/python -m playwright install chromium                    # -> ~/.cache/ms-playwright/chromium-1255
```

Extra packages go only into this venv: `uv pip install --python /workspace/camlive/.venv/bin/python <pkg>`.
[not run: the environment exists; `uv pip install` and `playwright install` need the network.]
`$P -m playwright --version` [exec: `Version 1.65.2`].

### 2.2 On a fresh machine (no agentrs venv)

The same versions as measured in §1.2, without the layering [not run: needs the network]:

```bash
git clone <repository> camlive && cd camlive && git checkout feat/cam-live
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv/bin/python -e 'cam_live[dev]' transformers==5.24.0 safetensors==0.8.0 \
    huggingface_hub==2.4.0 tokenizers==0.23.3 numpy==2.5.5 fastapi==0.165.0 uvicorn==0.54.0 playwright==1.65.2 \
    pillow==13.0.0 cryptography==50.0.2 beautifulsoup4==4.15.0 pypdf==7.2.0 websockets==17.4 httpx==0.28.1 \
    pyyaml==6.0.4 scipy==1.18.1 pytest==9.2.0 pytest-timeout==2.4.0
.venv/bin/python -m playwright install --with-deps chromium
export CAM_LIVE_GPU_LOCK=<the machine's shared GPU lock file>     # §1.4: never created by CAM Live
```

The GPU must support the wheel's CUDA (cu132 here; Blackwell GPUs need a CUDA >= 12.8 build).

### 2.3 Chromium for the browser uid (separated mode, once, as root)

```bash
$P -m cam_live.harness.browser --install-shared     # copy this user's Playwright Chromium to .venv/share/cam_live/ms-playwright   [help]
$P -m cam_live.harness.browser --find cambrowser    # [exec: exit 0, .venv/share/cam_live/ms-playwright/chromium-1255/chrome-linux64/chrome]
```

### 2.4 Model download

```bash
export HF_HOME=/workspace/.hf_home
$P -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c')"
# grounders (visual computer use only): the same call with the repositories and revisions of §1.3
```

[not run: the snapshots are present (doctor ok); downloading needs the network.]

### 2.5 Configuration and state root

* `cam_live/configs/cam_live.yaml` — the full candidate: Qwen3-4B + `50m` core, `device: auto`,
  `nn.max_prompt_tokens: 6144` (the planning task's proposer prompt is 4.1–5.3 k tokens), `storage.min_free_gb: 20`,
  `storage.warn_free_gb: 50`, `runtime.keep_checkpoints: 2`, `runtime.gpu_lock: auto`, `runtime.gpu_hold_s: 840`,
  `runtime.gpu_yield_s: 15`. Every `cam-live` command takes `--config FILE`, a complete replacement, not an overlay
  (`runtime/cli.py:_cfg`; `tools/planning_rehearsal.py prepare --config` is the exception: it merges over the default).
* State root: `home:` in the configuration, else `$CAM_LIVE_HOME`, else `<repo>/runs/cam_live`
  (`config.py:CamLiveConfig.resolved_home`). Run data is never in git.
* `cam-live init [--home H] [--no-agent] [--agent-factory M:F]` [help] — creates `H/{runs,checkpoints,exports,validation}`,
  `config.resolved.json` and the dashboard token; without `--no-agent` it builds the agent and saves
  `H/checkpoints/init` (under the GPU lock for a pretrained backbone on CUDA, `runtime/cli.py:cmd_init`).
  `cam-live init --no-agent --home H` [exec: exit 0, state root and `dashboard.token` created].
* A CPU debug configuration for plumbing checks (tiny random backbone, debug core) — the one used for §9.2 here:

  ```yaml
  seed: 0
  device: cpu
  home: ""
  backbone: {name: tiny, revision: "", lora_rank: 2, bridge_tokens: 2, max_text_tokens: 48}
  core: {scale: debug, max_context: 48}
  ```

---

## 3. Tests

```bash
cd /workspace/camlive
.venv/bin/python -m pytest cam_live/tests/<dir> -q --rootdir=cam_live --junitxml=cam_live/docs/evidence/test_results/<id>.xml
.venv/bin/python -m pytest cam_live/tests -q --rootdir=cam_live --junitxml=<file>      # whole default suite, one process
```

The whole suite collects in one process: `pyproject.toml` sets `addopts = "--import-mode=importlib"` (module names
from the path) and `tests/conftest.py` puts the helper directories on `sys.path`. Collected tests per directory
[exec: `pytest <dir> --collect-only -q -p no:cacheprovider --rootdir=cam_live`, each exit 0]:

| `<dir>` | collected | `<dir>` | collected |
|---|---|---|---|
| `credit` | 152 | `predict` | 176 |
| `diagnostics` | 8 | `reward` | 146 |
| `eval` | 42 | `runtime` | 299 |
| `harness` (with `auth`, `compute`, `env`, `skills`) | 629 | `storage` | 4 |
| `integration` | 68 | `test_secret_scan.py` | 6 |
| `nn` | 122 | `test_verify_completion.py` | 27 |
| **whole suite, one process** (`cam_live/tests`) | **1679** (= the sum; exit 0, 17 s) | | |

`harness` = `harness/auth` 305 + `harness/compute` 74 + `harness/env` 99 + `harness/skills` 151 (each collected on
its own, exit 0).

* **Latest recorded results** (not re-run here):
  * whole default suite in one process on an archive of the final freeze `257267c`:
    **1649 tests, 1542 passed, 107 skipped, 0 failed, 0 errors**, 2026 s
    (`docs/evidence/test_results/FINAL_suite_257267c.xml`, committed in `fbe2156`);
  * `tests/runtime` + `tests/integration` in one process on the code of `2247328` (= the code of `fdab7d3`):
    **367 tests, 355 passed, 12 skipped, 0 failed**, 688 s (`docs/evidence/test_results/INT3B_fix3.xml`, committed in
    `fdab7d3`). The run before it with the same command (`INT3B_fix3_run1.xml`) had **2 failures**,
    `tests/integration/test_int2_wiring.py::test_runtime_executes_in_the_computer_environment_without_grant_refusals`
    and `tests/integration/test_int3a_lrn6.py::test_a_refused_promotion_keeps_psi_the_meta_optimizer_and_an_unjudged_policy`;
    both pass alone and in the second run (order / process-state dependent, not reproduced; `docs/status/F.json`
    `int3b_round3.junit`).
  * No whole-suite run exists after the code commits between `257267c` and `fdab7d3` (`59807fc`, `7265843`,
    `e091b41`, `929434c`, `2247328`; evaluation code `67d7297`, `1488a98`, `d6d42e6`, `2e54bc4`). Partial runs recorded
    with them: `tests/nn` + `tests/runtime` 380 passed (commit messages of `59807fc` and `7265843`; no JUnit file) and
    `INT3B_fix3.xml` above.
* `cam-live test [--suite <dir> …] [-m MARK] [-k EXPR] [--results-dir D]` [help] runs one pytest process per
  `--suite` (without `--suite`: the whole tree in one process) and writes `D/cli_<dir>.xml` (default
  `cam_live/docs/evidence/test_results/`) [exec: `--suite storage --results-dir <scratch>`: exit 0, 4 passed].
* Opt-in marks (`tests/conftest.py:pytest_collection_modifyitems`; the default run is CPU-only and offline; unselected
  opt-in tests are collected and skipped): `-m gpu` (real Qwen3-4B; the test takes the shared GPU lock itself, §1.4),
  `-m slow`, `-m browser` (Playwright Chromium), `-m privileged` (root; the separated service uids). **Never
  `-m network`.** Per-test timeout 900 s (`pyproject.toml`); `tests/integration/test_agent_runtime.py` sets 2400 s.
* Independent verification probes: `$P cam_live/docs/verification/repro/<lens>/<probe>.py` (lenses `cl`, `env`,
  `g2decision`, `g2final`, `g2harness`, `g2integration`, `g2learn`, `g2runtime`, `nn`, `pr`, `rt`, `sec`; each
  documented in `docs/verification/G_*.md` / `G2_*.md`) [not run here except probe d03, §14; owned by the verifiers].

---

## 4. Health, harness inventory, preflight

| command | what | check |
|---|---|---|
| `cam-live doctor [--json] [--strict]` | hardware, storage, models, dependencies, sandbox (Landlock, seccomp, setuid), service users, harness modules, GPU lock | [exec: exit 0, 49/52 ok, §1.1; `--strict` exit 0; `--json` exit 0] |
| `cam-live inspect-harness [--task T] [--run-dir R] [--lab] [--no-browser]` | tools, permissions, missing integrations (in-process inspection environment unless `--run-dir` names a run with running services) | [exec `--lab`: exit 0, 5 LabWorld tools; missing integration: Discord (no Discord executor or credential configured)] |
| `cam-live preflight --task T [--no-doctor] [--accept-ephemeral-storage] [--json]` | the readiness report alone; offline the services are not running, so their items are "not confirmed"; exit 0 when the task's own mode is ready, else 3 | see the table below |
| `cam-live run --task T --preflight-only [--no-doctor]` | the same report through `run` (writes nothing) | [exec `lab_staging.yaml`: exit 0] |

Preflight of every task file [exec, with the doctor, offline, scratch `CAM_LIVE_HOME`; `--no-doctor` leaves out the
doctor's `persistent_storage` item, so each LIVE count is one lower]:

| task file | mode | exit | LIVE | STAGING |
|---|---|---|---|---|
| `tasks/discord_100_planning.yaml` (planning phase) | live | 3 | BLOCKED, 9 items: authorization service, `discord_bot_token` in the vault, computer environment, verifier credential copy, bot membership, privileged intent GUILD_MEMBERS, bot permissions, independent verifier (all need the running services), `persistent_storage` | ready |
| same, `--accept-ephemeral-storage` | live | 3 | BLOCKED, 8 items (the same without `persistent_storage`, which is listed as ACCEPTED RISK "operator (command line --accept-ephemeral-storage)") | ready |
| `tasks/discord_launch_check.yaml` (launch-path check: one decision, two external actions, 30 min) | live | 3 | BLOCKED, 9 items (as the planning task); 8 with `--accept-ephemeral-storage` | ready |
| `tasks/discord_100.yaml` (execution phase) | live | 3 | BLOCKED, 25 items: its 16 values marked REQUIRED are unset (5 of them limits) plus the 9 service / storage items (the guild item replaces the vault item) | ready |
| `tasks/staging/discord_100_planning_staging.yaml` (staging rehearsal of the planning task) | staging | **0** | BLOCKED, 10 items (the 9 + "mode: a live run needs mode: live") | ready |
| `tasks/lab_staging.yaml` (LabWorld) | diagnostic | 0 | (diagnostic) | ready |
| `tasks/utility_staging.yaml` (template) | staging | 3 | BLOCKED, 8 items | not ready, 2 items: `success.evidence_source`, `resources.staging_site` unset (marked REQUIRED) |

With the services running, the supervisor of `cam-live run` repeats the preflight against them, including the
read-only Discord identity / intent / permission check, and starts the learner only if every live item is confirmed
(`runtime/cli.py:cmd_supervise`).

---

## 5. Staging validation and rehearsals (controlled services only)

```bash
cam-live validate --mode staging [--real-model | --debug] [--decisions N] [--seed S] [--out D] [--evidence-dir E] \
    [--budget-s 840] [--meta-min-independent 3] [--lab-lifetimes 6] [--compute-limit-s X] [--computer-share 0.25]   # [help]
```

Defaults (`runtime/cli.py:build_parser`, `validation.py:run_staging`; 0 on the command line means "the validation's
default"): `--real-model` (the configured backbone and core on the GPU); 48 decisions (`validation.py:DEFAULT_DECISIONS`:
12 in the computer environment — 8 with operator approvals, then 4 under a compute limit — and the rest shared by the
LabWorld lifetimes); `--budget-s` 840 (`DEFAULT_BUDGET_S`: seconds the GPU lock is held at most, building and final
measurements included); `--meta-min-independent` 3 (`DEFAULT_META_MIN_INDEPENDENT`); `--lab-lifetimes` 6 (targets
`DEFAULT_LAB_TARGETS`); `--compute-limit-s` 8 s per decision of the limit objective (2 s with `--debug`);
`--computer-share` 0.25. With `--out D`: run directory `D`, report `D/staging_report.json`, evidence file
`D/staging_<UTC>[_cpu].json` (also in `E` with `--evidence-dir E`); without `--out` the run directory is
`<home>/validation/staging_<UTC>` and the report `<home>/validation/staging_report.json`
(`runtime/cli.py:cmd_validate`). Exit 0 only when every check holds. `--mode live` is refused [exec: exit 2]:
live readiness comes only from `preflight` / `run`.

* **Real configuration** (Qwen3-4B + 50m; the shared GPU lock inside `validation.py:run_staging`, needs >= 18 GB free
  GPU memory) [not run here: GPU lock budget]. Last recorded runs, both **executed, not passed, 13 of 16 checks**:
  INT3B at `a3844c8` (`docs/evidence/validation/staging_20261128T051702Z.json`: lock 685 s, 948 s wall, peak 18.0 GB;
  failing `approvals_exercised`, `meta_update_applied`, `meta_step_changes_policy`) and G2-integration round 2 at
  `257267c` (`docs/verification/repro/g2integration/out/round2_staging_20261128T055027Z.json`, lock 812 s; same three
  failures; LX-11 failed, `docs/verification/G2_integration.md` R2.2).
* **CPU debug configuration**: `--debug` / `--cpu` always uses the tiny CPU configuration, whatever `--config` says
  (`validation.py:staging_config` keeps only the state root of a given configuration; the earlier defect KF-E11 is
  fixed). [exec: `cam-live validate --mode staging --debug --decisions 12 --out <new scratch dir>` with the GPU hidden
  (`CUDA_VISIBLE_DEVICES=`) and `CAM_LIVE_GPU_LOCK` naming an absent file, twice: both exit 1, `executed: true`,
  backbone `tiny`, core `debug`; 07:28 UTC 13 of 16 checks in 77 s (`approvals_exercised`, `meta_update_applied`,
  `meta_step_changes_policy` fail), 07:45 UTC 15 of 16 in 114 s (`approvals_exercised` fails) — at 12 decisions
  (3 computer decisions, 2 per LabWorld lifetime) the result varies between runs.] **Use a new `--out`
  directory:** a run into a directory that already holds a validation run does not start from scratch (measured: a
  second run into the same directory ended after 9.5 s with 12 of 16). The last recorded full CPU run (48
  decisions) held 16 of 16 checks (`docs/evidence/validation/staging_20261128T051613Z_cpu.json`, `a3844c8`, 289 s;
  not re-checked). A plumbing diagnostic, not evidence about learning.
* **Separated-services staging run** (root; the launcher starts authorization, verifier, sandbox host, executor and
  learner under their uids; the script approves / rejects as the operator and stops the run):
  `$P cam_live/docs/evidence/validation/separated/run_separated.py --out X.json [--decisions N] [--base-port P]
  [--timeout S] [--keep]` [help; not run: starts services as root]. Last result
  `docs/evidence/validation/separated/separated_20261127T183623Z.json` (not re-checked).
* **Staging rehearsal of the planning phase** (`tools/planning_rehearsal.py`; staging only — a local staging Discord
  on 127.0.0.1:18471 and a neutral local web on 127.0.0.1:18472; the tool never answers an approval or a request and
  writes nothing CAM reads; task `tasks/staging/discord_100_planning_staging.yaml`, the operator's objective verbatim):

  ```bash
  $P cam_live/tools/planning_rehearsal.py prepare --root R [--config C] [--hold-s 840]   # run directory, staging token, policy, config
  $P cam_live/tools/planning_rehearsal.py serve --root R                                 # staging Discord + neutral web
  $P cam_live/tools/planning_rehearsal.py holds --root R [--decisions 25] [--max-holds 6] # cam-live run, then cam-live resume per hold
  $P cam_live/tools/planning_rehearsal.py check --root R [--out F]                       # counts and verbatim excerpts of what the run recorded
  ```

  [help: the tool and all four subcommands exit 0; not run: it starts services and the real model.] The separated
  variant (master, 2026-11-28 06:05–06:25 UTC, code `59807fc`): `cam-live run --task
  tasks/staging/discord_100_planning_staging.yaml` without `--inproc`, staging services from `planning_rehearsal.py
  serve`; the learner (camlearn) waited 849 s for the shared lock, ran 8+ decisions; Discord writes were queued for
  approval, none reached the staging Discord; `cam-live plan-report` exported the report; `cam-live stop`
  checkpointed. An artefact of the rehearsal tool (not of `tools/launch_live.sh`): `prepare` creates root-owned files
  in `<run>/auth`, which had to be chowned to camauth by hand (`docs/evidence/INT3B/rehearsal_separated_README.md`,
  `rehearsal_separated_20261128.json`; not re-checked).
* Local staging sites alone: `$P -m cam_live.harness.staging [--base-port P]` [help; not run: serves until stopped].

---

## 6. Evaluation

All write JSON with `"executed": true` only after the experiment ran. GPU runs take the shared lock per hold
(<= ~15 min, §1.4). LabWorld / tiny-backbone runs are engineering diagnostics only (`docs/S/PROTOCOLS.md`,
Protocol v3). `CAM_LIVE_EVAL_HOME` sets where runs, checkpoints and results go (default `<repo>/runs/cam_live/eval`,
`eval/experiments.py:EVAL_HOME`); `CAM_LIVE_EVAL_EVIDENCE` the evidence directory of `eval.cli analyse`
(`eval/cli.py:EVIDENCE`).

| command | purpose | check |
|---|---|---|
| `$P -m cam_live.eval.oe_batch [--plan {v3,v3x5}] [--only x0,x1] [--concurrency 3] [--max-gpu-gb 36] [--hold-s 810] [--yield-s 20] [--threads 6]` | Protocol v3 scheduler: one parent holds the GPU lease (`runtime/gpu_cycle.py:GpuLease`), up to 3 `oe_runner sequence --max-objectives 1 --no-lock --resume` children per lease; jobs in `eval/oe_batch.py:PLANS`. The S2 run on `257267c`: `--plan v3 --concurrency 3 --max-gpu-gb 36` with `CAM_LIVE_EVAL_HOME=runs/cam_live/eval/v5` (`docs/S/RESULTS.md` §1) | [help] |
| `$P -m cam_live.eval.oe_runner sequence --condition E --seed 0 --plan SW:rle,MECH:7 --decisions 20 --out R.json [--root D] [--tag T] [--resume] [--save-exports] [--max-objectives N] [--hold-s 870] [--min-free-gb 14] [--lock-timeout-s 7200] [--cpu] [--no-skills] [--threads 8] [--no-lock] [--allow-unavailable] [--from-export DIR]` | substantive runs: full-size agent, open-ended generation, one objective per GPU-lock hold, learner state carried by its checkpoint; `--from-export` is the 16.8 fresh-process child | [help] |
| `$P -m cam_live.eval.oe_runner transfer --writer-root ROOT --plan P --out-dir D [--seeds 0,1,2] [--decisions 20] [--cpu]` | fresh vs experienced in independent processes (16.8) | [help] |
| `$P -m cam_live.eval.v3_report --out-dir D [--results-dir DIR] [--budget 20] [--expect-commit SHA] [--markdown F]` | prespecified summaries of the v3 runs (reads `CAM_LIVE_EVAL_HOME/results/v3` or `--results-dir`) | [help] |
| `$P -m cam_live.eval.meta_checks --root ROOT --result R.json --out O.json [--max-support 16] [--threads 6] [--lock-timeout-s S] [--min-free-gb 12]` | the six 16.5 meta-learning checks on an `oe_runner` run root (GPU lock only for a GPU / real-backbone export) | [help] |
| `$P -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4 --out P.json [--family F --problem X] [--tokens N] [--k K] [--root D] [--allow-unavailable] [--cpu] [--lock-timeout S] [--lock-poll S]` | what the backbone proposer writes (executes nothing; GPU lock) | [help] |
| `$P -m cam_live.eval.cli run --condition E --family F1 --seed 0 --lifetimes 8 --decisions 16 --out R.json [--k K] [--real] [--scale S] [--save-dir D] [--threads N] [--budget-s S] [--device D] [--gpu-lock] [--min-free-gb G] [--lock-timeout-s S]` | one condition on one LabWorld family (diagnostic) | [help] |
| `$P -m cam_live.eval.cli batch --plan {main_cpu,pilot} [--jobs J] [--threads T] [--commit SHA] [--snapshot DIR]` | prespecified CPU batch | [help] |
| `$P -m cam_live.eval.cli transfer-batch --name transfer_cpu [--writer-plan P] [--seeds S] [--frozen] [--jobs J] [--threads T] [--commit SHA] [--snapshot DIR]` | fresh-process transfer / before-after of saved E checkpoints | [help] |
| `$P -m cam_live.eval.cli anchors --family F1 --world-seeds 0-7 --out A.json [--decisions N] [--k K]` | uniform / oracle anchors | [help] |
| `$P -m cam_live.eval.cli analyse [--plan P] [--tag T]` | prespecified analyses → `docs/evidence/S/*.json` | [help] |
| `$P -m cam_live.eval.cli integrity [--seeds 0-4] [--jobs J]` · `meta [--plan P] [--seeds S] [--jobs J]` · `scaling [--scales 5m,20m,50m,150m] [--seeds S] [--lifetimes N] [--device cuda] [--jobs J] [--threads T]` · `large [--scales 500m,1b] [--decisions N] [--min-free-gb G]` | reward integrity battery and trivial world; 16.5 checks on the CPU plan; scaling; large cores (GPU lock) | [help] |
| `$P -m cam_live.eval.report [--tag main_cpu]` | text digest of `docs/evidence/S` | [help] |

The docstrings of `eval/cli.py` and `eval/oe_runner.py` now name only commands that exist (checked against the
parsers; the earlier `oe_runner fresh`, `batch --plan baselines_cpu` and `eval.cli continual | shadow | skills` are
gone from them). `eval/extra.py` (`--one-integrity`, `--one-meta`) is an internal child entry point of `eval.cli`.

Component diagnostics (each writes its own evidence file; the owner's status file records the command used):
`$P -m cam_live.nn.bench [--skip-gpu] [--config C] [--out O] [--lock-timeout S]` (parameter counts and benchmarks;
GPU lock unless `--skip-gpu`), `$P -m cam_live.nn.fast_compare [--steps N] [--seeds …] [--noise-ablation] [--out O]`,
`$P -m cam_live.predict.diagnose [--quick] [--only X] [--update] [--out O]`, `$P -m cam_live.reward.diagnose [--seeds S]
[--lifetimes N] [--gpu-cost] [--out O]`, `$P -m cam_live.reward.evidence --out O [--only {exact,cost}]`,
`$P -m cam_live.credit.diagnose [--only X] [--real] [--out O]`, `$P -m cam_live.harness.grounding.bench [--grounders G]
[--families F] [--lock-budget-s S] [--out O]` — all [help].

---

## 7. Checkpoints and transfer

| command | check |
|---|---|
| `cam-live export-checkpoint [--run-dir R] [--out P] [--checkpoint ckpt_<n>]` — with a live runtime process: a signed operator command to the runtime's own hook; without one (`runtime/cli.py:_export_offline`): the checkpoint is chosen by name among the real directories of `R/checkpoints` (default: the committed one, else the newest), copied with nothing followed into a private staging directory, checked against its MANIFEST, and exported through `credit/checkpoint.py:export_weights_only` → `P/weights.safetensors` + `P/config.json` (+ `P.export.json`). The destination must be outside the run directory and must not exist; the storage guard needs twice the checkpoint size free. Safe to run as root over a directory the learner writes (HX rounds 4–5). | [exec as root on the scratch run: exit 0, `ckpt_00000004` exported, both files written] |
| `cam-live evaluate-transfer --checkpoint EXPERIENCED [--baseline PRE] [--config C] [--domains T1,T2] [--seeds 0-4] [--lifetimes N] [--decisions N] [--k K] [--frozen] [--real] [--threads T] [--jobs J] [--device D] [--out R.json]` (= `$P -m cam_live.eval.transfer …`) — every version × domain × seed in its own fresh process that receives only a weights-only export | [exec, minimal CPU plumbing: the same debug export as both versions, T1, 1 seed, 1 lifetime, 2 decisions, `--jobs 2`: exit 0 in 11–21 s (three runs), `executed: true`] |
| load in a fresh process: `credit.checkpoint.load_exported_model(dest, register_heads=agent.register_agent_heads)` | cited (`credit/checkpoint.py:load_exported_model`) |

---

## 8. Credentials, operator key, live authorization (the operator's acts)

No command below prints a secret; values are never taken from the command line.

```bash
# operator signing key (once; the private key stays with the operator, 0600; the public key 0644)
$P -m cam_live.harness.auth.opctl init-operator-key --private /root/.config/camlive/operator_ed25519 \
    --public /root/.config/camlive/operator_ed25519.pub                                        # [help]

# a credential into a run's vault (R/vault, owned by camauth, 0600) and the verifier's own copy (R/verifier/secrets, camverify, 0600)
$P -m cam_live.harness.credentials set discord_bot_token --run-dir R                     # hidden prompt   [help]
$P -m cam_live.harness.credentials set discord_bot_token --run-dir R --from-file /root/.config/camlive/discord_bot_token
$P -m cam_live.harness.credentials list --run-dir R                                     # names and bindings only
$P -m cam_live.harness.credentials revoke discord_bot_token --run-dir R
#   options: --dest PATTERN (repeatable; default for discord_bot_token: discord.com, gateway.discord.gg, *.discord.gg),
#            --scope S, --verifier / --no-verifier, --note T; --from-file must be a root-only 0600 file
#   with the services running, through the authorization service (value on stdin):
#   $P -m cam_live.harness.auth.opctl --run-dir R put-secret NAME --dest PATTERN [--scope S]

# the signed live authorization record, bound to the task file's hash, its limits and a time window
$P cam_live/tools/authorize_live.py --task cam_live/tasks/discord_100_planning.yaml \
    --private /root/.config/camlive/operator_ed25519 --hours 8 --out R.signed.json \
    [--read 'https://*' ...] [--owned P ...] [--preauthorize P ...] [--account A ...] [--issued-by NAME] [--note T]  # [help]
```

`tools/authorize_live.py:main` prints what is authorized and asks the operator to **type the objective id**
(`discord-100-planning` for the planning task) before it signs; a wrong answer exits 4 "not signed"; `--yes` skips the
question and is not to be used on the operator's behalf (signing is the operator's act). It refuses a task that does not load for a run or is not `mode: live`
(exit 3). Discord destinations of the task (`account:discord:<guild>`, `channel:discord:*`) are added to `owned`
automatically; `--read` defaults to the owned destinations (`tools/authorize_live.py:build_record`). The record id is
`live-<objective_id>-<unix time>`; the task field `resources.operator_authorization_record` is validated as a slug only
(`runtime/tasks.py`) and is not compared with that id.

The authorization service verifies the record against `R/auth/keys/operator.pub`. `configs/cam_live.yaml` has no
`launcher.operator_pubkey`, so the public key must be placed there before `cam-live run` (`tools/launch_live.sh` does
it, §9.1) or named in a configuration section `launcher: {operator_pubkey: <path>}`; otherwise the supervisor's
registration of the record fails and nothing starts. Also available: `opctl sign-live record.json --private K >
signed.json`, `opctl --run-dir R register-live signed.json` [help of `opctl`].

---

## 9. Launch

### 9.1 Live, planning phase (the operator, as root, in an interactive terminal)

The procedure as it exists at `fdab7d3` (`tools/launch_live.sh`; ledger `docs/CAM_LIVE_LEDGER.md`, session 4):

```bash
cd /workspace/camlive
# optional first: the launch-path check (one decision, two external actions, 30 min)
bash cam_live/tools/launch_live.sh --accept-ephemeral-storage cam_live/tasks/discord_launch_check.yaml 1 \
     --read 'https://*' --read 'http://*'
# the planning phase (8-hour authorization window; the task's own limits still bound the run)
bash cam_live/tools/launch_live.sh --accept-ephemeral-storage cam_live/tasks/discord_100_planning.yaml 8 \
     --read 'https://*' --read 'http://*'
```

`tools/launch_live.sh [--accept-ephemeral-storage] TASK_FILE HOURS [authorize_live.py options: --read / --owned /
--preauthorize / --account / --note]`:

* `--accept-ephemeral-storage` must be the **first** argument (`tools/launch_live.sh:31`). It is passed to
  `cam-live run`, where it moves only the doctor's `persistent_storage` item to the preflight's accepted risks
  (`runtime/preflight.py:ACCEPTABLE_RISKS`, live mode only; recorded in `R/run.json`, `R/preflight.json` and shown by
  `cam-live status`). It cannot come from a task file or the environment. Without it the planning launch is blocked on
  this machine (no persistent volume). Run data is then lost with the machine: copy `cam-live plan-report --out F`
  and an export off the box (§12).
* `HOURS`: length of the authorization window.
* Environment: `CAM_LIVE_RUN_DIR` (default `<repo>/runs/cam_live/live/<objective>-<UTC time>`), `CAM_LIVE_OPERATOR_KEY`
  (default `$CAM_LIVE_CONF/operator_ed25519`), `CAM_LIVE_CONF` (default `/root/.config/camlive`; holds
  `discord_bot_token` and `operator_ed25519.pub`). `CAM_LIVE_GPU_LOCK` is not set by the script: on this machine the
  default lock path is the shared lock (§1.4); elsewhere export it first. `CAM_LIVE_HOME` is not set either: the
  state root is `<repo>/runs/cam_live` (`current_run`, dashboard tokens there).
* It refuses a non-interactive terminal (exit 2), a run directory that already holds a run (exit 3), and a task that
  does not load for a run.

What it executes (lines 40–50), equivalent by hand [not run: live run, the operator's signature, real credentials]:

```bash
cd /workspace/camlive
REPO=/workspace/camlive; CONF=${CAM_LIVE_CONF:-/root/.config/camlive}; PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO/cam_live/src
TASK=$(readlink -f cam_live/tasks/discord_100_planning.yaml)
OBJ=$($PY -c "import sys; from cam_live.runtime.tasks import load_task; print(load_task(sys.argv[1], for_run=True).objective_id)" "$TASK")
R=$REPO/runs/cam_live/live/$OBJ-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$R"
$PY -m cam_live.harness.credentials set discord_bot_token --run-dir "$R" --from-file "$CONF/discord_bot_token"
mkdir -p "$R/auth/keys"                                                   # the key directory does not exist yet
install -m 0644 "$CONF/operator_ed25519.pub" "$R/auth/keys/operator.pub"
$PY $REPO/cam_live/tools/authorize_live.py --task "$TASK" --private "$CONF/operator_ed25519" --hours 8 \
    --out "$R.signed.json" --read 'https://*' --read 'http://*'           # the operator types the objective id
$PY -m cam_live.runtime.cli run --task "$TASK" --authorization "$R.signed.json" --run-dir "$R" --accept-ephemeral-storage
```

[`bash cam_live/tools/launch_live.sh` without arguments: exec, prints its header, exit 2.]

`cam-live run --task T [--run-dir R | --run-id ID] [--authorization F] [--accept-ephemeral-storage] [--config C]
[--agent-factory M:F] [--verifier-source S] [--no-doctor] [--preflight-only] [--inproc] [--foreground]
[--max-iterations N] [--start-timeout 120] [--json]` [help]. As root with the service users present and without
`--inproc` it is the separated mode: a detached supervisor (`runtime/cli.py:cmd_supervise`) starts the launcher's
services in the order authorization (camauth), verifier (camverify), sandbox host (root; runs generated code as
camsbx), executor / computer environment (cambrowser), learner (camlearn) (`harness/auth/launcher.py:START_ORDER`,
each started through `harness/service_boot.py` in isolated Python), registers the task and the record, runs the
preflight against the running services, writes `R/learner.go` only if every live item is confirmed, and otherwise
prints the missing items to `R/logs/supervisor.log`, stops the services and starts no learner. A live task is
refused without the separated services or without `--authorization` (exit 3). `<state root>/current_run`
records R.

**Start timeout** (code reading, not run): `cam-live run` waits at most `--start-timeout` seconds (default 120;
`tools/launch_live.sh` does not change it) for the learner's `R/state.json`, and the learner writes it only after it
holds the shared GPU lock and has built the agent (`runtime/loop_service.py:run_service`; the separated rehearsal's
learner waited 849 s for the lock, §5). When that takes longer, `cam-live run` exits 5 ("the runtime did not start",
with the last log lines) although the detached supervisor and the learner keep starting — and `launch_live.sh` stops
there (`set -e`) without its closing hint. Check `cam-live status --run-dir R` and `R/logs/supervisor.log` before
anything else; do not launch a second time. `tools/planning_rehearsal.py holds` passes `--start-timeout 2400` for
this reason; in the by-hand sequence the same option can be added to the last command.

Afterwards: `cam-live status --run-dir R`, `cam-live dashboard --run-dir R`, `cam-live plan-report --run-dir R --out
<root-only dir>/plan.md`, `$P -m cam_live.harness.operator_requests --run-dir R list`, `cam-live stop --run-dir R` (§10).

Other live tools: `tasks/discord_100.yaml` (execution phase) is refused until its 16 REQUIRED values are filled
(§4). Read-only Discord adapter check through the real environment: `$P cam_live/tools/discord_live_check.py --run-dir R
--task T [--policy P] [--skip-writes] [--print-record-template] [--print-policy-rule] [--write-policy FILE]
[--await-reply-s S] [--approve-wait-s S] [--cleanup-wait-s 900] [--listen-s 1..60] [--out F]`, or locally `--staging
[--live-policy]` [help; not run: live service]. Read-only live observation script (prepared, never executed):
`$P cam_live/docs/evidence/validation/discord_live/run_live_observe.py --out F [--decisions N]` [help].

### 9.2 Staging / diagnostic runs of the runtime

```bash
cam-live run --task cam_live/tasks/lab_staging.yaml --inproc [--foreground --max-iterations N] [--config C] [--run-dir R]
```

`--inproc` is single-process (never for live tasks). With the default configuration the runtime leases the shared GPU
lock (§1.4). [exec: `--inproc --foreground --max-iterations 4 --config <debug configuration of §2.5> --run-dir
<scratch>/run_lab --no-doctor`, GPU hidden and the lock path absent: exit 0 in 62–142 s (three runs; machine load), 4 decisions, checkpoints
`ckpt_00000003`, `ckpt_00000004`, "learner: run ended with status running".]

Lower-level entry points (the supervisor and the launcher start them; by hand for diagnosis only), all [help]:
`$P -m cam_live.runtime.loop_service --run-dir R [--task T] [--config C] [--agent-factory M:F] [--inproc]
[--wait-for F] [--wait-timeout S] [--max-iterations N] [--socket S]`, `$P -m cam_live.runtime.verifier_service
--run-dir R [--task T] [--source S] [--socket S]`, `$P -m cam_live.harness.service --run-dir R --socket S
[--auth-socket A] [--provenance {staging,live,diagnostic}] [--task T] [--staging-sites] [--no-browser] …`,
`$P -m cam_live.harness.sandbox_host --run-dir R --socket S [--client-uid U] [--workspace W]`, `$P -m
cam_live.harness.auth.service --run-dir R --socket S [--policy P] [--audit-anchor A] …`, `$P -m
cam_live.harness.auth.launcher --run-dir R [--services auth,exec,verifier,learner] [--policy P] [--operator-pubkey K]
[--no-separation]`.

---

## 10. Operation: status, control, approvals, stop, resume, report, dashboard

`--run-dir R` defaults to the run started last from the state root. Commands go through the signed operator command
channel (`runtime/control.py:OperatorControl`); a command to a run whose process is gone is queued and applied when
it starts.

| command | check |
|---|---|
| `cam-live status [--run-dir R] [--json]` — objective, verified value, activity (incl. `waiting_for_gpu_lock`), resources, learning, checkpoint, pending items, GPU lock, accepted risks | [exec on the scratch run: exit 0, "status: running (process not running)", 4 decisions, checkpoint `ckpt_00000004`; no run → exit 1] |
| `cam-live pause [--reason T]` · `cam-live stop [--reason T]` · `cam-live stop --emergency` | [help] |
| `cam-live resume [--run-dir R] [--accept-ephemeral-storage]` — paused: resume; process gone: restart from the persisted state (separated: through a new supervisor); a stopped run is refused (exit 3). A live run started with `--accept-ephemeral-storage` needs it again on `resume`, otherwise the restarted preflight blocks (`runtime/cli.py:cmd_resume`) | [help] |
| `cam-live approve ID [--note T]` · `cam-live reject ID [--note T]` | [help] |
| `cam-live journal [--print]` — regenerate `R/journal.md` | [exec: exit 0] |
| `cam-live plan-report [--run-dir R] [--out F] [--json]` — CAM's objective, notes, predictions, experiments, operator requests, actions and learning updates, verbatim with provenance, every text through the run logs' redaction (`runtime/plan_report.py:write_report`); default `R/plan_report.md` (`.json` with `--json`); refused below the storage minimum. Safe to run as root (HX). | [exec as root on the scratch run: exit 0, Markdown report written] |
| `$P -m cam_live.harness.auth.opctl --run-dir R approvals \| approve ID \| reject ID \| stop "reason" \| reset-stop \| pause "reason" \| resume \| revoke SCOPE \| reload-policy [P] \| limits \| secrets \| audit-tail [n] \| verify-audit \| put-secret NAME \| register-live F \| local-service add HOST:PORT \| local-service remove HOST:PORT \| local-service list` — `stop` and `verify-audit` also work with the service down | [help] |
| `$P -m cam_live.harness.operator_requests --run-dir R list [--all] \| grant ID --note T \| deny ID --note T \| answer ID --note T` — CAM's requests to the operator (an answer grants nothing by itself) | [help] |

**Dashboard** (127.0.0.1 only; FastAPI):

```bash
cam-live dashboard [--run-dir R] [--port 8787] [--view-only]
# viewing token: <state root>/dashboard.token  (http://127.0.0.1:8787/?token=<token> or Authorization: Bearer)
# operator actions: credential <state root>/dashboard.operator (0600), not loaded with --view-only
# state root (section 2.5): $CAM_LIVE_HOME, else <repo>/runs/cam_live (tools/launch_live.sh does not set CAM_LIVE_HOME)
ssh -p $VAST_TCP_PORT_22 -L 8787:127.0.0.1:8787 root@$PUBLIC_IPADDR     # from the operator's machine
```

[exec `--view-only` on the scratch run, port 18787: no token → 401, `?token=` → 200, `Bearer` on `/api/status` → 200;
stopped afterwards.] Its health panel shows free disk with the guard level (ok / warning / low against the run's
`min_free_gb` / `warn_free_gb`).

---

## 11. Retention and pruning

* Automatic: `storage.py:DiskGuard` before every runtime checkpoint and every learner save / promotion
  (`storage.min_free_gb` 20: below it nothing is written and the run pauses with a resource obstacle;
  `warn_free_gb` 50: warning); the runtime keeps `runtime.keep_checkpoints: 2` (optimizer state only in the newest;
  `runtime/loop.py:RuntimeConfig.keep_checkpoints`); `eval/oe_runner.py` applies `storage.py:apply_retention` after
  every objective. Exports and plan reports check the guard too (§7, §10).
* Manual, finished run trees (dry run by default; records are always kept, only checkpoint directories are touched,
  symlinked checkpoints are never followed):

  ```bash
  $P cam_live/tools/prune_runs.py runs/cam_live/eval                                     # report only
  $P cam_live/tools/prune_runs.py runs/cam_live/eval --apply --keep-optim 0 --drop-protected-optim [--keep 1] \
      [--frozen-glob 'A_*' ...] [--drop-group '*/lt[0-9][0-9]/checkpoints' ...] [--report F.json]
  ```

  [exec: dry run on the scratch state root and on the scratch run: `DRY RUN: 0 removals, 0.0 GiB`; `--apply` not run.]

---

## 12. Off-box copy, completion checker, secret scan, push

* **Run data and checkpoints have no off-box copy** (no persistent volume). Only git is pushed; anything to keep from
  `runs/` must be exported (`cam-live export-checkpoint`, `cam-live plan-report`) and copied off the machine by the
  operator (e.g. `rsync` / `rclone` to their storage) [not run: needs an operator destination].
* Completion checker (read-only without `--merge-status`): `cam-live verify-completion [--only IDS] [--json F]
  [--completion-md F] [--quiet]` (= `$P -m cam_live.verify_completion`) [exec `--only IN-05,CM-13,DL-02 --quiet`:
  exit 1 = level 1 not reached, by design; first run (07:2x UTC) "partial: 3", later runs (07:33–07:42 UTC)
  "implemented-needs-reverification: 3" — the categories follow the current documents and verdicts, which other
  commits changed in between].
  `--merge-status` rewrites the manifest and `docs/COMPLETION.md` (MAN only) [help; not run]. Manifest regeneration
  and verdict merge (MAN / G): `cd cam_live && ../.venv/bin/python tools/gen_manifest.py`, `.venv/bin/python
  cam_live/tools/merge_verdicts.py` [not run: neither has `--help`; both write their files when started].
* Secret scan of a push range: `python3 cam_live/tools/secret_scan.py BASE SHA` — every blob any commit of
  `BASE..SHA` adds or modifies, raw bytes, gzip / xz / bzip2 / zip expanded up to three levels, each blob judged at
  every path it occurs at; (1) the real credentials of this machine (exact match, read from root-only files, never
  printed), (2) credential-shaped patterns outside `tools/secret_scan_allow.txt`. Exit 0 clean, 4 a real credential,
  3 a pattern hit, 2 usage / git error; output names paths and commit ids only (`tools/secret_scan.py:main`). It has no
  argparse: `--help` prints the usage line and exits 2 [exec]. [exec, pattern part only:
  `tools/secret_scan.py:main(['fdab7d3~3', 'fdab7d3'])` in-process with `tools/secret_scan.py:real_secrets` replaced by
  an empty list: exit 0, "19 blobs of fdab7d3~3..fdab7d3 scanned". The full scan was not run: its real-credential
  part reads `/root/.config/camlive` and `/root/.config/agentrs`, which this check does not read. The scan itself
  uses only `git log` / `git cat-file` (offline, read-only).]
* Push (operator-authorized milestones only): `bash cam_live/tools/push_milestone.sh` — on `feat/cam-live` only,
  pushes `HEAD` by explicit sha, never forces, runs `tools/secret_scan.py` over remote-head..HEAD first and aborts on
  any hit; credential helper `$CAM_LIVE_CRED_HELPER` (default `/root/.config/camlive/gh_cred_helper.sh`). [not run:
  pushing is the master's act at milestones.]

---

## 13. Inventory of entry points at `fdab7d3`

Found by searching `pyproject.toml` (`[project.scripts]`), `runtime/cli.py:build_parser` and every `if __name__ ==
"__main__"` block under `cam_live/src` and `cam_live/tools`. All are documented above; marks in the sections.

| kind | entry points |
|---|---|
| console script | `cam-live` with `doctor`, `init`, `test`, `validate`, `inspect-harness`, `preflight`, `run`, `status`, `pause`, `resume`, `stop`, `approve`, `reject`, `export-checkpoint`, `journal`, `plan-report`, `dashboard`, `evaluate-transfer`, `verify-completion` (all [help]); `_supervise` is internal (started by `run` / `resume`) |
| runtime / services (`python -m`) | `cam_live.runtime.cli`, `runtime.loop_service`, `runtime.verifier_service`, `harness.service`, `harness.sandbox_host`, `harness.auth.service`, `harness.auth.launcher` (all [help]) |
| operator (`python -m`) | `harness.credentials`, `harness.auth.opctl`, `harness.operator_requests`, `harness.browser`, `harness.staging` (all [help]) |
| evaluation (`python -m`) | `eval.oe_batch`, `eval.oe_runner` (`sequence`, `transfer`), `eval.v3_report`, `eval.meta_checks`, `eval.proposal_probe`, `eval.cli` (`run`, `batch`, `transfer-batch`, `anchors`, `analyse`, `integrity`, `meta`, `scaling`, `large`), `eval.transfer`, `eval.report`, `verify_completion` (all [help]) |
| diagnostics (`python -m`) | `nn.bench`, `nn.fast_compare`, `predict.diagnose`, `reward.diagnose`, `reward.evidence`, `credit.diagnose`, `harness.grounding.bench` (all [help]) |
| internal children (not for the operator) | `harness.grounding.worker` (a grounder process; started by `IsolatedGrounder`), `harness/service_boot.py` (isolated start of every separated service), `eval/extra.py` (`--one-integrity` / `--one-meta` children of `eval.cli`) [not run: they expect their parent's protocol] |
| tools (`cam_live/tools/`) | `launch_live.sh` [exec: usage, exit 2], `authorize_live.py` [help], `planning_rehearsal.py` (`prepare`, `serve`, `holds`, `check`) [help], `discord_live_check.py` [help], `prune_runs.py` [help, exec dry run], `secret_scan.py` [exec], `push_milestone.sh` [not run], `gen_manifest.py`, `merge_verdicts.py` [not run] |
| evidence scripts named here | `docs/evidence/validation/separated/run_separated.py`, `docs/evidence/validation/discord_live/run_live_observe.py` [help]. The other scripts under `docs/evidence/<workstream>/` reproduce that workstream's recorded evidence and are documented in its status file. |

---

## 14. Check of this document by probe d03

`PYTHONPATH=/workspace/camlive/cam_live/src .venv/bin/python
cam_live/docs/verification/repro/g2integration/docs/d03_reproduce_commands.py /workspace/camlive <empty scratch dir>`
encodes the claims of the previous version of this document (`d3ce519` / `55a0f4c`, §13/§14 addendum `5ae3b59`) and
tests whether they still hold at the code. [exec 2026-11-28 07:32–07:34 UTC on `fdab7d3`, empty scratch directory:
87 commands, 81 HOLDS, 7 BROKEN; "help summary: 66/66 exit 0"; exit 0.] Every BROKEN line is a claim of the previous
version that is false of the code, and this version does not make it:

| d03 line | old claim the probe encodes | what is true at `fdab7d3` (this document) |
|---|---|---|
| `doctor` | "48/51 checks ok" | 52 checks; 49/52 in this machine's environment (§1.1). The probe hides the GPU and points `CAM_LIVE_GPU_LOCK` at an absent file, so it sees 47/52 (`gpu`, `gpu_lock` fail in addition) — the fail-closed lock of §1.4 |
| `preflight_staging_planning` | staging planning task: exit 3, 9 live items | exit 0 (a staging task whose staging readiness holds); LIVE lists 10 items with the doctor, 9 with `--no-doctor` as the probe runs it (§4) |
| `docstrings_name_missing_commands` | the eval docstrings name commands that do not exist | they no longer do (§6); the five names are still not commands (the probe's five "absent" lines hold) |
| `collect_runtime` | `tests/runtime`: 215 tests | 299 (§3) |
| `collect_storage` | `tests/storage`: 3 tests | 4 (§3) |
| `collect_all_fails` | the whole suite cannot be collected in one process (exit 2) | 1679 collected, exit 0 (§3) |
| `cam-live_test_storage` | `cam-live test --suite storage`: 3 passed | 4 passed (§3) |
