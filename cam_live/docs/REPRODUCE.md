# CAM Live 1.0 — dependencies and exact commands (IN-05, CM-13)

One place for everything needed to set up, test, validate, evaluate, authorize, launch, operate, back up and verify
CAM Live. Commands were checked on this machine at commit **`d3ce519`** (branch `feat/cam-live`, 2026-11-27
23:15–23:40 UTC); `tools/launch_live.sh` and `tasks/discord_launch_check.yaml` were added afterwards (`d086a55`,
`55a0f4c`) and were checked separately at `55a0f4c`. The record of every check (exit code, first output line) is
`docs/evidence/DOC/command_check.txt`.

Each command carries a mark:

* **[help]** — `--help` ran with exit 0 (argparse parses `--help` before doing any work);
* **[exec]** — executed here, offline, with `CAM_LIVE_HOME` in a scratch directory; the result is stated;
* **[not run: …]** — not executed, with the reason (network, live service, root services, writes owned by another
  workstream, or the operator's own act).

All commands run from the worktree root `/workspace/camlive` unless stated otherwise. `P=.venv/bin/python`.
`cam-live` is `.venv/bin/cam-live` (console script `cam_live.runtime.cli:main`, also `$P -m cam_live.runtime.cli`).
Exit codes of `cam-live`: 0 ok, 1 the command ran and reports a failure, 2 usage, 3 refused (preflight /
authorization / task validation), 4 a needed component is not installed, 5 the runtime did not start.

---

## 1. Dependencies

### 1.1 Hardware and host (measured by `cam-live doctor` [exec] and `nvidia-smi`)

| item | this machine | needed for |
|---|---|---|
| GPU | 1x NVIDIA RTX A6000, 47.4 GB, driver 595.104.02; **shared** with another tenant (heavy jobs take `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock`) | real backbone (Qwen3-4B bf16 + 50m core): measured peaks 10.4 GB (meta pathway) to 18.6 GB (staging validation, INTEGRATION.md §7.3, §7.6) |
| CPU / RAM | 48 logical CPUs, 126 GB RAM | CPU test suite, debug-scale runs |
| disk | 300 GB overlay, 168 GB free at check time | runs + checkpoints; `storage.min_free_gb` 20 / `warn_free_gb` 50 (configs/cam_live.yaml) |
| persistent volume | **none** (`workspace_is_volume=False`; doctor FAIL `persistent_storage`, blocking for live) | live runs need durable state or an off-box copy (§12) |
| isolation | root, setuid/setgid, Landlock ABI 1, seccomp; **no** CAP_NET_ADMIN, **no** user namespaces | separated services under 5 uids; per-destination network filtering is policy-level only (doctor FAIL `kernel_network_filtering`: a VM is recommended for live operation) |
| service users | `camlearn` 999, `cambrowser` 997, `camsbx` 996, `camauth` 995, `camverify` 994 | separated mode; created by the launcher (`harness/auth/launcher.py:_ensure_users`) on the first `cam-live run` as root |

### 1.2 Python packages

`cam_live/pyproject.toml`: `requires-python >= 3.11`; runtime dependencies torch>=2.4, transformers>=4.51,
safetensors, numpy, pyyaml>=6, httpx>=0.27, fastapi>=0.115, uvicorn>=0.30, playwright>=1.48, pillow>=10,
cryptography>=42, beautifulsoup4>=4.12, pypdf>=4, websockets>=12; extra `dev` = pytest>=8, pytest-timeout>=2, scipy.
Installed versions (measured with `importlib.metadata` in `.venv/bin/python`; "site" = where the package is loaded from):

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

`uv` 0.13.4 is at `/usr/local/bin/uv`. Nothing under `cam_live/` imports `superlearner`, `crucible` or `agentrs`; the
agentrs venv is used only as a package directory.

### 1.3 Models (pinned)

| model | repository @ revision | licence | size | where |
|---|---|---|---|---|
| backbone (frozen) | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` (`config.py:BackboneConfig`, `configs/cam_live.yaml`) | Apache-2.0 | 7.6 GB (3 safetensors shards) | `$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a72…` |
| visual grounder, lightweight | `HongxinLi/GoClick-Base` @ `05633a495f9d2ce823372d0c7fd9b0b4bb67c6fa` | MIT | — | `$HF_HOME/hub` |
| visual grounder, advanced | `Tongyi-MAI/MAI-UI-2B` @ `503050934809558c8dfd2ddedaf9621fa74ac2de` | Apache-2.0 | — | `$HF_HOME/hub` |
| visual grounder (baseline only) | `vocaela/Vocaela-2-500M-1024R2` @ `b0280f2ce8e2f126a1c3c446c70e63ee7bbeaed2` | CC BY-NC-SA 4.0 (non-commercial) | — | `$HF_HOME/hub` |

`HF_HOME` is `/workspace/.hf_home` (the default set by `nn/backbone.py:HFBackbone.__init__` and `tests/conftest.py`
when unset). The backbone loads with `local_files_only: true` and sets `HF_HUB_OFFLINE=1`, so nothing is downloaded
at run time. Doctor [exec]: `backbone: Qwen/Qwen3-4B@1cfa9a720891: 3 weight file(s), 7.5 GB` and all three grounders
`available: true`.

---

## 2. Environment setup

### 2.1 As built on this machine (layered venv)

Reconstructed from `.venv/pyvenv.cfg` and the `.pth` files present (the creation itself was not re-executed):

```bash
cd /workspace/camlive
/workspace/agentrs/.venv/bin/python -m venv --without-pip /workspace/camlive/.venv     # recorded in .venv/pyvenv.cfg
echo /workspace/agentrs/.venv/lib/python3.12/site-packages \
    > .venv/lib/python3.12/site-packages/_shared_base.pth                                # base packages (torch, transformers, ...)
uv pip install --python .venv/bin/python -e 'cam_live[dev]'                              # editable cam_live + numpy/scipy/pytest/pypdf
.venv/bin/python -m playwright install chromium                                          # -> ~/.cache/ms-playwright/chromium-1255
```

Extra packages go only into this venv: `uv pip install --python /workspace/camlive/.venv/bin/python <pkg>`
(CAM_LIVE_ARCHITECTURE.md §8). [not run: the environment exists; `uv pip install` and `playwright install` need the
network.]

### 2.2 On a fresh machine (no agentrs venv)

Same versions as measured above, without the layering [not run: needs the network]:

```bash
git clone <repository> camlive && cd camlive && git checkout feat/cam-live
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv/bin/python -e 'cam_live[dev]' transformers==5.24.0 safetensors==0.8.0 \
    huggingface_hub==2.4.0 numpy==2.5.5 fastapi==0.165.0 uvicorn==0.54.0 playwright==1.65.2 pillow==13.0.0 \
    cryptography==50.0.2 beautifulsoup4==4.15.0 pypdf==7.2.0 websockets==17.4 httpx==0.28.1 pyyaml==6.0.4 \
    scipy==1.18.1 pytest==9.2.0 pytest-timeout==2.4.0
.venv/bin/python -m playwright install --with-deps chromium
```

The GPU must support the wheel's CUDA (cu132 here; on Blackwell use a CUDA >= 12.8 build).

### 2.3 Chromium for the browser uid (separated mode, once, as root)

```bash
.venv/bin/python -m cam_live.harness.browser --install-shared      # copy this user's Playwright Chromium to
                                                                   # .venv/share/cam_live/ms-playwright   [help]
.venv/bin/python -m cam_live.harness.browser --find cambrowser     # [exec] -> .venv/share/cam_live/ms-playwright/
                                                                   #    chromium-1255/chrome-linux64/chrome
```

### 2.4 Model download

```bash
export HF_HOME=/workspace/.hf_home
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
d('Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c')"
# grounders (only for visual computer use): same call with the repositories and revisions of §1.3
```

[not run: the snapshots are present (doctor [exec] ok); downloading needs the network.] `Qwen/Qwen3-8B` is a
configuration change (`backbone.name`, `backbone.revision`).

### 2.5 Configuration and state root

* `cam_live/configs/cam_live.yaml` — the full candidate (Qwen3-4B + `50m` core, `device: auto`); every `cam-live`
  command takes `--config FILE` (a complete replacement, not an overlay). Sections other than
  seed/device/home/backbone/core/optim are read by their owners (`storage`, `runtime`, `agent`, `credit`, `launcher`).
* State root: `home:` in the config, else `$CAM_LIVE_HOME`, else `<repo>/runs/cam_live` (`config.py:resolved_home`).
  Run data is never in git.
* `cam-live init --home H` [help] — creates `H/{runs,checkpoints,exports,validation}`, `config.resolved.json`, the
  dashboard token, and (without `--no-agent`) builds the agent and saves `H/checkpoints/init` (takes the GPU lock
  for the real backbone). `cam-live init --no-agent --home H` [exec: exit 0, state root and token created].

---

## 3. Tests

```bash
cd /workspace/camlive
.venv/bin/python -m pytest cam_live/tests/<dir> -q --rootdir=cam_live --junitxml=cam_live/docs/evidence/test_results/<id>.xml
```

`<dir>` ∈ `credit`, `diagnostics`, `eval`, `harness` (subdirectories `auth`, `compute`, `env`, `skills`),
`integration`, `nn`, `predict`, `reward`, `runtime`, `storage`, and the file `test_verify_completion.py`.
[exec: `--collect-only -q` of `runtime` (215 tests) and `storage` (3); `storage` run with JUnit: 3 passed.]

* **Whole default suite: one pytest process per test directory.** Collecting `cam_live/tests` in one process fails
  (INT2-5: `tests/predict/test_providers.py` and `tests/credit/test_providers.py` share a basename) — [exec at
  `d3ce519`: `1459 tests collected, 1 error`, exit 2]. Run the directories separately (in parallel if wanted):

  ```bash
  for d in predict reward credit eval integration storage diagnostics nn runtime harness test_verify_completion.py; do
    .venv/bin/python -m pytest cam_live/tests/$d -q --rootdir=cam_live --junitxml=/tmp/junit_${d%.py}.xml &
  done; wait
  ```

  Last recorded whole-suite result (this procedure, frozen copy of commit `1039a6f`): **1468 tests, 1378 passed,
  90 skipped, 0 failed, 0 errors**, 442 s wall (load-dependent), JUnit `docs/evidence/test_results/INT2_full.xml`,
  per-directory times in `docs/status/M-int.json` (`measured.full_suite`). Not re-run here.
* `cam-live test --suite <dir> [--suite <dir> …] [-m MARK] [-k EXPR] [--results-dir D]` runs one pytest process per
  `--suite` and writes `D/cli_<dir>.xml` (default `cam_live/docs/evidence/test_results/`) [exec: `--suite storage
  --results-dir <scratch>`: exit 0, 3 passed]. **Without `--suite` it runs the whole tree in one process and hits
  the INT2-5 collection error.**
* Opt-in marks (`tests/conftest.py:pytest_collection_modifyitems`; default runs are CPU-only, offline):
  `-m gpu` (real Qwen3-4B; the test takes the shared GPU lock itself), `-m slow`, `-m browser` (Playwright Chromium),
  `-m privileged` (root; separated service uids). **Never `-m network`.** Never wrap pytest in `flock` on the GPU
  lock file (the tests take it and would deadlock). Per-test timeout 900 s (`pyproject.toml`; the runtime
  integration module sets 2400 s).
* Independent verification probes: `.venv/bin/python cam_live/docs/verification/repro/<lens>/<probe>.py`
  (lenses `cl`, `env`, `g2learn`, `nn`, `pr`, `rt`, `sec`; documented per lens in `docs/verification/G_<lens>.md`)
  [not run: owned by G, cited only].

---

## 4. Health, harness inventory, preflight

| command | what | check |
|---|---|---|
| `cam-live doctor [--json] [--strict]` | hardware, storage, models, dependencies, sandbox (Landlock, seccomp, setuid), service users, harness modules, GPU lock | [exec: exit 0; 48/51 ok; failed `persistent_storage` (blocking for live), `kernel_network_filtering`, `user_namespaces`; nothing blocking for staging; `--strict` exit 0] |
| `cam-live inspect-harness [--task T] [--run-dir R] [--lab] [--no-browser]` | tools, permissions, missing integrations (in-process inspection environment unless `--run-dir` of running services) | [exec `--lab`: exit 0, LabWorld tools listed; missing integration: Discord (no executor / credential)] |
| `cam-live preflight --task T [--no-doctor] [--json]` | the readiness report alone (offline: services not running → unconfirmed items) | [exec: `tasks/discord_100_planning.yaml` exit 3, LIVE BLOCKED 9 items (services not started, credential and Discord checks unconfirmed, persistent storage), STAGING ready; `tasks/discord_100.yaml` exit 3, 25 items incl. its 16 values marked REQUIRED (5 of them limits); `tasks/lab_staging.yaml` exit 0] |
| `cam-live run --task T --preflight-only` | the same report through `run` | [help] |

---

## 5. Staging validation (controlled services only)

```bash
cam-live validate --mode staging --real-model [--decisions 40] [--seed 0] [--out D] [--evidence-dir E]   # [help]
```

Real configuration (Qwen3-4B + 50m, takes the shared GPU lock inside `validation.py:run_staging`, needs ≥ 18 GB
free GPU memory): last executed run 630 s under the lock, 1030 s wall, peak 18.6 GB, **executed, not passed**
(`docs/evidence/validation/staging_20261127T223510Z.json`, INTEGRATION.md §7.6). [not run here: GPU lock budget.]
Report: `D/staging_report.json`; the evidence file `staging_<UTC>.json` goes to `D`, or to `E` with `--evidence-dir`. `--mode live` is refused
(exit 2): live readiness comes only from `preflight` / `run`.

**CPU plumbing check — use the library call, not `--debug`.** [exec at `d3ce519`] `cam-live validate --mode
staging --debug` passes the loaded default configuration to `run_staging`, and `validation.py:staging_config` keeps
a given configuration, so `--debug` builds **Qwen3-4B + 50m on CUDA without the GPU lock** (measured:
`staging_config(_cfg(''), False, 0)` → `Qwen/Qwen3-4B 50m cuda`; `staging_config(None, False, 0)` → `tiny debug
cpu`). The equivalent that really uses the tiny CPU configuration:

```bash
CAM_LIVE_HOME=<state root> .venv/bin/python -c "from cam_live.validation import run_staging; \
r = run_staging(None, real_model=False, n_decisions=12, seed=0, out_dir='<dir>'); print(r['ok'])"
```

[exec: 80 s, executed, `ok: false` — 14 of 16 checks hold; `approvals_exercised` false (1 approval resolved, 0
actions executed after approval) and `resource_limit_exercised` false (the decision budget ran out first) at
`n_decisions=12`. A diagnostic of the plumbing, not evidence about learning.]

**Separated-services staging rehearsal** (root; the launcher starts authorization, executor, verifier and learner
under their uids; the script approves / rejects as the operator and stops the run):

```bash
.venv/bin/python cam_live/docs/evidence/validation/separated/run_separated.py --out X.json [--decisions N] [--base-port P] [--keep]
```

[help; not run: starts the services as root.] Last result: `docs/evidence/validation/separated/
separated_20261127T183623Z.json` (INTEGRATION.md §7.1). Local staging sites alone:
`.venv/bin/python -m cam_live.harness.staging [--base-port P]` [help].

---

## 6. Evaluation

All write JSON with `"executed": true` only after the experiment ran. GPU runs take the shared lock per hold
(≤ ~15 min). LabWorld / tiny-backbone runs are engineering diagnostics only (docs/S/PROTOCOLS.md, Protocol v3).

| command | purpose | check |
|---|---|---|
| `$P -m cam_live.eval.oe_runner sequence --condition E --seed 0 --plan SW:rle,MECH:7 --decisions 20 --out R.json [--resume] [--save-exports] [--hold-s 870] [--min-free-gb 14] [--no-skills] [--from-export DIR] [--cpu]` | substantive runs: full-size agent, open-ended generation, one objective per GPU-lock hold, learner state carried by its checkpoint | [help] |
| `$P -m cam_live.eval.oe_runner transfer --writer-root ROOT --plan P --seeds 0,1,2 --out-dir D [--cpu]` | fresh vs experienced in independent processes (16.8) | [help] |
| `$P -m cam_live.eval.proposal_probe --family SW --problem rle --rounds 4 --out P.json [--tokens N]` | what the backbone proposer writes (executes nothing; GPU lock) | [help] |
| `$P -m cam_live.eval.cli run --condition E --family F1 --seed 0 --lifetimes 8 --decisions 16 --out R.json [--real] [--scale S] [--gpu-lock] [--budget-s S]` | one condition on one family | [help] |
| `$P -m cam_live.eval.cli batch --plan {main_cpu,pilot} [--jobs 12] [--threads 3] [--commit SHA]` | prespecified CPU batch (`main_cpu` in PROTOCOLS.md) | [help] |
| `$P -m cam_live.eval.cli transfer-batch --name transfer_cpu [--frozen] [--writer-plan P] [--seeds S]` | fresh-process transfer / before-after of saved E checkpoints | [help] |
| `$P -m cam_live.eval.cli anchors --family F1 --world-seeds 0-7 --out A.json` | uniform / oracle anchors | [help] |
| `$P -m cam_live.eval.cli analyse [--plan P] [--tag T]` | prespecified analyses → `docs/evidence/S/*.json` (`CAM_LIVE_EVAL_EVIDENCE` overrides the directory) | [help] |
| `$P -m cam_live.eval.cli integrity --seeds 0-4` · `meta --plan main_cpu` · `scaling --scales 5m,20m,50m,150m --device cuda` · `large --scales 500m,1b` | reward integrity battery; 16.5 meta checks; scaling; large cores (GPU lock) | [help] |
| `$P -m cam_live.eval.report [--tag main_cpu]` | text digest of `docs/evidence/S` | [help] |

Component diagnostics (each writes its own evidence file; see the owner's status file for the recorded command):
`$P -m cam_live.nn.bench [--skip-gpu] [--config C] [--out O]` (measured parameter counts and benchmarks, GPU lock
unless `--skip-gpu`), `$P -m cam_live.nn.fast_compare`, `$P -m cam_live.predict.diagnose [--quick] [--only X]`,
`$P -m cam_live.reward.diagnose`, `$P -m cam_live.reward.evidence --out O`, `$P -m cam_live.credit.diagnose [--real]`,
`$P -m cam_live.harness.grounding.bench [--grounders G]` — all [help].

---

## 7. Checkpoints and transfer

| command | check |
|---|---|
| `cam-live export-checkpoint [--run-dir R] [--out P]` — running runtime: operator command to the same hook; otherwise the committed checkpoint through `credit/checkpoint.py:export_weights_only` → `weights.safetensors` + `config.json` (+ `P.export.json`) | [exec on a scratch CPU run: exit 0, both files written] |
| `cam-live evaluate-transfer --checkpoint EXPERIENCED [--baseline PRE] [--domains T1,T2] [--seeds 0-4] [--lifetimes N] [--decisions N] [--frozen] [--real] [--jobs J] [--out R.json]` (= `$P -m cam_live.eval.transfer …`) — every version × domain × seed in its own fresh process with only a weights-only export | [exec, minimal CPU plumbing: same debug export as both versions, T1, 1 seed, 1 lifetime, 2 decisions: exit 0 in 13 s, `executed: true`] |
| load in a fresh process: `credit.checkpoint.load_exported_model(dest, register_heads=agent.register_agent_heads)` | cited (INTEGRATION.md §6) |

---

## 8. Credentials, operator key, live authorization (the operator's acts)

No command below prints a secret; values are never taken from the command line.

```bash
# operator signing key (once; the private key stays with the operator, mode 0600; the public key 0644)
$P -m cam_live.harness.auth.opctl init-operator-key --private /root/.config/camlive/operator_ed25519 \
    --public /root/.config/camlive/operator_ed25519.pub                                          # [help]

# a credential into a run's vault (owned by camauth, 0600) and the verifier's own copy (camverify, 0600)
$P -m cam_live.harness.credentials set discord_bot_token --run-dir R                      # hidden prompt   [help]
$P -m cam_live.harness.credentials set discord_bot_token --run-dir R --from-file /root/.config/camlive/discord_bot_token
$P -m cam_live.harness.credentials list --run-dir R                                      # names and bindings only
$P -m cam_live.harness.credentials revoke discord_bot_token --run-dir R
#   options: --dest PATTERN (repeatable; default for discord_bot_token: discord.com, gateway.discord.gg, *.discord.gg),
#            --scope S, --verifier / --no-verifier, --note; --from-file must be mode 0600
#   with the services running, the same through the authorization service (value on stdin):
#   $P -m cam_live.harness.auth.opctl --run-dir R put-secret NAME --dest PATTERN [--scope S]   (value on stdin)

# the signed live authorization record, bound to the task file's hash, its limits and a time window
$P cam_live/tools/authorize_live.py --task cam_live/tasks/discord_100_planning.yaml \
    --private /root/.config/camlive/operator_ed25519 --hours 8 --out R.signed.json \
    [--read 'https://*' ...] [--owned P ...] [--preauthorize P ...] [--account A ...] [--issued-by NAME] [--note T]
#   prints what is authorized and asks the operator to type the objective id (--yes skips that); refuses a task that
#   is not mode: live or does not load for a run (exit 3)                                        # [help]
```

The authorization service verifies the record against `<run>/auth/keys/operator.pub`. `configs/cam_live.yaml` has no
`launcher.operator_pubkey`, so with a plain `cam-live run` the key must be placed there first (as
`tools/launch_live.sh` does) or given in a config section `launcher: {operator_pubkey: <path>}`; otherwise the
supervisor's `register_live_authorization` fails ("operator public key not installed") and nothing starts.
Also available: `opctl sign-live record.json --private K > signed.json` and `opctl --run-dir R register-live
signed.json` [help].

---

## 9. Launch

### 9.1 Live (operator; separated services as root)

```bash
bash cam_live/tools/launch_live.sh cam_live/tasks/discord_launch_check.yaml 1 --read 'https://*' --read 'http://*'   # launch-path check
bash cam_live/tools/launch_live.sh cam_live/tasks/discord_100_planning.yaml 8 --read 'https://*' --read 'http://*'   # planning phase
#   env: CAM_LIVE_RUN_DIR (default runs/cam_live/live/<objective>-<UTC>), CAM_LIVE_OPERATOR_KEY, CAM_LIVE_CONF
```

[exec without arguments at `55a0f4c`: prints its usage, exit 2. Not run otherwise: live run, the operator's typed
confirmation, real credentials.] It refuses a non-interactive terminal. Steps: (1) `credentials set
discord_bot_token --run-dir R --from-file $CAM_LIVE_CONF/discord_bot_token`; (2) `install -m 0644
$CAM_LIVE_CONF/operator_ed25519.pub R/auth/keys/operator.pub`; (3) `tools/authorize_live.py … --out R.signed.json`;
(4) `cam-live run --task T --authorization R.signed.json --run-dir R`. The equivalent by hand is those four commands.

`cam-live run --task T [--run-dir R | --run-id ID] [--authorization F] [--config C] [--agent-factory M:F]
[--verifier-source S] [--no-doctor] [--start-timeout 120]` [help]: as root with the service users it is the
separated mode (`_supervise`: starts authorization, executor, verifier; registers the task and the record; runs the
preflight against the running services, including the read-only Discord identity / permission check; writes
`R/learner.go` only if every live item is confirmed). A live task is refused without the separated services or
without `--authorization` (exit 3). Detached; logs `R/logs/supervisor.log`; `$CAM_LIVE_HOME/current_run` records R.

Task files: `tasks/discord_launch_check.yaml` (one decision, two external actions, 30 min; launch-path check),
`tasks/discord_100_planning.yaml` (planning phase; operator values filled; preflight offline: 9 unconfirmed items,
all needing the running services or a persistent volume), `tasks/discord_100.yaml` (execution phase; its 16
values marked REQUIRED, 5 of them limits, are unset — refused). Read-only Discord adapter check through the real environment:
`$P cam_live/tools/discord_live_check.py --run-dir R --task T [--skip-writes] [--print-record-template]
[--write-policy FILE] [--out F]`, or locally `--staging [--live-policy]` [help; not run: live service / not needed].
Read-only live observation script (prepared, never executed): `docs/evidence/validation/discord_live/
run_live_observe.py --out F` [help].

### 9.2 Staging / diagnostic runs of the runtime

```bash
cam-live run --task cam_live/tasks/lab_staging.yaml --inproc [--foreground --max-iterations N] [--config C]   # [help]
```

`--inproc` is single-process (never for live tasks). The runtime and `runtime/loop_service.py` do **not** take the
shared GPU lock: with the default configuration a run holds the real backbone on the shared GPU for its whole
lifetime; pass a debug configuration (`--config`) or coordinate with the other tenant. Lower-level services:
`$P -m cam_live.runtime.loop_service --run-dir R [--inproc] [--wait-for F]`, `$P -m
cam_live.runtime.verifier_service --run-dir R [--source S]`, `$P -m cam_live.harness.service --run-dir R --socket S
…`, `$P -m cam_live.harness.auth.service --run-dir R --socket S`, `$P -m cam_live.harness.auth.launcher --run-dir R
[--services auth,exec,verifier,learner] [--policy P] [--operator-pubkey K] [--no-separation]` — all [help].

---

## 10. Operation: status, control, approvals, stop, resume

`--run-dir R` defaults to the run started last from the state root. Commands go through the signed operator
command channel (`runtime/control.py:OperatorControl`); a command to a run whose process is gone is queued and
applied when it starts.

| command | check |
|---|---|
| `cam-live status [--run-dir R] [--json]` | [exec on a scratch CPU run: exit 0; no run → exit 1] |
| `cam-live pause [--reason T]` · `cam-live stop [--reason T]` · `cam-live stop --emergency` | [help] |
| `cam-live resume [--run-dir R]` — paused: resume; process gone: restart from the persisted state (separated: through the supervisor); a stopped run is refused (exit 3) | [help] |
| `cam-live approve ID [--note T]` · `cam-live reject ID [--note T]` | [help] |
| `cam-live journal [--print]` — regenerate `R/journal.md` | [exec: exit 0] |
| `$P -m cam_live.harness.auth.opctl --run-dir R approvals \| approve ID \| reject ID \| stop "reason" \| reset-stop \| pause "reason" \| resume \| revoke SCOPE \| reload-policy [P] \| limits \| secrets \| audit-tail [n] \| verify-audit \| local-service add HOST:PORT \| local-service remove HOST:PORT \| local-service list` — `stop` and `verify-audit` work with the service down | [help] |
| `$P -m cam_live.harness.operator_requests --run-dir R list [--all] \| grant ID --note T \| deny ID --note T \| answer ID --note T` — CAM's requests to the operator (an answer grants nothing by itself) | [help] |

**Dashboard** (127.0.0.1 only; FastAPI):

```bash
cam-live dashboard [--run-dir R] [--port 8787] [--view-only]
# viewing token: $CAM_LIVE_HOME/dashboard.token  (http://127.0.0.1:8787/?token=<token> or Authorization: Bearer)
# operator actions: credential $CAM_LIVE_HOME/dashboard.operator (0600), not loaded with --view-only
ssh -p $VAST_TCP_PORT_22 -L 8787:127.0.0.1:8787 root@$PUBLIC_IPADDR     # from the operator's machine
```

[exec `--view-only` on a scratch run, port 18787: without token 401, with token 200, `/api/status` 200; stopped
afterwards.]

---

## 11. Retention and pruning

* Automatic: `cam_live.storage.DiskGuard` before every runtime checkpoint and every learner save / promotion
  (`storage.min_free_gb` 20: below it nothing is written and the run pauses with a resource obstacle;
  `warn_free_gb` 50); runtime keeps `runtime.keep_checkpoints: 2` (optimizer state only in the newest); learner
  promotions keep the last 2 + the promoted one; `eval.oe_runner` applies `storage.apply_retention` after every
  objective (INTEGRATION.md §5a).
* Manual, finished run trees (dry run by default; records are always kept, only checkpoint directories are touched):

  ```bash
  $P cam_live/tools/prune_runs.py runs/cam_live/eval                                            # report only
  $P cam_live/tools/prune_runs.py runs/cam_live/eval --apply --keep-optim 0 --drop-protected-optim [--keep 1] \
      [--frozen-glob 'A_*' ...] [--drop-group '*/lt[0-9][0-9]/checkpoints' ...] [--report F.json]
  ```

  [exec: dry run on the scratch state root: `DRY RUN: 0 removals, 0.0 GiB`; `--apply` not run.]

---

## 12. Off-box copy, completion checker, push

* **Run data and checkpoints have no off-box copy** (no persistent volume; rental ends 2026-11-30 03:00 UTC). Only
  git is pushed; anything to keep from `runs/` must be exported (`cam-live export-checkpoint`) and copied off the
  machine by the operator (e.g. `rsync`/`rclone` to their storage) [not run: needs an operator destination].
* Completion checker (read-only without `--merge-status`):
  `cam-live verify-completion [--only IDS] [--json F] [--quiet]` (= `$P -m cam_live.verify_completion`)
  [exec `--only IN-05,CM-13,DL-02 --quiet`: exit 1 = status level 1 not reached, by design; 2 partial, 1 missing].
  `--merge-status` rewrites the manifest and `docs/COMPLETION.md` (MAN only) [help; not run]. Manifest regeneration
  and verdict merge (MAN / G): `cd cam_live && ../.venv/bin/python tools/gen_manifest.py`, `.venv/bin/python
  cam_live/tools/merge_verdicts.py` [not run: no `--help`, they write their files when started].
* Push (operator-authorized milestones only): `bash cam_live/tools/push_milestone.sh` — on `feat/cam-live` only,
  pushes `HEAD` by explicit sha, never forces; aborts on any real credential of this machine (exact match, never
  printed) or a credential-like pattern outside `tools/secret_scan_allow.txt`; credential helper
  `$CAM_LIVE_CRED_HELPER` (default `/root/.config/camlive/gh_cred_helper.sh`). [not run: pushing is the master's act
  at milestones.]

---

## 13. Differences between documented and actual commands (found while checking)

* `cam-live validate --mode staging --debug` is not a CPU run (§5; defect of `runtime/cli.py:cmd_validate` +
  `validation.py:staging_config`; the test `tests/integration/test_int2_wiring.py::test_cli_validate_passes_its_options`
  replaces `run_staging` and does not see it).
* `cam-live test` without `--suite` and `pytest cam_live/tests` in one process fail collection (INT2-5, still open).
* Not present at `d3ce519` / `55a0f4c`: `cam-live plan-report` (named in CAM_LIVE_LEDGER.md session 4 as planned);
  `python -m cam_live.eval.oe_runner fresh` (docstring of `eval/oe_runner.py`; the 16.8 child is `sequence
  --from-export`); `eval.cli batch --plan baselines_cpu` (docstring; the choices are `main_cpu`, `pilot`);
  `eval.cli continual | shadow | skills` (docstring; not subcommands — `transfer-batch` exists and is not in the
  docstring).
* `resources.operator_authorization_record` in the task files is validated as a slug only; nothing compares it with
  the `record_id` that `tools/authorize_live.py` writes (`live-<objective_id>-<unix time>`).

---

## 14. Changes committed after the check (INT3B `5ae3b59`, checked by DOC 2026-11-27 23:45 UTC)

* **`cam-live plan-report [--run-dir R] [--out F] [--json] [--timeout S]`** now exists (`runtime/plan_report.py`).
  It exports, verbatim and with provenance, CAM's objective, notes, predictions, experiments, operator requests,
  actions and learning updates. [help: exit 0. Not run: no planning run exists.] The `plan-report` item of §13 is
  resolved.
* **GPU lock in the runtime.** `runtime.gpu_lock` (`auto` / `on` / `off`; `auto` = a pretrained backbone on CUDA)
  makes the runtime process (`runtime/loop_service.py`) hold the shared GPU lock from before the agent is built until
  it exits. `runtime.gpu_hold_s > 0` bounds the hold: the runtime then commits its checkpoint, pauses and exits, and
  `cam-live resume` continues for the next hold. `runtime.policy_file` sets the authorization policy of an
  in-process run. None of these keys is set in `configs/cam_live.yaml`, so the defaults apply (`auto`, unbounded
  hold). [code reading; not run.] The GPU-lock item of §13 and of `docs/KNOWN_FAILURES.md` KF-D07 is updated
  accordingly.
* **Staging rehearsal of the planning phase.** Task `tasks/staging/discord_100_planning_staging.yaml` and tool
  `tools/planning_rehearsal.py prepare --root R [--config C] [--hold-s 840]`, `serve --root R`,
  `holds --root R --decisions N [--max-holds H]`, `check --root R [--out F]`. Staging only: it serves a local
  staging Discord on 127.0.0.1:18471 and a neutral web on 127.0.0.1:18472. [help of the tool and of all four
  subcommands: exit 0. `cam-live preflight --task cam_live/tasks/staging/discord_100_planning_staging.yaml
  --no-doctor`: exit 3, 9 live items, mode staging. Not run further.]
