"""G2-integration / docs probe d03: the commands of docs/REPRODUCE.md, run with --help or offline, against a tree.

usage (from /workspace/camlive):
  PYTHONPATH=<SRC>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2integration/docs/d03_reproduce_commands.py \
      <SRC> <SCRATCH>
SRC = tree with cam_live/ (e.g. the frozen copy of a5ab772); SCRATCH = an empty directory for CAM_LIVE_HOME and logs.
Nothing here starts a service, contacts a network or uses the GPU (CUDA_VISIBLE_DEVICES is emptied, CAM_LIVE_GPU_LOCK
points into SCRATCH). Prints HOLDS / BROKEN per documented claim and writes SCRATCH/d03_results.tsv.
"""
import os
import subprocess
import sys

SRC, SCR = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
PY = "/workspace/camlive/.venv/bin/python"
CL = "/workspace/camlive/.venv/bin/cam-live"
HOME = os.path.join(SCR, "home")
os.makedirs(os.path.join(SCR, "logs"), exist_ok=True)
ENV = dict(os.environ, PYTHONPATH=os.path.join(SRC, "cam_live", "src"), CAM_LIVE_HOME=HOME, PYTHONDONTWRITEBYTECODE="1",
           CAM_LIVE_GPU_LOCK=os.path.join(SCR, "gpu.lock"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
rows = []


def run(cid, argv, timeout=600):
    log = os.path.join(SCR, "logs", cid + ".txt")
    try:
        p = subprocess.run(argv, cwd=SRC, env=ENV, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", -9
    open(log, "w").write(out)
    first = next((ln for ln in out.splitlines() if ln.strip()), "")[:140]
    rows.append((code, cid, " ".join(argv), first))
    return code, out


def claim(cid, ok, what):
    print(f"{'HOLDS ' if ok else 'BROKEN'}  {cid}: {what}")


# ---- [help] commands (REPRODUCE sections 2-12, 14)
helps = [["cam-live", CL, "--help"]]
for s in ("doctor init test validate inspect-harness preflight run status pause resume stop approve reject "
          "export-checkpoint journal dashboard evaluate-transfer verify-completion plan-report").split():
    helps.append([f"cam-live_{s}", CL, s, "--help"])
for m in ("cam_live.runtime.cli cam_live.runtime.loop_service cam_live.runtime.verifier_service cam_live.harness.credentials "
          "cam_live.harness.auth.opctl cam_live.harness.auth.launcher cam_live.harness.auth.service cam_live.harness.service "
          "cam_live.harness.operator_requests cam_live.harness.staging cam_live.harness.browser cam_live.harness.grounding.bench "
          "cam_live.eval.cli cam_live.eval.oe_runner cam_live.eval.proposal_probe cam_live.eval.transfer cam_live.eval.report "
          "cam_live.verify_completion cam_live.nn.bench cam_live.nn.fast_compare cam_live.predict.diagnose "
          "cam_live.reward.diagnose cam_live.reward.evidence cam_live.credit.diagnose").split():
    helps.append([m, PY, "-m", m, "--help"])
for s in "run anchors batch analyse integrity meta scaling large transfer-batch".split():
    helps.append([f"eval.cli_{s}", PY, "-m", "cam_live.eval.cli", s, "--help"])
for s in ("sequence", "transfer"):
    helps.append([f"oe_runner_{s}", PY, "-m", "cam_live.eval.oe_runner", s, "--help"])
for t in ("authorize_live", "discord_live_check", "prune_runs", "planning_rehearsal"):
    helps.append([f"tools_{t}", PY, f"cam_live/tools/{t}.py", "--help"])
for s in ("prepare", "serve", "holds", "check"):
    helps.append([f"planning_rehearsal_{s}", PY, "cam_live/tools/planning_rehearsal.py", s, "--help"])
helps.append(["run_separated", PY, "cam_live/docs/evidence/validation/separated/run_separated.py", "--help"])
helps.append(["run_live_observe", PY, "cam_live/docs/evidence/validation/discord_live/run_live_observe.py", "--help"])
helps.append(["opctl_init_key", PY, "-m", "cam_live.harness.auth.opctl", "init-operator-key", "--help"])
n_ok = 0
for h in helps:
    code, _ = run("help_" + h[0], h[1:], timeout=300)
    claim("help_" + h[0], code == 0, f"--help exits 0 (exit {code})")
    n_ok += code == 0
print(f"help summary: {n_ok}/{len(helps)} exit 0")

# ---- offline executions with a documented result
c, o = run("doctor", [CL, "doctor"])
claim("doctor", c == 0 and "/51 checks ok" in o, f"REPRODUCE 4: '48/51 ok' -> got: {[l for l in o.splitlines() if 'checks ok' in l]}")
c, o = run("doctor_strict", [CL, "doctor", "--strict"])
claim("doctor_strict", c == 0, f"REPRODUCE 4: --strict exit 0 (exit {c})")
c, o = run("preflight_lab", [CL, "preflight", "--task", "cam_live/tasks/lab_staging.yaml", "--no-doctor"])
claim("preflight_lab", c == 0, f"REPRODUCE 4: lab_staging exit 0 (exit {c})")
c, o = run("preflight_staging_planning", [CL, "preflight", "--task",
                                          "cam_live/tasks/staging/discord_100_planning_staging.yaml", "--no-doctor"])
claim("preflight_staging_planning", c == 3 and "BLOCKED — 9 item" in o and "mode staging" in o,
      f"REPRODUCE 14: exit 3, 9 live items, mode staging (exit {c}; {[l for l in o.splitlines() if l.startswith('LIVE')]})")
c, o = run("init_no_agent", [CL, "init", "--no-agent", "--home", HOME])
claim("init_no_agent", c == 0 and os.path.exists(os.path.join(HOME, "dashboard.token")),
      f"REPRODUCE 2.5: exit 0, state root and token created (exit {c})")
c, o = run("inspect_harness_lab", [CL, "inspect-harness", "--lab"])
claim("inspect_harness_lab", c == 0 and "discord" in o.lower(), f"REPRODUCE 4: exit 0, Discord listed missing (exit {c})")
c, o = run("browser_find", [PY, "-m", "cam_live.harness.browser", "--find", "cambrowser"])
claim("browser_find", c == 0 and "chromium-1255" in o, f"REPRODUCE 2.3: .../chromium-1255/... (exit {c}: {o.strip()[:100]})")
c, o = run("prune_dry", [PY, "cam_live/tools/prune_runs.py", HOME])
claim("prune_dry", c == 0 and "DRY RUN" in o, f"REPRODUCE 11: 'DRY RUN: 0 removals' (exit {c})")
c, o = run("status_no_run", [CL, "status", "--run-dir", os.path.join(HOME, "runs", "none")])
claim("status_no_run", c == 1, f"REPRODUCE 10: no run -> exit 1 (exit {c})")
c, o = run("verify_completion_only", [CL, "verify-completion", "--only", "IN-05,CM-13,DL-02", "--quiet"])
claim("verify_completion_only", c == 1, f"REPRODUCE 12: exit 1 (exit {c}: {o.strip()[:120]})")
c, o = run("launch_live_noargs", ["bash", "cam_live/tools/launch_live.sh"])
claim("launch_live_noargs", c == 2, f"REPRODUCE 9.1: no arguments -> usage, exit 2 (exit {c})")
c, o = run("validate_live_refused", [CL, "validate", "--mode", "live"])
claim("validate_live_refused", c == 2, f"REPRODUCE 5: --mode live refused, exit 2 (exit {c})")

# ---- documented-but-missing commands (REPRODUCE 13) and their docstrings
c, o = run("oe_runner_fresh", [PY, "-m", "cam_live.eval.oe_runner", "fresh", "--help"])
claim("oe_runner_fresh_absent", c != 0, f"REPRODUCE 13: 'oe_runner fresh' does not exist (exit {c})")
c, o = run("eval_batch_baselines", [PY, "-m", "cam_live.eval.cli", "batch", "--plan", "baselines_cpu", "--help"])
claim("batch_baselines_absent", c != 0, f"REPRODUCE 13: '--plan baselines_cpu' refused (exit {c})")
for s in ("continual", "shadow", "skills"):
    c, o = run(f"eval_{s}", [PY, "-m", "cam_live.eval.cli", s, "--help"])
    claim(f"eval_{s}_absent", c != 0, f"REPRODUCE 13: 'eval.cli {s}' is not a subcommand (exit {c})")
doc_cli = open(os.path.join(SRC, "cam_live/src/cam_live/eval/cli.py")).read().split('"""')[1]
doc_oe = open(os.path.join(SRC, "cam_live/src/cam_live/eval/oe_runner.py")).read().split('"""')[1]
claim("docstrings_name_missing_commands",
      ("baselines_cpu" in doc_cli or " continual" in doc_cli or "oe_runner fresh" in doc_oe),
      "REPRODUCE 13 / KF-E12: the eval docstrings still name the missing commands")

# ---- tests (REPRODUCE 3)
c, o = run("collect_runtime", [PY, "-m", "pytest", "cam_live/tests/runtime", "--collect-only", "-q", "-p",
                               "no:cacheprovider", "--rootdir=cam_live"])
tail = [l for l in o.splitlines() if "collected" in l or "tests" in l][-1:]
claim("collect_runtime", c == 0 and "215 tests" in o, f"REPRODUCE 3: runtime 215 tests (exit {c}: {tail})")
c, o = run("collect_storage", [PY, "-m", "pytest", "cam_live/tests/storage", "--collect-only", "-q", "-p",
                               "no:cacheprovider", "--rootdir=cam_live"])
claim("collect_storage", c == 0 and "3 tests" in o, f"REPRODUCE 3: storage 3 tests (exit {c})")
c, o = run("collect_all_one_process", [PY, "-m", "pytest", "cam_live/tests", "--collect-only", "-q", "-p",
                                       "no:cacheprovider", "--rootdir=cam_live"], timeout=900)
tail = [l for l in o.splitlines() if "collected" in l or "error" in l.lower()][-2:]
claim("collect_all_fails", c == 2, f"REPRODUCE 3 / KF-E10: one-process collection fails, exit 2 (exit {c}: {tail})")
c, o = run("cam-live_test_storage", [CL, "test", "--suite", "storage", "--results-dir", os.path.join(SCR, "junit")])
claim("cam-live_test_storage", c == 0 and "3 passed" in o, f"REPRODUCE 3: exit 0, 3 passed (exit {c})")

with open(os.path.join(SCR, "d03_results.tsv"), "w") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"commands run: {len(rows)}; results in {os.path.join(SCR, 'd03_results.tsv')}")
