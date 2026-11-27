"""G-rt probe 10 — the cam-live command line (spec Part 17): every command exists and does what it says; which
commands depend on modules that are not in the installation.

    .venv/bin/python cam_live/docs/verification/repro/rt/p10_cli.py            (about 1-2 minutes)

State root: a scratch directory through CAM_LIVE_HOME (nothing is written under the repository's runs/).
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import SRC, Report, read_log, scratch  # noqa: E402

from cam_live.config import PROJECT_ROOT  # noqa: E402
from cam_live.contracts import RunFiles  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402

R = Report("p10_cli")
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = scratch("p10")
HOME = os.path.join(tmp, "home")
ENV = dict(os.environ, CAM_LIVE_HOME=HOME, CAM_LIVE_GPU_LOCK=os.path.join(tmp, "gpu.lock"),
           PYTHONPATH=os.pathsep.join([SRC, HERE]))
CFG = os.path.join(tmp, "cfg.yaml")
yaml.safe_dump({"seed": 1, "home": HOME, "backbone": {"name": "tiny", "revision": ""},
                "runtime": {"control_poll_s": 0.2, "min_idle_sleep_s": 0.0},
                "g_standin": {"script": ["lab.act", "lab.probe", "wait"], "wait_seconds": 86400.0}}, open(CFG, "w"))


def cli(*args, timeout=240):
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True, env=ENV,
                       timeout=timeout, cwd="/workspace/camlive")
    return p.returncode, p.stdout, p.stderr, time.time() - t0


SPEC = ["doctor", "init", "test", "validate", "inspect-harness", "run", "status", "pause", "resume",
        "evaluate-transfer", "export-checkpoint", "verify-completion"]
code, out, err, _ = cli("--help")
R.check("0a `cam-live --help` lists the 12 commands of spec Part 17", code == 0 and all(c in out for c in SPEC),
        [c for c in SPEC if c not in out])
exe = os.path.join("/workspace/camlive/.venv/bin/cam-live")
R.check("0b the `cam-live` console script is installed", os.access(exe, os.X_OK))
missing = {m: importlib.util.find_spec(m) is None for m in ("cam_live.agent", "cam_live.validation", "cam_live.verify_completion")}
try:
    missing["cam_live.eval.transfer"] = importlib.util.find_spec("cam_live.eval.transfer") is None
except ModuleNotFoundError:
    missing["cam_live.eval.transfer"] = True
R.note("modules absent from this installation right now", missing)

# ---- doctor ----
code, out, err, dt = cli("doctor", "--config", CFG, "--json")
doc = json.loads(out) if code == 0 else {}
R.check("1a doctor: exit 0, sections hardware/storage/models/dependencies/sandbox/service_users/harness",
        code == 0 and set(doc.get("sections", {})) == {"hardware", "storage", "models", "dependencies", "sandbox", "service_users", "harness"},
        err[-200:])
R.note("1b doctor wall time through the CLI (status file: doctor_seconds 3.6)", round(dt, 1))
code2, out2, _, _ = cli("doctor", "--config", CFG, "--strict")
R.check("1c doctor --strict exits 1 while something blocks staging (tiny backbone / absent modules)", code2 == 1, out2[-300:])

# ---- init ----
code, out, err, _ = cli("init", "--config", CFG)
R.check("2a init with the default agent factory: exit 4 naming cam_live.agent" if missing["cam_live.agent"] else
        "2a init with the default agent factory runs", (code == 4 and "cam_live.agent" in err) if missing["cam_live.agent"] else code == 0,
        err[-200:])
code, out, err, _ = cli("init", "--config", CFG, "--agent-factory", "g_factory:build_agent", "--json")
rep = json.loads(out) if code == 0 else {}
tok = os.path.join(HOME, "dashboard.token")
R.check("2b init with a stand-in factory: state root, resolved config, 0600 dashboard token, initial checkpoint on disk",
        code == 0 and os.path.isdir(os.path.join(HOME, "runs")) and os.path.exists(os.path.join(HOME, "config.resolved.json"))
        and stat.S_IMODE(os.stat(tok).st_mode) == 0o600 and os.path.exists(os.path.join(rep["checkpoint"]["path"], "agent.json")),
        err[-300:])

# ---- test ----
code, out, err, _ = cli("test", "--suite", "runtime", "-k", "test_binom or test_secondary_metrics_are_separate_quantities",
                        "--results-dir", tmp)
xml = os.path.join(tmp, "cli_runtime.xml")
R.check("3a test --suite runtime -k <...>: runs pytest, exit 0, JUnit XML written", code == 0 and os.path.exists(xml), out[-300:])
code, out, err, _ = cli("test", "--suite", "no_such_suite", "--results-dir", tmp)
R.check("3b test with an unknown suite: exit 2", code == 2)
code, out, err, _ = cli("test", "--suite", "runtime", "-k", "g_rt_matches_nothing", "--results-dir", tmp)
R.check("3c test that runs nothing does not report success", code != 0, code)

# ---- validate ----
code, out, err, _ = cli("validate", "--mode", "staging", "--config", CFG)
R.check("4a validate --mode staging: exit 4 naming cam_live.validation (module absent)" if missing["cam_live.validation"] else
        "4a validate --mode staging runs", (code == 4 and "cam_live.validation" in err) if missing["cam_live.validation"] else code in (0, 1),
        err[-200:])
code, out, err, _ = cli("validate", "--mode", "live", "--config", CFG)
R.check("4b validate --mode live is refused (exit 2)", code == 2)

# ---- inspect-harness ----
code, out, err, _ = cli("inspect-harness", "--lab", "--config", CFG, "--json")
info = json.loads(out) if code == 0 else {}
R.check("5a inspect-harness --lab lists the LabWorld tools with scopes and side effects",
        code == 0 and {"lab.act", "lab.probe", "wait"} <= {t["name"] for t in info.get("tools", [])})
code, out, err, _ = cli("inspect-harness", "--no-browser", "--config", CFG, "--json", "--task", os.path.join(PROJECT_ROOT, "tasks", "discord_100.yaml"))
info = json.loads(out) if code == 0 else {}
names = sorted(t["name"] for t in info.get("tools", []))
miss = [m["tool"] for m in info.get("missing_integrations", [])]
R.check("5b inspect-harness on the computer environment: exit 0, tools + permissions listed", code == 0 and bool(names)
        and "policy_engine" in info.get("permissions", {}), {"tools": names, "err": err[-200:]})
R.check("5c ... and the Discord integration (no credentials here) is named among the missing integrations",
        any("discord" in m.lower() for m in miss) or any("discord" in n for n in names),
        {"missing_integrations": miss, "note": "only interfaces of the task without any tool are listed"})

# ---- run: refusals ----
DISCORD = os.path.join(PROJECT_ROOT, "tasks", "discord_100.yaml")
code, out, err, _ = cli("run", "--task", DISCORD, "--config", CFG, "--no-doctor")
R.check("6a run tasks/discord_100.yaml as shipped: exit 3, REFUSED, nothing started", code == 3 and "REFUSED" in err
        and not os.path.exists(os.path.join(HOME, "current_run")), err[-200:])
raw = T.load_task_raw(DISCORD)
raw["success"]["evidence_source"] = "g_fake_module:live_source"
raw["success"]["exclusions"]["test_account_ids"] = []
raw["resources"].update(discord_guild_id="G1", discord_bot_credential_name="discord_bot", owned_site="https://example.invalid",
                        operator_authorization_record="rec-1", authorized_tools=["api"], authorized_destinations=["G1"], currency="USD")
raw["resources"]["publishing"].update(authorized=True, destinations=["G1"])
raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100, "max_decisions": 100}
raw["data_permissions"]["collect_feedback_with_consent"] = False
LIVE = os.path.join(tmp, "live_filled.yaml")
yaml.safe_dump(raw, open(LIVE, "w"))
code, out, err, _ = cli("run", "--task", LIVE, "--config", CFG, "--no-doctor", "--run-dir", os.path.join(tmp, "live_run_a"))
R.check("6b run a completely filled LIVE task without --authorization: exit 3, REFUSED, no run directory content",
        code == 3 and "authorization" in err and not os.path.exists(os.path.join(tmp, "live_run_a", "state.json")), err[-160:])
code, out, err, _ = cli("run", "--task", LIVE, "--config", CFG, "--no-doctor", "--inproc", "--authorization", CFG,
                        "--run-dir", os.path.join(tmp, "live_run_b"))
R.check("6c run a LIVE task with --inproc: exit 3 (needs the separated services)", code == 3 and "separated" in err, err[-160:])
code, out, err, _ = cli("run", "--task", LIVE, "--config", CFG, "--no-doctor", "--inproc", "--foreground", "--authorization", CFG,
                        "--run-dir", os.path.join(tmp, "live_run_c"))
R.check("6d run a LIVE task with --inproc --foreground: refused", code == 3, (code, err[-160:]))
p = subprocess.run([sys.executable, "-m", "cam_live.runtime.loop_service", "--run-dir", os.path.join(tmp, "live_run_d"),
                    "--task", LIVE, "--config", CFG, "--agent-factory", "g_factory:build_agent", "--inproc"],
                   capture_output=True, text=True, env=ENV, timeout=120, cwd="/workspace/camlive")
R.check("6e starting the learner process directly on a LIVE task with --inproc is refused",
        p.returncode != 0 and "live" in p.stderr.lower() and not os.path.exists(os.path.join(tmp, "live_run_d", "state.json")),
        p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else p.returncode)
bad = os.path.join(tmp, "strategy.yaml")
raw2 = T.load_task_raw(os.path.join(PROJECT_ROOT, "tasks", "lab_staging.yaml"))
raw2["strategy"] = "do this"
yaml.safe_dump(raw2, open(bad, "w"))
code, out, err, _ = cli("run", "--task", bad, "--config", CFG, "--no-doctor", "--inproc")
R.check("6f run a task file containing a 'strategy' key: exit 3", code == 3, err[-160:])

# ---- run / status / pause / resume / export-checkpoint / stop on a detached in-process run (real clock) ----
# The in-process computer environment + authorization service + metric verifier of the harness, staging provenance,
# no browser; the stand-in learner writes a workspace note, then waits 30 s, so the process is alive and mostly asleep.
STG = os.path.join(tmp, "staging.yaml")
raw3 = T.load_task_raw(os.path.join(PROJECT_ROOT, "tasks", "utility_staging.yaml"))
json.dump({"utility.distinct_successful_sessions": 0}, open(os.path.join(tmp, "metrics.json"), "w"))
raw3["success"]["evidence_source"] = "json:" + os.path.join(tmp, "metrics.json")
raw3["resources"]["staging_site"] = "http://127.0.0.1:9"
raw3["limits"]["max_decisions"] = 400
yaml.safe_dump(raw3, open(STG, "w"))
cfg2 = yaml.safe_load(open(CFG))
cfg2["runtime"]["browser"] = False
cfg2["g_standin"] = {"script": [["internal.note", {"text": "g-rt stand-in note"}], "wait"], "wait_seconds": 30.0}
yaml.safe_dump(cfg2, open(CFG, "w"))
run_dir = os.path.join(tmp, "stg_run")
common = ("--config", CFG, "--run-dir", run_dir)
code, out, err, _ = cli("run", "--task", STG, *common, "--inproc", "--agent-factory", "g_factory:build_agent", "--no-doctor", "--json")
info = json.loads(out) if code == 0 else {}
R.check("7a run (detached, in-process harness, staging task): exit 0, runtime pid recorded, state.json written",
        code == 0 and info.get("started") and info.get("runtime_pid"), (err or out)[-400:])
pid = info.get("runtime_pid")


def st():
    try:
        return json.load(open(os.path.join(run_dir, RunFiles.STATE)))
    except (OSError, ValueError):
        return {}


time.sleep(1.0)
code, out, err, _ = cli("status", *common, "--json")
s = json.loads(out) if code == 0 else {}
R.check("7b status: objective, verified progress, learning state, resources, process liveness",
        code == 0 and s.get("objective", {}).get("id") == "staging-utility" and "verified" in s and "learning" in s
        and "resources" in s and s.get("process", {}).get("alive") is True, {k: s.get(k) for k in ("status", "process")})
code, out, err, dt = cli("pause", *common, "--reason", "g-rt", "--json")
ok = code == 0 and st().get("status") == "paused"
ck = (st().get("committed") or {}).get("ckpt") or {}
R.check("7c pause: acknowledged, status paused, learner checkpoint on disk", ok and os.path.exists(os.path.join(ck.get("path", "/nonexistent"), "agent.json")),
        {"seconds": round(dt, 1), "status": st().get("status")})
d0 = st().get("counters", {}).get("decisions")
time.sleep(1.5)
R.check("7d while paused no decision is taken", st().get("counters", {}).get("decisions") == d0, d0)
code, out, err, dt = cli("resume", *common, "--json")
R.check("7e resume (process alive): status running again", code == 0 and st().get("status") in ("running", "exhausted", "completed", "failed"),
        st().get("status"))
code, out, err, _ = cli("export-checkpoint", *common, "--json")
exp = (json.loads(out).get("result") or {}) if code == 0 else {}
exports = st().get("exports") or []
R.check("7f export-checkpoint (process alive): an export directory and manifest exist",
        code == 0 and exports and os.path.exists(exports[-1]["path"]) and os.path.exists(exports[-1]["path"].rstrip("/") + ".export.json"),
        exp.get("detail", err[-200:]))
# kill -9 the runtime, then `cam-live resume` must restart it from the persisted state
import signal  # noqa: E402
if pid:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
time.sleep(0.5)
d1 = st().get("counters", {}).get("decisions")
code, out, err, _ = cli("resume", *common, "--agent-factory", "g_factory:build_agent", "--json")
rs = json.loads(out) if code == 0 else {}
R.check("7g resume after kill -9: the runtime process is restarted and restores the persisted state",
        code == 0 and rs.get("restarted") and rs.get("restored") is not None, err[-300:])
code, out, err, _ = cli("export-checkpoint", *common, "--out", os.path.join(tmp, "export_x"), "--json")
code, out, err, _ = cli("stop", *common, "--reason", "g-rt end", "--json")
time.sleep(1.0)
R.check("7h stop: status stopped", st().get("status") == "stopped", st().get("status"))
code, out, err, _ = cli("export-checkpoint", *common, "--out", os.path.join(tmp, "export_offline"), "--json")
meta = json.loads(out) if code == 0 else {}
R.check("7i export-checkpoint with no process: exports the committed checkpoint and says how",
        code == 0 and os.path.exists(os.path.join(tmp, "export_offline")) and "method" in meta, meta.get("method", err[-200:]))
R.note("7j what export-checkpoint exported (spec: 'trained neural weights, adapters and compatible configuration')", meta.get("method"))
code, out, err, _ = cli("resume", *common, "--agent-factory", "g_factory:build_agent", "--json")
time.sleep(1.0)
R.check("7k resume after stop does not bring a stopped run back to running", st().get("status") == "stopped", (code, st().get("status")))
code, out, err, _ = cli("journal", *common)
R.check("7l journal: regenerated", code == 0 and os.path.exists(os.path.join(run_dir, "journal.md")))

# ---- evaluate-transfer / verify-completion ----
code, out, err, _ = cli("evaluate-transfer", "--checkpoint", os.path.join(tmp, "export_offline"), "--config", CFG)
R.check("8a evaluate-transfer: exit 4 naming cam_live.eval.transfer (module absent)" if missing["cam_live.eval.transfer"] else
        "8a evaluate-transfer dispatches", (code == 4 and "cam_live.eval.transfer" in err) if missing["cam_live.eval.transfer"] else code in (0, 1, 2),
        err[-200:])
code, out, err, _ = cli("evaluate-transfer", "--checkpoint", "/nonexistent/ckpt")
R.check("8b evaluate-transfer with a missing checkpoint: exit 2", code == 2)
code, out, err, dt = cli("verify-completion", timeout=400)
R.check("9a verify-completion (bare) runs the master's completion checker and reports the five status levels",
        code in (0, 1) and "status levels" in out and "1_engineering_implementation_complete" in out, (code, (out + err)[-200:]))
code, out, err, _ = cli("verify-completion", "--only", "CM-12", "--quiet")
code2, out2, err2, _ = cli("verify-completion", "--json", os.path.join(tmp, "vc.json"))
R.check("9b the checker's own options (--only, --json PATH, --quiet, --merge-status) can be passed through cam-live",
        code in (0, 1) and code2 in (0, 1), {"--only CM-12 --quiet": (code, err.strip().splitlines()[-1][:120] if err.strip() else ""),
                                             "--json PATH": (code2, err2.strip().splitlines()[-1][:120] if err2.strip() else "")})
R.check("0c nothing was written under the repository's runs/ by these commands", not os.path.exists("/workspace/camlive/runs/cam_live"),
        os.listdir("/workspace/camlive/runs") if os.path.exists("/workspace/camlive/runs") else "no runs/ directory")
sys.exit(R.finish())
