"""doctor, preflight and the cam-live command line (the commands that do not need the integrated agent)."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from cam_live.config import CamLiveConfig
from cam_live.contracts import RunFiles
from cam_live.diagnostics.world import AllowAllPolicy, LabWorld
from cam_live.runtime import cli, doctor, preflight
from cam_live.runtime import tasks as T
from cam_live.runtime.control import pid_alive, read_pid
from cam_live.runtime.logs import RunReader, read_json

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "..", "..", "tasks")
DISCORD = os.path.join(TASKS, "discord_100.yaml")
LAB = os.path.join(TASKS, "lab_staging.yaml")


def test_doctor_reports_what_is_on_this_machine():
    rep = doctor.run_doctor(CamLiveConfig())
    assert set(rep["sections"]) == {"hardware", "storage", "models", "dependencies", "sandbox", "service_users", "harness"}
    by = {c["name"]: c for s in rep["sections"].values() for c in s}
    for name in ("gpu", "memory", "disk", "persistent_storage", "backbone", "grounding_models", "python", "dep:torch",
                 "dep:fastapi", "chromium", "landlock", "seccomp", "setuid_setgid", "kernel_network_filtering",
                 "user:learner", "user:verifier", "module:cam_live.harness.env", "module:cam_live.runtime.loop_service",
                 "config_file"):
        assert name in by, name
        assert by[name]["ok"] in (True, False, None) and by[name]["detail"]
    assert by["python"]["ok"] is True and by["dep:fastapi"]["ok"] is True
    assert by["module:cam_live.runtime.loop_service"]["ok"] is True
    # an unknown is never counted as a pass, and failures required for live are listed as blocking
    s = rep["summary"]
    assert s["ok"] + len(s["failed"]) + len(s["unknown"]) == s["checks"]
    for name in s["unknown"]:
        if by[name]["needed_for"]:
            assert name in s["blocking_live"]
    text = doctor.render(rep)
    assert "[hardware]" in text and "blocking for live" in text
    # the tiny test backbone is never reported as an acceptable backbone
    tiny = doctor.model_checks(CamLiveConfig.debug())
    assert tiny[0]["ok"] is False and "tiny" in tiny[0]["detail"]


def test_preflight_blocks_live_with_the_explicit_missing_list():
    rep = preflight.preflight(DISCORD, CamLiveConfig(), with_doctor=False)
    assert rep["live"]["blocked"] is True and rep["live"]["ready"] is False and rep["mode"] == "live"
    items = {(m["kind"], m["item"]) for m in rep["live"]["missing"]}
    for expect in (("limit", "limits.duration_s"), ("limit", "limits.money"), ("limit", "limits.compute_s"),
                   ("limit", "limits.max_external_actions"), ("limit", "limits.max_decisions"),
                   ("authorization", "resources.authorized_tools"), ("authorization", "resources.publishing.authorized"),
                   ("data_permission", "data_permissions.collect_feedback_with_consent"),
                   ("credential", "resources.discord_bot_credential_name"), ("resource", "resources.discord_guild_id"),
                   ("resource", "resources.owned_site"), ("authorization", "authorization service"),
                   ("integration", "computer environment"), ("verification", "independent verifier")):
        assert expect in items, expect
    text = preflight.render(rep)
    assert "LIVE: BLOCKED" in text and "limits.money" in text
    # staging readiness is a separate statement
    assert "staging" in rep and rep["staging"]["missing"] == []


def _filled(tmp_path):
    raw = T.load_task_raw(DISCORD)
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 1000, "max_external_actions": 10, "max_decisions": 10}
    raw["resources"].update(discord_guild_id="G", discord_bot_credential_name="discord_bot", owned_site="https://x.test",
                            operator_authorization_record="rec-1", authorized_tools=["discord"],
                            authorized_destinations=["https://x.test/*"], currency="USD",
                            publishing={"authorized": True, "approval_required": True, "destinations": ["https://x.test"]})
    raw["success"]["evidence_source"] = "pkg.mod:factory"
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["data_permissions"]["collect_feedback_with_consent"] = True
    p = tmp_path / "filled.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def test_preflight_never_reports_live_readiness_from_staging_or_diagnostic_components(tmp_path):
    path = _filled(tmp_path)
    assert T.validate_task(T.load_task_raw(path), for_run=True) == []
    # a fully filled task, handed diagnostic services that answer "yes" to everything: still blocked, with reasons
    rep = preflight.preflight(path, CamLiveConfig(), policy=AllowAllPolicy(), env=LabWorld(),
                              verifier_info={"objective_id": "discord-100", "source": "jsonl:members", "source_live": False},
                              with_doctor=False)
    assert rep["live"]["blocked"] is True
    details = " | ".join(f"{m['item']}: {m['detail']}" for m in rep["live"]["missing"])
    assert "diagnostic policy" in details and "can never authorize live operation" in details
    assert "not a live environment" in details and "do not count towards live readiness" in details
    assert "not a live evidence source" in details
    assert not [m for m in rep["live"]["missing"] if m["kind"] in ("limit", "data_permission")]

    # an authorization service that holds no operator record for this task, and lacks the named credential
    class Policy:
        def limits_state(self):
            return {"policy": {"valid": True}, "stopped": False,
                    "tasks": {"discord-100": {"unavailable": ["no operator authorization record for this task"]}}}

        def secret_names(self):
            return [{"name": "something_else"}]

    rep = preflight.preflight(path, CamLiveConfig(), policy=Policy(), with_doctor=False)
    items = {(m["kind"], m["item"]): m["detail"] for m in rep["live"]["missing"]}
    assert items[("authorization", "operator authorization record")] == "no operator authorization record for this task"
    assert ("credential", "discord_bot") in items


def _cli(*args, env=None, timeout=120):
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "..", "src"), HERE])
    e.update(env or {})
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True, env=e,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def test_cli_has_every_command_of_the_specification():
    code, out, _ = _cli("--help")
    assert code == 0
    for cmd in ("doctor", "init", "test", "validate", "inspect-harness", "run", "status", "pause", "resume",
                "evaluate-transfer", "export-checkpoint", "verify-completion"):
        assert cmd in out, cmd
    entry = open(os.path.join(HERE, "..", "..", "pyproject.toml")).read()
    assert 'cam-live = "cam_live.runtime.cli:main"' in entry


def test_cli_doctor_preflight_and_refusals(tmp_path):
    code, out, _ = _cli("doctor", "--json")
    assert code == 0 and "sections" in json.loads(out)
    code, out, _ = _cli("preflight", "--task", DISCORD, "--no-doctor")
    assert code == 3 and "LIVE: BLOCKED" in out and "limits.duration_s" in out
    # run refuses the unfilled live task and starts nothing
    home = str(tmp_path / "home")
    code, out, err = _cli("run", "--task", DISCORD, "--no-doctor", "--run-dir", str(tmp_path / "r1"),
                          env={"CAM_LIVE_HOME": home})
    assert code == 3 and "REFUSED" in err and "limits.money" in err and not os.path.exists(tmp_path / "r1")
    # a filled live task without the operator authorization record is refused as well
    code, out, err = _cli("run", "--task", _filled(tmp_path), "--no-doctor", "--run-dir", str(tmp_path / "r2"),
                          env={"CAM_LIVE_HOME": home})
    assert code == 3 and "operator authorization record" in err and not os.path.exists(tmp_path / "r2")
    # ... and never in-process
    code, out, err = _cli("run", "--task", _filled(tmp_path), "--no-doctor", "--inproc", "--run-dir",
                          str(tmp_path / "r3"), env={"CAM_LIVE_HOME": home})
    assert code == 3 and "separated services" in err and not os.path.exists(tmp_path / "r3")
    # a task file with strategy content is refused
    bad = tmp_path / "bad.yaml"
    bad.write_text(open(LAB).read() + "\nplaybook: anything\n")
    code, out, err = _cli("run", "--task", str(bad), "--no-doctor", "--inproc", env={"CAM_LIVE_HOME": home})
    assert code == 3 and "playbook" in err
    # validate is for staging only, and components that are not installed are reported as such (exit 4), not faked
    code, _, err = _cli("validate", "--mode", "live")
    assert code == 2 and "never by a validation run" in err
    ck = tmp_path / "ck"
    ck.mkdir()
    for args, module in ((("validate", "--mode", "staging"), "cam_live.validation"),
                         (("evaluate-transfer", "--checkpoint", str(ck)), "cam_live.eval.transfer")):
        try:
            __import__(module)
            continue                                     # the component has landed; its own tests cover it
        except ImportError:
            code, _, err = _cli(*args)
            assert code == 4 and "not present in this installation" in err, args
    code, _, err = _cli("evaluate-transfer", "--checkpoint", str(tmp_path / "missing"))
    assert code == 2


def test_cli_inspect_harness_lists_tools_permissions_and_missing_integrations():
    code, out, _ = _cli("inspect-harness", "--lab", "--task", DISCORD, "--json")
    assert code == 0
    info = json.loads(out)
    names = {t["name"] for t in info["tools"]}
    assert {"lab.act", "lab.probe", "wait"} <= names and info["capabilities"]["live"] is False
    assert all({"modality", "auth_scope", "side_effects", "available"} <= set(t) for t in info["tools"])
    missing = {m["tool"] for m in info["missing_integrations"]}
    assert {"interface:structured_browser", "interface:visual", "interface:code", "interface:skill"} <= missing
    assert info["permissions"]["policy_engine"] == "AllowAllPolicy"
    assert "lab.write" in info["permissions"]["scopes_requested_by_tools"]


def test_cli_test_command_records_junit(tmp_path):
    suite = os.path.join("runtime", "test_rt_tasks.py")
    code, out, err = _cli("test", "--suite", suite, "-k", "second_objective", "--results-dir", str(tmp_path))
    assert code == 0, err
    xml = tmp_path / "cli_runtime_test_rt_tasks.py.xml"
    assert xml.exists() and 'failures="0"' in xml.read_text() and "pytest exit code 0" in out
    code, _, err = _cli("test", "--suite", "no_such_suite")
    assert code == 2


def _wait(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_cli_persistent_run_status_pause_resume_export_stop(tmp_path):
    """The runtime as an independently persistent process, driven only through the command line."""
    home, run_dir = str(tmp_path / "home"), str(tmp_path / "run")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"seed": 4, "home": home, "backbone": {"name": "tiny", "revision": ""},
                                   "runtime": {"sleeper": "wall", "control_poll_s": 0.1, "min_idle_sleep_s": 0.0},
                                   "lab_agent": {"only_ops": ["lab.act", "wait"], "wait_seconds": 86400.0}}))
    common = ("--config", str(cfg), "--run-dir", run_dir)
    factory = ("--agent-factory", "rt_lab_agent:build_agent")
    try:
        code, out, err = _cli("init", "--config", str(cfg), *factory, "--json")
        assert code == 0, err
        init = json.loads(out)
        assert os.path.exists(os.path.join(init["checkpoint"]["path"], "agent.json")) and os.path.exists(init["config"])
        assert os.stat(init["dashboard_token_file"]).st_mode & 0o777 == 0o600

        code, out, err = _cli("run", "--task", LAB, "--inproc", "--no-doctor", *common, *factory, "--json")
        assert code == 0, err
        started = json.loads(out)
        pid = started["runtime_pid"]
        assert started["started"] and pid and pid != os.getpid() and pid_alive(pid)
        assert os.getsid(pid) != os.getsid(0)                               # detached from this session
        # it acts, then waits (a real sleep, interruptible) — the process stays alive
        assert _wait(lambda: (read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}).get("activity", {}).get("phase")
                     == "sleeping")
        code, out, _ = _cli("status", *common, "--json")
        st = json.loads(out)
        assert code == 0 and st["status"] == "running" and st["process"]["alive"] and st["objective"]["id"] == "lab-staging"
        assert st["counters"]["decisions"] >= 1 and st["verified"]["status"] == "pending"
        code, out, _ = _cli("status", "--run-dir", run_dir)
        assert code == 0 and "objective:" in out and "verified:" in out and "learning:" in out

        code, out, err = _cli("pause", *common, "--reason", "test", "--json")
        res = json.loads(out)["result"]
        assert code == 0 and res["ok"] and res["status"] == "paused", err
        st = read_json(os.path.join(run_dir, RunFiles.STATE))
        assert st["status"] == "paused" and os.path.exists(os.path.join(st["committed"]["ckpt"]["path"], "agent.json"))
        decisions_paused = st["counters"]["decisions"]

        # infrastructure interruption while paused: the process is killed outright
        os.kill(pid, signal.SIGKILL)
        assert _wait(lambda: not pid_alive(pid))
        code, out, err = _cli("resume", *common, *factory, "--json")
        assert code == 0, err
        resumed = json.loads(out)
        assert resumed["restarted"] and resumed["restored"]["checkpoint"] == st["committed"]["ckpt"]["id"]
        new_pid = None
        assert _wait(lambda: pid_alive(read_pid(run_dir)) and read_pid(run_dir) != pid)
        new_pid = read_pid(run_dir)
        assert _wait(lambda: (read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}).get("status") == "running")
        st2 = read_json(os.path.join(run_dir, RunFiles.STATE))
        assert st2["counters"]["decisions"] >= decisions_paused        # the persistent state was restored, not reset
        assert len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_started"])) == 1

        code, out, err = _cli("export-checkpoint", *common, "--json")
        assert code == 0 and json.loads(out)["result"]["ok"], err
        exports = read_json(os.path.join(run_dir, RunFiles.STATE))["exports"]
        assert exports and os.path.exists(os.path.join(exports[-1]["path"], "agent.json"))

        code, out, err = _cli("stop", *common, "--reason", "finished", "--json")
        assert code == 0 and json.loads(out)["result"]["status"] == "stopped", err
        assert _wait(lambda: not pid_alive(new_pid))                     # the process ends after a stop
        # with no process: the committed checkpoint can still be exported, and the journal regenerated
        code, out, err = _cli("export-checkpoint", *common, "--out", str(tmp_path / "offline"), "--json")
        meta = json.loads(out)
        assert code == 0 and os.path.exists(tmp_path / "offline" / "agent.json") and meta["checkpoint_id"], err
        code, out, _ = _cli("journal", *common, "--print")
        assert code == 0 and "[FACT]" in out and "Run status: stopped" in out
        code, out, _ = _cli("status", *common, "--json")
        assert json.loads(out)["status"] == "stopped" and json.loads(out)["process"]["alive"] is False
    finally:
        for p in {read_pid(run_dir)}:
            if p and pid_alive(p) and p != os.getpid():
                os.kill(p, signal.SIGKILL)
