"""cam_live.runtime.cli — the ``cam-live`` command (spec Part 17).

    cam-live doctor                         inspect hardware, models, storage, dependencies, harness
    cam-live init --config C                create the state root, the initial learner checkpoint, the dashboard token
    cam-live test [--suite S] [-m MARK]     run the acceptance tests (JUnit XML under docs/evidence/test_results/)
    cam-live validate --mode staging        exercise the complete runtime against the controlled services
    cam-live inspect-harness                list tools, permissions and missing integrations
    cam-live preflight --task T             the readiness report alone
    cam-live run --task T                   preflight, then start the persistent runtime (detached, pid file)
    cam-live status | pause | resume | stop | approve ID | reject ID | journal | dashboard
    cam-live evaluate-transfer --checkpoint P
    cam-live export-checkpoint [--out P]
    cam-live verify-completion

Exit codes: 0 ok; 1 the command ran and reports a failure; 2 usage; 3 refused (preflight / authorization / task
validation); 4 a component this command needs is not present in the installation; 5 the runtime did not start.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..config import GPU_LOCK_PATH, PROJECT_ROOT, REPO_ROOT, CamLiveConfig
from ..contracts import RunFiles, WallClock, new_id
from ..harness import SOCK_AUTH, SOCK_EXEC, SOCK_VERIFY, users_exist
from . import doctor as _doctor
from . import preflight as _preflight
from .control import PID_FILE, OperatorControl, pid_alive, read_pid
from .logs import RunReader, atomic_write_json, atomic_write_text, read_json
from .tasks import TaskValidationError, load_task, load_task_raw, strategy_findings

CURRENT = "current_run"
SUPERVISOR_PID = "supervisor.pid"
TEST_SUITES_DIR = os.path.join(PROJECT_ROOT, "tests")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "docs", "evidence", "test_results")


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
def _cfg(path: str = "") -> CamLiveConfig:
    if path:
        return CamLiveConfig.load(path)
    default = os.path.join(PROJECT_ROOT, "configs", "cam_live.yaml")
    return CamLiveConfig.load(default) if os.path.exists(default) else CamLiveConfig()


def _out(obj: Any, as_json: bool, text: str = "") -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str) if as_json or not text else text)


def _run_dir(a: argparse.Namespace, cfg: Optional[CamLiveConfig] = None) -> str:
    """--run-dir, or the run started last from this state root."""
    if getattr(a, "run_dir", ""):
        return os.path.abspath(a.run_dir)
    home = (cfg or _cfg(getattr(a, "config", ""))).resolved_home()
    try:
        with open(os.path.join(home, CURRENT)) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"no run directory: give --run-dir (nothing recorded in {os.path.join(home, CURRENT)})")


def _missing(component: str, e: BaseException) -> int:
    print(f"cam-live: {component} is not present in this installation ({type(e).__name__}: {e})", file=sys.stderr)
    return 4


@contextlib.contextmanager
def gpu_lock(needed: bool, timeout_s: float = 1800.0) -> Iterator[None]:
    """The shared lock other tenants of this GPU use for heavy jobs (see ``config.GPU_LOCK_PATH``)."""
    if not needed:
        yield
        return
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    with open(GPU_LOCK_PATH, "a+") as f:
        end = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > end:
                    raise TimeoutError(f"shared GPU lock {GPU_LOCK_PATH} not acquired in {timeout_s:.0f}s")
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _policy_handle(run_dir: str) -> Any:
    """The operator's handle on the authorization service of a separated run, or None (in-process run)."""
    if not os.path.exists(os.path.join(run_dir, "sock", SOCK_AUTH)):
        return None
    from ..harness.auth.client import connect
    return connect(run_dir)


def _spawn_detached(argv: List[str], log_path: str, env: Optional[Dict[str, str]] = None) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as log:
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env or dict(os.environ), cwd=REPO_ROOT)
    return p.pid


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    src = os.path.join(PROJECT_ROOT, "src")
    env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _tail(path: str, n: int = 30) -> str:
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------------------
def cmd_doctor(a: argparse.Namespace) -> int:
    rep = _doctor.run_doctor(_cfg(a.config), a.config)
    _out(rep, a.json, _doctor.render(rep))
    return 1 if a.strict and rep["summary"]["blocking_staging"] else 0


def cmd_init(a: argparse.Namespace) -> int:
    cfg = _cfg(a.config)
    if a.home:
        cfg.home = os.path.abspath(a.home)
    home = cfg.resolved_home()
    for d in ("runs", "checkpoints", "exports", "validation"):
        os.makedirs(os.path.join(home, d), exist_ok=True)
    atomic_write_json(os.path.join(home, "config.resolved.json"),
                      {"config_path": os.path.abspath(a.config) if a.config else "", "config": cfg.to_dict()})
    from .dashboard import load_or_create_token
    load_or_create_token(home)
    report: Dict[str, Any] = {"home": home, "config": os.path.join(home, "config.resolved.json"),
                              "dashboard_token_file": os.path.join(home, "dashboard.token"), "checkpoint": None}
    if a.no_agent:
        report["agent"] = "skipped (--no-agent): no initial checkpoint was written"
        _out(report, a.json, "\n".join(f"{k}: {v}" for k, v in report.items()))
        return 0
    from .loop_service import DEFAULT_AGENT_FACTORY, call_factory, load_factory
    factory = a.agent_factory or cfg.section("runtime").get("agent_factory") or DEFAULT_AGENT_FACTORY
    try:
        build = load_factory(factory)
    except (ImportError, AttributeError) as e:
        report["agent"] = f"not initialised: {factory} is not present"
        _out(report, a.json, "\n".join(f"{k}: {v}" for k, v in report.items()))
        return _missing(f"the agent factory {factory}", e)
    ckpt = os.path.join(home, "checkpoints", "init")
    with gpu_lock(cfg.backbone.name != "tiny" and cfg.resolved_device() == "cuda"):
        agent = call_factory(build, cfg=cfg, run_dir=os.path.join(home, "runs", "_init"), clock=WallClock())
        cid = agent.save(ckpt)
        report["agent_status"] = agent.status()
    report.update(checkpoint={"id": cid, "path": ckpt}, agent=factory)
    atomic_write_json(os.path.join(home, "init.json"), report)
    _out(report, a.json, "\n".join(f"{k}: {v}" for k, v in report.items()))
    return 0


def cmd_test(a: argparse.Namespace) -> int:
    suites = a.suite or [""]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    worst = 0
    for s in suites:
        target = os.path.join(TEST_SUITES_DIR, s) if s else TEST_SUITES_DIR
        if not os.path.exists(target):
            print(f"cam-live test: no such suite {s!r} under {TEST_SUITES_DIR}", file=sys.stderr)
            return 2
        xml = os.path.join(a.results_dir or RESULTS_DIR, f"cli_{(s or 'all').replace('/', '_')}.xml")
        cmd = [sys.executable, "-m", "pytest", target, "-q", f"--rootdir={PROJECT_ROOT}", f"--junitxml={xml}"]
        if a.mark:
            cmd += ["-m", a.mark]
        if a.keyword:
            cmd += ["-k", a.keyword]
        print("$ " + " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=REPO_ROOT, env=_child_env())
        print(f"suite {s or 'all'}: pytest exit code {code}; JUnit XML: {xml}")
        worst = max(worst, 0 if code == 0 else 1)
    return worst


def cmd_validate(a: argparse.Namespace) -> int:
    if a.mode != "staging":
        print("cam-live validate exercises the controlled staging services only. Live readiness is established by "
              "`cam-live preflight` / `cam-live run`, never by a validation run.", file=sys.stderr)
        return 2
    cfg = _cfg(a.config)
    try:
        run_staging = importlib.import_module("cam_live.validation").run_staging
    except (ImportError, AttributeError) as e:
        return _missing("cam_live.validation (staging validation)", e)
    report = run_staging(cfg)
    out = os.path.join(cfg.resolved_home(), "validation", "staging_report.json")
    atomic_write_json(out, report)
    _out(report, True)
    print(f"report: {out}")
    return 0 if isinstance(report, dict) and (report.get("ok") or report.get("passed")) is True else 1


def cmd_inspect_harness(a: argparse.Namespace) -> int:
    cfg = _cfg(a.config)
    info: Dict[str, Any] = {"source": "", "tools": [], "missing_integrations": [], "permissions": {}, "capabilities": {}}
    env = policy = None
    cleanup = None
    run_dir = os.path.abspath(a.run_dir) if a.run_dir else ""
    try:
        if run_dir and os.path.exists(os.path.join(run_dir, "sock", SOCK_EXEC)):
            from ..harness.service import RemoteEnvironment
            env, policy = RemoteEnvironment(os.path.join(run_dir, "sock", SOCK_EXEC), connect_timeout=5.0), _policy_handle(run_dir)
            info["source"] = f"running services of {run_dir}"
        elif a.lab:
            from ..diagnostics.world import AllowAllPolicy, LabWorld
            env, policy = LabWorld(), AllowAllPolicy()
            info["source"] = "diagnostic LabWorld (in-process)"
        else:
            import tempfile
            from ..harness.auth.policy import PolicyService
            from ..harness.service import build_environment
            tmp = tempfile.mkdtemp(prefix="cam_live_inspect_")
            cleanup = lambda: shutil.rmtree(tmp, ignore_errors=True)  # noqa: E731
            policy = PolicyService(tmp)
            env = build_environment(tmp, policy, provenance="staging",
                                    executors=tuple(cfg.section("runtime").get("executors", ())),
                                    browser=not a.no_browser)
            info["source"] = "in-process inspection environment (staging provenance; nothing is executed)"
    except Exception as e:  # noqa: BLE001
        if cleanup:
            cleanup()
        return _missing("the computer environment / authorization service", e)
    try:
        tools = list(env.discover_tools())
        info["capabilities"] = env.get_capabilities()
        info["tools"] = [{"name": t.name, "modality": t.modality, "level": t.level, "auth_scope": t.auth_scope,
                          "side_effects": t.side_effects, "available": t.available,
                          "unavailable_reason": t.unavailable_reason, "description": t.description} for t in tools]
        info["missing_integrations"] = [{"tool": t.name, "reason": t.unavailable_reason or "listed as unavailable"}
                                        for t in tools if not t.available]
        if a.task:
            task = load_task(a.task, for_run=False)
            have = {t.modality for t in tools if t.available}
            info["missing_integrations"] += [{"tool": f"interface:{m}", "reason": "permitted by the task, but no "
                                              "available tool offers it"} for m in task.interfaces if m not in have]
        info["permissions"] = {"policy_engine": type(policy).__name__,
                               "scopes_requested_by_tools": sorted({t.auth_scope for t in tools if t.auth_scope}),
                               "limits_state": policy.limits_state() if policy is not None else None,
                               "pending_approvals": len(policy.approvals()) if policy is not None else None}
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        if cleanup:
            cleanup()
    lines = [f"harness: {info['source']}", f"{len(info['tools'])} tools:"]
    lines += [f"  {'ok ' if t['available'] else 'N/A'} {t['name']:<28} {t['modality']:<18} scope={t['auth_scope'] or '-':<16} "
              f"effects={t['side_effects']}" + ("" if t["available"] else f"  [{t['unavailable_reason']}]")
              for t in info["tools"]]
    lines.append(f"missing integrations: {len(info['missing_integrations'])}")
    lines += [f"  - {m['tool']}: {m['reason']}" for m in info["missing_integrations"]]
    lines.append(f"permissions: {json.dumps(info['permissions'], default=str)[:1500]}")
    _out(info, a.json, "\n".join(lines))
    return 0


def cmd_preflight(a: argparse.Namespace) -> int:
    rep = _preflight.preflight(a.task, _cfg(a.config), with_doctor=not a.no_doctor)
    _out(rep, a.json, _preflight.render(rep))
    return 0 if (rep["live"]["ready"] if rep.get("mode") == "live" else rep["staging"]["ready"]) else 3


def cmd_run(a: argparse.Namespace) -> int:
    cfg = _cfg(a.config)
    try:
        raw = load_task_raw(a.task)
    except (OSError, TaskValidationError) as e:
        print(f"cam-live run: {e}", file=sys.stderr)
        return 3
    mode = raw.get("mode", "staging")
    rep = _preflight.preflight(a.task, cfg, with_doctor=not a.no_doctor)
    if a.preflight_only:
        _out(rep, a.json, _preflight.render(rep))
        return 0 if (rep["live"]["ready"] if mode == "live" else rep["staging"]["ready"]) else 3
    try:
        task = load_task(a.task, for_run=True)
    except TaskValidationError as e:
        print(_preflight.render(rep), file=sys.stderr)
        print("cam-live run: REFUSED — the task cannot be run as configured:\n  - " + "\n  - ".join(e.problems),
              file=sys.stderr)
        return 3
    separated = os.geteuid() == 0 and users_exist() and not a.inproc
    if task.mode == "live":
        if not separated:
            print("cam-live run: REFUSED — live operation needs the separated services (root, service users, no "
                  "--inproc).", file=sys.stderr)
            return 3
        if not a.authorization or not os.path.isfile(a.authorization):
            print(_preflight.render(rep), file=sys.stderr)
            print("cam-live run: REFUSED — live operation needs the operator authorization record "
                  "(--authorization FILE).", file=sys.stderr)
            return 3
    home = cfg.resolved_home()
    run_dir = os.path.abspath(a.run_dir) if a.run_dir else os.path.join(
        home, "runs", a.run_id or f"{task.objective_id}-{new_id('run')[4:12]}")
    if pid_alive(read_pid(run_dir)):
        print(f"cam-live run: a runtime is already running in {run_dir} (pid {read_pid(run_dir)})", file=sys.stderr)
        return 3
    os.makedirs(run_dir, exist_ok=True)
    task_copy = os.path.join(run_dir, "task.yaml")
    if os.path.abspath(a.task) != task_copy:
        shutil.copyfile(a.task, task_copy)
    atomic_write_json(os.path.join(run_dir, "run.json"),
                      {"task_file": os.path.abspath(a.task), "config": os.path.abspath(a.config) if a.config else "",
                       "agent_factory": a.agent_factory, "separated": separated, "mode": task.mode,
                       "created_at": WallClock().now()})
    atomic_write_json(os.path.join(run_dir, "preflight.json"), rep)
    os.makedirs(home, exist_ok=True)
    atomic_write_text(os.path.join(home, CURRENT), run_dir)
    common = ["--run-dir", run_dir, "--task", task_copy] + (["--config", os.path.abspath(a.config)] if a.config else [])
    if a.agent_factory:
        common += ["--agent-factory", a.agent_factory]
    if a.foreground:
        if separated:
            print("cam-live run: --foreground is only available with --inproc", file=sys.stderr)
            return 2
        from .loop_service import main as loop_main
        return loop_main(common + ["--inproc"] + (["--max-iterations", str(a.max_iterations)] if a.max_iterations else []))
    if separated:
        argv = [sys.executable, "-m", "cam_live.runtime.cli", "_supervise"] + common
        if a.authorization:
            argv += ["--authorization", os.path.abspath(a.authorization)]
        if a.verifier_source:
            argv += ["--verifier-source", a.verifier_source]
        log = os.path.join(run_dir, "logs", "supervisor.log")
        pid = _spawn_detached(argv, log, _child_env())
        atomic_write_text(os.path.join(run_dir, SUPERVISOR_PID), str(pid))
    else:
        argv = [sys.executable, "-m", "cam_live.runtime.loop_service"] + common + ["--inproc"]
        if a.max_iterations:
            argv += ["--max-iterations", str(a.max_iterations)]
        log = os.path.join(run_dir, "logs", "runtime.log")
        pid = _spawn_detached(argv, log, _child_env())
    end = time.monotonic() + a.start_timeout
    state_path = os.path.join(run_dir, RunFiles.STATE)
    while time.monotonic() < end and not os.path.exists(state_path) and pid_alive(pid):
        time.sleep(0.1)
    started = os.path.exists(state_path)
    out = {"run_dir": run_dir, "mode": task.mode, "separated": separated, "launcher_pid": pid,
           "runtime_pid": read_pid(run_dir), "log": log, "started": started}
    _out(out, a.json, "\n".join(f"{k}: {v}" for k, v in out.items()))
    if not started:
        print(f"cam-live run: the runtime did not start; last lines of {log}:\n{_tail(log)}", file=sys.stderr)
        return 5
    return 0


def cmd_supervise(a: argparse.Namespace) -> int:
    """(internal) Root-side supervisor of a separated run: starts the services, performs the operator's
    registrations, runs the preflight against the running services, and only then lets the learner start."""
    from ..harness.auth.client import connect
    from ..harness.auth.launcher import LauncherConfig, launch
    from ..harness.service import RemoteEnvironment
    from .verifier_service import RemoteVerifier
    cfg = _cfg(a.config)
    run_dir = os.path.abspath(a.run_dir)
    task = load_task(a.task, for_run=True)
    go = os.path.join(run_dir, "learner.go")
    with contextlib.suppress(FileNotFoundError):
        os.remove(go)
    lc = LauncherConfig.coerce(cfg)
    sock = os.path.join(run_dir, "sock")
    extra = {k: list(v) for k, v in lc.extra_args.items()}
    extra.setdefault("exec", [])
    if "--auth-socket" not in extra["exec"]:
        extra["exec"] += ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", task.mode]
    extra["verifier"] = extra.get("verifier", []) + ["--task", a.task] + (
        ["--source", a.verifier_source] if a.verifier_source else [])
    extra["learner"] = extra.get("learner", []) + ["--task", a.task, "--wait-for", go] + (
        ["--config", a.config] if a.config else []) + (["--agent-factory", a.agent_factory] if a.agent_factory else [])
    lc.extra_args = extra
    handle = launch(run_dir, None, lc)
    stop = {"flag": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    code = 0
    try:
        auth = connect(run_dir)
        auth.register_task(task, "operator")
        if a.authorization:
            with open(a.authorization) as f:
                auth.register_live_authorization(json.load(f), "operator")
        env = RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=120.0)
        try:
            vinfo = RemoteVerifier(os.path.join(sock, SOCK_VERIFY)).info()
        except Exception as e:  # noqa: BLE001
            vinfo = None
            print(f"supervisor: verifier not reachable: {type(e).__name__}: {e}", flush=True)
        rep = _preflight.preflight(a.task, cfg, policy=auth, env=env, verifier_info=vinfo)
        atomic_write_json(os.path.join(run_dir, "preflight.json"), rep)
        print(_preflight.render(rep), flush=True)
        if task.mode == "live" and not rep["live"]["ready"]:
            print("supervisor: live operation is BLOCKED; nothing was started for the learner", flush=True)
            return 3
        atomic_write_text(go, "go")
        while not stop["flag"]:
            time.sleep(1.0)
            st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
            if st.get("status") == "stopped" and not pid_alive(read_pid(run_dir)):
                break
    except Exception as e:  # noqa: BLE001
        print(f"supervisor: {type(e).__name__}: {e}", flush=True)
        code = 5
    finally:
        print(json.dumps(handle.stop(), default=str), flush=True)
    return code


def cmd_status(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a)
    from .dashboard import RunView
    v = RunView(run_dir)
    st = v.status()
    if st["status"] == "no state":
        print(f"cam-live status: no runtime state in {run_dir}", file=sys.stderr)
        return 1
    learn, meta, health = v.learning(3), v.meta(3), v.health()
    rep = {"run_dir": run_dir, **st,
           "learning": {k: learn[k] for k in ("n_updates", "n_with_behavioural_change")},
           "meta_learning": {k: meta[k] for k in ("n_updates", "n_applied", "skipped_reasons", "bootstrap_mix")},
           "calibration": v.calibration(), "process": health["process"]}
    vo = st["verified"]
    text = "\n".join([
        f"run:        {run_dir}",
        f"status:     {st['status']}" + (f" — {st['status_reason']}" if st["status_reason"] else "")
        + f" (process {'alive' if health['process']['alive'] else 'not running'})",
        f"objective:  [{st['objective']['id']}, {st['objective']['mode']}] {st['objective']['text'] or '(none assigned)'}",
        f"verified:   {vo['value']} of {vo['target']} — {vo['status'] or 'no signed verification yet'}"
        f" (uncertainty {vo['uncertainty']})",
        f"activity:   {st['activity'].get('phase')} {st['activity'].get('detail', '')}",
        "resources:  " + "; ".join(f"{k} {r['used']:.6g}/{r['limit']:.6g}" for k, r in st["resources"].items()),
        f"learning:   {learn['n_updates']} updates ({learn['n_with_behavioural_change']} with a measured behavioural "
        f"change); meta: {meta['n_applied']} applied of {meta['n_updates']}, bootstrap mix {meta['bootstrap_mix']}",
        f"checkpoint: {(st['checkpoint'] or {}).get('id', 'none')}",
        f"pending:    {st['pending_outcomes']} outcomes, {st['pending_approvals']} approvals",
    ])
    _out(rep, a.json, text)
    return 0


def _command(a: argparse.Namespace, send, expect: Optional[Sequence[str]] = None) -> int:
    run_dir = _run_dir(a)
    ctl = OperatorControl(run_dir, policy=_policy_handle(run_dir))
    cid = send(ctl)
    alive = pid_alive(read_pid(run_dir))
    res = ctl.wait_result(cid, timeout_s=a.timeout if alive else 0.0)
    out = {"run_dir": run_dir, "command_id": cid, "runtime_alive": alive, "result": res}
    _out(out, getattr(a, "json", False),
         f"command {cid}: " + (f"{'ok' if res['ok'] else 'NOT applied'} — {res['detail']} (status {res['status']})"
                               if res else ("queued; the runtime has not acknowledged it yet" if alive else
                                            "queued; no runtime process is running — it is applied when one starts")))
    if res is None:
        return 1 if alive else 0
    return 0 if res["ok"] else 1


def cmd_pause(a: argparse.Namespace) -> int:
    return _command(a, lambda c: c.pause(a.reason))


def cmd_stop(a: argparse.Namespace) -> int:
    return _command(a, lambda c: c.stop(a.reason, emergency=a.emergency))


def cmd_approve(a: argparse.Namespace) -> int:
    return _command(a, lambda c: c.approve(a.approval_id, a.note))


def cmd_reject(a: argparse.Namespace) -> int:
    return _command(a, lambda c: c.reject(a.approval_id, a.note))


def cmd_resume(a: argparse.Namespace) -> int:
    """Resume a paused runtime; if no runtime process is alive, restart it from the persisted state."""
    run_dir = _run_dir(a)
    if pid_alive(read_pid(run_dir)):
        return _command(a, lambda c: c.resume())
    meta = read_json(os.path.join(run_dir, "run.json"), {}) or {}
    st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
    if not st:
        print(f"cam-live resume: no runtime state in {run_dir}", file=sys.stderr)
        return 1
    if st.get("status") == "paused":
        OperatorControl(run_dir).channel.send("resume", wake=False)
    common = ["--run-dir", run_dir, "--task", os.path.join(run_dir, "task.yaml")]
    if meta.get("config"):
        common += ["--config", meta["config"]]
    factory = a.agent_factory or meta.get("agent_factory")
    if factory:
        common += ["--agent-factory", factory]
    if meta.get("separated"):
        argv = [sys.executable, "-m", "cam_live.runtime.cli", "_supervise"] + common
        log = os.path.join(run_dir, "logs", "supervisor.log")
    else:
        argv = [sys.executable, "-m", "cam_live.runtime.loop_service"] + common + ["--inproc"]
        log = os.path.join(run_dir, "logs", "runtime.log")
    before = len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"]))
    pid = _spawn_detached(argv, log, _child_env())
    end = time.monotonic() + a.timeout
    while time.monotonic() < end and pid_alive(pid):
        if len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"])) > before:
            break
        time.sleep(0.1)
    resumed = RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"])
    ok = len(resumed) > before
    out = {"run_dir": run_dir, "restarted": True, "launcher_pid": pid, "restored": resumed[-1]["data"] if ok else None}
    _out(out, a.json, f"restarted the runtime in {run_dir} (pid {pid}); restored: {out['restored']}")
    if not ok:
        print(f"cam-live resume: the runtime did not come back; last lines of {log}:\n{_tail(log)}", file=sys.stderr)
        return 5
    return 0


def cmd_export_checkpoint(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a)
    if pid_alive(read_pid(run_dir)):
        return _command(a, lambda c: c.export_checkpoint(a.out))
    # no runtime process: export the last committed checkpoint from disk
    st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
    ckpt = (st.get("committed") or {}).get("ckpt")
    if not ckpt or not os.path.exists(ckpt["path"]):
        print(f"cam-live export-checkpoint: no committed checkpoint in {run_dir}", file=sys.stderr)
        return 1
    dest = os.path.abspath(a.out) if a.out else os.path.join(run_dir, "exports", f"export_{ckpt['id']}")
    how = "copy of the committed checkpoint"
    try:
        export = importlib.import_module("cam_live.credit.checkpoint").export_weights
        export(ckpt["path"], dest)
        how = "cam_live.credit.checkpoint.export_weights (weights only)"
    except (ImportError, AttributeError):
        if os.path.isdir(ckpt["path"]):
            shutil.copytree(ckpt["path"], dest, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(ckpt["path"], dest)
    meta = {"checkpoint_id": ckpt["id"], "source": ckpt["path"], "path": dest, "method": how,
            "task": st.get("task"), "provenance": (st.get("task") or {}).get("mode"), "exported_at": WallClock().now()}
    atomic_write_json(dest.rstrip("/") + ".export.json", meta)
    _out(meta, a.json, f"exported checkpoint {ckpt['id']} to {dest} ({how})")
    return 0


def cmd_journal(a: argparse.Namespace) -> int:
    from .journal import write_journal
    path = write_journal(_run_dir(a))
    if a.print:
        with open(path) as f:
            print(f.read())
    else:
        print(path)
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from .dashboard import load_or_create_token, serve
    cfg = _cfg(a.config)
    run_dir = _run_dir(a, cfg)
    token = load_or_create_token(cfg.resolved_home())
    print(f"dashboard for {run_dir}: http://127.0.0.1:{a.port}/?token=<contents of "
          f"{os.path.join(cfg.resolved_home(), 'dashboard.token')}>", flush=True)
    serve(run_dir, token, "127.0.0.1", a.port, policy=_policy_handle(run_dir))
    return 0


def _delegate(module: str, what: str, argv: List[str]) -> int:
    try:
        main_fn = importlib.import_module(module).main
    except (ImportError, AttributeError) as e:
        return _missing(f"{module} ({what})", e)
    code = main_fn(argv)
    return int(code or 0)


def cmd_evaluate_transfer(a: argparse.Namespace) -> int:
    if not os.path.exists(a.checkpoint):
        print(f"cam-live evaluate-transfer: checkpoint {a.checkpoint} does not exist", file=sys.stderr)
        return 2
    return _delegate("cam_live.eval.transfer", "fresh-process transfer evaluation",
                     ["--checkpoint", a.checkpoint] + (["--config", a.config] if a.config else []) + a.rest)


def cmd_verify_completion(a: argparse.Namespace) -> int:
    return _delegate("cam_live.verify_completion", "the completion checker", a.rest)


# --------------------------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cam-live", description="CAM Live 1.0 — operator command line")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_: str, run_dir: bool = False, config: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        if config:
            p.add_argument("--config", default="", help="configuration file (default: configs/cam_live.yaml)")
        if run_dir:
            p.add_argument("--run-dir", default="", help="run directory (default: the run started last)")
            p.add_argument("--timeout", type=float, default=30.0)
        p.add_argument("--json", action="store_true")
        return p

    p = add("doctor", cmd_doctor, "inspect hardware, model availability, storage, dependencies and harness configuration")
    p.add_argument("--strict", action="store_true", help="exit 1 if anything blocks staging")
    p = add("init", cmd_init, "initialise the model and the persistent research state")
    p.add_argument("--home", default="")
    p.add_argument("--agent-factory", default="")
    p.add_argument("--no-agent", action="store_true", help="create the state root only")
    p = add("test", cmd_test, "run the implementation and mathematical acceptance tests", config=False)
    p.add_argument("--suite", action="append", default=[], help="test directory under tests/ (repeatable)")
    p.add_argument("-m", dest="mark", default="")
    p.add_argument("-k", dest="keyword", default="")
    p.add_argument("--results-dir", default="")
    p = add("validate", cmd_validate, "exercise the complete runtime against controlled services")
    p.add_argument("--mode", required=True)
    p = add("inspect-harness", cmd_inspect_harness, "list available tools, permissions and missing integrations")
    p.add_argument("--run-dir", default="")
    p.add_argument("--task", default="")
    p.add_argument("--lab", action="store_true", help="inspect the diagnostic LabWorld instead of the computer environment")
    p.add_argument("--no-browser", action="store_true")
    p = add("preflight", cmd_preflight, "report staging and live readiness for a task")
    p.add_argument("--task", required=True)
    p.add_argument("--no-doctor", action="store_true")
    p = add("run", cmd_run, "start the configured experiment after authorization and preflight verification")
    p.add_argument("--task", required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--inproc", action="store_true", help="single-process mode (never for live tasks)")
    p.add_argument("--foreground", action="store_true")
    p.add_argument("--agent-factory", default="")
    p.add_argument("--authorization", default="", help="operator authorization record (required for live tasks)")
    p.add_argument("--verifier-source", default="")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--no-doctor", action="store_true")
    p.add_argument("--max-iterations", type=int, default=0)
    p.add_argument("--start-timeout", type=float, default=120.0)
    add("status", cmd_status, "report current objective, learning state and verified progress", run_dir=True)
    p = add("pause", cmd_pause, "safely pause operation (checkpoints the learner)", run_dir=True)
    p.add_argument("--reason", default="")
    p = add("resume", cmd_resume, "resume, restoring the complete persistent state if the process is gone", run_dir=True)
    p.add_argument("--agent-factory", default="")
    p = add("stop", cmd_stop, "stop the run (checkpoints the learner)", run_dir=True)
    p.add_argument("--reason", default="")
    p.add_argument("--emergency", action="store_true")
    p = add("approve", cmd_approve, "approve an action awaiting approval", run_dir=True)
    p.add_argument("approval_id")
    p.add_argument("--note", default="")
    p = add("reject", cmd_reject, "reject an action awaiting approval", run_dir=True)
    p.add_argument("approval_id")
    p.add_argument("--note", default="")
    p = add("export-checkpoint", cmd_export_checkpoint, "export trained weights, adapters and compatible configuration",
            run_dir=True)
    p.add_argument("--out", default="")
    p = add("journal", cmd_journal, "regenerate the research journal", run_dir=True)
    p.add_argument("--print", action="store_true")
    p = add("dashboard", cmd_dashboard, "serve the monitoring interface on 127.0.0.1", run_dir=True)
    p.add_argument("--port", type=int, default=8787)
    p = add("evaluate-transfer", cmd_evaluate_transfer, "run fresh-process transfer evaluations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = add("verify-completion", cmd_verify_completion, "run the executable completion checker", config=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = add("_supervise", cmd_supervise, "(internal) supervisor of a separated run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--agent-factory", default="")
    p.add_argument("--authorization", default="")
    p.add_argument("--verifier-source", default="")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except TaskValidationError as e:
        print("cam-live: task validation failed:\n  - " + "\n  - ".join(e.problems), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
