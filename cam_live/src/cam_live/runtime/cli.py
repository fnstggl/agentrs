"""cam_live.runtime.cli — the ``cam-live`` command (spec Part 17).

    cam-live doctor                         inspect hardware, models, storage, dependencies, harness
    cam-live init --config C                create the state root, the initial learner checkpoint, the dashboard token
    cam-live test [--suite S] [-m MARK]     run the acceptance tests (JUnit XML under docs/evidence/test_results/)
    cam-live validate --mode staging [--debug|--real-model] [--decisions N] [--out D] [--seed S]
                                            exercise the complete runtime against the controlled services
    cam-live inspect-harness                list tools, permissions and missing integrations
    cam-live preflight --task T             the readiness report alone
    cam-live run --task T                   preflight, then start the persistent runtime (detached, pid file)
    cam-live status | pause | resume | stop | approve ID | reject ID | journal | dashboard
    cam-live plan-report [--run-dir R] [--out F] [--json]
                                            CAM's objective, notes, predictions, experiments, operator requests,
                                            actions and learning updates, verbatim with provenance (no summary)
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
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..config import GPU_LOCK_PATH, PROJECT_ROOT, REPO_ROOT, CamLiveConfig
from ..contracts import RunFiles, WallClock, new_id
from ..harness import SOCK_AUTH, SOCK_EXEC, SOCK_VERIFY, users_exist
from . import doctor as _doctor
from . import preflight as _preflight
from .control import PID_FILE, OperatorControl, OperatorKey, pid_alive, read_pid
from .logs import RunReader, atomic_write_json, atomic_write_text, read_json
from .tasks import TaskValidationError, has_discord, load_task, load_task_raw, strategy_findings

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


def _accepted(fn: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
    """The keyword arguments of ``kw`` that ``fn`` accepts (an installed hook may be older than this command)."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kw)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in params}


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


def _absent(module: str, e: BaseException) -> bool:
    """True when ``e`` says that ``module`` itself (or a package above it) is not installed — not that it exists
    and failed to import."""
    if not isinstance(e, ModuleNotFoundError):
        return False
    name = getattr(e, "name", "") or ""
    return bool(name) and (module == name or module.startswith(name + "."))


def load_hook(spec: str, what: str) -> Any:
    """``module:attribute`` of a component other workstreams provide (``cam_live.agent:build_agent``,
    ``cam_live.validation:run_staging``, ``cam_live.eval.transfer:main``). Raises :class:`ComponentMissing` with a
    clear "not available yet" message only when the module is not installed; an import error *inside* an existing
    module is reported as such."""
    module, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        if _absent(module, e):
            raise ComponentMissing(f"{what} is not available yet: module {module} is not in this installation "
                                   f"(needed: {spec})") from e
        raise ComponentBroken(f"{what}: module {module} exists but failed to import: {type(e).__name__}: {e}") from e
    if not hasattr(mod, attr):
        raise ComponentBroken(f"{what}: module {module} is installed but does not define {attr} (needed: {spec})")
    return getattr(mod, attr)


class ComponentMissing(RuntimeError):
    """A component of another workstream is not installed (exit code 4, "not available yet")."""


class ComponentBroken(RuntimeError):
    """A component is installed but cannot be used (exit code 4, with the real error)."""


AGENT_HOOK = "cam_live.agent:build_agent"
VALIDATION_HOOK = "cam_live.validation:run_staging"
TRANSFER_HOOK = "cam_live.eval.transfer:main"
COMPLETION_HOOK = "cam_live.verify_completion:main"


def _hook_failed(e: BaseException) -> int:
    print(f"cam-live: {e}", file=sys.stderr)
    return 4


def _clean(text: Any) -> str:
    """Terminal-safe text: learner- or environment-derived strings never carry control sequences to the terminal."""
    import re
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "?", str(text))


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
    from ..harness import safefs
    with safefs.open_append(log_path, 0o640) as log:          # never through a planted symlink / FIFO (R4 sweep)
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env or dict(os.environ), cwd=REPO_ROOT)
    return p.pid


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    src = os.path.join(PROJECT_ROOT, "src")
    env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _tail(path: str, n: int = 30) -> str:
    from ..harness import safefs
    try:
        return "".join(safefs.read_nofollow(path).decode("utf-8", "replace").splitlines(True)[-n:])
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
    from .loop_service import DEFAULT_AGENT_FACTORY, call_factory
    factory = a.agent_factory or cfg.section("runtime").get("agent_factory") or DEFAULT_AGENT_FACTORY
    try:
        build = load_hook(factory, "the agent factory")
    except (ComponentMissing, ComponentBroken) as e:
        report["agent"] = f"not initialised: {e}"
        _out(report, a.json, "\n".join(f"{k}: {v}" for k, v in report.items()))
        return _hook_failed(e)
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
        run_staging = load_hook(VALIDATION_HOOK, "staging validation")
    except (ComponentMissing, ComponentBroken) as e:
        return _hook_failed(e)
    kw: Dict[str, Any] = {"real_model": not a.debug, "seed": int(a.seed)}
    if a.decisions:
        kw["n_decisions"] = int(a.decisions)
    if a.out:
        kw["out_dir"] = os.path.abspath(a.out)
    if a.evidence_dir:
        kw["evidence_dir"] = os.path.abspath(a.evidence_dir)
    if getattr(a, "budget_s", 0.0):
        kw["budget_s"] = float(a.budget_s)
    if getattr(a, "meta_min_independent", 0):
        kw["meta_min_independent"] = int(a.meta_min_independent)
    if getattr(a, "computer_share", 0.0):
        kw["computer_share"] = float(a.computer_share)
    if getattr(a, "lab_lifetimes", 0):
        kw["lab_lifetimes"] = int(a.lab_lifetimes)
    if getattr(a, "compute_limit_s", 0.0):
        kw["compute_limit_s"] = float(a.compute_limit_s)
    report = run_staging(cfg, **_accepted(run_staging, kw))
    out = os.path.join(kw.get("out_dir") or os.path.join(cfg.resolved_home(), "validation"), "staging_report.json")
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
        task = load_task(a.task, for_run=False) if a.task else None
        info["missing_integrations"] += _preflight.missing_integrations(task, tools)
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


ACCEPT_EPHEMERAL = "--accept-ephemeral-storage"


def _accepted_risks(a: argparse.Namespace) -> List[str]:
    """Risks the operator acknowledged on THIS command line (never from a file or the environment)."""
    return ["persistent_storage"] if getattr(a, "accept_ephemeral_storage", False) else []


def _risk_record(rep: Dict[str, Any], a: argparse.Namespace) -> List[Dict[str, Any]]:
    """What run.json records: each accepted risk with the finding that was accepted, by whom, when."""
    now = WallClock().now()
    return [{"check": m["item"], "finding": m["detail"], "accepted_by": m["accepted_by"], "accepted_at": now,
             "reason": "operator acknowledgement on the command line: the state root is not on a persistent volume; "
                       "checkpoints and logs must be copied off the machine to survive it"}
            for m in (rep.get("live") or {}).get("accepted_risks", [])]


def cmd_preflight(a: argparse.Namespace) -> int:
    rep = _preflight.preflight(a.task, _cfg(a.config), with_doctor=not a.no_doctor, accept_risks=_accepted_risks(a))
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
    rep = _preflight.preflight(a.task, cfg, with_doctor=not a.no_doctor, accept_risks=_accepted_risks(a))
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
    sup = read_json(os.path.join(run_dir, SUPERVISOR_PID))
    if pid_alive(read_pid(run_dir)) or pid_alive(sup if isinstance(sup, int) else None):
        print(f"cam-live run: a runtime or its supervisor is already running in {run_dir}", file=sys.stderr)
        return 3
    os.makedirs(run_dir, exist_ok=True)
    task_copy = os.path.join(run_dir, "task.yaml")
    if os.path.abspath(a.task) != task_copy:
        from ..harness import safefs
        with open(a.task, "rb") as f:                         # the operator's file
            safefs.write_atomic(task_copy, f.read(), 0o644)    # into the run directory: never through a planted link
    atomic_write_json(os.path.join(run_dir, "run.json"),
                      {"task_file": os.path.abspath(a.task), "config": os.path.abspath(a.config) if a.config else "",
                       "agent_factory": a.agent_factory, "separated": separated, "mode": task.mode,
                       "authorization": os.path.abspath(a.authorization) if a.authorization else "",
                       "verifier_source": a.verifier_source, "created_at": WallClock().now(),
                       "accepted_risks": _risk_record(rep, a) if task.mode == "live" else [],
                       "accept_ephemeral_storage_given": bool(_accepted_risks(a))})
    OperatorKey(run_dir).ensure()          # the operator's command key exists before any service or learner starts
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
        if _accepted_risks(a):
            argv.append(ACCEPT_EPHEMERAL)          # the operator's acknowledgement on this command line, passed on
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
    while time.monotonic() < end and pid_alive(pid) and not (os.path.exists(state_path) and read_pid(run_dir)):
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
        extra["exec"] += ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", task.mode,
                          "--task", os.path.join(run_dir, "task.yaml")]          # E-1: Discord / publishing config
    if "--run-dir" not in extra["exec"]:
        # the executor's own state (workspace, execution log) lives in <run_dir>/exec, the directory the launcher
        # creates for its uid; the run directory itself holds the runtime's records
        extra["exec"] += ["--run-dir", os.path.join(run_dir, "exec")]
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
        # HX-5: a staging task's own sites on this machine are reachable only once registered (never in live mode)
        from .tasks import register_staging_sites
        for r in register_staging_sites(auth, task):
            print(f"supervisor: registered staging site {r.get('address')} as a local service", flush=True)
        if a.authorization:
            with open(a.authorization) as f:
                auth.register_live_authorization(json.load(f), "operator")
        env = RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=120.0)
        try:
            vinfo = RemoteVerifier(os.path.join(sock, SOCK_VERIFY)).info()
        except Exception as e:  # noqa: BLE001
            vinfo = None
            print(f"supervisor: verifier not reachable: {type(e).__name__}: {e}", flush=True)
        rep = _preflight.preflight(a.task, cfg, policy=auth, env=env, verifier_info=vinfo, run_dir=run_dir,
                                   accept_risks=_accepted_risks(a))
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
    text = _clean("\n".join([
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
    ] + [f"GPU lock:   {'held' if (st.get('gpu') or {}).get('held') else 'not held'}"
         + (f", waiting {st['gpu']['waiting_for_s']} s" if (st.get('gpu') or {}).get('waiting_for_s') is not None else "")
         + f" (holds <= {(st.get('gpu') or {}).get('hold_s')} s; waited {(st.get('gpu') or {}).get('waited_s_total')} s "
           f"in total)" for _ in [0] if st.get("gpu")]
      + [f"ACCEPTED RISK: {r.get('check')} — {r.get('finding')} ({r.get('accepted_by')})"
         for r in st.get("accepted_risks", [])]))
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
         f"command {cid}: " + (f"{'ok' if res['ok'] else 'NOT applied'} — {_clean(res['detail'])} (status {res['status']})"
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
    """Resume a paused runtime; if no runtime process is alive, restart it from the persisted state. A stopped run
    (operator stop or emergency stop) is not restarted."""
    run_dir = _run_dir(a)
    if pid_alive(read_pid(run_dir)):
        return _command(a, lambda c: c.resume())
    meta = read_json(os.path.join(run_dir, "run.json"), {}) or {}
    st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
    if not st:
        print(f"cam-live resume: no runtime state in {run_dir}", file=sys.stderr)
        return 1
    if st.get("status") == "stopped":
        print(f"cam-live resume: REFUSED — the run in {run_dir} was stopped ({_clean(st.get('status_reason', ''))}); a "
              "stopped run is not restarted. Its committed checkpoint can be exported (cam-live export-checkpoint).",
              file=sys.stderr)
        return 3
    sup = read_json(os.path.join(run_dir, SUPERVISOR_PID))
    if meta.get("separated") and pid_alive(sup if isinstance(sup, int) else None):
        print(f"cam-live resume: the supervisor of {run_dir} (pid {sup}) is still running; it restarts the learner "
              "itself", file=sys.stderr)
        return 3
    if st.get("status") == "paused":
        OperatorControl(run_dir).resume(wake=False)      # signed; applied when the restarted runtime reads it
    common = ["--run-dir", run_dir, "--task", os.path.join(run_dir, "task.yaml")]
    if meta.get("config"):
        common += ["--config", meta["config"]]
    factory = a.agent_factory or meta.get("agent_factory")
    if factory:
        common += ["--agent-factory", factory]
    if meta.get("separated"):
        argv = [sys.executable, "-m", "cam_live.runtime.cli", "_supervise"] + common
        if meta.get("authorization"):
            argv += ["--authorization", meta["authorization"]]
        if meta.get("verifier_source"):
            argv += ["--verifier-source", meta["verifier_source"]]
        if _accepted_risks(a):
            argv.append(ACCEPT_EPHEMERAL)          # only when given again on this command line
        elif meta.get("accepted_risks"):
            print(f"cam-live resume: the run was started with {ACCEPT_EPHEMERAL}; a restarted live preflight blocks "
                  f"again unless it is given on this command line too", file=sys.stderr)
        log = os.path.join(run_dir, "logs", "supervisor.log")
    else:
        argv = [sys.executable, "-m", "cam_live.runtime.loop_service"] + common + ["--inproc"]
        log = os.path.join(run_dir, "logs", "runtime.log")
    before = len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"]))
    pid = _spawn_detached(argv, log, _child_env())
    if meta.get("separated"):
        atomic_write_text(os.path.join(run_dir, SUPERVISOR_PID), str(pid))
    end = time.monotonic() + a.timeout
    while time.monotonic() < end and pid_alive(pid):
        if len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"])) > before:
            break
        time.sleep(0.1)
    resumed = RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"])
    ok = len(resumed) > before
    out = {"run_dir": run_dir, "restarted": True, "launcher_pid": pid, "restored": resumed[-1]["data"] if ok else None,
           "supervisor_pid": pid if meta.get("separated") else None}
    _out(out, a.json, f"restarted the runtime in {run_dir} (pid {pid}); restored: {out['restored']}")
    if not ok:
        print(f"cam-live resume: the runtime did not come back; last lines of {log}:\n{_tail(log)}", file=sys.stderr)
        return 5
    return 0


_CKPT_NAME = re.compile(r"^ckpt_\d{1,12}$")
#: bounds of one offline checkpoint copy (entries, bytes)
_EXPORT_MAX_FILES, _EXPORT_MAX_BYTES = 20000, 256 * 1024 ** 3


class ExportRefused(RuntimeError):
    pass


def _checkpoint_names(run_dir: str) -> List[str]:
    """Names of checkpoint directories directly in ``<run>/checkpoints`` (real directories, no symlinks)."""
    from ..harness import safefs
    root = os.path.join(run_dir, RunFiles.CHECKPOINTS)
    try:
        dfd = safefs.open_dir_read(root)
    except (FileNotFoundError, NotADirectoryError):
        return []
    try:
        out = []
        for n in os.listdir(dfd):
            if not _CKPT_NAME.match(n):
                continue
            st = os.stat(n, dir_fd=dfd, follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                out.append(n)
        return sorted(out)
    finally:
        os.close(dfd)


def _copy_tree_safe(src_dfd: int, dst: str, budget: Dict[str, int]) -> None:
    """Copy the directory open as ``src_dfd`` into the new directory ``dst`` (created 0700): regular files and
    directories only, nothing followed; a symlink, FIFO, socket or device anywhere refuses the whole export."""
    os.mkdir(dst, 0o700)
    for n in sorted(os.listdir(src_dfd)):
        st = os.stat(n, dir_fd=src_dfd, follow_symlinks=False)
        budget["files"] -= 1
        if budget["files"] < 0:
            raise ExportRefused("the checkpoint has too many entries")
        if stat.S_ISDIR(st.st_mode):
            sub = os.open(n, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=src_dfd)
            try:
                _copy_tree_safe(sub, os.path.join(dst, n), budget)
            finally:
                os.close(sub)
        elif stat.S_ISREG(st.st_mode):
            fd = os.open(n, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_dfd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ExportRefused(f"{n} changed type while it was copied")
                out = os.open(os.path.join(dst, n), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    while True:
                        chunk = os.read(fd, 1 << 20)
                        if not chunk:
                            break
                        budget["bytes"] -= len(chunk)
                        if budget["bytes"] < 0:
                            raise ExportRefused("the checkpoint is larger than an export may be")
                        os.write(out, chunk)
                    os.fsync(out)
                finally:
                    os.close(out)
            finally:
                os.close(fd)
        else:
            kind = "a symlink" if stat.S_ISLNK(st.st_mode) else "a FIFO, socket or device"
            raise ExportRefused(f"the checkpoint contains {kind} ({n}); nothing is exported")


def _disk_guard(a: argparse.Namespace, path: str) -> Any:
    """The storage guard with the configuration's thresholds (section ``storage``)."""
    from ..storage import DEFAULT_MIN_FREE_GB, DEFAULT_WARN_FREE_GB, DiskGuard
    st = _cfg(getattr(a, "config", "")).section("storage") or {}
    return DiskGuard(path, float(st.get("min_free_gb", DEFAULT_MIN_FREE_GB)),
                     float(st.get("warn_free_gb", DEFAULT_WARN_FREE_GB)))


def _tree_bytes(path: str) -> int:
    n = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                n += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return n


def _export_offline(run_dir: str, a: argparse.Namespace) -> Dict[str, Any]:
    """Export the committed checkpoint of a run with no runtime process (G2-harness R4-C1). Root runs this on the
    operator's behalf over a directory the learner writes, so: the checkpoint is chosen by NAME among the real
    directories of ``<run>/checkpoints`` (the learner's state.json may only name which one, by its base name, and
    is never used as a path); it is copied with nothing followed into a private staging directory; a checkpoint
    with a MANIFEST must match its hashes; the destination is the operator's ``--out`` (or the state root's
    ``exports``), normalised, outside the run directory, and must not exist (never merged into a tree)."""
    from ..harness import safefs
    names = _checkpoint_names(run_dir)
    st = read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}
    ck = (st.get("committed") or {}).get("ckpt") or {}
    wanted = getattr(a, "checkpoint", "") or os.path.basename(str(ck.get("path") or ""))
    if wanted and not _CKPT_NAME.match(wanted):
        raise ExportRefused(f"{wanted!r} is not a checkpoint name (ckpt_<digits>)")
    if not names:
        raise ExportRefused(f"no checkpoint in {os.path.join(run_dir, RunFiles.CHECKPOINTS)}")
    name = wanted if wanted in names else ("" if getattr(a, "checkpoint", "") else names[-1])
    if not name:
        raise ExportRefused(f"no checkpoint named {wanted!r} in the run")
    run_real = os.path.realpath(run_dir)
    if a.out:
        dest = os.path.abspath(a.out)
    else:
        home = _cfg(getattr(a, "config", "")).resolved_home()
        dest = os.path.join(home, "exports", f"{os.path.basename(run_real)}_{name}")
    dest_real = os.path.realpath(dest)
    if dest_real == run_real or dest_real.startswith(run_real.rstrip("/") + "/"):
        raise ExportRefused("the export destination must be outside the run directory (the learner can write there)")
    if os.path.lexists(dest):
        if not (os.path.isdir(dest) and not os.path.islink(dest) and not os.listdir(dest)
                and os.lstat(dest).st_uid == os.geteuid()):
            raise ExportRefused(f"{dest} exists; an export is never merged into an existing tree")
        os.rmdir(dest)
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    # storage rule (G2-integration): an export needs about twice the checkpoint (staging copy + export) and must leave
    # the configured minimum free
    _disk_guard(a, parent).check(2 * _tree_bytes(os.path.join(run_dir, RunFiles.CHECKPOINTS, name)), "checkpoint export")
    import tempfile
    staging = tempfile.mkdtemp(prefix=".cam-export-", dir=parent)        # 0700, this uid's, next to the destination
    try:
        cdfd = safefs.open_dir_read(os.path.join(run_dir, RunFiles.CHECKPOINTS, name))
        try:
            copy = os.path.join(staging, "checkpoint")
            _copy_tree_safe(cdfd, copy, {"files": _EXPORT_MAX_FILES, "bytes": _EXPORT_MAX_BYTES})
        finally:
            os.close(cdfd)
        verified: Any = "no MANIFEST.json in the checkpoint (not verifiable)"
        if os.path.isfile(os.path.join(copy, "MANIFEST.json")):
            from ..credit.checkpoint import verify_checkpoint
            _manifest_paths_inside(copy)        # R5-m1: before anything stats / hashes a path the MANIFEST names
            bad = verify_checkpoint(copy)
            if bad:
                raise ExportRefused(f"the checkpoint does not match its MANIFEST: {bad[:3]}")
            verified = True
        try:
            mod = importlib.import_module("cam_live.credit.checkpoint")
            state = mod.load_checkpoint(copy)
            mod.export_weights_only(state, dest, config=(state.extra or {}).get("config") or {})
            how = ("cam_live.credit.checkpoint.export_weights_only (weights, adapters and the model configuration; "
                   "load with credit.checkpoint.load_exported_model)")
        except ExportRefused:
            raise
        except Exception as e:  # noqa: BLE001 — not a checkpoint of that format: the verified copy is the export
            how = f"complete copy of the committed checkpoint (weights-only export not applicable: {type(e).__name__})"
            if os.path.lexists(dest):
                shutil.rmtree(dest)
            os.rename(copy, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    meta = {"checkpoint_id": str(ck.get("id") or name) if os.path.basename(str(ck.get("path") or "")) == name else name,
            "checkpoint": name, "source": os.path.join(run_dir, RunFiles.CHECKPOINTS, name), "path": dest,
            "method": how, "manifest_verified": verified, "task": st.get("task"),
            "provenance": (st.get("task") or {}).get("mode") if isinstance(st.get("task"), dict) else None,
            "exported_at": WallClock().now()}
    atomic_write_json(dest.rstrip("/") + ".export.json", meta)
    return meta


def _manifest_paths_inside(copy: str) -> None:
    """R5-m1: every path a (learner-written) MANIFEST names must be relative, without ``..``, and resolve inside the
    checkpoint copy without following a symbolic link; otherwise the export is refused (root would otherwise stat
    and hash files elsewhere and print their sizes)."""
    try:
        with open(os.path.join(copy, "MANIFEST.json")) as f:
            m = json.load(f)
        files = m.get("files") if isinstance(m, dict) else None
    except (OSError, ValueError) as e:
        raise ExportRefused(f"the checkpoint's MANIFEST is unreadable ({type(e).__name__})")
    if not isinstance(files, dict):
        raise ExportRefused("the checkpoint's MANIFEST has no file list")
    root = os.path.realpath(copy)
    for rel in files:
        r = str(rel)
        parts = r.replace("\\", "/").split("/")
        if not r or os.path.isabs(r) or r.startswith("~") or ".." in parts or "\x00" in r:
            raise ExportRefused(f"the checkpoint's MANIFEST names a path outside the checkpoint: {r[:120]!r}")
        cur = root
        for part in [x for x in parts if x not in ("", ".")]:
            cur = os.path.join(cur, part)
            if os.path.islink(cur):
                raise ExportRefused(f"the checkpoint's MANIFEST names a path through a symbolic link: {r[:120]!r}")
        if not (os.path.realpath(cur) + os.sep).startswith(root + os.sep):
            raise ExportRefused(f"the checkpoint's MANIFEST names a path outside the checkpoint: {r[:120]!r}")


def cmd_export_checkpoint(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a)
    if pid_alive(read_pid(run_dir)):
        return _command(a, lambda c: c.export_checkpoint(a.out))
    from ..storage import DiskSpaceLow
    try:
        meta = _export_offline(run_dir, a)
    except (ExportRefused, OSError, DiskSpaceLow) as e:
        print(f"cam-live export-checkpoint: {e}", file=sys.stderr)
        return 1
    _out(meta, a.json, f"exported checkpoint {meta['checkpoint']} to {meta['path']} ({meta['method']})")
    return 0


def cmd_journal(a: argparse.Namespace) -> int:
    from .journal import write_journal
    path = write_journal(_run_dir(a))
    if a.print:
        from ..harness import safefs
        print(safefs.read_nofollow(path).decode("utf-8", "replace"))
    else:
        print(path)
    return 0


def cmd_plan_report(a: argparse.Namespace) -> int:
    from .plan_report import write_report
    run_dir = _run_dir(a)
    if not os.path.exists(os.path.join(run_dir, RunFiles.STATE)):
        print(f"cam-live plan-report: no runtime state in {run_dir}", file=sys.stderr)
        return 1
    from ..storage import DiskSpaceLow
    out = os.path.abspath(a.out) if a.out else os.path.join(run_dir, "plan_report." + ("json" if a.json else "md"))
    try:
        _disk_guard(a, os.path.dirname(out)).check(8 * 2 ** 20, "plan report")   # a report is a few MB at most
    except DiskSpaceLow as e:
        print(f"cam-live plan-report: {e}", file=sys.stderr)
        return 1
    path = write_report(run_dir, out, as_json=a.json)
    print(path)
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from .dashboard import load_or_create_operator_token, load_or_create_token, serve
    cfg = _cfg(a.config)
    run_dir = _run_dir(a, cfg)
    home = cfg.resolved_home()
    token = load_or_create_token(home)
    operator = None if a.view_only else load_or_create_operator_token(home)
    print(f"dashboard for {run_dir}: http://127.0.0.1:{a.port}/?token=<contents of "
          f"{os.path.join(home, 'dashboard.token')}>", flush=True)
    print("operator actions: " + ("disabled (--view-only)" if operator is None else
                                  f"enter the operator credential from {os.path.join(home, 'dashboard.operator')} "
                                  "(mode 0600; never share it with the viewing token)"), flush=True)
    serve(run_dir, token, "127.0.0.1", a.port, policy=_policy_handle(run_dir), operator_token=operator)
    return 0


def _delegate(spec: str, what: str, argv: List[str]) -> int:
    try:
        main_fn = load_hook(spec, what)
    except (ComponentMissing, ComponentBroken) as e:
        return _hook_failed(e)
    code = main_fn(argv)
    return int(code or 0)


def cmd_evaluate_transfer(a: argparse.Namespace) -> int:
    if not os.path.exists(a.checkpoint):
        print(f"cam-live evaluate-transfer: checkpoint {a.checkpoint} does not exist", file=sys.stderr)
        return 2
    return _delegate(TRANSFER_HOOK, "fresh-process transfer evaluation",
                     ["--checkpoint", a.checkpoint] + (["--config", a.config] if a.config else []) + list(a.rest))


def cmd_verify_completion(a: argparse.Namespace) -> int:
    return _delegate(COMPLETION_HOOK, "the completion checker", list(a.rest))


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
    g = p.add_mutually_exclusive_group()
    g.add_argument("--real-model", dest="debug", action="store_false", default=False,
                   help="the configured backbone and core on the GPU (default)")
    g.add_argument("--debug", "--cpu", dest="debug", action="store_true",
                   help="tiny random backbone + debug core on the CPU (plumbing check, not evidence about learning)")
    p.add_argument("--decisions", type=int, default=0, help="number of decisions (0: the validation's default, 48: "
                   "12 in the computer environment - 8 with operator approvals, then 4 under a compute limit - and "
                   "the rest shared by the LabWorld lifetimes)")
    p.add_argument("--out", default="", help="directory for the run (default: <home>/validation/staging_<ts>)")
    p.add_argument("--evidence-dir", default="", help="also write the evidence file here")
    p.add_argument("--budget-s", type=float, default=0.0, help="seconds the shared GPU lock is held at most, building "
                   "the agent and the final measurements included (0: the validation's default, 840; keep it <= 840 "
                   "on a shared GPU)")
    p.add_argument("--meta-min-independent", type=int, default=0, help="independent verified groups a reward "
                   "meta-update needs in this validation (0: the validation's default, 3; one per LabWorld lifetime)")
    p.add_argument("--lab-lifetimes", type=int, default=0, help="LabWorld lifetimes, each an independently verified "
                   "objective (0: the validation's default, 6)")
    p.add_argument("--compute-limit-s", type=float, default=0.0, help="learner seconds allowed to the computer "
                   "environment's limit objective (0: the validation's default, 8 per decision of it)")
    p.add_argument("--computer-share", type=float, default=0.0, help="share of the decisions in the computer phase")
    p.add_argument("--seed", type=int, default=0)
    p = add("inspect-harness", cmd_inspect_harness, "list available tools, permissions and missing integrations")
    p.add_argument("--run-dir", default="")
    p.add_argument("--task", default="")
    p.add_argument("--lab", action="store_true", help="inspect the diagnostic LabWorld instead of the computer environment")
    p.add_argument("--no-browser", action="store_true")
    p = add("preflight", cmd_preflight, "report staging and live readiness for a task")
    p.add_argument("--task", required=True)
    p.add_argument("--no-doctor", action="store_true")
    p.add_argument(ACCEPT_EPHEMERAL, dest="accept_ephemeral_storage", action="store_true",
                   help="live mode only: accept that the state root is not on a persistent volume (recorded)")
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
    p.add_argument(ACCEPT_EPHEMERAL, dest="accept_ephemeral_storage", action="store_true",
                   help="live mode only: accept that the state root is not on a persistent volume (recorded)")
    p.add_argument("--max-iterations", type=int, default=0)
    p.add_argument("--start-timeout", type=float, default=120.0)
    add("status", cmd_status, "report current objective, learning state and verified progress", run_dir=True)
    p = add("pause", cmd_pause, "safely pause operation (checkpoints the learner)", run_dir=True)
    p.add_argument("--reason", default="")
    p = add("resume", cmd_resume, "resume, restoring the complete persistent state if the process is gone", run_dir=True)
    p.add_argument("--agent-factory", default="")
    p.add_argument(ACCEPT_EPHEMERAL, dest="accept_ephemeral_storage", action="store_true",
                   help="live mode only: accept that the state root is not on a persistent volume (recorded)")
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
    p.add_argument("--checkpoint", default="", help="checkpoint directory name (ckpt_<digits>) when no runtime runs; "
                                                    "default: the committed one, else the newest")
    p = add("journal", cmd_journal, "regenerate the research journal", run_dir=True)
    p.add_argument("--print", action="store_true")
    p = add("plan-report", cmd_plan_report, "export CAM's objective, notes, predictions, experiments, operator requests, "
            "actions and learning updates verbatim, with provenance", run_dir=True)
    p.add_argument("--out", default="", help="output file (.md, or .json with --json; default <run-dir>/plan_report.*)")
    p = add("dashboard", cmd_dashboard, "serve the monitoring interface on 127.0.0.1", run_dir=True)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--view-only", action="store_true", help="no operator actions (no operator credential is loaded)")
    p = add("evaluate-transfer", cmd_evaluate_transfer, "run fresh-process transfer evaluations "
            "(further options are passed to cam_live.eval.transfer)")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(rest=[])
    sub.add_parser("verify-completion", help="run the executable completion checker (every option is passed to "
                   "cam_live.verify_completion; `cam-live verify-completion --help` shows the checker's own help)",
                   add_help=False).set_defaults(fn=cmd_verify_completion, rest=[])
    p = add("_supervise", cmd_supervise, "(internal) supervisor of a separated run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--agent-factory", default="")
    p.add_argument("--authorization", default="")
    p.add_argument("--verifier-source", default="")
    p.add_argument(ACCEPT_EPHEMERAL, dest="accept_ephemeral_storage", action="store_true")
    return ap


#: commands whose options belong to another component: everything after the command name is passed through
PASS_THROUGH = {"verify-completion": None, "evaluate-transfer": ("--checkpoint", "--config")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    if args and args[0] in PASS_THROUGH:
        own = PASS_THROUGH[args[0]]
        if own is None:                                     # verify-completion: the checker parses everything
            a = argparse.Namespace(fn=cmd_verify_completion, rest=args[1:])
        else:
            a, rest = ap.parse_known_args(args)
            a.rest = rest
    else:
        a = ap.parse_args(args)
    try:
        return int(a.fn(a) or 0)
    except TaskValidationError as e:
        print("cam-live: task validation failed:\n  - " + "\n  - ".join(e.problems), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
