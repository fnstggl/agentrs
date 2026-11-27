"""cam_live.runtime.loop_service — the learner process: builds the agent and runs :class:`LiveRuntime`.

    python -m cam_live.runtime.loop_service --run-dir R [--config C] [--task T] [--agent-factory M:F] [--inproc]

Separated mode (started by the launcher under the learner uid): the computer environment, the authorization
service and the verifier are other processes reached through ``RemoteEnvironment`` / ``RemotePolicy`` /
``RemoteVerifier`` on the sockets in ``R/sock``. This process holds no credential and no signing key.

In-process mode (``--inproc``; development, diagnostics and staging only — never live): everything is built in this
process. LabWorld tasks (``success.verifier: labworld``) get the diagnostic world with a simulated clock; other
tasks get the harness' in-process authorization service and computer environment.

The agent comes from ``--agent-factory`` (default ``cam_live.agent:build_agent``), called with those of
``cfg, run_dir, clock, task, verifier`` it accepts (the verifier only to *check* signatures: in separated mode it is
the ``RemoteVerifier`` client, which holds no key).

Shared GPU (INT3B). ``runtime.gpu_lock`` (``auto``: when the configuration loads a pretrained backbone on CUDA;
``on``; ``off``) makes this process lease the machine's shared GPU lock (``runtime.gpu_cycle``): taken before the
agent is built, held at most ``runtime.gpu_hold_s`` (default 840 s) at a time — at a decision boundary the runtime
commits, releases, yields ``runtime.gpu_yield_s`` and re-acquires it in the same process — and released whenever the
runtime sleeps. ``runtime.policy_file``: the authorization policy of an in-process run (default
``configs/authorization.yaml``), e.g. a staging rehearsal's policy.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ..config import CamLiveConfig
from ..contracts import SimClock, TaskSpec, WallClock
from ..harness import SOCK_AUTH, SOCK_EXEC, SOCK_VERIFY
from .loop import LiveRuntime, RuntimeConfig
from .tasks import load_task

DEFAULT_AGENT_FACTORY = "cam_live.agent:build_agent"


def load_factory(spec: str) -> Callable[..., Any]:
    mod, _, name = spec.partition(":")
    if not mod or not name:
        raise ValueError(f"factory must be given as module:callable, got {spec!r}")
    return getattr(importlib.import_module(mod), name)


def call_factory(fn: Callable[..., Any], **available: Any) -> Any:
    """Call ``fn`` with the keyword arguments it accepts out of ``available``."""
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**available)
    return fn(**{k: v for k, v in available.items() if k in params})


def build_lab(task: TaskSpec, cfg: CamLiveConfig) -> Tuple[Any, Any, Any, Any]:
    """The diagnostic LabWorld with its verifier and allow-all policy, sized from the task's own boundaries."""
    from ..diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld
    clock = SimClock()
    lab = LabConfig(seed=int(cfg.seed), target=int(task.success.get("target", 30)),
                    budget=float(task.limits.money or 100.0), horizon_s=float(task.limits.duration_s))
    world = LabWorld(lab, clock)
    return world, AllowAllPolicy(), LabVerifier(world), clock


def build_inproc(task: TaskSpec, cfg: CamLiveConfig, run_dir: str) -> Tuple[Any, Any, Any, Any]:
    """(env, policy, verifier, clock) inside this process. Never for a live task."""
    if task.mode == "live":
        raise PermissionError("a live task cannot run in-process: it needs the separated services")
    sec = cfg.section("runtime")
    if sec.get("inproc_factory"):
        return call_factory(load_factory(sec["inproc_factory"]), task=task, cfg=cfg, run_dir=run_dir)
    if task.success.get("verifier") == "labworld":
        return build_lab(task, cfg)
    from ..harness.auth.policy import PolicyService
    from ..harness.service import build_environment
    from .tasks import evidence_source
    from .verifier import build_verifier, source_from_spec
    clock = WallClock()
    policy = PolicyService(run_dir, sec.get("policy_file") or None, clock=clock)
    policy.register_task(task, "operator")
    # the environment keeps its own state and execution log under <run_dir>/exec, apart from the runtime's records
    env = build_environment(os.path.join(run_dir, "exec"), policy, provenance=task.mode, clock=clock,
                            secrets=policy.secrets_handle() if hasattr(policy, "secrets_handle") else None,
                            compose="all" if not sec.get("executors") else "minimal", task=task,   # E-2
                            executors=tuple(sec.get("executors", ())), browser=bool(sec.get("browser", True)))
    source = source_from_spec(evidence_source(task), task, run_dir)
    verifier = build_verifier(task, os.path.join(run_dir, "verifier"), clock, source,
                              log_path=os.path.join(run_dir, "verifier.jsonl"))
    return env, policy, verifier, clock


def build_remote(run_dir: str, connect_timeout: float = 60.0) -> Tuple[Any, Any, Any, Any]:
    from ..harness.service import RemoteEnvironment, RemotePolicy
    from .verifier_service import RemoteVerifier
    sock = os.path.join(run_dir, "sock")
    return (RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=connect_timeout),
            RemotePolicy(os.path.join(sock, SOCK_AUTH), connect_timeout=connect_timeout),
            RemoteVerifier(os.path.join(sock, SOCK_VERIFY)), WallClock())


def needs_gpu_lock(cfg: CamLiveConfig) -> bool:
    """``runtime.gpu_lock``: on / off / auto (a pretrained backbone on CUDA)."""
    mode = str(cfg.section("runtime").get("gpu_lock", "auto")).lower()
    if mode in ("on", "true", "1", "yes"):
        return True
    if mode in ("off", "false", "0", "no"):
        return False
    try:
        return cfg.backbone.name != "tiny" and cfg.resolved_device() == "cuda"
    except Exception:  # noqa: BLE001 — no CUDA information: nothing heavy runs on a GPU
        return False


def run_service(run_dir: str, task: TaskSpec, cfg: CamLiveConfig, *, agent_factory: str, inproc: bool,
                max_iterations: Optional[int] = None, block: Optional[bool] = None) -> str:
    lease = None
    if needs_gpu_lock(cfg):
        from .gpu_cycle import GpuLease, lease_settings
        lease = GpuLease(str(cfg.section("runtime").get("gpu_lock_path", "") or ""), **lease_settings(cfg))
        print(f"learner: waiting for the shared GPU lock {lease.path} (holds of at most {lease.hold_s:.0f} s)",
              flush=True)
        waited = lease.acquire()
        print(f"learner: shared GPU lock held (waited {waited:.0f} s)", flush=True)
    try:
        return _run_service(run_dir, task, cfg, agent_factory=agent_factory, inproc=inproc,
                            max_iterations=max_iterations, block=block, lease=lease)
    finally:
        if lease is not None:
            lease.release()
            from ..nn.gpu import free_gpu_memory
            free_gpu_memory()


def _run_service(run_dir: str, task: TaskSpec, cfg: CamLiveConfig, *, agent_factory: str, inproc: bool,
                 max_iterations: Optional[int] = None, block: Optional[bool] = None, lease: Any = None) -> str:
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    env, policy, verifier, clock = build_inproc(task, cfg, run_dir) if inproc else build_remote(run_dir)
    # the verifier is passed so that the agent can check signatures itself before an outcome supervises (M-int-1);
    # call_factory passes only the keywords a factory accepts
    agent = call_factory(load_factory(agent_factory), cfg=cfg, run_dir=run_dir, clock=clock, task=task,
                         verifier=verifier)
    wake = threading.Event()
    rt = LiveRuntime(agent, env, policy, verifier, task, run_dir, clock, RuntimeConfig.from_any(cfg), wake=wake)
    if lease is not None:
        from .gpu_cycle import attach
        attach(rt, lease)                  # bounded holds, released while sleeping; re-acquired in this process
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGUSR1, lambda *_: wake.set())
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, _f: rt.request_stop(f"signal {signal.Signals(s).name}"))
    try:
        return rt.run(max_iterations=max_iterations, block=block)
    finally:
        rt.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.runtime.loop_service", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", default="", help="unused (the learner serves no socket)")
    ap.add_argument("--config", default="")
    ap.add_argument("--task", default="", help="task file (default: <run-dir>/task.yaml)")
    ap.add_argument("--agent-factory", default="")
    ap.add_argument("--inproc", action="store_true")
    ap.add_argument("--wait-for", default="", help="start only once this file exists (written by the supervisor "
                                                   "after the operator's registrations and the preflight)")
    ap.add_argument("--wait-timeout", type=float, default=600.0)
    ap.add_argument("--max-iterations", type=int, default=0)
    a = ap.parse_args(argv)
    run_dir = os.path.abspath(a.run_dir)
    cfg = CamLiveConfig.load(a.config) if a.config else (
        CamLiveConfig.load() if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs",
                                                            "cam_live.yaml")) else CamLiveConfig())
    task = load_task(a.task or os.path.join(run_dir, "task.yaml"), for_run=True)
    if a.wait_for:
        end = time.monotonic() + a.wait_timeout
        while not os.path.exists(a.wait_for):
            if time.monotonic() > end:
                print(f"learner: {a.wait_for} did not appear; not starting", file=sys.stderr)
                return 3
            time.sleep(0.2)
    factory = a.agent_factory or cfg.section("runtime").get("agent_factory") or DEFAULT_AGENT_FACTORY
    module = factory.partition(":")[0]
    try:
        load_factory(factory)
    except ModuleNotFoundError as e:
        if e.name and (module == e.name or module.startswith(e.name + ".")):
            print(f"learner: the agent factory {factory} is not available yet: module {module} is not in this "
                  "installation", file=sys.stderr)
            return 4
        raise
    except AttributeError:
        print(f"learner: module {module} is installed but does not define {factory.partition(':')[2]}",
              file=sys.stderr)
        return 4
    status = run_service(run_dir, task, cfg, agent_factory=factory, inproc=a.inproc,
                         max_iterations=a.max_iterations or None)
    print(f"learner: run ended with status {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
