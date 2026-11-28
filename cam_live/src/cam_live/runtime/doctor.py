"""cam_live.runtime.doctor — ``cam-live doctor``: inspect hardware, models, storage, dependencies and harness.

Every check reports what was actually found on this machine. ``ok`` is True / False, or None when the check could
not be made (which is reported as unknown, never as a pass). ``needed_for`` says whether a failing check blocks
staging runs, live runs, or neither.
"""
from __future__ import annotations

import ctypes
import glob
import importlib
import importlib.util
import json
import os
import platform
import pwd
import shutil
import sys
from typing import Any, Dict, List, Optional

from ..config import GPU_LOCK_PATH, PROJECT_ROOT, CamLiveConfig
from ..harness import SERVICE_MODULES, SERVICE_USERS

DEPENDENCIES = ("torch", "transformers", "safetensors", "numpy", "yaml", "httpx", "fastapi", "uvicorn",
                "playwright", "PIL", "cryptography")
HARNESS_MODULES = ("cam_live.harness.env", "cam_live.harness.service", "cam_live.harness.browser",
                   "cam_live.harness.sandbox", "cam_live.harness.jobs", "cam_live.harness.visual",
                   "cam_live.harness.skills", "cam_live.harness.publishing", "cam_live.harness.discord",
                   "cam_live.harness.auth.policy", "cam_live.harness.auth.service", "cam_live.harness.auth.launcher",
                   "cam_live.agent", "cam_live.validation")
CAP_SYS_ADMIN, CAP_NET_ADMIN, CAP_SETUID, CAP_SETGID = 21, 12, 7, 6


def _check(name: str, ok: Optional[bool], detail: str, needed_for: str = "") -> Dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, "needed_for": needed_for}


def _meminfo() -> Dict[str, float]:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                out[k] = float(v.split()[0]) / 2**20      # GiB
    except OSError:
        pass
    return out


def _proc_status() -> Dict[str, str]:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                out[k] = v.strip()
    except OSError:
        pass
    return out


def landlock_abi() -> Optional[int]:
    """Landlock ABI version from the kernel (landlock_create_ruleset with the VERSION flag), None if unavailable."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        r = libc.syscall(444, None, 0, 1)
        return int(r) if r > 0 else None
    except (OSError, AttributeError):
        return None


def hardware_checks(cfg: CamLiveConfig) -> List[Dict[str, Any]]:
    out = []
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out.append(_check("gpu", True, f"{torch.cuda.get_device_name(0)}; {total / 2**30:.1f} GB total, "
                                           f"{free / 2**30:.1f} GB free now (shared: heavy jobs take {GPU_LOCK_PATH})",
                              "live"))
        else:
            out.append(_check("gpu", False, "CUDA is not available to torch; only the tiny CPU configuration can run",
                              "live"))
    except Exception as e:  # noqa: BLE001
        out.append(_check("gpu", None, f"could not query torch: {type(e).__name__}: {e}", "live"))
    mem = _meminfo()
    if mem:
        out.append(_check("memory", mem.get("MemAvailable", 0) >= 16, f"{mem.get('MemTotal', 0):.0f} GB RAM total, "
                                                                      f"{mem.get('MemAvailable', 0):.0f} GB available", "staging"))
    out.append(_check("cpu", True, f"{os.cpu_count()} logical CPUs; {platform.machine()}; Linux {platform.release()}"))
    return out


def storage_checks(cfg: CamLiveConfig) -> List[Dict[str, Any]]:
    home = cfg.resolved_home()
    probe = home
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    du = shutil.disk_usage(probe)
    st = (cfg.section("storage") if hasattr(cfg, "section") else {}) or {}
    from ..storage import DEFAULT_MIN_FREE_GB, DEFAULT_WARN_FREE_GB
    min_gb = float(st.get("min_free_gb", DEFAULT_MIN_FREE_GB))      # the storage rule's thresholds, not a constant
    warn_gb = float(st.get("warn_free_gb", DEFAULT_WARN_FREE_GB))
    free = du.free / 2**30
    out = [_check("disk", free >= min_gb, f"{free:.0f} GB free of {du.total / 2**30:.0f} GB at {probe} "
                                          f"(minimum {min_gb:g} GB, warning below {warn_gb:g} GB"
                                          + ("; below the warning level" if free < warn_gb else "") + ") "
                                                 f"(state root {home})", "staging"),
           _check("state_root_writable", os.access(probe, os.W_OK), f"{probe} writable by uid {os.geteuid()}", "staging")]
    persistent: Optional[bool] = None
    detail = "could not determine whether the state root survives a rebuild of this machine"
    caps = "/etc/vast_capabilities.json"
    if os.path.exists(caps):
        try:
            with open(caps) as f:
                inst = (json.load(f) or {}).get("instance", {})
            ws = os.environ.get("WORKSPACE", "/workspace")
            if "workspace_is_volume" in inst:
                under_ws = os.path.abspath(home).startswith(os.path.abspath(ws) + os.sep)
                persistent = bool(inst["workspace_is_volume"]) and under_ws
                detail = (f"the instance manifest reports workspace_is_volume={inst['workspace_is_volume']}; state root is "
                          f"{'under' if under_ws else 'outside'} {ws}. " +
                          ("State survives recycle/destroy." if persistent else
                           "State on this machine is lost on recycle/destroy: checkpoints and logs must be synced "
                           "off-box (git, object storage) to be durable."))
        except (OSError, ValueError) as e:
            detail = f"instance manifest unreadable: {e}"
    out.append(_check("persistent_storage", persistent, detail, "live"))
    return out


def model_checks(cfg: CamLiveConfig) -> List[Dict[str, Any]]:
    out = []
    bb = cfg.backbone
    if bb.name == "tiny":
        out.append(_check("backbone", False, "configured backbone is the tiny random model: CPU tests only, never a "
                                             "live or staging learner", "staging"))
    else:
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        snap = os.path.join(hub, "models--" + bb.name.replace("/", "--"), "snapshots", bb.revision)
        if os.path.isdir(snap):
            weights = glob.glob(os.path.join(snap, "*.safetensors"))
            size = sum(os.path.getsize(os.path.realpath(w)) for w in weights) / 2**30
            ok = bool(weights) and os.path.exists(os.path.join(snap, "config.json"))
            out.append(_check("backbone", ok, f"{bb.name}@{bb.revision[:12]}: {len(weights)} weight file(s), "
                                              f"{size:.1f} GB at {snap}", "staging"))
        else:
            out.append(_check("backbone", False, f"{bb.name}@{bb.revision[:12]} is not in the local cache ({snap})",
                              "staging"))
    try:
        g = importlib.import_module("cam_live.harness.grounding")
        lister = getattr(g, "doctor", None) or getattr(g, "available_grounders", None)
        if callable(lister):
            info = lister()
            out.append(_check("grounding_models", bool(info), f"{json.dumps(info, default=str)[:600]}", "live"))
        else:
            out.append(_check("grounding_models", None, "cam_live.harness.grounding is importable but offers no "
                                                        "inventory function; availability not determined", "live"))
    except Exception as e:  # noqa: BLE001
        out.append(_check("grounding_models", False, f"grounding package not importable: {type(e).__name__}: {e}", "live"))
    return out


def dependency_checks() -> List[Dict[str, Any]]:
    out = [_check("python", sys.version_info >= (3, 11), f"{platform.python_version()} at {sys.executable}", "staging")]
    for name in DEPENDENCIES:
        try:
            m = importlib.import_module(name)
            out.append(_check(f"dep:{name}", True, str(getattr(m, "__version__", "installed")), "staging"))
        except Exception as e:  # noqa: BLE001
            out.append(_check(f"dep:{name}", False, f"{type(e).__name__}: {e}", "staging"))
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or os.path.expanduser("~/.cache/ms-playwright")
    exes = [p for pat in ("chromium-*/chrome-linux*/chrome", "chromium_headless_shell-*/chrome-*/headless_shell",
                          "chromium_headless_shell-*/chrome-linux*/chrome-headless-shell")
            for p in glob.glob(os.path.join(root, pat)) if os.access(p, os.X_OK)]
    out.append(_check("chromium", bool(exes), (exes[0] if exes else f"no Playwright Chromium under {root}; "
                                                                    "run: python -m playwright install chromium"), "staging"))
    return out


def sandbox_checks() -> List[Dict[str, Any]]:
    st = _proc_status()
    cap = int(st.get("CapEff", "0"), 16) if st.get("CapEff") else 0
    has = lambda bit: bool(cap >> bit & 1)  # noqa: E731
    abi = landlock_abi()
    out = [
        _check("landlock", abi is not None, f"Landlock ABI {abi}" if abi else "Landlock is not available", "live"),
        _check("seccomp", "Seccomp" in st, f"seccomp supported (this process: mode {st.get('Seccomp', '?')})", "live"),
        _check("setuid_setgid", os.geteuid() == 0 and has(CAP_SETUID) and has(CAP_SETGID),
               f"euid {os.geteuid()}, CAP_SETUID={has(CAP_SETUID)}, CAP_SETGID={has(CAP_SETGID)} — needed to start the "
               "services under separate uids", "live"),
        _check("kernel_network_filtering", has(CAP_NET_ADMIN),
               ("CAP_NET_ADMIN present" if has(CAP_NET_ADMIN) else
                "no CAP_NET_ADMIN: per-destination network filtering cannot be enforced by the kernel here; "
                "destinations are enforced by the authorization layer only. A VM is recommended for live operation."),
               ""),
        _check("user_namespaces", has(CAP_SYS_ADMIN), f"CAP_SYS_ADMIN={has(CAP_SYS_ADMIN)}; isolation uses separate "
                                                      "uids + file modes + rlimits + Landlock + seccomp", ""),
    ]
    return out


def service_user_checks() -> List[Dict[str, Any]]:
    out = []
    for role, user in SERVICE_USERS.items():
        try:
            out.append(_check(f"user:{role}", True, f"{user} uid {pwd.getpwnam(user).pw_uid}", "live"))
        except KeyError:
            out.append(_check(f"user:{role}", False, f"{user} does not exist (created by the launcher when run as root)",
                              "live"))
    return out


def harness_checks(cfg: CamLiveConfig, config_path: str = "") -> List[Dict[str, Any]]:
    out = []
    path = config_path or os.path.join(PROJECT_ROOT, "configs", "cam_live.yaml")
    out.append(_check("config_file", os.path.exists(path), path if os.path.exists(path) else f"{path} not found; "
                                                                                             "built-in defaults in use", ""))
    for mod in HARNESS_MODULES + tuple(SERVICE_MODULES.values()):
        try:
            found = importlib.util.find_spec(mod) is not None
            out.append(_check(f"module:{mod}", found, "present" if found else "not present in this installation",
                              "staging"))
        except Exception as e:  # noqa: BLE001
            out.append(_check(f"module:{mod}", False, f"{type(e).__name__}: {e}", "staging"))
    lock_dir = os.path.dirname(GPU_LOCK_PATH)
    out.append(_check("gpu_lock", os.path.isdir(lock_dir), f"shared GPU lock file {GPU_LOCK_PATH}", ""))
    return out


def run_doctor(cfg: Optional[CamLiveConfig] = None, config_path: str = "") -> Dict[str, Any]:
    cfg = cfg or CamLiveConfig()
    sections = {"hardware": hardware_checks(cfg), "storage": storage_checks(cfg), "models": model_checks(cfg),
                "dependencies": dependency_checks(), "sandbox": sandbox_checks(), "service_users": service_user_checks(),
                "harness": harness_checks(cfg, config_path)}
    flat = [c for s in sections.values() for c in s]
    blocking = lambda mode: [c["name"] for c in flat  # noqa: E731
                             if c["ok"] is not True and (c["needed_for"] == mode
                                                         or (mode == "live" and c["needed_for"] == "staging"))]
    return {"sections": sections,
            "summary": {"checks": len(flat), "ok": sum(1 for c in flat if c["ok"] is True),
                        "failed": [c["name"] for c in flat if c["ok"] is False],
                        "unknown": [c["name"] for c in flat if c["ok"] is None],
                        "blocking_staging": blocking("staging"), "blocking_live": blocking("live")}}


def render(report: Dict[str, Any]) -> str:
    lines = []
    for sec, checks in report["sections"].items():
        lines.append(f"[{sec}]")
        for c in checks:
            mark = {True: "ok  ", False: "FAIL", None: "??  "}[c["ok"]]
            lines.append(f"  {mark} {c['name']}: {c['detail']}")
    s = report["summary"]
    lines.append(f"{s['ok']}/{s['checks']} checks ok; failed: {', '.join(s['failed']) or 'none'}; "
                 f"unknown: {', '.join(s['unknown']) or 'none'}")
    lines.append(f"blocking for staging: {', '.join(s['blocking_staging']) or 'nothing'}")
    lines.append(f"blocking for live:    {', '.join(s['blocking_live']) or 'nothing found by doctor (see preflight)'}")
    return "\n".join(lines)
