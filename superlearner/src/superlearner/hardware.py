"""`superlearner doctor`: what this host actually has, checked rather than assumed."""
from __future__ import annotations

import os
import shutil
import subprocess
import time


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def hardware() -> dict:
    import torch
    gpus = []
    q = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"])
    for line in q.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            gpus.append({"name": p[0], "memory_mib": int(p[1]), "used_mib": int(p[2]), "driver": p[3]})
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = round(int(v.split()[0]) / 1e6, 1)
    pids = None
    for p in ("/sys/fs/cgroup/pids/pids.max", "/sys/fs/cgroup/pids.max"):
        if os.path.exists(p):
            pids = open(p).read().strip()
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    models = sorted(d.replace("models--", "").replace("--", "/") for d in (os.listdir(hub) if os.path.isdir(hub) else [])
                    if d.startswith("models--"))
    return {"gpus": gpus, "cpu_count": os.cpu_count(), "cpu_model": _run(["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2"]).strip(),
            "ram_gb": mem, "pids_max": pids, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "local_models": models,
            "disk_free_gb": round(shutil.disk_usage(os.path.dirname(os.path.abspath(__file__))).free / 1e9, 1)}


def isolation() -> dict:
    """Which process-isolation mechanisms this host offers for untrusted candidate code (see meta/workspace.py)."""
    from .meta.workspace import probe_isolation
    return probe_isolation()


def doctor(quick: bool = False) -> dict:
    from . import ensure_crucible
    rep: dict = {"hardware": hardware(), "checks": {}}
    t0 = time.time()
    try:
        ensure_crucible()
        import crucible
        rep["checks"]["crucible"] = {"ok": True, "path": os.path.dirname(crucible.__file__)}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["crucible"] = {"ok": False, "error": str(e)}
    try:
        rep["isolation"] = isolation()
        rep["checks"]["isolation"] = {"ok": bool(rep["isolation"].get("usable")), "level": rep["isolation"].get("level")}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["isolation"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        from .cam.agent import AgentConfig, CAMAgent, ModelChooser
        from .cam.model import CAMController, EmbStore
        from .llm.server import EmbedOnly
        from .world.core import CoreConfig, SessionCore
        from .world.tasks import task
        m = CAMController()
        m.eval()
        ag = CAMAgent(m, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False))
        ep = ag.run(SessionCore(task("found", "tiny", 1), CoreConfig(max_decisions=4)), ModelChooser(seed=0), "doctor")
        rep["checks"]["end_to_end_cpu"] = {"ok": len(ep["decisions"]) == 4, "trainable_parameters": m.n_params()["trainable"],
                                           "decisions": len(ep["decisions"])}
    except Exception as e:  # noqa: BLE001
        rep["checks"]["end_to_end_cpu"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    from .cli import DEFAULT_LLM
    if os.path.exists(DEFAULT_LLM["socket"]) and not quick:
        try:
            from .llm.server import RemoteLLM
            inf = RemoteLLM(DEFAULT_LLM["socket"], DEFAULT_LLM["cache"]).info()
            rep["checks"]["llm_server"] = {"ok": True, "model": inf["model"], "fingerprint": inf["fingerprint"],
                                           "embedding": inf.get("emb_ns")}
        except Exception as e:  # noqa: BLE001
            rep["checks"]["llm_server"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        rep["checks"]["llm_server"] = {"ok": True, "note": "not running (start with `superlearner llm-serve`)"}
    rep["ok"] = all(c.get("ok") for k, c in rep["checks"].items() if k != "isolation")
    rep["wall_s"] = round(time.time() - t0, 1)
    return rep
