"""cam_live.storage — disk-space guard and checkpoint retention for long-running work. MASTER-OWNED.

On 2026-11-27 evaluation runs filled the instance's 300 GB disk with per-lifetime checkpoints and optimizer states;
every process on the machine (other tenants included) stopped mid-write. Two rules follow:

1. **Check before writing.** Anything that writes checkpoints or large artifacts calls :meth:`DiskGuard.check`
   first. Below ``min_free_gb`` it raises :class:`DiskSpaceLow` *before* the write starts, so the caller can pause
   cleanly (runtime: pause the run with a resource obstacle; evaluation: stop scheduling runs) instead of producing
   a torn checkpoint. Between ``min_free_gb`` and ``warn_free_gb`` it returns a warning the caller logs.
2. **Retain little.** :func:`apply_retention` keeps the newest ``keep`` checkpoints of every sibling group with
   optimizer state only in the newest ``keep_optim`` (the policy of ``tools/prune_runs.py``); protected checkpoints
   (``ckpt_pre`` / ``ckpt_post`` / ``export_*`` / ``LATEST`` targets) keep their weights. Records (JSONL, SQLite
   experience stores, ledgers, results) are never touched.

Thresholds: ``CAM_LIVE_MIN_FREE_GB`` (default 20) and ``CAM_LIVE_WARN_FREE_GB`` (default 50).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MIN_FREE_GB = float(os.environ.get("CAM_LIVE_MIN_FREE_GB", "20"))
DEFAULT_WARN_FREE_GB = float(os.environ.get("CAM_LIVE_WARN_FREE_GB", "50"))
GB = 1024 ** 3


class DiskSpaceLow(OSError):
    """Free space at a write location is below the configured minimum; nothing was written."""

    def __init__(self, path: str, free_gb: float, need_gb: float, minimum_gb: float):
        super().__init__(f"only {free_gb:.1f} GB free at {path}; {need_gb:.1f} GB needed for this write and "
                         f"{minimum_gb:.1f} GB must stay free — not writing")
        self.path, self.free_gb, self.need_gb, self.minimum_gb = path, free_gb, need_gb, minimum_gb


def free_gb(path: str) -> float:
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free / GB


@dataclass
class DiskGuard:
    path: str
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    warn_free_gb: float = DEFAULT_WARN_FREE_GB

    def check(self, need_bytes: int = 0, what: str = "write") -> Dict[str, Any]:
        """Raise :class:`DiskSpaceLow` if ``need_bytes`` would leave less than ``min_free_gb``; otherwise return
        ``{"free_gb", "level": "ok" | "warn", "what"}``."""
        free = free_gb(self.path)
        need = need_bytes / GB
        if free - need < self.min_free_gb:
            raise DiskSpaceLow(self.path, free, need, self.min_free_gb)
        return {"free_gb": round(free, 2), "need_gb": round(need, 3), "what": what,
                "level": "warn" if free - need < self.warn_free_gb else "ok"}

    def status(self) -> Dict[str, Any]:
        free = free_gb(self.path)
        level = "low" if free < self.min_free_gb else "warn" if free < self.warn_free_gb else "ok"
        return {"free_gb": round(free, 2), "min_free_gb": self.min_free_gb, "warn_free_gb": self.warn_free_gb,
                "level": level}


def dir_bytes(path: str) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _prune_module() -> Any:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "tools", "prune_runs.py")
    spec = importlib.util.spec_from_file_location("cam_live_prune_runs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def retention_plan(root: str, *, keep: int = 2, keep_optim: int = 1, drop_protected_optim: bool = False,
                   frozen_globs: Tuple[str, ...] = (), drop_groups: Tuple[str, ...] = ()) -> List[Tuple[str, str, int]]:
    if keep < 1:
        raise ValueError("keep must be >= 1")
    return _prune_module().plan(os.path.abspath(root), keep, keep_optim, drop_protected_optim, list(frozen_globs),
                                list(drop_groups))


def apply_retention(root: str, *, keep: int = 2, keep_optim: int = 1, drop_protected_optim: bool = False,
                    frozen_globs: Tuple[str, ...] = (), drop_groups: Tuple[str, ...] = (),
                    dry_run: bool = False) -> Dict[str, Any]:
    """Apply the retention policy under ``root``. Returns ``{"removed": n, "bytes": freed, "paths": [...]}``."""
    actions = retention_plan(root, keep=keep, keep_optim=keep_optim, drop_protected_optim=drop_protected_optim,
                             frozen_globs=frozen_globs, drop_groups=drop_groups)
    if not dry_run:
        for _act, path, _b in actions:
            shutil.rmtree(path, ignore_errors=True)
    return {"removed": len(actions), "bytes": sum(b for _a, _p, b in actions),
            "paths": [p for _a, p, _b in actions], "dry_run": dry_run}


def guard_for(run_dir: str, cfg: Optional[Any] = None) -> DiskGuard:
    """The guard for a run directory, thresholds from config section ``storage`` when given."""
    sec = cfg.section("storage") if cfg is not None and hasattr(cfg, "section") else {}
    return DiskGuard(run_dir, float(sec.get("min_free_gb", DEFAULT_MIN_FREE_GB)),
                     float(sec.get("warn_free_gb", DEFAULT_WARN_FREE_GB)))
