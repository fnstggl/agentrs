"""Shared fit-time bookkeeping for E2's fitting scripts: provenance, plan references, card and report writing."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

from .registry_e2 import CARD_SCHEMA, CRUCIBLE_ROOT, MODELS_DIR, REPORTS_DIR, rel, sha256_file


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=CRUCIBLE_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def code_state() -> dict:
    dirty = [l[3:] for l in git("status", "--porcelain").splitlines()
             if "docs/v5/calibration/models/" not in l and "docs/v5/calibration/reports/" not in l]
    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": dirty[:20], "clean": not dirty}


def plan_ref(plan_file: str) -> dict:
    path = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "plans", plan_file)
    first = git("log", "--diff-filter=A", "--format=%H %cI", "--", rel(path)).splitlines()
    commit, at = (first[-1].split() if first else (None, None))
    return {"path": rel(path), "sha256": sha256_file(path), "commit": commit, "committed_at": at}


def environment() -> dict:
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform()}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and o != o:
        return None
    return o


def write_checkpoint(name: str, arrays: dict[str, np.ndarray], max_bytes: int = 5_000_000) -> dict:
    d = os.path.join(MODELS_DIR, name)
    os.makedirs(d, exist_ok=True)
    fname = f"{name}.npz"
    path = os.path.join(d, fname)
    np.savez_compressed(path, **arrays)
    size = os.path.getsize(path)
    if size > max_bytes:
        raise RuntimeError(f"checkpoint {rel(path)} is {size} bytes (> {max_bytes}); reduce the model")
    return {"file": fname, "sha256": sha256_file(path), "bytes": size, "format": "npz (allow_pickle=False)"}


def write_card(name: str, card: dict) -> str:
    card = {"schema": CARD_SCHEMA, **card}
    path = os.path.join(MODELS_DIR, name, "model_card.json")
    with open(path, "w") as fh:
        json.dump(jsonable(card), fh, indent=1)
    return path


def write_report(model_id: str, report: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{model_id}.json")
    with open(path, "w") as fh:
        json.dump(jsonable(report), fh, indent=1)
    if os.path.getsize(path) > 5_000_000:
        raise RuntimeError(f"report {rel(path)} exceeds 5 MB")
    return path


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
