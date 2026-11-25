"""Evidence records for acceptance requirements.

A gate test measures something and calls `record(requirement, metrics, ...)`. The record states where it came from
(commit, whether the tree was clean apart from evidence files, the pytest node or command, the author), the measured
metrics, and the artifacts it relied on with their hashes. It does not state whether it passed: the checker decides
that from the manifest's criteria and the metrics. `python -m crucible.acceptance.v5 run` attaches the JUnit result
of the run that produced the record.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

from .manifest import crucible_root, docs_dir

SCHEMA = "crucible.v5.evidence/1"
EVIDENCE_PATHS = ("docs/v5/evidence/", "docs/v5/reviews/", "docs/v5/ACCEPTANCE_MANIFEST.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: str | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd or crucible_root(), stderr=subprocess.DEVNULL).decode().strip()


def tree_state(cwd: str | None = None) -> dict:
    """HEAD commit and whether anything other than evidence files is uncommitted."""
    head = git("rev-parse", "HEAD", cwd=cwd)
    top = git("rev-parse", "--show-toplevel", cwd=cwd)
    prefix = os.path.relpath(cwd or crucible_root(), top)
    dirty = []
    for line in git("status", "--porcelain", "--untracked-files=all", cwd=cwd).splitlines():
        path = line[3:].strip().strip('"')
        rel = os.path.relpath(os.path.join(top, path), os.path.join(top, prefix))
        if not any(rel.startswith(p) for p in EVIDENCE_PATHS):
            dirty.append(path)
    return {"commit": head, "dirty_paths": dirty[:50], "clean": not dirty}


def gate_of(requirement: str) -> str:
    return "SP" if requirement.startswith("SP-") else requirement.split(".", 1)[0]


def evidence_path(requirement: str, root: str | None = None) -> str:
    d = os.path.join(docs_dir(root), "evidence", f"gate_{gate_of(requirement)}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{requirement}.json")


def environment() -> dict:
    from ...scale.slots import cgroup_cpus
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except Exception:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_quota": cgroup_cpus(),
            "gpu": gpu, "host": platform.node()}


def record(requirement: str, metrics: dict, *, command: str | None = None, artifacts: list[str] = (),
           notes: str = "", root: str | None = None) -> str:
    """Write the evidence record for `requirement`; returns its path. Metrics must be JSON numbers/bools/strings."""
    root = root or crucible_root()
    for k, v in metrics.items():
        if not isinstance(v, (int, float, bool, str)) or (isinstance(v, float) and v != v):
            raise TypeError(f"metric {k!r} must be a finite number, bool or string, not {v!r}")
    arts = []
    for a in artifacts:
        p = a if os.path.isabs(a) else os.path.join(root, a)
        arts.append({"path": os.path.relpath(p, root), "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
    rec = {"schema": SCHEMA, "requirement": requirement, "gate": gate_of(requirement),
           "produced_by": os.environ.get("PYTEST_CURRENT_TEST", "").split(" (")[0] or command or "unknown",
           "command": command, "author": os.environ.get("CRUCIBLE_V5_AUTHOR", "unknown"),
           "run_id": os.environ.get("CRUCIBLE_V5_RUN_ID"), "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "tree": tree_state(root), "environment": environment(), "metrics": metrics, "artifacts": arts,
           "notes": notes, "junit": None}
    path = evidence_path(requirement, root)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return path
