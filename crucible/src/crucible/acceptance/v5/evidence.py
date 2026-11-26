"""Evidence records for acceptance requirements.

A gate test measures something on the real system and calls `record(requirement, metrics, ...)`. The record states
what was measured (the metrics) and, computed here rather than asserted by the test, where it came from: the commit of
the checkout it ran in, the pytest node that produced it, every `crucible` (and test) module the process had imported
with its git blob id, `crucible` modules imported from outside the checkout, the environment (Python, CPU quota, GPU
visibility), checksums of the datasets the test declares, and the hashes of cited artifacts. It does not state whether
it passed: the checker evaluates the manifest's criteria on the metrics.

Evidence is filed only by an official run (`python -m crucible.acceptance.v5 run`): the runner clones the committed
tree into a fresh directory, marks the clone (a marker inside the clone's git directory, outside the work tree), runs
the requirement's tests there with `CRUCIBLE_V5_RUN_ID` and `CRUCIBLE_V5_EVIDENCE_OUT` set, and installs the records,
its own run record and the JUnit file into `docs/v5/evidence/`. Anywhere else (an ordinary test run, a working tree,
a run id without a matching clone marker) the record is written to a temporary directory and is never evidence.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

from .manifest import crucible_root, docs_dir

SCHEMA = "crucible.v5.evidence/2"
RUN_SCHEMA = "crucible.v5.run/1"
REVIEW_SCHEMA = "crucible.v5.review/2"
MARKER = "crucible_v5_run.json"                     # inside the clone's git dir: <clone>/.git/crucible_v5_run.json
EVIDENCE_DIR = "docs/v5/evidence"
RUNS_DIR = "docs/v5/evidence/official_runs"         # not `runs/`: that name is git-ignored repository-wide
REVIEWS_DIR = "docs/v5/reviews"
REPRO_DIR = "docs/v5/reviews/reproductions"
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_blob_id(data: bytes) -> str:
    """The id git gives a file with these bytes (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git(*args: str, cwd: str | None = None, strip: bool = True) -> str:
    out = subprocess.check_output(["git", *args], cwd=cwd or crucible_root(), stderr=subprocess.DEVNULL).decode()
    return out.strip() if strip else out


def tree_state(cwd: str | None = None) -> dict:
    """HEAD commit of the checkout and every uncommitted path (no exemptions)."""
    head = git("rev-parse", "HEAD", cwd=cwd)
    dirty = []
    for line in git("status", "--porcelain", "--untracked-files=all", cwd=cwd, strip=False).splitlines():
        if line.strip():
            dirty.append(line[3:].strip().strip('"'))
    return {"commit": head, "dirty_paths": dirty[:50], "clean": not dirty}


def gate_of(requirement: str) -> str:
    return "SP" if requirement.startswith("SP-") else requirement.split(".", 1)[0]


def evidence_relpath(requirement: str) -> str:
    return f"{EVIDENCE_DIR}/gate_{gate_of(requirement)}/{requirement}.json"


def identity() -> dict:
    """Who produced something: the declared agent name plus process facts. The session id is hashed, never stored."""
    sess = os.environ.get("CRUCIBLE_V5_SESSION") or os.environ.get("CLAUDE_CODE_SESSION_ID") or ""
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "")
    return {"agent": os.environ.get("CRUCIBLE_V5_AGENT") or os.environ.get("CRUCIBLE_V5_AUTHOR") or "",
            "session_sha256": hashlib.sha256(sess.encode()).hexdigest()[:16] if sess else "",
            "uid": os.getuid(), "user": user, "host": platform.node(), "pid": os.getpid(), "ppid": os.getppid()}


def environment() -> dict:
    try:
        from ...scale.slots import cgroup_cpus
        cpus = cgroup_cpus()
    except Exception:
        cpus = float(os.cpu_count() or 1)
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL, timeout=10).decode().strip() or None
    except Exception:
        pass
    torch_cuda = None
    if "torch" in sys.modules:                       # never import torch just to ask
        try:
            torch_cuda = bool(sys.modules["torch"].cuda.is_available())
        except Exception:
            torch_cuda = None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_quota": cpus, "gpu": gpu,
            "cuda_visible_devices": vis, "gpu_visible": bool(gpu) and vis not in ("", "-1"),
            "torch_cuda_available": torch_cuda, "host": platform.node(),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def imported_modules(root: str) -> tuple[dict[str, str], list[str]]:
    """(path relative to `root` -> git blob id) of every imported module file under `root`, and the `crucible`
    modules imported from anywhere else (which would mean the run measured code outside its checkout)."""
    root = os.path.realpath(root)
    inside, outside = {}, []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or f.endswith(".pyc"):
            continue
        rf = os.path.realpath(f)
        if rf.startswith(root + os.sep):
            try:
                with open(rf, "rb") as fh:
                    inside[os.path.relpath(rf, root)] = git_blob_id(fh.read())
            except OSError:
                pass
        elif name == "crucible" or name.startswith("crucible."):
            outside.append(f"{name}:{rf}")
    return dict(sorted(inside.items())), sorted(outside)


_DS_CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-dataset-sums.json")


def _file_sum(p: str, cache: dict) -> str:
    st = os.stat(p)
    key = f"{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        cache[key] = sha256_file(p)
    return cache[key]


def dataset_checksums(datasets: dict | None) -> dict:
    """name -> {path, files, bytes, sha256}; a directory's sha256 covers every file's relative path and content hash.
    Content hashes are cached per (path, size, mtime) in a temporary file so a 1 GB dataset is not re-read per test."""
    if not datasets:
        return {}
    try:
        cache = json.load(open(_DS_CACHE))
    except (OSError, ValueError):
        cache = {}
    out = {}
    for name, path in sorted(datasets.items()):
        if not os.path.exists(path):
            out[name] = {"path": path, "missing": True}
            continue
        if os.path.isfile(path):
            files = [("", path)]
        else:
            files = []
            for dp, dn, fn in os.walk(path):
                dn.sort()
                for f in sorted(fn):
                    full = os.path.join(dp, f)
                    files.append((os.path.relpath(full, path), full))
        h, total = hashlib.sha256(), 0
        for rel, full in files:
            h.update(f"{rel}\0{_file_sum(full, cache)}\n".encode())
            total += os.path.getsize(full)
        out[name] = {"path": path, "files": len(files), "bytes": total, "sha256": h.hexdigest()}
    try:
        tmp = f"{_DS_CACHE}.{os.getpid()}"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, _DS_CACHE)
    except OSError:
        pass
    return out


def official_run(root: str | None = None) -> dict | None:
    """The official run this process belongs to, or None. Requires the runner's environment AND the marker the
    runner wrote into the git directory of the clone this code runs from (at that clone's HEAD)."""
    rid, out = os.environ.get("CRUCIBLE_V5_RUN_ID"), os.environ.get("CRUCIBLE_V5_EVIDENCE_OUT")
    if not rid or not out:
        return None
    root = root or crucible_root()
    try:
        gd = git("rev-parse", "--absolute-git-dir", cwd=root)
        marker = json.load(open(os.path.join(gd, MARKER)))
        head = git("rev-parse", "HEAD", cwd=root)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    if marker.get("run_id") != rid or marker.get("commit") != head:
        return None
    return {"run_id": rid, "out": out, "kind": marker.get("kind", "evidence"), "commit": head}


def _check_metrics(metrics: dict) -> None:
    if not isinstance(metrics, dict) or not metrics:
        raise TypeError("metrics must be a non-empty dict")
    for k, v in metrics.items():
        if not isinstance(k, str):
            raise TypeError(f"metric name {k!r} must be a string")
        if not isinstance(v, (int, float, bool, str)) or (isinstance(v, float) and not math.isfinite(v)):
            raise TypeError(f"metric {k!r} must be a finite number, bool or string, not {v!r}")


def _manifest_binding(root: str, requirement: str) -> dict:
    try:
        m = json.load(open(os.path.join(docs_dir(root), "ACCEPTANCE_MANIFEST.json")))
        e = next(x for x in m["requirements"] if x["id"] == requirement)
    except (OSError, ValueError, StopIteration, KeyError):
        return {"requirement_sha256": None, "criteria_sha256": None, "criteria_version": None}
    return {"requirement_sha256": sha256_text(e["requirement"]), "criteria_sha256": sha256_text(canonical(e["criteria"])),
            "criteria_version": len(e.get("criteria_history", []))}


def record(requirement: str, metrics: dict, *, command: str | None = None, artifacts: list[str] = (),
           notes: str = "", datasets: dict | None = None, labels: dict | None = None, root: str | None = None) -> str:
    """Write the evidence record for `requirement`; returns its path.

    metrics: JSON numbers/bools/strings, finite. datasets: {name: path} whose checksums are recorded. labels: how the
    result was obtained, e.g. {"scripted": True} for scripted reference policies or {"learned": True}; the final report
    separates results by these labels. artifacts: files the record relies on (committed files of the checkout, or
    files written under the run's output directory, at most 5 MB each). `root` only says which checkout to describe;
    it never makes a record official."""
    _check_metrics(metrics)
    root = root or crucible_root()
    run = official_run(root)
    arts = []
    for a in artifacts:
        p = a if os.path.isabs(a) else os.path.join(root, a)
        size = os.path.getsize(p)
        rp = os.path.realpath(p)
        if run and rp.startswith(os.path.realpath(run["out"]) + os.sep):
            rel = "@run/" + os.path.relpath(rp, os.path.realpath(run["out"]))
        else:
            rel = os.path.relpath(p, root)
        arts.append({"path": rel, "sha256": sha256_file(p), "bytes": size, "too_large": size > MAX_ARTIFACT_BYTES})
    mods, outside = imported_modules(root)
    rec = {"schema": SCHEMA, "requirement": requirement, "gate": gate_of(requirement),
           "produced_by": os.environ.get("PYTEST_CURRENT_TEST", "").split(" (")[0] or command or "unknown",
           "command": command, "run_id": run["run_id"] if run else None, "official": bool(run),
           "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "tree": tree_state(root), "environment": environment(), "identity": identity(),
           "author": os.environ.get("CRUCIBLE_V5_AGENT") or os.environ.get("CRUCIBLE_V5_AUTHOR") or "",
           "binding": _manifest_binding(root, requirement),
           "modules": mods, "modules_outside_root": outside, "datasets": dataset_checksums(datasets),
           "metrics": metrics, "artifacts": arts, "labels": dict(labels or {}), "notes": notes,
           "junit": None, "run": None}
    if run:
        d = os.path.join(run["out"], "evidence", f"gate_{gate_of(requirement)}")
    else:                                        # an ordinary test run: measured, but never filed as evidence
        d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-unofficial-evidence", gate_of(requirement))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{requirement}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path
