"""Official runs and independent reviews.

`run`: clone the committed tree (default HEAD; uncommitted work is never measured) into a fresh directory, mark the
clone, collect (`pytest --collect-only`) and run the requirement's test files there with the clone's own `src` on
PYTHONPATH, then install into the checkout: the evidence records, the run record (commit, clone state before and after,
collected node ids, exit codes, environment, identity) and the JUnit file whose suite name carries the run id.

`review`: an independent reviewer (a different agent name and session from the evidence's producer) re-executes the
requirement's tests in another fresh clone of the evidence commit, compares the reproduced metrics with the evidence
within the requirement's declared tolerance, checks the criteria on the reproduction, and writes a review bound to the
requirement text, criteria version, evidence hash and commit, plus the reproduction's own run record, JUnit and
records. The reviewer commits them; the checker re-verifies all of it (`checker.py`).

Both commands only write inside `docs/v5/evidence/` and `docs/v5/reviews/` of the checkout they are run from.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from . import manifest as M
from .evidence import (MARKER, REPRO_DIR, REVIEW_SCHEMA, RUN_SCHEMA, RUNS_DIR, canonical, evidence_relpath, gate_of,
                       identity, sha256_file, sha256_text)

DEFAULT_REL, DEFAULT_ABS = 1e-6, 1e-9


def _git(cwd: str, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def require_agent(agent: str | None) -> str:
    a = agent or os.environ.get("CRUCIBLE_V5_AGENT") or ""
    if not a.strip():
        raise SystemExit("an official run or review needs an agent name: --agent NAME or CRUCIBLE_V5_AGENT")
    return a.strip()


def clone(root: str, commit: str, dest: str, run_id: str, kind: str) -> str:
    """Fresh clone of the repository containing `root`, checked out (detached) at `commit`; returns the clone's
    equivalent of `root`. The marker lives in the clone's git directory, so the work tree stays clean."""
    top = _git(root, "rev-parse", "--show-toplevel")
    prefix = os.path.relpath(os.path.realpath(root), os.path.realpath(top))
    sha = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    _git(dest if os.path.isdir(dest) else os.path.dirname(dest), "clone", "--quiet", "--shared", "--no-checkout",
         top, dest)
    _git(dest, "checkout", "--quiet", "--detach", sha)
    gd = _git(dest, "rev-parse", "--absolute-git-dir")
    with open(os.path.join(gd, MARKER), "w") as fh:
        json.dump({"run_id": run_id, "commit": sha, "kind": kind, "created_at": _now()}, fh)
    return os.path.join(dest, prefix) if prefix != "." else dest


def _status(cwd: str) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd, capture_output=True,
                         text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _child_env(clone_root: str, run_id: str, out: str, agent: str) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CRUCIBLE_V5_") or k.startswith("PYTEST_") or k == "PYTHONPATH")}
    env.update({"PYTHONPATH": os.path.join(clone_root, "src"), "PYTHONDONTWRITEBYTECODE": "1",
                "CRUCIBLE_V5_RUN_ID": run_id, "CRUCIBLE_V5_EVIDENCE_OUT": out, "CRUCIBLE_V5_AGENT": agent,
                "CRUCIBLE_V5_AUTHOR": agent})
    for k in ("CRUCIBLE_V5_SESSION",):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def parse_collected(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if "::" in ln and not ln.startswith(" ") and
            not ln.startswith("=") and not ln.lower().startswith(("error", "warning"))]


def execute(root: str, tests: list[str], *, commit: str = "HEAD", kind: str = "evidence", agent: str,
            requirement_ids: list[str], scratch: str | None = None, keep: bool = False,
            pytest_args: list[str] = ()) -> dict:
    """Clone, collect, run. Returns {"record": run record, "out": output dir, "junit": path} (files not installed)."""
    run_id = uuid.uuid4().hex[:16]
    base = scratch or os.environ.get("CRUCIBLE_V5_SCRATCH") or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"crucible-v5-{kind}-{run_id}-", dir=base)
    dest, out = os.path.join(work, "clone"), os.path.join(work, "out")
    os.makedirs(out)
    t0 = time.time()
    croot = clone(root, commit, dest, run_id, kind)
    sha = _git(croot, "rev-parse", "HEAD")
    before = _status(dest)
    present = [t for t in tests if os.path.exists(os.path.join(croot, t))]
    missing = [t for t in tests if t not in present]
    env = _child_env(croot, run_id, out, agent)
    junit = os.path.join(out, "junit.xml")
    suite = f"crucible-v5-{run_id}"
    col_rc, collected, run_rc = None, [], None
    if present:
        col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                              *present], cwd=croot, env=env, capture_output=True, text=True)
        col_rc, collected = col.returncode, parse_collected(col.stdout)
        run_rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                  "-o", f"junit_suite_name={suite}", f"--junitxml={junit}", *pytest_args, *present],
                                 cwd=croot, env=env)
    after = _status(dest)
    rec = {"schema": RUN_SCHEMA, "run_id": run_id, "kind": kind, "requirement_ids": list(requirement_ids),
           "commit": sha, "source_head": _git(root, "rev-parse", "HEAD"),
           "clone": {"path": dest, "method": "git clone --shared --no-checkout + detached checkout",
                     "clean_before": not before, "status_before": before[:20], "status_after": after[:20]},
           "tests": present, "tests_missing_at_commit": missing,
           "command": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", f"junit_suite_name={suite}",
                       "--junitxml=<junit>", *pytest_args, *present],
           "env": {"PYTHONPATH": env["PYTHONPATH"], "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                   "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS")},
           "collect_rc": col_rc, "collected": collected, "pytest_rc": run_rc,
           "junit_suite": suite, "identity": identity() | {"agent": agent},
           "started_at": _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).isoformat(), "finished_at": _now(),
           "wall_s": round(time.time() - t0, 3), "evidence": {}}
    if not keep:
        shutil.rmtree(dest, ignore_errors=True)
        rec["clone"]["removed"] = True
    return {"record": rec, "out": out, "junit": junit if os.path.exists(junit) else None, "work": work}


def _install_records(root: str, res: dict, wanted: set[str], dest_dir: str, *, evidence_layout: bool) -> dict:
    """Copy records produced by the run into the checkout; returns {rid: relpath}."""
    rec, out = res["record"], res["out"]
    run_id = rec["run_id"]
    installed = {}
    edir = os.path.join(out, "evidence")
    for dp, _, files in os.walk(edir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            ev = json.load(open(os.path.join(dp, fn)))
            rid = ev.get("requirement")
            if rid not in wanted:
                print(f"note: the run recorded {rid}, which is not in this selection; not installed", file=sys.stderr)
                continue
            for a in ev.get("artifacts", []):
                if a["path"].startswith("@run/"):
                    src = os.path.join(out, a["path"][5:])
                    rel = f"{RUNS_DIR}/{run_id}/{a['path'][5:]}"
                    if a.get("too_large"):
                        a["external"] = src
                        a["path"] = rel
                        continue
                    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                    shutil.copy2(src, os.path.join(root, rel))
                    a["path"] = rel
            run_rel = f"{dest_dir}/{run_id}.json"
            ev["junit"] = ({"path": f"{dest_dir}/{run_id}.junit.xml", "sha256": sha256_file(res["junit"])}
                           if res["junit"] else None)
            ev["run"] = {"path": run_rel}
            rel = evidence_relpath(rid) if evidence_layout else f"{dest_dir}/{run_id}/{rid}.json"
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(ev, fh, indent=1, sort_keys=True)
            rec["evidence"][rid] = sha256_file(p)
            installed[rid] = rel
    if res["junit"]:
        jp = os.path.join(root, dest_dir, f"{run_id}.junit.xml")
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        shutil.copy2(res["junit"], jp)
    rp = os.path.join(root, dest_dir, f"{run_id}.json")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return installed


def select(m: dict, requirement: str | None = None, gate: str | None = None) -> list[dict]:
    if requirement:
        return [e for e in m["requirements"] if e["id"] == requirement]
    return [e for e in m["requirements"] if e["gate"] == gate or (gate == "SP" and e["id"].startswith("SP-"))]


def run(root: str | None = None, *, requirement: str | None = None, gate: str | None = None, agent: str | None = None,
        commit: str = "HEAD", scratch: str | None = None, keep: bool = False) -> dict:
    root = root or M.crucible_root()
    agent = require_agent(agent)
    m = M.load(root)
    reqs = select(m, requirement, gate)
    if not reqs:
        raise SystemExit("no such requirement or gate")
    tests = sorted({t for e in reqs for t in e["tests"]})
    res = execute(root, tests, commit=commit, kind="evidence", agent=agent,
                  requirement_ids=[e["id"] for e in reqs], scratch=scratch, keep=keep)
    installed = _install_records(root, res, {e["id"] for e in reqs}, RUNS_DIR, evidence_layout=True)
    shutil.rmtree(res["work"], ignore_errors=True) if not keep else None
    return {"run_id": res["record"]["run_id"], "pytest_rc": res["record"]["pytest_rc"], "installed": installed,
            "commit": res["record"]["commit"], "tests_missing_at_commit": res["record"]["tests_missing_at_commit"]}


# ------------------------------------------------------------------------------------------------- reproduction
def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_metrics(ev: dict, rep: dict, tolerance: dict | None = None) -> list[str]:
    """Mismatches between evidence and reproduced metrics: exact for bools, strings and integer pairs; floats within
    max(abs, rel * max(|a|, |b|)) using the declared per-metric tolerance (default rel 1e-6, abs 1e-9)."""
    tolerance = tolerance or {}
    out = []
    for k, a in sorted(ev.items()):
        if k not in rep:
            out.append(f"{k}: not reproduced")
            continue
        b = rep[k]
        if _isnum(a) and _isnum(b):
            if not (math.isfinite(a) and math.isfinite(b)):
                out.append(f"{k}: non-finite ({a!r} vs {b!r})")
                continue
            if isinstance(a, int) and isinstance(b, int) and k not in tolerance:
                if a != b:
                    out.append(f"{k}: {a!r} vs reproduced {b!r}")
                continue
            t = tolerance.get(k, {})
            lim = max(float(t.get("abs", DEFAULT_ABS)), float(t.get("rel", DEFAULT_REL)) * max(abs(a), abs(b)))
            if abs(a - b) > lim:
                out.append(f"{k}: {a!r} vs reproduced {b!r} (tolerance {lim:.3g})")
        elif a != b or type(a) is not type(b):
            out.append(f"{k}: {a!r} vs reproduced {b!r}")
    return out


def reproduce(root: str, e: dict, ev: dict, *, agent: str, scratch: str | None = None, install: bool = True,
              keep: bool = False) -> dict:
    """Re-execute the requirement's tests at the evidence commit in a fresh clone and compare."""
    from .checker import criterion_met
    res = execute(root, list(e["tests"]), commit=ev["tree"]["commit"], kind="reproduction", agent=agent,
                  requirement_ids=[e["id"]], scratch=scratch, keep=keep)
    rec = res["record"]
    rep_path = os.path.join(res["out"], "evidence", f"gate_{gate_of(e['id'])}", f"{e['id']}.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else None
    mism = ["the reproduction produced no record"] if rep is None else compare_metrics(ev["metrics"], rep["metrics"],
                                                                                      e.get("tolerance"))
    crit = [] if rep is None else [w for ok, w in (criterion_met(c, rep["metrics"]) for c in e["criteria"]) if not ok]
    installed = _install_records(root, res, {e["id"]}, REPRO_DIR, evidence_layout=False) if install else {}
    if not keep:
        shutil.rmtree(res["work"], ignore_errors=True)
    return {"run_id": rec["run_id"], "pytest_rc": rec["pytest_rc"], "record": rec, "reproduced": rep,
            "mismatches": mism, "criteria_failures": crit, "installed": installed}


def review(root: str | None = None, *, requirement: str, agent: str | None = None, verdict: str = "approved",
           notes: str = "", scratch: str | None = None) -> dict:
    root = root or M.crucible_root()
    agent = require_agent(agent)
    m = M.load(root)
    e = next((x for x in m["requirements"] if x["id"] == requirement), None)
    if e is None:
        raise SystemExit("no such requirement")
    ev_rel = evidence_relpath(requirement)
    ev_path = os.path.join(root, ev_rel)
    if not os.path.exists(ev_path):
        raise SystemExit("no evidence to review")
    ev = json.load(open(ev_path))
    me = identity() | {"agent": agent}
    prod = ev.get("identity") or {}
    if agent == (prod.get("agent") or ev.get("author")):
        raise SystemExit("the reviewer must not be the evidence's producer (same agent name)")
    if me["session_sha256"] and me["session_sha256"] == prod.get("session_sha256"):
        raise SystemExit("the reviewer must not be the evidence's producer (same session)")
    rp = reproduce(root, e, ev, agent=agent, scratch=scratch)
    ok = rp["pytest_rc"] == 0 and not rp["mismatches"] and not rp["criteria_failures"]
    final = "approved" if (verdict == "approved" and ok) else "rejected"
    rec = {"schema": REVIEW_SCHEMA, "requirement": requirement,
           "requirement_sha256": sha256_text(e["requirement"]),
           "criteria_sha256": sha256_text(canonical(e["criteria"])), "criteria_version": len(e["criteria_history"]),
           "evidence_path": ev_rel, "evidence_sha256": sha256_file(ev_path), "evidence_commit": ev["tree"]["commit"],
           "evidence_run_id": ev.get("run_id"),
           "reproduction": {"run_path": f"{REPRO_DIR}/{rp['run_id']}.json", "run_id": rp["run_id"],
                            "evidence_path": rp["installed"].get(requirement), "pytest_rc": rp["pytest_rc"],
                            "mismatches": rp["mismatches"], "criteria_failures": rp["criteria_failures"],
                            "metrics_sha256": sha256_text(canonical(rp["reproduced"]["metrics"]))
                            if rp["reproduced"] else None},
           "reviewer": me, "requested_verdict": verdict, "verdict": final, "notes": notes, "reviewed_at": _now()}
    d = os.path.join(root, "docs", "v5", "reviews")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{requirement}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec
