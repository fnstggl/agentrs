"""Official runs and independent reviews.

`run`: clone the committed tree (default HEAD; uncommitted work is never measured) into a fresh directory, mark the
clone, collect (`pytest --collect-only`) and run the requirement's test files there with the clone's own `src` on
PYTHONPATH, then install into the checkout: the evidence records, the run record (commit, clone state before and after,
collected node ids, exit codes, environment, identity) and the JUnit file whose suite name carries the run id.

`review`: an independent reviewer (a different agent name and session from the evidence's producer) re-executes the
requirement's tests in another fresh clone of the evidence commit, compares the reproduced metrics with the evidence
within the requirement's declared tolerance, checks the criteria on the reproduction, and writes a review bound to the
requirement text, criteria version, evidence hash and commit, plus the reproduction's own run record, JUnit and
records. The review also ablates the implementation: a third fresh clone in which every function of the requirement's
listed implementation modules raises when it is executed; the producing test is run there alone, and the measurement
must not survive (no record, or different criteria metrics) - a test whose metrics do not depend on its implementation
measures nothing. The ablation blocks execution and leaves the implementation's files and code objects byte-identical
(`ABLATION_HOOK`, D-83): a measurement computed from the implementation's text or bytecode instead of its behaviour
survives it. The reviewer commits all of it; the checker re-verifies all of it (`checker.py`).

The measured process gets a scrubbed environment: no PYTHON* variable except the clone's PYTHONPATH and the runner's
own settings, no PYTEST_* variable, no LD_PRELOAD / LD_AUDIT, no user site-packages (`python -s`,
PYTHONNOUSERSITE=1), so nothing outside the committed tree is injected through the environment (the P skeptic's S1
attack); what remains - the interpreter and its start-up hooks - is recorded in every evidence record.

Both commands only write inside `docs/v5/evidence/` and `docs/v5/reviews/` of the checkout they are run from.
"""
from __future__ import annotations

import ast
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
ABLATIONS_DIR = "docs/v5/reviews/ablations"
ABLATION_EXC = "__CrucibleAblated__"
SCRUB_PREFIXES = ("PYTHON", "PYTEST_", "CRUCIBLE_V5_", "COV_CORE_")
SCRUB_NAMES = ("LD_PRELOAD", "LD_AUDIT", "COVERAGE_PROCESS_START",
               "CRUCIBLE_HISTORICAL_BASELINE")          # a 4.0 reproduction switch (D-82) never reaches a gate
PYTEST = ["-s", "-m", "pytest", "-p", "crucible.acceptance.v5.evidence"]   # -s: no user site; the plugin installs
#                                                                         the files-read audit hook before collection


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


def require_session() -> str:
    s = identity()["session_sha256"]
    if not s:
        raise SystemExit("an official run or review needs a session identity (CRUCIBLE_V5_SESSION or "
                         "CLAUDE_CODE_SESSION_ID): without one, a self-review under a second name is indistinguishable")
    return s


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
    with open(os.path.join(gd, MARKER), "w") as fh:          # no run kind: the measured process need not know
        json.dump({"run_id": run_id, "commit": sha, "created_at": _now()}, fh)   # whether it is being ablated
    return os.path.join(dest, prefix) if prefix != "." else dest


def _status(cwd: str) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd, capture_output=True,
                         text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def scrubbed(env: dict) -> tuple[dict, list[str]]:
    """The environment without code-injection variables, and the names removed."""
    gone = sorted(k for k in env if k.startswith(SCRUB_PREFIXES) or k in SCRUB_NAMES)
    return {k: v for k, v in env.items() if k not in gone}, gone


def _child_env(clone_root: str, run_id: str, out: str, agent: str) -> tuple[dict, list[str]]:
    env, gone = scrubbed(dict(os.environ))
    env.update({"PYTHONPATH": os.path.join(clone_root, "src"), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "CRUCIBLE_V5_RUN_ID": run_id, "CRUCIBLE_V5_EVIDENCE_OUT": out,
                "CRUCIBLE_V5_AGENT": agent, "CRUCIBLE_V5_AUTHOR": agent})
    for k in ("CRUCIBLE_V5_SESSION",):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env, [k for k in gone if not k.startswith("CRUCIBLE_V5_")]


def parse_collected(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if "::" in ln and not ln.startswith(" ") and
            not ln.startswith("=") and not ln.lower().startswith(("error", "warning"))]


def ablate_source(src: str, label: str) -> tuple[str, int]:
    """`src` with the body of every function and method replaced by `raise __CrucibleAblated__(...)` (a BaseException,
    so `except Exception` in the measured code cannot absorb it); module-level code and class bodies are kept."""
    tree = ast.parse(src)
    count = 0

    class _T(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            nonlocal count
            count += 1
            doc = node.body[:1] if (node.body and isinstance(node.body[0], ast.Expr) and
                                    isinstance(node.body[0].value, ast.Constant) and
                                    isinstance(node.body[0].value.value, str)) else []
            node.body = doc + [ast.Raise(exc=ast.Call(func=ast.Name(ABLATION_EXC, ast.Load()),
                                                      args=[ast.Constant(f"{label}:{node.name}")], keywords=[]),
                                         cause=None)]
            return node                                 # nested definitions go with the body
        visit_AsyncFunctionDef = visit_FunctionDef

    tree = _T().visit(tree)
    i = 1 if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
              and isinstance(tree.body[0].value.value, str)) else 0
    while i < len(tree.body) and isinstance(tree.body[i], ast.ImportFrom) and tree.body[i].module == "__future__":
        i += 1
    tree.body.insert(i, ast.parse(f"class {ABLATION_EXC}(BaseException):\n    pass\n").body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", count


# The behavioural ablation (D-83, wave 3b): appended to the clone's `src/crucible/__init__.py`, so every process that
# imports Crucible from the clone (the measured pytest process: the evidence plugin imports it first; worker and
# subprocesses; clones made from the ablated clone) blocks the implementation. A sys.monitoring PY_START callback
# raises a BaseException when a function (a code object with CO_OPTIMIZED: functions, methods, lambdas, generator
# bodies - not module or class bodies) of a listed implementation file starts, except while a module is being imported
# (an importlib bootstrap frame is on the stack), so modules still load. Nothing of the implementation is rewritten:
# its files, code objects, docstrings and line numbers are the evidence commit's. Not covered: processes that do not
# import Crucible from a clone (the sandbox's `python -I -S` boot and jail scripts; every requirement listing them
# also lists the in-process modules that start them) and generators started during an import and resumed later.
ABLATION_HOOK = """

# --- crucible-v5 review: behavioural ablation (acceptance/v5/official.py; exists only in a throwaway clone) ---
def _crucible_v5_ablation(suffixes):
    import sys
    mon = sys.monitoring
    tool = next((t for t in (5, 3, 2) if mon.get_tool(t) is None), None)
    if tool is None:
        raise RuntimeError("crucible-v5 ablation: no free sys.monitoring tool id")

    class __CrucibleAblated__(BaseException):
        pass

    def start(code, offset):
        if not (code.co_flags & 1 and code.co_filename.endswith(suffixes)):
            return mon.DISABLE
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename.startswith("<frozen importlib._bootstrap"):
                return None
            f = f.f_back
        raise __CrucibleAblated__(code.co_filename + ":" + code.co_qualname)
    mon.use_tool_id(tool, "crucible-v5-ablation")
    mon.register_callback(tool, mon.events.PY_START, start)
    mon.set_events(tool, mon.events.PY_START)


_crucible_v5_ablation(%r)
del _crucible_v5_ablation
"""
ABLATION_PACKAGE_INIT = "src/crucible/__init__.py"


def function_count(src: str) -> int:
    """Functions and methods defined in a module (what the behavioural ablation blocks; lambdas not counted)."""
    return sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(ast.parse(src)))


def ablatable(root: str, e: dict, commit: str) -> list[str]:
    """The requirement's implementation modules (Python files) present at `commit`."""
    out = []
    for f in e["implementation_files"]:
        if f.endswith(".py") and subprocess.run(["git", "cat-file", "-e", f"{commit}:./{f}"], cwd=root,
                                                capture_output=True).returncode == 0:
            out.append(f)
    return sorted(out)


def execute(root: str, tests: list[str], *, commit: str = "HEAD", kind: str = "evidence", agent: str,
            requirement_ids: list[str], scratch: str | None = None, keep: bool = False,
            pytest_args: list[str] = (), nodes: list[str] | None = None, ablate: list[str] | None = None) -> dict:
    """Clone, collect, run. Returns {"record": run record, "out": output dir, "junit": path} (files not installed).
    `nodes`: run only these node ids. `ablate`: implementation files to ablate in the clone (committed there, so the
    clone is clean and its records are official for the ablated commit)."""
    run_id = uuid.uuid4().hex[:16]
    base = scratch or os.environ.get("CRUCIBLE_V5_SCRATCH") or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"crucible-v5-{kind}-{run_id}-", dir=base)
    dest, out = os.path.join(work, "clone"), os.path.join(work, "out")
    os.makedirs(out)
    t0 = time.time()
    croot = clone(root, commit, dest, run_id, kind)
    sha = _git(croot, "rev-parse", "HEAD")
    ablation = None
    if ablate:
        done = {}
        init = os.path.join(croot, ABLATION_PACKAGE_INIT)
        mode = "block" if os.path.exists(init) else "rewrite"
        for f in ablate:
            fp = os.path.join(croot, f)
            src = open(fp, encoding="utf-8").read()
            if mode == "block":
                done[f] = function_count(src)
            else:                       # a tree without the package's __init__.py: D-26 file ablation
                new, done[f] = ablate_source(src, f)
                with open(fp, "w", encoding="utf-8") as fh:
                    fh.write(new)
        if mode == "block":
            sufs = tuple(sorted({os.sep + os.path.normpath(f) for f in ablate}))
            with open(init, "a", encoding="utf-8") as fh:
                fh.write(ABLATION_HOOK % (sufs,))
        _git(dest, "-c", "user.name=crucible-v5-ablation", "-c", "user.email=ablation@invalid", "commit", "-qam",
             "ablation (never pushed)")
        ablated_sha = _git(croot, "rev-parse", "HEAD")
        gd = _git(dest, "rev-parse", "--absolute-git-dir")
        with open(os.path.join(gd, MARKER), "w") as fh:
            json.dump({"run_id": run_id, "commit": ablated_sha, "created_at": _now()}, fh)
        ablation = {"base_commit": sha, "ablated_commit": ablated_sha, "functions_ablated": done, "mode": mode}
    before = _status(dest)
    present = [t for t in tests if os.path.exists(os.path.join(croot, t))]
    missing = [t for t in tests if t not in present]
    env, removed = _child_env(croot, run_id, out, agent)
    junit = os.path.join(out, "junit.xml")
    suite = f"crucible-v5-{run_id}"
    col_rc, collected, run_rc = None, [], None
    targets = list(nodes) if nodes else present
    if present:
        col = subprocess.run([sys.executable, *PYTEST, "--collect-only", "-q", "-p", "no:cacheprovider", *targets],
                             cwd=croot, env=env, capture_output=True, text=True)
        col_rc, collected = col.returncode, parse_collected(col.stdout)
        run_rc = subprocess.call([sys.executable, *PYTEST, "-q", "-p", "no:cacheprovider",
                                  "-o", f"junit_suite_name={suite}", f"--junitxml={junit}", *pytest_args, *targets],
                                 cwd=croot, env=env)
    after = _status(dest)
    rec = {"schema": RUN_SCHEMA, "run_id": run_id, "kind": kind, "requirement_ids": list(requirement_ids),
           "commit": sha, "source_head": _git(root, "rev-parse", "HEAD"),
           "clone": {"path": dest, "method": "git clone --shared --no-checkout + detached checkout",
                     "clean_before": not before, "status_before": before[:20], "status_after": after[:20]},
           "tests": present, "tests_missing_at_commit": missing,
           "command": ["python", *PYTEST, "-q", "-p", "no:cacheprovider", "-o", f"junit_suite_name={suite}",
                       "--junitxml=<junit>", *pytest_args, *targets],
           "env": {"PYTHONPATH": env["PYTHONPATH"], "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                   "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"), "removed": removed,
                   "crucible": {k: v for k, v in sorted(env.items()) if k.startswith("CRUCIBLE_") and
                                not k.startswith("CRUCIBLE_V5_")}},
           "interpreter": {"executable": os.path.realpath(sys.executable), "prefix": sys.prefix},
           "nodes": list(nodes) if nodes else None, "ablation": ablation,
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
    require_session()
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
              keep: bool = False, commit: str | None = None) -> dict:
    """Re-execute the requirement's tests in a fresh clone (default: at the evidence commit; the release check passes
    HEAD) and compare with the evidence."""
    from .checker import criterion_met
    at = commit or ev["tree"]["commit"]
    res = execute(root, list(e["tests"]), commit=at, kind="reproduction", agent=agent,
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
            "mismatches": mism, "criteria_failures": crit, "installed": installed, "commit": rec["commit"]}


def ablation_survivors(ev_metrics: dict, ablated: dict | None, tolerance: dict | None,
                       criteria: list[dict] | None = None) -> list[str]:
    """The criteria metrics (all metrics when no criteria are given) that the ablated run recorded unchanged. The
    requirement's measurement is its criteria metrics: an auxiliary metric that changes under ablation (a hash of the
    implementation's source, the commit) says nothing about whether the criteria were measured (wave 3b, D-83)."""
    if ablated is None:
        return []
    got = ablated.get("metrics") or {}
    names = [n for n in dict.fromkeys(c["metric"] for c in criteria or []) if n in ev_metrics] or list(ev_metrics)
    return [n for n in names if not compare_metrics({n: ev_metrics[n]}, got, tolerance)]


def ablation_sensitive(ev_metrics: dict, ablated: dict | None, tolerance: dict | None,
                       criteria: list[dict] | None = None) -> bool:
    """The measurement depends on the implementation: the ablated run recorded nothing, or every criteria metric
    differs from the evidence."""
    return ablated is None or not ablation_survivors(ev_metrics, ablated, tolerance, criteria)


def ablate(root: str, e: dict, ev: dict, *, agent: str, scratch: str | None = None, install: bool = True,
           commit: str | None = None) -> dict:
    """Run the producing test alone in a fresh clone (default: the evidence commit) whose implementation modules are
    ablated; `sensitive` is true when the evidence's measurement does not survive."""
    at = commit or ev["tree"]["commit"]
    files = ablatable(root, e, at)
    node = ev.get("produced_by") or ""
    if not files:
        return {"applicable": False, "why": "no implementation module (.py) listed", "ablated_files": [],
                "sensitive": None}
    res = execute(root, [node.split("::", 1)[0]], commit=at, kind="ablation", agent=agent,
                  requirement_ids=[e["id"]], scratch=scratch, nodes=[node], ablate=files)
    rec = res["record"]
    path = os.path.join(res["out"], "evidence", f"gate_{gate_of(e['id'])}", f"{e['id']}.json")
    got = json.load(open(path)) if os.path.exists(path) else None
    installed = _install_records(root, res, {e["id"]}, ABLATIONS_DIR, evidence_layout=False) if install else {}
    shutil.rmtree(res["work"], ignore_errors=True)
    return {"applicable": True, "run_id": rec["run_id"], "run_path": f"{ABLATIONS_DIR}/{rec['run_id']}.json",
            "commit": rec["commit"], "ablated_commit": (rec.get("ablation") or {}).get("ablated_commit"),
            "ablated_files": files, "functions_ablated": sum((rec.get("ablation") or {}).get("functions_ablated",
                                                                                          {}).values()),
            "node": node, "pytest_rc": rec["pytest_rc"], "record_produced": got is not None,
            "record_path": installed.get(e["id"]),
            "survivors": ablation_survivors(ev["metrics"], got, e.get("tolerance"), e.get("criteria")),
            "sensitive": ablation_sensitive(ev["metrics"], got, e.get("tolerance"), e.get("criteria"))}


def review(root: str | None = None, *, requirement: str, agent: str | None = None, verdict: str = "approved",
           notes: str = "", scratch: str | None = None) -> dict:
    root = root or M.crucible_root()
    agent = require_agent(agent)
    require_session()
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
    ab = ablate(root, e, ev, agent=agent, scratch=scratch)
    ok = rp["pytest_rc"] == 0 and not rp["mismatches"] and not rp["criteria_failures"] and ab["sensitive"] is not False
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
           "ablation": ab,
           "reviewer": me, "requested_verdict": verdict, "verdict": final, "notes": notes, "reviewed_at": _now()}
    d = os.path.join(root, "docs", "v5", "reviews")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{requirement}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec
