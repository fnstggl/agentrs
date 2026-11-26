"""Verify every manifest requirement against its evidence. Nothing is taken on trust.

Project level (`consistency`): the manifest equals its derivation from MASTER_SPEC.md (anchored by SHA-256) and
`requirements.py`, and the append-only, hash-chained change-control ledger (anchored in git history); no requirement
was removed; every criteria change is recorded, cites a decision naming it and is provably at least as strict; test and
implementation lists match the ledger; the working tree is committed.

Per requirement, independently of the manifest's `status` string:

1. the implementation files and test files exist (the canonical gate test is always listed);
2. an evidence record exists, is committed, and was produced by an official run: a run record (committed) binds it to
   a commit in HEAD's history, a fresh clone that was clean before and after the run, the collected node ids, and a
   JUnit file (committed, hash-checked, suite name = run id);
3. completeness: every test function of the requirement's test files at that commit was collected, and every
   collected node passed in the JUnit (none skipped, none missing); the producing test is one of the requirement's
   own tests;
4. the producing test asserts on measured values and its record() metrics are not constants; it does not skip,
   return early, swallow exceptions, mock or leave patches in place (`testscan`);
5. the evidence's imported-module hashes equal the commit's blobs, no crucible module came from outside the clone,
   at least one listed implementation module was imported, and none of the imported modules, implementation files
   or test files changed after the evidence commit (transitive staleness);
6. every criterion holds on the evidence's own metrics (finite numbers only); cited artifacts hash-match;
7. an independent review is committed, bound to this requirement text, criteria version, evidence hash and commit;
   its reviewer is a different agent and session; it carries its own reproduction (committed run record, complete
   passing JUnit, reproduced record) whose metrics the checker compares with the evidence within tolerance;
8. every dependency is VERIFIED (greatest fixed point; X depends on everything except X, SP-18 and SP-20).

`verify_all(reproduce=True)` additionally re-executes every VERIFIED requirement's tests in a fresh clone itself.
Project status: CRUCIBLE_5_COMPLETE only if there is no project-level problem and every requirement is VERIFIED;
INCOMPLETE_BLOCKED if there is no project-level problem and every unverified requirement is BLOCKED by an open blocker
of BLOCKERS.md; otherwise IN_PROGRESS.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from . import manifest as M
from . import testscan
from .evidence import (REVIEW_SCHEMA, RUN_SCHEMA, SCHEMA, canonical, evidence_relpath, git_blob_id, sha256_file,
                       sha256_text)

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
PROJECT_STATUSES = ("IN_PROGRESS", "INCOMPLETE_BLOCKED", "CRUCIBLE_5_COMPLETE")
X_EXCLUDED = ("SP-18", "SP-20")


def _git_ok(*args: str, cwd: str) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _git(*args: str, cwd: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _show(cwd: str, commit: str, rel: str) -> str | None:
    p = subprocess.run(["git", "show", f"{commit}:./{rel}"], cwd=cwd, capture_output=True)
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def _finite(v) -> bool:
    return not (isinstance(v, float) and not math.isfinite(v))


def criterion_met(c: dict, metrics: dict) -> tuple[bool, str]:
    if c["metric"] not in metrics:
        return False, f"metric {c['metric']} missing from evidence"
    v = metrics[c["metric"]]
    if not _finite(v) or (isinstance(v, (list, dict)) or v is None):
        return False, f"{c['metric']}={v!r} is not a finite scalar"
    if isinstance(c["value"], bool) != isinstance(v, bool):
        return False, f"{c['metric']}={v!r} has the wrong type for {c['op']} {c['value']!r}"
    try:
        ok = OPS[c["op"]](v, c["value"])
    except TypeError:
        return False, f"{c['metric']}={v!r} cannot be compared with {c['value']!r}"
    return ok, "" if ok else f"{c['metric']}={v!r} fails {c['op']} {c['value']!r}"


# ------------------------------------------------------------------------------------------------- JUnit
def junit_cases(path: str) -> list[dict]:
    out = []
    for tc in ET.parse(path).getroot().iter("testcase"):
        outcome = "passed"
        for kind in ("failure", "error", "skipped"):
            if tc.find(kind) is not None:
                outcome = kind
        out.append({"classname": tc.get("classname") or "", "name": tc.get("name") or "",
                    "file": tc.get("file") or "", "outcome": outcome})
    return out


def junit_suites(path: str) -> list[str]:
    r = ET.parse(path).getroot()
    suites = [r] if r.tag == "testsuite" else list(r.iter("testsuite"))
    return [s.get("name") or "" for s in suites]


def _mod(test_file: str) -> str:
    return test_file[:-3].replace("/", ".") if test_file.endswith(".py") else test_file


def node_key(node: str) -> tuple[str, str]:
    """pytest node id -> (JUnit classname, name)."""
    f, rest = node.split("::", 1)
    parts = rest.split("::")
    return ".".join([_mod(f)] + parts[:-1]), parts[-1]


def _base(name: str) -> str:
    return name.split("[", 1)[0]


def _case_in(case: dict, test_file: str) -> bool:
    mod = _mod(test_file)
    return case["classname"] == mod or case["classname"].startswith(mod + ".") or case["file"] == test_file


# ------------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, root: str):
        self.root = root
        self.head = _git("rev-parse", "HEAD", cwd=root)
        self.tracked = set(_git("ls-files", cwd=root).splitlines())
        self.dirty = set()
        top = os.path.realpath(_git("rev-parse", "--show-toplevel", cwd=root))
        for line in subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "."], cwd=root,
                                   capture_output=True, text=True).stdout.splitlines():
            if line.strip():
                p = line[3:].strip().strip('"').split(" -> ")[-1]
                self.dirty.add(os.path.relpath(os.path.join(top, p), os.path.realpath(root)))
        self.sections = M.L.decision_sections(M.docs_dir(root))
        self.open_blockers = open_blockers(root)
        self._anc: dict = {}
        self._tree: dict = {}

    def committed(self, rel: str) -> bool:
        return rel in self.tracked and rel not in self.dirty

    def ancestor(self, a: str, b: str = "HEAD") -> bool:
        k = (a, b)
        if k not in self._anc:
            self._anc[k] = _git_ok("merge-base", "--is-ancestor", a, b, cwd=self.root)
        return self._anc[k]

    def blobs(self, commit: str, paths: list[str]) -> dict[str, str]:
        out = {}
        for i in range(0, len(paths), 200):
            chunk = paths[i:i + 200]
            for line in _git("ls-tree", "-r", commit, "--", *chunk, cwd=self.root).splitlines():
                meta, name = line.split("\t", 1)
                out[name] = meta.split()[2]
        return out

    def changed_since(self, commit: str, paths: list[str]) -> list[str]:
        out = []
        for i in range(0, len(paths), 200):
            out += _git("diff", "--name-only", "--relative", commit, "HEAD", "--", *paths[i:i + 200],
                        cwd=self.root).splitlines()
        return out


def open_blockers(root: str) -> set[str]:
    try:
        text = open(os.path.join(M.docs_dir(root), "BLOCKERS.md"), encoding="utf-8").read()
    except OSError:
        return set()
    m = re.search(r"^## Open blockers\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return set(re.findall(r"^###\s+(BLK-\d+)\b", m.group(1), re.M)) if m else set()


def _load(path: str):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------- run checks
def run_problems(ctx: Ctx, e: dict, rec: dict, run_rel: str | None, kind: str, commit: str,
                 rec_rel: str) -> tuple[list[str], dict | None]:
    """Problems with the run record behind an evidence or reproduction record, and the run record itself."""
    probs = []
    if not run_rel:
        return ["no run record: not produced by an official run"], None
    if not ctx.committed(run_rel):
        probs.append(f"run record {run_rel} is not committed")
    run = _load(os.path.join(ctx.root, run_rel))
    if run is None:
        return probs + [f"run record {run_rel} missing or unreadable"], None
    if run.get("schema") != RUN_SCHEMA or run.get("kind") != kind:
        probs.append("run record schema or kind does not match")
    if run.get("run_id") != rec.get("run_id"):
        probs.append("run id of the record and of the run record differ")
    if run.get("commit") != commit:
        probs.append("run record commit differs from the record's commit")
    if e["id"] not in (run.get("requirement_ids") or []):
        probs.append("run record does not cover this requirement")
    cl = run.get("clone") or {}
    if not cl.get("clean_before") or cl.get("status_after"):
        probs.append("the run's clone was not clean before and after the run")
    if (run.get("evidence") or {}).get(e["id"]) != sha256_file(os.path.join(ctx.root, rec_rel)):
        probs.append("the run record does not hold this record's hash")
    if run.get("pytest_rc") != 0:
        probs.append(f"pytest exit code {run.get('pytest_rc')} in the run")
    if run.get("collect_rc") != 0:
        probs.append(f"collection exit code {run.get('collect_rc')} in the run")
    if run.get("tests_missing_at_commit"):
        probs.append(f"test files missing at the run's commit: {run['tests_missing_at_commit'][:3]}")
    return probs, run


def junit_problems(ctx: Ctx, e: dict, rec: dict, run: dict | None, commit: str) -> list[str]:
    probs = []
    j = rec.get("junit")
    if not j:
        return ["no JUnit result attached"]
    if not ctx.committed(j["path"]):
        probs.append(f"JUnit {j['path']} is not committed")
    jp = os.path.join(ctx.root, j["path"])
    if not os.path.exists(jp) or sha256_file(jp) != j.get("sha256"):
        return probs + ["JUnit file missing or changed"]
    if f"crucible-v5-{rec.get('run_id')}" not in junit_suites(jp):
        probs.append("JUnit suite name is not bound to the record's run id")
    cases = junit_cases(jp)
    by_key = {(c["classname"], c["name"]): c for c in cases}
    collected = (run or {}).get("collected") or []
    if run is not None and not collected:
        probs.append("the run collected no tests")
    for node in collected:
        k = node_key(node)
        c = by_key.get(k)
        if c is None:
            probs.append(f"collected test {node} has no JUnit result")
        elif c["outcome"] != "passed":
            probs.append(f"{node}: {c['outcome']}")
    col_keys = {(cl, _base(n)) for cl, n in (node_key(x) for x in collected)}
    for t in e["tests"]:
        mine = [c for c in cases if _case_in(c, t)]
        if not mine:
            probs.append(f"no test case of {t} in the JUnit file")
        bad = [f"{c['name']}:{c['outcome']}" for c in mine if c["outcome"] != "passed"]
        if bad:
            probs.append(f"tests not passed in {t}: {bad[:5]}")
        src = _show(ctx.root, commit, t)
        if src is None:
            probs.append(f"{t} does not exist at the record's commit")
            continue
        try:
            funcs = testscan.scan_source(src)
        except SyntaxError:
            probs.append(f"{t} does not parse at the record's commit")
            continue
        for f in funcs:
            cl = _mod(t) + (f".{f['class']}" if f["class"] else "")
            if (cl, f["name"]) not in col_keys:
                probs.append(f"{t}::{f['name']} exists at the commit but was not collected and run")
            if not any(c["classname"] == cl and _base(c["name"]) == f["name"] for c in mine):
                probs.append(f"{t}::{f['name']} has no JUnit result")
    pb = rec.get("produced_by") or ""
    if "::" not in pb or pb.split("::", 1)[0] not in e["tests"]:
        probs.append(f"the producing test {pb!r} is not one of the requirement's tests")
    else:
        c = by_key.get(node_key(pb))
        if c is None or c["outcome"] != "passed":
            probs.append(f"the producing test {pb} did not pass in the JUnit file")
    return probs


def integrity_problems(ctx: Ctx, e: dict, rec: dict, commit: str) -> list[str]:
    """Static integrity of the producing test function at the record's commit."""
    pb = rec.get("produced_by") or ""
    if "::" not in pb:
        return []
    f, rest = pb.split("::", 1)
    parts = rest.split("::")
    name, cls = _base(parts[-1]), (parts[0] if len(parts) > 1 else None)
    src = _show(ctx.root, commit, f)
    if src is None:
        return [f"producing test file {f} absent at the commit"]
    try:
        funcs = testscan.scan_source(src)
    except SyntaxError:
        return [f"{f} does not parse"]
    fn = next((x for x in funcs if x["name"] == name and x["class"] == cls), None)
    if fn is None:
        return [f"producing test {pb} not found in {f} at the commit"]
    probs = []
    bad = testscan.blocking(fn["findings"])
    if bad:
        probs.append(f"producing test {name} is not a genuine measurement: {', '.join(bad)}")
    if fn["measured_asserts"] < 1:
        probs.append(f"producing test {name} has no assertion on measured values")
    if fn["records"] < 1:
        probs.append(f"producing test {name} does not call record()")
    return probs


def module_problems(ctx: Ctx, e: dict, rec: dict, commit: str) -> list[str]:
    probs = []
    mods = rec.get("modules") or {}
    if not mods:
        return ["no imported-module hashes in the record"]
    if rec.get("modules_outside_root"):
        probs.append(f"crucible modules imported from outside the clone: {rec['modules_outside_root'][:2]}")
    blobs = ctx.blobs(commit, sorted(mods))
    wrong = [p for p, b in mods.items() if blobs.get(p) != b]
    if wrong:
        probs.append(f"imported modules do not match the commit's files: {wrong[:3]}")
    impl_py = [f for f in e["implementation_files"] if f.endswith(".py")]
    if impl_py and not any(f in mods for f in impl_py):
        probs.append("the producing process imported none of the listed implementation modules")
    files = sorted(set(mods) | {f for f in e["implementation_files"] + e["tests"]
                                if os.path.exists(os.path.join(ctx.root, f))})
    changed = ctx.changed_since(commit, files)
    if changed:
        probs.append(f"stale evidence: {len(changed)} imported/listed files changed after the evidence commit "
                     f"({changed[:3]})")
    return probs


# ------------------------------------------------------------------------------------------------- review
def review_problems(ctx: Ctx, e: dict, ev: dict, ev_rel: str) -> list[str]:
    from .official import compare_metrics
    rv_rel = f"docs/v5/reviews/{e['id']}.json"
    rv = _load(os.path.join(ctx.root, rv_rel))
    if rv is None:
        return ["no independent review"]
    probs = []
    if not ctx.committed(rv_rel):
        probs.append("the review is not committed")
    if rv.get("schema") != REVIEW_SCHEMA:
        probs.append("review schema does not match")
    if rv.get("verdict") != "approved":
        probs.append(f"review verdict {rv.get('verdict')!r}")
    if rv.get("evidence_sha256") != sha256_file(os.path.join(ctx.root, ev_rel)):
        probs.append("review is for a different version of the evidence")
    if rv.get("requirement_sha256") != sha256_text(e["requirement"]):
        probs.append("review is bound to different requirement wording")
    if rv.get("criteria_sha256") != sha256_text(canonical(e["criteria"])) or \
            rv.get("criteria_version") != len(e["criteria_history"]):
        probs.append("review is bound to a different criteria version")
    commit = (ev.get("tree") or {}).get("commit")
    if rv.get("evidence_commit") != commit:
        probs.append("review is bound to a different commit")
    who, prod = rv.get("reviewer") or {}, ev.get("identity") or {}
    if not who.get("agent"):
        probs.append("review names no reviewer")
    elif who.get("agent") == (prod.get("agent") or ev.get("author")):
        probs.append("the reviewer is the evidence's producer (same agent name)")
    if who.get("session_sha256") and who.get("session_sha256") == prod.get("session_sha256"):
        probs.append("the reviewer is the evidence's producer (same session)")
    rp = rv.get("reproduction") or {}
    rep_rel = rp.get("evidence_path")
    rep = _load(os.path.join(ctx.root, rep_rel)) if rep_rel else None
    if rep is None:
        return probs + ["the review carries no reproduction record"]
    if not ctx.committed(rep_rel):
        probs.append("the reproduction record is not committed")
    if rep.get("schema") != SCHEMA or rep.get("requirement") != e["id"] or not rep.get("official"):
        probs.append("reproduction record schema, requirement or official flag does not match")
    if (rep.get("tree") or {}).get("commit") != commit:
        probs.append("the reproduction ran at a different commit")
    if rep.get("run_id") == ev.get("run_id"):
        probs.append("the reproduction is the evidence's own run")
    rprobs, rrun = run_problems(ctx, e, rep, (rep.get("run") or {}).get("path"), "reproduction", commit, rep_rel)
    probs += [f"reproduction: {p}" for p in rprobs]
    probs += [f"reproduction: {p}" for p in junit_problems(ctx, e, rep, rrun, commit)]
    if rrun is not None and (rrun.get("identity") or {}).get("agent") != who.get("agent"):
        probs.append("the reproduction was not run by the reviewer")
    if rep.get("produced_by") != ev.get("produced_by"):
        probs.append("the reproduction was produced by a different test")
    mism = compare_metrics(ev.get("metrics") or {}, rep.get("metrics") or {}, e.get("tolerance"))
    if mism:
        probs.append(f"reproduction does not match the evidence: {mism[:3]}")
    for c in e["criteria"]:
        ok, why = criterion_met(c, rep.get("metrics") or {})
        if not ok:
            probs.append(f"reproduction criterion: {why}")
    return probs


# ------------------------------------------------------------------------------------------------- requirement
def check_requirement(e: dict, ctx: Ctx) -> dict:
    root = ctx.root
    problems: list[str] = []
    missing = [f for f in M.ENTRY_FIELDS if f not in e]
    if missing:
        return {"id": e.get("id"), "status": "FAILED", "problems": [f"malformed entry, missing {missing}"]}
    if e["status"] not in M.STATUSES:
        problems.append(f"unknown status {e['status']!r}")
    blocked = False
    if e["status"] == "BLOCKED":
        if e.get("blocker") in ctx.open_blockers:
            blocked = True
            problems.append(f"blocked: {e['blocker']}")
        else:
            problems.append(f"BLOCKED without an open blocker in BLOCKERS.md ({e.get('blocker')!r})")
    impl = e["implementation_files"]
    if not impl:
        problems.append("no implementation files listed")
    absent = [f for f in impl if not os.path.exists(os.path.join(root, f))]
    if absent:
        problems.append(f"implementation files missing: {absent[:5]}")
    tests = e["tests"]
    absent_t = [t for t in tests if not os.path.exists(os.path.join(root, t))]
    if not tests or absent_t:
        problems.append(f"test files missing: {absent_t[:5] or 'none listed'}")

    ev_rel = evidence_relpath(e["id"])
    ev_path = os.path.join(root, ev_rel)
    row = {"id": e["id"], "problems": problems, "evidence": None, "review": None}
    if not os.path.exists(ev_path):
        problems.append("no evidence record")
        started = bool(impl) and not absent
        row["status"] = "BLOCKED" if blocked else ("IN_PROGRESS" if started else "NOT_STARTED")
        return row
    ev = _load(ev_path)
    if not isinstance(ev, dict):
        row.update(status="FAILED", problems=problems + ["evidence is not valid JSON"])
        return row
    row["evidence"] = ev_rel
    row["labels"] = ev.get("labels") or {}
    row["metrics"] = ev.get("metrics") or {}
    if ev.get("schema") != SCHEMA or ev.get("requirement") != e["id"]:
        problems.append("evidence schema or requirement id does not match")
    if not ev.get("official"):
        problems.append("evidence was not recorded by an official run")
    if not ctx.committed(ev_rel):
        problems.append("the evidence record is not committed")

    crit_fail = []
    metrics = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else {}
    bad_vals = [k for k, v in metrics.items() if not _finite(v)]
    if bad_vals:
        crit_fail.append(f"non-finite metrics {bad_vals[:3]}")
    for c in e["criteria"]:
        ok, why = criterion_met(c, metrics)
        if not ok:
            crit_fail.append(why)
    problems += [f"criterion: {w}" for w in crit_fail]

    commit = (ev.get("tree") or {}).get("commit")
    commit_ok = bool(commit) and _git_ok("cat-file", "-e", f"{commit}^{{commit}}", cwd=root)
    if not commit_ok:
        problems.append("evidence commit unknown")
    elif not ctx.ancestor(commit):
        problems.append("evidence commit is not in HEAD's history")
    if not (ev.get("tree") or {}).get("clean"):
        problems.append("the evidence's checkout was not clean")
    row["commit"] = commit
    if commit_ok:
        rprobs, run = run_problems(ctx, e, ev, (ev.get("run") or {}).get("path"), "evidence", commit, ev_rel)
        problems += rprobs
        problems += junit_problems(ctx, e, ev, run, commit)
        problems += integrity_problems(ctx, e, ev, commit)
        problems += module_problems(ctx, e, ev, commit)
    for a in ev.get("artifacts", []):
        p = a.get("external") or os.path.join(root, a["path"])
        if not os.path.exists(p):
            problems.append(f"artifact missing: {a['path']}")
        elif sha256_file(p) != a["sha256"]:
            problems.append(f"artifact changed since recorded: {a['path']}")
        elif not a.get("external") and not ctx.committed(a["path"]):
            problems.append(f"artifact not committed: {a['path']}")
    rv = review_problems(ctx, e, ev, ev_rel)
    problems += rv
    row["review"] = None if "no independent review" in rv else f"docs/v5/reviews/{e['id']}.json"

    if not problems:
        status = "VERIFIED"
    elif crit_fail:
        status = "FAILED"
    elif blocked:
        status = "BLOCKED"
    else:
        status = "IMPLEMENTED_UNVERIFIED"
    row["status"] = status
    return row


def _dep_ids(dep: str, rid: str, all_ids: list[str], gates: dict[str, str]) -> list[str] | None:
    if dep == "*":
        return [i for i in all_ids if i != rid and gates[i] != "X" and i not in X_EXCLUDED]
    if dep in all_ids:
        return [dep]
    members = [i for i in all_ids if gates[i] == dep and i != rid]
    return members or None


def verify_all(root: str | None = None, *, reproduce: bool = False, agent: str | None = None,
               scratch: str | None = None) -> dict:
    root = root or M.crucible_root()
    head = _git("rev-parse", "HEAD", cwd=root)
    try:
        m = M.load(root)
    except (OSError, ValueError) as exc:
        return {"ok": False, "head": head, "requirements": 0, "counts": {}, "rows": [], "project_status": "IN_PROGRESS",
                "global_problems": [f"manifest unreadable: {exc}"], "reproduced": reproduce}
    ctx = Ctx(root)
    cons = M.consistency(root, m)
    global_problems = list(cons["global"])
    if ctx.dirty:
        global_problems.append(f"the working tree has uncommitted changes ({sorted(ctx.dirty)[:3]}): only a "
                               "committed tree can be verified")
    entries = [e for e in m.get("requirements", []) if isinstance(e, dict) and "id" in e]
    rows = [check_requirement(e, ctx) for e in entries]
    by_id = {r["id"]: r for r in rows}
    for d in cons["derived"]:                       # requirements pruned from the manifest still count
        if d["id"] not in by_id:
            r = {"id": d["id"], "status": "NOT_STARTED", "problems": ["missing from the manifest"],
                 "evidence": None, "review": None}
            rows.append(r)
            by_id[d["id"]] = r
    for rid, ps in cons["per_id"].items():
        r = by_id.get(rid)
        if r is None:
            continue
        r["problems"] = ps + r["problems"]
        if r["status"] == "VERIFIED":
            r["status"] = "IMPLEMENTED_UNVERIFIED"
    entry_by = {e["id"]: e for e in entries}
    derived_by = {d["id"]: d for d in cons["derived"]}
    all_ids = [r["id"] for r in rows]
    gates = {i: (entry_by.get(i) or derived_by.get(i) or {}).get("gate", "") for i in all_ids}
    # dependencies: greatest fixed point over the self-verified set
    verified = {r["id"] for r in rows if r["status"] == "VERIFIED"}
    changed = True
    while changed:
        changed = False
        for rid in sorted(verified):
            deps = (derived_by.get(rid) or entry_by.get(rid) or {}).get("dependencies", [])
            for dep in deps:
                ids = _dep_ids(dep, rid, all_ids, gates)
                if ids is None:
                    by_id[rid]["problems"].append(f"dependency {dep!r} names no requirement")
                    bad = [dep]
                else:
                    bad = [i for i in ids if i not in verified]
                if bad:
                    by_id[rid]["problems"].append(f"dependency {dep} not verified ({bad[:3]})")
                    by_id[rid]["status"] = "IMPLEMENTED_UNVERIFIED"
                    verified.discard(rid)
                    changed = True
                    break
    if reproduce:
        from .official import reproduce as _repro, require_agent
        who = require_agent(agent or "acceptance-checker")
        for r in rows:
            if r["status"] != "VERIFIED":
                continue
            e = entry_by[r["id"]]
            ev = _load(os.path.join(root, r["evidence"]))
            rp = _repro(root, e, ev, agent=who, scratch=scratch, install=False)
            r["reproduction"] = {"run_id": rp["run_id"], "pytest_rc": rp["pytest_rc"], "mismatches": rp["mismatches"],
                                 "criteria_failures": rp["criteria_failures"]}
            if rp["pytest_rc"] != 0 or rp["mismatches"] or rp["criteria_failures"]:
                r["status"] = "FAILED"
                r["problems"].append(f"clean-checkout reproduction failed: {(rp['mismatches'] + rp['criteria_failures'])[:3]}"
                                     f" (pytest exit {rp['pytest_rc']})")
    claimed = {e["id"]: e.get("status") for e in entries}
    for r in rows:
        if claimed.get(r["id"]) == "VERIFIED" and r["status"] != "VERIFIED":
            r["problems"].insert(0, "manifest claims VERIFIED but the evidence does not support it")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    unverified = [r for r in rows if r["status"] != "VERIFIED"]
    if not global_problems and rows and not unverified:
        status = "CRUCIBLE_5_COMPLETE"
    elif not global_problems and rows and all(r["status"] == "BLOCKED" for r in unverified):
        status = "INCOMPLETE_BLOCKED"
    else:
        status = "IN_PROGRESS"
    return {"ok": status == "CRUCIBLE_5_COMPLETE", "head": head, "requirements": len(rows), "counts": counts,
            "rows": rows, "project_status": status, "global_problems": global_problems, "reproduced": reproduce}


def unmet(rid: str, metrics: dict, root: str | None = None) -> list[str]:
    """The manifest criteria of `rid` that `metrics` do not meet (gate tests assert this is empty)."""
    e = next(x for x in M.load(root)["requirements"] if x["id"] == rid)
    return [why for ok, why in (criterion_met(c, metrics) for c in e["criteria"]) if not ok]
