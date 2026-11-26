"""Verify every manifest requirement against its evidence. Nothing is taken on trust.

For each requirement the checker confirms, independently of the manifest's `status` string:

1. the entry is well formed and its criteria match the last change-controlled version;
2. the implementation files and test files exist;
3. an evidence record exists for it, produced from a clean tree at a commit that is in HEAD's history and that
   already contains the latest change to the implementation and test files (stale evidence is rejected);
4. every criterion holds when evaluated on the record's own metrics;
5. every artifact the record cites exists with the recorded SHA-256;
6. the record cites a JUnit file (hash-checked) in which its producing test and every test case of the
   requirement's test files passed, none skipped;
7. a review approves this exact evidence (matching SHA-256) and the reviewer is not the evidence's author.

The computed status is VERIFIED only if all seven hold. `verify_all` returns a nonzero exit code otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET

from . import manifest as M
from .evidence import SCHEMA, gate_of, sha256_file

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def _git_ok(*args: str, cwd: str) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _git(*args: str, cwd: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def criterion_met(c: dict, metrics: dict) -> tuple[bool, str]:
    if c["metric"] not in metrics:
        return False, f"metric {c['metric']} missing from evidence"
    v = metrics[c["metric"]]
    if isinstance(c["value"], bool) != isinstance(v, bool):
        return False, f"{c['metric']}={v!r} has the wrong type for {c['op']} {c['value']!r}"
    try:
        ok = OPS[c["op"]](v, c["value"])
    except TypeError:
        return False, f"{c['metric']}={v!r} cannot be compared with {c['value']!r}"
    return ok, "" if ok else f"{c['metric']}={v!r} fails {c['op']} {c['value']!r}"


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


def _case_in(case: dict, test_file: str) -> bool:
    mod = test_file[:-3].replace("/", ".") if test_file.endswith(".py") else test_file
    return case["classname"] == mod or case["classname"].startswith(mod + ".") or case["file"] == test_file


def _case_is(case: dict, node: str) -> bool:
    if "::" not in node:
        return False
    f, rest = node.split("::", 1)
    name = rest.split("::")[-1]
    return _case_in(case, f) and case["name"] == name


def check_requirement(e: dict, root: str, decisions: set[str], blockers_text: str) -> dict:
    problems: list[str] = []
    missing = [f for f in M.ENTRY_FIELDS if f not in e]
    if missing:
        return {"id": e.get("id"), "status": "FAILED", "problems": [f"malformed entry, missing {missing}"]}
    if e["status"] not in M.STATUSES:
        problems.append(f"unknown status {e['status']!r}")
    hist = e["criteria_history"]
    if not hist or hist[-1]["criteria"] != e["criteria"]:
        problems.append("criteria differ from the last change-controlled version")
    for h in hist[1:]:
        if h.get("decision") not in decisions:
            problems.append(f"criteria change cites no recorded decision ({h.get('decision')})")
    if e["status"] == "BLOCKED":
        if not e.get("blocker") or e["blocker"] not in blockers_text:
            problems.append("BLOCKED without a blocker recorded in BLOCKERS.md")
        problems.append(f"blocked: {e.get('blocker')}")

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

    ev_path = os.path.join(root, "docs", "v5", "evidence", f"gate_{gate_of(e['id'])}", f"{e['id']}.json")
    if not os.path.exists(ev_path):
        problems.append("no evidence record")
        started = bool(impl) and not absent
        return {"id": e["id"], "status": "BLOCKED" if e["status"] == "BLOCKED" else
                ("IN_PROGRESS" if started else "NOT_STARTED"), "problems": problems}
    try:
        ev = json.load(open(ev_path))
    except ValueError:
        return {"id": e["id"], "status": "FAILED", "problems": problems + ["evidence is not valid JSON"]}
    if ev.get("schema") != SCHEMA or ev.get("requirement") != e["id"]:
        problems.append("evidence schema or requirement id does not match")

    crit_fail = []
    for c in e["criteria"]:
        ok, why = criterion_met(c, ev.get("metrics", {}))
        if not ok:
            crit_fail.append(why)
    problems += [f"criterion: {w}" for w in crit_fail]

    tree = ev.get("tree", {})
    commit = tree.get("commit")
    if not tree.get("clean"):
        problems.append(f"evidence produced from an uncommitted tree: {tree.get('dirty_paths', [])[:3]}")
    if not commit or not _git_ok("cat-file", "-e", f"{commit}^{{commit}}", cwd=root):
        problems.append("evidence commit unknown")
    elif not _git_ok("merge-base", "--is-ancestor", commit, "HEAD", cwd=root):
        problems.append("evidence commit is not in HEAD's history")
    else:
        files = [f for f in impl + tests if os.path.exists(os.path.join(root, f))]
        if files:
            last = _git("log", "-1", "--format=%H", "HEAD", "--", *files, cwd=root)
            if last and not _git_ok("merge-base", "--is-ancestor", last, commit, cwd=root):
                problems.append(f"stale evidence: implementation or tests changed in {last[:9]} after the evidence commit")

    for a in ev.get("artifacts", []):
        p = os.path.join(root, a["path"])
        if not os.path.exists(p):
            problems.append(f"artifact missing: {a['path']}")
        elif sha256_file(p) != a["sha256"]:
            problems.append(f"artifact changed since recorded: {a['path']}")

    j = ev.get("junit")
    if not j:
        problems.append("no JUnit result attached to the evidence")
    else:
        jp = os.path.join(root, j["path"])
        if not os.path.exists(jp) or sha256_file(jp) != j["sha256"]:
            problems.append("JUnit file missing or changed")
        else:
            cases = junit_cases(jp)
            own = [c for c in cases if _case_is(c, ev.get("produced_by", ""))]
            if not own:
                problems.append("the producing test is not in the JUnit file")
            for t in tests:
                mine = [c for c in cases if _case_in(c, t)]
                if not mine:
                    problems.append(f"no test case of {t} in the JUnit file")
                bad = [f"{c['name']}:{c['outcome']}" for c in mine if c["outcome"] != "passed"]
                if bad:
                    problems.append(f"tests not passed in {t}: {bad[:5]}")

    rv_path = os.path.join(root, "docs", "v5", "reviews", f"{e['id']}.json")
    if not os.path.exists(rv_path):
        problems.append("no independent review")
    else:
        rv = json.load(open(rv_path))
        if rv.get("verdict") != "approved":
            problems.append(f"review verdict {rv.get('verdict')!r}")
        if rv.get("evidence_sha256") != sha256_file(ev_path):
            problems.append("review is for a different version of the evidence")
        if not rv.get("reviewer") or rv.get("reviewer") == ev.get("author"):
            problems.append("reviewer is missing or is the evidence's author")

    if not problems:
        status = "VERIFIED"
    elif crit_fail:
        status = "FAILED"
    elif e["status"] == "BLOCKED":
        status = "BLOCKED"
    else:
        status = "IMPLEMENTED_UNVERIFIED"
    return {"id": e["id"], "status": status, "problems": problems}


def verify_all(root: str | None = None) -> dict:
    root = root or M.crucible_root()
    m = M.load(root)
    decisions = M.decisions(root)
    try:
        blockers = open(os.path.join(M.docs_dir(root), "BLOCKERS.md")).read()
    except OSError:
        blockers = ""
    rows = [check_requirement(e, root, decisions, blockers) for e in m["requirements"]]
    claimed = {e["id"]: e["status"] for e in m["requirements"]}
    for r in rows:
        if claimed.get(r["id"]) == "VERIFIED" and r["status"] != "VERIFIED":
            r["problems"].insert(0, "manifest claims VERIFIED but the evidence does not support it")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    ok = all(r["status"] == "VERIFIED" for r in rows)
    head = _git("rev-parse", "HEAD", cwd=root)
    return {"ok": ok, "head": head, "requirements": len(rows), "counts": counts, "rows": rows,
            "project_status": "CRUCIBLE_5_COMPLETE" if ok else "IN_PROGRESS"}
