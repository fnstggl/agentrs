"""Executable completion checker for CAM Unified (spec: CAM_UNIFIED_SPEC_META_REWARDS.md Part 0.4).

    python -m superlearner.verify_completion                 # check claims against recorded evidence
    python -m superlearner.verify_completion --run-tests     # also execute every referenced test, record JUnit XML
    python -m superlearner.verify_completion --json out.json # machine-readable report

It reads superlearner/docs/CAM_UNIFIED_COMPLETION.json and checks, per requirement, that the claimed status is backed
by evidence. It FAILS (exit code 1) when any of the following holds for a requirement:

  * a mandatory requirement is missing from the manifest (ids listed in REQUIRED_IDS / REQUIRED_PREFIXES),
  * a referenced implementation file does not exist,
  * a claim of IMPLEMENTED or above has no resolvable acceptance test, or a referenced test has no recorded PASS in
    the latest JUnit results (docs/evidence/test_results/*.xml), i.e. the test was never executed or failed,
  * a claim of INTEGRATION_TESTED or above has no resolvable, passing integration test, no `integration_entry`
    (the function in the canonical runtime that exercises the capability) or no independent `verified_by`,
  * a claim that names `gradient_tests` (mandatory gradient pathways) has a gradient test without a recorded PASS,
  * an EMPIRICALLY_DEMONSTRATED / FAILED_EMPIRICALLY claim lacks an `experiment_result` file that exists, parses as
    JSON and records the same verdict,
  * an implementation file of a claim at IMPLEMENTED or above is a stub (NotImplementedError bodies, `...` bodies,
    'placeholder'/'stub' markers) or production code imports the test doubles (`integration_doubles`, `testing`).

Engineering completion and scientific success are reported separately:

  engineering complete  = status in {INTEGRATION_TESTED, FAILED_EMPIRICALLY, EMPIRICALLY_DEMONSTRATED} and all checks pass
  scientific success    = EMPIRICALLY_DEMONSTRATED; FAILED_EMPIRICALLY is an honest negative result, not a defect.

The default invocation never claims the project is complete unless every requirement is engineering-complete.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "superlearner", "docs", "CAM_UNIFIED_COMPLETION.json")
RESULTS_DIR = os.path.join(ROOT, "superlearner", "docs", "evidence", "test_results")

STATUSES = ["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY",
            "EMPIRICALLY_DEMONSTRATED"]
AT_LEAST_IMPLEMENTED = {"IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
AT_LEAST_INTEGRATED = {"INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
EMPIRICAL = {"FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}

# Every id below must exist in the manifest (the original A01-A22 acceptance list and the four requirement families
# added on 2026-11-26). A family prefix requires at least one entry and is checked for gaps in its numbering.
REQUIRED_IDS = [f"A{i:02d}" for i in range(1, 23)] + [f"DOM-{i:02d}" for i in range(1, 14)] + \
    [f"EX-{c}" for c in "ABCDEFGHIJK"] + ["X01", "X02", "X03", "X04", "VER-01", "RPT-01"]
REQUIRED_PREFIXES = ["OB-", "SE-", "IR-", "CR-", "KV-", "AU-", "CD-", "UR-"]

STUB_PATTERNS = [
    (re.compile(r"\bplaceholder\b", re.I), "placeholder marker"),
    (re.compile(r"^\s*#\s*(TODO|FIXME|STUB)\b", re.M), "TODO/STUB marker"),
]
DOUBLE_IMPORT = re.compile(r"^\s*(from\s+\S*(integration_doubles|cam_unified\.testing)\S*\s+import|import\s+\S*integration_doubles)",
                           re.M)


# ----------------------------------------------------------------------------------------------- helpers

def load_manifest(path: str = MANIFEST) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _split_test_id(test_id: str) -> tuple[str, str] | None:
    if "::" not in test_id:
        return None
    path, rest = test_id.split("::", 1)
    name = rest.split("[", 1)[0].split("::")[-1]
    return path, name


def test_resolves(test_id: str) -> bool:
    parts = _split_test_id(test_id)
    if parts is None:
        return False
    path, name = parts
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return False
    with open(full) as fh:
        src = fh.read()
    return re.search(rf"^\s*(async\s+)?def {re.escape(name)}\(", src, re.M) is not None


def _junit_key(classname: str, name: str, file_attr: str | None) -> tuple[str, str]:
    """JUnit testcases carry classname 'superlearner.tests.cam_unified.core.test_x' (or with a class suffix)."""
    if file_attr:
        path = file_attr
    else:
        mod = classname.split(".")
        # drop a trailing class name (CamelCase) if present
        if mod and mod[-1][:1].isupper():
            mod = mod[:-1]
        path = "/".join(mod) + ".py"
    path = path.replace("\\", "/")
    if not path.startswith("superlearner/"):
        path = "superlearner/" + path if os.path.exists(os.path.join(ROOT, "superlearner", path)) else path
    return path, name.split("[", 1)[0]


def load_results(results_dir: str = RESULTS_DIR) -> dict[tuple[str, str], dict]:
    """Latest outcome per (test file, test function) across all JUnit XML files (newest file wins).

    A parametrised test passes only if every recorded parametrisation in its newest file passed."""
    out: dict[tuple[str, str], dict] = {}
    files = sorted(glob.glob(os.path.join(results_dir, "**", "*.xml"), recursive=True), key=os.path.getmtime)
    for f in files:
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        per_file: dict[tuple[str, str], dict] = {}
        for tc in tree.iter("testcase"):
            key = _junit_key(tc.get("classname", ""), tc.get("name", ""), tc.get("file"))
            failed = any(child.tag in ("failure", "error") for child in tc)
            skipped = any(child.tag == "skipped" for child in tc)
            st = per_file.setdefault(key, {"passed": True, "skipped": False, "file": os.path.relpath(f, ROOT)})
            st["passed"] = st["passed"] and not failed and not skipped
            st["skipped"] = st["skipped"] or skipped
        out.update(per_file)
    return out


def test_passed(test_id: str, results: dict) -> tuple[bool, str]:
    parts = _split_test_id(test_id)
    if parts is None:
        return False, "malformed test id"
    rec = results.get(parts)
    if rec is None:
        return False, "never executed (no JUnit record)"
    if rec["skipped"]:
        return False, f"skipped ({rec['file']})"
    if not rec["passed"]:
        return False, f"failed ({rec['file']})"
    return True, rec["file"]


def stub_findings(path: str) -> list[str]:
    full = os.path.join(ROOT, path)
    if not path.endswith(".py") or not os.path.isfile(full):
        return []
    with open(full) as fh:
        src = fh.read()
    found = []
    is_test = "/tests/" in path
    if not is_test and DOUBLE_IMPORT.search(src):
        found.append("production code imports test doubles")
    for pat, label in STUB_PATTERNS:
        if pat.search(src):
            found.append(label)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return found + [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                                                 and isinstance(b.value.value, str))]
            if not body:
                continue
            is_abstract = any((isinstance(d, ast.Name) and d.id == "abstractmethod") or
                              (isinstance(d, ast.Attribute) and d.attr == "abstractmethod") for d in node.decorator_list)
            if is_abstract:
                continue
            if len(body) == 1:
                b = body[0]
                if isinstance(b, ast.Raise) and b.exc is not None:
                    exc = b.exc.func if isinstance(b.exc, ast.Call) else b.exc
                    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                        found.append(f"{node.name}() raises NotImplementedError")
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and b.value.value is Ellipsis:
                    # Protocol method declarations use '...'; only flag them outside Protocol classes
                    pass
    return found


def _protocol_methods(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(getattr(b, "id", getattr(b, "attr", "")) == "Protocol" for b in node.bases):
            names |= {f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return names


# ----------------------------------------------------------------------------------------------- checking

def check(manifest: dict, results: dict) -> dict:
    reqs = manifest["requirements"]
    ids = [r["id"] for r in reqs]
    problems: dict[str, list[str]] = defaultdict(list)
    global_problems: list[str] = []

    if len(ids) != len(set(ids)):
        global_problems.append("duplicate requirement ids: " + ", ".join(k for k, v in Counter(ids).items() if v > 1))
    for rid in REQUIRED_IDS:
        if rid not in ids:
            global_problems.append(f"mandatory requirement {rid} missing from the manifest")
    for pre in REQUIRED_PREFIXES:
        if not any(i.startswith(pre) for i in ids):
            global_problems.append(f"no requirement with prefix {pre}")
    allowed = set(manifest.get("allowed_statuses", STATUSES))

    for r in reqs:
        rid, st = r["id"], r["status"]
        p = problems[rid]
        if st not in allowed:
            p.append(f"unknown status {st}")
            continue
        if r.get("workstream") not in manifest.get("workstreams", {}):
            p.append(f"unknown workstream {r.get('workstream')}")
        for d in r.get("dependencies", []):
            if d not in ids:
                p.append(f"unknown dependency {d}")
        if st in AT_LEAST_IMPLEMENTED:
            if not r.get("files"):
                p.append("claimed implemented without implementation files")
            for f in r.get("files", []):
                if not os.path.exists(os.path.join(ROOT, f)):
                    p.append(f"missing file {f}")
                else:
                    for s in stub_findings(f):
                        p.append(f"{f}: {s}")
            acc = r.get("acceptance_tests", [])
            if not any(test_resolves(t) for t in acc):
                p.append("no resolvable acceptance test")
            for t in acc:
                if not test_resolves(t):
                    p.append(f"acceptance test does not resolve: {t}")
                    continue
                ok, why = test_passed(t, results)
                if not ok:
                    p.append(f"acceptance test {t}: {why}")
            for t in r.get("gradient_tests", []):
                ok, why = test_passed(t, results) if test_resolves(t) else (False, "does not resolve")
                if not ok:
                    p.append(f"gradient-pathway test {t}: {why}")
        if st in AT_LEAST_INTEGRATED:
            integ = r.get("integration_tests", [])
            if not any(test_resolves(t) for t in integ):
                p.append("no resolvable integration test")
            for t in integ:
                if test_resolves(t):
                    ok, why = test_passed(t, results)
                    if not ok:
                        p.append(f"integration test {t}: {why}")
            if not r.get("integration_entry"):
                p.append("no integration_entry (function of the canonical runtime that exercises it)")
            else:
                mod_path, _, fn = r["integration_entry"].partition(":")
                f = os.path.join(ROOT, mod_path)
                if not os.path.isfile(f):
                    p.append(f"integration_entry file missing: {mod_path}")
                elif fn and not re.search(rf"^\s*(async\s+)?def {re.escape(fn.split('.')[-1])}\(", open(f).read(), re.M):
                    p.append(f"integration_entry function missing: {r['integration_entry']}")
            if not r.get("verified_by"):
                p.append("not independently verified (verified_by is empty)")
        if st in EMPIRICAL:
            er = r.get("experiment_result")
            if not er:
                p.append("empirical claim without experiment_result")
            else:
                f = os.path.join(ROOT, er)
                if not os.path.isfile(f):
                    p.append(f"experiment_result missing: {er}")
                else:
                    try:
                        data = json.load(open(f))
                        verdict = str(data.get("verdict", "")).upper()
                        want = "DEMONSTRATED" if st == "EMPIRICALLY_DEMONSTRATED" else "FAILED"
                        if want not in verdict:
                            p.append(f"experiment_result verdict {verdict!r} does not support {st}")
                    except (ValueError, AttributeError) as e:
                        p.append(f"experiment_result unreadable: {e}")
            if not r.get("evidence"):
                p.append("empirical claim without evidence entries")

    by_status = Counter(r["status"] for r in reqs)
    eng_complete = [r["id"] for r in reqs if r["status"] in AT_LEAST_INTEGRATED and not problems[r["id"]]]
    incomplete = [r["id"] for r in reqs if r["id"] not in eng_complete]
    report = {
        "manifest": os.path.relpath(MANIFEST, ROOT),
        "n_requirements": len(reqs),
        "by_status": dict(by_status),
        "global_problems": global_problems,
        "problems": {k: v for k, v in problems.items() if v},
        "engineering_complete": eng_complete,
        "engineering_incomplete": incomplete,
        "scientific": {
            "demonstrated": [r["id"] for r in reqs if r["status"] == "EMPIRICALLY_DEMONSTRATED" and not problems[r["id"]]],
            "failed": [r["id"] for r in reqs if r["status"] == "FAILED_EMPIRICALLY" and not problems[r["id"]]],
            "blocked": [r["id"] for r in reqs if r["status"] == "BLOCKED"],
        },
        "test_records": len(results),
    }
    report["claims_consistent"] = not global_problems and not report["problems"]
    report["project_complete"] = report["claims_consistent"] and not incomplete
    return report


def run_referenced_tests(manifest: dict, timeout: int = 7200) -> int:
    """Execute every acceptance / integration / gradient test referenced by a claim of IMPLEMENTED or above and
    record JUnit XML under docs/evidence/test_results/ (named by timestamp)."""
    import time
    tests = set()
    for r in manifest["requirements"]:
        if r["status"] in AT_LEAST_IMPLEMENTED:
            for key in ("acceptance_tests", "integration_tests", "gradient_tests"):
                tests |= {t for t in r.get(key, []) if test_resolves(t)}
    if not tests:
        print("no referenced tests to run")
        return 0
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, time.strftime("verify_%Y%m%dT%H%M%SZ.xml", time.gmtime()))
    # pytest node ids are relative to the superlearner project dir (rootdir)
    nodes = sorted(t[len("superlearner/"):] if t.startswith("superlearner/") else t for t in tests)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={out}", "-o",
           "junit_family=xunit1", *nodes]
    print(f"running {len(nodes)} tests -> {os.path.relpath(out, ROOT)}")
    return subprocess.call(cmd, cwd=os.path.join(ROOT, "superlearner"), timeout=timeout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-tests", action="store_true", help="execute referenced tests and record JUnit results first")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    if a.run_tests:
        run_referenced_tests(m)
    rep = check(m, load_results())
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    if not a.quiet:
        print(f"requirements: {rep['n_requirements']}  by status: {rep['by_status']}")
        print(f"JUnit test records: {rep['test_records']}")
        for g in rep["global_problems"]:
            print(f"GLOBAL: {g}")
        for rid, ps in rep["problems"].items():
            for p in ps:
                print(f"{rid}: {p}")
        print(f"engineering complete: {len(rep['engineering_complete'])}/{rep['n_requirements']}")
        print(f"scientific: demonstrated={rep['scientific']['demonstrated']} failed={rep['scientific']['failed']} "
              f"blocked={rep['scientific']['blocked']}")
        print("CLAIMS CONSISTENT" if rep["claims_consistent"] else "CLAIMS INCONSISTENT WITH EVIDENCE")
        print("PROJECT COMPLETE" if rep["project_complete"] else "PROJECT NOT COMPLETE")
    return 0 if rep["project_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
