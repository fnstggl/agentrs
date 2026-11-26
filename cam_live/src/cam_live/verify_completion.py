"""cam_live.verify_completion — the executable completion checker (spec 0.3). MASTER-OWNED.

    python -m cam_live.verify_completion                 # check and print; exit 1 unless engineering-complete
    python -m cam_live.verify_completion --merge-status  # fold docs/status/*.json and G's verdicts into the manifest
    python -m cam_live.verify_completion --json out.json --only MR-07,RC-03

A requirement is engineering-complete only if ALL hold:
  1. it is claimed ``implemented`` by its workstream's status file;
  2. every claimed file exists and contains no placeholder marker (``raise NotImplementedError``, TODO / FIXME,
     or a function whose whole body is ``pass`` / ``...`` outside a Protocol or abstract base class);
  3. it names at least one acceptance test, every named test resolves to a real test function, and each has a
     recorded *passing* result in ``docs/evidence/test_results/*.xml`` (skipped counts as not passed);
  4. it names at least one integration point that imports, and whose module is statically import-reachable from
     a runtime entry point (so a disconnected module cannot be counted);
  5. the independent verifier (workstream G, ``docs/verification/verdicts.json``) marked it ``verified``.

The five project statuses are computed separately and none implies the next:
  1 engineering implementation complete, 2 controlled experimental validation complete,
  3 ready for authorized live deployment, 4 live experiment executed, 5 scientific hypothesis supported.
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(PROJECT, "src")
MANIFEST = os.path.join(PROJECT, "docs", "CAM_LIVE_COMPLETION_MANIFEST.json")
STATUS_DIR = os.path.join(PROJECT, "docs", "status")
RESULTS_DIR = os.path.join(PROJECT, "docs", "evidence", "test_results")
VERDICTS = os.path.join(PROJECT, "docs", "verification", "verdicts.json")
EVIDENCE = os.path.join(PROJECT, "docs", "evidence")

#: modules a running system actually starts from
ENTRY_POINTS = ["cam_live.agent", "cam_live.validation", "cam_live.runtime.cli", "cam_live.runtime.loop_service",
                "cam_live.runtime.verifier_service", "cam_live.runtime.dashboard", "cam_live.harness.service",
                "cam_live.harness.auth.service", "cam_live.harness.auth.launcher", "cam_live.eval.cli"]

#: requirements whose claim is an executed experiment: need an evidence JSON with "executed": true
VALIDATION_REQS = ["NA-13", "MR-07", "RC-03", "RC-04", "SV-05", "SV-06", "SV-07", "SV-08", "SV-09", "SV-10",
                   "SV-12", "CU-33", "CU-34", "LX-11", "IN-07"]

_PLACEHOLDER = re.compile(r"raise\s+NotImplementedError|\bTODO\b|\bFIXME\b")


# ---------------------------------------------------------------------------------------------------------------
def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST) as f:
        return json.load(f)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT, p)


def merge_status(manifest: Dict[str, Any]) -> List[str]:
    """Fold workstream status files and G's verdicts into the manifest. Returns human-readable notes."""
    notes: List[str] = []
    by_id = {r["id"]: r for r in manifest["requirements"]}
    claims: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for path in sorted(glob.glob(os.path.join(STATUS_DIR, "*.json"))):
        try:
            st = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            notes.append(f"unreadable status file {os.path.basename(path)}: {e}")
            continue
        ws = st.get("workstream") or os.path.splitext(os.path.basename(path))[0]
        for rid, claim in (st.get("requirements") or {}).items():
            if rid not in by_id:
                notes.append(f"{ws} claims unknown requirement {rid}")
                continue
            claims.setdefault(rid, []).append((ws, claim))
    order = {"not_started": 0, "partial": 1, "implemented": 2}
    for rid, req in by_id.items():
        cl = claims.get(rid, [])
        files, tests, ips, cmds, ev, lim = [], [], [], [], [], []
        owner_state: Optional[str] = None
        best = "not_started"
        for ws, c in cl:
            files += c.get("files", []); tests += c.get("tests", []); ips += c.get("integration_points", [])
            cmds += c.get("commands", []); ev += c.get("evidence", [])
            if c.get("limitations"):
                lim.append(f"[{ws}] {c['limitations']}")
            s = c.get("state", "not_started")
            s = s if s in order else "partial"
            if ws == req["workstream"]:
                owner_state = s
            if order[s] > order[best]:
                best = s
        req["files"] = sorted(set(files)); req["acceptance_tests"] = sorted(set(tests))
        req["integration_points"] = sorted(set(ips)); req["commands"] = cmds; req["evidence"] = sorted(set(ev))
        req["limitations"] = " | ".join(lim)
        req["claimed_state"] = owner_state if owner_state is not None else best
        req["claimed_by"] = sorted({ws for ws, _ in cl})
    if os.path.exists(VERDICTS):
        for rid, v in json.load(open(VERDICTS)).items():
            if rid in by_id:
                by_id[rid]["verification"] = {"status": v.get("status", "unverified"), "by": v.get("by", "G"),
                                              "notes": v.get("notes", "")}
    return notes


# ---------------------------------------------------------------------------------------------------------------
def placeholder_findings(path: str) -> List[str]:
    out: List[str] = []
    if not path.endswith(".py") or not os.path.exists(path):
        return out
    text = open(path, encoding="utf-8", errors="replace").read()
    for i, line in enumerate(text.splitlines(), 1):
        if _PLACEHOLDER.search(line) and "verify_completion" not in path:
            out.append(f"{os.path.relpath(path, PROJECT)}:{i}: {line.strip()[:80]}")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return out + [f"{os.path.relpath(path, PROJECT)}: syntax error {e}"]

    def abstract(cls: ast.ClassDef) -> bool:
        names = {getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases}
        return bool(names & {"Protocol", "ABC"}) or any("abc" in ast.dump(b) for b in cls.bases)

    def walk(node: ast.AST, in_abstract: bool) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                walk(ch, in_abstract or abstract(ch))
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in ch.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                                                   and isinstance(b.value.value, str))]
                deco = {ast.dump(d) for d in ch.decorator_list}
                is_abs = in_abstract or any("abstractmethod" in d or "overload" in d for d in deco)
                empty = all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                        and b.value.value is Ellipsis) for b in body)
                if empty and not is_abs and not ch.name.startswith("__"):
                    out.append(f"{os.path.relpath(path, PROJECT)}:{ch.lineno}: empty function {ch.name}()")
                walk(ch, in_abstract)
            else:
                walk(ch, in_abstract)
    walk(tree, False)
    return out


def import_graph() -> Dict[str, Set[str]]:
    """module -> cam_live modules it imports (static)."""
    graph: Dict[str, Set[str]] = {}
    for root, _dirs, files in os.walk(os.path.join(SRC, "cam_live")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, SRC)[:-3].replace(os.sep, ".")
            mod = rel[:-9] if rel.endswith(".__init__") else rel
            pkg = mod if fn == "__init__.py" else mod.rsplit(".", 1)[0]
            deps: Set[str] = set()
            try:
                tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                graph[mod] = deps
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    deps |= {a.name for a in node.names if a.name.startswith("cam_live")}
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = pkg.split(".")
                        base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                        target = ".".join(base + ([node.module] if node.module else []))
                    else:
                        target = node.module or ""
                    if target.startswith("cam_live"):
                        deps.add(target)
                        deps |= {f"{target}.{a.name}" for a in node.names}
                elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                        "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                        and str(node.args[0].value).startswith("cam_live"):
                    deps.add(str(node.args[0].value))
            graph[mod] = deps
    return graph


def reachable(graph: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = [e for e in ENTRY_POINTS if e in graph]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        parts = m.split(".")
        for i in range(1, len(parts)):
            p = ".".join(parts[:i])
            if p in graph and p not in seen:
                stack.append(p)
        for d in graph.get(m, ()):  # a dep may be module.symbol: keep the longest prefix that is a module
            while d and d not in graph:
                d = d.rpartition(".")[0]
            if d and d not in seen:
                stack.append(d)
    return seen


def resolve_integration_point(ip: str, graph: Dict[str, Set[str]], reach: Set[str]) -> Tuple[bool, str]:
    mod = ip
    while mod and mod not in graph:
        mod = mod.rpartition(".")[0]
    if not mod:
        return False, f"{ip}: no such module"
    try:
        obj: Any = importlib.import_module(mod)
        for attr in [a for a in ip[len(mod):].split(".") if a]:
            obj = getattr(obj, attr)
    except Exception as e:  # noqa: BLE001
        return False, f"{ip}: {type(e).__name__}: {e}"
    if mod not in reach:
        return False, f"{ip}: module {mod} is not import-reachable from any runtime entry point"
    return True, ""


def recorded_results() -> Dict[Tuple[str, str], Tuple[float, str]]:
    """(classname, test name) -> (mtime of the newest report mentioning it, outcome)."""
    out: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.xml")):
        mt = os.path.getmtime(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for tc in root.iter("testcase"):
            kinds = {c.tag for c in tc}
            outcome = "failed" if kinds & {"failure", "error"} else "skipped" if "skipped" in kinds else "passed"
            key = (tc.get("classname", ""), tc.get("name", ""))
            if key not in out or mt >= out[key][0]:
                out[key] = (mt, outcome)
    return out


def check_test(node_id: str, results: Dict[Tuple[str, str], Tuple[float, str]]) -> Tuple[bool, str]:
    parts = node_id.split("::")
    path = parts[0][len("cam_live/"):] if parts[0].startswith("cam_live/") else parts[0]
    full = _abs(path)
    if not os.path.exists(full):
        return False, f"{node_id}: test file missing"
    if len(parts) < 2:
        return False, f"{node_id}: not a test node id (file::test)"
    fn = parts[-1].split("[")[0]
    if not re.search(rf"def\s+{re.escape(fn)}\s*\(", open(full, encoding="utf-8", errors="replace").read()):
        return False, f"{node_id}: test function not found"
    cls = path[:-3].replace("/", ".") + ("." + ".".join(parts[1:-1]) if len(parts) > 2 else "")
    outcomes = [o for (c, n), (_, o) in results.items()
                if (c == cls or c.endswith("." + cls) or cls.endswith("." + c)) and (n == parts[-1] or n.split("[")[0] == fn)]
    if not outcomes:
        return False, f"{node_id}: no recorded result"
    if any(o == "failed" for o in outcomes):
        return False, f"{node_id}: recorded FAILED"
    if all(o == "skipped" for o in outcomes):
        return False, f"{node_id}: only skipped results recorded (needs a gpu / privileged / browser run)"
    return True, ""


def check_requirement(req: Dict[str, Any], graph, reach, results) -> Dict[str, Any]:
    problems: List[str] = []
    state = req.get("claimed_state", "not_started")
    if state != "implemented":
        problems.append(f"claimed state is '{state}'")
    files = req.get("files", [])
    if not files:
        problems.append("no implementation files named")
    for f in files:
        if not os.path.exists(_abs(f)):
            problems.append(f"missing file {f}")
        else:
            problems += placeholder_findings(_abs(f))
    tests = req.get("acceptance_tests", [])
    if not tests:
        problems.append("no acceptance test named")
    n_pass = 0
    for t in tests:
        ok, why = check_test(t, results)
        n_pass += ok
        if not ok:
            problems.append(why)
    ips = req.get("integration_points", [])
    if not ips:
        problems.append("no runtime integration point named")
    ok_ip = False
    for ip in ips:
        ok, why = resolve_integration_point(ip, graph, reach)
        ok_ip |= ok
        if not ok:
            problems.append(why)
    if ips and ok_ip:   # one reachable integration point is enough; others must at least import
        problems = [p for p in problems if "not import-reachable" not in p]
    for e in req.get("evidence", []):
        if not os.path.exists(_abs(e)):
            problems.append(f"missing evidence file {e}")
    eng_ok = not problems
    verified = req.get("verification", {}).get("status") == "verified"
    if req.get("verification", {}).get("status") == "rejected":
        problems.append("independent verification REJECTED: " + req["verification"].get("notes", ""))
    integrated = state == "implemented" and ok_ip
    req["engineering_status"] = ("integrated" if integrated else state)
    req["test_results"] = [f"{n_pass}/{len(tests)} named tests have a recorded pass"]
    return {"id": req["id"], "workstream": req["workstream"], "title": req["title"], "engineering_ok": eng_ok,
            "verified": verified, "complete": eng_ok and verified, "problems": problems}


def executed_evidence() -> Dict[str, bool]:
    """requirement id -> an evidence JSON exists that declares the experiment executed for it."""
    found: Dict[str, bool] = {}
    for path in glob.glob(os.path.join(EVIDENCE, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("executed") is True:
            for rid in d.get("requirements", []):
                found[rid] = True
    return found


def run(only: Optional[List[str]] = None, do_merge: bool = False) -> Dict[str, Any]:
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    manifest = load_manifest()
    notes = merge_status(manifest)
    graph = import_graph()
    reach = reachable(graph)
    results = recorded_results()
    rows = [check_requirement(r, graph, reach, results) for r in manifest["requirements"]
            if not only or r["id"] in only]
    eng = all(r["complete"] for r in rows) and not only
    ev = executed_evidence()
    missing_val = [r for r in VALIDATION_REQS if not ev.get(r)]
    live = _live_status()
    levels = {
        "1_engineering_implementation_complete": bool(eng),
        "2_controlled_experimental_validation_complete": bool(eng and not missing_val),
        "3_ready_for_authorized_live_deployment": bool(eng and live["preflight_ready"]),
        "4_live_experiment_executed": bool(live["live_run_executed"]),
        "5_scientific_hypothesis_supported": bool(live["live_run_executed"] and live["transfer_supported"]),
    }
    manifest["status_levels"] = levels
    if do_merge:
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, MANIFEST)
    return {"levels": levels, "rows": rows, "notes": notes, "missing_validation_evidence": missing_val,
            "live": live, "counts": {"total": len(rows), "engineering_ok": sum(r["engineering_ok"] for r in rows),
                                     "verified": sum(r["verified"] for r in rows),
                                     "complete": sum(r["complete"] for r in rows)},
            "unreachable_modules": sorted(m for m in graph if m not in reach and not m.endswith("__main__")
                                          and ".diagnostics" not in m and not m.endswith(".bench"))}


def _live_status() -> Dict[str, Any]:
    out = {"preflight_ready": False, "live_run_executed": False, "transfer_supported": False, "blockers": []}
    p = os.path.join(EVIDENCE, "preflight.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            out["preflight_ready"] = bool(d.get("live_ready"))
            out["blockers"] = d.get("blockers", [])
        except Exception:  # noqa: BLE001
            pass
    p = os.path.join(EVIDENCE, "live_run.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            out["live_run_executed"] = bool(d.get("mode") == "live" and d.get("verified_outcome_signature_ok"))
            out["transfer_supported"] = bool(d.get("transfer_supported"))
        except Exception:  # noqa: BLE001
            pass
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam-live verify-completion", description=__doc__.split("\n\n")[0])
    ap.add_argument("--merge-status", action="store_true", help="write merged claims and statuses into the manifest")
    ap.add_argument("--only", default="", help="comma-separated requirement ids")
    ap.add_argument("--json", default="", help="write the full report to this path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    rep = run([x.strip() for x in a.only.split(",") if x.strip()] or None, a.merge_status)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    if not a.quiet:
        c = rep["counts"]
        print(f"requirements: {c['total']}  engineering-ok: {c['engineering_ok']}  independently verified: "
              f"{c['verified']}  complete: {c['complete']}")
        for r in rep["rows"]:
            if not r["complete"]:
                why = r["problems"][:3] + ([] if r["verified"] else ["not independently verified"])
                print(f"  [{'eng-ok' if r['engineering_ok'] else 'OPEN  '}] {r['id']} ({r['workstream']}) {r['title'][:70]}")
                for w in why[:4]:
                    print(f"           - {w}")
        for n in rep["notes"]:
            print("note:", n)
        if rep["missing_validation_evidence"]:
            print("validation evidence missing for:", ", ".join(rep["missing_validation_evidence"]))
        if rep["live"]["blockers"]:
            print("live blockers:", "; ".join(map(str, rep["live"]["blockers"]))[:600])
        print("status levels:")
        for k, v in rep["levels"].items():
            print(f"  {k}: {'YES' if v else 'no'}")
    return 0 if rep["levels"]["1_engineering_implementation_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
