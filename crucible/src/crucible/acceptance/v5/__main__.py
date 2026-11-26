"""Crucible 5.0 acceptance command line.

    python -m crucible.acceptance.v5 --verify-all              # exit 0 only if every requirement is VERIFIED
    python -m crucible.acceptance.v5 report [--gate B]         # status table, writes docs/v5/evidence/checker_report.json
    python -m crucible.acceptance.v5 build-manifest [--decision D-7]
    python -m crucible.acceptance.v5 run --gate B | --requirement B.3   # run gate tests, attach JUnit to evidence
    python -m crucible.acceptance.v5 review --requirement B.3 --reviewer NAME --verdict approved --notes "..."

Run from the `crucible/` directory with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import uuid

from . import manifest as M
from .checker import verify_all
from .evidence import gate_of, sha256_file


def _report(res: dict, gate: str | None, out=sys.stdout) -> None:
    rows = [r for r in res["rows"] if gate is None or gate_of(r["id"]) == gate]
    by_gate: dict[str, dict[str, int]] = {}
    for r in rows:
        g = gate_of(r["id"])
        by_gate.setdefault(g, {})
        by_gate[g][r["status"]] = by_gate[g].get(r["status"], 0) + 1
    print(f"HEAD {res['head'][:9]}  requirements {len(rows)}  project status: {res['project_status']}", file=out)
    for g in sorted(by_gate, key=lambda x: (x == "SP", x)):
        cells = ", ".join(f"{k} {v}" for k, v in sorted(by_gate[g].items()))
        print(f"  gate {g:3s} {cells}", file=out)
    if gate:
        for r in rows:
            print(f"  {r['id']:8s} {r['status']:22s} {'; '.join(r['problems'][:3])[:160]}", file=out)


def _run(args) -> int:
    m = M.load()
    if args.requirement:
        reqs = [e for e in m["requirements"] if e["id"] == args.requirement]
    else:
        reqs = [e for e in m["requirements"] if e["gate"] == args.gate]
    if not reqs:
        print("no such requirement or gate", file=sys.stderr)
        return 2
    tests = sorted({t for e in reqs for t in e["tests"] if os.path.exists(os.path.join(M.crucible_root(), t))})
    if not tests:
        print("no test files exist yet for this selection", file=sys.stderr)
        return 2
    run_id = uuid.uuid4().hex[:12]
    gate = gate_of(reqs[0]["id"])
    jdir = os.path.join(M.docs_dir(), "evidence", f"gate_{gate}")
    os.makedirs(jdir, exist_ok=True)
    junit = os.path.join(jdir, f"junit_{run_id}.xml")
    env = dict(os.environ, CRUCIBLE_V5_RUN_ID=run_id)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *tests]
    rc = subprocess.call(cmd, cwd=M.crucible_root(), env=env)
    attached = 0
    for e in m["requirements"]:
        p = os.path.join(jdir, f"{e['id']}.json")
        if not os.path.exists(p):
            continue
        ev = json.load(open(p))
        if ev.get("run_id") != run_id or not os.path.exists(junit):
            continue
        ev["junit"] = {"path": os.path.relpath(junit, M.crucible_root()), "sha256": sha256_file(junit)}
        json.dump(ev, open(p, "w"), indent=1)
        rel = os.path.relpath(p, M.crucible_root())
        if rel not in e["evidence"]:
            e["evidence"].append(rel)
        e["commit"] = ev["tree"]["commit"]
        attached += 1
    M.save(m)
    print(f"pytest exit {rc}; JUnit {os.path.relpath(junit, M.crucible_root())}; evidence records attached: {attached}")
    return rc


def _review(args) -> int:
    rid = args.requirement
    ev_path = os.path.join(M.docs_dir(), "evidence", f"gate_{gate_of(rid)}", f"{rid}.json")
    if not os.path.exists(ev_path):
        print("no evidence to review", file=sys.stderr)
        return 2
    ev = json.load(open(ev_path))
    if args.reviewer == ev.get("author"):
        print("the reviewer must not be the evidence's author", file=sys.stderr)
        return 2
    d = os.path.join(M.docs_dir(), "reviews")
    os.makedirs(d, exist_ok=True)
    rec = {"schema": "crucible.v5.review/1", "requirement": rid, "evidence_sha256": sha256_file(ev_path),
           "reviewer": args.reviewer, "verdict": args.verdict, "notes": args.notes,
           "reviewed_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    json.dump(rec, open(os.path.join(d, f"{rid}.json"), "w"), indent=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.acceptance.v5", description=__doc__.splitlines()[0])
    ap.add_argument("--verify-all", action="store_true", help="check every requirement; nonzero exit unless all VERIFIED")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("report")
    r.add_argument("--gate")
    b = sub.add_parser("build-manifest")
    b.add_argument("--decision")
    ru = sub.add_parser("run")
    g = ru.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate")
    g.add_argument("--requirement")
    rv = sub.add_parser("review")
    rv.add_argument("--requirement", required=True)
    rv.add_argument("--reviewer", required=True)
    rv.add_argument("--verdict", choices=("approved", "rejected"), required=True)
    rv.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    if a.verify_all or a.cmd == "report":
        res = verify_all()
        _report(res, getattr(a, "gate", None))
        out = os.path.join(M.docs_dir(), "evidence", "checker_report.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        return 0 if (res["ok"] or a.cmd == "report") else 1
    if a.cmd == "build-manifest":
        m = M.build(decision=a.decision)
        print(f"{len(m['requirements'])} requirements written to {M.manifest_path()}")
        return 0
    if a.cmd == "run":
        return _run(a)
    if a.cmd == "review":
        return _review(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
