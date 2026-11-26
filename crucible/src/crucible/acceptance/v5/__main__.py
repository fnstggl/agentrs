"""Crucible 5.0 acceptance command line.

    python -m crucible.acceptance.v5 --verify-all [--reproduce]   # exit 0 only when CRUCIBLE_5_COMPLETE
    python -m crucible.acceptance.v5 report [--gate B] [--write]   # status table (--write: docs/v5/evidence/checker_report.json)
    python -m crucible.acceptance.v5 build-manifest [--decision D-7]
    python -m crucible.acceptance.v5 merge-workstreams             # implementation files/tests from docs/v5/workstreams/*.json
    python -m crucible.acceptance.v5 run --gate B | --requirement B.3 --agent ws-B [--commit HEAD]
    python -m crucible.acceptance.v5 review --requirement B.3 --agent qa-1 --verdict approved --notes "..."
    python -m crucible.acceptance.v5 sync-status                   # write the checker's statuses into the manifest
    python -m crucible.acceptance.v5 final-report [--out docs/v5/FINAL_REPORT.md]
    python -m crucible.acceptance.v5 progress validate | progress milestone --title ... --completed ... --next ...
    python -m crucible.acceptance.v5 progress interrupted --workstream B --why ... --recovery-branch ... --reassigned-to ...

Exit codes of --verify-all: 0 CRUCIBLE_5_COMPLETE, 1 IN_PROGRESS, 3 INCOMPLETE_BLOCKED.
Run from the `crucible/` directory with PYTHONPATH=src. `run` and `review` work in fresh clones of committed trees;
commit what they write (docs/v5/evidence/, docs/v5/reviews/) for the checker to accept it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from . import manifest as M
from .checker import verify_all
from .evidence import gate_of

EXIT = {"CRUCIBLE_5_COMPLETE": 0, "IN_PROGRESS": 1, "INCOMPLETE_BLOCKED": 3}


def _report(res: dict, gate: str | None, out=sys.stdout) -> None:
    rows = [r for r in res["rows"] if gate is None or gate_of(r["id"]) == gate]
    by_gate: dict[str, dict[str, int]] = {}
    for r in rows:
        g = gate_of(r["id"])
        by_gate.setdefault(g, {})
        by_gate[g][r["status"]] = by_gate[g].get(r["status"], 0) + 1
    print(f"HEAD {res['head'][:9]}  requirements {len(rows)}  project status: {res['project_status']}", file=out)
    for p in res.get("global_problems", [])[:12]:
        print(f"  project problem: {p}", file=out)
    for g in sorted(by_gate, key=lambda x: (x == "SP", x)):
        cells = ", ".join(f"{k} {v}" for k, v in sorted(by_gate[g].items()))
        print(f"  gate {g:3s} {cells}", file=out)
    if gate:
        for r in rows:
            print(f"  {r['id']:8s} {r['status']:22s} {'; '.join(r['problems'][:3])[:160]}", file=out)


def _sync_status(res: dict) -> int:
    m = M.load()
    by = {r["id"]: r for r in res["rows"]}
    for e in m["requirements"]:
        r = by.get(e["id"])
        if not r:
            continue
        if e.get("status") == "BLOCKED" and r["status"] == "BLOCKED":
            continue
        e["status"] = r["status"] if r["status"] in M.STATUSES else "IN_PROGRESS"
        e["evidence"] = [r["evidence"]] if r.get("evidence") else []
        e["review"] = r.get("review")
        e["commit"] = r.get("commit")
    M.save(m)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.acceptance.v5", description=__doc__.splitlines()[0])
    ap.add_argument("--verify-all", action="store_true", help="check every requirement; exit 0 only when complete")
    ap.add_argument("--reproduce", action="store_true", help="with --verify-all: re-execute every VERIFIED requirement "
                                                             "in a fresh clone and compare")
    ap.add_argument("--agent", help="agent name for --reproduce runs")
    ap.add_argument("--out", help="where --verify-all writes its JSON report (default: a temporary file)")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("report")
    r.add_argument("--gate")
    r.add_argument("--write", action="store_true")
    b = sub.add_parser("build-manifest")
    b.add_argument("--decision")
    sub.add_parser("merge-workstreams")
    sub.add_parser("sync-status")
    ru = sub.add_parser("run")
    g = ru.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate")
    g.add_argument("--requirement")
    ru.add_argument("--agent")
    ru.add_argument("--commit", default="HEAD")
    ru.add_argument("--keep", action="store_true", help="keep the clone (for debugging)")
    rv = sub.add_parser("review")
    rv.add_argument("--requirement", required=True)
    rv.add_argument("--agent", "--reviewer", dest="agent")
    rv.add_argument("--verdict", choices=("approved", "rejected"), required=True)
    rv.add_argument("--notes", default="")
    fr = sub.add_parser("final-report")
    fr.add_argument("--out", default=None)
    pg = sub.add_parser("progress")
    pg.add_argument("action", choices=("validate", "milestone", "interrupted"))
    pg.add_argument("--title", default="")
    pg.add_argument("--completed", action="append", default=[])
    pg.add_argument("--next", dest="next_actions", action="append", default=[])
    pg.add_argument("--problem", dest="problems", action="append", default=[])
    pg.add_argument("--workstream", dest="workstreams", action="append", default=[])
    pg.add_argument("--junit", action="append", default=[])
    pg.add_argument("--why", default="", help="interrupted: what interrupted the workstream")
    pg.add_argument("--recovery-branch", default="", help="interrupted: the branch holding its committed work")
    pg.add_argument("--reassigned-to", default="", help="interrupted: who took the work over")
    pg.add_argument("--status", default="reassigned", help="interrupted: reassigned | resumed | completed | abandoned")
    a = ap.parse_args(argv)
    if a.verify_all or a.cmd in ("report", "sync-status"):
        res = verify_all(reproduce=bool(a.verify_all and a.reproduce), agent=a.agent)
        _report(res, getattr(a, "gate", None))
        if a.cmd == "sync-status":
            return _sync_status(res)
        if a.cmd == "report" and a.write:
            out = os.path.join(M.docs_dir(), "evidence", "checker_report.json")
        else:
            out = a.out or os.path.join(tempfile.gettempdir(), "crucible-v5-checker-report.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"report: {out}")
        return 0 if a.cmd == "report" else EXIT[res["project_status"]]
    if a.cmd == "build-manifest":
        m = M.build(decision=a.decision)
        print(f"{len(m['requirements'])} requirements written to {M.manifest_path()}")
        return 0
    if a.cmd == "merge-workstreams":
        rep = M.merge_workstreams()
        print(f"files added for {len(rep['added'])} requirements; skipped paths for {len(rep['skipped'])}; "
              f"unknown ids {rep['unknown_ids']}")
        for rid, sk in sorted(rep["skipped"].items()):
            for p, why in sk[:4]:
                print(f"  {rid}: skipped {p!r} ({why})")
        return 0
    if a.cmd == "run":
        from .official import run
        res = run(requirement=a.requirement, gate=a.gate, agent=a.agent, commit=a.commit, keep=a.keep)
        print(f"run {res['run_id']} at {res['commit'][:9]}: pytest exit {res['pytest_rc']}; evidence installed: "
              f"{sorted(res['installed'])}; test files missing at the commit: {res['tests_missing_at_commit']}")
        return 0 if res["pytest_rc"] == 0 else 1
    if a.cmd == "review":
        from .official import review
        rec = review(requirement=a.requirement, agent=a.agent, verdict=a.verdict, notes=a.notes)
        rp = rec["reproduction"]
        print(f"review of {a.requirement}: {rec['verdict']} (reproduction {rp['run_id']}: pytest exit {rp['pytest_rc']}, "
              f"mismatches {rp['mismatches'][:3]}, criteria failures {rp['criteria_failures'][:3]})")
        return 0 if rec["verdict"] == a.verdict else 1
    if a.cmd == "final-report":
        from .report import write_report
        path = write_report(out=a.out)
        print(f"final report: {path}")
        return 0
    if a.cmd == "progress":
        from . import progress
        if a.action == "validate":
            probs = progress.validate()
            for p in probs:
                print(f"  {p}")
            print(f"{len(probs)} problems")
            return 1 if probs else 0
        if a.action == "interrupted":
            if len(a.workstreams) != 1:
                raise SystemExit("progress interrupted takes exactly one --workstream")
            e = progress.add_interrupted(workstream=a.workstreams[0], why=a.why, recovery_branch=a.recovery_branch,
                                         reassigned_to=a.reassigned_to, status=a.status)
            print(f"interrupted workstream {e['workstream']} ({e['recovery_branch']}): {e['status']}, "
                  f"reassigned to {e['reassigned_to']}")
            return 0
        ms = progress.add_milestone(title=a.title, completed=a.completed, next_actions=a.next_actions,
                                    problems=a.problems, workstreams=a.workstreams, junit=a.junit)
        print(f"milestone {ms['id']} recorded at {ms['commit'][:9]}: {len(ms['remaining_requirements'])} requirements "
              "remaining")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
