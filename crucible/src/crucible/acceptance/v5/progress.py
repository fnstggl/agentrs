"""The machine-readable progress ledger docs/v5/PROGRESS.json (Part 1.5, Clarification 3.7) and its validator.

Each milestone records the eight things Part 1.5 requires: completed work, branch and commit, active workstreams,
failed tests, known integration problems, remaining requirements (exact ids), next concrete actions and compute usage.
`interrupted_workstreams` tracks sub-agents that terminated early and who took their work over (Part 1.5).

    python -m crucible.acceptance.v5 progress validate
    python -m crucible.acceptance.v5 progress milestone --title "..." --completed "..." --next "..." [--junit x.xml]
    python -m crucible.acceptance.v5 progress interrupted --workstream B --why "usage limit" --recovery-branch c5c/w1-x \
        --reassigned-to "P (c5c/w3-qa)" --status reassigned
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import xml.etree.ElementTree as ET

from . import manifest as M

PROGRESS_REL = "docs/v5/PROGRESS.json"
SCHEMA = "crucible.v5.progress/1"
FIELDS = {"completed_work": list, "branch": str, "commit": str, "active_workstreams": list, "failed_tests": list,
          "integration_problems": list, "remaining_requirements": list, "next_actions": list, "compute_usage": dict}
NONEMPTY = ("completed_work", "branch", "commit", "next_actions", "compute_usage")


def _git(root: str, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)


def load(root: str | None = None) -> dict:
    with open(os.path.join(root or M.crucible_root(), PROGRESS_REL)) as fh:
        return json.load(fh)


def save(d: dict, root: str | None = None) -> None:
    p = os.path.join(root or M.crucible_root(), PROGRESS_REL)
    with open(p, "w") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def manifest_ids_at(root: str, commit: str) -> set[str] | None:
    p = _git(root, "show", f"{commit}:./docs/v5/ACCEPTANCE_MANIFEST.json")
    if p.returncode:
        return None
    try:
        return {e["id"] for e in json.loads(p.stdout)["requirements"]}
    except (ValueError, KeyError):
        return None


def reviewed_at(root: str, commit: str) -> set[str]:
    p = _git(root, "ls-tree", "--name-only", commit, "docs/v5/reviews/")
    return {os.path.basename(x)[:-5] for x in p.stdout.split() if x.endswith(".json")}


def validate(root: str | None = None) -> list[str]:
    """Problems with the ledger; [] when valid."""
    root = root or M.crucible_root()
    try:
        d = load(root)
    except (OSError, ValueError) as exc:
        return [f"progress ledger unreadable: {exc}"]
    probs = []
    if d.get("schema") != SCHEMA:
        probs.append("schema mismatch")
    ms = d.get("milestones") or []
    if not ms:
        probs.append("no milestones")
    seen = set()
    for m in ms:
        mid = m.get("id", "?")
        if mid in seen:
            probs.append(f"{mid}: duplicate id")
        seen.add(mid)
        for f, typ in FIELDS.items():
            if f not in m:
                probs.append(f"{mid}: missing field {f}")
            elif not isinstance(m[f], typ):
                probs.append(f"{mid}: field {f} is not a {typ.__name__}")
            elif f in NONEMPTY and not m[f]:
                probs.append(f"{mid}: field {f} is empty")
        c = m.get("commit")
        if c and _git(root, "cat-file", "-e", f"{c}^{{commit}}").returncode:
            probs.append(f"{mid}: commit {c} unknown")
        ids = manifest_ids_at(root, c) if c else None
        rem = set(m.get("remaining_requirements") or [])
        if ids is not None:
            extra = sorted(rem - ids)
            if extra:
                probs.append(f"{mid}: remaining ids not in the manifest at that commit: {extra[:5]}")
    for w in d.get("interrupted_workstreams", []):
        if not w.get("reassigned_to") or not w.get("status"):
            probs.append(f"interrupted workstream {w.get('workstream')}: no reassignment or status")
    return probs


def metrics(root: str | None = None) -> dict:
    root = root or M.crucible_root()
    probs = validate(root)
    try:
        d = load(root)
    except (OSError, ValueError):
        d = {"milestones": []}
    ms = d.get("milestones") or []
    missing = sum(1 for m in ms for f in FIELDS if f not in m or (f in NONEMPTY and not m.get(f)))
    omits = None
    if ms:
        last = ms[-1]
        ids = manifest_ids_at(root, last.get("commit", "")) or set()
        unverified = ids - reviewed_at(root, last.get("commit", ""))
        omits = len(unverified - set(last.get("remaining_requirements") or []))
    iw = d.get("interrupted_workstreams", [])
    return {"milestones_recorded": len(ms), "ledger_valid": not probs, "milestones_missing_fields": missing,
            "latest_milestone_omits_unverified_requirements": omits if omits is not None else 1,
            "ledger_lists_remaining_requirements": bool(ms) and bool(ms[-1].get("remaining_requirements")) and omits == 0,
            "interrupted_workstreams_recorded": len(iw),
            "interrupted_workstreams_unassigned": sum(1 for w in iw if not w.get("reassigned_to"))}, probs


def compute_usage(since: float | None = None) -> dict:
    """Summary of the slot scheduler's usage log (crucible.scale.slots)."""
    from ...scale.slots import slots_dir
    p = os.path.join(slots_dir(), "usage.jsonl")
    n = cpu_s = wall_s = 0.0
    gpu_jobs = 0
    try:
        for line in open(p):
            r = json.loads(line)
            if since and r.get("end", 0) < since:
                continue
            n += 1
            cpu_s += float(r.get("cpu_s") or 0)
            wall_s += float(r.get("wall_s") or 0)
            gpu_jobs += 1 if r.get("gpu_mb") else 0
    except (OSError, ValueError):
        pass
    return {"source": p, "scheduled_runs": int(n), "cpu_hours": round(cpu_s / 3600, 2),
            "wall_hours": round(wall_s / 3600, 2), "gpu_jobs": gpu_jobs}


def failed_tests(junit: list[str]) -> list[str]:
    out = []
    for j in junit:
        for tc in ET.parse(j).getroot().iter("testcase"):
            if tc.find("failure") is not None or tc.find("error") is not None:
                out.append(f"{tc.get('classname')}::{tc.get('name')}")
    return out


def add_milestone(*, title: str, completed: list[str], next_actions: list[str], problems: list[str] = (),
                  workstreams: list[str] = (), junit: list[str] = (), root: str | None = None) -> dict:
    from .checker import verify_all
    root = root or M.crucible_root()
    d = load(root)
    res = verify_all(root)
    prev = d["milestones"][-1] if d["milestones"] else None
    since = None
    if prev:
        try:
            since = _dt.datetime.fromisoformat(prev["date"]).timestamp()
        except (KeyError, ValueError):
            since = None
    ms = {"id": f"M{len(d['milestones'])}", "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
          "title": title, "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
          "commit": _git(root, "rev-parse", "HEAD").stdout.strip(), "completed_work": list(completed),
          "active_workstreams": list(workstreams), "failed_tests": failed_tests(list(junit)),
          "integration_problems": list(problems) + res["global_problems"],
          "remaining_requirements": [r["id"] for r in res["rows"] if r["status"] != "VERIFIED"],
          "next_actions": list(next_actions), "compute_usage": compute_usage(since),
          "project_status": res["project_status"]}
    d["milestones"].append(ms)
    save(d, root)
    return ms


INTERRUPTED_STATUSES = ("reassigned", "resumed", "completed", "abandoned")


def add_interrupted(*, workstream: str, why: str, recovery_branch: str, reassigned_to: str, status: str = "reassigned",
                    root: str | None = None) -> dict:
    """Record (or update) a sub-agent that terminated early and who took its work over (Part 1.5). An entry is keyed
    by workstream and recovery branch; earlier states are kept in its `history`."""
    if status not in INTERRUPTED_STATUSES:
        raise ValueError(f"status must be one of {INTERRUPTED_STATUSES}")
    if not (workstream and why and recovery_branch and reassigned_to):
        raise ValueError("workstream, why, recovery branch and reassignment are all required")
    root = root or M.crucible_root()
    d = load(root)
    iw = d.setdefault("interrupted_workstreams", [])
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    e = next((w for w in iw if w.get("workstream") == workstream and w.get("recovery_branch") == recovery_branch), None)
    if e is None:
        e = {"workstream": workstream, "interrupted": why, "recovery_branch": recovery_branch,
             "reassigned_to": reassigned_to, "status": status, "recorded_at": now}
        iw.append(e)
    else:
        e.setdefault("history", []).append({k: e.get(k) for k in ("interrupted", "reassigned_to", "status")} |
                                           {"until": now})
        e.update(interrupted=why, reassigned_to=reassigned_to, status=status, recorded_at=now)
    save(d, root)
    return e
