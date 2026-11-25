"""The acceptance manifest (`docs/v5/ACCEPTANCE_MANIFEST.json`): one entry per mandatory requirement.

`build()` derives the entries from MASTER_SPEC.md (gate wording), MASTER_SPEC_CLARIFICATIONS.md (Clarification 1) and
`requirements.py` (criteria, owners, dependencies), and merges the result with an existing manifest so that recorded
status, files, tests, evidence, reviews and commits are kept. Criteria are under change control: an entry's
`criteria_history` holds every version, and each change after the first must cite a decision (`D-<n>`) recorded in
DECISION_LOG.md. `build()` refuses to change criteria silently.
"""
from __future__ import annotations

import json
import os
import re

from . import requirements as R

STATUSES = ("NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED_UNVERIFIED", "FAILED", "BLOCKED", "VERIFIED")
ENTRY_FIELDS = ("id", "gate", "requirement", "workstream", "dependencies", "implementation_files", "tests",
                "verify_command", "criteria", "criteria_history", "status", "evidence", "review", "commit", "source")


def crucible_root() -> str:
    """The `crucible/` directory of this checkout."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


def docs_dir(root: str | None = None) -> str:
    return os.path.join(root or crucible_root(), "docs", "v5")


def manifest_path(root: str | None = None) -> str:
    return os.path.join(docs_dir(root), "ACCEPTANCE_MANIFEST.json")


def spec_gates(root: str | None = None) -> dict:
    """Gate letter -> {"title", "bullets"} from MASTER_SPEC.md Part 17."""
    s = open(os.path.join(docs_dir(root), "MASTER_SPEC.md")).read()
    p17 = s[s.index("# PART 17"):s.index("# PART 18")]
    out = {}
    for g in re.split(r"\n## Gate ", p17)[1:]:
        head, body = g.split("\n", 1)
        letter, title = head.split(":", 1)
        req = body.split("Required:", 1)[1] if "Required:" in body else body
        out[letter.strip()] = {"title": title.strip(),
                               "bullets": [b.strip()[2:].strip() for b in req.split("\n") if b.strip().startswith("- ")]}
    return out


def _verify_command(gate: str, rid: str) -> str:
    return f"PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement {rid}"


def _tests_for(gate: str) -> list[str]:
    return [f"tests/v5/test_gate_{gate.replace('-', '_').replace('.', '_')}.py"]


def derive(root: str | None = None) -> list[dict]:
    gates = spec_gates(root)
    entries = []
    for letter, (ws, deps, crits) in R.GATE_ITEMS.items():
        bullets = gates[letter]["bullets"]
        if len(bullets) != len(crits):
            raise ValueError(f"gate {letter}: spec has {len(bullets)} items, requirements.py has {len(crits)}")
        for i, (text, criteria) in enumerate(zip(bullets, crits), 1):
            rid = f"{letter}.{i}"
            clarified = rid in R.CLARIFIED
            entries.append({
                "id": rid, "gate": letter, "requirement": R.CLARIFIED.get(rid, text), "workstream": ws,
                "dependencies": list(deps), "criteria": criteria,
                "source": {"spec": f"MASTER_SPEC.md Part 17, Gate {letter} item {i}", "spec_text": text,
                           "gate_title": R.GATE_TITLE_OVERRIDE.get(letter, gates[letter]["title"]),
                           "clarification": "MASTER_SPEC_CLARIFICATIONS.md Clarification 1" if clarified else None}})
    for rid, gate, text, ws, criteria in R.GATE_NOTES:
        entries.append({"id": rid, "gate": gate, "requirement": text, "workstream": ws, "dependencies": [gate],
                        "criteria": criteria, "source": {"spec": f"MASTER_SPEC.md Part 17, Gate {gate} (mandatory note)",
                                                         "spec_text": text, "gate_title": gates[gate]["title"],
                                                         "clarification": None}})
    for rid, text, ws, deps, criteria in R.SPEC_ITEMS:
        part = rid.split("-", 1)[1]
        entries.append({"id": rid, "gate": "SP", "requirement": text, "workstream": ws, "dependencies": list(deps),
                        "criteria": criteria, "source": {"spec": f"MASTER_SPEC.md Part {part}", "spec_text": text,
                                                         "gate_title": "Mandatory requirements outside Part 17",
                                                         "clarification": None}})
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate requirement ids")
    return entries


def load(root: str | None = None) -> dict:
    with open(manifest_path(root)) as fh:
        return json.load(fh)


def save(m: dict, root: str | None = None) -> None:
    path = manifest_path(root)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def build(root: str | None = None, decision: str | None = None) -> dict:
    """Create or refresh the manifest. A criteria change needs `decision` = a D-<n> id present in DECISION_LOG.md."""
    fresh = derive(root)
    try:
        old = {e["id"]: e for e in load(root)["requirements"]}
    except (OSError, ValueError, KeyError):
        old = {}
    out = []
    for e in fresh:
        prev = old.get(e["id"], {})
        history = list(prev.get("criteria_history", []))
        if not history:
            history = [{"criteria": e["criteria"], "decision": None, "note": "initial criteria"}]
        elif history[-1]["criteria"] != e["criteria"]:
            if not decision or decision not in decisions(root):
                raise ValueError(f"{e['id']}: criteria changed; cite a decision recorded in DECISION_LOG.md")
            history.append({"criteria": e["criteria"], "decision": decision, "note": "changed under change control"})
        out.append({
            "id": e["id"], "gate": e["gate"], "requirement": e["requirement"], "workstream": e["workstream"],
            "dependencies": e["dependencies"],
            "implementation_files": prev.get("implementation_files", []),
            "tests": prev.get("tests", _tests_for(e["gate"] if e["gate"] != "SP" else e["id"])),
            "verify_command": prev.get("verify_command", _verify_command(e["gate"], e["id"])),
            "criteria": e["criteria"], "criteria_history": history,
            "status": prev.get("status", "NOT_STARTED"),
            "evidence": prev.get("evidence", []), "review": prev.get("review"), "commit": prev.get("commit"),
            "blocker": prev.get("blocker"), "source": e["source"]})
    m = {"schema": "crucible.v5.manifest/1",
         "statuses": list(STATUSES),
         "rules": "A requirement is VERIFIED only when the checker (python -m crucible.acceptance.v5 --verify-all) "
                  "confirms: implementation files exist; the evidence was produced from a clean, committed tree that "
                  "includes the latest change to those files; every criterion is met by the evidence's own metrics; "
                  "the listed tests passed in a recorded JUnit file; and a reviewer other than the evidence's author "
                  "approved this exact evidence.",
         "requirements": out}
    save(m, root)
    return m


def decisions(root: str | None = None) -> set[str]:
    try:
        text = open(os.path.join(docs_dir(root), "DECISION_LOG.md")).read()
    except OSError:
        return set()
    return set(re.findall(r"^##+\s*(D-\d+)\b", text, re.M))
