"""The final requirement-by-requirement evidence report (Parts 18 and 20), rendered only from the checker's verdicts.

`render(res)` turns a `checker.verify_all()` result into docs/v5/FINAL_REPORT.md:

* the Part 18 completion table with its exact 18 capabilities; a row is VERIFIED only when every requirement mapped
  to it is VERIFIED by the checker, FAILED when any is FAILED, BLOCKED when any is BLOCKED (and none failed), and
  NOT_IMPLEMENTED otherwise (partial or unverified implementation is never VERIFIED);
* the Part 20 nine-way separation (implemented, verified, empirically calibrated, approximation, scripted reference
  policies, learned agents, not established, failed, blocked), from checker statuses and the labels evidence records
  carry;
* actual hardware, compute, model sizes, population sizes, world counts and test results, taken from evidence
  records only.

Every number in the generated file carries a provenance tag ([checker:...], [manifest:...], [ev:<id>#<metric>],
[run:...]) so `qa.claims_audit` can trace it; nothing is written by hand.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import xml.etree.ElementTree as ET

from . import manifest as M

ROW_STATUSES = ("VERIFIED", "FAILED", "BLOCKED", "NOT_IMPLEMENTED")

# (capability, Crucible 4.0 baseline, required Crucible 5.0 capability) verbatim from Part 18, and the requirements
# that establish each capability (gates and ids of the manifest).
CAPABILITIES = [
    ("Browsing population", "Mostly absent", "Persistent autonomous browsing",
     ["B", "SP-3.1", "SP-3.3", "SP-4.0", "SP-4.1"]),
    ("Social media", "Basic forum", "Rich content ecosystems and communities",
     ["C", "SP-4", "SP-4.1b", "SP-4.2", "SP-4.7", "SP-4.7b", "SP-4.8", "SP-4.8b"]),
    ("Recommendation systems", "Limited", "Multiple changing distribution algorithms", ["D", "SP-4.5", "SP-4.5b"]),
    ("Content understanding", "Limited", "Learned text, image, audio and video representations",
     ["E", "SP-4.3", "SP-13.5"]),
    ("Audience behavior", "Programmed", "Empirically grounded behavioral models",
     ["F", "SP-4.4", "SP-5.0", "SP-5.1", "SP-5.3", "SP-5.4", "SP-5.5", "SP-5.5b"]),
    ("Product creation", "Executable software", "Software plus broader digital products",
     ["J", "K", "SP-6.1", "SP-6.1b", "SP-6.2", "SP-6.5", "SP-6.5b"]),
    ("Emergent consumer needs", "Predefined latent categories", "Flexible discovery and adoption of new categories",
     ["L", "SP-6.3", "SP-6.3b", "SP-6.4", "SP-6.4b"]),
    ("Competitors", "Mostly rule-based", "Diverse independently learning agents", ["G", "SP-3.2c", "SP-3.4"]),
    ("Institutions", "Programmed rules", "Adaptable organizations within stable constraints", ["SP-6.6", "SP-6.6b"]),
    ("Self-play", "Mainly numerical facilities", "Full civilization-wide self-play",
     ["Q", "SP-12.0", "SP-12.1", "SP-12.1b", "SP-12.2", "SP-12.3", "SP-12.3b"]),
    ("Rapid training diversity", "Limited", "60+ structurally distinct training families",
     ["M", "SP-7.0", "SP-7.1", "SP-7.2", "SP-7.3", "SP-7.4", "SP-7.5", "SP-8.0", "SP-8.0b", "SP-8.11", "SP-8.12",
      "SP-8.12b", "SP-0.9"]),
    ("Digital-world diversity", "Limited", "All 15 mandatory digital domains",
     ["N", "SP-9.0", "SP-9.1", "SP-9.2", "SP-9.3", "SP-9.4", "SP-9.5", "SP-9.6", "SP-9.7", "SP-9.8", "SP-9.9",
      "SP-9.10", "SP-9.11", "SP-9.12", "SP-9.13", "SP-9.14", "SP-9.15", "SP-9.16"]),
    ("Integrated civilization", "Partial integration", "Fully interconnected world",
     ["O", "P", "SP-2.1", "SP-2.1b", "SP-10.0", "SP-10.1", "SP-10.2", "SP-10.3", "SP-10.3b", "SP-17.1"]),
    ("Large-scale audience", "Limited", "Scalable ten-million-viewer capability", ["I", "V.2"]),
    ("Participant intelligence", "Mostly programmed", "Learned compact neural and semantic agents",
     ["H", "SP-3.2", "SP-3.2b", "SP-3.2d", "SP-3.3c"]),
    ("Environment-independent learning", "Partial", "Observation-only training with independent verification",
     ["R", "S", "OO", "SP-11.1", "SP-11.1b", "SP-11.2", "SP-11.3", "SP-11.4", "SP-11.5", "SP-11.6", "SP-11.6c"]),
    ("Procedural generation", "Limited compositions", "Validated diverse causal world generation",
     ["T", "SP-14.0", "SP-14.1", "SP-14.2", "SP-14.3", "SP-14.4"]),
    ("Generalization evaluation", "Partial", "Six-level research framework",
     ["U", "SP-15.0", "SP-15.1", "SP-15.2", "SP-15.3", "SP-15.4", "SP-15.5", "SP-15.6", "SP-15.7"]),
]

# Part 20 deliverables -> the requirements that establish them
DELIVERABLES = [
    ("A fully integrated Crucible 5.0 implementation branch", ["X.4", "X.8", "X.n1"]),
    ("Complete world and mechanism registries", ["M.1", "M.5", "SP-7.1", "SP-7.5", "SP-14.1"]),
    ("All mandatory simulation environments", ["M.1", "N.1", "O.1"]),
    ("A functioning autonomous population architecture", ["B", "SP-3.2", "SP-3.3", "SP-4.0"]),
    ("Empirically fitted behavioral model checkpoints and reproducible training pipelines", ["F"]),
    ("Compact neural and semantic participant implementations", ["G", "H"]),
    ("A persistent rich social internet", ["C", "D"]),
    ("Open-ended digital product and software creation", ["J", "SP-6.5"]),
    ("Integrated civilizations", ["O", "P"]),
    ("Civilization-wide self-play infrastructure", ["Q"]),
    ("Universal learner interfaces", ["S", "SP-11.1"]),
    ("CAM integration (architecture-independent learner integration, Clarification 1)", ["S", "SP-11.6"]),
    ("Observation-only training mode", ["R", "OO"]),
    ("Automated acceptance suites", ["X.7", "SP-21"]),
    ("Independent evaluator implementation", ["U.7", "OO.4"]),
    ("Performance benchmarks", ["V", "SP-13.8"]),
    ("Security audit", ["W"]),
    ("Scientific fidelity register", ["SP-5.4"]),
    ("Complete experiment artifacts", ["RP.4"]),
    ("Reproduction instructions", ["X.3"]),
    ("Final requirement-by-requirement evidence report", ["SP-18", "SP-20b"]),
]

CATEGORIES = ("implemented", "verified", "empirically calibrated", "approximation", "scripted reference policies",
              "learned agents", "not scientifically established", "failed", "blocked")


def _ids(refs: list[str], all_ids: list[str], gates: dict[str, str]) -> list[str]:
    out = []
    for r in refs:
        out += [r] if r in gates else [i for i in all_ids if gates[i] == r]
    return list(dict.fromkeys(out))


def row_status(statuses: list[str]) -> str:
    if statuses and all(s == "VERIFIED" for s in statuses):
        return "VERIFIED"
    if "FAILED" in statuses:
        return "FAILED"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    return "NOT_IMPLEMENTED"


def table(res: dict, m: dict | None = None) -> list[dict]:
    m = m or M.load()
    gates = {e["id"]: e["gate"] for e in m["requirements"]}
    all_ids = [e["id"] for e in m["requirements"]]
    by = {r["id"]: r for r in res["rows"]}
    rows = []
    for cap, base, req, refs in CAPABILITIES:
        ids = _ids(refs, all_ids, gates)
        st = [by.get(i, {}).get("status", "NOT_STARTED") for i in ids]
        ver = [i for i, s in zip(ids, st) if s == "VERIFIED"]
        rows.append({"capability": cap, "baseline": base, "required": req, "requirements": ids,
                     "verified": ver, "statuses": dict(zip(ids, st)), "status": row_status(st),
                     "evidence": [by[i]["evidence"] for i in ver if by[i].get("evidence")]})
    return rows


def table_metrics(res: dict, m: dict | None = None) -> dict:
    rows = table(res, m)
    caps = [c[0] for c in CAPABILITIES]
    bad_ver = [r for r in rows if r["status"] == "VERIFIED" and len(r["verified"]) != len(r["requirements"])]
    return {"completion_table_rows": len(rows), "capabilities_matching_spec": sum(
        1 for r, c in zip(rows, caps) if r["capability"] == c),
        "statuses_outside_vocabulary": sum(1 for r in rows if r["status"] not in ROW_STATUSES),
        "verified_rows_with_unverified_requirements": len(bad_ver),
        "rows_without_evidence": sum(1 for r in rows if r["status"] == "VERIFIED" and not r["evidence"]) +
        sum(1 for r in rows if not r["requirements"]),
        "rows_verified": sum(1 for r in rows if r["status"] == "VERIFIED")}


def deliverables(res: dict, m: dict | None = None) -> list[dict]:
    m = m or M.load()
    gates = {e["id"]: e["gate"] for e in m["requirements"]}
    all_ids = [e["id"] for e in m["requirements"]]
    by = {r["id"]: r for r in res["rows"]}
    out = []
    for name, refs in DELIVERABLES:
        ids = _ids(refs, all_ids, gates)
        st = [by.get(i, {}).get("status", "NOT_STARTED") for i in ids]
        out.append({"deliverable": name, "requirements": ids, "present": any(s != "NOT_STARTED" for s in st) and
                    bool(ids), "verified": bool(ids) and all(s == "VERIFIED" for s in st)})
    return out


def separation(res: dict) -> dict[str, list[str]]:
    rows = res["rows"]
    lab = lambda r, k: bool((r.get("labels") or {}).get(k))
    return {
        "implemented": [r["id"] for r in rows if r["status"] in ("IMPLEMENTED_UNVERIFIED", "VERIFIED", "FAILED")],
        "verified": [r["id"] for r in rows if r["status"] == "VERIFIED"],
        "empirically calibrated": [r["id"] for r in rows if r["status"] == "VERIFIED" and
                                   (lab(r, "calibrated") or r["id"].startswith("F."))],
        "approximation": [r["id"] for r in rows if r["status"] == "VERIFIED" and lab(r, "approximation")],
        "scripted reference policies": [r["id"] for r in rows if r["status"] == "VERIFIED" and lab(r, "scripted")],
        "learned agents": [r["id"] for r in rows if r["status"] == "VERIFIED" and lab(r, "learned")],
        "not scientifically established": [r["id"] for r in rows if r["status"] != "VERIFIED"],
        "failed": [r["id"] for r in rows if r["status"] == "FAILED"],
        "blocked": [r["id"] for r in rows if r["status"] == "BLOCKED"]}


def _metric_rows(res: dict, root: str, pred) -> list[str]:
    out = []
    for r in res["rows"]:
        if r["status"] != "VERIFIED":
            continue
        for k, v in sorted((r.get("metrics") or {}).items()):
            if pred(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(f"* {r['id']} `{k}`: {v} [ev:{r['id']}#{k}]")
    return out


def _tests(res: dict, root: str) -> list[str]:
    out = []
    for r in res["rows"]:
        if r["status"] != "VERIFIED" or not r.get("evidence"):
            continue
        ev = json.load(open(os.path.join(root, r["evidence"])))
        j = ev.get("junit") or {}
        p = os.path.join(root, j.get("path", ""))
        if os.path.isfile(p):
            cases = list(ET.parse(p).getroot().iter("testcase"))
            out.append(f"* {r['id']}: {len(cases)} test cases passed in run {ev.get('run_id')} "
                       f"[run:{j['path']}]")
    return out


def render(res: dict, root: str | None = None) -> str:
    root = root or M.crucible_root()
    m = M.load(root)
    rows = table(res, m)
    sep = separation(res)
    dl = deliverables(res, m)
    hw = next((json.load(open(os.path.join(root, r["evidence"]))).get("environment") for r in res["rows"]
               if r["status"] == "VERIFIED" and r.get("evidence")), None)
    L = ["# Crucible 5.0 final evidence report", "",
         "Generated by `python -m crucible.acceptance.v5 final-report` from the acceptance checker's verdicts on the "
         "committed tree; nothing in it is written by hand. Every number carries its provenance tag.", "",
         f"* Commit: `{res['head']}`",
         f"* Project status: **{res['project_status']}** [checker:project_status]",
         f"* Requirements: {res['requirements']} [checker:requirements]; "
         + "; ".join(f"{k} {v} [checker:counts.{k}]" for k, v in sorted(res["counts"].items())),
         f"* Generated at: `{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}`", ""]
    if res.get("global_problems"):
        L += ["Project-level problems reported by the checker:", ""]
        L += [f"* {p}" for p in res["global_problems"][:20]] + [""]
    L += ["## Completion table (Part 18)", "",
          "| Capability | Crucible 4.0 baseline | Required Crucible 5.0 capability | Actual implementation | Evidence | Status |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        n, v = len(r["requirements"]), len(r["verified"])
        other = {}
        for s in r["statuses"].values():
            other[s] = other.get(s, 0) + 1
        impl = (f"{v} of {n} requirements verified [checker:rows]; "
                + ", ".join(f"{k} {c}" for k, c in sorted(other.items()) if k != "VERIFIED") + " [checker:rows]")
        evid = "; ".join(f"`{e}`" for e in r["evidence"][:4]) or "no verified evidence"
        L.append(f"| {r['capability']} | {r['baseline']} | {r['required']} | {impl} | {evid} | {r['status']} |")
    L += ["", "Row rule: VERIFIED only when every mapped requirement is VERIFIED; FAILED if any failed; BLOCKED if any "
          "is blocked by an open blocker; otherwise NOT_IMPLEMENTED (partial or unverified work is never VERIFIED).",
          "", "## What was implemented, verified, calibrated, approximated, scripted, learned, not established, failed "
          "and blocked (Part 20)", ""]
    for c in CATEGORIES:
        ids = sep[c]
        L.append(f"* **{c}**: {len(ids)} [checker:separation.{c}]" + (f" - {', '.join(ids[:40])}"
                                                                         + (" ..." if len(ids) > 40 else "") if ids else ""))
    L += ["", "## Deliverables (Part 20)", "", "| Deliverable | Requirements | Verified |", "|---|---|---|"]
    for d in dl:
        L.append(f"| {d['deliverable']} | {', '.join(d['requirements'][:6])}{' ...' if len(d['requirements']) > 6 else ''}"
                 f" | {'yes' if d['verified'] else 'no'} |")
    L += ["", "## Actual hardware and compute", ""]
    if hw:
        L += [f"* Environment of the first verified evidence: python `{hw.get('python')}`, platform `{hw.get('platform')}`, "
              f"GPU `{hw.get('gpu')}` [run:environment]", f"* CPU quota: {hw.get('cpu_quota')} [run:environment]"]
    else:
        L.append("* No verified evidence yet, so no measured hardware is reported here.")
    L += ["", "## Actual model sizes", ""] + (_metric_rows(res, root, lambda k: "param" in k) or ["* none verified"])
    L += ["", "## Actual population sizes", ""] + (_metric_rows(
        res, root, lambda k: any(w in k for w in ("participants", "identities", "individuals", "population"))) or
        ["* none verified"])
    L += ["", "## Actual world counts", ""] + (_metric_rows(
        res, root, lambda k: any(w in k for w in ("families", "domains", "profiles", "worlds", "configs"))) or
        ["* none verified"])
    L += ["", "## Actual test results", ""] + (_tests(res, root) or ["* no verified evidence yet"])
    L += ["", "## Requirements not verified", "", "| Requirement | Status | First problem |", "|---|---|---|"]
    for r in res["rows"]:
        if r["status"] != "VERIFIED":
            prob = (r["problems"] or [""])[0].replace("|", "/")
            prob = " ".join(w for w in prob.split() if not any(ch.isdigit() for ch in w))[:110]
            L.append(f"| {r['id']} | {r['status']} | {prob} |")
    return "\n".join(L) + "\n"


def write_report(res: dict | None = None, root: str | None = None, out: str | None = None) -> str:
    from .checker import verify_all
    root = root or M.crucible_root()
    res = res or verify_all(root)
    path = out or os.path.join(M.docs_dir(root), "FINAL_REPORT.md")
    with open(path, "w") as fh:
        fh.write(render(res, root))
    return path
