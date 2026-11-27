"""The acceptance manifest (`docs/v5/ACCEPTANCE_MANIFEST.json`): one entry per mandatory requirement.

`derive()` computes the entries from MASTER_SPEC.md (gate wording, anchored by its SHA-256), the clarifications and
`requirements.py` (criteria, tolerances, owners, dependencies) of the checkout being checked. `build()` records every
addition and change in the append-only ledger (`ledger.py`) and writes the manifest as a projection of the ledger plus
each entry's state fields (status, evidence, review, commit, blocker), which the checker never trusts.

Change control: a requirement is never removed; a criteria change needs `decision` = a D-<n> recorded in
DECISION_LOG.md whose text names the requirement, and must be provably at least as strict; removing a test file,
implementation file or dependency, or changing wording or tolerances, also needs such a decision. `build()` refuses
anything else, and the checker re-verifies all of it from the ledger and git history (`consistency`).
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import re

from . import ledger as L
from . import requirements as R

STATUSES = ("NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED_UNVERIFIED", "FAILED", "BLOCKED", "VERIFIED")
ENTRY_FIELDS = ("id", "gate", "requirement", "workstream", "dependencies", "implementation_files", "tests",
                "verify_command", "criteria", "criteria_history", "tolerance", "status", "evidence", "review",
                "commit", "source")
DERIVED_FIELDS = ("gate", "requirement", "workstream", "dependencies", "criteria", "tolerance", "source",
                  "verify_command")
STATE_FIELDS = ("status", "evidence", "review", "commit", "blocker")
SPEC_REL = "docs/v5/MASTER_SPEC.md"


def crucible_root() -> str:
    """The `crucible/` directory of this checkout."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


def docs_dir(root: str | None = None) -> str:
    return os.path.join(root or crucible_root(), "docs", "v5")


def manifest_path(root: str | None = None) -> str:
    return os.path.join(docs_dir(root), "ACCEPTANCE_MANIFEST.json")


_RQ_CACHE: dict = {}


def requirements_module(root: str | None = None):
    """The `requirements.py` of the checkout at `root` (this package's own when root is this checkout)."""
    if root is None or os.path.realpath(root) == os.path.realpath(crucible_root()):
        return R
    p = os.path.join(root, "src", "crucible", "acceptance", "v5", "requirements.py")
    key = (os.path.realpath(p), os.path.getmtime(p))
    if key not in _RQ_CACHE:
        name = "_crucible_v5_requirements_" + hashlib.sha256(key[0].encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RQ_CACHE[key] = mod
    return _RQ_CACHE[key]


def spec_sha256(root: str | None = None) -> str:
    with open(os.path.join(root or crucible_root(), SPEC_REL), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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


def default_test(rid: str) -> str:
    g = "SP" if rid.startswith("SP-") else rid.split(".", 1)[0]
    key = rid if g == "SP" else g
    return f"tests/v5/test_gate_{key.replace('-', '_').replace('.', '_')}.py"


def canonical_test(rid: str, root: str | None = None) -> str:
    """The gate test file that must produce a requirement's evidence: tests/v5/test_gate_<G>.py (SP items: their id),
    unless requirements.GATE_TEST_OVERRIDE names another gate file for it."""
    return getattr(requirements_module(root), "GATE_TEST_OVERRIDE", {}).get(rid, default_test(rid))


CLAR_TITLES = {1: "Architecture independence from CAM", 2: "Observation-only as the universal default",
               3: "Recovery, audit and persistence"}


def derive(root: str | None = None) -> list[dict]:
    RQ = requirements_module(root)
    tol = getattr(RQ, "TOLERANCES", {})
    entries = []
    gates = spec_gates(root) if (RQ.GATE_ITEMS or RQ.GATE_NOTES) else {}
    for letter, (ws, deps, crits) in RQ.GATE_ITEMS.items():
        bullets = gates[letter]["bullets"]
        if len(bullets) != len(crits):
            raise ValueError(f"gate {letter}: spec has {len(bullets)} items, requirements.py has {len(crits)}")
        for i, (text, criteria) in enumerate(zip(bullets, crits), 1):
            rid = f"{letter}.{i}"
            clarified = rid in RQ.CLARIFIED
            entries.append({
                "id": rid, "gate": letter, "requirement": RQ.CLARIFIED.get(rid, text), "workstream": ws,
                "dependencies": list(deps), "criteria": criteria,
                "source": {"spec": f"MASTER_SPEC.md Part 17, Gate {letter} item {i}", "spec_text": text,
                           "gate_title": RQ.GATE_TITLE_OVERRIDE.get(letter, gates[letter]["title"]),
                           "clarification": "MASTER_SPEC_CLARIFICATIONS.md Clarification 1" if clarified else None}})
    for rid, gate, text, ws, criteria in RQ.GATE_NOTES:
        entries.append({"id": rid, "gate": gate, "requirement": text, "workstream": ws, "dependencies": [gate],
                        "criteria": criteria, "source": {"spec": f"MASTER_SPEC.md Part 17, Gate {gate} (mandatory note)",
                                                         "spec_text": text, "gate_title": gates[gate]["title"],
                                                         "clarification": None}})
    for item in RQ.SPEC_ITEMS:
        rid, text, ws, deps, criteria = item[:5]
        extra = item[5] if len(item) > 5 else {}
        part = rid.split("-", 1)[1]
        entries.append({"id": rid, "gate": "SP", "requirement": text, "workstream": ws, "dependencies": list(deps),
                        "criteria": criteria,
                        "source": {"spec": extra.get("spec", f"MASTER_SPEC.md Part {part}"), "spec_text": text,
                                   "gate_title": "Mandatory requirements outside Part 17",
                                   "clarification": extra.get("clarification"),
                                   **({"audit": extra["audit"]} if extra.get("audit") else {})}})
    for item in getattr(RQ, "AMENDMENT_ITEMS", []):
        rid, gate, text, ws, deps, criteria, clar = item[:7]
        extra = item[7] if len(item) > 7 else {}
        entries.append({"id": rid, "gate": gate, "requirement": text, "workstream": ws, "dependencies": list(deps),
                        "criteria": criteria,
                        "source": {"spec": f"MASTER_SPEC_CLARIFICATIONS.md Clarification {clar}", "spec_text": text,
                                   "gate_title": CLAR_TITLES.get(clar, f"Clarification {clar}"),
                                   "clarification": f"MASTER_SPEC_CLARIFICATIONS.md Clarification {clar}",
                                   **({"audit": extra["audit"]} if extra.get("audit") else {})}})
    for e in entries:
        e["tolerance"] = tol.get(e["id"], {})
        e["verify_command"] = _verify_command(e["gate"], e["id"])
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate requirement ids: {dup}")
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


def decisions(root: str | None = None) -> set[str]:
    return set(L.decision_sections(docs_dir(root)))


RULES = ("A requirement is VERIFIED only when the checker (python -m crucible.acceptance.v5 --verify-all) confirms, "
         "against the committed tree and never against status strings: the manifest equals its derivation from the "
         "spec and requirements.py and the append-only change-control ledger; the implementation files exist; an "
         "official run in a fresh clone of a committed tree produced the evidence, ran every collected test of the "
         "requirement's test files (all passed, none skipped) and recorded the imported modules, none of which changed "
         "since; the producing test asserts on measured values and records non-constant metrics; every criterion holds "
         "on the evidence's metrics; every dependency is VERIFIED; and an independent reviewer (a different agent and "
         "session) re-executed the tests in another fresh clone, reproduced the metrics within the declared tolerance "
         "and committed a review bound to the requirement text, criteria version, evidence hash and commit.")


def _project(fresh: list[dict], st: dict, old: dict) -> dict:
    out = []
    for e in fresh:
        r, prev = st[e["id"]], old.get(e["id"], {})
        out.append({
            "id": e["id"], "gate": e["gate"], "requirement": e["requirement"], "workstream": e["workstream"],
            "dependencies": e["dependencies"],
            "implementation_files": list(r["implementation_files"] or []), "tests": list(r["tests"] or []),
            "verify_command": e["verify_command"], "criteria": e["criteria"],
            "criteria_history": [{"criteria": h["criteria"], "decision": h["decision"], "note": h["note"],
                                  "ledger_hash": h["hash"]} for h in r["criteria_history"]],
            "tolerance": e["tolerance"],
            "status": prev.get("status", "NOT_STARTED") if prev.get("status") in STATUSES else "NOT_STARTED",
            "evidence": prev.get("evidence", []), "review": prev.get("review"), "commit": prev.get("commit"),
            "blocker": prev.get("blocker"), "source": e["source"]})
    return {"schema": "crucible.v5.manifest/2", "statuses": list(STATUSES), "rules": RULES,
            "ledger": L.LEDGER_REL, "requirements": out}


def build(root: str | None = None, decision: str | None = None) -> dict:
    """Create or refresh the manifest, recording every addition and change in the ledger first."""
    fresh = derive(root)
    sections = L.decision_sections(docs_dir(root))
    try:
        old = {e["id"]: e for e in load(root)["requirements"]}
    except (OSError, ValueError, KeyError):
        old = {}
    root_ = root or crucible_root()
    entries = L.read(root_)
    probs = L.chain_problems(entries)
    if probs:
        raise ValueError(f"ledger is corrupt: {probs[:3]}")
    st = L.state(entries)
    items = []
    if "MASTER_SPEC.md" not in st["__anchors__"] and os.path.exists(os.path.join(root_, SPEC_REL)):
        RQ = requirements_module(root)
        want = getattr(RQ, "MASTER_SPEC_SHA256", None)
        have = spec_sha256(root)
        if want and want != have:
            raise ValueError("MASTER_SPEC.md differs from the verbatim anchor in requirements.py")
        items.append({"id": "MASTER_SPEC.md", "kind": "anchor", "value": have,
                      "note": "sha256 of the verbatim specification"})
    fresh_ids = {e["id"] for e in fresh}
    gone = sorted(i for i in st if not i.startswith("__") and i not in fresh_ids)
    if gone:
        raise ValueError(f"requirements may never be removed: {gone[:10]}")

    def need(rid: str, what: str) -> str:
        if not decision or decision not in sections:
            raise ValueError(f"{rid}: {what}; cite a decision recorded in DECISION_LOG.md")
        if not L.mentions(sections[decision], rid):
            raise ValueError(f"{rid}: {what}; decision {decision} does not name {rid}")
        return decision

    for e in fresh:
        rid, r, prev = e["id"], st.get(e["id"]), old.get(e["id"], {})
        if r is None:
            items.append({"id": rid, "kind": "add", "value": {"gate": e["gate"], "workstream": e["workstream"]},
                          "note": "requirement added"})
            items.append({"id": rid, "kind": "text", "value": e["requirement"]})
            hist = prev.get("criteria_history") or []
            if hist and hist[-1].get("criteria") == e["criteria"]:
                for k, h in enumerate(hist):          # carry the pre-ledger history of the manifest over verbatim
                    items.append({"id": rid, "kind": "criteria", "value": h["criteria"],
                                  "decision": h.get("decision") if k else None,
                                  "note": h.get("note") or ("initial criteria" if k == 0 else "")})
            else:
                items.append({"id": rid, "kind": "criteria", "value": e["criteria"], "note": "initial criteria"})
            items.append({"id": rid, "kind": "dependencies", "value": e["dependencies"]})
            items.append({"id": rid, "kind": "tolerance", "value": e["tolerance"]})
            tests = list(prev.get("tests") or [])
            if canonical_test(rid, root) not in tests:
                tests.insert(0, canonical_test(rid, root))
            items.append({"id": rid, "kind": "tests", "value": tests})
            items.append({"id": rid, "kind": "implementation_files", "value": list(prev.get("implementation_files") or [])})
            continue
        if r["text"] != e["requirement"]:
            items.append({"id": rid, "kind": "text", "value": e["requirement"],
                          "decision": need(rid, "requirement wording changed")})
        cur = r["criteria_history"][-1]["criteria"] if r["criteria_history"] else None
        if cur != e["criteria"]:
            weaker = L.not_implied(e["criteria"], cur or [])
            if weaker:
                raise ValueError(f"{rid}: new criteria are not provably at least as strict (unimplied: {weaker[:3]})")
            items.append({"id": rid, "kind": "criteria", "value": e["criteria"],
                          "decision": need(rid, "criteria changed"), "note": "changed under change control"})
        if r["dependencies"] != e["dependencies"]:
            removed = set(r["dependencies"] or []) - set(e["dependencies"])
            items.append({"id": rid, "kind": "dependencies", "value": e["dependencies"],
                          "decision": need(rid, "dependencies removed") if removed else decision})
        if r["tolerance"] != e["tolerance"]:
            items.append({"id": rid, "kind": "tolerance", "value": e["tolerance"],
                          "decision": need(rid, "tolerance changed")})
        canon, cur_tests = canonical_test(rid, root), list(r["tests"] or [])
        if canon not in cur_tests:
            dflt = default_test(rid)
            if canon != dflt and dflt in cur_tests and decision and decision in sections and \
                    L.mentions(sections[decision], rid):            # the canonical gate file was reassigned
                items.append({"id": rid, "kind": "tests", "value": [canon] + [t for t in cur_tests if t != dflt],
                              "decision": decision, "note": "canonical gate test reassigned"})
            else:
                items.append({"id": rid, "kind": "tests", "value": [canon] + cur_tests,
                              "note": "canonical gate test restored"})
    L.append(root_, items)
    st = L.state(L.read(root_))
    m = _project(fresh, st, old)
    save(m, root)
    return m


def add_files(root: str | None, rid: str, *, tests: list[str] = (), implementation_files: list[str] = (),
              note: str = "") -> list[dict]:
    """Append test/implementation files to a requirement's change-controlled lists (additions need no decision)."""
    root_ = root or crucible_root()
    st = L.state(L.read(root_))
    if rid not in st:
        raise KeyError(rid)
    items = []
    for kind, new in (("tests", tests), ("implementation_files", implementation_files)):
        cur = list(st[rid][kind] or [])
        add = [p for p in new if p not in cur]
        if add:
            items.append({"id": rid, "kind": kind, "value": cur + add, "note": note})
    return L.append(root_, items)


def sync(root: str | None = None) -> dict:
    """Rewrite the manifest from the ledger (keeping each entry's state fields)."""
    root_ = root or crucible_root()
    try:
        old = {e["id"]: e for e in load(root)["requirements"]}
    except (OSError, ValueError, KeyError):
        old = {}
    m = _project(derive(root), L.state(L.read(root_)), old)
    save(m, root)
    return m


def _clean_path(p: str) -> str:
    p = p.strip().split(" (", 1)[0].split(" ", 1)[0].strip()
    if p.startswith("crucible/"):
        p = p[len("crucible/"):]
    return p.split("::", 1)[0]


def workstream_files(root: str | None = None) -> dict[str, dict]:
    """rid -> {"tests": [...], "implementation_files": [...], "sources": [...], "skipped": [(path, why)]} from every
    docs/v5/workstreams/*.json report (both the list form and the {"requirements": {id: {...}}} form)."""
    root_ = root or crucible_root()
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(docs_dir(root), "workstreams", "*.json"))):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        reqs = d.get("requirements", [])
        items = reqs.items() if isinstance(reqs, dict) else [(r.get("id"), r) for r in reqs if isinstance(r, dict)]
        for rid, info in items:
            if not rid or not isinstance(info, dict):
                continue
            o = out.setdefault(rid, {"tests": [], "implementation_files": [], "sources": [], "skipped": []})
            o["sources"].append(os.path.basename(f))
            for kind in ("tests", "implementation_files"):
                paths = []
                for p in info.get(kind, []) or []:
                    if not isinstance(p, str):
                        continue
                    c = _clean_path(p)
                    if c and any(ch in c for ch in "*?[") and not os.path.isabs(c) and ".." not in c.split("/"):
                        hits = sorted(os.path.relpath(h, root_) for h in glob.glob(os.path.join(root_, c), recursive=True)
                                      if os.path.isfile(h))       # a pattern names every file it matches
                        if hits:
                            paths += [(p, h) for h in hits]
                            continue
                    paths.append((p, c))
                for p, c in paths:
                    if not c or os.path.isabs(c) or ".." in c.split("/"):
                        o["skipped"].append((p, "not a path relative to crucible/"))
                    elif not os.path.exists(os.path.join(root_, c)):
                        o["skipped"].append((p, "does not exist in this checkout"))
                    elif kind == "tests" and not (c.endswith(".py") and os.path.basename(c).startswith("test")):
                        o["skipped"].append((p, "not a test file"))
                    elif c not in o[kind]:
                        o[kind].append(c)
    return out


def merge_workstreams(root: str | None = None) -> dict:
    """Merge implementation files and tests from the workstream reports into the ledger and the manifest.
    Idempotent: rerun it after each merge of a workstream branch. Returns what was added and what was skipped."""
    root_ = root or crucible_root()
    st = L.state(L.read(root_))
    ws = workstream_files(root)
    added, skipped, unknown = {}, {}, []
    for rid, info in sorted(ws.items()):
        if rid not in st:
            unknown.append(rid)
            continue
        new = add_files(root_, rid, tests=info["tests"], implementation_files=info["implementation_files"],
                        note="merged from " + ", ".join(sorted(set(info["sources"]))))
        if new:
            added[rid] = [x for e in new for x in e["value"]]
        if info["skipped"]:
            skipped[rid] = info["skipped"]
    sync(root)
    return {"added": added, "skipped": skipped, "unknown_ids": unknown}


# ------------------------------------------------------------------------------------------------- history
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_HIST_CACHE: dict = {}


def history_problems(root: str, fresh: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Every committed version of the manifest in HEAD's history (including versions that predate the ledger): no
    requirement it lists may be missing from derive(), and today's criteria must imply each criteria version it
    records (current criteria and criteria_history)."""
    import subprocess
    rel = "docs/v5/ACCEPTANCE_MANIFEST.json"
    run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
    commits = run("rev-list", "--full-history", "HEAD", "--", rel).stdout.split()
    want = {e["id"]: e for e in fresh}
    g: list[str] = []
    per: dict[str, list[str]] = {}
    removed: dict[str, str] = {}
    seen = set()
    for c in commits:
        blob = run("rev-parse", "--verify", "-q", f"{c}:./{rel}").stdout.strip()
        if not blob or blob in seen:
            continue
        seen.add(blob)
        if blob not in _HIST_CACHE:
            try:
                _HIST_CACHE[blob] = json.loads(run("cat-file", "blob", blob).stdout)
            except ValueError:
                _HIST_CACHE[blob] = None
        m = _HIST_CACHE[blob]
        if not isinstance(m, dict):
            continue
        for e in m.get("requirements") or []:
            rid = e.get("id") if isinstance(e, dict) else None
            if not rid:
                continue
            if rid not in want:
                removed.setdefault(rid, c[:9])
                continue
            versions = [e.get("criteria") or []] + [h.get("criteria") or [] for h in e.get("criteria_history") or []]
            for crit in versions:
                weaker = L.not_implied(want[rid]["criteria"], crit)
                if weaker:
                    msg = (f"criteria are weaker than those committed in manifest history at {c[:9]}: nothing implies "
                           + "; ".join(f"{x.get('metric')} {x.get('op')} {x.get('value')!r}" for x in weaker[:3]))
                    if msg not in per.get(rid, []):
                        per.setdefault(rid, []).append(msg)
    if removed:
        g.append(f"{len(removed)} requirements committed in manifest history are missing from derive() (removed): "
                 f"{sorted(removed)[:8]}")
    return g, per


# ------------------------------------------------------------------------------------------------- consistency
def consistency(root: str | None = None, m: dict | None = None) -> dict:
    """Everything that anchors the manifest: returns {"global": [...], "per_id": {id: [...]}, "derived": [...]}.
    Global problems make project completion impossible regardless of evidence."""
    root_ = root or crucible_root()
    g: list[str] = []
    per: dict[str, list[str]] = {}
    try:
        m = m if m is not None else load(root)
    except (OSError, ValueError) as exc:
        return {"global": [f"manifest unreadable: {exc}"], "per_id": {}, "derived": []}
    try:
        fresh = derive(root)
    except Exception as exc:                                   # noqa: BLE001 - reported, not swallowed
        return {"global": [f"manifest.derive() failed: {exc!r}"], "per_id": {}, "derived": []}
    reqs = m.get("requirements") or []
    if not reqs:
        g.append("the manifest has no requirements")
    if not fresh:
        g.append("manifest.derive() produced no requirements")
    have = {e.get("id"): e for e in reqs}
    want = {e["id"]: e for e in fresh}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    if missing:
        g.append(f"{len(missing)} requirements of derive() are missing from the manifest: {missing[:8]}")
    if extra:
        g.append(f"{len(extra)} manifest requirements are not in derive(): {extra[:8]}")
    if len(have) != len(reqs):
        g.append("the manifest contains duplicate ids")
    for rid, e in want.items():
        if rid not in have:
            per.setdefault(rid, []).append("missing from the manifest")
            continue
        for f in DERIVED_FIELDS:
            if have[rid].get(f) != e[f]:
                per.setdefault(rid, []).append(f"manifest field {f!r} differs from derive()")
    # the spec anchor
    RQ = requirements_module(root)
    try:
        sha = spec_sha256(root)
    except OSError:
        sha = None
    want_sha = getattr(RQ, "MASTER_SPEC_SHA256", None)
    entries = L.read(root_)
    st = L.state(entries)
    anch = st["__anchors__"].get("MASTER_SPEC.md")             # the first anchor: anchors never change
    for aid, vals in L.anchor_history(entries).items():
        if len({canonical_json(v) for v in vals}) > 1:
            g.append(f"the ledger re-anchors {aid} ({len(vals)} anchor entries): anchors are immutable")
    if RQ.GATE_ITEMS:
        if not want_sha or sha != want_sha:
            g.append("MASTER_SPEC.md does not match the verbatim anchor (requirements.MASTER_SPEC_SHA256)")
    if (RQ.GATE_ITEMS or anch is not None) and anch != sha:
        g.append("MASTER_SPEC.md does not match the ledger's anchor")
    # the ledger
    if not entries:
        g.append(f"no change-control ledger ({L.LEDGER_REL})")
    g += L.chain_problems(entries)
    g += L.git_problems(root_)
    rules = L.rule_problems(st, L.decision_sections(docs_dir(root)))
    for rid, ps in rules.items():
        per.setdefault(rid, []).extend(ps)
    led_ids = {i for i in st if not i.startswith("__")}
    lost = sorted(led_ids - set(want))
    if lost:
        g.append(f"{len(lost)} requirements recorded in the ledger were removed: {lost[:8]}")
    unrecorded = sorted(set(want) - led_ids)
    if unrecorded:
        g.append(f"{len(unrecorded)} requirements are not recorded in the ledger: {unrecorded[:8]}")
    hg, hper = history_problems(root_, fresh)
    g += hg
    for rid, ps in hper.items():
        per.setdefault(rid, []).extend(ps)
    for rid, e in want.items():
        r = st.get(rid)
        if r is None:
            per.setdefault(rid, []).append("not recorded in the change-control ledger")
            continue
        led_crit = r["criteria_history"][-1]["criteria"] if r["criteria_history"] else None
        if led_crit != e["criteria"]:
            per.setdefault(rid, []).append("criteria in requirements.py differ from the ledger (unrecorded change)")
        if r["text"] != e["requirement"]:
            per.setdefault(rid, []).append("wording differs from the ledger (unrecorded change)")
        if r["dependencies"] != e["dependencies"]:
            per.setdefault(rid, []).append("dependencies differ from the ledger (unrecorded change)")
        if r["tolerance"] != e["tolerance"]:
            per.setdefault(rid, []).append("tolerance differs from the ledger (unrecorded change)")
        mine = have.get(rid)
        if mine is None:
            continue
        hist = [{"criteria": h["criteria"], "decision": h["decision"], "note": h["note"], "ledger_hash": h["hash"]}
                for h in r["criteria_history"]]
        if mine.get("criteria_history") != hist:
            per.setdefault(rid, []).append("criteria_history differs from the ledger")
        for f in ("tests", "implementation_files"):
            if list(mine.get(f) or []) != list(r[f] or []):
                per.setdefault(rid, []).append(f"{f} differ from the ledger (lists are change-controlled)")
        if canonical_test(rid, root) not in (mine.get("tests") or []):
            per.setdefault(rid, []).append(f"the canonical gate test {canonical_test(rid, root)} is not listed")
    return {"global": g, "per_id": per, "derived": fresh}
