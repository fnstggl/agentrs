"""Change control for the acceptance manifest: an append-only, hash-chained ledger (`docs/v5/ACCEPTANCE_LEDGER.jsonl`).

Every requirement's existence, wording, criteria, reproduction tolerances, dependencies, test files and implementation
files are recorded as ledger entries. Each line is one JSON entry

    {"seq", "id", "kind", "value", "decision", "note", "prev", "hash"}

where `hash = sha256(canonical(entry without hash))` and `prev` is the previous line's hash, so no entry can be changed
or removed without breaking the chain. The chain is anchored in git: the checker requires that every committed version
of the file extends each of its parents' versions (append-only across history, merges included), that the working
copy equals the committed version, and that a pushed copy on the integration branch is a prefix of it.

The rules the checker enforces on the ledger's content:

* a requirement, once added, is never removed (the manifest and `manifest.derive()` must still contain it);
* a criteria change cites a decision recorded in DECISION_LOG.md whose text names the requirement, and is provably at
  least as strict as the version it replaces (every old criterion is implied by a new one: `ledger.not_implied`);
* removing a test file, an implementation file or a dependency, changing a tolerance or the requirement's wording
  cites such a decision; additions need none.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

LEDGER_REL = "docs/v5/ACCEPTANCE_LEDGER.jsonl"
GENESIS = "0" * 64
KINDS = ("anchor", "add", "text", "criteria", "tolerance", "dependencies", "tests", "implementation_files",
         "startup_hook")
HOOK_SHA_PREFIX = 16            # a startup_hook pin's decision must quote this many leading hex digits of its sha256
INTEGRATION_REFS = ("refs/remotes/origin/feat/crucible-5-complete",)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_hash(e: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in e.items() if k != "hash"}).encode()).hexdigest()


def path(root: str) -> str:
    return os.path.join(root, LEDGER_REL)


def parse(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read(root: str) -> list[dict]:
    try:
        return parse(open(path(root), encoding="utf-8").read())
    except OSError:
        return []


def append(root: str, items: list[dict]) -> list[dict]:
    """Append entries ({id, kind, value, decision?, note?}) to the ledger, chaining them; returns the new entries."""
    old = read(root)
    prev = old[-1]["hash"] if old else GENESIS
    seq = old[-1]["seq"] + 1 if old else 0
    out = []
    for it in items:
        if it["kind"] not in KINDS:
            raise ValueError(f"unknown ledger kind {it['kind']!r}")
        e = {"seq": seq, "id": it["id"], "kind": it["kind"], "value": it["value"], "decision": it.get("decision"),
             "note": it.get("note", ""), "prev": prev}
        e["hash"] = entry_hash(e)
        out.append(e)
        prev, seq = e["hash"], seq + 1
    if out:
        os.makedirs(os.path.dirname(path(root)), exist_ok=True)
        with open(path(root), "a", encoding="utf-8") as fh:
            for e in out:
                fh.write(canonical(e) + "\n")
    return out


def chain_problems(entries: list[dict]) -> list[str]:
    probs, prev = [], GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            probs.append(f"ledger entry {i} has seq {e.get('seq')}")
        if e.get("prev") != prev:
            probs.append(f"ledger entry {i} ({e.get('id')}) does not chain to its predecessor")
        if e.get("hash") != entry_hash(e):
            probs.append(f"ledger entry {i} ({e.get('id')}) hash does not match its content")
        if e.get("kind") not in KINDS:
            probs.append(f"ledger entry {i} has unknown kind {e.get('kind')!r}")
        prev = e.get("hash")
    return probs


def state(entries: list[dict]) -> dict:
    """id -> {"text", "criteria_history": [{criteria, decision, note, hash}], "tests", "implementation_files",
    "dependencies", "tolerance", "changes": [(kind, old, new, decision)]} plus the special keys "__anchors__" and
    "__hooks__" (start-up hook path -> its pins in ledger order: [{sha256, decision, seq}])."""
    st: dict = {"__anchors__": {}, "__hooks__": {}}
    for e in entries:
        rid, kind, val = e["id"], e["kind"], e["value"]
        if kind == "anchor":                     # anchors are immutable: the first one counts
            st["__anchors__"].setdefault(rid, val)
            continue
        if kind == "startup_hook":               # not a requirement: a pinned start-up hook file (D-83 third addendum)
            st["__hooks__"].setdefault(rid, []).append({
                "sha256": val.get("sha256") if isinstance(val, dict) else None, "decision": e.get("decision"),
                "seq": e["seq"]})
            continue
        if kind == "add":
            st.setdefault(rid, {"added": e["seq"], "text": None, "criteria_history": [], "tests": None,
                                "implementation_files": None, "dependencies": None, "tolerance": None, "changes": [],
                                "duplicate_add": False})
            if st[rid]["added"] != e["seq"]:
                st[rid]["duplicate_add"] = True
            continue
        r = st.setdefault(rid, {"added": None, "text": None, "criteria_history": [], "tests": None,
                                "implementation_files": None, "dependencies": None, "tolerance": None,
                                "changes": [], "duplicate_add": False})
        if kind == "criteria":
            old = r["criteria_history"][-1]["criteria"] if r["criteria_history"] else None
            r["criteria_history"].append({"criteria": val, "decision": e["decision"], "note": e["note"],
                                          "hash": e["hash"]})
            if old is not None:
                r["changes"].append(("criteria", old, val, e["decision"]))
        else:
            old = r[kind]
            r[kind] = val
            if old is not None:
                r["changes"].append((kind, old, val, e["decision"]))
    return st


def anchor_history(entries: list[dict]) -> dict[str, list]:
    """Every value each anchor id was given, in ledger order (more than one distinct value is a re-anchoring)."""
    out: dict[str, list] = {}
    for e in entries:
        if e.get("kind") == "anchor":
            out.setdefault(e["id"], []).append(e["value"])
    return out


# ------------------------------------------------------------------------------------------------- strictness
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def implies(new: dict, old: dict) -> bool:
    """Does criterion `new` imply criterion `old` (same metric) for every possible value of the metric?"""
    if new["metric"] != old["metric"]:
        return False
    no, a, op, b = new["op"], new["value"], old["op"], old["value"]
    if not (_num(a) and _num(b)):
        if op == "==":
            return no == "==" and a == b and type(a) is type(b)
        if op == "!=":
            return (no == "!=" and a == b) or (no == "==" and a != b and type(a) is type(b))
        return False
    if op == ">=":
        return (no in (">=", "==", ">") and a >= b)
    if op == ">":
        return (no == ">" and a >= b) or (no in (">=", "==") and a > b)
    if op == "<=":
        return (no in ("<=", "==", "<") and a <= b)
    if op == "<":
        return (no == "<" and a <= b) or (no in ("<=", "==") and a < b)
    if op == "==":
        return no == "==" and a == b
    if op == "!=":
        return (no == "!=" and a == b) or (no == "==" and a != b) or (no == ">" and a >= b) or (no == "<" and a <= b)
    return False


def not_implied(new: list[dict], old: list[dict]) -> list[dict]:
    """The old criteria that no new criterion implies (empty: the change is at least as strict)."""
    return [o for o in old if not any(implies(n, o) for n in new)]


# ------------------------------------------------------------------------------------------------- decisions
def decision_sections(docs_dir: str) -> dict[str, str]:
    """D-<n> -> the text of its section in DECISION_LOG.md."""
    try:
        text = open(os.path.join(docs_dir, "DECISION_LOG.md"), encoding="utf-8").read()
    except OSError:
        return {}
    return decision_sections_text(text)


def decision_sections_text(text: str) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^##+\s*(D-\d+)\b", line)
        if m:
            if cur:
                out[cur] = out.get(cur, "") + "\n".join(buf)
            cur, buf = m.group(1), [line]
        elif cur:
            if re.match(r"^#\s", line):
                out[cur] = out.get(cur, "") + "\n".join(buf)
                cur, buf = None, []
            else:
                buf.append(line)
    if cur:
        out[cur] = out.get(cur, "") + "\n".join(buf)
    return out


def mentions(section: str, rid: str) -> bool:
    return re.search(rf"(?<![\w.-]){re.escape(rid)}(?![\w-]|\.\d)", section) is not None


def rule_problems(st: dict, sections: dict[str, str]) -> dict[str, list[str]]:
    """Per requirement id, the change-control rules its ledger history breaks."""
    out: dict[str, list[str]] = {}
    for rid, r in st.items():
        if rid.startswith("__"):
            continue
        probs = []
        if r["added"] is None:
            probs.append("ledger records changes for a requirement that was never added")
        if r["duplicate_add"]:
            probs.append("requirement added twice in the ledger")
        if not r["criteria_history"]:
            probs.append("no criteria recorded in the ledger")
        for kind, old, new, dec in r["changes"]:
            def need_decision(what: str) -> None:
                if not dec or dec not in sections:
                    probs.append(f"{what} cites no recorded decision ({dec})")
                elif not mentions(sections[dec], rid):
                    probs.append(f"{what} cites {dec}, whose text does not name {rid}")
            if kind == "criteria":
                need_decision("criteria change")
                weaker = not_implied(new, old)
                if weaker:
                    probs.append("criteria change is not provably at least as strict: no new criterion implies "
                                 + "; ".join(f"{c['metric']} {c['op']} {c['value']!r}" for c in weaker[:4]))
            elif kind in ("tests", "implementation_files", "dependencies"):
                removed = sorted(set(old) - set(new))
                if removed:
                    need_decision(f"removal from {kind} ({removed[:3]})")
            elif kind in ("text", "tolerance"):
                need_decision(f"{kind} change")
        if probs:
            out[rid] = probs
    return out


def hook_pins(st: dict, sections: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Start-up hook pins (D-83 third addendum): path -> the sha256 of its latest pin, and the pins' problems. A pin is
    valid only when it cites a decision recorded in DECISION_LOG.md whose text names the path and quotes the first
    HOOK_SHA_PREFIX hex digits of the pinned sha256: re-pinning edited hook content needs a new, visible decision that
    states the new content's hash - the hook, the ledger entry and the decision text change together, and the ledger
    is append-only and anchored in git history. A path with an invalid latest pin is not pinned."""
    pins, probs = {}, []
    for hp, hist in sorted((st.get("__hooks__") or {}).items()):
        last = hist[-1]
        sha, dec = last.get("sha256"), last.get("decision")
        if not isinstance(sha, str) or len(sha) != 64:
            probs.append(f"start-up hook pin of {hp} carries no sha256")
            continue
        text = sections.get(dec or "")
        if not text:
            probs.append(f"start-up hook pin of {hp} cites no recorded decision ({dec})")
        elif hp not in text and os.path.basename(os.path.dirname(hp)) + "/" + os.path.basename(hp) not in text:
            probs.append(f"start-up hook pin of {hp} cites {dec}, whose text does not name the hook")
        elif sha[:HOOK_SHA_PREFIX] not in text:
            probs.append(f"start-up hook pin of {hp} cites {dec}, whose text does not quote the pinned hash "
                         f"{sha[:HOOK_SHA_PREFIX]}")
        else:
            pins[hp] = sha
    return pins, probs


# ------------------------------------------------------------------------------------------------- git anchoring
def _git(cwd: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True)


def _show(cwd: str, rev: str, rel: str) -> str | None:
    p = _git(cwd, "show", f"{rev}:./{rel}")                     # relative to cwd (the crucible root)
    return p.stdout.decode("utf-8") if p.returncode == 0 else None


def _is_prefix(older: str, newer: str) -> bool:
    a, b = older.splitlines(), newer.splitlines()
    return len(a) <= len(b) and b[:len(a)] == a


def git_problems(root: str) -> list[str]:
    """Append-only across every committed version (each version extends each parent's version, merges included),
    working copy committed, and pushed integration-branch copies are prefixes."""
    probs = []
    top = _git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    if not top:
        return ["not a git checkout: the ledger cannot be anchored"]
    rel = LEDGER_REL
    head = _show(root, "HEAD", rel)
    try:
        work = open(path(root), encoding="utf-8").read()
    except OSError:
        work = None
    if head is None:
        probs.append("the ledger is not committed")
    elif work != head:
        probs.append("the ledger has uncommitted changes")
    commits = _git(root, "rev-list", "--full-history", "HEAD", "--", rel).stdout.decode().split()
    for c in commits:
        cur = _show(root, c, rel)
        parents = _git(root, "rev-parse", f"{c}^@").stdout.decode().split()
        for p in parents:
            prev = _show(root, p, rel)
            if prev is None:
                continue
            if cur is None:
                probs.append(f"commit {c[:9]} deletes the ledger")
            elif not _is_prefix(prev, cur):
                probs.append(f"commit {c[:9]} rewrites the ledger (its parent {p[:9]}'s version is not a prefix)")
    for ref in INTEGRATION_REFS:
        if _git(root, "rev-parse", "--verify", "-q", ref).returncode != 0:
            continue
        pushed = _show(root, ref, rel)
        if pushed is not None and head is not None and not _is_prefix(pushed, head):
            probs.append(f"the ledger pushed to {ref} is not a prefix of HEAD's ledger (history rewritten)")
    return probs
