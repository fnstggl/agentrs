"""Claim integrity and outcome integrity (MASTER_SPEC Parts 15.1, 15.2, 11.6; Gate I, K and S notes).

* `scan_claims(kind, files)` - finds statements of a claim kind and decides per *clause* whether the clause asserts
  it or disclaims it. A clause is a claim when it names the claim's subject and its assertion; it is disclaimed only
  when a negation governs the assertion in the same clause (a negator up to six words before the assertion term,
  or a disclaiming phrase such as "no realism is claimed", "unrealistic", "unvalidated", "is not evidence"). A
  "no" elsewhere in the sentence ("... with no hand tuning") disclaims nothing. Kinds:

      realism_cascades     empirically realistic large-scale viral dynamics (Gate I note)
      real_world           real-world / open-internet generalisation (Part 15.1)
      integration_success  integration presented as learning success or general intelligence (Gate S note)
      reference_learning   a scripted reference presented as evidence of learning (Gate K note)

  `CONTROLS[kind]` are planted sentences (claims and disclaimers) the scanner must classify correctly before its
  verdict on the corpus counts.
* `corpus()` - every document a claim could hide in: README, IMPLEMENTATION_STATUS, ARCHITECTURE, docs/**/*.md
  (except the specification, its clarifications and the audits, which quote requirements and findings), evidence
  notes (`notes` of every evidence JSON under docs/v5/evidence), and module docstrings of src/crucible.
* `banned_proxy_checks()` - the outcome functions the instrument uses read only verifier records: synthetic verifier
  records with the seven banned proxies injected give the same outcome (Part 15.2), and the instrument's own
  modules import no language-model judge.
* `classify_failure(exc_or_record)` - separates algorithmic limitations from environment, interface,
  infrastructure and partition failures (Part 11.6); `unclassified` is the only other answer and counts as a
  defect of the report.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
from typing import Iterable, Mapping

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

NEG = r"(?:not|no|nothing|never|nor|cannot|can't|isn't|aren't|doesn't|don't|won't|without|neither|unless|none)"
KINDS = {
    "realism_cascades": {
        "subject": r"(viral|virality|cascade|cascades|ten[- ]million|10[ ,]?000[ ,]?000|10\s?m\b|million[- ]view|large[- ]scale reach)",
        "assert": r"(realistic|realism|empirically|real[- ]world|faithful|lifelike|matches real|reproduces real)",
        "disclaim": r"(unrealistic|unvalidated|no realism|not (?:a )?realis|realism is not|is not claimed|not claimed|"
                    r"computational capacity, not realism|known unrealistic|makes no claim|claims? no)"},
    "real_world": {
        "subject": r"(real[- ]world|open internet|actual internet|real internet|outside the simulat)",
        "assert": r"(generali[sz]|transfer|works? (?:on|in)|perform(?:s|ance)? (?:on|in)|succeed|competen)",
        "disclaim": r"(no claim|not claim|\bclaims? of\b|unless .{0,40}(?:authori[sz]ed|executed|performed)|has not been|"
                    r"have not been|not (?:been )?(?:shown|demonstrated|performed|executed)|without authori)"},
    "integration_success": {
        "subject": r"(\bcam\b|reference learner|integration|integrated|interacts?|interaction|gate s\b)",
        "assert": r"(general intelligence|learning success|has learned|have learned|"
                  r"learned to (?:run|achieve|solve|master|succeed)|achiev\w* (?:agi|general)|scientific success|"
                  r"proves? .{0,30}learn)",
        "disclaim": r"(not (?:the )?(?:scientific )?success|not evidence|does not (?:show|claim|demonstrate)|"
                    r"integration, not|no claim|not a claim|not that)"},
    "reference_learning": {
        "subject": r"(reference (?:strategy|policy|founder)|scripted)",
        "assert": r"(evidence (?:that|of) .{0,30}learn|learned to achieve|shows? .{0,30}learn|cam learned|"
                  r"learner (?:achieved|learned))",
        "disclaim": r"(not (?:a learner|evidence)|is not evidence|no evidence|labelled scripted|not treated as)"},
}
CONTROLS = {
    "realism_cascades": [
        ("Crucible reproduces empirically realistic ten-million-view viral cascades with no hand tuning.", True),
        ("Our viral cascades are realistic at ten-million scale.", True),
        ("The 10 million view cascades match real platforms empirically.", True),
        ("The scale test establishes computational capacity, not realism, of ten-million-view cascades.", False),
        ("No realism is claimed for the cascade sizes.", False),
        ("The cascade model is known unrealistic for viral reach and is not validated.", False)],
    "real_world": [
        ("The learner generalises to the real world.", True),
        ("These results show transfer to the open internet.", True),
        ("Real-world generalisation has not been demonstrated: no authorised Level 6 experiment was executed.", False),
        ("We make no claim of real-world generalisation unless authorised experiments are performed.", False)],
    "integration_success": [
        ("The integration shows that CAM has learned to run a business.", True),
        ("This proves the reference learner achieved general intelligence.", True),
        ("This gate verifies integration, not the scientific success of the learner.", False),
        ("Interacting with the environment is not evidence that the learner has learned anything.", False)],
    "reference_learning": [
        ("The reference strategy is evidence that CAM learned to achieve the objective.", True),
        ("The scripted reference policy shows the learner learned recurring revenue.", True),
        ("LABELLED SCRIPTED reference strategy (not a learner, not evidence of learning).", False),
        ("The reference strategy is not evidence that any learner can achieve the objective.", False)],
}
EXCLUDE = ("docs/v5/MASTER_SPEC.md", "docs/v5/MASTER_SPEC_CLARIFICATIONS.md", "docs/v5/audit/", "docs/v5/reviews/",
           "docs/v5/ACCEPTANCE_MANIFEST.json", "docs/v5/DEFECTS.md")


def _units(text: str) -> list[str]:
    """Paragraph text with table cells, list items and headings as separate units; code blocks dropped."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    units, para = [], []
    for line in text.splitlines():
        st = line.strip()
        if not st:
            if para:
                units.append(" ".join(para))
                para = []
            continue
        if st.startswith("|"):
            if para:
                units.append(" ".join(para))
                para = []
            units += [c for c in st.split("|") if c.strip()]
            continue
        if re.match(r"^([-*+]|\d+\.|#+)\s", st):
            if para:
                units.append(" ".join(para))
            para = [re.sub(r"^([-*+]|\d+\.|#+)\s+", "", st)]
            continue
        para.append(st)
    if para:
        units.append(" ".join(para))
    return units


def _clauses(text: str) -> list[str]:
    out = []
    for u in _units(text):
        u = re.sub(r"\s+", " ", u)
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'`*])", u):
            out += [c for c in re.split(r";|:\s|\s(?:but|whereas|although|while|however)\s|,\s(?=which\b)", sent)
                    if c.strip()]
    return out


HYPOTHETICAL = r"(?:whether|would|could|should|must|might|may|if|ask|aims?|goal|want|wants|can)"


def classify_clause(clause: str, kind: str) -> str:
    """'claim', 'disclaimed' or 'none' for one clause."""
    k = KINDS[kind]
    c = clause.lower()
    if not re.search(k["subject"], c):
        return "none"
    m = re.search(k["assert"], c)
    if not m:
        return "none"
    if re.search(k["disclaim"], c) or re.match(r"\s*(?:do not|don't|never|no|not)\b", c):
        return "disclaimed"
    if re.search(rf"\b{HYPOTHETICAL}\b", c[: m.start()]):
        return "none"                                   # a question, an aim or a condition is not a claim
    before = c[: m.start()].split()[-6:]
    if any(re.fullmatch(NEG, w.strip(",.;()'\"`")) for w in before):
        return "disclaimed"
    if re.search(rf"\b{NEG}\b[^.]{{0,25}}$", c[: m.start()]):
        return "disclaimed"
    return "claim"


def scan_text(text: str, kind: str) -> dict:
    claims, disclaimed = [], 0
    for cl in _clauses(text):
        r = classify_clause(cl, kind)
        if r == "claim":
            claims.append(cl.strip()[:200])
        elif r == "disclaimed":
            disclaimed += 1
    return {"claims": claims, "disclaimed": disclaimed}


def control_check(kind: str) -> dict:
    """The planted sentences: every claim found, every disclaimer cleared."""
    wrong = []
    for sent, is_claim in CONTROLS[kind]:
        found = bool(scan_text(sent, kind)["claims"])
        if found != is_claim:
            wrong.append(sent)
    return {"controls": len(CONTROLS[kind]), "misclassified": wrong, "ok": not wrong}


def _docstrings(path: str) -> str:
    try:
        tree = ast.parse(open(path).read())
    except (SyntaxError, UnicodeDecodeError):
        return ""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(n)
            if d:
                out.append(d)
    return "\n\n".join(out)


def _evidence_notes(path: str) -> str:
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return ""
    if isinstance(d, dict):
        return "\n".join(str(d.get(k) or "") for k in ("notes", "note", "summary"))
    return ""


def corpus(root: str | None = None) -> dict[str, str]:
    root = root or ROOT
    files: dict[str, str] = {}
    pats = ["README.md", "IMPLEMENTATION_STATUS.md", "ARCHITECTURE.md", "docs/**/*.md"]
    for pat in pats:
        for p in glob.glob(os.path.join(root, pat), recursive=True):
            rel = os.path.relpath(p, root)
            if any(rel.startswith(x) for x in EXCLUDE):
                continue
            try:
                files[rel] = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
    for p in glob.glob(os.path.join(root, "docs/v5/evidence/**/*.json"), recursive=True):
        files[os.path.relpath(p, root)] = _evidence_notes(p)
    for p in glob.glob(os.path.join(root, "src/crucible/**/*.py"), recursive=True):
        rel = os.path.relpath(p, root)
        if rel.endswith("research/v5/integrity.py"):
            continue                              # this module states the patterns and the planted sentences
        files[rel] = _docstrings(p)
    return files


def scan_claims(kind: str, files: Mapping[str, str] | None = None, supported: bool = False) -> dict:
    """Unsupported claims of `kind` in the corpus (every claim is unsupported unless `supported`)."""
    files = corpus() if files is None else files
    found, checked = [], 0
    for rel, text in files.items():
        r = scan_text(text, kind)
        checked += len(r["claims"]) + r["disclaimed"]
        for c in r["claims"]:
            found.append({"file": rel, "clause": c})
    ctl = control_check(kind)
    return {"kind": kind, "files_scanned": len(files), "statements_checked": checked,
            "claims": found, "unsupported_claims": 0 if supported else len(found), "controls": ctl}


# ====================================================================== Part 15.2: banned proxy outcomes
BANNED_PROXIES = {
    "tool_call_count": {"tool_calls": 10_000, "executes": 10_000, "actions": {"a0": 10_000}},
    "reasoning_length": {"reasoning_chars": 10 ** 7, "reasoning": "x" * 2000},
    "posts": {"posts": 500, "messages": [{"kind": "post"}] * 50},
    "websites_created": {"websites": 40, "sites_created": 40},
    "agent_reported_success": {"agent_reported_success": True, "success_claimed": True, "journal": ["I succeeded"]},
    "llm_judgment": {"llm_judgment": 1.0, "judge_score": 10.0},
    "unverified_milestones": {"milestones": ["launched", "profitable"], "milestones_reached": 9},
}


def banned_proxy_checks() -> dict:
    """For each banned proxy kind: inject it into verifier records (top level and inside the outcome) and check that
    the instrument's outcome function returns the same value; and check that no instrument module imports a
    language-model judge. Returns {kinds_checked, outcomes_using_proxies, details}."""
    from .episodes import verified_value
    base_rapid = {"verifier": "crucible.rapid RapidEnv.outcome", "outcome": {"net": 1234.5, "value": 1234.5,
                                                                            "success": None}}
    base_mech = {"verifier": "research.tasks.Task", "current": {"net_cents": -77.0}}
    details, used = {}, 0
    for kind, inj in BANNED_PROXIES.items():
        changed = False
        for base in (base_rapid, base_mech):
            v0 = verified_value(base)
            top = dict(base, **inj)
            nested = json.loads(json.dumps(base))
            for key in ("outcome", "current"):
                if key in nested:
                    nested[key] = dict(nested[key], **{k: v for k, v in inj.items() if k not in ("actions",)})
            for rec in (top, nested):
                try:
                    changed = changed or verified_value(rec) != v0
                except ValueError:
                    changed = True
        details[kind] = {"outcome_changed": changed}
        used += int(changed)
    pkg = os.path.dirname(os.path.abspath(__file__))
    judge = []
    for p in glob.glob(os.path.join(pkg, "*.py")):
        tree = ast.parse(open(p).read())
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mod = getattr(n, "module", None) or ""
                names = [a.name for a in n.names]
                if any(re.search(r"llm|judge|semantic\.(?:model|infer)|openai|anthropic", x or "")
                       for x in [mod, *names]):
                    judge.append(f"{os.path.basename(p)}:{mod or names}")
    details["llm_judgment"]["instrument_imports_judge"] = judge
    if judge:
        used += 1
    return {"kinds_checked": len(BANNED_PROXIES), "outcomes_using_proxies": used, "details": details}


# ====================================================================== Part 11.6: failure classification
FAILURE_CAUSES = ("algorithmic_limitation", "environment_failure", "interface_failure", "infrastructure_failure",
                  "partition_refusal")


def classify_failure(x) -> dict:
    """Cause of a failed run. `x` is an exception raised while running, or a run record {completed, score, q,
    censored, refused, error}."""
    from .distribution import PartitionError
    if isinstance(x, BaseException):
        tb = x.__traceback__
        frames = []
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_filename)
            tb = tb.tb_next
        where = frames[-1] if frames else ""
        name = type(x).__name__
        if isinstance(x, PartitionError):
            return {"cause": "partition_refusal", "detail": str(x)[:200]}
        if name in ("TierUnavailable",) or any(p in where for p in ("/rapid/", "/civ/", "/domains/", "/core/",
                                                                     "/mechanisms/", "/worldgen/", "/population/",
                                                                     "/experience/")):
            return {"cause": "environment_failure", "detail": f"{name} in {os.path.relpath(where, ROOT)}"}
        if name in ("InterfaceError", "ActionError") or "/research/v5/episodes.py" in where:
            return {"cause": "interface_failure", "detail": f"{name}: {str(x)[:160]}"}
        if name in ("EvaluatorError", "BrokenProcessPool", "TimeoutExpired", "OSError", "EOFError",
                    "ConnectionError", "BrokenPipeError"):
            return {"cause": "infrastructure_failure", "detail": f"{name}: {str(x)[:160]}"}
        if "/research/v5/learners.py" in where:
            return {"cause": "algorithmic_limitation", "detail": f"learner raised {name}"}
        return {"cause": "unclassified", "detail": f"{name} in {where}"}
    r = dict(x)
    if r.get("refused"):
        return {"cause": "partition_refusal", "detail": str(r["refused"])[:200]}
    if r.get("error"):
        return {"cause": "environment_failure", "detail": str(r["error"])[:200]}
    if r.get("completed") and (r.get("censored") or (r.get("score") is not None and r.get("q") is not None
                                                     and r["score"] < r["q"])):
        return {"cause": "algorithmic_limitation", "detail": "the run completed and the learner did not reach "
                                                             "the competence threshold"}
    return {"cause": "unclassified", "detail": "no failure cause recognised"}
