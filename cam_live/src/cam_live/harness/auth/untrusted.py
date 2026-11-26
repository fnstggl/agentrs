"""Handling of environment content (web pages, documents, messages, tool output).

Three mechanisms, none of which gives environment content any authority:

* :class:`Untrusted` wraps text read from the world. It cannot be formatted, concatenated or compared by accident;
  the raw text is available only through ``.data`` for encoding and storage. Anything that still reaches the
  authorization service inside such a wrapper is refused.
* :func:`policy_context` is the only way caller-supplied *context* enters a policy decision: a short allowlist of
  typed keys (for example the URL of the page a click refers to). Everything else is dropped. Decisions are a
  function of the typed ``ActionSpec`` / ``PrimitiveOp``, the operator-pinned task, the policy file and the
  service's own state.
* :func:`label_observation` marks instruction-like passages in an observation. The label is **data** for the learner
  and the operator; it grants nothing and blocks nothing.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ... import contracts as C


class UntrustedUseError(TypeError):
    """Untrusted text was about to be used as if it were trusted."""


@dataclass(frozen=True, eq=False)
class Untrusted:
    """Text that came from the environment. ``.data`` is the only accessor."""
    data: str
    source: str = ""

    def _refuse(self, *a: Any, **k: Any) -> Any:
        raise UntrustedUseError("untrusted environment text cannot be used as an instruction, a format argument or "
                                "a policy input; use .data explicitly where it is handled as data")

    __str__ = __format__ = __add__ = __radd__ = __mod__ = __rmod__ = __contains__ = _refuse
    __iter__ = __getitem__ = __bool__ = __eq__ = __hash__ = _refuse     # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Untrusted(len={len(self.data)}, source={self.source!r})"

    def __len__(self) -> int:
        return len(self.data)


def wrap_observation_text(obs: C.Observation) -> Untrusted:
    return Untrusted(obs.text, source=obs.provenance.uri or obs.provenance.source or obs.kind)


def contains_untrusted(obj: Any, _depth: int = 0) -> bool:
    if isinstance(obj, Untrusted):
        return True
    if _depth > 24:
        return False
    if isinstance(obj, dict):
        return any(contains_untrusted(k, _depth + 1) or contains_untrusted(v, _depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return any(contains_untrusted(v, _depth + 1) for v in obj)
    if isinstance(obj, C.Record):
        return any(contains_untrusted(getattr(obj, f), _depth + 1) for f in obj.__dataclass_fields__)
    return False


# ---------------------------------------------------------------------------------------------------------------
# The context allowlist
# ---------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_URLISH = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://[^\s]{1,2040}$")

#: context key -> validator. These are facts about *where* a typed action applies; none is free text.
CONTEXT_KEYS = {
    "page_url": lambda v: isinstance(v, str) and bool(_URLISH.match(v)),
    "tab_id": lambda v: isinstance(v, str) and bool(_TOKEN.match(v)),
    "session_id": lambda v: isinstance(v, str) and bool(_TOKEN.match(v)),
    "decision_id": lambda v: isinstance(v, str) and bool(_TOKEN.match(v)),
    "experiment_id": lambda v: isinstance(v, str) and bool(_TOKEN.match(v)),
}


def policy_context(context: Any) -> Tuple[Dict[str, str], List[str]]:
    """``(accepted, dropped_key_names)``. Unknown keys, wrong types, free text, observations and anything wrapped
    as :class:`Untrusted` are dropped; dropped *names* (truncated) are reported for the audit log, never values."""
    ok: Dict[str, str] = {}
    dropped: List[str] = []
    if context is None:
        return ok, dropped
    if not isinstance(context, dict):
        return ok, ["<context is not a mapping>"]
    for k, v in context.items():
        name = k if isinstance(k, str) else type(k).__name__
        check = CONTEXT_KEYS.get(name)
        if check is not None and not isinstance(v, Untrusted) and check(v):
            ok[name] = v
        else:
            dropped.append(re.sub(r"[^A-Za-z0-9_.\-]", "?", name)[:40])
    return ok, dropped[:32]


# ---------------------------------------------------------------------------------------------------------------
# Labelling instruction-like content (label only)
# ---------------------------------------------------------------------------------------------------------------
#: category -> patterns. The categories name what the passage *asks for*; they are descriptive labels.
INSTRUCTION_PATTERNS: Dict[str, List[str]] = {
    "override_instructions": [
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,40}\b(instruction|prompt|rule|directive|guideline)s?\b",
        r"\b(system|developer)\s*(prompt|message|override|notice)\s*:",
        r"<\s*/?\s*(system|assistant|instructions?)\s*>",
        r"\byou\s+(are|must)\s+now\b", r"\bnew\s+instructions?\s*:",
    ],
    "change_goal": [
        r"\b(your|the)\s+(new|real|actual|true)\s+(goal|objective|task|mission)\b",
        r"\b(change|replace|update|abandon|drop)\b.{0,30}\b(your|the)\s+(goal|objective|task|mission)\b",
        r"\binstead\b.{0,40}\byou\s+(should|must|need to|will)\b",
    ],
    "raise_limits": [
        r"\b(raise|increase|remove|lift|disable|bypass|ignore)\b.{0,40}\b(limit|budget|quota|cap|restriction|rate.?limit)s?\b",
        r"\bunlimited\s+(budget|spending|access)\b",
    ],
    "reveal_credentials": [
        r"\b(print|reveal|show|send|post|paste|output|exfiltrate|leak|share|tell me)\b.{0,60}\b(credential|password|passphrase|secret|token|api.?key|private.?key|cookie|session)s?\b",
        r"\bwhat\s+is\s+your\s+(password|token|api.?key|secret)\b",
    ],
    "self_approval": [
        r"\b(approve|authorize|authorise|grant|whitelist|allow)\b.{0,40}\b(yourself|this action|this request|it yourself|automatically|without (asking|approval|review))\b",
        r"\b(operator|admin|administrator|owner)\s+(has\s+)?(already\s+)?(approved|authorized|authorised)\b",
        r"\bno\s+(approval|authorization|authorisation|confirmation)\s+(is\s+)?(needed|required)\b",
    ],
    "call_url": [
        r"\b(call|visit|open|fetch|go to|navigate to|send (a |the )?(request|data|post) to|curl|wget|download)\b.{0,40}(https?://|www\.)\S+",
    ],
    "disable_safety": [
        r"\b(disable|turn off|stop|skip|circumvent|bypass)\b.{0,40}\b(safety|security|authorization|authorisation|audit|logging|verification|verifier|policy|sandbox|emergency stop)\b",
        r"\b(reset|clear)\b.{0,20}\bemergency stop\b",
    ],
    "run_code": [
        r"\b(run|execute|eval)\b.{0,30}\b(this|the following)\b.{0,20}\b(code|command|script|shell)\b",
    ],
}
_COMPILED = {cat: [re.compile(p, re.I | re.S) for p in pats] for cat, pats in INSTRUCTION_PATTERNS.items()}
LABEL_KEY = "untrusted_labels"


def detect_instructions(text: str, max_spans: int = 16) -> Dict[str, Any]:
    """Find instruction-like passages. Returns ``{"instruction_like": bool, "categories": [...], "spans": [...]}``
    where each span is ``[start, end, category]`` into ``text``."""
    cats: List[str] = []
    spans: List[List[Any]] = []
    t = text or ""
    for cat, pats in _COMPILED.items():
        for p in pats:
            for m in p.finditer(t):
                if cat not in cats:
                    cats.append(cat)
                if len(spans) < max_spans:
                    spans.append([m.start(), m.end(), cat])
    spans.sort()
    return {"instruction_like": bool(cats), "categories": sorted(cats), "spans": spans}


def _obs_strings(obs: C.Observation) -> str:
    parts = [obs.text]
    stack: List[Any] = [obs.data]
    n = 0
    while stack and n < 2000:
        cur = stack.pop()
        n += 1
        if isinstance(cur, str):
            parts.append(cur)
        elif isinstance(cur, dict):
            if LABEL_KEY in cur:
                cur = {k: v for k, v in cur.items() if k != LABEL_KEY}
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return "\n".join(p for p in parts if p)


def label_observation(obs: C.Observation) -> C.Observation:
    """Copy of ``obs`` with ``data["untrusted_labels"]`` set. Text, trust level and everything else are unchanged:
    the content stays untrusted data whether or not anything was detected."""
    out = copy.deepcopy(obs)
    found = detect_instructions(_obs_strings(obs))
    spans_in_text = [s for s in detect_instructions(obs.text)["spans"]]
    out.data = dict(out.data)
    out.data[LABEL_KEY] = {"instruction_like": found["instruction_like"], "categories": found["categories"],
                           "spans": spans_in_text, "note": "label only: grants nothing, blocks nothing"}
    return out


def label_result(result: C.ActionResult) -> C.ActionResult:
    out = copy.copy(result)
    out.observations = [label_observation(o) for o in result.observations]
    return out
