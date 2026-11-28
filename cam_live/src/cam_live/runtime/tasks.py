"""cam_live.runtime.tasks — external task configuration: YAML -> :class:`~cam_live.contracts.TaskSpec` (spec Part 13).

A task file states the objective, how success is verified independently of CAM, which resources and interfaces are
authorized, the experiment boundaries, the measurement protocol and the data permissions. It never contains a way
of achieving the objective. Three mechanisms enforce that:

* a **closed schema** (:data:`SCHEMA`): every key at every depth must be one the runtime, the harness or the verifier
  reads; anything else is rejected with its dotted path. Apart from three bounded free-text fields (``objective``,
  ``success.definition``, ``measurement.protocol``) every value is a number, a boolean, an enumerated word or an
  identifier without spaces, so prose cannot be hidden in an unlisted key, a list item or a label;
* a **mechanical text scan** (:func:`strategy_findings`) of every string and mapping key: forbidden keys and
  strategy-like text (procedures, directives, cadences, recruitment channels, platforms, audiences) are rejected;
* the **objective text** is given to CAM verbatim, so strategy-like content in it (:func:`objective_flags`) is
  refused unless the operator confirms that exact text with ``objective_confirmation: sha256:<hex>``.

YAML anchors, aliases, merge keys and duplicate keys are refused (they could make the text that is read differ from
the text that was reviewed). Unset items are written with the marker ``REQUIRED`` (optionally ``REQUIRED: what the
operator must provide``). A task in mode ``live`` with any unset item cannot be loaded for a run; resources are never
assumed unlimited; a limit must be a finite non-negative number (0 = "not set", refused for a live run).

What the learner receives is :func:`learner_view`: the objective text, limits, interfaces, data permissions, stop
conditions, the names of the secondary metrics, the objective metric and target, and the resources it acts on —
never the success definition, the exclusion rules, the evidence source, the measurement protocol, credential names,
test-account identifiers or the authorization record.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import unicodedata
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import yaml

from ..contracts import Limits, Modality, TaskSpec, Trust

UNSET_MARKER = "REQUIRED"
MODES = ("diagnostic", "staging", "live")
LIMIT_FIELDS = ("duration_s", "money", "compute_s", "max_external_actions", "max_decisions")
TOP_LEVEL_KEYS = ("version", "objective_id", "objective", "objective_confirmation", "success", "family", "resources",
                  "interfaces", "limits", "measurement", "data_permissions", "stop_conditions", "secondary_metrics",
                  "mode", "origin")
#: stop conditions the runtime knows how to enforce (anything else in a task file is rejected)
STOP_CONDITIONS = ("verified_success", "verified_failure", "duration_exhausted", "money_exhausted",
                   "compute_exhausted", "external_actions_exhausted", "decisions_exhausted", "operator_stop",
                   "emergency_stop", "authorization_revoked")
#: data permissions that must be stated explicitly (true or false) before a live run
REQUIRED_DATA_PERMISSIONS = ("store_message_content", "train_on_message_content", "store_member_identifiers",
                             "store_aggregate_metrics", "collect_feedback_with_consent")
#: every data permission a component reads (runtime.logs, harness.discord, harness/auth/default_policy.yaml data_use)
DATA_PERMISSIONS = REQUIRED_DATA_PERMISSIONS + ("aggregate_metrics", "read_message_content", "member_list",
                                                "collect_personal_data", "discord_message_content")
#: the evidence source the verifier uses for a task with Discord resources when the task names none
DEFAULT_DISCORD_EVIDENCE_SOURCE = "cam_live.harness.discord_membership:membership_source"

#: the objective is given to CAM verbatim; the operator's planning-phase objective (2026-11-27) is ~1300 characters
TEXT_LIMITS = {"objective": 2000, "success.definition": 2000, "measurement.protocol": 2000}
MAX_ID = 200                       # identifiers, names, URLs, paths
MAX_LIST = 256                     # ordinary lists
MAX_ID_LIST = 100_000              # lists of account identifiers
#: spellings of "fill this in" other than the documented marker; refused so that they are never taken as values
PLACEHOLDERS = ("todo", "tbd", "tbc", "fixme", "changeme", "change_me", "change-me", "xxx", "placeholder",
                "fill me in", "fill in", "<fill in>", "???", "...")

# ---- rejection lists (this module only *refuses* such content; nothing here is ever shown to the learner) -----
#: mapping keys that would carry a solution; matched as whole keys and as key tokens
REJECTED_KEYS = tuple(TaskSpec.FORBIDDEN_KEYS) + (
    "tactic", "tactics", "approach", "method", "methods", "recipe", "instructions", "guidance", "hints", "tips",
    "roadmap", "marketing", "topic", "theme", "calendar", "howto", "procedure", "workflow", "sequence",
    "framework", "suggestions", "ideas", "recommended", "examples", "strategies", "plans", "milestone", "milestones",
    "todo", "todos", "outline", "cadence", "growth", "growthplan", "audience", "focus", "niches", "moves", "tasks",
    "checklist", "agenda", "itinerary", "goals", "objectives", "estrategia", "plan_de")
#: schema keys that contain a rejected token but are part of the contract itself
_SCHEMA_SAFE_KEYS = ("max_external_actions",)
#: text patterns that read as a way of achieving the objective (label -> regex)
REJECTED_TEXT: Dict[str, str] = {
    "enumerated procedure": r"(?im)^\s*\d+[.)]\s+\S|\bstep\s*\d+\b|\b(?:first|second|third)(?:ly)?,"
                            r"|\bstep\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
    "ordered instruction": r"(?i)\bfirst\b[^.]{0,80}\b(?:then|next|after that|finally)\b|\bthen\s+(?:post|share|invite|run|"
                           r"open|ask|launch|write|publish|send)\b",
    "directive to the agent": r"(?i)\b(?:you|cam|the agent|the model|the learner)\s+(?:should|must|shall|ought to|"
                              r"needs? to|has to|is to|will|can|could|may want to)\b"
                              r"|\b(?:you|cam|the agent|the model|the learner)\s+(?:is|are)\s+(?:expected|supposed|"
                              r"required|asked|encouraged|meant|told|advised|instructed)\s+to\b",
    "timing directive": r"(?i)\b(?:every|each)\s+(?:day|week|hour|morning|evening|night|weekday|weekend|"
                        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b(?:daily|weekly|hourly|nightly)\b"
                        r"|\b\d+\s*(?:x|times)\s*(?:a|per)\s*(?:day|week|hour)\b|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
                        r"|\b(?:per|a|each|every)\s+(?:day|week|month|fortnight)\b|\b(?:mornings|evenings|nights|weekends)\b"
                        r"|\bcada\s+d[ií]a\b",
    "strategy vocabulary": r"(?i)\b(?:niche|tactic\w*|e?strat[eé]g\w*|marketing|playbook|growth hack\w*|funnel\w*|"
                           r"cross[- ]?promot\w*|giveaways?|advertis\w*|outreach|influencers?|seo|viral\w*)\b",
    "suggested solution": r"(?i)\b(?:for example|e\.g\.|such as|consider|try|recommend\w*|suggest\w*|best practice\w*|"
                          r"proven|playbook)\b[^.]{0,60}\b(?:post\w*|invit\w*|shar\w*|launch\w*|build\w*|publish\w*|messag\w*)\b",
    "recruitment channel": r"(?i)\brecruit\w*\b[^.]{0,80}\b(?:through|via|on|in|from|at|by)\b"
                           r"|\b(?:through|via|by)\s+(?:\w+\s+){0,3}(?:threads?|posts?|forums?|subreddits?|ads?|"
                           r"campaigns?|newsletters?|partners?|partnerships?|amas?|cold (?:dm|message)s?)\b",
    "platform or subreddit": r"(?i)(?<![\w/])r/[A-Za-z0-9_]{2,}|\b(?:reddit|twitter|tiktok|youtube|facebook|instagram|"
                             r"telegram|whatsapp|linkedin|twitch|mastodon|hacker ?news|product ?hunt|subreddits?|"
                             r"4chan|quora|medium\.com|threads\.net|bluesky|bsky|snapchat|pinterest|tumblr|kick\.com|"
                             r"x\.com|steam ?community|github ?discussions|stack ?overflow)\b",
    # G2R-1b: Discord server listing / bump sites and the moves made on them
    "listing or bump site": r"(?i)\b(?:disboard|top[ .]?gg|discord[ .]?me|discords[ .]?com|discordservers?|discadia|"
                            r"disforge|discordhome|discordlist\w*|discord ?bot ?list|server ?lists?|listing ?sites?|"
                            r"server ?directory|bump(?:s|ed|ing|er)?|cool ?down)\b",
    "outreach move": r"(?i)\b(?:cold[ -]?(?:dm|message|email)s?|mass[ -]?(?:dm|message|invite)s?|dm(?:s|ing|ed)? "
                     r"(?:every|each|all|new|them|people|members|users)|invite ?links?|shill\w*|spam\w*|promot\w*|"
                     r"cross ?post\w*|referrals?|collabs?|partnerships?|shout ?outs?|boost(?:s|ing)? (?:the|our|it)|"
                     r"word of mouth)\b",
}
#: further patterns for the objective text, which CAM receives verbatim (flagged; confirmable by the operator)
OBJECTIVE_TEXT: Dict[str, str] = {
    "audience or topic of the community": r"(?i)\b(?:community|server|guild|group|channel|site|utility|space|hub)\s+"
                                          r"(?:is\s+|will be\s+)?(?:for|about|around|on the topic of|focused on|"
                                          r"dedicated to|aimed at|targeting|centred on|centered on)\b"
                                          r"|\b(?:for|among)\s+(?:\w+\s+){0,3}(?:fans|developers|gamers|players|students|"
                                          r"teens|teenagers|parents|artists|creators|enthusiasts|beginners|"
                                          r"professionals|hobbyists|speedrunners|collectors|readers|writers)\b",
    "more than one sentence": r"[.!?]\s+\S",
    "imperative steps": r"(?i)(?:^|[.;:!?]\s+)(?:post|share|invite|ask|send|advertise|promote|recruit|spam|dm|"
                        r"message|reach out|partner|cross-post|run a|host a)\b",
}
#: patterns for the verifier-only free text (success.definition, measurement.protocol): moves stated as commands
INSTRUCTION_TEXT: Dict[str, str] = {
    "imperative instruction": r"(?i)(?:^|[.;:!?]\s+)(?:open|create|post|write|share|invite|run|launch|ask|send|start|"
                              r"build|publish|advertise|promote|recruit|spam|dm|message|reach out|partner|cross-post|"
                              r"use)\s",
}
_REJECTED_TEXT = [(label, re.compile(rx)) for label, rx in REJECTED_TEXT.items()]
_OBJECTIVE_TEXT = [(label, re.compile(rx)) for label, rx in OBJECTIVE_TEXT.items()]
_INSTRUCTION_TEXT = [(label, re.compile(rx)) for label, rx in INSTRUCTION_TEXT.items()]
_FREE_TEXT_PATHS = ("objective", "success.definition", "measurement.protocol")


class TaskValidationError(ValueError):
    def __init__(self, problems: List[str], path: str = ""):
        self.problems, self.path = list(problems), path
        super().__init__((f"{path}: " if path else "") + "; ".join(self.problems))


def is_unset(v: Any) -> bool:
    return v is None or (isinstance(v, str) and (not v.strip() or v.strip().upper().startswith(UNSET_MARKER)))


def _walk(node: Any, path: str = "") -> Iterator[Tuple[str, str, Any]]:
    """Yield (path, key, value) for every mapping entry and list element of a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, str(k), v
            yield from _walk(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            p = f"{path}[{i}]"
            yield p, "", v
            yield from _walk(v, p)


# ==============================================================================================================
# closed schema
# ==============================================================================================================
_ID_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+-]{0,199}$")           # identifiers without spaces
_SLUG_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NAME_RX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")                    # destination / channel labels
_NOSPACE_RX = re.compile(r"^\S{1,%d}$" % MAX_ID)
# G2R-1: bounded (a label, a metric name and a tool pattern are names, not sentences)
_FAMILY_RX = re.compile(r"^(?=.{1,64}$)[a-z0-9_-]+(?:/[a-z0-9_-]+)*$")
_METRIC_RX = re.compile(r"^(?=.{1,80}$)[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
_TOOL_RX = re.compile(r"^(?=.{1,80}$)(?:\*|[a-z][a-z0-9_]*(?:\.(?:[a-z0-9_]+|\*))*(?:\.?\*)?)$")
_SOURCE_RX = re.compile(r"^(?:jsonl?:\S{1,1000}|[A-Za-z_][\w.]{0,199}:[A-Za-z_]\w{0,99})$")
_CONFIRM_RX = re.compile(r"^sha256:[0-9a-f]{64}$")
_PERM_RX = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")                        # platform permission names (Discord scope)


class _Spec:
    """One node of the schema. ``check(value, path, out)`` appends problem texts."""

    def check(self, v: Any, path: str, out: List[str]) -> None:
        raise NotImplementedError        # pragma: no cover — every concrete spec overrides it


class _Scalar(_Spec):
    def __init__(self, what: str, test: Callable[[Any], bool]):
        self.what, self.test = what, test

    def check(self, v: Any, path: str, out: List[str]) -> None:
        if is_unset(v):
            return
        if not self.test(v):
            out.append(f"{path}: must be {self.what} (got {_show(v)})")


class _List(_Spec):
    def __init__(self, item: _Spec, max_len: int = MAX_LIST, exact: Optional[int] = None):
        self.item, self.max_len, self.exact = item, max_len, exact

    def check(self, v: Any, path: str, out: List[str]) -> None:
        if is_unset(v):
            return
        if not isinstance(v, list):
            out.append(f"{path}: must be a list (got {_show(v)})")
            return
        if len(v) > self.max_len:
            out.append(f"{path}: at most {self.max_len} items (got {len(v)})")
        if self.exact is not None and len(v) != self.exact:
            out.append(f"{path}: must have exactly {self.exact} items")
        for i, x in enumerate(v[: self.max_len]):
            if is_unset(x) and isinstance(x, str) and x.strip():
                continue                                   # REQUIRED inside a list: reported by unset_paths
            self.item.check(x, f"{path}[{i}]", out)


class _Map(_Spec):
    """A mapping with a fixed set of keys (``fields``), or free-form keys matching ``key_rx`` (``values``)."""

    def __init__(self, fields: Optional[Dict[str, _Spec]] = None, *, key_rx: Optional["re.Pattern[str]"] = None,
                 values: Optional[_Spec] = None, max_keys: int = MAX_LIST, top: bool = False):
        self.fields, self.key_rx, self.values, self.max_keys, self.top = fields, key_rx, values, max_keys, top

    def check(self, v: Any, path: str, out: List[str]) -> None:
        if is_unset(v):
            return
        if not isinstance(v, dict):
            out.append(f"{path or 'task file'}: must be a mapping (got {_show(v)})")
            return
        if len(v) > self.max_keys:
            out.append(f"{path}: at most {self.max_keys} keys (got {len(v)})")
        for k, x in v.items():
            p = f"{path}.{k}" if path else str(k)
            if not isinstance(k, str):
                out.append(f"{p}: keys must be text (got {_show(k)})")
                continue
            if self.fields is not None:
                spec = self.fields.get(k)
                if spec is None:
                    out.append(f"unknown top-level key {k!r}" if self.top else
                               f"{p}: unknown key {k!r} (not part of the task schema; known: {', '.join(self.fields)})")
                    continue
                spec.check(x, p, out)
            else:
                if self.key_rx is not None and not self.key_rx.match(k):
                    out.append(f"{p}: key {k!r} must match {self.key_rx.pattern}")
                    continue
                if self.values is not None:
                    self.values.check(x, p, out)


def _show(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _num(lo: Optional[float] = None, hi: Optional[float] = None, what: str = "") -> _Scalar:
    def ok(v: Any) -> bool:
        return _finite_number(v) and (lo is None or v >= lo) and (hi is None or v <= hi)
    rng = what or ("a finite number" + (f" >= {lo:g}" if lo is not None else "") + (f" <= {hi:g}" if hi is not None else ""))
    return _Scalar(rng, ok)


def _str(rx: "re.Pattern[str]", what: str) -> _Scalar:
    return _Scalar(what, lambda v: isinstance(v, str) and bool(rx.match(v)))


def _text(max_len: int) -> _Scalar:
    return _Scalar(f"text of at most {max_len} characters",
                   lambda v: isinstance(v, str) and len(v) <= max_len and "\x00" not in v)


def _enum(values: Tuple[str, ...]) -> _Scalar:
    return _Scalar(f"one of {', '.join(values)}", lambda v: isinstance(v, str) and v in values)


_BOOL = _Scalar("true or false", lambda v: isinstance(v, bool))
_INT = _Scalar("an integer", lambda v: isinstance(v, int) and not isinstance(v, bool))
_ID = _Scalar("an identifier without spaces (text or integer)",
              lambda v: (isinstance(v, int) and not isinstance(v, bool) and v >= 0)
              or (isinstance(v, str) and bool(_ID_RX.match(v))))
_SLUG = _str(_SLUG_RX, "an identifier (letters, digits, _ . : -; no spaces)")
_NOSPACE = _str(_NOSPACE_RX, f"one word without spaces (at most {MAX_ID} characters: a URL, path, pattern or name)")
_TOOL = _str(_TOOL_RX, "a tool name or tool-name pattern such as 'discord.*' or 'api'")
_METRIC = _str(_METRIC_RX, "a metric name such as 'secondary.retention_7d'")
_PATTERNS = _List(_NOSPACE)
_IDS = _List(_ID, MAX_ID_LIST)

SCHEMA = _Map({
    "version": _INT,
    "objective_id": _SLUG,
    "objective": _text(TEXT_LIMITS["objective"]),
    "objective_confirmation": _str(_CONFIRM_RX, "'sha256:<64 hex digits>' of the exact objective text"),
    "success": _Map({
        "verifier": _SLUG,
        "metric": _METRIC,
        "target": _num(0.0),
        "confidence": _num(0.0, 1.0, "a number strictly between 0 and 1"),
        "definition": _text(TEXT_LIMITS["success.definition"]),
        "evidence_source": _str(_SOURCE_RX, "'jsonl:<path>', 'json:<path>' or '<module>:<factory>'"),
        "exclusions": _Map({
            "exclude_bots": _BOOL,
            "test_account_ids": _IDS,
            "count_departed": _BOOL,
            "exclude_flags": _List(_SLUG),
            "uncertain_flags": _List(_SLUG),
            "pending_screening": _enum(("uncertain", "exclude", "count")),
            "suspect_invite_codes": _List(_SLUG, MAX_ID_LIST),
            "suspect_member_ids": _IDS,
            "min_account_age_s": _num(0.0),
            "uncertain_p_range": _List(_num(0.0, 1.0), exact=2),
            "confidence": _num(0.0, 1.0, "a number strictly between 0 and 1"),
            "retention_window_s": _num(0.0),
        }),
    }),
    "family": _str(_FAMILY_RX, "a label such as 'community/membership' (lowercase words separated by '/')"),
    "resources": _Map({
        "authorized_tools": _List(_TOOL),
        "authorized_destinations": _PATTERNS,
        "currency": _str(re.compile(r"^[A-Za-z]{3,10}$"), "a currency code such as 'USD'"),
        "owned_site": _NOSPACE,
        "owned_sites": _List(_Map({"name": _str(_NAME_RX, "a lowercase label"), "root": _NOSPACE,
                                   "base_url": _NOSPACE, "access_log": _NOSPACE})),
        "staging_site": _NOSPACE,
        "operator_authorization_record": _SLUG,
        "discord_guild_id": _ID,
        "discord_bot_credential_name": _SLUG,
        "publishing": _Map({"authorized": _BOOL, "approval_required": _BOOL, "destinations": _PATTERNS}),
        "discord": _Map({
            "guild_id": _ID, "token_secret": _SLUG, "base_url": _NOSPACE,
            "scope": _List(_str(_PERM_RX, "a platform permission name such as 'SEND_MESSAGES'")),
            "test_account_ids": _IDS, "state_dir": _NOSPACE, "metric_channel_id": _ID, "application_id": _ID,
            "channels": _Map(key_rx=_NAME_RX, values=_ID),
        }),
        **{k: _PATTERNS for k in ("owned", "owned_destinations", "accounts", "read", "allowed", "allowed_destinations",
                                  "preauthorized", "deny", "denied_destinations", "opted_in_recipients")},
    }),
    "interfaces": _List(_Scalar("a modality (" + ", ".join(Modality.ALL) + ") or a tool-name pattern",
                                lambda v: isinstance(v, str) and (v in Modality.ALL or bool(_TOOL_RX.match(v))))),
    "limits": _Map({f: _num(0.0, what="a finite number >= 0 (0 = not set)") for f in LIMIT_FIELDS}),
    "measurement": _Map({
        "primary": _METRIC, "source": _SLUG, "verify_interval_s": _num(0.0), "evidence_poll_interval_s": _num(0.0),
        "window_start_ts": _num(0.0), "deadline_ts": _num(0.0), "secondary_kept_separate": _BOOL,
        "protocol": _text(TEXT_LIMITS["measurement.protocol"]),
    }),
    "data_permissions": _Map({k: _BOOL for k in DATA_PERMISSIONS}),
    "stop_conditions": _List(_enum(STOP_CONDITIONS)),
    "secondary_metrics": _List(_METRIC),
    "mode": _enum(MODES),
    # written by TaskSpec.to_dict(); a task file is always the operator's (Trust.SELF is set by the runtime only)
    "origin": _enum((Trust.OPERATOR,)),
}, top=True)


def schema_problems(raw: Any) -> List[str]:
    """Every place where the document departs from the closed schema (unknown keys with their path, wrong types,
    over-long text, malformed identifiers)."""
    out: List[str] = []
    SCHEMA.check(raw, "", out)
    return out


# ==============================================================================================================
# strategy content
# ==============================================================================================================
def objective_confirmation(text: str) -> str:
    """The value of ``objective_confirmation`` that confirms exactly this objective text."""
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# ==============================================================================================================
# normalisation before the content checks (G2R-1)
# ==============================================================================================================
#: characters that render as nothing or reorder text: zero-width, joiners, bidi controls, soft hyphen, BOM
_INVISIBLE = re.compile("[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e"
                        "\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0]")
#: separators inside identifiers, labels, metric names, tool patterns and URLs: read as spaces
_ID_SEPARATORS = re.compile(r"[_\-.:/@+=?&#%~|\\]+")


def invisible_characters(text: str) -> List[str]:
    """Code points in ``text`` that render as nothing or change the direction of text (any format character)."""
    return sorted({f"U+{ord(c):04X}" for c in str(text)
                   if _INVISIBLE.match(c) or unicodedata.category(c) == "Cf"})


def normalise_for_scan(text: str) -> str:
    """What a reader sees: NFKC (full-width / compatibility forms folded), invisible and bidi-control characters
    removed, identifier separators (``_ - . : / @ + = ? & # % ~ |``) and camelCase boundaries turned into spaces."""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(c for c in t if not (_INVISIBLE.match(c) or unicodedata.category(c) == "Cf"))
    t = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", t)
    t = _ID_SEPARATORS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


#: G2R-1b: Unicode TR39-style confusables — letters of other scripts (Cyrillic, Greek, Armenian, Latin extensions)
#: that render like Latin letters, mapped to the Latin letter before the content checks
_CONFUSABLES = str.maketrans({
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "һ": "h", "і": "i", "ї": "i", "ј": "j", "к": "k",
    "ӏ": "l", "м": "m", "н": "h", "о": "o", "р": "p", "ԛ": "q", "г": "r", "ѕ": "s", "т": "t", "ц": "u", "ѵ": "v",
    "ԝ": "w", "х": "x", "у": "y", "ү": "y", "з": "3", "ь": "b", "п": "n", "ɡ": "g", "ı": "i", "ȷ": "j", "ℓ": "l",
    "ꞁ": "l", "ɑ": "a", "ɩ": "i", "ʋ": "u", "ɒ": "a", "ο": "o", "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i",
    "κ": "k", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "γ": "y", "ω": "w", "ϲ": "c", "ϳ": "j", "օ": "o",
    "ս": "u", "ո": "n", "հ": "h", "ց": "g", "զ": "q", "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I",
    "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X", "У": "Y", "Ԁ": "D", "Ԛ": "Q",
    "Ԝ": "W", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ϲ": "C", "Ꭺ": "A", "Ᏼ": "B", "Ꮯ": "C", "Ꭼ": "E", "Ꮋ": "H", "Ꭻ": "J",
    "Ꮶ": "K", "Ꮇ": "M", "Ꮲ": "P", "Ꮪ": "S", "Ꭲ": "T", "Ꮃ": "W",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
                       "|": "l", "€": "e"})
_LEET_TOKEN = re.compile(r"[A-Za-z0-9@$!|€]*[A-Za-z][A-Za-z0-9@$!|€]*")
_SPELLED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-]){2,}[A-Za-z](?![A-Za-z])")
#: words that occur in concatenated identifiers and labels ("postmemesdaily"): used only to split such a token into
#: words before the content checks (nothing here is shown to the learner)
_SEGMENT_WORDS = frozenset("""
a about ad ads advertise advertising all and announce any app art artists ask at audience back be best boost bot bots
build bump bumping campaign campaigns channel channels chat clip clips cold collab community contest contests content
create cross crosspost daily day days developer developers discord dm dms each email engage event events every
everyone fan fans feed first for forum forums free friend friends from fun game gamer gamers games get giveaway
giveaways group grow growth guide hack help host hourly in influencer influencers invite invites inviting it join
joiner joiners joining link links list listing make many marketing mass meme memes member members message messages
monthly more music new news newsletter niche nitro of on or our outreach partner partners people personally plan
play player players post posting posts prize prizes promo promote promotion recruit referral reddit reward rewards
role roles run running send server servers share sharing site sites social spam start strategy stream streams
student students subreddit sunday team the their them then thread threads through tiktok tips to top topic twitch
twitter up users via video videos viral weekly weekend welcome with youtube your
""".split())


def _segment(token: str) -> str:
    """A lowercase run split into known words (dynamic programming: the most characters covered); characters not
    covered stay together as their own word. The token itself unless two or more known words of three or more
    letters cover at least half of it."""
    n = len(token)
    best: List[Optional[Tuple[int, List[str]]]] = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(n):
        if best[i] is None:
            continue
        cov, words = best[i]
        for j in range(min(n, i + 14), i, -1):
            w = token[i:j]
            if w in _SEGMENT_WORDS and (best[j] is None or best[j][0] < cov + len(w)):
                best[j] = (cov + len(w), words + [w])
        if best[i + 1] is None or best[i + 1][0] < cov:      # one character not covered by a known word
            best[i + 1] = (cov, words + [token[i]])
    cov, words = best[n] or (0, [token])
    known = [w for w in words if w in _SEGMENT_WORDS and len(w) >= 3]
    if len(known) < 2 or cov < 0.5 * n:
        return token
    out, run = [], ""
    for w in words:                                        # uncovered characters stay together
        if w in _SEGMENT_WORDS:
            if run:
                out.append(run)
                run = ""
            out.append(w)
        else:
            run += w
    return " ".join(out + ([run] if run else []))


def skeleton(text: str) -> str:
    """The text as it reads to a person (G2R-1b): NFKC, invisible characters removed, confusable letters of other
    scripts mapped to Latin."""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(c for c in t if not (_INVISIBLE.match(c) or unicodedata.category(c) == "Cf"))
    return t.translate(_CONFUSABLES)


def _scan_variants(text: str, *, ident: bool) -> List[str]:
    """Readings of ``text`` the content rules are applied to besides the text itself (G2R-1, G2R-1b): the
    confusable skeleton, leetspeak folded (tokens mixing letters with 0 1 3 4 5 7 @ $), letters spelled one by one
    joined, and — for identifiers — separators as spaces and concatenated words split."""
    sk = skeleton(text)
    if ident:
        sk = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", sk)
        sk = re.sub(r"\s+", " ", _ID_SEPARATORS.sub(" ", sk)).strip()
    leet = _LEET_TOKEN.sub(lambda m: m.group(0).translate(_LEET) if re.search(r"[0-9@$!|€]", m.group(0))
                           else m.group(0), sk)
    joined = _SPELLED.sub(lambda m: re.sub(r"[ .\-]", "", m.group(0)), leet)
    out = [sk, leet, joined]
    if ident:
        out.append(" ".join(_segment(w) if w.isalpha() and len(w) >= 7 else w
                            for w in joined.lower().split(" ")))
    return [v for i, v in enumerate(out) if v and v != text and v not in out[:i]]


def mixed_script_words(text: str) -> List[str]:
    """Words that mix Latin letters with letters of another script (homoglyph spellings, G2R-1b)."""
    out = []
    for w in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", str(text))):
        scripts = {("latin" if "LATIN" in unicodedata.name(c, "") else unicodedata.name(c, "?").split(" ")[0])
                   for c in w}
        if "latin" in scripts and len(scripts) > 1:
            out.append(w)
    return sorted(set(out))


_IMPERATIVE_RULES = [(l, r) for l, r in _OBJECTIVE_TEXT if l == "imperative steps"] + _INSTRUCTION_TEXT
_AUDIENCE_RULE = [(l, r) for l, r in _OBJECTIVE_TEXT if l == "audience or topic of the community"]


def _content_hits(value: str, *, ident: bool) -> List[str]:
    """The content rules on the text as written and as normalised. ``ident``: a value without spaces (identifier,
    label, metric, tool pattern, URL): read as words, it is also checked for commands when it has three or more."""
    hits = _text_hits(value, _REJECTED_TEXT)
    norm = normalise_for_scan(value)
    readings = ([norm] if norm and norm != value else []) + [v for v in _scan_variants(value, ident=ident) if v != norm]
    for r in readings:
        hits += [h for h in _text_hits(r, _REJECTED_TEXT) if h not in hits]
        if ident and len(re.findall(r"[A-Za-z]{2,}", r)) >= 3:
            hits += [h for h in _text_hits(r, _IMPERATIVE_RULES + _AUDIENCE_RULE) if h not in hits]
    if ident:
        # each separated part of an identifier ("community/<part>/<part>"), read as words (concatenations split),
        # is checked for commands and audiences as well (G2R-1b)
        for part in _ID_SEPARATORS.split(skeleton(value)):
            if len(part) < 7:
                continue
            for r in _scan_variants(part, ident=True)[-1:]:
                if len(re.findall(r"[A-Za-z]{2,}", r)) >= 3:
                    hits += [h for h in _text_hits(r, _IMPERATIVE_RULES + _AUDIENCE_RULE) if h not in hits]
    mixed = mixed_script_words(value)
    if mixed:
        hits.append(f"letters of another script inside a word (homoglyphs): {', '.join(mixed[:5])}")
    return hits


def _text_hits(text: str, rules: List[Tuple[str, "re.Pattern[str]"]]) -> List[str]:
    out = []
    for label, rx in rules:
        m = rx.search(text)
        if m:
            out.append(f"{label}: {m.group(0).strip()!r}")
    return out


def objective_flags(text: str) -> List[str]:
    """Strategy-like content in an objective text (which CAM receives verbatim). [] = nothing flagged. The text is
    also checked as a reader sees it (NFKC, invisible / bidi-control characters removed); such characters are
    themselves flagged (G2R-1)."""
    if not isinstance(text, str) or not text.strip():
        return []
    out = _text_hits(text, _REJECTED_TEXT) + _text_hits(text.strip(), _OBJECTIVE_TEXT)
    hidden = invisible_characters(text)
    for seen in _scan_variants(text, ident=False):         # NFKC, confusables, leetspeak, spelled-out letters
        out += [h for h in _text_hits(seen, _REJECTED_TEXT) + _text_hits(seen.strip(), _OBJECTIVE_TEXT) if h not in out]
    if hidden:
        out.append(f"invisible or direction-control characters: {', '.join(hidden)}")
    mixed = mixed_script_words(text)
    if mixed:
        out.append(f"letters of another script inside a word (homoglyphs): {', '.join(mixed[:5])}")
    return out


def _confirmed(raw: Dict[str, Any]) -> bool:
    c = raw.get("objective_confirmation") if isinstance(raw, dict) else None
    return isinstance(c, str) and c == objective_confirmation(raw.get("objective") or "")


def strategy_findings(raw: Any) -> List[str]:
    """Every place in a task document that carries (or looks like) a solution strategy. [] = clean.

    Keys are matched against :data:`REJECTED_KEYS` (whole key and key tokens, camelCase split); every mapping key and
    every string value is scanned with :data:`REJECTED_TEXT`; the verifier-only texts are also scanned for commands;
    the objective text is scanned with :func:`objective_flags` and reported unless the operator confirmed it."""
    out: List[str] = []
    rejected = {k.lower() for k in REJECTED_KEYS}
    for path, key, value in _walk(raw):
        k = key.lower()
        if k and k not in _SCHEMA_SAFE_KEYS:
            split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
            tokens = set(re.split(r"[^a-z0-9]+", split)) | set(re.split(r"[^a-z0-9]+", k)) | {k, re.sub(r"[^a-z0-9]", "", k)}
            hit = sorted(tokens & rejected)
            if hit:
                out.append(f"{path}: key {key!r} is not allowed in a task file ({', '.join(hit)})")
            for h in _content_hits(key, ident=True):
                out.append(f"{path}: key {key!r} reads as a solution strategy ({h})")
            for w in sorted(set(re.split(r"\s+", normalise_for_scan(key).lower())) & rejected - {k}):
                if k not in _SCHEMA_SAFE_KEYS and w not in hit:
                    out.append(f"{path}: key {key!r} is not allowed in a task file ({w})")
        if not isinstance(value, str) or is_unset(value):
            continue
        if path == "objective" and isinstance(raw, dict):
            flags = objective_flags(value)
            if flags and not _confirmed(raw):
                out.append(f"objective: text reads as a solution strategy ({'; '.join(flags)}); CAM receives the "
                           "objective verbatim, so it is refused unless the operator confirms this exact text with "
                           f"objective_confirmation: {objective_confirmation(value)}")
            continue
        for h in _content_hits(value, ident=not re.search(r"\s", value.strip())):
            out.append(f"{path}: text reads as a solution strategy ({h})")
        if path in _FREE_TEXT_PATHS:
            for h in _text_hits(value.strip(), _INSTRUCTION_TEXT):
                out.append(f"{path}: text reads as a solution strategy ({h})")
    return out


def unset_paths(raw: Any) -> List[str]:
    """Dotted paths of every value still carrying the ``REQUIRED`` marker (or left empty)."""
    return [p for p, _k, v in _walk(raw) if not isinstance(v, (dict, list)) and is_unset(v)]


# ==============================================================================================================
# Discord resources and the evidence source
# ==============================================================================================================
def _resources_of(task_or_raw: Union[TaskSpec, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(task_or_raw, TaskSpec):
        r = task_or_raw.resources
    elif isinstance(task_or_raw, dict):
        r = task_or_raw.get("resources")
    else:
        r = None
    return r if isinstance(r, dict) else {}


def _success_of(task_or_raw: Union[TaskSpec, Dict[str, Any]]) -> Dict[str, Any]:
    s = task_or_raw.success if isinstance(task_or_raw, TaskSpec) else (
        task_or_raw.get("success") if isinstance(task_or_raw, dict) else None)
    return s if isinstance(s, dict) else {}


def has_discord(task_or_raw: Union[TaskSpec, Dict[str, Any]]) -> bool:
    """True when the task authorizes Discord resources (``resources.discord`` or ``resources.discord_guild_id``)."""
    r = _resources_of(task_or_raw)
    return isinstance(r.get("discord"), dict) or "discord_guild_id" in r


def evidence_source(task_or_raw: Union[TaskSpec, Dict[str, Any]]) -> str:
    """The verifier's evidence source: ``success.evidence_source`` when it is set; otherwise
    :data:`DEFAULT_DISCORD_EVIDENCE_SOURCE` for a task with Discord resources; otherwise ''."""
    v = _success_of(task_or_raw).get("evidence_source")
    if isinstance(v, str) and not is_unset(v):
        return v.strip()
    return DEFAULT_DISCORD_EVIDENCE_SOURCE if has_discord(task_or_raw) else ""


# ==============================================================================================================
# requirements for a run
# ==============================================================================================================
def _limit_problem(v: Any) -> str:
    """Why a limit value cannot be enforced ('' = acceptable: a finite number >= 0)."""
    if isinstance(v, bool):
        return "a boolean is not a limit"
    if not isinstance(v, (int, float)):
        return f"not a number ({_show(v)})"
    if not math.isfinite(float(v)):
        return f"not a finite number ({v!r}); an infinite or NaN limit is never enforced"
    if v < 0:
        return f"negative ({v!r})"
    return ""


def limit_problems(task: TaskSpec) -> Dict[str, str]:
    """Limits of a TaskSpec (possibly built in code) whose value cannot be enforced: field -> reason."""
    lim = task.limits
    out: Dict[str, str] = {}
    for f in LIMIT_FIELDS:
        why = _limit_problem(getattr(lim, f, None))
        if why:
            out[f] = why
    return out


def live_requirements(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """Everything a live run needs from the task file that is absent or unset: [{kind, item, detail}]."""
    miss: List[Dict[str, str]] = []

    def need(kind: str, item: str, detail: str) -> None:
        miss.append({"kind": kind, "item": item, "detail": detail})

    limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    for f in LIMIT_FIELDS:
        v = limits.get(f)
        if is_unset(v) or _limit_problem(v) or v <= 0:
            need("limit", f"limits.{f}", "must be an explicit positive finite number; resources are never assumed "
                                         "unlimited")
    res = raw.get("resources") if isinstance(raw.get("resources"), dict) else {}
    tools = res.get("authorized_tools")
    if is_unset(tools) or not isinstance(tools, list) or not tools or any(is_unset(t) for t in tools):
        need("authorization", "resources.authorized_tools", "explicit list of authorized tools / tool families")
    pub = res.get("publishing")
    if not isinstance(pub, dict) or not isinstance(pub.get("authorized"), bool):
        need("authorization", "resources.publishing.authorized", "publishing authorization must be stated (true or false)")
    elif pub.get("authorized") and (is_unset(pub.get("destinations")) or not pub.get("destinations")):
        need("authorization", "resources.publishing.destinations", "authorized publishing destinations")
    perms = raw.get("data_permissions") if isinstance(raw.get("data_permissions"), dict) else {}
    for k in REQUIRED_DATA_PERMISSIONS:
        if not isinstance(perms.get(k), bool):
            need("data_permission", f"data_permissions.{k}", "must be stated explicitly (true or false)")
    stops = raw.get("stop_conditions")
    if not isinstance(stops, list) or not stops:
        need("stop_condition", "stop_conditions", "explicit list of stop conditions")
    if not isinstance(raw.get("interfaces"), list) or not raw.get("interfaces"):
        need("authorization", "interfaces", "permitted environment interfaces")
    success = raw.get("success") if isinstance(raw.get("success"), dict) else {}
    if not success:
        need("verification", "success", "independent success criteria (verifier configuration)")
    elif not evidence_source(raw):
        need("verification", "success.evidence_source", "the verifier's live evidence source ('<module>:<factory>'); "
             "tasks with Discord resources default to " + DEFAULT_DISCORD_EVIDENCE_SOURCE)
    if not (raw.get("measurement") or {}):
        need("verification", "measurement", "outcome measurement protocol")
    seen = {m["item"] for m in miss}
    for p in unset_paths(raw):
        if p not in seen and not any(p.startswith(s + "[") or p.startswith(s + ".") for s in seen):
            kind = ("credential" if "credential" in p else "limit" if p.startswith("limits.") else
                    "data_permission" if p.startswith("data_permissions.") else "resource")
            need(kind, p, "unset (marked REQUIRED in the task file)")
    return miss


def structural_problems(raw: Any) -> List[str]:
    """Closed-schema check plus the required top-level items."""
    if not isinstance(raw, dict):
        return ["task file must be a mapping"]
    out = []
    for p in schema_problems(raw):
        if p.startswith("mode:"):
            p = f"mode must be one of {MODES} (got {_show(raw.get('mode'))})"
        elif p.startswith("limits.") and ": unknown key " in p:
            p = f"unknown limit {p.split(': unknown key ', 1)[1].split(' ', 1)[0]} (known: {', '.join(LIMIT_FIELDS)})"
        elif p.startswith("stop_conditions[") and ": must be one of" in p:
            got = p.rsplit("(got ", 1)[-1].rstrip(")")
            p = f"unknown stop condition {got} (known: {', '.join(STOP_CONDITIONS)})"
        out.append(p)
    for path, k, v in _walk(raw):
        if isinstance(v, str) and v.strip().lower() in PLACEHOLDERS:
            out.append(f"{path}: {v!r} looks like a placeholder; write REQUIRED (or the real value)")
        # G2R-1: identifiers, labels and keys never contain characters a reader cannot see
        if k and invisible_characters(k):
            out.append(f"{path}: key contains invisible or direction-control characters "
                       f"({', '.join(invisible_characters(k))})")
        if isinstance(v, str) and path not in _FREE_TEXT_PATHS and invisible_characters(v):
            out.append(f"{path}: contains invisible or direction-control characters "
                       f"({', '.join(invisible_characters(v))})")
    if is_unset(raw.get("objective_id")):
        out.append("objective_id is required")
    if raw.get("objective") is not None and not isinstance(raw.get("objective"), str):
        out.append("objective must be text")
    return out


def run_problems(raw: Dict[str, Any]) -> List[str]:
    """Why this task cannot be *run* as configured (beyond structure and strategy content)."""
    mode = raw.get("mode", "staging")
    if mode == "live":
        return [f"{m['item']}: {m['detail']}" for m in live_requirements(raw)]
    out: List[str] = [f"{p}: unset (marked REQUIRED in the task file)" for p in unset_paths(raw)]
    limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    for f in ("duration_s", "max_decisions"):
        v = limits.get(f)
        if f not in limits or (not is_unset(v) and (_limit_problem(v) or v <= 0)):
            out.append(f"limits.{f}: must be an explicit positive number in every mode")
    if not str(raw.get("objective") or "").strip():
        # autonomous learning (spec 11.4): no objective, but a complete fixed budget
        for f in ("money", "compute_s", "max_external_actions"):
            v = limits.get(f)
            if is_unset(v) or _limit_problem(v):
                out.append(f"limits.{f}: autonomous learning needs every budget stated explicitly")
    elif not (raw.get("success") or {}):
        out.append("success: independent success criteria are required")
    return out


def _limit_value(v: Any) -> float:
    if is_unset(v) or isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0.0
    return float(v)


def parse_task(raw: Dict[str, Any]) -> TaskSpec:
    """Build the TaskSpec. Unset limits become 0 (= "must be set explicitly before a live run"). Invalid limit
    values are carried through unchanged in kind (NaN stays NaN) so that :func:`check_runnable` refuses them; a
    validated file never contains one."""
    lim = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    perms = {k: bool(v) for k, v in (raw.get("data_permissions") or {}).items() if isinstance(v, bool)} \
        if isinstance(raw.get("data_permissions"), dict) else {}

    def as_int(v: Any) -> Any:
        x = _limit_value(v)
        return int(x) if math.isfinite(x) else x

    return TaskSpec(
        objective_id=str(raw.get("objective_id", "")), objective=str(raw.get("objective") or ""),
        success=copy.deepcopy(dict(raw.get("success") or {})), family=str(raw.get("family") or ""),
        resources=copy.deepcopy(dict(raw.get("resources") or {})),
        interfaces=[str(x) for x in raw.get("interfaces") or []],
        limits=Limits(duration_s=_limit_value(lim.get("duration_s")), money=_limit_value(lim.get("money")),
                      compute_s=_limit_value(lim.get("compute_s")),
                      max_external_actions=as_int(lim.get("max_external_actions")),
                      max_decisions=as_int(lim.get("max_decisions"))),
        measurement=copy.deepcopy(dict(raw.get("measurement") or {})), data_permissions=perms,
        stop_conditions=[str(s) for s in raw.get("stop_conditions") or []],
        secondary_metrics=[str(s) for s in raw.get("secondary_metrics") or []], mode=str(raw.get("mode", "staging")))


# ==============================================================================================================
# YAML loading
# ==============================================================================================================
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses merge keys and duplicate keys (anchors and aliases are refused before composing)."""

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        for key_node, _value in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise yaml.constructor.ConstructorError(
                    None, None, "YAML merge keys ('<<') are not allowed in task files", key_node.start_mark)
        super().flatten_mapping(node)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
        self.flatten_mapping(node)
        seen = set()
        for key_node, _value in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                dup = key in seen
            except TypeError:
                dup = False
            if dup:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r} (the later value would silently replace the earlier one)",
                    key_node.start_mark)
            if not isinstance(key, (list, dict)):
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


_StrictLoader.add_constructor("tag:yaml.org,2002:map", _StrictLoader.construct_yaml_map)


def _yaml_problems(text: str) -> List[str]:
    out: List[str] = []
    try:
        for ev in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(ev, yaml.AliasEvent):
                out.append(f"line {ev.start_mark.line + 1}: YAML alias '*{ev.anchor}' is not allowed in task files "
                           "(every value must be written out where it applies)")
            elif getattr(ev, "anchor", None):
                out.append(f"line {ev.start_mark.line + 1}: YAML anchor '&{ev.anchor}' is not allowed in task files")
    except yaml.YAMLError as e:
        out.append(f"not valid YAML: {e}")
    return out


def load_task_text(text: str, path: str = "") -> Dict[str, Any]:
    problems = _yaml_problems(text)
    if problems:
        raise TaskValidationError(problems, path)
    try:
        raw = yaml.load(text, Loader=_StrictLoader)       # noqa: S506 — a SafeLoader subclass
    except yaml.YAMLError as e:
        raise TaskValidationError([f"not a valid task file: {e}"], path)
    if not isinstance(raw, dict):
        raise TaskValidationError(["task file must be a mapping"], path)
    return raw


def load_task_raw(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return load_task_text(text, path)


def validate_task(raw: Dict[str, Any], *, for_run: bool = True) -> List[str]:
    problems = structural_problems(raw)
    problems += strategy_findings(raw)
    if for_run and not problems:
        problems += run_problems(raw)
    return problems


def load_task(path: str, *, for_run: bool = True) -> TaskSpec:
    """Load and validate a task file. ``for_run=False`` allows unset items (inspection / preflight) but still
    rejects strategy content and malformed files."""
    raw = load_task_raw(path)
    problems = validate_task(raw, for_run=for_run)
    if problems:
        raise TaskValidationError(problems, path)
    return parse_task(raw)


#: largest task file read for an assignment (a task file is a few kilobytes)
MAX_TASK_FILE_BYTES = 1 << 20


class TaskFileChanged(TaskValidationError):
    """The task file's bytes are not the ones the operator's command names (sha256)."""


def read_task_file(path: str) -> bytes:
    """The task file's bytes, read once (G2R-18: what is hashed is what is parsed)."""
    with open(path, "rb") as f:
        data = f.read(MAX_TASK_FILE_BYTES + 1)
    if len(data) > MAX_TASK_FILE_BYTES:
        raise TaskValidationError([f"task file larger than {MAX_TASK_FILE_BYTES} bytes"], path)
    return data


def load_task_bytes(data: bytes, *, sha256: str = "", for_run: bool = True, path: str = "") -> TaskSpec:
    """Validate and parse a task file from its bytes; with ``sha256`` the bytes must have exactly that hash."""
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise TaskFileChanged(["the task file differs from the one the operator's command names"], path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise TaskValidationError([f"task file is not UTF-8: {e}"], path)
    raw = load_task_text(text, path)
    problems = validate_task(raw, for_run=for_run)
    if problems:
        raise TaskValidationError(problems, path)
    return parse_task(raw)


def task_from_dict(raw: Dict[str, Any], *, for_run: bool = True) -> TaskSpec:
    problems = validate_task(raw, for_run=for_run)
    if problems:
        raise TaskValidationError(problems)
    return parse_task(raw)


def check_runnable(task: TaskSpec) -> List[str]:
    """Runtime-side re-check of a TaskSpec object (a TaskSpec may be built in code, not only from YAML)."""
    out: List[str] = []
    bad = limit_problems(task)
    for f, why in bad.items():
        out.append(f"limits.{f} is invalid: {why}")
    lim = task.limits

    def unset(f: str) -> bool:
        return f in bad or getattr(lim, f) <= 0

    if task.mode == "live":
        for f in LIMIT_FIELDS:
            if f not in bad and getattr(lim, f) <= 0:
                out.append(f"limits.{f} is not set; a live run needs every limit")
        if not task.stop_conditions:
            out.append("stop_conditions are not set")
        if not task.interfaces:
            out.append("interfaces are not set")
        if unset_paths({"resources": task.resources, "success": task.success, "measurement": task.measurement}):
            out.append("unset items: " + ", ".join(unset_paths({"resources": task.resources, "success": task.success,
                                                               "measurement": task.measurement})))
    elif unset("duration_s") and unset("max_decisions"):
        out.append("neither limits.duration_s nor limits.max_decisions is set; unbounded runs are refused")
    if not task.objective.strip():
        if unset("duration_s") or unset("max_decisions"):
            out.append("autonomous learning (no objective) needs explicit duration and decision budgets")
    for k in tuple(task.success) + tuple(task.resources) + tuple(task.measurement):
        if str(k).lower() in {x.lower() for x in TaskSpec.FORBIDDEN_KEYS}:
            out.append(f"forbidden key {k!r}")
    return out


# ==============================================================================================================
# what the learner receives
# ==============================================================================================================
_LEARNER_RESOURCES = ("authorized_tools", "authorized_destinations", "currency", "owned_site", "staging_site",
                      "publishing", "discord_guild_id", "owned", "owned_destinations", "accounts", "read", "allowed",
                      "allowed_destinations", "preauthorized", "deny", "denied_destinations", "opted_in_recipients")
_LEARNER_DISCORD = ("guild_id", "channels", "metric_channel_id", "application_id")


def learner_view(task: TaskSpec) -> TaskSpec:
    """The TaskSpec the learner may receive (``agent.set_objective``): objective text, limits, interfaces, data
    permissions, stop conditions, secondary metric names, the objective metric and target, and the resources it acts
    on. Withheld: success definition, exclusion rules, evidence source and verifier settings, the measurement
    protocol, credential names, test-account identifiers and the operator authorization record. Pure (deep copies)."""
    res_in = task.resources if isinstance(task.resources, dict) else {}
    res: Dict[str, Any] = {k: copy.deepcopy(res_in[k]) for k in _LEARNER_RESOURCES if k in res_in}
    if isinstance(res_in.get("owned_sites"), list):
        res["owned_sites"] = [{k: copy.deepcopy(s[k]) for k in ("name", "base_url") if k in s}
                              for s in res_in["owned_sites"] if isinstance(s, dict)]
    if isinstance(res_in.get("discord"), dict):
        res["discord"] = {k: copy.deepcopy(res_in["discord"][k]) for k in _LEARNER_DISCORD if k in res_in["discord"]}
    succ_in = task.success if isinstance(task.success, dict) else {}
    success = {k: copy.deepcopy(succ_in[k]) for k in ("metric", "target") if k in succ_in}
    return TaskSpec(objective_id=task.objective_id, objective=task.objective, success=success, family=task.family,
                    resources=res, interfaces=list(task.interfaces), limits=copy.deepcopy(task.limits),
                    measurement={}, data_permissions=dict(task.data_permissions),
                    stop_conditions=list(task.stop_conditions), secondary_metrics=list(task.secondary_metrics),
                    mode=task.mode, **({"origin": task.origin} if hasattr(task, "origin") else {}))


def autonomous_task(objective_id: str, limits: Limits, *, interfaces: Optional[List[str]] = None,
                    mode: str = "staging", family: str = "autonomous") -> TaskSpec:
    """Task for the optional learning mode without an assigned objective (spec 11.4): a fixed budget and nothing
    else. Self-chosen learning goals stay inside the learner; usefulness is judged on later assigned objectives."""
    t = TaskSpec(objective_id=objective_id, objective="", family=family, interfaces=list(interfaces or []),
                 limits=limits, mode=mode, stop_conditions=["duration_exhausted", "decisions_exhausted"])
    problems = check_runnable(t)
    if problems:
        raise TaskValidationError(problems)
    return t


def scan_file(path: str) -> List[str]:
    """Mechanical strategy scan of one task file (used by tests and ``cam-live``)."""
    return strategy_findings(load_task_raw(path))


def default_tasks_dir() -> str:
    from ..config import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "tasks")


# ==============================================================================================================
# staging sites the operator's run registers as local services (change request HX-5)
# ==============================================================================================================
def staging_local_services(task: TaskSpec) -> List[str]:
    """``host:port`` of every staging / owned site of a *staging* task that is an address of this machine
    (``resources.staging_site``, ``resources.owned_site``, ``resources.owned_sites[].base_url``). The authorization
    service refuses such addresses (P-LOCAL-ADDRESS) unless the operator registers them; ``cam-live run`` does that
    for the operator before the learner starts. Always empty outside staging mode: a live task never opens a
    loopback address to the learner."""
    from urllib.parse import urlsplit
    from ..harness.auth import destinations as D
    from ..harness.auth import localnet
    if task.mode != "staging":
        return []
    res = task.resources if isinstance(task.resources, dict) else {}
    urls: List[str] = [res.get("staging_site"), res.get("owned_site")]
    urls += [s.get("base_url") for s in (res.get("owned_sites") or []) if isinstance(s, dict)]
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str) or not u.strip() or is_unset(u):
            continue
        try:
            p = urlsplit(u if "://" in u else f"http://{u}")
            host, port = p.hostname or "", p.port or {"http": 80, "https": 443}.get(p.scheme, 0)
        except ValueError:
            continue
        try:
            canon = D.parse_pattern(f"{host}:{port}").host if host and port else ""
        except Exception:  # noqa: BLE001 — not a valid host: nothing to register
            canon = ""
        if canon and localnet.local_kind(canon):
            addr = f"[{canon}]:{port}" if ":" in canon else f"{canon}:{port}"
            if addr not in out:
                out.append(addr)
    return out


def register_staging_sites(policy: Any, task: TaskSpec, by: str = "operator") -> List[Dict[str, Any]]:
    """Register :func:`staging_local_services` at the authorization service (the operator's
    ``opctl local-service add``). Returns what was registered (empty outside staging mode)."""
    done = []
    for addr in staging_local_services(task):
        done.append(policy.register_local_service(addr, by, f"staging site of task {task.objective_id}"))
    return done
