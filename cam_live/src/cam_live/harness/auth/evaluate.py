"""Stateless rule evaluation of one action node against the policy and the operator-pinned task.

Inputs are only: the typed ``ActionSpec`` (or the ``PrimitiveOp`` turned into one node), the compiled policy, the
task context built from the pinned ``TaskSpec`` / live authorization record, the secret broker (to recognise secret
values) and the browser location tracked from executor reports. No observation text is ever looked at.

Every finding carries a policy id and a reason. ``deny`` findings deny; ``approve`` findings need the operator.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
import zlib
from urllib.parse import parse_qsl
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ... import contracts as C
from . import destinations as D
from . import localnet
from . import platform_constraints
from .schema import EFFECT_RANK, ModeRule, Policy, RateRule, ToolRule, norm_field

_OP_RE = re.compile(r"^[A-Za-z0-9_.\-:]{1,128}$")
_READ_METHODS = ("GET", "HEAD", "OPTIONS")
#: a tunnel (HTTPS / WebSocket CONNECT) is judged by host and port only, as a read: what travels through it is
#: judged per request by the layer that sees the requests (E->EA-1)
_TUNNEL_METHODS = ("CONNECT",)
#: findings that say "this target is not permitted" (as opposed to "this content must change" or "not now"):
#: these are remembered so the same target cannot be reached through another modality or spelling
INTENT_PIDS = ("P-DEST-DENIED", "P-DEST-NOT-ALLOWED", "P-MODE-ENVELOPE", "P-WRITE-NON-OWNED", "P-INTERACT-DENIED",
               "P-FILE-ROOT", "P-LOCAL-ADDRESS")
#: of those, the ones that deny only *writing* to the target (reading it may still be allowed)
WRITE_ONLY_PIDS = ("P-WRITE-NON-OWNED", "P-INTERACT-DENIED")


@dataclass
class TaskCtx:
    task: C.TaskSpec
    task_hash: str
    mode: ModeRule
    started_at: float
    owned: List[D.Pattern]
    read: List[D.Pattern]
    deny: List[D.Pattern]
    preauth: List[D.Pattern]
    interactive: List[D.Pattern]
    envelope: Optional[List[D.Pattern]]
    opted_in: Set[str]
    modalities: Set[str]
    op_patterns: List[str]
    limits: Dict[str, float]
    strict_unset: bool
    window: Optional[Tuple[float, float]] = None
    record_id: str = ""
    file_base: str = ""
    local_services: Set[Tuple[str, int]] = field(default_factory=set)   # registered (canonical host, port)
    own_process: bool = False            # loopback ports served only by this process count as registered
    rates: Dict[str, RateRule] = field(default_factory=dict)            # effective write / write_global rules

    @property
    def task_id(self) -> str:
        return self.task.objective_id


@dataclass
class Finding:
    pid: str
    reason: str
    dest: Optional[D.Destination] = None
    cls: str = ""                # approval class: write | input | content | spend | network


@dataclass
class NodeEval:
    node: C.ActionSpec
    tool: Optional[ToolRule] = None
    effect: str = "none"         # resolved side-effect class of this node
    potential: str = "none"      # strongest effect the node could have on its destinations
    dests: List[D.Destination] = field(default_factory=list)
    deny: List[Finding] = field(default_factory=list)
    approve: List[Finding] = field(default_factory=list)
    effects: List[Tuple[str, D.Destination]] = field(default_factory=list)   # ("read"|"write"|"input", dest)
    money: float = 0.0
    content: str = ""
    blob: str = ""               # lower-cased text of every string in the node (opaque tools only)
    disclose: bool = False       # the destination's platform requires the configured disclosure
    method: str = ""             # HTTP method of a by_method tool ("" otherwise): part of the operation class
    location_used: str = ""      # destination taken from the tracked browser location (approval binds to it)


def truthy(v: Any) -> bool:
    return v not in (None, "", False, 0, 0.0) and v != [] and v != {}


#: values of a network flag that ask for *no* network (S-3: ``sandbox.python {"network": "none"}``, the first value of
#: the tool's own enum, is the most restrictive request and must not count as a network request)
NO_NETWORK = ("none", "no", "off", "false", "disabled", "deny", "denied", "0")


def network_requested(v: Any) -> bool:
    return truthy(v) and not (isinstance(v, str) and v.strip().lower() in NO_NETWORK)


def walk_keys(obj: Any, depth: int = 0) -> Iterator[Tuple[str, Any]]:
    """(normalised key, value) for every mapping key at any depth."""
    if depth > 16:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield norm_field(k), v
            yield from walk_keys(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_keys(v, depth + 1)


_FORM_RE = re.compile(r"^[\w.\-\[\]%]+=[^&=]*(&[\w.\-\[\]%]+=[^&=]*)*$")


def expand(obj: Any, depth: int = 0) -> Any:
    """``obj`` with string values that are JSON objects / arrays or form-encoded bodies replaced by what they
    encode, so that field rules see keys inside a request body sent as a string."""
    if depth > 8:
        return obj
    if isinstance(obj, dict):
        return {k: expand(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [expand(v, depth + 1) for v in obj]
    if isinstance(obj, str) and 2 <= len(obj) <= 262144:
        t = obj.strip()
        if t[:1] in "{[" and t[-1:] in "}]":
            try:
                return expand(json.loads(t), depth + 1)
            except ValueError:
                return obj
        if "=" in t and _FORM_RE.match(t):
            pairs = parse_qsl(t, keep_blank_values=True)
            if pairs:
                return {k: v for k, v in pairs}
    return obj


_EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[A-Za-z]{2,}$")
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")
_MENTION_RE = re.compile(r"^<@[!&]?\d+>$")
_HANDLE_RE = re.compile(r"^@[A-Za-z0-9_][A-Za-z0-9_.]{1,31}$")
_SNOWFLAKE_RE = re.compile(r"^\d{15,21}$")
_PERSON_KEY_RE = re.compile(r"(user|member|rcpt|recipient|people|person|contact|target|audience|follower|friend|"
                            r"invitee|subscriber|dm|to$|^to|cc$|bcc|customer|lead|account)")


def _recipient_shaped(v: str, person_key: bool) -> bool:
    v = v.strip()
    if not v or len(v) > 254:
        return False
    if _EMAIL_RE.match(v) or _MENTION_RE.match(v) or _HANDLE_RE.match(v) or v.startswith("account:"):
        return True
    if _PHONE_RE.match(v) and sum(c.isdigit() for c in v) >= 7:
        return True
    return person_key and bool(_SNOWFLAKE_RE.match(v))


def recipient_lists(obj: Any, depth: int = 0, key: str = "") -> Iterator[Tuple[str, int]]:
    """(key, number of recipient-shaped values) for every list (or separator-joined string under a person-like
    key) at any depth of an (expanded) argument tree."""
    if depth > 16:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from recipient_lists(v, depth + 1, norm_field(k))
    elif isinstance(obj, (list, tuple)):
        person = bool(_PERSON_KEY_RE.search(key))
        n = sum(1 for x in obj if isinstance(x, (str, int)) and not isinstance(x, bool)
                and _recipient_shaped(str(x), person))
        if n:
            yield key, n
        for x in obj:
            if isinstance(x, (dict, list, tuple)):
                yield from recipient_lists(x, depth + 1, key)
    elif isinstance(obj, str) and key and _PERSON_KEY_RE.search(key) and re.search(r"[,;\s]", obj.strip()):
        n = sum(1 for x in re.split(r"[,;\s]+", obj) if _recipient_shaped(x, True))
        if n:
            yield key, n


def ident_words(text: str) -> Set[Tuple[str, ...]]:
    """Word sequences of identifiers and string words in program text: ``per_user_traits`` / ``perUserTraits`` ->
    every contiguous run of (per, user, traits)."""
    out: Set[Tuple[str, ...]] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,80}", text or ""):
        words = [w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", tok.replace("_", " ")) if w]
        for i in range(len(words)):
            for j in range(i + 1, min(len(words), i + 4) + 1):
                out.add(tuple(words[i:j]))
    return out


def shingles(text: str, cap: int = 600) -> List[int]:
    """Order-insensitive fingerprint: CRC of word bigrams of the normalised text."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(toks) < 2:
        grams = toks
    else:
        grams = [toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)]
    return sorted({zlib.crc32(g.encode()) for g in grams})[:cap]


def similarity(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _count_items(v: Any) -> int:
    if isinstance(v, (list, tuple, set)):
        return len(v)
    if isinstance(v, dict):
        return len(v)
    if isinstance(v, str):
        return len([x for x in re.split(r"[,;\s]+", v) if x])
    return 1 if truthy(v) else 0


def _as_strings(v: Any) -> List[str]:
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    if isinstance(v, dict):
        return [str(x) for x in list(v.keys()) + list(v.values())]
    return [str(v)] if truthy(v) else []


class Evaluator:
    def __init__(self, policy: Policy, vault: Any, sensitive_paths: Sequence[str] = ()):
        self.p, self.vault = policy, vault
        self.sensitive = [s.lower() for s in sensitive_paths if s]
        self._term_res = [re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
                          for t in policy.prohibited["vulnerable_terms"]]
        self._purpose_res = [re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
                             for t in policy.prohibited["denied_purposes"]]
        #: profiling field names as word sequences (``user_traits`` -> ("user", "traits")) for program text
        self._profiling_words = [tuple(w for w in re.split(r"[^a-z0-9]+", f.lower()) if w)
                                 for f in policy_profiling_names(policy)]

    # -- structure -------------------------------------------------------------------------------------------
    def structure(self, action: Any) -> List[str]:
        """Problems that make the action unjudgeable. [] = well-formed."""
        errs: List[str] = []
        st = self.p.structure
        count = 0

        def visit(a: Any, depth: int, where: str) -> None:
            nonlocal count
            count += 1
            if count > st["max_nodes"]:
                if count == st["max_nodes"] + 1:
                    errs.append(f"more than {st['max_nodes']} actions in one program")
                return
            if depth > st["max_depth"]:
                errs.append(f"{where}: steps nested deeper than {st['max_depth']}")
                return
            if not isinstance(a, C.ActionSpec):
                errs.append(f"{where}: not an ActionSpec")
                return
            if not isinstance(a.operation, str) or not _OP_RE.match(a.operation):
                errs.append(f"{where}: operation is not a tool name")
            if a.modality not in C.Modality.ALL:
                errs.append(f"{where}: unknown modality {a.modality!r}")
            if a.level not in C.Level.ALL:
                errs.append(f"{where}: unknown level {a.level!r}")
            if not isinstance(a.target, str) or len(a.target) > 4096:
                errs.append(f"{where}: target must be a string of at most 4096 characters")
            if not isinstance(a.action_id, str) or not a.action_id or len(a.action_id) > 200:
                errs.append(f"{where}: action_id missing")
            if not isinstance(a.arguments, dict) or any(not isinstance(k, str) for k in a.arguments):
                errs.append(f"{where}: arguments must be a mapping with string keys")
            else:
                try:
                    n = len(json.dumps(a.arguments, allow_nan=False))
                    if n > st["max_argument_bytes"]:
                        errs.append(f"{where}: arguments larger than {st['max_argument_bytes']} bytes")
                except (TypeError, ValueError):
                    errs.append(f"{where}: arguments are not plain JSON values")
            if isinstance(a.timeout_s, bool) or not isinstance(a.timeout_s, (int, float)) or not (0 <= a.timeout_s <= 86400):
                errs.append(f"{where}: timeout_s out of range")
            if not isinstance(a.est_cost, C.Cost):
                errs.append(f"{where}: est_cost is not a Cost")
            else:
                for f in ("seconds", "money", "compute_s", "primitive_ops", "tokens"):
                    v = getattr(a.est_cost, f)
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                        errs.append(f"{where}: est_cost.{f} must be a finite number >= 0")
            if not isinstance(a.steps, list):
                errs.append(f"{where}: steps must be a list")
            else:
                for i, s in enumerate(a.steps):
                    visit(s, depth + 1, f"{where}.steps[{i}]")

        visit(action, 0, "action")
        return errs[:20]

    @staticmethod
    def flatten(action: C.ActionSpec) -> List[C.ActionSpec]:
        out, stack = [], [action]
        while stack:
            a = stack.pop()
            out.append(a)
            stack.extend(reversed(a.steps))
        return out

    # -- one node --------------------------------------------------------------------------------------------
    def node(self, node: C.ActionSpec, tc: TaskCtx, location: str = "", primitive: bool = False) -> NodeEval:
        ne = NodeEval(node=node)
        p, args = self.p, node.arguments
        tool = p.tool_for(node.operation)
        ne.tool = tool
        strings = D.string_leaves(args) + [node.target]
        text_all = "\n".join(strings)
        self._credentials(ne, args, text_all)
        if tool is None:
            ne.deny.append(Finding("P-TOOL-UNKNOWN", f"operation {node.operation!r} is not in the policy's tool table"))
            for pid, why, d in platform_constraints.findings(node, [], "none", text_all, expand(args), raw_tool=False,
                                                             opaque=True):
                ne.deny.append(Finding(pid, why, d))
            return ne
        if tool.opaque or tool.effect == "by_method":
            ne.blob = text_all.lower()      # payloads whose meaning the tool table cannot see: program text, raw requests
        if node.modality not in tool.modalities:
            ne.deny.append(Finding("P-MODALITY", f"{node.operation!r} is not an operation of modality "
                                                 f"{node.modality!r} ({tool.id})"))
        if tool.modes and tc.mode.name not in tool.modes:
            ne.deny.append(Finding("P-TOOL-MODE", f"{node.operation!r} is not available in {tc.mode.name} mode"))
        if node.modality != C.Modality.INTERNAL and not (
                node.modality in tc.modalities or any(fnmatch.fnmatchcase(node.operation, pat) for pat in tc.op_patterns)):
            ne.deny.append(Finding("P-INTERFACE", f"the task does not permit {node.operation!r} "
                                                  f"(modality {node.modality!r}) among its interfaces"))

        # destinations ------------------------------------------------------------------------------------------
        context_ids: Set[int] = set()
        for rule in tool.dests:
            found: List[D.Destination] = []
            invalid = False
            for path in rule.frm:
                ok, val = D.dig(node, path)
                if not ok or not truthy(val):
                    continue
                vals = list(val) if (rule.multi and isinstance(val, (list, tuple))) else [val]
                for v in vals:
                    try:
                        found.append(D.parse_destination(v, rule.kind, rule.platform, file_base=tc.file_base))
                    except D.DestinationError as e:
                        invalid = True
                        ne.deny.append(Finding("P-DEST-INVALID", f"{path} is not a usable {rule.kind} destination: {e}"))
                if found or invalid:
                    break
            if not found and rule.location and location:
                try:
                    found.append(D.normalise_url(location))
                    ne.location_used = location
                except D.DestinationError:
                    pass
            if not found and rule.required and not invalid:
                hint = " (no page location is known: navigate first)" if rule.location else ""
                ne.deny.append(Finding("P-DEST-MISSING", f"{node.operation!r} names no {rule.kind} destination in "
                                                         f"{' / '.join(rule.frm)}{hint}"))
            if rule.context:
                context_ids.update(id(d) for d in found)
            ne.dests.extend(found)
        if tool.require_dest and not ne.dests and not ne.deny:
            ne.deny.append(Finding("P-DEST-MISSING", f"{node.operation!r} names no destination"))

        # effect ------------------------------------------------------------------------------------------------
        base = tool.effect
        if base == "by_method":
            method = str(args.get("method") or node.operation.rsplit(".", 1)[-1]).upper()
            ne.method = method if method in _READ_METHODS + _TUNNEL_METHODS + ("POST", "PUT", "PATCH", "DELETE") \
                else "WRITE"
            base = "external_read" if method in _READ_METHODS + _TUNNEL_METHODS else "external_write"
        network = bool(tool.network_flags) and any(k in tool.network_flags and network_requested(v)
                                                   for k, v in walk_keys(args))
        rank = EFFECT_RANK[base if base != "interact" else "external_read"]
        pot = rank
        if base == "external_write" or (base == "interact" and tool.interact_class == "input"):
            pot = EFFECT_RANK["external_write"]
        if tool.opaque:
            pot = EFFECT_RANK["external_write"]

        specific = [d for d in ne.dests if d.kind not in ("file", "skill", "sim") and id(d) not in context_ids]
        for d in ne.dests:
            if d.kind == "file":
                self._file(ne, d)
                continue
            if d.kind in ("skill", "sim"):
                continue
            eff = base if base in ("external_read", "external_write", "interact") else "external_read"
            if D.match_any(d, p.deny) or D.match_any(d, tc.deny):
                ne.deny.append(Finding("P-DEST-DENIED", f"{d.key} is on the deny list", d))
                continue
            lk = self.local_denial(d, tc)
            if lk:
                ne.deny.append(Finding("P-LOCAL-ADDRESS", f"{d.key} is an address of this machine or its network "
                                                          f"({lk}) and not a registered local service; the "
                                                          f"instance's own services are never reachable", d))
                continue
            if d.kind == "url" and tc.envelope is not None and D.match_any(d, tc.envelope) is None:
                ne.deny.append(Finding("P-MODE-ENVELOPE", f"{d.key} is outside what {tc.mode.name} mode may reach", d))
                continue
            owned = D.match_any(d, tc.owned) is not None
            readable = owned or D.match_any(d, tc.read) is not None
            pre = owned and D.match_any(d, tc.preauth) is not None
            interactive = D.match_any(d, p.interactive) is not None or D.match_any(d, tc.interactive) is not None
            if id(d) in context_ids and specific:
                # the place the operation acts *in* (e.g. the server of a channel): it must be one the task may act
                # on, but the effect, approval and rate accounting belong to the more specific destination
                if eff == "external_read" and not readable:
                    ne.deny.append(Finding("P-DEST-NOT-ALLOWED", f"{d.key} is not an allowed destination for this task", d))
                elif eff != "external_read" and not owned and not interactive:
                    ne.deny.append(Finding("P-WRITE-NON-OWNED", f"external write in {d.key}, which the operator does "
                                                                f"not own", d))
                continue
            self._platform(ne, d, tool, eff)
            if eff == "external_read":
                if not readable:
                    ne.deny.append(Finding("P-DEST-NOT-ALLOWED", f"{d.key} is not an allowed destination for this task", d))
                else:
                    ne.effects.append(("read", d))
                    rank = max(rank, EFFECT_RANK["external_read"])
                continue
            if owned and eff == "interact":            # clicks and typing on an owned page: an interaction
                rank = max(rank, EFFECT_RANK["external_write"])
                ne.effects.append(("input", d))
                if not pre:
                    ne.approve.append(Finding("P-INTERACT-APPROVAL", f"interaction with owned page {d.key} needs the "
                                                                     f"operator's approval (not pre-authorized)", d,
                                              "input"))
                continue
            if owned:                                   # an external write to an owned destination
                rank = max(rank, EFFECT_RANK["external_write"])
                ne.effects.append(("write", d))
                always = p.publishing_always and tool.kind == "publish"
                if not pre or always:
                    pid = "P-PUBLISH-APPROVAL" if tool.kind == "publish" else "P-WRITE-APPROVAL"
                    ne.approve.append(Finding(pid, f"external write to owned destination {d.key} needs the operator's "
                                                   f"approval (the destination class is not pre-authorized)", d, "write"))
                continue
            if eff == "external_write" and not interactive:
                ne.deny.append(Finding("P-WRITE-NON-OWNED", f"external write to {d.key}, which the operator does not own", d))
                continue
            if not readable:
                ne.deny.append(Finding("P-DEST-NOT-ALLOWED", f"{d.key} is not an allowed destination for this task", d))
                continue
            rule3 = "allow" if interactive else tc.mode.interact_non_owned[tool.interact_class]
            rank = max(rank, EFFECT_RANK["external_read"])
            if rule3 == "deny":
                ne.deny.append(Finding("P-INTERACT-DENIED", f"sending input to non-owned {d.key} is not permitted in "
                                                            f"{tc.mode.name} mode", d))
            elif rule3 == "needs_approval":
                ne.effects.append(("input", d))
                ne.approve.append(Finding("P-INTERACT-APPROVAL", f"sending input to non-owned {d.key} needs the "
                                                                 f"operator's approval", d, "input"))
            else:
                ne.effects.append(("input" if tool.interact_class == "input" else "read", d))

        if not [d for d in ne.dests if d.kind not in ("file", "skill", "sim")]:
            self._platform(ne, None, tool, base)
            if base == "external_write" and not any(f.pid.startswith("P-DEST") for f in ne.deny):
                ne.deny.append(Finding("P-DEST-MISSING", f"{node.operation!r} writes externally but names no destination"))
        if network:
            rank = max(rank, EFFECT_RANK["external_read"])
            if tc.mode.code_network == "deny":
                ne.deny.append(Finding("P-CODE-NETWORK", f"network access for generated code is not permitted in "
                                                         f"{tc.mode.name} mode"))
            elif tc.mode.code_network == "needs_approval":
                ne.approve.append(Finding("P-CODE-NETWORK", "network access for generated code needs the operator's "
                                                            "approval", None, "network"))
            if not [d for d in ne.dests if d.kind == "url"]:
                ne.deny.append(Finding("P-CODE-NETWORK", "network access requested without naming the hosts"))
        if tool.opaque:
            for d in D.urls_in_text(text_all):
                hit = D.match_any(d, p.deny) or D.match_any(d, tc.deny)
                if hit is not None:
                    ne.deny.append(Finding("P-DEST-DENIED", f"the payload refers to {d.key}, which is on the deny list", d))
        # hard platform constraints (no user-account automation, no CAPTCHA / human-verification circumvention):
        # fixed in code, not part of the operator-editable policy, so nothing can permit them
        for pid, why, d in platform_constraints.findings(
                node, ne.dests, base, text_all, expand(args), raw_tool=tool.id == "T-NET-REQUEST" or
                tool.effect == "by_method" and node.modality not in (C.Modality.STRUCTURED_BROWSER, C.Modality.VISUAL),
                opaque=bool(tool.opaque)):
            ne.deny.append(Finding(pid, why, d))
        ne.effect = [e for e, r in EFFECT_RANK.items() if r == rank][0]
        ne.potential = [e for e, r in EFFECT_RANK.items() if r == max(pot, rank)][0]
        if rank > EFFECT_RANK[tc.mode.max_effect]:
            ne.deny.append(Finding("P-MODE-EFFECT", f"{node.operation!r} has side effect {ne.effect!r}; "
                                                    f"{tc.mode.name} mode permits at most {tc.mode.max_effect!r}"))

        # money -------------------------------------------------------------------------------------------------
        money = float(node.est_cost.money)
        for path in tool.amount_from:
            ok, val = D.dig(node, path)
            if ok and isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                money = max(money, float(val))
        ne.money = max(0.0, money)
        if (ne.money > 0 or tool.kind == "payment") and not tc.mode.simulated_money:
            if ne.money > p.spending["auto_approve_below"] or tool.kind == "payment":
                ne.approve.append(Finding("P-SPEND-APPROVAL", f"spending {ne.money:g} needs the operator's approval",
                                          None, "spend"))

        # content, data use, prohibited classes -----------------------------------------------------------------
        outbound = rank >= EFFECT_RANK["external_write"] or base in ("interact", "external_write")
        if outbound or tool.kind in ("publish", "direct_message", "draft"):
            parts: List[str] = []
            if tool.content_from:
                for path in tool.content_from:
                    ok, val = D.dig(node, path)
                    if ok and truthy(val):
                        parts.extend(D.string_leaves(val) if not isinstance(val, str) else [val])
            else:
                dest_raw = {d.raw for d in ne.dests}
                parts = [s for s in _values_only(args) if s not in dest_raw]
            ne.content = "\n".join(parts)
        self._data_use(ne, tool, tc, args)
        self._prohibited(ne, tool, tc, args, outbound)
        return ne

    # -- helpers ---------------------------------------------------------------------------------------------
    def local_denial(self, d: D.Destination, tc: TaskCtx) -> str:
        """'' when ``d`` may be reached as far as the local-address rule goes, else why not (its local class)."""
        if d.kind != "url":
            return ""
        lk = localnet.local_kind(d.host)
        if not lk:
            return ""
        port = d.port or D._DEFAULT_PORTS.get(d.scheme, 0)
        if (d.host, port) in tc.local_services:
            return ""
        if lk == "loopback" and tc.own_process and localnet.own_listener(port):
            return ""
        return lk

    def _file(self, ne: NodeEval, d: D.Destination) -> None:
        """Inside a file root and not inside a protected path -- a file root that lies *inside* a protected path
        (the environment's workspace below its private directory) is more specific and wins."""
        low = d.path.lower()

        def depth_in(root: str) -> int:
            rl = root.lower().rstrip("/") or "/"
            return len(rl) if (low == rl or low.startswith(rl.rstrip("/") + "/")) else -1

        prot = max([depth_in(pp) for pp in self.p.protected_paths] or [-1])
        root = max([depth_in(r.path) for r in self.p.file_roots] or [-1])
        if prot >= 0 and root <= prot:
            ne.deny.append(Finding("P-FILE-ROOT", f"{d.path} is a protected path", d))
        elif root < 0:
            ne.deny.append(Finding("P-FILE-ROOT", f"{d.path} is outside the permitted file roots", d))

    def _platform(self, ne: NodeEval, d: Optional[D.Destination], tool: ToolRule, eff: str) -> None:
        plat = self.p.platform_of(d, tool)
        if plat is None:
            return
        node = ne.node
        for r in plat.rules:
            if r.modalities and node.modality not in r.modalities:
                continue
            if r.effects and eff not in r.effects and tool.effect not in r.effects:
                continue
            if r.kinds and tool.kind not in r.kinds:
                continue
            if r.ops and not any(fnmatch.fnmatchcase(node.operation, o) for o in r.ops):
                continue
            if r.data and not (set(r.data) & set(self._data_classes(tool, node.arguments))):
                continue
            f = Finding(r.id, f"{plat.name} rule: {r.reason}", d, "content")
            if any(x.pid == r.id for x in ne.deny + ne.approve):
                continue
            (ne.deny if r.verdict == "deny" else ne.approve).append(f)
        if plat.require_disclosure and eff in ("external_write", "interact"):
            ne.disclose = True

    def _data_classes(self, tool: ToolRule, args: Dict[str, Any]) -> List[str]:
        classes = list(tool.data)
        if "message_content_read" in classes:
            keys = {k for k, v in walk_keys(args) if truthy(v)}
            if keys & set(self.p.store_flags):
                classes.append("message_content_store")
            if keys & set(self.p.train_flags):
                classes.append("message_content_train")
        return classes

    def _data_use(self, ne: NodeEval, tool: ToolRule, tc: TaskCtx, args: Dict[str, Any]) -> None:
        perms = tc.task.data_permissions if isinstance(tc.task.data_permissions, dict) else {}
        classes = self._data_classes(tool, args)
        pr = self.p.prohibited
        personal = set(pr["personal_fields"])
        for k, v in walk_keys(args):
            if k in pr["field_list_args"]:
                asked = {norm_field(x) for x in _as_strings(v)}
                if asked & {"privatemessages", "dms", "dmhistory"}:
                    classes.append("private_messages")
                if asked & personal:
                    classes.append("personal_data")
        for cls in dict.fromkeys(classes):
            spec = self.p.data_classes.get(cls)
            if spec is None or spec["never"]:
                ne.deny.append(Finding("P-DATA-USE", f"data class {cls!r} is never permitted"))
            elif perms.get(spec["permission"], spec["default"]) is not True:
                pid = "X-DATA-COLLECTION" if cls in ("personal_data", "member_list") else "P-DATA-USE"
                ne.deny.append(Finding(pid, f"data permission {spec['permission']!r} is not granted for this task "
                                            f"(needed for {cls})"))

    def _credentials(self, ne: NodeEval, args: Dict[str, Any], text_all: str) -> None:
        pr = self.p.prohibited
        if self.vault is not None and self.vault.find(text_all):
            ne.deny.append(Finding("X-CREDENTIAL-LEAK", "a stored secret value appears in the action"))
        elif self.vault is not None and hasattr(self.vault, "find_loose") and self.vault.find_loose(text_all):
            ne.deny.append(Finding("X-CREDENTIAL-LEAK", "a stored secret value appears in the action (spaced out, "
                                                        "re-cased, punctuated or in parts)"))
        low = text_all.lower()
        if any(s in low for s in self.sensitive):
            ne.deny.append(Finding("X-CREDENTIAL-LEAK", "the action refers to the credential vault or the "
                                                        "authorization service's private files"))
        for kid, rx in self.p.credential_patterns:
            if rx.search(text_all):
                ne.deny.append(Finding("X-CREDENTIAL-LEAK", f"the action contains what looks like a credential ({kid})"))
                break
        prefix = pr["credential_ref_prefix"]
        for k, v in walk_keys(expand(args)):
            if k in pr["credential_fields"] and isinstance(v, str) and v and not v.startswith(prefix):
                ne.deny.append(Finding("X-CREDENTIAL-LEAK", f"a credential field carries a literal value; credentials "
                                                            f"are referenced by vault name ({prefix}<name>) and "
                                                            f"supplied to the executor at use"))
                break

    def _prohibited(self, ne: NodeEval, tool: ToolRule, tc: TaskCtx, args: Dict[str, Any], outbound: bool) -> None:
        pr, p = self.p.prohibited, self.p
        xargs = expand(args)                   # JSON / form bodies sent as strings are judged by their keys too
        keys = [(k, v) for k, v in walk_keys(xargs) if truthy(v)]
        present = {k for k, _ in keys}
        # unsolicited mass outreach ---------------------------------------------------------------------------
        n_rcpt = sum(_count_items(v) for k, v in keys if k in pr["recipient_fields"])
        shaped = sorted(recipient_lists(xargs), key=lambda kv: -kv[1])
        n_shaped = sum(n for _k, n in shaped)
        if max(n_rcpt, n_shaped) > pr["max_recipients"]:
            where = f" (recipient-shaped values under {', '.join(sorted({k or '?' for k, _ in shaped}))[:120]})" \
                if n_shaped > n_rcpt else ""
            ne.deny.append(Finding("X-MASS-OUTREACH", f"{max(n_rcpt, n_shaped)} recipients addressed in one action "
                                                      f"(at most {pr['max_recipients']}){where}"))
        if tool.kind == "direct_message" and pr["dm_require_opt_in"]:
            rcpts = {d.ident for d in ne.dests if d.kind == "account"}
            for k, v in keys:
                if k in pr["recipient_fields"]:
                    rcpts.update(_as_strings(v))
            missing = sorted(r for r in rcpts if r not in tc.opted_in)
            if missing or not rcpts:
                ne.deny.append(Finding("X-DM-OPT-IN", "direct message to recipients who have not opted in",
                                       next((d for d in ne.dests if d.kind == "account"), None)))
        low = ne.content.lower()
        broadcast = outbound and any(t in low for t in pr["broadcast_tokens"])
        users: Set[str] = set()
        if outbound and ne.content:
            for _mid, kind, rx in p.mention_patterns:
                found = rx.findall(ne.content) if kind == "user" else ([1] if rx.search(ne.content) else [])
                if kind == "broadcast" and found:
                    broadcast = True
                elif kind == "user":
                    users.update(m if isinstance(m, str) else "".join(m) for m in found)
        if broadcast:
            ne.approve.append(Finding("X-MASS-OUTREACH-BROADCAST", "the content notifies every member (or a whole "
                                                                   "role / channel) at once", None, "content"))
        if len(users) > pr["max_mentions"]:
            ne.deny.append(Finding("X-MASS-OUTREACH", f"the content mentions {len(users)} people individually "
                                                      f"(at most {pr['max_mentions']})"))
        # deceptive identity ------------------------------------------------------------------------------------
        imp = sorted(present & set(pr["impersonation_fields"]))
        if imp:
            ne.deny.append(Finding("X-IDENTITY", f"identity-override field(s) present: {', '.join(imp)}"))
        for k, v in keys:
            if k in ("identities", "identityids", "actingaccounts") and _count_items(v) > p.acting_identities:
                ne.deny.append(Finding("X-IDENTITY", "more than one acting identity"))
        need_disclosure = (tool.kind in p.disclosure_kinds and outbound) or ne.disclose
        if need_disclosure:
            if not p.disclosure_text:
                ne.deny.append(Finding("X-DISCLOSURE", "a disclosure is required for this publication but none is "
                                                       "configured by the operator"))
            elif p.disclosure_text.casefold() not in ne.content.casefold():
                ne.deny.append(Finding("X-DISCLOSURE", "the publication does not carry the configured disclosure"))
        # artificial membership / join-for-join -------------------------------------------------------------------
        if tool.kind in pr["membership_deny_kinds"]:
            ne.deny.append(Finding("X-MEMBERSHIP", f"operations of kind {tool.kind!r} (creating accounts, adding "
                                                   f"members, joining on someone's behalf) are not permitted"))
        mem = sorted(present & set(pr["membership_fields"]))
        if mem:
            ne.deny.append(Finding("X-MEMBERSHIP", f"artificial-membership field(s) present: {', '.join(mem)}"))
        # payment or reward for engagement ------------------------------------------------------------------------
        inc = sorted(present & set(pr["incentive_fields"]))
        if inc:
            ne.deny.append(Finding("X-INCENTIVE", f"engagement-incentive field(s) present: {', '.join(inc)}"))
        if tool.kind == "payment" or ne.money > 0:
            for k, v in keys:
                if k in pr["purpose_fields"] and isinstance(v, str):
                    if any(rx.search(v.lower()) for rx in self._purpose_res):
                        ne.deny.append(Finding("X-INCENTIVE", "payment whose stated purpose is membership or engagement"))
                        break
        # profiling / vulnerable audiences -----------------------------------------------------------------------
        prof = sorted(present & set(pr["profiling_fields"]))
        if prof:
            ne.deny.append(Finding("X-PROFILING", f"individual-profiling field(s) present: {', '.join(prof)}"))
        elif tool.opaque and ne.blob:
            words = ident_words(ne.blob)
            hits = sorted({" ".join(w) for w in self._profiling_words if w in words})
            if hits:
                ne.approve.append(Finding("X-PROFILING-CODE", f"generated code names individual-profiling data "
                                                              f"({', '.join(hits[:4])}): the operator decides",
                                          None, "content"))
        aud = [(k, v) for k, v in keys if k in pr["audience_fields"]]
        if aud:
            vulnerable = False
            for k, v in aud:
                if k in ("minage", "maxage", "agerange"):
                    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", json.dumps(v, default=str))]
                    vulnerable = vulnerable or any(n < 18 for n in nums)
                txt = json.dumps(v, default=str).lower()
                vulnerable = vulnerable or any(rx.search(txt) for rx in self._term_res)
            if vulnerable:
                ne.deny.append(Finding("X-VULNERABLE", "the action targets a vulnerable audience"))
            elif outbound:
                ne.approve.append(Finding("X-AUDIENCE-TARGETING", "the action selects an audience by attribute", None,
                                          "content"))
        if outbound and ne.content:
            vul = {t for t, rx in zip(pr["vulnerable_terms"], self._term_res) if rx.search(low)}
            exploit = [t for t in pr["exploit_terms"] if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low)]
            if len(vul) >= 2 or (vul and exploit):
                ne.approve.append(Finding("X-VULNERABLE-CONTENT", f"the content addresses vulnerable people "
                                                                  f"({', '.join(sorted(vul)[:4])}"
                                                                  f"{'; ' + ', '.join(exploit[:2]) if exploit else ''})",
                                          None, "content"))
        # content patterns (judgement) ----------------------------------------------------------------------------
        if outbound and ne.content:
            for cp in p.content_patterns:
                if cp.regex.search(ne.content):
                    f = Finding(cp.id, f"content matches prohibited class {cp.cls!r} ({cp.id})", None, "content")
                    (ne.deny if cp.verdict == "deny" else ne.approve).append(f)


def policy_profiling_names(policy: Policy) -> List[str]:
    """The operator's profiling field names with word boundaries kept (the compiled policy stores them squashed)."""
    return list(policy.prohibited.get("profiling_raw") or [])


def _values_only(obj: Any, depth: int = 0) -> List[str]:
    """String *values* (not keys) of a JSON-like object."""
    out: List[str] = []
    if depth > 16:
        return out
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_values_only(v, depth + 1))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_values_only(v, depth + 1))
    return out
