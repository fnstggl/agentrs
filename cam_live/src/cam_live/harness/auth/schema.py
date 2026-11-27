"""Strict loader for the authorization policy file (schema documented in docs/EA/POLICY_SCHEMA.md).

Anything unexpected — an unknown key, a wrong type, an unknown effect, a pattern or regular expression that does
not compile — raises :class:`PolicyError`; the service then denies everything until a valid policy is loaded.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ... import contracts as C
from . import destinations as D

EFFECTS = ("none", "local", "external_read", "external_write")
TOOL_EFFECTS = EFFECTS + ("interact", "by_method")
EFFECT_RANK = {e: i for i, e in enumerate(EFFECTS)}
MODES = ("diagnostic", "staging", "live")
VERDICTS3 = ("allow", "needs_approval", "deny")
CAP_KEYS = ("money", "money_per_action", "money_per_day", "compute_s", "max_external_actions", "max_decisions",
            "duration_s")
#: rate rules a task (staging) or a signed live record may set, within the mode's ``rate_caps``
RATE_CAP_KEYS = ("write", "write_global")
DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_policy.yaml")
#: the run policy of a source checkout (same content as the packaged default; docs/EA/README.md "Run policy")
RUN_POLICY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                               "configs", "authorization.yaml"))


def run_policy_path() -> str:
    """The policy a run uses when none is named: ``cam_live/configs/authorization.yaml`` if it exists (a source
    checkout), else the packaged ``default_policy.yaml``."""
    return RUN_POLICY_PATH if os.path.isfile(RUN_POLICY_PATH) else DEFAULT_POLICY_PATH


class PolicyError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------------------------------------------
def _keys(d: Any, where: str, required: Sequence[str] = (), optional: Sequence[str] = ()) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise PolicyError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - set(required) - set(optional)
    if unknown:
        raise PolicyError(f"{where}: unknown key(s) {sorted(map(str, unknown))}")
    missing = [k for k in required if k not in d]
    if missing:
        raise PolicyError(f"{where}: missing key(s) {missing}")
    return d


def _num(v: Any, where: str, lo: float = 0.0, integer: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < lo or v == float("inf"):
        raise PolicyError(f"{where}: expected a number >= {lo}, got {v!r}")
    if integer and int(v) != v:
        raise PolicyError(f"{where}: expected an integer, got {v!r}")
    return int(v) if integer else float(v)


def _strs(v: Any, where: str) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, list) or any(not isinstance(x, str) or not x for x in v):
        raise PolicyError(f"{where}: expected a list of non-empty strings")
    return list(v)


def _str(v: Any, where: str, allow_empty: bool = True) -> str:
    if not isinstance(v, str) or (not allow_empty and not v):
        raise PolicyError(f"{where}: expected a string")
    return v


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise PolicyError(f"{where}: expected true/false")
    return v


def _choice(v: Any, where: str, choices: Sequence[str]) -> str:
    if v not in choices:
        raise PolicyError(f"{where}: expected one of {list(choices)}, got {v!r}")
    return v


def _patterns(v: Any, where: str, run_dir: str = "") -> List[D.Pattern]:
    out = []
    for s in _strs(v, where):
        try:
            out.append(D.parse_pattern(s.replace("{run_dir}", run_dir) if run_dir else s))
        except D.DestinationError as e:
            raise PolicyError(f"{where}: {e}") from e
    return out


def _regex(v: Any, where: str) -> "re.Pattern[str]":
    try:
        return re.compile(_str(v, where, allow_empty=False))
    except re.error as e:
        raise PolicyError(f"{where}: bad regular expression: {e}") from e


def norm_field(name: str) -> str:
    """``asUser`` / ``as-user`` / ``AS_USER`` -> ``asuser`` (how argument names are compared with field lists)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ---------------------------------------------------------------------------------------------------------------
# compiled policy
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class DestRule:
    kind: str
    frm: List[str]
    platform: str = ""
    required: bool = False
    multi: bool = False
    location: bool = False       # may fall back to the browser location tracked from executor reports
    context: bool = False        # checked like any destination, but when the node names another (more specific)
                                 # destination the effect lands there: e.g. the server of a channel


@dataclass
class ToolRule:
    id: str
    ops: List[str]
    modalities: List[str]
    effect: str
    dests: List[DestRule] = field(default_factory=list)
    require_dest: bool = False
    kind: str = ""
    platform: str = ""
    data: List[str] = field(default_factory=list)
    content_from: List[str] = field(default_factory=list)
    amount_from: List[str] = field(default_factory=list)
    opaque: bool = False
    network_flags: List[str] = field(default_factory=list)
    interact_class: str = "input"
    modes: List[str] = field(default_factory=list)       # [] = every mode
    sets_location: bool = False  # an allowed primitive of this tool moves the tracked browser location
    counts_as_action: bool = True  # false: a primitive of this tool is not charged as an external action (a
                                   # subresource of a page the action already loads); rate limits still apply

    def matches(self, op: str) -> bool:
        return any(fnmatch.fnmatchcase(op, p) for p in self.ops)


@dataclass
class ModeRule:
    name: str
    max_effect: str
    url_envelope: Optional[List[D.Pattern]]
    owned: List[D.Pattern]
    read: List[D.Pattern]
    preauthorized: List[D.Pattern]
    honour_task_resources: bool
    default_interfaces: List[str]
    simulated_money: bool
    interact_non_owned: Dict[str, str]
    code_network: str
    requires_authorization_record: bool
    caps: Dict[str, float]
    own_process_services: bool = False      # loopback services of the authorization service's own process count
                                            # as registered local services (in-process staging only)
    rate_caps: Dict[str, "RateRule"] = field(default_factory=dict)   # most a task / live record may configure


@dataclass
class PlatformRule:
    id: str
    verdict: str
    reason: str
    modalities: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)
    ops: List[str] = field(default_factory=list)
    data: List[str] = field(default_factory=list)


@dataclass
class TransportRule:
    """``ops`` of a platform are realised as ``method`` requests to ``path`` (glob; ``{id}`` = the logical
    destination's id). A grant for the logical operation covers exactly these requests."""
    ops: List[str]
    method: str
    path: str


@dataclass
class Platform:
    name: str
    match: List[D.Pattern]
    require_disclosure: bool
    rules: List[PlatformRule]
    transport: List[TransportRule] = field(default_factory=list)


@dataclass
class RateRule:
    count: int
    window_s: float
    match: Optional[D.Pattern] = None
    effects: List[str] = field(default_factory=list)


@dataclass
class ContentPattern:
    id: str
    cls: str
    verdict: str
    regex: "re.Pattern[str]"


@dataclass
class Policy:
    policy_id: str
    sha256: str
    path: str
    grants: Dict[str, float]
    approvals_ttl_s: float
    structure: Dict[str, int]
    modes: Dict[str, ModeRule]
    deny: List[D.Pattern]
    interactive: List[D.Pattern]
    file_roots: List[D.Pattern]
    protected_paths: List[str]
    tools: List[ToolRule]
    publishing_always: bool
    spending: Dict[str, float]
    rate_default: RateRule
    rate_write: RateRule
    rate_primitive: RateRule
    rate_rules: List[RateRule]
    max_primitives_per_action: int
    data_classes: Dict[str, Dict[str, Any]]
    store_flags: List[str]
    train_flags: List[str]
    platforms: List[Platform]
    disclosure_text: str
    disclosure_kinds: List[str]
    acting_identities: int
    prohibited: Dict[str, Any]
    content_patterns: List[ContentPattern]
    credential_patterns: List[Tuple[str, "re.Pattern[str]"]]
    local_services: List[Tuple[str, int]] = field(default_factory=list)
    rate_write_global: Optional[RateRule] = None
    max_external_primitives_per_action: int = 0
    mention_patterns: List[Tuple[str, str, "re.Pattern[str]"]] = field(default_factory=list)   # (id, kind, regex)

    def tool_for(self, op: str) -> Optional[ToolRule]:
        for t in self.tools:
            if t.matches(op):
                return t
        return None

    def platform_of(self, dest: Optional[D.Destination], tool: Optional[ToolRule]) -> Optional[Platform]:
        if tool is not None and tool.platform:
            for p in self.platforms:
                if p.name == tool.platform:
                    return p
        if dest is not None:
            for p in self.platforms:
                if D.match_any(dest, p.match) is not None:
                    return p
        return None


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def _mode(name: str, d: Any, run_dir: str) -> ModeRule:
    w = f"modes.{name}"
    d = _keys(d, w, required=("max_effect", "url_envelope", "owned", "read", "preauthorized",
                              "honour_task_resources", "default_interfaces", "simulated_money",
                              "interact_non_owned", "code_network", "requires_authorization_record", "caps"),
              optional=("own_process_services", "rate_caps"))
    rc = _keys(d.get("rate_caps") or {}, f"{w}.rate_caps", optional=RATE_CAP_KEYS)
    ino = _keys(d["interact_non_owned"], f"{w}.interact_non_owned", required=("navigate", "input"))
    caps = _keys(d["caps"], f"{w}.caps", required=CAP_KEYS)
    ifaces = _strs(d["default_interfaces"], f"{w}.default_interfaces")
    return ModeRule(
        name=name, max_effect=_choice(d["max_effect"], f"{w}.max_effect", EFFECTS),
        url_envelope=None if d["url_envelope"] is None else _patterns(d["url_envelope"], f"{w}.url_envelope"),
        owned=_patterns(d["owned"], f"{w}.owned", run_dir), read=_patterns(d["read"], f"{w}.read", run_dir),
        preauthorized=_patterns(d["preauthorized"], f"{w}.preauthorized", run_dir),
        honour_task_resources=_bool(d["honour_task_resources"], f"{w}.honour_task_resources"),
        default_interfaces=ifaces, simulated_money=_bool(d["simulated_money"], f"{w}.simulated_money"),
        interact_non_owned={k: _choice(ino[k], f"{w}.interact_non_owned.{k}", VERDICTS3) for k in ("navigate", "input")},
        code_network=_choice(d["code_network"], f"{w}.code_network", VERDICTS3),
        requires_authorization_record=_bool(d["requires_authorization_record"], f"{w}.requires_authorization_record"),
        caps={k: float(_num(caps[k], f"{w}.caps.{k}")) for k in CAP_KEYS},
        own_process_services=_bool(d.get("own_process_services", False), f"{w}.own_process_services"),
        rate_caps={k: _rate(v, f"{w}.rate_caps.{k}") for k, v in rc.items()})


def _tool(i: int, d: Any) -> ToolRule:
    w = f"tools[{i}]"
    d = _keys(d, w, required=("id", "ops", "modalities", "effect"),
              optional=("dests", "require_dest", "kind", "platform", "data", "content_from", "amount_from", "opaque",
                        "network_flags", "interact_class", "modes", "sets_location", "counts_as_action"))
    mods = _strs(d["modalities"], f"{w}.modalities")
    for m in mods:
        _choice(m, f"{w}.modalities", C.Modality.ALL)
    dests = []
    for jx, r in enumerate(d.get("dests") or []):
        ww = f"{w}.dests[{jx}]"
        r = _keys(r, ww, required=("kind", "from"), optional=("platform", "required", "multi", "location", "context"))
        kind = _choice(r["kind"], f"{ww}.kind", D.KINDS)
        if kind in ("account", "channel") and not r.get("platform"):
            raise PolicyError(f"{ww}: account/channel destinations need a platform")
        dests.append(DestRule(kind=kind, frm=_strs(r["from"], f"{ww}.from"), platform=_str(r.get("platform", ""), ww),
                              required=_bool(r.get("required", False), ww), multi=_bool(r.get("multi", False), ww),
                              location=_bool(r.get("location", False), ww), context=_bool(r.get("context", False), ww)))
    ops = _strs(d["ops"], f"{w}.ops")
    if not ops:
        raise PolicyError(f"{w}.ops: empty")
    return ToolRule(
        id=_str(d["id"], f"{w}.id", allow_empty=False), ops=ops, modalities=mods,
        effect=_choice(d["effect"], f"{w}.effect", TOOL_EFFECTS), dests=dests,
        require_dest=_bool(d.get("require_dest", False), f"{w}.require_dest"),
        kind=_str(d.get("kind", ""), f"{w}.kind"), platform=_str(d.get("platform", ""), f"{w}.platform"),
        data=_strs(d.get("data"), f"{w}.data"), content_from=_strs(d.get("content_from"), f"{w}.content_from"),
        amount_from=_strs(d.get("amount_from"), f"{w}.amount_from"), opaque=_bool(d.get("opaque", False), f"{w}.opaque"),
        network_flags=[norm_field(x) for x in _strs(d.get("network_flags"), f"{w}.network_flags")],
        interact_class=_choice(d.get("interact_class", "input"), f"{w}.interact_class", ("navigate", "input")),
        modes=[_choice(m, f"{w}.modes", MODES) for m in _strs(d.get("modes"), f"{w}.modes")],
        sets_location=_bool(d.get("sets_location", False), f"{w}.sets_location"),
        counts_as_action=_bool(d.get("counts_as_action", True), f"{w}.counts_as_action"))


def _rate(d: Any, w: str, with_match: bool = False) -> RateRule:
    d = _keys(d, w, required=("count", "window_s") + (("match",) if with_match else ()),
              optional=("effects",) if with_match else ())
    m = None
    if with_match:
        (m,) = _patterns([d["match"]], f"{w}.match")
    eff = [_choice(e, f"{w}.effects", EFFECTS) for e in _strs(d.get("effects"), f"{w}.effects")] if with_match else []
    return RateRule(count=int(_num(d["count"], f"{w}.count", integer=True)),
                    window_s=float(_num(d["window_s"], f"{w}.window_s", lo=1e-6)), match=m, effects=eff)


def compile_policy(doc: Any, *, run_dir: str = "", path: str = "", raw: bytes = b"") -> Policy:
    top = _keys(doc, "policy", required=("version", "policy_id", "grants", "approvals", "structure", "modes",
                                         "destinations", "tools", "publishing", "spending", "rate_limits", "limits",
                                         "data_use", "platforms", "identity", "prohibited"))
    if top["version"] != 1:
        raise PolicyError(f"policy.version: unsupported version {top['version']!r}")
    g = _keys(top["grants"], "grants", required=("ttl_s", "exec_window_s", "exec_slack_s"))
    grants = {k: float(_num(g[k], f"grants.{k}", lo=0.0)) for k in g}
    if grants["ttl_s"] <= 0 or grants["exec_window_s"] <= 0:
        raise PolicyError("grants: ttl_s and exec_window_s must be positive")
    a = _keys(top["approvals"], "approvals", required=("ttl_s",))
    st = _keys(top["structure"], "structure", required=("max_depth", "max_nodes", "max_argument_bytes"))
    structure = {k: int(_num(st[k], f"structure.{k}", lo=1, integer=True)) for k in st}

    md = _keys(top["modes"], "modes", required=MODES)
    modes = {m: _mode(m, md[m], run_dir) for m in MODES}
    if not modes["live"].requires_authorization_record:
        raise PolicyError("modes.live.requires_authorization_record must be true")

    ds = _keys(top["destinations"], "destinations", required=("deny", "interactive", "file_roots", "protected_paths"),
               optional=("local_services",))
    local_services = []
    for i, txt in enumerate(_strs(ds.get("local_services"), "destinations.local_services")):
        try:
            pat = D.parse_pattern(txt)
        except D.DestinationError as e:
            raise PolicyError(f"destinations.local_services[{i}]: {e}") from e
        if pat.kind != "url" or not pat.port or pat.path or pat.scheme or any(c in pat.host for c in "*?"):
            raise PolicyError(f"destinations.local_services[{i}]: expected an exact host:port, got {txt!r}")
        local_services.append((pat.host, pat.port))
    if not isinstance(top["tools"], list) or not top["tools"]:
        raise PolicyError("tools: expected a non-empty list")
    tools = [_tool(i, t) for i, t in enumerate(top["tools"])]
    ids = [t.id for t in tools]
    if len(set(ids)) != len(ids):
        raise PolicyError("tools: duplicate id")

    pub = _keys(top["publishing"], "publishing", required=("require_approval",))
    _choice(pub["require_approval"], "publishing.require_approval", ("unless_preauthorized", "always"))
    sp = _keys(top["spending"], "spending", required=("auto_approve_below", "per_action", "per_day"))
    spending = {k: float(_num(sp[k], f"spending.{k}")) for k in sp}
    rl = _keys(top["rate_limits"], "rate_limits", required=("default", "write", "primitive", "per_destination"),
               optional=("write_global",))
    if not isinstance(rl["per_destination"], list):
        raise PolicyError("rate_limits.per_destination: expected a list")
    lm = _keys(top["limits"], "limits", required=("max_primitives_per_action",),
               optional=("max_external_primitives_per_action",))

    du = _keys(top["data_use"], "data_use", required=("classes", "store_flags", "train_flags"))
    classes: Dict[str, Dict[str, Any]] = {}
    for name, c in _keys(du["classes"], "data_use.classes", optional=tuple(du["classes"] or {})).items():
        c = _keys(c, f"data_use.classes.{name}", optional=("permission", "default", "never"))
        never = _bool(c.get("never", False), f"data_use.classes.{name}.never")
        if not never and not c.get("permission"):
            raise PolicyError(f"data_use.classes.{name}: needs 'permission' or 'never: true'")
        classes[name] = {"permission": _str(c.get("permission", ""), f"data_use.classes.{name}.permission"),
                         "default": _bool(c.get("default", False), f"data_use.classes.{name}.default"), "never": never}
    for t in tools:
        for dc in t.data:
            if dc not in classes:
                raise PolicyError(f"tool {t.id}: unknown data class {dc!r}")

    platforms = []
    for name, p in _keys(top["platforms"], "platforms", optional=tuple(top["platforms"] or {})).items():
        w = f"platforms.{name}"
        p = _keys(p, w, required=("match", "rules"), optional=("require_disclosure", "transport"))
        transport = []
        if not isinstance(p.get("transport") or [], list):
            raise PolicyError(f"{w}.transport: expected a list")
        for i, t in enumerate(p.get("transport") or []):
            ww = f"{w}.transport[{i}]"
            t = _keys(t, ww, required=("ops", "method", "path"))
            path = _str(t["path"], f"{ww}.path", allow_empty=False)
            if not path.startswith("/") or path.count("{id}") != 1:
                raise PolicyError(f"{ww}.path: must start with '/' and contain {{id}} exactly once")
            transport.append(TransportRule(ops=_strs(t["ops"], f"{ww}.ops"),
                                           method=_str(t["method"], f"{ww}.method", allow_empty=False).upper(),
                                           path=path))
        rules = []
        if not isinstance(p["rules"], list):
            raise PolicyError(f"{w}.rules: expected a list")
        for i, r in enumerate(p["rules"]):
            ww = f"{w}.rules[{i}]"
            r = _keys(r, ww, required=("id", "when", "verdict", "reason"))
            wh = _keys(r["when"], f"{ww}.when", optional=("modalities", "effects", "kinds", "ops", "data"))
            if not wh:
                raise PolicyError(f"{ww}.when: at least one condition is required")
            rules.append(PlatformRule(
                id=_str(r["id"], f"{ww}.id", allow_empty=False),
                verdict=_choice(r["verdict"], f"{ww}.verdict", ("deny", "needs_approval")),
                reason=_str(r["reason"], f"{ww}.reason", allow_empty=False),
                modalities=[_choice(m, f"{ww}.when.modalities", C.Modality.ALL) for m in _strs(wh.get("modalities"), ww)],
                effects=[_choice(e, f"{ww}.when.effects", TOOL_EFFECTS) for e in _strs(wh.get("effects"), ww)],
                kinds=_strs(wh.get("kinds"), ww), ops=_strs(wh.get("ops"), ww), data=_strs(wh.get("data"), ww)))
        platforms.append(Platform(name=str(name), match=_patterns(p["match"], f"{w}.match"),
                                  require_disclosure=_bool(p.get("require_disclosure", False), w), rules=rules,
                                  transport=transport))

    idn = _keys(top["identity"], "identity", required=("disclosure_text", "require_disclosure_for_kinds",
                                                       "acting_identities"))
    pr = _keys(top["prohibited"], "prohibited", required=("mass_outreach", "direct_message", "identity", "membership",
                                                          "incentives", "data_collection", "profiling", "vulnerable",
                                                          "credentials", "content_patterns"))
    mo = _keys(pr["mass_outreach"], "prohibited.mass_outreach",
               required=("recipient_fields", "max_recipients", "max_write_destinations", "duplicate", "broadcast_tokens"),
               optional=("max_mentions", "mention_patterns"))
    dup = _keys(mo["duplicate"], "prohibited.mass_outreach.duplicate",
                required=("similarity", "max_destinations", "window_s", "min_chars"))
    sim = _num(dup["similarity"], "prohibited.mass_outreach.duplicate.similarity")
    if not 0.0 < sim <= 1.0:
        raise PolicyError("prohibited.mass_outreach.duplicate.similarity must be in (0, 1]")
    dm = _keys(pr["direct_message"], "prohibited.direct_message", required=("require_opt_in",))
    ident = _keys(pr["identity"], "prohibited.identity", required=("impersonation_fields",))
    mem = _keys(pr["membership"], "prohibited.membership", required=("deny_kinds", "fields"))
    inc = _keys(pr["incentives"], "prohibited.incentives", required=("fields", "purpose_fields", "denied_purposes"))
    dc = _keys(pr["data_collection"], "prohibited.data_collection", required=("personal_fields", "field_list_args"))
    prof = _keys(pr["profiling"], "prohibited.profiling", required=("fields",))
    vul = _keys(pr["vulnerable"], "prohibited.vulnerable", required=("audience_fields", "terms"),
                optional=("exploit_terms",))
    cred = _keys(pr["credentials"], "prohibited.credentials", required=("field_names", "reference_prefix", "patterns"))

    def fields(v: Any, w: str) -> List[str]:
        return [norm_field(x) for x in _strs(v, w)]

    prohibited = {
        "recipient_fields": fields(mo["recipient_fields"], "prohibited.mass_outreach.recipient_fields"),
        "max_recipients": int(_num(mo["max_recipients"], "prohibited.mass_outreach.max_recipients", integer=True)),
        "max_write_destinations": int(_num(mo["max_write_destinations"], "prohibited.mass_outreach.max_write_destinations",
                                           lo=1, integer=True)),
        "dup_similarity": float(sim),
        "dup_max_destinations": int(_num(dup["max_destinations"], "duplicate.max_destinations", lo=1, integer=True)),
        "dup_window_s": float(_num(dup["window_s"], "duplicate.window_s")),
        "dup_min_chars": int(_num(dup["min_chars"], "duplicate.min_chars", integer=True)),
        "broadcast_tokens": [t.lower() for t in _strs(mo["broadcast_tokens"], "prohibited.mass_outreach.broadcast_tokens")],
        "dm_require_opt_in": _bool(dm["require_opt_in"], "prohibited.direct_message.require_opt_in"),
        "impersonation_fields": fields(ident["impersonation_fields"], "prohibited.identity.impersonation_fields"),
        "membership_deny_kinds": _strs(mem["deny_kinds"], "prohibited.membership.deny_kinds"),
        "membership_fields": fields(mem["fields"], "prohibited.membership.fields"),
        "incentive_fields": fields(inc["fields"], "prohibited.incentives.fields"),
        "purpose_fields": fields(inc["purpose_fields"], "prohibited.incentives.purpose_fields"),
        "denied_purposes": [t.lower() for t in _strs(inc["denied_purposes"], "prohibited.incentives.denied_purposes")],
        "personal_fields": fields(dc["personal_fields"], "prohibited.data_collection.personal_fields"),
        "field_list_args": fields(dc["field_list_args"], "prohibited.data_collection.field_list_args"),
        "profiling_fields": fields(prof["fields"], "prohibited.profiling.fields"),
        "profiling_raw": [x.lower() for x in _strs(prof["fields"], "prohibited.profiling.fields")],
        "audience_fields": fields(vul["audience_fields"], "prohibited.vulnerable.audience_fields"),
        "vulnerable_terms": [t.lower() for t in _strs(vul["terms"], "prohibited.vulnerable.terms")],
        "exploit_terms": [t.lower() for t in _strs(vul.get("exploit_terms"), "prohibited.vulnerable.exploit_terms")],
        "max_mentions": int(_num(mo.get("max_mentions", 10), "prohibited.mass_outreach.max_mentions", integer=True)),
        "credential_fields": fields(cred["field_names"], "prohibited.credentials.field_names"),
        "credential_ref_prefix": _str(cred["reference_prefix"], "prohibited.credentials.reference_prefix",
                                      allow_empty=False),
    }
    cps = []
    if not isinstance(pr["content_patterns"], list) or not isinstance(cred["patterns"], list):
        raise PolicyError("prohibited: content_patterns and credentials.patterns must be lists")
    for i, c in enumerate(pr["content_patterns"]):
        w = f"prohibited.content_patterns[{i}]"
        c = _keys(c, w, required=("id", "class", "verdict", "regex"))
        cps.append(ContentPattern(id=_str(c["id"], w, allow_empty=False), cls=_str(c["class"], w, allow_empty=False),
                                  verdict=_choice(c["verdict"], f"{w}.verdict", ("deny", "needs_approval")),
                                  regex=_regex(c["regex"], f"{w}.regex")))
    mps = []
    if not isinstance(mo.get("mention_patterns") or [], list):
        raise PolicyError("prohibited.mass_outreach.mention_patterns: expected a list")
    for i, c in enumerate(mo.get("mention_patterns") or []):
        w = f"prohibited.mass_outreach.mention_patterns[{i}]"
        c = _keys(c, w, required=("id", "kind", "regex"))
        mps.append((_str(c["id"], w, allow_empty=False), _choice(c["kind"], f"{w}.kind", ("broadcast", "user")),
                    _regex(c["regex"], f"{w}.regex")))
    kps = []
    for i, c in enumerate(cred["patterns"]):
        w = f"prohibited.credentials.patterns[{i}]"
        c = _keys(c, w, required=("id", "regex"))
        kps.append((_str(c["id"], w, allow_empty=False), _regex(c["regex"], f"{w}.regex")))

    return Policy(
        policy_id=_str(top["policy_id"], "policy_id", allow_empty=False),
        sha256=hashlib.sha256(raw).hexdigest() if raw else "", path=path, grants=grants,
        approvals_ttl_s=float(_num(a["ttl_s"], "approvals.ttl_s", lo=1.0)), structure=structure, modes=modes,
        deny=_patterns(ds["deny"], "destinations.deny", run_dir),
        interactive=_patterns(ds["interactive"], "destinations.interactive", run_dir),
        file_roots=_patterns(["file:" + s if not s.startswith("file:") else s
                              for s in _strs(ds["file_roots"], "destinations.file_roots")
                              if run_dir or "{run_dir}" not in s],
                             "destinations.file_roots", run_dir),
        protected_paths=[os.path.normpath(s.replace("{run_dir}", run_dir)) for s in
                         _strs(ds["protected_paths"], "destinations.protected_paths") if run_dir or "{run_dir}" not in s],
        tools=tools, publishing_always=pub["require_approval"] == "always", spending=spending,
        rate_default=_rate(rl["default"], "rate_limits.default"), rate_write=_rate(rl["write"], "rate_limits.write"),
        rate_primitive=_rate(rl["primitive"], "rate_limits.primitive"),
        rate_rules=[_rate(r, f"rate_limits.per_destination[{i}]", with_match=True)
                    for i, r in enumerate(rl["per_destination"])],
        max_primitives_per_action=int(_num(lm["max_primitives_per_action"], "limits.max_primitives_per_action",
                                           lo=1, integer=True)),
        data_classes=classes, store_flags=fields(du["store_flags"], "data_use.store_flags"),
        train_flags=fields(du["train_flags"], "data_use.train_flags"), platforms=platforms,
        disclosure_text=_str(idn["disclosure_text"], "identity.disclosure_text"),
        disclosure_kinds=_strs(idn["require_disclosure_for_kinds"], "identity.require_disclosure_for_kinds"),
        acting_identities=int(_num(idn["acting_identities"], "identity.acting_identities", lo=1, integer=True)),
        prohibited=prohibited, content_patterns=cps, credential_patterns=kps, local_services=local_services,
        rate_write_global=_rate(rl["write_global"], "rate_limits.write_global") if rl.get("write_global") else None,
        max_external_primitives_per_action=int(_num(lm.get("max_external_primitives_per_action",
                                                            lm["max_primitives_per_action"]),
                                                     "limits.max_external_primitives_per_action", lo=1, integer=True)),
        mention_patterns=mps)


def load_policy(path: Optional[str] = None, run_dir: str = "") -> Policy:
    path = path or DEFAULT_POLICY_PATH
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise PolicyError(f"cannot read policy file {path}: {e.strerror}") from e
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise PolicyError(f"policy file is not valid YAML: {e}") from e
    return compile_policy(doc, run_dir=os.path.abspath(run_dir) if run_dir else "", path=path, raw=raw)
