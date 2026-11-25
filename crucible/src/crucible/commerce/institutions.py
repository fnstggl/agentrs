"""Adaptable institutions within stable constraints (module `inst`, host `inst`; MASTER_SPEC Part 6.6).

Built on the 4.0 identity, bank, contracts and governance code, never beside it: organisations are identity orgs
(`id.create_org`), membership is identity membership (invitations are accepted, ownership is accepted), delegation
is identity grants, money moves only through `bank`, and a company's winding up is `business`'s procedure. This
module adds what an organisation needs to *change how it works* - and nothing that lets it change the laws it
works under.

**Charter.** An organisation may adopt a charter (`inst.adopt`): named decision procedures, rules saying which
procedure decides which class of action, offices with monthly budgets, membership admission and the basis of
distribution on dissolution. A charter is operational: its schema is closed and every field is validated.

**Decision procedures** (the `type` of a procedure):

* `majority` / `supermajority` - share of the votes cast (default thresholds 1/2 and 2/3) with a quorum;
* `unanimous` - every elector approves;
* `weighted` - votes weighted by shares held, recorded contributions (money paid into the organisation's projects,
  and labour valued at its wage) or tenure;
* `delegated` - liquid democracy: an elector may delegate its vote (`inst.delegate_vote`); weight flows along the
  chain to whoever votes, a direct vote overrides the delegation, cycles abstain;
* `consensus` - passes when the window closes without an objection (a single `no` blocks it);
* `officer` - an office holder decides alone, within the office's monthly budget (delegated authority).

The electorate and its weights are fixed when a decision is proposed, so recruiting voters or buying shares after
a proposal cannot change its outcome.

**Enforcement is part of authorisation.** The module chains itself into the identity hook `gov_check`: any tool
run in an organisation's name (by owners too, and also under an approved 4.0 `gov` proposal) that a charter rule
governs is refused unless it runs as the execution of an approved `inst` decision or by an officer within budget.
Where both a 4.0 `gov` constitution and a charter govern an action, both must be satisfied (the stricter wins).

**What decisions can do** (`action.kind`): run a tool as the organisation (`tool`), amend the charter (`amend`),
admit or expel members and change roles, create / staff / dissolve units, appoint officers and set their budgets,
allocate money to projects and tasks, merge with another organisation (`merge` + `absorb`, both organisations
decide), spin a unit off into a new organisation (`spin_off`) and dissolve the organisation (`dissolve`).

**Shared projects and cooperative ventures.** A project has a pool held in escrow (`acct_inst_escrow`), funded by
recorded contributions, a lead, tasks with skill requirements, effort and dependencies (a task can only be worked
once the tasks it depends on are done), and rewards allocated from the pool and paid on verified completion. A
project with partner organisations is a cooperative venture: partners join by consent, contribute, and - when the
venture has a procedure - decide allocations and closure together (weighted by contribution if they choose);
on closure what is left in the pool returns to contributors pro rata.

**Stable constraints** (refused, whatever an organisation decides): a charter cannot contain anything but
operational rules (no accounting, ledger, fee, overdraft, permission, hook, evaluation or world-state field);
charter conditions may read only metrics about the organisation itself (never evaluator-only metrics); decisions
execute only real tools through the kernel, as the organisation, under ordinary authorisation - operator (`admin`)
tools, other principals' resources, privilege escalation and unknown action kinds are refused; money only moves
through `bank` with its conservation and overdraft laws.
"""
from __future__ import annotations

import numbers
import re

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, UnknownTool, contain
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
MONTH = 30 * DAY
ESCROW = "acct_inst_escrow"
MAX_AMOUNT = 10**12

PROC_TYPES = ("majority", "supermajority", "unanimous", "weighted", "delegated", "consensus", "officer")
ELECTORATES = ("members", "admins", "owners", "shareholders", "partners")
WEIGHTS = ("equal", "shares", "contribution", "tenure")
ACTION_KINDS = ("tool", "amend", "admit", "expel", "role", "unit_create", "unit_assign", "unit_dissolve",
                "office", "office_remove", "allocate", "project_close", "merge", "absorb", "spin_off", "dissolve")
VENTURE_KINDS = ("allocate", "project_close", "admit_partner")
CHARTER_KEYS = ("procedures", "rules", "default", "membership", "offices", "dissolution")
DISTRIBUTIONS = ("equal", "shares", "contribution")
AMOUNT_KEYS = ("amount", "amount_cents", "salary_cents_month", "price_cents", "reward_cents", "budget_cents",
               "cents")
NAME = r"[a-z][a-z0-9_-]{0,39}"
UNIT_ID = r"unit_[0-9]{6}"
HANDLE = r"[a-z][a-z0-9-]{1,30}"

# What a charter can never touch, by the kind of law it would change (the message names the law; the key set is
# not exhaustive - every key outside CHARTER_KEYS is refused - it only explains common attempts).
PROTECTED_TOPICS = {
    "core accounting law": ("accounting", "ledger", "balance", "balances", "overdraft", "money", "mint", "fees",
                            "fee", "currency", "conservation", "bank", "postings", "revenue", "profit"),
    "security boundary": ("permissions", "permission", "security", "authorize", "auth", "hooks", "principals",
                          "act_as", "may_act_as", "grants", "admin", "sandbox", "capabilities", "visibility"),
    "hidden evaluation criteria": ("evaluation", "evaluator", "metrics", "metric", "score", "scoring", "reward",
                                   "objective", "goals", "independence", "related_parties", "verification"),
    "world state": ("state", "world", "clock", "time", "tables", "events", "schedule", "seed", "rng", "params"),
}


def _protected_topic(key: str) -> str | None:
    k = str(key).lower()
    for law, keys in PROTECTED_TOPICS.items():
        if k in keys or any(k.startswith(x + "_") or k.endswith("_" + x) for x in keys):
            return law
    return None


def _num(x, lo: float, hi: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise InvalidAction(f"{what} must be a number")
    x = float(x)
    if not lo <= x <= hi:
        raise InvalidAction(f"{what} must be between {lo} and {hi}")
    return x


def _int(x, lo: int, hi: int, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise InvalidAction(f"{what} must be an integer")
    x = int(x)
    if not lo <= x <= hi:
        raise InvalidAction(f"{what} must be between {lo} and {hi}")
    return x


def _str(x, pattern: str, what: str) -> str:
    if not isinstance(x, str) or not re.fullmatch(pattern, x):
        raise InvalidAction(f"{what} is not valid")
    return x


def _amount(args) -> int:
    if not isinstance(args, dict):
        return 0
    return max([int(v) for k, v in args.items() if k in AMOUNT_KEYS and isinstance(v, int)
                and not isinstance(v, bool)] or [0])


def _tool_match(pattern: str, tool: str) -> bool:
    return pattern in ("*", tool) or (pattern.endswith(".*") and tool.startswith(pattern[:-1]))


IMPLICIT = {"type": "unanimous", "electorate": "owners", "threshold": 1.0, "quorum": 0.0, "window_s": 2 * DAY,
            "weights": "equal", "office": None}


@register_module
class Institutions(Module):
    name = "inst"
    depends = ("identity", "bank", "contracts")

    def install(self, world):
        st = world.state
        st.declare_table("inst_charters")
        st.declare_table("inst_decisions", index=("org", "status"))
        st.declare_table("inst_ballots", index=("decision",))
        st.declare_table("inst_delegations", index=("org", "delegate"))
        st.declare_table("inst_units", index=("org", "status"))
        st.declare_table("inst_unit_members", index=("unit", "person"))
        st.declare_table("inst_offices", index=("org", "holder"))
        st.declare_table("inst_office_spend")
        st.declare_table("inst_projects", index=("owner", "status"))
        st.declare_table("inst_partners", index=("project", "partner"))
        st.declare_table("inst_tasks", index=("project", "assignee", "status"))
        st.declare_table("inst_contributions", index=("org", "contributor", "project"))
        st.declare_table("inst_merge_offers", index=("into",))
        world.hooks["inst"] = self
        self._chain_gov_check(world)
        cur = world.hooks.get("on_member_left")

        def left(ctx, org, person, _inner=cur):
            if _inner is not None:
                _inner(ctx, org, person)
            self.on_member_left(ctx, org, person)
        world.hooks["on_member_left"] = left

    def on_load(self, world):
        # the 4.0 `gov` module *sets* (does not chain) `gov_check` in its install: if it was installed after this
        # module it replaced the chain, which is restored here (before any command can run)
        gov = world.modules.get("gov")
        cur = world.hooks.get("gov_check")
        if gov is not None and getattr(cur, "__self__", None) is gov:
            self._chain_gov_check(world)

    def _chain_gov_check(self, world):
        """Charter enforcement joins the authorisation hook `gov_check`, after whatever is already there (the 4.0
        `gov` constitution check), so both institutions' rules apply."""
        cur = world.hooks.get("gov_check")

        def chained(ctx, owner, tool, args, _inner=cur):
            if _inner is not None:
                _inner(ctx, owner, tool, args)
            self.check(ctx, owner, tool, args)
        world.hooks["gov_check"] = chained

    def genesis(self, ctx):
        ctx.emit("bank.account", {"id": ESCROW, "owner": SYSTEM, "kind": "escrow", "name": "institution escrow",
                                  "overdraft": 0}, vis="internal")
        ctx.schedule(ctx.now + 3600, "inst.tick", {}, prio=58, period=3600, key="inst.tick")

    # ================================================================== charter validation (the stable constraints)
    def validate_charter(self, ctx: Ctx, org: str, charter) -> dict:
        if not isinstance(charter, dict):
            raise InvalidAction("a charter is an object")
        for k in sorted(charter):
            if k not in CHARTER_KEYS:
                law = _protected_topic(k)
                if law:
                    raise Forbidden(f"a charter cannot change the {law} ({k!r}): institutions adapt their "
                                    f"operational rules only ({', '.join(CHARTER_KEYS)})", law=law, key=k)
                raise InvalidAction(f"unknown charter field {k!r}: a charter holds only {', '.join(CHARTER_KEYS)}")
        procs_in = charter.get("procedures") or {}
        if not isinstance(procs_in, dict) or len(procs_in) > 12:
            raise InvalidAction("procedures: an object of at most 12 named procedures")
        procs = {}
        for name in sorted(procs_in):
            _str(name, NAME, f"procedure name {name!r}")
            procs[name] = self.validate_procedure(procs_in[name], f"procedure {name}")
        rules = []
        rin = charter.get("rules") or []
        if not isinstance(rin, list) or len(rin) > 30:
            raise InvalidAction("rules: a list of at most 30 rules")
        for i, r in enumerate(rin):
            if not isinstance(r, dict) or set(r) - {"match", "procedure"}:
                raise InvalidAction(f"rule {i}: {{match, procedure}}")
            m = r.get("match") or {}
            if not isinstance(m, dict) or set(m) - {"kind", "tool", "min_amount", "when"}:
                raise InvalidAction(f"rule {i}: match has kind, tool, min_amount and when only")
            kind = m.get("kind", "tool" if "tool" in m else "*")
            if kind != "*" and kind not in ACTION_KINDS:
                raise InvalidAction(f"rule {i}: unknown action kind {kind!r}")
            mm = {"kind": kind}
            if "tool" in m:
                if kind != "tool":
                    raise InvalidAction(f"rule {i}: a tool pattern only applies to kind 'tool'")
                mm["tool"] = _str(m["tool"], r"\*|[a-z]+\.(\*|[a-z_]+)", f"rule {i}: tool pattern")
            if "min_amount" in m:
                mm["min_amount"] = _int(m["min_amount"], 0, MAX_AMOUNT, f"rule {i}: min_amount")
            if "when" in m:
                mm["when"] = self.validate_condition(ctx, org, m["when"], f"rule {i}")
            pn = r.get("procedure")
            if pn not in procs:
                raise InvalidAction(f"rule {i}: procedure {pn!r} is not defined")
            rules.append({"match": mm, "procedure": pn})
        default = charter.get("default")
        if default is not None and default not in procs:
            raise InvalidAction(f"default procedure {default!r} is not defined")
        mem = charter.get("membership") or {}
        if not isinstance(mem, dict) or set(mem) - {"admission", "max_members"}:
            raise InvalidAction("membership has admission and max_members only")
        membership = {"admission": mem.get("admission", "invite"),
                      "max_members": _int(mem.get("max_members", 10_000), 1, 10_000_000, "max_members")}
        if membership["admission"] not in ("invite", "open", "decision"):
            raise InvalidAction("admission must be invite, open or decision")
        offices = {}
        oin = charter.get("offices") or {}
        if not isinstance(oin, dict) or len(oin) > 20:
            raise InvalidAction("offices: an object of at most 20 offices")
        for title in sorted(oin):
            offices[title] = self.validate_office(ctx, org, title, oin[title])
        for pn, p in procs.items():
            if p["type"] == "officer" and p["office"] != "*" and p["office"] not in offices:
                raise InvalidAction(f"procedure {pn}: office {p['office']!r} is not defined")
        dis = charter.get("dissolution") or {}
        if not isinstance(dis, dict) or set(dis) - {"distribution"}:
            raise InvalidAction("dissolution has distribution only")
        dist = dis.get("distribution", "shares")
        if dist not in DISTRIBUTIONS:
            raise InvalidAction(f"distribution must be one of {DISTRIBUTIONS}")
        return {"procedures": procs, "rules": rules, "default": default, "membership": membership,
                "offices": offices, "dissolution": {"distribution": dist}}

    def validate_procedure(self, p, what: str = "procedure") -> dict:
        if not isinstance(p, dict) or set(p) - {"type", "electorate", "threshold", "quorum", "window_s", "weights",
                                               "office"}:
            raise InvalidAction(f"{what}: type, electorate, threshold, quorum, window_s, weights, office")
        t = p.get("type")
        if t not in PROC_TYPES:
            raise InvalidAction(f"{what}: type must be one of {PROC_TYPES}")
        el = p.get("electorate", "members")
        if not isinstance(el, str) or not (el in ELECTORATES or re.fullmatch("unit:" + UNIT_ID, el)):
            raise InvalidAction(f"{what}: electorate must be one of {ELECTORATES} or unit:<unit id>")
        thr = {"supermajority": 2 / 3, "unanimous": 1.0}.get(t, 0.5)
        thr = _num(p.get("threshold", thr), 0.5, 1.0, f"{what}: threshold")
        weights = p.get("weights", "shares" if t == "weighted" else "equal")
        if weights not in WEIGHTS:
            raise InvalidAction(f"{what}: weights must be one of {WEIGHTS}")
        office = p.get("office")
        if t == "officer":
            office = _str(office, r"\*|" + NAME + "|unit:" + UNIT_ID, f"{what}: office")
        elif office is not None:
            raise InvalidAction(f"{what}: only officer procedures name an office")
        return {"type": t, "electorate": el, "threshold": thr,
                "quorum": _num(p.get("quorum", 0.0), 0.0, 1.0, f"{what}: quorum"),
                "window_s": _int(p.get("window_s", 2 * DAY), 3600, 30 * DAY, f"{what}: window_s"),
                "weights": weights, "office": office}

    def validate_office(self, ctx, org, title, o) -> dict:
        _str(title, NAME, f"office title {title!r}")
        if not isinstance(o, dict) or set(o) - {"holder", "budget_cents_month", "tools"}:
            raise InvalidAction(f"office {title}: holder, budget_cents_month, tools")
        holder = o.get("holder")
        if holder is not None:
            holder = ident(ctx.world).resolve_handle(ctx.state, holder)
            if ident(ctx.world).role(ctx.state, org, holder) is None:
                raise InvalidAction(f"office {title}: the holder must be a member")
        tools = o.get("tools", ["*"])
        if not isinstance(tools, list) or len(tools) > 20:
            raise InvalidAction(f"office {title}: tools is a list of tool patterns")
        tools = [_str(t, r"\*|[a-z]+\.(\*|[a-z_]+)", f"office {title}: tool pattern") for t in tools]
        return {"holder": holder, "budget_cents_month": _int(o.get("budget_cents_month", 0), 0, MAX_AMOUNT,
                                                             f"office {title}: budget"), "tools": tools}

    def validate_condition(self, ctx, org, pred, what) -> dict:
        """A charter condition is a metrics predicate about the organisation itself (goals/metrics.py): metrics
        without a declared subject are evaluator-only and cannot be read by an institution."""
        from ..goals.metrics import check_predicate, check_predicate_scope
        errs = check_predicate(pred)
        if errs:
            raise InvalidAction(f"{what}: " + "; ".join(errs[:3]))
        errs = check_predicate_scope(ctx.world, pred, [org])
        if errs:
            raise Forbidden(f"{what}: an institution may only condition on its own data - " + "; ".join(errs[:3]),
                            law="hidden evaluation criteria")
        return pred

    # ================================================================== charter lookup and the authorisation hook
    def charter(self, st, org) -> dict | None:
        c = st.get("inst_charters", org)
        return None if c is None else c["charter"]

    def rule_for(self, world, charter: dict, kind: str, tool: str | None, args) -> dict | None:
        from ..goals.metrics import evaluate_predicate
        amt = _amount(args)
        for r in charter["rules"]:
            m = r["match"]
            if m["kind"] not in ("*", kind):
                continue
            if kind == "tool" and "tool" in m and not _tool_match(m["tool"], tool or ""):
                continue
            if amt < m.get("min_amount", 0):
                continue
            if "when" in m:
                try:
                    if not evaluate_predicate(world, m["when"]):
                        continue
                except Exception:  # noqa: BLE001 - a condition that cannot be evaluated: the rule applies
                    pass
            return r
        return None

    def procedure_for(self, world, org, kind, tool=None, args=None) -> tuple[str, dict]:
        c = self.charter(world.state, org)
        if c is None:
            return "implicit", dict(IMPLICIT)
        r = self.rule_for(world, c, kind, tool, args)
        if r is not None:
            return r["procedure"], c["procedures"][r["procedure"]]
        if c["default"]:
            return c["default"], c["procedures"][c["default"]]
        return "implicit", dict(IMPLICIT)

    def executing(self, world, org) -> bool:
        return org in world.runtime.get("inst_exec", ())

    def consented(self, ctx: Ctx, org: str, fn, *args):
        """Run `fn(sub, *args)` as `org` for the mechanical consequence of something the organisation already
        decided under its rules (signing the employment contract of an offer it made, which was authorised - and
        governed - when the offer was made). Inside, neither the charter nor a 4.0 constitution asks again."""
        stack = ctx.world.runtime.setdefault("inst_exec", [])
        stack.append(org)

        def run(sub):
            sub.gov_approved = org
            return fn(sub, *args)
        try:
            return ctx.run_as(org, run)
        finally:
            stack.pop()

    def check(self, ctx: Ctx, org: str, tool: str, args: dict) -> None:
        """Called (through `gov_check`) before a tool runs in an organisation's name. A tool a charter rule governs
        runs only as an approved decision's execution, or by an officer the rule names, within budget."""
        st = ctx.state
        if not st.has("inst_charters", org) or self.executing(ctx.world, org):
            return
        c = self.charter(st, org)
        r = self.rule_for(ctx.world, c, "tool", tool, args)
        if r is None:
            return
        proc = c["procedures"][r["procedure"]]
        if proc["type"] == "officer":
            title = self.office_of(st, org, ctx.origin, proc["office"], tool)
            if title is not None:
                self.spend_office(ctx, org, title, _amount(args))
                return
        raise Forbidden(f"{org}'s charter requires a {proc['type']} decision ({r['procedure']}) for {tool}; "
                        "use inst.propose", procedure=r["procedure"])

    # ------------------------------------------------------------------ offices
    def office_rows(self, st, org) -> dict[str, dict]:
        return {o["title"]: dict(o) for o in st.find("inst_offices", "org", org)}

    def office_of(self, st, org, person, office: str, tool: str | None = None) -> str | None:
        """The office `person` holds that satisfies an officer procedure naming `office` ('*' = any office)."""
        rows = self.office_rows(st, org)
        for title in sorted(rows):
            o = rows[title]
            if o["holder"] != person or (office != "*" and title != office):
                continue
            if ident(self.world).role(st, org, person) is None:
                continue                                       # holders who left hold nothing
            if tool is not None and not any(_tool_match(p, tool) for p in o["tools"]):
                continue
            return title
        return None

    def spend_office(self, ctx, org, title, amount: int) -> None:
        if amount <= 0:
            return
        o = ctx.state.get("inst_offices", f"{org}|{title}")
        key = f"{org}|{title}|{ctx.now // MONTH}"
        used = (ctx.state.get("inst_office_spend", key) or {}).get("used", 0)
        if used + amount > o["budget_cents_month"]:
            raise Forbidden(f"office {title} of {org}: monthly budget exhausted ({used}/{o['budget_cents_month']}); "
                            "use inst.propose")
        ctx.emit("inst.office_spend", {"id": key, "amount": int(amount)}, vis=(org, o["holder"]))

    # ================================================================== electorates and weights
    def contribution_of(self, st, org, who) -> int:
        return sum(int(r["cents"]) for r in st.find("inst_contributions", "contributor", who) if r["org"] == org)

    def electorate(self, world, org: str, proc: dict) -> dict[str, float]:
        st = world.state
        el = proc["electorate"]
        if proc["type"] == "officer":
            rows = self.office_rows(st, org)
            return {o["holder"]: 1.0 for t, o in sorted(rows.items())
                    if o["holder"] and (proc["office"] in ("*", t)) and ident(world).role(st, org, o["holder"])}
        if org.startswith("prj_"):
            joined = {r["partner"]: r["t"] for r in st.find("inst_partners", "project", org)
                      if r["status"] == "joined"}
            w = proc["weights"]
            out = {}
            for p in sorted(joined):
                if w == "contribution":
                    x = float(sum(r["cents"] for r in st.find("inst_contributions", "project", org)
                                  if r["contributor"] == p))
                elif w == "tenure":
                    x = float(1 + (world.now - joined[p]) // DAY)
                else:
                    x = 1.0
                if x > 0:
                    out[p] = x
            return out
        if el == "shareholders":
            base = sorted({s["holder"] for s in st.find("shares", "org", org) if s["n"] > 0}) \
                if "shares" in st.tables else []
        elif el.startswith("unit:"):
            base = sorted(m["person"] for m in st.find("inst_unit_members", "unit", el[5:]))
        else:
            roles = {"members": ("owner", "admin", "member"), "admins": ("owner", "admin"), "owners": ("owner",),
                     "partners": ("owner", "admin", "member")}[el]
            base = sorted(m["person"] for m in st.find("members", "org", org) if m["role"] in roles)
        w = proc["weights"]
        out = {}
        for p in base:
            if w == "equal":
                x = 1.0
            elif w == "shares":
                rec = st.get("shares", f"{org}|{p}") if "shares" in st.tables else None
                x = float(rec["n"]) if rec else 0.0
            elif w == "contribution":
                x = float(self.contribution_of(st, org, p))
            else:
                m = st.get("members", f"{org}|{p}")
                x = float(1 + (world.now - m["since"]) // DAY) if m else 0.0
            if x > 0:
                out[p] = x
        return out

    # ================================================================== action validation
    def validate_action(self, ctx: Ctx, org: str, action) -> dict:
        """Normalise and check an action an institution proposes. Anything outside the closed set of institutional
        actions, any operator tool and any malformed tool call is refused here, before anyone votes."""
        st = ctx.state
        if not isinstance(action, dict):
            raise InvalidAction("an action is an object with a kind")
        kind = action.get("kind")
        R = lambda x: ident(ctx.world).resolve_handle(st, x)  # noqa: E731
        venture = org.startswith("prj_")
        if venture and kind not in VENTURE_KINDS:
            raise InvalidAction(f"a venture decides {', '.join(VENTURE_KINDS)}")
        if kind not in ACTION_KINDS and kind not in VENTURE_KINDS:
            raise InvalidAction(f"{kind!r} is not an institutional action: institutions can only take the actions "
                                f"{', '.join(ACTION_KINDS)} - they cannot write world state directly",
                                law="world state")
        try:
            if kind == "tool":
                tool = action.get("tool")
                if not isinstance(tool, str) or len(tool) > 80:
                    raise InvalidAction("tool: a tool name")
                try:
                    ep = ctx.world.resolve(tool, org)
                except UnknownTool:
                    raise InvalidAction(f"unknown tool {tool!r}")
                if ep.host == "admin":
                    raise Forbidden("operator tools are outside every institution's reach", law="security boundary")
                if ep.read_only:
                    raise InvalidAction(f"{tool} is read-only; there is nothing to decide")
                args = action.get("args") or {}
                if not isinstance(args, dict):
                    raise InvalidAction("args: an object")
                full = S.with_defaults(ep.params, args)
                errs = S.validate(ep.params, full)
                if errs:
                    raise InvalidAction("; ".join(errs[:4]))
                return {"kind": kind, "tool": tool, "args": args}
            if kind == "amend":
                return {"kind": kind, "charter": self.validate_charter(ctx, org, action.get("charter"))}
            if kind == "admit":
                p = R(action["person"])
                if not p.startswith("p:"):
                    raise InvalidAction("only persons can be members")
                role = action.get("role", "member")
                if role not in ("member", "admin"):
                    raise InvalidAction("admit as member or admin (ownership is accepted, never assigned)")
                return {"kind": kind, "person": p, "role": role}
            if kind == "expel":
                p = R(action["person"])
                if ident(ctx.world).role(st, org, p) in (None, "owner"):
                    raise InvalidAction("expel a non-owner member")
                return {"kind": kind, "person": p}
            if kind == "role":
                p = R(action["person"])
                role = action.get("role")
                if role not in ("member", "admin"):
                    raise InvalidAction("a decision sets member or admin roles; ownership must be accepted "
                                        "(id.invite with role=owner)", law="security boundary")
                if ident(ctx.world).role(st, org, p) in (None, "owner"):
                    raise InvalidAction("change the role of a non-owner member")
                return {"kind": kind, "person": p, "role": role}
            if kind == "unit_create":
                head = R(action["head"]) if action.get("head") else None
                if head is not None and ident(ctx.world).role(st, org, head) is None:
                    raise InvalidAction("the head of a unit must be a member")
                parent = action.get("parent")
                if parent is not None and (not st.has("inst_units", parent) or st.tables["inst_units"][parent]["org"] != org):
                    raise InvalidAction("parent unit not found")
                return {"kind": kind, "name": _str(action["name"], NAME, "unit name"), "head": head,
                        "parent": parent,
                        "budget_cents_month": _int(action.get("budget_cents_month", 0), 0, MAX_AMOUNT, "budget")}
            if kind == "unit_assign":
                u = self._unit(st, org, action["unit"])
                p = R(action["person"])
                if ident(ctx.world).role(st, org, p) is None:
                    raise InvalidAction("only members join units")
                return {"kind": kind, "unit": u, "person": p, "remove": bool(action.get("remove", False))}
            if kind == "unit_dissolve":
                return {"kind": kind, "unit": self._unit(st, org, action["unit"])}
            if kind == "office":
                title = _str(action["title"], NAME, "office title")
                o = self.validate_office(ctx, org, title, {k: action[k] for k in ("holder", "budget_cents_month",
                                                                                   "tools") if k in action})
                return {"kind": kind, "title": title, **o}
            if kind == "office_remove":
                title = _str(action["title"], NAME, "office title")
                if not st.has("inst_offices", f"{org}|{title}"):
                    raise InvalidAction("no such office")
                return {"kind": kind, "title": title}
            if kind == "allocate":
                if venture:
                    action = dict(action, project=action.get("project", org))
                prj = self._project(st, action["project"])
                if not venture and prj["owner"] != org:
                    raise InvalidAction("allocate to a project of this organisation")
                if venture and action["project"] != org:
                    raise InvalidAction("a venture allocates within itself")
                amount = _int(action["amount"], 1, MAX_AMOUNT, "amount")
                task = action.get("task")
                if venture and task is None:
                    raise InvalidAction("a venture allocates its pool to tasks (partners fund it with "
                                        "inst.project_fund)")
                if task is not None:
                    t = st.get("inst_tasks", task)
                    if t is None or t["project"] != action["project"] or t["status"] in ("done", "cancelled"):
                        raise InvalidAction("allocate to an open task of the project")
                acct = None
                if not venture and task is None:
                    acct = action.get("account") or ctx.world.modules["bank"].primary_account(st, org)
                    if acct is None or ctx.world.modules["bank"].account_owner(st, acct) != org:
                        raise InvalidAction("name an account of the organisation to fund the project from")
                return {"kind": kind, "project": action["project"], "task": task, "amount": amount, "account": acct}
            if kind == "project_close":
                if venture:
                    action = dict(action, project=action.get("project", org))
                prj = self._project(st, action["project"])
                if (venture and action["project"] != org) or (not venture and prj["owner"] != org):
                    raise InvalidAction("close a project of this organisation")
                return {"kind": kind, "project": action["project"]}
            if kind == "admit_partner":
                p = R(action["partner"])
                return {"kind": kind, "partner": p}
            if kind == "merge":
                into = R(action["into"])
                if not into.startswith("o:") or into == org or not st.has("orgs", into) \
                        or st.tables["orgs"][into]["status"] != "active":
                    raise InvalidAction("merge into another active organisation")
                return {"kind": kind, "into": into}
            if kind == "absorb":
                other = R(action["org"])
                if not st.has("inst_merge_offers", other) or st.tables["inst_merge_offers"][other]["into"] != org:
                    raise InvalidAction(f"{other} has not decided to merge into {org}")
                return {"kind": kind, "org": other}
            if kind == "spin_off":
                u = self._unit(st, org, action["unit"])
                handle = _str(action["handle"], HANDLE, "handle")
                if st.find_ids("orgs", "handle", handle) or st.find_ids("persons", "handle", handle):
                    raise Conflict(f"handle {handle} taken")
                stake = _num(action.get("parent_stake", 0.0), 0.0, 0.49, "parent_stake")
                return {"kind": kind, "unit": u, "handle": handle,
                        "name": str(action.get("name") or handle)[:120],
                        "budget_cents": _int(action.get("budget_cents", 0), 0, MAX_AMOUNT, "budget_cents"),
                        "parent_stake": stake}
            if kind == "dissolve":
                return {"kind": kind}
        except KeyError as e:
            raise InvalidAction(f"{kind}: missing {e}")
        raise InvalidAction(f"unsupported action {kind!r}")

    def _unit(self, st, org, uid) -> str:
        u = st.get("inst_units", uid) if isinstance(uid, str) else None
        if u is None or u["org"] != org or u["status"] != "active":
            raise InvalidAction("no such active unit of this organisation")
        return uid

    def _project(self, st, pid):
        p = st.get("inst_projects", pid) if isinstance(pid, str) else None
        if p is None:
            raise NotFound("no such project")
        if p["status"] != "open":
            raise Conflict("the project is closed")
        return p

    # ================================================================== proposing, voting, deciding
    def may_propose(self, st, org, who) -> bool:
        idm = ident(self.world)
        if org.startswith("prj_"):
            for r in st.find("inst_partners", "project", org):
                if r["status"] == "joined" and (r["partner"] == who or (r["partner"].startswith("o:") and
                                                idm.role(st, r["partner"], who) in ("owner", "admin"))):
                    return True
            return False
        if idm.role(st, org, who) is not None:
            return True
        return "shares" in st.tables and bool((st.get("shares", f"{org}|{who}") or {}).get("n"))

    def propose(self, ctx: Ctx, org: str, action: dict, reason: str = "", proposer: str | None = None,
                applicant: bool = False) -> dict:
        st = ctx.state
        proposer = proposer or ctx.actor
        if org.startswith("prj_"):
            prj = self._project(st, org)
            if not prj["procedure"]:
                raise InvalidAction("this project is run by its lead; it has no decision procedure")
        elif not st.has("orgs", org) or st.tables["orgs"][org]["status"] != "active":
            raise NotFound("no active organisation " + org)
        if not applicant and not self.may_propose(st, org, proposer):
            raise Forbidden("only members (or shareholders, or venture partners) can propose")
        a = self.validate_action(ctx, org, action)
        if org.startswith("prj_"):
            pname, proc = "venture", dict(st.tables["inst_projects"][org]["procedure"])
        else:
            pname, proc = self.procedure_for(ctx.world, org, a["kind"], a.get("tool"), a.get("args", a))
        el = self.electorate(ctx.world, org, proc)
        did = st.new_id("dec")
        closes = ctx.now + int(proc["window_s"])
        ctx.emit("inst.proposed", {"id": did, "org": org, "action": a, "procedure": proc, "proc_name": pname,
                                   "electorate": [[p, el[p]] for p in sorted(el)], "by": proposer,
                                   "reason": str(reason)[:2000], "closes": closes},
                 vis=tuple(sorted(set(el) | {proposer, org})))
        if not el:
            return self.tally(ctx, did, final=True)
        hook = ctx.world.hooks.get("on_inst_decision")
        if hook and not (proc["type"] == "officer" and proposer in el):
            for voter in sorted(el):
                if voter != proposer:
                    hook(ctx, did, voter)
        if proposer in el:
            # a proposer who is an elector votes for its own proposal (an officer thereby decides at once)
            return self.vote(ctx, did, proposer, "yes")
        return {"decision": did, "status": "open", "closes": closes}

    def vote(self, ctx: Ctx, did: str, voter: str, choice: str) -> dict:
        st = ctx.state
        d = st.get("inst_decisions", did)
        if d is None or d["status"] != "open":
            raise NotFound("no open decision")
        el = dict((p, w) for p, w in d["electorate"])
        if voter not in el:
            raise Forbidden("not in this decision's electorate")
        if choice not in ("yes", "no"):
            raise InvalidAction("vote yes or no")
        ctx.emit("inst.ballot", {"id": f"{did}|{voter}", "decision": did, "voter": voter, "choice": choice},
                 vis=tuple(sorted(set(el) | {d["org"]})))
        return self.tally(ctx, did, final=False)

    def counts(self, st, d) -> tuple[float, float, float]:
        """(yes, no, total) weight of a decision, resolving vote delegations for delegated procedures."""
        el = dict((p, w) for p, w in d["electorate"])
        total = sum(el.values())
        ballots = {b["voter"]: b["choice"] for b in st.find("inst_ballots", "decision", d["id"])}
        yes = no = 0.0
        if d["procedure"]["type"] == "delegated" and not d["org"].startswith("prj_"):
            for p in sorted(el):
                seen, cur = set(), p
                while cur not in ballots:
                    seen.add(cur)
                    row = st.get("inst_delegations", f"{d['org']}|{cur}")
                    nxt = row["delegate"] if row else None
                    if nxt is None or nxt in seen or nxt not in el:
                        cur = None
                        break
                    cur = nxt
                if cur is None:
                    continue
                if ballots[cur] == "yes":
                    yes += el[p]
                else:
                    no += el[p]
        else:
            for p, c in ballots.items():
                if p in el:
                    if c == "yes":
                        yes += el[p]
                    else:
                        no += el[p]
        return yes, no, total

    def tally(self, ctx: Ctx, did: str, final: bool) -> dict:
        st = ctx.state
        d = dict(st.tables["inst_decisions"][did], id=did)
        proc = d["procedure"]
        yes, no, total = self.counts(st, d)
        t = proc["type"]
        thr, quorum = float(proc["threshold"]), float(proc["quorum"])
        decided = None
        if total <= 0:
            decided = False
        elif t == "officer":
            decided = True if yes > 0 else (False if no > 0 else None)
        elif t == "unanimous":
            decided = True if yes >= total else (False if no > 0 else None)
        elif t == "consensus":
            if no > 0:
                decided = False
            elif yes >= total:
                decided = True
            elif final:
                decided = (yes + no) / total >= quorum
        else:
            if yes / total > thr and (yes + no) / total >= quorum:
                decided = True
            elif no / total >= 1 - thr and thr < 1.0:
                decided = False
            elif final:
                cast = yes + no
                decided = bool(cast > 0 and yes / cast > thr and cast / total >= quorum)
        if decided is None and final:
            decided = False
        if decided is True:
            self.execute(ctx, did)
        elif decided is False:
            ctx.emit("inst.decided", {"id": did, "status": "rejected" if (yes + no) > 0 or t == "officer"
                                      else "lapsed", "result": {"yes": yes, "no": no, "total": total}},
                     vis=(d["org"], d["by"]))
        return {"decision": did, "yes": yes, "no": no, "electorate": total,
                "status": st.tables["inst_decisions"][did]["status"]}

    def execute(self, ctx: Ctx, did: str) -> None:
        st = ctx.state
        d = st.tables["inst_decisions"][did]
        org, a = d["org"], d["action"]
        stack = ctx.world.runtime.setdefault("inst_exec", [])
        stack.append(org)
        try:
            if org.startswith("prj_"):
                res = ctx.run_as(ctx.actor, lambda sub: self.apply_venture(sub, org, a))
            elif a["kind"] == "tool":
                res = ctx.run_as(org, self._run_tool, org, a, did)
            else:
                res = ctx.run_as(org, self._run_action, org, a, did)
            ctx.emit("inst.decided", {"id": did, "status": "executed", "result": res}, vis=(org, d["by"]))
        except CrucibleError as e:
            ctx.emit("inst.decided", {"id": did, "status": "failed", "result": e.as_dict()}, vis=(org, d["by"]))
        finally:
            stack.pop()

    def _officer_spend(self, sub: Ctx, org: str, did: str, tool: str | None, amount: int) -> None:
        """An officer's decision spends from the approving officer's monthly budget, exactly as the officer's own
        direct action would."""
        d = sub.state.tables["inst_decisions"][did]
        if d["procedure"]["type"] != "officer" or amount <= 0:
            return
        yes = sorted(b["voter"] for b in sub.state.find("inst_ballots", "decision", did) if b["choice"] == "yes")
        for officer in yes:
            title = self.office_of(sub.state, org, officer, d["procedure"]["office"], tool)
            if title is not None:
                self.spend_office(sub, org, title, amount)
                return
        raise Forbidden("no approving officer's office covers this action")

    def _run_tool(self, sub: Ctx, org: str, a: dict, did: str) -> object:
        self._officer_spend(sub, org, did, a["tool"], _amount(a["args"]))
        return sub.call(a["tool"], a["args"])

    def _run_action(self, sub: Ctx, org: str, a: dict, did: str) -> object:
        # institutional actions are governed like tools named `inst.<kind>` by a 4.0 gov constitution, if the
        # organisation has one: where both institutions govern an action, both must be satisfied
        gov = sub.world.modules.get("gov")
        if gov is not None:
            gov.check(sub, org, f"inst.{a['kind']}", a)
        self._officer_spend(sub, org, did, None, _amount(a))
        return self.apply_action(sub, org, a, did)

    # ================================================================== institutional actions
    def apply_action(self, ctx: Ctx, org: str, a: dict, did: str):
        st = ctx.state
        idm = ident(ctx.world)
        k = a["kind"]
        if k == "amend":
            ctx.emit("inst.charter", {"org": org, "charter": a["charter"], "by": did}, vis="public")
            self._sync_offices(ctx, org, a["charter"]["offices"])
            return {"amended": True}
        if k == "admit":
            c = self.charter(st, org)
            cap = c["membership"]["max_members"] if c else 10_000
            if len(st.find_ids("members", "org", org)) >= cap:
                raise Conflict("the organisation is at its membership limit")
            if st.has("members", f"{org}|{a['person']}"):
                raise Conflict("already a member")
            if st.tables["inst_decisions"][did]["by"] == a["person"]:
                # the person applied (inst.join): its consent is the application
                ident(ctx.world).add_member(ctx, org, a["person"], a["role"], "")
                return {"joined": a["person"]}
            iid = st.new_id("inv")
            ctx.emit("id.invite", {"id": iid, "org": org, "person": a["person"], "role": a["role"],
                                   "title": "", "by": did}, vis=(a["person"], org))
            return {"invite": iid}
        if k == "expel":
            idm.remove_member(ctx, org, a["person"], f"expelled by decision {did}")
            for gid in st.find_ids("grants", "grantee", a["person"]):
                g = st.tables["grants"][gid]
                if g["owner"] == org and g["status"] == "active":
                    ctx.emit("id.revoke", {"id": gid}, vis=(a["person"], org))
            self.on_member_left(ctx, org, a["person"])
            return {"expelled": a["person"]}
        if k == "role":
            ctx.emit("id.role", {"org": org, "person": a["person"], "role": a["role"]}, vis=(a["person"], org))
            return {"role": a["role"]}
        if k == "unit_create":
            uid = st.new_id("unit")
            ctx.emit("inst.unit", {"id": uid, "org": org, "name": a["name"], "head": a["head"], "parent": a["parent"]},
                     vis=(org,) + ((a["head"],) if a["head"] else ()))
            if a["head"]:
                ctx.emit("inst.unit_member", {"unit": uid, "person": a["head"], "remove": False}, vis=(org, a["head"]))
                ctx.emit("inst.office", {"org": org, "title": f"unit:{uid}", "holder": a["head"],
                                         "budget_cents_month": a["budget_cents_month"], "tools": ["*"], "unit": uid},
                         vis=(org, a["head"]))
            return {"unit": uid}
        if k == "unit_assign":
            ctx.emit("inst.unit_member", {"unit": a["unit"], "person": a["person"], "remove": a["remove"]},
                     vis=(org, a["person"]))
            return {"ok": True}
        if k == "unit_dissolve":
            self._dissolve_unit(ctx, org, a["unit"])
            return {"ok": True}
        if k == "office":
            ctx.emit("inst.office", {"org": org, "title": a["title"], "holder": a["holder"],
                                     "budget_cents_month": a["budget_cents_month"], "tools": a["tools"],
                                     "unit": None}, vis=(org,) + ((a["holder"],) if a["holder"] else ()))
            return {"office": a["title"]}
        if k == "office_remove":
            ctx.emit("inst.office_removed", {"org": org, "title": a["title"]}, vis=(org,))
            return {"ok": True}
        if k in ("allocate", "project_close"):
            return self.apply_venture(ctx, org, a)
        if k == "merge":
            ctx.emit("inst.merge_offer", {"org": org, "into": a["into"], "decision": did}, vis=(org, a["into"]))
            return {"offered": a["into"]}
        if k == "absorb":
            return self._absorb(ctx, org, a["org"], did)
        if k == "spin_off":
            return self._spin_off(ctx, org, a, did)
        if k == "dissolve":
            return self.dissolve(ctx, org, f"decision {did}")
        raise InvalidAction(f"unsupported action {k!r}")

    def apply_venture(self, ctx: Ctx, org: str, a: dict):
        k = a["kind"]
        if k == "allocate":
            if a["task"] is None:
                # money from the organisation's own account into the project's pool (an org-level decision)
                self.fund(ctx, a["project"], org, a["account"], a["amount"])
                return {"funded": a["amount"]}
            self.allocate(ctx, a["project"], a["task"], a["amount"])
            return {"allocated": a["amount"], "task": a["task"]}
        if k == "project_close":
            return self.close_project(ctx, a["project"])
        if k == "admit_partner":
            self.invite_partner(ctx, org, a["partner"])
            return {"invited": a["partner"]}
        raise InvalidAction(f"unsupported venture action {k!r}")

    def _sync_offices(self, ctx, org, offices: dict) -> None:
        st = ctx.state
        cur = {o["title"]: o for o in st.find("inst_offices", "org", org) if not o.get("unit")}
        for title in sorted(cur):
            if title not in offices:
                ctx.emit("inst.office_removed", {"org": org, "title": title}, vis=(org,))
        for title in sorted(offices):
            o = offices[title]
            ctx.emit("inst.office", {"org": org, "title": title, "holder": o["holder"],
                                     "budget_cents_month": o["budget_cents_month"], "tools": o["tools"],
                                     "unit": None}, vis=(org,) + ((o["holder"],) if o["holder"] else ()))

    def _dissolve_unit(self, ctx, org, uid) -> None:
        st = ctx.state
        for m in st.find("inst_unit_members", "unit", uid):
            ctx.emit("inst.unit_member", {"unit": uid, "person": m["person"], "remove": True}, vis=(org, m["person"]))
        for cid in sorted(k for k, u in st.tables["inst_units"].items() if u.get("parent") == uid and u["status"] == "active"):
            self._dissolve_unit(ctx, org, cid)
        if st.has("inst_offices", f"{org}|unit:{uid}"):
            ctx.emit("inst.office_removed", {"org": org, "title": f"unit:{uid}"}, vis=(org,))
        ctx.emit("inst.unit_status", {"id": uid, "status": "dissolved"}, vis=(org,))

    def on_member_left(self, ctx, org, person) -> None:
        """A member who leaves (or is removed) gives up its offices, unit seats and vote delegations."""
        st = ctx.state
        for o in st.find("inst_offices", "holder", person):
            if o["org"] == org:
                ctx.emit("inst.office", {"org": org, "title": o["title"], "holder": None,
                                         "budget_cents_month": o["budget_cents_month"], "tools": o["tools"],
                                         "unit": o.get("unit")}, vis=(org,))
        for m in st.find("inst_unit_members", "person", person):
            if st.tables["inst_units"][m["unit"]]["org"] == org:
                ctx.emit("inst.unit_member", {"unit": m["unit"], "person": person, "remove": True}, vis=(org, person))
        if st.has("inst_delegations", f"{org}|{person}"):
            ctx.emit("inst.delegation", {"org": org, "member": person, "delegate": None}, vis=(org, person))

    # ------------------------------------------------------------------ restructuring: merge, spin-off, dissolve
    def _absorb(self, ctx, into: str, other: str, did: str) -> dict:
        """`other` decided to merge into `into`, and `into` decided to absorb it: `other`'s members are invited
        (membership is accepted), its money and its projects pass to `into`, its offices and units end, and it
        stops existing as a separate organisation. A registered company that still owes anything cannot be
        absorbed this way (its creditors must be paid first)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        biz = ctx.world.modules.get("business")
        owed = self.owed_by(st, other)
        if owed:
            raise Conflict(f"{other} owes {owed} cents; settle its debts before it is absorbed")
        self._settle_outside(ctx, other, f"merged into {into}")
        moved = 0
        dst = bank.primary_account(st, into)
        for aid in st.find_ids("accounts", "owner", other):
            a = st.tables["accounts"][aid]
            if a["status"] == "open" and a["balance"] > 0:
                if dst is None:
                    raise Conflict(f"{into} has no account to receive {other}'s money")
                bank.move(ctx, aid, dst, a["balance"], "merger", f"{other} merged into {into}", ref=did)
                moved += a["balance"]
        for pid in st.find_ids("inst_projects", "owner", other):
            if st.tables["inst_projects"][pid]["status"] == "open":
                ctx.emit("inst.project_owner", {"id": pid, "owner": into}, vis=(other, into))
        for uid in st.find_ids("inst_units", "org", other):
            if st.tables["inst_units"][uid]["status"] == "active":
                self._dissolve_unit(ctx, other, uid)
        invited = []
        for m in st.find("members", "org", other):
            if not st.has("members", f"{into}|{m['person']}"):
                iid = st.new_id("inv")
                ctx.emit("id.invite", {"id": iid, "org": into, "person": m["person"], "role": "member",
                                       "title": m["title"], "by": did}, vis=(m["person"], into))
                invited.append(m["person"])
        for m in st.find("members", "org", other):
            ident(ctx.world).remove_member(ctx, other, m["person"], f"merged into {into}")
        for o in st.find("inst_offices", "org", other):
            ctx.emit("inst.office_removed", {"org": other, "title": o["title"]}, vis=(other,))
        if biz is not None and biz.is_registered(st, other):
            for aid in st.find_ids("accounts", "owner", other):
                if st.tables["accounts"][aid]["status"] == "open":
                    ctx.emit("bank.account_status", {"id": aid, "status": "frozen"}, vis=(other,))
            ctx.emit("biz.status", {"org": other, "status": "dissolved", "cash": 0, "owed": 0}, vis="public")
        ctx.emit("inst.merged", {"org": other, "into": into, "moved": moved, "invited": invited}, vis="public")
        ctx.emit("id.org_status", {"id": other, "status": "merged"}, vis="public")
        return {"absorbed": other, "moved_cents": moved, "invited": invited}

    def _spin_off(self, ctx, org: str, a: dict, did: str) -> dict:
        """A unit becomes an organisation of its own: its head founds it (the head proposed the spin-off, so the
        ownership is accepted), its members are invited, and - for a registered company with an account - the
        parent may fund it and keep a minority stake."""
        st = ctx.state
        idm = ident(ctx.world)
        u = st.tables["inst_units"][a["unit"]]
        head = u["head"]
        d = st.tables["inst_decisions"][did]
        if head is None or d["by"] != head:
            raise Forbidden("a spin-off is proposed by the unit's head, who founds the new organisation")
        new = idm.create_org(ctx, a["handle"], a["name"], head, "company" if a["budget_cents"] else "group")
        biz = ctx.world.modules.get("business")
        bank = ctx.world.modules["bank"]
        funded = 0
        if a["budget_cents"]:
            src = bank.primary_account(st, org)
            if biz is None or src is None:
                raise Conflict("funding a spin-off needs a company registry and an account of the parent")
            biz.register_company(ctx, new, head, "spin-off", pay_from=src)     # the ordinary fee, paid by the parent
            acct = bank.open_account(ctx, new, "checking")
            bank.move(ctx, src, acct, a["budget_cents"], "investment", f"spin-off {new} of {org}", ref=did)
            funded = a["budget_cents"]
            if a["parent_stake"] > 0:
                from ..services.business import SHARES
                biz.transfer_shares(ctx, new, head, org, int(SHARES * a["parent_stake"]))
        invited = []
        for m in st.find("inst_unit_members", "unit", a["unit"]):
            if m["person"] != head:
                iid = st.new_id("inv")
                ctx.emit("id.invite", {"id": iid, "org": new, "person": m["person"], "role": "member",
                                       "title": "", "by": did}, vis=(m["person"], new))
                invited.append(m["person"])
        self._dissolve_unit(ctx, org, a["unit"])
        ctx.emit("inst.spun_off", {"org": org, "unit": a["unit"], "new": new, "funded": funded}, vis="public")
        return {"org": new, "funded": funded, "invited": invited}

    def dissolve(self, ctx, org: str, why: str) -> dict:
        """Wind an organisation up: its projects close (pools return to contributors), units and offices end; a
        registered company goes through the company registry's winding up (creditors first, then shareholders;
        refused if it cannot pay what it owes); any other organisation's money is distributed by the charter's
        basis, and its members are released."""
        st = ctx.state
        for pid in st.find_ids("inst_projects", "owner", org):
            if st.tables["inst_projects"][pid]["status"] == "open":
                self.close_project(ctx, pid)
        for uid in st.find_ids("inst_units", "org", org):
            if st.tables["inst_units"][uid]["status"] == "active":
                self._dissolve_unit(ctx, org, uid)
        for o in st.find("inst_offices", "org", org):
            ctx.emit("inst.office_removed", {"org": org, "title": o["title"]}, vis=(org,))
        biz = ctx.world.modules.get("business")
        if biz is not None and biz.is_registered(st, org):
            commit = ctx.world.modules.get("commit")
            if commit is not None:
                commit.settle_for_wind_up(ctx, org)
            biz._wind_up(ctx, org, "dissolved")        # creditors, then shareholders; refused if insolvent
            ctx.emit("inst.dissolved", {"org": org, "why": why[:200], "via": "company registry", "paid": 0},
                     vis="public")
            return {"dissolved": org, "via": "company registry"}
        bank = ctx.world.modules["bank"]
        self._settle_outside(ctx, org, "dissolved")
        accts = [a for a in st.find_ids("accounts", "owner", org) if st.tables["accounts"][a]["status"] == "open"]
        cash = sum(max(0, st.tables["accounts"][a]["balance"]) for a in accts)
        claims = [(cid, c) for cid in st.find_ids("claims", "debtor", org) for c in [st.tables["claims"][cid]]
                  if c["status"] == "open" and isinstance(c["amount"], int) and c["amount"] > 0] \
            if "claims" in st.tables else []
        owed = sum(c["amount"] for _, c in claims)
        if owed > cash:
            raise Conflict(f"{org} owes {owed} cents and holds {cash}: an organisation that cannot pay what it "
                           "owes cannot dissolve itself")
        for cid, cl in claims:                               # creditors first
            due = cl["amount"]
            dst = bank.primary_account(st, cl["creditor"]) if cl["creditor"] != SYSTEM else "acct_fees"
            for a in accts:
                x = min(st.tables["accounts"][a]["balance"], due)
                if x > 0 and dst is not None:
                    bank.move(ctx, a, dst, x, "claim", f"claim {cid} on dissolution of {org}")
                    due -= x
            ctx.emit("contracts.claim_status", {"id": cid, "status": "settled", "paid": cl["amount"]},
                     vis=(org, cl["creditor"]))
        c = self.charter(st, org)
        basis = (c or {}).get("dissolution", {}).get("distribution", "equal")
        members = sorted(m["person"] for m in st.find("members", "org", org))
        if basis == "contribution":
            w = {p: self.contribution_of(st, org, p) for p in members}
        elif basis == "shares" and "shares" in st.tables:
            w = {s["holder"]: s["n"] for s in st.find("shares", "org", org) if s["n"] > 0}
        else:
            w = {p: 1 for p in members}
        w = {p: x for p, x in w.items() if x > 0 and bank.primary_account(st, p) is not None}
        if not w:
            w = {p: 1 for p in members if bank.primary_account(st, p) is not None}
        tot = sum(w.values())
        paid = 0
        for aid in accts:
            left = max(0, st.tables["accounts"][aid]["balance"])
            names = sorted(w)
            for i, who in enumerate(names):
                share = left * w[who] // tot if i < len(names) - 1 else \
                    max(0, st.tables["accounts"][aid]["balance"])      # the last takes the rounding remainder
                dst = bank.primary_account(st, who)
                if share > 0 and dst != aid:
                    bank.move(ctx, aid, dst, share, "distribution", f"dissolution of {org}")
                    paid += share
            ctx.emit("bank.account_status", {"id": aid, "status": "frozen"}, vis=(org,))
        for m in st.find("members", "org", org):
            ident(ctx.world).remove_member(ctx, org, m["person"], "dissolved")
        ctx.emit("inst.dissolved", {"org": org, "why": why[:200], "via": basis, "paid": paid}, vis="public")
        ctx.emit("id.org_status", {"id": org, "status": "dissolved"}, vis="public")
        return {"dissolved": org, "distributed_cents": paid}

    def _settle_outside(self, ctx, org: str, why: str) -> None:
        """Before an organisation's accounts close: money the world holds for or from it elsewhere (marketplace
        orders, bounties, escrowed commitments) is settled, and its open-ended contracts end."""
        st = ctx.state
        for name, fn in (("market", "settle_for_wind_up"), ("jobs", "cancel_for_wind_up"),
                         ("commit", "settle_for_wind_up")):
            m = ctx.world.modules.get(name)
            if m is not None:
                getattr(m, fn)(ctx, org)
        contracts = ctx.world.modules["contracts"]
        for cid, i, creditor, amount, what in self.fixed_obligations(st, org):
            contracts.add_claim(ctx, org, creditor, amount, f"contract {cid} clause {i}: {what} due ({why})")
        for row in st.find("contract_parties", "party", org):
            k = st.tables["contracts"][row["contract"]]
            if k["status"] in ("active", "terminating", "proposed"):
                contracts.end(ctx, row["contract"], "terminated", f"{org} {why}")

    @staticmethod
    def fixed_obligations(st, org) -> list[tuple[str, int, str, int, str]]:
        """What `org` owes in a definite amount under active 4.0 contracts: remaining instalments of payments with
        a fixed number of instalments, and penalties of deliverables it promised and has not delivered (the same
        rule the company registry applies on winding up)."""
        from ..services.contracts import remaining_fixed
        out = []
        for row in st.find("contract_parties", "party", org):
            k = st.tables["contracts"][row["contract"]]
            if k["status"] not in ("active", "terminating"):
                continue
            for i, cl in enumerate(k["clauses"]):
                if cl["type"] == "payment" and cl["from"] == org:
                    rem = remaining_fixed(k, i)
                    if type(rem) is int and rem > 0:
                        out.append((row["contract"], i, cl["to"], rem, "remaining payments"))
                elif cl["type"] == "deliverable" and cl["by"] == org and cl.get("penalty") \
                        and str(i) not in k["delivered"]:
                    pen = cl["penalty"]
                    if pen["from"] == org and pen["to"] in k["parties"] and pen["to"] != org \
                            and type(pen["amount"]) is int and pen["amount"] > 0:
                        out.append((row["contract"], i, pen["to"], pen["amount"], "penalty"))
        return out

    def owed_by(self, st, org) -> int:
        open_claims = sum(c["amount"] for c in st.find("claims", "debtor", org)
                          if c["status"] == "open" and type(c["amount"]) is int and c["amount"] > 0)
        return open_claims + sum(x[3] for x in self.fixed_obligations(st, org))

    # ================================================================== projects (shared projects, ventures)
    def project_of(self, st, pid):
        return st.get("inst_projects", pid)

    def is_lead(self, ctx, prj, pid) -> bool:
        if ctx.actor == prj["lead"]:
            return True
        return ident(ctx.world).allowed(ctx, ctx.actor, prj["owner"], "inst.lead", 0, pid)

    def fund(self, ctx, pid, contributor: str, account: str, amount: int) -> None:
        st = ctx.state
        bank = ctx.world.modules["bank"]
        prj = self._project(st, pid)
        bank.move(ctx, account, ESCROW, amount, "contribution", f"contribution to {pid}", ref=pid)
        n = st.next_counter(f"contrib:{pid}")
        ctx.emit("inst.contribution", {"id": f"{pid}|{contributor}|{n}", "project": pid, "org": prj["owner"],
                                       "contributor": contributor, "cents": int(amount), "kind": "money",
                                       "account": account}, vis=tuple(sorted({contributor, prj["owner"]})))

    def record_labor(self, ctx, org: str, person: str, cents: int, project: str | None = None) -> None:
        """Labour contributed to an organisation, valued at the worker's wage (used for contribution weights and
        profit sharing by labour): called by the labour market when recorded hours are worked."""
        if cents <= 0:
            return
        n = ctx.state.next_counter(f"labor:{org}")
        ctx.emit("inst.contribution", {"id": f"labor|{org}|{person}|{n}", "project": project, "org": org,
                                       "contributor": person, "cents": int(cents), "kind": "labor", "account": None},
                 vis=(org, person))

    def allocate(self, ctx, pid, task, amount) -> None:
        st = ctx.state
        prj = self._project(st, pid)
        t = st.get("inst_tasks", task)
        if t is None or t["project"] != pid or t["status"] in ("done", "cancelled"):
            raise InvalidAction("allocate to an open task of the project")
        free = prj["pool"] - prj["allocated"]
        if amount > free:
            raise Conflict(f"the project's pool has {free} cents unallocated")
        ctx.emit("inst.allocated", {"project": pid, "task": task, "amount": int(amount)},
                 vis=self._pvis(st, pid))

    def _pvis(self, st, pid) -> tuple:
        prj = st.tables["inst_projects"][pid]
        out = {prj["owner"], prj["lead"]}
        out.update(r["partner"] for r in st.find("inst_partners", "project", pid))
        out.update(t["assignee"] for t in st.find("inst_tasks", "project", pid) if t["assignee"])
        return tuple(sorted(x for x in out if x))

    def invite_partner(self, ctx, pid, partner) -> None:
        st = ctx.state
        if st.has("inst_partners", f"{pid}|{partner}"):
            raise Conflict("already a partner or invited")
        ctx.emit("inst.partner", {"project": pid, "partner": partner, "status": "invited"},
                 vis=tuple(sorted(set(self._pvis(st, pid)) | {partner})))

    def close_project(self, ctx, pid) -> dict:
        """Close a project: open tasks are cancelled and what is left in its pool returns to the contributors of
        money, pro rata to what each contributed."""
        st = ctx.state
        prj = st.tables["inst_projects"][pid]
        bank = ctx.world.modules["bank"]
        rest = prj["pool"]
        money = {}
        accts = {}
        for r in st.find("inst_contributions", "project", pid):
            if r["kind"] == "money":
                money[r["contributor"]] = money.get(r["contributor"], 0) + r["cents"]
                accts.setdefault(r["contributor"], r["account"])
        tot = sum(money.values())
        refunded = 0
        for i, who in enumerate(sorted(money)):
            share = rest * money[who] // tot if tot else 0
            if i == len(money) - 1:
                share = rest - refunded                     # the last contributor takes the rounding remainder
            if share <= 0:
                continue
            dst = accts[who] if accts[who] and st.tables["accounts"][accts[who]]["status"] == "open" \
                else bank.primary_account(st, who)
            if dst is None:
                dst = "acct_fees"                           # nobody left to receive it: forfeited, on record
            bank.move(ctx, ESCROW, dst, share, "refund", f"closing {pid}", ref=pid)
            refunded += share
        for tid in st.find_ids("inst_tasks", "project", pid):
            if st.tables["inst_tasks"][tid]["status"] not in ("done", "cancelled"):
                ctx.emit("inst.task_status", {"id": tid, "status": "cancelled", "quality": None}, vis=self._pvis(st, pid))
        ctx.emit("inst.project_status", {"id": pid, "status": "closed", "refunded": refunded}, vis=self._pvis(st, pid))
        return {"closed": pid, "refunded": refunded}

    def complete_task(self, ctx, tid: str, quality: float | None, by: str) -> dict:
        """Record a task as done (by the labour market for simulated work, or on the lead's acceptance) and pay
        its allocated reward from the project's pool to whoever did it."""
        st = ctx.state
        t = st.tables["inst_tasks"][tid]
        prj = st.tables["inst_projects"][t["project"]]
        if t["status"] in ("done", "cancelled"):
            raise Conflict("task already finished")
        if not self.deps_done(st, tid):
            raise Conflict("the tasks this one depends on are not done")
        paid = 0
        bank = ctx.world.modules["bank"]
        if t["reward"] > 0 and t["assignee"]:
            dst = bank.primary_account(st, t["assignee"])
            if dst is not None:
                bank.move(ctx, ESCROW, dst, t["reward"], "project_reward", f"task {tid} of {t['project']}",
                          ref=t["project"])
                paid = t["reward"]
        ctx.emit("inst.task_status", {"id": tid, "status": "done", "by": by, "paid": paid,
                                      "quality": None if quality is None else round(float(quality), 4),
                                      "late": bool(prj["deadline"] is not None and ctx.now > prj["deadline"])},
                 vis=self._pvis(st, t["project"]))
        return {"task": tid, "paid": paid}

    def set_progress(self, ctx, tid: str, progress: float) -> None:
        st = ctx.state
        ctx.emit("inst.task_progress", {"id": tid, "progress": round(float(progress), 4)},
                 vis=self._pvis(st, st.tables["inst_tasks"][tid]["project"]))

    def unassign(self, ctx, tid: str, why: str) -> None:
        """Put an assigned task back to open (its assignee declined it or left); progress made so far is kept."""
        st = ctx.state
        t = st.tables["inst_tasks"][tid]
        if t["status"] in ("assigned", "submitted"):
            ctx.emit("inst.task_unassigned", {"id": tid, "from": t["assignee"], "why": why[:200]},
                     vis=self._pvis(st, t["project"]))

    def deps_done(self, st, tid) -> bool:
        t = st.tables["inst_tasks"][tid]
        return all(st.tables["inst_tasks"][d]["status"] == "done" for d in t["depends_on"])

    # ================================================================== endpoints: charters and decisions
    def _org_arg(self, ctx, org) -> str:
        o = ident(ctx.world).resolve_handle(ctx.state, org)
        if not o.startswith("o:"):
            raise InvalidAction("an organisation is required")
        return o

    @endpoint("adopt", S.obj({"org": S.s_str(maxLength=40), "charter": S.s_any_obj(
        "{procedures: {name: {type, electorate, threshold, quorum, window_s, weights, office}}, rules: [{match: "
        "{kind, tool, min_amount, when}, procedure}], default, membership: {admission, max_members}, offices: "
        "{title: {holder, budget_cents_month, tools}}, dissolution: {distribution}}")}, ["org", "charter"]), cost_s=120)
    def ep_adopt(self, ctx, org, charter):
        """Adopt an organisation's first charter (owners/admins; afterwards it changes only by an `amend` decision
        under the charter's own rules)."""
        st = ctx.state
        org = self._org_arg(ctx, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if st.has("inst_charters", org):
            raise Conflict("already chartered: amend it with inst.propose {kind: amend}")
        c = self.validate_charter(ctx, org, charter)
        ctx.emit("inst.charter", {"org": org, "charter": c, "by": ctx.actor}, vis="public")
        self._sync_offices(ctx, org, c["offices"])
        return {"ok": True, "procedures": sorted(c["procedures"]), "rules": len(c["rules"])}

    @endpoint("charter", S.obj({"org": S.s_str(maxLength=40)}, ["org"]), read_only=True)
    def ep_charter(self, ctx, org):
        """An organisation's charter, offices and units (public)."""
        st = ctx.state
        org = self._org_arg(ctx, org)
        rec = st.get("inst_charters", org)
        return {"org": org, "charter": dict(rec["charter"]) if rec else None, "version": rec["version"] if rec else 0,
                "offices": [{k: o[k] for k in ("title", "holder", "budget_cents_month", "tools", "unit")}
                            for o in st.find("inst_offices", "org", org)],
                "units": [dict(u, id=uid) for uid in st.find_ids("inst_units", "org", org)
                          for u in [st.tables["inst_units"][uid]] if u["status"] == "active"]}

    @endpoint("propose", S.obj({"org": S.s_str("organisation, or a project id for venture decisions", maxLength=40),
                                "action": S.s_any_obj("{kind, ...}: tool {tool, args} | amend {charter} | admit "
                                                      "{person, role} | expel {person} | role {person, role} | "
                                                      "unit_create {name, head, parent, budget_cents_month} | "
                                                      "unit_assign {unit, person, remove} | unit_dissolve {unit} | "
                                                      "office {title, holder, budget_cents_month, tools} | "
                                                      "office_remove {title} | allocate {project, task?, amount, "
                                                      "account?} | project_close {project} | admit_partner "
                                                      "{partner} | merge {into} | absorb {org} | spin_off {unit, "
                                                      "handle, name, budget_cents, parent_stake} | dissolve {}"),
                                "reason": S.s_str(maxLength=2000)}, ["org", "action"]), cost_s=60)
    def ep_propose(self, ctx, org, action, reason=""):
        """Propose a decision. It is decided by the procedure the charter assigns to this kind of action and
        executes (as the organisation) as soon as it passes; it is rejected or lapses otherwise."""
        o = org if org.startswith("prj_") else self._org_arg(ctx, org)
        return self.propose(ctx, o, action, reason)

    @endpoint("vote", S.obj({"decision": S.s_str(maxLength=40), "choice": S.s_enum(["yes", "no"]),
                             "as_org": S.s_str("vote for an organisation you administer (a shareholder or partner)",
                                               maxLength=40)}, ["decision", "choice"]))
    def ep_vote(self, ctx, decision, choice, as_org=None):
        """Vote on an open decision you are an elector of. A decision executes the moment it is reached."""
        voter = ctx.actor
        if as_org:
            voter = self._org_arg(ctx, as_org)
            ident(ctx.world).authorize(ctx, voter, "org.admin")
        return self.vote(ctx, decision, voter, choice)

    @endpoint("delegate_vote", S.obj({"org": S.s_str(maxLength=40),
                                      "to": S.s_str("member to delegate your vote to; empty to withdraw",
                                                    maxLength=40)}, ["org"]))
    def ep_delegate_vote(self, ctx, org, to=""):
        """Delegate your vote in an organisation's delegated (liquid) procedures to another member, or withdraw."""
        st = ctx.state
        org = self._org_arg(ctx, org)
        if ident(ctx.world).role(st, org, ctx.actor) is None:
            raise Forbidden("not a member")
        dele = None
        if to:
            dele = ident(ctx.world).resolve_handle(st, to)
            if dele == ctx.actor or ident(ctx.world).role(st, org, dele) is None:
                raise InvalidAction("delegate to another member")
        ctx.emit("inst.delegation", {"org": org, "member": ctx.actor, "delegate": dele}, vis=(org, ctx.actor)
                 + ((dele,) if dele else ()))
        return {"delegate": dele}

    @endpoint("decisions", S.obj({"org": S.s_str(maxLength=40), "status": S.s_enum(["open", "any"])}, ["org"]),
              read_only=True)
    def ep_decisions(self, ctx, org, status="open"):
        """Decisions of an organisation or venture (for its members, electors and partners)."""
        st = ctx.state
        o = org if org.startswith("prj_") else self._org_arg(ctx, org)
        mine = ctx.world.principals_of(ctx.actor)
        if not (o in mine or self.may_propose(st, o, ctx.actor)):
            raise Forbidden("not a member")
        out = []
        for did in st.find_ids("inst_decisions", "org", o):
            d = st.tables["inst_decisions"][did]
            if status == "open" and d["status"] != "open":
                continue
            yes, no, total = self.counts(st, dict(d, id=did))
            out.append({"decision": did, "action": d["action"], "procedure": d["proc_name"], "type":
                        d["procedure"]["type"], "status": d["status"], "by": d["by"], "closes": d["closes"],
                        "yes": yes, "no": no, "electorate": total, "result": d["result"]})
        return out[-50:]

    @endpoint("join", S.obj({"org": S.s_str(maxLength=40), "note": S.s_str(maxLength=1000)}, ["org"]))
    def ep_join(self, ctx, org, note=""):
        """Join an organisation whose charter admits members openly, or apply where admission is decided."""
        st = ctx.state
        org = self._org_arg(ctx, org)
        if not ctx.actor.startswith("p:"):
            raise Forbidden("persons join organisations")
        c = self.charter(st, org)
        adm = (c or {}).get("membership", {}).get("admission", "invite")
        if st.has("members", f"{org}|{ctx.actor}"):
            raise Conflict("already a member")
        if adm == "invite":
            raise Forbidden("membership is by invitation")
        if len(st.find_ids("members", "org", org)) >= c["membership"]["max_members"]:
            raise Conflict("the organisation is at its membership limit")
        if adm == "open":
            ident(ctx.world).add_member(ctx, org, ctx.actor, "member", "")
            return {"joined": True}
        r = self.propose(ctx, org, {"kind": "admit", "person": ctx.actor, "role": "member"}, note, applicant=True)
        return {"applied": True, **r}

    # ================================================================== endpoints: projects and ventures
    @endpoint("project_create", S.obj({
        "title": S.s_str(maxLength=120), "as_org": S.s_str("the organisation that owns the project", maxLength=40),
        "partners": S.s_arr(S.s_str(maxLength=40), "organisations or persons invited as venture partners",
                            maxItems=8),
        "lead": S.s_str("person who runs it (default: you)", maxLength=40),
        "deadline_days": S.s_int(minimum=1, maximum=3650),
        "procedure": S.s_any_obj("venture decision procedure over partners (e.g. weighted by contribution); "
                                 "without one the lead runs the project")}, ["title"]), cost_s=60)
    def ep_project_create(self, ctx, title, as_org=None, partners=None, lead=None, deadline_days=None, procedure=None):
        """Start a shared project with a pooled budget held in escrow, optionally a cooperative venture with partner
        organisations who join by consent."""
        st = ctx.state
        owner = ctx.actor
        if as_org:
            owner = self._org_arg(ctx, as_org)
            ident(ctx.world).authorize(ctx, owner, "org.admin")
        lead = ident(ctx.world).resolve_handle(st, lead) if lead else ctx.actor
        if not lead.startswith("p:"):
            raise InvalidAction("the lead is a person")
        if owner.startswith("o:") and ident(ctx.world).role(st, owner, lead) is None:
            raise InvalidAction("the lead must be a member of the owning organisation")
        proc = None
        if procedure:
            proc = self.validate_procedure(dict(procedure, electorate="partners"), "venture procedure")
            if proc["type"] == "officer" or proc["weights"] == "shares":
                raise InvalidAction("a venture decides by vote among partners (equal, contribution or tenure weights)")
        pid = st.new_id("prj")
        ps = sorted({ident(ctx.world).resolve_handle(st, p) for p in (partners or [])} - {owner})
        ctx.emit("inst.project", {"id": pid, "owner": owner, "title": title[:120], "lead": lead, "procedure": proc,
                                  "deadline": None if not deadline_days else ctx.now + deadline_days * DAY},
                 vis=tuple(sorted({owner, lead} | set(ps))))
        ctx.emit("inst.partner", {"project": pid, "partner": owner, "status": "joined"}, vis=(owner,))
        for p in ps:
            ctx.emit("inst.partner", {"project": pid, "partner": p, "status": "invited"}, vis=(owner, p))
        return {"project": pid, "invited": ps}

    @endpoint("project_join", S.obj({"project": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40),
                                     "accept": S.s_bool()}, ["project"]))
    def ep_project_join(self, ctx, project, as_org=None, accept=True):
        """Accept (or decline) an invitation to be a partner of a cooperative venture."""
        st = ctx.state
        who = ctx.actor
        if as_org:
            who = self._org_arg(ctx, as_org)
            ident(ctx.world).authorize(ctx, who, "org.admin")
        self._project(st, project)
        r = st.get("inst_partners", f"{project}|{who}")
        if r is None or r["status"] != "invited":
            raise NotFound("no invitation to this project")
        ctx.emit("inst.partner", {"project": project, "partner": who, "status": "joined" if accept else "declined"},
                 vis=self._pvis(st, project) + (who,))
        return {"status": "joined" if accept else "declined"}

    @endpoint("project_fund", S.obj({"project": S.s_str(maxLength=40), "account": S.s_str(maxLength=40),
                                     "amount": S.s_int("cents", minimum=1, maximum=MAX_AMOUNT)},
                                    ["project", "account", "amount"]), cost_s=30)
    def ep_project_fund(self, ctx, project, account, amount):
        """Contribute money from an account you control to a project's pool (recorded as your contribution)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        self._project(st, project)
        owner = bank.account_owner(st, account)
        r = st.get("inst_partners", f"{project}|{owner}")
        member_of_owner = ident(ctx.world).role(st, st.tables["inst_projects"][project]["owner"], owner) is not None
        if not (r and r["status"] == "joined") and not member_of_owner:
            raise Forbidden("only partners and members of the owning organisation fund a project")
        ident(ctx.world).authorize(ctx, owner, "bank.spend", amount=amount, scope=account)
        self.fund(ctx, project, owner, account, amount)
        return {"pool": st.tables["inst_projects"][project]["pool"]}

    @endpoint("task_add", S.obj({"project": S.s_str(maxLength=40), "title": S.s_str(maxLength=120),
                                 "skills": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0,
                                                                                       "maximum": 1},
                                            "description": "skill -> required level (0..1)"},
                                 "effort_hours": S.s_int(minimum=1, maximum=5000),
                                 "depends_on": S.s_arr(S.s_str(maxLength=40), maxItems=20)},
                                ["project", "title", "effort_hours"]), cost_s=30)
    def ep_task_add(self, ctx, project, title, effort_hours, skills=None, depends_on=None):
        """Add a task to a project you lead (or hold `inst.lead` for): required skills, effort and the tasks it
        depends on (which must already exist, so dependencies cannot form a cycle)."""
        st = ctx.state
        prj = self._project(st, project)
        if not self.is_lead(ctx, prj, project):
            raise Forbidden("only the project's lead (or a delegate holding inst.lead) adds tasks")
        deps = sorted(set(depends_on or []))
        for d in deps:
            t = st.get("inst_tasks", d)
            if t is None or t["project"] != project:
                raise InvalidAction(f"dependency {d} is not a task of this project")
        from .labor import SKILLS
        sk = {}
        for k in sorted(skills or {}):
            if k not in SKILLS:
                raise InvalidAction(f"unknown skill {k!r} (skills: {', '.join(SKILLS)})")
            sk[k] = round(float(skills[k]), 4)
        tid = st.new_id("task")
        ctx.emit("inst.task", {"id": tid, "project": project, "title": title[:120], "skills": sk,
                               "effort_hours": int(effort_hours), "depends_on": deps}, vis=self._pvis(st, project))
        return {"task": tid}

    @endpoint("allocate", S.obj({"project": S.s_str(maxLength=40), "task": S.s_str(maxLength=40),
                                 "amount": S.s_int("cents", minimum=1, maximum=MAX_AMOUNT)},
                                ["project", "task", "amount"]))
    def ep_allocate(self, ctx, project, task, amount):
        """Reserve part of a lead-run project's pool as the reward for a task (paid when the task is done).
        Ventures with a procedure allocate by decision (inst.propose {kind: allocate})."""
        st = ctx.state
        prj = self._project(st, project)
        if prj["procedure"]:
            raise Forbidden("this venture allocates by decision of its partners: inst.propose")
        if not self.is_lead(ctx, prj, project):
            raise Forbidden("only the project's lead allocates its pool")
        self.allocate(ctx, project, task, amount)
        return {"allocated": amount}

    @endpoint("task_assign", S.obj({"task": S.s_str(maxLength=40), "assignee": S.s_str(maxLength=40)},
                                   ["task", "assignee"]))
    def ep_task_assign(self, ctx, task, assignee):
        """Assign a task to a member of the owning organisation or a partner (a lead or `inst.lead` delegate)."""
        st = ctx.state
        t = st.get("inst_tasks", task)
        if t is None:
            raise NotFound("no task")
        prj = self._project(st, t["project"])
        if not self.is_lead(ctx, prj, t["project"]):
            raise Forbidden("only the project's lead (or a delegate holding inst.lead) assigns tasks")
        who = ident(ctx.world).resolve_handle(st, assignee)
        ok = who == prj["lead"] or (prj["owner"].startswith("o:") and ident(ctx.world).role(st, prj["owner"], who))
        for r in st.find("inst_partners", "project", t["project"]):
            if r["status"] == "joined" and (r["partner"] == who or (r["partner"].startswith("o:") and
                                                                    ident(ctx.world).role(st, r["partner"], who))):
                ok = True
        if not ok:
            raise Forbidden("assign to members of the owning organisation or of a partner")
        if t["status"] in ("done", "cancelled"):
            raise Conflict("task finished")
        ctx.emit("inst.task_assigned", {"id": task, "assignee": who}, vis=self._pvis(st, t["project"]) + (who,))
        hook = ctx.world.hooks.get("on_task_assigned")
        if hook:
            hook(ctx, task, who)
        return {"ok": True}

    @endpoint("task_submit", S.obj({"task": S.s_str(maxLength=40), "note": S.s_str(maxLength=2000)}, ["task"]))
    def ep_task_submit(self, ctx, task, note=""):
        """The assignee reports a task done; the lead reviews it (inst.task_review)."""
        st = ctx.state
        t = st.get("inst_tasks", task)
        if t is None or t["assignee"] != ctx.actor:
            raise NotFound("no task assigned to you")
        if t["status"] != "assigned":
            raise Conflict("task not in progress")
        if not self.deps_done(st, task):
            raise Conflict("the tasks this one depends on are not done")
        ctx.emit("inst.task_status", {"id": task, "status": "submitted", "quality": None, "note": note[:2000]},
                 vis=self._pvis(st, t["project"]))
        return {"ok": True}

    @endpoint("task_review", S.obj({"task": S.s_str(maxLength=40), "accept": S.s_bool()}, ["task", "accept"]))
    def ep_task_review(self, ctx, task, accept):
        """Accept a submitted task (pays its reward) or send it back."""
        st = ctx.state
        t = st.get("inst_tasks", task)
        if t is None or t["status"] != "submitted":
            raise NotFound("no submitted task")
        prj = self._project(st, t["project"])
        if not self.is_lead(ctx, prj, t["project"]) or ctx.actor == t["assignee"]:
            raise Forbidden("the lead reviews (not the assignee)")
        if not accept:
            ctx.emit("inst.task_status", {"id": task, "status": "assigned", "quality": None},
                     vis=self._pvis(st, t["project"]))
            return {"status": "returned"}
        return self.complete_task(ctx, task, None, ctx.actor)

    @endpoint("project", S.obj({"project": S.s_str(maxLength=40)}, ["project"]), read_only=True)
    def ep_project(self, ctx, project):
        """A project's partners, pool, tasks and their state (for its partners, lead, assignees and members)."""
        st = ctx.state
        prj = st.get("inst_projects", project)
        if prj is None:
            raise NotFound("no project")
        mine = ctx.world.principals_of(ctx.actor)
        if not (mine & set(self._pvis(st, project)) or (prj["owner"].startswith("o:") and
                                                       ident(ctx.world).role(st, prj["owner"], ctx.actor))):
            raise Forbidden("not involved in this project")
        tasks = [{"task": tid, **{k: t[k] for k in ("title", "skills", "effort_hours", "depends_on", "assignee",
                                                    "reward", "status", "progress", "quality")}}
                 for tid in st.find_ids("inst_tasks", "project", project) for t in [st.tables["inst_tasks"][tid]]]
        return {"project": project, **{k: prj[k] for k in ("owner", "title", "lead", "pool", "allocated", "status",
                                                           "deadline", "procedure")},
                "partners": [{"partner": r["partner"], "status": r["status"]}
                             for r in st.find("inst_partners", "project", project)],
                "contributions": self.contributions(st, project), "tasks": tasks}

    def contributions(self, st, pid) -> dict:
        out = {}
        for r in st.find("inst_contributions", "project", pid):
            out[r["contributor"]] = out.get(r["contributor"], 0) + r["cents"]
        return out

    @endpoint("project_close", S.obj({"project": S.s_str(maxLength=40)}, ["project"]))
    def ep_project_close(self, ctx, project):
        """Close a lead-run project (its lead); ventures with a procedure close by decision."""
        st = ctx.state
        prj = self._project(st, project)
        if prj["procedure"]:
            raise Forbidden("this venture closes by decision of its partners: inst.propose")
        if not self.is_lead(ctx, prj, project):
            raise Forbidden("only the project's lead closes it")
        return self.close_project(ctx, project)

    # ================================================================== process
    @process("tick")
    def pr_tick(self, ctx):
        for did in ctx.state.find_ids("inst_decisions", "status", "open"):
            if ctx.state.tables["inst_decisions"][did]["closes"] <= ctx.now:
                err = contain(ctx, self.tally, ctx, did, True)
                if err is not None:
                    ctx.emit("inst.tick_failed", {"id": did, "error": f"{type(err).__name__}: {err}"[:200]},
                             vis="internal")

    # ================================================================== appliers
    @applier("inst.charter")
    def ap_charter(self, ev):
        st = self.world.state
        p = ev.payload
        old = st.tables["inst_charters"].get(p["org"])
        st.put("inst_charters", p["org"], {"charter": p["charter"], "version": (old["version"] + 1) if old else 1,
                                      "t": ev.t, "by": p["by"]})

    @applier("inst.proposed")
    def ap_proposed(self, ev):
        p = ev.payload
        self.world.state.insert("inst_decisions", p["id"], {
            "org": p["org"], "action": p["action"], "procedure": p["procedure"], "proc_name": p["proc_name"],
            "electorate": p["electorate"], "by": p["by"], "reason": p["reason"], "closes": p["closes"],
            "status": "open", "t": ev.t, "decided_t": None, "result": None})

    @applier("inst.ballot")
    def ap_ballot(self, ev):
        p = ev.payload
        self.world.state.put("inst_ballots", p["id"], {"decision": p["decision"], "voter": p["voter"],
                                                       "choice": p["choice"], "t": ev.t})

    @applier("inst.decided")
    def ap_decided(self, ev):
        self.world.state.patch("inst_decisions", ev.payload["id"], status=ev.payload["status"], decided_t=ev.t,
                               result=ev.payload["result"])

    @applier("inst.delegation")
    def ap_delegation(self, ev):
        p = ev.payload
        k = f"{p['org']}|{p['member']}"
        if p["delegate"] is None:
            self.world.state.delete("inst_delegations", k)
        else:
            self.world.state.put("inst_delegations", k, {"org": p["org"], "member": p["member"],
                                                         "delegate": p["delegate"], "t": ev.t})

    @applier("inst.office")
    def ap_office(self, ev):
        p = ev.payload
        self.world.state.put("inst_offices", f"{p['org']}|{p['title']}", {
            "org": p["org"], "title": p["title"], "holder": p["holder"], "budget_cents_month": p["budget_cents_month"],
            "tools": p["tools"], "unit": p.get("unit"), "t": ev.t})

    @applier("inst.office_removed")
    def ap_office_removed(self, ev):
        self.world.state.delete("inst_offices", f"{ev.payload['org']}|{ev.payload['title']}")

    @applier("inst.office_spend")
    def ap_office_spend(self, ev):
        st = self.world.state
        r = st.tables["inst_office_spend"].get(ev.payload["id"]) or {"used": 0}
        st.put("inst_office_spend", ev.payload["id"], {"used": r["used"] + ev.payload["amount"]})

    @applier("inst.unit")
    def ap_unit(self, ev):
        p = ev.payload
        self.world.state.insert("inst_units", p["id"], {"org": p["org"], "name": p["name"], "head": p["head"],
                                                  "parent": p["parent"], "status": "active", "t": ev.t})

    @applier("inst.unit_member")
    def ap_unit_member(self, ev):
        p = ev.payload
        k = f"{p['unit']}|{p['person']}"
        if p["remove"]:
            self.world.state.delete("inst_unit_members", k)
            u = self.world.state.tables["inst_units"][p["unit"]]
            if u["head"] == p["person"]:
                self.world.state.patch("inst_units", p["unit"], head=None)
        else:
            self.world.state.put("inst_unit_members", k, {"unit": p["unit"], "person": p["person"], "t": ev.t})

    @applier("inst.unit_status")
    def ap_unit_status(self, ev):
        self.world.state.patch("inst_units", ev.payload["id"], status=ev.payload["status"])

    @applier("inst.project")
    def ap_project(self, ev):
        p = ev.payload
        self.world.state.insert("inst_projects", p["id"], {"owner": p["owner"], "title": p["title"], "lead": p["lead"],
                                                     "procedure": p["procedure"], "deadline": p["deadline"],
                                                     "pool": 0, "allocated": 0, "status": "open", "t": ev.t})

    @applier("inst.project_owner")
    def ap_project_owner(self, ev):
        self.world.state.patch("inst_projects", ev.payload["id"], owner=ev.payload["owner"])

    @applier("inst.partner")
    def ap_partner(self, ev):
        p = ev.payload
        self.world.state.put("inst_partners", f"{p['project']}|{p['partner']}", {
            "project": p["project"], "partner": p["partner"], "status": p["status"], "t": ev.t})

    @applier("inst.contribution")
    def ap_contribution(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("inst_contributions", p["id"], {"project": p["project"], "org": p["org"], "contributor": p["contributor"],
                                             "cents": p["cents"], "kind": p["kind"], "account": p["account"],
                                             "t": ev.t})
        if p["kind"] == "money" and p["project"]:
            prj = st.tables["inst_projects"][p["project"]]
            st.patch("inst_projects", p["project"], pool=prj["pool"] + p["cents"])
        if p["kind"] == "labor":
            # profit sharing by labour (4.0 gov distribution basis 'labor') reads these hours-valued weights
            k = f"labor_hours:{p['org']}"
            cur = dict(st.meta.get(k) or {})
            cur[p["contributor"]] = cur.get(p["contributor"], 0) + p["cents"]
            st.meta_set(k, cur)

    @applier("inst.task")
    def ap_task(self, ev):
        p = ev.payload
        self.world.state.insert("inst_tasks", p["id"], {
            "project": p["project"], "title": p["title"], "skills": p["skills"], "effort_hours": p["effort_hours"],
            "depends_on": p["depends_on"], "assignee": None, "reward": 0, "status": "open", "progress": 0.0,
            "quality": None, "t": ev.t, "done_t": None, "late": False})

    @applier("inst.task_assigned")
    def ap_task_assigned(self, ev):
        self.world.state.patch("inst_tasks", ev.payload["id"], assignee=ev.payload["assignee"], status="assigned")

    @applier("inst.task_progress")
    def ap_task_progress(self, ev):
        self.world.state.patch("inst_tasks", ev.payload["id"], progress=ev.payload["progress"])

    @applier("inst.task_unassigned")
    def ap_task_unassigned(self, ev):
        self.world.state.patch("inst_tasks", ev.payload["id"], assignee=None, status="open")

    @applier("inst.allocated")
    def ap_allocated(self, ev):
        st = self.world.state
        p = ev.payload
        t = st.tables["inst_tasks"][p["task"]]
        prj = st.tables["inst_projects"][p["project"]]
        st.patch("inst_tasks", p["task"], reward=t["reward"] + p["amount"])
        st.patch("inst_projects", p["project"], allocated=prj["allocated"] + p["amount"])

    @applier("inst.task_status")
    def ap_task_status(self, ev):
        st = self.world.state
        p = ev.payload
        t = st.tables["inst_tasks"][p["id"]]
        prj = st.tables["inst_projects"][t["project"]]
        ch = {"status": p["status"]}
        if p["status"] == "done":
            ch.update(quality=p["quality"], done_t=ev.t, late=p.get("late", False), progress=1.0)
            st.patch("inst_projects", t["project"], pool=prj["pool"] - p.get("paid", 0),
                     allocated=prj["allocated"] - t["reward"])
        elif p["status"] == "cancelled":
            st.patch("inst_projects", t["project"], allocated=prj["allocated"] - t["reward"])
            ch["reward"] = 0
        st.patch("inst_tasks", p["id"], **ch)

    @applier("inst.project_status")
    def ap_project_status(self, ev):
        p = ev.payload
        prj = self.world.state.tables["inst_projects"][p["id"]]
        self.world.state.patch("inst_projects", p["id"], status=p["status"], pool=prj["pool"] - p["refunded"],
                               closed_t=ev.t)

    @applier("inst.merge_offer")
    def ap_merge_offer(self, ev):
        p = ev.payload
        self.world.state.put("inst_merge_offers", p["org"], {"into": p["into"], "decision": p["decision"], "t": ev.t})

    @applier("inst.merged")
    def ap_merged(self, ev):
        self.world.state.delete("inst_merge_offers", ev.payload["org"])

    @applier("inst.spun_off")
    def ap_spun_off(self, ev):
        return None

    @applier("inst.dissolved")
    def ap_dissolved(self, ev):
        return None

    @applier("inst.tick_failed")
    def ap_tick_failed(self, ev):
        return None

    # ================================================================== invariants and observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        held = sum(p["pool"] for p in st.tables["inst_projects"].values() if p["status"] == "open")
        closed_left = sum(p["pool"] for p in st.tables["inst_projects"].values() if p["status"] != "open")
        bal = st.tables["accounts"][ESCROW]["balance"] if st.has("accounts", ESCROW) else 0
        if held != bal:
            errs.append(f"institution escrow {bal} != open project pools {held}")
        if closed_left:
            errs.append(f"closed projects still hold {closed_left} cents")
        for pid, p in st.tables["inst_projects"].items():
            if p["allocated"] < 0 or p["allocated"] > p["pool"]:
                errs.append(f"{pid}: allocated {p['allocated']} outside [0, pool {p['pool']}]")
        for did, d in st.tables["inst_decisions"].items():
            if d["status"] == "executed" and d["action"]["kind"] == "tool" and d["procedure"]["type"] != "officer":
                yes = sum(w for p, w in d["electorate"]
                          if (st.get("inst_ballots", f"{did}|{p}") or {}).get("choice") == "yes")
                if yes <= 0 and d["procedure"]["type"] not in ("delegated", "consensus"):
                    errs.append(f"{did} executed without a single approving vote")
        return errs

    def observe(self, world, actor):
        st = world.state
        if not actor.startswith("p:"):
            return None
        pending = []
        for did in st.find_ids("inst_decisions", "status", "open"):
            d = st.tables["inst_decisions"][did]
            if any(p == actor for p, _ in d["electorate"]) and not st.has("inst_ballots", f"{did}|{actor}"):
                pending.append(did)
        tasks = [tid for tid in st.find_ids("inst_tasks", "assignee", actor)
                 if st.tables["inst_tasks"][tid]["status"] == "assigned"]
        if not pending and not tasks:
            return None
        return {"decisions_to_vote": pending[-20:], "tasks_assigned": tasks[-20:]}


def inst(world) -> Institutions:
    return world.modules["inst"]  # type: ignore[return-value]
