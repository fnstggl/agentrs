"""Institutions (module `gov`, host `gov`): composable governance for organisations.

An organisation may adopt a **constitution**: a list of rules, each matching a class of actions taken in the
organisation's name and stating what is required before the action takes effect:

    {"match": {"tool": "bank.transfer", "min_amount": 100000}, "require": {"type": "vote", "electorate": "members",
     "threshold": 0.5, "quorum": 0.5, "window_s": 172800}}
    {"match": {"tool": "jobs.offer"}, "require": {"type": "approval", "by": "owner"}}
    {"match": {"tool": "*"}, "require": {"type": "delegate", "to": ["p:alice"], "budget_cents_month": 500000}}
    {"match": {"tool": "gov.amend"}, "require": {"type": "vote", "electorate": "shareholders", "threshold": 0.66}}

Enforcement is part of authorisation: a governed action taken for the organisation by anyone (owners included)
is refused unless it runs under an approved proposal (`gov.propose` -> votes -> automatic execution) or within a
delegated budget. Decision costs therefore emerge from the rules themselves: a vote takes as long as it takes the
electorate (people with limited attention, or agents) to respond; a single approver is a bottleneck; delegation is
fast but spends from a capped budget. The constitution can only be changed by `gov.amend`, which is itself
governed by the constitution.

**Profit sharing** (`distribution` in the constitution): on a monthly schedule, a share of the month's positive
operating result is paid out by equity (cap table), equally among members, or by members' recorded labour hours.
Simulated members' effort responds to how much of the marginal output they keep (see `effort_share` in
`crucible.actors.producers`), so free-riding in large equal-share cooperatives is a measurable consequence, not a
parameter.

Crucible 5.0 write-through (workstream F, SP-2.3 migration; `primitives.OWNERS["institution"]` is the v5 charter
table `inst.inst_charters`): in a world that also has the v5 institutions module `inst`, this module keeps no
constitution, proposal, vote or delegated-spend record of its own. `gov.adopt` / `gov.amend` adopt or propose an
`inst` charter translated from the constitution (`commerce.institutions.charter_from_constitution`), `gov.propose`
/ `gov.vote` are `inst` decisions and ballots, `gov.constitution` shows the charter in the 4.0 form, enforcement and
profit sharing are `inst`'s, and `effort` reads the charter. Worlds without `inst` (every 4.0 family) are unchanged.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

DAY = 86400
MONTH = 30 * DAY
RULE = {"type": "object", "additionalProperties": True}


def _matches(rule_match: dict, tool: str, args: dict) -> bool:
    t = rule_match.get("tool", "*")
    if t != "*" and not (t == tool or (t.endswith(".*") and tool.startswith(t[:-1]))):
        return False
    amt = max([v for k, v in args.items() if k in ("amount", "amount_cents", "salary_cents_month", "price_cents",
                                                     "reward_cents") and isinstance(v, int)] or [0])
    if "min_amount" in rule_match and amt < int(rule_match["min_amount"]):
        return False
    return True


@register_module
class Governance(Module):
    name = "gov"
    depends = ("identity", "bank")

    def _inst(self):
        """The v5 institutions module when installed: this module then writes through it (module docstring)."""
        return self.world.modules.get("inst")

    def install(self, world):
        st = world.state
        st.declare_table("constitutions")
        st.declare_table("proposals", index=("org", "status"))
        st.declare_table("votes", index=("proposal",))
        st.declare_table("gov_spend")
        world.hooks["gov"] = self
        world.hooks["gov_check"] = self.check

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 3600, "gov.tick", {}, prio=59, period=3600, key="gov.tick")
        ctx.schedule(ctx.now + MONTH, "gov.distribute", {}, prio=81, period=MONTH, key="gov.distribute")

    # ------------------------------------------------------------------ the authorisation hook
    def rule_for(self, st, org, tool, args) -> dict | None:
        c = st.get("constitutions", org)
        if c is None:
            return None
        for r in c["rules"]:
            if _matches(r.get("match", {}), tool, args):
                return r
        return None

    def check(self, ctx: Ctx, org: str, tool: str, args: dict) -> None:
        """Called before a tool runs on behalf of an organisation. Raises Forbidden if the constitution requires a
        decision that has not been made."""
        st = ctx.state
        if getattr(ctx, "gov_approved", None) == org or self._inst() is not None:
            return                       # with inst installed its charters govern (chained on gov_check)
        r = self.rule_for(st, org, tool, args)
        if r is None:
            return
        req = r["require"]
        if req["type"] == "delegate":
            budget = int(req.get("budget_cents_month", 0))
            amt = max([v for k, v in args.items() if k in ("amount", "amount_cents", "salary_cents_month") and
                       isinstance(v, int)] or [0])
            who = ctx.origin
            if who not in req.get("to", []):
                raise Forbidden(f"{tool} for {org} is delegated to {req.get('to')}; propose it with gov.propose")
            k = f"{org}|{who}|{ctx.now // MONTH}"
            used = (st.get("gov_spend", k) or {}).get("used", 0)
            if used + amt > budget:
                raise Forbidden(f"delegated budget exhausted ({used}/{budget}); propose it with gov.propose")
            ctx.emit("gov.spend", {"id": k, "amount": int(amt)}, vis=(org,))
            return
        raise Forbidden(f"{org}'s constitution requires {req['type']} ({req.get('electorate') or req.get('by')}) "
                        f"for {tool}; use gov.propose")

    # ------------------------------------------------------------------ constitutions
    def _validate_rules(self, rules: list) -> list:
        out = []
        for r in rules:
            req = r.get("require") or {}
            if req.get("type") not in ("vote", "approval", "delegate"):
                raise InvalidAction("require.type must be vote, approval or delegate")
            if req["type"] == "vote" and req.get("electorate", "members") not in ("members", "shareholders", "admins"):
                raise InvalidAction("electorate must be members, shareholders or admins")
            out.append({"match": dict(r.get("match") or {"tool": "*"}),
                        "require": {k: req[k] for k in req if k in ("type", "electorate", "threshold", "quorum",
                                                                     "window_s", "by", "to", "budget_cents_month")}})
        return out

    @endpoint("constitution", S.obj({"org": S.s_str(maxLength=40)}, ["org"]), read_only=True)
    def ep_constitution(self, ctx, org):
        """An organisation's constitution (public)."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        if self._inst() is not None:
            from ..commerce.institutions import constitution_view
            return constitution_view(self._inst().charter(ctx.state, org))
        return dict(ctx.state.get("constitutions", org) or {"rules": [], "distribution": None})

    @endpoint("adopt", S.obj({"org": S.s_str(maxLength=40), "rules": S.s_arr(RULE, maxItems=20),
                              "distribution": S.s_any_obj("{basis: equity|equal|labor, share: 0..1}")},
                             ["org", "rules"]), cost_s=120)
    def ep_adopt(self, ctx, org, rules, distribution=None):
        """Adopt a first constitution (only while the organisation has none; afterwards use gov.amend)."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if self._inst() is not None:
            from ..commerce.institutions import charter_from_constitution
            if st.has("inst_charters", org):
                raise Conflict("already constituted; amend it with gov.amend")
            self._inst().adopt_charter(ctx, org, charter_from_constitution(self._validate_rules(rules),
                                                                            self._dist(distribution)))
            return {"ok": True}
        if st.has("constitutions", org):
            raise Conflict("already constituted; amend it with gov.amend")
        ctx.emit("gov.constitution", {"org": org, "rules": self._validate_rules(rules),
                                      "distribution": self._dist(distribution)}, vis="public")
        return {"ok": True}

    def _dist(self, d):
        if not d:
            return None
        if d.get("basis") not in ("equity", "equal", "labor"):
            raise InvalidAction("distribution basis must be equity, equal or labor")
        return {"basis": d["basis"], "share": float(min(1.0, max(0.0, d.get("share", 0.5))))}

    @endpoint("amend", S.obj({"org": S.s_str(maxLength=40), "rules": S.s_arr(RULE, maxItems=20),
                              "distribution": S.s_any_obj()}, ["org", "rules"]), cost_s=120)
    def ep_amend(self, ctx, org, rules, distribution=None):
        """Replace the constitution. Governed like any other action: usually needs an approved proposal."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if self._inst() is not None:
            from ..commerce.institutions import charter_from_constitution
            ch = charter_from_constitution(self._validate_rules(rules), self._dist(distribution))
            if not st.has("inst_charters", org):
                self._inst().adopt_charter(ctx, org, ch)
                return {"ok": True}
            return {"ok": True, **self._inst().propose(ctx, org, {"kind": "amend", "charter": ch}, "gov.amend")}
        self.check(ctx, org, "gov.amend", {})
        ctx.emit("gov.constitution", {"org": org, "rules": self._validate_rules(rules),
                                      "distribution": self._dist(distribution)}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ proposals & votes
    def electorate(self, st, org, req) -> dict[str, float]:
        kind = req.get("electorate", "members")
        if req["type"] == "approval":
            by = req.get("by", "owner")
            return {m["person"]: 1.0 for m in st.find("members", "org", org) if m["role"] == by}
        if kind == "shareholders":
            w = {}
            for s in st.find("shares", "org", org):
                w[s["holder"]] = w.get(s["holder"], 0.0) + float(s["shares"])
            return w
        roles = ("owner", "admin") if kind == "admins" else ("owner", "admin", "member")
        return {m["person"]: 1.0 for m in st.find("members", "org", org) if m["role"] in roles}

    @endpoint("propose", S.obj({"org": S.s_str(maxLength=40), "tool": S.s_str(maxLength=80), "args": S.s_any_obj(),
                                "reason": S.s_str(maxLength=2000)}, ["org", "tool"]), cost_s=60)
    def ep_propose(self, ctx, org, tool, args=None, reason=""):
        """Propose an action for the organisation; it executes automatically (as the organisation) once the
        constitution's requirement is met, or lapses when the voting window closes."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        if self._inst() is not None:
            r = self._inst().propose(ctx, org, {"kind": "tool", "tool": tool, "args": args or {}}, reason)
            d = st.get("inst_decisions", r.get("decision")) if r.get("decision") else None
            return {"proposal": r.get("decision"), "closes": d["closes"] if d else ctx.now, "status": r.get("status")}
        if ident(ctx.world).role(st, org, ctx.actor) is None and ctx.actor not in self.electorate(
                st, org, {"type": "vote", "electorate": "shareholders"}):
            raise Forbidden("only members or shareholders can propose")
        args = args or {}
        r = self.rule_for(st, org, tool, args) or {"require": {"type": "approval", "by": "owner"}}
        req = r["require"]
        pid = st.new_id("prop")
        ctx.emit("gov.proposal", {"id": pid, "org": org, "tool": tool, "args": args, "reason": reason[:2000],
                                  "by": ctx.actor, "require": req,
                                  "closes": ctx.now + int(req.get("window_s", 2 * DAY))},
                 vis=tuple(sorted(set(self.electorate(st, org, req)) | {ctx.actor, org})))
        hook = ctx.world.hooks.get("on_gov_proposal")
        if hook:
            for voter in sorted(self.electorate(st, org, req)):
                hook(ctx, pid, voter)
        return {"proposal": pid, "closes": ctx.now + int(req.get("window_s", 2 * DAY))}

    @endpoint("vote", S.obj({"proposal": S.s_str(maxLength=40), "approve": S.s_bool()}, ["proposal", "approve"]))
    def ep_vote(self, ctx, proposal, approve):
        """Vote on a proposal (if you are in its electorate). Decisions execute as soon as they are reached."""
        st = ctx.state
        if self._inst() is not None:
            return self._inst().vote(ctx, proposal, ctx.actor, "yes" if approve else "no")
        p = st.get("proposals", proposal)
        if p is None or p["status"] != "open":
            raise NotFound("no open proposal")
        el = self.electorate(st, p["org"], p["require"])
        if ctx.actor not in el:
            raise Forbidden("not in the electorate")
        ctx.emit("gov.vote", {"id": f"{proposal}|{ctx.actor}", "proposal": proposal, "voter": ctx.actor,
                              "approve": bool(approve), "weight": el[ctx.actor]}, vis=tuple(sorted(set(el) | {p["org"]})))
        return self.tally(ctx, proposal, final=False)

    def tally(self, ctx, pid, final: bool) -> dict:
        st = ctx.state
        p = st.tables["proposals"][pid]
        req = p["require"]
        el = self.electorate(st, p["org"], req)
        total = sum(el.values()) or 1.0
        yes = sum(v["weight"] for v in st.find("votes", "proposal", pid) if v["approve"])
        no = sum(v["weight"] for v in st.find("votes", "proposal", pid) if not v["approve"])
        thr = float(req.get("threshold", 0.5 if req["type"] == "vote" else 1.0))
        quorum = float(req.get("quorum", 0.0))
        decided = None
        if req["type"] == "approval":
            decided = True if yes > 0 else (False if no > 0 else None)
        else:
            if yes / total > thr and (yes + no) / total >= quorum:
                decided = True
            elif no / total >= 1 - thr:
                decided = False
        if decided is None and final:
            decided = False
        if decided is True:
            self.execute(ctx, pid)
        elif decided is False:
            ctx.emit("gov.decided", {"id": pid, "status": "rejected", "result": None}, vis=(p["org"], p["by"]))
        return {"yes": yes, "no": no, "electorate": total, "status": st.tables["proposals"][pid]["status"]}

    def execute(self, ctx, pid):
        st = ctx.state
        p = st.tables["proposals"][pid]
        org = p["org"]

        def run(sub):
            sub.gov_approved = org
            return sub.call(p["tool"], p["args"])
        try:
            res = ctx.run_as(org, run)
            ctx.emit("gov.decided", {"id": pid, "status": "executed", "result": res}, vis=(org, p["by"]))
        except CrucibleError as e:
            ctx.emit("gov.decided", {"id": pid, "status": "failed", "result": e.as_dict()}, vis=(org, p["by"]))

    @endpoint("proposals", S.obj({"org": S.s_str(maxLength=40), "status": S.s_enum(["open", "any"])}, ["org"]),
              read_only=True)
    def ep_proposals(self, ctx, org, status="open"):
        """Proposals of an organisation (members and shareholders)."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        if self._inst() is not None:
            return [{"proposal": k, "tool": d["action"].get("tool"), "args": d["action"].get("args"),
                     "reason": d["reason"], "by": d["by"], "status": d["status"], "closes": d["closes"], "t": d["t"],
                     "decided_t": d["decided_t"]}
                    for k in st.find_ids("inst_decisions", "org", org) for d in [st.tables["inst_decisions"][k]]
                    if d["action"].get("kind") == "tool" and (status == "any" or d["status"] == "open")]
        return [{"proposal": k, **{x: p[x] for x in ("tool", "args", "reason", "by", "status", "closes", "t",
                                                      "decided_t")}}
                for k in st.find_ids("proposals", "org", org) for p in [st.tables["proposals"][k]]
                if status == "any" or p["status"] == "open"]

    @process("tick")
    def pr_tick(self, ctx):
        for pid in ctx.state.find_ids("proposals", "status", "open"):
            if ctx.state.tables["proposals"][pid]["closes"] <= ctx.now:
                self.tally(ctx, pid, final=True)

    # ------------------------------------------------------------------ profit sharing
    @process("distribute")
    def pr_distribute(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        ops = ctx.world.modules.get("ops")
        for org, c in st.scan("constitutions"):
            d = c.get("distribution")
            if not d or ops is None:
                continue
            profit = ops.window(st, org, 30, ctx.now // DAY + 1)["operating"]
            acct = bank.primary_account(st, org)
            if profit <= 0 or acct is None:
                continue
            pool = min(int(profit * d["share"]), max(0, bank.balance(st, acct)))
            if d["basis"] == "equity":
                w = {}
                for s in st.find("shares", "org", org):
                    w[s["holder"]] = w.get(s["holder"], 0) + s["shares"]
            elif d["basis"] == "equal":
                w = {m["person"]: 1 for m in st.find("members", "org", org)}
            else:
                w = dict((st.meta.get(f"labor_hours:{org}") or {}))
            tot = sum(w.values())
            paid = 0
            for who in sorted(w):
                amt = int(pool * w[who] / tot) if tot else 0
                dst = bank.primary_account(st, who)
                if amt > 0 and dst and dst != acct:
                    bank.move(ctx, acct, dst, amt, "distribution", f"profit share of {org}")
                    paid += amt
            ctx.emit("gov.distributed", {"org": org, "pool": pool, "paid": paid, "basis": d["basis"],
                                         "recipients": len(w)}, vis=(org,))

    def effort(self, st, org: str, n_members: int) -> float:
        """Effort of an organisation's working members relative to salaried staff under ordinary supervision (=1).
        Each member chooses effort e to maximise s*Y*e - c*e^2/2, where s is the share of the marginal output it
        keeps under the organisation's profit-sharing rule; the optimum is linear in s (here e = 0.55 + 2.5 s,
        clipped to [0.4, 1.15]). Equal sharing among many members dilutes s (free-riding); paying by recorded labour
        hours restores it."""
        if self._inst() is not None:
            return self._inst().effort(st, org, n_members)
        c = st.get("constitutions", org)
        d = (c or {}).get("distribution")
        if not d or d["basis"] == "equity":
            return 1.0
        s = d["share"] / max(1, n_members)
        if d["basis"] == "labor":
            s += 0.25
        return float(min(1.15, max(0.4, 0.55 + 2.5 * s)))

    # ------------------------------------------------------------------ appliers
    @applier("gov.constitution")
    def ap_constitution(self, ev):
        p = ev.payload
        self.world.state.put("constitutions", p["org"], {"rules": p["rules"], "distribution": p["distribution"],
                                                         "t": ev.t})

    @applier("gov.proposal")
    def ap_proposal(self, ev):
        p = dict(ev.payload)
        p.update(status="open", t=ev.t, decided_t=None)
        self.world.state.insert("proposals", p.pop("id"), p)

    @applier("gov.vote")
    def ap_vote(self, ev):
        p = ev.payload
        self.world.state.put("votes", p["id"], {"proposal": p["proposal"], "voter": p["voter"], "approve": p["approve"],
                                                "weight": p["weight"], "t": ev.t})

    @applier("gov.decided")
    def ap_decided(self, ev):
        self.world.state.patch("proposals", ev.payload["id"], status=ev.payload["status"], decided_t=ev.t)

    @applier("gov.spend")
    def ap_spend(self, ev):
        st = self.world.state
        r = st.tables["gov_spend"].get(ev.payload["id"]) or {"used": 0}
        st.put("gov_spend", ev.payload["id"], {"used": r["used"] + ev.payload["amount"]})

    @applier("gov.distributed")
    def ap_distributed(self, ev):
        return None
