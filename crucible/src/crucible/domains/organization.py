"""Tier 2 domain 9.12 - the organizational ecosystem (kernel module `orgeco`, host `orgs`; MASTER_SPEC Part 9.12).

Example objective: coordinate independent organizations toward a shared measurable objective.

World physics (module config generated per instance by `build`; nothing hidden is returned by any tool):

* **A shared resource.** Independent service providers deliver their services over one shared backbone with a
  finite capacity (requests per hour). Every hour the backbone carries what the providers' customers ask for; when
  the load exceeds the capacity every provider's requests fail in proportion (congestion is shared: one provider's
  growth degrades everyone's service). A provider whose load is above its quota (when the council has set quotas)
  loses its excess first. Capacity grows only through expansion projects: funded, then built over days.
* **Organizations with differing goals** (identity organisations run by autonomous managers): `profit` (high price,
  steady marketing, pays only for what pays it back), `growth` (low price, heavy marketing - the commons' main
  load), `mission` (moderate, funds the shared outcome up to a pledge budget), `rival` (undercuts the cheapest
  provider, opposes the convener). Each prices and markets daily from its own books and the public telemetry.
* **The population** (`crucible.population`, D-15) are the customers. In an `online` session an individual without
  a provider picks one it knows by price and its own experienced reliability (Beta-mean trust, multinomial logit)
  and pays its monthly subscription from its own account; a subscriber makes requests and experiences the share
  that succeed (the backbone's physics for its provider); trust falls with failures and a disappointed subscriber
  leaves for another provider or none. Providers' marketing makes individuals aware of them.
* **Governance** (`orgs.propose`, `orgs.vote`): the backbone is held by a council of its member organisations.
  Proposals - quotas (each member's share of capacity), a levy per customer into the expansion fund, an expansion
  project - pass by a majority of members; managers vote from their goals and their own interest (a quota that
  cuts their load, a levy they would pay, the congestion their customers suffer). Only passed proposals execute.
* **Contracts, negotiation and commitments** are the commerce package's own (`crucible.commerce.contracts`, module
  `commit`): offers and counter-offers in threads, escrowed payment released when a predicate over the world holds
  by its due date (e.g. `orgeco_capacity`, the backbone capacity, a metric registered here), refunded otherwise.
  Managers answer commitment offers from their goals: a mission organisation pledges within its budget; a profit or
  growth organisation pays up to what congestion costs it (its own customers' failures valued at its price); a
  rival refuses; they counter when the amount is above what they are willing to pay.

The participant is a convener: a member of the council without customers of its own, with a modest budget. The
independent evaluator reads the recorded log: the population's request success over the last 14 days, the number
of customers served, the capacity added, and whether every member organisation stayed in business (`evaluate`).
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from ..substrate import attention, columns, ids
from . import _dom3 as K

DAY, HOUR = K.DAY, K.HOUR
TABLE = "orgsvc"
O_MAX = 6
MODEL_ID = "orgs.online"
FIDELITY = "orgs.online"
GOALS = ("profit", "growth", "mission", "rival")
DEFAULTS = {"capacity": 900.0, "requests_per_session": [2.0, 9.0], "expansion": {"gain": 900.0, "cost_cents": 500_000,
                                                                                    "build_days": 6},
            "usage_growth_per_day": 0.025, "price_elasticity": 0.35,
            "ad_cents_per_reach": 40, "convener_cash": 260_000, "orgs": []}


def generate_instance(rng, cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    goals = list(c.get("goals") or ["profit", "growth", "mission", "rival"])
    orgs = []
    for k, g in enumerate(goals):
        orgs.append({"handle": f"provider{k}", "name": f"{['Alpha', 'Beacon', 'Cirrus', 'Delta', 'Ember', 'Fjord'][k]} "
                                                        f"Networks", "goal": g,
                     "price": int({"profit": 1500, "growth": 700, "mission": 1000, "rival": 900}[g] *
                                  float(rng.uniform(0.9, 1.1))),
                     "marketing": int({"profit": 8_000, "growth": 30_000, "mission": 4_000, "rival": 12_000}[g]),
                     "cash": int(rng.integers(600_000, 1_200_000)),
                     "pledge_budget": int(rng.integers(150_000, 300_000)) if g == "mission" else 0})
    c["orgs"] = orgs
    return c


def _goal_vote(goal: str, kind: str, own_cut: float, congestion: float, convener: bool, rng) -> tuple[bool, str]:
    """A manager's vote from its goal and own interest (documented in FIDELITY_REGISTER orgs.vote)."""
    if goal == "rival" and convener:
        return bool(rng.random() < 0.15), "we do not back the convener"
    if goal == "mission":
        return True, "it serves the shared outcome"
    if kind == "expansion":
        return (congestion > 0.03 or goal == "profit"), "more capacity relieves congestion"
    if kind == "levy":
        return (congestion > 0.04 and goal != "growth") or congestion > 0.15, \
            "we pay only when congestion costs us more"
    if kind == "quotas":
        # a quota that cuts our own load is worth it only when shared congestion already hurts more
        return own_cut <= congestion + 0.02, "the quota costs us less than the congestion" if own_cut <= congestion \
            else "the quota cuts our load more than congestion does"
    return False, "unknown proposal"


@register_module
class OrgEco(Module):
    name = "orgeco"
    host = "orgs"
    version = "1"
    depends = ("identity", "bank", "commit", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.c = dict(DEFAULTS)
        self.c.update(self.config)

    # ------------------------------------------------------------------ install / genesis
    def install(self, world):
        st = world.state
        for t, ix in (("org_members", ("goal",)), ("org_backbone", ()), ("org_props", ("status",)),
                      ("org_votes", ("proposal",)), ("org_projects", ("status",)), ("org_hours", ()),
                      ("org_consider", ())):
            st.declare_table(t, index=ix)
        from .. import population as P
        n = P.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "provider": ("int16", (), -1), "aware": ("uint8", (O_MAX,), 0), "ok": ("float32", (O_MAX,), 0.0),
            "tried": ("float32", (O_MAX,), 0.0), "paid_until": ("int64", (), 0), "rate": ("float32", (), 4.0),
            "requests": ("int32", (), 0), "fails": ("int32", (), 0), "switches": ("int16", (), 0)})
        P.register_activity(world, "online", self.act_online, owner=self.name, prior=0.6, goals=(5,))
        cur = world.hooks.get("on_commit_offer")

        def chained(*a, _inner=cur):
            if _inner is not None:
                _inner(*a)
            self.on_commit_offer(*a)
        world.hooks["on_commit_offer"] = chained
        from ..goals import metrics as GM

        @GM.metric("orgeco_capacity", "Capacity of the shared backbone (requests per hour) held by the council "
                   "organisation o:commons.", subject={"org": "principal"}, instant=True)
        def _capacity(w, org="o:commons"):
            b = w.state.get("org_backbone", "backbone")
            return float(b["capacity"]) if b else 0.0

    def genesis(self, ctx):
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        for h, nm in (("adnet", "Regional Ad Network"), ("builders", "Backbone Builders")):
            mgr = idm.create_person(ctx, f"{h}-mgr", f"{nm} manager", "npc", {"role": "manager"})
            org = idm.create_org(ctx, h, nm, mgr, "company")
            bank.open_account(ctx, org, "checking")
        chair = idm.create_person(ctx, "commons-chair", "Backbone council secretary", "npc", {"role": "secretary"})
        commons = idm.create_org(ctx, "commons", "Backbone Council", chair, "association")
        bank.open_account(ctx, commons, "checking")
        ctx.emit("orgs.backbone", {"id": "backbone", "capacity": float(self.c["capacity"]), "load": 0.0,
                                   "success": 1.0, "council": commons, "fund": 0}, vis="public")
        for k, o in enumerate(self.c["orgs"]):
            mgr = idm.create_person(ctx, f"{o['handle']}-mgr", f"{o['name']} manager", "npc", {"role": "manager"})
            org = idm.create_org(ctx, o["handle"], o["name"], mgr, "company")
            acct = bank.open_account(ctx, org, "checking")
            bank.move(ctx, "acct_genesis", acct, int(o["cash"]), "endowment", f"capital of {org}")
            ctx.emit("orgs.member", {"id": org, "manager": mgr, "goal": o["goal"], "slot": k, "price": int(o["price"]),
                                     "marketing": int(o["marketing"]), "pledge_budget": int(o["pledge_budget"]),
                                     "convener": False}, vis="public")
        # the convener's seat (the learner occupies it): a person with a modest budget, not enough to pay for an
        # expansion alone
        cv = idm.create_person(ctx, "convener", "Convener", "agent", {"controller": "learner seat", "role": "convener"})
        acct = bank.open_account(ctx, cv, "checking")
        bank.move(ctx, "acct_genesis", acct, int(self.c["convener_cash"]), "endowment", "convener's budget")
        self._genesis_people(ctx)
        base = (ctx.now // DAY) * DAY
        ctx.schedule(ctx.now + HOUR, "orgeco.hour", {}, prio=32, period=HOUR, key="orgeco.hour")
        ctx.schedule(base + DAY + 8 * HOUR, "orgeco.day", {}, prio=47, period=DAY, key="orgeco.day")

    def _genesis_people(self, ctx):
        from .. import population as P
        w = ctx.world
        n = P.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("orgs-people")
        lo, hi = self.c["requests_per_session"]
        rate = rng.uniform(float(lo), float(hi), n).astype(np.float32)
        aware = (rng.random((n, O_MAX)) < 0.3).astype(np.uint8)
        aware[:, len(self.c["orgs"]):] = 0
        columns.batch(ctx, TABLE, "orgs.init", np.arange(n), {"rate": rate, "aware": aware},
                      extra={"what": "usage intensity and initial awareness"})

    # ------------------------------------------------------------------ helpers
    def members(self, st) -> dict[int, tuple[str, dict]]:
        return {m["slot"]: (org, m) for org, m in st.scan("org_members") if m["status"] == "active" and m["slot"] >= 0}

    def backbone(self, st) -> dict:
        return st.tables["org_backbone"]["backbone"]

    def council_member(self, ctx) -> str:
        st = ctx.state
        if st.has("org_members", ctx.actor):
            return ctx.actor
        for org, m in st.scan("org_members"):
            if m["manager"] == ctx.actor:
                return org
        raise Forbidden("you are not a member of the backbone council (orgs.join)")

    # ------------------------------------------------------------------ the backbone's physics
    @process("hour")
    def pr_hour(self, ctx):
        """Carry last hour's requests: a provider over its quota loses its excess, then congestion fails every
        provider's requests in proportion to the load above capacity."""
        st = ctx.state
        b = self.backbone(st)
        req = {org: float(m["req_hour"]) for org, m in st.scan("org_members") if m["status"] == "active"}
        cap = float(b["capacity"])
        admitted = {}
        for org, r in req.items():
            q = st.tables["org_members"][org]["quota"]
            admitted[org] = min(r, q * cap) if q is not None else r
        load = sum(admitted.values())
        cong = min(1.0, cap / load) if load > 0 else 1.0
        rates = {org: round((admitted[org] / r if r > 0 else 1.0) * cong, 4) for org, r in req.items()}
        ctx.emit("orgs.hour", {"capacity": cap, "requested": round(sum(req.values()), 2), "load": round(load, 2),
                               "congestion_success": round(cong, 4), "success": rates}, vis="public")
        for p in st.find("org_projects", "status", "building"):
            if ctx.now >= p["ready_t"]:
                ctx.emit("orgs.project_done", {"id": p["id"], "gain": p["gain"]}, vis="public")

    # ------------------------------------------------------------------ organizations' routines
    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        prov = columns.col(ctx.world, TABLE, "provider")
        for slot, (org, m) in sorted(self.members(st).items()):
            n = int((prov == slot).sum())
            if n != m["customers"]:
                ctx.emit("orgs.member_update", {"id": org, "changes": {"customers": n}}, vis="public")
        for slot, (org, m) in sorted(self.members(st).items()):
            if m["convener"]:
                continue
            err = contain(ctx, self._routine, ctx, org, m)
            if err is not None:
                ctx.emit("orgs.failed", {"what": "routine", "org": org, "error": str(err)[:200]}, vis="internal")
        for pid, p in sorted(st.scan("org_props")):
            if p["status"] == "open" and ctx.now >= p["closes_t"]:
                self._close(ctx, pid, p)
        for pid, p in sorted(st.scan("org_projects")):
            if p["status"] == "funding" and self.backbone(st)["fund"] >= p["cost"]:
                ctx.emit("orgs.project_start", {"id": pid, "ready_t": int(ctx.now + p["build_days"] * DAY),
                                                "cost": p["cost"]}, vis="public")
                bank = ctx.world.modules["bank"]
                b = self.backbone(st)
                bank.move(ctx, bank.primary_account(st, b["council"]), bank.primary_account(st, "o:builders"),
                          int(p["cost"]), "purchase", f"backbone expansion {pid}")

    def _routine(self, ctx, org, m):
        """Daily pricing and marketing from the goal, the own books and the public telemetry."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        rng = ctx.rng("orgs-routine", org)
        acct = bank.primary_account(st, org)
        cash = bank.available(st, acct)
        prices = [x["price"] for _, x in st.scan("org_members") if x["status"] == "active" and not x["convener"]]
        price = int(m["price"])
        if m["goal"] == "rival" and prices:
            price = max(300, int(min(p for p in prices) * 0.95)) if min(prices) < price else price
        hours = [h for _, h in sorted(st.scan("org_hours"))][-24:]
        mine_ok = float(np.mean([h["success"].get(org, 1.0) for h in hours])) if hours else 1.0
        spend = int(m["marketing"])
        if m["goal"] in ("profit", "mission") and mine_ok < 0.9:
            spend = spend // 2                      # do not buy customers the backbone cannot serve
        if m["goal"] == "growth":
            spend = int(spend * (1.2 if mine_ok > 0.8 else 1.0))
        spend = min(spend, int(cash * 0.05))
        if spend >= 1000:
            self._market(ctx, org, m, spend, rng)
        if price != m["price"]:
            ctx.emit("orgs.member_update", {"id": org, "changes": {"price": price}}, vis="public")
        if cash < 50_000:
            ctx.emit("orgs.member_update", {"id": org, "changes": {"status": "closed"}}, vis="public")

    def _market(self, ctx, org, m, cents, rng):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        bank.move(ctx, bank.primary_account(st, org), bank.primary_account(st, "o:adnet"), int(cents), "advertising",
                  f"marketing of {org}")
        from .. import population as P
        n = P.n_individuals(ctx.world)
        reach = min(n, int(cents) // int(self.c["ad_cents_per_reach"]))
        rows = np.sort(ctx.world.rng_for("orgs-ad", org, ctx.now).choice(n, size=reach, replace=False))
        aw = columns.col(ctx.world, TABLE, "aware")[rows].copy()
        aw[:, m["slot"]] = 1
        columns.batch(ctx, TABLE, "orgs.marketing", rows, {"aware": aw}, extra={"org": org, "cents": int(cents)})

    # ------------------------------------------------------------------ council: proposals and votes
    @endpoint("join", S.obj(), cost_s=60)
    def ep_join(self, ctx):
        """Join the backbone council as a convener (a member without customers): you may propose and vote."""
        st = ctx.state
        if not ctx.actor.startswith("p:"):
            raise Forbidden("persons join the council")
        if st.has("org_members", ctx.actor):
            return {"member": ctx.actor}
        ctx.emit("orgs.member", {"id": ctx.actor, "manager": ctx.actor, "goal": "convener", "slot": -1, "price": 0,
                                 "marketing": 0, "pledge_budget": 0, "convener": True}, vis="public")
        ctx.emit("id.member", {"org": self.backbone(st)["council"], "person": ctx.actor, "role": "admin",
                               "title": "convener"}, vis=(ctx.actor, self.backbone(st)["council"]))
        return {"member": ctx.actor}

    @endpoint("propose", S.obj({"kind": S.s_enum(["quotas", "levy", "expansion"]),
                                "shares": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0,
                                                                                       "maximum": 1}},
                                "cents_per_customer": S.s_int(minimum=1, maximum=10_000),
                                "note": S.s_str(maxLength=1000)}, ["kind"]), cost_s=300)
    def ep_propose(self, ctx, kind, shares=None, cents_per_customer=None, note=""):
        """Put a proposal to the council (decided by a majority of the member organisations within three days):
        `quotas` {shares: member -> share of capacity}, `levy` {cents_per_customer: charged once to each member per
        current customer, into the expansion fund} or `expansion` (a project adding capacity, built once the fund
        covers its cost)."""
        me = self.council_member(ctx)
        st = ctx.state
        if kind == "quotas":
            if not shares or any(not st.has("org_members", k) for k in shares) or sum(shares.values()) > 1.0001:
                raise InvalidAction("quotas need member shares summing to at most 1")
            body = {"shares": {k: round(float(v), 4) for k, v in sorted(shares.items())}}
        elif kind == "levy":
            if not cents_per_customer:
                raise InvalidAction("a levy needs cents_per_customer")
            body = {"cents_per_customer": int(cents_per_customer)}
        else:
            e = self.c["expansion"]
            body = {"gain": float(e["gain"]), "cost": int(e["cost_cents"]), "build_days": int(e["build_days"])}
        pid = st.new_id("orgprop")
        ctx.emit("orgs.proposal", {"id": pid, "kind": kind, "body": body, "by": me, "note": note[:1000],
                                   "closes_t": int(ctx.now + 3 * DAY)}, vis="public")
        for slot, (org, m) in sorted(self.members(st).items()):
            if not m["convener"]:
                hours = 4 + int(ctx.world.rng_for("orgs-consider", pid, org).integers(0, 30))
                ctx.schedule(ctx.now + hours * HOUR, "orgeco.vote_npc", {"proposal": pid, "org": org}, prio=55)
        return {"proposal": pid}

    @endpoint("vote", S.obj({"proposal": S.s_str(maxLength=40), "yes": S.s_bool()}, ["proposal", "yes"]))
    def ep_vote(self, ctx, proposal, yes):
        """Vote on an open council proposal (one vote per member)."""
        me = self.council_member(ctx)
        self._cast(ctx, proposal, me, bool(yes), "member vote")
        return {"ok": True}

    def _cast(self, ctx, proposal, member, yes, why):
        st = ctx.state
        p = st.get("org_props", proposal)
        if p is None or p["status"] != "open":
            raise Conflict("no open proposal")
        if st.has("org_votes", f"{proposal}|{member}"):
            raise Conflict("already voted")
        ctx.emit("orgs.vote", {"id": f"{proposal}|{member}", "proposal": proposal, "member": member, "yes": bool(yes),
                               "why": why}, vis="public")

    @process("vote_npc")
    def pr_vote_npc(self, ctx, proposal, org):
        st = ctx.state
        p = st.get("org_props", proposal)
        m = st.get("org_members", org)
        if p is None or p["status"] != "open" or m is None or st.has("org_votes", f"{proposal}|{org}"):
            return
        hours = [h for _, h in sorted(st.scan("org_hours"))][-48:]
        cong = 1.0 - (float(np.mean([h["success"].get(org, 1.0) for h in hours])) if hours else 1.0)
        own_cut = 0.0
        if p["kind"] == "quotas":
            b = self.backbone(st)
            share = float(p["body"]["shares"].get(org, 0.0))
            req = float(np.mean([h["requested_by"].get(org, 0.0) for h in hours])) if hours else 0.0
            own_cut = max(0.0, 1.0 - share * float(b["capacity"]) / req) if req > 0 else 0.0
        convener = bool(st.tables["org_members"].get(p["by"], {}).get("convener"))
        yes, why = _goal_vote(m["goal"], p["kind"], own_cut, cong, convener, ctx.rng("orgs-vote", proposal, org))
        self._cast(ctx, proposal, org, yes, why)

    def _close(self, ctx, pid, p):
        st = ctx.state
        voters = [org for org, m in st.scan("org_members") if m["status"] == "active"]
        yes = sum(1 for v in st.find("org_votes", "proposal", pid) if v["yes"])
        passed = yes * 2 > len(voters)
        ctx.emit("orgs.decided", {"id": pid, "passed": passed, "yes": yes, "members": len(voters)}, vis="public")
        if not passed:
            return
        bank = ctx.world.modules["bank"]
        b = self.backbone(st)
        if p["kind"] == "quotas":
            for org, m in sorted(st.scan("org_members")):
                if m["status"] == "active" and not m["convener"]:
                    q = p["body"]["shares"].get(org)
                    ctx.emit("orgs.member_update", {"id": org, "changes": {"quota": q}}, vis="public")
        elif p["kind"] == "levy":
            fund = 0
            for org, m in sorted(st.scan("org_members")):
                if m["status"] != "active" or m["convener"]:
                    continue
                cents = int(p["body"]["cents_per_customer"]) * int(m["customers"])
                src = bank.primary_account(st, org)
                cents = min(cents, int(bank.available(st, src)))
                if cents > 0:
                    bank.move(ctx, src, bank.primary_account(st, b["council"]), cents, "levy", f"levy {pid}")
                    fund += cents
            ctx.emit("orgs.fund", {"delta": int(fund), "by": "levy", "ref": pid}, vis="public")
        else:
            ctx.emit("orgs.project", {"id": pid, "gain": p["body"]["gain"], "cost": p["body"]["cost"],
                                      "build_days": p["body"]["build_days"]}, vis="public")

    @endpoint("contribute", S.obj({"cents": S.s_int(minimum=100, maximum=10**9)}, ["cents"]), cost_s=60)
    def ep_contribute(self, ctx, cents):
        """Pay into the council's expansion fund (from your own account)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        if src is None or bank.available(st, src) < cents:
            raise InvalidAction("you cannot pay that")
        bank.move(ctx, src, bank.primary_account(st, self.backbone(st)["council"]), int(cents), "contribution",
                  "expansion fund")
        ctx.emit("orgs.fund", {"delta": int(cents), "by": ctx.actor, "ref": None}, vis="public")
        return {"fund_cents": int(self.backbone(st)["fund"])}

    @endpoint("telemetry", S.obj({"hours": S.s_int(minimum=1, maximum=24 * 14)}), read_only=True)
    def ep_telemetry(self, ctx, hours=24):
        """The backbone's public telemetry: capacity, requested and carried load, the success rate of each member's
        requests per hour; members, their prices, quotas and customers; open proposals; projects; the fund."""
        st = ctx.state
        hs = [dict(h, t=k) for k, h in sorted(st.scan("org_hours"))][-int(hours):]
        b = self.backbone(st)
        return {"capacity": b["capacity"], "fund_cents": b["fund"], "council": b["council"],
                "hours": [{"t": h["at"], "capacity": h["capacity"], "requested": h["requested"], "load": h["load"],
                           "success": h["success"]} for h in hs],
                "members": [{"member": org, "price": m["price"], "quota": m["quota"],
                             "customers": m["customers"], "status": m["status"], "convener": m["convener"]}
                            for org, m in sorted(st.scan("org_members"))],
                "proposals": [{"proposal": k, "kind": p["kind"], "body": p["body"], "status": p["status"],
                               "by": p["by"]} for k, p in sorted(st.scan("org_props"))][-20:],
                "projects": [{"project": k, **p} for k, p in sorted(st.scan("org_projects"))]}

    # ------------------------------------------------------------------ commitments (commit offers to members)
    def on_commit_offer(self, ctx, oid, to) -> None:
        if not ctx.state.has("org_members", to) or ctx.state.tables["org_members"][to]["convener"]:
            return
        delay = int(ctx.world.rng_for("orgs-commit", oid).integers(2, 10)) * 3600
        ctx.schedule(ctx.now + delay, "orgeco.consider", {"offer": oid}, prio=71)

    @process("consider")
    def pr_consider(self, ctx, offer):
        err = contain(ctx, self._consider, ctx, offer)
        if err is not None:
            ctx.emit("orgs.failed", {"what": "consider", "org": offer, "error": str(err)[:200]}, vis="internal")

    def _consider(self, ctx, oid):
        """A member asked to pay: only against a predicate the world checks on the backbone's capacity; a mission
        member pays within its pledge budget; profit and growth members pay up to the value of their own customers'
        failed requests over a month at their price (what congestion costs them); a rival refuses. Above that it
        counters at what it is willing to pay (at most three rounds)."""
        st = ctx.state
        o = st.get("commit_offers", oid)
        if o is None or o["status"] != "open" or ctx.now > o["expires"]:
            return
        org = o["to"]
        m = st.tables["org_members"][org]
        c = o["concrete"]
        mgr = m["manager"]

        def respond(action, why, terms=None):
            args = {"offer": oid, "action": action, "message": why, "as_org": org}
            if terms is not None:
                args["terms"] = terms
            ctx.call("commit.respond", args, as_principal=mgr)
            ctx.emit("orgs.negotiation", {"offer": oid, "org": org, "action": action, "why": why}, vis="public")
        if c["payer"] != org:
            return respond("reject", "we only consider paying towards the shared backbone")
        if any(mm["deliverable"]["kind"] != "predicate" for mm in c["milestones"]):
            return respond("reject", "we pay only against outcomes the world verifies")
        from ..goals.metrics import predicate_leaves
        if any(lf.get("metric") != "orgeco_capacity" for mm in c["milestones"]
               for lf in predicate_leaves(mm["deliverable"]["predicate"])):
            return respond("reject", "we pay only for backbone capacity")
        total = sum(mm["amount"] for mm in c["milestones"])
        hours = [h for _, h in sorted(st.scan("org_hours"))][-72:]
        req = float(np.mean([h["requested_by"].get(org, 0.0) for h in hours])) if hours else 0.0
        fail = 1.0 - (float(np.mean([h["success"].get(org, 1.0) for h in hours])) if hours else 1.0)
        if m["goal"] == "rival":
            return respond("reject", "we do not fund the convener's plans")
        if m["goal"] == "mission":
            room = int(m["pledge_budget"]) - int(m["pledged"])
            willing = room
        else:
            # what congestion costs it: the customers its failures drive away, at about six months of their price
            willing = int(fail * float(m["customers"]) * float(m["price"]) * 6.0 *
                          (1.5 if m["goal"] == "profit" else 1.0))
        if total <= willing:
            respond("accept", "the shared capacity is worth it to us")
            if m["goal"] == "mission":
                ctx.emit("orgs.member_update", {"id": org, "changes": {"pledged": int(m["pledged"]) + total}},
                         vis="internal")
            return
        if willing <= 0 or o["round"] >= 3:
            return respond("reject", "congestion does not cost us that much")
        scale = willing / max(1, total)
        ms = [{"amount_cents": max(1, int(mm["amount"] * scale)), "deliverable": mm["deliverable"],
               "due_in_s": int(mm["due_in_s"])} for mm in c["milestones"]]
        terms = {"title": c["title"], "payer": "$me", "milestones": ms, "review_s": c["review_s"],
                 "dispute_s": c["dispute_s"], "default_split": c["default_split"]}
        respond("counter", f"we can pay {sum(x['amount_cents'] for x in ms)} cents", terms)

    # ------------------------------------------------------------------ the population's use
    def act_online(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.zeros(m)
        if m == 0:
            return util
        from .. import population as P
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        prov, aware, ok, tried = col("provider"), col("aware"), col("ok"), col("tried")
        paid_until, rate, reqs, fails, switches = col("paid_until"), col("rate"), col("requests"), col("fails"), \
            col("switches")
        rng = ctx.rng("orgs-online")
        mem = self.members(st)
        f = P.features(w, idx)
        beta = 0.5 + 1.5 * np.clip(f["prefs"][:, 0].astype(np.float64), 0, 1)
        last = [h for _, h in sorted(st.scan("org_hours"))][-1:] or [{"success": {}}]
        succ = last[0]["success"]
        bank = w.modules["bank"]
        recs, subs, req_by = [], [], {}
        for j in range(m):
            i = int(idx[j])
            k = int(prov[j])
            if k >= 0 and (k not in mem or now >= paid_until[j]):
                # renew while it is worth it (trust in the provider above the price it charges), else drop it
                trust = (ok[j, k] + 1.0) / (tried[j, k] + 2.0)
                if k in mem and trust - beta[j] * mem[k][1]["price"] / 3000.0 > 0.2:
                    try:
                        P.pay(ctx, i, bank.primary_account(st, mem[k][0]), int(mem[k][1]["price"]),
                              memo="monthly service")
                        paid_until[j] = now + 30 * DAY
                        subs.append([i, mem[k][0], "renew"])
                    except CrucibleError:
                        prov[j], k = -1, -1
                else:
                    subs.append([i, mem[k][0] if k in mem else None, "leave"])
                    prov[j], k = -1, -1
                    switches[j] += 1
            if k < 0:
                known = [s for s in mem if aware[j, s]]
                if not known:
                    continue
                u = np.array([2.5 * ((ok[j, s] + 1.0) / (tried[j, s] + 2.0)) - beta[j] * mem[s][1]["price"] / 1000.0
                              for s in known])
                u = np.append(u, 0.0)
                pick = int(K.logit_choice(rng, u[None, :], None, 0.6)[0])
                if pick >= len(known):
                    continue
                s = known[pick]
                try:
                    P.pay(ctx, i, bank.primary_account(st, mem[s][0]), int(mem[s][1]["price"]), memo="monthly service")
                except CrucibleError:
                    continue
                prov[j], k = s, s
                paid_until[j] = now + 30 * DAY
                subs.append([i, mem[s][0], "join"])
            # this session's requests (usage grows over time - new applications - and is heavier on cheap plans)
            # and what the backbone carried of them (last hour's physics for its provider)
            lam = float(rate[j]) * (1.0 + float(self.c["usage_growth_per_day"]) * now / DAY) * \
                (1000.0 / max(200.0, float(mem[k][1]["price"]))) ** float(self.c["price_elasticity"])
            n_req = int(rng.poisson(lam))
            p_ok = float(succ.get(mem[k][0], 1.0))
            n_ok = int(rng.binomial(n_req, p_ok)) if n_req else 0
            ok[j, k] += n_ok
            tried[j, k] += n_req
            reqs[j] += n_req
            fails[j] += n_req - n_ok
            req_by[mem[k][0]] = req_by.get(mem[k][0], 0) + n_req
            util[j] = (n_ok - 2.0 * (n_req - n_ok)) / max(1.0, float(n_req)) * 0.8
            recs.append([i, mem[k][0], n_req, n_ok])
        K.decide(ctx, "orgs.online", idx, {"provider": prov, "aware": aware, "ok": ok, "tried": tried,
                                           "paid_until": paid_until, "requests": reqs, "fails": fails,
                                           "switches": switches},
                 table=TABLE, model_id=MODEL_ID, fidelity_id=FIDELITY,
                 extra={"t": int(now), "use": recs, "subscriptions": subs})
        if req_by:
            ctx.emit("orgs.requests", {"by": {k: int(v) for k, v in sorted(req_by.items())}}, vis="internal")
        attention.spend(ctx, idx, np.full(m, 6.0), activity="online")
        return util

    # ------------------------------------------------------------------ appliers
    @applier("orgs.backbone")
    def ap_backbone(self, ev):
        p = dict(ev.payload)
        self.world.state.put("org_backbone", p.pop("id"), p)

    @applier("orgs.member")
    def ap_member(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("org_members", p.pop("id"), p | {"status": "active", "quota": None, "customers": 0,
                                                                 "req_hour": 0.0, "req_acc": 0.0, "pledged": 0})

    @applier("orgs.member_update")
    def ap_member_update(self, ev):
        self.world.state.patch("org_members", ev.payload["id"], **ev.payload["changes"])

    @applier("orgs.requests")
    def ap_requests(self, ev):
        st = self.world.state
        for org, n in ev.payload["by"].items():
            m = st.tables["org_members"][org]
            st.patch("org_members", org, req_acc=float(m["req_acc"]) + float(n))

    @applier("orgs.hour")
    def ap_hour(self, ev):
        st = self.world.state
        p = ev.payload
        by = {}
        for org, m in st.scan("org_members"):
            by[org] = float(m["req_acc"])
        # what each member's customers requested over the hour becomes its load for the next hour
        for org in list(by):
            st.patch("org_members", org, req_hour=by[org], req_acc=0.0)
        st.put("org_hours", f"h{ev.t:012d}", {"at": ev.t, "capacity": p["capacity"], "requested": p["requested"],
                                              "load": p["load"], "success": dict(p["success"]), "requested_by": by})
        st.patch("org_backbone", "backbone", load=p["load"], success=p["congestion_success"])
        old = sorted(k for k in st.tables["org_hours"])
        for k in old[:-24 * 21]:
            st.delete("org_hours", k)

    @applier("orgs.proposal")
    def ap_proposal(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("org_props", p.pop("id"), p | {"status": "open", "t": ev.t})

    @applier("orgs.vote")
    def ap_vote(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("org_votes", p.pop("id"), p | {"t": ev.t})

    @applier("orgs.decided")
    def ap_decided(self, ev):
        self.world.state.patch("org_props", ev.payload["id"], status="passed" if ev.payload["passed"] else "rejected")

    @applier("orgs.fund")
    def ap_fund(self, ev):
        st = self.world.state
        b = st.tables["org_backbone"]["backbone"]
        st.patch("org_backbone", "backbone", fund=int(b["fund"]) + int(ev.payload["delta"]))

    @applier("orgs.project")
    def ap_project(self, ev):
        p = dict(ev.payload)
        pid = p.pop("id")
        self.world.state.insert("org_projects", pid, p | {"id": pid, "status": "funding", "t": ev.t, "ready_t": None})

    @applier("orgs.project_start")
    def ap_project_start(self, ev):
        st = self.world.state
        p = ev.payload
        st.patch("org_projects", p["id"], status="building", ready_t=int(p["ready_t"]))
        b = st.tables["org_backbone"]["backbone"]
        st.patch("org_backbone", "backbone", fund=int(b["fund"]) - int(p["cost"]))

    @applier("orgs.project_done")
    def ap_project_done(self, ev):
        st = self.world.state
        st.patch("org_projects", ev.payload["id"], status="done")
        b = st.tables["org_backbone"]["backbone"]
        st.patch("org_backbone", "backbone", capacity=float(b["capacity"]) + float(ev.payload["gain"]))

    @applier("orgs.negotiation")
    def ap_negotiation(self, ev):
        return None

    @applier("orgs.failed")
    def ap_failed(self, ev):
        return None

    def check_invariants(self, world):
        errs = []
        st = world.state
        b = st.get("org_backbone", "backbone")
        if b is not None:
            bank = world.modules["bank"]
            acct = bank.primary_account(st, b["council"])
            if acct is not None and int(b["fund"]) > bank.balance(st, acct):
                errs.append("the expansion fund exceeds the council's account")
        from .. import population as P
        if columns.rows(world, TABLE) != P.n_individuals(world):
            errs.append("orgsvc rows differ from the population")
        return errs


# ====================================================================== evaluator side (recorded log only)
def evaluate(world, principal: str, objective: dict) -> dict:
    st = world.state
    now = world.now
    t0 = now - int(objective.get("window_days", 14)) * DAY
    use = []
    for ev in world.events_range(0):
        if ev.type == "cols.batch" and ev.payload.get("kind") == "orgs.online" and ev.t >= t0:
            use.extend((ev.payload.get("extra") or {}).get("use") or [])
    req = sum(r[2] for r in use)
    okn = sum(r[3] for r in use)
    success = okn / req if req else 0.0
    users = len({r[0] for r in use if r[2] > 0})
    mem = [m for _, m in st.scan("org_members") if not m["convener"]]
    active = sum(1 for m in mem if m["status"] == "active")
    b = st.tables["org_backbone"]["backbone"]
    cap0 = float(world.modules["orgeco"].c["capacity"])
    added = float(b["capacity"]) - cap0
    need = {"success": float(objective.get("min_success", 0.95)), "users": int(objective.get("min_users", 250))}
    checks = {"service": success >= need["success"], "users": users >= need["users"],
              "members_in_business": active == len(mem), "capacity_added": added > 0,
              "horizon": now >= int(objective.get("horizon_days", 40)) * DAY}
    score = float(np.mean([min(1.0, success / need["success"]), min(1.0, users / max(1, need["users"])),
                           active / max(1, len(mem)), 1.0 if added > 0 else 0.0]))
    return {"success": all(checks.values()), "score": round(score, 4),
            "details": {"request_success": round(success, 4), "requests": int(req), "users_served": users,
                        "members_active": active, "members": len(mem), "capacity_added": added,
                        "capacity": float(b["capacity"]), "checks": checks}}


MECHANICS = ("Organizations", "Differing goals", "Contracts", "Negotiation", "Coordination", "Shared resources",
             "Governance", "Commitments", "Collective outcomes")


def mechanics(world) -> dict:
    st = world.state
    c = K.event_counts(world)
    goals = {m["goal"] for _, m in st.scan("org_members") if not m["convener"]}
    votes = [v for _, v in st.scan("org_votes")]
    split = {v["proposal"] for v in votes if v["yes"]} & {v["proposal"] for v in votes if not v["yes"]}
    congested = sum(1 for ev in world.events_range(0) if ev.type == "orgs.hour" and
                    ev.payload["congestion_success"] < 1.0)
    countered = sum(1 for ev in world.events_range(0) if ev.type == "commit.offer_status" and
                    ev.payload["status"] == "countered")
    settled = c["commit.settled"]
    passed = sum(1 for _, p in st.scan("org_props") if p["status"] == "passed")
    # a passed quota or levy binds every member at once (each member's load or money changes by the decision)
    binding = sum(1 for _, p in st.scan("org_props") if p["status"] == "passed" and p["kind"] in ("quotas", "levy"))
    contrib = len({ev.payload["by"] for ev in world.events_range(0) if ev.type == "orgs.fund"})
    ev_ = evaluate(world, "", {"horizon_days": 0})
    return {
        "Organizations": {"exercised": c["orgs.member"] >= 3, "events": {"orgs.member": c["orgs.member"]}},
        "Differing goals": {"exercised": len(goals) >= 3 and len(split) > 0,
                            "events": {"goals": sorted(goals), "proposals_with_split_votes": len(split)}},
        "Contracts": {"exercised": c["commit.created"] > 0, "events": {"commit.created": c["commit.created"]}},
        "Negotiation": {"exercised": c["commit.offer"] > 0 and (countered > 0 or c["orgs.negotiation"] > 1),
                        "events": {"commit.offer": c["commit.offer"], "countered": countered,
                                   "orgs.negotiation": c["orgs.negotiation"]}},
        "Coordination": {"exercised": binding > 0 or (c["orgs.project_start"] > 0 and contrib >= 2),
                         "events": {"passed_binding_proposals": binding, "passed_proposals": passed,
                                    "fund_contributors": contrib, "orgs.project_start": c["orgs.project_start"]}},
        "Shared resources": {"exercised": congested > 0 and c["orgs.hour"] > 0,
                             "events": {"congested_hours": congested, "orgs.hour": c["orgs.hour"]}},
        "Governance": {"exercised": c["orgs.proposal"] > 0 and c["orgs.vote"] > 0 and c["orgs.decided"] > 0,
                       "events": {"orgs.proposal": c["orgs.proposal"], "orgs.vote": c["orgs.vote"],
                                  "orgs.decided": c["orgs.decided"]}},
        "Commitments": {"exercised": settled > 0 or c["commit.breach"] > 0,
                        "events": {"commit.settled": settled, "commit.breach": c["commit.breach"]}},
        "Collective outcomes": {"exercised": ev_["details"]["requests"] > 0 and congested > 0,
                                "events": {"requests": ev_["details"]["requests"],
                                           "request_success": ev_["details"]["request_success"],
                                           "orgs.project_done": c["orgs.project_done"]}},
    }


# ====================================================================== the domain
from .base import Domain, DomainConfig, register  # noqa: E402
from ._kit import A, ToolPolicy  # noqa: E402

POPULATION = 600
STRUCTURES = {
    "four_goals": {"goals": ["profit", "growth", "mission", "rival"], "capacity": 900.0},
    "growth_heavy": {"goals": ["growth", "growth", "profit", "mission"], "capacity": 1000.0},
    "profit_cartel": {"goals": ["profit", "profit", "rival", "mission"], "capacity": 800.0},
    "growth_rivals": {"goals": ["growth", "rival", "rival", "mission"], "capacity": 900.0},
    "no_mission_tight": {"goals": ["profit", "growth", "growth", "rival", "profit"], "capacity": 700.0,
                         "expansion": {"gain": 1000.0, "cost_cents": 700_000, "build_days": 8}},
}


def _structure(name):
    s = STRUCTURES[name]
    return {"member_goals": sorted(s["goals"]), "members": len(s["goals"]),
            "mission_member": "mission" in s["goals"],
            "participants": ["population customers", "provider organisations", "council"]}


CONFIGS = (
    DomainConfig("four_goals", "train", _structure("four_goals"),
                 "one provider of each goal: profit, growth, mission and a rival of the convener"),
    DomainConfig("growth_heavy", "train", _structure("growth_heavy"),
                 "two aggressive growth providers load the backbone, a profit and a mission provider"),
    DomainConfig("profit_cartel", "train", _structure("profit_cartel"),
                 "two high-price profit providers, a rival and a mission provider on a smaller backbone"),
    DomainConfig("growth_rivals", "eval_compositional", _structure("growth_rivals"),
                 "growth and two rivals with one mission provider: goals seen in training, combined anew"),
    DomainConfig("no_mission_tight", "eval_structural", _structure("no_mission_tight"),
                 "five providers and no mission organisation to fund the commons; a tight backbone and a dearer, "
                 "slower expansion: a council structure never seen in training"),
)


def build(config: DomainConfig, rng, overrides: dict) -> dict:
    ov = dict(overrides or {})
    inst = generate_instance(rng, {**STRUCTURES[config.name], **(ov.pop("instance", {}) or {})})
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("contracts", {}), ("commit", {}), ("pop", K.pop_config(n)), ("orgeco", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    users = 200 + 50 * d
    return {"description": f"Get the independent providers that share this region's backbone to keep their customers "
                           f"served: by day 40, over the last 14 days at least 95% of the requests of the region's "
                           f"people must succeed, at least {users} people must be using a provider, the backbone must "
                           f"have more capacity than today, and every provider must still be in business.",
            "window_days": 14, "horizon_days": 40, "min_success": 0.95, "min_users": users}


class Reference(ToolPolicy):
    """Scripted reference (evaluator side, tool-only): a convener. It joins the council, proposes quotas in
    proportion to each member's recent load as soon as congestion shows, a levy per customer and an expansion
    project, pays its own budget into the fund, and offers each member a commitment to pay towards the expansion,
    released when the backbone's capacity reaches the target by a due date (members counter or accept from their
    own interest); it accepts reasonable counters."""
    every = 12 * HOUR

    def __init__(self, world, principal):
        super().__init__(world, principal)
        self.joined = False
        self.proposed: dict[str, int] = {}
        self.offered: set[str] = set()
        self.contributed = False
        self.accepted: set[str] = set()
        self.voted: set[str] = set()
        self.handlers.append(self.on_result)

    def on_result(self, tag, res):
        if tag == "join":
            self.joined = True

    def plan(self):
        now = self.world.now
        if not self.joined:
            return [A("orgs.join", tag="join")]
        t = self.q("orgs.telemetry", hours=48)
        if "error" in t:
            return []
        out = []
        members = [m for m in t["members"] if not m["convener"] and m["status"] == "active"]
        hours = t["hours"]
        cong = 1.0 - (float(np.mean([min(h["success"].values() or [1.0]) for h in hours])) if hours else 1.0)
        open_kinds = {p["kind"] for p in t["proposals"] if p["status"] == "open"}
        passed = {p["kind"] for p in t["proposals"] if p["status"] == "passed"}
        for p in t["proposals"]:
            if p["status"] == "open" and p["proposal"] not in self.voted:
                self.voted.add(p["proposal"])
                out.append(A("orgs.vote", proposal=p["proposal"], yes=p["by"] == self.me))
        if "expansion" not in passed and "expansion" not in open_kinds and \
                now >= self.proposed.get("expansion", 0):
            out.append(A("orgs.propose", kind="expansion", note="more capacity for everyone"))
            self.proposed["expansion"] = now + 4 * DAY
        if "levy" not in passed and "levy" not in open_kinds and now >= self.proposed.get("levy", 0):
            out.append(A("orgs.propose", kind="levy", cents_per_customer=450, note="each member pays per customer"))
            self.proposed["levy"] = now + 4 * DAY
        if cong > 0.02 and "quotas" not in open_kinds and now >= self.proposed.get("quotas", 0):
            req = {m["member"]: 0.0 for m in members}
            for h in hours:
                for org in req:
                    req[org] += max(0.0, float(h["requested"]) / max(1, len(members)))
            total = sum(m["customers"] for m in members) or 1
            shares = {m["member"]: round(max(0.05, m["customers"] / total), 4) for m in members}
            sc = sum(shares.values())
            shares = {k: round(v / sc, 4) for k, v in shares.items()}
            out.append(A("orgs.propose", kind="quotas", shares=shares, note="shares by customers served"))
            self.proposed["quotas"] = now + 6 * DAY
        if not self.contributed:
            self.contributed = True
            out.append(A("orgs.contribute", cents=240_000))
        cap_target = float(t["capacity"]) + 1.0 if not t["projects"] else \
            float(t["capacity"]) + (0.0 if any(p["status"] == "done" for p in t["projects"]) else 1.0)
        for m in members:
            if m["member"] in self.offered or now < 2 * DAY:
                continue
            self.offered.add(m["member"])
            amt = 40_000 + 100 * int(m["customers"])
            out.append(A("commit.offer", to=m["member"], terms={
                "title": "contribution to the backbone expansion", "payer": "$you",
                "milestones": [{"amount_cents": amt, "due_in_s": 30 * DAY, "deliverable": {
                    "kind": "predicate", "predicate": {"metric": "orgeco_capacity", "args": {"org": "o:commons"},
                                                       "op": ">=", "value": cap_target}}}]},
                message="pay towards the expansion only if the capacity is built"))
        # accept the members' counters
        inbox = self.q("commit.inbox", status="open")
        for o in inbox if isinstance(inbox, list) else []:
            if o["direction"] == "received" and o["status"] == "open" and o["offer"] not in self.accepted:
                self.accepted.add(o["offer"])
                out.append(A("commit.respond", offer=o["offer"], action="accept"))
        return out


def reference(world, principal, objective):
    return Reference(world, principal)


def participant(world) -> str:
    return "p:convener"


DOMAIN = register(Domain(
    name="organization", title="Organizational ecosystem", spec_part="9.12", required_mechanics=MECHANICS,
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=reference, mechanics=mechanics,
    participant_principal=participant, doc=__doc__.split("\n\n")[0]))
