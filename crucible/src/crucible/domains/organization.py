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
* **Governance** is the commerce package's institutions (`crucible.commerce.institutions`, module `inst`): the
  backbone is held by the council organisation `o:commons`, whose charter (adopted at genesis) decides every action
  in its name by a majority of its members within three days (each provider's manager represents its
  organisation; membership is open to a convener, up to a cap). The council's binding acts are tools only the
  council itself can run, and only as the execution of an approved `inst` decision (`inst.propose` with action
  `{kind: tool, tool: orgs.set_quotas | orgs.levy | orgs.expand, args}`): quotas (each member's share of
  capacity), a levy per customer into the expansion fund, an expansion project. Managers vote (`inst.vote`) from
  their goals and their own interest (a quota that cuts their load, a levy they would pay, the congestion their
  customers suffer; anything else they reject). The charter also binds the council's money: nothing leaves its
  accounts without a decision (`inst`'s posting guard).
* **Contracts, negotiation and commitments** are the commerce package's own (`crucible.commerce.contracts`, module
  `commit`): offers and counter-offers in threads, escrowed payment released when a predicate over the world holds
  by its due date (e.g. `orgeco_capacity` of the paying member: the capacity of the backbone it uses, a metric
  registered here), refunded otherwise.
  Managers answer commitment offers from their goals: a mission organisation pledges within its budget; a profit or
  growth organisation pays up to what congestion costs it (its own customers' failures valued at its price); a
  rival refuses; they counter when the amount is above what they are willing to pay.

The participant is a convener: it may join the council (`inst.join`) without customers of its own, with a modest
budget. The
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
        yes = congestion > 0.03 or goal == "profit"
        return yes, "more capacity relieves congestion" if yes else "congestion does not hurt us enough yet"
    if kind == "levy":
        yes = (congestion > 0.04 and goal != "growth") or congestion > 0.15
        return yes, "congestion costs us more than the levy" if yes else "we pay only when congestion costs us more"
    if kind == "quotas":
        # a quota that cuts our own load is worth it only when shared congestion already hurts more
        return own_cut <= congestion + 0.02, "the quota costs us less than the congestion" if own_cut <= congestion \
            else "the quota cuts our load more than congestion does"
    return False, "unknown proposal"


# the council's governed acts (tools only `o:commons` runs, as the execution of an approved inst decision) and the
# proposal kind a manager weighs each as
GOVERNED = {"orgs.set_quotas": "quotas", "orgs.levy": "levy", "orgs.expand": "expansion"}
COMMONS = "o:commons"


def council_charter(n_members: int) -> dict:
    """The council's `inst` charter: one majority procedure of the members (quorum one half, three days) decides
    every action in the council's name, money leaving its accounts included; membership is open up to the
    providers' representatives, the secretary and two conveners."""
    council = {"type": "majority", "electorate": "members", "threshold": 0.5, "quorum": 0.5, "window_s": 3 * DAY}
    return {"procedures": {"council": council},
            "rules": [{"match": {"kind": "tool", "tool": "orgs.*"}, "procedure": "council"},
                      {"match": {"kind": "tool", "tool": "*", "min_amount": 1}, "procedure": "council"},
                      {"match": {"kind": "*"}, "procedure": "council"}],
            "default": "council", "membership": {"admission": "open", "max_members": int(n_members) + 3}}


@register_module
class OrgEco(Module):
    name = "orgeco"
    host = "orgs"
    version = "1"
    depends = ("identity", "bank", "commit", "inst", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.c = dict(DEFAULTS)
        self.c.update(self.config)

    # ------------------------------------------------------------------ install / genesis
    def install(self, world):
        st = world.state
        for t, ix in (("org_members", ("goal",)), ("org_backbone", ()), ("org_projects", ("status",)),
                      ("org_hours", ()), ("org_consider", ()), ("org_use", ())):
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
        cur_d = world.hooks.get("on_inst_decision")

        def decision(ctx, did, voter, _inner=cur_d):
            if _inner is not None:
                _inner(ctx, did, voter)
            self.on_inst_decision(ctx, did, voter)
        world.hooks["on_inst_decision"] = decision
        from ..goals import metrics as GM

        @GM.metric("orgeco_capacity", "Capacity (requests per hour) of the shared backbone an organisation carries "
                   "its customers over (a member provider) or holds (the council o:commons); 0 for any other.",
                   subject={"org": "principal"}, instant=True)
        def _capacity(w, org=COMMONS):
            b = w.state.get("org_backbone", "backbone")
            if b is None or not (org == COMMONS or w.state.has("org_members", org)):
                return 0.0
            return float(b["capacity"])

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
        assert commons == COMMONS
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
            # the provider's manager represents it on the council (an ordinary identity membership)
            idm.add_member(ctx, commons, mgr, "member", f"representative of {org}")
        # the council governs itself by its charter (adopted by its secretary, the founding owner)
        ctx.call("inst.adopt", {"org": commons, "charter": council_charter(len(self.c["orgs"]))}, as_principal=chair)
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

    def manager_of(self, st, person: str) -> str | None:
        """The provider organisation a person manages (None for anyone else)."""
        for org, m in st.scan("org_members"):
            if m["manager"] == person:
                return org
        return None

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

    # ------------------------------------------------------------------ council: governed acts and votes
    def _governed(self, ctx) -> None:
        """The council's binding acts run only as the council organisation itself executing an approved `inst`
        decision (its charter governs every `orgs.*` tool in its name)."""
        if ctx.actor != COMMONS or not ctx.world.modules["inst"].executing(ctx.world, COMMONS):
            raise Forbidden("only the council decides this: propose it with inst.propose {org: o:commons, action: "
                            "{kind: tool, tool: ..., args: ...}}")
        ident(ctx.world).authorize(ctx, COMMONS, "orgs.govern")

    def _decision(self, ctx) -> str | None:
        """The council decision being executed: the latest open one whose action is this tool call."""
        st = ctx.state
        for did in reversed(st.find_ids("inst_decisions", "org", COMMONS)):
            d = st.tables["inst_decisions"][did]
            a = d["action"]
            if d["status"] == "open" and a.get("kind") == "tool" and a.get("tool") == ctx.tool and \
                    (a.get("args") or {}) == (ctx.tool_args or {}):
                return did
        return None

    @endpoint("set_quotas", S.obj({"shares": {"type": "object", "additionalProperties": {
        "type": "number", "minimum": 0, "maximum": 1}}}, ["shares"]), cost_s=60)
    def ep_set_quotas(self, ctx, shares):
        """Council act (only as an approved decision of o:commons): set each member organisation's share of the
        backbone's capacity (members left out lose their quota; shares sum to at most 1)."""
        self._governed(ctx)
        st = ctx.state
        if any(not st.has("org_members", k) for k in shares) or sum(shares.values()) > 1.0001:
            raise InvalidAction("quotas need member organisations' shares summing to at most 1")
        q = {k: round(float(v), 4) for k, v in sorted(shares.items())}
        for org, m in sorted(st.scan("org_members")):
            if m["status"] == "active":
                ctx.emit("orgs.member_update", {"id": org, "changes": {"quota": q.get(org)}}, vis="public")
        ctx.emit("orgs.act", {"kind": "quotas", "decision": self._decision(ctx), "body": {"shares": q}}, vis="public")
        return {"quotas": q}

    @endpoint("levy", S.obj({"cents_per_customer": S.s_int(minimum=1, maximum=10_000)}, ["cents_per_customer"]),
              cost_s=60)
    def ep_levy(self, ctx, cents_per_customer):
        """Council act (only as an approved decision of o:commons): charge every member organisation once per
        current customer into the expansion fund (each pays what its account holds, up to its levy)."""
        self._governed(ctx)
        st = ctx.state
        bank = ctx.world.modules["bank"]
        b = self.backbone(st)
        fund = 0
        for org, m in sorted(st.scan("org_members")):
            if m["status"] != "active":
                continue
            cents = int(cents_per_customer) * int(m["customers"])
            src = bank.primary_account(st, org)
            cents = min(cents, int(bank.available(st, src)))
            if cents > 0:
                bank.move(ctx, src, bank.primary_account(st, b["council"]), cents, "levy", "council levy")
                fund += cents
        ctx.emit("orgs.fund", {"delta": int(fund), "by": "levy", "ref": self._decision(ctx)}, vis="public")
        ctx.emit("orgs.act", {"kind": "levy", "decision": self._decision(ctx),
                              "body": {"cents_per_customer": int(cents_per_customer), "raised": int(fund)}},
                 vis="public")
        return {"raised_cents": int(fund)}

    @endpoint("expand", S.obj(), cost_s=60)
    def ep_expand(self, ctx):
        """Council act (only as an approved decision of o:commons): commission an expansion project (built by the
        contractor once the expansion fund covers its cost)."""
        self._governed(ctx)
        st = ctx.state
        if any(p["status"] in ("funding", "building") for _, p in st.scan("org_projects")):
            raise Conflict("an expansion project is already under way")
        e = self.c["expansion"]
        pid = st.new_id("orgprj")
        ctx.emit("orgs.project", {"id": pid, "gain": float(e["gain"]), "cost": int(e["cost_cents"]),
                                  "build_days": int(e["build_days"]), "decision": self._decision(ctx)}, vis="public")
        ctx.emit("orgs.act", {"kind": "expansion", "decision": self._decision(ctx), "body": {"project": pid}},
                 vis="public")
        return {"project": pid}

    def on_inst_decision(self, ctx, did, voter) -> None:
        """A provider's manager asked to vote on a council decision considers it within a day or so."""
        st = ctx.state
        d = st.get("inst_decisions", did)
        if d is None or d["org"] != COMMONS:
            return
        org = self.manager_of(st, voter)
        if org is None:
            return
        hours = 4 + int(ctx.world.rng_for("orgs-consider", did, org).integers(0, 30))
        ctx.schedule(ctx.now + hours * HOUR, "orgeco.vote_npc", {"proposal": did, "org": org}, prio=55)

    @process("vote_npc")
    def pr_vote_npc(self, ctx, proposal, org):
        st = ctx.state
        d = st.get("inst_decisions", proposal)
        m = st.get("org_members", org)
        if d is None or d["status"] != "open" or m is None or m["status"] != "active" or \
                st.has("inst_ballots", f"{proposal}|{m['manager']}"):
            return
        a = d["action"]
        kind = GOVERNED.get(a.get("tool")) if a.get("kind") == "tool" else None
        body = a.get("args") or {}
        hours = [h for _, h in sorted(st.scan("org_hours"))][-48:]
        cong = 1.0 - (float(np.mean([h["success"].get(org, 1.0) for h in hours])) if hours else 1.0)
        own_cut = 0.0
        if kind == "quotas":
            b = self.backbone(st)
            share = float((body.get("shares") or {}).get(org, 0.0))
            req = float(np.mean([h["requested_by"].get(org, 0.0) for h in hours])) if hours else 0.0
            own_cut = max(0.0, 1.0 - share * float(b["capacity"]) / req) if req > 0 else 0.0
        # a proposal by a convener (a council member that represents no provider and is not its secretary)
        convener = d["by"].startswith("p:") and self.manager_of(st, d["by"]) is None and \
            d["by"] not in ident(ctx.world).org_owner_persons(st, COMMONS)
        yes, why = _goal_vote(m["goal"], kind or "other", own_cut, cong, convener,
                              ctx.rng("orgs-vote", proposal, org))
        err = contain(ctx, ctx.call, "inst.vote", {"decision": proposal, "choice": "yes" if yes else "no"},
                      m["manager"])
        ctx.emit("orgs.vote_reason", {"decision": proposal, "org": org, "yes": bool(yes), "why": why,
                                      "error": str(err)[:200] if err is not None else None}, vis="public")

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
        requests per hour; members, their prices, quotas and customers; projects; the fund. The council's decisions
        are its members' business (inst.decisions {org: o:commons})."""
        st = ctx.state
        hs = [dict(h, t=k) for k, h in sorted(st.scan("org_hours"))][-int(hours):]
        b = self.backbone(st)
        return {"capacity": b["capacity"], "fund_cents": b["fund"], "council": b["council"],
                "governed_acts": sorted(GOVERNED),
                "hours": [{"t": h["at"], "capacity": h["capacity"], "requested": h["requested"], "load": h["load"],
                           "success": h["success"]} for h in hs],
                "members": [{"member": org, "price": m["price"], "quota": m["quota"],
                             "customers": m["customers"], "status": m["status"], "convener": m["convener"]}
                            for org, m in sorted(st.scan("org_members"))],
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
        if any(lf.get("metric") != "orgeco_capacity" or (lf.get("args") or {}).get("org") not in (org, COMMONS)
               for mm in c["milestones"] for lf in predicate_leaves(mm["deliverable"]["predicate"])):
            return respond("reject", "we pay only for the capacity of the backbone we use")
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
        if recs:
            # what this session's customers asked and got (the evaluator's record, kept as world state)
            ctx.emit("orgs.usage", {"t": int(now), "requests": int(sum(r[2] for r in recs)),
                                    "ok": int(sum(r[3] for r in recs)),
                                    "users": sorted({int(r[0]) for r in recs if r[2] > 0})}, vis="internal")
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

    @applier("orgs.act")
    def ap_act(self, ev):
        return None

    @applier("orgs.usage")
    def ap_usage(self, ev):
        st = self.world.state
        st.put("org_use", f"{int(ev.payload['t']):012d}|{len(st.tables['org_use']):06d}", dict(ev.payload))

    @applier("orgs.vote_reason")
    def ap_vote_reason(self, ev):
        return None

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
@K.cached_evaluator
def evaluate(world, principal: str, objective: dict) -> dict:
    st = world.state
    now = world.now
    t0 = now - int(objective.get("window_days", 14)) * DAY
    # the customers' sessions in the window, from world state (equal on a world rebuilt from its log)
    use = [u for _, u in st.scan("org_use") if u["t"] >= t0]
    req = sum(u["requests"] for u in use)
    okn = sum(u["ok"] for u in use)
    success = okn / req if req else 0.0
    users = len({i for u in use for i in u["users"]})
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
    return mechanics_report(world, 0)


def mechanics_report(world, since_seq: int = 0) -> dict:
    """Each required mechanic of 9.12 and the recorded events (after `since_seq`) that exercise it."""
    st = world.state
    c = K.event_counts(world, since_seq)
    goals = {m["goal"] for _, m in st.scan("org_members") if not m["convener"]}
    # provider organisations acting in the run: pricing and closing (member updates), marketing, commitment answers
    acting: set[str] = set()
    council = {"proposed": 0, "ballots": 0, "decided": 0}
    ballots: dict[str, set] = {}
    executed: list[str] = []
    for ev in world.events_range(since_seq):
        p = ev.payload
        if ev.type == "orgs.member_update" and set(p["changes"]) & {"price", "status"}:
            acting.add(p["id"])
        elif ev.type == "cols.batch" and p.get("kind") == "orgs.marketing":
            acting.add((p.get("extra") or {}).get("org"))
        elif ev.type == "orgs.negotiation":
            acting.add(p["org"])
        elif ev.type == "inst.proposed" and p["org"] == COMMONS:
            council["proposed"] += 1
        elif ev.type == "inst.ballot" and st.tables["inst_decisions"].get(p["decision"], {}).get("org") == COMMONS:
            council["ballots"] += 1
            ballots.setdefault(p["decision"], set()).add(p["choice"])
        elif ev.type == "inst.decided" and st.tables["inst_decisions"].get(p["id"], {}).get("org") == COMMONS:
            council["decided"] += 1
            if p["status"] == "executed":
                executed.append(p["id"])
    acting.discard(None)
    split = sum(1 for v in ballots.values() if v == {"yes", "no"})
    congested = sum(1 for ev in world.events_range(since_seq) if ev.type == "orgs.hour" and
                    ev.payload["congestion_success"] < 1.0)
    countered = sum(1 for ev in world.events_range(since_seq) if ev.type == "commit.offer_status" and
                    ev.payload["status"] == "countered")
    settled = c["commit.settled"]
    acts = [ev.payload for ev in world.events_range(since_seq) if ev.type == "orgs.act"]
    # an executed quota or levy binds every member at once (each member's load or money changes by the decision)
    binding = sum(1 for a in acts if a["kind"] in ("quotas", "levy"))
    contrib = len({ev.payload["by"] for ev in world.events_range(since_seq) if ev.type == "orgs.fund"})
    ev_ = evaluate(world, "", {"horizon_days": 0})
    return {
        "Organizations": {"exercised": len(acting) >= 3, "events": {"provider_organisations_acting": len(acting),
                                                                    "orgs.member_update": c["orgs.member_update"]}},
        "Differing goals": {"exercised": len(goals) >= 3 and split > 0,
                            "events": {"goals": sorted(goals), "decisions_with_split_ballots": split}},
        "Contracts": {"exercised": c["commit.created"] > 0, "events": {"commit.created": c["commit.created"]}},
        "Negotiation": {"exercised": c["commit.offer"] > 0 and (countered > 0 or c["orgs.negotiation"] > 1),
                        "events": {"commit.offer": c["commit.offer"], "countered": countered,
                                   "orgs.negotiation": c["orgs.negotiation"]}},
        "Coordination": {"exercised": binding > 0 or (c["orgs.project_start"] > 0 and contrib >= 2),
                         "events": {"executed_binding_acts": binding, "executed_council_decisions": len(executed),
                                    "fund_contributors": contrib, "orgs.project_start": c["orgs.project_start"]}},
        "Shared resources": {"exercised": congested > 0 and c["orgs.hour"] > 0,
                             "events": {"congested_hours": congested, "orgs.hour": c["orgs.hour"]}},
        "Governance": {"exercised": council["proposed"] > 0 and council["ballots"] > 0 and council["decided"] > 0,
                       "events": {"inst.proposed": council["proposed"], "inst.ballot": council["ballots"],
                                  "inst.decided": council["decided"], "orgs.act": len(acts)}},
        "Commitments": {"exercised": settled > 0 or c["commit.breach"] > 0,
                        "events": {"commit.settled": settled, "commit.breach": c["commit.breach"]}},
        "Collective outcomes": {"exercised": ev_["details"]["requests"] > 0 and congested > 0,
                                "events": {"requests": ev_["details"]["requests"],
                                           "request_success": ev_["details"]["request_success"],
                                           "orgs.project_done": c["orgs.project_done"]}},
    }


# ====================================================================== the domain
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register  # noqa: E402

A = K.A

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


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    ov = dict(overrides or {})
    rng = K.seeded_rng(config.name, seed)
    inst = generate_instance(rng, {**STRUCTURES[config.name], **(ov.pop("instance", {}) or {})})
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("contracts", {}), ("commit", {}), ("inst", {}), ("pop", K.pop_config(n)),
                            ("orgeco", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    users = 200 + 50 * d
    text = (f"Get the independent providers that share this region's backbone to keep their customers "
                           f"served: by day 40, over the last 14 days at least 95% of the requests of the region's "
                           f"people must succeed, at least {users} people must be using a provider, the backbone must "
                           f"have more capacity than today, and every provider must still be in business.")
    return {"text": text, "description": text, "difficulty": d, "window_days": 14, "horizon_days": 40, "min_success": 0.95, "min_users": users}


class CouncilConvener(K.ReadingReference):
    """Scripted reference (evaluator side, tool-only): a convener. It joins the council (`inst.join`), proposes an
    expansion project and a levy per customer as council decisions (`inst.propose`), proposes quotas in proportion
    to each member's customers as soon as congestion shows, pays its own budget into the fund, and offers each
    member a commitment to pay towards the expansion, released when the backbone's capacity reaches the target by a
    due date (members accept, counter or refuse from their own interest); it accepts the counters it receives."""
    cycle_s = 12 * HOUR
    pledges = True

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.joined = False
        self.proposed: dict[str, int] = {}
        self.offered: set[str] = set()
        self.contributed = False
        self.accepted: set[str] = set()

    def reads(self, t):
        if not self.joined:
            return []
        return [("tel", "orgs.telemetry", {"hours": 48}), ("inbox", "commit.inbox", {"status": "open"}),
                ("dec", "inst.decisions", {"org": COMMONS, "status": "any"})]

    def _propose(self, kind, tool, args, reason):
        return A("inst.propose", None, org=COMMONS, action={"kind": "tool", "tool": tool, "args": args},
                 reason=reason)

    def decide(self, t, got):
        if not self.joined:
            return [A("inst.join", "join", org=COMMONS, note="convener for the backbone's capacity")]
        tl = got.get("tel")
        if not tl:
            return []
        out = []
        members = [m for m in tl["members"] if not m["convener"] and m["status"] == "active"]
        hours = tl["hours"]
        cong = 1.0 - (float(np.mean([min(h["success"].values() or [1.0]) for h in hours])) if hours else 1.0)
        decs = got.get("dec") if isinstance(got.get("dec"), list) else []
        kinds = {}
        for d in decs:
            k = GOVERNED.get((d.get("action") or {}).get("tool"))
            if k:
                kinds.setdefault(k, set()).add(d["status"])
        open_kinds = {k for k, v in kinds.items() if "open" in v}
        passed = {k for k, v in kinds.items() if "executed" in v}
        if "expansion" not in passed and "expansion" not in open_kinds and t >= self.proposed.get("expansion", 0):
            out.append(self._propose("expansion", "orgs.expand", {}, "more capacity for everyone"))
            self.proposed["expansion"] = t + 4 * DAY
        if self.pledges and "levy" not in passed and "levy" not in open_kinds and t >= self.proposed.get("levy", 0):
            out.append(self._propose("levy", "orgs.levy", {"cents_per_customer": 450},
                                     "each member pays per customer into the expansion fund"))
            self.proposed["levy"] = t + 4 * DAY
        if self.pledges and cong > 0.02 and "quotas" not in open_kinds and t >= self.proposed.get("quotas", 0):
            total = sum(m["customers"] for m in members) or 1
            shares = {m["member"]: round(max(0.05, m["customers"] / total), 4) for m in members}
            sc = sum(shares.values())
            out.append(self._propose("quotas", "orgs.set_quotas",
                                     {"shares": {k: round(v / sc, 4) for k, v in shares.items()}},
                                     "shares by customers served"))
            self.proposed["quotas"] = t + 6 * DAY
        if not self.contributed:
            self.contributed = True
            out.append(A("orgs.contribute", None, cents=240_000))
        target = float(tl["capacity"]) + (0.0 if any(p["status"] == "done" for p in tl["projects"]) else 1.0)
        for m in members if self.pledges else []:
            if m["member"] in self.offered or t < 2 * DAY:
                continue
            self.offered.add(m["member"])
            amt = 40_000 + 100 * int(m["customers"])
            out.append(A("commit.offer", None, to=m["member"], terms={
                "title": "contribution to the backbone expansion", "payer": "$you",
                "milestones": [{"amount_cents": amt, "due_in_s": 30 * DAY, "deliverable": {
                    "kind": "predicate", "predicate": {"metric": "orgeco_capacity", "args": {"org": m["member"]},
                                                       "op": ">=", "value": target}}}]},
                message="pay towards the expansion only if the capacity is built"))
        inbox = got.get("inbox")
        for o in inbox if isinstance(inbox, list) else []:
            if o["direction"] == "received" and o["status"] == "open" and o["offer"] not in self.accepted:
                self.accepted.add(o["offer"])
                out.append(A("commit.respond", None, offer=o["offer"], action="accept"))
        return out

    def handle(self, tag, res, action):
        if tag == "join" and res is not None:
            self.joined = True


class SoloFunder(CouncilConvener):
    """Degenerate-strategy probe: the convener pays its own budget into the fund and proposes an expansion, but
    asks nothing of the members (no levy, no quotas, no commitments)."""
    pledges = False


def structure(world) -> dict:
    """Measured on the built world: the goals present among the providers and whether a mission organisation is
    among them."""
    goals = sorted({m["goal"] for _, m in world.state.scan("org_members") if not m["convener"]})
    return {"member_goals": goals, "mission_member": "mission" in goals}


Reference = CouncilConvener

DOMAIN = register(Domain(
    name="organization", title="Organizational ecosystem", spec_part="9.12", required_mechanics=MECHANICS,
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=CouncilConvener,
    mechanics=tuple(Mechanic(m, (), f"measured by organization.mechanics_report: {m}",
                             count=K.measured(mechanics_report, m)) for m in MECHANICS),
    seats=(Seat("convener", DEFAULTS["convener_cash"], "Convener"),),
    actors=(Actor("customers", ("cols.batch:orgs.online",), "population individuals using a provider"),
            Actor("providers", ("cols.batch:orgs.marketing", "orgs.member_update"),
                  "provider organisations pricing and marketing from their goals")),
    degenerate=(Probe("solo_funder", "pays its own budget and asks nothing of the members", SoloFunder),),
    structure=structure, mode="B", horizon_days=40.0, slowest_effect_days=8.0, dt=6 * HOUR,
    reference_seeds=(0, 1), sources=("organization.py", "_dom3.py"),
    doc=__doc__.split("\n\n")[0]))
