"""Tier-2 actor cognition: persistent, heterogeneous minds for simulated organisations and people (module `society`).

Each mind is a row of authoritative state, updated only through events:

    traits    risk tolerance, openness to novelty, propensity to share evidence, sociability, honesty,
              attention (how many decisions/messages per day it can process), patience
    goals     acquired, re-prioritised and dropped from the actor's own situation (cash runway, growth, the
              largest measured loss, ...), never handed down by the simulator
    beliefs   about capabilities, counterparties and products: (estimated relative effect, weight, source)
    trust     Beta(a, b) per counterparty, updated by experience (kept and broken commitments, trial outcomes)
    memory    a bounded episodic log of interactions that later decisions consult
    contacts  people/organisations it has dealt with (the channels knowledge travels through)

Minds act exclusively through the same tools as learning agents (`ctx.call(..., as_principal=...)`), with the
same permissions, prices and capacity limits. They never read hidden state: an organisation's mind sees its own
journal (`ops`), its own analytics, its mail, public market/registry data and public evidence.

The **adoption engine** (generic; it knows nothing about what any capability does):

1. measure: the organisation's own operating journal and loss journal over the last week;
2. publish pain: when a loss mechanism costs more than a threshold share of revenue, post a public request
   describing it with journal evidence, the resources it would delegate and a budget;
3. discover: responses to its requests, capability search on its own words, recommendations from contacts;
4. screen: expected net value = evidence-weighted relative effect x own revenue - price - delegation risk
   (sensitivity of the requested capabilities x distrust of the provider x risk aversion) + exploration value;
5. trial: install as a randomised switchback trial; the world computes the outcome from the journal;
6. decide: adopt if effect - cost exceeds z x standard error (z from risk tolerance), else uninstall;
7. propagate: share (or withhold) the verified evidence with contacts by mail; update trust in the provider;
8. monitor: drop capabilities whose service stops answering (measured dependency failures).

Default routines (capacity, pricing, staffing, ...) stand down on resources delegated to an installed capability.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Ctx, Module, applier, process, register_module

DAY = 86400
SENSITIVITY = {"analytics.read": 0.05, "cloud.manage": 0.4, "market.manage": 0.5, "make.read": 0.05,
               "make.operate": 0.5, "make.maintain": 0.3, "make.trade": 0.8, "freight.ship": 0.5,
               "support.answer": 0.4, "mail.read": 0.6, "mail.send": 0.6, "code.read": 0.3, "bank.read": 0.4,
               "bank.spend": 1.5, "org.admin": 4.0}
LOSS_TEXT = {
    "capacity_overload": "Our service {res} turned away {n} requests in the last {d} days because its instance was "
                         "overloaded (about {v} in lost sales). Demand varies a lot over the day.",
    "service_unavailable": "Our service {res} was unavailable for {n} requests in the last {d} days "
                           "(about {v} in lost sales).",
    "dependency_failure": "Software we depend on failed {n} times in the last {d} days.",
    "machine_downtime": "Our machines were broken down for {n} hours in the last {d} days (about {v} of lost output).",
    "stockout": "Customers wanted {n} units we did not have in stock in the last {d} days (about {v} in lost sales).",
    "supply_shortfall": "Suppliers failed to deliver {n} units we had ordered in the last {d} days.",
    "late_delivery": "{n} shipments arrived late in the last {d} days.",
    "defects": "{n} units failed inspection in the last {d} days (about {v} of wasted output).",
    "energy_curtailment": "Production runs were refused {n} times in the last {d} days because the grid was full.",
    "capacity_unavailable": "Our compute provider had no free cores {n} times in the last {d} days when our service "
                            "needed more; we could not scale at the busiest hours.",
}
COST_TEXT = "We spend about {v} per day on {k} ({p:.0f}% of our revenue) and would like to spend less."
SEARCH_WORDS = {"cost:compute": "compute cost instance cores idle capacity cloud scaling",
                "cost:energy": "energy efficiency power cost process",
                "cost:freight": "shipping logistics freight cost routing",
                "cost:contract": "automation labour staff cost workflow",
                "cost:subscription": "subscription cost cheaper alternative",
                "cost:materials": "materials procurement supplier cost cheaper","capacity_overload": "overload capacity scaling instance demand peak",
                "service_unavailable": "availability uptime outage failover",
                "dependency_failure": "reliability fallback dependency failover monitoring",
                "machine_downtime": "machine maintenance failure breakdown downtime predictive",
                "stockout": "inventory stock reorder demand forecast",
                "supply_shortfall": "supplier procurement sourcing reliability",
                "late_delivery": "logistics shipping routing delivery",
                "defects": "quality inspection defects process",
                "energy_curtailment": "energy efficiency power grid process",
                "capacity_unavailable": "provider capacity full cores unavailable placement migrate scaling peak"}


def _beta_mean(t):
    return t[0] / (t[0] + t[1])


@register_module
class Society(Module):
    name = "society"
    depends = ("identity", "bank", "ops", "caps", "wf")

    def install(self, world):
        st = world.state
        st.declare_table("minds", index=("kind", "org"))
        world.hooks["society"] = self
        register_default_routines(world)
        from .dynamics import register_dynamics
        register_dynamics(world)
        from .innovators import register_innovators
        register_innovators(world)
        world.hooks.setdefault("layoff", _layoff)
        from . import negotiation as neg
        prev_o = world.hooks.get("on_deal_offered")

        def offered(ctx, did, recipient):
            if prev_o:
                prev_o(ctx, did, recipient)
            neg.on_deal_offered(self, ctx, did, recipient)
        world.hooks["on_deal_offered"] = offered
        prev_b = world.hooks.get("on_breach")

        def breach(ctx, cid, who):
            if prev_b:
                prev_b(ctx, cid, who)
            neg.on_breach(self, ctx, cid, who)
        world.hooks["on_breach"] = breach
        prev_g = world.hooks.get("on_gov_proposal")

        def gprop(ctx, pid, voter):
            if prev_g:
                prev_g(ctx, pid, voter)
            neg.on_gov_proposal(self, ctx, pid, voter)
        world.hooks["on_gov_proposal"] = gprop

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY - 1800, "society.firms", {}, prio=78, period=DAY, key="society.firms")
        ctx.schedule(ctx.now + DAY - 3600, "society.people", {}, prio=77, period=DAY, key="society.people")

    @process("people")
    def pr_people(self, ctx):
        """Daily cycle of simulated people: their own routines (looking for opportunities, reading)."""
        st = ctx.state
        rng = ctx.rng("society-people")
        routines = ctx.world.hooks.get("mind_routines", {})
        for mid in sorted(st.find_ids("minds", "kind", "person")):
            m = dict(st.tables["minds"][mid])
            for name in sorted(m["routines"]):
                fn = routines.get(name)
                if fn is not None:
                    fn(self, ctx, mid, m, rng)
                    m = dict(st.tables["minds"][mid])

    def p(self, key, default):
        return self.config.get(key, self.world.spec.params.get(key, default))

    # ------------------------------------------------------------------ creation
    @staticmethod
    def draw_traits(rng) -> dict:
        return {"risk": round(float(rng.beta(2, 2)), 3), "openness": round(float(rng.beta(2, 2)), 3),
                "share": round(float(rng.beta(2, 2.5)), 3), "sociability": round(float(rng.beta(2, 2)), 3),
                "honesty": round(float(rng.beta(5, 1.5)), 3), "attention": int(rng.integers(2, 6)),
                "patience": round(float(rng.uniform(0.85, 0.99)), 3)}

    def add_mind(self, ctx: Ctx, mid: str, kind: str, org: str | None, owner: str, traits: dict, role: str = "",
                 extra: dict | None = None) -> None:
        rec = {"kind": kind, "org": org, "owner": owner, "role": role, "traits": traits, "goals": [], "beliefs": {},
               "trust": {}, "memory": [], "contacts": [], "tried": [], "posted": {}, "routines": {},
               "stats": {"trials": 0, "adopted": 0, "uninstalled": 0, "requests": 0, "shared": 0}}
        rec.update(extra or {})
        ctx.emit("society.add", {"id": mid, "rec": rec}, vis="internal")

    def set(self, ctx: Ctx, mid: str, **fields) -> None:
        ctx.emit("society.mind", {"id": mid, "set": fields}, vis="internal")

    def remember(self, ctx, mid, m, kind, with_, data, limit=40):
        mem = (list(m["memory"]) + [{"t": ctx.now, "kind": kind, "with": with_, "data": data}])[-limit:]
        self.set(ctx, mid, memory=mem)
        return mem

    # ------------------------------------------------------------------ acting
    def act(self, ctx: Ctx, who: str, tool: str, args: dict):
        try:
            return ctx.call(tool, args, as_principal=who)
        except CrucibleError as e:
            ctx.emit("society.failed", {"who": who, "tool": tool, "error": e.as_dict()}, vis="internal")
            return None

    def trust(self, m, who) -> float:
        t = m["trust"].get(who)
        return _beta_mean(t) if t else 0.5

    def update_trust(self, ctx, mid, m, who, good: float, bad: float):
        tr = dict(m["trust"])
        a, b = tr.get(who, [1.0, 1.0])
        tr[who] = [round(a + good, 3), round(b + bad, 3)]
        self.set(ctx, mid, trust=tr)
        m = dict(m, trust=tr)
        return m

    def add_contact(self, ctx, mid, m, who):
        if who and who not in m["contacts"] and who != m.get("org") and who != m.get("owner"):
            c = (list(m["contacts"]) + [who])[-30:]
            self.set(ctx, mid, contacts=c)
            return dict(m, contacts=c)
        return m

    # ------------------------------------------------------------------ resources
    def resources(self, st, org: str) -> dict[str, list[str]]:
        out = {"instance": [i for i in st.find_ids("instances", "owner", org)
                            if st.tables["instances"][i]["status"] == "running"],
               "host": [h for h in st.find_ids("hosts", "owner", org)],
               "listing": [l for l in st.find_ids("listings", "seller", org)
                           if st.tables["listings"][l]["status"] == "active"],
               "account": [a for a in st.find_ids("accounts", "owner", org)
                           if st.tables["accounts"][a]["status"] == "open"]}
        extra = self.world.hooks.get("resource_lister")
        if extra:
            for k, v in extra(st, org).items():
                out[k] = v
        return out

    def bind(self, st, org, params: dict, prefer: str | None = None) -> dict | None:
        """Bind a capability's parameters to own resources (related resources preferred: the instance that hosts a
        bound host, a machine at a bound facility). Returns None if some parameter cannot be bound."""
        res = self.resources(st, org)
        b: dict = {}
        order = sorted(params, key=lambda k: {"host": 0, "facility": 1}.get(params[k]["kind"], 2))
        for k in order:
            kind = params[k]["kind"]
            cands = list(res.get(kind, []))
            if kind == "value" or not cands:
                return None
            pick = None
            if kind == "host" and prefer in cands:
                pick = prefer
            if kind == "instance":
                for hv in [v for kk, v in b.items() if params[kk]["kind"] == "host"]:
                    h = st.get("hosts", hv)
                    d = st.get("deployments", h["deployment"]) if h and h.get("deployment") else None
                    if d and d["instance"] in cands:
                        pick = d["instance"]
            if kind == "listing":
                for hv in [v for kk, v in b.items() if params[kk]["kind"] == "host"]:
                    ls = [l for l in cands if st.tables["listings"][l].get("host") == hv]
                    if ls:
                        pick = ls[0]
            rel = self.world.hooks.get("resource_relation")
            if pick is None and rel:
                pick = rel(st, kind, cands, b, params)
            b[k] = pick or cands[0]
        return b

    # ------------------------------------------------------------------ the daily firm cycle
    @process("firms")
    def pr_firms(self, ctx):
        st = ctx.state
        rng = ctx.rng("society-firms")
        for mid in sorted(st.find_ids("minds", "kind", "firm")):
            m = dict(st.tables["minds"][mid])
            o = st.get("orgs", m["org"])
            if o is None or o["status"] != "active":
                continue
            self.firm_day(ctx, mid, m, rng)

    def firm_day(self, ctx, mid, m, rng):
        st = ctx.state
        org, owner = m["org"], m["owner"]
        ops = ctx.world.modules["ops"]
        today = ctx.now // DAY
        W = ops.window(st, org, 7, today + 1)        # the last 7 days including today (the process runs at 23:30)
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(st, org)
        cash = bank.balance(st, acct) if acct else 0
        rev_day = W["revenue"] / 7.0
        m = self.update_goals(ctx, mid, m, W, cash, rev_day)
        # 1. default routines on resources that are not delegated
        for name, routine in sorted(ctx.world.hooks.get("mind_routines", {}).items()):
            m = dict(st.tables["minds"][mid])          # each routine sees the effects of the previous ones
            if name in m["routines"]:
                routine(self, ctx, mid, m, rng)
        m = dict(st.tables["minds"][mid])
        if self.p("ablate_adoption", False):
            return
        attention = int(m["traits"]["attention"])
        # 2. knowledge arriving by mail (evidence references), bounded by attention
        m = self.read_mail(ctx, mid, m, attention)
        # 3. trials that finished: decide, share
        m = self.conclude_trials(ctx, mid, m, rng)
        # 4. monitor what is installed
        m = self.monitor(ctx, mid, m, W)
        # 5. publish the largest pain
        if not self.p("ablate_requests", False):
            m = self.post_pain(ctx, mid, m, W, rev_day, cash)
        # 6. discover and maybe start a trial (one at a time)
        running = [i for i in st.find("cap_installs", "installer", org) if i["status"] == "trial"]
        if not running and cash > 0:
            self.consider(ctx, mid, m, W, rev_day, cash, rng)

    def update_goals(self, ctx, mid, m, W, cash, rev_day):
        goals = []
        burn = -min(0.0, W["operating"] / 7.0)
        if burn > 0 and cash / burn < 45:
            goals.append({"kind": "survive", "key": "cash_runway", "priority": 1.0})
        losses = sorted(((v / 7.0, k) for k, v in W["losses"].items() if v > 0), reverse=True)
        for v, k in losses[:2]:
            sev = v / max(rev_day, 1.0)
            if sev > 0.02 or W["loss_n"].get(k, 0) >= 7:
                goals.append({"kind": "reduce_loss", "key": k, "priority": round(min(1.0, sev * 3), 3),
                              "stake": round(v, 1)})
        # cost structure: a spending category that takes a large share of revenue becomes something to reduce
        for k, v in sorted(W["flows"].items()):
            if v < 0 and k not in ("transfer", "endowment", "escrow", "sale", "refund") and rev_day > 0 \
                    and -v / 7.0 > 0.10 * rev_day:
                goals.append({"kind": "reduce_cost", "key": f"cost:{k}", "priority": round(min(1.0, -v / 7.0 / rev_day), 3),
                              "stake": round(-v / 7.0, 1)})
        if W["operating"] > 0:
            goals.append({"kind": "grow", "key": "operating_result", "priority": 0.3})
        old = {(g["kind"], g["key"]): g for g in m["goals"]}
        new = [dict(g, since=old.get((g["kind"], g["key"]), {}).get("since", ctx.now)) for g in goals]
        if new != list(m["goals"]):
            self.set(ctx, mid, goals=new)
            m = dict(m, goals=new)
        return m

    # ------------------------------------------------------------------ pain -> public request
    def post_pain(self, ctx, mid, m, W, rev_day, cash):
        st = ctx.state
        org = m["org"]
        for g in sorted(m["goals"], key=lambda g: -g.get("stake", 0)):
            if g["kind"] not in ("reduce_loss", "reduce_cost"):
                continue
            k = g["key"]
            last = m["posted"].get(k)
            if last is not None and ctx.now - last < 21 * DAY:
                continue
            open_reqs = [r for r in st.find("ops_requests", "by", org) if r["status"] == "open"]
            if len(open_reqs) >= 3:
                return m
            res_kinds = self.resources(st, org)
            v = W["losses"].get(k, 0)
            if g["kind"] == "reduce_cost":
                text = COST_TEXT.format(v=f"${g['stake'] / 100:,.0f}", k=k.split(":", 1)[1],
                                        p=100.0 * g["stake"] / max(rev_day, 1.0))
                v = g["stake"] * 7
            else:
                text = LOSS_TEXT.get(k, "We keep losing money to '{k}' ({n} events, about {v}, last {d} days).").format(
                    res=", ".join(res_kinds.get("host", [])[:2] + res_kinds.get("facility", [])[:2]) or org,
                    n=W["loss_n"].get(k, 0), d=7, v=f"${v / 100:,.0f}", k=k)
            offered = sorted({"analytics.read", "make.read"} | ({"cloud.manage"} if res_kinds.get("instance") else set())
                             | ({"make.maintain", "make.operate"} if res_kinds.get("machine") else set()))
            budget = int(max(0.0, v / 7.0 * 30 * 0.3))
            r = self.act(ctx, m["owner"], "ops.request", {
                "org": org, "title": f"{org.split(':', 1)[1]} needs help: {k.replace('_', ' ')}",
                "body": text + " We are open to software that can work on our resources with delegated permissions; "
                               "we evaluate everything with a trial against our own books.",
                "tags": [k.replace("cost:", "cost-")] + sorted(x for x in res_kinds if res_kinds[x])[:4],
                "evidence_days": 7,
                "resources": (res_kinds.get("host", []) + res_kinds.get("instance", []) + res_kinds.get("facility", [])
                              + res_kinds.get("machine", []))[:12],
                "offered_caps": offered, "budget_cents_month": budget, "expires_in_days": 30})
            if r:
                posted = dict(m["posted"], **{k: ctx.now})
                stats = dict(m["stats"], requests=m["stats"]["requests"] + 1)
                self.set(ctx, mid, posted=posted, stats=stats)
                m = dict(m, posted=posted, stats=stats)
            return m
        return m

    # ------------------------------------------------------------------ discovery & screening
    def candidates(self, ctx, mid, m) -> dict[str, float]:
        """Candidate capabilities -> the stake (cents/day) they plausibly address, from: answers to our own
        requests (stake recorded when we asked), contacts' recommendations, and search on our goals' words."""
        from ..services.capabilities import tokens
        st = ctx.state
        org = m["org"]
        out: dict[str, float] = {}
        goals = [g for g in m["goals"] if g["kind"] in ("reduce_loss", "reduce_cost")]
        for rid in st.find_ids("ops_requests", "by", org):
            r = st.tables["ops_requests"][rid]
            stake = r.get("budget_cents_month", 0) / 30.0 / 0.3
            for resp in st.find("ops_responses", "request", rid):
                if resp.get("capability") and r["status"] == "open":
                    out[resp["capability"]] = max(out.get(resp["capability"], 0.0), stake)
        for cid, b in sorted(m["beliefs"].items()):
            if b.get("src") == "contact" and cid not in out:
                out[cid] = 0.0
        words = {g["key"]: SEARCH_WORDS.get(g["key"], g["key"].replace("_", " ").replace(":", " ")) for g in goals}
        if words:
            found = self.act(ctx, m["owner"], "caps.search", {"q": " ".join(sorted(words.values())), "limit": 6}) or []
            for f in found:
                out.setdefault(f["capability"], 0.0)
        for cid in list(out):       # relevance of a candidate to each goal: vocabulary overlap with its descriptor
            c = st.get("capabilities", cid)
            if c is None:
                continue
            tk = tokens(" ".join([c["title"], c["description"], " ".join(c["tags"])]))
            for g in goals:
                if tokens(words[g["key"]]) & tk:
                    out[cid] = max(out[cid], g.get("stake", 0.0))
        tried = set(m["tried"])
        keep = [c for c in sorted(out, key=lambda c: (-out[c], c)) if c not in tried and st.has("capabilities", c)
                and st.tables["capabilities"][c]["status"] == "active"
                and st.tables["capabilities"][c]["provider"] != org][: int(m["traits"]["attention"]) + 2]
        return {c: out[c] for c in keep}

    def prior(self, ctx, m, cid) -> tuple[float, float]:
        """(expected relative effect, evidence weight) from public verified outcomes and contacts' reports."""
        st = ctx.state
        num, den = 0.0, 0.0
        c = st.tables["capabilities"][cid]
        for eid in st.find_ids("cap_evidence", "capability", cid):
            e = st.tables["cap_evidence"][eid]
            if e["kind"] != "outcome" or not e["public"] or not e["verified"] or e["by"] == m["org"]:
                continue
            rel = e["data"].get("rel_effect")
            if rel is None:
                continue
            w = (1.0 if e["independent"] else 0.1) * math.exp(-(ctx.now - e["t"]) / (60 * DAY))
            num += w * rel
            den += w
        b = m["beliefs"].get(cid)
        if b is not None and b.get("src") == "contact":
            w = b["w"] * self.trust(m, b.get("from"))
            num += w * b["rel"]
            den += w
        for eid in st.find_ids("cap_evidence", "capability", cid):
            e = st.tables["cap_evidence"][eid]
            if e["kind"] == "test" and e["public"] and e["independent"] and e["data"].get("failures", 0) > \
                    0.3 * e["data"].get("cases", 1):
                num -= 0.05
                den += 0.5
        del c
        return (num / den if den else 0.0), den

    def daily_cost(self, st, cid) -> float:
        c = st.tables["capabilities"][cid]
        cost = c["monthly_cents"] / 30.0
        for t in c["triggers"]:
            price = c["endpoints"].get(t["endpoint"], {}).get("price_cents", 0)
            per_day = (DAY / t["every_s"]) if t.get("every_s") else 24.0
            cost += price * per_day
        return cost

    def risk(self, m, c) -> float:
        r = 0.0
        for q in c["requires"]:
            s = SENSITIVITY.get(q["cap"], 1.0) * (3.0 if q["scope"] == "*" else 1.0)
            r += s
        return r * (1.0 - self.trust(m, c["provider"])) * (1.0 - m["traits"]["risk"])

    def consider(self, ctx, mid, m, W, rev_day, cash, rng):
        st = ctx.state
        org = m["org"]
        best = None
        p0 = 0.15 + 0.35 * m["traits"]["openness"]   # belief that an untested, plausible solution works
        for cid, stake in self.candidates(ctx, mid, m).items():
            c = st.tables["capabilities"][cid]
            b = self.bind(st, org, c["params"], None)
            if b is None:
                continue
            rel, w = self.prior(ctx, m, cid)
            if w == 0 and m["traits"]["openness"] < self.p("social_proof_openness", 0.25):
                continue                 # wait for evidence from others before trying something new
            cost = self.daily_cost(st, cid)
            risk = self.risk(m, c)
            gain = (w * rel * rev_day + p0 * stake) / (w + 1.0) if w > 0 else p0 * stake
            ev = gain - cost - risk * 0.01 * max(rev_day, 1.0)
            if cost * 30 > 0.5 * max(cash, 0):
                continue
            if best is None or ev > best[0]:
                best = (ev, cid, b, rel, w, cost, risk)
        if best is None:
            return
        ev, cid, b, rel, w, cost, risk = best
        if ev <= 0 or rng.random() > 0.6 + 0.4 * m["traits"]["openness"]:
            return
        days = 10 + 2 * int(3 * (1.0 - m["traits"]["risk"]))     # cautious firms run longer trials
        c = st.tables["capabilities"][cid]
        # capabilities that change the state of physical resources (maintenance, production, trade) have effects
        # that outlast a day: use blocks with washout, extended while inconclusive
        stateful = any(q["cap"] in ("make.maintain", "make.operate", "make.trade", "cloud.deploy") for q in c["requires"])
        design = {"design": "sequential", "block_days": 4, "washout_days": 2, "max_days": 40, "trial_days": 16} \
            if stateful and not self.p("trial_design_switchback", False) else {"design": "switchback", "trial_days": days}
        r = self.act(ctx, m["owner"], "caps.install", {"capability": cid, "org": org, "bindings": b, "trial": True,
                                                        "publish_outcome": False, **design})
        tried = sorted(set(m["tried"]) | {cid})
        stats = dict(m["stats"], trials=m["stats"]["trials"] + (1 if r else 0))
        self.set(ctx, mid, tried=tried, stats=stats)
        c = st.tables["capabilities"][cid]
        m = self.add_contact(ctx, mid, dict(m, tried=tried, stats=stats), c["provider"])
        if r:
            self.remember(ctx, mid, m, "trial_start", c["provider"],
                          {"capability": cid, "prior_rel": round(rel, 4), "weight": round(w, 3), "ev": round(ev, 1)})

    # ------------------------------------------------------------------ conclusions, sharing, monitoring
    def conclude_trials(self, ctx, mid, m, rng):
        st = ctx.state
        org = m["org"]
        for iid in sorted(st.find_ids("cap_installs", "installer", org)):
            i = st.tables["cap_installs"][iid]
            if i["status"] != "evaluated" or not i.get("evidence"):
                continue
            e = st.tables["cap_evidence"][i["evidence"]]
            d = e["data"]
            c = st.tables["capabilities"][i["capability"]]
            cost = self.daily_cost(st, i["capability"])
            z = 1.0 + 0.6 * (1.0 - m["traits"]["risk"])
            effect, se = d.get("effect_per_day", 0.0), d.get("se", 0.0)
            keep = effect - (c["monthly_cents"] / 30.0) > z * se and effect > 0
            # per-call fees are already inside the measured operating result; the subscription is not charged in
            # the free trial days, so it is subtracted explicitly
            del cost
            if keep:
                self.act(ctx, m["owner"], "caps.adopt", {"install": iid})
            else:
                self.act(ctx, m["owner"], "caps.uninstall", {"install": iid, "reason": "trial did not pay"})
            beliefs = dict(m["beliefs"], **{i["capability"]: {"rel": d.get("rel_effect") or 0.0, "w": 3.0,
                                                              "src": "own", "t": ctx.now}})
            stats = dict(m["stats"], **({"adopted": m["stats"]["adopted"] + 1} if keep else
                                        {"uninstalled": m["stats"]["uninstalled"] + 1}))
            self.set(ctx, mid, beliefs=beliefs, stats=stats)
            m = dict(m, beliefs=beliefs, stats=stats)
            m = self.update_trust(ctx, mid, m, c["provider"], 1.0 if keep else 0.0, 0.0 if keep else 0.5)
            self.remember(ctx, mid, m, "trial_end", c["provider"], {"capability": i["capability"], "kept": keep,
                                                                     "effect": effect, "se": se})
            m = dict(st.tables["minds"][mid])
            share = rng.random() < m["traits"]["share"]
            if share and not self.p("ablate_sharing", False):
                self.act(ctx, m["owner"], "caps.publish_evidence", {"evidence": i["evidence"]})
                if "forum" in ctx.world.modules and rng.random() < 0.5 + 0.5 * m["traits"]["sociability"]:
                    verdict = "worth it" if keep else "not worth it"
                    self.act(ctx, m["owner"], "forum.post", {
                        "board": "general", "title": f"Our trial of {c['title'][:60]}: {verdict}",
                        "body": f"We ran a randomised trial of capability {i['capability']} on our own books: "
                                f"{effect:+.0f} cents/day (+/- {se:.0f}). Verified outcome: #evidence {i['evidence']}"})
            if not self.p("ablate_sharing", False):
                m = self.tell_contacts(ctx, mid, m, i["capability"], i["evidence"], keep, rng)
        return m

    def tell_contacts(self, ctx, mid, m, cid, eid, kept, rng):
        st = ctx.state
        n = 0
        for who in m["contacts"]:
            if n >= 3 or rng.random() > m["traits"]["sociability"]:
                continue
            if who == st.tables["capabilities"][cid]["provider"]:
                continue
            if self.competitor(st, m["org"], who) and rng.random() > 0.2:
                continue            # strategic withholding: rarely help competitors
            target = self.mailbox_of(st, who)
            if not target:
                continue
            verdict = "it paid for itself" if kept else "it did not pay off for us"
            self.act(ctx, m["owner"], "mail.send", {
                "to": [target], "subject": f"Our trial of {st.tables['capabilities'][cid]['title'][:60]}",
                "body": f"We trialled capability {cid} on our own operations; {verdict}. "
                        f"Verified outcome: #evidence {eid} #capability {cid}"})
            n += 1
        if n:
            stats = dict(m["stats"], shared=m["stats"]["shared"] + n)
            self.set(ctx, mid, stats=stats)
            m = dict(m, stats=stats)
        return m

    def competitor(self, st, a, b) -> bool:
        ca, cb = st.get("companies", a), st.get("companies", b)
        return bool(ca and cb and ca.get("sector") and ca.get("sector") == cb.get("sector"))

    def mailbox_of(self, st, who):
        if who.startswith("o:"):
            mm = st.get("minds", who)
            return mm["owner"] if mm else None
        return who

    def read_mail(self, ctx, mid, m, attention):
        """Read unread mail; references to verifiable evidence (#evidence <id>) update beliefs, weighted by trust in
        the sender. Unverifiable claims carry little weight."""
        st = ctx.state
        if self.p("ablate_sharing", False):
            return m
        msgs = self.act(ctx, m["owner"], "mail.inbox", {"unread_only": True, "limit": max(1, attention)}) or []
        beliefs = dict(m["beliefs"])
        changed = False
        for msg in msgs:
            full = self.act(ctx, m["owner"], "mail.read", {"msg": msg["msg"]})
            if not full:
                continue
            body = full.get("body", "")
            sender = msg["from"]
            sender_org = self.org_of(st, sender)
            m = self.add_contact(ctx, mid, m, sender_org or sender)
            for eid in re.findall(r"#evidence\s+(evid_\d+)", body):
                e = st.get("cap_evidence", eid)
                if e is None or e["kind"] != "outcome" or not e["verified"]:
                    continue
                if not (e["public"] or e["by"] == sender_org):
                    continue
                rel = e["data"].get("rel_effect")
                if rel is None:
                    continue
                cid = e["capability"]
                if cid in beliefs and beliefs[cid].get("src") == "own":
                    continue
                beliefs[cid] = {"rel": rel, "w": 1.0, "src": "contact", "from": sender_org or sender, "t": ctx.now,
                                "evidence": eid}
                changed = True
        if changed:
            self.set(ctx, mid, beliefs=beliefs)
            m = dict(m, beliefs=beliefs)
        return m

    def org_of(self, st, person):
        if person.startswith("o:"):
            return person
        for mm in st.find("members", "person", person):
            if mm["role"] in ("owner", "admin"):
                return mm["org"]
        return None

    def monitor(self, ctx, mid, m, W):
        """Watch installed capabilities. A transient outage of the provider (its triggers paused after repeated
        failures) is tolerated: resume once the service answers again, at a cost in trust. Drop it after three
        outage episodes within 30 days, or if it has been down for two days."""
        st = ctx.state
        org = m["org"]
        cloud = ctx.world.modules["cloud"]
        for iid in sorted(st.find_ids("cap_installs", "installer", org)):
            i = st.tables["cap_installs"][iid]
            if i["status"] != "active":
                continue
            c = st.tables["capabilities"][i["capability"]]
            dep = cloud.live_deployment(st, c["host"])
            up = dep is not None and cloud.availability(st, dep) is None
            paused = [t for t in i["triggers"] if (st.get("wf_triggers", t) or {}).get("status") == "paused"]
            episodes = [e for e in m["memory"] if e["kind"] == "outage" and e["data"].get("capability") == i["capability"]
                        and e["t"] > ctx.now - 30 * DAY]
            if paused and up:
                for t in paused:
                    self.act(ctx, m["owner"], "wf.resume", {"trigger": t})
                m = self.update_trust(ctx, mid, m, c["provider"], 0.0, 0.3)
                self.remember(ctx, mid, m, "outage", c["provider"], {"capability": i["capability"]})
                m = dict(st.tables["minds"][mid])
                episodes.append(True)
            down_since = (m.get("down_since") or {}).get(iid)
            if not up and down_since is None:
                self.set(ctx, mid, down_since=dict(m.get("down_since") or {}, **{iid: ctx.now}))
                m = dict(st.tables["minds"][mid])
            elif up and down_since is not None:
                ds = dict(m.get("down_since") or {})
                ds.pop(iid, None)
                self.set(ctx, mid, down_since=ds)
                m = dict(st.tables["minds"][mid])
            long_down = not up and down_since is not None and ctx.now - down_since > 2 * DAY
            if len(episodes) >= 3 or long_down:
                self.act(ctx, m["owner"], "caps.uninstall", {"install": iid, "reason": "the service stopped working"})
                m = self.update_trust(ctx, mid, m, c["provider"], 0.0, 1.0)
                self.remember(ctx, mid, m, "dropped", c["provider"], {"capability": i["capability"]})
                m = dict(st.tables["minds"][mid])
        return m

    @process("deal_reply")
    def pr_deal_reply(self, ctx, deal, mind):
        from . import negotiation as neg
        st = ctx.state
        m = st.get("minds", mind)
        if m is not None and m["kind"] == "person" and "npc_workers" in st.tables and st.has("npc_workers", mind):
            emp = st.tables["npc_workers"][mind]["employer"]
            if emp != m.get("employer"):
                self.set(ctx, mind, employer=emp)
        neg.deal_reply(self, ctx, deal, mind)

    @process("gov_vote")
    def pr_gov_vote(self, ctx, proposal, mind):
        from . import negotiation as neg
        neg.gov_vote(self, ctx, proposal, mind)

    # ------------------------------------------------------------------ appliers
    @applier("society.add")
    def ap_add(self, ev):
        self.world.state.insert("minds", ev.payload["id"], dict(ev.payload["rec"], t=ev.t))

    @applier("society.mind")
    def ap_mind(self, ev):
        self.world.state.patch("minds", ev.payload["id"], **ev.payload["set"])

    @applier("society.failed")
    def ap_failed(self, ev):
        return None

    @applier("society.layoff")
    def ap_layoff(self, ev):
        return None

    @applier("society.llm_decision")
    def ap_llm_decision(self, ev):
        return None


# ====================================================================== default routines for digital firms
def capacity_routine(soc: Society, ctx: Ctx, mid: str, m: dict, rng) -> None:
    """Daily, coarse capacity management of the firm's own services: add a core after a day with overloads (if the
    extra core costs less than the day's revenue), remove one after a day whose busiest hour used < 40% of it."""
    st = ctx.state
    caps = ctx.world.modules["caps"]
    for host in sorted(st.find_ids("hosts", "owner", m["org"])):
        h = st.tables["hosts"][host]
        dep = st.get("deployments", h["deployment"]) if h.get("deployment") else None
        if dep is not None and dep["status"] == "stopped" and not soc.p("ablate_recovery", False):
            _recover_or_exit(soc, ctx, m, host, dep)
            continue
        if dep is None or dep["status"] != "live":
            continue
        inst = st.get("instances", dep["instance"])
        if inst is None or inst["status"] != "running" or caps.is_delegated(st, m["org"], dep["instance"], "cloud.manage") \
                or caps.is_delegated(st, m["org"], host, "cloud.manage"):
            continue
        met = soc.act(ctx, m["owner"], "cloud.metrics", {"host": host, "hours": 24})
        if not met:
            continue
        over = sum(r["overload"] for r in met["hours"])
        peak = max((r["cu"] for r in met["hours"]), default=0.0)
        rev = sum(r["revenue_cents"] for r in met["hours"])
        cap = met["capacity_cu_per_hour"]
        if over > 0 and rev > inst["price"] * 24 * 1.2:
            soc.act(ctx, m["owner"], "cloud.resize", {"instance": dep["instance"], "cores": min(64, inst["cores"] + 1)})
        elif inst["cores"] > 1 and peak < 0.4 * cap * (inst["cores"] - 1) / inst["cores"] and over == 0:
            soc.act(ctx, m["owner"], "cloud.resize", {"instance": dep["instance"], "cores": inst["cores"] - 1})


def _recover_or_exit(soc, ctx, m, host, dep) -> None:
    """A firm whose service lost its instance (e.g. suspended for non-payment, then terminated) either restarts it
    on one core at the cheapest provider with room - if it can pay for a week of it - or winds the company up.
    Without this, a firm could sit for months with a dead service and an open listing (found in a 120-day run)."""
    st = ctx.state
    bank = ctx.world.modules["bank"]
    acct = bank.primary_account(st, m["org"])
    cash = bank.balance(st, acct) if acct else 0
    provs = sorted((p["price"], k) for k, p in st.scan("providers") if p["status"] == "active" and
                   p["cores"] - sum(i["cores"] for i in st.find("instances", "provider", k) if i["status"] == "running") >= 1)
    if provs and acct and cash >= provs[0][0] * 24 * 7:
        inst = soc.act(ctx, m["owner"], "cloud.rent", {"provider": provs[0][1], "cores": 1, "account": acct,
                                                      "owner": m["org"]})
        if inst:
            soc.act(ctx, m["owner"], "cloud.deploy", {"repo": dep["repo"], "host": host, "instance": inst["instance"]})
        return
    if st.get("companies", m["org"]) and st.tables["companies"][m["org"]]["status"] == "registered":
        soc.act(ctx, m["owner"], "biz.dissolve", {"org": m["org"]})


def pricing_routine(soc: Society, ctx: Ctx, mid: str, m: dict, rng) -> None:
    """Price toward the competition: undercut a cheaper, comparably rated rival; raise slowly when overloaded."""
    st = ctx.state
    caps = ctx.world.modules["caps"]
    for lid in sorted(st.find_ids("listings", "seller", m["org"])):
        l = st.tables["listings"][lid]
        if l["status"] != "active" or l["kind"] != "api" or caps.is_delegated(st, m["org"], lid, "market.manage"):
            continue
        step = m["routines"].get("pricing", {}).get("step", 0.06)
        rivals = [x for x in (soc.act(ctx, m["owner"], "market.browse", {"need": l["need"], "sort": "price",
                                                                          "limit": 10}) or [])
                  if x["seller"] != m["org"]] if l.get("need") else []
        price = l["price"]
        new = price
        if rivals:
            cheapest = min(rivals, key=lambda x: (x["price_cents"], x["listing"]))
            if cheapest["price_cents"] < price and rng.random() < 0.5:
                new = max(1, int(price * (1 - step)))
        if new != price:
            soc.act(ctx, m["owner"], "market.update", {"listing": lid, "price_cents": int(new)})


def staffing_routine(soc: Society, ctx: Ctx, mid: str, m: dict, rng) -> None:
    """Hire support staff when the ticket backlog grows; let one go after a week with almost no backlog (labour
    demand responds to how much work there actually is - e.g. after automating it)."""
    st = ctx.state
    if "sup_tickets" not in st.tables:
        return
    org = m["org"]
    open_t = len([t for t in st.find("sup_tickets", "owner", org) if t["status"] != "resolved"])
    staff = [p for p, w in st.scan("npc_workers") if w["employer"] == org and w["role"] == "support"] \
        if "npc_workers" in st.tables else []
    r = m["routines"].get("staffing", {})
    low = int(r.get("low_days", 0))
    if open_t > 15 * (len(staff) + 1):
        npcs = ctx.world.modules.get("npcs")
        if npcs is not None:
            npcs._hire(ctx, org, m["owner"], "support", 250_000)
        low = 0
    elif open_t <= 2 * len(staff) and staff:
        low += 1
        if low >= 7:
            soc.set(ctx, mid, routines=dict(m["routines"], staffing=dict(r, low_days=0)))
            layoff = ctx.world.hooks.get("layoff")
            if layoff:
                layoff(ctx, org, staff[-1], "not enough work")
            return
    else:
        low = 0
    if low != int(r.get("low_days", 0)):
        soc.set(ctx, mid, routines=dict(m["routines"], staffing=dict(r, low_days=low)))


def register_default_routines(world) -> None:
    rt = world.hooks.setdefault("mind_routines", {})
    rt.setdefault("capacity", capacity_routine)
    rt.setdefault("pricing", pricing_routine)
    rt.setdefault("staffing", staffing_routine)


def _layoff(ctx: Ctx, org: str, person: str, reason: str) -> None:
    """Terminate a person's employment with the contract's notice period (as the organisation's owner)."""
    st = ctx.state
    owner = next((mm["person"] for mm in st.find("members", "org", org) if mm["role"] == "owner"), None)
    for cid in sorted(st.find_ids("contracts", "employee", person)):
        c = st.tables["contracts"][cid]
        if c["status"] == "active" and c.get("employer") == org and owner:
            try:
                ctx.call("contracts.terminate", {"contract": cid, "as_org": org}, as_principal=owner)
                ctx.emit("society.layoff", {"org": org, "person": person, "reason": reason}, vis=(org, person))
            except CrucibleError:
                pass
