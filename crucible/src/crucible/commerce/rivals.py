"""Adaptive incumbents (module `rivals`): world-generated competitors that respond to the market by explicit rules.

The incumbents themselves are created by `ventures` (config `incumbents`): companies running a reference-tier
solver of a public format as a SaaS product with subscription plans. This module makes them *respond*. Once a day
each incumbent's management looks only at what any operator sees - its own books (live subscriptions, open support
tickets, its service's overloads in `cloud.metrics`) and the public market (competitors' plans and verified ratings in
`prod.catalog`) - and acts through the ordinary tools as the company's owner:

* **price** - when it lost at least `lose_share` of its subscribers over `window_days` while a competitor sells the
  same format cheaper, it opens a cheaper monthly plan (x `cut`, never below `floor` x its starting price) and
  archives the old one (existing subscribers keep their price); when it grew by `grow_share` with a good rating, it
  raises the price of new subscriptions by `raise_`;
* **quality** - when its verified rating is below `min_rating` (with at least `min_ratings` ratings) or it lost
  subscribers while a competitor is rated higher, it starts an engineering project: after `dev_days` its engineers
  ship the next reference tier of the solver (a new commit, redeployed);
* **capacity** - overloads in the last day add a core (`cloud.resize`), up to `max_cores`;
* **support** - it answers open tickets with probability `reply_p` a day and resolves them; it grants refund
  requests that name an invoice with probability `refund_p`.

These rules are explicit and labelled scripted (learned competitors are wave 3). The state they keep - when the
last change happened and the subscriber history - is their own memory, recorded by `rivals.observed`.
"""
from __future__ import annotations

import re

from ..core.errors import CrucibleError, contain
from ..core.kernel import Module, applier, process, register_module

DAY = 86400
DEFAULTS = {"window_days": 14, "lose_share": 0.15, "cut": 0.9, "floor": 0.55, "grow_share": 0.3, "raise_": 1.05,
            "min_rating": 3.2, "min_ratings": 5, "dev_days": 21, "max_cores": 12, "reply_p": 0.7, "refund_p": 0.6,
            "first_day": 3}
REFUND_RE = re.compile(r"(sinv_[0-9]+)")


@register_module
class Rivals(Module):
    name = "rivals"
    depends = ("identity", "bank", "products", "subs", "cloud", "code", "ventures")

    def cfg(self, k):
        return self.config.get(k, DEFAULTS[k])

    def install(self, world):
        world.state.declare_table("rival_memory")

    def genesis(self, ctx):
        ctx.schedule((ctx.now // DAY + int(self.cfg("first_day"))) * DAY + 7 * 3600, "rivals.daily", {}, prio=48,
                     period=DAY, key="rivals.daily")

    def incumbents(self, st) -> list[dict]:
        """Incumbent companies created by `ventures` (trusted-runtime SaaS firms with a product)."""
        out = []
        for oid, o in sorted(st.scan("orgs")):
            if not (o.get("info") or {}).get("trusted_runtime") or o["status"] != "active":
                continue
            prods = [pid for pid in st.find_ids("products", "owner", oid)
                     if st.tables["products"][pid]["status"] == "launched"]
            if not prods:
                continue
            owner = next((m["person"] for m in st.find("members", "org", oid) if m["role"] == "owner"), None)
            if owner is None:
                continue
            out.append({"org": oid, "owner": owner, "product": prods[0]})
        return out

    @process("daily")
    def pr_daily(self, ctx):
        for inc in self.incumbents(ctx.state):
            err = contain(ctx, self._manage, ctx, inc)
            if err is not None:
                ctx.emit("rivals.failed", {"org": inc["org"], "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def _do(self, ctx, inc, tool, **args):
        return ctx.call(tool, args, as_principal=inc["owner"])

    def _manage(self, ctx, inc):
        st = ctx.state
        org, pid = inc["org"], inc["product"]
        rng = ctx.rng("rivals", org)
        mem = dict(st.get("rival_memory", org) or {"history": [], "last_change": -10**9, "dev_until": None,
                                                  "start_price": None, "tier": None})
        mine = self._do(ctx, inc, "prod.mine")
        me = next((m for m in mine if m["product"] == pid), None)
        if me is None:
            return
        subs_now = len([s for s in self._do(ctx, inc, "subs.mine", role="merchant", limit=500)
                        if s["status"] in ("trialing", "active", "past_due")])
        monthly = [pl for pl in me["plans"] if pl["interval"] == "month"]
        price = min((pl["price_cents"] for pl in monthly), default=None)
        if mem["start_price"] is None and price is not None:
            mem["start_price"] = price
        need = me.get("need")
        rivals = [c for c in self._do(ctx, inc, "prod.catalog", need=need, limit=50) if c["product"] != pid] \
            if need else []
        rival_prices = [pl["price_cents"] for c in rivals for pl in c["plans"] if pl["interval"] == "month"]
        rival_ratings = [c["rating"] for c in rivals if c.get("rating") is not None and c.get("ratings", 0) >= 3]
        hist = list(mem["history"])[-60:]
        hist.append([ctx.now // DAY, subs_now])
        mem["history"] = hist
        past = [n for d, n in hist if d <= ctx.now // DAY - int(self.cfg("window_days"))]
        before = past[-1] if past else None
        actions = []
        cooled = ctx.now - int(mem["last_change"]) >= int(self.cfg("window_days")) * DAY
        lost = before is not None and before >= 5 and subs_now <= before * (1 - float(self.cfg("lose_share")))
        grew = before is not None and before >= 5 and subs_now >= before * (1 + float(self.cfg("grow_share")))
        rating = me.get("rating")
        # price
        if price is not None and cooled:
            new_price = None
            if lost and rival_prices and min(rival_prices) < price:
                new_price = max(int(price * float(self.cfg("cut"))), int(mem["start_price"] * float(self.cfg("floor"))))
            elif grew and (rating or 0) >= 4.0:
                new_price = int(price * float(self.cfg("raise_")))
            if new_price is not None and new_price != price:
                old = min(monthly, key=lambda x: x["price_cents"])
                self._do(ctx, inc, "subs.create_plan", product=pid, price_cents=int(new_price), interval="month",
                         trial_days=int(old["trial_days"]))
                self._do(ctx, inc, "subs.archive_plan", plan=old["plan"])
                mem["last_change"] = ctx.now
                actions.append(["price", price, int(new_price)])
        # quality: ship the next tier after an engineering project
        bad = rating is not None and me.get("ratings", 0) >= int(self.cfg("min_ratings")) and \
            rating < float(self.cfg("min_rating"))
        outrated = lost and rival_ratings and rating is not None and max(rival_ratings) > rating + 0.5
        if mem["dev_until"] is None and (bad or outrated) and need:
            mem["dev_until"] = ctx.now + int(self.cfg("dev_days")) * DAY
            actions.append(["engineering_started", int(mem["dev_until"])])
        elif mem["dev_until"] is not None and ctx.now >= int(mem["dev_until"]):
            shipped = self._ship_next_tier(ctx, inc, need, mem)
            mem["dev_until"] = None
            if shipped is not None:
                actions.append(["shipped_tier", shipped])
        # capacity
        host = ((st.tables["products"][pid]["delivery"]) or {}).get("host")
        if host:
            m = self._do(ctx, inc, "cloud.metrics", host=host, hours=24)
            over = sum(int(h.get("overload", 0)) for h in (m.get("hours") or []))
            if over:
                inst = next((x for x in self._do(ctx, inc, "cloud.instances") if x.get("owner") == org
                             and x.get("status") == "running"), None)
                if inst is not None and int(inst["cores"]) < int(self.cfg("max_cores")):
                    self._do(ctx, inc, "cloud.resize", instance=inst["instance"], cores=int(inst["cores"]) + 1)
                    actions.append(["cores", int(inst["cores"]) + 1])
        # support and refunds
        for t in self._do(ctx, inc, "support.list", owner=org, status="unresolved", limit=50):
            if rng.random() >= float(self.cfg("reply_p")):
                continue
            m = REFUND_RE.search(t.get("subject") or "")
            if m and rng.random() < float(self.cfg("refund_p")):
                try:
                    self._do(ctx, inc, "subs.refund", invoice=m.group(1), reason="refund on request")
                    actions.append(["refund", m.group(1)])
                except CrucibleError:
                    pass
            self._do(ctx, inc, "support.reply", ticket=t["ticket"], body="Thank you for the report; we are on it.")
            self._do(ctx, inc, "support.resolve", ticket=t["ticket"], note="answered")
        ctx.emit("rivals.observed", {"org": org, "memory": mem, "subscribers": subs_now, "actions": actions},
                 vis=(org,))

    def _ship_next_tier(self, ctx, inc, need, mem) -> int | None:
        """The incumbent's engineers ship the next reference tier of the solver (world-generated firms run
        kernel-authored solvers; agents never see them unless they read the firm's code)."""
        from ..economy.needs import NEEDS
        from ..worlds.digital_civilization import service_repo_files
        st = ctx.state
        org = inc["org"]
        repos = [r for r in st.find_ids("repos", "owner", org)]
        if not repos:
            return None
        rid = repos[0]
        cur = mem.get("tier")
        if cur is None:
            src = self.world.modules["code"].files(st, rid).get("service.py", "")
            cur = next((k for k, (_, s) in enumerate(NEEDS[need].tiers) if s == src), 0)
        nxt = int(cur) + 1
        if nxt >= len(NEEDS[need].tiers):
            return None
        files = service_repo_files(need, nxt, 500)
        self.world.modules["code"].commit_files(ctx, rid, files, f"tier {nxt}", inc["owner"], trusted=True)
        host = ((st.tables["products"][inc["product"]]["delivery"]) or {}).get("host")
        dep = self.world.modules["cloud"].live_deployment(st, host)
        if dep is None:
            return None
        self._do(ctx, inc, "cloud.deploy", repo=rid, host=host, instance=dep["instance"])
        mem["tier"] = nxt
        return nxt

    @applier("rivals.observed")
    def ap_observed(self, ev):
        self.world.state.put("rival_memory", ev.payload["org"], dict(ev.payload["memory"]))

    @applier("rivals.failed")
    def ap_failed(self, ev):
        pass
