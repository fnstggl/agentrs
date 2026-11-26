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
  requests that name an invoice with probability `refund_p`;
* **staff** (wave 3b, worlds with the labour market `labor`) - the founder answers at most `founder_tickets`
  tickets a day; when the tickets left unresolved average `hire_backlog` or more over two days and its cash covers
  six months of the top salary, it posts a part-time support job (`labor.post`), offers the applicant with the best
  verified support record (then: declared writing skill, then first applicant) the middle of its salary range,
  accepts counters within the range, and closes the posting when it hired; it dismisses its latest hire when no
  ticket was left over for 21 days and it has two or more staff, or when its cash falls below two months of
  payroll (staff serving their notice, given by either side, are leaving: they are paid until the notice date but
  neither dismissed again nor counted as staying). The employees work the queue in their own shifts
  (`labor.shift`). Each wave-3b rule runs contained: a refused step rolls back that rule alone (`rivals.failed`
  names it), never the day's support and pricing;
* **display advertising** (wave 3b, worlds with `ads` and the population's `work`; config `display_ads`) - every
  `display_every_days` it runs a `display_days` display campaign for its product to the workers of its format
  (`work.display_campaign` on an `ads` campaign of kind `display`, bid `display_bid` per impression, daily budget
  `display_budget`) when its cash covers twenty days of that budget;
* **marketplaces** (wave 3b, worlds with `xchg`) - it sells a 30-day pass of its service on one open exchange: the
  venue with the most buyers in the last 30 days by the venue's public statistics (ties: the lowest take rate), at
  its cheapest monthly price x `pass_markup`, grossed up for the venue's take rate; it re-lists when its monthly
  price moved by more than 5% or the venue raised its take rate, and leaves a venue that refunded more than half of
  its orders in 30 days (and does not return to it for 30 days).

These rules are explicit and labelled scripted (learned competitors are wave 3). The state they keep - when the
last change happened and the subscriber history - is their own memory, recorded by `rivals.observed`.
"""
from __future__ import annotations

import json
import re

from ..core.errors import CrucibleError, contain
from ..core.kernel import Module, applier, process, register_module

DAY = 86400
DEFAULTS = {"window_days": 14, "lose_share": 0.15, "cut": 0.9, "floor": 0.55, "grow_share": 0.3, "raise_": 1.05,
            "min_rating": 3.2, "min_ratings": 5, "dev_days": 21, "max_cores": 12, "reply_p": 0.7, "refund_p": 0.6,
            "first_day": 3, "pass_markup": 1.25, "pass_days": 30, "founder_tickets": 3, "hire_backlog": 3,
            "max_staff": 4, "salary_min": 120_000, "salary_max": 220_000, "job_hours": 20, "display_ads": True,
            "display_every_days": 30, "display_days": 14, "display_bid": 20, "display_budget": 3_000}
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
        # support and refunds (with a labour market the founder answers at most `founder_tickets` a day)
        staffed = "labor" in self.world.modules
        answered = 0
        for t in self._do(ctx, inc, "support.list", owner=org, status="unresolved", limit=50):
            if staffed and answered >= int(self.cfg("founder_tickets")):
                break
            if rng.random() >= float(self.cfg("reply_p")):
                continue
            answered += 1
            m = REFUND_RE.search(t.get("subject") or "")
            if m and rng.random() < float(self.cfg("refund_p")):
                try:
                    self._do(ctx, inc, "subs.refund", invoice=m.group(1), reason="refund on request")
                    actions.append(["refund", m.group(1)])
                except CrucibleError:
                    pass
            self._do(ctx, inc, "support.reply", ticket=t["ticket"], body="Thank you for the report; we are on it.")
            self._do(ctx, inc, "support.resolve", ticket=t["ticket"], note="answered")
        # the wave-3b rules run one by one, each contained: a refused step of one rule (e.g. the labour market
        # refusing a dismissal) rolls back that rule only, never the day's support, pricing or other rules
        rules = []
        if staffed:
            rules.append(("staff", lambda: self._staff(ctx, inc, mem, actions)))
        if self.cfg("display_ads") and need and "ads" in self.world.modules and "work_display" in st.tables:
            rules.append(("display", lambda: self._advertise(ctx, inc, pid, need, mem, actions)))
        if "xchg" in self.world.modules and price is not None:
            rules.append(("marketplace", lambda: self._list_pass(ctx, inc, pid, int(price), actions, mem)))
        for rule, fn in rules:
            snap, n_act = json.dumps(mem, sort_keys=True), len(actions)
            err = contain(ctx, fn)
            if err is not None:
                mem.clear()
                mem.update(json.loads(snap))
                del actions[n_act:]
                ctx.emit("rivals.failed", {"org": org, "error": f"{rule}: {type(err).__name__}: {err}"[:200]},
                         vis="internal")
        ctx.emit("rivals.observed", {"org": org, "memory": mem, "subscribers": subs_now, "actions": actions},
                 vis=(org,))

    def _advertise(self, ctx, inc, pid: str, need: str, mem: dict, actions: list) -> None:
        """The display-advertising rule (module docstring)."""
        from ..economy.needs import NEEDS
        st = ctx.state
        org = inc["org"]
        last = mem.get("last_display")
        if last is not None and ctx.now - int(last) < int(self.cfg("display_every_days")) * DAY:
            return
        if self._cash(st, org) < 20 * int(self.cfg("display_budget")):
            return
        acct = self.world.modules["bank"].primary_account(st, org)
        name = st.tables["products"][pid]["name"]
        c = self._do(ctx, inc, "ads.create_campaign", account=acct, keywords=list(NEEDS[need].keywords[:5]) or [need],
                     bid_cents=int(self.cfg("display_bid")), daily_budget_cents=int(self.cfg("display_budget")),
                     title=name[:120], text=f"{NEEDS[need].title} as a service: try it on your own tasks."[:500],
                     ref=pid, kind="display")
        cid = c["campaign"]
        self._do(ctx, inc, "work.display_campaign", campaign=cid, need=need, days=int(self.cfg("display_days")))
        mem["last_display"] = int(ctx.now)
        actions.append(["display_campaign", cid])

    def _cash(self, st, org: str) -> int:
        return sum(int(st.tables["accounts"][a]["balance"]) for a in st.find_ids("accounts", "owner", org)
                   if st.tables["accounts"][a]["status"] == "open")

    def _staff(self, ctx, inc, mem: dict, actions: list) -> None:
        """The staffing rule (module docstring), from the firm's own books and the public labour market."""
        st = ctx.state
        org = inc["org"]
        left = len(self._do(ctx, inc, "support.list", owner=org, status="unresolved", limit=50))
        hist = (list(mem.get("backlog") or []) + [int(left)])[-21:]
        mem["backlog"] = hist
        team = self._do(ctx, inc, "labor.team", org=org)
        payroll = sum(int(m["salary"]) for m in team)
        cash = self._cash(st, org)
        smin, smax = int(self.cfg("salary_min")), int(self.cfg("salary_max"))
        posts = [pid for pid in st.find_ids("labor_posts", "org", org) if st.tables["labor_posts"][pid]["status"] == "open"]
        # counters from applicants: accepted within the range, otherwise rejected
        for o in self._do(ctx, inc, "labor.offers", as_org=org):
            if o["from"] == "worker" and o["status"] == "open":
                ok = int(o["salary"]) <= smax and cash >= 6 * (payroll + int(o["salary"]))
                self._do(ctx, inc, "labor.respond", offer=o["offer"], action="accept" if ok else "reject",
                         message="" if ok else "above our range", as_org=org)
                actions.append(["counter", o["worker"], "accepted" if ok else "rejected"])
        team = self._do(ctx, inc, "labor.team", org=org)
        payroll = sum(int(m["salary"]) for m in team)              # salary is due until the notice date
        staff = [m for m in team if not m.get("on_notice")]        # who stays (notice given by either side)
        if staff and (cash < 2 * payroll or (len(staff) >= 2 and len(hist) >= 21 and max(hist) == 0)):
            last = max(staff, key=lambda m: (-(m["tenure_days"] or 0), m["worker"]))
            self._do(ctx, inc, "labor.dismiss", worker=last["worker"], org=org)
            actions.append(["dismissed", last["worker"], "cash" if cash < 2 * payroll else "no backlog"])
            return
        busy = len(hist) >= 2 and sum(hist[-2:]) / 2.0 >= float(self.cfg("hire_backlog"))
        if busy and not posts and len(staff) < int(self.cfg("max_staff")) and cash >= 6 * (payroll + smax):
            r = self._do(ctx, inc, "labor.post", org=org, title="Customer support (part-time)",
                         skills={"writing": 0.3}, salary_min=smin, salary_max=smax,
                         hours_week=int(self.cfg("job_hours")),
                         description="Answer and resolve our customers' support tickets.")
            actions.append(["posted", r["post"]])
            return
        for post in posts:
            apps = self._do(ctx, inc, "labor.applicants", post=post)
            if any(a["status"] == "offered" for a in apps):
                pending = [o for o in self._do(ctx, inc, "labor.offers", as_org=org)
                           if o["post"] == post and int(o["expires"]) > ctx.now]
                if pending:
                    continue
            if any(a["status"] == "hired" for a in apps) or not busy:
                self._do(ctx, inc, "labor.close_post", post=post)
                actions.append(["closed", post])
                continue
            cands = [a for a in apps if a["status"] == "applied"]
            if not cands:
                continue

            def rank(a):
                rec = (a["profile"].get("verified_record") or {}).get("operations") or {}
                return (-float(rec.get("mean_quality") or 0.0), -int("writing" in a["profile"]["declared_skills"]),
                        a["application"])
            best = min(cands, key=rank)
            r = self._do(ctx, inc, "labor.offer", application=best["application"],
                         salary_cents_month=(smin + smax) // 2, title="Support agent", notice_days=14)
            actions.append(["offered", best["application"], r["offer"]])

    def _list_pass(self, ctx, inc, pid: str, price: int, actions: list, mem: dict) -> None:
        """The marketplace rule (module docstring), from public venue statistics only."""
        today = ctx.now // DAY
        left = {k: d for k, d in (mem.get("left_venues") or {}).items() if today - int(d) < 30}
        venues = [v for v in self._do(ctx, inc, "xchg.exchanges", limit=50) if v["operator"] != inc["org"]]
        mine = [x for x in self._do(ctx, inc, "xchg.browse", product=pid, limit=50) if x["seller"] == inc["org"]]

        def target(v):
            return int(round(price * float(self.cfg("pass_markup")) / (1.0 - int(v["take_rate_bps"]) / 10_000)))
        by_id = {v["exchange"]: v for v in venues}
        for x in mine:
            v = by_id.get(x["exchange"])
            bad = v is None or (v["orders_30d"] >= 4 and v["disputes_refunded_30d"] > 0.5 * v["orders_30d"])
            moved = v is not None and abs(int(x["price_cents"]) - target(v)) > 0.05 * target(v)
            if bad or moved:
                self._do(ctx, inc, "xchg.delist", listing=x["listing"])
                actions.append(["delisted", x["listing"], "venue refunds" if bad else "price moved"])
                mine = [y for y in mine if y["listing"] != x["listing"]]
                if bad:
                    left[x["exchange"]] = int(today)
        mem["left_venues"] = left
        venues = [v for v in venues if v["exchange"] not in left]
        if mine or not venues:
            return
        best = min(venues, key=lambda v: (-int(v["buyers_30d"]), int(v["take_rate_bps"]), v["exchange"]))
        r = self._do(ctx, inc, "xchg.list", exchange=best["exchange"], product=pid, price_cents=target(best),
                     term_days=int(self.cfg("pass_days")))
        actions.append(["listed", r["listing"], best["exchange"], target(best)])

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
