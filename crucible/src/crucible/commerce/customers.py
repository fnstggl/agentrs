"""Independent customers with their own money (module `patrons`, host `patrons`).

The 4.0 consumer population pays from one pooled household account, one transfer per offering-hour, so revenue
cannot be attributed to distinct customers. These participants are individual principals (persons, `npc`) with
their own bank accounts, savings from genesis and a monthly exogenous income (ledger kind `income`, paid by the
rest-of-world account in one transaction). They act only through the ordinary tools, as themselves, from the
daily process. When the v5 population (workstream B) lands, it can drive the same tools for its individuals
(each of which gets a principal and an account through `substrate.ids.ensure_principal`); this module then is
one behavioural class among others.

Subscription buyers. Each has one need (a public need format of `economy.needs`), a willingness to pay per month,
a quality requirement and a preference for annual commitment, all drawn from the world seed. On its active days a
buyer:
* without a subscription, sometimes searches the catalogue (`prod.catalog`) for launched products serving its
  need; it skips products it has rejected, ranks the rest by expected value (public paying-customer ratings, or
  an uninformed prior) minus the monthly-equivalent price, prefers a free trial, and subscribes from its own
  account when it can afford it;
* with a subscription (trialing or active), *uses* the product (`prod.use`) on problem instances of its own need
  and scores every response with the need's exact verifier - the measured usefulness. It cancels during the trial
  (no charge) or at period end when the measured quality is below its requirement, when the price exceeds its
  willingness to pay, or when it cannot pay; it opens a support ticket (`prod.support`) when calls fail; it rates
  products it has paid for from what it measured; satisfied annual-minded buyers move to an annual plan
  (`subs.change_plan`) after two paid months.

Goods buyers and sellers on exchanges (`xchg`). Sellers hold stock of a good (minted at genesis, conserved),
sell it as an industrial product and list it on the exchange where they expect the most net revenue: recent
buyers (+1) x (1 - take rate), from the exchanges' public statistics; they move when another exchange is better
by a margin, and price at cost x markup / (1 - take rate). Buyers who need units pick the cheapest listing whose
seller is rated acceptably, among exchanges whose dispute record they trust; they rate sellers and dispute
orders that did not deliver. Network effects arise from these choices.

Nothing here reads hidden state: buyers see what any participant sees (catalogue, plans, ratings, exchange
statistics) plus their own results.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError, contain
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module

DAY = 86400
MONTH = 30 * DAY
DEFAULTS = {"n": 0, "prefix": "cust", "needs": {"sort.records": 1.0}, "income_month_cents": [30_000, 150_000],
            "savings_cents": [5_000, 60_000], "wtp_month_cents": [800, 8_000], "min_quality": [0.75, 0.97],
            "search_p": 0.2, "act_every_days": 2, "checks": 2, "size": 0.4, "annual_share": 0.3, "rate_p": 0.5,
            "prior_quality": 0.8, "support_p": 0.5, "first_day": 1,
            "goods": None, "sellers": None}
GOODS_DEFAULTS = {"n": 0, "sku": "widget", "units_per_month": [1, 4], "wtp_unit_cents": [600, 2_000],
                  "income_month_cents": [20_000, 60_000], "buy_p": 0.5, "min_rating": 2.5, "max_dispute_share": 0.5}
SELLER_DEFAULTS = {"n": 0, "sku": "widget", "stock": [40, 120], "unit_cost_cents": [300, 900],
                   "markup": [1.2, 1.6], "act_every_days": 3, "switch_margin": 1.25, "cash": 50_000}


def _draw(rng, lohi):
    lo, hi = lohi
    return float(rng.uniform(lo, hi))


@register_module
class Patrons(Module):
    name = "patrons"
    host = "patrons"
    depends = ("identity", "bank", "products", "subs")

    def install(self, world):
        st = world.state
        st.declare_table("patrons", index=("role",))

    def cfg(self, k):
        return self.config.get(k, DEFAULTS.get(k))

    def genesis(self, ctx):
        from ..economy.needs import NEEDS
        w = ctx.world
        idm, bank = w.modules["identity"], w.modules["bank"]
        rng = w.rng_for("patrons-genesis")
        rows = []
        prefix = str(self.cfg("prefix"))
        needs = dict(self.cfg("needs") or {})
        for nd in needs:
            if nd not in NEEDS:
                raise ValueError(f"patrons: unknown need {nd!r}")
        names = sorted(needs)
        weights = np.array([float(needs[k]) for k in names])
        weights = weights / weights.sum() if len(names) else weights
        endow: list[tuple[str, int]] = []
        for i in range(int(self.cfg("n"))):
            pid = idm.create_person(ctx, f"{prefix}-{i:05d}", f"Customer {i}", "npc", {"role": "customer"})
            acct = bank.open_account(ctx, pid, "checking")
            need = str(rng.choice(names, p=weights))
            rows.append({"id": pid, "role": "subscriber", "account": acct, "need": need,
                         "wtp": int(_draw(rng, self.cfg("wtp_month_cents"))),
                         "min_q": round(_draw(rng, self.cfg("min_quality")), 4),
                         "income": int(_draw(rng, self.cfg("income_month_cents"))),
                         "annual": bool(rng.random() < float(self.cfg("annual_share"))), "slot": i})
            endow.append((acct, int(_draw(rng, self.cfg("savings_cents")))))
        g = dict(GOODS_DEFAULTS, **(self.config.get("goods") or {}))
        for i in range(int(g["n"])):
            pid = idm.create_person(ctx, f"{prefix}-g{i:04d}", f"Buyer {i}", "npc", {"role": "goods_buyer"})
            acct = bank.open_account(ctx, pid, "checking")
            rows.append({"id": pid, "role": "goods_buyer", "account": acct, "sku": g["sku"],
                         "units": int(rng.integers(g["units_per_month"][0], g["units_per_month"][1] + 1)),
                         "wtp": int(_draw(rng, g["wtp_unit_cents"])), "income": int(_draw(rng, g["income_month_cents"])),
                         "slot": i})
            endow.append((acct, int(_draw(rng, g["income_month_cents"]))))
        s = dict(SELLER_DEFAULTS, **(self.config.get("sellers") or {}))
        market = w.modules.get("market")
        if int(s["n"]) and market is None:
            raise ValueError("patrons: goods sellers need the market module (inventory)")
        for i in range(int(s["n"])):
            pid = idm.create_person(ctx, f"{prefix}-s{i:04d}", f"Seller {i}", "npc", {"role": "seller"})
            acct = bank.open_account(ctx, pid, "checking")
            stock = int(rng.integers(s["stock"][0], s["stock"][1] + 1))
            market.mint_goods(ctx, pid, s["sku"], stock)
            prod = ctx.call("prod.create", {"name": f"{s['sku']} from seller {i}", "kind": "industrial",
                                            "delivery": {"type": "goods", "sku": s["sku"]}}, as_principal=pid)["product"]
            ctx.call("prod.update", {"product": prod, "status": "launched"}, as_principal=pid)
            rows.append({"id": pid, "role": "seller", "account": acct, "sku": s["sku"], "product": prod,
                         "cost": int(_draw(rng, s["unit_cost_cents"])), "markup": round(_draw(rng, s["markup"]), 3),
                         "slot": i})
            endow.append((acct, int(s["cash"])))
        if endow:
            total = sum(c for _, c in endow)
            bank.post(ctx, [("acct_genesis", -total)] + [(a, c) for a, c in endow], "endowment", "customer savings")
        for r in rows:
            r.update(current=None, rejected=[], rated=[], q=None, checks=0, listing=None, bought=0)
        if rows:
            ctx.emit("patrons.genesis", {"rows": rows}, vis="internal")
            ctx.schedule(int(self.cfg("first_day")) * DAY + 7 * 3600, "patrons.day", {}, prio=70, period=DAY,
                         key="patrons.day")
            ctx.schedule(MONTH, "patrons.income", {}, prio=5, period=MONTH, key="patrons.income")

    # ------------------------------------------------------------------ processes
    @process("income")
    def pr_income(self, ctx):
        st = ctx.state
        posts = []
        for pid, r in st.scan("patrons"):
            a = st.get("accounts", r["account"])
            if a is not None and a["status"] == "open" and r.get("income"):
                posts.append((r["account"], int(r["income"])))
        if posts:
            ctx.world.modules["bank"].post(ctx, [("acct_external", -sum(c for _, c in posts))] + posts, "income",
                                           "monthly income")

    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        day = ctx.now // DAY
        every = max(1, int(self.cfg("act_every_days")))
        sev = max(1, int(dict(SELLER_DEFAULTS, **(self.config.get("sellers") or {}))["act_every_days"]))
        for pid, r in st.scan("patrons"):
            k = sev if r["role"] == "seller" else every
            if (int(r["slot"]) + day) % k:
                continue
            fn = {"subscriber": self._subscriber, "goods_buyer": self._goods_buyer, "seller": self._seller}[r["role"]]
            err = contain(ctx, fn, ctx, pid)
            if err is not None:
                ctx.emit("patrons.failed", {"id": pid, "error": f"{type(err).__name__}: {err}"[:200]}, vis="internal")

    def _set(self, ctx: Ctx, pid: str, **changes) -> None:
        ctx.emit("patrons.state", {"id": pid, "changes": changes}, vis="internal")

    def _call(self, ctx: Ctx, who: str, tool: str, args: dict):
        """A tool call as the participant; an in-world refusal is returned as (None, error)."""
        try:
            return ctx.call(tool, args, as_principal=who), None
        except CrucibleError as e:
            return None, e

    # ------------------------------------------------------------------ subscription buyers
    def _subscriber(self, ctx: Ctx, pid: str) -> None:
        from ..economy.needs import NEEDS
        st = ctx.state
        r = st.tables["patrons"][pid]
        sm = ctx.world.modules["subs"]
        rng = ctx.rng("patron", pid)
        sid = None
        for x in st.find_ids("sub_subs", "customer", pid):
            if st.tables["sub_subs"][x]["status"] in ("trialing", "active", "past_due"):
                sid = x
        if sid is None:
            if rng.random() < float(self.cfg("search_p")):
                self._search(ctx, pid, r, rng)
            return
        s = st.tables["sub_subs"][sid]
        bank = ctx.world.modules["bank"]
        if s["status"] == "past_due":
            if bank.available(st, r["account"]) < int(s["price"]):
                self._call(ctx, pid, "subs.cancel", {"subscription": sid, "at_period_end": False})
            return
        nd = NEEDS[r["need"]]
        qs, failed = [], 0
        for _ in range(max(1, int(self.cfg("checks")))):
            inp, hidden = nd.generate(rng, float(self.cfg("size")))
            res, err = self._call(ctx, pid, "prod.use", {"product": s["product"], "input": inp})
            if err is not None or res.get("status") != 200:
                failed += 1
                qs.append(0.0)
                continue
            qs.append(float(nd.score(inp, hidden, res.get("body"))))
        q = float(np.mean(qs))
        prev = r["q"]
        qe = q if prev is None else 0.5 * prev + 0.5 * q
        self._set(ctx, pid, q=round(qe, 4), checks=int(r["checks"]) + 1, current=s["product"])
        if failed and rng.random() < float(self.cfg("support_p")):
            self._call(ctx, pid, "prod.support", {"product": s["product"], "subject": "requests are failing",
                                                  "body": f"{failed} of {len(qs)} requests failed", "severity": 2})
        monthly = int(s["price"]) / (12 if s["interval"] == "year" else 1)
        bad = qe < float(r["min_q"])
        if bad or monthly > int(r["wtp"]):
            if s["status"] == "trialing":          # a failed trial ends now, before anything is charged
                self._call(ctx, pid, "subs.cancel", {"subscription": sid, "at_period_end": False})
            elif not s["cancel_at_period_end"]:     # a paid period is used up, then not renewed
                self._call(ctx, pid, "subs.cancel", {"subscription": sid, "at_period_end": True})
            if bad:
                self._set(ctx, pid, rejected=sorted(set(r["rejected"]) | {s["product"]}), q=None, current=None)
            return
        if s["product"] not in r["rated"] and int(s["periods_paid"]) >= 1 and rng.random() < float(self.cfg("rate_p")):
            stars = int(1 + round(4 * min(1.0, max(0.0, (qe - 0.5) / 0.5))))
            _, err = self._call(ctx, pid, "prod.rate", {"product": s["product"], "stars": stars})
            if err is None:
                self._set(ctx, pid, rated=sorted(set(r["rated"]) | {s["product"]}))
        if r["annual"] and s["interval"] == "month" and int(s["periods_paid"]) >= 2 and s["status"] == "active":
            for pl in sm.public_plans(st, s["product"]):
                if pl["interval"] == "year" and pl["price_cents"] / 12 < int(s["price"]):
                    self._call(ctx, pid, "subs.change_plan", {"subscription": sid, "plan": pl["plan"]})
                    break

    def _search(self, ctx: Ctx, pid: str, r: dict, rng) -> None:
        st = ctx.state
        cands, err = self._call(ctx, pid, "prod.catalog", {"need": r["need"], "limit": 50})
        if err is not None or not cands:
            return
        bank = ctx.world.modules["bank"]
        cash = bank.available(st, r["account"])
        prior = float(self.cfg("prior_quality"))
        best = None
        for c in cands:
            if c["product"] in r["rejected"] or not c["plans"]:
                continue
            qexp = c["rating"] / 5.0 if c["rating"] is not None and c["ratings"] >= 3 else prior
            plans = sorted(c["plans"], key=lambda p: (p["interval"] != ("year" if r["annual"] else "month"),
                                                     p["monthly_equivalent_cents"], p["plan"]))
            for pl in plans:
                monthly = float(pl["monthly_equivalent_cents"])
                if monthly > r["wtp"]:
                    continue
                if pl["trial_days"] == 0 and cash < pl["price_cents"]:
                    continue
                value = r["wtp"] * qexp - monthly + (0.1 * r["wtp"] if pl["trial_days"] else 0.0)
                value += float(rng.normal(0.0, 0.05 * r["wtp"]))
                if best is None or value > best[0]:
                    best = (value, pl["plan"], c["product"])
                break
        if best is None:
            return
        res, err = self._call(ctx, pid, "subs.subscribe", {"plan": best[1]})
        if err is None:
            self._set(ctx, pid, current=best[2], q=None)

    # ------------------------------------------------------------------ goods buyers
    def _goods_buyer(self, ctx: Ctx, pid: str) -> None:
        st = ctx.state
        r = st.tables["patrons"][pid]
        g = dict(GOODS_DEFAULTS, **(self.config.get("goods") or {}))
        rng = ctx.rng("goods", pid)
        if "xchg_listings" not in st.tables or rng.random() >= float(g["buy_p"]):
            return
        held = st.get("inventory", f"{pid}|{r['sku']}")
        month_units = sum(o["qty"] for o in st.find("xchg_orders", "buyer", pid) if o["t"] > ctx.now - MONTH)
        if month_units >= int(r["units"]):
            return
        exs, err = self._call(ctx, pid, "xchg.exchanges", {"limit": 50})
        if err is not None or not exs:
            return
        trusted = {e["exchange"] for e in exs
                   if e["disputes_30d"] == 0 or e["disputes_refunded_30d"] / e["disputes_30d"] >= g["max_dispute_share"]}
        rows, err = self._call(ctx, pid, "xchg.browse", {"delivery": "goods", "sort": "price", "limit": 50})
        if err is not None:
            return
        for x in rows or []:
            if x["exchange"] not in trusted or x["price_cents"] > int(r["wtp"]):
                continue
            if x["seller_rating"] is not None and x["seller_ratings"] >= 3 and x["seller_rating"] < g["min_rating"]:
                continue
            res, err = self._call(ctx, pid, "xchg.buy", {"listing": x["listing"], "qty": 1})
            if err is None:
                after = st.get("inventory", f"{pid}|{r['sku']}")
                got = (after["qty"] if after else 0) - (held["qty"] if held else 0)
                if got < 1:
                    self._call(ctx, pid, "xchg.dispute", {"order": res["order"], "reason": "nothing was delivered"})
                elif rng.random() < 0.5:
                    self._call(ctx, pid, "xchg.rate", {"order": res["order"], "stars": 4 if x["price_cents"] <
                                                       0.8 * int(r["wtp"]) else 3})
                self._set(ctx, pid, bought=int(r["bought"]) + 1)
                return

    # ------------------------------------------------------------------ goods sellers
    def _seller(self, ctx: Ctx, pid: str) -> None:
        st = ctx.state
        r = st.tables["patrons"][pid]
        s = dict(SELLER_DEFAULTS, **(self.config.get("sellers") or {}))
        if "xchg_exchanges" not in st.tables:
            return
        inv = st.get("inventory", f"{pid}|{r['sku']}")
        cur = st.get("xchg_listings", r["listing"] or "")
        if inv is None or inv["qty"] <= 0:
            if cur is not None and cur["status"] == "active":
                self._call(ctx, pid, "xchg.delist", {"listing": r["listing"]})
            return
        exs, err = self._call(ctx, pid, "xchg.exchanges", {"limit": 50})
        if err is not None or not exs:
            return
        score = {e["exchange"]: (e["buyers_30d"] + 1) * (1 - e["take_rate_bps"] / 10_000) for e in exs}
        fee = {e["exchange"]: e["take_rate_bps"] for e in exs}
        best = max(sorted(score), key=lambda k: score[k])
        here = cur["exchange"] if cur is not None and cur["status"] == "active" else None
        want_price = int(round(r["cost"] * r["markup"] / max(0.05, 1 - fee[best] / 10_000)))
        if here is not None and here in score:
            if score[best] <= float(s["switch_margin"]) * score[here]:
                price = int(round(r["cost"] * r["markup"] / max(0.05, 1 - fee[here] / 10_000)))
                if price != cur["price"]:
                    self._call(ctx, pid, "xchg.delist", {"listing": r["listing"]})
                    res, e2 = self._call(ctx, pid, "xchg.list", {"exchange": here, "product": r["product"],
                                                                 "price_cents": price})
                    if e2 is None:
                        self._set(ctx, pid, listing=res["listing"])
                return
            self._call(ctx, pid, "xchg.delist", {"listing": r["listing"]})
        res, err = self._call(ctx, pid, "xchg.list", {"exchange": best, "product": r["product"],
                                                      "price_cents": want_price})
        if err is None:
            self._set(ctx, pid, listing=res["listing"])

    # ------------------------------------------------------------------ endpoint
    @endpoint("census", S.obj(), read_only=True)
    def ep_census(self, ctx):
        """How many independent customers, goods buyers and sellers live in this world (public statistics)."""
        st = ctx.state
        out = {"subscriber": 0, "goods_buyer": 0, "seller": 0}
        for r in st.tables["patrons"].values():
            out[r["role"]] = out.get(r["role"], 0) + 1
        return out

    # ------------------------------------------------------------------ appliers
    @applier("patrons.genesis")
    def ap_genesis(self, ev):
        st = self.world.state
        for r in ev.payload["rows"]:
            st.insert("patrons", r["id"], {k: v for k, v in r.items() if k != "id"})

    @applier("patrons.state")
    def ap_state(self, ev):
        self.world.state.patch("patrons", ev.payload["id"], **ev.payload["changes"])

    @applier("patrons.failed")
    def ap_failed(self, ev):
        pass
