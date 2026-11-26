"""Tier 2 domain 9.13 - the digital marketplace (kernel module `mktplace`, host `mkt`; MASTER_SPEC Part 9.13).

Example objective: build a sustainable exchange that independently attracts buyers and sellers.

The exchanges are the commerce package's own (`crucible.commerce.marketplace`, module `xchg`: listings, orders,
settlement through `bank`, the take rate, buyer protection and disputes decided by the operator, ratings from
buyers of settled orders, public 30-day statistics); what is sold are launched `products` of goods delivery, held as
conserved `market` inventory. This module adds the two sides of the market and the physics of the goods:

* **Goods** come in categories. A seller's production process has a hidden quality level and spread and a unit
  cost that rises with quality; every unit a buyer receives has a measured quality drawn from its seller's process
  (`mkt.unit`, visible to the buyer). Nothing is declared about quality except what sellers claim in their titles.
* **Sellers** are autonomous traders (persons with their own money) with heterogeneous costs, quality, capacity and
  honesty: an honest seller claims the grade its process delivers, a dishonest one claims the top grade whatever it
  makes. Each week a seller decides where to list from the exchanges' public statistics (recent distinct buyers -
  the cross-side network effect - and the take rate), sets its price from its cost, the take rate and its own sales
  (stock piling up lowers it, selling out raises it), produces to its recent sales, and leaves an exchange where it
  stopped selling. New traders enter while the market's buyer base grows.
* **Buyers** are the population (`crucible.population`, D-15). A `shop` session of an individual with a due need in
  a category: it picks an exchange it knows (its own experienced trust - the Beta mean of good outcomes there -
  the number of listings in the category it can see, loyalty), browses that exchange's listings in the category and
  chooses by claimed grade (discounted by the seller's public rating and its own trust in the exchange), price
  against its willingness to pay and price sensitivity (multinomial logit with an outside option), and buys with its
  own money. The unit it receives has the quality it has; a unit well below the claimed grade is disputed, and the
  outcome (refund or rejection by the operator, or the automatic refund when the operator does not decide) feeds
  its trust; it rates the seller from what it got; it tells contacts who ask which exchange served it well; it
  learns of exchanges from contacts, promotions and the public directory.
* **Operators**: an incumbent exchange run by an autonomous operator with a high take rate and a seller-friendly
  dispute rule; any participant may open others (`xchg.create`) and run them: fees (`xchg.set_fee`), dispute
  decisions (`xchg.resolve`), removing sellers (`mkt.ban`), paying promotions that reach individuals
  (`mkt.promote`), seller incentives (`mkt.incentive`: a paid bonus per sale for a period) and test purchases
  (`xchg.buy` + `mkt.inspect`).

The independent evaluator reads the recorded log: independent population buyers and independent sellers with sales
on the participant's exchanges, its operating result (commissions minus refunds it paid, promotions and incentives),
and buyer retention. The scripted reference (`Reference`) acts only through tools.
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
TABLE = "mkt"
X_MAX = 6                      # exchanges an individual can know
MODEL_ID = "mkt.shop.logit"
FIDELITY = "mkt.shop"
GRADES = ("C", "B", "A")       # claimed grade thresholds on measured quality: C >= 0.0, B >= 0.45, A >= 0.75
GRADE_Q = {"C": 0.0, "B": 0.45, "A": 0.75}
CATEGORIES = {
    "household": {"items": ("kettle", "desk lamp", "blender", "toaster", "fan"), "value": 4000},
    "outdoor": {"items": ("backpack", "tent", "water bottle", "headlamp", "sleeping bag"), "value": 6000},
    "electronics": {"items": ("headphones", "charger", "speaker", "keyboard", "webcam"), "value": 8000},
    "crafts": {"items": ("ceramic mug", "wool scarf", "leather wallet", "candle set", "print"), "value": 3000},
}
DEFAULTS = {"categories": ["household", "outdoor"], "sellers": 14, "honest_share": 0.6, "entrants_per_week": 1,
            "incumbents": [{"name": "BigMart", "take_rate_bps": 1500, "refund_share": 0.35, "resolve_days": 3}],
            "need_rate_per_day": [0.03, 0.12], "promote_cents_per_reach": 40, "start_listing": "incumbent"}


def grade_of(q: float) -> str:
    return "A" if q >= GRADE_Q["A"] else "B" if q >= GRADE_Q["B"] else "C"


def generate_instance(rng, cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    cats = list(c["categories"])
    sellers = []
    for k in range(int(c["sellers"]) + 8 * int(c["entrants_per_week"])):
        cat = cats[k % len(cats)]
        q = float(np.clip(rng.beta(2.2, 2.0), 0.05, 0.97))
        honest = bool(rng.random() < float(c["honest_share"]))
        base = CATEGORIES[cat]["value"]
        sellers.append({"handle": f"trader{k:02d}", "category": cat,
                        "item": CATEGORIES[cat]["items"][int(rng.integers(0, 5))],
                        "q": round(q, 3), "spread": round(float(rng.uniform(0.04, 0.15)), 3),
                        "unit_cost": int(base * (0.25 + 0.45 * q) * float(rng.uniform(0.85, 1.15))),
                        "capacity_day": int(rng.integers(2, 8)), "honest": honest,
                        "markup": round(float(rng.uniform(0.15, 0.6)), 3),
                        "entry_week": 0 if k < int(c["sellers"]) else 1 + (k - int(c["sellers"])) //
                        max(1, int(c["entrants_per_week"])), "cash": int(rng.integers(150_000, 600_000))})
    c["seller_specs"] = sellers
    return c


@register_module
class Marketplace(Module):
    name = "mktplace"
    host = "mkt"
    version = "1"
    depends = ("identity", "bank", "products", "market", "xchg", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.c = dict(DEFAULTS)
        self.c.update(self.config)
        self.cats = list(self.c["categories"])
        self.specs = {s["handle"]: s for s in self.c.get("seller_specs") or []}

    def install(self, world):
        st = world.state
        for t, ix in (("mkt_sellers", ("status",)), ("mkt_units", ("buyer", "seller")), ("mkt_slots", ()),
                      ("mkt_bans", ("exchange",)), ("mkt_promos", ("exchange",)), ("mkt_incentives", ("exchange",))):
            st.declare_table(t, index=ix)
        from .. import population as P
        n = P.n_individuals(world)
        C = len(self.cats)
        columns.declare(world, TABLE, n, {
            "need_at": ("int64", (C,), 0), "rate": ("float16", (C,), 0.05), "wtp": ("float32", (), 1.0),
            "beta": ("float16", (), 1.0), "aware": ("int16", (X_MAX,), -1), "x_n": ("int16", (X_MAX,), 0),
            "x_ok": ("int16", (X_MAX,), 0), "last_x": ("int16", (), -1), "last_ok_x": ("int16", (), -1),
            "bought": ("int32", (), 0), "disputed": ("int32", (), 0), "spent": ("int64", (), 0)})
        P.register_activity(world, "shop", self.act_shop, owner=self.name, prior=0.5, goals=(2,))

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        for h, nm in (("factory", "Contract Manufacturing"), ("adnet", "Regional Ad Network")):
            mgr = idm.create_person(ctx, f"{h}-mgr", f"{nm} manager", "npc", {"role": "manager"})
            bank.open_account(ctx, mgr, "checking")
            ctx.emit("mkt.slot", {"id": f"_{h}", "principal": mgr, "slot": -1}, vis="internal")
        # the incumbent exchanges, each run by an autonomous operator with its own dispute rule
        for k, inc in enumerate(self.c["incumbents"]):
            op = idm.create_person(ctx, f"incumbent{k}-op", f"{inc['name']} operator", "npc", {"role": "operator"})
            acct = bank.open_account(ctx, op, "checking")
            bank.move(ctx, "acct_genesis", acct, 2_000_000, "endowment", "operator capital")
            ex = ctx.call("xchg.create", {"name": inc["name"], "take_rate_bps": int(inc["take_rate_bps"]),
                                          "description": f"{inc['name']}: an established online market.",
                                          "resolve_days": int(inc["resolve_days"]) + 4}, as_principal=op)["exchange"]
            ctx.emit("mkt.incumbent", {"exchange": ex, "operator": op, "k": k}, vis="public")
            self._slot(ctx, ex)
        for h, s in sorted(self.specs.items()):
            if s["entry_week"] == 0:
                self._enter(ctx, h)
        self._genesis_people(ctx)
        base = (ctx.now // DAY) * DAY
        ctx.schedule(base + 6 * HOUR, "mktplace.day", {}, prio=45, period=DAY, key="mktplace.day")
        ctx.schedule(base + 7 * DAY + 5 * HOUR, "mktplace.week", {}, prio=44, period=7 * DAY, key="mktplace.week")

    def _slot(self, ctx, ex: str) -> int:
        st = ctx.state
        if st.has("mkt_slots", ex):
            return st.tables["mkt_slots"][ex]["slot"]
        n = sum(1 for _, r in st.scan("mkt_slots") if r["slot"] >= 0)
        if n >= X_MAX:
            return -1
        ctx.emit("mkt.slot", {"id": ex, "principal": None, "slot": n}, vis="internal")
        return n

    def _enter(self, ctx, handle: str) -> None:
        """A trader enters: its account and savings, its product (goods of its own sku) and first production run;
        it lists on the incumbent (or where its rule sends it) at its first weekly review."""
        w = ctx.world
        s = self.specs[handle]
        idm = ident(w)
        bank = w.modules["bank"]
        p = idm.create_person(ctx, handle, f"{s['item'].title()} trader {handle[-2:]}", "npc", {"role": "seller"})
        acct = bank.open_account(ctx, p, "checking")
        bank.move(ctx, "acct_genesis", acct, int(s["cash"]), "endowment", f"savings of {p}")
        sku = f"{s['item'].replace(' ', '-')}-{handle}"
        prd = ctx.call("prod.create", {"name": f"{s['item'].title()} by {handle}", "kind": "industrial",
                                       "description": f"A {s['item']}.",
                                       "delivery": {"type": "goods", "sku": sku}}, as_principal=p)["product"]
        ctx.emit("mkt.seller", {"id": p, "handle": handle, "category": s["category"], "item": s["item"], "sku": sku,
                                "product": prd}, vis="public")
        self._produce(ctx, p, max(4, 3 * int(s["capacity_day"])))
        ctx.call("prod.update", {"product": prd, "status": "launched"}, as_principal=p)
        self._review(ctx, p, first=True)

    def _produce(self, ctx, seller: str, units: int) -> int:
        st = ctx.state
        r = st.tables["mkt_sellers"][seller]
        s = self.specs[r["handle"]]
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, seller)
        units = min(int(units), int(bank.available(st, src) // max(1, s["unit_cost"])))
        if units <= 0:
            return 0
        fac = st.tables["mkt_slots"]["_factory"]["principal"]
        bank.move(ctx, src, bank.primary_account(st, fac), units * int(s["unit_cost"]), "purchase",
                  f"production of {units} {s['item']}")
        ctx.emit("market.stock", {"owner": seller, "sku": r["sku"], "delta": int(units), "minted": True},
                 vis=(seller,))
        ctx.emit("mkt.produced", {"seller": seller, "units": int(units)}, vis=(seller,))
        return units

    def _genesis_people(self, ctx):
        from .. import population as P
        w = ctx.world
        n = P.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("mkt-people")
        f = P.features(w, np.arange(n))
        prefs = f["prefs"].astype(np.float64)
        C = len(self.cats)
        lo, hi = self.c["need_rate_per_day"]
        rate = rng.uniform(lo, hi, (n, C)) * (0.6 + 0.8 * rng.random((n, C)))
        need = (rng.exponential(1.0, (n, C)) / rate * DAY).astype(np.int64)
        wtp = np.clip(rng.lognormal(0.0, 0.3, n), 0.4, 2.5)
        beta = 0.5 + 1.5 * prefs[:, 0]
        aware = np.full((n, X_MAX), -1, np.int16)
        aware[:, 0] = 0                                         # everyone knows the incumbent
        columns.batch(ctx, TABLE, "mkt.init", np.arange(n), {
            "need_at": need, "rate": rate.astype(np.float16), "wtp": wtp.astype(np.float32),
            "beta": beta.astype(np.float16), "aware": aware}, extra={"what": "needs and willingness to pay"})

    # ------------------------------------------------------------------ helpers
    def exchanges(self, st) -> dict[int, str]:
        return {r["slot"]: ex for ex, r in st.scan("mkt_slots") if r["slot"] >= 0
                and st.tables["xchg_exchanges"][ex]["status"] == "open"}

    def _listings(self, st, ex: str, cat: str) -> list[dict]:
        out = []
        for lid in st.find_ids("xchg_listings", "exchange", ex):
            x = st.tables["xchg_listings"][lid]
            if x["status"] != "active":
                continue
            r = st.get("mkt_sellers", x["seller"])
            if r is None or r["category"] != cat:
                continue
            inv = st.get("inventory", f"{x['seller']}|{r['sku']}")
            if inv is None or inv["qty"] <= 0:
                continue
            out.append(dict(x, id=lid))
        return out

    def _claimed(self, title: str) -> str:
        for g in ("A", "B", "C"):
            if f"grade {g}" in title:
                return g
        return "C"

    # ------------------------------------------------------------------ sellers' routines
    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        for seller, r in sorted(st.scan("mkt_sellers")):
            if r["status"] != "active":
                continue
            err = contain(ctx, self._restock, ctx, seller, r)
            if err is not None:
                ctx.emit("mkt.failed", {"what": "restock", "seller": seller, "error": str(err)[:200]}, vis="internal")
        for iid, inc in sorted(st.scan("mkt_incentives")):
            if inc["status"] == "active" and inc["until"] <= ctx.now:
                ctx.emit("mkt.incentive_status", {"id": iid, "status": "ended"}, vis="public")

    def _restock(self, ctx, seller, r):
        st = ctx.state
        s = self.specs[r["handle"]]
        inv = (st.get("inventory", f"{seller}|{r['sku']}") or {}).get("qty", 0)
        recent = sum(u["qty"] for u in st.find("mkt_units", "seller", seller) if u["t"] >= ctx.now - 7 * DAY)
        target = max(3, int(math.ceil(recent / 7.0 * 4)))
        make = min(int(s["capacity_day"]), target - inv)
        if make > 0:
            self._produce(ctx, seller, make)

    @process("week")
    def pr_week(self, ctx):
        st = ctx.state
        week = int(ctx.now // (7 * DAY))
        for h, s in sorted(self.specs.items()):
            if s["entry_week"] == week and not st.has("mkt_sellers", f"p:{h}"):
                if self._market_attractive(st):
                    err = contain(ctx, self._enter, ctx, h)
                    if err is not None:
                        ctx.emit("mkt.failed", {"what": "entry", "seller": h, "error": str(err)[:200]},
                                 vis="internal")
        self._pay_weekly(ctx)
        for seller, r in sorted(st.scan("mkt_sellers")):
            if r["status"] == "active":
                err = contain(ctx, self._review, ctx, seller)
                if err is not None:
                    ctx.emit("mkt.failed", {"what": "review", "seller": seller, "error": str(err)[:200]},
                             vis="internal")

    def _pay_weekly(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        for iid, inc in sorted(st.scan("mkt_incentives")):
            if inc["status"] != "active" or not inc["cents_per_week"]:
                continue
            ex = inc["exchange"]
            op = st.tables["xchg_exchanges"][ex]["operator"]
            src = bank.primary_account(st, op)
            listed = sorted({st.tables["xchg_listings"][lid]["seller"] for lid in st.find_ids("xchg_listings",
                                                                                              "exchange", ex)
                             if st.tables["xchg_listings"][lid]["status"] == "active"})
            for seller in listed:
                dst = bank.primary_account(st, seller)
                if src is None or dst is None or bank.available(st, src) < inc["cents_per_week"]:
                    break
                bank.move(ctx, src, dst, int(inc["cents_per_week"]), "incentive", f"weekly listing bonus {ex}")
                ctx.emit("mkt.incentive_paid", {"id": iid, "cents": int(inc["cents_per_week"]), "seller": seller},
                         vis="public")

    def _market_attractive(self, st) -> bool:
        """Traders enter while the exchanges' public buyer counts keep up with the number of active sellers."""
        xs = self.world.modules["xchg"]
        buyers = sum(xs.stats(st, ex, self.world.now)["buyers_30d"] for ex in self.exchanges(st).values())
        sellers = sum(1 for _, r in st.scan("mkt_sellers") if r["status"] == "active")
        return buyers >= 6 * max(1, sellers)

    def _review(self, ctx, seller, first: bool = False):
        """A trader's weekly review from public statistics and its own books: list where buyers are (expected
        buyers per seller in its category after the take rate), price from cost, take rate and own stock, leave
        exchanges where it sold nothing for two weeks or where it was banned."""
        st = ctx.state
        w = ctx.world
        xs = w.modules["xchg"]
        r = st.tables["mkt_sellers"][seller]
        s = self.specs[r["handle"]]
        rng = ctx.rng("mkt-review", seller)
        now = ctx.now
        inv = (st.get("inventory", f"{seller}|{r['sku']}") or {}).get("qty", 0)
        sold = sum(u["qty"] for u in st.find("mkt_units", "seller", seller) if u["t"] >= now - 7 * DAY)
        mark = float(r["markup"])
        if not first:
            mark = mark * (0.9 if inv > 4 * max(1, sold) else 1.08 if inv == 0 or sold > 2 * s["capacity_day"] else 1.0)
            mark = float(np.clip(mark, 0.05, 1.5))
        claim = grade_of(s["q"]) if s["honest"] else "A"
        mine = {x["exchange"]: lid for lid in st.find_ids("xchg_listings", "seller", seller)
                for x in [st.tables["xchg_listings"][lid]] if x["status"] == "active"}
        decisions = []
        for slot, ex in sorted(self.exchanges(st).items()):
            e = st.tables["xchg_exchanges"][ex]
            banned = st.has("mkt_bans", f"{ex}|{seller}")
            stt = xs.stats(st, ex, now)
            same_cat = max(1, sum(1 for x in self._listings(st, ex, r["category"])))
            take = int(e["take_rate_bps"]) / 10_000.0
            incentive = sum(i["cents_per_sale"] for i in st.find("mkt_incentives", "exchange", ex)
                            if i["status"] == "active")
            weekly = sum(i["cents_per_week"] for i in st.find("mkt_incentives", "exchange", ex)
                         if i["status"] == "active")
            # cross-side network effect: expected sales from the exchange's recent buyers shared among the
            # category's sellers, valued at the margin left after the take rate (plus any incentive)
            exp_sales = (stt["buyers_30d"] + 0.5) / same_cat / 4.0
            margin = s["unit_cost"] * mark * (1 - take) - s["unit_cost"] * take + incentive
            value = exp_sales * margin + weekly
            ex_sold = sum(u["qty"] for u in st.find("mkt_units", "seller", seller)
                          if u["exchange"] == ex and u["t"] >= now - 14 * DAY)
            listed_since = (st.tables["xchg_listings"][mine[ex]]["t"] if ex in mine else now)
            stay = ex in mine and not banned and (ex_sold > 0 or now - listed_since < 14 * DAY)
            # try a new exchange when its buyers can be expected to cover a week's listing effort (~a unit's margin)
            join = ex not in mine and not banned and (first and slot == 0 or value >= 0.6 * s["unit_cost"] * mark
                                                      * float(rng.uniform(0.7, 1.3)))
            price = int(math.ceil(s["unit_cost"] * (1 + mark) / max(0.5, 1 - take)))
            title = f"{s['item'].title()} - grade {claim}"
            if ex in mine and not stay:
                ctx.call("xchg.delist", {"listing": mine[ex]}, as_principal=seller)
                decisions.append({"exchange": ex, "action": "leave", "expected_value": round(value, 1),
                                  "buyers_30d": stt["buyers_30d"], "take_rate_bps": e["take_rate_bps"]})
            elif ex in mine:
                cur = st.tables["xchg_listings"][mine[ex]]
                if abs(cur["price"] - price) > 0.02 * price:
                    ctx.call("xchg.delist", {"listing": mine[ex]}, as_principal=seller)
                    ctx.call("xchg.list", {"exchange": ex, "product": r["product"], "price_cents": price,
                                           "title": title}, as_principal=seller)
                    decisions.append({"exchange": ex, "action": "reprice", "price": price, "old": cur["price"]})
            elif join and inv > 0:
                ctx.call("xchg.list", {"exchange": ex, "product": r["product"], "price_cents": price,
                                       "title": title}, as_principal=seller)
                decisions.append({"exchange": ex, "action": "join", "expected_value": round(value, 1),
                                  "buyers_30d": stt["buyers_30d"], "take_rate_bps": e["take_rate_bps"],
                                  "price": price})
        ctx.emit("mkt.review", {"seller": seller, "markup": round(mark, 4), "inventory": int(inv), "sold_7d": int(sold),
                                "decisions": decisions}, vis="internal")

    # ------------------------------------------------------------------ the incumbent's dispute rule
    def _incumbent_disputes(self, ctx):
        st = ctx.state
        rng = ctx.rng("mkt-incumbent")
        incs = {r["exchange"]: (r["principal"], self.c["incumbents"][r["k"]]) for key, r in st.scan("mkt_slots")
                if key.startswith("_incumbent")}
        for oid in st.find_ids("xchg_orders", "status", "disputed"):
            o = st.tables["xchg_orders"][oid]
            if o["exchange"] not in incs:
                continue
            op, inc = incs[o["exchange"]]
            if ctx.now - o["updated"] < int(inc["resolve_days"]) * DAY:
                continue
            pct = 100 if rng.random() < float(inc["refund_share"]) else 0
            err = contain(ctx, lambda: ctx.call("xchg.resolve", {"order": oid, "refund_pct": pct,
                                                                 "note": "operator review"}, as_principal=op))
            if err is not None:
                ctx.emit("mkt.failed", {"what": "incumbent dispute", "order": oid, "error": str(err)[:200]},
                         vis="internal")

    # ------------------------------------------------------------------ the population's shopping
    def act_shop(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.zeros(m)
        if m == 0:
            return util
        self._incumbent_disputes(ctx)
        from .. import population as P
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        need_at, rate, wtp, beta = col("need_at"), col("rate").astype(np.float64), col("wtp"), \
            col("beta").astype(np.float64)
        aware, x_n, x_ok, last_x, last_ok = col("aware"), col("x_n"), col("x_ok"), col("last_x"), col("last_ok_x")
        bought, disputed, spent = col("bought"), col("disputed"), col("spent")
        rng = ctx.rng("mkt-shop")
        f = P.features(w, idx)
        loyal = np.clip(f["prefs"][:, 5].astype(np.float64), 0, 1)
        contacts = columns.col(w, "pop", "contacts")
        lok_all = columns.col(w, TABLE, "last_ok_x")
        exs = self.exchanges(st)
        wallet = P.spendable(w, idx)
        recs = []
        self._resolve_my_units(ctx, idx, aware, x_n, x_ok, last_ok, disputed, recs)
        for j in range(m):
            due = np.flatnonzero(need_at[j] <= now)
            if due.size == 0:
                continue
            c = int(due[0])
            cat = self.cats[c]
            i = int(idx[j])
            # awareness from a contact who was served well
            if rng.random() < 0.2:
                cs = contacts[i][contacts[i] >= 0]
                if cs.size:
                    k = int(lok_all[int(cs[int(rng.integers(0, cs.size))])])
                    if k >= 0 and k not in aware[j]:
                        free = np.flatnonzero(aware[j] < 0)
                        if free.size:
                            aware[j, int(free[0])] = k
            known = [int(k) for k in aware[j] if k >= 0 and int(k) in exs]
            need_at[j, c] = now + int(rng.exponential(1.0) / max(float(rate[j, c]), 1e-3) * DAY)
            if not known:
                recs.append([i, cat, None, None, "no exchange"])
                continue
            # ---- exchange choice: own trust, visible depth of the category, loyalty
            u = []
            depth = {}
            for k in known:
                ls = self._listings(st, exs[k], cat)
                depth[k] = ls
                kk = list(aware[j]).index(k)
                trust = (x_ok[j, kk] + 1.0) / (x_n[j, kk] + 2.0)
                u.append(3.0 * (trust - 0.5) + 0.8 * math.log1p(len(ls)) + 1.0 * loyal[j] * (k == last_x[j]))
            kx = known[int(K.logit_choice(rng, np.array(u)[None, :], None, 0.7)[0])]
            kk = list(aware[j]).index(kx)
            ls = depth[kx]
            if not ls:
                recs.append([i, cat, exs[kx], None, "empty"])
                x_n[j, kk] += 1
                continue
            # ---- listing choice: claimed grade discounted by public rating and own trust, price vs wtp
            trust = (x_ok[j, kk] + 1.0) / (x_n[j, kk] + 2.0)
            base = CATEGORIES[cat]["value"] * float(wtp[j])
            uu, opts = [], []
            xs = w.modules["xchg"]
            for x in ls[:30]:
                g = self._claimed(x["title"])
                avg, nr = xs.seller_rating(st, exs[kx], x["seller"])
                believed = GRADE_Q[g] + 0.2 if g != "C" else 0.3
                if avg is not None:
                    w_r = nr / (nr + 3.0)
                    believed = (1 - w_r) * believed * (0.5 + trust) + w_r * (avg - 1) / 4.0
                else:
                    believed = believed * (0.5 + trust)
                val = base * (0.4 + 0.9 * believed)
                uu.append((val - beta[j] * x["price"]) / 1000.0)
                opts.append(x)
            uu.append(0.0)                                      # outside option: do not buy now
            pick = int(K.logit_choice(rng, np.array(uu)[None, :], None, 0.8)[0])
            if pick >= len(opts):
                recs.append([i, cat, exs[kx], None, "no purchase"])
                continue
            x = opts[pick]
            if wallet[j] < x["price"]:
                recs.append([i, cat, exs[kx], x["id"], "cannot afford"])
                continue
            try:
                from ..population.core import money_deposit
                p = ids.ensure_principal(ctx, i, account=True)
                money_deposit(ctx, i)                       # accrued income into its own account
                res = ctx.call("xchg.buy", {"listing": x["id"], "qty": 1}, as_principal=p)
            except CrucibleError as e:
                recs.append([i, cat, exs[kx], x["id"], f"failed: {str(e)[:60]}"])
                continue
            q = self._unit(ctx, res["order"], x, p)
            wallet[j] -= x["price"]
            bought[j] += 1
            spent[j] += x["price"]
            last_x[j] = kx
            util[j] = float(np.clip((base * (0.4 + 0.9 * q) - x["price"]) / 2000.0, -1.5, 1.5))
            recs.append([i, cat, exs[kx], x["id"], "bought", res["order"], round(q, 3), x["price"]])
        K.decide(ctx, "mkt.shop", idx, {
            "need_at": need_at, "aware": aware, "x_n": x_n, "x_ok": x_ok, "last_x": last_x, "last_ok_x": last_ok,
            "bought": bought, "disputed": disputed, "spent": spent},
            table=TABLE, model_id=MODEL_ID, fidelity_id=FIDELITY, extra={"t": int(now), "shopping": recs})
        attention.spend(ctx, idx, np.full(m, 8.0), activity="shop")
        return util

    def _unit(self, ctx, order: str, x: dict, buyer: str) -> float:
        """The unit the buyer received: its measured quality drawn from the seller's process (world physics)."""
        st = ctx.state
        r = st.tables["mkt_sellers"][x["seller"]]
        s = self.specs[r["handle"]]
        g = ctx.world.rng_for("mkt-unit", order)
        q = float(np.clip(g.normal(s["q"], s["spread"]), 0.0, 1.0))
        ctx.emit("mkt.unit", {"id": order, "buyer": buyer, "seller": x["seller"], "exchange": x["exchange"],
                              "category": r["category"], "quality": round(q, 4), "claimed": self._claimed(x["title"]),
                              "price": int(x["price"]), "qty": 1}, vis=(buyer,))
        # per-sale incentives of the exchange's operator (paid while it can pay)
        bank = ctx.world.modules["bank"]
        for iid, inc in sorted(st.scan("mkt_incentives")):
            if inc["exchange"] != x["exchange"] or inc["status"] != "active" or not inc["cents_per_sale"]:
                continue
            op = st.tables["xchg_exchanges"][x["exchange"]]["operator"]
            src, dst = bank.primary_account(st, op), bank.primary_account(st, x["seller"])
            if src is not None and dst is not None and bank.available(st, src) >= inc["cents_per_sale"]:
                bank.move(ctx, src, dst, int(inc["cents_per_sale"]), "incentive", f"sale bonus {order}", ref=order)
                ctx.emit("mkt.incentive_paid", {"id": iid, "cents": int(inc["cents_per_sale"]), "seller": x["seller"]},
                         vis="public")
        return q

    def _resolve_my_units(self, ctx, idx, aware, x_n, x_ok, last_ok, disputed, recs):
        """Individuals of this session look at what they bought: rate it, dispute a unit well below its claimed
        grade, and learn from how their open disputes ended (trust in the exchange)."""
        st = ctx.state
        w = ctx.world
        rng = ctx.rng("mkt-after")
        slots = {ex: r["slot"] for ex, r in st.scan("mkt_slots") if r["slot"] >= 0}
        for j, i in enumerate(idx):
            p = f"p{int(i)}"
            if not st.has("principals", p):
                continue
            for u in st.find("mkt_units", "buyer", p):
                if u["done"]:
                    continue
                o = st.tables["xchg_orders"].get(u["order"])
                if o is None:
                    continue
                k = slots.get(u["exchange"])
                if k is None or k not in aware[j]:
                    continue
                kk = list(aware[j]).index(k)
                short = GRADE_Q[u["claimed"]] - u["quality"]
                if o["status"] == "settled" and not o["disputed"] and short > 0.15 and \
                        ctx.now <= o["dispute_until"] and rng.random() < 0.8:
                    try:
                        ctx.call("xchg.dispute", {"order": u["order"], "reason": f"received grade {grade_of(u['quality'])}"
                                                                             f", listed as grade {u['claimed']}"},
                                 as_principal=p)
                        disputed[j] += 1
                        ctx.emit("mkt.unit_status", {"id": u["order"], "stage": "disputed"}, vis=(p,))
                    except CrucibleError:
                        pass
                    continue
                if o["status"] == "disputed":
                    continue
                good = (o["refunded"] >= o["amount"]) if o["disputed"] else short <= 0.15
                x_n[j, kk] += 1
                x_ok[j, kk] += int(good)
                if good:
                    last_ok[j] = k
                if not o["disputed"] and o["status"] in ("settled", "resolved") and rng.random() < 0.5:
                    stars = int(np.clip(round(1 + 4 * u["quality"] - 3 * max(0.0, short)), 1, 5))
                    try:
                        ctx.call("xchg.rate", {"order": u["order"], "stars": stars}, as_principal=p)
                    except CrucibleError:
                        pass
                ctx.emit("mkt.unit_status", {"id": u["order"], "stage": "done", "good": bool(good)}, vis=(p,))
                recs.append([int(i), u["category"], u["exchange"], None, "outcome", u["order"], bool(good)])

    # ------------------------------------------------------------------ operators' tools
    def _operator_of(self, ctx, exchange: str) -> dict:
        e = ctx.state.get("xchg_exchanges", exchange)
        if e is None:
            raise NotFound(f"no exchange {exchange}")
        ident(ctx.world).authorize(ctx, e["operator"], "org.admin")
        return e

    @endpoint("register", S.obj({"exchange": S.s_str(maxLength=40)}, ["exchange"]), cost_s=60)
    def ep_register(self, ctx, exchange):
        """List an exchange you operate in the public market directory (buyers and traders consider only listed
        exchanges)."""
        e = self._operator_of(ctx, exchange)
        if e["status"] != "open":
            raise Conflict("exchange is closed")
        slot = self._slot(ctx, exchange)
        if slot < 0:
            raise Conflict("the directory is full")
        return {"exchange": exchange, "listed": True}

    @endpoint("directory", S.obj(), read_only=True)
    def ep_directory(self, ctx):
        """The public market directory: listed exchanges with their take rate and public 30-day statistics."""
        st = ctx.state
        xs = ctx.world.modules["xchg"]
        out = []
        for slot, ex in sorted(self.exchanges(st).items()):
            e = st.tables["xchg_exchanges"][ex]
            out.append({"exchange": ex, "name": e["name"], "operator": e["operator"],
                        "take_rate_bps": e["take_rate_bps"], **xs.stats(st, ex, ctx.now)})
        return {"exchanges": out, "categories": self.cats}

    @endpoint("traders", S.obj(), read_only=True)
    def ep_traders(self, ctx):
        """Public trader profiles: what each trader sells and where it is listed."""
        st = ctx.state
        out = []
        for seller, r in sorted(st.scan("mkt_sellers")):
            where = sorted({st.tables["xchg_listings"][lid]["exchange"] for lid in st.find_ids("xchg_listings",
                                                                                               "seller", seller)
                            if st.tables["xchg_listings"][lid]["status"] == "active"})
            out.append({"trader": seller, "category": r["category"], "item": r["item"], "product": r["product"],
                        "listed_on": where, "status": r["status"]})
        return {"traders": out}

    @endpoint("ban", S.obj({"exchange": S.s_str(maxLength=40), "seller": S.s_str(maxLength=40)},
                           ["exchange", "seller"]), cost_s=60)
    def ep_ban(self, ctx, exchange, seller):
        """Remove a trader from an exchange you operate (its listings close; it cannot list there again)."""
        self._operator_of(ctx, exchange)
        st = ctx.state
        for lid in st.find_ids("xchg_listings", "seller", seller):
            x = st.tables["xchg_listings"][lid]
            if x["exchange"] == exchange and x["status"] == "active":
                ctx.emit("xchg.listing_status", {"id": lid, "status": "closed"}, vis="public")
        ctx.emit("mkt.ban", {"id": f"{exchange}|{seller}", "exchange": exchange, "seller": seller}, vis="public")
        return {"ok": True}

    @endpoint("promote", S.obj({"exchange": S.s_str(maxLength=40), "cents": S.s_int(minimum=100, maximum=10**8)},
                               ["exchange", "cents"]), cost_s=60)
    def ep_promote(self, ctx, exchange, cents):
        """Buy promotion of a listed exchange from the ad network: each reached individual learns of it (reach =
        cents / price per reach, individuals drawn at random; those who already know it are reached again)."""
        e = self._operator_of(ctx, exchange)
        st = ctx.state
        if not st.has("mkt_slots", exchange):
            raise Conflict("register the exchange in the directory first")
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, e["operator"])
        if src is None or bank.available(st, src) < cents:
            raise InvalidAction("the operator cannot pay for that")
        ad = st.tables["mkt_slots"]["_adnet"]["principal"]
        bank.move(ctx, src, bank.primary_account(st, ad), int(cents), "advertising", f"promotion of {exchange}")
        from .. import population as P
        n = P.n_individuals(ctx.world)
        reach = min(n, int(cents) // int(self.c["promote_cents_per_reach"]))
        rows = np.sort(ctx.world.rng_for("mkt-promo", exchange, ctx.now).choice(n, size=reach, replace=False))
        slot = st.tables["mkt_slots"][exchange]["slot"]
        aw = columns.col(ctx.world, TABLE, "aware")[rows].copy()
        for j in range(rows.size):
            if slot not in aw[j]:
                free = np.flatnonzero(aw[j] < 0)
                if free.size:
                    aw[j, int(free[0])] = slot
        columns.batch(ctx, TABLE, "mkt.promotion", rows, {"aware": aw}, extra={"exchange": exchange, "cents": int(cents)})
        pid = st.new_id("mkpromo")
        ctx.emit("mkt.promo", {"id": pid, "exchange": exchange, "cents": int(cents), "reach": int(reach),
                               "by": ctx.actor}, vis="public")
        return {"reached": int(reach)}

    @endpoint("incentive", S.obj({"exchange": S.s_str(maxLength=40),
                                  "cents_per_sale": S.s_int(minimum=0, maximum=100_000),
                                  "cents_per_week": S.s_int("paid weekly to every trader listed on the exchange",
                                                            minimum=0, maximum=1_000_000),
                                  "days": S.s_int(minimum=1, maximum=60)}, ["exchange", "days"]), cost_s=60)
    def ep_incentive(self, ctx, exchange, days, cents_per_sale=0, cents_per_week=0):
        """Pay traders on your exchange for some days: a bonus per sale (at each sale) and/or a weekly amount to
        every trader listed there (at each weekly review), from the operator's account while it can pay. Traders
        see both when they decide where to list."""
        self._operator_of(ctx, exchange)
        if not cents_per_sale and not cents_per_week:
            raise InvalidAction("give a per-sale or a weekly amount")
        iid = ctx.state.new_id("mkinc")
        ctx.emit("mkt.incentive", {"id": iid, "exchange": exchange, "cents_per_sale": int(cents_per_sale),
                                   "cents_per_week": int(cents_per_week), "until": int(ctx.now + days * DAY)},
                 vis="public")
        return {"incentive": iid}

    @endpoint("inspect", S.obj({"order": S.s_str(maxLength=40)}, ["order"]), read_only=True)
    def ep_inspect(self, ctx, order):
        """Measure a unit you bought (a test purchase): its quality on the 0-1 scale and the grade it meets."""
        st = ctx.state
        u = st.get("mkt_units", order)
        if u is None or u["buyer"] not in ctx.world.principals_of(ctx.actor):
            raise NotFound("you bought no such unit")
        return {"order": order, "seller": u["seller"], "quality": u["quality"], "grade": grade_of(u["quality"]),
                "claimed": u["claimed"]}

    @endpoint("buy_sample", S.obj({"listing": S.s_str(maxLength=40)}, ["listing"]), cost_s=60)
    def ep_buy_sample(self, ctx, listing):
        """Buy one unit of a listing as a test purchase (you pay as any buyer does) and receive it measured."""
        st = ctx.state
        x = st.get("xchg_listings", listing)
        if x is None or x["status"] != "active":
            raise NotFound("no active listing")
        res = ctx.call("xchg.buy", {"listing": listing, "qty": 1})
        q = self._unit(ctx, res["order"], dict(x, id=listing), ctx.actor)
        return {"order": res["order"], "quality": round(q, 4), "grade": grade_of(q),
                "claimed": self._claimed(x["title"])}

    # ------------------------------------------------------------------ appliers
    @applier("mkt.slot")
    def ap_slot(self, ev):
        p = dict(ev.payload)
        self.world.state.put("mkt_slots", p.pop("id"), p)

    @applier("mkt.incumbent")
    def ap_incumbent(self, ev):
        self.world.state.put("mkt_slots", f"_incumbent{ev.payload['k']}", {"exchange": ev.payload["exchange"],
                                                                           "principal": ev.payload["operator"],
                                                                           "slot": -1, "k": ev.payload["k"]})

    @applier("mkt.seller")
    def ap_seller(self, ev):
        p = dict(ev.payload)
        s = self.specs[p["handle"]]
        self.world.state.insert("mkt_sellers", p.pop("id"), p | {"status": "active", "markup": s["markup"],
                                                                 "t": ev.t})

    @applier("mkt.produced")
    def ap_produced(self, ev):
        return None

    @applier("mkt.review")
    def ap_review(self, ev):
        self.world.state.patch("mkt_sellers", ev.payload["seller"], markup=ev.payload["markup"])

    @applier("mkt.unit")
    def ap_unit(self, ev):
        p = dict(ev.payload)
        oid = p.pop("id")
        self.world.state.insert("mkt_units", oid, p | {"order": oid, "t": ev.t, "done": False, "stage": "received"})

    @applier("mkt.unit_status")
    def ap_unit_status(self, ev):
        p = ev.payload
        ch = {"stage": p["stage"]}
        if p["stage"] == "done":
            ch.update(done=True, good=p.get("good"))
        self.world.state.patch("mkt_units", p["id"], **ch)

    @applier("mkt.ban")
    def ap_ban(self, ev):
        p = dict(ev.payload)
        self.world.state.put("mkt_bans", p.pop("id"), p | {"t": ev.t})

    @applier("mkt.promo")
    def ap_promo(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("mkt_promos", p.pop("id"), p | {"t": ev.t})

    @applier("mkt.incentive")
    def ap_incentive(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("mkt_incentives", p.pop("id"), p | {"status": "active", "t": ev.t, "paid": 0})

    @applier("mkt.incentive_status")
    def ap_incentive_status(self, ev):
        self.world.state.patch("mkt_incentives", ev.payload["id"], status=ev.payload["status"])

    @applier("mkt.incentive_paid")
    def ap_incentive_paid(self, ev):
        st = self.world.state
        r = st.tables["mkt_incentives"][ev.payload["id"]]
        st.patch("mkt_incentives", ev.payload["id"], paid=r["paid"] + int(ev.payload["cents"]))

    @applier("mkt.failed")
    def ap_failed(self, ev):
        return None

    def check_invariants(self, world):
        errs = []
        st = world.state
        for oid, u in st.scan("mkt_units"):
            if not 0.0 <= u["quality"] <= 1.0:
                errs.append(f"{oid}: quality outside [0, 1]")
        from .. import population as P
        if columns.rows(world, TABLE) != P.n_individuals(world):
            errs.append("mkt rows differ from the population")
        return errs


# ====================================================================== evaluator side (recorded log only)
OPERATING_KINDS = ("take_rate", "refund", "advertising", "incentive")


def operated(world, principal: str) -> list[str]:
    st = world.state
    mine = {principal}
    if "members" in st.tables:
        mine |= {m["org"] for m in st.find("members", "person", principal) if m["role"] == "owner"}
    return sorted(ex for ex, e in st.scan("xchg_exchanges") if e["operator"] in mine)


def evaluate(world, principal: str, objective: dict) -> dict:
    """Independent outcome: independent population buyers and independent traders with sales on the
    participant's exchanges in the window, its operating result (commissions minus refunds it paid, promotion and
    incentive spending) over the last 15 days, and buyer retention."""
    st = world.state
    now = world.now
    win = int(objective.get("window_days", 30)) * DAY
    t0 = now - win
    exs = set(operated(world, principal))
    ops = {st.tables["xchg_exchanges"][e]["operator"] for e in exs}
    rel: dict[str, bool] = {}

    def indep(p):
        if p not in rel:
            rel[p] = all(not K.related(world, o, p) for o in ops)
        return rel[p]
    orders = [o for _, o in st.scan("xchg_orders") if o["exchange"] in exs]
    recent = [o for o in orders if o["t"] >= t0]
    buyers = {o["buyer"] for o in recent if K.population_index(o["buyer"]) is not None and indep(o["buyer"])}
    sellers = {o["seller"] for o in recent if st.has("mkt_sellers", o["seller"]) and indep(o["seller"])}
    half = now - win // 2
    early = {o["buyer"] for o in orders if t0 <= o["t"] < half and o["buyer"] in buyers}
    late = {o["buyer"] for o in orders if o["t"] >= half and o["buyer"] in buyers}
    retention = len(early & late) / len(early) if early else 0.0
    accts = {a for a, r in st.scan("accounts") if r["owner"] in ops}
    t15 = now - 15 * DAY
    result = 0
    for _, x in st.scan("acct_txns"):
        if x["acct"] in accts and x["t"] >= t15:
            tx = st.tables["txns"][x["txn"]]
            if tx["kind"] in OPERATING_KINDS:
                result += int(x["amount"])
    gmv = sum(int(o["amount"]) for o in recent)
    need = {"buyers": int(objective.get("min_buyers", 40)), "sellers": int(objective.get("min_sellers", 4)),
            "retention": float(objective.get("min_retention", 0.25))}
    checks = {"buyers": len(buyers) >= need["buyers"], "sellers": len(sellers) >= need["sellers"],
              "sustainable": result > 0 and gmv > 0, "retention": retention >= need["retention"],
              "horizon": now >= int(objective.get("horizon_days", 45)) * DAY}
    score = float(np.mean([min(1.0, len(buyers) / need["buyers"]), min(1.0, len(sellers) / need["sellers"]),
                           1.0 if result >= 0 else 0.0, min(1.0, retention / max(1e-6, need["retention"]))]))
    return {"success": all(checks.values()), "score": round(score, 4),
            "details": {"independent_buyers": len(buyers), "independent_sellers": len(sellers),
                        "operating_result_15d_cents": int(result), "gmv_window_cents": int(gmv),
                        "orders_window": len(recent), "retention": round(retention, 4), "exchanges": sorted(exs),
                        "checks": checks}}


MECHANICS = ("Two-sided participation", "Discovery", "Transactions", "Trust", "Quality", "Disputes", "Pricing",
             "Network effects", "Retention")


def mechanics(world) -> dict:
    return mechanics_report(world, 0)


def mechanics_report(world, since_seq: int = 0) -> dict:
    st = world.state
    c = K.event_counts(world, since_seq)
    shop = K.batch_records(world, "mkt.shop", "shopping", since_seq)
    bought = [r for r in shop if len(r) > 4 and r[4] == "bought"]
    outcomes = [r for r in shop if len(r) > 4 and r[4] == "outcome"]
    by_ex: dict[str, set] = {}
    for _, o in st.scan("xchg_orders"):
        by_ex.setdefault(o["exchange"], set()).add(o["seller"])
    two_sided = sum(1 for ex, sel in by_ex.items() if sel and any(
        K.population_index(o["buyer"]) is not None for _, o in st.scan("xchg_orders") if o["exchange"] == ex))
    q = [u["quality"] for _, u in st.scan("mkt_units")]
    disputed = sum(1 for _, o in st.scan("xchg_orders") if o["disputed"])
    resolved = sum(1 for _, o in st.scan("xchg_orders") if o["disputed"] and o["status"] != "disputed")
    reviews = [ev.payload for ev in world.events_range(since_seq) if ev.type == "mkt.review"]
    reprices = sum(1 for r in reviews for d in r["decisions"] if d["action"] == "reprice")
    joins = [d for r in reviews for d in r["decisions"] if d["action"] == "join" and d.get("buyers_30d", 0) > 0]
    per: dict[tuple, int] = {}
    for r in bought:
        per[(r[0], r[2])] = per.get((r[0], r[2]), 0) + 1
    repeat = sum(1 for v in per.values() if v >= 2)
    return {
        "Two-sided participation": {"exercised": two_sided >= 1 and len(by_ex) >= 1,
                                    "events": {"exchanges_with_both_sides": two_sided,
                                               "xchg.listing": c["xchg.listing"]}},
        "Discovery": {"exercised": len(shop) > 0 and (c["batch:mkt.promotion"] > 0 or c["batch:mkt.shop"] > 0),
                      "events": {"shopping_decisions": len(shop), "batch:mkt.promotion": c["batch:mkt.promotion"]}},
        "Transactions": {"exercised": c["xchg.order"] > 0, "events": {"xchg.order": c["xchg.order"]}},
        "Trust": {"exercised": len(outcomes) > 0 and c["xchg.rating"] > 0,
                  "events": {"trust_updates": len(outcomes), "xchg.rating": c["xchg.rating"]}},
        "Quality": {"exercised": len(q) > 1 and min(q) < GRADE_Q["B"] <= max(q),
                    "events": {"mkt.unit": c["mkt.unit"]}},
        "Disputes": {"exercised": disputed > 0 and resolved > 0, "events": {"disputed": disputed,
                                                                            "resolved_or_refunded": resolved}},
        "Pricing": {"exercised": reprices > 0, "events": {"reprices": reprices,
                                                          "xchg.exchange_update": c["xchg.exchange_update"]}},
        "Network effects": {"exercised": len(joins) > 0, "events": {"joins_on_buyer_counts": len(joins)}},
        "Retention": {"exercised": repeat > 0, "events": {"repeat_buyer_exchange_pairs": repeat}},
    }


# ====================================================================== the domain
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register  # noqa: E402

A = K.A

POPULATION = 700
STRUCTURES = {
    "two_categories": {"categories": ["household", "outdoor"], "sellers": 14, "honest_share": 0.6,
                       "entrants_per_week": 1},
    "electronics_fraud": {"categories": ["electronics"], "sellers": 10, "honest_share": 0.35, "entrants_per_week": 1,
                          "incumbents": [{"name": "BigMart", "take_rate_bps": 1500, "refund_share": 0.2,
                                          "resolve_days": 3}]},
    "crafts_small_traders": {"categories": ["crafts", "household"], "sellers": 22, "honest_share": 0.7,
                             "entrants_per_week": 2,
                             "incumbents": [{"name": "BigMart", "take_rate_bps": 2000, "refund_share": 0.35,
                                             "resolve_days": 3}]},
    "electronics_crafts_entry": {"categories": ["electronics", "crafts"], "sellers": 12, "honest_share": 0.35,
                                 "entrants_per_week": 2},
    "two_incumbents": {"categories": ["household", "outdoor"], "sellers": 16, "honest_share": 0.5,
                       "entrants_per_week": 1,
                       "incumbents": [{"name": "BigMart", "take_rate_bps": 1500, "refund_share": 0.35,
                                       "resolve_days": 3},
                                      {"name": "CheapDeals", "take_rate_bps": 600, "refund_share": 0.05,
                                       "resolve_days": 5}]},
}


def _structure(name):
    s = STRUCTURES[name]
    incs = s.get("incumbents", DEFAULTS["incumbents"])
    return {"categories": sorted(s["categories"]), "incumbents": len(incs),
            "incumbent_dispute_rules": sorted({round(i["refund_share"], 2) for i in incs}),
            "seller_honesty": s["honest_share"], "entry": s["entrants_per_week"],
            "participants": ["population buyers", "autonomous traders", "incumbent operators"]}


CONFIGS = (
    DomainConfig("two_categories", "train", _structure("two_categories"),
                 "household and outdoor goods, one incumbent, mostly honest traders"),
    DomainConfig("electronics_fraud", "train", _structure("electronics_fraud"),
                 "electronics only, most traders overstate their grade, a seller-friendly incumbent"),
    DomainConfig("crafts_small_traders", "train", _structure("crafts_small_traders"),
                 "many small craft and household traders, fast entry, an expensive incumbent"),
    DomainConfig("electronics_crafts_entry", "eval_compositional", _structure("electronics_crafts_entry"),
                 "electronics and crafts together (each seen in training, never together) with dishonest traders "
                 "and fast entry"),
    DomainConfig("two_incumbents", "eval_structural", _structure("two_incumbents"),
                 "two established exchanges compete (a strict expensive one and a cheap one that rejects almost "
                 "every dispute): a market structure never seen in training"),
)


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    ov = dict(overrides or {})
    rng = K.seeded_rng(config.name, seed)
    inst = generate_instance(rng, {**STRUCTURES[config.name], **(ov.pop("instance", {}) or {})})
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("products", {}), ("market", {}), ("xchg", {}), ("pop", K.pop_config(n)),
                            ("mktplace", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    buyers, sellers = 50 + 25 * d, 4 + d
    text = (f"Build an online exchange that buyers and traders choose on their own: by day 45, over "
                           f"the last 30 days at least {buyers} different people must have bought on your exchange "
                           f"from at least {sellers} independent traders, at least 25% of those who bought in the "
                           f"first half of that period must have come back to buy again in the second half, and the "
                           f"exchange must pay for itself over the last 15 days (its commissions must exceed the "
                           f"refunds, promotions and incentives you pay).")
    return {"text": text, "description": text, "difficulty": d, "window_days": 30, "horizon_days": 45, "min_buyers": buyers, "min_sellers": sellers,
            "min_retention": 0.25}


class ExchangeOperator(K.ReadingReference):
    """Scripted reference (evaluator side, tool-only): a low-fee, buyer-protecting exchange. It opens an exchange
    at a 4% take rate and lists it in the directory, subsidises the trader side for three weeks (a weekly listing
    bonus and a per-sale bonus) and promotes the exchange to buyers, refunds every dispute whose buyer received a
    lower grade than listed, removes traders with repeated upheld disputes, then raises its fee to 9% and stops
    the subsidies so that the exchange pays for itself."""
    cycle_s = 12 * HOUR
    fee_bps, late_fee_bps, subsidy = 400, 900, True

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.ex = None
        self.registered = False
        self.started = None
        self.raised = False
        self.promos = 0
        self.upheld: dict[str, int] = {}
        self.banned: set[str] = set()
        self.resolved: set[str] = set()

    def reads(self, t):
        return [("orders", "xchg.mine", {"role": "operator"})] if self.ex else []

    def decide(self, t, got):
        if self.ex is None:
            return [A("xchg.create", "create", name="FairCart", take_rate_bps=self.fee_bps,
                      description="Low fees; every buyer is protected: if what arrives is below the listed grade, "
                                  "you get your money back.", dispute_days=14, resolve_days=3)]
        out = []
        if not self.registered:
            self.registered, self.started = True, t
            out.append(A("mkt.register", None, exchange=self.ex))
            if self.subsidy:
                out.append(A("mkt.incentive", None, exchange=self.ex, cents_per_week=3000, cents_per_sale=250,
                             days=21))
        weeks = (t - self.started) // (7 * DAY)
        if weeks >= self.promos and self.promos < 4:
            self.promos += 1
            out.append(A("mkt.promote", None, exchange=self.ex, cents=40_000 if self.promos == 1 else 16_000))
        if not self.raised and t - self.started >= 24 * DAY and self.late_fee_bps != self.fee_bps:
            self.raised = True
            out.append(A("xchg.set_fee", None, exchange=self.ex, take_rate_bps=self.late_fee_bps))
        orders = got.get("orders")
        for o in orders if isinstance(orders, list) else []:
            if o.get("status") != "disputed" or o["id"] in self.resolved:
                continue
            self.resolved.add(o["id"])
            note = str(o.get("note") or "")
            if "listed as grade" in note:
                out.append(A("xchg.resolve", None, order=o["id"], refund_pct=100, note="below the listed grade"))
                self.upheld[o["seller"]] = self.upheld.get(o["seller"], 0) + 1
            else:
                out.append(A("xchg.resolve", None, order=o["id"], refund_pct=50, note="partial refund"))
        for seller, n in sorted(self.upheld.items()):
            if n >= 3 and seller not in self.banned:
                self.banned.add(seller)
                out.append(A("mkt.ban", None, exchange=self.ex, seller=seller))
        return out

    def handle(self, tag, res, action):
        if tag == "create" and res is not None:
            self.ex = res["exchange"]


class FreeExchange(ExchangeOperator):
    """Degenerate-strategy probe: the same exchange that never charges a fee (buyers and traders come, but it never
    pays for itself)."""
    fee_bps, late_fee_bps = 0, 0


def structure(world) -> dict:
    """Measured on the built world: the goods categories traded, the incumbents' dispute rules (lenient or strict)
    and whether most traders are honest."""
    mod = world.modules["mktplace"]
    rules = sorted({"strict" if float(i["refund_share"]) >= 0.3 else "lenient" for i in mod.c["incumbents"]})
    honest = sum(1 for s in mod.specs.values() if s["honest"]) * 2 > len(mod.specs)
    return {"categories": sorted(mod.cats), "incumbent_rules": rules,
            "incumbents": "one" if len(mod.c["incumbents"]) == 1 else "several", "honest_majority": honest}


Reference = ExchangeOperator

DOMAIN = register(Domain(
    name="marketplace", title="Digital marketplace", spec_part="9.13", required_mechanics=MECHANICS,
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=ExchangeOperator,
    mechanics=tuple(Mechanic(m, (), f"measured by marketplace_domain.mechanics_report: {m}",
                             count=K.measured(mechanics_report, m)) for m in MECHANICS),
    seats=(Seat("agent", 1_000_000, "Exchange founder"),),
    actors=(Actor("buyers", ("cols.batch:mkt.shop",), "population individuals shopping on exchanges they know"),
            Actor("traders", ("mkt.review", "mkt.produced"), "sellers deciding where to list, prices, production"),
            Actor("incumbent operators", ("xchg.order_status",), "orders settled and disputes decided")),
    degenerate=(Probe("free_exchange", "the same exchange without any fee", FreeExchange),),
    structure=structure, mode="B", horizon_days=45.0, slowest_effect_days=14.0, dt=6 * HOUR,
    reference_seeds=(0, 1), sources=("marketplace_domain.py", "_dom3.py"),
    doc=__doc__.split("\n\n")[0]))
