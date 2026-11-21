"""Household demand for physical goods (module `shoppers`): a vectorised population with preferences over item
*attributes*, not over product categories.

Each household has a region, a preference weight vector over the four attributes (durability, performance,
efficiency, appeal), a price sensitivity, an outside option (how good an alternative use of the money is), a
purchase frequency, a small awareness set of retail listings, and neighbours. On a purchase occasion it chooses the
aware listing with the highest utility

    u = 10 * (w . a) - beta * price / price_scale - distance_penalty * hops + Gumbel noise

and buys only if that beats its outside option. `a` is the measured attribute vector of the stock on offer, so a
new design that serves an under-served region of preference space finds buyers without anybody defining a
category for it. Categories are descriptive clusters of what sells (`shoppers.census`).

Awareness spreads through regional discovery, search (retail listings are indexed), advertising impressions and
word of mouth from neighbours' purchases. Purchases are capped by the household budget pool and by stock; unmet
demand is a measured `stockout` loss of the seller. Employed workers' surplus income flows back into the
household budget (wages become demand).
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import Module, applier, process, register_module

DAY = 86400
K = 8


@register_module
class Shoppers(Module):
    name = "shoppers"
    depends = ("identity", "bank", "industry")

    def install(self, world):
        st = world.state
        st.declare_table("shop_slots", index=("slot",))
        st.declare_table("shop_days")
        self.cfg = {"n": 20000, "rate": [0.005, 0.03], "beta": [0.5, 1.8], "u0": [3.5, 1.2], "price_scale": 2500.0,
                    "hop_penalty": 1.5, "discover": 0.15, "wom": 0.08, "income_per_day_cents": 2_000_000,
                    "worker_spend": 0.3}
        self.cfg.update(self.config)

    def genesis(self, ctx):
        n = int(self.cfg["n"])
        ctx.emit("shoppers.genesis", {"n": n}, vis="internal")
        ctx.schedule(ctx.now + DAY - 6 * 3600, "shoppers.day", {}, prio=40, period=DAY, key="shoppers.day")

    @applier("shoppers.genesis")
    def ap_genesis(self, ev):
        st = self.world.state
        c = self.cfg
        n = ev.payload["n"]
        rng = self.world.rng_for("shoppers-genesis")
        regions = sorted(st.tables["regions"]) if "regions" in st.tables else ["r0"]
        st.meta_set("shop_regions", regions)
        st.set_array("s_region", rng.integers(0, len(regions), n).astype(np.int16))
        conc = np.array(c.get("taste_concentration", [0.6, 0.6, 0.6, 0.6]), float)
        st.set_array("s_w", rng.dirichlet(conc, n).astype(np.float32))
        st.set_array("s_beta", rng.uniform(*c["beta"], n).astype(np.float32))
        st.set_array("s_u0", rng.normal(c["u0"][0], c["u0"][1], n).astype(np.float32))
        st.set_array("s_rate", rng.uniform(*c["rate"], n).astype(np.float32))
        st.set_array("s_aware", np.full((n, K), -1, np.int32))
        st.set_array("s_last", np.full(n, -1, np.int32))
        nbr = (np.arange(n)[:, None] + np.array([-2, -1, 1, 2])[None, :]) % n
        rew = rng.random((n, 4)) < 0.2
        nbr[rew] = rng.integers(0, n, int(rew.sum()))
        st.set_array("s_nbr", nbr.astype(np.int32))

    # ------------------------------------------------------------------ helpers
    def hops(self, st) -> dict[tuple[str, str], int]:
        regs = st.meta_get("shop_regions") or []
        h = {(a, a): 0 for a in regs}
        if "links" in st.tables:
            for _, l in st.scan("links"):
                h[(l["a"], l["b"])] = 1
                h[(l["b"], l["a"])] = 1
        return h

    def _slots(self, ctx):
        st = ctx.state
        for rid in st.ids("retail"):
            if not st.has("shop_slots", rid):
                ctx.emit("shoppers.slot", {"id": rid, "slot": st.count("shop_slots")}, vis="internal")

    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        world = ctx.world
        bank = world.modules["bank"]
        ind = world.modules["industry"]
        c = self.cfg
        bank.move(ctx, "acct_external", "acct_households", int(c["income_per_day_cents"]), "income",
                  "household income (goods)")
        self._worker_spending(ctx, bank)
        self._slots(ctx)
        slots = {s["slot"]: rid for rid, s in st.scan("shop_slots")}
        L = len(slots)
        if L == 0:
            return
        regs = st.meta_get("shop_regions")
        hops = self.hops(st)
        price = np.zeros(L, np.float32)
        attr = np.zeros((L, 4), np.float32)
        avail = np.zeros(L, np.int64)
        live = np.zeros(L, bool)
        reg_hops = np.full((len(regs), L), 9, np.int16)
        for k in range(L):
            r = st.tables["retail"][slots[k]]
            s = ind.stock(st, r["fac"], r["sku"])
            if r["status"] != "active":
                continue
            live[k] = True
            price[k] = r["price"]
            avail[k] = s["qty"] if s else 0
            if s is not None and s["v"]:
                attr[k] = s["v"]
            for ri, rg in enumerate(regs):
                reg_hops[ri, k] = hops.get((rg, r["region"]), 9)
        rng = ctx.rng("shoppers-day")
        n = len(st.arrays["s_region"])
        buy = rng.random(n) < st.arrays["s_rate"]
        idx = np.flatnonzero(buy)
        aware = st.arrays["s_aware"][idx]
        valid = aware >= 0
        a = np.where(valid, aware, 0)
        h = reg_hops[st.arrays["s_region"][idx][:, None], a]
        ok = valid & live[a] & (avail[a] > 0) & (h <= 1)
        u = 10.0 * np.einsum("nkd,nd->nk", attr[a], st.arrays["s_w"][idx]) \
            - st.arrays["s_beta"][idx][:, None] * price[a] / float(c["price_scale"]) \
            - float(c["hop_penalty"]) * h + rng.gumbel(0, 0.6, a.shape).astype(np.float32)
        u = np.where(ok | (valid & live[a] & (h <= 1)), u, -1e9)    # sold-out items are still wanted (stockout)
        best = np.argmax(u, axis=1)
        ub = u[np.arange(len(idx)), best]
        choose = ub > st.arrays["s_u0"][idx]
        chosen_slot = a[np.arange(len(idx)), best]
        demand = np.bincount(chosen_slot[choose], minlength=L)
        hh = bank.balance(st, "acct_households")
        sold = {}
        ops = world.hooks.get("ops")
        for k in np.flatnonzero(demand):
            rid = slots[int(k)]
            r = st.tables["retail"][rid]
            q = int(min(demand[k], avail[k], hh // max(1, int(price[k]))))
            if q > 0:
                ind.take_stock(ctx, r["fac"], r["sku"], q)
                dst = bank.primary_account(st, r["seller"])
                if dst:
                    bank.move(ctx, "acct_households", dst, int(q * price[k]), "sale", f"{q} x {r['sku']}", ref=rid)
                    hh -= int(q * price[k])
            sold[str(int(k))] = q
            miss = int(demand[k]) - q
            if miss > 0 and ops is not None:
                ops.record_loss(ctx, r["seller"], "stockout", int(miss * price[k]), miss, resource=rid)
        buyers = idx[choose]
        ctx.emit("shoppers.day", {"day": ctx.now // DAY, "sold": sold,
                                  "demand": {str(int(k)): int(v) for k, v in enumerate(demand) if v},
                                  "buyers": buyers.astype(int).tolist()[:5000],
                                  "bought": chosen_slot[choose].astype(int).tolist()[:5000],
                                  "live": np.flatnonzero(live).astype(int).tolist()}, vis="internal")

    def _worker_spending(self, ctx, bank):
        st = ctx.state
        if "npc_workers" not in st.tables:
            return
        share = float(self.cfg["worker_spend"])
        for pid, w in st.scan("npc_workers"):
            acct = bank.primary_account(st, pid)
            if acct is None:
                continue
            surplus = bank.balance(st, acct) - 2 * int(w["reservation"])
            if surplus > 1000:
                bank.move(ctx, acct, "acct_households", int(surplus * share), "consumption", "household spending")

    @applier("shoppers.slot")
    def ap_slot(self, ev):
        self.world.state.insert("shop_slots", ev.payload["id"], {"slot": ev.payload["slot"], "t": ev.t})

    @applier("shoppers.day")
    def ap_day(self, ev):
        """Awareness dynamics (deterministic given the payload): regional discovery, word of mouth, memory."""
        st = self.world.state
        p = ev.payload
        c = self.cfg
        rng = self.world.rng_for("shoppers-pop", p["day"])
        n = len(st.arrays["s_region"])
        aware = st.array_mut("s_aware")
        last = st.array_mut("s_last")
        live = np.array(p["live"], np.int64)
        for k, v in p["sold"].items():
            rid = next((r for r, s in st.scan("shop_slots") if s["slot"] == int(k)), None)
            if rid:
                rec = st.tables["retail"][rid]
                st.patch("retail", rid, sold_y=v, sold_total=rec.get("sold_total", 0) + v)
        if p["buyers"]:
            b = np.array(p["buyers"], np.int64)
            last[b] = np.array(p["bought"], np.int32)
        if len(live) == 0:
            return
        regs = st.meta_get("shop_regions")
        slot_region = {}
        for rid, s in st.scan("shop_slots"):
            slot_region[s["slot"]] = regs.index(st.tables["retail"][rid]["region"]) \
                if st.tables["retail"][rid]["region"] in regs else -1
        sr = np.array([slot_region.get(int(k), -1) for k in live])
        # regional discovery: learn of a random live listing in your own region
        disc = np.flatnonzero(rng.random(n) < c["discover"])
        if len(disc):
            for ri in range(len(regs)):
                cand = live[sr == ri]
                sel = disc[st.arrays["s_region"][disc] == ri]
                if len(cand) and len(sel):
                    self._learn(aware, sel, cand[rng.integers(0, len(cand), len(sel))], rng)
        # word of mouth: learn what a neighbour last bought
        wom = np.flatnonzero(rng.random(n) < c["wom"])
        if len(wom):
            nb = st.arrays["s_nbr"][wom, rng.integers(0, 4, len(wom))]
            item = last[nb]
            m = item >= 0
            self._learn(aware, wom[m], item[m], rng)
        # search/advertising: a few households look up listings matching their dominant attribute
        srch = np.flatnonzero(rng.random(n) < c.get("search", 0.02))
        if len(srch):
            self._learn(aware, srch, live[rng.integers(0, len(live), len(srch))], rng)

    @staticmethod
    def _learn(aware, who, slot, rng):
        for i, s in zip(who.tolist(), slot.tolist()):
            row = aware[i]
            if s in row:
                continue
            free = np.flatnonzero(row < 0)
            j = int(free[0]) if len(free) else int(rng.integers(0, K))
            row[j] = s

    def census(self, world, eps: float = 0.12) -> list[dict]:
        """Descriptive product clusters: retail items that sold, grouped by attribute-vector proximity."""
        st = world.state
        ind = world.modules["industry"]
        items = []
        for rid, r in st.scan("retail"):
            if r.get("sold_total", 0) <= 0:
                continue
            s = ind.stock(st, r["fac"], r["sku"])
            v = list(s["v"]) if s and s["v"] else None
            if v:
                items.append((rid, r, np.array(v)))
        groups: list[list] = []
        for it in items:
            for g in groups:
                if np.abs(g[0][2] - it[2]).sum() < eps * 4:
                    g.append(it)
                    break
            else:
                groups.append([it])
        return [{"centre": np.mean([x[2] for x in g], axis=0).round(3).tolist(),
                 "listings": [x[0] for x in g], "sellers": sorted({x[1]["seller"] for x in g}),
                 "units_sold": int(sum(x[1].get("sold_total", 0) for x in g))} for g in groups]
