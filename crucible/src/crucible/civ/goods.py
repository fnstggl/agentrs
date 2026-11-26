"""Kernel module `civgoods` (host `goods`): the population buys physical goods from the industrial supply chain.

In a civilisation the households that buy what the supply chain produces are the one population
(`crucible.population`, D-15), not 4.0's separate `shoppers` table. This module registers the population activity
`goods` (ONE_WORLD_CONTRACTS section 1): an individual who spends a session on it

1. learns of listings (awareness, its own column state): browsing local stores (a random live listing of its own
   region), search (any live listing), and - when the coupling `goods_wom` exists - word of mouth: what one of its
   own contacts (`pop.contacts`) last bought;
2. chooses among the listings it knows that it can reach (its region or a neighbouring one, through the freight
   network) by the measured attribute vector of the stock on offer (durability, performance, efficiency, appeal -
   the output of the hidden production physics, never a declared quality), the price relative to its price
   sensitivity, the distance, and noise (multinomial logit, as the 4.0 household model; fidelity row
   `civ.goods.choice`), against its own outside option;
3. buys one unit with its own money (`industry.retail_sale`: stock taken, bank move from its own account to the
   seller) if the stock is there and it can afford it; an out-of-stock choice is a stockout loss of the seller
   (`ops`), which the producers' pricing and production routines read.

The coupling `goods_supply` is this module's link between the population and the supply chain: removed (in a
counterfactual), sessions on `goods` buy nothing and the industry has no household demand.

Per-person state lives in this module's column table `goods` (one row per individual: region, awareness, last
purchase, attribute weights, price sensitivity, outside option); no fact of the population table is copied.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError, NotFound, ProtocolError
from ..core.kernel import Module, applier, endpoint, process, register_module
from ..substrate import attention, columns, ids
from . import popapi

DAY = 86400
TABLE = "goods"
K = 8
MODEL_ID = "civ.goods.attribute_logit"
DEFAULTS = {"placement": "mixed", "discover": 0.35, "search": 0.08, "wom": 0.35, "value_scale": 14.0,
            "price_scale": 2500.0, "hop_penalty": 1.5, "noise": 0.6, "minutes": [8.0, 25.0], "prior": 0.35,
            "regions": None}


def _active(world, coupling: str) -> bool:
    from . import couplings
    return couplings.active(world, coupling)


@register_module
class Goods(Module):
    name = "civgoods"
    host = "goods"
    version = "1"
    depends = ("pop", "identity", "bank", "industry", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"civgoods: unknown config key {k!r}")
            self.cfg[k] = v
        if self.cfg["placement"] not in ("blocks", "scattered", "mixed", "hub"):
            raise ProtocolError("civgoods: placement must be blocks, scattered, mixed or hub")

    def install(self, world):
        n = popapi.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "region": ("int16", (), -1), "aware": ("int32", (K,), -1), "last": ("int32", (), -1),
            "w": ("float16", (4,), 0.25), "beta": ("float16", (), 1.0), "u0": ("float16", (), 3.5),
            "bought_n": ("int32", (), 0), "spent": ("int64", (), 0), "stockouts": ("int32", (), 0),
            "last_t": ("int64", (), -1), "last_seq": ("int64", (), -1)})
        world.state.declare_table("goods_slots", index=("slot",))
        popapi.register_activity(world, "goods", self.act_goods, owner=self.name, prior=float(self.cfg["prior"]),
                                 goals=(2,))

    def genesis(self, ctx):
        w = ctx.world
        n = popapi.n_individuals(w)
        regs = self._regions(w)
        if n and regs:
            rng = w.rng_for("civgoods-init")
            R = len(regs)
            rows = np.arange(n)
            pl = self.cfg["placement"]
            if pl == "blocks":
                reg = (rows * R) // max(1, n)
            elif pl == "scattered":
                reg = rng.integers(0, R, n)
            elif pl == "hub":
                reg = np.where(rng.random(n) < 0.5, 0, rng.integers(0, R, n))
            else:
                reg = np.where(rng.random(n) < 0.3, rng.integers(0, R, n), (rows * R) // max(1, n))
            f = popapi.features(w, rows)
            prefs = f["prefs"].astype(np.float64)
            # attribute weights from the individual's own preferences: durability <- quality sensitivity,
            # performance <- novelty seeking, efficiency <- price sensitivity, appeal <- social conformity
            base = np.stack([prefs[:, 3], prefs[:, 1], prefs[:, 0], prefs[:, 4]], axis=1)
            logit = 2.0 * base + rng.normal(0, 0.5, (n, 4))
            wts = np.exp(logit - logit.max(axis=1, keepdims=True))
            wts /= wts.sum(axis=1, keepdims=True)
            beta = 0.5 + 1.3 * prefs[:, 0]
            u0 = rng.normal(3.5, 1.2, n)
            popapi.record_decisions(ctx, "goods.init", rows, {
                "region": reg.astype(np.int16), "w": wts.astype(np.float16), "beta": beta.astype(np.float16),
                "u0": u0.astype(np.float16)}, table=TABLE, model_id="civ.goods.init", model_version="1",
                fidelity_id="civ.goods.choice", extra={"placement": pl, "regions": regs})
        ctx.schedule((ctx.now // DAY + 1) * DAY + 60, "civgoods.day", {}, prio=29, period=DAY, key="civgoods.day")

    # ------------------------------------------------------------------ helpers
    def _regions(self, world) -> list[str]:
        if self.cfg.get("regions"):
            return list(self.cfg["regions"])
        return sorted(world.state.tables["regions"]) if "regions" in world.state.tables else []

    def _hops(self, st) -> dict[tuple[str, str], int]:
        regs = sorted(st.tables["regions"])
        h = {(a, a): 0 for a in regs}
        if "links" in st.tables:
            for _, l in st.scan("links"):
                h[(l["a"], l["b"])] = 1
                h[(l["b"], l["a"])] = 1
        return h

    def _slots(self, ctx) -> dict[int, str]:
        st = ctx.state
        for rid in sorted(st.ids("retail")):
            if not st.has("goods_slots", rid):
                ctx.emit("goods.slot", {"id": rid, "slot": st.count("goods_slots")}, vis="internal")
        return {s["slot"]: rid for rid, s in st.scan("goods_slots")}

    # ------------------------------------------------------------------ the activity
    def act_goods(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        ind = w.modules["industry"]
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.full(m, -0.05)
        if m == 0:
            return util
        rng = ctx.rng("civgoods")
        c = self.cfg
        mins = rng.uniform(c["minutes"][0], c["minutes"][1], m)
        granted = attention.spend(ctx, idx, mins, activity="goods")
        if not _active(w, "goods_supply"):
            # without the link between households and the supply chain a goods session finds nothing to buy
            popapi.record_decisions(ctx, "goods.session", idx, {"last_t": np.full(m, now, np.int64)}, table=TABLE,
                                    model_id=MODEL_ID, model_version="1", fidelity_id="civ.goods.choice",
                                    extra={"t": int(now), "sessions": int(m), "purchases": [], "coupling": "off"})
            return util
        slots = self._slots(ctx)
        L = len(slots)
        regs = sorted(st.tables["regions"])
        ri = {r: k for k, r in enumerate(regs)}
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        region, aware, last = col("region").astype(np.int64), col("aware").astype(np.int64), col("last")
        wts, beta, u0 = col("w").astype(np.float64), col("beta").astype(np.float64), col("u0").astype(np.float64)
        bought_n, spent, stockouts = col("bought_n"), col("spent"), col("stockouts")
        last_seq = col("last_seq")
        live = np.zeros(L, bool)
        price = np.zeros(L, np.float64)
        avail = np.zeros(L, np.int64)
        attr = np.zeros((L, 4), np.float64)
        lreg = np.full(L, -1, np.int64)
        for k in range(L):
            r = st.tables["retail"][slots[k]]
            if r["status"] != "active":
                continue
            s = ind.stock(st, r["fac"], r["sku"])
            live[k] = True
            price[k] = r["price"]
            avail[k] = s["qty"] if s else 0
            if s is not None and s["v"]:
                attr[k] = s["v"]
            lreg[k] = ri.get(r["region"], -1)
        hops = self._hops(st)
        hop_m = np.full((len(regs), L), 9, np.int64)
        for a, ra in enumerate(regs):
            for k in range(L):
                if lreg[k] >= 0:
                    hop_m[a, k] = hops.get((ra, regs[lreg[k]]), 9)
        live_ix = np.flatnonzero(live)
        # ---- awareness (only listings that exist; a person's own region or anywhere through search)
        if live_ix.size:
            for j in range(m):
                learn = []
                rr = region[j]
                if rng.random() < c["discover"] and rr >= 0:
                    local = live_ix[lreg[live_ix] == rr]
                    if local.size:
                        learn.append(int(local[rng.integers(0, local.size)]))
                if rng.random() < c["search"]:
                    learn.append(int(live_ix[rng.integers(0, live_ix.size)]))
                if _active(w, "goods_wom") and rng.random() < c["wom"]:
                    cont = columns.col(w, "pop", "contacts")[idx[j]]
                    cont = cont[cont >= 0]
                    if cont.size:
                        other = int(cont[rng.integers(0, cont.size)])
                        lb = int(columns.col(w, TABLE, "last")[other])
                        if lb >= 0 and lb < L and lb not in aware[j]:
                            learn.append(lb)
                            src = int(columns.col(w, TABLE, "last_seq")[other])
                            ctx.emit("goods.wom", {"learner": int(idx[j]), "source": other, "retail": slots[lb],
                                                   "refs": [src] if src > 0 else []}, vis="internal")
                for s_ in learn:
                    row = aware[j]
                    if s_ in row:
                        continue
                    free = np.flatnonzero(row < 0)
                    row[int(free[0]) if free.size else int(rng.integers(0, K))] = s_
        # ---- choice among the listings one knows and can reach
        valid = (aware >= 0) & (aware < max(L, 1))
        a = np.where(valid, aware, 0)
        if L:
            h = hop_m[np.maximum(region, 0)[:, None], a]
            reach = valid & live[a] & (h <= 1) & (region[:, None] >= 0)
            u = c["value_scale"] * np.einsum("nkd,nd->nk", attr[a], wts) - beta[:, None] * price[a] / c["price_scale"] \
                - c["hop_penalty"] * h + rng.gumbel(0, c["noise"], a.shape)
            u = np.where(reach, u, -1e9)
            best = np.argmax(u, axis=1)
            ub = u[np.arange(m), best]
            pick = a[np.arange(m), best]
        else:
            ub = np.full(m, -1e9)
            pick = np.full(m, -1)
        wallet = popapi.spendable(w, idx)
        purchases = []
        sold_out = []
        for j in np.flatnonzero((ub > u0) & (granted > 0)):
            k = int(pick[j])
            rid = slots[k]
            i = int(idx[j])
            if avail[k] <= 0:
                stockouts[j] += 1
                sold_out.append([i, rid])
                util[j] = -0.2
                r = st.tables["retail"][rid]
                ops = w.hooks.get("ops")
                if ops is not None:
                    ops.record_loss(ctx, r["seller"], "stockout", int(price[k]), 1, resource=rid)
                continue
            if wallet[j] < price[k]:
                util[j] = -0.1
                continue
            try:
                p = ids.ensure_principal(ctx, i, account=True)
                acct = popapi.account_of(ctx, i)
                ctx.run_as(p, lambda sub: ind.retail_sale(sub, rid, 1, acct, buyer=p))
            except CrucibleError:
                util[j] = -0.1
                continue
            avail[k] -= 1
            last[j] = k
            last_seq[j] = w.seq
            bought_n[j] += 1
            spent[j] += int(price[k])
            purchases.append([i, rid, int(price[k])])
            util[j] = float(np.clip((ub[j] - u0[j]) / 4.0, 0.0, 1.5))
        popapi.record_decisions(ctx, "goods.session", idx, {
            "aware": aware.astype(np.int32), "last": last.astype(np.int32), "bought_n": bought_n.astype(np.int32),
            "spent": spent.astype(np.int64), "stockouts": stockouts.astype(np.int32),
            "last_t": np.full(m, now, np.int64), "last_seq": last_seq.astype(np.int64)}, table=TABLE, model_id=MODEL_ID, model_version="1",
            fidelity_id="civ.goods.choice",
            extra={"t": int(now), "sessions": int(m), "purchases": purchases[:2000], "stockouts": sold_out[:2000],
                   "listings": int(live.sum())})
        return util

    # ------------------------------------------------------------------ daily close of retail sales
    @process("day")
    def pr_day(self, ctx):
        ctx.world.modules["industry"].retail_close_day(ctx, ctx.now // DAY - 1)

    @applier("goods.wom")
    def ap_wom(self, ev):
        pass

    @applier("goods.slot")
    def ap_slot(self, ev):
        self.world.state.insert("goods_slots", ev.payload["id"], {"slot": ev.payload["slot"], "t": ev.t})

    # ------------------------------------------------------------------ the individual's own view
    @endpoint("mine", S.obj(), read_only=True, cost_s=0)
    def ep_mine(self, ctx):
        """Goods you know of and what you bought (members of the population only)."""
        i = ids.index(ctx.actor)
        if i is None or i >= columns.rows(ctx.world, TABLE):
            raise NotFound("you are not a member of the population")
        slots = {s["slot"]: rid for rid, s in ctx.state.scan("goods_slots")}
        aw = [slots[int(s)] for s in columns.col(ctx.world, TABLE, "aware")[i] if int(s) in slots]
        lb = int(columns.col(ctx.world, TABLE, "last")[i])
        return {"known_listings": aw, "last_bought": slots.get(lb), "purchases": int(columns.col(ctx.world, TABLE,
                                                                                                  "bought_n")[i])}

    def check_invariants(self, world):
        errs = []
        n = columns.rows(world, TABLE)
        if n != popapi.n_individuals(world):
            errs.append(f"goods has {n} rows, the population {popapi.n_individuals(world)}")
        L = world.state.count("goods_slots")
        aw = columns.col(world, TABLE, "aware")
        if aw.size and int(aw.max()) >= max(L, 1) and L:
            errs.append("awareness of an unknown listing slot")
        if int(columns.col(world, TABLE, "spent").min(initial=0)) < 0:
            errs.append("negative spending")
        return errs


def purchases_total(world) -> int:
    return int(columns.col(world, TABLE, "bought_n").sum())


def mean_price(world, since_t: int = 0) -> float:
    st = world.state
    ps = [r["price"] for _, r in st.scan("retail") if r["status"] == "active"]
    return float(np.mean(ps)) if ps else math.nan
