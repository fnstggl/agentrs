"""Consumers: a persistent, vectorised background population that creates demand (module `consumers`).

Each simulated consumer has a need (one of the public need formats), a request rate, a willingness to pay,
preference weights over quality / latency / price, loyalty (switching cost), a current provider, satisfaction,
a small awareness set of offerings it knows about, and neighbours in a social graph. Consumers are numpy arrays
inside the authoritative state, so they persist across snapshots and forks and cost no LLM calls.

Every simulated hour (process `consumers.hour`):
1. Requests per offering ~ Poisson(sum of customers' rates), reduced by price elasticity and capped by the
   household budget (acct_households is funded by exogenous income; it cannot go negative).
2. A sample of real requests is *executed* on the offering's deployment (sandbox or trusted runtime) through the
   same metering path as any other call, and each output is scored by the need's verifier. The remaining requests
   are extrapolated from the sample (and consume the matching compute; overload is enforced).
3. Households pay `price x successful requests` to the seller.
4. The population update (applier, deterministic given the event payload): satisfaction from experienced utility,
   churn and switching using only *observable* signals (price, public ratings, word of mouth, ads, search results),
   awareness diffusion over the social graph, turnover.
5. Consequences recorded as world events: reviews on the forum and verified ratings on the market (posted by
   persona accounts), support tickets for failures.

Consumers never see hidden state: they choose from what they know, and they know only what reached them.

Crucible 4.0 (all of it off unless configured; a world without these keys evolves byte for byte as in 3.0):

* `behavior: {"model": name, "params": {...}}` or `segments: [{"name", "share", "model", "params"}]` - the rules
  of step 4 come from `economy/behavior.py` (`BEHAVIOR_MODELS`). Each consumer is assigned to a segment at genesis
  (seeded, array `c_seg`); each segment follows its own model and parameter overrides, including how its members
  are drawn (`rate_per_day`, `wtp_mult`, `loyalty`) and a `budget_share` (an envelope of the household balance).
* `latent: {need: base_active_fraction}` and `derived: [{"src", "dst", "weight", "scale", "lag_days"}]` - each
  consumer has an `active` flag (array `c_active`): an inactive consumer does not have its need for now (no
  requests, no shopping, no provider). Once a day (process `consumers.demand`, one event) the active fraction of
  every need relaxes at `activation_rate` (down: `decay_rate`, default the same) per day toward
  `clip(base + sum_src weight * (1 - exp(-verified_src / scale)), 0, 1)`, where `verified_src` is the verified
  successful volume served in `src` (successful requests x mean verifier score) on the last complete day, `lag_days`
  earlier. Activation creates no income: household income is a constant of the configuration and the household
  balance still caps what every hour can spend.

The edges, the targets and the segment of a consumer are hidden parameters: only the evaluator-side helpers at the
bottom of this file (`demand_state`, `segment_aggregates`) return them; no tool does.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, ProtocolError
from ..core.kernel import Module, applier, endpoint, process, register_module
from .behavior import (BEHAVIOR_MODELS, DEG, K_AWARE, Hour, Pop, budget_envelopes, build_demand,  # noqa: F401
                       build_segments, learn, learn_pairs, segment_counts)
from .needs import NEEDS

DAY = 86400


@register_module
class Consumers(Module):
    name = "consumers"
    host = "research"
    depends = ("identity", "bank", "cloud", "market")

    def install(self, world):
        st = world.state
        st.declare_table("cons_offers", index=("need",))
        st.declare_table("cons_summary")
        st.declare_table("cons_voices")
        self.cfg = {"n": 5000, "needs": None, "rate_per_day": [5.0, 30.0], "wtp_mult": [0.6, 3.0],
                    "k_samples": 4, "search_rate": 0.02, "review_rate": 0.0005, "ticket_rate": 0.01,
                    "wom_rate": 0.01, "turnover_per_day": 0.002, "income_per_day_cents": None, "personas": 12,
                    "shop_threshold": 0.35, "size": [0.3, 1.0], "loyalty": [0.0, 0.9]}
        self.cfg.update(self.config)
        # 4.0: behaviour models per segment (default: the single 3.0 `legacy` segment) and latent / derived demand
        # (default: none). State for them is declared only when configured, so 3.0 worlds keep their digests.
        self.segments = build_segments(self.cfg)
        self.envelopes = budget_envelopes(self.segments)
        self.demand = build_demand(self.cfg)
        self._xarrays: dict[str, tuple] = {}
        for seg in self.segments:
            for name, decl in seg.model.arrays.items():
                if self._xarrays.setdefault(name, decl) != decl:
                    raise ProtocolError(f"behaviour models declare array {name} differently")
        self._members: tuple | None = None
        if self.demand is not None:
            st.declare_table("cons_ok_days")
            st.declare_table("cons_demand")

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        c = self.cfg
        rng = ctx.world.rng_for("consumers-genesis")
        needs = c["needs"] or {n: 1.0 for n in _builtin_needs()}
        names = sorted(needs)
        w = np.array([needs[n] for n in names], dtype=float)
        n = int(c["n"])
        need_idx = rng.choice(len(names), size=n, p=w / w.sum())
        ctx.emit("consumers.genesis", {"n": n, "need_names": names, "seed_key": "consumers-genesis"}, vis="internal")
        idm = ctx.world.modules["identity"]
        for k in range(int(c["personas"])):
            idm.create_person(ctx, f"customer-{k:02d}", f"Customer voice {k}", "npc", {"role": "consumer-voice"})
        bank = ctx.world.modules["bank"]
        income = self.daily_income()
        bank.move(ctx, "acct_external", "acct_households", income, "income", "household income (day 0)")
        ctx.schedule(ctx.now + 3600, "consumers.hour", {}, prio=10, period=3600, key="consumers.hour")
        ctx.schedule(ctx.now + 86400, "consumers.income", {}, prio=5, period=86400, key="consumers.income")
        if self.demand is not None:
            ctx.schedule(ctx.now + DAY, "consumers.demand", {}, prio=20, period=DAY, key="consumers.demand")
        del need_idx

    def daily_income(self) -> int:
        c = self.cfg
        if c.get("income_per_day_cents"):
            return int(c["income_per_day_cents"])
        # default: enough to buy ~70% of nominal demand at reference prices (the budget binds when prices rise).
        # The reference is the built-in need formats, as in 3.0: need categories generated and registered later
        # in the same process (4.0, names `gen.*`) must not change the income of worlds that do not use them. A
        # world whose own needs include generated ones is priced on its own needs.
        avg_rate = sum(c["rate_per_day"]) / 2
        own = sorted(c["needs"] or ())
        ref = own if any(n.startswith("gen.") for n in own) else _builtin_needs()
        avg_price = np.mean([NEEDS[n].base_price_cents for n in ref])
        return int(0.7 * c["n"] * avg_rate * avg_price * 1.5)

    @applier("consumers.genesis")
    def ap_genesis(self, ev):
        c = self.cfg
        st = self.world.state
        n = ev.payload["n"]
        names = ev.payload["need_names"]
        rng = self.world.rng_for(ev.payload["seed_key"], "arrays")
        w = np.array([(c["needs"] or {}).get(x, 1.0) for x in names], dtype=float)
        st.meta_set("cons_needs", names)
        st.set_array("c_need", rng.choice(len(names), size=n, p=w / w.sum()).astype(np.int16))
        st.set_array("c_rate", rng.uniform(*c["rate_per_day"], n).astype(np.float32))
        base = np.array([NEEDS[x].base_price_cents for x in names], dtype=np.float32)
        st.set_array("c_wtp", (base[st.arrays["c_need"]] * rng.uniform(*c["wtp_mult"], n)).astype(np.float32))
        prefs = rng.dirichlet([2.0, 1.0, 1.5], n).astype(np.float32)   # quality, latency, price
        st.set_array("c_pref", prefs)
        st.set_array("c_loyal", rng.uniform(*c["loyalty"], n).astype(np.float32))
        st.set_array("c_prov", np.full(n, -1, np.int32))
        st.set_array("c_sat", np.full(n, 0.5, np.float32))
        st.set_array("c_aware", np.full((n, K_AWARE), -1, np.int32))
        st.set_array("c_since", np.zeros(n, np.int32))
        # small-world social graph: ring lattice with random rewiring
        nbr = (np.arange(n)[:, None] + np.array([-3, -2, -1, 1, 2, 3])[None, :]) % n
        rew = rng.random((n, DEG)) < float(c.get("rewire", 0.15))
        nbr[rew] = rng.integers(0, n, int(rew.sum()))
        st.set_array("c_nbr", nbr.astype(np.int32))
        self._genesis_population(st, n, names, ev.payload["seed_key"])

    def _genesis_population(self, st, n, names, seed_key) -> None:
        """4.0 state, drawn from generators of their own after the 3.0 arrays (which are therefore the same as in
        the 3.0 world of this seed): segment membership, per-segment draws, model traits, the `active` flag."""
        segs = self.segments
        if len(segs) > 1:
            rng = self.world.rng_for(seed_key, "segments")
            order = rng.permutation(n)
            seg = np.zeros(n, np.int8)
            a = 0
            for s, k in zip(segs, segment_counts(segs, n)):
                seg[order[a:a + k]] = s.index
                a += k
            st.set_array("c_seg", seg)
        for name in sorted(self._xarrays):
            dtype, shape, fill, _ = self._xarrays[name]
            st.set_array(name, np.full((n, *shape), fill, dtype))
        base = np.array([NEEDS[x].base_price_cents for x in names], dtype=np.float32)
        for s, pop in self._pops(st, mutable=True):
            rng = self.world.rng_for(seed_key, "segment", s.index, s.model.name)
            m = pop.members()
            if "rate_per_day" in s.draw:
                st.array_mut("c_rate")[m] = rng.uniform(*s.draw["rate_per_day"], len(m)).astype(np.float32)
            if "wtp_mult" in s.draw:
                st.array_mut("c_wtp")[m] = (base[st.arrays["c_need"][m]] *
                                            rng.uniform(*s.draw["wtp_mult"], len(m))).astype(np.float32)
            if "loyalty" in s.draw:
                st.array_mut("c_loyal")[m] = rng.uniform(*s.draw["loyalty"], len(m)).astype(np.float32)
            s.model.init(pop, rng)
        if self.demand is not None:
            rng = self.world.rng_for(seed_key, "latent")
            active = np.ones(n, bool)
            need = st.arrays["c_need"]
            for name in sorted(self.demand.latent):
                if name not in names:
                    continue
                idx = np.flatnonzero(need == names.index(name))
                keep = int(round(self.demand.latent[name] * len(idx)))
                active[rng.permutation(idx)[keep:]] = False
            st.set_array("c_active", active)

    # ------------------------------------------------------------------ segments (4.0)
    def _seg_members(self, st) -> list:
        """Consumer ids of each segment (None for a single segment: the whole population). Membership is fixed at
        genesis, so the index arrays are cached against the identity of the state's `c_seg` array."""
        if len(self.segments) == 1:
            return [None]
        seg = st.arrays["c_seg"]
        if self._members is None or self._members[0] is not seg:
            self._members = (seg, [np.flatnonzero(seg == s.index) for s in self.segments])
        return self._members[1]

    def _pops(self, st, mutable: bool) -> list:
        """[(segment, Pop)]: every segment's view of the population arrays (writable inside an applier)."""
        a = st.arrays
        if "c_prov" not in a:
            return []
        own = st.array_mut if mutable else st.array
        arrays = {"prov": own("c_prov"), "sat": own("c_sat"), "aware": own("c_aware"), "since": own("c_since"),
                  "need": a["c_need"], "pref": a["c_pref"], "wtp": a["c_wtp"], "loyal": a["c_loyal"],
                  "nbr": a["c_nbr"]}
        if "c_active" in a:
            arrays["active"] = a["c_active"]
        x = {name: (st.array_mut(name) if mutable and decl[3] else a[name]) for name, decl in self._xarrays.items()}
        n = len(a["c_prov"])
        return [(s, Pop(n, m, s.index, s.name, s.params, arrays, x))
                for s, m in zip(self.segments, self._seg_members(st))]

    # ------------------------------------------------------------------ offerings registry
    def offerings(self, st) -> dict[int, dict]:
        """slot -> offering (active api listings serving a known need)."""
        out = {}
        for lid, o in st.scan("cons_offers"):
            l = st.get("listings", lid)
            if l is None or l["status"] != "active":
                continue
            out[o["slot"]] = {"listing": lid, "need": o["need"], "host": l["host"], "endpoint": l["endpoint"],
                              "price": l["price"], "seller": l["seller"],
                              "rating": (l["rating_sum"] / l["rating_n"]) if l["rating_n"] else None,
                              "n_ratings": l["rating_n"]}
        return out

    def _register_new_offers(self, ctx):
        st = ctx.state
        names = st.meta_get("cons_needs")
        for lid in st.find_ids("listings", "kind", "api"):
            l = st.tables["listings"][lid]
            if l["need"] and l["need"] in names and not st.has("cons_offers", lid):
                slot = st.count("cons_offers")
                ctx.emit("consumers.offer", {"id": lid, "slot": slot, "need": l["need"]}, vis="internal")

    @applier("consumers.offer")
    def ap_offer(self, ev):
        p = ev.payload
        self.world.state.insert("cons_offers", p["id"], {"slot": p["slot"], "need": p["need"], "t": ev.t})

    # ------------------------------------------------------------------ the hourly market
    @process("income")
    def pr_income(self, ctx):
        ctx.world.modules["bank"].move(ctx, "acct_external", "acct_households", self.daily_income(), "income",
                                       "household income")

    @process("hour")
    def pr_hour(self, ctx):
        st = ctx.state
        world = ctx.world
        self._register_new_offers(ctx)
        offers = self.offerings(st)
        names = st.meta_get("cons_needs")
        cloud = world.modules["cloud"]
        bank = world.modules["bank"]
        rng = ctx.rng("demand")
        prov = st.array("c_prov")
        rate = st.array("c_rate")
        wtp = st.array("c_wtp")
        hh = bank.balance(st, "acct_households")
        # what each spending envelope may still spend this hour: one pool (3.0) unless segments set budget shares
        if self.envelopes is None:
            cenv, pools = None, [hh]
        else:
            pools = [hh * ppm // 1_000_000 for ppm in self.envelopes[1]]
            cenv = np.asarray(self.envelopes[0], np.int16)[st.array("c_seg")] if len(pools) > 1 else None
        pools0 = list(pools)
        k_samples = int(self.cfg["k_samples"])
        amp = float(self.cfg.get("diurnal", 0.0))
        diurnal = 1.0 + amp * math.cos(2 * math.pi * ((ctx.now // 3600) % 24 - float(self.cfg.get("peak_hour", 14))) / 24.0) \
            if amp else 1.0
        ops = world.hooks.get("ops")
        slots = {}
        for slot in sorted(offers):
            o = offers[slot]
            cust = prov == slot
            ncust = int(cust.sum())
            if ncust == 0:
                continue
            price = max(1, o["price"])
            R, parts = 0, []
            for e in range(len(pools)):
                ce = cust if cenv is None else cust & (cenv == e)
                elastic = np.clip(2.0 - price / np.maximum(wtp[ce], 0.01), 0.0, 1.0)
                lam = float((rate[ce] * elastic).sum() / 24.0) * diurnal
                r_e = int(rng.poisson(lam)) if lam > 0 else 0
                r_e = min(r_e, int(pools[e] // price))
                parts.append(r_e)
                R += r_e
            summ = {"slot": slot, "requests": R, "ok": 0, "errors": 0, "unavailable": 0, "overload": 0,
                    "q": 0.0, "cpu_ms": 0.0, "revenue": 0, "customers": ncust}
            if self.envelopes is not None:
                summ["parts"] = parts             # requests placed by each spending envelope (audit trail)
            if R > 0:
                self._serve(ctx, o, R, k_samples, rng, summ, cloud)
                # 3.0: every answered request is paid. With `accept_floor`, customers pay only for answers whose
                # verified score reaches the floor; the rest are refused (a measured loss of the seller).
                cents = summ.get("accepted", summ["ok"]) * price
                if cents > 0:
                    dst = bank.primary_account(st, o["seller"])
                    if dst is not None:
                        bank.move(ctx, "acct_households", dst, cents, "sale",
                                  f"{summ.get('accepted', summ['ok'])} requests to {o['host']}", ref=f"svc:{o['host']}")
                        _charge(pools, parts, R, cents, price)
                        summ["revenue"] = cents
                        dep = cloud.live_deployment(st, o["host"])
                        if dep is not None:
                            cloud.record_revenue(ctx, st.get("hosts", o["host"])["deployment"], cents)
            slots[str(slot)] = summ
            if ops is not None and R > 0:     # demand that arrived but could not be served is a measured loss
                if summ["overload"]:
                    ops.record_loss(ctx, o["seller"], "capacity_overload", summ["overload"] * price, summ["overload"],
                                    resource=o["host"])
                if summ["unavailable"]:
                    ops.record_loss(ctx, o["seller"], "service_unavailable", summ["unavailable"] * price,
                                    summ["unavailable"], resource=o["host"])
                refused = summ["ok"] - summ.get("accepted", summ["ok"])
                if refused > 0:
                    ops.record_loss(ctx, o["seller"], "result_refused", refused * price, refused, resource=o["host"])
        # who searches, and what they see (search engine and sponsored results are real world mechanisms)
        search = world.modules.get("search")
        seen: dict[str, list[int]] = {}
        sponsored: dict[str, list[int]] = {}
        ads = world.modules.get("ads")
        pops = self._pops(st, mutable=False)
        for ni, need in enumerate(names):
            n_search = self._n_searchers(pops, ni)
            if n_search == 0 or search is None:
                continue
            q = " ".join(NEEDS[need].keywords[:3])
            sp_res, org_res, _ = search.search(world, q, None, 10)
            res = sp_res + org_res
            lid_to_slot = {o["listing"]: s for s, o in offers.items() if o["need"] == need}
            host_to_slot = {o["host"]: s for s, o in offers.items() if o["need"] == need}

            def slot_of(r):
                # listings map directly; posts/docs/pages that mention an offering's host lead to it too
                if r.get("ref") in lid_to_slot:
                    return lid_to_slot[r["ref"]]
                for h in r.get("hosts") or ():
                    if h in host_to_slot:
                        return host_to_slot[h]
                return None
            organic = []
            for r in res:
                s_ = slot_of(r)
                if s_ is not None and not r.get("sponsored") and s_ not in organic:
                    organic.append(s_)
            seen[str(ni)] = organic[:4]
            sp = [r for r in res if r.get("sponsored") and slot_of(r) is not None]
            sponsored[str(ni)] = [slot_of(r) for r in sp][:2]
            if sp and ads is not None:
                camp = [r["campaign"] for r in sp if r.get("campaign")]
                if camp:
                    ads.record_impressions(ctx, camp, n_search)
        public = {str(s): {"price": o["price"], "rating": o["rating"], "n": o["n_ratings"], "need": names.index(o["need"])}
                  for s, o in offers.items()}
        payload = {"t": ctx.now, "slots": slots, "seen": seen, "sponsored": sponsored, "public": public}
        if self.envelopes is not None:
            payload["pools"] = pools0             # what each envelope could spend this hour
        ctx.emit("consumers.hour", payload, vis="internal")
        self._voices(ctx, offers)

    def _n_searchers(self, pops, ni) -> int:
        """Consumers of need `ni` who search this hour (the applier samples exactly this many per segment)."""
        tot = 0
        for seg, pop in pops:
            if seg.model.uses_search:
                tot += int(round(float(seg.model.searchers(pop, ni).sum()) * pop.params["search_rate"]))
        return tot

    def _serve(self, ctx, o, R, k, rng, summ, cloud):
        st = ctx.state
        need = NEEDS[o["need"]]
        dep = cloud.live_deployment(st, o["host"])
        if dep is None or o["endpoint"] not in dep["manifest"]["endpoints"]:
            summ["unavailable"] = R
            return
        dep_id = st.get("hosts", o["host"])["deployment"]
        why = cloud.availability(st, dep)
        if why:
            summ["unavailable"] = R
            cloud._usage(ctx, dep_id, 0, 0.0, 0.0, "unavailable", 0)
            return
        n = min(R, k)
        qs, cpus, ok, err, over = [], [], 0, 0, 0
        for _ in range(n):
            inp, hidden = need.generate(rng, float(rng.uniform(*self.cfg["size"])))
            res = cloud.execute(ctx, dep_id, o["endpoint"], inp, "households", need.work_scale,
                                post=lambda body, inp=inp, hidden=hidden: {"q": float(need.score(inp, hidden, body))})
            if res["status"] == 200:
                q = float((res["body"] or {}).get("q", 0.0)) if isinstance(res["body"], dict) else 0.0
                qs.append(q)
                cpus.append(res.get("cpu_ms", 0.0))
                ok += 1
            elif res["status"] == 503:
                over += 1
            else:
                err += 1
                cpus.append(res.get("cpu_ms", 0.0))
        rest = R - n
        if rest > 0:
            frac_ok = ok / n if n else 0.0
            frac_err = err / n if n else 0.0
            mean_cu = (float(np.mean(cpus)) if cpus else 0.0) * need.work_scale
            inst = st.get("instances", dep["instance"])
            hour = ctx.now // 3600
            used = inst["used_cu"] if inst["hour"] == hour else 0.0
            cap = inst["cores"] * 100_000 - used
            served = rest if mean_cu <= 0 else int(min(rest, max(0.0, cap) // max(mean_cu, 1e-9)))
            r_ok = int(round(served * frac_ok))
            r_err = int(round(served * frac_err))
            r_over = rest - served + (served - r_ok - r_err if (frac_ok + frac_err) < 1 else 0)
            ctx.emit("cloud.usage_bulk", {"dep": dep_id, "ok": r_ok, "errors": r_err, "unavailable": 0,
                                          "overload": max(0, r_over), "cu": round(served * mean_cu, 3)}, vis="internal")
            ok, err, over = ok + r_ok, err + r_err, over + max(0, r_over)
        summ["ok"], summ["errors"], summ["overload"] = ok, err, over
        floor = self.cfg.get("accept_floor")
        if floor is not None:             # answers the customers accept: the sampled acceptance rate, extrapolated
            share = (sum(1 for q in qs if q >= float(floor)) / len(qs)) if qs else 0.0
            summ["accepted"] = int(round(ok * share))
        summ["q"] = float(np.mean(qs)) if qs else 0.0
        summ["cpu_ms"] = float(np.mean(cpus)) if cpus else 0.0

    # ------------------------------------------------------------------ population dynamics (deterministic applier)
    @applier("consumers.hour")
    def ap_hour(self, ev):
        """The hourly population update. The steps a rule set can differ in (experience, awareness, choice, voice)
        are delegated to each segment's behaviour model (economy/behavior.py); what is physical (an offering that
        closed is lost, people leave the market) is common. With the default single `legacy` segment this performs
        the same operations and the same random draws, in the same order, as Crucible 3.0."""
        st = self.world.state
        p = ev.payload
        hour = p["t"] // 3600
        rng = self.world.rng_for("consumers-pop", hour)
        n = len(st.arrays["c_prov"])
        pops = self._pops(st, mutable=True)
        prov = st.array_mut("c_prov")
        sat = st.array_mut("c_sat")
        aware = st.array_mut("c_aware")
        min_slots = max(int(prov.max(initial=-1)) + 1, int(aware.max(initial=-1)) + 1)
        for seg in self.segments:
            for name in seg.model.slot_arrays:
                min_slots = max(min_slots, int(st.arrays[name].max(initial=-1)) + 1)
        mk = Hour(p, min_slots, len(st.meta_get("cons_needs")))
        n_slots = mk.n_slots
        # 1. experienced utility for current customers
        has = prov >= 0
        ps = np.where(has, prov, n_slots)
        for seg, pop in pops:
            seg.model.experience(pop, mk, rng)
        # providers that disappeared: customers lose them
        gone = has & ~mk.live[ps]
        prov[gone] = -1
        sat[gone] = 0.3
        # 2. awareness (search results, sponsored results, word of mouth, neighbours: whatever the model listens to)
        for seg, pop in pops:
            seg.model.awareness(pop, mk, rng)
        # 3. shopping: first trials, switching, abandonment
        for seg, pop in pops:
            seg.model.choose(pop, mk, rng)
        # 4. turnover: some consumers leave the market, replaced by newcomers with fresh preferences
        churn = np.flatnonzero(rng.random(n) < self.cfg["turnover_per_day"] / 24.0)
        if len(churn):
            prov[churn] = -1
            sat[churn] = 0.5
            aware[churn] = -1
            if self._xarrays:
                seg_of = st.arrays.get("c_seg")
                for seg, pop in pops:
                    seg.model.forget(pop, churn if seg_of is None else churn[seg_of[churn] == seg.index])
        # 5. summary + voices (reviews, ratings, tickets) for the handler to turn into world events
        counts = np.bincount(np.where(prov >= 0, prov, n_slots), minlength=n_slots + 1)
        voices = []
        seg_of = st.arrays.get("c_seg") if len(pops) > 1 else None
        for k, sm in sorted(p["slots"].items(), key=lambda kv: int(kv[0])):
            s = int(k)
            cust = np.flatnonzero(prov == s)
            st.put("cons_summary", k, {"slot": s, "customers": int(counts[s]), "t": p["t"],
                                      "mean_sat": float(sat[cust].mean()) if len(cust) else None,
                                      "requests": sm["requests"], "ok": sm["ok"], "q": sm["q"]})
            if not len(cust):
                continue
            for seg, pop in pops:
                mine = cust if seg_of is None else cust[seg_of[cust] == seg.index]
                if len(mine):
                    seg.model.voice(pop, mk, rng, s, mine, sm, voices)
        for j, v in enumerate(voices):
            st.put("cons_voices", f"{hour}|{j:04d}", v)
        st.meta_set("cons_hour", hour)
        if self.demand is not None:
            self._record_volume(st, p, mk)

    # ------------------------------------------------------------------ latent needs and derived demand (4.0)
    def _record_volume(self, st, p, mk) -> None:
        """Add this hour's served volume to the per-need totals of the day. `verified` is the verified successful
        volume: requests that succeeded times the mean verifier score of the executed sample (a service that
        answers every request with garbage has served nothing)."""
        names = st.meta_get("cons_needs")
        day = p["t"] // DAY
        add: dict[str, list] = {}
        for k, sm in sorted(p["slots"].items(), key=lambda kv: int(kv[0])):
            ni = int(mk.slot_need[int(k)])
            if 0 <= ni < len(names) and sm["ok"]:
                rec = add.setdefault(names[ni], [0, 0.0])
                rec[0] += int(sm["ok"])
                rec[1] += float(sm["ok"]) * float(sm["q"])
        for need in sorted(add):
            key = f"{need}|{day}"
            old = st.tables["cons_ok_days"].get(key) or {"need": need, "day": day, "ok": 0, "verified": 0.0}
            st.put("cons_ok_days", key, {"need": need, "day": day, "ok": old["ok"] + add[need][0],
                                         "verified": round(old["verified"] + add[need][1], 4)})

    @process("demand")
    def pr_demand(self, ctx):
        """Once a day: read the verified volume each derived-demand edge listens to (the last complete day,
        `lag_days` earlier) and emit one event; its applier moves the active fractions."""
        st = ctx.state
        day = ctx.now // DAY
        inputs = []
        for e in self.demand.edges:
            d = day - 1 - e["lag_days"]
            rec = st.get("cons_ok_days", f"{e['src']}|{d}")
            inputs.append({"src": e["src"], "dst": e["dst"], "day": d, "ok": rec["ok"] if rec else 0,
                           "verified": rec["verified"] if rec else 0.0})
        ctx.emit("consumers.demand", {"day": day, "inputs": inputs}, vis="internal")

    @applier("consumers.demand")
    def ap_demand(self, ev):
        st = self.world.state
        dm = self.demand
        p = ev.payload
        if "c_active" not in st.arrays:
            return
        names = st.meta_get("cons_needs")
        targets = dm.targets([i["verified"] for i in p["inputs"]])
        rng = self.world.rng_for("consumers-demand", p["day"])
        need = st.arrays["c_need"]
        active = st.array_mut("c_active")
        prov = st.array_mut("c_prov")
        sat = st.array_mut("c_sat")
        for ni, name in enumerate(names):
            tg = targets.get(name) or {"base": 1.0, "lift": 0.0, "target": 1.0}
            members = np.flatnonzero(need == ni)
            is_on = active[members]
            n_on = int(is_on.sum())
            gap = tg["target"] * len(members) - n_on
            move = int(round((dm.activation_rate if gap > 0 else dm.decay_rate) * gap))
            if move == 0 and abs(gap) >= 1:       # small populations still converge: at least one consumer a day
                move = 1 if gap > 0 else -1
            pool = members[~is_on] if move > 0 else members[is_on]
            if len(pool) == 0:
                move = 0
            if move > 0:
                pick = rng.choice(pool, size=min(move, len(pool)), replace=False)
                active[pick] = True
                n_on += len(pick)
            elif move < 0:
                pick = rng.choice(pool, size=min(-move, len(pool)), replace=False)
                active[pick] = False              # the need is gone for now: no provider, no requests, no shopping
                prov[pick] = -1
                sat[pick] = 0.5
                n_on -= len(pick)
            if name in targets or st.has("cons_demand", name):
                st.put("cons_demand", name, {"day": p["day"], "n": int(len(members)), "active": n_on,
                                             "base": tg["base"], "lift": round(tg["lift"], 6),
                                             "target": round(tg["target"], 6),
                                             "inputs": [dict(i) for i in p["inputs"] if i["dst"] == name]})
        horizon = p["day"] - dm.max_lag - 3
        for key in [k for k, r in st.scan("cons_ok_days") if r["day"] < horizon]:
            st.delete("cons_ok_days", key)

    @applier("consumers.seed")
    def ap_seed(self, ev):
        """An evaluator intervention (`seed_adopters`): the listed consumers start using an offering now."""
        st = self.world.state
        p = ev.payload
        who = np.asarray(p["who"], np.int64)
        prov = st.array_mut("c_prov")
        prov[who] = p["slot"]
        st.array_mut("c_sat")[who] = 0.5
        st.array_mut("c_since")[who] = ev.t // 3600
        learn(st.array_mut("c_aware"), who, int(p["slot"]))

    def _voices(self, ctx, offers):
        st = ctx.state
        hour = st.meta_get("cons_hour")
        prefix = f"{hour}|"
        keys = [k for k in st.ids("cons_voices") if k.startswith(prefix)]
        market = ctx.world.modules.get("market")
        forum = ctx.world.modules.get("forum")
        support = ctx.world.modules.get("support")
        by_slot = {o_slot: o for o_slot, o in offers.items()}
        for key in keys:
            v = st.tables["cons_voices"][key]
            o = by_slot.get(v["slot"])
            if o is None:
                continue
            persona = f"p:customer-{v['persona']:02d}"
            if v["kind"] == "review":
                rid = f"{o['listing']}|{persona}|{hour // 24}"
                if market is not None and not st.has("ratings", rid):
                    market.add_rating(ctx, rid, o["listing"], persona, v["stars"],
                                      f"verified customer review of {o['host']}.crucible")
                if forum is not None and hasattr(forum, "post"):
                    mood = {1: "terrible", 2: "disappointing", 3: "okay", 4: "good", 5: "excellent"}[v["stars"]]
                    try:
                        forum.post(ctx, persona, "market", f"{o['host']}.crucible was {mood}",
                                   f"My experience with {o['host']}.crucible for {o['need']}: {v['stars']}/5.",
                                   tags=("review", o["need"]), hosts=(o["host"],))
                    except Exception:  # noqa: BLE001 - forum may reject (rate limits); reviews still stand
                        pass
            elif v["kind"] == "ticket" and support is not None and hasattr(support, "open_ticket"):
                support.open_ticket(ctx, o["host"], o["seller"], v["customer"], f"{o['host']} failing",
                                    f"About {int(v['fail'] * 100)}% of my requests failed this hour.", severity=2)
        if keys:
            ctx.emit("consumers.voices_done", {"keys": keys}, vis="internal")

    @applier("consumers.voices_done")
    def ap_voices_done(self, ev):
        for k in ev.payload["keys"]:
            self.world.state.delete("cons_voices", k)

    # ------------------------------------------------------------------ paid market research (noisy)
    @endpoint("survey", S.obj({"need": S.s_str(maxLength=40), "account": S.s_str(maxLength=40)}, ["need", "account"]),
              cost_s=600)
    def ep_survey(self, ctx, need, account):
        """Commission a market survey ($250): noisy estimates of demand for a need (how many consumers currently
        have it), willingness to pay, how many customers each known offering has and their satisfaction.
        Estimates are sampled, not exact."""
        st = ctx.state
        names = st.meta_get("cons_needs")
        if need not in names:
            raise InvalidAction("unknown need")
        bank = ctx.world.modules["bank"]
        owner = bank.account_owner(st, account)
        ctx.world.modules["identity"].authorize(ctx, owner, "bank.spend", amount=25_000, scope=account)
        bank.move(ctx, account, "acct_fees", 25_000, "fee", f"market survey {need}")
        rng = ctx.rng("survey")
        ni = names.index(need)
        mask = st.array("c_need") == ni
        if "c_active" in st.arrays:           # latent needs: only consumers who have the need right now are found
            mask = mask & st.array("c_active")
        sample = np.flatnonzero(mask)
        if len(sample) == 0:
            return {"need": need, "consumers_est": 0}
        pick = rng.choice(sample, size=min(200, len(sample)), replace=False)
        prov = st.array("c_prov")[pick]
        offers = self.offerings(st)
        shares = {}
        for s in np.unique(prov):
            if s >= 0 and int(s) in offers:
                o = offers[int(s)]
                sel = prov == s
                shares[o["host"]] = {"share_est": round(float(sel.mean()), 3),
                                     "satisfaction_est": round(float(st.array("c_sat")[pick][sel].mean()), 2)}
        return {"need": need, "consumers_est": int(mask.sum() * rng.uniform(0.85, 1.15)),
                "requests_per_day_est": round(float(st.array("c_rate")[pick].mean() * rng.uniform(0.9, 1.1)), 1),
                "median_willingness_to_pay_cents": round(float(np.median(st.array("c_wtp")[pick])), 2),
                "unserved_share_est": round(float((prov < 0).mean()), 3), "offerings": shares,
                "sample_size": int(len(pick))}

    # ------------------------------------------------------------------ analytics helpers (evaluator / dashboards)
    def market_shares(self, world, need: str | None = None) -> dict[str, int]:
        st = world.state
        offers = self.offerings(st)
        prov = st.arrays["c_prov"]
        out: dict[str, int] = {}
        for s, o in offers.items():
            if need and o["need"] != need:
                continue
            out[o["host"]] = out.get(o["host"], 0) + int((prov == s).sum())
        return out

    def check_invariants(self, world):
        st = world.state
        errs = []
        n = len(st.arrays.get("c_prov", []))
        if n:
            for a in ("c_need", "c_rate", "c_wtp", "c_sat", "c_loyal", "c_since"):
                if len(st.arrays[a]) != n:
                    errs.append(f"array {a} has wrong length")
            sat = st.arrays["c_sat"]
            if not np.all(np.isfinite(sat)) or sat.min() < -1e-6 or sat.max() > 1 + 1e-6:
                errs.append("satisfaction out of [0,1]")
            if st.arrays["c_prov"].max() >= max(1, st.count("cons_offers")):
                errs.append("provider index out of range")
            if len(self.segments) > 1:
                seg = st.arrays.get("c_seg")
                if seg is None or len(seg) != n or seg.min() < 0 or seg.max() >= len(self.segments):
                    errs.append("segment membership missing or out of range")
            for name, (dtype, shape, _, _) in self._xarrays.items():
                arr = st.arrays.get(name)
                if arr is None or arr.shape != (n, *shape) or arr.dtype != dtype:
                    errs.append(f"array {name} missing or malformed")
                elif not np.issubdtype(arr.dtype, np.integer) and not np.all(np.isfinite(arr)):
                    errs.append(f"array {name} has non-finite values")
            for seg in self.segments:
                for name in seg.model.slot_arrays:
                    if name in st.arrays and st.arrays[name].max(initial=-1) >= max(1, st.count("cons_offers")):
                        errs.append(f"{name} refers to an unknown offering")
            if self.demand is not None:
                act = st.arrays.get("c_active")
                if act is None or len(act) != n or act.dtype != bool:
                    errs.append("active flags missing or malformed")
                elif (st.arrays["c_prov"][~act] >= 0).any():
                    errs.append("a consumer without the need holds a provider")
        if st.tables["accounts"]["acct_households"]["balance"] < 0:
            errs.append("households overspent")
        return errs


def need_index(world, need: str) -> int:
    return world.state.meta_get("cons_needs").index(need)


def _builtin_needs() -> list[str]:
    """The need formats of the 3.0 registry, sorted (generated 4.0 categories are named `gen.<family>.<hash>`)."""
    return [n for n in sorted(NEEDS) if not n.startswith("gen.")]


def _charge(pools: list, parts: list, R: int, cents: int, price: int) -> None:
    """Take `cents` actually paid for one offering-hour out of the spending envelopes, in proportion to the
    requests each envelope placed (`parts`, summing to R). With one envelope this is `pool -= cents` (3.0). No
    envelope is charged more than it reserved (`parts[e] * price`), so none goes negative."""
    if len(pools) == 1:
        pools[0] -= cents
        return
    share = [cents * r // R for r in parts]
    left = cents - sum(share)
    for e, r in enumerate(parts):
        extra = min(left, r * price - share[e])
        share[e] += extra
        left -= extra
    for e, c in enumerate(share):
        pools[e] -= c


# ====================================================================== evaluator-side helpers (never agent tools)
def _slot_of(world, host_or_listing: str) -> int:
    st = world.state
    for lid, o in st.scan("cons_offers"):
        l = st.get("listings", lid)
        if lid == host_or_listing or (l is not None and l["host"] == host_or_listing and l["status"] == "active"):
            return int(o["slot"])
    raise KeyError(f"no consumer-facing offering {host_or_listing!r}")


def segment_aggregates(world, need: str | None = None) -> list[dict]:
    """Per-segment aggregates for the evaluator: size, how many currently have a need, how many are served, by
    which host, mean satisfaction of the served, and the nominal demand (requests per day) of the served.
    `need` restricts every number to consumers of one need. Segment membership is hidden from agents."""
    st = world.state
    cons = world.modules["consumers"]
    if "c_prov" not in st.arrays:
        return []
    prov, sat, rate, c_need = st.arrays["c_prov"], st.arrays["c_sat"], st.arrays["c_rate"], st.arrays["c_need"]
    active = st.arrays.get("c_active")
    slot_host = {s: o["host"] for s, o in cons.offerings(st).items()}
    names = st.meta_get("cons_needs")
    out = []
    for seg, m in zip(cons.segments, cons._seg_members(st)):
        idx = np.arange(len(prov)) if m is None else m
        if need is not None:
            idx = idx[c_need[idx] == names.index(need)] if need in names else idx[:0]
        p = prov[idx]
        served = p >= 0
        customers: dict[str, int] = {}
        for s, c in zip(*np.unique(p[served], return_counts=True)):
            h = slot_host.get(int(s))
            if h is not None:
                customers[h] = customers.get(h, 0) + int(c)
        n_active = int(active[idx].sum()) if active is not None else int(len(idx))
        out.append({"segment": seg.name, "model": seg.model.name, "share": round(seg.share, 6), "n": int(len(idx)),
                    "active": n_active, "served": int(served.sum()),
                    "served_share": round(float(served.sum()) / n_active, 4) if n_active else 0.0,
                    "mean_satisfaction": round(float(sat[idx][served].mean()), 4) if served.any() else None,
                    "requests_per_day": round(float(rate[idx][served].sum()), 1),
                    "customers": dict(sorted(customers.items()))})
    return out


def demand_state(world) -> dict:
    """Latent / derived demand as the evaluator sees it: for every need, how many consumers there are, how many
    currently have the need, the base and target active fraction, and the verified volumes (per edge: source
    need, the day it was measured, successful and verified successful requests) that produced the target at the
    last daily update. `targets_now` recomputes the targets from the volumes the *next* update will read.
    No agent-callable tool returns any of this."""
    st = world.state
    cons = world.modules["consumers"]
    dm = cons.demand
    if dm is None or "c_active" not in st.arrays:
        return {"enabled": False, "needs": {}, "edges": []}
    names = st.meta_get("cons_needs")
    need, active = st.arrays["c_need"], st.arrays["c_active"]
    day = world.now // DAY
    pending = []
    for e in dm.edges:
        rec = st.get("cons_ok_days", f"{e['src']}|{day - e['lag_days']}")
        pending.append(rec["verified"] if rec else 0.0)
    now = dm.targets(pending)
    out = {}
    for ni, name in enumerate(names):
        mem = need == ni
        n = int(mem.sum())
        n_on = int(active[mem].sum())
        rec = st.get("cons_demand", name)
        out[name] = {"n": n, "active": n_on, "active_fraction": round(n_on / n, 6) if n else 0.0,
                     "base": dm.base(name), "target": rec["target"] if rec else dm.base(name),
                     "lift": rec["lift"] if rec else 0.0, "updated_day": rec["day"] if rec else None,
                     "inputs": [dict(i) for i in rec["inputs"]] if rec else [],
                     "target_now": round(now[name]["target"], 6) if name in now else 1.0}
    return {"enabled": True, "day": day, "activation_rate": dm.activation_rate, "decay_rate": dm.decay_rate,
            "edges": [dict(e) for e in dm.edges], "needs": out,
            "volumes": {k: dict(r) for k, r in st.scan("cons_ok_days")}}


def seed_adopters(world, host: str, who) -> int:
    """Evaluator intervention: make consumers `who` (ids) customers of the offering at `host` (or listing id) now.
    Only consumers who have the offering's need (and currently have a need at all) are moved; returns how many.
    Recorded as one internal `consumers.seed` event, so the intervened world replays exactly (`World.replay`).
    Like every direct intervention it is not a command, so `World.regenerate` does not re-execute it: use it on
    forks, as the counterfactual tools do."""
    st = world.state
    slot = _slot_of(world, host)
    names = st.meta_get("cons_needs")
    o_need = next(o["need"] for _, o in st.scan("cons_offers") if o["slot"] == slot)
    who = np.unique(np.asarray(list(who), np.int64))
    who = who[(who >= 0) & (who < len(st.arrays["c_prov"]))]
    ok = st.arrays["c_need"][who] == names.index(o_need)
    if "c_active" in st.arrays:
        ok &= st.arrays["c_active"][who]
    ids = [int(i) for i in who[ok]]
    if ids:
        world._run_system("consumers.seed", lambda ctx: ctx.emit("consumers.seed", {"slot": slot, "who": ids},
                                                                 vis="internal"))
    return len(ids)
