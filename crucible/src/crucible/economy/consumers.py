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
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction
from ..core.kernel import Module, applier, endpoint, process, register_module
from .needs import NEEDS

K_AWARE = 6
DEG = 6


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

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        c = self.cfg
        rng = ctx.world.rng_for("consumers-genesis")
        needs = c["needs"] or {n: 1.0 for n in sorted(NEEDS)}
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
        del need_idx

    def daily_income(self) -> int:
        c = self.cfg
        if c.get("income_per_day_cents"):
            return int(c["income_per_day_cents"])
        # default: enough to buy ~70% of nominal demand at reference prices (the budget binds when prices rise)
        avg_rate = sum(c["rate_per_day"]) / 2
        avg_price = np.mean([NEEDS[n].base_price_cents for n in sorted(NEEDS)])
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
            elastic = np.clip(2.0 - price / np.maximum(wtp[cust], 0.01), 0.0, 1.0)
            lam = float((rate[cust] * elastic).sum() / 24.0) * diurnal
            R = int(rng.poisson(lam)) if lam > 0 else 0
            R = min(R, int(hh // price))
            summ = {"slot": slot, "requests": R, "ok": 0, "errors": 0, "unavailable": 0, "overload": 0,
                    "q": 0.0, "cpu_ms": 0.0, "revenue": 0, "customers": ncust}
            if R > 0:
                self._serve(ctx, o, R, k_samples, rng, summ, cloud)
                cents = summ["ok"] * price
                if cents > 0:
                    dst = bank.primary_account(st, o["seller"])
                    if dst is not None:
                        bank.move(ctx, "acct_households", dst, cents, "sale", f"{summ['ok']} requests to {o['host']}",
                                  ref=f"svc:{o['host']}")
                        hh -= cents
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
        # who searches, and what they see (search engine and sponsored results are real world mechanisms)
        search = world.modules.get("search")
        seen: dict[str, list[int]] = {}
        sponsored: dict[str, list[int]] = {}
        need_arr = st.array("c_need")
        sat = st.array("c_sat")
        ads = world.modules.get("ads")
        for ni, need in enumerate(names):
            n_search = self._n_searchers(need_arr, prov, sat, ni)
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
        ctx.emit("consumers.hour", {"t": ctx.now, "slots": slots, "seen": seen, "sponsored": sponsored,
                                    "public": public}, vis="internal")
        self._voices(ctx, offers)

    def _n_searchers(self, need_arr, prov, sat, ni) -> int:
        mask = (need_arr == ni) & ((prov < 0) | (sat < self.cfg["shop_threshold"]))
        return int(round(float(mask.sum()) * self.cfg["search_rate"]))

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
        summ["q"] = float(np.mean(qs)) if qs else 0.0
        summ["cpu_ms"] = float(np.mean(cpus)) if cpus else 0.0

    # ------------------------------------------------------------------ population dynamics (deterministic applier)
    @applier("consumers.hour")
    def ap_hour(self, ev):
        st = self.world.state
        p = ev.payload
        hour = p["t"] // 3600
        rng = self.world.rng_for("consumers-pop", hour)
        n = len(st.arrays["c_prov"])
        prov = st.array_mut("c_prov")
        sat = st.array_mut("c_sat")
        aware = st.array_mut("c_aware")
        since = st.array_mut("c_since")
        need = st.arrays["c_need"]
        pref = st.arrays["c_pref"]
        wtp = st.arrays["c_wtp"]
        loyal = st.arrays["c_loyal"]
        nbr = st.arrays["c_nbr"]
        public = {int(k): v for k, v in p["public"].items()}
        n_slots = max([s for s in public] + [int(k) for k in p["slots"]] + [-1]) + 1
        # slots that closed still appear in consumers' current provider and awareness sets: size the per-slot
        # arrays to cover them (an offering closing at the top of the slot range used to crash every hour)
        n_slots = max(n_slots, int(prov.max(initial=-1)) + 1, int(aware.max(initial=-1)) + 1)
        price = np.zeros(n_slots + 1, np.float32)
        rating = np.full(n_slots + 1, 3.0, np.float32)
        conf = np.zeros(n_slots + 1, np.float32)
        live = np.zeros(n_slots + 1, bool)
        for s, v in public.items():
            price[s] = v["price"]
            if v["rating"] is not None:
                rating[s] = v["rating"]
                conf[s] = v["n"] / (v["n"] + 5.0)
            live[s] = True
        # 1. experienced utility for current customers
        q_eff = np.zeros(n_slots + 1, np.float32)
        lat = np.zeros(n_slots + 1, np.float32)
        served = np.zeros(n_slots + 1, bool)
        for k, sm in p["slots"].items():
            s = int(k)
            tot = sm["ok"] + sm["errors"] + sm["unavailable"] + sm["overload"]
            if tot == 0:
                continue
            served[s] = True
            q_eff[s] = sm["q"] * sm["ok"] / tot
            lat[s] = min(1.0, sm["cpu_ms"] / 50.0)
        has = prov >= 0
        ps = np.where(has, prov, n_slots)
        exp = has & served[ps]
        u = pref[:, 0] * q_eff[ps] - pref[:, 1] * lat[ps] - pref[:, 2] * np.minimum(2.0, price[ps] / np.maximum(wtp, 0.01)) * 0.5
        u = u + rng.normal(0, 0.03, n).astype(np.float32)
        sat[exp] = 0.85 * sat[exp] + 0.15 * np.clip(u[exp] * 1.6 + 0.2, 0, 1)
        # providers that disappeared: customers lose them
        gone = has & ~live[ps]
        prov[gone] = -1
        sat[gone] = 0.3
        # 2. awareness from search results, sponsored results and word of mouth
        for key, field in (("seen", 3), ("sponsored", 2)):
            for ni_s, slots_seen in p[key].items():
                if not slots_seen:
                    continue
                ni = int(ni_s)
                mask = (need == ni) & ((prov < 0) | (sat < self.cfg["shop_threshold"]))
                idx = np.flatnonzero(mask)
                m = int(round(len(idx) * self.cfg["search_rate"]))
                if m == 0:
                    continue
                who = rng.choice(idx, size=min(m, len(idx)), replace=False)
                for s in slots_seen[:field]:
                    self._learn(aware, who, s)
        happy = np.flatnonzero((prov >= 0) & (sat > 0.6))
        if len(happy):
            talk = happy[rng.random(len(happy)) < self.cfg["wom_rate"]]
            if len(talk):
                listener = nbr[talk, rng.integers(0, DEG, len(talk))]
                same_need = need[listener] == need[talk]
                self._learn_pairs(aware, listener[same_need], prov[talk][same_need])
        # 3. shopping: unserved or dissatisfied consumers evaluate the offerings they know
        shop = (prov < 0) | ((sat < self.cfg["shop_threshold"]) & (rng.random(n) < (1.0 - loyal) * 0.3))
        idx = np.flatnonzero(shop)
        if len(idx):
            A = aware[idx]                                        # [m, K]
            valid = A >= 0
            Ac = np.where(valid, A, n_slots)
            ok_need = np.zeros_like(valid)
            for s, v in public.items():
                ok_need |= (Ac == s) & (need[idx][:, None] == v["need"])
            valid &= ok_need & live[Ac]
            expected_q = (conf[Ac] * rating[Ac] / 5.0 + (1 - conf[Ac]) * 0.6)
            eu = pref[idx, 0:1] * expected_q - pref[idx, 2:3] * np.minimum(2.0, price[Ac] / np.maximum(wtp[idx, None], 0.01)) * 0.5
            eu = eu + rng.gumbel(0, 0.05, eu.shape).astype(np.float32)
            eu = np.where(valid, eu, -np.inf)
            best = np.argmax(eu, axis=1)
            best_eu = eu[np.arange(len(idx)), best]
            choice = A[np.arange(len(idx)), best]
            cur = prov[idx]
            switch = np.isfinite(best_eu) & (choice != cur) & (best_eu > -0.05)
            switch &= (cur < 0) | (best_eu > (sat[idx] - 0.5) * 0.4 + loyal[idx] * 0.1)
            sw = idx[switch]
            prov[sw] = choice[switch]
            sat[sw] = 0.5
            since[sw] = hour
        # 4. turnover: some consumers leave the market, replaced by newcomers with fresh preferences
        churn = np.flatnonzero(rng.random(n) < self.cfg["turnover_per_day"] / 24.0)
        if len(churn):
            prov[churn] = -1
            sat[churn] = 0.5
            aware[churn] = -1
        # 5. summary + voices (reviews, ratings, tickets) for the handler to turn into world events
        counts = np.bincount(np.where(prov >= 0, prov, n_slots), minlength=n_slots + 1)
        voices = []
        for k, sm in sorted(p["slots"].items(), key=lambda kv: int(kv[0])):
            s = int(k)
            cust = np.flatnonzero(prov == s)
            st.put("cons_summary", k, {"slot": s, "customers": int(counts[s]), "t": p["t"],
                                      "mean_sat": float(sat[cust].mean()) if len(cust) else None,
                                      "requests": sm["requests"], "ok": sm["ok"], "q": sm["q"]})
            if not len(cust):
                continue
            n_rev = int(rng.binomial(len(cust), self.cfg["review_rate"]))
            for i in rng.choice(cust, size=min(n_rev, len(cust)), replace=False) if n_rev else []:
                voices.append({"kind": "review", "slot": s, "stars": int(1 + round(4 * float(sat[i]))),
                               "persona": int(i % max(1, self.cfg["personas"]))})
            tot = sm["ok"] + sm["errors"] + sm["unavailable"] + sm["overload"]
            fail = (sm["errors"] + sm["unavailable"] + sm["overload"]) / tot if tot else 0.0
            if fail > 0.05:
                n_t = int(rng.binomial(len(cust), min(1.0, self.cfg["ticket_rate"] * fail)))
                for i in (rng.choice(cust, size=min(n_t, len(cust), 20), replace=False) if n_t else []):
                    voices.append({"kind": "ticket", "slot": s, "customer": f"consumer:{int(i)}",
                                   "fail": round(fail, 3), "persona": int(i % max(1, self.cfg["personas"]))})
        for j, v in enumerate(voices):
            st.put("cons_voices", f"{hour}|{j:04d}", v)
        st.meta_set("cons_hour", hour)

    def _learn(self, aware, who, slot):
        if len(who) == 0:
            return
        rows = aware[who]
        known = (rows == slot).any(axis=1)
        who = who[~known]
        if len(who) == 0:
            return
        rows = aware[who]
        empty = rows < 0
        pos = np.where(empty.any(axis=1), empty.argmax(axis=1), (aware[who, 0] % K_AWARE + K_AWARE) % K_AWARE)
        aware[who, pos] = slot

    def _learn_pairs(self, aware, who, slots):
        for s in np.unique(slots):
            self._learn(aware, who[slots == s], int(s))

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
        """Commission a market survey ($250): noisy estimates of demand for a need, willingness to pay, how many
        customers each known offering has and their satisfaction. Estimates are sampled, not exact."""
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
        if st.tables["accounts"]["acct_households"]["balance"] < 0:
            errs.append("households overspent")
        return errs


def need_index(world, need: str) -> int:
    return world.state.meta_get("cons_needs").index(need)
