"""Consumer behaviour models: structurally different rules by which the simulated population finds, tries, keeps
and drops offerings (registry `BEHAVIOR_MODELS`), plus the configuration of segments and of latent / derived demand.

The `consumers` module (economy/consumers.py) owns the state and the event; this file owns the *rules*. Every
simulated hour the `consumers.hour` applier decodes what happened in the market into an `Hour` (only things a
consumer could observe: public price, public rating and number of ratings, which offerings exist, what the search
engine and the ads showed, and the quality its own provider delivered), and then calls, for every segment, the
four steps of that segment's model:

    experience(pop, mk, rng)   own experience of the current provider -> satisfaction (and private memory)
    awareness(pop, mk, rng)    which signal makes whom aware of (or try) what
    choose(pop, mk, rng)       first trial, switching, abandonment
    voice(pop, mk, rng, ...)   reviews and support tickets

`pop` (`Pop`) is one segment's view of the population arrays; all state lives in the state store's arrays, and all
randomness comes from the generator the applier passes in (seeded from the world seed and the hour).

The models differ in causal structure, not in constants:

    model            awareness comes from            a first trial needs                  retention / what feeds back
    ---------------  ------------------------------  -----------------------------------  -----------------------------------
    legacy           search, ads, happy neighbours   best expected utility among known    satisfaction; ratings; word of mouth
                                                     offers (public rating, price)
    contagion        neighbours' visible use         share of neighbours using it >= own  experienced quality -> abandonment,
                     (search/ads only seed a few)    threshold                            and abandonment spreads to neighbours
    reputation_gate  search, ads, happy neighbours   enough public ratings AND rating >=  habit: the bar to switch rises with
                                                     own trust bar (ads never suffice;    tenure; own reviews gate the others
                                                     a few pioneers try the unrated)
    explorer         the public directory            spontaneous, at a personal rate      own remembered experience only
                                                     (ratings and ads ignored)            (a small bandit memory)
    bargain          the public price list           being the cheapest acceptable        price; quality only below a floor
                                                                                          (then the offering is avoided)

`legacy` is the Crucible 3.0 rule set, moved here without changing a single random draw: a world configured
without `behavior`, `segments`, `latent` or `derived` evolves byte for byte as it did in 3.0
(tests/test_behavior.py compares digests recorded from the 3.0 tree).

None of these models is calibrated against data; see `CALIBRATION`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.errors import ProtocolError

K_AWARE = 6          # size of a consumer's awareness set
DEG = 6              # neighbours per consumer in the social graph
MEM = 4              # offerings an `explorer` remembers

# parameters every model reads; a segment may override them, the default is the module configuration
COMMON_PARAMS = ("search_rate", "wom_rate", "review_rate", "ticket_rate", "shop_threshold")
# per-segment overrides of how the segment's members are drawn at genesis, and of its spending envelope
POPULATION_PARAMS = ("rate_per_day", "wtp_mult", "loyalty", "budget_share")


# ====================================================================== awareness sets (shared by several models)
def learn(aware: np.ndarray, who: np.ndarray, slot: int) -> None:
    """Consumers `who` become aware of offering `slot` (first empty place of their awareness set, else one is
    overwritten)."""
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


def learn_pairs(aware: np.ndarray, who: np.ndarray, slots: np.ndarray) -> None:
    for s in np.unique(slots):
        learn(aware, who[slots == s], int(s))


# ====================================================================== what a consumer can observe in one hour
class Hour:
    """One simulated hour of the market, decoded from a `consumers.hour` event payload.

    Per-offering arrays are indexed by slot and have one extra trailing entry (index `n_slots`) that stands for
    "no offering", so `arr[np.where(prov >= 0, prov, n_slots)]` is always valid.

    public (anyone can look it up): `price`, `rating` (mean stars, 3.0 when unrated), `nrat` (number of ratings),
    `conf` (nrat / (nrat + 5)), `live`, `slot_need`, and the search results `payload["seen"]` /
    `payload["sponsored"]` per need. Experienced only by an offering's own customers: `q_eff` (verified quality x
    share of requests that succeeded), `lat`, `served`.
    """
    __slots__ = ("payload", "t", "hour", "public", "n_slots", "n_needs", "price", "rating", "conf", "nrat", "live",
                 "slot_need", "q_eff", "lat", "served", "_cand")

    def __init__(self, payload: dict, min_slots: int, n_needs: int):
        p = payload
        self.payload = p
        self.t = p["t"]
        self.hour = p["t"] // 3600
        public = {int(k): v for k, v in p["public"].items()}
        self.public = public
        n_slots = max([s for s in public] + [int(k) for k in p["slots"]] + [-1]) + 1
        # slots that closed still appear in consumers' current provider, awareness sets and memories: size the
        # per-slot arrays to cover them (an offering closing at the top of the slot range used to crash every hour)
        n_slots = max(n_slots, min_slots)
        self.n_slots = n_slots
        self.n_needs = n_needs
        price = np.zeros(n_slots + 1, np.float32)
        rating = np.full(n_slots + 1, 3.0, np.float32)
        conf = np.zeros(n_slots + 1, np.float32)
        nrat = np.zeros(n_slots + 1, np.int32)
        live = np.zeros(n_slots + 1, bool)
        slot_need = np.full(n_slots + 1, -1, np.int32)
        for s, v in public.items():
            price[s] = v["price"]
            if v["rating"] is not None:
                rating[s] = v["rating"]
                conf[s] = v["n"] / (v["n"] + 5.0)
            nrat[s] = v["n"]
            live[s] = True
            slot_need[s] = v["need"]
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
        self.price, self.rating, self.conf, self.nrat, self.live, self.slot_need = price, rating, conf, nrat, live, \
            slot_need
        self.q_eff, self.lat, self.served = q_eff, lat, served
        self._cand = None

    def directory(self) -> tuple[np.ndarray, np.ndarray]:
        """The public directory: (`tab` [n_needs, C] slot ids of the live offerings of each need, -1 padded;
        `cnt` [n_needs] how many each need has). This is what the market's browse tool shows to anyone."""
        if self._cand is None:
            by: dict[int, list[int]] = {}
            for s in sorted(self.public):
                by.setdefault(int(self.public[s]["need"]), []).append(s)
            width = max([len(v) for v in by.values()] + [1])
            tab = np.full((max(1, self.n_needs), width), -1, np.int64)
            cnt = np.zeros(max(1, self.n_needs), np.int64)
            for ni, v in by.items():
                if 0 <= ni < self.n_needs:
                    tab[ni, :len(v)] = v
                    cnt[ni] = len(v)
            self._cand = (tab, cnt)
        return self._cand


class Pop:
    """One segment's view of the population. The arrays are the state store's (shared by all segments, indexed by
    consumer id); `m` holds the ids of this segment's members, or None when the segment is the whole population
    (then every operation runs on the arrays themselves, exactly as the 3.0 code did).

    Models may read their own members' rows, `nbr` (the social graph) and their neighbours' *visible* state: the
    provider a neighbour currently uses (`prov`), whether it has the need (`need`, `active`) and the offering it
    recently dropped (`x["c_quit"]`). They never read another consumer's satisfaction, preferences or memory.
    """
    __slots__ = ("n", "m", "index", "name", "params", "prov", "sat", "aware", "since", "need", "pref", "wtp", "loyal",
                 "nbr", "active", "x", "_all")

    def __init__(self, n, m, index, name, params, arrays, x):
        self.n, self.m, self.index, self.name, self.params = n, m, index, name, params
        self.prov, self.sat, self.aware, self.since = arrays["prov"], arrays["sat"], arrays["aware"], arrays["since"]
        self.need, self.pref, self.wtp, self.loyal = arrays["need"], arrays["pref"], arrays["wtp"], arrays["loyal"]
        self.nbr = arrays["nbr"]
        self.active = arrays.get("active")
        self.x = x
        self._all = None

    def g(self, a: np.ndarray) -> np.ndarray:
        """The members' rows of a population array (the array itself when the segment is everyone)."""
        return a if self.m is None else a[self.m]

    def ix(self, local: np.ndarray) -> np.ndarray:
        """Positions within the segment (indices or a mask over `g(...)`) -> consumer ids."""
        return local if self.m is None else self.m[local]

    def put(self, a: np.ndarray, local: np.ndarray, value) -> None:
        if self.m is None:
            a[local] = value
        else:
            a[self.m[local]] = value

    def members(self, active_only: bool = False) -> np.ndarray:
        if self.m is None:
            if self._all is None:
                self._all = np.arange(self.n)
            m = self._all
        else:
            m = self.m
        if active_only and self.active is not None:
            m = m[self.active[m]]
        return m


# ====================================================================== the interface, and the 3.0 rule set
class BehaviorModel:
    """Interface of a population update rule. The base class *is* the Crucible 3.0 rule set (`legacy`); the other
    models override the steps whose causal structure differs.

    Class attributes:
      `defaults`     model parameters (a segment's `params` may override them)
      `arrays`       extra per-consumer state: name -> (dtype, trailing shape, fill value, written every hour?)
      `slot_arrays`  those of `arrays` that hold offering slots (they bound the per-slot arrays of `Hour`)
      `uses_search`  whether members look at search results (the handler bills ad impressions per searcher)
    """
    name = "legacy"
    uses_search = True
    prune = False          # drop shoppers with nothing to consider before drawing choice noise (3.0 does not)
    defaults: dict = {}
    arrays: dict = {}
    slot_arrays: tuple = ()

    # ------------------------------------------------------------------ genesis / turnover
    def init(self, pop: Pop, rng: np.random.Generator) -> None:
        """Draw the members' model-specific traits (seeded heterogeneity). Called once, at genesis."""

    def forget(self, pop: Pop, who: np.ndarray) -> None:
        """Consumers `who` (ids, members of this segment) left and were replaced by newcomers."""

    # ------------------------------------------------------------------ who searches
    def searchers(self, pop: Pop, ni: int) -> np.ndarray:
        """Mask over the segment: consumers of need `ni` who are looking (no provider, or dissatisfied)."""
        mask = (pop.g(pop.need) == ni) & ((pop.g(pop.prov) < 0) | (pop.g(pop.sat) < pop.params["shop_threshold"]))
        if pop.active is not None:
            mask &= pop.g(pop.active)
        return mask

    # ------------------------------------------------------------------ 1. experience
    def utility(self, pop: Pop, mk: Hour, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Utility each member experienced from its provider this hour (verified quality, latency, price against
        willingness to pay, idiosyncratic noise). Returns (u, experienced mask, provider slot or n_slots)."""
        P = pop.g(pop.prov)
        has = P >= 0
        ps = np.where(has, P, mk.n_slots)
        exp = has & mk.served[ps]
        pref = pop.g(pop.pref)
        wtp = pop.g(pop.wtp)
        u = pref[:, 0] * mk.q_eff[ps] - pref[:, 1] * mk.lat[ps] - pref[:, 2] * np.minimum(
            2.0, mk.price[ps] / np.maximum(wtp, 0.01)) * 0.5
        u = u + rng.normal(0, 0.03, len(P)).astype(np.float32)
        return u, exp, ps

    def experience(self, pop: Pop, mk: Hour, rng) -> None:
        u, exp, _ = self.utility(pop, mk, rng)
        S = pop.g(pop.sat)
        pop.put(pop.sat, exp, 0.85 * S[exp] + 0.15 * np.clip(u[exp] * 1.6 + 0.2, 0, 1))

    # ------------------------------------------------------------------ 2. awareness
    def awareness(self, pop: Pop, mk: Hour, rng) -> None:
        prm = pop.params
        for key, field in (("seen", 3), ("sponsored", 2)):
            # in need order: the order the handler wrote them in, and (unlike plain dict order) also the order
            # after the event has been read back from a store, where keys come sorted as strings ("10" < "2")
            for ni_s, slots_seen in sorted(mk.payload[key].items(), key=lambda kv: int(kv[0])):
                if not slots_seen:
                    continue
                ni = int(ni_s)
                idx = pop.ix(np.flatnonzero(self.searchers(pop, ni)))
                m = int(round(len(idx) * prm["search_rate"]))
                if m == 0:
                    continue
                who = rng.choice(idx, size=min(m, len(idx)), replace=False)
                self.exposed(pop, mk, rng, key, who, slots_seen[:field])
        self.word_of_mouth(pop, mk, rng)

    def exposed(self, pop: Pop, mk: Hour, rng, key: str, who: np.ndarray, slots: list) -> None:
        """Consumers `who` saw offerings `slots` in organic (`key == "seen"`) or sponsored results."""
        for s in slots:
            learn(pop.aware, who, s)

    def word_of_mouth(self, pop: Pop, mk: Hour, rng) -> None:
        happy = pop.ix(np.flatnonzero((pop.g(pop.prov) >= 0) & (pop.g(pop.sat) > 0.6)))
        if len(happy):
            talk = happy[rng.random(len(happy)) < pop.params["wom_rate"]]
            if len(talk):
                listener = pop.nbr[talk, rng.integers(0, DEG, len(talk))]
                same_need = pop.need[listener] == pop.need[talk]
                learn_pairs(pop.aware, listener[same_need], pop.prov[talk][same_need])

    # ------------------------------------------------------------------ 3. choice
    def consider(self, pop: Pop, mk: Hour, idx: np.ndarray, Ac: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Restrict the known offerings a shopper is willing to consider ([len(idx), K_AWARE] mask)."""
        return valid

    def switch_bar(self, pop: Pop, mk: Hour, idx: np.ndarray) -> np.ndarray | None:
        """Extra utility a served shopper demands before leaving its provider (None: nothing extra)."""
        return None

    def choose(self, pop: Pop, mk: Hour, rng) -> None:
        """Unserved or dissatisfied consumers evaluate the offerings they know."""
        prm = pop.params
        P = pop.g(pop.prov)
        shop = (P < 0) | ((pop.g(pop.sat) < prm["shop_threshold"]) &
                          (rng.random(len(P)) < (1.0 - pop.g(pop.loyal)) * 0.3))
        if pop.active is not None:
            shop &= pop.g(pop.active)
        idx = pop.ix(np.flatnonzero(shop))
        if len(idx):
            A = pop.aware[idx]                                    # [m, K]
            valid = A >= 0
            Ac = np.where(valid, A, mk.n_slots)
            valid &= (mk.slot_need[Ac] == pop.need[idx][:, None]) & mk.live[Ac]
            valid = self.consider(pop, mk, idx, Ac, valid)
            if self.prune:
                keep = valid.any(axis=1)
                idx, A, Ac, valid = idx[keep], A[keep], Ac[keep], valid[keep]
            expected_q = (mk.conf[Ac] * mk.rating[Ac] / 5.0 + (1 - mk.conf[Ac]) * 0.6)
            eu = pop.pref[idx, 0:1] * expected_q - pop.pref[idx, 2:3] * np.minimum(
                2.0, mk.price[Ac] / np.maximum(pop.wtp[idx, None], 0.01)) * 0.5
            eu = eu + rng.gumbel(0, 0.05, eu.shape).astype(np.float32)
            eu = np.where(valid, eu, -np.inf)
            best = np.argmax(eu, axis=1)
            best_eu = eu[np.arange(len(idx)), best]
            choice = A[np.arange(len(idx)), best]
            cur = pop.prov[idx]
            switch = np.isfinite(best_eu) & (choice != cur) & (best_eu > -0.05)
            bar = (pop.sat[idx] - 0.5) * 0.4 + pop.loyal[idx] * 0.1
            extra = self.switch_bar(pop, mk, idx)
            if extra is not None:
                bar = bar + extra
            switch &= (cur < 0) | (best_eu > bar)
            sw = idx[switch]
            pop.prov[sw] = choice[switch]
            pop.sat[sw] = 0.5
            pop.since[sw] = mk.hour

    # ------------------------------------------------------------------ 4. voice
    def review_rate(self, prm: dict) -> float:
        return prm["review_rate"]

    def voice(self, pop: Pop, mk: Hour, rng, slot: int, cust: np.ndarray, sm: dict, out: list) -> None:
        """Reviews (stars from own satisfaction) and support tickets (when requests failed) of this segment's
        customers `cust` (ids) of offering `slot`; appended to `out` for the handler to publish."""
        prm = pop.params
        personas = max(1, prm["personas"])
        n_rev = int(rng.binomial(len(cust), self.review_rate(prm)))
        for i in rng.choice(cust, size=min(n_rev, len(cust)), replace=False) if n_rev else []:
            out.append({"kind": "review", "slot": slot, "stars": int(1 + round(4 * float(pop.sat[i]))),
                        "persona": int(i % personas)})
        tot = sm["ok"] + sm["errors"] + sm["unavailable"] + sm["overload"]
        fail = (sm["errors"] + sm["unavailable"] + sm["overload"]) / tot if tot else 0.0
        if fail > 0.05:
            n_t = int(rng.binomial(len(cust), min(1.0, prm["ticket_rate"] * fail)))
            for i in (rng.choice(cust, size=min(n_t, len(cust), 20), replace=False) if n_t else []):
                out.append({"kind": "ticket", "slot": slot, "customer": f"consumer:{int(i)}",
                            "fail": round(fail, 3), "persona": int(i % personas)})


BEHAVIOR_MODELS: dict[str, BehaviorModel] = {}


def register_behavior(cls: type[BehaviorModel]) -> type[BehaviorModel]:
    BEHAVIOR_MODELS[cls.name] = cls()
    return cls


register_behavior(BehaviorModel)       # "legacy"


def _fresh(hour: int, when: np.ndarray, window) -> np.ndarray:
    return (hour - when) <= window


# ====================================================================== contagion
@register_behavior
class Contagion(BehaviorModel):
    """Adoption is social (a threshold model on the social graph).

    * awareness / trial: a consumer takes up an offering when the share of its neighbours *with the same need*
      who visibly use it exceeds the share using its current provider (0 if it has none) by at least its personal
      threshold `c_thr` (drawn once, uniform in `threshold`; a served consumer's bar is raised by its loyalty).
      Public ratings play no part, and price only as affordability (`price < afford * willingness to pay`).
    * search and sponsored results do not persuade: of the consumers they reach, a fraction `seed_rate` of those
      without a provider simply try what they were shown. These are the only adoptions that do not come from a
      neighbour.
    * retention: own experience. A dissatisfied customer abandons at rate `quit_rate * (1 - loyalty)` per hour and
      remembers what it dropped (`c_quit`, `c_quit_h`).
    * feedback: abandonment is visible too. A customer leaves when, among its neighbours with the same need,
      those who dropped its provider within `quit_memory_h` outnumber those still using it by at least its
      threshold share (scaled by `quit_spread`; 0 switches this channel off), and nobody re-adopts an offering it
      dropped within that window. Good quality therefore spreads by use, bad quality un-spreads by defection.
    """
    name = "contagion"
    defaults = {"threshold": [0.15, 0.55], "seed_rate": 0.03, "afford": 2.0, "quit_rate": 0.3,
                "quit_memory_h": 72, "quit_spread": 1.0, "switch_loyalty": 1.0}
    arrays = {"c_thr": (np.float32, (), 0.0, False), "c_quit": (np.int32, (), -1, True),
              "c_quit_h": (np.int32, (), 0, True)}
    slot_arrays = ("c_quit",)

    def init(self, pop, rng):
        m = pop.members()
        lo, hi = pop.params["threshold"]
        pop.x["c_thr"][m] = rng.uniform(lo, hi, len(m)).astype(np.float32)

    def forget(self, pop, who):
        pop.x["c_quit"][who] = -1

    def word_of_mouth(self, pop, mk, rng):
        return None            # influence travels through visible use (see `choose`), not through recommendations

    def exposed(self, pop, mk, rng, key, who, slots):
        prm = pop.params
        take = who[rng.random(len(who)) < prm["seed_rate"]]
        pick = np.asarray(slots, np.int64)[rng.integers(0, len(slots), len(take))]
        if len(take) == 0:
            return
        dropped = (pop.x["c_quit"][take] == pick) & _fresh(mk.hour, pop.x["c_quit_h"][take], prm["quit_memory_h"])
        ok = (pop.prov[take] < 0) & mk.live[pick] & (mk.slot_need[pick] == pop.need[take]) & ~dropped & \
            (mk.price[pick] < prm["afford"] * pop.wtp[take])
        g = take[ok]
        pop.prov[g] = pick[ok]
        pop.sat[g] = 0.5
        pop.since[g] = mk.hour

    def choose(self, pop, mk, rng):
        prm = pop.params
        hour = mk.hour
        P = pop.g(pop.prov)
        k = len(P)
        r = rng.random(k)
        if k == 0:
            return
        thr = pop.g(pop.x["c_thr"])
        quit_s, quit_h = pop.x["c_quit"], pop.x["c_quit_h"]
        nb = pop.g(pop.nbr)                                       # [k, DEG] neighbour ids
        NP = pop.prov.take(nb)                                    # the offering each neighbour visibly uses
        served = P >= 0

        def peers(rows):
            """How many neighbours of members `rows` face the same need (the reference group; at least 1)."""
            g = pop.ix(rows)
            nbr = nb[rows]
            same = pop.need.take(nbr) == pop.need[g][:, None]
            if pop.active is not None:
                same &= pop.active.take(nbr)
            return np.maximum(1, same.sum(axis=1)).astype(np.float32)

        # (a) abandonment: own bad experience, or neighbours who visibly dropped the offering outnumber those
        # who still use it by the consumer's threshold share
        out = served & (pop.g(pop.sat) < prm["shop_threshold"]) & (r < (1.0 - pop.g(pop.loyal)) * prm["quit_rate"])
        recent = (quit_s >= 0) & _fresh(hour, quit_h, prm["quit_memory_h"])
        if prm["quit_spread"] > 0 and recent.any():
            dropped = np.where(recent, quit_s, -1)                # what each consumer visibly dropped (or -1)
            rs = np.flatnonzero(served)
            left = (dropped.take(nb[rs]) == P[rs][:, None]).sum(axis=1)
            rs, left = rs[left > 0], left[left > 0]
            if len(rs):
                stay = (NP[rs] == P[rs][:, None]).sum(axis=1)
                net = prm["quit_spread"] * (left - stay) / peers(rs)
                out[rs[net >= thr[rs] - 1e-6]] = True
        if out.any():
            g = pop.ix(np.flatnonzero(out))
            quit_s[g] = P[out]
            quit_h[g] = hour
            pop.prov[g] = -1
            P = pop.g(pop.prov)
            NP = pop.prov.take(nb)
        # (b) adoption and switching by the share of neighbours using each offering
        cand = (NP >= 0) & (NP != P[:, None])
        if pop.active is not None:
            cand &= pop.g(pop.active)[:, None]
        rows = np.flatnonzero(cand.any(axis=1))
        if len(rows) == 0:
            return
        g = pop.ix(rows)
        NP, cur = NP[rows], P[rows]
        NPc = np.where(NP >= 0, NP, mk.n_slots)
        ok = cand[rows] & mk.live[NPc] & (mk.slot_need[NPc] == pop.need[g][:, None])
        ok &= mk.price[NPc] < prm["afford"] * pop.wtp[g][:, None]
        ok &= ~((NP == quit_s[g][:, None]) & _fresh(hour, quit_h[g], prm["quit_memory_h"])[:, None])
        keep = ok.any(axis=1)
        if not keep.all():
            rows, g, NP, cur, ok = rows[keep], g[keep], NP[keep], cur[keep], ok[keep]
            if len(rows) == 0:
                return
        denom = peers(rows)
        cnt = np.zeros(NP.shape, np.int8)                         # neighbours using the same offering as column j
        for j in range(NP.shape[1]):
            cnt += NP == NP[:, j:j + 1]
        share = np.where(ok, cnt, 0) / denom[:, None]
        own_share = ((NP == cur[:, None]) & (cur[:, None] >= 0)).sum(axis=1) / denom
        ar = np.arange(len(rows))
        j = share.argmax(axis=1)
        top = share[ar, j]
        bar = thr[rows] * np.where(cur >= 0, 1.0 + prm["switch_loyalty"] * pop.loyal[g], 1.0)
        go = (top > 0) & (top - own_share >= bar - 1e-6)
        gg = g[go]
        pop.prov[gg] = NP[ar, j][go]
        pop.sat[gg] = 0.5
        pop.since[gg] = hour


# ====================================================================== reputation gate
@register_behavior
class ReputationGate(BehaviorModel):
    """Trial is gated by public reputation; retention by habit.

    * awareness: as in `legacy` (search results, sponsored results, happy neighbours). Awareness is necessary but
      never sufficient: advertising cannot produce a trial on its own.
    * trial: a consumer considers an offering only if it has at least `c_evid` public ratings *and* its public
      rating is at least the consumer's personal trust bar `c_trust` (stars, uniform in `trust`). Among the
      offerings that pass, the choice is the legacy expected utility (rating, price). `c_evid` is `min_ratings`,
      except for a share `pioneer_share` of consumers who will try an offering nobody has rated yet (they still
      refuse one that is rated below their bar). Pioneers are what lets a market of gated consumers start at
      all; with `pioneer_share: 0` the gate is absolute and somebody else has to supply the first ratings.
    * retention: habit. A served consumer demands, on top of the legacy switching cost, an extra
      `habit * (1 - exp(-tenure_hours / habit_tau_h))` of utility before it leaves.
    * feedback: customers' own reviews are the public rating that opens or closes the gate for everyone else
      (members review `review_mult` times as often as the module's `review_rate`).
    """
    name = "reputation_gate"
    prune = True
    defaults = {"trust": [2.5, 4.0], "min_ratings": 3, "pioneer_share": 0.1, "habit": 0.3, "habit_tau_h": 72.0,
                "review_mult": 4.0}
    arrays = {"c_trust": (np.float32, (), 0.0, False), "c_evid": (np.int16, (), 0, False)}

    def init(self, pop, rng):
        m = pop.members()
        lo, hi = pop.params["trust"]
        pop.x["c_trust"][m] = rng.uniform(lo, hi, len(m)).astype(np.float32)
        pioneer = rng.random(len(m)) < pop.params["pioneer_share"]
        pop.x["c_evid"][m] = np.where(pioneer, 0, int(pop.params["min_ratings"])).astype(np.int16)

    def consider(self, pop, mk, idx, Ac, valid):
        evid = pop.x["c_evid"][idx][:, None]
        rated = mk.nrat[Ac] >= np.maximum(evid, 1)
        return valid & np.where(rated, mk.rating[Ac] >= pop.x["c_trust"][idx][:, None], evid == 0)

    def switch_bar(self, pop, mk, idx):
        tenure = np.maximum(0, mk.hour - pop.since[idx]).astype(np.float32)
        return pop.params["habit"] * (1.0 - np.exp(-tenure / float(pop.params["habit_tau_h"])))

    def review_rate(self, prm):
        return min(1.0, prm["review_rate"] * prm["review_mult"])


# ====================================================================== explorer
@register_behavior
class Explorer(BehaviorModel):
    """Experience-driven choice with spontaneous exploration (a small bandit per consumer).

    * awareness: the public directory of live offerings for the consumer's need. Search rank, advertising and
      public ratings are ignored.
    * trial: spontaneous. Each hour a consumer tries a random offering of its need with probability
      `1 - exp(-c_eps / 24)` (`c_eps` per day, uniform in `explore_per_day`; `unserved_boost` times faster while
      it has no provider), preferring one it does not remember.
    * retention: own remembered experience only. `c_mslot` / `c_mval` hold up to MEM offerings and a running mean
      (step `learn`, starting from `prior`) of the utility the consumer itself experienced with each; column 0 is
      the current provider. After `trial_h` hours with a provider it switches back to the best remembered
      alternative if that beats the current one by `margin`, and leaves if the current one is worth less than
      `outside`.
    * feedback: none through public signals (explorers do write reviews, which other models read).
    """
    name = "explorer"
    uses_search = False
    defaults = {"explore_per_day": [0.3, 1.5], "learn": 0.25, "prior": 0.2, "margin": 0.03, "outside": 0.0,
                "trial_h": 6, "afford": 2.0, "unserved_boost": 4.0}
    arrays = {"c_eps": (np.float32, (), 0.0, False), "c_mslot": (np.int32, (MEM,), -1, True),
              "c_mval": (np.float32, (MEM,), 0.0, True)}
    slot_arrays = ("c_mslot",)

    def init(self, pop, rng):
        m = pop.members()
        lo, hi = pop.params["explore_per_day"]
        pop.x["c_eps"][m] = rng.uniform(lo, hi, len(m)).astype(np.float32)

    def forget(self, pop, who):
        pop.x["c_mslot"][who] = -1
        pop.x["c_mval"][who] = 0.0

    def awareness(self, pop, mk, rng):
        return None

    @staticmethod
    def _recall(pop, g, slot):
        """Consumers `g` (ids) now use offerings `slot`: bring each one's memory of that offering to column 0
        (a new memory starts at `prior`; when memory is full the worst remembered offering is forgotten)."""
        if len(g) == 0:
            return
        mslot, mval = pop.x["c_mslot"], pop.x["c_mval"]
        ms, mv = mslot[g], mval[g]
        ar = np.arange(len(g))
        hit = ms == slot[:, None]
        has = hit.any(axis=1)
        empty = ms < 0
        j = np.where(has, hit.argmax(axis=1), np.where(empty.any(axis=1), empty.argmax(axis=1), mv.argmin(axis=1)))
        new_v = np.where(has, mv[ar, j], np.float32(pop.params["prior"])).astype(np.float32)
        ms[ar, j], mv[ar, j] = ms[:, 0], mv[:, 0]          # what was in column 0 moves to the freed place
        ms[:, 0], mv[:, 0] = slot, new_v
        mslot[g], mval[g] = ms, mv

    def experience(self, pop, mk, rng):
        u, exp, ps = self.utility(pop, mk, rng)
        S = pop.g(pop.sat)
        pop.put(pop.sat, exp, 0.85 * S[exp] + 0.15 * np.clip(u[exp] * 1.6 + 0.2, 0, 1))
        g = pop.ix(np.flatnonzero(exp))
        if len(g) == 0:
            return
        mslot, mval = pop.x["c_mslot"], pop.x["c_mval"]
        slot = ps[exp]
        lost = mslot[g, 0] != slot        # a provider the consumer did not choose itself (an evaluator seeding)
        if lost.any():
            self._recall(pop, g[lost], slot[lost])
        mval[g, 0] += np.float32(pop.params["learn"]) * (u[exp].astype(np.float32) - mval[g, 0])

    def choose(self, pop, mk, rng):
        prm = pop.params
        hour = mk.hour
        P = pop.g(pop.prov)
        k = len(P)
        r = rng.random(k)
        if k == 0:
            return
        act = None if pop.active is None else pop.g(pop.active)
        need = pop.g(pop.need)
        served = P >= 0
        ms, mv = pop.g(pop.x["c_mslot"]), pop.g(pop.x["c_mval"])
        ar = np.arange(k)
        mine = served & (ms[:, 0] == P)                       # column 0 is the memory of the current provider
        cval = mv[:, 0]
        judged = mine & ((hour - pop.g(pop.since)) >= prm["trial_h"])
        val = np.where(ms >= 0, mv, -np.inf)                  # remembered alternatives
        val[mine, 0] = -np.inf
        b = val.argmax(axis=1)
        bval, bslot = val[ar, b], ms[ar, b]
        bs = np.where(bslot >= 0, bslot, mk.n_slots)
        alive = mk.live[bs] & (mk.slot_need[bs] == need)
        gone = np.isfinite(bval) & ~alive                     # remembered offering no longer exists: forget it
        if gone.any():
            pop.x["c_mslot"][pop.ix(np.flatnonzero(gone)), b[gone]] = -1
        better = np.isfinite(bval) & alive & (bval > prm["outside"]) & \
            (~served | (judged & (bval > cval + prm["margin"])))
        new = np.where(better, bslot, np.where(judged & (cval < prm["outside"]), -1, P))
        # exploration: a random offering from the public directory, preferably one not remembered
        rate = pop.g(pop.x["c_eps"]) * np.where(served, 1.0, prm["unserved_boost"]) / 24.0
        explore = r < -np.expm1(-rate)
        e = np.flatnonzero(explore)
        draw = rng.integers(0, 1 << 30, (len(e), 2))
        if len(e):
            tab, cnt = mk.directory()
            ne, cu = need[e], P[e]
            c_n = np.maximum(cnt[ne], 1)
            wtp = pop.g(pop.wtp)[e]
            c1, c2 = tab[ne, draw[:, 0] % c_n], tab[ne, draw[:, 1] % c_n]
            v1 = (c1 >= 0) & (c1 != cu) & (mk.price[np.where(c1 >= 0, c1, mk.n_slots)] < prm["afford"] * wtp)
            v2 = (c2 >= 0) & (c2 != cu) & (mk.price[np.where(c2 >= 0, c2, mk.n_slots)] < prm["afford"] * wtp)
            n1, n2 = ~(ms[e] == c1[:, None]).any(axis=1), ~(ms[e] == c2[:, None]).any(axis=1)
            pick = np.where(v1 & n1, c1, np.where(v2 & n2, c2, np.where(v1, c1, np.where(v2, c2, -1))))
            new[e[pick >= 0]] = pick[pick >= 0]
        ch = new != P
        if act is not None:
            ch &= act
        i = np.flatnonzero(ch)
        if len(i) == 0:
            return
        g = pop.ix(i)
        pop.prov[g] = new[i]
        ad = new[i] >= 0
        ga = g[ad]
        pop.sat[ga] = 0.5
        pop.since[ga] = hour
        self._recall(pop, ga, new[i][ad].astype(np.int32))


# ====================================================================== bargain
@register_behavior
class Bargain(BehaviorModel):
    """Price-dominant, disloyal, constantly re-shopping.

    * awareness: the public price list of live offerings for the consumer's need (no search, no advertising).
    * trial / switching: with probability `1 - exp(-reshop_per_day / 24)` per hour (always, while unserved) the
      consumer compares prices (perceived with relative noise `price_noise`) and moves to the cheapest offering it
      can afford and is not avoiding, if that is cheaper than its provider by more than
      `switch_cost * (1 + loyalty)`. Public ratings are ignored.
    * retention: none beyond that switching cost.
    * quality matters only below a floor: an hour in which the quality the consumer experienced is under
      `quality_floor` is worth nothing to it, otherwise the hour is worth its value for money
      (`1 - 0.5 * price / willingness to pay`), whatever the quality. When satisfaction falls under
      `shop_threshold` the consumer drops the offering and avoids it for `avoid_h` hours (`c_quit`, `c_quit_h`).
    """
    name = "bargain"
    uses_search = False
    defaults = {"reshop_per_day": 12.0, "quality_floor": 0.4, "price_noise": 0.03, "switch_cost": 0.05,
                "avoid_h": 72, "afford": 2.0}
    arrays = {"c_quit": (np.int32, (), -1, True), "c_quit_h": (np.int32, (), 0, True)}
    slot_arrays = ("c_quit",)

    def forget(self, pop, who):
        pop.x["c_quit"][who] = -1

    def awareness(self, pop, mk, rng):
        return None

    def experience(self, pop, mk, rng):
        P = pop.g(pop.prov)
        has = P >= 0
        ps = np.where(has, P, mk.n_slots)
        exp = has & mk.served[ps]
        q = mk.q_eff[ps] + rng.normal(0, 0.05, len(P)).astype(np.float32)
        value = np.clip(1.0 - 0.5 * mk.price[ps] / np.maximum(pop.g(pop.wtp), 0.01), 0, 1)
        worth = np.where(q >= pop.params["quality_floor"], value, 0.0).astype(np.float32)
        S = pop.g(pop.sat)
        pop.put(pop.sat, exp, 0.7 * S[exp] + 0.3 * worth[exp])

    def choose(self, pop, mk, rng):
        prm = pop.params
        hour = mk.hour
        m = pop.members(active_only=True)
        k = len(m)
        r = rng.random(k)
        if k == 0:
            return
        quit_s, quit_h = pop.x["c_quit"], pop.x["c_quit_h"]
        cur = pop.prov[m]
        low = (cur >= 0) & (pop.sat[m] < prm["shop_threshold"])   # quality fell through the floor: drop and avoid
        if low.any():
            g = m[low]
            quit_s[g] = cur[low]
            quit_h[g] = hour
            pop.prov[g] = -1
            cur = pop.prov[m]
        i = np.flatnonzero((cur < 0) | (r < -math.expm1(-float(prm["reshop_per_day"]) / 24.0)))
        tab, _ = mk.directory()
        noise = rng.gumbel(0.0, 1.0, (len(i), tab.shape[1]))
        if len(i) == 0:
            return
        g, cu = m[i], cur[i]
        S = tab[pop.need[g]]                                      # [k', C] the price list for each shopper's need
        Sc = np.where(S >= 0, S, mk.n_slots)
        ok = (S >= 0) & (mk.price[Sc] < prm["afford"] * pop.wtp[g][:, None])
        ok &= ~((S == quit_s[g][:, None]) & _fresh(hour, quit_h[g], prm["avoid_h"])[:, None])
        seen = np.where(ok, np.log(np.maximum(mk.price[Sc], 1e-6)) + prm["price_noise"] * noise, np.inf)
        ar = np.arange(len(i))
        b = seen.argmin(axis=1)
        ch = S[ar, b]
        p_new = mk.price[np.where(ch >= 0, ch, mk.n_slots)]
        p_cur = mk.price[np.where(cu >= 0, cu, mk.n_slots)]
        go = np.isfinite(seen[ar, b]) & (ch != cu) & \
            ((cu < 0) | (p_new < p_cur * (1.0 - prm["switch_cost"] * (1.0 + pop.loyal[g]))))
        gg = g[go]
        pop.prov[gg] = ch[go]
        pop.sat[gg] = 0.5
        pop.since[gg] = hour


# ====================================================================== honesty about realism
CALIBRATION = {
    "_note": "No behaviour model in this file is calibrated against real consumer data. Every parameter default "
             "is a design choice made so that the mechanism is visible within days of simulated time in worlds of "
             "a few hundred to a few thousand consumers; none is an empirical estimate. The models are "
             "hypotheses about how a market might work, used to test whether a learner can find out which one "
             "it is in and whether its success survives a change of participant model.",
    "legacy": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "Random-utility choice over a limited awareness (consideration) set: argmax of utility plus "
                     "Gumbel noise is the form behind multinomial logit (McFadden 1974, 'Conditional logit "
                     "analysis of qualitative choice behavior'). The satisfaction update, awareness diffusion and "
                     "review process are ad hoc.",
    },
    "contagion": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "Threshold models of collective behaviour (Granovetter 1978, 'Threshold Models of Collective "
                     "Behavior', American Journal of Sociology 83(6)) placed on a network (Watts 2002, 'A simple "
                     "model of global cascades on random networks', PNAS 99(9)). Thresholds above one neighbour "
                     "make it a complex contagion in the sense of Centola and Macy (2007). The externally seeded "
                     "adopters play the role of the innovators in the Bass (1969) diffusion model. Abandonment "
                     "by the same threshold rule is this project's own addition.",
    },
    "reputation_gate": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "Markets in which buyers rely on a public feedback score (reputation systems: Resnick, "
                     "Zeckhauser, Friedman and Kuwabara 2000, Communications of the ACM 43(12)), combined with "
                     "consumer switching costs (Klemperer 1987, 'Markets with consumer switching costs'). The "
                     "hard gate, the personal trust bar and the tenure-dependent habit term are stylised.",
    },
    "explorer": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "An epsilon-greedy multi-armed bandit with recency-weighted value estimates (Sutton and "
                     "Barto, 'Reinforcement Learning: An Introduction', chapter 2), one per consumer.",
    },
    "bargain": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "The fully informed 'shoppers' who buy from the lowest-price seller in Varian (1980), 'A "
                     "Model of Sales' (American Economic Review 70(4)); price competition for them is close to "
                     "Bertrand competition. The quality floor and the avoidance window are ad hoc.",
    },
    "segments": {
        "calibrated": False, "status": "design choice",
        "resembles": "Finite-mixture (latent-class) descriptions of heterogeneous markets in general; the "
                     "segment shares used in world families are not estimates of any real market.",
    },
    "derived_demand": {
        "calibrated": False, "status": "plausible hypothesis",
        "resembles": "No specific published model. The saturating form 1 - exp(-volume / scale), the first-order "
                     "relaxation at `activation_rate` per day and the lag are chosen for transparency, not "
                     "fitted to anything.",
    },
}


# ====================================================================== configuration: segments
@dataclass
class Segment:
    index: int
    name: str
    share: float
    model: BehaviorModel
    params: dict          # resolved: module defaults <- model defaults <- the segment's overrides
    draw: dict            # genesis overrides for the members (rate_per_day, wtp_mult, loyalty)
    budget_share: float | None = None


def build_segments(cfg: dict) -> list[Segment]:
    """Segments from the `consumers` module configuration. No `behavior` and no `segments` gives the single
    `legacy` segment covering everyone (Crucible 3.0)."""
    beh, segs = cfg.get("behavior"), cfg.get("segments")
    if beh and segs:
        raise ProtocolError("consumers: configure either `behavior` or `segments`, not both")
    raw = [dict(s) for s in segs] if segs else [{"name": "all", "share": 1.0, **dict(beh or {})}]
    out: list[Segment] = []
    names = set()
    for i, s in enumerate(raw):
        unknown = set(s) - {"name", "share", "model", "params"}
        if unknown:
            raise ProtocolError(f"consumers: unknown segment keys {sorted(unknown)}")
        mname = s.get("model", "legacy")
        model = BEHAVIOR_MODELS.get(mname)
        if model is None:
            raise ProtocolError(f"consumers: unknown behaviour model {mname!r} (known: {sorted(BEHAVIOR_MODELS)})")
        over = dict(s.get("params") or {})
        allowed = set(model.defaults) | set(COMMON_PARAMS) | set(POPULATION_PARAMS)
        if set(over) - allowed:
            raise ProtocolError(f"consumers: model {mname!r} has no parameters {sorted(set(over) - allowed)} "
                                f"(it has {sorted(allowed)})")
        share = float(s.get("share", 1.0))
        if not (share > 0 and math.isfinite(share)):
            raise ProtocolError("consumers: segment shares must be positive")
        params = {k: cfg[k] for k in COMMON_PARAMS}
        params.update(model.defaults)
        params.update({k: v for k, v in over.items() if k not in POPULATION_PARAMS})
        params["personas"] = cfg["personas"]
        for k in ("rate_per_day", "wtp_mult", "loyalty"):
            if k in over and not (isinstance(over[k], (list, tuple)) and len(over[k]) == 2 and over[k][0] <= over[k][1]):
                raise ProtocolError(f"consumers: segment parameter {k} must be a [low, high] range")
        bs = over.get("budget_share")
        if bs is not None and not 0 < float(bs) <= 1:
            raise ProtocolError("consumers: budget_share must be in (0, 1]")
        name = str(s.get("name") or f"seg{i}")
        if name in names:
            raise ProtocolError(f"consumers: duplicate segment name {name!r}")
        names.add(name)
        out.append(Segment(i, name, share, model, params,
                           {k: list(over[k]) for k in ("rate_per_day", "wtp_mult", "loyalty") if k in over},
                           None if bs is None else float(bs)))
    if len(out) > 100:
        raise ProtocolError("consumers: at most 100 segments")
    tot = sum(s.share for s in out)
    for s in out:
        s.share = s.share / tot
    if sum(s.budget_share or 0.0 for s in out) > 1.0 + 1e-9:
        raise ProtocolError("consumers: budget shares add up to more than 1")
    return out


def budget_envelopes(segments: list[Segment]) -> tuple[list[int], list[int]] | None:
    """Spending envelopes: (envelope of each segment, parts per million of the household balance each envelope
    may spend in an hour). None when no segment sets `budget_share` (one pooled household budget, as in 3.0).
    Segments without a share draw together on what the others leave."""
    if not any(s.budget_share is not None for s in segments):
        return None
    env_of, ppm = [], []
    rest = None
    for s in segments:
        if s.budget_share is None:
            if rest is None:
                rest = len(ppm)
                ppm.append(0)
            env_of.append(rest)
        else:
            env_of.append(len(ppm))
            ppm.append(int(s.budget_share * 1_000_000))
    if rest is not None:
        ppm[rest] = max(0, 1_000_000 - sum(ppm))
    return env_of, ppm


def segment_counts(segments: list[Segment], n: int) -> list[int]:
    """Exact segment sizes for a population of n (largest remainders)."""
    exact = [s.share * n for s in segments]
    counts = [int(math.floor(x)) for x in exact]
    order = sorted(range(len(segments)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


# ====================================================================== configuration: latent and derived demand
@dataclass
class Demand:
    latent: dict           # need -> base active fraction
    edges: list            # [{"src", "dst", "weight", "scale", "lag_days"}]
    activation_rate: float
    decay_rate: float

    @property
    def max_lag(self) -> int:
        return max([e["lag_days"] for e in self.edges] + [0])

    def base(self, need: str) -> float:
        return float(self.latent.get(need, 1.0))

    def targets(self, verified: list[float]) -> dict[str, dict]:
        """Target active fraction of every need that has a base below 1 or an incoming edge, from the verified
        successful volume (one number per edge, in the order of `edges`):
        clip(base + sum_src weight * (1 - exp(-volume_src / scale)), 0, 1)."""
        lift: dict[str, float] = {}
        for e, v in zip(self.edges, verified):
            lift[e["dst"]] = lift.get(e["dst"], 0.0) + e["weight"] * (1.0 - math.exp(-max(0.0, float(v)) / e["scale"]))
        out = {}
        for need in sorted(set(self.latent) | set(lift)):
            b = self.base(need)
            out[need] = {"base": b, "lift": lift.get(need, 0.0),
                         "target": min(1.0, max(0.0, b + lift.get(need, 0.0)))}
        return out


def build_demand(cfg: dict) -> Demand | None:
    """Latent / derived demand from the module configuration; None when neither is configured (3.0: every
    consumer has its need forever)."""
    latent = dict(cfg.get("latent") or {})
    derived = list(cfg.get("derived") or [])
    if not latent and not derived:
        return None
    for need, b in latent.items():
        if not (isinstance(b, (int, float)) and 0.0 <= float(b) <= 1.0):
            raise ProtocolError(f"consumers: latent[{need!r}] must be a fraction in [0, 1]")
    edges = []
    for e in derived:
        unknown = set(e) - {"src", "dst", "weight", "scale", "lag_days"}
        if unknown or "src" not in e or "dst" not in e:
            raise ProtocolError(f"consumers: a derived-demand edge needs src and dst (bad keys: {sorted(unknown)})")
        w, sc, lag = float(e.get("weight", 1.0)), float(e.get("scale", 1000.0)), int(e.get("lag_days", 0))
        if not (math.isfinite(w) and math.isfinite(sc) and sc > 0 and lag >= 0):
            raise ProtocolError("consumers: derived-demand edges need a finite weight, scale > 0 and lag_days >= 0")
        edges.append({"src": str(e["src"]), "dst": str(e["dst"]), "weight": w, "scale": sc, "lag_days": lag})
    rate = float(cfg.get("activation_rate", 0.2))
    decay = float(cfg.get("decay_rate", rate))
    if not (0 < rate <= 1 and 0 < decay <= 1):
        raise ProtocolError("consumers: activation_rate and decay_rate are fractions per day in (0, 1]")
    return Demand({str(k): float(v) for k, v in latent.items()}, edges, rate, decay)
