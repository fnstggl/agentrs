"""Market entry by prospective founders (module `entry`).

The 3.0 long runs went stale because firms that exited were never replaced: entrepreneurs (dynamics.py) only
enter emergent *capability* markets, so a vacated consumer market stayed empty. This module adds a population of
prospective founders: persistent simulated persons created at genesis with their own savings (the same genesis
endowment workers get) and heterogeneous traits

    risk        appetite: lowers the personal hurdle, raises the share of savings committed, undercuts on price
    competence  technical skill: decides how an implementation is obtained and at which reference tier, and
                whether the founder tests its code and prices from measured cost before launching
    informed    how much research it does: how many markets it looks at, whether it pays for the market survey,
                and how noisy its estimate is
    patience    how long it carries a loss-making venture before winding it up

Everything a founder does is an ordinary tool call by its own principal (`ctx.call(..., as_principal=founder)`):
it has no privileged action and no access to hidden state.

On its own schedule a founder that is not running a venture

1. **screens** markets from free public signals: the public format documents and the listings (which needs exist),
   active sellers with their prices and verified ratings (`market.browse`), which of them a customer finds on the
   first page of the public search engine, customer reviews on the public forum board, and the public company
   registry (who exited);
2. **researches** the most promising market: a well-informed founder pays for `research.survey` (the fee is
   debited from its own account) and reads the estimates of market size, willingness to pay, unserved share and
   the incumbents' shares and satisfaction; a badly informed founder uses a rule of thumb instead;
3. **estimates** monthly profit: customers it expects to win (unserved customers shared with the other sellers,
   plus dissatisfied or overcharged customers of incumbents) x request rate x its planned price, minus compute,
   multiplied by a personal noise factor that shrinks with `informed`;
4. **enters** when the estimate exceeds its personal hurdle on the capital at risk: it founds and registers a
   company, funds it from its savings, asks an existing lender for the shortfall if it is under-capitalised,
   obtains an implementation in a way that depends on its competence (writes a reference tier bounded by its
   competence, forks public open-source code, buys a code licence, or hires a developer from the labour pool),
   rents compute, deploys, lists and prices;
5. afterwards the company is an ordinary firm: it is registered with the same firm logic the incumbents use
   (a Tier-2 mind in open worlds, an `npcs` firm in legacy worlds) and is subject to the same recovery-or-exit and
   bankruptcy rules. The founder, as owner, reviews it: it tops the firm up, asks for a bridging loan, or winds
   it up when losses exceed what its patience tolerates. Failed entrants lose the capital they burned.

Entry is a function of state only, so it applies to any market: one an incumbent vacated, one that is badly
served, or one whose customers only recently appeared. Nothing guarantees success: saturated markets are not
entered by well-informed founders, and founders that misjudge a market fail.

The module does nothing in worlds that do not list it. Configuration (all optional):

    founders              number of founders drawn at genesis (default 6), or
    founder_specs         explicit list of {capital, risk, competence, informed, patience[, first_scan_day]}
    capital_cents         [lo, hi] savings at genesis
    risk / competence / informed / patience   [lo, hi] trait ranges
    scan_interval_days    mean days between a founder's looks at the market (default 3)
    first_scan_day        nobody looks before this day (default 10: by then the customers present at genesis
                          have found the incumbents, so the first looks do not mistake a young world for a gap)
    paused                start with entry paused (the operator tool `admin.entry_pause` switches it; this is how
                          matched forks with and without entry are made)
    hurdle                required expected monthly return on the capital at risk (default 0.25); the personal
                          hurdle is hurdle x (1.6 - risk)
    noise_sigma           log-normal estimate noise of a founder with informed = 0 (default 0.9)
    max_ventures, cooldown_days, resurvey_days, ...   see DEFAULTS
"""
from __future__ import annotations

import copy
import json
import math
import re

from ..core import schema as S
from ..core.canonical import to_jsonable
from ..core.errors import CrucibleError
from ..core.kernel import Ctx, Endpoint, Module, applier, process, register_module
from ..economy.needs import NEEDS, best_tier, tier_source

DAY = 86400
PHASES = ("scouting", "hiring", "building", "operating", "retired")
SURVEY_FEE = 25_000

DEFAULTS = {
    "founders": 6, "founder_specs": None,
    "capital_cents": [400_000, 1_600_000],
    "risk": [0.15, 0.9], "competence": [0.05, 0.95], "informed": [0.1, 0.95], "patience": [0.2, 0.95],
    "scan_interval_days": 3.0, "first_scan_day": 10.0, "scan_hour": 8, "paused": False,
    "hurdle": 0.25, "noise_sigma": 0.9,
    "attention": 4,                 # a founder with informed = 1 examines 1 + attention markets per look
    "screen_min": 0.45,             # minimum free-signal score before a market is researched
    "survey_min_informed": 0.35,    # founders below this never pay for the survey
    "resurvey_days": 30,            # a rejected market is not reconsidered sooner unless its sellers changed
    "prior_customers": 1500, "prior_requests_per_day": 15.0, "prior_price_cents": 3,
    "prior_cu_per_request": 150.0,  # compute units per request assumed before the code exists
    "write_min_competence": 0.2, "test_min_competence": 0.3, "prefer_own_code": 0.6,
    "min_runway_days": 10, "margin": 1.5,
    "max_ventures": 2, "cooldown_days": 20,
    "dev_salary_cents_month": 300_000, "hire_patience_days": 12,
    "loan_rate_month": 0.035, "loan_months": 6,
}


class _Abort(Exception):
    """A step of an entry failed; the founder unwinds what it had started."""

    def __init__(self, step: str):
        super().__init__(step)
        self.step = step


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _stem(need: str) -> str:
    part = need.split(".")[1] if "." in need else need
    s = re.sub(r"[^a-z0-9]", "", part.lower())[:6] or "svc"
    return s if s[0].isalpha() else "s" + s[:5]


@register_module
class Entry(Module):
    name = "entry"
    depends = ("identity", "bank", "code", "cloud", "market", "business")

    def install(self, world):
        world.state.declare_table("founders", index=("phase",))
        self.cfg = dict(DEFAULTS)
        self.cfg.update(self.config)
        # operator tool (host `admin`: the kernel refuses it for every in-world actor and hides it from them)
        world.endpoints["admin.entry_pause"] = Endpoint(
            "admin.entry_pause", "admin", S.obj({"paused": S.s_bool()}, ["paused"]), self._ep_pause,
            "Operator only: pause or resume market entry by founders (recorded intervention).", 0, False)

    def _ep_pause(self, ctx, paused):
        ctx.emit("entry.paused", {"paused": bool(paused)}, vis="internal")
        return {"paused": bool(paused)}

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        cfg = self.cfg
        w = ctx.world
        idm, bank = w.modules["identity"], w.modules["bank"]
        rng = w.rng_for("entry-genesis")
        specs = cfg["founder_specs"]
        if specs is None:
            specs = []
            for _ in range(int(cfg["founders"])):
                specs.append({"capital": int(rng.integers(int(cfg["capital_cents"][0]), int(cfg["capital_cents"][1]) + 1)),
                              **{k: round(float(rng.uniform(*cfg[k])), 3)
                                 for k in ("risk", "competence", "informed", "patience")}})
        n = max(1, len(specs))
        for i, s in enumerate(specs):
            person = idm.create_person(ctx, f"founder-{i:02d}", f"Founder {i}", "npc", {"role": "founder"})
            acct = bank.open_account(ctx, person, "checking")
            capital = int(s.get("capital", cfg["capital_cents"][0]))
            if capital > 0:
                bank.move(ctx, "acct_genesis", acct, capital, "endowment", "savings")
            first = float(s.get("first_scan_day", cfg["first_scan_day"] + cfg["scan_interval_days"] * i / n))
            ctx.emit("entry.founder", {"id": person, "rec": {
                "person": person, "account": acct, "idx": i, "capital0": capital,
                "traits": {k: round(float(s.get(k, 0.5)), 3) for k in ("risk", "competence", "informed", "patience")},
                "phase": "scouting", "next_scan": int(first * DAY), "venture": None, "ventures": [],
                "estimates": [], "considered": {}, "seen_exits": [],
                "stats": {"scans": 0, "surveys": 0, "research_cents": 0, "entries": 0, "exits": 0, "failed": 0,
                          "aborted": 0, "loans_asked": 0, "loans_granted": 0}}}, vis="internal")
        if cfg["paused"]:
            ctx.emit("entry.paused", {"paused": True}, vis="internal")
        ctx.schedule(ctx.now + int(cfg["scan_hour"]) * 3600, "entry.scan", {}, prio=76, period=DAY, key="entry.scan")

    # ------------------------------------------------------------------ helpers
    def act(self, ctx: Ctx, who: str, tool: str, args: dict):
        """One tool call by `who`; a refusal is recorded and returns None (exactly what the caller would see)."""
        try:
            return ctx.call(tool, args, as_principal=who)
        except CrucibleError as e:
            ctx.emit("entry.failed", {"who": who, "tool": tool, "error": e.as_dict()}, vis="internal")
            return None

    def must(self, ctx: Ctx, who: str, tool: str, args: dict):
        r = self.act(ctx, who, tool, args)
        if r is None:
            raise _Abort(tool)
        return r

    def _set(self, ctx: Ctx, fid: str, **changes) -> None:
        # the payload is a deep plain copy: nothing emitted may alias an object that is mutated later
        ctx.emit("entry.founder_update", {"id": fid, "changes": to_jsonable(copy.deepcopy(changes))}, vis="internal")

    def _rec(self, st, fid: str) -> dict:
        return copy.deepcopy(dict(st.tables["founders"][fid]))

    def _bump(self, ctx, fid, f, **inc) -> dict:
        stats = dict(f["stats"])
        for k, v in inc.items():
            stats[k] = stats.get(k, 0) + v
        self._set(ctx, fid, stats=stats)
        f["stats"] = stats
        return f

    def _balance(self, ctx, who, account) -> int:
        r = self.act(ctx, who, "bank.balance", {"account": account})
        return int(r["balance"]) if r and r.get("status") == "open" else 0

    def _interval(self, rng) -> int:
        return int(self.cfg["scan_interval_days"] * DAY * float(rng.uniform(0.7, 1.3)))

    def _fee(self, key: str, default: int) -> int:
        return int(self.world.spec.params.get(key, default))

    def _core_price(self, ctx, me) -> tuple[str | None, int]:
        provs = sorted((p["price_core_hour_cents"], p["provider"]) for p in (self.act(ctx, me, "cloud.providers", {}) or [])
                       if p["cores_free"] >= 1 and not p["in_outage"])
        return (provs[0][1], int(provs[0][0])) if provs else (None, 0)

    # ------------------------------------------------------------------ the scheduled look
    @process("scan")
    def pr_scan(self, ctx):
        st = ctx.state
        if st.meta_get("entry_paused", False):
            return
        for fid in st.ids("founders"):
            f = st.tables["founders"][fid]
            if f["phase"] == "retired" or ctx.now < f["next_scan"]:
                continue
            rng = ctx.rng("entry", fid)
            f = self._rec(st, fid)
            self._set(ctx, fid, next_scan=ctx.now + self._interval(rng))
            try:
                if f["phase"] == "scouting":
                    self._scout(ctx, fid, f, rng)
                else:
                    self._manage(ctx, fid, f, rng)
            except _Abort as a:
                self._unwind(ctx, fid, self._rec(st, fid), a.step)

    # ================================================================== 1-3: signals and the estimate
    def _known_needs(self, ctx, me) -> tuple[list[str], dict[str, list[str]]]:
        """Markets an outsider can know about (format documents, listings, the registry) and exited firms by
        sector."""
        names: set[str] = set()
        mods = ctx.world.modules
        if "docs" in mods:
            for d in self.act(ctx, me, "docs.list", {"scope": "public", "limit": 50}) or []:
                if str(d.get("title", "")).startswith("Format: "):
                    names.add(str(d["title"])[8:].strip())
        for l in self.act(ctx, me, "market.browse", {"kind": "api", "limit": 30}) or []:
            if l.get("need"):
                names.add(str(l["need"]))
        exited: dict[str, list[str]] = {}
        for status in ("bankrupt", "dissolved"):
            for c in self.act(ctx, me, "biz.registry", {"status": status, "limit": 100}) or []:
                if c.get("sector"):
                    names.add(str(c["sector"]))
                    exited.setdefault(str(c["sector"]), []).append(str(c["org"]))
        return sorted(n for n in names if n in NEEDS), exited

    def _reviews(self, ctx, me) -> dict[str, list[int]]:
        """Stars of recent public customer reviews by need (the market board of the forum)."""
        out: dict[str, list[int]] = {}
        if "forum" not in ctx.world.modules:
            return out
        for p in self.act(ctx, me, "forum.list", {"board": "market", "sort": "new", "limit": 30}) or []:
            if "review" not in p.get("tags", ()) or ctx.now - p["t"] > 21 * DAY:
                continue
            m = re.search(r"(\d)/5", p.get("snippet", ""))
            if not m:
                continue
            for tag in p["tags"]:
                if tag in NEEDS:
                    out.setdefault(tag, []).append(int(m.group(1)))
        return out

    def _screen(self, ctx, me, need, reviews, new_exits) -> dict:
        """Free signals for one market: sellers, prices, verified ratings, complaints, recent exits."""
        rows = [r for r in (self.act(ctx, me, "market.browse", {"need": need, "sort": "price", "limit": 10}) or [])
                if r["kind"] == "api"]
        sellers = [{"host": r["host"], "seller": r["seller"], "price": int(r["price_cents"]),
                    "rating": r["rating"], "ratings": int(r["ratings"])} for r in rows]
        n = len(sellers)
        # what a customer who searches for this kind of service finds: at most the first four sellers
        visible = None
        if "search" in ctx.world.modules and n:
            by_listing = {r["listing"]: r["host"] for r in rows}
            hosts = set(by_listing.values())
            found: list[str] = []
            q = " ".join(NEEDS[need].keywords[:3])
            for r in (self.act(ctx, me, "search.query", {"q": q, "limit": 10}) or {}).get("results", []):
                if r.get("sponsored"):
                    continue
                h = by_listing.get(r.get("ref")) or next((x for x in r.get("hosts", ()) if x in hosts), None)
                if h is not None and h not in found:
                    found.append(h)
            visible = min(4, len(found))
        rated = [(s["rating"], s["ratings"]) for s in sellers if s["rating"] is not None and s["ratings"] > 0]
        mean_rating = sum(r * k for r, k in rated) / sum(k for _, k in rated) if rated else None
        stars = reviews.get(need, [])
        complaints = sum(1 for s in stars if s <= 2) / len(stars) if len(stars) >= 2 else 0.0
        # three stars is what customers call "okay": only ratings below that read as a quality gap
        qgap = _clip((3.0 - mean_rating) / 2.0, 0.0, 1.0) if mean_rating is not None else 0.1
        x = min(2, new_exits)
        score = 2.0 + 0.5 * x if n == 0 else 0.6 / n + 0.8 * qgap + 0.6 * complaints + 0.5 * x \
            + (0.5 if visible == 0 else 0.0)                 # sellers nobody can find leave the field open
        return {"need": need, "n": n, "visible": visible, "sellers": sellers,
                "p_min": min((s["price"] for s in sellers), default=None),
                "mean_rating": mean_rating, "complaints": round(complaints, 3), "qgap": round(qgap, 3),
                "new_exits": new_exits, "score": round(score, 3)}

    def _estimate(self, f, sig, survey, core_price, rng) -> dict:
        """Expected monthly profit of entering this market, as this founder sees it."""
        cfg, tr = self.cfg, f["traits"]
        n, p_min = sig["n"], sig["p_min"]
        if survey:
            N = float(survey.get("consumers_est") or 0)
            # consumers who currently have the need, when the survey reports it (latent / derived demand)
            for k in ("active_consumers_est", "consumers_with_need_est"):
                if survey.get(k) is not None:
                    N = float(survey[k])
            rate = float(survey.get("requests_per_day_est") or 0.0)
            wtp = float(survey.get("median_willingness_to_pay_cents") or 0.0) or None
            u = float(survey.get("unserved_share_est") or 0.0)
            offs = survey.get("offerings") or {}
        else:   # an outsider's rule of thumb
            N, rate = float(cfg["prior_customers"]), float(cfg["prior_requests_per_day"])
            wtp = p_min * 1.25 if p_min else None
            u = 1.0 if n == 0 else min(0.6, 0.1 + 0.5 * sig["qgap"] + 0.3 * sig["complaints"])
            offs = {}
        if n:
            price = p_min if tr["risk"] < 0.7 else max(1, p_min - 1)      # bold founders undercut by a cent
            if wtp:
                price = min(price, max(1, int(round(wtp))))
        else:
            price = max(1, int(0.8 * wtp)) if wtp else int(cfg["prior_price_cents"])
        unserved = N * u
        # customers looking for a provider choose among the sellers they can find (at most four per search); with
        # four or more ahead of it a newcomer is not always on that list
        rivals = n if sig.get("visible") is None else min(n, sig["visible"])
        win = unserved * 0.7 / (1 + rivals) * min(1.0, 4.0 / (n + 1))
        steal = 0.0
        for s in sig["sellers"]:
            o = offs.get(s["host"])
            if not o:
                continue
            adv = _clip(0.5 - float(o.get("satisfaction_est", 0.5)), 0.0, 0.5) * 1.2 \
                + _clip((s["price"] - price) / max(s["price"], 1), 0.0, 1.0) * 0.3
            steal += N * float(o.get("share_est", 0.0)) * min(0.8, adv)
        sigma = cfg["noise_sigma"] * (1.0 - tr["informed"]) * (1.0 if survey else 1.5)
        noise = float(math.exp(rng.normal(0.0, sigma))) if sigma > 0 else 1.0
        customers = min(max(N, 1.0) * (3.0 if not survey else 1.0), (win + steal) * noise)
        elastic = _clip(2.0 - price / wtp, 0.0, 1.0) * 0.85 if wtp else 0.8
        req_day = customers * rate * elastic
        cores = max(1, int(math.ceil(req_day * 1.7 / 24.0 * cfg["prior_cu_per_request"] / 100_000)))
        cost_day = cores * core_price * 24
        rev_day = req_day * price
        monthly = 30.0 * (rev_day - cost_day) * 0.6            # the first month is a ramp
        return {"need": sig["need"], "t": 0, "n_sellers": n, "visible": sig.get("visible"), "p_min": p_min,
                "score": sig["score"],
                "survey": bool(survey), "market_est": round(N, 1), "unserved_est": round(unserved, 1),
                "win_est": round(win, 1), "steal_est": round(steal, 1), "noise": round(noise, 4),
                "customers_est": round(customers, 1), "price": int(price), "wtp_est": wtp,
                "rev_day": round(rev_day, 1), "cost_day": round(cost_day, 1), "monthly": round(monthly, 1)}

    def _plan_route(self, ctx, f, need, cash) -> dict | None:
        """How this founder would obtain an implementation (depends on its competence and on what is public)."""
        cfg, tr, me = self.cfg, f["traits"], f["person"]
        mods = ctx.world.modules
        own = None
        if tr["competence"] >= cfg["write_min_competence"]:
            own = {"route": "write", "tier": min(best_tier(need), int(tr["competence"] * len(NEEDS[need].tiers))),
                   "cost": 0}
        if own is not None and tr["competence"] >= cfg["prefer_own_code"]:
            return own
        if "search" in mods:        # public open-source code serving this format
            q = f"{need} {NEEDS[need].title}"
            for r in (self.act(ctx, me, "search.query", {"q": q, "kind": "repo", "limit": 5}) or {}).get("results", []):
                ls = self.act(ctx, me, "code.ls", {"repo": r["ref"]}) if r.get("ref") else None
                if not ls or ls["license"] != "open" or ls["visibility"] != "public":
                    continue
                if not any(x["path"] == "crucible.json" for x in ls["files"]):
                    continue
                man = self.act(ctx, me, "code.read", {"repo": r["ref"], "path": "crucible.json"})
                text = f"{ls['description']} {(man or {}).get('content', '')}"
                if need in text or NEEDS[need].title in text:
                    return {"route": "fork", "tier": None, "cost": 0, "src": r["ref"]}
        lic = sorted((x["price_cents"], x["listing"]) for x in
                     (self.act(ctx, me, "market.browse", {"kind": "code", "need": need, "limit": 10}) or [])
                     if x["price_cents"] <= 0.25 * cash)
        if lic:
            return {"route": "licence", "tier": None, "cost": int(lic[0][0]), "src": lic[0][1]}
        if own is not None:
            return own
        if all(m in mods for m in ("jobs", "contracts", "npcs")):
            salary = int(cfg["dev_salary_cents_month"])
            for j in self.act(ctx, me, "jobs.openings", {"role": "developer", "limit": 10}) or []:
                salary = max(salary, int(j["salary_cents_month"]))     # match the going rate
            return {"route": "hire", "tier": None, "cost": salary, "salary": salary}
        return None

    def _scout(self, ctx, fid, f, rng) -> None:
        cfg, tr, me, st = self.cfg, f["traits"], f["person"], ctx.state
        f = self._bump(ctx, fid, f, scans=1)
        cash = self._balance(ctx, me, f["account"])
        prov, core_price = self._core_price(ctx, me)
        fee = self._fee("registration_fee_cents", 50_000) + self._fee("listing_fee_cents", 500)
        min_cap = fee + int(cfg["min_runway_days"]) * core_price * 24
        if prov is None:
            return
        if cash < min_cap:
            self._set(ctx, fid, phase="retired", retired_reason="savings too small to start a company")
            return
        needs, exited = self._known_needs(ctx, me)
        seen = set(f["seen_exits"])
        new_exits = {s: sum(1 for o in orgs if o not in seen) for s, orgs in exited.items()}
        all_exits = sorted({o for orgs in exited.values() for o in orgs})
        if all_exits != f["seen_exits"]:
            self._set(ctx, fid, seen_exits=all_exits[-200:])
        if not needs:
            return
        k = 1 + int(tr["informed"] * cfg["attention"])
        look = sorted(needs[int(i)] for i in rng.permutation(len(needs))[:k])
        reviews = self._reviews(ctx, me)
        sigs = sorted((self._screen(ctx, me, need, reviews, new_exits.get(need, 0)) for need in look),
                      key=lambda s: (-s["score"], s["need"]))
        considered = dict(f["considered"])
        for sig in sigs:
            need = sig["need"]
            if sig["score"] < cfg["screen_min"]:
                break
            last = considered.get(need)
            if last and last["n"] == sig["n"] and ctx.now - last["t"] < cfg["resurvey_days"] * DAY \
                    and not sig["new_exits"]:
                continue                        # looked recently, said no, nothing visible has changed
            survey = None
            if tr["informed"] >= cfg["survey_min_informed"] and "consumers" in ctx.world.modules \
                    and cash - SURVEY_FEE >= min_cap:
                survey = self.act(ctx, me, "research.survey", {"need": need, "account": f["account"]})
                if survey is not None:
                    cash -= SURVEY_FEE
                    f = self._bump(ctx, fid, f, surveys=1, research_cents=SURVEY_FEE)
            est = self._estimate(f, sig, survey, core_price, rng)
            route = self._plan_route(ctx, f, need, cash)
            willing = int(cash * (0.4 + 0.5 * tr["risk"]))
            setup = fee + (route["cost"] if route else 0)
            runway = int((20 + 20 * tr["patience"]) * core_price * 24)
            commit = max(0, min(willing, setup + runway) - self._fee("registration_fee_cents", 50_000))
            at_risk = commit + self._fee("registration_fee_cents", 50_000)
            hurdle = cfg["hurdle"] * (1.6 - tr["risk"])
            enter = bool(route is not None and est["monthly"] >= hurdle * at_risk
                         and at_risk >= setup + int(cfg["min_runway_days"]) * core_price * 24)
            est.update(t=ctx.now, route=route["route"] if route else None, at_risk=at_risk,
                       hurdle=round(hurdle, 4), enter=enter)
            ctx.emit("entry.estimate", {"founder": fid, **to_jsonable(est)}, vis="internal")
            considered[need] = {"t": ctx.now, "n": sig["n"]}
            self._set(ctx, fid, estimates=(f["estimates"] + [est])[-12:], considered=considered)
            if enter:
                short = max(0, setup + runway - at_risk)
                self._found(ctx, fid, self._rec(st, fid), need, est, route, commit, short, rng)
            return                              # one market per look

    # ================================================================== 4: entering
    def _found(self, ctx, fid, f, need, est, route, commit, short, rng) -> None:
        me, st = f["person"], ctx.state
        k = len(f["ventures"]) + 1
        host = f"{_stem(need)}{f['idx']:02d}{'abcdefghij'[min(k, 10) - 1]}"
        while st.has("hosts", host) or st.find_ids("orgs", "handle", host + "co"):
            host += "x"
        fee = self._fee("registration_fee_cents", 50_000)
        v = {"need": need, "host": host, "org": None, "account": None, "repo": None, "listing": None,
             "route": route["route"], "tier": route.get("tier"), "price": int(est["price"]), "t": ctx.now,
             "invested": 0, "stage": "founding", "wtp_est": est.get("wtp_est"), "est_monthly": est["monthly"],
             "loans": [], "topups": 0}
        org = self.must(ctx, me, "id.create_org", {"handle": host + "co", "name": f"{host.title()} Services",
                                                   "kind": "company"})["org"]
        v["org"] = org
        self._set(ctx, fid, venture=v, phase="building")
        self.must(ctx, me, "biz.register", {"org": org, "account": f["account"], "sector": need})
        v["invested"] += fee
        self._set(ctx, fid, venture=v)
        v["account"] = self.must(ctx, me, "bank.open_account", {"owner": org})["account"]
        self.must(ctx, me, "bank.transfer", {"from_account": f["account"], "to": v["account"], "amount": int(commit),
                                             "memo": f"capital for {org}"})
        v["invested"] += int(commit)
        self._set(ctx, fid, venture=v)
        f = self._bump(ctx, fid, f, entries=1)
        ctx.emit("entry.entered", {"founder": fid, "org": org, "need": need, "host": host, "route": v["route"],
                                   "invested": v["invested"], "est_monthly": est["monthly"]}, vis="internal")
        if short > 0:
            v = self._seek_loan(ctx, fid, f, v, short, "start-up capital", rng)
        from ..worlds.digital_civilization import service_repo_files
        if v["route"] == "write":
            v["repo"] = self.must(ctx, me, "code.create_repo", {"name": "svc", "owner": org,
                                                                "description": f"{need} service"})["repo"]
            self.must(ctx, me, "code.commit", {"repo": v["repo"], "files": service_repo_files(need, v["tier"], v["price"]),
                                               "message": f"implement {need}"})
        elif v["route"] == "fork":
            v["repo"] = self.must(ctx, me, "code.fork", {"repo": route["src"], "name": "svc", "owner": org})["repo"]
        elif v["route"] == "licence":
            self.must(ctx, me, "market.buy", {"listing": route["src"], "account": v["account"]})
            held = [g for g in (self.act(ctx, me, "id.me", {}) or {}).get("grants_held", [])
                    if g["cap"] == "code.read" and g["scope"] not in ("*", None)]
            mine = [g for g in (self.act(ctx, org, "id.me", {}) or {}).get("grants_held", [])
                    if g["cap"] == "code.read" and g["scope"] not in ("*", None)]
            src = sorted(g["scope"] for g in held + mine)
            if not src:
                raise _Abort("licence")
            v["repo"] = self.must(ctx, me, "code.fork", {"repo": src[-1], "name": "svc", "owner": org})["repo"]
        elif v["route"] == "hire":
            v["repo"] = self.must(ctx, me, "code.create_repo", {"name": "svc", "owner": org,
                                                                "description": f"{need} service"})["repo"]
            files = service_repo_files(need, 0, v["price"])
            del files["service.py"]                     # the developer writes it
            self.must(ctx, me, "code.commit", {"repo": v["repo"], "files": files, "message": "project skeleton"})
            v.update(stage="hiring", salary=int(route["salary"]), offer=None, dev=None, hire_t0=ctx.now)
            v["posting"] = self.must(ctx, me, "jobs.post", {
                "org": org, "title": "Developer wanted", "role": "developer", "salary_cents_month": v["salary"],
                "description": f"Implement a {need} service."})["posting"]
            self._set(ctx, fid, venture=v, phase="hiring")
            return
        self._set(ctx, fid, venture=v)
        self._launch(ctx, fid, f, v, rng)

    def _launch(self, ctx, fid, f, v, rng) -> None:
        """Test (if the founder is able to), rent compute, deploy, list, and hand the firm to the firm logic."""
        cfg, tr, me, org, need = self.cfg, f["traits"], f["person"], v["org"], v["need"]
        man = json.loads(self.must(ctx, me, "code.read", {"repo": v["repo"], "path": "crucible.json"})["content"])
        eps = man.get("endpoints") or {}
        if not eps:
            raise _Abort("manifest")
        ep = "solve" if "solve" in eps else sorted(eps)[0]
        prov, core_price = self._core_price(ctx, me)
        if prov is None:
            raise _Abort("no compute")
        price = int(v["price"])
        if tr["competence"] >= cfg["test_min_competence"]:
            # a founder who can read code tries it on a request written from the public format before paying for
            # compute, and prices from the measured cost
            inp, _ = NEEDS[need].generate(rng, 0.6)
            r = self.must(ctx, me, "code.run", {"repo": v["repo"],
                                                "target": f"{man.get('entry', 'service.py')}:{eps[ep].get('handler', ep)}",
                                                "input": to_jsonable(inp)})
            if not r["ok"]:
                raise _Abort("the implementation failed its own test")
            unit = r["cpu_ms"] * NEEDS[need].work_scale / 100_000 * core_price
            floor = max(1, int(math.ceil(unit * cfg["margin"])))
            if v.get("wtp_est") and floor > 2 * v["wtp_est"]:
                raise _Abort("cannot serve at a price customers would pay")
            price = max(price, floor)
            v["unit_cost"] = round(unit, 4)
        inst = self.must(ctx, me, "cloud.rent", {"provider": prov, "cores": 1, "account": v["account"], "owner": org})
        d = self.must(ctx, me, "cloud.deploy", {"repo": v["repo"], "host": v["host"], "instance": inst["instance"]})
        if d.get("status") != "live":
            raise _Abort("deploy")
        nd = NEEDS[need]
        v["listing"] = self.must(ctx, me, "market.list", {
            "kind": "api", "title": f"{v['host'].title()}: {nd.title}",
            "description": f"{nd.title}. Call {v['host']}.{ep}. Keywords: {' '.join(nd.keywords)}",
            "price_cents": price, "seller": org, "host": v["host"], "endpoint": ep, "need": need})["listing"]
        v.update(price=price, stage="operating", launched=ctx.now)
        self._attach_firm_logic(ctx, f, v, rng)
        if "forum" in ctx.world.modules:
            self.act(ctx, me, "forum.post", {"board": "market", "title": f"New: {v['host']}.crucible",
                                             "body": f"{nd.title} at {price} cents per request: {v['host']}.crucible",
                                             "tags": ["launch"]})
        self._set(ctx, fid, venture=v, phase="operating")
        ctx.emit("entry.launched", {"founder": fid, "org": org, "need": need, "host": v["host"],
                                    "listing": v["listing"], "route": v["route"], "tier": v.get("tier"),
                                    "price": price}, vis="internal")

    def _attach_firm_logic(self, ctx, f, v, rng) -> None:
        """From here on the company is run by the same firm logic as the incumbents (reused, not copied)."""
        mods = ctx.world.modules
        civ = mods.get("civ")
        if "society" in mods and (civ is None or civ.config.get("firm_minds")):
            soc = mods["society"]
            traits = soc.draw_traits(rng)
            traits["risk"] = round(float(f["traits"]["risk"]), 3)
            soc.add_mind(ctx, v["org"], "firm", v["org"], f["person"], traits, role="service",
                         extra={"routines": {"capacity": {}, "pricing": {"step": 0.06}, "staffing": {},
                                             "forum_reader": {}}})
            v["logic"] = "society"
        elif "npcs" in mods:
            mods["npcs"].add_firm(ctx, v["org"], f["person"], v["need"], v["host"], v["listing"], v["repo"],
                                  int(v["tier"] or 0), {"outsource": False})
            v["logic"] = "npcs"
        else:
            v["logic"] = None

    def _seek_loan(self, ctx, fid, f, v, amount, why, rng) -> dict:
        """Ask an existing lender (found in the public registry) for a term loan to the company. The lender
        decides by its own rules; the founder learns the answer from the contract's status."""
        cfg, me, org = self.cfg, f["person"], v["org"]
        if "contracts" not in ctx.world.modules or amount < 10_000:
            return v
        lenders = sorted(c["org"] for c in (self.act(ctx, me, "biz.registry", {"status": "registered", "limit": 100}) or [])
                         if c.get("sector") == "finance")
        if not lenders:
            return v
        lender = lenders[int(rng.integers(0, len(lenders)))]
        if lender not in v.get("bank_read", []):
            if self.act(ctx, me, "id.grant", {"grantee": lender, "cap": "bank.read", "owner": org}) is None:
                return v
            v["bank_read"] = sorted(set(v.get("bank_read", [])) | {lender})
        n = int(cfg["loan_months"])
        inst = int(math.ceil(amount * (1.0 + cfg["loan_rate_month"] * n) / n))
        r = self.act(ctx, me, "contracts.propose", {
            "parties": [org, lender], "title": f"Loan to {org}", "text": why, "as_org": org, "expires_in_s": 3 * DAY,
            "clauses": [{"type": "payment", "from": lender, "to": org, "amount": int(amount), "start_in_s": 0},
                        {"type": "payment", "from": org, "to": lender, "amount": inst, "every_s": 30 * DAY,
                         "count": n, "start_in_s": 30 * DAY}]})
        if r is None:
            return v
        v["loans"] = list(v["loans"]) + [{"contract": r["contract"], "amount": int(amount), "t": ctx.now,
                                          "status": "asked", "why": why}]
        self._set(ctx, fid, venture=v)
        self._bump(ctx, fid, f, loans_asked=1)
        ctx.emit("entry.loan", {"founder": fid, "org": org, "lender": lender, "contract": r["contract"],
                                "amount": int(amount), "why": why}, vis="internal")
        return v

    # ================================================================== 5: owning a company
    def _manage(self, ctx, fid, f, rng) -> None:
        st = ctx.state
        v = f["venture"]
        if v is None:
            self._set(ctx, fid, phase="scouting")
            return
        me, org = f["person"], v["org"]
        comp = self.act(ctx, me, "biz.company", {"org": org}) if org else None
        if comp is None or comp["status"] != "registered":
            self._close(ctx, fid, f, v, comp["status"] if comp else "never registered", rng)
            return
        v = self._loan_answers(ctx, fid, f, v)
        if v["stage"] == "hiring":
            self._hiring(ctx, fid, f, v, rng)
        elif v["stage"] == "building":
            self._building(ctx, fid, f, v, rng)
        elif v["stage"] == "operating":
            self._review(ctx, fid, f, v, rng)
        else:
            raise _Abort(f"stalled in stage {v['stage']}")
        del st

    def _loan_answers(self, ctx, fid, f, v) -> dict:
        changed = False
        for L in v["loans"]:
            if L["status"] != "asked":
                continue
            c = self.act(ctx, f["person"], "contracts.view", {"contract": L["contract"]})
            if c is None:
                continue
            if c["status"] in ("active", "completed"):
                L["status"] = "granted"
                self._bump(ctx, fid, f, loans_granted=1)
                changed = True
            elif c["status"] != "proposed" or ctx.now > c["expires"]:
                L["status"] = "refused"
                changed = True
        if changed:
            self._set(ctx, fid, venture=v)
        return v

    def _hiring(self, ctx, fid, f, v, rng) -> None:
        cfg, me, org = self.cfg, f["person"], v["org"]
        if v.get("offer"):
            c = self.act(ctx, me, "contracts.view", {"contract": v["offer"]["contract"]})
            if c is not None and c["status"] == "active":
                v.update(stage="building", dev=v["offer"]["applicant"], dev_contract=v["offer"]["contract"],
                         hired=ctx.now, progress=0.0, build_t=ctx.now, offer=None)
                self.act(ctx, me, "jobs.close", {"posting": v["posting"]})
                self._set(ctx, fid, venture=v, phase="building")
                return
            if c is not None and c["status"] == "proposed" and ctx.now <= c["expires"]:
                return                                   # the applicant has not answered yet
            v["refused"] = sorted(set(v.get("refused", [])) | {v["offer"]["applicant"]})
            v["offer"] = None
            v["salary"] = int(v["salary"] * 1.15)        # the offer was refused: pay more
        if ctx.now - v["hire_t0"] > (cfg["hire_patience_days"] + 20 * f["traits"]["patience"]) * DAY \
                or v["salary"] > self._balance(ctx, me, v["account"]):
            raise _Abort("could not hire a developer")
        apps = sorted(a["application"] for a in (self.act(ctx, me, "jobs.applications", {"posting": v["posting"]}) or [])
                      if a["status"] in ("applied", "offered") and a["applicant"] not in v.get("refused", []))
        if apps:
            r = self.act(ctx, me, "jobs.offer", {"application": apps[0], "salary_cents_month": int(v["salary"]),
                                                 "title": "Developer"})
            if r is not None:
                v["offer"] = {"contract": r["contract"], "applicant": apps[0].split("|", 1)[1]}
        self._set(ctx, fid, venture=v)

    def _building(self, ctx, fid, f, v, rng) -> None:
        """The hired developer writes the service. Output per half-day and the reachable reference tier follow
        the developer's own skill with the same rule the labour pool uses for assigned issues; the developer
        commits it under its own name. The issue tracker is not used: finishing an assigned issue in
        `npcs._work` reads `it['id']`, a key `issues.assigned` does not return, and the resulting KeyError fails
        the `npcs.workers` process on every later run."""
        st, me = ctx.state, f["person"]
        if v.get("dev") is None:                        # the firm was already launched (write/fork/licence)
            raise _Abort("build")
        c = self.act(ctx, me, "contracts.view", {"contract": v["dev_contract"]})
        if c is None or c["status"] != "active":
            v.update(stage="hiring", dev=None, offer=None, hire_t0=ctx.now)
            v["posting"] = self.must(ctx, me, "jobs.post", {
                "org": v["org"], "title": "Developer wanted", "role": "developer",
                "salary_cents_month": int(v["salary"]), "description": f"Implement a {v['need']} service."})["posting"]
            self._set(ctx, fid, venture=v, phase="hiring")
            return
        worker = st.get("npc_workers", v["dev"]) if "npc_workers" in st.tables else None
        skill = float((worker or {}).get("skills", {}).get("developer", 0.3))
        halves = max(1, int((ctx.now - v["build_t"]) // (DAY // 2)))
        prog = float(v["progress"])
        for _ in range(halves):
            prog += (0.15 + skill) * float(rng.uniform(0.5, 1.5)) / 2.5
        v.update(progress=round(prog, 4), build_t=ctx.now)
        if prog < 1.0:
            self._set(ctx, fid, venture=v)
            return
        tiers = len(NEEDS[v["need"]].tiers)
        v["tier"] = min(tiers - 1, int(skill * tiers))
        self.must(ctx, v["dev"], "code.commit", {"repo": v["repo"], "files": {"service.py": tier_source(v["need"], v["tier"])},
                                                 "message": f"implement {v['need']}"})
        self._set(ctx, fid, venture=v)
        self._launch(ctx, fid, f, v, rng)

    def _review(self, ctx, fid, f, v, rng) -> None:
        """The owner's periodic look at its own company's books (its account, its service analytics)."""
        cfg, tr, me, org = self.cfg, f["traits"], f["person"], v["org"]
        if v.get("dev") and v.get("hired") is not None and ctx.now - v["hired"] > 31 * DAY:
            # the build is done and the first salary is paid: let the developer go with notice
            self.act(ctx, me, "contracts.terminate", {"contract": v["dev_contract"], "as_org": org})
            v["dev"] = None
        cash = self._balance(ctx, me, v["account"])
        status = self.act(ctx, me, "cloud.status", {"host": v["host"], "days": 8}) or {}
        today = ctx.now // DAY
        days = [d for d in status.get("days", []) if d["day"] < today][-7:]
        rev_day = sum(d["revenue_cents"] for d in days) / max(1, len(days)) if days else 0.0
        cost_day = sum(i["cores"] * i["price_core_hour_cents"] for i in (self.act(ctx, me, "cloud.instances", {}) or [])
                       if i["owner"] == org and i["status"] in ("running", "suspended")) * 24
        age = (ctx.now - v.get("launched", v["t"])) / DAY
        grace = 7 + 28 * tr["patience"]
        burn = cost_day - rev_day
        lost = 1.0 - cash / max(1, v["invested"])
        v["last"] = {"t": ctx.now, "cash": cash, "rev_day": round(rev_day, 1), "cost_day": cost_day}
        if age >= grace and burn > 0 and lost >= 0.3 + 0.5 * tr["patience"]:
            self._wind_up(ctx, fid, f, v, "losses beyond what the founder tolerates", rng)
            return
        if burn > 0 and cash < 5 * burn:
            mine = self._balance(ctx, me, f["account"])
            top = min(int(mine * (0.2 + 0.5 * tr["risk"])), int(14 * burn))
            asked = [L for L in v["loans"] if ctx.now - L["t"] < 20 * DAY]
            if (age < grace or rev_day >= 0.6 * cost_day) and top >= 3 * burn and v["topups"] < 2:
                if self.act(ctx, me, "bank.transfer", {"from_account": f["account"], "to": v["account"], "amount": top,
                                                       "memo": f"more capital for {org}"}) is not None:
                    v["invested"] += top
                    v["topups"] += 1
            elif rev_day > 0 and not asked:
                v = self._seek_loan(ctx, fid, f, v, int(30 * cost_day), "bridging loan against current sales", rng)
            elif age >= grace and cash < 2 * burn:
                self._wind_up(ctx, fid, f, v, "out of cash", rng)
                return
        self._set(ctx, fid, venture=v)

    # ------------------------------------------------------------------ leaving
    def _wind_up(self, ctx, fid, f, v, reason, rng) -> None:
        if self.act(ctx, f["person"], "biz.dissolve", {"org": v["org"]}) is None:
            self._set(ctx, fid, venture=v)
            return
        self._close(ctx, fid, f, v, f"dissolved by its founder: {reason}", rng)

    def _unwind(self, ctx, fid, f, step) -> None:
        """An entry step failed after the company existed: wind it up and take back what is left."""
        v = f["venture"]
        if v is None:
            self._set(ctx, fid, phase="scouting")
            return
        if v.get("posting") and v.get("stage") == "hiring":
            self.act(ctx, f["person"], "jobs.close", {"posting": v["posting"]})
        comp = self.act(ctx, f["person"], "biz.company", {"org": v["org"]}) if v.get("org") else None
        if comp is not None and comp["status"] == "registered":
            self.act(ctx, f["person"], "biz.dissolve", {"org": v["org"]})
        self._bump(ctx, fid, f, aborted=1)
        self._close(ctx, fid, self._rec(ctx.state, fid), v, f"entry aborted at {step}", ctx.rng("entry-unwind", fid))

    def _close(self, ctx, fid, f, v, reason, rng) -> None:
        cfg, me = self.cfg, f["person"]
        recovered = 0
        for row in self.act(ctx, me, "bank.statement", {"account": f["account"], "limit": 200}) or []:
            if row["kind"] == "distribution" and v.get("org") and row["memo"].endswith(v["org"]) and row["amount"] > 0 \
                    and row["t"] >= v["t"]:
                recovered += int(row["amount"])
        failed = recovered < v["invested"]
        done = {k: v.get(k) for k in ("org", "need", "host", "listing", "route", "tier", "price", "t", "invested",
                                      "launched", "logic", "est_monthly")}
        done.update(t_end=ctx.now, recovered=recovered, reason=str(reason)[:160],
                    outcome="failed" if failed else "exited_whole",
                    loans=[{"amount": L["amount"], "status": L["status"]} for L in v.get("loans", [])])
        ventures = f["ventures"] + [done]
        cash = self._balance(ctx, me, f["account"])
        _, core_price = self._core_price(ctx, me)
        min_cap = self._fee("registration_fee_cents", 50_000) + int(cfg["min_runway_days"]) * max(core_price, 1) * 24
        again = len(ventures) < int(cfg["max_ventures"]) and cash >= min_cap
        stats = dict(f["stats"], exits=f["stats"]["exits"] + 1, failed=f["stats"]["failed"] + (1 if failed else 0))
        self._set(ctx, fid, venture=None, ventures=ventures, stats=stats,
                  phase="scouting" if again else "retired",
                  next_scan=ctx.now + int(cfg["cooldown_days"] * DAY * float(rng.uniform(0.8, 1.2))),
                  **({} if again else {"retired_reason": "no further venture"}))
        ctx.emit("entry.exited", {"founder": fid, "org": v.get("org"), "need": v["need"], "reason": done["reason"],
                                  "invested": v["invested"], "recovered": recovered, "outcome": done["outcome"],
                                  "launched": v.get("launched") is not None,
                                  "age_days": round((ctx.now - v["t"]) / DAY, 2)}, vis="internal")

    # ------------------------------------------------------------------ appliers
    @applier("entry.founder")
    def ap_founder(self, ev):
        self.world.state.insert("founders", ev.payload["id"], dict(ev.payload["rec"]))

    @applier("entry.founder_update")
    def ap_founder_update(self, ev):
        self.world.state.patch("founders", ev.payload["id"], **ev.payload["changes"])

    @applier("entry.paused")
    def ap_paused(self, ev):
        self.world.state.meta_set("entry_paused", bool(ev.payload["paused"]))

    @applier("entry.estimate")
    def ap_estimate(self, ev):
        return None

    @applier("entry.entered")
    def ap_entered(self, ev):
        return None

    @applier("entry.launched")
    def ap_launched(self, ev):
        return None

    @applier("entry.exited")
    def ap_exited(self, ev):
        return None

    @applier("entry.loan")
    def ap_loan(self, ev):
        return None

    @applier("entry.failed")
    def ap_failed(self, ev):
        return None

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        for fid, f in st.tables["founders"].items():
            a = st.get("accounts", f["account"])
            if a is None or a["owner"] != fid:
                errs.append(f"founder {fid} does not own its account")
            if f["phase"] not in PHASES:
                errs.append(f"founder {fid} in unknown phase {f['phase']!r}")
            v = f["venture"]
            if f["phase"] in ("hiring", "building", "operating") and (v is None or not v.get("org")):
                errs.append(f"founder {fid} is {f['phase']} without a company")
            if f["phase"] in ("scouting", "retired") and v is not None:
                errs.append(f"founder {fid} is {f['phase']} but still holds a venture")
            if v is not None and v.get("org") and st.has("orgs", v["org"]) and \
                    not st.has("members", f"{v['org']}|{fid}"):
                errs.append(f"founder {fid} is not a member of its company {v['org']}")
            if f["stats"]["failed"] > f["stats"]["exits"] or f["stats"]["exits"] > f["stats"]["entries"]:
                errs.append(f"founder {fid} has inconsistent counters {f['stats']}")
            if len(f["ventures"]) != f["stats"]["exits"]:
                errs.append(f"founder {fid}: {len(f['ventures'])} closed ventures but {f['stats']['exits']} exits")
        return errs


# ====================================================================== evaluator-side summaries (not tools)
def entry_stats(world) -> dict:
    """Counts for reports: founders, companies founded, launched, still operating, exits and failures."""
    st = world.state
    if "founders" not in st.tables:
        return {"founders": 0, "entries": 0, "launched": 0, "operating": 0, "exits": 0, "failed": 0,
                "failure_rate": None, "ventures": []}
    rows, tot = [], {"scans": 0, "surveys": 0, "research_cents": 0, "entries": 0, "exits": 0, "failed": 0,
                     "aborted": 0, "loans_asked": 0, "loans_granted": 0}
    operating = launched = 0
    for fid, f in st.scan("founders"):
        for k in tot:
            tot[k] += f["stats"].get(k, 0)
        for v in f["ventures"]:
            launched += v.get("launched") is not None
            rows.append({"founder": fid, "status": v["outcome"], **{k: v.get(k) for k in (
                "org", "need", "route", "tier", "t", "t_end", "invested", "recovered", "reason", "launched")}})
        v = f["venture"]
        if v is not None:
            launched += v.get("launched") is not None
            operating += v.get("stage") == "operating"
            rows.append({"founder": fid, "status": v.get("stage"), **{k: v.get(k) for k in (
                "org", "need", "route", "tier", "t", "invested", "launched")}})
    return {"founders": st.count("founders"), **tot, "launched": int(launched), "operating": int(operating),
            "failure_rate": round(tot["failed"] / tot["entries"], 3) if tot["entries"] else None,
            "retired": len(st.find_ids("founders", "phase", "retired")),
            "ventures": sorted(rows, key=lambda r: (r["t"], r["founder"]))}


def entrant_orgs(world) -> set[str]:
    """Organisations founded through this module (closed and current)."""
    st = world.state
    out: set[str] = set()
    if "founders" in st.tables:
        for _, f in st.scan("founders"):
            out |= {v["org"] for v in f["ventures"] if v.get("org")}
            if f["venture"] is not None and f["venture"].get("org"):
                out.add(f["venture"]["org"])
    return out


__all__ = ["Entry", "entry_stats", "entrant_orgs", "DEFAULTS"]
