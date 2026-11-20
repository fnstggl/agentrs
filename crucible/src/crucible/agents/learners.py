"""Small online learners for multi-agent experiments (competition and cooperation).

Both learners start with a fixed, documented scripted prefix (found a company, deploy a strong service for a need,
list it) and then *learn* one daily decision from consequences they can observe:

* `PricingLearner`: daily price multiplier in {0.75, 0.9, 1.0, 1.15, 1.35}; reward = its service's revenue from
  `cloud.status` (observable analytics) minus nothing else.
* `AdLearner`: daily contribution to a *joint* advertising campaign promoting its own and a partner's service
  (a public good inside the world mechanics: sponsored results lead searching consumers to either host);
  reward = own revenue - own ad spend. Under an optional cost-sharing contract, the partner reimburses part of it.

Policy: softmax over a linear function of observable features, trained by REINFORCE with a running baseline
(learning rate `lr`). Parameters persist for the agent's whole lifetime (no resets between days); `weights()` is a
checkpoint. These are deliberately tiny: the experiment tests whether multi-agent learning through real world
mechanics is possible and measurable, not a strong learner.
"""
from __future__ import annotations

import numpy as np

from ..economy.needs import NEEDS, best_tier, tier_source
from .scripted import ScriptPolicy, call, cheapest_rival, ok, service_files

DAY = 86400


class _Softmax:
    def __init__(self, n_feat, n_act, lr, rng):
        self.W = np.zeros((n_act, n_feat))
        self.lr = lr
        self.rng = rng
        self.baseline = 0.0
        self.history = []

    def act(self, phi):
        z = self.W @ phi
        p = np.exp(z - z.max())
        p /= p.sum()
        a = int(self.rng.choice(len(p), p=p))
        return a, p

    def update(self, phi, a, p, reward):
        adv = reward - self.baseline
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        g = -p[:, None] * phi[None, :]
        g[a] += phi
        self.W += self.lr * adv * g
        self.history.append({"a": a, "r": float(reward), "adv": float(adv), "p": p.round(3).tolist()})


class _LaunchMixin:
    def launch(self, suffix, need, capital, price_factor=0.95):
        r = yield call("market.browse", need=need, sort="price", limit=10)
        price = cheapest_rival(r["result"] if ok(r) else [], need, price_factor)
        org, acct, _ = yield from self.found_company(suffix, capital)
        if acct is None:
            return None
        r = yield call("code.create_repo", name=need.replace(".", "-"), owner=org)
        repo = r["result"]["repo"]
        yield call("code.commit", repo=repo, files=service_files(need, tier_source(need, best_tier(need)), price))
        svc = yield from self.launch_service(org, acct, repo, f"{self.handle[:12]}-{need.split('.')[1][:6]}", need, price)
        if svc is None:
            return None
        svc.update(org=org, acct=acct, repo=repo, need=need, price=price)
        return svc

    def day_revenue(self, svc):
        r = yield call("cloud.status", host=svc["host"], days=2)
        if not ok(r):
            return 0.0, {}
        days = r["result"]["days"]
        prev = days[-2] if len(days) > 1 else {}
        return float(prev.get("revenue_cents", 0)), prev


class PricingLearner(_LaunchMixin, ScriptPolicy):
    name = "pricing_learner"
    kind = "rl"
    LEVELS = (1, 2, 3, 4, 5, 6, 8)   # absolute price levels in cents (per-request prices are a few cents)

    def __init__(self, need, capital=400_000, lr=0.05, seed=0):
        self.need, self.capital, self.lr, self.seed = need, capital, lr, seed

    def script(self):
        svc = yield from self.launch("pl", self.need, self.capital)
        if svc is None:
            return
        self.svc = svc
        self.pol = _Softmax(4, len(self.LEVELS), self.lr, np.random.default_rng(self.seed))
        prev_rev, prev = None, None
        while True:
            yield from self.wait(23)
            rev, day = yield from self.day_revenue(svc)
            r = yield call("market.browse", need=self.need, sort="price", limit=10)
            others = [x["price_cents"] for x in (r["result"] if ok(r) else []) if x["host"] != svc["host"]]
            rel = svc["price"] / max(1, min(others)) if others else 1.0
            phi = np.array([1.0, np.tanh((rev - (prev_rev or rev)) / 2000.0), np.tanh(np.log(rel)),
                            min(1.0, day.get("overload", 0) / 100.0)])
            if prev is not None:
                self.pol.update(prev[0], prev[1], prev[2], rev / 1000.0)
            a, p = self.pol.act(phi)
            new_price = self.LEVELS[a]
            if new_price != svc["price"] and svc.get("listing"):
                rr = yield call("market.update", listing=svc["listing"], price_cents=new_price)
                if ok(rr):
                    svc["price"] = new_price
            self.prices = getattr(self, "prices", []) + [svc["price"]]
            prev, prev_rev = (phi, a, p), rev

    def weights(self):
        return self.pol.W.tolist() if hasattr(self, "pol") else None


class AdLearner(_LaunchMixin, ScriptPolicy):
    """Public-goods player: daily budget for a joint campaign that advertises both partners' hosts."""
    name = "ad_learner"
    kind = "rl"
    BUDGETS = (0, 3_000)    # pause, or contribute a 3,000-cent daily budget

    def __init__(self, need, partner_handle, capital=400_000, lr=0.05, seed=0):
        self.need, self.partner, self.capital, self.lr, self.seed = need, partner_handle, capital, lr, seed

    def script(self):
        svc = yield from self.launch("ad", self.need, self.capital)
        if svc is None:
            return
        self.svc = svc
        self.pol = _Softmax(3, len(self.BUDGETS), self.lr, np.random.default_rng(self.seed))
        partner_host = None
        campaign = None
        self._active = False
        prev = None
        self.contrib = []
        while True:
            yield from self.wait(23)
            if partner_host is None:
                r = yield call("market.browse", kind="api", limit=30)
                for x in (r["result"] if ok(r) else []):
                    if x["seller"].startswith(f"o:{self.partner}-"):
                        partner_host = x["host"]
            rev, day = yield from self.day_revenue(svc)
            spent = 0
            if campaign:
                r = yield call("ads.stats", campaign=campaign)
                if ok(r):
                    total = int(r["result"][0]["total_spent"])
                    spent = total - getattr(self, "_last_total", 0)
                    self._last_total = total
            reward = (rev - spent) / 1000.0
            phi = np.array([1.0, np.tanh(rev / 5000.0), float(partner_host is not None)])
            if prev is not None:
                self.pol.update(prev[0], prev[1], prev[2], reward)
            a, p = self.pol.act(phi)
            budget = self.BUDGETS[a]
            self.contrib.append(budget)
            hosts = [svc["host"]] + ([partner_host] if partner_host else [])
            if campaign is None and budget > 0:
                kw = sorted(set(NEEDS[self.need].keywords[:2]))
                r = yield call("ads.create_campaign", account=svc["acct"], keywords=kw, bid_cents=1,
                               daily_budget_cents=budget, title="Two great services", text="Joint campaign",
                               hosts=hosts, ref=svc.get("listing") or "")
                if ok(r):
                    campaign = r["result"]["campaign"]
                    self._active = True
            elif campaign is not None:
                if budget == 0 and self._active:
                    yield call("ads.pause", campaign=campaign)
                    self._active = False
                elif budget > 0 and not self._active:
                    yield call("ads.resume", campaign=campaign)
                    self._active = True
            prev = (phi, a, p)
