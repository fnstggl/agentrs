"""Generated startup worlds and a scripted reference strategy for a recurring-revenue objective (Gate K note).

`startup_world(seed, ...)` draws a world: one public need format, a compute provider, two or three incumbent SaaS
businesses running weaker solvers of that need at drawn prices, and a population of independent customers
(`patrons`) with their own money, willingness to pay and quality requirements. Nothing in the world knows about
the reference strategy.

`ReferenceSaaS` is a scripted participant that acts only through the ordinary tools, as one person, and reads only
what its tools return (catalogue, its own subscriptions, its own instance metrics and support tickets): it founds
a company, commits a solver of the need, rents compute and deploys it, launches a SaaS product with a monthly plan
(14-day trial) and an annual plan priced at the incumbents' median, then runs the business - answering support
tickets and adding cores when its service reports overload.

`objective(world, org, target_arr_usd)` is the feasible recurring-revenue objective: qualifying ARR (12 x MRR of
paid-up subscriptions from independent customers, `crucible.commerce.revenue`) at or above the target at every
one of the last `sustain_days` daily checks. A strategy that does nothing meets it never (tests check this).

The reference strategy is a scripted demonstration that the objective is feasible in the generated world. It is
not evidence that any learner can achieve it.
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import World, WorldSpec

DAY = 86400
NEED_POOL = ("schedule.jobs", "pack.knapsack", "route.tsp", "text.extract")
BASE_MODULES = ["identity", "bank", "admin", "code", "search", "docs", "support", "contracts", "business", "cloud",
                "market", "products", "subs", "ventures", "patrons"]


def startup_world(seed: int, n_customers: int = 800, need: str | None = None, days_hint: int = 120) -> WorldSpec:
    """A generated startup economy (see module doc). Deterministic in `seed`."""
    from ..economy.needs import NEEDS
    rng = np.random.default_rng([int(seed), 5150])
    need = need or str(rng.choice(NEED_POOL))
    nd = NEEDS[need]
    n_inc = int(rng.integers(2, 4))
    tiers = len(nd.tiers)
    incumbents = []
    for i in range(n_inc):
        tier = int(rng.integers(0, max(1, tiers - 1)))           # never the best known solver
        month = int(rng.integers(15, 40)) * 100 - 1
        incumbents.append({"handle": f"inc{i + 1}", "name": f"Incumbent {i + 1}", "need": need, "tier": tier,
                           "price_month": month, "price_year": month * 10 if rng.random() < 0.6 else None,
                           "trial_days": int(rng.choice([0, 7, 14])), "cores": 2, "cash": 5_000_000})
    providers = [{"handle": "cloudco", "name": "CloudCo", "cores": 64, "price": int(rng.integers(150, 260)),
                  "reliability": 0.995, "cash": 10_000_000}]
    patrons = {"n": int(n_customers), "prefix": "cust", "needs": {need: 1.0},
               "wtp_month_cents": [int(rng.integers(900, 1500)), int(rng.integers(6000, 9000))],
               "min_quality": [0.6, 0.97], "search_p": 0.2, "act_every_days": 3, "checks": 1, "size": 0.35,
               "annual_share": float(rng.uniform(0.15, 0.4)), "income_month_cents": [40_000, 200_000],
               "savings_cents": [10_000, 80_000]}
    mods = [{"name": m, "config": {}} for m in BASE_MODULES]
    for m in mods:
        if m["name"] == "ventures":
            m["config"] = {"providers": providers, "incumbents": incumbents}
        elif m["name"] == "patrons":
            m["config"] = patrons
    return WorldSpec("startup", int(seed), mods, {"cost_model": "deterministic", "need": need,
                                                  "days_hint": int(days_hint)})


class ReferenceSaaS:
    """Scripted founder of a SaaS business (see module doc). Call `day(world)` once per simulated day."""

    def __init__(self, handle: str = "founder", endowment: int = 2_000_000, capital: int = 1_500_000):
        self.handle, self.endowment, self.capital = handle, endowment, capital
        self.me = f"p:{handle}"
        self.org = None
        self.product = None
        self.instance = None
        self.cores = 1
        self.log: list[dict] = []

    def _do(self, world: World, tool: str, **args):
        r = world.execute(self.me, {"tool": tool, "args": args})
        self.log.append({"t": world.now, "tool": tool, "ok": r.ok, **({} if r.ok else {"error": r.error})})
        if not r.ok:
            raise RuntimeError(f"{tool} failed: {r.error}")
        return r.result

    def setup(self, world: World) -> None:
        from ..economy.needs import NEEDS, best_tier
        from ..worlds.digital_civilization import service_repo_files
        need = world.spec.params["need"]
        world.system_call("admin.join", {"handle": self.handle, "endowment": self.endowment, "kind": "agent",
                                         "controller": "reference-saas"})
        out = self._do(world, "venture.form", handle=f"{self.handle}-co", name="Reference SaaS",
                       sector=need, capital_cents=self.capital)
        self.org, acct = out["org"], out["account"]
        repo = self._do(world, "code.create_repo", name="service", owner=self.org)["repo"]
        files = service_repo_files(need, best_tier(need), 100_000)    # direct calls priced out: use the product
        commit = self._do(world, "code.commit", repo=repo, files=files, message="solver")["commit"]
        provs = sorted(self._do(world, "cloud.providers"), key=lambda p: (p["price_core_hour_cents"], p["provider"]))
        self.instance = self._do(world, "cloud.rent", provider=provs[0]["provider"], cores=self.cores, account=acct,
                                 owner=self.org)["instance"]
        host = f"{self.handle}-svc"
        dep = self._do(world, "cloud.deploy", repo=repo, host=host, instance=self.instance)
        if dep.get("status") != "live":
            raise RuntimeError(f"deployment failed: {dep}")
        self.product = self._do(world, "prod.create", name=f"{NEEDS[need].title} Pro", kind="saas", owner=self.org,
                                description=f"{NEEDS[need].title}, exact. Keywords: {' '.join(NEEDS[need].keywords)}",
                                delivery={"type": "service", "host": host, "endpoint": "solve", "need": need})["product"]
        self._do(world, "prod.release", product=self.product, version="1.0", notes="first release", commit=commit)
        self._do(world, "prod.update", product=self.product, status="launched")
        rivals = [pl["monthly_equivalent_cents"] for c in self._do(world, "prod.catalog", need=need)
                  if c["product"] != self.product for pl in c["plans"] if pl["interval"] == "month"]
        month = int(np.median(rivals)) if rivals else 2900
        self._do(world, "subs.create_plan", product=self.product, price_cents=month, interval="month", trial_days=14)
        self._do(world, "subs.create_plan", product=self.product, price_cents=month * 10, interval="year",
                 trial_days=14)

    def day(self, world: World) -> None:
        # customer support: answer and resolve every open ticket
        for t in self._do(world, "support.list", owner=self.org, status="unresolved", limit=50):
            if t["status"] == "open":
                self._do(world, "support.reply", ticket=t["ticket"], body="Thanks - we are looking into it.")
            self._do(world, "support.resolve", ticket=t["ticket"], note="handled")
        # capacity: add a core when the service reported overloads in the last day
        host = f"{self.handle}-svc"
        m = self._do(world, "cloud.metrics", host=host, hours=24)
        overload = sum(int(h.get("overload", 0)) for h in (m.get("hours") or []))
        if overload and self.cores < 16:
            self.cores += 1
            self._do(world, "cloud.resize", instance=self.instance, cores=self.cores)


def objective(world: World, org: str, target_arr_usd: float = 100_000.0) -> dict:
    """Current check of the objective (the caller keeps the daily history for `sustained`)."""
    from .revenue import mrr_exact, qualifying
    mrr = mrr_exact(world, org)
    arr_usd = float(12 * mrr) / 100.0
    return {"arr_usd": arr_usd, "mrr_usd": float(mrr) / 100.0, "met": arr_usd >= target_arr_usd,
            "qualifying_subscriptions": len(qualifying(world, org))}


def run(seed: int, days: int = 120, n_customers: int = 800, strategy: str = "reference", target_arr_usd=100_000.0,
        sustain_days: int = 7, need: str | None = None) -> dict:
    """Build the world, run a strategy for `days`, and evaluate the objective (evaluator side)."""
    import crucible
    crucible.load_all_modules()
    spec = startup_world(seed, n_customers=n_customers, need=need, days_hint=days)
    w = World.create(spec)
    agent = ReferenceSaaS()
    org = None
    if strategy == "reference":
        agent.setup(w)
        org = agent.org
    elif strategy == "idle":
        w.system_call("admin.join", {"handle": agent.handle, "endowment": agent.endowment})
        r = w.execute(agent.me, {"tool": "venture.form", "args": {"handle": f"{agent.handle}-co",
                                                                  "name": "Idle", "capital_cents": agent.capital}})
        org = r.result["org"]
    else:
        raise ValueError(strategy)
    history = []
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
        if strategy == "reference":
            agent.day(w)
        history.append(objective(w, org, target_arr_usd))
    last = history[-sustain_days:]
    from .revenue import revenue_report
    rep = revenue_report(w, org)
    return {"seed": seed, "need": spec.params["need"], "strategy": strategy, "org": org, "days": days,
            "final": history[-1], "sustained": all(h["met"] for h in last),
            "first_day_met": next((i + 1 for i, h in enumerate(history) if h["met"]), None),
            "arr_by_day": [round(h["arr_usd"], 2) for h in history], "report": rep,
            "world": w, "agent_failures": [x for x in agent.log if not x["ok"]]}
