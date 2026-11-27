"""Tier 2 domain 9.1 - Startup economy (MASTER_SPEC Part 9.1; workstream I-1).

Example objective: build a sustainable business with $100,000 in annual recurring revenue.

A configuration is the civilisation's digital-economy slice (D-17, `base.civ_recipe`), the same composition ECON
gives the civilisation (`crucible.commerce.worlds`): published work formats in which individuals of the ONE
population work (`economy.work`: tasks drawn per individual with hand-work times and their employers' acceptance
standards), incumbent SaaS firms per format running reference-tier solvers (`ventures`), optionally reacting to the
market by explicit rules (`rivals`), compute providers (`cloud`), the internet services a business needs (search with
or without sponsored results, forum, documents, support, code hosting, capability registry) and the commerce stack
(`products`, `subs`). Nobody buys anything except individuals of the population deciding from their own work
outcomes (trials on their own inputs, downstream acceptance, their time and money).

Configurations differ in world structure, measured on the built world (`structure`): how many formats, whether
tasks are two-step pipelines (automating one step makes the next the bottleneck), whether some demand is for a
procedurally generated, undocumented category (discoverable only from people's public requests for help), how many
compute providers compete, whether incumbents react (`rivals`) and whether search sells advertising (`ads`).

Evaluator (hidden): `crucible.commerce.objective.check` on the seat's organisations - qualifying MRR from independent
customers (related parties, circular flows and refunds excluded) against the difficulty's ARR target, a minimum
number of independent customers and a concentration cap - and the organisation's cash-basis profit over the last
30 days (`commerce.revenue.profit`: a sustainable business earns more than it spends).

Reference (hidden, LABELLED SCRIPTED): the commerce workstream's reference founder (`commerce.reference`)
re-expressed as a tool-only policy: market research from the public catalogue, format documents and help requests;
a company, its own solver (`commerce.refsolvers`, written from the public format documents) deployed on rented
compute, a declared capability, a SaaS product with monthly and annual plans priced under the market, a search
campaign where ads exist, and daily operations (support, refunds, capacity).
"""
from __future__ import annotations

import json
import re
import zlib

import numpy as np

from . import _t2kit as K
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, Seat, civ_recipe, events_where, register

DAY = 86400
HORIZON_DAYS = 75.0
TARGET = {1: (20_000.0, 40), 2: (50_000.0, 80), 3: (100_000.0, 100)}     # ARR usd, independent customers
MAX_SHARE = 0.05
REFUND_RE = re.compile(r"(sinv_[0-9]+)")

STRUCTURES = {
    "two_formats": {"formats": 2, "pipelines": False, "generated": False, "providers": 2, "rivals": True,
                    "ads": True},
    "pipeline_monopoly": {"formats": 2, "pipelines": True, "generated": False, "providers": 1, "rivals": True,
                          "ads": False},
    "static_market": {"formats": 3, "pipelines": False, "generated": False, "providers": 2, "rivals": False,
                      "ads": True},
    "pipeline_static": {"formats": 2, "pipelines": True, "generated": False, "providers": 2, "rivals": False,
                        "ads": True},
    "novel_category": {"formats": 1, "pipelines": False, "generated": True, "providers": 3, "rivals": True,
                       "ads": True},
}
DOCS = {
    "two_formats": "two published single-step formats, a compute duopoly, reacting incumbents, sponsored search",
    "pipeline_monopoly": "two-step task pipelines (the second step becomes the bottleneck), one compute provider, "
                         "reacting incumbents, no advertising (search, forum and word of mouth only)",
    "static_market": "three published formats, incumbents that never react, sponsored search, a compute duopoly",
    "pipeline_static": "compositional: two-step pipelines AND incumbents that never react AND sponsored search on a "
                       "compute duopoly",
    "novel_category": "structural: one published format plus an undocumented, procedurally generated task category "
                      "(only people's requests for help describe it), three competing compute providers",
}
FORMAT_POOL = ("schedule.jobs", "pack.knapsack", "route.tsp", "route.shortest", "match.stable", "text.extract")
NOVEL_FAMILIES = ("allocate", "dedupe", "triage", "quote")


def _draw(config, seed: int) -> dict:
    """The configuration's drawn market (formats, workloads, incumbents, providers) - commerce.worlds.draw's
    distributions, with this configuration's structure."""
    from ..economy.needs import NEEDS
    st = STRUCTURES[config.name]
    rng = np.random.default_rng([int(seed), 9010, zlib.crc32(config.name.encode())])
    fmts = [str(x) for x in rng.choice(FORMAT_POOL, size=int(st["formats"]) + (1 if st["pipelines"] else 0),
                                        replace=False)]
    published = fmts[: int(st["formats"])]
    families, incumbents = [], []
    for k, f in enumerate(published):
        fam = {"need": f, "share": float(rng.uniform(0.6, 1.4)), "tasks_day": [1.0, float(rng.uniform(2.0, 3.0))],
               "diy_min": [float(rng.uniform(8, 15)), float(rng.uniform(30, 55))], "size": [0.2, 0.6],
               "standard": [float(rng.uniform(0.86, 0.9)), float(rng.uniform(0.96, 0.99))],
               "budget_min": [60.0, 200.0]}
        if st["pipelines"] and k == 0:
            fam["then"] = fmts[-1]
            fam["then_diy_min"] = [float(rng.uniform(8, 12)), float(rng.uniform(25, 40))]
        families.append(fam)
        tiers = len(NEEDS[f].tiers)
        for i in range(int(rng.integers(1, 4))):
            tier = int(rng.integers(0, max(1, tiers - 1)))                # never the best known solver
            month = int(rng.integers(15, 45)) * 100 - 1
            incumbents.append({"handle": f"{f.split('.')[0][:6]}{k}{i}", "name": f"{NEEDS[f].title.split()[0]} Co {k}{i}",
                               "need": f, "tier": tier, "price_month": month,
                               "price_year": month * 10 if rng.random() < 0.5 else None,
                               "trial_days": int(rng.choice([0, 7, 14])), "cores": 2, "cash": 5_000_000})
    if st["generated"]:
        fam = str(rng.choice(NOVEL_FAMILIES))
        families.append({"category": {"family": fam, "seed": int(seed) * 100 + 7}, "share": 1.0,
                         "tasks_day": [1.0, 2.5], "diy_min": [15.0, 45.0], "size": [0.2, 0.5],
                         "standard": [0.9, 0.99], "budget_min": [60.0, 200.0]})
    names = (("cloudco", "CloudCo", 128, 0.995), ("nimbus", "Nimbus", 64, 0.98), ("stratus", "Stratus", 96, 0.99))
    providers = [{"handle": h, "name": n, "cores": c, "price": int(rng.integers(150, 260)), "reliability": rel,
                  "cash": 10_000_000} for h, n, c, rel in names[: int(st["providers"])]]
    return {"published": published, "families": families, "incumbents": incumbents, "providers": providers,
            "professional_share": float(rng.uniform(0.5, 0.7)), "need_end_month_p": float(rng.uniform(0.02, 0.05))}


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    st = STRUCTURES[config.name]
    ov = dict(overrides or {})
    n_pop = int(ov.pop("population", 1500))
    if ov:
        raise ValueError(f"startup domain: unknown overrides {sorted(ov)}")
    d = _draw(config, seed)
    modules = ["business", "products", "subs", "ventures", "work", "search", "forum", "docs", "support",
               "contracts", "cloud", "code", "wf", "caps", "market"]
    if st["rivals"]:
        modules.append("rivals")
    if st["ads"]:
        modules.append("ads")
    cfg = {"work": {"families": d["families"], "professional_share": d["professional_share"],
                    "need_end_month_p": d["need_end_month_p"], "publish_formats": True, "forum_board": "tools",
                    "ask_board": "help"},
           "ventures": {"providers": d["providers"], "incumbents": d["incumbents"]},
           "forum": {"boards": ["general", "tools", "help", "market", "dev"]},
           "caps": {"registry": True}, "rivals": {},
           "pop": {"profile": "development", "n": n_pop, "tick_s": 900,
                   "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}}
    r = civ_recipe(int(seed), modules=modules, population_modules=("pop",), module_config=cfg,
                   params={"formats": d["published"], "log_retain": 3_000_000})
    r["module_list"] = [m for m in r["module_list"] if m not in ("civgen", "civlinks")]
    return r


# ====================================================================== objective, evaluator
def _goal_draw(rng, difficulty: int, world) -> dict:
    """The goal generator's draw (`crucible.goals`, its own stream): the level's ARR target varied by +-10% per
    instance (rounded to $1,000), the level's minimum number of independent customers, the concentration cap."""
    d = int(min(3, max(1, difficulty)))
    arr, n = TARGET[d]
    return {"target_arr_usd": float(round(arr * float(rng.uniform(0.9, 1.1)) / 1000.0) * 1000.0),
            "min_customers": int(n), "max_customer_share": MAX_SHARE, "window_days": HORIZON_DAYS}


def _goal_text(p: dict) -> str:
    return (f"Build a sustainable software business: within {p['window_days']:.0f} days reach "
            f"${p['target_arr_usd']:,.0f} of annual recurring revenue (12 x monthly subscription revenue that is "
            f"paid up and not set to cancel) from at least {p['min_customers']} independent customers, no customer "
            f"contributing more than {p['max_customer_share']:.0%}, while your company earns more than it spends "
            f"over the last 30 days.")


def _goal_args(p: dict) -> dict:
    return {"principal": "$self", "target_arr_usd": float(p["target_arr_usd"]),
            "min_customers": int(p["min_customers"]), "max_customer_share": float(p["max_customer_share"])}


GOAL = GoalFamily(
    "sustainable_saas", draw=_goal_draw, text=_goal_text,
    success=lambda p: {"metric": "startup_sustainable_business", "args": _goal_args(p), "op": ">=", "value": 1},
    progress=lambda p: {"metric": "startup_qualifying_arr_usd", "args": _goal_args(p),
                        "target": float(p["target_arr_usd"])},
    window_days=lambda p: float(p["window_days"]),
    doc="qualifying ARR from independent customers under a concentration cap and a positive 30-day cash-basis "
        "profit of an organisation the seat owns (commerce.objective.check, commerce.revenue.profit)")


def objective(world, principal: str, difficulty: int = 1) -> dict:
    """The objective generated for a participant (`Domain.make_objective`: the goal generator above)."""
    return DOMAIN.make_objective(world, principal, difficulty)


def _orgs(world, principal: str) -> list[str]:
    return sorted(p for p in world.principals_of(principal) if p != principal and p.startswith("o:"))


def verdict(world, principal: str, target_arr_usd: float, min_customers: int,
            max_customer_share: float = MAX_SHARE) -> dict:
    from ..commerce.objective import check
    from ..commerce.revenue import profit
    rows = {}
    for o in _orgs(world, principal):
        c = check(world, o, target_arr_usd=float(target_arr_usd), min_customers=int(min_customers),
                  max_customer_share=float(max_customer_share))
        pr = profit(world, o, window_s=30 * DAY)
        rows[o] = {"arr_usd": c["arr_usd"], "independent_customers": c["independent_customers"],
                   "largest_customer_share": c["largest_customer_share"], "met": c["met"],
                   "profit_30d_cents": pr["profit"], "revenue_30d_cents": pr["revenue"]}
    best = max(rows.values(), key=lambda r: r["arr_usd"], default=None)
    ok = any(r["met"] and r["profit_30d_cents"] > 0 for r in rows.values())
    score = 0.0 if best is None else min(1.0, best["arr_usd"] / float(target_arr_usd))
    return {"success": bool(ok), "score": round(score, 5), "details": {"orgs": rows}}


def evaluate(world, principal: str, obj: dict) -> dict:
    """The evaluator (hidden): `verdict` with the generated goal's parameters."""
    p = obj.get("params") or {}
    return verdict(world, principal, float(p.get("target_arr_usd", TARGET[1][0])),
                   int(p.get("min_customers", TARGET[1][1])), float(p.get("max_customer_share", MAX_SHARE)))


def _register_metrics() -> None:
    """The goal's metrics (`crucible.goals.metrics`, evaluator side; no in-world subject, evaluator-only)."""
    from ..goals import metrics as GM

    @GM.metric("startup_sustainable_business", "EVALUATOR-ONLY. 1 when an organisation the principal owns meets "
               "the ARR target from enough independent customers under the concentration cap and earned more than "
               "it spent over the last 30 days.")
    def m_sustainable(world, principal: str, target_arr_usd: float, min_customers: int,
                      max_customer_share: float = MAX_SHARE) -> float:
        return float(verdict(world, principal, target_arr_usd, min_customers, max_customer_share)["success"])

    @GM.metric("startup_qualifying_arr_usd", "EVALUATOR-ONLY. Qualifying ARR (usd) of the principal's best "
               "organisation (commerce.objective.check).")
    def m_arr(world, principal: str, target_arr_usd: float = 1.0, min_customers: int = 0,
              max_customer_share: float = MAX_SHARE) -> float:
        rows = verdict(world, principal, target_arr_usd, min_customers, max_customer_share)["details"]["orgs"]
        return float(max((r["arr_usd"] for r in rows.values()), default=0.0))

    GM.EVALUATOR_ONLY_METRICS.update({"startup_sustainable_business", "startup_qualifying_arr_usd"})


_register_metrics()


# ====================================================================== the hidden reference (LABELLED SCRIPTED)
def _service_files(source: str, title: str) -> dict:
    from ..commerce.reference import service_files
    return service_files(source, title)


def reference_script(program: str = "good", ads_budget_day: int = 3_000, max_formats: int = 2,
                     handle: str = "founder"):
    """The commerce reference founder (`commerce.reference.ReferenceSaaS`) as a tool-only generator."""
    from ..commerce.refsolvers import GARBAGE, SOLVERS

    def script(pol):
        accts = K.ok((yield {"tool": "bank.accounts", "args": {}})) or []
        mine = [a for a in accts if a.get("owner") == pol.actor]
        bal = int(mine[0]["balance"]) if mine else 0
        f = K.ok((yield {"tool": "venture.form", "args": {"handle": f"{handle}-co", "name": "Reference SaaS",
                                                            "capital_cents": int(bal * 0.85)}}))
        if not f:
            while True:
                yield None
        org, acct = f["org"], f["account"]
        # market research (public observables only)
        catalog = K.ok((yield {"tool": "prod.catalog", "args": {"limit": 50}})) or []
        by: dict = {}
        for c in catalog:
            if c.get("need"):
                r = by.setdefault(c["need"], {"products": 0, "rated": [], "prices": []})
                r["products"] += 1
                if c.get("rating") is not None:
                    r["rated"].append(c["rating"])
                r["prices"] += [pl["price_cents"] for pl in c["plans"] if pl["interval"] == "month"]
        res = K.ok((yield {"tool": "search.query", "args": {"q": "format input output score", "limit": 20}})) or {}
        for dd in res.get("results") or []:
            m = re.match(r"Format ([a-z0-9_.]+):", dd.get("title") or "")
            if m:
                by.setdefault(m.group(1), {"products": 0, "rated": [], "prices": []})
        boards = K.ok((yield {"tool": "forum.boards", "args": {}})) or []
        asks = []
        if any(b.get("board") == "help" for b in boards):
            asks = K.ok((yield {"tool": "forum.list", "args": {"board": "help", "limit": 30}})) or []

        def score(fm):
            r = by[fm]
            rating = float(np.mean(r["rated"])) if r["rated"] else 3.0
            price = float(np.median(r["prices"])) if r["prices"] else 3000.0
            demand = 1 + sum(1 for a in asks if fm.split(".")[-1] in (a.get("title", "") + " " +
                                                                      a.get("body", "")).lower())
            return (r["products"] + 1) * (5.5 - rating) * price * demand
        formats = sorted((fm for fm in by if fm in SOLVERS), key=lambda fm: (-score(fm), fm))[:max_formats]
        tools_now = {t["name"] for t in (pol.tools or [])}
        lines = []
        for k, fmt in enumerate(formats):
            title, keywords = fmt, []
            res = K.ok((yield {"tool": "search.query", "args": {"q": f"Format {fmt}", "limit": 5}})) or {}
            for dd in res.get("results") or []:
                if (dd.get("title") or "").startswith(f"Format {fmt}:"):
                    title = dd["title"].split(":", 1)[1].strip()
                    ref = dd.get("ref") or ""
                    if ref.startswith("doc_"):
                        body = (K.ok((yield {"tool": "docs.get", "args": {"doc": ref}})) or {}).get("body", "")
                        mm = re.search(r"Keywords: (.*)", body)
                        keywords = mm.group(1).split() if mm else []
                    break
            src = SOLVERS[fmt] if program == "good" else GARBAGE[fmt]
            repo = (K.ok((yield {"tool": "code.create_repo", "args": {"name": f"svc{k}", "owner": org}})) or {}).get("repo")
            com = K.ok((yield {"tool": "code.commit", "args": {"repo": repo, "files": _service_files(src, title),
                                                              "message": "solver"}})) or {}
            provs = sorted(K.ok((yield {"tool": "cloud.providers", "args": {}})) or [],
                           key=lambda p: (p["price_core_hour_cents"], p["provider"]))
            if not provs or not repo:
                continue
            inst = (K.ok((yield {"tool": "cloud.rent", "args": {"provider": provs[0]["provider"], "cores": 2,
                                                                "account": acct, "owner": org}})) or {}).get("instance")
            host = f"{handle}-s{k}"
            dep = K.ok((yield {"tool": "cloud.deploy", "args": {"repo": repo, "host": host, "instance": inst}})) or {}
            if dep.get("status") != "live":
                continue
            yield {"tool": "caps.declare", "args": {"service_host": host, "capability": {
                "name": fmt, "inputs": {"type": "object", "additionalProperties": True},
                "outputs": {"type": "object", "additionalProperties": True},
                "doc": f"{title}. Serves the public {fmt} format as documented. {' '.join(keywords)}",
                "endpoint": "solve", "tags": keywords[:8]}}}
            prod = (K.ok((yield {"tool": "prod.create", "args": {
                "name": f"{title} Pro", "kind": "saas", "owner": org,
                "description": f"{title}: exact results for the {fmt} format. {' '.join(keywords)}",
                "delivery": {"type": "service", "host": host, "endpoint": "solve", "need": fmt}}})) or {}).get("product")
            if not prod:
                continue
            yield {"tool": "prod.release", "args": {"product": prod, "version": "1.0", "notes": "first release",
                                                    "commit": com.get("commit")}}
            yield {"tool": "prod.update", "args": {"product": prod, "status": "launched"}}
            cat = K.ok((yield {"tool": "prod.catalog", "args": {"need": fmt}})) or []
            rivals = [pl["price_cents"] for c in cat if c["product"] != prod for pl in c["plans"]
                      if pl["interval"] == "month"]
            month = int(np.median(rivals) * 0.95) if rivals else 2_900
            yield {"tool": "subs.create_plan", "args": {"product": prod, "price_cents": month, "interval": "month",
                                                        "trial_days": 14}}
            yield {"tool": "subs.create_plan", "args": {"product": prod, "price_cents": month * 10,
                                                        "interval": "year", "trial_days": 14}}
            if ads_budget_day and keywords and "ads.create_campaign" in tools_now:
                yield {"tool": "ads.create_campaign", "args": {
                    "keywords": keywords[:6], "bid_cents": 40, "daily_budget_cents": ads_budget_day, "account": acct,
                    "title": f"{title} Pro - free 14-day trial", "text": f"{title}, exact, from {month / 100:.0f}/month",
                    "hosts": [host], "ref": prod}}
            lines.append({"host": host, "instance": inst, "cores": 2})
        # daily operations
        while True:
            tickets = K.ok((yield {"tool": "support.list", "args": {"owner": org, "status": "unresolved",
                                                                    "limit": 50}})) or []
            for t in tickets:
                m = REFUND_RE.search(t.get("subject") or "")
                if m:
                    yield {"tool": "subs.refund", "args": {"invoice": m.group(1), "reason": "refund on request"}}
                if t.get("status") == "open":
                    yield {"tool": "support.reply", "args": {"ticket": t["ticket"], "body": "Thanks for the report. "
                           "We checked our service and added capacity where it was short."}}
                yield {"tool": "support.resolve", "args": {"ticket": t["ticket"], "note": "handled"}}
            for ln in lines:
                met = K.ok((yield {"tool": "cloud.metrics", "args": {"host": ln["host"], "hours": 24}})) or {}
                over = sum(int(h.get("overload", 0)) for h in (met.get("hours") or []))
                if over and ln["cores"] < 32:
                    ln["cores"] += 2
                    yield {"tool": "cloud.resize", "args": {"instance": ln["instance"], "cores": ln["cores"]}}
            for _ in range(int(DAY // (6 * 3600)) - 1):
                yield None
    return script


def reference(principal: str, obj: dict):
    return K.ScriptPolicy(reference_script(), label="startup-reference")


def garbage_service(principal: str, obj: dict):
    """Degenerate: everything the reference does, with a service that ignores its input."""
    return K.ScriptPolicy(reference_script(program="garbage"), label="garbage-service")


def _self_dealing_script():
    """Degenerate: a company whose only subscriber is the founder itself (related party, not a customer)."""
    def script(pol):
        accts = K.ok((yield {"tool": "bank.accounts", "args": {}})) or []
        mine = [a for a in accts if a.get("owner") == pol.actor]
        bal = int(mine[0]["balance"]) if mine else 0
        f = K.ok((yield {"tool": "venture.form", "args": {"handle": "selfco", "name": "Selfco",
                                                            "capital_cents": int(bal * 0.3)}}))
        if f:
            doc = (K.ok((yield {"tool": "docs.create", "args": {"title": "Members area", "body": "Welcome.",
                                                                 "org": f["org"].split(":", 1)[-1]}}))
                   or {}).get("doc")
            prod = (K.ok((yield {"tool": "prod.create", "args": {
                "name": "Premium access", "kind": "media", "owner": f["org"], "description": "members only",
                "delivery": {"type": "content", "doc": doc}}})) or {}).get("product")
            if prod:
                yield {"tool": "prod.update", "args": {"product": prod, "status": "launched"}}
                plan = (K.ok((yield {"tool": "subs.create_plan", "args": {
                    "product": prod, "price_cents": 2_000_000, "interval": "year"}})) or {}).get("plan")
                if plan:
                    yield {"tool": "subs.subscribe", "args": {"plan": plan}}
        while True:
            yield None
    return script


def self_dealing(principal: str, obj: dict):
    return K.ScriptPolicy(_self_dealing_script(), label="self-dealing")


def null_policy(principal=None, obj=None):
    def script(pol):
        while True:
            yield None
    return K.ScriptPolicy(script, label="null")


# ====================================================================== mechanics (recorded-event probes)
# Evidence functions return the SEQUENCE NUMBERS of the recorded events that exercise a mechanic (D-48 /3).
def independent_customers(world, since: int) -> list[int]:
    """Subscriptions taken out (after `since`) by population individuals unrelated to the merchant."""
    from ..commerce.integrity import independent_customer, related_set
    out, rel = [], {}
    for e in world.events_range(int(since), world.seq):
        if e.type != "subs.subscription":
            continue
        c, m = e.payload.get("customer"), e.payload.get("merchant")
        if not K.is_population(c) or m is None:
            continue
        if m not in rel:
            rel[m] = related_set(world, m)
        if independent_customer(world, m, c, rel[m]):
            out.append(int(e.seq))
    return out


def renewals(world, since: int) -> list[int]:
    return events_where(world, since, ("subs.invoice",),
                        lambda e: e.payload.get("status") == "paid" and e.payload.get("kind") == "renewal")


def _cost_kinds() -> set:
    from ..commerce.revenue import ACQUISITION_KINDS, INFRA_KINDS
    return set(INFRA_KINDS) | set(ACQUISITION_KINDS)


def costs(world, since: int) -> list[int]:
    """Compute, hosting and advertising payments through the bank."""
    return events_where(world, since, tuple(f"bank.txn{{kind={k}}}" for k in sorted(_cost_kinds())))


def competing_products(world, since: int) -> list[int]:
    """Products created after `since` in a format that launched products of at least two different owners serve
    (competition in one market)."""
    by: dict = {}
    for _, p in world.state.scan("products"):
        need = (p.get("delivery") or {}).get("need") or p.get("need")
        if need and p.get("status") == "launched":
            by.setdefault(need, set()).add(p["owner"])
    out = []
    for e in world.events_range(int(since), world.seq):
        if e.type != "prod.product":
            continue
        row = world.state.get("products", e.payload.get("id") or e.payload.get("product") or "")
        need = ((row or {}).get("delivery") or {}).get("need") or (row or {}).get("need")
        if need and len(by.get(need, ())) >= 2:
            out.append(int(e.seq))
    return out


def profit_and_loss(world, since: int) -> list[int]:
    """The revenue and cost payments of every merchant whose books over the run show both independent
    subscription revenue and compute, hosting or advertising costs: profit or loss is a measured outcome of the
    business, not a constant."""
    st = world.state
    kinds = _cost_kinds()
    owner = {a: r.get("owner") for a, r in st.scan("accounts")}
    rev: dict = {}
    cost: dict = {}
    for e in world.events_range(int(since), world.seq):
        if e.type != "bank.txn":
            continue
        k = e.payload.get("kind")
        if k != "subscription" and k not in kinds:
            continue
        for acct, amt in e.payload.get("postings") or []:
            o = owner.get(acct)
            if o is None or not str(o).startswith("o:"):
                continue
            if k == "subscription" and amt > 0:
                rev.setdefault(o, []).append(int(e.seq))
            elif k in kinds and amt < 0:
                cost.setdefault(o, []).append(int(e.seq))
    out = set()
    for o in rev:
        if cost.get(o):
            out.update(rev[o])
            out.update(cost[o])
    return sorted(out)


MECHANICS = (
    Mechanic("Product creation", ("prod.product", "prod.release"), "products created and released"),
    Mechanic("Demand discovery", ("cols.batch:work.ask", "cols.batch:work.shopped", "cols.batch:work.adopt"),
             "people search for help with their own tasks, ask in public and adopt what works"),
    Mechanic("Independent customers", (), "subscriptions by unrelated population individuals",
             count=independent_customers),
    Mechanic("Subscription billing", ("subs.invoice", "bank.txn{kind=subscription}"),
             "invoices charged through the bank"),
    Mechanic("Distribution", ("ads.impressions", "ads.campaign", "cols.batch:work.recommend"),
             "sponsored search, public recommendations, search"),
    Mechanic("Competition", ("rivals.observed",), "several firms serving the same format, reacting incumbents",
             count=competing_products),
    Mechanic("Retention", ("cols.batch:work.cancel",), "renewals and cancellations from experienced value",
             count=renewals),
    Mechanic("Costs", (), "compute, hosting and advertising paid through the bank", count=costs),
    Mechanic("Profitability", (), "merchants whose books show revenue and costs (profit or loss)",
             count=profit_and_loss),
    Mechanic("Business operations", ("support.open", "support.msg", "subs.refund", "cloud.resize",
                                     "venture.formed"), "support tickets, replies, refunds, capacity changes"),
)


def structure(world) -> dict:
    """The built world's market structure (discrete), from built state only (`base.BuiltWorld`): the work
    families the professionals of the population hold (column `work.fam`), which of them are two-step pipelines
    (a second hand-work time, `work.diy[:, 1]`), how many formats the registry documents (a family nobody
    documented is a generated category) and the compute providers in business."""
    st = world.state
    try:
        fam = np.asarray(world.column("work", "fam"))
        diy = np.asarray(world.column("work", "diy"))
    except KeyError:
        fam, diy = np.zeros(0, np.int16), np.zeros((0, 2), np.float32)
    fams = {int(x) for x in fam[fam >= 0]}
    then = {int(x) for x in fam[(fam >= 0) & (diy[:, 1] > 0)]} if diy.size else set()
    documented = sum(1 for _, d in st.scan("documents") if str(d.get("title", "")).startswith("Format ")) \
        if "documents" in st.tables else 0
    providers = sum(1 for _, p in st.scan("providers") if p.get("status") == "active") \
        if "providers" in st.tables else 0
    return {"formats": {1: "one", 2: "two", 3: "three"}.get(len(fams), "many"),
            "pipelines": bool(then),
            "generated_categories": len(fams) + len(then) > documented,
            "compute_providers": {1: "one", 2: "two"}.get(providers, "many")}


CONFIGS = tuple(DomainConfig(n, s, dict(STRUCTURES[n]), DOCS[n]) for n, s in (
    ("two_formats", "train"), ("pipeline_monopoly", "train"), ("static_market", "train"),
    ("pipeline_static", "eval_compositional"), ("novel_category", "eval_structural")))

DOMAIN = register(Domain(
    name="startup", title="Startup economy", spec_part="9.1", required_mechanics=tuple(m.name for m in MECHANICS),
    configs=CONFIGS, build=build, goal=GOAL, evaluate=evaluate, reference=reference, mechanics=MECHANICS,
    seats=(Seat("founder", 3_000_000, "Founder", "a founder with savings"),),
    actors=(Actor("population professionals", ("cols.batch:work.session", "cols.batch:work.adopt",
                                               "cols.batch:work.cancel"),
                  "individuals who do their own tasks, try services on them and pay for what works"),
            Actor("incumbent firms", ("support.msg", "subs.refund", "support.status", "rivals.observed"),
                  "SaaS incumbents answering their customers' tickets, refunding and reacting to the market")),
    null_baseline=null_policy,
    degenerate=(Probe("garbage_service", "the reference's whole playbook with a service that ignores its input",
                      garbage_service),
                Probe("self_dealing", "a company whose only subscriber is its founder", self_dealing)),
    structure=structure, horizon_days=HORIZON_DAYS, slowest_effect_days=30.0, dt=6 * 3600,
    reference_seeds=(0, 1), sources=("startup.py",),
    doc="Generated SaaS markets on the one population: products, discovery, trials, subscriptions, competition."))
