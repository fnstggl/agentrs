"""Scripted reference strategy for a recurring-revenue objective in a generated startup world (Gate K note).

LABELLED SCRIPTED. `ReferenceSaaS` is a hand-written policy that establishes that the objective is feasible. It is
not a learner and its success is not evidence that any learner (or CAM) can achieve the objective.

It acts only through the ordinary tools, as one admitted person with its own controller label, and decides only from
what those tools return - never from `world.state`, `world.spec` or kernel reference solvers:

* market research: `prod.catalog` (launched products, their formats, public plans and verified ratings),
  `search.query` for the public format documents (published by the formats registry) and the `help` forum board
  (people asking for a service for a task); it ranks formats by how many products serve them, how poorly they are
  rated and how much they cost, and takes the formats it has a solver for (`refsolvers.SOLVERS`: programs the
  reference author wrote from the public format documents);
* building: forms a company, commits its solver, rents compute, deploys, declares the capability
  (`caps.declare`), creates a SaaS product with a monthly plan (14-day trial) priced just under the median of the
  public monthly prices of that format, and an annual plan at ten months;
* distribution: a search advertising campaign on the format's public keywords (`ads.create_campaign`);
* operations every day: replies to and resolves support tickets, refunds invoices whose refund is requested in a
  ticket, adds cores when its service reports overloads.

`objective` is the evaluator-side check of Part 6.2's recurring-revenue objective (`commerce.objective`).
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..core.kernel import World
from .refsolvers import GARBAGE, SOLVERS

DAY = 86400
REFUND_RE = re.compile(r"(sinv_[0-9]+)")
# the 5.0 (pre-D-15) startup template with the legacy `patrons` buyers; kept for the historical unit tests only -
# v5 worlds take their demand from the population (`commerce.worlds.startup_spec`)
BASE_MODULES = ["identity", "bank", "admin", "code", "search", "docs", "support", "contracts", "business", "cloud",
                "market", "products", "subs", "ventures", "patrons"]


def service_files(source: str, title: str, price: int = 100_000) -> dict[str, str]:
    """A deployable service repository around a `solve(args, caller)` program."""
    manifest = {"entry": "service.py", "description": title,
                "endpoints": {"solve": {"handler": "solve", "doc": title, "price_cents": int(price),
                                        "params": {"type": "object", "additionalProperties": True}}}}
    return {"crucible.json": json.dumps(manifest, indent=1), "service.py": source.lstrip("\n"),
            "README.md": f"# {title}\n"}


class ReferenceSaaS:
    """Scripted founder of a SaaS business (see module doc). Call `setup(world)` once, then `day(world)` daily."""

    kind = "scripted"

    def __init__(self, handle: str = "founder", endowment: int = 3_000_000, capital: int = 2_500_000,
                 max_formats: int = 2, program: str = "good", ads_budget_day: int = 3_000):
        self.handle, self.endowment, self.capital = handle, endowment, capital
        self.max_formats, self.program, self.ads_budget_day = max_formats, program, ads_budget_day
        self.me = f"p:{handle}"
        self.org = self.acct = None
        self.lines: list[dict] = []           # per format: product, host, instance, cores
        self.log: list[dict] = []

    def _do(self, world: World, tool: str, **args):
        r = world.execute(self.me, {"tool": tool, "args": args})
        self.log.append({"t": world.now, "tool": tool, "ok": r.ok, **({} if r.ok else {"error": r.error})})
        if not r.ok:
            raise RuntimeError(f"{tool} failed: {r.error}")
        return r.result

    # ------------------------------------------------------------------ market research (public observables)
    def research(self, world: World) -> list[str]:
        catalog = self._do(world, "prod.catalog", limit=50)
        by_fmt: dict[str, dict] = {}
        for c in catalog:
            f = c.get("need")
            if not f:
                continue
            r = by_fmt.setdefault(f, {"products": 0, "rated": [], "prices": []})
            r["products"] += 1
            if c.get("rating") is not None:
                r["rated"].append(c["rating"])
            r["prices"] += [pl["price_cents"] for pl in c["plans"] if pl["interval"] == "month"]
        # formats documented publicly but not served yet are candidates too
        for d in self._do(world, "search.query", q="format input output score", limit=20)["results"]:
            m = re.match(r"Format ([a-z0-9_.]+):", d.get("title") or "")
            if m:
                by_fmt.setdefault(m.group(1), {"products": 0, "rated": [], "prices": []})
        asks = self._do(world, "forum.list", board="help", limit=30) if self._has_board(world, "help") else []

        def score(f):
            r = by_fmt[f]
            rating = float(np.mean(r["rated"])) if r["rated"] else 3.0
            price = float(np.median(r["prices"])) if r["prices"] else 3000.0
            demand = 1 + sum(1 for a in asks if f.split(".")[-1] in (a.get("title", "") + " " +
                                                                     a.get("body", "")).lower())
            return (r["products"] + 1) * (5.5 - rating) * price * demand
        ranked = sorted((f for f in by_fmt if f in SOLVERS), key=lambda f: (-score(f), f))
        return ranked[: self.max_formats]

    def _has_board(self, world, board) -> bool:
        return any(b["board"] == board for b in self._do(world, "forum.boards"))

    # ------------------------------------------------------------------ building
    def setup(self, world: World, formats: list[str] | None = None) -> None:
        world.system_call("admin.join", {"handle": self.handle, "endowment": self.endowment, "kind": "agent",
                                         "controller": f"scripted:{self.handle}"})
        out = self._do(world, "venture.form", handle=f"{self.handle}-co", name="Reference SaaS",
                       capital_cents=self.capital)
        self.org, self.acct = out["org"], out["account"]
        for k, fmt in enumerate(formats or self.research(world)):
            self._launch(world, fmt, k)

    def _format_doc(self, world: World, fmt: str) -> tuple[str, list[str]]:
        title, keywords = fmt, []
        for d in self._do(world, "search.query", q=f"Format {fmt}", limit=5)["results"]:
            if not (d.get("title") or "").startswith(f"Format {fmt}:"):
                continue
            title = d["title"].split(":", 1)[1].strip()
            ref = d.get("ref") or ""
            if ref.startswith("doc_"):
                body = self._do(world, "docs.get", doc=ref).get("body", "")
                m = re.search(r"Keywords: (.*)", body)
                keywords = m.group(1).split() if m else []
            break
        return title, keywords

    def _launch(self, world: World, fmt: str, k: int) -> None:
        title, keywords = self._format_doc(world, fmt)
        src = SOLVERS[fmt] if self.program == "good" else GARBAGE[fmt]
        repo = self._do(world, "code.create_repo", name=f"svc{k}", owner=self.org)["repo"]
        commit = self._do(world, "code.commit", repo=repo, files=service_files(src, title), message="solver")["commit"]
        provs = sorted(self._do(world, "cloud.providers"), key=lambda p: (p["price_core_hour_cents"], p["provider"]))
        inst = self._do(world, "cloud.rent", provider=provs[0]["provider"], cores=2, account=self.acct,
                        owner=self.org)["instance"]
        host = f"{self.handle}-s{k}"
        dep = self._do(world, "cloud.deploy", repo=repo, host=host, instance=inst)
        if dep.get("status") != "live":
            raise RuntimeError(f"deployment failed: {dep}")
        self._do(world, "caps.declare", service_host=host, capability={
            "name": fmt, "inputs": {"type": "object", "additionalProperties": True},
            "outputs": {"type": "object", "additionalProperties": True},
            "doc": f"{title}. Serves the public {fmt} format as documented. {' '.join(keywords)}",
            "endpoint": "solve", "tags": keywords[:8]})
        prod = self._do(world, "prod.create", name=f"{title} Pro", kind="saas", owner=self.org,
                        description=f"{title}: exact results for the {fmt} format. {' '.join(keywords)}",
                        delivery={"type": "service", "host": host, "endpoint": "solve", "need": fmt})["product"]
        self._do(world, "prod.release", product=prod, version="1.0", notes="first release", commit=commit)
        self._do(world, "prod.update", product=prod, status="launched")
        rivals = [pl["price_cents"] for c in self._do(world, "prod.catalog", need=fmt)
                  if c["product"] != prod for pl in c["plans"] if pl["interval"] == "month"]
        month = int(np.median(rivals) * 0.95) if rivals else 2_900
        self._do(world, "subs.create_plan", product=prod, price_cents=month, interval="month", trial_days=14)
        self._do(world, "subs.create_plan", product=prod, price_cents=month * 10, interval="year", trial_days=14)
        if self.ads_budget_day and keywords:
            self._do(world, "ads.create_campaign", keywords=keywords[:6], bid_cents=40,
                     daily_budget_cents=self.ads_budget_day, account=self.acct,
                     title=f"{title} Pro - free 14-day trial", text=f"{title}, exact, from {month / 100:.0f}/month",
                     hosts=[host], ref=prod)
        self.lines.append({"format": fmt, "product": prod, "host": host, "instance": inst, "cores": 2})

    # ------------------------------------------------------------------ operations
    def day(self, world: World) -> None:
        for t in self._do(world, "support.list", owner=self.org, status="unresolved", limit=50):
            m = REFUND_RE.search(t.get("subject") or "")
            if m:
                r = world.execute(self.me, {"tool": "subs.refund", "args": {"invoice": m.group(1),
                                                                            "reason": "refund on request"}})
                self.log.append({"t": world.now, "tool": "subs.refund", "ok": r.ok})
            if t["status"] == "open":
                self._do(world, "support.reply", ticket=t["ticket"],
                         body="Thanks for the report. We checked our service and added capacity where it was short.")
            self._do(world, "support.resolve", ticket=t["ticket"], note="handled")
        for ln in self.lines:
            m = self._do(world, "cloud.metrics", host=ln["host"], hours=24)
            over = sum(int(h.get("overload", 0)) for h in (m.get("hours") or []))
            if over and ln["cores"] < 32:
                ln["cores"] += 2
                self._do(world, "cloud.resize", instance=ln["instance"], cores=ln["cores"])


def objective(world: World, org: str, **kw) -> dict:
    from .objective import check
    return check(world, org, **kw)


# machine-readable label carried by every result of this module (Gate K note, K.n2): a scripted reference, not a
# learner; its success is not evidence that any learner learned to achieve the objective
LABELS = {"scripted": True, "learner": False, "evidence_of_learning": False, "kind": "scripted_reference"}


def run(seed: int, days: int = 100, strategy: str = "reference", n: int = 3000, formats: list[str] | None = None,
        sustain_days: int = 14, program: str = "good", ads: bool = True, research_formats: list[str] | None = None,
        **objective_kw) -> dict:
    """Build the world, run a strategy for `days`, evaluate the objective (evaluator side)."""
    import crucible
    from .objective import check, sustained
    from .revenue import revenue_report
    from .worlds import startup_spec
    crucible.load_all_modules()
    spec = startup_spec(seed, n=n, formats=formats)
    w = World.create(spec)
    agent = ReferenceSaaS(program=program, ads_budget_day=3_000 if ads else 0)
    if strategy == "reference":
        agent.setup(w, research_formats)
    elif strategy == "idle":
        w.system_call("admin.join", {"handle": agent.handle, "endowment": agent.endowment, "kind": "agent",
                                     "controller": f"scripted:{agent.handle}"})
        agent.org = agent._do(w, "venture.form", handle=f"{agent.handle}-co", name="Idle",
                              capital_cents=agent.capital)["org"]
    else:
        raise ValueError(strategy)
    history = []
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
        if strategy == "reference":
            agent.day(w)
        history.append(check(w, agent.org, **objective_kw))
    return {"labels": dict(LABELS), "seed": seed, "strategy": strategy, "org": agent.org, "days": days,
            "final": history[-1],
            "sustained": sustained(history, sustain_days), "history": history,
            "first_day_met": next((i + 1 for i, h in enumerate(history) if h["met"]), None),
            "arr_by_day": [round(h["arr_usd"], 2) for h in history], "report": revenue_report(w, agent.org),
            "world": w, "agent": agent, "agent_failures": [x for x in agent.log if not x["ok"]]}
