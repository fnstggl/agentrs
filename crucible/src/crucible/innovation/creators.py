"""Scripted creators for Gates J and L (LABELLED SCRIPTED: feasibility demonstrations, not learners).

`NoveltyCreator` is an entrepreneur that acts only through the ordinary tools, as one admitted person whose
controller label starts with `scripted:` (so the evaluator reports its products as scripted), and decides only
from public observables:

* every day it reads the public `help` forum board (`forum.list` / `forum.read`) where individuals of the population
  ask for services for tasks they do by hand, and groups the requests by task title;
* when at least `min_asks` people ask for a task it has a program for, it builds one: a service whose solver is the
  creator's own program for the described task (`PROGRAMS`, written by the module author from the task description
  people post - not from any hidden rule or kernel reference solver), deployed on rented compute, a capability
  declaration under a name of its own choosing whose input schema is inferred from the example inputs people
  posted, a price per call, and a reply to each request naming its host;
* it answers support tickets and adds cores when its service reports overloads.

`program="garbage"` swaps in an input-ignoring program with the same interface (for counterfactual arms that test
whether measured behaviour drives adoption).
"""
from __future__ import annotations

import json
import re

from ..commerce.reference import service_files

DAY = 86400
EX_RE = re.compile(r"Example input: (\{.*\})", re.S)

PROGRAMS: dict[str, dict] = {
    "Allocate a budget across options": {
        "name": "budget.split", "host": "budget-split", "keywords": ["allocation", "budget", "returns", "spend",
                                                                      "portfolio", "optimisation"],
        "doc": "Splits a budget across options with diminishing returns (log, saturating and square-root return "
               "shapes, unit costs and caps) and returns the units to buy of every option (marginal-return "
               "allocation).",
        "outputs": {"type": "object", "required": ["units"], "properties": {"units": {"type": "array"}}},
        "source": '''
import heapq, math
def _f(k, a, b, x):
    if k == "log":
        return a * math.log(1.0 + b * x)
    if k == "sat":
        return a * (1.0 - math.exp(-b * x))
    return a * math.sqrt(b * x)
def solve(args, caller):
    # concave returns: buy the unit with the best marginal return per cost until the budget is spent
    left = int(args["budget"])
    opts = args["options"]
    units = [0] * len(opts)
    heap = []
    for j, o in enumerate(opts):
        heapq.heappush(heap, (-_f(o["kind"], float(o["a"]), float(o["b"]), 1) / int(o["cost"]), j))
    while heap:
        _, j = heapq.heappop(heap)
        o = opts[j]
        c = int(o["cost"])
        if c > left or units[j] >= int(o["cap"]):
            continue
        units[j] += 1
        left -= c
        if units[j] < int(o["cap"]):
            k, a, b = o["kind"], float(o["a"]), float(o["b"])
            heapq.heappush(heap, (-(_f(k, a, b, units[j] + 1) - _f(k, a, b, units[j])) / c, j))
    return {"units": units}
''',
        "garbage": "def solve(args, caller):\n    return {'units': [0 for _ in args['options']]}\n"},
}


def _published_programs() -> None:
    """Programs for public formats (the founder wrote them from the published format documents)."""
    from ..commerce.refsolvers import GARBAGE, SOLVERS
    from ..economy.needs import NEEDS
    for fmt, src in SOLVERS.items():
        nd = NEEDS[fmt]
        PROGRAMS.setdefault(nd.title, {
            "name": fmt, "host": fmt.replace(".", "-") + "-pro", "keywords": list(nd.keywords[:6]),
            "doc": f"Serves the public {fmt} format as documented ({nd.title.lower()}).",
            "outputs": {"type": "object", "additionalProperties": True}, "source": src, "garbage": GARBAGE[fmt]})


_published_programs()


def _json_type(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def infer_schema(examples: list[dict]) -> dict:
    """The input schema an entrepreneur can infer from example inputs: keys present in every example are
    required, with the JSON type they had."""
    keys = set(examples[0])
    for e in examples[1:]:
        keys &= set(e)
    props = {}
    for k in sorted(keys):
        ts = {_json_type(e[k]) for e in examples}
        if ts <= {"integer", "number"} and "number" in ts:
            ts = {"number"}
        props[k] = {"type": sorted(ts)[0] if len(ts) == 1 else sorted(ts)}
    return {"type": "object", "required": sorted(keys), "properties": props}


class NoveltyCreator:
    kind = "scripted"

    def __init__(self, handle: str = "maker", endowment: int = 1_500_000, capital: int = 1_000_000,
                 min_asks: int = 3, price: int = 4, program: str = "good", skip: tuple = ()):
        self.handle, self.endowment, self.capital = handle, endowment, capital
        self.min_asks, self.price, self.program, self.skip = min_asks, price, program, tuple(skip)
        self.me = f"p:{handle}"
        self.org = self.acct = None
        self.built: dict[str, dict] = {}
        self.log: list[dict] = []

    def _do(self, world, tool, **args):
        r = world.execute(self.me, {"tool": tool, "args": args})
        self.log.append({"t": world.now, "tool": tool, "ok": r.ok, **({} if r.ok else {"error": r.error})})
        if not r.ok:
            raise RuntimeError(f"{tool} failed: {r.error}")
        return r.result

    def setup(self, world) -> None:
        world.system_call("admin.join", {"handle": self.handle, "endowment": self.endowment, "kind": "agent",
                                         "controller": f"scripted:{self.handle}"})
        out = self._do(world, "venture.form", handle=f"{self.handle}-co", name="Maker Labs",
                       capital_cents=self.capital)
        self.org, self.acct = out["org"], out["account"]

    def asks(self, world) -> dict[str, list[dict]]:
        """Public requests on the help board, grouped by task title."""
        out: dict[str, list[dict]] = {}
        for p in self._do(world, "forum.list", board="help", limit=30, sort="new"):
            m = re.match(r"Looking for a service: (.*)", p.get("title") or "")
            if not m:
                continue
            title = next((t for t in PROGRAMS if m.group(1).startswith(t)), None)
            if title is None:
                continue
            out.setdefault(title, []).append(p)
        return out

    def day(self, world) -> None:
        for title, posts in sorted(self.asks(world).items()):
            if title in self.skip:
                continue
            if title not in self.built and len(posts) >= self.min_asks:
                self._build(world, title, posts)
            elif title in self.built:
                self._reply(world, title, posts)
        for t in self._do(world, "support.list", owner=self.org, status="unresolved", limit=50):
            self._do(world, "support.reply", ticket=t["ticket"], body="Thanks - looking into it.")
            self._do(world, "support.resolve", ticket=t["ticket"], note="handled")
        for b in self.built.values():
            m = self._do(world, "cloud.metrics", host=b["host"], hours=24)
            if sum(int(h.get("overload", 0)) for h in (m.get("hours") or [])) and b["cores"] < 32:
                b["cores"] = min(32, b["cores"] * 2)        # autoscale: double on overload
                self._do(world, "cloud.resize", instance=b["instance"], cores=b["cores"])

    def _build(self, world, title: str, posts: list[dict]) -> None:
        spec = PROGRAMS[title]
        examples = []
        for p in posts[:8]:
            full = self._do(world, "forum.read", post=p["post"])
            m = EX_RE.search(full.get("body") or "")
            if m:
                try:
                    examples.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    continue
        if not examples:
            return
        src = spec["source"] if self.program == "good" else spec["garbage"]
        repo = self._do(world, "code.create_repo", name=spec["host"].replace("-", "")[:39], owner=self.org)["repo"]
        self._do(world, "code.commit", repo=repo, files=service_files(src, title, price=self.price), message="v1")
        provs = sorted(self._do(world, "cloud.providers"), key=lambda p: (p["price_core_hour_cents"], p["provider"]))
        inst = self._do(world, "cloud.rent", provider=provs[0]["provider"], cores=2, account=self.acct,
                        owner=self.org)["instance"]
        host = spec["host"]
        dep = self._do(world, "cloud.deploy", repo=repo, host=host, instance=inst)
        if dep.get("status") != "live":
            raise RuntimeError(f"deployment failed: {dep}")
        self._do(world, "caps.declare", service_host=host, capability={
            "name": spec["name"], "inputs": infer_schema(examples), "outputs": spec["outputs"],
            "doc": f"{title}. {spec['doc']} Keywords: {' '.join(spec['keywords'])}", "endpoint": "solve",
            "tags": spec["keywords"]})
        self.built[title] = {"host": host, "instance": inst, "cores": 2, "repo": repo, "replied": set(),
                             "t": world.now}
        self._reply(world, title, posts)

    def _reply(self, world, title, posts) -> None:
        b = self.built[title]
        for p in posts:
            if p["post"] in b["replied"]:
                continue
            self._do(world, "forum.reply", post=p["post"],
                     body=f"We built a service for this: {b['host']}.crucible ({PROGRAMS[title]['name']}, "
                          f"{self.price} cents a call). Try it on one of your own inputs.")
            b["replied"].add(p["post"])


def drive(world, agents: list, until: int) -> None:
    """Advance day by day to `until`, letting each scripted agent act once a day."""
    while world.now < until:
        world.advance_time(min(until, (world.now // DAY + 1) * DAY))
        for a in agents:
            a.day(world)
