"""Scripted reference agents for the innovation mechanisms (feasibility only - labelled `kind = "scripted"`).

They act exclusively through ordinary tools (`World.execute` as their own principal) and decide only from what
those tools return. They establish that the mechanisms can be driven end to end; they are not learners, and
results obtained with them are reported as scripted-reference results.

* `ServiceCreator` writes, deploys and declares a service for a public problem format; optionally it publishes a
  forum announcement, scales its instance when it holds a supply need, or builds only in response to public need
  requests (`needs.requests`) for that format (an entrepreneur reacting to a market signal).
* `BatchCreator` builds a composite `route.batch` service on top of an existing `route.shortest` service (calls it
  once per query through `crucible_sdk.call`): a technology whose adoption multiplies the load on another service.
* `drive(world, agents, until, dt)` advances the clock hour by hour and lets every agent act after each step.
"""
from __future__ import annotations

import json

from ..core.kernel import World
from .formats import FORMATS
from .worlds import FORMAT_SCHEMAS

HOUR, DAY = 3600, 86400


def drive(world: World, agents: list, until: int, dt: int = HOUR) -> None:
    t = world.now
    while t < until:
        t = min(until, (t // dt + 1) * dt)
        world.advance_time(t)
        for a in agents:
            a.act(world)


def join(world: World, handle: str, endowment: int, kind: str = "agent") -> str:
    return world.system_call("admin.join", {"handle": handle, "endowment": int(endowment), "kind": kind})["principal"]


class _Agent:
    kind = "scripted"

    def __init__(self, handle: str, endowment: int = 3_000_000):
        self.handle = handle
        self.actor = f"p:{handle}"
        self.endowment = endowment
        self.log: list[dict] = []
        self.joined = False

    def ex(self, world, tool, **args):
        r = world.execute(self.actor, {"tool": tool, "args": args})
        self.log.append({"t": world.now, "tool": tool, "ok": r.ok, "error": None if r.ok else r.error.get("message")})
        return r

    def ensure(self, world):
        if not self.joined:
            if not world.state.has("principals", self.actor):
                join(world, self.handle, self.endowment)
            self.joined = True

    def act(self, world):
        raise NotImplementedError


def service_files(need: str, tier: int, price: int) -> dict[str, str]:
    from ..worlds.digital_civilization import service_repo_files
    return service_repo_files(need, tier, price)


def batch_files(base_host: str, price: int) -> dict[str, str]:
    src = ("import crucible_sdk\n\n"
           "def solve(args, caller):\n"
           "    paths = []\n"
           "    for q in args['queries']:\n"
           f"        r = crucible_sdk.call('{base_host}.solve', **q)\n"
           "        paths.append((r.get('body') or {}).get('path', []) if isinstance(r, dict) else [])\n"
           "    return {'paths': paths}\n")
    manifest = {"entry": "service.py", "description": "Batch route planning",
                "endpoints": {"solve": {"handler": "solve", "doc": FORMATS["route.batch"].doc, "price_cents": price,
                                        "params": {"type": "object", "additionalProperties": True}}}}
    return {"crucible.json": json.dumps(manifest, indent=1), "service.py": src,
            "README.md": f"# Batch routes\n\nPlans many routes per request using {base_host}.crucible.\n"}


class ServiceCreator(_Agent):
    """Builds and declares a service for `need` at time `at` (or on a public request, if `on_request`)."""

    def __init__(self, handle, need, host, *, tier=1, price=3, at=0, cores=1, endowment=3_000_000,
                 announce=False, scale_on_need=False, max_cores=4, on_request=False, min_requesters=1,
                 files=None, schema=None):
        super().__init__(handle, endowment)
        self.need, self.host, self.tier, self.price, self.at, self.cores = need, host, tier, price, at, cores
        self.announce, self.scale_on_need, self.max_cores = announce, scale_on_need, max_cores
        self.on_request, self.min_requesters = on_request, min_requesters
        self.files, self.schema = files, schema
        self.instance = None
        self.declaration = None
        self.built_t = None
        self.post = None

    def wants_to_build(self, world) -> bool:
        if world.now < self.at:
            return False
        if not self.on_request:
            return True
        r = self.ex(world, "needs.requests", limit=50)
        if not r.ok:
            return False
        return any(q["target"] == f"cap:{self.need}" and q["status"] in ("open", "addressed")
                   and q["kind"] in ("supply", "provide", "alternative") and q["requesters"] >= self.min_requesters
                   for q in r.result)

    def build(self, world) -> bool:
        repo = self.ex(world, "code.create_repo", name=self.host, visibility="private")
        if not repo.ok:
            return False
        files = self.files or service_files(self.need, self.tier, self.price)
        if not self.ex(world, "code.commit", repo=repo.result["repo"], files=files, message="first version").ok:
            return False
        provs = world.query(self.actor, "cloud.providers")
        prov = sorted(provs, key=lambda p: (-p["cores_free"], p["provider"]))[0]["provider"]
        acct = world.modules["bank"].primary_account(world.state, self.actor)
        inst = self.ex(world, "cloud.rent", provider=prov, cores=self.cores, account=acct)
        if not inst.ok:
            return False
        self.instance = inst.result["instance"]
        dep = self.ex(world, "cloud.deploy", repo=repo.result["repo"], host=self.host, instance=self.instance)
        if not dep.ok or dep.result.get("status") != "live":
            return False
        sch = self.schema or FORMAT_SCHEMAS.get(self.need) or {"inputs": FORMATS[self.need].inputs,
                                                               "outputs": FORMATS[self.need].outputs}
        from .formats import lookup
        fmt = lookup(self.need)
        d = self.ex(world, "caps.declare", service_host=self.host,
                    capability={"name": self.need, "inputs": sch["inputs"], "outputs": sch["outputs"],
                                "doc": f"{fmt.title}. {fmt.doc}", "tags": list(fmt.keywords[:6])})
        if not d.ok:
            return False
        self.declaration = d.result["declaration"]
        self.built_t = world.now
        if self.announce and "forum" in world.modules:
            p = self.ex(world, "forum.post", board="market", title=f"New service for {self.need}: {self.host}",
                        body=f"{self.host}.crucible solves {self.need} ({fmt.title}) for {self.price} cents per call.")
            self.post = p.result["post"] if p.ok else None
        return True

    def act(self, world):
        self.ensure(world)
        if self.declaration is None:
            if self.wants_to_build(world):
                self.build(world)
            return
        if self.scale_on_need and self.instance is not None:
            r = self.ex(world, "needs.mine", status="open")
            if r.ok and any(n["kind"] == "supply" and "operator" in n["roles"] for n in r.result):
                inst = world.state.get("instances", self.instance)
                if inst and inst["cores"] < self.max_cores:
                    self.ex(world, "cloud.resize", instance=self.instance, cores=inst["cores"] + 1)


class BatchCreator(ServiceCreator):
    """A composite technology: `route.batch` built on top of an existing route.shortest service."""

    def __init__(self, handle, host, base_host, *, price=8, at=0, endowment=3_000_000, **kw):
        super().__init__(handle, "route.batch", host, price=price, at=at, endowment=endowment,
                         files=batch_files(base_host, price),
                         schema={"inputs": FORMATS["route.batch"].inputs, "outputs": FORMATS["route.batch"].outputs},
                         **kw)
        self.base_host = base_host

    def build(self, world) -> bool:
        if not super().build(world):
            return False
        # the composite's service principal pays the base service from its operator's account
        self.ex(world, "id.grant", grantee=f"svc:{self.host}", cap="bank.spend", limit=10_000_000)
        return True
