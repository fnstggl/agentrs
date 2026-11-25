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


# ====================================================================== a scripted software-ecosystem history
VEC_LIB = ("def sub(a, b):\n    return [x - y for x, y in zip(a, b)]\n\n\n"
           "def norm(v):\n    return sum(x * x for x in v) ** 0.5\n")
VEC_TESTS = "from lib import norm, sub\n\n\ndef test_norm():\n    assert norm(sub([3, 4], [0, 0])) == 5\n"
GEO_LIB_V1 = ("from deps.vecmath.lib import norm, sub\n\n\n"
              "def dist(a, b):\n    return norm(sub(a, b))\n\n\n"
              "def path_length(points):\n    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))\n\n\n"
              "def parse_point(s):\n    \"\"\"'x,y' -> (x, y)\"\"\"\n    return tuple(eval('(' + s + ')'))\n")
GEO_LIB_FIXED = GEO_LIB_V1.replace("    return tuple(eval('(' + s + ')'))\n",
                                   "    parts = s.split(',')\n    if len(parts) != 2:\n        raise ValueError('x,y')\n"
                                   "    return (float(parts[0]), float(parts[1]))\n")
GEO_LIB_RENAMED = GEO_LIB_FIXED.replace("def parse_point(s):", "def parse(s):")
GEO_TESTS = ("from lib import dist, parse_point, path_length\n\n\n"
             "def test_dist():\n    assert dist((0, 0), (3, 4)) == 5\n\n\n"
             "def test_path():\n    assert path_length([(0, 0), (3, 4), (3, 0)]) == 9\n\n\n"
             "def test_parse():\n    assert parse_point('1,2') == (1, 2)\n")
GEO_README = ("# geokit\n\nPlanar geometry for routing code: `dist(a, b)`, `path_length(points)`, "
              "`parse_point('x,y')`.\nImport as `from deps.geokit.lib import dist`.\n")
APP_SERVICE = ("from deps.geokit.lib import parse_point, path_length\n\n\n"
               "def length(args, caller):\n    return {'length': path_length([parse_point(s) for s in args['points']])}\n")
APP_TESTS = ("from service import length\n\n\n"
             "def test_length():\n    assert abs(length({'points': ['0,0', '3,4', '3,0']}, None)['length'] - 9) < 1e-9\n")
AUDIT_TESTS = ("from deps.geokit.lib import parse_point\n\n\n"
               "def test_parse_point_does_not_execute_input():\n"
               "    try:\n        r = parse_point(\"__import__('os').getpid(), 1\")\n    except Exception:\n        return\n"
               "    raise AssertionError('parse_point evaluated code: %r' % (r,))\n")


def software_history(world: World, step=None, prefix: str = "") -> dict:
    """A scripted history through the pkg tools only (scripted reference developers ana, ben, cat, dan, eve):
    two packages with a transitive dependency, an independent dependent that executes and deploys them, a
    verified security advisory, an independent review of a fix from a fork, a patch release that fixes the
    advisory, a breaking minor release caught by compatibility checks, a fork registered as a substitute package,
    a star (not adoption) and issue maintenance. `step(world, seconds)` advances time between acts (default:
    advance the clock). Returns the results of the main acts."""
    def adv(s):
        if step is not None:
            step(world, s)
        else:
            world.advance_time(world.now + s)
    devs = {}
    for h in ("ana", "ben", "cat", "dan", "eve"):
        handle = prefix + h
        if not world.state.has("principals", f"p:{handle}"):
            join(world, handle, 2_000_000)
        devs[h] = _Agent(handle)
        devs[h].joined = True
    ana, ben, cat, dan, eve = (devs[h] for h in ("ana", "ben", "cat", "dan", "eve"))
    hn = {h: devs[h].handle for h in devs}

    def must(agent, tool, **args):
        r = agent.ex(world, tool, **args)
        if not r.ok:
            raise RuntimeError(f"{agent.actor} {tool}: {r.error}")
        return r.result
    out: dict = {}
    must(eve, "code.create_repo", name="vecmath", visibility="public", license="open", description="vector maths")
    must(eve, "code.commit", repo=f"{hn['eve']}/vecmath", files={"lib.py": VEC_LIB, "tests/test_lib.py": VEC_TESTS,
                                                                 "README.md": "# vecmath\n\n`sub(a, b)`, `norm(v)`.\n"})
    must(eve, "pkg.register", name="vecmath", repo=f"{hn['eve']}/vecmath", description="Small vector maths: sub, norm")
    must(eve, "pkg.release", package="vecmath", version="1.0.0")
    must(ana, "code.create_repo", name="geokit", visibility="public", license="open", description="planar geometry")
    must(ana, "pkg.search", query="vector norm")
    must(ana, "code.commit", repo=f"{hn['ana']}/geokit", files={"lib.py": GEO_LIB_V1, "tests/test_lib.py": GEO_TESTS,
                                                                "README.md": GEO_README})
    must(ana, "pkg.depend", repo=f"{hn['ana']}/geokit", package="vecmath", constraint="^1.0.0")
    must(ana, "pkg.register", name="geokit", repo=f"{hn['ana']}/geokit",
         description="Planar geometry for routing: distances, path lengths, point parsing")
    out["geokit_1_0_0"] = must(ana, "pkg.release", package="geokit", version="1.0.0", notes="first release")
    adv(3600)
    # an independent developer finds it, incorporates it and runs it
    found = must(ben, "pkg.search", query="distance path length points geometry")
    out["ben_found"] = [r["package"] for r in found]
    app = f"{hn['ben']}/routecalc"
    must(ben, "code.create_repo", name="routecalc", visibility="private", description="route length service")
    manifest = {"entry": "service.py", "endpoints": {"length": {"handler": "length", "doc": "length of a polyline",
                                                                 "price_cents": 1,
                                                                 "params": {"type": "object", "additionalProperties": True}}}}
    must(ben, "code.commit", repo=app, files={"service.py": APP_SERVICE, "tests/test_service.py": APP_TESTS,
                                              "crucible.json": json.dumps(manifest)})
    out["ben_depend"] = must(ben, "pkg.depend", repo=app, package="geokit", constraint="^1.0.0")
    out["ben_ci"] = must(ben, "pkg.ci", repo=app)
    out["ben_exec"] = must(ben, "pkg.exec", repo=app, target="service:length", input={"points": ["0,0", "6,8"]})
    provs = world.query(ben.actor, "cloud.providers")
    acct = world.modules["bank"].primary_account(world.state, ben.actor)
    inst = must(ben, "cloud.rent", provider=sorted(provs, key=lambda p: -p["cores_free"])[0]["provider"], cores=1,
                account=acct)
    host = f"{prefix}routecalc"
    out["ben_deploy"] = must(ben, "pkg.deploy", repo=app, host=host, instance=inst["instance"])
    out["cat_call"] = must(cat, f"{host}.length", points=["0,0", "0,5"])
    must(dan, "pkg.star", package="geokit")
    adv(6 * 3600)
    # a security researcher demonstrates a defect with an executable reproduction
    audit = f"{hn['cat']}/geokit-audit"
    must(cat, "code.create_repo", name="geokit-audit", visibility="public", license="open", description="audit")
    must(cat, "code.commit", repo=audit, files={"tests/test_inject.py": AUDIT_TESTS})
    must(cat, "pkg.report", package="geokit", title="parse_point evaluates its input", labels=["bug"])
    out["advisory"] = must(cat, "pkg.advisory", package="geokit", version="1.0.0", kind="security", severity="high",
                           description="parse_point passes its argument to eval: code execution in dependents",
                           repro_repo=audit)
    adv(12 * 3600)
    # a contributor fixes it in a fork and proposes it; an independent developer reviews it; the maintainer merges
    must(dan, "pkg.fork", package="geokit", name="geokit-fix")
    fork = f"{hn['dan']}/geokit-fix"
    must(dan, "code.commit", repo=fork, files={"lib.py": GEO_LIB_FIXED}, message="parse without eval")
    pr = must(dan, "pkg.pr_open", repo=f"{hn['ana']}/geokit", source_repo=fork, title="Parse points without eval",
              body="Fixes the advisory.")
    out["pr"] = pr
    must(ben, "pkg.pr_diff", pr=pr["pr"])
    must(ben, "pkg.pr_review", pr=pr["pr"], verdict="approve", comment="no eval any more; tests pass")
    out["merge"] = must(ana, "pkg.pr_merge", pr=pr["pr"])
    out["geokit_1_0_1"] = must(ana, "pkg.release", package="geokit", version="1.0.1", notes="security fix")
    project = world.state.get("pkg_packages", "geokit")["project"]
    for i in world.query(ana.actor, "issues.list", {"project": project}):
        must(ana, "issues.close", issue=i["issue"], comment="fixed in 1.0.1")
    adv(6 * 3600)
    out["ben_upgrade"] = must(ben, "pkg.upgrade", repo=app, package="geokit")
    must(ben, "pkg.ci", repo=app)
    # a breaking change released as a minor version: caught by the API comparison and by the dependent's tests
    must(ana, "code.commit", repo=f"{hn['ana']}/geokit", files={"lib.py": GEO_LIB_RENAMED}, message="rename parse_point")
    out["geokit_1_1_0"] = must(ana, "pkg.release", package="geokit", version="1.1.0")
    out["ben_upgrade_breaking"] = must(ben, "pkg.upgrade", repo=app, package="geokit")
    # the fork becomes its own package (a substitute with the same API)
    must(dan, "pkg.register", name="geokitx", repo=fork, description="geokit with safe parsing")
    must(dan, "pkg.release", package="geokitx", version="1.0.0")
    adv(12 * 3600)
    out["developers"] = {h: devs[h].actor for h in devs}
    return out
