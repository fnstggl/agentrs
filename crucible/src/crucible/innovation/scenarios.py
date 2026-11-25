"""Reproducible scenarios that exercise the innovation mechanisms end to end (used by the gate tests).

Each scenario builds a world with `innovation_spec`, drives scripted reference agents (labelled: feasibility, not
learning) and the need-driven participants of `crucible.innovation.adopt`, and returns the live world together
with the agents and what was observed. Counterfactual arms are forks of one snapshot (`measure.arms`), exact under
the deterministic cost model.

* `market`     - an agent creates a route.shortest service; participants discover, test, adopt and use it.
* `capacity`   - tight compute: growing use overloads the service, a supply need emerges, participants publish it,
                 an entrepreneur builds a substitute, participants switch, the need resolves.
* `batch`      - a composite technology (route.batch over the route.shortest service) multiplies compute demand
                 on the base service until its constraint binds.
* `social`     - half of the participants discover only through what others post on the forum.
* `software`   - the scripted package history (`reference.software_history`).
"""
from __future__ import annotations

import copy
from typing import Callable

from ..core.kernel import SYSTEM, World
from . import measure as M
from .reference import ServiceCreator, BatchCreator, drive, software_history

HOUR, DAY = 3600, 86400
NEVER = 10 ** 12

SUITE_ROUTE = r'''
import heapq
import random


def generate(args, caller):
    rng = random.Random(int(args["seed"]))
    n = rng.randint(12, 30)
    edges = [[i, i + 1, rng.randint(1, 20)] for i in range(n - 1)]
    for _ in range(2 * n):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.append([u, v, rng.randint(1, 20)])
    s, t = rng.randrange(n), rng.randrange(n)
    while t == s:
        t = rng.randrange(n)
    return {"n": n, "edges": edges, "source": s, "target": t}


def _best(inp):
    adj = {}
    for u, v, w in inp["edges"]:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    dist = {inp["source"]: 0}
    q = [(0, inp["source"])]
    while q:
        d, u = heapq.heappop(q)
        if d > dist.get(u, 1e18):
            continue
        for v, w in adj.get(u, []):
            if d + w < dist.get(v, 1e18):
                dist[v] = d + w
                heapq.heappush(q, (d + w, v))
    return dist.get(inp["target"]), adj


def check(args, caller):
    inp, out = args["input"], args["output"]
    best, adj = _best(inp)
    try:
        path = list(out["path"])
    except Exception:
        return 0.0
    if not path or path[0] != inp["source"] or path[-1] != inp["target"]:
        return 0.0
    length = 0
    for a, b in zip(path, path[1:]):
        ws = [w for v, w in adj.get(a, []) if v == b]
        if not ws:
            return 0.0
        length += min(ws)
    return 1.0 if length == 0 else min(1.0, best / length)
'''

BROKEN_ROUTE = "def solve(args, caller):\n    return {'path': [args['source'], args['target']]}\n"


def _world(seed, **kw):
    from .worlds import innovation_spec
    return World.create(innovation_spec(seed, **kw))


class Remover:
    """A recorded intervention used in counterfactual arms: deletes every forum post that mentions `host` as soon
    as it appears (as its author would), i.e. removes the social channel for that product."""
    kind = "intervention"

    def __init__(self, host: str, board: str = "market"):
        self.host, self.board = host, board
        self.removed: list[str] = []

    def act(self, world):
        st = world.state
        for pid in st.find_ids("forum_posts", "board", self.board):
            p = st.tables["forum_posts"][pid]
            if p["status"] == "visible" and self.host in p["hosts"]:
                world.system_call("forum.delete", {"post": pid}, as_principal=p["author"])
                self.removed.append(pid)


def arms(world: World, agents: list, modifiers: dict[str, Callable], horizon_s: int, probe: Callable) -> dict:
    """Fork `world` once per arm from one snapshot; `modifiers[name](fork, agents_copy)` changes that arm (None: no
    change), then every arm runs the same horizon with copies of the same agents and is measured by `probe`."""
    blob = world.snapshot_blob()
    t0 = world.now
    out = {}
    for name, mod in modifiers.items():
        w = World.from_blob(blob, world_id=f"{world.world_id}~{name}")
        ag = copy.deepcopy(agents)
        try:
            extra = mod(w, ag) if mod else None
            drive(w, ag, t0 + horizon_s)
            out[name] = {**probe(w), "modifier": extra, "invariants": w.check_invariants()}
        finally:
            w.close()
    return {"fork_t": t0, "horizon_s": horizon_s, "arms": out}


# ====================================================================== market (Gate J, SP-6.3)
def market(seed: int = 11, participants: int = 24, days: float = 4.0, broken: bool = False, tester: bool = False,
           contrast: bool = True, software: bool = False) -> dict:
    w = _world(seed, participants=participants, adopt={"start_hours": [8, 40], "per_day": [6, 12]})
    ana = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=3, at=6 * HOUR)
    agents: list = [ana]
    if broken:
        agents.append(ServiceCreator("mal", "route.shortest", "malroute", price=1, at=6 * HOUR,
                                     files={**_files("route.shortest", 1, 1), "service.py": BROKEN_ROUTE}))
    drive(w, agents, 5 * HOUR)

    def probe(x):
        prods = [p for p in M.agent_products(x) if p.get("host") == "anaroute"]
        return {"adopters": len(M.adopters(x, prods[0])) if prods else 0,
                "users": len(M.users(x, prods[0])) if prods else 0, "compute": M.compute_use(x, 5 * HOUR),
                "participants": x.state.count("adopt_people"),
                "satisfied_requests": _satisfied(x), "open_provide_needs": _open(x, "provide", "cap:route.shortest")}
    res = {}
    if contrast:
        def degrade(x, ag):
            ag[0].tier = 0
            return "route.shortest service built from the weaker reference tier (same price, same declaration)"

        def absent(x, ag):
            ag[0].at = NEVER
            return "the agent does not create the service"
        res["arms"] = arms(w, agents, {"product": None, "degraded": degrade, "absent": absent},
                           int(days * DAY) - 5 * HOUR, probe)
    if tester:
        drive(w, agents, 2 * DAY)
    else:
        drive(w, agents, int(days * DAY))
    if tester:
        tess = ServiceCreator("tess", "route.shortest", "tessnone", at=NEVER)
        tess.ensure(w)
        tess.ex(w, "code.create_repo", name="route-suite", visibility="public", license="open",
                description="independent route.shortest test suite")
        tess.ex(w, "code.commit", repo="tess/route-suite", files={"suite.py": SUITE_ROUTE})
        res["verifications"] = {}
        for d in w.query("p:tess", "caps.declarations", {}):
            r = tess.ex(w, "caps.verify", declaration=d["declaration"], suite_repo="tess/route-suite", cases=8)
            res["verifications"][d["host"]] = r.result if r.ok else {"error": r.error}
        drive(w, agents + [tess], int(days * DAY))
    if software:
        res["software"] = software_history(w, step=lambda x, s: drive(x, agents, x.now + s), prefix="d")
    return {"world": w, "agents": agents, **res}


def _files(need, tier, price):
    from .reference import service_files
    return service_files(need, tier, price)


def _satisfied(w) -> int:
    """Participants' task requests answered with a score of at least their own threshold (their experience)."""
    n = 0
    for _, r in w.state.scan("adopt_people"):
        n += sum(1 for x in r["recent"] if x >= r["threshold"])
    return n + sum(1 for _, d in w.state.scan("adopt_decisions") if d["adopted"])


def _open(w, kind, target) -> int:
    return sum(1 for _, n in w.state.scan("needs_open") if n["kind"] == kind and n["target"] == target
               and n["status"] != "resolved")


# ====================================================================== capacity (Gate L chain A, SP-6.4)
def capacity(seed: int = 2, participants: int = 36, days: float = 7.0, counterfactual: bool = True,
             names: dict | None = None) -> dict:
    """`names` renames the agents and hosts ({"ana": ..., "anaroute": ..., "bo": ..., "boroute": ...}) without
    changing anything else (used for the rename-invariance check of SP-6.4)."""
    nm = {"ana": "ana", "anaroute": "anaroute", "bo": "bo", "boroute": "boroute", **(names or {})}
    w = _world(seed, participants=participants, params={"cu_per_ms": 5000.0},
               adopt={"per_day": [10, 16], "start_hours": [1, 72], "publish_p": 0.6})
    ana = ServiceCreator(nm["ana"], "route.shortest", nm["anaroute"], tier=1, price=3, at=6 * HOUR)
    bo = ServiceCreator(nm["bo"], "route.shortest", nm["boroute"], tier=1, price=3, on_request=True, min_requesters=2,
                        cores=2)
    agents = [ana, bo]
    drive(w, agents, 7 * HOUR)
    res = {}
    if counterfactual:
        def remove(x, ag):
            x.system_call("cloud.undeploy", {"host": nm["anaroute"]}, as_principal=ag[0].actor)
            return f"{nm['anaroute']} withdrawn right after it was created"

        def probe(x):
            st = x.state
            sup = [n for _, n in st.scan("needs_open") if n["kind"] == "supply"]
            new = [d for _, d in st.scan("cap_decls") if d["host"] == nm["boroute"]]
            return {"adoptions_first_service": sum(1 for _, a in st.scan("cap_adoptions") if a["host"] == nm["anaroute"]),
                    "supply_needs": len(sup), "new_service_built": len(new),
                    "adoptions_new_service": sum(1 for _, a in st.scan("cap_adoptions") if a["host"] == nm["boroute"]),
                    "needs_addressed_by_new": len(M.needs_addressed_by_new_services(x))}
        res["counterfactual"] = arms(w, agents, {"factual": None, "removed": remove}, int(days * DAY) - 7 * HOUR, probe)
    drive(w, agents, int(days * DAY))
    return {"world": w, "agents": agents, "names": nm, **res}


# ====================================================================== batch (Gate L chain C, SP-6.4)
def batch(seed: int = 5, days: float = 5.0, counterfactual: bool = True) -> dict:
    w = _world(seed, participants=24, needs={"route.shortest": 1.0, "route.batch": 1.0},
               params={"cu_per_ms": 5000.0}, adopt={"per_day": [8, 12], "start_hours": [1, 24], "wtp_cents": [10, 20]})
    base = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=1, at=2 * HOUR)
    tech = BatchCreator("bix", "batchroute", "anaroute", price=6, at=DAY)
    agents = [base, tech]
    drive(w, agents, 20 * HOUR)
    res = {}
    if counterfactual:
        def remove(x, ag):
            ag[1].at = NEVER
            return "the composite batch technology is never created"

        def probe(x):
            st = x.state
            cu = sum(r["cu"] for k, r in st.scan("svc_hours") if r["host"] == "anaroute" and r["hour"] >= 20)
            cu_days = sum(r["cu"] for k, r in st.scan("svc_days") if r["host"] == "anaroute" and int(k.split("|")[1]) >= 1)
            sup = [n for _, n in st.scan("needs_open") if n["kind"] == "supply" and n["target"] == "cap:route.shortest"]
            return {"base_cu_since_fork": round(cu_days, 1), "base_cu_last_week_hours": round(cu, 1),
                    "batch_adoptions": sum(1 for _, a in st.scan("cap_adoptions") if a["host"] == "batchroute"),
                    "base_overload": sum(r.get("overload", 0) for k, r in st.scan("svc_days") if r["host"] == "anaroute"),
                    "supply_needs_on_base": len(sup)}
        res["counterfactual"] = arms(w, agents, {"factual": None, "removed": remove}, int(days * DAY) - 20 * HOUR, probe)
    drive(w, agents, int(days * DAY))
    return {"world": w, "agents": agents, **res}


# ====================================================================== social (Gate L chain B)
def social(seed: int = 7, days: float = 5.0, counterfactual: bool = True) -> dict:
    w = _world(seed, participants=24, needs={"sort.records": 1.0}, channels={"search": 1.0, "forum": 1.0},
               adopt={"recommend_p": 1.0, "start_hours": [4, 60], "retry_hours": [6, 12]})
    ana = ServiceCreator("ana", "sort.records", "anasort", tier=1, price=2, at=2 * HOUR)
    agents = [ana]
    drive(w, agents, 3 * HOUR)
    res = {}
    if counterfactual:
        def remove(x, ag):
            ag.append(Remover("anasort"))
            return "every forum post mentioning anasort is deleted by its author as soon as it appears"

        def probe(x):
            st = x.state
            forum_people = {p for p, r in st.scan("adopt_people") if r["channel"] == "forum"}
            ad = {a["adopter"] for _, a in st.scan("cap_adoptions") if a["host"] == "anasort"}
            prod = [p for p in M.agent_products(x) if p.get("host") == "anasort"]
            tx = 0
            if prod:
                for tid in st.find_ids("txns", "kind", "sale"):
                    t = st.tables["txns"][tid]
                    if t["ref"] == "svc:anasort":
                        payer = st.tables["accounts"][[a for a, d in t["postings"] if d < 0][0]]["owner"]
                        tx += payer in forum_people
            posts = sum(1 for _, p in st.scan("forum_posts") if "anasort" in p["hosts"])
            return {"forum_channel_participants": len(forum_people), "forum_channel_adopters": len(ad & forum_people),
                    "search_channel_adopters": len(ad - forum_people), "forum_channel_paid_calls": tx,
                    "posts_mentioning_product": posts}
        res["counterfactual"] = arms(w, agents, {"factual": None, "removed": remove}, int(days * DAY) - 3 * HOUR, probe)
    drive(w, agents, int(days * DAY))
    return {"world": w, "agents": agents, **res}


# ====================================================================== software (SP-6.5, Gate L chain D)
def software(seed: int = 3) -> dict:
    w = _world(seed, with_adopters=False)
    out = software_history(w)
    return {"world": w, "history": out}


def close(res: dict) -> None:
    w = res.get("world")
    if w is not None:
        w.close()
