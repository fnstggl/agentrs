"""Tasks: verifiable outcomes in Crucible worlds, organised into disjoint splits. PROTECTED (not candidate-editable).

A `TaskSpec` names a task family, a world family, a seed, a level and a start state. `build(spec)` returns the
Crucible `GoalContract`, the world, the actor and what the actor already did (for scripted start states).

Task families. Each is an *outcome* checked on authoritative state; none prescribes how to reach it.

    found     own a registered company
    fund      a company you own holds at least $1,000
    code      own a repository whose `solve` endpoint solves a public need format with verified quality
    apicall   make paid calls to a service operated by somebody else
    deploy    operate a live deployment
    list      have an active market listing for a service you operate
    service   Crucible's `service` goal: serve k requests in 7 days with availability >= 0.95
    revenue   Crucible's `revenue` goal: independent revenue over a 30-day window
    repair    the `service` goal, starting from a service that is listed but failing (down, or badly overpriced)

Start states (``start``): ``scratch`` is the goal's own initial condition. The others run a documented scripted
prefix as the actor, through ordinary recorded commands (found a company / commit a reference implementation /
deploy / list), and stop there. They exist so that a curriculum can put the learner near an outcome before it can
reach it from scratch; the learner sees the identifiers the prefix created, never its actions as instructions.

Splits. Seeds and world families are disjoint by construction and `check_split` enforces it:

    train       seeds 10,000,000+     world families: tiny, bazaar, cheap_compute_closed_markets,
                                      expensive_compute_open_distribution
    curriculum  seeds 5,000,000+      same families (curriculum competence estimates only)
    select      seeds 1,000,000+      train families and entrenched_incumbents, fragmented_networks
                                      (algorithm selection by the Meta-Researcher)
    final       seeds < 1,000,000     train families, labor_rich_fragile_infra, novel_needs
                                      (locked: sealed list, used only for final reports)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .. import ensure_crucible
from ..evalkit import metrics as _m  # noqa: F401  (registers the sl_* metrics)

ensure_crucible()
from crucible.goals.contracts import GoalContract, generate_goal  # noqa: E402

DAY = 86400
TRAIN_FAMILIES = ("tiny", "bazaar", "cheap_compute_closed_markets", "expensive_compute_open_distribution")
SELECT_FAMILIES = TRAIN_FAMILIES + ("entrenched_incumbents", "fragmented_networks")
FINAL_FAMILIES = TRAIN_FAMILIES + ("labor_rich_fragile_infra", "novel_needs")
SPLITS = {"train": (10_000_000, 2**30, TRAIN_FAMILIES), "curriculum": (5_000_000, 6_000_000, TRAIN_FAMILIES),
          "select": (1_000_000, 2_000_000, SELECT_FAMILIES), "final": (0, 1_000_000, FINAL_FAMILIES)}
TASK_FAMILIES = ("found", "fund", "code", "apicall", "deploy", "list", "service", "revenue", "repair")
STARTS = ("scratch", "company", "coded", "deployed", "listed", "overpriced", "outage")
STAGE_ORDER = ("company", "coded", "deployed", "listed")


@dataclass(frozen=True)
class TaskSpec:
    family: str
    world_family: str = "tiny"
    seed: int = 10_000_000
    level: int = 0
    start: str = "scratch"
    split: str = "train"

    @property
    def key(self) -> str:
        return f"{self.family}:{self.world_family}:{self.seed}:{self.level}:{self.start}"

    @property
    def arm(self) -> tuple:
        return (self.family, self.level, self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        return TaskSpec(**{k: d[k] for k in ("family", "world_family", "seed", "level", "start", "split") if k in d})


def check_split(spec: TaskSpec) -> None:
    lo, hi, fams = SPLITS[spec.split]
    if not (lo <= spec.seed < hi):
        raise ValueError(f"seed {spec.seed} is outside the {spec.split} split [{lo}, {hi})")
    if spec.world_family not in fams:
        raise ValueError(f"world family {spec.world_family} is not part of the {spec.split} split")
    if spec.family not in TASK_FAMILIES or spec.start not in STARTS:
        raise ValueError(f"unknown task family/start {spec.family}/{spec.start}")


def task(family: str, world_family: str, index: int, split: str = "train", level: int = 0,
         start: str = "scratch") -> TaskSpec:
    """The `index`-th task of a split (seed = split base + index)."""
    spec = TaskSpec(family, world_family, SPLITS[split][0] + int(index), level, start, split)
    check_split(spec)
    return spec


# ------------------------------------------------------------------------------------------------ goals
def world_needs(world_family: str, seed: int) -> list[str]:
    from crucible.worldgen import generate
    spec = generate({"family": world_family}, seed)
    civ = [m for m in spec.modules if m["name"] == "civ"]
    firms = civ[0]["config"].get("firms", []) if civ else []
    return sorted({f["need"] for f in firms})


def _goal(family, desc, world, success, progress, horizon_d, endowment, split, level, params=None, every=0):
    return GoalContract(f"sl.{family}", desc, world, "$self", success, progress, int(horizon_d * DAY), 1,
                        endowment=int(endowment), split=split, difficulty=0.0, params=dict(params or {}, level=level),
                        progress_every_s=every)


def build_goal(spec: TaskSpec) -> GoalContract:
    rng = np.random.default_rng([spec.seed, spec.level, int(hashlib.sha1(spec.family.encode()).hexdigest()[:6], 16)])
    world = {"family": spec.world_family, "seed": int(spec.seed), "overrides": {}}
    endow = int(rng.choice([300_000, 500_000, 1_000_000]))
    me = {"principal": "$self"}
    f, lvl = spec.family, spec.level
    if f == "found":
        return _goal(f, "Own an organisation that is registered as a company.", world,
                     {"metric": "sl_companies", "args": me, "op": ">=", "value": 1},
                     {"metric": "sl_companies", "args": me, "target": 1}, 2, endow, spec.split, lvl)
    if f == "fund":
        amt = [100_000, 200_000][min(lvl, 1)]
        return _goal(f, f"Own a registered company whose own bank accounts hold at least ${amt // 100:,}.", world,
                     {"all": [{"metric": "sl_companies", "args": me, "op": ">=", "value": 1},
                              {"metric": "sl_org_cash", "args": me, "op": ">=", "value": amt}]},
                     {"metric": "sl_org_cash", "args": me, "target": amt}, 3, endow, spec.split, lvl, {"amount": amt})
    if f == "code":
        needs = world_needs(spec.world_family, spec.seed)
        need = str(needs[int(rng.integers(0, len(needs)))])
        thr = 0.8
        args = {"principal": "$self", "need": need, "n": 3}
        return _goal(f, f"Own a repository that can be deployed as a service (a crucible.json manifest and code) whose "
                        f"endpoint `solve` solves the public need format `{need}` with verified quality of at least {thr}.",
                     world, {"metric": "sl_solver_quality", "args": args, "op": ">=", "value": thr},
                     {"metric": "sl_solver_quality", "args": args, "target": thr}, 3, endow, spec.split, lvl,
                     {"need": need, "threshold": thr})
    if f == "apicall":
        n = [3, 10][min(lvl, 1)]
        return _goal(f, f"Make at least {n} successful paid calls to services operated by other organisations.", world,
                     {"metric": "sl_api_calls", "args": me, "op": ">=", "value": n},
                     {"metric": "sl_api_calls", "args": me, "target": n}, 2, endow, spec.split, lvl, {"n": n})
    if f == "deploy":
        return _goal(f, "Operate a live service: a deployment of your own code that is running and answers requests.",
                     world, {"metric": "sl_live_services", "args": me, "op": ">=", "value": 1},
                     {"metric": "sl_ladder", "args": me, "target": 0.8}, 4, endow, spec.split, lvl)
    if f == "list":
        return _goal(f, "Have an active market listing for a live service that you operate.", world,
                     {"metric": "sl_listed_services", "args": me, "op": ">=", "value": 1},
                     {"metric": "sl_ladder", "args": me, "target": 1.0}, 5, endow, spec.split, lvl)
    if f in ("service", "repair", "revenue"):
        g = generate_goal("revenue" if f == "revenue" else "service", spec.world_family, spec.seed, lvl, split=spec.split)
        g = copy.deepcopy(g)
        g.world = dict(g.world, seed=int(g.world["seed"]))
        g.family = f"sl.{f}"
        g.endowment = max(int(g.endowment), 500_000 if f != "revenue" else int(g.endowment))
        return g
    raise ValueError(f"unknown task family {f}")


# ------------------------------------------------------------------------------------------------ start states
def run_prefix(world, actor: str, stage: str, need: str | None = None) -> dict:
    """Scripted start state. Runs as the actor through ordinary recorded commands, one per simulated hour.

    Stages are cumulative: ``company`` (found, register, open an account, move 70% of the personal balance),
    ``coded`` (+ repository with the best reference implementation of `need`), ``deployed`` (+ rent a core and
    deploy), ``listed`` (+ market listing at the cheapest rival's price). ``overpriced`` is ``listed`` at four times
    that price; ``outage`` is ``listed`` followed by releasing the instance, so the service is down."""
    from crucible.agents.base import Runner
    from crucible.agents.scripted import ScriptPolicy, call, cheapest_rival, ok, service_files
    from crucible.economy.needs import NEEDS, best_tier, tier_source

    target = {"overpriced": "listed", "outage": "listed"}.get(stage, stage)
    depth = STAGE_ORDER.index(target)
    info: dict = {"stage": stage}

    class _Prefix(ScriptPolicy):
        name = f"prefix:{stage}"

        def script(self):
            nd = need
            r = yield call("market.browse", kind="api", sort="price", limit=20)
            rows = r["result"] if ok(r) else []
            if nd is None:
                seen = sorted({x["need"] for x in rows if x.get("need") in NEEDS})
                nd = seen[0] if seen else "route.shortest"
            info["need"] = nd
            mult = 4.0 if stage == "overpriced" else 1.0
            price = max(1, int(round(cheapest_rival(rows, nd, 1.0) * mult)))
            r = yield call("bank.accounts")
            mine = [a for a in (r or {}).get("result", []) if a["owner"] == self.actor]
            capital = int(0.7 * mine[0]["balance"]) if mine else 0
            org, acct, personal = yield from self.found_company("labs", capital)
            info.update(org=org, account=acct, personal_account=personal, price=price)
            if acct is None or depth < 1:
                return
            r = yield call("code.create_repo", name=nd.replace(".", "-"), owner=org, description=NEEDS[nd].title)
            if not ok(r):
                return
            repo = r["result"]["repo"]
            info["repo"] = repo
            yield call("code.commit", repo=repo, files=service_files(nd, tier_source(nd, best_tier(nd)), price),
                       message="first version")
            if depth < 2:
                return
            host = f"{self.handle[:12]}-{nd.split('.')[1][:6]}"
            r = yield call("cloud.providers")
            provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= 1 and not p["in_outage"]),
                           key=lambda p: (p["price_core_hour_cents"], p["provider"]))
            if not provs:
                return
            r = yield call("cloud.rent", provider=provs[0]["provider"], cores=1, account=acct, owner=org)
            if not ok(r):
                return
            inst = r["result"]["instance"]
            r = yield call("cloud.deploy", repo=repo, host=host, instance=inst)
            if not ok(r):
                return
            info.update(host=host, instance=inst)
            if depth < 3:
                return
            r = yield call("market.list", kind="api", title=f"{host}: {NEEDS[nd].title}", price_cents=int(price),
                           seller=org, host=host, endpoint="solve", need=nd,
                           description=f"{NEEDS[nd].title}. Keywords: {' '.join(NEEDS[nd].keywords)} {host}.crucible")
            if ok(r):
                info["listing"] = r["result"]["listing"]
            if stage == "outage":
                yield call("cloud.release", instance=inst)

    pol = _Prefix()
    run = Runner(world, [(actor, pol)], dt=3600, record=True)
    for _ in range(60):
        run.round()
        if pol.finished:
            break
    info["history"] = [(s.action, s.result) for s in run.trajs[actor].steps if s.action]
    info["errors"] = list(pol.log)
    return info


def build(spec: TaskSpec, handle: str = "agent"):
    """Instantiate a task: ``(goal, world, actor, start_info)``."""
    from crucible.goals.contracts import instantiate
    check_split(spec)
    goal = build_goal(spec)
    world, actor = instantiate(goal, handle)
    info: dict = {"stage": "scratch", "history": []}
    start = spec.start
    if spec.family == "repair" and start == "scratch":
        start = "outage" if (spec.seed % 2) else "overpriced"
    if start != "scratch":
        info = run_prefix(world, actor, start, goal.params.get("need"))
    return goal, world, actor, info


# ------------------------------------------------------------------------------------------------ locked final set
def lock_final(path: str, specs: list[TaskSpec]) -> str:
    """Write the locked final evaluation list with a SHA-256 seal and make the file read-only."""
    for s in specs:
        if s.split != "final":
            raise ValueError("only final-split tasks can be locked")
        check_split(s)
    body = json.dumps([s.to_dict() for s in specs], sort_keys=True)
    seal = hashlib.sha256(body.encode()).hexdigest()
    if os.path.exists(path):
        os.chmod(path, 0o644)
    with open(path, "w") as f:
        json.dump({"seal": seal, "n": len(specs), "tasks": json.loads(body)}, f, indent=1)
    os.chmod(path, 0o444)
    return seal


def load_final(path: str) -> list[TaskSpec]:
    d = json.load(open(path))
    body = json.dumps(d["tasks"], sort_keys=True)
    if hashlib.sha256(body.encode()).hexdigest() != d["seal"]:
        raise ValueError(f"locked final task list {path} was altered (seal mismatch)")
    return [TaskSpec.from_dict(t) for t in d["tasks"]]


def assert_disjoint(train: list[TaskSpec], heldout: list[TaskSpec]) -> None:
    a = {(s.world_family, s.seed) for s in train}
    b = {(s.world_family, s.seed) for s in heldout}
    if a & b:
        raise ValueError(f"{len(a & b)} worlds appear in both training and held-out tasks")
