"""The Crucible 4.0 numeric view of numeric-lever families (INTERFACE_CONTRACTS section 8).

`RapidLeverEngine` implements the 4.0 mechanism `Engine` contract (`crucible.mechanisms.base`) on top of a Tier 1
engine: settings are the family's levers, readings its visible gauges, revenue and cost its money flows. Because it
is an ordinary registered engine (`engine = "rapid"`), everything 4.0 built on that contract works unchanged:

* `MechSim` / `BatchSim` run it, so `research.envs.MechVecEnv` and every learner in `crucible.research.learners`
  (hold, random, probe_exploit, hill_climb, ppo_gru, model_based, ...) train and evaluate on it with the 61-float
  `EnvSpec` layout;
* the kernel module `rig` hosts it as a facility with real ledger money (`kernel_world`), which is the kernel twin
  of the fast simulator.

`make_task(family, seed)` builds a `research.tasks.Task` for it; with `validate=True` the task carries the private
default and reference nets (from the evaluator-side rapid oracle), so `Task.score` normalises results.

Other action kinds of a family (discrete decisions, queries, experiments, programs) are not part of this view:
discrete decisions stay at their defaults, and a lever left unset follows nothing - every lever is always set.
"""
from __future__ import annotations

from dataclasses import dataclass


from ..mechanisms.base import ENGINES, Engine, Gauge, Lever, StepResult, register_engine
from .engines import build as build_rapid
from .env import world_seed_of
from .families import FAMILIES, build_config

VIEW_VERSION = "rapid-lever/1"


def _view_parts(inner) -> tuple[list[dict], list[str]]:
    c = inner.config
    if inner.kind == "primitive_graph":
        return inner.public_levers(), inner.visible_gauges()
    if inner.kind == "graph_dynamical" and inner.sc == "routing":
        return inner.public_levers(), list(c["gauge_names"])
    if inner.kind == "state_machine" and inner.sc == "sign_switch":
        return inner.public_levers(), list(c["gauge_names"])
    if inner.kind == "discrete_event" and inner.sc == "inventory":
        lv = [{"name": c["order_name"], "min": 0.0, "max": float(c["max_order"]), "default": 0.0, "integer": True}]
        return lv, list(c["gauge_names"][:4 if c.get("shelf_life") else 3])
    raise ValueError(f"no numeric lever view for engine {inner.kind}")


@register_engine
class RapidLeverEngine(Engine):
    engine = "rapid"
    version = "1"

    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = build_rapid(config["rapid"])
        lv, gs = _view_parts(self.inner)
        self._levers = [Lever(x["name"], float(x["min"]), float(x["max"]), float(x["default"]), bool(x["integer"]))
                        for x in lv]
        self._gauges = [Gauge(g) for g in gs]
        self.agent = self.inner.agents[0]
        # array states travel in the facility's tick events; an event queue (lists, a heap) is recomputed on replay
        self.state_in_event = self.inner.kind != "discrete_event"

    @classmethod
    def generate(cls, rng, params):
        raise NotImplementedError("rapid lever mechanisms come from crucible.rapid families (lever_mech)")

    def levers(self, seat):
        return list(self._levers)

    def gauges(self, seat):
        return list(self._gauges)

    def brief(self, seat):
        return self.inner.brief(self.agent)

    @property
    def horizon(self):
        return self.inner.horizon

    @property
    def demand_income_cents_per_tick(self):
        return self.config.get("income_cents_per_tick")

    def init_state(self, rng):
        return self.inner.init_state(rng)

    def step(self, state, settings, rng, t):
        s = self.inner.copy_state(state)
        self.inner.apply(s, self.agent, {"kind": "levers", "values": dict(settings[0])}, rng, t)
        out = self.inner.advance(s, rng, t)
        nums = out["obs"][self.agent].get("numbers", {})
        rd = {}
        for g in self._gauges:
            v = nums.get(g.name)
            rd[g.name] = float(v) if v is not None and v == v else 0.0
        rev, cost = float(out["revenue"][self.agent]), float(out["cost"][self.agent])
        return StepResult(s, [rd], [rev], [cost], [rev - cost], {})

    def truth(self):
        tr = dict(self.inner.truth())
        tr.setdefault("nodes", [])
        tr.setdefault("edges", [])
        tr.setdefault("regimes", [])
        tr["family"] = self.config.get("family")
        tr["engine"] = self.engine
        tr["inner_engine"] = self.inner.kind
        tr["notes"] = "Tier 1 rapid family seen through the 4.0 numeric lever view"
        return tr


assert ENGINES["rapid"] is RapidLeverEngine


def lever_families() -> list[str]:
    return sorted(n for n, f in FAMILIES.items() if f.lever_view)


def lever_mech(family: str, seed: int, **difficulty) -> dict:
    """A 4.0 mechanism dict for a rapid lever family (what `rig`, `MechSim` and `BatchSim` take)."""
    if not FAMILIES[family].lever_view:
        raise ValueError(f"{family} has no numeric lever view")
    cfg = build_config(family, seed, **difficulty)
    inner = build_rapid(cfg)
    inc = cfg.get("income_cents_per_tick")
    if inc is None and inner.kind == "state_machine":
        inc = int(cfg["price"] * (max(cfg["base"]) + 10 * max(abs(s) for s in cfg["slope"]))) + 1
    if inc is None and inner.kind == "discrete_event":
        inc = int(cfg["price"] * cfg["demand"] * (1 + cfg["season_amp"]) * 4) + 1
    return {"engine": "rapid", "family": f"rapid:{family}", "seed": int(seed),
            "config": {"rapid": cfg, "family": family, "difficulty": dict(difficulty), "n_seats": 1,
                       "horizon": inner.horizon, "income_cents_per_tick": int(inc), "view": VIEW_VERSION}}


def _research():
    from ..research.tasks import ENDOWMENT_TICKS, Task
    return ENDOWMENT_TICKS, Task


def make_task(family: str, seed: int, validate: bool = False, **difficulty):
    """A `research.tasks.Task` over the lever view of a rapid family. With `validate`, the private default net and
    reference net come from the rapid evaluator (oracle) and the public goal sits half way between them."""
    ENDOWMENT_TICKS, Task = _research()
    mech = lever_mech(family, seed, **difficulty)

    @dataclass(frozen=True)
    class RapidTask(Task):
        def mechanism(self) -> dict:
            return lever_mech(family, self.seed, **difficulty)

    inc = int(mech["config"]["income_cents_per_tick"])
    h = int(mech["config"]["horizon"])
    meta = {"engine": "rapid", "transfer": "base" if FAMILIES[family].split == "train" else "structural",
            "split": FAMILIES[family].split}
    ws = world_seed_of(family, seed)
    if not validate:
        return RapidTask(f"rapid:{family}", int(seed), int(seed), ws, inc, ENDOWMENT_TICKS * inc, h, 1, False, **meta)
    from .families import make
    from .oracles import fork, oracle
    from .policies import run, wait
    env = make(family, seed, **difficulty)
    d = float(run(fork(env), wait)["value"]) / h
    r = float(oracle(fork(env))["value"]) / h
    return RapidTask(f"rapid:{family}", int(seed), int(seed), ws, inc, ENDOWMENT_TICKS * inc, h, 1, True,
                     int(round(h * (d + 0.5 * (r - d)))), d, r, None, "yes" if r > d else "unknown", (), **meta)


def kernel_world(family: str, seed: int, *, holder: str = "agent", tick_s: int = 3600, endowment: int = 10 ** 9,
                 **difficulty):
    """An event-sourced kernel world hosting the lever view of a rapid family as a `rig` facility with real ledger
    money; the world seed equals the rapid world seed, so the facility's tick generators are the fast simulator's.
    Returns (world, principal of the holder, mech)."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World, WorldSpec
    from ..mechanisms.module import rig_entry
    mech = lever_mech(family, seed, **difficulty)
    ws = world_seed_of(family, seed)
    entry = rig_entry(mech, id="r1", tick_s=tick_s, income_cents_per_tick=mech["config"]["income_cents_per_tick"],
                      holders=[holder])
    spec = WorldSpec(f"rapid_{family}", ws, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                             {"name": "admin", "config": {}}, {"name": "rig", "config": {"rigs": [entry]}}],
                     params={"family_split": FAMILIES[family].split})
    w = World.create(spec, world_id=f"rapid-{family}-{seed}", strict=True)
    r = w.system_call("admin.join", {"handle": holder, "endowment": int(endowment)})
    return w, r["principal"], mech
