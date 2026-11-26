"""Tier 2 domain 9.7 Engineering infrastructure (workstream I-2): substantially reduce infrastructure costs while
preserving reliability.

The world (a restriction of the civilisation generator, D-17): the computing-infrastructure subsystem with the
datacenter fleet (`fleet`, domains.infra_world) of an incumbent operator organisation whose engineers the participant
seat (`sre`) joins as an administrator, the tenants whose services run there, and the one population as the end
users of those services (`fleet` registers the population activity `apps`: individuals experience their region's
service quality, abandon services that fail them and adopt what their contacts use - their adoption IS the tenants'
traffic), plus the population's outside option (`elsewhere`).

Physics (executed, no judge): leased machines with boot lag, procurement lead time, wear-driven failures repaired by
technicians, M/M/1 latency against SLOs over a routed network, PUE and facility power, flat / time-of-use / spot
tariffs (spot: an autonomous utility prices from grid load including the operator's own draw), power caps and
breakers, fibre cuts, batch jobs with deadlines, tenant releases that change CPU per request unannounced, tenants
moving traffic after a week of bad service, SLA credits. Production runs a controller program from a repository
(trusted incumbent with parameters, or any participant program in the sandbox), deployed at the maintenance
window; `fleet.experiment` replays the operator's own recorded traffic and telemetry (never the hidden demand).

Objective (public): cut the operator's total cost (leases, energy, repairs, wages, SLA credits) over the window by a
target fraction relative to running the fleet as it is, without lowering reliability. Evaluator (evaluator side):
the same world instance run with nobody acting (the incumbent operation: same seed, same exogenous hours) is the
counterfactual; reliability is the share of requests served within the SLO, the batch on-time rate, and the fleet's
repair backlog at the end (a cut that lets failed machines pile up is a deferred outage, not a saving).

Configurations (structure measured on the built world: topology shape, number of sites (one / several), tariff,
batch, thermal PUE, power caps, breakers, fibre cuts, follow-the-sun demand, heat waves):

    train   single_site_tou        one site, star network, time-of-use tariff
    train   multisite_spot_batch   three sites, mesh network, spot tariff from the utility, batch jobs
    train   ring_thermal_caps      two sites on a ring, flat tariff, thermal PUE, power caps with breakers, cuts
    eval_compositional  mesh_tou_thermal_batch   mesh, several sites, time-of-use, thermal, power caps, batch
    eval_structural     follow_sun_heat          four sites on a ring, spot, demand peaking at different hours per
                                                 region (follow the sun) and heat waves (never seen in training)
"""
from __future__ import annotations

import json

from . import base
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "infrastructure"
HORIZON_DAYS = 35.0
SEAT = "sre"
N_POP = 600
TARGET = {1: 0.06, 2: 0.12, 3: 0.20}          # cost reduction vs. the incumbent operation over the window

_COMMON = {"thermal": False, "breakers": False, "power_cap": None, "link_cuts": False, "follow_sun": False,
           "heat_waves": 0.0, "batch": False}
CONFIGS = {
    "single_site_tou": {**_COMMON, "n_sites": 1, "n_regions": 3, "topology": "star", "tariff": "tou"},
    "multisite_spot_batch": {**_COMMON, "n_sites": 3, "n_regions": 4, "topology": "mesh", "tariff": "spot",
                             "batch": True},
    "ring_thermal_caps": {**_COMMON, "n_sites": 2, "n_regions": 3, "topology": "ring", "tariff": "flat",
                          "thermal": True, "breakers": True, "power_cap": [1.05, 1.25], "link_cuts": True},
    "mesh_tou_thermal_batch": {**_COMMON, "n_sites": 3, "n_regions": 4, "topology": "mesh", "tariff": "tou",
                               "thermal": True, "power_cap": [1.05, 1.25], "batch": True},
    "follow_sun_heat": {**_COMMON, "n_sites": 4, "n_regions": 4, "topology": "ring", "tariff": "spot",
                        "thermal": True, "link_cuts": True, "follow_sun": True, "heat_waves": 0.05},
}
SPLIT = {"single_site_tou": "train", "multisite_spot_batch": "train", "ring_thermal_caps": "train",
         "mesh_tou_thermal_batch": "eval_compositional", "follow_sun_heat": "eval_structural"}
DOCS = {
    "single_site_tou": "one datacenter behind a star network serving three regions; time-of-use energy prices",
    "multisite_spot_batch": "three datacenters in a mesh serving four regions; energy bought at the utility's spot "
                            "price (which the operator's own draw moves); tenants' batch jobs with deadlines",
    "ring_thermal_caps": "two datacenters on a ring with fibre cuts; flat tariff; PUE rises with heat and load; "
                         "power caps with breakers that trip a site dark",
    "mesh_tou_thermal_batch": "a mesh of three sites with time-of-use prices, thermal PUE, power caps and batch jobs: "
                              "every part seen in training, never together",
    "follow_sun_heat": "four sites on a ring, spot prices, regional demand peaking at different hours (follow the "
                       "sun) and heat waves: structure no training world has",
}


# ============================================================================================ world
def build(config, seed, overrides):
    fleet = dict(CONFIGS[config.name])
    fleet["operators"] = [SEAT]
    fleet.update(overrides.get("fleet") or {})
    pop = {"n": int(overrides.get("n", N_POP)), "dims": {"H": 16}}
    r = base.civ_recipe(seed, subsystems=["computing_infrastructure"], modules=["code", "docs", "seat", "elsewhere"],
                        population_modules=("pop",), module_config={"fleet": fleet, "pop": pop},
                        params=overrides.get("params"))
    # the population and the seats exist before the fleet's genesis (its users and its engineers)
    mods = [m for m in r["module_list"] if m != "fleet"] + ["fleet"]
    r["module_list"] = mods
    return r


def tier3(config, merged):
    """In a civilisation the fleet keeps the configuration's structure and the seat as its engineer."""
    mine = build(config, 0, {})
    merged["modules"]["fleet"] = {**merged["modules"].get("fleet", {}), **mine["modules"]["fleet"]}
    mods = [m for m in merged["module_list"] if m != "fleet"] + ["fleet"]
    merged["module_list"] = mods
    return merged


def structure(world):
    cfg = world.modules["fleet"].cfg
    return {"topology": cfg["topology"], "sites": "one" if int(cfg["n_sites"]) == 1 else "several",
            "tariff": cfg["tariff"], "batch": bool(cfg["batch"]), "thermal": bool(cfg["thermal"]),
            "power_caps": bool(cfg.get("power_cap")), "breakers": bool(cfg["breakers"]),
            "fibre_cuts": bool(cfg["link_cuts"]), "follow_sun": bool(cfg["follow_sun"]),
            "heat_waves": bool(cfg["heat_waves"]), "end_users": "pop" in world.modules}


# ============================================================================================ objective and evaluator
def objective(world, principal, difficulty):
    t = TARGET.get(int(difficulty), TARGET[3])
    return {"text": (f"You are an infrastructure engineer of the datacenter operator. Within {HORIZON_DAYS:.0f} days, "
                     f"reduce the operator's total infrastructure cost (machine leases, energy, repairs, staff and "
                     f"SLA credits) by at least {100 * t:.0f}% compared with running the fleet as it is, while "
                     f"preserving reliability: the share of requests served within their SLO, batch deadlines and the "
                     f"fleet's repair backlog must not get worse."),
            "cost_reduction": t, "window_days": HORIZON_DAYS, "difficulty": int(difficulty)}


_SHADOW: dict[str, object] = {}


def shadow(world):
    """The counterfactual: the same world instance (spec and seed) where nobody acts, advanced to `world.now`
    (evaluator side; cached per spec and advanced incrementally)."""
    from ..core.kernel import World
    key = world.spec.hash()
    w0 = _SHADOW.get(key)
    if w0 is None or w0.now > world.now:
        for k in list(_SHADOW):
            if k != key:
                _SHADOW.pop(k).close()
        w0 = World.create(world.spec, world_id=f"shadow-{key[:12]}")
        _SHADOW[key] = w0
    if w0.now < world.now:
        w0.advance_time(world.now)
    return w0


def _backlog(world) -> int:
    import numpy as np
    from .infra_world import FAILED, REPAIR
    status = world.state.arrays["fl_status"]
    return int(np.sum(np.isin(status, (FAILED, REPAIR))))


def evaluate(world, principal, obj):
    from .infra_world import operator_costs, reliability
    day = 86400
    t1 = (world.now // day) * day
    if t1 < day:
        return {"success": False, "score": 0.0, "details": {"note": "no full day yet"}}
    w0 = shadow(world)
    c, c0 = operator_costs(world, 0, t1), operator_costs(w0, 0, t1)
    r, r0 = reliability(world, 0, t1 // 3600), reliability(w0, 0, t1 // 3600)
    b, b0 = _backlog(world), _backlog(w0)
    red = 1.0 - c["total"] / c0["total"] if c0["total"] > 0 else 0.0
    target = float(obj.get("cost_reduction", TARGET[1]))
    rel_ok = r["attainment"] >= r0["attainment"] - 0.001 and r["batch_on_time"] >= r0["batch_on_time"] - 0.05
    health_ok = b <= b0 + 2
    ok = red >= target and rel_ok and health_ok
    score = max(0.0, min(1.0, red / target)) * (1.0 if rel_ok and health_ok else 0.25)
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {"cost": c, "baseline_cost": c0, "cost_reduction": round(red, 4),
                        "attainment": round(r["attainment"], 6), "baseline_attainment": round(r0["attainment"], 6),
                        "batch_on_time": r["batch_on_time"], "baseline_batch_on_time": r0["batch_on_time"],
                        "repair_backlog": b, "baseline_repair_backlog": b0, "reliability_ok": rel_ok,
                        "health_ok": health_ok, "days": t1 // day}}


# ============================================================================================ policies
TUNED = {"power": "follow_demand", "headroom": 2.2, "min_on": 2, "forecast": "day_ago_max", "placement": "nearest",
         "batch": "cheapest", "spread": 0.0}


class FleetReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): watch two days of the operator's own dashboard, retune
    the incumbent controller (power machines to the forecast need at a utilisation the latency SLOs tolerate, run
    batch in cheap hours), test it on the recorded traffic with `fleet.experiment`, deploy it at the maintenance
    window, then return surplus machines beyond the observed peak at that utilisation plus failure spares. It never
    fires the repair crew."""
    burst = 6
    replan_s = 6 * 3600
    params = TUNED
    retire_headroom = 2.3          # machines kept per machine of peak work (latency: utilisation near 0.45)
    spares = 4

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.repo = None
        self.target = None
        self.stage = "status"
        self.peak = {}
        self.cores = 32
        self.machines = {}
        self.deployed_at = None

    def plan(self, obs):
        day = 86400
        if self.stage == "status":
            return [act_("fleet.status", "status")]
        if self.stage == "watch" and self.t >= 2 * day:
            self.stage = "measure"
            return [act_("fleet.measure", "measure", window_h=48)]
        if self.stage == "commit":
            self.stage = "wait"
            return [act_("code.commit", "commit", repo=self.repo, files={"controller.json": json.dumps(self.params)},
                         message="follow demand with headroom; batch in cheap hours"),
                    act_("fleet.experiment", "experiment", repo=self.repo, target=self.target, hours=48),
                    act_("fleet.propose_change", "proposed", repo=self.repo, target=self.target,
                         note="retuned incumbent controller")]
        if self.stage == "deployed" and self.deployed_at is not None and self.t >= self.deployed_at + 2 * day:
            self.stage = "resize"
            return [act_("fleet.measure", "measure2", window_h=72), act_("fleet.status", "status2")]
        if self.stage == "retire":
            self.stage = "steady"
            acts = []
            for site, (installed, need) in sorted(self.machines.items()):
                surplus = installed - need
                if surplus > 0:
                    acts.append(act_("fleet.retire", None, site=site, n=int(surplus)))
            return acts
        if self.stage == "wait" and self.deployed_at is not None and self.t >= self.deployed_at:
            self.stage = "deployed"
        if self.stage in ("wait", "deployed", "steady", "watch"):
            return [act_("fleet.status", "poll")]
        return []

    def on_result(self, tag, result, action):
        if result is None:
            return
        if tag == "status":
            if not result.get("you_operate"):
                self.stage = "stuck"
                return
            dep = result["deployed"]
            self.repo, self.target = dep["repo"], dep["target"]
            self.stage = "watch"
        elif tag in ("measure", "measure2"):
            peak = {}
            for row in result.get("rows", []):
                for s, v in row["sites"].items():
                    load = float(v["util"]) * float(v["on"])         # machines' worth of work
                    peak[s] = max(peak.get(s, 0.0), load)
            self.peak = peak
            if tag == "measure":
                self.stage = "commit"
        elif tag == "proposed":
            self.deployed_at = int(result.get("deploy_at") or self.t)
        elif tag == "status2":
            m = result.get("machines") or {}
            self.machines = {}
            for s, c in m.items():
                installed = sum(int(c.get(k, 0)) for k in ("on", "booting", "off", "failed", "repairing"))
                need = int(-(-self.peak.get(s, 0.0) * self.retire_headroom // 1)) + self.spares
                self.machines[s] = (installed, min(installed, need))
            self.stage = "retire"


class FireRepairCrew(ScriptedReference):
    """Degenerate: the one-call staffing cut (no technicians: failed machines are never repaired)."""
    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.done = False

    def plan(self, obs):
        if self.done:
            return []
        self.done = True
        return [act_("fleet.staff", None, technicians=0)]


class StarveCapacity(FleetReference):
    """Degenerate: power to the bare forecast and return every machine that is off (no headroom, no spares)."""
    params = {**TUNED, "headroom": 1.0, "min_on": 1, "forecast": "last_hour"}
    retire_headroom = 1.0
    spares = 0


# ============================================================================================ mechanics and actors
def _hours(world, since_seq):
    for e in world.events_range(int(since_seq), world.seq):
        if e.type == "fleet.hour":
            yield e.payload


def _net_hours(world, since_seq):
    return sum(1 for p in _hours(world, since_seq) if p["exact"]["network"]["volume_gbps"] > 0)


def _energy_hours(world, since_seq):
    return sum(1 for p in _hours(world, since_seq) if sum(p["energy_cents"].values()) > 0)


def _capacity_hours(world, since_seq):
    n = 0
    for p in _hours(world, since_seq):
        busy = any(v["util"] >= 0.8 or v["throttle"] < 1.0 for v in p["exact"]["sites"].values())
        short = any(v["delivered"] < v["offered"] - 1e-6 for v in p["exact"]["services"].values())
        n += int(busy or short)
    return n


def _failure_hours(world, since_seq):
    return sum(1 for p in _hours(world, since_seq) if p["failed"])


def _deployments(world, since_seq):
    return sum(1 for e in world.events_range(int(since_seq), world.seq)
               if e.type == "fleet.deployed" and e.payload.get("change"))


MECHANICS = (
    Mechanic("Compute", ("fleet.hour",), "machines on, cores serving interactive and batch work every hour"),
    Mechanic("Networking", ("fleet.cut",), "traffic routed over the network (hours with volume)", count=_net_hours),
    Mechanic("Workloads", ("cols.batch:fleet.use", "fleet.job", "fleet.release"),
             "population users adopting and abandoning services, batch jobs, tenant releases"),
    Mechanic("Capacity", ("fleet.ordered", "fleet.delivered", "fleet.retired"),
             "hours where capacity binds (utilisation, throttling, undelivered requests); fleet resizing",
             count=_capacity_hours),
    Mechanic("Energy", (), "facility energy bought every hour at the site's price", count=_energy_hours),
    Mechanic("Failures", ("fleet.repair", "fleet.trip"), "machine failures, repairs, breaker trips",
             count=_failure_hours),
    Mechanic("Technical optimization", ("fleet.experiment", "fleet.change"),
             "controller experiments on recorded traffic and proposed changes"),
    Mechanic("Delayed effects", ("fleet.ready", "fleet.delivered", "fleet.share", "fleet.dashboard"),
             "boot lag, procurement lead time, tenants reacting to a week of service, delayed dashboards"),
    Mechanic("Deployment of new controllers and software", ("fleet.release",),
             "controllers deployed at the maintenance window; tenants' software releases", count=_deployments),
)

ACTORS = (
    Actor("end_users", ("cols.batch:fleet.use",), "population individuals using the tenants' services"),
    Actor("tenants", ("fleet.release", "fleet.statement"), "tenant organisations releasing software, billed daily"),
    Actor("repair_crew", ("fleet.repair",), "technicians repairing failed machines"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (str(v) if isinstance(v, list) else v) for k, v in CONFIGS[n].items()},
                              DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Engineering infrastructure", spec_part="9.7", required_mechanics=(), configs=_configs(),
    build=build, objective=objective, evaluate=evaluate, reference=FleetReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 100_000, "Site reliability engineer", "an engineer of the incumbent operator"),),
    actors=ACTORS,
    degenerate=(Probe("fire_repair_crew", "set the technicians to zero", FireRepairCrew),
                Probe("starve_capacity", "no headroom, return every idle machine", StarveCapacity)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=21.0, dt=6 * 3600,
    reference_seeds=(0, 1), reference_configs=("single_site_tou",), sources=("infra_world.py",), doc=__doc__)

base.register(DOMAIN)
