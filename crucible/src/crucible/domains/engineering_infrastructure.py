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
import math
import re

import numpy as np

from . import base
from ..goals.metrics import metric as goal_metric
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

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
    """Measured on the built world (a `base.BuiltWorld`): the fleet's built configuration row (`fleet.conf`: topology,
    sites, tariff, thermal behaviour, breakers, power caps, batch work) and its regime row (`fleet.regime`: the
    network's and sites' hazards, the regional demand phases)."""
    from .base import graph_shape
    st = world.state
    conf = st.get("fleet", "conf") or {}
    reg = (st.get("fleet", "regime") or {}).get("params") or {}
    topo = conf.get("topology") or {}
    nodes = [n["id"] if isinstance(n, dict) else str(n) for n in topo.get("nodes", [])]
    edges = [(l["u"], l["v"]) for l in topo.get("links", []) if isinstance(l, dict) and "u" in l and "v" in l]
    sites = conf.get("sites") or []
    hz = reg.get("hazards") or {}
    phases = set()
    for svc in (reg.get("services") or {}).values():
        ph = sorted((svc.get("peak_hour") or {}).values())
        phases.add("spread" if ph and max(ph) - min(ph) > 6 else "aligned")
    return {"network": graph_shape(nodes, edges, directed=False) if nodes else "none",
            "sites": "one" if len(sites) == 1 else "several", "tariff": conf.get("tariff"),
            "batch": bool(conf.get("batch")), "thermal": bool(conf.get("thermal")),
            "power_caps": any(x.get("power_cap_kw") for x in sites), "breakers": bool(conf.get("breakers")),
            "fibre_cuts": float(hz.get("link_cut_rate_per_day", 0.0)) > 0,
            "follow_sun": "spread" in phases, "heat_waves": float(hz.get("heat_wave_rate", 0.0)) > 0,
            "end_users": "pop" in world.modules}


# ============================================================================================ objective and evaluator
def _draw(rng, difficulty, world):
    """The goal generator's draw: the cost reduction of the level over the window."""
    return {"cost_reduction": float(TARGET.get(int(difficulty), TARGET[3])), "window_days": HORIZON_DAYS}


def _text(p):
    return (f"You are an infrastructure engineer of the datacenter operator. Within {p['window_days']:.0f} days, "
            f"reduce the operator's total infrastructure cost (machine leases, energy, repairs, staff and SLA credits) "
            f"by at least {100 * p['cost_reduction']:.0f}% compared with running the fleet as it is, while preserving "
            f"reliability: the share of requests served within their SLO, batch deadlines, the traffic the tenants "
            f"keep here and their users bring, and the fleet's repair backlog must not get worse.")


@goal_metric("infra_cost_reduction", "EVALUATOR-ONLY. The operator's total cost over the full days so far relative "
             "to the same instance run with nobody acting (1 - cost / counterfactual cost).")
def _m_cost(world, principal):
    return float(evaluate(world, principal, {})["details"].get("cost_reduction", 0.0))


@goal_metric("infra_reliability_preserved", "EVALUATOR-ONLY. 1 when attainment, batch deadlines, the tenants' own "
             "recorded share decisions, users' departures and the failure debt are no worse than in the "
             "counterfactual operation, else 0.")
def _m_rel(world, principal):
    d = evaluate(world, principal, {})["details"]
    return 1.0 if d.get("reliability_ok") and d.get("health_ok") else 0.0


GOAL = GoalFamily(
    "cheaper_reliable_fleet", draw=_draw, text=_text,
    success=lambda p: {"all": [{"metric": "infra_cost_reduction", "args": {"principal": "$self"}, "op": ">=",
                                "value": float(p["cost_reduction"])},
                               {"metric": "infra_reliability_preserved", "args": {"principal": "$self"}, "op": ">=",
                                "value": 1}]},
    progress=lambda p: {"metric": "infra_cost_reduction", "args": {"principal": "$self"},
                        "target": float(p["cost_reduction"])},
    window_days=lambda p: float(p["window_days"]),
    doc="cost reduction against the counterfactual operation with reliability as users and tenants experience it")


def objective(world, principal, difficulty):
    """The objective generated for a participant (what `Domain.make_objective` returns)."""
    return DOMAIN.make_objective(world, principal, difficulty)


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
    """Failure debt: machines failed or under repair now, plus failed machines handed back to the lessor unrepaired
    during the window (returning a broken machine repairs nothing: the fleet lost it to a failure)."""
    import numpy as np
    from .infra_world import FAILED, REPAIR
    status = world.state.arrays["fl_status"]
    returned = sum(len(e.payload.get("broken") or ()) for e in world.events_range(0, world.seq)
                   if e.type == "fleet.retired")
    return int(np.sum(np.isin(status, (FAILED, REPAIR)))) + int(returned)


SHARE_SLACK = 0.02      # tenants' kept share of their traffic (mean over services) vs. the incumbent operation's
LEFT_SLACK = (1.25, 20)  # users who abandoned a service because it failed them: at most 1.25x the incumbent's + 20


def customers(world, since_h: int = 0) -> dict:
    """What the users and tenants did about the service (evaluator side, from recorded decisions): the tenants'
    shares of their traffic kept on this operator now (their own deterministic decisions from a week of experienced
    service) and the population users who abandoned a service because its quality fell below their tolerance (the
    `fleet.use` decision records). Total traffic is not compared: the population's own lives diverge between two
    worlds as soon as anything differs (a seat that only reads already moves it by about 1%)."""
    fl = world.modules["fleet"]
    st = world.state
    conf = fl.conf(st)
    shares = [fl.demand_mult(st, svc["id"]) for svc in conf["services"]]
    left = sum(int((e.payload.get("extra") or {}).get("abandoned_quality", 0)) for e in world.events_range(0, world.seq)
               if e.type == "cols.batch" and e.payload.get("kind") == "fleet.use" and e.t >= since_h * 3600)
    # the tenants' own verdicts: every service's tenant reviews it daily and records how much traffic it keeps (as
    # the tenant's lead); a week without any such decision for a service means no customer judged the service
    leads = {t["lead"] for t in fl.regime(st)["tenants"].values()}
    t0 = world.now - 7 * 86400
    reviewed = {e.payload.get("service") for e in world.events_range(0, world.seq)
                if e.type == "fleet.share" and e.t >= t0 and e.actor in leads}
    return {"share": float(sum(shares) / len(shares)) if shares else 1.0, "left_for_quality": int(left),
            "services_reviewed": len(reviewed & {svc["id"] for svc in conf["services"]}),
            "services": len(conf["services"])}


def first_change_hour(world, principal) -> int | None:
    """The hour of the participant's first recorded state-changing command (a tool that is not read-only), or None.
    Before it the participant only looked: whatever differs from the counterfactual there is not its doing. (Every
    recorded command - reading ones too - shifts the main-line position the population's random draws are keyed by,
    so two worlds diverge a little as soon as a seat calls any tool: on a stressed instance a seat that only reads
    lost 0.0036 and 0.020 of a day's attainment against the idle counterfactual in its first two days.)"""
    mine = set(world.principals_of(principal)) | {principal}
    for e in world.events_range(0, world.seq):
        if e.type != "cmd" or e.actor not in mine:
            continue
        try:
            ro = bool(world.resolve(e.payload.get("tool"), e.actor).read_only)
        except Exception:  # noqa: BLE001 - an unknown tool changed nothing
            continue
        if not ro:
            return int(e.t // 3600)
    return None


def evaluate(world, principal, obj):
    from .infra_world import operator_costs, reliability
    day = 86400
    t1 = (world.now // day) * day
    if t1 < day:
        return {"success": False, "score": 0.0, "details": {"note": "no full day yet"}}
    w0 = shadow(world)
    c, c0 = operator_costs(world, 0, t1), operator_costs(w0, 0, t1)
    # reliability over the hours from the participant's first change on (the whole window when it changed nothing)
    h0 = first_change_hour(world, principal) or 0
    r, r0 = reliability(world, h0, t1 // 3600), reliability(w0, h0, t1 // 3600)
    b, b0 = _backlog(world), _backlog(w0)
    cu, cu0 = customers(world, since_h=h0), customers(w0, since_h=h0)
    red = 1.0 - c["total"] / c0["total"] if c0["total"] > 0 else 0.0
    target = float(((obj or {}).get("params") or {}).get("cost_reduction", TARGET[1]))
    # reliability as the users and tenants experience it: requests within the SLO, batch deadlines, and whether they
    # stay (individuals abandon services that fail them and tenants move traffic away: a cost cut that drives
    # customers off is lost business, not a saving)
    kept_ok = cu["share"] >= cu0["share"] - SHARE_SLACK and \
        cu["left_for_quality"] <= LEFT_SLACK[0] * cu0["left_for_quality"] + LEFT_SLACK[1] and \
        cu["services_reviewed"] == cu["services"]
    rel_ok = r["attainment"] >= r0["attainment"] - 0.001 and r["batch_on_time"] >= r0["batch_on_time"] - 0.05 \
        and kept_ok
    health_ok = b <= b0 + 2
    ok = red >= target and rel_ok and health_ok
    score = max(0.0, min(1.0, red / target)) * (1.0 if rel_ok and health_ok else 0.25)
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {"cost": c, "baseline_cost": c0, "cost_reduction": round(red, 4),
                        "attainment": round(r["attainment"], 6), "baseline_attainment": round(r0["attainment"], 6),
                        "batch_on_time": r["batch_on_time"], "baseline_batch_on_time": r0["batch_on_time"],
                        "repair_backlog": b, "baseline_repair_backlog": b0, "tenant_share": round(cu["share"], 4),
                        "baseline_tenant_share": round(cu0["share"], 4), "users_left": cu["left_for_quality"],
                        "baseline_users_left": cu0["left_for_quality"],
                        "services_reviewed_by_tenants": cu["services_reviewed"], "services": cu["services"],
                        "reliability_ok": rel_ok, "reliability_from_hour": h0,
                        "health_ok": health_ok, "days": t1 // day}}


# ============================================================================================ policies
TUNED = {"power": "always_on", "headroom": 2.2, "min_on": 2, "forecast": "day_ago_max", "placement": "nearest",
         "batch": "cheapest", "spread": 0.0}


class FleetReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): right-size the fleet by experiment. Watch three days of
    the operator's own dashboard and take each site's peak work (utilisation x machines on). Candidate controllers
    keep every machine on (no power cycling: cycling wears machines) but cap each site at `k` machines per machine
    of peak work, plus, with several sites, a warm share of the largest other site's work for traffic that moves when
    a site is cut off or trips; batch runs in cheap hours. Replay the recorded hours with each candidate
    (`fleet.experiment`), take the smallest `k` whose replayed attainment stays within a small margin of the deployed
    controller's, deploy it at the maintenance window and return the machines beyond the cap and a reserve (for
    tenants' releases, which change the work per request unannounced, the demand growth the dashboard shows and
    failure spares). Every day re-check: when a site's busiest hour ran hot, raise its cap into the reserve and lease
    more when the reserve runs low. It never fires the repair crew and returns no
    failed machine."""
    burst = 12
    replan_s = 6 * 3600
    params = TUNED
    ks = (1.4, 1.6, 1.8, 2.0, 2.2, 2.45, 2.7, 3.0)  # machines per machine of peak work, tested on recorded traffic
    att_margin = 0.0006                           # replayed attainment allowed below the deployed controller's
    failover = 1.2                                # warm machines per machine of work another site may hand over
    spares = 4
    reserve = 0.1                                 # machines kept off beyond the cap: tenants' releases change CPU per
                                                  # request unannounced (and the dashboard shows demand growing)
    hot = 0.04                                    # a site running this much above 1/k at its busiest hour is raised

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.repo = None
        self.target = None
        self.stage = "status"
        self.sites: list[str] = []
        self.work: dict[str, float] = {}
        self.cands: dict[float, dict] = {}
        self.k = None
        self.caps: dict[str, int] = {}
        self.deployed_at = None
        self.growth = 0.0
        self.lead_days = 5.0
        self.t_start = None
        self.next_check = None
        m = re.search(r"Within (\d+) days", str(self.objective.get("text") or ""))
        self.window_days = float(m.group(1)) if m else HORIZON_DAYS
        self.installed0: dict[str, int] = {}
        self.cuts_seen: set = set()

    def caps_for(self, k: float) -> dict[str, int]:
        out = {}
        # warm failover: a share of the largest other site's work; when fibre cuts were seen, a cut hands a whole
        # region to its next-nearest site rather than spreading it over every other site
        spread = max(1, len(self.sites) - (2 if self.cuts_seen else 1))
        for s in self.sites:
            others = [v for o, v in self.work.items() if o != s]
            reserve = self.failover * max(others) / spread if others else 0.0
            out[s] = int(math.ceil(k * self.work.get(s, 0.0) + reserve))
        return out

    def commit(self, tag, caps, msg):
        return act_("code.commit", tag, repo=self.repo, files={"controller.json": json.dumps({**self.params,
                                                                                             "max_on": caps})},
                    message=msg)

    def plan(self, obs):
        day = 86400
        if self.t_start is None:
            self.t_start = self.t
        if self.stage == "status":
            return [act_("fleet.status", "status")]
        if self.stage == "watch" and self.t >= self.t_start + 3 * day:
            # three days: the replayed 48 hours start with a recorded day before them
            self.stage = "measured"
            return [act_("fleet.measure", "measure", window_h=72)]
        if self.stage == "candidates":
            self.stage = "experiments"
            return [self.commit(f"commit:{k}", self.caps_for(k), f"right-size: {k} machines per machine of peak "
                                f"work") for k in self.ks]
        if self.stage == "experiments" and all("ref" in c for c in self.cands.values()) and \
                len(self.cands) == len(self.ks):
            self.stage = "testing"
            return [act_("fleet.experiment", f"exp:{k}", repo=self.repo, target=self.target, ref=c["ref"], hours=48)
                    for k, c in sorted(self.cands.items())]
        if self.stage == "testing" and all("att" in c for c in self.cands.values()):
            ok = [k for k, c in sorted(self.cands.items()) if c["att"] >= c["att0"] - self.att_margin]
            self.k = min(ok) if ok else max(self.cands)
            self.caps = self.caps_for(self.k)
            self.stage = "wait"
            return [act_("fleet.propose_change", "proposed", repo=self.repo, target=self.target,
                         ref=self.cands[self.k]["ref"], note=f"right-size at {self.k} per machine of peak work")]
        if self.stage == "wait" and self.deployed_at is not None and self.t >= self.deployed_at + 6 * 3600:
            self.stage = "resize"
            return [act_("fleet.status", "status2")]
        if self.stage == "steady" and self.next_check is not None and self.t >= self.next_check:
            self.next_check = self.t + day
            self.stage = "check"
            return [act_("fleet.measure", "measure", window_h=24), act_("fleet.status", "status3")]
        return [act_("fleet.status", "poll")]

    def _stats(self, rows):
        """Per site: peak work (machines' worth: utilisation x machines on) and the hottest hour's utilisation;
        the demand growth per day from daily request totals (log-linear fit over full days)."""
        work: dict[str, float] = {}
        utils: dict[str, list] = {}
        daily: dict[int, float] = {}
        for row in rows:
            for s, v in row["sites"].items():
                work[s] = max(work.get(s, 0.0), float(v["util"]) * float(v["on"]))
                if not v.get("trip"):
                    utils.setdefault(s, []).append(float(v["util"]))
            daily[row["hour"] // 24] = daily.get(row["hour"] // 24, 0.0) + float(sum(row["requests"].values()))
        # a busy hour, not an incident: the third-busiest hour of the window (a cut or a trip elsewhere moves a
        # region's traffic for a few hours)
        hot = {s: sorted(u)[-3] if len(u) >= 3 else max(u) for s, u in utils.items() if u}
        days = sorted(daily)[1:-1]
        g = 0.0
        if len(days) >= 3:
            x = np.array(days, float)
            y = np.log(np.maximum(1.0, np.array([daily[d] for d in days])))
            g = float(np.polyfit(x, y, 1)[0])
        return work, hot, max(0.0, min(0.03, g))

    def on_result(self, tag, result, action):
        if result is None or not tag:
            return
        if isinstance(result, dict) and action.get("tool") == "fleet.status":
            self.cuts_seen |= set(result.get("cut_links") or [])
        if tag == "status":
            if not result.get("you_operate"):
                self.stage = "stuck"
                return
            dep = result["deployed"]
            self.repo, self.target = dep["repo"], dep["target"]
            self.sites = [s["id"] for s in result.get("sites") or []]
            self.installed0 = {s: sum(int(c.get(k, 0)) for k in ("on", "booting", "off", "failed", "repairing"))
                               for s, c in (result.get("machines") or {}).items()}
            self.lead_days = float((result.get("prices") or {}).get("lead_days", 5))
            self.stage = "watch"
        elif tag == "measure":
            work, hot, g = self._stats(result.get("rows", []))
            self.growth = g
            if self.stage == "measured":
                self.work = work
                self.stage = "candidates"
            else:
                self.hot_now = hot
        elif tag.startswith("commit:"):
            self.cands[float(tag[7:])] = {"ref": result["commit"]}
        elif tag.startswith("exp:"):
            c = self.cands[float(tag[4:])]
            c["att"], c["att0"] = float(result["candidate"]["attainment"]), float(result["deployed"]["attainment"])
        elif tag == "proposed":
            self.deployed_at = int(result.get("deploy_at") or self.t)
        elif tag in ("status2", "status3"):
            machines = result.get("machines") or {}
            grow = math.exp(self.growth * min(self.lead_days + 7, max(0.0, self.t_start + self.window_days * 86400
                                                                       - self.t) / 86400))
            raised = False
            for site, c in sorted(machines.items()):
                installed = sum(int(c.get(k, 0)) for k in ("on", "booting", "off", "failed", "repairing"))
                cap = self.caps.get(site, installed)
                keep = int(math.ceil(cap * (1.0 + self.reserve) * grow)) + self.spares
                if tag == "status2":
                    # return what lies beyond the cap and its reserve (machines that are off; never a failed one)
                    if installed > keep and int(c.get("off", 0)) > 0:
                        self.queue.append(act_("fleet.retire", None, site=site, n=int(min(installed - keep,
                                                                                        c["off"]))))
                elif getattr(self, "hot_now", {}).get(site, 0.0) > 1.0 / self.k + self.hot and \
                        self.t >= getattr(self, "raised_at", 0) + 2 * 86400:     # the last raise deployed and measured
                    # the busiest hour ran hot (a release, growth): raise the cap into the reserve now and lease
                    # more when the reserve runs low (lead time)
                    # (never beyond the fleet the operator had: the target is a cheaper operation than that one)
                    top = self.installed0.get(site, installed)
                    cap = min(int(math.ceil(cap * 1.12)), top)
                    raised = raised or cap > self.caps.get(site, 0)
                    self.caps[site] = cap
                    keep = min(top, int(math.ceil(cap * (1.0 + self.reserve) * grow)) + self.spares)
                    if keep > installed:
                        self.queue.append(act_("fleet.order", None, site=site, n=int(min(200, keep - installed))))
            if raised:
                self.raised_at = self.t
                self.queue.append(self.commit("recommit", dict(self.caps), "raise the caps of hot sites"))
            self.stage = "steady"
            if self.next_check is None:
                self.next_check = self.t + 86400
        elif tag == "recommit":
            self.queue.append(act_("fleet.propose_change", None, repo=self.repo, target=self.target,
                                   ref=result["commit"], note="raise the caps of hot sites"))


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


class FireAndReturnBroken(ScriptedReference):
    """Degenerate: fire the repair crew and, every six hours, hand every failed machine back to the lessor (paying
    its refurbishment charge), so failed machines never pile up in the repair backlog - the fleet just shrinks. No
    controller change, no experiment (the verifier's `fire_and_retire`, which passed before returns of broken
    machines were charged)."""
    burst = 8
    replan_s = 6 * 3600

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.fired = False

    def plan(self, obs):
        acts = []
        if not self.fired:
            self.fired = True
            acts.append(act_("fleet.staff", None, technicians=0))
        acts.append(act_("fleet.status", "st"))
        return acts

    def on_result(self, tag, result, action):
        if tag == "st" and result:
            for site, c in sorted((result.get("machines") or {}).items()):
                n = int(c.get("failed", 0))
                if n > 0:
                    self.queue.append(act_("fleet.retire", None, site=site, n=n, include_failed=True))


class StarveCapacity(ScriptedReference):
    """Degenerate: power to the bare forecast (follow demand, no headroom) and return every machine that is off."""
    burst = 6
    replan_s = 6 * 3600

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.stage = "status"
        self.repo = self.target = None
        self.deployed_at = None

    def plan(self, obs):
        if self.stage == "status":
            self.stage = "wait_status"
            return [act_("fleet.status", "status")]
        if self.stage == "commit":
            self.stage = "wait"
            return [act_("code.commit", "commit", repo=self.repo, files={"controller.json": json.dumps(
                {**TUNED, "power": "follow_demand", "headroom": 1.0, "min_on": 1, "forecast": "last_hour"})},
                message="bare forecast")]
        if self.stage == "deployed" and self.t >= self.deployed_at + 6 * 3600:
            self.stage = "done"
            return [act_("fleet.status", "status2")]
        return []

    def on_result(self, tag, result, action):
        if result is None or not tag:
            return
        if tag == "status":
            self.repo, self.target = result["deployed"]["repo"], result["deployed"]["target"]
            self.stage = "commit"
        elif tag == "commit":
            self.queue.append(act_("fleet.propose_change", "proposed", repo=self.repo, target=self.target,
                                   ref=result["commit"]))
        elif tag == "proposed":
            self.deployed_at = int(result.get("deploy_at") or self.t)
            self.stage = "deployed"
        elif tag == "status2":
            for site, c in sorted((result.get("machines") or {}).items()):
                if int(c.get("off", 0)) > 0:
                    self.queue.append(act_("fleet.retire", None, site=site, n=int(c["off"])))


# ============================================================================================ mechanics and actors
def _hour_seqs(world, since_seq, pred):
    return base.events_where(world, since_seq, ("fleet.hour",), lambda e: pred(e.payload))


def _net_hours(world, since_seq):
    """Hours whose traffic was routed over the network (seqs of those fleet.hour events)."""
    return _hour_seqs(world, since_seq, lambda p: p["exact"]["network"]["volume_gbps"] > 0)


def _capacity_hours(world, since_seq):
    """Hours where capacity binds: a busy or throttled site, or requests not delivered."""
    def binds(p):
        busy = any(v["util"] >= 0.8 or v["throttle"] < 1.0 for v in p["exact"]["sites"].values())
        short = any(v["delivered"] < v["offered"] - 1e-6 for v in p["exact"]["services"].values())
        return busy or short
    return _hour_seqs(world, since_seq, binds)


def _failure_hours(world, since_seq):
    """Hours in which machines failed."""
    return _hour_seqs(world, since_seq, lambda p: bool(p["failed"]))


def _deployments(world, since_seq):
    """Controller changes deployed at the maintenance window."""
    return base.events_where(world, since_seq, ("fleet.deployed",), lambda e: bool(e.payload.get("change")))


MECHANICS = (
    Mechanic("Compute", ("fleet.hour", "fleet.ready"), "machines on, cores serving interactive and batch work every "
             "hour; machines coming into service after booting or repair"),
    Mechanic("Networking", ("fleet.cut",), "traffic routed over the network (hours with volume)", count=_net_hours),
    Mechanic("Workloads", ("cols.batch:fleet.use", "fleet.job", "fleet.release"),
             "population users adopting and abandoning services, batch jobs, tenant releases"),
    Mechanic("Capacity", ("fleet.ordered", "fleet.delivered", "fleet.retired"),
             "hours where capacity binds (utilisation, throttling, undelivered requests); fleet resizing",
             count=_capacity_hours),
    Mechanic("Energy", ("bank.txn{kind=energy}",), "facility energy bought from the utility every hour at the site's "
             "price (the operator's payments)"),
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
    Actor("tenants", ("fleet.release", "fleet.share"), "tenant organisations reviewing their service and deciding "
          "how much traffic to keep here; releasing software"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (str(v) if isinstance(v, list) else v) for k, v in CONFIGS[n].items()},
                              DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Engineering infrastructure", spec_part="9.7", required_mechanics=(), configs=_configs(),
    build=build, goal=GOAL, evaluate=evaluate, reference=FleetReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 100_000, "Site reliability engineer", "an engineer of the incumbent operator"),),
    actors=ACTORS,
    degenerate=(Probe("fire_repair_crew", "set the technicians to zero", FireRepairCrew),
                Probe("fire_and_return_broken", "no technicians; return every failed machine to the lessor",
                      FireAndReturnBroken),
                Probe("starve_capacity", "no headroom, return every idle machine", StarveCapacity)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=21.0, dt=6 * 3600,
    reference_seeds=(0, 1), reference_configs=("single_site_tou",), sources=("infra_world.py",), doc=__doc__)

base.register(DOMAIN)
