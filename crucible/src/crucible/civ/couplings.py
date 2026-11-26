"""Cross-domain couplings of a civilisation (MASTER_SPEC Part 10.2; ONE_WORLD_CONTRACTS section 3).

A coupling is a reusable relation over shared world state between two subsystems, written once as a record
(`Coupling`) and executed by world mechanisms - never a trigger that fires on a particular event, never a bonus.
Every coupling of a world is a row of the `links` table of kernel module `civlinks`; removing one
(`remove(world, id)`, a recorded system command) stops its mechanism from that moment on, which is how a
counterfactual experiment asks "what happens without this relation" (crucible.civ.counterfactual).

The relation kinds (`KINDS`) are the generic ones of the contract's relation graph:

    informs      knowledge published in one domain is used as an input to decisions in another
    loads        use of one resource draws on the capacity of another (and pays its price)
    constrains   exhausted capacity of one resource limits the operation of another
    exposes      an activity in one domain makes something of another domain known to participants
    supplies     the output of one domain is what participants of another consume
    coordinates  the size of an organisation sets how much of its members' work goes to coordination
    skills       learning activity changes the skills a labour market sees

The catalogue (`CATALOGUE`) covers the eight example categories of Part 10.2. Where a mechanism lives:

* `science_process` (informs; science -> industry): a public paper on a phenomenon that IS the physics of an
  industrial machine model (one fact: the generator writes the model's hidden quality response as the
  phenomenon's hidden law, and `check_invariants` verifies they agree) is read by producer firms' R&D routine
  (`rnd_routine`, an organisational mind routine acting only through tools on public data); it trials the paper's
  predicted optimum on its own machine and keeps it only if its own measured yield improves (D-18).
* `compute_energy` (loads; computing infrastructure -> energy): every hour each datacenter draws energy for its
  powered cores and for the compute its deployments used (`cloud` usage records) from the grid of its region
  (`industry.draw_energy`: the same capacity and utilisation price that production runs see), paid by the provider.
* `grid_curtailment` (constrains; energy -> computing infrastructure -> software businesses): when the grid refuses
  that draw, the provider's datacenter is curtailed for the hour (a `cloud.outage` of the provider, the same state
  cloud's own outages use), so its deployments fail requests. Removed, the datacenter runs on backup generation
  bought outside the world at a premium instead.
* `goods_supply` (supplies; industry -> population/households): the population's `goods` activity buys the supply
  chain's retail stock (crucible.civ.goods).
* `goods_wom` (exposes; social -> product adoption): individuals learn which goods their own contacts last bought.
* `org_coordination` (coordinates; organisations -> employment/production): an organisation's effective staff is
  reduced by coordination overhead that grows with its operator headcount (Ringelmann-type losses), halved for
  organisations that run workflows (`wf`); wraps the `gov` effort hook industry reads, so no production code changes.
* `education_skills` (skills; education -> employment): PENDING - learners of `learn` are not yet members of the
  population and the population's employment is not yet the labour market (Tier 2 education, wave 2b; CORE-POP/ECON
  `pop.employer` -> `labor`). The record is kept so profiles state it; `pending` says what it waits for and the gate
  tests count it as not demonstrated.

Part 10.2 category 2 (technology adoption changes industrial production costs) is the downstream half of
`science_process` measured with a different intervention (reverting an adopted process setting), see
crucible.civ.experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import schema as S
from ..core.errors import CapacityExceeded, CrucibleError, Forbidden, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module

DAY = 86400
HOUR = 3600
KINDS = ("informs", "loads", "constrains", "exposes", "supplies", "coordinates", "skills")


@dataclass(frozen=True)
class Coupling:
    id: str
    category: int                 # Part 10.2 example number (1-8)
    kind: str                     # one of KINDS
    source: str                   # subsystem
    target: str                   # subsystem
    observable: str               # what the source side is (shared state)
    effect: str                   # what changes on the target side
    mechanism: str                # where it runs
    params: dict = field(default_factory=dict)
    pending: str = ""             # non-empty: not executable in this checkout, and why

    def record(self) -> dict:
        return {"id": self.id, "category": self.category, "kind": self.kind, "source": self.source,
                "target": self.target, "observable": self.observable, "effect": self.effect,
                "mechanism": self.mechanism, "params": dict(self.params), "pending": self.pending}


CATALOGUE: dict[str, Coupling] = {c.id: c for c in [
    Coupling("science_process", 1, "informs", "scientific_research", "industry",
             "public papers and replication reports on a phenomenon that is a machine model's process physics",
             "process settings of producers using that machine model (trialled, kept if own yield improves)",
             "crucible.civ.couplings.rnd_routine (society mind routine 'rnd')",
             {"every_days": 3, "min_gain": 0.03, "trial_runs": 3, "grid": 21, "max_rel_err": 0.12}),
    Coupling("compute_energy", 3, "loads", "computing_infrastructure", "energy",
             "powered cores and metered compute units of each provider's deployments (cloud instances, svc_hours)",
             "energy drawn from the datacenter region's grid (capacity, utilisation price) and paid by the provider",
             "civlinks.hour -> industry.draw_energy",
             {"idle_kwh_per_core_hour": 0.1, "kwh_per_1000_cu": 50.0}),
    Coupling("grid_curtailment", 4, "constrains", "energy", "computing_infrastructure",
             "a refused datacenter energy draw (grid capacity used up today)",
             "the provider's datacenter is curtailed for the hour: its deployments fail requests (software "
             "businesses lose sales, users leave)",
             "civlinks.hour -> cloud.outage of the provider", {"backup_price_mult": 3.0}),
    Coupling("goods_supply", 8, "supplies", "industry", "population",
             "retail listings served from industrial stock (availability and price set by producers)",
             "what households of the population can buy, and the demand producers see",
             "crucible.civ.goods (activity 'goods')"),
    Coupling("goods_wom", 5, "exposes", "social_media", "digital_economy",
             "what an individual's own contacts last bought",
             "awareness, then purchases by that individual with its own money",
             "crucible.civ.goods (word of mouth)", {"rate": 0.35}),
    Coupling("org_coordination", 7, "coordinates", "organizations", "industry",
             "operator headcount of an organisation (and whether it runs workflows)",
             "effective operator hours available for production (coordination overhead)",
             "civlinks wraps the `gov` effort hook read by industry.operators",
             {"k": 0.035, "k_with_workflows": 0.0175}),
    Coupling("education_skills", 6, "skills", "education", "employment",
             "mastery that learners reach in `learn` courses",
             "the skills the labour market sees when it matches people to jobs",
             "pending", pending="learners of `learn` are not population members yet (Tier 2 education, wave 2b) and "
                                "the population's employment is not the labour market yet (CORE-POP/ECON: "
                                "pop.employer -> labor)"),
]}


def coupling_ids() -> list[str]:
    return sorted(CATALOGUE)


# --------------------------------------------------------------------------------------------- state queries
def records(world) -> dict[str, dict]:
    """{id: record + enabled} of the couplings of a world."""
    if "civlinks" not in world.modules:
        return {}
    return {k: dict(v) for k, v in world.state.scan("civ_links")}


def active(world, cid: str) -> bool:
    """Is coupling `cid` present and enabled (and executable) in this world?"""
    if "civlinks" not in world.modules:
        return False
    r = world.state.get("civ_links", cid)
    return bool(r and r["enabled"] and not r.get("pending"))


def remove(world, cid: str) -> None:
    """Counterfactual removal: a recorded system command disables coupling `cid` from now on."""
    world.system_call("admin.links_remove", {"coupling": cid})


def restore(world, cid: str) -> None:
    world.system_call("admin.links_restore", {"coupling": cid})


def datacenter_region(world, provider: str) -> str | None:
    r = world.state.get("civ_dc", provider) if "civ_dc" in world.state.tables else None
    return r["region"] if r else None


# --------------------------------------------------------------------------------------------- R&D routine
def _bindings(world) -> dict[str, dict]:
    mod = world.modules.get("civlinks")
    return {b["model"]: b for b in (mod.config.get("bindings") or [])} if mod else {}


def _paper_support(st, pid: str, tol: float) -> tuple:
    """(independent replications whose relative error is within `tol`, minus their mean relative error).
    Graded validity: the institution's chi-square verdict tests the model as an exact law, which an approximate
    but useful model fails; what a producer needs is that the model predicts independent data well."""
    reps = [st.tables["sci_reps"][r] for r in st.find_ids("sci_reps", "paper", pid)]
    errs = [float(r["rel_err"]) for r in reps if r["independent"] and r.get("rel_err") is not None]
    close = [e for e in errs if e <= tol]
    return len(close), -(float(np.mean(errs)) if errs else 1e9)


def _optimum(model_text: str, phen: dict, grid: int):
    """Arg-max of a published model over the phenomenon's public ranges (participant-side computation)."""
    from ..domains.lawlang import LawError, compile_expr
    names = [v["name"] for v in phen["vars"]]
    try:
        f = compile_expr(model_text, names)
    except LawError:
        return None
    axes = [np.linspace(v["lo"], v["hi"], grid) for v in phen["vars"]]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = {n: m.ravel() for n, m in zip(names, mesh)}
    try:
        vals = np.asarray(f(pts), float).reshape(-1)
    except Exception:  # noqa: BLE001 - a model that does not evaluate is not usable
        return None
    if vals.size != mesh[0].size or not np.isfinite(vals).any():
        return None
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    k = int(np.argmax(vals))
    return {n: float(pts[n][k]) for n in names}, float(vals[k]), f


def rnd_routine(soc, ctx, mid, m, rng):
    """Producer R&D: use replicated public science about one's own machine model, keep it only if it works here."""
    w = ctx.world
    if not active(w, "science_process") or "sci" not in w.modules:
        return
    st = ctx.state
    par = CATALOGUE["science_process"].params
    binds = _bindings(w)
    cfg = dict(m["routines"].get("rnd") or {})
    day = ctx.now // DAY
    tried = list(cfg.get("tried") or [])
    trials = dict(cfg.get("trials") or {})
    changed = False
    last_check = int(cfg.get("last_day", -999))
    due = day - last_check >= int(CATALOGUE["science_process"].params["every_days"])
    plans = (m["routines"].get("production") or {}).get("plans") or []
    for plan in plans:
        mach = st.get("machines", plan["machine"])
        d = st.get("designs", plan["design"])
        if mach is None or d is None:
            continue
        b = binds.get(mach["model"])
        if b is None:
            continue
        runs = sorted(((r["t0"], rid, r) for rid, r in st.scan("runs")
                       if r["machine"] == plan["machine"] and r["status"] in ("done", "broke") and r["units"] > 0),
                      key=lambda x: (x[0], x[1]))
        tr = trials.get(plan["design"])
        if tr is not None:
            after = [r for t0, _, r in runs if t0 >= tr["t"]]
            if len(after) < int(par["trial_runs"]):
                continue
            y_new = float(np.mean([r["made"] / r["units"] for r in after[-int(par["trial_runs"]):]]))
            keep = y_new >= tr["base_yield"] * (1.0 - 0.01)
            if not keep:
                soc.act(ctx, m["owner"], "make.tune", {"design": plan["design"], "speed": tr["from"][0],
                                                       "temp": tr["from"][1]})
            soc.remember(ctx, mid, dict(st.tables["minds"][mid]), "rnd_trial", tr["paper"],
                         {"kept": bool(keep), "base_yield": round(tr["base_yield"], 4), "yield": round(y_new, 4)})
            ctx.emit("links.rnd_result", {"org": m["org"], "design": plan["design"], "paper": tr["paper"],
                                          "kept": bool(keep), "base_yield": round(tr["base_yield"], 4),
                                          "yield": round(y_new, 4)}, vis=(m["org"],))
            trials.pop(plan["design"])
            changed = True
            continue
        if not due:
            continue
        changed = True                   # a literature check happened today
        phen = st.get("sci_phen", b["phenomenon"]) if "sci_phen" in st.tables else None
        if phen is None:
            continue
        cands = [(p_id, p) for p_id, p in st.scan("sci_papers")
                 if p["phenomenon"] == b["phenomenon"] and p_id not in tried]
        if not cands:
            continue
        tol = float(par["max_rel_err"])
        cands = [c for c in cands if _paper_support(st, c[0], tol)[0] >= 1]
        if not cands:
            continue                     # nothing independently reproduced yet: wait (papers stay untried)
        cands.sort(key=lambda x: (_paper_support(st, x[0], tol), x[0]), reverse=True)
        pid, paper = cands[0]
        tried.append(pid)
        changed = True
        opt = _optimum(paper["model"], phen, int(par["grid"]))
        if opt is None:
            continue
        point, best, f = opt
        cur = {b["speed_var"]: float(d["speed"]), b["temp_var"]: float(d["temp"])}
        try:
            now_val = float(np.asarray(f({k: np.array([v]) for k, v in cur.items()}), float).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(now_val) or best <= now_val * (1.0 + float(par["min_gain"])):
            continue
        recent = [r for _, _, r in runs][-int(par["trial_runs"]):]
        if not recent:
            continue
        base = float(np.mean([r["made"] / r["units"] for r in recent]))
        s_new = min(1.0, max(0.0, point[b["speed_var"]]))
        t_new = min(1.0, max(0.0, point[b["temp_var"]]))
        out = soc.act(ctx, m["owner"], "make.tune", {"design": plan["design"], "speed": round(s_new, 4),
                                                     "temp": round(t_new, 4)})
        if out is not None:
            trials[plan["design"]] = {"paper": pid, "t": ctx.now, "base_yield": base,
                                      "from": [float(d["speed"]), float(d["temp"])]}
            ctx.emit("links.rnd", {"org": m["org"], "design": plan["design"], "paper": pid,
                                   "from": [float(d["speed"]), float(d["temp"])],
                                   "to": [round(s_new, 4), round(t_new, 4)]}, vis=(m["org"],))
    if changed:
        m2 = dict(st.tables["minds"][mid])
        soc.set(ctx, mid, routines=dict(m2["routines"], rnd={"last_day": day if due else last_check,
                                                             "tried": tried[-60:], "trials": trials}))


# --------------------------------------------------------------------------------------------- coordination
class _CoordinatedGov:
    """The `gov` hook as industry sees it, with coordination overhead (coupling `org_coordination`) applied."""

    def __init__(self, world, inner):
        self.world, self.inner = world, inner

    def __getattr__(self, name):
        if self.inner is None:
            raise AttributeError(name)
        return getattr(self.inner, name)

    def effort(self, st, org: str, n_members: int) -> float:
        base = self.inner.effort(st, org, n_members) if self.inner is not None else 1.0
        return base * coordination_factor(self.world, st, org)


def headcount(world, st, org: str) -> int:
    n = 0
    if "npc_workers" in st.tables:
        n = sum(1 for _, w in st.scan("npc_workers") if w["employer"] == org and w["role"] == "operator")
    return n + int((st.meta.get("own_operators") or {}).get(org, 0))


def coordination_factor(world, st, org: str) -> float:
    if not active(world, "org_coordination"):
        return 1.0
    par = CATALOGUE["org_coordination"].params
    n = headcount(world, st, org)
    if n <= 1:
        return 1.0
    flows = st.find_ids("wf_flows", "owner", org) if "wf_flows" in st.tables else []
    k = float(par["k_with_workflows"] if flows else par["k"])
    return 1.0 / (1.0 + k * (n - 1))


# --------------------------------------------------------------------------------------------- the module
@register_module
class CivLinks(Module):
    """The couplings of a civilisation (see the module docstring). Config: {couplings: [ids], bindings:
    [{phenomenon, model, speed_var, temp_var}], params: {coupling id: {...}}}. Its two tools live on the operator
    host `admin` (removal and restoration for counterfactual experiments); participants have none."""
    name = "civlinks"
    host = "admin"
    version = "1"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("civ_links")
        st.declare_table("civ_dc")
        st.declare_table("civ_rnd", index=("org",))
        for cid in self.config.get("couplings") or []:
            if cid not in CATALOGUE:
                raise ProtocolError(f"civlinks: unknown coupling {cid!r}; known: {coupling_ids()}")
        cs = set(self.config.get("couplings") or [])
        if "science_process" in cs and "society" in world.modules:
            world.hooks.setdefault("mind_routines", {})["rnd"] = rnd_routine
        if "org_coordination" in cs and "industry" in world.modules:
            world.hooks["gov"] = _CoordinatedGov(world, world.hooks.get("gov"))

    def genesis(self, ctx):
        for cid in sorted(self.config.get("couplings") or []):
            rec = CATALOGUE[cid].record()
            rec["params"].update((self.config.get("params") or {}).get(cid, {}))
            ctx.emit("links.coupling", {**rec, "enabled": True}, vis="internal")
        if any(c in (self.config.get("couplings") or []) for c in ("compute_energy", "grid_curtailment")):
            ctx.schedule((ctx.now // HOUR + 1) * HOUR + 120, "civlinks.hour", {}, prio=22, period=HOUR,
                         key="civlinks.hour")
        if "org_coordination" in (self.config.get("couplings") or []):
            ctx.schedule((ctx.now // DAY + 1) * DAY + 300, "civlinks.day", {}, prio=31, period=DAY,
                         key="civlinks.day")

    def register_datacenter(self, ctx, provider: str, region: str) -> None:
        ctx.emit("links.dc", {"provider": provider, "region": region}, vis="public")

    def _params(self, st, cid) -> dict:
        r = st.get("civ_links", cid)
        return dict(r["params"]) if r else dict(CATALOGUE[cid].params)

    # ------------------------------------------------------------------ operator tools (counterfactuals)
    @endpoint("links_remove", S.obj({"coupling": S.s_str(maxLength=40)}, ["coupling"]), cost_s=0)
    def ep_remove(self, ctx, coupling):
        """Operator: disable a coupling from now on (counterfactual experiments)."""
        if ctx.actor != SYSTEM:
            raise Forbidden("operator only")
        if not ctx.state.has("civ_links", coupling):
            raise NotFound(f"no coupling {coupling} in this world")
        ctx.emit("links.removed", {"id": coupling}, vis="internal")
        return {"removed": coupling}

    @endpoint("links_restore", S.obj({"coupling": S.s_str(maxLength=40)}, ["coupling"]), cost_s=0)
    def ep_restore(self, ctx, coupling):
        """Operator: re-enable a removed coupling."""
        if ctx.actor != SYSTEM:
            raise Forbidden("operator only")
        if not ctx.state.has("civ_links", coupling):
            raise NotFound(f"no coupling {coupling} in this world")
        ctx.emit("links.restored", {"id": coupling}, vis="internal")
        return {"restored": coupling}

    @endpoint("links_placebo", S.obj({"n": S.s_int(minimum=1, maximum=50)}, ["n"]), cost_s=0)
    def ep_placebo(self, ctx, n):
        """Operator: record `n` no-op events (the placebo arm of a counterfactual experiment: the same kind of
        footprint in the event log as an intervention, no change to the world)."""
        if ctx.actor != SYSTEM:
            raise Forbidden("operator only")
        for k in range(int(n)):
            ctx.emit("links.placebo", {"k": k}, vis="internal")
        return {"placebo": int(n)}

    @endpoint("civ_grid", S.obj({"region": S.s_str(maxLength=10), "capacity_kwh_day": S.s_int(minimum=0)},
                                ["region", "capacity_kwh_day"]), cost_s=0)
    def ep_grid(self, ctx, region, capacity_kwh_day):
        """Operator intervention (experiments only): set a regional grid's daily capacity (a disruption)."""
        if ctx.actor != SYSTEM:
            raise Forbidden("operator only")
        r = ctx.state.get("regions", region) if "regions" in ctx.state.tables else None
        if not r or not r.get("grid"):
            raise NotFound(f"no grid in region {region}")
        g = r["grid"]
        ctx.world.modules["industry"].add_grid(ctx, region, g["owner"], int(capacity_kwh_day), g["base_price"])
        return {"region": region, "capacity_kwh_day": int(capacity_kwh_day)}

    # ------------------------------------------------------------------ hourly: datacenter energy
    @process("hour")
    def pr_hour(self, ctx):
        w = ctx.world
        st = ctx.state
        if not active(w, "compute_energy") or "cloud" not in w.modules or "industry" not in w.modules:
            return
        ind = w.modules["industry"]
        bank = w.modules["bank"]
        par = self._params(st, "compute_energy")
        backup = float(self._params(st, "grid_curtailment").get("backup_price_mult", 3.0)) \
            if st.has("civ_links", "grid_curtailment") else 3.0
        curtail = active(w, "grid_curtailment")
        hour = ctx.now // HOUR - 1
        cores: dict[str, int] = {}
        for iid in st.find_ids("instances", "status", "running"):
            inst = st.tables["instances"][iid]
            cores[inst["provider"]] = cores.get(inst["provider"], 0) + int(inst["cores"])
        used: dict[str, float] = {}
        loads: dict[str, list] = {}
        for dep_id, dep in sorted(st.scan("deployments")):
            h = st.get("svc_hours", f"{dep['host']}|{hour}")
            if h is None or not h.get("cu"):
                continue
            inst = st.get("instances", dep["instance"])
            if inst is None:
                continue
            used[inst["provider"]] = used.get(inst["provider"], 0.0) + float(h["cu"])
            loads.setdefault(inst["provider"], []).append([dep_id, round(float(h["cu"]), 3)])
        for prov, rec in sorted(st.scan("civ_dc")):
            p = st.get("providers", prov)
            if p is None or p["status"] != "active":
                continue
            kwh = cores.get(prov, 0) * float(par["idle_kwh_per_core_hour"]) + \
                used.get(prov, 0.0) / 1000.0 * float(par["kwh_per_1000_cu"])
            if kwh <= 0:
                continue
            acct = bank.primary_account(st, prov)
            if acct is None:
                continue
            region = rec["region"]
            # what this hour's draw powers (provenance: the deployments whose metered compute it includes)
            info = {"provider": prov, "region": region, "kwh": round(kwh, 3), "hour": int(hour),
                    "powered_cores": int(cores.get(prov, 0)), "loads": loads.get(prov, [])}
            try:
                ctx.run_as(prov, lambda sub: ind.draw_energy(sub, region, kwh, acct, f"datacenter {prov} hour {hour}",
                                                             consumer=prov))
                ctx.emit("links.energy", info, vis=(prov,))
                continue
            except CapacityExceeded:
                pass
            except CrucibleError as e:
                ctx.emit("links.energy_failed", {"provider": prov, "region": region, "error": e.message[:200]},
                         vis=(prov,))
                continue
            if curtail:
                ctx.emit("cloud.outage", {"id": prov, "start": ctx.now, "until": ctx.now + HOUR}, vis="public")
                ctx.emit("links.curtailed", info, vis=(prov,))
            else:
                g = st.tables["regions"][region]["grid"]
                cost = max(1, int(math.ceil(kwh * g["base_price"] * backup)))
                try:
                    bank.move(ctx, acct, "acct_external", cost, "energy", f"backup generation {prov} hour {hour}")
                    ctx.emit("links.backup", dict(info, cost=cost), vis=(prov,))
                except CrucibleError:
                    ctx.emit("cloud.outage", {"id": prov, "start": ctx.now, "until": ctx.now + HOUR}, vis="public")
                    ctx.emit("links.curtailed", dict(info, reason="cannot pay backup"), vis=(prov,))

    # ------------------------------------------------------------------ daily: what coordination costs
    @process("day")
    def pr_day(self, ctx):
        w = ctx.world
        st = ctx.state
        if not active(w, "org_coordination") or "industry" not in w.modules:
            return
        orgs = sorted({f["owner"] for _, f in st.scan("facilities")})
        rows = []
        for org in orgs:
            n = headcount(w, st, org)
            if n < 2:
                continue
            f = coordination_factor(w, st, org)
            rows.append({"org": org, "operators": n, "effective_share": round(f, 4),
                         "hours_lost": round(8.0 * n * (1 - f), 2)})
        for r in rows:
            ctx.emit("links.coordination", r, vis=(r["org"],))

    # ------------------------------------------------------------------ appliers
    @applier("links.coupling")
    def ap_coupling(self, ev):
        p = dict(ev.payload)
        self.world.state.put("civ_links", p.pop("id"), p)

    @applier("links.removed")
    def ap_removed(self, ev):
        self.world.state.patch("civ_links", ev.payload["id"], enabled=False, removed_t=ev.t)

    @applier("links.restored")
    def ap_restored(self, ev):
        self.world.state.patch("civ_links", ev.payload["id"], enabled=True)

    @applier("links.dc")
    def ap_dc(self, ev):
        self.world.state.put("civ_dc", ev.payload["provider"], {"region": ev.payload["region"], "t": ev.t})

    @applier("links.rnd")
    def ap_rnd(self, ev):
        p = ev.payload
        self.world.state.put("civ_rnd", p["design"], {"org": p["org"], "paper": p["paper"], "from": p["from"],
                                                      "to": p["to"], "t": ev.t, "kept": None})

    @applier("links.rnd_result")
    def ap_rnd_result(self, ev):
        p = ev.payload
        if self.world.state.has("civ_rnd", p["design"]):
            self.world.state.patch("civ_rnd", p["design"], kept=bool(p["kept"]), yield_after=p["yield"],
                                   yield_before=p["base_yield"])

    @applier("links.energy")
    def ap_energy(self, ev):
        pass

    @applier("links.placebo")
    def ap_placebo(self, ev):
        pass

    @applier("links.curtailed")
    def ap_curtailed(self, ev):
        pass

    @applier("links.backup")
    def ap_backup(self, ev):
        pass

    @applier("links.energy_failed")
    def ap_energy_failed(self, ev):
        pass

    @applier("links.coordination")
    def ap_coordination(self, ev):
        pass

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        st = world.state
        for cid, r in st.scan("civ_links"):
            if cid not in CATALOGUE:
                errs.append(f"unknown coupling {cid}")
        # one fact: a bound phenomenon's hidden law IS its machine model's process physics
        if "sci" in world.modules and "industry" in world.modules and "phys" in st.meta:
            sci = world.modules["sci"]
            ind = world.modules["industry"]
            g = np.linspace(0.0, 1.0, 5)
            for b in self.config.get("bindings") or []:
                mdl = ind.phys["models"].get(b["model"])
                if mdl is None or b["phenomenon"] not in sci.laws:
                    errs.append(f"binding {b} refers to a missing model or phenomenon")
                    continue
                pts = [{b["speed_var"]: float(s), b["temp_var"]: float(t)} for s in g for t in g]
                law = sci.truth(b["phenomenon"], pts)
                phys = np.array([100.0 * ind.quality(mdl, p[b["speed_var"]], p[b["temp_var"]]) for p in pts])
                if not np.allclose(law, phys, rtol=1e-6, atol=1e-6):
                    errs.append(f"phenomenon {b['phenomenon']} disagrees with the physics of {b['model']}")
        return errs
