"""Counterfactual invention experiments in worlds where invention is live (Gates J and L, SP-6.3b, SP-6.4b;
EVALUATOR SIDE; wave 3b, workstream G).

Worlds: the integrated civilisation with the population's developers (`inventworld.civ_spec`, kind "civ") or, for
fast unit tests, the ECON creation world with developers (`inventworld.creation_spec`, kind "creation"). Nobody is
scripted: every product is made by an autonomous developer of the population (`innovation.devs`) from public
requests, and every user is an individual of the population doing its own work (`economy.work`).

Arms of one spec and seed differ only in the developers' module config (an evaluator-side intervention; every
difference between arms is caused by it):

* `factual`       - nothing changed;
* `garbage`       - every developer ships an input-ignoring program with the same interface (measured behaviour
                    changed, everything else equal; J.5);
* `ban_novel`     - nobody builds for the procedurally generated (undocumented) task categories (removal of those
                    inventions; J.7, L.5);
* `ban_published` - nobody builds for the documented formats (removal of those inventions; J.7, L.5).

`run_arm` drives an arm day by day and records, every day, the evaluator's needs labelling
(`innovation.relations.label`), the population's work resource use per activity, incumbents' prices and live
subscriptions, the compute the developers' services draw from the providers and the projects that exist.
`outcome` summarises an arm; `experiments.mine_chains` (generic: need -> public request -> artifact -> adoption ->
relief, adoption creating a need, artifact -> competitors' response) and `package_chains` (package -> independent
developer's adoption by execution -> service -> individuals' adoption) find chains; the gate tests confirm them
against the arm where the chain's first change was removed.
"""
from __future__ import annotations

import numpy as np

DAY = 86400
ARMS = ("factual", "garbage", "ban_novel", "ban_published")
LOG_RETAIN = 6_000_000


def published_formats(spec) -> list[str]:
    """Documented formats among the world's work activities (steps of families that name a format)."""
    out = set()
    for m in spec.modules:
        if m["name"] == "work":
            for f in m["config"].get("families") or []:
                if f.get("need"):
                    out.add(f["need"])
                if f.get("then"):
                    out.add(f["then"])
    return sorted(out)


def arm_devs(arm: str, spec) -> dict:
    if arm == "factual":
        return {}
    if arm == "garbage":
        return {"program": "garbage"}
    if arm == "ban_novel":
        return {"ban": ["gen.*"]}
    if arm == "ban_published":
        return {"ban": published_formats(spec)}
    raise KeyError(arm)


def arm_spec(kind: str, arm: str, seed: int, *, profile: str = "balanced", n: int = 600, **kw):
    from . import inventworld as IW
    if kind == "civ":
        base = IW.civ_spec(profile, seed, **kw)
        return IW.civ_spec(profile, seed, devs=arm_devs(arm, base), **kw)
    base = IW.creation_spec(seed, n=n, **kw)
    return IW.creation_spec(seed, n=n, devs=arm_devs(arm, base), **kw)


def family_minutes(world) -> dict[str, dict]:
    """Per work activity (keyed by its steps' formats, `a+b`): working and hand-work minutes, tasks done/lost."""
    from ..substrate import columns
    w = world.modules.get("work")
    if w is None:
        return {}
    fam = columns.col(world, "work", "fam")
    out = {}
    for j, g in enumerate(w.fams):
        rows = np.flatnonzero(fam == j)
        key = f"{g['need']}+{g['then']}" if g["then"] else g["need"]
        out[key] = {k: float(columns.col(world, "work", k)[rows].sum()) for k in ("minutes", "diy_minutes", "done",
                                                                                  "lost", "delegated", "spent")}
        out[key]["individuals"] = int(rows.size)
    return out


def developer_compute(world) -> dict:
    """Compute the developers' services drew from the providers (CU and CPU ms metered on their hosts) and the
    cores their instances hold now."""
    st = world.state
    hosts = {r["host"] for _, r in st.scan("dev_projects")} if "dev_projects" in st.tables else set()
    cu = sum(float(r["cu"]) for k, r in st.scan("svc_days") if r["host"] in hosts)
    cores = 0
    for _, r in st.scan("dev_projects") if "dev_projects" in st.tables else ():
        inst = st.get("instances", r["instance"])
        if inst and inst["status"] == "running":
            cores += int(inst["cores"])
    return {"cu": round(cu, 3), "cores": cores, "hosts": len(hosts)}


def snapshot_day(world) -> dict:
    from .experiments import snapshot_day as base
    h = base(world)
    h["families"] = family_minutes(world)
    h["dev_compute"] = developer_compute(world)
    st = world.state
    h["projects"] = sorted((pid, r["status"]) for pid, r in st.scan("dev_projects")) if "dev_projects" in st.tables else []
    return h


def run_arm(kind: str, arm: str, seed: int, *, days: int = 24, profile: str = "balanced", n: int = 600,
            log=None, **kw) -> dict:
    import time

    import crucible
    from ..core.kernel import World
    crucible.load_all_modules()
    # the measurements read the whole event log (adoptions after accepted trials, income deposits, chains): keep it
    # all (a civilisation retains 300,000 events by default and drops the older half beyond that)
    kw["params"] = {**dict(kw.get("params") or {}), "log_retain": LOG_RETAIN}
    sp = arm_spec(kind, arm, seed, profile=profile, n=n, **kw)
    t0 = time.time()
    w = World.create(sp, world_id=f"inv-{kind}-{arm}-{sp.hash()[:10]}")
    history = [snapshot_day(w)]
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
        if w.log_base:
            raise RuntimeError(f"{kind}/{arm}: the event log was truncated on day {d} ({w.log_base} events dropped); "
                               "the measurements need the whole log")
        history.append(snapshot_day(w))
        if log is not None:
            st = w.state
            log(f"{kind}/{arm} seed {seed} day {d} ({time.time() - t0:.0f}s): projects "
                f"{len(history[-1]['projects'])}, needs {len(history[-1]['needs'])}, "
                f"relied {sum(1 for _, r in st.scan('work_exp') if r['active'])}")
    return {"kind": kind, "arm": arm, "seed": seed, "profile": profile, "world": w, "history": history,
            "wall_s": round(time.time() - t0, 1)}


def product_category(world, product: dict) -> str | None:
    """The task a product serves, from what the population used it for (the step of its users' work records);
    for a package, the task of the services that run it."""
    from . import measure as M
    st = world.state
    provs = M._provider_ids(world, product)
    if product["kind"] == "package":
        repos = {a["repo"] for a in st.find("pkg_adoptions", "package", product["package"])}
        hosts = {d["host"] for _, d in st.scan("pkg_deploys") if d["repo"] in repos}
        provs = {did for did, d in st.scan("cap_decls") if d["host"] in hosts}
    counts: dict[str, int] = {}
    for _, r in st.scan("work_exp"):
        if r["prov"] in provs and r["uses"] > 0 and r["step_need"]:
            counts[r["step_need"]] = counts.get(r["step_need"], 0) + int(r["uses"])
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])


def outcome(res: dict) -> dict:
    from . import experiments as X
    from . import measure as M
    w = res["world"]
    out = X.outcome(res)
    out["families"] = res["history"][-1]["families"]
    out["dev_compute"] = res["history"][-1]["dev_compute"]
    for pid, p in {p["id"]: p for p in M.agent_products(w)}.items():
        if pid in out["products"]:
            out["products"][pid].update(category=product_category(w, p), creator_kind=p["creator_kind"],
                                        kind=p["kind"], published=p["published"], executable=p["executable"],
                                        created_t=p["created_t"])
    return out


def minutes_for(out: dict, need: str) -> float:
    """Hand-work minutes of the activities that have `need` as a step."""
    return float(sum(v["diy_minutes"] for k, v in out["families"].items() if need in k.split("+")))


def removal_arm_for(need: str) -> str:
    return "ban_novel" if need.startswith("gen.") else "ban_published"


def _rate(hist: list[dict], need: str) -> list[tuple[int, float]]:
    """(t, hand-work minutes per day over the preceding day) of the activities that have `need` as a step."""
    out = []
    for a, b in zip(hist, hist[1:]):
        m0 = sum(v["diy_minutes"] for k, v in a["families"].items() if need in k.split("+"))
        m1 = sum(v["diy_minutes"] for k, v in b["families"].items() if need in k.split("+"))
        out.append((b["t"], (m1 - m0) / max(1.0, (b["t"] - a["t"]) / DAY)))
    return out


# ====================================================================== software chains (technical)
def package_chains(res: dict) -> list[dict]:
    """For every package adopted by execution by an independent developer whose service the population then relies
    on: [package released (software)] -> [independent developer's service executes it (software: CI across the
    attested boundary)] -> [service declared (technology)] -> [individuals adopt the service (work)] -> [their
    hand work on that step falls below what it was when the package was released]."""
    from . import experiments as X
    from . import measure as M
    w, hist = res["world"], res["history"]
    st = w.state
    out = []
    for a in M.adoptions_by_execution(w):
        need = None
        p = st.tables["pkg_packages"][a["package"]]
        rel = min((r["t"] for r in w.modules["pkg"].releases(st, a["package"])), default=None)
        stages = [{"name": f"package {a['package']} released", "subsystem": "software", "t": rel,
                   "creator_kind": M.creator_kind(w, p["owner"])},
                  {"name": f"independent developer's repository {a['repo']} executes it", "subsystem": "software",
                   "t": a["first_t"]}]
        hosts = sorted({d["host"] for _, d in st.scan("pkg_deploys") if d["repo"] == a["repo"]})
        decls = sorted(did for did, d in st.scan("cap_decls") if d["host"] in hosts)
        if decls:
            stages.append({"name": f"service {hosts[0]} declared", "subsystem": "innovation",
                           "t": min(st.tables["cap_decls"][d]["t"] for d in decls)})
            t_ad = next((e.t for e in w.log if e.type == "work.rely" and e.payload.get("active")
                         and e.payload["prov"] in set(decls)), None)
            if t_ad is not None:
                stages.append({"name": "individuals adopt the service", "subsystem": "work activities", "t": t_ad})
                need = next((r["step_need"] for _, r in st.scan("work_exp") if r["prov"] in set(decls)
                             and r["step_need"]), None)
                if need is not None:
                    t_fall = hand_work_falls(hist, need, rel or 0, t_ad)
                    if t_fall is not None:
                        stages.append({"name": "hand work on the step falls", "subsystem": "work activities",
                                       "t": t_fall})
        ch = X._chain(f"package:{a['package']}->{a['repo']}", stages)
        ch.update(need=need, released_t=rel, adopted_t=next((s["t"] for s in stages if s["name"] ==
                                                             "individuals adopt the service"), None))
        out.append(ch)
    return out


def hand_work_falls(hist: list[dict], need: str, t_rel: int, t_ad: int) -> int | None:
    """The first day after `t_ad` on which the hand-work minutes per day of the activities with step `need` fall
    below 80% of their mean over the (up to) three days before max(t_rel, t_ad); None if they never do."""
    rate = _rate(hist, need)
    r0 = [r for t, r in rate if t <= max(t_rel, t_ad)][-3:]
    if not r0:
        return None
    ref = sum(r0) / len(r0)
    return next((t for t, r in rate if t > t_ad and r < 0.8 * ref), None)
