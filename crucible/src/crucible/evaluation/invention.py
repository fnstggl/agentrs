"""Invention trials: what follows from introducing an artifact into a world, established by matched forks.

An *invention* is a real executable artifact (a deployed service, a published capability, a listing, an
operational change) that some participant introduces. This module does not know what any particular invention is
for. It answers four questions about any of them, from authoritative state only:

1. **Did the world change?** A structural `census` of each fork (losses by mechanism, public problem reports,
   capabilities and who installed them, organisations founded and wound up, providers' prices and capacity,
   listings, served and latent demand per kind of work, service moves between providers) is diffed against a
   matched control fork that started from the same bytes and received the same schedule, minus the inventor.
2. **In what order?** Each difference is placed by its *causal position*, never by its name: the artifact itself
   (order 0); its adoption by independent parties, its imitation (derivatives that declare it as their origin)
   and the direct change in what adopters serve, lose and consume (order 1); conditions that exist only with it
   (order 2: a loss mechanism or a kind of problem report absent from the control, a shared resource driven to
   its limit, demand for another kind of work switched on or off); artifacts and organisations that other
   participants created after such a condition appeared and that do not derive from the artifact (order 3); and
   reactions of the rest of the economy (order 4: prices, capacity investment, firms wound up or saved). The
   evidence (identifiers, actors, times) is kept with every item so the placement can be audited.
3. **Does it depend on the artifact?** Optional further arms: `no_response` (the inventor acts, would-be
   responders stay idle: isolates what the response changed), `removal` (the artifact is withdrawn by a recorded
   operator call partway: do the order-2 conditions fade?), and any placebo the caller supplies.
4. **Is it robust?** `invention_suite` repeats the trial over seeds and reports, for each consequence, in how
   many seeds it appeared with the artifact and how many without.

What this is and is not. Under the deterministic cost model both forks are exact functions of the same snapshot,
so a difference between them is caused by the difference in what the arms' participants did: a *controlled
simulated counterfactual*. It is not evidence about any real economy, and it identifies the effect of the whole
arm (everything the inventor did), not of one action inside it. Participants adapt to each other, so later
differences are downstream of earlier ones; the order labels describe position in time and dependency, and the
`no_response` and `removal` arms are what separates them causally. Every participant's provenance (scripted,
language model, trained policy) is recorded in the result, because a chain produced by scripted controls shows
that the *world* supports the phenomenon, not that a learner can find it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.kernel import World
from .counterfactual import Intervention

DAY = 86400


# ====================================================================== census
def census(world: World, t0: int) -> dict:
    """Structural facts about `world` since time `t0`, keyed so that two forks can be diffed. Reads only tables
    that exist: any world family can be passed."""
    st = world.state
    tb = st.tables
    d0 = t0 // DAY
    out: dict[str, Any] = {"time": world.now, "t0": t0}

    losses: dict[str, dict] = {}
    for _, r in sorted(tb.get("ops_days", {}).items()):
        if r["day"] < d0:
            continue
        for kind, n in r["loss_n"].items():
            x = losses.setdefault(kind, {"events": 0, "cents": 0, "orgs": set(), "first_day": r["day"], "by_day": {}})
            x["events"] += int(n)
            x["cents"] += int(r["losses"].get(kind, 0))
            x["orgs"].add(r["owner"])
            x["first_day"] = min(x["first_day"], r["day"])
            x["by_day"][r["day"]] = x["by_day"].get(r["day"], 0) + int(n)
    out["losses"] = {k: {**v, "orgs": sorted(v["orgs"])} for k, v in sorted(losses.items())}

    reqs: dict[str, dict] = {}
    for rid, r in sorted(tb.get("ops_requests", {}).items()):
        if r["t"] < t0:
            continue
        kinds = sorted(set((r.get("evidence") or {}).get("loss_events", {})) | set(r.get("tags") or ()))
        for k in kinds:
            x = reqs.setdefault(k, {"n": 0, "by": set(), "first_t": r["t"], "ids": []})
            x["n"] += 1
            x["by"].add(r["by"])
            x["first_t"] = min(x["first_t"], r["t"])
            x["ids"].append(rid)
    out["requests"] = {k: {**v, "by": sorted(v["by"])} for k, v in sorted(reqs.items())}

    caps = {}
    for cid, c in sorted(tb.get("capabilities", {}).items()):
        if c["t"] >= t0:
            caps[cid] = {"host": c.get("host"), "provider": c.get("provider"), "t": c["t"],
                         "improves": c.get("improves"), "status": c.get("status")}
    out["capabilities"] = caps
    installs: dict[str, dict] = {}
    for _, i in sorted(tb.get("cap_installs", {}).items()):
        if i["t"] < t0:
            continue
        x = installs.setdefault(i["capability"], {"ever": set(), "active": set(), "removed": set(), "first_t": i["t"]})
        x["ever"].add(i["installer"])
        x["first_t"] = min(x["first_t"], i["t"])
        if i["status"] == "active":
            x["active"].add(i["installer"])
        elif i["status"] == "removed":
            x["removed"].add(i["installer"])
    out["installs"] = {k: {a: (sorted(b) if isinstance(b, set) else b) for a, b in v.items()}
                       for k, v in sorted(installs.items())}
    out["markets"] = {k: {"t": m.get("t"), "label": m.get("label")} for k, m in sorted(tb.get("cap_markets", {}).items())
                      if m.get("t", 0) >= t0}

    orgs_new, orgs_status = {}, {}
    for oid, o in sorted(tb.get("orgs", {}).items()):
        orgs_status[oid] = o["status"]
        if o["created"] >= t0:
            orgs_new[oid] = {"founder": o.get("founder"), "kind": o.get("kind"), "t": o["created"]}
    out["orgs_new"] = orgs_new
    out["orgs_inactive"] = sorted(o for o, s in orgs_status.items() if s != "active")

    out["providers"] = {k: {"price": p["price"], "cores": p["cores"],
                            "used": sum(i["cores"] for i in st.find("instances", "provider", k)
                                        if i["status"] == "running")}
                        for k, p in sorted(tb.get("providers", {}).items())}
    out["hw_orders"] = sorted((o["provider"], int(o.get("cores", 0))) for _, o in tb.get("hw_orders", {}).items())

    listings = {}
    for lid, l in sorted(tb.get("listings", {}).items()):
        if l.get("t", l.get("created", 0)) >= t0:
            listings[lid] = {"seller": l["seller"], "need": l.get("need"), "kind": l["kind"], "host": l.get("host"),
                             "price": l["price"], "status": l["status"]}
    out["listings_new"] = listings

    # where every live service runs, and the moves between providers since t0 (from the log, when it is in memory)
    moves = []
    if world.log_base <= 0 or (world.log and world.log[0].t <= t0):
        last_prov: dict[str, str] = {}
        for e in world.log:
            if e.type != "cloud.deployment":
                continue
            inst = tb.get("instances", {}).get(e.payload["instance"])
            prov = inst["provider"] if inst else None
            h = e.payload["host"]
            if e.t >= t0 and h in last_prov and prov is not None and prov != last_prov[h]:
                moves.append({"t": e.t, "host": h, "owner": e.payload["owner"], "from": last_prov[h], "to": prov,
                              "by": e.actor})
            if prov is not None:
                last_prov[h] = prov
    out["moves"] = moves

    needs: dict[str, dict] = {}
    names = st.meta_get("cons_needs") if "c_need" in st.arrays else None
    if names:
        need_arr, prov = st.arrays["c_need"], st.arrays["c_prov"]
        active = st.arrays.get("c_active")
        host_need = {}
        for lid, o in tb.get("cons_offers", {}).items():
            l = tb["listings"].get(lid)
            if l is not None:
                host_need[l["host"]] = o["need"]
        ok: dict[str, int] = {}
        for k, r in tb.get("svc_days", {}).items():
            h, d = k.rsplit("|", 1)
            if int(d) >= d0 and h in host_need:
                ok[host_need[h]] = ok.get(host_need[h], 0) + int(r.get("ok", 0))
        for i, n in enumerate(names):
            m = need_arr == i
            tot = int(m.sum())
            act = int((m & (active.astype(bool))).sum()) if active is not None else tot
            needs[n] = {"consumers": tot, "active": act, "customers": int((m & (prov >= 0)).sum()),
                        "served_ok": int(ok.get(n, 0))}
    out["needs"] = needs
    return out


# ====================================================================== consequences
@dataclass
class Consequence:
    kind: str                 # what differs (see _diff)
    key: str                  # the thing that differs
    order: int                # causal position, 0..4 (see module docstring); -1 = not placed
    treatment: Any
    control: Any
    first_t: int | None = None
    actors: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "order": self.order, "treatment": self.treatment,
                "control": self.control, "first_day": None if self.first_t is None else round(self.first_t / DAY, 2),
                "actors": self.actors, "evidence": self.evidence}


def _ratio_differs(a: float, b: float, rel: float, floor: float) -> bool:
    return abs(a - b) > max(floor, rel * max(abs(a), abs(b)))


def _derives_from(cid: str, caps: dict, roots: set[str]) -> bool:
    """Does capability `cid` declare (transitively, through `improves`) one of `roots` as its origin?"""
    seen = set()
    while cid is not None and cid not in seen:
        if cid in roots:
            return True
        seen.add(cid)
        cid = (caps.get(cid) or {}).get("improves")
    return False


def consequences(treat: dict, control: dict, inventors: set[str], rel: float = 0.25) -> list[Consequence]:
    """Differences between a treatment census and its matched control, placed by causal position.

    `inventors` are the principals (and organisations) whose artifact is being tested: what they publish is order
    0, what others do is imitation or a response. `rel` is the relative difference below which a count is treated
    as equal."""
    out: list[Consequence] = []
    caps = treat["capabilities"]
    new_caps = {cid: c for cid, c in caps.items() if cid not in control["capabilities"]}
    inv_caps = {cid for cid, c in new_caps.items() if c["provider"] in inventors}
    imitations = {cid for cid in new_caps if cid not in inv_caps and _derives_from(cid, caps, inv_caps)}
    lineage = inv_caps | imitations

    # ---- order 0: the artifact ------------------------------------------------------------------------
    for cid in sorted(inv_caps):
        c = new_caps[cid]
        out.append(Consequence("artifact_published", cid, 0, c["host"], None, c["t"], [c["provider"]]))

    # ---- order 1: adoption, imitation, and the direct change in adopters' operations ------------------
    adopters: set[str] = set()
    for cid in sorted(lineage):
        ins = treat["installs"].get(cid)
        if not ins:
            continue
        indep = [o for o in ins["ever"] if o not in inventors and o != caps[cid]["provider"]]
        if indep:
            adopters |= set(indep)
            base = control["installs"].get(cid, {"ever": [], "active": []})
            out.append(Consequence("artifact_adopted" if cid in inv_caps else "imitation_adopted", cid, 1,
                                   {"tried": indep, "kept": [o for o in ins["active"] if o in indep]},
                                   {"tried": base["ever"], "kept": base["active"]}, ins["first_t"], indep))
    for cid in sorted(imitations):
        c = new_caps[cid]
        out.append(Consequence("artifact_imitated", cid, 1, c["host"], None, c["t"], [c["provider"]],
                               {"improves": c["improves"]}))
    for mid, m in treat["markets"].items():
        if mid not in control["markets"] and mid in lineage:
            out.append(Consequence("market_formed", mid, 1, m, None, m["t"], [], {"around": "the artifact"}))
    for kind, x in treat["losses"].items():
        b = control["losses"].get(kind)
        if b and b["events"] > 0 and _ratio_differs(x["events"], b["events"], rel, 2):
            out.append(Consequence("existing_loss_changed", kind, 1,
                                   {"events": x["events"], "cents": x["cents"]},
                                   {"events": b["events"], "cents": b["cents"]}, None, x["orgs"]))
    for kind, b in control["losses"].items():
        if kind not in treat["losses"] and b["events"] > 2:
            out.append(Consequence("existing_loss_changed", kind, 1, {"events": 0, "cents": 0},
                                   {"events": b["events"], "cents": b["cents"]}, None, b["orgs"]))
    for pid, x in treat["providers"].items():
        b = control["providers"].get(pid)
        if b is not None and x["used"] != b["used"]:
            out.append(Consequence("resource_use_changed", pid, 1, {"used": x["used"], "of": x["cores"]},
                                   {"used": b["used"], "of": b["cores"]}))
    for need, x in treat["needs"].items():
        b = control["needs"].get(need)
        if b is not None and _ratio_differs(x["served_ok"], b["served_ok"], rel, 20):
            out.append(Consequence("work_served_changed", need, 1, {"served_ok": x["served_ok"]},
                                   {"served_ok": b["served_ok"]}))

    # ---- order 2: conditions that exist only with the artifact -----------------------------------------
    conditions: dict[str, Consequence] = {}
    for kind, x in treat["losses"].items():
        b = control["losses"].get(kind)
        if (b is None or b["events"] == 0) and x["events"] > 2:
            c = Consequence("loss_mechanism_new", kind, 2, {"events": x["events"], "cents": x["cents"], "orgs": x["orgs"]},
                            {"events": 0}, x["first_day"] * DAY, x["orgs"],
                            {"affects_non_adopters": sorted(set(x["orgs"]) - adopters - inventors)})
            out.append(c)
            conditions[kind] = c
    for kind, x in treat["requests"].items():
        b = control["requests"].get(kind, {"n": 0, "by": []})
        if x["n"] > b["n"] and b["n"] == 0:
            c = Consequence("problem_reported", kind, 2, {"n": x["n"], "by": x["by"]}, {"n": 0}, x["first_t"],
                            x["by"], {"requests": x["ids"][:5]})
            out.append(c)
            if kind not in conditions:
                conditions[kind] = c
    for pid, x in treat["providers"].items():
        b = control["providers"].get(pid)
        if b is not None and x["cores"] > 0 and x["used"] >= x["cores"] and b["used"] < b["cores"]:
            out.append(Consequence("resource_saturated", pid, 2, {"used": x["used"], "of": x["cores"]},
                                   {"used": b["used"], "of": b["cores"]}))
    for need, x in treat["needs"].items():
        b = control["needs"].get(need)
        if b is not None and _ratio_differs(x["active"], b["active"], rel, 5):
            c = Consequence("demand_changed", need, 2, {"active": x["active"], "of": x["consumers"]},
                            {"active": b["active"], "of": b["consumers"]})
            out.append(c)
            if x["active"] > b["active"]:
                conditions.setdefault(f"demand:{need}", c)

    # ---- order 3: what others built after a condition appeared, not derived from the artifact ----------
    times = [c.first_t for c in conditions.values() if c.first_t is not None]
    t_cond = min(times) if times else (treat["t0"] if conditions else None)
    affected: set[str] = set()
    for cond in conditions.values():
        affected |= set(cond.actors)
    responders: set[str] = set()
    for cid, c in sorted(new_caps.items()):
        if cid in lineage:
            continue
        after = t_cond is not None and c["t"] >= t_cond
        if after:
            responders.add(c["provider"])
        out.append(Consequence("response_published" if after else "other_capability_published", cid,
                               3 if after else 1, c["host"], None, c["t"], [c["provider"]],
                               {"after_condition": after}))
        ins = treat["installs"].get(cid)
        if ins and after:
            out.append(Consequence("response_adopted", cid, 3, {"tried": ins["ever"], "kept": ins["active"]}, None,
                                   ins["first_t"], ins["ever"],
                                   {"adopters_affected_by_condition": sorted(set(ins["ever"]) & affected)}))
    imitator_orgs = {caps[cid]["provider"] for cid in imitations}
    for oid, o in treat["orgs_new"].items():
        if oid in control["orgs_new"] or o["founder"] in inventors or oid in inventors:
            continue
        if oid in imitator_orgs:
            out.append(Consequence("imitator_founded", oid, 1, o, None, o["t"], [str(o["founder"])]))
            continue
        after = t_cond is not None and o["t"] >= t_cond
        out.append(Consequence("organisation_founded", oid, 3 if after else 1, o, None, o["t"], [str(o["founder"])],
                               {"after_condition": after, "published_a_response": oid in responders}))
    for lid, l in treat["listings_new"].items():
        if lid in control["listings_new"] or l["seller"] in inventors:
            continue
        out.append(Consequence("offering_listed", lid, 3 if t_cond is not None else 1, l, None, None, [l["seller"]]))
    for mid, m in treat["markets"].items():
        if mid not in control["markets"] and mid not in lineage:
            out.append(Consequence("market_formed", mid, 3, m, None, m["t"], [], {"around": "a response"}))
    tm = [m for m in treat["moves"] if m["by"] not in inventors]
    cm = control["moves"]
    if len(tm) != len(cm):
        out.append(Consequence("services_moved", "providers", 3, {"n": len(tm), "moves": tm[:8]}, {"n": len(cm)},
                               tm[0]["t"] if tm else None, sorted({m["by"] for m in tm})))

    # ---- order 4: reactions of the rest of the economy -------------------------------------------------
    for pid, x in treat["providers"].items():
        b = control["providers"].get(pid)
        if b is None:
            continue
        if x["price"] != b["price"]:
            out.append(Consequence("provider_price_changed", pid, 4, x["price"], b["price"]))
        if x["cores"] != b["cores"]:
            out.append(Consequence("provider_capacity_changed", pid, 4, x["cores"], b["cores"]))
    if treat["hw_orders"] != control["hw_orders"]:
        out.append(Consequence("hardware_investment", "providers", 4, treat["hw_orders"], control["hw_orders"]))
    gone_t, gone_c = set(treat["orgs_inactive"]), set(control["orgs_inactive"])
    for oid in sorted(gone_t - gone_c):
        out.append(Consequence("organisation_wound_up", oid, 4, "inactive", "active"))
    for oid in sorted(gone_c - gone_t):
        out.append(Consequence("organisation_survived", oid, 4, "active", "inactive"))
    out.sort(key=lambda c: (c.order, c.first_t if c.first_t is not None else 10 ** 18, c.kind, c.key))
    return out


# ====================================================================== trials
@dataclass
class Arm:
    """One future of the world. `agents(world) -> [(actor, policy)]` builds fresh policies for this fork;
    `at_day`/`intervention` optionally applies a recorded operator intervention partway (e.g. withdrawing the
    artifact). `provenance` says what kind of thing each actor's policy is."""
    name: str
    agents: Callable[[World], list]
    intervention: Intervention | Callable[[World], Intervention] | None = None
    at_day: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)


def run_arm(blob: bytes, arm: Arm, horizon_days: float, dt: int = 3600, seed: int = 1, world_id: str = "arm") -> World:
    from ..agents.base import Runner
    w = World.from_blob(blob, world_id=world_id)
    run = Runner(w, arm.agents(w), dt=dt, seed=seed, record=False)
    end = w.now + int(horizon_days * DAY)
    cut = None if arm.at_day is None else w.now + int(arm.at_day * DAY)
    done = False
    while w.now + dt <= end:
        if cut is not None and not done and w.now >= cut:
            iv = arm.intervention(w) if callable(arm.intervention) else arm.intervention
            if iv is not None:
                iv.apply(w)
            done = True
        run.round()
    return w


def invention_trial(world: World, arms: list[Arm], inventors: set[str], horizon_days: float, *, control: str = "control",
                    dt: int = 3600, seed: int = 1, rel: float = 0.25, check_determinism: bool = False) -> dict:
    """Fork `world` once per arm from one snapshot, advance all arms on the same schedule, and report every
    arm's consequences relative to the control arm."""
    t_wall = time.perf_counter()
    blob = world.snapshot_blob()
    t0 = world.now
    exact = world.spec.params.get("cost_model") == "deterministic"
    names = [a.name for a in arms]
    if control not in names:
        raise ValueError(f"no arm named {control!r}")
    finals: dict[str, World] = {}
    cens: dict[str, dict] = {}
    try:
        for a in arms:
            finals[a.name] = run_arm(blob, a, horizon_days, dt, seed, f"{world.world_id}~{a.name}")
            cens[a.name] = census(finals[a.name], t0)
        det = None
        if check_determinism:
            again = run_arm(blob, arms[names.index(control)], horizon_days, dt, seed, f"{world.world_id}~{control}2")
            det = again.digest() == finals[control].digest()
            again.close()
        inventors = set(inventors)          # organisations the inventors founded during the trial are theirs too
        for w in finals.values():
            inventors |= {oid for oid, o in w.state.tables.get("orgs", {}).items() if o.get("founder") in inventors}
        res = {"fork_time": t0, "fork_day": round(t0 / DAY, 2), "horizon_days": horizon_days, "exact": exact,
               "determinism_check": det, "control": control, "inventors": sorted(inventors),
               "provenance": {a.name: a.provenance for a in arms},
               "invariants": {n: w.check_invariants() for n, w in finals.items()},
               "digests": {n: w.digest() for n, w in finals.items()}, "arms": {}}
        for a in arms:
            if a.name == control:
                continue
            cs = consequences(cens[a.name], cens[control], inventors, rel)
            res["arms"][a.name] = {"consequences": [c.to_dict() for c in cs],
                                   "by_order": {str(k): sum(1 for c in cs if c.order == k) for k in range(5)},
                                   "kinds": sorted({f"{c.order}:{c.kind}:{c.key}" for c in cs})}
        for a in arms:                      # an arm that withdrew something: did the new conditions fade?
            ref = "treatment" if "treatment" in cens else None
            if a.at_day is None or a.name == control or ref is None or a.name == ref:
                continue
            cut = int(t0 // DAY + a.at_day)
            window = int(max(1, min(7, (finals[a.name].now // DAY) - cut - 1)))
            new_kinds = [c["key"] for c in res["arms"][ref]["consequences"] if c["kind"] == "loss_mechanism_new"]
            res["arms"][a.name]["withdrawn_at_day"] = cut
            res["arms"][a.name]["fades"] = [fades_after(cens[a.name], cens[ref], k, cut, window) for k in new_kinds]
        res["census"] = {n: _slim(c) for n, c in cens.items()}
        res["wall_s"] = round(time.perf_counter() - t_wall, 2)
        return res
    finally:
        for w in finals.values():
            w.close()


def _slim(c: dict) -> dict:
    """Census without the long per-day series (kept: totals, what exists)."""
    out = dict(c)
    out["losses"] = {k: {a: b for a, b in v.items() if a != "by_day"} for k, v in c["losses"].items()}
    out["requests"] = {k: {a: b for a, b in v.items() if a != "ids"} for k, v in c["requests"].items()}
    return out


def fades_after(removal: dict, treatment: dict, key: str, cut_day: int, window: int = 5) -> dict:
    """Did loss mechanism `key` fade once the artifact was withdrawn at `cut_day`? Compares events per day in the
    `window` days after the cut in the removal arm against the same days in the arm that kept the artifact."""
    def rate(c):
        by = c["losses"].get(key, {}).get("by_day", {})
        days = range(cut_day + 1, cut_day + 1 + window)
        return float(np.mean([by.get(d, 0) for d in days]))
    r, t = rate(removal), rate(treatment)
    return {"key": key, "cut_day": cut_day, "window_days": window, "per_day_after_removal": r, "per_day_kept": t,
            "faded": bool(r < 0.5 * t) if t > 0 else None}


def invention_suite(build: Callable[[int], tuple[World, list[Arm], set[str]]], seeds: list[int], horizon_days: float,
                    **kw) -> dict:
    """Repeat a trial over seeds. `build(seed) -> (world at the fork point, arms, inventors)`.

    Returns per-seed results and, for every consequence signature `order:kind` (the key is dropped: identifiers
    differ between seeds), the number of seeds in which it appeared in each arm."""
    per_seed, table = {}, {}
    for s in seeds:
        world, arms, inventors = build(s)
        try:
            r = invention_trial(world, arms, inventors, horizon_days, **kw)
        finally:
            world.close()
        per_seed[str(s)] = r
        for arm, a in r["arms"].items():
            for sig in sorted({f"{c['order']}:{c['kind']}" for c in a["consequences"]}):
                table.setdefault(arm, {}).setdefault(sig, []).append(s)
    from ..research.stats import proportion_ci
    summary = {arm: {sig: {"seeds": ss, **{k: v for k, v in proportion_ci(len(ss), len(seeds)).items()
                                           if k in ("p", "lo", "hi", "low_n")}}
                     for sig, ss in sorted(sigs.items())} for arm, sigs in table.items()}
    return {"seeds": seeds, "horizon_days": horizon_days, "summary": summary, "per_seed": per_seed}
