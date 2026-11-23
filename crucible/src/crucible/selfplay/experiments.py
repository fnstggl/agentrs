"""Jobs of the self-play experiments (`examples/ow_selfplay.py`).

Every job is a plain function of one JSON-serialisable dict, safe to run in a worker process, that writes what it
trains into a league directory under `job["work"]` and returns a JSON-serialisable result. Training jobs come
first; evaluation jobs load the leagues those wrote. Nothing here tunes anything on a family: one set of learner
options (the defaults of `selfplay.learners`) is used for every family, test split included.

    pair       two individuals trained against each other (optionally with a mid-training branch in which one of
               them is replaced by a scripted profile: the intervention behind "a change in one policy is followed
               by a change in the other's")
    fixed      one individual trained against a frozen scripted profile (or a partner that holds the defaults)
    br         best responses against the frozen final policy of a trained pair (exploitability)
    a1         the 2x2 of experiment A(i)
    coop       the controlled comparison of experiment B
    commons    experiment C: learned pairs between scripted restraint and scripted over-use
    crossplay  the full cross-play matrix of the pairs of one world
    history    archived stages of a lineage against each other and against the final stage
    heldout    final individuals against the held-out opponent population
    audit      classification of pairs by the collusion audit
    population, historical   the two remaining interaction modes
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Any, Sequence

import numpy as np

from ..research.learner import EnvSpec
from . import audit as AU
from . import modes as M
from .arena import Individual, arena_task
from .league import SelfPlayLeague
from .profiles import available, natural_view, profile_individual

SPEC = EnvSpec()
LEAGUE_MODE = {"abm_duopoly": "competition", "abm_duopoly_fx": "competition", "tech_handoff": "cooperation",
               "abm_complement": "cooperation", "abm_commons": "mixed", "tech_shared_pool": "mixed"}
NATIVE_OBJECTIVE = {"tech_handoff": "joint"}        # every other family: own net is the mechanism's own score
NONCOOP = {"abm_duopoly": "compete", "abm_duopoly_fx": "compete", "abm_commons": "grab", "tech_shared_pool": "grab",
           "abm_complement": "default", "tech_handoff": "default"}
HELDOUT_PROFILES = {"abm_duopoly": ("aggressive", "passive", "compete", "collude"),
                    "abm_duopoly_fx": ("aggressive", "passive", "compete", "collude"),
                    "abm_commons": ("restrain", "grab"), "abm_complement": ("partner",),
                    "tech_handoff": ("tuned", "capacity"), "tech_shared_pool": ("restraint", "grab")}


def _threads() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                   # noqa: BLE001
        pass


def pair_name(j: dict) -> str:
    return (f"{j['family']}_{j['task_seed']}_{j['algos'][0]}{j['seeds'][0]}-{j['algos'][1]}{j['seeds'][1]}"
            f"_{j.get('objective', 'own')}")


def league_root(work: str, name: str) -> str:
    return os.path.join(work, "leagues", name)


def _rows(ev: dict) -> list[dict]:
    return ev["rows"]


def _views(task, ev: dict) -> list[dict]:
    return [natural_view(task, s, a) for s, a in enumerate(ev["mean_action"])]


def _slim_curve(curve: Sequence[dict]) -> list[dict]:
    out = []
    for r in curve:
        row = {"it": r["it"], "net": r["net"], "joint": r["joint"], "view": r.get("view")}
        for k in ("eval_net", "eval_joint", "eval_view"):
            if k in r:
                row[k] = r[k]
        out.append(row)
    return out


# ====================================================================== training jobs
def job_pair(j: dict) -> dict:
    _threads()
    t0 = time.time()
    task = arena_task(j["family"], j["task_seed"])
    name = pair_name(j)
    root = league_root(j["work"], name)
    if os.path.isdir(root):
        shutil.rmtree(root)
    lg = SelfPlayLeague(root, task, SPEC, LEAGUE_MODE[j["family"]], j["n_slots"])
    bound = M.seat_bound(task)
    obj = j.get("objective", "own")
    ids = [lg.new(f"{j['algos'][s]}{j['seeds'][s]}.s{s}", j["algos"][s], j["seeds"][s], obj,
                  seat=s if bound else None).id for s in range(2)]
    view = lambda s, a: natural_view(task, s, a)      # noqa: E731
    iters, off = int(j["iters"]), int(j.get("seed_offset", 0))
    K = j.get("shock_at")
    kw = dict(seed_offset=off, eval_every=5, snapshot_every=j.get("snapshot_every", 10), view=view)
    branch = root + f"@branch{K}"
    if K:
        curve = M.train(lg, ids, int(K), **kw)
        lg.save()
        if os.path.isdir(branch):
            shutil.rmtree(branch)
        shutil.copytree(root, branch)
        curve += M.train(lg, ids, iters - int(K), start=int(K), **kw)
    else:
        curve = M.train(lg, ids, iters, **kw)
    lg.save()
    inds = [lg.get(p) for p in ids]
    final = M.evaluate(task, inds, n_slots=j["n_slots"], spec=SPEC)
    base = M.evaluate(task, [None, None], n_slots=j["n_slots"], spec=SPEC)
    out = {"job": j, "name": name, "root": root, "ids": ids, "arch": [i.arch for i in inds], "bound": bound,
           "curve": _slim_curve(curve), "final": M.light(final), "final_rows": _rows(final),
           "final_view": _views(task, final), "defaults": M.light(base), "default_rows": _rows(base),
           "conservation": AU.conservation(final["records"])}
    if K:
        out["shock"] = {}
        for prof in j.get("shocks", []):
            lg2 = SelfPlayLeague.load(branch, SPEC)
            p = profile_individual(task, 0, prof, SPEC, id=f"shock:{prof}")
            if not bound:
                p.seat = None
            lg2.add(p, role="fixture")
            c2 = M.train(lg2, [p.id, ids[1]], min(iters - int(K), int(j.get("shock_len", iters))), start=int(K),
                         seed_offset=off, eval_every=0, snapshot_every=None, view=view)
            out["shock"][prof] = {"at": int(K), "profile_view": natural_view(task, 0, p.learner.action),
                                  "curve": [{"it": r["it"], "view": r["view"], "net": r["net"]} for r in c2]}
        # the same branch continued without any intervention must reproduce the main run exactly (resumability)
        lg3 = SelfPlayLeague.load(branch, SPEC)
        c3 = M.train(lg3, ids, min(3, iters - int(K)), start=int(K), seed_offset=off, eval_every=0,
                     snapshot_every=None, view=view)
        main = [r for r in curve if int(K) <= r["it"] < int(K) + len(c3)]
        out["branch_reproduces_main"] = bool(all(a["net"] == b["net"] and a["action"] == b["action"]
                                                 for a, b in zip(main, c3)))
        shutil.rmtree(branch)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def fixed_name(j: dict) -> str:
    return f"{j['family']}_{j['task_seed']}_{j['algo']}{j['seed']}.s{j['seat']}_vs_{j['profile']}_{j.get('objective', 'own')}"


def job_fixed(j: dict) -> dict:
    """One individual trained in `seat` against a frozen scripted profile in the other seat."""
    _threads()
    t0 = time.time()
    task = arena_task(j["family"], j["task_seed"])
    name = fixed_name(j)
    root = league_root(j["work"], name)
    if os.path.isdir(root):
        shutil.rmtree(root)
    lg = SelfPlayLeague(root, task, SPEC, LEAGUE_MODE[j["family"]], j["n_slots"])
    bound = M.seat_bound(task)
    seat = int(j["seat"])
    me = lg.new(f"{j['algo']}{j['seed']}.vs_{j['profile']}", j["algo"], j["seed"], j.get("objective", "own"),
                seat=seat if bound else None)
    opp = profile_individual(task, 1 - seat, j["profile"], SPEC, id=f"fx:{j['profile']}")
    if not bound:
        opp.seat = None
    lg.add(opp, role="fixture")
    seats = [me.id, opp.id] if seat == 0 else [opp.id, me.id]
    view = lambda s, a: natural_view(task, s, a)      # noqa: E731
    curve = M.train(lg, seats, int(j["iters"]), seed_offset=int(j.get("seed_offset", 0)), eval_every=5,
                    snapshot_every=None, view=view)
    lg.save()
    final = M.evaluate(task, [lg.get(p) for p in seats], n_slots=j["n_slots"], spec=SPEC)
    return {"job": j, "name": name, "root": root, "id": me.id, "seat": seat, "seats": seats,
            "curve": _slim_curve(curve), "final": M.light(final), "final_rows": _rows(final),
            "final_view": _views(task, final), "wall_s": round(time.time() - t0, 1)}


def job_br(j: dict) -> dict:
    """Exploitability of the seat-1 individual of a trained pair: a fresh best response and a warm one (a copy
    of its training partner that keeps learning while the target is frozen)."""
    _threads()
    t0 = time.time()
    lg = SelfPlayLeague.load(j["pair_root"], SPEC)
    task = lg.task
    ids = j["ids"]
    partner, target = lg.get(ids[0]), lg.get(ids[1])
    n = lg.n_slots
    base = M.evaluate(task, [partner, target], n_slots=n, spec=SPEC)
    out = {"job": j, "target": ids[1], "partner": ids[0], "target_arch": target.arch,
           "with_partner": {"partner_net": base["net_per_tick"][0], "target_net": base["net_per_tick"][1],
                            "rows": _rows(base)}, "responses": {}}
    root = league_root(j["work"], "br_" + os.path.basename(j["pair_root"]))
    for algo in j.get("fresh", ["spsa"]):
        r = M.best_response(task, target, 0, f"{root}_fresh_{algo}", algo=algo, seed=j.get("seed", 77),
                            iters=int(j["iters"]), n_slots=n, spec=SPEC)
        r.pop("individual")
        out["responses"][f"fresh_{algo}"] = r
    if j.get("warm_iters"):
        r = M.best_response(task, target, 0, f"{root}_warm", warm=partner, iters=int(j["warm_iters"]), n_slots=n,
                            spec=SPEC)
        r.pop("individual")
        out["responses"]["warm"] = r
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ====================================================================== evaluation jobs
def _load(root: str) -> SelfPlayLeague:
    return SelfPlayLeague.load(root, SPEC)


def _learner_of(lg: SelfPlayLeague) -> Individual:
    return next(i for i in lg.live.values())


def job_a1(j: dict) -> dict:
    """A(i): B trained against the aggressive profile and B trained against the passive one, each evaluated
    against both profiles on the same noise."""
    _threads()
    task = arena_task(j["family"], j["task_seed"])
    n = j["n_slots"]
    seat = int(j["seat"])
    bs = {p: _learner_of(_load(j["roots"][p])) for p in ("aggressive", "passive")}
    out = {"job": j, "net": {}, "opp_net": {}, "rows": {}, "view": {}}
    for trained, b in bs.items():
        for opp in ("aggressive", "passive"):
            a = profile_individual(task, 1 - seat, opp, SPEC)
            a.seat = b.seat = None
            seats = [b, a] if seat == 0 else [a, b]
            ev = M.evaluate(task, seats, n_slots=n, spec=SPEC)
            key = f"B_{trained}|A_{opp}"
            out["net"][key], out["opp_net"][key] = ev["net_per_tick"][seat], ev["net_per_tick"][1 - seat]
            out["rows"][key] = [r["net"][seat] for r in ev["rows"]]
            if opp == trained:
                out["view"][trained] = natural_view(task, seat, ev["mean_action"][seat])
    return out


def job_coop(j: dict) -> dict:
    """Experiment B on one world and one algorithm. Conditions, all on identical evaluation noise:

    pair_native      the pair trained together with the family's native objective
    pair_ablated     the pair trained together with the other objective (cooperative channel removed or added)
    A+default, default+B     each trained individual of the native pair with a partner that holds the defaults
    A+soloB, soloA+B         each with a self-interested learner trained alone against a default partner
    solo+solo        the two self-interested learners together
    default+default  nobody operates anything
    """
    _threads()
    task = arena_task(j["family"], j["task_seed"])
    n = j["n_slots"]
    nat = _load(j["roots"]["native"])
    a, b = nat.get(j["ids"]["native"][0]), nat.get(j["ids"]["native"][1])
    conds: dict[str, list] = {"pair_native": [a, b], "A+default": [a, None], "default+B": [None, b],
                              "default+default": [None, None]}
    if j["roots"].get("ablated"):
        ab = _load(j["roots"]["ablated"])
        conds["pair_ablated"] = [ab.get(j["ids"]["ablated"][0]), ab.get(j["ids"]["ablated"][1])]
    if j["roots"].get("solo0") and j["roots"].get("solo1"):
        s0, s1 = _learner_of(_load(j["roots"]["solo0"])), _learner_of(_load(j["roots"]["solo1"]))
        s0.id, s1.id = "solo.s0", "solo.s1"
        conds.update({"A+soloB": [a, s1], "soloA+B": [s0, b], "solo+solo": [s0, s1]})
    out = {"job": j, "cond": {}, "audit": {}}
    recs = {}
    for name, inds in conds.items():
        ev = M.evaluate(task, inds, n_slots=n, spec=SPEC)
        recs[name] = ev["records"]
        out["cond"][name] = {"net": ev["net_per_tick"], "joint": ev["joint_net_per_tick"], "rows": _rows(ev),
                             "private": ev["private"], "view": _views(task, ev)}
    # the audit: the trained pair against each of its members with a non-cooperating partner
    for ref_name, r0, r1 in (("default_partner", "A+default", "default+B"), ("solo_partner", "A+soloB", "soloA+B")):
        if r0 in recs and r1 in recs:
            ref = AU.matched_reference(recs[r0], recs[r1])
            for pair_name_ in ("pair_native", "pair_ablated"):
                if pair_name_ in recs:
                    c = AU.classify_pair(recs[pair_name_], ref)
                    out["audit"][f"{pair_name_}|vs_{ref_name}"] = _audit_light(c)
    return out


def _audit_light(c: dict) -> dict:
    ev = c["evidence"]
    return {"class": c["class"], "flags": c["flags"], "joint": ev["joint_net_per_tick"], "net": ev["net_per_tick"],
            "value_delivered": ev["value_delivered"], "price_per_unit": ev["price_per_unit"],
            "market_units": ev.get("market_units"), "paired": ev.get("paired"),
            "circular_manipulation": c["circular_manipulation"]["possible"]}


def job_commons(j: dict) -> dict:
    """Experiment C on one trained pair: the pair, both scripted restraint, both scripted over-use, and each
    learned individual against a scripted over-user, on identical noise; plus the audit."""
    _threads()
    lg = _load(j["root"])
    task = lg.task
    n = lg.n_slots
    a, b = lg.get(j["ids"][0]), lg.get(j["ids"][1])
    names = available(task)
    lo, hi = ("grab", "restrain") if "restrain" in names else ("grab", "restraint")
    prof = {nm: [profile_individual(task, s, nm, SPEC) for s in range(2)] for nm in (lo, hi)}
    conds = {"pair": [a, b], "both_restrain": prof[hi], "both_grab": prof[lo], "A+grab": [a, prof[lo][1]],
             "grab+B": [prof[lo][0], b], "restrain+grab": [prof[hi][0], prof[lo][1]],
             "grab+restrain": [prof[lo][0], prof[hi][1]], "default+default": [None, None]}
    for inds in conds.values():
        for i in inds:
            if i is not None and not M.seat_bound(task):
                i.seat = None
    out = {"job": j, "arch": [a.arch, b.arch], "cond": {}}
    recs = {}
    for name, inds in conds.items():
        ev = M.evaluate(task, inds, n_slots=n, spec=SPEC)
        recs[name] = ev["records"]
        out["cond"][name] = {"net": ev["net_per_tick"], "joint": ev["joint_net_per_tick"], "rows": _rows(ev),
                             "private": ev["private"], "view": _views(task, ev)}
    c = out["cond"]
    out["position"] = M.commons_position(c["pair"]["joint"], c["both_restrain"]["joint"], c["both_grab"]["joint"])
    ref = AU.matched_reference(recs["A+grab"], recs["grab+B"])
    out["audit"] = {"pair|vs_grab_partner": _audit_light(AU.classify_pair(recs["pair"], ref)),
                    "both_restrain|vs_both_grab": _audit_light(AU.classify_pair(recs["both_restrain"], recs["both_grab"]))}
    return out


def job_crossplay(j: dict) -> dict:
    _threads()
    task = arena_task(j["family"], j["task_seed"])
    s0, s1 = [], []
    for k, (root, ids) in enumerate(zip(j["roots"], j["ids"])):
        lg = _load(root)
        a, b = lg.get(ids[0]), lg.get(ids[1])
        a.id, b.id = f"p{k}:{a.id}", f"p{k}:{b.id}"
        s0.append(a)
        s1.append(b)
    cp = M.cross_play(task, s0, s1, n_slots=j["n_slots"], spec=SPEC)
    return {"job": j, "matrix": cp}


def job_history(j: dict) -> dict:
    """Archived stages of both lineages of a pair: the round robin of each lineage (seat-free worlds), and every
    stage of each lineage against the final stage of the other."""
    _threads()
    lg = _load(j["root"])
    task = lg.task
    out = {"job": j, "arch": [lg.info[p]["arch"] for p in j["ids"]], "lineage": {}, "vs_final": {}}
    bound = M.seat_bound(task)
    for s, pid in enumerate(j["ids"]):
        if not bound and s == 0:                         # the round robin of one lineage per pair (seat-0's)
            out["lineage"][pid] = M.history_ratings(lg, pid)
        other = j["ids"][1 - s]
        stages = [x for x in lg.archive(of=pid) if lg.entries[x].parent == pid]
        stages.sort(key=lambda x: int(lg.info[x]["stage"]))
        last = [x for x in lg.archive(of=other) if lg.entries[x].parent == other]
        last.sort(key=lambda x: int(lg.info[x]["stage"]))
        rows = []
        for st in stages:
            seats = [st, last[-1]] if s == 0 else [last[-1], st]
            ev = M.evaluate(task, [lg.get(p) for p in seats], n_slots=lg.n_slots, spec=SPEC)
            rows.append({"stage": lg.info[st]["stage"], "net": ev["net_per_tick"][s],
                         "final_opponent_net": ev["net_per_tick"][1 - s], "joint": ev["joint_net_per_tick"]})
        out["vs_final"][pid] = rows
    return out


def job_heldout(j: dict) -> dict:
    """The final individuals of one pair against the held-out opponent population of their world: an idle
    holder, a random operator, a scripted climber (also with a quarter and with four times the starting funds),
    operator-side profiles, earlier checkpoints of another pair that were set aside, and the final individuals of
    a pair trained with a different algorithm. Nobody learns; the evaluation is logged by the league."""
    _threads()
    src = _load(j["root"])
    task = src.task
    bound = M.seat_bound(task)
    root = league_root(j["work"], "heldout_" + os.path.basename(j["root"]))
    if os.path.isdir(root):
        shutil.rmtree(root)
    lg = SelfPlayLeague(root, task, SPEC, LEAGUE_MODE[task.family], src.n_slots)
    finals = []
    for pid in j["ids"]:
        ind = src.get(pid)
        ind.learner.freeze(True)
        lg.add(ind, role="fixture")
        finals.append(pid)
    lg.add_heldout_population(seats=(0, 1) if bound else (None,), profiles=HELDOUT_PROFILES[task.family])
    for k, (oroot, oids, stages) in enumerate(j.get("others", [])):
        olg = _load(oroot)
        for pid in oids:
            for st in stages:                              # earlier checkpoints set aside / other algorithms
                sid = pid if st == "final" else f"{pid}@{st}"
                ind = olg.get(sid)
                ind.learner.freeze(True)
                ind.id = f"ho:{'other_algo' if st == 'final' else 'set_aside'}:{k}:{sid}"
                lg.add(ind, heldout=True, role="fixture", tags=("heldout", "set_aside" if st != "final" else "other_algorithm"))
    out = {"job": j, "arch": [lg.info[p]["arch"] for p in finals], "heldout": lg.heldout_pool(), "eval": {}}
    for pid in finals:
        out["eval"][pid] = lg.evaluate_heldout(pid, M.HELDOUT_SEEDS, both_seatings=False)
    base = M.evaluate(task, [None, None], M.HELDOUT_SEEDS, n_slots=lg.n_slots, spec=SPEC)
    out["default_net"] = base["net_per_tick"]
    # the scripted climber in the same chairs, as the yardstick a learned policy should beat
    out["climber"] = {}
    for s in ((0, 1) if bound else (None,)):
        cid = "ho:climber" + ("" if s is None else f":s{s}")
        rows = {}
        for h in lg.heldout_pool():
            if h == cid or (s is not None and lg.info[h]["seat"] == s) or \
                    not (h.startswith("ho:hold") or lg.info[h]["privileged"]):
                continue                                  # the yardstick meets the idle holder and the profiles
            rec = lg.play([cid, h], M.HELDOUT_SEEDS[0], heldout_eval=True, tag="heldout_yardstick")
            rows[h] = rec.outcomes[cid]["net_per_tick"]
        out["climber"][str(s)] = rows
    out["log"] = lg.heldout_log
    out["training_side_records_with_heldout"] = int(sum(1 for r in lg.records if not r.heldout_eval and
                                                        any(lg.entries[p].heldout for p in r.participants)))
    lg.save()
    return out


def job_audit(j: dict) -> dict:
    """The collusion audit of one trained pair of a market: the pair against each member with a non-cooperating
    scripted partner; and, as calibration of the audit itself, two scripted pairs whose nature is known."""
    _threads()
    lg = _load(j["root"])
    task = lg.task
    n = lg.n_slots
    a, b = lg.get(j["ids"][0]), lg.get(j["ids"][1])
    non = NONCOOP[task.family]

    def prof(name, seat):
        p = profile_individual(task, seat, name, SPEC)
        if not M.seat_bound(task):
            p.seat = None
        return p
    ev = {"pair": M.evaluate(task, [a, b], n_slots=n, spec=SPEC),
          "A+non": M.evaluate(task, [a, prof(non, 1)], n_slots=n, spec=SPEC),
          "non+B": M.evaluate(task, [prof(non, 0), b], n_slots=n, spec=SPEC),
          "non+non": M.evaluate(task, [prof(non, 0), prof(non, 1)], n_slots=n, spec=SPEC)}
    out = {"job": j, "arch": [a.arch, b.arch], "noncooperating_partner": non, "view": _views(task, ev["pair"]),
           "cond": {k: {"net": v["net_per_tick"], "joint": v["joint_net_per_tick"], "private": v["private"]}
                    for k, v in ev.items()}}
    ref = AU.matched_reference(ev["A+non"]["records"], ev["non+B"]["records"])
    out["pair"] = _audit_light(AU.classify_pair(ev["pair"]["records"], ref))
    if task.family.startswith("abm_duopoly"):
        col = M.evaluate(task, [prof("collude", 0), prof("collude", 1)], n_slots=n, spec=SPEC)
        out["calibration_collude_vs_compete"] = _audit_light(AU.classify_pair(col["records"], ev["non+non"]["records"]))
    return out


def job_population(j: dict) -> dict:
    _threads()
    t0 = time.time()
    task = arena_task(j["family"], j["task_seed"])
    root = league_root(j["work"], f"population_{j['family']}_{j['task_seed']}_{j.get('tag', 'a')}")
    if os.path.isdir(root):
        shutil.rmtree(root)
    r = M.run_population(task, root, members=j["members"], iters=int(j["iters"]), n_slots=j["n_slots"],
                         matchmaking=j.get("matchmaking", "uniform"), spec=SPEC, seed=j.get("seed", 0),
                         snapshot_every=None, seed_offset=int(j.get("seed_offset", 0)))
    lg = r.pop("league")
    lg.save()
    out = M.strip(r)
    out.update({"job": j, "root": root, "wall_s": round(time.time() - t0, 1)})
    return out


def job_historical(j: dict) -> dict:
    _threads()
    t0 = time.time()
    task = arena_task(j["family"], j["task_seed"])
    root = league_root(j["work"], f"historical_{j['family']}_{j['task_seed']}_{j['algos'][0]}{j['seeds'][0]}")
    if os.path.isdir(root):
        shutil.rmtree(root)
    r = M.run_historical(task, root, algos=j["algos"], seeds=j["seeds"], iters=int(j["iters"]), n_slots=j["n_slots"],
                         snapshot_every=int(j.get("snapshot_every", 10)), p_archive=float(j.get("p_archive", 0.5)),
                         sampling=j.get("sampling", "pfsp"), spec=SPEC, seed_offset=int(j.get("seed_offset", 0)))
    lg = r.pop("league")
    inds = [lg.get(p) for p in r["individuals"]]
    final = M.evaluate(task, inds, n_slots=j["n_slots"], spec=SPEC)
    lg.save()
    out = M.strip(r)
    out.update({"job": j, "root": root, "ids": r["individuals"], "final": M.light(final), "final_rows": _rows(final),
                "final_view": _views(task, final), "wall_s": round(time.time() - t0, 1)})
    return out


def job_kernel(j: dict) -> dict:
    """A trained pair's frozen match played again in a kernel world, both individuals operating their seats with
    `rig.set` / `rig.read`: the registered metric `rig_net` of each holder against the arena's evaluation of the
    same match (same noise). Also the ledger's conservation and whether the event log replays."""
    _threads()
    import numpy as np
    from ..core.kernel import World
    from .arena import KernelSelfPlayEnv, SelfPlayEnv, with_noise
    lg = _load(j["root"])
    task = with_noise(lg.task, int(j.get("noise", 7)))
    inds = [lg.get(p) for p in j["ids"]]
    nets = {}
    for label, env in (("arena", SelfPlayEnv([task], SPEC)), ("kernel", KernelSelfPlayEnv(task, SPEC))):
        obs = env.reset()
        for ind in inds:
            ind.learner.reset_context(1)
        tot = np.zeros(2)
        while not env.done.all():
            obs, _, _, info = env.step([ind.learner.act(o, explore=False) for ind, o in zip(inds, obs)])
            tot += info["net_cents"][:, 0]
        nets[label] = [float(x) for x in tot]
        if label == "kernel":
            w = env.world
            out = {"rig_net": [env.rig_net(s) for s in range(2)], "tool_calls": [len(c) for c in env.calls],
                   "rig_set_calls": [sum(1 for c in env.calls[s] if c["tool"] == "rig.set") for s in range(2)],
                   "invariants": w.check_invariants(), "ledger": AU.ledger_conservation(w),
                   "payments_between_holders_cents": AU.bilateral_transfers(w, env.actors)["total_cents"],
                   "log_replays": World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()}
    out.update({"job": j, "arch": [i.arch for i in inds], "arena_net_cents": nets["arena"],
                "kernel_net_cents": nets["kernel"], "ticks": int(task.horizon),
                "equal": nets["arena"] == nets["kernel"] == out["rig_net"]})
    return out


def job_registry(j: dict) -> dict:
    """Integration of the `research.learners` registry: (a) two independent `ppo_gru` individuals trained against
    each other in the arena, and a `ppo_gru` individual with an `spsa` partner; (b) the final individuals of two
    trained pairs against the registry's scripted experimenters (`hill_climb`, `probe_exploit`) as further
    held-out opponents. Every registry learner is used through the learner contract alone, with its defaults
    (except the rollout length of `ppo_gru`, set to one batch of matches)."""
    _threads()
    t0 = time.time()
    try:
        from ..research import learners as registry
        names = registry.available()
    except Exception as e:                              # noqa: BLE001
        return {"job": j, "available": False, "why": f"{type(e).__name__}: {e}"}
    task = arena_task(j["family"], j["task_seed"])
    # registry learners cost several milliseconds per tick (recurrent networks, per-slot scripts): training is a
    # short run, and only in the cheap technical world unless the job says otherwise
    n, iters = j["n_slots"], min(int(j["iters"]), 12)
    do_train = bool(j.get("train", j["family"].startswith("tech")))
    bound = M.seat_bound(task)
    obj = NATIVE_OBJECTIVE.get(j["family"], "own")
    out = {"job": j, "available": True, "registry": names, "iters": iters, "trained": {}, "vs_scripted": {}}
    base = M.evaluate(task, [None, None], n_slots=n, spec=SPEC)
    out["defaults"] = M.light(base)
    if "ppo_gru" in names and do_train:
        steps = n * task.horizon
        for label, algos in (("ppo_gru+ppo_gru", ("ppo_gru", "ppo_gru")), ("ppo_gru+spsa", ("ppo_gru", "spsa"))):
            root = league_root(j["work"], f"registry_{j['family']}_{j['task_seed']}_{label}")
            if os.path.isdir(root):
                shutil.rmtree(root)
            lg = SelfPlayLeague(root, task, SPEC, LEAGUE_MODE[j["family"]], n)
            ids = [lg.new(f"{a}{91 + s}.s{s}", a, 91 + s, obj, seat=s if bound else None,
                          **({"rollout_steps": steps} if a == "ppo_gru" else {})).id for s, a in enumerate(algos)]
            curve = M.train(lg, ids, iters, seed_offset=500_000, eval_every=5, snapshot_every=None)
            lg.save()
            back = SelfPlayLeague.load(root, SPEC)       # the registry learner's checkpoint restores
            final = M.evaluate(task, [lg.get(p) for p in ids], n_slots=n, spec=SPEC)
            again = M.evaluate(task, [back.get(p) for p in ids], n_slots=n, spec=SPEC)
            out["trained"][label] = {"ids": ids, "final": M.light(final),
                                     "curve": [{"it": r["it"], "eval_joint": r["eval_joint"], "eval_net": r["eval_net"]}
                                               for r in curve if "eval_joint" in r],
                                     "updates": [int(lg.get(p).learner.usage().updates) for p in ids],
                                     "reloaded_checkpoint_plays_identically": final["net_per_tick"] == again["net_per_tick"]}
    for root, ids in zip(j.get("roots", []), j.get("ids", [])):
        src = _load(root)
        for s, pid in enumerate(ids):
            me = src.get(pid)
            me.learner.freeze(True)
            for name in ("hill_climb", "probe_exploit"):
                if name not in names:
                    continue
                opp = Individual(f"reg:{name}", registry.make(name, SPEC, seed=5), "own", seat=(1 - s) if bound else None)
                seats = [me, opp] if s == 0 else [opp, me]
                was = me.seat
                me.seat = s if bound else None
                ev = M.evaluate(task, seats, M.HELDOUT_SEEDS, n_slots=n, spec=SPEC)
                me.seat = was
                out["vs_scripted"][f"{pid}|{name}"] = {"arch": me.arch, "net": ev["net_per_tick"][s],
                                                       "opponent_net": ev["net_per_tick"][1 - s],
                                                       "joint": ev["joint_net_per_tick"]}
    out["wall_s"] = round(time.time() - t0, 1)
    return out


JOBS = {"registry": job_registry, "kernel": job_kernel, "pair": job_pair, "fixed": job_fixed, "br": job_br, "a1": job_a1, "coop": job_coop, "commons": job_commons,
        "crossplay": job_crossplay, "history": job_history, "heldout": job_heldout, "audit": job_audit,
        "population": job_population, "historical": job_historical}


def run_job(j: dict) -> dict:
    """Run one job and return `{"id", "ok", "result" | "error"}` (errors are reported, never swallowed)."""
    import traceback
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        return {"id": j["id"], "ok": True, "result": JOBS[j["kind"]](j)}
    except Exception as e:                              # noqa: BLE001
        return {"id": j["id"], "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
