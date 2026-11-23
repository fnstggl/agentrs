"""Crucible 4.0 self-play experiments: independent learners competing and cooperating on hidden mechanisms.

    cd crucible
    CRUCIBLE_CACHE=<dir> nohup python examples/ow_selfplay.py --work <scratch dir> --workers 3 > log 2>&1 &
    python examples/ow_selfplay.py --work <scratch dir> --report          # aggregate what has finished so far

Experiments (results in docs/experiments/ow_selfplay_{A,B,C,D,summary}.json):

  A  autonomous competition (abm_duopoly): (i) a learner trained against a frozen aggressive firm and the same
     learner trained against a frozen passive firm, each evaluated against both (2x2); (ii) joint training, the
     lagged relation between the two policies and a mid-training intervention on one of them; (iii) best
     responses against the frozen final policies (exploitability).
  B  cooperation (tech_handoff, abm_complement): the trained pair against each member with a partner that holds
     the defaults, with a self-interested partner trained alone, and against a pair trained with the cooperative
     channel removed (or added), all on identical worlds and noise.
  C  mixed motive (abm_commons, tech_shared_pool): where independent learners settle between scripted joint
     restraint and scripted mutual over-use.
  D  population, history and cross-play: four independently trained pairs per world (two algorithms), the full
     cross-play matrix, archived stages against each other and against the final stage, the final individuals
     against a held-out opponent population, and the same on the test-split families (abm_duopoly_fx,
     tech_shared_pool) with every learner option left at the value used everywhere else.

The run is resumable: every job writes `<work>/results/<id>.json` when it is done and is skipped next time.
Every statistic is over runs (a run = one world and one set of learner seeds), with `research.stats` intervals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np  # noqa: E402

PAIRS = [("spsa", "spsa", (11, 12)), ("spsa", "spsa", (21, 22)), ("pg", "pg", (31, 32)), ("pg", "pg", (41, 42))]
ABLATED_PAIRS = (0, 2)            # one pair per algorithm is also trained with the other objective
NATIVE = {"tech_handoff": "joint"}

FULL = {"n_slots": 8, "iters": {"abm": 40, "tech": 50}, "shock_at": 20, "shock_len": 15,
        "tasks": {"abm_duopoly": [0, 1, 2], "tech_handoff": [0, 1, 2], "tech_shared_pool": [0, 1, 2],
                  "abm_duopoly_fx": [0, 1], "abm_commons": [0, 1], "abm_complement": [0, 1]},
        "a1_tasks": [0, 1, 2], "br_tasks": [0, 1], "br_iters": (30, 15), "pop_tasks": [0],
        "hist": [(0, "spsa"), (0, "pg")], "history_pairs": (0, 2), "history_tasks": 2, "heldout_pairs": (0, 2),
        "heldout_tasks": 2, "registry": ("tech_handoff", "abm_duopoly"),
        "heldout_families": ["abm_duopoly", "tech_handoff", "abm_duopoly_fx", "tech_shared_pool"]}
SMOKE = {"n_slots": 4, "iters": {"abm": 4, "tech": 4}, "shock_at": 2, "shock_len": 2,
         "tasks": {"abm_duopoly": [0], "abm_duopoly_fx": [0], "tech_handoff": [0], "abm_complement": [0],
                   "abm_commons": [0], "tech_shared_pool": [0]},
         "a1_tasks": [0], "br_tasks": [0], "pop_tasks": [0], "hist": [(0, "spsa")], "history_pairs": (0,),
         "registry": ("tech_handoff",),
         "heldout_pairs": (0, 2), "heldout_families": ["abm_duopoly", "tech_handoff", "abm_duopoly_fx", "tech_shared_pool"]}


def _iters(cfg, family):
    return cfg["iters"]["abm" if family.startswith("abm") else "tech"]


def _other(family):
    return "own" if NATIVE.get(family, "own") == "joint" else "joint"


# ====================================================================== the plan
def plan_training(cfg: dict, work: str) -> list[dict]:
    from crucible.selfplay import experiments as X
    jobs = []
    n = cfg["n_slots"]

    def pair(family, ts, k, objective, **extra):
        a, b, seeds = PAIRS[k]
        j = {"kind": "pair", "work": work, "family": family, "task_seed": ts, "algos": [a, b], "seeds": list(seeds),
             "objective": objective, "iters": _iters(cfg, family), "n_slots": n, "seed_offset": 10_000 * (k + 1),
             "pair_index": k, **extra}
        j["id"] = "pair_" + X.pair_name(j)
        return j

    def fixed(family, ts, algo, seed, seat, profile, tag):
        j = {"kind": "fixed", "work": work, "family": family, "task_seed": ts, "algo": algo, "seed": seed,
             "seat": seat, "profile": profile, "objective": "own", "iters": _iters(cfg, family), "n_slots": n,
             "seed_offset": 200_000 + 1000 * seed, "tag": tag}
        j["id"] = "fixed_" + X.fixed_name(j)
        return j

    # A: competition (with the mid-training intervention), then A(i)
    for ts in cfg["tasks"]["abm_duopoly"]:
        for k in range(len(PAIRS)):
            jobs.append(pair("abm_duopoly", ts, k, "own", shock_at=cfg["shock_at"], shock_len=cfg["shock_len"],
                             shocks=["aggressive", "passive"]))
    for ts in cfg["a1_tasks"]:
        for algo, seed in (("spsa", 51), ("pg", 52)):
            for prof in ("aggressive", "passive"):
                jobs.append(fixed("abm_duopoly", ts, algo, seed, 1, prof, "a1"))
    # B (cooperation), C (mixed motive) and the test-split market of D, cheapest worlds first
    for family in cfg["tasks"]:
        if family == "abm_duopoly":
            continue
        native = NATIVE.get(family, "own")
        for ts in cfg["tasks"][family]:
            for k in range(len(PAIRS)):
                jobs.append(pair(family, ts, k, native))
            if family in ("tech_handoff", "abm_complement"):
                for k in ABLATED_PAIRS:
                    jobs.append(pair(family, ts, k, _other(family)))
                for algo, seed in (("spsa", 61), ("pg", 62)):
                    for seat in (0, 1):
                        jobs.append(fixed(family, ts, algo, seed, seat, "default", "solo"))
    for ts in cfg["pop_tasks"]:
        members = [{"id": "spsa.v1", "algo": "spsa", "seed": 71}, {"id": "spsa.v2", "algo": "spsa", "seed": 72, "c": 0.2, "a": 0.1},
                   {"id": "pg.v1", "algo": "pg", "seed": 73}, {"id": "pg.v2", "algo": "pg", "seed": 74, "hold": 12, "sigma0": 0.3}]
        jobs.append({"kind": "population", "id": f"population_abm_duopoly_{ts}", "work": work, "family": "abm_duopoly",
                     "task_seed": ts, "members": members, "iters": _iters(cfg, "abm_duopoly"), "n_slots": n,
                     "seed_offset": 300_000})
    for ts, algo in cfg["hist"]:
        seeds = (81, 82) if algo == "spsa" else (83, 84)
        jobs.append({"kind": "historical", "id": f"historical_abm_duopoly_{ts}_{algo}", "work": work,
                     "family": "abm_duopoly", "task_seed": ts, "algos": [algo, algo], "seeds": list(seeds),
                     "iters": _iters(cfg, "abm_duopoly"), "n_slots": n, "snapshot_every": max(1, _iters(cfg, "abm_duopoly") // 4),
                     "seed_offset": 400_000})
    return jobs


def plan_evaluation(cfg: dict, work: str, res: dict) -> list[dict]:
    """Evaluation jobs for whatever training has finished (`res`: job id -> result)."""
    from crucible.selfplay import experiments as X
    jobs = []
    n = cfg["n_slots"]

    def pr(family, ts, k, objective):
        a, b, seeds = PAIRS[k]
        return res.get("pair_" + X.pair_name({"family": family, "task_seed": ts, "algos": [a, b], "seeds": seeds,
                                              "objective": objective}))

    def fx(family, ts, algo, seed, seat, profile):
        return res.get("fixed_" + X.fixed_name({"family": family, "task_seed": ts, "algo": algo, "seed": seed,
                                                "seat": seat, "profile": profile}))

    for ts in cfg["a1_tasks"]:
        for algo, seed in (("spsa", 51), ("pg", 52)):
            ra, rp = fx("abm_duopoly", ts, algo, seed, 1, "aggressive"), fx("abm_duopoly", ts, algo, seed, 1, "passive")
            if ra and rp:
                jobs.append({"kind": "a1", "id": f"a1_abm_duopoly_{ts}_{algo}", "family": "abm_duopoly", "task_seed": ts,
                             "algo": algo, "seat": 1, "n_slots": n, "roots": {"aggressive": ra["root"], "passive": rp["root"]}})
    for ts in cfg["br_tasks"]:
        for k in range(len(PAIRS)):
            r = pr("abm_duopoly", ts, k, "own")
            if r:
                fresh_it, warm_it = cfg.get("br_iters", (_iters(cfg, "abm_duopoly"), max(2, _iters(cfg, "abm_duopoly") // 2)))
                jobs.append({"kind": "br", "id": f"br_{r['name']}", "work": work, "pair_root": r["root"], "ids": r["ids"],
                             "iters": fresh_it, "warm_iters": warm_it,
                             "fresh": ["spsa"], "pair_index": k, "task_seed": ts})
    for family in ("tech_handoff", "abm_complement"):
        for ts in cfg["tasks"][family]:
            for k in range(len(PAIRS)):
                nat = pr(family, ts, k, NATIVE.get(family, "own"))
                if not nat:
                    continue
                algo = PAIRS[k][0]
                roots, ids = {"native": nat["root"]}, {"native": nat["ids"]}
                ab = pr(family, ts, k, _other(family)) if k in ABLATED_PAIRS else None
                if ab:
                    roots["ablated"], ids["ablated"] = ab["root"], ab["ids"]
                seed = 61 if algo == "spsa" else 62
                s0, s1 = fx(family, ts, algo, seed, 0, "default"), fx(family, ts, algo, seed, 1, "default")
                if s0 and s1:
                    roots["solo0"], roots["solo1"] = s0["root"], s1["root"]
                jobs.append({"kind": "coop", "id": f"coop_{nat['name']}", "family": family, "task_seed": ts,
                             "algo": algo, "pair_index": k, "n_slots": n, "roots": roots, "ids": ids})
    for family in ("abm_commons", "tech_shared_pool"):
        for ts in cfg["tasks"][family]:
            for k in range(len(PAIRS)):
                r = pr(family, ts, k, "own")
                if r:
                    jobs.append({"kind": "commons", "id": f"commons_{r['name']}", "family": family, "task_seed": ts,
                                 "pair_index": k, "root": r["root"], "ids": r["ids"]})
    for family in cfg["tasks"]:
        obj = NATIVE.get(family, "own")
        for ts in cfg["tasks"][family]:
            rs = [pr(family, ts, k, obj) for k in range(len(PAIRS))]
            if all(rs):
                jobs.append({"kind": "crossplay", "id": f"crossplay_{family}_{ts}", "family": family, "task_seed": ts,
                             "n_slots": n, "roots": [r["root"] for r in rs], "ids": [r["ids"] for r in rs],
                             "algos": [p[0] for p in PAIRS]})
            first = cfg["tasks"][family].index(ts)
            if family in cfg["heldout_families"] and first < cfg.get("heldout_tasks", 99):
                for k in cfg["heldout_pairs"]:
                    # the opponents set aside come from a pair that was trained with the other algorithm and is
                    # not itself evaluated here: an early checkpoint of it, and its final individuals
                    r, o = rs[k], rs[3 if PAIRS[k][0] == "spsa" else 1]
                    if r and o:
                        stage = 10 if _iters(cfg, family) >= 20 else 0
                        jobs.append({"kind": "heldout", "id": f"heldout_{r['name']}", "work": work, "family": family,
                                     "task_seed": ts, "pair_index": k, "root": r["root"], "ids": r["ids"],
                                     "others": [[o["root"], o["ids"], [stage, "final"]]]})
            if ts == cfg["tasks"][family][0]:
                for k in (0, 2):
                    if rs[k]:
                        jobs.append({"kind": "kernel", "id": f"kernel_{rs[k]['name']}", "family": family,
                                     "task_seed": ts, "pair_index": k, "root": rs[k]["root"], "ids": rs[k]["ids"]})
            if family in cfg.get("registry", ()) and ts == cfg["tasks"][family][0] and rs[0] and rs[2]:
                # registry learners are slow per tick (recurrent networks, per-slot scripts): a short training run
                # in the cheap world only, and the scripted experimenters as opponents in both
                jobs.append({"kind": "registry", "id": f"registry_{family}_{ts}", "work": work, "family": family,
                             "task_seed": ts, "n_slots": n, "iters": min(12, _iters(cfg, family)),
                             "train": family.startswith("tech"),
                             "roots": [rs[0]["root"], rs[2]["root"]], "ids": [rs[0]["ids"], rs[2]["ids"]]})
            if family in ("abm_duopoly", "abm_duopoly_fx"):
                for k in range(len(PAIRS)):
                    if rs[k]:
                        jobs.append({"kind": "audit", "id": f"audit_{rs[k]['name']}", "family": family, "task_seed": ts,
                                     "pair_index": k, "root": rs[k]["root"], "ids": rs[k]["ids"]})
                for k in cfg["history_pairs"]:
                    if rs[k] and first < (cfg.get("history_tasks", 99) if family == "abm_duopoly" else 1):
                        jobs.append({"kind": "history", "id": f"history_{rs[k]['name']}", "family": family,
                                     "task_seed": ts, "pair_index": k, "root": rs[k]["root"], "ids": rs[k]["ids"]})
    return jobs


# ====================================================================== running
def _path(work, jid):
    return os.path.join(work, "results", jid + ".json")


def load_results(work: str) -> dict:
    out = {}
    d = os.path.join(work, "results")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                with open(os.path.join(d, f)) as fh:
                    out[f[:-5]] = json.load(fh)
    return out


def _claimed(j: dict) -> dict:
    """Run a job unless another driver instance has claimed it (several instances may share one work directory:
    a job belongs to whoever creates `<work>/claims/<id>` first)."""
    from crucible.selfplay.experiments import run_job
    claim = os.path.join(j["work_dir"], "claims", j["id"])
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return {"id": j["id"], "ok": None}
    return run_job(j)


def run(jobs: list[dict], work: str, workers: int) -> None:
    for sub in ("results", "claims"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    todo = [dict(j, work_dir=work) for j in jobs if not os.path.exists(_path(work, j["id"]))]
    print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run",
          flush=True)
    if not todo:
        return
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, workers), maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(_claimed, todo, chunksize=1)):
            if r["ok"] is None:
                continue
            if r["ok"]:
                tmp = _path(work, r["id"]) + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(r["result"], fh, default=_json)
                os.replace(tmp, _path(work, r["id"]))
            else:
                with open(os.path.join(work, "errors.log"), "a") as fh:
                    fh.write(f"{r['id']}: {r['error']}\n{r['trace']}\n")
                open(_path(work, r["id"]) + ".failed", "w").close()
            print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(todo)} {r['id']} "
                  f"{'ok' if r['ok'] else 'FAILED ' + r['error']} ({time.time() - t0:.0f}s)", flush=True)
    # jobs claimed by another instance: wait until they have finished (or failed) before the next stage is planned
    while True:
        pending = [j["id"] for j in todo if not os.path.exists(_path(work, j["id"]))
                   and not os.path.exists(_path(work, j["id"]) + ".failed")]
        if not pending:
            break
        print(f"[{time.strftime('%H:%M:%S')}] waiting for {len(pending)} job(s) claimed elsewhere", flush=True)
        time.sleep(20)


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch directory for leagues, checkpoints and job results")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scale", default="full", choices=["full", "smoke"])
    ap.add_argument("--report", action="store_true", help="only aggregate finished jobs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "experiments"))
    ap.add_argument("--only", default="", help="comma-separated job-id prefixes to run (default: all)")
    ap.add_argument("--export", action="store_true", help="also copy small final checkpoints next to the results")
    ap.add_argument("--fresh", action="store_true", help="forget the claims and failures of earlier instances "
                                                         "(use only when no other instance is running)")
    a = ap.parse_args()
    cfg = FULL if a.scale == "full" else SMOKE
    work = os.path.abspath(a.work)
    if a.fresh and os.path.isdir(os.path.join(work, "claims")):
        import shutil
        shutil.rmtree(os.path.join(work, "claims"))
        for f in os.listdir(os.path.join(work, "results")) if os.path.isdir(os.path.join(work, "results")) else []:
            if f.endswith(".failed"):
                os.remove(os.path.join(work, "results", f))
    if not a.report:
        jobs = plan_training(cfg, work)
        if a.only:
            jobs = [j for j in jobs if j["id"].startswith(tuple(a.only.split(",")))]
        run(jobs, work, a.workers)
        ev = plan_evaluation(cfg, work, load_results(work))
        if a.only:
            ev = [j for j in ev if j["id"].startswith(tuple(a.only.split(",")))]
        run(ev, work, a.workers)
    from crucible.selfplay.report import build_report, export_checkpoints
    res = load_results(work)
    files = build_report(res, cfg, os.path.abspath(a.out), work, tag="" if a.scale == "full" else "_smoke")
    print("wrote", files)
    if a.export:
        print("checkpoints:", len(export_checkpoints(res, cfg, os.path.join(os.path.abspath(a.out), "checkpoints", "selfplay"))))


if __name__ == "__main__":
    main()
