"""Crucible 4.0 long runs: liveness relative to recent levels, an invariant monitor attached throughout, an optional
recorded shock, with and without market entry by founders.

Each run advances one world with no focal agent and writes `<out>/<prefix>_<family>_s<seed>_<arm>.json` (weekly
tables, collapses with their proximate causes, flags, entry statistics, shock outcome, cost) and
`..._days.csv` (one row per simulated day). A combined `<out>/<prefix>_<family>_summary.json` puts the arms side
by side. Arms: `base` (the family as it is) and `entry` (the same family with the `entry` module added).

    PYTHONPATH=crucible/src python crucible/examples/long_run4.py --family open_digital --seeds 1,2 --days 90 \
        --out crucible/docs/experiments --arms base,entry --shock-day 45 --jobs 2

`--snapshot-every N` writes a snapshot every N simulated days; `--resume` continues a run from its last snapshot.
`--controls` adds the two scripted controls of the 3.0 long runs, so that those runs can be repeated, e.g.

    ... long_run4.py --family open_tight_compute --seeds 1 --days 120 --controls --prefix ow_longrun_tight \
        --overrides '{"cheap_cores": 40, "module_config": {"civ": {"innovators": 2}, "webgen": {"sites": 3},
                      "adversary": {"injection_pages": 2, "fraud_capability": true, "spam_posts": 2}}}'

Runs that execute founders' services need the sandbox (run outside any tool sandbox).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def one(job: dict) -> dict:
    import crucible
    crucible.load_all_modules()
    from crucible.actors import entry  # noqa: F401  (registers the module)
    from crucible.evaluation.longrun import liveness_table, run_long
    t0 = time.time()
    label = job["label"]

    def progress(row):
        if (row["day"] + 1) % 7 == 0:
            print(f"[{label}] day {row['day'] + 1} served {row['served_total']} firms {row['active_firms']} "
                  f"collapsed {row['collapsed']} wall {round(time.time() - t0)}s", flush=True)
    agents = None
    if job["controls"]:     # the two scripted controls of the 3.0 long runs, acting as ordinary agents
        from crucible.agents.inventors import CapacityInventor, PlacementBroker
        agents = [("ada", CapacityInventor(open_source=True), 2_000_000),
                  ("bea", PlacementBroker(open_source=True), 2_000_000)]
    s = run_long(job["family"], job["seed"], job["days"], overrides=job["overrides"], out_dir=job["out"],
                 extra_modules=["entry"] if job["arm"] == "entry" else [], shock=job["shock"], label=label,
                 agents=agents,
                 snapshot_every_days=job["snapshot_every"], resume=job["resume"], progress=progress,
                 check_every_days=job["check_every_days"])
    print(f"[{label}] done in {s['wall_s']}s healthy={s['healthy']}\n{liveness_table(s)}", flush=True)
    return s


def side_by_side(runs: list[dict]) -> dict:
    """Arms next to each other per seed: weekly output per sector, entries and exits, collapses, flags."""
    out = {}
    for seed in sorted({r["seed"] for r in runs}):
        arms = {r["label"].rsplit("_", 1)[1]: r for r in runs if r["seed"] == seed}
        sectors = sorted({s for r in arms.values() for wk in r["weeks"] for s in wk["sectors"]})
        weeks = []
        for i in range(max(len(r["weeks"]) for r in arms.values())):
            row = {"week": i + 1}
            for arm, r in sorted(arms.items()):
                if i >= len(r["weeks"]):
                    continue
                wk = r["weeks"][i]
                row[arm] = {"served": wk["served"], "firms": wk["active_firms_end"], "entries": wk["entries"],
                            "exits": wk["exits"], "collapsed_days": wk["collapsed_days"],
                            "process_failures": sum(wk["process_failures"].values()),
                            "violations": wk["violations_new"],
                            "sectors": {s: {"served": wk["sectors"].get(s, {}).get("served", 0),
                                            "sellers": wk["sectors"].get(s, {}).get("sellers_end", 0)} for s in sectors}}
            weeks.append(row)
        out[str(seed)] = {
            "weeks": weeks,
            "arms": {arm: {"output": r["output"], "collapses": [
                {k: e[k] for k in ("sector", "day", "ratio", "recovered_day", "days_to_recover")} |
                {"cause": e["cause"].get("label", e["cause"]["kind"])} for e in r["liveness"]["collapses"]],
                "unrecovered": r["liveness"]["unrecovered"], "economy": r["economy"],
                "household_budget_bound_from_day": r["household_budget"]["bound_from_day"],
                "entry": {k: v for k, v in (r.get("entry") or {}).items() if k != "ventures"} or None,
                "shock": r.get("shock"), "repeated_process_failures": r["flags"]["repeated_process_failures"],
                "underserved_sectors": r["flags"].get("underserved_sectors", []),
                "invariant_violations": r["invariants"]["violations"], "replay_identical": (r["replay"] or {}).get("identical"),
                "events": r["events"], "wall_s": r["wall_s"], "s_per_sim_day": r["cost"]["s_per_sim_day"],
                "rss_mb_max": r["cost"]["rss_mb_max"], "log_bytes": r["cost"]["log_bytes"], "healthy": r["healthy"]}
                for arm, r in sorted(arms.items())}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="open_digital")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "experiments"))
    ap.add_argument("--arms", default="base,entry")
    ap.add_argument("--prefix", default="ow_longrun")
    ap.add_argument("--overrides", default="{}", help="JSON object of family overrides")
    ap.add_argument("--shock-day", type=int, default=None)
    ap.add_argument("--shock-kind", default="bankrupt_largest",
                    choices=["bankrupt_largest", "bankrupt_sector", "provider_outage"])
    ap.add_argument("--shock-sector", default=None)
    ap.add_argument("--shock-hours", type=int, default=48)
    ap.add_argument("--controls", action="store_true",
                    help="add the two scripted controls of the 3.0 long runs (an inventor and a placement broker)")
    ap.add_argument("--snapshot-every", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-every-days", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    shock = None
    if a.shock_day is not None:
        shock = {"day": a.shock_day, "kind": a.shock_kind, "hours": a.shock_hours,
                 **({"sector": a.shock_sector} if a.shock_sector else {})}
    os.makedirs(a.out, exist_ok=True)
    jobs = [{"family": a.family, "seed": int(s), "days": a.days, "overrides": json.loads(a.overrides), "out": a.out,
             "arm": arm, "shock": shock, "label": f"{a.prefix}_{a.family}_s{int(s)}_{arm}",
             "snapshot_every": a.snapshot_every, "resume": a.resume, "check_every_days": a.check_every_days,
             "controls": a.controls}
            for s in a.seeds.split(",") for arm in a.arms.split(",")]
    t0 = time.time()
    if a.jobs > 1:
        with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            runs = pool.map(one, jobs, chunksize=1)
    else:
        runs = [one(j) for j in jobs]
    combined = {"family": a.family, "days": a.days, "seeds": [int(s) for s in a.seeds.split(",")],
                "arms": a.arms.split(","), "shock": shock, "wall_s": round(time.time() - t0, 1),
                "runs": [r["label"] for r in runs], "by_seed": side_by_side(runs)}
    path = os.path.join(a.out, f"{a.prefix}_{a.family}_summary.json")
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
