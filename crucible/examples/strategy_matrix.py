"""Solvability / multiple-strategies / exploit-resistance matrix (measured; see docs/EXPERIMENTS.md §2).

    PYTHONPATH=crucible/src python crucible/examples/strategy_matrix.py --out crucible/docs/experiments/strategy_matrix.json

For every (goal family, level, world family, seed, policy): build the goal's world, admit the agent, run until
success or horizon (max_days cap), record the evaluator's RunResult. Policies: scripted Builder, Composer,
Employer, BountyHunter and the Random / Idle probes. Runs in parallel processes.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")


def one(job):
    import crucible
    crucible.load_all_modules()
    from dataclasses import asdict
    from crucible.agents.scripted import Builder, BountyHunter, Composer, Employer, IdlePolicy, RandomPolicy
    from crucible.evaluation.harness import run_goal
    from crucible.goals.contracts import generate_goal
    gf, level, wf, seed, pol, max_days = job
    g = generate_goal(gf, wf, seed, level)
    need = g.params.get("need")
    mk = {"builder": lambda: Builder(need=need), "composer": lambda: Composer(need=need or "route.shortest"),
          "employer": lambda: Employer(need=need or "route.shortest"), "bounty": BountyHunter,
          "random": RandomPolicy, "idle": IdlePolicy}[pol]
    if pol == "composer" and need is None:
        # pick a need the world has (the composer builds on existing providers of it)
        from crucible.worldgen import generate
        spec = generate({"family": wf}, seed)
        firms = [m for m in spec.modules if m["name"] == "civ"][0]["config"]["firms"]
        mk = (lambda n=sorted({f['need'] for f in firms} - {"sort.records"})[0]: Composer(need=n))
    t = time.time()
    try:
        r, _ = run_goal(g, mk(), dt=3600, max_days=max_days)
        d = asdict(r)
        d["error"] = None
    except Exception as e:  # noqa: BLE001 - recorded, never hidden
        d = {"goal_family": gf, "world_family": wf, "policy": pol, "success": False, "error": f"{type(e).__name__}: {e}"}
    d.update({"goal": g.id, "goal_family": gf, "level": level, "seed": seed, "policy": pol, "world_family": wf,
              "description": g.description, "wall_s": round(time.time() - t, 1)})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crucible/docs/experiments/strategy_matrix.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default="bazaar,cheap_compute_closed_markets,expensive_compute_open_distribution")
    ap.add_argument("--goals", default="revenue,service,outperform")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--policies", default="builder,composer,employer,bounty,random,idle")
    ap.add_argument("--max-days", type=float, default=21)
    a = ap.parse_args()
    jobs = [(gf, 0, wf, int(s), p, a.max_days) for gf in a.goals.split(",") for wf in a.families.split(",")
            for s in a.seeds.split(",") for p in a.policies.split(",")]
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = []
        for d in pool.imap_unordered(one, jobs):
            res.append(d)
            print(f"[{len(res)}/{len(jobs)}] {d['goal_family']:10s} {d['world_family']:36s} s{d['seed']} {d['policy']:9s} "
                  f"success={d.get('success')} day={d.get('success_day')} prog={d.get('progress')} err={d.get('error')}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"jobs": len(jobs), "wall_s": round(time.time() - t, 1), "runs": res}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
