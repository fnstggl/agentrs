"""Train and evaluate the baseline learners of `crucible.research.learners` on hidden-mechanism tasks.

    # one training run with a learning curve (resumable: rerun the same command to continue)
    python examples/ow_train_learners.py train --learner ppo_gru --seed 0 --steps 1000000

    # the comparison table on validated tasks of the training families (same tasks for every learner)
    python examples/ow_train_learners.py table

    # the same frozen checkpoints on held-out families, per transfer kind (run once, at the end)
    python examples/ow_train_learners.py heldout

    # merge the per-run curves into one file
    python examples/ow_train_learners.py curves

Everything is written under `docs/experiments/` (`ow_learners_*.json`, checkpoints in `checkpoints/`). Training
uses the CPU only (the GPU is hidden from the process) with `OW_THREADS` torch threads per process (default 1:
the networks are small and one thread is the most CPU-efficient). Set `CRUCIBLE_CACHE` to share validated tasks
between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # these learners never use a GPU
THREADS = int(os.environ.get("OW_THREADS", "1"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "experiments")
CKPT = os.path.join(OUT, "checkpoints")
TRAIN_FAMILIES = ["scm_additive", "scm_delay", "scm_gate", "abm_utility", "abm_wom", "abm_congestion"]
HELD_OUT = ["scm_delay_gate", "scm_invu", "scm_feedback", "scm_regime", "abm_trust_gate", "abm_network_fx",
            "tech_pipeline", "tech_contention"]
POOL = {"scm": 300, "abm": 300}            # distinct training tasks per family (cycled in shuffled order)
DEV_OFFSET, DEV_N = 100, 6                 # curve / model-selection tasks: validation seeds 1_000_100 ...
TABLE_N, HELD_N = 12, 6                    # table: validation seeds 1_000_000 ... 1_000_011


def _setup():
    import torch
    torch.set_num_threads(THREADS)
    from crucible.research import learners, tasks as T
    from crucible.research.learner import EnvSpec
    return learners, T, EnvSpec()


def _dump(path: str, obj) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _pool(T, families):
    from crucible.research.learners.training import train_pool
    out = []
    for eng in ("scm", "abm"):
        fams = [f for f in families if f.startswith(eng)]
        if fams:
            out += train_pool(fams, POOL[eng], workers=2)
    return out


def _dev_tasks(T, families):
    return T.build_tasks([(f, T.VAL_SEED0 + DEV_OFFSET + i) for f in families for i in range(DEV_N)], workers=2)


def _run_name(a) -> str:
    return a.tag or f"{a.learner}_s{a.seed}"


# ---------------------------------------------------------------------------- train
def cmd_train(a) -> None:
    learners, T, spec = _setup()
    from crucible.research.learners.training import pool_stream, train
    fams = a.families.split(",") if a.families else TRAIN_FAMILIES
    kw = json.loads(a.kwargs) if a.kwargs else {}
    name = _run_name(a)
    os.makedirs(CKPT, exist_ok=True)
    curve_path = os.path.join(OUT, f"ow_learners_curve_{name}.json")
    ckpt_path = os.path.join(CKPT, f"ow_learners_{name}.pt")
    lr = learners.make(a.learner, spec, seed=a.seed, **kw)
    run = {"learner": a.learner, "seed": a.seed, "kwargs": kw, "families": fams, "n_envs": a.n_envs,
           "budget_steps": a.steps, "eval_every": a.eval_every, "pool_per_family": POOL,
           "dev_tasks": f"validation seeds {T.VAL_SEED0 + DEV_OFFSET}..+{DEV_N - 1} of each training family",
           "threads": THREADS, "task_version": T.TASK_VERSION, "resumes": 0, "curve": []}
    if os.path.exists(curve_path) and os.path.exists(ckpt_path) and not a.fresh:
        with open(curve_path) as fh:
            run = json.load(fh)
        lr.load(ckpt_path)
        last = run["curve"][-1]
        u = lr.usage()
        u.train_cpu_s, u.infer_cpu_s, u.env_steps = last["train_cpu_s"], last["infer_cpu_s"], last.get("env_steps", 0)
        run["resumes"] += 1
        run["budget_steps"], run["threads"] = a.steps, THREADS
        run.setdefault("resumed_at", []).append({"steps": last["steps"], "threads": THREADS})
        print(f"resuming {name} at {last['steps']} steps", flush=True)
    pool = _pool(T, fams)
    dev = _dev_tasks(T, fams)
    print(f"{name}: {len(pool)} training tasks, {len(dev)} dev tasks, params={lr.usage().params}", flush=True)
    cpu0, wall0 = time.process_time(), time.time()
    base_cpu, base_wall = run.get("process_cpu_s", 0.0), run.get("wall_s", 0.0)

    def on_point(p: dict) -> None:
        p["env_steps"] = int(lr.usage().env_steps)
        p["search_evals"] = int(lr.usage().search_evals)
        run["curve"] = curve
        run["process_cpu_s"] = base_cpu + time.process_time() - cpu0
        run["wall_s"] = base_wall + time.time() - wall0
        run["usage"] = lr.usage().to_dict()
        run["finished"] = bool(p["steps"] >= a.steps)
        lr.save(ckpt_path)
        if a.keep_checkpoints:
            lr.save(os.path.join(CKPT, f"ow_learners_{name}_{p['steps']}.pt"))
        _dump(curve_path, run)

    def log(s: str) -> None:
        extra = getattr(lr, "last_stats", None) or getattr(lr, "last_loss", None)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {s} | {extra}", flush=True)

    curve = list(run["curve"])
    stream = pool_stream(pool, shuffle_seed=1000 * a.seed + run["resumes"])
    # `train` appends to the list it is given a copy of; keep our reference in step through on_point
    def on_point_wrap(p: dict) -> None:
        curve.append(p)
        on_point(p)
    train(lr, stream, a.steps, n_envs=a.n_envs, eval_tasks=dev, eval_every=a.eval_every, on_point=on_point_wrap,
          curve=run["curve"], log=log)
    print(f"{name}: done, {run['curve'][-1]['steps']} steps, wall {run['wall_s']:.0f}s, "
          f"process cpu {run['process_cpu_s']:.0f}s", flush=True)


# ---------------------------------------------------------------------------- the learners of the tables
TRAINABLE = ("ppo_gru", "ppo_gru_flat", "model_based", "evo_probe")


def _roster(learners, spec, seeds=(0, 1, 2)) -> list[tuple[str, callable]]:
    """(row name, factory) for every row of the tables. A trained row exists only if its run finished."""
    rows: list[tuple[str, callable]] = [
        ("hold", lambda: learners.make("hold", spec)),
        ("random", lambda: learners.make("random", spec, seed=1)),
        ("probe_exploit", lambda: learners.make("probe_exploit", spec)),
        ("hill_climb", lambda: learners.make("hill_climb", spec)),
    ]

    def trained(name: str, run: str, **kw):
        def build():
            lr = learners.make(name, spec, seed=0, **kw)
            lr.load(os.path.join(CKPT, f"ow_learners_{run}.pt"))
            return lr
        return build

    def finished(run: str) -> bool:
        cp = os.path.join(OUT, f"ow_learners_curve_{run}.json")
        if not (os.path.exists(cp) and os.path.exists(os.path.join(CKPT, f"ow_learners_{run}.pt"))):
            return False
        with open(cp) as fh:
            return bool(json.load(fh).get("finished"))
    for name in TRAINABLE:
        rows.append((f"{name} untrained", (lambda n=name: learners.make(n, spec, seed=0))))
        for s in seeds:
            if finished(f"{name}_s{s}"):
                rows.append((f"{name} trained s{s}", trained(name, f"{name}_s{s}")))
    # the recurrent policy acting on its mean (no dithering inside the episode), for comparison
    rows.append(("ppo_gru untrained (mean action)", lambda: learners.make("ppo_gru", spec, seed=0, eval_sample=False)))
    for s in seeds:
        if finished(f"ppo_gru_s{s}"):
            rows.append((f"ppo_gru trained s{s} (mean action)", trained("ppo_gru", f"ppo_gru_s{s}", eval_sample=False)))
    return rows


def _stats(S, recs_like: dict) -> dict:
    """Summary of one row from per-task scores and successes (lists aligned with the task list)."""
    sc, ok, fam, kind = recs_like["scores"], recs_like["successes"], recs_like["families"], recs_like["transfers"]

    def part(sel):
        return {"score": S.mean_ci([x for x, m in zip(sc, sel) if m]),
                "success": S.proportion_ci(int(sum(1 for x, m in zip(ok, sel) if m and x)), int(sum(sel)))}
    return {"score": S.mean_ci(sc), "success": S.proportion_ci(int(sum(ok)), len(ok)),
            "by_family": {f: part([x == f for x in fam]) for f in sorted(set(fam))},
            "by_transfer": {k: part([x == k for x in kind]) for k in sorted(set(kind))}}


def _evaluate_rows(rows, tasks, spec, out_path: str, meta: dict) -> dict:
    from crucible.research import stats as S
    from crucible.research.learners.training import evaluate
    res = {"meta": meta, "tasks": [t.id for t in tasks], "rows": {}, "paired": {}}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            old = json.load(fh)
        if old.get("tasks") == res["tasks"] and old.get("meta", {}).get("task_version") == meta.get("task_version"):
            res["rows"] = {k: v for k, v in old["rows"].items() if "mean of" not in k}
    fams, kinds = [t.family for t in tasks], [t.transfer for t in tasks]
    for name, build in rows:
        if name in res["rows"]:
            continue
        lr = build()
        t0, c0 = time.time(), time.process_time()
        recs = evaluate(lr, tasks, spec)
        u = lr.usage()
        row = {"scores": [r["score"] for r in recs], "successes": [bool(r["success"]) for r in recs],
               "families": fams, "transfers": kinds}
        row.update(_stats(S, row))
        row.update({"score_first_half": S.mean_ci([r["score_first_half"] for r in recs]),
                    "score_second_half": S.mean_ci([r["score_second_half"] for r in recs]),
                    "ticks_fallback_or_idle": [r["ticks_fallback"] + r["ticks_idle"] for r in recs],
                    "usage": {"env_steps": int(u.env_steps), "infer_cpu_s": float(u.infer_cpu_s),
                              "search_evals": int(u.search_evals), "params": int(u.params),
                              "updates": int(u.updates)},
                    "wall_s": time.time() - t0, "process_cpu_s": time.process_time() - c0})
        res["rows"][name] = row
        print(f"{name:36s} score {row['score']['mean']:+.3f} [{row['score']['lo']:+.3f}, {row['score']['hi']:+.3f}]  "
              f"success {row['success']['p']:.2f} [{row['success']['lo']:.2f}, {row['success']['hi']:.2f}]", flush=True)
        _dump(out_path, res)
    # a trained learner over its training seeds: per-task mean score over seeds (uncertainty over tasks), and
    # successes pooled over (seed, task) pairs
    rows_ = res["rows"]
    groups: dict[str, list[str]] = {}
    for name in rows_:
        if " trained s" in name:
            base, _, rest = name.partition(" trained s")
            groups.setdefault(base + " trained" + rest[1:], []).append(name)
    for label, members in groups.items():
        sc = np.mean([rows_[m]["scores"] for m in members], axis=0).tolist()
        ok = [bool(x) for m in members for x in rows_[m]["successes"]]
        agg = {"scores": sc, "successes": ok, "families": fams * len(members), "transfers": kinds * len(members),
               "members": members}
        st = _stats(S, {"scores": sc * len(members), "successes": ok, "families": agg["families"],
                        "transfers": agg["transfers"]})
        agg.update(_stats(S, {"scores": sc, "successes": [False] * len(sc), "families": fams, "transfers": kinds}))
        agg["success"] = st["success"]
        for k in agg["by_family"]:
            agg["by_family"][k]["success"] = st["by_family"][k]["success"]
        for k in agg["by_transfer"]:
            agg["by_transfer"][k]["success"] = st["by_transfer"][k]["success"]
        agg["per_seed_mean_score"] = {m: rows_[m]["score"]["mean"] for m in members}
        rows_[f"{label} (mean of {len(members)} seeds)"] = agg
    # paired differences on identical tasks
    res["paired"] = {}
    pairs = [("probe_exploit", "hold"), ("hill_climb", "hold"), ("model_based untrained", "hold"),
             ("model_based untrained", "probe_exploit")]
    for name in rows_:
        if " trained" in name:
            base = name.split(" trained")[0]
            untrained = f"{base} untrained" + (" (mean action)" if "(mean action)" in name else "")
            pairs += [(name, untrained), (name, "hold")]
    for a_, b_ in pairs:
        if a_ in rows_ and b_ in rows_:
            res["paired"][f"{a_} - {b_}"] = S.paired_diff(rows_[a_]["scores"], rows_[b_]["scores"])
    _dump(out_path, res)
    return res


def _print_table(res: dict, by: str = "by_family") -> None:
    cols = sorted(next(iter(res["rows"].values()))[by])
    print("\n| learner | score (95% CI) | success (Wilson 95%) | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for name, r in res["rows"].items():
        s, p = r["score"], r["success"]
        print(f"| {name} | {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | {p['p']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}] | "
              + " | ".join(f"{r[by][c]['score']['mean']:+.2f} / {r[by][c]['success']['p']:.2f}" for c in cols) + " |")
    print("\n| paired difference (same tasks) | n | mean | 95% CI | p (sign-flip) | better / worse |")
    print("|---|---|---|---|---|---|")
    for k, d in res["paired"].items():
        print(f"| {k} | {d['n']} | {d['mean_diff']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p_value']:.4f} | "
              f"{d['frac_better']:.2f} / {d['frac_worse']:.2f} |")


def cmd_table(a) -> None:
    learners, T, spec = _setup()
    tasks = T.build_tasks([(f, T.VAL_SEED0 + i) for f in TRAIN_FAMILIES for i in range(TABLE_N)], workers=2)
    res = _evaluate_rows(_roster(learners, spec), tasks, spec, os.path.join(OUT, "ow_learners_table.json"),
                         {"what": "frozen evaluation, fresh context, one episode per task; per-family cells are "
                                  "mean score / success rate",
                          "tasks": f"{TABLE_N} validated tasks per training family, validation seeds "
                                   f"{T.VAL_SEED0}..{T.VAL_SEED0 + TABLE_N - 1}", "families": TRAIN_FAMILIES,
                          "task_version": T.TASK_VERSION})
    _print_table(res)


def cmd_heldout(a) -> None:
    learners, T, spec = _setup()
    tasks = T.build_tasks([(f, T.VAL_SEED0 + i) for f in HELD_OUT for i in range(HELD_N)], workers=2,
                          skip_failures=True)
    res = _evaluate_rows(_roster(learners, spec), tasks, spec, os.path.join(OUT, "ow_learners_heldout.json"),
                         {"what": "frozen evaluation on held-out families, fresh context, nothing tuned on them; "
                                  "cells are mean score / success rate",
                          "tasks": f"{HELD_N} validated tasks per family", "families": HELD_OUT,
                          "transfer": {f: T.MECH_FAMILIES[f].transfer for f in HELD_OUT},
                          "task_version": T.TASK_VERSION, "build_failures": getattr(T.build_tasks, "failures", [])})
    _print_table(res, "by_transfer")
    _print_table(res, "by_family")


def cmd_curves(a) -> None:
    from crucible.research import stats as S
    out = {"runs": {}, "bands": {}}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("ow_learners_curve_") and fn.endswith(".json"):
            with open(os.path.join(OUT, fn)) as fh:
                run = json.load(fh)
            out["runs"][fn[len("ow_learners_curve_"):-5]] = run
    by: dict[str, list] = {}
    for name, run in out["runs"].items():
        by.setdefault(run["learner"], []).append(([p["steps"] for p in run["curve"]], [p["score"] for p in run["curve"]]))
    for learner, curves in by.items():
        grid = sorted({x for xs, _ in curves for x in xs})
        out["bands"][learner] = S.curve_band(curves, grid)
        print(learner, "runs", len(curves))
        for x, m, lo, hi, n in zip(grid, *(out["bands"][learner][k] for k in ("mean", "lo", "hi", "n"))):
            print(f"  steps {int(x):8d}  score {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")
    _dump(os.path.join(OUT, "ow_learners_curves.json"), out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--learner", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=1_000_000)
    t.add_argument("--eval-every", type=int, default=50_000)
    t.add_argument("--n-envs", type=int, default=16)
    t.add_argument("--families", default="")
    t.add_argument("--kwargs", default="", help="JSON of learner options")
    t.add_argument("--tag", default="", help="run name (default <learner>_s<seed>)")
    t.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    t.add_argument("--keep-checkpoints", action="store_true", help="also keep one checkpoint per curve point")
    t.set_defaults(fn=cmd_train)
    for name, fn in (("table", cmd_table), ("heldout", cmd_heldout), ("curves", cmd_curves)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
