"""Retention and cross-family interference of the shared weights (spec 14.5 "Retention", "Cross-domain
interference"; manifest F04) on the toy pipeline.

Two task families = two hidden mechanisms of `ToyWorld` (different context prototypes and best tools). One actor per
family feeds the one central learner through the real pipeline (`ActorProcess` → durable queue → `CentralLearner` →
`TinyLearnerModel` → `CheckpointPublisher` → actors refresh at trajectory boundaries), run in one process so that
the schedule of which family's experience feeds each update is exact. Every ``eval_every`` updates the current
weights are evaluated on held-out worlds of *both* families (stochastic policy, fixed seeds never trained on), and the
scores are appended to ``<run>/eval/heldout.jsonl`` as ``{"metric": "heldout_payout", "family", "version", "value",
"condition"}`` rows — the format the dashboard's retention/transfer section reads.

Conditions (``n`` = updates per family budget):

* ``single0`` / ``single1`` — n updates on one family only;
* ``mixed``      — 2n updates alternating the families (n updates' worth of each family's data);
* ``sequential`` — n updates on family 0, then n updates on family 1.

Measures (per seed, then averaged): ``interference[m] = final(single_m) − final(mixed)[m]`` (both after n updates'
worth of family-m data; positive = sharing the weights with the other family cost family m), ``retention0 =
score0(end of sequential) / score0(end of phase 1)`` and ``forgetting0`` = their difference, ``zero_shot1`` = family-1
score at the end of phase 1 (transfer from family 0 alone) against its initial score. ``final`` = the last
evaluation; chance = 1 / n_tools.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from .actor import ActorProcess, ActorSpec, run_paths
from .learner import CentralLearner
from .sinks import JsonlArchive
from .toyworld import evaluate_policy, make_tiny_learner

FAMILIES = (0, 1)


def run_condition(out_dir: str, condition: str, seed: int = 0, n: int = 30, envs: int = 16, episode_len: int = 16,
                  lr: float = 1e-2, eval_every: int = 5, eval_envs: int = 64, keep_run: bool = False) -> dict:
    if condition == "single0":
        schedule = [0] * n
    elif condition == "single1":
        schedule = [1] * n
    elif condition == "mixed":
        schedule = [i % 2 for i in range(2 * n)]
    elif condition == "sequential":
        schedule = [0] * n + [1] * n
    else:
        raise ValueError(condition)
    run = os.path.join(out_dir, f"{condition}_seed{seed}")
    shutil.rmtree(run, ignore_errors=True)
    nd = envs * episode_len
    model = make_tiny_learner(seed=seed, lr=lr)
    L = CentralLearner(run, model, sink=JsonlArchive(run_paths(run)["archive"], fsync=False), min_decisions=nd,
                       max_batch_decisions=nd, record_traces=False, queue_kwargs={"fsync": False}, fsync=False)
    actors = {}
    for m in sorted(set(schedule)):
        a = ActorProcess(ActorSpec(f"fam{m}", run, env_kwargs={"B": envs, "episode_len": episode_len,
                                                                "mechanism_seed": m, "env_prefix": f"m{m}toy"},
                                   seed=1000 + 10 * seed + m, batch_records=10 ** 6, record_traces=False,
                                   queue_kwargs={"fsync": False}))
        a.client.wait_for_version(a.policy, 0, timeout_s=10)
        actors[m] = a
    eval_path = os.path.join(run, "eval", "heldout.jsonl")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    curves = {m: [] for m in FAMILIES}

    def evaluate(u):
        sd = model.state_dict()
        with open(eval_path, "a") as fh:
            for m in FAMILIES:
                s = evaluate_policy(sd, m, 20_000 + seed, eval_envs, episode_len)
                curves[m].append((u, s))
                fh.write(json.dumps({"metric": "heldout_payout", "family": f"mech{m}", "version": u, "value": s,
                                     "condition": condition, "seed": seed}) + "\n")
    evaluate(0)
    t0 = time.time()
    trained_by_family = {m: 0 for m in FAMILIES}
    for i, fam in enumerate(schedule):
        actors[fam]._episode()
        for _ in range(8):
            r = L.step()
            if r is not None:
                for a, k in r["by_actor"].items():
                    trained_by_family[int(a[3:])] += k
                break
        else:
            raise RuntimeError("the learner did not update on a full episode batch")
        u = L.counts["updates"]
        if u % eval_every == 0 or (condition == "sequential" and u == n):
            evaluate(u)
    L.close()
    for a in actors.values():
        a.writer.close()
    if not keep_run:
        shutil.rmtree(run, ignore_errors=True)
    return {"condition": condition, "seed": seed, "n": n, "curves": {f"mech{m}": c for m, c in curves.items()},
            "trained_by_family": trained_by_family, "wall_s": time.time() - t0, "run_dir": run if keep_run else None}


def _at(curve: list, u: int) -> float:
    return dict(curve)[u]


def measure(out_dir: str, seeds=(0, 1, 2), n: int = 30, **kw) -> dict:
    runs = []
    for seed in seeds:
        for cond in ("single0", "single1", "mixed", "sequential"):
            r = run_condition(out_dir, cond, seed, n, **kw)
            runs.append(r)
            print(f"{cond:10s} seed={seed} final mech0={r['curves']['mech0'][-1][1]:.3f} "
                  f"mech1={r['curves']['mech1'][-1][1]:.3f} trained={r['trained_by_family']} {r['wall_s']:.1f}s",
                  flush=True)
    per_seed = []
    for seed in seeds:
        R = {r["condition"]: r for r in runs if r["seed"] == seed}
        fin = lambda c, m: R[c]["curves"][f"mech{m}"][-1][1]       # noqa: E731
        p1 = _at(R["sequential"]["curves"]["mech0"], n)
        end = fin("sequential", 0)
        per_seed.append({"seed": seed,
                         "interference_mech0": fin("single0", 0) - fin("mixed", 0),
                         "interference_mech1": fin("single1", 1) - fin("mixed", 1),
                         "mech0_after_phase1": p1, "mech0_after_phase2": end,
                         "retention0": end / p1 if p1 > 0 else None, "forgetting0": p1 - end,
                         "mech1_initial": R["sequential"]["curves"]["mech1"][0][1],
                         "zero_shot1": _at(R["sequential"]["curves"]["mech1"], n),
                         "mech1_after_phase2": fin("sequential", 1)})
    keys = [k for k in per_seed[0] if k != "seed"]
    mean = {k: sum(p[k] for p in per_seed if p[k] is not None) / max(1, sum(p[k] is not None for p in per_seed))
            for k in keys}
    return {"per_seed": per_seed, "mean": mean, "n": n, "seeds": list(seeds), "runs": runs}


def _main(argv=None) -> None:
    import argparse
    import torch
    ap = argparse.ArgumentParser(prog="cam_unified.dist.interference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    res = measure(a.out, tuple(int(x) for x in a.seeds.split(",")), a.n)
    with open(os.path.join(a.out, "interference.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"mean": res["mean"], "per_seed": res["per_seed"]}, indent=1))


if __name__ == "__main__":
    _main()
