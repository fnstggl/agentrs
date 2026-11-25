"""Policy-lag validation sweep: how much learning per update survives a fixed policy lag, with V-trace correction and
without (spec 14.4 "discard samples outside validated training conditions"; manifest F03).

For each (correction, lag, seed) the real pipeline runs in one process, deterministically:
`ActorProcess` (toy world, recurrent policy) → durable queue → `CentralLearner` (lag routing with no lag limit, so
every lag is trained) → `TinyLearnerModel` update → `CheckpointPublisher`. The actor's `LaggedPolicyClient` adopts, at
each trajectory boundary, the validated version exactly ``lag`` updates behind LATEST, so every trained decision has
policy lag ``lag`` (once ``lag`` updates exist). One actor episode batch (``envs`` x ``episode_len`` decisions) feeds
one update. Every ``eval_every`` updates the learner's current weights are evaluated (stochastic policy, as acted) on
held-out world instances of the same mechanism: mean payout rate (chance = 1 / n_tools).

``correction="none"`` trains stale decisions as if they were on-policy (log ρ = 0) — the failure mode spec 14.4
forbids — to show what the correction buys.

Criterion (fixed before running the full sweep; a pilot showed the final score saturates at 1.0 within 60 updates
for lags up to 8, so the final score cannot separate lags): a lag L is *validated* when, averaged over seeds, the
V-trace learner's learning-curve area over chance at lag L — the mean evaluation score over the ``updates`` updates
(evaluated every ``eval_every``), minus chance — is at least ``keep_frac`` (0.9) of that at lag 0, i.e. learning per
update stays within 10% of on-policy training; ``validated_max_lag`` is the largest L such that every tested lag
<= L is validated. `write_validation` stores the sweep and the result in ``dist/lag_validation.json``, which
`offpolicy.LagRouter(max_lag=None)` reads.

CLI::

    python -m superlearner.cam_unified.dist.lagsweep --out runs/cam_unified/lagsweep --write
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

import torch

from .actor import ActorProcess, ActorSpec, run_paths
from .learner import CentralLearner
from .offpolicy import VALIDATION_FILE
from .publish import PolicyClient, read_latest
from .sinks import JsonlArchive
from .toyworld import evaluate_policy, make_tiny_learner


class LaggedPolicyClient(PolicyClient):
    """PolicyClient that adopts the version ``lag`` updates behind LATEST (an injected, fixed policy lag)."""

    def __init__(self, root: str, lag: int, **kw):
        super().__init__(root, **kw)
        self.lag = int(lag)

    def newer(self) -> Optional[dict]:
        lt = read_latest(self.root)
        if lt is None:
            return None
        target = max(0, int(lt["version"]) - self.lag)
        if target <= self.version:
            return None
        d = f"v{target:08d}"
        with open(os.path.join(self.root, d, "meta.json")) as fh:
            m = json.load(fh)
        if not m.get("validated"):
            return None
        return {"version": target, "dir": d, "sha256": m["sha256"]}


def run_one(out_dir: str, lag: int, seed: int, correction: str = "vtrace", updates: int = 60, envs: int = 16,
            episode_len: int = 16, lr: float = 1e-2, eval_every: int = 5, eval_envs: int = 64,
            keep_run: bool = False) -> dict:
    run = os.path.join(out_dir, f"{correction}_lag{lag}_seed{seed}")
    shutil.rmtree(run, ignore_errors=True)
    n = envs * episode_len
    model = make_tiny_learner(seed=seed, lr=lr, correction=correction)
    L = CentralLearner(run, model, sink=JsonlArchive(run_paths(run)["archive"], fsync=False), max_lag=10 ** 6,
                       min_decisions=n, max_batch_decisions=n, keep_versions=lag + 3, record_traces=False,
                       queue_kwargs={"fsync": False}, fsync=False)
    spec = ActorSpec("a0", run, env_kwargs={"B": envs, "episode_len": episode_len, "mechanism_seed": seed},
                     seed=1000 + seed, batch_records=10 ** 6, record_traces=False, queue_kwargs={"fsync": False})
    actor = ActorProcess(spec)
    actor.client = LaggedPolicyClient(run_paths(run)["ckpt"], lag)
    actor.client.wait_for_version(actor.policy, 0, timeout_s=10)
    curve = [(0, evaluate_policy(model.state_dict(), seed, 10_000 + seed, eval_envs, episode_len))]
    lags_seen = []
    t0 = time.time()
    while L.counts["updates"] < updates:
        actor._episode()
        for _ in range(8):
            r = L.step()
            if r is not None:
                lags_seen.append(r["lag"]["lag_max"])
                break
        else:
            raise RuntimeError("the learner did not update on a full episode batch")
        u = L.counts["updates"]
        if u % eval_every == 0:
            curve.append((u, evaluate_policy(model.state_dict(), seed, 10_000 + seed, eval_envs, episode_len)))
    st = L.stats()
    L.close()
    actor.writer.close()
    if not keep_run:
        shutil.rmtree(run, ignore_errors=True)
    finals = [s for _, s in curve[-2:]]
    return {"lag": lag, "seed": seed, "correction": correction, "updates": updates, "curve": curve,
            "final": sum(finals) / len(finals), "auc": sum(s for _, s in curve[1:]) / max(1, len(curve) - 1),
            "trained_lag_hist": st["router"]["lag_hist"], "lag_max_per_update": lags_seen[-5:],
            "exact_on_policy": st["router"]["counts"].get("exact_on_policy", 0), "wall_s": time.time() - t0}


def lag_sweep(out_dir: str, lags=(0, 1, 2, 4, 8, 16, 32), seeds=(0, 1, 2), corrections=("vtrace", "none"),
              **kw) -> list:
    os.makedirs(out_dir, exist_ok=True)
    res = []
    for corr in corrections:
        for lag in lags:
            for seed in seeds:
                r = run_one(out_dir, lag, seed, corr, **kw)
                res.append(r)
                print(f"{corr:6s} lag={lag:3d} seed={seed} final={r['final']:.3f} auc={r['auc']:.3f} "
                      f"lag_hist={r['trained_lag_hist']} {r['wall_s']:.1f}s", flush=True)
    return res


def summarize(results: list, keep_frac: float = 0.9, chance: float = 0.25) -> dict:
    """Mean final/AUC per (correction, lag) and the validated max lag of the V-trace learner (criterion in the
    module docstring)."""
    table: dict = {}
    for r in results:
        t = table.setdefault(r["correction"], {}).setdefault(r["lag"], {"final": [], "auc": []})
        t["final"].append(r["final"])
        t["auc"].append(r["auc"])
    means = {c: {lag: {"final": sum(v["final"]) / len(v["final"]), "auc": sum(v["auc"]) / len(v["auc"]),
                       "n_seeds": len(v["final"])} for lag, v in sorted(d.items())} for c, d in table.items()}
    vt = means.get("vtrace", {})
    validated = None
    if 0 in vt:
        base = vt[0]["auc"] - chance
        for lag in sorted(vt):
            if vt[lag]["auc"] - chance >= keep_frac * base:
                validated = lag
            else:
                break
    return {"means": {c: {str(k): v for k, v in d.items()} for c, d in means.items()},
            "validated_max_lag": validated, "keep_frac": keep_frac, "chance": chance}


def write_validation(results: list, path: str = VALIDATION_FILE, keep_frac: float = 0.9, note: str = "") -> dict:
    s = summarize(results, keep_frac)
    if s["validated_max_lag"] is None:
        raise RuntimeError("no lag validated (lag 0 missing or below the criterion)")
    doc = {"validated_max_lag": s["validated_max_lag"],
           "criterion": f"largest lag L such that every tested lag <= L keeps >= {keep_frac:.0%} of the lag-0 "
                        f"learning-curve area over chance ({s['chance']}) with V-trace, mean over seeds",
           "model": "dist.toyworld.TinyLearnerModel on ToyWorld (4 contexts, 4 tools, 16-step episodes)",
           "note": note, "t": time.time(), "summary": s["means"],
           "runs": [{k: r[k] for k in ("correction", "lag", "seed", "final", "auc", "curve", "trained_lag_hist")}
                    for r in results]}
    with open(path + ".tmp", "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(path + ".tmp", path)
    return doc


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="cam_unified.dist.lagsweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lags", default="0,1,2,4,8,16,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--write", action="store_true", help="store the result in dist/lag_validation.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    res = lag_sweep(a.out, tuple(int(x) for x in a.lags.split(",")), tuple(int(x) for x in a.seeds.split(",")),
                    updates=a.updates)
    with open(os.path.join(a.out, "lag_sweep.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    s = summarize(res)
    print(json.dumps(s, indent=1))
    if a.write:
        write_validation(res, note=f"python -m superlearner.cam_unified.dist.lagsweep --lags {a.lags} "
                                   f"--seeds {a.seeds} --updates {a.updates}")


if __name__ == "__main__":
    _main()
