"""Experiment 1, condition A: how efficiently does each learner turn identical evidence into predictions?

For every task, one fixed experimental design is run in the simulator: ``n_support`` arms, each holding a setting for
``dwell`` ticks and recording the mean net income per tick (in the task's public money scale) over the hold after its
first tick. Then the simulator is forked once per query arm, so every query is answered by the same world from the
same moment with the same noise (common random numbers). Forks are evaluator-side only: learners see support pairs and
query settings, never query outcomes, mechanisms or references.

Learners predict query outcomes from the first ``k`` support arms, for k in 1, 2, 4, ..., 64. Families: the six
training families supply meta-training data; the four validation families are the held-out test. Crucible's sealed test
families are not used.

    python -m superlearner.unified.experiments.fewshot data
    python -m superlearner.unified.experiments.fewshot run --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np

OUT = "/workspace/agentrs/runs/cam7/fewshot"
TRAIN = ["abm_congestion", "abm_utility", "abm_wom", "scm_additive", "scm_delay", "scm_gate"]
HELDOUT = ["abm_trust_gate", "scm_delay_gate", "scm_invu", "tech_pipeline"]
KS = (1, 2, 4, 8, 16, 32, 64)
L_MAX = 12


# ---------------------------------------------------------------------------- data (evaluator side)
def _arm(rng: np.random.Generator, x0: np.ndarray, n: int) -> np.ndarray:
    """Change one to three levers from the defaults by up to 0.6 of the half-range; the others stay at their defaults."""
    x = x0.copy()
    for j in rng.choice(n, size=min(n, int(rng.integers(1, 4))), replace=False):
        x[j] = np.clip(x0[j] + rng.uniform(-0.6, 0.6), -1, 1)
    return x


def collect(args) -> dict:
    family, seed, n_support, n_query, dwell = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from crucible.research.envs import money_scale
    from crucible.research.tasks import make_task
    task = make_task(family, seed, validate=False)
    sim = task.sim()
    levers = sim.engine.levers(0)
    n = len(levers)
    lo = np.array([lv.lo for lv in levers])
    hi = np.array([lv.hi for lv in levers])
    span = np.where(hi > lo, hi - lo, 1.0)
    x0 = 2 * (np.array([sim.settings[0][lv.name] for lv in levers]) - lo) / span - 1
    scale = money_scale(task)
    rng = np.random.default_rng([seed, 7])

    def hold(s, x):
        vals = {lv.name: float(lo[j] + (x[j] + 1) / 2 * (hi[j] - lo[j])) for j, lv in enumerate(levers)}
        out = []
        for _ in range(dwell):
            r = s.step(vals if s.n_seats == 1 else [vals if q == 0 else None for q in range(s.n_seats)])
            out.append((float(r.revenue[0]) - float(r.cost[0])) / scale)
        return float(np.mean(out[1:])) if dwell > 1 else out[0]

    xs, ys = [], []
    for _ in range(n_support):
        x = _arm(rng, x0, n)
        xs.append(x)
        ys.append(hold(sim, x))
    xq = [_arm(rng, x0, n) for _ in range(n_query)]
    yq = [hold(copy.deepcopy(sim), x) for x in xq]
    ydef = hold(copy.deepcopy(sim), x0)
    pad = lambda a: np.pad(np.asarray(a, dtype=np.float32), [(0, 0), (0, L_MAX - n)])
    return {"family": family, "seed": seed, "x0": np.pad(x0, (0, L_MAX - n)).astype(np.float32),
            "mask": np.pad(np.ones(n), (0, L_MAX - n)).astype(np.float32), "xs": pad(xs), "ys": np.array(ys, np.float32),
            "xq": pad(xq), "yq": np.array(yq, np.float32), "ydef": float(ydef)}


def build(n_train: int = 150, n_test: int = 30, n_support: int = 64, n_query: int = 16, dwell: int = 2, workers: int = 16):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    # "familiar": unused training-range seeds of the training families (in-distribution check, never trained on)
    for name, fams, n, start in (("train", TRAIN, n_train, 0), ("heldout", HELDOUT, n_test, 1_000_000),
                                 ("familiar", TRAIN, 20, 500_000),
                                 # untouched confirmation set for pre-declared criterion C-1 (ledger)
                                 ("confirm", HELDOUT, 30, 1_050_000)):
        path = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(path):
            continue
        jobs = [(f, start + i, n_support, n_query, dwell) for f in fams for i in range(n)]
        with mp.get_context("spawn").Pool(workers) as pool:
            recs = pool.map(collect, jobs, chunksize=4)
        np.savez_compressed(path, **{k: np.stack([r[k] for r in recs]) for k in ("x0", "mask", "xs", "ys", "xq", "yq")},
                            ydef=np.array([r["ydef"] for r in recs]), family=np.array([r["family"] for r in recs]),
                            seed=np.array([r["seed"] for r in recs]))
        print(f"{name}: {len(recs)} tasks", flush=True)


# ---------------------------------------------------------------------------- learners
def _scale(ys_k: "torch.Tensor") -> "torch.Tensor":
    """The reward spread known from the support alone (no query or future information)."""
    import torch
    k = ys_k.shape[1]
    if k >= 2:
        sd = ys_k.std(1)
    else:
        sd = torch.zeros(ys_k.shape[0], device=ys_k.device)
    return torch.maximum(sd, 0.02 * ys_k.abs().mean(1) + 1e-4)


class FixedBLR:
    """Bayesian regression on fixed features: each lever's move from its default and the move squared, plus a bias."""
    name = "blr_fixed"

    def __init__(self, prec: float, noise: float):
        self.prec, self.noise = prec, noise

    def features(self, x, x0, mask):
        import torch
        d = (x - x0.unsqueeze(1)) * mask.unsqueeze(1)
        return torch.cat([d, d * d, torch.ones_like(d[..., :1])], -1)

    def predict(self, xs, ys, xq, x0, mask):
        import torch
        ps, pq = self.features(xs, x0, mask), self.features(xq, x0, mask)
        D = ps.shape[-1]
        prior = torch.full((D,), self.prec, device=xs.device)
        prior[-1] = 1e-6
        s = _scale(ys)
        P = torch.diag(prior) + ps.transpose(1, 2) @ ps / self.noise
        Lc = torch.linalg.cholesky(P)
        # Prior and noise variances both scale with s^2, so the mean is computed in raw units
        mean = torch.cholesky_solve((ps.transpose(1, 2) @ ys.unsqueeze(-1)) / self.noise, Lc).squeeze(-1)
        mu = (pq * mean.unsqueeze(1)).sum(-1)
        var = (pq * torch.cholesky_solve(pq.transpose(1, 2), Lc).transpose(1, 2)).sum(-1) + self.noise
        return mu, var * (s ** 2).unsqueeze(1)


class MeanOnly:
    name = "running_mean"

    def predict(self, xs, ys, xq, x0, mask):
        s = _scale(ys)
        return ys.mean(1, keepdim=True).expand(-1, xq.shape[1]), ((s ** 2) * (1 + 1 / ys.shape[1])).unsqueeze(1).expand(-1, xq.shape[1])


class MetaBLR:
    """CAM's within-episode regression layer, meta-trained by support/query likelihood on the training families."""
    name = "blr_meta"

    def __init__(self, k: int = 4, device: str = "cuda", fixed_features: bool = False):
        from ..core.adaptive import EpisodeRegression
        self.m = EpisodeRegression(L_MAX, k, fixed_features=fixed_features).to(device)
        self.m.decay_logit.data.fill_(30.0)          # arms here are exchangeable: no forgetting

    def predict(self, xs, ys, xq, x0, mask, scale=None):
        m = self.m
        ps = m.phi(xs, x0.unsqueeze(1).expand_as(xs), mask.unsqueeze(1).expand_as(xs))
        pq = m.phi(xq, x0.unsqueeze(1).expand_as(xq), mask.unsqueeze(1).expand_as(xq))
        psd = ps.double()                                # float64 statistics, as in the agent's accumulator
        A = psd.transpose(1, 2) @ psd
        b = (psd * ys.double().unsqueeze(-1)).sum(1)
        mu, var = m.predict(A, b, pq)
        s = _scale(ys) if scale is None else scale
        return mu, var * (s ** 2).unsqueeze(1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation (ties broken by order); 0 when one side is constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _nll(mu, var, y):
    return 0.5 * (np.log(2 * math.pi * var) + (y - mu) ** 2 / var)


def _tensors(d: dict, idx, device):
    import torch
    t = lambda k: torch.as_tensor(d[k][idx], device=device, dtype=torch.float32)
    return t("xs"), t("ys"), t("xq"), t("yq"), t("x0"), t("mask")


def meta_train(model: MetaBLR, train: dict, steps: int, seed: int, device: str, task_scale: bool = False) -> list[float]:
    """``task_scale``: measure training errors in each training task's full-data spread (a training-time label) instead
    of the spread estimated from the k support points, which at small k rewards inflating the variance."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.m.parameters(), lr=1e-3)
    n = len(train["ys"])
    losses = []
    for i in range(steps):
        idx = rng.choice(n, 32, replace=False)
        xs, ys, xq, yq, x0, mask = _tensors(train, idx, device)
        k = int(rng.choice(KS))
        scale = torch.cat([ys, yq], 1).std(1).clamp(min=1e-4) if task_scale else None
        mu, var = model.predict(xs[:, :k], ys[:, :k], xq, x0, mask, scale)
        loss = (0.5 * (torch.log(2 * math.pi * var) + (yq - mu) ** 2 / var)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.m.parameters(), 10.0)
        opt.step()
        losses.append(float(loss))
    return losses


def evaluate(model, data: dict, device: str) -> dict:
    """Per k: predictive NLL (in units of each task's query spread), standardised RMSE, effect error, rank correlation
    of predicted versus true query outcomes, and 90% interval coverage. One value per task, so tasks can be paired."""
    import torch
    out = {}
    n = len(data["ys"])
    xs, ys, xq, yq, x0, mask = _tensors(data, np.arange(n), device)
    unit = np.maximum(data["yq"].std(1), 1e-6)                # evaluator-side unit, for reporting only
    for k in KS:
        with torch.no_grad():
            mu, var = model.predict(xs[:, :k], ys[:, :k], xq, x0, mask)
        mu, var = mu.cpu().numpy(), var.cpu().numpy()
        y = data["yq"]
        z = (y - mu) / np.sqrt(var)
        eff_true = y - data["ydef"][:, None]
        eff_pred = mu - mu.mean(1, keepdims=True) + (y.mean(1, keepdims=True) - data["ydef"][:, None])
        rho = np.array([_spearman(mu[i], y[i]) for i in range(n)])
        out[k] = {"nll": (_nll(mu, var, y) + np.log(unit)[:, None]).mean(1).tolist(),
                  "rmse": (np.sqrt(((y - mu) ** 2).mean(1)) / unit).tolist(),
                  "effect_rmse": (np.sqrt(((eff_true - eff_pred) ** 2).mean(1)) / unit).tolist(),
                  "rank": np.nan_to_num(rho).tolist(),
                  "coverage90": (np.abs(z) < 1.645).mean(1).tolist()}
    return out


def tune_fixed(train: dict, device: str) -> FixedBLR:
    """Generic regression gets its two settings tuned on the training families, the same data the meta-learner sees."""
    best, best_nll = None, float("inf")
    for prec in (0.01, 0.1, 1.0, 10.0):
        for noise in (0.05, 0.2, 0.5, 1.0, 2.0):
            m = FixedBLR(prec, noise)
            ev = evaluate(m, train, device)
            nll = float(np.mean([np.mean(ev[k]["nll"]) for k in KS if k >= 4]))
            if nll < best_nll:
                best, best_nll = m, nll
    return best


def run(seeds, steps: int, device: str, test: str = "heldout", tag: str = "condition_a") -> dict:
    tr, te = (dict(np.load(os.path.join(OUT, f"{s}.npz"))) for s in ("train", test))
    res = {"families": {"train": TRAIN, "heldout": HELDOUT}, "n_heldout_tasks": int(len(te["ys"])), "ks": list(KS),
           "learners": {}}
    fixed = tune_fixed(tr, device)
    res["blr_fixed_settings"] = {"prec": fixed.prec, "noise": fixed.noise}
    res["learners"]["running_mean"] = {"seeds": {"-": evaluate(MeanOnly(), te, device)}}
    res["learners"]["blr_fixed"] = {"seeds": {"-": evaluate(fixed, te, device)}}
    res["learners"]["blr_meta_untrained"] = {"seeds": {}}
    res["learners"]["blr_meta"] = {"seeds": {}, "train_seconds": []}
    res["learners"]["blr_meta_plus_fixed"] = {"seeds": {}, "train_seconds": []}
    res["learners"]["blr_meta_taskscale"] = {"seeds": {}, "train_seconds": []}
    import torch
    for s in seeds:
        for name, fixed, ts in (("blr_meta", False, False), ("blr_meta_plus_fixed", True, False),
                                ("blr_meta_taskscale", True, True)):
            torch.manual_seed(s)
            m = MetaBLR(device=device, fixed_features=fixed)
            if name == "blr_meta":
                res["learners"]["blr_meta_untrained"]["seeds"][str(s)] = evaluate(m, te, device)
            t0 = time.time()
            meta_train(m, tr, steps, s, device, task_scale=ts)
            res["learners"][name]["train_seconds"].append(time.time() - t0)
            res["learners"][name]["seeds"][str(s)] = evaluate(m, te, device)
    res["test_set"] = test
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(res, fh)
    report(res)
    return res


def _seed_task_boot(per_seed: list[np.ndarray], n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a two-level bootstrap: resample training seeds, then tasks, so both sources of variation count."""
    rng = np.random.default_rng(seed)
    arr = np.stack(per_seed)                         # [seeds, tasks]
    S, T = arr.shape
    boots = [arr[rng.integers(S, size=S)][:, rng.integers(T, size=T)].mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _at(per_k: dict, k):
    """Results keyed by k, whether freshly computed (int keys) or loaded from JSON (string keys)."""
    return per_k[str(k)] if str(k) in per_k else per_k[k]


def report(res: dict) -> None:
    print("held-out families, identical support sets; mean [95% interval over seeds and tasks]")
    for metric in ("nll", "rank", "effect_rmse", "coverage90"):
        print(f"\n{metric}")
        print(f"{'k':>4} " + " ".join(f"{n:>26}" for n in res["learners"]))
        for k in res["ks"]:
            cells = []
            for n, lr in res["learners"].items():
                runs = [np.array(_at(v, k)[metric]) for v in lr["seeds"].values()]
                m, lo, hi = _seed_task_boot(runs)
                cells.append(f"{m:+8.3f} [{lo:+.3f},{hi:+.3f}]")
            print(f"{k:>4} " + " ".join(f"{c:>26}" for c in cells))
    base = res["learners"]["blr_fixed"]["seeds"]["-"]
    for name in ("blr_meta", "blr_meta_plus_fixed", "blr_meta_taskscale"):
        if name not in res["learners"]:
            continue
        print(f"\n{name} minus blr_fixed (rank: positive = better; effect_rmse: negative = better):")
        for k in res["ks"]:
            row = []
            for metric in ("rank", "effect_rmse"):
                a = [np.array(_at(v, k)[metric]) for v in res["learners"][name]["seeds"].values()]
                d = _seed_task_boot([x - np.array(_at(base, k)[metric]) for x in a])
                row.append(f"{metric} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
            print(f"  k={k:<3} " + "   ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "run", "report"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--test", default="heldout", choices=["heldout", "familiar", "confirm"])
    a = ap.parse_args()
    if a.cmd == "data":
        build()
    elif a.cmd == "run":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        run([int(s) for s in a.seeds.split(",")], a.steps, dev, a.test, f"condition_a_{a.test}")
    else:
        with open(os.path.join(OUT, f"condition_a_{a.test}.json")) as fh:
            report(json.load(fh))
