"""Experiment 2: active causal discovery. Does the learner choose experiments that identify what matters for the goal?

Each learner (frozen weights, fresh context, no mechanism description) runs whole episodes on worlds with hidden
confounders, lagged feedback, gates, interactions, delays, regime switches and distractor levers (`cam7.worlds`) and on
the held-out validation families. Measured by the independent evaluator:

1. Verified outcome: the whole-episode score (0 = holding the defaults, 1 = the private reference policy).
2. Explore-then-commit at a budget of B ticks: at tick B the evaluator forks the simulator (privileged; the learner never
   sees a fork) and
   (a) holds the setting the learner would commit to (its posterior-mean optimum) for the rest of the horizon:
       commit score = task.score(mean net per tick of that hold); experimentation cost = score of ticks [0, B);
   (b) holds each of 8 probe interventions (random moves of 1-3 levers from the learner's current settings) for 12 ticks
       and records the mean net of the last 8: the learner's predicted values of the same probes are compared with the
       truth (Spearman rank correlation; regret of the probe it predicts best, in score units).
3. Experimentation: number of distinct settings tried and total lever movement before B.

Exploration rules compared on one trained CAM 7.0 (same model, same posterior; only the choice of experiments differs):
random, fixed one-lever-at-a-time design, greedy, Thompson sampling, expected information gain, knowledge gradient
(decision-relevant value of information) and the full learner (knowledge gradient + learned residual). Plus the trained
`model_based` baseline with its own Thompson schedule (its posterior-mean model supplies commit and predictions).

    python -m superlearner.cam7.experiments.exp2 run --set dev --out DIR name=spec ...
    python -m superlearner.cam7.experiments.exp2 report --out DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np

from .exp1b import _boot2
from .train import HELDOUT, make_learner

OFFSETS = {"dev": 60_000, "confirm": 70_000}
BUDGET = 24
N_PROBES = 8
PROBE_HOLD, PROBE_SKIP = 12, 4


# ------------------------------------------------------------------------------------------ learner beliefs
def cam7_beliefs(L, probes: np.ndarray):
    """Commit setting (posterior-mean optimum in the trust region) and predicted mean net per tick (money scale) of
    holding each probe, from the learner's current fast state."""
    import torch
    from ..agent import history_batch
    slots = L.slots
    m, cfg = L.model, L.cfg
    with torch.no_grad():
        H = history_batch(slots, cfg, L.device, L.rng, live=True)
        cut = torch.tensor([s.rows for s in slots], device=L.device)
        T = H.x.shape[1]
        keep = (torch.arange(T, device=L.device)[None] >= torch.tensor([s.keep_from for s in slots], device=L.device)[:, None]).float()
        pre = m.prepare(H, cut, keep)
        cand, valid, flags = m.propose(H, cut, pre["post"], pre["ctx"], pre["x_now"], 1, 0, L.eval_gen, L.ctx_rng)
        from .. import posterior as P
        loc, _ = P.mean_cov(pre["post"], m.candidate_features(pre["ctx"], cand, H, cut))
        mean = (loc[..., 0] - loc[..., 1]).masked_fill(~valid, -1e30)
        best = mean.argmax(-1)
        commit = cand[torch.arange(len(slots)), best].cpu().numpy()
        pr = torch.tensor(probes, device=L.device, dtype=torch.float32)
        lp, _ = P.mean_cov(pre["post"], m.candidate_features(pre["ctx"], pr, H, cut))
        pred = (lp[..., 0] - lp[..., 1]).cpu().numpy()
    return commit, pred


def mb_beliefs(L, probes: np.ndarray):
    """The same for `model_based`: its posterior-mean model (temperature 0) over its own features."""
    from crucible.research.learners.model_based import BIAS_SD
    commits, preds = [], []
    for i, s in enumerate(L._slots if hasattr(L, "_slots") else L.slots):
        m, p = L._np, L.p
        n_l = len(s.levers)
        s2 = np.maximum((m.n0 * m.s2 + s.ss) / (m.n0 + s.n_held), 1e-8)
        ws = []
        for k, c in ((0, s.cr), (1, s.cc)):
            lam = np.concatenate([np.tile(m.lam[k], n_l), [1.0 / BIAS_SD ** 2]])
            mu = np.concatenate([np.tile(m.mu[k], n_l), [0.0]])
            A = s.G / s2[k]
            A[np.diag_indices_from(A)] += lam
            ws.append(np.linalg.solve(A, lam * mu + c / s2[k]))
        wnet = ws[0] - ws[1]
        x = s.target.copy() if s.target is not None else s.x0.copy()
        g = int(p["grid"])
        cand = np.clip(x[:, None] + np.linspace(-float(p["trust"]), float(p["trust"]), g)[None, :], -1.0, 1.0)
        d = cand - s.x0[:, None]
        val = np.einsum("lgf,lf->lg", m.feats(d, d, np.broadcast_to(s.x0[:, None], d.shape)), wnet[:-1].reshape(n_l, m.f))
        full = np.zeros(probes.shape[-1])
        full[s.levers] = cand[np.arange(n_l), np.argmax(val, axis=1)]
        commits.append(full)
        pv = []
        for q in probes[i]:
            dq = q[s.levers] - s.x0
            f = m.feats(dq[:, None], dq[:, None], s.x0[:, None])[:, 0]
            pv.append(float(np.concatenate([f.ravel(), [1.0]]) @ wnet))
        preds.append(pv)
    return np.array(commits), np.array(preds)


# ------------------------------------------------------------------------------------------ evaluator
def _hold_net(sim, env, i, x, ticks):
    lo, hi, names = env._lo[i], env._hi[i], env._names[i]
    vals = {nm: float(lo[j] + (np.clip(x[j], -1, 1) + 1) / 2 * (hi[j] - lo[j])) for j, nm in enumerate(names)}
    out = []
    for _ in range(ticks):
        r = sim.step(vals if sim.n_seats == 1 else [vals if q == 0 else None for q in range(sim.n_seats)])
        out.append(float(r.revenue[0]) - float(r.cost[0]))
    return np.array(out)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def run_batch(L, kind: str, tasks, spec, budget: int, rng: np.random.Generator) -> list[dict]:
    from crucible.research.envs import MechVecEnv
    env = MechVecEnv(tasks, spec)
    h = int(tasks[0].horizon)
    obs = env.reset()
    L.reset_context(env.n)
    nets = np.zeros((h, env.n))
    settings = np.zeros((h, env.n, spec.max_levers))
    extra = [None] * env.n
    for t in range(h):
        act = L.act(obs, explore=False)
        if t == budget:
            lm = obs[:, spec.lever_mask] > 0.5
            cur = obs[:, spec.last_action]
            probes = np.repeat(cur[:, None], N_PROBES, 1)
            for i in range(env.n):
                lev = np.flatnonzero(lm[i])
                for p in range(N_PROBES):
                    for j in rng.choice(lev, size=min(len(lev), int(rng.integers(1, 4))), replace=False):
                        probes[i, p, j] = np.clip(cur[i, j] + rng.uniform(-0.5, 0.5), -1, 1)
            commit, pred = (cam7_beliefs if kind == "cam7" else mb_beliefs)(L, probes)
            for i in range(env.n):
                task = tasks[i]
                base = copy.deepcopy(env.sims[i])
                c_net = _hold_net(copy.deepcopy(base), env, i, commit[i] * lm[i], h - budget)
                truth = [_hold_net(copy.deepcopy(base), env, i, probes[i, p] * lm[i], PROBE_HOLD)[PROBE_SKIP:].mean()
                         for p in range(N_PROBES)]
                truth = np.array(truth)
                sc = lambda v: task.score(float(v))
                best_pred = int(np.argmax(pred[i]))
                extra[i] = {"commit_score": sc(c_net.mean()), "probe_spearman": _spearman(pred[i], truth),
                            "probe_regret": sc(truth.max()) - sc(truth[best_pred]),
                            "probe_pred_err": float(np.sqrt(np.mean((pred[i] - truth / env._scale[i]) ** 2)))}
        obs, rew, done, info = env.step(act)
        nets[t] = info["net_cents"]
        settings[t] = obs[:, spec.last_action]
    fin = {r["task"]: r for r in env.drain()}
    recs = []
    for i, task in enumerate(tasks):
        s = settings[:budget, i][:, obs[i, spec.lever_mask] > 0.5]
        distinct = len({tuple(np.round(row, 3)) for row in s})
        recs.append({"task": task.id, "family": task.family, "score": fin[task.id]["score"],
                     "explore_score": task.score(float(nets[:budget, i].mean())),
                     "distinct_settings": distinct, "movement": float(np.abs(np.diff(s, axis=0)).sum()),
                     **(extra[i] or {})})
    return recs


def run(args) -> None:
    import torch
    from crucible.research.learner import EnvSpec
    from crucible.research.tasks import eval_tasks
    from .. import worlds
    spec = EnvSpec()
    fams = list(worlds.FAMILIES) + (HELDOUT if args.with_heldout else [])
    # the Experiment 2 families exist only in this process (worker processes would not know them): build them here
    tasks = eval_tasks(list(worlds.FAMILIES), args.n, offset=OFFSETS[args.set], workers=1, skip_failures=False)
    if args.with_heldout:
        tasks += eval_tasks(HELDOUT, args.n, offset=OFFSETS[args.set], workers=8, skip_failures=True)
    if not tasks:
        raise RuntimeError("no tasks")
    os.makedirs(args.out, exist_ok=True)
    for item in args.items:
        name, rest = item.split("=", 1)
        ckpts = [None]
        if "@" in rest:
            rest, ck = rest.split("@", 1)
            ckpts = ck.split(",")
        learner, over = (rest.split(":", 1) + ["{}"])[:2]
        over = json.loads(over)
        scale = over.pop("scale", "base")
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            continue
        per_seed = {}
        for si, ck in enumerate(ckpts):
            torch.manual_seed(si)
            L = make_learner(learner, spec, si, "cuda", {}, scale)
            if ck:
                L.load(ck)
            if learner == "cam7" and over:
                L.cfg = L.cfg.with_(**over)
                L.model.cfg = L.cfg
            L.freeze(True)
            rng = np.random.default_rng([si, 99])
            recs = []
            by_h: dict = {}
            for t in tasks:
                by_h.setdefault(int(t.horizon), []).append(t)
            t0 = time.time()
            for h, group in sorted(by_h.items()):
                for i in range(0, len(group), 32):
                    recs += run_batch(L, "cam7" if learner == "cam7" else "mb", group[i:i + 32], spec, args.budget, rng)
            per_seed[str(si)] = {"checkpoint": ck, "records": recs, "usage": L.usage().to_dict(), "wall_s": time.time() - t0}
            print(name, si, {k: round(float(np.nanmean([r[k] for r in recs if r.get(k) is not None])), 3)
                             for k in ("score", "commit_score", "explore_score", "probe_spearman", "probe_regret")}, flush=True)
        with open(path, "w") as fh:
            json.dump({"name": name, "learner": learner, "overrides": over, "set": args.set, "families": fams,
                       "budget": args.budget, "seeds": per_seed}, fh)


METRICS = ("score", "commit_score", "explore_score", "probe_spearman", "probe_regret", "distinct_settings")


def report(args) -> dict:
    res = {}
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".json") and not f.startswith("report"):
            d = json.load(open(os.path.join(args.out, f)))
            res[d["name"]] = d
    out = {}
    fams = sorted({r["family"] for d in res.values() for s in d["seeds"].values() for r in s["records"]})
    for fam in ["all"] + fams:
        print(f"\n== {fam}")
        print(f"{'learner':24s}" + "".join(f"{m:>26s}" for m in METRICS))
        for name, d in res.items():
            cells = []
            for mtr in METRICS:
                rows = []
                for s in d["seeds"].values():
                    recs = sorted([r for r in s["records"] if fam == "all" or r["family"] == fam], key=lambda r: r["task"])
                    rows.append([np.nan if r.get(mtr) is None else float(r[mtr]) for r in recs])
                m, lo, hi = _boot2(np.array(rows))
                out.setdefault(fam, {}).setdefault(name, {})[mtr] = (m, lo, hi)
                cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]")
            print(f"{name:24s}" + "".join(f"{c:>26s}" for c in cells))
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="dev", choices=list(OFFSETS))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--with-heldout", action="store_true")
    ap.add_argument("items", nargs="*")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
