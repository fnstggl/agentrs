"""cam_live.nn.fast_compare — does the learned-prior Bayesian fast layer earn its place? (spec 2.1, NA-13)

    python -m cam_live.nn.fast_compare [--steps 2500] [--seeds 0 1] [--out docs/evidence/A/fast_adaptation.json]

Synthetic meta-regression: every task is a function drawn from a *family*; a learner sees the task's points one at a
time and must predict the next one (prequential evaluation at support sizes 0 .. T-1). Methods are meta-trained on
tasks of the training families and evaluated on new tasks of (a) the same families and (b) **held-out families**
never seen in training.

Methods
    learned_prior_bayes  features Phi_theta(x) + the Bayesian last layer of ``cam_live.nn.fast`` with a learned
                         prior, trained ALPaCA-style through the online updates (``fast.online_nll``)
    blr_raw              exact Bayesian linear regression (Gaussian) / Laplace GLM (count) on the raw inputs [x, 1],
                         prior fitted on the training tasks (empirical Bayes)
    gru                  a GRU that reads (x, y) pairs and predicts the next y (recurrent adaptation): Gaussian
                         mean / variance head, negative-binomial head for counts; four times the meta-train steps
    running_stats        running mean / variance (normal-inverse-gamma) for Gaussian targets, Gamma-Poisson running
                         rate for counts; prior fitted on the training tasks

Targets: Gaussian (heteroscedastic across tasks) and Poisson counts with exposure.

Scale protocol: Gaussian targets of a task are standardised with that task's **full-data** mean and spread (a large
independent sample of the task), identically for every method. Standardising with the spread estimated from the
few support points rewarded variance inflation in an earlier project and is deliberately not used.

The result is written as measured; if the learned prior loses, the file says so.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from ..config import PROJECT_ROOT
from ..contracts import OutcomeKind
from . import fast
from .fast import F64, FastBeliefState, FastPrior

TRAIN_FAMILIES = ("sine", "linear", "quadratic", "tanh")
HELDOUT_FAMILIES = ("abs_ridge", "sine_product", "cubic")
D_IN = 2
SUPPORT_SIZES = (0, 1, 2, 3, 5, 8, 12, 16, 24, 31)
#: the recurrent baseline converges more slowly per step and is cheap per step: it gets four times the updates
STEP_FACTOR = {"gru": 4}


# ------------------------------------------------------------------------------------------------------------------
# Task families
# ------------------------------------------------------------------------------------------------------------------
def _unit(n: int, g: torch.Generator) -> Tensor:
    u = torch.randn(n, 1, D_IN, generator=g)
    return u / u.norm(dim=-1, keepdim=True)


def _family_fn(family: str, n: int, g: torch.Generator) -> Callable[[Tensor], Tensor]:
    """A batch of ``n`` random functions of the family: f(x [n, T, 2]) -> [n, T]."""
    U = lambda lo, hi: torch.rand(n, 1, generator=g) * (hi - lo) + lo
    N = lambda s=1.0: torch.randn(n, 1, generator=g) * s
    if family == "sine":
        A, w, p, c = U(0.5, 2.0), U(0.8, 1.6), U(0.0, 2 * math.pi), U(-1.0, 1.0)
        return lambda x: A * torch.sin(w * x[..., 0] + p) + c * x[..., 1]
    if family == "linear":
        a, b, c = N(), N(), N()
        return lambda x: a * x[..., 0] + b * x[..., 1] + c
    if family == "quadratic":
        a, b, c, d = N(0.5), N(0.5), N(0.5), N(0.5)
        return lambda x: a * x[..., 0] ** 2 + b * x[..., 0] * x[..., 1] + c * x[..., 1] ** 2 + d
    if family == "tanh":
        u, A, k, s = _unit(n, g), U(0.5, 2.0), U(0.5, 2.0), N()
        return lambda x: A * torch.tanh(k * (x * u).sum(-1) + s)
    if family == "abs_ridge":
        u, A, s, b = _unit(n, g), U(0.5, 2.0), N(0.7), N()
        return lambda x: A * ((x * u).sum(-1) - s).abs() + b
    if family == "sine_product":
        A, w1, w2, p = U(0.5, 2.0), U(1.6, 2.4), U(1.6, 2.4), U(0.0, 2 * math.pi)
        return lambda x: A * torch.sin(w1 * x[..., 0] + p) * torch.cos(w2 * x[..., 1])
    if family == "cubic":
        u, v, a, b = _unit(n, g), _unit(n, g), N(0.5), N(0.7)
        return lambda x: a * (x * u).sum(-1) ** 3 / 4.0 + b * (x * v).sum(-1)
    raise KeyError(family)


def sample_tasks(families: Sequence[str], n: int, T: int, kind: str, g: torch.Generator, full: int = 256
                 ) -> Dict[str, Tensor]:
    """``n`` tasks (families cycled) with ``T`` points each. Gaussian: y standardised by the task's full-data
    statistics (``full`` extra points); count: Poisson with exposure, log-rate = offset + 0.7 f(x)."""
    xs, ys, es = [], [], []
    per = [n // len(families) + (1 if i < n % len(families) else 0) for i in range(len(families))]
    for fam, m in zip(families, per):
        if m == 0:
            continue
        f = _family_fn(fam, m, g)
        x = torch.rand(m, T + full, D_IN, generator=g) * 4.0 - 2.0
        fx = f(x)
        if kind == OutcomeKind.REAL:
            noise = (torch.rand(m, 1, generator=g) * 0.25 + 0.05) * fx[:, T:].std(1, keepdim=True).clamp_min(1e-3)
            y_raw = fx + noise * torch.randn(m, T + full, generator=g)
            scale = torch.exp(torch.rand(m, 1, generator=g) * math.log(1000.0) + math.log(0.1))
            offset = torch.randn(m, 1, generator=g) * 3.0 * scale
            y_raw = offset + scale * y_raw                                  # arbitrary units per task
            mu, sd = y_raw[:, T:].mean(1, keepdim=True), y_raw[:, T:].std(1, keepdim=True)   # full-data statistics
            ys.append(((y_raw[:, :T] - mu) / sd))
            es.append(torch.ones(m, T))
        else:
            off = torch.rand(m, 1, generator=g) * 3.0 - 1.0
            ex = torch.rand(m, T, generator=g) * 3.5 + 0.5
            rate = torch.exp(off + 0.7 * fx[:, :T]).clamp(max=200.0)
            ys.append(torch.poisson(ex * rate, generator=g))
            es.append(ex)
        xs.append(x[:, :T])
    perm = torch.randperm(n, generator=g)
    return {"x": torch.cat(xs)[perm], "y": torch.cat(ys)[perm].to(F64), "exposure": torch.cat(es)[perm].to(F64)}


# ------------------------------------------------------------------------------------------------------------------
# Methods. Each exposes parameters() and prequential(batch) -> (nll [n, T], predictive mean [n, T]).
# ------------------------------------------------------------------------------------------------------------------
def _bayes_prequential(prior: Dict[str, Tensor], phi: Tensor, y: Tensor, kind: str, exposure: Tensor,
                       noise_mode: str, n_newton: int = 6) -> Tuple[Tensor, Tensor]:
    """Predict each row from the rows before it with the fast layer's functional updates."""
    lead, T, Fd = phi.shape[:-2], phi.shape[-2], phi.shape[-1]
    mu = prior["mu"].expand(*lead, Fd)
    gauss = kind == OutcomeKind.REAL
    nig = gauss and noise_mode == "nig"
    cov = (prior["V"] if nig else prior["cov"]).expand(*lead, Fd, Fd)
    a = prior["a"].expand(lead) if nig else None
    b = prior["b"].expand(lead) if nig else None
    nlls, means = [], []
    for t in range(T):
        f, yt, ex = phi[..., t, :].to(F64), y[..., t], exposure[..., t]
        if nig:
            means.append((f * mu).sum(-1))
            mu, cov, a, b, info = fast.nig_update(mu, cov, a, b, f, yt)
        elif gauss:
            means.append((f * mu).sum(-1))
            mu, cov, info = fast.gaussian_update(mu, cov, f, yt, prior["noise"])
        else:
            means.append(fast.predict_glm(mu, cov, f, kind, ex)[0])
            mu, cov, info = fast.glm_update(mu, cov, f, yt, kind, ex, None, prior["dispersion"], n_newton)
        nlls.append(info["surprise"])
    return torch.stack(nlls, -1), torch.stack(means, -1)


class LearnedPriorBayes(nn.Module):
    """Phi_theta(x): MLP with unit-norm output and a constant feature (the same normalisation as the model's
    feature map) + the learned prior of the Bayesian fast layer."""

    def __init__(self, kind: str, n_features: int = 32, hidden: int = 64, noise_mode: str = "nig"):
        super().__init__()
        self.kind, self.noise_mode, self.F = kind, noise_mode, n_features
        self.net = nn.Sequential(nn.Linear(D_IN, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_features - 1))
        self.prior = FastPrior(n_features, noise_mode=noise_mode)

    def features(self, x: Tensor) -> Tensor:
        raw = self.net(x)
        raw = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
        return torch.cat([raw / math.sqrt(self.F), torch.full_like(raw[..., :1], 1.0 / math.sqrt(self.F))], -1)

    def prequential(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return _bayes_prequential(self.prior.prior(self.kind), self.features(batch["x"]), batch["y"], self.kind,
                                  batch["exposure"], self.noise_mode)


class RawBayes(nn.Module):
    """Bayesian (generalised) linear regression on fixed raw features; only the prior is fitted."""

    def __init__(self, kind: str, constant_only: bool = False, noise_mode: str = "nig"):
        super().__init__()
        self.kind, self.noise_mode, self.constant_only = kind, noise_mode, constant_only
        self.prior = FastPrior(1 if constant_only else D_IN + 1, noise_mode=noise_mode)

    def features(self, x: Tensor) -> Tensor:
        one = torch.ones_like(x[..., :1])
        return one if self.constant_only else torch.cat([x, one], -1)

    def prequential(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return _bayes_prequential(self.prior.prior(self.kind), self.features(batch["x"]), batch["y"], self.kind,
                                  batch["exposure"], self.noise_mode)


class GammaPoissonStats(nn.Module):
    """Running rate with a conjugate Gamma(alpha, beta) prior: predictive is negative binomial."""

    def __init__(self) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.log_beta = nn.Parameter(torch.zeros(()))

    def prequential(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        y, ex = batch["y"], batch["exposure"]
        a0, b0 = self.log_alpha.to(F64).exp(), self.log_beta.to(F64).exp()
        zeros = torch.zeros_like(y[..., :1])
        a = a0 + torch.cat([zeros, y.cumsum(-1)[..., :-1]], -1)
        b = b0 + torch.cat([zeros, ex.cumsum(-1)[..., :-1]], -1)
        logp = (torch.lgamma(y + a) - torch.lgamma(a) - torch.lgamma(y + 1.0)
                + a * torch.log(b / (b + ex)) + y * torch.log(ex / (b + ex)))
        return -logp, ex * a / b


class GRUAdapter(nn.Module):
    """Recurrent adaptation: h_t = GRU(h_{t-1}, [x_t, y_t, log exposure_t]); the next point is predicted from
    [h_{t-1}, x_t, log exposure_t]."""

    def __init__(self, kind: str, hidden: int = 64):
        super().__init__()
        self.kind = kind
        self.gru = nn.GRU(D_IN + 2, hidden, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(nn.Linear(hidden + D_IN + 1, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def prequential(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y, ex = batch["x"], batch["y"], batch["exposure"]
        loge = torch.log(ex).to(torch.float32).unsqueeze(-1)
        yin = (torch.log1p(y) if self.kind == OutcomeKind.COUNT else y).to(torch.float32).unsqueeze(-1)
        n = x.shape[0]
        h0 = self.h0.expand(1, n, -1).contiguous()
        hs, _ = self.gru(torch.cat([x, yin, loge], -1), h0)
        prev = torch.cat([h0.transpose(0, 1), hs[:, :-1]], 1)                    # state before each point
        out = self.head(torch.cat([prev, x, loge], -1)).to(F64)
        if self.kind == OutcomeKind.COUNT:
            # negative binomial (mean m, inverse dispersion r): a point-estimated Poisson rate would be penalised
            # for having no way to express its uncertainty, which is not what this comparison is about
            m = torch.exp(out[..., 0].clamp(-10.0, 10.0) + torch.log(ex))
            r = torch.exp(out[..., 1].clamp(-6.0, 8.0))
            logp = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
                    + r * torch.log(r / (r + m)) + y * torch.log(m / (r + m)))
            return -logp, m
        mean, logvar = out[..., 0], out[..., 1].clamp(-10.0, 6.0)
        return 0.5 * (math.log(2 * math.pi) + logvar + (y - mean) ** 2 / logvar.exp()), mean


def build_methods(kind: str) -> Dict[str, nn.Module]:
    return {"learned_prior_bayes": LearnedPriorBayes(kind), "blr_raw": RawBayes(kind), "gru": GRUAdapter(kind),
            "running_stats": RawBayes(kind, constant_only=True) if kind == OutcomeKind.REAL else GammaPoissonStats()}


# ------------------------------------------------------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------------------------------------------------------
def train(method: nn.Module, kind: str, steps: int, batch: int, T: int, seed: int, lr: float = 2e-3) -> List[float]:
    g = torch.Generator().manual_seed(10_000 + seed)
    opt = torch.optim.Adam(method.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps), eta_min=lr * 0.05)
    curve = []
    for step in range(steps):
        data = sample_tasks(TRAIN_FAMILIES, batch, T, kind, g, full=128)
        nll, _ = method.prequential(data)
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(method.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append(round(float(loss.detach()), 4))
    return curve


@torch.no_grad()
def evaluate(method: nn.Module, data: Dict[str, Tensor]) -> Dict[str, Any]:
    nll, mean = method.prequential(data)
    err2 = (data["y"] - mean) ** 2
    n = nll.shape[0]
    per_k = {}
    for k in SUPPORT_SIZES:
        if k < nll.shape[1]:
            per_k[str(k)] = {"nll": round(float(nll[:, k].mean()), 4),
                             "nll_se": round(float(nll[:, k].std() / math.sqrt(n)), 4),
                             "rmse": round(float(err2[:, k].mean().sqrt()), 4)}
    task_mean = nll[:, 1:].mean(1)                              # per-task average over support sizes >= 1
    return {"per_support_size": per_k, "mean_nll_k_ge_1": round(float(task_mean.mean()), 4),
            "mean_nll_k_ge_1_se": round(float(task_mean.std() / math.sqrt(n)), 4),
            "mean_rmse_k_ge_1": round(float(err2[:, 1:].mean().sqrt()), 4), "_task_nll": task_mean}


def _stateful_agreement(method: LearnedPriorBayes, data: Dict[str, Tensor], n_tasks: int = 4) -> float:
    """Max |difference| between the functional prequential NLL used here and the stateful ``FastBeliefState`` the
    agent runs (same mathematics, two code paths)."""
    with torch.no_grad():
        nll, _ = method.prequential({k: v[:n_tasks] for k, v in data.items()})
        phi = method.features(data["x"][:n_tasks])
    worst = 0.0
    for i in range(n_tasks):
        st = FastBeliefState(method.prior, forgetting=1.0, change_h=0.0, n_newton=6)
        ex = data["exposure"][i] if method.kind == OutcomeKind.COUNT else None
        out = st.update(phi[i], data["y"][i], method.kind, exposure=ex)
        worst = max(worst, float((torch.tensor(out["surprises"], dtype=F64) - nll[i]).abs().max()))
    return worst


def run(steps: int = 2500, seeds: Sequence[int] = (0, 1), batch: int = 48, T: int = 32, n_eval: int = 600,
        kinds: Sequence[str] = (OutcomeKind.REAL, OutcomeKind.COUNT), log: Callable[[str], None] = print
        ) -> Dict[str, Any]:
    t_start = time.time()
    result: Dict[str, Any] = {
        "experiment": "fast adaptation on synthetic meta-regression task families",
        "protocol": {"train_families": list(TRAIN_FAMILIES), "heldout_families": list(HELDOUT_FAMILIES),
                     "inputs": f"x uniform in [-2, 2]^{D_IN}", "points_per_task": T, "support_sizes": list(SUPPORT_SIZES),
                     "meta_train_steps": steps, "step_factor": dict(STEP_FACTOR), "tasks_per_step": batch,
                     "eval_tasks_per_split": n_eval,
                     "seeds": list(seeds), "optimizer": "Adam lr 2e-3 cosine, clip 10",
                     "gaussian_scale": "each task standardised with its full-data mean and spread (256 extra "
                                       "points of the task); never with the support-estimated spread",
                     "count": "y ~ Poisson(exposure * exp(offset + 0.7 f(x))), exposure in [0.5, 4]",
                     "metric": "prequential NLL of the next point (nats) and RMSE of the predictive mean, by "
                               "support size; summary = mean over support sizes 1..T-1"},
        "targets": {}}
    for kind in kinds:
        tname = "gaussian" if kind == OutcomeKind.REAL else "count"
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            methods = build_methods(kind)
            ge = torch.Generator().manual_seed(777 + seed)
            splits = {"in_distribution": sample_tasks(TRAIN_FAMILIES, n_eval, T, kind, ge),
                      "held_out_families": sample_tasks(HELDOUT_FAMILIES, n_eval, T, kind, ge)}
            entry: Dict[str, Any] = {"seed": seed, "methods": {}}
            for name, m in methods.items():
                t0 = time.time()
                curve = train(m, kind, steps * STEP_FACTOR.get(name, 1), batch, T, seed)
                res = {s: evaluate(m, d) for s, d in splits.items()}
                entry["methods"][name] = {"params": sum(p.numel() for p in m.parameters()),
                                          "train_s": round(time.time() - t0, 1), "train_curve": curve, **res}
                log(f"[{tname} seed {seed}] {name}: held-out NLL {res['held_out_families']['mean_nll_k_ge_1']:.4f} "
                    f"in-dist NLL {res['in_distribution']['mean_nll_k_ge_1']:.4f} ({time.time() - t0:.0f} s)")
            entry["stateful_vs_functional_max_abs_nll_diff"] = _stateful_agreement(
                methods["learned_prior_bayes"], splits["held_out_families"])   # type: ignore[arg-type]
            per_seed.append(entry)
        # aggregate over seeds; paired comparison of the learned prior against each baseline on held-out tasks
        summary: Dict[str, Any] = {}
        for split in ("in_distribution", "held_out_families"):
            agg = {}
            for name in per_seed[0]["methods"]:
                vals = [e["methods"][name][split]["mean_nll_k_ge_1"] for e in per_seed]
                rm = [e["methods"][name][split]["mean_rmse_k_ge_1"] for e in per_seed]
                agg[name] = {"mean_nll_k_ge_1": round(sum(vals) / len(vals), 4), "per_seed_nll": vals,
                             "mean_rmse_k_ge_1": round(sum(rm) / len(rm), 4)}
                for k in ("0", "1", "3", "8", "31"):
                    ks = [e["methods"][name][split]["per_support_size"][k]["nll"] for e in per_seed
                          if k in e["methods"][name][split]["per_support_size"]]
                    if ks:
                        agg[name][f"nll_at_k{k}"] = round(sum(ks) / len(ks), 4)
            ranking = sorted(agg, key=lambda n: agg[n]["mean_nll_k_ge_1"])
            paired = {}
            for name in agg:
                if name == "learned_prior_bayes":
                    continue
                d = torch.cat([e["methods"]["learned_prior_bayes"][split]["_task_nll"]
                               - e["methods"][name][split]["_task_nll"] for e in per_seed])
                paired[name] = {"mean_diff_nll": round(float(d.mean()), 4),
                                "se": round(float(d.std() / math.sqrt(d.numel())), 4),
                                "z": round(float(d.mean() / (d.std() / math.sqrt(d.numel()))), 2),
                                "note": "negative = learned prior better; tasks pooled over seeds (the seeds share "
                                        "no tasks, but tasks within a seed share one trained model)"}
            summary[split] = {"methods": agg, "ranking_by_nll": ranking, "winner": ranking[0],
                              "learned_prior_minus_baseline": paired,
                              "learned_prior_wins": ranking[0] == "learned_prior_bayes"}
        for e in per_seed:
            for m in e["methods"].values():
                for split in ("in_distribution", "held_out_families"):
                    m[split].pop("_task_nll", None)
        result["targets"][tname] = {"summary": summary, "per_seed": per_seed}
    verdict = {}
    for tname, t in result["targets"].items():
        s = t["summary"]["held_out_families"]
        verdict[tname] = (f"held-out families: winner = {s['winner']}; ranking = {s['ranking_by_nll']}; "
                          f"learned prior {'WINS' if s['learned_prior_wins'] else 'LOSES'}")
    result["verdict"] = verdict
    result["runtime_s"] = round(time.time() - t_start, 1)
    return result


def run_noise_ablation(steps: int = 2000, seeds: Sequence[int] = (0, 1), batch: int = 48, T: int = 32,
                       n_eval: int = 600, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Gaussian targets only: the learned-prior layer with the noise variance inferred per task (normal-inverse-
    gamma, the agent's default) against the same layer with one learned, fixed noise variance. Same tasks, seeds
    and training as :func:`run`."""
    kind = OutcomeKind.REAL
    t_start = time.time()
    per_seed = []
    for seed in seeds:
        ge = torch.Generator().manual_seed(777 + seed)
        splits = {"in_distribution": sample_tasks(TRAIN_FAMILIES, n_eval, T, kind, ge),
                  "held_out_families": sample_tasks(HELDOUT_FAMILIES, n_eval, T, kind, ge)}
        entry: Dict[str, Any] = {"seed": seed, "modes": {}}
        for mode in ("nig", "fixed"):
            torch.manual_seed(seed)
            m = LearnedPriorBayes(kind, noise_mode=mode)
            curve = train(m, kind, steps, batch, T, seed)
            res = {s: evaluate(m, d) for s, d in splits.items()}
            for r in res.values():
                r.pop("_task_nll", None)
            entry["modes"][mode] = {"train_curve": curve, "learned_noise_variance": round(float(m.prior.noise().detach()), 5),
                                    "learned_nig_shape_a0": round(float(m.prior.a0().detach()), 4), **res}
            log(f"[noise ablation seed {seed}] {mode}: held-out NLL {res['held_out_families']['mean_nll_k_ge_1']:.4f} "
                f"in-dist NLL {res['in_distribution']['mean_nll_k_ge_1']:.4f}")
        per_seed.append(entry)
    summary: Dict[str, Any] = {}
    for split in ("in_distribution", "held_out_families"):
        summary[split] = {}
        for mode in ("nig", "fixed"):
            vals = [e["modes"][mode][split]["mean_nll_k_ge_1"] for e in per_seed]
            row = {"mean_nll_k_ge_1": round(sum(vals) / len(vals), 4), "per_seed_nll": vals}
            for k in ("0", "1", "2", "3", "8", "31"):
                ks = [e["modes"][mode][split]["per_support_size"][k]["nll"] for e in per_seed
                      if k in e["modes"][mode][split]["per_support_size"]]
                if ks:
                    row[f"nll_at_k{k}"] = round(sum(ks) / len(ks), 4)
            summary[split][mode] = row
    return {"experiment": "noise model of the learned-prior Bayesian layer on Gaussian targets (ablation)",
            "protocol": {"meta_train_steps": steps, "tasks_per_step": batch, "points_per_task": T, "seeds": list(seeds),
                         "eval_tasks_per_split": n_eval, "reference_nll_of_unit_normal": round(0.5 * math.log(2 * math.pi * math.e), 4),
                         "note": "same task generator, evaluation tasks and optimiser as the main comparison"},
            "summary": summary, "per_seed": per_seed, "runtime_s": round(time.time() - t_start, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--eval-tasks", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--noise-ablation", action="store_true",
                    help="run only the noise-model ablation (default output: fast_adaptation_noise_mode.json)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    name = "fast_adaptation_noise_mode.json" if args.noise_ablation else "fast_adaptation.json"
    args.out = args.out or os.path.join(PROJECT_ROOT, "docs", "evidence", "A", name)
    if args.noise_ablation:
        res = run_noise_ablation(steps=args.steps, seeds=args.seeds, batch=args.batch, n_eval=args.eval_tasks)
        res["command"] = "python -m cam_live.nn.fast_compare " + " ".join(sys.argv[1:] if argv is None else argv)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res["summary"], indent=1))
        print("written", args.out)
        return 0
    res = run(steps=args.steps, seeds=args.seeds, batch=args.batch, n_eval=args.eval_tasks)
    res["command"] = "python -m cam_live.nn.fast_compare " + " ".join(sys.argv[1:] if argv is None else argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["verdict"], indent=1))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
