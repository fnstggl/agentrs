"""cam_live.predict.labdata — diagnostic-scale data and training utilities for Workstream B's evidence.

Used by ``python -m cam_live.predict.diagnose`` and by the tests in ``tests/predict``. It collects honest LabWorld
experience with the master-owned collector, asks every executed intervention about its consequence at more than one
horizon, and runs B's loss providers with a plain optimiser. The production learner is Workstream D's
``ContinualLearner``; :func:`train` exists so prediction can be measured without it. Everything here is
``provenance="diagnostic"`` and none of it is evidence about a live task.
"""
from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..config import CamLiveConfig
from ..contracts import ExperienceRecord, OutcomeKind, OutcomeQuery, new_id
from ..diagnostics.collect import LabCollector, Lifetime, uniform_policy
from ..diagnostics.memstore import MemoryStore
from ..diagnostics.world import LabConfig, LabWorld
from ..learning_api import encode_decisions
from ..nn.api import GROUP_EXPERIENTIAL, GROUP_HEADS, LatentModelAPI
from . import dists as D
from .heads import PredictHeads, attach_heads, world_batch
from .providers import record_rows

DAY = 86400.0
SHORT_HORIZON_S = 6 * 3600.0


def debug_config(seed: int = 0, tiny_vocab: int = 4096) -> CamLiveConfig:
    """``CamLiveConfig.debug()`` with a larger hash vocabulary for the tiny random backbone: at the default 512
    buckets distinct argument values (e.g. two channel numbers) collide into one token and become indistinguishable
    to everything downstream."""
    cfg = CamLiveConfig.debug(seed=seed)
    cfg.backbone.tiny_vocab = int(tiny_vocab)
    return cfg


def build(seed: int = 0, **section: Any) -> Tuple[LatentModelAPI, PredictHeads]:
    from ..nn.model import build_model
    cfg = debug_config(seed)
    if section:
        cfg.sections["predict"] = dict(section)
    model = build_model(cfg)
    return model, attach_heads(model)


def add_horizons(horizons_s: Sequence[float]) -> Callable[[ExperienceRecord], None]:
    """Collector ``on_step`` hook: for every outcome query the environment issued for the executed action, also
    ask the same question at each shorter horizon in ``horizons_s``. The extra queries are measured by the
    environment like any other; nothing is filled in."""
    def hook(rec: ExperienceRecord) -> None:
        if rec.result is None:
            return
        extra: List[OutcomeQuery] = []
        for q in list(rec.result.expected_outcomes):
            for h in horizons_s:
                if h < q.horizon_s:
                    extra.append(dataclasses.replace(q, query_id=new_id("q"), horizon_s=float(h)))
        rec.result.expected_outcomes.extend(extra)
        rec.pending_queries.extend(extra)
    return hook


def lab_config(**over: Any) -> LabConfig:
    base = dict(n_channels=4, key_door=False, noisy_tv=True, budget=1e9, target=30)
    base.update(over)
    return LabConfig(**base)


def collect(store: MemoryStore, seeds: Sequence[int], n_decisions: int, *, base: Optional[LabConfig] = None,
            mechanism_seed: Optional[int] = None, horizons_s: Sequence[float] = (SHORT_HORIZON_S,),
            policy_fn: Optional[Callable[[int], Any]] = None, max_context: int = 40) -> List[Lifetime]:
    """One lifetime per seed. ``mechanism_seed`` fixes the hidden mechanism (same world, different behaviour
    sequences and therefore different sampled consequences); otherwise every seed is a new mechanism."""
    base = base or lab_config()
    out: List[Lifetime] = []
    for s in seeds:
        world = LabWorld(dataclasses.replace(base, seed=s if mechanism_seed is None else mechanism_seed))
        col = LabCollector(store, max_context=max_context, seq_start=1000 * int(s))
        pol = policy_fn(s) if policy_fn is not None else uniform_policy(s)
        out.append(col.run(world, pol, n_decisions, seed=s, on_step=add_horizons(horizons_s)))
    return out


def train(model: LatentModelAPI, providers: Sequence[Any], store: Any,
          sample: Callable[[int], Sequence[ExperienceRecord]], steps: int, *, lr: float = 2e-3,
          clip: float = 1.0, log_every: int = 0) -> List[Dict[str, float]]:
    """Plain Adam over the ``experiential`` and ``heads`` groups with per-group clipping. Returns per-step metrics."""
    groups = model.param_groups()
    opt = torch.optim.Adam([{"params": groups[GROUP_EXPERIENTIAL]}, {"params": groups[GROUP_HEADS]}], lr=lr)
    with_next = any(p.batch_kwargs.get("with_next") for p in providers)
    history: List[Dict[str, float]] = []
    for step in range(steps):
        recs = list(sample(step))
        if not recs:
            continue
        batch = encode_decisions(model, store, recs, with_next=with_next)
        total, metrics = None, {}
        for p in providers:
            loss, m = p.loss(model, batch, store)
            metrics.update(m)
            if loss is not None:
                total = p.weight * loss if total is None else total + p.weight * loss
        if total is None:
            continue
        opt.zero_grad(set_to_none=True)
        total.backward()
        model.clip_gradients({GROUP_EXPERIENTIAL: clip, GROUP_HEADS: clip})
        opt.step()
        metrics["loss"] = float(total.detach())
        history.append(metrics)
        if log_every and step % log_every == 0:
            print(f"step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())
                                              if k in ("loss", "world_nll_count", "evaluator_loss",
                                                       "fast_online_nll_count", "transition_nmse")), flush=True)
    return history


def lifetime_sampler(lifetimes: Sequence[Lifetime], n_lifetimes: int, n_per: int, seed: int = 0,
                     eligible: Optional[Callable[[ExperienceRecord], bool]] = None
                     ) -> Callable[[int], List[ExperienceRecord]]:
    """Batches made of a few records from each of a few lifetimes (context-coherent, as the fast prior needs)."""
    pools = [[r for r in lt.records if eligible is None or eligible(r)] for lt in lifetimes]
    pools = [p for p in pools if p]
    rng = random.Random(seed)

    def sample(step: int) -> List[ExperienceRecord]:
        out: List[ExperienceRecord] = []
        for pool in rng.sample(pools, min(n_lifetimes, len(pools))):
            out += rng.sample(pool, min(n_per, len(pool)))
        return out
    return sample


# ----------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def count_targets(model: LatentModelAPI, heads: PredictHeads, store: Any, records: Sequence[ExperienceRecord],
                  batch_size: int = 32) -> Dict[str, Tensor]:
    """Per matured *count* outcome of ``records``: the head ensemble's mixture NLL and mean, the outcome, its
    horizon and exposure, and the index of its record."""
    nll, mean, ys, hz, ex, ridx, epi = [], [], [], [], [], [], []
    for s in range(0, len(records), batch_size):
        chunk = list(records[s:s + batch_size])
        rows = record_rows(chunk)
        if not any(rows):
            continue
        batch = encode_decisions(model, store, chunk, grad=False)
        wb = world_batch(model, heads, batch.z_chosen, rows)
        m = wb.mask() & (wb.kind_ids == D_KIND_COUNT)
        lp = wb.out.log_prob(wb.targets.y, event=wb.targets.observed)[:, :, 0]            # [M, B, Q]
        mix = -(torch.logsumexp(lp, 0) - math.log(lp.shape[0]))
        pred = wb.out.predictive()
        bi, qi = torch.nonzero(m, as_tuple=True)
        nll.append(mix[bi, qi])
        mean.append(pred["mean"][:, 0][bi, qi])
        epi.append(pred["epistemic_std"][:, 0][bi, qi])
        ys.append(wb.targets.y[bi, qi])
        hz.append(wb.horizon_s[bi, qi])
        ex.append(wb.meta.exposure[bi, qi])
        ridx.append(bi + s)
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)                             # noqa: E731
    return {"nll": cat(nll).double(), "mean": cat(mean).double(), "y": cat(ys).double(), "horizon_s": cat(hz).double(),
            "exposure": cat(ex).double(), "record": cat(ridx).long(), "epistemic_std": cat(epi).double()}


D_KIND_COUNT = list(OutcomeKind.ALL).index(OutcomeKind.COUNT)


def fit_marginal_nb(y: Tensor, exposure: Tensor, steps: int = 300) -> Tuple[float, float]:
    """Maximum-likelihood negative binomial with one rate per unit exposure for *all* actions (action-independent
    baseline). Returns (log rate, log concentration)."""
    y, exposure = y.double(), exposure.double()
    lr0 = math.log(max(float(y.sum() / exposure.sum().clamp_min(1e-9)), 1e-6))
    p = torch.tensor([lr0, 1.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -D.neg_binomial_log_prob_from_log(y, p[0] + torch.log(exposure), p[1]).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(p[0].detach()), float(p[1].detach().clamp(-6.0, 9.0))


def marginal_nll(y: Tensor, exposure: Tensor, log_rate: float, log_conc: float) -> Tensor:
    y, exposure = y.double(), exposure.double()
    return -D.neg_binomial_log_prob_from_log(y, log_rate + torch.log(exposure), torch.full_like(y, log_conc))


def oracle_count_mean(lifetime: Lifetime, record: ExperienceRecord, horizon_s: float) -> float:
    """Ground-truth expected count of a LabWorld intervention at a horizon (tests / diagnostics only)."""
    w = lifetime.world
    info = w._actions.get(record.decision.chosen.action_id)
    if info is None:
        return float("nan")
    ch = info["channel"]
    return w._rate(ch, info["at"]) * info["effort"] * (1.0 - math.exp(-horizon_s / w._delay[ch]))
