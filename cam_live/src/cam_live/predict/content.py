"""cam_live.predict.content — a generic, content-dependent, self-supervised predictive target (IMP, ST's request).

The world model's outcome questions so far are scalars the executor or the verifier reports (``action.succeeded``,
bookkeeping metrics, the objective metric). None of them depends on *what an observation says*, so the information
an action returns had no measurable value and imagination had nothing content-dependent to predict (ST measured
|realised information value| <= 5e-4 nats on the real model). This module adds a prediction of the **content of
what an action returns**:

    target y (K dims)  = P^T  mean_items e(text)  +  mean_items sum_q symlog(value_q) h(name_q) / sqrt(n_q)

over the experience items the action added to the context (its result line and the observations it returned):
``e`` is the frozen backbone's standardised semantic embedding of the item text (``model.embed_texts``; a fixed
target, not one the learner can move), ``P`` a fixed random projection (seeded), ``h(name)`` a fixed pseudo-random
unit vector per quantity name (sha256 of the name) and ``symlog`` keeps exact numbers on a bounded scale. Nothing
names a task, a tool or a metric.

:class:`ContentHead` predicts y from the candidate's decision latent ``z`` with a bootstrapped ensemble (randomised,
zero-mean prior functions as in ``LatentTransition``) and a per-dimension Gaussian: predictive NLL (mixture over
members), epistemic variance (members' disagreement) and the **information gain** of observing the content,
``IG = 0.5 sum_d log(1 + var_m(mu_d) / mean_m sigma^2_d)`` nats (mutual information between the content and the
model, Gaussian approximation) — large for an action whose returned content the model cannot yet predict but could
(reducible), small for one whose content is predictable or pure noise (aleatoric).

:class:`ContentLoss` (registered by ``predict.providers.default_providers``) trains it with the continual learner on
every executed decision whose result added items; gradients also reach the decision trunk, so the shared
representation learns what actions will return. ``Predictor.evaluate_candidates`` records ``content_ig`` and
``content_epistemic`` per candidate; :func:`record_content_nll` gives a record's content NLL under the current weights
(for realised information value: the same record's NLL with and without a piece of context).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import ExperienceItem, ExperienceRecord, ExperienceStore, ItemKind
from .world import _EnsembleMLP

HEAD_CONTENT = "predict_content"
CONTENT_DIM = 16
F64 = torch.float64


def _symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _name_vector(name: str, k: int) -> Tensor:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % (2 ** 62)
    v = torch.randn(k, generator=torch.Generator().manual_seed(seed), dtype=F64)
    return v / v.norm().clamp_min(1e-12)


class ContentTarget:
    """The fixed map from returned items to a K-dimensional content vector (module docstring)."""

    def __init__(self, d_sem: int, k: int = CONTENT_DIM, seed: int = 0):
        self.d_sem, self.k = int(d_sem), int(k)
        g = torch.Generator().manual_seed(1_000_003 + int(seed))
        self.proj = torch.randn(self.d_sem, self.k, generator=g, dtype=F64) / math.sqrt(self.d_sem)
        self._names: Dict[str, Tensor] = {}

    def _h(self, name: str) -> Tensor:
        if name not in self._names:
            self._names[name] = _name_vector(name, self.k)
        return self._names[name]

    @torch.no_grad()
    def __call__(self, model: Any, groups: Sequence[Sequence[ExperienceItem]]) -> Tuple[Tensor, Tensor]:
        """(y [B, K] float64, mask [B] bool) for each group of returned items (an empty group is masked)."""
        b = len(groups)
        y = torch.zeros(b, self.k, dtype=F64)
        mask = torch.zeros(b, dtype=torch.bool)
        texts = [it.text or "" for g in groups for it in g]
        emb = model.embed_texts(texts).detach().to("cpu", F64) if texts else torch.zeros(0, self.d_sem, dtype=F64)
        pos = 0
        for i, g in enumerate(groups):
            if not g:
                continue
            e = emb[pos:pos + len(g)]
            pos += len(g)
            v = e.mean(0) @ self.proj
            num = torch.zeros(self.k, dtype=F64)
            for it in g:
                qs = [q for q in it.quantities if q.value is not None and math.isfinite(float(q.value))]
                for q in qs:
                    num = num + _symlog(float(q.value)) * self._h(str(q.name)) / math.sqrt(len(qs))
            y[i] = v + num / len(g)
            mask[i] = True
        return y, mask


def returned_items(store: ExperienceStore, rec: ExperienceRecord) -> List[ExperienceItem]:
    """The items a decision's result added to the context (its result line and returned observations; outcome
    readings of earlier questions are left out — they are what the world model's scalar questions predict)."""
    if not rec.post_item_refs:
        return []
    before = set(rec.context_item_refs)
    new = [r for r in rec.post_item_refs if r not in before]
    return [it for it in store.items(new) if it.kind != ItemKind.OUTCOME]


class ContentOutput:
    def __init__(self, mean: Tensor, log_var: Tensor):
        self.member_mean, self.member_log_var = mean, log_var                       # [M, B, K]

    @property
    def epistemic_var(self) -> Tensor:
        mm = self.member_mean
        return mm.var(0, unbiased=True) if mm.shape[0] > 1 else torch.zeros_like(mm[0])

    @property
    def aleatoric_var(self) -> Tensor:
        return torch.exp(self.member_log_var).mean(0)

    def information_gain(self) -> Tensor:
        """[B] nats: 0.5 sum_d log(1 + epistemic_d / aleatoric_d)."""
        return 0.5 * torch.log1p(self.epistemic_var / self.aleatoric_var.clamp_min(1e-8)).sum(-1)

    def member_nll(self, y: Tensor) -> Tensor:
        """[M, B] Gaussian NLL of y [B, K] under each member (summed over dimensions)."""
        lv = self.member_log_var
        return 0.5 * (((y.unsqueeze(0) - self.member_mean) ** 2) * torch.exp(-lv) + lv + math.log(2 * math.pi)).sum(-1)

    def predictive_nll(self, y: Tensor) -> Tensor:
        """[B] NLL of y under the members' mixture (the model's predictive distribution of the content)."""
        m = self.member_mean.shape[0]
        return -(torch.logsumexp(-self.member_nll(y), 0) - math.log(m))


class ContentHead(nn.Module):
    """z [..., D] -> distribution over the content vector y [..., K] (ensemble with zero-mean randomised priors)."""

    def __init__(self, d_model: int, k: int = CONTENT_DIM, hidden: int = 128, n_members: int = 5,
                 prior_scale: float = 1.0, seed: int = 0, min_log_var: float = -8.0, max_log_var: float = 4.0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) + 7)
        self.k, self.n_members, self.prior_scale = int(k), int(n_members), float(prior_scale)
        self.min_log_var, self.max_log_var = min_log_var, max_log_var
        self.inp = nn.Linear(d_model, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.members = _EnsembleMLP(n_members, hidden, hidden, 2 * k, out_std=0.1 / math.sqrt(hidden), generator=g)
        gp = torch.Generator().manual_seed(int(seed) * 7919 + 15485863)
        self.register_buffer("prior_w1", torch.randn(n_members, d_model, 64, generator=gp) / math.sqrt(d_model),
                             persistent=False)
        self.register_buffer("prior_w2", torch.randn(n_members, 64, k, generator=gp) / 8.0, persistent=False)

    def forward(self, z: Tensor) -> ContentOutput:
        lead = z.shape[:-1]
        x = F.silu(self.norm(self.inp(z)))
        o = self.members(x.unsqueeze(0).expand((self.n_members,) + x.shape))
        mean, log_var = o[..., :self.k], o[..., self.k:]
        if self.prior_scale > 0 and self.n_members > 1:
            zf = z.detach().reshape(1, -1, z.shape[-1]).expand(self.n_members, -1, -1).to(self.prior_w1.dtype)
            p = torch.bmm(torch.tanh(torch.bmm(zf, self.prior_w1)), self.prior_w2)
            p = (p - p.mean(0, keepdim=True)) * self.prior_scale
            mean = mean + p.reshape((self.n_members,) + lead + (self.k,)).to(mean.dtype)
        return ContentOutput(mean.to(F64), log_var.clamp(self.min_log_var, self.max_log_var).to(F64))


def ensure_content_head(model: Any, seed: int = 0) -> ContentHead:
    """The model's content head, registered (group heads) on first use. Older checkpoints have no weights for it:
    the non-strict restore reports them missing and it starts from its initialisation."""
    try:
        return model.head(HEAD_CONTENT)
    except (KeyError, AttributeError):
        pass
    from ..nn.api import GROUP_HEADS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3_000_017 + int(seed))
        head = ContentHead(model.d_model, seed=seed)
    return model.register_head(HEAD_CONTENT, head, GROUP_HEADS)


def content_target_for(model: Any, seed: int = 0) -> ContentTarget:
    t = getattr(model, "_content_target", None)
    if not isinstance(t, ContentTarget):
        d_sem = int(getattr(model, "d_sem", 0) or model.embed_texts(["x"]).shape[-1])
        t = ContentTarget(d_sem, seed=seed)
        object.__setattr__(model, "_content_target", t)
    return t


@torch.no_grad()
def record_content_nll(model: Any, store: ExperienceStore, records: Sequence[ExperienceRecord]
                       ) -> List[Optional[float]]:
    """Per record: the predictive NLL (nats) of the content its action returned under the current weights, from the
    record's own decision context; None without returned items."""
    from ..learning_api import encode_decisions
    recs = list(records)
    groups = [returned_items(store, r) for r in recs]
    keep = [i for i, g in enumerate(groups) if g]
    out: List[Optional[float]] = [None] * len(recs)
    if not keep:
        return out
    head = ensure_content_head(model)
    y, mask = content_target_for(model)(model, [groups[i] for i in keep])
    batch = encode_decisions(model, store, [recs[i] for i in keep], grad=False)
    nll = head(batch.z_chosen).predictive_nll(y.to(batch.z_chosen.device))
    for j, i in enumerate(keep):
        if bool(mask[j]):
            out[i] = float(nll[j])
    return out


class ContentLoss:
    """Loss provider: content NLL of executed decisions (bootstrap weights per member), frequent timescale."""
    name = "content"
    weight = 0.25
    batch_kwargs: Dict[str, Any] = {"with_next": False, "with_grad_semantic": False}

    def __init__(self, heads: Any, weight: Optional[float] = None, allowed_provenance: Optional[Sequence[str]] = None):
        from ..learning_api import TIMESCALE_FREQUENT
        self.timescale = TIMESCALE_FREQUENT
        self.heads = heads
        if weight is not None:
            self.weight = float(weight)
        self.allowed_provenance = tuple(allowed_provenance or heads.cfg.allowed_provenance)

    def eligible(self, record: ExperienceRecord) -> bool:
        return record.provenance in self.allowed_provenance and record.result is not None and \
            bool(record.post_item_refs)

    def loss(self, model: Any, batch: Any, store: ExperienceStore) -> Tuple[Optional[Tensor], Dict[str, float]]:
        from .world import bootstrap_weights
        recs = batch.records
        groups = [returned_items(store, r) for r in recs]
        if not any(groups):
            return None, {"content_n": 0.0}
        head = ensure_content_head(model)
        y, mask = content_target_for(model)(model, groups)
        z = batch.z_chosen
        out = head(z)
        dev = out.member_mean.device
        y, mask = y.to(dev), mask.to(dev)
        nll = out.member_nll(y) / head.k                                                 # [M, B] per dimension
        w = bootstrap_weights([r.exp_id for r in recs], head.n_members, 29).to(dev, nll.dtype)
        w = w.reshape(head.n_members, -1) * mask.to(nll.dtype)[None]
        loss = (nll * w).sum() / w.sum().clamp_min(1e-12)
        with torch.no_grad():
            pnll = out.predictive_nll(y)
            metrics = {"content_nll": float(loss.detach()), "content_n": float(mask.sum()),
                       "content_predictive_nll": float(pnll[mask].mean()) if bool(mask.any()) else float("nan"),
                       "content_ig": float(out.information_gain()[mask].mean()) if bool(mask.any()) else 0.0}
        tot = loss.to(z.dtype)
        return tot / max(1.0, abs(float(tot.detach()))), metrics


# ----------------------------------------------------------------------------------------------------------------
# debug-scale diagnostic: python -m cam_live.predict.content --out <json>
# ----------------------------------------------------------------------------------------------------------------
def _realised_iv(model: Any, store: ExperienceStore, lifetimes: Sequence[Sequence[ExperienceRecord]]
                 ) -> Dict[str, List[float]]:
    """Per operation of decision t: content NLL of decision t+1's returned content with decision t's returned items
    removed from t+1's context minus with them (nats; > 0: what t returned helped predict what t+1 returned)."""
    import dataclasses
    out: Dict[str, List[float]] = {}
    for recs in lifetimes:
        recs = [r for r in recs if r.post_item_refs]
        for a, b in zip(recs, recs[1:]):
            drop = {it.ref for it in returned_items(store, a)}
            if not drop or not returned_items(store, b):
                continue
            cut = dataclasses.replace(b, context_item_refs=[x for x in b.context_item_refs if x not in drop])
            full, part = record_content_nll(model, store, [b])[0], record_content_nll(model, store, [cut])[0]
            if full is not None and part is not None:
                out.setdefault(a.decision.chosen.operation, []).append(part - full)
    return out


def diagnose(seed: int = 0, n_lifetimes: int = 8, n_decisions: int = 12, steps: int = 300, batch: int = 16,
             lr: float = 2e-3) -> Dict[str, Any]:
    """Train the content head (and the decision trunk through it) on LabWorld lifetimes collected with a debug-scale
    model, and measure on held-out lifetimes, before and after: predictive NLL of the returned content per
    operation, information gain per operation, and realised information value per operation."""
    import random as _random
    import time as _time
    from ..diagnostics.memstore import MemoryStore
    from ..learning_api import encode_decisions
    from . import labdata as L
    t0 = _time.time()
    torch.manual_seed(seed)
    model, heads = L.build(seed)
    store = MemoryStore()
    lts = L.collect(store, range(n_lifetimes), n_decisions)
    n_tr = max(1, int(0.75 * len(lts)))
    train = [r for lt in lts[:n_tr] for r in lt.records if r.post_item_refs]
    held = [[r for r in lt.records if r.post_item_refs] for lt in lts[n_tr:]]
    held_flat = [r for lt in held for r in lt]
    head = ensure_content_head(model, seed)

    def measure() -> Dict[str, Any]:
        nll = record_content_nll(model, store, held_flat)
        with torch.no_grad():
            bt = encode_decisions(model, store, held_flat, grad=False)
            ig = head(bt.z_chosen).information_gain().cpu()
        by: Dict[str, Dict[str, List[float]]] = {}
        for r, v, g in zip(held_flat, nll, ig):
            d = by.setdefault(r.decision.chosen.operation, {"nll": [], "ig": []})
            if v is not None:
                d["nll"].append(v)
            d["ig"].append(float(g))
        iv = _realised_iv(model, store, held)
        avg = lambda xs: sum(xs) / len(xs) if xs else None                              # noqa: E731
        return {op: {"n": len(d["ig"]), "content_nll": avg(d["nll"]), "information_gain": avg(d["ig"]),
                     "realised_iv_next": avg(iv.get(op, [])), "n_iv": len(iv.get(op, []))} for op, d in by.items()}

    before = measure()
    loss_fn = ContentLoss(heads)
    params = list(head.parameters()) + [p for n, p in model.named_parameters()
                                        if p.requires_grad and n.startswith("exp.")]
    opt = torch.optim.Adam(params, lr=lr)
    rng = _random.Random(seed)
    trace = []
    for step in range(steps):
        b = encode_decisions(model, store, rng.sample(train, min(batch, len(train))), grad=True)
        loss, m = loss_fn.loss(model, b, store)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            trace.append({"step": step, **{k: m[k] for k in ("content_predictive_nll", "content_ig") if k in m}})
    after = measure()
    return {"seed": seed, "n_lifetimes": n_lifetimes, "n_decisions": n_decisions, "steps": steps, "train_records":
            len(train), "held_out_records": len(held_flat), "before": before, "after": after, "trace": trace,
            "seconds": _time.time() - t0, "provenance": "diagnostic (LabWorld, debug-scale model)"}


def diagnose_prequential(seed: int = 0, n_lifetimes: int = 10, n_decisions: int = 12, steps_per_record: int = 4,
                         batch: int = 8, lr: float = 1e-3) -> Dict[str, Any]:
    """The learner's regime: records arrive in time order; for each new record its content NLL and information gain
    are measured *before* it is trained on, then a few steps on a replay batch that includes it. Per operation:
    the NLL over time (does the content become predictable?), the IG over time, and whether the IG predicted before
    observing a record tracks the actual NLL reduction on that operation's next record (Spearman rank correlation)."""
    import random as _random
    import time as _time
    from ..diagnostics.memstore import MemoryStore
    from ..learning_api import encode_decisions
    from . import labdata as L
    t0 = _time.time()
    torch.manual_seed(seed)
    model, heads = L.build(seed)
    store = MemoryStore()
    lts = L.collect(store, range(n_lifetimes), n_decisions)
    stream = [r for lt in lts for r in lt.records if r.post_item_refs and returned_items(store, r)]
    head = ensure_content_head(model, seed)
    loss_fn = ContentLoss(heads)
    params = list(head.parameters()) + [p for n, p in model.named_parameters() if p.requires_grad and
                                        n.startswith("exp.")]
    opt = torch.optim.Adam(params, lr=lr)
    rng = _random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen: List[ExperienceRecord] = []
    for t, rec in enumerate(stream):
        with torch.no_grad():
            bt = encode_decisions(model, store, [rec], grad=False)
            ig = float(head(bt.z_chosen).information_gain()[0])
        nll = record_content_nll(model, store, [rec])[0]
        rows.append({"t": t, "op": rec.decision.chosen.operation, "nll": nll, "ig": ig})
        seen.append(rec)
        for _ in range(steps_per_record):
            b = encode_decisions(model, store, [rec] + rng.sample(seen, min(batch - 1, len(seen))), grad=True)
            loss, _ = loss_fn.loss(model, b, store)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    # IG before observing record i vs the drop of NLL from op's occurrence i to its next occurrence
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["op"], []).append(r)
    pairs = []
    for op, rs in by.items():
        for a, b in zip(rs, rs[1:]):
            if a["nll"] is not None and b["nll"] is not None:
                pairs.append((a["ig"], a["nll"] - b["nll"], op))

    def rank(xs: Sequence[float]) -> List[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r

    def spearman(ps: Sequence[Tuple[float, float, str]]) -> Optional[float]:
        if len(ps) < 3:
            return None
        a, b = rank([p[0] for p in ps]), rank([p[1] for p in ps])
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den if den > 0 else None
    third = max(1, len(rows) // 3)
    summary = {}
    for op, rs in by.items():
        first, last = rs[:max(1, len(rs) // 3)], rs[-max(1, len(rs) // 3):]
        avg = lambda xs: sum(xs) / len(xs) if xs else None                               # noqa: E731
        summary[op] = {"n": len(rs), "nll_first_third": avg([r["nll"] for r in first if r["nll"] is not None]),
                       "nll_last_third": avg([r["nll"] for r in last if r["nll"] is not None]),
                       "ig_first_third": avg([r["ig"] for r in first]), "ig_last_third": avg([r["ig"] for r in last]),
                       "spearman_ig_vs_next_nll_drop": spearman([p for p in pairs if p[2] == op])}
    return {"seed": seed, "records": len(stream), "steps_per_record": steps_per_record, "per_operation": summary,
            "spearman_ig_vs_next_nll_drop_all": spearman(pairs), "n_pairs": len(pairs), "rows": rows,
            "stream_third": third, "seconds": _time.time() - t0, "provenance": "diagnostic (LabWorld, debug scale)"}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m cam_live.predict.content")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lifetimes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prequential", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.prequential:
        res = diagnose_prequential(a.seed, a.lifetimes)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        raise SystemExit(0)
    res = diagnose(a.seed, a.lifetimes, steps=a.steps)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"before": res["before"], "after": res["after"]}, indent=1))
