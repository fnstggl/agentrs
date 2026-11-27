"""cam_live.credit.retention — retention of earlier capabilities: anchors, functional regularisation, measured
forgetting, and the metrics that gate checkpoint promotion (spec 10.3; CL-04, CL-05).

* :class:`AnchorSet` keeps a sample of earlier-objective experience (record ids, by objective family) together with
  the *outputs the model produced on it at the time* (reference outputs of named functions: latents, predictions,
  action probabilities).
* :class:`RetentionProvider` (``LossProvider`` "retention") penalises the change of those outputs on a few anchors
  per update: squared error relative to the reference scale for real-valued functions, KL(reference || current)
  for probability vectors. Together with the store's family-balanced replay this is the retention mechanism.
* :func:`evaluate_families` / :func:`measure_forgetting` measure — on records that were never trained on — how
  prediction NLL, calibration and the decisions themselves changed for each earlier family after learning
  something new. Forgetting is reported, never assumed away.
* :func:`retention_metrics` / :func:`validation_metrics` orient those numbers "higher = better" for
  ``CheckpointManager.promote``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..contracts import ExperienceRecord
from ..learning_api import TIMESCALE_FREQUENT, DecisionBatch, encode_decisions
from ..nn.api import LatentModelAPI

FnOut = Union[Tensor, Tuple[Tensor, Tensor]]
OutputFn = Callable[[LatentModelAPI, DecisionBatch, Any], FnOut]
EvalFn = Callable[[LatentModelAPI, DecisionBatch, Any], Dict[str, Tensor]]


@dataclass
class FunctionSpec:
    """One function of the model whose value on the anchors is to be preserved.

    fn      (model, batch, store) -> values [B, M] or (values [B, M], mask [B, M]); differentiable
    kind    "mse": squared change relative to the mean square of the reference; "kl": values are probabilities
    """
    name: str
    fn: OutputFn
    kind: str = "mse"
    weight: float = 1.0


def latent_functions() -> List[FunctionSpec]:
    """Always available: the belief and the decision latent of the executed candidate."""
    return [FunctionSpec("latent_b", lambda m, b, s: b.b), FunctionSpec("latent_z", lambda m, b, s: b.z_chosen)]


def _split(out: FnOut) -> Tuple[Tensor, Tensor]:
    v, m = out if isinstance(out, tuple) else (out, None)
    v = v.reshape(v.shape[0], -1)
    m = torch.ones_like(v, dtype=torch.bool) if m is None else m.reshape(v.shape[0], -1).bool()
    return v, m


class AnchorSet:
    """Earlier-objective experience with stored reference outputs. Anchors are grouped by objective family."""

    def __init__(self, functions: Optional[Sequence[FunctionSpec]] = None, *, max_per_family: int = 256):
        self.functions: List[FunctionSpec] = list(functions) if functions is not None else latent_functions()
        self.max_per_family = int(max_per_family)
        self.family_of: Dict[str, str] = {}                     # exp_id -> family
        self.refs: Dict[str, Dict[str, Tensor]] = {}            # exp_id -> function -> reference values (valid entries)
        self.scale: Dict[str, float] = {}                       # function -> mean square of its reference values
        self._n: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def families(self) -> List[str]:
        return sorted(set(self.family_of.values()))

    def ids(self, family: Optional[str] = None) -> List[str]:
        return [e for e, f in self.family_of.items() if family is None or f == family]

    @torch.no_grad()
    def add(self, model: LatentModelAPI, store: Any, records: Sequence[ExperienceRecord], *, family: Optional[str] = None,
            batch_size: int = 32) -> int:
        """Store the current model's outputs on ``records`` as references (records already anchored keep their
        reference: the anchor says what the model did *then*). Returns the number of anchors added."""
        fresh = [r for r in records if r.exp_id not in self.refs]
        room: Dict[str, int] = {}
        keep = []
        for r in fresh:
            fam = family or r.objective_family or "default"
            left = room.setdefault(fam, self.max_per_family - len(self.ids(fam)))
            if left > 0:
                keep.append((r, fam))
                room[fam] = left - 1
        was = model.training
        model.eval()
        try:
            for i in range(0, len(keep), batch_size):
                chunk = keep[i:i + batch_size]
                batch = encode_decisions(model, store, [r for r, _ in chunk], grad=False)
                outs = {f.name: _split(f.fn(model, batch, store)) for f in self.functions}
                for j, (r, fam) in enumerate(chunk):
                    self.family_of[r.exp_id] = fam
                    self.refs[r.exp_id] = {n: v[j][m[j]].detach().float().cpu().clone() for n, (v, m) in outs.items()}
                    for n, t in self.refs[r.exp_id].items():
                        if t.numel():
                            k = self._n.get(n, 0)
                            ms = float(t.pow(2).mean())
                            self.scale[n] = (self.scale.get(n, 0.0) * k + ms) / (k + 1)
                            self._n[n] = k + 1
        finally:
            model.train(was)
        return len(keep)

    def penalty(self, model: LatentModelAPI, store: Any, exp_ids: Sequence[str]) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """Functional distance between the current outputs and the references on the given anchors."""
        recs = [r for r in store.get_many(list(exp_ids)) if r.exp_id in self.refs]
        if not recs:
            return None, {"n_anchors": 0.0}
        batch = encode_decisions(model, store, recs)
        total: Optional[Tensor] = None
        metrics: Dict[str, float] = {"n_anchors": float(len(recs))}
        for f in self.functions:
            v, m = _split(f.fn(model, batch, store))
            terms = []
            for j, r in enumerate(recs):
                ref = self.refs[r.exp_id].get(f.name)
                cur = v[j][m[j]]
                if ref is None or ref.numel() == 0 or ref.numel() != cur.numel():
                    continue
                ref = ref.to(cur.device, cur.dtype)
                if f.kind == "kl":
                    terms.append((ref * (ref.clamp_min(1e-12).log() - cur.clamp_min(1e-12).log())).sum())
                else:
                    terms.append((cur - ref).pow(2).mean() / max(self.scale.get(f.name, 1.0), 1e-8))
            if not terms:
                continue
            d = torch.stack(terms).mean()
            metrics[f.name] = float(d.detach())
            total = f.weight * d if total is None else total + f.weight * d
        return total, metrics

    def state_dict(self) -> Dict[str, Any]:
        return {"family_of": dict(self.family_of), "refs": {e: dict(d) for e, d in self.refs.items()},
                "scale": dict(self.scale), "n": dict(self._n), "functions": [f.name for f in self.functions]}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.family_of = dict(st.get("family_of", {}))
        self.refs = {e: dict(d) for e, d in st.get("refs", {}).items()}
        self.scale = {k: float(v) for k, v in st.get("scale", {}).items()}
        self._n = {k: int(v) for k, v in st.get("n", {}).items()}


class RetentionProvider:
    """``LossProvider``: functional regularisation on a few anchors per update, drawn evenly over the anchored
    families. It reads its anchors, not the replay batch (``uses_batch = False``)."""
    timescale = TIMESCALE_FREQUENT
    uses_batch = False
    adapter_grad = False
    needs_latents = False

    def __init__(self, anchors: AnchorSet, *, weight: float = 1.0, n_anchors: int = 8, name: str = "retention", seed: int = 0):
        self.anchors, self.weight, self.n_anchors, self.name = anchors, float(weight), int(n_anchors), name
        self._gen = torch.Generator().manual_seed(9_000_011 + int(seed))

    def eligible(self, record: ExperienceRecord) -> bool:
        return False                     # it never asks for replay records of its own

    def draw(self) -> List[str]:
        fams = self.anchors.families()
        if not fams:
            return []
        per = max(1, self.n_anchors // len(fams))
        out: List[str] = []
        for f in fams:
            ids = self.anchors.ids(f)
            take = torch.randperm(len(ids), generator=self._gen)[:per].tolist()
            out += [ids[i] for i in take]
        return out[:max(self.n_anchors, len(fams))]

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Optional[Tensor], Dict[str, float]]:
        ids = self.draw()
        if not ids:
            return None, {"n_anchors": 0.0}
        return self.anchors.penalty(model, store, ids)

    def state_dict(self) -> Dict[str, Any]:
        return {"anchors": self.anchors.state_dict(), "generator": self._gen.get_state()}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.anchors.load_state_dict(st.get("anchors", {}))
        if st.get("generator") is not None:
            self._gen.set_state(st["generator"])


# ------------------------------------------------------------------------------------------------ measured forgetting
@dataclass
class FamilyEval:
    """Per-record evaluation of one family at one moment (CPU tensors, aligned with ``exp_ids``)."""
    family: str
    exp_ids: List[str]
    nll: Tensor                  # [n] prediction NLL of the realised outcomes (NaN: the record has no matured outcome)
    z: Tensor                    # [n] standardised residual (y - mean) / std (NaN: none)
    action_probs: List[Tensor] = field(default_factory=list)     # per record, over its recorded candidates

    def summary(self) -> Dict[str, float]:
        nll = self.nll[torch.isfinite(self.nll)]
        z = self.z[torch.isfinite(self.z)]
        msz = float(z.pow(2).mean()) if z.numel() else float("nan")
        return {"n": float(len(self.exp_ids)), "n_scored": float(nll.numel()),
                "nll": float(nll.mean()) if nll.numel() else float("nan"), "mean_sq_z": msz,
                "calibration_error": abs(math.log(max(msz, 1e-12))) if z.numel() else float("nan"),
                "coverage90": float((z.abs() < 1.6449).float().mean()) if z.numel() else float("nan")}


@torch.no_grad()
def evaluate_families(model: LatentModelAPI, store: Any, families: Dict[str, Sequence[str]], eval_fn: EvalFn, *,
                      batch_size: int = 32) -> Dict[str, FamilyEval]:
    """Evaluate ``eval_fn`` on the given records of each family. ``eval_fn(model, batch, store)`` returns
    ``{"nll": [B], "z": [B], "action_probs": [B, N]}`` (any subset; missing parts are NaN / empty)."""
    out: Dict[str, FamilyEval] = {}
    was = model.training
    model.eval()
    try:
        for fam, ids in families.items():
            recs = store.get_many(list(ids))
            nll, z, probs = [], [], []
            for i in range(0, len(recs), batch_size):
                batch = encode_decisions(model, store, recs[i:i + batch_size], grad=False)
                r = eval_fn(model, batch, store)
                n = len(batch.records)
                nll.append(r["nll"].detach().float().cpu() if "nll" in r else torch.full((n,), float("nan")))
                z.append(r["z"].detach().float().cpu() if "z" in r else torch.full((n,), float("nan")))
                if "action_probs" in r:
                    p = r["action_probs"].detach().float().cpu()
                    k = batch.a_mask.sum(1).tolist()
                    probs += [p[j, :int(k[j])].clone() for j in range(n)]
            out[fam] = FamilyEval(fam, [r.exp_id for r in recs], torch.cat(nll) if nll else torch.zeros(0),
                                  torch.cat(z) if z else torch.zeros(0), probs)
    finally:
        model.train(was)
    return out


def measure_forgetting(model: LatentModelAPI, store: Any, before: Dict[str, FamilyEval], eval_fn: EvalFn, *,
                       batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    """Re-evaluate the records of ``before`` with the current model and report, per earlier family, the change of
    prediction NLL (positive = forgotten), of calibration, and how many decisions would now be taken differently
    (agreement of the most probable candidate, mean total variation between the action distributions)."""
    after = evaluate_families(model, store, {f: e.exp_ids for f, e in before.items()}, eval_fn, batch_size=batch_size)
    rep: Dict[str, Dict[str, float]] = {}
    for fam, b in before.items():
        a = after[fam]
        sb, sa = b.summary(), a.summary()
        both = torch.isfinite(b.nll) & torch.isfinite(a.nll)
        d = (a.nll - b.nll)[both]
        r = {"n": sb["n"], "n_scored": float(both.sum()), "nll_before": sb["nll"], "nll_after": sa["nll"],
             "nll_change": float(d.mean()) if d.numel() else float("nan"),
             "nll_change_se": float(d.std(unbiased=True) / math.sqrt(d.numel())) if d.numel() > 1 else float("nan"),
             "calibration_error_before": sb["calibration_error"], "calibration_error_after": sa["calibration_error"],
             "calibration_change": sa["calibration_error"] - sb["calibration_error"],
             "mean_sq_z_before": sb["mean_sq_z"], "mean_sq_z_after": sa["mean_sq_z"],
             "coverage90_before": sb["coverage90"], "coverage90_after": sa["coverage90"]}
        if b.action_probs and len(b.action_probs) == len(a.action_probs):
            agree = [float(int(torch.argmax(p)) == int(torch.argmax(q))) for p, q in zip(b.action_probs, a.action_probs)
                     if p.numel() and p.shape == q.shape]
            tv = [0.5 * float((p - q).abs().sum()) for p, q in zip(b.action_probs, a.action_probs)
                  if p.numel() and p.shape == q.shape]
            if agree:
                r["decision_agreement"] = sum(agree) / len(agree)
                r["action_tv"] = sum(tv) / len(tv)
        rep[fam] = r
    return rep


def retention_metrics(report: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Retention metrics for ``CheckpointManager.promote`` (higher = better): negated NLL after the update and
    decision agreement, per earlier family. NaN entries are left out."""
    out: Dict[str, float] = {}
    for fam, r in report.items():
        if math.isfinite(r.get("nll_after", float("nan"))):
            out[f"{fam}/neg_nll"] = -r["nll_after"]
        if "decision_agreement" in r:
            out[f"{fam}/decision_agreement"] = r["decision_agreement"]
    return out


def validation_metrics(validation_losses: Dict[str, float]) -> Dict[str, float]:
    """Held-out provider losses (``ContinualLearner.validate``) as higher-is-better validation metrics."""
    return {f"heldout/neg_{k}": -float(v) for k, v in validation_losses.items() if math.isfinite(float(v))}
