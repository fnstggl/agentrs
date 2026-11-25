"""Combined staged optimisation of the whole CAM Unified model (spec 12.5).

``L_total = sum_i w_i L_i`` over the objectives the active stage enables (weights from ``CamConfig.loss_weights``),
one AdamW optimiser with separate parameter groups (core backbone / decision & world heads / semantic bridge + LoRA
at ``cfg.lr_sem``), bf16 autocast, gradient clipping, and — every ``grad_probe_every`` updates — the gradient each
weighted objective sends into the shared backbone (``autograd.grad`` of ``w_i L_i`` with respect to the backbone
parameters), their norms, shares and pairwise cosines, so a dominating or conflicting objective is visible. Every
update gets an id that is appended to the training provenance of the records it consumed.

Stages (spec 12.5; cumulative — later stages keep the earlier objectives, so no component is dropped):
  1 grounding (semantic, action) · 2 world · 3 adapt · 4 actor_critic (actor, value) · 5 explore_meta (meta; LP reward
  on) · 6 imagination (imagined actor-critic on calibrated rollouts) · 7 continual (memory, retention).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    objectives: frozenset
    flags: tuple = ()          # (("lp_reward", True), ...)

    def flag(self, name: str, default=False):
        return dict(self.flags).get(name, default)


def _stages() -> tuple:
    spec = [("grounding", {"semantic", "action"}, {}),
            ("world", {"world"}, {}),
            ("adapt", {"adapt"}, {}),
            ("actor_critic", {"actor", "value"}, {}),
            ("explore_meta", {"meta"}, {"lp_reward": True}),
            ("imagination", {"imagined"}, {"imagination": True}),
            ("continual", {"memory", "retention"}, {"consolidation": True})]
    out, objs, flags = [], set(), {}
    for i, (name, o, f) in enumerate(spec, start=1):
        objs |= o
        flags.update(f)
        out.append(Stage(i, name, frozenset(objs), tuple(sorted(flags.items()))))
    return tuple(out)


STAGES = _stages()
STAGE_BY_NAME = {s.name: s for s in STAGES}
DEFAULT_EXTRA_WEIGHTS = {"imagined": 0.5}


@dataclass
class UpdateInfo:
    update_id: str
    n: int
    stage: str
    losses: dict                         # raw objective values (float)
    weighted: dict                       # w_i * L_i (float)
    total: float
    grad_norm: float                     # pre-clip global norm
    group_grad_norm: dict
    clipped: bool
    backbone: Optional[dict] = None      # {"norm": {obj: float}, "share": {...}, "cosine": {"a|b": float}, "dominant": str|None}
    disabled: list = field(default_factory=list)
    skipped: bool = False
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _params(x) -> list:
    if x is None:
        return []
    if isinstance(x, nn.Module):
        return [p for p in x.parameters() if p.requires_grad]
    return [p for p in x if p.requires_grad]


class CombinedTrainer:
    """Optimise all trainable parts of the model with the staged combined objective.

    Args:
      cfg: `CamConfig` (lr, lr_sem, weight_decay, grad_clip, loss_weights; ``value_coef`` must equal
        ``loss_weights["value"]`` unless the trainer's ``loss_weights`` sets "value" explicitly).
      core / heads / semantic: modules or parameter iterables for the three optimiser groups (disjoint).
      backbone: parameters whose per-objective gradient contributions are measured (default: ``core``).
      stage: index 1..7 or stage name; ``stage_schedule``: [(update_count, stage)] automatic advancement.
      grad_probe_every: measure backbone contributions every N updates (0 = never). Norms are exact; pairwise
        cosines are exact up to ``cos_exact_numel`` backbone parameters and otherwise estimated on a fixed random
        sample of ``cos_sample`` coordinates (memory O(n_objectives * cos_sample) instead of the full backbone).
      on_nonfinite: "skip" (no step, counted, logged) or "raise".
    """

    def __init__(self, cfg, core=None, heads=None, semantic=None, backbone=None, stage=1, loss_weights=None,
                 extra_loss_weights=None, amp: bool = True, device=None, grad_probe_every: int = 10,
                 on_nonfinite: str = "skip", stage_schedule=None, log_path: Optional[str] = None,
                 history: int = 10_000, betas=(0.9, 0.95), lr_warmup: int = 0, dominance: float = 0.9,
                 cos_exact_numel: int = 20_000_000, cos_sample: int = 2_000_000):
        self.cfg = cfg
        groups = {"core": _params(core), "heads": _params(heads), "semantic": _params(semantic)}
        seen: dict = {}
        for g, ps in groups.items():
            for p in ps:
                if id(p) in seen:
                    raise ValueError(f"parameter in two optimiser groups ({seen[id(p)]} and {g})")
                seen[id(p)] = g
        if not seen:
            raise ValueError("no trainable parameters")
        self.groups = groups
        lr = {"core": cfg.lr, "heads": cfg.lr, "semantic": cfg.lr_sem}
        pg = []
        for g, ps in groups.items():
            if not ps:
                continue
            dec = [p for p in ps if p.dim() >= 2]
            nod = [p for p in ps if p.dim() < 2]
            if dec:
                pg.append({"params": dec, "lr": lr[g], "weight_decay": cfg.weight_decay, "name": f"{g}/decay",
                           "base_lr": lr[g]})
            if nod:
                pg.append({"params": nod, "lr": lr[g], "weight_decay": 0.0, "name": f"{g}/no_decay",
                           "base_lr": lr[g]})
        self._pg_spec = pg
        self.betas = betas
        self.opt = torch.optim.AdamW(pg, betas=betas)
        self.backbone = _params(backbone) if backbone is not None else list(groups["core"])
        # the critic weight appears twice in CamConfig (value_coef and loss_weights["value"]); refuse an ambiguous
        # config instead of silently applying one of them
        vc = getattr(cfg, "value_coef", None)
        if vc is not None and "value" in cfg.loss_weights and not (loss_weights and "value" in loss_weights) \
                and abs(float(vc) - float(cfg.loss_weights["value"])) > 1e-12:
            raise ValueError(f"cfg.value_coef={vc} disagrees with cfg.loss_weights['value']="
                             f"{cfg.loss_weights['value']}; set both (or pass loss_weights={{'value': w}})")
        self.weights = dict(cfg.loss_weights)
        self.weights.update(DEFAULT_EXTRA_WEIGHTS if extra_loss_weights is None else extra_loss_weights)
        if loss_weights:
            self.weights.update(loss_weights)
        self.amp = amp
        dev = device if device is not None else self._all_params()[0].device
        self.device_type = torch.device(dev).type
        self.grad_probe_every = int(grad_probe_every)
        if on_nonfinite not in ("skip", "raise"):
            raise ValueError("on_nonfinite must be 'skip' or 'raise'")
        self.on_nonfinite = on_nonfinite
        self.stage = self._resolve(stage)
        self.stage_schedule = sorted([(int(n), self._resolve(s)) for n, s in (stage_schedule or [])],
                                     key=lambda x: x[0])
        self.n_updates = 0
        self.n_skipped = 0
        self.lr_warmup = int(lr_warmup)
        self.dominance = float(dominance)
        self.cos_exact_numel, self.cos_sample, self._cos_idx = int(cos_exact_numel), int(cos_sample), None
        self.history: deque = deque(maxlen=history)
        self.log_path = log_path
        self.run_id = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------ stages
    @staticmethod
    def _resolve(stage) -> Stage:
        if isinstance(stage, Stage):
            return stage
        if isinstance(stage, str):
            return STAGE_BY_NAME[stage]
        i = int(stage)
        if not 1 <= i <= len(STAGES):
            raise ValueError(f"stage must be 1..{len(STAGES)}")
        return STAGES[i - 1]

    def set_stage(self, stage) -> Stage:
        self.stage = self._resolve(stage)
        return self.stage

    def enabled(self, objective: str) -> bool:
        return objective in self.stage.objectives

    def flag(self, name: str) -> bool:
        return bool(self.stage.flag(name))

    def autocast(self):
        """bf16 autocast for the forward pass that produces the losses (no-op when disabled/unsupported)."""
        if not self.amp:
            return contextlib.nullcontext()
        if self.device_type == "cuda" and not torch.cuda.is_bf16_supported():
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ the update
    def _all_params(self) -> list:
        return [p for ps in self.groups.values() for p in ps]

    def _sample_index(self) -> list:
        """Fixed random coordinates per backbone tensor (proportional to its size) for cosine estimates on large
        backbones; norms are always exact."""
        if self._cos_idx is None:
            g = torch.Generator().manual_seed(1234)
            tot = sum(p.numel() for p in self.backbone)
            idx = []
            for p in self.backbone:
                k = min(p.numel(), max(1, int(round(self.cos_sample * p.numel() / tot))))
                idx.append(torch.randperm(p.numel(), generator=g)[:k].to(p.device))
            self._cos_idx = idx
        return self._cos_idx

    def _backbone_probe(self, weighted: dict) -> dict:
        names = list(weighted)
        exact = sum(p.numel() for p in self.backbone) <= self.cos_exact_numel
        idx = None if exact else self._sample_index()
        vec, norms = {}, {}
        for n in names:
            if weighted[n].requires_grad:
                gs = torch.autograd.grad(weighted[n], self.backbone, retain_graph=True, allow_unused=True)
            else:
                gs = [None] * len(self.backbone)
            sq = torch.zeros((), device=self.backbone[0].device)
            parts = []
            for i, (g, p) in enumerate(zip(gs, self.backbone)):
                k = p.numel() if exact else idx[i].numel()
                if g is None:
                    parts.append(torch.zeros(k, device=p.device))
                    continue
                gf = g.detach().reshape(-1).float()
                sq = sq + (gf * gf).sum()
                parts.append(gf if exact else gf[idx[i]])
            norms[n] = float(sq.sqrt())
            vec[n] = torch.cat(parts)
        tot = sum(norms.values())
        share = {n: (norms[n] / tot if tot > 0 else 0.0) for n in names}
        cos = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                den = vec[a].norm() * vec[b].norm()
                cos[f"{a}|{b}"] = float((vec[a] @ vec[b]) / den) if float(den) > 0 else 0.0
        dom = max(share, key=share.get) if tot > 0 and max(share.values()) > self.dominance and len(names) > 1 else None
        return {"norm": norms, "share": share, "cosine": cos, "dominant": dom, "cosine_exact": exact}

    def step(self, losses: dict, consumed: Optional[Iterable] = None, probe: Optional[bool] = None) -> UpdateInfo:
        """One optimiser update from the objective tensors in ``losses`` (keys from the loss-weight table).
        Objectives the active stage disables are not optimised (listed in ``disabled``)."""
        t0 = time.time()
        for n_, st in self.stage_schedule:
            if self.n_updates >= n_ and st.index > self.stage.index:
                self.stage = st
        unknown = [k for k in losses if k not in self.weights]
        if unknown:
            raise KeyError(f"objectives without a loss weight: {unknown}")
        active = {k: v for k, v in losses.items() if self.enabled(k)}
        disabled = sorted(k for k in losses if not self.enabled(k))
        if not active:
            raise ValueError(f"no objective enabled in stage {self.stage.name}: got {sorted(losses)}")
        weighted = {k: self.weights[k] * v.float() for k, v in active.items() if self.weights[k] != 0}
        raw = {k: float(v.detach().float()) for k, v in active.items()}
        total = sum(weighted.values()) if weighted else None
        if total is None:
            raise ValueError("all enabled objectives have weight 0")
        self.n_updates += 1
        uid = f"upd-{self.run_id}-{self.n_updates:08d}"
        finite = math.isfinite(float(total.detach()))
        if not finite:
            self.opt.zero_grad(set_to_none=True)
            if self.on_nonfinite == "raise":
                raise FloatingPointError(f"non-finite loss {raw}")
            self.n_skipped += 1
            info = UpdateInfo(uid, self.n_updates, self.stage.name, raw, {k: float(v.detach()) for k, v in
                                                                            weighted.items()}, float("nan"),
                              float("nan"), {}, False, None, disabled, True, time.time() - t0)
            self._log(info)
            return info
        do_probe = (self.grad_probe_every > 0 and (self.n_updates - 1) % self.grad_probe_every == 0) \
            if probe is None else probe
        bb = self._backbone_probe(weighted) if (do_probe and self.backbone) else None
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = {}
        for g, ps in self.groups.items():
            gs = [p.grad.detach().float().norm() for p in ps if p.grad is not None]
            gnorm[g] = float(torch.stack(gs).norm()) if gs else 0.0
        params = self._all_params()
        gn = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip))
        if not math.isfinite(gn):
            self.opt.zero_grad(set_to_none=True)
            if self.on_nonfinite == "raise":
                raise FloatingPointError("non-finite gradient")
            self.n_skipped += 1
            info = UpdateInfo(uid, self.n_updates, self.stage.name, raw, {k: float(v.detach()) for k, v in
                                                                            weighted.items()}, float(total.detach()),
                              gn, gnorm, False, bb, disabled, True, time.time() - t0)
            self._log(info)
            return info
        if self.lr_warmup > 0:
            f = min(1.0, self.n_updates / self.lr_warmup)
            for grp in self.opt.param_groups:
                grp["lr"] = grp["base_lr"] * f
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        info = UpdateInfo(uid, self.n_updates, self.stage.name, raw,
                          {k: float(v.detach()) for k, v in weighted.items()}, float(total.detach()), gn, gnorm,
                          gn > self.cfg.grad_clip, bb, disabled, False, time.time() - t0)
        if consumed is not None:
            attach_provenance(consumed, uid)
        self._log(info)
        return info

    def _log(self, info: UpdateInfo) -> None:
        self.history.append(info)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(info.to_dict(), default=float) + "\n")

    # ------------------------------------------------------------------ state
    def reset_optimizer(self) -> None:
        """Fresh optimiser moments (spec 11.5 / 13.7 condition B vs C), same groups and learning rates."""
        self.opt = torch.optim.AdamW([{"params": list(g["params"]), "lr": g["base_lr"],
                                       "weight_decay": g["weight_decay"], "name": g["name"], "base_lr": g["base_lr"]}
                                      for g in self._pg_spec], betas=self.betas)

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "n_updates": self.n_updates, "n_skipped": self.n_skipped,
                "stage": self.stage.index, "weights": dict(self.weights), "run_id": self.run_id}

    def load_state_dict(self, s: dict) -> None:
        self.opt.load_state_dict(s["opt"])
        self.n_updates, self.n_skipped = int(s["n_updates"]), int(s.get("n_skipped", 0))
        self.stage = self._resolve(int(s["stage"]))
        self.weights.update(s.get("weights", {}))


def attach_provenance(records: Iterable, update_id: str) -> int:
    """Append ``update_id`` to the training provenance of consumed records: `EventRecord.training_provenance
    ["used_in_updates"]` and `DecisionTrace.update_refs` (spec 18.1). Returns the number of records marked."""
    n = 0
    for r in records:
        if hasattr(r, "update_refs"):
            r.update_refs.append(update_id)
            n += 1
        elif hasattr(r, "training_provenance"):
            r.training_provenance.setdefault("used_in_updates", []).append(update_id)
            n += 1
        else:
            raise TypeError(f"cannot attach provenance to {type(r).__name__}")
    return n
