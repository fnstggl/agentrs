"""Combined staged optimisation of the whole CAM Unified model (spec 12.5).

``L_total = sum_i w_i L_i`` over the objectives the active stage enables (weights from ``CamConfig.loss_weights``),
one AdamW optimiser with separate parameter groups (core backbone / decision & world heads / semantic bridge + LoRA
at ``cfg.lr_sem``), bf16 autocast, gradient clipping, and — every ``grad_probe_every`` updates — the gradient each
weighted objective sends into the shared backbone (``autograd.grad`` of ``w_i L_i`` with respect to the backbone
parameters), their norms, shares and pairwise cosines, so a dominating or conflicting objective is visible. Every
update gets an id that is appended to the training provenance of the records it consumed.

W7 extension (plan 2026-11-26 §2 "Optimisers"; additive, the three-group constructor behaves as before):
``groups={name: params}`` gives named, disjoint parameter groups (e.g. "core", "heads", "dec_cond", "pred", "psi"),
``group_lr`` their learning rates (default: cfg.lr_core for "core*", cfg.lr_psi for "psi", cfg.lr_heads otherwise, 0
meaning cfg.lr), ``roles={role: [group names]}`` one AdamW per role ("slow" = everything the integrated update trains,
"meta" = psi, stepped only by the outer meta objective; `apply` refuses a slow step while a meta parameter holds a
gradient, so no actor loss can move psi), ``clip_mode="per_group"`` clips each group at ``clip_median_mult`` x the
running median of its own pre-clip norms (instead of one global norm), and a linear learning-rate warmup applies to
the ``warmup_groups`` only (default: the core groups, cfg.lr_warmup_updates). `apply(role)` steps gradients the
caller computed itself (the integrated learner's per-objective accumulation), `step(losses)` keeps its contract.

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


class SplitOptimizer:
    """One role's optimiser as AdamW for the core groups + Adafactor for the satellite groups (cfg.sat_optimizer
    "adafactor"; plan 2026-11-26 §4 G3): the trainer's per-group lr warmup, clipping and zero_grad see the union of
    both optimisers' param groups (the same dicts)."""

    def __init__(self, core, sat):
        self.parts = [o for o in (core, sat) if o is not None]

    @property
    def param_groups(self) -> list:
        return [g for o in self.parts for g in o.param_groups]

    @property
    def state(self) -> dict:
        st: dict = {}
        for o in self.parts:
            st.update(o.state)
        return st

    @property
    def defaults(self) -> dict:
        return self.parts[0].defaults

    def step(self, closure=None):
        for o in self.parts:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.parts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"split": [o.state_dict() for o in self.parts], "kinds": [type(o).__name__ for o in self.parts]}

    def load_state_dict(self, sd: dict) -> None:
        if "split" not in sd or [type(o).__name__ for o in self.parts] != list(sd.get("kinds") or []):
            raise ValueError("optimiser state of a different cfg.sat_optimizer")
        for o, s in zip(self.parts, sd["split"]):
            o.load_state_dict(s)


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
    clip: Optional[dict] = None          # per_group mode: {group: {"norm", "threshold", "clipped"}}
    role: str = "slow"

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
                 cos_exact_numel: int = 20_000_000, cos_sample: int = 2_000_000, *, groups: Optional[dict] = None,
                 group_lr: Optional[dict] = None, roles: Optional[dict] = None, clip_mode: Optional[str] = None,
                 clip_median_mult: Optional[float] = None, clip_window: int = 100, clip_min_history: int = 3,
                 warmup_groups: Optional[Iterable[str]] = None, fused: Optional[bool] = None):
        self.cfg = cfg
        named = groups is not None
        if named:
            if core is not None or heads is not None or semantic is not None:
                raise ValueError("pass either groups={name: params} or core/heads/semantic, not both")
            groups = {str(g): _params(ps) for g, ps in groups.items()}
        else:
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
        self._group_of = seen
        if named:
            lr = {g: self._default_lr(cfg, g) for g in groups}
        else:
            lr = {"core": cfg.lr, "heads": cfg.lr, "semantic": cfg.lr_sem}
        lr.update({str(k): float(v) for k, v in (group_lr or {}).items()})
        self.group_lr = lr
        if roles is None:
            meta = [g for g in groups if g == "psi" and groups[g]]
            roles = {"slow": [g for g in groups if g not in meta]}
            if meta:
                roles["meta"] = meta
        roles = {str(r): [str(g) for g in gs] for r, gs in roles.items()}
        in_role: dict = {}
        for r, gs in roles.items():
            for g in gs:
                if g not in groups:
                    raise KeyError(f"role {r!r} names an unknown group {g!r}")
                if g in in_role:
                    raise ValueError(f"group {g!r} in two roles ({in_role[g]} and {r})")
                in_role[g] = r
        missing = [g for g, ps in groups.items() if ps and g not in in_role]
        if missing:
            raise ValueError(f"groups without a role: {missing}")
        if "slow" not in roles:
            raise ValueError("roles must define 'slow' (the integrated update's optimiser)")
        self.roles = roles
        self.role_of_group = in_role
        self.betas = betas
        self.fused = fused
        self._pg_spec = {r: self._param_groups(cfg, gs) for r, gs in roles.items()}
        self.opts = {r: self._make_opt(pg) for r, pg in self._pg_spec.items() if pg}
        if "slow" not in self.opts:
            raise ValueError("the 'slow' role has no trainable parameters")
        self.opt = self.opts["slow"]
        self.backbone = _params(backbone) if backbone is not None else list(groups.get("core", []))
        self.clip_mode = str(clip_mode if clip_mode is not None else getattr(cfg, "clip_mode", "global"))
        if self.clip_mode not in ("global", "per_group"):
            raise ValueError("clip_mode must be 'global' or 'per_group'")
        self.clip_median_mult = float(clip_median_mult if clip_median_mult is not None
                                      else getattr(cfg, "clip_median_mult", 5.0))
        self.clip_min_history = int(clip_min_history)
        self.clip_hist: dict = {g: deque(maxlen=int(clip_window)) for g in groups}
        self.clip_counts = {"steps": 0, "groups": 0, "clipped": 0}
        if named and not lr_warmup:
            lr_warmup = int(getattr(cfg, "lr_warmup_updates", 0) or 0)
        self.warmup_groups = set(warmup_groups) if warmup_groups is not None else (
            {g for g in groups if g.startswith("core")} if named else set(groups))
        self.role_steps = {r: 0 for r in roles}
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

    # ------------------------------------------------------------------ groups / roles
    @staticmethod
    def _default_lr(cfg, group: str) -> float:
        base = float(cfg.lr)
        if group.startswith("core"):
            return float(getattr(cfg, "lr_core", 0.0) or base)
        if group == "psi":
            return float(getattr(cfg, "lr_psi", 0.0) or base)
        if group == "semantic":
            return float(cfg.lr_sem)
        return float(getattr(cfg, "lr_heads", 0.0) or base)

    def _param_groups(self, cfg, names) -> list:
        pg = []
        for g in names:
            ps = self.groups[g]
            if not ps:
                continue
            dec = [p for p in ps if p.dim() >= 2]
            nod = [p for p in ps if p.dim() < 2]
            if dec:
                pg.append({"params": dec, "lr": self.group_lr[g], "weight_decay": cfg.weight_decay,
                           "name": f"{g}/decay", "base_lr": self.group_lr[g], "group": g})
            if nod:
                pg.append({"params": nod, "lr": self.group_lr[g], "weight_decay": 0.0, "name": f"{g}/no_decay",
                           "base_lr": self.group_lr[g], "group": g})
        return pg

    def _make_opt(self, pg: list):
        if not pg:
            return None
        mode = str(getattr(self.cfg, "sat_optimizer", "adamw") or "adamw")
        if mode not in ("adamw", "adafactor", "adafactor_all"):
            raise ValueError(f"cfg.sat_optimizer {mode!r}: 'adamw', 'adafactor' or 'adafactor_all'")
        if mode != "adamw" and any(str(g.get("group", "")) for g in pg):
            # fresh dicts with the trainer's own keys only (torch writes its defaults, e.g. AdamW's eps, into the
            # dicts it is given, and those are the trainer's _pg_spec entries)
            base = [{k: g[k] for k in ("params", "lr", "weight_decay", "name", "base_lr", "group") if k in g}
                    for g in pg]
            is_core = (lambda g: False) if mode == "adafactor_all" else \
                (lambda g: str(g.get("group", "")).startswith("core"))
            core = [g for g in base if is_core(g)]
            sat = [g for g in base if not is_core(g)]
            if sat:
                return SplitOptimizer(self._make_adamw(core) if core else None, torch.optim.Adafactor(sat))
        return self._make_adamw(pg)

    def _make_adamw(self, pg: list):
        fused = self.fused
        if fused is None:
            fused = all(p.is_cuda and p.dtype in (torch.float32, torch.float16, torch.bfloat16)
                        for g in pg for p in g["params"])
        if fused:
            try:
                return torch.optim.AdamW(pg, betas=self.betas, fused=True)
            except (RuntimeError, TypeError):             # fused unsupported for this build / device
                pass
        return torch.optim.AdamW(pg, betas=self.betas)

    def role_params(self, role: str) -> list:
        return [p for g in self.roles.get(role, []) for p in self.groups[g]]

    def group_of(self, p) -> Optional[str]:
        return self._group_of.get(id(p))

    def zero_grad(self, role: Optional[str] = None) -> None:
        for r, opt in self.opts.items():
            if role is None or r == role:
                opt.zero_grad(set_to_none=True)

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
        self.zero_grad()
        total.backward()
        res = self.apply("slow", count_update=False)
        if res["skipped"]:
            self.n_skipped += 1
            info = UpdateInfo(uid, self.n_updates, self.stage.name, raw, {k: float(v.detach()) for k, v in
                                                                            weighted.items()}, float(total.detach()),
                              res["grad_norm"], res["group_grad_norm"], False, bb, disabled, True, time.time() - t0,
                              res["clip"])
            self._log(info)
            return info
        info = UpdateInfo(uid, self.n_updates, self.stage.name, raw,
                          {k: float(v.detach()) for k, v in weighted.items()}, float(total.detach()),
                          res["grad_norm"], res["group_grad_norm"], res["clipped"], bb, disabled, False,
                          time.time() - t0, res["clip"])
        if consumed is not None:
            attach_provenance(consumed, uid)
        self._log(info)
        return info

    def apply(self, role: str = "slow", count_update: bool = True) -> dict:
        """Clip and step the gradients currently held by `role`'s parameters (computed by the caller, e.g. the
        integrated learner's per-objective accumulation, or by `step`). Non-finite gradients skip the step (or
        raise with on_nonfinite="raise"). A slow step refuses to run while any parameter of another role (psi)
        holds a nonzero gradient: actor / world losses must never reach the meta parameters.
        Returns {"grad_norm" (pre-clip, the role's global norm), "group_grad_norm", "clipped", "clip", "skipped",
        "lr"}."""
        if self.opts.get(role) is None:
            raise KeyError(f"unknown or empty role {role!r}")
        if role == "slow":
            leak = [g for r, gs in self.roles.items() if r != role for g in gs
                    for p in self.groups[g] if p.grad is not None and bool((p.grad != 0).any())]
            if leak:
                raise RuntimeError(f"slow step with gradients on other-role parameters (groups {sorted(set(leak))})")
        if count_update:
            self.n_updates += 1
        names = self.roles[role]
        gnorm = {}
        for g in names:
            gs = [p.grad.detach().float().norm() for p in self.groups[g] if p.grad is not None]
            gnorm[g] = float(torch.stack(gs).norm()) if gs else 0.0
        params = [p for p in self.role_params(role) if p.grad is not None]
        tot = math.sqrt(sum(v * v for v in gnorm.values()))
        out = {"grad_norm": tot, "group_grad_norm": gnorm, "clipped": False, "clip": None, "skipped": False,
               "lr": {}}
        if not math.isfinite(tot):
            self.zero_grad(role)
            if self.on_nonfinite == "raise":
                raise FloatingPointError("non-finite gradient")
            out["skipped"] = True
            return out
        if self.clip_mode == "per_group":
            clip = {}
            for g in names:
                n = gnorm[g]
                if n <= 0:
                    continue
                hist = self.clip_hist[g]
                thr = (self.clip_median_mult * float(sorted(hist)[len(hist) // 2])
                       if len(hist) >= self.clip_min_history else float("inf"))
                c = n > thr
                if c:
                    sc = thr / (n + 1e-12)
                    for p in self.groups[g]:
                        if p.grad is not None:
                            p.grad.mul_(sc)
                hist.append(n)
                clip[g] = {"norm": n, "threshold": thr, "clipped": bool(c)}
                self.clip_counts["groups"] += 1
                self.clip_counts["clipped"] += int(c)
            out["clip"] = clip
            out["clipped"] = any(v["clipped"] for v in clip.values())
        elif params:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            out["clipped"] = tot > self.cfg.grad_clip
            self.clip_counts["groups"] += 1
            self.clip_counts["clipped"] += int(out["clipped"])
        self.clip_counts["steps"] += 1
        self.role_steps[role] = self.role_steps.get(role, 0) + 1
        opt = self.opts[role]
        f = min(1.0, self.role_steps[role] / self.lr_warmup) if self.lr_warmup > 0 else 1.0
        for grp in opt.param_groups:
            warm = grp.get("group", grp["name"].split("/")[0]) in self.warmup_groups
            grp["lr"] = grp["base_lr"] * (f if warm else 1.0)
            out["lr"][grp["name"]] = grp["lr"]
        opt.step()
        self.zero_grad(role)
        return out

    def clip_rate(self) -> float:
        """Fraction of clipped (group, step) pairs so far (global mode: of steps)."""
        n = self.clip_counts["groups"]
        return self.clip_counts["clipped"] / n if n else 0.0

    def _log(self, info: UpdateInfo) -> None:
        self.history.append(info)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(info.to_dict(), default=float) + "\n")

    # ------------------------------------------------------------------ state
    def reset_optimizer(self) -> None:
        """Fresh optimiser moments (spec 11.5 / 13.7 condition B vs C), same groups and learning rates (every role)."""
        self.opts = {r: self._make_opt([{**{k: v for k, v in g.items() if k != "params"},
                                          "params": list(g["params"]), "lr": g["base_lr"]} for g in pg])
                     for r, pg in self._pg_spec.items() if pg}
        self.opt = self.opts["slow"]
        self.role_steps = {r: 0 for r in self.roles}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "n_updates": self.n_updates, "n_skipped": self.n_skipped,
                "stage": self.stage.index, "weights": dict(self.weights), "run_id": self.run_id,
                "opts": {r: o.state_dict() for r, o in self.opts.items() if r != "slow"},
                "role_steps": dict(self.role_steps), "clip_hist": {g: list(h) for g, h in self.clip_hist.items()},
                "clip_counts": dict(self.clip_counts)}

    def load_state_dict(self, s: dict) -> None:
        self.opt.load_state_dict(s["opt"])
        for r, st in (s.get("opts") or {}).items():
            if r in self.opts:
                self.opts[r].load_state_dict(st)
        self.n_updates, self.n_skipped = int(s["n_updates"]), int(s.get("n_skipped", 0))
        self.stage = self._resolve(int(s["stage"]))
        self.weights.update(s.get("weights", {}))
        self.role_steps.update({k: int(v) for k, v in (s.get("role_steps") or {}).items()})
        for g, h in (s.get("clip_hist") or {}).items():
            if g in self.clip_hist:
                self.clip_hist[g].clear()
                self.clip_hist[g].extend(float(x) for x in h)
        self.clip_counts.update(s.get("clip_counts") or {})


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
