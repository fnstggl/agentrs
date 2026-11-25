"""Plasticity instruments and optional plasticity mechanisms (spec 11.3, architecture §8).

Instruments (``PlasticityMonitor`` + functions; they never change the network):

    dormant units          Sokar et al. 2023: unit i of a hidden layer is dormant when its mean |activation| divided by
                           the layer's mean is <= tau. Measured by forward pre-hooks on the layer that CONSUMES the
                           hidden units (its input = the post-activation hidden vector), which covers ReLU/GELU MLPs,
                           nn.TransformerEncoderLayer (linear1/linear2) and SwiGLU (w1, w3 -> w2) alike.
    activation diversity   effective rank of the captured hidden activations per unit layer
    effective rank         ``effective_rank`` (Roy & Vetterli 2007) and ``srank`` (Kumar et al. 2021) of any [N, D]
                           representation (e.g. Z_t over a probe batch)
    gradient norms         per parameter group (after a backward pass)
    weight change          per group L2 / relative distance to a reference snapshot; per-step update magnitudes
    gradient interference  cosine between the gradients of several objectives / tasks (overall and per group)
    new-task learning speed  ``learning_speed``: a COPY trained from a fresh optimiser on a probe task
    historical retention   ``RetentionTracker``: score matrix R[task][after-training-on], forgetting, backward transfer

Mechanisms (each behind its own switch in ``PlasticitySwitches``, all off by default; tested independently):

    continual_backprop     Dohare et al. 2024: running contribution utility per unit; a fraction of mature, least
                           useful units is re-initialised every step (incoming fresh, outgoing zero + bias correction,
                           optimiser moments reset)
    selective_reinit       ReDo: re-initialise units that the monitor finds dormant
    adaptive_lr            per optimiser-group learning-rate multipliers steering the relative update size toward a
                           target (groups whose updates vanish get larger steps, exploding ones smaller)
    pcgrad                 Yu et al. 2020: project away conflicting components between objectives' gradients
    replay_distill         distillation of a frozen teacher snapshot's outputs on replayed inputs (retention)
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ unit layers
@dataclass
class UnitLayer:
    """A hidden layer of units: ``fc_in`` (and ``fc_gate`` for SwiGLU) produce them (rows), ``fc_out`` consumes them
    (columns)."""
    name: str
    fc_in: nn.Linear
    fc_out: nn.Linear
    fc_gate: Optional[nn.Linear] = None

    @property
    def n_units(self) -> int:
        return self.fc_out.in_features


_PAIRS = (("linear1", "linear2", None), ("fc1", "fc2", None), ("w1", "w2", "w3"), ("gate_proj", "down_proj", "up_proj"),
          ("up", "down", None))


def find_unit_layers(model: nn.Module) -> list:
    """Discover hidden unit layers: attribute pairs (linear1/linear2, fc1/fc2, w1/w2(/w3), gate/down(/up), up/down)
    and Linear -> activation(s) -> Linear runs inside nn.Sequential."""
    out, seen = [], set()
    for name, m in model.named_modules():
        for a_in, a_out, a_gate in _PAIRS:
            fi, fo = getattr(m, a_in, None), getattr(m, a_out, None)
            if isinstance(fi, nn.Linear) and isinstance(fo, nn.Linear) and fi.out_features == fo.in_features:
                fg = getattr(m, a_gate, None) if a_gate else None
                key = (id(fi), id(fo))
                if key not in seen:
                    seen.add(key)
                    out.append(UnitLayer(f"{name}.{a_in}" if name else a_in, fi, fo,
                                         fg if isinstance(fg, nn.Linear) else None))
        if isinstance(m, nn.Sequential):
            mods = list(m.children())
            for i, mi in enumerate(mods):
                if not isinstance(mi, nn.Linear):
                    continue
                j = i + 1
                while j < len(mods) and not isinstance(mods[j], nn.Linear):
                    j += 1
                if j < len(mods) and j > i + 1 and mi.out_features == mods[j].in_features:
                    key = (id(mi), id(mods[j]))
                    if key not in seen:
                        seen.add(key)
                        out.append(UnitLayer(f"{name}.{i}" if name else str(i), mi, mods[j]))
    return out


def default_group(name: str) -> str:
    return name.split(".", 1)[0]


# ============================================================================ functions
def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of a centred [N, D] matrix (1 .. min(N, D))."""
    x = x.detach().float()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    if float(s.sum()) <= 1e-12:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))


def srank(x: torch.Tensor, delta: float = 0.01) -> int:
    """Smallest k whose top-k singular values hold a (1 - delta) share of the total (Kumar et al. 2021)."""
    s = torch.linalg.svdvals(x.detach().float())
    if float(s.sum()) <= 1e-12:
        return 0
    c = torch.cumsum(s, 0) / s.sum()
    return int((c < 1 - delta).sum()) + 1


def _flat_grads(loss, params, retain=True):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, params)]


def gradient_interference(losses: dict, named_params: Sequence, group_fn: Callable = default_group) -> dict:
    """Cosine similarity between the gradients of several objectives/tasks.

    losses: name -> scalar loss (sharing one graph is fine); named_params: [(name, Parameter)].
    Returns {"names", "cos" [n][n], "norms", "by_group": {group: [n][n]}}."""
    names = list(losses)
    params = [p for _, p in named_params]
    groups = [group_fn(n) for n, _ in named_params]
    gl = [_flat_grads(losses[k], params, retain=True) for k in names]
    flat = torch.stack([torch.cat([g.reshape(-1).float() for g in gs]) for gs in gl])
    norms = flat.norm(dim=1)
    cos = (flat @ flat.T) / (norms[:, None] * norms[None, :]).clamp(min=1e-12)
    by = {}
    for gname in sorted(set(groups)):
        sel = [i for i, g in enumerate(groups) if g == gname]
        fg = torch.stack([torch.cat([gs[i].reshape(-1).float() for i in sel]) for gs in gl])
        nn_ = fg.norm(dim=1)
        by[gname] = ((fg @ fg.T) / (nn_[:, None] * nn_[None, :]).clamp(min=1e-12)).tolist()
    return {"names": names, "cos": cos.tolist(), "norms": norms.tolist(), "by_group": by}


def learning_speed(model: nn.Module, batch_fn: Callable, loss_fn: Callable, steps: int = 100, lr: float = 1e-3,
                   threshold: Optional[float] = None, optimizer_cls=torch.optim.Adam, copy_model: bool = True) -> dict:
    """New-task learning speed of (a copy of) ``model``: fresh optimiser, ``steps`` updates on ``batch_fn(step)``.

    loss_fn(model, batch) -> scalar. Returns {"curve", "auc" (mean loss), "final", "first", "steps_to_threshold"}."""
    m = copy.deepcopy(model) if copy_model else model
    m.train()
    opt = optimizer_cls([p for p in m.parameters() if p.requires_grad], lr=lr)
    curve, hit = [], None
    for t in range(steps):
        loss = loss_fn(m, batch_fn(t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        curve.append(v)
        if threshold is not None and hit is None and v <= threshold:
            hit = t + 1
    return {"curve": curve, "auc": sum(curve) / len(curve), "final": curve[-1], "first": curve[0],
            "steps_to_threshold": hit}


class RetentionTracker:
    """Sequential-task bookkeeping: ``record(task, after, score)`` = score on ``task`` after training on ``after``
    (higher = better); ``record_acquisition(task, **speed)`` = learning-speed numbers when the task was learned."""

    def __init__(self):
        self.order: list = []
        self.R: dict = {}
        self.acq: dict = {}

    def trained(self, task: str) -> None:
        if task not in self.order:
            self.order.append(task)

    def record(self, task: str, after: str, score: float) -> None:
        self.R.setdefault(task, {})[after] = float(score)

    def record_acquisition(self, task: str, **speed) -> None:
        self.acq[task] = dict(speed)

    def report(self) -> dict:
        last = self.order[-1] if self.order else None
        forgetting, bwt = {}, []
        for i, t in enumerate(self.order[:-1]):
            hist = [self.R.get(t, {}).get(a) for a in self.order[i:]]
            hist = [h for h in hist if h is not None]
            if len(hist) >= 2:
                forgetting[t] = max(hist[:-1]) - hist[-1]
                bwt.append(hist[-1] - hist[0])
        final = {t: self.R.get(t, {}).get(last) for t in self.order}
        vals = [v for v in final.values() if v is not None]
        return {"order": list(self.order), "matrix": {t: dict(r) for t, r in self.R.items()}, "final": final,
                "avg_final": sum(vals) / len(vals) if vals else None, "forgetting": forgetting,
                "backward_transfer": sum(bwt) / len(bwt) if bwt else None, "acquisition": dict(self.acq)}


# ============================================================================ monitor
class PlasticityMonitor:
    """Hooks + metrics over a model. ``attach()`` before forward passes that should be measured; ``report()``."""

    def __init__(self, model: nn.Module, layers: Optional[Sequence[UnitLayer]] = None,
                 group_fn: Callable = default_group, tau: float = 0.025, rank_samples: int = 2048, seed: int = 0):
        self.model = model
        self.layers = list(layers) if layers is not None else find_unit_layers(model)
        self.group_fn = group_fn
        self.tau = tau
        self.rank_samples = rank_samples
        self.rng = random.Random(seed)
        self._hooks: list = []
        self._ref: Optional[dict] = None
        self._pre: Optional[dict] = None
        self.updates: list = []
        self.reset_activation_stats()

    def reset_activation_stats(self) -> None:
        self._sum = {l.name: torch.zeros(l.n_units) for l in self.layers}
        self._cnt = {l.name: 0 for l in self.layers}
        self._samples = {l.name: [] for l in self.layers}
        self._seen = {l.name: 0 for l in self.layers}

    def attach(self) -> "PlasticityMonitor":
        self.detach()
        for l in self.layers:
            self._hooks.append(l.fc_out.register_forward_pre_hook(self._hook(l)))
        return self

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        return self.attach()

    def __exit__(self, *a):
        self.detach()

    def _hook(self, layer: UnitLayer):
        name = layer.name

        def fn(_mod, inp):
            h = inp[0].detach().reshape(-1, layer.n_units).float()
            self._sum[name] += h.abs().sum(0).cpu()
            self._cnt[name] += h.shape[0]
            buf = self._samples[name]
            for row in h[: max(1, self.rank_samples // 8)].cpu():
                self._seen[name] += 1
                if len(buf) < self.rank_samples:
                    buf.append(row)
                else:
                    j = self.rng.randrange(self._seen[name])
                    if j < self.rank_samples:
                        buf[j] = row
        return fn

    # ------------------------------------------------------------------ activation metrics
    def mean_activation(self) -> dict:
        return {n: (self._sum[n] / max(1, self._cnt[n])) for n in self._sum}

    def dormant(self, tau: Optional[float] = None) -> dict:
        """{"fraction", "by_layer": {name: fraction}, "mask": {name: bool [units]}} over the activations seen."""
        tau = self.tau if tau is None else tau
        by, masks, tot, dead = {}, {}, 0, 0
        for n, a in self.mean_activation().items():
            if self._cnt[n] == 0:
                continue
            s = a / a.mean().clamp(min=1e-12)
            m = s <= tau
            masks[n] = m
            by[n] = float(m.float().mean())
            tot += m.numel()
            dead += int(m.sum())
        return {"fraction": dead / tot if tot else 0.0, "by_layer": by, "mask": masks}

    def activation_rank(self) -> dict:
        return {n: effective_rank(torch.stack(b)) for n, b in self._samples.items() if len(b) >= 2}

    # ------------------------------------------------------------------ parameter metrics
    def _groups(self) -> dict:
        by: dict = {}
        for n, p in self.model.named_parameters():
            by.setdefault(self.group_fn(n), []).append((n, p))
        return by

    def grad_norms(self) -> dict:
        return {g: math.sqrt(sum(float(p.grad.detach().float().pow(2).sum()) for _, p in items if p.grad is not None))
                for g, items in self._groups().items()}

    def weight_norms(self) -> dict:
        return {g: math.sqrt(sum(float(p.detach().float().pow(2).sum()) for _, p in items))
                for g, items in self._groups().items()}

    def set_reference(self) -> None:
        self._ref = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}

    def weight_change(self, ref: Optional[dict] = None) -> dict:
        """Per group {"l2", "relative", "params"} distance to the reference snapshot, plus "all"."""
        ref = ref if ref is not None else self._ref
        if ref is None:
            raise ValueError("no reference snapshot: call set_reference() first")
        out, td, tn, tp = {}, 0.0, 0.0, 0
        for g, items in self._groups().items():
            d = sum(float((p.detach().cpu().float() - ref[n].float()).pow(2).sum()) for n, p in items if n in ref)
            nr = sum(float(ref[n].float().pow(2).sum()) for n, _ in items if n in ref)
            np_ = sum(p.numel() for _, p in items)
            out[g] = {"l2": math.sqrt(d), "relative": math.sqrt(d / nr) if nr > 0 else 0.0, "params": np_}
            td, tn, tp = td + d, tn + nr, tp + np_
        out["all"] = {"l2": math.sqrt(td), "relative": math.sqrt(td / tn) if tn > 0 else 0.0, "params": tp}
        return out

    def begin_step(self) -> None:
        """Call before optimizer.step(); ``end_step`` then records per-group update magnitudes."""
        self._pre = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def end_step(self) -> dict:
        if self._pre is None:
            raise ValueError("begin_step() was not called")
        by: dict = {}
        for n, p in self.model.named_parameters():
            if n in self._pre:
                g = self.group_fn(n)
                d = float((p.detach() - self._pre[n]).float().pow(2).sum())
                w = float(self._pre[n].float().pow(2).sum())
                a = by.setdefault(g, [0.0, 0.0])
                a[0] += d
                a[1] += w
        self._pre = None
        rec = {g: {"l2": math.sqrt(d), "relative": math.sqrt(d / w) if w > 0 else 0.0} for g, (d, w) in by.items()}
        self.updates.append(rec)
        return rec

    def report(self, features: Optional[torch.Tensor] = None) -> dict:
        d = self.dormant()
        out = {"dormant_fraction": d["fraction"], "dormant_by_layer": d["by_layer"],
               "activation_rank": self.activation_rank(), "grad_norm": self.grad_norms(),
               "weight_norm": self.weight_norms()}
        if self._ref is not None:
            out["weight_change"] = self.weight_change()
        if self.updates:
            out["last_update"] = self.updates[-1]
        if features is not None:
            out["effective_rank"] = effective_rank(features)
            out["srank"] = srank(features)
        return out


# ============================================================================ mechanisms
@dataclass
class PlasticitySwitches:
    continual_backprop: bool = False
    selective_reinit: bool = False
    adaptive_lr: bool = False
    pcgrad: bool = False
    replay_distill: bool = False


def _zero_opt_state(optimizer, param: torch.Tensor, idx: torch.Tensor, dim: int) -> None:
    if optimizer is None:
        return
    st = optimizer.state.get(param)
    if not st:
        return
    for v in st.values():
        if torch.is_tensor(v) and v.shape == param.shape:
            v.index_fill_(dim, idx.to(v.device), 0)


@torch.no_grad()
def reinit_units(layer: UnitLayer, idx: torch.Tensor, optimizer=None, generator: Optional[torch.Generator] = None,
                 out_bias_correction: Optional[torch.Tensor] = None) -> None:
    """Fresh incoming weights (U(-1/sqrt(fan_in), +)), zero incoming bias, zero outgoing weights for units ``idx``;
    optimiser moments of the touched rows/columns reset. ``out_bias_correction`` [units] (mean activation) is folded
    into the output bias so the network function is preserved in expectation."""
    if len(idx) == 0:
        return
    idx = idx.to(layer.fc_out.weight.device)
    for fc in (layer.fc_in, layer.fc_gate):
        if fc is None:
            continue
        b = 1.0 / math.sqrt(fc.in_features)
        w = (torch.rand(len(idx), fc.in_features, generator=generator) * 2 - 1) * b
        fc.weight[idx] = w.to(fc.weight.device, fc.weight.dtype)
        _zero_opt_state(optimizer, fc.weight, idx, 0)
        if fc.bias is not None:
            fc.bias[idx] = 0
            _zero_opt_state(optimizer, fc.bias, idx, 0)
    if out_bias_correction is not None and layer.fc_out.bias is not None:
        layer.fc_out.bias += layer.fc_out.weight[:, idx] @ out_bias_correction[idx].to(layer.fc_out.weight)
    layer.fc_out.weight[:, idx] = 0
    _zero_opt_state(optimizer, layer.fc_out.weight, idx, 1)


class ContinualBackprop:
    """Continual backpropagation (Dohare et al. 2024). Call ``step()`` after each optimiser step (the forward pass
    of that step feeds the utility through hooks)."""

    def __init__(self, layers: Sequence[UnitLayer], optimizer=None, replacement_rate: float = 1e-3,
                 decay: float = 0.99, maturity: int = 100, seed: int = 0):
        self.layers = list(layers)
        self.optimizer = optimizer
        self.rate, self.decay, self.maturity = replacement_rate, decay, maturity
        self.gen = torch.Generator().manual_seed(seed)
        self.util = {l.name: torch.zeros(l.n_units) for l in self.layers}
        self.mean_act = {l.name: torch.zeros(l.n_units) for l in self.layers}
        self.age = {l.name: torch.zeros(l.n_units, dtype=torch.long) for l in self.layers}
        self.acc = {l.name: 0.0 for l in self.layers}
        self._last = {}
        self.replaced = {l.name: 0 for l in self.layers}
        self._hooks = [l.fc_out.register_forward_pre_hook(self._hook(l)) for l in self.layers]

    def _hook(self, layer):
        def fn(_m, inp):
            h = inp[0].detach().reshape(-1, layer.n_units).float()
            self._last[layer.name] = (h.abs().mean(0).cpu(), h.mean(0).cpu())
        return fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def step(self) -> dict:
        out = {}
        for l in self.layers:
            n = l.name
            if n not in self._last:
                continue
            a_abs, a_mean = self._last[n]
            contrib = a_abs * l.fc_out.weight.detach().abs().sum(0).float().cpu()
            self.util[n] = self.decay * self.util[n] + (1 - self.decay) * contrib
            self.mean_act[n] = self.decay * self.mean_act[n] + (1 - self.decay) * a_mean
            self.age[n] += 1
            eligible = torch.nonzero(self.age[n] > self.maturity).squeeze(1)
            self.acc[n] += self.rate * len(eligible)
            k = int(self.acc[n])
            if k <= 0 or len(eligible) == 0:
                out[n] = 0
                continue
            self.acc[n] -= k
            k = min(k, len(eligible))
            worst = eligible[torch.argsort(self.util[n][eligible])[:k]]
            reinit_units(l, worst, self.optimizer, self.gen, out_bias_correction=self.mean_act[n])
            self.util[n][worst] = 0
            self.mean_act[n][worst] = 0
            self.age[n][worst] = 0
            self.replaced[n] += k
            out[n] = k
        return out


class SelectiveReinit:
    """ReDo-style re-initialisation of dormant units found by a ``PlasticityMonitor``."""

    def __init__(self, layers: Sequence[UnitLayer], optimizer=None, tau: float = 0.025, seed: int = 0):
        self.layers = {l.name: l for l in layers}
        self.optimizer = optimizer
        self.tau = tau
        self.gen = torch.Generator().manual_seed(seed)
        self.total = 0

    @torch.no_grad()
    def apply(self, monitor: PlasticityMonitor) -> dict:
        d = monitor.dormant(self.tau)
        out = {}
        for n, m in d["mask"].items():
            if n not in self.layers:
                continue
            idx = torch.nonzero(m).squeeze(1)
            reinit_units(self.layers[n], idx, self.optimizer, self.gen)
            out[n] = int(len(idx))
            self.total += int(len(idx))
        monitor.reset_activation_stats()
        return out


class AdaptiveGroupLR:
    """Per optimiser param-group LR multipliers that steer the EMA of the relative update ||dW|| / ||W|| toward
    ``target`` (multiplicative, damped by ``beta``, per-step factor clipped to [1/2, 2], overall [min_mult, max_mult]).

    The multiplier is applied on top of whatever lr the group holds (an LR scheduler's or the user's) only for the
    duration of ``optimizer.step()``: ``before_step`` scales ``lr`` by the multiplier, ``after_step`` restores the
    scheduled value. Schedules therefore run unchanged, optimiser state dicts never contain a multiplied lr, and a
    fresh-optimiser reset is not affected by the multipliers (``state_dict`` / ``reset`` cover them separately).
    Usage: ``alr.before_step(); opt.step(); alr.after_step(); sched.step()``."""

    def __init__(self, optimizer, target: float = 1e-3, beta: float = 0.5, min_mult: float = 0.05,
                 max_mult: float = 20.0, ema: float = 0.9):
        self.opt = optimizer
        self.target, self.beta, self.min_mult, self.max_mult, self.ema = target, beta, min_mult, max_mult, ema
        self.mult = [1.0] * len(optimizer.param_groups)
        self.r_ema: list = [None] * len(optimizer.param_groups)
        self._pre = None
        self._sched = None

    def effective_lr(self) -> list:
        """lr each group will use at the next step: scheduled lr x multiplier."""
        return [g["lr"] * m for g, m in zip(self.opt.param_groups, self.mult)]

    def before_step(self) -> None:
        if self._sched is not None:
            raise ValueError("after_step() was not called for the previous before_step()")
        self._pre = [[p.detach().clone() for p in g["params"]] for g in self.opt.param_groups]
        self._sched = [g["lr"] for g in self.opt.param_groups]
        for g, m in zip(self.opt.param_groups, self.mult):
            g["lr"] = g["lr"] * m

    @torch.no_grad()
    def after_step(self) -> list:
        if self._pre is None:
            raise ValueError("before_step() was not called")
        info = []
        for i, (g, pre, lr_s) in enumerate(zip(self.opt.param_groups, self._pre, self._sched)):
            used = g["lr"]
            g["lr"] = lr_s                                  # the schedule's value, untouched
            d = math.sqrt(sum(float((p - q).float().pow(2).sum()) for p, q in zip(g["params"], pre)))
            w = math.sqrt(sum(float(q.float().pow(2).sum()) for q in pre))
            r = d / (w + 1e-12)
            self.r_ema[i] = r if self.r_ema[i] is None else self.ema * self.r_ema[i] + (1 - self.ema) * r
            if self.r_ema[i] > 0:
                f = (self.target / self.r_ema[i]) ** self.beta
                self.mult[i] = min(self.max_mult, max(self.min_mult, self.mult[i] * min(2.0, max(0.5, f))))
            info.append({"group": g.get("name", i), "ratio": r, "ratio_ema": self.r_ema[i], "mult": self.mult[i],
                         "lr_used": used, "lr_scheduled": lr_s})
        self._pre = None
        self._sched = None
        return info

    def state_dict(self) -> dict:
        return {"mult": list(self.mult), "r_ema": list(self.r_ema)}

    def load_state_dict(self, st: dict) -> None:
        self.mult, self.r_ema = list(st["mult"]), list(st["r_ema"])

    def reset(self) -> None:
        self.mult = [1.0] * len(self.opt.param_groups)
        self.r_ema = [None] * len(self.opt.param_groups)


class PCGrad:
    """Gradient surgery across objectives: each objective's gradient loses its component along any other objective's
    gradient it conflicts with (negative dot product); the projected gradients are summed into ``.grad``."""

    def __init__(self, params: Sequence[torch.Tensor], seed: int = 0):
        self.params = [p for p in params if p.requires_grad]
        self.rng = random.Random(seed)

    def backward(self, losses) -> dict:
        losses = list(losses.values()) if isinstance(losses, dict) else list(losses)
        grads = []
        for i, L in enumerate(losses):
            gs = _flat_grads(L, self.params, retain=i < len(losses) - 1)
            grads.append(torch.cat([g.reshape(-1) for g in gs]))
        G = torch.stack(grads)
        P = G.clone()
        conflicts = 0
        for i in range(len(losses)):
            order = [j for j in range(len(losses)) if j != i]
            self.rng.shuffle(order)
            for j in order:
                dot = torch.dot(P[i], G[j])
                if dot < 0:
                    P[i] -= dot / G[j].pow(2).sum().clamp(min=1e-20) * G[j]
                    conflicts += 1
        tot = P.sum(0)
        off = 0
        for p in self.params:
            n = p.numel()
            g = tot[off:off + n].view_as(p).to(p.dtype)
            p.grad = g.clone() if p.grad is None else p.grad + g
            off += n
        nrm = G.norm(dim=1)
        cos = (G @ G.T) / (nrm[:, None] * nrm[None, :]).clamp(min=1e-12)
        return {"conflicts": conflicts, "cos": cos.tolist()}


class ReplayDistillation:
    """Retention by distillation: match a frozen teacher snapshot's outputs on replayed inputs.

    kind "mse" for real-valued outputs, "kl" for logits (temperature-scaled, times T^2)."""

    def __init__(self, teacher: nn.Module, kind: str = "mse", temperature: float = 1.0, weight: float = 1.0):
        if kind not in ("mse", "kl"):
            raise ValueError("kind must be 'mse' or 'kl'")
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.kind, self.T, self.weight = kind, temperature, weight

    @classmethod
    def from_model(cls, model: nn.Module, **kw) -> "ReplayDistillation":
        return cls(copy.deepcopy(model), **kw)

    def loss(self, student: nn.Module, inputs, forward: Optional[Callable] = None) -> torch.Tensor:
        fwd = forward or (lambda m, x: m(x))
        with torch.no_grad():
            t = fwd(self.teacher, inputs)
        s = fwd(student, inputs)
        if self.kind == "mse":
            return self.weight * F.mse_loss(s.float(), t.float())
        T = self.T
        return self.weight * F.kl_div(F.log_softmax(s.float() / T, -1), F.log_softmax(t.float() / T, -1),
                                      log_target=True, reduction="batchmean") * T * T


class PlasticityMechanisms:
    """Applies exactly the switched-on mechanisms around a training step::

        mech.backward({"world": lw, "policy": lp})   # PCGrad if on, else the plain sum
        mech.before_step(); opt.step(); mech.after_step()
    """

    def __init__(self, model: nn.Module, optimizer, switches: PlasticitySwitches,
                 layers: Optional[Sequence[UnitLayer]] = None, monitor: Optional[PlasticityMonitor] = None,
                 teacher: Optional[nn.Module] = None, reinit_interval: int = 1000, cbp_kwargs: Optional[dict] = None,
                 lr_kwargs: Optional[dict] = None, distill_kwargs: Optional[dict] = None, seed: int = 0):
        self.model, self.opt, self.sw = model, optimizer, switches
        layers = list(layers) if layers is not None else find_unit_layers(model)
        self.cbp = ContinualBackprop(layers, optimizer, seed=seed, **(cbp_kwargs or {})) \
            if switches.continual_backprop else None
        self.monitor = monitor
        if switches.selective_reinit and self.monitor is None:
            self.monitor = PlasticityMonitor(model, layers).attach()
        self.redo = SelectiveReinit(layers, optimizer, seed=seed) if switches.selective_reinit else None
        self.reinit_interval = reinit_interval
        self.alr = AdaptiveGroupLR(optimizer, **(lr_kwargs or {})) if switches.adaptive_lr else None
        self.pcg = PCGrad([p for p in model.parameters()], seed=seed) if switches.pcgrad else None
        self.distill = None
        if switches.replay_distill:
            self.distill = ReplayDistillation(teacher if teacher is not None else copy.deepcopy(model),
                                              **(distill_kwargs or {}))
        self.n_steps = 0

    def backward(self, losses: dict) -> dict:
        if self.pcg is not None:
            return self.pcg.backward(losses)
        sum(losses.values()).backward()
        return {}

    def distill_loss(self, inputs, forward: Optional[Callable] = None):
        if self.distill is None:
            return None
        return self.distill.loss(self.model, inputs, forward)

    def before_step(self) -> None:
        if self.alr is not None:
            self.alr.before_step()

    def after_step(self) -> dict:
        self.n_steps += 1
        info = {}
        if self.alr is not None:
            info["adaptive_lr"] = self.alr.after_step()
        if self.cbp is not None:
            info["continual_backprop"] = self.cbp.step()
        if self.redo is not None and self.n_steps % self.reinit_interval == 0:
            info["selective_reinit"] = self.redo.apply(self.monitor)
        return info
