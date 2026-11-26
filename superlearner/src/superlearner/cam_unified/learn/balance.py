"""Probe-calibrated loss balancing on the shared core (plan 2026-11-26 §2 "Loss weights are recomputed every 25 updates
from the named-group probe: w_i <- w_i * clip((target_i / share_i)^0.5, 0.5, 2)"; audit 'learning': the world model
took 96-99.6% of the core gradient and the actor 0.25-0.95%).

The balancer multiplies the configured loss weights by per-objective factors so that every objective's share of the
core gradient (`grad_probe.GradTable.share`, norm_i / sum_j norm_j over the core groups) satisfies its target:
``("max", s)`` (world <= 0.45) or ``("min", s)`` (actor >= 0.1, predictors >= 0.1 when present). Under Adam the
absolute scale of a loss barely matters for parameters only it reaches (its own heads); the weights therefore act on
the ratio of the objectives inside the shared core, which is what the targets constrain.

* Calibration (probes with `calibrate=True` from the caller, i.e. the warmup probes, once every constrained
  objective has a nonzero core gradient): the factors are solved directly from that probe, assuming each
  core-gradient norm scales linearly with its weight, by iterative proportional fitting toward the targets with a
  safety `margin` (world -> 0.45 * margin, actor -> 0.1 / margin). Without it the clipped rule would need ~25 x
  log2(40) updates to lift the actor share from 0.25% to 10%. Re-solving on every warmup probe matters on the real
  stack: at update 1 the actor's core gradient is the entropy bonus alone (zero-initialised value output -> zero
  advantages), and its unit-weight norm grows ~200x once the critic moves (measured, dom_dev).
* Refinement (later probes): the clipped square-root rule of the plan, w_i <- w_i * clip((target_i / share_i)^0.5,
  lo, hi), on an EMA of the unit-weight core norms (`ema`), applied only to objectives that violate their target.

Factors stay within `bounds`; every change is logged (`history`).
"""
from __future__ import annotations

import math
from typing import Optional

DEFAULT_TARGETS = {"world": ("max", 0.45), "actor": ("min", 0.10), "pred": ("min", 0.10)}


class LossBalancer:
    def __init__(self, targets: Optional[dict] = None, calibrate: bool = True, step_clip: tuple = (0.5, 2.0),
                 bounds: tuple = (1e-4, 1e4), margin: float = 0.85, iters: int = 200, ema: float = 0.5):
        self.targets = dict(DEFAULT_TARGETS if targets is None else targets)
        for k, (kind, s) in self.targets.items():
            if kind not in ("max", "min") or not 0 < float(s) < 1:
                raise ValueError(f"target {k}: ({kind!r}, {s}) must be ('max'|'min', share in (0, 1))")
        self.calibrate = bool(calibrate)
        self.step_clip = (float(step_clip[0]), float(step_clip[1]))
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.margin = float(margin)
        self.iters = int(iters)
        self.ema = float(ema)
        self.factors: dict = {}
        self.unit: dict = {}                 # EMA of core_norm_i / factor_i (the norm at factor 1)
        self.calibrated = False
        self.history: list = []

    def factor(self, name: str) -> float:
        return float(self.factors.get(name, 1.0))

    def apply(self, weights: dict) -> dict:
        """Base weights -> balanced weights (objectives without a factor keep their weight)."""
        return {k: float(v) * self.factor(k) for k, v in weights.items()}

    def _clamp(self, f: float) -> float:
        return min(self.bounds[1], max(self.bounds[0], f))

    def _active(self, core_norm: dict) -> dict:
        return {k: t for k, t in self.targets.items() if k in core_norm}

    def _solve(self, core_norm: dict) -> dict:
        """Factors g (relative to the measured state) with share_i(g) on the safe side of every target."""
        act = self._active(core_norm)
        g = {k: 1.0 for k in core_norm}
        for _ in range(self.iters):
            tot = sum(g[k] * core_norm[k] for k in core_norm)
            if tot <= 0:
                break
            moved = False
            for k, (kind, s) in act.items():
                share = g[k] * core_norm[k] / tot
                goal = s * self.margin if kind == "max" else min(0.99, s / self.margin)
                if (kind == "max" and share > goal) or (kind == "min" and share < goal):
                    if share <= 0:
                        continue
                    # the factor that puts objective k exactly at `goal` with the others fixed
                    rest = tot - g[k] * core_norm[k]
                    want = goal * rest / (1.0 - goal) / max(core_norm[k], 1e-30)
                    g[k] = want
                    tot = rest + g[k] * core_norm[k]
                    moved = True
            if not moved:
                break
        return g

    def update(self, n_update: int, table, calibrate: Optional[bool] = None) -> dict:
        """Adjust the factors from one probe (`GradTable` or {"core_norm": {...}}). `core_norm` is measured with the
        weights in force (factors included). `calibrate`: solve directly from this probe (default: only until the
        first successful calibration); otherwise the clipped square-root rule on the EMA shares. Returns the log
        entry (empty when nothing changed)."""
        core_norm = dict(table.core_norm if hasattr(table, "core_norm") else table["core_norm"])
        core_norm = {k: float(v) for k, v in core_norm.items() if math.isfinite(float(v))}
        tot = sum(core_norm.values())
        if tot <= 0:
            return {}
        share = {k: v / tot for k, v in core_norm.items()}
        for k, v in core_norm.items():                  # unit-weight norms, smoothed
            u = v / self.factor(k)
            self.unit[k] = u if k not in self.unit else (1 - self.ema) * self.unit[k] + self.ema * u
        act = self._active(core_norm)
        if not act:
            return {}
        entry = {"update": int(n_update), "share_before": share, "mode": None, "factors_before": dict(self.factors)}
        live = all(core_norm.get(k, 0.0) > 0 for k in act)
        do_cal = (not self.calibrated) if calibrate is None else bool(calibrate)
        if self.calibrate and do_cal and live:
            g = self._solve(core_norm)
            for k, v in g.items():
                if k in act and v != 1.0:
                    self.factors[k] = self._clamp(self.factor(k) * v)
            self.calibrated = True
            entry["mode"] = "calibrate"
        else:
            pred = {k: self.unit.get(k, 0.0) * self.factor(k) for k in core_norm}
            ptot = sum(pred.values())
            sm = {k: (v / ptot if ptot > 0 else 0.0) for k, v in pred.items()}
            entry["share_ema"] = sm
            lo, hi = self.step_clip
            changed = False
            for k, (kind, s) in act.items():
                sh = sm.get(k, 0.0)
                bad = (kind == "max" and sh > s) or (kind == "min" and sh < s)
                if not bad:
                    continue
                r = hi if sh <= 0 else min(hi, max(lo, math.sqrt(s / sh)))
                self.factors[k] = self._clamp(self.factor(k) * r)
                changed = True
            if not changed:
                return {}
            entry["mode"] = "clipped_sqrt"
        entry["factors_after"] = dict(self.factors)
        self.history.append(entry)
        return entry

    def state_dict(self) -> dict:
        return {"factors": dict(self.factors), "calibrated": self.calibrated, "targets": dict(self.targets),
                "unit": dict(self.unit), "history": list(self.history[-100:])}

    def load_state_dict(self, s: dict) -> None:
        self.factors = {k: float(v) for k, v in (s.get("factors") or {}).items()}
        self.unit = {k: float(v) for k, v in (s.get("unit") or {}).items()}
        self.calibrated = bool(s.get("calibrated", False))
        self.history = list(s.get("history") or [])
