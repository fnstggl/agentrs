"""Probe-calibrated loss balancing on the shared core (plan 2026-11-26 §2 "Loss weights are recomputed every 25 updates
from the named-group probe: w_i <- w_i * clip((target_i / share_i)^0.5, 0.5, 2)"; audit 'learning': the world model
took 96-99.6% of the core gradient and the actor 0.25-0.95%).

The balancer multiplies the configured loss weights by per-objective factors so that every constrained objective's
share of the core gradient (`grad_probe.GradTable.share`, norm_i / sum_j norm_j over the core groups) stays inside
its target range ``(lo, hi)``. Defaults: world (0.20, 0.45) - the plan's cap "world share <= 0.45" -, adapt
(0, 0.45) - the same cap for the other replay objective: on dom_dev the uncapped adaptation loss drifted to 0.79 of
the core within 20 updates (run 2, 2026-11-26) -, actor (0.10, 0.40) - the plan's floor "actor share >= 0.1" -,
predictors (0.10, 0.40) when present. The upper bound on the
decision objectives is not in the plan: without it the solve hands the actor ~98% of the core on the real stack
(measured, dom_dev, run of 2026-11-26), starving the world and adaptation objectives; the plan's own rule moves a
weight toward its target from both sides. ``("max", s)`` / ``("min", s)`` are shorthands for (0, s) / (s, 1).
Under Adam the absolute scale of a loss barely matters for parameters only it reaches (its own heads); the weights
therefore act on the ratio of the objectives inside the shared core, which is what the targets constrain.

* Calibration (probes with `calibrate=True` from the caller, i.e. the warmup probes, once every constrained
  objective has a nonzero core gradient): the factors are solved directly, assuming each core-gradient norm scales
  linearly with its weight, by iterative proportional fitting into the ranges with a safety `margin` (world ->
  0.45 * margin, actor -> 0.1 / margin), on the EMA of the unit-weight core norms (= the measurement on the first
  probe). Without it the clipped rule would need ~25 x log2(40) updates to lift the actor share from 0.25% to 10%.
  Re-solving on the warmup probes matters on the real stack: at update 1 the actor's core gradient is the entropy
  bonus alone (zero-initialised value output -> zero advantages), and its unit-weight norm grows ~200x once the
  critic moves (measured, dom_dev).
* Refinement (later probes): the clipped square-root rule of the plan, w_i <- w_i * clip((bound_i / share_i)^0.5,
  lo, hi) toward the violated bound, on the EMA shares, applied only to objectives outside their range.

Factors stay within `bounds`; every change is logged (`history`).
"""
from __future__ import annotations

import math
from typing import Optional

DEFAULT_TARGETS = {"world": (0.20, 0.45), "adapt": (0.0, 0.45), "actor": (0.10, 0.40), "pred": (0.10, 0.40)}


def _range(t) -> tuple:
    if len(t) == 2 and isinstance(t[0], str):
        kind, s = t
        if kind == "max":
            return 0.0, float(s)
        if kind == "min":
            return float(s), 1.0
        raise ValueError(f"target kind {kind!r} (use 'max' | 'min' or a (lo, hi) range)")
    lo, hi = float(t[0]), float(t[1])
    return lo, hi


class LossBalancer:
    def __init__(self, targets: Optional[dict] = None, calibrate: bool = True, step_clip: tuple = (0.5, 2.0),
                 bounds: tuple = (1e-4, 1e4), margin: float = 0.85, iters: int = 200, ema: float = 0.5):
        raw = dict(DEFAULT_TARGETS if targets is None else targets)
        self.targets = {}
        for k, t in raw.items():
            lo, hi = _range(t)
            if not (0.0 <= lo < hi <= 1.0) or (lo == 0.0 and hi == 1.0):
                raise ValueError(f"target {k}: range ({lo}, {hi}) must satisfy 0 <= lo < hi <= 1 and constrain")
            self.targets[k] = (lo, hi)
        if sum(lo for lo, _ in self.targets.values()) >= 1.0:
            raise ValueError("the lower bounds of the targets sum to >= 1 (infeasible)")
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

    def violated(self, share: dict, slack: float = 1.0) -> list:
        """Objectives whose share lies outside its range (widened by `slack` >= 1: hi * slack, lo / slack)."""
        out = []
        for k, (lo, hi) in self.targets.items():
            if k not in share:
                continue
            if share[k] > min(1.0, hi * slack) or share[k] < lo / slack:
                out.append(k)
        return out

    def _solve(self, core_norm: dict) -> dict:
        """Factors g (relative to the given norms) with share_i(g) inside every range (with the margin)."""
        act = self._active(core_norm)
        g = {k: 1.0 for k in core_norm}
        for _ in range(self.iters):
            tot = sum(g[k] * core_norm[k] for k in core_norm)
            if tot <= 0:
                break
            moved = False
            for k, (lo, hi) in act.items():
                share = g[k] * core_norm[k] / tot
                if share > hi:
                    goal = hi * self.margin
                elif share < lo:
                    goal = min(0.99, lo / self.margin)
                else:
                    continue
                if share <= 0:
                    continue
                rest = tot - g[k] * core_norm[k]
                if rest <= 0:
                    continue
                # the factor that puts objective k exactly at `goal` with the others fixed
                g[k] = goal * rest / (1.0 - goal) / max(core_norm[k], 1e-30)
                tot = rest + g[k] * core_norm[k]
                moved = True
            if not moved:
                break
        return g

    def update(self, n_update: int, table, calibrate: Optional[bool] = None) -> dict:
        """Adjust the factors from one probe (`GradTable` or {"core_norm": {...}}). `core_norm` is measured with the
        weights in force (factors included). `calibrate`: solve directly (default: only until the first successful
        calibration); otherwise the clipped square-root rule on the EMA shares. Returns the log entry (empty when
        nothing changed)."""
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
        pred = {k: self.unit.get(k, 0.0) * self.factor(k) for k in core_norm}
        ptot = sum(pred.values())
        sm = {k: (v / ptot if ptot > 0 else 0.0) for k, v in pred.items()}
        entry["share_ema"] = sm
        live = all(pred.get(k, 0.0) > 0 for k in act)
        do_cal = (not self.calibrated) if calibrate is None else bool(calibrate)
        if self.calibrate and do_cal and live:
            g = self._solve(pred)
            changed = False
            for k, v in g.items():
                if k in act and abs(v - 1.0) > 1e-12:
                    self.factors[k] = self._clamp(self.factor(k) * v)
                    changed = True
            self.calibrated = True
            if not changed:
                return {}
            entry["mode"] = "calibrate"
        else:
            lo_c, hi_c = self.step_clip
            changed = False
            for k, (lo, hi) in act.items():
                sh = sm.get(k, 0.0)
                if lo <= sh <= hi:
                    continue
                bound = hi if sh > hi else lo
                r = hi_c if sh <= 0 else min(hi_c, max(lo_c, math.sqrt(bound / sh)))
                self.factors[k] = self._clamp(self.factor(k) * r)
                changed = True
            if not changed:
                return {}
            entry["mode"] = "clipped_sqrt"
        entry["factors_after"] = dict(self.factors)
        self.history.append(entry)
        return entry

    def state_dict(self) -> dict:
        return {"factors": dict(self.factors), "calibrated": self.calibrated,
                "targets": {k: list(v) for k, v in self.targets.items()}, "unit": dict(self.unit),
                "history": list(self.history[-100:])}

    def load_state_dict(self, s: dict) -> None:
        self.factors = {k: float(v) for k, v in (s.get("factors") or {}).items()}
        self.unit = {k: float(v) for k, v in (s.get("unit") or {}).items()}
        self.calibrated = bool(s.get("calibrated", False))
        self.history = list(s.get("history") or [])
