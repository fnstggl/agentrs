"""Operator-side scripted settings: fixtures computed from the hidden mechanism.

A *profile* is a constant settings vector for one seat, written down by someone who knows what every lever is: an
aggressive or a passive firm in a market, restraint or grabbing on a shared resource, a tuned pipeline. Profiles
are experimental fixtures: controlled opponents to train against, and reference points to place learned behaviour
between. They are built from private information (`engine.encode`, `techsys.reference_settings`), so an individual
that plays one is marked `privileged` and is never counted as a learning system. No learner is ever given a profile.

Market profiles (agent-based engine), in natural units (`p0` is the market's reference price, `out_hi` the top of
the outreach range):

    aggressive   price 0.95 p0, effort 0.25, outreach 0.30 out_hi   undercuts and advertises hard
    passive      price 1.45 p0, effort 0.00, outreach 0.03 out_hi   dear, barely present
    compete      price 0.90 p0, effort 0.10, outreach 0.10 out_hi   a sharp competitor: in the sampled duopolies, two
                                                                     such firms are at the price from which
                                                                     undercutting by a further 0.1 p0 no longer pays
    collude      price 1.15 p0, effort 0.10, outreach 0.10 out_hi   the same firm at about the price that maximises
                                                                     the two seats' joint net (undercutting it pays)
    restrain     price 1.10 p0, effort 0.10, outreach 0.05 out_hi   little outreach (spares a shared attention pool)
    grab         price 1.10 p0, effort 0.10, outreach 0.50 out_hi   heavy outreach (depletes it)
    partner      price 0.75 x default, effort 0.10, outreach 0.08 out_hi   a component maker doing its part of a bundle

Technical-system profiles: the styles of `techsys.reference_settings` (`tuned`, `capacity`, `restraint`, `grab`, ...).
`default` (any engine) is the resting settings.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..research.learner import EnvSpec
from ..research.tasks import Task
from .arena import Individual, _sim
from .learners import FixedLearner, learner_config

ABM_PROFILES: dict[str, dict[str, float]] = {
    "aggressive": {"price": 0.95, "quality": 0.25, "outreach": 0.30},
    "passive": {"price": 1.45, "quality": 0.0, "outreach": 0.03},
    "compete": {"price": 0.90, "quality": 0.10, "outreach": 0.10},
    "collude": {"price": 1.15, "quality": 0.10, "outreach": 0.10},
    "restrain": {"price": 1.10, "quality": 0.10, "outreach": 0.05},
    "grab": {"price": 1.10, "quality": 0.10, "outreach": 0.50},
    "partner": {"price": -0.75, "quality": 0.10, "outreach": 0.08},     # negative price: a multiple of the default
}


def available(task: Task) -> list[str]:
    eng = _sim(task, [None] * task.n_seats).engine
    if eng.engine == "abm":
        return ["default"] + sorted(ABM_PROFILES)
    if eng.engine == "techsys":
        from ..mechanisms.techsys import _STYLES
        names = sorted(_STYLES)
        if "restraint" not in eng.config.get("design", {}):
            names = [n for n in names if n not in ("restraint", "grab")]
        return ["default"] + names
    return ["default"]


def profile_settings(task: Task, seat: int, name: str, **natural: float) -> dict[str, float]:
    """The settings (lever name -> value) of a profile for one seat. For markets, `natural` overrides or adds
    natural values (`price` as a multiple of p0, `quality` as effort, `outreach` as a fraction of its range)."""
    eng = _sim(task, [None] * task.n_seats).engine
    if name == "default" and not natural:
        return eng.defaults(seat)
    if eng.engine == "abm":
        k = eng._k
        nat = dict(ABM_PROFILES[name]) if name != "default" else {}
        nat.update(natural)
        out: dict[str, float] = {}
        if "price" in nat:
            p = float(nat["price"])
            if p < 0:                                # relative to the seat's resting price
                d = eng.roles(seat)["levers"]["price"]
                dial = eng.defaults(seat)[d["public"]]
                frac = (100.0 - dial) / 100.0 if d["flip"] else dial / 100.0
                rest = d["natural_lo"] + frac * (d["natural_hi"] - d["natural_lo"])
                out["price"] = -p * rest
            else:
                out["price"] = p * k.p0
        if "quality" in nat:
            out["quality"] = float(nat["quality"])
        if "outreach" in nat:
            out["outreach"] = float(nat["outreach"]) * k.out_hi
        return eng.encode(out, base=eng.defaults(seat))
    if eng.engine == "techsys":
        from ..mechanisms.techsys import reference_settings
        return dict(eng.defaults(seat), **reference_settings(eng.config, name)[seat])
    raise ValueError(f"no profiles for engine {eng.engine!r}")


def profile_action(task: Task, seat: int, name: str, spec: EnvSpec | None = None, **natural: float) -> np.ndarray:
    """A profile as an action vector in [-1, 1] (the `research.learner` action layout)."""
    spec = spec or EnvSpec()
    eng = _sim(task, [None] * task.n_seats).engine
    st = profile_settings(task, seat, name, **natural)
    out = np.zeros(spec.act_dim, dtype=np.float32)
    for j, lv in enumerate(eng.levers(seat)):
        span = lv.hi - lv.lo
        out[j] = 0.0 if span <= 0 else 2.0 * (st[lv.name] - lv.lo) / span - 1.0
    return np.clip(out, -1.0, 1.0)


def profile_individual(task: Task, seat: int, name: str, spec: EnvSpec | None = None, id: str | None = None,
                       funds_scale: float = 1.0, **natural: float) -> Individual:
    """A privileged individual that plays a profile in one seat of `task`."""
    spec = spec or EnvSpec()
    act = profile_action(task, seat, name, spec, **natural)
    cfg = learner_config("fixed", 0, action=[float(v) for v in act], label=name)
    ind = Individual(id or f"profile:{name}:s{seat}", FixedLearner(spec, 0, action=act, label=name), "own", seat=seat,
                     funds_scale=funds_scale, tags=("profile", name), privileged=True, world=task.id, config=cfg)
    ind.learner.freeze(True)
    return ind


def natural_view(task: Task, seat: int, action: Sequence[float]) -> dict[str, float]:
    """Operator-side reading of an action vector: for markets the price as a multiple of p0, the effort and the
    outreach as a fraction of its range; for technical systems the lever kinds with their values. Used only to
    describe learned policies in reports."""
    eng = _sim(task, [None] * task.n_seats).engine
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    levers = eng.levers(seat)
    val = {lv.name: lv.lo + (a[j] + 1.0) * 0.5 * (lv.hi - lv.lo) for j, lv in enumerate(levers)}
    if eng.engine == "abm":
        out = {}
        for role, d in eng.roles(seat)["levers"].items():
            if role.startswith("noop"):
                continue
            frac = val[d["public"]] / 100.0
            if d["flip"]:
                frac = 1.0 - frac
            nat = d["natural_lo"] + frac * (d["natural_hi"] - d["natural_lo"])
            out[role] = float(nat / eng._k.p0) if role == "price" else float(frac)
        return out
    if eng.engine == "techsys":
        out = {}
        for lv in eng.config["levers"]:
            if lv["name"] in val:
                key = f"{lv['kind']}{''.join(str(t) for t in lv['targets'])}"
                out[key] = float(val[lv["name"]])
        return out
    return {k: float(v) for k, v in val.items()}
