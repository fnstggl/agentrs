"""Baseline learners behind the architecture-neutral contract (`research.learner.Learner`).

An instrument that compares learning systems needs learners that span the range from "does nothing" to "learns
across worlds". Everything here works through the contract alone (observation, action, reward; no lever names, no
private task data), so every learner has the same information and permissions:

    name            kind          weights  learns across episodes   what it is
    hold            idle          no       no     never touches a lever (score 0 by construction)
    random          random        no       no     uniformly random settings
    probe_exploit   scripted      no       no     one-lever-at-a-time experiments from the defaults, commit to the
                                                  best combination, re-probe when net collapses
    hill_climb      scripted      no       no     stochastic finite-difference ascent (SPSA) on net, keeps the best
                                                  iterate
    ppo_gru         model_free    yes      yes    recurrent PPO; the recurrent state spans the whole episode, the
                                                  network treats lever and gauge slots symmetrically
    ppo_gru_flat    model_free    yes      yes    the same algorithm with a flat GRU over the padded observation
                                                  (the architecture comparison)
    model_based     model_based   yes      yes    Bayesian regression of revenue and cost on lever features fitted
                                                  inside the episode, planning by search with Thompson sampling;
                                                  the feature map, prior and noise model are meta-learned across
                                                  episodes
    evo_probe       search        yes      yes    cross-entropy (population) search over the parameters of the
                                                  probe-and-commit procedure

The registry is the public surface::

    from crucible.research import learners
    learners.available()                       # names
    learners.describe("ppo_gru")               # kind, has_weights, learns_across_episodes, defaults, doc
    lr = learners.make("ppo_gru", spec, seed=0, device="cpu", hidden=64)

Torch is imported only when a learner that needs it is made.
"""
from __future__ import annotations

from typing import Callable

from ..learner import EnvSpec, Learner

_REGISTRY: dict[str, dict] = {}


def register(name: str, factory: Callable[..., Learner], *, kind: str, has_weights: bool,
             learns_across_episodes: bool, defaults: dict | Callable[[], dict] | None = None, doc: str = "",
             adapts_within_episode: bool = True) -> None:
    """Add a learner. `factory(spec, seed, device, **kwargs)` builds it; `defaults` are its keyword defaults (a
    dict, or a callable returning one so that heavy modules are imported only on demand)."""
    if name in _REGISTRY:
        raise ValueError(f"learner {name!r} is already registered")
    _REGISTRY[name] = {"factory": factory, "kind": kind, "has_weights": bool(has_weights),
                       "learns_across_episodes": bool(learns_across_episodes),
                       "adapts_within_episode": bool(adapts_within_episode), "defaults": defaults or {}, "doc": doc}


def available() -> list[str]:
    return sorted(_REGISTRY)


def describe(name: str) -> dict:
    e = _entry(name)
    d = e["defaults"]
    return {"name": name, "kind": e["kind"], "has_weights": e["has_weights"],
            "learns_across_episodes": e["learns_across_episodes"],
            "adapts_within_episode": e["adapts_within_episode"], "defaults": dict(d() if callable(d) else d),
            "doc": e["doc"]}


def make(name: str, spec: EnvSpec | None = None, seed: int = 0, device: str = "cpu", **kwargs) -> Learner:
    """Build a registered learner. Unknown keyword arguments are an error (they would silently change nothing)."""
    e = _entry(name)
    d = e["defaults"]
    known = d() if callable(d) else d
    bad = sorted(set(kwargs) - set(known))
    if bad:
        raise TypeError(f"learner {name!r} has no option(s) {bad}; options: {sorted(known)}")
    lr = e["factory"](spec or EnvSpec(), int(seed), device, **kwargs)
    lr.name = name
    return lr


def _entry(name: str) -> dict:
    if name not in _REGISTRY:
        raise KeyError(f"unknown learner {name!r}; available: {available()}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------- the built-in learners
def _hold(spec, seed, device, **kw):
    from ..baselines import HoldLearner
    return HoldLearner(spec, seed, device, **kw)


def _random(spec, seed, device, **kw):
    from ..baselines import RandomLearner
    return RandomLearner(spec, seed, device, **kw)


def _probe(spec, seed, device, **kw):
    from .scripted import ProbeExploitLearner
    return ProbeExploitLearner(spec, seed, device, **kw)


def _hill(spec, seed, device, **kw):
    from .scripted import HillClimbLearner
    return HillClimbLearner(spec, seed, device, **kw)


def _ppo(arch: str):
    def build(spec, seed, device, **kw):
        from .ppo import PPOGRULearner
        kw.setdefault("arch", arch)
        return PPOGRULearner(spec, seed, device, **kw)
    return build


def _ppo_defaults(arch: str):
    def defaults():
        from .ppo import PPO_DEFAULTS
        return dict(PPO_DEFAULTS, arch=arch)
    return defaults


def _mb(spec, seed, device, **kw):
    from .model_based import ModelBasedLearner
    return ModelBasedLearner(spec, seed, device, **kw)


def _mb_defaults():
    from .model_based import MB_DEFAULTS
    return dict(MB_DEFAULTS)


def _evo(spec, seed, device, **kw):
    from .evo import EvoProbeLearner
    return EvoProbeLearner(spec, seed, device, **kw)


def _evo_defaults():
    from .evo import EVO_DEFAULTS
    return dict(EVO_DEFAULTS)


def _scripted_defaults(which: str):
    def defaults():
        from . import scripted
        return dict(getattr(scripted, which))
    return defaults


register("hold", _hold, kind="idle", has_weights=False, learns_across_episodes=False, adapts_within_episode=False,
         doc="Leaves every lever where it is: the zero of the normalised score.")
register("random", _random, kind="random", has_weights=False, learns_across_episodes=False,
         adapts_within_episode=False, defaults={"hold": 1},
         doc="Uniformly random settings, redrawn every `hold` ticks.")
register("probe_exploit", _probe, kind="scripted", has_weights=False, learns_across_episodes=False,
         defaults=_scripted_defaults("PROBE_DEFAULTS"),
         doc="One-lever-at-a-time experiments from the defaults (a fixed number of ticks per arm), then the best "
             "combination is checked against the baseline and held; probing restarts when net collapses.")
register("hill_climb", _hill, kind="scripted", has_weights=False, learns_across_episodes=False,
         defaults=_scripted_defaults("HILL_DEFAULTS"),
         doc="Simultaneous-perturbation stochastic approximation (SPSA) on net with a normalised step; the best "
             "iterate is held for the last part of the episode.")
register("ppo_gru", _ppo("set"), kind="model_free", has_weights=True, learns_across_episodes=True,
         defaults=_ppo_defaults("set"),
         doc="Recurrent PPO (clipped surrogate, GAE, whole-episode backpropagation through time). Shared "
             "per-lever and per-gauge recurrent encoders with pooling: slots are treated symmetrically.")
register("ppo_gru_flat", _ppo("flat"), kind="model_free", has_weights=True, learns_across_episodes=True,
         defaults=_ppo_defaults("flat"),
         doc="The same recurrent PPO with one flat GRU over the padded observation (architecture comparison).")
register("model_based", _mb, kind="model_based", has_weights=True, learns_across_episodes=True, defaults=_mb_defaults,
         doc="Within an episode: Bayesian linear regression of revenue and cost on per-lever features, settings "
             "chosen by coordinate search on a posterior sample (Thompson sampling). Across episodes: the feature "
             "network, the prior over weights and the noise model are trained by maximising the posterior "
             "predictive likelihood of held-out transitions (meta-learned Bayesian last layer).")
register("evo_probe", _evo, kind="search", has_weights=True, learns_across_episodes=True, defaults=_evo_defaults,
         doc="Cross-entropy method over the parameters of the probe-and-commit procedure: a population of "
             "parameter vectors is scored by episode returns on the training stream.")
