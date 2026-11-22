"""Interface every learning algorithm implements, and a registry.  [candidate-editable]

An algorithm receives finished episodes (the decision records written by the agent, plus the evaluator's summary)
and changes the controller's weights. On-policy algorithms use only the episodes of the current iteration;
algorithms with ``uses_replay = True`` also keep episodes in a replay buffer between iterations.
"""
from __future__ import annotations

REGISTRY: dict = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


class Algorithm:
    name = "base"
    uses_replay = False

    def __init__(self, model, cfg: dict | None = None, device: str = "cpu"):
        self.model, self.cfg, self.device = model, dict(cfg or {}), device

    def update(self, episodes: list[dict], store, demos: list[dict] | None = None) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, s: dict) -> None:
        pass


def make(name: str, model, cfg: dict | None = None, device: str = "cpu") -> Algorithm:
    from . import continual, ppo, sil  # noqa: F401  (registers the built-in algorithms)
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](model, cfg, device)
