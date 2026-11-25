"""`make(tier, name, seed, reward_mode, **kw)`: one constructor for every world a learner can be put in.

    tier 1  "mech:<family>"     a 4.0 hidden-mechanism task (`research.tasks.make_task(family, seed)`) through
                                `MechExperienceEnv`; ``validate=False`` builds an unvalidated training task
            "rapid:<family>"    a v5 Tier 1 family through `crucible.rapid.make` (workstream H)
            "<family>"          rapid if `crucible.rapid` knows it, else a mechanism family
    tier 2  "<domain>[:<config>]" or "d5_<domain>_<config>"   a v5 domain (`crucible.domains.<domain>.DOMAIN`,
                                workstream I): `DOMAIN.build(config, seed)` -> kernel world; the domain's objective
                                and evaluator become the agents' objective and the independent verifier
    tier 3  "<profile>"         a v5 civilisation (`crucible.civ.generate(profile, seed)`, workstream J)
    tier 0  "<world family>"    any 4.0 world family or WorldSpec through `KernelExperienceEnv` (not one of the
                                three tiers; used for the 4.0 open worlds and tests)

The same `ExperienceEnv` methods, observation record and action shape come back in every case. A tier whose
package is not merged yet raises `TierUnavailable` naming what is missing - it never falls back silently to another
world type.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .modes import LEGACY, check_mode


class TierUnavailable(LookupError):
    """The requested tier or world is not available in this checkout (its workstream has not landed)."""


def _rapid():
    try:
        mod = importlib.import_module("crucible.rapid")
    except ImportError:
        return None
    return mod if hasattr(mod, "make") else None


def domains() -> dict[str, Any]:
    """v5 Tier 2 domains present in this checkout: name -> DOMAIN object (modules of `crucible.domains` that define
    a module-level ``DOMAIN``)."""
    out = {}
    try:
        pkg = importlib.import_module("crucible.domains")
    except ImportError:
        return out
    for info in sorted(pkgutil.iter_modules(getattr(pkg, "__path__", [])), key=lambda i: i.name):
        if info.ispkg or info.name.startswith("_") or info.name == "modules":
            continue
        try:
            mod = importlib.import_module(f"crucible.domains.{info.name}")
        except ImportError:
            continue
        dom = getattr(mod, "DOMAIN", None)
        if dom is not None:
            out[getattr(dom, "name", info.name)] = dom
    return out


def civ_available() -> bool:
    try:
        mod = importlib.import_module("crucible.civ")
    except ImportError:
        return False
    return callable(getattr(mod, "generate", None))


def tiers_available() -> dict[int, bool]:
    from ..mechanisms.base import MECH_FAMILIES  # noqa: F401 - Tier 1 always exists (4.0 mechanisms)
    return {1: True, 2: bool(domains()), 3: civ_available()}


def _domain_and_config(name: str, doms: dict) -> tuple[Any, Any]:
    if name.startswith("d5_"):
        rest = name[3:]
        for dn in sorted(doms, key=len, reverse=True):
            if rest.startswith(dn + "_"):
                return doms[dn], rest[len(dn) + 1:]
        raise TierUnavailable(f"no v5 domain for world family {name!r}; domains: {sorted(doms)}")
    dn, _, cfg = name.partition(":")
    if dn not in doms:
        raise TierUnavailable(f"no v5 domain {dn!r}; domains present: {sorted(doms) or 'none (workstream I)'}")
    dom = doms[dn]
    if not cfg:
        cfg = (getattr(dom, "configs", {}) or {}).get("train", [None])[0]
    return dom, cfg


def _config_by_name(dom, cfg):
    if not isinstance(cfg, str):
        return cfg
    for split in (getattr(dom, "configs", {}) or {}).values():
        for c in split:
            if c == cfg or (isinstance(c, dict) and c.get("name") == cfg):
                return c
    return cfg


def make(tier: int, name: str, seed: int = 0, reward_mode: str = LEGACY, **kw):
    """Build an `ExperienceEnv`. See the module docstring for names; `kw` goes to the adapter (agents, endowment,
    objectives, validate, ...)."""
    mode = check_mode(reward_mode)
    tier = int(tier)
    if tier == 1:
        from .tier1 import MechExperienceEnv, RapidExperienceEnv
        kind, _, fam = name.partition(":") if ":" in name else ("", "", name)
        rapid = _rapid()
        if kind == "rapid" or (not kind and rapid is not None and fam in getattr(getattr(rapid, "families", None),
                                                                              "REGISTRY", {})):
            if rapid is None:
                raise TierUnavailable("crucible.rapid (workstream H) is not merged: rapid families are unavailable")
            return RapidExperienceEnv(fam, seed, reward_mode=mode, **kw)
        from ..research.tasks import make_task
        validate = bool(kw.pop("validate", True))
        task = kw.pop("task", None) or make_task(fam, int(seed), validate=validate)
        return MechExperienceEnv(task, reward_mode=mode, **kw)
    from .kernel import KernelExperienceEnv
    if tier == 2:
        doms = domains()
        if not doms:
            raise TierUnavailable("no v5 Tier 2 domain is present (crucible.domains, workstream I, not merged)")
        dom, cfg = _domain_and_config(name, doms)
        cfg = _config_by_name(dom, cfg)
        difficulty = kw.pop("difficulty", 0.5)
        spec = dom.build(cfg, int(seed))
        label = cfg.get("name") if isinstance(cfg, dict) else cfg
        env = KernelExperienceEnv(spec, int(seed), reward_mode=mode, tier=2, name=f"{dom.name}:{label}", **kw)
        _attach_domain(env, dom, difficulty)
        return env
    if tier == 3:
        if not civ_available():
            raise TierUnavailable("no v5 Tier 3 civilisation generator (crucible.civ.generate, workstream J)")
        civ = importlib.import_module("crucible.civ")
        pop = kw.pop("population_profile", "standard")
        spec = civ.generate(name, int(seed), pop)
        return KernelExperienceEnv(spec, int(seed), reward_mode=mode, tier=3, name=f"civ:{name}", **kw)
    if tier == 0:
        return KernelExperienceEnv(kw.pop("spec", None) or name, int(seed), reward_mode=mode, tier=0, name=name, **kw)
    raise ValueError(f"tier must be 0, 1, 2 or 3, got {tier}")


class _DomainObjective:
    """A v5 domain's objective behind the begin/step_reward/status protocol: the domain's own evaluator is the
    verifier; the legacy shaped reward is the change of its score, the terminal outcome its success."""

    def __init__(self, dom, objective: dict):
        self.dom, self.objective = dom, objective
        self.text = objective.get("description") or objective.get("text")
        self._score: dict[str, float] = {}
        self._last: dict[str, dict] = {}

    def begin(self, world, actor):
        self._last[actor] = self.dom.evaluate(world, actor, self.objective)
        self._score[actor] = float(self._last[actor].get("score") or 0.0)

    def step_reward(self, world, actor) -> float:
        ev = self.dom.evaluate(world, actor, self.objective)
        self._last[actor] = ev
        s = float(ev.get("score") or 0.0)
        r, self._score[actor] = s - self._score[actor], s
        return r

    def status(self, world, actor) -> dict:
        ev = self._last.get(actor) or {}
        return {"success": bool(ev.get("success")), "failed": False, "score": ev.get("score"),
                "details": ev.get("details")}


def _attach_domain(env, dom, difficulty) -> None:
    objective = dom.objective(env.world, difficulty)
    for a in env.agents():
        env.set_objective(a, _DomainObjective(dom, objective))
