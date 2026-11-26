"""`make(tier, name, seed, reward_mode=None, **kw)`: one constructor for every world a learner can be put in.

    tier 1  "mech:<family>"     a 4.0 hidden-mechanism task (`research.tasks.make_task(family, seed)`) through
                                `MechExperienceEnv`; ``validate=False`` builds an unvalidated training task
            "rapid:<family>"    a v5 Tier 1 family through `crucible.rapid.make` (workstream H)
            "<family>"          rapid if `crucible.rapid` knows it, else a mechanism family
    tier 2  "<domain>[:<config>]" or "d5_<domain>_<config>"   a v5 domain (`crucible.domains.base.Domain`, the
                                canonical contract by decision D-17): the world is `domains.base.spec(domain, config,
                                seed, overrides)`; the learner occupies the domain's participant seat
                                (`Domain.participant_principal`, else a new participant); the domain's public
                                objective (`Domain.objective(world, principal, difficulty)`) is the agent's objective
                                and `Domain.evaluate(world, principal, objective)` the independent evaluator. The
                                domain's scripted `reference` is evaluator-side and never reachable.
    tier 3  "<profile>"         a v5 civilisation: `crucible.civ.generate(profile, seed, population=...,
                                overrides=...) -> World` with ``crucible.civ.PROFILES`` (workstream J; see
                                `CIV_ENTRY_POINT`)
    tier 0  "<world family>"    any 4.0 world family or WorldSpec through `KernelExperienceEnv` (not one of the
                                three tiers; used for the 4.0 open worlds and tests)

`reward_mode` defaults to ``observation_only`` (Clarification 2, D-19); ``legacy_shaped_reward`` and
``verified_terminal_outcome`` exist only when named. The same `ExperienceEnv` methods, observation record and action
shape come back in every case. A tier whose package is not merged yet raises `TierUnavailable` naming what is
missing - it never falls back silently to another world type.
"""
from __future__ import annotations

import importlib
from typing import Any

from .modes import check_mode, evaluator_access

# What the experience interface expects of workstream J's Tier 3 package (D-17). Tier 2 domains will become
# configurations of the same generator; until then they come from `crucible.domains.base`.
CIV_ENTRY_POINT = {
    "function": "crucible.civ.generate(profile: str, seed: int, *, population: str = 'development', "
                "overrides: dict | None = None) -> crucible.core.kernel.World",
    "profiles": "crucible.civ.PROFILES: dict[str, ...] (names accepted by generate)",
    "overrides": "the experience registry passes {'params': {'evaluator_access': 'evaluator' | 'participants'}} "
                 "and expects them merged into the world spec's params before World.create (so replay and "
                 "regeneration keep the observation-only policy); a generator that ignores them is adopted with a "
                 "runtime override (crucible.experience.modes.prepare_world)",
    "participants": "optional crucible.civ.participant_seats(world) -> list[str] of principals a learner may occupy; "
                    "otherwise the registry admits new participants (admin.join)",
    "evaluator": "optional crucible.civ.evaluate(world, principal) -> dict, evaluator side only",
}


class TierUnavailable(LookupError):
    """The requested tier or world is not available in this checkout (its workstream has not landed)."""


def _rapid():
    try:
        mod = importlib.import_module("crucible.rapid")
    except ImportError:
        return None
    return mod if hasattr(mod, "make") else None


def domains(include_fixtures: bool = False) -> dict[str, Any]:
    """v5 Tier 2 domains present in this checkout: name -> `crucible.domains.base.Domain` (every domain module of
    `crucible.domains` registers itself with `base.register` at import; `load_all()` imports them). Only the Part 9
    domains count as Tier 2 (`Domain.is_part9`): test fixtures (`fixture=True`, e.g. tests/v5/_domain_fixture.py) and
    test domains outside Part 9 are left out unless `include_fixtures` - so a fixture registered by another test in
    the same process never stands in for a real domain in a gate."""
    try:
        pkg = importlib.import_module("crucible.domains")
        base = importlib.import_module("crucible.domains.base")
    except ImportError:
        return {}
    load = getattr(pkg, "load_all", None)
    if callable(load):
        load()
    reg = dict(sorted(getattr(base, "REGISTRY", {}).items()))
    if include_fixtures:
        return reg
    return {n: d for n, d in reg.items() if (d.is_part9() if hasattr(d, "is_part9") else not getattr(d, "fixture",
                                                                                                         False))}


def civ_available() -> bool:
    try:
        mod = importlib.import_module("crucible.civ")
    except ImportError:
        return False
    return callable(getattr(mod, "generate", None))


def civ_profiles() -> list[str]:
    if not civ_available():
        return []
    mod = importlib.import_module("crucible.civ")
    return sorted(getattr(mod, "PROFILES", {}) or [])


def tiers_available() -> dict[int, bool]:
    from ..mechanisms.base import MECH_FAMILIES  # noqa: F401 - Tier 1 always exists (4.0 mechanisms)
    return {1: True, 2: bool(domains()), 3: civ_available()}


def domain_and_config(name: str, doms: dict | None = None) -> tuple[Any, Any]:
    """(Domain, DomainConfig) for ``"<domain>"``, ``"<domain>:<config>"`` or ``"d5_<domain>_<config>"``; a bare
    domain name means its first training configuration. Any registered domain can be built by name (fixtures
    included); only `domains()` decides what counts as Tier 2."""
    doms = domains(include_fixtures=True) if doms is None else doms
    if not doms:
        raise TierUnavailable("no v5 Tier 2 domain is present (crucible.domains.base.REGISTRY is empty; "
                              "workstream I / the civilisation generator, D-17)")
    if name.startswith("d5_"):
        rest = name[3:]
        for dn in sorted(doms, key=len, reverse=True):
            if rest.startswith(dn + "_"):
                try:
                    return doms[dn], doms[dn].config(rest[len(dn) + 1:])
                except KeyError as e:
                    raise TierUnavailable(str(e)) from e
        raise TierUnavailable(f"no v5 domain for world family {name!r}; domains: {sorted(doms)}")
    dn, _, cfg = name.partition(":")
    if dn not in doms:
        raise TierUnavailable(f"no v5 domain {dn!r}; domains present: {sorted(doms)}")
    dom = doms[dn]
    if not cfg:
        train = dom.split("train")
        if not train:
            raise TierUnavailable(f"domain {dn!r} has no training configuration")
        return dom, train[0]
    try:
        return dom, dom.config(cfg)
    except KeyError as e:
        raise TierUnavailable(str(e)) from e


def make(tier: int, name: str, seed: int = 0, reward_mode: str | None = None, **kw):
    """Build an `ExperienceEnv`. See the module docstring for names; `kw` goes to the adapter (agents, endowment,
    objectives, validate, difficulty, overrides, ...)."""
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
        return _make_domain(name, int(seed), mode, kw)
    if tier == 3:
        return _make_civ(name, int(seed), mode, kw)
    if tier == 0:
        world = kw.pop("spec", None)
        if world is None:
            world = {"family": name, "overrides": kw.pop("overrides")} if "overrides" in kw else name
        return KernelExperienceEnv(world, int(seed), reward_mode=mode, tier=0, name=name, **kw)
    raise ValueError(f"tier must be 0, 1, 2 or 3, got {tier}")


# ====================================================================== Tier 2
def _make_domain(name: str, seed: int, mode: str, kw: dict):
    from ..core.kernel import World
    from ..domains import base
    from .kernel import KernelExperienceEnv
    from .modes import prepare_spec
    dom, cfg = domain_and_config(name)
    difficulty = int(kw.pop("difficulty", 1))
    spec = prepare_spec(base.spec(dom.name, cfg.name, seed, kw.pop("overrides", None)), mode)
    # learner-side modules (as Tier 3 worlds get them, `LEARNER_MODULES`): self-generated experiments and self-set
    # goals read only what the agent may read and pay nothing (OO.11); recorded in the spec, so replay keeps them
    have = {m["name"] for m in spec.modules}
    for m in kw.pop("learner_modules", LEARNER_MODULES) or ():
        if m not in have:
            spec.modules.append({"name": m, "config": {}})
            have.add(m)
    import crucible
    crucible.load_all_modules()
    from ..api.local import ensure_modules
    ensure_modules([m["name"] for m in spec.modules])
    world = World.create(spec, store=kw.pop("store", None))
    agents = _domain_agents(dom, world, kw.pop("agents", None))
    env = KernelExperienceEnv(world, seed, reward_mode=mode, tier=2, name=f"{dom.name}:{cfg.name}", agents=agents,
                              **kw)
    attach_domain(env, dom, difficulty)
    env.domain, env.domain_config = dom.name, cfg.name
    return env


def _domain_agents(dom, world, agents):
    """Who a Tier 2 learner is (workstream I hook, D-48): by default every seat of the domain (the first is the
    primary learner; several seats make a multi-learner world), else the domain's `participant_principal`, else a
    new participant. An int n means the first n seats (n new participants when the domain has fewer seats);
    explicit names are seats or principals ("p:...") or handles of new participants."""
    seats = list(dom.seat_principals()) if hasattr(dom, "seat_principals") else []
    if isinstance(agents, int):
        return seats[:agents] if seats and agents <= len(seats) else agents
    if agents is not None:
        return list(agents)
    if seats:
        return seats
    if dom.participant_principal:
        return [dom.participant_principal(world)]
    return ["agent"]


class DomainObjective:
    """A v5 domain's objective behind the begin/step_reward/status protocol. The domain's own evaluator is the
    verifier: its verdict stays on the evaluator side (`status` is read only by `verify()` and the reward gate);
    the participant sees the objective's public text. The legacy reward (explicit legacy mode only) is the change of
    the evaluator's numeric score, the verified outcome its success. The evaluator runs at most once per world
    state (event sequence number and clock) and principal."""

    def __init__(self, dom, objective: dict):
        self.dom, self.objective = dom, dict(objective)
        text = objective.get("description") or objective.get("text") or objective.get("objective")
        self.text = text if isinstance(text, str) else None
        self._score: dict[str, float] = {}
        self._last: dict[str, dict] = {}
        self._at: dict[str, tuple] = {}

    def _eval(self, world, actor) -> dict:
        key = (int(world.seq), int(world.now))
        hit = self._at.get(actor)
        if hit is not None and hit[0] == key:
            return hit[1]
        ev = self.dom.evaluate(world, actor, self.objective)
        ev = ev if isinstance(ev, dict) else {"value": ev}
        self._at[actor] = (key, ev)
        return ev

    @staticmethod
    def _num(ev: dict) -> float:
        for k in ("score", "value", "progress"):
            v = ev.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 1.0 if ev.get("success") else 0.0

    def begin(self, world, actor):
        self._last[actor] = self._eval(world, actor)
        self._score[actor] = self._num(self._last[actor])

    def step_reward(self, world, actor) -> float:
        ev = self._eval(world, actor)
        self._last[actor] = ev
        s = self._num(ev)
        r, self._score[actor] = s - self._score.get(actor, s), s
        return r

    def status(self, world, actor) -> dict:
        ev = self._last.get(actor) or {}
        return {"success": bool(ev.get("success")), "failed": bool(ev.get("failed", False)),
                "progress": self._score.get(actor), "evaluation": ev}


def attach_domain(env, dom, difficulty: int = 1) -> None:
    """Give every agent of `env` the domain's public objective (evaluated per principal) and make the domain's
    evaluator part of `env.verify()`."""
    objectives = {}
    for a in env.agents():
        objectives[a] = dom.objective(env.world, a, difficulty)
        env.set_objective(a, DomainObjective(dom, objectives[a]))
    env.evaluator = lambda w, a: dom.evaluate(w, a, objectives[a])


# ====================================================================== Tier 3
# Learner-side modules every Tier 3 world gets (as KernelExperienceEnv adds `exp` to Tier 0 worlds): self-generated
# experiments (`exp`) and self-set learning goals (`goals`, without its evaluator tool `goals.check_now` in an
# observation-only world). They read only what the agent itself may read and pay nothing (Clarification 2, OO.11).
LEARNER_MODULES = ("goals", "exp")


def civ_world(name: str, seed: int, *, population: str = "development", overrides: dict | None = None,
              mode: str | None = None, learner_modules=LEARNER_MODULES, store=None):
    """A Tier 3 civilisation world: `crucible.civ.generate(name, seed, population=..., overrides=...)` with the
    evaluator-access policy of `mode` merged into the spec params (recorded in the spec, so replay and regeneration
    keep it). When learner-side modules are requested and the civilisation lacks them, the world is built from
    `crucible.civ.spec(...)` (the WorldSpec `generate` creates) plus those modules, with `generate`'s world-id rule."""
    from ..core.kernel import World
    civ = importlib.import_module("crucible.civ")
    ov = dict(overrides or {})
    ov["params"] = dict(ov.get("params") or {}, evaluator_access=evaluator_access(check_mode(mode)))
    add = [m for m in (learner_modules or ())]
    spec_fn = getattr(civ, "spec", None)
    if not add or not callable(spec_fn):
        return civ.generate(name, int(seed), population=population, overrides=ov, store=store)
    sp = spec_fn(name, int(seed), population, overrides=ov)
    have = {m["name"] for m in sp.modules}
    extra = [m for m in add if m not in have]
    if not extra:
        return civ.generate(name, int(seed), population=population, overrides=ov, store=store)
    import crucible
    crucible.load_all_modules()
    from ..api.local import ensure_modules
    for m in extra:
        sp.modules.append({"name": m, "config": {}})
    ensure_modules([m["name"] for m in sp.modules])
    return World.create(sp, world_id=f"civ-{sp.hash()[:12]}", store=store)


def _make_civ(name: str, seed: int, mode: str, kw: dict):
    from .kernel import KernelExperienceEnv
    if not civ_available():
        raise TierUnavailable("no v5 Tier 3 civilisation generator (crucible.civ.generate, workstream J); expected "
                              f"entry point: {CIV_ENTRY_POINT['function']}")
    civ = importlib.import_module("crucible.civ")
    profiles = getattr(civ, "PROFILES", None)
    if profiles is not None and name not in profiles:
        raise TierUnavailable(f"no civilisation profile {name!r}; profiles: {sorted(profiles)}")
    population = kw.pop("population", kw.pop("population_profile", "development"))
    world = civ_world(name, int(seed), population=population, overrides=kw.pop("overrides", None), mode=mode,
                      learner_modules=kw.pop("learner_modules", LEARNER_MODULES), store=kw.pop("store", None))
    agents = kw.pop("agents", None)
    if agents is None:
        seats = getattr(civ, "participant_seats", None)
        agents = list(seats(world)) if callable(seats) else ["agent"]
    ev = getattr(civ, "evaluate", None)
    return KernelExperienceEnv(world, int(seed), reward_mode=mode, tier=3, name=f"civ:{name}", agents=agents,
                               evaluator=kw.pop("evaluator", None) or (ev if callable(ev) else None), **kw)
