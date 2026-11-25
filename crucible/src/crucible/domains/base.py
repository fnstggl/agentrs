"""The Tier 2 domain framework (INTERFACE_CONTRACTS section 9).

A domain is a family of standalone semantic digital worlds (execution mode B) with:

* configurations: at least three structurally different training configurations, at least one compositionally
  unfamiliar and at least one structurally unfamiliar evaluation configuration. A configuration's `structure` states
  what is structurally different about it (which mechanisms, couplings, topologies, participant types); two
  configurations of a domain may not share a structure, and changing only numbers is not a new configuration;
* `build(config, rng, overrides)`: the world recipe, in the 4.0 world-family form {"modules", "params",
  "module_list"}; each configuration becomes the world family `d5_<domain>_<config>` with split `train`,
  `v5eval_compositional` or `v5eval_structural` (never 4.0's `val`/`test`);
* `objective(world, principal, difficulty)`: the public objective given to a participant (no hidden data);
* `evaluate(world, principal, objective)`: the independent, evaluator-side outcome measurement;
* `reference(world, principal, objective)`: a scripted reference policy that shows the objective is feasible. It is
  evaluator-side: never registered as a tool, never named in an observation, never a required solution path;
* `mechanics(world)`: for each of the domain's required mechanics (MASTER_SPEC Part 9), whether the built world
  contains and exercises it, measured on the world, not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.canonical import digest
from ..core.kernel import World, WorldSpec

SPLITS = {"train": "train", "eval_compositional": "v5eval_compositional", "eval_structural": "v5eval_structural"}


@dataclass(frozen=True)
class DomainConfig:
    name: str
    split: str                      # train | eval_compositional | eval_structural
    structure: dict                 # what is structurally different about this configuration
    doc: str
    overrides: dict = field(default_factory=dict)


@dataclass
class Domain:
    name: str
    title: str
    spec_part: str                  # e.g. "9.1"
    required_mechanics: tuple[str, ...]
    configs: tuple[DomainConfig, ...]
    build: Callable[[DomainConfig, Any, dict], dict]
    objective: Callable[[World, str, int], dict]
    evaluate: Callable[[World, str, dict], dict]
    reference: Callable[[World, str, dict], Any]
    mechanics: Callable[[World], dict]
    participant_principal: Callable[[World], str] | None = None   # the seat a learner or reference occupies
    doc: str = ""

    def family(self, config: str) -> str:
        return f"d5_{self.name}_{config}"

    def config(self, name: str) -> DomainConfig:
        for c in self.configs:
            if c.name == name:
                return c
        raise KeyError(f"domain {self.name}: no configuration {name!r}")

    def split(self, split: str) -> list[DomainConfig]:
        return [c for c in self.configs if c.split == split]


REGISTRY: dict[str, Domain] = {}


def validate(d: Domain) -> list[str]:
    """Structural problems with a domain definition (empty when it may be registered)."""
    out = []
    if len(d.split("train")) < 3:
        out.append("needs at least three training configurations")
    if not d.split("eval_compositional"):
        out.append("needs a compositionally unfamiliar evaluation configuration")
    if not d.split("eval_structural"):
        out.append("needs a structurally unfamiliar evaluation configuration")
    names = [c.name for c in d.configs]
    if len(names) != len(set(names)):
        out.append("configuration names repeat")
    for c in d.configs:
        if c.split not in SPLITS:
            out.append(f"{c.name}: unknown split {c.split!r}")
        if not c.structure:
            out.append(f"{c.name}: no structure stated")
    structs = [digest(c.structure) for c in d.configs]
    if len(structs) != len(set(structs)):
        out.append("two configurations have the same structure (numbers alone do not make a configuration)")
    if not d.required_mechanics:
        out.append("no required mechanics listed")
    return out


def register(d: Domain) -> Domain:
    """Register a domain and its world families. Raises on an invalid definition."""
    from ..worldgen.generator import FAMILIES, Family
    problems = validate(d)
    if problems:
        raise ValueError(f"domain {d.name}: " + "; ".join(problems))
    REGISTRY[d.name] = d
    for c in d.configs:
        def build(rng, overrides, _c=c, _d=d):
            ov = dict(_c.overrides)
            ov.update(overrides or {})
            r = _d.build(_c, rng, ov)
            r.setdefault("params", {})
            r["params"] = dict(r["params"], domain=_d.name, domain_config=_c.name)
            return r
        FAMILIES[d.family(c.name)] = Family(d.family(c.name), SPLITS[c.split], build,
                                            f"{d.title} - {c.name}: {c.doc}")
    return d


def spec(domain: str, config: str, seed: int = 0, overrides: dict | None = None) -> WorldSpec:
    from ..worldgen.generator import generate
    d = REGISTRY[domain]
    d.config(config)
    return generate({"family": d.family(config), "overrides": dict(overrides or {})}, seed)


def world(domain: str, config: str, seed: int = 0, overrides: dict | None = None, store=None) -> World:
    return World.create(spec(domain, config, seed, overrides), store=store)


def run_policy(world: World, principal: str, policy, *, days: float, dt: int = 3600, seed: int = 0) -> dict:
    """Drive one policy for `days` simulated days with the 4.0 runner; returns the runner's summary."""
    from ..agents.base import Runner
    rounds = int(round(days * 86400 / dt))
    return Runner(world, [(principal, policy)], dt=dt, seed=seed).run(rounds)


def run_reference(domain: str, config: str, seed: int = 0, *, days: float, difficulty: int = 1, dt: int = 3600,
                  overrides: dict | None = None) -> dict:
    """Build a world, give the reference policy the domain's objective, run it, and evaluate the outcome."""
    d = REGISTRY[domain]
    w = world(domain, config, seed, overrides)
    who = d.participant_principal(w) if d.participant_principal else "agent"
    obj = d.objective(w, who, difficulty)
    pol = d.reference(w, who, obj)
    run = run_policy(w, who, pol, days=days, dt=dt, seed=seed)
    out = d.evaluate(w, who, obj)
    return {"domain": domain, "config": config, "seed": seed, "objective": obj, "outcome": out, "run": run,
            "mechanics": d.mechanics(w), "events": w.seq, "scripted_reference": True}


def summary() -> dict:
    return {n: {"title": d.title, "spec_part": d.spec_part, "configs": {c.name: c.split for c in d.configs},
                "required_mechanics": list(d.required_mechanics)} for n, d in sorted(REGISTRY.items())}
