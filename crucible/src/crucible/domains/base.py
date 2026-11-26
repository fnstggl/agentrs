"""The Tier 2 domain framework: the one contract of a Crucible 5.0 domain (D-17, D-48) and the measurements that
check it on BUILT and RUN worlds (Gate N, SP-9.0, SP-9.16). The authoring guide is docs/v5/components/I.md.

A domain (`Domain`) is a family of worlds for one Part 9 subject. Each of its configurations is

* a civilisation recipe restricted to the subsystems, mechanism variants, couplings, topology and population the
  configuration needs (`civ_recipe`, built on `crucible.civ.recipe`; execution mode C composition) - or, where the
  spec allows a standalone semantic world (mode B), any recipe in the same 4.0 world-family form;
* plus the participant seats a learner (or the scripted reference) occupies (`Seat`; module `seat`, created at
  genesis, several per world);
* plus a public objective (`objective(world, principal, difficulty) -> {"text": ..., ...}`: everything in it is
  public; the learner reads its text) and an evaluator-side outcome evaluator (`evaluate(world, principal,
  objective) -> {"success": bool, "score": float, "details": {...}}`), never reachable by participants: in the default
  ``observation_only`` mode (Clarification 2, D-19) the Gym reward is 0.0 and the verdict reaches only `verify()`;
* plus a hidden feasible reference (`reference(principal, objective) -> policy`): a scripted policy that acts ONLY
  through the learner interface (it is handed observations and tool results, never the World), never registered as
  a tool, never named in an observation, never a required solution path (N.n1), and labelled scripted in evidence;
* plus a null baseline (default `Idle`) that must fail, and degenerate-strategy probes (`Probe`) that must fail too
  (SP-9.16: feasible, not guaranteed);
* plus the recorded-event evidence of every required mechanic of its Part 9 section (`Mechanic`) and of its
  autonomous participants (`Actor`), counted by the framework in the event log of integrated runs.

Configurations are procedurally generated instances per seed (`spec(domain, config, seed)`); their STRUCTURE is
measured on the built world (`fingerprint`), never taken from the declaration: module set, couplings, participant
types, graph shapes and the domain's own measured structure, with every magnitude reduced to presence/absence
("numbers alone do not make a configuration"). A domain needs >= 3 structurally distinct training configurations,
>= 1 compositionally unfamiliar evaluation configuration (every structural atom seen in training, the combination
never) and >= 1 structurally unfamiliar one (at least one structural atom no training configuration has).

`validate(d)` is the cheap definition check run by `register`; `check(d)` builds and runs the domain and returns
every Gate N / SP-9.0 / SP-9.16 measurement (the gate tests in tests/v5 iterate over `part9_domains()`).
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import re
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..core.canonical import digest
from ..core.kernel import World, WorldSpec

API_VERSION = "domains-framework/2"

SPLITS = {"train": "train", "eval_compositional": "v5eval_compositional", "eval_structural": "v5eval_structural"}
DAY, HOUR = 86400, 3600

# MASTER_SPEC Part 9: the fifteen domains and their required mechanics, verbatim (order as in the spec).
PART9: dict[str, dict] = {
    "9.1": {"title": "Startup economy", "mechanics": (
        "Product creation", "Demand discovery", "Independent customers", "Subscription billing", "Distribution",
        "Competition", "Retention", "Costs", "Profitability", "Business operations")},
    "9.2": {"title": "Software ecosystem", "mechanics": (
        "Executable code", "Repositories", "Dependencies", "Issues", "Independent testing", "Developer preferences",
        "Documentation", "Adoption", "Compatibility", "Maintenance")},
    "9.3": {"title": "Content ecosystem", "mechanics": (
        "Actual content artifacts", "Learned content representations", "Autonomous consumption",
        "Recommendation systems", "Competing creators", "Social sharing", "Audience feedback",
        "Distinct verified viewers")},
    "9.4": {"title": "Community ecosystem", "mechanics": (
        "Social networks", "Persistent memberships", "Independent participants", "Shared activities", "Communication",
        "Moderation", "Governance", "Retention", "Community evolution")},
    "9.5": {"title": "Scientific research", "mechanics": (
        "Experimental apparatus", "Hidden laws", "Measurement noise", "Experimental costs", "Data collection",
        "Hypothesis testing", "Independent replication", "Novel-condition evaluation")},
    "9.6": {"title": "Technology invention", "mechanics": (
        "Executable inventions", "Technical constraints", "Independent evaluation", "Adoption", "Compatibility",
        "Economic consequences", "Downstream resource changes")},
    "9.7": {"title": "Engineering infrastructure", "mechanics": (
        "Compute", "Networking", "Workloads", "Capacity", "Energy", "Failures", "Technical optimization",
        "Delayed effects", "Deployment of new controllers and software")},
    "9.8": {"title": "Industrial economy", "mechanics": (
        "Raw materials", "Production", "Energy", "Logistics", "Inventory", "Equipment", "Labor", "Market demand",
        "Resource scarcity", "Supply disruptions")},
    "9.9": {"title": "Labor marketplace", "mechanics": (
        "Worker skills", "Preferences", "Compensation", "Recruitment", "Delegation", "Project dependencies",
        "Performance", "Retention", "Independent agency")},
    "9.10": {"title": "Education ecosystem", "mechanics": (
        "Learner knowledge state", "Educational artifacts", "Prerequisites", "Participation", "Learning dynamics",
        "Independent assessments", "Retention of acquired knowledge")},
    "9.11": {"title": "Information ecosystem", "mechanics": (
        "Search", "Documents", "Knowledge artifacts", "Queries", "Source provenance", "Information quality",
        "Updating facts", "Independent user satisfaction", "Verified retrieval performance")},
    "9.12": {"title": "Organizational ecosystem", "mechanics": (
        "Organizations", "Differing goals", "Contracts", "Negotiation", "Coordination", "Shared resources",
        "Governance", "Commitments", "Collective outcomes")},
    "9.13": {"title": "Digital marketplace", "mechanics": (
        "Two-sided participation", "Discovery", "Transactions", "Trust", "Quality", "Disputes", "Pricing",
        "Network effects", "Retention")},
    "9.14": {"title": "Crisis recovery", "mechanics": (
        "Cascading disruptions", "Scarce resources", "Prioritization", "Coordination", "Recovery", "Resilience",
        "Independent service-restoration metrics")},
    "9.15": {"title": "Research commercialization", "mechanics": (
        "Research", "Experimentation", "Technical development", "Product creation", "Financing", "Deployment",
        "Independent adoption", "Commercial sustainability")},
}
N_PART9 = len(PART9)


# ============================================================================================ the contract
@dataclass(frozen=True)
class DomainConfig:
    """One configuration. `doc` is its independent characterisation (what world it is and why it differs);
    `structure` is the DECLARED structural intent (documentation, shown by the research registry) - the framework
    measures structure on the built world and never trusts this field; `overrides` go to `Domain.build`."""
    name: str
    split: str                      # train | eval_compositional | eval_structural
    structure: dict
    doc: str
    overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Seat:
    """A participant seat: an ordinary person with a checking account funded at genesis (module `seat`). A learner
    occupies a seat through `crucible.experience.make(2, ...)`; several seats make a multi-learner world."""
    handle: str
    endowment: int = 1_000_000      # cents
    name: str = ""
    doc: str = ""


@dataclass(frozen=True)
class Mechanic:
    """Recorded-event evidence of one required mechanic: the event types whose occurrence after genesis in an
    integrated run exercises it. A pattern is an exact type (``sci.reading``), a prefix (``sci.*``) or a column
    batch kind (``cols.batch:goods.session``, ``cols.batch:goods.*``). `count(world, since_seq) -> int` may add a
    measured count from recorded state (rows written by appliers after `since_seq`); it is never a declaration."""
    name: str
    events: tuple[str, ...] = ()
    doc: str = ""
    count: Callable[[World, int], int] | None = None


@dataclass(frozen=True)
class Actor:
    """An autonomous participant kind of the domain (population individuals, firms, labs, tenants, ...) and the
    recorded events its own decisions produce. Measured in a null-baseline run: the seats do nothing, so every such
    event is autonomous activity."""
    name: str
    events: tuple[str, ...] = ()
    doc: str = ""
    count: Callable[[World, int], int] | None = None


@dataclass(frozen=True)
class Probe:
    """A degenerate strategy for the objective (SP-9.16): `policy(principal, objective) -> policy object`. The
    evaluator must NOT report success for it on the probed instances."""
    name: str
    doc: str
    policy: Callable[[str, dict], Any]


@dataclass
class Domain:
    name: str
    title: str
    spec_part: str                  # "9.1" .. "9.15" (anything else: not counted by Gate N)
    required_mechanics: tuple[str, ...]   # () = the Part 9 list of `spec_part`
    configs: tuple[DomainConfig, ...]
    build: Callable[[DomainConfig, int, dict], dict]          # (config, seed, overrides) -> recipe
    objective: Callable[[World, str, int], dict]              # public objective {"text": ..., ...}
    evaluate: Callable[[World, str, dict], dict]              # evaluator side {"success", "score", "details"}
    reference: Callable[..., Any]                             # (principal, objective) -> tool-only policy
    mechanics: Any = ()                                       # tuple[Mechanic] | dict[str, Mechanic|tuple]
    participant_principal: Callable[[World], str] | None = None   # default: the first seat
    doc: str = ""
    seats: tuple[Seat, ...] = ()
    actors: tuple[Actor, ...] = ()
    null_baseline: Callable[[str, dict], Any] | None = None   # default: Idle
    degenerate: tuple[Probe, ...] = ()
    structure: Callable[[World], dict] | None = None          # domain-measured discrete structure of a built world
    tier3: Callable[[DomainConfig, dict], dict] | None = None  # merge this configuration into a civ recipe
    mode: str = "C"                 # C: a restriction of the civilisation generator; B: standalone semantic world
    horizon_days: float = 30.0      # evaluation window of the objective
    slowest_effect_days: float = 0.0   # slowest delayed effect of the world (the horizon must exceed it)
    dt: int = HOUR                  # simulated seconds between a policy's decision points
    difficulties: tuple[int, ...] = (1, 2, 3)
    reference_difficulty: int = 1
    reference_seeds: tuple[int, ...] = (0, 1)
    reference_configs: tuple[str, ...] = ()   # default: the first training configuration
    sources: tuple[str, ...] = ()   # extra source files of the domain's mechanisms (dice scan, SP-9.16)
    fixture: bool = False           # a test fixture (never counted by the gates)

    def __post_init__(self):
        if not self.required_mechanics and self.spec_part in PART9:
            self.required_mechanics = tuple(PART9[self.spec_part]["mechanics"])
        self.required_mechanics = tuple(self.required_mechanics)
        self.configs = tuple(self.configs)

    # ---------------------------------------------------------------- names
    def family(self, config: str) -> str:
        return f"d5_{self.name}_{config}"

    def config(self, name: str) -> DomainConfig:
        for c in self.configs:
            if c.name == name:
                return c
        raise KeyError(f"domain {self.name}: no configuration {name!r}")

    def split(self, split: str) -> list[DomainConfig]:
        return [c for c in self.configs if c.split == split]

    # ---------------------------------------------------------------- normalised views
    def mechanic_probes(self) -> dict[str, Mechanic] | None:
        """{mechanic name: Mechanic}; None when the domain only self-reports (a callable `mechanics`)."""
        m = self.mechanics
        if callable(m) and not isinstance(m, (tuple, list, dict)):
            return None
        out: dict[str, Mechanic] = {}
        items = m.items() if isinstance(m, dict) else ((x.name, x) for x in m)
        for name, v in items:
            out[name] = v if isinstance(v, Mechanic) else Mechanic(name, tuple(v))
        return out

    def seat_principals(self) -> list[str]:
        return [f"p:{s.handle}" for s in self.seats]

    def principal_of(self, world: World) -> str | None:
        if self.participant_principal is not None:
            return self.participant_principal(world)
        return f"p:{self.seats[0].handle}" if self.seats else None

    def reference_policy(self, principal: str, objective: dict):
        return _call_policy_factory(self.reference, principal, objective)

    def null_policy(self, principal: str, objective: dict):
        return _call_policy_factory(self.null_baseline, principal, objective) if self.null_baseline else Idle()

    def is_part9(self) -> bool:
        return self.spec_part in PART9 and not self.fixture


def _call_policy_factory(fn, principal: str, objective: dict):
    """A policy factory gets (principal, objective). For callers of the earlier 3-argument contract
    (world, principal, objective) the world is dropped here: a reference never receives the World."""
    try:
        n = len([p for p in inspect.signature(fn).parameters.values()
                 if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        n = 2
    if n >= 3:
        return fn(None, principal, objective)
    return fn(principal, objective)


REGISTRY: dict[str, Domain] = {}


# ============================================================================================ static validation
def validate(d: Domain) -> list[str]:
    """Definition problems (empty when the domain may be registered). Cheap: nothing is built. Structure is checked
    on built worlds by `check` (`structure_report`)."""
    out = []
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", d.name or ""):
        out.append("name must match [a-z][a-z0-9_]{1,30}")
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
        if not (c.doc or "").strip():
            out.append(f"{c.name}: no characterisation (doc)")
    structs = [digest(c.structure) for c in d.configs]
    if len(structs) != len(set(structs)):
        out.append("two configurations have the same structure (numbers alone do not make a configuration)")
    if not d.required_mechanics:
        out.append("no required mechanics listed")
    if d.spec_part in PART9 and tuple(d.required_mechanics) != tuple(PART9[d.spec_part]["mechanics"]):
        out.append(f"required mechanics differ from MASTER_SPEC Part {d.spec_part}: "
                   f"{list(PART9[d.spec_part]['mechanics'])}")
    probes = d.mechanic_probes()
    if (d.spec_part in PART9 and not d.fixture) or probes:
        if probes is None:
            out.append("mechanics must be recorded-event probes (Mechanic), not a self-report")
        else:
            missing = [m for m in d.required_mechanics if m not in probes or not (probes[m].events or probes[m].count)]
            extra = [m for m in probes if m not in d.required_mechanics]
            if missing:
                out.append(f"no event probe for required mechanics {missing}")
            if extra:
                out.append(f"event probes for mechanics the spec does not list: {extra}")
    if d.mode not in ("B", "C"):
        out.append("mode must be 'B' (standalone semantic world) or 'C' (restriction of the civilisation generator)")
    if d.horizon_days <= 0 or d.horizon_days < d.slowest_effect_days:
        out.append("the evaluation horizon must be positive and exceed the slowest delayed effect")
    handles = [s.handle for s in d.seats]
    if len(handles) != len(set(handles)):
        out.append("seat handles repeat")
    if not callable(d.reference):
        out.append("no reference policy factory")
    if d.spec_part in PART9 and not d.fixture:
        if not d.seats and d.participant_principal is None:
            out.append("no participant seat")
        if not d.degenerate:
            out.append("no degenerate-strategy probe (SP-9.16)")
        if not d.actors:
            out.append("no autonomous participant kind (Actor) declared")
    return out


def register(d: Domain, *, replace: bool = True) -> Domain:
    """Register a domain. Raises ValueError on an invalid definition. Registration is all a domain needs to be
    reachable through `crucible.experience.make(2, '<domain>:<config>')`, mode B (`crucible.domains.modes`), the
    research distribution registry and the gates."""
    problems = validate(d)
    if problems:
        raise ValueError(f"domain {d.name}: " + "; ".join(problems))
    if d.name in REGISTRY and not replace and REGISTRY[d.name] is not d:
        raise ValueError(f"domain {d.name} is already registered")
    REGISTRY[d.name] = d
    _CACHE.clear()
    return d


def unregister(name: str) -> None:
    REGISTRY.pop(name, None)
    _CACHE.clear()


def part9_domains() -> dict[str, Domain]:
    """The registered Part 9 domains (fixtures and non-Part-9 test domains excluded), after importing every domain
    module of the package."""
    from . import load_all
    load_all()
    return {n: d for n, d in sorted(REGISTRY.items()) if d.is_part9()}


# ============================================================================================ recipes and specs
ALWAYS = ("identity", "bank", "admin", "artifacts", "cols", "space")


def _ensure(names) -> None:
    import crucible
    from ..core.kernel import MODULES
    crucible.load_all_modules()
    missing = [n for n in names if n not in MODULES]
    if missing:
        from ..api.local import ensure_modules
        ensure_modules(missing)


def _closure(names: Iterable[str]) -> list[str]:
    from ..core.kernel import MODULES
    out: list[str] = []

    def add(m):
        if m in out:
            return
        if m not in MODULES:
            raise ValueError(f"unknown kernel module {m!r}")
        for dep in MODULES[m].depends:
            add(dep)
        out.append(m)
    for m in names:
        add(m)
    return out


def civ_recipe(seed: int, *, subsystems: Iterable[str] = (), modules: Iterable[str] = (), drop: Iterable[str] = (),
               population: str = "development", population_modules: Iterable[str] = ("pop",),
               profile: Any = "balanced", module_config: dict | None = None, params: dict | None = None,
               couplings: Iterable[str] | None = None) -> dict:
    """A civilisation recipe (`crucible.civ.recipe`) restricted to what a domain configuration needs (D-17).

    Kept: the substrate (`ALWAYS`), the named Mode C `subsystems` (`crucible.civ.subsystems.SUBSYSTEMS`), the
    population modules (`population_modules`; () for a world without the population), the extra `modules` (modules
    the civilisation does not compose, e.g. a domain's own kernel module, are added with `module_config`), and every
    declared dependency of these. Removed: `drop`. `couplings` restricts `civlinks` to those relations (None: the
    profile's). The result is the 4.0 world-family form plus `subsystems`, `profile` and `kept` for inspection.
    Undeclared runtime needs of a module (an index another module provides) show up as failed processes in
    `check`; add that module to `modules`."""
    from .. import civ
    from ..civ import subsystems as SUB
    _ensure(["identity"])
    modules = list(modules)
    full = civ.recipe(profile, int(seed), population=population)
    want: list[str] = list(ALWAYS)
    for s in subsystems:
        if s not in SUB.SUBSYSTEMS:
            raise KeyError(f"unknown civilisation subsystem {s!r}; known: {sorted(SUB.SUBSYSTEMS)}")
        want += [m for m in SUB.SUBSYSTEMS[s]["modules"] if m in full["module_list"] or m in modules]
    want += list(population_modules) + modules
    _ensure(want)
    keep = _closure(want)
    dropped = set(drop)
    mods = [m for m in keep if m not in dropped]
    cfgs = {m: copy.deepcopy(full["modules"].get(m, {})) for m in mods}
    for m, c in (module_config or {}).items():
        cfgs[m] = {**cfgs.get(m, {}), **copy.deepcopy(c)}
    if "civlinks" in cfgs and couplings is not None:
        allowed = set(couplings)
        cfgs["civlinks"]["couplings"] = [c for c in cfgs["civlinks"].get("couplings", []) if c in allowed]
    return {"module_list": mods, "modules": cfgs, "params": {**full["params"], **(params or {})},
            "subsystems": [s for s in SUB.SUBSYSTEMS if any(m in mods for m in SUB.SUBSYSTEMS[s]["modules"])],
            "profile": full["profile"], "population": population, "kept": sorted(mods)}


def ordered_spec(recipe: dict, family: str, seed: int, params: dict | None = None) -> WorldSpec:
    """A WorldSpec from a recipe: modules in dependency order (every declared dependency must be in the recipe)."""
    names = list(recipe.get("module_list") or list(recipe["modules"]))
    _ensure(names)
    from ..core.kernel import MODULES
    order: list[str] = []

    def add(m):
        if m in order or m not in names:
            return
        for dep in MODULES[m].depends:
            add(dep)
        order.append(m)
    for m in names:
        add(m)
    missing = sorted({dep for m in order for dep in MODULES[m].depends if dep not in order})
    if missing:
        raise ValueError(f"recipe modules miss dependencies {missing}")
    p = dict(recipe.get("params") or {})
    p.update(params or {})
    return WorldSpec(family, int(seed), [{"name": m, "config": copy.deepcopy(recipe["modules"].get(m, {}))}
                                         for m in order], p)


def recipe(domain: str | Domain, config: str, seed: int = 0, overrides: dict | None = None) -> dict:
    """The recipe of one configuration instance with the framework's additions (seats, domain params)."""
    d = REGISTRY[domain] if isinstance(domain, str) else domain
    c = d.config(config)
    ov = copy.deepcopy(dict(c.overrides))
    extra = copy.deepcopy(dict(overrides or {}))
    post_params = extra.pop("params", {}) or {}
    post_cfg = extra.pop("module_config", {}) or {}
    post_drop = set(extra.pop("drop_modules", []) or [])
    ov.update(extra)
    r = d.build(c, int(seed), ov)
    if not isinstance(r, dict) or "modules" not in r:
        raise TypeError(f"domain {d.name}: build must return a recipe dict with 'modules'")
    r = copy.deepcopy(r)
    mods = list(r.get("module_list") or list(r["modules"]))
    if d.seats:
        if "seat" not in mods:
            mods.append("seat")
        seats = [{"handle": s.handle, "name": s.name or s.handle, "endowment": int(s.endowment), "kind": "agent"}
                 for s in d.seats]
        r["modules"]["seat"] = {**r["modules"].get("seat", {}), "seats": seats}
    for m, cfg in post_cfg.items():
        r["modules"][m] = {**r["modules"].get(m, {}), **cfg}
    r["module_list"] = [m for m in mods if m not in post_drop]
    r["params"] = dict(r.get("params") or {}, domain=d.name, domain_config=c.name, domain_split=c.split,
                       family_split=SPLITS.get(c.split, c.split), **post_params)
    return r


def spec(domain: str, config: str, seed: int = 0, overrides: dict | None = None) -> WorldSpec:
    """The WorldSpec of instance `seed` of a configuration. `overrides`: ``params`` and ``module_config`` are merged
    after the build, ``drop_modules`` removes modules, every other key goes to `Domain.build`."""
    d = REGISTRY[domain]
    r = recipe(d, config, seed, overrides)
    return ordered_spec(r, d.family(config), seed)


def world(domain: str, config: str, seed: int = 0, overrides: dict | None = None, store=None) -> World:
    return World.create(spec(domain, config, seed, overrides), world_id=f"d5-{domain}-{config}-{seed}", store=store)


def tier3_spec(domain: str | Domain, config: str, seed: int = 0, profile: str = "balanced",
               population: str = "development") -> WorldSpec:
    """The configuration composed into a Tier 3 civilisation: the civilisation's recipe, plus the configuration's
    modules the civilisation does not have (with their configuration), plus the seats; `Domain.tier3` may adjust
    the merged recipe. Modules both have keep the civilisation's configuration."""
    from .. import civ
    d = REGISTRY[domain] if isinstance(domain, str) else domain
    c = d.config(config)
    mine = recipe(d, config, seed)
    full = civ.recipe(profile, int(seed), population=population)
    mods = list(full["module_list"])
    cfgs = copy.deepcopy(full["modules"])
    for m in mine["module_list"]:
        if m not in mods:
            mods.append(m)
            cfgs[m] = copy.deepcopy(mine["modules"].get(m, {}))
    merged = {"module_list": mods, "modules": cfgs,
              "params": dict(full["params"], domain=d.name, domain_config=c.name, domain_tier3=profile)}
    if d.tier3 is not None:
        merged = d.tier3(c, merged)
    return ordered_spec(merged, f"civ_{profile}+{d.name}_{c.name}", seed)


# ============================================================================================ policies
class Idle:
    """The null baseline: never acts."""
    kind = "null"
    name = "idle"

    def act(self, obs, tools):
        return None

    def observe(self, reward, result, next_obs, done):
        pass


class ScriptedReference:
    """Base of tool-only scripted references. A reference sees exactly what a learner sees: the observation
    (`obs`, a `crucible.experience` Observation) and the results of its own actions. It never holds the World.

    Subclasses implement `plan(obs) -> list[action]` (an action: ``{"tool": ..., "args": {...}}``, optionally with
    ``"tag"``, which the world never sees; build one with `act_`); the actions run in order, up to `burst` per
    decision point, and `on_result(tag, result, action)` receives each result (None when the call failed; the error
    is appended to `failures`). `plan` is called again when the queue is empty and `replan_s` simulated seconds
    have passed since the last plan."""
    kind = "scripted"
    name = "reference"
    burst = 8
    replan_s = 0

    def __init__(self, principal: str, objective: dict):
        self.me = principal
        self.objective = dict(objective or {})
        self.queue: list[dict] = []
        self.next_plan = -1
        self.failures: list[dict] = []
        self._pending: dict | None = None
        self.t = 0

    def plan(self, obs) -> list[dict]:
        raise NotImplementedError

    def on_result(self, tag: str | None, result: Any, action: dict) -> None:
        pass

    def act(self, obs, tools):
        self.t = int(obs["t"]) if isinstance(obs, dict) and "t" in obs else self.t
        if not self.queue and self.t >= self.next_plan:
            self.queue = list(self.plan(obs) or [])
            self.next_plan = self.t + int(self.replan_s)
        if not self.queue:
            return None
        a = self.queue.pop(0)
        self._pending = a
        return {"tool": a["tool"], "args": dict(a.get("args") or {})}

    def observe(self, reward, result, next_obs, done):
        a, self._pending = self._pending, None
        if a is None or result is None:
            return
        if not result.get("ok"):
            self.failures.append({"t": self.t, "tool": a["tool"], "error": result.get("error")})
            self.on_result(a.get("tag"), None, a)
            return
        self.on_result(a.get("tag"), result.get("result"), a)


def act_(tool: str, tag: str | None = None, **args) -> dict:
    """An action for `ScriptedReference.plan`."""
    out = {"tool": tool, "args": args}
    if tag:
        out["tag"] = tag
    return out


class RandomTools:
    """A degenerate generic strategy (SP-9.16): uniformly random tool calls with schema-filled arguments, drawn
    from its own seeded generator. It shows that acting at random does not achieve the objective."""
    kind = "random"
    name = "random_tools"
    burst = 4

    def __init__(self, principal: str, objective: dict, seed: int = 0):
        import numpy as np
        self.me = principal
        self.rng = np.random.default_rng([int(seed), int(digest(principal)[:8], 16)])

    def act(self, obs, tools):
        names = [t for t in tools if not str(t.get("name", "")).startswith(("admin.", "world."))]
        if not names:
            return None
        t = names[int(self.rng.integers(len(names)))]
        return {"tool": t["name"], "args": _fill_args(tool_params(t), self.rng, obs)}

    def observe(self, reward, result, next_obs, done):
        pass


def _fill_args(params: dict, rng, obs) -> dict:
    ids = sorted({e["id"] for e in (obs.get("entities") or []) if isinstance(e, dict) and "id" in e}) \
        if isinstance(obs, dict) else []
    out = {}
    for k, p in (params or {}).items():
        if not isinstance(p, dict):
            continue
        if "default" in p and rng.random() < 0.5:
            continue
        typ = p.get("type")
        typ = next((x for x in typ if x != "null"), None) if isinstance(typ, list) else typ
        if p.get("enum"):
            out[k] = p["enum"][int(rng.integers(len(p["enum"])))]
        elif typ in ("int", "integer"):
            out[k] = int(rng.integers(0, 1000))
        elif typ in ("float", "number"):
            out[k] = float(rng.uniform(0, 1000))
        elif typ in ("bool", "boolean"):
            out[k] = bool(rng.integers(2))
        elif typ in ("str", "string"):
            out[k] = ids[int(rng.integers(len(ids)))] if ids and rng.random() < 0.7 else f"x{int(rng.integers(99))}"
        elif typ in ("list", "array"):
            out[k] = []
        elif typ in ("dict", "object"):
            out[k] = {}
    return out


GENERIC_PROBES = (Probe("random_tools", "uniformly random tool calls with schema-filled arguments",
                        lambda principal, objective: RandomTools(principal, objective)),)


# ============================================================================================ driving
def drive(env, policies: dict[str, Any], *, days: float, dt: int = HOUR, max_burst: int | None = None,
          trace: bool = True) -> dict:
    """Run policies (``act(obs, tools) -> action | None``, ``observe(reward, result, next_obs, done)``) for
    `days` simulated days through an `ExperienceEnv`. At each decision point every policy acts on its own
    observation, up to its `burst` actions (each followed by the next observation), then time passes `dt` seconds.
    Policies see only observations and their own results. Returns per-agent counts and (when `trace`) each agent's
    tool sequence."""
    agents = [a for a in env.agents() if a in policies]
    steps = int(round(days * DAY / int(dt)))
    out = {a: {"actions": 0, "ok": 0, "rejected": 0, "tools": [] if trace else None, "errors": Counter()}
           for a in agents}
    t0 = time.perf_counter()
    for _ in range(steps):
        for a in agents:
            pol = policies[a]
            n = int(max_burst if max_burst is not None else getattr(pol, "burst", 1))
            obs = env.observe(a)
            for _k in range(max(1, n)):
                action = pol.act(obs, obs["tools"])
                if action is None:
                    break
                res = env.execute(a, action)
                o = out[a]
                o["actions"] += 1
                o["ok" if res.get("ok") else "rejected"] += 1
                if not res.get("ok"):
                    o["errors"][str((res.get("error") or {}).get("code") or "error")] += 1
                if trace:
                    o["tools"].append(str(action.get("tool")))
                obs = env.observe(a)
                if hasattr(pol, "observe"):
                    pol.observe(0.0, res, obs, False)
        env.advance(int(dt))
    for a in agents:
        out[a]["errors"] = dict(out[a]["errors"])
    return {"agents": out, "steps": steps, "wall_s": round(time.perf_counter() - t0, 3)}


def make_env(domain: str, config: str, seed: int = 0, *, reward_mode: str | None = None, agents=None,
             difficulty: int = 1, overrides: dict | None = None):
    """The learner's own path: `crucible.experience.make(2, '<domain>:<config>', seed)` (observation_only by
    default)."""
    from ..experience.registry import make
    kw: dict = {"difficulty": int(difficulty)}
    if agents is not None:
        kw["agents"] = agents
    if overrides:
        kw["overrides"] = overrides
    return make(2, f"{domain}:{config}", int(seed), reward_mode, **kw)


def run(domain: str, config: str, seed: int = 0, *, policy: str = "reference", days: float | None = None,
        difficulty: int | None = None, keep_env: bool = False) -> dict:
    """One evaluated run through the learner interface: the first seat is driven by the domain's reference
    (`policy="reference"`), its null baseline (``"null"``), a degenerate probe (its name) or a generic probe
    (``"random_tools"``); other seats idle. Returns the evaluator's outcome (evaluator side), the actions, the
    measured mechanics and autonomous activity after genesis, and the run's labels."""
    d = REGISTRY[domain]
    days = float(d.horizon_days if days is None else days)
    difficulty = int(d.reference_difficulty if difficulty is None else difficulty)
    env = make_env(domain, config, seed, difficulty=difficulty)
    w = env.world
    since = int(w.seq)
    seats = env.agents()
    me = seats[0]
    obj = d.objective(w, me, difficulty)
    if policy == "reference":
        pol = d.reference_policy(me, obj)
    elif policy == "null":
        pol = d.null_policy(me, obj)
    else:
        probes = {p.name: p for p in tuple(d.degenerate) + GENERIC_PROBES}
        if policy not in probes:
            raise KeyError(f"domain {domain}: no policy {policy!r}")
        pol = _call_policy_factory(probes[policy].policy, me, obj)
    pols = {me: pol, **{s: Idle() for s in seats[1:]}}
    dr = drive(env, pols, days=days, dt=d.dt)
    ver = env.verify()
    ev = (ver.get("evaluation") or {}).get(me) or {}
    out = {"domain": domain, "config": config, "seed": int(seed), "policy": policy, "days": days,
           "difficulty": difficulty, "objective": obj, "outcome": ev, "success": bool(ev.get("success")),
           "score": ev.get("score"), "drive": dr, "seats": seats, "events": int(w.seq), "since_seq": since,
           "invariant_violations": len(ver.get("invariant_violations") or []),
           "proc_failed": count_events(w, ("proc.failed",), since),
           "mechanics": mechanics_exercised(w, d, since), "actors": actors_active(w, d, since),
           "population": population_decisions(w, since), "reward_mode": env.reward_mode,
           "labels": {"scripted": policy == "reference", "policy_kind": getattr(pol, "kind", "unknown")}}
    if keep_env:
        out["env"] = env
    else:
        env.close()
    return out


# ============================================================================================ measurements
def _match(pattern: str, etype: str, ekind: str | None) -> bool:
    if pattern.startswith("cols.batch:"):
        if etype != "cols.batch" or ekind is None:
            return False
        k = pattern[len("cols.batch:"):]
        return ekind.startswith(k[:-1]) if k.endswith("*") else ekind == k
    if pattern.endswith(".*"):
        return etype.startswith(pattern[:-1])
    return etype == pattern


def count_events(world: World, patterns: Iterable[str], since_seq: int = 0) -> int:
    pats = tuple(patterns)
    if not pats:
        return 0
    n = 0
    for e in world.events_range(int(since_seq), world.seq):
        k = e.payload.get("kind") if e.type == "cols.batch" else None
        if any(_match(p, e.type, k) for p in pats):
            n += 1
    return n


def event_counts(world: World, since_seq: int = 0) -> Counter:
    """Event types after `since_seq`, plus ``cols.batch:<kind>`` for every column batch."""
    c: Counter = Counter()
    for e in world.events_range(int(since_seq), world.seq):
        c[e.type] += 1
        if e.type == "cols.batch":
            c[f"cols.batch:{e.payload.get('kind')}"] += 1
    return c


def _probe_count(world: World, probe, since_seq: int, counts: Counter) -> int:
    n = 0
    for p in probe.events:
        if p.startswith("cols.batch:"):
            if p.endswith("*"):
                n += sum(v for k, v in counts.items() if k.startswith(p[:-1]))
            else:
                n += int(counts.get(p, 0))
        elif p.endswith(".*"):
            n += sum(v for k, v in counts.items() if k.startswith(p[:-1]) and not k.startswith("cols.batch:"))
        else:
            n += int(counts.get(p, 0))
    if probe.count is not None:
        n += int(probe.count(world, int(since_seq)))
    return n


def mechanics_exercised(world: World, d: Domain, since_seq: int = 0) -> dict:
    """For every required mechanic: the recorded events (after `since_seq`) its probe counts. Self-reported
    mechanics (a callable) count as not exercised."""
    probes = d.mechanic_probes()
    counts = event_counts(world, since_seq)
    rows = {}
    for m in d.required_mechanics:
        pr = (probes or {}).get(m)
        rows[m] = int(_probe_count(world, pr, since_seq, counts)) if pr is not None else 0
    ex = sorted(m for m, n in rows.items() if n > 0)
    return {"listed": len(d.required_mechanics), "exercised": len(ex), "missing": sorted(set(rows) - set(ex)),
            "counts": rows, "self_reported": probes is None}


def actors_active(world: World, d: Domain, since_seq: int = 0) -> dict:
    counts = event_counts(world, since_seq)
    rows = {a.name: int(_probe_count(world, a, since_seq, counts)) for a in d.actors}
    return {"declared": len(rows), "active": sum(1 for n in rows.values() if n > 0), "counts": rows}


POP_INTERNAL = ("pop.", "attention.")


def population_decisions(world: World, since_seq: int = 0) -> dict:
    """Recorded decisions of the one population after `since_seq` (column batches with decision provenance,
    D-15): all of them, and those of activities other than the population's own bookkeeping."""
    total, domain_kinds = 0, Counter()
    models = set()
    for e in world.events_range(int(since_seq), world.seq):
        if e.type != "cols.batch":
            continue
        dec = (e.payload.get("extra") or {}).get("decision")
        kind = str(e.payload.get("kind") or "")
        if not dec or kind.endswith(".init"):
            continue
        total += 1
        models.add(str(dec.get("model_id")))
        if not kind.startswith(POP_INTERNAL):
            domain_kinds[kind] += 1
    return {"decisions": total, "activity_decisions": int(sum(domain_kinds.values())),
            "activity_kinds": dict(domain_kinds), "models": sorted(models)}


# ---------------------------------------------------------------------------------------------- structure
def discrete(x: Any) -> Any:
    """The structural form of a value: magnitudes become presence/absence ("zero"/"nonzero"), numeric sequences
    their set of degree classes (0, 1, 2, "3+"), other sequences their set of distinct elements; strings and
    booleans stay. Two values with equal discrete forms differ only in numbers."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return "zero" if x == 0 else "nonzero"
    if isinstance(x, dict):
        return {str(k): discrete(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple, set, frozenset)):
        xs = list(x)
        if xs and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xs):
            return sorted({("3+" if v >= 3 else str(int(v))) for v in xs})
        return sorted({json.dumps(discrete(v), sort_keys=True) for v in xs})
    return str(type(x).__name__)


def graph_shape(nodes: Iterable[str], edges: Iterable[tuple[str, str]], directed: bool = True) -> str:
    """A scale-invariant shape signature of a graph: the set of (in, out) degree classes, cycles, components."""
    nodes = sorted(set(nodes))
    edges = sorted({(a, b) for a, b in edges})
    if not directed:
        edges = sorted({tuple(sorted(e)) for e in edges})
    deg_in, deg_out = Counter(), Counter()
    adj: dict[str, set] = {n: set() for n in nodes}
    for a, b in edges:
        deg_out[a] += 1
        deg_in[b] += 1
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    cls = lambda v: "3+" if v >= 3 else str(v)  # noqa: E731
    pats = sorted({(cls(deg_in[n]), cls(deg_out[n])) if directed else (cls(deg_in[n] + deg_out[n]),) for n in adj})
    seen, comps = set(), 0
    for n in adj:
        if n in seen:
            continue
        comps += 1
        stack = [n]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x] - seen)
    cyclic = len(edges) > len(adj) - comps          # an (undirected) cycle exists
    return json.dumps({"degrees": pats, "cyclic": cyclic, "components": "one" if comps <= 1 else "many"})


def participant_types(world: World) -> list[str]:
    """Kinds of participants of a built world: principal kind with person/org kind or role, and the population."""
    st = world.state
    out = set()
    for pid, p in (st.scan("principals") if "principals" in st.tables else []):
        role = ""
        for t in ("persons", "orgs"):
            if t in st.tables and pid in st.tables[t]:
                r = st.tables[t][pid]
                role = str(r.get("kind") or (r.get("meta") or {}).get("role") or "")
        out.add(f"{p.get('kind')}:{role}")
    if "pop" in world.modules:
        out.add("population:individual")
    return sorted(out)


def fingerprint(world: World, d: Domain | None = None) -> dict:
    """The discrete structure of a BUILT world: module set, participant types, the civilisation structure
    (`crucible.civ.integration.fingerprint`: couplings, supply-chain and freight shapes, regimes) when civilisation
    modules are present, and the domain's own measured `structure(world)`; every value in discrete form."""
    fp: dict = {"modules": sorted(world.modules), "participant_types": participant_types(world)}
    if any(m in world.modules for m in ("civgen", "civlinks", "industrygen")):
        from ..civ import integration as I
        c = I.fingerprint(world)
        c.pop("modules", None)
        c["supply_chain"] = _supply_shape(world)
        c["freight"] = _freight_shape(world)
        fp["civ"] = c
    if d is not None and d.structure is not None:
        fp["domain"] = d.structure(world)
    return discrete(fp)


def _supply_shape(world: World) -> str:
    st = world.state
    designs = st.tables.get("designs", {}) if "designs" in st.tables else {}
    produces = {}
    for did, dd in designs.items():
        produces[f"i:{did}"] = dd["owner"]
        if dd.get("material"):
            produces[dd["material"]] = dd["owner"]
    nodes, edges = set(), set()
    for did, dd in designs.items():
        nodes.add(dd["owner"])
        for x in dd.get("inputs") or []:
            src = produces.get(x["sku"])
            if src and src != dd["owner"]:
                edges.add((src, dd["owner"]))
    return graph_shape(nodes, edges)


def _freight_shape(world: World) -> str:
    st = world.state
    regs = set(st.tables.get("regions", {})) if "regions" in st.tables else set()
    links = [(r["a"], r["b"]) for r in (st.tables.get("links", {}) if "links" in st.tables else {}).values()]
    return graph_shape(regs, links, directed=False)


def structure_digest(fp: dict) -> str:
    return digest(fp)


def atoms(fp: Any, path: str = "") -> set[tuple[str, str]]:
    """(path, value) atoms of a discrete fingerprint (a list contributes one atom per element)."""
    out: set[tuple[str, str]] = set()
    if isinstance(fp, dict):
        for k, v in fp.items():
            out |= atoms(v, f"{path}.{k}" if path else str(k))
    elif isinstance(fp, list):
        for v in fp:
            out.add((path, v if isinstance(v, str) else json.dumps(v, sort_keys=True)))
        if not fp:
            out.add((path, "[]"))
    else:
        out.add((path, json.dumps(fp, sort_keys=True)))
    return out


def unfamiliarity(train: list[dict], ev: dict) -> dict:
    """How an evaluation fingerprint relates to the training fingerprints: `novel_atoms` (structural elements no
    training configuration has: structurally unfamiliar when > 0) and `compositional` (every atom seen in
    training, the combination in none)."""
    tr = [atoms(f) for f in train]
    seen = set().union(*tr) if tr else set()
    a = atoms(ev)
    novel = sorted(a - seen)
    same = any(a == t for t in tr)
    return {"novel_atoms": [f"{p}={v}" for p, v in novel], "structural": bool(novel),
            "compositional": (not novel) and not same, "equal_to_training": same}


# ---------------------------------------------------------------------------------------------- open-ended actions
CREATION_CATEGORIES = ("file_creation", "content_creation", "code_execution", "program_deployment")


def tool_params(tool: dict) -> dict:
    """{name: JSON schema} of a tool's arguments (MCP `inputSchema.properties`, or 4.0 `params`)."""
    sch = tool.get("inputSchema") or {}
    props = sch.get("properties") if isinstance(sch, dict) else None
    return dict(props or tool.get("params") or {})


def _free_form(p: dict) -> bool:
    typ = p.get("type")
    types = set(typ) if isinstance(typ, list) else {typ}
    if p.get("enum"):
        return False
    if types & {"object", "dict"}:
        return True
    if types & {"string", "str"}:
        return int(p.get("maxLength") or 10**6) >= 200
    return False


def open_ended(tools: list[dict]) -> dict:
    """The seat's action space is open-ended when it is the universal tool interface, not a menu: >= 10 tools over
    >= 4 of the 13 MASTER_SPEC 11.2 categories, at least one authoring category (files, content, code,
    deployments), and at least one tool that takes free-form content (text of >= 200 characters or a structured
    object, e.g. files of a commit, an artifact, a post)."""
    from ..experience.features import tool_category
    cats = Counter(tool_category(t) for t in tools)
    free = sorted(t["name"] for t in tools if any(isinstance(p, dict) and _free_form(p)
                                                  for p in tool_params(t).values()))
    creation = sorted(c for c in cats if c in CREATION_CATEGORIES)
    named = [c for c in cats if c not in ("other",)]
    ok = len(tools) >= 10 and len(named) >= 4 and bool(creation) and bool(free)
    return {"tools": len(tools), "categories": sorted(cats), "creation_categories": creation,
            "free_form_tools": len(free), "free_form_examples": free[:8], "open_ended": ok}


# ---------------------------------------------------------------------------------------------- isolation (N.n1)
FORBIDDEN_ATTRS = frozenset({"state", "spec", "world", "_w", "modules", "tables", "system_call", "rng_for",
                             "events_range", "runtime", "hooks", "endpoints", "evaluate", "evaluator"})
FORBIDDEN_NAMES = frozenset({"World", "WorldSpec", "REGISTRY"})
_FRAMEWORK_POLICIES: tuple = ()


def _classes_of(obj) -> list[type]:
    out = []
    for c in type(obj).__mro__:
        if c is object or c in _FRAMEWORK_POLICIES:
            continue
        out.append(c)
    return out


def reads_world(obj_or_fn) -> list[str]:
    """Static check that a policy (its classes) or a factory reads nothing but observations: attribute accesses of
    world internals (`FORBIDDEN_ATTRS`), kernel names (`FORBIDDEN_NAMES`) and kernel imports, by AST. A target
    whose source cannot be read is reported (it cannot be checked)."""
    if obj_or_fn is None:
        return []
    targets = [obj_or_fn] if inspect.isfunction(obj_or_fn) or inspect.isclass(obj_or_fn) else _classes_of(obj_or_fn)
    hits = []
    for t in targets:
        name = getattr(t, "__qualname__", str(t))
        try:
            src = textwrap.dedent(inspect.getsource(t))
            tree = ast.parse(src)
        except (OSError, TypeError, SyntaxError):
            if getattr(t, "__module__", "") == "builtins":
                continue
            hits.append(f"{name}: source unavailable (cannot be checked)")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                hits.append(f"{name}:{node.lineno}: .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                hits.append(f"{name}:{node.lineno}: {node.id}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                if "kernel" in mod or "core.state" in mod or any("kernel" in n for n in names):
                    hits.append(f"{name}:{node.lineno}: imports the kernel")
    return hits


def reference_canaries(d: Domain, policy) -> list[str]:
    """Distinctive identifiers of a domain's reference: its policy classes, their module, the factory."""
    out = set()
    for c in _classes_of(policy):
        if len(c.__name__) >= 6:
            out.add(c.__name__)
        out.add(f"{c.__module__}.{c.__qualname__}")
    q = getattr(d.reference, "__qualname__", "")
    if q and "<lambda>" not in q and len(q) >= 6:
        out.add(q)
        out.add(f"{d.reference.__module__}.{q}")
    return sorted(out)


def exposure(texts: Iterable[str], canaries: Iterable[str], trajectory: list[str]) -> dict:
    """Reference exposure in participant-visible text: canary identifiers, or the reference's own ordered tool
    sequence (its first >= 3 distinct tools, in order) spelled out in one text."""
    can = [c for c in canaries if c]
    seq: list[str] = []
    for t in trajectory:
        if t not in seq:
            seq.append(t)
    seq = seq[:6]
    hits = []
    for tx in texts:
        s = str(tx)
        for c in can:
            if c in s:
                hits.append(f"canary {c!r}")
        if len(seq) >= 3:
            pos, found = 0, 0
            for tool in seq:
                i = s.find(tool, pos)
                if i < 0:
                    break
                pos, found = i + len(tool), found + 1
            if found >= 3:
                hits.append(f"ordered reference tool sequence {seq[:found]}")
    hits = sorted(set(hits))
    return {"hits": hits, "exposed": len(hits)}


def sandbox_can_import_crucible() -> bool:
    """Whether participant code in the sandbox could import the crucible package (and so a reference): the crucible
    source tree on the sandbox's readable paths or sys.path, or crucible among its approved packages."""
    import os
    import crucible
    from ..sandbox import runner
    root = os.path.realpath(os.path.dirname(crucible.__file__))
    cfg = runner.SandboxConfig()
    reads, syspath = runner._read_paths(cfg.packages)
    under = any(root == os.path.realpath(p) or root.startswith(os.path.realpath(p) + os.sep)
                for p in list(reads) + list(syspath))
    return bool(under or "crucible" in tuple(cfg.packages))


# ---------------------------------------------------------------------------------------------- dice (SP-9.16)
DICE_WORDS = re.compile(r"fail|success|succeed|error|break|crash|reject|accept|win|lose|luck|chance|outage|bug|"
                        r"\bok\b|\bpass(es|ed)?\b", re.I)
PRAGMA = re.compile(r"#\s*fidelity:\s*([A-Za-z0-9_.:\-]+)")


def _fidelity_ids() -> set[str]:
    import os
    import crucible
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(crucible.__file__))), "docs", "v5",
                     "FIDELITY_REGISTER.md")
    try:
        txt = open(p, encoding="utf-8").read()
    except OSError:
        return set()
    return set(re.findall(r"`([A-Za-z0-9_.:\-]+)`", txt)) | set(re.findall(r"\|\s*([A-Za-z0-9_.:\-]+)\s*\|", txt))


def _is_draw(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
        node.func.attr in ("random", "uniform", "rand", "random_sample")


def dice_scan(paths: Iterable[str]) -> dict:
    """Arbitrary failure probabilities in source files: Bernoulli draws (``rng.random() < p``, ``rng.binomial(1,
    p)``) whose expression or assigned name speaks of failure/success (or whose branch raises or fails) and that
    carry no ``# fidelity: <id>`` pragma (on the line or the one above) naming a FIDELITY_REGISTER record - an
    empirically grounded hazard. Difficulty must come from complexity, scarcity and uncertainty instead."""
    ids = None
    hits, exempt, files = [], [], 0
    for path in paths:
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError) as e:
            hits.append({"file": path, "line": 0, "code": f"unreadable: {type(e).__name__}"})
            continue
        files += 1
        lines = src.splitlines()
        parents = {}
        for p in ast.walk(tree):
            for ch in ast.iter_child_nodes(p):
                parents[ch] = p
        for node in ast.walk(tree):
            bern = None
            if isinstance(node, ast.Compare) and (_is_draw(node.left) or any(_is_draw(c) for c in node.comparators)):
                bern = node
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "binomial" and node.args and isinstance(node.args[0], ast.Constant) and \
                    node.args[0].value == 1:
                bern = node
            if bern is None:
                continue
            seg = ast.get_source_segment(src, bern) or ""
            ctx_txt = seg
            par = parents.get(bern)
            while par is not None and not isinstance(par, (ast.If, ast.IfExp, ast.Assign, ast.FunctionDef,
                                                           ast.Return)):
                par = parents.get(par)
            body_fail = False
            if isinstance(par, ast.If):
                body = "\n".join(ast.get_source_segment(src, b) or "" for b in par.body)
                body_fail = bool(re.search(r"\braise\b|\.fail\(|fail", body))
            elif isinstance(par, (ast.Assign, ast.IfExp, ast.Return)):
                ctx_txt += " " + (ast.get_source_segment(src, par) or "")
            if not (DICE_WORDS.search(ctx_txt) or body_fail):
                continue
            ln = bern.lineno
            m = PRAGMA.search(" ".join(lines[max(0, ln - 2):ln]))
            if m:
                ids = _fidelity_ids() if ids is None else ids
                if m.group(1) in ids:
                    exempt.append({"file": path, "line": ln, "fidelity": m.group(1)})
                    continue
            hits.append({"file": path, "line": ln, "code": seg[:120]})
    return {"files": files, "mechanisms": len(hits), "hits": hits, "exempt": exempt}


def domain_sources(d: Domain) -> list[str]:
    """Source files of a domain: the files defining its build, objective and evaluator, plus `sources` (relative
    paths are taken from crucible/domains/)."""
    import os
    out = []
    for fn in (d.build, d.evaluate, d.objective):
        try:
            f = inspect.getsourcefile(fn)
        except TypeError:
            f = None
        if f and f not in out:
            out.append(f)
    root = os.path.dirname(os.path.abspath(__file__))
    for s in d.sources:
        p = s if os.path.isabs(s) else os.path.join(root, s)
        if p not in out:
            out.append(p)
    return out


# ============================================================================================ the deep check
_CACHE: dict[tuple, Any] = {}


def cached_run(domain: str, config: str, seed: int, policy: str, days: float | None = None) -> dict:
    key = (domain, config, int(seed), policy, days)
    if key not in _CACHE:
        _CACHE[key] = run(domain, config, seed, policy=policy, days=days)
    return _CACHE[key]


def structure_report(d: Domain, seeds: Iterable[int] = (0, 1, 2)) -> dict:
    """Build every configuration on `seeds` (genesis only): fingerprints, structure digests (stable across seeds?),
    distinct instances per configuration, distinct training structures, and the unfamiliarity of each evaluation
    configuration (N.2, N.6, N.7, SP-9.0)."""
    seeds = [int(s) for s in seeds]
    fps, digs, inst, stable, errors = {}, {}, {}, {}, {}
    for c in d.configs:
        ds, ins = [], set()
        for s in seeds:
            try:
                w = world(d.name, c.name, s)
            except Exception as e:  # noqa: BLE001 - reported: a configuration that cannot be built counts as failing
                errors[f"{c.name}@{s}"] = f"{type(e).__name__}: {e}"[:300]
                continue
            fp = fingerprint(w, d)
            if c.name not in fps:
                fps[c.name] = fp
            ds.append(structure_digest(fp))
            ins.add(w.digest())
            w.close()
        digs[c.name] = ds[0] if ds else None
        stable[c.name] = len(set(ds)) == 1 and len(ds) == len(seeds)
        inst[c.name] = len(ins)
    train = [c.name for c in d.split("train") if c.name in fps]
    unf = {c.name: unfamiliarity([fps[t] for t in train], fps[c.name]) for c in d.configs
           if c.split != "train" and c.name in fps}
    same_pairs = []
    names = [c.name for c in d.configs if digs.get(c.name)]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if digs[a] == digs[b]:
                same_pairs.append(f"{a}={b}")
    return {"fingerprints": fps, "digests": digs, "stable_across_seeds": stable, "instances": inst,
            "seeds": seeds, "train_names": [c.name for c in d.split("train")], "train_structures": len({digs[c] for c in train if digs.get(c)}),
            "train_configs": len(d.split("train")), "unfamiliarity": unf, "same_structure_pairs": same_pairs,
            "errors": errors,
            "compositional_eval": sum(1 for c in d.split("eval_compositional")
                                      if unf.get(c.name, {}).get("compositional")),
            "structural_eval": sum(1 for c in d.split("eval_structural") if unf.get(c.name, {}).get("structural"))}


def evaluator_report(d: Domain, config: str, seed: int, days: float | None = None) -> dict:
    """The evaluator is an independent, evaluator-side function of recorded world state: run the reference (the
    run is cached for `check`), rebuild the world from its event log alone (`World.replay`) and evaluate both; the
    verdicts must be equal. The evaluator must not be a participant tool, and in the default mode evaluator data are
    hidden from participants."""
    from ..core import evaluator_only as EO
    key = (d.name, config, int(seed), "reference", days)
    r = run(d.name, config, seed, policy="reference", days=days, keep_env=True)
    env = r.pop("env")
    _CACHE.setdefault(key, r)
    w = env.world
    me = r["seats"][0]
    live = d.evaluate(w, me, r["objective"])
    events = [e.to_dict() for e in w.events_range(0, w.seq)]
    rw = World.replay(w.spec, events, world_id="evaluator-replay", store=getattr(w, "store", None))
    again = d.evaluate(rw, me, r["objective"])
    canon = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
    names = sorted(t["name"] for t in env.tools(me))
    as_tool = [n for n in names if re.search(r"evaluat|verif|grade|score", n.split(".", 1)[-1])]
    out = {"config": config, "seed": int(seed), "replay_equal": canon(live) == canon(again),
           "replay_digest_equal": rw.digest() == w.digest(), "evaluator_hidden": bool(EO.hidden(w)),
           "verdict_keys": sorted(live) if isinstance(live, dict) else [], "participant_tools_named_like_evaluator":
           as_tool, "has_success_and_score": isinstance(live, dict) and isinstance(live.get("success"), bool)
           and isinstance(live.get("score"), (int, float))}
    rw.close()
    env.close()
    out["ok"] = bool(out["replay_equal"] and out["replay_digest_equal"] and out["evaluator_hidden"]
                     and out["has_success_and_score"])
    return out


def tier3_report(d: Domain, profile: str = "balanced", seed: int = 0, hours: int = 6) -> dict:
    """The first training configuration composed into a Tier 3 civilisation: it builds, runs `hours` without failed
    processes or invariant errors, its seat is a participant there and the domain's evaluator runs on it."""
    c = (d.split("train") or list(d.configs))[0]
    try:
        sp = tier3_spec(d, c.name, seed, profile)
        w = World.create(sp, world_id=f"t3-{d.name}-{c.name}-{seed}")
        since = w.seq
        w.advance_time(w.now + int(hours) * HOUR)
        me = d.principal_of(w)
        ok_seat = me is not None and w.principal(me) is not None
        ev = d.evaluate(w, me, d.objective(w, me, d.reference_difficulty)) if ok_seat else None
        out = {"config": c.name, "profile": profile, "modules": len(w.modules), "seat": me, "seat_present": ok_seat,
               "proc_failed": count_events(w, ("proc.failed",), since),
               "invariant_errors": len(w.check_invariants()), "evaluator_ran": isinstance(ev, dict)}
        w.close()
    except Exception as e:  # noqa: BLE001 - reported: the domain does not compose
        return {"config": c.name, "profile": profile, "composes": False, "error": f"{type(e).__name__}: {e}"[:300]}
    out["composes"] = bool(out["seat_present"] and out["evaluator_ran"] and out["proc_failed"] == 0
                           and out["invariant_errors"] == 0)
    return out


def check(d: Domain | str, *, structure_seeds: Iterable[int] = (0, 1, 2), seeds: Iterable[int] | None = None,
          days: float | None = None, configs: Iterable[str] | None = None, tier3: bool = True,
          tier3_profile: str = "balanced") -> dict:
    """Every Gate N / SP-9.0 / SP-9.16 measurement of one domain, on built and run worlds:

    * `static`: `validate` problems;
    * `structure`: `structure_report`;
    * `runs`: reference, null baseline, every degenerate probe and the generic probes on the reference
      configurations and seeds (through `crucible.experience.make(2, ...)`, observation_only);
    * `feasible` (reference succeeds on >= 1 instance and on a majority), `null_fails`, `probes_fail`;
    * `mechanics` (union over reference runs), `autonomy` (null runs: declared actors active, population decisions),
      `open_ended` (the seat's tools), `observation_only` (default mode, evaluator hidden, reward 0.0);
    * `isolation`: the reference reads no world internals; reference exposure in the seat's observation, tool docs
      and objective; tools named after the reference; sandbox importability;
    * `dice`: arbitrary failure probabilities in the domain's sources;
    * `tier3`: composition into a civilisation."""
    d = REGISTRY[d] if isinstance(d, str) else d
    t0 = time.perf_counter()
    rep: dict = {"domain": d.name, "spec_part": d.spec_part, "static": validate(d)}
    rep["structure"] = structure_report(d, structure_seeds)
    cfgs = list(configs or d.reference_configs or [c.name for c in d.split("train")[:1]])
    seeds = [int(s) for s in (seeds if seeds is not None else d.reference_seeds)]
    runs: dict[str, list] = {"reference": [], "null": []}
    probe_names = [p.name for p in d.degenerate] + [p.name for p in GENERIC_PROBES]
    for p in probe_names:
        runs[p] = []
    errors = {}
    try:
        rep["evaluator"] = evaluator_report(d, cfgs[0], seeds[0], days)
    except Exception as e:  # noqa: BLE001 - reported: the evaluator could not be checked
        rep["evaluator"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    for c in cfgs:
        for s in seeds:
            for pol in ("reference", "null"):
                try:
                    runs[pol].append(cached_run(d.name, c, s, pol, days))
                except Exception as e:  # noqa: BLE001 - a run that crashes is a failed run, reported
                    errors[f"{pol}:{c}@{s}"] = f"{type(e).__name__}: {e}"[:300]
        for p in probe_names:
            try:
                runs[p].append(cached_run(d.name, c, seeds[0], p, days))
            except Exception as e:  # noqa: BLE001
                errors[f"{p}:{c}@{seeds[0]}"] = f"{type(e).__name__}: {e}"[:300]
    rep["run_errors"] = errors
    keys = ("config", "seed", "policy", "success", "score", "events", "proc_failed", "invariant_violations")
    rep["runs"] = {k: [{x: r[x] for x in keys} for r in v] for k, v in runs.items()}
    ref, null = runs["reference"], runs["null"]
    n_ok = sum(1 for r in ref if r["success"])
    rep["reference_successes"], rep["reference_runs"] = n_ok, len(ref)
    rep["feasible"] = bool(ref) and n_ok >= 1 and 2 * n_ok > len(ref) and \
        not any(k.startswith("reference") for k in errors)
    rep["null_fails"] = bool(null) and not any(r["success"] for r in null) and \
        not any(k.startswith("null") for k in errors)
    rep["probes"] = {p: {"runs": len(runs[p]), "successes": sum(1 for r in runs[p] if r["success"])}
                     for p in probe_names}
    rep["degenerate_declared"] = len(d.degenerate)
    rep["probes_fail"] = bool(d.degenerate) and all(v["runs"] > 0 and v["successes"] == 0
                                                    for v in rep["probes"].values())
    rep["process_failures"] = sum(r["proc_failed"] for v in runs.values() for r in v)
    rep["invariant_violations"] = sum(r["invariant_violations"] for v in runs.values() for r in v)
    # mechanics: union of the reference runs' recorded events
    counts: Counter = Counter()
    for r in ref:
        for m, n in r["mechanics"]["counts"].items():
            counts[m] += n
    rep["mechanics"] = {"listed": len(d.required_mechanics),
                        "exercised": sum(1 for m in d.required_mechanics if counts.get(m, 0) > 0),
                        "missing": sorted(m for m in d.required_mechanics if counts.get(m, 0) == 0),
                        "counts": dict(counts), "self_reported": d.mechanic_probes() is None,
                        "spec_list_matches": tuple(d.required_mechanics) == tuple(PART9[d.spec_part]["mechanics"])
                        if d.spec_part in PART9 else True}
    # autonomy: what happens while the seats do nothing
    act: Counter = Counter()
    popd, popk = 0, Counter()
    for r in null:
        for a, n in r["actors"]["counts"].items():
            act[a] += n
        popd += r["population"]["decisions"]
        popk.update(r["population"]["activity_kinds"])
    rep["autonomy"] = {"actors_declared": len(d.actors), "actors_active": sum(1 for a in d.actors if act[a.name] > 0),
                       "actor_counts": dict(act), "population_decisions": popd,
                       "population_activity_kinds": dict(popk)}
    rep["autonomous"] = bool(d.actors) and rep["autonomy"]["actors_active"] == len(d.actors) and popd > 0 \
        and sum(popk.values()) > 0
    # open-ended action space, observation-only default, reward 0.0
    from ..core import evaluator_only as EO
    from ..experience.modes import OBSERVATION_ONLY
    env = make_env(d.name, cfgs[0], seeds[0], difficulty=d.reference_difficulty)
    me = env.agents()[0]
    tools = env.tools(me)
    rep["open_ended"] = open_ended(tools)
    env.advance(HOUR)
    r0 = float(env.reward(me))
    rep["observation_only"] = {"default_mode": env.reward_mode, "evaluator_hidden": bool(EO.hidden(env.world)),
                               "reward": r0, "ok": env.reward_mode == OBSERVATION_ONLY and bool(EO.hidden(env.world))
                               and r0 == 0.0}
    obs = env.observe(me)
    obj = d.objective(env.world, me, d.reference_difficulty)
    env.close()
    # isolation of the reference
    pol = d.reference_policy(me, obj)
    reads = reads_world(pol) + reads_world(d.reference)
    canaries = reference_canaries(d, pol)
    traj = ref[0]["drive"]["agents"][ref[0]["seats"][0]]["tools"] if ref else []
    texts = [obs.get("objective") or "", json.dumps(obs.get("missing") or []), json.dumps(obj, default=str)]
    texts += [json.dumps({k: t.get(k) for k in ("name", "doc", "description", "inputSchema")}) for t in tools]
    ex = exposure(texts, canaries, traj)
    names = {t.get("name") for t in tools}
    named = sorted(c for c in canaries if c in names or any(c.lower() in str(n).lower() for n in names))
    sbx = sandbox_can_import_crucible()
    rep["isolation"] = {"reference_reads_world": reads, "canaries": canaries, "exposure_hits": ex["hits"],
                        "tools_named_after_reference": named, "sandbox_imports_crucible": sbx,
                        "trajectory_head": traj[:6], "texts_scanned": len(texts)}
    rep["reference_paths_exposed"] = ex["exposed"] + len(named) + int(sbx)
    rep["reference_tool_only"] = not reads
    rep["dice"] = dice_scan(domain_sources(d))
    rep["tier3"] = tier3_report(d, tier3_profile) if tier3 else None
    rep["wall_s"] = round(time.perf_counter() - t0, 2)
    return rep


# ============================================================================================ gate aggregation
_REPORTS: dict[str, dict] = {}


def gate_reports(domains: dict[str, Domain] | None = None, **kw) -> dict[str, dict]:
    """`check` of every registered Part 9 domain (cached in-process), keyed by domain name."""
    doms = part9_domains() if domains is None else domains
    out = {}
    for n, d in sorted(doms.items()):
        if n not in _REPORTS:
            _REPORTS[n] = check(d, **kw)
        out[n] = _REPORTS[n]
    return out


def verdicts(rep: dict) -> dict:
    """Per-domain verdicts of one `check` report (what each gate criterion counts)."""
    st = rep["structure"]
    stable_train = {st["digests"][c] for c, s in st["stable_across_seeds"].items()
                    if s and st["digests"].get(c) and c in st.get("train_names", st["digests"])}
    ev = rep.get("evaluator") or {}
    t3 = rep.get("tier3") or {}
    v = {
        "static_ok": not rep["static"],
        "builds": not st["errors"],
        "runs_clean": not rep["run_errors"] and rep["process_failures"] == 0 and rep["invariant_violations"] == 0,
        "train_structures": len(stable_train),
        "min_instances": min(st["instances"].values()) if st["instances"] else 0,
        "compositional_eval": st["compositional_eval"] > 0,
        "structural_eval": st["structural_eval"] > 0,
        "same_structure_pairs": len(st["same_structure_pairs"]) + sum(1 for s in st["stable_across_seeds"].values()
                                                                      if not s),
        "mechanics_ok": rep["mechanics"]["exercised"] == rep["mechanics"]["listed"] and
        rep["mechanics"]["spec_list_matches"] and not rep["mechanics"]["self_reported"],
        "mechanics_missing": len(rep["mechanics"]["missing"]) if rep["mechanics"]["spec_list_matches"] else
        rep["mechanics"]["listed"],
        "evaluator_ok": bool(ev.get("ok")) and rep["feasible"] and rep["null_fails"],
        "reference_ok": rep["feasible"] and rep["reference_tool_only"],
        "exposed": rep["reference_paths_exposed"],
        "autonomous": rep["autonomous"],
        "open_ended": rep["open_ended"]["open_ended"],
        "observation_only": rep["observation_only"]["ok"],
        "tier3": bool(t3.get("composes")),
        "null_fails": rep["null_fails"],
        "probes_ok": rep["probes_fail"],
        "dice": rep["dice"]["mechanisms"],
    }
    v["implemented"] = all((v["static_ok"], v["builds"], v["runs_clean"], v["train_structures"] >= 3,
                            v["min_instances"] >= 3, v["compositional_eval"], v["structural_eval"],
                            v["mechanics_ok"], v["evaluator_ok"], v["reference_ok"], v["exposed"] == 0,
                            v["observation_only"]))
    return v


def gate_metrics(reports: dict[str, dict]) -> dict:
    """The Gate N / SP-9.0 / SP-9.16 metrics over the fifteen Part 9 sections: a section counts when one of its
    domains satisfies the criterion; a section without a domain counts as failing (0 configurations, every
    mechanic missing)."""
    by_part: dict[str, list[dict]] = {p: [] for p in PART9}
    for n, rep in reports.items():
        if rep["spec_part"] in by_part:
            by_part[rep["spec_part"]].append(verdicts(rep))
    part = lambda key: sum(1 for vs in by_part.values() if any(v[key] for v in vs))  # noqa: E731
    best = lambda key, f, miss: [f(v[key] for v in vs) if vs else miss for vs in by_part.values()]  # noqa: E731
    m = {
        "domains_registered": len(reports), "sections_with_domain": sum(1 for vs in by_part.values() if vs),
        "sections_missing": ",".join(p for p, vs in by_part.items() if not vs) or "none",
        "domains_implemented": part("implemented"),
        "min_train_configs_per_domain": min(best("train_structures", max, 0)),
        "domains_with_executable_mechanisms": part("mechanics_ok"),
        "missing_required_mechanics": sum(best("mechanics_missing", min, None)[i] if by_part[p] else
                                          len(PART9[p]["mechanics"]) for i, p in enumerate(by_part)),
        "domains_with_independent_evaluator": part("evaluator_ok"),
        "domains_with_successful_reference": part("reference_ok"),
        "min_seeds_per_config": min(best("min_instances", max, 0)),
        "domains_with_compositional_eval": part("compositional_eval"),
        "domains_with_structural_eval": part("structural_eval"),
        "reference_paths_exposed": sum(v["exposed"] for vs in by_part.values() for v in vs),
        "domains_scanned_for_exposure": sum(1 for vs in by_part.values() if vs),
        "domains_with_autonomous_participants": part("autonomous"),
        "domains_with_open_ended_actions": part("open_ended"),
        "configurations_differing_only_in_objective_text": sum(v["same_structure_pairs"] for vs in by_part.values()
                                                               for v in vs),
        "domains_composable_into_tier3": part("tier3"),
        "domains_where_null_baseline_fails": part("null_fails"),
        "arbitrary_failure_probability_mechanisms": sum(v["dice"] for vs in by_part.values() for v in vs),
        "domains_with_degenerate_strategy_probe": part("probes_ok"),
        "domains_observation_only_default": part("observation_only"),
    }
    return m


def summary() -> dict:
    return {n: {"title": d.title, "spec_part": d.spec_part, "configs": {c.name: c.split for c in d.configs},
                "required_mechanics": list(d.required_mechanics), "seats": [s.handle for s in d.seats],
                "mode": d.mode, "fixture": d.fixture} for n, d in sorted(REGISTRY.items())}


_FRAMEWORK_POLICIES = (ScriptedReference, Idle, RandomTools)
