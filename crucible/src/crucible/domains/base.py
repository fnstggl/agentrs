"""The Tier 2 domain framework: the one contract of a Crucible 5.0 domain (D-17, D-48) and the measurements that
check it on BUILT and RUN worlds (Gate N, SP-9.0, SP-9.16). The authoring guide is docs/v5/components/I.md.

A domain (`Domain`) is a family of worlds for one Part 9 subject. Each of its configurations is

* a civilisation recipe restricted to the subsystems, mechanism variants, couplings, topology and population the
  configuration needs (`civ_recipe`, built on `crucible.civ.recipe`; D-17) - or, where the spec allows a standalone
  semantic world, any recipe in the same 4.0 world-family form (`Domain.construction == "standalone"`). Either way
  every configuration world is an execution-mode B world of MASTER_SPEC Part 2.1 (a subset of civilisation-wide
  activities in one kernel world);
* plus the participant seats a learner (or the scripted reference) occupies (`Seat`; module `seat`, created at
  genesis, several per world);
* plus objectives from the independent goal generator (`GoalFamily`, MASTER_SPEC Part 7:
  ``objective = IndependentlyGenerateGoal(world, difficulty)``): the goal's parameters are drawn for a difficulty
  level from the goal generator's own random stream (never the world's), the outcome is a predicate over
  `crucible.goals.metrics` (registered, evaluator-side metrics of recorded state), and the evaluator is that
  predicate evaluated by `crucible.goals` (optionally refined by a domain `evaluate` that adds details). Only the
  objective's public text reaches participants; in the default ``observation_only`` mode (Clarification 2, D-19) the
  Gym reward is 0.0 and the verdict reaches only `verify()`;
* plus a hidden feasible reference (`reference(principal, public_objective) -> policy`): a scripted policy that acts
  ONLY through the learner interface. It runs in a separate process (`IsolatedPolicy`), is handed the objective
  exactly as the learner sees it (its text) plus observations and its own tool results - never the World - is
  never registered as a tool, never named in an observation, never a required solution path (N.n1), and is
  labelled scripted in evidence;
* plus a null baseline (default `Idle`) and degenerate-strategy probes (`Probe`) that must fail where the reference
  succeeds (SP-9.16: feasible, not guaranteed);
* plus the recorded-event evidence of every required mechanic of its Part 9 section (`Mechanic`) and of its
  autonomous participants (`Actor`): event patterns and evidence functions that return the sequence numbers of
  recorded events, checked by the framework (bookkeeping never counts, substrate-generic events count only through a
  payload filter, an actor's events must be decisions of a participant other than the seats, two mechanics with the
  same evidence are one).

Configurations are procedurally generated instances per seed (`spec(domain, config, seed)`): distinct instances are
distinct genesis state CONTENT (`instance_digest`, the event hash chain excluded). Their STRUCTURE is measured on
the built world (`fingerprint`), never taken from the declaration: the modules whose mechanisms are present
(`engaged_modules`), participant types, civilisation couplings and graph shapes, and the domain's own `structure`
hook, which sees only built state (`BuiltWorld`: no spec, no module configuration), with every magnitude reduced to
presence/absence ("numbers alone do not make a configuration"). A domain needs >= 3 structurally distinct training
configurations, >= 1 compositionally unfamiliar evaluation configuration (every structural atom seen in training,
the combination never) and >= 1 structurally unfamiliar one (an atom no training configuration has); an evaluation
configuration counts only when it runs clean and its reference succeeds there while the null baseline fails.

`validate(d)` is the cheap definition check run by `register`; `check(d)` builds and runs the domain (every
configuration) and returns every Gate N / SP-9.0 / SP-9.16 measurement (the gate tests in tests/v5 iterate over
`part9_domains()`).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..core.canonical import canonical_bytes, digest
from ..core.kernel import SYSTEM, World, WorldSpec

API_VERSION = "domains-framework/3"

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

# Kernel bookkeeping: never evidence of anything (crucible.civ.chains.BOOKKEEPING plus the kernel's own types).
BOOKKEEPING = frozenset({"cmd", "cmd.rejected", "ext", "idem", "ctr", "know", "note", "sched.add", "sched.del",
                         "time", "proc", "proc.failed", "principal.add", "principal.update"})
# Substrate events every world records whatever its mechanisms (money moves, identities, stored blobs, recorded
# representations): a pattern naming one counts only with a payload filter (``bank.txn{kind=subscription}``).
GENERIC = frozenset({"bank.txn", "bank.account", "id.person", "id.org", "id.member", "artifact.stored", "cols.grow",
                     "space.recorded"})
GENERIC_NAMESPACES = ("bank.", "id.", "artifact.", "sched.", "proc", "principal.", "space.", "cols.")
GENERIC_BATCHES = ("pop.", "attention.")       # the population's own bookkeeping column batches


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
    """Recorded-event evidence of one required mechanic.

    `events`: patterns of the event types whose occurrence after genesis in an integrated run exercises it - an
    exact type (``sci.reading``), a prefix (``sci.*``), a column batch kind (``cols.batch:goods.session``,
    ``cols.batch:goods.*``), optionally with a payload filter (``bank.txn{kind=subscription}``,
    ``pkg.executed{kind=ci}``). Kernel bookkeeping never matches; a bare pattern of a substrate-generic type
    (`GENERIC`: money moves, identities, stored blobs) matches nothing - filter it.

    `count(world, since_seq)` (alias: evidence function) returns the SEQUENCE NUMBERS of recorded events after
    `since_seq` that exercise the mechanic (e.g. the events that wrote the rows the domain measures); the framework
    checks each one (recorded after `since_seq`, not bookkeeping). A function that returns a number instead of
    events is a self-report and counts nothing (it is listed as unverified)."""
    name: str
    events: tuple[str, ...] = ()
    doc: str = ""
    count: Callable[[World, int], Iterable[int]] | None = None


@dataclass(frozen=True)
class Actor:
    """An autonomous participant kind of the domain (population individuals, firms, labs, tenants, ...) and the
    recorded events its own decisions produce (patterns as for `Mechanic`; `count` returns event sequence numbers).
    Measured in null-baseline runs (the seats do nothing). An event counts only when it is a decision of a
    participant: its actor is a principal of the world other than the seats and their organisations, or it is a
    population column batch carrying decision provenance (D-15). Kernel bookkeeping and system events never count."""
    name: str
    events: tuple[str, ...] = ()
    doc: str = ""
    count: Callable[[World, int], Iterable[int]] | None = None


@dataclass(frozen=True)
class Probe:
    """A degenerate strategy for the objective (SP-9.16): `policy(principal, objective) -> policy object`. The
    evaluator must NOT report success for it on the probed instances."""
    name: str
    doc: str
    policy: Callable[[str, dict], Any]


@dataclass(frozen=True)
class GoalFamily:
    """The independent goal generator of a domain (MASTER_SPEC Part 7, ``IndependentlyGenerateGoal(world,
    difficulty)``; `crucible.goals`). For a world, a participant and a difficulty level the framework draws the
    goal's parameters with `draw(rng, difficulty, world_view)` from the generator's own random stream (`goal_rng`:
    keyed by the world's seed, the domain, the participant and the level - never one of the world's streams, never
    the reference), states the outcome as `success(params)`, a `crucible.goals.metrics` predicate over recorded state
    (``$self``, ``$org`` - every organisation the participant owns - and ``$orgs`` are bound at evaluation), and
    publishes `text(params)` as the objective's public description. `progress(params)` (optional,
    ``{"metric", "args", "target"}``) gives the evaluator's numeric score. `world_view` is a `BuiltWorld` (built
    state only)."""
    name: str
    draw: Callable[..., dict]
    text: Callable[[dict], str]
    success: Callable[[dict], dict]
    progress: Callable[[dict], dict] | None = None
    window_days: Callable[[dict], float] | float | None = None
    doc: str = ""


@dataclass
class Domain:
    name: str
    title: str
    spec_part: str                  # "9.1" .. "9.15" (anything else: not counted by Gate N)
    required_mechanics: tuple[str, ...]   # () = the Part 9 list of `spec_part`
    configs: tuple[DomainConfig, ...]
    build: Callable[[DomainConfig, int, dict], dict]          # (config, seed, overrides) -> recipe
    objective: Callable[[World, str, int], dict] | None = None  # legacy free objective (prefer `goal`)
    evaluate: Callable[[World, str, dict], dict] | None = None  # evaluator side {"success", "score", "details"}
    reference: Callable[..., Any] | None = None               # (principal, public objective) -> tool-only policy
    mechanics: Any = ()                                       # tuple[Mechanic] | dict[str, Mechanic|tuple]
    participant_principal: Callable[[World], str] | None = None   # default: the first seat
    doc: str = ""
    seats: tuple[Seat, ...] = ()
    actors: tuple[Actor, ...] = ()
    null_baseline: Callable[[str, dict], Any] | None = None   # default: Idle
    degenerate: tuple[Probe, ...] = ()
    structure: Callable[[Any], dict] | None = None            # domain-measured structure of a BuiltWorld
    tier3: Callable[[DomainConfig, dict], dict] | None = None  # merge this configuration into a civ recipe
    mode: str = "C"                 # construction: C = restriction of the civ generator, B = standalone world
    horizon_days: float = 30.0      # evaluation window of the objective
    slowest_effect_days: float = 0.0   # slowest delayed effect of the world (the horizon must exceed it)
    dt: int = HOUR                  # simulated seconds between a policy's decision points
    difficulties: tuple[int, ...] = (1, 2, 3)
    reference_difficulty: int = 1
    reference_seeds: tuple[int, ...] = (0, 1)
    reference_configs: tuple[str, ...] = ()   # default: the first training configuration
    sources: tuple[str, ...] = ()   # extra source files of the domain's mechanisms (dice scan, SP-9.16)
    fixture: bool = False           # a test fixture (never counted by the gates)
    goal: GoalFamily | None = None  # the independent goal generator (crucible.goals)

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

    @property
    def construction(self) -> str:
        """How the configuration worlds are built: a restriction of the civilisation generator (D-17) or a
        standalone semantic world. Both are execution-mode B worlds (Part 2.1)."""
        return "standalone" if self.mode == "B" else "civ_restriction"

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

    # ---------------------------------------------------------------- objectives and verdicts
    def make_objective(self, world: World, principal: str, difficulty: int) -> dict:
        """The objective of `principal` at `difficulty`: generated by the goal generator (`goal`) when the domain
        has one, else the domain's own `objective` (labelled ``generator: "domain"``)."""
        return generate_objective(self, world, principal, difficulty)

    def judge(self, world: World, principal: str, objective: dict) -> dict:
        """The evaluator-side verdict: the domain's `evaluate` when given, else the generated goal's predicate."""
        if self.evaluate is not None:
            return self.evaluate(world, principal, objective)
        return goal_verdict(world, principal, objective)

    def reference_policy(self, principal: str, objective: dict):
        """The reference built IN THIS PROCESS from the public objective (static inspection only; runs use
        `IsolatedPolicy`)."""
        return _call_policy_factory(self.reference, principal, public_objective(objective))

    def null_policy(self, principal: str, objective: dict):
        return _call_policy_factory(self.null_baseline, principal, public_objective(objective)) \
            if self.null_baseline else Idle()

    def is_part9(self) -> bool:
        return self.spec_part in PART9 and not self.fixture


def public_objective(objective: dict | None) -> dict:
    """The objective exactly as a learner sees it (the observation's `objective` is its text): nothing else of the
    objective dict reaches a policy (the dict is computed with full World access, so any other field would be a
    privileged channel)."""
    o = objective or {}
    text = o.get("text") if isinstance(o, dict) else None
    return {"text": text if isinstance(text, str) else ""}


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


# ============================================================================================ goals (crucible.goals)
def goal_rng(world: World, domain: str, principal: str, difficulty: int, family: str):
    """The goal generator's own random stream: keyed by the world's seed, the domain, the participant, the level and
    the goal family - independent of every stream the world's mechanisms draw from (`World.rng_for`)."""
    import numpy as np
    key = int(digest(["goal", domain, family, principal, int(difficulty)])[:12], 16)
    return np.random.default_rng([int(world.spec.seed), key])


def generate_objective(d: Domain, world: World, principal: str, difficulty: int) -> dict:
    """``IndependentlyGenerateGoal(world, difficulty)`` for one participant (see `GoalFamily`)."""
    if d.goal is None:
        if d.objective is None:
            raise ValueError(f"domain {d.name}: neither a goal generator nor an objective")
        o = dict(d.objective(world, principal, int(difficulty)))
        o.setdefault("generator", "domain")
        return o
    from ..goals.metrics import check_predicate
    g = d.goal
    rng = goal_rng(world, d.name, principal, difficulty, g.name)
    params = g.draw(rng, int(difficulty), BuiltWorld(world)) if _arity(g.draw) >= 3 else g.draw(rng, int(difficulty))
    params = json.loads(json.dumps(params))
    pred = g.success(params)
    problems = check_predicate(pred)
    if problems:
        raise ValueError(f"domain {d.name}: goal {g.name}: invalid success predicate: {problems}")
    win = g.window_days(params) if callable(g.window_days) else g.window_days
    out = {"text": str(g.text(params)), "generator": "crucible.goals", "goal_family": g.name, "params": params,
           "success": pred, "difficulty": int(difficulty),
           "window_days": float(win if win is not None else d.horizon_days)}
    if g.progress is not None:
        out["progress"] = g.progress(params)
    return out


def _arity(fn) -> int:
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return 2


def goal_verdict(world: World, principal: str, objective: dict) -> dict:
    """The generated goal's verdict (evaluator side, a function of recorded state): the success predicate with
    ``$self``/``$org``/``$orgs`` bound to the participant and each organisation it owns (`goals.contracts`);
    success when it holds for one of them; score = the progress metric over its target (or the success)."""
    from ..goals.contracts import _bind, bindings
    from ..goals.metrics import evaluate_metric, evaluate_predicate
    pred = objective.get("success")
    if not isinstance(pred, dict):
        raise ValueError("objective has no generated success predicate")
    env = bindings(world, principal)
    rows, best = {}, 0.0
    ok = False
    prog = objective.get("progress")
    for org in env["$orgs"] or ["o:none"]:
        e = dict(env, **{"$org": org})
        try:
            holds = bool(evaluate_predicate(world, _bind(pred, e)))
        except Exception:  # noqa: BLE001 - a missing entity makes the predicate false (as goals.contracts)
            holds = False
        val = None
        if isinstance(prog, dict):
            try:
                val = float(evaluate_metric(world, prog["metric"], _bind(prog.get("args", {}), e)))
            except Exception:  # noqa: BLE001
                val = 0.0
        rows[org] = {"holds": holds, "value": val}
        ok = ok or holds
        if val is not None:
            best = max(best, val)
    if isinstance(prog, dict) and prog.get("target"):
        score = max(0.0, min(1.0, best / float(prog["target"])))
    else:
        score = 1.0 if ok else 0.0
    return {"success": bool(ok), "score": float(score),
            "details": {"goal_family": objective.get("goal_family"), "metric": (prog or {}).get("metric"),
                        "value": best if isinstance(prog, dict) else None,
                        "target": (prog or {}).get("target"), "by_org": rows}}


# ============================================================================================ evidence patterns
_PAT = re.compile(r"^(?P<base>[^{}\s]+)(\{(?P<flt>[^{}]*)\})?$")


@dataclass(frozen=True)
class _Pattern:
    raw: str
    base: str
    filters: tuple
    bookkeeping: bool
    generic: bool

    @property
    def ignored(self) -> bool:
        return self.bookkeeping or (self.generic and not self.filters)


def parse_pattern(p: str) -> _Pattern:
    m = _PAT.match(str(p).strip())
    if not m:
        raise ValueError(f"bad event pattern {p!r}")
    b = m.group("base")
    flt = []
    for part in (m.group("flt") or "").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            flt.append((k.strip(), v.strip()))
    if b.startswith("cols.batch:"):
        k = b[len("cols.batch:"):]
        generic = k in ("", "*") or k.startswith(GENERIC_BATCHES)
        book = False
    else:
        book = b in BOOKKEEPING
        generic = b in GENERIC or b in ("*", ".*") or (b.endswith(".*") and b[:-1].startswith(GENERIC_NAMESPACES)) \
            or b == "cols.batch"
    return _Pattern(str(p), b, tuple(flt), book, generic)


def _pat_match(pt: _Pattern, etype: str, kind: str | None, payload: dict) -> bool:
    if pt.ignored or etype in BOOKKEEPING:
        return False
    b = pt.base
    if b.startswith("cols.batch:"):
        if etype != "cols.batch" or kind is None:
            return False
        k = b[len("cols.batch:"):]
        if not (kind.startswith(k[:-1]) if k.endswith("*") else kind == k):
            return False
    elif b.endswith(".*"):
        if not etype.startswith(b[:-1]):
            return False
    elif etype != b:
        return False
    for key, val in pt.filters:
        if str(payload.get(key)) != val:
            return False
    return True


def _match(pattern: str, etype: str, ekind: str | None) -> bool:
    """Earlier helper (type and batch kind only; payload filters need `_pat_match`)."""
    return _pat_match(parse_pattern(pattern), etype, ekind, {})


def _effective_patterns(probe) -> list[_Pattern]:
    return [pt for pt in (parse_pattern(p) for p in (probe.events or ())) if not pt.ignored]


# ============================================================================================ static validation
def validate(d: Domain) -> list[str]:
    """Definition problems (empty when the domain may be registered). Cheap: nothing is built. Structure, evidence
    and feasibility are checked on built and run worlds by `check`."""
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
            out += _pattern_problems("mechanic", probes.values())
            seen: dict[tuple, str] = {}
            for name, pr in probes.items():
                key = tuple(sorted(p.raw for p in _effective_patterns(pr)))
                if key and pr.count is None:
                    if key in seen:
                        out.append(f"mechanics {seen[key]!r} and {name!r} have the same evidence {list(key)} "
                                   f"(one mechanism, not two)")
                    seen.setdefault(key, name)
    out += _pattern_problems("actor", d.actors)
    if d.mode not in ("B", "C"):
        out.append("mode must be 'B' (standalone semantic world) or 'C' (restriction of the civilisation generator)")
    if d.horizon_days <= 0 or d.horizon_days < d.slowest_effect_days:
        out.append("the evaluation horizon must be positive and exceed the slowest delayed effect")
    handles = [s.handle for s in d.seats]
    if len(handles) != len(set(handles)):
        out.append("seat handles repeat")
    if not callable(d.reference):
        out.append("no reference policy factory")
    if d.goal is None and d.objective is None:
        out.append("no objective: give a goal generator (GoalFamily) or an objective")
    if d.goal is None and d.evaluate is None:
        out.append("no evaluator: a domain without a goal generator needs evaluate")
    if d.goal is not None and not isinstance(d.goal, GoalFamily):
        out.append("goal must be a GoalFamily")
    if d.spec_part in PART9 and not d.fixture:
        if not d.seats and d.participant_principal is None:
            out.append("no participant seat")
        if not d.degenerate:
            out.append("no degenerate-strategy probe (SP-9.16)")
        if not d.actors:
            out.append("no autonomous participant kind (Actor) declared")
    return out


def _pattern_problems(what: str, probes) -> list[str]:
    out = []
    for pr in probes:
        try:
            pats = [parse_pattern(p) for p in (pr.events or ())]
        except ValueError as e:
            out.append(f"{what} {pr.name!r}: {e}")
            continue
        if pr.count is None and pats and all(p.ignored for p in pats):
            out.append(f"{what} {pr.name!r} is evidenced only by kernel bookkeeping or substrate-generic events "
                       f"{[p.raw for p in pats]} (name the mechanism's own event types or add a payload filter)")
        elif pr.count is None and not pats:
            out.append(f"{what} {pr.name!r} has no evidence (no event pattern, no evidence function)")
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
    _forget(d.name)
    return d


def unregister(name: str) -> None:
    REGISTRY.pop(name, None)
    _forget(name)


def _forget(name: str) -> None:
    """Drop the cached runs and gate report of one domain (its definition changed)."""
    for k in [k for k in _CACHE if k[0] == name]:
        del _CACHE[k]
    _REPORTS.pop(name, None)


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


def world(domain: str, config: str, seed: int = 0, overrides: dict | None = None, store=None,
          world_id: str | None = None) -> World:
    return World.create(spec(domain, config, seed, overrides),
                        world_id=world_id or f"d5-{domain}-{config}-{seed}", store=store)


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


# ============================================================================================ built-state view
class DeclarationAccess(AttributeError):
    """A structure hook or goal generator reached for a declaration (spec, module configuration)."""


class _ModuleNames(frozenset):
    def __getitem__(self, name):
        raise DeclarationAccess(f"module {name!r}: a structure hook sees built state only (tables, columns, engaged "
                                f"module names), never module objects or their configuration (D-49)")


class _Rows(Mapping):
    def __init__(self, rows: dict):
        self._rows = rows

    def __getitem__(self, rid):
        return MappingProxyType(self._rows[rid])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class _Tables(Mapping):
    def __init__(self, tables: dict):
        self._tables = tables

    def __getitem__(self, name):
        return _Rows(self._tables[name])

    def __iter__(self):
        return iter(self._tables)

    def __len__(self):
        return len(self._tables)


class _ReadState:
    """Read-only view of a world's state tables and arrays (no meta, no journal, no world)."""

    def __init__(self, st):
        self.get, self.has, self.count, self.ids = st.get, st.has, st.count, st.ids
        self.scan, self.find_ids, self.find, self.find_one = st.scan, st.find_ids, st.find, st.find_one
        self.tables = _Tables(st.tables)
        self._arrays = st.arrays

    def array(self, name: str):
        a = self._arrays[name].view()
        a.flags.writeable = False
        return a

    @property
    def arrays(self):
        return MappingProxyType({k: self.array(k) for k in self._arrays})


class BuiltWorld:
    """What a structure hook (and a goal generator's `draw`) may read: the BUILT world's recorded state - tables,
    column tables, the names of the modules whose mechanisms are present (`engaged_modules`), the clock - and
    nothing declared: no spec, no parameters, no module objects or configuration (D-49)."""

    def __init__(self, world: World):
        self.state = _ReadState(world.state)
        self.modules = _ModuleNames(engaged_modules(world))
        self.now = int(world.now)
        self.seq = int(world.seq)
        reg = (getattr(world, "runtime", {}) or {}).get("cols", {})
        self._cols = {t: list(rec.get("columns") or {}) for t, rec in reg.items()}

    def column_tables(self) -> list[str]:
        return sorted(self._cols)

    def column(self, table: str, name: str):
        if name not in self._cols.get(table, ()):
            raise KeyError(f"no column {table}.{name}")
        return self.state.array(f"{table}.{name}")

    def __getattr__(self, name):
        raise DeclarationAccess(f"BuiltWorld has no {name!r}: structure is measured on built state only (D-49)")


def engaged_modules(world: World, since_seq: int = 0, until_seq: int | None = None, *,
                    scheduled: bool = False) -> list[str]:
    """The modules whose mechanisms are present in a world: a module counts when one of its appliers recorded a
    (non-bookkeeping) event in the window, a table it declared holds rows, or a column table it declared has rows
    (and, with `scheduled`, when one of its processes is scheduled). A module that only adds tools nobody used,
    tables that stay empty or a process that records nothing is not part of the measured structure (the probe of
    inert modules, D-49); `check` adds the modules a configuration's null run engaged (`fingerprint(activity=)`)."""
    st = world.state
    out: set[str] = set()
    owner: dict[str, str] = {}
    for et, fn in world.appliers.items():
        m = getattr(getattr(fn, "__self__", None), "name", None)
        if isinstance(m, str) and m in world.modules:
            owner[et] = m
    for e in world.events_range(int(since_seq), until_seq):
        if e.type in BOOKKEEPING:
            continue
        m = owner.get(e.type)
        if m:
            out.add(m)
    for t, mods in getattr(st, "declared_by", {}).items():
        if mods and mods[0] in world.modules and st.tables.get(t):
            out.add(mods[0])
    reg = (getattr(world, "runtime", {}) or {}).get("cols", {})
    for t, rec in reg.items():
        cols = list(rec.get("columns") or {})
        if not cols:
            continue
        a = st.arrays.get(f"{t}.{cols[0]}")
        if a is not None and a.shape[0] > 0:
            for c, who in (rec.get("by") or {}).items():
                if who in world.modules:
                    out.add(who)
    if scheduled:
        for j in st.tables.get("sched", {}).values():
            m = str(j.get("proc", "")).split(".", 1)[0]
            if m in world.modules:
                out.add(m)
    return sorted(out)


def instance_digest(world: World) -> str:
    """Content digest of a world's state - tables and arrays, without the kernel's meta (event hash chain, sequence
    number, clock, counters) and idempotency records. Two seeds of a configuration are distinct instances only when
    the generated content differs (N.6); the hash chain differs for every seed and says nothing."""
    st = world.state
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(st.tables):
        if name == "idem":
            continue
        h.update(b"T" + name.encode())
        t = st.tables[name]
        for rid in sorted(t):
            h.update(str(rid).encode())
            h.update(canonical_bytes(t[rid]))
    import numpy as np
    for name in sorted(st.arrays):
        a = np.ascontiguousarray(st.arrays[name])
        h.update(b"A" + name.encode() + str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def table_digests(world: World) -> dict[str, str]:
    st = world.state
    out = {}
    for name in sorted(st.tables):
        if name == "idem":
            continue
        h = hashlib.blake2b(digest_size=8)
        for rid in sorted(st.tables[name]):
            h.update(str(rid).encode())
            h.update(canonical_bytes(st.tables[name][rid]))
        out[name] = h.hexdigest()
    return out


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
    (`obs`, a `crucible.experience` Observation), the objective's public text (`self.objective["text"]`) and the
    results of its own actions. It never holds the World (it runs in a separate process, `IsolatedPolicy`).

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


# ---------------------------------------------------------------------------------------------- isolation
class ReferenceNotIsolated(RuntimeError):
    """The reference factory cannot be imported by module path, so it cannot run in a separate process."""


def _json_default(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return str(x)


def factory_path(fn) -> tuple[str, str]:
    """(module, qualname) of an importable policy factory, else raises ReferenceNotIsolated."""
    mod = getattr(fn, "__module__", None)
    q = getattr(fn, "__qualname__", None)
    if not mod or not q or "<" in q or mod == "__main__":
        raise ReferenceNotIsolated(f"policy factory {fn!r} is not importable by module path (module {mod!r}, "
                                   f"qualname {q!r}): define it at module level of an importable module")
    obj = sys.modules.get(mod)
    for part in q.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
    if obj is not fn:
        raise ReferenceNotIsolated(f"policy factory {mod}.{q} does not resolve to the same object")
    return mod, q


class IsolatedPolicy:
    """A policy that runs in a SEPARATE PROCESS: the parent sends the learner's observations (JSON) and the results
    of the policy's own calls; the child holds no World, no environment and no event log, so whatever the policy's
    code does (introspection, garbage-collector walks, string-built attribute names), it can reach only what a
    learner reaches. Used for every reference run (N.5, N.n1)."""
    kind = "scripted"

    def __init__(self, factory, principal: str, objective: dict, timeout_s: float = 600.0):
        mod, q = factory_path(factory)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        path = [p for p in sys.path if p and os.path.isdir(p)]
        env["PYTHONPATH"] = os.pathsep.join(path)
        import tempfile
        self._err = tempfile.TemporaryFile(mode="w+")
        self._p = subprocess.Popen([sys.executable, "-c", "from crucible.domains.base import _serve_policy; "
                                    "_serve_policy()"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=self._err, env=env, text=True, bufsize=1)
        self._tools_key = None
        self.timeout_s = float(timeout_s)
        r = self._call({"op": "init", "module": mod, "qualname": q, "principal": principal,
                        "objective": public_objective(objective), "path": path})
        self.kind = r.get("kind") or "scripted"
        self.name = r.get("name") or q
        self.burst = int(r.get("burst") or 1)
        self.classes = list(r.get("classes") or [])
        self.child_pid = r.get("pid")
        self.calls = 0

    def _call(self, msg: dict) -> dict:
        if self._p.poll() is not None:
            raise RuntimeError(f"isolated policy process exited ({self._p.returncode}): {self._stderr()}")
        self._p.stdin.write(json.dumps(msg, default=_json_default) + "\n")
        self._p.stdin.flush()
        import select
        ready, _, _ = select.select([self._p.stdout], [], [], self.timeout_s)
        if not ready:
            self._p.kill()
            raise RuntimeError(f"isolated policy did not answer {msg.get('op')!r} within {self.timeout_s:.0f} s")
        line = self._p.stdout.readline()
        if not line:
            raise RuntimeError(f"isolated policy process died: {self._stderr()}")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(f"isolated policy raised {r.get('error')}\n{r.get('trace', '')}")
        return r

    def _stderr(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read()[-3000:]
        except Exception:  # noqa: BLE001
            return ""

    def _tools(self, tools):
        key = hashlib.blake2b(json.dumps(tools, sort_keys=True, default=_json_default).encode(),
                              digest_size=12).hexdigest()
        if key == self._tools_key:
            return None
        self._tools_key = key
        return tools

    @staticmethod
    def _strip(obs):
        return {k: v for k, v in obs.items() if k != "tools"} if isinstance(obs, dict) else obs

    def act(self, obs, tools):
        self.calls += 1
        return self._call({"op": "act", "obs": self._strip(obs), "tools": self._tools(tools)}).get("action")

    def observe(self, reward, result, next_obs, done):
        nt = next_obs.get("tools") if isinstance(next_obs, dict) else None
        self._call({"op": "observe", "reward": float(reward), "result": result, "next_obs": self._strip(next_obs),
                    "tools": self._tools(nt) if nt is not None else None, "done": bool(done)})

    def close(self):
        if self._p.poll() is None:
            try:
                self._p.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._p.stdin.flush()
                self._p.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self._p.kill()
        self._err.close()

    def __del__(self):
        try:
            if self._p.poll() is None:
                self._p.kill()
        except Exception:  # noqa: BLE001
            pass


def _serve_policy() -> None:
    """Child side of `IsolatedPolicy` (JSON lines on stdin/stdout; policy prints go to stderr)."""
    import importlib
    import traceback
    out = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr
    pol, tools = None, None
    for line in sys.stdin:
        msg = json.loads(line)
        op = msg.get("op")
        try:
            if op == "init":
                for p in reversed(msg.get("path") or []):
                    if p not in sys.path:
                        sys.path.insert(0, p)
                fn = importlib.import_module(msg["module"])
                for part in msg["qualname"].split("."):
                    fn = getattr(fn, part)
                pol = _call_policy_factory(fn, msg["principal"], dict(msg.get("objective") or {}))
                resp = {"ok": True, "kind": getattr(pol, "kind", "unknown"), "name": getattr(pol, "name", None),
                        "burst": int(getattr(pol, "burst", 1) or 1), "pid": os.getpid(),
                        "classes": [f"{c.__module__}.{c.__qualname__}" for c in type(pol).__mro__ if c is not object]}
            elif op == "act":
                if msg.get("tools") is not None:
                    tools = msg["tools"]
                obs = dict(msg.get("obs") or {}, tools=tools)
                resp = {"ok": True, "action": pol.act(obs, tools)}
            elif op == "observe":
                if msg.get("tools") is not None:
                    tools = msg["tools"]
                nobs = dict(msg.get("next_obs") or {}, tools=tools)
                if hasattr(pol, "observe"):
                    pol.observe(msg.get("reward", 0.0), msg.get("result"), nobs, msg.get("done", False))
                resp = {"ok": True}
            elif op == "close":
                out.write(json.dumps({"ok": True}) + "\n")
                break
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001 - reported to the parent, which raises
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        out.write(json.dumps(resp, default=_json_default) + "\n")


# ============================================================================================ driving
class ExposureTrace:
    """Collects, from everything a participant receives (observations, minus the tool list, and tool results), the
    ordered tool names each text mentions and every canary identifier found. A text that mentions at least half of
    the seat's tools is a catalogue (a help listing), not a recipe: it is scanned for canaries only."""

    def __init__(self, tool_names: Iterable[str], canaries: Iterable[str] = (), max_sequences: int = 4000):
        names = sorted({str(n) for n in tool_names if n}, key=lambda s: (-len(s), s))
        self.n_tools = len(names)
        self._re = re.compile("|".join(re.escape(n) for n in names)) if names else None
        self.canaries = [c for c in canaries if c]
        self.sequences: set[tuple] = set()
        self.canary_hits: set[str] = set()
        self.texts = 0
        self.max_sequences = int(max_sequences)

    def feed(self, x, where: str = "") -> None:
        s = x if isinstance(x, str) else json.dumps(x, default=_json_default, sort_keys=True)
        self.texts += 1
        for c in self.canaries:
            if c in s:
                self.canary_hits.add(f"canary {c!r} in {where or 'text'}")
        if self._re is None:
            return
        seq = [m.group(0) for m in self._re.finditer(s)]
        if len(set(seq)) >= 3 and len(set(seq)) < max(10, self.n_tools // 2) and \
                len(self.sequences) < self.max_sequences:
            self.sequences.add(tuple(seq[:400]))

    def summary(self) -> dict:
        return {"texts": self.texts, "sequences": [list(s) for s in sorted(self.sequences)],
                "canary_hits": sorted(self.canary_hits)}


def drive(env, policies: dict[str, Any], *, days: float, dt: int = HOUR, max_burst: int | None = None,
          trace: bool = True, exposure: dict | None = None) -> dict:
    """Run policies (``act(obs, tools) -> action | None``, ``observe(reward, result, next_obs, done)``) for
    `days` simulated days through an `ExperienceEnv`. At each decision point every policy acts on its own
    observation, up to its `burst` actions (each followed by the next observation), then time passes `dt` seconds.
    Policies see only observations and their own results. Every reward the environment hands out is read (after
    each action and each time step): `nonzero_rewards` counts the nonzero ones. `exposure` ({agent:
    ExposureTrace}) receives every observation and result of those agents. Returns per-agent counts and (when
    `trace`) each agent's tool sequence."""
    agents = [a for a in env.agents() if a in policies]
    steps = int(round(days * DAY / int(dt)))
    out = {a: {"actions": 0, "ok": 0, "rejected": 0, "tools": [] if trace else None, "errors": Counter(),
               "nonzero_rewards": 0} for a in agents}
    exposure = exposure or {}
    t0 = time.perf_counter()

    def seen(a, kind, x):
        tr = exposure.get(a)
        if tr is not None:
            tr.feed({k: v for k, v in x.items() if k != "tools"} if isinstance(x, dict) and kind == "obs" else x,
                    kind)
    for _ in range(steps):
        for a in agents:
            pol = policies[a]
            n = int(max_burst if max_burst is not None else getattr(pol, "burst", 1))
            obs = env.observe(a)
            seen(a, "obs", obs)
            for _k in range(max(1, n)):
                action = pol.act(obs, obs["tools"])
                if action is None:
                    break
                res = env.execute(a, action)
                seen(a, "result", res)
                r = float(env.reward(a))
                o = out[a]
                o["nonzero_rewards"] += int(r != 0.0)
                o["actions"] += 1
                o["ok" if res.get("ok") else "rejected"] += 1
                if not res.get("ok"):
                    o["errors"][str((res.get("error") or {}).get("code") or "error")] += 1
                if trace:
                    o["tools"].append(str(action.get("tool")))
                obs = env.observe(a)
                seen(a, "obs", obs)
                if hasattr(pol, "observe"):
                    pol.observe(r, res, obs, False)
        env.advance(int(dt))
        for a in agents:
            out[a]["nonzero_rewards"] += int(float(env.reward(a)) != 0.0)
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


def _env_objective(env, d: Domain, agent: str, difficulty: int) -> dict:
    o = getattr(env, "_objectives", {}).get(agent)
    obj = getattr(o, "objective", None)
    return dict(obj) if isinstance(obj, dict) else d.make_objective(env.world, agent, difficulty)


def freeze_processes(world: World, procs: Iterable[str]) -> list[str]:
    """Intervention arm: the named scheduled processes do nothing when they run (their autonomous decisions are
    switched off). Evaluator-side only; returns the processes frozen."""
    done = []
    for p in sorted(set(procs)):
        if p in world.processes:
            world.processes[p] = _frozen
            done.append(p)
    return done


def _frozen(ctx, **_args):
    return None


def run(domain: str, config: str, seed: int = 0, *, policy: str = "reference", days: float | None = None,
        difficulty: int | None = None, keep_env: bool = False, overrides: dict | None = None,
        isolate: bool | None = None, freeze: Iterable[str] = (), queries: bool = False) -> dict:
    """One evaluated run through the learner interface: the first seat is driven by the domain's reference
    (`policy="reference"`, in a separate process unless `isolate=False`), its null baseline (``"null"``), a
    degenerate probe (its name) or a generic probe (``"random_tools"``); other seats idle. Returns the evaluator's
    outcome (evaluator side), the actions, the measured mechanics and autonomous activity after genesis, the
    rewards handed out, what the seat received (`exposure`: tool-name sequences and canaries, plus - with
    `queries` - the results of every argument-free read-only tool at the end), and the run's labels. `overrides`
    (as for `spec`) build an intervention arm, e.g. ``{"module_config": {"sci": {"replicators": {"n": 0}}}}``;
    `freeze` names scheduled processes that do nothing (the autonomy ablation of `evaluator_report`)."""
    d = REGISTRY[domain]
    days = float(d.horizon_days if days is None else days)
    difficulty = int(d.reference_difficulty if difficulty is None else difficulty)
    isolate = (policy == "reference") if isolate is None else bool(isolate)
    env = make_env(domain, config, seed, difficulty=difficulty, overrides=overrides)
    pol = None
    try:
        w = env.world
        frozen = freeze_processes(w, freeze) if freeze else []
        since = int(w.seq)
        seats = env.agents()
        me = seats[0]
        obj = _env_objective(env, d, me, difficulty)
        if policy == "reference":
            factory = d.reference
        elif policy == "null":
            factory = d.null_baseline or (lambda p, o: Idle())
        else:
            probes = {p.name: p for p in tuple(d.degenerate) + GENERIC_PROBES}
            if policy not in probes:
                raise KeyError(f"domain {domain}: no policy {policy!r}")
            factory = probes[policy].policy
        pol = IsolatedPolicy(factory, me, obj) if isolate else _call_policy_factory(factory, me, public_objective(obj))
        tools0 = env.tools(me)
        canaries = reference_canaries(d)
        tr = ExposureTrace([t.get("name") for t in tools0], canaries)
        pols = {me: pol, **{s: Idle() for s in seats[1:]}}
        dr = drive(env, pols, days=days, dt=d.dt, exposure={me: tr})
        ver = env.verify()
        ev = (ver.get("evaluation") or {}).get(me) or {}
        q = query_results(env, me) if queries else None
        if q:
            for name, res in q.items():
                tr.feed(res, f"query {name}")
        out = {"domain": domain, "config": config, "seed": int(seed), "policy": policy, "days": days,
               "difficulty": difficulty, "objective": obj, "outcome": ev, "success": bool(ev.get("success")),
               "score": ev.get("score"), "drive": dr, "seats": seats, "events": int(w.seq), "since_seq": since,
               "invariant_violations": len(ver.get("invariant_violations") or []),
               "proc_failed": count_events(w, ("proc.failed",), since),
               "mechanics": mechanics_exercised(w, d, since), "actors": actors_active(w, d, since),
               "population": population_decisions(w, since), "reward_mode": env.reward_mode,
               "nonzero_rewards": int(dr["agents"][me]["nonzero_rewards"]),
               "exposure": tr.summary(), "queried_tools": sorted(q) if q else [],
               "engaged_modules": engaged_modules(w, since), "frozen": frozen, "isolated": bool(isolate),
               "overrides": copy.deepcopy(overrides) if overrides else None,
               "labels": {"scripted": policy == "reference", "policy_kind": getattr(pol, "kind", "unknown"),
                          "isolated_process": bool(isolate)}}
    finally:
        if pol is not None and hasattr(pol, "close"):
            pol.close()
        if not keep_env:
            env.close()
    if keep_env:
        out["env"] = env
    return out


def query_results(env, agent: str) -> dict:
    """The results of every argument-free read-only tool of `agent` (`World.query`: never recorded)."""
    out = {}
    w = env.world
    for t in env.tools(agent):
        ann = t.get("annotations") or {}
        req = (t.get("inputSchema") or {}).get("required") or []
        if not ann.get("readOnlyHint") or req:
            continue
        try:
            out[t["name"]] = w.query(agent, t["name"], {})
        except Exception as e:  # noqa: BLE001 - a refused query is reported, not fatal
            out[t["name"]] = {"error": f"{type(e).__name__}"}
    return out


# ============================================================================================ evaluator helpers
def seat_orgs(world: World, principal: str) -> list[str]:
    """Organisations a seat acts for (evaluator side): the `o:` principals among `world.principals_of(principal)`."""
    return sorted(p for p in world.principals_of(principal) if p != principal and p.startswith("o:"))


def independent_revenue(world: World, principal: str, window_s: int) -> float:
    """Revenue from unrelated payers of a seat's organisations over a window (evaluator side;
    `goals.metrics.independent_revenue`: transfers among related parties count for nothing)."""
    from ..goals.metrics import evaluate_metric
    return float(sum(evaluate_metric(world, "independent_revenue", {"principal": o, "window_s": int(window_s)})
                     for o in seat_orgs(world, principal)))


def events_where(world: World, since_seq: int, types: Iterable[str], pred: Callable | None = None) -> list[int]:
    """Sequence numbers of recorded events after `since_seq` matching `types` (patterns as for `Mechanic`) and
    `pred(event)` - the building block of evidence functions."""
    pats = [parse_pattern(p) for p in types]
    out = []
    for e in world.events_range(int(since_seq), world.seq):
        k = e.payload.get("kind") if e.type == "cols.batch" else None
        if any(_pat_match(p, e.type, k, e.payload) for p in pats) and (pred is None or pred(e)):
            out.append(int(e.seq))
    return out


# ============================================================================================ measurements
def count_events(world: World, patterns: Iterable[str], since_seq: int = 0) -> int:
    """Events after `since_seq` matching the patterns (bookkeeping types are matched literally here: this is a
    plain counter, e.g. of ``proc.failed``)."""
    pats = tuple(patterns)
    if not pats:
        return 0
    n = 0
    for e in world.events_range(int(since_seq), world.seq):
        k = e.payload.get("kind") if e.type == "cols.batch" else None
        for p in pats:
            if p == e.type or (p.endswith(".*") and e.type.startswith(p[:-1])) or \
                    (p.startswith("cols.batch:") and e.type == "cols.batch" and k is not None and
                     (k.startswith(p[11:-1]) if p.endswith("*") else k == p[11:])):
                n += 1
                break
    return n


def event_counts(world: World, since_seq: int = 0) -> Counter:
    """Event types after `since_seq`, plus ``cols.batch:<kind>`` for every column batch."""
    c: Counter = Counter()
    for e in world.events_range(int(since_seq), world.seq):
        c[e.type] += 1
        if e.type == "cols.batch":
            c[f"cols.batch:{e.payload.get('kind')}"] += 1
    return c


def _excluded_principals(world: World, d: Domain) -> set[str]:
    out = set(d.seat_principals())
    for s in list(out):
        try:
            out |= set(world.principals_of(s))
        except Exception:  # noqa: BLE001
            pass
    try:
        p = d.principal_of(world)
        if p:
            out.add(p)
            out |= set(world.principals_of(p))
    except Exception:  # noqa: BLE001
        pass
    return out


def _attributed(e, principals: Mapping, excluded: set[str]) -> bool:
    if e.type == "cols.batch":
        return bool((e.payload.get("extra") or {}).get("decision"))
    a = e.actor
    return isinstance(a, str) and a != SYSTEM and a in principals and a not in excluded


def _evidence(world: World, d: Domain, probes: dict, since_seq: int, *, attributed: bool) -> dict:
    """Evidence of each probe after `since_seq`: matching events plus the event sequence numbers its evidence
    function returns, each checked against the recorded log (after `since_seq`, not bookkeeping and, for actors,
    a decision of a participant other than the seats). Returns sets and the problems found."""
    since = int(since_seq)
    from_fn: dict[str, set[int]] = {}
    unverified, errors = {}, {}
    for name, pr in probes.items():
        fn = getattr(pr, "count", None) if pr is not None else None
        if fn is None:
            continue
        try:
            got = fn(world, since)
        except Exception as e:  # noqa: BLE001 - a crashing evidence function gives no evidence (reported)
            errors[name] = f"{type(e).__name__}: {e}"[:200]
            continue
        if got is None or isinstance(got, (bool, int, float)) or type(got).__name__ in ("int64", "int32", "float64"):
            unverified[name] = got if isinstance(got, (bool, int, float, type(None))) else float(got)
            continue
        try:
            from_fn[name] = {int(x) for x in got}
        except (TypeError, ValueError):
            unverified[name] = str(type(got).__name__)
    pats = {n: _effective_patterns(pr) for n, pr in probes.items() if pr is not None}
    sets: dict[str, set[int]] = {n: set() for n in probes}
    procs: dict[str, set[str]] = {n: set() for n in probes}
    principals = world.state.tables.get("principals", {})
    excluded = _excluded_principals(world, d) if attributed else set()
    accepted = {n: 0 for n in from_fn}
    cur = None
    for e in world.events_range(since, world.seq):
        if e.type == "proc":
            cur = e.payload.get("proc")
            continue
        if e.type in ("cmd", "time"):
            cur = None
        if e.type in BOOKKEEPING:
            continue
        if attributed and not _attributed(e, principals, excluded):
            continue
        k = e.payload.get("kind") if e.type == "cols.batch" else None
        for n, pl in pats.items():
            if pl and any(_pat_match(p, e.type, k, e.payload) for p in pl):
                sets[n].add(int(e.seq))
                if cur:
                    procs[n].add(cur)
        for n, s in from_fn.items():
            if e.seq in s:
                sets[n].add(int(e.seq))
                accepted[n] += 1
                if cur:
                    procs[n].add(cur)
    rejected = {n: len(s) - accepted[n] for n, s in from_fn.items() if len(s) != accepted[n]}
    return {"sets": sets, "unverified": unverified, "errors": errors, "rejected": rejected,
            "procs": {n: sorted(v) for n, v in procs.items()}}


def mechanics_exercised(world: World, d: Domain, since_seq: int = 0) -> dict:
    """For every required mechanic: the recorded events (after `since_seq`) that evidence it (see `Mechanic`).
    Self-reported mechanics (a callable `mechanics`, or an evidence function returning a number) count nothing; a
    mechanic whose evidence is exactly another one's is `indistinct` and counts nothing (one mechanism, not two)."""
    probes = d.mechanic_probes()
    if probes is None:
        rows = {m: 0 for m in d.required_mechanics}
        return {"listed": len(rows), "exercised": 0, "missing": sorted(rows), "counts": rows, "self_reported": True,
                "unverified": {}, "rejected": {}, "indistinct": {}, "errors": {}}
    req = {m: probes.get(m) for m in d.required_mechanics}
    ev = _evidence(world, d, {m: p for m, p in req.items() if p is not None}, since_seq, attributed=False)
    rows, indistinct, first = {}, {}, {}
    for m in d.required_mechanics:
        s = ev["sets"].get(m) or set()
        key = hashlib.blake2b(json.dumps(sorted(s)).encode(), digest_size=12).hexdigest() if s else None
        if key and key in first:
            indistinct[m] = first[key]
            rows[m] = 0
            continue
        if key:
            first[key] = m
        rows[m] = len(s)
    ex = sorted(m for m, n in rows.items() if n > 0)
    return {"listed": len(d.required_mechanics), "exercised": len(ex), "missing": sorted(set(rows) - set(ex)),
            "counts": rows, "self_reported": False, "unverified": ev["unverified"], "rejected": ev["rejected"],
            "indistinct": indistinct, "errors": ev["errors"]}


def actors_active(world: World, d: Domain, since_seq: int = 0) -> dict:
    """For every declared actor: its attributed decision events after `since_seq` (see `Actor`), and the scheduled
    processes that made them (`processes`, used by the autonomy ablation)."""
    ev = _evidence(world, d, {a.name: a for a in d.actors}, since_seq, attributed=True)
    rows = {a.name: len(ev["sets"].get(a.name) or ()) for a in d.actors}
    procs = sorted({p for a in d.actors for p in ev["procs"].get(a.name, [])})
    return {"declared": len(rows), "active": sum(1 for n in rows.values() if n > 0), "counts": rows,
            "unverified": ev["unverified"], "rejected": ev["rejected"], "errors": ev["errors"], "processes": procs}


POP_INTERNAL = ("pop.", "attention.")


def population_decisions(world: World, since_seq: int = 0) -> dict:
    """Recorded decisions of the one population after `since_seq` (column batches with decision provenance,
    D-15): all of them, and those of activities other than the population's own bookkeeping; `processes` are the
    scheduled processes that made them."""
    total, domain_kinds = 0, Counter()
    models, procs = set(), set()
    cur = None
    for e in world.events_range(int(since_seq), world.seq):
        if e.type == "proc":
            cur = e.payload.get("proc")
            continue
        if e.type in ("cmd", "time"):
            cur = None
        if e.type != "cols.batch":
            continue
        dec = (e.payload.get("extra") or {}).get("decision")
        kind = str(e.payload.get("kind") or "")
        if not dec or kind.endswith(".init"):
            continue
        total += 1
        models.add(str(dec.get("model_id")))
        if cur:
            procs.add(cur)
        if not kind.startswith(POP_INTERNAL):
            domain_kinds[kind] += 1
    return {"decisions": total, "activity_decisions": int(sum(domain_kinds.values())),
            "activity_kinds": dict(domain_kinds), "models": sorted(models), "processes": sorted(procs)}


# ---------------------------------------------------------------------------------------------- structure
def discrete(x: Any) -> Any:
    """The structural form of a value: magnitudes become presence/absence ("zero"/"nonzero"), numeric sequences
    their set of degree classes (0, 1, 2, "3+"), other sequences their set of distinct elements; strings and
    booleans stay. Two values with equal discrete forms differ only in numbers."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return "zero" if x == 0 else "nonzero"
    if isinstance(x, (dict, Mapping)):
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


def fingerprint(world: World, d: Domain | None = None, activity: Iterable[str] = ()) -> dict:
    """The discrete structure of a BUILT world: the modules whose mechanisms are present (`engaged_modules` at
    genesis plus `activity`, the modules a run of the configuration engaged - never the declared module list),
    participant types, the civilisation structure (`crucible.civ.integration.fingerprint`: couplings, supply-chain
    and freight shapes, regimes) when civilisation modules are present, and the domain's own
    `structure(BuiltWorld)`; every value in discrete form."""
    mods = sorted(set(engaged_modules(world)) | {m for m in activity if m in world.modules})
    fp: dict = {"modules": mods, "participant_types": participant_types(world)}
    if any(m in world.modules for m in ("civgen", "civlinks", "industrygen")):
        from ..civ import integration as I
        c = I.fingerprint(world)
        c.pop("modules", None)
        c["supply_chain"] = _supply_shape(world)
        c["freight"] = _freight_shape(world)
        fp["civ"] = c
    if d is not None and d.structure is not None:
        fp["domain"] = d.structure(BuiltWorld(world))
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
                             "events_range", "runtime", "hooks", "endpoints", "evaluate", "evaluator", "REGISTRY",
                             "_CACHE", "_REPORTS", "__dict__", "__globals__", "__closure__", "f_back", "f_locals",
                             "f_globals", "gi_frame", "tb_frame", "get_objects", "get_referrers", "get_referents",
                             "_getframe", "processes", "appliers", "make_env", "judge"})
FORBIDDEN_NAMES = frozenset({"World", "WorldSpec", "REGISTRY", "globals", "vars", "locals", "eval", "exec",
                             "compile", "__import__", "breakpoint", "__builtins__"})
FORBIDDEN_IMPORTS = ("gc", "inspect", "importlib", "builtins", "ctypes", "sys", "crucible.core", "crucible.domains",
                     "crucible.experience", "crucible.api", "crucible.civ", "..core", "..experience")
_FRAMEWORK_POLICIES: tuple = ()


def _classes_of(obj) -> list[type]:
    out = []
    for c in type(obj).__mro__:
        if c is object or c in _FRAMEWORK_POLICIES:
            continue
        out.append(c)
    return out


def _ast_hits(t, name: str, src: str) -> list[str]:
    hits = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hits.append(f"{name}:{node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            hits.append(f"{name}:{node.lineno}: {node.id}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("getattr", "setattr", "delattr", "hasattr") and len(node.args) >= 2 and \
                not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) and
                     node.args[1].value not in FORBIDDEN_ATTRS):
            hits.append(f"{name}:{node.lineno}: {node.func.id} with a computed or forbidden attribute name")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (("." * getattr(node, "level", 0)) + (getattr(node, "module", None) or "")) \
                if isinstance(node, ast.ImportFrom) else ""
            names = [a.name for a in node.names]
            full = [mod] if mod else names
            if any(m == f or m.startswith(f + ".") or ("kernel" in m) for m in full + names
                   for f in FORBIDDEN_IMPORTS) or "kernel" in mod or "core.state" in mod:
                hits.append(f"{name}:{node.lineno}: imports {', '.join(full)}")
    return hits


def _helpers(t) -> list:
    """Module-level functions and classes of the same module that `t`'s code refers to by name."""
    try:
        src = textwrap.dedent(inspect.getsource(t))
    except (OSError, TypeError):
        return []
    g = getattr(sys.modules.get(getattr(t, "__module__", ""), None), "__dict__", {})
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Name):
            obj = g.get(node.id)
            if obj is not None and obj is not t and (inspect.isfunction(obj) or inspect.isclass(obj)) and \
                    getattr(obj, "__module__", None) == t.__module__ and obj not in _FRAMEWORK_POLICIES:
                out.append(obj)
    return out


def reads_world(obj_or_fn) -> list[str]:
    """Static check that a policy (its classes) or a factory reads nothing but observations: attribute accesses of
    world internals and introspection (`FORBIDDEN_ATTRS`), kernel and introspection names (`FORBIDDEN_NAMES`),
    getattr/setattr with computed names, and imports of the kernel, the framework or introspection modules, by AST -
    in the policy's classes AND every module-level helper function or class of the same module they refer to
    (transitively). A target whose source cannot be read is reported. The dynamic guarantee is `IsolatedPolicy`."""
    if obj_or_fn is None:
        return []
    todo = [obj_or_fn] if inspect.isfunction(obj_or_fn) or inspect.isclass(obj_or_fn) else _classes_of(obj_or_fn)
    seen, hits = set(), []
    while todo:
        t = todo.pop(0)
        if id(t) in seen:
            continue
        seen.add(id(t))
        name = getattr(t, "__qualname__", str(t))
        try:
            src = textwrap.dedent(inspect.getsource(t))
            hits += _ast_hits(t, name, src)
        except (OSError, TypeError, SyntaxError):
            if getattr(t, "__module__", "") == "builtins":
                continue
            hits.append(f"{name}: source unavailable (cannot be checked)")
            continue
        todo += [h for h in _helpers(t) if id(h) not in seen]
    return hits


def reference_canaries(d: Domain, policy=None) -> list[str]:
    """Distinctive identifiers of a domain's reference: its factory and (when given) its policy classes and their
    module paths. Plain words ("Reference") are skipped."""
    out = set()
    fac = d.reference
    targets = []
    if policy is not None:
        targets += _classes_of(policy)
    if inspect.isclass(fac):
        targets += [c for c in fac.__mro__ if c is not object and c not in _FRAMEWORK_POLICIES]
    for c in targets:
        if len(c.__name__) >= 8 and sum(ch.isupper() for ch in c.__name__) >= 2:
            out.add(c.__name__)
        out.add(f"{c.__module__}.{c.__qualname__}")
    q = getattr(fac, "__qualname__", "")
    if q and "<" not in q:
        if len(q) >= 8 and (sum(ch.isupper() for ch in q) >= 2 or "_" in q.strip("_")):
            out.add(q)
        out.add(f"{fac.__module__}.{q}")
    return sorted(out)


def _first_distinct(trajectory: Iterable[str], n: int = 6) -> list[str]:
    seq: list[str] = []
    for t in trajectory:
        if t not in seq:
            seq.append(t)
    return seq[:n]


def _ordered_prefix(found_in: list[str], seq: list[str]) -> int:
    """How many tools of `seq` (from its start) appear in order in the sequence `found_in`."""
    pos, k = 0, 0
    for tool in seq:
        try:
            pos = found_in.index(tool, pos) + 1
        except ValueError:
            break
        k += 1
    return k


def exposure(texts: Iterable[str], canaries: Iterable[str], trajectory: list[str]) -> dict:
    """Reference exposure in participant-visible text: canary identifiers, or the reference's own ordered tool
    sequence (its first >= 3 distinct tools, in order) spelled out in one text."""
    can = [c for c in canaries if c]
    seq = _first_distinct(trajectory)
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


def trajectory_in_sequences(sequences: Iterable[Iterable[str]], trajectory: list[str]) -> list[str]:
    """Hits of the reference's ordered tool sequence (>= 3 of its first distinct tools, in order) among the tool-name
    sequences an `ExposureTrace` extracted."""
    seq = _first_distinct(trajectory)
    if len(seq) < 3:
        return []
    out = set()
    for s in sequences:
        k = _ordered_prefix(list(s), seq)
        if k >= 3:
            out.add(f"ordered reference tool sequence {seq[:k]}")
    return sorted(out)


def sandbox_can_import_crucible() -> bool:
    """Whether participant code in the sandbox could import the crucible package (and so a reference): the crucible
    source tree on the sandbox's readable paths or sys.path, or crucible among its approved packages."""
    import crucible
    from ..sandbox import runner
    root = os.path.realpath(os.path.dirname(crucible.__file__))
    cfg = runner.SandboxConfig()
    reads, syspath = runner._read_paths(cfg.packages)
    under = any(root == os.path.realpath(p) or root.startswith(os.path.realpath(p) + os.sep)
                for p in list(reads) + list(syspath))
    return bool(under or "crucible" in tuple(cfg.packages))


EVALUATOR_TOOL_NAME = re.compile(r"(^|[._])(score|scores|scoring|grade|grades|grading|evaluat\w*|verdict|reward|"
                                 r"rewards|objective|objectives|progress|success)([._]|$)", re.I)
EVALUATOR_TOOL_DOC = re.compile(r"\byour (current )?(objective|goal)\b|\b(objective|goal) (score|progress|status)\b|"
                                r"\bevaluator\b|\byour (current )?(score|grade|reward)\b", re.I)


def evaluator_like_tools(tools: list[dict]) -> list[str]:
    """Participant tools named or described like the evaluator (a score, grade, verdict, reward, objective
    progress). The kernel keeps evaluator-only endpoints away from participants (D-19); a domain must not add one."""
    out = []
    for t in tools:
        n = str(t.get("name") or "")
        tail = n.split(".", 1)[-1]
        doc = str(t.get("description") or t.get("doc") or "")
        if EVALUATOR_TOOL_NAME.search(tail) or EVALUATOR_TOOL_DOC.search(doc):
            out.append(n)
    return sorted(out)


def _calls_evaluator(world: World, d: Domain, principal: str) -> list[str]:
    """Participant endpoints whose handler is, or calls by name, the domain's evaluator."""
    out = []
    ev = [f for f in (d.evaluate,) if f is not None]
    names = {getattr(f, "__name__", "") for f in ev} - {"", "<lambda>"}
    for t in world.available_tools(principal):
        ep = world.endpoints.get(t["name"])
        h = getattr(ep, "handler", None)
        f = getattr(h, "__func__", h)
        if f is None:
            continue
        if any(f is e for e in ev):
            out.append(t["name"])
            continue
        code = getattr(f, "__code__", None)
        if code is not None and names & set(code.co_names) and \
                any(getattr(e, "__module__", None) == getattr(f, "__module__", None) for e in ev):
            out.append(t["name"])
    return sorted(out)


# ---------------------------------------------------------------------------------------------- dice (SP-9.16)
DICE_WORDS = re.compile(r"fail|success|succeed|error|break|broke|crash|reject|accept|win|lose|luck|chance|outage|"
                        r"bug|defect|fault|offline|\bdown\b|\bok\b|\bpass(es|ed)?\b", re.I)
BODY_WORDS = re.compile(r"\braise\b|\.fail\(|fail|\bdown\b|offline|\bstatus\b|broken|crash|outage|\bok\b|error|"
                        r"reject", re.I)
PRAGMA = re.compile(r"#\s*fidelity:\s*([A-Za-z0-9_.:\-]+)")
DRAWS = frozenset({"random", "uniform", "rand", "random_sample", "integers", "randint", "normal", "standard_normal",
                   "exponential", "beta", "gamma", "weibull", "lognormal", "triangular", "randrange"})


def _fidelity_ids() -> set[str]:
    import crucible
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(crucible.__file__))), "docs", "v5",
                     "FIDELITY_REGISTER.md")
    try:
        txt = open(p, encoding="utf-8").read()
    except OSError:
        return set()
    return set(re.findall(r"`([A-Za-z0-9_.:\-]+)`", txt)) | set(re.findall(r"\|\s*([A-Za-z0-9_.:\-]+)\s*\|", txt))


def _is_draw(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in DRAWS


def _contains_draw(node) -> bool:
    return any(_is_draw(n) for n in ast.walk(node))


def _bool_choice(node) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "choice"):
        return False
    if not any(k.arg == "p" for k in node.keywords):
        return False
    opts = node.args[0] if node.args else None
    if isinstance(opts, (ast.List, ast.Tuple)) and opts.elts and all(
            isinstance(e, ast.Constant) and (isinstance(e.value, bool) or
                                             (isinstance(e.value, str) and DICE_WORDS.search(e.value)))
            for e in opts.elts):
        return True
    return False


def dice_scan(paths: Iterable[str]) -> dict:
    """Arbitrary failure probabilities in source files: random draws that decide a failure or success - a draw
    compared with a threshold (``rng.random() < p``, ``rng.integers(0, 10) < 3``, ``rng.uniform() > q``), a
    binomial draw, a weighted choice between True/False or outcome words - whose expression, assigned name or branch
    speaks of failure/success (or whose branch raises, fails, rejects, marks something down or returns
    ``ok: False``), and that carry no ``# fidelity: <id>`` pragma (on the line or the one above) naming a
    FIDELITY_REGISTER record - an empirically grounded hazard. Difficulty must come from complexity, scarcity and
    uncertainty instead."""
    ids = None
    hits, exempt, files = [], [], 0
    seen_sites = set()
    for path in dict.fromkeys(paths):
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
            cand, always = None, False
            if isinstance(node, ast.Compare) and (_contains_draw(node.left) or
                                                  any(_contains_draw(c) for c in node.comparators)):
                cand = node
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ("binomial", "bernoulli"):
                cand = node
            elif _bool_choice(node):
                cand, always = node, True
            if cand is None:
                continue
            seg = ast.get_source_segment(src, cand) or ""
            ctx_txt = seg
            par = parents.get(cand)
            while par is not None and not isinstance(par, (ast.If, ast.IfExp, ast.Assign, ast.AugAssign,
                                                           ast.AnnAssign, ast.Return, ast.FunctionDef, ast.While)):
                par = parents.get(par)
            body_fail = False
            if isinstance(par, (ast.If, ast.While)):
                body = "\n".join(ast.get_source_segment(src, b) or "" for b in par.body)
                body_fail = bool(BODY_WORDS.search(body))
            elif isinstance(par, ast.FunctionDef):
                ctx_txt += " " + par.name + " " + " ".join(a.arg for a in par.args.args)
            elif par is not None:
                ctx_txt += " " + (ast.get_source_segment(src, par) or "")
                fn = par
                while fn is not None and not isinstance(fn, ast.FunctionDef):
                    fn = parents.get(fn)
                if isinstance(par, ast.Return) and fn is not None:
                    ctx_txt += " " + fn.name + " " + " ".join(a.arg for a in fn.args.args)
            if not (always or DICE_WORDS.search(ctx_txt) or body_fail):
                continue
            ln = cand.lineno
            if (path, ln) in seen_sites:
                continue
            seen_sites.add((path, ln))
            m = PRAGMA.search(" ".join(lines[max(0, ln - 2):ln]))
            if m:
                ids = _fidelity_ids() if ids is None else ids
                if m.group(1) in ids:
                    exempt.append({"file": path, "line": ln, "fidelity": m.group(1)})
                    continue
            hits.append({"file": path, "line": ln, "code": seg[:120]})
    return {"files": files, "mechanisms": len(hits), "hits": hits, "exempt": exempt}


def domain_sources(d: Domain) -> list[str]:
    """Source files of a domain: the files defining its build, objective, goal generator and evaluator, plus
    `sources` (relative paths are taken from crucible/domains/)."""
    out = []
    fns = [d.build, d.evaluate, d.objective]
    if d.goal is not None:
        fns += [d.goal.draw, d.goal.success, d.goal.text]
    for fn in fns:
        if fn is None:
            continue
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


def world_sources(world: World) -> list[str]:
    """Source files of the kernel modules of a built world (each module class and its bases, the kernel excluded):
    where the mechanisms of a civilisation-backed domain actually live."""
    import crucible
    core = os.path.join(os.path.dirname(os.path.abspath(crucible.__file__)), "core")
    out = []
    for m in world.modules.values():
        for c in type(m).__mro__:
            if c is object:
                continue
            try:
                f = inspect.getsourcefile(c)
            except TypeError:
                continue
            if f and not os.path.abspath(f).startswith(core + os.sep) and f not in out:
                out.append(f)
    return sorted(out)


# ============================================================================================ the deep check
_CACHE: dict[tuple, Any] = {}


def cached_run(domain: str, config: str, seed: int, policy: str, days: float | None = None) -> dict:
    key = (domain, config, int(seed), policy, days)
    if key not in _CACHE:
        _CACHE[key] = run(domain, config, seed, policy=policy, days=days, queries=(policy == "null"))
    return _CACHE[key]


def structure_report(d: Domain, seeds: Iterable[int] = (0, 1, 2), activity: dict | None = None) -> dict:
    """Build every configuration on `seeds` (genesis only, one world id per configuration so that only the seed
    differs): fingerprints, structure digests (stable across seeds?), distinct instances per configuration
    (distinct genesis CONTENT, `instance_digest`) and the tables that vary, distinct training structures, the
    unfamiliarity of each evaluation configuration, and the framework-measured core digests (no domain hook) used
    to compare configurations across domains (N.2, N.6, N.7, SP-9.0). `activity` ({config: modules}) adds the
    modules each configuration's null run engaged (`check` passes them). A structure hook that reaches for a
    declaration (spec, module objects) is an error of its configuration."""
    activity = activity or {}
    seeds = [int(s) for s in seeds]
    fps, digs, core, inst, stable, errors, vary, sources = {}, {}, {}, {}, {}, {}, {}, set()
    for c in d.configs:
        ds, ins, tabs = [], set(), []
        for s in seeds:
            try:
                w = world(d.name, c.name, s, world_id=f"d5-{d.name}-{c.name}")
            except Exception as e:  # noqa: BLE001 - reported: a configuration that cannot be built counts as failing
                errors[f"{c.name}@{s}"] = f"{type(e).__name__}: {e}"[:300]
                continue
            try:
                act = activity.get(c.name, ())
                fp = fingerprint(w, d, act)
                if c.name not in fps:
                    fps[c.name] = fp
                    core[c.name] = structure_digest(fingerprint(w, None, act))
                    sources.update(world_sources(w))
                ds.append(structure_digest(fp))
                ins.add(instance_digest(w))
                tabs.append(table_digests(w))
            except Exception as e:  # noqa: BLE001 - a hook that fails or reads declarations
                errors[f"{c.name}@{s}"] = f"structure: {type(e).__name__}: {e}"[:300]
            finally:
                w.close()
        digs[c.name] = ds[0] if ds else None
        stable[c.name] = len(set(ds)) == 1 and len(ds) == len(seeds)
        inst[c.name] = len(ins)
        vary[c.name] = sorted(t for t in (tabs[0] if tabs else {}) if len({tb.get(t) for tb in tabs}) > 1)
    hook = reads_declarations(d.structure) if d.structure is not None else []
    if hook:
        errors["structure_hook"] = "; ".join(hook)[:300]
    train = [c.name for c in d.split("train") if c.name in fps]
    unf = {c.name: unfamiliarity([fps[t] for t in train], fps[c.name]) for c in d.configs
           if c.split != "train" and c.name in fps}
    same_pairs = []
    names = [c.name for c in d.configs if digs.get(c.name)]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if digs[a] == digs[b]:
                same_pairs.append(f"{a}={b}")
    return {"fingerprints": fps, "digests": digs, "core_digests": core, "stable_across_seeds": stable,
            "instances": inst, "varying_tables": vary, "seeds": seeds,
            "train_names": [c.name for c in d.split("train")],
            "train_structures": len({digs[c] for c in train if digs.get(c)}),
            "train_configs": len(d.split("train")), "unfamiliarity": unf, "same_structure_pairs": same_pairs,
            "errors": errors, "hook_reads_declarations": hook, "world_sources": sorted(sources),
            "compositional_eval": sum(1 for c in d.split("eval_compositional")
                                      if unf.get(c.name, {}).get("compositional")),
            "structural_eval": sum(1 for c in d.split("eval_structural") if unf.get(c.name, {}).get("structural"))}


DECLARATION_ATTRS = frozenset({"spec", "config", "cfg", "params", "runtime", "_w", "world", "hooks", "processes",
                               "endpoints", "__dict__", "__globals__", "get_objects", "_getframe", "f_back"})


def reads_declarations(fn) -> list[str]:
    """Static check of a structure hook: attribute accesses of declarations (spec, parameters, module
    configuration), introspection and computed getattr. The hook also only ever receives a `BuiltWorld`."""
    if fn is None:
        return []
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return [f"{getattr(fn, '__qualname__', fn)}: source unavailable (cannot be checked)"]
    hits = []
    name = getattr(fn, "__qualname__", str(fn))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in DECLARATION_ATTRS:
            hits.append(f"{name}:{node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("getattr", "vars", "globals") and \
                not (node.func.id == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                     and node.args[1].value not in DECLARATION_ATTRS):
            hits.append(f"{name}:{node.lineno}: {node.func.id}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [getattr(node, "module", None) or ""] + [a.name for a in node.names]
            if any(m in ("gc", "inspect", "sys") for m in mods):
                hits.append(f"{name}:{node.lineno}: imports {mods}")
    return hits


def evaluator_report(d: Domain, config: str, seed: int, days: float | None = None,
                     ablate: Iterable[str] | None = None) -> dict:
    """The evaluator is an independent, evaluator-side function of recorded world state:

    * run the reference (cached for `check`), rebuild the world from its event log alone (`World.replay`) and
      evaluate both; the verdicts must be equal;
    * no participant tool is named or described like the evaluator, and none calls the domain's evaluator;
      in the default mode evaluator data are hidden from participants;
    * its verdict needs the autonomous participants: the reference is run again with the scheduled processes that
      made the actors' and the population's decisions in the null run frozen (`ablate`, default: measured from the
      null run); success there means the verdict is obtainable by the seat alone (a participant-controlled fact,
      not an independent outcome)."""
    from ..core import evaluator_only as EO
    key = (d.name, config, int(seed), "reference", days)
    r = run(d.name, config, seed, policy="reference", days=days, keep_env=True)
    env = r.pop("env")
    _CACHE.setdefault(key, r)
    try:
        w = env.world
        me = r["seats"][0]
        obj = r["objective"]
        live = d.judge(w, me, obj)
        events = [e.to_dict() for e in w.events_range(0, w.seq)]
        rw = World.replay(w.spec, events, world_id="evaluator-replay", store=getattr(w, "store", None))
        again = d.judge(rw, me, obj)
        canon = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
        tools = env.tools(me)
        out = {"config": config, "seed": int(seed), "replay_equal": canon(live) == canon(again),
               "replay_digest_equal": rw.digest() == w.digest(), "evaluator_hidden": bool(EO.hidden(w)),
               "verdict_keys": sorted(live) if isinstance(live, dict) else [],
               "participant_tools_named_like_evaluator": evaluator_like_tools(tools),
               "participant_tools_calling_evaluator": _calls_evaluator(w, d, me),
               "has_success_and_score": isinstance(live, dict) and isinstance(live.get("success"), bool)
               and isinstance(live.get("score"), (int, float)), "reference_success": bool(r["success"])}
        rw.close()
    finally:
        env.close()
    if ablate is None:
        null = cached_run(d.name, config, seed, "null", days)
        ablate = sorted(set(null["actors"].get("processes") or []) | set(null["population"].get("processes") or []))
    ablate = sorted(set(ablate))
    out["ablated_processes"] = ablate
    if ablate:
        ab = run(d.name, config, seed, policy="reference", days=days, freeze=ablate)
        out["ablation_success"] = bool(ab["success"])
        out["ablation_score"] = ab.get("score")
        out["needs_autonomous_participants"] = bool(r["success"]) and not ab["success"]
    else:
        out["ablation_success"] = None
        out["needs_autonomous_participants"] = None
    out["ok"] = bool(out["replay_equal"] and out["replay_digest_equal"] and out["evaluator_hidden"]
                     and out["has_success_and_score"] and not out["participant_tools_named_like_evaluator"]
                     and not out["participant_tools_calling_evaluator"] and out["needs_autonomous_participants"])
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
        ev = d.judge(w, me, d.make_objective(w, me, d.reference_difficulty)) if ok_seat else None
        out = {"config": c.name, "profile": profile, "modules": len(w.modules), "seat": me, "seat_present": ok_seat,
               "proc_failed": count_events(w, ("proc.failed",), since),
               "invariant_errors": len(w.check_invariants()), "evaluator_ran": isinstance(ev, dict)}
        w.close()
    except Exception as e:  # noqa: BLE001 - reported: the domain does not compose
        return {"config": c.name, "profile": profile, "composes": False, "error": f"{type(e).__name__}: {e}"[:300]}
    out["composes"] = bool(out["seat_present"] and out["evaluator_ran"] and out["proc_failed"] == 0
                           and out["invariant_errors"] == 0)
    return out


def goal_report(d: Domain, seed: int = 0) -> dict:
    """The objective generator: crucible.goals (`GoalFamily`) or the domain's own callable; generated objectives
    for every difficulty level on the reference configuration's genesis world - their texts must differ (a
    difficulty knob), every success predicate must be valid, and the generator must not draw from the world's
    streams (the framework hands it `goal_rng`)."""
    c = (list(d.reference_configs) or [cc.name for cc in d.split("train")[:1]])[0]
    out: dict = {"generator": "crucible.goals" if d.goal is not None else "domain",
                 "family": d.goal.name if d.goal is not None else None}
    try:
        w = world(d.name, c, seed)
        try:
            me = d.principal_of(w) or "p:none"
            objs = {int(k): d.make_objective(w, me, int(k)) for k in d.difficulties}
        finally:
            w.close()
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        return out
    texts = {k: o.get("text") for k, o in objs.items()}
    out["texts"] = texts
    out["distinct_texts"] = len(set(texts.values()))
    out["params"] = {k: o.get("params") for k, o in objs.items()}
    out["ok"] = bool(d.goal is not None and len(objs) >= 2 and out["distinct_texts"] == len(objs)
                     and all(isinstance(o.get("success"), dict) for o in objs.values()))
    return out


def check(d: Domain | str, *, structure_seeds: Iterable[int] = (0, 1, 2), seeds: Iterable[int] | None = None,
          days: float | None = None, configs: Iterable[str] | None = None, tier3: bool = True,
          tier3_profile: str = "balanced", all_configs: bool = True) -> dict:
    """Every Gate N / SP-9.0 / SP-9.16 measurement of one domain, on built and run worlds:

    * `static`: `validate` problems; `structure`: `structure_report`;
    * `runs`: the reference and the null baseline on EVERY configuration (the first reference seed; the reference
      configurations on every reference seed), every degenerate probe and the generic probes on the reference
      configurations (through `crucible.experience.make(2, ...)`, observation_only; references in a separate
      process);
    * `feasible` (reference succeeds on >= 1 and on a majority of the reference instances), `null_fails` (no null
      run succeeds anywhere), `probes_fail`; `per_config`: success, null and cleanliness per configuration (an
      evaluation configuration counts only when it runs clean, its reference succeeds and its null baseline fails);
    * `mechanics` (union over reference runs), `autonomy` (null runs: declared actors active by attributed
      decisions, population decisions), `open_ended` (the seat's tools), `observation_only` (default mode,
      evaluator hidden, no nonzero reward in any run);
    * `evaluator`: `evaluator_report` (replay, evaluator tools, autonomy ablation); `goal`: `goal_report`;
    * `isolation`: the reference reads no world internals (static) and ran isolated; reference exposure in
      everything the seat received (canaries in every run; its ordered tool path in the null runs' observations
      and results, the read-only query results, the tool docs and the objectives); tools named after it; sandbox
      importability;
    * `dice`: arbitrary failure probabilities in the domain's sources AND the kernel module files of its worlds;
    * `tier3`: composition into a civilisation."""
    d = REGISTRY[d] if isinstance(d, str) else d
    t0 = time.perf_counter()
    rep: dict = {"domain": d.name, "spec_part": d.spec_part, "api": API_VERSION, "static": validate(d)}
    ref_cfgs = list(configs or d.reference_configs or [c.name for c in d.split("train")[:1]])
    seeds = [int(s) for s in (seeds if seeds is not None else d.reference_seeds)]
    s0 = seeds[0]
    every = [c.name for c in d.configs] if all_configs else ref_cfgs
    probe_names = [p.name for p in d.degenerate] + [p.name for p in GENERIC_PROBES]
    runs: dict[str, list] = {"reference": [], "null": [], **{p: [] for p in probe_names}}
    errors = {}

    def go(pol, c, s):
        try:
            r = cached_run(d.name, c, s, pol, days)
            runs[pol].append(r)
            return r
        except Exception as e:  # noqa: BLE001 - a run that crashes is a failed run, reported
            errors[f"{pol}:{c}@{s}"] = f"{type(e).__name__}: {e}"[:400]
            return None
    plan = [(c, s) for c in ref_cfgs for s in seeds] + [(c, s0) for c in every if c not in ref_cfgs]
    try:        # first: its reference and null runs are cached for the loops below
        rep["evaluator"] = evaluator_report(d, ref_cfgs[0], s0, days)
    except Exception as e:  # noqa: BLE001 - reported: the evaluator could not be checked
        rep["evaluator"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    for c, s in plan:
        go("reference", c, s)
    for c, s in plan:
        go("null", c, s)
    for c in ref_cfgs:
        for p in probe_names:
            go(p, c, s0)
    rep["run_errors"] = errors
    activity = {r["config"]: r.get("engaged_modules") or [] for r in runs["null"] if r["seed"] == s0}
    rep["structure"] = structure_report(d, structure_seeds, activity)
    keys = ("config", "seed", "policy", "success", "score", "events", "proc_failed", "invariant_violations",
            "nonzero_rewards", "isolated")
    rep["runs"] = {k: [{x: r.get(x) for x in keys} for r in v] for k, v in runs.items()}
    ref_main = [r for r in runs["reference"] if r["config"] in ref_cfgs]
    null = runs["null"]
    n_ok = sum(1 for r in ref_main if r["success"])
    rep["reference_successes"], rep["reference_runs"] = n_ok, len(ref_main)
    rep["feasible"] = bool(ref_main) and n_ok >= 1 and 2 * n_ok > len(ref_main) and \
        not any(k.startswith("reference:") and k.split(":", 1)[1].split("@")[0] in ref_cfgs for k in errors)
    rep["null_fails"] = bool(null) and not any(r["success"] for r in null) and \
        not any(k.startswith("null:") for k in errors)
    rep["probes"] = {p: {"runs": len(runs[p]), "successes": sum(1 for r in runs[p] if r["success"])}
                     for p in probe_names}
    rep["degenerate_declared"] = len(d.degenerate)
    rep["probes_fail"] = bool(d.degenerate) and all(v["runs"] > 0 and v["successes"] == 0
                                                    for v in rep["probes"].values())
    allruns = [r for v in runs.values() for r in v]
    rep["process_failures"] = sum(r["proc_failed"] for r in allruns)
    rep["invariant_violations"] = sum(r["invariant_violations"] for r in allruns)
    per = {}
    for c in d.configs:
        rr = [r for r in runs["reference"] if r["config"] == c.name]
        nn = [r for r in null if r["config"] == c.name]
        mine = [r for r in allruns if r["config"] == c.name]
        per[c.name] = {"split": c.split, "reference_success": [r["success"] for r in rr],
                       "null_success": [r["success"] for r in nn],
                       "proc_failed": sum(r["proc_failed"] for r in mine),
                       "invariant_violations": sum(r["invariant_violations"] for r in mine),
                       "errors": sorted(k for k in errors if k.split(":", 1)[1].split("@")[0] == c.name)}
        p = per[c.name]
        p["clean"] = bool(rr) and bool(nn) and not p["errors"] and p["proc_failed"] == 0 and \
            p["invariant_violations"] == 0
        p["feasible"] = p["clean"] and any(p["reference_success"]) and not any(p["null_success"])
    rep["per_config"] = per
    # mechanics: union of the reference runs' recorded evidence on the reference configurations
    counts: Counter = Counter()
    for r in ref_main:
        for m, n in r["mechanics"]["counts"].items():
            counts[m] += n
    rep["mechanics"] = {"listed": len(d.required_mechanics),
                        "exercised": sum(1 for m in d.required_mechanics if counts.get(m, 0) > 0),
                        "missing": sorted(m for m in d.required_mechanics if counts.get(m, 0) == 0),
                        "counts": dict(counts), "self_reported": d.mechanic_probes() is None,
                        "unverified": sorted({m for r in ref_main for m in r["mechanics"].get("unverified", {})}),
                        "rejected": sorted({m for r in ref_main for m in r["mechanics"].get("rejected", {})}),
                        "indistinct": sorted({f"{m}={o}" for r in ref_main
                                              for m, o in r["mechanics"].get("indistinct", {}).items()}),
                        "spec_list_matches": tuple(d.required_mechanics) == tuple(PART9[d.spec_part]["mechanics"])
                        if d.spec_part in PART9 else True}
    # autonomy: what happens while the seats do nothing (the reference configurations' null runs)
    act: Counter = Counter()
    popd, popk = 0, Counter()
    null_main = [r for r in null if r["config"] in ref_cfgs]
    for r in null_main:
        for a, n in r["actors"]["counts"].items():
            act[a] += n
        popd += r["population"]["decisions"]
        popk.update(r["population"]["activity_kinds"])
    rep["autonomy"] = {"actors_declared": len(d.actors), "actors_active": sum(1 for a in d.actors if act[a.name] > 0),
                       "actor_counts": dict(act), "population_decisions": popd,
                       "population_activity_kinds": dict(popk),
                       "unverified": sorted({a for r in null_main for a in r["actors"].get("unverified", {})})}
    rep["autonomous"] = bool(d.actors) and rep["autonomy"]["actors_active"] == len(d.actors) and popd > 0 \
        and sum(popk.values()) > 0
    # open-ended action space, observation-only default, no reward
    from ..core import evaluator_only as EO
    from ..experience.modes import OBSERVATION_ONLY
    env = make_env(d.name, ref_cfgs[0], s0, difficulty=d.reference_difficulty)
    try:
        me = env.agents()[0]
        tools = env.tools(me)
        rep["open_ended"] = open_ended(tools)
        hidden = bool(EO.hidden(env.world))
        mode = env.reward_mode
        objs = [d.make_objective(env.world, me, k) for k in d.difficulties]
    finally:
        env.close()
    nonzero = sum(int(r.get("nonzero_rewards") or 0) for r in allruns)
    rep["observation_only"] = {"default_mode": mode, "evaluator_hidden": hidden, "nonzero_rewards": nonzero,
                               "runs": len(allruns), "ok": mode == OBSERVATION_ONLY and hidden and nonzero == 0
                               and bool(allruns)}
    rep["goal"] = goal_report(d, s0)
    # isolation of the reference
    try:
        pol = d.reference_policy(me, objs[0] if objs else {})
        reads = reads_world(pol) + reads_world(d.reference)
    except Exception as e:  # noqa: BLE001
        pol, reads = None, [f"reference could not be built for inspection: {type(e).__name__}: {e}"[:200]]
    canaries = reference_canaries(d, pol)
    first = next((r for r in ref_main if r["config"] == ref_cfgs[0] and r["seed"] == s0), None)
    traj = first["drive"]["agents"][first["seats"][0]]["tools"] if first else []
    hits = set()
    for r in allruns:
        hits.update(r["exposure"]["canary_hits"])
    for r in null:
        hits.update(trajectory_in_sequences(r["exposure"]["sequences"], traj))
    static_texts = [json.dumps({k: t.get(k) for k in ("name", "doc", "description", "inputSchema")}) for t in tools]
    static_texts += [o.get("text") or "" for o in objs] + [json.dumps(public_objective(o)) for o in objs]
    hits.update(exposure(static_texts, canaries, traj)["hits"])
    names = {t.get("name") for t in tools}
    named = sorted(c for c in canaries if c in names or any(c.lower() in str(n).lower() for n in names))
    sbx = sandbox_can_import_crucible()
    isolated = bool(ref_main) and all(r.get("isolated") for r in runs["reference"])
    rep["isolation"] = {"reference_reads_world": reads, "canaries": canaries, "exposure_hits": sorted(hits),
                        "tools_named_after_reference": named, "sandbox_imports_crucible": sbx,
                        "reference_isolated_process": isolated, "trajectory_head": _first_distinct(traj),
                        "texts_scanned": sum(r["exposure"]["texts"] for r in allruns) + len(static_texts),
                        "queried_tools": sorted({q for r in null for q in r.get("queried_tools") or []})}
    rep["reference_paths_exposed"] = len(hits) + len(named) + int(sbx)
    rep["reference_tool_only"] = not reads and isolated
    srcs = domain_sources(d) + list(rep["structure"].get("world_sources") or [])
    rep["dice"] = dice_scan(srcs)
    rep["tier3"] = tier3_report(d, tier3_profile) if tier3 else None
    rep["wall_s"] = round(time.perf_counter() - t0, 2)
    return rep


# ============================================================================================ per-section (SP-9.<n>)
def section_metrics(domain: str | Domain, *, configs: Iterable[str] | None = None, seeds: Iterable[int] | None = None,
                    days: float | None = None) -> tuple[dict, list[dict]]:
    """The five metrics every SP-9.<n> requirement shares, measured on integrated reference runs through
    `crucible.experience.make(2, ...)`: required_mechanics_listed / _exercised / _missing (recorded evidence summed
    over the runs), observation_only_default (default mode, evaluator hidden, no nonzero reward in the runs) and
    reachable_through_experience (the seats are the learner's agents and have tools). Returns (metrics, runs);
    per-mechanic evidence counts are included as ``events_<mechanic>``."""
    d = REGISTRY[domain] if isinstance(domain, str) else domain
    cfgs = list(configs or d.reference_configs or [c.name for c in d.split("train")[:1]])
    seeds = [int(s) for s in (seeds if seeds is not None else d.reference_seeds)]
    runs = [cached_run(d.name, c, s, "reference", days) for c in cfgs for s in seeds]
    counts = {m: sum(int(r["mechanics"]["counts"].get(m, 0)) for r in runs) for m in d.required_mechanics}
    from ..core import evaluator_only as EO
    from ..experience.modes import OBSERVATION_ONLY
    env = make_env(d.name, cfgs[0], seeds[0], difficulty=d.reference_difficulty)
    agents = env.agents()
    reach = bool(agents) and (not d.seats or agents == d.seat_principals()) and bool(env.tools(agents[0]))
    env.advance(HOUR)
    oo = env.reward_mode == OBSERVATION_ONLY and bool(EO.hidden(env.world)) and env.reward(agents[0]) == 0.0 and \
        all(int(r.get("nonzero_rewards") or 0) == 0 for r in runs)
    env.close()
    m = {"required_mechanics_listed": len(d.required_mechanics),
         "required_mechanics_exercised": sum(1 for n in counts.values() if n > 0),
         "required_mechanics_missing": sum(1 for n in counts.values() if n == 0),
         "observation_only_default": bool(oo), "reachable_through_experience": bool(reach),
         "spec_list_matches": d.spec_part in PART9 and tuple(d.required_mechanics) ==
         tuple(PART9[d.spec_part]["mechanics"]), "reference_runs": len(runs),
         "reference_successes": sum(1 for r in runs if r["success"]),
         "process_failures": sum(r["proc_failed"] for r in runs),
         "unverified_evidence": ",".join(sorted({x for r in runs for x in r["mechanics"].get("unverified", {})}))
         or "none",
         "indistinct_mechanics": ",".join(sorted({x for r in runs for x in r["mechanics"].get("indistinct", {})}))
         or "none"}
    for name, n in counts.items():
        m["events_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")] = int(n)
    return m, runs


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
    per = rep.get("per_config") or {}

    def eval_ok(split, key):
        return any(per.get(c, {}).get("feasible") and st["unfamiliarity"].get(c, {}).get(key)
                   for c, p in per.items() if p.get("split") == split)
    feasible = bool(rep["feasible"])
    v = {
        "static_ok": not rep["static"],
        "builds": not st["errors"],
        "runs_clean": not rep["run_errors"] and rep["process_failures"] == 0 and rep["invariant_violations"] == 0,
        "train_structures": len(stable_train),
        "min_instances": min(st["instances"].values()) if st["instances"] else 0,
        "compositional_eval": eval_ok("eval_compositional", "compositional"),
        "structural_eval": eval_ok("eval_structural", "structural"),
        "same_structure_pairs": len(st["same_structure_pairs"]) + sum(1 for s in st["stable_across_seeds"].values()
                                                                      if not s),
        "mechanics_ok": rep["mechanics"]["exercised"] == rep["mechanics"]["listed"] and
        rep["mechanics"]["spec_list_matches"] and not rep["mechanics"]["self_reported"],
        "mechanics_missing": len(rep["mechanics"]["missing"]) if rep["mechanics"]["spec_list_matches"] else
        rep["mechanics"]["listed"],
        "evaluator_ok": bool(ev.get("ok")) and feasible and bool(rep["null_fails"]),
        "reference_ok": feasible and bool(rep["reference_tool_only"]),
        "exposed": rep["reference_paths_exposed"],
        "autonomous": rep["autonomous"],
        "open_ended": rep["open_ended"]["open_ended"],
        "observation_only": rep["observation_only"]["ok"],
        "tier3": bool(t3.get("composes")),
        # SP-9.16: a null baseline (or a probe) failing shows something only where the objective is feasible
        "null_fails": feasible and bool(rep["null_fails"]),
        "probes_ok": feasible and bool(rep["probes_fail"]),
        "dice": rep["dice"]["mechanisms"],
        "objective_generated": bool((rep.get("goal") or {}).get("ok")),
    }
    v["implemented"] = all((v["static_ok"], v["builds"], v["runs_clean"], v["train_structures"] >= 3,
                            v["min_instances"] >= 3, v["compositional_eval"], v["structural_eval"],
                            v["mechanics_ok"], v["evaluator_ok"], v["reference_ok"], v["exposed"] == 0,
                            v["observation_only"], v["objective_generated"]))
    return v


def cross_domain_same_structure(reports: dict[str, dict]) -> list[str]:
    """Configurations of DIFFERENT domains whose built worlds have the same framework-measured structure (core
    fingerprint: engaged modules, participant types, civilisation structure): the same world under another task
    prompt."""
    rows = [(n, c, dg) for n, rep in sorted(reports.items())
            for c, dg in sorted((rep["structure"].get("core_digests") or {}).items()) if dg]
    out = []
    for i, (n1, c1, g1) in enumerate(rows):
        for n2, c2, g2 in rows[i + 1:]:
            if n1 != n2 and g1 == g2:
                out.append(f"{n1}:{c1}={n2}:{c2}")
    return out


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
    cross = cross_domain_same_structure({n: r for n, r in reports.items() if r["spec_part"] in by_part})
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
                                                               for v in vs) + len(cross),
        "cross_domain_same_structure": ";".join(cross) or "none",
        "domains_composable_into_tier3": part("tier3"),
        "domains_where_null_baseline_fails": part("null_fails"),
        "arbitrary_failure_probability_mechanisms": sum(v["dice"] for vs in by_part.values() for v in vs),
        "domains_with_degenerate_strategy_probe": part("probes_ok"),
        "domains_observation_only_default": part("observation_only"),
        "domains_with_generated_objectives": part("objective_generated"),
        "domain_load_errors": _load_errors(),
    }
    return m


def _load_errors() -> str:
    from . import LOAD_ERRORS
    return ";".join(f"{k}: {v}" for k, v in sorted(LOAD_ERRORS.items()))[:3000] or "none"


def summary() -> dict:
    return {n: {"title": d.title, "spec_part": d.spec_part, "configs": {c.name: c.split for c in d.configs},
                "required_mechanics": list(d.required_mechanics), "seats": [s.handle for s in d.seats],
                "mode": d.mode, "construction": d.construction, "fixture": d.fixture,
                "goal": d.goal.name if d.goal is not None else None} for n, d in sorted(REGISTRY.items())}


_FRAMEWORK_POLICIES = (ScriptedReference, Idle, RandomTools)
