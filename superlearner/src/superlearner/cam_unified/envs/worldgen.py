"""Procedurally generated hidden-mechanism worlds with disjoint structural holdouts (spec 13.1-13.3; manifest W01).

Families are compositions of causal *motifs* over Crucible's three independently written engines, registered with
`crucible.mechanisms.base.mech_family` under versioned names (``cw1_<engine>_<motifs>``; a recipe change gets a new
version prefix because Crucible's task caches are keyed by name):

* **scm** (rule-compositional causal graphs): recipes that require motif blocks from {delay, stock, feedback, gate,
  threshold, interaction, confounder, invu} plus the regime switch, with lagged gauges (``gauge_lag_p``), gauge
  noise and distractor levers drawn per seed (partial observability varies within a family). Training contains
  every training motif alone and in most pairs and some triples; selected pairs and triples are **held out
  entirely** (compositional holdouts: every motif seen in training, the combination never), and the inverted-U
  response appears only in held-out families (structural holdout).
* **abm** (agent-based markets): recombinations of the training rules (word of mouth, congestion, private-utility
  choice) on other base kinds for training; the rating gate and local network effects only in held-out families.
* **techsys** (queueing systems): scenarios with gauge kinds hidden (``hide``), held out only (every Crucible techsys
  scenario is already a val/test family; training on any of them would leak Crucible's own holdouts).

`catalogue()` is deterministic; `register_families()` registers it idempotently (call it in every process that
builds tasks - spawned workers do not inherit in-process registrations; use ``workers=1`` or this as the worker
initializer). `validate_catalogue()` checks the holdout disjointness invariants and builds/runs tasks.

`SplitGuard` refuses validation/test/held-out families (Crucible's mechanism families, Crucible's world families
and the catalogue's) and validation-range seeds in any training stream. `Surface` gives per-episode presentation
randomisation (names, units, ordering, wording) for invariance pairs: two surfaces over one task are different
appearances of identical dynamics.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

import numpy as np

VERSION = "cw1"
SCM_TRAIN_MOTIFS = ("delay", "stock", "feedback", "regime", "gate", "threshold", "interaction", "confounder")
SCM_HELDOUT_MOTIFS = ("invu",)
VAL_PAIRS = (("delay", "gate"), ("feedback", "threshold"), ("stock", "interaction"))
TEST_PAIRS = (("feedback", "confounder"), ("stock", "gate"), ("regime", "interaction"), ("delay", "confounder"))
N_TRAIN_TRIPLES = 6
N_VAL_TRIPLES = 2
N_TEST_TRIPLES = 3
STRUCTURAL = {"val": (("invu",), ("invu", "delay")), "test": (("invu", "feedback"), ("invu", "stock", "threshold"))}

# ABM: (kind, extra rules) -> split. Training rules: wom, congest (and the base private-utility choice). The rating
# gate (abm_trust_gate, val) and local network effects (abm_network_fx, test) stay out of training.
ABM_RECOMBINATIONS = {
    "util_congest": ("utility", {"congest": True}, "train", ("private_utility", "congestion")),
    "util_wom": ("utility", {"wom": True}, "train", ("private_utility", "word_of_mouth")),
    "congest_wom": ("congestion", {"wom": True}, "train", ("congestion", "word_of_mouth")),
    "wom_congest": ("wom", {"congest": True}, "train", ("word_of_mouth", "congestion")),
    "congest_gate": ("congestion", {"gate": True}, "val", ("congestion", "rating_gate")),
    "util_gate": ("utility", {"gate": True}, "val", ("private_utility", "rating_gate")),
    "wom_nfx": ("wom", {"nfx": True}, "test", ("word_of_mouth", "network_effects")),
    "congest_nfx": ("congestion", {"nfx": True}, "test", ("congestion", "network_effects")),
}
TECH_HIDDEN = {
    "pipeline_hidequeue": ("pipeline", ["queue"], "val"),
    "contention_hidequeue": ("contention", ["queue", "late"], "test"),
    "cascade_hideerr": ("cascade", ["err", "late"], "test"),
    "redundant_hidelat": ("redundant", ["lat50", "lat95", "late"], "test"),
    "lostwork_hideshed": ("lostwork", ["shed", "offered", "late"], "test"),
}

# which of the spec 13.1 diversity dimensions a motif exercises (documentation and coverage accounting)
DIMENSIONS = {
    "delay": ("delays", "partial_observability"), "stock": ("delays", "hidden_state"),
    "feedback": ("feedback_loops", "hidden_state"), "regime": ("changing_mechanisms",),
    "gate": ("nonlinearities", "dependencies"), "threshold": ("nonlinearities",),
    "interaction": ("nonlinearities", "dependencies"), "confounder": ("hidden_state", "partial_observability"),
    "invu": ("nonlinearities",), "private_utility": ("behavioral_populations", "competing_agents", "market_structures"),
    "congestion": ("resource_constraints", "feedback_loops", "competing_agents"),
    "word_of_mouth": ("network_effects", "feedback_loops", "behavioral_populations"),
    "rating_gate": ("nonlinearities", "behavioral_populations"), "network_effects": ("network_effects",),
    "queue": ("technical_failures", "dependencies", "resource_constraints"), "hidden_gauges": ("partial_observability",),
}


@dataclass(frozen=True)
class FamilyDef:
    name: str
    engine: str
    split: str                      # train | val | test
    motifs: tuple                   # motif vocabulary of this module
    relation: str                   # relation to the catalogue's training set: base | compositional | structural |
                                    # cross_impl
    params: dict = field(default_factory=dict, hash=False, compare=False)
    doc: str = ""

    @property
    def dimensions(self) -> set:
        return {d for m in self.motifs for d in DIMENSIONS.get(m, ())}


def _scm_recipe(motifs: tuple) -> dict:
    req = [m for m in motifs if m != "regime"]
    pool = {"direct": 2, "chain": 1, "fork": 1, "mediator": 1}
    for m in req:
        pool[m] = pool.get(m, 0) + 1
    dyn = any(m in motifs for m in ("delay", "stock", "feedback"))
    return {"required": req or ["direct", "chain"], "pool": pool,
            "n_blocks": [max(3, len(req) + 1), max(4, len(req) + 2)], "regime": "regime" in motifs,
            "horizon": 96 + (48 if dyn else 0) + (48 if "regime" in motifs else 0),
            "harmful_p": 0.8 if "invu" in motifs else 0.5, "dynamic": dyn}


def _name(engine: str, parts: Iterable[str]) -> str:
    return f"{VERSION}_{engine}_" + "_".join(parts)


def catalogue() -> list[FamilyDef]:
    """The full, deterministic family catalogue (registration-free)."""
    held_pairs = {tuple(sorted(p)) for p in VAL_PAIRS + TEST_PAIRS}
    train_pairs = [p for p in itertools.combinations(sorted(SCM_TRAIN_MOTIFS), 2) if p not in held_pairs]
    tp = set(train_pairs)
    triples = [t for t in itertools.combinations(sorted(SCM_TRAIN_MOTIFS), 3)
               if all(tuple(sorted(x)) in tp for x in itertools.combinations(t, 2))]
    rng = np.random.default_rng(7919)
    order = rng.permutation(len(triples))
    triples = [triples[i] for i in order]
    test_tri = triples[:N_TEST_TRIPLES]
    val_tri = triples[N_TEST_TRIPLES:N_TEST_TRIPLES + N_VAL_TRIPLES]
    train_tri = triples[N_TEST_TRIPLES + N_VAL_TRIPLES:N_TEST_TRIPLES + N_VAL_TRIPLES + N_TRAIN_TRIPLES]
    out: list[FamilyDef] = []

    def scm(ms: tuple, split: str, relation: str, doc: str):
        out.append(FamilyDef(_name("scm", ms), "scm", split, tuple(ms), relation, _scm_recipe(tuple(ms)), doc))
    for m in sorted(SCM_TRAIN_MOTIFS):
        if m == "regime":     # alone it would duplicate Crucible's test family scm_regime; trained in combinations
            continue
        scm((m,), "train", "base", f"Single motif: {m}.")
    for p in train_pairs:
        scm(p, "train", "base", f"Pair {p[0]} + {p[1]}.")
    for t in train_tri:
        scm(t, "train", "base", f"Triple {' + '.join(t)}.")
    for p in VAL_PAIRS:
        scm(tuple(sorted(p)), "val", "compositional", f"Held-out pair {p[0]} + {p[1]} (both trained alone).")
    for p in TEST_PAIRS:
        scm(tuple(sorted(p)), "test", "compositional", f"Held-out pair {p[0]} + {p[1]} (both trained alone).")
    for t in val_tri:
        scm(t, "val", "compositional", f"Held-out triple {' + '.join(t)} (every pair trained).")
    for t in test_tri:
        scm(t, "test", "compositional", f"Held-out triple {' + '.join(t)} (every pair trained).")
    for split, combos in STRUCTURAL.items():
        for c in combos:
            scm(tuple(c), split, "structural", f"Structural holdout: inverted-U response never in training ({c}).")
    for key, (kind, rules, split, ms) in ABM_RECOMBINATIONS.items():
        rel = "base" if split == "train" else "structural"
        out.append(FamilyDef(_name("abm", [key]), "abm", split, ms, rel, {"kind": kind, "rules": rules},
                             f"Agent-based market: {kind} kind with extra rules {sorted(rules)}."))
    for key, (scn, hide, split) in TECH_HIDDEN.items():
        out.append(FamilyDef(_name("tech", [key]), "techsys", split, ("queue", "hidden_gauges"), "cross_impl",
                             {"scenario": scn, "hide": list(hide)},
                             f"Queueing system ({scn}) with gauge kinds {hide} hidden."))
    return out


def _builder(fd: FamilyDef):
    if fd.engine == "scm":
        base = dict(fd.params)
        dyn = base.pop("dynamic")

        def build(rng, o, base=base, dyn=dyn):
            r = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                 for k, v in base.items()}
            # partial observability varies within the family: lagged gauges, gauge noise, distractor levers
            r["gauge_lag_p"] = float(rng.choice([0.0, 0.25, 0.45] if dyn else [0.0, 0.1, 0.25]))
            hi = float(rng.choice([0.02, 0.04, 0.08]))
            r["gauge_noise"] = [0.005, hi]
            r["distractors"] = [int(rng.integers(0, 2)), int(rng.integers(2, 4))]
            r["seats"] = int(o.get("seats", 1))
            return r
    elif fd.engine == "abm":
        def build(rng, o, p=fd.params):
            n = int(rng.choice([300, 450, 600]))
            return {**(o or {}), "kind": p["kind"], "rules": dict(p["rules"]), "n_customers": int(o.get("n_customers", n))}
    else:
        def build(rng, o, p=fd.params):
            return {**(o or {}), "scenario": p["scenario"], "hide": list(p["hide"])}
    build.__name__ = f"build_{fd.name}"
    return build


_REGISTERED: set = set()


def register_families(kernel_worlds: bool = False) -> list[str]:
    """Register every catalogue family with Crucible (idempotent). With `kernel_worlds`, also (re)register the
    kernel world families ``mech_<name>`` so tasks can be hosted in OpenWorld kernels (`Task.world_config`)."""
    import crucible
    crucible.load_all_modules()
    from crucible.mechanisms import load_engines
    from crucible.mechanisms.base import MECH_FAMILIES, mech_family
    load_engines()
    names = []
    for fd in catalogue():
        if fd.name not in MECH_FAMILIES:
            motifs = tuple(fd.motifs) + (("lagged_gauge",) if fd.engine == "scm" else ())
            mech_family(fd.name, fd.engine, fd.split, fd.relation, fd.doc, motifs=motifs)(_builder(fd))
        _REGISTERED.add(fd.name)
        names.append(fd.name)
    if kernel_worlds:
        from crucible.worldgen.mech_families import register_mech_world_families
        register_mech_world_families()
    return names


def families(split: Optional[str] = None, engine: Optional[str] = None) -> list[str]:
    return [f.name for f in catalogue() if (split is None or f.split == split) and (engine is None or f.engine == engine)]


def holdout_report() -> dict:
    """Disjointness invariants of the catalogue (all must hold): no held-out combination occurs inside any
    training family; every motif of a compositional holdout is trained alone; structural motifs never trained."""
    cat = catalogue()
    train = [f for f in cat if f.split == "train"]
    train_sets = [set(f.motifs) for f in train]
    problems = []
    for f in cat:
        if f.split == "train" or f.engine != "scm":
            continue
        ms = set(f.motifs)
        if f.relation == "compositional":
            if any(ms <= s for s in train_sets):
                problems.append(f"{f.name}: combination occurs in training")
            for m in ms:
                if not any(m in s for s in train_sets):
                    problems.append(f"{f.name}: motif {m} never trained")
        if f.relation == "structural" and not (ms & set(SCM_HELDOUT_MOTIFS)):
            problems.append(f"{f.name}: structural family without a held-out motif")
    for s in train_sets:
        if s & set(SCM_HELDOUT_MOTIFS):
            problems.append(f"training family uses held-out motif: {sorted(s)}")
    for key, (_, rules, split, _) in ABM_RECOMBINATIONS.items():
        if split == "train" and ({"gate", "nfx", "bundle", "commons"} & {k for k, v in rules.items() if v}):
            problems.append(f"abm {key}: held-out rule in training")
    covered = set().union(*(f.dimensions for f in train))
    return {"problems": problems, "crucible_relations": crucible_relations(), "n": {s: sum(1 for f in cat if f.split == s) for s in ("train", "val", "test")},
            "train_dimensions": sorted(covered),
            "heldout_only_dimensions": sorted(set().union(*(f.dimensions for f in cat)) - covered)}


_CRUCIBLE_VOCAB = {"delay", "stock", "feedback", "regime", "gate", "threshold", "interaction", "confounder", "invu"}


def crucible_relations() -> dict:
    """Relation of each of Crucible's own scm families to *this catalogue's* training set (the family split
    itself is unchanged and still enforced by `SplitGuard`): ``structural`` if one of its motifs never occurs in
    catalogue training, ``compositional`` if each does but never in this combination, ``within`` otherwise."""
    try:
        from crucible.mechanisms import load_engines
        from crucible.mechanisms.base import MECH_FAMILIES
        load_engines()
    except Exception:  # noqa: BLE001
        return {}
    train_sets = [set(f.motifs) for f in catalogue() if f.split == "train" and f.engine == "scm"]
    out = {}
    for n, f in sorted(MECH_FAMILIES.items()):
        if f.engine != "scm" or n.startswith(VERSION):
            continue
        ms = set(f.motifs) & _CRUCIBLE_VOCAB
        if not all(any(m in s for s in train_sets) for m in ms):
            rel = "structural"
        elif ms and not any(ms <= s for s in train_sets):
            rel = "compositional"
        else:
            rel = "within"
        out[n] = {"crucible_split": f.split, "motifs": sorted(ms), "relation_to_catalogue_training": rel}
    return out


# ============================================================================================== split guard
class HeldOutFamilyError(RuntimeError):
    """A validation / test / held-out family (or a validation-range seed) was about to enter a training stream."""


TRAINABLE_SPLITS = ("train", "dev")


class SplitGuard:
    """Refuses held-out experience in training streams. Knows the splits of Crucible's mechanism families
    (`MECH_FAMILIES`), Crucible's world families (`worldgen.generator.FAMILIES`), this catalogue, Crucible's goal
    held-out families, and any `extra` {family: split} declarations (task protocols add their own). Unknown names
    are refused unless ``allow_unknown``."""

    def __init__(self, extra: Optional[dict] = None, allow_unknown: bool = False, seed_limit: Optional[int] = None):
        import crucible
        crucible.load_all_modules()
        from crucible.goals.contracts import HELDOUT_GOAL_FAMILIES
        from crucible.mechanisms import load_engines
        from crucible.mechanisms.base import MECH_FAMILIES
        from crucible.research.tasks import TRAIN_SEED_MAX
        from crucible.worldgen.generator import FAMILIES
        load_engines()
        self.splits: dict[str, str] = {}
        for n, f in FAMILIES.items():
            self.splits[n] = f.split
        for n, f in MECH_FAMILIES.items():
            self.splits[n] = f.split
            self.splits[f"mech_{n}"] = f.split
            self.splits[f"mech_{n}_tools"] = f.split
        for f in catalogue():
            self.splits[f.name] = f.split
        for g in HELDOUT_GOAL_FAMILIES:
            self.splits[g] = "test"
        self.splits.update(extra or {})
        self.allow_unknown = allow_unknown
        self.seed_limit = TRAIN_SEED_MAX if seed_limit is None else int(seed_limit)
        self.checked = 0

    def split_of(self, family: str) -> Optional[str]:
        return self.splits.get(family)

    def check(self, family: str, seed: Optional[int] = None, what: str = "family") -> None:
        sp = self.splits.get(family)
        if sp is None and not self.allow_unknown:
            raise HeldOutFamilyError(f"{what} {family!r} has no declared split; refusing it in a training stream")
        if sp is not None and sp not in TRAINABLE_SPLITS:
            raise HeldOutFamilyError(f"{what} {family!r} is a {sp} family; it must never enter a training stream")
        if seed is not None and int(seed) > self.seed_limit:
            raise HeldOutFamilyError(f"{what} {family!r} seed {seed} is outside the training seed range "
                                     f"(0..{self.seed_limit}); validation/sealed seeds never train")
        self.checked += 1

    def check_task(self, task: Any) -> None:
        """A mechanism `Task`, a GoalContract-like OpenWorld task (``world = {family, seed}``), or a name."""
        if isinstance(task, str):
            return self.check(task)
        if hasattr(task, "mech_seed") and hasattr(task, "family"):
            return self.check(task.family, getattr(task, "seed", None), "mechanism family")
        world = getattr(task, "world", None)
        if isinstance(world, dict) and "family" in world:
            self.check(world["family"], None, "world family")
            gf = getattr(task, "family", None)
            if gf in self.splits:
                self.check(gf, None, "goal family")
            return None
        raise HeldOutFamilyError(f"cannot determine the family of {type(task).__name__}; refusing it")

    def guard(self, stream: Iterable[Any]) -> Iterator[Any]:
        """Wrap a training stream: every item is checked before it is yielded."""
        for item in stream:
            self.check_task(item)
            yield item


def training_stream(families_: Iterable[str], start: int = 0, guard: Optional[SplitGuard] = None,
                    validate: bool = False) -> Iterator[Any]:
    """Unvalidated training tasks cycling over `families_` (seed-major from `start`), split-guarded."""
    from crucible.research.tasks import make_task
    register_families()
    g = guard or SplitGuard()
    fams = list(families_)
    for f in fams:
        g.check(f)
    s = int(start)
    while True:
        for f in fams:
            t = make_task(f, s, validate=validate)
            g.check_task(t)
            yield t
        s += 1


def validate_catalogue(seeds: Iterable[int] = (0, 1), families_: Optional[Iterable[str]] = None,
                       ticks: int = 24) -> dict:
    """Build (unvalidated) tasks of every family and run them: lever/gauge counts, net per tick of the defaults
    and of random settings, and whether the random settings move the outcome (the mechanism responds)."""
    from crucible.research.envs import MechVecEnv
    from crucible.research.learner import EnvSpec
    from crucible.research.tasks import make_task
    register_families()
    out = {}
    for f in (families_ or [fd.name for fd in catalogue()]):
        rows = []
        for s in seeds:
            t = make_task(f, int(s), validate=False)
            eng = t.sim().engine
            nl, ng = len(eng.levers(0)), len(eng.gauges(0))
            spec = EnvSpec(max_levers=max(12, nl), max_gauges=max(16, ng))
            env = MechVecEnv([t, t], spec)
            env.reset()
            rng = np.random.default_rng(int(s))
            nets = np.zeros(2)
            for _ in range(min(ticks, t.horizon)):
                a = np.zeros((2, spec.max_levers))
                a[0] = np.asarray(env._norm_settings(0))          # defaults held
                a[1] = rng.uniform(-1, 1, spec.max_levers)
                _, _, _, info = env.step(a)
                nets += info["net_cents"]
            rows.append({"seed": int(s), "levers": nl, "gauges": ng, "horizon": int(t.horizon),
                         "default_net": float(nets[0]), "random_net": float(nets[1]),
                         "responds": bool(abs(nets[0] - nets[1]) > 1e-6)})
        out[f] = rows
    return out


# ============================================================================================== surfaces
_UNITS = (("reading", 1.0), ("milli-reading", 1000.0), ("kilo-reading", 0.001), ("centi-reading", 100.0),
          ("deci-reading", 10.0))
WORDING = {
    "lever": ("Set control {name} (allowed {lo} to {hi}, default {default}{integer}). Applies from the next tick.",
              "Adjust the setting of {name}: any value from {lo} to {hi}{integer}; it rests at {default}. Takes "
              "effect at the next tick.",
              "Change dial {name} to a value within [{lo}, {hi}]{integer} (resting value {default}); used from the "
              "next period on."),
    "goal": ("Operate the facility so that your net income (revenue minus cost) over the first {horizon} ticks is "
             "at least {goal} cents.",
             "Over {horizon} periods, make the facility earn a net of at least {goal} cents (what it is paid minus "
             "what it costs).",
             "Target: cumulative revenue minus operating cost of {goal} cents or more within {horizon} ticks."),
    "goal_open": ("Operate the facility to earn as much net income (revenue minus cost) as you can over {horizon} "
                  "ticks.",
                  "Make the facility as profitable as possible over {horizon} periods.",
                  "Maximise cumulative revenue minus operating cost within {horizon} ticks."),
    "report": ("tick {tick}: readings and money of the facility", "period {tick} report",
               "measurements after tick {tick}"),
}


@dataclass
class Surface:
    """Presentation of one episode: neutral names, observation/tool ordering, gauge units (a declared presentation
    scale with its unit label: the value shown is reading x scale) and wording. Two surfaces over the same task are
    an invariance pair: identical dynamics, different appearance."""
    seed: int
    rename: bool = True
    shuffle: bool = True
    scale_units: bool = True
    reword: bool = True

    def __post_init__(self):
        self.rng = np.random.default_rng([int(self.seed), 0x5EED])
        self.variant = int(self.rng.integers(0, 3)) if self.reword else 0

    def names(self, original: list) -> list:
        if not self.rename:
            return list(original)
        from crucible.mechanisms.base import neutral_names
        return neutral_names(self.rng, len(original), used=set(original))

    def order(self, n: int) -> list:
        return [int(i) for i in self.rng.permutation(n)] if self.shuffle else list(range(n))

    def unit(self) -> tuple:
        if not self.scale_units:
            return _UNITS[0]
        return _UNITS[int(self.rng.integers(0, len(_UNITS)))]

    def phrase(self, key: str, **kw) -> str:
        return WORDING[key][self.variant].format(**kw)


def invariance_pair(seed: int) -> tuple:
    """Two different surfaces for the same task (different names, order, units, wording)."""
    a, b = Surface(2 * int(seed) + 1), Surface(2 * int(seed) + 2)
    if b.variant == a.variant:
        b.variant = (a.variant + 1) % 3
    return a, b
