"""The versioned cross-tier training-distribution registry (MASTER_SPEC Part 14.1; SP-14.1, SP-14.1b, T.7).

One registry describes everything a learner can be trained or evaluated on, in nine kinds (Part 14.1):

    engines                  transition implementations (rapid engine kinds, 4.0 mechanism engines, the kernel)
    mechanism_families       4.0 hidden-mechanism families and the v5 causal primitives
    world_families           Tier 1 rapid families (and generated structures), 4.0 mechanism tasks, Tier 3
                             civilisation profiles - the things an episode is drawn from
    domain_configurations    Tier 2 domain configurations (`crucible.domains.base`, D-17)
    objectives               objective types: the families' assigned objectives and the goal types of the
                             independent goal generator (`crucible.rapid.goals`)
    participant_policies     distributions of other participants (rapid opponents and populations, generated
                             populations, civilisation population profiles)
    action_interfaces        action kinds (rapid action kinds, the 4.0 lever interface, kernel tools)
    difficulty_levels        named levels that set a family's difficulty axes
    partitions               train / validation / evaluation / sealed

Partitions. Every episode is an item (world family, seed, difficulty level). A family is either a *training* family
(rapid split ``train``, 4.0 split ``train``, Tier 2 ``train`` configurations, Tier 3 profiles not held out) or an
*evaluation* family (rapid ``val``/``test`` = v5-eval-1, 4.0 ``val``, Tier 2 ``eval_compositional`` /
``eval_structural``, the held-out Tier 3 profiles ``EVAL_CIV_PROFILES``); 4.0 ``test`` families are *sealed*.
Seeds then partition each family: training families give ``train`` items on seeds 0..999_999 and ``validation``
items on seeds 1_000_000..1_999_999_999; evaluation families give ``evaluation`` items below 2_000_000_000; every
family gives ``sealed`` items only on seeds the evaluator derives from a secret salt (>= 2_000_000_000,
`crucible.research.v5.evaluator`). Only ``train`` items may be handed to a learner for training
(`check_trainable`); a training family whose measured structural fingerprint equals an evaluation family's
(distance 0) is refused too (a renamed copy of an evaluation world is not a new training world).

Versioning. The registry is built from the code of this checkout, frozen (read-only mappings and tuples: entries
cannot be relabelled in place), and identified by the SHA-256 of its canonical JSON (`digest`, `version`). Adding
generated worlds makes a child registry (`extend`) whose `parent` is the old digest. A sampler records the version
it was built from and refuses to resume on another.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Iterable, Mapping

SCHEMA = "crucible.v5.distribution/1"
KINDS = ("engines", "mechanism_families", "world_families", "domain_configurations", "objectives",
         "participant_policies", "action_interfaces", "difficulty_levels", "partitions")
PARTITIONS = ("train", "validation", "evaluation", "sealed")
TRAIN_SEEDS = (0, 1_000_000)
VALIDATION_SEEDS = (1_000_000, 2_000_000_000)
EVALUATION_SEEDS = (0, 2_000_000_000)
SEALED_SEED0 = 2_000_000_000
# Tier 3 profiles held out of training (the "new domain" of Level 5 among civilisations). Chosen once, here; the
# choice is part of the registry digest, so changing it makes a new distribution version.
EVAL_CIV_PROFILES = ("fragmented_social", "infrastructure_constrained")
# difficulty levels: multipliers / increments applied to the axes a family declares (missing axes are ignored)
DIFFICULTY_LEVELS = {
    "0": {"doc": "the family's default difficulty", "mul": {}, "add": {}},
    "1": {"doc": "noisier readings, two distractor variables, costlier experiments",
          "mul": {"observation_noise": 2.0, "experiment_cost": 1.5}, "add": {"n_variables": 2}},
    "2": {"doc": "much noisier readings, four distractors, stronger opponents, half the resources, dearer experiments",
          "mul": {"observation_noise": 3.0, "experiment_cost": 2.0, "opponent_capability": 1.5, "resources": 0.5},
          "add": {"n_variables": 4}},
}
DIVERSITY_DIMENSIONS = ("parameters", "causal_structure", "composition", "objective", "participant_policies",
                        "action_space")


class PartitionError(PermissionError):
    """An evaluation, validation or sealed item was requested where only training items may go."""


# ====================================================================== freezing
def freeze(x: Any) -> Any:
    if isinstance(x, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple, set, frozenset)):
        items = [freeze(v) for v in x]
        return tuple(sorted(items, key=repr)) if isinstance(x, (set, frozenset)) else tuple(items)
    return x


def thaw(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {k: thaw(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [thaw(v) for v in x]
    return x


def canonical(x: Any) -> str:
    return json.dumps(thaw(x), sort_keys=True, separators=(",", ":"), default=str)


def sha(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


# ====================================================================== sources
def _rapid_entries() -> list[dict]:
    from ...rapid import families as RF
    out = []
    for n, f in sorted(RF.FAMILIES.items()):
        if n.startswith("gen_"):                    # generated structures enter through `extend`
            continue
        e = RF.REGISTRY.get(n) if hasattr(RF.REGISTRY, "get") else None
        kinds = sorted(e["action_kinds"]) if e else []
        out.append({"id": f"t1:rapid:{n}", "tier": 1, "source": "crucible.rapid", "name": n,
                    "make": {"tier": 1, "name": f"rapid:{n}"}, "engine": f.engine, "category": f.category,
                    "family_partition": "train" if f.split == "train" else "evaluation",
                    "source_split": f.split, "objective": f.objective, "axes": dict(f.axes),
                    "participants": sorted(set(f.participants)), "action_kinds": kinds,
                    "lever_view": bool(f.lever_view), "version": f.version})
    return out


def _mech_entries() -> list[dict]:
    from ...mechanisms.base import MECH_FAMILIES
    from ...mechanisms.sim import ensure_engines
    ensure_engines()
    out = []
    part = {"train": "train", "val": "evaluation", "dev": "evaluation", "test": "sealed"}
    for n, f in sorted(MECH_FAMILIES.items()):
        out.append({"id": f"t1:mech:{n}", "tier": 1, "source": "crucible.mechanisms", "name": n,
                    "make": {"tier": 1, "name": f"mech:{n}", "kw": {"validate": False}}, "engine": f.engine,
                    "category": "4.0", "family_partition": part.get(f.split, "evaluation"), "source_split": f.split,
                    "objective": "maximize_net", "axes": {}, "participants": [f"seats:{f.seats}"],
                    "action_kinds": ["mech_levers"], "lever_view": True, "version": "mech/1"})
    return out


UNAVAILABLE: dict[int, str] = {}       # tier -> why its package could not be loaded in this process (reported)


def _domain_entries() -> list[dict]:
    from ...experience.registry import domains
    out = []
    try:
        doms = domains()
    except Exception as e:  # noqa: BLE001 - a domain package that fails to register: Tier 2 is reported unavailable
        UNAVAILABLE[2] = f"crucible.domains failed to load: {type(e).__name__}: {str(e)[:300]}"
        return []
    UNAVAILABLE.pop(2, None)
    for dn, dom in doms.items():
        for c in dom.configs:
            out.append({"id": f"t2:{dn}:{c.name}", "tier": 2, "source": "crucible.domains", "name": f"{dn}:{c.name}",
                        "make": {"tier": 2, "name": f"{dn}:{c.name}"}, "engine": "kernel", "category": dom.spec_part,
                        "family_partition": "train" if c.split == "train" else "evaluation", "source_split": c.split,
                        "objective": f"domain:{dn}", "axes": {}, "participants": ["population"],
                        "action_kinds": ["kernel_tools"], "structure": dict(c.structure), "version": "domain/1"})
    return out


def _civ_entries() -> list[dict]:
    try:
        import crucible.civ as civ
    except Exception:  # noqa: BLE001 - Tier 3 absent: no entries (reported by `tiers`)
        return []
    if not callable(getattr(civ, "generate", None)):
        return []
    pops = list(getattr(civ, "POPULATION_PROFILES", ("development",)))
    out = []
    for n in sorted(getattr(civ, "PROFILES", {}) or {}):
        out.append({"id": f"t3:{n}", "tier": 3, "source": "crucible.civ", "name": n,
                    "make": {"tier": 3, "name": n, "kw": {"population": "development"}}, "engine": "kernel",
                    "category": "10.1", "family_partition": "evaluation" if n in EVAL_CIV_PROFILES else "train",
                    "source_split": "heldout" if n in EVAL_CIV_PROFILES else "train", "objective": "civ:open",
                    "axes": {}, "participants": [f"population:{p}" for p in pops], "action_kinds": ["kernel_tools"],
                    "version": "civ/1"})
    return out


def _kind_tables(worlds: list[dict], domain_cfgs: list[dict]) -> dict[str, list[dict]]:
    from ...rapid import goals as G
    from ...rapid.actions import KINDS as RAPID_ACTIONS
    from ...rapid.engines import ENGINE_KINDS
    from ...rapid import primitives as PR
    from ...mechanisms.base import ENGINES, MECH_FAMILIES
    from . import compose as C
    engines = [{"id": f"engine:rapid:{k}", "name": k, "source": "crucible.rapid", "tiers": [1]}
               for k in sorted(ENGINE_KINDS)]
    engines += [{"id": f"engine:mech:{k}", "name": k, "source": "crucible.mechanisms", "tiers": [1]}
                for k in sorted(ENGINES)]
    if any(w["tier"] in (2, 3) for w in worlds + domain_cfgs):
        engines.append({"id": "engine:kernel", "name": "kernel", "source": "crucible.core.kernel", "tiers": [2, 3]})
    mechs = [{"id": f"mech:{n}", "name": n, "source": "crucible.mechanisms", "engine": f.engine, "split": f.split}
             for n, f in sorted(MECH_FAMILIES.items())]
    mechs += [{"id": f"primitive:{n}", "name": n, "source": "crucible.rapid.primitives"}
              for n in sorted(PR.REGISTRY)]
    objs = sorted({w["objective"] for w in worlds + domain_cfgs})
    objectives = [{"id": f"objective:assigned:{o}", "name": o, "generator": "family"} for o in objs]
    objectives += [{"id": f"objective:generated:{t}", "name": t, "generator": "crucible.rapid.goals.generate"}
                   for t in G.GOAL_TYPES]
    parts = sorted({p for w in worlds + domain_cfgs for p in w["participants"]})
    policies = [{"id": f"participants:{p}", "name": p, "source": "world families"} for p in parts]
    policies += [{"id": f"participants:generated:{p}", "name": p, "source": "crucible.research.v5.compose"}
                 for p in C.POPULATIONS]
    acts = sorted(set(RAPID_ACTIONS) | {k for w in worlds + domain_cfgs for k in w["action_kinds"]})
    actions = [{"id": f"action:{a}", "name": a} for a in acts]
    levels = [{"id": f"difficulty:{k}", "name": k, **v} for k, v in sorted(DIFFICULTY_LEVELS.items())]
    partitions = [
        {"id": "partition:train", "name": "train", "seeds": list(TRAIN_SEEDS), "families": "training",
         "use": "training experience"},
        {"id": "partition:validation", "name": "validation", "seeds": list(VALIDATION_SEEDS), "families": "training",
         "use": "development evaluation, curriculum measurement (never training)"},
        {"id": "partition:evaluation", "name": "evaluation", "seeds": list(EVALUATION_SEEDS),
         "families": "evaluation", "use": "confirmation runs through the evaluator process only"},
        {"id": "partition:sealed", "name": "sealed", "seeds": [SEALED_SEED0, None], "families": "all",
         "use": "final confirmation; seeds derived from a salt held only by the evaluator process; access budget"},
    ]
    return {"engines": engines, "mechanism_families": mechs, "world_families": worlds,
            "domain_configurations": domain_cfgs, "objectives": objectives, "participant_policies": policies,
            "action_interfaces": actions, "difficulty_levels": levels, "partitions": partitions}


# ====================================================================== the registry
class Registry:
    """A frozen, versioned training distribution. Build with `build_registry()`."""

    def __init__(self, tables: Mapping[str, Iterable[Mapping]], parent: str | None = None,
                 generated: Iterable[Mapping] = ()):
        t = {k: [thaw(e) for e in tables.get(k, ())] for k in KINDS}
        gen = [thaw(g) for g in generated]
        t["world_families"] = t["world_families"] + gen
        body = {"schema": SCHEMA, "parent": parent, "tables": t}
        self.digest = sha(body)
        self.version = f"{SCHEMA}+{self.digest[:12]}"
        self.parent = parent
        self.tables = freeze(t)
        self._by_id = MappingProxyType({e["id"]: e for k in KINDS for e in self.tables[k]})
        self._fp_guard: dict[str, dict] = {}
        self._unavailable = dict(UNAVAILABLE)

    # ---------------------------------------------------------------- reading
    @property
    def entries(self) -> Mapping[str, Mapping]:
        return self._by_id

    def __getitem__(self, entry_id: str) -> Mapping:
        return self._by_id[entry_id]

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._by_id

    def kind_counts(self) -> dict[str, int]:
        return {k: len(self.tables[k]) for k in KINDS}

    def nonempty_kinds(self) -> list[str]:
        return [k for k in KINDS if self.tables[k]]

    def worlds(self, tier: int | None = None, family_partition: str | None = None,
               source: str | None = None) -> list[Mapping]:
        rows = list(self.tables["world_families"]) + list(self.tables["domain_configurations"])
        return [w for w in rows if (tier is None or w["tier"] == tier)
                and (family_partition is None or w["family_partition"] == family_partition)
                and (source is None or w["source"] == source)]

    def tiers(self) -> dict[int, int]:
        """Training world families per tier (a tier with none cannot be trained on)."""
        return {t: len(self.worlds(tier=t, family_partition="train")) for t in (1, 2, 3)}

    @property
    def unavailable(self) -> dict[int, str]:
        """Tiers whose package failed to load when the registry was built, with the reason."""
        return dict(self._unavailable)

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "version": self.version, "digest": self.digest, "parent": self.parent,
                "tables": thaw(self.tables)}

    # ---------------------------------------------------------------- partitions
    def partition_of(self, entry_id: str, seed: int) -> str:
        """The partition of item (entry, seed); raises KeyError for an unknown entry."""
        e = self._by_id[entry_id]
        seed = int(seed)
        if seed < 0:
            raise ValueError("seeds are non-negative")
        if seed >= SEALED_SEED0 or e["family_partition"] == "sealed":
            return "sealed"
        if e["family_partition"] == "evaluation":
            return "evaluation"
        return "train" if seed < TRAIN_SEEDS[1] else "validation"

    def check_trainable(self, entry_id: str, seed: int) -> None:
        """Raise PartitionError unless (entry, seed) is a training item whose structure is not an evaluation
        family's."""
        if entry_id not in self._by_id:
            raise PartitionError(f"{entry_id!r} is not in distribution {self.version}")
        p = self.partition_of(entry_id, seed)
        if p != "train":
            raise PartitionError(f"{entry_id} seed {seed} is in the {p} partition; only train items may be used "
                                 "for training")
        g = self.structure_guard(entry_id)
        if not g["ok"]:
            raise PartitionError(f"{entry_id} has the structure of evaluation family {g['nearest']} "
                                 f"(fingerprint distance {g['distance']})")

    def structure_guard(self, entry_id: str) -> dict:
        """Whether a training family is structurally distinct (measured fingerprint distance > 0) from every
        evaluation and sealed rapid family. Families without a measurable rapid fingerprint (4.0 mechanism tasks,
        kernel worlds) are guarded by identity and seed range only (`ok` true, `measured` false)."""
        if entry_id in self._fp_guard:
            return self._fp_guard[entry_id]
        e = self._by_id[entry_id]
        res = {"ok": True, "measured": False, "nearest": None, "distance": None}
        if e["tier"] == 1 and e["source"] in ("crucible.rapid", "generated"):
            from ...rapid import fingerprints as FP
            fp = entry_fingerprint(e)
            best = (None, float("inf"))
            for ev in self.worlds(tier=1, source="crucible.rapid"):
                if ev["family_partition"] == "train" or ev["id"] == entry_id:
                    continue
                d = FP.distance(fp, entry_fingerprint(ev))
                if d < best[1]:
                    best = (ev["id"], d)
            res = {"ok": not (best[1] <= 0.0), "measured": True, "nearest": best[0],
                   "distance": None if best[0] is None else float(best[1])}
        self._fp_guard[entry_id] = res
        return res

    # ---------------------------------------------------------------- extension and audits
    def extend(self, generated: Iterable[Mapping]) -> "Registry":
        """A child registry with generated world entries (`generated_entry`) added to the training families.
        Entries whose id is already present are refused."""
        gen = [thaw(g) for g in generated]
        for g in gen:
            if g["id"] in self._by_id:
                raise ValueError(f"{g['id']} is already in {self.version}")
            if g.get("family_partition") != "train":
                raise PartitionError("generated worlds join the training partition only")
        base = {k: thaw(self.tables[k]) for k in KINDS}
        return Registry(base, parent=self.digest, generated=gen)

    def audit(self, items: Iterable[Mapping]) -> dict:
        """Contamination audit of a training log: every item {entry, seed} must be a training item. Returns
        {checked, violations: [...]} (an unknown entry is a violation)."""
        bad, n = [], 0
        for it in items:
            n += 1
            eid, seed = it.get("entry"), it.get("seed")
            try:
                self.check_trainable(eid, int(seed))
            except (PartitionError, KeyError, TypeError, ValueError) as e:
                bad.append({"entry": eid, "seed": seed, "why": str(e)[:200]})
        return {"checked": n, "violations": bad}


def generated_entry(structure: Mapping, *, proposer: str | None = None) -> dict:
    """A registry entry for a generated Tier 1 structure (`crucible.research.v5.compose`)."""
    from . import compose as C
    spec = C.family_spec(dict(structure), proposer=proposer)
    s = spec["structure"]
    return {"id": f"t1:gen:{spec['name']}", "tier": 1, "source": "generated", "name": spec["name"],
            "make": {"tier": 1, "name": f"gen:{spec['name']}", "structure": s}, "engine": "primitive_graph",
            "category": spec["category"], "family_partition": "train", "source_split": "generated",
            "objective": "maximize_net", "axes": dict(spec["axes"]), "participants": sorted(set(spec["participants"])),
            "action_kinds": sorted({"levers"} | ({"experiment"} if "pilot_experiments" in s["drivers"] else set()) |
                                   ({"query"} if "on_demand" in s["channels"] else set())),
            "lever_view": True, "version": C.COMPOSE_VERSION, "proposer": proposer, "structure": s}


def entry_fingerprint(e: Mapping) -> dict:
    """Measured structural fingerprint of a Tier 1 entry (rapid family or generated structure)."""
    from ...rapid import fingerprints as FP
    if e["source"] == "crucible.rapid":
        return FP.family_fingerprint(e["name"])
    if e["source"] == "generated":
        return _generated_fingerprint(canonical(e["structure"]))
    raise ValueError(f"no rapid fingerprint for {e['id']}")


@lru_cache(maxsize=512)
def _generated_fingerprint(structure_json: str) -> dict:
    from ...rapid import fingerprints as FP
    from . import compose as C
    s = json.loads(structure_json)
    with C.registered(s) as names:
        return FP.family_fingerprint(names[0])


def difficulty_for(entry: Mapping, level: str | int) -> dict:
    """The difficulty axes of `entry` at a named level (only axes the family declares)."""
    lv = DIFFICULTY_LEVELS[str(level)]
    out = {}
    for ax, default in dict(entry.get("axes") or {}).items():
        if not isinstance(default, (int, float)) or isinstance(default, bool):
            continue
        v = default
        if ax in lv["mul"]:
            v = default * lv["mul"][ax] if default else (lv["mul"][ax] * 0.1 if ax == "observation_noise" else 0)
        if ax in lv["add"]:
            v = v + lv["add"][ax]
        if v != default:
            out[ax] = type(default)(v) if isinstance(default, int) else float(v)
    return out


@lru_cache(maxsize=1)
def build_registry() -> Registry:
    """The registry of this checkout (built once per process from the code; frozen)."""
    worlds = _rapid_entries() + _mech_entries() + _civ_entries()
    doms = _domain_entries()
    return Registry(_kind_tables(worlds, doms))


def source_digest() -> str:
    """Digest of the code the registry is built from (rapid, mechanisms, domains, civ and this package)."""
    import os
    h = hashlib.sha256()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for pkg in ("rapid", "mechanisms", "domains", "civ", os.path.join("research", "v5")):
        base = os.path.join(root, pkg)
        for dp, _, files in sorted(os.walk(base)):
            for f in sorted(files):
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    h.update(os.path.relpath(p, root).encode())
                    h.update(open(p, "rb").read())
    return h.hexdigest()
