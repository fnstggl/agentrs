"""Hidden-mechanism engines: the contract shared by every engine, the facility module (`rig`), the private
evaluator and the research environments.

A *mechanism* is a hidden causal process behind a facility that participants operate through ordinary tools: they
set **levers**, read **gauges**, and are paid for a verified **outcome**. The mechanism (its causal graph, its
functional forms, its delays, feedback loops and regime changes) is never shown to a participant; it has to be
discovered by acting. The environment operator knows it exactly, which is what makes discovery measurable.

Three properties are required of every engine, and checked by `mechanisms.validity`:

1. **Purity.** `step` is a pure function of (config, state, settings, rng draws, t): no global randomness, no
   wall-clock, no dependence on dict ordering or hash seeds. The same inputs give bit-identical outputs, so a
   facility hosted in the event-sourced kernel replays exactly and a fork is a controlled counterfactual.
2. **Opacity.** Public names (`Lever.name`, `Gauge.name`, the brief) are neutral: they carry no hint of role, sign
   or structure. Everything that reveals structure lives in `truth()` and `StepResult.info`, which only the
   evaluator reads.
3. **Conservation.** Revenue is paid from the independent demand the engine models, and can never exceed what
   that demand could pay; cost is what the settings consume. The facility module moves real ledger money for both.

Engines are *independently authored*: they share this contract and nothing else (no shared helper decides a
causal form). `ENGINES` and `MECH_FAMILIES` are filled when the engine modules are imported
(`crucible.mechanisms.load_engines()`).

Structural variation is expressed by **families** (`MechFamily`): a family fixes which motifs are composed and how;
the seed draws the concrete graph, parameters and names. Families are partitioned train / val / test, and every
non-train family declares its *relation* to the training set (`transfer`):

    "base"           a training family
    "within"         same family, unseen seeds (reported from held-out seeds of a train family)
    "compositional"  motifs all seen in training, in a combination never seen
    "structural"     at least one causal motif never seen in training
    "cross_impl"     a different engine implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.canonical import digest

CONTRACT_VERSION = "mech/1"
TRANSFER_KINDS = ("base", "within", "compositional", "structural", "cross_impl")


# ====================================================================== public surface of a facility
@dataclass(frozen=True)
class Lever:
    """A quantity a seat can set. `default` is where it rests if never touched."""
    name: str
    lo: float
    hi: float
    default: float
    integer: bool = False
    doc: str = ""

    def clip(self, v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = self.default
        if not np.isfinite(x):
            x = self.default
        x = min(self.hi, max(self.lo, x))
        return float(round(x)) if self.integer else float(x)

    def public(self) -> dict:
        return {"name": self.name, "min": self.lo, "max": self.hi, "default": self.default,
                "integer": self.integer, "doc": self.doc}


@dataclass(frozen=True)
class Gauge:
    """A quantity a seat can read each tick (possibly noisy, possibly lagged; the engine decides)."""
    name: str
    doc: str = ""

    def public(self) -> dict:
        return {"name": self.name, "doc": self.doc}


@dataclass
class StepResult:
    state: dict[str, np.ndarray]        # next hidden state (fresh arrays; the input state is never mutated)
    readings: list[dict[str, float]]    # per seat: gauge name -> value shown to that seat
    revenue: list[float]                # per seat: cents earned from the engine's independent demand this tick
    cost: list[float]                   # per seat: cents of operating cost the settings incurred this tick
    score: list[float]                  # per seat: the objective quantity this tick (what goals count)
    info: dict = field(default_factory=dict)   # PRIVATE diagnostics: evaluator only


class Engine:
    """Base class. Subclasses set `engine`, implement `generate`, and the instance methods below."""
    engine: str = ""
    version: str = "1"

    def __init__(self, config: dict):
        self.config = config

    # ---- construction ------------------------------------------------------------------------------
    @classmethod
    def generate(cls, rng: np.random.Generator, params: dict) -> dict:
        """Draw a complete mechanism. Returns a JSON-serialisable config (plain dicts, lists, numbers, strings):
        it is stored in the world spec, hashed, and is all an instance needs."""
        raise NotImplementedError

    # ---- public surface ----------------------------------------------------------------------------
    @property
    def n_seats(self) -> int:
        return int(self.config.get("n_seats", 1))

    def levers(self, seat: int) -> list[Lever]:
        raise NotImplementedError

    def gauges(self, seat: int) -> list[Gauge]:
        raise NotImplementedError

    def brief(self, seat: int) -> str:
        """What the seat is paid for, in neutral words. No strategy, no structure."""
        raise NotImplementedError

    @property
    def horizon(self) -> int:
        """Suggested episode length in ticks (regime changes and delays fit inside it)."""
        return int(self.config.get("horizon", 120))

    # ---- dynamics ----------------------------------------------------------------------------------
    def init_state(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def step(self, state: dict[str, np.ndarray], settings: list[dict[str, float]], rng: np.random.Generator,
             t: int) -> StepResult:
        """Advance one tick. `settings[seat]` is a complete, range-clipped mapping lever name -> value."""
        raise NotImplementedError

    # ---- private (evaluator only) ------------------------------------------------------------------
    def truth(self) -> dict:
        """Canonical description of the hidden mechanism::

            {"engine": str, "family": str | None,
             "nodes": [{"id": str, "role": "lever|latent|gauge|outcome|exogenous", "seat": int | None,
                        "public": str | None}],
             "edges": [{"src": id, "dst": id, "sign": +1 | -1 | 0, "lag": int, "kind": str}],
             "regimes": [{"at": tick, "what": str}], "motifs": [str], "notes": str}

        `sign` 0 means non-monotone or gated. `public` is the name a participant sees (None for latent nodes)."""
        raise NotImplementedError

    # ---- helpers -----------------------------------------------------------------------------------
    def defaults(self, seat: int) -> dict[str, float]:
        return {lv.name: lv.default for lv in self.levers(seat)}

    def clip_settings(self, seat: int, raw: dict | None, base: dict[str, float] | None = None) -> dict[str, float]:
        """Complete, validated settings for a seat: `base` (or defaults) overlaid with the known keys of `raw`."""
        out = dict(base) if base is not None else self.defaults(seat)
        raw = raw or {}
        for lv in self.levers(seat):
            if lv.name in raw:
                out[lv.name] = lv.clip(raw[lv.name])
            else:
                out[lv.name] = lv.clip(out.get(lv.name, lv.default))
        return out

    def public_spec(self, seat: int) -> dict:
        return {"levers": [lv.public() for lv in self.levers(seat)],
                "gauges": [g.public() for g in self.gauges(seat)],
                "brief": self.brief(seat), "seats": self.n_seats, "horizon_ticks": self.horizon}


# ====================================================================== registries
ENGINES: dict[str, type[Engine]] = {}


def register_engine(cls: type[Engine]) -> type[Engine]:
    if not cls.engine:
        raise ValueError("engine class without a name")
    ENGINES[cls.engine] = cls
    return cls


@dataclass
class MechFamily:
    name: str
    engine: str
    split: str                 # train | val | test
    transfer: str              # see TRANSFER_KINDS
    build: Callable[[np.random.Generator, dict], dict]   # (rng, overrides) -> params for Engine.generate
    doc: str
    motifs: tuple[str, ...] = ()
    seats: int = 1


MECH_FAMILIES: dict[str, MechFamily] = {}


def mech_family(name: str, engine: str, split: str, transfer: str, doc: str, motifs: tuple[str, ...] = (),
                seats: int = 1):
    if split not in ("train", "val", "test"):
        raise ValueError(f"bad split {split!r}")
    if transfer not in TRANSFER_KINDS:
        raise ValueError(f"bad transfer kind {transfer!r}")
    if split == "train" and transfer != "base":
        raise ValueError("training families have transfer='base'")

    def deco(fn):
        MECH_FAMILIES[name] = MechFamily(name, engine, split, transfer, fn, doc, tuple(motifs), seats)
        return fn
    return deco


def mech_families(split: str | None = None, engine: str | None = None, transfer: str | None = None) -> list[str]:
    return sorted(n for n, f in MECH_FAMILIES.items()
                  if (split is None or f.split == split) and (engine is None or f.engine == engine)
                  and (transfer is None or f.transfer == transfer))


def make_mechanism(family: str, seed: int, overrides: dict | None = None) -> dict:
    """A concrete hidden mechanism: {"engine", "family", "seed", "config"} - JSON-serialisable and deterministic in
    (family, seed, overrides)."""
    fam = MECH_FAMILIES[family]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(["mech", family])[:8], 16)]))
    params = fam.build(rng, dict(overrides or {}))
    cfg = ENGINES[fam.engine].generate(rng, params)
    cfg.setdefault("n_seats", fam.seats)
    cfg["family"] = family
    return {"engine": fam.engine, "family": family, "seed": int(seed), "config": cfg}


def build_engine(mech: dict) -> Engine:
    return ENGINES[mech["engine"]](mech["config"])


# ====================================================================== randomness shared by every host
def tick_rng(world_seed: int, rig_id: str, t: int | str) -> np.random.Generator:
    """The generator for tick `t` of facility `rig_id` in a world with seed `world_seed`. The kernel-hosted
    facility and the direct simulator both use exactly this derivation, which is why their trajectories agree
    bit for bit (it equals `World.rng_for("rig", rig_id, t)`)."""
    h = digest([int(world_seed), "rig", str(rig_id), str(t)], size=16)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(h, 16))))


def neutral_names(rng: np.random.Generator, n: int, used: set[str] | None = None) -> list[str]:
    """`n` distinct opaque identifiers (a consonant-vowel syllable and two digits, e.g. 'zu47'). The same pool
    serves levers and gauges, so a name says nothing about what it is."""
    cons, vow = "bdfgjklmnprstvz", "aeiou"
    used = set(used or ())
    out: list[str] = []
    while len(out) < n:
        s = f"{cons[int(rng.integers(len(cons)))]}{vow[int(rng.integers(len(vow)))]}{int(rng.integers(10, 100))}"
        if s not in used:
            used.add(s)
            out.append(s)
    return out


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in state.items()}
