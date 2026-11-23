"""Tasks on hidden mechanisms: one (mechanism family, seed) made into a fully specified, reproducible problem.

A `Task` fixes the mechanism draw, the world seed, the money rules (the facility's demand income per tick and the
operator's starting balance) and a **public objective**: total net income to reach within the horizon. The same
task runs in a kernel world (`Task.world_config()` -> `crucible.worldgen.generate`, operated with `rig.*` tools)
and in the kernel-free simulator (`Task.sim()`), with identical readings and money, tick for tick.

Private to the operator, and never part of what a learner is given: the reference policy's net (found by search on
the true mechanism), the net of leaving the defaults alone, and the net of uniformly random settings. They exist to
*normalise* results (`Task.score`: 0 = left the defaults alone, 1 = as good as the reference) and to set the
public objective half way between the two.

Validated tasks come from `mechanisms.validity.sample_valid`: draws that fail a validity check are redrawn
deterministically and the rejected draws are recorded in the task. Validation costs seconds of CPU per task, so
tasks are cached on disk (`CRUCIBLE_CACHE`, default `~/.cache/crucible`); the cache is a pure function of
(family, seed, versions) and can be deleted at any time. Unvalidated tasks (`validate=False`) have no reference
and are meant for training streams, where a learner needs experience and not a normalised score.

Seeds partition experience from evaluation:

    training streams       seeds 0 ... 999_999          `train_stream`
    validation sets        seeds 1_000_000 ...          `eval_tasks(split="val")`
    final (sealed) sets    seeds derived from a salt kept outside the repository (`research.sealed`)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Iterator, Sequence

from ..core.canonical import digest
from ..mechanisms.base import CONTRACT_VERSION, MECH_FAMILIES, build_engine, make_mechanism

TASK_VERSION = "task/2"
ENDOWMENT_TICKS = 20            # the operator starts with this many ticks' worth of the facility's demand income
GOAL_FRACTION = 0.5             # the public objective sits this far from the default net toward the reference net
VAL_SEED0 = 1_000_000
TRAIN_SEED_MAX = 999_999


@dataclass(frozen=True)
class Task:
    family: str
    seed: int                       # the seed asked for
    mech_seed: int                  # the draw that was accepted (differs from `seed` only after redraws)
    world_seed: int
    income: int                     # demand income of the facility, cents per tick
    endowment: int                  # operator's starting balance, cents
    horizon: int                    # ticks
    n_seats: int = 1
    validated: bool = False
    goal_net_cents: int | None = None       # PUBLIC objective (None for unvalidated tasks)
    # ---- private (operator side) ----
    default_net: float | None = None        # cents per tick, settings left at their defaults
    reference_net: float | None = None      # cents per tick, the reference policy found by search
    random_net: float | None = None         # cents per tick, uniformly random settings
    solvable: str = "unknown"
    rejected: tuple = ()                    # draws that failed validity before this one: ((mech_seed, reasons), ...)
    engine: str = ""
    transfer: str = ""
    split: str = ""

    @property
    def id(self) -> str:
        return f"{self.family}:{self.seed}"

    # ---- the problem itself ------------------------------------------------------------------------
    def mechanism(self) -> dict:
        return _mech(self.family, self.mech_seed)

    def world_config(self, tools: bool = False, holders: Sequence[str | None] = ("agent",)) -> tuple[dict, int]:
        """`(config, seed)` for `crucible.worldgen.generate`: the kernel world hosting this task."""
        return ({"family": f"mech_{self.family}" + ("_tools" if tools else ""),
                 "overrides": {"mech_seed": self.mech_seed, "income_cents_per_tick": self.income,
                               "holders": list(holders)}}, self.world_seed)

    def sim(self, money: bool = True, held: Sequence[bool] | None = None, funds: Sequence[int | None] | None = None):
        """The kernel-free twin: seat 0 held by an operator with the task's endowment, other seats empty."""
        from ..mechanisms.sim import MechSim
        if held is None:
            held = [True] + [False] * (self.n_seats - 1)
        if funds is None:
            funds = [self.endowment if h else None for h in held]
        if not money:
            return MechSim(self.mechanism(), self.world_seed, "r1")
        return MechSim(self.mechanism(), self.world_seed, "r1", income_cents_per_tick=self.income, funds=list(funds),
                       held=list(held))

    def public(self) -> dict:
        """What a participant may be told about the task: the objective and the money rules. Nothing else."""
        return {"task": self.id, "horizon_ticks": self.horizon, "starting_balance_cents": self.endowment,
                "objective": None if self.goal_net_cents is None else
                f"Operate the facility so that your net income (revenue minus cost) over the first {self.horizon} "
                f"ticks is at least {self.goal_net_cents} cents.", "goal_net_cents": self.goal_net_cents}

    # ---- private measures --------------------------------------------------------------------------
    def score(self, net_per_tick: float) -> float | None:
        """Normalised result: 0 = the net of leaving the defaults alone, 1 = the reference policy's net."""
        if self.reference_net is None or self.default_net is None:
            return None
        gap = self.reference_net - self.default_net
        return float((net_per_tick - self.default_net) / gap) if gap > 0 else None

    def success(self, total_net_cents: float) -> bool | None:
        return None if self.goal_net_cents is None else bool(total_net_cents >= self.goal_net_cents)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejected"] = [list(x) for x in self.rejected]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["rejected"] = tuple((int(a), tuple(b)) for a, b in d.get("rejected", ()))
        return cls(**d)


_MECHS: dict[tuple[str, int], dict] = {}


def _mech(family: str, mech_seed: int) -> dict:
    key = (family, int(mech_seed))
    if key not in _MECHS:
        if len(_MECHS) > 4096:
            _MECHS.clear()
        _ensure()
        _MECHS[key] = make_mechanism(family, int(mech_seed))
    return _MECHS[key]


def _ensure() -> None:
    from ..mechanisms.sim import ensure_engines
    ensure_engines()


def world_seed_of(family: str, seed: int) -> int:
    """Every task gets its own world seed: per-tick noise is a function of (world seed, facility, tick), so two
    tasks sharing a world seed would share their noise."""
    return int(digest(["task-world", family, int(seed)])[:8], 16)


def _income(mech: dict) -> int:
    from ..mechanisms.sim import demand_income
    return demand_income(mech)


def make_task(family: str, seed: int, validate: bool = True, max_tries: int = 8) -> Task:
    """Build one task. With `validate`, the mechanism is drawn with `sample_valid` (seconds of CPU) and the task
    carries the private reference; raises ValueError if no draw passes."""
    _ensure()
    fam = MECH_FAMILIES[family]
    meta = {"engine": fam.engine, "transfer": fam.transfer, "split": fam.split}
    if not validate:
        mech = _mech(family, seed)
        eng = build_engine(mech)
        inc = _income(mech)
        return Task(family, int(seed), int(seed), world_seed_of(family, seed), inc, ENDOWMENT_TICKS * inc,
                    int(eng.horizon), int(eng.n_seats), False, **meta)
    from ..mechanisms.validity import sample_valid
    got = sample_valid(family, int(seed), max_tries)
    if got["mech"] is None:
        raise ValueError(f"no valid {family} mechanism for seed {seed} in {max_tries} draws: {got['rejected']}")
    mech, rep = got["mech"], got["report"]
    _MECHS[(family, int(got["seed"]))] = mech
    eng = build_engine(mech)
    econ = rep["economics"][0]
    inc = _income(mech)
    h = int(eng.horizon)
    d, r = float(econ["default_net"]), float(econ["reference_net"])
    return Task(family, int(seed), int(got["seed"]), world_seed_of(family, seed), inc, ENDOWMENT_TICKS * inc, h,
                int(eng.n_seats), True, int(round(h * (d + GOAL_FRACTION * (r - d)))), d, r, float(econ["random_net"]),
                rep["solvable"], tuple((int(x["seed"]), tuple(x["reasons"])) for x in got["rejected"]), **meta)


# ---------------------------------------------------------------------------- cache
def cache_dir() -> str:
    return os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")


def _cache_path(family: str, seed: int) -> str:
    v = digest([TASK_VERSION, CONTRACT_VERSION, getattr(MECH_FAMILIES.get(family), "engine", "")], 6)
    return os.path.join(cache_dir(), "tasks", v, family, f"{int(seed)}.json")


def _load(family: str, seed: int) -> Task | None:
    try:
        with open(_cache_path(family, seed)) as fh:
            return Task.from_dict(json.load(fh))
    except (OSError, ValueError, TypeError):
        return None


def _store(t: Task) -> None:
    path = _cache_path(t.family, t.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(t.to_dict(), fh)
    os.replace(tmp, path)


def _build_one(args) -> dict:
    family, seed, max_tries = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        t = make_task(family, seed, True, max_tries)
        _store(t)
        return {"ok": True, "task": t.to_dict()}
    except Exception as e:                                       # noqa: BLE001 - reported, never swallowed
        return {"ok": False, "family": family, "seed": seed, "error": f"{type(e).__name__}: {e}"}


def build_tasks(pairs: Sequence[tuple[str, int]], workers: int = 1, max_tries: int = 8, use_cache: bool = True,
                skip_failures: bool = False) -> list[Task]:
    """Validated tasks for (family, seed) pairs, from the cache where possible, the rest built (in `workers`
    processes). A pair for which no draw passes validity raises, unless `skip_failures` (then it is left out and
    listed in `build_tasks.failures`)."""
    _ensure()
    out: dict[tuple[str, int], Task] = {}
    todo = []
    for fam, seed in pairs:
        t = _load(fam, seed) if use_cache else None
        if t is not None and t.validated:
            out[(fam, int(seed))] = t
        else:
            todo.append((fam, int(seed), max_tries))
    failures = []
    if todo:
        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
                res = pool.map(_build_one, todo, chunksize=1)
        else:
            res = [_build_one(a) for a in todo]
        for r in res:
            if r["ok"]:
                t = Task.from_dict(r["task"])
                out[(t.family, t.seed)] = t
            else:
                failures.append(r)
    build_tasks.failures = failures                              # type: ignore[attr-defined]
    if failures and not skip_failures:
        raise ValueError(f"{len(failures)} task(s) could not be built: {failures[:3]}")
    return [out[(f, int(s))] for f, s in pairs if (f, int(s)) in out]


def eval_tasks(families: Sequence[str], n: int, split: str = "val", offset: int = 0, **kw) -> list[Task]:
    """`n` validated tasks per family from the validation seed range (never used by training streams)."""
    if split != "val":
        raise ValueError("only the validation range is public; final sets come from research.sealed")
    return build_tasks([(f, VAL_SEED0 + offset + i) for f in families for i in range(n)], **kw)


def train_stream(families: Sequence[str], start: int = 0, shuffle_seed: int | None = None) -> Iterator[Task]:
    """An endless stream of unvalidated training tasks cycling over `families`, seeds `start`, `start+1`, ...
    (each seed is used once per family). Stays inside the training seed range."""
    import numpy as np
    fams = list(families)
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    s = int(start)
    while True:
        if s > TRAIN_SEED_MAX:
            raise RuntimeError("training seed range exhausted")
        order = list(rng.permutation(len(fams))) if rng is not None else range(len(fams))
        for i in order:
            yield make_task(fams[i], s, validate=False)
        s += 1


def families_by(split: str | None = None, engine: str | None = None, transfer: str | None = None,
                seats: int | None = 1) -> list[str]:
    """Mechanism family names by partition (`seats=None`: any number of seats)."""
    _ensure()
    return sorted(n for n, f in MECH_FAMILIES.items()
                  if (split is None or f.split == split) and (engine is None or f.engine == engine)
                  and (transfer is None or f.transfer == transfer) and (seats is None or f.seats == seats))
