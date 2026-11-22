"""Kernel-free simulation of a hidden mechanism, bit-identical to the kernel-hosted facility (`rig`).

Everything that decides what a tick *is* lives here and is used by both hosts:

* `TickRng`      the generator of tick t, equal to `base.tick_rng(world_seed, rig_id, t)` but ~4x cheaper;
* `settle_tick`  one tick of a facility: run the engine at each seat's settings, fall back to defaults for a holder
                 who cannot pay for its settings, idle a holder who cannot pay even that, pay revenue out of the
                 demand budget (pro rata when it does not cover what was earned);
* `MechSim`      one facility without a kernel (what batched learning, the evaluator and validity checks use);
* `BatchSim`     many independent facilities stepped together through padded arrays.

The facility module calls the same `settle_tick` and then moves ledger money for exactly the integer amounts it
returns, so the two hosts cannot drift apart by construction; `tests/test_mech_rig.py` checks it tick for tick.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Callable, Sequence

import numpy as np

from ..core.canonical import canonical_bytes, digest
from .base import ENGINES, Engine, StepResult, build_engine, tick_rng

MAX_CENTS = 10 ** 11            # no single per-tick flow is ever larger (keeps ledger amounts sane)
_INF = float("inf")


# ====================================================================== engines that exist right now
def ensure_engines() -> list[str]:
    """Import every engine module that can be imported and return the names of the registered engines. Engines
    are written independently: a missing or broken engine file must not take the others down, and a later call
    picks up engines that have appeared since."""
    from . import load_engines
    try:
        load_engines()
    except Exception:                                   # noqa: BLE001 - another engine's file is not ready
        import importlib
        for name in ("scm", "abm", "techsys"):
            try:
                importlib.import_module(f"{__package__}.{name}")
            except Exception:                           # noqa: BLE001
                pass
    return sorted(ENGINES)


# ====================================================================== the per-tick generator
_M32 = 0xFFFFFFFF
_M128 = (1 << 128) - 1
_INIT_A, _MULT_A, _INIT_B, _MULT_B = 0x43B0D7E5, 0x931E8875, 0x8B51F9DD, 0x58F38DED
_MIX_L, _MIX_R = 0xCA01F9DD, 0x4973F715
_PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def _make_pcg64_state():
    """Build `(e0, e1, e2, e3) -> (state, inc)`: the state of `numpy.random.PCG64(numpy.random.SeedSequence(x))`
    for the 128-bit entropy x whose little-endian 32-bit words are e0..e3. numpy's SeedSequence hash and PCG64
    seeding are documented and version-stable; the hash constants do not depend on the data, so they are folded
    in here and the whole computation is straight-line integer arithmetic."""
    ca = [_INIT_A]
    for _ in range(16):
        ca.append((ca[-1] * _MULT_A) & _M32)
    cb = [_INIT_B]
    for _ in range(8):
        cb.append((cb[-1] * _MULT_B) & _M32)
    lines = ["def _pcg64_state(e0, e1, e2, e3):"]
    for i in range(4):
        lines += [f"    v = ((e{i} ^ {ca[i]}) * {ca[i + 1]}) & {_M32}", f"    p{i} = v ^ (v >> 16)"]
    k = 4
    for i_src in range(4):
        for i_dst in range(4):
            if i_src != i_dst:
                lines += [f"    v = ((p{i_src} ^ {ca[k]}) * {ca[k + 1]}) & {_M32}", "    v ^= v >> 16",
                          f"    v = ({_MIX_L} * p{i_dst} - {_MIX_R} * v) & {_M32}", f"    p{i_dst} = v ^ (v >> 16)"]
                k += 1
    for i in range(8):
        lines += [f"    v = ((p{i & 3} ^ {cb[i]}) * {cb[i + 1]}) & {_M32}", f"    w{i} = v ^ (v >> 16)"]
    lines += ["    inc = ((((w4 | (w5 << 32)) << 64) | w6 | (w7 << 32)) << 1 | 1) & M128",
              "    return ((inc + (((w0 | (w1 << 32)) << 64) | w2 | (w3 << 32))) * MULT + inc) & M128, inc"]
    ns = {"M128": _M128, "MULT": _PCG_MULT}
    exec(compile("\n".join(lines), "<pcg64-seed>", "exec"), ns)        # noqa: S102 - constants and arithmetic only
    return ns["_pcg64_state"]


_pcg64_state = _make_pcg64_state()
_unpack_words = struct.Struct(">4I").unpack


def _seed_words_block(digests: bytes, n: int) -> list[tuple[int, int, int, int]]:
    """The same hash as `_pcg64_state` for n entropies at once (uint32 arithmetic wraps exactly like the C code):
    per entropy the four uint64 words numpy feeds to the PCG64 seeding routine."""
    e = np.frombuffer(digests, dtype=">u4").reshape(n, 4).astype(np.uint32)
    p = []
    for i in range(4):
        v = (e[:, 3 - i] ^ _CA[i]) * _CA[i + 1]
        p.append(v ^ (v >> _S16))
    k = 4
    for i_src in range(4):
        for i_dst in range(4):
            if i_src != i_dst:
                v = (p[i_src] ^ _CA[k]) * _CA[k + 1]
                v ^= v >> _S16
                v = _L32 * p[i_dst] - _R32 * v
                p[i_dst] = v ^ (v >> _S16)
                k += 1
    w = []
    for i in range(8):
        v = (p[i & 3] ^ _CB[i]) * _CB[i + 1]
        w.append((v ^ (v >> _S16)).astype(np.uint64))
    u = [(w[2 * i] | (w[2 * i + 1] << _S32)).tolist() for i in range(4)]
    return list(zip(*u))


_S16, _S32 = np.uint32(16), np.uint64(32)
_L32, _R32 = np.uint32(_MIX_L), np.uint32(_MIX_R)
_CA = [np.uint32(_INIT_A)]
for _ in range(16):
    _CA.append(np.uint32((int(_CA[-1]) * _MULT_A) & _M32))
_CB = [np.uint32(_INIT_B)]
for _ in range(8):
    _CB.append(np.uint32((int(_CB[-1]) * _MULT_B) & _M32))


class TickRng:
    """`rng = TickRng(world_seed, rig_id); rng(t)` returns a generator in exactly the state of
    `tick_rng(world_seed, rig_id, t)`, several times cheaper: the seed hashes of consecutive ticks are computed a
    block at a time and one generator object is re-seeded in place. Because that object is reused, finish
    drawing for one tick before asking for the next. Falls back to `tick_rng` itself if this numpy does not
    reproduce the fast path (checked once at import)."""

    def __init__(self, world_seed: int, rig_id: str):
        self.world_seed, self.rig_id = int(world_seed), str(rig_id)
        self._prefix = canonical_bytes([self.world_seed, "rig", self.rig_id])[:-1] + b',"'
        self._bg = np.random.PCG64(0)
        self._gen = np.random.Generator(self._bg)
        self._st = {"bit_generator": "PCG64", "state": {"state": 0, "inc": 1}, "has_uint32": 0, "uinteger": 0}
        self._t0, self._blk, self._last, self._run = 0, [], -2, 0

    def prefetch(self, t: int, n: int) -> None:
        """Compute the seeds of ticks t .. t+n-1 in one vectorised block (worth it from about 16 ticks)."""
        if FAST_RNG and n > 0:
            pre = self._prefix
            self._blk = _seed_words_block(b"".join(
                [hashlib.blake2b(pre + b'%d"]' % i, digest_size=16).digest() for i in range(t, t + n)]), n)
            self._t0 = t

    def __call__(self, t: int | str) -> np.random.Generator:
        if not FAST_RNG:
            return tick_rng(self.world_seed, self.rig_id, t)
        s = self._st["state"]
        if type(t) is int and t >= 0:
            k = t - self._t0
            if not 0 <= k < len(self._blk):
                self._run = self._run + 1 if t == self._last + 1 else 0
                if self._run >= 32:                    # sustained consecutive ticks: switch to blocks
                    self.prefetch(t, min(256, max(64, 2 * len(self._blk))))
                    k = 0
                else:
                    k = -1
            self._last = t
        else:
            k = -1
        if k >= 0:
            u0, u1, u2, u3 = self._blk[k]
            inc = ((((u2 << 64) | u3) << 1) | 1) & _M128
            s["state"], s["inc"] = ((inc + ((u0 << 64) | u1)) * _PCG_MULT + inc) & _M128, inc
        else:
            e3, e2, e1, e0 = _unpack_words(hashlib.blake2b(self._prefix + str(t).encode() + b'"]',
                                                           digest_size=16).digest())
            s["state"], s["inc"] = _pcg64_state(e0, e1, e2, e3)
        self._bg.state = self._st
        return self._gen


def _fast_rng_ok() -> bool:
    try:
        for seed, rig, t in ((0, "r1", 0), (7, "r1", "init"), (2**40 + 3, 'a"b/é', 123456), (1, "", 9)):
            a = tick_rng(seed, rig, t)
            pre = canonical_bytes([int(seed), "rig", str(rig)])[:-1] + b',"'
            e3, e2, e1, e0 = _unpack_words(hashlib.blake2b(pre + str(t).encode() + b'"]', digest_size=16).digest())
            bg = np.random.PCG64(0)
            st, inc = _pcg64_state(e0, e1, e2, e3)
            bg.state = {"bit_generator": "PCG64", "state": {"state": st, "inc": inc}, "has_uint32": 0, "uinteger": 0}
            b = np.random.Generator(bg)
            if a.standard_normal(5).tobytes() != b.standard_normal(5).tobytes() or \
                    int(a.integers(1 << 62)) != int(b.integers(1 << 62)) or a.random() != b.random():
                return False
        global FAST_RNG
        FAST_RNG = True                                 # the block path is checked through the class itself
        try:
            tr = TickRng(11, "r1")
            for t in list(range(40)) + [500, 3]:
                if tr(t).bit_generator.state != tick_rng(11, "r1", t).bit_generator.state:
                    return False
            tr.prefetch(90, 5)
            if tr(92).bit_generator.state != tick_rng(11, "r1", 92).bit_generator.state:
                return False
        finally:
            FAST_RNG = False
        return True
    except Exception:                                   # noqa: BLE001
        return False


FAST_RNG = False
FAST_RNG = _fast_rng_ok()


# ====================================================================== settings
def lever_specs(engine: Engine, seat: int) -> list[tuple[str, float, float, float, bool]]:
    return [(lv.name, float(lv.lo), float(lv.hi), float(lv.default), bool(lv.integer)) for lv in engine.levers(seat)]


def clip_value(v: Any, lo: float, hi: float, default: float, integer: bool) -> float:
    """Same result as `Lever.clip`."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = default
    if x != x or x == _INF or x == -_INF:
        x = default
    x = hi if x > hi else (lo if x < lo else x)
    return float(round(x)) if integer else x


def cents(x: Any) -> int:
    """A per-tick amount an engine reported, as whole non-negative ledger cents."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0
    if not (x > 0.0) or x == _INF:
        return 0
    return min(MAX_CENTS, int(round(x)))


def _num(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


# ====================================================================== one tick of a facility
def settle_tick(engine: Engine, state: dict[str, np.ndarray], t: int, rng_for: Callable[[int], np.random.Generator],
                desired: list[dict[str, float]], defaults: list[dict[str, float]], held: Sequence[bool],
                funds: Sequence[int | None], demand: int | None) -> tuple[StepResult, list[dict[str, float]], list[dict]]:
    """Run tick `t`.

    `desired[s]` is the complete, clipped settings seat s asks for (defaults for an empty seat), `held[s]` whether
    somebody operates the seat, `funds[s]` what that holder can pay this tick (None: unlimited), `demand` what the
    facility's demand account holds after this tick's income (None: unlimited).

    Returns (engine result of the run that counts, the settings every seat actually ran at, one record per seat)::

        {"status": "ok" | "fallback" | "idle" | "empty", "earned": int, "revenue": int, "shortfall": int,
         "cost": int, "score": float, "readings": {gauge: float}}

    * ok        ran at its own settings;
    * fallback  the holder could not pay the cost of its settings: the seat ran at defaults this tick;
    * idle      the holder could not pay even the cost at defaults: nothing produced, nothing charged or paid;
    * empty     nobody holds the seat: it runs at defaults and earns nothing for anyone.

    `earned` is what the engine's independent demand would pay, `revenue` what is actually paid (never more than
    the demand budget; several seats share a short budget pro rata), `shortfall` the difference. All money is in
    whole cents. If a second engine run is needed it starts from the same state with the same tick generator.
    """
    n = engine.n_seats
    res = engine.step(state, desired, rng_for(t), t)
    used = desired
    status = ["ok" if held[s] else "empty" for s in range(n)]
    cost = [cents(res.cost[s]) for s in range(n)]
    short = [s for s in range(n) if held[s] and funds[s] is not None and cost[s] > funds[s]]
    if short:
        used = list(desired)
        rerun = False
        for s in short:
            status[s] = "fallback"
            if used[s] != defaults[s]:
                used[s] = defaults[s]
                rerun = True
        if rerun:
            res = engine.step(state, used, rng_for(t), t)
            cost = [cents(res.cost[s]) for s in range(n)]
        for s in range(n):
            if held[s] and funds[s] is not None and cost[s] > funds[s]:
                status[s] = "idle"
    earned = [cents(res.revenue[s]) if status[s] in ("ok", "fallback") else 0 for s in range(n)]
    total = sum(earned)
    if demand is not None and total > demand:
        d = max(0, int(demand))
        paid = [e * d // total for e in earned]
    else:
        paid = earned
    seats = []
    for s in range(n):
        live = status[s] in ("ok", "fallback")
        rd = res.readings[s]
        try:                                   # the common case: plain finite floats under string names
            clean = math.isfinite(math.fsum(rd.values())) and all(type(v) is float for v in rd.values())
        except (TypeError, ValueError, OverflowError):
            clean = False
        if not clean:
            rd = {str(k): _num(v) for k, v in rd.items()}
        seats.append({"status": status[s], "earned": earned[s], "revenue": paid[s], "shortfall": earned[s] - paid[s],
                      "cost": cost[s] if live else 0, "score": _num(res.score[s]) if live else 0.0, "readings": rd})
    return res, used, seats


# ====================================================================== one facility
class MechSim:
    """A hidden mechanism run directly: the same readings, revenue, cost and score, tick for tick, as the facility
    `rig_id` hosting `mech` in a kernel world with seed `world_seed`.

    Settings persist between ticks, exactly like `rig.set`: `step(x)` overlays the partial mapping `x` on the
    seat's current settings and then runs one tick.

    By default every seat is held by an operator with unlimited funds and the demand budget is unlimited (a pure
    mechanism). Pass `income_cents_per_tick` (the facility's demand income), `funds` (each holder's starting
    balance in cents; the simulator then tracks it as balance + revenue - cost) and `held` to reproduce a
    particular hosted facility including its money constraints.
    """

    def __init__(self, mech: dict, world_seed: int, rig_id: str = "r1", *, income_cents_per_tick: int | None = None,
                 funds: Sequence[int | None] | None = None, held: Sequence[bool] | None = None,
                 engine: Engine | None = None):
        if engine is None:
            if mech["engine"] not in ENGINES:
                ensure_engines()
            engine = build_engine(mech)
        self.mech, self.engine = mech, engine
        self.world_seed, self.rig_id = int(world_seed), str(rig_id)
        self.n_seats = engine.n_seats
        self._rng = TickRng(world_seed, rig_id)
        self._specs = [lever_specs(engine, s) for s in range(self.n_seats)]
        self._by_name = [{sp[0]: sp for sp in specs} for specs in self._specs]
        self._defaults = [engine.defaults(s) for s in range(self.n_seats)]
        self._income = None if income_cents_per_tick is None else int(income_cents_per_tick)
        self._funds0 = None if funds is None else list(funds)
        self._held = [True] * self.n_seats if held is None else [bool(x) for x in held]
        self.reset()

    # ---- control -----------------------------------------------------------------------------------
    def reset(self) -> "MechSim":
        self.t = 0
        self.state = self.engine.init_state(self._rng("init"))
        self.settings = [dict(d) for d in self._defaults]
        self.funds: list[int | None] = [None] * self.n_seats if self._funds0 is None else list(self._funds0)
        self.demand: int | None = None if self._income is None else 0
        self.last: list[dict] | None = None        # per-seat records of the latest tick (see settle_tick)
        self.info: dict = {}                       # the engine's private diagnostics of the latest tick
        return self

    def snapshot(self) -> tuple:
        """Everything needed to come back to this point (`restore`). Engine states are never mutated in place,
        so the snapshot shares them."""
        return (self.t, self.state, [dict(x) for x in self.settings], list(self.funds), self.demand, self.last)

    def restore(self, snap: tuple) -> "MechSim":
        self.t, self.state, settings, funds, self.demand, self.last = snap
        self.settings, self.funds = [dict(x) for x in settings], list(funds)
        return self

    def set(self, seat: int, settings: dict[str, Any] | None) -> dict[str, float]:
        """Overlay a partial mapping on a seat's settings (values clipped to range). Unknown names are errors."""
        cur = self.settings[seat]
        if settings:
            by = self._by_name[seat]
            for k, v in settings.items():
                sp = by.get(k)
                if sp is None:
                    raise ValueError(f"unknown lever {k!r} for seat {seat}; valid names: {sorted(by)}")
                cur[k] = clip_value(v, sp[1], sp[2], sp[3], sp[4])
        return cur

    def step(self, settings_per_seat: Any = None) -> StepResult:
        """Run one tick. `settings_per_seat`: None (keep), a list with one partial mapping (or None) per seat, or
        a single mapping when the mechanism has one seat. The returned `revenue` and `cost` are the whole cents
        actually paid and charged (floats), i.e. what the hosted facility moves on the ledger."""
        if settings_per_seat is not None:
            if isinstance(settings_per_seat, dict):
                self.set(0, settings_per_seat)
            else:
                for s, x in enumerate(settings_per_seat):
                    if x:
                        self.set(s, x)
        if self.demand is not None:
            self.demand += self._income
        desired = [dict(x) for x in self.settings]
        res, used, seats = settle_tick(self.engine, self.state, self.t, self._rng, desired, self._defaults,
                                       self._held, self.funds, self.demand)
        for s, r in enumerate(seats):
            r["settings"] = used[s]
            if self.funds[s] is not None:
                self.funds[s] += r["revenue"] - r["cost"]
            if self.demand is not None:
                self.demand -= r["revenue"]
        self.state, self.last, self.info = res.state, seats, res.info
        self.t += 1
        return StepResult(res.state, [r["readings"] for r in seats], [float(r["revenue"]) for r in seats],
                          [float(r["cost"]) for r in seats], [r["score"] for r in seats], res.info)

    def run(self, policy: Any, ticks: int) -> dict:
        """Run `ticks` ticks. `policy` is None (keep current settings), a constant (a mapping for one seat, or a
        list per seat), or a callable `policy(t, last) -> settings_per_seat` where `last` is the list of per-seat
        records of the previous tick (None before the first). Returns arrays of shape (ticks, seats) for
        `revenue`, `cost`, `net`, `score`, plus the per-tick `readings` and `status` lists."""
        n = self.n_seats
        rev, cost, score = np.zeros((ticks, n)), np.zeros((ticks, n)), np.zeros((ticks, n))
        readings, status = [], []
        fn = policy if callable(policy) else None
        if ticks >= 16:
            self._rng.prefetch(self.t, ticks)
        for k in range(ticks):
            x = fn(self.t, self.last) if fn is not None else (policy if k == 0 else None)
            r = self.step(x)
            rev[k], cost[k], score[k] = r.revenue, r.cost, r.score
            readings.append(r.readings)
            status.append([s["status"] for s in self.last])
        return {"revenue": rev, "cost": cost, "net": rev - cost, "score": score, "readings": readings,
                "status": status}


def suggest_income(mech: dict, seeds: int = 2, margin: float = 1.25) -> int:
    """A demand income per tick for a mechanism whose engine declares none (`config.income_cents_per_tick`):
    `margin` times the largest total revenue any tick earned in seeded runs at default, extreme and random
    settings over the suggested horizon."""
    key = (digest(mech), int(seeds), float(margin))
    if key in _INCOME_CACHE:
        return _INCOME_CACHE[key]
    top = 0.0
    for k in range(seeds):
        sim = MechSim(mech, 7919 + k, "income-probe")
        eng = sim.engine
        rng = np.random.default_rng([4243, k])
        lo = [{sp[0]: sp[1] for sp in specs} for specs in sim._specs]
        hi = [{sp[0]: sp[2] for sp in specs} for specs in sim._specs]
        for mode in ("default", "lo", "hi", "random"):
            sim.reset()
            for t in range(eng.horizon):
                if mode == "random":
                    x = [{sp[0]: float(rng.uniform(sp[1], sp[2])) for sp in specs} for specs in sim._specs]
                else:
                    x = None if t else {"default": None, "lo": lo, "hi": hi}[mode]
                top = max(top, sum(sim.step(x).revenue))
    if len(_INCOME_CACHE) >= 64:
        _INCOME_CACHE.clear()
    _INCOME_CACHE[key] = int(math.ceil(margin * top)) + 1
    return _INCOME_CACHE[key]


_INCOME_CACHE: dict[tuple, int] = {}       # the probe is a pure function of the mechanism; engines can be slow


# ====================================================================== many facilities
class BatchSim:
    """Independent facilities stepped together for a learner. Mechanisms differ in how many levers and gauges
    they have, so numeric views are padded to the widest and masked::

        lo, hi, default   (n, max_levers)   lever ranges in lever order of the controlled seat
        lever_mask        (n, max_levers)   True where a lever exists
        gauge_mask        (n, max_gauges)   True where a gauge exists

    `step(actions)` takes an (n, max_levers) array in lever units (or in [0, 1] with `normalized=True`), or a list
    of partial mappings, and returns arrays `readings` (n, max_gauges), `revenue`, `cost`, `net`, `score` (n,).
    Only seat `seat` is controlled; other seats of a multi-seat mechanism stay at their current settings."""

    def __init__(self, mechs: Sequence[dict], world_seeds: Sequence[int] | int = 0, rig_id: str = "r1", seat: int = 0,
                 **kw: Any):
        if isinstance(world_seeds, (int, np.integer)):
            world_seeds = [int(world_seeds) + i for i in range(len(mechs))]
        if len(world_seeds) != len(mechs):
            raise ValueError("one world seed per mechanism")
        self.seat, self.rig_id, self._kw = int(seat), rig_id, kw
        self.sims = [MechSim(m, s, rig_id, **kw) for m, s in zip(mechs, world_seeds)]
        self._layout()

    def _layout(self) -> None:
        self.n = len(self.sims)
        self.lever_names = [[sp[0] for sp in s._specs[self.seat]] for s in self.sims]
        self.gauge_names = [[g.name for g in s.engine.gauges(self.seat)] for s in self.sims]
        self.max_levers = max((len(x) for x in self.lever_names), default=0)
        self.max_gauges = max((len(x) for x in self.gauge_names), default=0)
        self.lo = np.zeros((self.n, self.max_levers))
        self.hi = np.ones((self.n, self.max_levers))
        self.default = np.zeros((self.n, self.max_levers))
        self.lever_mask = np.zeros((self.n, self.max_levers), dtype=bool)
        self.gauge_mask = np.zeros((self.n, self.max_gauges), dtype=bool)
        for i, s in enumerate(self.sims):
            for j, (_, lo, hi, d, _) in enumerate(s._specs[self.seat]):
                self.lo[i, j], self.hi[i, j], self.default[i, j] = lo, hi, d
                self.lever_mask[i, j] = True
            self.gauge_mask[i, :len(self.gauge_names[i])] = True

    def replace(self, i: int, mech: dict, world_seed: int) -> None:
        """Put a new mechanism in slot i (padded shapes may grow)."""
        self.sims[i] = MechSim(mech, world_seed, self.rig_id, **self._kw)
        self._layout()

    def reset(self, indices: Sequence[int] | None = None) -> None:
        for i in (range(self.n) if indices is None else indices):
            self.sims[i].reset()

    def step(self, actions: Any = None, normalized: bool = False) -> dict[str, np.ndarray]:
        n, seat = self.n, self.seat
        out = {"readings": np.zeros((n, self.max_gauges)), "revenue": np.zeros(n), "cost": np.zeros(n),
               "score": np.zeros(n), "t": np.zeros(n, dtype=np.int64)}
        arr = None
        if isinstance(actions, np.ndarray):
            arr = actions.astype(np.float64, copy=False)
            if normalized:
                arr = self.lo + np.clip(arr, 0.0, 1.0) * (self.hi - self.lo)
            arr = arr.tolist()
        for i, sim in enumerate(self.sims):
            if arr is not None:
                x = dict(zip(self.lever_names[i], arr[i]))
            else:
                x = actions[i] if actions is not None else None
            if x is not None and sim.n_seats > 1:
                x = [x if s == seat else None for s in range(sim.n_seats)]
            elif x is not None and seat != 0:
                raise ValueError("seat out of range")
            r = sim.step(x)
            rd = r.readings[seat]
            out["readings"][i, :len(rd)] = [rd[g] for g in self.gauge_names[i]]
            out["revenue"][i], out["cost"][i], out["score"][i] = r.revenue[seat], r.cost[seat], r.score[seat]
            out["t"][i] = sim.t
        out["net"] = out["revenue"] - out["cost"]
        return out
