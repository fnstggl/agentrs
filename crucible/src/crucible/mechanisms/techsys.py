"""Executable technical systems with hidden operational constraints (engine ``techsys``).

A mechanism of this engine is a small *service system*: jobs arrive from outside, pass through a generated graph
of processing stages and are worth money only if they leave the last stage before their deadline. The seat that
operates the facility sets operational quantities (how many workers a stage has, how work is batched, how much is
cached, how often a failed attempt is repeated, how long a job may stay, how many jobs may be inside at once,
how traffic is split between two alternative paths) and reads operational gauges. Nothing tells it which lever is
which: names are opaque and the constraints that decide whether a setting helps or hurts are hidden.

The simulation is a discrete-time queueing model over **integer job counts**, which is why job conservation is
exact. One tick is `substeps` service rounds; in each round

1. arrivals of each line are offered at the door: a job is *shed* if the number of jobs already inside the line
   has reached the admission limit, *dropped* if the first buffer is full, and otherwise queued with age 0;
2. every stage, in topological order, serves its queue oldest-first. What it can serve this round is its capacity
   credit (jobs per tick / substeps, fractions carried), cut to whole batches when the stage batches, and cut by
   the free space of the next buffer (back-pressure: a full downstream buffer blocks the stage above it; a stage
   that splits its output between two paths is blocked by whichever path is full);
3. each served attempt fails with a probability made of the stage's base error rate (which a scheduled
   degradation of its dependency can raise) and of *staleness*: when the time needed to clear the stage's queue
   exceeds the patience of the callers, attempts are abandoned while they wait and the work spent on them is
   wasted. A failed attempt is repeated (the job stays queued and consumes capacity again) or given up according
   to the retry limit.

At the end of the tick every queued job ages by one tick. A job leaving the last stage is delivered a hidden
number of ticks later (`settle`: its latency is its queue age plus that), and it is *completed* (paid for by the
customers) if its latency is within its deadline, otherwise *late* (worth nothing, but it consumed capacity all
the way). Deadlines differ between jobs: all are still wanted at latency <= `deadline`, a hidden fraction one
and two ticks later, none after that. The timeout is a bound on latency: a queued job that can no longer be
delivered within it is discarded at the end of the tick, and a job served too late for it is discarded at the
door of the last stage. A timeout below `settle` therefore delivers nothing, one just above it sheds everything
that has to wait, and a long one lets worthless late work clog the stages.

Hidden constraints (which of them are active and how they are wired is the family's and the seed's choice):

* **service rates** per stage and worker, with small noise; a stage may also have a hard ceiling (what its
  dependency can take), beyond which more workers add nothing;
* **start-up lag**: a worker counts only after it has been requested for `lag` consecutive ticks (scale-up is
  delayed, scale-down is immediate, the requested number is what is paid for and what draws on the pool);
* **batching**: per-worker throughput is ``(1+h) b / (h+b)`` up to a hidden knee and falls beyond it
  (``/ (1 + over ((b-knee)/knee)^2)``); a stage only starts whole batches, so a large batch also makes jobs wait
  for the batch to fill;
* a **shared pool**: workers (more with larger batches) and caches of all stages draw on one pool; above a hidden
  utilisation every stage slows by ``1 / (1 + kappa x^power)``, ``x = (u - theta)/(1 - theta)`` (thrashing);
* **caches** with diminishing returns, ``hit = hmax c / (c + half)``, rate ``/ (1 - save hit)``, drawing on the pool;
* **retries**: with retry limit R and per-attempt failure probability e, a failed attempt is given up with
  probability ``e^R (1-e) / (1 - e^(R+1))``, which gives exactly the loss ``e^(R+1)`` and the load amplification
  ``(1 - e^(R+1)) / (1 - e)`` of "at most R retries" at constant e without tracking attempts per job. Together
  with staleness this produces **metastable collapse**: once the queue is long, almost every attempt is stale,
  every job is attempted R+1 times, the amplified load keeps the queue long, and the stage stays collapsed after
  whatever caused the backlog is gone, until load is shed;
* **timeout** (bound on latency, above), **admission limit** (maximum jobs inside a line), **routing weight**
  (share of a split's output sent to its first successor; exact up to a carried fraction);
* **regimes**: scheduled, hidden changes of a dependency (rate or error rate of a stage) or of the arrivals;
* **arrivals**: Poisson around a daily cycle with a slow drift and bursts from a two-state chain; nothing a seat
  does changes them.

Money. Revenue is `value` cents per completed job, paid by the customers, so cumulative revenue can never exceed
``value x arrivals``, and what is earned in one tick can never exceed `config["income_cents_per_tick"]` (the last
stage of every line has a delivery ceiling; both bounds are tested). Cost is workers + pool use + cache. Objectives: `net` (own revenue minus own cost),
`team_net` (every seat scores the facility's net divided by the number of seats), `ontime_value` (the value
completed this tick, counted only while a running balance of revenue minus cost, started with a reserve and
clamped to +-`cap`, is not negative).

Private accounting, `StepResult.info`, per tick: `arrivals`, `completed`, `late`, `dropped`, `shed`, `queued`,
`queued_before` (job counts; arrivals = completed + late + dropped + shed + queued - queued_before, exactly),
`value_arrived`, `value_completed`, `value_lost` and `lost_by_reason` with exactly the keys `overload` (buffer
overflow at the door, attempts abandoned as stale, and jobs delivered late), `timeout` (discarded by the
timeout), `failure` (failed after the allowed retries) and `shed` (refused by the admission limit); the same per
line under `lines` (plus `admitted`, `overflow`, `timed_out`, `failed`, `abandoned`, noise-free `latency_p50` /
`latency_p95`); and the latent state of the tick: `pool_util`, `slowdown`, `capacity`, `workers_effective`,
`queue`, `throughput`, `attempts`, `failures`, `failure_prob` per stage, `error_rate` per seat, `reserve`.

Lever kinds (private; `config["levers"][i]["kind"]`, `targets` are stage indices, or line indices for the last
two): `workers`, `batch`, `cache`, `retry`, `route`, `noop` (a distractor: no effect at all), `timeout`, `admit`.
Gauge kinds: `done` (completed in time), `late`, `lat50` / `lat95` (noisy latency percentiles of what was
delivered this tick; they hold their last value when nothing was), `err` (noisy share of the seat's attempts
that failed), `queue` (a stage's backlog after the end-of-tick sweep), `pool` (utilisation), `shed` (refused at
the door, limit or overflow), `tout` (discarded by the timeout), `offered` (arrivals), `handoff` (a stage's
output). Families differ in which gauges exist: what can be seen is part of the structure.

Everything public is opaque (`neutral_names`, seeded order, empty docs). The config is plain JSON; the instance
compiles lookup tables from it and keeps no other state. `step` uses no numpy arithmetic on the hot path: state
arrays are converted to Python lists, advanced with integer and float arithmetic, and returned as fresh arrays.

This module is written independently of the other engines and shares nothing with them but `base`.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Engine, Gauge, Lever, StepResult, mech_family, neutral_names, register_engine

_TWO_PI = 2.0 * math.pi
LEVER_KINDS = ("workers", "batch", "cache", "retry", "route", "timeout", "admit", "noop")
GAUGE_KINDS = ("done", "late", "lat50", "lat95", "err", "queue", "pool", "shed", "tout", "offered", "handoff")
LOSS_REASONS = ("overload", "timeout", "failure", "shed")


# ====================================================================== functional forms (shared with truth/tests)
def batch_gain(bt: dict | None, b: float) -> float:
    """Per-worker throughput multiplier of batch size `b` (1.0 at b = 1; peaks at the knee)."""
    if bt is None:
        return 1.0
    h, kn = bt["setup"], bt["knee"]
    if b <= kn:
        return (1.0 + h) * b / (h + b)
    x = (b - kn) / kn
    return (1.0 + h) * kn / (h + kn) / (1.0 + bt["over"] * x * x)


def cache_gain(ch: dict | None, c: float) -> float:
    """Service-rate multiplier of cache size `c` (1.0 at c = 0; saturates at 1 / (1 - save * hmax))."""
    if ch is None:
        return 1.0
    return 1.0 / (1.0 - ch["save"] * ch["hmax"] * c / (c + ch["half"]))


def pool_slowdown(pool: dict | None, util: float) -> float:
    """Factor (<= 1) applied to every stage's rate at pool utilisation `util`."""
    if pool is None or util <= pool["theta"]:
        return 1.0
    x = (util - pool["theta"]) / (1.0 - pool["theta"])
    return 1.0 / (1.0 + pool["kappa"] * x ** pool["power"])


def giveup_probability(e: float, limit: int) -> float:
    """Probability that a failed attempt is the last one allowed, for retry limit `limit` and failure prob `e`."""
    if limit <= 0:
        return 1.0
    if e >= 0.999999:
        return 1.0 / (limit + 1.0)
    er = e ** limit
    return er * (1.0 - e) / (1.0 - er * e)


def _percentile(hist: list[int], total: int, frac: float) -> float:
    """Percentile of a histogram over integer ages, linear within a bin."""
    target = frac * total
    cum = 0
    for a, c in enumerate(hist):
        if c:
            if cum + c >= target:
                return a + (target - cum) / c
            cum += c
    return float(len(hist))


# ====================================================================== compiled plan
def _clean(v, lo: float, hi: float, isint: bool, dflt: float) -> float:
    """Slow path for a setting that is not a number inside the lever's range (same rule as `Lever.clip`)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = dflt
    if x - x != 0.0:                 # nan or inf
        x = dflt
    x = lo if x < lo else (hi if x > hi else x)
    return float(round(x)) if isint else x


class _Plan:
    """Lookup tables compiled once from the JSON config. Nothing here is state."""

    def __init__(self, cfg: dict):
        st = cfg["stages"]
        self.S = S = len(st)
        self.NL = NL = len(cfg["lines"])
        self.n_seats = int(cfg.get("n_seats", 1))
        self.K = int(cfg["substeps"])
        self.A1 = int(cfg["ages"]) + 1
        self.H = float(cfg["horizon"])
        self.objective = cfg["objective"]
        self.mu = [float(s["mu"]) for s in st]
        self.lag = [int(s["lag"]) for s in st]
        self.Lmax = max(self.lag)
        self.jit = [min(0.06, max(0.0, float(s["jitter"]))) for s in st]
        self.buf = [int(s["buf"]) for s in st]
        self.wmem = [float(s["wmem"]) for s in st]
        self.wcost = [float(s["wcost"]) for s in st]
        self.eb = [float(s["err"]) for s in st]
        self.line_of = [int(s["line"]) for s in st]
        self.seat_of = [int(s["seat"]) for s in st]
        self.nxt = [tuple(int(j) for j in s["next"]) for s in st]
        self.bat = []       # (stage, setup, knee, over, gain at knee, pool factor, span)
        self.cch = []       # (stage, save * hmax, half, pool units per unit, cost per unit)
        for i, s in enumerate(st):
            bt, ch = s["batch"], s["cache"]
            if bt is not None:
                self.bat.append((i, float(bt["setup"]), float(bt["knee"]), float(bt["over"]),
                                 batch_gain(bt, bt["knee"]), float(bt["mem"]), float(max(1, int(bt["max"]) - 1))))
            if ch is not None:
                self.cch.append((i, float(ch["save"]) * float(ch["hmax"]), float(ch["half"]), float(ch["mem"]),
                                 float(ch["cost"])))
        self.capped = [(i, float(s["cap"])) for i, s in enumerate(st) if s.get("cap") is not None]
        self.stl = [None if s["stale"] is None else
                    (float(s["stale"]["tau"]), float(s["stale"]["tau"]) * float(s["stale"]["ramp"]),
                     float(s["stale"]["max"])) for s in st]
        pool = cfg.get("pool")
        self.pool = None if pool is None else (float(pool["size"]), float(pool["theta"]), float(pool["kappa"]),
                                               float(pool["power"]), float(pool["cost"]))
        for i, nx in enumerate(self.nxt):
            if len(nx) > 2:
                raise ValueError("a stage has at most two successors")
            for j in nx:
                if j <= i:
                    raise ValueError("stages must be listed in topological order")
        self.lines = cfg["lines"]
        self.src = [int(ln["source"]) for ln in self.lines]
        # a job is delivered `settle` ticks after it leaves the last stage, so queue age a means latency
        # a + settle; deadlines and the timeout are about latency and are stored here in queue-age terms
        self.settle = [int(ln.get("settle", 0)) for ln in self.lines]
        self.deadline = [int(ln["deadline"]) - int(ln.get("settle", 0)) for ln in self.lines]
        # jobs have their own deadlines: all are still wanted at latency <= deadline, a fraction tail[k] at
        # latency deadline + 1 + k, none later
        self.tail = [[(d + 1 + k, float(f)) for k, f in enumerate(ln.get("tail", [])) if d + 1 + k < self.A1]
                     for d, ln in zip(self.deadline, self.lines)]
        self.sink_line = [int(ln["sink"]) for ln in self.lines]
        self.value = [float(ln["value"]) for ln in self.lines]
        self.share = [[float(x) for x in ln["share"]] for ln in self.lines]
        self.arr = []
        for ln in self.lines:
            a = ln["arrival"]
            peak = float(a["base"]) * (1.0 + abs(float(a["amp"]))) * (1.0 + 1.5 * max(0.0, float(a["drift"]))) \
                * max(1.0, float(a["burst_x"]))
            self.arr.append((float(a["base"]), float(a["amp"]), float(a["period"]), float(a["phase"]),
                             float(a["drift"]), float(a["burst_on"]), float(a["burst_off"]), float(a["burst_x"]),
                             int(4.0 * peak + 30.0)))
        self.regimes = []
        for r in cfg.get("regimes", []):
            code = {"rate": 0, "error": 1, "arrival": 2}[r["what"]]
            until = r.get("until")
            self.regimes.append((int(r["at"]), int(until) if until is not None else 1 << 60, code,
                                 int(r["target"]), float(r["value"])))
        # the operating point is one flat vector: workers, batch, cache, retry limit, routing weight per stage,
        # then timeout and admission limit per line; levers write into it, the rest keeps its fixed value
        self.o_b, self.o_c, self.o_r, self.o_rho, self.o_ttl, self.o_adm, self.o_end = \
            S, 2 * S, 3 * S, 4 * S, 5 * S, 5 * S + NL, 5 * S + 2 * NL
        op = [float(s.get("workers", 1.0)) for s in st] + [1.0] * S + [0.0] * S \
            + [float(s["retry"]) for s in st] + [1.0] * S + [float(ln["ttl"]) for ln in self.lines] \
            + [float(ln["limit"]) for ln in self.lines]
        base = {"workers": 0, "batch": S, "cache": 2 * S, "retry": 3 * S, "route": 4 * S, "timeout": 5 * S,
                "admit": 5 * S + NL}
        self.levers = []    # (seat, name, lo, hi, integer, default, slot); one entry per target
        self.lever_objs: list[list[Lever]] = [[] for _ in range(self.n_seats)]
        w_def = op[:S]
        for lv in cfg["levers"]:
            if lv["kind"] not in LEVER_KINDS:
                raise ValueError(f"unknown lever kind {lv['kind']!r}")
            self.lever_objs[int(lv["seat"])].append(
                Lever(lv["name"], float(lv["lo"]), float(lv["hi"]), float(lv["default"]), bool(lv["integer"]), ""))
            targets = [-1] if lv["kind"] == "noop" else [base[lv["kind"]] + int(j) for j in lv["targets"]]
            for slot in targets:
                self.levers.append((int(lv["seat"]), lv["name"], float(lv["lo"]), float(lv["hi"]),
                                    bool(lv["integer"]), float(lv["default"]), slot))
                if 0 <= slot < S:
                    w_def[slot] = float(lv["default"])
        self.op = op + [0.0]        # last slot swallows the levers that control nothing
        self.w_def = w_def
        # noise layout: [0, S) capacity jitter, [S, S + K*S) failure noise, then one draw per noisy gauge
        zi = S + self.K * S
        # readings are looked up in one flat vector of this tick's quantities
        self.v_done, self.v_late, self.v_l50, self.v_l95, self.v_shed, self.v_tout, self.v_off = \
            0, NL, 2 * NL, 3 * NL, 4 * NL, 5 * NL, 6 * NL
        self.v_queue = 7 * NL
        self.v_flow = 7 * NL + S
        self.v_err = 7 * NL + 2 * S
        self.v_pool = self.v_err + self.n_seats
        slot_of = {"done": 0, "late": NL, "lat50": 2 * NL, "lat95": 3 * NL, "shed": 4 * NL, "tout": 5 * NL,
                   "offered": 6 * NL, "queue": self.v_queue, "handoff": self.v_flow}
        self.exact = []     # (seat, name, slot)
        self.noisy = []     # (seat, name, slot, noise index, sigma, upper bound)
        self.gauge_objs: list[list[Gauge]] = [[] for _ in range(self.n_seats)]
        nz = cfg.get("noise", {})
        for g in cfg["gauges"]:
            kind, seat = g["kind"], int(g["seat"])
            if kind not in GAUGE_KINDS:
                raise ValueError(f"unknown gauge kind {kind!r}")
            self.gauge_objs[seat].append(Gauge(g["name"], ""))
            if kind == "err":
                slot, sigma, top = self.v_err + seat, float(nz.get("err", 0.0)), 1.0
            elif kind == "pool":
                slot, sigma, top = self.v_pool, float(nz.get("pool", 0.0)), 1e9
            else:
                slot = slot_of[kind] + int(g["targets"][0])
                sigma, top = (float(nz.get("lat", 0.0)) if kind in ("lat50", "lat95") else 0.0), 1e9
            if sigma > 0.0:
                self.noisy.append((seat, g["name"], slot, zi, sigma, top))
                zi += 1
            else:
                self.exact.append((seat, g["name"], slot))
        self.nz = zi
        self.seat_stages = [[i for i in range(S) if self.seat_of[i] == j] for j in range(self.n_seats)]
        rs = cfg.get("reserve")
        self.reserve = None if rs is None else (float(rs["start"]), float(rs["cap"]))
        self.ones_S = [1.0] * S
        self.ones_NL = [1.0] * NL
        self.zero_tail = [[0] * n for n in range(self.A1 + 1)]
        # layout of the small float state vector
        self.x_l50, self.x_l95, self.x_dl, self.x_res = NL, 2 * NL, 3 * NL, 4 * NL


# ====================================================================== the engine
@register_engine
class TechSys(Engine):
    engine = "techsys"
    version = "1"

    def __init__(self, config: dict):
        super().__init__(config)
        self._plan = _Plan(config)

    @property
    def claims(self) -> tuple[str, ...]:
        """Properties every generated system is calibrated to have (checked by simulation in the tests). With two
        seats the random-settings property is calibrated for both seats acting at random together, not for one
        seat at random against the other's defaults, so it is not claimed per seat."""
        return ("default_zero", "random_loses", "corners_bad") if self.n_seats == 1 else ("default_zero", "corners_bad")

    def constructive_optimum(self, seat: int) -> dict:
        """Settings an operator who knew the mechanism would choose, as a schedule with one entry per stretch
        between scheduled changes: `reference_settings(config, "tuned", at=t)`, which reckons only with the
        permanent changes that have happened by tick t. A strong starting point for a search, not a proven
        optimum. Evaluator only: it is computed from the hidden parameters."""
        cuts = sorted({int(r["at"]) for r in self.truth()["regimes"] if 0 < int(r["at"]) < self.horizon})
        return {"schedule": [{"from": t, "settings": reference_settings(self.config, "tuned", at=t)[seat]}
                             for t in [0] + cuts]}

    # ---- construction ------------------------------------------------------------------------------
    @classmethod
    def generate(cls, rng: np.random.Generator, params: dict) -> dict:
        """Draw a complete system. `params["scenario"]` picks the builder (one per family, see `_SCENARIOS`).
        Optional keys, each a number or a `[lo, hi]` range unless noted: `horizon`, `ages` (largest queue age
        tracked, also the top of the timeout lever), `substeps`, `period` (integers); `arrivals` (mean jobs per
        tick), `amp`, `drift`, `burst_on`, `burst_x`, `deadline` (in queue-age ticks), `settle`, `lag`,
        `default_loss` (share of arriving value the resting configuration should lose, where the scenario
        calibrates it), `default_margin` (how far below break-even the resting configuration sits), `stages`,
        `batch_stages`, `distractors`, `intake` (scenario-specific counts and flags), `hide` (gauge kinds to leave
        out), `no_calibration` (skip the two simulation-based calibrations; for benchmarks only)."""
        p = dict(params or {})
        scenario = p.get("scenario", "pipeline")
        if scenario not in _SCENARIOS:
            raise ValueError(f"unknown techsys scenario {scenario!r}; known: {sorted(_SCENARIOS)}")
        draft = _SCENARIOS[scenario](rng, p)
        return _finalize(cls, rng, draft, p)

    # ---- public surface ----------------------------------------------------------------------------
    def levers(self, seat: int) -> list[Lever]:
        return list(self._plan.lever_objs[seat])

    def gauges(self, seat: int) -> list[Gauge]:
        return list(self._plan.gauge_objs[seat])

    def brief(self, seat: int) -> str:
        obj = self.config["objective"]
        head = ("You operate a facility that performs work for outside customers. Each tick you are paid for the "
                "work it delivers to them while that work is still wanted, and you are charged for what your "
                "settings consume.")
        if self.n_seats > 1 and obj == "team_net":
            return ("You operate one part of a facility that performs work for outside customers; another operator "
                    "runs the other part. Each tick the facility is paid for the work it delivers while that work "
                    "is still wanted, and each operator is charged for what their own settings consume. Both "
                    "operators receive the same score: half of the facility's payments minus half of all charges.")
        if self.n_seats > 1:
            return ("You run one of several operations housed in the same facility; another operator runs each of "
                    "the others. Each tick you are paid for the work your operation delivers to its customers "
                    "while that work is still wanted, and you are charged for what your settings consume. Your "
                    "score is your own payments minus your own charges.")
        if obj == "ontime_value":
            return (head + " Your score is the value of the work delivered while it is still wanted. That work "
                    "counts toward your score only while the operation pays its way: your running balance of "
                    "payments minus charges, which starts with a small reserve, must not be exhausted.")
        return head + " Your score is what you are paid minus what you are charged."

    # ---- dynamics ----------------------------------------------------------------------------------
    def init_state(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Empty queues, every stage running its default number of workers, no burst in progress."""
        p = self._plan
        aux = [0.0] * (4 * p.NL + 1)
        if p.reserve is not None:
            aux[-1] = p.reserve[0]
        return {"q": np.zeros((p.S, p.A1), dtype=np.int64),
                "wh": np.array([[w] * (p.Lmax + 1) for w in p.w_def], dtype=np.float64),
                "car": np.zeros((4, p.S), dtype=np.float64),
                "aux": np.array(aux, dtype=np.float64)}

    def step(self, state, settings, rng, t) -> StepResult:
        p = self._plan
        S, NL, K, A1 = p.S, p.NL, p.K, p.A1
        q = np.asarray(state["q"]).tolist()
        wh = np.asarray(state["wh"]).tolist()
        credit, fcar, dcar, rcar = np.asarray(state["car"]).tolist()
        aux = np.asarray(state["aux"]).tolist()
        t = int(t)

        # ---- 1. operating point from the settings (re-clipped: any input keeps the state bounded)
        op = p.op[:]
        try:
            for seat, name, lo, hi, isint, dflt, slot in p.levers:
                v = float(settings[seat][name])
                if not lo <= v <= hi:
                    v = _clean(v, lo, hi, isint, dflt)
                elif isint:
                    v = float(round(v))
                op[slot] = v
        except (KeyError, IndexError, TypeError, ValueError):
            for seat, name, lo, hi, isint, dflt, slot in p.levers:
                try:
                    v = settings[seat][name]
                except (KeyError, IndexError, TypeError):
                    v = dflt
                op[slot] = _clean(v, lo, hi, isint, dflt)
        w = op[:S]
        b = [int(x) for x in op[S:p.o_c]]
        cz = op[p.o_c:p.o_r]
        rl = [int(x) for x in op[p.o_r:p.o_rho]]
        rho = op[p.o_rho:p.o_ttl]
        ttl = [int(op[p.o_ttl + i]) - p.settle[i] for i in range(NL)]      # in queue-age terms; may be < 0
        adm = [int(x) for x in op[p.o_adm:p.o_end]]

        # ---- 2. randomness of this tick (fixed number and order of draws)
        z = rng.standard_normal(p.nz).tolist()

        # ---- 3. scheduled regimes and arrivals
        rate = None
        eb = p.eb
        arrf = p.ones_NL
        for at, until, code, tg, val in p.regimes:
            if at <= t < until:
                if code == 0:
                    if rate is None:
                        rate = p.ones_S[:]
                    rate[tg] *= val
                elif code == 1:
                    if eb is p.eb:
                        eb = eb[:]
                    eb[tg] = val
                else:
                    if arrf is p.ones_NL:
                        arrf = arrf[:]
                    arrf[tg] *= val
        N = [0] * NL
        xt = t / p.H
        if xt > 1.5:
            xt = 1.5
        for i in range(NL):
            base, amp, per, ph, drift, pon, poff, bx, nmax = p.arr[i]
            u = rng.random()
            burst = aux[i]
            if burst > 0.5:
                if u < poff:
                    burst = 0.0
            elif u < pon:
                burst = 1.0
            aux[i] = burst
            lam = base * (1.0 + amp * math.sin(_TWO_PI * (t + ph) / per)) * (1.0 + drift * xt) * arrf[i]
            if burst > 0.5:
                lam *= bx
            n = int(rng.poisson(lam)) if lam > 0.0 else 0
            N[i] = n if n < nmax else nmax

        # ---- 4. pool and capacity
        gain = p.ones_S[:]
        use = [w[s] * p.wmem[s] for s in range(S)]
        for s, h, kn, ov, gk, bm, bspan in p.bat:
            bb = b[s]
            if bb <= kn:
                gain[s] = (1.0 + h) * bb / (h + bb)
            else:
                x = (bb - kn) / kn
                gain[s] = gk / (1.0 + ov * x * x)
            if bm:
                use[s] *= 1.0 + bm * (bb - 1) / bspan
        ccost = None
        for s, sh, half, cmem, cc in p.cch:
            c = cz[s]
            gain[s] /= 1.0 - sh * c / (c + half)
            use[s] += c * cmem
            if ccost is None:
                ccost = [0.0] * S
            ccost[s] = c * cc
        util = 0.0
        phi = 1.0
        pool_cost = 0.0
        if p.pool is not None:
            size, th, kap, pw, pool_cost = p.pool
            util = sum(use) / size
            if util > th:
                phi = 1.0 / (1.0 + kap * ((util - th) / (1.0 - th)) ** pw)
        cap = [0.0] * S
        eff = [0.0] * S
        lag, mu, jit = p.lag, p.mu, p.jit
        for s in range(S):
            h = wh[s]
            ws = w[s]
            h.insert(0, ws)
            h.pop()
            lg = lag[s]
            ef = min(h[:lg + 1]) if lg else ws
            eff[s] = ef
            cap[s] = ef * mu[s] * gain[s] * phi * (1.0 + jit[s] * z[s])
        for s, x in p.capped:           # what the stage's dependency can take, whatever the workers
            if cap[s] > x:
                cap[s] = x
        if rate is not None:
            cap = [cap[s] * rate[s] for s in range(S)]
        capk = cap if K == 1 else [c / K for c in cap]

        # ---- 5. service rounds
        nxt, buf, line_of, stl, src = p.nxt, p.buf, p.line_of, p.stl, p.src
        qtot = [sum(row) for row in q]
        queued_before = sum(qtot)
        infl = [0] * NL
        hi_age = [0] * S
        top = A1 - 1
        if queued_before:
            for s in range(S):
                if qtot[s]:
                    infl[line_of[s]] += qtot[s]
                    qs = q[s]
                    a = top
                    while not qs[a]:
                        a -= 1
                    hi_age[s] = a
        hist = [[0] * A1 for _ in range(NL)]
        shed = [0] * NL
        over = [0] * NL
        d_fail = [0] * NL
        d_stale = [0] * NL
        admitted = [0] * NL
        att = [0] * S
        fails = [0] * S
        thr = [0] * S
        eprob = eb[:]
        for k in range(K):
            for i in range(NL):
                n = N[i] if K == 1 else N[i] // K + (1 if k < N[i] % K else 0)
                if n:
                    room = adm[i] - infl[i]
                    if room < n:
                        a = room if room > 0 else 0
                        shed[i] += n - a
                    else:
                        a = n
                    s0 = src[i]
                    sp = buf[s0] - qtot[s0]
                    if a > sp:
                        if sp < 0:
                            sp = 0
                        over[i] += a - sp
                        a = sp
                    if a:
                        q[s0][0] += a
                        qtot[s0] += a
                        infl[i] += a
                        admitted[i] += a
            zb = S + k * S
            for s in range(S):
                bs = b[s]
                cs = credit[s] + capk[s]
                x = capk[s] + bs
                if cs > x:
                    cs = x
                Q = qtot[s]
                if Q == 0:
                    credit[s] = cs
                    continue
                n = int(cs)
                if n > Q:
                    n = Q
                if bs > 1:
                    n -= n % bs
                nx = nxt[s]
                nn = len(nx)
                if nn == 1:
                    ja = nx[0]
                    spa = buf[ja] - qtot[ja]
                    if n > spa:
                        n = spa
                elif nn == 2:
                    ja, jb = nx
                    r = rho[s]
                    spa = buf[ja] - qtot[ja]
                    spb = buf[jb] - qtot[jb]
                    if r > 0.0:
                        m = int(spa / r)
                        if n > m:
                            n = m
                    if r < 1.0:
                        m = int(spb / (1.0 - r))
                        if n > m:
                            n = m
                if n <= 0:
                    credit[s] = cs
                    continue
                credit[s] = cs - n
                # failure probability of an attempt: base error and staleness
                e = eb[s]
                sl = stl[s]
                if sl is not None:
                    cp = cap[s]
                    x = ((Q / cp if cp > 1e-9 else 1e9) - sl[0]) / sl[1]
                    if x > 0.0:
                        e = 1.0 - (1.0 - e) * (1.0 - (sl[2] if x >= 1.0 else sl[2] * x))
                    eprob[s] = e
                F = 0
                Dn = 0
                if e > 0.0:
                    m = n * e
                    fr = m + math.sqrt(m * (1.0 - e)) * z[zb + s] + fcar[s]
                    if fr <= 0.0:
                        fcar[s] = 0.0
                    else:
                        F = int(fr)
                        if F > n:
                            F = n
                            fcar[s] = 0.0
                        else:
                            fcar[s] = fr - F
                    if F:
                        lim = rl[s]
                        if lim <= 0:
                            Dn = F
                        else:
                            if e >= 0.999999:
                                dp = 1.0 / (lim + 1.0)
                            else:
                                er = e ** lim
                                dp = er * (1.0 - e) / (1.0 - er * e)
                            dr = F * dp + dcar[s]
                            Dn = int(dr)
                            dcar[s] = dr - Dn
                            if Dn > F:
                                Dn = F
                        if Dn:
                            df = int(Dn * (eb[s] / e) + 0.5)
                            li = line_of[s]
                            d_fail[li] += df
                            d_stale[li] += Dn - df
                # oldest first; failures and give-ups are spread over the served cohorts in proportion
                qs = q[s]
                a = hi_age[s]
                if nn:
                    if hi_age[ja] < a:
                        hi_age[ja] = a
                    qa = q[ja]
                    if nn == 2:
                        if hi_age[jb] < a:
                            hi_age[jb] = a
                        qb = q[jb]
                        rc = rcar[s]
                        nat = 0
                else:
                    qa = hist[line_of[s]]
                rem = n
                okt = n - F
                if F:
                    cum = 0
                    cfp = 0
                    cdp = 0
                    while rem > 0 and a >= 0:
                        x = qs[a]
                        if x:
                            tk = x if x < rem else rem
                            rem -= tk
                            cum += tk
                            cf = F * cum // n
                            ok = tk - cf + cfp
                            if Dn:
                                cd = Dn * cf // F
                                qs[a] = x - ok - cd + cdp
                                cdp = cd
                            else:
                                qs[a] = x - ok
                            cfp = cf
                            if ok:
                                if nn < 2:
                                    qa[a] += ok
                                else:
                                    fr = ok * r + rc
                                    na = int(fr)
                                    rc = fr - na
                                    nb = ok - na
                                    if na > spa:
                                        nb += na - spa
                                        na = spa
                                    if nb > spb:
                                        na += nb - spb
                                        nb = spb
                                    spa -= na
                                    spb -= nb
                                    nat += na
                                    qa[a] += na
                                    qb[a] += nb
                        a -= 1
                elif nn < 2:
                    while rem > 0 and a >= 0:
                        x = qs[a]
                        if x:
                            if x <= rem:
                                rem -= x
                                qs[a] = 0
                                qa[a] += x
                            else:
                                qs[a] = x - rem
                                qa[a] += rem
                                rem = 0
                        a -= 1
                else:
                    while rem > 0 and a >= 0:
                        x = qs[a]
                        if x:
                            ok = x if x < rem else rem
                            rem -= ok
                            qs[a] = x - ok
                            fr = ok * r + rc
                            na = int(fr)
                            rc = fr - na
                            nb = ok - na
                            if na > spa:
                                nb += na - spa
                                na = spa
                            if nb > spb:
                                na += nb - spb
                                nb = spb
                            spa -= na
                            spb -= nb
                            nat += na
                            qa[a] += na
                            qb[a] += nb
                        a -= 1
                att[s] += n
                fails[s] += F
                thr[s] += okt
                qtot[s] -= okt + Dn
                if nn == 1:
                    qtot[ja] += okt
                    if Dn:
                        infl[line_of[s]] -= Dn
                elif nn == 0:
                    infl[line_of[s]] -= okt + Dn
                else:
                    rcar[s] = rc
                    qtot[ja] += nat
                    qtot[jb] += okt - nat
                    if Dn:
                        infl[line_of[s]] -= Dn

        # ---- 6. ageing and timeouts
        kills = [0] * NL
        ztail = p.zero_tail
        for s in range(S):
            if qtot[s]:
                li = line_of[s]
                T = ttl[li]
                if T < 0:
                    T = 0
                qs = q[s]
                if hi_age[s] >= T:
                    kl = sum(qs[T:])
                    if kl:
                        kills[li] += kl
                        qtot[s] -= kl
                nq = [0]
                nq.extend(qs[:T])
                nq.extend(ztail[A1 - 1 - T])
                q[s] = nq

        # ---- 7. accounting
        ns = p.n_seats
        revenue = [0.0] * ns
        cost = [0.0] * ns
        seat_of, wcost = p.seat_of, p.wcost
        if ns == 1:
            c = 0.0
            for s in range(S):
                c += w[s] * wcost[s]
            if pool_cost:
                c += sum(use) * pool_cost
            if ccost is not None:
                c += sum(ccost)
            cost[0] = c
        else:
            for s in range(S):
                c = w[s] * wcost[s] + use[s] * pool_cost
                if ccost is not None:
                    c += ccost[s]
                cost[seat_of[s]] += c
        lines_info = []
        tot_arr = tot_done = tot_late = tot_drop = tot_shed = 0
        value_arrived = value_done = 0.0
        lo_over = lo_time = lo_fail = lo_shed = 0.0
        vals = [0.0] * (p.v_pool + 1)
        for i in range(NL):
            h = hist[i]
            T = ttl[i]
            if T < top and hi_age[p.sink_line[i]] > T:
                # served, but the timeout fires before delivery: the work was done for nothing
                x = sum(h[T + 1:]) if T >= 0 else sum(h)
                if x:
                    kills[i] += x
                    h = h[:T + 1] + ztail[top - T] if T >= 0 else ztail[A1]
            D = p.deadline[i]
            o = sum(h[:D + 1])
            lt = 0
            a = D
            for a, fr in p.tail[i]:
                x = h[a]
                if x:
                    y = x * fr + aux[p.x_dl + i]
                    m = int(y)
                    aux[p.x_dl + i] = y - m
                    o += m
                    lt += x - m
            if a < top:
                lt += sum(h[a + 1:])
            v = p.value[i]
            if o + lt:
                l50 = _percentile(h, o + lt, 0.5) + p.settle[i]
                l95 = _percentile(h, o + lt, 0.95) + p.settle[i]
                aux[p.x_l50 + i] = l50
                aux[p.x_l95 + i] = l95
            else:
                l50 = aux[p.x_l50 + i]
                l95 = aux[p.x_l95 + i]
            rv = v * o
            if ns == 1:
                revenue[0] += rv
            else:
                sh = p.share[i]
                for j in range(ns):
                    revenue[j] += rv * sh[j]
            dropped = over[i] + d_fail[i] + d_stale[i] + kills[i]
            l_over = v * (over[i] + d_stale[i] + lt)
            l_time = v * kills[i]
            l_fail = v * d_fail[i]
            l_shed = v * shed[i]
            lo_over += l_over
            lo_time += l_time
            lo_fail += l_fail
            lo_shed += l_shed
            tot_arr += N[i]
            tot_done += o
            tot_late += lt
            tot_drop += dropped
            tot_shed += shed[i]
            value_arrived += v * N[i]
            value_done += rv
            vals[i] = o
            vals[NL + i] = lt
            vals[2 * NL + i] = l50
            vals[3 * NL + i] = l95
            vals[4 * NL + i] = shed[i] + over[i]
            vals[5 * NL + i] = kills[i]
            vals[6 * NL + i] = N[i]
            lines_info.append({"arrivals": N[i], "admitted": admitted[i], "completed": o, "late": lt,
                               "dropped": dropped, "shed": shed[i], "overflow": over[i], "timed_out": kills[i],
                               "failed": d_fail[i], "abandoned": d_stale[i], "latency_p50": l50, "latency_p95": l95,
                               "value_lost": l_over + l_time + l_fail + l_shed,
                               "lost_by_reason": {"overload": l_over, "timeout": l_time, "failure": l_fail,
                                                  "shed": l_shed}})
        obj = p.objective
        reserve = None
        if obj == "net":
            score = [revenue[j] - cost[j] for j in range(ns)]
        elif obj == "team_net":
            score = [(sum(revenue) - sum(cost)) / ns] * ns
        else:   # ontime_value: value delivered in time, counted while the running balance is not exhausted
            rcap = p.reserve[1]
            reserve = aux[p.x_res] + revenue[0] - cost[0]
            if reserve > rcap:
                reserve = rcap
            elif reserve < -rcap:
                reserve = -rcap
            aux[p.x_res] = reserve
            score = [value_done if reserve >= 0.0 else 0.0]

        # ---- 8. readings
        vals[p.v_queue:p.v_flow] = qtot
        vals[p.v_flow:p.v_err] = thr
        err_seat = [0.0] * ns
        if ns == 1:
            n = sum(att)
            if n:
                err_seat[0] = sum(fails) / n
        else:
            for j in range(ns):
                n = 0
                m = 0
                for s in p.seat_stages[j]:
                    n += att[s]
                    m += fails[s]
                if n:
                    err_seat[j] = m / n
        vals[p.v_err:p.v_pool] = err_seat
        vals[p.v_pool] = util
        readings: list[dict[str, float]] = [{} for _ in range(ns)]
        for seat, name, slot in p.exact:
            readings[seat][name] = float(vals[slot])
        for seat, name, slot, idx, sigma, vmax in p.noisy:
            v = vals[slot] + sigma * z[idx]
            readings[seat][name] = 0.0 if v < 0.0 else (vmax if v > vmax else v)

        info = {"t": t, "arrivals": tot_arr, "completed": tot_done, "late": tot_late, "dropped": tot_drop,
                "shed": tot_shed, "queued": sum(qtot), "queued_before": queued_before,
                "value_arrived": value_arrived, "value_completed": value_done,
                "value_lost": lo_over + lo_time + lo_fail + lo_shed,
                "lost_by_reason": dict(zip(LOSS_REASONS, (lo_over, lo_time, lo_fail, lo_shed))),
                "lines": lines_info, "pool_util": util, "slowdown": phi, "capacity": cap,
                "workers_effective": eff, "queue": qtot, "throughput": thr, "attempts": att, "failures": fails,
                "failure_prob": eprob, "error_rate": err_seat, "reserve": reserve}
        new_state = {"q": np.array(q, dtype=np.int64), "wh": np.array(wh, dtype=np.float64),
                     "car": np.array([credit, fcar, dcar, rcar], dtype=np.float64),
                     "aux": np.array(aux, dtype=np.float64)}
        return StepResult(new_state, readings, revenue, cost, score, info)

    # ---- private (evaluator only) ------------------------------------------------------------------
    def truth(self) -> dict:
        return _truth(self.config)


# ====================================================================== truth
def _truth(cfg: dict) -> dict:
    """The causal graph of a generated system.

    Conventions. An edge sign is the direction of the *direct* dependence with the other parents held fixed:
    +1 / -1 mean weakly monotone (possibly saturating or active only beyond a threshold, never of the opposite
    sign); 0 means non-monotone, or of a direction that depends on the state or the regime. Edges of sign 0 that
    start at a lever are the declared non-monotone lever responses (`kind` in knee, thrash, expiry, storm, split;
    a `thrash` edge is declared only for a lever that can push the pool over its threshold on its own). The total
    effect of a lever on a node is monotone only when every directed path between them has the same non-zero
    sign product and the node is not inside a feedback loop through a sign-0 edge (`implied_effects`); the tests
    check exactly those implied claims by simulation.
    """
    st = cfg["stages"]
    S = len(st)
    ns = int(cfg.get("n_seats", 1))
    pool = cfg.get("pool")
    obj = cfg["objective"]
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()
    rest = {(lv["kind"], tg): lv["default"] for lv in cfg["levers"] for tg in lv["targets"]}

    def can_thrash(lv: dict) -> bool:
        """Whether this lever alone, at its top with everything else at rest, slows the pool materially."""
        at = dict(rest)
        for tg in lv["targets"]:
            at[(lv["kind"], tg)] = lv["hi"]
        use = _pool_use(st, {i: at.get(("workers", i), 1.0) for i in range(S)},
                        {i: at.get(("batch", i), 1) for i in range(S)}, {i: at.get(("cache", i), 0.0) for i in range(S)})
        return pool_slowdown(pool, use / pool["size"]) < 0.85

    def node(nid, role, seat=None, public=None):
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "role": role, "seat": seat, "public": public})

    def edge(src, dst, sign, lag=0, kind="direct"):
        edges.append({"src": src, "dst": dst, "sign": int(sign), "lag": int(lag), "kind": kind})

    for i, s in enumerate(st):
        node(f"capacity.s{i}", "latent")
        node(f"backlog.s{i}", "latent")
        node(f"flow.s{i}", "latent")
    for li, ln in enumerate(cfg["lines"]):
        node(f"arrivals.l{li}", "exogenous")
        for q in ("admitted", "shed", "done", "late", "expired", "latency"):
            node(f"{q}.l{li}", "latent")
    if pool is not None:
        node("pool", "latent")
    for j in range(ns):
        node(f"errors.seat{j}", "latent")
        node(f"revenue.seat{j}", "latent")
        node(f"cost.seat{j}", "latent")
        node(f"score.seat{j}", "outcome", j)

    # structure of the flow
    for i, s in enumerate(st):
        li, seat = s["line"], s["seat"]
        edge(f"capacity.s{i}", f"flow.s{i}", +1, 0, "limit")
        edge(f"capacity.s{i}", f"backlog.s{i}", -1, 0, "drain")
        edge(f"backlog.s{i}", f"latency.l{li}", +1, 0, "wait")
        edge(f"backlog.s{i}", f"admitted.l{li}", -1, 0, "limit")
        edge(f"backlog.s{i}", f"shed.l{li}", +1, 0, "limit")
        edge(f"backlog.s{i}", f"expired.l{li}", +1, 1, "age")
        edge(f"expired.l{li}", f"backlog.s{i}", -1, 0, "discard")
        edge(f"expired.l{li}", f"flow.s{i}", -1, 1, "discard")
        for j in s["next"]:
            edge(f"flow.s{i}", f"flow.s{j}", +1, 0, "flow")
            edge(f"flow.s{i}", f"backlog.s{j}", +1, 0, "flow")
            edge(f"backlog.s{j}", f"flow.s{i}", -1, 0, "blocking")
            edge(f"backlog.s{j}", f"backlog.s{i}", +1, 0, "blocking")
        if not s["next"]:
            edge(f"flow.s{i}", f"done.l{li}", +1, 0, "flow")
            edge(f"flow.s{i}", f"late.l{li}", +1, 0, "flow")
        if s["stale"] is not None:
            edge(f"backlog.s{i}", f"flow.s{i}", 0, 0, "collapse")
            edge(f"backlog.s{i}", f"errors.seat{seat}", +1, 0, "stale")
        if pool is not None:
            edge("pool", f"capacity.s{i}", -1, 0, "threshold")
    for li, ln in enumerate(cfg["lines"]):
        s0 = ln["source"]
        edge(f"arrivals.l{li}", f"admitted.l{li}", +1, 0, "flow")
        edge(f"arrivals.l{li}", f"shed.l{li}", +1, 0, "flow")
        edge(f"admitted.l{li}", f"backlog.s{s0}", +1, 0, "flow")
        edge(f"admitted.l{li}", f"flow.s{s0}", +1, 0, "flow")
        edge(f"latency.l{li}", f"done.l{li}", -1, 0, "deadline")
        edge(f"latency.l{li}", f"late.l{li}", +1, 0, "deadline")
        for j in range(ns):
            if ln["share"][j] > 0:
                edge(f"done.l{li}", f"revenue.seat{j}", +1, 0, "payment")
    for j in range(ns):
        if obj == "net":
            edge(f"revenue.seat{j}", f"score.seat{j}", +1, 0, "money")
            edge(f"cost.seat{j}", f"score.seat{j}", -1, 0, "money")
        elif obj == "team_net":
            for k in range(ns):
                edge(f"revenue.seat{k}", f"score.seat{j}", +1, 0, "money")
                edge(f"cost.seat{k}", f"score.seat{j}", -1, 0, "money")
        else:
            for li, ln in enumerate(cfg["lines"]):
                edge(f"done.l{li}", f"score.seat{j}", +1, 0, "value")
            edge(f"revenue.seat{j}", f"score.seat{j}", +1, 0, "reserve")
            edge(f"cost.seat{j}", f"score.seat{j}", -1, 0, "reserve")

    # levers
    noop = 0
    for lv in cfg["levers"]:
        kind, seat, tg = lv["kind"], lv["seat"], lv["targets"]
        if kind == "noop":
            node(f"lever.noop.{noop}", "lever", seat, lv["name"])
            noop += 1
            continue
        tag = ("l" if kind in ("timeout", "admit") else "s") + "+".join(str(j) for j in tg)
        nid = f"lever.{kind}.{tag}"
        node(nid, "lever", seat, lv["name"])
        if kind == "workers":
            for i in tg:
                edge(nid, f"capacity.s{i}", +1, st[i]["lag"], "delayed")
                if pool is not None:
                    edge(nid, "pool", +1, 0, "draw")
                    if can_thrash(lv):
                        edge(nid, f"done.l{st[i]['line']}", 0, st[i]["lag"], "thrash")
                edge(nid, f"cost.seat{st[i]['seat']}", +1, 0, "cost")
        elif kind == "batch":
            for i in tg:
                edge(nid, f"capacity.s{i}", 0, 0, "knee")
                edge(nid, f"backlog.s{i}", +1, 0, "batch-wait")
                if pool is not None and st[i]["batch"]["mem"] > 0:
                    edge(nid, "pool", +1, 0, "draw")
                    edge(nid, f"cost.seat{st[i]['seat']}", +1, 0, "cost")
        elif kind == "cache":
            for i in tg:
                edge(nid, f"capacity.s{i}", +1, 0, "saturating")
                edge(nid, f"cost.seat{st[i]['seat']}", +1, 0, "cost")
                if pool is not None:
                    edge(nid, "pool", +1, 0, "draw")
                    if can_thrash(lv):
                        edge(nid, f"done.l{st[i]['line']}", 0, 0, "thrash")
        elif kind == "retry":
            for i in tg:
                # repeating failed attempts recovers work; where attempts go stale it can also start a storm
                if st[i]["stale"] is not None:
                    edge(nid, f"flow.s{i}", 0, 0, "storm")
                else:
                    edge(nid, f"flow.s{i}", +1, 0, "repeat")
                edge(nid, f"backlog.s{i}", +1, 0, "repeat")
        elif kind == "timeout":
            for li in tg:
                edge(nid, f"done.l{li}", 0, 0, "expiry")
                edge(nid, f"expired.l{li}", -1, 1, "age")
                edge(nid, f"late.l{li}", +1, 0, "age")
                for i, s in enumerate(st):
                    if s["line"] == li:
                        edge(nid, f"backlog.s{i}", +1, 1, "age")
        elif kind == "admit":
            for li in tg:
                edge(nid, f"admitted.l{li}", +1, 0, "limit")
                edge(nid, f"shed.l{li}", -1, 0, "limit")
        elif kind == "route":
            for i in tg:
                ja, jb = st[i]["next"]
                edge(nid, f"backlog.s{ja}", +1, 0, "split")
                edge(nid, f"flow.s{ja}", +1, 0, "split")
                edge(nid, f"backlog.s{jb}", -1, 0, "split")
                edge(nid, f"flow.s{jb}", -1, 0, "split")
                edge(nid, f"done.l{st[i]['line']}", 0, 0, "split")
    if pool is not None and pool["cost"] > 0:
        for j in range(ns):
            edge("pool", f"cost.seat{j}", +1, 0, "cost")

    # dependencies and regimes
    regimes = []
    for r in cfg.get("regimes", []):
        what, tg = r["what"], r["target"]
        if what == "arrival":
            regimes.append({"at": int(r["at"]), "what": f"arrivals of line {tg} x{r['value']}"})
            if r.get("until") is not None:
                regimes.append({"at": int(r["until"]), "what": f"arrivals of line {tg} back to normal"})
            continue
        nid = f"dependency.s{tg}"
        node(nid, "exogenous")
        if what == "rate":
            edge(nid, f"capacity.s{tg}", +1, 0, "regime")
            regimes.append({"at": int(r["at"]), "what": f"dependency of stage {tg} degrades: service rate x{r['value']}"})
        else:
            edge(nid, f"flow.s{tg}", +1, 0, "regime")
            edge(nid, f"errors.seat{st[tg]['seat']}", -1, 0, "regime")
            regimes.append({"at": int(r["at"]), "what": f"dependency of stage {tg} degrades: error rate {r['value']}"})
        if r.get("until") is not None:
            regimes.append({"at": int(r["until"]), "what": f"dependency of stage {tg} recovers"})
    regimes.sort(key=lambda r: (r["at"], r["what"]))

    # gauges
    quantity = {"done": "done.l{}", "late": "late.l{}", "lat50": "latency.l{}", "lat95": "latency.l{}",
                "queue": "backlog.s{}", "shed": "shed.l{}", "tout": "expired.l{}", "offered": "arrivals.l{}",
                "handoff": "flow.s{}"}
    for g in cfg["gauges"]:
        nid = f"gauge.{g['name']}"
        node(nid, "gauge", g["seat"], g["name"])
        if g["kind"] == "pool":
            edge("pool", nid, +1, 0, "reading")
        elif g["kind"] == "err":
            edge(f"errors.seat{g['seat']}", nid, +1, 0, "reading")
        else:
            edge(quantity[g["kind"]].format(g["targets"][0]), nid, +1, 0, "reading")
    return {"engine": "techsys", "family": cfg.get("family"), "nodes": nodes, "edges": edges, "regimes": regimes,
            "motifs": list(cfg.get("motifs", [])),
            "notes": ("Queueing model over integer job counts. Edge signs are direct, weakly monotone dependences; "
                      "sign 0 is non-monotone or state-dependent. Sign-0 edges that start at a lever are the "
                      "declared non-monotone responses. A worker counts after `lag` consecutive ticks of being "
                      "requested (the lag of the lever->capacity edge). Stage roles: "
                      + ", ".join(f"s{i}={s['role']}" for i, s in enumerate(st)) + ".")}


def implied_effects(truth: dict, max_steps: int = 2_000_000) -> list[tuple[str, str, int]]:
    """The monotone total effects a truth graph implies: (lever id, node id, sign) for every pair such that
    (a) every simple directed path from the lever to the node has the same non-zero sign product, and (b) the
    node does not lie on a directed cycle that contains a sign-0 edge (feedback through a non-monotone link can
    reverse an effect; feedback through monotone links only attenuates or amplifies it). Gauges and exogenous
    nodes are left out: a gauge is a reading of the quantity it hangs on. Evaluator and tests only."""
    adj: dict[str, list[tuple[str, int]]] = {}
    for e in truth["edges"]:
        adj.setdefault(e["src"], []).append((e["dst"], e["sign"]))
    role = {n["id"]: n["role"] for n in truth["nodes"]}

    def reach(start: str) -> set[str]:
        seen, stack = set(), [start]
        while stack:
            u = stack.pop()
            for v, _ in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    reach_of = {n: reach(n) for n in role}
    unsafe: set[str] = set()
    for e in truth["edges"]:
        if e["sign"] == 0 and e["src"] in reach_of[e["dst"]] | {e["dst"]}:
            # nodes on a cycle through this edge: reachable from its head and reaching its tail
            for n in role:
                if (n == e["dst"] or n in reach_of[e["dst"]]) and (n == e["src"] or e["src"] in reach_of[n]):
                    unsafe.add(n)
    out = []
    for lever in sorted(n for n, r in role.items() if r == "lever"):
        signs: dict[str, set[int]] = {}
        steps = 0
        on_path = {lever}
        stack = [(lever, 1, iter(adj.get(lever, ())))]
        while stack:
            u, sg, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                on_path.discard(u)
                continue
            v, s = nxt
            if v in on_path:
                continue
            steps += 1
            if steps > max_steps:
                signs = {}
                break
            signs.setdefault(v, set()).add(sg * s)
            on_path.add(v)
            stack.append((v, sg * s, iter(adj.get(v, ()))))
        for node in sorted(signs):
            sg = signs[node]
            if len(sg) == 1 and 0 not in sg and role[node] in ("latent", "outcome") and node not in unsafe:
                out.append((lever, node, next(iter(sg))))
    return out


def node_value(res: StepResult, node_id: str) -> float:
    """The value of a truth node in one step result (evaluator and tests only; gauges are read noise-free from
    the underlying quantity)."""
    info = res.info
    head, _, tail = node_id.partition(".")
    if node_id == "pool":
        return float(info["pool_util"])
    idx = int("".join(ch for ch in tail if ch.isdigit()) or 0)
    if head == "capacity":
        return float(info["capacity"][idx])
    if head == "backlog":
        return float(info["queue"][idx])
    if head == "flow":
        return float(info["throughput"][idx])
    if head in ("revenue", "cost", "score"):
        return float({"revenue": res.revenue, "cost": res.cost, "score": res.score}[head][idx])
    if head == "errors":
        return float(info["error_rate"][idx])
    ln = info["lines"][idx]
    if head == "shed":
        return float(ln["shed"] + ln["overflow"])
    key = {"arrivals": "arrivals", "admitted": "admitted", "done": "completed", "late": "late",
           "expired": "timed_out", "latency": "latency_p50"}[head]
    return float(ln[key])


# ====================================================================== generation
def _r(x, nd: int = 4) -> float:
    return float(round(float(x), nd))


def _u(rng, lo, hi, nd: int = 4) -> float:
    return _r(rng.uniform(lo, hi), nd)


def _i(rng, lo, hi) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _rng_range(p: dict, key: str, default):
    v = p.get(key, default)
    if isinstance(v, (int, float)):
        return (v, v)
    return (v[0], v[1])


def _common(rng, p: dict) -> dict:
    return {"lam": _u(rng, *_rng_range(p, "arrivals", (30.0, 54.0))), "value": _u(rng, 6.0, 14.0, 2),
            "deadline": _i(rng, *_rng_range(p, "deadline", (2, 4))), "ages": int(p.get("ages", 10)),
            "horizon": int(p.get("horizon", 144)), "substeps": int(p.get("substeps", 1)),
            "tail": [_u(rng, 0.45, 0.7), _u(rng, 0.1, 0.3)], "settle": _i(rng, *_rng_range(p, "settle", (2, 3)))}


def _ttl(c: dict, slack: int) -> int:
    """A resting timeout `slack` ticks above the deadline (both in latency terms)."""
    return min(c["ages"], c["deadline"] + c["settle"] + slack)


def _arrival(rng, p: dict, base: float, amp=(0.25, 0.45), burst_x=(1.25, 1.6), burst_on=(0.015, 0.04),
             drift=(-0.1, 0.15)) -> dict:
    """Arrival intensity: a daily cycle, a slow drift over the horizon, and bursts that switch on and off at
    random (a two-state chain)."""
    period = int(p.get("period", 24))
    return {"base": _r(base), "amp": _u(rng, *_rng_range(p, "amp", amp)), "period": period,
            "phase": _u(rng, 0.0, float(period)), "drift": _u(rng, *_rng_range(p, "drift", drift)),
            "burst_on": _u(rng, *_rng_range(p, "burst_on", burst_on)), "burst_off": _u(rng, 0.25, 0.5),
            "burst_x": _u(rng, *_rng_range(p, "burst_x", burst_x))}


def _line(c: dict, source: int, sink: int, arrival: dict, share: list[float]) -> dict:
    return {"source": source, "sink": sink, "deadline": c["deadline"] + c["settle"], "settle": c["settle"],
            "tail": list(c["tail"]), "value": c["value"], "share": share, "arrival": arrival}


def _stage(rng, role: str, line: int, seat: int, nxt: list[int], lam: float, lag, buf_ticks, n_ref=(6, 11)) -> dict:
    """A stage sized so that `n_ref` workers at batch 1, no cache, serve `lam` jobs per tick. Its buffer holds
    `buf_ticks` ticks of mean arrivals (with one service round per tick a buffer caps the flow per tick, so no
    buffer is smaller than two ticks of flow)."""
    n = _i(rng, *n_ref)
    return {"role": role, "line": line, "seat": seat, "next": list(nxt), "mu": _r(lam / n),
            "lag": _i(rng, *lag), "jitter": _u(rng, 0.02, 0.05), "buf": int(round(_u(rng, *buf_ticks) * lam)),
            "wmem": _u(rng, 0.8, 1.25), "wcost": _r(_u(rng, 0.7, 1.3) / n, 5), "batch": None, "cache": None,
            "err": 0.0, "retry": 0, "stale": None, "cap": None}


def _batch(rng, mem=(0.3, 0.8)) -> dict:
    knee = _i(rng, 5, 12)
    return {"setup": _u(rng, 1.5, 4.0), "knee": knee, "over": _u(rng, 0.8, 2.0), "mem": _u(rng, *mem),
            "max": 4 * knee}


def _cache(rng, hmax=(0.6, 0.9), save=(0.6, 0.9)) -> dict:
    half = _u(rng, 4.0, 10.0, 2)
    return {"hmax": _u(rng, *hmax), "half": half, "save": _u(rng, *save), "mem": 0.0, "cost": _r(0.3 / half, 5),
            "max": int(round(8 * half))}


def _rate_at(stage: dict, b: float = 1, c: float = 0.0) -> float:
    return stage["mu"] * batch_gain(stage["batch"], b) * cache_gain(stage["cache"], c)


def _workers_for(stage: dict, demand: float, b: float = 1, c: float = 0.0) -> float:
    return demand / _rate_at(stage, b, c)


def _pool_use(stages: list[dict], w: dict, b: dict, c: dict) -> float:
    tot = 0.0
    for i, s in enumerate(stages):
        us = w.get(i, 1.0) * s["wmem"]
        if s["batch"] is not None:
            us *= 1.0 + s["batch"]["mem"] * (b.get(i, 1) - 1) / max(1, s["batch"]["max"] - 1)
        if s["cache"] is not None:
            us += c.get(i, 0.0) * s["cache"]["mem"]
        tot += us
    return tot


def _lever(kind: str, targets: list[int], seat: int, lo, hi, default, integer: bool = True) -> dict:
    return {"kind": kind, "targets": list(targets), "seat": seat, "lo": float(lo), "hi": float(hi),
            "default": float(min(hi, max(lo, default))), "integer": bool(integer)}


def _worker_lever(stage_idx: int, stage: dict, default: float, peak_need: float, hi: int | None = None) -> dict:
    d = max(1, int(round(default)))
    if hi is None:
        hi = max(d + 4, int(math.ceil(2.6 * peak_need)))
    return _lever("workers", [stage_idx], stage["seat"], 1, max(hi, d), d)


def _line_levers(li: int, line: dict, stages: list[dict], seat_admit: int, seat_timeout: int, ages: int,
                 ttl_default: int, limit_default: int | None = None) -> list[dict]:
    """Admission limit and timeout of a line. The limit's upper end exceeds what the buffers can hold, so at
    the top of its range it never binds."""
    hi = int(round(1.15 * sum(s["buf"] for s in stages if s["line"] == li)))
    line["limit"] = hi
    line["ttl"] = int(ttl_default)
    return [_lever("admit", [li], seat_admit, 0, hi, hi if limit_default is None else limit_default),
            _lever("timeout", [li], seat_timeout, 1, ages, ttl_default)]


def _resting_limit(rng, c: dict, lam: float) -> int:
    """A resting admission limit that is neither absent nor right: between a little more than one tick of mean
    arrivals (sheds at every peak) and enough backlog to make the last admitted jobs late (`deadline` here is in
    queue-age terms, as everywhere in the scenario builders)."""
    return int(round(_u(rng, 1.3, c["deadline"] + 1.5) * lam))


def _noop_lever(rng, seat: int = 0) -> dict:
    if rng.random() < 0.5:
        hi = _i(rng, 8, 40)
        return _lever("noop", [], seat, 0, hi, _i(rng, 0, hi))
    return _lever("noop", [], seat, 0.0, 1.0, _u(rng, 0.0, 1.0), integer=False)


def _standard_gauges(stages: list[dict], lines: list[dict], pool, n_seats: int, hide) -> list[dict]:
    """Every gauge the engine can show, each given to the seat that owns the thing it measures."""
    gs = []
    for li, ln in enumerate(lines):
        s_in, s_out = stages[ln["source"]]["seat"], stages[ln["sink"]]["seat"]
        for kind in ("done", "late", "lat50", "lat95", "tout"):
            gs.append({"kind": kind, "targets": [li], "seat": s_out})
        for kind in ("shed", "offered"):
            gs.append({"kind": kind, "targets": [li], "seat": s_in})
    for i, s in enumerate(stages):
        gs.append({"kind": "queue", "targets": [i], "seat": s["seat"]})
    for j in range(n_seats):
        gs.append({"kind": "err", "targets": [], "seat": j})
        if pool is not None:
            gs.append({"kind": "pool", "targets": [], "seat": j})
    hide = set(hide)
    return [g for g in gs if g["kind"] not in hide]


_ROLES = {2: ["work", "store"], 3: ["intake", "transform", "store"], 4: ["intake", "parse", "index", "store"],
          5: ["intake", "parse", "enrich", "index", "store"]}


def _linear(rng, n: int, lam: float, lag, line: int = 0, seat: int = 0, offset: int = 0, roles=None) -> list[dict]:
    roles = roles or _ROLES[n]
    out = []
    for k, role in enumerate(roles):
        nxt = [offset + k + 1] if k < n - 1 else []
        out.append(_stage(rng, role, line, seat, nxt, lam, lag, (6.0, 8.0) if k == 0 else (2.5, 4.0)))
    return out


def _fork(rng, lam: float, lag, with_intake: bool) -> tuple[list[dict], int]:
    """[intake ->] split -> (path_a | path_b) -> store. Returns the stages and the index of the split."""
    stages = []
    k = 0
    if with_intake:
        stages.append(_stage(rng, "intake", 0, 0, [1], lam, lag, (6.0, 8.0)))
        k = 1
    stages.append(_stage(rng, "split", 0, 0, [k + 1, k + 2], lam, lag, (6.0, 8.0) if k == 0 else (2.5, 4.0)))
    stages.append(_stage(rng, "path_a", 0, 0, [k + 3], lam, lag, (2.0, 3.0)))
    stages.append(_stage(rng, "path_b", 0, 0, [k + 3], lam, lag, (2.0, 3.0)))
    stages.append(_stage(rng, "store", 0, 0, [], lam, lag, (2.5, 4.0)))
    return stages, k


# ---------------------------------------------------------------------- scenarios
def _scn_pipeline(rng, p: dict) -> dict:
    """Linear pipeline; batching knee on one or two stages; slow worker start-up; one stage short of capacity at
    the daily peak while the others are over-staffed."""
    c = _common(rng, p)
    n = int(p.get("stages", _i(rng, 3, 4)))
    lam = c["lam"]
    stages = _linear(rng, n, lam, _rng_range(p, "lag", (2, 6)))
    line = _line(c, 0, n - 1, _arrival(rng, p, lam), [1.0])
    amp = line["arrival"]["amp"]
    nb = int(p.get("batch_stages", _i(rng, 1, 2)))
    bstages = sorted(int(x) for x in rng.choice(n, size=nb, replace=False))
    weak = int(rng.integers(n))
    levers = []
    for i, s in enumerate(stages):
        s["err"] = _u(rng, 0.0, 0.02)
        s["retry"] = 1
        b0 = 1
        if i in bstages:
            s["batch"] = _batch(rng, mem=(0.0, 0.0))
            if rng.random() < 0.3:
                b0 = int(round(s["batch"]["knee"] * _u(rng, 1.8, 2.4)))     # resting beyond the knee
            levers.append(_lever("batch", [i], 0, 1, s["batch"]["max"], b0))
        need = _workers_for(s, lam, b0)
        f = _u(rng, 1.02, 1.15) if i == weak else (1.0 + amp) * _u(rng, 1.25, 1.6)
        levers.append(_worker_lever(i, s, f * need, (1.0 + amp) * _workers_for(s, lam)))
    levers += _line_levers(0, line, stages, 0, 0, c["ages"], _ttl(c, _i(rng, 1, 2)),
                           _resting_limit(rng, c, lam))
    return {"common": c, "stages": stages, "lines": [line], "pool": None, "regimes": [], "levers": levers,
            "gauges": _standard_gauges(stages, [line], None, 1, p.get("hide", [])), "objective": "net",
            "motifs": ["queue", "deadline", "batch_knee", "startup_lag", "bottleneck"],
            "calibrate_loss": _rng_range(p, "default_loss", (0.12, 0.25)), "n_seats": 1, "design": {"weak": weak}}


def _scn_contention(rng, p: dict) -> dict:
    """Linear pipeline whose stages share one pool that thrashes above a hidden utilisation. At rest the pool is
    just below the threshold: one stage hoards it (too many workers, an oversized cache), another is short of
    capacity, and giving the short stage more of anything pushes the pool over the edge."""
    c = _common(rng, p)
    n = int(p.get("stages", 4))
    lam = c["lam"]
    stages = _linear(rng, n, lam, _rng_range(p, "lag", (1, 3)))
    line = _line(c, 0, n - 1, _arrival(rng, p, lam), [1.0])
    amp = line["arrival"]["amp"]
    order = [int(x) for x in rng.permutation(n)]
    weak, hog, bst = order[0], order[1], order[2]
    w0: dict = {}
    b0: dict = {}
    c0: dict = {}
    levers = []
    for i, s in enumerate(stages):
        s["err"] = _u(rng, 0.0, 0.02)
        s["retry"] = 1
        if i in (weak, hog):
            s["cache"] = _cache(rng)
            s["cache"]["mem"] = _u(rng, 0.15, 0.3)
            c0[i] = int(round(s["cache"]["half"] * (_u(rng, 0.0, 0.4) if i == weak else _u(rng, 3.0, 5.0))))
            levers.append(_lever("cache", [i], 0, 0, s["cache"]["max"], c0[i]))
        if i == bst:
            s["batch"] = _batch(rng)
            b0[i] = _i(rng, 1, 2)
            levers.append(_lever("batch", [i], 0, 1, s["batch"]["max"], b0[i]))
        need = _workers_for(s, lam, b0.get(i, 1), c0.get(i, 0.0))
        f = _u(rng, 1.02, 1.15) if i == weak else (1.0 + amp) * (_u(rng, 1.5, 2.0) if i == hog else _u(rng, 1.2, 1.5))
        lv = _worker_lever(i, s, f * need, (1.0 + amp) * _workers_for(s, lam))
        w0[i] = lv["default"]
        levers.append(lv)
    theta = _u(rng, 0.7, 0.82)
    use0 = _pool_use(stages, w0, b0, c0)
    pool = {"size": _r(use0 / (theta - _u(rng, 0.02, 0.05))), "theta": theta, "kappa": _u(rng, 8.0, 14.0),
            "power": _u(rng, 1.0, 1.3),
            "cost": _r(_u(rng, 0.1, 0.2) * sum(w0[i] * s["wcost"] for i, s in enumerate(stages)) / use0, 5)}
    levers += _line_levers(0, line, stages, 0, 0, c["ages"], _ttl(c, _i(rng, 1, 2)),
                           _resting_limit(rng, c, lam))
    levers += [_noop_lever(rng) for _ in range(int(p.get("distractors", 1)))]
    return {"common": c, "stages": stages, "lines": [line], "pool": pool, "regimes": [], "levers": levers,
            "gauges": _standard_gauges(stages, [line], pool, 1, p.get("hide", ["late", "offered"])),
            "objective": "net",
            "motifs": ["queue", "deadline", "shared_pool", "thrash_threshold", "cache", "batch_knee", "bottleneck",
                       "distractor"],
            "calibrate_loss": _rng_range(p, "default_loss", (0.12, 0.25)), "n_seats": 1,
            "design": {"weak": weak, "hog": hog}}


def _scn_cascade(rng, p: dict) -> dict:
    """A pipeline with a stage that calls a flaky dependency and repeats failed attempts. At rest it is healthy
    with a generous retry limit. A scheduled, temporary slowdown of the dependency builds a backlog; from then on
    attempts go stale while they wait, every job is attempted again and again, and the stage stays collapsed
    after the dependency has recovered. The dependency has a ceiling, so more workers cannot clear the storm."""
    c = _common(rng, p)
    n = int(p.get("stages", _i(rng, 3, 4)))
    lam = c["lam"]
    roles = ["intake", "enrich", "store"] if n == 3 else ["intake", "parse", "enrich", "store"]
    stages = _linear(rng, n, lam, _rng_range(p, "lag", (1, 4)), roles=roles)
    dep = roles.index("enrich")
    line = _line(c, 0, n - 1, _arrival(rng, p, lam, amp=(0.12, 0.22), burst_x=(1.1, 1.2), burst_on=(0.01, 0.025)),
                 [1.0])
    amp = line["arrival"]["amp"]
    bst = int(rng.choice([i for i in range(n) if i != dep]))
    levers = []
    stages[dep]["err"] = _u(rng, 0.07, 0.12)
    stages[dep]["stale"] = {"tau": _u(rng, 1.3, 1.6), "ramp": _u(rng, 0.3, 0.5), "max": _u(rng, 0.93, 0.97)}
    stages[dep]["buf"] = int(round(_u(rng, 5.0, 7.0) * lam))       # calls pile up in front of the dependency
    r0 = _i(rng, 5, 7)
    levers.append(_lever("retry", [dep], 0, 0, 8, r0))
    for i, s in enumerate(stages):
        b0 = 1
        if i != dep:
            s["err"] = _u(rng, 0.0, 0.01)
            s["retry"] = 1
        if i == bst:
            s["batch"] = _batch(rng, mem=(0.0, 0.0))
            b0 = _i(rng, 1, 3)
            levers.append(_lever("batch", [i], 0, 1, s["batch"]["max"], b0))
        demand = lam / (1.0 - s["err"])
        lv = _worker_lever(i, s, (1.0 + amp) * _u(rng, 1.3, 1.5) * _workers_for(s, demand, b0),
                           (1.0 + amp) * _workers_for(s, demand))
        if i == dep:        # the dependency takes little more than the resting workers already send it
            s["cap"] = _r(lv["default"] * s["mu"] * _u(rng, 1.0, 1.15))
        levers.append(lv)
    H = c["horizon"]
    at = int(_u(rng, 0.5, 0.65) * H)
    regimes = [{"at": at, "until": at + _i(rng, 8, 14), "what": "rate", "target": dep, "value": _u(rng, 0.25, 0.4)}]
    levers += _line_levers(0, line, stages, 0, 0, c["ages"], _ttl(c, _i(rng, 3, 4)))
    levers += [_noop_lever(rng) for _ in range(int(p.get("distractors", 1)))]
    return {"common": c, "stages": stages, "lines": [line], "pool": None, "regimes": regimes, "levers": levers,
            "gauges": _standard_gauges(stages, [line], None, 1, p.get("hide", ["late"])), "objective": "net",
            "motifs": ["queue", "deadline", "retry_storm", "metastable_collapse", "regime_change", "batch_knee",
                       "distractor"],
            "calibrate_loss": None, "n_seats": 1, "design": {"dependency": dep}}


def _fork_scenario(rng, p: dict, c: dict, with_intake: bool, err_main, err_alt, alt_price, share_main):
    lam = c["lam"]
    stages, k = _fork(rng, lam, _rng_range(p, "lag", (1, 4)), with_intake)
    main = k + 1 + int(rng.integers(2))          # the cheaper, less reliable path (which successor is seeded)
    alt = 2 * k + 3 - main
    stages[main]["err"] = _u(rng, *err_main)
    stages[alt]["err"] = _u(rng, *err_alt)
    stages[alt]["wcost"] = _r(stages[alt]["wcost"] * _u(rng, *alt_price), 5)
    sm = _u(rng, *share_main)
    rho0 = sm if main == k + 1 else _r(1.0 - sm)
    line = _line(c, 0, len(stages) - 1, _arrival(rng, p, lam), [1.0])
    return stages, line, k, main, alt, sm, rho0


def _scn_redundant(rng, p: dict) -> dict:
    """Two alternative paths between a split and a join: a cheap one that fails more often and whose dependency
    degrades for good in mid-episode, and a dearer reliable one. At rest most traffic takes the cheap path."""
    c = _common(rng, p)
    lam = c["lam"]
    stages, line, k, main, alt, sm, rho0 = _fork_scenario(
        rng, p, c, bool(p.get("intake", rng.random() < 0.5)), (0.05, 0.12), (0.005, 0.02), (2.5, 3.5), (0.7, 0.85))
    amp = line["arrival"]["amp"]
    r0 = 3
    levers = [_lever("retry", [k + 1, k + 2], 0, 0, 6, r0),
              _lever("route", [k], 0, 0.0, 1.0, rho0, integer=False)]
    for i, s in enumerate(stages):
        share = sm if i == main else ((1.0 - sm) if i == alt else 1.0)
        if i not in (main, alt):
            s["err"] = _u(rng, 0.0, 0.01)
            s["retry"] = 1
        demand = lam * share / (1.0 - s["err"])
        f = (1.0 + amp) * (_u(rng, 1.1, 1.3) if i in (main, alt) else _u(rng, 1.2, 1.5))
        levers.append(_worker_lever(i, s, f * _workers_for(s, demand), (1.0 + amp) * _workers_for(s, lam)))
    at = int(_u(rng, 0.4, 0.6) * c["horizon"])
    regimes = [{"at": at, "until": None, "what": "error", "target": main, "value": _u(rng, 0.6, 0.75)},
               {"at": at, "until": None, "what": "rate", "target": main, "value": _u(rng, 0.5, 0.7)}]
    levers += _line_levers(0, line, stages, 0, 0, c["ages"], _ttl(c, _i(rng, 1, 2)))
    return {"common": c, "stages": stages, "lines": [line], "pool": None, "regimes": regimes, "levers": levers,
            "gauges": _standard_gauges(stages, [line], None, 1, p.get("hide", ["late", "offered"])),
            "objective": "net",
            "motifs": ["queue", "deadline", "redundant_paths", "routing", "unequal_reliability", "regime_change",
                       "retry"],
            "calibrate_loss": None, "n_seats": 1, "design": {"split": k, "main": main, "alt": alt}}


def _scn_lostwork(rng, p: dict) -> dict:
    """The redundant layout scored on the value of work delivered in time, subject to paying its way. At rest
    most traffic takes a path that fails often and is short of capacity, with batching and caching unused, a
    stingy retry limit, no load shedding and a long timeout: several independent ways to lose less work."""
    c = _common(rng, p)
    lam = c["lam"]
    stages, line, k, main, alt, sm, rho0 = _fork_scenario(
        rng, p, c, True, (0.15, 0.25), (0.01, 0.03), (1.3, 1.7), (0.7, 0.8))
    amp = line["arrival"]["amp"]
    r0 = 1
    levers = [_lever("retry", [k + 1, k + 2], 0, 0, 6, r0),
              _lever("route", [k], 0, 0.0, 1.0, rho0, integer=False)]
    stages[main]["batch"] = _batch(rng, mem=(0.2, 0.5))
    stages[main]["cache"] = _cache(rng)
    levers.append(_lever("batch", [main], 0, 1, stages[main]["batch"]["max"], 1))
    levers.append(_lever("cache", [main], 0, 0, stages[main]["cache"]["max"], 0))
    w0: dict = {}
    for i, s in enumerate(stages):
        if i not in (main, alt):
            s["err"] = _u(rng, 0.0, 0.01)
            s["retry"] = 1
        if i == main:
            d = _u(rng, 0.95, 1.05) * _workers_for(s, lam * sm * (1.0 + s["err"]))
        elif i == alt:      # staffed for about half the traffic although it receives much less
            d = (1.0 + amp) * _u(rng, 1.0, 1.15) * _workers_for(s, lam * 0.5)
        else:
            d = (1.0 + amp) * _u(rng, 1.2, 1.5) * _workers_for(s, lam)
        lv = _worker_lever(i, s, d, (1.0 + amp) * _workers_for(s, lam))
        w0[i] = lv["default"]
        levers.append(lv)
    use_w = _pool_use(stages, w0, {}, {})
    stages[main]["cache"]["mem"] = _r(_u(rng, 0.25, 0.35) * use_w / stages[main]["cache"]["max"], 5)
    pool = {"size": _r(use_w / _u(rng, 0.45, 0.55)), "theta": _u(rng, 0.75, 0.85), "kappa": _u(rng, 3.0, 6.0),
            "power": _u(rng, 1.0, 1.6),
            "cost": _r(_u(rng, 0.08, 0.15) * sum(w0[i] * s["wcost"] for i, s in enumerate(stages)) / use_w, 5)}
    levers += _line_levers(0, line, stages, 0, 0, c["ages"], _ttl(c, 2))
    return {"common": c, "stages": stages, "lines": [line], "pool": pool, "regimes": [], "levers": levers,
            "gauges": _standard_gauges(stages, [line], pool, 1, p.get("hide", [])), "objective": "ontime_value",
            "motifs": ["queue", "deadline", "redundant_paths", "routing", "unequal_reliability", "retry",
                       "batch_knee", "cache", "shared_pool", "bottleneck", "sustainability_constraint"],
            "calibrate_loss": _rng_range(p, "default_loss", (0.4, 0.55)), "n_seats": 1,
            "design": {"split": k, "main": main, "alt": alt, "weak": main}}


def _scn_handoff(rng, p: dict) -> dict:
    """One pipeline operated by two seats: seat 0 the upstream half, seat 1 the downstream half, with a small
    buffer between them. The customers pay for what leaves the last stage; both seats receive the same score."""
    c = _common(rng, p)
    lam = c["lam"]
    stages = _linear(rng, 4, lam, _rng_range(p, "lag", (1, 4)), roles=["intake", "prepare", "process", "store"])
    for i, s in enumerate(stages):
        s["seat"] = 0 if i < 2 else 1
    stages[2]["buf"] = int(round(_u(rng, 1.5, 1.9) * lam))      # the hand-off buffer is the tight one
    line = _line(c, 0, 3, _arrival(rng, p, lam), [0.5, 0.5])
    amp = line["arrival"]["amp"]
    bst = [int(rng.integers(0, 2)), int(rng.integers(2, 4))]
    lean = int(rng.integers(2))                  # the half that is short of capacity at rest
    weak = int(rng.integers(2)) + 2 * lean
    levers = []
    for i, s in enumerate(stages):
        s["err"] = _u(rng, 0.0, 0.02)
        s["retry"] = 1
        b0 = 1
        if i in bst:
            s["batch"] = _batch(rng, mem=(0.0, 0.0))
            b0 = _i(rng, 1, 2)
            levers.append(_lever("batch", [i], s["seat"], 1, s["batch"]["max"], b0))
        need = _workers_for(s, lam, b0)
        if i == weak:
            f = _u(rng, 1.02, 1.15)
        elif s["seat"] == lean:
            f = (1.0 + amp) * _u(rng, 1.2, 1.4)
        else:
            f = (1.0 + amp) * _u(rng, 1.6, 2.0)
        levers.append(_worker_lever(i, s, f * need, (1.0 + amp) * _workers_for(s, lam)))
    levers += _line_levers(0, line, stages, 0, 1, c["ages"], _ttl(c, _i(rng, 1, 2)),
                           _resting_limit(rng, c, lam))
    gauges = _standard_gauges(stages, [line], None, 2, p.get("hide", []))
    gauges.append({"kind": "handoff", "targets": [1], "seat": 0})
    return {"common": c, "stages": stages, "lines": [line], "pool": None, "regimes": [], "levers": levers,
            "gauges": gauges, "objective": "team_net",
            "motifs": ["queue", "deadline", "handoff", "back_pressure", "matched_capacity", "batch_knee",
                       "private_gauges", "bottleneck"],
            "calibrate_loss": _rng_range(p, "default_loss", (0.15, 0.3)), "n_seats": 2,
            "design": {"lean_seat": lean, "weak": weak}}


def _scn_shared_pool(rng, p: dict) -> dict:
    """Two independent pipelines, one per seat, drawing on one pool. Each has more demand than it can serve
    with the small cache it rests at, and the pool then sits just below its threshold; a large cache serves much
    more but takes a large slice of the pool. The pool's response is drawn so that one seat taking the large
    slice gains while the other loses, and both taking it lose."""
    c = _common(rng, p)
    n = int(p.get("stages", 3))
    roles = ["intake", "work", "store"] if n == 3 else ["work", "store"]
    theta = _u(rng, 0.7, 0.8)
    grab = _u(rng, 0.35, 0.5)          # extra pool taken by the large cache, as a fraction of a seat's draw
    phi1, phi2 = _u(rng, 0.72, 0.8), _u(rng, 0.45, 0.55)   # slowdown when one / both seats take it
    margin = _u(rng, 0.02, 0.04)
    stages: list[dict] = []
    lines: list[dict] = []
    levers: list[dict] = []
    use_r = 0.0
    wcost_total = 0.0
    design = {"restraint": {}, "grab": {}}
    for li in range(2):
        lam = _u(rng, *_rng_range(p, "arrivals", (22.0, 36.0)))
        off = li * n
        st = _linear(rng, n, lam, _rng_range(p, "lag", (1, 3)), line=li, seat=li, offset=off, roles=roles)
        wk = roles.index("work")
        work = st[wk]
        nref = _i(rng, 9, 14)
        work["mu"] = _r(lam / nref)
        work["wcost"] = _r(_u(rng, 0.7, 1.3) / nref, 5)
        work["cache"] = _cache(rng, hmax=(0.8, 0.9), save=(0.8, 0.9))
        half = work["cache"]["half"]
        work["cache"]["cost"] = _r(0.05 / half, 5)        # cheap in money: what it costs is pool
        c_r, c_g = max(1, int(round(0.5 * half))), int(round(3.0 * half))
        over = _u(rng, 1.2, 1.35)      # mean demand relative to what the small cache can serve
        w_work = max(2, int(round(lam / over / _rate_at(work, 1, c_r))))
        lam = _r(over * w_work * _rate_at(work, 1, c_r))
        arr = _arrival(rng, p, lam, amp=(0.1, 0.25), burst_x=(1.15, 1.4))
        amp = arr["amp"]
        cl = dict(c)
        cl["value"] = _u(rng, 6.0, 14.0, 2)
        line = _line(cl, off, off + n - 1, arr, [1.0 if j == li else 0.0 for j in range(2)])
        w0: dict = {}
        for k, s in enumerate(st):
            s["err"] = _u(rng, 0.0, 0.02)
            s["retry"] = 1
            if k == wk:
                lv = _worker_lever(off + k, s, w_work, (1.0 + amp) * nref)
            else:
                s["wmem"] = _r(s["wmem"] * 0.05)       # the work stage is what draws on the pool
                lv = _worker_lever(off + k, s, (1.0 + amp) * _u(rng, 1.6, 2.0) * _workers_for(s, lam),
                                   (1.0 + amp) * _workers_for(s, lam))
            w0[k] = lv["default"]
            levers.append(lv)
        u_w = _pool_use(st, w0, {}, {})
        work["cache"]["mem"] = _r(grab * u_w / ((c_g - c_r) - grab * c_r), 5)
        use_r += u_w + c_r * work["cache"]["mem"]
        wcost_total += sum(w0[k] * s["wcost"] for k, s in enumerate(st))
        levers.append(_lever("cache", [off + wk], li, 0, work["cache"]["max"], c_r))
        stages += st
        lines.append(line)
        cap_r = w_work * _rate_at(work, 1, c_r)
        levers += _line_levers(li, line, stages, li, li, c["ages"], _ttl(c, 1),
                               int(round(_u(rng, 1.6, 2.4) * cap_r)))
        design["restraint"][str(off + wk)] = c_r
        design["grab"][str(off + wk)] = c_g
    u1, u2 = (theta - margin) * (1.0 + grab / 2.0), (theta - margin) * (1.0 + grab)
    x1, x2 = (u1 - theta) / (1.0 - theta), (u2 - theta) / (1.0 - theta)
    a1, a2 = 1.0 / phi1 - 1.0, 1.0 / phi2 - 1.0
    power = math.log(a2 / a1) / math.log(x2 / x1)
    pool = {"size": _r(use_r / (theta - margin)), "theta": theta, "kappa": _r(a2 / x2 ** power),
            "power": _r(power), "cost": _r(_u(rng, 0.08, 0.15) * wcost_total / use_r, 5)}
    return {"common": c, "stages": stages, "lines": lines, "pool": pool, "regimes": [], "levers": levers,
            "gauges": _standard_gauges(stages, lines, pool, 2, p.get("hide", ["late", "offered"])),
            "objective": "net",
            "motifs": ["queue", "deadline", "shared_pool", "thrash_threshold", "cache", "commons",
                       "private_gauges"],
            "calibrate_loss": None, "n_seats": 2, "design": design}


_SCENARIOS = {"pipeline": _scn_pipeline, "contention": _scn_contention, "cascade": _scn_cascade,
              "redundant": _scn_redundant, "lostwork": _scn_lostwork, "handoff": _scn_handoff,
              "shared_pool": _scn_shared_pool}


def _simulate(engine: "TechSys", settings: list[dict], ticks: int, seed: int) -> dict:
    """Totals of a constant-settings run driven by one private random stream (calibration only)."""
    g = np.random.default_rng(int(seed))
    state = engine.init_state(g)
    ns = engine.n_seats
    arrived = lost = 0.0
    revenue = [0.0] * ns
    cost = [0.0] * ns
    for t in range(ticks):
        r = engine.step(state, settings, g, t)
        state = r.state
        arrived += r.info["value_arrived"]
        lost += r.info["value_lost"]
        for j in range(ns):
            revenue[j] += r.revenue[j]
            cost[j] += r.cost[j]
    return {"arrived": arrived, "lost": lost, "revenue": revenue, "cost": cost}


def _finalize(cls, rng, d: dict, p: dict) -> dict:
    """Names, seeded order, and the two calibrations that make the resting configuration marginal: the arrival
    level is set so that the defaults lose a drawn fraction of the arriving value (when the scenario asks for
    it), and the price level is set so that the defaults roughly break even (they lose a little). Both are
    measured by simulating the drawn system at its defaults with private random streams."""
    c = d["common"]
    ns = int(d["n_seats"])
    stages, lines = d["stages"], d["lines"]
    levers, gauges = d["levers"], d["gauges"]
    names = neutral_names(rng, len(levers) + len(gauges))
    order = [int(x) for x in rng.permutation(len(levers))]
    levers = [dict(levers[k]) for k in order]
    order = [int(x) for x in rng.permutation(len(gauges))]
    gauges = [dict(gauges[k]) for k in order]
    for k, lv in enumerate(levers):
        lv["name"] = names[k]
    for k, g in enumerate(gauges):
        g["name"] = names[len(levers) + k]
    # What the customers can take per tick is finite: the last stage of each line has a ceiling of three times
    # the peak of the daily cycle (never reached in normal operation; it bounds how fast a backlog can be flushed).
    # Hence a bound on what can be earned in one tick, declared as the demand income the facility needs.
    income = 0.0
    for ln in lines:
        sink = stages[ln["sink"]]
        if sink.get("cap") is None:
            sink["cap"] = _r(3.0 * ln["arrival"]["base"] * (1.0 + ln["arrival"]["amp"]))
        top_batch = max([lv["hi"] for lv in levers if lv["kind"] == "batch" and ln["sink"] in lv["targets"]] + [1.0])
        income += ln["value"] * (math.ceil(sink["cap"]) + c["substeps"] * top_batch + 1.0)
    cfg = {"engine": "techsys", "schema": 1, "scenario": p.get("scenario", "pipeline"), "n_seats": ns,
           "income_cents_per_tick": int(math.ceil(income)),
           "horizon": c["horizon"], "substeps": c["substeps"], "ages": c["ages"], "objective": d["objective"],
           "stages": stages, "lines": lines, "pool": d["pool"], "regimes": d["regimes"], "levers": levers,
           "gauges": gauges, "noise": {"lat": _u(rng, 0.08, 0.2), "err": _u(rng, 0.005, 0.015), "pool": 0.0},
           "reserve": None, "motifs": list(d["motifs"]), "design": d.get("design", {})}
    if d["objective"] == "ontime_value":
        cfg["reserve"] = {"start": 0.0, "cap": 1.0}
    seeds = [int(x) for x in rng.integers(1, 2 ** 31 - 1, size=2)]
    margin = float(rng.uniform(*_rng_range(p, "default_margin", (0.0, 0.07))))
    goal = None if d.get("calibrate_loss") is None else float(rng.uniform(*d["calibrate_loss"]))
    reserve_ticks = float(rng.uniform(4.0, 8.0))
    ticks = c["horizon"]

    def run_defaults(n_seeds):
        eng = cls(cfg)
        dflt = [eng.defaults(j) for j in range(ns)]
        tots = [_simulate(eng, dflt, ticks, sd) for sd in seeds[:n_seeds]]
        return {"arrived": sum(t["arrived"] for t in tots) / n_seeds, "lost": sum(t["lost"] for t in tots) / n_seeds,
                "revenue": [sum(t["revenue"][j] for t in tots) / n_seeds for j in range(ns)],
                "cost": [sum(t["cost"][j] for t in tots) / n_seeds for j in range(ns)]}

    if not p.get("no_calibration"):
        if goal is not None:
            weak = stages[d["design"]["weak"]]
            mu0 = weak["mu"]
            lo, hi = 0.7, 1.5           # the weak stage is made 1/m as fast
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                weak["mu"] = _r(mu0 / mid)
                tot = run_defaults(1)
                if tot["lost"] / max(1.0, tot["arrived"]) < goal:
                    lo = mid
                else:
                    hi = mid
            m = 0.5 * (lo + hi)
            weak["mu"] = _r(mu0 / m)
            for lv in levers:
                if lv["kind"] == "workers" and lv["targets"] == [d["design"]["weak"]]:
                    lv["hi"] = float(max(lv["hi"], math.ceil(lv["hi"] * m)))
        tot = run_defaults(2)
        rev, cost = tot["revenue"], tot["cost"]
        if min(cost) > 0 and min(rev) > 0:
            # one price level for the pool, then each seat's own prices so that each seat's defaults lose a little
            overall = sum(rev) * (1.0 + margin) / sum(cost)
            pool_part = [0.0] * ns
            if cfg["pool"] is not None:
                dflt = {(lv["kind"], tg): lv["default"] for lv in levers for tg in lv["targets"]}
                for i, s in enumerate(stages):
                    one = _pool_use([s], {0: dflt.get(("workers", i), 1.0)}, {0: dflt.get(("batch", i), 1)},
                                    {0: dflt.get(("cache", i), 0.0)})
                    pool_part[s["seat"]] += one * cfg["pool"]["cost"] * ticks
                cfg["pool"]["cost"] = _r(cfg["pool"]["cost"] * overall, 5)
            for j in range(ns):
                own = cost[j] - pool_part[j]
                scale = (rev[j] * (1.0 + margin) - pool_part[j] * overall) / own if own > 0 else overall
                if scale <= 0:
                    scale = overall
                for s in stages:
                    if s["seat"] == j:
                        s["wcost"] = _r(s["wcost"] * scale, 5)
                        if s["cache"] is not None:
                            s["cache"]["cost"] = _r(s["cache"]["cost"] * scale, 5)
    if cfg["reserve"] is not None:
        per_tick = sum(ln["arrival"]["base"] * ln["value"] for ln in lines)
        cfg["reserve"] = {"start": _r(per_tick * reserve_ticks, 2), "cap": _r(per_tick * 40.0, 2)}
    return cfg


# ====================================================================== reference settings (private)
_STYLES = {"tuned": ("workers", "batch", "cache", "retry", "timeout", "admit", "route"),
           "capacity": ("workers",), "batching": ("batch",), "cache": ("cache",), "shedding": ("timeout", "admit"),
           "routing": ("route",), "retry": ("retry",), "restraint": ("workers", "cache", "timeout", "admit"),
           "grab": ("workers", "cache", "timeout", "admit")}


def reference_settings(cfg: dict, style: str = "tuned", at: int | None = None) -> list[dict[str, float]]:
    """Settings computed from the hidden parameters (evaluator and tests only; never reachable by a seat).
    With `at`, only the permanent scheduled changes that have happened by that tick are reckoned with (the
    settings for that stretch of the episode); without it, all of them (settings for the whole episode).

    `tuned` sets every lever sensibly. The other styles start from the defaults and move one class of levers
    only: `capacity` (workers), `batching`, `cache`, `shedding` (admission limit and timeout), `routing`, `retry`;
    `restraint` and `grab` are the profiles of the shared-pool scenario: both right-size the stages that hardly
    touch the pool, keep the pool-hungry stage's workers as they are, and set its cache small or large. None of
    them is claimed optimal: they are what an operator who knew the mechanism would try first."""
    if style not in _STYLES:
        raise ValueError(f"unknown style {style!r}")
    if style == "tuned" and "restraint" in cfg.get("design", {}):
        style = "restraint"             # with a commons, the sensible joint setting is restraint
    allowed = set(_STYLES[style])
    st, lines, pool = cfg["stages"], cfg["lines"], cfg.get("pool")
    S = len(st)
    by = {}
    for lv in cfg["levers"]:
        for tg in lv["targets"]:
            by[(lv["kind"], tg)] = lv
    val = {lv["name"]: lv["default"] for lv in cfg["levers"]}

    def get(kind, tg, fallback):
        lv = by.get((kind, tg))
        return val[lv["name"]] if lv is not None else fallback

    def put(kind, tg, x):
        lv = by.get((kind, tg))
        if lv is not None and kind in allowed:
            x = min(lv["hi"], max(lv["lo"], float(x)))
            val[lv["name"]] = float(round(x)) if lv["integer"] else x

    permanent = {}
    for r in cfg.get("regimes", []):
        if r.get("until") is None and r["what"] in ("rate", "error") and (at is None or r["at"] <= at):
            permanent.setdefault(r["target"], {})[r["what"]] = r["value"]

    def err(i):
        return permanent.get(i, {}).get("error", st[i]["err"])

    def amplification(i):
        e, lim = err(i), int(get("retry", i, st[i]["retry"]))
        return (1.0 - e ** (lim + 1)) / (1.0 - e) if e < 1.0 else lim + 1.0

    def rate(i):        # successful jobs per worker per tick
        return (_rate_at(st[i], get("batch", i, 1), get("cache", i, 0.0)) * permanent.get(i, {}).get("rate", 1.0)
                / amplification(i))

    def attempts(i):    # attempts per tick the stage can make as it is staffed now
        a = get("workers", i, 1) * _rate_at(st[i], get("batch", i, 1), get("cache", i, 0.0))
        return min(a, st[i]["cap"]) if st[i].get("cap") is not None else a

    profile = style in ("restraint", "grab")
    if profile:
        for key, c in cfg.get("design", {}).get(style, {}).items():
            put("cache", int(key), c)
    for i in range(S):
        if st[i]["batch"] is not None:
            put("batch", i, st[i]["batch"]["knee"])
        put("retry", i, 2)
    # routing
    for i in range(S):
        if len(st[i]["next"]) == 2 and ("route", i) in by:
            ja, jb = st[i]["next"]
            if style == "routing":       # balance traffic to what each path can carry as it is staffed now
                ca, cb = get("workers", ja, 1) * rate(ja), get("workers", jb, 1) * rate(jb)
                put("route", i, ca / (ca + cb))
            elif ja in permanent or jb in permanent:
                put("route", i, 0.0 if ja in permanent else 1.0)
            else:                        # most traffic on the path that is cheaper per delivered job
                ua, ub = st[ja]["wcost"] / rate(ja), st[jb]["wcost"] / rate(jb)
                put("route", i, 0.8 if ua <= ub else 0.2)
    share = [0.0] * S
    for ln in lines:
        share[ln["source"]] = 1.0
    for i in range(S):
        nx = st[i]["next"]
        if len(nx) == 1:
            share[nx[0]] += share[i]
        elif len(nx) == 2:
            r = get("route", i, 1.0)
            share[nx[0]] += share[i] * r
            share[nx[1]] += share[i] * (1.0 - r)

    def design_rate(li):
        a = lines[li]["arrival"]
        return a["base"] * (1.0 + 0.6 * a["amp"]) * (1.0 + 0.5 * max(0.0, a["drift"]))

    # caches
    pc = pool["cost"] if pool is not None else 0.0
    for i in range(S):
        ch = st[i]["cache"]
        if ch is None or "cache" not in allowed or profile:
            continue
        if style == "cache":
            put("cache", i, 3.0 * ch["half"])
            continue
        demand = design_rate(st[i]["line"]) * share[i] * amplification(i)
        best = None
        for mult in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            cc = round(mult * ch["half"])
            wn = math.ceil(demand / _rate_at(st[i], get("batch", i, 1), cc))
            total = wn * (st[i]["wcost"] + st[i]["wmem"] * pc) + cc * (ch["cost"] + ch["mem"] * pc)
            if best is None or total < best[0]:
                best = (total, cc)
        put("cache", i, best[1])
    # workers
    for i in range(S):
        if profile and st[i]["cache"] is not None:
            continue
        need = design_rate(st[i]["line"]) * share[i] / rate(i)
        if st[i].get("cap") is not None:
            need = min(need, st[i]["cap"] / _rate_at(st[i], get("batch", i, 1), get("cache", i, 0.0)) / 1.1)
        if style == "capacity":
            put("workers", i, max(get("workers", i, 1), math.ceil(1.1 * need)))
        else:
            put("workers", i, math.ceil(1.1 * need))

    def utilisation():
        w = {i: get("workers", i, 1.0) for i in range(S)}
        b = {i: get("batch", i, 1) for i in range(S)}
        cz = {i: get("cache", i, 0.0) for i in range(S)}
        return _pool_use(st, w, b, cz) / pool["size"]

    if pool is not None and style in ("tuned", "capacity", "cache"):
        guard = 0
        while utilisation() > pool["theta"] - 0.03 and guard < 60:
            guard += 1
            caches = [(get("cache", i, 0.0), i) for i in range(S) if st[i]["cache"] is not None
                      and "cache" in allowed and get("cache", i, 0.0) > 0]
            if caches:
                cmax, i = max(caches)
                put("cache", i, math.floor(cmax * 0.7))
            elif "workers" in allowed:
                for i in range(S):
                    put("workers", i, max(1, math.floor(get("workers", i, 1) * 0.95)))
            else:
                break
    # timeout and admission limit
    for li, ln in enumerate(lines):
        D = ln["deadline"] - ln.get("settle", 0)        # how long a job may queue and still be in time
        put("timeout", li, ln["deadline"] + (1 if ln.get("tail") and ln["tail"][0] >= 0.4 else 0))
        through = min(attempts(i) * permanent.get(i, {}).get("rate", 1.0) / amplification(i) / share[i]
                      for i in range(S) if st[i]["line"] == li and share[i] > 1e-9)
        limit = (0.7 if profile else 0.9) * (D + 1) * through
        for i in range(S):
            if st[i]["line"] == li and st[i]["stale"] is not None:
                limit = min(limit, 0.9 * st[i]["stale"]["tau"] * attempts(i))
        put("admit", li, max(1, int(limit)))
    out: list[dict[str, float]] = [{} for _ in range(int(cfg.get("n_seats", 1)))]
    for lv in cfg["levers"]:
        out[lv["seat"]][lv["name"]] = float(val[lv["name"]])
    return out


# ====================================================================== families
# Every family is `cross_impl`: the canonical training set contains no techsys family, so whatever a learner
# knows about these systems it learnt from other engines.
@mech_family("tech_pipeline", "techsys", "val", "cross_impl",
             "A linear processing pipeline with a deadline on every job: one stage is short of capacity at the "
             "daily peak, batching helps up to a hidden knee, and added workers start late.",
             motifs=("queue", "deadline", "batch_knee", "startup_lag", "bottleneck"))
def _fam_pipeline(rng, overrides):
    return {"scenario": "pipeline", **overrides}


@mech_family("tech_contention", "techsys", "test", "cross_impl",
             "Stages share one pool that thrashes above a hidden utilisation: adding workers or cache where "
             "capacity is short slows every stage; the fix is to take resources away somewhere else first.",
             motifs=("queue", "deadline", "shared_pool", "thrash_threshold", "cache", "batch_knee", "bottleneck",
                     "distractor"))
def _fam_contention(rng, overrides):
    return {"scenario": "contention", **overrides}


@mech_family("tech_cascade", "techsys", "test", "cross_impl",
             "Retries against a flaky dependency: a temporary slowdown tips the system into a retry storm that "
             "persists after the dependency recovers, until load is shed.",
             motifs=("queue", "deadline", "retry_storm", "metastable_collapse", "regime_change", "batch_knee",
                     "distractor"))
def _fam_cascade(rng, overrides):
    return {"scenario": "cascade", **overrides}


@mech_family("tech_redundant", "techsys", "test", "cross_impl",
             "Two alternative paths with different hidden reliabilities and prices; the dependency of the cheap "
             "one degrades for good in mid-episode.",
             motifs=("queue", "deadline", "redundant_paths", "routing", "unequal_reliability", "regime_change",
                     "retry"))
def _fam_redundant(rng, overrides):
    return {"scenario": "redundant", **overrides}


@mech_family("tech_lostwork", "techsys", "test", "cross_impl",
             "Scored on the value of work delivered in time while paying its way: the resting configuration loses "
             "much of the arriving work, and capacity, batching, caching, routing, retries or load shedding can "
             "each recover a large part of it.",
             motifs=("queue", "deadline", "redundant_paths", "routing", "unequal_reliability", "retry", "batch_knee",
                     "cache", "shared_pool", "bottleneck", "sustainability_constraint"))
def _fam_lostwork(rng, overrides):
    return {"scenario": "lostwork", **overrides}


@mech_family("tech_handoff", "techsys", "val", "cross_impl",
             "Cooperative, two seats: one operates the upstream half of a pipeline and the other the downstream "
             "half; only matched capacity produces value and each seat sees only its own gauges.",
             motifs=("queue", "deadline", "handoff", "back_pressure", "matched_capacity", "batch_knee",
                     "private_gauges", "bottleneck"), seats=2)
def _fam_handoff(rng, overrides):
    return {"scenario": "handoff", **overrides}


@mech_family("tech_shared_pool", "techsys", "test", "cross_impl",
             "Mixed motive, two seats: two independent pipelines draw on one pool; each gains by taking more "
             "until the pool thrashes for both.",
             motifs=("queue", "deadline", "shared_pool", "thrash_threshold", "cache", "commons", "private_gauges"),
             seats=2)
def _fam_shared_pool(rng, overrides):
    return {"scenario": "shared_pool", **overrides}
