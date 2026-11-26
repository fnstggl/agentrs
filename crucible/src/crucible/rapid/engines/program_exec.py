"""Engine `program_exec`: executable program environments (MASTER_SPEC Part 7.3, Part 8.3 item 6, Part 8.11).

The participant writes Python programs - controllers, schedulers, data processors, routers, policies, solvers - and
the *world* executes them: a deployed program is called by the transition code itself, at the world's own rate
(several inner steps per tick, once per arriving job, once per raw sample, once per test record), against dynamics
or test suites the participant cannot see. This is the difference from the controller action of the other engines
(`crucible.rapid.program`), where a program merely issues the participant's ordinary per-tick actions: here the
program sits inside the dynamics, sees inputs no observation carries (every raw sample, every inner step) and its
outputs drive the hidden system directly.

Execution goes through `crucible.sandbox` (separate unprivileged process, Landlock + seccomp, rlimits, wall-clock
timeouts, no network, numpy only). Each call runs the program's source in a fresh namespace (a small harness
compiles `program.py` once and executes it anew for every call), so a program is a function of what it is given:
the world's input and `caller` = {"memory", "step", "tick"}. A program keeps state only by returning a dict with
a `memory` entry (JSON, at most 4 kB), which the world hands back on the next call. A fresh sandbox process serves
each tick's calls and is closed at the end of the tick (nothing is shared between ticks or between copies of a
world). Work is measured by the parent outside the sandboxed process (CPU time in work units, D-20,
`crucible.sandbox.metering`; identical computations are billed identically, `SandboxConfig.count_lines`) and charged as
compute cost; a call that raises, times out or returns something malformed is answered by the world's built-in
fallback and reported.

    {"kind": "program", "source": "def act(observation, caller):\\n    ...", "ticks": 20}

deploys (or, in solver worlds, submits) a program. Scenarios (one transition function each, below):

    fast_control      an open-loop unstable plant at 10 inner steps per tick; the controller is called every step
    deadtime_control  a stable plant with hidden dead time and lag tracking a public setpoint schedule
    contention        a shared node: a scheduler picks jobs to start; hidden resource classes interfere
    stream            a data processor turns raw multi-sensor samples (drifting, failing sensors) into estimates
    cascade_guard     a load-shedding program protects a ring of units with hidden true capacities
    router            a router sends each arriving job to one of several lines with hidden size-dependent speeds
    rule_induction    a solver program is scored on hidden test suites of a hidden rule (shifted holdout suite)
    fleet             a maintenance policy for a fleet with hidden wear seen through noisy vibration readings
    rollout           a targeting program exposes users; adoption is delayed, early exposure can be rejected
"""
from __future__ import annotations

import json
import math

import numpy as np

from .. import actions as A
from ...sandbox.runner import SandboxConfig, SandboxUnavailable, Worker
from .base import ActionError, RapidEngine, register

MEMORY_BYTES = 4096
HARNESS = '''\
_CODE = compile(open("program.py").read(), "program.py", "exec")


def call(args, caller):
    ns = {"__name__": "program"}
    exec(_CODE, ns)
    fn = ns.get("act")
    if not callable(fn):
        raise NameError("the program defines no act(observation, caller)")
    return fn(args, caller)
'''


def sandbox_config() -> SandboxConfig:
    return SandboxConfig(job_wall_s=5.0, mem_mb=768, cpu_s=120, count_lines=True)


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class ProgramRunner:
    """Calls of one deployed program during one tick (or one submission): a fresh sandbox process, started on the
    first call and closed by `close()`. Never stored in a world state."""

    def __init__(self, source: str, memory=None):
        self.source, self.memory = source, memory
        self.worker: Worker | None = None
        self.calls = self.errors = self.lines = 0
        self.first_error: str | None = None
        self.dead = False

    def _fail(self, why: str):
        self.errors += 1
        if self.first_error is None:
            self.first_error = why[:300]
        return None

    def __call__(self, args, **caller):
        if self.dead:
            return self._fail(self.first_error or "the program stopped")
        if self.worker is None:
            try:
                self.worker = Worker({"program.py": self.source, "harness.py": HARNESS}, sandbox_config(), "rapid-pe")
                self.worker.start()
            except (SandboxUnavailable, OSError, ValueError) as e:
                self.dead = True
                return self._fail(f"sandbox unavailable: {e}")
        r = self.worker.run({"kind": "call", "target": "harness.py:call", "args": args,
                             "caller": {"memory": self.memory, **caller}})
        self.calls += 1
        self.lines += max(0, int(r.get("work_lines") or 0))     # parent-measured work units, never self-reported
        if not r.get("ok"):
            err = r.get("error") or {}
            if err.get("type") in ("Timeout", "WorkerDied", "WorkerDead"):
                self.dead = True
            return self._fail(f"{err.get('type')}: {str(err.get('message', ''))[:200]}")
        out = r.get("result")
        if isinstance(out, dict) and "memory" in out:
            try:
                txt = json.dumps(out["memory"])
            except (TypeError, ValueError):
                txt = None
            if txt is None or len(txt) > MEMORY_BYTES:
                return self._fail(f"memory must be JSON of at most {MEMORY_BYTES} bytes")
            self.memory = json.loads(txt)
        return out

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def field(out, key: str):
    """A numeric field of a program's output: `out[key]` of a dict, or the output itself when it is a number."""
    if isinstance(out, dict):
        return _finite(out.get(key))
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        return _finite(out)
    return None


def index_list(out, key: str, valid, limit: int) -> list:
    """A list of identifiers a program chose (`out[key]` or the output itself), filtered to `valid`, deduplicated,
    at most `limit`."""
    xs = out.get(key) if isinstance(out, dict) else out
    if not isinstance(xs, list):
        return []
    seen, res = set(), []
    for x in xs:
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x == int(x):
            x = int(x)
        if x in valid and x not in seen:
            seen.add(x)
            res.append(x)
        if len(res) >= limit:
            break
    return res


def program_kind(max_ticks: int, doc: str, submission: bool = False) -> dict:
    props = {"kind": {"type": "string", "enum": ["program"],
                      "description": ("submit a program that the world runs on its test suites: " if submission else
                                      "deploy a program that the world calls inside its dynamics: ") + doc},
             "source": A.S.s_str("Python source defining act(observation, caller)", maxLength=A.MAX_PROGRAM_BYTES)}
    req = ["kind", "source"]
    if not submission:
        props["ticks"] = A.S.s_int("how many ticks the program stays deployed", minimum=1, maximum=max_ticks)
        req.append("ticks")
    return A.S.obj(props, req)


# ====================================================================== scenarios
class Scenario:
    """One executable-program world. Subclasses hold the transition code; `self.c` is the config, `self.h` its
    hidden part (evaluator data)."""
    name = ""
    doc = ""                 # what act() receives and returns (public)
    submission = False       # True: programs are submitted and scored on test suites instead of deployed
    motifs_: tuple = ()
    edges_: tuple = ()       # (src, dst, kind, lag) of the hidden mechanism, for the evaluator's fingerprint

    def __init__(self, cfg: dict):
        self.c, self.h = cfg, cfg["hidden"]

    def manual(self, state) -> tuple[dict, dict, dict]:
        """(kinds, limits, query targets) of the manual interface besides `program`."""
        return {}, {}, {}

    def init_state(self, rng) -> dict:
        return {}

    def apply(self, st, action, rng, t) -> dict:
        raise ActionError(f"{action['kind']} is not available in this world")

    def query(self, st, target, args, rng, t) -> dict:
        raise ActionError(f"unknown query target {target}")

    def submit(self, st, source, rng, t) -> dict:
        raise ActionError("this world runs deployed programs; submissions are not scored here")

    def advance(self, st, rng, t, run) -> dict:
        raise NotImplementedError

    def outcome(self, st, net) -> dict:
        return {"value": net}

    def conservation(self, before, after, out) -> list[str]:
        return []

    def latent(self, st):
        return None

    def truth(self) -> dict:
        nodes = sorted({e[0] for e in self.edges_} | {e[1] for e in self.edges_})
        return {"nodes": nodes, "edges": [{"src": a, "dst": b, "kind": k, "lag": lag} for a, b, k, lag in self.edges_]}


SCENARIOS: dict[str, type[Scenario]] = {}


def scenario(cls):
    SCENARIOS[cls.name] = cls
    return cls


# ---------------------------------------------------------------------- fast_control (8.3.6)
@scenario
class FastControl(Scenario):
    """x[k+1] = x[k] + a x[k] + b u[k] + d(t) + w[k] at `inner` steps per tick, a > 0 (open-loop unstable); the
    sign and size of b, the load disturbance d(t) (piecewise constant, stepping at hidden ticks) and the noise are
    hidden; the reading y = x + noise. Each step inside the band pays in proportion to how close to zero the state
    is; past the trip level the plant trips and halts for `halt_ticks` ticks. The manual lever holds one actuator
    value for the whole tick; a deployed controller is called every inner step."""
    name = "fast_control"
    doc = ("act(observation, caller) is called every inner step with observation = {\"reading\": y, \"step\": k, "
           "\"tick\": t, \"last_u\": previous actuator value} and returns {\"u\": actuator value} (clipped to the "
           "actuator range); return {\"u\": ..., \"memory\": ...} to keep state. If it fails, the manual lever "
           "value is used for that step.")
    motifs_ = ("feedback", "measurement_noise", "partial_observability", "threshold", "linear_nonlinear")
    edges_ = (("actuator", "state", "causal", 0), ("state", "state", "feedback", 1), ("bias", "state", "causal", 0),
              ("state", "reading", "sensor", 0), ("state", "trip", "threshold", 0))

    def manual(self, state):
        c = self.c
        lv = {"name": c["u_name"], "min": -c["umax"], "max": c["umax"], "default": 0.0}
        return ({"levers": A.levers_schema([lv])}, {},
                {"trace": "the readings of every inner step of the last tick (also in arrays)"})

    def init_state(self, rng):
        return {"x": 0.0, "halt": 0, "u": 0.0, "trace": [0.0] * int(self.c["inner"]), "trips": 0,
                "inband": 0, "steps": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "levers":
            v = action["values"].get(self.c["u_name"])
            if v is not None:
                st["u"] = float(v)
            return {"set": {self.c["u_name"]: st["u"]}}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"trace": list(st["trace"])}

    def bias(self, t: int) -> float:
        """The load disturbance: piecewise constant, changing at hidden ticks."""
        v = 0.0
        for start, val in self.h["d"]:
            if t >= start:
                v = val
        return v

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        K = int(c["inner"])
        z = rng.standard_normal((K, 2))
        x, umax = float(st["x"]), float(c["umax"])
        trace, inband, quality, energy, tripped, last_u = [], 0, 0.0, 0.0, False, float(st["u"])
        for k in range(K):
            y = x + c["sensor_sd"] * float(z[k, 1])
            trace.append(round(y, 6))
            if st["halt"] > 0:
                x = 0.0
                continue
            u = float(st["u"])
            if run is not None:
                v = field(run({"reading": round(y, 6), "step": k, "tick": t, "last_u": last_u}, step=k, tick=t), "u")
                if v is not None:
                    u = v
            u = min(umax, max(-umax, u))
            last_u = u
            energy += abs(u)
            x = x + h["a"] * x + h["b"] * u + self.bias(t) + h["w_sd"] * float(z[k, 0])
            if abs(x) < c["band"]:
                inband += 1
                quality += 1.0 - abs(x) / c["band"]
            if abs(x) > c["trip"]:
                tripped = True
                x = 0.0
                st["halt"] = int(c["halt_ticks"]) + 1
        if st["halt"] > 0:
            st["halt"] -= 1
        st["x"], st["trace"] = x, trace
        st["trips"] += int(tripped)
        st["inband"] += inband
        st["steps"] += K
        rev = c["price"] * quality
        cost = c["energy_price"] * energy + (c["trip_cost"] if tripped else 0.0)
        ob = {"numbers": {c["y_name"]: trace[-1]}, "arrays": {"trace": trace},
              "text": ["the plant tripped and is halted"] if tripped else []}
        return {"obs": ob, "revenue": rev, "cost": cost}

    def latent(self, st):
        return [float(st["x"])]


# ---------------------------------------------------------------------- rule_induction (8.2.1)
@scenario
class RuleInduction(Scenario):
    """A hidden rule maps records (two numeric fields and a category) to a value: a category offset, a
    category-specific slope in the second field, and a response to the first field that is linear up to a hidden
    threshold and gains curvature beyond it. Probes (costly, noisy) answer the rule at chosen records inside the
    probe box. A submitted program is run on a validation suite drawn from the probe box (its pass count is
    reported) and on a holdout suite of records beyond the probe box (never reported); the last submission's
    holdout pass rate is the objective."""
    name = "rule_induction"
    submission = True
    doc = ("act(record, caller) is called once per test record, record = {field: value, ...} with the two numeric "
           "fields and the category field, and returns the predicted value (a number, or {\"value\": number}). A "
           "prediction passes when it is within the public tolerance of the rule's value.")
    motifs_ = ("threshold", "linear_nonlinear", "multiplicative", "measurement_noise", "partial_observability")
    edges_ = (("field_a", "value", "threshold", 0), ("field_b", "value", "causal", 0),
              ("category", "value", "causal", 0), ("category", "slope_b", "modifies", 0),
              ("slope_b", "value", "interaction", 0))

    def manual(self, state):
        c = self.c
        q = {"probe": f"the rule's value at a record inside the probe box (args: {{\"record\": {{...}}}}), with "
                      f"measurement noise; costs {c['probe_cost']} budget",
             "examples": "the public worked examples"}
        return {}, {}, q

    def init_state(self, rng):
        return {"budget": float(self.c["budget"]), "submissions": 0, "holdout_pass": 0.0, "validation_pass": 0.0,
                "src": None}

    def value(self, rec: dict) -> float:
        """The rule (noise-free)."""
        c, h = self.c, self.h
        fa, fb, fc = c["fields"]
        k = c["categories"].index(rec[fc])
        a, b = float(rec[fa]), float(rec[fb])
        return h["off"][k] + h["slope"][k] * b + h["k1"] * a + h["k2"] * max(0.0, a - h["theta"]) ** h["p"]

    def _record(self, args) -> dict:
        c = self.c
        fa, fb, fc = c["fields"]
        rec = (args or {}).get("record")
        if not isinstance(rec, dict):
            raise ActionError("probe needs args {\"record\": {field: value, ...}}")
        out = {}
        for f in (fa, fb):
            v = _finite(rec.get(f))
            lo, hi = c["box"][f]
            if v is None or not (lo <= v <= hi):
                raise ActionError(f"field {f} must be a number in the probe box [{lo}, {hi}]")
            out[f] = v
        if rec.get(fc) not in c["categories"]:
            raise ActionError(f"field {fc} must be one of {c['categories']}")
        out[fc] = rec[fc]
        return out

    def query(self, st, target, args, rng, t):
        c = self.c
        if target == "examples":
            return {"examples": c["examples"]}
        cost = float(c["probe_cost"])
        if st["budget"] < cost - 1e-12:
            raise ActionError(f"not enough budget ({st['budget']:.3g} left, a probe costs {cost:g})")
        rec = self._record(args)
        st["budget"] -= cost
        v = self.value(rec) + c["probe_sd"] * float(rng.standard_normal())
        return {"record": rec, "value": round(v, 6), "cost": cost, "budget_left": st["budget"]}

    def _score(self, run, suite) -> tuple[int, int]:
        ok = 0
        for rec in suite:
            v = field(run(dict(rec), step=0, tick=0), "value")
            if v is not None and abs(v - self.value(rec)) <= self.c["tol"]:
                ok += 1
        return ok, len(suite)

    def submit(self, st, source, rng, t):
        c = self.c
        cost = float(c["submit_cost"])
        if st["budget"] < cost - 1e-12:
            raise ActionError(f"not enough budget ({st['budget']:.3g} left, a submission costs {cost:g})")
        st["budget"] -= cost
        run = ProgramRunner(source)
        try:
            kv, nv = self._score(run, self.h["validation"])
            kh, nh = self._score(run, self.h["holdout"])
        finally:
            run.close()
        st["submissions"] += 1
        st["validation_pass"], st["holdout_pass"] = kv / nv, kh / nh
        st["src"] = source
        res = {"submitted": True, "validation": {"passed": kv, "of": nv}, "errors": run.errors,
               "compute_lines": run.lines, "budget_left": st["budget"]}
        if run.first_error:
            res["first_error"] = run.first_error
        return res, run.lines

    def advance(self, st, rng, t, run):
        """Observational data: every tick a few records seen in the field (inside the probe box) and their noisy
        values arrive for free; probes are the interventions."""
        c = self.c
        fa, fb, fc = c["fields"]
        n = int(c.get("field_records", 1))
        z, e = rng.random((n, 3)), rng.standard_normal(n)
        (alo, ahi), (blo, bhi) = c["box"][fa], c["box"][fb]
        nums, ents = {}, []
        for i in range(n):
            rec = {fa: round(alo + (ahi - alo) * float(z[i, 0]), 4), fb: round(blo + (bhi - blo) * float(z[i, 1]), 4),
                   fc: c["categories"][min(len(c["categories"]) - 1, int(z[i, 2] * len(c["categories"])))]}
            v = round(self.value(rec) + c["probe_sd"] * float(e[i]), 6)
            nums.update({f"{fa}_{i}": rec[fa], f"{fb}_{i}": rec[fb], f"{c['value_name']}_{i}": v})
            ents.append(dict(rec, kind="field_record", value=v))
        return {"obs": {"numbers": nums, "entities": ents, "resources": {"budget": st["budget"]}},
                "revenue": 0.0, "cost": 0.0}

    def outcome(self, st, net):
        return {"value": float(st["holdout_pass"]), "holdout_pass": float(st["holdout_pass"]),
                "validation_pass": float(st["validation_pass"]), "submissions": int(st["submissions"])}

    def conservation(self, before, after, out):
        return ["budget grew"] if after["budget"] > before["budget"] + 1e-9 else []


# ---------------------------------------------------------------------- deadtime_control (8.3.3)
@scenario
class DeadtimeControl(Scenario):
    """z[k+1] = z[k] + (G u[k-L] + d[k] - z[k]) / tau + w[k]: a stable first-order process with hidden gain G,
    time constant tau and dead time L inner steps, and an unmeasured load d that drifts (AR(1)). The setpoint
    follows a public schedule; each inner step pays in proportion to closeness to it; actuator moves wear the
    valve. The manual lever holds one valve position per tick; a deployed controller runs every inner step."""
    name = "deadtime_control"
    doc = ("act(observation, caller) is called every inner step with observation = {\"reading\": y, \"setpoint\": r, "
           "\"step\": k, \"tick\": t, \"last_u\": previous valve position} and returns {\"u\": position} (clipped "
           "to [0, umax]); return {\"u\": ..., \"memory\": ...} to keep state (integrators, histories). If it "
           "fails, the manual lever value is used for that step.")
    motifs_ = ("delay", "decay", "feedback", "hidden_confounding", "measurement_noise", "partial_observability")
    edges_ = (("valve", "process", "causal", 2), ("process", "process", "decay", 1), ("load", "process", "causal", 0),
              ("load", "load", "autoregressive", 1), ("process", "reading", "sensor", 0),
              ("setpoint", "payoff", "compare", 0), ("process", "payoff", "compare", 0))

    def manual(self, state):
        c = self.c
        return {"levers": A.levers_schema([{"name": c["u_name"], "min": 0.0, "max": c["umax"], "default": 0.0}])}, \
            {}, {"trace": "the readings of every inner step of the last tick (also in arrays)"}

    def init_state(self, rng):
        return {"z": 0.0, "load": 0.0, "hist": [0.0] * (int(self.c["max_dead"]) + 1), "u": 0.0, "last_u": 0.0,
                "trace": [0.0] * int(self.c["inner"]), "err": 0.0, "steps": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "levers":
            v = action["values"].get(self.c["u_name"])
            if v is not None:
                st["u"] = float(v)
            return {"set": {self.c["u_name"]: st["u"]}}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"trace": list(st["trace"])}

    def setpoint(self, t: int) -> float:
        v = 0.0
        for start, val in self.c["setpoints"]:
            if t >= start:
                v = val
        return v

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        K, L = int(c["inner"]), int(h["L"])
        z = rng.standard_normal((K, 3))
        r = self.setpoint(t)
        trace, quality, moves, energy = [], 0.0, 0.0, 0.0
        x, last = float(st["z"]), float(st["last_u"])
        for k in range(K):
            y = x + c["sensor_sd"] * float(z[k, 1])
            trace.append(round(y, 6))
            u = float(st["u"])
            if run is not None:
                v = field(run({"reading": round(y, 6), "setpoint": r, "step": k, "tick": t, "last_u": last},
                              step=k, tick=t), "u")
                if v is not None:
                    u = v
            u = min(c["umax"], max(0.0, u))
            moves += abs(u - last)
            energy += u
            last = u
            st["hist"] = (st["hist"] + [u])[-(int(c["max_dead"]) + 1):]
            ud = st["hist"][-(L + 1)]
            st["load"] = h["phi"] * st["load"] + h["load_sd"] * float(z[k, 2])
            x = x + (h["G"] * ud + st["load"] - x) / h["tau"] + h["w_sd"] * float(z[k, 0])
            quality += max(0.0, 1.0 - abs(x - r) / c["band"])
            st["err"] += abs(x - r)
        st["z"], st["last_u"], st["trace"] = x, last, trace
        st["steps"] += K
        ob = {"numbers": {c["y_name"]: trace[-1], c["r_name"]: r}, "arrays": {"trace": trace}}
        return {"obs": ob, "revenue": c["price"] * quality,
                "cost": c["move_price"] * moves + c["energy_price"] * energy}

    def latent(self, st):
        return [float(st["z"]), float(st["load"])]


# ---------------------------------------------------------------------- contention (8.3.4)
@scenario
class Contention(Scenario):
    """A shared node with a fixed number of slots. Jobs of public types arrive (Bernoulli per inner step) with
    public sizes; each type belongs to a hidden resource class, and jobs of one class running together slow each
    other: a job's speed is 1 / (1 + kappa (n_same_class - 1)). Completed jobs pay; waiting jobs cost; jobs that
    wait too long are dropped. The manual lever caps concurrency for the built-in oldest-first rule; a deployed
    scheduler is called at every inner step with a free slot and waiting jobs."""
    name = "contention"
    doc = ("act(observation, caller) is called at every inner step that has a free slot and waiting jobs, with "
           "observation = {\"queue\": [{\"id\", \"type\", \"size\", \"waited\"}, ...] (oldest first), \"running\": "
           "[{\"id\", \"type\", \"left\"}, ...], \"free\": free slots, \"step\": k, \"tick\": t} and returns "
           "{\"start\": [job ids]} (at most `free`; unknown ids are ignored). If it fails, the built-in rule starts "
           "the oldest jobs up to the concurrency cap.")
    motifs_ = ("congestion", "resource_competition", "non_monotonic", "partial_observability", "stock",
               "multiplicative")
    edges_ = (("arrivals", "queue", "flow", 0), ("queue", "running", "dispatch", 0), ("running", "speed", "congestion", 0),
              ("class", "speed", "modifies", 0), ("speed", "completions", "causal", 0), ("queue", "drops", "threshold", 1),
              ("completions", "running", "flow", 0))

    def manual(self, state):
        c = self.c
        return {"levers": A.levers_schema([{"name": c["cap_name"], "min": 1.0, "max": float(c["slots"]),
                                            "default": float(c["slots"]), "integer": True}])}, {}, \
            {"queue": "the waiting and running jobs (also in entities)"}

    def init_state(self, rng):
        return {"queue": [], "running": [], "next": 0, "cap": int(self.c["slots"]), "arrived": 0, "done": 0,
                "dropped": 0, "busy_steps": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "levers":
            v = action["values"].get(self.c["cap_name"])
            if v is not None:
                st["cap"] = int(round(float(v)))
            return {"set": {self.c["cap_name"]: st["cap"]}}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"queue": [dict(j) for j in st["queue"]], "running": [dict(j) for j in st["running"]]}

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        K, C = int(c["inner"]), int(c["slots"])
        z = rng.random((K, 3))
        types = c["types"]
        done_now = 0
        for k in range(K):
            if z[k, 0] < c["arrival_p"]:
                cum, ty = 0.0, types[-1]
                for name, p in zip(types, c["mix"]):
                    cum += p
                    if z[k, 1] < cum:
                        ty = name
                        break
                lo, hi = c["size"]
                st["queue"].append({"id": int(st["next"]), "type": ty, "size": round(lo + (hi - lo) * float(z[k, 2]), 4),
                                    "waited": 0})
                st["next"] += 1
                st["arrived"] += 1
            free = C - len(st["running"])
            if free > 0 and st["queue"]:
                ids = None
                if run is not None:
                    out = run({"queue": [dict(j) for j in st["queue"]], "running": [dict(j) for j in st["running"]],
                               "free": free, "step": k, "tick": t}, step=k, tick=t)
                    if out is not None:
                        ids = index_list(out, "start", {j["id"] for j in st["queue"]}, free)
                if ids is None:
                    n = max(0, min(free, int(st["cap"]) - len(st["running"])))
                    ids = [j["id"] for j in st["queue"][:n]]
                pick = set(ids)
                for j in [j for j in st["queue"] if j["id"] in pick]:
                    st["running"].append({"id": j["id"], "type": j["type"], "left": j["size"]})
                st["queue"] = [j for j in st["queue"] if j["id"] not in pick]
            cls = [h["cls"][types.index(j["type"])] for j in st["running"]]
            keep = []
            for j, cl in zip(st["running"], cls):
                n_same = sum(1 for x in cls if x == cl)
                j["left"] = round(j["left"] - c["step_work"] / (1.0 + h["kappa"] * (n_same - 1)), 6)
                if j["left"] <= 1e-9:
                    done_now += 1
                else:
                    keep.append(j)
            st["busy_steps"] += len(st["running"])
            st["running"] = keep
        st["done"] += done_now
        for j in st["queue"]:
            j["waited"] += 1
        drop = [j for j in st["queue"] if j["waited"] > c["max_wait"]]
        st["queue"] = [j for j in st["queue"] if j["waited"] <= c["max_wait"]]
        st["dropped"] += len(drop)
        rev = c["price"] * done_now
        cost = c["hold_cost"] * len(st["queue"]) + c["drop_cost"] * len(drop)
        ents = [dict(j, kind="waiting") for j in st["queue"]] + [dict(j, kind="running") for j in st["running"]]
        ob = {"numbers": {"queue_length": float(len(st["queue"])), "running": float(len(st["running"])),
                          "completed_last_tick": float(done_now), "dropped_last_tick": float(len(drop))},
              "entities": ents}
        return {"obs": ob, "revenue": rev, "cost": cost}

    def conservation(self, before, after, out):
        n = after["done"] + after["dropped"] + len(after["queue"]) + len(after["running"])
        return [] if n == after["arrived"] else ["jobs are not conserved"]

    def latent(self, st):
        return [float(len(st["queue"])), float(len(st["running"]))]


# ---------------------------------------------------------------------- stream (8.3.2)
@scenario
class Stream(Scenario):
    """A process value (AR(1) around a public mean) is read by several sensors whose gains and offsets drift as
    hidden random walks of sensor-specific speed; a sensor can fail and stick at its last reading (hidden
    hazard). A data processor is called for every raw sample - the participant's observations carry only tick
    means - and its estimate is paid by closeness to the true value. Each tick a laboratory audit reports the true
    value and raw readings of one sample; recalibrating a sensor (maintenance, costly) resets it."""
    name = "stream"
    doc = ("act(observation, caller) is called for every raw sample with observation = {\"readings\": [one value "
           "per sensor], \"sample\": s, \"tick\": t, \"audit\": {\"readings\": [...], \"value\": true value} or "
           "null} and returns {\"estimate\": number}; the audit (the laboratory's check of the last sample of the "
           "previous tick) comes with the first sample of each tick. Return {\"estimate\": ..., \"memory\": ...} to "
           "keep calibrations. If it fails, the built-in estimate (the mean of the readings) is used.")
    motifs_ = ("measurement_noise", "decay", "regime_change", "missing_observation", "partial_observability",
               "linear_nonlinear")
    edges_ = (("process", "reading", "sensor", 0), ("gain", "reading", "multiplicative", 0),
              ("offset", "reading", "causal", 0), ("gain", "gain", "random_walk", 1),
              ("offset", "offset", "random_walk", 1), ("failure", "reading", "regime", 0),
              ("process", "process", "autoregressive", 1), ("estimate", "payoff", "compare", 0),
              ("maintenance", "gain", "reset", 0))

    def manual(self, state):
        names = list(self.c["sensor_names"])
        return {"discrete": A.discrete_schema({"recalibrate": names})}, {"decisions": {"recalibrate": names}}, \
            {"audit": "the last laboratory audit (true value and raw readings of one sample)"}

    def init_state(self, rng):
        N = len(self.c["sensor_names"])
        return {"g": [1.0] * N, "o": [0.0] * N, "stuck": [False] * N, "stuck_val": [0.0] * N,
                "v": float(self.c["mean"]), "audit": None, "due": 0.0, "err": 0.0, "samples": 0, "recals": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "discrete":
            i = self.c["sensor_names"].index(action["choice"])
            st["g"][i], st["o"][i], st["stuck"][i] = 1.0, 0.0, False
            st["due"] += self.c["recal_cost"]
            st["recals"] += 1
            return {"recalibrated": action["choice"], "cost": self.c["recal_cost"]}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"audit": st["audit"]}

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        N, S = len(c["sensor_names"]), int(c["samples"])
        zp = rng.standard_normal(S)
        zs = rng.standard_normal((S, N))
        zg, zo = rng.standard_normal(N), rng.standard_normal(N)
        uf = rng.random(N)
        quality, sums, last = 0.0, [0.0] * N, None
        for s in range(S):
            st["v"] = c["mean"] + h["phi"] * (st["v"] - c["mean"]) + h["proc_sd"] * float(zp[s])
            rd = [round(st["stuck_val"][i] if st["stuck"][i] else
                        st["g"][i] * st["v"] + st["o"][i] + c["sensor_sd"] * float(zs[s, i]), 6) for i in range(N)]
            e = None
            if run is not None:
                e = field(run({"readings": rd, "sample": s, "tick": t, "audit": st["audit"] if s == 0 else None},
                              step=s, tick=t), "estimate")
            if e is None:
                e = sum(rd) / N
            quality += max(0.0, 1.0 - abs(e - st["v"]) / c["tol"])
            st["err"] += abs(e - st["v"])
            sums = [a + b for a, b in zip(sums, rd)]
            last = rd
        st["samples"] += S
        st["audit"] = {"readings": last, "value": round(st["v"], 6)}
        for i in range(N):
            st["g"][i] += h["g_sd"][i] * float(zg[i])
            st["o"][i] += h["o_sd"][i] * float(zo[i])
            if not st["stuck"][i] and uf[i] < h["hazard"][i]:
                st["stuck"][i], st["stuck_val"][i] = True, last[i]
        cost, st["due"] = st["due"], 0.0
        nums = {f"{n}_mean": round(x / S, 6) for n, x in zip(c["sensor_names"], sums)}
        nums["audit_value"] = st["audit"]["value"]
        ob = {"numbers": nums, "arrays": {"audit_readings": list(last)}}
        return {"obs": ob, "revenue": c["price"] * quality, "cost": cost}

    def latent(self, st):
        return [float(st["v"])] + [float(x) for x in st["o"]]


# ---------------------------------------------------------------------- router (8.3.1)
@scenario
class Router(Scenario):
    """Parallel processing lines (FIFO, one job in service each). A job of size s takes alpha + beta s^gamma steps
    on a line (hidden per line), lengthened on the congestible line when its queue is long; a little service noise.
    Jobs arrive every inner step with some probability; completed jobs pay by size; jobs in the system cost every
    step and are dropped past a wait limit. The manual `allocate` sets the built-in split (weighted round robin);
    a deployed router is called for every arriving job."""
    name = "router"
    doc = ("act(observation, caller) is called for every arriving job with observation = {\"size\": s, \"queues\": "
           "[jobs on each line, in service included], \"sizes\": [[sizes of the jobs on each line], ...], \"step\": "
           "k, \"tick\": t} and returns {\"line\": index}. If it fails, the built-in split is used.")
    motifs_ = ("congestion", "linear_nonlinear", "resource_competition", "partial_observability", "stock",
               "threshold")
    edges_ = (("arrivals", "router", "flow", 0), ("router", "line", "dispatch", 0), ("size", "service", "nonlinear", 0),
              ("line", "service", "modifies", 0), ("queue", "service", "congestion", 0),
              ("service", "completions", "causal", 1), ("queue", "drops", "threshold", 1))

    def manual(self, state):
        names = list(self.c["line_names"])
        return {"allocate": A.allocate_schema(names, 1.0, "the built-in split of arriving jobs")}, \
            {"allocate_total": 1.0}, {"lines": "the jobs on each line"}

    def init_state(self, rng):
        M = len(self.c["line_names"])
        return {"lines": [[] for _ in range(M)], "split": [1.0 / M] * M, "credit": [0.0] * M, "next": 0,
                "arrived": 0, "done": 0, "dropped": 0, "log": []}

    def apply(self, st, action, rng, t):
        if action["kind"] == "allocate":
            w = [max(0.0, float(action["amounts"].get(n, 0.0))) for n in self.c["line_names"]]
            s = sum(w)
            if s <= 0:
                raise ActionError("the split needs a positive share somewhere")
            st["split"] = [x / s for x in w]
            return {"split": dict(zip(self.c["line_names"], st["split"]))}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"lines": [[dict(j) for j in q] for q in st["lines"]]}

    def service_time(self, m: int, s: float, qlen: int, z: float) -> float:
        h = self.h
        base = h["alpha"][m] + h["beta"][m] * s ** h["gamma"][m]
        if m == h["congested"] and qlen > h["qstar"]:
            base *= 1.0 + h["cong"] * (qlen - h["qstar"])
        return max(0.5, base * (1.0 + self.c["service_noise"] * z))


    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        K, M = int(c["inner"]), len(c["line_names"])
        ua = rng.random((K, 2))
        zs = rng.standard_normal((K, M))
        done, log, cost = 0, [], 0.0
        for k in range(K):
            if ua[k, 0] < c["arrival_p"]:
                lo, hi = c["size"]
                size = round(lo + (hi - lo) * float(ua[k, 1]), 4)
                m = None
                if run is not None:
                    out = run({"size": size, "queues": [len(q) for q in st["lines"]],
                               "sizes": [[j["size"] for j in q] for q in st["lines"]], "step": k, "tick": t},
                              step=k, tick=t)

                    v = field(out, "line")
                    if v is not None and float(v).is_integer() and 0 <= int(v) < M:
                        m = int(v)
                if m is None:
                    st["credit"] = [cr + sp for cr, sp in zip(st["credit"], st["split"])]
                    m = max(range(M), key=lambda i: (st["credit"][i], -i))
                    st["credit"][m] -= 1.0
                st["lines"][m].append({"id": int(st["next"]), "size": size, "age": 0, "left": None})
                st["next"] += 1
                st["arrived"] += 1
            for m in range(M):
                q = st["lines"][m]
                if q and q[0]["left"] is None:
                    q[0]["left"] = round(self.service_time(m, q[0]["size"], len(q), float(zs[k, m])), 6)
                    q[0]["service"] = q[0]["left"]
                if q:
                    q[0]["left"] = round(q[0]["left"] - 1.0, 6)
                    if q[0]["left"] <= 1e-9:
                        j = q.pop(0)
                        done += 1
                        st["revenue_due"] = st.get("revenue_due", 0.0) + c["price_base"] + c["price_size"] * j["size"]
                        log.append({"kind": "completed", "line": c["line_names"][m], "size": j["size"],
                                    "service_steps": round(j["service"], 4), "sojourn_steps": j["age"] + 1})
            for q in st["lines"]:
                for j in q:
                    j["age"] += 1
                cost += c["hold_cost"] * len(q)
            for m, q in enumerate(st["lines"]):
                keep = [j for j in q if j["age"] <= c["max_wait"] or (j is q[0] and j["left"] is not None)]
                st["dropped"] += len(q) - len(keep)
                cost += c["drop_cost"] * (len(q) - len(keep))
                st["lines"][m] = keep
        st["done"] += done
        rev, st["revenue_due"] = st.get("revenue_due", 0.0), 0.0
        ob = {"numbers": {f"{n}_jobs": float(len(q)) for n, q in zip(c["line_names"], st["lines"])},
              "entities": log}
        ob["numbers"]["completed_last_tick"] = float(done)
        return {"obs": ob, "revenue": rev, "cost": cost}

    def conservation(self, before, after, out):
        n = after["done"] + after["dropped"] + sum(len(q) for q in after["lines"])
        return [] if n == after["arrived"] else ["jobs are not conserved"]

    def latent(self, st):
        return [float(len(q)) for q in st["lines"]]


# ---------------------------------------------------------------------- fleet (8.6.2)
@scenario
class Fleet(Scenario):
    """Machines accumulate hidden wear at machine-specific rates; past a hidden level the failure hazard rises
    steeply; a failure stops a machine for several ticks and costs a repair (wear then starts afresh). Wear shows
    only through noisy vibration readings with machine-specific sensitivity. Preventive service takes a machine
    off for one tick and removes most (a hidden fraction) of its wear; a crew can service a few machines per tick.
    Benefits of service appear only later, as failures that do not happen. The manual interface services one
    machine per tick; a deployed policy is called every tick for the whole fleet."""
    name = "fleet"
    doc = ("act(observation, caller) is called once per tick with observation = {\"vibration\": [one reading per "
           "machine, last tick], \"since_service\": [ticks], \"down\": [ticks each machine is still down], \"crew\": "
           "n, \"tick\": t} and returns {\"service\": [machine indices]} (at most `crew`; machines that are down are "
           "skipped). If it fails, nothing is serviced by the program that tick.")
    motifs_ = ("stock", "threshold", "delay", "measurement_noise", "partial_observability", "path_dependence")
    edges_ = (("usage", "wear", "accumulate", 1), ("wear", "wear", "stock", 1), ("wear", "hazard", "threshold", 0),
              ("hazard", "failure", "causal", 0), ("failure", "downtime", "delay", 2), ("wear", "vibration", "sensor", 0),
              ("service", "wear", "reset", 1), ("failure", "wear", "reset", 2))

    def manual(self, state):
        names = list(self.c["machine_names"])
        return {"discrete": A.discrete_schema({"service": names})}, {"decisions": {"service": names}}, \
            {"fleet": "every machine's last vibration reading, ticks since service and downtime left"}

    def init_state(self, rng):
        M = len(self.c["machine_names"])
        w = [float(x) for x in self.h["w_init"]]
        return {"w": w, "down": [0] * M, "since": [0] * M, "vib": [1.0] * M, "want": [], "failures": 0,
                "services": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "discrete":
            i = self.c["machine_names"].index(action["choice"])
            if st["down"][i] > 0:
                raise ActionError(f"{action['choice']} is down")
            if i not in st["want"]:
                st["want"].append(i)
            return {"service_scheduled": action["choice"]}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"vibration": list(st["vib"]), "since_service": list(st["since"]), "down": list(st["down"])}

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        M = len(c["machine_names"])
        uf = rng.random(M)
        zv = rng.standard_normal(M)
        want = list(st["want"])[:int(c["crew"])]
        st["want"] = []
        if run is not None:
            out = run({"vibration": list(st["vib"]), "since_service": list(st["since"]), "down": list(st["down"]),
                       "crew": int(c["crew"]), "tick": t}, step=0, tick=t)
            if out is not None:
                for i in index_list(out, "service", set(range(M)), int(c["crew"])):
                    if i not in want and len(want) < int(c["crew"]):
                        want.append(i)
        rev, cost, fails = 0.0, 0.0, []
        for i in range(M):
            if st["down"][i] > 0:
                st["down"][i] -= 1
                if st["down"][i] == 0 and st["w"][i] < 0:
                    st["w"][i] = 0.0                     # repaired
                continue
            if i in want:
                st["w"][i] *= 1.0 - h["eff"]
                st["since"][i] = 0
                st["services"] += 1
                cost += c["service_cost"]
                continue
            rev += c["output"]
            st["w"][i] += h["rate"][i]
            st["since"][i] += 1
            p = h["p0"] + h["k"] * max(0.0, st["w"][i] - h["w0"]) ** 2
            if uf[i] < p:
                fails.append(c["machine_names"][i])
                st["failures"] += 1
                st["down"][i] = int(c["repair_ticks"])
                st["w"][i] = -1.0                        # marker: reset when the repair is done
                cost += c["repair_cost"]
        st["vib"] = [round(1.0 + h["sens"][i] * 3.0 * max(0.0, st["w"][i]) ** 1.5 + c["vib_sd"] * float(zv[i]), 6)
                     if st["down"][i] == 0 else 0.0 for i in range(M)]
        ob = {"numbers": {f"{n}_vibration": v for n, v in zip(c["machine_names"], st["vib"])},
              "arrays": {"down": [float(x) for x in st["down"]], "since_service": [float(x) for x in st["since"]]},
              "text": [f"{n} failed" for n in fails]}
        return {"obs": ob, "revenue": rev, "cost": cost}

    def latent(self, st):
        return [max(0.0, float(x)) for x in st["w"]]


# ---------------------------------------------------------------------- cascade_guard (8.3.5)
@scenario
class CascadeGuard(Scenario):
    """Units on a ring carry fluctuating demand. A unit's true capacity is its public rating times a hidden factor;
    loaded past it, the unit trips and is out for several inner steps, and its demand moves to the nearest
    working units on either side - which can trip them in turn (a cascade). Shed demand is not served (lost
    revenue) but protects units. The manual lever sets the built-in protection margin (shed what exceeds margin x
    rating); a deployed guard program is called every inner step."""
    name = "cascade_guard"
    doc = ("act(observation, caller) is called every inner step with observation = {\"loads\": [demand on each "
           "unit, redistribution included], \"up\": [1 if working], \"ratings\": [...], \"step\": k, \"tick\": t} and "
           "returns {\"shed\": [fraction in [0, 1] for each unit]}. If it fails, the built-in protection is used.")
    motifs_ = ("cascading_failure", "threshold", "network_effect", "partial_observability", "measurement_noise",
               "delay")
    edges_ = (("demand", "load", "causal", 0), ("load", "trip", "threshold", 0), ("trip", "neighbour_load", "cascade", 1),
              ("neighbour_load", "trip", "threshold", 0), ("shed", "load", "causal", 0), ("trip", "restart", "delay", 2))

    def manual(self, state):
        c = self.c
        return {"levers": A.levers_schema([{"name": c["margin_name"], "min": 0.6, "max": 1.3, "default": 1.3}])}, \
            {}, {"units": "ratings, working flags and the loads of the last inner step"}

    def init_state(self, rng):
        N = len(self.c["ratings"])
        return {"out": [0] * N, "margin": 1.3, "loads": [0.0] * N, "trips": 0, "served": 0.0, "steps": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "levers":
            v = action["values"].get(self.c["margin_name"])
            if v is not None:
                st["margin"] = float(v)
            return {"set": {self.c["margin_name"]: st["margin"]}}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"ratings": list(self.c["ratings"]), "up": [int(o == 0) for o in st["out"]], "loads": list(st["loads"])}

    def redistribute(self, demand: list, up: list) -> list:
        N = len(demand)
        load = [demand[i] if up[i] else 0.0 for i in range(N)]
        if not any(up):
            return load
        for i in range(N):
            if up[i]:
                continue
            l, r = (i - 1) % N, (i + 1) % N
            while not up[l]:
                l = (l - 1) % N
            while not up[r]:
                r = (r + 1) % N
            load[l] += demand[i] / 2
            load[r] += demand[i] / 2
        return load

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        K, N = int(c["inner"]), len(c["ratings"])
        z = rng.standard_normal((K, N))
        served, trips, cost = 0.0, [], 0.0
        for k in range(K):
            g = t * K + k
            demand = [c["base"][i] * (1.0 + c["amp"] * math.sin(2 * math.pi * g / c["period"] + c["phase"][i]))
                      + c["noise_sd"] * float(z[k, i]) for i in range(N)]
            demand = [max(0.0, x) for x in demand]
            up = [o == 0 for o in st["out"]]
            load = self.redistribute(demand, up)
            shed = None
            if run is not None:
                out = run({"loads": [round(x, 4) for x in load], "up": [int(u) for u in up],
                           "ratings": list(c["ratings"]), "step": k, "tick": t}, step=k, tick=t)
                xs = out.get("shed") if isinstance(out, dict) else None
                if isinstance(xs, list) and len(xs) == N and all(_finite(x) is not None for x in xs):
                    shed = [min(1.0, max(0.0, float(x))) for x in xs]
            if shed is None:
                shed = [max(0.0, x - st["margin"] * r) / x if x > 0 else 0.0 for x, r in zip(load, c["ratings"])]
            for i in range(N):
                if st["out"][i] > 0:
                    st["out"][i] -= 1
                    continue
                s = load[i] * (1.0 - shed[i])
                if s > c["ratings"][i] * h["factor"][i]:
                    st["out"][i] = int(c["restart_steps"])
                    trips.append({"unit": c["unit_names"][i], "load": round(s, 4)})
                    cost += c["trip_cost"]
                else:
                    served += s
            st["loads"] = [round(x, 4) for x in load]
        st["trips"] += len(trips)
        st["served"] += served
        st["steps"] += K
        ob = {"numbers": {"working_units": float(sum(o == 0 for o in st["out"]))},
              "arrays": {"loads": list(st["loads"]), "up": [float(o == 0) for o in st["out"]]},
              "entities": [dict(x, kind="trip") for x in trips],
              "text": [f"{x['unit']} tripped at load {x['load']}" for x in trips]}
        return {"obs": ob, "revenue": c["price"] * served, "cost": cost}

    def latent(self, st):
        return [float(sum(o == 0 for o in st["out"]))]


# ---------------------------------------------------------------------- rollout (8.6.3)
@scenario
class Rollout(Scenario):
    """Users in groups, with public features. An exposed user decides after a hidden delay: adopt (then pays a
    subscription every tick) with a hidden feature-dependent propensity, or reject for good. A group that has not
    yet reached a hidden share of adopters is not ready: exposures there are rejected much more often. Users also
    adopt by word of mouth in proportion to their group's share. The manual interface exposes the next users of
    one group; a deployed targeting program is called every tick for the whole population."""
    name = "rollout"
    doc = ("act(observation, caller) is called once per tick with observation = {\"users\": [{\"id\", \"group\", "
           "\"f1\", \"f2\", \"status\"}, ...] (status: unexposed, pending, adopted or rejected), \"group_share\": "
           "[share of each group that has adopted], \"capacity\": n, \"tick\": t} and returns {\"expose\": [user "
           "ids]} (at most `capacity`, unexposed users only). If it fails, nobody is exposed by it that tick.")
    motifs_ = ("delay", "network_effect", "threshold", "learning_behaviour", "path_dependence", "selection_effect",
               "linear_nonlinear")
    edges_ = (("exposure", "decision", "causal", 2), ("features", "decision", "linear_nonlinear", 0),
              ("group_share", "decision", "threshold", 0), ("decision", "adoption", "delay", 2),
              ("adoption", "group_share", "accumulate", 1), ("group_share", "adoption", "network_effect", 1),
              ("adoption", "revenue", "stock", 1))

    STATUS = ("unexposed", "pending", "adopted", "rejected")

    def manual(self, state):
        g = list(self.c["group_names"])
        return {"discrete": A.discrete_schema({"expose_group": g})}, {"decisions": {"expose_group": g}}, \
            {"users": "every user's group, features and status"}

    def init_state(self, rng):
        c = self.c
        U = len(c["users"])
        status = [0] * U
        for k, g in enumerate(c["group_names"]):
            idx = [i for i, u in enumerate(c["users"]) if u["group"] == g]
            for i in idx[:int(round(self.h["seed"][k] * len(idx)))]:
                status[i] = 2                        # early adopters before the participant arrives
        return {"status": status, "reveal": [-1] * U, "fate": [0] * U, "want": [], "adopters": sum(s == 2 for s in status),
                "exposed": 0}

    def shares(self, st) -> list[float]:
        c = self.c
        out = []
        for g in range(len(c["group_names"])):
            idx = [i for i, u in enumerate(c["users"]) if u["group"] == c["group_names"][g]]
            out.append(sum(st["status"][i] == 2 for i in idx) / max(1, len(idx)))
        return out

    def view(self, st) -> list[dict]:
        return [dict(u, id=i, status=self.STATUS[st["status"][i]]) for i, u in enumerate(self.c["users"])]

    def apply(self, st, action, rng, t):
        if action["kind"] == "discrete":
            g = action["choice"]
            cand = [i for i, u in enumerate(self.c["users"]) if u["group"] == g and st["status"][i] == 0
                    and i not in st["want"]]
            room = int(self.c["capacity"]) - len(st["want"])
            st["want"] += cand[:max(0, room)]
            return {"exposing": cand[:max(0, room)]}
        return super().apply(st, action, rng, t)

    def query(self, st, target, args, rng, t):
        return {"users": self.view(st), "group_share": [round(x, 4) for x in self.shares(st)]}

    def advance(self, st, rng, t, run):
        c, h = self.c, self.h
        U, cap = len(c["users"]), int(c["capacity"])
        u_org, u_fate, u_delay = rng.random(U), rng.random(U), rng.random(U)
        share = self.shares(st)
        want = [i for i in st["want"] if st["status"][i] == 0][:cap]
        st["want"] = []
        if run is not None:
            out = run({"users": self.view(st), "group_share": [round(x, 4) for x in share], "capacity": cap,
                       "tick": t}, step=0, tick=t)
            if out is not None:
                for i in index_list(out, "expose", {i for i in range(U) if st["status"][i] == 0}, cap):
                    if i not in want and len(want) < cap:
                        want.append(i)
        cost = c["exposure_cost"] * len(want)
        gi = {g: k for k, g in enumerate(c["group_names"])}
        for i in want:
            u = c["users"][i]
            k = gi[u["group"]]
            p = 1.0 / (1.0 + math.exp(-(h["w"][0] * u["f1"] + h["w"][1] * u["f2"] + h["bias"][k])))
            if share[k] < h["tau"][k]:
                p *= 1.0 - h["q"]
            st["status"][i] = 1
            st["fate"][i] = 2 if u_fate[i] < p else 3
            st["reveal"][i] = t + 1 + int(-math.log(max(1e-12, 1.0 - u_delay[i])) * h["delay"])
            st["exposed"] += 1
        for i in range(U):
            if st["status"][i] == 1 and st["reveal"][i] <= t:
                st["status"][i] = st["fate"][i]
            elif st["status"][i] == 0 and i not in want:
                k = gi[c["users"][i]["group"]]
                if u_org[i] < h["org0"] + h["org1"] * share[k]:
                    st["status"][i] = 2
        st["adopters"] = sum(s == 2 for s in st["status"])
        rev = c["fee"] * st["adopters"]
        sh = self.shares(st)
        ob = {"numbers": {f"{g}_share": round(x, 4) for g, x in zip(c["group_names"], sh)},
              "entities": [dict(x, kind="user") for x in self.view(st) if x["status"] != "unexposed"]}
        ob["numbers"]["adopters"] = float(st["adopters"])
        return {"obs": ob, "revenue": rev, "cost": cost}

    def latent(self, st):
        return [float(st["adopters"])]


# ====================================================================== the engine
@register
class ProgramExec(RapidEngine):
    kind = "program_exec"
    handles_programs = True          # `program` actions reach `apply` (the world runs them), see env.step
    SECRET_KEYS = ("hidden",)        # config keys whose numbers a participant must never receive

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = SCENARIOS[config["scenario"]](config)
        if type(self.sc).latent is not Scenario.latent:
            self.latent = self._latent          # hidden state for the informativeness check (evaluator side)

    # ------------------------------------------------------------------ public surface
    def action_space(self, agent, state=None):
        c = self.config
        kinds, limits, q = self.sc.manual(state)
        kinds = dict(kinds)
        kinds["program"] = program_kind(self.horizon, self.sc.doc, self.sc.submission)
        q = dict(q)
        q.update({"tools": "the action kinds available to you", "history": "your last observations (args: n)",
                  "spec": "the public description of this world", "program_api": "what act() receives and returns"})
        if not self.sc.submission:
            q["program_status"] = "the deployed program: ticks left, calls, errors and compute of the last tick"
        kinds["query"] = A.query_schema(q)
        inner = sorted(k for k in kinds if k not in ("program",))
        kinds["sequence"] = A.sequence_schema(inner)
        limits = dict(limits)
        limits["sequence_kinds"] = inner
        return A.ActionSpace(kinds, limits)

    # ------------------------------------------------------------------ state
    def init_state(self, rng):
        st = self.sc.init_state(rng)
        st.update({"prog": None, "deploy_left": 0, "memory": None, "pstat": {"calls": 0, "errors": 0, "lines": 0},
                   "compute_due": 0.0})
        return st

    # ------------------------------------------------------------------ actions
    def apply(self, st, agent, action, rng, t):
        k = action["kind"]
        if k == "program":
            if self.sc.submission:
                res, lines = self.sc.submit(st, action["source"], rng, t)
                st["compute_due"] += self.config["compute_price"] * lines / 1000.0
                return res
            st["prog"], st["deploy_left"], st["memory"] = action["source"], int(action["ticks"]), None
            return {"deployed": True, "ticks": int(action["ticks"]), "bytes": len(action["source"])}
        if k == "query":
            tg = action["target"]
            if tg == "program_api":
                return {"program_api": self.sc.doc}
            if tg == "program_status":
                return {"deployed": st["prog"] is not None, "ticks_left": int(st["deploy_left"]), **st["pstat"]}
            return self.sc.query(st, tg, action.get("args") or {}, rng, t)
        return self.sc.apply(st, action, rng, t)

    # ------------------------------------------------------------------ dynamics
    def advance(self, st, rng, t):
        c = self.config
        run = ProgramRunner(st["prog"], st["memory"]) if st["prog"] is not None else None
        try:
            out = self.sc.advance(st, rng, t, run)
        finally:
            if run is not None:
                run.close()
        ob = out["obs"]
        res = dict(ob.get("resources", {}))
        cost = float(out["cost"]) + st["compute_due"]
        st["compute_due"] = 0.0
        text = list(ob.get("text", []))
        if run is not None:
            cost += c["compute_price"] * run.lines / 1000.0
            st["memory"] = run.memory
            st["pstat"] = {"calls": run.calls, "errors": run.errors, "lines": run.lines}
            st["deploy_left"] -= 1
            res.update({"program_ticks_left": int(st["deploy_left"]), "program_calls_last_tick": run.calls,
                        "program_errors_last_tick": run.errors, "compute_lines_last_tick": run.lines})
            if run.first_error:
                text.append(f"program error: {run.first_error}")
            if st["deploy_left"] <= 0:
                st["prog"], st["memory"] = None, None
                text.append("the program's deployment ended")
        ob = dict(ob, resources=res, text=text)
        a = self.agents[0]
        return {"obs": {a: ob}, "revenue": {a: float(out["revenue"])}, "cost": {a: cost}, "info": {}}

    def history(self, state):
        ex = self.config.get("examples")
        if ex:
            return {"text": [f"worked examples: {json.dumps(ex)}"]}
        return {}

    # ------------------------------------------------------------------ evaluator side
    def truth(self):
        tr = self.sc.truth()
        tr.update({"engine": self.kind, "scenario": self.sc.name, "motifs": self.motifs()})
        return tr

    def motifs(self):
        return sorted(self.sc.motifs_)

    def _latent(self, state):
        return np.asarray(self.sc.latent(state), dtype=float)

    def outcome(self, state, ledger):
        a = self.agents[0]
        rev, cost = float(ledger["revenue"].get(a, 0.0)), float(ledger["cost"].get(a, 0.0))
        out = {"net": rev - cost, "revenue": rev, "cost": cost}
        out.update(self.sc.outcome(state, rev - cost))
        return out

    def conservation(self, before, after, out):
        errs = []
        for a in self.agents:
            if out["revenue"][a] < 0 or out["cost"][a] < 0:
                errs.append("negative money flow")
        return errs + self.sc.conservation(before, after, out)

    def check_state(self, state):
        errs = []
        for k, v in state.items():
            if isinstance(v, float) and (not math.isfinite(v) or abs(v) > 1e12):
                errs.append(f"state {k} is not finite or unbounded")
        return errs
