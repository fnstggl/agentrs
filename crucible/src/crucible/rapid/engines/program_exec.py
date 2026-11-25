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
world). Work is accounted deterministically - executed Python lines, `SandboxConfig.count_lines` - and charged as
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
        self.lines += int(r.get("work_lines") or 0)
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
        """Observational data: every tick one record seen in the field (inside the probe box) and its noisy
        value arrive for free; probes are the interventions."""
        c = self.c
        fa, fb, fc = c["fields"]
        z, e = rng.random(3), float(rng.standard_normal())
        (alo, ahi), (blo, bhi) = c["box"][fa], c["box"][fb]
        rec = {fa: round(alo + (ahi - alo) * float(z[0]), 4), fb: round(blo + (bhi - blo) * float(z[1]), 4),
               fc: c["categories"][min(len(c["categories"]) - 1, int(z[2] * len(c["categories"])))]}
        v = self.value(rec) + c["probe_sd"] * e
        nums = {fa: rec[fa], fb: rec[fb], c["value_name"]: round(v, 6)}
        return {"obs": {"numbers": nums, "entities": [dict(rec, kind="field_record", value=round(v, 6))],
                        "resources": {"budget": st["budget"]}}, "revenue": 0.0, "cost": 0.0}

    def outcome(self, st, net):
        return {"value": float(st["holdout_pass"]), "holdout_pass": float(st["holdout_pass"]),
                "validation_pass": float(st["validation_pass"]), "submissions": int(st["submissions"])}

    def conservation(self, before, after, out):
        return ["budget grew"] if after["budget"] > before["budget"] + 1e-9 else []


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
