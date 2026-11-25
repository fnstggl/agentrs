"""Self-generated experiments (module `exp`, host `exp`; MASTER_SPEC 11.4).

An agent states a testable prediction about a quantity it can measure, sets a deadline, acts, and later retrieves
what actually happened. This extends the 4.0 self-goal interface (`goals.propose`, seven predefined metrics) to any
quantity the agent can read through an ordinary **read-only tool** it is allowed to call: a balance from
`bank.accounts`, a facility's revenue from `rig.read`, a service's requests from `cloud.status`, a count of search
results, ... The measurement is made by calling that tool *as the agent*, so it can never reveal more than the agent
could see by calling it itself; the module adds no information, only a record.

    exp.register {hypothesis, measure, prediction, deadline_in_s, plan?, samples?}
        measure    = {"tool": "bank.accounts", "args": {}, "path": [], "where": {"owner": "$self"},
                      "field": "balance", "aggregate": "sum"}
        prediction = {"op": ">=" | "<=" | ">" | "<" | "==" | "!=", "value": number}
                   | {"op": "increase" | "decrease" | "change", "min_change": number}
    exp.result {experiment}  -> the evidence: baseline, samples, measured value, whether the prediction held,
                                how many actions the agent took during the experiment, times
    exp.mine {status?}, exp.cancel {experiment}

At the deadline a scheduled process measures again and records `exp.resolved` (visible only to the agent). The
module never pays, rewards, scores or ranks anything: it returns evidence. Whether a subgoal was worth setting is
for the learner to judge; an easy prediction that holds earns exactly what a hard one earns - nothing. Experiments
are private (events are visible to their owner only) and the evaluator ignores this table.
"""
from __future__ import annotations

import math
from typing import Any

from ..core import schema as S
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, process, register_module

OPS = (">=", "<=", ">", "<", "==", "!=", "increase", "decrease", "change")
AGGREGATES = ("value", "sum", "count", "min", "max", "mean", "first", "last")
MAX_OPEN = 32
FORBIDDEN_HOSTS = ("admin", "exp")

MEASURE = {"type": "object", "properties": {
    "tool": S.s_str("a read-only tool you may call, e.g. bank.accounts or rig.read", maxLength=80),
    "args": S.s_any_obj("arguments for that tool"),
    "path": {"type": "array", "items": {"type": ["string", "integer"]}, "maxItems": 12,
             "description": "keys / list indices to follow in the tool's result"},
    "where": S.s_any_obj("keep only list elements whose fields equal these values ('$self' = you)"),
    "field": S.s_str("field to read from the selected element(s)", maxLength=80),
    "aggregate": S.s_enum(list(AGGREGATES), "how to combine several selected values (default: sum for a list)")},
    "required": ["tool"], "additionalProperties": False}
PREDICTION = {"type": "object", "properties": {
    "op": S.s_enum(list(OPS)), "value": {"type": "number"}, "min_change": {"type": "number"}},
    "required": ["op"], "additionalProperties": False}


def select(result: Any, measure: dict, owner: str) -> float:
    """The number a measure picks out of a tool result. Raises InvalidAction if it is not a finite number."""
    node = result
    for seg in measure.get("path") or []:
        if isinstance(node, dict) and isinstance(seg, str) and seg in node:
            node = node[seg]
        elif isinstance(node, list) and isinstance(seg, int) and -len(node) <= seg < len(node):
            node = node[seg]
        else:
            raise InvalidAction(f"measure path segment {seg!r} not found in the result")
    where = {k: (owner if v == "$self" else v) for k, v in (measure.get("where") or {}).items()}
    field = measure.get("field")
    agg = measure.get("aggregate")
    if isinstance(node, list):
        items = [x for x in node if not where or (isinstance(x, dict) and all(x.get(k) == v for k, v in where.items()))]
        if agg == "count":
            return float(len(items))
        vals = [x.get(field) if (field and isinstance(x, dict)) else x for x in items]
        vals = [_num(v) for v in vals]
        if any(v is None for v in vals):
            raise InvalidAction("the selected values are not all numbers")
        agg = agg or "sum"
        if agg in ("sum", "value"):
            return float(math.fsum(vals))
        if not vals:
            raise InvalidAction(f"nothing selected to take the {agg} of")
        return float({"min": min, "max": max, "first": lambda v: v[0], "last": lambda v: v[-1],
                      "mean": lambda v: math.fsum(v) / len(v)}[agg](vals))
    if field:
        if not isinstance(node, dict) or field not in node:
            raise InvalidAction(f"field {field!r} not found in the result")
        node = node[field]
    v = _num(node)
    if v is None:
        raise InvalidAction("the measured value is not a number")
    return v


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def holds(prediction: dict, measured: float, baseline: float) -> bool:
    op = prediction["op"]
    if op in ("increase", "decrease", "change"):
        d = measured - baseline
        m = float(prediction.get("min_change", 0.0))
        return d > m if op == "increase" else (-d > m if op == "decrease" else abs(d) > m)
    v = float(prediction["value"])
    return {">=": measured >= v, "<=": measured <= v, ">": measured > v, "<": measured < v, "==": measured == v,
            "!=": measured != v}[op]


@register_module
class Experiments(Module):
    name = "exp"
    host = "exp"
    version = "1"
    depends = ("identity",)

    def install(self, world):
        world.state.declare_table("experiments", index=("owner", "status"))

    # ------------------------------------------------------------------ measuring
    def _endpoint(self, ctx, tool: str):
        ep = ctx.world.resolve(tool, ctx.actor)
        if ep.host in FORBIDDEN_HOSTS:
            raise Forbidden(f"{tool} cannot be used as a measure")
        if not ep.read_only:
            raise Forbidden(f"{tool} is not a read-only tool: a measure may only read")
        return ep

    def _measure(self, ctx, measure: dict, owner: str) -> float:
        res = ctx.call(measure["tool"], dict(measure.get("args") or {}), as_principal=owner)
        return select(res, measure, owner)

    # ------------------------------------------------------------------ endpoints
    @endpoint("register", S.obj({"hypothesis": S.s_str("what you expect and why, in words", maxLength=2000),
                                 "measure": MEASURE, "prediction": PREDICTION,
                                 "deadline_in_s": S.s_int("seconds from now until the result is measured",
                                                          minimum=60, maximum=365 * 86400),
                                 "plan": S.s_str("what you will do, for your own records", maxLength=2000),
                                 "samples": S.s_int("intermediate measurements before the deadline", minimum=0,
                                                    maximum=48)},
                                ["hypothesis", "measure", "prediction", "deadline_in_s"]), cost_s=30)
    def ep_register(self, ctx, hypothesis, measure, prediction, deadline_in_s, plan="", samples=0):
        """Register a private experiment: a testable prediction about a quantity you can read with a read-only
        tool, measured now (baseline) and again at the deadline. Example: measure {"tool": "bank.accounts",
        "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"} with prediction {"op": "increase",
        "min_change": 10000}. Returns the experiment id and the baseline. The result is evidence, never a reward."""
        if prediction["op"] not in ("increase", "decrease", "change") and "value" not in prediction:
            raise InvalidAction(f"prediction op {prediction['op']!r} needs a 'value'")
        self._endpoint(ctx, measure["tool"])
        open_n = sum(1 for g in ctx.state.find("experiments", "owner", ctx.actor) if g["status"] == "open")
        if open_n >= MAX_OPEN:
            raise InvalidAction(f"at most {MAX_OPEN} open experiments")
        baseline = self._measure(ctx, measure, ctx.actor)
        eid = ctx.state.new_id("exp")
        due = ctx.now + int(deadline_in_s)
        p = ctx.state.get("principals", ctx.actor) or {}
        ctx.emit("exp.registered", {"id": eid, "owner": ctx.actor, "hypothesis": hypothesis, "measure": measure,
                                    "prediction": prediction, "plan": plan, "baseline": baseline, "due": due,
                                    "actions_at_start": int(p.get("actions", 0)), "samples": int(samples)},
                 vis=(ctx.actor,))
        ctx.schedule(due, "exp.resolve", {"id": eid}, prio=85)
        for k in range(1, int(samples) + 1):
            ctx.schedule(ctx.now + int(deadline_in_s) * k // (int(samples) + 1), "exp.sample", {"id": eid}, prio=85)
        return {"experiment": eid, "baseline": baseline, "due": due}

    @endpoint("result", S.obj({"experiment": S.s_str(maxLength=40)}, ["experiment"]), read_only=True)
    def ep_result(self, ctx, experiment):
        """The evidence of one of your experiments: the baseline, intermediate samples, the measured value at the
        deadline, whether the prediction held, how many actions you took in between, and the times."""
        g = ctx.state.get("experiments", experiment)
        if g is None or g["owner"] != ctx.actor:
            raise NotFound("no such experiment")
        return dict(g, experiment=experiment)

    @endpoint("mine", S.obj({"status": S.s_enum(["open", "resolved", "failed", "cancelled"])}), read_only=True)
    def ep_mine(self, ctx, status=None):
        """Your experiments (all, or those with a status)."""
        return [dict(ctx.state.tables["experiments"][i], experiment=i)
                for i in ctx.state.find_ids("experiments", "owner", ctx.actor)
                if status is None or ctx.state.tables["experiments"][i]["status"] == status]

    @endpoint("cancel", S.obj({"experiment": S.s_str(maxLength=40)}, ["experiment"]), cost_s=5)
    def ep_cancel(self, ctx, experiment):
        """Cancel an open experiment of yours; it will not be measured again."""
        g = ctx.state.get("experiments", experiment)
        if g is None or g["owner"] != ctx.actor:
            raise NotFound("no such experiment")
        if g["status"] != "open":
            raise InvalidAction(f"experiment is {g['status']}")
        ctx.emit("exp.cancelled", {"id": experiment}, vis=(ctx.actor,))
        return {"experiment": experiment, "status": "cancelled"}

    # ------------------------------------------------------------------ processes
    @process("sample")
    def pr_sample(self, ctx, id):
        g = ctx.state.get("experiments", id)
        if g is None or g["status"] != "open":
            return
        try:
            v = self._measure(ctx, g["measure"], g["owner"])
        except CrucibleError as e:
            ctx.emit("exp.sampled", {"id": id, "value": None, "error": e.as_dict()}, vis=(g["owner"],))
            return
        ctx.emit("exp.sampled", {"id": id, "value": v}, vis=(g["owner"],))

    @process("resolve")
    def pr_resolve(self, ctx, id):
        g = ctx.state.get("experiments", id)
        if g is None or g["status"] != "open":
            return
        p = ctx.state.get("principals", g["owner"]) or {}
        acts = int(p.get("actions", 0)) - int(g["actions_at_start"])
        try:
            v = self._measure(ctx, g["measure"], g["owner"])
        except CrucibleError as e:
            ctx.emit("exp.resolved", {"id": id, "measured": None, "held": None, "error": e.as_dict(),
                                      "actions_during": acts}, vis=(g["owner"],))
            return
        ctx.emit("exp.resolved", {"id": id, "measured": v, "held": holds(g["prediction"], v, g["baseline"]),
                                  "actions_during": acts}, vis=(g["owner"],))

    # ------------------------------------------------------------------ appliers
    @applier("exp.registered")
    def ap_registered(self, ev):
        p = ev.payload
        self.world.state.insert("experiments", p["id"], {
            "owner": p["owner"], "hypothesis": p["hypothesis"], "measure": p["measure"], "prediction": p["prediction"],
            "plan": p["plan"], "baseline": p["baseline"], "baseline_t": ev.t, "due": p["due"],
            "actions_at_start": p["actions_at_start"], "status": "open", "samples": [], "measured": None,
            "measured_t": None, "held": None, "actions_during": None, "error": None})

    @applier("exp.sampled")
    def ap_sampled(self, ev):
        g = self.world.state.tables["experiments"][ev.payload["id"]]
        self.world.state.patch("experiments", ev.payload["id"],
                               samples=list(g["samples"]) + [{"t": ev.t, "value": ev.payload.get("value")}])

    @applier("exp.resolved")
    def ap_resolved(self, ev):
        p = ev.payload
        self.world.state.patch("experiments", p["id"], status="failed" if p.get("error") else "resolved",
                               measured=p["measured"], measured_t=ev.t, held=p["held"],
                               actions_during=p["actions_during"], error=p.get("error"))

    @applier("exp.cancelled")
    def ap_cancelled(self, ev):
        self.world.state.patch("experiments", ev.payload["id"], status="cancelled")

    # ------------------------------------------------------------------ observation & invariants
    def observe(self, world, actor):
        rows = [world.state.tables["experiments"][i] for i in world.state.find_ids("experiments", "owner", actor)]
        if not rows:
            return None
        return {"open": sum(1 for r in rows if r["status"] == "open"),
                "resolved": sum(1 for r in rows if r["status"] == "resolved"),
                "next_due": min((r["due"] for r in rows if r["status"] == "open"), default=None)}

    def check_invariants(self, world):
        errs = []
        for i, g in world.state.scan("experiments"):
            if g["status"] == "resolved" and g["measured_t"] is None:
                errs.append(f"experiment {i} resolved without a measurement time")
            if g["status"] == "open" and g["due"] < world.now:
                errs.append(f"experiment {i} overdue")
        return errs
