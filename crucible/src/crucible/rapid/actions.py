"""The general structured action interface of Tier 1 (MASTER_SPEC Part 8.11, INTERFACE_CONTRACTS section 8).

An action is a JSON object with a `kind` and the fields of that kind. An `ActionSpace` maps each kind an environment
offers to the JSON schema of one action of that kind (the strict schema subset of `crucible.core.schema`, the same
one tool arguments use). The schemas are specialised per environment: the lever schema names that environment's
levers with their ranges, the discrete schema its decisions and options, the query schema its query targets, and so
on. Nothing in a schema is private: names are the neutral public names, never roles.

The ten kinds:

    levers      set named continuous quantities (a partial mapping)
    discrete    pick one option of a named decision
    graph_edit  add or remove an edge or node, set an edge weight, reroute a flow
    sequence    an ordered list of other actions, executed in order within one tick
    query       ask for an observation: a sensor, a probe, a test, the action catalogue (tool discovery), history
    experiment  run a designed trial on a pilot copy of the system: settings to impose, what to measure, how long,
                how many replicates; costs experiment budget, never changes the real system
    message     send structured content or text to another participant (or file a claim with the journal)
    program     submit a controller program (Python source) that runs in the sandbox and acts for a number of ticks
    allocate    split a limited budget over named targets
    coordinate  propose or accept a joint plan; it takes effect only when every party it names commits to it

`SPEC_ACTION_TYPES` maps the ten action types named by Part 8.11 onto these kinds (tool discovery is the `tools`
target of `query`, whose answer also arrives in the observation's `tools`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import schema as S

KINDS = ("levers", "discrete", "graph_edit", "sequence", "query", "experiment", "message", "program", "allocate",
         "coordinate")

SPEC_ACTION_TYPES = {
    "graph modifications": "graph_edit",
    "discrete decisions": "discrete",
    "sequential actions": "sequence",
    "tool discovery": "query:tools",
    "experiment design": "experiment",
    "queries and observations": "query",
    "communication": "message",
    "executable program construction": "program",
    "resource allocation": "allocate",
    "multi-agent coordination": "coordinate",
}

MAX_SEQUENCE = 16
MAX_PROGRAM_BYTES = 20_000
NAME = {"type": "string", "maxLength": 40, "description": "a public name"}


def _kind(name: str, doc: str) -> dict:
    return {"type": "string", "enum": [name], "description": doc}


# ====================================================================== per-kind schema builders
def levers_schema(levers: list[dict]) -> dict:
    """`levers`: [{"name", "min", "max", "default", "integer"?}] -> schema of {"kind": "levers", "values": {...}}."""
    props = {}
    for lv in levers:
        props[lv["name"]] = {"type": "number", "minimum": lv["min"], "maximum": lv["max"],
                             "description": f"default {lv['default']}" + (", rounded to an integer"
                                                                          if lv.get("integer") else "")}
    return S.obj({"kind": _kind("levers", "set continuous levers (any subset)"),
                  "values": S.obj(props, description="lever name -> value")}, ["kind", "values"])


def discrete_schema(decisions: dict[str, list[str]]) -> dict:
    """`decisions`: decision name -> option names."""
    opts = sorted({o for v in decisions.values() for o in v})
    return S.obj({"kind": _kind("discrete", "choose one option of a decision"),
                  "decision": S.s_enum(sorted(decisions), "which decision"),
                  "choice": S.s_enum(opts, "the option chosen (must be one of that decision's options)")},
                 ["kind", "decision", "choice"])


def graph_edit_schema(ops: list[str], nodes: list[str], max_weight: float = 1e6) -> dict:
    return S.obj({"kind": _kind("graph_edit", "modify the graph"),
                  "op": S.s_enum(ops, "the edit"),
                  "src": S.s_enum(nodes, "source node"), "dst": S.s_enum(nodes, "destination node"),
                  "weight": S.s_num("edge weight or capacity (set_weight, add_edge)", minimum=0, maximum=max_weight)},
                 ["kind", "op", "src"])


def sequence_schema(inner: list[str], max_steps: int = MAX_SEQUENCE) -> dict:
    return S.obj({"kind": _kind("sequence", f"several actions in order within one tick; step kinds: {inner}"),
                  "steps": {"type": "array", "minItems": 1, "maxItems": max_steps,
                            "items": {"type": "object", "additionalProperties": True},
                            "description": "each step is a complete action of another kind"}},
                 ["kind", "steps"])


def query_schema(targets: dict[str, str]) -> dict:
    """`targets`: target name -> what it returns."""
    doc = "; ".join(f"{k}: {v}" for k, v in sorted(targets.items()))
    return S.obj({"kind": _kind("query", "ask for an observation"),
                  "target": S.s_enum(sorted(targets), doc),
                  "args": {"type": "object", "additionalProperties": True, "description": "target arguments"}},
                 ["kind", "target"])


def experiment_schema(variables: list[dict], measures: list[str], max_ticks: int, max_replicates: int,
                      cost_per_replicate_tick: float, instruments: list[str] | None = None) -> dict:
    props = {v["name"]: {"type": "number", "minimum": v["min"], "maximum": v["max"]} for v in variables}
    design = {"set": S.obj(props, description="variables imposed in the trial (others stay as they are)"),
              "measure": {"type": "array", "items": S.s_enum(measures), "minItems": 1, "maxItems": len(measures),
                          "description": "what to measure"},
              "ticks": S.s_int("trial length", minimum=1, maximum=max_ticks),
              "replicates": S.s_int("independent replicates", minimum=1, maximum=max_replicates)}
    req = ["set", "measure"]
    if instruments:
        design["instrument"] = S.s_enum(instruments, "measuring instrument (differ in cost and precision)")
    return S.obj({"kind": _kind("experiment", f"a designed trial on a pilot copy; costs {cost_per_replicate_tick} "
                                              "budget per replicate-tick (instrument-dependent where offered)"),
                  "design": S.obj(design, req)}, ["kind", "design"])


def message_schema(recipients: list[str], max_chars: int = 2000) -> dict:
    return S.obj({"kind": _kind("message", "send content to another participant or file it with the journal"),
                  "to": S.s_enum(recipients, "recipient"),
                  "text": S.s_str("free text", maxLength=max_chars),
                  "content": {"type": "object", "additionalProperties": True, "description": "structured content"}},
                 ["kind", "to"])


def program_schema(max_ticks: int, doc: str = "") -> dict:
    return S.obj({"kind": _kind("program", "a controller program run in the sandbox: define act(observation, caller) "
                                           "returning one action dict; it acts once per tick. " + doc),
                  "source": S.s_str("Python source", maxLength=MAX_PROGRAM_BYTES),
                  "ticks": S.s_int("how many ticks the controller runs", minimum=1, maximum=max_ticks)},
                 ["kind", "source", "ticks"])


def allocate_schema(targets: list[str], total: float, what: str = "budget") -> dict:
    props = {t: S.s_num("", minimum=0, maximum=total) for t in targets}
    return S.obj({"kind": _kind("allocate", f"split at most {total} units of {what} over targets"),
                  "amounts": S.obj(props, description=f"target -> amount; the amounts sum to at most {total}")},
                 ["kind", "amounts"])


def coordinate_schema(parties: list[str], plan: dict) -> dict:
    return S.obj({"kind": _kind("coordinate", "propose or accept a joint plan; it is carried out when every party "
                                              "it names submits the same plan in the same tick"),
                  "with": {"type": "array", "items": S.s_enum(parties), "minItems": 1, "maxItems": len(parties),
                           "description": "the other parties"},
                  "plan": S.obj(plan, list(plan), description="the joint plan")},
                 ["kind", "with", "plan"])


# ====================================================================== the space
@dataclass
class ActionSpace:
    """`kinds`: kind name -> JSON schema of one action of that kind. `limits` holds semantic constraints the schema
    subset cannot express (allocation totals, which kinds a sequence may contain)."""
    kinds: dict[str, dict]
    limits: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in self.kinds:
            if k not in KINDS:
                raise ValueError(f"unknown action kind {k!r}")
            errs = S.check_schema(self.kinds[k])
            if errs:
                raise ValueError(f"schema of {k}: {errs}")

    def validate(self, action: Any, depth: int = 0) -> list[str]:
        """Errors (empty when valid). Checks the schema of the action's kind and the semantic limits."""
        if not isinstance(action, dict):
            return ["an action is a JSON object"]
        kind = action.get("kind")
        if kind not in self.kinds:
            return [f"kind {kind!r} is not available here; available: {sorted(self.kinds)}"]
        errs = S.validate(self.kinds[kind], action, "action")
        if errs:
            return errs
        if kind == "sequence":
            if depth >= 1:
                return ["a sequence cannot contain another sequence"]
            inner = self.limits.get("sequence_kinds", [k for k in self.kinds if k not in ("sequence", "program")])
            for i, st in enumerate(action["steps"]):
                if not isinstance(st, dict) or st.get("kind") not in inner:
                    errs.append(f"action.steps[{i}]: kind must be one of {inner}")
                    continue
                errs += [f"steps[{i}]: {e}" for e in self.validate(st, depth + 1)]
        elif kind == "allocate":
            total = float(self.limits.get("allocate_total", float("inf")))
            s = sum(float(v) for v in action["amounts"].values())
            if s > total * (1 + 1e-9):
                errs.append(f"action.amounts: sum {s:g} exceeds the available {total:g}")
        elif kind == "discrete":
            opts = self.limits.get("decisions", {}).get(action["decision"])
            if opts is not None and action["choice"] not in opts:
                errs.append(f"action.choice: {action['choice']!r} is not an option of {action['decision']!r} "
                            f"(options: {opts})")
        elif kind == "graph_edit":
            ops2 = self.limits.get("edge_ops", ("add_edge", "remove_edge", "set_weight", "reroute"))
            if action["op"] in ops2 and "dst" not in action:
                errs.append("action.dst: required for this edit")
            if action.get("dst") is not None and action.get("dst") == action["src"]:
                errs.append("action: src and dst must differ")
        elif kind == "program":
            try:
                compile(action["source"], "<controller>", "exec")
            except SyntaxError as e:
                errs.append(f"action.source: syntax error: {e.msg} (line {e.lineno})")
            if "def act" not in action["source"]:
                errs.append("action.source: must define act(observation, caller)")
        elif kind == "coordinate":
            parties = self.limits.get("parties")
            if parties is not None and any(p not in parties for p in action["with"]):
                errs.append("action.with: unknown party")
        return errs

    def expand(self, action: dict) -> list[dict]:
        """The atomic actions of a (validated) action, in execution order."""
        return list(action["steps"]) if action.get("kind") == "sequence" else [action]

    def tools(self) -> list[dict]:
        """The action kinds as tool specs (what an Observation's `tools` lists)."""
        out = []
        for k in sorted(self.kinds):
            sch = self.kinds[k]
            out.append({"name": k, "description": sch["properties"]["kind"].get("description", ""),
                        "inputSchema": sch})
        return out

    def public(self) -> dict:
        return {"kinds": {k: self.kinds[k] for k in sorted(self.kinds)}, "limits": dict(self.limits)}


def spec_coverage(spaces: list[ActionSpace]) -> dict[str, bool]:
    """Which of the Part 8.11 action types the given spaces offer (`query:tools` needs a `tools` query target)."""
    out = {}
    for item, k in SPEC_ACTION_TYPES.items():
        if k == "query:tools":
            out[item] = any("query" in s.kinds and "tools" in s.kinds["query"]["properties"]["target"]["enum"]
                            for s in spaces)
        else:
            out[item] = any(k in s.kinds for s in spaces)
    return out
