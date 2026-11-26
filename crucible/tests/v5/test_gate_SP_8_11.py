"""Gate SP-8.11 - the Tier 1 action interface covers ten action types (workstream H).

An action type counts when an admitted family offers it and taking it measurably changes the world or what the
participant learns, compared with not taking it (each effect is checked on a real world, not on a schema)."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import report as RP
from crucible.rapid.actions import SPEC_ACTION_TYPES, spec_coverage
from crucible.sandbox.runner import capabilities

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def eff_graph_edit():
    env = F.make("net_repair", 0)
    names = env._config["node_names"]
    pair = next((a, b) for a in range(len(names)) for b in range(len(names)) if a < b and
                [a, b] not in env._config["edges"] and [b, a] not in env._config["edges"])
    r = env.step({"kind": "graph_edit", "op": "add_edge", "src": names[pair[0]], "dst": names[pair[1]]})
    return r["result"]["ok"] and any(e["id"] == "new0" for e in r["observation"]["entities"])


def eff_discrete():
    env = F.make("des_maintenance", 0)
    m = env._config["machine_names"][0]
    r = env.step({"kind": "discrete", "decision": "service", "choice": m})
    return r["result"]["ok"] and env._state["services"] == 1


def eff_sequence():
    env = F.make("posm_unlock", 0)
    c = env._config
    steps = [{"kind": "discrete", "decision": c["decision_name"], "choice": c["options"][k]} for k in c["code"]]
    env.step({"kind": "sequence", "steps": steps})
    r = env.step(None)
    return r["observation"]["resources"]["revenue_cents_last_tick"] > 0 and env._state["s"][0] == c["stages"]


def eff_tool_discovery():
    env = F.make("srch_tools", 0)
    before = env.action_space().limits["decisions"]["use"]
    r = env.step({"kind": "query", "target": "catalogue"})
    after = env.action_space().limits["decisions"]["use"]
    listed = env.step({"kind": "query", "target": "tools"})["result"]["results"]["tools"]
    return (r["result"]["ok"] and len(after) == len(before) + 1 and r["observation"]["tools"]
            and any(t["name"] == "discrete" for t in listed))


def eff_experiment():
    env = F.make("sci_functional_law", 0)
    sp = env.action_space()
    var = sorted(sp.kinds["experiment"]["properties"]["design"]["properties"]["set"]["properties"])
    g = env._config["objective"]["gauge"]
    b0 = float(env._state["budget"][0])
    r = env.step({"kind": "experiment", "design": {"set": {var[0]: 5.0, var[1]: 1.0}, "measure": [g], "ticks": 2,
                                                   "replicates": 4}})
    m = r["result"]["results"]["measured"][g]
    return r["result"]["ok"] and m["n"] == 8 and m["mean"] is not None and float(env._state["budget"][0]) < b0


def eff_query():
    env = F.make("des_bottleneck", 0)
    for _ in range(5):
        env.step(None)
    n = env._config["station_names"][0]
    r = env.step({"kind": "query", "target": f"station:{n}"})
    return r["result"]["ok"] and "queue" in r["result"]["results"]


def eff_message():
    env = F.make("coop_split_info", 0)
    r = env.step({"a0": {"kind": "message", "to": "a1", "content": {"x": 3}, "text": "x is about 3"}})
    got = [e for e in r["observations"]["a1"]["entities"] if e.get("kind") == "message"]
    none = [e for e in r["observations"]["a0"]["entities"] if e.get("kind") == "message"]
    return bool(got) and got[0]["content"] == {"x": 3} and not none


def eff_program():
    if capabilities()["landlock_abi"] < 1 or not capabilities()["seccomp"]:
        raise RuntimeError("the sandbox (Landlock + seccomp) is required for program actions")
    env = F.make("eng_delayed_controller", 0)
    h = next(iter(env.action_space().kinds["levers"]["properties"]["values"]["properties"]))
    src = ("def act(observation, caller):\n"
           f"    return {{'kind': 'levers', 'values': {{'{h}': 7.5}}}}\n")
    r = env.step({"kind": "program", "source": src, "ticks": 5})
    return r["result"]["ok"] and r["result"]["program"]["ticks_run"] == 5 and env.t == 5 and \
        float(env._state["lev"][0, 0]) == 7.5


def eff_allocate():
    env = F.make("des_bottleneck", 0)
    names = env._config["station_names"]
    r = env.step({"kind": "allocate", "amounts": {names[0]: 6.0, names[1]: 2.0}})
    return r["result"]["ok"] and env._state["servers"][0] == 6 and r["observation"]["arrays"]["servers"][0] == 6.0


def eff_coordinate():
    from crucible.rapid.engines.search import cell_name
    env = F.make("coop_split_info", 0)
    tx, ty = (int(v) for v in env._state["target"])
    plan = {"cell": cell_name(tx, ty)}
    solo = F.make("coop_split_info", 0)
    r1 = solo.step({"a0": {"kind": "coordinate", "with": ["a1"], "plan": plan}})
    r2 = env.step({"a0": {"kind": "coordinate", "with": ["a1"], "plan": plan},
                   "a1": {"kind": "coordinate", "with": ["a0"], "plan": plan}})
    return (not r1["results"]["a0"]["result"]["agreed"]) and r2["results"]["a0"]["result"]["agreed"] and \
        r2["results"]["a0"]["result"]["hit"]


EFFECTS = {"graph modifications": eff_graph_edit, "discrete decisions": eff_discrete,
           "sequential actions": eff_sequence, "tool discovery": eff_tool_discovery,
           "experiment design": eff_experiment, "queries and observations": eff_query,
           "communication": eff_message, "executable program construction": eff_program,
           "resource allocation": eff_allocate, "multi-agent coordination": eff_coordinate}


def test_SP_8_11_action_types():
    adm = RP.admission_report(seeds=(0, 1, 2))
    admitted = [n for n, r in adm["reports"].items() if r["admitted"]]
    spaces = [F.make(n, 0).action_space(a) for n in admitted for a in F.make(n, 0).agents()]
    offered = spec_coverage(spaces)
    worked = {k: bool(fn()) for k, fn in EFFECTS.items()}
    hosts = {"graph modifications": "net_repair", "discrete decisions": "des_maintenance",
             "sequential actions": "posm_unlock", "tool discovery": "srch_tools",
             "experiment design": "sci_functional_law", "queries and observations": "des_bottleneck",
             "communication": "coop_split_info", "executable program construction": "eng_delayed_controller",
             "resource allocation": "des_bottleneck", "multi-agent coordination": "coop_split_info"}
    counted = [k for k in SPEC_ACTION_TYPES if offered[k] and worked[k] and hosts[k] in admitted]
    record("SP-8.11", {"action_types": len(counted), "offered_by_admitted_families": sum(offered.values()),
                       "effects_verified": sum(worked.values())},
           notes="Spec type -> kind: " + ", ".join(f"{k}: {v}" for k, v in SPEC_ACTION_TYPES.items()) +
                 f". Not counted: {[k for k in SPEC_ACTION_TYPES if k not in counted]}.")
    assert len(counted) == 10
