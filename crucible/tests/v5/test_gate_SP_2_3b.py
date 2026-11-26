"""Gate SP-2.3b (MASTER_SPEC Part 2.3): all systems communicate through typed events and approved state interfaces
(no cross-module state writes), and simulation is deterministic wherever possible (the same spec and seed give the
same event hashes in every v5 world kind).

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json): cross_module_state_writes == 0, world_kinds_checked_for_determinism >= 3,
determinism_mismatches == 0. Measured on the real worlds of each v5 world kind (`crucible.substrate.integrity.KINDS`:
the integrated civilisation, the population world, the social internet, the commerce world; a Tier 2 domain when one
is registered; a Tier 1 rapid environment for determinism):

* cross-module state writes: every write to a world's state during the run is attributed to the module whose
  applier performs it (or to the handler, outside any applier) and checked against the declaring owner of the table,
  array, column or kernel metadata it changes (`integrity.trace_writes`). Approved interfaces are the columns
  module's batch events whose recorded writer the column ownership rule admits (an interface only from the module
  that implements it), and the kernel's counters. The metric counts distinct offending (writer, target) pairs, plus
  one per traced world whose own log does not replay to its state (`integrity.replay_check`: a write that bypasses
  the state store's methods, e.g. a nested value of a row mutated in place, leaves no trace but breaks replay).
  Positive control: a module whose applier writes another module's table is found.
* determinism: each kind is built from the same spec and seed in two fresh processes (`python -m
  crucible.substrate.integrity hashes`) and run for the same simulated time; event hash and state digest must be
  equal. A kind counts as checked only if its world ran (events > 0) in both processes.

Typed-event coverage (`bus.coverage`: emitted event types that have a registered schema) is reported alongside.
"""
import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5 import evidence
from crucible.core import schema as S
from crucible.core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import bus, integrity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_3b.py"
HOURS = {"civ": 12.0, "population": 24.0, "social": 12.0, "commerce": 24.0, "domain": 12.0, "rapid": 48.0}


def _run(mode: str, kind: str, seed: int, hours: float) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="random")
    out = subprocess.run([sys.executable, "-m", "crucible.substrate.integrity", mode, kind, str(seed), str(hours)],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=3000)
    if out.returncode != 0:
        return {"kind": kind, "error": (out.stderr or out.stdout)[-1500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------- positive control
@register_module
class Sp23bOwner(Module):
    name = "sp23b_owner"

    def install(self, world):
        world.state.declare_table("sp23b_owned")


@register_module
class Sp23bIntruder(Module):
    """Its applier writes a table another module declared: a cross-module state write the trace must find."""
    name = "sp23b_intruder"
    host = "sp23b"
    depends = ("sp23b_owner",)

    def install(self, world):
        world.state.declare_table("sp23b_mine")

    @endpoint("poke", S.obj())
    def ep_poke(self, ctx):
        ctx.emit("sp23b.poked", {"n": 1}, vis="internal")
        return {}

    @endpoint("scribble", S.obj())
    def ep_scribble(self, ctx):
        ctx.state.put("sp23b_mine", "direct", {"n": 2})        # a handler writing state: replay cannot reproduce it
        return {}

    @applier("sp23b.poked")
    def ap_poked(self, ev):
        self.world.state.put("sp23b_mine", f"m{ev.seq}", {"n": 1})
        self.world.state.put("sp23b_owned", f"x{ev.seq}", {"n": 1})       # not its table


def _control() -> dict:
    w = World.create(WorldSpec("sp23b-control", 1, [{"name": m, "config": {}} for m in
                                                     ("identity", "bank", "admin", "sp23b_owner", "sp23b_intruder")]))
    p = w.system_call("admin.join", {"handle": "ctl", "endowment": 0})["principal"]
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "sp23b.poke", "args": {}}).ok
        assert w.execute(p, {"tool": "sp23b.scribble", "args": {}}).ok
    return tr.report()


@pytest.mark.slow
@pytest.mark.timeout(10800)
def test_SP_2_3b_typed_events_approved_interfaces_and_determinism():
    ctl = _control()
    # the trace finds the intruder's write into the owner's table and its handler's direct write, and only those
    assert ctl["cross_module_state_writes"] == 2, ctl
    assert [(r["writer"], r["target"]) for r in ctl["cross"]] == [("sp23b_intruder", "sp23b_owned")]
    assert [(r["writer"], r["target"]) for r in ctl["handler"]] == [("handler", "sp23b_mine")]

    kinds = list(integrity.KINDS)
    traced, hashed, problems = {}, {}, {}
    for kind in kinds:
        tr = _run("trace", kind, 0, HOURS[kind])
        if "error" in tr:
            problems[kind] = tr["error"]
            continue
        traced[kind] = tr
    for kind in kinds + list(integrity.ENV_KINDS):
        a, b = _run("hashes", kind, 0, HOURS[kind]), _run("hashes", kind, 0, HOURS[kind])
        if "error" in a or "error" in b:
            problems.setdefault(kind, (a.get("error") or b.get("error")))
            continue
        hashed[kind] = (a, b)
    # the domain kind is not available while no Tier 2 domain is registered; every other kind must run
    missing = sorted(set(kinds) - set(traced) - {"domain"})
    assert not missing, {k: problems.get(k, "")[-600:] for k in missing}
    checked = {k: v for k, v in hashed.items() if v[0]["events"] > 0 and v[1]["events"] > 0}
    mismatches = sorted(k for k, (a, b) in checked.items()
                        if (a["last_hash"], a["digest"]) != (b["last_hash"], b["digest"]))
    unreplayable = sorted(k for k, t in traced.items() if (t.get("replay") or {}).get("replay_exact") is not True)
    cross = sum(t["writes"]["cross_module_state_writes"] for t in traced.values()) + len(unreplayable)
    offenders = {k: t["writes"]["cross"][:8] + t["writes"]["handler"][:4] + t["writes"]["refused"][:4]
                 for k, t in traced.items() if t["writes"]["cross_module_state_writes"]}
    writes = sum(t["writes"]["writes_traced"] for t in traced.values())
    assert writes > 10_000 and all(t["writes"]["events_applied"] > 0 for t in traced.values())
    # typed-event coverage of the traced worlds' logs (reported; the requirement's metric is the write discipline)
    cov = bus.types()
    metrics = {
        "cross_module_state_writes": int(cross),
        "world_kinds_checked_for_determinism": len(checked),
        "determinism_mismatches": len(mismatches),
        "world_kinds_traced": len(traced),
        "state_writes_traced": int(writes),
        "handler_writes": int(sum(t["writes"]["handler_writes"] for t in traced.values())),
        "column_writes_refused_by_rule": int(sum(t["writes"]["column_writes_refused_by_rule"]
                                                 for t in traced.values())),
        "control_cross_module_writes_found": int(ctl["cross_module_state_writes"]),
        "traced_worlds_not_replaying_to_their_state": len(unreplayable),
        "registered_event_types": len(cov),
        "kinds_unavailable": len(problems),
    }
    print(json.dumps({"metrics": metrics, "mismatches": mismatches, "offenders": offenders,
                      "unavailable": {k: v[-300:] for k, v in problems.items()},
                      "kinds": {k: {"events": v[0]["events"], "hash": v[0]["last_hash"][:16]}
                                for k, v in checked.items()}}, indent=1, default=str))
    evidence.record("SP-2.3b", metrics, command=CMD,
                    notes=f"kinds checked {sorted(checked)}; mismatches {mismatches}; unreplayable {unreplayable}; "
                          f"offenders {json.dumps(offenders, default=str)[:1500]}; unavailable {sorted(problems)}")
    assert len(checked) >= 3
