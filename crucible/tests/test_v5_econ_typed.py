"""ECON worlds run typed and strict, as civilisations do from wave 3b (c5c/w3b-civ: params.typed_events, strict
worlds): the population's exchanges, labour market and institutions emit only valid payloads of the registered
types (labor.hired / quit / left, inst.charter, column batches), raise no handler bug, and validation changes
nothing - the typed strict world runs exactly the trajectory of the plain one."""
import collections

import crucible

crucible.load_all_modules()
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.substrate import bus  # noqa: E402

DAY = 86400


def _run(typed: bool, days: int = 26):
    spec = startup_spec(21, n=450, formats=["schedule.jobs"], extra=("xchg", "inst", "labor"),
                        rivals={"display_ads": True})
    if typed:
        spec.params["typed_events"] = True
    w = World.create(spec, strict=typed)
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
    return w


def test_econ_world_is_typed_strict_and_unchanged_by_validation():
    typed, plain = _run(True), _run(False)
    t = bus.types()
    for name in ("labor.hired", "labor.quit", "labor.left", "inst.charter"):
        assert name in t and t[name]["owner"] in ("labor", "inst")
    kinds = collections.Counter(e.type for e in typed.log)
    assert kinds["inst.charter"] and kinds["cols.batch"], "the typed events were emitted (and validated)"
    assert not [e for e in typed.log if e.type in ("proc.failed", "rivals.failed")]
    assert typed.check_invariants() == []
    # the two specs differ only in the typed_events flag (the genesis note hashes the spec, and with it every event
    # hash of the chain): every event after it is the same event
    def body(w):
        return [(e.type, e.t, e.actor, e.payload) for e in w.log[1:]]
    assert typed.log[0].type == plain.log[0].type and body(typed) == body(plain)

