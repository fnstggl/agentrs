"""Crucible 5.0 substrate: the events of the shared primitives are typed, and 4.0 logs satisfy their schemas."""
import numpy as np
import pytest

import crucible
from crucible.agents.scripted import _random_args
from crucible.core.kernel import World
from crucible.substrate import bus
from crucible.substrate import primitives as P
from crucible.worldgen.generator import generate

crucible.load_all_modules()
FAMILIES = ["tiny", "ow_tiny", "open_mixed", "open_weak_institutions", "open_industry"]


def session(w, n=300, seed=1):
    rng = np.random.default_rng(seed)
    ps = [w.system_call("admin.join", {"handle": f"ty{i}", "endowment": 50_000})["principal"] for i in range(3)]
    eps = sorted(e for e in w.endpoints if not e.startswith(("admin.", "world.")))
    for i in range(n):
        ep = w.endpoints[eps[int(rng.integers(0, len(eps)))]]
        w.execute(ps[i % 3], {"tool": ep.name, "args": _random_args(ep.params, rng, ps[i % 3])})
        if i % 100 == 99:
            w.advance_time(w.now + 6 * 3600)
    w.advance_time(w.now + 86400)


def test_every_primitive_event_type_is_registered_with_its_owner():
    t = bus.types()
    for name, (kind, mod, _) in P.PRIMITIVE_EVENTS.items():
        assert t[name]["owner"] == mod and kind in P.OWNERS
        assert mod == P.KERNEL or mod == P.owner(kind)["module"]
    covered = {k for k, _, _ in P.PRIMITIVE_EVENTS.values()} | {"content_artifact", "asset"}
    assert len(covered) >= 15


@pytest.mark.parametrize("family", FAMILIES)
def test_4_0_logs_satisfy_the_primitive_schemas(family):
    w = World.create(generate({"family": family, "overrides": {"params": {"cost_model": "deterministic"}}}, 2))
    session(w)
    n = errs = 0
    bad = []
    for e in w.log:
        if e.type in P.PRIMITIVE_EVENTS:
            n += 1
            v = bus.validate(e.type, e.payload)
            if v:
                errs += 1
                bad.append((e.type, v[:2]))
    assert n > 50 and errs == 0, bad[:5]


def test_a_world_enforcing_every_type_runs_4_0_sessions_unchanged():
    from crucible.core.kernel import WorldSpec
    spec = generate({"family": "ow_tiny", "overrides": {"params": {"cost_model": "deterministic"}}}, 4)
    plain = World.create(spec, world_id="plain")
    session(plain, n=200, seed=4)
    checked = World.create(WorldSpec.from_dict(spec.to_dict()), world_id="checked")
    checked.spec.params["typed_events"] = True      # validate from now on (the genesis is identical to `plain`)
    assert bus.enforce(checked) >= len(P.PRIMITIVE_EVENTS) - 1      # sched.add cannot be subscribed to
    calls = []
    fn = bus._validator("bank.txn")
    checked.subscribers["bank.txn"] = [("kernel", lambda ctx, ev: (calls.append(ev.seq), fn(ctx, ev))[1])]
    session(checked, n=200, seed=4)
    # every validated event passed (a failing validator would have turned the command into a rejection), and
    # validation adds no events: the two logs are identical event for event
    assert calls and [e.to_dict() for e in checked.log] == [e.to_dict() for e in plain.log]
