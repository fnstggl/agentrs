"""Autonomous NPC invention (19E): simulated developers identify a reported problem, generate a controller program
from a grammar, deploy and publish it, read the trial evidence and ship mutated versions. No agent is present."""
import pytest

import crucible
from crucible.actors.innovators import FORECASTS, SPACE, mutate, program, sample
from crucible.agents.base import Runner
from crucible.core.kernel import World
from crucible.worldgen import generate

crucible.load_all_modules()


def _world(seed=1, **params):
    return World.create(generate({"family": "open_tight_compute", "overrides": {
        "module_config": {"civ": {"innovators": 2}}, "params": params}}, seed))


def test_program_grammar_generates_valid_distinct_controllers():
    import numpy as np
    rng = np.random.default_rng(0)
    seen = set()
    for kind in ("capacity_overload", "capacity_unavailable", "cost:compute"):
        for _ in range(12):
            p = sample(rng)
            src = program(kind, p)
            compile(src, "service.py", "exec")
            assert FORECASTS[p["forecast"]] in src
            assert ("cloud.rent" in src) == (kind == "capacity_unavailable")
            seen.add(src)
            q = mutate(rng, p)
            assert set(q) == set(SPACE) and all(q[k] in SPACE[k] for k in q)
    assert len(seen) > 20


@pytest.mark.slow
def test_npc_innovators_invent_deploy_and_iterate_on_evidence():
    w = _world()
    st = w.state
    inn = [k for k, m in st.scan("minds") if "innovator" in m["routines"]]
    assert len(inn) == 2
    Runner(w, [], dt=3600, record=False).run(24 * 18)
    caps = dict(st.scan("capabilities"))
    npc_orgs = {st.tables["minds"][k]["routines"]["innovator"].get("org") for k in inn}
    mine = {k: c for k, c in caps.items() if c["provider"] in npc_orgs}
    assert mine                                                          # invented, not copied: no `improves` of others
    assert all(c.get("improves") in (None, *mine) for c in mine.values())
    # the artifact is executable code produced by the grammar, deployed and live
    cloud, code = w.modules["cloud"], w.modules["code"]
    for c in mine.values():
        dep = cloud.live_deployment(st, c["host"]) or {}
        if dep:
            src = code.files(st, dep["repo"], None)["service.py"]
            assert "sdk.call(\"cloud.metrics\"" in src and "def tick" in src
    # independent firms trialled it on their own books
    trials = [i for _, i in st.scan("cap_installs") if i["capability"] in mine and i["independent"]]
    assert trials
    # iteration: a later version declares it improves an earlier one and changes the program
    later = [c for c in mine.values() if c.get("improves")]
    notes = [e.payload for e in w.log if e.type == "note" and e.payload.get("what") == "invented"]
    if later:
        v1 = next(n for n in notes if n["capability"] == later[0]["improves"])
        v2 = next(n for n in notes if n["capability"] == [k for k, c in mine.items() if c is later[0]][0])
        assert v1["params"] != v2["params"]
    assert w.check_invariants() == []
    # the routine's state is never aliased with emitted payloads: the hash chain replays exactly
    assert World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()
    assert World.regenerate(w.spec, [e.to_dict() for e in w.log]).last_hash == w.last_hash


def test_innovation_can_be_ablated():
    w = _world(ablate_innovation=True)
    Runner(w, [], dt=3600, record=False).run(24 * 5)
    assert w.state.count("capabilities") == 0
