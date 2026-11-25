"""Scenarios that pin the 5.0 kernel to Crucible 4.0 byte for byte (used by tests/test_v5_substrate_kernel.py).

`KERNEL40_DIGESTS` was recorded by running `scenario(name)` against the untouched Crucible 4.0 tree (c07098b):

    git archive c07098b crucible/src | tar -x -C /tmp/k40
    PYTHONPATH=/tmp/k40/crucible/src python tests/v5_kernel_scenarios.py
"""
import json
import sys

KERNEL40_DIGESTS = {}  # filled below by the recorded values

SCENARIOS = {
    "ow_tiny_2d": {"family": "ow_tiny", "seed": 3, "days": 2},
    "tiny_session": {"family": "tiny", "seed": 5, "days": 1, "session": 300},
    "mech_scm_additive_tools_1d": {"family": "mech_scm_additive_tools", "seed": 2, "days": 1},
    "open_mixed_1d": {"family": "open_mixed", "seed": 1, "days": 1},
}


def scenario(name: str) -> dict:
    import numpy as np
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World
    from crucible.worldgen.generator import generate
    sc = SCENARIOS[name]
    spec = generate({"family": sc["family"], "overrides": {"params": {"cost_model": "deterministic"}}}, sc["seed"])
    w = World.create(spec, world_id=f"k40-{name}")
    if sc.get("session"):
        # a randomised, schema-valid session (accepted and rejected commands, idempotency keys)
        from crucible.agents.scripted import _random_args
        rng = np.random.default_rng(sc["seed"])
        ps = [w.system_call("admin.join", {"handle": f"k{i}", "endowment": 50_000})["principal"] for i in range(3)]
        eps = sorted(e for e in w.endpoints if not e.startswith(("admin.", "world.")))
        for i in range(sc["session"]):
            ep = w.endpoints[eps[int(rng.integers(0, len(eps)))]]
            act = {"tool": ep.name, "args": _random_args(ep.params, rng, ps[i % 3])}
            if i % 7 == 0:
                act["idempotency_key"] = f"k{i % 11}"
            w.execute(ps[i % 3], act)
            if i % 60 == 59:
                w.advance_time(w.now + 3600)
    w.advance_time(w.now + sc["days"] * 86400)
    return {"events": w.seq, "last_hash": w.last_hash, "digest": w.digest()}


KERNEL40_DIGESTS.update({
    "ow_tiny_2d": {"events": 1967, "last_hash": "bf5529f5dd0f81590fcbe7abfb6b7b4c", "digest": "f278ea3ef9ec364f470a5531133c821c"},
    "tiny_session": {"events": 1593, "last_hash": "3149c8c77a4e80c6fa80397df743ba15", "digest": "84d076d46291a142b4ed503cbe688274"},
    "mech_scm_additive_tools_1d": {"events": 866, "last_hash": "6fe8a2d6a89e4dc5d38b00de2156fd0d", "digest": "808602900ea235a634abd8ce8fa0770d"},
    "open_mixed_1d": {"events": 2695, "last_hash": "ec94efb621e6511d02d6f1f9870f8dbb", "digest": "85b86535f33394b19c36b4302f752e7a"},
})

if __name__ == "__main__":
    out = {n: scenario(n) for n in SCENARIOS}
    json.dump(out, sys.stdout, indent=1)
