"""INT3A item 2 (decision cost): the value-of-information computation reuses the predictive means the evaluation
already computed and runs its sampled-outcome loop on the fast state's device. Same numbers: on the CPU the outputs are
bit-identical with and without the reuse; on the real configuration the old and new code were compared on the same
captured inputs (``docs/INT3A.md`` section 2)."""
from __future__ import annotations

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task


def _agent(tmp_path):
    import os
    import sys
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "integration")
    sys.path.insert(0, here)
    try:
        from int_helpers import DriverState, agent_config, drive, lab_config
    finally:
        sys.path.remove(here)
    clock = SimClock()
    world = LabWorld(lab_config(4), clock)
    task = lab_task(world, "obj_cost")
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock,
                         task=task, verifier=verifier)
    drive(agent, world, clock, verifier, DriverState(), 3)
    return agent, world


def test_voi_with_the_evaluations_means_is_bit_identical(tmp_path):
    agent, world = _agent(tmp_path)
    try:
        pred = agent.predictor
        seen = {}
        real = pred._voi

        def cap(*a, **kw):
            seen["args"], seen["kw"] = a, kw
            seen["gstate"] = a[10].get_state().clone()
            seen["fast"] = pred.fast.clone()
            return real(*a, **kw)
        pred._voi = cap
        cands = agent.propose(world.discover_tools(), 8)
        agent.evaluate([c for c in cands if not world.validate_action(c)])
        pred._voi = real
        assert "means" in seen["kw"]                                    # the evaluation passes its means
        outs = []
        for means in (seen["kw"]["means"], None):
            g = torch.Generator()
            g.set_state(seen["gstate"])
            pred.fast = seen["fast"].clone()
            args = list(seen["args"])
            args[10] = g
            outs.append(real(*args, means=means))
        (v1, i1), (v2, i2) = outs
        assert torch.equal(v1, v2) and set(i1) == set(i2)
        assert all(torch.equal(i1[k], i2[k]) for k in i1)
        assert float(v1.abs().max()) >= 0.0 and float(i1["info_gain_nats"].abs().max()) > 0.0
    finally:
        agent.close()
