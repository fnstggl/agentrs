"""G3-act: are two fresh agents with the same seed on the same LabWorld seed the same experiment? (SG's goal-free
ablation pairs arms by seed; re-running its script with the same seeds gave other numbers.)

Runs the driver of tests/integration (agent.learn(1e9): no wall-clock budget) for N decisions with
(a) two fresh agents, same config seed, and (b) the same, but with the agent's random id ``uid`` (os.urandom at
construction, agent.py ``self.uid``) forced equal. Prints the chosen operations and the verified value.

    PYTHONPATH=<tree>/cam_live/src .venv/bin/python determinism_probe.py [N]
"""
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "integration"))


def one(seed, n, uid=None):
    from int_helpers import DriverState, agent_config, drive, lab_config
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
    clock = SimClock()
    agent = CamLiveAgent(agent_config(seed=seed), tempfile.mkdtemp(prefix="g3act_det_"), clock=clock)
    if uid is not None:
        agent.uid = uid
    try:
        world = LabWorld(lab_config(300 + seed, target=30), clock)
        ver = LabVerifier(world)
        agent.attach_verifier(ver)
        task = lab_task(world, f"assigned_{seed}")
        agent.set_objective(task)
        out = drive(agent, world, clock, ver, DriverState(), n)
        return [d["operation"] for d in out], float(ver.verify(task).value)
    finally:
        agent.close()


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a1, a2 = one(0, n), one(0, n)
    b1, b2 = one(0, n, uid="aaaaaa"), one(0, n, uid="aaaaaa")
    print(json.dumps({"fresh_same_seed": {"run1": a1, "run2": a2, "identical": a1 == a2},
                      "same_seed_same_uid": {"run1": b1, "run2": b2, "identical": b1 == b2}}, indent=1))
