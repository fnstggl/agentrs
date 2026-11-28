"""G3A-3 (G3-act, major): two fresh agents with the same seed and salt in the same environment are the same experiment.

Before: ``CamLiveAgent.__init__`` drew its id prefix from ``os.urandom``; the ids enter id-keyed steps, so two fresh
agents with seed 0 in the same LabWorld diverged (verified 4 vs 1 after 8 decisions; G3-act's
``repro/g3act/determinism_probe.py``) and every "paired, same seed" ablation carried that noise. Now every random
draw of the agent is a function of the configured seed and a salt: ``agent.id_salt`` when configured (paired
evaluation arms set the same one), else the absolute run directory (separate runs never share id prefixes or the
Discord nonces derived from action ids). The driver is the integration tests' own (``agent.learn(1e9)``: no
wall-clock budget, so nothing depends on machine load).
"""
from __future__ import annotations

import tempfile

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
from int_helpers import DriverState, agent_config, drive, lab_config


def _run(root, seed, n, **settings):
    root.mkdir(parents=True, exist_ok=True)
    clock = SimClock()
    agent = CamLiveAgent(agent_config(seed=seed, **settings), tempfile.mkdtemp(dir=str(root)), clock=clock)
    try:
        world = LabWorld(lab_config(300 + seed, target=30), clock)
        ver = LabVerifier(world)
        agent.attach_verifier(ver)
        task = lab_task(world, f"assigned_{seed}")
        agent.set_objective(task)
        out = drive(agent, world, clock, ver, DriverState(), n)
        seq = [(d["operation"], d["arguments"], [round(float(x), 12) for x in d["probabilities"]], d["model_version"])
               for d in out]
        return agent.uid, seq, float(ver.verify(task).value)
    finally:
        agent.close()


def test_two_fresh_agents_with_the_same_seed_and_salt_make_the_same_decisions(tmp_path):
    uid1, seq1, v1 = _run(tmp_path / "a", 0, 8, id_salt="pair-0")        # paired arms: the same explicit salt
    uid2, seq2, v2 = _run(tmp_path / "b", 0, 8, id_salt="pair-0")
    assert seq1 == seq2, [(a[0], b[0]) for a, b in zip(seq1, seq2)]    # decisions, probabilities, learning
    assert v1 == v2                                                     # the verified outcome
    assert uid1 == uid2


def test_the_id_prefix_depends_on_the_seed_and_the_salt_only(tmp_path):
    def uid(seed, run_dir=None, **kw):
        d = run_dir or tempfile.mkdtemp(dir=str(tmp_path))
        a = CamLiveAgent(agent_config(seed=seed, **kw), d, clock=SimClock())
        try:
            return a.uid
        finally:
            a.close()
    assert uid(0, id_salt="x") == uid(0, id_salt="x") != uid(1, id_salt="x")
    assert uid(0, id_salt="x") != uid(0, id_salt="y")
    # not configured: the run directory is the salt -> separate runs never share id prefixes, the same run does
    d = str(tmp_path / "run1")
    assert uid(0) != uid(0)                                               # two different run directories
    assert uid(0, run_dir=d) == uid(0, run_dir=d)
