"""Demo-init transcripts: non-privileged ExampleInductor only, training worlds only, tagged and re-checked."""
from __future__ import annotations

import json

import pytest

import crucible
crucible.load_all_modules()
from crucible.agents.scripted import Builder                    # noqa: E402
from crucible.research import llm_eval as L                     # noqa: E402

from superlearner.cam_unified.interfaces import ActionSource, ObsKind, Provenance   # noqa: E402
from superlearner.cam_unified.envs.demos import DEMO_CONDITION, PrivilegedDemoError, load_demos, record_demos   # noqa: E402
from superlearner.cam_unified.envs.worldgen import HeldOutFamilyError   # noqa: E402


def tiny(seed=1, family="ow_tiny"):
    return L.open_economy_goal(world_family=family, seed=seed, overrides={"n_consumers": 800} if family == "ow_tiny"
                               else {}, days=6)


def test_record_and_load_demos(tmp_path):
    path = str(tmp_path / "demos.jsonl")
    seen = []
    out = record_demos([tiny(1), tiny(2)], path, sink=seen.extend)
    assert out["episodes"] == 2 and out["demonstrator"] == "example_inductor"
    assert out["successes"][0] is True                  # seed 1: the inductor succeeds on day 4
    recs = list(load_demos(path))
    assert len(recs) == out["records"] == len(seen) > 100
    acted = [r for r in recs if r.action is not None]
    assert all(r.action.source is ActionSource.SCRIPTED and r.provenance is Provenance.REAL for r in acted)
    assert {r.extra["demo_condition"] for r in recs} == {DEMO_CONDITION}
    assert {r.env_id for r in recs} and len({r.env_id for r in recs}) == 2
    tools = {r.action.tool for r in acted}
    assert {"use.examples", "code.commit", "cloud.deploy", "market.list"} <= tools
    # the commit carries the program the demonstrator wrote (the long free-form argument a decoder must learn)
    commit = next(r for r in acted if r.action.tool == "code.commit")
    assert "def solve(args, caller)" in commit.action.args["files"]["service.py"]
    # every demo record carries the observations the agent had before acting
    assert all(isinstance(r.observations, list) for r in recs) and any(
        o.kind is ObsKind.TOOL_RESULT for r in recs for o in (r.result.observations if r.result else []))


def test_privileged_and_heldout_demos_refused(tmp_path):
    with pytest.raises(PrivilegedDemoError):
        record_demos([tiny()], str(tmp_path / "x.jsonl"), policy_factory=lambda: Builder())
    with pytest.raises(HeldOutFamilyError):
        record_demos([tiny(1, family="ow_heldout")], str(tmp_path / "y.jsonl"))


def test_load_demos_rejects_untagged_or_privileged(tmp_path):
    path = str(tmp_path / "d.jsonl")
    record_demos([tiny(1)], path, days=1.0)
    lines = open(path).read().splitlines()
    bad = json.loads(lines[3])
    bad["extra"]["privileged"] = True
    p2 = str(tmp_path / "bad.jsonl")
    with open(p2, "w") as fh:
        fh.write(lines[0] + "\n" + json.dumps(bad) + "\n")
    it = load_demos(p2)
    next(it)
    with pytest.raises(PrivilegedDemoError):
        next(it)
