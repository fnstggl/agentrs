"""Every economic task family: the scripted expert passes the hidden verifier, an immediate-finish
policy does not, splits are deterministic, and the sandbox rejects unsafe edits."""
import json

import pytest

from agentrs.econ.core import load_all, make_task
from agentrs.econ.rollout import make_env, run_expert

FAMS = sorted(load_all())


@pytest.mark.parametrize("key", FAMS)
def test_expert_succeeds_and_null_policy_fails(key):
    for split in ("train", "val", "test"):
        t = make_task(key, split, 3)
        assert run_expert(t).success, (key, split)
        env = make_env(t)
        term = [n for n, tl in getattr(env, "tools", {}).items() if tl.terminal]
        if term:
            args = {"resolve": {"outcome": "completed"}, "submit_answer": {"answer": 0},
                    "submit": {"answer": "x", "evidence": []}}.get(term[0], {})
            env.step(json.dumps({"tool": term[0], "args": args}))
            assert not env.verify()[0], key


def test_tasks_are_deterministic_and_splits_differ():
    a = make_task("finance/payment_matching", "train", 7)
    b = make_task("finance/payment_matching", "train", 7)
    c = make_task("finance/payment_matching", "test", 7)
    assert a.goal == b.goal and a.private["expect"] == b.private["expect"]
    assert a.private["tables"] != c.private["tables"]


def test_sandbox_rejects_unsafe_edit():
    t = make_task("software/bugfix", "train", 0)
    env = make_env(t)
    path = [p for p in env.files if not p.startswith("test_")][0]
    first = env.files[path].split("\n")[0]
    out = env.step(json.dumps({"tool": "replace", "args": {"path": path, "old": first,
                                                            "new": first + "\n    import os"}}))
    assert "rejected" in out.observation
