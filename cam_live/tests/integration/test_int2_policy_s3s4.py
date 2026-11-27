"""S-3 / S-4 (applied by INT2 in the authorization policy): ``network: "none"`` is not a network request, and
``sandbox.list_files`` without a path lists the workspace root; a path outside the file roots is still refused."""
from __future__ import annotations

import os

from cam_live.contracts import ActionSpec, Limits, TaskSpec
from cam_live.harness.auth import PolicyService


def _verdict(p, task, op, args):
    a = ActionSpec(action_id="a" + os.urandom(4).hex(), operation=op, modality="code", arguments=args)
    return p.authorize(a, task, {}).verdict


def test_sandbox_requests_s3_s4(tmp_path):
    task = TaskSpec(objective_id="o", objective="x", interfaces=["code", "internal"], mode="staging",
                    limits=Limits(duration_s=60, max_decisions=5))
    p = PolicyService(str(tmp_path / "auth"))
    p.register_task(task, "operator")
    assert _verdict(p, task, "sandbox.python", {"code": "print(1)", "network": "none"}) == "allow"
    assert _verdict(p, task, "sandbox.python", {"code": "print(1)", "network": "scoped"}) == "deny"
    assert _verdict(p, task, "sandbox.list_files", {}) == "allow"
    assert _verdict(p, task, "sandbox.list_files", {"path": "/etc"}) == "deny"
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    a = open(os.path.join(root, "src", "cam_live", "harness", "auth", "default_policy.yaml"), "rb").read()
    b = open(os.path.join(root, "configs", "authorization.yaml"), "rb").read()
    assert a == b                                     # the two policy files stay byte-identical
