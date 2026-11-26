"""Gate A machinery of the D-82 resolution: the Crucible 4.0 regression suite runs as 4.0 wrote it, as an explicitly
selected historical baseline. These tests check the pieces on the real tree; the full measurement is Gate A.1
(tests/v5/test_gate_A.py)."""
from __future__ import annotations

import os
import subprocess

import pytest

from crucible.acceptance.v5 import baseline as B, checker, official
from crucible.experience import modes


def test_the_historical_baseline_is_selected_only_explicitly(monkeypatch):
    monkeypatch.delenv(modes.HISTORICAL_BASELINE_ENV, raising=False)
    assert modes.check_mode(None) == modes.OBSERVATION_ONLY and not modes.is_explicit_legacy(None)
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "4.0")
    assert modes.check_mode(None) == modes.LEGACY and modes.is_explicit_legacy(None)
    assert modes.check_mode(modes.OBSERVATION_ONLY) == modes.OBSERVATION_ONLY   # a named mode always wins
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "5.0")
    with pytest.raises(ValueError):
        modes.check_mode(None)
    # it never reaches a measured process, and a record made under it is rejected
    env, gone = official.scrubbed({"CRUCIBLE_HISTORICAL_BASELINE": "4.0", "PATH": "/bin"})
    assert "CRUCIBLE_HISTORICAL_BASELINE" in gone and "CRUCIBLE_HISTORICAL_BASELINE" not in env
    probs = checker.env_problems(type("C", (), {"site_hooks": {}})(), {
        "schema": checker.SCHEMA, "interpreter": {"flags": {"no_user_site": True}}, "site_hooks": {},
        "environment": {"historical_baseline": "4.0"}})
    assert any("historical baseline" in p for p in probs)


def test_the_sandbox_counts_executed_lines_only_in_a_4_0_baseline_process(monkeypatch):
    from crucible.sandbox.runner import SandboxConfig, run_program
    src = "def f(n, c):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "4.0")
    a = run_program({"s.py": src}, "s:f", 1000, cfg=SandboxConfig(count_lines=True))
    b = run_program({"s.py": src}, "s:f", 3000, cfg=SandboxConfig(count_lines=True))
    assert a["ok"] and a["meter"]["cost_model"].startswith("crucible-4.0")
    assert a["work_lines"] >= 2000 and b["work_lines"] - a["work_lines"] == 2 * 2000   # 2 lines per iteration
    monkeypatch.delenv(modes.HISTORICAL_BASELINE_ENV)
    c = run_program({"s.py": src}, "s:f", 1000, cfg=SandboxConfig(count_lines=True))
    assert c["ok"] and "cost_model" not in c["meter"] and c["meter"]["billed_ns"] >= 0   # 5.0: parent-measured


def test_the_4_0_test_files_are_restored_byte_identical_in_a_fresh_clone(tmp_path):
    croot, restored, differ = B.v4_test_clone(B.M.crucible_root(), str(tmp_path / "clone"))
    assert differ == []
    assert set(restored) == set(B.v4_test_files_changed(B.M.crucible_root()))
    for f in restored:
        want = subprocess.run(["git", "rev-parse", f"{B.V4_COMMIT}:./{f}"], cwd=B.M.crucible_root(),
                              capture_output=True, text=True).stdout.strip()
        assert B._blob(os.path.join(croot, f)) == want
    nodes = B.baseline_nodes()
    assert len(nodes) == 1076 and all(os.path.exists(os.path.join(croot, f)) for f in B.baseline_files(nodes))
