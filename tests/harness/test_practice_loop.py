"""Integration test 18 (mechanics, small scale): the full autonomous sandbox practice loop as separate
services over real HTTP — lab, shared inference service, agents through the gateway, delayed
follow-ups, verification, learner, held-out comparison and promotion gate. The measured learning
result at research scale comes from the same script (results/harness/)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_18_autonomous_practice_loop_end_to_end(tmp_path):
    out = tmp_path / "res.json"
    cmd = [sys.executable, str(ROOT / "scripts/harness/sandbox_learning_experiment.py"), "--model", "scratch",
           "--cycles", "2", "--groups-per-cycle", "2", "--group", "2", "--promote-every", "2", "--eval-tasks", "3",
           "--test-tasks", "3", "--bc-demos", "16", "--bc-steps", "5", "--agents", "2", "--threads", "2",
           "--out", str(tmp_path / "run"), "--results-name", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    log = json.loads(out.read_text())
    assert log["account"]["provisioning_state"] == "ACTIVE"
    c0, c1 = log["cycles"]
    assert c0["executed"] == 4 and c0["verification"]["pending"] == 4  # rewards not yet mature
    assert c1["verification"]["matured"] >= 4  # the first cycle's posts matured 24 h later
    assert log["throughput"]["inference"]["requests"] >= 8
    assert log["throughput"]["action_status"].get("executed", 0) >= 8
    assert set(log["final"]["results"]) >= {"bc-v0"} and all(v["n"] == 3 for v in log["final"]["results"].values())
    regs = log["throughput"]["registry"]
    assert regs[0]["status"] in ("promoted", "production", "retired")
