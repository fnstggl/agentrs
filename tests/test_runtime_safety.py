"""Regression tests for the 2026-11-18 late-stage crash and the crash-prevention machinery."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentrs.runtime_safety import CrashLedger, crash_signature, main_guard_problems, smoke_run, state_phase_done

SRC = Path(__file__).resolve().parents[1] / "src"
TINY = dict(d=48, deter=64, stoch=4, classes=8, hidden=64, layers=1, heads=2, n_options=3, ens=3)


def test_no_module_defines_anything_after_its_main_guard():
    assert main_guard_problems(SRC) == []


def test_main_guard_check_catches_the_original_failure(tmp_path):
    (tmp_path / "bad.py").write_text("def main():\n    return Foo.x()\n\nclass Foo: pass\n\n"
                                     "if __name__ == '__main__':\n    main()\n\ndef late():\n    pass\n\nFoo.x = late\n")
    assert main_guard_problems(tmp_path) and "statements after the __main__ guard" in main_guard_problems(tmp_path)[0]


def _tiny_cfg(run_dir: Path) -> dict:
    return dict(run_dir=str(run_dir), model=TINY, workers=8, steps_per_iter=120, max_iters=2, updates_per_iter=2,
                batch_episodes=4, learn_start_steps=50, eval_every=100, eval_episodes=2, test_episodes=2, cf_states=1,
                audit_episodes=20, learner={"actor": "ppo", "epochs": 1, "mb_episodes": 8}, final_reserve_h=0,
                eval_at_interactions=[100], final_reset_evals=True, final_nl_episodes=2, final_frontends=["parser"])


def test_production_cli_runs_start_to_finish_including_all_end_of_run_diagnostics(tmp_path):
    """The exact production entry point (python -m agentrs.cam.v2.train --config ...), not an import."""
    import yaml

    cfgf = tmp_path / "cfg.yaml"
    cfgf.write_text(yaml.safe_dump(_tiny_cfg(tmp_path / "run")))
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    res = smoke_run([sys.executable, "-u", "-m", "agentrs.cam.v2.train", "--config", str(cfgf)], cwd=str(SRC.parent), env=env,
                    run_dir=tmp_path / "run", done_check=state_phase_done, timeout_s=900, log_path=tmp_path / "smoke.log")
    assert res["ok"], (res, (tmp_path / "smoke.log").read_text()[-3000:])
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    assert {"counterfactual", "planning_diagnosis", "hybrid_final"} <= set(st)
    assert {("final", "test"), ("reset_all-train", "val"), ("frontend-parser-heldout", "test")} <= {(e["tag"], e["split"]) for e in st["evals"]}


def test_failing_diagnostic_is_recorded_and_never_crashes_a_finished_run(tmp_path, monkeypatch):
    from agentrs.cam.v2 import train as T

    def boom(self):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(T.Trainer2, "hybrid_final", boom)
    t = T.Trainer2(T.TrainConfig2(**_tiny_cfg(tmp_path / "r")))
    assert t.run() == 0
    st = json.loads((tmp_path / "r" / "state.json").read_text())
    assert st["phase"] == "done" and "counterfactual" in st
    assert "synthetic diagnostic failure" in st["diagnostic_errors"]["hybrid_final"]["error"]
    # a restart of the finished run does nothing (no repeated training or diagnostics)
    assert T.main(["--config", "/dev/null", "--set", f"run_dir={tmp_path / 'r'}"]) == 0


def test_interrupted_diagnostic_is_retried_once_then_abandoned(tmp_path):
    from agentrs.cam.v2 import train as T

    t = T.Trainer2(T.TrainConfig2(**_tiny_cfg(tmp_path / "r")))
    calls = []
    t.state["diagnostic_attempts"] = {"x": 1}  # one attempt was interrupted by a crash
    t.optional_diagnostic("x", lambda: calls.append(1) or t.state.__setitem__("x", {"ok": 1}))
    assert calls == [1] and t.state["x"] == {"ok": 1}
    t.state["diagnostic_attempts"]["y"] = 2  # interrupted twice: abandoned, not run again
    t.optional_diagnostic("y", lambda: calls.append(2))
    assert calls == [1] and "abandoned" in t.state["diagnostic_errors"]["y"]["error"]


def test_crash_ledger_stops_a_repeating_failure_with_a_report(tmp_path):
    log = tmp_path / "stdout.log"
    tb = ('Traceback (most recent call last):\n  File "/x/train.py", line 600, in <module>\n    main()\n'
          '  File "/x/train.py", line 573, in run\n    self.hybrid_final()\n'
          "AttributeError: 'Trainer2' object has no attribute 'hybrid_final'\n")
    log.write_text("training...\n" + tb)
    assert crash_signature(log.read_text())["exception"].startswith("AttributeError")
    led, js = CrashLedger(repeat_limit=2, max_starts=5), {"starts": 1}
    assert led.record(js, tmp_path, log) is False
    js["starts"] = 2
    log.write_text(log.read_text() + "resumed\n" + tb)
    assert led.record(js, tmp_path, log) is True
    assert "same failure occurred 2 times" in (tmp_path / "CRASH_REPORT.md").read_text()


def _queue_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sl_queue", Path(__file__).resolve().parents[1] / "scripts" / "sl_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_queue_waits_for_a_checkpoint_then_starts_and_skips_unreachable_ones(tmp_path):
    """Transfer jobs start once their source checkpoint exists (source still training) and are skipped when the source
    finished without it (amendment 5)."""
    import yaml
    qm = _queue_module()
    runs = tmp_path / "runs"
    for name, ph in (("SRC_a", "train"), ("SRC_b", "train")):
        (runs / name / "ckpt").mkdir(parents=True)
        (runs / name / "state.json").write_text(json.dumps({"phase": ph}))
    want_a, want_b = runs / "SRC_a" / "ckpt" / "life_10", runs / "SRC_b" / "ckpt" / "life_20"
    q = {"runs_root": str(runs), "src": str(Path(__file__).resolve().parents[1] / "src"), "module": "json.tool",
         "max_jobs": 4, "min_free_mib": 0, "min_free_disk_gb": 0, "smoke": False, "gh_backup": False,
         "jobs": [{"name": "T_a", "config": "x.yaml", "set": {}, "wait_for": [str(want_a)]},
                  {"name": "T_b", "config": "x.yaml", "set": {}, "wait_for": [str(want_b)]}]}
    qp = tmp_path / "q.yaml"
    qp.write_text(yaml.safe_dump(q))
    qu = qm.Queue(str(qp))
    for _ in range(3):
        qu.tick()
    assert qu.st["T_a"]["status"] == qu.st["T_b"]["status"] == "pending"   # sources still training, no checkpoints yet
    want_a.mkdir()
    (want_a / "weights.pt").write_text("partial")
    qu.tick(); qu.tick()
    assert qu.st["T_a"]["status"] == "pending"                              # incomplete checkpoint (no DONE marker)
    (want_a / "DONE").write_text("1")
    (runs / "SRC_b" / "state.json").write_text(json.dumps({"phase": "done"}))  # SRC_b finished without life_20
    qu.tick(); qu.tick()
    assert qu.st["T_a"]["status"] == "running" and qu.st["T_a"]["starts"] == 1
    assert qu.st["T_b"]["status"] == "skipped"
    assert "T_b skipped" in (runs / "queue.log").read_text()
