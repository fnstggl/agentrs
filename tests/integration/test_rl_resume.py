"""Crash safety: an experiment killed mid-RL resumes from its last RL-state checkpoint."""
import json
from pathlib import Path

import pytest
import torch

from agentrs.config import load_config
from agentrs.training import pipeline
from agentrs.training.pipeline import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path, name):
    return load_config(ExperimentConfig, ROOT / "configs/experiments/smoke.yaml",
                       [f"out_dir={tmp_path / 'runs'}", f"cache_dir={tmp_path / 'cache'}", f"name={name}",
                        "rl_iterations=3", "use_imagination=false", "eval_every_iters=10"])


def _weights(run_dir):
    from safetensors.torch import load_file

    return load_file(str(run_dir / "checkpoints" / "policy_final" / "weights.safetensors"))


def test_crash_mid_rl_then_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    ref = run_experiment(_cfg(tmp_path, "ref"))

    calls = {"n": 0}
    orig_update = pipeline.RLTrainer.update

    def crashing_update(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # die during the second RL iteration (after iteration 1 was checkpointed)
            raise KeyboardInterrupt("simulated instance loss")
        return orig_update(self, *a, **kw)

    monkeypatch.setattr(pipeline.RLTrainer, "update", crashing_update)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(_cfg(tmp_path, "crash"))
    run_dir = tmp_path / "runs" / "crash"
    done = json.loads((run_dir / "checkpoints" / "rl_state" / "DONE").read_text())
    assert done["completed_iterations"] == 1
    assert any("_rl1_" in p.name for p in (run_dir / "experience" / "real").glob("*.parquet"))

    monkeypatch.setattr(pipeline.RLTrainer, "update", orig_update)
    res = run_experiment(_cfg(tmp_path, "crash"))
    assert res["resumed_from"] == [1]
    assert res["real_episodes"] == ref["real_episodes"]
    # the re-collected iteration's shards replaced the partial ones (no duplicates)
    tags = [p.name.split("_")[1] for p in (run_dir / "experience" / "real").glob("*_rl*_*.parquet")]
    assert sorted(tags) == sorted(set(tags))
    # same weights and same evaluation as a run that never crashed
    a, b = _weights(tmp_path / "runs" / "ref"), _weights(run_dir)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    for k, v in ref["final"].items():
        assert res["final"][k]["success_rate"] == v["success_rate"], k


def test_resume_disabled_starts_over(tmp_path):
    run_experiment(_cfg(tmp_path, "r"))
    res = run_experiment(_cfg(tmp_path, "r").model_copy(update={"resume": False}))
    assert "resumed_from" not in res
