"""The benchmark script: preset counts on CPU (slow: instantiates up to 1e9 weights) and the recorded evidence."""
import json
import os

import pytest

from cam_live.config import PROJECT_ROOT, CamLiveConfig
from cam_live.nn import bench


def test_backbone_width_is_read_without_loading_weights():
    try:
        bw = bench.backbone_width(CamLiveConfig())
    except OSError:
        pytest.skip("Qwen3-4B config not in the local Hugging Face cache")
    assert bw == {"hidden": 2560, "n_layers": 36, "n_taps": 3}


def test_recorded_parameter_evidence_matches_the_code():
    path = os.path.join(PROJECT_ROOT, "docs", "evidence", "A", "params.json")
    if not os.path.exists(path):
        pytest.skip("run python -m cam_live.nn.bench first")
    res = json.load(open(path))
    import torch
    from cam_live.config import CoreConfig
    from cam_live.nn.model import ExperientialNet
    from cam_live.nn.presets import NNConfig, preset_nn, resolve_core
    for scale, entry in res["presets"].items():
        if scale == "debug":
            continue
        core = CoreConfig().apply_scale(scale)
        with torch.device("meta"):
            net = ExperientialNet(resolve_core(core), NNConfig(**preset_nn(core)), res["d_sem"], 3)
        # A's recorded counts predate PS's per-slot retention of the carried core state (n_slots parameters)
        n = sum(p.numel() for p in net.parameters())
        assert n - net.core.persist_rho.numel() == entry["experiential_params"], scale
        assert net.core.persist_rho.numel() == net.core.n_slots
        assert abs(entry["ratio_to_target"] - 1.0) < 0.05
    real = res["real_configuration"]
    if real.get("status") == "measured":
        p = real["parameters"]
        assert 3.9e9 < p["frozen"] < 4.1e9 and 10e6 <= p["adapters"] <= 20e6 and 45e6 < p["experiential"] < 55e6
        assert p["total_persistent_trainable"] == p["experiential"] + p["adapters"] + p["reward"] + p["heads_total"]
        assert real["bridge_noop_check"]["bridge_logits_bit_identical_at_init"] is True
        assert real["gpu_peak_total_gb"] < 24.0


@pytest.mark.slow
def test_bench_cpu_part_runs(tmp_path):
    out = tmp_path / "params.json"
    assert bench.main(["--skip-gpu", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert set(res["presets"]) == {"debug", "5m", "20m", "50m", "150m", "500m", "1b"}
    assert res["real_configuration"]["status"].startswith("skipped")
