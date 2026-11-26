"""Scale presets: measured experiential parameter counts for every preset (NA-02, NA-10)."""
import dataclasses

import pytest
import torch

from cam_live.config import SCALE_PRESETS, CamLiveConfig, CoreConfig
from cam_live.nn.model import ExperientialNet, build_experiential, build_model
from cam_live.nn.presets import NN_PRESETS, NNConfig, preset_nn, resolve_core

QWEN_D_SEM, TAPS = 3 * 2560, 3          # Qwen3-4B hidden width x three depth taps


def _count(scale: str, d_sem: int = QWEN_D_SEM) -> int:
    core = CoreConfig().apply_scale(scale)
    with torch.device("meta"):          # shapes only: exact counts without allocating up to 1e9 weights
        net = ExperientialNet(resolve_core(core), NNConfig(**preset_nn(core)), d_sem, TAPS)
    return sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("scale", ["5m", "20m", "50m", "150m", "500m", "1b"])
def test_preset_parameter_counts_hit_their_targets(scale):
    n, target = _count(scale), SCALE_PRESETS[scale]["target_params"]
    assert abs(n - target) / target < 0.05, (scale, n, target)


def test_fifty_million_preset_is_the_default_and_near_768_by_6():
    cfg = CamLiveConfig()
    core = resolve_core(cfg.core)
    assert cfg.core.scale == "50m" and core.d_model == 768 and core.n_layers == 6 and core.n_heads == 12
    n = _count("50m")
    assert 47.5e6 < n < 52.5e6, n
    assert n == 50_953_765                      # the measured count recorded in docs/evidence/A/params.json


def test_real_instantiation_matches_the_shape_only_count():
    core = CoreConfig().apply_scale("5m")
    net = build_experiential(core, d_sem=QWEN_D_SEM, n_taps=TAPS)
    assert sum(p.numel() for p in net.parameters()) == _count("5m")
    assert all(torch.isfinite(p).all() for p in net.parameters())
    again = build_experiential(core, d_sem=QWEN_D_SEM, n_taps=TAPS)
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), again.parameters()))    # seeded init
    other = build_experiential(core, d_sem=QWEN_D_SEM, n_taps=TAPS, seed=1)
    assert not torch.equal(net.core.slot_init, other.core.slot_init)


def test_debug_preset_and_resolution_rules():
    m = build_model(CamLiveConfig.debug())
    n = m.count_parameters()["experiential"]
    assert abs(n - 0.2e6) / 0.2e6 < 0.1, n
    # an untouched master preset is replaced by the corrected one; a hand-edited core is respected
    core = CoreConfig().apply_scale("50m")
    assert resolve_core(core).d_ff == NN_PRESETS["50m"]["d_ff"] and preset_nn(core) == {"sem_rank": 384}
    custom = dataclasses.replace(core, d_ff=2048, n_layers=3)
    assert resolve_core(custom) is custom and preset_nn(custom) == {}
    cfg = CamLiveConfig.from_dict({"core": {"scale": "5m"}, "nn": {"bridge_layers": 2}})
    nn_cfg = NNConfig.from_cfg(cfg)
    assert nn_cfg.sem_rank == 96 and nn_cfg.bridge_layers == 2
    with pytest.raises(KeyError):
        NNConfig.from_cfg(CamLiveConfig.from_dict({"nn": {"no_such_option": 1}}))
    assert set(NN_PRESETS) == set(SCALE_PRESETS)
    for s in NN_PRESETS:
        assert NN_PRESETS[s]["target_params"] == SCALE_PRESETS[s]["target_params"]
        assert NN_PRESETS[s]["d_model"] == SCALE_PRESETS[s]["d_model"]


def test_scales_share_one_algorithm():
    """A larger preset is the same model with other numbers: same modules, same API (spec 16.10)."""
    names = None
    for scale in ("debug", "5m", "150m"):
        core = CoreConfig().apply_scale(scale)
        with torch.device("meta"):
            net = ExperientialNet(resolve_core(core), NNConfig(), 192, TAPS)
        top = sorted({n.split(".")[0] for n, _ in net.named_parameters()})
        names = names or top
        assert top == names
