"""W6 real end-to-end check as a pytest acceptance test (plan 2026-11-26 §3 W6): `capabilities.causal.run_causal` on
the real dom_dev stack (QwenSemantic Qwen3-0.6B, ObservationTokenizer, UnifiedCore 8 x 384, Reasoner, BeliefModule,
BayesLastLayer(fusion), OutcomeHead, WorldModel with the block transition on R_t and no EMA target), observation-only
(the verified return is never a target; payouts are read from the observable points counter), three seeds, every
world. Criteria: t1 > 0.5, t2 max error < 0.3, |t3 contrast - 2| < 0.4, t4 switch correct, consistency MSE/Var < 0.1.

GPU and the cached Qwen3 weights are required; it runs for hours, so it is marked gpu + slow (CAM_RUN_SLOW=1). The
evidence run uses the same driver one (world, seed) per process (`python -m superlearner.cam_unified.capabilities.causal
run ...`), each capped at 3.8 GB of GPU memory; the results are in docs/evidence/dom/W6/e2e_dom_dev_causal.json (t1 and
t3 pass, t2 / t4 / the no-EMA consistency gap do not).
"""
import os

import pytest
import torch

from superlearner.cam_unified.capabilities.causal import DOM_SETTINGS, CausalSettings, criteria, run_causal

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("CAM_RUN_SLOW") != "1",
                    reason="needs CUDA, cached Qwen3-0.6B and CAM_RUN_SLOW=1")
@pytest.mark.parametrize("world", ["delay", "interaction", "switch"])
def test_dom_dev_causal_criteria_hold_on_three_seeds(world, tmp_path):
    fails = []
    for seed in (0, 1, 2):
        res = run_causal(CausalSettings(seed=seed, worlds=(world,), **DOM_SETTINGS), root=str(tmp_path / str(seed)))
        assert res["worlds"][world]["config"]["preset"] == "dom_dev"
        assert res["worlds"][world]["config"]["reward_mode"] == "observation_only"
        assert not res["worlds"][world]["world_model_has_target_core"]
        for k, v in criteria(res["worlds"]).items():
            if not v["pass"]:
                fails.append((seed, k, v["value"]))
    assert not fails, fails
