"""W1 budget acceptance at the 1B scale (plan 2026-11-26 §2 memory budget, §4 G3): the real `build_agent("dom")`
system (32 x 1536 x 24 core, Qwen3-4B) on 16 real rapid worlds through the canonical loop, then the system's own
update (IntegratedLearner.step: replay world / adapt losses, the policy recompute in cfg.microbatch microbatches, PPO
epochs, the named-group probe and share guard, role optimisers) over the 128 collected decisions, twice (the second
update runs with the optimiser moments resident: the steady-state peak) - core/bench.benchmark_full.

    peak memory of a full optimiser step (both updates) <= 40 GB
    decide at B=16 with K=4 candidates per row sharing one encode (agent.act, cfg.n_candidates = 4) <= 1.5 s
    an update of 128 decisions <= 90 s (the slower of the two)
    the core changes

Marked slow + gpu: it needs the GPU (it takes runs/cam_unified/gpu_heavy.lock itself, around the GPU part only,
for about 10 minutes) and the locally cached Qwen3-4B weights. Free-memory gate: after taking the lock it skips
(with the reason) unless >= 36 GB are free (CAM_GPU_WAIT_S seconds of waiting, default 0), so other tenants never
make it OOM. The learner runs in the configuration G3 selected (OVERRIDES / VARIANT below; every variant measured is
in docs/evidence/dom/G3G4/)."""
import contextlib
import fcntl
import os
import time

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

LOCK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "runs", "cam_unified",
                                    "gpu_heavy.lock"))
BUDGET_GB = 40.0
MIN_FREE_GB = 36.0          # free-memory gate: other tenants of the shared GPU must leave room for the 1B step
OVERRIDES: dict = {}            # build overrides of the selected G3 configuration (filled from the G3 evidence)
VARIANT = "adafactor+mb2+ols2"  # learner variant of core/bench (sat_optimizer, cfg.microbatch, open-loop starts)


@contextlib.contextmanager
def gpu_lock(timeout_s: float = 3 * 3600):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    fd = open(LOCK, "a")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > timeout_s:
                fd.close()
                pytest.skip("GPU lock busy")
            time.sleep(5)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


@pytest.fixture(scope="module")
def bench():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    if not os.path.isdir(os.path.join(hub, "models--Qwen--Qwen3-4B")):
        pytest.skip("Qwen3-4B weights not cached")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from superlearner.cam_unified.core.bench import benchmark_full, wait_free
    with gpu_lock():
        gate = wait_free(MIN_FREE_GB, float(os.environ.get("CAM_GPU_WAIT_S", "0")))
        if not gate["ok"]:
            pytest.skip(f"GPU contention: {gate['free_GB']} GB free < {MIN_FREE_GB} GB (other tenants); the budget "
                        "check needs the GPU to itself (set CAM_GPU_WAIT_S to wait)")
        return benchmark_full("dom", B=16, K=4, rounds=8, overrides=OVERRIDES, variants=["none"],
                              agent_update_rows=128, learner_variants=[VARIANT])


def test_dom_is_the_1b_core_with_qwen3_4b(bench):
    core = bench["parameters"]["components"]["core"]["total"]
    assert 0.85e9 < core < 1.2e9
    assert bench["semantic_model"] == "Qwen/Qwen3-4B"
    assert bench["acting"]["reward_mode"] == "observation_only"


def test_one_full_optimiser_step_fits_the_budget(bench):
    u = bench["agent_updates"][VARIANT]
    assert u["error"] is None, u
    for st in (u["first"], u["second"]):
        assert st["updated"] and st["decisions_recomputed"] == 128 and st["core_hash_changed"], st
    assert u["peak_GB"] <= BUDGET_GB, u["peak_GB"]


def test_decide_at_B16_with_K4_candidates_within_1_5_s(bench):
    a = bench["acting"]
    assert a["n_candidates_cfg"] == 4 and a["n_candidates_actions"] == [4]
    assert a["decide_s_warm_median"] <= 1.5, a["decide_s"]


def test_update_of_128_decisions_within_90_s(bench):
    assert bench["agent_updates"][VARIANT]["s_max"] <= 90.0
