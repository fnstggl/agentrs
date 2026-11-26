"""W1 budget acceptance at the 1B scale (plan 2026-11-26 §2 memory budget, §4 G3): the real `build_agent("dom")`
system (32 x 1536 x 24 core, Qwen3-4B) on 16 real rapid worlds through the canonical loop, then one full optimiser
step over the 128 collected decisions (core/bench.benchmark_full).

    peak memory of one full optimiser step <= 40 GB
    decide at B=16 (agent.act) and at B*K = 64 rows (the cost model of K=4 candidates) <= 1.5 s
    an update of 128 decisions (fwd+bwd in microbatches of 4 + world/adapt losses + optimiser step) <= 90 s
    the core changes

Marked slow + gpu: it needs the GPU (it takes runs/cam_unified/gpu_heavy.lock itself, around the GPU part only,
for about 10 minutes) and the locally cached Qwen3-4B weights. Free-memory gate: after taking the lock it skips
(with the reason) unless >= 36 GB are free (CAM_GPU_WAIT_S seconds of waiting, default 0), so other tenants never
make it OOM. The step is measured in the plan configuration
available before the owners' merges: EMA target removed (shared online core; W6 deletes it), AdamW for the core and
Adafactor for the satellites (the plan's fallback), the not-yet-wired DOM adapters' parameters included
(--planned-adapters). Measured numbers of every variant tried are in docs/evidence/dom/W1/."""
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
VARIANT = "share_ema+adafactor"


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
        return benchmark_full("dom", B=16, K=4, rounds=8, microbatch=4, planned_adapters=True, variants=[VARIANT])


def test_dom_is_the_1b_core_with_qwen3_4b(bench):
    core = bench["parameters"]["components"]["core"]["total"]
    assert 0.85e9 < core < 1.2e9
    assert bench["semantic_model"] == "Qwen/Qwen3-4B"
    assert bench["acting"]["reward_mode"] == "observation_only"


def test_one_full_optimiser_step_fits_the_budget(bench):
    st = bench["steps"][VARIANT]
    assert "error" not in st, st
    assert st["decisions"] == 128 and st["core_hash_changed"]
    assert st["peak_GB"] <= BUDGET_GB, st["peak_GB"]


def test_decide_at_B16_and_B16xK4_within_1_5_s(bench):
    assert bench["acting"]["decide_s_warm_median"] <= 1.5
    assert bench["decide_BK"]["rows"] == 64 and bench["decide_BK"]["decide_s_warm_median"] <= 1.5


def test_update_of_128_decisions_within_90_s(bench):
    assert bench["steps"][VARIANT]["step_s"] <= 90.0
