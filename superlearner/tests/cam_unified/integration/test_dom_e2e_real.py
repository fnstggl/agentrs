"""W1 real end-to-end check at dom_dev (8 x 384 core, real Qwen3-0.6B, real Crucible rapid and ow_tiny worlds):
build_agent -> one ConcatEnv batch -> runtime.run_lifetime observation-only -> two learner updates through the
LearnerHook (stage1/dom_e2e.run_dom_e2e). No test doubles. The 1B variant is the CLI evidence run
(`python -m superlearner.cam_unified.stage1.dom_e2e --preset dom ...`, docs/evidence/dom/W1/).

Marked slow + gpu (Qwen3-0.6B weights; the update peaks near 9 GB, so the GPU part takes
runs/cam_unified/gpu_heavy.lock)."""
import pytest

from test_memory_budget_1b import gpu_lock

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from superlearner.cam_unified.stage1.dom_e2e import run_dom_e2e
    with gpu_lock():
        return run_dom_e2e("dom_dev", n_rapid=4, n_ow=1, rounds=16, update_every=8, require_cam_eval=False,
                           run_dir=str(tmp_path_factory.mktemp("dom_e2e")))


def test_observation_only_through_the_canonical_loop(e2e):
    assert e2e["reward_mode"] == "observation_only"
    assert e2e["rewards_none_every_round"] and e2e["record_rewards_zero"]
    assert e2e["B"] == 5 and e2e["rounds"] == 16 and e2e["decisions"] == 80
    assert set(e2e["versions"]) == {"policy", "reward_model", "evaluator", "env", "code"}
    assert e2e["versions"]["reward_model"] == 0          # the seam RewardModel stub until W8 merges


def test_two_updates_change_the_core(e2e):
    assert e2e["n_updates"] == 2 and e2e["core_hash_changed"] and e2e["agent_version"] == 2
    assert all(u["decisions"] > 0 and {"world", "actor", "value"} <= set(u["losses"] or {}) for u in e2e["updates"])


def test_no_private_world_identity_in_actor_side_content(e2e):
    hits = e2e["canary_scan"]["content_hits"]
    assert e2e["canary_scan"]["records_scanned"] >= 80
    assert not [k for k in hits if k.startswith("archive:")], hits


def test_unstamped_actions_are_never_executed_silently(tmp_path):
    """With the authorisation armed by the dom_dev config, a run either executes only cam_eval-stamped actions (W5's
    decision path) or refuses before the first execution."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from superlearner.cam_unified.runtime import ActionNotAuthorised
    from superlearner.cam_unified.stage1.dom_e2e import run_dom_e2e
    with gpu_lock():
        try:
            r = run_dom_e2e("dom_dev", n_rapid=2, n_ow=0, rounds=2, update_every=0, run_dir=str(tmp_path))
        except ActionNotAuthorised:
            return
    assert r["authorisation"]["required"] and r["actions_stamped"] == r["actions"] > 0
