"""Verifier F3 / DOM finding 2 (2026-11-27): the canonical factory attaches the actor reward model R_psi to the model,
so Phi enters the evaluator's Q and every DOM decision stamps the psi version; the learner trains the same object."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu]


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from superlearner.cam_unified.build import build_agent
    return build_agent("dom_dev", device="cuda", run_dir=str(tmp_path_factory.mktemp("rm_attached")))


def test_factory_attaches_the_learners_reward_model(system):
    rm = system.model.__dict__.get("reward_model")
    assert rm is not None and rm is system.reward_model
    assert getattr(system.learner, "reward_model", rm) is rm
    assert system.model.decision_versions()["reward_model"] is not None


def test_reward_model_is_not_a_submodule_of_the_model(system):
    ids = {id(p) for p in system.model.parameters()}
    assert not any(id(p) in ids for p in system.reward_model.parameters())
