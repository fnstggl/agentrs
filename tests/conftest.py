import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True, scope="session")
def _threads():
    import torch

    torch.set_num_threads(int(os.environ.get("AGENTRS_TEST_THREADS", "2")))


@pytest.fixture(scope="session")
def scratch_bb():
    from agentrs.models.backbone import BackboneConfig, ScratchSettings

    return BackboneConfig(name="scratch", scratch=ScratchSettings(hidden_size=48, num_layers=2, num_heads=4,
                                                                  vocab_size=512), max_prompt_tokens=384)


@pytest.fixture(scope="session")
def demo_data():
    """Teacher demonstrations (train) and validation episodes for a few families."""
    from agentrs.agent.runtime import RandomActor, TeacherActor, run_episode
    from agentrs.data.schemas import Provenance, Source
    from agentrs.demonstrations.teacher import ScriptedTeacher
    from agentrs.environments.miniweb.tasks import task_set
    from agentrs.evaluation.benchmark import make_env

    env = make_env("miniweb")
    fams = ["click_button", "fill_field", "menu"]
    prov = Provenance(source=Source.TEACHER, source_id="teacher:test")
    train, val = [], []
    for i, t in enumerate(task_set("train", fams, n_per_family=12)):
        train += run_episode(env, t, TeacherActor(ScriptedTeacher(0.2, i)), prov).transitions
        train += run_episode(env, t, RandomActor(i)).transitions
    for i, t in enumerate(task_set("val", fams, n_per_family=4)):
        val += run_episode(env, t, TeacherActor(ScriptedTeacher(0.2, i)), prov).transitions
    return train, val
