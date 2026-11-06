import torch

from agentrs.agent.runtime import TeacherActor, run_episode
from agentrs.data.schemas import Source
from agentrs.demonstrations.teacher import ScriptedTeacher
from agentrs.environments.miniweb.tasks import make_task, task_set
from agentrs.evaluation.benchmark import make_env
from agentrs.memory.skills import SkillExecutor, SkillLibrary, discover_skills
from agentrs.models.idm import IDMConfig, InverseDynamicsModel
from agentrs.training.common import OptimConfig
from agentrs.training.idm import IDMTrainConfig, evaluate_idm, label_recording, train_idm


def test_idm_learns_and_labels_with_confidence(scratch_bb, demo_data):
    torch.manual_seed(0)
    train, val = demo_data
    idm = InverseDynamicsModel(IDMConfig(backbone=scratch_bb, abstain_threshold=0.5))
    before = evaluate_idm(idm, val)
    r = train_idm(idm, train, IDMTrainConfig(steps=150, optim=OptimConfig(lr=3e-3, total_steps=150)))
    assert r["loss_last10"] < r["loss_first10"]
    after = evaluate_idm(idm, val)
    assert after["accuracy"] > before["accuracy"] + 0.2
    # label an action-free recording (observations only)
    ep = run_episode(make_env("miniweb"), make_task("fill_field", "val", 7), TeacherActor(ScriptedTeacher()))
    obs_seq = [ep.transitions[0].observation] + [t.next_observation for t in ep.transitions]
    labelled = label_recording(idm, ep.goal, obs_seq, split="val")
    for t in labelled:
        assert t.provenance.source == Source.INFERRED and 0.5 <= t.action_confidence <= 1.0


def test_skill_executor_expands_into_guarded_primitive_actions():
    lib = SkillLibrary()

    class AlwaysSkill:
        name = "pick-skill"

        def begin_episode(self, b):
            pass

        def decide(self, belief, cands):
            from agentrs.agent.runtime import Decision

            sk = [c for c in cands if c.to_str() == "skill(fill_and_submit)"]
            return Decision(action=sk[0] if sk else cands[-1])

    env = make_env("miniweb")
    ex = SkillExecutor(AlwaysSkill(), lib)
    succ = 0
    for t in task_set("train", ["login", "fill_field"], n_per_family=4):
        ep = run_episode(env, t, ex)
        ex.end_episode(bool(ep.success))
        succ += bool(ep.success)
        assert all(not tr.action.startswith("skill(") for tr in ep.transitions)  # only primitives executed
        assert any(tr.selection.startswith("skill:") for tr in ep.transitions)
    assert succ >= 4
    sk = lib.skills["fill_and_submit"]
    assert sk.successes + sk.failures == 8 and 0 < sk.reliability < 1


def test_skill_discovery_from_successful_trajectories():
    env = make_env("miniweb")
    trajs = []
    for t in task_set("train", ["login", "fill_field", "checkout"], n_per_family=5):
        ep = run_episode(env, t, TeacherActor(ScriptedTeacher()))
        if ep.success:
            trajs.append([tr.action for tr in ep.transitions])
    found = dict(discover_skills(trajs, min_count=5))
    assert ("type", "click") in found and ("type", "type") in found
