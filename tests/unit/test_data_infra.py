import json

import pytest

from agentrs.agent.runtime import RandomActor, TeacherActor, run_episode
from agentrs.checkpoints.registry import CheckpointRegistry, PromotionGate
from agentrs.config import apply_overrides, load_config
from agentrs.data.replay import LeakageError, ReplayBuffer
from agentrs.data.schemas import Provenance, Source, Transition
from agentrs.data.storage import ExperienceStore, migrate_record
from agentrs.demonstrations.teacher import ScriptedTeacher
from agentrs.environments.miniweb.core import MiniWebEnv
from agentrs.environments.miniweb.tasks import make_task, task_set
from agentrs.evaluation.metrics import auroc, bootstrap_ci, expected_calibration_error
from agentrs.memory.episodic import EpisodicMemory
from agentrs.training.curriculum import Curriculum, CurriculumConfig
from agentrs.training.returns import discounted_returns, gae


@pytest.fixture
def episodes():
    env = MiniWebEnv()
    out = []
    for i, t in enumerate(task_set("train", ["fill_field", "login"], n_per_family=3)):
        out.append(run_episode(env, t, TeacherActor(ScriptedTeacher(0.2, i)),
                               Provenance(source=Source.TEACHER, source_id="t")))
        out.append(run_episode(env, t, RandomActor(i)))
    return out


def test_store_roundtrip_and_partitions(tmp_path, episodes):
    st = ExperienceStore(tmp_path)
    st.write_episodes(episodes)
    back = {e.episode_id: e for e in st.load_episodes()}
    for e in episodes:
        b = back[e.episode_id]
        assert [t.model_dump() for t in b.transitions] == [t.model_dump() for t in e.transitions]
    # private human demonstrations are physically separated
    t0 = episodes[0].transitions[0]
    priv = t0.model_copy(update={"provenance": Provenance(source=Source.HUMAN, consent=True, privacy="private")})
    st.write_transitions([priv])
    assert st.summary()["private"]["transitions"] == 1
    assert all(t.provenance.privacy != "private" for t in st.load_transitions("real"))


def test_blob_store(tmp_path):
    st = ExperienceStore(tmp_path)
    h = st.put_blob(b"\x89PNG fake screenshot")
    assert st.get_blob(h) == b"\x89PNG fake screenshot" and len(h) == 64


def test_migration_v1(episodes):
    d = json.loads(episodes[0].transitions[0].model_dump_json())
    d["schema_version"] = 1
    d.pop("selection")
    d.pop("behavior_logprob")
    d["policy_logprob"] = -0.5
    t = Transition.model_validate(migrate_record(d))
    assert t.schema_version == 3 and t.behavior_logprob == -0.5 and t.selection == "policy"
    assert t.semantic_action is None and t.content_outcome is None


def test_imagined_and_human_schema_guards(episodes):
    t = episodes[0].transitions[0]
    with pytest.raises(ValueError):  # imagined experience can never carry a verified outcome
        Transition.model_validate({**t.model_dump(), "provenance": {"source": "imagined"}, "episode_success": True})
    with pytest.raises(ValueError):
        Transition.model_validate({**t.model_dump(), "provenance": {"source": "imagined"},
                                   "verification": {"success": True}})
    with pytest.raises(ValueError):
        Transition.model_validate({**t.model_dump(), "provenance": {"source": "human", "consent": False}})


def test_replay_leakage_and_strategies(episodes):
    rb = ReplayBuffer(seed=0)
    for e in episodes:
        rb.add_episode(e)
    assert len(rb) == sum(e.num_steps for e in episodes)
    env = MiniWebEnv()
    test_ep = run_episode(env, make_task("fill_field", "test", 0), RandomActor(0))
    with pytest.raises(LeakageError):
        rb.add_episode(test_ep)
    for s in ("uniform", "recent", "historical", "success", "failure", "stratified", "balanced_old_new"):
        assert len(rb.sample(8, s)) == 8
    succ = rb.sample(30, "success")
    assert all(t.episode_success for t in succ)
    assert rb.sample(5, "uniform", source_weights={"teacher": 1.0, "agent": 0.0})[0].provenance.source == Source.TEACHER


def test_returns_and_gae():
    assert discounted_returns([0, 0, 1], 0.9, terminated=True) == pytest.approx([0.81, 0.9, 1.0])
    # truncation bootstraps with V(s_T)
    assert discounted_returns([0, 0], 0.5, terminated=False, last_value=1.0) == pytest.approx([0.25, 0.5])
    adv, rets = gae([0, 1], [0.5, 0.5], gamma=1.0, lam=1.0, terminated=True)
    assert rets == pytest.approx([1.0, 1.0]) and adv == pytest.approx([0.5, 0.5])
    adv, rets = gae([0, 0], [0.0, 0.0], gamma=1.0, lam=1.0, terminated=False, last_value=0.7)
    assert rets == pytest.approx([0.7, 0.7])


def test_metrics():
    assert auroc([0.1, 0.9], [0, 1]) == 1.0
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert expected_calibration_error([0.5] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.0)
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi


def test_registry_promotion_gate_and_rollback(tmp_path):
    reg = CheckpointRegistry(tmp_path / "r.sqlite")
    a = reg.register("policy", "/a")
    reg.promote(a)
    b = reg.register("policy", "/b", parent=a)
    gate = PromotionGate(max_family_regression=0.2)
    ok, why = gate.decide({"success_rate": 0.6, "invalid_action_rate": 0.0, "by_family": {"x": 0.1}},
                          {"success_rate": 0.5, "by_family": {"x": 0.9}})
    assert not ok and "regression" in why
    ok, _ = gate.decide({"success_rate": 0.6, "invalid_action_rate": 0.0}, {"success_rate": 0.5})
    assert ok
    reg.promote(b)
    assert reg.production("policy").id == b
    assert reg.rollback("policy") == a
    assert reg.production("policy").id == a
    assert reg.get(b).status == "rolled_back"


def test_curriculum_advances_only_on_competence():
    cur = Curriculum(CurriculumConfig(stages=[["a"], ["b"]], advance_threshold=0.6, min_episodes=10, window=10))
    for _ in range(50):
        cur.record("a", False)
    assert not cur.maybe_advance() and cur.stage == 0
    for _ in range(10):
        cur.record("a", True)
    assert cur.maybe_advance() and cur.stage == 1


def test_episodic_memory_retrieval_and_no_eval_memorisation(tmp_path, episodes):
    mem = EpisodicMemory(tmp_path / "m.sqlite")
    for e in episodes:
        mem.add_episode(e)
    n = len(mem)
    mem.add_episode(run_episode(MiniWebEnv(), make_task("login", "test", 0), RandomActor(0)))
    assert len(mem) == n  # held-out episodes are never stored
    r = mem.retrieve('Log in with username "zed1" and password "x#11".', k=1)
    assert r and r[0]["family"] == "login"


def test_config_overrides(tmp_path):
    from agentrs.training.pipeline import ExperimentConfig

    p = tmp_path / "c.yaml"
    p.write_text("variant: C\nrl: {algo: reinforce}\n")
    cfg = load_config(ExperimentConfig, p, ["rl.algo=grpo", "seed=3", "planner.horizon=2"])
    assert cfg.rl.algo == "grpo" and cfg.seed == 3 and cfg.planner.horizon == 2 and cfg.variant == "C"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):  # typos fail loudly
        load_config(ExperimentConfig, p, ["rl.clip_epsilon=0.1"])
    with pytest.raises(ValidationError):
        load_config(ExperimentConfig, p, ["rl.clip_eps=not-a-number"])
    assert apply_overrides({}, ["a.b=2"]) == {"a": {"b": 2}}
