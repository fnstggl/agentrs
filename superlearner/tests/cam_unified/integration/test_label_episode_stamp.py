"""run_lifetime stamps EventRecord.extra["label_episode"] from the label source's `episode_of` (W7/W8 change
request, wave-3 merge): label_join then matches labels to decisions by (env_id, episode) key, not archive order."""
from conftest import Stack, make_cfg
from superlearner.cam_unified import build
from superlearner.cam_unified import runtime as R
from superlearner.cam_unified.envs.factory import make_env
from superlearner.cam_unified.envs.labels import LabelStore


class EpisodeLabels:
    """A LabelSource that knows the label episode of every env id (no labels released)."""

    def __init__(self, ep):
        self.ep, self.asked = ep, []

    def take_labels(self):
        return []

    def episode_of(self, env_id):
        self.asked.append(env_id)
        return self.ep


def test_decisions_carry_the_label_episode_of_their_source(tmp_path):
    st = Stack(cfg=make_cfg(reward_mode="observation_only", reward_in_beliefs=False))
    src = EpisodeLabels(3)
    R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, labels=src, label_store=LabelStore(str(tmp_path / "l")),
                   max_decisions=3, update_every=0)
    recs = [r for r in st.archive.rows if r.action is not None]
    assert recs and all(r.extra.get("label_episode") == 3 for r in recs)
    assert set(src.asked) == {r.env_id for r in recs}


def test_no_episode_of_leaves_records_unstamped(tmp_path):
    st = Stack()
    R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=2, update_every=0)
    assert all("label_episode" not in r.extra for r in st.archive.rows)


def test_experience_operator_and_label_adapters_answer_episode_of():
    ad, op, labels = make_env("rapid", ["dl_delayed_returns"], compute_references=False)
    ad.reset()
    eid = ad.observe().env_ids[0]
    e0 = op.episode_of(eid)
    assert isinstance(e0, int) and op.episode_of("not-an-id") is None
    op.reset_world(0)
    assert op.episode_of(eid) == e0 + 1                       # the next decision belongs to the next episode
    assert R.label_episode_of(labels, eid) in (None, e0 + 1)  # the factory's LabelSource (operator or wrapper)
    assert build.OperatorLabels(op).episode_of(eid) == e0 + 1
    assert build.ConcatLabels([EpisodeLabels(None), build.OperatorLabels(op)]).episode_of(eid) == e0 + 1
