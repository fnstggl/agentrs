"""Each direct tool result reaches the next decision exactly once through the canonical loop (wave-3 merge fix).

W2's adapters re-deliver every direct result at the head of the next ObservationBatch (`ow_convert.delivered_copy`);
run_lifetime used to merge `ExecutionResult.observations` as well, so the agent saw every result twice (W2's
integration measurement: 36/36 results duplicated). `runtime.drop_delivered` drops the ExecutionResult copy when the
adapter has delivered it; a non-delivering adapter still gets its results merged."""
from collections import Counter

from conftest import Stack
from superlearner.cam_unified import runtime as R
from superlearner.cam_unified.dist import actor as dist_actor
from superlearner.cam_unified.envs.factory import make_env
from superlearner.cam_unified.envs.ow_convert import delivered_copy, is_delivered_result, result_key
from superlearner.cam_unified.interfaces import Action, ObservationBatch, ObsKind


class Delivering:
    """Wraps an adapter the way W2's adapters behave: direct results also arrive in the next batch."""

    delivers_results = True

    def __init__(self, inner):
        self.inner, self.pending = inner, None

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def execute(self, actions):
        res = self.inner.execute(actions)
        self.pending = [list(r.observations) if r is not None else [] for r in res]
        return res

    def advance_time(self, duration):
        nb = self.inner.advance_time(duration)
        if self.pending is None:
            return nb
        obs = [[delivered_copy(o) for o in p] + list(o2) for p, o2 in zip(self.pending, nb.observations)]
        self.pending = None
        return ObservationBatch(list(nb.env_ids), obs, list(nb.t), list(nb.done))


def _spy_inputs(agent):
    inputs, act0 = [], agent.act

    def spy(obs, tools):
        inputs.append([list(o) for o in obs.observations])
        return act0(obs, tools)
    agent.act = spy
    return inputs


def _result_counts(rows):
    return Counter(result_key(o) for o in rows if o.kind in (ObsKind.TOOL_RESULT, ObsKind.ERROR))


def test_run_lifetime_delivers_each_result_once_with_a_delivering_adapter():
    st = Stack()
    inputs = _spy_inputs(st.agent)
    R.run_lifetime(st.agent, Delivering(st.env), evaluator=st.evaluator, max_decisions=4, update_every=0)
    n = 0
    for rows in inputs[1:]:
        for b in range(len(rows)):
            c = _result_counts(rows[b])
            assert c and max(c.values()) == 1, c            # every row sees its result, never twice
            n += sum(c.values())
            assert all(is_delivered_result(o) for o in rows[b]                   # the adapter's copy is the one kept
                       if o.kind in (ObsKind.TOOL_RESULT, ObsKind.ERROR))
    assert n >= 2 * 3


def test_run_lifetime_still_merges_results_of_a_non_delivering_adapter():
    st = Stack()
    inputs = _spy_inputs(st.agent)
    R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=3, update_every=0)
    for rows in inputs[1:]:
        for b in range(len(rows)):
            c = _result_counts(rows[b])
            assert c and max(c.values()) == 1


def test_drop_delivered_on_a_real_experience_adapter_and_concat_flag():
    ad, op, labels = make_env("rapid", ["dl_delayed_returns"], compute_references=False)
    ad.reset()
    tool = ad.available_tools()[0][0].name
    a = Action(tool, {"kind": "levers", "values": {}})
    (r,) = ad.execute([a])
    nb = ad.advance_time([1 / 24])
    assert r.observations and R.drop_delivered(r.observations, nb.observations[0]) == []
    merged, _ = R.merge_by_seq(-1, R.drop_delivered(r.observations, nb.observations[0]), nb.observations[0])
    assert max(_result_counts(merged).values()) == 1
    assert R.drop_delivered(r.observations, []) == list(r.observations)   # nothing delivered: keep them all
    assert R.ConcatEnv([ad]).delivers_results and R.GuardedEnv(ad).delivers_results
    assert dist_actor.merge_by_seq is R.merge_by_seq and dist_actor.drop_delivered is R.drop_delivered
