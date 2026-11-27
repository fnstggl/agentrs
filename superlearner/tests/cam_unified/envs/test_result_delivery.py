"""W2 (plan 2026-11-26 §2 step 5): a tool's direct result reaches the next decision's input, once, with
``caused_by`` = the action id - for OpenWorld (in-process and process-isolated) and the Crucible 5.0 experience
adapter, whichever loop drives the adapter (the audit measured 0 of 9 decision inputs carrying a tool result)."""
from __future__ import annotations

import os
import sys

import pytest

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

from superlearner.cam_unified.envs import ow_convert as C  # noqa: E402
from superlearner.cam_unified.envs.factory import make_env, merge_results  # noqa: E402
from superlearner.cam_unified.interfaces import Action, ObservationBatch, ObsKind  # noqa: E402

THINK = 600 / 86400


def _results_of(batch, b, action):
    return [o for o in batch.observations[b] if o.caused_by == action.action_id and C.is_delivered_result(o)]


@pytest.fixture()
def ow():
    es = make_env("openworld", tasks=["open_economy#100000"], isolated=False)
    yield es
    es.close()


def test_openworld_result_arrives_once_at_the_head_of_the_next_batch(ow):
    ad, op, ev = ow
    assert ad.delivers_results
    ad.reset()
    a = Action("world.hosts", {})
    (r,) = ad.execute([a])
    assert r.ok and all(o.caused_by == a.action_id for o in r.observations)
    nb = ad.advance_time([r.duration + THINK])
    got = _results_of(nb, 0, a)
    assert [o.kind for o in got] == [o.kind for o in r.observations]          # TOOL_RESULT + its NUMERIC leaves
    assert [o.content for o in got] == [o.content for o in r.observations]
    assert all(o.meta["result_of"] == a.action_id for o in got)
    assert nb.observations[0][:len(got)] == got                               # first in the batch
    assert got[0].kind is ObsKind.TOOL_RESULT and "host" in got[0].content
    nb2 = ad.advance_time([THINK])
    assert not [o for o in nb2.observations[0] if o.caused_by == a.action_id and C.is_delivered_result(o)]


def test_errors_waits_and_observe_deliver_too(ow):
    ad, op, ev = ow
    ad.reset()
    bad = Action("no.such_tool", {})
    (r,) = ad.execute([bad])
    assert not r.ok
    ob = ad.observe()                                    # observe() is a delivery as well
    (e,) = _results_of(ob, 0, bad)
    assert e.kind is ObsKind.ERROR and e.content.startswith("unknown_tool")
    w = Action("wait", {"seconds": 120})
    (rw,) = ad.execute([w])
    nb = ad.advance_time([rw.duration + THINK])
    (o,) = _results_of(nb, 0, w)
    assert o.kind is ObsKind.TOOL_RESULT and '"waited_s": 120' in o.content


def test_harness_decision_after_world_hosts_sees_its_result(ow):
    """Driven by the lifetime harness (W1's legacy shim, `batch = nxt`): the next act() input carries the result."""
    from superlearner.cam_unified.envs.harness import run_lifetime

    class Scripted:
        def __init__(self):
            self.inputs, self.actions = [], []

        def act(self, obs: ObservationBatch, tools):
            self.inputs.append(obs)
            a = Action("world.hosts" if len(self.actions) == 0 else "world.whoami", {})
            self.actions.append(a)
            return [a]

        def observe(self, transition):
            pass

        def update(self):
            return {}

    ad, op, ev = ow
    ag = Scripted()
    run_lifetime(ag, ad, ev, max_decisions=3)
    first = ag.actions[0]
    hits = [o for o in ag.inputs[1].observations[0] if o.kind is ObsKind.TOOL_RESULT and o.caused_by == first.action_id]
    assert len(hits) == 1
    assert not [o for o in ag.inputs[2].observations[0] if o.kind is ObsKind.TOOL_RESULT
                and o.caused_by == first.action_id]
    assert [o for o in ag.inputs[2].observations[0] if o.caused_by == ag.actions[1].action_id]


def test_process_isolated_operator_delivers_results():
    es = make_env("openworld", tasks=["open_economy#100001"], isolated=True)
    ad, rop, rev = es
    try:
        assert rop.pid != os.getpid() and ad.delivers_results
        ad.reset()
        a = Action("world.whoami", {})
        (r,) = ad.execute([a])
        nb = ad.advance_time([r.duration + THINK])
        got = _results_of(nb, 0, a)
        assert got and got[0].kind is ObsKind.TOOL_RESULT and got[0].content == r.observations[0].content
    finally:
        es.close()


def test_undelivered_results_survive_an_operator_checkpoint(ow, tmp_path):
    ad, op, ev = ow
    ad.reset()
    a = Action("world.hosts", {})
    ad.execute([a])
    op.checkpoint(str(tmp_path / "op.pkl"))
    op.restore(str(tmp_path / "op.pkl"))
    nb = ad.advance_time([THINK])
    assert _results_of(nb, 0, a)


def test_legacy_constructor_is_unchanged():
    from superlearner.cam_unified.envs.openworld import open_world
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    ad, op = open_world([open_economy_task(seed=100_002)])
    try:
        assert not ad.delivers_results
        ad.reset()
        a = Action("world.whoami", {})
        ad.execute([a])
        nb = ad.advance_time([THINK])
        assert not [o for o in nb.observations[0] if o.kind is ObsKind.TOOL_RESULT and o.caused_by == a.action_id]
    finally:
        ad.close()


def test_experience_adapter_delivers_results():
    es = make_env("rapid", ["dl_delayed_returns"], compute_references=False)
    ad, op, labels = es
    assert ad.delivers_results
    ad.reset()
    tool = ad.available_tools()[0][0].name
    a = Action(tool, {"kind": "levers", "values": {}})
    (r,) = ad.execute([a])
    nb = ad.advance_time([1 / 24])
    got = _results_of(nb, 0, a)
    assert len(got) == 1 and got[0].content == r.observations[0].content and got[0] is not r.observations[0]
    nb2 = ad.advance_time([1 / 24])
    assert not _results_of(nb2, 0, a)


def test_merge_results_is_idempotent(ow):
    ad, op, ev = ow
    ad.reset()
    a = Action("world.whoami", {})
    res = ad.execute([a])
    nb = ad.advance_time([THINK])
    merged = merge_results(res, nb)
    assert len(merged[0]) == len(nb.observations[0])           # already delivered: nothing added
    bare = ObservationBatch(nb.env_ids, [[o for o in nb.observations[0] if not C.is_delivered_result(o)]], nb.t,
                            nb.done)
    merged2 = merge_results(res, bare)
    assert merged2[0][:len(res[0].observations)] == res[0].observations    # a non-delivering loop gets them added


def test_capability_env_payouts_are_observable_and_results_delivered_with_their_seq():
    from superlearner.cam_unified.capabilities.envs import ProbeEnv
    es = make_env("capability", ["probe"], n_envs=2, seed=3)
    env, _, ev = es
    assert env.delivers_results and env.observable_payouts
    env.reset()
    acts = [Action("play_safe", {}), Action("run_test", {})]
    res = env.execute(acts)
    for b, r in enumerate(res):
        pay = [o for o in r.observations if o.source == "payout"]
        bal = [o for o in r.observations if o.source == "balance"]
        assert pay and bal and pay[0].kind is ObsKind.NUMERIC and pay[0].value == pytest.approx(env.returns[b])
        assert all(o.caused_by == acts[b].action_id for o in r.observations)
    nb = env.advance_time([THINK, THINK])
    for b, r in enumerate(res):
        assert [o.seq for o in nb.observations[b]] == [o.seq for o in r.observations]      # same seq: dedupable
    assert env.advance_time([THINK, THINK]).observations == [[], []]
    legacy = ProbeEnv(2, seed=3)                                  # the default diagnostics are unchanged
    legacy.reset()
    lr = legacy.execute(acts)
    assert not [o for r in lr for o in r.observations if o.source in ("payout", "balance") or o.caused_by]
    assert legacy.advance_time([THINK, THINK]).observations == [[], []]
