"""label_join.join: verified labels -> every prefix of the resolved episode; pending excluded (plan §3 W8)."""
import pytest
import torch

from superlearner.cam_unified.envs.experience import OutcomeLabel
from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore
from superlearner.cam_unified.interfaces import Action, EventRecord, ExecutionResult
from superlearner.cam_unified.selfeval import label_join as LJ
from superlearner.cam_unified.selfeval.evaluator import FAILURE, SUCCESS, UNCERTAIN


def _rec(env, ep, step, t, action=True, kind=None, extra=None):
    ex = dict(extra or {})
    if kind:
        ex["kind"] = kind
    return EventRecord(event_id=f"{env}/{ep}/{step}/{kind or 'd'}", env_id=env, actor_id="a", episode_id=ep, step=step,
                       model_version=0, t_sim=t, observations=[],
                       action=Action("shop.status") if action else None,
                       result=ExecutionResult(ok=True) if action else None, extra=ex)


def _archive():
    out = []
    for env in ("w#0", "w#1"):
        for e, ep in enumerate(("ep_a", "ep_b")):
            for s in range(4):
                out.append(_rec(env, f"{env}:{ep}", s, 10.0 * e + s))
            out.append(_rec(env, f"{env}:{ep}", 4, 10.0 * e + 4, action=False, kind="post_action_observations"))
    return out


def test_every_prefix_of_resolved_episodes_pending_and_unlabelled_excluded(tmp_path):
    st = LabelStore(str(tmp_path))
    st.append(LabelRecord("w#0", 1, "success", 0.8, 120.0, 3.5, "op"))       # first episode of w#0 (episode_base 1)
    st.append(LabelRecord("w#1", 2, "failure", 0.1, 5.0, 14.0, "op"))        # second episode of w#1
    st.append(LabelRecord("w#1", 1, "uncertain", None, None, None, "op"))
    lb = LJ.join(st, _archive(), episode_base=1)
    # w#0 ep1: decisions at t=0..3 all <= 3.5; w#1 ep2: t=10..13 <= 14; w#1 ep1 (uncertain): 4 rows
    assert len(lb) == 12
    assert lb.stats["labels_joined"] == 3 and lb.stats["labels_unmatched"] == 0
    rows = {(e, k): [i for i in range(len(lb)) if (lb.env_ids[i], lb.episodes[i]) == (e, k)]
            for e, k in lb.episode_keys()}
    assert set(rows) == {("w#0", 1), ("w#1", 1), ("w#1", 2)}
    for idx in rows.values():
        assert float(lb.weight[idx].sum()) == pytest.approx(1.0)                # 1 / prefixes per episode
    i0 = rows[("w#0", 1)]
    assert lb.status[i0].tolist() == [SUCCESS] * 4
    assert lb.time_days[i0].tolist() == pytest.approx([3.5, 2.5, 1.5, 0.5])
    assert lb.utility[i0].tolist() == pytest.approx([0.8] * 4)
    assert lb.remaining_utility[i0].tolist() == pytest.approx([0.8] * 4)
    assert lb.status[rows[("w#1", 2)]].tolist() == [FAILURE] * 4
    iu = rows[("w#1", 1)]
    assert lb.status[iu].tolist() == [UNCERTAIN] * 4
    assert all(torch.isnan(lb.utility[iu]))
    assert not any("post_action" in e for e in lb.event_ids)                     # world-event records are not decisions


def test_pending_outcome_labels_are_never_joined_and_label_episode_stamp_wins():
    arch = _archive()
    for r in arch:
        if r.env_id == "w#0":
            r.extra["label_episode"] = 7 if r.episode_id.endswith("ep_a") else 8
    labs = [OutcomeLabel("w#0", 7, "pending", None, None, None), OutcomeLabel("w#0", 8, "success", 1.0, 1.0, 20.0)]
    lb = LJ.join(labs, arch)
    assert len(lb) == 4 and set(lb.episodes) == {8}
    assert set(lb.episode_ids) == {"w#0:ep_b"}


def test_decisions_after_resolution_and_other_label_kinds():
    arch = _archive()
    labs = [LabelRecord("w#0", 0, "success", 1.0, 1.0, 1.5, "op"),
            LabelRecord("w#0", 0, "success", 0.25, None, 0.5, "op:self", goal_id="g", kind="self_goal")]
    lb = LJ.join(labs, arch)                                                     # only terminal labels by default
    assert lb.t_sim.tolist() == [0.0, 1.0] and lb.stats["rows_after_resolution"] == 2
    lb2 = LJ.join(labs, arch, kinds=("terminal", "self_goal"))
    # remaining utility: the self-goal (resolved at 0.5) still counts from t=0, not from t=1
    assert lb2.remaining_utility.tolist() == pytest.approx([1.25, 1.0])
    lb3 = LJ.join(labs, arch, only_before_resolution=False)
    assert len(lb3) == 4 and lb3.time_days.tolist() == pytest.approx([1.5, 0.5, 0.0, 0.0])


def test_split_by_episode_is_disjoint_and_features_attach():
    labs = [LabelRecord(e, k, "success" if (k + int(e[-1])) % 2 else "failure", 0.5, 1.0, 100.0, "op")
            for e in ("w#0", "w#1") for k in (0, 1)]
    lb = LJ.join(labs, _archive())
    tr, ho = LJ.split_by_episode(lb, 0.5, seed=1)
    assert len(tr) + len(ho) == len(lb) and len(ho) > 0 and len(tr) > 0
    assert not set(tr.episode_keys()) & set(ho.episode_keys())
    lb2 = lb.with_features(torch.randn(len(lb), 3, 8))
    assert lb2.R.shape == (16, 3, 8) and lb.R is None
    assert lb2.select([0, 5]).R.shape == (2, 3, 8)
    with pytest.raises(ValueError):
        lb.with_features(torch.randn(3, 3, 8))
