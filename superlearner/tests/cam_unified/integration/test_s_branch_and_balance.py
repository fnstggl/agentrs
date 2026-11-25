"""Stage-1 integration pieces that need no GPU: the abandoned-branch archive view over E's real EventArchive and
ReplaySampler, and the agent's per-objective magnitude cap."""
import pytest
import torch

from conftest import Stack
from superlearner.cam_unified.interfaces import Action, EventRecord, ExecutionResult, Observation, ObsKind
from superlearner.cam_unified.runtime import run_steps
from superlearner.cam_unified.stage1.branch import BranchView, abandoned_after


def _rec(i, step, ep="ep_1"):
    return EventRecord(event_id=f"e{i}", env_id="env", actor_id="a", episode_id=ep, step=step, model_version=0,
                       t_sim=0.01 * i, observations=[Observation(ObsKind.TEXT, content=f"obs {i}")],
                       action=Action("bank.accounts", {}), result=ExecutionResult(True), extra={"outcomes": []})


def test_branch_view_hides_abandoned_events_from_replay_and_consolidation_reads(tmp_path):
    from superlearner.cam_unified.memory.archive import EventArchive
    from superlearner.cam_unified.memory.replay import ReplaySampler
    arc = EventArchive(str(tmp_path / "arc"), chunk_rows=4)
    arc.append([_rec(i, i) for i in range(6)])            # the lifetime up to the checkpoint (count 6)
    arc.append([_rec(i, i - 0) for i in range(6, 9)])      # steps 6..8 of the interrupted process (abandoned)
    arc.append([_rec(i, i - 3) for i in range(9, 12)])     # the resumed lifetime re-lives steps 6..8
    arc.flush()
    ab = abandoned_after(arc, 6)
    assert [e for e, _ in ab] == [f"e{i}" for i in range(6, 12)]         # everything after the checkpoint
    view = BranchView(arc)
    view.add(ab[:3])                                                    # only the interrupted process's events
    rows = [r for b in view.scan_meta(columns=["gseq", "event_id"]) for r in b]
    assert [r["event_id"] for r in rows] == [f"e{i}" for i in list(range(6)) + list(range(9, 12))]
    rows = [r for b in view.scan_meta(columns=["gseq", "step"]) for r in b]   # filtered by gseq without event_id
    assert len(rows) == 9
    recs = [r for b in view.scan() for r in b]
    assert {r.event_id for r in recs}.isdisjoint({"e6", "e7", "e8"})
    win, i = view.stream_window("env", "a", "ep_1", 7, 10, 3, 3)       # anchor e10 = resumed step 7
    assert [w[1] for w in win] == ["e4", "e5", "e9", "e10", "e11"] and win[i][1] == "e10"   # widened, one branch
    raw, j = arc.stream_window("env", "a", "ep_1", 7, 10, 3, 3)
    assert {"e6", "e7", "e8"} & {w[1] for w in raw}                    # the raw archive interleaves both branches
    assert view.count() == arc.count() == 12 and view.get(["e7"])[0].event_id == "e7"   # writers / exact reads
    sampler = ReplaySampler(view, seed=0)
    sampler.sync()
    assert sampler.stratum_sizes()["domain"] == 9                       # membership strata: one branch only
    # E's "uniform" / "recent" strata are gseq ranges (they can still name an abandoned anchor); the agent drops
    # abandoned records from every drawn slice, so training sequences hold one branch with unique steps
    st = Stack()
    st.agent.replay, st.agent.exclude_events = sampler, view.exclude
    seqs = st.agent._sample(32, 4)
    assert seqs
    for recs in seqs:
        ids = {r.event_id for r in recs}
        assert not ids & {"e6", "e7", "e8"}
        steps = [r.step for r in recs]
        assert len(steps) == len(set(steps)) and steps == sorted(steps)
    assert st.agent.provenance_stats["replay_records_excluded"] > 0
    arc.close()


def test_loss_cap_bounds_each_objective_and_is_logged():
    st = Stack(loss_cap=0.05)
    rep = run_steps(st.agent, st.env, 4, evaluator=st.evaluator, update_every=2)
    ups = [u for u in rep.updates if u["updated"]]
    assert ups
    for u in ups:
        for k, raw in u["losses"].items():
            sc = u[f"balance/{k}"]
            assert 0 < sc <= 1.0
            assert u["scaled_losses"][k] == pytest.approx(raw * sc, rel=1e-5, abs=1e-7)
        assert any(u[f"balance/{k}"] < 1.0 for k in u["losses"])
    st2 = Stack()
    rep2 = run_steps(st2.agent, st2.env, 2, evaluator=st2.evaluator, update_every=2)
    assert not any(k.startswith("balance/") for k in rep2.updates[0])     # off by default
