"""TraceRecorder: decision traces linked to event ids, later outcomes and update refs, multi-writer directories,
Parquet export, and trace-grounded retrospective summaries (spec 18.1; manifest A20)."""
from __future__ import annotations

import json

import pytest

from superlearner.cam_unified.dist.queue import decode_segment, encode_segment
from superlearner.cam_unified.interfaces import DecisionTrace, new_id
from superlearner.cam_unified.observe.trace import (INTERPRETATION_LABEL, TraceRecorder, load_traces,
                                                    prediction_errors, retrospective_summary)


def _tr(i, env="w0", version=0, tool="lever_1", p=0.6, reward=0.0):
    return DecisionTrace(decision_id=f"d{i:04d}", event_id=f"e{i:04d}", env_id=env, policy_version=version,
                         t_sim=i * 0.1, objective="earn payouts", context_refs=[f"e{i - 1:04d}"] if i else [],
                         retrieved=[{"id": "m1", "score": 0.3}],
                         candidates=[{"action": {"tool": "lever_0"}, "score": 1 - p},
                                     {"action": {"tool": tool}, "score": p}],
                         predicted={"payoff": p}, uncertainty={"payoff": 0.2},
                         imagined=[{"candidate": 1, "horizon": 1, "value": p, "outcomes": {"payoff": p}}],
                         selected={"tool": tool, "args": {}}, reasoning={"iters": 2},
                         belief_before={"ess": 8.0}, outcome={"ok": True, "reward": reward})


def test_record_fill_outcomes_update_refs_and_fields(tmp_path):
    r = TraceRecorder(str(tmp_path / "traces"), name="learner")
    for i in range(5):
        r.record(_tr(i))
    r.fill_outcome({"payoff": 1.0, "delayed": [{"event_id": "x1", "value": 1.0}]}, event_id="e0002")
    r.fill_outcome({"delayed": [{"event_id": "x2", "value": 0.0}]}, event_id="e0002")     # list keys append
    r.fill_outcome({"payoff": 0.0}, decision_id="d0003")
    r.add_update_refs(["e0001", "e0002"], "u00000004")
    r.add_update_refs(["e0002"], "u00000005")
    r.add_update_refs(["e0002"], "u00000005")                # idempotent
    r.set_field("belief_after", {"ess": 3.0}, event_id="e0004")
    with pytest.raises(KeyError):
        r.set_field("not_a_field", 1, event_id="e0004")
    with pytest.raises(ValueError):
        r.fill_outcome({"x": 1})
    ts = {t.decision_id: t for t in r.load()}
    assert ts["d0002"].outcome["payoff"] == 1.0 and ts["d0002"].outcome["ok"] is True
    assert [d["event_id"] for d in ts["d0002"].outcome["delayed"]] == ["x1", "x2"]
    assert ts["d0003"].outcome["payoff"] == 0.0
    assert ts["d0001"].update_refs == ["u00000004"] and ts["d0002"].update_refs == ["u00000004", "u00000005"]
    assert ts["d0004"].belief_after == {"ess": 3.0}
    assert ts["d0000"].update_refs == []


def test_multiple_writers_wire_traces_and_parquet(tmp_path):
    root = str(tmp_path / "traces")
    a, b = TraceRecorder(root, name="a", rows_per_file=3), TraceRecorder(root, name="b")
    for i in range(7):
        a.record(_tr(i, env="w0"))
    # a trace arriving through the queue is stored from its wire JSON, without re-serialisation
    t = _tr(100, env="w1")
    _, items = decode_segment(encode_segment("x", 0, 1, [("trace", t)]), lazy_traces=True)
    b.record(items[0][1])
    b.fill_outcome({"payoff": 1.0}, event_id="e0100")
    a.close()
    b.close()
    ts = load_traces(root)
    assert len(ts) == 8 and len([f for f in (tmp_path / "traces").iterdir()]) >= 4
    w1 = [x for x in ts if x.env_id == "w1"][0]
    assert w1.to_dict() == {**t.to_dict(), "outcome": {**t.outcome, "payoff": 1.0}}
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    path = TraceRecorder(root, name="c").to_parquet(str(tmp_path / "t.parquet"))
    tab = pq.read_table(path)
    assert tab.num_rows == 8
    row = tab.to_pylist()[0]
    assert json.loads(row["candidates"])[1]["action"]["tool"] == "lever_1" and row["policy_version"] == 0


def test_retrospective_summary_is_grounded_and_labelled():
    ts = []
    for i in range(40):                       # strategy shift: lever_0 early (bad), lever_2 late (good)
        late = i >= 20
        tr = _tr(i, version=0 if not late else 3, tool="lever_2" if late else "lever_0", p=0.9 if late else 0.5,
                 reward=1.0 if late else 0.0)
        tr.outcome["payoff"] = 1.0 if late else (1.0 if i == 7 else 0.0)
        tr.update_refs = ["u1"] if i < 10 else []
        ts.append(tr)
    s = retrospective_summary(ts, window=10)
    assert s["label"] == INTERPRETATION_LABEL and "not direct access" in s["label"]
    assert s["versions"] == [0, 3] and len(s["phases"]) == 4
    assert s["phases"][0]["top_actions"][0] == ("lever_0", 10) and s["phases"][-1]["top_actions"][0] == ("lever_2", 10)
    assert s["phases"][0]["mean_reward"] == 0.0 and s["phases"][-1]["mean_reward"] == 1.0
    errs = prediction_errors(ts)
    assert len(errs) == 40
    assert s["largest_misses"][0]["decision_id"] != "d0007" and abs(s["largest_misses"][0]["observed"] -
                                                                    s["largest_misses"][0]["predicted"]) == 0.5
    text = " ".join(s["statements"])
    assert "lever_0 x10" in text and "lever_2 x10" in text
    assert "from 0.000 (first phase) to 1.000 (last phase)" in text
    assert "consumed by 1 learner updates" in text
    empty = retrospective_summary([])
    assert empty["n_decisions"] == 0 and empty["statements"]
