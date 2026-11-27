"""C-12: the verifier signs the origin of the task it verified; changing it breaks the signature (INT2)."""
from __future__ import annotations

from cam_live.contracts import Quantity, SimClock, TaskSpec, Trust
from cam_live.runtime.verifier import MetricThresholdVerifier


class _Src:
    name, live = "test", False

    def read(self):
        return [Quantity("m.count", 2.0, kind="count")]


def test_verifier_signs_the_task_origin(tmp_path):
    task = TaskSpec(objective_id="obj_o", objective="Reach 3.", success={"verifier": "metric_threshold",
                                                                         "metric": "m.count", "target": 3},
                    mode="staging", origin=Trust.OPERATOR)
    v = MetricThresholdVerifier(_Src(), SimClock(), str(tmp_path / "v"), pinned_task=task)
    out = v.verify(task)
    assert out.origin == Trust.OPERATOR and v.check_signature(out)
    out.origin = Trust.SELF
    assert not v.check_signature(out)                     # a relabelled origin does not verify
    self_task = TaskSpec(objective_id="obj_s", objective="self goal", success={"verifier": "metric_threshold",
                                                                               "metric": "m.count", "target": 3},
                         mode="staging", origin=Trust.SELF)
    v2 = MetricThresholdVerifier(_Src(), SimClock(), str(tmp_path / "v2"), pinned_task=self_task)
    o2 = v2.verify(self_task)
    assert o2.origin == Trust.SELF and v2.check_signature(o2)
