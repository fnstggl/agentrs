"""G2D-1 (PI-11): the change detector re-arms once predictions are calibrated again."""
import numpy as np

from cam_live.contracts import OutcomeStatus
from cam_live.predict.errors import ChangeDetector, classify_error


def test_a_detected_shift_does_not_latch_forever():
    det = ChangeDetector()
    for _ in range(40):
        det.update(3.0)
    assert det.changed
    for r in np.random.default_rng(0).normal(size=2000):
        det.update(float(r))
    assert not det.changed
    att = classify_error(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 0.1, det)
    assert att.label == "model_wrong"


def test_a_persisting_shift_stays_detected():
    det = ChangeDetector()
    for _ in range(200):
        det.update(3.0)
        assert det.changed or det.n < 5
    assert det.changed
