"""Smaller checks (PI-11 error attribution, MR-05/MR-10 outer baseline after G-pr D1, MR-08 promotion bound).

1. PI-11: Predictor.observe_outcomes keeps one ChangeDetector per metric and updates it with every resolved
   standardised residual before classify_error (predict/predictor.py:694-697); nothing in predict/ or agent.py
   calls ChangeDetector.reset(). Reproduced with the same two calls: after one sustained shift, 2000 later
   well-calibrated residuals do not clear the state, and a confident miss is labelled changed_conditions, never
   model_wrong again for that metric.
2. MR-05 / MR-10 (G-pr D1 re-check): MetaLearner.outer_baselines with return-to-go chains: the baseline of a row
   does not change when that row's own measured outcome (and therefore the utilities of the earlier rows of its
   chain) changes.
3. MR-08: RewardMixer.update decisions == my one-sided 95 % Student-t non-inferiority rule (scipy quantile).
"""
import random
import sys

import numpy as np

import _g2d as C
from cam_live.contracts import OutcomeStatus
from cam_live.predict import errors as E
from cam_live.reward.bootstrap import RewardMixer
from cam_live.reward.metalearner import MetaLearner

# 1 ----------------------------------------------------------------------------------------------- PI-11
rng = np.random.default_rng(0)
det = E.ChangeDetector()
for _ in range(40):
    det.update(3.0)                                  # a sustained 3-sigma shift: detected
fired = det.changed
for r in rng.normal(size=2000):                      # the model has adapted: residuals well calibrated again
    det.update(float(r))
lab = E.classify_error(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 0.1, det).label
fresh = E.classify_error(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 0.1, E.ChangeDetector()).label
C.info("detector fired / still 'changed' after 2000 calibrated residuals / label of a confident miss / same "
       "miss with a fresh detector", (fired, det.changed, lab, fresh))
C.check("PI-11: after a detected change and 2000 calibrated residuals a confident miss can again be model_wrong",
        lab == E.WRONG, f"label={lab} (the detector is never reset by the Predictor; detected_at={det.detected_at})")


# 2 ----------------------------------------------------------------------------------------------- D1 re-check
class Ep:
    def __init__(self, utility, group, prefix_base):
        self.utility, self.group, self.prefix_base = utility, group, prefix_base


def rtg_utilities(meas, gamma=0.9):
    n = len(meas)
    return [sum(gamma ** (j - a) * meas[j] for j in range(a, n)) for a in range(n)]


ok_all, worst = True, 0.0
random.seed(1)
for trial in range(200):
    n1, n2 = random.randint(2, 8), random.randint(0, 6)
    m1 = [random.gauss(0, 1) for _ in range(n1)]
    m2 = [random.gauss(0, 1) for _ in range(n2)]
    pb1 = [random.gauss(0, 1) for _ in range(n1)]

    def bases(m1_):
        eps = [Ep(rtg_utilities(m1_), [f"rtg:life=1:held=0"] * n1, pb1)]
        if n2:
            eps.append(Ep(rtg_utilities(m2), [f"rtg:life=2:held=0"] * n2, [0.0] * n2))
        rows = [(0, a) for a in range(n1)] + [(1, a) for a in range(n2)]
        return MetaLearner.outer_baselines(eps, rows)
    a = random.randrange(n1)
    b0 = bases(m1)
    m1b = list(m1)
    m1b[a] += 5.0                                   # the row's own outcome changes
    b1 = bases(m1b)
    d = abs(b1[(0, a)] - b0[(0, a)])
    worst = max(worst, d)
    ok_all &= d < 1e-12
C.check("D1: the outer baseline of a row does not depend on its own measured outcome (return-to-go chains, 200 "
        "random lifetimes with and without a second lifetime)", ok_all, f"max |db| = {worst:.2e}")

# 3 ----------------------------------------------------------------------------------------------- promotion
from scipy import stats  # noqa: E402

agree, n_cases = 0, 0
for trial in range(300):
    n = random.randint(2, 40)
    mu = random.gauss(0, 1)
    d = [random.gauss(mu, abs(random.gauss(1, 0.5)) + 0.05) for _ in range(n)]
    scale = abs(random.gauss(2, 1))
    m = RewardMixer(mix=0.75, step=0.25, min_independent=4, rel_margin=0.05)
    dec = m.update(d, scale=scale)
    if n < 4:
        mine = "hold"
    else:
        mean, se = np.mean(d), np.std(d, ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.95, n - 1)
        lo, hi = mean - t * se, mean + t * se
        mine = "lower" if lo >= -0.05 * scale else ("raise" if hi < 0 else "hold")
    n_cases += 1
    # tolerate the 3-decimal t table only at the boundary
    agree += int(dec.action == mine)
C.check("promotion decisions == my one-sided 95 % t non-inferiority rule (300 random buffers)", agree >= n_cases - 2,
        f"{agree}/{n_cases}")
sys.exit(C.finish())
