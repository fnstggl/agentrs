"""Learning-progress curriculum: prefers learnable arms, keeps a diversity floor, never touches held-out families."""
import math
import random

import pytest

from superlearner.cam_unified.learn.curriculum import HeldOutFamilyError, LearningProgressCurriculum, SplitGuard

ARMS = [("mastered", 0), ("learnable", 1), ("learnable", 2), ("impossible", 3)]


class _SimLearner:
    """Competence per arm as a function of the experience spent on it."""

    def __init__(self, seed=0):
        self.n = {}
        self.rng = random.Random(seed)

    def score(self, arm):
        fam, diff = arm
        k = self.n.get(arm, 0)
        self.n[arm] = k + 1
        noise = self.rng.gauss(0, 0.03)
        if fam == "mastered":
            return 0.97 + noise
        if fam == "impossible":
            return 0.0 + abs(noise)
        tau = 25.0 if diff == 1 else 60.0
        return 1.0 - math.exp(-k / tau) + noise


def test_curriculum_prefers_learnable_arms_and_keeps_the_diversity_floor():
    guard = SplitGuard(held_out={"test_family"})
    cur = LearningProgressCurriculum(ARMS, guard, family_floor=0.05, seed=0)
    sim = _SimLearner()
    counts = {a: 0 for a in ARMS}
    for _ in range(400):
        arm = cur.sample(1)[0]
        counts[arm] += 1
        cur.update(arm[0], arm[1], sim.score(arm))
    learn = counts[("learnable", 1)] + counts[("learnable", 2)]
    assert learn > 0.6 * 400, counts
    assert counts[("mastered", 0)] < 0.2 * 400 and counts[("impossible", 3)] < 0.2 * 400
    w = cur.weights()
    fams = {}
    for a, p in zip(cur.arms, w):
        fams[a.family] = fams.get(a.family, 0.0) + p
    assert all(p >= 0.05 for p in fams.values()) and sum(w) == pytest.approx(1.0)
    # once the learnable arms are mastered too, the mass spreads back out (no collapse on one family)
    for _ in range(1500):
        arm = cur.sample(1)[0]
        cur.update(arm[0], arm[1], sim.score(arm))
    tab = {(r["family"], r["difficulty"]): r for r in cur.table()}
    assert tab[("learnable", 1.0)]["competence"] > 0.9
    assert max(r["p"] for r in tab.values()) < 0.6


def test_held_out_families_are_refused_everywhere():
    guard = SplitGuard(held_out={"val_fam", "test_fam"}, allowed={"a", "b", "val_fam"})
    with pytest.raises(HeldOutFamilyError):
        LearningProgressCurriculum([("a", 0), ("test_fam", 0)], guard)
    cur = LearningProgressCurriculum([("a", 0), ("b", 0)], guard)
    with pytest.raises(HeldOutFamilyError):
        cur.add_arm("val_fam", 1)
    with pytest.raises(HeldOutFamilyError):
        cur.update("val_fam", 1, 0.5)
    with pytest.raises(HeldOutFamilyError):
        cur.add_arm("undeclared", 0)                      # not a declared training family
    st = cur.state_dict()
    st["arms"].append(("test_fam", 0.0, 3, 0.5, 0.4, 0.5))
    with pytest.raises(HeldOutFamilyError):
        LearningProgressCurriculum([("a", 0), ("b", 0)], guard).load_state_dict(st)
    with pytest.raises(ValueError):
        LearningProgressCurriculum([("a", 0)], None)


def test_curriculum_state_round_trip():
    guard = SplitGuard(held_out=set())
    cur = LearningProgressCurriculum(ARMS, guard, seed=3)
    for i in range(50):
        a = cur.sample(1)[0]
        cur.update(a[0], a[1], 0.01 * i)
    cur2 = LearningProgressCurriculum(ARMS, guard, seed=99)
    cur2.load_state_dict(cur.state_dict())
    assert cur2.weights() == pytest.approx(cur.weights())
    assert cur2.sample(5) == cur.sample(5)
