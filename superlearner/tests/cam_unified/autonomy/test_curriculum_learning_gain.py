"""C_chi trained on MEASURED learning gain (AU-03, KV-02): a real learner is updated on the experience each chosen
self-goal produces and evaluated on independent query tasks against a matched-compute control; the curriculum and the
gain predictor learn only from those measurements (no completion or difficulty reward)."""
import math

import torch

from superlearner.cam_unified.autonomy.curriculum import (CurriculumPolicy, CurriculumTrainer, LearningGainPredictor,
                                                          measure_learning_gain)

N_Q, SCALES = 6, (0.5, 1.0, 2.0, 4.0)
QUERY_Q = (3, 4)                          # what the (undisclosed) query tasks ask about
SAMPLES = {0.5: 1, 1.0: 3, 2.0: 8, 4.0: 2}  # easy goals finish at once; impossible ones are abandoned early


class MeanLearner:
    """A real (tiny) learner: running estimates of hidden quantities, SGD on squared error."""

    def __init__(self, lr=0.25):
        self.m = torch.zeros(N_Q, dtype=torch.float64)
        self.lr = lr


def update(learner, data):
    for q, x in data:
        learner.m[q] += learner.lr * (x - learner.m[q])


def evaluate(learner, task):
    mu, q = task
    return -float((learner.m[q] - mu[q]) ** 2)


def candidates():
    out = []
    for q in range(N_Q):
        for k in SCALES:
            for op in (1.0, -1.0):
                out.append((q, k, op))
    return out


def goal_feats(cands):
    return torch.tensor([[q / (N_Q - 1), 0.0, k, 0.0, op] for q, k, op in cands], dtype=torch.float32)


def pursue(cand, mu, g):
    q, k, _ = cand
    return [(q, float(mu[q] + torch.randn((), generator=g, dtype=torch.float64))) for _ in range(SAMPLES[k])]


def _measure(cand, g, cands):
    mu = torch.randn(N_Q, generator=g, dtype=torch.float64)
    data = pursue(cand, mu, g)
    rnd = cands[int(torch.randint(len(cands), (1,), generator=g))]
    ctrl = [(rnd[0], float(mu[rnd[0]] + torch.randn((), generator=g, dtype=torch.float64))) for _ in range(len(data))]
    tasks = [(mu, q) for q in QUERY_Q for _ in range(2)]
    return measure_learning_gain(MeanLearner(), data, tasks, update_fn=update, evaluate_fn=evaluate,
                                 control_data=ctrl).gain


def _expected_gain(cand, g, n=200):
    cands = candidates()
    return sum(_measure(cand, g, cands) for _ in range(n)) / n


def test_curriculum_learns_from_measured_gain_and_the_predictor_is_calibrated_on_held_out_measurements():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    cands = candidates()
    F = goal_feats(cands)
    Cn = len(cands)
    D = 8
    pol = CurriculumPolicy(D, 64, d_goal_emb=N_Q)              # goal embedding: which quantity the goal is about
    pred = LearningGainPredictor(D, 64, n_goal_feats=5 + N_Q)
    tr = CurriculumTrainer(pol, pred, lr=3e-3, pred_lr=3e-3, entropy_coef=3e-3, holdout_every=5, pred_steps=2)
    B = 16
    b = torch.zeros(B, D)
    life = torch.zeros(B, 6)
    gf = F[None].expand(B, Cn, 5)
    ge = torch.nn.functional.one_hot(torch.tensor([q for q, _, _ in cands]), N_Q).float()[None].expand(B, Cn, N_Q)
    FE = torch.cat([F, ge[0]], -1)                                    # predictor descriptor
    mask = torch.ones(B, Cn, dtype=torch.bool)
    good = torch.tensor([q in QUERY_Q for q, k, _ in cands])            # goals about what the queries need
    p0 = float(pol.choose(b, life, gf, mask, ge).probs.mean(0)[good].sum())
    gains_first, gains_last = [], []
    for it in range(250):
        ch = tr.choose(b, life, gf, mask, ge, generator=g)
        # each chosen goal is pursued in a fresh world; its experience updates a learner that is then evaluated on
        # independent query tasks against a control updated on as many examples from a random goal (matched compute)
        meas = [_measure(cands[int(ch.index[i])], g, cands) for i in range(B)]
        gain = torch.tensor(meas, dtype=torch.float32)
        outcome = torch.zeros(B, 4)
        tr.record(b, FE[ch.index], outcome, gain)
        tr.policy_step(b, life, gf, mask, ch.index, tr.signal(b, FE[ch.index], outcome, gain), goal_emb=ge)
        (gains_first if it < 25 else gains_last if it >= 225 else []).append(float(gain.mean()))
    probs = pol.choose(b, life, gf, mask, ge).probs.mean(0)
    p1 = float(probs[good].sum())
    assert p0 < 0.4 and p1 > 0.7, (p0, p1)
    assert sum(gains_last) / len(gains_last) > sum(gains_first) / len(gains_first) + 0.2
    cal = tr.calibration()
    # single measurements are heavy-tailed (differences of squared errors; the best possible correlation with one
    # measurement is ~0.17 here), so the predictor's MEAN is judged: held-out RMSE below the constant predictor's and,
    # per candidate, correlation with Monte-Carlo expected gains. Its Gaussian NLL is over-confident on these tails
    # (reported by calibration(), not asserted).
    assert cal["n"] >= 500 and cal["rmse"] < cal["rmse_constant"], cal
    true = torch.tensor([_expected_gain(c, g) for c in cands])
    with torch.no_grad():
        mu, _ = pred(torch.zeros(Cn, D), FE, torch.zeros(Cn, 4))
    r = float(torch.corrcoef(torch.stack([mu, true.float()]))[0, 1])
    assert r > 0.7, r


def test_measurement_never_modifies_the_learner_and_matches_a_hand_computation():
    L = MeanLearner(lr=0.5)
    mu = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    m = measure_learning_gain(L, [(0, 1.0), (0, 1.0)], [(mu, 0), (mu, 1)], update_fn=update, evaluate_fn=evaluate,
                              control_data=[(1, 2.0), (1, 2.0)])
    assert torch.equal(L.m, torch.zeros(N_Q, dtype=torch.float64))
    # after D_g: m0 = 0.75, m1 = 0; control: m0 = 0, m1 = 1.5
    ja = [-(0.75 - 1) ** 2, -(0 - 2) ** 2]
    jc = [-(0 - 1) ** 2, -(1.5 - 2) ** 2]
    assert m.per_task == [a - c for a, c in zip(ja, jc)]
    assert math.isclose(m.gain, sum(m.per_task) / 2)
