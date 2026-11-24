"""`meta_np`: the properties its results depend on. The learner-contract tests in `test_learners.py` also cover it."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crucible.research import learners  # noqa: E402
from crucible.research.learner import EnvSpec  # noqa: E402
from crucible.research.learners.meta_np import _Slot, from_z, to_z  # noqa: E402

torch.set_num_threads(2)
SPEC = EnvSpec()
SMALL = dict(d=24, heads=2, enc_blocks=1, dec_blocks=1)


def make(**kw):
    return learners.make("meta_np", SPEC, seed=0, **dict(SMALL, **kw))


def table(seed=0, b=2, c=7, q=5, l=4):
    rng = np.random.default_rng(seed)
    t = lambda *shape: torch.as_tensor(rng.uniform(-1, 1, shape), dtype=torch.float32)
    return dict(xc=t(b, c, l), yc=t(b, c, 3), xq=t(b, q, l), x0=t(b, l), lm=torch.ones(b, l, dtype=torch.bool),
                cm=torch.ones(b, c, dtype=torch.bool), pc=t(b, c, l), pq=t(b, q, l))


def predict(net, x, a=1.0, b=1.0):
    net.eval()
    with torch.no_grad():
        return net(x["xc"], x["yc"], x["xq"], x["x0"], x["lm"], x["cm"], x["pc"], x["pq"], a, b)


def perturbed(lr):
    """Give the correction head non-zero weights, so symmetry is not trivially true of a zero output."""
    with torch.no_grad():
        for q in lr.net.head[-1].parameters():
            q.add_(0.1 * torch.randn_like(q))
    return lr


def test_scaled_log_round_trips():
    y = np.array([-1000.0, -1.5, -0.01, 0.0, 0.01, 2.0, 500.0])
    assert np.allclose(from_z(to_z(y, 0.01), 0.01), y)


def test_an_untrained_learner_predicts_exactly_the_regression():
    lr = make()
    x = table()
    m1, s1 = predict(lr.net, x, 1.0, 1.0)
    m0, s0 = predict(lr.net, x, 0.0, 0.0)
    assert torch.equal(m1, m0) and torch.equal(s1, s0)


def test_the_regression_settings_are_not_trained():
    lr = make()
    assert not lr.net.log_noise.requires_grad and not lr.net.log_tau.requires_grad


def test_the_regression_recovers_an_additive_response():
    lr = make(ridge_noise=0.02, ridge_tau=3.0)
    rng = np.random.default_rng(1)
    w = np.array([0.8, -0.5, 0.2])
    f = lambda v: 1.0 + (v * w).sum(-1) - 0.4 * (v[..., 0] ** 2)
    xc, xq = rng.uniform(-1, 1, (1, 40, 3)), rng.uniform(-1, 1, (1, 20, 3))
    t = lambda v: torch.as_tensor(v, dtype=torch.float32)
    y = np.stack([f(xc), 0 * f(xc), f(xc)], -1)
    mean, _ = lr.net.ridge(t(xc), t(y), t(xq), torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool),
                           torch.ones(1, 40, dtype=torch.bool))
    assert np.abs(mean[0, :, 0].numpy() - f(xq)[0]).max() < 0.05


def test_predictions_do_not_depend_on_the_order_of_levers():
    lr = perturbed(make())
    x = table()
    m, s = predict(lr.net, x)
    perm = torch.tensor([2, 0, 3, 1])
    y = dict(x)
    for k in ("xc", "xq", "pc", "pq"):
        y[k] = x[k][:, :, perm]
    y["x0"] = x["x0"][:, perm]
    m2, s2 = predict(lr.net, y)
    assert torch.allclose(m, m2, atol=1e-4) and torch.allclose(s, s2, atol=1e-4)


def test_predictions_do_not_depend_on_the_order_of_measured_arms():
    lr = perturbed(make())
    x = table()
    m, _ = predict(lr.net, x)
    perm = torch.randperm(x["xc"].shape[1], generator=torch.Generator().manual_seed(3))
    y = dict(x, xc=x["xc"][:, perm], yc=x["yc"][:, perm], pc=x["pc"][:, perm])
    m2, _ = predict(lr.net, y)
    assert torch.allclose(m, m2, atol=1e-4)


def test_padded_levers_and_arms_change_nothing():
    lr = perturbed(make())
    x = table(b=1)
    m, _ = predict(lr.net, x)
    pad_l = lambda v: torch.cat([v, 0.7 * torch.ones(*v.shape[:-1], 2)], dim=-1)
    y = {k: pad_l(x[k]) for k in ("xc", "xq", "pc", "pq", "x0")}
    y["lm"] = torch.cat([x["lm"], torch.zeros(1, 2, dtype=torch.bool)], dim=1)
    extra = torch.zeros(1, 3, y["xc"].shape[2])
    y["xc"], y["pc"] = torch.cat([y["xc"], extra + 0.3], 1), torch.cat([y["pc"], extra - 0.3], 1)
    y["yc"] = torch.cat([x["yc"], torch.ones(1, 3, 3)], 1)
    y["cm"] = torch.cat([x["cm"], torch.zeros(1, 3, dtype=torch.bool)], 1)
    m2, _ = predict(lr.net, y)
    assert torch.allclose(m, m2, atol=1e-3)


def _held_out_error(lr, held, a):
    """Mean squared error (scaled log revenue) on the last 4 arms of each episode, given its first 12."""
    lr.net.eval()
    out = []
    for e in held:
        t = lambda v: torch.as_tensor(v, dtype=torch.float32)[None]
        r, c = to_z(e["r"], 0.01), to_z(e["c"], 0.01)
        before = np.vstack([e["x0"][None], e["x"][:-1]])
        yc = np.stack([r[:12], c[:12], to_z(e["r"][:12] - e["c"][:12], 0.01)], 1)
        with torch.no_grad():
            mu, _ = lr.net(t(e["x"][:12]), t(yc), t(e["x"][12:]), t(e["x0"]),
                           torch.ones(1, e["x"].shape[1], dtype=torch.bool), torch.ones(1, 12, dtype=torch.bool),
                           t(before[:12]), t(before[12:]), a, 0.0)
        out.append(float(((mu[0, :, 0].numpy() - r[12:]) ** 2).mean()))
    lr.net.train()
    return float(np.mean(out))


def _episode(x, r):
    return {"x": x.astype(np.float32), "x0": np.zeros(x.shape[1], np.float32), "r": r.astype(np.float32),
            "c": np.full(len(x), 0.2, np.float32)}


@pytest.mark.slow
def test_training_learns_structure_shared_across_worlds_that_the_regression_cannot_represent():
    """About 100 s on two CPU threads. Learning starts after a plateau of a few thousand updates, so a short run shows
    nothing. `examples/ow_meta_np_synthetic.py` records this and two harder structures at the full network size."""
    lr = make(min_episodes=8, batch=16, query=8, lr=1e-3, ridge_noise=0.3, updates_per_episode=0.0)
    rng = np.random.default_rng(5)

    def episode():
        l = int(rng.integers(2, 5))
        x = rng.uniform(-1, 1, (16, l))
        return _episode(x, 1.0 + 0.5 * np.cos(4.0 * x).mean(1))

    lr._buffer = [episode() for _ in range(1000)]
    held = [episode() for _ in range(40)]
    regression = _held_out_error(lr, held, 0.0)
    for _ in range(6000):
        lr._step()
    assert _held_out_error(lr, held, 1.0) < 0.2 * regression


def slot(lr, **kw):
    s = _Slot(lr, np.random.default_rng(0))
    s.levers, s.x0, s.h, s.t = np.arange(2), np.zeros(2), 100, 10
    s.X, s.R, s.C, s.N = [np.zeros(2)], [1.0], [0.5], [1]
    s.base_net = 0.5
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_exploration_stops_once_it_has_lost_its_allowance():
    lr = make(loss_floor=0.01, loss_share=0.5)
    assert slot(lr, spent=0.5).may_explore()
    assert not slot(lr, spent=1.5).may_explore()


def test_a_demonstrated_gain_buys_more_exploration():
    lr = make(loss_floor=0.01, loss_share=0.5)
    s = slot(lr, spent=5.0)
    assert not s.may_explore()
    s.X.append(np.ones(2)), s.R.append(1.5), s.C.append(0.5), s.N.append(1)          # net 1.0 against a base of 0.5
    assert s.may_explore()


def test_no_exploration_after_the_commit_point():
    lr = make(explore_until=0.5)
    assert not slot(lr, t=60).may_explore()


def test_calibration_tracks_how_wrong_the_predictions_were():
    lr = make()
    assert slot(lr).calibration() == pytest.approx(1.0)
    s = slot(lr)
    for _ in range(9):
        s.pending = (1.0, 0.1)
        s._score_arm(1.3)                                # three spreads off, every time
    assert s.calibration() == pytest.approx(np.sqrt((1 + 9 * 9.0) / 10.0), rel=1e-6)


def test_coordinate_candidates_move_one_lever_within_the_trust_region():
    lr = make()
    s = slot(lr, radius=0.2)
    x = np.array([0.1, -0.3])
    g = s.coord_grid(x)
    assert (np.abs(g - x) > 1e-9).sum(1).max() == 1
    assert np.abs(g - x).max() <= 0.2 + 1e-9
