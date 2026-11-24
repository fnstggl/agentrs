"""CAM 7.0 integrated model: learned adaptive features, support/query meta-training, the learner contract."""
import numpy as np
import pytest
import torch

from superlearner.cam7.config import small
from superlearner.cam7.model import Cam7Model, History

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _interaction_world(B, T=64, L=12, n_active=4, seed=0):
    """Revenue depends on the product of two levers (non-additive); cost is linear. Each task draws its own sign and
    size of the interaction, so it must be inferred from the episode."""
    g = np.random.default_rng(seed)
    x = np.zeros((B, T, L), np.float32)
    x[:, :, :n_active] = g.uniform(-0.6, 0.6, (B, T, n_active))
    beta = g.choice([-1, 1], B) * g.uniform(0.5, 1.5, B)
    rates = g.uniform(0.0, 0.2, (B, n_active))
    rev = 1.0 + beta[:, None] * x[:, :, 0] * x[:, :, 1] + 0.02 * g.standard_normal((B, T))
    cost = 0.5 + (x[:, :, :n_active] * rates[:, None]).sum(-1) + 0.005 * g.standard_normal((B, T))
    lm = np.zeros((B, L), np.float32)
    lm[:, :n_active] = 1
    ids = g.standard_normal((B, L, 16)).astype(np.float32)
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)
    return History(x=t(x), ema=t(x), out=t(np.stack([rev, cost], -1)), gauges=t(np.zeros((B, T, 16))),
                   tau=t(np.broadcast_to(np.arange(T) / 96, (B, T))), funds=t(np.ones((B, T))), pred=t(np.zeros((B, T, 2))),
                   x0=t(np.zeros((B, L))), lmask=t(lm), gmask=t(np.zeros((B, 16))), lid=t(ids),
                   gid=t(np.zeros((B, 16, 16))))


def _train_eval(cfg, steps=300, seed=0):
    torch.manual_seed(seed)
    m = Cam7Model(cfg).to(DEV)
    opt = torch.optim.Adam(m.param_groups(1e-3))
    for i in range(steps):
        H = _interaction_world(32, seed=1000 + i)
        k = int(np.random.default_rng(i).choice([4, 8, 16, 32]))
        cut = torch.full((32,), k, device=DEV)
        q = torch.zeros(32, 64, dtype=torch.bool, device=DEV)
        q[:, k:] = True
        loss, _, _ = m.meta_loss(H, cut, q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        H = _interaction_world(256, seed=7)
        cut = torch.full((256,), 16, device=DEV)
        q = torch.zeros(256, 64, dtype=torch.bool, device=DEV)
        q[:, 16:] = True
        _, info, _ = m.meta_loss(H, cut, q)
    return info


@pytest.mark.slow
def test_learned_features_capture_an_interaction_fixed_regression_cannot():
    base = small().with_(n_layers=1, d_model=64, n_heads=4)
    fixed = _train_eval(base.with_(use_learned_features=False))
    learned = _train_eval(base)
    # identical support (16 rows) and queries; learned features must explain most of the interaction variance
    assert learned["meta_r2"] > 0.5 and learned["meta_r2"] > fixed["meta_r2"] + 0.3, (fixed, learned)
    assert learned["meta_nll"] < fixed["meta_nll"] - 0.3, (fixed, learned)


def test_learner_contract_freeze_and_checkpoint(tmp_path):
    from crucible.research.envs import MechVecEnv, run_episodes
    from crucible.research.learner import EnvSpec
    from crucible.research.tasks import make_task
    from superlearner.cam7.agent import Cam7Learner
    spec = EnvSpec()
    tasks = [make_task("scm_additive", 900 + i) for i in range(4)]
    L = Cam7Learner(spec, 0, DEV, scale="small", min_episodes=2, n_layers=1)
    run_episodes(MechVecEnv(tasks, spec), L, explore=True, learn=True)
    assert len(L.replay) == 4 and L.replay[0]["decisions"], "episodes and decisions are recorded"
    u = L.update()
    assert u["updates"] > 0
    w0 = [p.detach().clone() for p in L.model.parameters()]
    L.freeze(True)
    run_episodes(MechVecEnv(tasks, spec), L, explore=False, learn=True)
    assert L.update()["updates"] == 0 and len(L.replay) == 4, "frozen: no weight change, no replay writes"
    assert all(torch.equal(a, b) for a, b in zip(w0, L.model.parameters()))
    # frozen evaluation is reproducible given the same context seed
    r1 = run_episodes(MechVecEnv(tasks, spec), L, explore=False)
    r2 = run_episodes(MechVecEnv(tasks, spec), L, explore=False)
    assert np.allclose([r["net_cents"] for r in r1], [r["net_cents"] for r in r2], rtol=0.05, atol=1e-6)
    path = str(tmp_path / "c.pt")
    L.save(path)
    L2 = Cam7Learner(spec, 5, DEV, scale="small", n_layers=1)
    L2.load(path)
    assert all(torch.equal(a, b) for a, b in zip(L.model.parameters(), L2.model.parameters()))
    assert len(L2.replay) == 4 and L2.version == L.version


def test_slot_permutation_does_not_change_the_posterior():
    """Nothing is tied to a lever slot: permuting slots (with their identities) permutes nothing in the predictions."""
    torch.manual_seed(0)
    m = Cam7Model(small().with_(n_layers=1, d_model=64, n_heads=4)).to(DEV).eval()
    H = _interaction_world(4, seed=3)
    perm = torch.tensor([3, 1, 0, 2] + list(range(4, 12)), device=DEV)
    Hp = History(x=H.x[..., perm], ema=H.ema[..., perm], out=H.out, gauges=H.gauges, tau=H.tau, funds=H.funds,
                 pred=H.pred, x0=H.x0[:, perm], lmask=H.lmask[:, perm], gmask=H.gmask, lid=H.lid[:, perm], gid=H.gid)
    cut = torch.full((4,), 20, device=DEV)
    q = torch.zeros(4, 64, dtype=torch.bool, device=DEV)
    q[:, 20:] = True
    with torch.no_grad():
        a = m.meta_loss(H, cut, q)[1]["meta_nll"]
        b = m.meta_loss(Hp, cut, q)[1]["meta_nll"]
    assert abs(a - b) < 1e-4
