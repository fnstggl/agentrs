"""RewardModel: the default observation-only actor reward of plan §2 (W8)."""
import inspect
import math

import pytest
import torch

from superlearner.cam_unified.config import make_config, make_preset
from superlearner.cam_unified.envs.labels import LabelRecord
from superlearner.cam_unified.interfaces import Action, EventRecord, RewardModelP, TokenKind, TokenSet
from superlearner.cam_unified.selfeval import label_join as LJ
from superlearner.cam_unified.selfeval.evaluator import calibration_report
from superlearner.cam_unified.selfeval.reward_model import RewardModel

torch.set_num_threads(4)
D = 32


def _rm(**kw):
    torch.manual_seed(0)
    cfg = make_config("dev", reward_mode="observation_only", d_model=D, n_heads=4)
    return RewardModel(cfg, d_hidden=32, **kw)


def _inputs(B=6, S=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(B, S, D, generator=g)
    Rn = torch.randn(B, S, D, generator=g)
    G = TokenSet.from_dense(torch.randn(B, 2, D, generator=g), int(TokenKind.GOAL))
    dt = torch.rand(B, generator=g)
    ig = torch.rand(B, generator=g)
    cost = 500 * torch.rand(B, generator=g)
    return R, Rn, G, dt, ig, cost


def test_legacy_mode_is_the_inert_null_model_and_dom_presets_build_the_full_one():
    assert RewardModel(make_config("dev")).stub and RewardModel(None).stub
    full = RewardModel(make_preset("dom_dev"))
    assert not full.stub and isinstance(full, RewardModelP)
    s = full.state_summary()
    assert s["lambda_ig"] == pytest.approx(0.05) and s["lambda_c"] == pytest.approx(0.01) and s["n_psi"] > 0


def test_reward_at_initialisation_is_constant_potential_shaping_plus_ig_minus_cost():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    r = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    phi = rm.potential(R, G)
    assert torch.allclose(phi, torch.full_like(phi, 0.5))                       # zero-initialised heads
    expect = torch.exp(-rm.kappa * dt) * 0.5 - 0.5 + 0.05 * ig - 0.01 * torch.log1p(cost / 100.0)
    assert torch.allclose(r, expect, atol=1e-6)
    t = rm.reward_terms(R, Rn, G, dt, info_gain=ig, cost=cost)
    assert float(t["rho"].abs().max()) == 0.0                                    # zero-initialised output layer


def test_reward_is_detached_so_actor_losses_never_reach_psi_or_phi():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    R.requires_grad_(True)
    r = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    assert not r.requires_grad
    logits = torch.zeros(6, 3, requires_grad=True)
    logp = torch.log_softmax(logits, -1)[:, 0]
    L_actor = -(logp * (r - r.mean())).sum()
    params = rm.meta_parameters() + rm.supervised_parameters()
    gs = torch.autograd.grad(L_actor, params + [logits], allow_unused=True)
    assert all(g is None for g in gs[:-1])                                       # d L_actor / d (psi, Phi) == 0
    assert gs[-1] is not None


def test_reward_psi_carries_the_graph_to_psi_only():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    rp = rm.reward_psi(R, Rn, G, dt, info_gain=ig, cost=cost)
    sup = rm.supervised_parameters()
    psi = rm.meta_parameters()
    gs = torch.autograd.grad(rp.sum(), psi + sup, allow_unused=True)
    g_psi, g_sup = gs[:len(psi)], gs[len(psi):]
    assert all(g is None for g in g_sup)                                         # Phi is stop-gradient inside r
    assert abs(float(g_psi[-2])) > 0 and abs(float(g_psi[-1])) > 0              # log lambda_ig, log lambda_c
    fc2 = rm.intrinsic.mlp.fc2.weight
    (gw,) = torch.autograd.grad(rm.reward_psi(R, Rn, G, dt, info_gain=ig, cost=cost).sum(), [fc2])
    assert float(gw.abs().sum()) > 0                                             # rho's output layer gets gradient


def test_reward_is_recomputed_under_the_current_psi_after_a_bump():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    r0 = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    v0 = rm.version
    opt = torch.optim.Adam(rm.meta_parameters(), lr=1e-2)
    (-rm.reward_psi(R, Rn, G, dt, info_gain=ig, cost=cost).mean()).backward()
    opt.step()
    rm.bump_version()
    r1 = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    assert rm.version == v0 + 1 and rm.intrinsic.version == v0 + 1
    assert float((r1 - r0).abs().max()) > 1e-4


def test_only_actor_visible_signals_are_accepted_and_no_label_argument_exists():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    for bad in ("utility", "reward", "label", "verified", "status"):
        with pytest.raises(ValueError):
            rm.reward(R, Rn, G, dt, signals={bad: torch.ones(6)})
    ok = rm.reward(R, Rn, G, dt, signals={"pred_error": torch.ones(6), "novelty": torch.zeros(6)})
    assert ok.shape == (6,)
    params = set(inspect.signature(rm.reward).parameters)
    assert not params & {"label", "labels", "utility", "verified", "label_batch"}
    a = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    b = rm.reward(R, Rn, G, dt, info_gain=ig, cost=cost)
    assert torch.equal(a, b)                                                     # a function of actor inputs only


def _label_batch(n_ep=80, T=4, seed=0):
    """Synthetic resolved episodes whose success is predictable from a feature direction of R (a stand-in for
    CAM's latent; unit test only)."""
    g = torch.Generator().manual_seed(seed)
    recs, labs, feats = [], [], {}
    u = torch.randn(D, generator=g)
    for e in range(n_ep):
        p = torch.rand((), generator=g)
        y = bool(torch.rand((), generator=g) < p)
        for s in range(T):
            eid = f"e{e}/{s}"
            recs.append(EventRecord(event_id=eid, env_id=f"w#{e}", actor_id="a", episode_id=f"ep{e}", step=s,
                                    model_version=0, t_sim=float(s), observations=[], action=Action("x")))
            x = torch.randn(4, D, generator=g) * 0.3
            x[:, :] += (torch.logit(p.clamp(0.02, 0.98)) * u / u.norm() ** 2 * 3.0)
            feats[eid] = x
        labs.append(LabelRecord(f"w#{e}", 0, "success" if y else "failure", float(y), float(y), float(T), "op"))
    lb = LJ.join(labs, recs)
    return lb.with_features(torch.stack([feats[e] for e in lb.event_ids]))


def test_losses_train_phi_and_lifetime_value_on_joined_labels_only():
    rm = _rm()
    lb = _label_batch(n_ep=300)
    tr, ho = LJ.split_by_episode(lb, 0.25, seed=0)
    L0 = rm.losses(tr)
    assert set(L0) >= {"prospect", "prospect/success", "lifetime_value"}
    gs = torch.autograd.grad(L0["prospect"] + L0["lifetime_value"], rm.meta_parameters(), allow_unused=True)
    assert all(g is None for g in gs)                                            # labels never train psi directly
    opt = torch.optim.Adam(rm.supervised_parameters(), lr=2e-3)
    for _ in range(60):
        L = rm.losses(tr)
        opt.zero_grad()
        (L["prospect"] + L["lifetime_value"]).backward()
        opt.step()
    L1 = rm.losses(tr)
    assert float(L1["prospect/success"]) < float(L0["prospect/success"]) - 0.05
    assert float(L1["lifetime_value"]) < float(L0["lifetime_value"])
    p = rm.predict(ho.R)
    y = (ho.status == 1).float()
    rep = calibration_report(p, y)
    assert rep["nll"] < rep["nll_prevalence"] + 0.05


def test_potential_target_freezes_phi_between_refreshes():
    rm = _rm()
    R, Rn, G, dt, ig, cost = _inputs()
    rm.refresh_potential_target()
    r0 = rm.reward(R, Rn, G, dt)
    with torch.no_grad():
        rm.evaluator.success.bias.fill_(2.0)                                     # the live evaluator moves
    assert torch.equal(rm.reward(R, Rn, G, dt), r0)
    rm.refresh_potential_target()
    assert not torch.equal(rm.reward(R, Rn, G, dt), r0)
    rm.clear_potential_target()
    assert float(rm.potential(R, G)[0]) == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-5)
