"""IntrinsicReward R_psi, LifetimeContext and the signal schema (spec 7.2; MR-01, MR-02, MR-03, MR-11)."""
import inspect
import io

import pytest
import torch

from cam_live.contracts import ActionResult, Cost, FailureClass, ResultStatus
from cam_live.nn.api import GROUP_REWARD
from cam_live.reward import meta as M
from cam_live.reward import signals as S
from cam_live.reward.intrinsic import IntrinsicReward, LifetimeContext, RunningNorm

D = 12


def _inputs(B=6, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    return dict(b_t=rn(B, D), e_a=rn(B, D), b_next=rn(B, D), e_g=rn(B, D), signals=rn(B, S.N_SIGNALS))


def _reward(seed=0, **kw):
    torch.manual_seed(seed)
    return IntrinsicReward(D, d_hidden=24, d_life=8, **kw).double()


# ------------------------------------------------------------------------------------------------- signals
def test_signal_schema_is_fixed_and_versioned():
    assert S.SIGNAL_SCHEMA_VERSION == "1.0" and S.N_SIGNALS == 33 == len(S.SIGNAL_NAMES) == len(S.SIGNAL_TRANSFORMS)
    assert S.SIGNAL_NAMES[:15] == (
        "success_before", "success_after", "delta_v", "action_quality", "info_gain", "voi", "cost_money",
        "cost_seconds", "cost_compute_s", "elapsed_s", "prediction_error_std", "epistemic_before",
        "epistemic_after", "verified_progress_delta", "context_novelty")
    assert S.SIGNAL_NAMES[15:21] == tuple("status." + s for s in ("ok", "failed", "denied", "needs_approval",
                                                                   "timeout", "cancelled"))
    assert S.SIGNAL_NAMES[21:] == tuple("failure." + f for f in FailureClass.ALL)
    # changing names, order or transforms must come with a version bump
    assert S.schema_hash() == "74965c537aa08753" + S.schema_hash()[16:] and len(S.schema_hash()) == 64
    assert S.RewardSignals.names() == S.SIGNAL_NAMES


def test_signals_to_tensor_order_transforms_masks_and_one_hots():
    res = ActionResult(action_id="a1", status=ResultStatus.FAILED, failure_class=FailureClass.PAGE_CHANGED,
                       cost=Cost(seconds=9.0, money=3.0, compute_s=1.0))
    s = S.RewardSignals.from_step(result=res, success_before=0.2, success_after=0.5, voi=-3.0,
                                  prediction_error_std=float("nan"))
    x, m = s.to_tensor(torch.float64), s.mask(torch.float64)
    i = S.SIGNAL_NAMES.index
    assert x.shape == m.shape == (1, S.N_SIGNALS)
    assert float(x[0, i("delta_v")]) == pytest.approx(0.3) and float(m[0, i("delta_v")]) == 1.0
    assert float(x[0, i("voi")]) == pytest.approx(-torch.log1p(torch.tensor(3.0)).item())        # symlog
    assert float(x[0, i("cost_seconds")]) == pytest.approx(torch.log1p(torch.tensor(9.0)).item())  # log1p
    assert float(m[0, i("prediction_error_std")]) == 0.0 and float(x[0, i("prediction_error_std")]) == 0.0  # NaN
    assert float(m[0, i("info_gain")]) == 0.0                                                  # never given
    assert float(x[0, i("status.failed")]) == 1.0 and float(x[0, 15:21].sum()) == 1.0
    assert float(x[0, i("failure.page_changed")]) == 1.0 and float(x[0, 21:].sum()) == 1.0
    two = S.RewardSignals.stack([s, S.RewardSignals.from_step(info_gain=2.0)])
    x2, m2 = two.to_tensor(torch.float64), two.mask(torch.float64)
    assert x2.shape == (2, S.N_SIGNALS) and torch.equal(x2[0], x[0])
    assert float(m2[1, i("info_gain")]) == 1.0 and float(m2[0, i("info_gain")]) == 0.0
    assert float(m2[1, 15:].sum()) == 0.0                                                      # no status known
    back = S.RewardSignals.from_dict(two.to_dict())
    assert torch.equal(back.to_tensor(torch.float64), x2)
    with pytest.raises(ValueError):
        S.RewardSignals.from_dict({"schema_version": "0.9"})
    with pytest.raises(KeyError):
        S.RewardSignals.from_step(extrinsic_return=1.0)          # no slot for the verified return


# ------------------------------------------------------------------------------------------------- reward
def test_reward_is_bounded_nonlinear_and_uses_every_input():
    R = _reward(bound=0.7)
    x = _inputs()
    r = R(**x)
    assert r.shape == (6,) and float(r.detach().abs().max()) < 0.7
    big = {k: 1e6 * v for k, v in x.items()}
    assert float(R(**big).detach().abs().max()) <= 0.7
    leaves = {k: v.clone().requires_grad_(True) for k, v in x.items()}
    ctx = torch.randn(6, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    grads = torch.autograd.grad(R(**leaves, life_ctx=ctx).sum(), list(leaves.values()) + [ctx])
    assert all(float(g.abs().max()) > 0 for g in grads)                      # every input matters
    y = _inputs(seed=5)
    mix = {k: 0.5 * (x[k] + y[k]) for k in x}
    with torch.no_grad():
        assert float((R(**mix) - 0.5 * (R(**x) + R(**y))).abs().max()) > 1e-6     # not an affine map
        assert float((R(**x, life_ctx=ctx.detach()) - r).abs().max()) > 1e-6
    no_goal = R(x["b_t"], x["e_a"], x["b_next"], None, x["signals"])          # goal-free: learned null goal
    assert no_goal.shape == (6,) and bool(torch.isfinite(no_goal).all())


def test_reward_has_no_input_for_the_verified_return():
    names = set(inspect.signature(IntrinsicReward.forward).parameters)
    assert names == {"self", "b_t", "e_a", "b_next", "e_g", "signals", "life_ctx", "signal_mask"}
    assert not any("extrinsic" in n or "verified_return" in n or n == "utility" for n in S.SIGNAL_NAMES)


def test_meta_parameters_are_exactly_the_reward_group():
    R = _reward()
    assert GROUP_REWARD == "reward"
    assert {id(p) for p in R.meta_parameters()} == {id(p) for p in R.parameters()}
    assert set(R.psi()) == {n for n, _ in R.named_parameters()}
    assert any(n.startswith("lifetime.cell") for n in R.psi())               # the recurrence is meta-learned too
    assert all(p.requires_grad for p in R.meta_parameters())
    assert not any(n in R.psi() for n in ("lifetime.state", "norm_s.mean"))  # state and statistics are buffers


def test_normalisation_statistics_update_and_freeze_inside_a_meta_window():
    R = _reward()
    x = _inputs(B=64)
    shifted = {k: 5.0 + 3.0 * v for k, v in x.items()}
    R.observe(x["b_t"], x["e_a"], x["b_next"], x["e_g"], x["signals"])
    assert float(R.norm_s.count.min()) == 64 and float((R.norm_s.mean - x["signals"].mean(0)).abs().max()) < 1e-9
    r0, v0 = R(**x), R.version
    with R.meta_window():
        assert R.stats_frozen
        R.observe(shifted["b_t"], shifted["e_a"], shifted["b_next"], shifted["e_g"], shifted["signals"])
        assert torch.equal(R(**x), r0) and R.version == v0                   # same function inside the window
    assert not R.stats_frozen
    R.observe(shifted["b_t"], shifted["e_a"], shifted["b_next"], shifted["e_g"], shifted["signals"])
    assert float((R(**x) - r0).detach().abs().max()) > 1e-6 and R.version != v0   # statistics are in the version
    n = RunningNorm(3)
    a = torch.randn(200, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 4 + 2
    n.update(a[:50]); n.update(a[50:])
    assert torch.allclose(n.mean, a.mean(0)) and torch.allclose(n.var, a.var(0, unbiased=False))
    m = torch.ones(200, 3, dtype=torch.float64); m[:, 1] = 0
    n2 = RunningNorm(3); n2.update(a, m)
    assert float(n2.count[1]) == 0 and float(n2.mean[1]) == 0.0 and float(n2(a, m)[:, 1].abs().max()) == 0.0


def test_version_is_a_hash_that_changes_exactly_when_parameters_change():
    R = _reward()
    v = R.version
    assert v.startswith("rpsi1-s1.0-") and R.version == v
    R(**_inputs())                                                            # using it does not change it
    assert R.version == v
    opt = torch.optim.SGD(R.meta_parameters(), lr=0.1)
    R(**_inputs()).sum().backward()
    opt.step()
    v2 = R.version
    assert v2 != v
    twin = _reward(seed=123)
    assert twin.version != v2
    twin.load_state_dict(R.state_dict())
    assert twin.version == v2                                                 # same parameters, same version
    x = _inputs()
    R.reward_sequence(x["b_t"], x["e_a"], x["b_next"], x["e_g"], x["signals"], advance=True)
    assert R.version == v2                                                    # lifetime state is not the function


def test_lifetime_context_persists_across_objectives_and_checkpoints():
    R = _reward()
    assert int(R.lifetime.steps) == 0 and float(R.lifetime.state.abs().max()) == 0.0
    a, b = _inputs(B=4, seed=1), _inputs(B=3, seed=2)                         # two different objectives (e_g)
    r_a, ctx_a, _ = R.reward_sequence(a["b_t"], a["e_a"], a["b_next"], a["e_g"], a["signals"], advance=True)
    mid = R.lifetime.state.clone()
    assert int(R.lifetime.steps) == 4 and float(mid.abs().max()) > 0
    r_b, ctx_b, _ = R.reward_sequence(b["b_t"], b["e_a"], b["b_next"], b["e_g"], b["signals"], advance=True)
    assert torch.allclose(ctx_b[0], mid) and int(R.lifetime.steps) == 7      # nothing was reset in between
    # the context of a step summarises only earlier steps
    x = R.step_features(a["b_t"], a["e_a"], a["b_next"], a["e_g"], a["signals"])
    c_full, _ = R.lifetime.encode(x, torch.zeros(8, dtype=torch.float64))
    c_cut, _ = R.lifetime.encode(x[:2], torch.zeros(8, dtype=torch.float64))
    assert torch.allclose(c_full[:2], c_cut) and torch.allclose(c_full, ctx_a)
    # rewards of the second objective depend on what happened under the first one
    fresh = _reward(); fresh.load_state_dict(R.state_dict()); fresh.lifetime.new_lifetime()
    r_b_fresh, _, _ = fresh.reward_sequence(b["b_t"], b["e_a"], b["b_next"], b["e_g"], b["signals"])
    assert float((r_b_fresh - r_b).detach().abs().max()) > 1e-8
    # the state is part of the checkpoint
    buf = io.BytesIO(); torch.save(R.state_dict(), buf); buf.seek(0)
    again = _reward(seed=9); again.load_state_dict(torch.load(buf))
    assert torch.equal(again.lifetime.state, R.lifetime.state) and int(again.lifetime.steps) == 7
    assert "lifetime.state" in R.state_dict() and "norm_s.mean" in R.state_dict()
    lc = LifetimeContext(5, 4); lc.advance(torch.randn(3, 5)); lc.new_lifetime()
    assert int(lc.steps) == 0 and float(lc.state.abs().max()) == 0.0


def test_meta_gradient_through_the_real_reward_module_reaches_every_parameter_including_the_recurrence():
    """R_psi inside the functional inner update of a small policy; outer objective on verified utilities."""
    torch.manual_seed(0)
    R = _reward()
    T_, A = 10, 3
    g = torch.Generator().manual_seed(3)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    policy = torch.nn.Linear(D, A).double()
    b_t, e_a_all, b_next = rn(T_, D), rn(T_, A, D), rn(T_, D)
    act = torch.randint(A, (T_,), generator=g)
    sig = rn(T_, S.N_SIGNALS)
    batch = {"x": b_t, "a": act,
             "reward_inputs": dict(b_t=b_t, e_a=e_a_all[torch.arange(T_), act], b_next=b_next, e_g=None, signals=sig)}
    logp = M.functional_logp(policy)

    reward_fn = R.as_reward_fn(sequence=True)              # consecutive lifetime steps, context in-graph
    mg = M.MetaGradient(M.policy_gradient_inner_loss(logp, reward_fn), logp, inner_lr=0.5)
    with R.meta_window():
        tr = mg.inner_update(dict(policy.named_parameters()), R.psi(), [batch], K=2)
        q = M.QueryData(batch={"x": rn(8, D), "a": torch.randint(A, (8,), generator=g)}, utility=rn(8))
        res = mg.meta_gradient(tr, q)
    assert res.supported and set(res.grads) == set(R.psi())
    for name, gr in res.grads.items():
        assert float(gr.abs().max()) > 0.0, name
    assert float(res.grads["lifetime.cell.weight_hh"].abs().max()) > 0.0
    with_goal = dict(batch, reward_inputs=dict(batch["reward_inputs"], e_g=rn(T_, D)))
    tr = mg.inner_update(dict(policy.named_parameters()), R.psi(), [with_goal], K=1)
    res = mg.meta_gradient(tr, q)
    assert float(res.grads["null_goal"].abs().max()) == 0.0        # unused in goal mode: honest zero
    assert all(float(v.abs().max()) > 0 for n, v in res.grads.items() if n != "null_goal")

