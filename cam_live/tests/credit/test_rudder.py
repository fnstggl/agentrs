"""ReturnDecomposer: causality, conservation under duration-aware discounting, the key decision gets the credit,
pending trajectories get nothing (RC-02, RC-03, RC-06)."""
import json
import math

import pytest
import torch

from cam_live.credit.replay import SqliteExperienceStore
from cam_live.credit.rudder import ReturnDecomposer, discounted_return, duration_discount, write_credit

D_IN = 4


def _data(n, T, seed, p_key=0.5, noise=0.05, var_len=False):
    """Delayed-reward sequences: one randomly placed key decision (taken or not); the return 1[key taken] + noise is
    only known at the end; all other decisions are distractors with random features."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, D_IN)
    x[:, :, 2] = torch.randn(n, T, generator=g)                       # distractor feature
    x[:, :, 3] = 1.0
    lengths = torch.randint(T // 2, T + 1, (n,), generator=g) if var_len else torch.full((n,), T)
    key_pos = (torch.rand(n, generator=g) * lengths).long()
    taken = (torch.rand(n, generator=g) < p_key)
    idx = torch.arange(n)
    x[idx, key_pos, 0] = 1.0
    x[idx, key_pos, 1] = taken.float() * 2 - 1
    mask = torch.arange(T)[None, :] < lengths[:, None]
    G = taken.double() + noise * torch.randn(n, generator=g, dtype=torch.float64)
    return x, mask, G, key_pos, taken


def _train(model, x, mask, G, matured, steps=350, lr=3e-3, times=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, x.shape[0], (128,), generator=gen)
        loss, _ = model.loss(x[i], mask[i], G[i], matured[i], times=None if times is None else times[i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def test_predictions_are_causal():
    torch.manual_seed(0)
    m = ReturnDecomposer(D_IN, d_model=32, d_goal=8)
    torch.nn.init.normal_(m.head.weight, std=0.5)
    x, mask, _, _, _ = _data(4, 9, seed=1)
    e_g, times = torch.randn(4, 8), torch.cumsum(torch.rand(4, 9, dtype=torch.float64) * 600, 1)
    base = m(x, mask, e_g, times)
    for j in range(9):
        x2, t2 = x.clone(), times.clone()
        x2[:, j:] = torch.randn(4, 9 - j, D_IN)                       # change decision j and everything after it
        t2[:, j:] = t2[:, j:] + 1234.0
        out = m(x2, mask, e_g, t2)
        assert torch.allclose(out[:, :j + 1], base[:, :j + 1], atol=1e-6), j     # predictions before j unchanged
        assert not torch.allclose(out[:, j + 1:], base[:, j + 1:], atol=1e-6)


@pytest.mark.parametrize("beta", [0.0, 2e-5, 1e-3])
def test_conservation_to_numerical_precision_under_duration_aware_discounting(beta):
    """sum_t exp(-beta * tau_t) c_t = G for an arbitrary (untrained, random) predictor, variable lengths and random
    decision durations; the return-to-go at the first decision is the whole return."""
    torch.manual_seed(1)
    m = ReturnDecomposer(D_IN, d_model=32, beta=beta, scale=3.0)
    torch.nn.init.normal_(m.head.weight, std=1.0)
    x, mask, G, _, _ = _data(64, 14, seed=2, var_len=True)
    G = G * 7.3 - 2.0
    times = 1.7e9 + torch.cumsum(torch.rand(64, 14, dtype=torch.float64) * 5000.0, 1)      # seconds, irregular
    red = m.redistribute(x, mask, G, torch.ones(64, dtype=torch.bool), times=times)
    assert float(red.conservation_error(G).max()) < 1e-9
    tau = times - times[:, :1]
    manual = (torch.exp(-beta * tau) * red.contrib * mask).sum(1)
    assert torch.allclose(manual, G, atol=1e-9, rtol=0)
    assert torch.allclose(red.discount[mask], duration_discount(tau, beta)[mask])
    assert torch.allclose(red.return_to_go[:, 0], G, atol=1e-9)                               # Gamma_first = 1
    # the learning signal is the return-to-go minus the prediction made before the decision, at decision time
    assert torch.allclose(red.signal[mask], ((G[:, None] - red.pred[:, :-1]) / red.discount)[mask], atol=1e-9)
    assert torch.allclose(red.return_to_go[:, 1:][mask[:, 1:]], red.signal[:, 1:][mask[:, 1:]], rtol=1e-9, atol=1e-9)
    assert torch.allclose((red.return_to_go - red.signal)[:, 0], red.offset, atol=1e-9)
    # padded steps carry nothing
    assert float(red.contrib[~mask].abs().max()) == 0.0
    if beta > 0:
        assert float((red.discount[mask] < 1).double().mean()) > 0.8
        # discounting an outcome to the start and conserving it are the same convention
        v, t_out = torch.tensor([10.0], dtype=torch.float64), torch.tensor([5000.0], dtype=torch.float64)
        assert float(discounted_return(v, t_out, torch.zeros(1), beta)) == pytest.approx(10 * math.exp(-beta * 5000))


def test_linear_scale_only_rescales_the_redistribution():
    torch.manual_seed(3)
    a = ReturnDecomposer(D_IN, d_model=32, scale=1.0)
    torch.nn.init.normal_(a.head.weight, std=1.0)
    b = ReturnDecomposer(D_IN, d_model=32, scale=1.0)
    b.load_state_dict(a.state_dict())
    b.scale.fill_(100.0)
    x, mask, G, _, _ = _data(8, 6, seed=4)
    ones = torch.ones(8, dtype=torch.bool)
    ra, rb = a.redistribute(x, mask, G, ones), b.redistribute(x, mask, 100 * G, ones)
    assert torch.allclose(rb.contrib, 100 * ra.contrib, atol=1e-9) and torch.allclose(rb.signal, 100 * ra.signal, atol=1e-9)
    assert b.fit_scale(torch.tensor([1.0, -250.0, 9999.0]), torch.tensor([True, True, False])) == 250.0


def test_key_decision_receives_the_credit():
    torch.manual_seed(0)
    x, mask, G, key_pos, taken = _data(4000, 10, seed=5)
    matured = torch.ones(4000, dtype=torch.bool)
    model = ReturnDecomposer(D_IN, d_model=32, n_layers=2, n_heads=4)
    model.fit_scale(G, matured)
    # before training (F = 0) the whole return sits in the final residual: the key decision gets nothing as an
    # immediate contribution, while the learning signal (return-to-go) already separates taken from not taken
    xt, mt, Gt, kt, tk = _data(1000, 10, seed=6)
    ones = torch.ones(1000, dtype=torch.bool)
    idx = torch.arange(1000)
    r0 = model.redistribute(xt, mt, Gt, ones)
    not_last = kt < 9
    assert float(r0.contrib[idx, kt][not_last].abs().max()) == 0.0
    assert float(r0.signal[idx, kt][tk].mean() - r0.signal[idx, kt][~tk].mean()) > 0.9
    _train(model, x, mask, G, matured)
    red = model.redistribute(xt, mt, Gt, ones)
    assert float(red.conservation_error(Gt).max()) < 1e-9
    c_key = red.diff[idx, kt]
    target = tk.double() - 0.5                                   # E[G | key decision] - E[G | before it]
    key_err = float((c_key - target).abs().mean())
    others = red.diff.clone()
    others[idx, kt] = 0.0
    other_mean = float(others.abs().sum(1).mean() / 9)
    top = float((red.diff.abs().argmax(1) == kt).double().mean())
    assert key_err < 0.08 and other_mean < 0.03 and top > 0.97, (key_err, other_mean, top)
    assert float(red.offset.mean()) == pytest.approx(0.5, abs=0.06)          # initial offset = E[G]
    assert float(red.residual.abs().mean()) < 0.1                            # residual = unexplained noise
    # squared loss on the scaled return gives the conditional mean after the key decision
    after = red.pred[idx, kt + 1]
    assert float(after[tk].mean()) == pytest.approx(1.0, abs=0.06) and float(after[~tk].mean()) == pytest.approx(0.0, abs=0.06)
    # the signal at the key decision is an unbiased advantage: mean +-0.5 by arm
    sig = red.signal[idx, kt]
    assert float(sig[tk].mean()) == pytest.approx(0.5, abs=0.06) and float(sig[~tk].mean()) == pytest.approx(-0.5, abs=0.06)
    print("RUDDER_KEY", json.dumps({"key_abs_err": key_err, "other_abs_mean": other_mean, "key_is_argmax": top,
                                    "offset": float(red.offset.mean()), "residual_abs": float(red.residual.abs().mean())}))


def test_pending_sequences_yield_no_credit_and_no_loss(tmp_path):
    torch.manual_seed(0)
    x, mask, G, _, _ = _data(12, 6, seed=7)
    matured = torch.tensor([True] * 6 + [False] * 6)
    model = ReturnDecomposer(D_IN, d_model=32)
    torch.nn.init.normal_(model.head.weight, std=0.3)
    # whatever stands in the return slot of a pending trajectory has no influence at all
    Ga, Gb = G.clone(), G.clone()
    Ga[6:], Gb[6:] = 0.0, 1e9
    la, _ = model.loss(x, mask, Ga, matured)
    lb, info = model.loss(x, mask, Gb, matured)
    ga = torch.autograd.grad(la, list(model.parameters()), allow_unused=True)
    gb = torch.autograd.grad(lb, list(model.parameters()), allow_unused=True)
    assert float(la.detach()) == float(lb.detach()) and info["n_matured"] == 6.0
    assert all(torch.equal(p, q) for p, q in zip(ga, gb) if p is not None)
    none, info = model.loss(x, mask, Gb, torch.zeros(12, dtype=torch.bool))
    assert none is None and info["n_matured"] == 0.0
    red = model.redistribute(x, mask, Gb, matured)
    assert red.matured.tolist() == [True] * 6 + [False] * 6
    for t in (red.contrib, red.signal, red.return_to_go, red.diff):
        assert torch.isnan(t[6:]).all() and torch.isfinite(t[:6]).all()
    assert torch.isnan(red.residual[6:]).all() and torch.isnan(red.conservation_error(Gb)[6:]).all()
    # nothing is written to the records of pending trajectories
    from conftest import build_record
    st = SqliteExperienceStore(str(tmp_path / "e.sqlite"), durable=False)
    ids = []
    with st.batch():
        for b in range(12):
            row = []
            for t in range(6):
                rec = build_record(b * 6 + t, objective=f"obj{b}")
                st.append(rec)
                row.append(rec.exp_id)
            ids.append(row)
    assert write_credit(st, ids, red) == 36
    for b in range(12):
        for t, e in enumerate(ids[b]):
            cr = st.get(e).credit
            if b < 6:
                assert cr["rudder"] == pytest.approx(float(red.signal[b, t])) and "rudder_contribution" in cr
            else:
                assert cr == {}
