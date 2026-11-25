"""Plasticity instruments on toy networks with known answers, sequential retention / acquisition measurement, and the
effect of each optional mechanism tested on its own (all others off)."""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.memory._nn import SwiGLU
from superlearner.cam_unified.memory.plasticity import (AdaptiveGroupLR, ContinualBackprop, PCGrad,
                                                        PlasticityMechanisms, PlasticityMonitor, PlasticitySwitches,
                                                        ReplayDistillation, RetentionTracker, SelectiveReinit,
                                                        effective_rank, find_unit_layers, gradient_interference,
                                                        learning_speed, reinit_units, srank)
from superlearner.cam_unified.memory.retrieval import Retriever
from superlearner.cam_unified.testing import rand_tokens, tiny_config


def _dead_net(seed=0, width=64, dead=32):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 1))
    with torch.no_grad():
        net[0].bias[:dead] = -100.0                          # these ReLU units never fire on N(0,1) inputs
    return net


def _x(n=256, seed=1):
    return torch.randn(n, 4, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------ instruments
def test_dormant_units_detected_exactly():
    net = _dead_net()
    mon = PlasticityMonitor(net).attach()
    net(_x())
    d = mon.dormant()
    assert d["fraction"] == 0.5
    (name, mask), = d["mask"].items()
    assert mask[:32].all() and not mask[32:].any()
    mon.detach()
    net(_x())                                                  # detached: no more accumulation
    assert mon._cnt[name] == 256


def test_unit_layers_found_in_mlp_transformer_and_swiglu():
    tl = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
    assert [(l.fc_in, l.fc_out) for l in find_unit_layers(tl)] == [(tl.linear1, tl.linear2)]
    sw = SwiGLU(16)
    (l,) = find_unit_layers(sw)
    assert l.fc_in is sw.w1 and l.fc_out is sw.w2 and l.fc_gate is sw.w3
    ret = Retriever(tiny_config())
    layers = find_unit_layers(ret)
    assert len(layers) >= 8
    mon = PlasticityMonitor(ret, layers).attach()
    ret.query(torch.randn(3, 8, 64), rand_tokens(3, 2, 64), rand_tokens(3, 4, 64))
    rep = mon.report()
    assert 0.0 <= rep["dormant_fraction"] < 0.5 and rep["activation_rank"]


def test_effective_rank_and_srank_of_known_spectra():
    torch.manual_seed(0)
    N, D = 400, 32

    def data(sv):
        H = torch.randn(N, len(sv))
        Q, _ = torch.linalg.qr(H - H.mean(0))
        V, _ = torch.linalg.qr(torch.randn(D, len(sv)))
        return Q @ torch.diag(torch.tensor(sv)) @ V.T
    assert effective_rank(data([1.0] * 5)) == pytest.approx(5.0, rel=1e-4)
    assert srank(data([1.0] * 5)) == 5
    assert effective_rank(data([1.0] + [1e-3] * 9)) < 1.2
    assert effective_rank(torch.randn(N, D)) > 0.9 * D
    assert effective_rank(torch.ones(N, D)) == 0.0          # collapsed representation


def test_gradient_norms_weight_change_and_update_magnitudes_per_group():
    torch.manual_seed(0)

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 3)
            self.b = nn.Linear(3, 1)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))
    m = Two()
    mon = PlasticityMonitor(m)
    mon.set_reference()
    loss = m(torch.randn(8, 3)).pow(2).mean()
    loss.backward()
    gn = mon.grad_norms()
    want_a = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.a.parameters()))
    assert gn["a"] == pytest.approx(want_a) and gn["b"] > 0
    mon.begin_step()
    with torch.no_grad():
        m.a.weight.add_(0.5)                                 # 9 entries changed by 0.5 -> L2 1.5
    up = mon.end_step()
    assert up["a"]["l2"] == pytest.approx(1.5) and up["b"]["l2"] == 0.0
    wc = mon.weight_change()
    assert wc["a"]["l2"] == pytest.approx(1.5) and wc["b"]["l2"] == 0.0 and wc["all"]["l2"] == pytest.approx(1.5)
    assert wc["a"]["relative"] == pytest.approx(1.5 / math.sqrt(sum(float(v.pow(2).sum()) for k, v in
                                                                    mon._ref.items() if k.startswith("a."))))


def test_gradient_interference_signs():
    w = nn.Parameter(torch.randn(5))
    g = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0])
    h = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    r = gradient_interference({"t1": (w * g).sum(), "t2": -(w * g).sum() * 3, "t3": (w * h).sum()}, [("core.w", w)])
    cos = torch.tensor(r["cos"])
    assert cos[0, 1] == pytest.approx(-1.0) and cos[0, 2] == pytest.approx(0.0, abs=1e-7)
    assert r["by_group"]["core"][0][1] == pytest.approx(-1.0) and r["norms"][1] == pytest.approx(3 * float(g.norm()))


def test_learning_speed_detects_lost_plasticity():
    xs = torch.linspace(-2, 2, 128).unsqueeze(1)
    ys = torch.sin(2 * xs)

    def mk(dead):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
        if dead:
            with torch.no_grad():
                net[0].bias.fill_(-100.0)
        return net
    loss_fn = lambda m, b: ((m(b[0]) - b[1]) ** 2).mean()   # noqa: E731
    healthy = learning_speed(mk(False), lambda t: (xs, ys), loss_fn, steps=300, lr=1e-2, threshold=0.05)
    dead = learning_speed(mk(True), lambda t: (xs, ys), loss_fn, steps=300, lr=1e-2, threshold=0.05)
    assert healthy["steps_to_threshold"] is not None and healthy["final"] < 0.05
    assert dead["steps_to_threshold"] is None and dead["final"] > 0.3 and dead["auc"] > healthy["auc"]


def test_sequential_retention_and_new_task_acquisition_are_measured():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    tasks = {"A": (lambda x: torch.sin(3 * x), (-1.0, 1.0)), "B": (lambda x: -torch.sin(3 * x), (-1.0, 1.0)),
             "C": (lambda x: torch.cos(2 * x), (-1.0, 1.0))}
    g = torch.Generator().manual_seed(1)

    def batch(t):
        f, (lo, hi) = tasks[t]
        x = torch.rand(128, 1, generator=g) * (hi - lo) + lo
        return x, f(x)

    def score(t):
        x, y = batch(t)
        with torch.no_grad():
            return -float(((net(x) - y) ** 2).mean())
    tr = RetentionTracker()
    for t in tasks:
        sp = learning_speed(net, lambda s, t=t: batch(t), lambda m, b: ((m(b[0]) - b[1]) ** 2).mean(), steps=300,
                            lr=3e-3, threshold=0.02, copy_model=False)
        tr.trained(t)
        tr.record_acquisition(t, steps_to_threshold=sp["steps_to_threshold"], auc=sp["auc"])
        for u in tasks:
            tr.record(u, t, score(u))
    rep = tr.report()
    assert rep["order"] == ["A", "B", "C"] and set(rep["matrix"]["A"]) == {"A", "B", "C"}
    assert rep["forgetting"]["A"] > 0.1                       # B contradicts A: naive training forgets it
    assert rep["acquisition"]["B"]["steps_to_threshold"] is not None
    assert rep["backward_transfer"] < 0 and rep["final"]["C"] > -0.05


# ------------------------------------------------------------------ mechanisms (each on its own)
def _train(net, steps, opt, mech_step=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(128, 4, generator=g)
        y = (x[:, :1] * 2 - x[:, 1:2]).sin()
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if mech_step is not None:
            mech_step()


def test_continual_backprop_replaces_useless_units():
    def dormant_after(cbp_on):
        net = _dead_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        cbp = ContinualBackprop(find_unit_layers(net), opt, replacement_rate=0.05, maturity=5, decay=0.9) \
            if cbp_on else None
        _train(net, 40, opt, cbp.step if cbp else None)
        mon = PlasticityMonitor(net).attach()
        net(_x())
        return mon.dormant()["fraction"], cbp
    off, _ = dormant_after(False)
    on, cbp = dormant_after(True)
    assert off == 0.5                                          # dead ReLU units never recover by gradient descent
    assert on < 0.15 and sum(cbp.replaced.values()) >= 32
    assert int((cbp.age[next(iter(cbp.age))] < 40).sum()) >= 32   # the replaced units restarted their maturity


def test_reinit_preserves_the_function():
    net = _dead_net()
    x = _x()
    y0 = net(x).detach()
    (layer,) = find_unit_layers(net)
    reinit_units(layer, torch.arange(0, 32))                  # dormant units: exact preservation
    assert torch.allclose(net(x), y0)
    live = torch.arange(40, 48)
    h = torch.relu(net[0](x)).detach()
    reinit_units(layer, live, out_bias_correction=h.mean(0))   # live units: preserved in expectation over the batch
    assert torch.allclose(net(x).mean(), y0.mean(), atol=1e-5) and not torch.allclose(net(x), y0)


def test_selective_reinit_revives_dormant_units_and_resets_their_moments():
    net = _dead_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _train(net, 3, opt)
    st = opt.state[net[0].weight]
    st["exp_avg"].add_(1.0)                                    # make the moments visibly non-zero
    mon = PlasticityMonitor(net).attach()
    x = _x()
    y0 = net(x).detach()
    redo = SelectiveReinit(find_unit_layers(net), opt, tau=0.025)
    out = redo.apply(mon)
    assert list(out.values()) == [32]
    assert torch.allclose(net(x), y0)                          # outgoing weights zero: same function
    assert (st["exp_avg"][:32] == 0).all() and (st["exp_avg"][32:] != 0).all()
    assert (opt.state[net[2].weight]["exp_avg"][:, :32] == 0).all()
    mon.reset_activation_stats()
    net(x)
    assert mon.dormant()["fraction"] < 0.1


def test_adaptive_group_lr_steers_relative_update_size():
    def run(on):
        torch.manual_seed(0)
        a, b = nn.Linear(4, 1), nn.Linear(4, 1)
        opt = torch.optim.SGD([{"params": a.parameters(), "name": "a"}, {"params": b.parameters(), "name": "b"}],
                              lr=0.01)
        alr = AdaptiveGroupLR(opt, target=1e-3) if on else None
        g = torch.Generator().manual_seed(2)
        ratios = []
        for _ in range(80):
            x = torch.randn(64, 4, generator=g)
            y = x.sum(1, keepdim=True)
            loss = ((a(x) + 1e-3 * b(x) - y) ** 2).mean()          # b's gradient is 1000x smaller
            opt.zero_grad()
            loss.backward()
            pre = [p.detach().clone() for p in b.parameters()]
            if alr:
                alr.before_step()
            opt.step()
            if alr:
                alr.after_step()
            ratios.append(float(sum((p - q).norm() ** 2 for p, q in zip(b.parameters(), pre)) ** 0.5 /
                                sum(q.norm() ** 2 for q in pre) ** 0.5))
        return opt, alr, ratios
    opt_off, _, r_off = run(False)
    opt_on, alr, r_on = run(True)
    assert [g["lr"] for g in opt_off.param_groups] == [0.01, 0.01]
    assert alr.mult[1] >= 10 and opt_on.param_groups[1]["lr"] == pytest.approx(0.01 * alr.mult[1])
    assert sum(r_on[-10:]) > 8 * sum(r_off[-10:])             # vanishing group's updates were revived


def test_pcgrad_removes_gradient_conflict():
    w = nn.Parameter(torch.zeros(2))
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    info = PCGrad([w]).backward({"world": (w * g1).sum(), "policy": (w * g2).sum()})
    assert torch.allclose(w.grad, torch.tensor([0.5, 1.5])) and info["conflicts"] == 2
    assert float(w.grad @ g1) > 0 and float(w.grad @ g2) > 0       # plain sum (0, 1) is orthogonal to g1
    w.grad = None
    g3 = torch.tensor([1.0, 1.0])
    info = PCGrad([w]).backward([(w * g1).sum(), (w * g3).sum()])
    assert torch.allclose(w.grad, g1 + g3) and info["conflicts"] == 0


def test_replay_distillation_improves_retention():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    xa = torch.linspace(-3, -1, 128).unsqueeze(1)
    xb = torch.linspace(1, 3, 128).unsqueeze(1)
    ya, yb = torch.sin(2 * xa), torch.cos(3 * xb) + 1

    def fit(m, x, y, steps, distill=None):
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(steps):
            loss = ((m(x) - y) ** 2).mean()
            if distill is not None:
                loss = loss + distill.loss(m, xa)                  # replayed task-A inputs only (no A targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    fit(net, xa, ya, 400)
    base_a = float(((net(xa) - ya) ** 2).mean())
    plain, dist = copy.deepcopy(net), copy.deepcopy(net)
    fit(plain, xb, yb, 400)
    fit(dist, xb, yb, 400, ReplayDistillation.from_model(net, kind="mse"))
    err = {k: float(((m(xa) - ya) ** 2).mean()) for k, m in (("plain", plain), ("distill", dist))}
    err_b = {k: float(((m(xb) - yb) ** 2).mean()) for k, m in (("plain", plain), ("distill", dist))}
    assert base_a < 0.05 and err["distill"] < 0.5 * err["plain"], (base_a, err)
    assert err_b["distill"] < 0.1                              # the new task is still learned


def test_kl_distillation_matches_teacher_logits():
    teacher = nn.Linear(3, 5)
    rd = ReplayDistillation(copy.deepcopy(teacher), kind="kl", temperature=2.0)
    assert float(rd.loss(teacher, torch.randn(10, 3))) == pytest.approx(0.0, abs=1e-6)
    assert float(rd.loss(nn.Linear(3, 5), torch.randn(10, 3))) > 0


def test_mechanism_wrapper_applies_exactly_the_switched_on_mechanisms():
    def build(**on):
        net = _dead_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        return net, opt, PlasticityMechanisms(net, opt, PlasticitySwitches(**on), reinit_interval=2,
                                              cbp_kwargs={"replacement_rate": 0.05, "maturity": 1})
    for on, key in ((dict(), None), (dict(continual_backprop=True), "continual_backprop"),
                    (dict(selective_reinit=True), "selective_reinit"), (dict(adaptive_lr=True), "adaptive_lr")):
        net, opt, mech = build(**on)
        infos = []
        for _ in range(4):
            x = _x(64)
            mech.backward({"fit": net(x).pow(2).mean()})
            mech.before_step()
            opt.step()
            opt.zero_grad()
            infos.append(mech.after_step())
        keys = set().union(*[set(i) for i in infos])
        assert keys == (set() if key is None else {key}), (on, keys)
    net, opt, mech = build(pcgrad=True)
    assert mech.pcg is not None and mech.cbp is None and mech.alr is None and mech.redo is None
    info = mech.backward({"a": net(_x(8)).sum(), "b": -net(_x(8)).sum()})
    assert "conflicts" in info
    net, opt, mech = build(replay_distill=True)
    assert float(mech.distill_loss(_x(8))) == pytest.approx(0.0, abs=1e-7)
    _, _, off = build()
    assert off.distill_loss(_x(8)) is None
