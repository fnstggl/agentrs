"""Experiential core: causality, persistent gated state, direct current-event path, gradient reach (NA-02, NA-05)."""
import torch
import torch.nn.functional as Fn

from cam_live.contracts import (ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity)
from cam_live.nn import fast
from cam_live.nn.api import CoreState, GROUP_EXPERIENTIAL


def test_h_is_causal(model, items):
    base = items(8)
    tb = model.tokenize([base])
    h0 = model.core_forward(tb).h
    for k in (3, 6, 7):
        changed = list(base)
        changed[k] = ExperienceItem(kind=ItemKind.FAILURE, text="something completely different happened",
                                    ts=base[k].ts + 500000.0, quantities=[Quantity("other", -9e6)])
        for j in range(k + 1, 8):               # later items move in time too
            changed[j] = ExperienceItem(kind=base[j].kind, text=base[j].text + " changed", ts=base[j].ts + 500000.0,
                                        quantities=base[j].quantities)
        h1 = model.core_forward(model.tokenize([changed])).h
        assert torch.equal(h0[:, :k], h1[:, :k]), k            # earlier latents are bit-identical
        assert not torch.allclose(h0[:, k:], h1[:, k:], atol=1e-3)
    # appending items never changes the latents of the prefix
    h_short = model.core_forward(model.tokenize([base[:5]])).h
    assert torch.allclose(h_short, h0[:, :5], atol=1e-6)


def test_padding_does_not_change_a_row(model, items):
    a, b = items(3, tag="a"), items(7, tag="b")
    alone = model.core_forward(model.tokenize([a]))
    both = model.core_forward(model.tokenize([a, b]))
    assert torch.allclose(alone.b[0], both.b[0], atol=1e-5)
    assert torch.allclose(alone.h[0], both.h[0, :3], atol=1e-5)
    assert torch.allclose(alone.state.slots[0], both.state.slots[0], atol=1e-5)
    assert float(both.h[0, 3:].abs().sum()) == 0.0


def test_belief_reads_the_current_event_directly(model, items):
    """b must depend on the newest event more strongly than on any earlier one (an earlier core could not read the
    current step through attention-mixed history), and the direct path alone already carries it."""
    tb = model.tokenize([items(8)])
    x = tb.x.detach().clone().requires_grad_(True)
    tb.x = x
    out = model.core_forward(tb)
    probe = torch.randn(out.b.shape, generator=torch.Generator().manual_seed(0))
    (out.b * probe).sum().backward()
    g = x.grad[0].norm(dim=-1)
    assert float(g[-1]) > 2.0 * float(g[:-1].max())
    core = model.exp.core
    direct = core.direct(core.direct_norm(tb.x[0, -1].detach()))
    assert float(direct.norm()) > 0.25 * float(out.h[0, -1].norm())      # not a vanishing contribution
    # changing only the newest item changes b a lot
    its = items(8)
    its2 = its[:-1] + [ExperienceItem(kind=ItemKind.OUTCOME, text="a new result", ts=its[-1].ts,
                                      quantities=[Quantity("x", -4.0)])]
    b1, b2 = model.core_forward(model.tokenize([its])).b, model.core_forward(model.tokenize([its2])).b
    assert float(Fn.cosine_similarity(b1, b2, -1)) < 0.9


def test_state_is_gated_input_dependent_and_carried(model, items):
    """An earlier ungated state was 99.5 % constant across inputs and decayed."""
    words = ("alpha", "river", "seven", "click", "error", "table", "green", "motor")
    ctxs = [[ExperienceItem(kind=ItemKind.ALL[i % len(ItemKind.ALL)], ts=10.0 + i,
                            text=" ".join(words[(i * (j + 3) + j) % 8] for j in range(1 + i % 5)),
                            quantities=[Quantity(f"m{i % 4}", float(i - 16))] if i % 3 else [])] for i in range(32)]
    out = model.core_forward(model.tokenize(ctxs))
    s = out.state.slots                                             # [32, K, D]
    delta = s - model.init_state(32).slots
    const_fraction = float(delta.mean(0).pow(2).mean() / delta.pow(2).mean())
    assert const_fraction < 0.5, const_fraction                    # most of the written state depends on the input
    assert float(delta.pow(2).mean().sqrt()) > 0.05                # and it is not a negligible write
    assert 0.02 < float(out.aux["gate_mean"]) < 0.3                # gate starts mostly closed (negative bias)
    assert float(model.exp.core.gate_cand.bias.mean()) < 0
    assert out.state.step == 1 and out.state.last_time.tolist() == [10.0 + i for i in range(32)]
    # carried between calls: the same second call gives a different belief after a different first call
    second = [[ExperienceItem(kind=ItemKind.ACTION, text="the same second event", ts=500.0)]] * 2
    tb2 = model.tokenize(second)
    o2 = model.core_forward(tb2, CoreState(s[:2], out.state.last_time[:2], 1))
    assert not torch.allclose(o2.b[0], o2.b[1], atol=1e-4)
    assert o2.state.step == 2
    # bounded, not decaying: 200 calls keep the state at a stable non-trivial scale
    state = None
    rms = []
    for t in range(200):
        o = model.core_forward(model.tokenize([[ctxs[t % 32][0]]]), state)
        state = CoreState(o.state.slots.detach(), o.state.last_time, o.state.step)
        rms.append(float(state.slots.pow(2).mean().sqrt()))
    assert 0.5 < rms[-1] < 2.0 and abs(rms[-1] - rms[100]) < 0.2
    # an empty context leaves the state untouched; elapsed time since the last call conditions the read
    empty = model.core_forward(model.tokenize([[]]), state)
    assert torch.equal(empty.state.slots, state.slots) and torch.equal(empty.state.last_time, state.last_time)
    soon = model.core_forward(model.tokenize([[ExperienceItem(kind=ItemKind.EVENT, text="next", ts=state.last_time.item() + 1.0)]]), state)
    late = model.core_forward(model.tokenize([[ExperienceItem(kind=ItemKind.EVENT, text="next", ts=state.last_time.item() + 2.6e6)]]), state)
    assert not torch.allclose(soon.b, late.b, atol=1e-5)


def test_persistent_state_is_learnable_eight_step_recall(cfg):
    """A symbol shown in call 1 must be reported after call 8; each call carries one token, so the only route is
    the persistent slots. Chance is 25 %."""
    from cam_live.nn.model import build_model
    torch.manual_seed(0)
    model = build_model(cfg)
    n_cls, n_steps = 4, 8
    head = torch.nn.Linear(model.d_model, n_cls)
    params = model.param_groups()[GROUP_EXPERIENTIAL] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    g = torch.Generator().manual_seed(0)
    sym = [ExperienceItem(kind=ItemKind.OBSERVATION, text=f"symbol {'abcd'[k]}", quantities=[Quantity("code", float(k))])
           for k in range(n_cls)]
    dis = [ExperienceItem(kind=ItemKind.EVENT, text=f"distractor {w}") for w in ("red", "green", "blue", "tall", "round")]

    def episode(B):
        y = torch.randint(0, n_cls, (B,), generator=g)
        state, out = None, None
        for t in range(n_steps):
            ctx = []
            for b in range(B):
                it = sym[int(y[b])] if t == 0 else dis[int(torch.randint(0, len(dis), (1,), generator=g))]
                ctx.append([ExperienceItem(kind=it.kind, text=it.text, ts=1000.0 + 60.0 * t, quantities=it.quantities)])
            out = model.core_forward(model.tokenize(ctx), state)
            state = out.state
        return head(out.b), y

    with torch.no_grad():
        logits, y = episode(128)
    acc_before = float((logits.argmax(-1) == y).float().mean())
    for _ in range(60):
        logits, y = episode(32)
        loss = Fn.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        logits, y = episode(256)
    acc = float((logits.argmax(-1) == y).float().mean())
    assert acc > 0.9 and acc_before < 0.6, (acc_before, acc)


def test_gradients_reach_every_experiential_parameter(model, items):
    ctx = [items(5, tag="a") + [ExperienceItem(kind=ItemKind.COST, text="", ts=2000.0,
                                               quantities=[Quantity("cost.money", 2.5, unit="usd")])],
           items(4, tag="b")]
    first = model.core_forward(model.tokenize(ctx))
    nxt = [[ExperienceItem(kind=ItemKind.OUTCOME, text="later result", ts=9000.0, quantities=[Quantity("m", 3.0)])]] * 2
    out = model.core_forward(model.tokenize(nxt), first.state)          # reads the slots written by the first call
    acts = [[ActionSpec("a1", "browser.click", arguments={"ref": 1}), ActionSpec("a2", "internal.wait")],
            [ActionSpec("a3", "sandbox.python", modality="code")]]
    e_a, a_mask = model.encode_actions(acts)
    e_g = model.encode_goals(["reach the target", "another objective"])
    e_q, q_mask, kind_ids, horizon = model.encode_queries(
        [[OutcomeQuery("q1", "members", kind=OutcomeKind.COUNT, horizon_s=86400.0, description="count after a day")],
         [OutcomeQuery("q2", "latency", horizon_s=60.0), OutcomeQuery("q3", "done", kind=OutcomeKind.BINARY)]])
    z = model.decision_latent(out.b, e_a, e_g)
    phi = model.features(z, e_q)
    assert phi.shape == (2, 2, 2, model.n_features)
    # ALPaCA-style loss through the Bayesian layer: features and the learned prior both get gradient
    seq = phi[:, :, 0, :]                                                   # [2, 2, F] as a sequence of two rows
    y = torch.tensor([[0.3, -1.0], [2.0, 0.5]], dtype=torch.float64)
    nll = fast.online_nll(model.exp.fast_prior.prior(OutcomeKind.REAL), seq, y, OutcomeKind.REAL,
                          noise_mode=model.exp.fast_prior.noise_mode)
    cnt = fast.online_nll(model.exp.fast_prior.prior(OutcomeKind.COUNT), phi[:, 0], y.abs().round(), OutcomeKind.COUNT)
    w = torch.randn(phi.shape, generator=torch.Generator().manual_seed(1))
    loss = nll.sum() + cnt.sum() + (phi * w).sum() + out.h.pow(2).mean()
    model.zero_grad()
    loss.backward()
    named = {n: p for n, p in model.exp.named_parameters()}
    dead = [n for n, p in named.items() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    # every experiential tensor (tokenizer, core, encoders, trunk, feature map, learned prior) is on the loss path
    assert dead == [], dead
    norms = model.grad_norms()
    assert norms[GROUP_EXPERIENTIAL] > 0 and norms["semantic_adapters"] == 0.0 and norms["backbone_frozen"] == 0.0
    # double backward through the core (needed for meta-gradients on cached latents)
    tb = model.tokenize([items(4)])
    tb.x = tb.x.detach().requires_grad_(True)
    b = model.core_forward(tb).b
    (g1,) = torch.autograd.grad(b.pow(2).sum(), tb.x, create_graph=True)
    (g2,) = torch.autograd.grad(g1.pow(2).sum(), tb.x)
    assert torch.isfinite(g2).all() and float(g2.abs().sum()) > 0
