"""PS — the experiential core's persistent latent state carried across decisions (amendment item 3, KF-E04).

What is pinned here (``cam_live.nn.persist``, ``ExperientialCore.persist_rho``):
* compatibility: no chain / disabled chain / nothing committed / rho = 0 compute exactly what the core computed before;
* decision-time and replayed beliefs are the same function (context window -> its position's state, post window ->
  the next position's state);
* the carried state transports information beyond the context window, also across objectives;
* gradients flow through the K recomputed transitions (finite differences), the anchor is a constant;
* durability: the chain survives a reopen of its database; commits are idempotent; a legacy checkpoint closes rho.
"""
import math

import pytest
import torch

from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.nn.api import CoreState
from cam_live.nn.model import build_model
from cam_live.nn.persist import PersistentStateChain, PersistSettings, window_key


def _item(ref, t, text, v=None, kind=ItemKind.OBSERVATION):
    qs = [Quantity("reading", float(v))] if v is not None else []
    return ExperienceItem(kind=kind, text=text, ts=float(t), ref=ref, quantities=qs)


class Stream:
    """A fake agent: items into the store, a sliding window of refs, decisions committed to the chain."""

    def __init__(self, model, store, chain, width=4):
        self.model, self.store, self.chain, self.width = model, store, chain, width
        self.ctx, self.t, self.k = [], 1000.0, 0
        self.live_b, self.windows, self.posts = [], [], []

    def put(self, text, v=None, kind=ItemKind.OBSERVATION):
        self.k += 1
        self.t += 60.0
        it = _item(f"it{self.k}", self.t, text, v, kind)
        self.store.put_item(it)
        self.ctx.append(it.ref)
        return it

    def window(self):
        return list(self.ctx[-self.width:])

    def decide(self, observe):
        for text, v in observe:
            self.put(text, v)
        w = self.window()
        with self.chain.live():
            b = self.model.core_forward(self.model.tokenize([self.store.items(w)]), None).b
        self.put(f"action {self.k}", None, ItemKind.ACTION)
        post = self.window()
        self.chain.commit(f"dec{len(self.windows)}", context_refs=w, post_refs=post, objective_id="o", ts=self.t)
        self.live_b.append(b.detach())
        self.windows.append(w)
        self.posts.append(post)
        return b

    def replay(self, refs):
        return self.model.core_forward(self.model.tokenize([self.store.items(refs)]), None).b


@pytest.fixture
def setup(tmp_path, cfg):
    torch.manual_seed(0)
    model = build_model(cfg)
    store = SqliteExperienceStore(str(tmp_path / "s.sqlite"), durable=False)
    chain = PersistentStateChain(model, store, PersistSettings(enabled=True, unroll=2), path=str(tmp_path / "core_state.sqlite"))
    return model, store, chain, Stream(model, store, chain)


def _run(stream, n=7, tag="a"):
    for i in range(n):
        stream.decide([(f"{tag} reading {i}", float(i * 3 - 4))])


def test_without_chain_or_commits_or_with_rho_zero_the_core_is_unchanged(setup, cfg):
    model, store, chain, s = setup
    ctx = [s.put("x one", 1.0), s.put("x two", 2.0), s.put("x three", None)]
    tok = model.tokenize([ctx])
    ref = model.exp.core(tok, None)
    with chain.live():                                            # nothing committed: the live state is the fresh one
        assert torch.equal(model.core_forward(tok, None).b, ref.b)
    assert torch.equal(model.core_forward(tok, None).b, ref.b)    # not live, not stored: fresh
    _run(s, 5)
    chain.settings.enabled = False
    with chain.live():
        assert torch.equal(model.core_forward(tok, None).b, ref.b)
    chain.settings.enabled = True
    with chain.live():
        carried = model.core_forward(tok, None).b
    assert not torch.allclose(carried, ref.b, atol=1e-5)          # the carried state changes the belief ...
    with torch.no_grad():
        model.exp.core.persist_rho.zero_()
    with chain.live():
        closed = model.core_forward(tok, None)
    assert torch.equal(closed.b, ref.b) and torch.equal(closed.h, ref.h)  # ... and rho = 0 is the old computation
    chain.detach()
    assert model.persistent is None and torch.equal(model.core_forward(tok, None).b, ref.b)


def test_decision_time_and_replayed_beliefs_are_the_same_function(setup):
    model, store, chain, s = setup
    _run(s, 8)
    with torch.no_grad():
        for t, w in enumerate(s.windows):
            assert torch.allclose(s.replay(w), s.live_b[t], atol=1e-6, rtol=0), t
        # a batch of stored windows of different positions resolves row by row (as the learner encodes records)
        tok = model.tokenize([store.items(w) for w in s.windows[2:6]])
        b = model.core_forward(tok, None).b
        assert torch.allclose(b, torch.cat(s.live_b[2:6]), atol=1e-6, rtol=0)
        # the post-result window of decision t is read with the state of position t + 1
        for t, p in enumerate(s.posts):
            st = chain.state_for([t + 1])
            want = model.exp.core(model.tokenize([store.items(p)]), st).b
            assert torch.allclose(s.replay(p), want, atol=1e-6, rtol=0)
            fresh = model.exp.core(model.tokenize([store.items(p)]), None).b
            if t >= 1:
                assert not torch.allclose(want, fresh, atol=1e-5)
    c = chain.counters
    assert c["rows_context"] >= 12 and c["rows_post"] >= 8 and c["rows_live"] == 8 and c["commits"] == 8


def test_information_beyond_the_window_reaches_later_beliefs(tmp_path, cfg):
    """Two streams differ only in what was observed 6 decisions ago — long out of the 4-item window — and in the
    same way across an objective boundary (a window that starts empty)."""
    beliefs = []
    for secret in (5.0, -5.0):
        torch.manual_seed(0)
        model = build_model(cfg)
        store = SqliteExperienceStore(str(tmp_path / f"s{secret}.sqlite"), durable=False)
        chain = PersistentStateChain(model, store, PersistSettings(enabled=True, unroll=2))
        s = Stream(model, store, chain)
        s.decide([("the hidden setting is", secret)])
        _run(s, 6, tag="same")
        s.ctx = []                                                # a new objective: the window starts empty
        b = s.decide([("new objective starts", 1.0)])
        beliefs.append(b)
        assert all(r not in s.windows[-1] for r in s.windows[0])
    assert float((beliefs[0] - beliefs[1]).detach().abs().max()) > 1e-4
    torch.manual_seed(0)                                          # without the chain both beliefs are identical
    model = build_model(cfg)
    store = SqliteExperienceStore(str(tmp_path / "plain.sqlite"), durable=False)
    s = Stream(model, store, PersistentStateChain(model, store, PersistSettings(enabled=False)))
    s.decide([("the hidden setting is", 5.0)])
    _run(s, 6, tag="same")
    s.ctx = []
    b = s.decide([("new objective starts", 1.0)])
    assert torch.allclose(b, s.replay(s.windows[-1]), atol=0)


def test_gradients_flow_through_the_recomputed_transitions(setup):
    """Finite differences of a replayed belief with respect to parameters that act only through the carried state
    (the slot-write gate and the retention), and to the content of a window K = 2 decisions back."""
    model, store, chain, s = setup
    _run(s, 6)
    core = model.exp.core
    w = s.windows[5]

    r = torch.randn(model.d_model, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    def f():                                                      # a projection (|b| is fixed by the output norm)
        return (s.replay(w).double() @ r).sum()

    with torch.no_grad():
        core.persist_rho.fill_(0.7)                               # inside (0, 1): the clamp is not active
    for p, idx in ((core.gate_cand.bias, 3), (core.persist_rho, 1), (core.write_attn.v.weight, (2, 5))):
        model.zero_grad()
        f().backward()
        g = float(p.grad[idx])
        eps = 1e-2
        with torch.no_grad():
            old = float(p[idx])
            p[idx] = old + eps
            up = float(f())
            p[idx] = old - eps
            dn = float(f())
            p[idx] = old
        fd = (up - dn) / (2 * eps)
        assert abs(g) > 1e-6 and abs(g - fd) <= 2e-2 * abs(fd) + 1e-5, (p.shape, g, fd)
    # the content of window 3 (K = 2 back from position 5) enters through the recomputed transitions: changing its
    # newest reading changes b_5 although no item of window 3 is in window 5
    assert not set(s.windows[3]) & set(w)
    with torch.no_grad():
        b0 = s.replay(w)
        it = store.items([s.windows[3][-1]])[0]
        store.put_item(ExperienceItem(kind=it.kind, text=it.text, ts=it.ts, ref="changed", quantities=[Quantity("reading", 99.0)]))
        chain._refs[3] = s.windows[3][:-1] + ["changed"]
        b1 = s.replay(w)
        chain._refs[3] = s.windows[3]
        # ... while a window more than K back acts only through its stored anchor (a constant)
        chain._refs[1] = s.windows[1][:-1] + ["changed"]
        b2 = s.replay(w)
        chain._refs[1] = s.windows[1]
    assert float((b1 - b0).abs().max()) > 1e-5
    assert torch.equal(b2, b0)


def test_anchor_is_the_state_the_decision_used_and_survives_a_reopen(tmp_path, setup, cfg):
    model, store, chain, s = setup
    _run(s, 5)
    live = chain.live_state()
    sd = chain.state_dict()
    path = chain.path
    chain.close()
    again = PersistentStateChain(model, store, PersistSettings(enabled=True, unroll=2), path=path)
    rep = again.load_state_dict(sd)
    assert rep["consistent"] and rep["positions_at_checkpoint"] == 5 and again.n == 5
    assert torch.allclose(again.live_state().slots, live.slots, atol=1e-6)
    for t, w in enumerate(s.windows):
        assert torch.allclose(s.replay(w), s.live_b[t], atol=1e-6)
    # idempotent per decision: a second delivery only updates the post window
    pos = again.commit("dec2", context_refs=s.windows[2], post_refs=s.posts[3], objective_id="o")
    assert pos == 2 and again.n == 5 and again.counters["commit_repeated"] == 1
    # anchors are stored rounded to float16 and read back the same way by every consumer
    a, t = again._anchor(3)
    assert a.dtype == torch.float32 and math.isfinite(t)


def test_legacy_checkpoint_closes_the_retention_and_reproduces_the_old_beliefs(tmp_path, cfg):
    from cam_live.credit.checkpoint import collect_weights, load_weights
    torch.manual_seed(0)
    model = build_model(cfg)
    w = collect_weights(model)
    for g in w.values():
        g.pop("exp.core.persist_rho", None)                       # a checkpoint written before PS
    torch.manual_seed(1)
    other = build_model(cfg)
    with pytest.warns(UserWarning):
        rep = load_weights(other, w)
    assert rep.missing == ["exp.core.persist_rho"]                # reported, kept at its initial value (1)
    store = SqliteExperienceStore(str(tmp_path / "s.sqlite"), durable=False)
    chain = PersistentStateChain(other, store, PersistSettings(enabled=True))
    out = chain.load_state_dict(None)
    assert out["legacy_checkpoint"] and out["rho"] == [0.0] * other.exp.core.n_slots
    s = Stream(other, store, chain)
    _run(s, 5)
    for t, win in enumerate(s.windows):                           # exactly the fresh-state beliefs of the old model
        ref = model.exp.core(model.tokenize([store.items(win)]), None).b
        assert torch.equal(s.live_b[t], ref)


def test_window_key_matches_what_the_tokenizer_sees(setup):
    model, store, chain, s = setup
    refs = [s.put(f"k{i}", float(i)).ref for i in range(3)]
    tok = model.tokenize([store.items(refs + ["never-stored"])])
    n = int(tok.pad_mask.sum())
    assert window_key(tok.refs[0][:n]) == chain._key_of(refs + ["never-stored"]) == window_key(refs)
    mc = model.core_cfg.max_context
    many = [s.put(f"m{i}").ref for i in range(mc + 5)]
    tok = model.tokenize([store.items(many)])
    assert window_key(tok.refs[0][:int(tok.pad_mask.sum())]) == chain._key_of(many)


def test_empty_core_call_keeps_the_carried_state(cfg):
    torch.manual_seed(0)
    model = build_model(cfg)
    st = CoreState(torch.randn(1, model.exp.core.n_slots, model.d_model), torch.tensor([5.0], dtype=torch.float64), 3)
    out = model.exp.core(model.tokenize([[]]), st)
    assert torch.equal(out.state.slots, st.slots) and torch.equal(out.state.last_time, st.last_time)


def test_initial_state_and_reset(setup, cfg):
    """``initial()`` evaluates at the initial chain state (what a weights-only export computes) without changing the
    chain; ``reset()`` starts a new segment: the next decision is made from the fresh state, nothing is recomputed
    across the boundary, earlier records keep their states."""
    model, store, chain, s = setup
    _run(s, 5)
    w = s.windows[4]
    with torch.no_grad():
        carried = s.replay(w)
        with chain.initial():
            at_initial = s.replay(w)
            with chain.live():
                live_initial = model.core_forward(model.tokenize([store.items(s.window())]), None).b
        fresh = model.exp.core(model.tokenize([store.items(w)]), None).b
        fresh_live = model.exp.core(model.tokenize([store.items(s.window())]), None).b
        assert torch.equal(at_initial, fresh) and torch.equal(live_initial, fresh_live)
        assert not torch.allclose(carried, fresh, atol=1e-5)
        assert torch.equal(s.replay(w), carried)                       # nothing changed
        before = [s.replay(x) for x in s.windows]
        boundary = chain.reset("test")
        assert boundary == 5 and chain.live_state().last_time.isnan().all()   # the next decision: fresh state
        _run(s, 3, tag="after")
        for t, x in enumerate(s.windows[:5]):                          # earlier records keep their own states
            assert torch.allclose(s.replay(x), before[t], atol=1e-6)
        st5 = chain.state_for([5])
        assert torch.equal(st5.slots, model.exp.core.init_state(1).slots)    # the boundary position starts fresh
        st6 = chain.state_for([6])            # K = 2 would reach window 4 and anchor 4; the boundary stops it at 5
        chain._refs[4] = ["changed-but-before-the-boundary"]
        assert torch.equal(chain.state_for([6]).slots, st6.slots)
        chain._refs[4] = s.windows[4]
    assert chain.state_dict()["boundaries"] == [5]
