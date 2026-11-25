"""Retriever: q = Q(Z,G,B), MEMORY tokens with event refs, usefulness-trained InfoNCE (delayed cues among thousands of
interfering segments), compression objective, target versions."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch

from memhelpers import light_events
from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.consolidate import Consolidator
from superlearner.cam_unified.memory.index import MemoryIndex
from superlearner.cam_unified.memory.retrieval import Retriever
from superlearner.cam_unified.testing import rand_tokens, tiny_config

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _pipeline(tmp_path, n_envs=3, n=240, seg=8, device="cpu"):
    cfg = tiny_config(mem_segment=seg, mem_segment_tokens=2)
    arc = EventArchive(str(tmp_path / "a"), chunk_rows=128)
    for k in range(n_envs):
        arc.append(light_events(n, env=f"env{k}", episode_len=60, seed=k))
    idx = MemoryIndex(str(tmp_path / "i"), cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model, device=device)
    ret = Retriever(cfg, idx, fanout=4).to(device)
    con = Consolidator(arc, idx, ret)
    con.step()
    con.ensure_current()
    return cfg, arc, idx, ret, con


def test_retrieve_returns_memory_tokens_with_event_refs(tmp_path):
    cfg, arc, idx, ret, con = _pipeline(tmp_path)
    B, k = 3, 4
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    q = ret.query(Z, rand_tokens(B, 2, cfg.d_model), rand_tokens(B, 3, cfg.d_model))
    assert q.shape == (B, cfg.mem_key_dim) and torch.allclose(q.norm(dim=-1), torch.ones(B), atol=1e-4)
    now = [20.0, 25.0, 30.0]
    M, ids, scores = ret.retrieve(q, k, filters=[None, {"env_id": "env2"}, {"level": 2}], now=now)
    T = cfg.mem_segment_tokens
    assert M.x.shape == (B, k * T, cfg.d_model) and (M.kind == int(TokenKind.MEMORY)).all()
    assert M.mask.all() and [len(r) for r in ids] == [k] * B and scores.shape == (B, k)
    assert (scores[:, :-1] >= scores[:, 1:]).all()
    assert all(idx.segment(s)["env_id"] == "env2" for s in ids[1])
    assert all(idx.segment(s)["level"] == 2 for s in ids[2])
    for b in range(B):
        for j, s in enumerate(ids[b]):
            meta = idx.segment(s)
            refs = M.refs[b][j * T]
            assert refs == tuple(idx.source_events(s)) and len(refs) == meta["n_events"]
            assert [r.event_id for r in arc.get(list(refs))] == list(refs)       # exact originals reachable
            assert torch.allclose(M.time[b, j * T:(j + 1) * T], torch.full((T,), meta["t_end"] - now[b]))
    vals = idx.values(idx.current_version, idx.kids_of(idx.current_version, ids[0][:1]))
    assert torch.allclose(M.x[0, :T], vals[0])
    arc.close()


def test_retrieve_is_empty_without_index_or_with_memory_disabled(tmp_path):
    cfg, arc, idx, ret, con = _pipeline(tmp_path, n_envs=1, n=40)
    q = torch.randn(2, cfg.mem_key_dim)
    ret.detach_index()
    M, ids, sc = ret.retrieve(q, 5)
    assert M.N == 0 and ids == [[], []] and sc.shape == (2, 0)
    off = Retriever(tiny_config(use_memory=False, mem_segment=8, mem_segment_tokens=2), idx)
    assert off.retrieve(q, 5)[0].N == 0
    arc.close()


def test_target_versions_and_compatibility():
    cfg = tiny_config(mem_segment=8, mem_segment_tokens=2)
    ret = Retriever(cfg)
    v0 = ret.key_version
    assert ret.compatible_versions() == {v0}
    with torch.no_grad():
        next(ret.segment_encoder.parameters()).add_(1.0)
    v1 = ret.sync_target()
    assert v1 != v0 and v1.startswith("seg-1-") and ret.compatible_versions() == {v0, v1}
    v2 = ret.sync_target()
    assert ret.compatible_versions() == {v1, v2}            # two versions back is obsolete
    sd = ret.state_dict()
    other = Retriever(cfg)
    other.load_state_dict(sd)
    assert other.key_version == v2                           # the version identity travels with the weights
    for p in ret.target_segment_encoder.parameters():
        assert not p.requires_grad


def test_measure_usefulness_identifies_the_helpful_memory():
    D, B = 8, 4
    y = torch.randn(B, D)

    def loss_fn(M):                                          # downstream: predict y from the memory tokens
        if M is None or M.N == 0:
            pred = torch.zeros(B, D)
        else:
            w = M.mask.float().unsqueeze(-1)
            pred = (M.x * w).sum(1) / w.sum(1).clamp(min=1)
        return ((pred - y) ** 2).sum(-1)
    cands = [TokenSet.from_dense(torch.randn(B, 2, D) * 0.1, TokenKind.MEMORY) for _ in range(5)]
    cands[3] = TokenSet.from_dense(y.unsqueeze(1).expand(B, 2, D).clone(), TokenKind.MEMORY)
    u = Retriever.measure_usefulness(loss_fn, cands)
    assert u.shape == (B, 5) and (u.argmax(1) == 3).all() and (u[:, 3] > 0).all()


def test_usefulness_loss_handles_rows_without_contrast():
    cfg = tiny_config()
    ret = Retriever(cfg)
    q = torch.nn.functional.normalize(torch.randn(3, cfg.mem_key_dim), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(3, 5, cfg.mem_key_dim), dim=-1)
    u = torch.tensor([[1., 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    loss, info = ret.usefulness_loss(q, k, u)
    assert info["rows"] == 1                                  # no positive / no negative rows are ignored
    loss.backward()
    assert q.grad[0].abs().sum() > 0 and q.grad[1:].abs().sum() == 0


# ------------------------------------------------------------------ learned delayed-cue retrieval
class _CueWorld:
    """Event-token segments (L events of width D); a cue segment holds one token carrying a code; queries carry the
    code in one of the latent state tokens Z. Everything else is interference."""

    def __init__(self, D, L, S, n_codes, device, seed=0):
        self.D, self.L, self.S, self.dev = D, L, S, device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.E = torch.randn(n_codes, D, generator=self.g, device=device)

    def segments(self, codes: torch.Tensor):
        n = len(codes)
        x = torch.randn(n, self.L, self.D, generator=self.g, device=self.dev)
        pos = torch.randint(0, self.L, (n,), generator=self.g, device=self.dev)
        has = codes >= 0
        r = torch.arange(n, device=self.dev)[has]
        x[r, pos[has]] = 2.0 * self.E[codes[has]] + 0.3 * torch.randn(int(has.sum()), self.D, generator=self.g,
                                                                      device=self.dev)
        mask = torch.ones(n, self.L, dtype=torch.bool, device=self.dev)
        dt = -torch.arange(self.L - 1, -1, -1, device=self.dev).float().expand(n, self.L) / 24
        return x, mask, dt

    def queries(self, codes: torch.Tensor):
        B = len(codes)
        Z = torch.randn(B, self.S, self.D, generator=self.g, device=self.dev)
        p = torch.randint(0, self.S, (B,), generator=self.g, device=self.dev)
        Z[torch.arange(B, device=self.dev), p] = 2.0 * self.E[codes] + 0.3 * torch.randn(B, self.D, generator=self.g,
                                                                                          device=self.dev)
        G = TokenSet.from_dense(torch.randn(B, 2, self.D, generator=self.g, device=self.dev), TokenKind.GOAL)
        Bt = TokenSet.from_dense(torch.randn(B, 4, self.D, generator=self.g, device=self.dev), TokenKind.BELIEF)
        return Z, G, Bt


def _eval_delayed(ret, world, cfg, tmp, n_seg, n_codes, k=5):
    """Store n_seg segments (one cue per code, spread over the stream) with the target encoder, then query every
    code after the whole stream: returns hit@k per cue and its delay (segments stored after the cue)."""
    idx = MemoryIndex(tmp, cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model, device=DEV)
    ret.attach_index(idx)
    v = ret.key_version
    codes = torch.full((n_seg,), -1, dtype=torch.long)
    slots = np.linspace(0, n_seg - 1, n_codes).astype(int)
    codes[slots] = torch.arange(n_codes)
    for i in range(0, n_seg, 512):
        x, m, dt = world.segments(codes[i:i + 512].to(DEV))
        mem, key = ret.encode_segments(x, m, dt, target=True)
        ids = [f"s{j:06d}" for j in range(i, i + len(x))]
        idx.add_segments([{"seg_id": s, "level": 1, "t_start": float(j), "t_end": float(j)}
                          for j, s in zip(range(i, i + len(x)), ids)])
        idx.add(v, ids, key, mem)
    idx.set_current(v)
    Z, G, Bt = world.queries(torch.arange(n_codes, device=DEV))
    with torch.no_grad():
        q = ret.query(Z, G, Bt)
    t0 = time.time()
    _M, ids, _ = ret.retrieve(q, k)
    lat = time.time() - t0
    hits = np.array([f"s{slots[c]:06d}" in ids[c] for c in range(n_codes)])
    idx.close()
    return hits, (n_seg - 1 - slots), lat


def test_usefulness_trained_retrieval_finds_delayed_cues(tmp_path):
    torch.manual_seed(0)
    cfg = tiny_config(mem_segment=8, mem_segment_tokens=2)
    n_codes, n_seg = 48, 3000                         # 24k interfering events
    world = _CueWorld(cfg.d_model, 8, cfg.n_state, n_codes, DEV, seed=1)
    ret = Retriever(cfg).to(DEV)
    fresh_hits, _, _ = _eval_delayed(ret, world, cfg, str(tmp_path / "fresh"), n_seg, n_codes)

    def answer_loss(codes):
        """Downstream loss of each query row given memory M: how far the code's embedding is from the best-matching
        memory content (1 - max cosine); without memory it is 1."""
        e = torch.nn.functional.normalize(world.E[codes], dim=-1)

        def fn(M):
            if M is None or M.N == 0:
                return torch.ones(len(codes), device=DEV)
            c = torch.einsum("bd,bnd->bn", e, torch.nn.functional.normalize(M.x, dim=-1))
            return 1 - c.masked_fill(~M.mask, -1).max(1).values
        return fn
    opt = torch.optim.Adam([p for p in ret.parameters() if p.requires_grad], lr=1e-3)
    B, t0, accs = 32, time.time(), []
    for step in range(400):
        codes = torch.randperm(n_codes, device=DEV, generator=world.g)[:B]
        x, m, dt = world.segments(torch.cat([codes, torch.full((B,), -1, device=DEV)]))
        # usefulness of each candidate's raw events for each query, measured through the downstream loss
        cands = [TokenSet.from_dense(x[j:j + 1].expand(B, -1, -1), TokenKind.EVENT) for j in range(2 * B)]
        util = Retriever.measure_usefulness(answer_loss(codes), cands)
        mem, key = ret.encode_segments(x, m, dt)
        Z, G, Bt = world.queries(codes)
        loss, info = ret.usefulness_loss(ret.query(Z, G, Bt), key, util, margin=0.5)
        assert info["rows"] == B and (util[torch.arange(B), torch.arange(B)] > 0.5).all()   # the cue always helps
        assert info["pos_frac"] < 1.5 / (2 * B)               # interference rarely passes the usefulness margin
        opt.zero_grad()
        loss.backward()
        if step == 0:
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ret.query_net.parameters())
            assert ret.segment_encoder.key_head[-1].weight.grad.abs().sum() > 0
        opt.step()
        accs.append(info["acc"])
    train_s = time.time() - t0
    ret.sync_target()
    hits, delay, lat = _eval_delayed(ret, world, cfg, str(tmp_path / "trained"), n_seg, n_codes)
    bins = {f"<{b}": float(hits[delay < b].mean()) for b in (100, 1000, 3000) if (delay < b).any()}
    print(f"\ndelayed-cue hit@5 among {n_seg} segments: fresh {fresh_hits.mean():.3f}, trained {hits.mean():.3f}; "
          f"by delay (segments after the cue): {bins}; train {train_s:.1f}s for 400 steps "
          f"(final batch acc {np.mean(accs[-20:]):.2f}); retrieval {lat * 1e3:.1f} ms for {n_codes} queries")
    assert fresh_hits.mean() <= 0.15
    assert hits.mean() >= 0.85
    assert hits[delay >= 1000].mean() >= 0.8                 # long delays are not forgotten


def test_compression_loss_trains_memory_tokens_to_keep_event_information():
    torch.manual_seed(0)
    cfg = tiny_config(mem_segment=8, mem_segment_tokens=4)
    ret = Retriever(cfg).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(3)

    def batch():
        x = torch.randn(32, 8, cfg.d_model, generator=g, device=DEV)
        m = torch.ones(32, 8, dtype=torch.bool, device=DEV)
        m[:4, 6:] = False                                       # ragged segments
        return x, m, torch.zeros(32, 8, device=DEV)

    def ident_acc():
        x, m, dt = batch()
        with torch.no_grad():
            mem, _ = ret.segment_encoder(x, m, dt)
            p = torch.nn.functional.normalize(ret.segment_encoder.reconstruct(mem, 8, dt)[m], dim=-1)
            t = torch.nn.functional.normalize(x[m], dim=-1)
            return float(((p @ t.T).argmax(1) == torch.arange(len(p), device=DEV)).float().mean())
    before = ident_acc()
    opt = torch.optim.Adam(ret.segment_encoder.parameters(), lr=2e-3)
    losses = []
    for _ in range(250):
        x, m, dt = batch()
        loss = ret.compression_loss(x, m, dt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    after = ident_acc()
    print(f"\nevent identification from memory tokens: {before:.3f} -> {after:.3f} "
          f"(chance {1 / 252:.4f}); loss {losses[0]:.2f} -> {np.mean(losses[-10:]):.2f}")
    assert after > 0.3 and after > 10 * max(before, 1 / 252) and np.mean(losses[-10:]) < losses[0] - 1.0


def test_memory_tokens_enter_the_core_context(tmp_path):
    """Contract check: M_t concatenates with other context TokenSets (refs preserved) and changes Z_t = F(Z, ctx).
    (The core here is the shared reference double; the real backbone is workstream B's.)"""
    from superlearner.cam_unified.testing import DoubleCore
    cfg, arc, idx, ret, con = _pipeline(tmp_path)
    torch.manual_seed(0)
    core = DoubleCore(cfg).eval()
    Z = core.init_state(2) + 0.1 * torch.randn(2, cfg.n_state, cfg.d_model)
    obs = rand_tokens(2, 5, cfg.d_model, seed=1)
    with torch.no_grad():
        M, ids, _ = ret.retrieve(ret.query(Z, None, None), 4, now=[10.0, 10.0])
        ctx = TokenSet.cat(obs, M)
        Z_mem, _ = core(Z, ctx)
        Z_nomem, _ = core(Z, obs)
    assert ctx.N == 5 + M.N and ctx.refs[0][5] == M.refs[0][0] and (ctx.kind[:, 5:] == int(TokenKind.MEMORY)).all()
    assert not torch.allclose(Z_mem, Z_nomem, atol=1e-4)
    arc.close()
