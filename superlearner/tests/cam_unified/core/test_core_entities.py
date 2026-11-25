"""Generalised entities and relations (A04): variable entity counts, relation tokens, binding by slot equality."""
import random

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.entities import EntityEncoder, entity_type_tokens
from superlearner.cam_unified.interfaces import EntityRegistry, TokenKind, TokenSet
from superlearner.cam_unified.testing import tiny_config


def _obs(rows, D, seed=0):
    """rows: list of (list of (slot, n_tokens), list of (i, j, n_tokens)). Returns TokenSet with OBS tokens for
    entities and RELATION tokens for pairs (entity2 set), padded."""
    g = torch.Generator().manual_seed(seed)
    toks = []
    for ents, rels in rows:
        r = []
        for s, n in ents:
            for _ in range(n):
                r.append((torch.randn(D, generator=g), int(TokenKind.OBS_NUM), s, -1, -float(torch.rand(1, generator=g)) * 3))
        for i, j, n in rels:
            for _ in range(n):
                r.append((torch.randn(D, generator=g), int(TokenKind.RELATION), i, j, -float(torch.rand(1, generator=g)) * 3))
        toks.append(r)
    B, N = len(rows), max(1, max(len(r) for r in toks))
    x = torch.zeros(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    kind = torch.zeros(B, N, dtype=torch.long)
    time = torch.zeros(B, N)
    ent = torch.full((B, N), -1, dtype=torch.long)
    ent2 = torch.full((B, N), -1, dtype=torch.long)
    for b, r in enumerate(toks):
        for n, (v, k, s, s2, t) in enumerate(r):
            x[b, n], kind[b, n], ent[b, n], ent2[b, n], time[b, n], mask[b, n] = v, k, s, s2, t, True
    return TokenSet(x, mask, kind, time, ent, [[f"o{b}_{n}" for n in range(N)] for b in range(B)], ent2)


def _perm(ts, perm):
    return TokenSet(ts.x[:, perm], ts.mask[:, perm], ts.kind[:, perm], ts.time[:, perm], ts.entity[:, perm], None,
                    ts.entity2[:, perm])


ROWS = [
    ([(3, 2), (7, 1)], [(3, 7, 2)]),
    ([(0, 1), (1, 3), (2, 2), (5, 1), (9, 2)], [(1, 2, 1), (2, 1, 1), (9, 0, 2), (5, 5, 1)]),
    ([], []),
    ([(4, 1)], [(4, 11, 1)]),           # slot 11 appears only as a relation target
]


def test_variable_entity_counts_and_relation_tokens():
    cfg = tiny_config()
    enc = EntityEncoder(cfg).eval()
    obs = _obs(ROWS, cfg.d_model)
    out = enc(obs)
    n_ent, n_rel = enc.last_counts
    assert n_ent.tolist() == [2, 5, 0, 2] and n_rel.tolist() == [1, 4, 0, 1]
    Eo = int(n_ent.max())
    assert out.N == Eo + int(n_rel.max())
    for b, (ents, rels) in enumerate(ROWS):
        want = sorted({s for s, _ in ents} | {j for _, j, _ in rels})
        got = out.entity[b, :Eo][out.mask[b, :Eo]].tolist()
        assert got == want, (b, got, want)
        assert (out.kind[b, :Eo][out.mask[b, :Eo]] == TokenKind.ENTITY).all()
        pairs = sorted(set((i, j) for i, j, _ in rels))
        rm = out.mask[b, Eo:]
        got_pairs = sorted(zip(out.entity[b, Eo:][rm].tolist(), out.entity2[b, Eo:][rm].tolist()))
        assert got_pairs == pairs
        assert (out.kind[b, Eo:][rm] == TokenKind.RELATION).all()
    assert not out.mask[2].any() and torch.all(out.x[2] == 0)
    assert torch.isfinite(out.x).all()
    assert out.refs[0][0] == "entity:3" and out.refs[0][Eo] == "relation:3->7"
    # entity tokens carry the time of their latest observation
    b, s = 0, 3
    sel = (obs.entity[b] == s) | (obs.entity2[b] == s)
    assert torch.isclose(out.time[b, 0], obs.time[b][sel & obs.mask[b]].max())


def test_token_order_invariance():
    cfg = tiny_config()
    enc = EntityEncoder(cfg).eval()
    obs = _obs(ROWS[:2], cfg.d_model, seed=1)
    out = enc(obs)
    perm = torch.randperm(obs.N, generator=torch.Generator().manual_seed(3))
    out2 = enc(_perm(obs, perm))
    assert torch.allclose(out.x, out2.x, atol=1e-5)
    assert torch.equal(out.entity, out2.entity) and torch.equal(out.entity2, out2.entity2)


def test_slot_relabeling_permutes_outputs_identity_is_binding_only():
    cfg = tiny_config()
    enc = EntityEncoder(cfg).eval()
    obs = _obs(ROWS[:2], cfg.d_model, seed=2)
    out = enc(obs)
    pi = torch.randperm(cfg.max_entities, generator=torch.Generator().manual_seed(4))
    relab = TokenSet(obs.x, obs.mask, obs.kind, obs.time, torch.where(obs.entity >= 0, pi[obs.entity.clamp(min=0)],
                                                                       obs.entity), None,
                     torch.where(obs.entity2 >= 0, pi[obs.entity2.clamp(min=0)], obs.entity2))
    out2 = enc(relab)
    for b in range(2):
        for n in range(out.N):
            if not out.mask[b, n]:
                continue
            key = (int(pi[out.entity[b, n]]), int(pi[out.entity2[b, n]]) if out.entity2[b, n] >= 0 else -1)
            m = [(int(out2.entity[b, k]), int(out2.entity2[b, k])) == key and bool(out2.mask[b, k])
                 for k in range(out2.N)].index(True)
            assert torch.allclose(out.x[b, n], out2.x[b, m], atol=1e-5)


def test_entity_depends_on_its_relations_and_neighbours():
    cfg = tiny_config()
    enc = EntityEncoder(cfg, n_rounds=2).eval()
    obs = _obs([([(1, 2), (2, 2), (6, 2)], [(1, 2, 1)])], cfg.d_model, seed=5)
    out = enc(obs)
    obs2 = TokenSet(obs.x.clone(), obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2)
    sel = (obs.entity[0] == 2) & (obs.kind[0] == TokenKind.OBS_NUM)
    obs2.x[0, sel] += 2.0                                    # change entity 2's own observations
    out2 = enc(obs2)
    pos = {int(s): k for k, s in enumerate(out.entity[0, :3].tolist())}
    d = (out.x - out2.x).norm(dim=-1)[0]
    assert d[pos[2]] > 1e-2                                   # its own token changes
    assert d[pos[1]] > 1e-3                                   # its relation partner changes
    enc0 = EntityEncoder(cfg, n_rounds=0).eval()
    a, b2 = enc0(obs), enc0(obs2)
    d0 = (a.x - b2.x).norm(dim=-1)[0]
    assert d0[pos[1]] < 1e-5 and d0[pos[6]] < 1e-5, "without relational rounds an entity sees only its own tokens"


def test_type_tokens_from_registry_condition_entities():
    cfg = tiny_config()
    enc = EntityEncoder(cfg, type_dim=16).eval()
    regs = [EntityRegistry(cfg.max_entities, seed=0), EntityRegistry(cfg.max_entities, seed=1)]
    s_a = regs[0].bind("prod_1", 0.0, "product")
    s_b = regs[0].bind("cust_7", 0.0, "customer group")
    s_c = regs[1].bind("repo_1", 0.0, "software repository")
    calls = []

    def embed(texts):
        calls.append(list(texts))
        g = torch.Generator().manual_seed(len(texts))
        return torch.randn(len(texts), 16, generator=g)
    cache = {}
    tt = entity_type_tokens(regs, embed, cache=cache)
    assert tt.x.shape == (2, 2, 16) and tt.mask.tolist() == [[True, True], [True, False]]
    assert set(tt.entity[0].tolist()) == {s_a, s_b} and int(tt.entity[1, 0]) == s_c
    entity_type_tokens(regs, embed, cache=cache)
    assert len(calls) == 1, "type embeddings are cached by type string"
    obs = _obs([([(s_a, 1), (s_b, 1)], []), ([(s_c, 2)], [])], cfg.d_model, seed=6)
    with_t = enc(obs, tt)
    without = enc(obs)
    assert (with_t.x - without.x)[with_t.mask].abs().max() > 1e-3


def test_relation_working_budget_keeps_most_recent_pairs_and_counts_drops():
    cfg = tiny_config()
    enc = EntityEncoder(cfg, max_relations=2).eval()
    obs = _obs([([(0, 1), (1, 1), (2, 1)], [(0, 1, 1), (1, 2, 1), (2, 0, 1)])], cfg.d_model, seed=7)
    rel = obs.kind[0] == TokenKind.RELATION
    obs.time[0, rel] = torch.tensor([-5.0, -1.0, -2.0])
    out = enc(obs)
    Eo = 3
    pairs = sorted(zip(out.entity[0, Eo:].tolist(), out.entity2[0, Eo:].tolist()))
    assert pairs == [(1, 2), (2, 0)] and enc.stats["dropped_relations"] == 1


def test_entity_tokens_feed_the_core_and_train_end_to_end():
    cfg = tiny_config()
    enc = EntityEncoder(cfg)
    core = UnifiedCore(cfg)
    obs = _obs(ROWS, cfg.d_model, seed=8)
    types = TokenSet(torch.randn(obs.B, 1, cfg.d_model), torch.ones(obs.B, 1, dtype=torch.bool),
                     torch.full((obs.B, 1), int(TokenKind.ENTITY)), torch.zeros(obs.B, 1),
                     torch.tensor([[3], [1], [0], [4]]), None)
    toks = enc(obs, types)
    ctx = TokenSet.cat(obs, toks)
    Z, co = core(core.init_state(obs.B), ctx)
    Z.pow(2).mean().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not missing, missing


def _relational_task(B, cfg, rng, D):
    """Each row: 2-5 entities with 1-3 numeric tokens (value v along a fixed direction); random directed relations
    i->j. Target of entity i: mean own value + sum over related j of mean value of j."""
    gen = torch.Generator().manual_seed(rng.randrange(10 ** 9))
    dirv = torch.ones(D) / D ** 0.5
    rows, targets = [], []
    for _ in range(B):
        n = rng.randint(2, 5)
        slots = rng.sample(range(cfg.max_entities), n)
        vals = {s: [rng.uniform(-2, 2) for _ in range(rng.randint(1, 3))] for s in slots}
        rels = [(i, j) for i in slots for j in slots if i != j and rng.random() < 0.3]
        rows.append((slots, vals, rels))
        targets.append({i: sum(vals[i]) / len(vals[i]) + sum(sum(vals[j]) / len(vals[j]) for a, j in rels if a == i)
                        for i in slots})
    N = max(sum(len(v) for v in r[1].values()) + len(r[2]) for r in rows)
    x = torch.zeros(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    kind = torch.zeros(B, N, dtype=torch.long)
    ent = torch.full((B, N), -1, dtype=torch.long)
    ent2 = torch.full((B, N), -1, dtype=torch.long)
    for b, (slots, vals, rels) in enumerate(rows):
        n = 0
        for s in slots:
            for v in vals[s]:
                x[b, n] = v * dirv * D ** 0.5 + 0.1 * torch.randn(D, generator=gen)
                kind[b, n], ent[b, n], mask[b, n] = int(TokenKind.OBS_NUM), s, True
                n += 1
        for i, j in rels:
            x[b, n] = torch.randn(D, generator=gen) * 0.1
            kind[b, n], ent[b, n], ent2[b, n], mask[b, n] = int(TokenKind.RELATION), i, j, True
            n += 1
    return TokenSet(x, mask, kind, torch.zeros(B, N), ent, None, ent2), targets


def _train_relational(n_rounds, steps, seed=0):
    torch.manual_seed(seed)
    cfg = tiny_config()
    enc = EntityEncoder(cfg, n_rounds=n_rounds)
    head = nn.Linear(cfg.d_model, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=2e-3)
    rng = random.Random(seed)

    def batch_loss(B):
        obs, targets = _relational_task(B, cfg, rng, cfg.d_model)
        out = enc(obs)
        pred = head(out.x).squeeze(-1)
        ys, ps = [], []
        for b, t in enumerate(targets):
            for k in range(out.N):
                if out.mask[b, k] and out.kind[b, k] == TokenKind.ENTITY:
                    ys.append(t[int(out.entity[b, k])])
                    ps.append(pred[b, k])
        y = torch.tensor(ys)
        p = torch.stack(ps)
        return ((p - y) ** 2).mean(), y.var()
    for _ in range(steps):
        loss, _ = batch_loss(32)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse, var = zip(*[batch_loss(64) for _ in range(4)])
    return float(sum(mse) / 4), float(sum(var) / 4)


@pytest.mark.slow
def test_relational_rounds_learn_relation_dependent_targets():
    """Relationships_i matter: a target that sums related entities' attributes is learned with the default single
    relational attention round and not without it (same budget)."""
    mse1, var = _train_relational(1, 600)
    mse0, _ = _train_relational(0, 600)
    print(f"relational task: var {var:.3f}  mse rounds=1 {mse1:.3f}  rounds=0 {mse0:.3f}")
    assert mse1 < 0.35 * var
    assert mse1 < 0.6 * mse0
