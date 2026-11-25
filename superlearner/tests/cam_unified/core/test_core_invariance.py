"""W02: surface-randomised but dynamically equivalent training pairs, and the invariance objective."""
import math

import pytest
import torch

from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.invariance import (InvarianceLoss, SurfaceRandomizer, UNIT_CONVERSIONS, _fmt,
                                                      invariance_loss_from_views)
from superlearner.cam_unified.interfaces import Action, GoalSpec, Observation, ObsKind, TokenKind, TokenSet
from superlearner.cam_unified.testing import sample_observations, sample_tools, tiny_config


def test_surface_map_preserves_dynamics_of_records():
    obs = sample_observations(7, seed=1) + [
        Observation(ObsKind.NUMERIC, value=12345.0, unit="cents", entity="acct_000017", source="bank"),
        Observation(ObsKind.NUMERIC, value=2000.0, unit="ms", entity="svc_1", source="monitor"),
        Observation(ObsKind.TEXT, "p:nova paid prod_1 invoice; latency 2000 ms on svc_1", entity="p:nova",
                    target="prod_1"),
        Observation(ObsKind.NUMERIC, value=3.0, unit="count", entity="prod_1")]
    tools = sample_tools(5)
    act = Action("set_price", {"product": "prod_1", "price": 19.5}, entities=["prod_1"])
    rnd = SurfaceRandomizer(seed=3)
    sm = rnd.make_map(obs, tools, [act])
    v = sm.apply_observations(obs)
    assert sorted(o.kind.value for o in v) == sorted(o.kind.value for o in obs)
    assert [o.kind for o in v] != [o.kind for o in obs] or sm.order_seed is not None
    by_src = {}
    for o in obs:
        t = sm.apply_observation(o)
        # identities: consistent bijection, prefixes kept, never the original
        if o.entity:
            assert t.entity == sm.entity[o.entity] and t.entity != o.entity
            if o.entity.startswith("p:"):
                assert t.entity.startswith("p:")
        if o.target:
            assert t.target == sm.entity[o.target]
        # units: value and unit string change together (dynamics preserved)
        if o.kind == ObsKind.NUMERIC and o.unit not in (None, "count"):
            nu, f = sm.unit[o.unit]
            assert t.unit == nu and math.isclose(t.value, o.value * f) and t.unit != o.unit
        if o.unit == "count":
            assert t.value == o.value and t.unit == "count"
        by_src[o.content] = t
    assert len(set(sm.entity.values())) == len(sm.entity)
    txt = by_src["p:nova paid prod_1 invoice; latency 2000 ms on svc_1"].content
    assert sm.entity["p:nova"] in txt and sm.entity["prod_1"] in txt and "prod_1" not in txt
    nu, f = sm.unit["ms"]
    assert f"{_fmt(2000 * f)} {nu}" in txt
    code = [o for o in obs if o.kind == ObsKind.CODE][0]
    assert sm.apply_observation(code).content == code.content, "code is never reworded"
    # rewording changes text without touching numbers
    msg = [o for o in obs if o.kind == ObsKind.MESSAGE][0]
    tm = sm.apply_observation(msg)
    assert tm.content != msg.content and "40" in tm.content
    # tools renamed consistently; actions round-trip
    tt = sm.apply_tools(tools)
    assert sorted(t.name for t in tt) == sorted(sm.tool[t.name] for t in tools)
    a2 = sm.apply_action(act)
    assert a2.tool == sm.tool["set_price"] and a2.args["product"] == sm.entity["prod_1"] and a2.args["price"] == 19.5
    back = sm.invert_action(a2)
    assert back.tool == act.tool and back.args == act.args and back.entities == act.entities
    g = sm.apply_goal(GoalSpec("g", "Increase revenue of prod_1 to 10000 cents per month"))
    nu, f = sm.unit["cents"]
    assert sm.entity["prod_1"] in g.text and f"{_fmt(10000 * f)} {nu}" in g.text and "prod_1" not in g.text


def test_unit_table_factors_are_consistent_round_trips():
    for u, convs in UNIT_CONVERSIONS.items():
        for nu, f in convs:
            if nu in UNIT_CONVERSIONS:
                back = dict(UNIT_CONVERSIONS[nu])
                if u in back:
                    assert math.isclose(f * back[u], 1.0, rel_tol=1e-9), (u, nu)
    sm = SurfaceRandomizer(seed=0).make_map([Observation(ObsKind.NUMERIC, value=5.0, unit="cents/core-hour"),
                                             Observation(ObsKind.NUMERIC, value=5.0, unit="widgets")])
    nu, f = sm.unit["cents/core-hour"]
    assert nu.endswith("/core-hour") and nu != "cents/core-hour"
    nu2, f2 = sm.unit["widgets"]
    assert nu2 in ("milli-widgets", "kilo-widgets") and f2 in (1000.0, 1e-3)


def test_infonce_negatives_only_from_different_worlds():
    torch.manual_seed(0)
    lf = InvarianceLoss(16, proj_dim=8, temperature=0.5)
    B, S = 6, 3
    Z1, Z2 = torch.randn(B, S, 16), torch.randn(B, S, 16)
    world = torch.tensor([0, 0, 1, 1, 2, 3])
    loss, st = lf(Z1, Z2, world)
    # manual NT-Xent with same-world (non-positive) candidates removed
    e = torch.cat([lf.embed(Z1), lf.embed(Z2)])
    sim = e @ e.t() / 0.5
    w = torch.cat([world, world])
    tot = 0.0
    for i in range(2 * B):
        p = (i + B) % (2 * B)
        cand = [j for j in range(2 * B) if j == p or w[j] != w[i]]
        tot = tot - (sim[i, p] - torch.logsumexp(sim[i, cand], 0))
    assert torch.allclose(loss, tot / (2 * B), atol=1e-5)
    # experiences of one world are never pushed apart: with a single world every non-positive candidate is
    # excluded, so the loss is exactly zero with zero gradient; with distinct worlds the same batch is penalised
    Zg1, Zg2 = Z1.clone().requires_grad_(), Z2.clone().requires_grad_()
    l_same, _ = lf(Zg1, Zg2, torch.zeros(B, dtype=torch.long))
    l_same.backward()
    assert float(l_same.detach()) == 0.0 and Zg1.grad.abs().max() == 0 and Zg2.grad.abs().max() == 0
    l_diff, _ = lf(Z1, Z2, torch.arange(B))
    assert float(l_diff.detach()) > 0.1 and 0 <= st["top1"] <= 1


def _views(B, n_worlds, mech, cfg, g, nuisance=1.5):
    """Token-level pairs: experience = 10 tokens c_n = [u_n, m_w * u_n] embedded by a fixed matrix; each view adds a
    view-specific surface vector to every token (wording/unit/interface shift), permutes the tokens and relabels the
    entity slots with a random bijection."""
    D = cfg.d_model
    W = torch.randn(8, D, generator=torch.Generator().manual_seed(123)) / math.sqrt(8)
    world = torch.arange(B) % n_worlds
    m = mech[world]                                                              # [B,4]
    u = torch.randn(B, 10, 4, generator=g)
    content = torch.cat([u, m.unsqueeze(1) * u], -1) @ W                         # [B,10,D]
    ent = torch.randint(0, 3, (B, 10), generator=g)
    out = []
    for _ in range(2):
        x = content + nuisance * torch.randn(B, 1, D, generator=g)
        perm = torch.argsort(torch.rand(B, 10, generator=g), dim=1)
        x = torch.gather(x, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        e = torch.gather(ent, 1, perm)
        relabel = torch.stack([torch.randperm(cfg.max_entities, generator=g)[:3] for _ in range(B)])
        e = torch.gather(relabel, 1, e)
        out.append(TokenSet(x, torch.ones(B, 10, dtype=torch.bool), torch.full((B, 10), int(TokenKind.OBS_NUM)),
                            torch.zeros(B, 10), e, None))
    return out[0], out[1], world


def test_invariance_training_matches_views_and_keeps_worlds_apart():
    torch.manual_seed(0)
    cfg = tiny_config()
    core = UnifiedCore(cfg)
    lf = InvarianceLoss(cfg.d_model, proj_dim=32, temperature=0.1)
    mech = torch.randn(64, 4, generator=torch.Generator().manual_seed(9)) * 1.5
    train_worlds, test_worlds = mech[:48], mech[48:]

    def evaluate():
        with torch.no_grad():
            v1, v2, w = _views(16, 16, test_worlds, cfg, torch.Generator().manual_seed(77))
            _, st, _, _ = invariance_loss_from_views(core, lf, v1, v2, w)
        return st
    st0 = evaluate()
    opt = torch.optim.Adam(list(core.parameters()) + list(lf.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(150):
        v1, v2, w = _views(32, 16, train_worlds[torch.randperm(48, generator=g)[:16]], cfg, g)
        loss, st, _, _ = invariance_loss_from_views(core, lf, v1, v2, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    st1 = evaluate()
    print(f"held-out worlds: before {st0}  after {st1}")
    assert st1["top1"] >= 0.9 and st1["top1"] > st0["top1"]
    assert st1["pos_cos"] - st1["neg_cos"] > (st0["pos_cos"] - st0["neg_cos"]) + 0.2
    assert st1["neg_cos"] < 0.6, "different mechanisms are not collapsed together"
