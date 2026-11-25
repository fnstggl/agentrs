"""W02: surface-randomised but dynamically equivalent training pairs, and the invariance objective."""
import hashlib
import json
import math
import random
import re

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.core.backbone import UnifiedCore

from superlearner.cam_unified.core.invariance import (PRESENTATIONS, SYNONYMS, InvarianceLoss,
                                                      PredictionEquivarianceLoss, SurfaceRandomizer, UNIT_CONVERSIONS,
                                                      _flatten, _fmt, invariance_loss_from_views, is_identifier_chunk,
                                                      render_json, unit_factors)
from superlearner.cam_unified.interfaces import (Action, EntityRegistry, GoalSpec, Observation, ObsKind, TokenKind,
                                                 TokenSet)
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
    for _ in range(300):
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


# ---------------------------------------------------------------------------- identifier safety of the surface views

_JSON = ('{"price": 1999, "build_ok": true, "note": "customers complain the price is too high", "product": "prod_1", '
         '"url": "https://shop.example/search?product=prod_1", "items": [{"product": "prod_1", "price": 5}, '
         '{"product": "prod_2", "price": 7}]}')
_PATH = 'File "/app/service/build/main.py", line 3: error in build of prod_1'
_URL_TEXT = "See https://shop.example/search?product=prod_1 for product details; customers search by price."
_MSG = "Can you deliver 40 units by Friday? Use the 'price' field, the build_ok flag and set_price.v2 at 10:30."
_GOAL = "Increase revenue by 20% by day 30"


def _records():
    obs = [Observation(ObsKind.TOOL_RESULT, _JSON, source="get_price", entity="prod_1"),
           Observation(ObsKind.ERROR, _PATH, entity="svc_1"),
           Observation(ObsKind.EXEC_RESULT, "build failed: upstream error in /srv/app/main.py", entity="svc_1"),
           Observation(ObsKind.TEXT, _URL_TEXT, entity="prod_1"),
           Observation(ObsKind.MESSAGE, _MSG, entity="cust_7"),
           Observation(ObsKind.HTML, '<a href="/search?product=prod_2">search products</a>', entity="prod_2")]
    act = Action("browse", {"url": "https://shop.example/search?product=prod_1", "path": "/app/service/build/main.py",
                            "product": "prod_2"}, entities=["prod_1"])
    return obs, act


def test_identifier_chunks_are_recognised():
    for c in ["build_ok", "/app/main.py", "main.py", "https://x.y/z?a=b", "p:nova", "prod1", "v2", "getUser",
              "TimeoutError", "--force", "'price'", '"build_ok":', "a=b", "{x}", "[0]", "$5"]:
        assert is_identifier_chunk(c), c
    for c in ["customers", "Customers,", "price.", "(search)", "don't", "well-known", "40", "12.5", "20%", "Friday?"]:
        assert not is_identifier_chunk(c), c


@pytest.mark.parametrize("seed", range(6))
def test_rewording_never_touches_identifiers_paths_urls_or_json_keys(seed):
    obs, act = _records()
    sm = SurfaceRandomizer(seed=seed, reword_prob=1.0, presentation=False, permute=False).make_map(obs, [], [act])
    out = {o.kind: sm.apply_observation(o) for o in obs}
    e = sm.entity
    # JSON tool result: same structure and keys, scalars and identifier values exact (ids renamed), prose reworded
    src, dst = json.loads(_JSON), json.loads(out[ObsKind.TOOL_RESULT].content)
    assert list(dst) == list(src) and [list(r) for r in dst["items"]] == [list(r) for r in src["items"]]
    assert dst["price"] == 1999 and dst["build_ok"] is True and dst["product"] == e["prod_1"]
    assert dst["url"] == f"https://shop.example/search?product={e['prod_1']}"
    assert dst["items"][1] == {"product": e["prod_2"], "price": 7}
    assert dst["note"] != src["note"] and "too high" not in dst["note"]
    # machine output: never reworded, paths verbatim
    assert out[ObsKind.ERROR].content == _PATH.replace("prod_1", e["prod_1"])
    assert out[ObsKind.EXEC_RESULT].content == "build failed: upstream error in /srv/app/main.py"
    # prose: URL stays copyable — the surface action carries exactly the URL the surface view shows
    a2 = sm.apply_action(act)
    surf_url = re.search(r"https://\S+", out[ObsKind.TEXT].content).group(0)
    assert surf_url == a2.args["url"] and a2.args["path"] == act.args["path"] and a2.args["product"] == e["prod_2"]
    assert sm.invert_action(a2).args == act.args and sm.invert_action(a2).tool == act.tool
    assert out[ObsKind.HTML].content == f'<a href="/search?product={e["prod_2"]}">search products</a>'
    msg = out[ObsKind.MESSAGE].content
    for ident in ("'price'", "build_ok", "set_price.v2", "10:30", "40", "Friday"):
        assert ident in msg, (ident, msg)
    assert msg != _MSG
    # goal semantics: numbers and the meaning-bearing "by" kept; only listed synonyms substituted
    g = sm.apply_goal(GoalSpec("g", _GOAL)).text
    assert re.findall(r"\d+%?", g) == re.findall(r"\d+%?", _GOAL) and g.count(" by ") == 2
    allowed = {w for alts in SYNONYMS.values() for a in alts for w in a.lower().split()}
    for o, t in ((_GOAL, g), (_URL_TEXT, out[ObsKind.TEXT].content), (_MSG, msg)):
        new_words = set(re.findall(r"[a-z]+", t.lower())) - set(re.findall(r"[a-z]+", o.lower()))
        new_words -= {w for v in e.values() for w in re.findall(r"[a-z]+", v)}
        assert new_words <= allowed, new_words


def test_synonym_table_excludes_meaning_changing_substitutions():
    for w in ("by", "did", "can", "long", "your", "set", "return", "cost", "account", "program", "you", "of"):
        assert w not in SYNONYMS, w
    for w, alts in SYNONYMS.items():
        assert w not in alts and all(a.strip() for a in alts), w


@pytest.mark.parametrize("style", PRESENTATIONS)
def test_presentation_rerenders_json_with_identical_content(style):
    obj = {"status": "ok", "total": 12, "items": [{"sku": "a_1", "qty": 2, "price": 5.5},
                                                   {"sku": "b_2", "qty": 1, "price": 7.0}]}
    txt = render_json(obj, style)
    if style.startswith("json"):
        assert json.loads(txt) == obj
    elif style == "kv_lines":
        assert sorted(txt.splitlines()) == sorted(f"{p}: {v}" for p, v in _flatten(obj))
    else:
        lines = txt.splitlines()
        assert "status: ok" in lines and "total: 12" in lines and "| sku | qty | price |" in lines
        assert "| a_1 | 2 | 5.5 |" in lines and "| b_2 | 1 | 7.0 |" in lines
    styles = {render_json(obj, st) for st in PRESENTATIONS}
    assert len(styles) == len(PRESENTATIONS), "every presentation is a different surface"
    # through a SurfaceMap: tool results change presentation, content is preserved
    sm = SurfaceRandomizer(seed=0, rename_entities=False, reword=False).make_map()
    sm.presentation = style
    o = Observation(ObsKind.TOOL_RESULT, json.dumps(obj))
    assert sm.apply_observation(o).content == txt


# ---------------------------------------------------------------------------- prediction equivariance

def test_prediction_equivariance_loss_is_zero_iff_predictions_transform_with_the_units():
    lf = PredictionEquivarianceLoss(robust=False)
    m1, v1 = torch.tensor([[1234.0, 5.0]]), torch.tensor([[100.0, 4.0]])
    f = torch.tensor([[0.01, 1.0]])                                  # cents -> dollars ; unit-free
    assert float(lf.gaussian(m1, v1, m1 * f, v1 * f * f, f)) < 1e-9
    wrong = lf.gaussian(m1, v1, m1, v1, f)                              # not rescaled: penalised
    ref = 0.0
    for q in range(2):
        a = torch.distributions.Normal(m1[0, q] * f[0, q], (v1[0, q] * f[0, q] ** 2).sqrt())
        b = torch.distributions.Normal(m1[0, q], v1[0, q].sqrt())
        ref += float(torch.distributions.kl_divergence(a, b) + torch.distributions.kl_divergence(b, a))
    assert math.isclose(float(wrong), ref / 2, rel_tol=1e-5)
    robust = PredictionEquivarianceLoss()
    per_q = [float(torch.distributions.kl_divergence(
        torch.distributions.Normal(m1[0, q] * f[0, q], (v1[0, q] * f[0, q] ** 2).sqrt()),
        torch.distributions.Normal(m1[0, q], v1[0, q].sqrt())) + torch.distributions.kl_divergence(
        torch.distributions.Normal(m1[0, q], v1[0, q].sqrt()),
        torch.distributions.Normal(m1[0, q] * f[0, q], (v1[0, q] * f[0, q] ** 2).sqrt()))) for q in range(2)]
    assert math.isclose(float(robust.gaussian(m1, v1, m1, v1, f)), sum(math.log1p(k) for k in per_q) / 2,
                        rel_tol=1e-5)
    assert float(robust.gaussian(m1, v1, m1 * f, v1 * f * f, f)) < 1e-9
    # scale-free: the same relative error costs the same in cents and in dollars
    e1 = lf.gaussian(m1[:, :1], v1[:, :1], 1.1 * m1[:, :1] * 0.01, v1[:, :1] * 1e-4, 0.01)
    e2 = lf.gaussian(m1[:, :1] * 0.01, v1[:, :1] * 1e-4, 1.1 * m1[:, :1] * 0.01, v1[:, :1] * 1e-4, 1.0)
    assert torch.allclose(e1, e2, rtol=1e-4)
    lg = torch.randn(3, 2, 5)
    assert float(lf.categorical(lg, lg)) < 1e-7 and float(lf.categorical(lg, lg.flip(-1))) > 1e-3
    assert float(lf.bernoulli(lg[..., 0], lg[..., 0])) < 1e-7
    sm = SurfaceRandomizer(seed=0).make_map([Observation(ObsKind.NUMERIC, value=5.0, unit="cents")])
    fac = unit_factors(sm, [["cents", "count", None]])
    assert math.isclose(float(fac[0, 0]), sm.unit["cents"][1]) and fac[0, 1:].tolist() == [1.0, 1.0]


def test_prediction_equivariance_transfers_supervision_across_units():
    """A predictor of y = 3 x (same unit as x) is supervised only on cents inputs; the equivariance term on unlabeled
    paired views (cents vs dollars, f = 0.01) makes its dollars predictions correct — without it they are not."""
    def symlog(v):
        return torch.sign(v) * torch.log1p(v.abs())

    def feats(v, unit):                                               # value + unit string features
        sl = symlog(v)
        fr = torch.arange(1, 9).float() * 0.3
        oh = torch.zeros(v.shape[0], 2)
        oh[:, unit] = 1.0
        return torch.cat([torch.sin(sl[:, None] * fr), torch.cos(sl[:, None] * fr), sl[:, None] / 5, oh], -1)

    def predict(net, v, unit):
        """Predictive in symlog space (mu_s, log sd_s) and its raw-space moments (delta method)."""
        o = net(feats(v, unit))
        mu_s, log_sd = o[:, 0], o[:, 1].clamp(-8, 3)
        mean = torch.sign(mu_s) * torch.expm1(mu_s.abs())
        sd = torch.exp(log_sd) * (mean.abs() + 1)
        return mean, sd ** 2, mu_s, log_sd

    lf = PredictionEquivarianceLoss()
    res = {}
    for use_eq in (False, True):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(19, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=1e-4)
        g = torch.Generator().manual_seed(1)
        for _ in range(1200):
            x = torch.exp(torch.rand(128, generator=g) * 4 + 5)        # 150 .. 8100 cents
            y = 3 * x * (1 + 0.02 * torch.randn(128, generator=g))
            m, v, mu_s, lsd = predict(net, x, 0)
            loss = (lsd + 0.5 * (symlog(y) - mu_s) ** 2 * torch.exp(-2 * lsd)).mean()
            if use_eq:
                m2, v2, _, _ = predict(net, x * 0.01, 1)
                loss = loss + lf.gaussian(m, v, m2, v2, 0.01)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            xc = torch.exp(torch.rand(512, generator=torch.Generator().manual_seed(9)) * 4 + 5)
            mc = predict(net, xc, 0)[0]
            md = predict(net, xc * 0.01, 1)[0]
            res[use_eq] = (float(((mc - 3 * xc).abs() / (3 * xc)).median()),
                           float(((md - 0.03 * xc).abs() / (0.03 * xc)).median()))
    print(f"relative error (cents inputs, dollars inputs): without equivariance {res[False]}, with {res[True]}")
    assert res[False][0] < 0.05 and res[True][0] < 0.05, "cents supervision is fitted either way"
    assert res[True][1] < 0.05 and res[False][1] > 0.3, res



# ---------------------------------------------------------------------------- end to end from records

_TEMPLATES = ["customers of {p} complain the price is too high", "customers say onboarding for {p} takes too long",
              "the upstream service {s} did not respond to the request",
              "workers ask to deliver more units of {p} quickly"]
_WVEC: dict = {}


def _wvec(w, d):
    """Fixed random embedding of a word / unit string (hashing tokenizer double; tests only)."""
    if (w, d) not in _WVEC:
        g = torch.Generator().manual_seed(int(hashlib.sha256(w.encode()).hexdigest()[:12], 16))
        _WVEC[(w, d)] = torch.randn(d, generator=g) / math.sqrt(d)
    return _WVEC[(w, d)]


def _experience(world, rng):
    """One experience of a world: prices (cents) and latencies (ms) around the world's levels, and the world's kind of
    complaint in text, about freshly named entities."""
    p, sv = f"prod_{rng.randrange(10 ** 4)}", f"svc_{rng.randrange(10 ** 4)}"
    obs = []
    for _ in range(3):
        obs.append(Observation(ObsKind.NUMERIC, value=round(world["price"] * (1 + 0.05 * rng.gauss(0, 1))),
                               unit="cents", entity=p))
        obs.append(Observation(ObsKind.NUMERIC, value=round(world["lat"] * (1 + 0.05 * rng.gauss(0, 1)), 1), unit="ms",
                               entity=sv))
    tpl = _TEMPLATES[world["type"]]
    for _ in range(2):
        obs.append(Observation(ObsKind.TEXT, tpl.format(p=p, s=sv), entity=p if "{p}" in tpl else sv))
    return obs


def _hash_tokenize(views, cfg, rng):
    """Observation records -> TokenSet: text = mean hashed word vectors; NUMERIC = unit-string vector + Fourier
    features of symlog(value) (the value exactly as observed, no unit normalisation); entities bound to random
    episode slots by an EntityRegistry per view."""
    D = cfg.d_model
    Dh = D - 16
    B, N = len(views), max(len(v) for v in views)
    x = torch.zeros(B, N, D)
    kind = torch.full((B, N), int(TokenKind.PAD))
    ent = torch.full((B, N), -1)
    mask = torch.zeros(B, N, dtype=torch.bool)
    k = torch.arange(1, 9).float() * 0.5
    for b, obs in enumerate(views):
        reg = EntityRegistry(cfg.max_entities, seed=rng.randrange(2 ** 31))
        for n, o in enumerate(obs):
            if o.kind == ObsKind.NUMERIC:
                x[b, n, :Dh] = _wvec("unit:" + o.unit, Dh)
                sl = math.copysign(math.log1p(abs(o.value)), o.value)
                x[b, n, Dh:] = torch.cat([torch.sin(k * sl), torch.cos(k * sl)])
                kind[b, n] = int(TokenKind.OBS_NUM)
            else:
                x[b, n, :Dh] = torch.stack([_wvec(w, Dh) for w in re.findall(r"[a-z]+", o.content.lower())]).mean(0)
                kind[b, n] = int(TokenKind.OBS_SEM)
            ent[b, n] = reg.bind(o.entity)
            mask[b, n] = True
    return TokenSet(x, mask, kind, torch.zeros(B, N), ent, None)


def _record_views(worlds, idx, cfg, rng, rnd):
    v1, v2 = [], []
    for i in idx:
        obs = _experience(worlds[i], rng)
        v1.append(rnd.make_map(obs).apply_observations(obs))
        v2.append(rnd.make_map(obs).apply_observations(obs))
    return _hash_tokenize(v1, cfg, rng), _hash_tokenize(v2, cfg, rng), torch.tensor(idx)


def test_invariance_from_surface_randomized_records_end_to_end():
    """Records -> SurfaceRandomizer views (renamed ids, rescaled units, reworded text, permuted order) -> tokens ->
    backbone -> InvarianceLoss. Held-out worlds (new price / latency levels): the trained representation retrieves
    the other view of the same experience among 16 worlds, which needs unit-invariant numeric identity (text type
    alone separates at most 4 groups)."""
    rng0 = random.Random(0)
    worlds = [{"price": math.exp(rng0.uniform(5, 9)), "lat": math.exp(rng0.uniform(4, 8)), "type": rng0.randrange(4)}
              for _ in range(64)]
    torch.manual_seed(0)
    cfg = tiny_config()
    core = UnifiedCore(cfg)
    lf = InvarianceLoss(cfg.d_model, proj_dim=32, temperature=0.1)
    rnd = SurfaceRandomizer(seed=1)

    def evaluate(seed=77):
        with torch.no_grad():
            v1, v2, w = _record_views(worlds, list(range(48, 64)), cfg, random.Random(seed), SurfaceRandomizer(seed=seed))
            _, st, _, _ = invariance_loss_from_views(core, lf, v1, v2, w)
        return st
    st0 = evaluate()
    opt = torch.optim.Adam(list(core.parameters()) + list(lf.parameters()), lr=1e-3)
    rng = random.Random(1)
    for _ in range(300):
        v1, v2, w = _record_views(worlds, rng.sample(range(48), 32), cfg, rng, rnd)
        loss, _, _, _ = invariance_loss_from_views(core, lf, v1, v2, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    st1 = evaluate()
    print(f"records end to end, held-out worlds: top1 {st0['top1']:.3f} -> {st1['top1']:.3f}; pos cos "
          f"{st1['pos_cos']:.3f} neg cos {st1['neg_cos']:.3f}")
    assert st0["top1"] < 0.3 and st1["top1"] >= 0.85
    assert st1["pos_cos"] - st1["neg_cos"] > 0.5
