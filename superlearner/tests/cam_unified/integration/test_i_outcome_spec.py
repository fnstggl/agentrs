"""outcome_spec: generic typed outcome keys, query tokens through the tokenizer, channel binding."""
import math

import torch

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.interfaces import (Action, EntityRegistry, ExecutionResult, Likelihood, Observation,
                                                 ObsKind, TokenKind)
from superlearner.cam_unified.outcome_spec import (CORE_KEYS, OutcomeQueryEncoder, OutcomeTracker, TokenizerTextEncoder,
                                                   bind_channels, build_outcome_batch, error_class,
                                                   outcomes_from_results, symlog)
from superlearner.cam_unified.testing import tiny_config, zero_fast_state


def num(v, metric, ent, unit, t=0.0, seq=None, etype=None):
    return Observation(ObsKind.NUMERIC, value=v, unit=unit, entity=ent, entity_type=etype, t=t, seq=seq,
                       source="dash", meta={"metric": metric})


def test_core_keys_likelihoods_and_targets():
    tr = OutcomeTracker(n_error_classes=16)
    a = Action("set_price", {"product": "p1", "price": 3.0}, logp=-1.0)
    r = ExecutionResult(False, [], error="not_found: no product", cost=250.0, duration=0.5,
                        meta={"error_code": "not_found"})
    specs = tr.realize([], a, r, [], 0.75)
    by = {s.key: s for s in specs}
    assert [s.key for s in specs[:5]] == list(CORE_KEYS)
    assert by["ok"].lik == Likelihood.BERNOULLI and by["ok"].y == 0.0
    assert by["error_code"].lik == Likelihood.CATEGORICAL and by["error_code"].n_classes == 16
    assert by["error_code"].y == error_class("not_found", 16) != 0
    assert error_class(None, 16) == 0 and error_class("not_found", 16) == error_class("not_found", 16)
    assert by["cost"].lik == Likelihood.GAUSSIAN and math.isclose(by["cost"].y, symlog(250.0))
    assert by["cost"].unit == "cents"
    assert math.isclose(by["duration"].y, symlog(0.5)) and by["duration"].unit == "days"
    assert by["reward"].y == 0.75 and by["reward"].space == "raw"
    assert all(s.observed for s in specs[:5])
    # error code parsed from the message when meta has none
    r2 = ExecutionResult(False, [], error="rate_limited: slow down")
    assert tr.realize([], a, r2, [], 0.0)[1].y == error_class("rate_limited", 16)


def test_numeric_deltas_levels_and_event_counts():
    tr = OutcomeTracker()
    prev = [num(100.0, "balance", "me", "cents", seq=1)]
    nxt = [num(130.0, "balance", "me", "cents", t=1.0, seq=2), num(160.0, "balance", "me", "cents", t=1.0, seq=3),
           num(0.25, "error_rate", "svc_1", "fraction", t=1.0, seq=4),
           Observation(ObsKind.TRANSACTION, "sold", t=1.0, seq=5, meta={"type": "sale"}),
           Observation(ObsKind.TRANSACTION, "sold", t=1.0, seq=6, meta={"type": "sale"}),
           Observation(ObsKind.TOOL_RESULT, "{}", source="x")]
    specs = {s.key: s for s in tr.realize(prev, Action("x"), ExecutionResult(True), nxt, 0.0)}
    d = specs["num:balance@me[cents]"]
    assert d.space == "symlog_delta" and math.isclose(d.y, symlog(60.0))     # latest value within the step
    lv = specs["lvl:error_rate@svc_1[fraction]"]
    assert lv.space == "symlog_level" and math.isclose(lv.y, symlog(0.25)) and lv.entity == "svc_1"
    assert "svc_1" not in lv.text                                             # identities bind via slots, not text
    c = specs["cnt:transaction/sale"]
    assert c.lik == Likelihood.COUNT and c.y == 2.0
    assert not any(k.startswith("cnt:tool_result") for k in specs)
    # next step: the tracked numeric becomes a delta query, the tracked event type gets an explicit zero count
    pro = {s.key for s in tr.prospective()}
    assert {"num:balance@me[cents]", "num:error_rate@svc_1[fraction]", "cnt:transaction/sale"} <= pro
    specs2 = {s.key: s for s in tr.realize([], Action("x"), ExecutionResult(True),
                                           [num(0.5, "error_rate", "svc_1", "fraction", t=2.0)], 0.0)}
    assert math.isclose(specs2["num:error_rate@svc_1[fraction]"].y, symlog(0.25))
    assert specs2["cnt:transaction/sale"].y == 0.0 and specs2["cnt:transaction/sale"].observed


def test_query_tokens_through_tokenizer_need_no_new_parameters():
    cfg = tiny_config()
    tok = dbl.DoubleTokenizer(cfg)
    enc = OutcomeQueryEncoder(cfg.d_model, TokenizerTextEncoder(tok))
    n_params = sum(p.numel() for p in list(tok.parameters()) + list(enc.parameters()))
    reg = EntityRegistry(8)
    slot = reg.bind("svc_1")
    tr = OutcomeTracker()
    specs = tr.realize([], Action("x"), ExecutionResult(True),
                       [num(1.0, "latency", "svc_1", "ms"), num(3.0, "brand_new_metric", None, "widgets/hour"),
                        Observation(ObsKind.EVENT, "outage", meta={"type": "never_seen_type"})], 0.0)
    before = tok.tokens_encoded
    ob = build_outcome_batch([specs], enc, [reg])
    assert tok.tokens_encoded > before                                        # queries went through the tokenizer
    assert sum(p.numel() for p in list(tok.parameters()) + list(enc.parameters())) == n_params
    Q = len(specs)
    assert ob.q.x.shape == (1, Q, cfg.d_model) and ob.lik.shape == (1, Q)
    assert (ob.q.kind == TokenKind.QUERY).all() and ob.q.mask.all() and ob.y_mask.all()
    keys = ob.keys[0]
    assert ob.q.entity[0, keys.index("lvl:latency@svc_1[ms]")] == slot
    x = ob.q.x[0]
    assert not torch.allclose(x[keys.index("cost")], x[keys.index("duration")])
    # gradients reach the tokenizer through the query path (training mode, no cache)
    ob.q.x.sum().backward()
    assert tok.text.emb.weight.grad is not None and tok.text.emb.weight.grad.abs().sum() > 0


def test_query_cache_is_versioned_and_only_without_grad():
    cfg = tiny_config()
    tok = dbl.DoubleTokenizer(cfg)
    ver = {"v": 0}
    te = TokenizerTextEncoder(tok, version_fn=lambda: ver["v"])
    with torch.no_grad():
        a = te(["cost in cents", "ok"])
        calls = te.calls
        b = te(["ok", "cost in cents"])
        assert te.calls == calls and torch.allclose(a[0], b[1])               # served from the cache
        with torch.no_grad():
            tok.text.emb.weight.add_(1.0)                                     # "update" the tokenizer
        c = te(["ok"])
        assert torch.allclose(c[0], b[0])                                     # same version -> cached (stale by design)
        ver["v"] = 1
        d = te(["ok"])
        assert te.calls == calls + 1 and not torch.allclose(d[0], b[0])       # new version -> recomputed
    e = te(["ok"])
    assert e.requires_grad                                                    # grad mode bypasses the cache


def test_channels_bound_in_fast_state_and_recycled_least_recent():
    cfg = tiny_config()
    fast = zero_fast_state(1, cfg, C=4)
    tr = OutcomeTracker()
    s1 = tr.realize([], Action("x"), ExecutionResult(True), [], 0.0)          # 5 core keys > 4 channels
    ch = bind_channels(fast, 0, s1[:4], tr.last_seen)
    assert sorted(ch) == [0, 1, 2, 3] and fast.channels[0]["ok"] == ch[0]
    fast.bayes_mu[0, ch[1]] = 5.0
    fast.bayes_n[0, ch[1]] = 3
    tr.last_seen.update({"ok": 10, "error_code": 1, "cost": 10, "duration": 10})
    ch2 = bind_channels(fast, 0, [s1[4]], tr.last_seen, prior_var=2.0, obs_var=0.5)   # 'reward' evicts error_code
    assert ch2 == [ch[1]] and "error_code" not in fast.channels[0] and fast.channels[0]["reward"] == ch[1]
    assert float(fast.bayes_mu[0, ch[1]].abs().sum()) == 0.0 and float(fast.bayes_n[0, ch[1]]) == 0
    assert torch.allclose(fast.bayes_P[0, ch[1]], torch.eye(cfg.bayes_feat, dtype=torch.float64) * 2.0)
    assert float(fast.bayes_noise[0, ch[1]]) == 0.5


def test_outcomes_from_results_functional_api_pads_and_binds():
    cfg = tiny_config()
    tok = dbl.DoubleTokenizer(cfg)
    fast = zero_fast_state(2, cfg, C=16)
    prev = [[num(10.0, "stock", "p1", "count")], []]
    nxt = [[num(7.0, "stock", "p1", "count", t=1)], [num(1.0, "a", "e", "u"), num(2.0, "b", "e", "u")]]
    acts = [Action("restock", {"product": "p1", "count": 3}), Action("wait", {"seconds": 60})]
    res = [ExecutionResult(True, cost=360.0, duration=0.02), ExecutionResult(True, duration=60 / 86400)]
    ob, specs = outcomes_from_results(prev, acts, res, nxt, [0.0, 1.0], tok, fast=fast, d_model=cfg.d_model,
                                      return_specs=True)
    Q = max(len(s) for s in specs)
    assert ob.y.shape == (2, Q) and ob.channel.shape == (2, Q)
    n0 = len(specs[0])
    assert n0 < Q and not ob.y_mask[0, n0:].any() and (ob.channel[0, n0:] == -1).all() and not ob.q.mask[0, n0:].any()
    assert (ob.channel[0, :n0] >= 0).all() and len(fast.channels[0]) == n0
    k = ob.keys[0].index("num:stock@p1[count]")
    assert math.isclose(float(ob.y[0, k]), symlog(-3.0), rel_tol=1e-6)
    assert math.isclose(float(ob.y[1, ob.keys[1].index("reward")]), 1.0)
