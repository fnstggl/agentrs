"""outcome_spec: generic typed outcome keys, query tokens through the tokenizer, channel binding."""
import math

import torch

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.interfaces import (Action, EntityRegistry, ExecutionResult, Likelihood, Observation,
                                                 ObsKind, TokenKind)
from superlearner.cam_unified.outcome_spec import (CORE_KEYS, OutcomeQueryEncoder, OutcomeTracker, TokenizerTextEncoder,
                                                   bind_channels, build_outcome_batch, error_class, lookup_channels,
                                                   make_room, outcomes_from_results, symlog)
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


def test_query_cache_never_serves_embeddings_of_other_weights():
    cfg = tiny_config()
    tok = dbl.DoubleTokenizer(cfg)
    epoch = {"v": 0}
    te = TokenizerTextEncoder(tok, epoch_fn=lambda: epoch["v"])
    with torch.no_grad():
        a = te(["cost in cents", "ok"])
        calls = te.calls
        b = te(["ok", "cost in cents"])
        assert te.calls == calls and torch.allclose(a[0], b[1]) and te.hits >= 2      # served from the cache
        tok.text.emb.weight.add_(1.0)            # in-place edit nobody announced (same epoch)
        c = te(["ok"])
        fresh = (lambda: (te.clear(), te(["ok"]))[1])()
        assert te.calls >= calls + 1 and not torch.allclose(c[0], b[0]) and torch.allclose(c[0], fresh[0])
        calls = te.calls
        te(["ok"])
        assert te.calls == calls                                              # cached again for the new weights
        epoch["v"] = 1                                                        # announced change (optimiser step)
        te(["ok"])
        assert te.calls == calls + 1
    e = te(["ok"])
    assert e.requires_grad                                                    # grad mode bypasses the cache


def test_channels_only_for_observed_continuous_keys_with_stable_lru():
    cfg = tiny_config()
    fast = zero_fast_state(1, cfg, C=4)
    tr = OutcomeTracker()
    stats = {}
    s1 = tr.realize([], Action("x"), ExecutionResult(True, cost=3.0), [num(1.0, "a", "e", "u")], 0.0)
    ch = bind_channels(fast, 0, s1, tr.last_seen, stats=stats)
    by = dict(zip([s.key for s in s1], ch))
    assert by["ok"] == -1 and by["error_code"] == -1                           # not continuous: particles only
    assert sorted(v for v in ch if v >= 0) == [0, 1, 2, 3]                     # cost, duration, reward, lvl:a
    assert set(fast.channels[0]) == {"cost", "duration", "reward", "lvl:a@e[u]"} and stats.get("unbound", 0) == 0
    # prospective (unobserved) keys only look up existing bindings, never allocate
    pro = tr.prospective()
    look = lookup_channels(fast, 0, pro)
    assert all((c >= 0) == (s.key in fast.channels[0]) for s, c in zip(pro, look))
    # a step realising 2 new continuous keys: they take the least recently *realised* slots not realised now
    fast.bayes_mu[0, by["lvl:a@e[u]"]] = 5.0
    fast.bayes_n[0, by["lvl:a@e[u]"]] = 3
    s2 = tr.realize([], Action("x"), ExecutionResult(True), [num(1.0, "b", "e", "u"), num(2.0, "c", "e", "u")], 0.0)
    keys2 = {s.key for s in s2 if s.observed and int(s.lik) in (0, 1)}
    assert keys2 == {"cost", "duration", "reward", "lvl:b@e[u]", "lvl:c@e[u]"}
    stats = {}
    ch2 = bind_channels(fast, 0, s2, tr.last_seen, prior_var=2.0, obs_var=0.5, stats=stats)
    # 5 realised continuous keys, 4 slots, only 1 non-realised victim (lvl:a): 1 evicted, 1 does not fit
    assert stats == {"evicted": 1, "unbound": 1} and "lvl:a@e[u]" not in fast.channels[0]
    slot = by["lvl:a@e[u]"]
    assert float(fast.bayes_mu[0, slot].abs().sum()) == 0.0 and float(fast.bayes_n[0, slot]) == 0   # reset to prior
    assert torch.allclose(fast.bayes_P[0, slot], torch.eye(cfg.bayes_feat, dtype=torch.float64) * 2.0)
    assert float(fast.bayes_noise[0, slot]) == 0.5
    # the same realised set again: nothing is evicted (keys realised now are protected), the overflow stays counted
    stats = {}
    before = dict(fast.channels[0])
    bind_channels(fast, 0, s2, tr.last_seen, stats=stats)
    assert stats == {"evicted": 0, "unbound": 1} and fast.channels[0] == before
    # make_room alone frees slots without allocating (the Bayes layer allocates)
    fast2 = zero_fast_state(1, cfg, C=3)
    fast2.channels[0] = {"old1": 0, "old2": 1, "cost": 2}
    tr.last_seen.update({"old1": 1, "old2": 5})
    ev = make_room(fast2, 0, [s for s in s2 if s.key in ("cost", "duration")], tr.last_seen)
    assert ev == ["old1"] and fast2.channels[0] == {"old2": 1, "cost": 2}


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
    cont = [int(s.lik) in (0, 1) and s.observed for s in specs[0]]
    assert all((int(c) >= 0) == k for c, k in zip(ob.channel[0, :n0].tolist(), cont))
    assert len(fast.channels[0]) == sum(cont) == 4                       # cost, duration, reward, num:stock
    k = ob.keys[0].index("num:stock@p1[count]")
    assert math.isclose(float(ob.y[0, k]), symlog(-3.0), rel_tol=1e-6)
    assert math.isclose(float(ob.y[1, ob.keys[1].index("reward")]), 1.0)
