"""Experience tokens: every record kind, the separate value channel, exact values, time (NA-05, NA-11)."""
import math

import pytest
import torch
import torch.nn.functional as Fn

from cam_live.contracts import (ActionResult, ActionSpec, CandidateEvaluation, Cost, Decision, Distribution,
                                ExperienceItem, ExperimentRecord, FailureClass, ItemKind, Limits, Observation,
                                ObstacleRecord, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus,
                                PredictionRecord, Provenance, Quantity, ResultStatus, TaskSpec, ToolDescriptor,
                                Trust, VerifiedOutcome)
from cam_live.nn.core import elapsed_features
from cam_live.nn.tokens import (gap_features, relation_item, resource_items, stable_hash, time_item, to_items,
                                value_features)

LONG = "a long description of a web page with many words " * 6


def _tok(model, item):
    return model.tokenize([[item]]).x[0, 0]


def _item(value, name="members.count", text=LONG):
    return ExperienceItem(kind=ItemKind.OUTCOME, text=text, ts=10.0, quantities=[Quantity(name, value)])


def test_value_channel_separates_plus_one_from_minus_one(model):
    """An earlier design gave cos(+1, -1) = 0.98 because the text embedding swamped the number."""
    for text in (LONG, "x", ""):
        cos = float(Fn.cosine_similarity(_tok(model, _item(1.0, text=text)), _tok(model, _item(-1.0, text=text)), 0))
        assert cos < 0.35, (text[:10], cos)
    tok = model.exp.tokenizer
    b = model._to_device(tok.collate([[_item(1.0)], [_item(-1.0)]]))
    v = tok.value_channel(b["values"], b["value_mask"], b["name_ids"], b["unit_ids"], b["qkind"], b["exposure"])
    assert float(Fn.cosine_similarity(v[0, 0], v[1, 0], 0)) < -0.5          # the value channel itself flips
    assert abs(float(v[0, 0].pow(2).mean().sqrt()) - 1.0) < 0.05           # unit RMS, like the content channel
    c = tok.content(model._semantic(b, False), b["has_text"])
    assert abs(float(c[0, 0].pow(2).mean().sqrt()) - 1.0) < 0.05
    # the same text with and without a number is not the same token either
    no_val = ExperienceItem(kind=ItemKind.OUTCOME, text=LONG, ts=10.0)
    assert float(Fn.cosine_similarity(_tok(model, _item(1.0)), _tok(model, no_val), 0)) < 0.9


def test_magnitudes_and_metric_names_are_resolved(model):
    def cos(a, b):
        return float(Fn.cosine_similarity(_tok(model, a), _tok(model, b), 0))

    assert cos(_item(100.0), _item(103.0)) < 0.999          # close values are still distinguishable
    assert cos(_item(100.0), _item(3.0)) < 0.95
    assert cos(_item(1e6), _item(1e9)) < 0.98
    assert cos(_item(0.001), _item(0.01)) < 0.999
    assert cos(_item(0.0), _item(1.0)) < 0.9
    assert cos(_item(5.0, "visits"), _item(5.0, "signups")) < 0.8       # same number, different metric
    odd, even = value_features(torch.tensor([3.7, -3.7, 0.0, 1e12, -1e-6]))
    assert torch.allclose(odd[0], -odd[1]) and torch.allclose(even[0], even[1])
    assert torch.isfinite(odd).all() and torch.isfinite(even).all() and float(odd.abs().max()) <= 1.0
    assert float(odd[2].abs().sum()) == 0.0


def test_exact_values_survive_in_token_batch(model):
    vals = [100.0, 3.0, 12.34, -7.25e-5]
    item = ExperienceItem(kind=ItemKind.OBSERVATION, text="an action produced visits and signups", ts=5.0,
                          quantities=[Quantity("visits", vals[0], kind=OutcomeKind.COUNT),
                                      Quantity("signups", vals[1], kind=OutcomeKind.COUNT),
                                      Quantity("cost", vals[2], unit="usd"), Quantity("delta", vals[3])])
    tb = model.tokenize([[item], [ExperienceItem(kind=ItemKind.EVENT, text="nothing numeric", ts=1.0)]])
    assert tb.values.dtype == torch.float64 and tb.values.shape == (2, 1, 4)
    assert tb.values[0, 0].tolist() == vals                      # bit-exact, not reconstructed from an embedding
    assert tb.value_mask[0, 0].all() and not tb.value_mask[1, 0].any()
    assert tb.refs == [[""], [""]] and tb.time.dtype == torch.float64 and tb.kind_ids.dtype == torch.long
    # float64 keeps a large exact integer and a tiny difference that float32 would destroy
    big = ExperienceItem(kind=ItemKind.OUTCOME, text="t", quantities=[Quantity("a", 16777217.0),
                                                                    Quantity("b", 1800000000.123)])
    v = model.tokenize([[big]]).values[0, 0]
    assert v[0].item() == 16777217.0 and v[1].item() == 1800000000.123
    # non-finite values are dropped and counted, never encoded
    before = model.exp.tokenizer.stats["nonfinite_values"]
    bad = ExperienceItem(kind=ItemKind.OUTCOME, text="t", quantities=[Quantity("a", float("nan")), Quantity("b", 2.0)])
    tb = model.tokenize([[bad]])
    assert tb.values[0, 0, 0].item() == 2.0 and int(tb.value_mask[0, 0].sum()) == 1
    assert model.exp.tokenizer.stats["nonfinite_values"] == before + 1 and torch.isfinite(tb.x).all()


def _records():
    q = OutcomeQuery("q1", "members.count", kind=OutcomeKind.COUNT, horizon_s=86400.0, issued_at=100.0,
                     description="members after one day", exposure=2.0)
    act = ActionSpec("a1", "browser.click", arguments={"ref": "e3"}, est_cost=Cost(seconds=2.0, primitive_ops=1),
                     steps=[ActionSpec("a1s", "browser.type")])
    obs = Observation("o1", 120.0, "page", text="a page", quantities=[Quantity("links", 12.0)],
                      entities=["Example Org", "Form"], provenance=Provenance(trust=Trust.UNTRUSTED))
    res = ActionResult("a1", ResultStatus.FAILED, failure_class=FailureClass.PAGE_CHANGED, failure_detail="gone",
                       started=100.0, ended=103.5, observations=[obs], cost=Cost(seconds=3.5, money=0.25),
                       verified=False, retries=1)
    pred = PredictionRecord("p1", q, Distribution("poisson", mean=4.0, std=2.0, epistemic_std=1.5), made_at=101.0)
    dec = Decision("d1", 99.0, "obj", act, candidates=[CandidateEvaluation(act)], probabilities=[0.7])
    return {
        "observation": obs, "action": act, "result": res, "query": q,
        "outcome": (q, OutcomeObservation("q1", OutcomeStatus.OBSERVED, value=7.0, observed_at=86500.0, verified=True)),
        "censored": (q, OutcomeObservation("q1", OutcomeStatus.CENSORED, value=86400.0, observed_at=86500.0)),
        "unobservable": (q, OutcomeObservation("q1", OutcomeStatus.UNOBSERVABLE, observed_at=86500.0)),
        "prediction": pred,
        "tool": ToolDescriptor("browser.click", "structured_browser", "clicks an element",
                               arg_schema={"properties": {"ref": {"type": "string"}}}, est_cost=Cost(seconds=1.0)),
        "task": TaskSpec("obj", "reach the stated target", limits=Limits(duration_s=3600.0, money=10.0)),
        "cost": Cost(seconds=1.5, money=2.0, tokens=300),
        "verified": VerifiedOutcome("obj", "achieved", value=31.0, target=30.0, evaluated_at=500.0),
        "obstacle": ObstacleRecord("ob1", 50.0, "a1", FailureClass.MISSING_PERMISSION, "not allowed"),
        "experiment": ExperimentRecord("e1", 10.0, "does x change y", "yes", arms=["a", "b"], horizon_s=60.0),
        "decision": dec,
    }


def test_to_items_covers_every_record_type_and_item_kind(model):
    r = _records()
    kinds = lambda rec, **kw: [i.kind for i in to_items(rec, **kw)]
    assert kinds(r["observation"]) == [ItemKind.OBSERVATION, ItemKind.ENTITY, ItemKind.ENTITY]
    assert kinds(r["action"]) == [ItemKind.ACTION] and kinds(r["decision"]) == [ItemKind.ACTION]
    assert kinds(r["result"]) == [ItemKind.ATTEMPT, ItemKind.ATTEMPT, ItemKind.FAILURE, ItemKind.OBSERVATION,
                                  ItemKind.ENTITY, ItemKind.ENTITY]
    assert kinds(r["query"]) == [ItemKind.OUTCOME] and kinds(r["outcome"]) == [ItemKind.OUTCOME]
    assert kinds(r["prediction"]) == [ItemKind.PREDICTION, ItemKind.UNCERTAINTY]
    assert kinds(r["tool"]) == [ItemKind.TOOL] and kinds(r["cost"], ts=3.0) == [ItemKind.COST]
    assert kinds(r["task"]) == [ItemKind.GOAL, ItemKind.RESOURCE]
    assert kinds(r["verified"]) == [ItemKind.OUTCOME] and kinds(r["obstacle"]) == [ItemKind.FAILURE]
    assert kinds(r["experiment"]) == [ItemKind.EVENT]
    extra = [relation_item("page", "links to", "form", ts=1.0), time_item(100.0, started=40.0, deadline=400.0),
             *resource_items({"budget": {"money": 5.5, "left": 3}, "paused": False}, ts=2.0),
             *to_items(Observation("o2", 1.0, "document", text="doc")),
             *to_items(Observation("o3", 1.0, "code", text="print(1)"))]
    assert [e.kind for e in extra] == [ItemKind.RELATION, ItemKind.TIME, ItemKind.RESOURCE, ItemKind.DOCUMENT,
                                       ItemKind.SOFTWARE]
    all_items = [i for rec in r.values() for i in to_items(rec)] + extra
    assert {i.kind for i in all_items} == set(ItemKind.ALL)           # every kind of spec 3.1 is produced
    tb = model.tokenize([sorted(all_items, key=lambda i: i.ts)])
    assert torch.isfinite(tb.x).all() and int(tb.pad_mask.sum()) == len(all_items)
    out = model.core_forward(tb)
    assert torch.isfinite(out.b).all()

    # exact evidence and trust
    o = to_items(r["outcome"])[0]
    assert o.trust == Trust.VERIFIER and o.ts == 86500.0 and o.ref == "q1"
    assert [(q.name, q.value, q.kind, q.exposure) for q in o.quantities][0] == ("members.count", 7.0, "count", 2.0)
    c = to_items(r["censored"])[0]
    assert {q.name: q.value for q in c.quantities}["outcome.censored"] == 1.0 and c.trust == Trust.UNTRUSTED
    # pending / unobservable outcomes never get an invented value
    for key in ("unobservable", "query"):
        names = [q.name for i in to_items(r[key]) for q in i.quantities]
        assert "members.count" not in names
    res_items = to_items(r["result"])
    qd = {q.name: q.value for i in res_items for q in i.quantities}
    assert qd["result.ok"] == 0.0 and qd["result.duration"] == 3.5 and qd["cost.money"] == 0.25
    assert qd["result.verified"] == 0.0 and qd["result.retries"] == 1.0
    assert all(len(i.quantities) <= 4 for i in res_items)             # long quantity lists continue in further items
    assert to_items(r["task"])[0].text == "reach the stated target" and to_items(r["task"])[0].trust == Trust.OPERATOR
    assert {q.name: q.value for q in to_items(r["prediction"])[0].quantities}["pred.mean"] == 4.0
    with pytest.raises(TypeError):
        to_items(object())


def test_time_encoding_resolves_seconds_and_months():
    day = 86400.0
    f = elapsed_features(torch.tensor([0.0, 1.0, 2.0, 3600.0, 15 * day, 30 * day, 31 * day], dtype=torch.float64))
    d = lambda i, j: float((f[i] - f[j]).norm())
    assert d(0, 1) > 0.3 and d(1, 2) > 0.3                 # one second matters
    assert d(4, 5) > 0.3 and d(5, 6) > 0.05                # so do 15 vs 30 days, and 30 vs 31 days
    assert d(2, 3) > 0.3 and torch.isfinite(f).all() and float(f.abs().max()) <= 1.0
    # absolute UNIX times are handled in float64: a one-second gap at t = 1.8e9 is not lost
    t = torch.tensor([[1.8e9, 1.8e9 + 1.0, 1.8e9 + 1.0 + 30 * day]], dtype=torch.float64)
    gap = torch.zeros_like(t)
    gap[:, 1:] = t[:, 1:] - t[:, :-1]
    g = gap_features(gap, torch.tensor([[True, False, False]]))
    assert float((g[0, 0] - g[0, 1]).norm()) > 0.3 and float((g[0, 1] - g[0, 2]).norm()) > 0.3
    assert float(elapsed_features(t[:, 1] - t[:, 0])[0].sub(elapsed_features(torch.tensor([1.0]))[0]).abs().max()) < 1e-6


def test_token_time_and_padding(model):
    a = [ExperienceItem(kind=ItemKind.EVENT, text="e", ts=1.8e9 + i) for i in range(3)]
    b = [ExperienceItem(kind=ItemKind.EVENT, text="e", ts=1.8e9 + 86400.0 * i) for i in range(2)]
    tb = model.tokenize([a, b, []])
    assert tb.x.shape[:2] == (3, 3) and tb.pad_mask.tolist() == [[True] * 3, [True, True, False], [False] * 3]
    assert tb.time[0].tolist() == [1.8e9, 1.8e9 + 1, 1.8e9 + 2] and tb.time[1, 2].item() == 1.8e9 + 86400.0
    assert float(tb.x[1, 2].abs().sum()) == 0.0 and float(tb.x[2].abs().sum()) == 0.0
    assert not torch.allclose(tb.x[0, 1], tb.x[1, 1])               # same text, different gap to the predecessor
    out = model.core_forward(tb)                                     # an empty context is handled
    assert torch.isfinite(out.b).all() and torch.isfinite(out.h).all()
    # the newest max_context items are kept
    n = model.core_cfg.max_context + 5
    long = [ExperienceItem(kind=ItemKind.EVENT, text=f"e{i % 7}", ts=float(i), ref=str(i)) for i in range(n)]
    tb = model.tokenize([long])
    assert tb.x.shape[1] == model.core_cfg.max_context and tb.refs[0][-1] == str(n - 1) and tb.refs[0][0] == "5"


def test_stable_hash_is_process_independent(model):
    assert stable_hash("members.count", "cam.name.a") == 7469528568118672412
    assert stable_hash("", "cam.unit") == 2094319368812617021
    tok = model.exp.tokenizer
    e = tok.name_embedding(["visits", "signups", "visits"])
    assert torch.equal(e[0], e[2]) and not torch.equal(e[0], e[1])
