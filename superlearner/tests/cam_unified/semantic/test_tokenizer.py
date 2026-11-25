"""ObservationTokenizer: every ObsKind, numeric sensitivity, entity binding, budget trimming, tool/goal/action tokens."""
from __future__ import annotations

import json

import pytest
import torch

from superlearner.cam_unified.interfaces import (Action, EntityRegistry, EvidenceKind, GoalSpec, Observation, ObsKind,
                                                 TokenKind, ToolSchema)
from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer, parse_obs_ref, render_observation
from superlearner.cam_unified.testing import sample_tools


def one_of_every_kind(t0: float = 1.0) -> list:
    K = ObsKind
    return [
        Observation(K.TEXT, "Customers complain that onboarding is slow.", t=t0, source="forum", entity="seg_a",
                    entity_type="customer group"),
        Observation(K.TOOL_SCHEMA, "search.query(query: string, limit: integer) - search the web", t=t0,
                    source="available_tools"),
        Observation(K.TOOL_RESULT, '{"ok": true, "result": {"hits": 3}}', t=t0, source="search.query",
                    evidence=EvidenceKind.CONTROLLED, caused_by="e_1"),
        Observation(K.HTML, "<html><body><h1>Pricing</h1><p>$10/month</p></body></html>", t=t0, source="web"),
        Observation(K.A11Y, "button 'Buy' [enabled]; textbox 'email' [empty]", t=t0, source="browser"),
        Observation(K.CODE, "def handler(req):\n    return {'ok': True}\n", t=t0, source="repo", entity="repo_1",
                    entity_type="repository"),
        Observation(K.FILE, "name,price\nwidget,12\n", t=t0, source="files", artifact_ref="sha256:" + "ab" * 32),
        Observation(K.EXEC_RESULT, "stdout:\n42\n", t=t0, source="code.run", evidence=EvidenceKind.CONTROLLED),
        Observation(K.ERROR, "TimeoutError: upstream did not respond in 2000 ms", t=t0, source="deploy",
                    entity="svc_1", entity_type="service"),
        Observation(K.IMAGE, "screenshot of the dashboard", t=t0, source="browser", artifact_ref="sha256:" + "cd" * 32),
        Observation(K.NUMERIC, value=1250.0, unit="cents", entity="acct_1", entity_type="bank account", t=t0,
                    source="bank.balance", meta={"metric": "balance"}),
        Observation(K.RELATION, "", entity="prod_1", target="seg_a", relation="sold_to", t=t0, source="crm"),
        Observation(K.MESSAGE, "Can you deliver 40 units by Friday?", entity="p:cust7", target="p:me", t=t0,
                    source="mail"),
        Observation(K.TRANSACTION, "bank.txn kind=sale memo=widget", value=499.0, unit="cents", entity="acct_1",
                    target="acct_9", t=t0, source="bank"),
        Observation(K.EVENT, "rig.report tick 4: net -300 cents", t=t0, source="rig", seq=1042),
        Observation(K.GOAL, "Earn 10,000 cents of independent revenue within 8 days.", t=t0, source="goal"),
        Observation(K.CONSTRAINT, "At most 240 accepted actions per simulated hour.", t=t0, source="rules"),
    ]


def test_every_obskind_tokens_kinds_refs_time_and_entities(enc, cfg):
    tok = ObservationTokenizer(cfg, enc, max_obs_tokens=256)
    obs = one_of_every_kind(t0=1.0)
    assert {o.kind for o in obs} == set(ObsKind)
    reg = EntityRegistry(cfg.max_entities, seed=3)
    ts, stats = tok.observations_with_stats([obs], [reg], [1.5])
    Q = cfg.sem_tokens_per_obs
    expect = sum((0 if o.kind == ObsKind.NUMERIC else Q) + (1 if o.value is not None else 0) for o in obs)
    assert ts.x.shape == (1, expect, cfg.d_model) and int(ts.mask.sum()) == expect
    assert stats["dropped_obs"] == [0] and stats["kept_obs"] == [len(obs)]
    assert torch.isfinite(ts.x).all()
    per_obs: dict = {}
    for j, ref in enumerate(ts.refs[0]):
        per_obs.setdefault(parse_obs_ref(ref), []).append(j)
    assert sorted(per_obs) == list(range(len(obs)))
    for i, o in enumerate(obs):
        js = per_obs[i]
        kinds = ts.kind[0, js].tolist()
        if o.kind == ObsKind.NUMERIC:
            assert kinds == [TokenKind.OBS_NUM]
        else:
            main = {ObsKind.RELATION: TokenKind.RELATION, ObsKind.EVENT: TokenKind.EVENT}.get(o.kind, TokenKind.OBS_SEM)
            assert kinds[:Q] == [main] * Q
            if o.value is not None:
                assert kinds[Q:] == [TokenKind.OBS_NUM]
        assert torch.allclose(ts.time[0, js], torch.full((len(js),), o.t - 1.5, device=ts.time.device))
        slot = reg.slot.get(o.entity, -1) if o.entity else -1
        assert set(ts.entity[0, js].tolist()) == {slot}
        if o.kind in (ObsKind.RELATION, ObsKind.MESSAGE, ObsKind.TRANSACTION):
            assert set(ts.entity2[0, js].tolist()) == {reg.slot[o.target]}
        else:
            assert set(ts.entity2[0, js].tolist()) == {-1}
    assert "#seq:1042" in ts.refs[0][per_obs[14][0]] and "@sha256:" in ts.refs[0][per_obs[6][0]]
    # semantic observations retain their original content in the rendered text (exactness lives in the record)
    assert "2000 ms" in render_observation(obs[8])
    # versioned schema: records round-trip through JSON, a newer schema version is refused explicitly
    assert all(Observation.from_dict(json.loads(o.to_json())) == o for o in obs)
    future = Observation(ObsKind.TEXT, "from the future", schema_version=99)
    with pytest.raises(ValueError):
        tok.observations([[future]], None, [0.0])


def test_numeric_tokens_are_sensitive_to_value_and_unit(tok, cfg):
    def num(v, unit):
        o = Observation(ObsKind.NUMERIC, value=v, unit=unit, entity="m1", t=0.0, source="monitor",
                        meta={"metric": "latency"})
        return tok.observations([[o]], None, [0.0]).x[0, 0]

    a, a2 = num(120.0, "ms"), num(120.0, "ms")
    assert torch.equal(a, a2)
    assert (a - num(121.0, "ms")).norm() > 1e-3
    assert (a - num(120000.0, "ms")).norm() > (a - num(121.0, "ms")).norm()
    assert (a - num(120.0, "cents")).norm() > 1e-2
    assert (a - num(-120.0, "ms")).norm() > 1e-2
    # the exact value is preserved in the record and its rendered text (never only in an embedding)
    o = Observation(ObsKind.NUMERIC, value=1234567.891, unit="cents", t=0.0)
    assert "1234567.891" in render_observation(o)


def test_entity_slot_codes_bind_within_episode_but_are_random_across_episodes(tok, cfg):
    def tokens(seed):
        reg = EntityRegistry(cfg.max_entities, seed=seed)
        obs = [Observation(ObsKind.NUMERIC, value=5.0, unit="count", entity="prod_7", t=0.0, source="a"),
               Observation(ObsKind.NUMERIC, value=5.0, unit="count", entity="prod_7", t=0.0, source="a"),
               Observation(ObsKind.NUMERIC, value=5.0, unit="count", entity="prod_8", t=0.0, source="a")]
        return tok.observations([obs], [reg], [0.0]), reg

    ts, reg = tokens(11)
    assert ts.entity[0, 0] == ts.entity[0, 1] != ts.entity[0, 2]
    assert torch.allclose(ts.x[0, 0], ts.x[0, 1])
    assert (ts.x[0, 0] - ts.x[0, 2]).norm() > 1e-2           # same content, different entity -> different token
    ts_b, _ = tokens(11)
    assert torch.allclose(ts.x, ts_b.x)                        # same episode seed -> same codes
    ts_c, _ = tokens(12)
    assert (ts.x[0, 0] - ts_c.x[0, 0]).norm() > 1e-2           # another episode: the identity code is re-drawn
    codes = tok.slot_codes(reg)
    gram = codes @ codes.t() / cfg.d_model
    off = gram - torch.diag(torch.diag(gram))
    assert off.abs().max() < 1e-4 and torch.allclose(torch.diag(gram), torch.ones_like(torch.diag(gram)), atol=1e-4)


def test_budget_trimming_keeps_most_recent_and_reports_drops(enc, cfg):
    small = ObservationTokenizer(cfg, enc, max_obs_tokens=3 * cfg.sem_tokens_per_obs)
    obs = [Observation(ObsKind.TEXT, f"message number {i}", t=float(i), source="mail") for i in range(7)]
    ts, stats = small.observations_with_stats([obs, obs[:2]], None, [7.0, 7.0])
    assert stats["kept_obs"] == [3, 2] and stats["dropped_obs"] == [4, 0]
    assert stats["dropped_tokens"] == [4 * cfg.sem_tokens_per_obs, 0]
    kept = sorted({parse_obs_ref(r) for r in ts.refs[0] if r is not None})
    assert kept == [4, 5, 6]
    assert small.dropped_total["obs"] == 4


def test_tool_tokens_one_per_tool_unseen_tools_need_no_parameters(tok, cfg):
    n_params = sum(p.numel() for p in tok.parameters())
    keys = set(tok.state_dict())
    tools = [sample_tools(5), sample_tools(3, randomize_names=True)]
    ts = tok.tools(tools)
    assert ts.x.shape == (2, 5, cfg.d_model)
    assert ts.mask.sum(1).tolist() == [5, 3]
    assert (ts.kind[ts.mask] == TokenKind.TOOL).all()
    assert ts.refs[0][:5] == [t.key for t in tools[0]]
    novel = ToolSchema("zq_frobnicate", "Rotate the quantum widget by an angle.",
                       {"type": "object", "properties": {"angle": {"type": "number", "minimum": 0, "maximum": 360}},
                        "required": ["angle"], "additionalProperties": False})
    t2 = tok.tools([[novel]])
    assert torch.isfinite(t2.x).all()
    assert sum(p.numel() for p in tok.parameters()) == n_params and set(tok.state_dict()) == keys
    # the renamed tool's token differs from the original only through the name text
    assert (ts.x[1, 0] - ts.x[0, 0]).norm() > 1e-3


def test_goal_and_action_tokens(tok, cfg):
    goals = [GoalSpec("g1", "Earn revenue by selling a service.", {"revenue_cents": 10_000, "requests": 100}, 8.0),
             None, GoalSpec("g3", "Keep the service up.", {})]
    g = tok.goal(goals)
    Q = min(cfg.sem_tokens_per_obs, tok.max_goal_tokens)
    assert g.mask.sum(1).tolist() == [Q + 3, 0, Q]
    assert (g.kind[g.mask] == TokenKind.GOAL).all()
    g_b = tok.goal([GoalSpec("g1", "Earn revenue by selling a service.", {"revenue_cents": 20_000, "requests": 100},
                             8.0)])
    assert (g.x[0, Q] - g_b.x[0, Q]).norm() > 1e-3           # numeric target sensitivity
    tok.register_tools(sample_tools(5))
    acts = [Action("set_price", {"product": "prod_1", "price": 12.5}), None, Action("noop", {}),
            Action("run_python", {"code": "print(1)"}, code="print(1)")]
    a = tok.action(acts)
    assert a.mask.sum(1).tolist() == [1 + tok.n_action_arg_tokens, 0, 1, 1 + tok.n_action_arg_tokens]
    assert (a.kind[a.mask] == TokenKind.ACTION).all()
    a2 = tok.action([Action("set_price", {"product": "prod_1", "price": 99.0})])
    assert (a.x[0, 1:] - a2.x[0, 1:]).norm() > 1e-3            # arguments matter
    assert torch.nn.functional.cosine_similarity(a.x[0, 0], a2.x[0, 0], dim=0) > 0.999   # same tool token


def test_gradients_reach_pooling_and_projections_not_qwen(tok, enc, cfg):
    tok.zero_grad(set_to_none=True)
    for p in enc.peft.parameters():
        p.grad = None
    obs = one_of_every_kind()
    ts = tok.observations([obs], [EntityRegistry(cfg.max_entities, seed=0)], [0.0])
    loss = ts.x.pow(2).mean() + tok.tools([sample_tools(2)]).x.pow(2).mean()
    loss.backward()
    for name in ("q_obs", "q_tool", "slot_gain"):
        assert getattr(tok, name).grad is not None and getattr(tok, name).grad.abs().sum() > 0, name
    assert tok.pool.kv.weight.grad.abs().sum() > 0 and tok.num_mlp[0].weight.grad.abs().sum() > 0
    assert tok.unit_proj[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in enc.peft.parameters())    # cached states are constants for the tokenizer
    assert not any(k.startswith("enc") for k in tok.state_dict())
