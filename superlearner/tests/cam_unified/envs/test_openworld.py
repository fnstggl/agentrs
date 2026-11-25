"""OpenWorldAdapter / OpenWorldOperator / OpenWorldEvaluator on the real Crucible OpenWorld (ow_tiny)."""
from __future__ import annotations

import gc
import json
import random

import numpy as np
import pytest

import crucible
crucible.load_all_modules()
from crucible.api import WorldServer                                   # noqa: E402
from crucible.core.kernel import World, WorldSpec                      # noqa: E402
from crucible.goals.contracts import GoalContract, GoalVerifier, RewardContract   # noqa: E402
from crucible.research import llm_eval as L                            # noqa: E402

from superlearner.cam_unified.interfaces import (Action, EnvAdapter, ExecutionResult, GoalEvaluator, GoalSpec,  # noqa: E402
                                                 IndependentGoalResult, ObservationBatch, ObsKind, ToolSchema)
from superlearner.cam_unified.envs import ow_convert as C              # noqa: E402
from superlearner.cam_unified.envs.openworld import OpenWorldAdapter, open_world, WAIT_TOOL   # noqa: E402
from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator   # noqa: E402


def tiny_goal(days=8, seed=1):
    return L.open_economy_goal(world_family="ow_tiny", seed=seed, overrides={"n_consumers": 800}, days=days,
                               window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2,
                               endowment=3_000_000)


@pytest.fixture()
def env():
    ad, op = open_world([tiny_goal()], seed=3)
    ev = OpenWorldEvaluator(op)
    yield ad, op, ev
    ad.close()


def test_protocols_satisfied(env):
    ad, op, ev = env
    assert isinstance(ad, EnvAdapter)
    assert isinstance(ev, GoalEvaluator)


def test_round_trip_observe_tools_execute_advance_collect_evaluate(env):
    ad, op, ev = env
    b0 = ad.reset()
    assert isinstance(b0, ObservationBatch) and len(b0.env_ids) == 1 and b0.t == [0.0] and b0.done == [False]
    obs = b0.observations[0]
    me = [o for o in obs if o.meta.get("self")]
    assert len(me) == 1 and me[0].entity.startswith("p:")
    principal = me[0].entity
    # genesis public events (need-format docs, listings, persons) are delivered once at reset
    types = {json.loads(o.content)["type"] for o in obs if o.kind in (ObsKind.EVENT, ObsKind.MESSAGE)}
    assert {"docs.create", "market.listing", "id.person"} <= types
    # the endowment arrives as a transaction in cents into an own account
    tx = [o for o in obs if o.kind is ObsKind.TRANSACTION]
    assert any(o.value == 3_000_000 and o.unit == "cents" and o.entity.startswith("acct_") for o in tx)
    # summaries: bank balance as NUMERIC cents
    bal = [o for o in obs if o.kind is ObsKind.NUMERIC and o.meta.get("field", "").startswith("bank.balances")]
    assert bal and bal[0].unit == "cents" and bal[0].value == 3_000_000
    tools = ad.available_tools()[0]
    assert all(isinstance(t, ToolSchema) for t in tools)
    names = {t.name for t in tools}
    assert len(tools) > 140 and WAIT_TOOL in names and "use.categories" in names
    assert not any(n.startswith("admin.") for n in names)
    cat = next(t for t in tools if t.name == "use.categories")
    assert cat.duration_hint is not None and cat.duration_hint > 0 and cat.parameters["type"] == "object"
    # execute: ok result, the clock does not move
    r = ad.execute([Action("use.categories", {})])[0]
    assert isinstance(r, ExecutionResult) and r.ok and r.duration == pytest.approx(1 / 86400)
    tr = [o for o in r.observations if o.kind is ObsKind.TOOL_RESULT][0]
    cats = json.loads(tr.content)
    assert isinstance(cats, list) and cats and "category" in cats[0]
    assert ad.observe().t == [0.0]
    # rejected call: structured error, not an exception
    bad = ad.execute([Action("bank.transfer", {"from_account": "acct_nope", "to": "acct_x", "amount": -5})])[0]
    assert not bad.ok and bad.error in ("invalid_action", "not_found", "forbidden")
    eobs = [o for o in bad.observations if o.kind is ObsKind.ERROR][0]
    assert eobs.meta["code"] == bad.error
    unknown = ad.execute([Action("no.such_tool", {})])[0]
    assert not unknown.ok and unknown.error == "unknown_tool"
    # a command with consequences: open an account -> bank.account event caused by this action
    a = Action("bank.open_account", {"owner": principal, "name": "savings"})
    r2 = ad.execute([a])[0]
    assert r2.ok
    nb = ad.observe()
    caused = [o for o in nb.observations[0] if o.caused_by == a.action_id]
    assert caused and all(o.evidence.value == "controlled" for o in caused)
    assert any(json.loads(o.content)["type"] == "bank.account" for o in caused)
    # advance: the operator clock moves, consequences arrive; evaluator gives a verified increment
    b2 = ad.advance_time([0.5])
    assert b2.t[0] == pytest.approx(0.5)
    assert ev.take_rewards() == [0.0]           # an idle agent earns nothing
    # collect_events is a stateless cursor API
    evs, cur = ad.collect_events([0])
    assert cur[0] > 0 and evs[0] and all(o.seq > 0 for o in evs[0])
    evs2, cur2 = ad.collect_events(cur)
    assert evs2[0] == [] and cur2 == cur
    res = ev.evaluate_goal(0)
    assert isinstance(res, IndependentGoalResult) and res.achieved is False and res.score == 0.0
    assert set(res.metrics) == {"independent_revenue", "use_independent_requests"}
    assert res.details["world"]["family"] == "ow_tiny"     # evaluator-only details exist ...
    g = ad.goals()[0]
    assert isinstance(g, GoalSpec) and g.deadline_days == 8 and g.metrics["independent_revenue"] == 10_000


def test_wait_tool_and_rate_limit_free_clock(env):
    ad, op, ev = env
    ad.reset()
    r = ad.execute([Action(WAIT_TOOL, {"seconds": 7200})])[0]
    assert r.ok and r.duration == pytest.approx(7200 / 86400) and r.meta["harness"]
    assert ad.observe().t == [0.0]                     # waiting is a duration, the harness advances
    bad = ad.execute([Action(WAIT_TOOL, {"seconds": 10 ** 9})])[0]
    assert not bad.ok and bad.error == "invalid_action"


def test_lossless_event_cursor_and_exactly_once_delivery():
    """Every actor-visible event is delivered exactly once through reset/observe/advance, collect_events(0)
    returns all of them, and both equal the server's actor-filtered log (minus the actor's own command records)."""
    ad, op = open_world([tiny_goal(days=3)], seed=5)
    try:
        from crucible.usefulness.agents import ExampleInductor
        from superlearner.cam_unified.envs.agent_protocol import PolicyAgent
        from superlearner.cam_unified.envs.harness import run_lifetime
        delivered = []
        b = ad.reset()
        delivered += [o.seq for o in b.observations[0] if o.seq is not None]
        agent = PolicyAgent([ExampleInductor()])
        agent.begin(b.env_ids, ad.goals())
        batch = b
        rng = random.Random(0)
        for _ in range(60):
            acts = agent.act(batch, ad.available_tools())
            res = ad.execute(acts)
            if rng.random() < 0.3:
                mid = ad.observe()
                delivered += [o.seq for o in mid.observations[0] if o.seq is not None]
            batch = ad.advance_time([(r.duration if r else 0) + rng.choice([600, 3600, 20000]) / 86400 for r in res])
            delivered += [o.seq for o in batch.observations[0] if o.seq is not None]
            from superlearner.cam_unified.envs.agent_protocol import Transition
            from superlearner.cam_unified.interfaces import EventRecord
            agent.observe(Transition([EventRecord("e", b.env_ids[0], "a", "ep", 0, 0, 0.0, [], acts[0], res[0])],
                                     batch, [0.0], batch.done, [0.0]))
            if batch.done[0]:
                break
        assert len(delivered) == len(set(delivered)), "an event was delivered twice"
        # ground truth from the server, actor-filtered, operator-side
        w = op.world(0)
        ident = op.envs[0].ident
        truth = [d["seq"] for d in op.srv.replay_events(op.slots[0].wid, 0, None, actor=ident)
                 if not (d["type"] in ("cmd", "cmd.rejected") and d["actor"] == op.principal(0))]
        assert sorted(delivered) == truth
        allc, cur = ad.collect_events([0])
        assert [o.seq for o in allc[0]] == truth and cur[0] == w.seq
        k = truth[len(truth) // 2]
        part, _ = ad.collect_events([k])
        assert [o.seq for o in part[0]] == [s for s in truth if s > k]
        # visibility: nothing internal or private to someone else
        ps = w.principals_of(op.principal(0))
        for ev in w.events_range(0, w.seq):
            if ev.seq in set(truth):
                assert w.visible(ev, ps)
        # buffer fallback: a tiny buffer still answers old cursors losslessly through the actor-filtered log
        op.buffer_limit = 10
        op._ingest(op.envs[0], [])
        old, _ = ad.collect_events([0])
        assert [o.seq for o in old[0]] == truth
    finally:
        ad.close()


def test_records_reconstruct_server_view_exactly():
    """Conversion is lossless: event dicts and result dicts rebuilt from records equal the server's."""
    srv = WorldServer("memory://", lease_ttl_s=None, max_events=100_000)
    g = tiny_goal()
    wid = srv.create_world({"family": "ow_tiny", "overrides": g.world["overrides"]}, 1)
    me = srv.join_world(wid, "lomi-42", endowment=1_000_000)
    for tool, args in [("use.categories", {}), ("bank.accounts", {}), ("market.browse", {"limit": 3}),
                       ("bank.transfer", {"from_account": "acct_x", "to": "acct_y", "amount": 1}),
                       ("no.such", {})]:
        raw = srv.execute_action(wid, me, {"tool": tool, "args": args})
        obs = C.result_observations(raw, tool, raw["t"], "a_1")
        er = ExecutionResult(raw["ok"], obs, None if raw["ok"] else raw["error"]["code"], 0.0,
                             raw.get("cost_s", 0) / 86400, raw["t"] / 86400, raw["t"] / 86400, meta={"tool": raw["tool"]})
        assert C.result_dict(er, tool) == raw
    srv.advance_time(wid, dt=3600)
    o = srv.observe(wid, me, include_tools="never")
    for d in o.events:
        assert C.event_dict(C.event_observation(d, set(), None)) == d
    so = C.summary_observations(o.summary, o.time, me.principal)
    assert C.summary_dict(so) == o.summary
    tools = srv.available_tools(wid, me)
    assert [C.tool_spec(C.tool_schema(t)) for t in tools] == [
        {**t, "annotations": {k: v for k, v in t["annotations"].items() if v is not None and
                              k in ("readOnlyHint", "costSeconds", "host", "builtin", "version")}} for t in tools]
    srv.close()


# ---------------------------------------------------------------------------------------------- firewall
_FORBIDDEN_TYPES = (World, WorldServer, WorldSpec, GoalContract, GoalVerifier, RewardContract)
_FORBIDDEN_ATTRS = {"state", "spec", "query", "system_call", "world_info", "family", "seed", "world", "srv",
                    "operator", "evaluator", "events_range", "snapshot_world", "fork_world", "replay_events"}


def _walk(root, max_nodes=200_000):
    """Everything reachable from `root` by attribute access and container traversal (what a learner holding the
    object could reach without introspecting function internals)."""
    seen, stack, found = set(), [root], []
    while stack and len(seen) < max_nodes:
        x = stack.pop()
        if id(x) in seen or isinstance(x, (str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(x))
        found.append(x)
        if isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, (list, tuple, set, frozenset)):
            stack.extend(x)
        elif callable(x) and not hasattr(x, "__dict__") and not hasattr(type(x), "__slots__"):
            continue
        else:
            d = getattr(x, "__dict__", None)
            if isinstance(d, dict) and not isinstance(x, type):
                stack.extend(d.values())
            for s in getattr(type(x), "__slots__", ()) or ():
                if hasattr(x, s):
                    stack.append(getattr(x, s))
    return found


def test_hidden_state_firewall():
    goal = tiny_goal()
    ad, op = open_world([goal, goal], seed=11)
    ev = OpenWorldEvaluator(op)
    try:
        batch = ad.reset()
        tools = ad.available_tools()
        res = ad.execute([Action("use.categories", {}), Action(WAIT_TOOL, {"seconds": 60})])
        nb = ad.advance_time([0.1, 0.1])
        evs, _ = ad.collect_events([0, 0])
        goals = ad.goals()
        # 1. nothing reachable from the adapter is a kernel, server, spec, contract or verifier
        reach = _walk(ad)
        assert not [x for x in reach if isinstance(x, _FORBIDDEN_TYPES)]
        # negative control: the same walk does find them from the operator-side evaluator
        assert [x for x in _walk(ev) if isinstance(x, World)]
        # 2. the adapter exposes none of the hidden-state attributes
        public = {n for n in dir(ad) if not n.startswith("__")}
        assert not public & _FORBIDDEN_ATTRS
        port = ad._port
        assert not hasattr(port, "__dict__") and set(type(port).__slots__) == {
            "reset", "observe", "tools", "execute", "collect", "advance", "goals", "now", "done", "env_ids", "close"}
        # 3. records carry neither the world family, the seed nor the kernel world id (a spec hash)
        wids = [s.wid for s in op.slots]
        spec_hash = op.world(0).spec.hash()[:12]
        blob = json.dumps([[o.to_dict() for o in olist] for olist in batch.observations + nb.observations + evs],
                          default=str)
        blob += json.dumps([[t.to_dict() for t in tl] for tl in tools]) + json.dumps([r.to_dict() for r in res])
        blob += json.dumps([g.to_dict() for g in goals]) + json.dumps(batch.env_ids)
        for secret in wids + [spec_hash, '"ow_tiny"', '"seed"', "n_consumers"]:
            assert secret not in blob, secret
        assert not any(g.family == "ow_tiny" for g in goals)
        # 4. tokens (bearer secrets) never appear in records
        for e in op.envs:
            assert e.ident.token not in blob
    finally:
        ad.close()


def test_shared_world_mode_joins_actors_to_one_world():
    goal = tiny_goal()
    ad, op = open_world([goal, goal, goal], shared_world=True, seed=2)
    ev = OpenWorldEvaluator(op)
    try:
        b = ad.reset()
        assert len(op.slots) == 1 and len({op.principal(i) for i in range(3)}) == 3
        assert len(set(b.env_ids)) == 3
        # actor 0 founds an org; actor 1 sees the public id.org event, actor 2 too, but not 0's private account
        p0 = op.principal(0)
        r = ad.execute([Action("id.create_org", {"handle": "zeta-works", "name": "Zeta", "kind": "company"}),
                        None, None])
        assert r[0].ok and r[1] is None
        nb = ad.advance_time([0.01, 0.01, 0.01])
        assert len(set(nb.t)) == 1                       # one operator clock
        seen1 = [json.loads(o.content) for o in nb.observations[1] if o.kind is ObsKind.EVENT]
        assert any(e["type"] == "id.org" and e["payload"]["id"] == "o:zeta-works" for e in seen1)
        assert not any(o.caused_by for o in nb.observations[1])     # provenance links only to own actions
        own0 = [o for o in nb.observations[0] if o.caused_by == r[0].observations[0].caused_by]
        assert own0
        acct_evs = [json.loads(o.content) for o in nb.observations[1] if o.kind is ObsKind.EVENT
                    and json.loads(o.content)["type"] == "bank.account"]
        assert all(e["payload"]["owner"] != p0 for e in acct_evs)
        assert all(ev.evaluate_goal(i).achieved is False for i in range(3))
    finally:
        ad.close()


def test_heterogeneous_participants_act_independently():
    from crucible.agents.scripted import RandomPolicy
    from crucible.usefulness.agents import ExampleInductor
    goal = tiny_goal()
    ad, op = open_world([goal], seed=4, participants=[("rival-01", ExampleInductor(), 3_000_000),
                                                      ("noise-02", RandomPolicy(), 100_000)])
    try:
        ad.reset()
        ad.advance_time([0.5])
        w = op.world(0)
        acts = {p["id"] if "id" in p else k: p.get("actions", 0) for k, p in w.state.scan("principals")}
        # one turn per participant per simulated hour: the inductor's script runs, the random actor's calls are
        # mostly rejected by schema validation but some are accepted
        assert acts.get("p:rival-01", 0) >= 10 and acts.get("p:noise-02", 0) >= 1
        assert acts.get(op.principal(0), 0) == 0         # the CAM seat did nothing; the others acted on their own
    finally:
        ad.close()


def test_process_isolated_operator_firewall_and_success():
    """The worlds and the evaluator live in another process: this process never holds a kernel object, and the
    agent still reaches verified success through the records alone."""
    import os
    from crucible.usefulness.agents import ExampleInductor
    from superlearner.cam_unified.envs.agent_protocol import PolicyAgent
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.openworld import open_world_isolated
    gc.collect()
    worlds_before = sum(1 for o in gc.get_objects() if type(o) is World)
    ad, rop, rev = open_world_isolated([tiny_goal(days=6)], evaluator={}, seed=0)
    try:
        assert rop.pid != os.getpid()
        res = run_lifetime(PolicyAgent([ExampleInductor()]), ad, rev, think_s=600)
        assert res.final[0]["achieved"] is True
        assert res.returns[0] == pytest.approx(2.0)
        gc.collect()
        assert sum(1 for o in gc.get_objects() if type(o) is World) == worlds_before
        assert not [x for x in _walk(ad) + _walk(rev) if isinstance(x, _FORBIDDEN_TYPES)]
    finally:
        ad.close()
