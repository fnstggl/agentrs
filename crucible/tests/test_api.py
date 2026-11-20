"""Agent-facing API: WorldServer, rendering/parsing, Gymnasium, PettingZoo, HTTP (OpenEnv, MCP, /v1), stdio MCP."""
from __future__ import annotations

import importlib
import json
import os
import queue
import subprocess
import sys
import threading

import numpy as np
import pytest

import crucible.services.admin  # noqa: F401  (registers identity, bank, admin)
from crucible.api import render as R
from crucible.api.local import (API_VERSION, ActorIdentity, ApiError, AuthError, ConflictError, EvaluationResult,
                                NotFoundError, WorldServer)

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SPEC = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
DISCO_SPEC = {"family": "test", "seed": 1,
              "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}, {"name": "testdisco"}]}

# A test-only module: an agent-deployed-style service at host "shop" that becomes a tool once discovered
# (world.describe {"host": "shop"} -> world.learn -> available_tools changes).
DISCO_SRC = '''
from crucible.core import schema as S
from crucible.core.kernel import Endpoint, Module, register_module


@register_module
class TestDisco(Module):
    name = "testdisco"

    def install(self, world):
        world.hooks["host_endpoints"] = self._host_endpoints
        world.resolvers.append(self._resolve)

    def _eps(self):
        def buy(ctx, item):
            ctx.emit("note", {"what": "bought", "item": item}, vis=(ctx.actor,))
            return {"bought": item}
        return [Endpoint("shop.buy", "shop", S.obj({"item": S.s_str(maxLength=40)}, ["item"]), buy,
                         "Buy an item from the shop.", 5, False, builtin=False)]

    def _host_endpoints(self, world, host, principal):
        return self._eps() if host == "shop" else []

    def _resolve(self, tool, principal):
        if tool.startswith("shop.") and self.world.knows(principal, "host", "shop"):
            return next((e for e in self._eps() if e.name == tool), None)
        return None
'''


@pytest.fixture(scope="module")
def disco_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("disco")
    (d / "crucible_test_disco.py").write_text(DISCO_SRC)
    sys.path.insert(0, str(d))
    importlib.import_module("crucible_test_disco")
    yield d
    sys.path.remove(str(d))


def balance(server, wid, principal):
    w = server.world(wid)
    bank = w.modules["bank"]
    return bank.balance(w.state, bank.primary_account(w.state, principal))


@pytest.fixture
def srv():
    s = WorldServer("memory://")
    yield s
    s.close()


@pytest.fixture
def world(srv):
    wid = srv.create_world(SPEC)
    alice = srv.join_world(wid, "alice", endowment=1000)
    bob = srv.join_world(wid, "bob")
    return srv, wid, alice, bob


# ============================================================================ WorldServer
def test_create_join_observe_act(world):
    srv, wid, alice, bob = world
    assert alice.principal == "p:alice" and alice.token.startswith("cru_") and alice.account
    o = srv.observe(wid, alice)
    assert o.api == API_VERSION and o.time == 0 and o.tools_changed and o.tools
    names = {t["name"] for t in o.tools}
    assert "bank.transfer" in names and not any(n.startswith("admin.") for n in names)
    assert any(e["type"] == "id.person" and e["payload"]["handle"] == "alice" for e in o.events)
    assert o.summary["bank"]["balances"][alice.account] == 1000
    assert "bank.transfer" in o.text and "p:alice" in o.text

    r = srv.execute_action(wid, alice, {"tool": "bank.transfer",
                                        "args": {"from_account": alice.account, "to": "bob", "amount": 100}})
    assert r["ok"] and r["result"]["balance"] == 900 and r["cost_s"] == 10
    text = 'Paying bob now.\n```json\n{"tool": "bank.transfer", "args": {"from_account": "%s", "to": "bob", ' \
           '"amount": 50}}\n```' % alice.account
    assert srv.execute_action(wid, alice.token, text)["ok"]
    bad = srv.execute_action(wid, alice, "I am not sure what to do")
    assert not bad["ok"] and bad["error"]["code"] == "parse_error"
    rej = srv.execute_action(wid, alice, {"tool": "admin.endow", "args": {"principal": "p:alice", "amount": 5}})
    assert not rej["ok"] and rej["error"]["code"] == "forbidden"
    nsf = srv.execute_action(wid, bob, {"tool": "bank.transfer",
                                        "args": {"from_account": bob.account, "to": "alice", "amount": 10**6}})
    assert not nsf["ok"] and nsf["error"]["code"] == "insufficient_funds"

    assert srv.advance_time(wid, dt=120)["time"] == 120
    assert srv.advance_time(wid, until=300)["time"] == 300
    with pytest.raises(ApiError):
        srv.advance_time(wid, until=10)

    o2 = srv.observe(wid, alice)
    assert o2.time == 300 and not o2.tools_changed and o2.tools is None
    assert o2.last_result["tool"] == "admin.endow" and not o2.last_result["ok"]
    assert all(e["seq"] > o.seq for e in o2.events)
    assert [e["type"] for e in o2.events].count("bank.txn") == 2
    assert not any(e["type"] == "cmd" for e in o2.events)  # own commands are hidden
    ob = srv.observe(wid, bob)
    assert not any(e["type"] == "bank.account" and e["payload"]["owner"] == "p:alice" for e in ob.events)
    assert ob.summary["bank"]["balances"][bob.account] == 150
    assert srv.observe(wid, alice).events == []  # cursor advanced
    o3 = srv.observe(wid, alice, include_tools="always")
    assert o3.tools and not o3.tools_changed

    small = srv.observe(wid, bob, max_events=0)
    assert small.events == [] and small.dropped == 0


def test_tokens_and_auth(world):
    srv, wid, alice, bob = world
    with pytest.raises(AuthError):
        srv.observe(wid, "cru_not-a-real-token")
    with pytest.raises(AuthError):
        srv.execute_action(wid, ActorIdentity(wid, "p:alice", "cru_forged"), {"tool": "world.whoami", "args": {}})
    with pytest.raises(AuthError):
        srv.available_tools(wid, "")
    assert srv.authenticate(wid, bob.token).principal == "p:bob"
    assert srv.execute_action(wid, bob.token, {"tool": "world.whoami", "args": {}})["result"]["principal"] == "p:bob"
    with pytest.raises(NotFoundError):
        srv.observe("no-such-world", alice)
    # only hashes are stored
    blob = srv.store.get_blob(f"crucible-api/tokens/{wid}").decode()
    assert alice.token not in blob and "p:alice" in blob
    # extra token for an existing principal; revocation
    t2 = srv.issue_token(wid, "p:alice")
    assert srv.authenticate(wid, t2.token).principal == "p:alice"
    assert srv.revoke_token(wid, t2.token)
    with pytest.raises(AuthError):
        srv.authenticate(wid, t2.token)
    with pytest.raises(ApiError) as e:
        srv.join_world(wid, "alice")
    assert e.value.code == "conflict"


def test_snapshot_fork_isolation_replay_restore_evaluate(world):
    srv, wid, alice, bob = world
    sid = srv.snapshot_world(wid)
    seq_at_snap = srv.world(wid).seq
    fid = srv.fork_world(sid)
    assert fid != wid and srv.world_info(fid)["parent"]["world"] == wid
    # actors keep their tokens in the fork; changes do not leak in either direction
    assert srv.execute_action(fid, alice, {"tool": "bank.transfer",
                                           "args": {"from_account": alice.account, "to": "bob", "amount": 300}})["ok"]
    assert balance(srv, fid, "p:alice") == 700 and balance(srv, wid, "p:alice") == 1000
    assert srv.execute_action(wid, bob.token, {"tool": "id.update_profile", "args": {"bio": "hi"}})["ok"]
    assert srv.world(fid).state.get("persons", "p:bob").get("bio") != "hi"
    # replay: operator sees everything; actor sees its own; fork lineage includes pre-fork history
    all_f = srv.replay_events(fid)
    assert [e["seq"] for e in all_f] == list(range(1, srv.world(fid).seq + 1))
    assert all_f[seq_at_snap - 1]["hash"] == srv.replay_events(wid, 0, seq_at_snap)[-1]["hash"]
    mine = srv.replay_events(fid, actor=bob)
    assert mine and len(mine) < len(all_f)
    assert all(e["vis"] == "public" or "p:bob" in e["vis"] for e in mine)
    assert [e["seq"] for e in srv.replay_events(wid, 5, 8)] == [6, 7, 8]
    # ephemeral fork disappears on close
    eph = srv.fork_world(sid, ephemeral=True)
    assert srv.world_info(eph)["ephemeral"]
    srv.close_world(eph)
    with pytest.raises(NotFoundError):
        srv.world(eph)
    # restore: the name now points at a fork of the snapshot; history is kept
    srv.execute_action(wid, alice, {"tool": "bank.transfer",
                                    "args": {"from_account": alice.account, "to": "bob", "amount": 1}})
    old_seq = srv.world(wid).seq
    new = srv.restore_world(sid)
    assert new != wid and srv.resolve(wid) == new
    assert balance(srv, wid, "p:alice") == 1000  # via the name
    assert srv.authenticate(wid, alice).world_id == new
    assert srv.store.world_info(wid)["status"] == "superseded"
    assert srv.store.last_seq(wid) == old_seq  # superseded log untouched
    assert any(w.get("alias") and w["name"] == wid for w in srv.list_worlds())
    # evaluation: builtin, then hook
    ev = srv.evaluate_world(wid)
    assert isinstance(ev, EvaluationResult) and ev.ok and ev.source == "builtin"
    assert ev.metrics["principals"]["person"] == 2
    srv.world(wid).hooks["evaluate"] = lambda w, ctx: {"score": 42, "ctx": ctx}
    ev2 = srv.evaluate_world(wid, {"task": "x"})
    assert ev2.source == "hook" and ev2.metrics == {"score": 42, "ctx": {"task": "x"}}


def test_persistence_sqlite_new_server(tmp_path):
    url = f"sqlite:///{tmp_path / 'w.db'}"
    s1 = WorldServer(url)
    wid = s1.create_world(SPEC)
    alice = s1.join_world(wid, "alice", endowment=1000)
    s1.join_world(wid, "bob")
    s1.observe(wid, alice)
    assert s1.execute_action(wid, alice, {"tool": "bank.transfer",
                                          "args": {"from_account": alice.account, "to": "bob", "amount": 250}})["ok"]
    s1.advance_time(wid, dt=3600)
    seq, digest = s1.world(wid).seq, s1.world(wid).digest()
    with pytest.raises(ConflictError):  # single writer: the lease is held by s1
        WorldServer(url).load_world(wid)
    s1.close()

    s2 = WorldServer(url)
    info = s2.load_world(wid)
    assert info["seq"] == seq and info["time"] == 3600 and s2.world(wid).digest() == digest
    assert s2.authenticate(wid, alice.token).principal == "p:alice"
    with pytest.raises(AuthError):
        s2.authenticate(wid, alice.token + "x")
    assert balance(s2, wid, "p:alice") == 750
    o = s2.observe(wid, alice.token)  # cursor was persisted: only the unseen transfer event
    assert [e["type"] for e in o.events] == ["bank.txn"] and not o.tools_changed
    assert o.last_result["tool"] == "bank.transfer" and o.last_result["ok"]  # session saved at close
    assert s2.execute_action(wid, alice, {"tool": "world.clock", "args": {}})["result"]["time"] == 3600
    sid = s2.snapshot_world(wid)
    new = s2.restore_world(sid)
    s2.close()
    s3 = WorldServer(url)
    assert s3.resolve(wid) == new and s3.authenticate(wid, alice.token).world_id == new
    s3.close()


# ============================================================================ rendering / parsing
def test_parse_action_formats():
    want = {"tool": "bank.balance", "args": {"account": "acct_000001"}}
    cases = [
        '{"tool": "bank.balance", "args": {"account": "acct_000001"}}',
        'Sure.\n```json\n{"tool": "bank.balance", "args": {"account": "acct_000001"}}\n```\nDone.',
        '<think>maybe {"tool": "x.y"}</think>I will check: {"name": "bank.balance", "arguments": {"account": "acct_000001"}}',
        '<tool_call>{"name": "bank.balance", "arguments": "{\\"account\\": \\"acct_000001\\"}"}</tool_call>',
        '{"function": {"name": "bank.balance", "arguments": {"account": "acct_000001"}}}',
        'bank.balance {"account": "acct_000001"}',
        "{'tool': 'bank.balance', 'args': {'account': 'acct_000001'},}",
        {"tool": "bank.balance", "args": {"account": "acct_000001"}},
    ]
    for c in cases:
        assert R.parse_action(c) == want, c
    assert R.parse_action('{"tool": "world.clock"}') == {"tool": "world.clock", "args": {}}
    assert R.parse_action('{"tool": "a.b", "args": {}, "idempotency_key": "k1"}')["idempotency_key"] == "k1"
    for bad in ["", "hello", '{"args": {}}', '{"tool": "a.b", "args": 3}', None, 17, '{"tool": "a.b", "args": "{bad"}']:
        out = R.parse_action(bad)
        assert "error" in out and out["error"]["code"] == "parse_error"


def test_render_deterministic(world):
    srv, wid, alice, _ = world
    o = srv.observe(wid, alice, peek=True)
    assert R.render_observation(o) == R.render_observation(o.to_dict()) == o.text
    assert srv.observe(wid, alice, peek=True).text == o.text  # peek does not move the cursor
    line = R.render_tool(next(t for t in o.tools if t["name"] == "bank.transfer"))
    assert line.startswith("bank.transfer(amount: int, from_account: str, to: str, memo?: str, ref?: str) [10s]")
    assert R.fmt_time(90061) == "d1 01:01:01"


# ============================================================================ Gymnasium
class PayBobObjective:
    """Reward = money bob received this step; success once bob holds >= target."""

    def __init__(self, target=200):
        self.target = target
        self.begun = 0

    def _bob(self, world):
        bank = world.modules["bank"]
        return bank.balance(world.state, bank.primary_account(world.state, "p:bob"))

    def begin(self, world, actor):
        self.begun += 1
        self.last = self._bob(world)

    def step_reward(self, world, actor):
        now = self._bob(world)
        r, self.last = float(now - self.last), now
        return r

    def status(self, world, actor):
        b = self._bob(world)
        return {"success": b >= self.target, "failed": False, "progress": min(1.0, b / self.target)}


def _pay(env, amount):
    return json.dumps({"tool": "bank.transfer",
                       "args": {"from_account": env.identity.account, "to": "bob", "amount": amount}})


def test_gym_continue_mode_world_persists(world):
    from crucible.api.gym_env import CrucibleEnv
    srv, wid, _, bob = world
    env = CrucibleEnv(srv, wid, handle="learner", endowment=1000, max_steps=3, dt=60)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and info["tools_changed"] and info["available_tools"]
    t0 = obs["time"][0]
    total = 0
    for i in range(3):
        obs, r, term, trunc, info = env.step(_pay(env, 100) if i < 2 else {"tool": "world.whoami", "args": {}})
        assert env.observation_space.contains(obs) and info["action_result"]["ok"]
        total += 1
    assert trunc and not term and obs["time"][0] == t0 + 180
    assert balance(srv, wid, "p:learner") == 800
    obs, info = env.reset()  # episode boundary only
    assert balance(srv, wid, "p:learner") == 800 and obs["time"][0] == t0 + 180 and info["episode"] == 2
    assert info["world_id"] == wid and not info["tools_changed"]
    obs, r, term, trunc, info = env.step(_pay(env, 100))
    assert balance(srv, wid, "p:learner") == 700 and balance(srv, wid, "p:bob") == 300
    obs, r, term, trunc, info = env.step("not json at all")
    assert info["action_result"]["error"]["code"] == "parse_error" and r == 0.0
    assert env.action_space.contains(_pay(env, 1))
    env.close()


def test_gym_fork_mode_and_objective(world):
    from crucible.api.gym_env import CrucibleEnv
    srv, wid, _, _ = world
    obj = PayBobObjective(target=200)
    env = CrucibleEnv(srv, wid, handle="learner", endowment=1000, reset_mode="fork", objective=obj, max_steps=10,
                      time_mode="cost")
    obs1, info1 = env.reset(seed=1)
    fork1 = info1["world_id"]
    assert fork1 != wid and obj.begun == 1
    rs = []
    for _ in range(5):
        obs, r, term, trunc, info = env.step(_pay(env, 100))
        rs.append(r)
        if term:
            break
    assert rs == [100.0, 100.0] and term and not trunc and info["objective"]["success"]
    assert obs["time"][0] == 20  # time_mode="cost": two 10 s transfers
    assert balance(srv, fork1, "p:learner") == 800
    obs2, info2 = env.reset(seed=1)
    assert info2["world_id"] != fork1 and balance(srv, info2["world_id"], "p:learner") == 1000
    assert obs2["text"] == obs1["text"]  # every episode starts from the same state
    with pytest.raises(NotFoundError):  # previous ephemeral fork was discarded
        srv.world(fork1)
    assert balance(srv, wid, "p:learner") == 1000 and balance(srv, wid, "p:bob") == 0  # base world untouched
    env.close()


def test_gym_env_checker():
    """Gymnasium's passive checker (spaces, reset/step return types and containment). The full `check_env`
    determinism check is not applicable: `info` deliberately carries the episode index and the fork's world id."""
    from gymnasium.utils.passive_env_checker import (check_action_space, check_observation_space,
                                                     env_reset_passive_checker, env_step_passive_checker)
    from crucible.api.gym_env import CrucibleEnv
    env = CrucibleEnv(spec=SPEC, seed=3, handle="learner", endowment=100, reset_mode="fork", max_steps=5)
    check_observation_space(env.observation_space)
    check_action_space(env.action_space)
    obs0, _ = env_reset_passive_checker(env, seed=5)
    env.action_space.seed(0)
    for _ in range(6):
        out = env_step_passive_checker(env, env.action_space.sample())
        assert env.observation_space.contains(out[0])
    obs1, _ = env.reset(seed=5)
    assert obs1["text"] == obs0["text"]  # fork mode: same start state every episode
    env.close()


# ============================================================================ PettingZoo
def test_pettingzoo_parallel(world):
    from pettingzoo.test import parallel_api_test
    from crucible.api.pettingzoo_env import CrucibleParallelEnv
    srv, wid, _, _ = world
    env = CrucibleParallelEnv(srv, wid, agents=["ann", "ben", "cat"], endowment=500, max_steps=5, order_seed=7,
                              renderers={"cat": lambda o: f"t={o.time}"})
    parallel_api_test(env, num_cycles=20)

    obs, infos = env.reset(seed=11)
    assert set(obs) == {"ann", "ben", "cat"} and obs["cat"]["text"].startswith("t=")
    acts = {a: json.dumps({"tool": "bank.transfer", "args": {"from_account": env.identities[a].account,
                                                             "to": "bob", "amount": 10}}) for a in env.agents}
    obs, rew, term, trunc, infos = env.step(acts)
    order = infos["ann"]["order"]
    assert sorted(order) == ["ann", "ben", "cat"] and all(infos[a]["order"] == order for a in order)
    assert all(infos[a]["action_result"]["ok"] for a in order)
    episode, t_before = env.episode, srv.world(wid).now
    env.reset(seed=11)
    env.step(acts)
    env.step(acts)
    assert srv.world(wid).now == t_before + 120  # world persisted across resets (continue mode)
    # same seed, episode index and step -> same order (recorded, reproducible)
    env2 = CrucibleParallelEnv(srv, wid, actors={a: env.identities[a] for a in env.possible_agents})
    env2.reset(seed=11)
    env2.episode = episode
    i2 = env2.step(acts)[-1]
    assert i2["ann"]["order"] == order
    assert balance(srv, wid, "p:bob") >= 90


# ============================================================================ HTTP: OpenEnv, MCP, /v1
@pytest.fixture
def client(world):
    from fastapi.testclient import TestClient
    from crucible.api.gym_env import CrucibleEnv
    from crucible.api.http_server import make_app
    srv, wid, alice, bob = world
    env = CrucibleEnv(srv, wid, alice, max_steps=4)
    app = make_app(srv, operator_token="op-secret", env=env)
    with TestClient(app) as c:
        yield c, srv, wid, alice, bob


def test_http_openenv_routes(client):
    c, srv, wid, alice, _ = client
    assert c.get("/health").json() == {"status": "healthy"}
    assert c.get("/metadata").json()["version"] == API_VERSION
    sch = c.get("/schema").json()
    assert set(sch) == {"action", "observation", "state"}
    r = c.post("/reset", json={"seed": 1, "episode_id": "e1"}).json()
    assert set(r) == {"observation", "reward", "done"} and r["done"] is False and r["reward"] is None
    assert "reward" not in r["observation"] and r["observation"]["principal"] == "p:alice"
    assert r["observation"]["metadata"]["episode_id"] == "e1" and r["observation"]["tools"]
    r = c.post("/step", json={"action": {"tool": "bank.transfer", "args": {"from_account": alice.account, "to": "bob",
                                                                         "amount": 5}}, "timeout_s": 10,
                              "request_id": "r1"}).json()
    assert r["observation"]["action_result"]["ok"] and r["reward"] == 0.0 and r["done"] is False
    r = c.post("/step", json={"action": {"text": '```json\n{"tool": "world.whoami", "args": {}}\n```'}}).json()
    assert r["observation"]["action_result"]["result"]["principal"] == "p:alice"
    st = c.get("/state").json()
    assert st["episode_id"] == "e1" and st["step_count"] == 2 and st["world_id"] == wid and st["time"] == 120
    assert c.post("/step", json={"nope": 1}).status_code == 422
    c.post("/step", json={"action": {"tool": "world.clock", "args": {}}})
    r = c.post("/step", json={"action": {"tool": "world.clock", "args": {}}}).json()
    assert r["done"] is True and r["observation"]["metadata"]["truncated"]


def test_http_websocket(client):
    c, srv, wid, alice, _ = client
    with c.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        m = ws.receive_json()
        assert m["type"] == "error" and m["data"]["code"] == "INVALID_JSON"
        ws.send_json({"type": "bogus"})
        assert ws.receive_json()["data"]["code"] == "UNKNOWN_TYPE"
        ws.send_json({"type": "reset", "data": {"seed": "x"}})
        assert ws.receive_json()["data"]["code"] == "VALIDATION_ERROR"
        ws.send_json({"type": "reset", "data": {"seed": 3}})
        m = ws.receive_json()
        assert m["type"] == "observation" and m["data"]["done"] is False
        ws.send_json({"type": "step", "data": {"action": {"tool": "world.whoami", "args": {}}}})
        m = ws.receive_json()
        assert m["type"] == "observation" and m["data"]["observation"]["action_result"]["ok"]
        ws.send_json({"type": "step", "data": {"tool": "world.clock", "args": {}}})  # action given directly
        assert ws.receive_json()["data"]["observation"]["action_result"]["result"]["time"] == 60
        ws.send_json({"type": "state"})
        m = ws.receive_json()
        assert m["type"] == "state" and m["data"]["step_count"] == 2
        ws.send_json({"type": "step", "data": {}})
        assert ws.receive_json()["data"]["code"] == "VALIDATION_ERROR"
        ws.send_json({"type": "close"})


def test_http_mcp(client):
    c, srv, wid, alice, bob = client
    rpc = lambda method, params=None, mid=1, **kw: c.post("/mcp", json={"jsonrpc": "2.0", "id": mid, "method": method,
                                                                         "params": params or {}}, **kw).json()
    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert init["result"]["capabilities"]["tools"]["listChanged"] is True
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
    tools = rpc("tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"bank.transfer", "crucible.observe"} <= names
    assert all({"name", "description", "inputSchema"} <= set(t) for t in tools)
    res = rpc("tools/call", {"name": "bank.balance", "arguments": {"account": alice.account},
                             "_meta": {"progressToken": 1}})["result"]
    assert res["isError"] is False and res["structuredContent"]["result"]["balance"] == 1000
    assert json.loads(res["content"][0]["text"])["ok"] is True
    err = rpc("tools/call", {"name": "bank.balance", "arguments": {"wrong": 1}})["result"]
    assert err["isError"] is True and err["structuredContent"]["error"]["code"] == "invalid_action"
    obs = rpc("tools/call", {"name": "crucible.observe", "arguments": {}})["result"]
    assert "you are p:alice" in obs["content"][0]["text"]
    assert rpc("nope/nope")["error"]["code"] == -32601
    # another actor via bearer token + world header
    who = rpc("tools/call", {"name": "world.whoami", "arguments": {}},
              headers={"Authorization": f"Bearer {bob.token}", "X-Crucible-World": wid})["result"]
    assert who["structuredContent"]["result"]["principal"] == "p:bob"
    bad = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                 headers={"Authorization": "Bearer cru_bogus", "X-Crucible-World": wid})
    assert bad.status_code == 401
    assert c.post("/mcp", content=b"{oops").json()["error"]["code"] == -32700


def test_http_v1(client):
    c, srv, _, _, _ = client
    op = {"Authorization": "Bearer op-secret"}
    assert c.post("/v1/worlds", json={"spec": SPEC}).status_code == 401
    assert c.post("/v1/worlds", json={"spec": SPEC}, headers={"Authorization": "Bearer nope"}).status_code == 403
    w = c.post("/v1/worlds", json={"spec": SPEC, "world_id": "w-http"}, headers=op).json()
    assert w["world_id"] == "w-http" and w["modules"] == ["identity", "bank", "admin"]
    assert any(x["world_id"] == "w-http" for x in c.get("/v1/worlds", headers=op).json()["worlds"])
    j = c.post("/v1/worlds/w-http/join", json={"handle": "carol", "endowment": 500}, headers=op).json()
    c.post("/v1/worlds/w-http/join", json={"handle": "dave"}, headers=op)
    me = {"Authorization": f"Bearer {j['token']}"}
    assert j["principal"] == "p:carol"
    assert c.get("/v1/worlds/w-http/observe").status_code == 401
    assert c.get("/v1/worlds/w-http/observe", headers={"Authorization": "Bearer cru_x"}).status_code == 401
    o = c.get("/v1/worlds/w-http/observe", headers=me).json()
    assert o["principal"] == "p:carol" and o["tools_changed"] and "text" in o
    assert any(t["name"] == "bank.transfer" for t in c.get("/v1/worlds/w-http/tools", headers=me).json()["tools"])
    r = c.post("/v1/worlds/w-http/act", json={"tool": "bank.transfer", "args": {"from_account": j["account"],
                                                                                "to": "dave", "amount": 70}},
               headers=me).json()
    assert r["ok"] and r["result"]["balance"] == 430
    assert c.post("/v1/worlds/w-http/advance", json={"dt": 600}, headers=me).status_code == 403
    assert c.post("/v1/worlds/w-http/advance", json={"dt": 600}, headers=op).json()["time"] == 600
    sid = c.post("/v1/worlds/w-http/snapshot", headers=op).json()["snapshot_id"]
    f = c.post(f"/v1/snapshots/{sid}/fork", json={"world_id": "w-http-f"}, headers=op).json()
    assert f["world_id"] == "w-http-f" and f["parent"]["world"] == "w-http"
    assert c.get("/v1/worlds/w-http-f/observe", headers=me).json()["principal"] == "p:carol"  # token inherited
    ev_actor = c.get("/v1/worlds/w-http/events", headers=me).json()["events"]
    ev_op = c.get("/v1/worlds/w-http/events", headers=op).json()["events"]
    assert 0 < len(ev_actor) < len(ev_op)
    assert c.post("/v1/worlds/w-http/evaluate", json={}, headers=op).json()["ok"] is True
    assert c.get("/v1/worlds/nope/observe", headers=me).status_code == 404
    assert c.delete("/v1/worlds/w-http-f", headers=op).json() == {"closed": "w-http-f"}


def test_http_operator_disabled(srv):
    from fastapi.testclient import TestClient
    from crucible.api.http_server import make_app
    with TestClient(make_app(srv)) as c:
        assert c.post("/v1/worlds", json={"spec": SPEC}, headers={"Authorization": "Bearer x"}).status_code == 403
        assert c.post("/reset", json={}).status_code == 409


# ============================================================================ MCP (in-process + stdio)
def test_mcp_handler_list_changed_inprocess(disco_dir, srv):
    from crucible.api.mcp_server import MCPHandler
    wid = srv.create_world(DISCO_SPEC)
    a = srv.join_world(wid, "alice")
    h = MCPHandler(srv, wid, a.token, name_style="underscores")
    names = {t["name"] for t in h.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})[0]["result"]["tools"]}
    assert "bank__transfer" in names and "shop__buy" not in names
    resp, notes = h.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "world__describe", "arguments": {"host": "shop"}}})
    assert not resp["result"]["isError"] and notes == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]
    names = {t["name"] for t in h.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})[0]["result"]["tools"]}
    assert "shop__buy" in names
    resp, notes = h.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "shop__buy", "arguments": {"item": "tea"}}})
    assert resp["result"]["structuredContent"]["result"] == {"bought": "tea"} and notes == []
    o = srv.observe(wid, a)
    assert any(e["type"] == "know" for e in o.events) and "shop.buy" in {t["name"] for t in o.tools}


class _Proc:
    def __init__(self, args, env):
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=env, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(json.loads(line))

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv(self, timeout=30):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError("no message from MCP server; stderr: " + self.p.stderr.read()[-2000:]
                                 if self.p.poll() is not None else "timeout waiting for MCP server")


def test_mcp_stdio_subprocess(disco_dir, tmp_path):
    db = tmp_path / "mcp.db"
    url = f"sqlite:///{db}"
    s = WorldServer(url)
    wid = s.create_world(DISCO_SPEC, world_id="w-mcp")
    a = s.join_world(wid, "alice", endowment=100)
    s.close()
    env = dict(os.environ, CRUCIBLE_TOKEN=a.token,
               PYTHONPATH=os.pathsep.join([SRC, str(disco_dir), os.environ.get("PYTHONPATH", "")]))
    proc = _Proc([sys.executable, "-m", "crucible.api.mcp_server", "--store", url, "--world", wid,
                  "--import", "crucible_test_disco"], env)
    try:
        proc.send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "pytest"},
                              "_meta": {"x": 1}}})
        init = proc.recv()
        assert init["id"] == 1 and init["result"]["capabilities"] == {"tools": {"listChanged": True}}
        assert init["result"]["protocolVersion"] == "2025-03-26"
        proc.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        proc.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = proc.recv()
        assert tools["id"] == 2
        names = {t["name"] for t in tools["result"]["tools"]}
        assert "bank.transfer" in names and "shop.buy" not in names
        proc.send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "shop.buy", "arguments": {"item": "tea"}}})
        r = proc.recv()
        assert r["result"]["isError"] and r["result"]["structuredContent"]["error"]["code"] == "unknown_tool"
        proc.send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                   "params": {"name": "world.describe", "arguments": {"host": "shop"}, "_meta": {"progressToken": 9}}})
        r = proc.recv()
        assert r["id"] == 4 and r["result"]["isError"] is False
        assert proc.recv() == {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
        proc.send({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
        assert "shop.buy" in {t["name"] for t in proc.recv()["result"]["tools"]}
        proc.send({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                   "params": {"name": "shop.buy", "arguments": {"item": "tea"}}})
        r = proc.recv()
        assert r["id"] == 6 and r["result"]["structuredContent"]["result"] == {"bought": "tea"}
        proc.send({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        assert proc.recv() == {"jsonrpc": "2.0", "id": 7, "result": {}}
    finally:
        proc.p.stdin.close()
        proc.p.wait(timeout=30)
    assert proc.p.returncode == 0
    s2 = WorldServer(url)  # the subprocess released its lease and persisted its events
    evs = s2.replay_events(wid)
    assert any(e["type"] == "cmd" and e["payload"]["tool"] == "shop.buy" for e in evs)
    assert s2.world(wid).knows("p:alice", "host", "shop")
    s2.close()
