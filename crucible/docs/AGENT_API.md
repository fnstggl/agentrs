# Agent API (`crucible-api/1`)

Code: `crucible/src/crucible/api/` (`local.py`, `render.py`, `gym_env.py`, `pettingzoo_env.py`, `http_server.py`,
`mcp_server.py`). Tests: `crucible/tests/test_api.py`.

Every way an agent can reach a Crucible world goes through one object, the **`WorldServer`**. The other surfaces
are thin adapters over it:

```
                 ┌──────────── CrucibleEnv (Gymnasium) ── one learner
                 ├──────────── CrucibleParallelEnv (PettingZoo) ── several learners
 WorldServer ────┼──────────── make_app (FastAPI): OpenEnv /reset /step /ws · MCP POST /mcp · REST /v1
 (local.py)      └──────────── MCPHandler: stdio MCP server (python -m crucible.api.mcp_server)
      │
  World kernel (core/kernel.py) ── Store (memory:// · sqlite:/// · postgresql://)
```

## 1. Lifetimes: world, agent, trial, episode, rollout

Crucible worlds are persistent. Five time scales must not be confused:

| Term | What it is | Begins / ends |
|---|---|---|
| **World lifetime** | One event log and its state (`world_id`). | `create_world` … never ends by itself. It does not reset. Snapshots, forks and restores create *new* world ids; history is never rewritten. |
| **Agent lifetime** | One actor (principal `p:<handle>`) inside one world: identity, accounts, reputation, knowledge. | `join_world` … the actor is suspended or the world is retired. It spans any number of episodes and controller (model) versions. |
| **Trial** | One experimental condition: a world (or a fork of a common snapshot) plus the controllers and objectives evaluated in it. | Chosen by the experimenter; usually a fork per trial so that trials are independent. |
| **Episode** | A bounded stretch of one actor's lifetime used for credit assignment (`reset` … `terminated`/`truncated`). | `env.reset()` … objective success/failure or `max_steps`. **Ending an episode never resets the world** (`reset_mode="continue"`). |
| **Rollout** | A sequence of steps collected for one learner update; may cover part of an episode or several. | Defined by the training loop, invisible to the world. |

`reset_mode="fork"` is the one deliberate exception for episodic training: every episode starts in a *fresh fork* of a
fixed base snapshot; the persistent world is untouched by those episodes, and ephemeral forks are discarded at the
next reset.

## 2. Versioning

* `crucible.api.API_VERSION == "crucible-api/1"`. It appears in every `Observation` (`api`), `GET /metadata`,
  `GET /v1/version`, MCP `serverInfo.version`, and the Gym/PettingZoo `metadata`.
* Within `/1`, changes are additive only (new optional fields, endpoints, tools). Removing or renaming a field,
  changing a default that alters episode semantics, or changing the token scheme bumps it to `crucible-api/2`
  (and the REST prefix to `/v2`).
* The kernel and event schema are versioned separately (`crucible-kernel/1` in `WorldSpec.kernel`).
* MCP: protocol versions `2025-06-18` (default), `2025-03-26`, `2024-11-05` are accepted at `initialize`.

## 3. Native Python API

```python
import crucible.services.admin  # registers identity, bank, admin (WorldServer also auto-imports known modules)
from crucible.api import WorldServer

spec = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
with WorldServer("memory://") as srv:                     # or "sqlite:///worlds.db", "postgresql://..."
    wid = srv.create_world(spec)
    alice = srv.join_world(wid, "alice", endowment=10_000)  # ActorIdentity(world_id, principal, token, account)
    bob = srv.join_world(wid, "bob")
    obs = srv.observe(wid, alice)                         # Observation; obs.text is the rendered form
    print(obs.text)
    r = srv.execute_action(wid, alice, {"tool": "bank.transfer",
                                        "args": {"from_account": alice.account, "to": "bob", "amount": 2_500}})
    assert r["ok"] and r["result"]["balance"] == 7_500
    srv.advance_time(wid, dt=3600)                        # run scheduled processes for one simulated hour
    print(srv.observe(wid, bob.token).events)             # a token string works as well as the ActorIdentity
    sid = srv.snapshot_world(wid)
    fork = srv.fork_world(sid)                            # isolated copy; actors keep their tokens
    print(srv.evaluate_world(wid).ok)
```

### Operations

Operator operations (in-process code is the operator):

| Method | Returns | Notes |
|---|---|---|
| `WorldServer(store_url="memory://", *, store=None, strict=False, autoflush=True, lease_ttl_s=600, max_events=50, max_scan=200_000, persist_sessions=True, renderer=None, hide_own_commands=True)` | server | `lease_ttl_s=None` disables the single-writer lease. |
| `create_world(config, seed=None, *, world_id=None)` | world id | `config`: `WorldSpec`, a WorldSpec dict (has `"modules"`), or a generator config for `crucible.worldgen.generate(config, seed)` (error code `worldgen_unavailable` if that is missing). |
| `load_world(world_id)` / `world(world_id)` | info dict / kernel `World` | Load = latest snapshot + replay with hash-chain verification. `world()` gives the live kernel object (objectives, hooks). |
| `world_info(world_id)`, `list_worlds()` | dicts | `list_worlds` also lists restore aliases. |
| `join_world(world_id, handle, kind="agent", endowment=0, *, name=None, password=None, controller="", info=None)` | `ActorIdentity` | Recorded `admin.join` command (identity, checking account, endowment in cents) + new token. |
| `issue_token(world_id, principal)`, `revoke_token(world_id, token)` | `ActorIdentity` / bool | Tokens for principals created by generators or NPC code. |
| `authenticate(world_id, token_or_identity)` | `ActorIdentity` | Raises `AuthError` for unknown tokens. |
| `advance_time(world_id, until=None, *, dt=None, max_jobs=None)` | `{world_id, time, runs, seq}` | Time never goes backwards. |
| `snapshot_world(world_id)` | snapshot id | Immutable, stored in the store. |
| `fork_world(snapshot_id, new_world_id=None, *, ephemeral=False, inherit_tokens=True)` | world id | `ephemeral=True`: private in-memory store, discarded by `close_world`. |
| `restore_world(snapshot_id, name=None)` | new world id | Forks the snapshot into `<name>.r<n>` and points `name` at it; the old world is kept with status `superseded`. Aliases persist in the store. |
| `replay_events(world_id, from_seq=0, to_seq=None, actor=None)` | event dicts | `from_seq < seq <= to_seq`, following fork lineage into parents. With `actor`, only events that actor may see. |
| `evaluate_world(world_id, evaluator_context=None)` | `EvaluationResult` | `world.hooks["evaluate"](world, ctx)` if registered (may return a metrics dict), else invariants + basic statistics. |
| `close_world(world_id)`, `close()` | – | Flush, persist sessions, release leases. |

Actor operations (credential = `ActorIdentity` or its token string):

| Method | Returns |
|---|---|
| `observe(world_id, actor, *, max_events=None, include_tools="changed", peek=False)` | `Observation` |
| `available_tools(world_id, actor)` | list of MCP-shaped tool specs `{name, description, inputSchema, annotations}` (operator-only `admin.*` tools are never listed) |
| `execute_action(world_id, actor, action)` | ActionResult dict `{ok, t, tool, cost_s, result | error[, replayed]}` |

`action` is `{"tool": str, "args": {...}, "idempotency_key"?: str}`, or model output text containing one (see
`parse_action` below). In-world failures never raise; they return `ok=False` with an error code (`invalid_action`,
`unknown_tool`, `forbidden`, `insufficient_funds`, ...; `parse_error` for unparseable text, which is not sent to
the world). API failures raise `ApiError` subclasses: `NotFoundError` (404), `AuthError` (401), `ForbiddenError`
(403), `ConflictError` (409).

### Observation

`Observation(world_id, principal, time, seq, events, dropped, summary, last_result, tools_changed, tools,
new_tools, removed_tools, unscanned, observed, text, api)`:

* `events`: the actor-visible events since its previous observation, compact (`seq, t, type, actor, payload`), at
  most `max_events` (newest kept; `dropped` counts the rest). The actor's own `cmd`/`cmd.rejected` records are
  omitted (`last_result` carries the outcome of its last action).
* `summary`: per-module summaries from `World.observe` (e.g. `bank.balances`, `identity.orgs`).
* `tools_changed`: true on the first observation and whenever the actor's tool list changed (e.g. it discovered a
  service host); `tools` is then included (`include_tools="always"` includes it every time).
* The cursor is per (world, actor) and persisted in the store: after a restart the next observation contains exactly
  the unseen events.

### Rendering and parsing (`crucible.api.render`)

```python
from crucible.api.render import parse_action, render_tools, action_format_help

fence = "`" * 3
reply = f'Let me check.\n{fence}json\n{{"tool": "bank.balance", "args": {{"account": "acct_000001"}}}}\n{fence}'
assert parse_action(reply) == {"tool": "bank.balance", "args": {"account": "acct_000001"}}
assert parse_action('<tool_call>{"name": "world.clock", "arguments": {}}</tool_call>')["tool"] == "world.clock"
assert parse_action("no idea")["error"]["code"] == "parse_error"   # never raises
print(action_format_help())
```

`render_observation(obs)` and `render_tools(tools)` are deterministic (sorted JSON keys, ASCII escapes): one line
per event (`[d0 01:00:00] #19 bank.txn by p:alice: {...}`) and per tool
(`bank.transfer(amount: int, from_account: str, to: str, memo?: str, ref?: str) [10s] - Send money...`).
`parse_action` accepts plain JSON, code fences, `<tool_call>` tags, `<think>` blocks (ignored), prose around the
JSON, OpenAI/MCP shapes (`name`/`arguments`, `function`, arguments as a JSON string), `tool.name {json}`, and
Python-style dicts.

## 4. Gymnasium: `CrucibleEnv`

```python
import json
from crucible.api import WorldServer
from crucible.api.gym_env import CrucibleEnv

spec = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
srv = WorldServer()
wid = srv.create_world(spec)
srv.join_world(wid, "bob")

class PayBob:                                   # objective protocol: begin / step_reward / status
    def _bob(self, w):
        bank = w.modules["bank"]
        return bank.balance(w.state, bank.primary_account(w.state, "p:bob"))
    def begin(self, world, actor): self.last = self._bob(world)
    def step_reward(self, world, actor):
        now, last = self._bob(world), self.last
        self.last = now
        return float(now - last)
    def status(self, world, actor):
        b = self._bob(world)
        return {"success": b >= 300, "failed": False, "progress": min(1.0, b / 300)}

env = CrucibleEnv(srv, wid, handle="learner", endowment=1_000, objective=PayBob(), max_steps=10, dt=60)
obs, info = env.reset(seed=0)                   # obs = {"text": str, "time": array([t])}
pay = json.dumps({"tool": "bank.transfer", "args": {"from_account": env.identity.account, "to": "bob", "amount": 100}})
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(pay)   # or a dict {"tool": ..., "args": ...}
    if terminated or truncated:
        break
print(info["objective"], obs["time"])           # {'success': True, ...} after three payments
obs, info = env.reset()                          # continue mode: money, clock, history all carry over
assert srv.world(wid).now == 180
```

Parameters: `server`, `world_id`, `actor` (ActorIdentity or token; omitted → `join_world(handle, endowment)`),
`spec`/`seed` (create a world when `world_id` is omitted), `objective`, `reset_mode` (`"continue"` | `"fork"`),
`base_snapshot` (fork mode; default: snapshot taken at construction), `ephemeral_forks=True`, `dt=60`,
`time_mode` (`"fixed"`: advance `dt` after each step; `"cost"`: advance by the action's `cost_s`, at least `min_dt`;
`"none"`: an external driver owns the clock), `max_steps` (truncation), `max_events`, `max_text_length`,
`renderer(obs) -> str`, `include_tools`, `driver(server, world_id)` (called after the agent's action, before the
clock moves: scripted NPCs, other policies), `render_mode="ansi"`.

Spaces: `observation_space = Dict(text=Text(max_length, charset=printable ASCII), time=Box(0, inf, (1,)))`,
`action_space = Text(8192)`. `info` holds `action_result`, `events`, `dropped_events`, `tools_changed`,
`available_tools` (only when changed), `world_id` (the fork's id in fork mode), `principal`, `episode`, `step`,
`time`, `seq`, `objective`. Gymnasium's passive checker passes; the strict `check_env` determinism test does not
apply because `info` deliberately carries the episode index and fork world id.

Fork mode for episodic training from a fixed start state:

```python
from crucible.api import WorldServer
from crucible.api.gym_env import CrucibleEnv

spec = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
env = CrucibleEnv(spec=spec, seed=1, handle="learner", endowment=500, reset_mode="fork", max_steps=5)
o1, i1 = env.reset(seed=0)
env.step({"tool": "bank.open_account", "args": {"name": "savings"}})
o2, i2 = env.reset(seed=0)
assert o1["text"] == o2["text"] and i1["world_id"] != i2["world_id"]   # same start, fresh fork
env.close()
```

## 5. PettingZoo: `CrucibleParallelEnv`

```python
import json
from pettingzoo.test import parallel_api_test
from crucible.api.pettingzoo_env import CrucibleParallelEnv

spec = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
env = CrucibleParallelEnv(spec=spec, agents=["ann", "ben"], endowment=500, max_steps=5, order_seed=7,
                          renderers={"ben": lambda o: f"t={o.time} balances={o.summary.get('bank')}"})
parallel_api_test(env, num_cycles=10)
obs, infos = env.reset(seed=3)
acts = {a: json.dumps({"tool": "bank.transfer", "args": {"from_account": env.identities[a].account,
                                                         "to": "ben" if a == "ann" else "ann", "amount": 10}})
        for a in env.agents}
obs, rewards, terms, truncs, infos = env.step(acts)
print(infos["ann"]["order"])   # execution order this step, from default_rng([order_seed, episode, step])
env.close()
```

All agents' actions in a step run in a seeded random order (recorded in every agent's `infos[a]["order"]`), then
the clock advances once by `dt`. `objectives` and `renderers` are per-agent dicts; `actors` may map agent names to
existing identities/tokens instead of joining new actors. `reset_mode` works as in `CrucibleEnv`.

## 6. HTTP server (`make_app`)

```bash
export CRUCIBLE_OPERATOR_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(24))')
python -m crucible.api.http_server --store sqlite:///worlds.db --port 8000          # binds 127.0.0.1
# optional default OpenEnv/MCP binding:  CRUCIBLE_TOKEN=cru_... ... --world <world id or name>
```

The same app in-process (runnable; uses FastAPI's TestClient):

```python
from fastapi.testclient import TestClient
from crucible.api import WorldServer
from crucible.api.gym_env import CrucibleEnv
from crucible.api.http_server import make_app

spec = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
srv = WorldServer()
wid = srv.create_world(spec)
me = srv.join_world(wid, "alice", endowment=1_000)
app = make_app(srv, operator_token="op-secret", env=CrucibleEnv(srv, wid, me, max_steps=50))
c = TestClient(app)
op = {"Authorization": "Bearer op-secret"}

# (a) OpenEnv surface for the default binding
r = c.post("/reset", json={"seed": 0, "episode_id": "ep-1"}).json()       # {observation, reward: None, done: False}
r = c.post("/step", json={"action": {"tool": "world.whoami", "args": {}}}).json()
print(r["observation"]["action_result"], r["reward"], r["done"], c.get("/state").json()["step_count"])
with c.websocket_connect("/ws") as ws:
    ws.send_json({"type": "step", "data": {"tool": "world.clock", "args": {}}})
    print(ws.receive_json()["type"])                                         # "observation"
    ws.send_json({"type": "close"})

# (b) MCP over HTTP (JSON-RPC 2.0)
tools = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()["result"]["tools"]
call = c.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "bank.balance", "arguments": {"account": me.account}}}).json()
print(call["result"]["structuredContent"]["result"]["balance"], call["result"]["isError"])

# (c) multi-world REST: operator creates and admits, actors act with their own token
w2 = c.post("/v1/worlds", json={"spec": spec, "world_id": "w-demo"}, headers=op).json()["world_id"]
carol = c.post(f"/v1/worlds/{w2}/join", json={"handle": "carol", "endowment": 500}, headers=op).json()
auth = {"Authorization": f"Bearer {carol['token']}"}
print(c.get(f"/v1/worlds/{w2}/observe", headers=auth).json()["text"][:80])
print(c.post(f"/v1/worlds/{w2}/act", json={"tool": "world.whoami", "args": {}}, headers=auth).json()["result"])
c.post(f"/v1/worlds/{w2}/advance", json={"dt": 600}, headers=op)
sid = c.post(f"/v1/worlds/{w2}/snapshot", headers=op).json()["snapshot_id"]
print(c.post(f"/v1/snapshots/{sid}/fork", json={}, headers=op).json()["world_id"])
```

### (a) OpenEnv-compatible routes (one default world/actor binding)

| Route | Request | Response |
|---|---|---|
| `POST /reset` | `{seed?, episode_id?, options?}` | `{"observation": {...}, "reward": null, "done": false}` |
| `POST /step` | `{action: {tool, args} \| {text: "<model output>"}, timeout_s?, request_id?}` | `{"observation": {...}, "reward": float, "done": bool}` |
| `GET /state` | – | `{episode_id, step_count, world_id, principal, time, seq, episode, api_version}` |
| `GET /health` | – | `{"status": "healthy"}` |
| `GET /metadata`, `GET /schema` | – | name/version/binding; `{action, observation, state}` JSON schemas |
| `WS /ws` | `{"type": "reset"\|"step"\|"state"\|"close", "data": {...}}` | `{"type": "observation"\|"state"\|"error", "data": {...}}`; error codes `INVALID_JSON`, `UNKNOWN_TYPE`, `VALIDATION_ERROR`, `EXECUTION_ERROR` |

The observation object holds `text, time, world_id, principal, seq, events, dropped_events, tools_changed, tools,
action_result, metadata{episode_id, episode, step, objective, terminated, truncated}` (no `reward`/`done`, which
are top-level). `done = terminated or truncated`. `timeout_s`/`request_id` are accepted and ignored (actions are
simulated, not wall-clock bound). For a WS `step`, `data` is the action itself or `{"action": {...}}`.
`make_app(..., ws_env_factory=...)` gives each WebSocket session its own environment.

### (b) MCP over HTTP: `POST /mcp`

JSON-RPC 2.0 with `initialize`, `ping`, `tools/list`, `tools/call` (+ client notifications, answered with 202).
Without a bearer token the default binding's actor is used; with `Authorization: Bearer <actor token>` plus
`X-Crucible-World: <world>` (or `?world=`) any actor can use it. HTTP cannot push notifications here, so a
`tools/call` that changed the tool list carries `_meta: {"crucible/toolsListChanged": true}` in its result.

### (c) REST API `/v1`

Authorization: `Bearer <actor token>` for actor routes; `Bearer <operator token>` (configured at startup; operator
routes return 403 when none is configured) for operator routes.

| Route | Who |
|---|---|
| `GET /v1/version` | anyone |
| `POST /v1/worlds` `{spec \| config, seed?, world_id?}`, `GET /v1/worlds` | operator |
| `GET /v1/worlds/{w}` | operator or an actor of `w` |
| `POST /v1/worlds/{w}/load`, `/join` `{handle, kind?, endowment?, name?, password?, controller?, info?}` | operator |
| `GET /v1/worlds/{w}/observe?max_events&include_tools&peek`, `GET /v1/worlds/{w}/tools`, `POST /v1/worlds/{w}/act` | actor |
| `POST /v1/worlds/{w}/advance` `{until? \| dt?}`, `/snapshot`, `/evaluate` `{context?}`, `DELETE /v1/worlds/{w}` | operator |
| `POST /v1/snapshots/{s}/fork` `{world_id?, inherit_tokens?}`, `POST /v1/snapshots/{s}/restore` `{name?}` | operator |
| `GET /v1/worlds/{w}/events?from_seq&to_seq` | operator: full log; actor: events it may see |

Errors: `{"error": {"code", "message"}}` with 400/401/403/404/409/422.

## 7. stdio MCP server

```bash
python -m crucible.api.mcp_server --store sqlite:///worlds.db --world w-1234 --token cru_...   # or $CRUCIBLE_TOKEN
#   --import my_pkg.my_modules   load custom world modules   --name-style underscores   bank__transfer
#   --advance-dt 60              move the clock after each call --no-observe-tool
```

Client configuration (e.g. an `mcpServers` entry):

```json
{"mcpServers": {"crucible": {"command": "python",
  "args": ["-m", "crucible.api.mcp_server", "--store", "sqlite:///worlds.db", "--world", "w-1234"],
  "env": {"CRUCIBLE_TOKEN": "cru_...", "PYTHONPATH": "crucible/src"}}}}
```

One JSON-RPC object per line. `initialize` answers `capabilities: {"tools": {"listChanged": true}}`; tool names are
Crucible tool names; `tools/call` returns `{content: [{type: "text", text: <JSON of the ActionResult>}],
structuredContent: <ActionResult>, isError: !ok}`; `_meta` on requests is ignored. Besides the world's tools there
is `crucible.observe` (the rendered observation). Whenever a call changes the actor's tool list (for example
`world.describe {"host": "shop"}` makes an agent-deployed host known), the server writes
`{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}` right after the call's response.

## 8. Security and concurrency

* Tokens (`cru_` + 32 random URL-safe characters) are shown once, at issue. The store keeps only SHA-256 hashes
  (`crucible-api/tokens/<world>` blob). A token authenticates one principal in one world and in forks/restores of
  it (`inherit_tokens`). `revoke_token` removes it.
* Actors can only act as themselves; `admin.*` tools are hidden from them and rejected by the kernel.
* The HTTP server binds `127.0.0.1` by default; there is no TLS. Put it behind an authenticating proxy before
  exposing it.
* A `WorldServer` takes a writer lease on each world it loads (`Store.acquire`, renewed on writes, released on
  close); a second server on the same world gets `ConflictError`. A crashed server's lease blocks others until
  it expires (`lease_ttl_s`, CLI `--lease-ttl`, default 600 s). Within a server one re-entrant lock serialises
  all calls, so the FastAPI threadpool is safe. Several agents sharing one live world should therefore use one
  server (HTTP `/v1` or `/mcp`), not one stdio MCP process each.

## 9. Limitations

* Observations scan at most `max_scan` log entries behind the cursor (`unscanned` reports the rest); events from
  before a fork point are not shown in the fork's observations (they are in `replay_events`).
* `restore_world` aliases are held per store in one blob; concurrent restores from different servers race.
* MCP over HTTP is plain JSON request/response (no SSE stream, no `Mcp-Session-Id`); list changes are reported via
  `_meta`. Pagination (`cursor`) of `tools/list` is not implemented (all tools are returned).
* `timeout_s` on OpenEnv `/step` is ignored; long-running world actions are bounded by the kernel, not the API.
