"""HTTP server for Crucible worlds (FastAPI). Three surfaces on one app:

(a) **OpenEnv-compatible** single-binding environment (one default `CrucibleEnv`: world + actor):
    ``POST /reset`` ``{seed?, episode_id?}`` and ``POST /step`` ``{action: {tool, args} | {text}, timeout_s?,
    request_id?}`` return ``{"observation": {...}, "reward": float|null, "done": bool}``; ``GET /state``,
    ``GET /health``, ``GET /metadata``, ``GET /schema``; WebSocket ``/ws`` with ``{"type": "reset"|"step"|"state"|
    "close", "data": {...}}`` -> ``{"type": "observation"|"state"|"error", "data": {...}}``.
(b) **MCP over HTTP**: ``POST /mcp`` (JSON-RPC 2.0: initialize, ping, tools/list, tools/call). The actor is the
    bearer token's actor (world from header ``X-Crucible-World`` or ``?world=``), else the default binding's.
(c) **Multi-world REST API** under ``/v1`` with per-actor bearer tokens; operator endpoints require the operator
    token given to `make_app` (they are disabled when none is configured).

`serve(...)` binds 127.0.0.1 by default. CLI: ``python -m crucible.api.http_server --store sqlite:///w.db``.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from typing import Any, Callable

from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from . import render as R
from .gym_env import CrucibleEnv
from .local import API_VERSION, ActorIdentity, ApiError, AuthError, ForbiddenError, WorldServer
from .mcp_server import MCPHandler

ACTION_SCHEMA = {
    "title": "CrucibleAction", "type": "object",
    "properties": {"tool": {"type": "string", "description": "tool name, e.g. bank.transfer"},
                   "args": {"type": "object", "description": "arguments (see the tool's inputSchema)"},
                   "idempotency_key": {"type": "string"},
                   "text": {"type": "string", "description": "alternatively: raw model output containing a JSON "
                                                             "tool call"},
                   "metadata": {"type": "object"}},
    "additionalProperties": False,
}
OBSERVATION_SCHEMA = {
    "title": "CrucibleObservation", "type": "object",
    "properties": {"text": {"type": "string"}, "time": {"type": "integer"}, "world_id": {"type": "string"},
                   "principal": {"type": "string"}, "seq": {"type": "integer"},
                   "events": {"type": "array", "items": {"type": "object"}}, "dropped_events": {"type": "integer"},
                   "tools_changed": {"type": "boolean"}, "tools": {"type": ["array", "null"]},
                   "action_result": {"type": ["object", "null"]}, "done": {"type": "boolean"},
                   "reward": {"type": ["number", "null"]}, "metadata": {"type": "object"}},
}
STATE_SCHEMA = {
    "title": "CrucibleState", "type": "object",
    "properties": {"episode_id": {"type": ["string", "null"]}, "step_count": {"type": "integer"},
                   "world_id": {"type": "string"}, "principal": {"type": "string"}, "time": {"type": "integer"},
                   "seq": {"type": "integer"}, "episode": {"type": "integer"}, "api_version": {"type": "string"}},
}


class _Binding:
    """The default (world, actor) environment behind the OpenEnv routes."""

    def __init__(self, env: CrucibleEnv):
        self.env = env
        self.lock = threading.RLock()
        self.episode_id: str | None = None
        self.started = False

    def _payload(self, obs: dict, info: dict, reward: float | None, done: bool, extra: dict | None = None) -> dict:
        o = {"text": obs["text"], "time": int(obs["time"][0]), "world_id": info.get("world_id"),
             "principal": info.get("principal"), "seq": info.get("seq"), "events": info.get("events", []),
             "dropped_events": info.get("dropped_events", 0), "tools_changed": info.get("tools_changed", False),
             "tools": info.get("available_tools"), "action_result": info.get("action_result"),
             "metadata": {"episode_id": self.episode_id, "episode": info.get("episode"), "step": info.get("step"),
                          "objective": info.get("objective", {}), **(extra or {})}}
        return {"observation": o, "reward": reward, "done": done}

    def reset(self, seed: int | None = None, episode_id: str | None = None, options: dict | None = None) -> dict:
        with self.lock:
            obs, info = self.env.reset(seed=seed, options=options)
            self.started = True
            self.episode_id = episode_id or f"ep-{self.env.episode}"
            return self._payload(obs, info, None, False)

    def step(self, action: Any) -> dict:
        with self.lock:
            if not self.started:
                self.reset()
            if isinstance(action, dict):
                action = {k: v for k, v in action.items() if k != "metadata"}
                if "text" in action and "tool" not in action:
                    action = action["text"]
            obs, reward, term, trunc, info = self.env.step(action)
            return self._payload(obs, info, float(reward), bool(term or trunc),
                                 {"terminated": term, "truncated": trunc})

    def state(self) -> dict:
        with self.lock:
            e = self.env
            wid = e.current_world_id or e.server.resolve(e.world_id)
            w = e.server.world(wid)
            return {"episode_id": self.episode_id, "step_count": e.steps, "world_id": wid, "principal": e.principal,
                    "time": w.now, "seq": w.seq, "episode": e.episode, "api_version": API_VERSION}


def _bearer(request) -> str | None:
    h = request.headers.get("authorization") or ""
    if h.lower().startswith("bearer "):
        return h[7:].strip() or None
    return None


def make_app(server: WorldServer, *, operator_token: str | None = None, env: CrucibleEnv | None = None,
             env_kwargs: dict | None = None, ws_env_factory: Callable[[], CrucibleEnv] | None = None,
             title: str = "Crucible") -> FastAPI:
    """Build the FastAPI app. `env` (or `env_kwargs` for a `CrucibleEnv(server=server, **env_kwargs)`) is the
    default OpenEnv/MCP binding; `ws_env_factory` gives every WebSocket session its own env instead."""
    if env is None and env_kwargs is not None:
        env = CrucibleEnv(server=server, **env_kwargs)
    binding = _Binding(env) if env is not None else None
    app = FastAPI(title=title, version=API_VERSION)
    app.state.server = server
    app.state.binding = binding
    mcp_handlers: dict[tuple[str, str], MCPHandler] = {}
    mcp_lock = threading.Lock()

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse({"error": exc.as_dict()}, status_code=exc.status)

    def need_binding() -> _Binding:
        if binding is None:
            raise ApiError("no default environment binding configured (make_app(env=...))", code="no_binding",
                           status=409)
        return binding

    def require_operator(request: Request) -> None:
        if not operator_token:
            raise ForbiddenError("operator API disabled (no operator token configured)")
        tok = _bearer(request)
        if tok is None:
            raise AuthError("operator token required")
        if not hmac.compare_digest(tok.encode(), operator_token.encode()):
            raise ForbiddenError("not the operator token")

    def is_operator(request: Request) -> bool:
        tok = _bearer(request)
        return bool(operator_token and tok and hmac.compare_digest(tok.encode(), operator_token.encode()))

    def actor_token(request: Request) -> str:
        tok = _bearer(request)
        if tok is None:
            raise AuthError("actor bearer token required")
        return tok

    # ================================================================== (a) OpenEnv
    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/metadata")
    def metadata():
        b = binding
        return {"name": "crucible", "description": "Crucible: a persistent, event-sourced synthetic internet. "
                "Actions are tool calls; the world persists across episodes.", "version": API_VERSION,
                "api_version": API_VERSION, "action": "tool call {tool, args}",
                "world_id": b.env.world_id if b else None, "principal": b.env.principal if b else None,
                "reset_mode": b.env.reset_mode if b else None}

    @app.get("/schema")
    def schema():
        return {"action": ACTION_SCHEMA, "observation": OBSERVATION_SCHEMA, "state": STATE_SCHEMA}

    @app.post("/reset")
    def reset(body: dict | None = Body(default=None)):
        body = body or {}
        seed = body.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ApiError("seed must be an integer", code="validation_error", status=422)
        return need_binding().reset(seed, body.get("episode_id"), body.get("options"))

    @app.post("/step")
    def step(body: dict = Body(...)):
        action = body.get("action")
        if not isinstance(action, (dict, str)):
            raise ApiError("body must contain 'action' (object)", code="validation_error", status=422)
        return need_binding().step(action)

    @app.get("/state")
    def state():
        return need_binding().state()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        b = _Binding(await run_in_threadpool(ws_env_factory)) if ws_env_factory else binding

        async def err(code: str, message: str):
            await websocket.send_json({"type": "error", "data": {"code": code, "message": message}})
        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    msg = json.loads(raw)
                except ValueError as e:
                    await err("INVALID_JSON", f"invalid JSON: {e}")
                    continue
                if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
                    await err("VALIDATION_ERROR", "message must be an object with a string 'type'")
                    continue
                kind, data = msg["type"], msg.get("data") or {}
                if kind == "close":
                    await websocket.close()
                    break
                if kind not in ("reset", "step", "state"):
                    await err("UNKNOWN_TYPE", f"unknown message type {kind!r}")
                    continue
                if b is None:
                    await err("EXECUTION_ERROR", "no default environment binding configured")
                    continue
                if not isinstance(data, dict):
                    await err("VALIDATION_ERROR", "'data' must be an object")
                    continue
                try:
                    if kind == "reset":
                        seed = data.get("seed")
                        if seed is not None and not isinstance(seed, int):
                            await err("VALIDATION_ERROR", "seed must be an integer")
                            continue
                        out = await run_in_threadpool(b.reset, seed, data.get("episode_id"), data.get("options"))
                        await websocket.send_json({"type": "observation", "data": out})
                    elif kind == "step":
                        action = data.get("action", data)
                        if not isinstance(action, (dict, str)) or (isinstance(action, dict) and not action):
                            await err("VALIDATION_ERROR", "step needs an action object")
                            continue
                        out = await run_in_threadpool(b.step, action)
                        await websocket.send_json({"type": "observation", "data": out})
                    else:
                        await websocket.send_json({"type": "state", "data": await run_in_threadpool(b.state)})
                except Exception as e:  # noqa: BLE001
                    await err("EXECUTION_ERROR", f"{type(e).__name__}: {e}")
        finally:
            if ws_env_factory and b is not None:
                await run_in_threadpool(b.env.close)

    # ================================================================== (b) MCP over HTTP
    def mcp_handler(request: Request) -> MCPHandler:
        tok = _bearer(request)
        if tok and not is_operator(request):
            wid = request.headers.get("x-crucible-world") or request.query_params.get("world")
            if not wid:
                raise ApiError("MCP with a bearer token needs the world (X-Crucible-World header or ?world=)",
                               status=400)
            ident = server.authenticate(wid, tok)
            key = (ident.world_id, tok)
            with mcp_lock:
                h = mcp_handlers.get(key)
                if h is None:
                    h = mcp_handlers[key] = MCPHandler(server, wid, tok)
            return h
        b = need_binding()
        key = ("__binding__", "")
        with mcp_lock:
            h = mcp_handlers.get(key)
            if h is None:
                h = mcp_handlers[key] = MCPHandler(
                    server, None, lambda: ActorIdentity(b.env.current_world_id or b.env.world_id, b.env.principal,
                                                        b.env.identity.token))
            return h

    @app.post("/mcp")
    async def mcp(request: Request):
        raw = (await request.body()).decode("utf-8", "replace")
        h = mcp_handler(request)
        msgs = await run_in_threadpool(h.handle_raw, raw)
        replies = [m for m in msgs if isinstance(m, list) or "id" in m]
        if not replies:
            return Response(status_code=202)
        return JSONResponse(replies[0])

    # ================================================================== (c) /v1 multi-world REST
    @app.get("/v1/version")
    def v1_version():
        return {"api_version": API_VERSION, "store": server.store_url}

    @app.post("/v1/worlds")
    def v1_create(request: Request, body: dict = Body(...)):
        require_operator(request)
        cfg = body.get("spec") or body.get("config")
        if cfg is None:
            raise ApiError("body needs 'spec' (WorldSpec dict) or 'config' (world-generator config)", status=422)
        wid = server.create_world(cfg, body.get("seed"), world_id=body.get("world_id"))
        return server.world_info(wid)

    @app.get("/v1/worlds")
    def v1_list(request: Request):
        require_operator(request)
        return {"worlds": server.list_worlds()}

    @app.get("/v1/worlds/{world_id}")
    def v1_info(world_id: str, request: Request):
        if not is_operator(request):
            server.authenticate(world_id, actor_token(request))
        return server.world_info(world_id)

    @app.post("/v1/worlds/{world_id}/load")
    def v1_load(world_id: str, request: Request):
        require_operator(request)
        return server.load_world(world_id)

    @app.post("/v1/worlds/{world_id}/join")
    def v1_join(world_id: str, request: Request, body: dict = Body(...)):
        require_operator(request)
        if not isinstance(body.get("handle"), str):
            raise ApiError("body needs 'handle'", status=422)
        ident = server.join_world(world_id, body["handle"], body.get("kind", "agent"), int(body.get("endowment", 0)),
                                  name=body.get("name"), password=body.get("password"),
                                  controller=body.get("controller", ""), info=body.get("info"))
        return ident.to_dict()

    @app.get("/v1/worlds/{world_id}/observe")
    def v1_observe(world_id: str, request: Request, max_events: int | None = None, include_tools: str = "changed",
                   peek: bool = False):
        return server.observe(world_id, actor_token(request), max_events=max_events, include_tools=include_tools,
                              peek=peek).to_dict()

    @app.get("/v1/worlds/{world_id}/tools")
    def v1_tools(world_id: str, request: Request):
        return {"tools": server.available_tools(world_id, actor_token(request))}

    @app.post("/v1/worlds/{world_id}/act")
    def v1_act(world_id: str, request: Request, body: Any = Body(...)):
        if isinstance(body, dict) and "action" in body and "tool" not in body:
            body = body["action"]
        return server.execute_action(world_id, actor_token(request), body)

    @app.post("/v1/worlds/{world_id}/advance")
    def v1_advance(world_id: str, request: Request, body: dict = Body(...)):
        require_operator(request)
        return server.advance_time(world_id, body.get("until"), dt=body.get("dt"), max_jobs=body.get("max_jobs"))

    @app.post("/v1/worlds/{world_id}/snapshot")
    def v1_snapshot(world_id: str, request: Request):
        require_operator(request)
        return {"snapshot_id": server.snapshot_world(world_id)}

    @app.post("/v1/snapshots/{snapshot_id}/fork")
    def v1_fork(snapshot_id: str, request: Request, body: dict | None = Body(default=None)):
        require_operator(request)
        body = body or {}
        wid = server.fork_world(snapshot_id, body.get("world_id"), inherit_tokens=body.get("inherit_tokens", True))
        return server.world_info(wid)

    @app.post("/v1/snapshots/{snapshot_id}/restore")
    def v1_restore(snapshot_id: str, request: Request, body: dict | None = Body(default=None)):
        require_operator(request)
        body = body or {}
        return {"world_id": server.restore_world(snapshot_id, body.get("name"))}

    @app.get("/v1/worlds/{world_id}/events")
    def v1_events(world_id: str, request: Request, from_seq: int = 0, to_seq: int | None = None):
        actor = None if is_operator(request) else actor_token(request)
        return {"events": server.replay_events(world_id, from_seq, to_seq, actor=actor)}

    @app.post("/v1/worlds/{world_id}/evaluate")
    def v1_evaluate(world_id: str, request: Request, body: dict | None = Body(default=None)):
        require_operator(request)
        return server.evaluate_world(world_id, (body or {}).get("context")).to_dict()

    @app.delete("/v1/worlds/{world_id}")
    def v1_close(world_id: str, request: Request):
        require_operator(request)
        server.close_world(world_id)
        return {"closed": world_id}

    @app.get("/v1/render/help")
    def v1_help():
        return {"action_format": R.action_format_help()}

    return app


def serve(server: WorldServer, *, host: str = "127.0.0.1", port: int = 8000, **app_kwargs) -> None:
    """Run the app with uvicorn (loopback only unless `host` says otherwise)."""
    import uvicorn
    try:
        uvicorn.run(make_app(server, **app_kwargs), host=host, port=port, log_level="info")
    finally:
        server.close()  # flush, persist sessions, release writer leases


def main(argv: list[str] | None = None) -> int:
    from ..hosting import engage
    engage("api.http_server")                  # Part 16: no real-world bridge in a process that hosts worlds
    ap = argparse.ArgumentParser(prog="python -m crucible.api.http_server")
    ap.add_argument("--store", default="sqlite:///crucible.db")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--operator-token-env", default="CRUCIBLE_OPERATOR_TOKEN",
                    help="environment variable holding the operator token (unset: operator API disabled)")
    ap.add_argument("--world", help="default binding: world id or name (OpenEnv / MCP routes)")
    ap.add_argument("--token-env", default="CRUCIBLE_TOKEN", help="env var with the default binding's actor token")
    ap.add_argument("--dt", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--reset-mode", choices=["continue", "fork"], default="continue")
    ap.add_argument("--lease-ttl", type=float, default=600.0,
                    help="writer lease seconds (0 disables; a crashed server's lease blocks others until expiry)")
    a = ap.parse_args(argv)
    server = WorldServer(a.store, lease_ttl_s=a.lease_ttl or None)
    env = None
    if a.world:
        tok = os.environ.get(a.token_env)
        if not tok:
            ap.error(f"--world needs an actor token in ${a.token_env}")
        env = CrucibleEnv(server=server, world_id=a.world, actor=tok, dt=a.dt, max_steps=a.max_steps,
                          reset_mode=a.reset_mode)
    serve(server, host=a.host, port=a.port, operator_token=os.environ.get(a.operator_token_env), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
