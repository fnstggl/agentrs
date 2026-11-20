"""Model Context Protocol (MCP) server for one actor in one world.

`MCPHandler` implements the JSON-RPC 2.0 methods ``initialize``, ``ping``, ``tools/list`` and ``tools/call``
(plus the ``notifications/*`` a client may send) for a fixed (world, actor token) binding. It is shared by:

* the stdio transport in this file (one JSON object per line on stdin/stdout; logs go to stderr), and
* the HTTP transport ``POST /mcp`` in `http_server`.

Tool names are Crucible tool names (``bank.transfer``); with ``name_style="underscores"`` they are exposed as
``bank__transfer`` for clients that do not accept dots. One extra tool, ``crucible.observe``, returns the actor's
observation (new events, status, tool changes) unless disabled. After a ``tools/call`` that changed the actor's
tool list (e.g. it discovered a new service host), the stdio server emits ``notifications/tools/list_changed``;
over HTTP the result carries ``_meta["crucible/toolsListChanged"] = true`` instead.

Run:  ``python -m crucible.api.mcp_server --store sqlite:///world.db --world W --token cru_...``
(the token may instead be given in ``$CRUCIBLE_TOKEN``; ``--import pkg.mod`` loads extra world modules).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable

from . import render as R
from .local import API_VERSION, ActorIdentity, ApiError, WorldServer

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
OBSERVE_TOOL = "crucible.observe"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


def _resp(mid: Any, result: Any = None, error: dict | None = None) -> dict:
    d: dict = {"jsonrpc": "2.0", "id": mid}
    if error is not None:
        d["error"] = error
    else:
        d["result"] = result
    return d


def _error(code: int, message: str, data: Any = None) -> dict:
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return e


class MCPHandler:
    """JSON-RPC/MCP request handler bound to one actor. `handle(msg)` returns (response | None, notifications)."""

    def __init__(self, server: WorldServer, world_id: str, actor: ActorIdentity | str | Callable[[], Any], *,
                 name_style: str = "dots", observe_tool: bool = True, advance_dt: int = 0,
                 server_name: str = "crucible"):
        if name_style not in ("dots", "underscores"):
            raise ValueError("name_style must be 'dots' or 'underscores'")
        self.server = server
        self.world_id = world_id
        self._actor_src = actor
        self.name_style = name_style
        self.observe_tool = observe_tool
        self.advance_dt = int(advance_dt)
        self.server_name = server_name
        self.initialized = False
        self._tool_names: list[str] | None = None

    # ------------------------------------------------------------------ binding
    @property
    def actor(self) -> ActorIdentity | str:
        a = self._actor_src
        return a() if callable(a) else a

    def _wid(self) -> str:
        a = self.actor
        return a.world_id if isinstance(a, ActorIdentity) and self.world_id is None else self.world_id

    def _ext(self, name: str) -> str:
        return name.replace(".", "__") if self.name_style == "underscores" else name

    def _int(self, name: str) -> str:
        return name.replace("__", ".") if self.name_style == "underscores" else name

    def _crucible_tools(self) -> list[dict]:
        return self.server.available_tools(self._wid(), self.actor)

    def mcp_tools(self) -> list[dict]:
        out = []
        tools = self._crucible_tools()
        self._tool_names = [t["name"] for t in tools]
        for t in tools:
            ann = t.get("annotations") or {}
            out.append({"name": self._ext(t["name"]), "description": t.get("description") or t["name"],
                        "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}},
                        "annotations": {"readOnlyHint": bool(ann.get("readOnlyHint"))},
                        "_meta": {"crucible/costSeconds": ann.get("costSeconds", 0),
                                  "crucible/host": ann.get("host"), "crucible/builtin": ann.get("builtin", True)}})
        if self.observe_tool:
            out.append({"name": self._ext(OBSERVE_TOOL),
                        "description": "Your observation: world time, new events you can see since you last "
                                       "observed (messages, payments, ...), account summaries and tool changes.",
                        "inputSchema": {"type": "object", "properties": {
                            "max_events": {"type": "integer", "minimum": 0, "maximum": 500}},
                            "additionalProperties": False},
                        "annotations": {"readOnlyHint": True}})
        return out

    def tools_changed(self) -> bool:
        """True when the actor's tool list differs from what this handler last reported (and records it)."""
        names = [t["name"] for t in self._crucible_tools()]
        changed = self._tool_names is not None and names != self._tool_names
        self._tool_names = names
        return changed

    # ------------------------------------------------------------------ dispatch
    def handle(self, msg: Any) -> tuple[dict | None, list[dict]]:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            mid = msg.get("id") if isinstance(msg, dict) else None
            return _resp(mid, error=_error(INVALID_REQUEST, "invalid JSON-RPC 2.0 request")), []
        method = msg["method"]
        mid = msg.get("id")
        is_note = "id" not in msg
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return (None if is_note else _resp(mid, error=_error(INVALID_PARAMS, "params must be an object"))), []
        params = {k: v for k, v in params.items() if k != "_meta"}
        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self.initialized = True
            return None, []
        try:
            fn = {"initialize": self._initialize, "ping": lambda p: ({}, []), "tools/list": self._tools_list,
                  "tools/call": self._tools_call}.get(method)
            if fn is None:
                return (None if is_note else _resp(mid, error=_error(METHOD_NOT_FOUND, f"unknown method {method}"))), []
            result, notes = fn(params)
        except ApiError as e:
            return (None if is_note else _resp(mid, error=_error(INVALID_PARAMS if e.status < 500 else INTERNAL_ERROR,
                                                                 e.message, e.as_dict()))), []
        except _ParamError as e:
            return (None if is_note else _resp(mid, error=_error(INVALID_PARAMS, str(e)))), []
        except Exception as e:  # noqa: BLE001
            return (None if is_note else _resp(mid, error=_error(INTERNAL_ERROR, f"{type(e).__name__}: {e}"))), []
        return (None if is_note else _resp(mid, result)), notes

    def _initialize(self, params: dict):
        want = params.get("protocolVersion")
        version = want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        ident = self.server.authenticate(self._wid(), self.actor)
        return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": self.server_name, "version": API_VERSION},
                "instructions": f"You are {ident.principal} in the persistent Crucible world {ident.world_id}. "
                                "Every tool call is an action in that world and takes simulated time. Call "
                                f"{self._ext(OBSERVE_TOOL)} to see what happened since you last looked."}, []

    def _tools_list(self, params: dict):
        return {"tools": self.mcp_tools()}, []

    def _tools_call(self, params: dict):
        name = params.get("name")
        args = params.get("arguments")
        if not isinstance(name, str):
            raise _ParamError("tools/call needs a string 'name'")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise _ParamError("'arguments' must be an object")
        if self._tool_names is None:
            self._tool_names = [t["name"] for t in self._crucible_tools()]
        tool = self._int(name)
        if self.observe_tool and tool == OBSERVE_TOOL:
            o = self.server.observe(self._wid(), self.actor, max_events=args.get("max_events"))
            d = o.to_dict()
            d.pop("text", None)
            return {"content": [{"type": "text", "text": o.text}], "structuredContent": d, "isError": False}, []
        res = self.server.execute_action(self._wid(), self.actor, {"tool": tool, "args": args})
        if self.advance_dt > 0:
            self.server.advance_time(self._wid(), dt=self.advance_dt)
        result = {"content": [{"type": "text", "text": R.compact_json(res, None)}], "structuredContent": res,
                  "isError": not res.get("ok", False)}
        notes = []
        if self.tools_changed():
            notes.append({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            result["_meta"] = {"crucible/toolsListChanged": True}
        return result, notes

    def handle_raw(self, line: str) -> list[dict]:
        """Parse one JSON text (object or batch array) and return every message to send back, in order."""
        try:
            msg = json.loads(line)
        except ValueError as e:
            return [_resp(None, error=_error(PARSE_ERROR, f"parse error: {e}"))]
        out: list[dict] = []
        notes: list[dict] = []
        if isinstance(msg, list):
            if not msg:
                return [_resp(None, error=_error(INVALID_REQUEST, "empty batch"))]
            batch = []
            for m in msg:
                r, n = self.handle(m)
                if r is not None:
                    batch.append(r)
                notes.extend(n)
            return ([batch] if batch else []) + notes  # type: ignore[list-item]
        r, n = self.handle(msg)
        if r is not None:
            out.append(r)
        return out + n


class _ParamError(Exception):
    pass


def serve_stdio(handler: MCPHandler, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        for m in handler.handle_raw(line):
            stdout.write(json.dumps(m, separators=(",", ":")) + "\n")
        stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.api.mcp_server", description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="store url: sqlite:///path.db, postgresql://..., memory://")
    ap.add_argument("--world", required=True, help="world id or name")
    ap.add_argument("--token", default=os.environ.get("CRUCIBLE_TOKEN"), help="actor token (or $CRUCIBLE_TOKEN)")
    ap.add_argument("--import", dest="imports", action="append", default=[],
                    help="extra Python module to import (registers custom world modules); repeatable")
    ap.add_argument("--name-style", choices=["dots", "underscores"], default="dots")
    ap.add_argument("--advance-dt", type=int, default=0, help="advance the clock this many seconds per call")
    ap.add_argument("--no-observe-tool", action="store_true")
    ap.add_argument("--lease-ttl", type=float, default=600.0, help="writer lease seconds (0 disables)")
    a = ap.parse_args(argv)
    if not a.token:
        ap.error("--token or $CRUCIBLE_TOKEN is required")
    for m in a.imports:
        importlib.import_module(m)
    server = WorldServer(a.store, lease_ttl_s=a.lease_ttl or None)
    try:
        ident = server.authenticate(a.world, a.token)
        print(f"crucible mcp: {ident.principal} in {ident.world_id}", file=sys.stderr, flush=True)
        handler = MCPHandler(server, a.world, a.token, name_style=a.name_style,
                             observe_tool=not a.no_observe_tool, advance_dt=a.advance_dt)
        serve_stdio(handler)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
