"""STAGING ONLY — local imitation of the part of the Discord Gateway that ``cam_live.harness.discord_gateway`` uses.

A ``websockets.sync.server`` on a loopback port that speaks, with Discord's payload shapes::

    -> Hello (opcode 10, heartbeat_interval)
    <- Heartbeat (opcode 1)            -> Heartbeat ACK (opcode 11)
    <- Identify (opcode 2)             -> Ready (Dispatch, opcode 0), then Guild Create for every guild of the bot
    -> Dispatch events for what happens in the staging state, filtered by the intents of the session

and closes with Discord's close codes: 4003 (payload before Identify), 4004 (wrong token), 4005 (second
Identify), 4013 (undefined intent bits), 4014 (privileged intent that the staging application has not enabled).

Test hooks (plain methods): :meth:`emit` / :meth:`emit_other_guild`, :meth:`send_op` (Reconnect 7, Invalid
Session 9, heartbeat request 1), :meth:`close_sessions`, ``identify_response`` (answer the next Identify with
opcode 9 or 7 instead of Ready), ``acknowledge_heartbeats``, :meth:`on_ready` and :meth:`wait_ready`.

The token a client sends is compared and discarded: it is never stored in ``identify_log`` or anywhere else.
This is a staging service. Results obtained against it are never live results.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
from typing import Any, Callable, Deque, Dict, List, Optional

from ..discord_perms import INTENT_BITS, PRIVILEGED_INTENTS, granted_intents

#: event name prefix -> the intent a session needs to be sent it (from the documented list of intents)
_EVENT_INTENT = (("GUILD_MEMBER_", "GUILD_MEMBERS"), ("GUILD_SCHEDULED_EVENT_", "GUILD_SCHEDULED_EVENTS"),
                 ("GUILD_BAN_", "GUILD_MODERATION"), ("GUILD_AUDIT_LOG_", "GUILD_MODERATION"),
                 ("INVITE_", "GUILD_INVITES"), ("MESSAGE_REACTION_", "GUILD_MESSAGE_REACTIONS"),
                 ("MESSAGE_", "GUILD_MESSAGES"), ("PRESENCE_", "GUILD_PRESENCES"), ("TYPING_", "GUILD_MESSAGE_TYPING"),
                 ("GUILD_", "GUILDS"), ("CHANNEL_", "GUILDS"), ("THREAD_", "GUILDS"))
_ALL_INTENT_BITS = 0
for _b in INTENT_BITS.values():
    _ALL_INTENT_BITS |= 1 << _b


#: the websocket library logs received frames at DEBUG level; Identify frames carry a token, so it gets no logger
_SILENT = logging.getLogger("cam_live.harness.staging_discord.gateway.transport")
_SILENT.addHandler(logging.NullHandler())
_SILENT.propagate = False
_SILENT.disabled = True


def intent_for(event: str) -> Optional[str]:
    for prefix, intent in _EVENT_INTENT:
        if event.startswith(prefix):
            return intent
    return None


class _Session:
    def __init__(self) -> None:
        self.outbox: Deque[Any] = collections.deque()
        self.identified = False
        self.intents = 0
        self.seq = 0
        self.closed = False


class StagingGateway:
    """Serves the staging Gateway for one :class:`StagingDiscord` state. ``url`` is what ``GET /gateway/bot``
    of the staging REST service returns."""

    environment = "staging"

    def __init__(self, state: Any, host: str = "127.0.0.1", port: int = 0, heartbeat_interval_ms: int = 45000):
        from websockets.sync.server import serve
        self.state = state
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.lock = threading.RLock()
        self.sessions: List[_Session] = []
        self.identify_log: List[Dict[str, Any]] = []          # intents and outcome of every Identify; never the token
        self.frames_received: List[Dict[str, Any]] = []       # opcode (and heartbeat sequence) of every client frame
        self.closes_received: List[Optional[int]] = []        # close code each client ended its connection with
        self.identify_response: Optional[str] = None          # "invalid_session" | "reconnect" for the next Identify
        self.acknowledge_heartbeats = True
        self._on_ready: List[Callable[[], None]] = []
        self._ready = threading.Event()
        self._server = serve(self._handle, host, port, ping_interval=None, close_timeout=1.0, logger=_SILENT,
                             server_header="cam-live-staging-discord-gateway")
        self.port = self._server.socket.getsockname()[1]
        self.url = f"ws://{host}:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="staging-discord-gateway")
        self._thread.start()

    # ---- test hooks --------------------------------------------------------------------------------------------
    def on_ready(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` once, right after the next session received Ready and its Guild Create events."""
        with self.lock:
            self._on_ready.append(fn)

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def emit(self, event: str, data: Dict[str, Any]) -> int:
        """Dispatch ``event`` to every session whose intents include it. Returns the number of sessions reached."""
        need = intent_for(event)
        n = 0
        with self.lock:
            for s in self.sessions:
                if s.closed or (need is not None and not s.intents >> INTENT_BITS[need] & 1):
                    continue
                d = json.loads(json.dumps(data))
                if event in ("MESSAGE_CREATE", "MESSAGE_UPDATE") and not s.intents >> INTENT_BITS["MESSAGE_CONTENT"] & 1 \
                        and (d.get("author") or {}).get("id") != self.state.bot_user["id"]:
                    d.update(content="", embeds=[], attachments=[], components=[])   # no message-content intent
                s.outbox.append({"op": 0, "t": event, "d": d})
                n += 1
        return n

    def emit_other_guild(self, event: str, data: Optional[Dict[str, Any]] = None, guild_id: str = "") -> int:
        """Dispatch an event that belongs to another guild (what a bot that is in several servers receives)."""
        gid = guild_id or (self.state.other_guilds[0]["id"] if self.state.other_guilds else "999000000000000001")
        d = dict(data or {})
        d["id" if event in ("GUILD_CREATE", "GUILD_UPDATE", "GUILD_DELETE") else "guild_id"] = gid
        return self.emit(event, d)

    def send_op(self, op: int, d: Any = None) -> None:
        """Send a non-dispatch payload to every session: 7 Reconnect, 9 Invalid Session, 1 heartbeat request."""
        with self.lock:
            for s in self.sessions:
                s.outbox.append({"op": op, "d": d})

    def close_sessions(self, code: int, reason: str = "") -> None:
        with self.lock:
            for s in self.sessions:
                s.outbox.append(("close", code, reason))

    def close(self) -> None:
        self._server.shutdown()

    # ---- one connection ----------------------------------------------------------------------------------------
    def _send(self, ws: Any, s: _Session, frame: Dict[str, Any]) -> None:
        if frame.get("op") == 0:
            s.seq += 1
            frame = dict(frame, s=s.seq)
        ws.send(json.dumps(frame))

    def _identify(self, ws: Any, s: _Session, d: Any) -> bool:
        st = self.state
        d = d if isinstance(d, dict) else {}
        intents = d.get("intents")
        entry = {"intents": intents, "token_ok": d.get("token") == st.token, "outcome": "",
                 "intent_names": sorted(n for n, b in INTENT_BITS.items() if isinstance(intents, int) and intents >> b & 1),
                 "properties": dict(d.get("properties") or {}), "ts": st.time()}
        with self.lock:
            self.identify_log.append(entry)
        with st.lock:
            st.session_starts_remaining = max(0, st.session_starts_remaining - 1)
        if s.identified:
            entry["outcome"] = "close:4005"
            ws.close(4005, "Already authenticated.")
            return False
        if not entry["token_ok"]:
            entry["outcome"] = "close:4004"
            ws.close(4004, "Authentication failed.")
            return False
        if not isinstance(intents, int) or isinstance(intents, bool) or intents & ~_ALL_INTENT_BITS:
            entry["outcome"] = "close:4013"
            ws.close(4013, "Invalid intent(s).")
            return False
        enabled = granted_intents(st.application["flags"])
        if any(intents >> INTENT_BITS[p] & 1 and not enabled[p] for p in PRIVILEGED_INTENTS):
            entry["outcome"] = "close:4014"
            ws.close(4014, "Disallowed intent(s).")
            return False
        with self.lock:
            scripted, self.identify_response = self.identify_response, None
        if scripted == "invalid_session":
            entry["outcome"] = "op:9"
            self._send(ws, s, {"op": 9, "d": False})
            return True
        if scripted == "reconnect":
            entry["outcome"] = "op:7"
            self._send(ws, s, {"op": 7, "d": None})
            return True
        s.identified, s.intents = True, intents
        entry["outcome"] = "ready"
        with st.lock:
            guild_ids = ([st.guild_id] if st.bot_in_guild else []) + [g["id"] for g in st.other_guilds]
            ready = {"v": 10, "user": dict(st.bot_user), "guilds": [{"id": g, "unavailable": True} for g in guild_ids],
                     "session_id": f"staging-session-{st._snowflake()}", "resume_gateway_url": self.url,
                     "application": {"id": st.application["id"], "flags": st.application["flags"]}}
            creates = ([st.gateway_guild_create()] if st.bot_in_guild else []) + [
                {"id": g["id"], "name": g["name"], "member_count": g.get("member_count", 1), "channels": [],
                 "members": [], "joined_at": st.members[st.bot_user["id"]]["joined_at"], "large": False}
                for g in st.other_guilds]
        self._send(ws, s, {"op": 0, "t": "READY", "d": ready})
        if intents >> INTENT_BITS["GUILDS"] & 1:
            for g in creates:
                self._send(ws, s, {"op": 0, "t": "GUILD_CREATE", "d": g})
        with self.lock:
            self.sessions.append(s)
            callbacks, self._on_ready = self._on_ready, []
        self._ready.set()
        for fn in callbacks:
            fn()
        return True

    def _handle(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed
        s = _Session()
        try:
            ws.send(json.dumps({"op": 10, "d": {"heartbeat_interval": self.heartbeat_interval_ms}}))
            while True:
                while s.outbox:
                    frame = s.outbox.popleft()
                    if isinstance(frame, tuple):
                        ws.close(frame[1], frame[2])
                        return
                    self._send(ws, s, frame)
                try:
                    raw = ws.recv(timeout=0.01)
                except TimeoutError:
                    continue
                try:
                    msg = json.loads(raw)
                    op = msg.get("op")
                except (ValueError, AttributeError):
                    ws.close(4002, "Error while decoding payload.")
                    return
                with self.lock:
                    self.frames_received.append({"op": op, "d": msg.get("d") if op == 1 else None})
                if op == 1:
                    if self.acknowledge_heartbeats:
                        ws.send(json.dumps({"op": 11}))
                elif op == 2:
                    if not self._identify(ws, s, msg.get("d")):
                        return
                elif not s.identified:
                    ws.close(4003, "Not authenticated.")
                    return
                else:
                    ws.close(4001, "Unknown opcode.")
                    return
        except ConnectionClosed as e:
            rcvd = getattr(e, "rcvd", None)
            with self.lock:
                self.closes_received.append(getattr(rcvd, "code", None))
        finally:
            s.closed = True
            with self.lock:
                if s in self.sessions:
                    self.sessions.remove(s)
                if not self.sessions:
                    self._ready.clear()


__all__ = ["StagingGateway", "intent_for"]
