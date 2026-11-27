"""cam_live.harness.discord_gateway — one bounded, receive-only window on the Discord Gateway.

Owned by workstream ES. Protocol facts are transcribed from the official API documentation
(github.com/discord/discord-api-docs, commit 62ab2d5, read 2026-11-27): ``developers/events/gateway.mdx``
(connection lifecycle, heartbeats, Identify, intents), ``developers/events/gateway-events.mdx`` (payloads) and
``developers/topics/opcodes-and-status-codes.mdx`` (opcodes, close event codes).

What :class:`GatewayWindow` does, in the documented order: connect to the URL of ``GET /gateway/bot`` with
``v`` and ``encoding=json``; receive Hello (opcode 10); start heartbeating (first heartbeat after
``heartbeat_interval * jitter``, then every interval, each carrying the last sequence number, and immediately when
the Gateway asks with opcode 1); send one Identify (opcode 2); collect Dispatch events (opcode 0) until the window
ends; close with code 1000, which invalidates the session. It never resumes and never reconnects: a Reconnect
(opcode 7) or Invalid Session (opcode 9) ends the window and is reported.

Constraints encoded here:

* **Receive-only and bounded.** The only payloads ever sent are Heartbeat and one Identify. There is no presence
  update, no voice state, no member request. The window has a fixed length and a fixed maximum number of events.
* **Never the presence or direct-message intents.** ``GUILD_PRESENCES`` and every ``DIRECT_MESSAGE*`` intent are
  refused at construction; ``MESSAGE_CONTENT`` is requested only with ``keep_content=True``.
* **Reduction before anything leaves the module.** Raw payloads never leave :meth:`GatewayWindow.run`. Members are
  one-way pseudonyms (``member_id_hash``), invites carry no inviter, messages carry a length (and text only with
  ``keep_content``), and an event of any guild other than the configured one is counted and dropped.
* **The token is sent in Identify and nowhere else.** It is fetched at the moment Identify is built, is not stored
  on the object, and is removed from the text of every exception this module raises.
"""
from __future__ import annotations

import json
import random
import time as _time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .discord_perms import INTENT_BITS, intents_value, member_id_hash

GATEWAY_VERSION = 10
OP_DISPATCH, OP_HEARTBEAT, OP_IDENTIFY, OP_RESUME = 0, 1, 2, 6
OP_RECONNECT, OP_INVALID_SESSION, OP_HELLO, OP_HEARTBEAT_ACK = 7, 9, 10, 11
#: Gateway close event codes: code -> (description, explanation, may reconnect)
CLOSE_CODES: Dict[int, Tuple[str, str, bool]] = {
    4000: ("Unknown error", "We're not sure what went wrong.", True),
    4001: ("Unknown opcode", "An invalid Gateway opcode or an invalid payload for an opcode was sent.", True),
    4002: ("Decode error", "An invalid payload was sent.", True),
    4003: ("Not authenticated", "A payload was sent prior to identifying, or this session has been invalidated.", True),
    4004: ("Authentication failed", "The account token sent with the identify payload is incorrect.", False),
    4005: ("Already authenticated", "More than one identify payload was sent.", True),
    4007: ("Invalid seq", "The sequence sent when resuming the session was invalid.", True),
    4008: ("Rate limited", "Payloads were sent too quickly.", True),
    4009: ("Session timed out", "The session timed out.", True),
    4010: ("Invalid shard", "An invalid shard was sent when identifying.", False),
    4011: ("Sharding required", "The session would have handled too many guilds; sharding is required.", False),
    4012: ("Invalid API version", "An invalid version for the gateway was sent.", False),
    4013: ("Invalid intent(s)", "An invalid intent for a Gateway Intent was sent; the bitwise value may be "
                                "incorrectly calculated.", False),
    4014: ("Disallowed intent(s)", "A disallowed intent for a Gateway Intent was sent: an intent that is not "
                                   "enabled or not approved for the application.", False),
}
#: close codes that state a missing authorization (hard failures: never retried, never worked around)
PERMISSION_CLOSE_CODES = (4004, 4013, 4014)
#: the intents of a window unless the caller narrows them
DEFAULT_INTENTS: Tuple[str, ...] = ("GUILDS", "GUILD_MEMBERS", "GUILD_INVITES", "GUILD_MESSAGES",
                                    "GUILD_SCHEDULED_EVENTS")
#: intents this adapter never requests
FORBIDDEN_INTENTS: Tuple[str, ...] = ("GUILD_PRESENCES", "DIRECT_MESSAGES", "DIRECT_MESSAGE_REACTIONS",
                                      "DIRECT_MESSAGE_TYPING", "DIRECT_MESSAGE_POLLS")
_CHANNEL_EVENTS = ("CHANNEL_CREATE", "CHANNEL_UPDATE", "CHANNEL_DELETE")
_SCHEDULED_EVENTS = ("GUILD_SCHEDULED_EVENT_CREATE", "GUILD_SCHEDULED_EVENT_UPDATE", "GUILD_SCHEDULED_EVENT_DELETE")
_SCHEDULED_USER_EVENTS = ("GUILD_SCHEDULED_EVENT_USER_ADD", "GUILD_SCHEDULED_EVENT_USER_REMOVE")
_MEMBER_EVENTS = ("GUILD_MEMBER_ADD", "GUILD_MEMBER_REMOVE", "GUILD_MEMBER_UPDATE")
#: events that are reduced and kept; every other event name is counted and dropped
KEPT_EVENTS: Tuple[str, ...] = (("READY", "GUILD_CREATE", "GUILD_UPDATE", "INVITE_CREATE", "INVITE_DELETE",
                                 "MESSAGE_CREATE") + _MEMBER_EVENTS + _CHANNEL_EVENTS + _SCHEDULED_EVENTS
                                + _SCHEDULED_USER_EVENTS)


class GatewayError(RuntimeError):
    """The window could not be opened or was refused. ``kind`` is ``permission`` (close codes 4004 / 4013 / 4014,
    or no token), ``limit``, ``network`` or ``protocol``. The message never contains the token."""

    def __init__(self, kind: str, message: str, *, close_code: Optional[int] = None):
        super().__init__(message)
        self.kind, self.close_code = kind, close_code


def describe_close(code: Optional[int], reason: str = "") -> str:
    """The documented meaning of a Gateway close code, for failure details."""
    if code is None:
        return "the connection was closed without a close code"
    if code in CLOSE_CODES:
        name, text, _ = CLOSE_CODES[code]
        return f"gateway close code {code} ({name}): {text}"
    return f"connection closed with code {code}" + (f" ({reason[:120]})" if reason else "")


def gateway_connect_url(url: str) -> str:
    """The connection URL with the API version and the JSON encoding passed explicitly."""
    return f"{url.split('?', 1)[0].rstrip('/')}/?v={GATEWAY_VERSION}&encoding=json"


class GatewayWindow:
    """One bounded listening window. ``token_provider`` is called once, when Identify is built."""

    def __init__(self, url: str, token_provider: Callable[[], Optional[str]], *, guild_id: str, seconds: float,
                 intents: Sequence[str] = DEFAULT_INTENTS, keep_content: bool = False, id_salt: str = "",
                 test_account_ids: Iterable[str] = (), max_events: int = 200, cancel: Any = None,
                 connect: Optional[Callable[..., Any]] = None, monotonic: Callable[[], float] = _time.monotonic,
                 rng: Optional[random.Random] = None, open_timeout_s: float = 10.0, hello_timeout_s: float = 10.0,
                 library: str = "cam-live"):
        if not guild_id:
            raise ValueError("a gateway window needs the authorized guild id")
        if not 0 < float(seconds) <= 60:
            raise ValueError("a gateway window lasts at most 60 seconds")
        names = list(dict.fromkeys(intents))
        if keep_content and "MESSAGE_CONTENT" not in names:
            names.append("MESSAGE_CONTENT")
        if not keep_content and "MESSAGE_CONTENT" in names:
            raise ValueError("MESSAGE_CONTENT is requested only together with keep_content")
        bad = [n for n in names if n in FORBIDDEN_INTENTS]
        if bad:
            raise ValueError(f"intent(s) never requested by this adapter: {', '.join(bad)}")
        self.url, self._token_provider = url, token_provider
        self.guild_id, self.seconds, self.intents = str(guild_id), float(seconds), tuple(names)
        self.intents_value = intents_value(names)
        self.keep_content, self.id_salt = bool(keep_content), id_salt
        self._tests = {str(t) for t in test_account_ids}
        self.max_events, self.cancel = max(1, int(max_events)), cancel
        self._connect, self._monotonic = connect, monotonic
        self._rng = rng or random.Random()
        self.open_timeout_s, self.hello_timeout_s, self.library = open_timeout_s, hello_timeout_s, library

    # ---- reduction ---------------------------------------------------------------------------------------------
    def _member(self, event: str, d: Dict[str, Any]) -> Dict[str, Any]:
        user = d.get("user") or {}
        uid = str(user.get("id", ""))
        out: Dict[str, Any] = {"event": event, "id_hash": member_id_hash(self.guild_id, uid, self.id_salt),
                               "bot": bool(user.get("bot", False)), "test_account": uid in self._tests}
        if event != "GUILD_MEMBER_REMOVE":
            out.update(joined_at=d.get("joined_at"), pending=bool(d.get("pending", False)))
        return out

    def reduce(self, event: str, d: Any) -> Optional[Dict[str, Any]]:
        """The record kept for one Dispatch event of the configured guild (``None``: nothing is kept)."""
        d = d if isinstance(d, dict) else {}
        if event == "GUILD_CREATE":
            return {"event": event, "member_count": d.get("member_count"), "channels": len(d.get("channels") or []),
                    "unavailable": bool(d.get("unavailable", False))}
        if event == "GUILD_UPDATE":
            return {"event": event, "name": d.get("name", "")}
        if event in _MEMBER_EVENTS:
            return self._member(event, d)
        if event == "INVITE_CREATE":
            return {"event": event, "code": d.get("code"), "channel_id": d.get("channel_id"),
                    "max_age": d.get("max_age"), "max_uses": d.get("max_uses")}
        if event == "INVITE_DELETE":
            return {"event": event, "code": d.get("code"), "channel_id": d.get("channel_id")}
        if event == "MESSAGE_CREATE":
            author = d.get("author") or {}
            text = d.get("content") or ""
            out = {"event": event, "channel_id": d.get("channel_id"),
                   "author_id_hash": member_id_hash(self.guild_id, str(author.get("id", "")), self.id_salt),
                   "bot": bool(author.get("bot", False)), "length": len(text)}
            if self.keep_content:
                out["content"] = str(text)
            return out
        if event in _CHANNEL_EVENTS:
            return {"event": event, "id": d.get("id"), "name": d.get("name", ""), "type": d.get("type")}
        if event in _SCHEDULED_EVENTS:
            return {"event": event, "id": d.get("id"), "name": d.get("name", "")}
        if event in _SCHEDULED_USER_EVENTS:
            return {"event": event, "id": d.get("guild_scheduled_event_id")}
        return None

    # ---- the window --------------------------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Open the window, collect, close. Returns the reduced summary; raises :class:`GatewayError` when no
        session could be started."""
        from websockets.exceptions import ConnectionClosed, WebSocketException
        connect = self._connect
        if connect is None:
            from websockets.sync.client import connect as ws_connect
            connect = ws_connect
        token = self._token_provider()
        if not token:
            raise GatewayError("permission", "blocked: no Discord bot token was released for the gateway destination")

        def clean(text: Any) -> str:
            return str(text).replace(token, "[token]")

        try:
            ws = connect(gateway_connect_url(self.url), open_timeout=self.open_timeout_s, close_timeout=2.0,
                         ping_interval=None, max_size=16 * 1024 * 1024, proxy=None,
                         user_agent_header=f"{self.library} (gateway window)")
        except (OSError, WebSocketException, TimeoutError) as e:
            raise GatewayError("network", clean(f"could not connect to the gateway: {type(e).__name__}: {e}")) from None
        out: Dict[str, Any] = {
            "session_ok": False, "ended_by": "", "requested_s": self.seconds, "window_s": 0.0,
            "intents": list(self.intents), "events": [], "counts": {}, "guilds": 0, "configured_guild_listed": False,
            "dropped": {"other_guild": 0, "no_guild": 0, "unreduced": {}}, "close_code": None, "resumable": None,
            "heartbeats": {"interval_s": None, "sent": 0, "acknowledged": 0, "requested_by_gateway": 0},
            "content_included": self.keep_content, "truncated": False}
        hb = out["heartbeats"]
        started = self._monotonic()
        end = started + self.seconds
        hello_by = started + self.hello_timeout_s
        next_hb: Optional[float] = None
        interval, awaiting_ack, seq, identified = 0.0, False, None, False
        close_code = 1000

        def send(payload: Dict[str, Any]) -> None:
            ws.send(json.dumps(payload))

        def heartbeat() -> None:
            nonlocal awaiting_ack
            send({"op": OP_HEARTBEAT, "d": seq})
            hb["sent"] += 1
            awaiting_ack = True

        try:
            while True:
                now = self._monotonic()
                if self.cancel is not None and self.cancel.is_set():
                    out["ended_by"] = "cancelled"
                    break
                if now >= end:
                    out["ended_by"] = "window"
                    break
                if not identified and now >= hello_by:
                    raise GatewayError("network", "the gateway sent no Hello within "
                                                  f"{self.hello_timeout_s:g} s of connecting")
                if next_hb is not None and now >= next_hb:
                    if awaiting_ack:        # no ACK between two heartbeats: a failed ("zombied") connection
                        out["ended_by"], close_code = "heartbeat_not_acknowledged", 4000
                        break
                    heartbeat()
                    next_hb = now + interval
                    continue
                wait_until = min(end, next_hb if next_hb is not None else end, now + 0.25)
                try:
                    raw = ws.recv(timeout=max(0.0, wait_until - now))
                except TimeoutError:
                    continue
                except ConnectionClosed as e:
                    rcvd = getattr(e, "rcvd", None)
                    code = getattr(rcvd, "code", None)
                    out["close_code"] = code
                    if code in PERMISSION_CLOSE_CODES:
                        raise GatewayError("permission", describe_close(code), close_code=code) from None
                    if not out["session_ok"]:
                        kind = "limit" if code == 4008 else "network" if (code is None or CLOSE_CODES.get(
                            code, ("", "", True))[2]) else "protocol"
                        raise GatewayError(kind, "the gateway closed the connection before a session started: "
                                           + describe_close(code, clean(getattr(rcvd, "reason", "") or "")),
                                           close_code=code) from None
                    out["ended_by"] = f"closed:{code if code is not None else 'none'}"
                    break
                try:
                    msg = json.loads(raw if isinstance(raw, str) else bytes(raw).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    out["dropped"]["unreduced"]["(undecodable)"] = out["dropped"]["unreduced"].get("(undecodable)", 0) + 1
                    continue
                op = msg.get("op") if isinstance(msg, dict) else None
                d = msg.get("d") if isinstance(msg, dict) else None
                if op == OP_HELLO and not identified:
                    interval = float((d or {}).get("heartbeat_interval", 45000)) / 1000.0
                    hb["interval_s"] = interval
                    next_hb = self._monotonic() + interval * self._rng.random()      # the documented initial jitter
                    send({"op": OP_IDENTIFY, "d": {"token": token, "intents": self.intents_value,
                                                   "properties": {"os": "linux", "browser": self.library,
                                                                  "device": self.library}}})
                    identified = True
                elif op == OP_HEARTBEAT_ACK:
                    awaiting_ack = False
                    hb["acknowledged"] += 1
                elif op == OP_HEARTBEAT:            # the gateway asks for a heartbeat now
                    hb["requested_by_gateway"] += 1
                    heartbeat()
                elif op == OP_RECONNECT:
                    out["ended_by"] = "reconnect_requested"
                    break
                elif op == OP_INVALID_SESSION:
                    out["ended_by"], out["resumable"] = "invalid_session", bool(d)
                    break
                elif op == OP_DISPATCH:
                    if msg.get("s") is not None:
                        seq = msg["s"]
                    self._dispatch(str(msg.get("t") or ""), d, out)
                    if len(out["events"]) >= self.max_events:
                        out["ended_by"], out["truncated"] = "max_events", True
                        break
        except GatewayError:
            close_code = 1000
            raise
        except (OSError, WebSocketException) as e:
            if not out["session_ok"]:
                raise GatewayError("network", clean(f"gateway connection failed: {type(e).__name__}: {e}")) from None
            out["ended_by"] = out["ended_by"] or f"error:{type(e).__name__}"
        finally:
            try:
                ws.close(close_code)        # 1000 invalidates the session: this adapter never resumes
            except Exception:  # noqa: BLE001 — the connection may already be gone
                pass
        out["window_s"] = round(self._monotonic() - started, 3)
        if not out["session_ok"]:
            why = {"invalid_session": "the gateway answered Identify with Invalid Session (opcode 9, resumable: "
                                      f"{out['resumable']})",
                   "reconnect_requested": "the gateway asked for a reconnect (opcode 7) before a session started"
                   }.get(out["ended_by"], f"no Ready event arrived before the window ended ({out['ended_by']})")
            raise GatewayError("limit" if out["ended_by"] == "invalid_session" else "network",
                               "no gateway session was started: " + why)
        return out

    def _dispatch(self, event: str, d: Any, out: Dict[str, Any]) -> None:
        dropped = out["dropped"]
        if event == "READY":
            guilds = [str(g.get("id")) for g in ((d or {}).get("guilds") or []) if isinstance(g, dict)]
            out.update(session_ok=True, guilds=len(guilds), configured_guild_listed=self.guild_id in guilds)
            rec: Optional[Dict[str, Any]] = {"event": "READY", "guilds": len(guilds),
                                             "configured_guild_listed": self.guild_id in guilds}
        else:
            body = d if isinstance(d, dict) else {}
            gid = body.get("id") if event in ("GUILD_CREATE", "GUILD_UPDATE", "GUILD_DELETE") else body.get("guild_id")
            if gid is None:
                dropped["no_guild"] += 1                 # e.g. anything about a direct-message channel
                return
            if str(gid) != self.guild_id:
                dropped["other_guild"] += 1              # another server: counted, never looked at
                return
            rec = self.reduce(event, body) if event in KEPT_EVENTS else None
        if rec is None:
            dropped["unreduced"][event] = dropped["unreduced"].get(event, 0) + 1
            return
        out["events"].append(rec)
        out["counts"][event] = out["counts"].get(event, 0) + 1


def intents_needed(keep_content: bool) -> List[str]:
    """Intent names of a window, with ``MESSAGE_CONTENT`` only under the content permission."""
    return list(DEFAULT_INTENTS) + (["MESSAGE_CONTENT"] if keep_content else [])


__all__ = ["GatewayWindow", "GatewayError", "CLOSE_CODES", "PERMISSION_CLOSE_CODES", "DEFAULT_INTENTS",
           "FORBIDDEN_INTENTS", "KEPT_EVENTS", "INTENT_BITS", "describe_close", "gateway_connect_url",
           "intents_needed", "GATEWAY_VERSION"]
