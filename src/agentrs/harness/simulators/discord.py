"""Local simulator of the Discord HTTP API (v10) subset used by ``DiscordConnector``.

Implements (under ``/api/v10``): ``GET /users/@me``, ``GET /channels/{id}/messages`` (+ ``/{mid}``),
``POST /channels/{id}/messages`` (``content`` <= 2000, ``message_reference``, ``allowed_mentions``,
``nonce`` + ``enforce_nonce`` de-duplication), ``POST /interactions/{id}/{token}/callback``
(``?with_response=true`` returns the created message; the initial response must come within 3 s and
only once).

Auth: ``Authorization: Bot <token>`` (401 ``{"message": "401: Unauthorized", "code": 0}``). The bot
sees and posts only in channels where it was granted VIEW_CHANNEL / SEND_MESSAGES (403 ``Missing
Access`` 50001 / ``Missing Permissions`` 50013; unknown channel 404 10003). Without the
MESSAGE_CONTENT intent the bot gets empty ``content`` for messages that do not mention it.
Per-route rate limits with ``X-RateLimit-Limit/Remaining/Reset/Reset-After/Bucket`` and 429 bodies
``{"message": "You are being rate limited.", "retry_after": s, "global": false}``.

Interactions: ``emit_interaction`` POSTs an APPLICATION_COMMAND signed with Ed25519
(``X-Signature-Ed25519`` over ``timestamp + body``, ``X-Signature-Timestamp``) to
``interactions_endpoint``; ``verify_endpoint`` reproduces Discord's endpoint check (a valid PING must
get ``{"type": 1}`` and a request with a bad signature must get 401). Other users:
``simulate_message``, ``simulate_reaction``, ``simulate_delete``.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone, UTC
from typing import Any, Optional

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from nacl.signing import SigningKey

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    WebhookTarget,
    deliver_webhook,
    jdump,
)

DISCORD_EPOCH_MS = 1420070400000
VIEW, SEND, HISTORY = "VIEW_CHANNEL", "SEND_MESSAGES", "READ_MESSAGE_HISTORY"


class DiscordError(Exception):
    def __init__(self, status: int, body: dict, headers: Optional[dict] = None):
        self.status, self.body, self.headers = status, body, headers or {}


def _ts(t: float) -> str:
    return datetime.fromtimestamp(t, tz=UTC).isoformat(timespec="microseconds")


class DiscordState:
    def __init__(self, clock: Optional[SimClock] = None, bot_token: str = "SIMULATED.bot-token.0001",
                 interactions_endpoint: WebhookTarget = None, message_content_intent: bool = True):
        self.clock = clock or SimClock()
        self._seq = 0
        self.bot_token = bot_token
        self.signing_key = SigningKey(hashlib.sha256(b"agentrs-discord-simulator").digest())
        self.public_key_hex = self.signing_key.verify_key.encode().hex()
        self.application_id = self.snowflake()
        self.bot_user = {"id": self.application_id, "username": "agentrs-sim", "discriminator": "0",
                         "global_name": "agentrs (simulated bot)", "avatar": None, "bot": True, "flags": 0}
        self.guild_id = self.snowflake()
        self.users: dict[str, dict] = {}
        self.channels: dict[str, dict] = {}
        self.bot_perms: dict[str, set[str]] = {}
        self.messages: dict[str, list[dict]] = {}
        self.interactions: dict[str, dict] = {}
        self.nonces: dict[tuple[str, str], str] = {}
        self.interactions_endpoint = interactions_endpoint
        self.message_content_intent = message_content_intent
        self.limits = {"post_message": (5, 5.0), "get_messages": (50, 1.0), "interaction": (50, 1.0)}
        self.limiter = RateLimiter(self.clock)
        self.deliveries: list[dict] = []

    def snowflake(self) -> str:
        self._seq += 1
        ms = int(self.clock.now() * 1000)
        return str(((ms - DISCORD_EPOCH_MS) << 22) | (self._seq & 0xFFF))

    # ------------------------------------------------------------------ setup / simulation
    def add_channel(self, name: str, bot_can_view: bool = True, bot_can_send: bool = True) -> str:
        cid = self.snowflake()
        self.channels[cid] = {"id": cid, "type": 0, "guild_id": self.guild_id, "name": name, "position": len(self.channels),
                              "topic": None, "nsfw": False, "last_message_id": None, "parent_id": None}
        self.bot_perms[cid] = ({VIEW, HISTORY} if bot_can_view else set()) | ({SEND} if bot_can_send and bot_can_view else set())
        self.messages[cid] = []
        return cid

    def user(self, name: str) -> dict:
        if name not in self.users:
            self.users[name] = {"id": self.snowflake(), "username": name, "discriminator": "0", "global_name": name.title(),
                                "avatar": None}
        return self.users[name]

    def _message(self, channel_id: str, author: dict, content: str, reply_to: Optional[str] = None,
                 mentions: Optional[list[dict]] = None, extra: Optional[dict] = None) -> dict:
        mid = self.snowflake()
        m: dict[str, Any] = {"id": mid, "channel_id": channel_id, "author": author, "content": content,
                             "timestamp": _ts(self.clock.now()), "t": self.clock.now(), "edited_timestamp": None,
                             "tts": False, "mention_everyone": False, "mentions": mentions or [], "mention_roles": [],
                             "attachments": [], "embeds": [], "pinned": False, "type": 19 if reply_to else 0, "flags": 0,
                             "reactions": {}, "deleted": False}
        if reply_to:
            m["message_reference"] = {"type": 0, "message_id": reply_to, "channel_id": channel_id,
                                      "guild_id": self.guild_id}
        if extra:
            m.update(extra)
        self.messages[channel_id].append(m)
        self.channels[channel_id]["last_message_id"] = mid
        return m

    def simulate_message(self, channel_id: str, text: str, by: str = "carol", reply_to: Optional[str] = None,
                         mention_bot: bool = False) -> str:
        mentions = []
        if mention_bot:
            text = f"<@{self.bot_user['id']}> {text}"
            mentions = [self.bot_user]
        if reply_to:
            parent = self.find(channel_id, reply_to)
            if parent and parent["author"]["id"] == self.bot_user["id"] and self.bot_user not in mentions:
                mentions.append(self.bot_user)  # replies ping the replied-to author by default
        return self._message(channel_id, self.user(by), text, reply_to, mentions)["id"]

    def simulate_reaction(self, channel_id: str, message_id: str, emoji: str = "\N{THUMBS UP SIGN}",
                          by: str = "erin") -> None:
        m = self.find(channel_id, message_id)
        m["reactions"].setdefault(emoji, set()).add(self.user(by)["id"])

    def simulate_delete(self, channel_id: str, message_id: str) -> None:
        self.find(channel_id, message_id)["deleted"] = True

    def find(self, channel_id: str, message_id: str) -> Optional[dict]:
        return next((m for m in self.messages.get(channel_id, []) if m["id"] == message_id), None)

    def message_view(self, m: dict) -> dict:
        out = {k: v for k, v in m.items() if k not in ("t", "deleted", "reactions")}
        if m["reactions"]:
            out["reactions"] = [{"count": len(u), "count_details": {"burst": 0, "normal": len(u)}, "me": self.bot_user["id"] in u,
                                 "me_burst": False, "emoji": {"id": None, "name": e}, "burst_colors": []}
                                for e, u in m["reactions"].items()]
        ref = m.get("message_reference")
        if ref:
            p = self.find(m["channel_id"], ref["message_id"])
            out["referenced_message"] = self.message_view(p) if p and not p["deleted"] else None
        mentioned = any(u["id"] == self.bot_user["id"] for u in m["mentions"])
        if not self.message_content_intent and not mentioned and m["author"]["id"] != self.bot_user["id"]:
            out.update(content="", embeds=[], attachments=[])
        return out

    # ------------------------------------------------------------------ interactions
    def _signed(self, body: bytes, ts: Optional[str] = None, bad: bool = False) -> dict[str, str]:
        ts = ts or str(int(self.clock.now()))
        sig = self.signing_key.sign(ts.encode() + (body + b" " if bad else body)).signature.hex()
        return {"Content-Type": "application/json", "User-Agent": "Discord-Interactions/1.0 (+https://discord.com)",
                "X-Signature-Ed25519": sig, "X-Signature-Timestamp": ts, "X-Agentrs-Simulator": "1"}

    def emit_interaction(self, channel_id: str, command: str = "ask", options: Optional[dict[str, Any]] = None,
                         by: str = "carol") -> dict:
        iid = self.snowflake()
        token = "aW50ZXJhY3Rpb246" + hashlib.sha256(f"tok{iid}".encode()).hexdigest()
        u = self.user(by)
        payload = {"id": iid, "application_id": self.application_id, "type": 2, "token": token, "version": 1,
                   "guild_id": self.guild_id, "channel_id": channel_id, "channel": self.channels[channel_id],
                   "member": {"user": u, "roles": [], "permissions": "2147483647", "nick": None},
                   "data": {"id": self.snowflake(), "name": command, "type": 1,
                            "options": [{"name": k, "type": 3, "value": v} for k, v in (options or {}).items()]},
                   "app_permissions": "2048", "locale": "en-US", "guild_locale": "en-US",
                   "entitlements": [], "authorizing_integration_owners": {"0": self.guild_id}, "context": 0}
        self.interactions[iid] = {"token": token, "channel_id": channel_id, "created": self.clock.now(),
                                  "responded": False, "user": u, "command": command}
        body = jdump(payload)
        rec = deliver_webhook(self.interactions_endpoint, self._signed(body), body, self.deliveries)
        try:
            resp = json.loads(rec.get("response") or b"null")
        except ValueError:
            resp = None
        if rec.get("status") == 200 and isinstance(resp, dict) and resp.get("type") in (4, 5):
            self._respond(iid, resp)
        return payload

    def verify_endpoint(self) -> dict[str, bool]:
        """Discord's Interactions Endpoint URL check: PING must be answered, a bad signature rejected."""
        body = jdump({"id": self.snowflake(), "application_id": self.application_id, "type": 1, "token": "ping",
                      "version": 1})
        ok = deliver_webhook(self.interactions_endpoint, self._signed(body), body, self.deliveries)
        try:
            ping_ok = ok["status"] == 200 and json.loads(ok["response"] or b"{}").get("type") == 1
        except ValueError:
            ping_ok = False
        bad = deliver_webhook(self.interactions_endpoint, self._signed(body, bad=True), body, self.deliveries)
        return {"ping_ok": ping_ok, "rejects_bad_signature": bad["status"] == 401}

    def _respond(self, iid: str, resp: dict) -> Optional[dict]:
        it = self.interactions[iid]
        it["responded"] = True
        if resp.get("type") != 4:
            return None
        data = resp.get("data") or {}
        ephemeral = bool(int(data.get("flags", 0)) & 64)
        meta = {"interaction_metadata": {"id": iid, "type": 2, "name": it["command"], "user": it["user"],
                                         "authorizing_integration_owners": {"0": self.guild_id}},
                "application_id": self.application_id, "webhook_id": self.application_id,
                "flags": 64 if ephemeral else 0}
        m = self._message(it["channel_id"], self.bot_user, str(data.get("content", "")), extra={**meta, "type": 20})
        if ephemeral:
            self.messages[it["channel_id"]].remove(m)
        it["message"] = m
        return m


def create_app(state: Optional[DiscordState] = None) -> FastAPI:
    st = state or DiscordState()
    app = FastAPI(title="Simulated Discord API (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="discord")

    @app.exception_handler(DiscordError)
    async def _err(request: Request, e: DiscordError):
        return JSONResponse(e.body, status_code=e.status, headers=e.headers)

    def _rl(route: str, key: str, response: Response) -> None:
        lim, win = st.limits[route]
        rs = st.limiter.hit(f"{route}:{key}", lim, win)
        after = max(0.0, rs.reset - st.clock.now())
        h = {"X-RateLimit-Limit": str(rs.limit), "X-RateLimit-Remaining": str(rs.remaining),
             "X-RateLimit-Reset": f"{rs.reset:.3f}", "X-RateLimit-Reset-After": f"{after:.3f}",
             "X-RateLimit-Bucket": hashlib.md5(route.encode()).hexdigest()}
        if not rs.allowed:
            h.update({"Retry-After": str(max(1, math.ceil(after))), "X-RateLimit-Scope": "user"})
            raise DiscordError(429, {"message": "You are being rate limited.", "retry_after": round(after, 3),
                                     "global": False}, h)
        response.headers.update(h)

    def bot(request: Request) -> dict:
        auth = request.headers.get("authorization", "")
        if auth != f"Bot {st.bot_token}":
            raise DiscordError(401, {"message": "401: Unauthorized", "code": 0})
        return st.bot_user

    def channel(cid: str, need: str) -> dict:
        ch = st.channels.get(cid)
        if ch is None:
            raise DiscordError(404, {"message": "Unknown Channel", "code": 10003})
        perms = st.bot_perms.get(cid, set())
        if VIEW not in perms:
            raise DiscordError(403, {"message": "Missing Access", "code": 50001})
        if need not in perms:
            raise DiscordError(403, {"message": "Missing Permissions", "code": 50013})
        return ch

    P = "/api/v10"

    @app.get(P + "/users/@me")
    async def me(_: dict = Depends(bot)):
        return st.bot_user

    @app.get(P + "/channels/{cid}/messages")
    async def list_messages(cid: str, response: Response, limit: int = 50, before: Optional[str] = None,
                            after: Optional[str] = None, _: dict = Depends(bot)):
        _rl("get_messages", cid, response)
        channel(cid, VIEW)
        if HISTORY not in st.bot_perms[cid]:
            return []
        if not 1 <= limit <= 100:
            raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035, "errors": {"limit": {"_errors": [
                {"code": "NUMBER_TYPE_MAX", "message": "int value should be less than or equal to 100."}]}}})
        msgs = [m for m in st.messages[cid] if not m["deleted"]]
        if after:
            msgs = [m for m in msgs if int(m["id"]) > int(after)][:limit]
        elif before:
            msgs = [m for m in msgs if int(m["id"]) < int(before)][-limit:]
        else:
            msgs = msgs[-limit:]
        return [st.message_view(m) for m in sorted(msgs, key=lambda m: int(m["id"]), reverse=True)]

    @app.get(P + "/channels/{cid}/messages/{mid}")
    async def get_message(cid: str, mid: str, response: Response, _: dict = Depends(bot)):
        _rl("get_messages", cid, response)
        channel(cid, HISTORY)
        m = st.find(cid, mid)
        if m is None or m["deleted"]:
            raise DiscordError(404, {"message": "Unknown Message", "code": 10008})
        return st.message_view(m)

    @app.post(P + "/channels/{cid}/messages")
    async def post_message(cid: str, request: Request, response: Response, _: dict = Depends(bot)):
        _rl("post_message", cid, response)
        channel(cid, SEND)
        try:
            p = await request.json()
        except ValueError:
            raise DiscordError(400, {"message": "400: Bad Request", "code": 0}) from None
        content = str(p.get("content") or "")
        if not content.strip() and not p.get("embeds"):
            raise DiscordError(400, {"message": "Cannot send an empty message", "code": 50006})
        if len(content) > 2000:
            raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035, "errors": {"content": {"_errors": [
                {"code": "BASE_TYPE_MAX_LENGTH", "message": "Must be 2000 or fewer in length."}]}}})
        nonce = p.get("nonce")
        if nonce is not None and len(str(nonce)) > 25:
            raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035, "errors": {"nonce": {"_errors": [
                {"code": "BASE_TYPE_MAX_LENGTH", "message": "Must be 25 or fewer in length."}]}}})
        if nonce is not None and p.get("enforce_nonce") and (cid, str(nonce)) in st.nonces:
            return st.message_view(st.find(cid, st.nonces[(cid, str(nonce))]))
        ref = (p.get("message_reference") or {}).get("message_id")
        if ref:
            parent = st.find(cid, str(ref))
            if parent is None or parent["deleted"]:
                raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035, "errors": {"message_reference": {
                    "_errors": [{"code": "MESSAGE_REFERENCE_UNKNOWN_MESSAGE", "message": "Unknown message"}]}}})
        am = p.get("allowed_mentions")
        parse = am.get("parse", []) if isinstance(am, dict) else ["users", "roles", "everyone"]
        mentions = [u for u in st.users.values() if f"<@{u['id']}>" in content and "users" in parse]
        m = st._message(cid, st.bot_user, content, str(ref) if ref else None, mentions,
                        {"nonce": nonce} if nonce is not None else None)
        m["mention_everyone"] = "everyone" in parse and ("@everyone" in content or "@here" in content)
        if nonce is not None:
            st.nonces[(cid, str(nonce))] = m["id"]
        return st.message_view(m)

    @app.post(P + "/interactions/{iid}/{token}/callback")
    async def callback(iid: str, token: str, request: Request, response: Response, with_response: bool = False):
        _rl("interaction", iid, response)
        it = st.interactions.get(iid)
        if it is None or it["token"] != token or st.clock.now() - it["created"] > 3.0:
            raise DiscordError(404, {"message": "Unknown interaction", "code": 10062})
        if it["responded"]:
            raise DiscordError(400, {"message": "Interaction has already been acknowledged.", "code": 40060})
        p = await request.json()
        if p.get("type") not in (1, 4, 5, 6, 7, 8, 9):
            raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035})
        content = str((p.get("data") or {}).get("content", ""))
        if len(content) > 2000:
            raise DiscordError(400, {"message": "Invalid Form Body", "code": 50035})
        m = st._respond(iid, p)
        if not with_response:
            return Response(status_code=204)
        out: dict[str, Any] = {"interaction": {"id": iid, "type": 2, "activity_instance_id": None,
                                               "response_message_id": m["id"] if m else None,
                                               "response_message_loading": p.get("type") == 5,
                                               "response_message_ephemeral": bool(m and m["flags"] & 64)}}
        if m:
            out["resource"] = {"type": p["type"], "message": st.message_view(m)}
        return out

    return app
