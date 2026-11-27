"""STAGING ONLY — local imitation of the Discord REST API subset used by ``cam_live.harness.discord``.

Implemented routes (prefix ``/api/v10``), with Discord's payload shapes and error bodies::

    GET    /users/@me                                       GET /users/@me/guilds
    GET    /applications/@me                                GET /gateway/bot
    GET    /guilds/{guild}?with_counts=true                 PATCH /guilds/{guild}                 (MANAGE_GUILD)
    GET    /guilds/{guild}/members?limit=&after=            (needs the members intent)
    GET    /guilds/{guild}/members/{user}
    GET    /guilds/{guild}/channels                         (omits channels the bot cannot view)
    POST   /guilds/{guild}/channels                         (MANAGE_CHANNELS)
    GET    /guilds/{guild}/roles                            POST /guilds/{guild}/roles            (MANAGE_ROLES)
    PATCH  /guilds/{guild}/roles/{role}                     DELETE /guilds/{guild}/roles/{role}   (MANAGE_ROLES)
    GET    /guilds/{guild}/invites                          (MANAGE_GUILD, or VIEW_AUDIT_LOG without metadata)
    GET    /guilds/{guild}/audit-logs?limit=                (VIEW_AUDIT_LOG)
    GET    /guilds/{guild}/scheduled-events?with_user_count=true
    POST   /guilds/{guild}/scheduled-events                 (CREATE_EVENTS; in a voice channel also CONNECT)
    GET    /channels/{channel}                              PATCH, DELETE /channels/{channel}     (MANAGE_CHANNELS)
    PUT    /channels/{channel}/permissions/{overwrite}      (MANAGE_ROLES; only permissions the bot holds)
    GET    /channels/{channel}/messages?limit=              (VIEW_CHANNEL; empty without READ_MESSAGE_HISTORY)
    POST   /channels/{channel}/messages                     (SEND_MESSAGES; nonce + enforce_nonce de-duplication)
    GET    /channels/{channel}/messages/{message}           (VIEW_CHANNEL, READ_MESSAGE_HISTORY)
    PATCH  /channels/{channel}/messages/{message}           (own messages)
    DELETE /channels/{channel}/messages/{message}           (own, or MANAGE_MESSAGES)
    PUT, DELETE /channels/{channel}/messages/pins/{message} (PIN_MESSAGES)
    POST   /channels/{channel}/invites                      (CREATE_INSTANT_INVITE)
    GET    /invites/{code}                                  DELETE /invites/{code}  (MANAGE_CHANNELS or MANAGE_GUILD)

Authentication is the ``Authorization: Bot <token>`` header. Per-route rate limits are reported with
``X-RateLimit-Limit / -Remaining / -Reset / -Reset-After / -Bucket`` and enforced with HTTP 429, ``Retry-After``
and the JSON body ``{"message", "retry_after", "global"}``.

**Permissions** are enforced the way the documentation describes them: the bot member's roles (and the
``@everyone`` role, whose id is the guild id) are OR-ed, ``ADMINISTRATOR`` grants everything, and a channel's
permission overwrites are applied in the documented order. This implementation is written independently of the
adapter's (``cam_live.harness.discord_perms``); only the table of bit positions is shared.

Test hooks (plain methods, not reachable over HTTP) add and remove human members, bot members and
operator-declared test accounts, post member messages, use invites, change roles, permissions, overwrites and
limits, remove the bot from the guild, and add another guild or an invite of another guild. Every state change
is also dispatched to the sessions of the staging Gateway (``staging_discord.gateway``) when one is attached.

This is a staging service. Results obtained against it are never live results.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from ..discord_perms import DEFAULT_SCOPE, PERMISSION_BITS

STAGING_LABEL = "cam-live-staging-discord"
STAGING_HEADER = "X-Staging"
API_PREFIX = "/api/v10"
#: what the bot's own (managed) role holds unless a test changes it: the permission integer 2269727275740401
DEFAULT_BOT_PERMISSIONS = DEFAULT_SCOPE
DEFAULT_EVERYONE_PERMISSIONS = ("VIEW_CHANNEL", "READ_MESSAGE_HISTORY")
GUILD_TEXT, DM, GUILD_VOICE, GROUP_DM, GUILD_CATEGORY, GUILD_ANNOUNCEMENT, GUILD_FORUM = 0, 1, 2, 3, 4, 5, 15
_ADMIN = 1 << PERMISSION_BITS["ADMINISTRATOR"]
_ALL = 0
for _b in PERMISSION_BITS.values():
    _ALL |= 1 << _b
#: audit log event values (developers/resources/audit-log.mdx)
_AUDIT = {"GUILD_UPDATE": 1, "CHANNEL_CREATE": 10, "CHANNEL_UPDATE": 11, "CHANNEL_DELETE": 12,
          "CHANNEL_OVERWRITE_CREATE": 13, "CHANNEL_OVERWRITE_UPDATE": 14, "ROLE_CREATE": 30, "ROLE_UPDATE": 31,
          "ROLE_DELETE": 32, "INVITE_CREATE": 40, "INVITE_DELETE": 42, "MESSAGE_DELETE": 72, "MESSAGE_PIN": 74,
          "MESSAGE_UNPIN": 75, "GUILD_SCHEDULED_EVENT_CREATE": 100}

Reply = Tuple[int, Dict[str, str], Any]


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()


def _mask(names: Iterable[str]) -> int:
    out = 0
    for n in names:
        out |= 1 << PERMISSION_BITS[n]
    return out


class StagingDiscord:
    """In-memory state of the staging service plus its request handler."""

    environment = "staging"

    def __init__(self, *, token: str = "staging-bot-token", guild_name: str = "staging guild",
                 members_intent: bool = True, rate_limit: Tuple[int, float] = (50, 1.0),
                 time_fn: Callable[[], float] = _time.time):
        self.token, self.time = token, time_fn
        self.lock = threading.RLock()
        self._seq = 1_000_000_000_000_000_000
        self.members_intent = members_intent
        #: permissions the service refuses although the bot's roles hold them (what a client sees whose view of
        #: the roles is out of date); changed with :meth:`set_permission`
        self.refused: set = set()
        self.rate_limit = rate_limit
        self.global_block_until = 0.0
        self._buckets: Dict[str, Tuple[float, int]] = {}       # bucket -> (window start, used)
        self.request_log: List[Dict[str, Any]] = []
        self.gateway: Any = None                                # a StagingGateway, when one is attached
        self.session_starts_remaining = 1000
        self.bot_user = {"id": self._snowflake(), "username": "staging-app", "discriminator": "0",
                         "global_name": None, "avatar": None, "bot": True}
        self.guild_id = self._snowflake()
        self.guild = {"id": self.guild_id, "name": guild_name, "description": None, "owner_id": self._snowflake(),
                      "features": [], "verification_level": 1, "default_message_notifications": 0,
                      "explicit_content_filter": 0, "mfa_level": 0, "system_channel_id": None,
                      "rules_channel_id": None, "public_updates_channel_id": None, "preferred_locale": "en-US",
                      "icon": None, "banner": None, "afk_channel_id": None, "afk_timeout": 300}
        #: bits 15 and 19: the limited members and message-content intents; bit 13 (presence) is set as well, as
        #: on the operator's application
        self.application = {"id": self.bot_user["id"], "name": "staging app", "bot_public": False,
                            "bot_require_code_grant": False, "description": "", "flags": (1 << 13) | (1 << 15) | (1 << 19)}
        self.bot_in_guild = True
        self.other_guilds: List[Dict[str, Any]] = []
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.members: Dict[str, Dict[str, Any]] = {}
        self.test_account_ids: List[str] = []                   # what the operator would list in the task file
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.invites: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.presence: Optional[int] = None
        self._nonces: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._role(self.guild_id, "@everyone", DEFAULT_EVERYONE_PERMISSIONS, position=0)
        self.bot_role_id = self.add_role("staging-app", DEFAULT_BOT_PERMISSIONS, managed=True)
        self._add_member(self.bot_user, pending=False)
        self.members[self.bot_user["id"]]["roles"] = [self.bot_role_id]
        self.general_id = self.add_channel("general")

    # ---- test hooks --------------------------------------------------------------------------------------------
    def _snowflake(self) -> str:
        with self.lock:
            self._seq += 1
            return str(self._seq)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        if self.gateway is not None:
            self.gateway.emit(event, data)

    def _add_member(self, user: Dict[str, Any], *, pending: bool, joined_at: Optional[float] = None) -> str:
        with self.lock:
            self.members[user["id"]] = {"user": user, "nick": None, "roles": [], "deaf": False, "mute": False,
                                        "flags": 0, "pending": pending,
                                        "joined_at": _iso(self.time() if joined_at is None else joined_at)}
            self._emit("GUILD_MEMBER_ADD", dict(self.members[user["id"]], guild_id=self.guild_id))
            return user["id"]

    def _user(self, username: Optional[str], bot: bool) -> Dict[str, Any]:
        uid = self._snowflake()
        user = {"id": uid, "username": username or f"user{uid[-6:]}", "discriminator": "0", "global_name": None,
                "avatar": None}
        if bot:
            user["bot"] = True
        return user

    def add_human(self, username: Optional[str] = None, *, joined_at: Optional[float] = None,
                  pending: bool = False) -> str:
        return self._add_member(self._user(username, False), pending=pending, joined_at=joined_at)

    def add_bot(self, username: Optional[str] = None, *, joined_at: Optional[float] = None) -> str:
        return self._add_member(self._user(username, True), pending=False, joined_at=joined_at)

    def add_test_account(self, username: Optional[str] = None, *, joined_at: Optional[float] = None) -> str:
        """An ordinary (non-bot) account that the operator declares as a test account."""
        user = self._user(username, False)
        self.test_account_ids.append(user["id"])
        return self._add_member(user, pending=False, joined_at=joined_at)

    def remove_member(self, user_id: str) -> None:
        with self.lock:
            m = self.members.pop(user_id, None)
            if m is not None:
                self._emit("GUILD_MEMBER_REMOVE", {"guild_id": self.guild_id, "user": dict(m["user"])})

    def update_member(self, user_id: str, *, pending: Optional[bool] = None, roles: Optional[List[str]] = None) -> None:
        with self.lock:
            m = self.members[user_id]
            if pending is not None:
                m["pending"] = pending
            if roles is not None:
                m["roles"] = list(roles)
            self._emit("GUILD_MEMBER_UPDATE", dict(m, guild_id=self.guild_id))

    def add_channel(self, name: str, type: int = GUILD_TEXT, *, guild: bool = True) -> str:
        cid = self._snowflake()
        with self.lock:
            self.channels[cid] = {"id": cid, "type": type, "guild_id": self.guild_id if guild else None, "name": name,
                                  "position": len(self.channels), "parent_id": None, "topic": None, "nsfw": False,
                                  "rate_limit_per_user": 0, "permission_overwrites": []}
            self.messages[cid] = []
            if guild:
                self._emit("CHANNEL_CREATE", dict(self.channels[cid]))
        return cid

    def add_dm_channel(self) -> str:
        """A direct-message channel (type 1): exists so tests can show the adapter never reads it."""
        cid = self.add_channel("", DM, guild=False)
        self.messages[cid].append(self._message(cid, self._user("dm-peer", False), "private words"))
        return cid

    def post_member_message(self, channel_id: str, user_id: str, content: str) -> str:
        with self.lock:
            msg = self._message(channel_id, self.members[user_id]["user"], content)
            self.messages[channel_id].append(msg)
            self._emit_message(msg)
            return msg["id"]

    def use_invite(self, code: str, username: Optional[str] = None) -> str:
        """A person accepts an invite: its use count rises and a human member joins."""
        with self.lock:
            self.invites[code]["uses"] += 1
        return self.add_human(username)

    def set_permission(self, name: str, allowed: bool) -> None:
        """``allowed=False``: the service refuses requests that need ``name`` (403, code 50013) while the bot's
        roles still show it — the answer a client gets when its view of the roles is out of date. To change what
        the roles hold, use :meth:`set_bot_permissions` / :meth:`revoke_bot` / :meth:`grant_bot`."""
        (self.refused.discard if allowed else self.refused.add)(name)

    # roles and permission bitfields
    def _role(self, role_id: str, name: str, permissions: Iterable[str], *, position: int, managed: bool = False,
              color: int = 0, hoist: bool = False, mentionable: bool = False) -> Dict[str, Any]:
        role = {"id": role_id, "name": name, "color": color,
                "colors": {"primary_color": color, "secondary_color": None, "tertiary_color": None}, "hoist": hoist,
                "icon": None, "unicode_emoji": None, "position": position, "permissions": str(_mask(permissions)),
                "managed": managed, "mentionable": mentionable, "flags": 0}
        if managed:
            role["tags"] = {"bot_id": self.bot_user["id"]}
        self.roles[role_id] = role
        return role

    def add_role(self, name: str, permissions: Iterable[str] = (), *, managed: bool = False,
                 position: Optional[int] = None) -> str:
        with self.lock:
            rid = self._snowflake()
            pos = position if position is not None else max((r["position"] for r in self.roles.values()), default=0) + 1
            self._role(rid, name, permissions, position=pos, managed=managed)
            return rid

    def set_role_permissions(self, role_id: str, names: Iterable[str]) -> None:
        with self.lock:
            self.roles[role_id]["permissions"] = str(_mask(names))

    def set_bot_permissions(self, names: Iterable[str]) -> None:
        """Replace what the bot's own role holds."""
        self.set_role_permissions(self.bot_role_id, names)

    def grant_bot(self, *names: str) -> None:
        with self.lock:
            self.roles[self.bot_role_id]["permissions"] = str(int(self.roles[self.bot_role_id]["permissions"]) | _mask(names))

    def revoke_bot(self, *names: str) -> None:
        with self.lock:
            self.roles[self.bot_role_id]["permissions"] = str(int(self.roles[self.bot_role_id]["permissions"]) & ~_mask(names))

    def make_bot_administrator(self, on: bool = True) -> None:
        (self.grant_bot if on else self.revoke_bot)("ADMINISTRATOR")

    def set_everyone_permissions(self, names: Iterable[str]) -> None:
        self.set_role_permissions(self.guild_id, names)

    def assign_role(self, user_id: str, role_id: str) -> None:
        with self.lock:
            if role_id not in self.members[user_id]["roles"]:
                self.members[user_id]["roles"].append(role_id)

    def set_overwrite(self, channel_id: str, target_id: str, *, allow: Iterable[str] = (), deny: Iterable[str] = (),
                      member: bool = False) -> None:
        """Set a channel permission overwrite for a role (or, with ``member=True``, for one member)."""
        with self.lock:
            ows = self.channels[channel_id]["permission_overwrites"]
            ows[:] = [o for o in ows if o["id"] != target_id]
            ows.append({"id": target_id, "type": 1 if member else 0, "allow": str(_mask(allow)), "deny": str(_mask(deny))})

    def remove_bot_from_guild(self) -> None:
        self.bot_in_guild = False

    def add_other_guild(self, name: str = "another server") -> str:
        """Another guild the bot is a member of (it then shows in ``/users/@me/guilds`` and in Ready)."""
        gid = self._snowflake()
        self.other_guilds.append({"id": gid, "name": name, "member_count": 3})
        return gid

    def add_foreign_invite(self, guild_id: str = "") -> str:
        """An invite that belongs to another guild."""
        gid = guild_id or self._snowflake()
        code = "fgn" + hashlib.sha256(self._snowflake().encode()).hexdigest()[:6]
        self.invites[code] = {"type": 0, "code": code, "guild": {"id": gid, "name": "elsewhere"},
                              "channel": {"id": self._snowflake(), "name": "lobby", "type": 0},
                              "inviter": self._user("someone", False), "uses": 0, "max_uses": 0, "max_age": 0,
                              "temporary": False, "created_at": _iso(self.time()), "expires_at": None}
        return code

    def set_rate_limit(self, limit: int, window_s: float) -> None:
        with self.lock:
            self.rate_limit, self._buckets = (limit, window_s), {}

    def block_globally(self, seconds: float) -> None:
        self.global_block_until = self.time() + seconds

    def humans(self) -> List[str]:
        return [uid for uid, m in self.members.items() if not m["user"].get("bot") and uid not in self.test_account_ids]

    def mark_interested(self, event_id: str, n: int = 1) -> None:
        with self.lock:
            self.events[event_id]["user_count"] += n

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _message(self, channel_id: str, author: Dict[str, Any], content: str) -> Dict[str, Any]:
        return {"id": self._snowflake(), "channel_id": channel_id, "author": dict(author), "content": content,
                "timestamp": _iso(self.time()), "edited_timestamp": None, "tts": False, "mention_everyone": False,
                "mentions": [], "mention_roles": [], "attachments": [], "embeds": [], "pinned": False, "type": 0}

    def _emit_message(self, msg: Dict[str, Any]) -> None:
        ch = self.channels.get(msg["channel_id"])
        if ch is not None and ch.get("guild_id") == self.guild_id:
            self._emit("MESSAGE_CREATE", dict(msg, guild_id=self.guild_id, channel_type=ch["type"]))

    def gateway_guild_create(self) -> Dict[str, Any]:
        """The Guild Create payload of the configured guild: the guild object with the documented extra fields."""
        return dict(self._guild_object(False), joined_at=self.members[self.bot_user["id"]]["joined_at"], large=False,
                    member_count=len(self.members), voice_states=[], presences=[], threads=[], stage_instances=[],
                    soundboard_sounds=[], members=[json.loads(json.dumps(self.members[self.bot_user["id"]]))],
                    channels=[dict(c) for c in self.channels.values() if c["guild_id"] == self.guild_id],
                    guild_scheduled_events=[dict(e) for e in self.events.values()])

    def _guild_object(self, with_counts: bool) -> Dict[str, Any]:
        g = dict(self.guild, roles=[dict(r) for r in self.roles.values()])
        if with_counts:
            g["approximate_member_count"] = len(self.members)
            g["approximate_presence_count"] = (self.presence if self.presence is not None
                                               else max(1, len(self.humans()) // 2))
        return g

    def _audit(self, action: str, target_id: Optional[str], reason: Optional[str]) -> None:
        entry = {"id": self._snowflake(), "user_id": self.bot_user["id"], "target_id": target_id,
                 "action_type": _AUDIT[action], "changes": []}
        if reason:
            entry["reason"] = unquote(reason)
        self.audit_log.append(entry)

    @staticmethod
    def _err(status: int, code: int, message: str, **extra: Any) -> Reply:
        return status, {}, dict({"message": message, "code": code}, **extra)

    # permissions, computed as documented (independent of the adapter's implementation)
    def _guild_perms(self, user_id: str) -> int:
        if user_id == self.guild["owner_id"]:
            return _ALL
        perms = int(self.roles[self.guild_id]["permissions"])
        for rid in self.members[user_id]["roles"]:
            if rid in self.roles:
                perms |= int(self.roles[rid]["permissions"])
        return _ALL if perms & _ADMIN else perms

    def _channel_perms(self, user_id: str, channel: Dict[str, Any]) -> int:
        base = self._guild_perms(user_id)
        if base & _ADMIN:
            return _ALL
        ows = {o["id"]: o for o in channel.get("permission_overwrites") or []}
        perms = base
        ev = ows.get(self.guild_id)
        if ev:
            perms = (perms & ~int(ev["deny"])) | int(ev["allow"])
        allow = deny = 0
        for rid in self.members[user_id]["roles"]:
            o = ows.get(rid)
            if o and o["type"] == 0:
                allow |= int(o["allow"])
                deny |= int(o["deny"])
        perms = (perms & ~deny) | allow
        own = ows.get(user_id)
        if own and own["type"] == 1:
            perms = (perms & ~int(own["deny"])) | int(own["allow"])
        return perms

    def bot_permissions(self, channel_id: Optional[str] = None) -> int:
        """The bot's effective permission integer in the guild, or in one of its channels."""
        with self.lock:
            me = self.bot_user["id"]
            return self._channel_perms(me, self.channels[channel_id]) if channel_id else self._guild_perms(me)

    def _has(self, permission: str, channel: Optional[Dict[str, Any]] = None) -> bool:
        if permission in self.refused:
            return False
        me = self.bot_user["id"]
        perms = self._channel_perms(me, channel) if channel is not None and channel.get("guild_id") else self._guild_perms(me)
        return bool(perms >> PERMISSION_BITS[permission] & 1)

    def _need(self, permission: str, channel: Optional[Dict[str, Any]] = None) -> Optional[Reply]:
        if not self._has(permission, channel):
            return self._err(403, 50013, "Missing Permissions")
        return None

    def _highest(self, user_id: str) -> int:
        return max([self.roles[r]["position"] for r in self.members[user_id]["roles"] if r in self.roles] or [0])

    def _rate(self, bucket_key: str) -> Tuple[Optional[Reply], Dict[str, str]]:
        now = self.time()
        if now < self.global_block_until:
            wait = self.global_block_until - now
            return (429, {"Retry-After": f"{wait:.3f}", "X-RateLimit-Global": "true", "X-RateLimit-Scope": "global"},
                    {"message": "You are being rate limited.", "retry_after": round(wait, 3), "global": True}), {}
        limit, window = self.rate_limit
        start, used = self._buckets.get(bucket_key, (now, 0))
        if now - start >= window:
            start, used = now, 0
        bucket = hashlib.sha256(bucket_key.encode()).hexdigest()[:16]
        reset_after = max(0.0, start + window - now)
        if used >= limit:
            headers = {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0",
                       "X-RateLimit-Reset": f"{start + window:.3f}", "X-RateLimit-Reset-After": f"{reset_after:.3f}",
                       "X-RateLimit-Bucket": bucket, "X-RateLimit-Scope": "user", "Retry-After": f"{reset_after:.3f}"}
            return (429, headers, {"message": "You are being rate limited.", "retry_after": round(reset_after, 3),
                                   "global": False}), {}
        used += 1
        self._buckets[bucket_key] = (start, used)
        return None, {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(limit - used),
                      "X-RateLimit-Reset": f"{start + window:.3f}", "X-RateLimit-Reset-After": f"{reset_after:.3f}",
                      "X-RateLimit-Bucket": bucket}

    # ---- request handling --------------------------------------------------------------------------------------
    def handle(self, method: str, url: str, headers: Dict[str, str], body: bytes) -> Reply:
        u = urlsplit(url)
        query = {k: v[-1] for k, v in parse_qs(u.query).items()}
        with self.lock:
            self.request_log.append({"method": method, "path": u.path, "query": query, "ts": self.time(),
                                     "authorized": headers.get("authorization") == f"Bot {self.token}",
                                     "audit_reason": headers.get("x-audit-log-reason")})
            status, hs, payload = self._route(method, u.path, query, headers, body)
            hs = dict(hs)
            hs[STAGING_HEADER] = STAGING_LABEL
            return status, hs, payload

    def _route(self, method: str, path: str, query: Dict[str, str], headers: Dict[str, str], body: bytes) -> Reply:
        if not path.startswith(API_PREFIX + "/"):
            return self._err(404, 0, "404: Not Found")
        if headers.get("authorization") != f"Bot {self.token}":
            return self._err(401, 0, "401: Unauthorized")
        parts = path[len(API_PREFIX) + 1:].strip("/").split("/")
        major = "/".join(parts[:2]) if parts[0] in ("guilds", "channels") else parts[0]
        template = "/".join(p if i < 2 or not p.isdigit() else "{id}" for i, p in enumerate(parts))
        limited, rate_headers = self._rate(f"{method} {major} {template}")
        if limited is not None:
            return limited
        payload: Any = None
        if body:
            try:
                payload = json.loads(body.decode())
            except ValueError:
                return self._err(400, 50109, "The request body contains invalid JSON.")
        status, hs, out = self._dispatch(method, parts, query, payload, headers.get("x-audit-log-reason"))
        return status, dict(rate_headers, **hs), out

    def _dispatch(self, method: str, parts: List[str], query: Dict[str, str], payload: Any,
                  reason: Optional[str] = None) -> Reply:
        if parts == ["users", "@me"] and method == "GET":
            return 200, {}, dict(self.bot_user)
        if parts == ["users", "@me", "guilds"] and method == "GET":
            mine = [{"id": self.guild_id, "name": self.guild["name"], "icon": None, "banner": None, "owner": False,
                     "permissions": str(self._guild_perms(self.bot_user["id"])), "features": list(self.guild["features"])}
                    ] if self.bot_in_guild else []
            mine += [{"id": g["id"], "name": g["name"], "icon": None, "banner": None, "owner": False,
                      "permissions": "0", "features": []} for g in self.other_guilds]
            return 200, {}, mine
        if parts == ["applications", "@me"] and method == "GET":
            return 200, {}, dict(self.application)
        if parts == ["gateway", "bot"] and method == "GET":
            if self.gateway is None:
                return self._err(503, 0, "503: the staging Gateway is not running")
            return 200, {}, {"url": self.gateway.url, "shards": 1,
                             "session_start_limit": {"total": 1000, "remaining": self.session_starts_remaining,
                                                     "reset_after": 14400000, "max_concurrency": 1}}
        if parts[0] == "invites" and len(parts) == 2:
            return self._invite_route(method, parts[1], reason)
        if parts[0] == "guilds" and len(parts) >= 2:
            if parts[1] != self.guild_id:
                return self._err(404, 10004, "Unknown Guild")
            if not self.bot_in_guild:                   # the application was not added to this server (yet)
                return self._err(404, 10004, "Unknown Guild") if len(parts) == 2 else self._err(403, 50001, "Missing Access")
            return self._guild_route(method, parts[2:], query, payload, reason)
        if parts[0] == "channels" and len(parts) >= 2:
            ch = self.channels.get(parts[1])
            if ch is None:
                return self._err(404, 10003, "Unknown Channel")
            if ch.get("guild_id") and not self.bot_in_guild:
                return self._err(403, 50001, "Missing Access")
            return self._channel_route(method, ch, parts[2:], query, payload, reason)
        return self._err(404, 0, "404: Not Found")

    # ---- guild routes ------------------------------------------------------------------------------------------
    def _guild_route(self, method: str, rest: List[str], query: Dict[str, str], payload: Any,
                     reason: Optional[str]) -> Reply:
        p = payload if isinstance(payload, dict) else {}
        if not rest and method == "GET":
            return 200, {}, self._guild_object(query.get("with_counts", "false").lower() == "true")
        if not rest and method == "PATCH":
            return self._modify_guild(p, reason)
        if rest == ["members"] and method == "GET":
            if not self.members_intent:
                return self._err(403, 50001, "Missing Access")
            try:
                limit, after = int(query.get("limit", "1")), int(query.get("after", "0"))
            except ValueError:
                return self._err(400, 50035, "Invalid Form Body")
            if not 1 <= limit <= 1000:
                return self._err(400, 50035, "Invalid Form Body",
                                 errors={"limit": {"_errors": [{"code": "NUMBER_TYPE_MAX",
                                                                "message": "int value should be between 1 and 1000."}]}})
            ids = sorted((i for i in self.members if int(i) > after), key=int)[:limit]
            return 200, {}, [json.loads(json.dumps(self.members[i])) for i in ids]
        if len(rest) == 2 and rest[0] == "members" and method == "GET":
            m = self.members.get(rest[1])
            if m is None:
                return self._err(404, 10007, "Unknown Member")
            return 200, {}, json.loads(json.dumps(m))
        if rest == ["channels"] and method == "GET":
            return 200, {}, [dict(c) for c in self.channels.values()
                             if c["guild_id"] == self.guild_id and self._has("VIEW_CHANNEL", c)]
        if rest == ["channels"] and method == "POST":
            denied = self._need("MANAGE_CHANNELS")
            if denied:
                return denied
            name = p.get("name")
            ctype = p.get("type", GUILD_TEXT)
            if not isinstance(name, str) or not 1 <= len(name) <= 100 or ctype not in (
                    GUILD_TEXT, GUILD_VOICE, GUILD_CATEGORY, GUILD_ANNOUNCEMENT, GUILD_FORUM):
                return self._err(400, 50035, "Invalid Form Body")
            if sum(1 for c in self.channels.values() if c["guild_id"] == self.guild_id) >= 500:
                return self._err(400, 30013, "Maximum number of guild channels reached (500)")
            parent = p.get("parent_id")
            if parent is not None and (parent not in self.channels or self.channels[parent]["type"] != GUILD_CATEGORY):
                return self._err(400, 50035, "Invalid Form Body")
            cid = self._snowflake()
            self.channels[cid] = {"id": cid, "type": ctype, "guild_id": self.guild_id, "name": name,
                                  "position": len(self.channels), "parent_id": parent, "topic": p.get("topic"),
                                  "nsfw": bool(p.get("nsfw", False)), "rate_limit_per_user": 0,
                                  "permission_overwrites": []}
            self.messages[cid] = []
            self._audit("CHANNEL_CREATE", cid, reason)
            self._emit("CHANNEL_CREATE", dict(self.channels[cid]))
            return 201, {}, dict(self.channels[cid])
        if rest == ["roles"] and method == "GET":
            return 200, {}, [dict(r) for r in self.roles.values()]
        if rest and rest[0] == "roles":
            return self._role_route(method, rest[1:], p, reason)
        if rest == ["invites"] and method == "GET":
            meta = self._has("MANAGE_GUILD")
            if "MANAGE_GUILD" in self.refused or (not meta and not self._has("VIEW_AUDIT_LOG")):
                return self._err(403, 50013, "Missing Permissions")
            mine = [i for i in self.invites.values() if i["guild"]["id"] == self.guild_id]
            drop = () if meta else ("uses", "max_uses", "max_age", "temporary", "created_at")
            return 200, {}, [{k: v for k, v in json.loads(json.dumps(i)).items() if k not in drop} for i in mine]
        if rest == ["audit-logs"] and method == "GET":
            denied = self._need("VIEW_AUDIT_LOG")
            if denied:
                return denied
            try:
                limit = int(query.get("limit", "50"))
            except ValueError:
                return self._err(400, 50035, "Invalid Form Body")
            if not 1 <= limit <= 100:
                return self._err(400, 50035, "Invalid Form Body")
            entries = [dict(e) for e in reversed(self.audit_log)][:limit]            # most recent first
            return 200, {}, {"audit_log_entries": entries, "users": [dict(self.bot_user)], "webhooks": [],
                             "integrations": [], "threads": [], "application_commands": [],
                             "auto_moderation_rules": [], "guild_scheduled_events": []}
        if rest == ["scheduled-events"] and method == "GET":
            with_count = query.get("with_user_count", "false").lower() == "true"
            return 200, {}, [{k: v for k, v in e.items() if with_count or k != "user_count"}
                             for e in self.events.values()]
        if rest == ["scheduled-events"] and method == "POST":
            return self._create_event(payload, reason)
        return self._err(404, 0, "404: Not Found")

    def _modify_guild(self, p: Dict[str, Any], reason: Optional[str]) -> Reply:
        denied = self._need("MANAGE_GUILD")
        if denied:
            return denied
        levels = {"verification_level": 4, "default_message_notifications": 1, "explicit_content_filter": 2}
        new: Dict[str, Any] = {}
        for k, v in p.items():
            if k == "name":
                if not isinstance(v, str) or not 2 <= len(v.strip()) <= 100:
                    return self._err(400, 50035, "Invalid Form Body",
                                     errors={"name": {"_errors": [{"code": "BASE_TYPE_BAD_LENGTH",
                                                                   "message": "Must be between 2 and 100 in length."}]}})
                new[k] = v.strip()
            elif k in levels:
                if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= levels[k]):
                    return self._err(400, 50035, "Invalid Form Body")
                new[k] = 0 if v is None else v
            elif k in ("system_channel_id", "rules_channel_id", "public_updates_channel_id"):
                if v is not None and (v not in self.channels or self.channels[v]["guild_id"] != self.guild_id):
                    return self._err(400, 50035, "Invalid Form Body")
                new[k] = v
            elif k in ("description", "preferred_locale"):
                if v is not None and not isinstance(v, str):
                    return self._err(400, 50035, "Invalid Form Body")
                new[k] = v if k == "description" or v else "en-US"
            else:
                return self._err(400, 50035, "Invalid Form Body")          # a field this imitation does not model
        self.guild.update(new)
        self._audit("GUILD_UPDATE", self.guild_id, reason)
        self._emit("GUILD_UPDATE", self._guild_object(False))
        return 200, {}, self._guild_object(False)

    def _role_route(self, method: str, rest: List[str], p: Dict[str, Any], reason: Optional[str]) -> Reply:
        me = self.bot_user["id"]
        denied = self._need("MANAGE_ROLES")
        if denied:
            return denied

        def fields(target: Dict[str, Any]) -> Optional[Reply]:
            if "name" in p:
                if not isinstance(p["name"], str) or len(p["name"]) > 100:
                    return self._err(400, 50035, "Invalid Form Body")
                target["name"] = p["name"]
            if "permissions" in p and p["permissions"] is not None:
                try:
                    want = int(p["permissions"])
                except (TypeError, ValueError):
                    return self._err(400, 50035, "Invalid Form Body")
                if want & ~self._guild_perms(me):                 # a bot can only grant permissions it has
                    return self._err(403, 50013, "Missing Permissions")
                target["permissions"] = str(want)
            color = (p.get("colors") or {}).get("primary_color", p.get("color"))
            if color is not None:
                if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 0xFFFFFF:
                    return self._err(400, 50035, "Invalid Form Body")
                target["color"] = color
                target["colors"] = {"primary_color": color, "secondary_color": None, "tertiary_color": None}
            for k in ("hoist", "mentionable"):
                if k in p and p[k] is not None:
                    target[k] = bool(p[k])
            return None

        if not rest and method == "POST":
            if len(self.roles) >= 250:
                return self._err(400, 30005, "Maximum number of guild roles reached (250)")
            # "permissions" defaults to the @everyone permissions of the guild, as documented
            role = {"name": "new role", "permissions": self.roles[self.guild_id]["permissions"], "color": 0,
                    "colors": {"primary_color": 0, "secondary_color": None, "tertiary_color": None}, "hoist": False,
                    "mentionable": False}
            bad = fields(role)
            if bad:
                return bad
            rid = self._snowflake()
            for r in self.roles.values():                         # a new role is placed just above @everyone
                if r["position"] >= 1:
                    r["position"] += 1
            self.roles[rid] = dict(role, id=rid, icon=None, unicode_emoji=None, position=1, managed=False, flags=0)
            self._audit("ROLE_CREATE", rid, reason)
            self._emit("GUILD_ROLE_CREATE", {"guild_id": self.guild_id, "role": dict(self.roles[rid])})
            return 200, {}, dict(self.roles[rid])
        if len(rest) == 1 and method in ("PATCH", "DELETE"):
            role = self.roles.get(rest[0])
            if role is None:
                return self._err(404, 10011, "Unknown Role")
            if method == "DELETE" and role["id"] == self.guild_id:
                return self._err(400, 50028, "Invalid Role")
            if role["id"] != self.guild_id and role["position"] >= self._highest(me) and not self.guild["owner_id"] == me:
                return self._err(403, 50013, "Missing Permissions")      # not below the bot's highest role
            if method == "DELETE":
                if role["managed"]:
                    return self._err(403, 50013, "Missing Permissions")
                del self.roles[role["id"]]
                for m in self.members.values():
                    m["roles"] = [r for r in m["roles"] if r != role["id"]]
                self._audit("ROLE_DELETE", role["id"], reason)
                self._emit("GUILD_ROLE_DELETE", {"guild_id": self.guild_id, "role_id": role["id"]})
                return 204, {}, None
            bad = fields(role)
            if bad:
                return bad
            self._audit("ROLE_UPDATE", role["id"], reason)
            self._emit("GUILD_ROLE_UPDATE", {"guild_id": self.guild_id, "role": dict(role)})
            return 200, {}, dict(role)
        return self._err(404, 0, "404: Not Found")

    # ---- channel routes ----------------------------------------------------------------------------------------
    def _channel_route(self, method: str, ch: Dict[str, Any], rest: List[str], query: Dict[str, str], payload: Any,
                       reason: Optional[str]) -> Reply:
        p = payload if isinstance(payload, dict) else {}
        in_guild = ch.get("guild_id") == self.guild_id
        if not rest and method == "GET":
            return 200, {}, dict(ch)
        if in_guild and not self._has("VIEW_CHANNEL", ch) and "VIEW_CHANNEL" not in self.refused:
            return self._err(403, 50001, "Missing Access")
        if not rest and method in ("PATCH", "DELETE"):
            if not in_guild:
                return self._err(403, 50003, "Cannot execute action on a DM channel")
            denied = self._need("MANAGE_CHANNELS", ch)
            if denied:
                return denied
            if method == "DELETE":
                if ch["id"] in (self.guild.get("rules_channel_id"), self.guild.get("public_updates_channel_id")):
                    return self._err(400, 50074, "Cannot delete a channel required for Community guilds")
                del self.channels[ch["id"]]
                self.messages.pop(ch["id"], None)
                for other in self.channels.values():
                    if other.get("parent_id") == ch["id"]:
                        other["parent_id"] = None
                self.invites = {c: i for c, i in self.invites.items() if i["channel"]["id"] != ch["id"]}
                self._audit("CHANNEL_DELETE", ch["id"], reason)
                self._emit("CHANNEL_DELETE", dict(ch))
                return 200, {}, dict(ch)
            return self._modify_channel(ch, p, reason)
        if len(rest) == 2 and rest[0] == "permissions" and method == "PUT":
            return self._edit_overwrite(ch, rest[1], p, reason)
        if rest == ["messages"] and method == "GET":
            if "READ_MESSAGE_HISTORY" in self.refused:
                return self._err(403, 50013, "Missing Permissions")
            try:
                limit = int(query.get("limit", "50"))
            except ValueError:
                return self._err(400, 50035, "Invalid Form Body")
            if not 1 <= limit <= 100:
                return self._err(400, 50035, "Invalid Form Body")
            if in_guild and not self._has("READ_MESSAGE_HISTORY", ch):
                return 200, {}, []                              # documented: no messages are returned
            return 200, {}, [json.loads(json.dumps(m)) for m in reversed(self.messages[ch["id"]][-limit:])]
        if rest == ["messages"] and method == "POST":
            if ch["type"] in (GUILD_VOICE, GUILD_CATEGORY):
                return self._err(400, 50008, "Cannot send messages in a non-text channel")
            denied = self._need("SEND_MESSAGES", ch)
            if denied:
                return denied
            content = p.get("content")
            if not isinstance(content, str) or not content.strip():
                return self._err(400, 50006, "Cannot send an empty message")
            if len(content) > 2000:
                return self._err(400, 50035, "Invalid Form Body",
                                 errors={"content": {"_errors": [{"code": "BASE_TYPE_MAX_LENGTH",
                                                                  "message": "Must be 2000 or fewer in length."}]}})
            nonce = p.get("nonce")
            if nonce is not None and p.get("enforce_nonce") and (ch["id"], str(nonce)) in self._nonces:
                return 200, {}, json.loads(json.dumps(self._nonces[(ch["id"], str(nonce))]))
            msg = self._message(ch["id"], self.bot_user, content)
            if nonce is not None:
                msg["nonce"] = nonce
                self._nonces[(ch["id"], str(nonce))] = msg
            self.messages[ch["id"]].append(msg)
            self._emit_message(msg)
            return 200, {}, json.loads(json.dumps(msg))
        if len(rest) == 3 and rest[:2] == ["messages", "pins"] and method in ("PUT", "DELETE"):
            msg = next((m for m in self.messages[ch["id"]] if m["id"] == rest[2]), None)
            if msg is None:
                return self._err(404, 10008, "Unknown Message")
            denied = self._need("PIN_MESSAGES", ch)
            if denied:
                return denied
            if method == "PUT" and not msg["pinned"] and sum(1 for m in self.messages[ch["id"]] if m["pinned"]) >= 250:
                return self._err(400, 30003, "Maximum number of pins reached for the channel (250)")
            msg["pinned"] = method == "PUT"
            self._audit("MESSAGE_PIN" if method == "PUT" else "MESSAGE_UNPIN", msg["author"]["id"], reason)
            self._emit("CHANNEL_PINS_UPDATE", {"guild_id": self.guild_id, "channel_id": ch["id"]})
            return 204, {}, None
        if len(rest) == 2 and rest[0] == "messages" and method in ("GET", "PATCH", "DELETE"):
            msg = next((m for m in self.messages[ch["id"]] if m["id"] == rest[1]), None)
            if msg is None:
                return self._err(404, 10008, "Unknown Message")
            own = msg["author"]["id"] == self.bot_user["id"]
            if method == "GET":
                denied = self._need("READ_MESSAGE_HISTORY", ch) if in_guild else None
                return denied or (200, {}, json.loads(json.dumps(msg)))
            if method == "DELETE":
                denied = None if own else self._need("MANAGE_MESSAGES", ch)
                if denied:
                    return denied
                self.messages[ch["id"]].remove(msg)
                if not own:
                    self._audit("MESSAGE_DELETE", msg["author"]["id"], None)
                self._emit("MESSAGE_DELETE", {"id": msg["id"], "channel_id": ch["id"], "guild_id": self.guild_id})
                return 204, {}, None
            if not own:
                return self._err(403, 50005, "Cannot edit a message authored by another user")
            content = p.get("content")
            if not isinstance(content, str) or not content.strip() or len(content) > 2000:
                return self._err(400, 50035, "Invalid Form Body")
            msg["content"], msg["edited_timestamp"] = content, _iso(self.time())
            return 200, {}, json.loads(json.dumps(msg))
        if rest == ["invites"] and method == "POST":
            if not in_guild:
                return self._err(403, 50003, "Cannot execute action on a DM channel")
            denied = self._need("CREATE_INSTANT_INVITE", ch)
            if denied:
                return denied
            max_age, max_uses = p.get("max_age", 86400), p.get("max_uses", 0)
            if not isinstance(max_age, int) or not 0 <= max_age <= 604800 or not isinstance(max_uses, int) \
                    or not 0 <= max_uses <= 100:
                return self._err(400, 50035, "Invalid Form Body")
            if not p.get("unique"):
                same = next((i for i in self.invites.values() if i["channel"]["id"] == ch["id"]
                             and i["max_age"] == max_age and i["max_uses"] == max_uses), None)
                if same is not None:
                    return 200, {}, json.loads(json.dumps(same))
            code = hashlib.sha256(self._snowflake().encode()).hexdigest()[:8]
            now = self.time()
            inv = {"type": 0, "code": code, "guild": {"id": self.guild_id, "name": self.guild["name"]},
                   "channel": {"id": ch["id"], "name": ch["name"], "type": ch["type"]},
                   "inviter": dict(self.bot_user), "uses": 0, "max_uses": max_uses, "max_age": max_age,
                   "temporary": bool(p.get("temporary", False)), "created_at": _iso(now),
                   "expires_at": _iso(now + max_age) if max_age else None}
            self.invites[code] = inv
            self._audit("INVITE_CREATE", None, reason)
            self._emit("INVITE_CREATE", {"channel_id": ch["id"], "code": code, "created_at": inv["created_at"],
                                         "guild_id": self.guild_id, "inviter": dict(self.bot_user),
                                         "max_age": max_age, "max_uses": max_uses, "temporary": inv["temporary"],
                                         "uses": 0, "expires_at": inv["expires_at"]})
            return 200, {}, json.loads(json.dumps(inv))
        return self._err(404, 0, "404: Not Found")

    def _modify_channel(self, ch: Dict[str, Any], p: Dict[str, Any], reason: Optional[str]) -> Reply:
        new: Dict[str, Any] = {}
        for k, v in p.items():
            if k == "name":
                if not isinstance(v, str) or not 1 <= len(v) <= 100:
                    return self._err(400, 50035, "Invalid Form Body")
            elif k == "topic":
                if v is not None and (not isinstance(v, str) or len(v) > 1024):
                    return self._err(400, 50035, "Invalid Form Body")
            elif k == "position":
                if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                    return self._err(400, 50035, "Invalid Form Body")
            elif k == "parent_id":
                if v is not None and (v not in self.channels or self.channels[v]["type"] != GUILD_CATEGORY):
                    return self._err(400, 50035, "Invalid Form Body")
            elif k == "nsfw":
                if v is not None and not isinstance(v, bool):
                    return self._err(400, 50035, "Invalid Form Body")
                v = bool(v)
            elif k == "rate_limit_per_user":
                if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 21600):
                    return self._err(400, 50035, "Invalid Form Body")
                v = v or 0
            else:
                return self._err(400, 50035, "Invalid Form Body")          # a field this imitation does not model
            new[k] = v
        ch.update(new)
        self._audit("CHANNEL_UPDATE", ch["id"], reason)
        self._emit("CHANNEL_UPDATE", dict(ch))
        return 200, {}, dict(ch)

    def _edit_overwrite(self, ch: Dict[str, Any], target: str, p: Dict[str, Any], reason: Optional[str]) -> Reply:
        if ch.get("guild_id") != self.guild_id:
            return self._err(403, 50003, "Cannot execute action on a DM channel")
        denied = self._need("MANAGE_ROLES", ch)
        if denied:
            return denied
        kind = p.get("type")
        try:
            allow, deny = int(p.get("allow") or 0), int(p.get("deny") or 0)
        except (TypeError, ValueError):
            return self._err(400, 50035, "Invalid Form Body")
        if kind not in (0, 1) or allow < 0 or deny < 0:
            return self._err(400, 50035, "Invalid Form Body")
        if kind == 0 and target not in self.roles:
            return self._err(404, 10011, "Unknown Role")
        if kind == 1 and target not in self.members:
            return self._err(404, 10007, "Unknown Member")
        me = self.bot_user["id"]
        own = next((o for o in ch["permission_overwrites"] if o["id"] in [me] + self.members[me]["roles"]
                    and int(o["allow"]) >> PERMISSION_BITS["MANAGE_ROLES"] & 1), None)
        if own is None and (allow | deny) & ~self._guild_perms(me):   # only permissions the bot has in the guild
            return self._err(403, 50013, "Missing Permissions")
        existed = any(o["id"] == target for o in ch["permission_overwrites"])
        ch["permission_overwrites"] = [o for o in ch["permission_overwrites"] if o["id"] != target] + [
            {"id": target, "type": kind, "allow": str(allow), "deny": str(deny)}]
        self._audit("CHANNEL_OVERWRITE_UPDATE" if existed else "CHANNEL_OVERWRITE_CREATE", ch["id"], reason)
        self._emit("CHANNEL_UPDATE", dict(ch))
        return 204, {}, None

    # ---- invites and events ------------------------------------------------------------------------------------
    def _invite_route(self, method: str, code: str, reason: Optional[str]) -> Reply:
        inv = self.invites.get(code)
        if inv is None:
            return self._err(404, 10006, "Unknown Invite")
        plain = {k: v for k, v in json.loads(json.dumps(inv)).items()
                 if k not in ("uses", "max_uses", "max_age", "temporary", "created_at")}
        if method == "GET":
            return 200, {}, plain
        if method == "DELETE":
            ch = self.channels.get(inv["channel"]["id"])
            mine = inv["guild"]["id"] == self.guild_id and self.bot_in_guild
            if not mine or not (self._has("MANAGE_GUILD") or (ch is not None and self._has("MANAGE_CHANNELS", ch))):
                return self._err(403, 50013, "Missing Permissions")
            del self.invites[code]
            self._audit("INVITE_DELETE", None, reason)
            self._emit("INVITE_DELETE", {"channel_id": inv["channel"]["id"], "guild_id": self.guild_id, "code": code})
            return 200, {}, plain
        return self._err(405, 0, "405: Method Not Allowed")

    def _create_event(self, p: Any, reason: Optional[str] = None) -> Reply:
        denied = self._need("CREATE_EVENTS")
        if denied:
            return denied
        if not isinstance(p, dict) or not isinstance(p.get("name"), str) or not 1 <= len(p["name"]) <= 100 \
                or p.get("privacy_level") != 2 or p.get("entity_type") not in (1, 2, 3) \
                or not isinstance(p.get("scheduled_start_time"), str):
            return self._err(400, 50035, "Invalid Form Body")
        meta = p.get("entity_metadata") or {}
        if p["entity_type"] == 3:
            if not meta.get("location") or not p.get("scheduled_end_time"):
                return self._err(400, 50035, "Invalid Form Body")
        elif p.get("channel_id") not in self.channels:
            return self._err(400, 50035, "Invalid Form Body")
        else:
            ch = self.channels[p["channel_id"]]
            for need in ("VIEW_CHANNEL", "CONNECT"):
                denied = self._need(need, ch)
                if denied:
                    return denied
        if sum(1 for e in self.events.values() if e["status"] in (1, 2)) >= 100:
            return self._err(400, 30038, "Maximum number of guild scheduled events reached (100)")
        eid = self._snowflake()
        ev = {"id": eid, "guild_id": self.guild_id, "channel_id": p.get("channel_id"),
              "creator_id": self.bot_user["id"], "name": p["name"], "description": p.get("description"),
              "scheduled_start_time": p["scheduled_start_time"], "scheduled_end_time": p.get("scheduled_end_time"),
              "privacy_level": 2, "status": 1, "entity_type": p["entity_type"], "entity_id": None,
              "entity_metadata": meta or None, "user_count": 0}
        self.events[eid] = ev
        self._audit("GUILD_SCHEDULED_EVENT_CREATE", eid, reason)
        self._emit("GUILD_SCHEDULED_EVENT_CREATE", dict(ev))
        return 200, {}, dict(ev)


class StagingDiscordServer:
    """Serves a :class:`StagingDiscord` on a loopback port, together with its staging Gateway. ``base_url`` is
    what ``DiscordClient`` is given; ``gateway.url`` is what ``GET /gateway/bot`` answers."""

    environment = "staging"

    def __init__(self, state: Optional[StagingDiscord] = None, host: str = "127.0.0.1", port: int = 0, *,
                 gateway: bool = True, heartbeat_interval_ms: int = 45000):
        self.state = state or StagingDiscord()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True       # headers and body are separate writes; no 40 ms delayed-ACK stall

            def _serve(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                status, headers, payload = outer.state.handle(self.command, self.path,
                                                              {k.lower(): v for k, v in self.headers.items()}, body)
                raw = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = _serve

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.base_url = f"http://{host}:{self.httpd.server_address[1]}{API_PREFIX}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True,
                                        name="staging-discord")
        self._thread.start()
        self.gateway: Any = None
        if gateway:
            from .gateway import StagingGateway
            self.gateway = StagingGateway(self.state, host, heartbeat_interval_ms=heartbeat_interval_ms)
            self.state.gateway = self.gateway

    def close(self) -> None:
        if self.gateway is not None:
            self.gateway.close()
            if self.state.gateway is self.gateway:
                self.state.gateway = None
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> "StagingDiscordServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
