"""STAGING ONLY — local imitation of the Discord REST API subset used by ``cam_live.harness.discord``.

Implemented routes (prefix ``/api/v10``), with Discord's payload shapes and error bodies::

    GET   /users/@me
    GET   /guilds/{guild}?with_counts=true
    GET   /guilds/{guild}/members?limit=&after=          (needs the members intent)
    GET   /guilds/{guild}/channels
    POST  /guilds/{guild}/channels                        (MANAGE_CHANNELS)
    GET   /channels/{channel}
    GET   /channels/{channel}/messages?limit=             (READ_MESSAGE_HISTORY)
    POST  /channels/{channel}/messages                    (SEND_MESSAGES; nonce + enforce_nonce de-duplication)
    PATCH /channels/{channel}/messages/{message}
    GET   /guilds/{guild}/invites                         (MANAGE_GUILD)
    POST  /channels/{channel}/invites                     (CREATE_INSTANT_INVITE)
    GET   /guilds/{guild}/scheduled-events?with_user_count=true
    POST  /guilds/{guild}/scheduled-events                (MANAGE_EVENTS)

Authentication is the ``Authorization: Bot <token>`` header. Per-route rate limits are reported with
``X-RateLimit-Limit / -Remaining / -Reset / -Reset-After / -Bucket`` and enforced with HTTP 429, ``Retry-After``
and the JSON body ``{"message", "retry_after", "global"}``.

Test hooks (plain methods, not reachable over HTTP) add and remove human members, bot members and
operator-declared test accounts, post member messages, use invites, and change permissions and limits.

This is a staging service. Results obtained against it are never live results.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

STAGING_LABEL = "cam-live-staging-discord"
STAGING_HEADER = "X-Staging"
API_PREFIX = "/api/v10"
ALL_PERMISSIONS = ("VIEW_CHANNEL", "MANAGE_CHANNELS", "SEND_MESSAGES", "READ_MESSAGE_HISTORY",
                   "CREATE_INSTANT_INVITE", "MANAGE_GUILD", "MANAGE_EVENTS")
GUILD_TEXT, DM, GUILD_VOICE, GROUP_DM, GUILD_CATEGORY, GUILD_ANNOUNCEMENT, GUILD_FORUM = 0, 1, 2, 3, 4, 5, 15

Reply = Tuple[int, Dict[str, str], Any]


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()


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
        self.permissions = set(ALL_PERMISSIONS)
        self.rate_limit = rate_limit
        self.global_block_until = 0.0
        self._buckets: Dict[str, Tuple[float, int]] = {}       # bucket -> (window start, used)
        self.request_log: List[Dict[str, Any]] = []
        self.bot_user = {"id": self._snowflake(), "username": "staging-app", "discriminator": "0",
                         "global_name": None, "avatar": None, "bot": True}
        self.guild_id = self._snowflake()
        self.guild = {"id": self.guild_id, "name": guild_name, "description": None, "owner_id": self._snowflake(),
                      "features": [], "verification_level": 1}
        self.members: Dict[str, Dict[str, Any]] = {}
        self.test_account_ids: List[str] = []                   # what the operator would list in the task file
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.invites: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self.presence: Optional[int] = None
        self._nonces: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._add_member(self.bot_user, pending=False)
        self.general_id = self.add_channel("general")

    # ---- test hooks ----------------------------------------------------------------------------------------
    def _snowflake(self) -> str:
        with self.lock:
            self._seq += 1
            return str(self._seq)

    def _add_member(self, user: Dict[str, Any], *, pending: bool, joined_at: Optional[float] = None) -> str:
        with self.lock:
            self.members[user["id"]] = {"user": user, "nick": None, "roles": [], "deaf": False, "mute": False,
                                        "flags": 0, "pending": pending,
                                        "joined_at": _iso(self.time() if joined_at is None else joined_at)}
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
        uid = self._add_member(self._user(username, False), pending=False, joined_at=joined_at)
        self.test_account_ids.append(uid)
        return uid

    def remove_member(self, user_id: str) -> None:
        with self.lock:
            self.members.pop(user_id, None)

    def add_channel(self, name: str, type: int = GUILD_TEXT, *, guild: bool = True) -> str:
        cid = self._snowflake()
        with self.lock:
            self.channels[cid] = {"id": cid, "type": type, "guild_id": self.guild_id if guild else None, "name": name,
                                  "position": len(self.channels), "parent_id": None, "topic": None, "nsfw": False,
                                  "permission_overwrites": []}
            self.messages[cid] = []
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
            return msg["id"]

    def use_invite(self, code: str, username: Optional[str] = None) -> str:
        """A person accepts an invite: its use count rises and a human member joins."""
        with self.lock:
            self.invites[code]["uses"] += 1
        return self.add_human(username)

    def set_permission(self, name: str, allowed: bool) -> None:
        (self.permissions.add if allowed else self.permissions.discard)(name)

    def set_rate_limit(self, limit: int, window_s: float) -> None:
        with self.lock:
            self.rate_limit, self._buckets = (limit, window_s), {}

    def block_globally(self, seconds: float) -> None:
        self.global_block_until = self.time() + seconds

    def humans(self) -> List[str]:
        return [uid for uid, m in self.members.items() if not m["user"].get("bot") and uid not in self.test_account_ids]

    # ---- helpers -------------------------------------------------------------------------------------------
    def _message(self, channel_id: str, author: Dict[str, Any], content: str) -> Dict[str, Any]:
        return {"id": self._snowflake(), "channel_id": channel_id, "author": dict(author), "content": content,
                "timestamp": _iso(self.time()), "edited_timestamp": None, "tts": False, "mention_everyone": False,
                "mentions": [], "mention_roles": [], "attachments": [], "embeds": [], "pinned": False, "type": 0}

    @staticmethod
    def _err(status: int, code: int, message: str, **extra: Any) -> Reply:
        return status, {}, dict({"message": message, "code": code}, **extra)

    def _need(self, permission: str) -> Optional[Reply]:
        if permission not in self.permissions:
            return self._err(403, 50013, "Missing Permissions")
        return None

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

    # ---- request handling ----------------------------------------------------------------------------------
    def handle(self, method: str, url: str, headers: Dict[str, str], body: bytes) -> Reply:
        u = urlsplit(url)
        query = {k: v[-1] for k, v in parse_qs(u.query).items()}
        with self.lock:
            self.request_log.append({"method": method, "path": u.path, "query": query, "ts": self.time(),
                                     "authorized": headers.get("authorization") == f"Bot {self.token}"})
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
        status, hs, out = self._dispatch(method, parts, query, payload)
        return status, dict(rate_headers, **hs), out

    def _dispatch(self, method: str, parts: List[str], query: Dict[str, str], payload: Any) -> Reply:
        if parts == ["users", "@me"] and method == "GET":
            return 200, {}, dict(self.bot_user)
        if parts[0] == "guilds" and len(parts) >= 2:
            if parts[1] != self.guild_id:
                return self._err(404, 10004, "Unknown Guild")
            rest = parts[2:]
            if not rest and method == "GET":
                g = dict(self.guild)
                if query.get("with_counts", "false").lower() == "true":
                    g["approximate_member_count"] = len(self.members)
                    g["approximate_presence_count"] = (self.presence if self.presence is not None
                                                       else max(1, len(self.humans()) // 2))
                return 200, {}, g
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
            if rest == ["channels"] and method == "GET":
                return 200, {}, [dict(c) for c in self.channels.values() if c["guild_id"] == self.guild_id]
            if rest == ["channels"] and method == "POST":
                denied = self._need("MANAGE_CHANNELS")
                if denied:
                    return denied
                name = (payload or {}).get("name") if isinstance(payload, dict) else None
                ctype = (payload or {}).get("type", GUILD_TEXT) if isinstance(payload, dict) else GUILD_TEXT
                if not isinstance(name, str) or not 1 <= len(name) <= 100 or ctype not in (
                        GUILD_TEXT, GUILD_VOICE, GUILD_CATEGORY, GUILD_ANNOUNCEMENT, GUILD_FORUM):
                    return self._err(400, 50035, "Invalid Form Body")
                if sum(1 for c in self.channels.values() if c["guild_id"] == self.guild_id) >= 500:
                    return self._err(400, 30013, "Maximum number of guild channels reached (500)")
                parent = payload.get("parent_id")
                if parent is not None and (parent not in self.channels or self.channels[parent]["type"] != GUILD_CATEGORY):
                    return self._err(400, 50035, "Invalid Form Body")
                cid = self.add_channel(name, ctype)
                self.channels[cid].update(topic=payload.get("topic"), parent_id=parent)
                return 201, {}, dict(self.channels[cid])
            if rest == ["invites"] and method == "GET":
                denied = self._need("MANAGE_GUILD")
                if denied:
                    return denied
                return 200, {}, [json.loads(json.dumps(i)) for i in self.invites.values()]
            if rest == ["scheduled-events"] and method == "GET":
                with_count = query.get("with_user_count", "false").lower() == "true"
                return 200, {}, [{k: v for k, v in e.items() if with_count or k != "user_count"}
                                 for e in self.events.values()]
            if rest == ["scheduled-events"] and method == "POST":
                denied = self._need("MANAGE_EVENTS")
                if denied:
                    return denied
                return self._create_event(payload)
            return self._err(404, 0, "404: Not Found")
        if parts[0] == "channels" and len(parts) >= 2:
            ch = self.channels.get(parts[1])
            if ch is None:
                return self._err(404, 10003, "Unknown Channel")
            rest = parts[2:]
            if not rest and method == "GET":
                return 200, {}, dict(ch)
            if rest == ["messages"] and method == "GET":
                denied = self._need("READ_MESSAGE_HISTORY")
                if denied:
                    return denied
                try:
                    limit = int(query.get("limit", "50"))
                except ValueError:
                    return self._err(400, 50035, "Invalid Form Body")
                if not 1 <= limit <= 100:
                    return self._err(400, 50035, "Invalid Form Body")
                return 200, {}, [json.loads(json.dumps(m)) for m in reversed(self.messages[ch["id"]][-limit:])]
            if rest == ["messages"] and method == "POST":
                if ch["type"] in (GUILD_VOICE, GUILD_CATEGORY):
                    return self._err(400, 50008, "Cannot send messages in a non-text channel")
                denied = self._need("SEND_MESSAGES")
                if denied:
                    return denied
                content = (payload or {}).get("content") if isinstance(payload, dict) else None
                if not isinstance(content, str) or not content.strip():
                    return self._err(400, 50006, "Cannot send an empty message")
                if len(content) > 2000:
                    return self._err(400, 50035, "Invalid Form Body",
                                     errors={"content": {"_errors": [{"code": "BASE_TYPE_MAX_LENGTH",
                                                                      "message": "Must be 2000 or fewer in length."}]}})
                nonce = payload.get("nonce")
                if nonce is not None and payload.get("enforce_nonce") and (ch["id"], str(nonce)) in self._nonces:
                    return 200, {}, json.loads(json.dumps(self._nonces[(ch["id"], str(nonce))]))
                msg = self._message(ch["id"], self.bot_user, content)
                if nonce is not None:
                    msg["nonce"] = nonce
                    self._nonces[(ch["id"], str(nonce))] = msg
                self.messages[ch["id"]].append(msg)
                return 200, {}, json.loads(json.dumps(msg))
            if len(rest) == 2 and rest[0] == "messages" and method == "PATCH":
                msg = next((m for m in self.messages[ch["id"]] if m["id"] == rest[1]), None)
                if msg is None:
                    return self._err(404, 10008, "Unknown Message")
                if msg["author"]["id"] != self.bot_user["id"]:
                    return self._err(403, 50005, "Cannot edit a message authored by another user")
                content = (payload or {}).get("content") if isinstance(payload, dict) else None
                if not isinstance(content, str) or not content.strip() or len(content) > 2000:
                    return self._err(400, 50035, "Invalid Form Body")
                msg["content"], msg["edited_timestamp"] = content, _iso(self.time())
                return 200, {}, json.loads(json.dumps(msg))
            if rest == ["invites"] and method == "POST":
                denied = self._need("CREATE_INSTANT_INVITE")
                if denied:
                    return denied
                p = payload if isinstance(payload, dict) else {}
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
                return 200, {}, json.loads(json.dumps(inv))
            return self._err(404, 0, "404: Not Found")
        return self._err(404, 0, "404: Not Found")

    def _create_event(self, p: Any) -> Reply:
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
        if sum(1 for e in self.events.values() if e["status"] in (1, 2)) >= 100:
            return self._err(400, 30038, "Maximum number of guild scheduled events reached (100)")
        eid = self._snowflake()
        ev = {"id": eid, "guild_id": self.guild_id, "channel_id": p.get("channel_id"),
              "creator_id": self.bot_user["id"], "name": p["name"], "description": p.get("description"),
              "scheduled_start_time": p["scheduled_start_time"], "scheduled_end_time": p.get("scheduled_end_time"),
              "privacy_level": 2, "status": 1, "entity_type": p["entity_type"], "entity_id": None,
              "entity_metadata": meta or None, "user_count": 0}
        self.events[eid] = ev
        return 200, {}, dict(ev)

    def mark_interested(self, event_id: str, n: int = 1) -> None:
        with self.lock:
            self.events[event_id]["user_count"] += n


class StagingDiscordServer:
    """Serves a :class:`StagingDiscord` on a loopback port. ``base_url`` is what ``DiscordClient`` is given."""

    environment = "staging"

    def __init__(self, state: Optional[StagingDiscord] = None, host: str = "127.0.0.1", port: int = 0):
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

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> "StagingDiscordServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
