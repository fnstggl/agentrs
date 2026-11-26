"""cam_live.harness.discord — Discord through the official bot REST API only (spec 12.5).

Owned by workstream ES. The constraints encoded here are derived in ``docs/ES/DISCORD_COMPLIANCE.md`` from the
Discord Community Guidelines and the Discord Developer Policy.

* :class:`DiscordClient` talks to ``<base_url>`` (``https://discord.com/api/v10`` or the local staging service) with
  ``Authorization: Bot <token>``. There is no code path for user tokens, user accounts, direct messages, friend or
  relationship endpoints, or the gateway.
* **Data minimisation at the client.** Member objects are reduced to ``{id_hash, bot, joined_at, pending,
  test_account}`` before they leave :meth:`DiscordClient.list_members`; usernames, avatars, roles and raw ids are
  dropped. Message objects are reduced to counts; message text leaves the client only when the caller passes
  ``keep_content=True``, which :class:`DiscordExecutor` does only if the task's ``data_permissions`` contain
  ``discord_message_content: true``.
* **Rate limits are honoured, never circumvented.** Limits are read from the response headers, exhausted buckets
  are waited for (bounded) and a 429 is retried only after the stated ``retry_after``.
* **No credentials, no call.** Without a bot token nothing is sent; :class:`DiscordExecutor` lists its tools as
  ``available=False`` with the exact reason when the token or the authorized guild is missing.
* The executor acts only inside the one authorized guild; a channel that is not part of it (a direct-message
  channel, a channel of another server) is refused before any request about it is made.
* Results obtained from anything but ``discord.com`` are labelled ``staging`` in every observation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import httpx

from ..contracts import (ActionResult, ActionSpec, Clock, Cost, ExecContext, FailureClass, Level, Modality,
                         Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, ResultStatus, TaskSpec,
                         ToolDescriptor, Trust, WallClock, new_id)
from .failures import (FailureEvidence, RetryPolicy, authorization_result, classify, failed_result,
                       run_with_retries, wait)
from .http_api import RateLimiter, RateLimitInfo, parse_rate_limit, resolve_secret, schema_errors

LIVE_BASE_URL = "https://discord.com/api/v10"
#: TaskSpec.data_permissions key that must be true before message text may leave the client. Setting it asserts
#: that the express permission Discord requires for model-training use of message content is on record.
CONTENT_PERMISSION = "discord_message_content"
CHANNEL_TYPES = {"text": 0, "voice": 2, "category": 4, "announcement": 5, "forum": 15}
_DM_TYPES = (1, 3)
#: Discord JSON error codes that state a platform rule or platform-defined maximum (never retried or worked around)
_PLATFORM_CODES = {20001, 20002, 20028, 40006, 50007, 200000, 200001, 240000}
_PERMISSION_CODES = {50001, 50013, 40001, 50005}


class DiscordUnavailable(RuntimeError):
    """The integration cannot be used (no token / no authorized guild). Raised before any network activity."""


class DiscordAPIError(RuntimeError):
    def __init__(self, status: int, code: int, message: str, *, retry_after_s: Optional[float] = None,
                 is_global: bool = False, sent: bool = True):
        super().__init__(f"Discord API error {status} (code {code}): {message}")
        self.status, self.code, self.message = status, code, message
        self.retry_after_s, self.is_global, self.sent = retry_after_s, is_global, sent


def environment_of(base_url: str) -> str:
    host = (urlsplit(base_url).hostname or "").lower()
    return "live" if host in ("discord.com", "canary.discord.com", "ptb.discord.com") else "staging"


def member_id_hash(guild_id: str, user_id: str, salt: str = "") -> str:
    """One-way pseudonym of a member, stable within a guild: enough to count distinct members and to see that the
    same member is still present later. It is never reversed."""
    return hashlib.sha256(f"{salt}:{guild_id}:{user_id}".encode()).hexdigest()[:32]


class DiscordClient:
    """Minimal bot REST client. ``token_provider`` is called at the moment of each request."""

    def __init__(self, base_url: str, token_provider: Callable[[], Optional[str]], *, clock: Optional[Clock] = None,
                 sleep: Optional[Callable[[float], None]] = None, transport: Optional[httpx.BaseTransport] = None,
                 timeout_s: float = 20.0, max_rate_limit_wait_s: float = 30.0, max_rate_limit_retries: int = 2,
                 user_agent: str = "DiscordBot (https://localhost, 1.0)", id_salt: str = ""):
        self.base_url = base_url.rstrip("/")
        self.environment = environment_of(self.base_url)
        self._token, self.clock, self._sleep = token_provider, clock or WallClock(), sleep
        self.max_rate_limit_wait_s, self.max_rate_limit_retries = max_rate_limit_wait_s, max_rate_limit_retries
        self.user_agent, self.id_salt, self.timeout_s = user_agent, id_salt, timeout_s
        self.limiter = RateLimiter()
        self.last_rate_limit = RateLimitInfo()
        self.requests_sent = 0
        self.rate_limit_waits_s = 0.0
        self._http = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)

    def close(self) -> None:
        self._http.close()

    # ---- transport -----------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, *, route: str, params: Optional[Dict[str, Any]] = None,
                 body: Optional[Dict[str, Any]] = None) -> Any:
        token = self._token()
        if not token:
            raise DiscordUnavailable("blocked: no Discord bot token provided")
        key = f"{method} {route}"
        for attempt in range(self.max_rate_limit_retries + 1):
            pause = self.limiter.wait_needed(key, self.clock.now())
            if pause > self.max_rate_limit_wait_s:
                raise DiscordAPIError(429, 0, f"rate limit: {pause:.1f}s until the bucket resets; request not sent",
                                      retry_after_s=pause, sent=False)
            if pause > 0:
                self.rate_limit_waits_s += pause
                wait(self.clock, pause, self._sleep)
            resp = self._http.request(method, self.base_url + path, params=params,
                                      content=json.dumps(body).encode() if body is not None else None,
                                      headers={"Authorization": f"Bot {token}", "User-Agent": self.user_agent,
                                               "Content-Type": "application/json", "Accept": "application/json"},
                                      timeout=self.timeout_s)
            self.requests_sent += 1
            try:
                data = resp.json() if resp.content else None
            except ValueError:
                data = None
            info = parse_rate_limit(resp.headers, self.clock.now(), data)
            self.last_rate_limit = info
            self.limiter.update(key, info, resp.status_code, self.clock.now())
            if resp.status_code == 429:
                delay = info.retry_after_s if info.retry_after_s is not None else 1.0
                if attempt < self.max_rate_limit_retries and delay <= self.max_rate_limit_wait_s:
                    continue                       # the limiter now holds the stated wait; honour it and repeat
                raise DiscordAPIError(429, int((data or {}).get("code", 0) or 0),
                                      str((data or {}).get("message", "rate limited")), retry_after_s=delay,
                                      is_global=info.is_global)
            if resp.status_code >= 400:
                d = data if isinstance(data, dict) else {}
                raise DiscordAPIError(resp.status_code, int(d.get("code", 0) or 0),
                                      str(d.get("message", resp.reason_phrase))[:300])
            return data
        raise DiscordAPIError(429, 0, "rate limited", retry_after_s=self.max_rate_limit_wait_s)

    # ---- guild ---------------------------------------------------------------------------------------------
    def guild(self, guild_id: str, with_counts: bool = True) -> Dict[str, Any]:
        g = self._request("GET", f"/guilds/{guild_id}", route=f"/guilds/{guild_id}",
                          params={"with_counts": "true"} if with_counts else None)
        return {"id": g["id"], "name": g.get("name", ""), "description": g.get("description"),
                "approximate_member_count": g.get("approximate_member_count"),
                "approximate_presence_count": g.get("approximate_presence_count")}

    def list_members(self, guild_id: str, *, test_account_ids: Iterable[str] = (), page_size: int = 1000,
                     max_pages: int = 1000) -> List[Dict[str, Any]]:
        """All members, paged with ``after``. Each is reduced to ``{id_hash, bot, joined_at, pending,
        test_account}``; nothing else about a member leaves this method."""
        tests = {str(t) for t in test_account_ids}
        out: List[Dict[str, Any]] = []
        after = "0"
        for _ in range(max_pages):
            page = self._request("GET", f"/guilds/{guild_id}/members", route=f"/guilds/{guild_id}/members",
                                 params={"limit": str(page_size), "after": after})
            for m in page:
                uid = str(m["user"]["id"])
                out.append({"id_hash": member_id_hash(guild_id, uid, self.id_salt), "bot": bool(m["user"].get("bot", False)),
                            "joined_at": m.get("joined_at"), "pending": bool(m.get("pending", False)),
                            "test_account": uid in tests})
            if len(page) < page_size:
                break
            after = max((str(m["user"]["id"]) for m in page), key=int)
        return out

    @staticmethod
    def member_counts(members: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        bots = sum(1 for m in members if m["bot"])
        tests = sum(1 for m in members if m["test_account"] and not m["bot"])
        humans = [m for m in members if not m["bot"] and not m["test_account"]]
        return {"total": len(members), "bots": bots, "test_accounts": tests, "humans": len(humans),
                "pending": sum(1 for m in humans if m["pending"]),
                "humans_not_pending": sum(1 for m in humans if not m["pending"]),
                "distinct": len({m["id_hash"] for m in humans})}

    @staticmethod
    def member_retention(earlier: Sequence[Mapping[str, Any]], later: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        """Aggregate retention between two member snapshots: how many of the people (not bots, not declared test
        accounts) of the earlier snapshot are still members in the later one. Works on pseudonyms only."""
        cohort = {m["id_hash"] for m in earlier if not m["bot"] and not m["test_account"]}
        present = {m["id_hash"] for m in later if not m["bot"] and not m["test_account"]}
        return {"cohort": len(cohort), "retained": len(cohort & present), "left": len(cohort - present),
                "joined_since": len(present - cohort)}

    # ---- channels and messages -----------------------------------------------------------------------------
    @staticmethod
    def _channel(c: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": c["id"], "name": c.get("name", ""), "type": c.get("type"), "guild_id": c.get("guild_id"),
                "parent_id": c.get("parent_id"), "position": c.get("position")}

    def list_channels(self, guild_id: str) -> List[Dict[str, Any]]:
        return [self._channel(c) for c in self._request("GET", f"/guilds/{guild_id}/channels",
                                                        route=f"/guilds/{guild_id}/channels")]

    def channel(self, channel_id: str) -> Dict[str, Any]:
        return self._channel(self._request("GET", f"/channels/{channel_id}", route=f"/channels/{channel_id}"))

    def create_channel(self, guild_id: str, name: str, type: int = 0, *, topic: Optional[str] = None,
                       parent_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "type": type}
        if topic is not None:
            body["topic"] = topic
        if parent_id is not None:
            body["parent_id"] = parent_id
        return self._channel(self._request("POST", f"/guilds/{guild_id}/channels",
                                           route=f"/guilds/{guild_id}/channels", body=body))

    @staticmethod
    def _sent(m: Dict[str, Any], content: str) -> Dict[str, Any]:
        return {"id": m["id"], "channel_id": m["channel_id"], "timestamp": m.get("timestamp"),
                "edited_timestamp": m.get("edited_timestamp"),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "content_length": len(content)}

    def send_message(self, channel_id: str, content: str, *, nonce: Optional[str] = None) -> Dict[str, Any]:
        """Post as the bot. With ``nonce`` the request is idempotent (``enforce_nonce``): repeating it after an
        uncertain failure cannot produce a second message."""
        body: Dict[str, Any] = {"content": content, "allowed_mentions": {"parse": []}}
        if nonce:
            body.update(nonce=nonce, enforce_nonce=True)
        m = self._request("POST", f"/channels/{channel_id}/messages", route=f"/channels/{channel_id}/messages", body=body)
        return self._sent(m, content)

    def edit_message(self, channel_id: str, message_id: str, content: str) -> Dict[str, Any]:
        m = self._request("PATCH", f"/channels/{channel_id}/messages/{message_id}",
                          route=f"/channels/{channel_id}/messages/:id", body={"content": content})
        return self._sent(m, content)

    def channel_activity(self, channel_id: str, limit: int = 100, *, guild_id: str = "",
                         keep_content: bool = False) -> Dict[str, Any]:
        """Aggregate view of the most recent messages of a channel. Message text and authors are dropped here
        unless ``keep_content`` is set; authors are only ever counted through one-way pseudonyms."""
        msgs = self._request("GET", f"/channels/{channel_id}/messages", route=f"/channels/{channel_id}/messages",
                             params={"limit": str(max(1, min(100, limit)))})
        authors = {member_id_hash(guild_id, str(m["author"]["id"]), self.id_salt) for m in msgs
                   if not m["author"].get("bot")}
        bot_n = sum(1 for m in msgs if m["author"].get("bot"))
        stamps = sorted(m["timestamp"] for m in msgs if m.get("timestamp"))
        out: Dict[str, Any] = {"messages": len(msgs), "bot_messages": bot_n, "human_messages": len(msgs) - bot_n,
                               "distinct_human_authors": len(authors), "oldest": stamps[0] if stamps else None,
                               "newest": stamps[-1] if stamps else None, "content_included": bool(keep_content)}
        if keep_content:
            out["contents"] = [str(m.get("content", "")) for m in msgs if not m["author"].get("bot")]
        return out

    # ---- invites and scheduled events ----------------------------------------------------------------------
    @staticmethod
    def _invite(i: Dict[str, Any]) -> Dict[str, Any]:
        return {"code": i["code"], "channel_id": (i.get("channel") or {}).get("id"), "uses": i.get("uses", 0),
                "max_uses": i.get("max_uses", 0), "max_age": i.get("max_age", 0), "temporary": i.get("temporary", False),
                "created_at": i.get("created_at"), "expires_at": i.get("expires_at"),
                "inviter_is_bot": bool((i.get("inviter") or {}).get("bot", False))}

    def list_invites(self, guild_id: str) -> List[Dict[str, Any]]:
        return [self._invite(i) for i in self._request("GET", f"/guilds/{guild_id}/invites",
                                                       route=f"/guilds/{guild_id}/invites")]

    def create_invite(self, channel_id: str, *, max_age: int = 86400, max_uses: int = 0, temporary: bool = False,
                      unique: bool = False) -> Dict[str, Any]:
        i = self._request("POST", f"/channels/{channel_id}/invites", route=f"/channels/{channel_id}/invites",
                          body={"max_age": max_age, "max_uses": max_uses, "temporary": temporary, "unique": unique})
        out = self._invite(i)
        out["url"] = f"https://discord.gg/{out['code']}"
        return out

    @staticmethod
    def _event(e: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": e["id"], "name": e.get("name", ""), "description": e.get("description"),
                "scheduled_start_time": e.get("scheduled_start_time"), "scheduled_end_time": e.get("scheduled_end_time"),
                "status": e.get("status"), "entity_type": e.get("entity_type"), "channel_id": e.get("channel_id"),
                "location": (e.get("entity_metadata") or {}).get("location"), "user_count": e.get("user_count")}

    def list_scheduled_events(self, guild_id: str, with_user_count: bool = True) -> List[Dict[str, Any]]:
        return [self._event(e) for e in self._request(
            "GET", f"/guilds/{guild_id}/scheduled-events", route=f"/guilds/{guild_id}/scheduled-events",
            params={"with_user_count": "true"} if with_user_count else None)]

    def create_scheduled_event(self, guild_id: str, name: str, scheduled_start_time: str, *,
                               scheduled_end_time: Optional[str] = None, description: Optional[str] = None,
                               location: Optional[str] = None, channel_id: Optional[str] = None,
                               channel_kind: str = "voice") -> Dict[str, Any]:
        """An external event (``location``) or one held in a voice / stage channel of the guild (``channel_id``)."""
        body: Dict[str, Any] = {"name": name, "privacy_level": 2, "scheduled_start_time": scheduled_start_time}
        if channel_id:
            body.update(entity_type=1 if channel_kind == "stage" else 2, channel_id=channel_id)
        else:
            body.update(entity_type=3, entity_metadata={"location": location or ""})
        if scheduled_end_time:
            body["scheduled_end_time"] = scheduled_end_time
        if description:
            body["description"] = description
        return self._event(self._request("POST", f"/guilds/{guild_id}/scheduled-events",
                                         route=f"/guilds/{guild_id}/scheduled-events", body=body))


def evidence_of(err: BaseException) -> FailureEvidence:
    """Observable evidence of a Discord failure, for :func:`cam_live.harness.failures.classify`."""
    if isinstance(err, DiscordUnavailable):
        return FailureEvidence(exception=PermissionError(str(err)), detail=str(err), modality=Modality.API)
    if isinstance(err, DiscordAPIError):
        platform = err.code in _PLATFORM_CODES or 30000 <= err.code < 31000
        ev = FailureEvidence(http_status=err.status, retry_after_s=err.retry_after_s, platform_rule=platform,
                             limit_exceeded=err.status == 429, modality=Modality.API,
                             detail=f"Discord {err.status} code {err.code}: {err.message}")
        if err.code in _PERMISSION_CODES and err.status not in (401, 403):
            ev.exception = PermissionError(err.message)
        return ev
    return FailureEvidence(exception=err, timed_out=isinstance(err, httpx.TimeoutException), modality=Modality.API)


# --------------------------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------------------------
_SNOWFLAKE = {"type": "string", "pattern": r"^\d{5,25}$"}
_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "discord.guild_stats": {"type": "object", "additionalProperties": False, "properties": {}},
    "discord.list_channels": {"type": "object", "additionalProperties": False, "properties": {}},
    "discord.create_channel": {"type": "object", "additionalProperties": False, "required": ["name"], "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "type": {"type": "string", "enum": sorted(CHANNEL_TYPES)},
        "topic": {"type": "string", "maxLength": 1024}, "parent_id": _SNOWFLAKE}},
    "discord.send_message": {"type": "object", "additionalProperties": False, "required": ["channel_id", "content"],
                             "properties": {"channel_id": _SNOWFLAKE,
                                            "content": {"type": "string", "minLength": 1, "maxLength": 2000}}},
    "discord.edit_message": {"type": "object", "additionalProperties": False,
                             "required": ["channel_id", "message_id", "content"],
                             "properties": {"channel_id": _SNOWFLAKE, "message_id": _SNOWFLAKE,
                                            "content": {"type": "string", "minLength": 1, "maxLength": 2000}}},
    "discord.channel_activity": {"type": "object", "additionalProperties": False, "required": ["channel_id"],
                                 "properties": {"channel_id": _SNOWFLAKE,
                                                "limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
    "discord.list_invites": {"type": "object", "additionalProperties": False, "properties": {}},
    "discord.create_invite": {"type": "object", "additionalProperties": False, "required": ["channel_id"], "properties": {
        "channel_id": _SNOWFLAKE, "max_age_s": {"type": "integer", "minimum": 0, "maximum": 604800},
        "max_uses": {"type": "integer", "minimum": 0, "maximum": 100}, "unique": {"type": "boolean"}}},
    "discord.list_events": {"type": "object", "additionalProperties": False, "properties": {}},
    "discord.create_event": {"type": "object", "additionalProperties": False, "required": ["name", "start_time"],
                             "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 100},
                                            "start_time": {"type": "string", "minLength": 10},
                                            "end_time": {"type": "string", "minLength": 10},
                                            "description": {"type": "string", "maxLength": 1000},
                                            "location": {"type": "string", "maxLength": 100},
                                            "channel_id": _SNOWFLAKE}},
}
_DESCRIPTIONS = {
    "discord.guild_stats": ("external_read", "Read aggregate statistics of the authorized server: approximate member "
                            "and presence counts and, where the members intent is granted, exact counts of human, "
                            "bot, pending and operator-declared test accounts."),
    "discord.list_channels": ("external_read", "List the channels of the authorized server: identifier, name, type, parent."),
    "discord.create_channel": ("external_write", "Create a text, voice, category, announcement or forum channel in the "
                               "authorized server."),
    "discord.send_message": ("external_write", "Post a message as the bot in a channel of the authorized server."),
    "discord.edit_message": ("external_write", "Replace the text of a message the bot posted earlier."),
    "discord.channel_activity": ("external_read", "Read aggregate activity of a channel of the authorized server: number "
                                 "of recent messages, of messages by bots and by people, and of distinct human "
                                 "authors. Message text is included only where the task's data permissions allow it."),
    "discord.list_invites": ("external_read", "List the invites of the authorized server with their use counts."),
    "discord.create_invite": ("external_write", "Create an invite link to a channel of the authorized server."),
    "discord.list_events": ("external_read", "List the scheduled events of the authorized server with the number of "
                            "interested users."),
    "discord.create_event": ("external_write", "Create a scheduled event in the authorized server, at an external "
                             "location or in one of its voice channels."),
}
_IDEMPOTENT = ("discord.guild_stats", "discord.list_channels", "discord.channel_activity", "discord.list_invites",
               "discord.list_events", "discord.send_message", "discord.edit_message")


class DiscordExecutor:
    """``ModalityExecutor`` (modality ``api``) exposing the Discord capabilities of one authorized guild."""

    modality = Modality.API

    def __init__(self, base_url: str = LIVE_BASE_URL, *, guild_id: str = "", token_secret: str = "discord_bot_token",
                 secrets: Any = None, test_account_ids: Sequence[str] = (),
                 data_permissions: Optional[Mapping[str, bool]] = None, id_salt: str = "",
                 transport: Optional[httpx.BaseTransport] = None, sleep: Optional[Callable[[float], None]] = None,
                 retry: Optional[RetryPolicy] = None, user_agent: str = "DiscordBot (https://localhost, 1.0)"):
        self.base_url, self.guild_id, self.token_secret = base_url.rstrip("/"), str(guild_id or ""), token_secret
        self.environment = environment_of(self.base_url)
        self._secrets, self.test_account_ids = secrets, tuple(str(t) for t in test_account_ids)
        self.data_permissions = dict(data_permissions or {})
        self.id_salt, self._transport, self._sleep, self.user_agent = id_salt, transport, sleep, user_agent
        self.retry = retry or RetryPolicy()
        self._client: Optional[DiscordClient] = None
        self._broker: Any = None
        self._channels: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_task(cls, task: TaskSpec, *, base_url: str = "", secrets: Any = None, **kw: Any) -> "DiscordExecutor":
        """Configuration from ``TaskSpec.resources["discord"]`` (``guild_id``, ``test_account_ids``, ``token_secret``,
        ``base_url``) and ``TaskSpec.data_permissions``."""
        r = dict(task.resources.get("discord") or {})
        return cls(base_url or r.get("base_url") or LIVE_BASE_URL, guild_id=str(r.get("guild_id") or ""),
                   token_secret=r.get("token_secret", "discord_bot_token"), secrets=secrets,
                   test_account_ids=r.get("test_account_ids") or (), data_permissions=task.data_permissions, **kw)

    # ---- availability --------------------------------------------------------------------------------------
    def availability(self, secrets: Any = None) -> Tuple[bool, str]:
        if not self.guild_id:
            return False, "blocked: no owner-authorized Discord guild is configured"
        if resolve_secret(secrets if secrets is not None else self._secrets, self.token_secret) is None:
            return False, f"blocked: no Discord bot token provided (secret {self.token_secret!r})"
        return True, ""

    def client(self, secrets: Any = None, clock: Optional[Clock] = None) -> DiscordClient:
        broker = secrets if secrets is not None else self._secrets
        if self._client is None or broker is not self._broker:
            if self._client is not None:
                self._client.close()
            self._broker = broker
            self._client = DiscordClient(self.base_url, lambda: resolve_secret(broker, self.token_secret), clock=clock,
                                         sleep=self._sleep, transport=self._transport, id_salt=self.id_salt,
                                         user_agent=self.user_agent)
        return self._client

    # ---- ModalityExecutor ----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        ok, why = self.availability()
        out = []
        for name, schema in _SCHEMAS.items():
            effect, text = _DESCRIPTIONS[name]
            out.append(ToolDescriptor(name=name, modality=Modality.API, description=text, arg_schema=schema,
                                      level=Level.PRIMITIVE, side_effects=effect,
                                      auth_scope="discord.read" if effect == "external_read" else "discord.write",
                                      est_cost=Cost(seconds=0.5, primitive_ops=1), available=ok, unavailable_reason=why))
        return out

    def validate(self, action: ActionSpec) -> List[str]:
        schema = _SCHEMAS.get(action.operation)
        if schema is None:
            return [f"unknown operation {action.operation!r} for the Discord executor"]
        errs = schema_errors(schema, action.arguments)
        a = action.arguments
        if not errs and action.operation == "discord.create_event" and not a.get("channel_id"):
            if not a.get("location") or not a.get("end_time"):
                errs.append("arguments: an event outside a channel needs 'location' and 'end_time'")
        return errs

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        clock, started = ctx.clock, ctx.clock.now()
        errs = self.validate(action)
        if errs:
            return failed_result(action, classify(FailureEvidence(arg_errors=errs, modality=Modality.API)), clock, started)
        ok, why = self.availability(ctx.secrets)
        if not ok:                                    # no credential or no authorized guild: nothing is sent
            rep = classify(FailureEvidence(exception=PermissionError(why), detail=why, modality=Modality.API))
            return failed_result(action, rep, clock, started, status=ResultStatus.DENIED)
        client = self.client(ctx.secrets, clock)
        name = action.operation
        policy = self.retry if name in _IDEMPOTENT else RetryPolicy({FailureClass.RESOURCE_LIMIT:
                                                                     RetryPolicy.DEFAULT_RULES[FailureClass.RESOURCE_LIMIT]})
        return run_with_retries(lambda n: self._attempt(action, ctx, client, started), policy, clock,
                                sleep=self._sleep, deadline=ctx.deadline, cancel=ctx.cancel)

    # ---- one attempt ---------------------------------------------------------------------------------------
    def _checked(self, action: ActionSpec, ctx: ExecContext, started: float, ops: List[PrimitiveOp], op_name: str,
                 args: Dict[str, Any], note: str = "") -> Optional[ActionResult]:
        op = PrimitiveOp(op=op_name, modality=Modality.API, started=ctx.clock.now(), note=note,
                         arguments=dict(args, guild_id=self.guild_id, environment=self.environment))
        ops.append(op)
        return authorization_result(action, ctx.check(op, action), ctx.clock, started, ops=ops)

    def _in_guild(self, client: DiscordClient, channel_id: str) -> Optional[Dict[str, Any]]:
        if channel_id not in self._channels:
            self._channels = {c["id"]: c for c in client.list_channels(self.guild_id)}
        c = self._channels.get(channel_id)
        if c is None or str(c.get("guild_id")) != self.guild_id or c.get("type") in _DM_TYPES:
            return None
        return c

    def _attempt(self, action: ActionSpec, ctx: ExecContext, client: DiscordClient, started: float) -> ActionResult:
        clock, a, name, ops = ctx.clock, action.arguments, action.operation, []
        check_args = {k: v for k, v in a.items()}
        if "content" in a:
            check_args["content_sha256"] = hashlib.sha256(a["content"].encode()).hexdigest()
        refused = self._checked(action, ctx, started, ops, name, check_args)
        if refused is not None:
            return refused
        sent_before = client.requests_sent
        try:
            if "channel_id" in a:
                if a["channel_id"] not in self._channels:
                    refused = self._checked(action, ctx, started, ops, "discord.list_channels", {},
                                            note="is the channel part of the authorized server")
                    if refused is not None:
                        return refused
                if self._in_guild(client, a["channel_id"]) is None:
                    why = "the channel is not a channel of the authorized server; nothing was read or sent"
                    rep = classify(FailureEvidence(exception=PermissionError(why), detail=why, modality=Modality.API))
                    ops[0].status, ops[0].failure_class, ops[0].ended = ResultStatus.DENIED, rep.failure_class, clock.now()
                    return failed_result(action, rep, clock, started, ops=ops, status=ResultStatus.DENIED)
            text, quantities, data = self._call(name, a, client, action)
        except (DiscordAPIError, DiscordUnavailable, httpx.HTTPError) as e:
            rep = classify(evidence_of(e))
            for op in ops:
                op.ended = op.ended or clock.now()
            ops[-1].status, ops[-1].failure_class = ResultStatus.FAILED, rep.failure_class
            return failed_result(action, rep, clock, started, ops=ops)
        now = clock.now()
        for op in ops:
            op.ended = op.ended or now
        rl = client.last_rate_limit
        quantities += rl.quantities("discord")
        data.update(environment=self.environment, staging=self.environment != "live", guild_id=self.guild_id,
                    requests=client.requests_sent - sent_before)
        obs = Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text[:4000], quantities=quantities,
                          provenance=Provenance(source=f"discord:{self.environment}", uri=f"{self.base_url}#{name}",
                                                retrieved_at=now, trust=Trust.UNTRUSTED),
                          parent_action_id=action.action_id, data=data)
        ops[0].obs_after = obs.obs_id
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=started, ended=now,
                            observations=[obs], ops=ops, verified=True,
                            cost=Cost(seconds=now - started, primitive_ops=len(ops)),
                            artifacts={k: str(v) for k, v in data.items() if k in ("message_id", "channel_id", "invite_code", "event_id")})

    def _call(self, name: str, a: Dict[str, Any], client: DiscordClient,
              action: ActionSpec) -> Tuple[str, List[Quantity], Dict[str, Any]]:
        g, label = self.guild_id, f"[{self.environment}] "
        count = OutcomeKind.COUNT
        if name == "discord.guild_stats":
            info = client.guild(g, with_counts=True)
            q = [Quantity("discord.members.approximate", float(info["approximate_member_count"] or 0), kind=count),
                 Quantity("discord.presence.approximate", float(info["approximate_presence_count"] or 0), kind=count)]
            data: Dict[str, Any] = {"guild": info, "members_listed": False}
            text = (f"{label}server statistics: about {info['approximate_member_count']} members, "
                    f"{info['approximate_presence_count']} present")
            try:
                counts = client.member_counts(client.list_members(g, test_account_ids=self.test_account_ids))
            except DiscordAPIError as e:
                if e.status != 403:
                    raise
                data["members_unavailable"] = f"member list not available ({e.message}); only approximate counts"
            else:
                data.update(members_listed=True, member_counts=counts)
                q += [Quantity(f"discord.members.{k}", float(v), kind=count) for k, v in counts.items()]
                text += (f"; listed {counts['total']}: {counts['humans']} people ({counts['pending']} pending), "
                         f"{counts['bots']} bots, {counts['test_accounts']} declared test accounts")
            return text, q, data
        if name == "discord.list_channels":
            chans = client.list_channels(g)
            self._channels = {c["id"]: c for c in chans}
            return (label + "channels: " + ", ".join(f"{c['name']} ({c['id']}, type {c['type']})" for c in chans),
                    [Quantity("discord.channels.count", float(len(chans)), kind=count)], {"channels": chans})
        if name == "discord.create_channel":
            c = client.create_channel(g, a["name"], CHANNEL_TYPES[a.get("type", "text")], topic=a.get("topic"),
                                      parent_id=a.get("parent_id"))
            self._channels[c["id"]] = c
            return f"{label}channel created: {c['name']} ({c['id']})", [], {"channel": c, "channel_id": c["id"]}
        if name == "discord.send_message":
            nonce = hashlib.sha256(action.action_id.encode()).hexdigest()[:25]
            m = client.send_message(a["channel_id"], a["content"], nonce=nonce)
            return (f"{label}message {m['id']} posted in channel {m['channel_id']} ({m['content_length']} characters)",
                    [Quantity("discord.message.length", float(m["content_length"]), kind=count)],
                    {"message": m, "message_id": m["id"], "channel_id": m["channel_id"]})
        if name == "discord.edit_message":
            m = client.edit_message(a["channel_id"], a["message_id"], a["content"])
            return (f"{label}message {m['id']} in channel {m['channel_id']} replaced ({m['content_length']} characters)",
                    [Quantity("discord.message.length", float(m["content_length"]), kind=count)],
                    {"message": m, "message_id": m["id"], "channel_id": m["channel_id"]})
        if name == "discord.channel_activity":
            keep = bool(self.data_permissions.get(CONTENT_PERMISSION, False))
            act = client.channel_activity(a["channel_id"], a.get("limit", 100), guild_id=g, keep_content=keep)
            q = [Quantity("discord.channel.messages", float(act["messages"]), kind=count),
                 Quantity("discord.channel.human_messages", float(act["human_messages"]), kind=count),
                 Quantity("discord.channel.bot_messages", float(act["bot_messages"]), kind=count),
                 Quantity("discord.channel.distinct_human_authors", float(act["distinct_human_authors"]), kind=count)]
            text = (f"{label}channel {a['channel_id']}: {act['messages']} recent messages, {act['human_messages']} by "
                    f"{act['distinct_human_authors']} people, {act['bot_messages']} by bots")
            if keep:
                text += "; texts: " + " | ".join(c[:200] for c in act["contents"][:20])
            else:
                text += "; message text not retained (no data permission)"
            return text, q, {"activity": act, "channel_id": a["channel_id"]}
        if name == "discord.list_invites":
            inv = client.list_invites(g)
            return (label + "invites: " + (", ".join(f"{i['code']} used {i['uses']} times" for i in inv) or "none"),
                    [Quantity("discord.invites.count", float(len(inv)), kind=count),
                     Quantity("discord.invites.uses", float(sum(i["uses"] for i in inv)), kind=count)], {"invites": inv})
        if name == "discord.create_invite":
            i = client.create_invite(a["channel_id"], max_age=a.get("max_age_s", 86400), max_uses=a.get("max_uses", 0),
                                     unique=a.get("unique", False))
            return (f"{label}invite {i['code']} to channel {i['channel_id']}: {i['url']}", [],
                    {"invite": i, "invite_code": i["code"], "channel_id": i["channel_id"]})
        if name == "discord.list_events":
            ev = client.list_scheduled_events(g)
            return (label + "scheduled events: " + (", ".join(f"{e['name']} at {e['scheduled_start_time']} "
                                                              f"({e['user_count']} interested)" for e in ev) or "none"),
                    [Quantity("discord.events.count", float(len(ev)), kind=count),
                     Quantity("discord.events.interested", float(sum(e["user_count"] or 0 for e in ev)), kind=count)],
                    {"events": ev})
        e = client.create_scheduled_event(g, a["name"], a["start_time"], scheduled_end_time=a.get("end_time"),
                                          description=a.get("description"), location=a.get("location"),
                                          channel_id=a.get("channel_id"))
        return f"{label}scheduled event created: {e['name']} ({e['id']})", [], {"event": e, "event_id": e["id"]}
