"""cam_live.harness.discord — Discord through the official bot interface only (spec 12.5).

Owned by workstream ES. The constraints encoded here are derived in ``docs/ES/DISCORD_COMPLIANCE.md`` from the
Discord Community Guidelines and the Discord Developer Policy. Routes, fields and permission requirements are
taken from the official API documentation (github.com/discord/discord-api-docs, commit 62ab2d5, read 2026-11-27).

* :class:`DiscordClient` talks to ``<base_url>`` (``https://discord.com/api/v10`` or the local staging service) with
  ``Authorization: Bot <token>``. There is no code path for user tokens, user accounts, direct messages, friend or
  relationship endpoints. The Gateway is used only for one bounded, receive-only window at a time
  (``cam_live.harness.discord_gateway``).
* **No operation on a member.** No tool gives or takes a role from a member, removes, bans, times out or renames a
  member, or writes to one. Role tools act on the role itself; channel permission overwrites are written for roles
  only (overwrite type 0), never for a member.
* **Data minimisation at the client.** Member objects are reduced to ``{id_hash, bot, joined_at, pending,
  test_account}`` before they leave :meth:`DiscordClient.list_members`; usernames, avatars, roles and raw ids are
  dropped. Message objects are reduced to counts; message text leaves the client only when the caller passes
  ``keep_content=True``, which :class:`DiscordExecutor` does only if the task's ``data_permissions`` contain
  ``discord_message_content: true``. Audit-log entries are reduced to counts per kind of action.
* **The operator's permission scope.** ``scope`` lists the Discord permissions the operator lets CAM use. A tool
  that needs a permission outside it is ``available=False`` and is refused before any request, even when the bot's
  role in the server would allow it.
* **Permissions are checked before the request.** The bot's effective permissions (roles, ``@everyone``, channel
  overwrites, computed as documented) are read once and cached for 300 s; a request the bot is not permitted to
  make is not sent. A 403 that arrives anyway drops the cache.
* **Rate limits are honoured, never circumvented.** Limits are read from the response headers, exhausted buckets
  are waited for (bounded) and a 429 is retried only after the stated ``retry_after``.
* **No credentials, no call.** Without a bot token nothing is sent; :class:`DiscordExecutor` lists its tools as
  ``available=False`` with the exact reason when the token or the authorized guild is missing.
* The executor acts only inside the one authorized guild; a channel, invite or role that is not part of it (a
  direct-message channel, something of another server) is refused before any request that would change it.
* Results obtained from anything but ``discord.com`` are labelled ``staging`` in every observation.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit

import httpx

from ..contracts import (ActionResult, ActionSpec, Clock, Cost, ExecContext, FailureClass, Level, Modality,
                         Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, ResultStatus, TaskSpec,
                         ToolDescriptor, Trust, WallClock, new_id)
from .discord_gateway import GatewayError, GatewayWindow, intents_needed
from .discord_perms import (ADMINISTRATOR, DEFAULT_SCOPE, NEVER_GRANTED, PERMISSION_BITS, PRIVILEGED_INTENTS,
                            base_permissions, bits_of, channel_permissions, granted_intents, member_id_hash,
                            names_of, normalise_scope, timed_out_permissions)
from .failures import (FailureEvidence, RetryPolicy, authorization_result, classify, failed_result,
                       run_with_retries, wait)
from .http_api import (RateLimiter, RateLimitInfo, grant_bound, granted_action, parse_rate_limit, resolve_secret,
                       schema_errors)

LIVE_BASE_URL = "https://discord.com/api/v10"
#: TaskSpec.data_permissions key that must be true before message text may leave the client. Setting it asserts
#: that the express permission Discord requires for model-training use of message content is on record.
CONTENT_PERMISSION = "discord_message_content"
CHANNEL_TYPES = {"text": 0, "voice": 2, "category": 4, "announcement": 5, "forum": 15}
_DM_TYPES = (1, 3)
#: Discord JSON error codes that state a platform rule or platform-defined maximum (never retried or worked around)
_PLATFORM_CODES = {20001, 20002, 20028, 40006, 50007, 200000, 200001, 240000}
_PERMISSION_CODES = {50001, 50013, 40001, 50005}
#: value of the ``X-Audit-Log-Reason`` header on every write whose route supports it
AUDIT_REASON = "CAM Live authorized action"
DISCORD_EPOCH_MS = 1420070400000
#: audit log event value -> name ("Audit Log Events", developers/resources/audit-log.mdx)
AUDIT_LOG_EVENTS: Dict[int, str] = {
    1: "GUILD_UPDATE", 10: "CHANNEL_CREATE", 11: "CHANNEL_UPDATE", 12: "CHANNEL_DELETE",
    13: "CHANNEL_OVERWRITE_CREATE", 14: "CHANNEL_OVERWRITE_UPDATE", 15: "CHANNEL_OVERWRITE_DELETE",
    16: "CHANNEL_POSITION_UPDATE", 20: "MEMBER_KICK", 21: "MEMBER_PRUNE", 22: "MEMBER_BAN_ADD",
    23: "MEMBER_BAN_REMOVE", 24: "MEMBER_UPDATE", 25: "MEMBER_ROLE_UPDATE", 26: "MEMBER_MOVE",
    27: "MEMBER_DISCONNECT", 28: "BOT_ADD", 30: "ROLE_CREATE", 31: "ROLE_UPDATE", 32: "ROLE_DELETE",
    33: "ROLE_POSITION_UPDATE", 40: "INVITE_CREATE", 41: "INVITE_UPDATE", 42: "INVITE_DELETE",
    50: "WEBHOOK_CREATE", 51: "WEBHOOK_UPDATE", 52: "WEBHOOK_DELETE", 60: "EMOJI_CREATE", 61: "EMOJI_UPDATE",
    62: "EMOJI_DELETE", 72: "MESSAGE_DELETE", 73: "MESSAGE_BULK_DELETE", 74: "MESSAGE_PIN", 75: "MESSAGE_UNPIN",
    80: "INTEGRATION_CREATE", 81: "INTEGRATION_UPDATE", 82: "INTEGRATION_DELETE", 83: "STAGE_INSTANCE_CREATE",
    84: "STAGE_INSTANCE_UPDATE", 85: "STAGE_INSTANCE_DELETE", 90: "STICKER_CREATE", 91: "STICKER_UPDATE",
    92: "STICKER_DELETE", 100: "GUILD_SCHEDULED_EVENT_CREATE", 101: "GUILD_SCHEDULED_EVENT_UPDATE",
    102: "GUILD_SCHEDULED_EVENT_DELETE", 110: "THREAD_CREATE", 111: "THREAD_UPDATE", 112: "THREAD_DELETE",
    121: "APPLICATION_COMMAND_PERMISSION_UPDATE", 130: "SOUNDBOARD_SOUND_CREATE", 131: "SOUNDBOARD_SOUND_UPDATE",
    132: "SOUNDBOARD_SOUND_DELETE", 140: "AUTO_MODERATION_RULE_CREATE", 141: "AUTO_MODERATION_RULE_UPDATE",
    142: "AUTO_MODERATION_RULE_DELETE", 143: "AUTO_MODERATION_BLOCK_MESSAGE",
    144: "AUTO_MODERATION_FLAG_TO_CHANNEL", 145: "AUTO_MODERATION_USER_COMMUNICATION_DISABLED",
    146: "AUTO_MODERATION_QUARANTINE_USER", 150: "CREATOR_MONETIZATION_REQUEST_CREATED",
    151: "CREATOR_MONETIZATION_TERMS_ACCEPTED", 163: "ONBOARDING_PROMPT_CREATE", 164: "ONBOARDING_PROMPT_UPDATE",
    165: "ONBOARDING_PROMPT_DELETE", 166: "ONBOARDING_CREATE", 167: "ONBOARDING_UPDATE",
    190: "HOME_SETTINGS_CREATE", 191: "HOME_SETTINGS_UPDATE", 192: "VOICE_CHANNEL_STATUS_CREATE",
    193: "VOICE_CHANNEL_STATUS_DELETE"}
GUILD_SETTING_FIELDS = ("name", "description", "verification_level", "default_message_notifications",
                        "explicit_content_filter", "system_channel_id", "rules_channel_id",
                        "public_updates_channel_id", "preferred_locale")
CHANNEL_SETTING_FIELDS = ("name", "topic", "position", "parent_id", "nsfw", "rate_limit_per_user")
ROLE_SETTING_FIELDS = ("name", "permissions", "color", "hoist", "mentionable")


class DiscordUnavailable(RuntimeError):
    """The integration cannot be used (no token / no authorized guild). Raised before any network activity."""


class DiscordAPIError(RuntimeError):
    def __init__(self, status: int, code: int, message: str, *, retry_after_s: Optional[float] = None,
                 is_global: bool = False, sent: bool = True):
        super().__init__(f"Discord API error {status} (code {code}): {message}")
        self.status, self.code, self.message = status, code, message
        self.retry_after_s, self.is_global, self.sent = retry_after_s, is_global, sent


class DiscordMissingPermission(PermissionError):
    """The service answered, but what it returned shows that the bot lacks a permission (for instance an invite
    list without use counts, which is what a bot without ``MANAGE_GUILD`` is given)."""


def environment_of(base_url: str) -> str:
    host = (urlsplit(base_url).hostname or "").lower()
    return "live" if host in ("discord.com", "canary.discord.com", "ptb.discord.com") else "staging"


def snowflake_iso(snowflake: Any) -> Optional[str]:
    """Creation time encoded in a snowflake id: ``(snowflake >> 22) + 1420070400000`` milliseconds."""
    try:
        ms = (int(snowflake) >> 22) + DISCORD_EPOCH_MS
    except (TypeError, ValueError):
        return None
    return _dt.datetime.fromtimestamp(ms / 1000.0, _dt.timezone.utc).isoformat()


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
        #: channel id -> (time read, permission overwrites): kept inside the adapter for the permission check
        self.overwrites: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._http = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)

    def close(self) -> None:
        self._http.close()

    # ---- transport -----------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, *, route: str, params: Optional[Dict[str, Any]] = None,
                 body: Optional[Dict[str, Any]] = None, reason: Optional[str] = None) -> Any:
        token = self._token()
        if not token:
            raise DiscordUnavailable("blocked: no Discord bot token provided")
        key = f"{method} {route}"
        headers = {"Authorization": f"Bot {token}", "User-Agent": self.user_agent,
                   "Content-Type": "application/json", "Accept": "application/json"}
        if reason:
            headers["X-Audit-Log-Reason"] = quote(reason[:512], safe=" ")
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
                                      headers=headers, timeout=self.timeout_s)
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

    # ---- identity ------------------------------------------------------------------------------------------
    def me(self) -> Dict[str, Any]:
        """The bot account the token belongs to."""
        u = self._request("GET", "/users/@me", route="/users/@me")
        return {"id": str(u["id"]), "name": u.get("username", ""), "bot": bool(u.get("bot", False))}

    def application(self) -> Dict[str, Any]:
        a = self._request("GET", "/applications/@me", route="/applications/@me")
        return {"id": str(a.get("id", "")), "name": a.get("name", ""), "bot_public": bool(a.get("bot_public", False)),
                "intents": granted_intents(a.get("flags"))}

    def guild_membership(self, guild_id: str, max_pages: int = 50) -> Tuple[bool, int]:
        """(is the bot a member of ``guild_id``, in how many other guilds is it). Only ids are looked at and only
        the count leaves this method."""
        inside, others, after = False, 0, None
        for _ in range(max_pages):
            params = {"limit": "200"}
            if after:
                params["after"] = after
            page = self._request("GET", "/users/@me/guilds", route="/users/@me/guilds", params=params)
            for g in page:
                if str(g["id"]) == str(guild_id):
                    inside = True
                else:
                    others += 1
            if len(page) < 200:
                break
            after = max((str(g["id"]) for g in page), key=int)
        return inside, others

    def gateway_bot(self) -> Dict[str, Any]:
        g = self._request("GET", "/gateway/bot", route="/gateway/bot")
        lim = g.get("session_start_limit") or {}
        return {"url": str(g.get("url", "")), "shards": g.get("shards"),
                "session_start_limit": {k: lim.get(k) for k in ("total", "remaining", "reset_after", "max_concurrency")}}

    # ---- guild ---------------------------------------------------------------------------------------------
    def guild(self, guild_id: str, with_counts: bool = True) -> Dict[str, Any]:
        g = self._request("GET", f"/guilds/{guild_id}", route=f"/guilds/{guild_id}",
                          params={"with_counts": "true"} if with_counts else None)
        return {"id": g["id"], "name": g.get("name", ""), "description": g.get("description"),
                "approximate_member_count": g.get("approximate_member_count"),
                "approximate_presence_count": g.get("approximate_presence_count")}

    @staticmethod
    def _settings(g: Dict[str, Any]) -> Dict[str, Any]:
        """The settings of a guild object. The owner id, role list and every user object are dropped."""
        out = {k: g.get(k) for k in GUILD_SETTING_FIELDS}
        out.update(id=g["id"], features=list(g.get("features") or []),
                   approximate_member_count=g.get("approximate_member_count"),
                   approximate_presence_count=g.get("approximate_presence_count"))
        return out

    def guild_settings(self, guild_id: str) -> Dict[str, Any]:
        return self._settings(self._request("GET", f"/guilds/{guild_id}", route=f"/guilds/{guild_id}",
                                            params={"with_counts": "true"}))

    def modify_guild(self, guild_id: str, fields: Mapping[str, Any], reason: str = AUDIT_REASON) -> Dict[str, Any]:
        body = {k: fields[k] for k in GUILD_SETTING_FIELDS if k in fields}
        return self._settings(self._request("PATCH", f"/guilds/{guild_id}", route=f"/guilds/{guild_id}", body=body,
                                            reason=reason))

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

    @staticmethod
    def member_changes(earlier: Sequence[Mapping[str, Any]], later: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Aggregate difference between two member snapshots, split into people, bots and declared test accounts.
        Works on pseudonyms only; the only per-member values returned are the join times of new people."""
        before = {m["id_hash"]: m for m in earlier}
        after = {m["id_hash"]: m for m in later}

        def split(ms: List[Mapping[str, Any]]) -> Dict[str, int]:
            bots = sum(1 for m in ms if m["bot"])
            tests = sum(1 for m in ms if m["test_account"] and not m["bot"])
            return {"people": len(ms) - bots - tests, "bots": bots, "test_accounts": tests}

        joined = [m for h, m in after.items() if h not in before]
        left = [m for h, m in before.items() if h not in after]
        return {"joined": split(joined), "left": split(left),
                "joined_at": sorted(str(m["joined_at"]) for m in joined
                                    if not m["bot"] and not m["test_account"] and m.get("joined_at"))}

    def own_member(self, guild_id: str, bot_id: str) -> Dict[str, Any]:
        """The bot's own member object, reduced to what the permission computation needs. Only ever called with the
        id of the bot itself."""
        m = self._request("GET", f"/guilds/{guild_id}/members/{bot_id}", route=f"/guilds/{guild_id}/members/:own")
        return {"roles": [str(r) for r in m.get("roles") or []],
                "communication_disabled_until": m.get("communication_disabled_until")}

    # ---- roles ---------------------------------------------------------------------------------------------
    @staticmethod
    def _role(r: Dict[str, Any], guild_id: str = "") -> Dict[str, Any]:
        colors = r.get("colors") or {}
        return {"id": str(r["id"]), "name": r.get("name", ""), "permissions": names_of(r.get("permissions")),
                "permission_bits": str(r.get("permissions") or "0"), "position": r.get("position", 0),
                "managed": bool(r.get("managed", False)), "mentionable": bool(r.get("mentionable", False)),
                "hoist": bool(r.get("hoist", False)), "color": colors.get("primary_color", r.get("color", 0)),
                "everyone": str(r["id"]) == str(guild_id)}

    def roles(self, guild_id: str) -> List[Dict[str, Any]]:
        return [self._role(r, guild_id) for r in self._request("GET", f"/guilds/{guild_id}/roles",
                                                               route=f"/guilds/{guild_id}/roles")]

    @staticmethod
    def _role_body(fields: Mapping[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if "name" in fields:
            body["name"] = fields["name"]
        if "permissions" in fields:
            body["permissions"] = str(bits_of(fields["permissions"]))
        if "color" in fields:
            body["colors"] = {"primary_color": int(fields["color"]), "secondary_color": None, "tertiary_color": None}
        for k in ("hoist", "mentionable"):
            if k in fields:
                body[k] = bool(fields[k])
        return body

    def create_role(self, guild_id: str, name: str, permissions: Iterable[str] = (), *, color: int = 0,
                    hoist: bool = False, mentionable: bool = False, reason: str = AUDIT_REASON) -> Dict[str, Any]:
        """Create a role. The permission set is always stated (an omitted one would default to the permissions of
        ``@everyone``)."""
        body = self._role_body({"name": name, "permissions": list(permissions), "color": color, "hoist": hoist,
                                "mentionable": mentionable})
        return self._role(self._request("POST", f"/guilds/{guild_id}/roles", route=f"/guilds/{guild_id}/roles",
                                        body=body, reason=reason), guild_id)

    def modify_role(self, guild_id: str, role_id: str, fields: Mapping[str, Any],
                    reason: str = AUDIT_REASON) -> Dict[str, Any]:
        return self._role(self._request("PATCH", f"/guilds/{guild_id}/roles/{role_id}",
                                        route=f"/guilds/{guild_id}/roles/:id", body=self._role_body(fields),
                                        reason=reason), guild_id)

    def delete_role(self, guild_id: str, role_id: str, reason: str = AUDIT_REASON) -> None:
        self._request("DELETE", f"/guilds/{guild_id}/roles/{role_id}", route=f"/guilds/{guild_id}/roles/:id",
                      reason=reason)

    # ---- channels and messages -----------------------------------------------------------------------------
    @staticmethod
    def _channel(c: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": c["id"], "name": c.get("name", ""), "type": c.get("type"), "guild_id": c.get("guild_id"),
                "parent_id": c.get("parent_id"), "position": c.get("position")}

    def _seen(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce a channel object; its permission overwrites stay inside the client."""
        self.overwrites[str(c["id"])] = (self.clock.now(), [
            {"id": str(o["id"]), "type": int(o.get("type", 0)), "allow": str(o.get("allow") or "0"),
             "deny": str(o.get("deny") or "0")} for o in c.get("permission_overwrites") or []])
        return self._channel(c)

    def list_channels(self, guild_id: str) -> List[Dict[str, Any]]:
        return [self._seen(c) for c in self._request("GET", f"/guilds/{guild_id}/channels",
                                                     route=f"/guilds/{guild_id}/channels")]

    def channel(self, channel_id: str) -> Dict[str, Any]:
        return self._seen(self._request("GET", f"/channels/{channel_id}", route=f"/channels/{channel_id}"))

    def create_channel(self, guild_id: str, name: str, type: int = 0, *, topic: Optional[str] = None,
                       parent_id: Optional[str] = None, reason: Optional[str] = AUDIT_REASON) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "type": type}
        if topic is not None:
            body["topic"] = topic
        if parent_id is not None:
            body["parent_id"] = parent_id
        return self._seen(self._request("POST", f"/guilds/{guild_id}/channels",
                                        route=f"/guilds/{guild_id}/channels", body=body, reason=reason))

    def modify_channel(self, channel_id: str, fields: Mapping[str, Any], reason: str = AUDIT_REASON) -> Dict[str, Any]:
        raw = self._request("PATCH", f"/channels/{channel_id}", route=f"/channels/{channel_id}",
                            body={k: fields[k] for k in CHANNEL_SETTING_FIELDS if k in fields}, reason=reason)
        out = self._seen(raw)
        out.update(topic=raw.get("topic"), nsfw=raw.get("nsfw"), rate_limit_per_user=raw.get("rate_limit_per_user"))
        return out

    def delete_channel(self, channel_id: str, reason: str = AUDIT_REASON) -> Dict[str, Any]:
        raw = self._request("DELETE", f"/channels/{channel_id}", route=f"/channels/{channel_id}", reason=reason)
        self.overwrites.pop(str(channel_id), None)
        return self._channel(raw) if isinstance(raw, dict) and "id" in raw else {"id": channel_id}

    def set_role_overwrite(self, channel_id: str, role_id: str, allow: Iterable[str], deny: Iterable[str],
                           reason: str = AUDIT_REASON) -> Dict[str, Any]:
        """Write the permission overwrite of one **role** in a channel (overwrite type 0). There is no way to
        write a member's overwrite through this client."""
        body = {"allow": str(bits_of(allow)), "deny": str(bits_of(deny)), "type": 0}
        self._request("PUT", f"/channels/{channel_id}/permissions/{role_id}",
                      route=f"/channels/{channel_id}/permissions/:id", body=body, reason=reason)
        self.overwrites.pop(str(channel_id), None)
        return {"channel_id": channel_id, "role_id": role_id, "allow": sorted(set(allow)), "deny": sorted(set(deny))}

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

    def message_meta(self, channel_id: str, message_id: str, bot_id: str) -> Dict[str, Any]:
        """Whose message it is, without its text or its author: ``own`` (posted by this bot), ``author_is_bot``,
        ``pinned`` and the time."""
        m = self._request("GET", f"/channels/{channel_id}/messages/{message_id}",
                          route=f"/channels/{channel_id}/messages/:id")
        author = m.get("author") or {}
        return {"id": str(m["id"]), "channel_id": str(m.get("channel_id", channel_id)),
                "own": str(author.get("id")) == str(bot_id), "author_is_bot": bool(author.get("bot", False)),
                "pinned": bool(m.get("pinned", False)), "timestamp": m.get("timestamp")}

    def delete_message(self, channel_id: str, message_id: str) -> None:
        self._request("DELETE", f"/channels/{channel_id}/messages/{message_id}",
                      route=f"/channels/{channel_id}/messages/:id")

    def pin_message(self, channel_id: str, message_id: str, reason: str = AUDIT_REASON) -> None:
        self._request("PUT", f"/channels/{channel_id}/messages/pins/{message_id}",
                      route=f"/channels/{channel_id}/messages/pins/:id", reason=reason)

    def unpin_message(self, channel_id: str, message_id: str, reason: str = AUDIT_REASON) -> None:
        self._request("DELETE", f"/channels/{channel_id}/messages/pins/{message_id}",
                      route=f"/channels/{channel_id}/messages/pins/:id", reason=reason)

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
        """The invites of the guild with their use counts. Use counts are invite metadata, which is included only
        with ``MANAGE_GUILD``; a list without them is reported as that missing permission, never as zero uses."""
        raw = self._request("GET", f"/guilds/{guild_id}/invites", route=f"/guilds/{guild_id}/invites")
        if any("uses" not in i for i in raw):
            raise DiscordMissingPermission("the invite list came without use counts: the bot lacks MANAGE_GUILD in "
                                           "this server; the server owner must grant it")
        return [self._invite(i) for i in raw]

    def create_invite(self, channel_id: str, *, max_age: int = 86400, max_uses: int = 0, temporary: bool = False,
                      unique: bool = False, reason: Optional[str] = AUDIT_REASON) -> Dict[str, Any]:
        i = self._request("POST", f"/channels/{channel_id}/invites", route=f"/channels/{channel_id}/invites",
                          body={"max_age": max_age, "max_uses": max_uses, "temporary": temporary, "unique": unique},
                          reason=reason)
        out = self._invite(i)
        out["url"] = f"https://discord.gg/{out['code']}"
        return out

    def invite(self, code: str) -> Dict[str, Any]:
        """Where an invite leads: its guild and channel ids. The inviter is dropped."""
        i = self._request("GET", f"/invites/{code}", route="/invites/:code")
        return {"code": i.get("code", code), "guild_id": str((i.get("guild") or {}).get("id") or ""),
                "channel_id": str((i.get("channel") or {}).get("id") or "")}

    def delete_invite(self, code: str, reason: str = AUDIT_REASON) -> None:
        self._request("DELETE", f"/invites/{code}", route="/invites/:code", reason=reason)

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
                               channel_kind: str = "voice", reason: Optional[str] = AUDIT_REASON) -> Dict[str, Any]:
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
                                         route=f"/guilds/{guild_id}/scheduled-events", body=body, reason=reason))

    # ---- audit log -----------------------------------------------------------------------------------------
    def audit_summary(self, guild_id: str, limit: int = 50) -> Dict[str, Any]:
        """Counts of the most recent audit-log entries per kind of action, and when the newest and the oldest of
        them were made. Who did what to whom, reasons and changes are dropped here."""
        log = self._request("GET", f"/guilds/{guild_id}/audit-logs", route=f"/guilds/{guild_id}/audit-logs",
                            params={"limit": str(max(1, min(100, limit)))})
        entries = (log or {}).get("audit_log_entries") or []
        counts: Dict[str, int] = {}
        for e in entries:
            kind = AUDIT_LOG_EVENTS.get(int(e.get("action_type", -1)), f"UNKNOWN_{e.get('action_type')}")
            counts[kind] = counts.get(kind, 0) + 1
        ids = sorted((int(e["id"]) for e in entries if str(e.get("id", "")).isdigit()))
        return {"entries": len(entries), "by_action": dict(sorted(counts.items())),
                "newest": snowflake_iso(ids[-1]) if ids else None, "oldest": snowflake_iso(ids[0]) if ids else None}


def evidence_of(err: BaseException) -> FailureEvidence:
    """Observable evidence of a Discord failure, for :func:`cam_live.harness.failures.classify`."""
    if isinstance(err, (DiscordUnavailable, DiscordMissingPermission)):
        return FailureEvidence(exception=PermissionError(str(err)), detail=str(err), modality=Modality.API)
    if isinstance(err, GatewayError):
        text = str(err)
        if err.kind == "permission":
            return FailureEvidence(exception=PermissionError(text), detail=text, modality=Modality.API)
        if err.kind == "limit":
            return FailureEvidence(limit_exceeded=True, detail=text, modality=Modality.API)
        if err.kind == "network":
            return FailureEvidence(exception=ConnectionError(text), detail=text, modality=Modality.API)
        return FailureEvidence(exception=RuntimeError(text), detail=text, modality=Modality.API)
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
# Tools
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Tool:
    effect: str                          # "external_read" | "external_write"
    text: str                            # what the tool does; never when or why to use it
    properties: Dict[str, Any]
    required: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()       # Discord permissions the route needs (per the API documentation)
    idempotent: bool = False             # safe to repeat after a transient failure


_SNOWFLAKE = {"type": "string", "pattern": r"^\d{5,25}$"}
_SNOWFLAKE_OR_NULL = {"type": ["string", "null"], "pattern": r"^\d{5,25}$"}
_PERMISSIONS = {"type": "array", "maxItems": len(PERMISSION_BITS),
                "items": {"type": "string", "enum": sorted(PERMISSION_BITS)}}
_MESSAGE = {"channel_id": _SNOWFLAKE, "message_id": _SNOWFLAKE}
_ROLE_FIELDS = {"name": {"type": "string", "minLength": 1, "maxLength": 100}, "permissions": _PERMISSIONS,
                "color": {"type": "integer", "minimum": 0, "maximum": 0xFFFFFF}, "hoist": {"type": "boolean"},
                "mentionable": {"type": "boolean"}}
_VIEW = ("VIEW_CHANNEL",)
TOOLS: Dict[str, _Tool] = {
    "discord.get_guild_stats": _Tool(
        "external_read", "Read aggregate statistics of the authorized server: approximate member and presence "
        "counts and, where the members intent is granted, exact counts of human, bot, pending and "
        "operator-declared test accounts.", {}, idempotent=True),
    "discord.list_channels": _Tool(
        "external_read", "List the channels of the authorized server: identifier, name, type, parent.", {},
        idempotent=True),
    "discord.create_channel": _Tool(
        "external_write", "Create a text, voice, category, announcement or forum channel in the authorized server.",
        {"name": {"type": "string", "minLength": 1, "maxLength": 100},
         "type": {"type": "string", "enum": sorted(CHANNEL_TYPES)},
         "topic": {"type": "string", "maxLength": 1024}, "parent_id": _SNOWFLAKE},
        ("name",), ("MANAGE_CHANNELS",)),
    "discord.send_message": _Tool(
        "external_write", "Post a message as the bot in a channel of the authorized server.",
        {"channel_id": _SNOWFLAKE, "content": {"type": "string", "minLength": 1, "maxLength": 2000}},
        ("channel_id", "content"), _VIEW + ("SEND_MESSAGES",), idempotent=True),
    "discord.edit_message": _Tool(
        "external_write", "Replace the text of a message the bot posted earlier.",
        dict(_MESSAGE, content={"type": "string", "minLength": 1, "maxLength": 2000}),
        ("channel_id", "message_id", "content"), _VIEW, idempotent=True),
    "discord.get_channel_activity": _Tool(
        "external_read", "Read aggregate activity of a channel of the authorized server: number of recent messages, "
        "of messages by bots and by people, and of distinct human authors. Message text is included only where the "
        "task's data permissions allow it.",
        {"channel_id": _SNOWFLAKE, "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        ("channel_id",), _VIEW + ("READ_MESSAGE_HISTORY",), idempotent=True),
    "discord.list_invites": _Tool(
        "external_read", "List the invites of the authorized server with their use counts.", {},
        requires=("MANAGE_GUILD",), idempotent=True),
    "discord.create_invite": _Tool(
        "external_write", "Create an invite link to a channel of the authorized server.",
        {"channel_id": _SNOWFLAKE, "max_age_s": {"type": "integer", "minimum": 0, "maximum": 604800},
         "max_uses": {"type": "integer", "minimum": 0, "maximum": 100}, "unique": {"type": "boolean"}},
        ("channel_id",), _VIEW + ("CREATE_INSTANT_INVITE",)),
    "discord.list_events": _Tool(
        "external_read", "List the scheduled events of the authorized server with the number of interested users.",
        {}, idempotent=True),
    "discord.create_event": _Tool(
        "external_write", "Create a scheduled event in the authorized server, at an external location or in one of "
        "its voice channels.",
        {"name": {"type": "string", "minLength": 1, "maxLength": 100},
         "start_time": {"type": "string", "minLength": 10}, "end_time": {"type": "string", "minLength": 10},
         "description": {"type": "string", "maxLength": 1000}, "location": {"type": "string", "maxLength": 100},
         "channel_id": _SNOWFLAKE},
        ("name", "start_time"), ("CREATE_EVENTS",)),
    # ---- identity, permissions, settings ---------------------------------------------------------------------
    "discord.get_identity": _Tool(
        "external_read", "Read which bot account and application the credential belongs to, which privileged "
        "gateway intents the application has, whether the bot is a member of the authorized server, and the number "
        "of other servers it is in.", {}, idempotent=True),
    "discord.get_permissions": _Tool(
        "external_read", "Read the permissions the bot holds in the authorized server, or in one of its channels "
        "with the channel's overwrites applied, which permissions of the operator's scope it lacks and which it "
        "holds beyond that scope.", {"channel_id": _SNOWFLAKE}, idempotent=True),
    "discord.get_guild_settings": _Tool(
        "external_read", "Read the settings of the authorized server: name, description, features, verification "
        "level, notification default, content filter, system, rules and updates channels, locale and approximate "
        "counts.", {}, idempotent=True),
    "discord.modify_guild": _Tool(
        "external_write", "Change settings of the authorized server: name, description, verification level, "
        "notification default, content filter, system, rules and updates channels, locale.",
        {"name": {"type": "string", "minLength": 2, "maxLength": 100},
         "description": {"type": ["string", "null"], "maxLength": 1000},
         "verification_level": {"type": "integer", "minimum": 0, "maximum": 4},
         "default_message_notifications": {"type": "integer", "minimum": 0, "maximum": 1},
         "explicit_content_filter": {"type": "integer", "minimum": 0, "maximum": 2},
         "system_channel_id": _SNOWFLAKE_OR_NULL, "rules_channel_id": _SNOWFLAKE_OR_NULL,
         "public_updates_channel_id": _SNOWFLAKE_OR_NULL,
         "preferred_locale": {"type": "string", "minLength": 2, "maxLength": 10}},
        requires=("MANAGE_GUILD",), idempotent=True),
    # ---- channels -----------------------------------------------------------------------------------------------
    "discord.modify_channel": _Tool(
        "external_write", "Change a channel of the authorized server: name, topic, position, parent category, age "
        "restriction, slow mode.",
        {"channel_id": _SNOWFLAKE, "name": {"type": "string", "minLength": 1, "maxLength": 100},
         "topic": {"type": ["string", "null"], "maxLength": 1024}, "position": {"type": "integer", "minimum": 0},
         "parent_id": _SNOWFLAKE_OR_NULL, "nsfw": {"type": "boolean"},
         "rate_limit_per_user": {"type": "integer", "minimum": 0, "maximum": 21600}},
        ("channel_id",), _VIEW + ("MANAGE_CHANNELS",), idempotent=True),
    "discord.delete_channel": _Tool(
        "external_write", "Delete one channel of the authorized server. A deleted channel cannot be restored.",
        {"channel_id": _SNOWFLAKE}, ("channel_id",), _VIEW + ("MANAGE_CHANNELS",)),
    "discord.set_channel_permissions": _Tool(
        "external_write", "Set what one role, or everyone, is allowed and denied in one channel of the authorized "
        "server.",
        {"channel_id": _SNOWFLAKE, "role_id": {"type": "string", "pattern": r"^(everyone|\d{5,25})$"},
         "allow": _PERMISSIONS, "deny": _PERMISSIONS},
        ("channel_id", "role_id"), _VIEW + ("MANAGE_ROLES",), idempotent=True),
    # ---- roles (the role itself; no tool gives a role to anyone) --------------------------------------------------
    "discord.list_roles": _Tool(
        "external_read", "List the roles of the authorized server with the permissions each holds.", {},
        idempotent=True),
    "discord.create_role": _Tool(
        "external_write", "Create a role in the authorized server. The role is not given to anyone.",
        dict(_ROLE_FIELDS), ("name",), ("MANAGE_ROLES",)),
    "discord.modify_role": _Tool(
        "external_write", "Change the name, permissions, colour or display of a role of the authorized server.",
        dict(_ROLE_FIELDS, role_id=_SNOWFLAKE), ("role_id",), ("MANAGE_ROLES",), idempotent=True),
    "discord.delete_role": _Tool(
        "external_write", "Delete a role of the authorized server.", {"role_id": _SNOWFLAKE}, ("role_id",),
        ("MANAGE_ROLES",)),
    # ---- messages and invites -----------------------------------------------------------------------------------
    "discord.delete_message": _Tool(
        "external_write", "Delete one message in a channel of the authorized server.", dict(_MESSAGE),
        ("channel_id", "message_id"), _VIEW + ("READ_MESSAGE_HISTORY",)),
    "discord.pin_message": _Tool(
        "external_write", "Pin a message in its channel.", dict(_MESSAGE), ("channel_id", "message_id"),
        _VIEW + ("PIN_MESSAGES",), idempotent=True),
    "discord.unpin_message": _Tool(
        "external_write", "Remove the pin of a message.", dict(_MESSAGE), ("channel_id", "message_id"),
        _VIEW + ("PIN_MESSAGES",), idempotent=True),
    "discord.delete_invite": _Tool(
        "external_write", "Delete an invite of the authorized server.",
        {"code": {"type": "string", "pattern": r"^[A-Za-z0-9-]{2,64}$"}}, ("code",), ("MANAGE_CHANNELS",)),
    # ---- events and audit ---------------------------------------------------------------------------------------
    "discord.get_member_events": _Tool(
        "external_read", "Read the membership changes of the authorized server since the previous reading: numbers "
        "of people, bots and declared test accounts that joined or left, join times of new people, current counts "
        "and the change of each invite's use count.", {}, idempotent=True),
    "discord.listen_events": _Tool(
        "external_read", "Receive the event stream of the authorized server for a fixed number of seconds and "
        "return what happened in that window: members joining, leaving or changing, invites, messages (lengths; "
        "text only where the task's data permissions allow it), channels, scheduled events.",
        {"seconds": {"type": "integer", "minimum": 1, "maximum": 60},
         "max_events": {"type": "integer", "minimum": 1, "maximum": 500}}, ("seconds",)),
    "discord.get_audit_summary": _Tool(
        "external_read", "Read how many recent audit-log entries of the authorized server there are per kind of "
        "action and when the newest was made.", {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        requires=("VIEW_AUDIT_LOG",), idempotent=True),
}
_ARTIFACT_KEYS = ("message_id", "channel_id", "invite_code", "event_id", "role_id")
_HOW_TO = {
    "scope": "add the permission(s) to resources.discord.scope in the task file (the bot's role must hold them too)",
    "permission": "the server owner grants the permission(s) to the bot's role: Server Settings > Roles",
    "intent": "the application owner enables the intent on the Bot page of the application in the Discord "
              "Developer Portal (Privileged Gateway Intents)",
    "guild": "the server owner adds the application to the server with the invite link of the application",
    "token": "the operator stores the bot token in the run's vault: python -m cam_live.harness.credentials set "
             "discord_bot_token --run-dir <run>",
    "guild_id": "the operator sets resources.discord.guild_id in the task file to the server the owner authorized",
    "gateway_secret": "the operator binds the token secret to the gateway host as well (credentials: --dest "
                      "discord.com --dest gateway.discord.gg)",
}


class _Refusal(Exception):
    """The adapter stops before (further) requests: ``status`` and the evidence of the reason."""

    def __init__(self, detail: str, *, denied: bool = False, evidence: Optional[FailureEvidence] = None):
        super().__init__(detail)
        self.status = ResultStatus.DENIED if denied else ResultStatus.FAILED
        self.evidence = evidence or FailureEvidence(exception=PermissionError(detail), detail=detail,
                                                    modality=Modality.API)


class _Stop(Exception):
    """An authorization verdict reached in the middle of an operation: its result is returned as it is."""

    def __init__(self, result: ActionResult):
        super().__init__("stopped by the authorization layer")
        self.result = result


def _not_in_guild(e: DiscordAPIError) -> bool:
    return (e.status == 404 and e.code in (10004, 10007)) or (e.status == 403 and e.code == 50001)


class DiscordExecutor:
    """``ModalityExecutor`` (modality ``api``) exposing the Discord capabilities of one authorized guild."""

    modality = Modality.API

    def __init__(self, base_url: str = LIVE_BASE_URL, *, guild_id: str = "", token_secret: str = "discord_bot_token",
                 secrets: Any = None, credential_probe: Optional[Callable[[str], bool]] = None,
                 test_account_ids: Sequence[str] = (), data_permissions: Optional[Mapping[str, bool]] = None,
                 id_salt: str = "", transport: Optional[httpx.BaseTransport] = None,
                 sleep: Optional[Callable[[float], None]] = None, retry: Optional[RetryPolicy] = None,
                 user_agent: str = "DiscordBot (https://localhost, 1.0)", scope: Optional[Sequence[str]] = None,
                 state_dir: str = "", permission_ttl_s: float = 300.0, identify_reserve: int = 50,
                 gateway_connect: Optional[Callable[..., Any]] = None):
        """``secrets`` is a broker handle used when ``ExecContext.secrets`` is absent. ``credential_probe(name)``
        says whether the named secret exists without releasing it (needed to report availability when the broker
        releases secrets only to a granted, executing action). ``scope`` is the list of Discord permission names
        the operator lets CAM use (default: :data:`discord_perms.DEFAULT_SCOPE`). ``state_dir`` holds the
        pseudonymous member snapshot of ``discord.get_member_events``."""
        self.base_url, self.guild_id, self.token_secret = base_url.rstrip("/"), str(guild_id or ""), token_secret
        self.environment = environment_of(self.base_url)
        self._secrets, self._probe = secrets, credential_probe
        self.test_account_ids = tuple(str(t) for t in test_account_ids)
        self.data_permissions = dict(data_permissions or {})
        self.id_salt, self._transport, self._sleep, self.user_agent = id_salt, transport, sleep, user_agent
        self.retry = retry or RetryPolicy()
        self.scope = normalise_scope(scope)
        self.state_dir, self.permission_ttl_s = str(state_dir or ""), float(permission_ttl_s)
        self.identify_reserve, self._gateway_connect = int(identify_reserve), gateway_connect
        self._client: Optional[DiscordClient] = None
        self._token_ctx: Tuple[Any, Any, Any] = (secrets, None, None)      # broker, granted action, its decision
        self._channels: Dict[str, Dict[str, Any]] = {}
        self._bot_id = ""
        self._perm_state: Optional[Dict[str, Any]] = None      # roles and the bot's own roles, with the time read
        self._app_state: Optional[Dict[str, Any]] = None       # application intents, with the time read
        self._memory_snapshot: Optional[Dict[str, Any]] = None
        #: what executed actions learned about the bot's standing (for :meth:`missing_capabilities`)
        self.facts: Dict[str, Any] = {"in_guild": None, "permissions": None, "intents": None,
                                      "members_list": None, "gateway_secret": None}
        gid = {"type": "string", "const": self.guild_id} if self.guild_id else {"type": "string"}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        for name, tool in TOOLS.items():                   # every tool names the server it acts on, explicitly
            self.schemas[name] = {"type": "object", "additionalProperties": False,
                                  "properties": dict(tool.properties, guild_id=gid),
                                  "required": list(tool.required) + (["guild_id"] if self.guild_id else [])}

    @classmethod
    def from_task(cls, task: TaskSpec, *, base_url: str = "", secrets: Any = None, **kw: Any) -> "DiscordExecutor":
        """Configuration from ``TaskSpec.resources["discord"]`` (``guild_id``, ``test_account_ids``, ``token_secret``,
        ``base_url``, ``scope``, ``state_dir``) and ``TaskSpec.data_permissions``. The flat keys of the task files
        (``resources.discord_guild_id``, ``resources.discord_bot_credential_name``) are read when the nested ones
        are absent."""
        res = task.resources or {}
        r = dict(res.get("discord") or {})
        kw.setdefault("scope", r.get("scope"))
        kw.setdefault("state_dir", r.get("state_dir") or "")
        guild_id = r.get("guild_id") or res.get("discord_guild_id") or ""
        secret = r.get("token_secret") or res.get("discord_bot_credential_name") or "discord_bot_token"
        if not isinstance(guild_id, (str, int)) or not str(guild_id).isdigit():
            guild_id = ""                                  # an unset item of a task file ("REQUIRED") is no guild
        return cls(base_url or r.get("base_url") or LIVE_BASE_URL, guild_id=str(guild_id),
                   token_secret=str(secret), secrets=secrets,
                   test_account_ids=r.get("test_account_ids") or (), data_permissions=task.data_permissions, **kw)

    # ---- availability --------------------------------------------------------------------------------------
    def _no_token(self) -> str:
        return f"blocked: no Discord bot token provided (secret {self.token_secret!r})"

    def availability(self, secrets: Any = None) -> Tuple[bool, str]:
        """Whether the tools can be used at all, and the exact reason if not; never contacts anything. With a
        ``credential_probe`` the token is not released to answer. ``secrets`` is the broker of an execution in
        progress: a grant-bound broker is then asked only at the moment of the request."""
        if not self.guild_id:
            return False, "blocked: no owner-authorized Discord guild is configured"
        broker = secrets if secrets is not None else self._secrets
        if self._probe is not None:
            present = bool(self._probe(self.token_secret))
        elif secrets is not None and grant_bound(broker):
            present = True
        else:
            present = resolve_secret(broker, self.token_secret) is not None
        return (True, "") if present else (False, self._no_token())

    def _token(self) -> Optional[str]:
        broker, action, decision = self._token_ctx
        return resolve_secret(broker, self.token_secret, action=action, decision=decision, destination=self.base_url)

    def client(self, clock: Optional[Clock] = None) -> DiscordClient:
        """The REST client. Its token is fetched from the broker at the moment of each request."""
        if self._client is None:
            self._client = DiscordClient(self.base_url, self._token, clock=clock, sleep=self._sleep,
                                         transport=self._transport, id_salt=self.id_salt, user_agent=self.user_agent)
        return self._client

    # ---- the operator's scope ------------------------------------------------------------------------------
    @staticmethod
    def route_requirements(name: str, arguments: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
        """Permissions the route needs, per the API documentation (an event held in a voice channel also needs
        ``VIEW_CHANNEL`` and ``CONNECT`` there)."""
        need = list(TOOLS[name].requires)
        if name == "discord.create_event" and (arguments or {}).get("channel_id"):
            need += ["VIEW_CHANNEL", "CONNECT"]
        return tuple(dict.fromkeys(need))

    @staticmethod
    def handed_out(name: str, arguments: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
        """Permissions the arguments name: a bot can put into an overwrite or a role only what it holds itself."""
        a = arguments or {}
        names: List[str] = []
        if name == "discord.set_channel_permissions":
            names = list(a.get("allow") or []) + list(a.get("deny") or [])
        if name in ("discord.create_role", "discord.modify_role"):
            names = list(a.get("permissions") or [])
        return tuple(dict.fromkeys(n for n in names if n in PERMISSION_BITS))

    def requirements(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
        """Every Discord permission an operation needs the bot to hold."""
        return tuple(dict.fromkeys(self.route_requirements(name, arguments) + self.handed_out(name, arguments)))

    def outside_scope(self, names: Iterable[str]) -> List[str]:
        return [n for n in names if n not in self.scope]

    @staticmethod
    def _scope_reason(names: Sequence[str]) -> str:
        return f"outside the operator's permission scope: needs {', '.join(names)}"

    # ---- ModalityExecutor ----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        ok, why = self.availability()
        out = []
        for name, schema in self.schemas.items():
            tool = TOOLS[name]
            outside = self.outside_scope(tool.requires)
            available, reason = ok, why
            if ok and outside:
                available, reason = False, self._scope_reason(outside)
            out.append(ToolDescriptor(name=name, modality=Modality.API, description=tool.text, arg_schema=schema,
                                      level=Level.PRIMITIVE, side_effects=tool.effect,
                                      auth_scope="+".join(tool.requires) or "discord.read",
                                      est_cost=Cost(seconds=0.5, primitive_ops=1), available=available,
                                      unavailable_reason=reason))
        return out

    def validate(self, action: ActionSpec) -> List[str]:
        schema = self.schemas.get(action.operation)
        if schema is None:
            return [f"unknown operation {action.operation!r} for the Discord executor"]
        errs = schema_errors(schema, action.arguments)
        if errs:
            return errs
        a, name = action.arguments, action.operation
        if name == "discord.create_event" and not a.get("channel_id"):
            if not a.get("location") or not a.get("end_time"):
                errs.append("arguments: an event outside a channel needs 'location' and 'end_time'")
        fields = {"discord.modify_guild": GUILD_SETTING_FIELDS, "discord.modify_channel": CHANNEL_SETTING_FIELDS,
                  "discord.modify_role": ROLE_SETTING_FIELDS}.get(name)
        if fields is not None and not any(k in a for k in fields):
            errs.append(f"arguments: nothing to change; give at least one of {', '.join(fields)}")
        if name == "discord.modify_guild" and "name" in a and not 2 <= len(a["name"].strip()) <= 100:
            errs.append("arguments.name: a server name has 2 to 100 characters, not counting surrounding spaces")
        if name in ("discord.modify_role", "discord.delete_role") and self.guild_id and a.get("role_id") == self.guild_id:
            errs.append("arguments.role_id: the @everyone role is never modified or deleted")
        if name == "discord.set_channel_permissions":
            allow, deny = list(a.get("allow") or []), list(a.get("deny") or [])
            both = sorted(set(allow) & set(deny))
            if both:
                errs.append(f"arguments: {', '.join(both)} cannot be allowed and denied at once")
            errs += self._grant_errors("arguments.allow", allow, granting=True)
            errs += self._grant_errors("arguments.deny", deny, granting=False)
        if name in ("discord.create_role", "discord.modify_role"):
            errs += self._grant_errors("arguments.permissions", list(a.get("permissions") or []), granting=True)
        return errs

    def _grant_errors(self, where: str, names: Sequence[str], *, granting: bool) -> List[str]:
        errs = []
        outside = sorted(set(self.outside_scope(names)))
        if outside:
            errs.append(f"{where}: {', '.join(outside)} is outside the operator's permission scope")
        never = sorted(set(names) & set(NEVER_GRANTED)) if granting else []
        if never:
            errs.append(f"{where}: {', '.join(never)} is never handed to a role by this adapter")
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
        broker = ctx.secrets if ctx.secrets is not None else self._secrets
        ok, why = self.availability(ctx.secrets)
        name = action.operation
        if ok:                                        # the operator's scope, whatever the bot's role would allow
            outside = self.outside_scope(self.requirements(name, action.arguments))
            if outside:
                ok, why = False, self._scope_reason(outside)
        if not ok:                                    # no credential, no authorized guild, out of scope: nothing is sent
            rep = classify(FailureEvidence(exception=PermissionError(why), detail=why, modality=Modality.API))
            return failed_result(action, rep, clock, started, status=ResultStatus.DENIED)
        self._token_ctx = (broker, granted_action(ctx, action), getattr(ctx, "authorization", None))
        client = self.client(clock)
        policy = self.retry if TOOLS[name].idempotent else RetryPolicy({FailureClass.RESOURCE_LIMIT:
                                                                         RetryPolicy.DEFAULT_RULES[FailureClass.RESOURCE_LIMIT]})
        return run_with_retries(lambda n: self._attempt(action, ctx, client, started), policy, clock,
                                sleep=self._sleep, deadline=ctx.deadline, cancel=ctx.cancel)

    # ---- permissions of the bot ----------------------------------------------------------------------------
    def _permission_state(self, client: DiscordClient, clock: Clock, refresh: bool = False) -> Dict[str, Any]:
        """Roles of the guild and the bot's own roles: read once, kept for ``permission_ttl_s``."""
        st = self._perm_state
        if st is not None and not refresh and clock.now() - st["ts"] < self.permission_ttl_s:
            return st
        g = self.guild_id
        try:
            if not self._bot_id:
                self._bot_id = client.me()["id"]
            roles = client.roles(g)
            own = client.own_member(g, self._bot_id)
        except DiscordAPIError as e:
            if _not_in_guild(e):
                self.facts["in_guild"] = False
                raise _Refusal("the bot is not a member of the authorized server; the server owner must add it") from None
            raise
        raw = 0
        by_id = {r["id"]: int(r["permission_bits"]) for r in roles}
        for rid in [g] + own["roles"]:
            raw |= by_id.get(rid, 0)
        until = own.get("communication_disabled_until")
        timed_out = False
        if until:
            try:
                timed_out = _dt.datetime.fromisoformat(str(until).replace("Z", "+00:00")).timestamp() > clock.now()
            except ValueError:
                timed_out = False
        base = base_permissions(g, [{"id": r["id"], "permissions": r["permission_bits"]} for r in roles], own["roles"])
        st = {"ts": clock.now(), "roles": roles, "own_roles": own["roles"], "raw": raw, "base": base,
              "timed_out": timed_out, "highest": max([r["position"] for r in roles if r["id"] in own["roles"]] or [0])}
        self._perm_state = st
        self.facts.update(in_guild=True, permissions=names_of(timed_out_permissions(base) if timed_out else base))
        return st

    def _effective(self, client: DiscordClient, clock: Clock, channel_id: str = "", refresh: bool = False) -> int:
        st = self._permission_state(client, clock, refresh)
        perms = st["base"]
        if channel_id:
            seen = client.overwrites.get(channel_id)
            if seen is None or refresh or clock.now() - seen[0] >= self.permission_ttl_s:
                client.channel(channel_id)
                seen = client.overwrites.get(channel_id)
            perms = channel_permissions(perms, seen[1] if seen else [], self.guild_id, st["own_roles"], self._bot_id)
        return timed_out_permissions(perms) if st["timed_out"] else perms

    def _require(self, client: DiscordClient, clock: Clock, names: Iterable[str], channel_id: str = "") -> None:
        """Stop unless every permission is inside the operator's scope and held by the bot (in the channel, when
        one is given). Nothing that needs the missing permission is sent."""
        names = list(dict.fromkeys(names))
        if not names:
            return
        outside = self.outside_scope(names)
        if outside:
            raise _Refusal(self._scope_reason(outside), denied=True)
        have = self._effective(client, clock, channel_id)
        missing = [n for n in names if not have >> PERMISSION_BITS[n] & 1]
        if missing:
            raise _Refusal(f"the bot lacks {', '.join(missing)} in this server; the server owner must grant it")

    def _application(self, client: DiscordClient, clock: Clock) -> Dict[str, Any]:
        st = self._app_state
        if st is None or clock.now() - st["ts"] >= self.permission_ttl_s:
            st = dict(client.application(), ts=clock.now())
            self._app_state = st
            self.facts["intents"] = dict(st["intents"])
        return st

    def missing_capabilities(self) -> List[Dict[str, str]]:
        """What cannot be used and what would make it usable, for the runtime's operator-request file. Built from
        the configuration and from what executed actions learned (``facts``); contacts nothing. Each entry is
        ``{capability, reason, how_to_grant}``; ``capability`` is a tool name or ``discord`` for all of them."""
        out: List[Dict[str, str]] = []
        ok, why = self.availability()
        if not ok:
            key = "guild_id" if not self.guild_id else "token"
            return [{"capability": "discord", "reason": why, "how_to_grant": _HOW_TO[key]}]
        if self.facts["in_guild"] is False:
            return [{"capability": "discord", "reason": "the bot is not a member of the authorized server",
                     "how_to_grant": _HOW_TO["guild"]}]
        held = self.facts["permissions"]
        intents = self.facts["intents"] or {}
        for name, tool in TOOLS.items():
            outside = self.outside_scope(tool.requires)
            if outside:
                out.append({"capability": name, "reason": self._scope_reason(outside), "how_to_grant": _HOW_TO["scope"]})
                continue
            lacking = [n for n in tool.requires if held is not None and n not in held]
            if lacking:
                out.append({"capability": name, "reason": f"the bot lacks {', '.join(lacking)} in this server; the "
                                                          f"server owner must grant it",
                            "how_to_grant": _HOW_TO["permission"]})
        if intents.get("GUILD_MEMBERS") is False or self.facts["members_list"] is False:
            for name in ("discord.listen_events", "discord.get_member_events", "discord.get_guild_stats"):
                out.append({"capability": name, "reason": "the application lacks the GUILD_MEMBERS privileged "
                                                          "gateway intent (exact member lists and member events)",
                            "how_to_grant": _HOW_TO["intent"]})
        if intents.get("MESSAGE_CONTENT") is False and self.data_permissions.get(CONTENT_PERMISSION):
            out.append({"capability": "discord.listen_events",
                        "reason": "the application lacks the MESSAGE_CONTENT privileged gateway intent",
                        "how_to_grant": _HOW_TO["intent"]})
        if self.facts["gateway_secret"] is False:
            out.append({"capability": "discord.listen_events",
                        "reason": f"the secret {self.token_secret!r} is not released for the gateway destination",
                        "how_to_grant": _HOW_TO["gateway_secret"]})
        return out

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
        content_ok = bool(self.data_permissions.get(CONTENT_PERMISSION, False))
        check_args = {k: v for k, v in a.items()}
        if "content" in a:
            check_args["content_sha256"] = hashlib.sha256(a["content"].encode()).hexdigest()
        keep = name in ("discord.get_channel_activity", "discord.listen_events") and content_ok
        if name == "discord.get_guild_stats":
            check_args.update(reads="guild counts and member list", aggregate_only=True)
        if name == "discord.get_channel_activity":
            check_args.update(reads="message history", aggregate_only=not keep)
        if name == "discord.get_member_events":
            check_args.update(reads="member list and invite use counts", aggregate_only=True)
        if name == "discord.get_audit_summary":
            check_args.update(reads="audit log", aggregate_only=True)
        if name == "discord.listen_events":
            check_args.update(reads="gateway events", intents=intents_needed(keep), aggregate_only=not keep)
        refused = self._checked(action, ctx, started, ops, name, check_args)
        if refused is not None:
            return refused
        if keep:        # message text will leave the adapter: that is a different data class and is judged as such
            # Text in an observation is kept: it enters the experience stream (recorded, replayed) and is learned
            # from. So retaining it is declared to the authorization layer as reading, storing AND training on
            # message content (data classes message_content_read / _store / _train), not as reading only: the
            # task's data permissions and the platform rule decide all three (spec 12.5, 12.6; IH-15).
            read_args: Dict[str, Any] = {"content_retained": True, "store": True, "train": True}
            if "channel_id" in a:
                read_args["channel_id"] = a["channel_id"]
            refused = self._checked(action, ctx, started, ops, "discord.read_messages", read_args,
                                    note="message text retained under the task's data permission")
            if refused is not None:
                return refused
        sent_before = client.requests_sent
        try:
            for key in ("channel_id", "parent_id", "system_channel_id", "rules_channel_id", "public_updates_channel_id"):
                cid = a.get(key)
                if not cid:
                    continue
                if cid not in self._channels:
                    refused = self._checked(action, ctx, started, ops, "discord.list_channels", {},
                                            note="is the channel part of the authorized server")
                    if refused is not None:
                        return refused
                if self._in_guild(client, cid) is None:
                    raise _Refusal("the channel is not a channel of the authorized server; nothing was read or sent",
                                   denied=True)
            if name != "discord.delete_invite":            # its channel is known only once the invite was read
                self._require(client, clock, self.route_requirements(name, a), a.get("channel_id", ""))
                self._require(client, clock, self.handed_out(name, a))
            text, quantities, data = self._call(name, a, client, action, ctx, ops, started)
        except _Refusal as r:
            rep = classify(r.evidence)
            now = clock.now()
            for op in ops:
                if not op.ended:
                    op.status, op.failure_class, op.ended = r.status, rep.failure_class, now
            return failed_result(action, rep, clock, started, ops=ops, status=r.status)
        except _Stop as s:
            return s.result
        except (DiscordAPIError, DiscordUnavailable, DiscordMissingPermission, GatewayError, httpx.HTTPError) as e:
            if isinstance(e, DiscordMissingPermission) or (isinstance(e, DiscordAPIError) and e.status == 403):
                self._perm_state = None            # what was cached about the bot's permissions is out of date
                client.overwrites.clear()
            rep = classify(evidence_of(e))
            if isinstance(e, DiscordAPIError) and e.status == 404 and e.code == 10004:      # Unknown Guild
                self.facts["in_guild"] = False
                why = ("the bot is not a member of the authorized server; the server owner must add it "
                       f"(Discord 404 code 10004: {e.message})")
                rep = classify(FailureEvidence(exception=PermissionError(why), detail=why, modality=Modality.API))
            for op in ops:
                op.ended = op.ended or clock.now()
            ops[-1].status, ops[-1].failure_class = ResultStatus.FAILED, rep.failure_class
            denied = isinstance(e, DiscordUnavailable) or (isinstance(e, GatewayError) and e.close_code is None
                                                           and e.kind == "permission")
            return failed_result(action, rep, clock, started, ops=ops, status=ResultStatus.DENIED if denied else "")
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
                            artifacts={k: str(v) for k, v in data.items() if k in _ARTIFACT_KEYS})

    # ---- the operations ------------------------------------------------------------------------------------
    def _call(self, name: str, a: Dict[str, Any], client: DiscordClient, action: ActionSpec,
              ctx: Optional[ExecContext] = None, ops: Optional[List[PrimitiveOp]] = None,
              started: float = 0.0) -> Tuple[str, List[Quantity], Dict[str, Any]]:
        g, label = self.guild_id, f"[{self.environment}] "
        count = OutcomeKind.COUNT
        clock = ctx.clock if ctx is not None else client.clock
        if name == "discord.get_guild_stats":
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
                self._perm_state = None            # any 403: what was cached about the bot's standing is dropped
                self.facts["members_list"] = False
                data["members_unavailable"] = f"member list not available ({e.message}); only approximate counts"
            else:
                self.facts["members_list"] = True
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
        if name == "discord.get_channel_activity":
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
        if name == "discord.create_event":
            e = client.create_scheduled_event(g, a["name"], a["start_time"], scheduled_end_time=a.get("end_time"),
                                              description=a.get("description"), location=a.get("location"),
                                              channel_id=a.get("channel_id"))
            return f"{label}scheduled event created: {e['name']} ({e['id']})", [], {"event": e, "event_id": e["id"]}
        handler = getattr(self, "_op_" + name.split(".", 1)[1])
        return handler(a, client, clock, label, action, ctx, ops if ops is not None else [], started)

    # identity, permissions, settings
    def _op_get_identity(self, a, client, clock, label, action, ctx, ops, started):
        me, app = client.me(), client.application()
        inside, others = client.guild_membership(self.guild_id)
        self._bot_id = me["id"]
        self._app_state = dict(app, ts=clock.now())
        self.facts.update(in_guild=inside, intents=dict(app["intents"]))
        warnings = []
        if others:
            warnings.append(f"the bot is a member of {others} other server(s); this adapter acts only in the "
                            f"authorized one")
        if app["bot_public"]:
            warnings.append("the application is public: anyone can add it to a server")
        q = [Quantity("discord.identity.in_guild", 1.0 if inside else 0.0, kind=OutcomeKind.BINARY),
             Quantity("discord.identity.other_guilds", float(others), kind=OutcomeKind.COUNT)]
        q += [Quantity(f"discord.identity.intent.{k.lower()}", 1.0 if v else 0.0, kind=OutcomeKind.BINARY)
              for k, v in app["intents"].items()]
        granted = [k for k, v in app["intents"].items() if v]
        text = (f"{label}bot {me['name']} ({me['id']}), application {app['name']} "
                f"({'public' if app['bot_public'] else 'not public'}); privileged intents granted: "
                f"{', '.join(granted) or 'none'}; member of the authorized server: {'yes' if inside else 'no'}; "
                f"other servers: {others}")
        return text, q, {"bot": me, "application": {"name": app["name"], "bot_public": app["bot_public"],
                                                    "intents": app["intents"]},
                         "in_guild": inside, "other_guilds": others, "warnings": warnings}

    def _op_get_permissions(self, a, client, clock, label, action, ctx, ops, started):
        cid = a.get("channel_id", "")
        perms = self._effective(client, clock, cid, refresh=True)
        st = self._perm_state or {}
        held = names_of(perms)
        admin = bool(st.get("raw", 0) & ADMINISTRATOR)
        missing = sorted(set(self.scope) - set(held))
        broader = sorted(set(held) - set(self.scope))
        q = [Quantity("discord.permissions.held", float(len(held)), kind=OutcomeKind.COUNT),
             Quantity("discord.permissions.missing_from_scope", float(len(missing)), kind=OutcomeKind.COUNT),
             Quantity("discord.permissions.broader_than_scope", float(len(broader)), kind=OutcomeKind.COUNT),
             Quantity("discord.permissions.administrator", 1.0 if admin else 0.0, kind=OutcomeKind.BINARY)]
        where = f"in channel {cid}" if cid else "in the server"
        text = (f"{label}the bot holds {len(held)} permissions {where}: {', '.join(held) or 'none'}; missing from "
                f"the operator's scope: {', '.join(missing) or 'none'}; held beyond the scope: "
                f"{', '.join(broader) or 'none'}; administrator: {'yes' if admin else 'no'}")
        data = {"permissions": held, "administrator": admin, "missing_from_scope": missing,
                "broader_than_scope": broader, "scope": sorted(self.scope), "timed_out": bool(st.get("timed_out"))}
        if cid:
            data["channel_id"] = cid
        return text, q, data

    @staticmethod
    def _settings_text(s: Dict[str, Any]) -> str:
        return (f"name {s['name']!r}, description {s['description']!r}, verification level {s['verification_level']}, "
                f"notification default {s['default_message_notifications']}, content filter "
                f"{s['explicit_content_filter']}, system channel {s['system_channel_id']}, rules channel "
                f"{s['rules_channel_id']}, updates channel {s['public_updates_channel_id']}, locale "
                f"{s['preferred_locale']}, features {', '.join(s['features']) or 'none'}")

    def _op_get_guild_settings(self, a, client, clock, label, action, ctx, ops, started):
        s = client.guild_settings(self.guild_id)
        q = [Quantity("discord.members.approximate", float(s["approximate_member_count"] or 0), kind=OutcomeKind.COUNT),
             Quantity("discord.presence.approximate", float(s["approximate_presence_count"] or 0), kind=OutcomeKind.COUNT),
             Quantity("discord.guild.verification_level", float(s["verification_level"] or 0))]
        return f"{label}server settings: {self._settings_text(s)}", q, {"settings": s}

    def _op_modify_guild(self, a, client, clock, label, action, ctx, ops, started):
        changed = [k for k in GUILD_SETTING_FIELDS if k in a]
        s = client.modify_guild(self.guild_id, a)
        return (f"{label}server settings changed ({', '.join(changed)}): {self._settings_text(s)}", [],
                {"settings": s, "changed": changed})

    # channels
    def _op_modify_channel(self, a, client, clock, label, action, ctx, ops, started):
        changed = [k for k in CHANNEL_SETTING_FIELDS if k in a]
        c = client.modify_channel(a["channel_id"], a)
        self._channels[c["id"]] = {k: c[k] for k in ("id", "name", "type", "guild_id", "parent_id", "position")}
        return (f"{label}channel {c['id']} changed ({', '.join(changed)}): name {c['name']!r}, topic {c.get('topic')!r}",
                [], {"channel": c, "channel_id": c["id"], "changed": changed})

    def _op_delete_channel(self, a, client, clock, label, action, ctx, ops, started):
        c = client.delete_channel(a["channel_id"])
        self._channels.pop(a["channel_id"], None)
        return (f"{label}channel {a['channel_id']} deleted", [],
                {"deleted_channel": c, "channel_id": a["channel_id"], "deleted": True})

    def _role_of_guild(self, client: DiscordClient, clock: Clock, role_id: str) -> Dict[str, Any]:
        """The role, read fresh; a role id that is not a role of the authorized server is refused."""
        st = self._permission_state(client, clock, refresh=True)
        role = next((r for r in st["roles"] if r["id"] == role_id), None)
        if role is None:
            why = "the role is not a role of the authorized server; nothing was changed"
            raise _Refusal(why, evidence=FailureEvidence(target_missing=True, detail=why, modality=Modality.API))
        return role

    def _op_set_channel_permissions(self, a, client, clock, label, action, ctx, ops, started):
        rid = self.guild_id if a["role_id"] == "everyone" else a["role_id"]
        role = self._role_of_guild(client, clock, rid)             # a role of this server: never a member id
        allow, deny = sorted(set(a.get("allow") or [])), sorted(set(a.get("deny") or []))
        out = client.set_role_overwrite(a["channel_id"], rid, allow, deny)
        who = "everyone" if role["everyone"] else f"role {role['name']} ({rid})"
        return (f"{label}channel {a['channel_id']}: {who} allowed {', '.join(allow) or 'nothing more'}, denied "
                f"{', '.join(deny) or 'nothing'}", [],
                {"overwrite": dict(out, everyone=role["everyone"]), "channel_id": a["channel_id"], "role_id": rid})

    # roles
    @staticmethod
    def _role_view(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: r[k] for k in ("id", "name", "permissions", "position", "managed", "mentionable", "hoist", "color",
                                  "everyone")}

    def _op_list_roles(self, a, client, clock, label, action, ctx, ops, started):
        roles = [self._role_view(r) for r in client.roles(self.guild_id)]
        return (label + "roles: " + ", ".join(f"{r['name']} ({r['id']}): {len(r['permissions'])} permissions"
                                              for r in roles),
                [Quantity("discord.roles.count", float(len(roles)), kind=OutcomeKind.COUNT)], {"roles": roles})

    def _op_create_role(self, a, client, clock, label, action, ctx, ops, started):
        r = self._role_view(client.create_role(self.guild_id, a["name"], a.get("permissions") or [],
                                               color=a.get("color", 0), hoist=a.get("hoist", False),
                                               mentionable=a.get("mentionable", False)))
        self._perm_state = None
        return (f"{label}role created: {r['name']} ({r['id']}) with {', '.join(r['permissions']) or 'no permissions'}; "
                f"it is not given to anyone", [], {"role": r, "role_id": r["id"]})

    def _editable_role(self, client: DiscordClient, clock: Clock, role_id: str) -> Dict[str, Any]:
        role = self._role_of_guild(client, clock, role_id)
        st = self._perm_state or {}
        if role["everyone"] or role["managed"]:
            raise _Refusal("the @everyone role and roles managed by an integration are never modified or deleted",
                           denied=True)
        if role["position"] > st.get("highest", 0):        # equal positions are ordered by id: left to the service
            raise _Refusal("the bot can only change roles below its own highest role; the server owner must move "
                           "the bot's role above this one")
        return role

    def _op_modify_role(self, a, client, clock, label, action, ctx, ops, started):
        self._editable_role(client, clock, a["role_id"])
        changed = [k for k in ROLE_SETTING_FIELDS if k in a]
        r = self._role_view(client.modify_role(self.guild_id, a["role_id"], a))
        self._perm_state = None
        return (f"{label}role {r['id']} changed ({', '.join(changed)}): {r['name']} with "
                f"{', '.join(r['permissions']) or 'no permissions'}", [], {"role": r, "role_id": r["id"], "changed": changed})

    def _op_delete_role(self, a, client, clock, label, action, ctx, ops, started):
        role = self._editable_role(client, clock, a["role_id"])
        client.delete_role(self.guild_id, a["role_id"])
        self._perm_state = None
        return (f"{label}role {role['name']} ({role['id']}) deleted", [],
                {"deleted_role": self._role_view(role), "role_id": role["id"], "deleted": True})

    # messages and invites
    def _op_delete_message(self, a, client, clock, label, action, ctx, ops, started):
        if not self._bot_id:
            self._bot_id = client.me()["id"]
        meta = client.message_meta(a["channel_id"], a["message_id"], self._bot_id)
        if not meta["own"]:        # another author's message: a different act, judged and permitted as such
            if ctx is not None:
                refused = self._checked(action, ctx, started, ops, "discord.delete_message",
                                        dict(a, own=False, author_is_bot=meta["author_is_bot"]),
                                        note="the message was posted by another author")
                if refused is not None:
                    raise _Stop(refused)
            self._require(client, clock, ["MANAGE_MESSAGES"], a["channel_id"])
        client.delete_message(a["channel_id"], a["message_id"])
        whose = "the bot's own message" if meta["own"] else "a message of another author"
        return (f"{label}message {a['message_id']} in channel {a['channel_id']} deleted ({whose})",
                [Quantity("discord.message.deleted_own", 1.0 if meta["own"] else 0.0, kind=OutcomeKind.BINARY)],
                {"message_id": a["message_id"], "channel_id": a["channel_id"], "own": meta["own"],
                 "author_is_bot": meta["author_is_bot"], "deleted": True})

    def _op_pin_message(self, a, client, clock, label, action, ctx, ops, started):
        client.pin_message(a["channel_id"], a["message_id"])
        return (f"{label}message {a['message_id']} pinned in channel {a['channel_id']}", [],
                {"message_id": a["message_id"], "channel_id": a["channel_id"], "pinned": True})

    def _op_unpin_message(self, a, client, clock, label, action, ctx, ops, started):
        client.unpin_message(a["channel_id"], a["message_id"])
        return (f"{label}pin of message {a['message_id']} removed in channel {a['channel_id']}", [],
                {"message_id": a["message_id"], "channel_id": a["channel_id"], "pinned": False})

    def _op_delete_invite(self, a, client, clock, label, action, ctx, ops, started):
        inv = client.invite(a["code"])
        if inv["guild_id"] != self.guild_id or self._in_guild(client, inv["channel_id"]) is None:
            raise _Refusal("the invite does not belong to the authorized server; nothing was deleted", denied=True)
        try:
            self._require(client, clock, ["MANAGE_CHANNELS"], inv["channel_id"])
        except _Refusal as first:                      # the documented alternative: MANAGE_GUILD in the server
            have = self._effective(client, clock)
            if first.status == ResultStatus.DENIED or "MANAGE_GUILD" not in self.scope \
                    or not have >> PERMISSION_BITS["MANAGE_GUILD"] & 1:
                raise
        client.delete_invite(a["code"])
        return (f"{label}invite {a['code']} to channel {inv['channel_id']} deleted", [],
                {"invite_code": a["code"], "channel_id": inv["channel_id"], "deleted": True})

    # membership changes
    def _snapshot_path(self, ctx: Optional[ExecContext]) -> str:
        root = self.state_dir or (os.path.join(ctx.run_dir, "discord_state") if ctx is not None and ctx.run_dir else "")
        if not root:
            return ""
        tag = hashlib.sha256(f"{self.environment}:{self.guild_id}".encode()).hexdigest()[:16]
        return os.path.join(root, f"member_snapshot_{tag}.json")

    def _read_snapshot(self, path: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """(previous snapshot or None, was an unreadable file found)."""
        if not path:
            return self._memory_snapshot, False
        try:
            with open(path, "r") as f:
                snap = json.load(f)
            if not isinstance(snap, dict) or not isinstance(snap.get("members"), list) or snap.get("version") != 1:
                raise ValueError("not a member snapshot")
            return snap, False
        except FileNotFoundError:
            return None, False
        except (OSError, ValueError):
            return None, True

    def _write_snapshot(self, path: str, snap: Dict[str, Any]) -> None:
        """Atomic, private write: temporary file (0600) in the same directory, fsync, rename."""
        self._memory_snapshot = snap
        if not path:
            return
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.write(fd, json.dumps(snap, sort_keys=True).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)

    def _op_get_member_events(self, a, client, clock, label, action, ctx, ops, started):
        g, count = self.guild_id, OutcomeKind.COUNT
        try:
            members = client.list_members(g, test_account_ids=self.test_account_ids)
        except DiscordAPIError as e:
            if e.status == 403:
                self.facts["members_list"] = False
            raise
        self.facts["members_list"] = True
        invites: Optional[Dict[str, int]] = None
        invites_note = ""
        try:
            self._require(client, clock, ["MANAGE_GUILD"])
            invites = {i["code"]: int(i["uses"] or 0) for i in client.list_invites(g)}
        except _Refusal as r:
            invites_note = f"invite use counts not read: {r}"
        except (DiscordAPIError, DiscordMissingPermission) as e:
            if isinstance(e, DiscordAPIError) and e.status != 403:
                raise
            self._perm_state = None
            invites, invites_note = None, f"invite use counts not read: {getattr(e, 'message', e)}"
        path = self._snapshot_path(ctx)
        prev, unreadable = self._read_snapshot(path)
        now = clock.now()
        counts = client.member_counts(members)
        snap = {"version": 1, "environment": self.environment,
                "guild_hash": hashlib.sha256(f"{self.id_salt}:{g}".encode()).hexdigest()[:32], "taken_at": now,
                "members": members, "invites": invites}
        self._write_snapshot(path, snap)
        q = [Quantity(f"discord.members.{k}", float(v), kind=count) for k, v in counts.items()]
        data: Dict[str, Any] = {"baseline": prev is None, "member_counts": counts, "persisted": bool(path),
                                "previous_snapshot_unreadable": unreadable}
        if invites_note:
            data["invites_unavailable"] = invites_note
        if prev is None:
            text = (f"{label}membership baseline established: {counts['humans']} people, {counts['bots']} bots, "
                    f"{counts['test_accounts']} declared test accounts; changes are reported from the next reading on")
            return text, q, data
        elapsed = max(1.0, now - float(prev.get("taken_at") or now))
        ch = client.member_changes(prev["members"], members)
        data.update(joined=ch["joined"], left=ch["left"], joined_at=ch["joined_at"], since_s=round(elapsed, 3))
        q += [Quantity("discord.members.joined", float(ch["joined"]["people"]), kind=count, exposure=elapsed),
              Quantity("discord.members.left", float(ch["left"]["people"]), kind=count, exposure=elapsed),
              Quantity("discord.members.joined_bots", float(ch["joined"]["bots"]), kind=count, exposure=elapsed),
              Quantity("discord.members.left_bots", float(ch["left"]["bots"]), kind=count, exposure=elapsed),
              Quantity("discord.members.joined_test_accounts", float(ch["joined"]["test_accounts"]), kind=count,
                       exposure=elapsed),
              Quantity("discord.members.left_test_accounts", float(ch["left"]["test_accounts"]), kind=count,
                       exposure=elapsed)]
        text = (f"{label}membership since the previous reading ({elapsed:.0f} s): joined {ch['joined']['people']} "
                f"people, {ch['joined']['bots']} bots, {ch['joined']['test_accounts']} declared test accounts; left "
                f"{ch['left']['people']} people, {ch['left']['bots']} bots, {ch['left']['test_accounts']} declared "
                f"test accounts; now {counts['humans']} people")
        before = prev.get("invites")
        if invites is not None and isinstance(before, dict):
            deltas = [{"code": c, "uses": u, "delta": u - int(before.get(c, 0)), "new": c not in before}
                      for c, u in sorted(invites.items())]
            data.update(invites=deltas, invites_gone=len(set(before) - set(invites)))
            total = sum(d["delta"] for d in deltas)
            q.append(Quantity("discord.invites.uses_delta", float(total), kind=count, exposure=elapsed))
            used = [f"{d['code']} +{d['delta']}" for d in deltas if d["delta"]]
            text += "; invite uses: " + (", ".join(used) or "no change")
        elif invites is not None:
            data["invites"] = [{"code": c, "uses": u, "delta": None, "new": True} for c, u in sorted(invites.items())]
        return text, q, data

    # gateway window
    def gateway_url_allowed(self, url: str) -> bool:
        """The token is sent to the Gateway: against ``discord.com`` only a ``wss`` URL under ``discord.gg`` is
        accepted, in staging only the host of the staging REST service."""
        u = urlsplit(url)
        host = (u.hostname or "").lower()
        if self.environment == "live":
            return u.scheme == "wss" and (host == "gateway.discord.gg" or host.endswith(".discord.gg"))
        return u.scheme in ("ws", "wss") and host == (urlsplit(self.base_url).hostname or "").lower()

    def _op_listen_events(self, a, client, clock, label, action, ctx, ops, started):
        keep = bool(self.data_permissions.get(CONTENT_PERMISSION, False))
        intents = intents_needed(keep)
        app = self._application(client, clock)
        lacking = [i for i in intents if i in PRIVILEGED_INTENTS and not app["intents"].get(i)]
        if lacking:        # an Identify with an intent the application does not have is closed with code 4014
            raise _Refusal(f"the application lacks the {', '.join(lacking)} privileged gateway intent; the "
                           f"application owner must enable it on the Bot page of the developer portal")
        gw = client.gateway_bot()
        url, limit = gw["url"], gw["session_start_limit"]
        if not self.gateway_url_allowed(url):
            why = "the gateway URL returned by the service is not a Discord gateway address; nothing was connected"
            raise _Refusal(why, denied=True)
        remaining = limit.get("remaining")
        if isinstance(remaining, (int, float)) and remaining <= self.identify_reserve:
            why = (f"only {int(remaining)} gateway session starts remain before the daily limit (reserve "
                   f"{self.identify_reserve}); no session was started")
            raise _Refusal(why, evidence=FailureEvidence(limit_exceeded=True, detail=why, modality=Modality.API,
                                                         retry_after_s=float(limit.get("reset_after") or 0) / 1000.0))
        seconds = float(a["seconds"])
        if ctx is not None and ctx.deadline:
            room = ctx.deadline - clock.now() - 2.0
            if room < 1.0:
                why = "the action's time limit leaves no room for a listening window"
                raise _Refusal(why, evidence=FailureEvidence(limit_exceeded=True, detail=why, modality=Modality.API))
            seconds = min(seconds, room)
        broker, granted, decision = self._token_ctx
        host = (urlsplit(url).hostname or "")

        def gateway_token() -> Optional[str]:
            token = resolve_secret(broker, self.token_secret, action=granted, decision=decision, destination=url)
            self.facts["gateway_secret"] = token is not None
            return token

        window = GatewayWindow(url, gateway_token, guild_id=self.guild_id, seconds=seconds, intents=intents_needed(False),
                               keep_content=keep, id_salt=self.id_salt, test_account_ids=self.test_account_ids,
                               max_events=a.get("max_events", 200), cancel=ctx.cancel if ctx is not None else None,
                               connect=self._gateway_connect)
        try:
            out = window.run()
        except GatewayError as e:
            if e.kind == "permission" and e.close_code is None:
                why = (f"blocked: the bot token (secret {self.token_secret!r}) was not released for the gateway "
                       f"destination {host}; the operator must bind the secret to it")
                raise _Refusal(why, denied=True) from None
            if e.close_code == 4014:
                self._app_state = None
                self.facts["intents"] = dict(self.facts["intents"] or {},
                                             **{i: False for i in intents if i in PRIVILEGED_INTENTS})
            raise
        q = [Quantity("discord.gateway.events", float(len(out["events"])), kind=OutcomeKind.COUNT, exposure=max(out["window_s"], 1e-3)),
             Quantity("discord.gateway.window_s", float(out["window_s"]), unit="s"),
             Quantity("discord.gateway.dropped_other_guild", float(out["dropped"]["other_guild"]), kind=OutcomeKind.COUNT),
             Quantity("discord.gateway.heartbeats_sent", float(out["heartbeats"]["sent"]), kind=OutcomeKind.COUNT)]
        q += [Quantity(f"discord.gateway.{k.lower()}", float(v), kind=OutcomeKind.COUNT, exposure=max(out["window_s"], 1e-3))
              for k, v in sorted(out["counts"].items())]
        if isinstance(remaining, (int, float)):
            q.append(Quantity("discord.gateway.session_starts_remaining", float(remaining) - 1.0, kind=OutcomeKind.COUNT))
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(out["counts"].items())) or "none"
        text = (f"{label}event window of {out['window_s']:.1f} s (ended by {out['ended_by']}): {len(out['events'])} "
                f"events ({kinds}); {out['dropped']['other_guild']} events of other servers dropped")
        text += "; message text retained" if keep else "; message text not retained (no data permission)"
        return text, q, {"window": out}

    def _op_get_audit_summary(self, a, client, clock, label, action, ctx, ops, started):
        s = client.audit_summary(self.guild_id, a.get("limit", 50))
        q = [Quantity("discord.audit.entries", float(s["entries"]), kind=OutcomeKind.COUNT)]
        q += [Quantity(f"discord.audit.{k.lower()}", float(v), kind=OutcomeKind.COUNT) for k, v in s["by_action"].items()]
        kinds = ", ".join(f"{k} {v}" for k, v in s["by_action"].items()) or "none"
        return (f"{label}audit log, {s['entries']} most recent entries: {kinds}; newest at {s['newest']}", q,
                {"audit": s})


__all__ = ["DiscordClient", "DiscordExecutor", "DiscordAPIError", "DiscordUnavailable", "DiscordMissingPermission",
           "LIVE_BASE_URL",
           "CONTENT_PERMISSION", "CHANNEL_TYPES", "AUDIT_REASON", "AUDIT_LOG_EVENTS", "TOOLS", "DEFAULT_SCOPE",
           "environment_of", "member_id_hash", "evidence_of", "snowflake_iso"]
