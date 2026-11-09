"""Discord connector: a bot user added (by a human admin) to servers whose admins agreed.

Credentials: ``bot_token`` (secret). Account fields: ``allowed_channel_ids`` (the only channels the
connector reads or posts in), ``application_id``, ``public_key`` (hex; for interaction signature
verification by the webhook receiver).

Interactions arrive by webhook (Interactions Endpoint URL, Ed25519-signed; verify with
``agentrs.harness.webhooks.verify_discord_signature``); channel messages are polled (the Gateway
websocket is not used).
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
    parse_time,
)
from agentrs.harness.connectors.base import ConnectorError, Subscription
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

_MASS_MENTION = re.compile(r"@everyone|@here|<@&\d+>")
EPHEMERAL = 64


class DiscordConnector(HarnessConnector):
    provider = "discord"
    operations = {
        "read_messages": OperationSpec(name="read_messages", tier=Tier.OBSERVE, mutates=False, params=["channel_id"],
                                       optional_params=["limit", "after", "before"], capability="read_messages",
                                       description="GET /channels/{id}/messages"),
        "send_message": OperationSpec(name="send_message", tier=Tier.EXTERNAL, mutates=True, params=["channel_id", "content"],
                                      optional_params=["reply_to_message_id"], content_param="content",
                                      capability="send_message", reversible=False,
                                      description="POST /channels/{id}/messages (no pings: allowed_mentions.parse = [])"),
        "respond_interaction": OperationSpec(name="respond_interaction", tier=Tier.EXTERNAL, mutates=True,
                                             params=["interaction_id", "interaction_token", "channel_id", "content"],
                                             optional_params=["ephemeral"], content_param="content",
                                             capability="respond_interaction", reversible=False,
                                             description="POST /interactions/{id}/{token}/callback (type 4)"),
    }
    outcome_profile = {"send_message": "forum_reply", "respond_interaction": "forum_reply"}
    text_ops = frozenset({"send_message"})

    def __init__(self, ctx):
        super().__init__(ctx)
        a = ctx.account or {}
        self.allowed_channels = {str(c) for c in a.get("allowed_channel_ids", [])}
        self.bot_id: Optional[str] = a.get("bot_user_id")

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bot {self._secret('bot_token')}",
                "User-Agent": "DiscordBot (https://github.com/agentrs, 0.1)"}

    def _channel(self, channel_id: str) -> str:
        cid = str(channel_id)
        if cid not in self.allowed_channels:
            raise ConnectorError(f"discord: channel {cid} is not configured for this bot", 403)
        return cid

    def _extra_policy(self, action: ActionRequest, entries: list[dict]) -> tuple[bool, str]:
        cid = str(action.params.get("channel_id", ""))
        if cid not in self.allowed_channels:
            return False, f"channel {cid or '?'} is not one of the configured channel ids"
        if _MASS_MENTION.search(str(action.params.get("content", ""))):
            return False, "mass mentions (@everyone, @here, role mentions) are refused"
        return True, ""

    async def validate_credentials(self) -> dict[str, Any]:
        me = await self._req("GET", "/users/@me", headers=self._h())
        if not me.get("bot"):
            raise ConnectorError("discord: the token does not belong to a bot user", 403)
        self.bot_id = me["id"]
        return {"ok": True, "id": me["id"], "username": me.get("username"), "bot": True,
                "allowed_channel_ids": sorted(self.allowed_channels)}

    async def _me(self) -> str:
        if not self.bot_id:
            await self.validate_credentials()
        return str(self.bot_id)

    async def op_read_messages(self, channel_id: str, limit: int = 50, after: Optional[str] = None,
                               before: Optional[str] = None) -> dict[str, Any]:
        cid = self._channel(channel_id)
        me = await self._me()
        params: dict[str, Any] = {"limit": int(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        msgs = await self._req("GET", f"/channels/{cid}/messages", headers=self._h(), params=params)
        for m in msgs:
            ref = m.get("referenced_message") or {}
            if (m.get("author") or {}).get("id") != me and (
                    any(u.get("id") == me for u in m.get("mentions", [])) or (ref.get("author") or {}).get("id") == me):
                self.note_addressed(f"{cid}/{m['id']}")
        return {"messages": msgs}

    async def op_send_message(self, channel_id: str, content: str,
                              reply_to_message_id: Optional[str] = None) -> dict[str, Any]:
        cid = self._channel(channel_id)
        if _MASS_MENTION.search(content):
            raise ConnectorError("discord: mass mentions are refused", 403)
        body: dict[str, Any] = {"content": content, "allowed_mentions": {"parse": [], "replied_user": False},
                                "nonce": self._idempotency_key(cid, content)[:25], "enforce_nonce": True}
        if reply_to_message_id:
            body["message_reference"] = {"message_id": str(reply_to_message_id), "fail_if_not_exists": True}
        m = await self._req("POST", f"/channels/{cid}/messages", headers=self._h(), json=body)
        return {"object_id": f"{cid}/{m['id']}", "message_id": m["id"], "channel_id": cid, "timestamp": m.get("timestamp")}

    async def op_respond_interaction(self, interaction_id: str, interaction_token: str, channel_id: str, content: str,
                                     ephemeral: bool = False) -> dict[str, Any]:
        cid = self._channel(channel_id)
        if _MASS_MENTION.search(content):
            raise ConnectorError("discord: mass mentions are refused", 403)
        data: dict[str, Any] = {"content": content, "allowed_mentions": {"parse": []}}
        if ephemeral:
            data["flags"] = EPHEMERAL
        # interaction callbacks are authenticated by the interaction token, not the bot token
        r = await self._req("POST", f"/interactions/{interaction_id}/{interaction_token}/callback",
                            params={"with_response": "true"}, json={"type": 4, "data": data})
        msg = ((r or {}).get("resource") or {}).get("message") or {}
        mid = msg.get("id") or ((r or {}).get("interaction") or {}).get("response_message_id")
        return {"object_id": f"{cid}/{mid}" if mid and not ephemeral else f"interaction:{interaction_id}",
                "message_id": mid, "ephemeral": bool(ephemeral)}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        if "/" not in object_ref:
            raise ConnectorError("discord: outcomes need a channel_id/message_id reference", 400)
        cid, mid = object_ref.split("/", 1)
        cid = self._channel(cid)
        me = await self._me()
        try:
            m = await self._req("GET", f"/channels/{cid}/messages/{mid}", headers=self._h())
        except ProviderHTTPError as e:
            if e.status == 404:
                return outcome_payload({}, "removed", {"removed": True}, raw={"ref": object_ref})
            raise
        later = await self._req("GET", f"/channels/{cid}/messages", headers=self._h(), params={"after": mid, "limit": 100})
        replies = [x for x in later if (x.get("message_reference") or {}).get("message_id") == mid
                   and (x.get("author") or {}).get("id") != me]
        reactions = sum(int(r.get("count", 0)) for r in m.get("reactions", []) or [])
        counts = {"replies": len(replies), "reactions": reactions}
        first = min((parse_time(x.get("timestamp")) for x in replies), default=None)
        return outcome_payload(counts, "visible", {"first_response": bool(replies), "positive_feedback": reactions > 0,
                                                   "removed": False}, {"first_response": first},
                               {"engagement": float(len(replies) + reactions)},
                               {"message_id": mid, "channel_id": cid, "observed_at": time.time()})

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="webhook", event_types=event_types or ["APPLICATION_COMMAND"],
                            detail={"interactions": "Interactions Endpoint URL (set by a human in the Developer Portal)",
                                    "signature_headers": ["X-Signature-Ed25519", "X-Signature-Timestamp"],
                                    "verify": "agentrs.harness.webhooks.verify_discord_signature",
                                    "public_key": (self.ctx.account or {}).get("public_key"),
                                    "messages": "poll GET /channels/{id}/messages?after=<last id> for allowed channels"})

    async def revoke(self) -> dict[str, Any]:
        return {"revoked_at_provider": False,
                "reason": ("Discord has no API to revoke a bot token; a human resets the token in the Developer "
                           "Portal (Bot -> Reset Token) or removes the bot from the server; local credentials removed")}
