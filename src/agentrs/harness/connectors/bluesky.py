"""Bluesky (AT Protocol) connector: app-password session via the account's PDS.

Credentials: ``app_password`` (secret; created by the human account holder), refreshed session
tokens ``access_jwt`` / ``refresh_jwt`` are persisted with ``ctx.credentials.update``.
Account fields: ``handle`` (identifier), optionally ``did``.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone, UTC
from typing import Any, Optional

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
    parse_time,
)
from agentrs.harness.connectors.base import ConnectorError, Subscription
from agentrs.harness.types import OperationSpec, Tier

_URL = re.compile(r"https?://[^\s<>\"')\]]+")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def link_facets(text: str) -> list[dict]:
    """Rich-text link facets (byte offsets into UTF-8), so URLs render as links."""
    out = []
    for m in _URL.finditer(text):
        start = len(text[:m.start()].encode())
        end = start + len(m.group(0).encode())
        out.append({"index": {"byteStart": start, "byteEnd": end},
                    "features": [{"$type": "app.bsky.richtext.facet#link", "uri": m.group(0)}]})
    return out


class BlueskyConnector(HarnessConnector):
    provider = "bluesky"
    operations = {
        "read_profile": OperationSpec(name="read_profile", tier=Tier.OBSERVE, mutates=False, optional_params=["actor"],
                                      capability="read_profile", description="app.bsky.actor.getProfile"),
        "read_thread": OperationSpec(name="read_thread", tier=Tier.OBSERVE, mutates=False, params=["uri"],
                                     optional_params=["depth"], capability="read_content",
                                     description="app.bsky.feed.getPostThread"),
        "read_author_feed": OperationSpec(name="read_author_feed", tier=Tier.OBSERVE, mutates=False,
                                          optional_params=["actor", "limit", "cursor"], capability="read_content",
                                          description="app.bsky.feed.getAuthorFeed"),
        "read_notifications": OperationSpec(name="read_notifications", tier=Tier.OBSERVE, mutates=False,
                                            optional_params=["limit", "cursor"], capability="read_content",
                                            description="app.bsky.notification.listNotifications"),
        "create_post": OperationSpec(name="create_post", tier=Tier.EXTERNAL, mutates=True, params=["text"],
                                     optional_params=["langs"], content_param="text", capability="create_post",
                                     reversible=False, description="createRecord app.bsky.feed.post"),
        "reply": OperationSpec(name="reply", tier=Tier.EXTERNAL, mutates=True, params=["parent_uri", "text"],
                               optional_params=["langs", "approved_reply"], content_param="text", capability="reply",
                               reversible=False, description="createRecord app.bsky.feed.post with reply refs"),
        # capability key 'update_profile' is not in config/platforms.yaml yet -> fails closed until added
        "set_bot_label": OperationSpec(name="set_bot_label", tier=Tier.REVERSIBLE, mutates=True,
                                       capability="update_profile", rate_key="write",
                                       description="putRecord app.bsky.actor.profile with self-label 'bot'"),
    }
    outcome_profile = {"create_post": "forum_post", "reply": "forum_reply"}
    text_ops = frozenset({"create_post", "reply"})
    reply_ops = frozenset({"reply"})
    reply_target_param = "parent_uri"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.did: Optional[str] = (ctx.account or {}).get("did")
        self.handle: Optional[str] = (ctx.account or {}).get("handle")

    # ------------------------------------------------------------------ session
    async def _create_session(self) -> None:
        ident = self.handle or self.did
        if not ident:
            raise ConnectorError("bluesky: account has no handle", 400)
        r = await self._req("POST", "/xrpc/com.atproto.server.createSession",
                            json={"identifier": ident, "password": self._secret("app_password")})
        self._session_from(r)

    def _session_from(self, r: dict) -> None:
        self._remember("access_jwt", r["accessJwt"])
        self._remember("refresh_jwt", r["refreshJwt"])
        self.did, self.handle = r.get("did", self.did), r.get("handle", self.handle)

    async def _refresh(self) -> None:
        ref = self._secret("refresh_jwt", required=False)
        if ref:
            try:
                r = await self._req("POST", "/xrpc/com.atproto.server.refreshSession",
                                    headers={"Authorization": f"Bearer {ref}"})
                self._session_from(r)
                return
            except ProviderHTTPError as e:
                if e.status == 429:
                    raise
        await self._create_session()

    async def _access(self) -> str:
        tok = self._secret("access_jwt", required=False)
        if not tok:
            await self._create_session()
            tok = self._secret("access_jwt")
        if not self.did:
            s = await self._req("GET", "/xrpc/com.atproto.server.getSession", headers={"Authorization": f"Bearer {tok}"})
            self.did, self.handle = s["did"], s.get("handle", self.handle)
        return tok

    async def _x(self, method: str, nsid: str, **kw) -> Any:
        extra = kw.pop("headers", {})
        for attempt in range(2):
            tok = await self._access()
            hdrs = {"Authorization": f"Bearer {tok}", **extra}
            try:
                return await self._req(method, f"/xrpc/{nsid}", headers=hdrs, **kw)
            except ProviderHTTPError as e:
                expired = e.code == "ExpiredToken" or (e.status == 401 and e.code in ("InvalidToken", "AuthenticationRequired"))
                if attempt == 0 and expired:
                    await self._refresh()
                    continue
                raise
        raise ConnectorError("bluesky: authentication failed", 401)

    # ------------------------------------------------------------------ interface
    async def validate_credentials(self) -> dict[str, Any]:
        await self._access()
        p = await self._x("GET", "app.bsky.actor.getProfile", params={"actor": self.did})
        labels = [lb.get("val") for lb in p.get("labels", []) if lb.get("src") == self.did]
        return {"ok": True, "did": self.did, "handle": p.get("handle"), "display_name": p.get("displayName"),
                "self_labels": labels, "bot_label": "bot" in labels}

    async def op_read_profile(self, actor: Optional[str] = None) -> dict[str, Any]:
        await self._access()
        return {"profile": await self._x("GET", "app.bsky.actor.getProfile", params={"actor": actor or self.did})}

    async def op_read_thread(self, uri: str, depth: int = 6) -> dict[str, Any]:
        r = await self._x("GET", "app.bsky.feed.getPostThread", params={"uri": uri, "depth": int(depth)})
        self._note_thread(r.get("thread") or {})
        return r

    def _note_thread(self, node: dict) -> None:
        post = node.get("post") or {}
        mine = (post.get("author") or {}).get("did") == self.did
        for child in node.get("replies") or []:
            cp = child.get("post") or {}
            if mine and (cp.get("author") or {}).get("did") != self.did:
                self.note_addressed(cp.get("uri"))
            self._note_thread(child)

    async def op_read_author_feed(self, actor: Optional[str] = None, limit: int = 30,
                                  cursor: Optional[str] = None) -> dict[str, Any]:
        await self._access()
        params: dict[str, Any] = {"actor": actor or self.did, "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        return await self._x("GET", "app.bsky.feed.getAuthorFeed", params=params)

    async def op_read_notifications(self, limit: int = 50, cursor: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        r = await self._x("GET", "app.bsky.notification.listNotifications", params=params)
        for n in r.get("notifications", []):
            if n.get("reason") in ("mention", "reply", "quote"):
                self.note_addressed(n.get("uri"))
        return r

    def _post_record(self, text: str, langs: Optional[list[str]]) -> dict[str, Any]:
        rec: dict[str, Any] = {"$type": "app.bsky.feed.post", "text": text, "createdAt": _now_iso()}
        facets = link_facets(text)
        if facets:
            rec["facets"] = facets
        if langs:
            rec["langs"] = list(langs)
        return rec

    async def _create(self, record: dict) -> dict[str, Any]:
        await self._access()
        r = await self._x("POST", "com.atproto.repo.createRecord",
                          json={"repo": self.did, "collection": "app.bsky.feed.post", "record": record})
        return {"object_id": r["uri"], "uri": r["uri"], "cid": r["cid"]}

    async def op_create_post(self, text: str, langs: Optional[list[str]] = None) -> dict[str, Any]:
        return await self._create(self._post_record(text, langs))

    def _addresses_us(self, post: dict) -> bool:
        rec = post.get("record") or {}
        if (post.get("author") or {}).get("did") == self.did:
            return True
        for fc in rec.get("facets") or []:
            for f in fc.get("features", []):
                if f.get("$type") == "app.bsky.richtext.facet#mention" and f.get("did") == self.did:
                    return True
        if self.handle and f"@{self.handle}".lower() in str(rec.get("text", "")).lower():
            return True
        parent = ((rec.get("reply") or {}).get("parent") or {}).get("uri", "")
        return bool(self.did) and parent.startswith(f"at://{self.did}/")

    async def op_reply(self, parent_uri: str, text: str, langs: Optional[list[str]] = None,
                       approved_reply: bool = False) -> dict[str, Any]:
        t = await self._x("GET", "app.bsky.feed.getPostThread", params={"uri": parent_uri, "depth": 0, "parentHeight": 0})
        parent = (t.get("thread") or {}).get("post")
        if not parent:
            raise ConnectorError(f"bluesky: parent post {parent_uri} is not available", 404)
        if not approved_reply and not self._addresses_us(parent):
            raise ConnectorError("bluesky: parent post did not mention or reply to this account; reply refused", 403)
        root = ((parent.get("record") or {}).get("reply") or {}).get("root") or {"uri": parent["uri"], "cid": parent["cid"]}
        rec = self._post_record(text, langs)
        rec["reply"] = {"root": {"uri": root["uri"], "cid": root["cid"]}, "parent": {"uri": parent["uri"], "cid": parent["cid"]}}
        return await self._create(rec)

    async def op_set_bot_label(self) -> dict[str, Any]:
        """Self-label the account's profile as 'bot', preserving the rest of the profile record."""
        await self._access()
        existing, swap = {}, None
        try:
            cur = await self._req("GET", "/xrpc/com.atproto.repo.getRecord",
                                  params={"repo": self.did, "collection": "app.bsky.actor.profile", "rkey": "self"})
            existing, swap = cur.get("value") or {}, cur.get("cid")
        except ProviderHTTPError as e:
            if e.code != "RecordNotFound" and e.status not in (400, 404):
                raise
        labels = existing.get("labels") or {"$type": "com.atproto.label.defs#selfLabels", "values": []}
        vals = [v for v in labels.get("values", []) if isinstance(v, dict)]
        changed = not any(v.get("val") == "bot" for v in vals)
        if changed:
            vals.append({"val": "bot"})
        record = {**existing, "$type": "app.bsky.actor.profile",
                  "labels": {"$type": "com.atproto.label.defs#selfLabels", "values": vals}}
        body: dict[str, Any] = {"repo": self.did, "collection": "app.bsky.actor.profile", "rkey": "self", "record": record}
        if swap:
            body["swapRecord"] = swap
        r = await self._x("POST", "com.atproto.repo.putRecord", json=body)
        return {"object_id": r["uri"], "uri": r["uri"], "cid": r["cid"], "changed": changed,
                "self_labels": [v["val"] for v in vals]}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        await self._access()
        try:
            t = await self._x("GET", "app.bsky.feed.getPostThread", params={"uri": object_ref, "depth": 1, "parentHeight": 0})
        except ProviderHTTPError as e:
            if e.code == "NotFound":
                return outcome_payload({}, "removed", {"removed": True}, raw={"uri": object_ref, "error": e.code})
            raise
        node = t.get("thread") or {}
        if node.get("$type") in ("app.bsky.feed.defs#notFoundPost", "app.bsky.feed.defs#blockedPost"):
            return outcome_payload({}, "removed", {"removed": True}, raw={"uri": object_ref, "type": node.get("$type")})
        post = node.get("post") or {}
        replies = [c.get("post") or {} for c in node.get("replies") or []]
        others = [p for p in replies if (p.get("author") or {}).get("did") != self.did]
        counts = {"replies": int(post.get("replyCount", 0)), "reposts": int(post.get("repostCount", 0)),
                  "likes": int(post.get("likeCount", 0)), "quotes": int(post.get("quoteCount", 0))}
        first = min((parse_time(p.get("indexedAt")) for p in others if p.get("indexedAt")), default=None)
        occurred = {"first_response": bool(others), "positive_feedback": counts["likes"] + counts["reposts"] > 0,
                    "removed": False}
        return outcome_payload(counts, "visible", occurred, {"first_response": first},
                               {"engagement": float(counts["likes"] + counts["reposts"] + counts["replies"])},
                               {"uri": post.get("uri"), "cid": post.get("cid"), "observed_at": time.time()})

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types,
                            detail={"endpoint": "app.bsky.notification.listNotifications", "cursor": "cursor",
                                    "interval_s": 60, "note": "firehose/Jetstream not used; notifications are polled"})

    async def revoke(self) -> dict[str, Any]:
        ref = self._secret("refresh_jwt", required=False)
        ended = False
        if ref:
            r = await self._call("POST", "/xrpc/com.atproto.server.deleteSession", headers={"Authorization": f"Bearer {ref}"})
            ended = r.status_code < 400
        self._tok.clear()
        return {"revoked_at_provider": ended, "scope": "session",
                "reason": ("session ended at the PDS; the app password itself can only be revoked by the account "
                           "holder in Settings -> Privacy and security -> App passwords")}
