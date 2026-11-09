"""X (API v2) connector, OAuth 2.0 user-context token of an approved developer app.

Credentials: ``access_token`` (+ optional ``refresh_token``; refreshed with the public-client
``refresh_token`` grant using account ``client_id``). Account fields: ``username``, optional ``id``,
``client_id``. ``base_url`` is the v2 root (``https://api.x.com/2``).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
    parse_time,
)
from agentrs.harness.connectors.base import ConnectorError, Subscription
from agentrs.harness.types import OperationSpec, Tier

FIELDS = "public_metrics,created_at,author_id,conversation_id,in_reply_to_user_id,entities,referenced_tweets"


class XConnector(HarnessConnector):
    provider = "x"
    operations = {
        "read_post": OperationSpec(name="read_post", tier=Tier.OBSERVE, mutates=False, params=["tweet_id"],
                                   capability="read_content", description="GET /2/tweets/:id"),
        "create_post": OperationSpec(name="create_post", tier=Tier.EXTERNAL, mutates=True, params=["text"],
                                     content_param="text", capability="create_post", reversible=False,
                                     description="POST /2/tweets"),
        "reply": OperationSpec(name="reply", tier=Tier.EXTERNAL, mutates=True, params=["in_reply_to_tweet_id", "text"],
                               optional_params=["approved_reply"], content_param="text", capability="reply",
                               reversible=False, description="POST /2/tweets with reply.in_reply_to_tweet_id"),
        "like": OperationSpec(name="like", tier=Tier.EXTERNAL, mutates=True, params=["tweet_id"], capability="like",
                              rate_key="like", description="POST /2/users/:id/likes"),
        # deleting one's own post undoes create_post; it uses the create_post capability
        "delete_post": OperationSpec(name="delete_post", tier=Tier.REVERSIBLE, mutates=True, params=["tweet_id"],
                                     capability="create_post", reversible=False, description="DELETE /2/tweets/:id"),
    }
    outcome_profile = {"create_post": "forum_post", "reply": "forum_reply"}
    text_ops = frozenset({"create_post", "reply"})
    reply_ops = frozenset({"reply"})
    like_ops = frozenset({"like"})
    reply_target_param = "in_reply_to_tweet_id"
    engage_target_param = "tweet_id"

    def __init__(self, ctx):
        super().__init__(ctx)
        a = ctx.account or {}
        self.user_id: Optional[str] = a.get("id")
        self.username: Optional[str] = a.get("username")

    async def _refresh(self) -> bool:
        rt = self._secret("refresh_token", required=False)
        cid = (self.ctx.account or {}).get("client_id")
        if not (rt and cid):
            return False
        r = await self._req("POST", "/oauth2/token", data={"grant_type": "refresh_token", "refresh_token": rt,
                                                           "client_id": cid})
        self._remember("access_token", r["access_token"])
        if r.get("refresh_token"):
            self._remember("refresh_token", r["refresh_token"])
        return True

    async def _api(self, method: str, path: str, **kw) -> Any:
        for attempt in range(2):
            hdrs = {"Authorization": f"Bearer {self._secret('access_token')}"}
            try:
                return await self._req(method, path, headers=hdrs, **kw)
            except ProviderHTTPError as e:
                if attempt == 0 and e.status == 401 and await self._refresh():
                    continue
                raise
        raise ConnectorError("x: authentication failed", 401)

    async def validate_credentials(self) -> dict[str, Any]:
        d = (await self._api("GET", "/users/me"))["data"]
        self.user_id, self.username = d["id"], d["username"]
        return {"ok": True, "id": d["id"], "username": d["username"], "name": d.get("name")}

    async def _me(self) -> str:
        if not self.user_id:
            await self.validate_credentials()
        return str(self.user_id)

    async def _tweet(self, tweet_id: str) -> Optional[dict]:
        r = await self._api("GET", f"/tweets/{tweet_id}", params={"tweet.fields": FIELDS})
        return r.get("data")  # missing / deleted tweets come back as 200 with an "errors" array

    async def op_read_post(self, tweet_id: str) -> dict[str, Any]:
        me = await self._me()
        t = await self._tweet(tweet_id)
        if t is None:
            raise ConnectorError(f"x: post {tweet_id} not found", 404)
        mentions = ((t.get("entities") or {}).get("mentions") or [])
        if t.get("author_id") != me and (t.get("in_reply_to_user_id") == me or any(m.get("id") == me for m in mentions)):
            self.note_addressed(t["id"])
        return {"post": t}

    async def op_create_post(self, text: str) -> dict[str, Any]:
        d = (await self._api("POST", "/tweets", json={"text": text}))["data"]
        return {"object_id": d["id"], "id": d["id"], "text": d.get("text")}

    async def op_reply(self, in_reply_to_tweet_id: str, text: str, approved_reply: bool = False) -> dict[str, Any]:
        me = await self._me()
        parent = await self._tweet(in_reply_to_tweet_id)
        if parent is None:
            raise ConnectorError(f"x: post {in_reply_to_tweet_id} not found", 404)
        mentions = ((parent.get("entities") or {}).get("mentions") or [])
        addressed = parent.get("author_id") == me or parent.get("in_reply_to_user_id") == me or any(
            m.get("id") == me for m in mentions)
        if not approved_reply and not addressed:
            raise ConnectorError("x: the post did not mention or reply to this account; reply refused", 403)
        d = (await self._api("POST", "/tweets", json={"text": text, "reply": {"in_reply_to_tweet_id": in_reply_to_tweet_id}}))["data"]
        return {"object_id": d["id"], "id": d["id"], "in_reply_to": in_reply_to_tweet_id}

    async def op_like(self, tweet_id: str) -> dict[str, Any]:
        me = await self._me()
        d = (await self._api("POST", f"/users/{me}/likes", json={"tweet_id": str(tweet_id)}))["data"]
        return {"liked": bool(d.get("liked")), "tweet_id": tweet_id}

    async def op_delete_post(self, tweet_id: str) -> dict[str, Any]:
        d = (await self._api("DELETE", f"/tweets/{tweet_id}"))["data"]
        return {"deleted": bool(d.get("deleted")), "tweet_id": tweet_id}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        me = await self._me()
        t = await self._tweet(object_ref)
        if t is None:
            return outcome_payload({}, "removed", {"removed": True}, raw={"id": object_ref})
        m = t.get("public_metrics") or {}
        counts = {"replies": int(m.get("reply_count", 0)), "reposts": int(m.get("retweet_count", 0)),
                  "likes": int(m.get("like_count", 0)), "quotes": int(m.get("quote_count", 0))}
        raw: dict[str, Any] = {"id": t["id"], "impressions": m.get("impression_count"), "observed_at": time.time()}
        first, others = None, None
        try:  # exact first-reply time needs search (paid access tiers); otherwise only counts
            s = await self._api("GET", "/tweets/search/recent",
                                params={"query": f"conversation_id:{t.get('conversation_id', t['id'])}",
                                        "tweet.fields": "author_id,created_at,referenced_tweets", "max_results": 100})
            others = [x for x in s.get("data", []) if x.get("author_id") != me and any(
                r.get("type") == "replied_to" and r.get("id") == t["id"] for r in x.get("referenced_tweets", []))]
            first = min((parse_time(x.get("created_at")) for x in others), default=None)
        except ProviderHTTPError as e:
            if e.status not in (400, 403):
                raise
            raw["search_unavailable"] = e.status
        responded = bool(others) if others is not None else counts["replies"] > 0
        return outcome_payload(counts, "visible", {"first_response": responded,
                                                   "positive_feedback": counts["likes"] + counts["reposts"] > 0,
                                                   "removed": False}, {"first_response": first},
                               {"engagement": float(sum(counts.values()))}, raw)

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types,
                            detail={"endpoint": "/2/tweets/:id (public_metrics)", "interval_s": 300,
                                    "note": "Account Activity webhooks need an enterprise/paid tier; not used"})

    async def revoke(self) -> dict[str, Any]:
        return {"revoked_at_provider": False,
                "reason": ("token revocation (POST /2/oauth2/revoke) needs the app's client credentials, which the "
                           "harness does not hold; a human revokes the app under Settings -> Security and account "
                           "access -> Apps and sessions. Local credentials removed")}
