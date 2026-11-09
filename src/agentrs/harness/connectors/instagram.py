"""Instagram connector (Graph API, professional accounts only, via an approved Meta app).

Credentials: ``access_token`` (long-lived user or system-user token). Account fields:
``ig_user_id``. ``base_url`` includes the Graph version (``https://graph.facebook.com/v21.0``).
Tokens are sent in the ``Authorization`` header, never in URLs.
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
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

PROFESSIONAL = {"BUSINESS", "MEDIA_CREATOR"}
THROTTLE_CODES = {"4", "17", "32", "613", "80002"}
_R = dict(tier=Tier.OBSERVE, mutates=False)


class InstagramConnector(HarnessConnector):
    provider = "instagram"
    operations = {
        "read_account": OperationSpec(name="read_account", capability="read_media", **_R),
        "read_media": OperationSpec(name="read_media", params=["media_id"], capability="read_media", **_R),
        "read_comments": OperationSpec(name="read_comments", params=["media_id"], capability="read_comments", **_R),
        "read_publishing_limit": OperationSpec(name="read_publishing_limit", capability="read_media", **_R),
        "publish_image": OperationSpec(name="publish_image", tier=Tier.EXTERNAL, mutates=True,
                                       params=["image_url", "caption"], content_param="caption",
                                       capability="publish_media", reversible=False,
                                       description="POST /{ig-user-id}/media then /media_publish"),
    }
    outcome_profile = {"publish_image": "forum_post"}
    text_ops = frozenset({"publish_image"})

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ig_user_id = str((ctx.account or {}).get("ig_user_id") or "")

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._secret('access_token')}"}

    async def _g(self, method: str, path: str, **kw) -> Any:
        try:
            return await self._req(method, path, headers=self._h(), **kw)
        except ProviderHTTPError as e:
            if e.code in THROTTLE_CODES:
                raise ProviderHTTPError(f"instagram: throttled ({e})", 429, e.body, e.code, e.retry_after) from e
            raise

    def _uid(self) -> str:
        if not self.ig_user_id:
            raise ConnectorError("instagram: account has no ig_user_id", 400)
        return self.ig_user_id

    async def _account_type(self) -> str:
        d = await self._g("GET", f"/{self._uid()}", params={"fields": "account_type,username"})
        t = str(d.get("account_type", "")).upper()
        self.memory.facts["account_type"] = t
        return t

    async def validate_credentials(self) -> dict[str, Any]:
        d = await self._g("GET", f"/{self._uid()}", params={"fields": "account_type,username,media_count"})
        t = str(d.get("account_type", "")).upper()
        self.memory.facts["account_type"] = t
        if t not in PROFESSIONAL:
            raise ConnectorError(f"instagram: account type {t or 'unknown'} is not supported "
                                 "(professional Business/Creator accounts only)", 403)
        return {"ok": True, "id": d["id"], "username": d.get("username"), "account_type": t,
                "media_count": d.get("media_count")}

    def _extra_policy(self, action: ActionRequest, entries: list[dict]) -> tuple[bool, str]:
        if action.operation == "publish_image":
            t = self.memory.facts.get("account_type") or str((self.ctx.account or {}).get("account_type", "")).upper()
            if not t:
                return False, "account type not verified yet; run validate_credentials first"
            if t not in PROFESSIONAL:
                return False, f"instagram account type {t} is not a professional account"
            if not str(action.params.get("image_url", "")).startswith("https://"):
                return False, "image_url must be an https URL"
        return True, ""

    async def op_read_account(self) -> dict[str, Any]:
        return {"account": await self._g("GET", f"/{self._uid()}",
                                         params={"fields": "account_type,username,media_count,followers_count"})}

    async def op_read_media(self, media_id: str) -> dict[str, Any]:
        return {"media": await self._g("GET", f"/{media_id}", params={
            "fields": "caption,media_type,permalink,timestamp,like_count,comments_count"})}

    async def op_read_comments(self, media_id: str) -> dict[str, Any]:
        r = await self._g("GET", f"/{media_id}/comments", params={"fields": "id,text,timestamp,username,like_count"})
        return {"comments": r.get("data", [])}

    async def op_read_publishing_limit(self) -> dict[str, Any]:
        r = await self._g("GET", f"/{self._uid()}/content_publishing_limit", params={"fields": "config,quota_usage"})
        d = (r.get("data") or [{}])[0]
        return {"quota_usage": d.get("quota_usage"), "config": d.get("config")}

    async def op_publish_image(self, image_url: str, caption: str) -> dict[str, Any]:
        uid = self._uid()
        t = await self._account_type()
        if t not in PROFESSIONAL:
            raise ConnectorError(f"instagram: account type {t} cannot publish (professional accounts only)", 403)
        lim = await self.op_read_publishing_limit()
        total = (lim.get("config") or {}).get("quota_total")
        if total is not None and (lim.get("quota_usage") or 0) >= total:
            raise ConnectorError("instagram: content publishing quota for the last 24 h is used up", 429)
        c = await self._g("POST", f"/{uid}/media", data={"image_url": image_url, "caption": caption})
        cid = c["id"]
        status: Optional[str] = None
        for _ in range(5):
            status = (await self._g("GET", f"/{cid}", params={"fields": "status_code"})).get("status_code")
            if status in ("FINISHED", "ERROR", "EXPIRED"):
                break
        if status != "FINISHED":
            raise ConnectorError(f"instagram: media container {cid} is {status}; not published", 422)
        m = await self._g("POST", f"/{uid}/media_publish", data={"creation_id": cid})
        return {"object_id": m["id"], "media_id": m["id"], "container_id": cid}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        try:
            m = await self._g("GET", f"/{object_ref}", params={"fields": "like_count,comments_count,timestamp,username"})
        except ProviderHTTPError as e:
            if e.code == "100":
                return outcome_payload({}, "removed", {"removed": True}, raw={"id": object_ref})
            raise
        comments = (await self._g("GET", f"/{object_ref}/comments",
                                  params={"fields": "id,text,timestamp,username"})).get("data", [])
        others = [c for c in comments if c.get("username") != m.get("username")]
        counts = {"likes": int(m.get("like_count", 0) or 0), "comments": int(m.get("comments_count", 0) or 0)}
        first = min((parse_time(c.get("timestamp")) for c in others), default=None)
        return outcome_payload(counts, "visible", {"first_response": bool(others), "positive_feedback": counts["likes"] > 0,
                                                   "removed": False}, {"first_response": first},
                               {"engagement": float(counts["likes"] + counts["comments"])},
                               {"id": object_ref, "published_at": parse_time(m.get("timestamp")), "observed_at": time.time()})

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types,
                            detail={"endpoint": "/{media-id}?fields=like_count,comments_count", "interval_s": 900,
                                    "note": "Meta webhooks (comments) need an app subscription set up by a human"})

    async def revoke(self) -> dict[str, Any]:
        return {"revoked_at_provider": False,
                "reason": ("a human removes the app under the Facebook/Instagram account's Business Integrations, or "
                           "revokes the system-user token in Business Settings; local credentials removed")}
