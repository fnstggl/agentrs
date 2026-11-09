"""Reddit connector (official OAuth API at ``oauth.reddit.com``). Refuses by default.

Every call raises ``ConnectorError(..., 403)`` unless the provider config is ``enabled``, has an
``authorization_ref`` (the operator's recorded approval for this use of the Data API) and
*explicitly declares* ``training_data_authorized`` (present in the config, True or False). When it is
False, reads are still possible for non-training use only: every result carries
``data_use: "non_training_only"`` and ``fetch_outcome`` (whose purpose is to produce learning
signals) is refused.

Credentials: ``access_token`` (+ optional ``refresh_token``, ``client_id`` / ``client_secret`` for the
refresh grant at ``www.reddit.com/api/v1/access_token``). Account fields: ``username``,
``user_agent`` (Reddit requires ``<platform>:<app id>:<version> (by /u/<username>)``).
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
)
from agentrs.harness.connectors.base import ConnectorError
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


class RedditConnector(HarnessConnector):
    provider = "reddit"
    operations = {
        "read_content": OperationSpec(name="read_content", tier=Tier.OBSERVE, mutates=False, params=["subreddit"],
                                      optional_params=["sort", "limit", "after"], capability="read_content",
                                      description="GET /r/{subreddit}/{sort}"),
        "create_post": OperationSpec(name="create_post", tier=Tier.EXTERNAL, mutates=True,
                                     params=["subreddit", "title", "text"], content_param="text",
                                     capability="create_post", reversible=False, description="POST /api/submit (kind=self)"),
        "reply": OperationSpec(name="reply", tier=Tier.EXTERNAL, mutates=True, params=["parent_fullname", "text"],
                               optional_params=["approved_reply"], content_param="text", capability="reply",
                               reversible=False, description="POST /api/comment"),
    }
    outcome_profile = {"create_post": "forum_post", "reply": "forum_reply"}
    text_ops = frozenset({"create_post", "reply"})
    reply_ops = frozenset({"reply"})
    reply_target_param = "parent_fullname"

    # ------------------------------------------------------------------ gate
    def gate(self) -> dict[str, Any]:
        cfg = self.ctx.config
        if not cfg.enabled:
            raise ConnectorError("reddit is disabled (enable it in config/platforms.yaml after approval)", 403)
        if not cfg.authorization_ref:
            raise ConnectorError("reddit: no authorization_ref recorded for this use of the Reddit Data API", 403)
        if "training_data_authorized" not in cfg.model_fields_set:
            raise ConnectorError("reddit: training_data_authorized must be explicitly declared (true or false)", 403)
        allowed = bool(cfg.training_data_authorized)
        return {"training_data_authorized": allowed, "data_use": "training_permitted" if allowed else "non_training_only",
                "authorization_ref": cfg.authorization_ref}

    async def _dispatch(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        self.gate()  # refuse before anything else, even parameter validation
        return await super()._dispatch(op, params)

    def provider_policy(self, action: ActionRequest, recent: list[dict]) -> tuple[bool, str]:
        try:
            self.gate()
        except ConnectorError as e:
            return False, str(e)
        return super().provider_policy(action, recent)

    def _h(self) -> dict[str, str]:
        ua = (self.ctx.account or {}).get("user_agent")
        if not ua:
            raise ConnectorError("reddit: account needs a descriptive user_agent", 400)
        return {"Authorization": f"bearer {self._secret('access_token')}", "User-Agent": ua}

    async def _refresh(self) -> bool:
        rt = self._secret("refresh_token", required=False)
        cid, cs = self._secret("client_id", required=False), self._secret("client_secret", required=False)
        if not (rt and cid and cs):
            return False
        basic = base64.b64encode(f"{cid}:{cs}".encode()).decode()
        r = await self._req("POST", TOKEN_URL, data={"grant_type": "refresh_token", "refresh_token": rt},
                            headers={"Authorization": f"Basic {basic}", "User-Agent": self._h()["User-Agent"]})
        self._remember("access_token", r["access_token"])
        return True

    async def _api(self, method: str, path: str, **kw) -> Any:
        for attempt in range(2):
            try:
                return await self._req(method, path, headers=self._h(), **kw)
            except ProviderHTTPError as e:
                if attempt == 0 and e.status == 401 and await self._refresh():
                    continue
                raise
        raise ConnectorError("reddit: authentication failed", 401)

    # ------------------------------------------------------------------ interface
    async def validate_credentials(self) -> dict[str, Any]:
        g = self.gate()
        me = await self._api("GET", "/api/v1/me", params={"raw_json": 1})
        return {"ok": True, "name": me.get("name"), "id": me.get("id"), **g}

    async def op_read_content(self, subreddit: str, sort: str = "new", limit: int = 25,
                              after: Optional[str] = None) -> dict[str, Any]:
        g = self.gate()
        if sort not in ("new", "hot", "top", "rising"):
            raise ConnectorError(f"reddit: unsupported sort {sort!r}", 400)
        params: dict[str, Any] = {"limit": int(limit), "raw_json": 1}
        if after:
            params["after"] = after
        listing = await self._api("GET", f"/r/{subreddit}/{sort}", params=params)
        items = [c.get("data", {}) for c in (listing.get("data") or {}).get("children", [])]
        return {"posts": [{k: p.get(k) for k in ("name", "title", "selftext", "author", "score", "num_comments",
                                                 "created_utc", "permalink")} for p in items],
                "after": (listing.get("data") or {}).get("after"), **g}

    async def op_create_post(self, subreddit: str, title: str, text: str) -> dict[str, Any]:
        g = self.gate()
        r = await self._api("POST", "/api/submit", data={"sr": subreddit, "kind": "self", "title": title, "text": text,
                                                         "api_type": "json", "resubmit": "false", "sendreplies": "true"})
        j = (r.get("json") or {})
        if j.get("errors"):
            raise ConnectorError(f"reddit: submit failed: {j['errors']}", 400)
        d = j.get("data") or {}
        return {"object_id": d.get("name"), "url": d.get("url"), **g}

    async def op_reply(self, parent_fullname: str, text: str, approved_reply: bool = False) -> dict[str, Any]:
        g = self.gate()
        r = await self._api("POST", "/api/comment", data={"thing_id": parent_fullname, "text": text, "api_type": "json"})
        j = (r.get("json") or {})
        if j.get("errors"):
            raise ConnectorError(f"reddit: comment failed: {j['errors']}", 400)
        things = (j.get("data") or {}).get("things") or [{}]
        return {"object_id": (things[0].get("data") or {}).get("name"), **g}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        g = self.gate()
        if not g["training_data_authorized"]:
            raise ConnectorError("reddit: outcomes are learning signals and training_data_authorized is false", 403)
        r = await self._api("GET", "/api/info", params={"id": object_ref, "raw_json": 1})
        children = (r.get("data") or {}).get("children") or []
        if not children:
            return outcome_payload({}, "removed", {"removed": True}, raw={"id": object_ref})
        d = children[0].get("data", {})
        removed = d.get("removed_by_category") is not None or d.get("author") == "[deleted]"
        counts = {"score": int(d.get("score", 0)), "comments": int(d.get("num_comments", 0))}
        return outcome_payload(counts, "removed" if removed else "visible",
                               {"first_response": counts["comments"] > 0, "removed": removed}, {},
                               {"score": float(counts["score"])}, {"id": object_ref, **g})

    async def revoke(self) -> dict[str, Any]:
        self.gate()
        tok = self._secret("access_token", required=False)
        cid, cs = self._secret("client_id", required=False), self._secret("client_secret", required=False)
        if not (tok and cid and cs):
            return {"revoked_at_provider": False, "reason": "client credentials unknown; revoke the app at "
                                                            "reddit.com/prefs/apps"}
        basic = base64.b64encode(f"{cid}:{cs}".encode()).decode()
        await self._req("POST", "https://www.reddit.com/api/v1/revoke_token",
                        data={"token": tok, "token_type_hint": "access_token"},
                        headers={"Authorization": f"Basic {basic}", "User-Agent": self._h()["User-Agent"]})
        return {"revoked_at_provider": True, "reason": "token revoked with /api/v1/revoke_token"}
