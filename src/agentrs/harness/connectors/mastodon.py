"""Mastodon connector (one instance per ``base_url``), OAuth 2 user token.

Credentials: ``access_token`` (+ optional ``refresh_token``), ``client_id`` / ``client_secret`` of the
registered application (needed for refresh and revocation). Account fields: ``username``, optional
``id`` and ``default_visibility`` (default ``unlisted``, which many instances require for bots).

The OAuth authorisation-code flow (with PKCE S256) is split so a human approves in the browser:
``register_app`` -> ``authorization_url`` (shown to the operator) -> ``exchange_code``.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any, Optional
from urllib.parse import urlencode

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
    parse_time,
    strip_html,
)
from agentrs.harness.connectors.base import ConnectorError, Subscription
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

_R = dict(tier=Tier.OBSERVE, mutates=False)


def make_pkce() -> tuple[str, str]:
    """(code_verifier, S256 code_challenge)."""
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


class MastodonConnector(HarnessConnector):
    provider = "mastodon"
    operations = {
        "read_profile": OperationSpec(name="read_profile", optional_params=["account_id"], capability="read_profile", **_R),
        "read_instance": OperationSpec(name="read_instance", capability="read_content",
                                       description="GET /api/v1/instance (incl. rules)", **_R),
        "read_status": OperationSpec(name="read_status", params=["status_id"], capability="read_content", **_R),
        "read_context": OperationSpec(name="read_context", params=["status_id"], capability="read_content", **_R),
        "read_home_timeline": OperationSpec(name="read_home_timeline", optional_params=["limit", "max_id"],
                                            capability="read_content", **_R),
        "read_notifications": OperationSpec(name="read_notifications", optional_params=["limit", "types", "since_id"],
                                            capability="read_notifications", **_R),
        "create_post": OperationSpec(name="create_post", tier=Tier.EXTERNAL, mutates=True, params=["text"],
                                     optional_params=["visibility", "language", "spoiler_text"], content_param="text",
                                     capability="create_post", reversible=False, description="POST /api/v1/statuses"),
        "reply": OperationSpec(name="reply", tier=Tier.EXTERNAL, mutates=True, params=["in_reply_to_id", "text"],
                               optional_params=["visibility", "language", "spoiler_text", "approved_reply"],
                               content_param="text", capability="reply", reversible=False,
                               description="POST /api/v1/statuses with in_reply_to_id"),
    }
    outcome_profile = {"create_post": "forum_post", "reply": "forum_reply"}
    text_ops = frozenset({"create_post", "reply"})
    reply_ops = frozenset({"reply"})
    reply_target_param = "in_reply_to_id"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.account_id: Optional[str] = (ctx.account or {}).get("id")

    # ------------------------------------------------------------------ OAuth (human-approved)
    async def register_app(self, client_name: str, redirect_uri: str, scopes: str = "read write",
                           website: Optional[str] = None) -> dict[str, Any]:
        body = {"client_name": client_name, "redirect_uris": redirect_uri, "scopes": scopes}
        if website:
            body["website"] = website
        r = await self._req("POST", "/api/v1/apps", data=body)
        self._remember("client_id", r["client_id"])
        self._remember("client_secret", r["client_secret"])
        return {"client_id": r["client_id"], "scopes": r.get("scopes"), "redirect_uris": r.get("redirect_uris")}

    def authorization_url(self, redirect_uri: str, scopes: str, state: str, code_challenge: str) -> str:
        q = {"response_type": "code", "client_id": self._secret("client_id"), "redirect_uri": redirect_uri,
             "scope": scopes, "state": state, "code_challenge": code_challenge, "code_challenge_method": "S256"}
        return f"{self.base_url}/oauth/authorize?{urlencode(q)}"

    async def exchange_code(self, code: str, redirect_uri: str, code_verifier: str) -> dict[str, Any]:
        r = await self._req("POST", "/oauth/token", data={
            "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
            "client_id": self._secret("client_id"), "client_secret": self._secret("client_secret"),
            "code_verifier": code_verifier})
        self._store_tokens(r)
        return {"scope": r.get("scope"), "token_type": r.get("token_type"), "created_at": r.get("created_at"),
                "expires_in": r.get("expires_in"), "refreshable": "refresh_token" in r}

    def _store_tokens(self, r: dict) -> None:
        self._remember("access_token", r["access_token"])
        if r.get("refresh_token"):
            self._remember("refresh_token", r["refresh_token"])

    async def _refresh(self) -> bool:
        rt = self._secret("refresh_token", required=False)
        cid, cs = self._secret("client_id", required=False), self._secret("client_secret", required=False)
        if not (rt and cid and cs):
            return False
        r = await self._req("POST", "/oauth/token", data={"grant_type": "refresh_token", "refresh_token": rt,
                                                          "client_id": cid, "client_secret": cs})
        self._store_tokens(r)
        return True

    async def _api(self, method: str, path: str, auth: bool = True, **kw) -> Any:
        extra = kw.pop("headers", {})
        for attempt in range(2):
            hdrs = dict(extra)
            if auth:
                hdrs["Authorization"] = f"Bearer {self._secret('access_token')}"
            try:
                return await self._req(method, path, headers=hdrs, **kw)
            except ProviderHTTPError as e:
                if attempt == 0 and auth and e.status == 401 and "expired" in str(e.code).lower() and await self._refresh():
                    continue
                raise
        raise ConnectorError("mastodon: authentication failed", 401)

    # ------------------------------------------------------------------ interface
    async def validate_credentials(self) -> dict[str, Any]:
        a = await self._api("GET", "/api/v1/accounts/verify_credentials")
        self.account_id = a["id"]
        self.memory.facts["bot"] = bool(a.get("bot"))
        self.memory.facts["acct"] = a.get("acct")
        return {"ok": True, "id": a["id"], "acct": a.get("acct"), "display_name": a.get("display_name"),
                "bot": bool(a.get("bot")),
                "warning": None if a.get("bot") else "account is not flagged as automated (bot); publishing is refused"}

    async def _me(self) -> str:
        if not self.account_id:
            await self.validate_credentials()
        return str(self.account_id)

    async def op_read_profile(self, account_id: Optional[str] = None) -> dict[str, Any]:
        if account_id:
            return {"account": await self._api("GET", f"/api/v1/accounts/{account_id}")}
        return {"account": await self._api("GET", "/api/v1/accounts/verify_credentials")}

    async def op_read_instance(self) -> dict[str, Any]:
        i = await self._api("GET", "/api/v1/instance", auth=False)
        return {"instance": {k: i.get(k) for k in ("uri", "title", "version", "registrations", "approval_required")},
                "rules": i.get("rules", []), "max_characters": ((i.get("configuration") or {}).get("statuses") or {})
                .get("max_characters")}

    async def op_read_status(self, status_id: str) -> dict[str, Any]:
        s = await self._api("GET", f"/api/v1/statuses/{status_id}")
        return {"status": s, "text": strip_html(s.get("content", ""))}

    async def op_read_context(self, status_id: str) -> dict[str, Any]:
        me = await self._me()
        c = await self._api("GET", f"/api/v1/statuses/{status_id}/context")
        for s in c.get("descendants", []):
            if s.get("in_reply_to_account_id") == me and (s.get("account") or {}).get("id") != me:
                self.note_addressed(s["id"])
        return c

    async def op_read_home_timeline(self, limit: int = 20, max_id: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(limit)}
        if max_id:
            params["max_id"] = max_id
        return {"statuses": await self._api("GET", "/api/v1/timelines/home", params=params)}

    async def op_read_notifications(self, limit: int = 30, types: Optional[list[str]] = None,
                                    since_id: Optional[str] = None) -> dict[str, Any]:
        params: list[tuple[str, Any]] = [("limit", int(limit))]
        params += [("types[]", t) for t in types or []]
        if since_id:
            params.append(("since_id", since_id))
        items = await self._api("GET", "/api/v1/notifications", params=params)
        for n in items:
            if n.get("type") == "mention" and n.get("status"):
                self.note_addressed(n["status"]["id"])
        return {"notifications": items}

    async def _post(self, text: str, in_reply_to_id: Optional[str], visibility: Optional[str], language: Optional[str],
                    spoiler_text: Optional[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"status": text,
                                "visibility": visibility or (self.ctx.account or {}).get("default_visibility", "unlisted")}
        if in_reply_to_id:
            body["in_reply_to_id"] = in_reply_to_id
        if language:
            body["language"] = language
        if spoiler_text:
            body["spoiler_text"] = spoiler_text
        key = self._idempotency_key(text, in_reply_to_id)
        s = await self._api("POST", "/api/v1/statuses", json=body, headers={"Idempotency-Key": key})
        return {"object_id": s["id"], "id": s["id"], "url": s.get("url"), "visibility": s.get("visibility"),
                "created_at": s.get("created_at")}

    async def op_create_post(self, text: str, visibility: Optional[str] = None, language: Optional[str] = None,
                             spoiler_text: Optional[str] = None) -> dict[str, Any]:
        return await self._post(text, None, visibility, language, spoiler_text)

    async def op_reply(self, in_reply_to_id: str, text: str, visibility: Optional[str] = None,
                       language: Optional[str] = None, spoiler_text: Optional[str] = None,
                       approved_reply: bool = False) -> dict[str, Any]:
        me = await self._me()
        parent = await self._api("GET", f"/api/v1/statuses/{in_reply_to_id}")
        author = parent.get("account") or {}
        addressed = (author.get("id") == me or parent.get("in_reply_to_account_id") == me
                     or any(m.get("id") == me for m in parent.get("mentions", [])))
        if not approved_reply and not addressed:
            raise ConnectorError("mastodon: parent status did not mention or reply to this account; reply refused", 403)
        if author.get("id") != me and f"@{author.get('acct')}" not in text:
            text = f"@{author.get('acct')} {text}"  # address the author, as Mastodon clients do
        if not visibility:  # never widen the audience of the conversation
            order = ["public", "unlisted", "private", "direct"]
            mine = (self.ctx.account or {}).get("default_visibility", "unlisted")
            theirs = parent.get("visibility", "public")
            visibility = max(mine, theirs, key=lambda v: order.index(v) if v in order else 1)
        return await self._post(text, in_reply_to_id, visibility, language, spoiler_text)

    def _extra_policy(self, action: ActionRequest, entries: list[dict]) -> tuple[bool, str]:
        if action.operation in self.text_ops and self.memory.facts.get("bot") is False:
            return False, "the Mastodon account is not flagged as automated (bot); set the flag before publishing"
        return True, ""

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        me = await self._me()
        try:
            s = await self._api("GET", f"/api/v1/statuses/{object_ref}")
        except ProviderHTTPError as e:
            if e.status == 404:
                return outcome_payload({}, "removed", {"removed": True}, raw={"id": object_ref})
            raise
        ctx = await self._api("GET", f"/api/v1/statuses/{object_ref}/context")
        others = [d for d in ctx.get("descendants", [])
                  if d.get("in_reply_to_id") == object_ref and (d.get("account") or {}).get("id") != me]
        counts = {"replies": int(s.get("replies_count", 0)), "reposts": int(s.get("reblogs_count", 0)),
                  "likes": int(s.get("favourites_count", 0))}
        first = min((parse_time(d.get("created_at")) for d in others), default=None)
        occurred = {"first_response": bool(others), "positive_feedback": counts["likes"] + counts["reposts"] > 0,
                    "removed": False, "edited": bool(s.get("edited_at"))}
        return outcome_payload(counts, "visible", occurred, {"first_response": first},
                               {"engagement": float(sum(counts.values()))},
                               {"id": s.get("id"), "url": s.get("url"), "visibility": s.get("visibility"),
                                "observed_at": time.time()})

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types,
                            detail={"endpoint": "/api/v1/notifications", "cursor_param": "since_id", "interval_s": 60,
                                    "note": "the streaming API (websocket) is not used"})

    async def revoke(self) -> dict[str, Any]:
        tok = self._secret("access_token", required=False)
        cid, cs = self._secret("client_id", required=False), self._secret("client_secret", required=False)
        if not (tok and cid and cs):
            return {"revoked_at_provider": False, "reason": "client credentials unknown; revoke the app in the "
                                                            "account's Settings -> Account -> Authorized apps"}
        await self._req("POST", "/oauth/revoke", data={"client_id": cid, "client_secret": cs, "token": tok})
        self._tok.pop("access_token", None)
        self._tok.pop("refresh_token", None)
        return {"revoked_at_provider": True, "reason": "token revoked with POST /oauth/revoke"}
