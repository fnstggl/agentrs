"""SocialWebLab connector: the self-hosted sandbox (tier 1) and default Reddit-like training
environment. Talks to the lab's JSON API over real HTTP with a session token held by the
credential handle; outcomes are read from public item pages (the same view any user has)."""

from __future__ import annotations

from typing import Any

from agentrs.harness.connectors.base import ConnectorError, PlatformConnector, Subscription
from agentrs.harness.types import OperationSpec, Tier
from agentrs.sensors.socialweblab_source import payload_from_item


class SocialWebLabConnector(PlatformConnector):
    provider = "socialweblab"
    operations = {
        "read_feed": OperationSpec(name="read_feed", tier=Tier.OBSERVE, mutates=False, optional_params=["space", "sort"],
                                   capability="read_content"),
        "read_item": OperationSpec(name="read_item", tier=Tier.OBSERVE, mutates=False, params=["item_id"],
                                   capability="read_content"),
        "read_user": OperationSpec(name="read_user", tier=Tier.OBSERVE, mutates=False, params=["handle"],
                                   capability="read_content"),
        "create_post": OperationSpec(name="create_post", tier=Tier.SANDBOX, mutates=True, params=["space", "title", "body"],
                                     optional_params=["client_ref"], content_param="body", capability="create_post"),
        "reply": OperationSpec(name="reply", tier=Tier.SANDBOX, mutates=True, params=["item_id", "body"],
                               optional_params=["kind"], content_param="body", capability="reply"),
        "react": OperationSpec(name="react", tier=Tier.SANDBOX, mutates=True, params=["item_id", "kind"], capability="react"),
    }

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.ctx.credentials.get('session_token')}"}

    async def validate_credentials(self) -> dict[str, Any]:
        n = await self._json("GET", "/api/notifications", headers=self._auth())
        return {"ok": True, "unread": n.get("unread")}

    async def op_read_feed(self, space=None, sort="hot") -> dict:
        q = {"sort": sort}
        if space:
            q["space"] = space
        return await self._json("GET", "/api/feed", headers=self._auth(), params=q)

    async def op_read_item(self, item_id) -> dict:
        return await self._json("GET", f"/api/items/{int(item_id)}", headers=self._auth())

    async def op_read_user(self, handle) -> dict:
        return await self._json("GET", f"/api/users/{handle}", headers=self._auth())

    async def op_create_post(self, space, title, body, client_ref=None) -> dict:
        payload = {"space": space, "title": title, "body": body}
        if client_ref:
            payload["client_ref"] = client_ref
        r = await self._json("POST", "/api/items", headers=self._auth(), json=payload)
        return {"object_id": str(r["id"]), "item_id": r["id"]}

    async def op_reply(self, item_id, body, kind="answer") -> dict:
        r = await self._json("POST", f"/api/items/{int(item_id)}/responses", headers=self._auth(),
                             json={"body": body, "kind": kind})
        return {"object_id": f"{item_id}#{r['id']}", "response_id": r["id"]}

    async def op_react(self, item_id, kind) -> dict:
        return await self._json("POST", f"/api/items/{int(item_id)}/react", headers=self._auth(), json={"kind": kind})

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        item = object_ref.split("#")[0]
        it = await self._json("GET", f"/api/items/{int(item)}", headers=self._auth())
        p = payload_from_item(it)
        p["raw"] = {"status": it["status"], "counts": it["counts"]}
        return p

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types, detail={"endpoint": "/api/events"})

    async def lab_now(self) -> float:
        r = await self._json("GET", "/api/events", headers=self._auth(), params={"since": 10 ** 12, "limit": 1})
        if "now" not in r:
            raise ConnectorError("lab clock unavailable")
        return float(r["now"])
