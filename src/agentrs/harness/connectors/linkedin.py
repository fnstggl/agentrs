"""LinkedIn connector (official REST API, approved app with ``openid profile w_member_social``). Refuses by default.

Every call raises ``ConnectorError(..., 403)`` unless the provider config is ``enabled`` and has an
``authorization_ref``. Credentials: ``access_token``. Account fields: optional ``person_urn``
(``urn:li:person:<id>``; otherwise derived from ``/v2/userinfo``), ``linkedin_version`` (``YYYYMM``).
"""

from __future__ import annotations

from typing import Any, Optional

from agentrs.harness.connectors._common import HarnessConnector
from agentrs.harness.connectors.base import ConnectorError
from agentrs.harness.types import ActionRequest, OperationSpec, Tier


class LinkedInConnector(HarnessConnector):
    provider = "linkedin"
    operations = {
        "read_profile": OperationSpec(name="read_profile", tier=Tier.OBSERVE, mutates=False, capability="read_profile",
                                      description="GET /v2/userinfo (OpenID Connect)"),
        "create_post": OperationSpec(name="create_post", tier=Tier.EXTERNAL, mutates=True, params=["text"],
                                     optional_params=["visibility"], content_param="text", capability="create_post",
                                     reversible=False, description="POST /rest/posts"),
    }
    outcome_profile = {"create_post": "forum_post"}
    text_ops = frozenset({"create_post"})

    def gate(self) -> None:
        cfg = self.ctx.config
        if not cfg.enabled:
            raise ConnectorError("linkedin is disabled (enable it in config/platforms.yaml after approval)", 403)
        if not cfg.authorization_ref:
            raise ConnectorError("linkedin: no authorization_ref recorded for this use of the LinkedIn API", 403)

    async def _dispatch(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        self.gate()  # refuse before anything else, even parameter validation
        return await super()._dispatch(op, params)

    def provider_policy(self, action: ActionRequest, recent: list[dict]) -> tuple[bool, str]:
        try:
            self.gate()
        except ConnectorError as e:
            return False, str(e)
        return super().provider_policy(action, recent)

    def _h(self, rest: bool = False) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._secret('access_token')}"}
        if rest:
            h.update({"LinkedIn-Version": str((self.ctx.account or {}).get("linkedin_version", "202410")),
                      "X-Restli-Protocol-Version": "2.0.0"})
        return h

    async def validate_credentials(self) -> dict[str, Any]:
        self.gate()
        u = await self._req("GET", "/v2/userinfo", headers=self._h())
        return {"ok": True, "sub": u.get("sub"), "name": u.get("name")}

    async def op_read_profile(self) -> dict[str, Any]:
        self.gate()
        return {"profile": await self._req("GET", "/v2/userinfo", headers=self._h())}

    async def _author(self) -> str:
        urn: Optional[str] = (self.ctx.account or {}).get("person_urn")
        if urn:
            return urn
        u = await self._req("GET", "/v2/userinfo", headers=self._h())
        return f"urn:li:person:{u['sub']}"

    async def op_create_post(self, text: str, visibility: str = "PUBLIC") -> dict[str, Any]:
        self.gate()
        body = {"author": await self._author(), "commentary": text, "visibility": visibility,
                "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [], "thirdPartyDistributionChannels": []},
                "lifecycleState": "PUBLISHED", "isReshareDisabledByAuthor": False}
        r = await self._call("POST", "/rest/posts", headers=self._h(rest=True), json=body)
        if r.status_code >= 400:
            raise self._error("POST", "/rest/posts", r)
        return {"object_id": r.headers.get("x-restli-id"), "status": r.status_code}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        self.gate()
        raise ConnectorError("linkedin: engagement metrics need the Community Management API; not implemented", 501)
