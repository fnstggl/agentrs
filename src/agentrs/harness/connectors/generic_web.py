"""Generic website connector (browser automation) for sites where automating the intended activity
is permitted. Requires an explicit site allowlist and a per-site permission profile
(``config/permissions.yaml: web_sites``); reachability is never permission.

Every operation names the site; the session for (account, task, site) is an isolated browser
session from the ``BrowserWorkerPool``. Tiers: reads 0; interactions use the site's profile tier
(default 3 = external); controls whose accessible name looks high impact (buy, pay, delete
account, transfer, …) are refused unless the action was approved at tier 4.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Optional

from agentrs.actions.schema import Action
from agentrs.harness.connectors.base import ConnectorError, PlatformConnector
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

HIGH_IMPACT = re.compile(r"(?i)\b(buy|purchase|pay|checkout|place order|transfer|delete (my )?account|close account|"
                         r"change password|disable 2fa|confirm payment|subscribe)\b")


class GenericWebConnector(PlatformConnector):
    provider = "web"
    pool: ClassVar[Any] = None  # BrowserWorkerPool, set by the harness wiring
    site_profiles: ClassVar[dict[str, dict]] = {}  # host -> {"tier": int, "start_url": str, "allowed_prefixes": [...]}
    operations = {
        "navigate": OperationSpec(name="navigate", tier=Tier.OBSERVE, mutates=False, params=["site", "url"]),
        "read_page": OperationSpec(name="read_page", tier=Tier.OBSERVE, mutates=False, params=["site"]),
        "screenshot": OperationSpec(name="screenshot", tier=Tier.OBSERVE, mutates=False, params=["site"]),
        "wait": OperationSpec(name="wait", tier=Tier.OBSERVE, mutates=False, params=["site"], capability="read_page"),
        "fill": OperationSpec(name="fill", tier=Tier.EXTERNAL, mutates=True, params=["site", "element", "text"],
                              content_param="text"),
        "click": OperationSpec(name="click", tier=Tier.EXTERNAL, mutates=True, params=["site", "element"],
                               optional_params=["approved_high_impact"]),
        "upload_file": OperationSpec(name="upload_file", tier=Tier.EXTERNAL, mutates=True,
                                     params=["site", "element", "object_key"]),
    }

    def __init__(self, ctx):
        super().__init__(ctx)
        self._sessions: dict[str, Any] = {}
        self.task_id = "default"

    @classmethod
    def tier_for(cls, action: ActionRequest, spec: OperationSpec) -> Tier:
        if not spec.mutates:
            return spec.tier
        prof = cls.site_profiles.get(str(action.params.get("site", "")))
        if prof is None:
            return Tier.HIGH_IMPACT  # unknown site: never a routine action
        t = Tier(int(prof.get("tier", Tier.EXTERNAL)))
        return Tier.HIGH_IMPACT if action.params.get("approved_high_impact") else t

    def _session(self, site: str):
        prof = self.site_profiles.get(site)
        if prof is None:
            raise ConnectorError(f"site {site!r} has no permission profile", 403)
        if self.pool is None:
            raise ConnectorError("no browser pool configured", 503)
        s = self._sessions.get(site)
        if s is None or s.id not in self.pool.sessions:
            s = self.pool.open(self.ctx.account["id"], agent_id="gateway", task_id=self.task_id,
                               start_url=prof["start_url"], allowed_prefixes=prof.get("allowed_prefixes", [prof["start_url"]]))
            s.env.reset_url = prof["start_url"]
            from agentrs.environments.base import TaskSpec

            s.env.reset(TaskSpec(task_id=f"web/{site}", env_id="web", family="web", goal="", max_steps=10_000,
                                 private={"start_url": prof["start_url"]}))
            self._sessions[site] = s
        s.use()
        return s

    async def validate_credentials(self) -> dict[str, Any]:
        return {"ok": True, "note": "browser sessions authenticate per site through stored session state"}

    def _obs(self, s) -> dict:
        o = s.env.observe()
        return {"url": o.url, "title": o.title, "page": o.to_text(80), "error": o.last_action_error}

    def _do(self, s, a: Action) -> dict:
        r = s.env.step(a)
        out = self._obs(s)
        out["action_error"] = r.error
        if r.error:
            raise ConnectorError(r.error, 400)
        return out

    async def op_navigate(self, site, url) -> dict:
        return self._do(self._session(site), Action.navigate(url))

    async def op_read_page(self, site) -> dict:
        return self._obs(self._session(site))

    async def op_wait(self, site) -> dict:
        return self._do(self._session(site), Action.wait())

    async def op_screenshot(self, site) -> dict:
        s = self._session(site)
        return {"screenshot_ref": self.pool.screenshot(s.id)}

    def _element_name(self, s, eid: str) -> str:
        e = s.env.observe().element(eid)
        if e is None:
            raise ConnectorError(f"element {eid} not found", 404)
        return e.name or ""

    async def op_fill(self, site, element, text) -> dict:
        s = self._session(site)
        return self._do(s, Action.type_text(element, text))

    async def op_click(self, site, element, approved_high_impact: Optional[bool] = None) -> dict:
        s = self._session(site)
        name = self._element_name(s, element)
        if HIGH_IMPACT.search(name) and not approved_high_impact:
            raise ConnectorError(f"control {name!r} looks high impact; needs tier-4 human approval", 403)
        return self._do(s, Action.click(element))

    async def op_upload_file(self, site, element, object_key) -> dict:
        s = self._session(site)
        data = self.pool.store.get(object_key)
        import tempfile
        from pathlib import Path

        p = Path(tempfile.mkdtemp()) / Path(object_key).name
        p.write_bytes(data)
        loc = f'[data-agentrs-id="{element}"]'
        s.env.run_in_browser(lambda page: page.locator(loc).first.set_input_files(str(p)))
        return self._obs(s)

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        raise ConnectorError("generic websites have no outcome API; use a site-specific verifier")

    def close(self) -> None:
        for s in self._sessions.values():
            self.pool.close(s.id)
        self._sessions.clear()
