"""Isolated browser workers (Playwright + Chromium).

* One Chromium **process and context per session**: sessions bound to different accounts never
  share cookies, storage, cache or service workers.
* Each session has an explicit task, agent, account, action budget, lifetime, network policy
  (egress allowlist + private-address blocking on every request the page makes) and artifact
  capture (screenshots on demand; Playwright traces when enabled).
* Persistent authentication: the context's storage state (cookies / local storage) is encrypted
  (Fernet) and written to object storage; it is never returned to the model.
* Pool limits: ``max_sessions`` concurrent sessions; CPU / memory limits are enforced on the
  browser-worker container (see ``deploy/``), not inside this process.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

from cryptography.fernet import Fernet

from agentrs.environments.web.playwright_env import PlaywrightEnv
from agentrs.harness.net import EgressDenied, EgressPolicy, check_url
from agentrs.harness.object_store import ObjectStore


class SessionError(PermissionError):
    pass


class IsolatedPlaywrightEnv(PlaywrightEnv):
    """PlaywrightEnv whose context is created from an (optional) storage state, whose every request is
    checked against the egress policy, and which can capture screenshots / traces."""

    def __init__(self, egress: EgressPolicy, storage_state: Optional[dict] = None, trace: bool = False, **kw):
        self.egress = egress
        self._storage_state = storage_state
        self._trace = trace
        super().__init__(**kw)

    def _request_allowed(self, url: str) -> bool:
        if url.startswith(("data:", "blob:", "about:")):
            return True
        try:
            check_url(url, self.egress)
        except EgressDenied:
            return False
        return super()._request_allowed(url)

    def _new_context(self) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception:  # noqa: BLE001
                pass
        w, h = self.viewport
        self._context = self._browser.new_context(viewport={"width": w, "height": h}, service_workers="block",
                                                  accept_downloads=True, storage_state=self._storage_state)
        self._context.set_default_timeout(self.action_timeout_ms)
        self._context.set_default_navigation_timeout(self.navigation_timeout_ms)
        self._context.route("**/*", self._route)
        if self._trace:
            self._context.tracing.start(screenshots=True, snapshots=True)
        self.page = self._context.new_page()
        self._inflight = 0
        self.page.on("request", self._on_request)
        self.page.on("requestfinished", self._on_request_done)
        self.page.on("requestfailed", self._on_request_done)
        self._context.on("page", lambda p: p.close() if p is not self.page else None)

    def storage_state(self) -> dict:
        return self._thread.call(lambda: self._context.storage_state())

    def screenshot(self) -> bytes:
        return self.run_in_browser(lambda page: page.screenshot(type="png"))

    def stop_trace(self, path: str) -> None:
        if self._trace:
            self._thread.call(lambda: self._context.tracing.stop(path=path))


@dataclass
class BrowserSession:
    id: str
    account_id: str
    agent_id: str
    task_id: str
    env: IsolatedPlaywrightEnv
    action_budget: int
    expires_at: float
    actions_used: int = 0
    artifacts: list[str] = field(default_factory=list)

    def use(self) -> None:
        if time.time() > self.expires_at:
            raise SessionError("browser session expired")
        if self.actions_used >= self.action_budget:
            raise SessionError("browser session action budget exhausted")
        self.actions_used += 1


class BrowserWorkerPool:
    def __init__(self, store: ObjectStore, key: bytes, egress: EgressPolicy, max_sessions: int = 4,
                 session_lifetime_s: float = 1800.0, default_budget: int = 50, headless: bool = True):
        self.store, self.fernet, self.egress = store, Fernet(key), egress
        self.max_sessions, self.lifetime, self.budget, self.headless = max_sessions, session_lifetime_s, default_budget, headless
        self.sessions: dict[str, BrowserSession] = {}
        self.revoked_accounts: set[str] = set()

    def _state_key(self, account_id: str) -> str:
        return f"browser-state/{account_id}.enc"

    def open(self, account_id: str, agent_id: str, task_id: str, start_url: str, allowed_prefixes: list[str],
             budget: Optional[int] = None, trace: bool = False) -> BrowserSession:
        if account_id in self.revoked_accounts:
            raise SessionError("account revoked")
        live = [s for s in self.sessions.values() if time.time() < s.expires_at]
        if len(live) >= self.max_sessions:
            raise SessionError("browser pool is at capacity")
        state = None
        if self.store.exists(self._state_key(account_id)):
            state = json.loads(self.fernet.decrypt(self.store.get(self._state_key(account_id))))
        env = IsolatedPlaywrightEnv(self.egress, storage_state=state, trace=trace, start_url=start_url,
                                    allowed_url_prefixes=allowed_prefixes, headless=self.headless)
        s = BrowserSession(id=uuid.uuid4().hex, account_id=account_id, agent_id=agent_id, task_id=task_id, env=env,
                           action_budget=budget or self.budget, expires_at=time.time() + self.lifetime)
        self.sessions[s.id] = s
        return s

    def save_state(self, session_id: str) -> str:
        s = self.sessions[session_id]
        st = s.env.storage_state()
        return self.store.put(self._state_key(s.account_id), self.fernet.encrypt(json.dumps(st).encode()))

    def screenshot(self, session_id: str) -> str:
        s = self.sessions[session_id]
        key = f"screenshots/{s.task_id}/{s.id}/{int(time.time() * 1000)}.png"
        ref = self.store.put(key, s.env.screenshot(), "image/png")
        s.artifacts.append(ref)
        return ref

    def close(self, session_id: str, save: bool = True) -> None:
        s = self.sessions.pop(session_id, None)
        if s is None:
            return
        try:
            if save and s.account_id not in self.revoked_accounts:
                st = s.env.storage_state()
                self.store.put(self._state_key(s.account_id), self.fernet.encrypt(json.dumps(st).encode()))
        finally:
            s.env.close()

    def revoke_account(self, account_id: str) -> int:
        """Terminate every session of the account and delete its stored browser state."""
        self.revoked_accounts.add(account_id)
        n = 0
        for sid in [k for k, v in self.sessions.items() if v.account_id == account_id]:
            self.close(sid, save=False)
            n += 1
        self.store.delete(self._state_key(account_id))
        return n

    def close_all(self) -> None:
        for sid in list(self.sessions):
            self.close(sid)

    def stats(self) -> dict[str, Any]:
        return {"sessions": len(self.sessions), "max_sessions": self.max_sessions,
                "hosts": sorted({urlsplit(s.env.start_url or "").hostname or "" for s in self.sessions.values()})}
