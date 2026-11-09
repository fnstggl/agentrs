"""Shared helpers for the provider connectors (token handling, provider-specific policy, outcome payloads).

``HarnessConnector`` adds to ``PlatformConnector``:

* credential helpers: ``_secret(name)`` reads a credential (refreshed values first) and
  ``_remember(name, value)`` persists a refreshed token through ``ctx.credentials.update`` (kept
  in memory only if the account's credentials are not updatable);
* ``_call`` / ``_req``: requests through ``ctx.http`` only, rate-limit header capture and
  ``ProviderHTTPError`` carrying the parsed provider error body;
* ``provider_policy(action, recent)``: provider-specific rules the tool gateway calls before
  execution (duplicate text, bulk likes/reposts, unsolicited replies, plus ``_extra_policy``);
* ``outcome_payload``: the temporal ledger payload format.

``recent`` passed to ``provider_policy`` is a list of dicts describing this account's recent
actions; recognised keys: ``provider``, ``operation``, ``params``, ``content`` (optional, the
generated text), ``status`` (entries with denied/failed/rejected/rate_limited are ignored) and a time
in ``executed_at`` / ``proposed_at`` / ``t`` (epoch seconds). In addition, every connector instance
records the texts it published, the likes it made and the objects that addressed the account in a
per-(provider, account) process-wide memory, so rules hold even when ``recent`` is incomplete.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from agentrs.harness.connectors.base import ConnectorError, PlatformConnector
from agentrs.harness.types import ActionRequest, ActionResult

log = logging.getLogger("agentrs.connectors")

_CURRENT_ACTION: contextvars.ContextVar[Optional[ActionRequest]] = contextvars.ContextVar("agentrs_action", default=None)
_IGNORED_STATUS = {"denied", "failed", "rejected", "rate_limited"}


class ProviderHTTPError(ConnectorError):
    """HTTP error from a provider, with the parsed error body (untrusted)."""

    def __init__(self, msg: str, status: int, body: Any = None, code: str = "", retry_after: Optional[float] = None):
        super().__init__(msg, status, retry_after)
        self.body = body
        self.code = code


class _AccountMemory:
    """Process-wide memory per (provider, account): published texts, likes/reposts, addressed objects."""

    def __init__(self):
        self.texts: dict[str, float] = {}
        self.engagements: list[tuple[str, float]] = []  # (operation, time)
        self.addressed: set[str] = set()
        self.own: set[str] = set()
        self.facts: dict[str, Any] = {}  # e.g. verified account type / bot flag from validate_credentials


_MEMORY: dict[tuple[str, str], _AccountMemory] = {}
_MEMORY_LOCK = threading.Lock()


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def parse_time(s: Any) -> Optional[float]:
    """Parse provider timestamps (ISO 8601 with Z / +00:00 / +0000, epoch numbers) to epoch seconds."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if re.fullmatch(r"\d+(\.\d+)?", txt):
        return float(txt)
    txt = txt.replace("Z", "+00:00")
    m = re.match(r"(.*[T ]\d\d:\d\d:\d\d(?:\.\d+)?)([+-]\d\d)(\d\d)$", txt)
    if m:
        txt = f"{m.group(1)}{m.group(2)}:{m.group(3)}"
    m = re.match(r"(.*\.\d{6})\d+(.*)$", txt)  # more than microsecond precision
    if m:
        txt = m.group(1) + m.group(2)
    try:
        return datetime.fromisoformat(txt).timestamp()
    except ValueError:
        return None


def outcome_payload(counts: dict, state: str, occurred: dict, occurred_at: Optional[dict] = None,
                    continuous: Optional[dict] = None, raw: Optional[dict] = None) -> dict[str, Any]:
    return {"counts": counts, "state": state, "occurred": occurred,
            "occurred_at": {k: v for k, v in (occurred_at or {}).items() if v is not None},
            "continuous": continuous or {}, "raw": raw or {}}


def strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html or "")
    text = re.sub(r"</p>\s*<p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    import html as _h

    return _h.unescape(text).strip()


class HarnessConnector(PlatformConnector):
    """Base for the provider connectors in this package."""

    text_ops: frozenset[str] = frozenset()  # operations that publish text (duplicate-content rule)
    like_ops: frozenset[str] = frozenset()
    repost_ops: frozenset[str] = frozenset()
    reply_ops: frozenset[str] = frozenset()  # operations that reply to someone else's object
    reply_target_param: str = ""
    engage_target_param: str = ""  # object parameter of like / repost operations
    default_engagements_per_hour = 5

    def __init__(self, ctx):
        super().__init__(ctx)
        self._tok: dict[str, str] = {}

    # ------------------------------------------------------------------ account memory
    def _account_key(self) -> str:
        a = self.ctx.account or {}
        for k in ("id", "account_id", "handle", "username", "app_id", "ig_user_id"):
            if a.get(k):
                return str(a[k])
        return "default"

    @property
    def memory(self) -> _AccountMemory:
        key = (self.provider, self._account_key())
        with _MEMORY_LOCK:
            return _MEMORY.setdefault(key, _AccountMemory())

    def note_addressed(self, *object_ids: Optional[str]) -> None:
        self.memory.addressed.update(o for o in object_ids if o)

    # ------------------------------------------------------------------ execution
    async def execute(self, action: ActionRequest) -> ActionResult:
        tok = _CURRENT_ACTION.set(action)
        try:
            res = await super().execute(action)
        finally:
            _CURRENT_ACTION.reset(tok)
        if res.status == "executed":
            spec = self.operations[action.operation]
            mem, now = self.memory, time.time()
            if action.operation in self.text_ops and spec.content_param:
                mem.texts[normalize_text(action.params.get(spec.content_param))] = now
            if action.operation in self.like_ops | self.repost_ops:
                mem.engagements.append((action.operation, now))
            if res.provider_object_id:
                mem.own.add(res.provider_object_id)
        return res

    def _idempotency_key(self, *parts: Any) -> str:
        a = _CURRENT_ACTION.get()
        if a is not None:
            return a.idempotency_key()
        return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:32]

    # ------------------------------------------------------------------ credentials
    def _secret(self, name: str, required: bool = True) -> Optional[str]:
        if name in self._tok:
            return self._tok[name]
        if self.ctx.credentials.has(name):
            return self.ctx.credentials.get(name)
        if required:
            raise ConnectorError(f"{self.provider}: credential {name!r} is not connected for this account", 401)
        return None

    def _remember(self, name: str, value: str) -> None:
        self._tok[name] = value
        try:
            self.ctx.credentials.update(name, value)
        except ConnectorError:
            log.info("%s: credential %s kept in memory only (account credentials are not updatable)", self.provider, name)

    # ------------------------------------------------------------------ HTTP
    def _note_rate_limits(self, headers) -> None:
        for k, v in headers.items():
            lk = k.lower()
            if ("ratelimit" in lk or "rate-limit" in lk or lk == "retry-after" or lk in ("x-app-usage",
                                                                                       "x-business-use-case-usage")):
                self.last_rate_limits[lk] = v

    async def _call(self, method: str, path: str, **kw) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        kw.setdefault("headers", {})
        r = await self.ctx.http.request(method, url, **kw)
        self._note_rate_limits(r.headers)
        return r

    def _error(self, method: str, path: str, r: httpx.Response) -> ProviderHTTPError:
        try:
            body = r.json()
        except ValueError:
            body = {"text": r.text[:300]}
        code = ""
        if isinstance(body, dict):
            e = body.get("error")
            code = str(e.get("code", "")) if isinstance(e, dict) else str(e or body.get("code") or body.get("title") or "")
        ra = r.headers.get("retry-after")
        retry = float(ra) if ra and re.fullmatch(r"\d+(\.\d+)?", ra) else None
        status = r.status_code
        msg = f"{self.provider} {method} {path.split('?')[0]}: HTTP {status}: {str(body)[:300]}"
        return ProviderHTTPError(msg, 429 if status == 429 else status, body, code, retry)

    async def _req(self, method: str, path: str, **kw) -> Any:
        r = await self._call(method, path, **kw)
        if r.status_code >= 400:
            raise self._error(method, path, r)
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"text": r.text}

    # ------------------------------------------------------------------ provider policy
    def provider_policy(self, action: ActionRequest, recent: list[dict]) -> tuple[bool, str]:
        """Provider-specific rules, called by the gateway before execution. (allowed, reason)."""
        spec = self.operations.get(action.operation)
        if spec is None:
            return False, f"{self.provider} does not support {action.operation!r}"
        entries = self._recent(recent)
        op, params = action.operation, action.params
        if op in self.text_ops and spec.content_param:
            text = normalize_text(params.get(spec.content_param))
            if not text:
                return False, "empty content"
            seen = text in self.memory.texts or any(e["content"] == text for e in entries if e["operation"] in self.text_ops)
            if seen:
                return False, "identical text was already posted by this account (duplicate content)"
        if op in self.like_ops | self.repost_ops:
            kind = self.like_ops if op in self.like_ops else self.repost_ops
            limit = self._engagement_limit(op)
            now = time.time()
            n_recent = sum(1 for e in entries if e["operation"] in kind and e["t"] is not None and now - e["t"] < 3600)
            n_mem = sum(1 for o, t in self.memory.engagements if o in kind and now - t < 3600)
            if max(n_recent, n_mem) >= limit:
                return False, f"bulk engagement refused: {max(n_recent, n_mem)} {op} actions in the last hour (limit {limit})"
            target = params.get(self.engage_target_param)
            if target is not None and any(e["operation"] == op and str(e["params"].get(self.engage_target_param)) == str(target)
                                          for e in entries):
                return False, f"already performed {op} on this object"
        if op in self.reply_ops and not params.get("approved_reply"):
            target = str(params.get(self.reply_target_param, ""))
            addressed = self.memory.addressed | self.memory.own | set(map(str, self.ctx.account.get("addressed_ids", [])))
            if target not in addressed:
                return False, ("reply refused: the target did not mention or reply to this account "
                               "(set approved_reply after human review to override)")
        return self._extra_policy(action, entries)

    def _extra_policy(self, action: ActionRequest, entries: list[dict]) -> tuple[bool, str]:
        return True, ""

    def _engagement_limit(self, op: str) -> int:
        acct = self.ctx.account or {}
        extra = self.ctx.config.limits.model_extra or {}
        for k in (f"max_{op}s_per_hour", f"{op}s_per_hour", "engagements_per_hour"):
            if k in acct:
                return int(acct[k])
            if k in extra:
                return int(extra[k])
        return self.default_engagements_per_hour

    def _recent(self, recent: list[dict]) -> list[dict]:
        out = []
        for d in recent or []:
            if not isinstance(d, dict):
                continue
            if d.get("provider") not in (None, self.provider) or d.get("status") in _IGNORED_STATUS:
                continue
            op = d.get("operation", "")
            params = d.get("params") or {}
            content = d.get("content")
            spec = self.operations.get(op)
            if content is None and spec and spec.content_param:
                content = params.get(spec.content_param)
            t = next((d[k] for k in ("executed_at", "proposed_at", "t", "time") if d.get(k) is not None), None)
            out.append({"operation": op, "params": params, "content": normalize_text(content) if content else None,
                        "t": float(t) if t is not None else None})
        return out
