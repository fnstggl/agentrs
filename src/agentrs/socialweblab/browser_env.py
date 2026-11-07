"""SocialWebLab in a real browser: ``SocialWebLabBrowserEnv(PlaywrightEnv)``.

* The browser talks to a ``SocialWebLabServer`` (started by the env when given a lab). Only the
  server origin is allowlisted and ``/__control/`` is blocked at the network layer (inherited
  ``blocked_path_markers``); the env itself reads the lab object directly and never needs the
  control channel.
* ``reset(task)``: ``task.private = {"account", "password", "start_path": "/", "verifier": {...},
  "space"?}``. With ``login_mode="form"`` (default) the env opens ``/login?next=<start_path>``,
  fills the real login form and submits it; with ``login_mode="cookie"`` it obtains a session
  through ``lab.login`` and installs the ``swl_session`` cookie before the first navigation.
  Either way the agent's first observation is ``start_path``, logged in.
* Observation URLs are shown as path (+ query) on the lab origin, e.g. ``/item/12``.

Content macros (``Action.submit(operation, content)``; all other actions are the inherited
Playwright primitives):

``create_post``
    In the real browser: go to ``/new``; select the space - an explicit first content line
    ``space: <name>`` (removed from the content) wins, else ``task.private["space"]``; fill
    *Title* with the first (remaining) content line truncated to 120 characters and *Body* with
    the full (remaining) content; click *Publish*; wait for the navigation to ``/item/<id>`` (the
    page's JS publishes through ``fetch('/api/items')``). The new id is stored in
    ``self.last_created_item`` and returned as ``StepResult.info["created_item_id"]``; a
    validation error shown in the page's ``role=alert`` becomes the step error.
``post_reply``
    On an item page: choose reply type *Answer*, fill *Your reply* with the content, click *Post
    reply* and wait for the reloaded item page; ``info["created_response_id"]``.

Verification is environment-side only, from the lab state (never from the page):
``item_created {space, must_contain}``, ``reacted {item_id, reaction}``, ``replied {item_id,
must_contain?}``, ``edited {item_id, must_contain}``, ``help_within {within_s, item_id?}``
(``lab.helpful_answer_within`` on ``last_created_item``; ``success=False, evidence.pending=True``
until ``within_s`` of lab time has passed since the item was created). Only actions taken after
``reset`` count (event / item id floors recorded at reset).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import quote, urlsplit

from agentrs.actions.schema import Action, ActionType
from agentrs.environments.base import (
    EnvironmentError_,
    StepResult,
    TaskSpec,
    Verification,
)
from agentrs.environments.web.playwright_env import PlaywrightEnv, _first_line
from agentrs.socialweblab.engine import LabError, SocialWebLab
from agentrs.socialweblab.server import (
    CONTROL_PREFIX,
    SESSION_COOKIE,
    SocialWebLabServer,
)

_ITEM_PATH = re.compile(r"^/item/(\d+)$")
_SPACE_LINE = re.compile(r"^\s*space\s*[:=]\s*([A-Za-z0-9_\-]+)\s*$", re.IGNORECASE)
TITLE_MAX = 120


class SocialWebLabBrowserEnv(PlaywrightEnv):
    env_id = "socialweblab"
    version = "1.0"

    def __init__(self, server_or_lab: SocialWebLabServer | SocialWebLab, headless: bool = True,
                 login_mode: str = "form", report_focus: bool = False, **kw):
        if login_mode not in ("form", "cookie"):
            raise ValueError("login_mode must be 'form' or 'cookie'")
        if isinstance(server_or_lab, SocialWebLabServer):
            self.server, self._owns_server = server_or_lab, False
        elif isinstance(server_or_lab, SocialWebLab):
            self.server, self._owns_server = SocialWebLabServer(server_or_lab), True
        else:
            raise TypeError("SocialWebLabBrowserEnv needs a SocialWebLabServer or a SocialWebLab")
        self.server.start()
        self.lab: SocialWebLab = self.server.lab
        self.origin = self.server.url
        super().__init__(allowed_url_prefixes=(self.origin + "/",), headless=headless, report_focus=report_focus, **kw)
        self.login_mode = login_mode
        self.last_created_item: int | None = None
        self.last_created_response: int | None = None
        self._account: str | None = None
        self._password: str | None = None
        self._session_token: str | None = None
        self.uid: int | None = None
        self._item_floor = 0
        self._event_floor = 0
        self._macro_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ episode setup
    def _prepare_episode(self, task: TaskSpec) -> str:
        p = task.private
        try:
            self._account, self._password = str(p["account"]), str(p["password"])
        except KeyError as e:
            raise EnvironmentError_(f"socialweblab task needs private[{e.args[0]!r}]") from e
        start = str(p.get("start_path", "/"))
        if not start.startswith("/") or start.startswith("//") or CONTROL_PREFIX in start:
            raise EnvironmentError_(f"bad start_path {start!r}")
        self.server.start()
        self.allowed_url_prefixes = (self.origin + "/",)
        self.last_created_item = self.last_created_response = None
        self._item_floor = self._max_id("items")
        self._event_floor = self._max_id("events")
        self._start_path = start
        if self.login_mode == "cookie":
            try:
                self._session_token = self.lab.login(self._account, self._password)
            except LabError as e:
                raise EnvironmentError_(f"cannot log in test account {self._account!r}: {e}") from e
            self.uid = self.lab.user_for_token(self._session_token)
            return self.origin + start
        self._session_token = None
        return self.origin + "/login?next=" + quote(start, safe="")

    def _new_context(self) -> None:
        super()._new_context()
        if self.login_mode == "cookie" and self._session_token:
            self._install_cookie(self._session_token)

    def _install_cookie(self, token: str) -> None:
        self._context.add_cookies([{"name": SESSION_COOKIE, "value": token, "url": self.origin + "/",
                                    "httpOnly": True, "sameSite": "Lax"}])

    def _reset_impl(self, task: TaskSpec):
        obs = super()._reset_impl(task)
        if self.login_mode == "form":
            self._form_login()
            obs = self._observe_impl()
        if self._path() == "/login":
            raise EnvironmentError_("the agent is not logged in after reset")
        return obs

    def _form_login(self) -> None:
        from playwright.sync_api import Error as PlaywrightError

        page = self.page
        try:
            page.fill("#login-handle", self._account or "")
            page.fill("#login-password", self._password or "")
            with page.expect_navigation(wait_until="load", timeout=self.navigation_timeout_ms):
                page.click("#login-submit")
            self._settle()
        except PlaywrightError as e:
            raise EnvironmentError_(f"login form flow failed: {_first_line(e)}") from e
        tok = next((c["value"] for c in self._context.cookies(self.origin) if c["name"] == SESSION_COOKIE), None)
        self.uid = self.lab.user_for_token(tok)
        if self.uid is None or self._path() == "/login":
            raise EnvironmentError_(f"cannot log in test account {self._account!r} through the login form")
        self._session_token = tok
        self._last_good_url = page.url

    # ------------------------------------------------------------------ helpers
    def _max_id(self, table: str) -> int:
        return int(self.lab._q(f"SELECT COALESCE(MAX(id), 0) AS m FROM {table}")[0]["m"])

    def _path(self) -> str:
        return urlsplit(self.page.url).path if self.page is not None else ""

    def _display_url(self, url: str) -> str:
        if url.startswith(self.origin + "/"):
            s = urlsplit(url)
            return s.path + (f"?{s.query}" if s.query else "")
        return url

    def _alert_text(self) -> str | None:
        for sel in ("#new-error:not([hidden])", "#page-error", "[role=alert]:not([hidden])"):
            loc = self.page.locator(sel)
            if loc.count():
                t = (loc.first.inner_text() or "").strip()
                if t:
                    return t
        return None

    # ------------------------------------------------------------------ actions
    def _step_impl(self, action: Action) -> StepResult:
        self._macro_info = {}
        res = super()._step_impl(action)
        if self._macro_info:
            res.info.update(self._macro_info)
        return res

    def _apply_impl(self, a: Action) -> str | None:
        if a.type != ActionType.SUBMIT:
            return super()._apply_impl(a)
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            if a.name == "create_post":
                return self._macro_create_post(a.text or "")
            if a.name == "post_reply":
                return self._macro_post_reply(a.text or "")
            return f"unsupported content operation {a.name!r} (supported: create_post, post_reply)"
        except PlaywrightTimeoutError:
            self._safe_settle()
            return self._alert_text() or f"timed out performing submit({a.name})"
        except PlaywrightError as e:
            self._safe_settle()
            return self._check_error_page() or f"submit({a.name}) failed: {_first_line(e)}"

    def _macro_create_post(self, content: str) -> str | None:
        from playwright.sync_api import Error as PlaywrightError

        lines = content.strip().splitlines()
        space = None
        if lines and _SPACE_LINE.match(lines[0]):
            space = _SPACE_LINE.match(lines[0]).group(1)
            lines = lines[1:]
        text = "\n".join(lines).strip()
        space = space or (self.spec.private.get("space") if self.spec is not None else None)
        if not text:
            return "create_post needs non-empty content"
        if not space:
            return "create_post: no space given (first line 'space: <name>' or task.private['space'])"
        title = text.splitlines()[0].strip()[:TITLE_MAX]
        page = self.page
        page.goto(self.origin + "/new", wait_until="load", timeout=self.navigation_timeout_ms)
        self._settle()
        options = page.eval_on_selector_all("#new-space option", "els => els.map(e => e.value)")
        if space not in options:
            return f"space {space!r} is not available"
        page.select_option("#new-space", value=space, timeout=self.action_timeout_ms)
        page.fill("#new-title", title, timeout=self.action_timeout_ms)
        page.fill("#new-body", text, timeout=self.action_timeout_ms)
        page.click("#publish", timeout=self.action_timeout_ms)
        deadline = time.monotonic() + self.navigation_timeout_ms / 1000
        while not _ITEM_PATH.match(self._path()):  # until the page navigates or shows its alert
            if time.monotonic() > deadline:
                return "publish failed: " + (self._alert_text() or "no confirmation from the page")
            try:
                if page.locator("#new-error:not([hidden])").count():
                    return "publish failed: " + (self._alert_text() or "unknown error")
            except PlaywrightError:  # navigation in progress
                pass
            page.wait_for_timeout(50)
        self._settle()
        iid = int(_ITEM_PATH.match(self._path()).group(1))
        self.last_created_item = iid
        self._macro_info = {"created_item_id": iid}
        return self._check_error_page()

    def _macro_post_reply(self, content: str) -> str | None:
        m = _ITEM_PATH.match(self._path())
        if not m:
            return "post_reply must be used on an item page (/item/<id>)"
        if not content.strip():
            return "post_reply needs non-empty content"
        page = self.page
        if page.locator("#reply-body").count() == 0:
            return "this item does not accept replies"
        page.select_option("#reply-kind", value="answer", timeout=self.action_timeout_ms)
        page.fill("#reply-body", content, timeout=self.action_timeout_ms)
        with page.expect_navigation(wait_until="load", timeout=self.navigation_timeout_ms):
            page.click("#reply-submit", timeout=self.action_timeout_ms)
        self._settle()
        frag = re.search(r"#response-(\d+)$", page.url)
        if not frag:
            return "reply failed: " + (self._alert_text() or "no confirmation from the page")
        self.last_created_response = int(frag.group(1))
        self._macro_info = {"created_response_id": self.last_created_response, "item_id": int(m.group(1))}
        return self._check_error_page()

    # ------------------------------------------------------------------ verification (lab state only)
    def _my_items(self, space: str | None = None) -> list[dict]:
        where, args = "author=? AND id>?", [self.uid, self._item_floor]
        if space:
            where += " AND space=?"
            args.append(space)
        return self.lab.items(where, args)

    def _my_events(self, item_id: int, types: tuple[str, ...]) -> list[dict]:
        return [e for e in self.lab.all_events(item_id)
                if e["id"] > self._event_floor and e["actor"] == self.uid and e["type"] in types]

    def _verify_impl(self) -> Verification:
        if self.spec is None:
            raise EnvironmentError_("reset() first")
        v = dict(self.spec.private.get("verifier") or {})
        kind = v.get("kind")
        vid = f"socialweblab/{kind}/v1"
        if self.uid is None:
            return Verification(success=False, verifier_id=vid, evidence={"reason": "not logged in"})
        lab = self.lab

        def has_all(text: str, words) -> list[str]:
            low = text.lower()
            return [w for w in (words or []) if w.lower() not in low]

        if kind == "item_created":
            best = None
            for it in self._my_items(v.get("space")):
                missing = has_all(it["title"] + " " + it["body"], v.get("must_contain"))
                if it["state"] != "deleted" and not missing:
                    best = it
                    break
            created = self._my_items(v.get("space"))
            if best is not None and self.last_created_item is None:
                self.last_created_item = best["id"]
            return Verification(success=best is not None, score=float(best is not None),
                                progress=1.0 if best else (0.5 if created else 0.0), verifier_id=vid,
                                evidence={"item_id": best["id"] if best else None,
                                          "created": [it["id"] for it in created]})
        if kind == "reacted":
            want = v.get("reaction", "up")
            evs = self._my_events(int(v["item_id"]), ("reaction_up", "reaction_down"))
            ok = any(e["type"] == f"reaction_{want}" for e in evs)
            return Verification(success=ok, score=float(ok), progress=float(ok), verifier_id=vid,
                                evidence={"reactions": [e["type"] for e in evs]})
        if kind == "replied":
            iid = int(v["item_id"])
            evs = self._my_events(iid, ("response_created",))
            hits = [e for e in evs if not has_all(_payload(e).get("body", ""), v.get("must_contain"))]
            return Verification(success=bool(hits), score=float(bool(hits)), progress=1.0 if hits else 0.5 * bool(evs),
                                verifier_id=vid, evidence={"responses": [e["response_id"] for e in evs]})
        if kind == "edited":
            iid = int(v["item_id"])
            evs = self._my_events(iid, ("item_edited",))
            rows = lab.items("id=?", (iid,))
            body = (rows[0]["title"] + " " + rows[0]["body"]) if rows else ""
            missing = has_all(body, v.get("must_contain"))
            ok = bool(evs) and not missing and bool(rows) and rows[0]["state"] != "deleted"
            return Verification(success=ok, score=float(ok), progress=1.0 if ok else 0.5 * bool(evs), verifier_id=vid,
                                evidence={"edits": len(evs), "missing": missing})
        if kind == "help_within":
            within = float(v.get("within_s", 86400))
            iid = v.get("item_id") or self.last_created_item
            if iid is None:
                mine = self._my_items()
                iid = mine[-1]["id"] if mine else None
            if iid is None:
                return Verification(success=False, verifier_id=vid, evidence={"reason": "no item created"})
            rows = lab.items("id=?", (int(iid),))
            age = lab.now() - rows[0]["created_at"]
            res = lab.helpful_answer_within(int(iid), within)
            ev = {"item_id": int(iid), "age_s": age, **res}
            if age < within:
                return Verification(success=False, verifier_id=vid, evidence={**ev, "pending": True})
            return Verification(success=res["success"], score=float(res["success"]), progress=float(res["success"]),
                                verifier_id=vid, evidence={**ev, "pending": False})
        return Verification(success=False, verifier_id="socialweblab/unverified",
                            evidence={"reason": f"unknown verifier kind {kind!r}"})

    # ------------------------------------------------------------------ observer access, snapshots
    def get_observable_events(self, since_id: int = 0, limit: int = 200) -> list[dict]:
        """Public events of the lab (what any user of the application could see)."""
        return self.lab.public_events(since_id, limit)

    def snapshot(self, path) -> str:
        return self.lab.snapshot(path)

    def restore(self, path) -> None:
        """Restore the lab database. The restored DB has the sessions of the snapshot, so the env
        logs the test account in again (new session cookie) and reloads the page - the browser
        would otherwise show pre-restore content and might hold an unknown session."""
        self._thread.call(self._restore_impl, path)

    def _restore_impl(self, path) -> None:
        self.lab.restore(path)
        if self.page is None:
            return
        try:
            tok = self.lab.login(self._account or "", self._password or "")
            self._install_cookie(tok)
            self._session_token, self.uid = tok, self.lab.user_for_token(tok)
        except LabError:  # the account did not exist at snapshot time
            self._context.clear_cookies()
            self._session_token = self.uid = None
        self._item_floor = min(self._item_floor, self._max_id("items"))
        self._event_floor = min(self._event_floor, self._max_id("events"))
        self.page.reload(wait_until="load")
        self._settle()

    def _close_impl(self) -> None:
        super()._close_impl()
        if self._owns_server:
            self.server.stop()


def _payload(e: dict) -> dict:
    try:
        return json.loads(e.get("payload") or "{}")
    except ValueError:
        return {}


__all__ = ["SocialWebLabBrowserEnv"]
