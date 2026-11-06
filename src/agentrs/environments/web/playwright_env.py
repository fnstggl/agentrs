"""Real-browser environments driven by Playwright's sync API.

``PlaywrightEnv``
    Generic environment for any web page. Observations come from the generic
    DOM extractor in ``dom_extract.js`` (``data-agentrs-id`` element ids, roles,
    accessible names, values, states; invisible elements skipped). Typed
    ``Action`` objects are executed by an allowlisted dispatcher that only calls
    fixed Playwright operations (click / fill / select_option / scrollBy / goto);
    model output is never evaluated as code - it only ever appears as a
    validated element id, a fill/select string or a navigation URL checked
    against the allowlist. Every network request outside the allowlist is
    aborted with ``context.route``. Stale ids, timeouts and navigation errors
    become observable ``last_action_error`` strings with ``action_valid=False``.

``LocalWebEnv``
    MiniWeb task families served as real HTML by ``LocalWebServer``. Reset /
    step / termination / reward semantics are those of ``MiniWebEnv``; the
    outcome is verified server-side through the token-protected control
    channel, which the browser (and therefore the agent) cannot reach.

Threading: Playwright's sync API objects are bound to the thread that created
them, while callers (e.g. ``GuardedEnvironment`` step timeouts) may call from
other threads. All browser work therefore runs on one dedicated, process-wide
``PlaywrightThread``; public methods marshal their calls onto it.

Browser discovery: Playwright pins a browser revision; if that build is not
installed (e.g. playwright 1.64 expects chromium-1248 but only 1243 exists),
``launch_chromium`` falls back to other installed builds under
``PLAYWRIGHT_BROWSERS_PATH`` (headless shell first when headless) and system
Chrome/Chromium. ``AGENTRS_CHROMIUM=/path/to/chrome`` forces a binary.
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import queue
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urljoin, urlsplit

from agentrs.actions.schema import MAX_TEXT_LEN, Action, ActionType
from agentrs.environments.base import (
    ElementState,
    Environment,
    EnvironmentError_,
    Observation,
    PublicTask,
    StepResult,
    TaskSpec,
    Verification,
)
from agentrs.environments.miniweb.core import (
    CLICKABLE_ROLES,
    EDITABLE_ROLES,
    _answer_proposals,
    goal_text_proposals,
)

DOM_EXTRACT_JS = Path(__file__).with_name("dom_extract.js").read_text(encoding="utf-8")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,31}$")
_ELEMENT_ACTIONS = (ActionType.CLICK, ActionType.TYPE, ActionType.CLEAR, ActionType.SELECT, ActionType.READ)
_SCROLL_JS = "(d) => window.scrollBy(0, d * Math.max(1, Math.round(window.innerHeight / 2)))"


class BrowserUnavailable(EnvironmentError_):
    """Chromium could not be launched (missing binary / missing system libraries)."""


# ====================================================================== browser discovery
def _browsers_root() -> Path:
    return Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or Path.home() / ".cache" / "ms-playwright")


def _revision(path: str) -> int:
    m = re.search(r"-(\d+)[/\\]", path)
    return int(m.group(1)) if m else -1


def chromium_candidates(headless: bool = True) -> list[str]:
    """Installed Chromium executables, best first (headless shell first when headless)."""
    root = _browsers_root()
    shells = sorted(glob.glob(str(root / "chromium_headless_shell-*" / "chrome-headless-shell-*" / "chrome-headless-shell")),
                    key=_revision, reverse=True)
    fulls = sorted(glob.glob(str(root / "chromium-*" / "chrome-linux*" / "chrome")), key=_revision, reverse=True)
    out = (shells + fulls) if headless else fulls  # the headless shell cannot open windows
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        w = shutil.which(name)
        if w:
            out.append(w)
    seen, res = set(), []
    for p in out:
        if p not in seen and os.access(p, os.X_OK):
            seen.add(p)
            res.append(p)
    return res


_RESOLVED_EXE: dict[bool, Optional[str]] = {}  # headless -> executable that worked (None = Playwright default)


def launch_chromium(pw, headless: bool = True, executable_path: Optional[str] = None, **kw):
    """Launch Chromium robustly. Order: explicit path / ``AGENTRS_CHROMIUM``; the
    build Playwright expects; other installed builds; system Chrome."""
    from playwright.sync_api import Error as PlaywrightError

    explicit = executable_path or os.environ.get("AGENTRS_CHROMIUM")
    if explicit:
        try:
            return pw.chromium.launch(headless=headless, executable_path=explicit, **kw)
        except PlaywrightError as e:
            raise BrowserUnavailable(f"cannot launch Chromium at {explicit!r}: {_first_line(e)}") from e
    errors: list[str] = []
    order: list[Optional[str]] = []
    if headless in _RESOLVED_EXE:
        order.append(_RESOLVED_EXE[headless])
    order += [None] + chromium_candidates(headless)
    tried: set = set()
    for exe in order:
        if exe in tried:
            continue
        tried.add(exe)
        try:
            b = pw.chromium.launch(headless=headless, **({"executable_path": exe} if exe else {}), **kw)
            _RESOLVED_EXE[headless] = exe
            return b
        except PlaywrightError as e:
            errors.append(f"{exe or 'playwright-default'}: {_first_line(e)}")
    raise BrowserUnavailable("could not launch Chromium (set AGENTRS_CHROMIUM). Tried: " + " | ".join(errors))


def resolved_chromium_executable(headless: bool = True) -> Optional[str]:
    """The executable used by the last successful launch (None = Playwright default)."""
    return _RESOLVED_EXE.get(headless)


def _first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


# ====================================================================== browser thread
class PlaywrightThread:
    """A single daemon thread owning the process-wide sync ``Playwright`` instance.

    Sync Playwright objects may only be used from the thread that created them.
    ``call`` runs a function on this thread and returns its result (re-entrant:
    calls made from the thread itself run inline)."""

    _instance: Optional["PlaywrightThread"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "PlaywrightThread":
        with cls._instance_lock:
            if cls._instance is None or not cls._instance._thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._pw = None
        self._thread = threading.Thread(target=self._loop, name="agentrs-playwright", daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, args, kw, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args, **kw))
            except BaseException as e:  # propagate to the caller
                fut.set_exception(e)

    def in_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def call(self, fn: Callable, *args, timeout: Optional[float] = None, **kw):
        if self.in_thread():
            return fn(*args, **kw)
        fut: Future = Future()
        self._q.put((fn, args, kw, fut))
        return fut.result(timeout)

    @property
    def playwright(self):
        """The sync Playwright instance (only valid on this thread)."""
        assert self.in_thread(), "Playwright objects must be used on the browser thread"
        if self._pw is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
        return self._pw

    def shutdown(self, timeout: float = 10.0) -> None:
        def _stop():
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
                self._pw = None

        if self._thread.is_alive():
            try:
                self.call(_stop, timeout=timeout)
            except Exception:
                pass
            self._q.put(None)


# ====================================================================== action mask
def candidate_actions(obs: Observation, goal: str, loading: Optional[bool] = None) -> list[Action]:
    """Action mask analogous to ``MiniWebEnv.available_actions``: valid primitive
    actions on visible enabled elements, goal-span text proposals, scroll when
    the page has more content, ``wait`` while loading, ``finish`` proposals."""
    acts: list[Action] = []
    proposals = goal_text_proposals(goal)

    def add(fn, *args):
        try:
            acts.append(fn(*args))
        except ValueError:  # e.g. an option label longer than MAX_TEXT_LEN
            pass

    for e in obs.elements:
        if e.disabled:
            continue
        if e.role in CLICKABLE_ROLES:
            add(Action.click, e.eid)
        if e.role in EDITABLE_ROLES:
            for p in proposals:
                add(Action.type_text, e.eid, p)
            if e.value:
                add(Action.clear, e.eid)
        if e.role == "combobox" and e.options:
            for opt in e.options:
                if len(opt) <= MAX_TEXT_LEN:
                    add(Action.select, e.eid, opt)
    if obs.scroll_info:
        if "below" in obs.scroll_info:
            acts.append(Action.scroll("down"))
        if "above" in obs.scroll_info:
            acts.append(Action.scroll("up"))
    if loading is None:
        loading = any(e.role == "status" for e in obs.elements)
    if loading:
        acts.append(Action.wait())
    acts.append(Action.finish())
    for p in _answer_proposals(obs):
        add(Action.finish, p)
    seen, out = set(), []
    for a in acts:
        s = a.to_str()
        if s not in seen:
            seen.add(s)
            out.append(a)
    return out


# ====================================================================== generic env
class PlaywrightEnv(Environment):
    """Generic Playwright-backed environment for arbitrary (allowlisted) pages.

    Without a ``verifier`` the outcome is unverified (``success=False``); use a
    subclass (``LocalWebEnv``) or pass ``verifier(page, answer) -> Verification``
    (runs on the browser thread) for verified tasks.
    """

    env_id = "playwright"
    version = "1.0"
    blocked_path_markers: tuple[str, ...] = ("/__control/",)

    def __init__(
        self,
        start_url: Optional[str] = None,
        allowed_url_prefixes: Sequence[str] = (),
        headless: bool = True,
        viewport: tuple[int, int] = (1024, 720),
        action_timeout_ms: int = 3000,
        navigation_timeout_ms: int = 15000,
        settle_timeout_ms: int = 5000,
        wait_ms: int = 500,
        id_prefix: str = "a",
        start_id_range: tuple[int, int] = (1, 400),
        report_focus: bool = True,
        viewport_only: bool = True,
        max_elements: int = 250,
        executable_path: Optional[str] = None,
        launch_args: Sequence[str] = (),
        verifier: Optional[Callable[[Any, Optional[str]], Verification]] = None,
    ):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,7}", id_prefix):
            raise ValueError("id_prefix must be a short identifier")
        self.start_url = start_url
        self._configured_prefixes = tuple(_norm_prefix(p) for p in allowed_url_prefixes)
        self.allowed_url_prefixes: tuple[str, ...] = self._configured_prefixes  # effective, per episode
        self.headless = headless
        self.viewport = viewport
        self.action_timeout_ms = action_timeout_ms
        self.navigation_timeout_ms = navigation_timeout_ms
        self.settle_timeout_ms = settle_timeout_ms
        self.wait_ms = wait_ms
        self.id_prefix = id_prefix
        self.start_id_range = start_id_range
        self.report_focus = report_focus
        self.viewport_only = viewport_only
        self.max_elements = max_elements
        self.executable_path = executable_path
        self.launch_args = list(launch_args)
        self.verifier = verifier

        self._thread = PlaywrightThread.get()
        self._browser = None
        self._context = None
        self.page = None  # only usable on the browser thread (see ``run_in_browser``)
        self.spec: Optional[TaskSpec] = None
        self.blocked_requests: list[str] = []
        self._inflight = 0
        self._last_good_url: Optional[str] = None
        self._init_episode(None)

    # ------------------------------------------------------------------ public API (any thread)
    def start(self) -> "PlaywrightEnv":
        """Launch the browser eagerly (raises ``BrowserUnavailable`` if impossible)."""
        self._thread.call(self._ensure_browser)
        return self

    def reset(self, task: TaskSpec) -> Observation:
        return self._thread.call(self._reset_impl, task)

    def observe(self) -> Observation:
        return self._thread.call(self._observe_impl)

    def available_actions(self) -> list[Action]:
        return self._thread.call(self._available_actions_impl)

    def step(self, action: Action) -> StepResult:
        return self._thread.call(self._step_impl, action)

    def verify_outcome(self) -> Verification:
        return self._thread.call(self._verify_impl)

    def get_task_metadata(self) -> PublicTask:
        assert self.spec is not None, "reset() first"
        return self.spec.public()

    def close(self) -> None:
        self._thread.call(self._close_impl)

    def run_in_browser(self, fn: Callable[[Any], Any]) -> Any:
        """Run ``fn(page)`` on the browser thread (tests, recorders, debugging)."""
        return self._thread.call(lambda: fn(self.page))

    # ------------------------------------------------------------------ hooks for subclasses
    def _prepare_episode(self, task: TaskSpec) -> str:
        """Return the start URL for ``task`` (and set up allowlists)."""
        url = self.start_url or task.private.get("start_url")
        if not url:
            raise EnvironmentError_("PlaywrightEnv needs start_url (constructor or task.private['start_url'])")
        p = urlsplit(url)  # default allowlist: the start URL's origin
        self.allowed_url_prefixes = self._configured_prefixes or (f"{p.scheme}://{p.netloc}/",)
        return url

    def _display_url(self, url: str) -> str:
        return url

    def _resolve_navigation(self, text: str) -> tuple[Optional[str], Optional[str]]:
        base = self.page.url if self.page is not None and self.page.url.startswith("http") else (self.start_url or "")
        url = urljoin(base, text.strip())
        if not self._request_allowed(url) or not url.startswith(("http://", "https://")):
            return None, f"navigation to {text!r} is not permitted"
        return url, None

    def _finish_impl(self, answer: Optional[str]) -> None:
        pass

    def _wait_impl(self) -> None:
        self.page.wait_for_timeout(self.wait_ms)

    def _after_action(self, action: Action) -> None:
        pass

    def _is_terminal(self, action: Action) -> bool:
        return action.type == ActionType.FINISH

    def _loading_hint(self) -> Optional[bool]:
        return None

    def _verify_impl(self) -> Verification:
        if self.verifier is not None:
            return self.verifier(self.page, self._answer)
        return Verification(success=False, score=0.0, verifier_id="playwright/unverified",
                            evidence={"reason": "no verifier configured", "answer": self._answer})

    def _settle(self) -> None:
        """Wait for the page to load and the network to go quiet (bounded)."""
        from playwright.sync_api import Error as PlaywrightError

        deadline = time.monotonic() + self.settle_timeout_ms / 1000
        try:
            self.page.wait_for_load_state("load", timeout=self.settle_timeout_ms)
        except PlaywrightError:
            pass
        quiet = 0
        while time.monotonic() < deadline and quiet < 2:
            quiet = quiet + 1 if self._inflight <= 0 else 0
            self.page.wait_for_timeout(25)

    # ------------------------------------------------------------------ browser-thread internals
    def _init_episode(self, task: Optional[TaskSpec]) -> None:
        seed = task.seed if task is not None else 0
        self._rng = random.Random(seed * 7919 + 17)  # same id-offset scheme as MiniWebEnv
        self._next_id = self._rng.randint(*self.start_id_range) + 1
        self._steps = 0
        self._done = False
        self._last_error: Optional[str] = None
        self._read_text: Optional[str] = None
        self._answer: Optional[str] = None
        self.blocked_requests = []
        self._blocked_nav = False  # a main-frame navigation was aborted by the allowlist

    def _ensure_browser(self) -> None:
        if self._browser is not None and self._browser.is_connected():
            return
        self._browser = launch_chromium(self._thread.playwright, headless=self.headless,
                                        executable_path=self.executable_path, args=self.launch_args)

    def _new_context(self) -> None:
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
        w, h = self.viewport
        self._context = self._browser.new_context(viewport={"width": w, "height": h}, service_workers="block",
                                                  accept_downloads=False)
        self._context.set_default_timeout(self.action_timeout_ms)
        self._context.set_default_navigation_timeout(self.navigation_timeout_ms)
        self._context.route("**/*", self._route)
        self.page = self._context.new_page()
        self._inflight = 0
        self.page.on("request", self._on_request)
        self.page.on("requestfinished", self._on_request_done)
        self.page.on("requestfailed", self._on_request_done)
        # Single-page agent: popups / new tabs are closed immediately.
        self._context.on("page", lambda p: p.close() if p is not self.page else None)

    def _on_request(self, _req) -> None:
        self._inflight += 1

    def _on_request_done(self, _req) -> None:
        self._inflight = max(0, self._inflight - 1)

    def _request_allowed(self, url: str) -> bool:
        if url.startswith(("data:", "blob:", "about:")):
            return True
        path = urlsplit(url).path
        if any(m in path for m in self.blocked_path_markers):
            return False
        return any(url.startswith(p) for p in self.allowed_url_prefixes)

    def _route(self, route, request) -> None:
        try:
            if self._request_allowed(request.url):
                route.continue_()
            else:
                self.blocked_requests.append(request.url)
                if request.is_navigation_request() and self.page is not None and request.frame == self.page.main_frame:
                    self._blocked_nav = True
                route.abort("blockedbyclient")
        except Exception:  # page closed mid-request etc.
            pass

    def _reset_impl(self, task: TaskSpec) -> Observation:
        from playwright.sync_api import Error as PlaywrightError

        self.spec = task
        self._init_episode(task)
        url = self._prepare_episode(task)
        self._ensure_browser()
        self._new_context()
        try:
            self.page.goto(url, wait_until="load")
            self._settle()
        except PlaywrightError as e:
            raise EnvironmentError_(f"could not open {url}: {_first_line(e)}") from e
        self._last_good_url = self.page.url
        return self._observe_impl()

    def _extract(self) -> dict:
        from playwright.sync_api import Error as PlaywrightError

        opts = {"mode": "extract", "next": self._next_id, "prefix": self.id_prefix,
                "viewportOnly": self.viewport_only, "maxElements": self.max_elements,
                "reportFocus": self.report_focus}
        for _ in range(4):
            try:
                res = self.page.evaluate(DOM_EXTRACT_JS, opts)
                self._next_id = max(self._next_id, int(res.get("next", self._next_id)))
                return res
            except PlaywrightError:  # e.g. execution context destroyed by a navigation
                try:
                    self.page.wait_for_load_state("load", timeout=self.settle_timeout_ms)
                except PlaywrightError:
                    pass
        return {"elements": [], "title": "", "url": self.page.url, "scroll": None}

    def _observe_impl(self) -> Observation:
        if self.page is None:
            raise EnvironmentError_("reset() first")
        err = self._check_error_page()
        if err and not self._last_error:
            self._last_error = err
        res = self._extract()
        elements = []
        for e in res.get("elements", []):
            try:
                elements.append(ElementState(**e))
            except Exception:  # malformed element: skip rather than crash
                continue
        return Observation(env_id=self.env_id, url=self._display_url(self.page.url), title=res.get("title", ""),
                           elements=elements, last_action_error=self._last_error, scroll_info=res.get("scroll"),
                           text=self._read_text, step=self._steps)

    def _available_actions_impl(self) -> list[Action]:
        assert self.spec is not None, "reset() first"
        return candidate_actions(self._observe_impl(), self.spec.goal, self._loading_hint())

    def _status(self, eid: str) -> dict:
        return self.page.evaluate(DOM_EXTRACT_JS, {"mode": "status", "id": eid, "viewportOnly": self.viewport_only})

    def _apply_impl(self, a: Action) -> Optional[str]:
        """Allowlisted dispatcher: typed action -> fixed Playwright operation."""
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        t = a.type
        try:
            if t in _ELEMENT_ACTIONS:
                eid = a.target or ""
                if not _ID_RE.match(eid):  # defensive: Action already validates ids
                    return f"element {eid!r} not found (stale or invalid id)"
                st = self._status(eid)
                if not st.get("found"):
                    return f"element {eid} not found (stale or invalid id)"
                if not st.get("visible"):
                    return f"element {eid} is not visible (scroll?)"
                if st.get("disabled"):
                    return f"element {eid} is disabled"
                loc = self.page.locator(f'[data-agentrs-id="{eid}"]').first
                if t == ActionType.READ:
                    self._read_text = st.get("text") or ""
                    return None
                if t == ActionType.CLICK:
                    loc.click(timeout=self.action_timeout_ms)
                elif t in (ActionType.TYPE, ActionType.CLEAR):
                    if not st.get("editable"):
                        return f"element {eid} is not editable"
                    loc.fill("" if t == ActionType.CLEAR else (a.text or ""), timeout=self.action_timeout_ms)
                elif t == ActionType.SELECT:
                    if not st.get("isSelect"):
                        return f"element {eid} is not a select box"
                    want = (a.text or "").strip().lower()
                    match = [o for o in (st.get("options") or []) if o.strip().lower() == want]
                    if not match:
                        return f"option {a.text!r} not found"
                    loc.select_option(label=match[0], timeout=self.action_timeout_ms)
            elif t == ActionType.SCROLL:
                self.page.evaluate(_SCROLL_JS, 1 if a.direction == "down" else -1)
            elif t == ActionType.NAVIGATE:
                url, err = self._resolve_navigation(a.text or "")
                if err:
                    return err
                self.page.goto(url, wait_until="load", timeout=self.navigation_timeout_ms)
            elif t == ActionType.WAIT:
                self._wait_impl()
            elif t == ActionType.FINISH:
                self._answer = a.text
                self._finish_impl(a.text)
            elif t == ActionType.SKILL:
                return "skills must be expanded by the agent runtime"
            else:  # pragma: no cover - exhaustive above
                return f"unsupported action {t.value}"
            self._settle()
            return self._check_error_page()
        except PlaywrightTimeoutError:
            self._safe_settle()
            return f"timed out performing {t.value}" + (f" on {a.target}" if a.target else "")
        except PlaywrightError as e:
            self._safe_settle()
            return self._check_error_page() or f"{t.value} failed: {_first_line(e)}"

    def _safe_settle(self) -> None:
        try:
            self._settle()
        except Exception:
            pass

    def _check_error_page(self) -> Optional[str]:
        """Blocked/failed navigations land on chrome-error://; go back and report it."""
        if self.page is None:
            return None
        if self._blocked_nav:  # the error page commits asynchronously after the aborted request
            self._blocked_nav = False
            end = time.monotonic() + 2.0
            while not self.page.url.startswith("chrome-error://") and time.monotonic() < end:
                self.page.wait_for_timeout(25)
        url = self.page.url
        if not url.startswith("chrome-error://"):
            if url.startswith(("http://", "https://")) and self._request_allowed(url):
                self._last_good_url = url
            return None
        blocked = self.blocked_requests[-1] if self.blocked_requests else "the requested page"
        self._recover()
        return f"navigation to {blocked!r} was blocked (outside the allowlist) or failed"

    def _recover(self) -> None:
        """Leave a browser error page: history back first, then the last good URL."""
        from playwright.sync_api import Error as PlaywrightError

        for attempt in range(8):
            try:
                self.page.wait_for_load_state("load", timeout=2000)
            except PlaywrightError:
                pass
            if not self.page.url.startswith("chrome-error://"):
                break
            try:
                if attempt == 0:
                    self.page.go_back(wait_until="load", timeout=self.navigation_timeout_ms)
                elif self._last_good_url:
                    self.page.goto(self._last_good_url, wait_until="load", timeout=self.navigation_timeout_ms)
                else:
                    break
            except PlaywrightError:  # e.g. still committing the error page
                self.page.wait_for_timeout(100)
        self._safe_settle()

    def _step_impl(self, action: Action) -> StepResult:
        assert self.spec is not None, "reset() first"
        if self._done:
            return StepResult(observation=self._observe_impl(), terminated=True, action_valid=False,
                              error="episode already finished")
        self._steps += 1
        self._last_error = None
        self._read_text = None
        try:
            error = self._apply_impl(action)
        except EnvironmentError_:
            raise
        except Exception as e:  # never crash on agent input
            error = f"environment error: {type(e).__name__}: {_first_line(e)}"
        self._last_error = error
        self._after_action(action)
        terminated = self._is_terminal(action)
        truncated = (not terminated) and self._steps >= self.spec.max_steps
        reward = 0.0
        info: dict[str, Any] = {}
        if terminated:
            ver = self._verify_impl()
            reward = 1.0 if ver.success else 0.0
            info["verification"] = ver.model_dump()
        if self.blocked_requests:
            info["blocked_requests"] = list(self.blocked_requests[-5:])
        self._done = terminated or truncated
        obs = self._observe_impl()
        return StepResult(observation=obs, reward=reward, terminated=terminated, truncated=truncated,
                          action_valid=error is None, error=error, info=info)

    def _close_impl(self) -> None:
        for obj in (self._context, self._browser):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._context = self._browser = self.page = None


def _norm_prefix(p: str) -> str:
    s = urlsplit(p)
    if s.scheme and s.netloc and not s.path:
        return p + "/"
    return p


def _paths_allow(paths: Sequence[str], url: str) -> bool:
    return any(url == p or (p.endswith("*") and url.startswith(p[:-1])) for p in paths)


# ====================================================================== MiniWeb in a real browser
_SETTLED_JS = "() => document.readyState === 'complete' && !document.documentElement.hasAttribute('data-busy')"
_VERSION_JS = "(v) => Number(document.documentElement.dataset.v || 0) >= v"


class LocalWebEnv(PlaywrightEnv):
    """MiniWeb families in Chromium, with MiniWebEnv semantics and server-side verification.

    By default an in-process ``LocalWebServer`` is started on 127.0.0.1 (random
    port). Pass ``server=`` to share one, or ``server_url=`` + ``control_token=``
    to use a server running elsewhere.
    """

    env_id = "localweb"
    version = "1.0"

    def __init__(self, server=None, server_url: Optional[str] = None, control_token: Optional[str] = None,
                 headless: bool = True, report_focus: bool = False, **kw):
        if server_url is not None and not control_token:
            raise ValueError("server_url requires control_token")
        super().__init__(headless=headless, report_focus=report_focus, **kw)
        self._server = server
        self._owns_server = server is None and server_url is None
        self._base = server_url.rstrip("/") if server_url else None
        self._token = control_token
        self._sid: Optional[str] = None
        self._prefix: Optional[str] = None  # absolute URL prefix of this session's app pages
        self._state: dict[str, Any] = {}
        # Loopback control traffic must never go through HTTP(S)_PROXY.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------------ control channel
    def _ensure_server(self) -> None:
        if self._base is not None and (self._server is None or self._server.running):
            return
        if self._server is None:
            from agentrs.environments.web.server import LocalWebServer

            self._server = LocalWebServer()
        self._server.start()
        self._base, self._token = self._server.base_url, self._server.token

    def _control(self, op: str, **payload) -> dict:
        req = urllib.request.Request(f"{self._base}/__control/{self._token}/{op}",
                                     data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise EnvironmentError_(f"control '{op}' failed: HTTP {e.code} {e.read()[:200]!r}") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise EnvironmentError_(f"control '{op}' failed: {e}") from e

    def session_state(self) -> dict:
        """Server-side session state (terminal / loading / url / version)."""
        return self._thread.call(lambda: self._control("state", sid=self._sid))

    # ------------------------------------------------------------------ hooks
    def _prepare_episode(self, task: TaskSpec) -> str:
        from agentrs.environments.miniweb.families import FAMILIES

        if task.family not in FAMILIES:
            raise ValueError(f"unknown MiniWeb family {task.family!r}")
        self._ensure_server()
        if self._sid is not None:
            try:
                self._control("close", sid=self._sid)
            except EnvironmentError_:
                pass
        st = self._control("session", task=json.loads(task.model_dump_json()))
        self._sid, self._state = st["sid"], st
        self._prefix = self._base + st["prefix"]
        self.allowed_url_prefixes = (self._prefix + "/",)  # the browser may only talk to this session
        return self._prefix + st["url"]

    def _display_url(self, url: str) -> str:
        if self._prefix and url.startswith(self._prefix):
            rest = urlsplit(url[len(self._prefix):]).path
            return rest or "/"
        return url

    def _resolve_navigation(self, text: str) -> tuple[Optional[str], Optional[str]]:
        path = text.strip()
        if self._prefix and path.startswith(self._prefix + "/"):  # absolute URL of this session
            path = path[len(self._prefix):]
        if not path.startswith("/") or not _paths_allow(self._state.get("allowed_paths", ()), path):
            return None, f"navigation to {text!r} is not permitted"
        return self._prefix + path, None

    def _finish_impl(self, answer: Optional[str]) -> None:
        self._state = self._control("finish", sid=self._sid, answer=answer)

    def _wait_impl(self) -> None:
        pass  # time passes through the app clock (two ticks), as in MiniWebEnv

    def _settle(self) -> None:
        from playwright.sync_api import Error as PlaywrightError

        try:
            self.page.wait_for_function(_SETTLED_JS, timeout=self.settle_timeout_ms)
        except PlaywrightError:
            super()._settle()

    def _advance_clock(self, n: int) -> dict:
        """Tick the app clock and bring the page up to date with the server."""
        from playwright.sync_api import Error as PlaywrightError

        st = self._control("tick", sid=self._sid, n=n)
        self._state = st
        if st.get("changed"):
            try:  # the page polls while loading; wait until it shows this version
                self.page.wait_for_function(_VERSION_JS, arg=st["version"], timeout=self.settle_timeout_ms)
            except PlaywrightError:
                try:
                    self.page.reload(wait_until="load")
                except PlaywrightError:
                    pass
            self._safe_settle()
        return st

    def _after_action(self, action: Action) -> None:
        # time passes on every step; ``wait`` lets one more tick pass (MiniWebEnv semantics)
        self._advance_clock(2 if action.type == ActionType.WAIT else 1)

    def _is_terminal(self, action: Action) -> bool:
        return bool(self._state.get("terminal")) or action.type == ActionType.FINISH

    def _loading_hint(self) -> Optional[bool]:
        return bool(self._state.get("loading"))

    def _verify_impl(self) -> Verification:
        if self._sid is None:
            raise EnvironmentError_("reset() first")
        r = self._control("verify", sid=self._sid)
        self._state = {k: v for k, v in r.items() if k != "verification"}
        return Verification.model_validate(r["verification"])

    def _close_impl(self) -> None:
        if self._sid is not None and self._base is not None:
            try:
                self._control("close", sid=self._sid)
            except EnvironmentError_:
                pass
            self._sid = None
        super()._close_impl()
        if self._owns_server and self._server is not None:
            self._server.stop()


__all__ = [
    "PlaywrightEnv",
    "LocalWebEnv",
    "PlaywrightThread",
    "BrowserUnavailable",
    "candidate_actions",
    "chromium_candidates",
    "launch_chromium",
    "resolved_chromium_executable",
    "DOM_EXTRACT_JS",
]
