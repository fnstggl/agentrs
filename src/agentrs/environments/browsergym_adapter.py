"""BrowserGym adapter: BrowserGym tasks (MiniWoB++ by default) behind ``Environment``.

* Observation: BrowserGym's merged accessibility tree + ``extra_element_properties``
  are mapped to ``ElementState`` lists. BrowserGym ids ("bids", e.g. ``"13"``)
  become element ids ``b<bid>`` (``b13``) so they satisfy the action grammar's
  identifier pattern; static text is aggregated per nearest bid-carrying element.
* Actions: typed ``Action`` objects are translated to BrowserGym high-level action
  strings (``click('13')``, ``fill('13', 'x')``, ``clear``, ``select_option``,
  ``scroll(0, dy)``, ``noop`` for wait, ``send_msg_to_user`` for finish answers).
  The string arguments are emitted with ``repr`` and BrowserGym re-parses them as
  literals; the action set is restricted to the ``chat`` and ``bid`` subsets (no
  ``goto``/tab actions), and navigation is refused before reaching BrowserGym.
* Verification: the benchmark's own reward/termination (MiniWoB's in-page task
  code via ``task.validate``), i.e. environment-side, never the agent's claim.

Compatibility workaround (playwright 1.64 / pre-installed Chromium)
-------------------------------------------------------------------
browsergym-core 0.14.3 pins playwright==1.44 but runs fine on 1.64 (installed
with ``--no-deps``). The only blocker is the browser binary: playwright 1.64
expects chromium build 1248 while only 1243 is installed. BrowserGym launches
two browsers (the task browser, which accepts ``pw_chromium_kwargs``, and its
chat window, which does not), both via a process-global Playwright instance
(``browsergym.core._get_global_playwright``). We therefore install, on our
dedicated Playwright thread, a thin proxy as that global instance whose
``chromium.launch`` adds ``executable_path`` when the caller did not pass one.
No BrowserGym code is modified.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agentrs.actions.schema import Action, ActionType
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
from agentrs.environments.web.playwright_env import (
    PlaywrightThread,
    _first_line,
    candidate_actions,
    launch_chromium,
    resolved_chromium_executable,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MINIWOB_DIR = _REPO_ROOT / "third_party" / "miniwob-plusplus" / "miniwob" / "html" / "miniwob"

_INTERACTIVE = {"button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "listbox", "menuitem",
                "menuitemcheckbox", "menuitemradio", "switch", "tab", "slider", "spinbutton", "treeitem"}
_INFORMATIVE = {"heading", "alert", "status", "dialog", "alertdialog", "img", "image", "tooltip", "note", "log",
                "timer", "progressbar", "meter"}
_ROLE_ALIASES = {"image": "img", "SearchBox": "searchbox", "ToggleButton": "button", "PopUpButton": "combobox"}
_OPTION_ROLES = {"option", "MenuListOption", "menuitem"}


def ensure_miniwob_url() -> str:
    """Point BrowserGym's MiniWoB tasks at the local MiniWoB++ checkout (unless set)."""
    if not os.environ.get("MINIWOB_URL"):
        os.environ["MINIWOB_URL"] = DEFAULT_MINIWOB_DIR.as_uri() + "/"
    return os.environ["MINIWOB_URL"]


class _BrowserTypeProxy:
    def __init__(self, bt, exe: Optional[str]):
        self._bt, self._exe = bt, exe

    def launch(self, *args, **kw):
        if self._exe and not kw.get("executable_path"):
            kw["executable_path"] = self._exe
        return self._bt.launch(*args, **kw)

    def __getattr__(self, name):
        return getattr(self._bt, name)


class _PlaywrightProxy:
    """Global Playwright instance for BrowserGym that injects our Chromium path."""

    def __init__(self, pw, exe: Optional[str]):
        self._pw, self._exe = pw, exe

    @property
    def chromium(self):
        return _BrowserTypeProxy(self._pw.chromium, self._exe)

    def __getattr__(self, name):
        return getattr(self._pw, name)


_PROBED: dict[bool, Optional[str]] = {}


def _chromium_for_browsergym(pw, headless: bool, executable_path: Optional[str]) -> Optional[str]:
    """Executable BrowserGym should use (None = Playwright's default works)."""
    if executable_path or os.environ.get("AGENTRS_CHROMIUM"):
        return executable_path or os.environ["AGENTRS_CHROMIUM"]
    if headless not in _PROBED:  # one probe launch per process and mode
        launch_chromium(pw, headless=headless).close()
        _PROBED[headless] = resolved_chromium_executable(headless)
    return _PROBED[headless]


def _ax_prop(node: dict, name: str):
    for p in node.get("properties") or []:
        if p.get("name") == name:
            return (p.get("value") or {}).get("value")
    return None


def _ax_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "mixed")


def _clip(s: str, n: int = 200) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


class BrowserGymEnv(Environment):
    """Run a BrowserGym task (default ``miniwob.click-test``) through our interface.

    ``reset(task)`` accepts a ``TaskSpec`` whose ``family`` is the BrowserGym task
    name (``"miniwob.click-button"``; a ``browsergym/`` prefix is accepted) and
    whose ``seed`` seeds the task. BrowserGym generates the goal at reset, so if
    ``task.goal`` is empty it is filled in from the benchmark's goal (the episode
    loop builds the agent's belief after ``reset`` and therefore sees it).
    ``reset()`` without a task uses the constructor's ``task_name`` / ``seed``.
    """

    env_id = "browsergym"
    version = "bg0.14.3-a1"

    def __init__(self, task_name: str = "miniwob.click-test", seed: int = 0, max_steps: int = 10,
                 headless: bool = True, slow_mo: Optional[int] = 0, pre_observation_delay: float = 0.1,
                 timeout_ms: Optional[int] = None, visibility_threshold: float = 0.5,
                 executable_path: Optional[str] = None, gym_kwargs: Optional[dict] = None):
        self.task_name = self._norm_name(task_name)
        self.seed = seed
        self.max_steps = max_steps
        self.headless = headless
        self.slow_mo = slow_mo
        self.pre_observation_delay = pre_observation_delay
        self.timeout_ms = timeout_ms
        self.visibility_threshold = visibility_threshold
        self.executable_path = executable_path
        self.gym_kwargs = dict(gym_kwargs or {})
        self._thread = PlaywrightThread.get()
        self._gym = None
        self._gym_name: Optional[str] = None
        self.spec: Optional[TaskSpec] = None
        self._bg_obs: Optional[dict] = None
        self._last_obs: Optional[Observation] = None
        self._task_info: dict = {}
        self._bg_done = False
        self._steps = 0
        self._done = False
        self._last_error: Optional[str] = None
        self._read_text: Optional[str] = None
        self.blocked_requests: list[str] = []

    @staticmethod
    def _norm_name(name: str) -> str:
        return name[len("browsergym/"):] if name.startswith("browsergym/") else name

    # ------------------------------------------------------------------ public API
    def reset(self, task: Optional[TaskSpec] = None) -> Observation:
        return self._thread.call(self._reset_impl, task)

    def observe(self) -> Observation:
        return self._thread.call(self._observe_impl)

    def available_actions(self) -> list[Action]:
        def _impl():
            obs = self._observe_impl()
            acts = [a for a in candidate_actions(obs, self.spec.goal if self.spec else "", loading=False)
                    if a.type != ActionType.NAVIGATE]
            return acts + ([Action.wait()] if not any(a.type == ActionType.WAIT for a in acts) else [])

        return self._thread.call(_impl)

    def step(self, action: Action) -> StepResult:
        return self._thread.call(self._step_impl, action)

    def verify_outcome(self) -> Verification:
        return self._thread.call(self._verify_impl)

    def get_task_metadata(self) -> PublicTask:
        assert self.spec is not None, "reset() first"
        return self.spec.public()

    def close(self) -> None:
        self._thread.call(self._close_impl)

    # ------------------------------------------------------------------ internals (browser thread)
    def _make_gym(self, name: str):
        import gymnasium as gym

        ensure_miniwob_url()
        import browsergym.core
        import browsergym.miniwob  # noqa: F401  (registers browsergym/miniwob.* ids)
        from browsergym.core.action.highlevel import HighLevelActionSet

        pw = self._thread.playwright
        exe = _chromium_for_browsergym(pw, self.headless, self.executable_path)
        browsergym.core._set_global_playwright(_PlaywrightProxy(pw, exe))  # see module docstring
        action_set = HighLevelActionSet(subsets=["chat", "bid"], multiaction=False, strict=True, demo_mode="off")
        kw: dict[str, Any] = dict(headless=self.headless, action_mapping=action_set.to_python_code,
                                  pre_observation_delay=self.pre_observation_delay)
        if self.slow_mo is not None:
            kw["slow_mo"] = self.slow_mo
        if self.timeout_ms is not None:
            kw["timeout"] = self.timeout_ms
        if exe:
            kw["pw_chromium_kwargs"] = {"executable_path": exe}
        kw.update(self.gym_kwargs)
        try:
            return gym.make(f"browsergym/{name}", **kw)
        except gym.error.Error as e:
            raise ValueError(f"unknown BrowserGym task {name!r}: {e}") from e

    def _reset_impl(self, task: Optional[TaskSpec]) -> Observation:
        if task is None:
            task = TaskSpec(task_id=f"browsergym/{self.task_name}/{self.seed}", env_id=self.env_id,
                            family=self.task_name, goal="", seed=self.seed, max_steps=self.max_steps)
        name = self._norm_name(task.family)
        if self._gym is None or self._gym_name != name:
            self._close_impl()
            self._gym = self._make_gym(name)
            self._gym_name = name
        try:
            obs, info = self._gym.reset(seed=task.seed)
        except Exception as e:
            raise EnvironmentError_(f"BrowserGym reset failed for {name}: {_first_line(e)}") from e
        self._install_network_guard()
        if not task.goal:
            task.goal = obs.get("goal") or ""
        self.spec = task
        self._bg_obs = obs
        self._task_info = dict(info.get("task_info") or {})
        self._bg_done = False
        self._steps = 0
        self._done = False
        self._last_error = None
        self._read_text = None
        return self._observe_impl()

    def _install_network_guard(self) -> None:
        """Abort network requests that leave the benchmark's own pages."""
        base = os.environ.get("MINIWOB_URL", "")
        ctx = self._gym.unwrapped.context
        self.blocked_requests = []

        def route(r, req):
            try:
                if req.url.startswith(("data:", "blob:", "about:")) or (base and req.url.startswith(base)):
                    r.continue_()
                else:
                    self.blocked_requests.append(req.url)
                    r.abort("blockedbyclient")
            except Exception:
                pass

        ctx.route("**/*", route)

    @property
    def _page(self):
        return self._gym.unwrapped.page

    def _elements(self, bg_obs: dict) -> list[ElementState]:
        nodes = (bg_obs.get("axtree_object") or {}).get("nodes") or []
        if not nodes:
            return []
        props = bg_obs.get("extra_element_properties") or {}
        focused = bg_obs.get("focused_element_bid")
        by_id = {n["nodeId"]: n for n in nodes}
        out: list[ElementState] = []
        texts: dict[str, ElementState] = {}

        def visible(bid: Optional[str]) -> bool:
            p = props.get(bid) if bid else None
            return p is None or float(p.get("visibility") or 0.0) >= self.visibility_threshold

        def role_of(n: dict) -> str:
            r = (n.get("role") or {}).get("value") or ""
            return _ROLE_ALIASES.get(r, r)

        def name_of(n: dict) -> str:
            return _clip((n.get("name") or {}).get("value") or "")

        def has_interactive(n: dict) -> bool:
            return any(role_of(by_id[c]) in _INTERACTIVE or has_interactive(by_id[c])
                       for c in n.get("childIds") or [] if c in by_id)

        def options(n: dict) -> list[str]:
            res: list[str] = []

            def walk(m):
                for c in m.get("childIds") or []:
                    ch = by_id.get(c)
                    if ch is None:
                        continue
                    if role_of(ch) in _OPTION_ROLES and name_of(ch):
                        res.append(name_of(ch))
                    walk(ch)

            walk(n)
            return res

        def emit(n: dict, role: str, bid: str) -> None:
            val = (n.get("value") or {}).get("value")
            el = ElementState(
                eid=f"b{bid}", role=role, name=name_of(n),
                value=None if val is None else str(val) if role in ("textbox", "searchbox", "combobox", "slider",
                                                                    "spinbutton", "listbox") else None,
                checked=_ax_bool(_ax_prop(n, "checked")) if role in ("checkbox", "radio", "switch",
                                                                      "menuitemcheckbox", "menuitemradio") else None,
                expanded=_ax_bool(_ax_prop(n, "expanded")) if role != "combobox" else None,
                disabled=bool(_ax_bool(_ax_prop(n, "disabled"))),
                focused=bool(focused and focused == bid),
                options=options(n) if role in ("combobox", "listbox") else None,
            )
            if role in ("textbox", "searchbox") and el.value is None:
                el = el.model_copy(update={"value": ""})
            out.append(el)

        def dfs(n: dict, anchor: Optional[str], in_label_with_control: bool) -> None:
            role = role_of(n)
            bid = n.get("browsergym_id")
            if not n.get("ignored"):
                if role in _INTERACTIVE and bid:
                    if visible(bid):
                        emit(n, role, bid)
                    return
                if role in _INFORMATIVE and bid and visible(bid):
                    emit(n, role, bid)
                if role == "StaticText" and anchor and not in_label_with_control:
                    txt = name_of(n)
                    if txt and visible(anchor):
                        el = texts.get(anchor)
                        if el is None:
                            el = ElementState(eid=f"b{anchor}", role="text", name=txt)
                            texts[anchor] = el
                            out.append(el)
                        else:
                            el.name = _clip(el.name + " " + txt)
                    return
            if role == "LabelText" and has_interactive(n):
                in_label_with_control = True  # the label names the control it wraps
            if bid and role not in _INFORMATIVE:
                anchor = bid
            for c in n.get("childIds") or []:
                if c in by_id:
                    dfs(by_id[c], anchor, in_label_with_control)

        dfs(nodes[0], None, False)
        return [e for e in out if not (e.role == "text" and not e.name)]

    def _scroll_info(self) -> Optional[str]:
        try:
            top, vh, sh = self._page.evaluate(
                "() => { const s = document.scrollingElement || document.documentElement;"
                " return [s.scrollTop, window.innerHeight, s.scrollHeight]; }")
        except Exception:
            return None
        parts = [p for p, c in (("more content above", top > 1), ("more content below", top + vh < sh - 1)) if c]
        return ", ".join(parts) or None

    def _observe_impl(self) -> Observation:
        if self._bg_obs is None:
            raise EnvironmentError_("reset() first")
        o = self._bg_obs
        titles = list(o.get("open_pages_titles") or [])
        idx = int(list(o.get("active_page_index") or [0])[0]) if len(o.get("active_page_index", [])) else 0
        obs = Observation(env_id=self.env_id, url=o.get("url") or "", title=titles[idx] if idx < len(titles) else "",
                          elements=self._elements(o), last_action_error=self._last_error,
                          scroll_info=self._scroll_info(), text=self._read_text, step=self._steps)
        self._last_obs = obs
        return obs

    def _translate(self, a: Action) -> tuple[Optional[str], Optional[str]]:
        """Typed action -> (BrowserGym action string, error)."""
        t = a.type
        if t in (ActionType.CLICK, ActionType.TYPE, ActionType.CLEAR, ActionType.SELECT, ActionType.READ):
            obs = self._last_obs or self._observe_impl()
            el = obs.element(a.target or "")
            if el is None or not (a.target or "").startswith("b"):
                return None, f"element {a.target} not found (stale or invalid id)"
            bid = a.target[1:]
            if el.disabled:
                return None, f"element {a.target} is disabled"
            if t == ActionType.READ:
                return None, None
            if t == ActionType.CLICK:
                return f"click({bid!r})", None
            if t in (ActionType.TYPE, ActionType.CLEAR):
                if el.role not in ("textbox", "searchbox"):
                    return None, f"element {a.target} is not editable"
                return (f"clear({bid!r})" if t == ActionType.CLEAR else f"fill({bid!r}, {a.text or ''!r})"), None
            if el.role not in ("combobox", "listbox"):
                return None, f"element {a.target} is not a select box"
            match = [o for o in (el.options or []) if o.strip().lower() == (a.text or "").strip().lower()]
            if not match:
                return None, f"option {a.text!r} not found"
            return f"select_option({bid!r}, {match[0]!r})", None
        if t == ActionType.SCROLL:
            return f"scroll(0, {200 if a.direction == 'down' else -200})", None
        if t == ActionType.WAIT:
            return "noop(500)", None
        if t == ActionType.FINISH:
            return (f"send_msg_to_user({a.text!r})" if a.text else "noop(0)"), None
        if t == ActionType.NAVIGATE:
            return None, f"navigation to {a.text!r} is not permitted"
        if t == ActionType.SKILL:
            return None, "skills must be expanded by the agent runtime"
        return None, f"unsupported action {t.value}"  # pragma: no cover

    def _step_impl(self, action: Action) -> StepResult:
        assert self.spec is not None, "reset() first"
        if self._done:
            return StepResult(observation=self._observe_impl(), terminated=True, action_valid=False,
                              error="episode already finished")
        self._steps += 1
        self._last_error = None
        self._read_text = None
        bg_action, error = self._translate(action)
        if action.type == ActionType.READ and error is None:
            el = (self._last_obs or self._observe_impl()).element(action.target or "")
            self._read_text = el.name if el else ""
        bg_terminated = False
        if bg_action is not None:
            try:
                obs, _r, bg_terminated, _trunc, info = self._gym.step(bg_action)
                self._bg_obs = obs
                self._task_info = dict(info.get("task_info") or {})
                if obs.get("last_action_error"):
                    error = _first_line(obs["last_action_error"])
            except Exception as e:  # BrowserGym/Playwright failure -> observable error
                error = f"environment error: {type(e).__name__}: {_first_line(e)}"
        self._bg_done = self._bg_done or bool(bg_terminated)
        self._last_error = error
        terminated = self._bg_done or action.type == ActionType.FINISH
        truncated = (not terminated) and self._steps >= self.spec.max_steps
        reward, info_out = 0.0, {}
        if terminated:
            ver = self._verify_impl()
            reward = 1.0 if ver.success else 0.0
            info_out["verification"] = ver.model_dump()
        if self.blocked_requests:
            info_out["blocked_requests"] = list(self.blocked_requests[-5:])
        self._done = terminated or truncated
        return StepResult(observation=self._observe_impl(), reward=reward, terminated=terminated, truncated=truncated,
                          action_valid=error is None, error=error, info=info_out)

    def _verify_impl(self) -> Verification:
        if self._gym is None:
            raise EnvironmentError_("reset() first")
        info = self._task_info
        try:  # fresh benchmark-side validation (reads the task's own reward state)
            u = self._gym.unwrapped
            _r, done, _msg, info = u.task.validate(u.page, u.chat.messages)
            info = dict(info or {})
            self._task_info = info
            self._bg_done = self._bg_done or bool(done)
        except Exception:
            pass
        raw = info.get("RAW_REWARD_GLOBAL")
        raw = float(raw) if isinstance(raw, (int, float)) else 0.0
        done = bool(info.get("DONE_GLOBAL", self._bg_done))
        ok = done and raw > 0
        return Verification(success=ok, score=max(0.0, raw), progress=1.0 if ok else 0.0,
                            verifier_id=f"browsergym/{self._gym_name}",
                            evidence={"raw_reward": raw, "done": done, "reason": info.get("REWARD_REASON")})

    def _close_impl(self) -> None:
        if self._gym is not None:
            try:
                self._gym.close()
            except Exception:
                pass
        self._gym = None
        self._gym_name = None


__all__ = ["BrowserGymEnv", "ensure_miniwob_url", "DEFAULT_MINIWOB_DIR"]
