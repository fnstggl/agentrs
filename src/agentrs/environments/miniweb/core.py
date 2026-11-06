"""MiniWeb: a deterministic, pure-Python simulated web-application environment.

Design goals
------------
* Fast (microseconds per step) so neural learning, not the simulator, dominates
  CPU time in learning experiments.
* Realistic *observable* dynamics shared across task families: links navigate,
  typing changes values, submitting incomplete forms raises validation errors,
  menus expand, dialogs open, content loads asynchronously, element ids change
  after navigation (stale references), long pages need scrolling.
* Independent verification: every app keeps *server-side* state that only the
  app's own handlers mutate. The verifier reads that state and the private task
  spec; the agent cannot write to it except by acting.
* Stochastic where useful (e.g. transient server errors) so the world model must
  learn distributions, not single outcomes.

Apps are written React-style: the page is re-rendered from app state after every
step. Widgets carry a stable ``key`` per page load; the environment maps keys to
element ids (``eid``). Ids are re-generated on every navigation, so references
to old ids become stale, as in a real browser.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agentrs.actions.schema import Action, ActionType
from agentrs.environments.base import (
    ElementState,
    Environment,
    Observation,
    PublicTask,
    StepResult,
    TaskSpec,
    Verification,
)

EDITABLE_ROLES = {"textbox", "searchbox"}
CLICKABLE_ROLES = {"button", "link", "menuitem", "checkbox", "switch", "radio", "tab", "menubutton"}


@dataclass
class W:
    """Widget in the simulated DOM."""

    key: str
    role: str
    name: str = ""
    value: Optional[str] = None
    checked: Optional[bool] = None
    expanded: Optional[bool] = None
    disabled: bool = False
    options: Optional[list[str]] = None
    href: Optional[str] = None
    on_click: Optional[Callable[[], None]] = None
    on_input: Optional[Callable[[str], None]] = None  # textbox / combobox value change
    full_text: Optional[str] = None  # returned by read()


@dataclass
class Page:
    url: str
    title: str
    widgets: list[W]
    scrollable: bool = False


class App:
    """Base class for a simulated web application (one task family)."""

    family: str = "base"
    category: str = ""
    allowed_paths: tuple[str, ...] = ()

    def __init__(self, spec: TaskSpec, rng: random.Random):
        self.spec = spec
        self.rng = rng
        self.server: dict[str, Any] = {}  # server-side state; verifier reads this
        self.url = "/"
        self.terminal = False  # app-declared terminal event happened
        self.tick = 0
        self._pending: list[tuple[int, Callable[[], None]]] = []
        self.field_values: dict[str, str] = {}  # client-side form state
        self.checks: dict[str, bool] = {}
        self.expanded: dict[str, bool] = {}
        self.setup()

    # --- to override --------------------------------------------------------
    def setup(self) -> None:
        pass

    def render(self) -> Page:  # pragma: no cover - abstract
        raise NotImplementedError

    def verify(self) -> Verification:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------------
    def goto(self, url: str) -> None:
        self.url = url
        self.field_values = {}
        self.checks = {}
        self.expanded = {}

    def later(self, delay: int, fn: Callable[[], None]) -> None:
        self._pending.append((self.tick + delay, fn))

    @property
    def loading(self) -> bool:
        return bool(self._pending)

    def advance(self, n: int = 1) -> None:
        for _ in range(n):
            self.tick += 1
            ready = [p for p in self._pending if p[0] <= self.tick]
            self._pending = [p for p in self._pending if p[0] > self.tick]
            for _, fn in ready:
                fn()

    def textbox(self, key: str, label: str, role: str = "textbox") -> W:
        def _set(v: str, k=key):
            self.field_values[k] = v

        return W(key=key, role=role, name=label, value=self.field_values.get(key, ""), on_input=_set)

    def checkbox(self, key: str, label: str, role: str = "checkbox") -> W:
        def _toggle(k=key):
            self.checks[k] = not self.checks.get(k, False)

        return W(key=key, role=role, name=label, checked=self.checks.get(key, False), on_click=_toggle)

    def combobox(self, key: str, label: str, options: list[str]) -> W:
        def _set(v: str, k=key):
            self.field_values[k] = v

        return W(
            key=key,
            role="combobox",
            name=label,
            value=self.field_values.get(key, ""),
            options=list(options),
            on_input=_set,
        )

    def link(self, key: str, label: str, href: str) -> W:
        return W(key=key, role="link", name=label, href=href)

    def button(self, key: str, label: str, fn: Optional[Callable[[], None]] = None, disabled=False) -> W:
        return W(key=key, role="button", name=label, on_click=fn, disabled=disabled)

    def text(self, key: str, s: str, role: str = "text") -> W:
        return W(key=key, role=role, name=s, full_text=s)


class MiniWebEnv(Environment):
    """Environment wrapper around a MiniWeb ``App``."""

    version = "1.0"

    def __init__(self, viewport: int = 12, start_id_range: tuple[int, int] = (1, 400)):
        self.env_id = "miniweb"
        self.viewport = viewport
        self.start_id_range = start_id_range
        self.app: Optional[App] = None
        self.spec: Optional[TaskSpec] = None
        self._id_counter = 0
        self._key_to_eid: dict[str, str] = {}
        self._eid_to_key: dict[str, str] = {}
        self._page_load = 0
        self._last_url = None
        self._scroll = 0
        self._last_error: Optional[str] = None
        self._read_text: Optional[str] = None
        self._steps = 0
        self._done = False
        self._rng = random.Random(0)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, task: TaskSpec) -> Observation:
        from agentrs.environments.miniweb.families import FAMILIES

        if task.family not in FAMILIES:
            raise ValueError(f"unknown MiniWeb family {task.family!r}")
        self.spec = task
        self._rng = random.Random(task.seed * 7919 + 17)
        self.app = FAMILIES[task.family](task, random.Random(task.seed))
        self._id_counter = self._rng.randint(*self.start_id_range)
        self._key_to_eid, self._eid_to_key = {}, {}
        self._page_load = 0
        self._last_url = None
        self._scroll = 0
        self._last_error = None
        self._read_text = None
        self._steps = 0
        self._done = False
        return self.observe()

    def get_task_metadata(self) -> PublicTask:
        assert self.spec is not None
        return self.spec.public()

    # ------------------------------------------------------------------ observation
    def _page(self) -> Page:
        assert self.app is not None
        page = self.app.render()
        if page.url != self._last_url:
            # navigation => new page load => fresh element ids
            self._page_load += 1
            self._key_to_eid, self._eid_to_key = {}, {}
            self._last_url = page.url
            self._scroll = 0
        return page

    def _eid(self, key: str) -> str:
        if key not in self._key_to_eid:
            self._id_counter += 1
            eid = f"e{self._id_counter}"
            self._key_to_eid[key] = eid
            self._eid_to_key[eid] = key
        return self._key_to_eid[key]

    def _visible(self, page: Page) -> list[W]:
        if not page.scrollable:
            return page.widgets
        return page.widgets[self._scroll : self._scroll + self.viewport]

    def observe(self) -> Observation:
        assert self.app is not None
        page = self._page()
        vis = self._visible(page)
        elements = []
        for w in vis:
            elements.append(
                ElementState(
                    eid=self._eid(w.key),
                    role=w.role,
                    name=w.name,
                    value=w.value,
                    checked=w.checked,
                    expanded=w.expanded,
                    disabled=w.disabled,
                    options=w.options,
                )
            )
        scroll_info = None
        if page.scrollable:
            above = self._scroll > 0
            below = self._scroll + self.viewport < len(page.widgets)
            if above or below:
                scroll_info = ", ".join(
                    s for s, c in (("more content above", above), ("more content below", below)) if c
                )
        if self.app.loading:
            elements.append(ElementState(eid=self._eid("__loading__"), role="status", name="Loading..."))
        return Observation(
            env_id=self.env_id,
            url=page.url,
            title=page.title,
            elements=elements,
            last_action_error=self._last_error,
            scroll_info=scroll_info,
            text=self._read_text,
            step=self._steps,
        )

    def _visible_widget(self, eid: str) -> Optional[W]:
        assert self.app is not None
        page = self._page()
        key = self._eid_to_key.get(eid)
        if key is None:
            return None
        for w in self._visible(page):
            if w.key == key:
                return w
        return None

    def available_actions(self) -> list[Action]:
        """Action mask: valid primitive actions on visible elements. Text-entry
        arguments are proposals from a generic copy mechanism (quoted spans of
        the goal); the agent may type any string."""
        assert self.app is not None and self.spec is not None
        obs = self.observe()
        acts: list[Action] = []
        proposals = goal_text_proposals(self.spec.goal)
        for e in obs.elements:
            if e.disabled:
                continue
            if e.role in CLICKABLE_ROLES:
                acts.append(Action.click(e.eid))
            if e.role in EDITABLE_ROLES:
                for p in proposals:
                    acts.append(Action.type_text(e.eid, p))
                if e.value:
                    acts.append(Action.clear(e.eid))
            if e.role == "combobox" and e.options:
                for o in e.options:
                    acts.append(Action.select(e.eid, o))
        if obs.scroll_info:
            if "below" in obs.scroll_info:
                acts.append(Action.scroll("down"))
            if "above" in obs.scroll_info:
                acts.append(Action.scroll("up"))
        if self.app.loading:
            acts.append(Action.wait())
        acts.append(Action.finish())
        for p in _answer_proposals(obs):
            acts.append(Action.finish(p))
        # de-duplicate, keep order
        seen, out = set(), []
        for a in acts:
            s = a.to_str()
            if s not in seen:
                seen.add(s)
                out.append(a)
        return out

    # ------------------------------------------------------------------ dynamics
    def step(self, action: Action) -> StepResult:
        assert self.app is not None and self.spec is not None
        if self._done:
            return StepResult(
                observation=self.observe(), terminated=True, action_valid=False, error="episode already finished"
            )
        self._steps += 1
        self._last_error = None
        self._read_text = None
        error = self._apply(action)
        self._last_error = error
        # time passes on every step; ``wait`` lets one more tick pass
        self.app.advance(2 if action.type == ActionType.WAIT else 1)
        terminated = self.app.terminal or action.type == ActionType.FINISH
        truncated = (not terminated) and self._steps >= self.spec.max_steps
        reward = 0.0
        info: dict[str, Any] = {}
        if terminated:
            ver = self.verify_outcome()
            reward = 1.0 if ver.success else 0.0
            info["verification"] = ver.model_dump()
        self._done = terminated or truncated
        obs = self.observe()
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            action_valid=error is None,
            error=error,
            info=info,
        )

    def _apply(self, a: Action) -> Optional[str]:
        app = self.app
        assert app is not None
        t = a.type
        if t in (ActionType.CLICK, ActionType.TYPE, ActionType.CLEAR, ActionType.SELECT, ActionType.READ):
            w = self._visible_widget(a.target or "")
            if w is None:
                if a.target in self._eid_to_key:
                    return f"element {a.target} is not visible (scroll?)"
                return f"element {a.target} not found (stale or invalid id)"
            if w.disabled:
                return f"element {a.target} is disabled"
            if t == ActionType.CLICK:
                if w.role == "link" and w.href is not None:
                    app.goto(w.href)
                    return None
                if w.on_click is not None:
                    w.on_click()
                    return None
                if w.role in EDITABLE_ROLES or w.role == "combobox":
                    return None  # focusing a field has no further effect
                if w.role in CLICKABLE_ROLES:
                    return None
                return None  # clicking static text: valid, no effect
            if t in (ActionType.TYPE, ActionType.CLEAR):
                if w.role not in EDITABLE_ROLES or w.on_input is None:
                    return f"element {a.target} is not editable"
                w.on_input("" if t == ActionType.CLEAR else (a.text or ""))
                return None
            if t == ActionType.SELECT:
                if w.role != "combobox" or w.on_input is None:
                    return f"element {a.target} is not a select box"
                opts = w.options or []
                match = [o for o in opts if o.strip().lower() == (a.text or "").strip().lower()]
                if not match:
                    return f"option {a.text!r} not found"
                w.on_input(match[0])
                return None
            if t == ActionType.READ:
                self._read_text = w.full_text or w.name
                return None
        if t == ActionType.SCROLL:
            page = self._page()
            if not page.scrollable:
                return None
            n = len(page.widgets)
            step = max(1, self.viewport // 2)
            if a.direction == "down":
                self._scroll = min(max(0, n - self.viewport), self._scroll + step)
            else:
                self._scroll = max(0, self._scroll - step)
            return None
        if t == ActionType.NAVIGATE:
            url = a.text or ""
            if not any(url == p or (p.endswith("*") and url.startswith(p[:-1])) for p in app.allowed_paths):
                return f"navigation to {url!r} is not permitted"
            app.goto(url)
            return None
        if t == ActionType.WAIT:
            return None
        if t == ActionType.FINISH:
            app.server["answer"] = a.text
            app.server["finished"] = True
            return None
        if t == ActionType.SKILL:
            return "skills must be expanded by the agent runtime"
        return f"unsupported action {a.type}"  # pragma: no cover

    def verify_outcome(self) -> Verification:
        assert self.app is not None
        v = self.app.verify()
        v.verifier_id = f"miniweb/{self.app.family}/v1"
        return v


def goal_text_proposals(goal: str) -> list[str]:
    """Generic copy mechanism: quoted spans in the goal are candidate text
    arguments. This is not task-specific logic: the policy must still decide
    which span goes in which field (or whether to type at all)."""
    import re

    spans = re.findall(r"[\"“']([^\"”']{1,64})[\"”']", goal)
    out = []
    for s in spans:
        if s not in out:
            out.append(s)
    return out


def _answer_proposals(obs: Observation) -> list[str]:
    """Candidate answers for retrieval tasks: values that look like prices,
    codes or numbers in the observation. Generic, not task-specific."""
    import re

    out: list[str] = []
    for e in obs.elements:
        for m in re.findall(r"(\$\d+(?:\.\d\d)?|#[A-Z0-9]{3,8}|\b[A-Z]{2,3}-\d{3,6}\b)", e.name):
            if m not in out:
                out.append(m)
    if obs.text:
        for m in re.findall(r"(\$\d+(?:\.\d\d)?|#[A-Z0-9]{3,8}|\b[A-Z]{2,3}-\d{3,6}\b)", obs.text):
            if m not in out:
                out.append(m)
    return out[:8]
