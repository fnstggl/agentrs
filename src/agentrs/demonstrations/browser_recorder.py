"""Consent-based recorder for human demonstrations in a real browser.

A consenting human (``headless=False``) - or a test driving real input events
programmatically - operates a ``LocalWebEnv`` page. Injected listeners capture
the demonstrator's gestures on observed elements (``data-agentrs-id``):

* click on a button / link / checkbox / switch / menu item / label -> ``click``
* a text field's committed value (``change``, i.e. on blur)         -> ``type`` / ``clear``
* a ``<select>`` change                                             -> ``select``
* a mouse-wheel gesture                                             -> ``scroll``
* idling while the page shows a loading indicator                    -> ``wait``
* the recorder toolbar's Finish button (headed) or ``finish()``      -> ``finish``

Each captured gesture is *intercepted* (the page does not act on it) and then
replayed as the equivalent typed ``Action`` through the environment's own
allowlisted dispatcher (``env.step`` semantics: same clock ticks, termination
and verification). Human transitions are therefore exactly comparable with
agent transitions: the observation before a step is the state the gesture was
applied to, the observation after is the environment's response.

Every transition carries ``Provenance(source=HUMAN, consent=True, privacy=...,
source_id=demonstrator_id)``; the final outcome comes only from server-side
verification. The recorder refuses to start without explicit consent and a
demonstrator id.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from agentrs.actions.schema import Action
from agentrs.data.diff import compute_diff
from agentrs.data.schemas import EpisodeRecord, Provenance, Source, Transition
from agentrs.environments.base import Observation, StepResult, TaskSpec
from agentrs.environments.web.playwright_env import LocalWebEnv, PlaywrightEnv

PRIVACY_LEVELS = ("private", "internal", "shareable")
RECORDER_ID = "agentrs.browser_recorder/v1"


class ConsentRequired(PermissionError):
    """Raised when recording is attempted without recorded consent / demonstrator id."""


# Capture-phase listeners on ``window`` run before any page handler. Gestures are
# stopped there (preventDefault + stopImmediatePropagation) and reported to Python
# through the ``__agentrsRecord`` binding; the gesture's target is tagged with
# ``data-agentrs-pending`` so Python can resolve its element id after observing.
# While Python replays an action the <html> element carries
# ``data-agentrs-replay`` and the listeners let events through.
_LISTENER_JS = r"""
(() => {
  if (window.__agentrsRecorderInstalled) return;
  window.__agentrsRecorderInstalled = true;
  let seq = 0;
  const CLICKABLE = 'a[href], button, input[type=checkbox], input[type=radio], input[type=button], ' +
    'input[type=submit], input[type=reset], summary, [role=button], [role=link], [role=menuitem], ' +
    '[role=menuitemcheckbox], [role=menuitemradio], [role=switch], [role=checkbox], [role=radio], [role=tab], ' +
    '[role=option], [role=treeitem]';
  const replaying = () => document.documentElement.hasAttribute('data-agentrs-replay');
  const ignored = (t) => !(t instanceof Element) || !!t.closest('[data-agentrs-ignore]');
  const report = (kind, el, value) => {
    const s = ++seq;
    if (el) el.setAttribute('data-agentrs-pending', String(s));
    try { window.__agentrsRecord({seq: s, kind, value: value === undefined ? null : value}); } catch (e) {}
  };
  window.addEventListener('click', (e) => {
    if (replaying() || ignored(e.target)) return;
    let t = e.target;
    const lab = t.closest('label');
    if (lab && lab.control && !lab.control.contains(t)) t = lab.control;  // label click -> its control
    const ctl = t.closest(CLICKABLE);
    if (!ctl) return;  // focusing a field or clicking static text is not an action
    e.preventDefault();
    e.stopImmediatePropagation();
    report('click', ctl);
  }, true);
  window.addEventListener('input', (e) => {
    if (replaying() || ignored(e.target)) return;
    e.stopImmediatePropagation();  // the app sees the value only once it is committed and replayed
  }, true);
  window.addEventListener('change', (e) => {
    if (replaying() || ignored(e.target)) return;
    const t = e.target;
    e.stopImmediatePropagation();
    // ``defaultValue`` / ``defaultSelected`` mirror the app's current (server-rendered)
    // value. A commit equal to it changes nothing - e.g. the blur that follows a
    // replayed fill - and is not an action.
    if (t instanceof HTMLSelectElement) {
      const op = t.selectedOptions[0];
      const label = op ? (op.label || op.text || '').trim() : '';
      const def = Array.from(t.options).find((o) => o.defaultSelected);
      if (op && op === def) return;
      t.value = def ? def.value : '';  // revert; the replayed action re-applies it
      report('select', t, label);
    } else if ((t instanceof HTMLInputElement && !['checkbox', 'radio'].includes(t.type)) ||
               t instanceof HTMLTextAreaElement) {
      const v = t.value;
      if (v === t.defaultValue) return;
      t.value = t.defaultValue;
      report('type', t, v);
    }
  }, true);
  let lastWheel = 0;
  window.addEventListener('wheel', (e) => {
    if (replaying() || ignored(e.target)) return;
    e.preventDefault();
    const now = Date.now();
    if (now - lastWheel < 400 || !e.deltaY) return;  // one scroll action per wheel gesture
    lastWheel = now;
    report('scroll', null, e.deltaY > 0 ? 'down' : 'up');
  }, {capture: true, passive: false});
})();
"""

# Headed mode only: a small toolbar to end the demonstration (with an optional answer).
_TOOLBAR_JS = r"""
(() => {
  const mount = () => {
    if (document.getElementById('__agentrs_toolbar') || !document.body) return;
    const bar = document.createElement('div');
    bar.id = '__agentrs_toolbar';
    bar.setAttribute('data-agentrs-ignore', '');
    bar.style.cssText = 'position:fixed;right:8px;bottom:8px;z-index:2147483647;background:#222;color:#fff;' +
      'padding:6px 8px;border-radius:6px;font:13px system-ui;display:flex;gap:6px;align-items:center';
    bar.innerHTML = '<span>&#9679; Recording demo</span><input placeholder="answer (optional)" ' +
      'style="font:inherit;width:140px"><button style="font:inherit">Finish</button>';
    bar.querySelector('button').addEventListener('click', () => {
      const v = bar.querySelector('input').value.trim();
      try { window.__agentrsRecord({seq: 0, kind: 'finish', value: v || null}); } catch (e) {}
    });
    document.body.appendChild(bar);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount); else mount();
})();
"""

_RESOLVE_JS = r"""
(seq) => {
  const el = document.querySelector(`[data-agentrs-pending="${seq}"]`);
  if (!el) return null;
  el.removeAttribute('data-agentrs-pending');
  const host = el.closest('[data-agentrs-id]');
  return host ? host.getAttribute('data-agentrs-id') : null;
}
"""
_SET_REPLAY_JS = "() => document.documentElement.setAttribute('data-agentrs-replay', '1')"
_CLEAR_REPLAY_JS = "() => document.documentElement.removeAttribute('data-agentrs-replay')"


class BrowserDemoRecorder:
    """Records one demonstration episode at a time on a ``LocalWebEnv``."""

    def __init__(self, *, consent: bool, demonstrator_id: str, privacy: str = "private", headless: bool = True,
                 env: Optional[PlaywrightEnv] = None, idle_wait_s: float = 1.0, notes: str = ""):
        if consent is not True:
            raise ConsentRequired("human demonstrations can only be recorded with explicit consent (consent=True)")
        if not isinstance(demonstrator_id, str) or not demonstrator_id.strip():
            raise ConsentRequired("a non-empty demonstrator_id is required to record a human demonstration")
        if privacy not in PRIVACY_LEVELS:
            raise ValueError(f"privacy must be one of {PRIVACY_LEVELS}")
        self.provenance = Provenance(source=Source.HUMAN, source_id=demonstrator_id.strip(), consent=True,
                                     privacy=privacy, collected_by=RECORDER_ID, notes=notes)
        self.headless = headless
        self.env: PlaywrightEnv = env if env is not None else LocalWebEnv(headless=headless)
        self._owns_env = env is None
        self.idle_wait_s = idle_wait_s
        self.task: Optional[TaskSpec] = None
        self._events: list[dict] = []
        self._installed_ctx = None  # browser context the listeners were installed in
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.episode: Optional[EpisodeRecord] = None
        self.observation: Optional[Observation] = None
        self._events = []
        self._done = False
        self._terminated = False
        self._truncated = False
        self._unmapped = 0
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ lifecycle
    def start(self, task: TaskSpec) -> Observation:
        """Open the task in the browser and start capturing the demonstrator's gestures."""
        self._reset_episode_state()
        self.task = task
        self.observation = self.env.reset(task)
        self.episode = EpisodeRecord(task_id=task.task_id, env_id=self.env.env_id, family=task.family,
                                     split=task.split, goal=task.goal, provenance=self.provenance,
                                     metadata={"recorder": RECORDER_ID, "headless": self.headless})
        self._call(self._install)
        return self.observation

    def _install(self) -> None:
        ctx, page = self.env._context, self.env.page
        if ctx is not self._installed_ctx:  # the env opens a fresh browser context per episode
            ctx.expose_binding("__agentrsRecord", self._on_event)
            ctx.add_init_script(_LISTENER_JS)
            if not self.headless:
                ctx.add_init_script(_TOOLBAR_JS)
            self._installed_ctx = ctx
        page.evaluate(_LISTENER_JS)
        if not self.headless:
            page.evaluate(_TOOLBAR_JS)

    def _on_event(self, _source, payload) -> None:  # runs on the browser thread
        if isinstance(payload, dict) and len(self._events) < 10_000:
            self._events.append(payload)

    def _call(self, fn, *args):
        return self.env._thread.call(fn, *args)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def transitions(self) -> list[Transition]:
        return self.episode.transitions if self.episode is not None else []

    # ------------------------------------------------------------------ event -> typed action -> replay
    def _pump(self, max_wait_s: float = 0.0, until: Optional[Callable[[], bool]] = None) -> None:
        """Process captured events; keep listening for up to ``max_wait_s`` seconds."""
        deadline = time.monotonic() + max_wait_s
        while True:
            while self._events and not self._done:
                self._handle(self._events.pop(0))
            if self._done or (until is not None and until()) or time.monotonic() >= deadline:
                return
            if self.env.page is None or self.env.page.is_closed():
                return
            self.env.page.wait_for_timeout(30)  # lets Playwright deliver binding calls

    def _handle(self, ev: dict) -> None:
        kind, value = ev.get("kind"), ev.get("value")
        env = self.env
        obs = env._observe_impl()  # the state the gesture applies to (also ids new elements)
        target = None
        if kind in ("click", "type", "select"):
            target = env.page.evaluate(_RESOLVE_JS, int(ev.get("seq") or 0))
            if not target or obs.element(target) is None:
                self._unmapped += 1  # e.g. a gesture on an element that is not observable
                return
        try:
            if kind == "click":
                action = Action.click(target)
            elif kind == "type":
                action = Action.clear(target) if not value else Action.type_text(target, str(value))
            elif kind == "select":
                action = Action.select(target, str(value or ""))
            elif kind == "scroll" and value in ("up", "down"):
                action = Action.scroll(value)
            elif kind == "wait":
                action = Action.wait()
            elif kind == "finish":
                action = Action.finish(str(value) if value else None)
            else:
                self._unmapped += 1
                return
        except ValueError:  # e.g. text longer than the action grammar allows
            self._unmapped += 1
            return
        self._commit(action, obs)

    def _commit(self, action: Action, obs: Observation) -> None:
        env, task, ep = self.env, self.task, self.episode
        assert task is not None and ep is not None
        candidates = env._available_actions_impl()
        cs = [c.to_str() for c in candidates]
        try:
            env.page.evaluate(_SET_REPLAY_JS)
        except Exception:
            pass
        result: StepResult = env._step_impl(action)
        try:
            env.page.evaluate(_CLEAR_REPLAY_JS)
        except Exception:
            pass
        diff = compute_diff(obs, result.observation)
        a = action.to_str()
        tr = Transition(
            episode_id=ep.episode_id, step=len(ep.transitions), task_id=task.task_id, env_id=env.env_id,
            env_version=getattr(env, "version", ""), family=task.family, split=task.split, goal=task.goal,
            observation=obs, action=a, action_type=action.type.value, candidates=cs,
            chosen_index=cs.index(a) if a in cs else None, selection="human",
            next_observation=result.observation, diff_text=diff.to_text(), events=diff.events(result.observation),
            action_valid=result.action_valid, error=result.error, reward=result.reward,
            terminated=result.terminated, truncated=result.truncated, provenance=self.provenance,
        )
        ep.transitions.append(tr)
        if not result.action_valid:
            ep.num_invalid += 1
        self.observation = result.observation
        if result.terminated or result.truncated:
            self._done = True
            self._terminated, self._truncated = result.terminated, result.truncated

    def _record_now(self, kind: str, value=None) -> Optional[Transition]:
        n = len(self.transitions)
        if not self._done:
            self._pump()  # earlier gestures first
            if not self._done:
                self._handle({"kind": kind, "value": value})
        return self.transitions[-1] if len(self.transitions) > n else None

    def _gesture(self, perform: Callable[[], None], timeout_s: float) -> Optional[Transition]:
        if self._done:
            return None
        n, u = len(self.transitions), self._unmapped
        perform()
        self._pump(timeout_s, until=lambda: len(self.transitions) > n or self._unmapped > u)
        return self.transitions[-1] if len(self.transitions) > n else None

    # ------------------------------------------------------------------ programmatic "human" gestures
    # These produce real input events in the page (mouse / keyboard), which go
    # through exactly the same capture path as a person's gestures. Intended for
    # tests and scripted collection; humans use the headed browser directly.
    def _loc(self, eid: str):
        if not eid or not all(c.isalnum() or c in "_-" for c in eid):
            raise ValueError(f"invalid element id {eid!r}")
        return self.env.page.locator(f'[data-agentrs-id="{eid}"]').first

    def click(self, eid: str, timeout_s: float = 5.0) -> Optional[Transition]:
        return self._call(lambda: self._gesture(lambda: self._loc(eid).click(), timeout_s))

    def type(self, eid: str, text: str, timeout_s: float = 5.0) -> Optional[Transition]:
        def perform():
            loc, kb = self._loc(eid), self.env.page.keyboard
            loc.click()
            kb.press("ControlOrMeta+a")
            if text:
                kb.type(text)
            else:
                kb.press("Backspace")
            kb.press("Tab")  # commit -> ``change``

        return self._call(lambda: self._gesture(perform, timeout_s))

    def select(self, eid: str, option: str, timeout_s: float = 5.0) -> Optional[Transition]:
        return self._call(lambda: self._gesture(lambda: self._loc(eid).select_option(label=option), timeout_s))

    def scroll(self, direction: str = "down", timeout_s: float = 5.0) -> Optional[Transition]:
        def perform():
            w, h = self.env.viewport
            self.env.page.mouse.move(w / 2, h / 2)
            self.env.page.mouse.wheel(0, 300 if direction == "down" else -300)

        return self._call(lambda: self._gesture(perform, timeout_s))

    def wait(self) -> Optional[Transition]:
        """The demonstrator waits (e.g. for results to load): one ``wait()`` step."""
        return self._call(lambda: self._record_now("wait"))

    def finish(self, answer: Optional[str] = None) -> Optional[Transition]:
        """The demonstrator declares the task done (optionally with an answer)."""
        return self._call(lambda: self._record_now("finish", answer))

    def wait_for_human(self, timeout_s: float = 600.0) -> None:
        """Headed mode: record until the task terminates, the demonstrator presses
        Finish, the page is closed or ``timeout_s`` passes. Idling while the page
        is loading is recorded as ``wait()`` every ``idle_wait_s`` seconds."""

        def loop():
            deadline = time.monotonic() + timeout_s
            last = time.monotonic()
            while not self._done and time.monotonic() < deadline:
                if self.env.page is None or self.env.page.is_closed():
                    return
                n = len(self.transitions)
                self._pump(0.1)
                if len(self.transitions) > n:
                    last = time.monotonic()
                elif (getattr(self.env, "_state", {}).get("loading")
                      and time.monotonic() - last >= self.idle_wait_s):
                    self._handle({"kind": "wait"})
                    last = time.monotonic()

        self._call(loop)

    # ------------------------------------------------------------------ result
    def stop(self) -> EpisodeRecord:
        """Finalize the episode: verify server-side and return the ``EpisodeRecord``."""

        def _finalize() -> EpisodeRecord:
            ep = self.episode
            assert ep is not None, "start() first"
            self._pump()
            ver = self.env._verify_impl()
            ep.success = bool(ver.success)
            ep.verification = ver.model_dump()
            if self._terminated:
                ep.termination_reason = "terminated:" + ("success" if ver.success else "failure")
            elif self._truncated:
                ep.termination_reason = "truncated:max_steps"
            else:
                ep.termination_reason = "truncated:stopped"
            ep.num_steps = len(ep.transitions)
            ep.wall_time = time.monotonic() - self._t0
            ep.metadata["unmapped_events"] = self._unmapped
            if ep.transitions:
                last = ep.transitions[-1]
                last.termination_reason = ep.termination_reason
                last.verification = ep.verification
                last.progress = ver.progress
            for tr in ep.transitions:
                tr.episode_success = ep.success
            self._done = True
            return ep

        return self._call(_finalize)

    def close(self) -> None:
        if self._owns_env:
            self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def record_episode(task: TaskSpec, *, consent: bool, demonstrator_id: str, privacy: str = "private",
                   headless: bool = True, driver: Optional[Callable[[BrowserDemoRecorder], None]] = None,
                   timeout_s: float = 600.0, env: Optional[PlaywrightEnv] = None) -> EpisodeRecord:
    """Record one human demonstration of ``task`` and return the verified ``EpisodeRecord``.

    With ``driver=None`` a (headed) browser waits for a person to perform the task
    (``wait_for_human``). With a ``driver`` callable, it receives the recorder and
    drives real input events (``rec.click(eid)``, ``rec.type(eid, text)``, ...),
    e.g. in tests. Raises ``ConsentRequired`` without consent / demonstrator id.
    """
    rec = BrowserDemoRecorder(consent=consent, demonstrator_id=demonstrator_id, privacy=privacy,
                              headless=headless, env=env)
    try:
        rec.start(task)
        if driver is not None:
            driver(rec)
        else:
            rec.wait_for_human(timeout_s)
        return rec.stop()
    finally:
        rec.close()


__all__ = ["BrowserDemoRecorder", "ConsentRequired", "record_episode", "PRIVACY_LEVELS"]
