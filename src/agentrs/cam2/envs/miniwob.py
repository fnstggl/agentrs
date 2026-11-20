"""BrowserGym / MiniWoB++ adapter (real Chromium, official MiniWoB task code and reward).

* Observation: BrowserGym's accessibility tree (``tags_to_mark="all"`` so clickable spans get ids), flattened by
  BrowserGym's own ``flatten_axtree_to_str`` and parsed into ``Element``s. Element text excludes the bid (bids are
  arbitrary numbers; CAM refers to elements by pointer).
* Tools: BrowserGym's generic high-level ``bid`` primitives (click, dblclick, hover, focus, fill, clear, press,
  select_option, scroll, noop) published as typed schemas. No task-specific actions exist.
* Reward / success: BrowserGym's ``MiniwobTask.validate`` (MiniWoB's in-page ``RAW_REWARD_GLOBAL > 0``).
* Safety: every request that is not ``file://`` or ``data:`` is aborted (the pages are local files).
* Snapshot/restore: BrowserGym cannot serialise a live page, so a snapshot is (task, seed, action history) and
  ``restore`` replays the history after ``reset`` (MiniWoB tasks are seeded; flagged non-deterministic tasks
  refuse).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from agentrs.cam2.envs.base import (DependencyUnavailable, EnvironmentAdapter, EvalResult, NotSupported, StepResult,
                                    TaskRef)
from agentrs.cam2.tools import ArgSpec, Element, Observation, ToolCall, ToolSpec, validate_call

DEFAULT_DIR = os.environ.get("CAM2_MINIWOB_DIR", "/workspace/ext/miniwob-plusplus/miniwob/html/miniwob")

SETUP = """MiniWoB++ via BrowserGym is not available: {why}
Setup:
  uv pip install browsergym-core==0.14.3 browsergym-miniwob==0.14.3 playwright
  python -m playwright install chromium   # plus system libraries: playwright install-deps chromium (or apt)
  git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /workspace/ext/miniwob-plusplus
  export CAM2_MINIWOB_DIR=/workspace/ext/miniwob-plusplus/miniwob/html/miniwob
"""

TEXT_ROLES = ("textbox", "searchbox", "combobox", "spinbutton", "textarea")
KEYS = ("Enter", "Tab", "Backspace", "ArrowDown", "ArrowUp")
SKIP_ROLES = {"RootWebArea", "generic", "none", "LineBreak"}


def miniwob_tools() -> list:
    el = lambda roles=(): ArgSpec("target", "element", "element to act on", roles=roles)  # noqa: E731
    return [
        ToolSpec("click", "click an element with the mouse", (el(),)),
        ToolSpec("dblclick", "double-click an element", (el(),)),
        ToolSpec("hover", "move the mouse over an element", (el(),)),
        ToolSpec("focus", "give keyboard focus to an element", (el(),)),
        ToolSpec("fill", "replace the content of an input field with text", (el(TEXT_ROLES),
                 ArgSpec("value", "text", "text to type"))),
        ToolSpec("clear", "clear an input field", (el(TEXT_ROLES),)),
        ToolSpec("press", "press a key while an element has focus", (el(TEXT_ROLES + ("button", "link")),
                 ArgSpec("key", "enum", "key", options=KEYS))),
        ToolSpec("select_option", "choose an option of a drop-down or list box", (el(("option",)),)),
        ToolSpec("scroll", "scroll the page", (ArgSpec("direction", "enum", "direction", options=("up", "down")),)),
        ToolSpec("wait", "do nothing for half a second", ()),
    ]


_LINE = re.compile(r"^(\t*)(?:\[([^\]]+)\] )?(.*)$")
_NAME = re.compile(r"^(\S+)(?: '(.*?)')?(.*)$")


def parse_axtree(flat: str, max_elements: int = 80) -> list:
    out: list = []
    stack: list = []  # [depth, eid, role, claimable]: claimable = a skipped unnamed container with an id whose
    #                   first text child stands for it (MiniWoB "links" are <span>s: generic[bid] > StaticText)
    for line in flat.splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        depth, bid, rest = len(m.group(1)), m.group(2), m.group(3).strip()
        nm = _NAME.match(rest)
        role = nm.group(1) if nm else rest.split(" ")[0]
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None
        parent_role = stack[-1][2] if stack else None
        if role in SKIP_ROLES and not (nm and nm.group(2)):
            if bid:
                stack.append([depth, bid, role, True])
            continue
        if not bid and role == "StaticText" and stack and stack[-1][3]:
            bid = stack[-1][1]          # the text is the visible face of its clickable container
            stack[-1][3] = False
            role = "text"
        text = rest
        if role == "option" and parent_role in ("combobox", "listbox"):
            text = f"{rest} (in {parent_role})"
        out.append(Element(eid=bid or f"t{len(out)}", role=role, text=text[:200], depth=depth,
                           targetable=bool(bid), parent=parent))
        if bid and role != "text":
            stack.append([depth, bid, role, False])
    # keep targetable elements first if the page is large, preserving document order within each group
    if len(out) > max_elements:
        keep = [e for e in out if e.targetable][:max_elements]
        rest_ = [e for e in out if not e.targetable][: max_elements - len(keep)]
        ids = {id(e) for e in keep + rest_}
        out = [e for e in out if id(e) in ids]
    return out


class _MemoryChat:
    """Drop-in for BrowserGym's ``Chat`` without its separate Chromium window. MiniWoB tasks only read the message
    list (goal messages); the chat window is a human-facing display. Removing it halves the browser processes per
    worker, which matters under this container's task limit (pids.max = 2816 threads)."""

    def __init__(self, *args, **kwargs):
        self.messages: list = []

    def add_message(self, role, msg):
        import time as _t

        self.messages.append({"role": role, "timestamp": _t.time(), "message": msg})

    def wait_for_user_message(self):
        raise RuntimeError("no interactive user in CAM 2.0 training")

    def close(self):
        pass


class MiniWoBAdapter(EnvironmentAdapter):
    name = "miniwob"

    def __init__(self, max_steps: int = 10, pre_observation_delay: float = 0.1, headless: bool = True,
                 miniwob_dir: str = DEFAULT_DIR, max_elements: int = 80):
        try:
            import gymnasium  # noqa: F401
            import browsergym.miniwob  # noqa: F401
            from browsergym.utils.obs import flatten_axtree_to_str  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise DependencyUnavailable(SETUP.format(why=f"import failed: {e!r}")) from e
        if not Path(miniwob_dir, "click-button.html").exists():
            raise DependencyUnavailable(SETUP.format(why=f"MiniWoB++ HTML not found under {miniwob_dir}"))
        os.environ["MINIWOB_URL"] = Path(miniwob_dir).as_uri() + "/"
        import browsergym.core.env as _bge

        _bge.Chat = _MemoryChat
        self.max_steps, self.delay, self.headless, self.max_elements = max_steps, pre_observation_delay, headless, max_elements
        self.env = None
        self.task: Optional[TaskRef] = None
        self.tools = miniwob_tools()
        self.tmap = {t.name: t for t in self.tools}
        self.history: list = []
        self._obs: Optional[Observation] = None
        self._last_reward = 0.0
        self._done = False
        self.blocked_requests = 0

    def describe(self) -> dict:
        import browsergym.core

        return {"name": self.name, "browsergym": getattr(browsergym.core, "__version__", "?"),
                "max_steps": self.max_steps, "pre_observation_delay": self.delay, "tags_to_mark": "all"}

    def _guard(self):
        ctx = self.env.unwrapped.context

        def route(r):
            if r.request.url.startswith(("file://", "data:", "about:")):
                r.continue_()
            else:
                self.blocked_requests += 1
                r.abort()

        ctx.route("**/*", route)

    def reset(self, task: TaskRef) -> Observation:
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet

        def make():
            return gym.make(f"browsergym/miniwob.{task.family}", headless=self.headless,
                            pre_observation_delay=self.delay, tags_to_mark="all",
                            action_mapping=HighLevelActionSet(subsets=["bid"], strict=False).to_python_code)

        if self.env is None or self.task is None or self.task.family != task.family:
            self.close()
            self.env = make()
        self.task = task
        try:
            bg, _ = self.env.reset(seed=task.seed)
        except Exception:  # noqa: BLE001  - a crashed browser is rebuilt once; a second failure propagates
            self.close()
            self.env = make()
            bg, _ = self.env.reset(seed=task.seed)
        self._guard()
        self.history = []
        self._done = False
        self._last_reward = 0.0
        self._obs = self._convert(bg, 0)
        return self._obs

    def _convert(self, bg: dict, step: int) -> Observation:
        from browsergym.utils.obs import flatten_axtree_to_str

        flat = flatten_axtree_to_str(bg["axtree_object"], extra_properties=bg.get("extra_element_properties"),
                                     filter_visible_only=False)
        els = parse_axtree(flat, self.max_elements)
        err = bg.get("last_action_error") or ""
        return Observation(goal=bg.get("goal") or "", elements=els, text=("error: " + err[:200]) if err else "",
                           step=step)

    def observe(self) -> Observation:
        assert self._obs is not None
        return self._obs

    def available_tools(self) -> list:
        return list(self.tools)

    def to_browsergym(self, call: ToolCall) -> str:
        t = call.tool
        if t in ("click", "dblclick", "hover", "focus", "clear"):
            return f"{t}({call.element!r})"
        if t == "fill":
            return f"fill({call.element!r}, {call.text or ''!r})"
        if t == "press":
            return f"press({call.element!r}, {call.enum!r})"
        if t == "select_option":
            el = next(e for e in self._obs.elements if e.eid == call.element)
            m = _NAME.match(el.text)
            label = m.group(2) if m and m.group(2) is not None else ""
            return f"select_option({el.parent!r}, {label!r})"
        if t == "scroll":
            return f"scroll(0, {-150 if call.enum == 'up' else 150})"
        if t == "wait":
            return "noop(500)"
        raise ValueError(t)

    def step(self, action: ToolCall) -> StepResult:
        assert self._obs is not None and not self._done
        err = validate_call(action, self.tmap, self._obs)
        info = {"invalid": err is not None}
        if err is not None:  # never reaches the environment; the agent sees the error, the step counts
            n = len(self.history) + 1
            self.history.append(None)
            obs = Observation(goal=self._obs.goal, elements=self._obs.elements, text="invalid action: " + err, step=n)
            self._obs = obs
            trunc = n >= self.max_steps
            self._done = trunc
            if trunc:
                info.update({"success": False, "score": 0.0})
            return StepResult(obs, 0.0, False, trunc, info)
        act = self.to_browsergym(action)
        self.history.append(action)
        bg, r, term, trunc, binfo = self.env.step(act)
        n = len(self.history)
        self._obs = self._convert(bg, n)
        self._last_reward = float(r)
        trunc = bool(trunc) or (not term and n >= self.max_steps)
        self._done = bool(term) or trunc
        info.update({"bg_action": act, "action_error": bool(bg.get("last_action_error"))})
        if self._done:
            info.update({"success": self._last_reward > 0, "score": float(self._last_reward > 0)})
        return StepResult(self._obs, float(r), bool(term), trunc, info)

    def evaluate(self) -> EvalResult:
        return EvalResult(success=self._last_reward > 0, score=float(self._last_reward > 0),
                          details={"reward": self._last_reward, "steps": len(self.history)})

    def snapshot(self):
        return {"task": self.task, "history": list(self.history)}

    def restore(self, snap) -> Observation:
        cls = getattr(self.env.unwrapped, "task_entrypoint", None)
        if cls is not None and getattr(cls, "nondeterministic", False):
            raise NotSupported(f"{snap['task'].family} is flagged non-deterministic; replay restore is unsound")
        obs = self.reset(snap["task"])
        for a in snap["history"]:
            if a is None:
                continue
            self.step(a)
        return self._obs

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001  - closing an already-crashed browser
                pass
            self.env = None


# Task-family split (fixed before any CAM 2.0 result). Families are held out at the family level for VAL and TEST;
# TRAIN families additionally have instance-level splits by seed range.
TRAIN_FAMILIES = ["click-button", "click-link", "click-test-2", "click-dialog", "click-checkboxes", "click-tab",
                  "click-option", "focus-text", "enter-text", "enter-password", "login-user", "click-collapsible",
                  "choose-list", "click-widget", "enter-text-dynamic", "click-button-sequence"]
VAL_FAMILIES = ["click-dialog-2", "focus-text-2", "enter-text-2", "click-tab-2-easy", "click-checkboxes-transfer",
                "multi-layouts"]
TEST_FAMILIES = ["unicode-test", "login-user-popup", "click-collapsible-2", "click-checkboxes-soft", "use-autocomplete",
                 "navigate-tree"]
SEED_RANGES = {"train": (0, 1_000_000), "val": (1_000_000, 1_100_000), "test": (2_000_000, 2_100_000)}
