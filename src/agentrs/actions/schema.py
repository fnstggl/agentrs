"""Typed action schema with a canonical, strictly parsed string grammar.

Actions are the only way an agent can affect an environment. Every action has a
typed representation (``Action``) and a canonical string form used as the
language-model target, e.g.::

    click(e12)
    type(e3, "alice@example.com")
    clear(e3)
    select(e7, "Express")
    scroll(down)
    navigate("/settings")
    read(e9)
    wait()
    finish("42.50")

The parser is a small hand-written recursive-descent parser. It never evaluates
model output as code.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    CLEAR = "clear"
    SELECT = "select"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    READ = "read"
    WAIT = "wait"
    FINISH = "finish"
    SKILL = "skill"  # hierarchical macro-action; expanded into primitive actions
    SUBMIT = "submit"  # semantic content action: submit(<operation>, "<complete generated content>")


# Which argument slots each action type uses.
_SIGNATURES: dict[ActionType, tuple[str, ...]] = {
    ActionType.CLICK: ("target",),
    ActionType.TYPE: ("target", "text"),
    ActionType.CLEAR: ("target",),
    ActionType.SELECT: ("target", "text"),
    ActionType.SCROLL: ("direction",),
    ActionType.NAVIGATE: ("text",),
    ActionType.READ: ("target",),
    ActionType.WAIT: (),
    ActionType.FINISH: ("text?",),
    ActionType.SKILL: ("name", "text?"),
    ActionType.SUBMIT: ("name", "text"),
}

SCROLL_DIRECTIONS = ("up", "down")
MAX_TEXT_LEN = 256
MAX_CONTENT_LEN = 65_536  # generated content carried by SUBMIT; longer content goes by reference
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,31}$")


class ActionParseError(ValueError):
    pass


class Action(BaseModel):
    """A typed agent action. Immutable and hashable."""

    model_config = ConfigDict(frozen=True)

    type: ActionType
    target: Optional[str] = None  # element id
    text: Optional[str] = None  # typed text / option / url / answer / skill args
    direction: Optional[str] = None  # scroll direction
    name: Optional[str] = None  # skill name

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _ID_RE.match(v):
            raise ValueError(f"invalid element id {v!r}")
        return v

    def model_post_init(self, __context) -> None:  # structural validation
        sig = _SIGNATURES[self.type]
        required = {s for s in sig if not s.endswith("?")}
        optional = {s.rstrip("?") for s in sig if s.endswith("?")}
        allowed = required | optional
        for slot in ("target", "text", "direction", "name"):
            val = getattr(self, slot)
            if slot in required and val is None:
                raise ValueError(f"{self.type.value} requires '{slot}'")
            if slot not in allowed and val is not None:
                raise ValueError(f"{self.type.value} does not take '{slot}'")
        limit = MAX_CONTENT_LEN if self.type == ActionType.SUBMIT else MAX_TEXT_LEN
        if self.text is not None and len(self.text) > limit:
            raise ValueError(f"text argument longer than {limit} chars")
        if self.type == ActionType.SKILL and self.name is not None and not _ID_RE.match(self.name):
            raise ValueError(f"invalid skill name {self.name!r}")
        if self.type == ActionType.SUBMIT and (self.name is None or not _ID_RE.match(self.name)):
            raise ValueError(f"invalid content operation {self.name!r}")
        if self.type == ActionType.SCROLL and self.direction not in SCROLL_DIRECTIONS:
            raise ValueError(f"scroll direction must be one of {SCROLL_DIRECTIONS}")

    # ------------------------------------------------------------------ string form
    def to_str(self) -> str:
        t = self.type
        if t in (ActionType.CLICK, ActionType.CLEAR, ActionType.READ):
            return f"{t.value}({self.target})"
        if t in (ActionType.TYPE, ActionType.SELECT):
            return f"{t.value}({self.target}, {_quote(self.text)})"
        if t == ActionType.SCROLL:
            return f"scroll({self.direction})"
        if t == ActionType.NAVIGATE:
            return f"navigate({_quote(self.text)})"
        if t == ActionType.WAIT:
            return "wait()"
        if t == ActionType.FINISH:
            return "finish()" if self.text is None else f"finish({_quote(self.text)})"
        if t == ActionType.SUBMIT:
            return f"submit({self.name}, {_quote(self.text)})"
        if t == ActionType.SKILL:
            if self.text is None:
                return f"skill({self.name})"
            return f"skill({self.name}, {_quote(self.text)})"
        raise AssertionError(t)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_str()

    @classmethod
    def parse(cls, s: str) -> "Action":
        return parse_action(s)

    # convenience constructors
    @classmethod
    def click(cls, target: str) -> "Action":
        return cls(type=ActionType.CLICK, target=target)

    @classmethod
    def type_text(cls, target: str, text: str) -> "Action":
        return cls(type=ActionType.TYPE, target=target, text=text)

    @classmethod
    def clear(cls, target: str) -> "Action":
        return cls(type=ActionType.CLEAR, target=target)

    @classmethod
    def select(cls, target: str, option: str) -> "Action":
        return cls(type=ActionType.SELECT, target=target, text=option)

    @classmethod
    def scroll(cls, direction: str) -> "Action":
        return cls(type=ActionType.SCROLL, direction=direction)

    @classmethod
    def navigate(cls, url: str) -> "Action":
        return cls(type=ActionType.NAVIGATE, text=url)

    @classmethod
    def read(cls, target: str) -> "Action":
        return cls(type=ActionType.READ, target=target)

    @classmethod
    def wait(cls) -> "Action":
        return cls(type=ActionType.WAIT)

    @classmethod
    def submit(cls, operation: str, content: str) -> "Action":
        return cls(type=ActionType.SUBMIT, name=operation, text=content)

    @classmethod
    def finish(cls, answer: Optional[str] = None) -> "Action":
        return cls(type=ActionType.FINISH, text=answer)


def _quote(s: Optional[str]) -> str:
    assert s is not None
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class _Lexer:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, ch: str) -> None:
        self.ws()
        if self.peek() != ch:
            raise ActionParseError(f"expected {ch!r} at {self.i} in {self.s!r}")
        self.i += 1

    def ident(self) -> str:
        self.ws()
        m = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*").match(self.s, self.i)
        if not m:
            raise ActionParseError(f"expected identifier at {self.i} in {self.s!r}")
        self.i = m.end()
        return m.group(0)

    def string(self) -> str:
        self.ws()
        if self.peek() != '"':
            raise ActionParseError(f"expected string at {self.i} in {self.s!r}")
        self.i += 1
        out = []
        while True:
            if self.i >= len(self.s):
                raise ActionParseError("unterminated string")
            c = self.s[self.i]
            if c == "\\":
                if self.i + 1 >= len(self.s):
                    raise ActionParseError("dangling escape")
                n = self.s[self.i + 1]
                out.append({"n": "\n", '"': '"', "\\": "\\"}.get(n, n))
                self.i += 2
                continue
            if c == '"':
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1

    def done(self) -> None:
        self.ws()
        if self.i != len(self.s):
            raise ActionParseError(f"trailing characters in {self.s!r}")


def parse_action(s: str) -> Action:
    """Parse a canonical action string. Raises ActionParseError on any deviation."""
    s = s.strip()
    lx = _Lexer(s)
    name = lx.ident()
    try:
        at = ActionType(name)
    except ValueError as e:
        raise ActionParseError(f"unknown action type {name!r}") from e
    lx.expect("(")
    try:
        if at in (ActionType.CLICK, ActionType.CLEAR, ActionType.READ):
            tgt = lx.ident()
            lx.expect(")")
            lx.done()
            return Action(type=at, target=tgt)
        if at in (ActionType.TYPE, ActionType.SELECT):
            tgt = lx.ident()
            lx.expect(",")
            txt = lx.string()
            lx.expect(")")
            lx.done()
            return Action(type=at, target=tgt, text=txt)
        if at == ActionType.SCROLL:
            d = lx.ident()
            lx.expect(")")
            lx.done()
            return Action(type=at, direction=d)
        if at == ActionType.NAVIGATE:
            url = lx.string()
            lx.expect(")")
            lx.done()
            return Action(type=at, text=url)
        if at == ActionType.WAIT:
            lx.expect(")")
            lx.done()
            return Action(type=at)
        if at == ActionType.FINISH:
            lx.ws()
            if lx.peek() == ")":
                lx.i += 1
                lx.done()
                return Action(type=at)
            txt = lx.string()
            lx.expect(")")
            lx.done()
            return Action(type=at, text=txt)
        if at == ActionType.SUBMIT:
            nm = lx.ident()
            lx.expect(",")
            txt = lx.string()
            lx.expect(")")
            lx.done()
            return Action(type=at, name=nm, text=txt)
        if at == ActionType.SKILL:
            nm = lx.ident()
            lx.ws()
            txt = None
            if lx.peek() == ",":
                lx.i += 1
                txt = lx.string()
            lx.expect(")")
            lx.done()
            return Action(type=at, name=nm, text=txt)
    except ValueError as e:  # pydantic validation errors
        if isinstance(e, ActionParseError):
            raise
        raise ActionParseError(str(e)) from e
    raise ActionParseError(f"unhandled action {s!r}")  # pragma: no cover


def try_parse_action(s: str) -> Optional[Action]:
    try:
        return parse_action(s)
    except ActionParseError:
        return None
