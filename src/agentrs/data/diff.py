"""Canonical observable state differences.

The world model predicts *decision-relevant changes* rather than full pages.
``StateDiff`` is a structured, id-aware difference between two observations with
a canonical text form (the generative target of the world model) and an
``apply`` operation (used to build imagined next observations for planning).

Text format (one item per line, deterministic order)::

    url: /cart
    title: Cart
    reload                      # every previous element disappeared (navigation)
    error: element e9 not found
    - [e12]                     # element removed
    ~ [e13] value="kitten3"     # attribute change(s)
    ~ [e14] checked
    + [new] alert "Email is required"
    (no change)                 # only when nothing changed

New elements get the placeholder id ``[new]``; ``apply`` assigns fresh imagined
ids (``n1``, ``n2``...). Exact ids of new elements are arbitrary in real
environments, so they are not prediction targets.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from agentrs.environments.base import ElementState, Observation

EVENT_NAMES = [
    "error",
    "url_changed",
    "reload",
    "elements_added",
    "elements_removed",
    "value_changed",
    "toggled",
    "expanded_changed",
    "alert_appeared",
    "dialog_appeared",
    "loading",
    "no_change",
]
N_EVENTS = len(EVENT_NAMES)
MAX_DIFF_ADDED = 14


class AttrChange(BaseModel):
    eid: str
    value: Optional[str] = None
    checked: Optional[bool] = None
    expanded: Optional[bool] = None
    disabled: Optional[bool] = None
    name: Optional[str] = None

    def to_line(self) -> str:
        parts = [f"~ [{self.eid}]"]
        if self.name is not None:
            parts.append(f"name={_q(self.name)}")
        if self.value is not None:
            parts.append(f"value={_q(self.value)}")
        if self.checked is not None:
            parts.append("checked" if self.checked else "unchecked")
        if self.expanded is not None:
            parts.append("expanded" if self.expanded else "collapsed")
        if self.disabled is not None:
            parts.append("disabled" if self.disabled else "enabled")
        return " ".join(parts)


def _q(s: str) -> str:
    s = s.replace("\n", " ")
    if len(s) > 80:
        s = s[:77] + "..."
    return '"' + s.replace('"', "'") + '"'


class StateDiff(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    reload: bool = False
    error: Optional[str] = None
    removed: list[str] = Field(default_factory=list)
    changed: list[AttrChange] = Field(default_factory=list)
    added: list[ElementState] = Field(default_factory=list)
    scroll_info: Optional[str] = None
    scroll_changed: bool = False

    # ------------------------------------------------------------------ text
    def to_text(self) -> str:
        lines = []
        if self.url is not None:
            lines.append(f"url: {self.url}")
        if self.title is not None:
            lines.append(f"title: {self.title}")
        if self.reload:
            lines.append("reload")
        if self.error:
            lines.append(f"error: {self.error[:100]}")
        if not self.reload:
            lines += [f"- [{eid}]" for eid in self.removed]
        lines += [c.to_line() for c in self.changed]
        for e in self.added[:MAX_DIFF_ADDED]:
            ln = e.to_line()
            lines.append("+ " + re.sub(r"^\[[^\]]+\]", "[new]", ln))
        if self.scroll_changed:
            lines.append(f"scroll: {self.scroll_info or 'none'}")
        if not lines:
            return "(no change)"
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str) -> "StateDiff":
        """Lenient parser for model-generated diffs. Unparseable lines are skipped."""
        d = cls()
        for raw in text.strip().splitlines():
            ln = raw.strip()
            if not ln or ln == "(no change)":
                continue
            if ln.startswith("url: "):
                d.url = ln[5:].strip()
            elif ln.startswith("title: "):
                d.title = ln[7:].strip()
            elif ln == "reload":
                d.reload = True
            elif ln.startswith("error: "):
                d.error = ln[7:].strip()
            elif ln.startswith("scroll: "):
                d.scroll_changed = True
                v = ln[8:].strip()
                d.scroll_info = None if v == "none" else v
            elif ln.startswith("- ["):
                m = re.match(r"- \[([^\]]+)\]", ln)
                if m:
                    d.removed.append(m.group(1))
            elif ln.startswith("~ ["):
                ch = _parse_change(ln)
                if ch is not None:
                    d.changed.append(ch)
            elif ln.startswith("+ "):
                el = parse_element_line(ln[2:])
                if el is not None:
                    d.added.append(el)
        return d

    # ------------------------------------------------------------------ apply
    def apply(self, obs: Observation, id_prefix: str = "n") -> Observation:
        """Build the predicted next observation. New elements get imagined ids."""
        if self.reload:
            elements: list[ElementState] = []
        else:
            removed = set(self.removed)
            elements = [e.model_copy() for e in obs.elements if e.eid not in removed]
            by_id = {e.eid: e for e in elements}
            for c in self.changed:
                e = by_id.get(c.eid)
                if e is None:
                    continue
                for attr in ("value", "checked", "expanded", "disabled", "name"):
                    v = getattr(c, attr)
                    if v is not None:
                        setattr(e, attr, v)
        used = {e.eid for e in elements} | obs.element_ids()
        k = 0
        for a in self.added:
            k += 1
            while f"{id_prefix}{k}" in used:
                k += 1
            elements.append(a.model_copy(update={"eid": f"{id_prefix}{k}"}))
            used.add(f"{id_prefix}{k}")
        return Observation(
            env_id=obs.env_id,
            url=self.url if self.url is not None else obs.url,
            title=self.title if self.title is not None else obs.title,
            elements=elements,
            last_action_error=self.error,
            scroll_info=self.scroll_info if self.scroll_changed else obs.scroll_info,
            step=obs.step + 1,
        )

    # ------------------------------------------------------------------ events
    def events(self, after: Optional[Observation] = None) -> list[float]:
        added_roles = {e.role for e in self.added}
        ev = {
            "error": self.error is not None,
            "url_changed": self.url is not None,
            "reload": self.reload,
            "elements_added": bool(self.added),
            "elements_removed": bool(self.removed) or self.reload,
            "value_changed": any(c.value is not None for c in self.changed),
            "toggled": any(c.checked is not None for c in self.changed),
            "expanded_changed": any(c.expanded is not None for c in self.changed),
            "alert_appeared": "alert" in added_roles,
            "dialog_appeared": "dialog" in added_roles,
            "loading": "status" in added_roles
            or (after is not None and any(e.role == "status" for e in after.elements)),
            "no_change": self.to_text() == "(no change)",
        }
        return [float(ev[n]) for n in EVENT_NAMES]


def _parse_change(ln: str) -> Optional[AttrChange]:
    m = re.match(r"~ \[([^\]]+)\](.*)$", ln)
    if not m:
        return None
    rest = m.group(2)
    ch = AttrChange(eid=m.group(1))
    mv = re.search(r'value="([^"]*)"', rest)
    if mv:
        ch.value = mv.group(1)
    mn = re.search(r'name="([^"]*)"', rest)
    if mn:
        ch.name = mn.group(1)
    toks = re.sub(r'(value|name)="[^"]*"', "", rest).split()
    for t in toks:
        if t == "checked":
            ch.checked = True
        elif t == "unchecked":
            ch.checked = False
        elif t == "expanded":
            ch.expanded = True
        elif t == "collapsed":
            ch.expanded = False
        elif t == "disabled":
            ch.disabled = True
        elif t == "enabled":
            ch.disabled = False
    return ch


_ELEM_RE = re.compile(r'^\[([^\]]+)\]\s+([a-z_]+)(?:\s+"([^"]*)")?(.*)$')


def parse_element_line(ln: str) -> Optional[ElementState]:
    m = _ELEM_RE.match(ln.strip())
    if not m:
        return None
    eid, role, name, rest = m.group(1), m.group(2), m.group(3) or "", m.group(4)
    el = ElementState(eid=eid if eid != "new" else "new", role=role, name=name)
    mv = re.search(r'value="([^"]*)"', rest)
    if mv:
        el.value = mv.group(1)
    mo = re.search(r"options=\[(.*)\]", rest)
    if mo:
        el.options = re.findall(r'"([^"]*)"', mo.group(1))
    rest2 = re.sub(r'value="[^"]*"|options=\[.*\]', "", rest).split()
    for t in rest2:
        if t == "checked":
            el.checked = True
        elif t == "unchecked":
            el.checked = False
        elif t == "expanded":
            el.expanded = True
        elif t == "collapsed":
            el.expanded = False
        elif t == "disabled":
            el.disabled = True
        elif t == "focused":
            el.focused = True
    return el


def compute_diff(before: Observation, after: Observation) -> StateDiff:
    d = StateDiff()
    if after.url != before.url:
        d.url = after.url
    if after.title != before.title:
        d.title = after.title
    d.error = after.last_action_error
    b_ids = [e.eid for e in before.elements]
    a_map = {e.eid: e for e in after.elements}
    shared = [i for i in b_ids if i in a_map]
    if before.elements and after.elements and not shared:
        d.reload = True
        d.added = list(after.elements)
    else:
        b_map = {e.eid: e for e in before.elements}
        d.removed = [i for i in b_ids if i not in a_map]
        for i in shared:
            x, y = b_map[i], a_map[i]
            ch = AttrChange(eid=i)
            changed = False
            for attr in ("value", "checked", "expanded", "disabled", "name"):
                if getattr(x, attr) != getattr(y, attr):
                    setattr(ch, attr, getattr(y, attr) if getattr(y, attr) is not None else "")
                    if attr in ("checked", "expanded", "disabled") and getattr(y, attr) is None:
                        setattr(ch, attr, False)
                    changed = True
            if changed:
                d.changed.append(ch)
        d.added = [e for e in after.elements if e.eid not in b_map]
    if after.scroll_info != before.scroll_info:
        d.scroll_changed = True
        d.scroll_info = after.scroll_info
    return d


def content_set(obs: Observation) -> set[tuple]:
    """Id-free multiset signature of an observation for comparing predicted
    and real observations."""
    return {(e.role, e.name, e.value or "", e.checked, e.expanded) for e in obs.elements} | {
        ("__url__", obs.url, "", None, None)
    }


def observation_f1(pred: Observation, real: Observation) -> float:
    a, b = content_set(pred), content_set(real)
    if not a and not b:
        return 1.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    p, r = tp / len(a), tp / len(b)
    return 2 * p * r / (p + r)
