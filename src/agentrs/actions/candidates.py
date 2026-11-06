"""Generic, environment-independent candidate action generation from an observation.

Used when an environment does not provide ``available_actions`` (e.g. action-free
recordings for inverse dynamics). Text arguments come only from generic sources:
quoted spans of the goal and values visible in a provided "after" observation.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from agentrs.actions.schema import Action
from agentrs.environments.base import Observation

CLICKABLE = {"button", "link", "menuitem", "checkbox", "switch", "radio", "tab", "menubutton"}
EDITABLE = {"textbox", "searchbox"}


def quoted_spans(text: str) -> list[str]:
    out = []
    for s in re.findall(r"[\"“]([^\"”]{1,64})[\"”]", text or ""):
        if s not in out:
            out.append(s)
    return out


def candidates_from_observation(obs: Observation, goal: str = "", extra_texts: Iterable[str] = (),
                                after: Optional[Observation] = None, include_finish: bool = True) -> list[Action]:
    texts = quoted_spans(goal) + [t for t in extra_texts if t]
    if after is not None:
        # values that appear in editable fields after the step are plausible typed texts
        for e in after.elements:
            if e.role in EDITABLE and e.value and e.value not in texts:
                texts.append(e.value)
    acts: list[Action] = []
    for e in obs.elements:
        if e.disabled:
            continue
        if e.role in CLICKABLE:
            acts.append(Action.click(e.eid))
        if e.role in EDITABLE:
            for t in texts:
                if len(t) <= 64:
                    acts.append(Action.type_text(e.eid, t))
            if e.value:
                acts.append(Action.clear(e.eid))
        if e.role == "combobox" and e.options:
            for o in e.options:
                acts.append(Action.select(e.eid, o))
    acts += [Action.scroll("down"), Action.scroll("up"), Action.wait()]
    if include_finish:
        acts.append(Action.finish())
    seen, out = set(), []
    for a in acts:
        k = a.to_str()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out
