"""Scripted teacher agents for MiniWeb.

A teacher is "another agent completing tasks in the environment": it sees only
the public goal and observations (never ``TaskSpec.private``), like a human
demonstrator. Teachers are deliberately imperfect when ``epsilon > 0`` (random
valid actions are injected), so demonstrations carry a realistic mix of
successful, failed and partial trajectories. Outcomes are always decided by the
environment verifier, not by the teacher.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from agentrs.actions.schema import Action
from agentrs.environments.base import Observation, PublicTask
from agentrs.environments.miniweb.families import FIELD_LABELS, SECTIONS


def quoted(goal: str) -> list[str]:
    return re.findall(r"[\"“]([^\"”]{1,64})[\"”]", goal)


def _find(obs: Observation, role=None, name=None, contains=None):
    for e in obs.elements:
        if role is not None and (e.role not in role if isinstance(role, (set, tuple)) else e.role != role):
            continue
        if name is not None and e.name.strip().lower() != name.strip().lower():
            continue
        if contains is not None and contains.lower() not in e.name.lower():
            continue
        return e
    return None


class ScriptedTeacher:
    """Family-aware scripted demonstrator (public information only)."""

    source_id = "teacher:scripted-v1"

    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.memory: dict[str, str] = {}

    def reset(self) -> None:
        self.memory = {}

    def act(self, task: PublicTask, obs: Observation, candidates: Optional[list[Action]] = None) -> Action:
        if candidates and self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)
        fn = getattr(self, f"_{task.family}", None)
        if fn is None:
            return Action.finish()
        a = fn(task.goal, obs)
        return a if a is not None else Action.finish()

    # ---------------------------------------------------------------- families
    def _click_button(self, goal, obs):
        t = quoted(goal)[0]
        e = _find(obs, role="button", name=t)
        if e:
            return Action.click(e.eid)
        if obs.scroll_info and "below" in obs.scroll_info:
            return Action.scroll("down")
        if obs.scroll_info and "above" in obs.scroll_info:
            return Action.scroll("up")
        return None

    _scroll_find = _click_button

    def _navigate_link(self, goal, obs):
        sub = quoted(goal)[0] if quoted(goal) else ""
        sec = next((s for s in SECTIONS if s in goal and s != sub), None)
        if sub and sec and obs.title.lower() == f"{sec} - {sub}".lower():
            return Action.finish()
        e = _find(obs, role="link", name=sub)
        if e and sec and obs.title.lower() == sec.lower():
            return Action.click(e.eid)
        e = _find(obs, role="link", name=sec) if sec else None
        if e:
            return Action.click(e.eid)
        e = _find(obs, role="link", name="Home")
        return Action.click(e.eid) if e else None

    def _fill_field(self, goal, obs):
        v = quoted(goal)[0]
        unq = re.sub(r"\"[^\"]*\"", "", goal)
        label = next((lab for lab in FIELD_LABELS if re.search(rf"\b{lab}\b", unq)), None)
        box = _find(obs, role="textbox", name=label) if label else None
        if box and (box.value or "") != v:
            return Action.type_text(box.eid, v)
        b = _find(obs, role="button", name="Submit")
        return Action.click(b.eid) if b else None

    def _login(self, goal, obs):
        q = quoted(goal)
        u, pw = q[0], q[1]
        ub, pb = _find(obs, role="textbox", name="Username"), _find(obs, role="textbox", name="Password")
        if ub and (ub.value or "") != u:
            return Action.type_text(ub.eid, u)
        if pb and (pb.value or "") != pw:
            return Action.type_text(pb.eid, pw)
        b = _find(obs, role="button", name="Sign in")
        return Action.click(b.eid) if b else Action.finish()

    def _checkboxes(self, goal, obs):
        targets = set(quoted(goal))
        for e in obs.elements:
            if e.role == "checkbox" and (e.name in targets) != bool(e.checked):
                return Action.click(e.eid)
        b = _find(obs, role="button", name="Submit")
        return Action.click(b.eid) if b else None

    def _dropdown(self, goal, obs):
        opt = quoted(goal)[0]
        cb = _find(obs, role="combobox")
        if cb and (cb.value or "") != opt:
            return Action.select(cb.eid, opt)
        b = _find(obs, role="button", name="Submit")
        return Action.click(b.eid) if b else None

    def _menu(self, goal, obs):
        item = quoted(goal)[0]
        e = _find(obs, role="menuitem", name=item)
        if e:
            return Action.click(e.eid)
        m = _find(obs, role="menubutton", name="Actions")
        return Action.click(m.eid) if m else None

    def _search_lookup(self, goal, obs):
        prod = quoted(goal)[0]
        for e in obs.elements:
            if e.role == "text" and e.name.lower().startswith(prod.lower() + " - "):
                return Action.finish(e.name.split(" - ")[-1])
        if _find(obs, role="status"):
            return Action.wait()
        sb = _find(obs, role="searchbox")
        if sb and (sb.value or "") != prod:
            return Action.type_text(sb.eid, prod)
        b = _find(obs, role="button", name="Search")
        if b and not self.memory.get("searched"):
            self.memory["searched"] = "1"
            return Action.click(b.eid)
        if b:
            self.memory.pop("searched", None)
            return Action.wait()
        return None

    def _form_validation(self, goal, obs):
        name, email = quoted(goal)[0], quoted(goal)[1]
        nb, eb = _find(obs, role="textbox", name="Full name"), _find(obs, role="textbox", name="Email")
        if nb and (nb.value or "") != name:
            return Action.type_text(nb.eid, name)
        if eb and (eb.value or "") != email:
            return Action.type_text(eb.eid, email)
        b = _find(obs, role="button", name="Create account")
        return Action.click(b.eid) if b else Action.finish()

    def _checkout(self, goal, obs):
        prod, name, city = quoted(goal)[:3]
        if obs.url == "/shop":
            cart = _find(obs, role="link", contains="Cart (")
            n = int(re.findall(r"\d+", cart.name)[0]) if cart else 0
            if n == 0:
                b = _find(obs, role="button", name=f"Add {prod} to cart")
                if b:
                    return Action.click(b.eid)
            return Action.click(cart.eid) if cart else None
        if obs.url == "/cart":
            c = _find(obs, role="link", name="Checkout")
            if c and not c.disabled:
                return Action.click(c.eid)
            b = _find(obs, role="link", name="Continue shopping")
            return Action.click(b.eid) if b else None
        if obs.url == "/checkout":
            nb, cb = _find(obs, role="textbox", name="Full name"), _find(obs, role="textbox", name="City")
            if nb and (nb.value or "") != name:
                return Action.type_text(nb.eid, name)
            if cb and (cb.value or "") != city:
                return Action.type_text(cb.eid, city)
            b = _find(obs, role="button", name="Place order")
            return Action.click(b.eid) if b else None
        return Action.finish()

    def _dialog_confirm(self, goal, obs):
        name = quoted(goal)[0]
        dlg = _find(obs, role="dialog")
        if dlg:
            btn = _find(obs, role="button", name="Confirm" if name in dlg.name else "Cancel")
            return Action.click(btn.eid) if btn else None
        b = _find(obs, role="button", name=f"Delete {name}")
        return Action.click(b.eid) if b else None

    def _toggle_settings(self, goal, obs):
        on, off = quoted(goal)[:2]
        for e in obs.elements:
            if e.role == "switch" and ((e.name == on and not e.checked) or (e.name == off and e.checked)):
                return Action.click(e.eid)
        b = _find(obs, role="button", name="Save")
        return Action.click(b.eid) if b else None

    def _lookup_then_fill(self, goal, obs):
        cust = quoted(goal)[0]
        if obs.url == "/orders":
            for e in obs.elements:
                if e.role == "text" and e.name.endswith(" - " + cust):
                    self.memory["code"] = e.name.split()[1]
            link = _find(obs, role="link", name="Support form")
            return Action.click(link.eid) if link else None
        code = self.memory.get("code")
        box = _find(obs, role="textbox", name="Order number")
        if code is None:
            link = _find(obs, role="link", name="Orders")
            return Action.click(link.eid) if link else None
        if box and (box.value or "") != code:
            return Action.type_text(box.eid, code)
        b = _find(obs, role="button", name="Submit")
        return Action.click(b.eid) if b else None
