"""Browser workflow task families on SocialWebLab and a scripted teacher for them.

Families (``env_id="socialweblab"``; goals only mention public information - titles, spaces,
words to include):

``upvote_post``    find a post (by a simulated user) by its title through the asynchronous
                   search page, open it and upvote it            -> verifier ``reacted``
``reply_to_post``  open a post from its space's "New" listing and answer it, mentioning a word
                                                                   -> verifier ``replied``
``edit_own_post``  the agent's account already has a post (created through the service layer
                   when the task is made); edit its body to mention a phrase -> ``edited``
``create_post``    publish a post in a named space about a topic -> ``item_created``

``make_task(lab, family, split, index)`` creates whatever the task needs in the lab (the agent's
test account, a target post) and returns a ``TaskSpec`` whose ``private`` holds the account,
password, ``start_path`` and verifier. Splits use disjoint seed ranges; the ``test`` split also
uses held-out spaces (gardening / cooking) when the lab has them.

``SocialWebLabTeacher`` is a scripted demonstrator in the style of
``demonstrations/teacher.py``: it sees only the public goal and the observation and emits typed
``Action``s (``run_teacher`` drives an env with it; ``TeacherActor`` + ``run_episode`` work too).
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from agentrs.actions.schema import MAX_TEXT_LEN, Action
from agentrs.environments.base import (
    Observation,
    PublicTask,
    StepResult,
    TaskSpec,
    Verification,
)
from agentrs.socialweblab.actors import STYLES, write_item
from agentrs.socialweblab.apps import SPACES
from agentrs.socialweblab.engine import LabError, SocialWebLab

FAMILIES = ("upvote_post", "reply_to_post", "edit_own_post", "create_post")
SEED_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000}
SPLIT_SPACES = {"train": ("python-help", "linux", "webdev", "databases"),
                "val": ("python-help", "linux", "webdev", "databases"),
                "test": ("gardening", "cooking")}
MAX_STEPS = {"upvote_post": 14, "reply_to_post": 14, "edit_own_post": 16, "create_post": 12}
CATEGORY = {"upvote_post": "retrieval", "reply_to_post": "content", "edit_own_post": "content",
            "create_post": "content"}
_CLEAN = {"spam": False, "offtopic": False, "shout": False, "clickbait": False, "question": True, "code": False}


# ====================================================================== task generation
def account_password(lab: SocialWebLab, handle: str) -> str:
    """Deterministic per-lab password of a generated test account."""
    return hashlib.sha256(f"swl:{lab.lab_id}:{handle}".encode()).hexdigest()[:16]


def ensure_account(lab: SocialWebLab, handle: str, spaces: Iterable[str], followers: int = 0) -> tuple[int, str]:
    pw = account_password(lab, handle)
    rows = lab._q("SELECT id FROM users WHERE handle=?", (handle,))
    if rows:
        return rows[0]["id"], pw
    return lab.create_account(handle, pw, kind="agent", spaces=list(spaces), followers=followers), pw


def _spaces_for(lab: SocialWebLab, split: str) -> list[str]:
    have = {s["name"] for s in lab.spaces()}
    pref = [s for s in SPLIT_SPACES[split] if s in have]
    return pref or sorted(s for s in have if SPACES.get(s, ("",))[0] == "forum") or sorted(have)


def _simulated_author(lab: SocialWebLab, rng: random.Random) -> int:
    rows = lab._q("SELECT id FROM users WHERE kind='simulated' AND role IN ('veteran','regular','newbie','expert') "
                  "ORDER BY id")
    if not rows:
        raise LabError("the lab has no simulated authors", 409)
    return rng.choice(rows)["id"]


def _unique_post(lab: SocialWebLab, rng: random.Random, space: str, max_body: int = 400) -> tuple[str, str]:
    title = body = ""
    for _ in range(30):
        title, body, _ = write_item(rng, space, STYLES["regular"], overrides=dict(_CLEAN))
        title = title.replace('"', "'").strip()
        if len(body) <= max_body and not any(r["title"] == title for r in lab.search(title, limit=50)):
            return title, body
    return f"{title} (#{rng.randint(100, 999)})", body[:max_body]


def make_task(lab: SocialWebLab, family: str, split: str = "train", index: int = 0,
              account: str | None = None, followers: int = 0) -> TaskSpec:
    if family not in FAMILIES:
        raise ValueError(f"unknown socialweblab family {family!r}")
    if split not in SEED_BASE:
        raise ValueError(f"unknown split {split!r}")
    seed = SEED_BASE[split] + index * 101 + sum(map(ord, family))
    rng = random.Random(seed)
    spaces = _spaces_for(lab, split)
    space = rng.choice(spaces)
    kws = SPACES[space][1]
    handle = account or f"agent-{split}-{family.split('_')[0]}-{index}"
    uid, pw = ensure_account(lab, handle, spaces, followers)
    private: dict = {"account": handle, "password": pw, "start_path": "/"}
    if family == "upvote_post":
        title, body = _unique_post(lab, rng, space)
        iid = lab.create_item(_simulated_author(lab, rng), space, title, body)
        goal = f'Use the search page to find the post titled "{title}", open it and upvote it.'
        private.update(target_item=iid, verifier={"kind": "reacted", "item_id": iid, "reaction": "up"})
    elif family == "reply_to_post":
        title, body = _unique_post(lab, rng, space)
        iid = lab.create_item(_simulated_author(lab, rng), space, title, body)
        word = rng.choice(kws)
        goal = (f'In the "{space}" space, open the newest post titled "{title}" and post an answer to it '
                f'that mentions "{word}".')
        private.update(target_item=iid, verifier={"kind": "replied", "item_id": iid, "must_contain": [word]})
    elif family == "edit_own_post":
        title, body = _unique_post(lab, rng, space, max_body=160)
        iid = lab.create_item(uid, space, title, body)  # the agent's own earlier post (service layer)
        phrase = f"version {rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
        goal = f'Edit your post titled "{title}" so that its body also mentions "{phrase}".'
        private.update(target_item=iid, verifier={"kind": "edited", "item_id": iid, "must_contain": [phrase]})
    else:  # create_post
        k1, k2 = rng.sample(kws, 2)
        goal = (f'Publish a new post in the "{space}" space asking for help with {k1} and {k2}. '
                f'The post must mention "{k1}" and "{k2}".')
        private.update(space=space, verifier={"kind": "item_created", "space": space, "must_contain": [k1, k2]})
    return TaskSpec(task_id=f"socialweblab/{family}/{split}/{index}", env_id="socialweblab", family=family,
                    goal=goal, seed=seed, split=split, max_steps=MAX_STEPS[family], category=CATEGORY[family],
                    difficulty=2 if family in ("upvote_post", "create_post") else 3, private=private)


def task_set(lab: SocialWebLab, split: str = "train", families: Iterable[str] | None = None,
             n_per_family: int = 1, start: int = 0) -> list[TaskSpec]:
    return [make_task(lab, f, split, i) for f in (families or FAMILIES) for i in range(start, start + n_per_family)]


# ====================================================================== scripted teacher
def quoted(goal: str) -> list[str]:
    return re.findall(r'"([^"]{1,200})"', goal)


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


def _path(obs: Observation) -> str:
    return obs.url.split("?", 1)[0].split("#", 1)[0]


def _loading(obs: Observation) -> bool:
    return any(e.role == "status" for e in obs.elements)


def _more_below(obs: Observation) -> bool:
    return bool(obs.scroll_info and "below" in obs.scroll_info)


class SocialWebLabTeacher:
    """Scripted demonstrator for the SocialWebLab families (public information only)."""

    source_id = "teacher:socialweblab-v1"

    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.memory: dict = {}

    def reset(self) -> None:
        self.memory = {}

    def act(self, task: PublicTask, obs: Observation, candidates: list[Action] | None = None) -> Action:
        if candidates and self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)
        fn = getattr(self, f"_{task.family}", None)
        a = fn(task.goal, obs) if fn is not None else None
        return a if a is not None else Action.finish()

    # ---------------------------------------------------------------- shared sub-skills
    def _on_item(self, obs: Observation, title: str) -> bool:
        path = _path(obs)
        if not path.startswith("/item/"):
            return False
        if _find(obs, "heading", name=title):
            self.memory["item_path"] = path
            return True
        return self.memory.get("item_path") == path

    def _open_via_search(self, title: str, obs: Observation) -> Action | None:
        """None once the item page of ``title`` is open; otherwise the next action to get there."""
        if self._on_item(obs, title):
            return None
        path = _path(obs)
        if path == "/search":
            if _loading(obs):
                return Action.wait()  # results are fetched asynchronously
            link = _find(obs, "link", name=title)
            if link:
                return Action.click(link.eid)
            box = _find(obs, ("searchbox", "textbox"), name="Search posts")
            if box is None:
                return Action.scroll("up") if obs.scroll_info and "above" in obs.scroll_info else None
            if (box.value or "") != title:
                return Action.type_text(box.eid, title[:MAX_TEXT_LEN])
            searched = any(e.role == "text" and re.match(r"^\d+ results? for ", e.name) for e in obs.elements)
            if self.memory.get("searched") == title and searched:
                return Action.scroll("down") if _more_below(obs) else None  # not found: give up
            btn = _find(obs, "button", name="Search")
            if btn is None:
                return None
            self.memory["searched"] = title
            return Action.click(btn.eid)
        nav = _find(obs, "link", name="Search")
        return Action.click(nav.eid) if nav else Action.navigate("/search")

    # ---------------------------------------------------------------- families
    def _upvote_post(self, goal: str, obs: Observation) -> Action | None:
        title = quoted(goal)[0]
        a = self._open_via_search(title, obs)
        if a is not None or not self._on_item(obs, title):
            return a
        if _find(obs, "text", contains="You upvoted this post"):
            return Action.finish()
        btn = _find(obs, "button", name="Upvote")
        if btn:
            return Action.click(btn.eid)
        return Action.scroll("down") if _more_below(obs) else None

    def _edit_own_post(self, goal: str, obs: Observation) -> Action | None:
        title, phrase = quoted(goal)[:2]
        a = self._open_via_search(title, obs)
        if a is not None or not self._on_item(obs, title):
            return a
        meta = _find(obs, "text", contains="Posted by")
        if self.memory.get("saved") and meta and "edited" in meta.name:
            return Action.finish()
        box = _find(obs, "textbox", name="Edit body")
        if box is None:
            return Action.scroll("down") if _more_below(obs) else None
        if phrase.lower() not in (box.value or "").lower():
            new = f"{(box.value or '').strip()} Update: {phrase}.".strip()
            return Action.type_text(box.eid, new if len(new) <= MAX_TEXT_LEN else f"Update: {phrase}.")
        btn = _find(obs, "button", name="Save changes")
        if btn is None:
            return Action.scroll("down") if _more_below(obs) else None
        self.memory["saved"] = True
        return Action.click(btn.eid)

    def _reply_to_post(self, goal: str, obs: Observation) -> Action | None:
        space, title, word = quoted(goal)[:3]
        reply = f"Answer: check the {word} settings first, then restart and try again. That fixed it for me."
        path = _path(obs)
        if self._on_item(obs, title):
            if self.memory.get("replied") and _find(obs, "text", contains=reply[:40]):
                return Action.finish()
            box = _find(obs, "textbox", name="Your reply")
            kind = _find(obs, "combobox", name="Reply type")
            btn = _find(obs, "button", name="Post reply")
            if box is None or kind is None or btn is None:
                return Action.scroll("down") if _more_below(obs) else None
            if (kind.value or "") != "Answer":
                return Action.select(kind.eid, "Answer")
            if (box.value or "") != reply:
                return Action.type_text(box.eid, reply)
            self.memory["replied"] = True
            return Action.click(btn.eid)
        if path == f"/s/{space}":
            if not obs.url.endswith("sort=new"):
                new = _find(obs, "link", name="New")
                if new:
                    return Action.click(new.eid)
            link = _find(obs, "link", name=title)
            if link:
                return Action.click(link.eid)
            return Action.scroll("down") if _more_below(obs) else None
        link = _find(obs, "link", name=space)
        if link:
            return Action.click(link.eid)
        return Action.scroll("down") if _more_below(obs) else Action.navigate(f"/s/{space}")

    def _create_post(self, goal: str, obs: Observation) -> Action | None:
        space, k1, k2 = quoted(goal)[:3]
        title = f"How do I use {k1} together with {k2}?"
        body = (f"I am trying to use {k1} with {k2} but it fails with error E1234 on version 2.1.0. "
                f"What am I missing? Thanks in advance!")
        path = _path(obs)
        if path.startswith("/item/"):
            return Action.finish() if (_find(obs, "heading", name=title) or self.memory.get("published")) else \
                Action.navigate("/new")
        if path == "/new":
            if _loading(obs):
                return Action.wait()  # "Publishing…"
            if self.memory.get("published") and _find(obs, "alert"):
                return None  # the page refused the post: give up (verifier decides)
            sp = _find(obs, "combobox", name="Space")
            t = _find(obs, "textbox", name="Title")
            b = _find(obs, "textbox", name="Body")
            pub = _find(obs, "button", name="Publish")
            if sp is None or t is None or b is None or pub is None:
                return Action.scroll("down") if _more_below(obs) else None
            if (sp.value or "") != space:
                return Action.select(sp.eid, space)
            if (t.value or "") != title:
                return Action.type_text(t.eid, title)
            if (b.value or "") != body:
                return Action.type_text(b.eid, body)
            if pub.disabled:
                return Action.wait()
            self.memory["published"] = True
            return Action.click(pub.eid)
        nav = _find(obs, "link", name="New post")
        return Action.click(nav.eid) if nav else Action.navigate("/new")


@dataclass
class TeacherRun:
    observations: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    verification: Verification | None = None

    @property
    def success(self) -> bool:
        return bool(self.verification and self.verification.success)


def run_teacher(env, task: TaskSpec, teacher: SocialWebLabTeacher | None = None) -> TeacherRun:
    """Drive ``env`` with the scripted teacher until termination or ``task.max_steps``."""
    teacher = teacher or SocialWebLabTeacher()
    teacher.reset()
    run = TeacherRun()
    obs = env.reset(task)
    run.observations.append(obs)
    public = task.public()
    for _ in range(task.max_steps):
        a = teacher.act(public, obs)
        r = env.step(a)
        run.actions.append(a)
        run.results.append(r)
        obs = r.observation
        run.observations.append(obs)
        if r.terminated or r.truncated:
            break
    run.verification = env.verify_outcome()
    return run


__all__ = [
    "FAMILIES",
    "SocialWebLabTeacher",
    "TeacherRun",
    "account_password",
    "ensure_account",
    "make_task",
    "quoted",
    "run_teacher",
    "task_set",
]
