"""Delayed-outcome content tasks for our agent in SocialWebLab.

``ask_for_help``: the agent has a concrete technical/practical problem (the goal states the
problem and the details it knows) and must get a *helpful answer within one day* from the lab's
community. It chooses where to post (operation ``post_in_<space>``) and writes the post (content).
Success is verified by the environment after 24 simulated hours (``helpful_answer_within`` —
answer helpfulness is privileged ground truth, never shown to the agent). The world model's
targets, in contrast, are publicly observable events (answered, replies, views, removal, ...).

``SocialWebLabContentEnv`` executes the post through the application's JSON API (in-process
transport: same handlers, authentication and rate limits as HTTP) under a test account. The
episode ends at submission; the reward is *pending* until it matures.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from agentrs.actions.schema import Action, ActionType
from agentrs.data.observed_actor_action import context_text
from agentrs.environments.base import (
    ElementState,
    Environment,
    Observation,
    PublicTask,
    StepResult,
    TaskSpec,
    Verification,
)
from agentrs.socialweblab.actors import PROBLEMS
from agentrs.socialweblab.apps import SPACES

FORUM_SPACES = [s for s, (app, _) in SPACES.items() if app == "forum"]
TRAIN_SPACES = ["python-help", "linux", "gardening", "cooking"]
HELDOUT_SPACES = ["databases", "webdev"]
SPLIT_SEED = {"train": 0, "val": 50_000, "test": 90_000, "test_env": 130_000}
HELP_WINDOW_S = 86400.0


def op_for(space: str) -> str:
    return "post_in_" + space.replace("-", "_")


def space_for(op: str) -> Optional[str]:
    for s in FORUM_SPACES:
        if op_for(s) == op:
            return s
    return None


def make_help_task(split: str, index: int, account: str, spaces: Optional[list[str]] = None) -> TaskSpec:
    rng = random.Random(SPLIT_SEED[split] + index * 7919)
    pool = spaces or (HELDOUT_SPACES if split == "test_env" else TRAIN_SPACES)
    space = pool[index % len(pool)]
    kws = SPACES[space][1]
    k, k2 = rng.sample(kws, 2)
    prob = rng.choice(PROBLEMS)
    detail = rng.choice([f"version {rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 9)}",
                         f"error E{rng.randint(1000, 9999)} on line {rng.randint(3, 300)}"])
    goal = (f"Get a helpful answer within 1 day from the community: your {k} {prob} when using {k2}. "
            f"You know: {detail}.")
    return TaskSpec(task_id=f"swl_help/{split}/{index}", env_id="socialweblab_content", family="ask_for_help",
                    goal=goal, seed=SPLIT_SEED[split] + index, split=split, max_steps=2, category="content",
                    private={"space": space, "k": k, "k2": k2, "problem": prob, "detail": detail, "account": account,
                             "within_s": HELP_WINDOW_S})


def own_context(lab, account_uid: int, space: str) -> str:
    """Context text of our own post, in exactly the format used for observed third-party posts."""
    app = SPACES[space][0]
    ctx = {"env": f"socialweblab/{app}", "space": space, "topic": ", ".join(SPACES[space][1][:5]),
           "author_followers": lab.followers(account_uid)}
    return context_text(ctx, lab.now(), lab.time_base)


class HelpTeacher:
    """Scripted author with mixed quality (so imitation does not already solve the task)."""

    def __init__(self, seed: int = 0, p_right_space: float = 0.8, p_detail: float = 0.5, p_polite: float = 0.5,
                 p_shout: float = 0.12, p_clickbait: float = 0.12):
        self.rng = random.Random(seed)
        self.p = (p_right_space, p_detail, p_polite, p_shout, p_clickbait)

    def write(self, task: TaskSpec) -> tuple[str, str]:
        r, (pr, pd, pp, ps, pc) = self.rng, self.p
        pv = task.private
        space = pv["space"] if r.random() < pr else r.choice(FORUM_SPACES)
        head = (f"How do I fix {pv['k']} when {pv['k2']} {pv['problem']}?" if r.random() < 0.8
                else f"My {pv['k']} {pv['problem']} with {pv['k2']}.")
        parts = [("URGENT: " if r.random() < pc else "") + head]
        if r.random() < pd:
            parts.append(f"I have {pv['detail']}.")
        if r.random() < ps:
            parts.append("NOTHING WORKS!!! HELP!!!")
        if r.random() < pp:
            parts.append(r.choice(["Thanks in advance!", "Any help appreciated."]))
        return op_for(space), " ".join(parts)


class TeacherHelpActor:
    """Runtime actor for the teacher; sees the public goal only (the task's private fields are
    used by the teacher to *write* like a person who has the problem, which the goal states)."""

    name = "help-teacher"

    def __init__(self, teacher: HelpTeacher, tasks: dict[str, TaskSpec]):
        self.t = teacher
        self.tasks = tasks

    def begin_episode(self, belief):
        pass

    def decide(self, belief, candidates):
        from agentrs.agent.runtime import Decision

        op, content = self.t.write(self.tasks[belief.task.task_id])
        return Decision(action=Action.submit(op, content), candidates=[c.to_str() for c in candidates],
                        chosen_index=None, selection="teacher")


class SocialWebLabContentEnv(Environment):
    env_id = "socialweblab_content"
    version = "1.0"

    def __init__(self, lab, app=None, account_passwords: Optional[dict[str, str]] = None):
        from agentrs.socialweblab.server import InProcessTransport, LabClient, app_for

        self.lab = lab
        self.app = app or app_for(lab)
        self._transport_cls = InProcessTransport
        self._client_cls = LabClient
        self.passwords = account_passwords or {}
        self.task: Optional[TaskSpec] = None
        self.created_item: Optional[int] = None
        self.created_at: Optional[float] = None
        self._done = False
        self._err: Optional[str] = None
        self._clients: dict[str, Any] = {}

    def ensure_account(self, handle: str, followers: int = 3, password: str = "agent-pass") -> int:
        try:
            uid = self.lab._uid(handle)
        except Exception:
            uid = self.lab.create_account(handle, password, kind="agent", spaces=FORUM_SPACES, followers=followers)
        self.passwords.setdefault(handle, password)
        return uid

    def _client(self, handle: str):
        if handle not in self._clients:
            t = self._transport_cls(self.app)
            c = self._client_cls(t)
            tok = c.login(handle, self.passwords[handle])
            t.session_token = tok if isinstance(tok, str) else (tok or {}).get("token")
            self._clients[handle] = c
        return self._clients[handle]

    def reset(self, task: TaskSpec) -> Observation:
        self.task = task
        self.created_item = None
        self.created_at = None
        self._done = False
        self._err = None
        self.ensure_account(task.private["account"])
        return self.observe()

    def account_uid(self) -> int:
        return self.lab._uid(self.task.private["account"])

    def observe(self) -> Observation:
        els = [ElementState(eid="g1", role="heading", name="Ask the community for help"),
               ElementState(eid="g2", role="text", name="Spaces: " + ", ".join(FORUM_SPACES)),
               ElementState(eid="g3", role="text", name=f"Your followers: {self.lab.followers(self.account_uid())}")]
        if self.created_item is not None:
            els.append(ElementState(eid="g4", role="status", name="Your post was published"))
        return Observation(env_id=self.env_id, url="/new", title="New post", elements=els,
                           last_action_error=self._err, step=int(self.created_item is not None))

    def available_actions(self) -> list[Action]:
        acts = [Action.submit(op_for(s), "") for s in FORUM_SPACES] if self.created_item is None else []
        return acts + [Action.finish()]

    def step(self, action: Action) -> StepResult:
        assert self.task is not None
        if self._done:
            return StepResult(observation=self.observe(), terminated=True, action_valid=False, error="episode finished")
        self._err = None
        info: dict[str, Any] = {}
        if action.type == ActionType.SUBMIT:
            space = space_for(action.name or "")
            text = (action.text or "").strip()
            if space is None:
                self._err = f"unknown operation {action.name!r}"
            elif not text:
                self._err = "empty post"
            else:
                title = text.split("?")[0][:100] + ("?" if "?" in text[:101] else "")
                c = self._client(self.task.private["account"])
                try:
                    res = c.create_item(space, title.strip() or text[:60], text,
                                        client_ref=f"{self.task.task_id}:{uuid.uuid4().hex}")  # one per attempt
                    self.created_item = int(res["id"] if isinstance(res, dict) else res)
                    self.created_at = self.lab.now()
                    info.update({"created_item_id": self.created_item, "reward_pending": True,
                                 "verify_after_s": self.task.private["within_s"], "space": space})
                except Exception as e:  # noqa: BLE001 - refused by the application (validation / rate limit)
                    self._err = f"post refused: {e}"
        elif action.type != ActionType.FINISH:
            self._err = f"action {action.type.value} not supported here"
        terminated = self.created_item is not None or action.type == ActionType.FINISH
        self._done = terminated
        ver = self.verify_outcome()
        info["verification"] = ver.model_dump()
        return StepResult(observation=self.observe(), reward=0.0, terminated=terminated,
                          action_valid=self._err is None, error=self._err, info=info)

    def verify_outcome(self) -> Verification:
        return verify_help(self.lab, self.created_item, self.created_at, self.task.private["within_s"])

    def get_task_metadata(self) -> PublicTask:
        return self.task.public()


def verify_help(lab, item_id: Optional[int], created_at: Optional[float], within_s: float) -> Verification:
    """Environment-side verification. Pending until ``within_s`` of lab time has elapsed."""
    if item_id is None:
        return Verification(success=False, verifier_id="socialweblab/help/v1", evidence={"reason": "nothing posted"})
    if lab.now() - (created_at or 0) < within_s - 1e-6:
        return Verification(success=False, verifier_id="socialweblab/help/v1",
                            evidence={"pending": True, "item_id": item_id})
    r = lab.helpful_answer_within(item_id, within_s)
    return Verification(success=r["success"], score=float(r["success"]), verifier_id="socialweblab/help/v1",
                        evidence={"pending": False, "item_id": item_id, **r})
