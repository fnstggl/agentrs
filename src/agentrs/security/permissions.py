"""Permissions, allowlists and resource budgets.

``PermissionPolicy`` is immutable and created from configuration only. Nothing
an environment returns (page text, tool output) is ever parsed into
permissions, so untrusted content cannot grant itself capabilities.

``GuardedEnvironment`` is the allowlisted action dispatcher every agent action
passes through. It enforces:
* allowed action types,
* navigation URL allowlist (scheme/host/path prefixes),
* per-episode action budget and wall-clock budget,
* per-step timeout and rate limit,
* crash containment: environment exceptions become observable errors,
* consequential actions require an approval callback (default: deny).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from agentrs.actions.schema import Action, ActionType
from agentrs.environments.base import (
    Environment,
    Observation,
    PublicTask,
    StepResult,
    TaskSpec,
    Verification,
)


class PermissionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_action_types: frozenset[ActionType] = frozenset(
        {ActionType.CLICK, ActionType.TYPE, ActionType.CLEAR, ActionType.SELECT, ActionType.SCROLL,
         ActionType.NAVIGATE, ActionType.READ, ActionType.WAIT, ActionType.FINISH}
    )
    # URL allowlist. Relative paths ("/...") are resolved against the environment origin.
    allowed_url_prefixes: tuple[str, ...] = ("/",)
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    max_actions_per_episode: int = 50
    max_episode_seconds: float = 300.0
    step_timeout_seconds: float = 30.0
    min_seconds_between_actions: float = 0.0
    # Actions whose text matches these markers are treated as consequential and
    # need explicit approval (e.g. payments). Default approval callback denies.
    consequential_markers: tuple[str, ...] = ()
    max_text_length: int = 256

    def url_allowed(self, url: str) -> bool:
        p = urlparse(url)
        if p.scheme in ("", None) and not p.netloc:
            return any(url.startswith(pre) for pre in self.allowed_url_prefixes if pre.startswith("/"))
        if p.scheme not in ("http", "https"):
            return False
        if p.hostname not in self.allowed_hosts:
            return False
        path = p.path or "/"
        return any(path.startswith(pre) for pre in self.allowed_url_prefixes if pre.startswith("/")) or any(
            url.startswith(pre) for pre in self.allowed_url_prefixes if not pre.startswith("/"))


class PermissionDenied(Exception):
    pass


def check_action(policy: PermissionPolicy, action: Action) -> Optional[str]:
    """Return a denial reason or None. Pure function of (policy, action)."""
    if action.type not in policy.allowed_action_types:
        return f"action type {action.type.value!r} is not permitted"
    if action.text is not None and len(action.text) > policy.max_text_length:
        return "text argument too long"
    if action.type == ActionType.NAVIGATE and not policy.url_allowed(action.text or ""):
        return f"navigation to {action.text!r} is outside the allowlist"
    if policy.consequential_markers and action.text:
        low = action.text.lower()
        if any(m in low for m in policy.consequential_markers):
            return "consequential action requires human approval"
    return None


class GuardedEnvironment(Environment):
    """Wraps any Environment with permission checks, budgets and crash containment."""

    def __init__(self, env: Environment, policy: PermissionPolicy,
                 approve: Optional[Callable[[Action], bool]] = None):
        self.inner = env
        self.policy = policy
        self.approve = approve or (lambda a: False)
        self.env_id = env.env_id
        self.version = env.version
        self._n_actions = 0
        self._t0 = time.monotonic()
        self._last_action_t = 0.0
        self._task: Optional[TaskSpec] = None
        self._last_obs: Optional[Observation] = None
        self.denied: list[tuple[str, str]] = []

    def reset(self, task: TaskSpec) -> Observation:
        self._task = task
        self._n_actions = 0
        self._t0 = time.monotonic()
        self.denied = []
        try:
            obs = self.inner.reset(task)
        except Exception as e:  # reset failure is unrecoverable for this episode
            from agentrs.environments.base import EnvironmentError_

            raise EnvironmentError_(f"reset failed: {e!r}") from e
        self._last_obs = obs
        return obs

    def observe(self) -> Observation:
        self._last_obs = self.inner.observe()
        return self._last_obs

    def available_actions(self) -> list[Action]:
        return [a for a in self.inner.available_actions() if check_action(self.policy, a) is None]

    def _error_result(self, msg: str, truncated: bool = False) -> StepResult:
        obs = (self._last_obs or self.inner.observe()).model_copy(update={"last_action_error": msg})
        return StepResult(observation=obs, action_valid=False, error=msg, truncated=truncated,
                          info={"guard": msg})

    def step(self, action: Action) -> StepResult:
        self._n_actions += 1
        budget_hit = self._n_actions >= self.policy.max_actions_per_episode
        if time.monotonic() - self._t0 > self.policy.max_episode_seconds:
            return self._error_result("episode wall-clock budget exhausted", truncated=True)
        reason = check_action(self.policy, action)
        if reason == "consequential action requires human approval" and self.approve(action):
            reason = None
        if reason is not None:
            self.denied.append((action.to_str(), reason))
            return self._error_result(f"permission denied: {reason}", truncated=budget_hit)
        wait = self.policy.min_seconds_between_actions - (time.monotonic() - self._last_action_t)
        if wait > 0:
            time.sleep(wait)
        self._last_action_t = time.monotonic()
        result: dict = {}

        def _run():
            try:
                result["r"] = self.inner.step(action)
            except Exception as e:  # environment crash -> observable failure
                result["e"] = e

        if self.policy.step_timeout_seconds and self.policy.step_timeout_seconds < 1e6:
            th = threading.Thread(target=_run, daemon=True)
            th.start()
            th.join(self.policy.step_timeout_seconds)
            if th.is_alive():
                return self._error_result("action timed out", truncated=True)
        else:
            _run()
        if "e" in result:
            return self._error_result(f"environment error: {result['e']!r}", truncated=True)
        r: StepResult = result["r"]
        self._last_obs = r.observation
        if budget_hit and not r.terminated:
            r = r.model_copy(update={"truncated": True})
        return r

    def verify_outcome(self) -> Verification:
        return self.inner.verify_outcome()

    def get_task_metadata(self) -> PublicTask:
        return self.inner.get_task_metadata()

    def close(self) -> None:
        self.inner.close()


def default_policy_for(env_id: str, **overrides) -> PermissionPolicy:
    base = dict(allowed_url_prefixes=("/",), allowed_hosts=("localhost", "127.0.0.1"))
    base.update(overrides)
    return PermissionPolicy(**base)


class PrivacyGuard:
    """Refuses to export private demonstration data unless explicitly authorized."""

    @staticmethod
    def check_export(privacy: str, authorized: bool = False) -> None:
        if privacy == "private" and not authorized:
            raise PermissionDenied("private demonstration data cannot be exported without explicit authorization")


__all__ = ["PermissionPolicy", "GuardedEnvironment", "check_action", "PermissionDenied", "PrivacyGuard",
           "default_policy_for"]