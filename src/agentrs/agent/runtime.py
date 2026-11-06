"""Agent runtime: actors and the episode loop.

``run_episode`` is the single place where an agent interacts with a real
environment. It records a complete ``Transition`` for every real step and asks
the environment's verifier for the outcome. Nothing an actor returns can mark a
task as successful: success comes from ``env.verify_outcome()`` only.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agentrs.actions.schema import Action, try_parse_action
from agentrs.agent.belief import BeliefState
from agentrs.data.diff import compute_diff
from agentrs.data.schemas import EpisodeRecord, Provenance, Source, Transition, WMPredictionRecord
from agentrs.environments.base import Environment, EnvironmentError_, TaskSpec


@dataclass
class Decision:
    action: Action
    candidates: list[str] = field(default_factory=list)
    chosen_index: Optional[int] = None
    policy_logprob: Optional[float] = None
    behavior_logprob: Optional[float] = None
    selection: str = "policy"
    wm_prediction: Optional[WMPredictionRecord] = None
    info: dict[str, Any] = field(default_factory=dict)


class Actor(Protocol):
    name: str

    def begin_episode(self, belief: BeliefState) -> None: ...

    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision: ...


class RandomActor:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def begin_episode(self, belief):
        pass

    def decide(self, belief, candidates):
        i = self.rng.randrange(len(candidates))
        lp = -math.log(len(candidates))
        return Decision(action=candidates[i], candidates=[c.to_str() for c in candidates], chosen_index=i,
                        policy_logprob=lp, behavior_logprob=lp, selection="random")


class TeacherActor:
    """Adapter for scripted teachers (public info only)."""

    def __init__(self, teacher):
        self.teacher = teacher
        self.name = getattr(teacher, "source_id", "teacher")

    def begin_episode(self, belief):
        self.teacher.reset()

    def decide(self, belief, candidates):
        assert belief.observation is not None
        a = self.teacher.act(belief.task, belief.observation, candidates)
        cs = [c.to_str() for c in candidates]
        idx = cs.index(a.to_str()) if a.to_str() in cs else None
        return Decision(action=a, candidates=cs, chosen_index=idx, selection="teacher")


class HumanCLIActor:
    """Interactive terminal demonstrator: a consenting human types actions."""

    name = "human-cli"

    def __init__(self, input_fn=input, print_fn=print):
        self.input_fn, self.print_fn = input_fn, print_fn

    def begin_episode(self, belief):
        pass

    def decide(self, belief, candidates):
        self.print_fn("\n" + belief.prompt())
        self.print_fn("candidates: " + ", ".join(c.to_str() for c in candidates[:30]))
        while True:
            s = self.input_fn("action> ").strip()
            a = try_parse_action(s)
            if a is not None:
                cs = [c.to_str() for c in candidates]
                return Decision(action=a, candidates=cs, chosen_index=cs.index(a.to_str()) if a.to_str() in cs
                                else None, selection="human")
            self.print_fn("could not parse; use e.g. click(e12) or type(e3, \"text\")")


def run_episode(
    env: Environment,
    task: TaskSpec,
    actor: Actor,
    provenance: Optional[Provenance] = None,
    checkpoints: Optional[dict[str, str]] = None,
    belief_kwargs: Optional[dict] = None,
    memory=None,
    step_callback=None,
    record_prompts: bool = True,
) -> EpisodeRecord:
    prov = provenance or Provenance(source=Source.AGENT, source_id=getattr(actor, "name", "agent"))
    ep = EpisodeRecord(task_id=task.task_id, env_id=env.env_id, family=task.family, split=task.split,
                       goal=task.goal, provenance=prov, checkpoints=dict(checkpoints or {}))
    t0 = time.monotonic()
    try:
        obs = env.reset(task)
    except EnvironmentError_ as e:
        ep.termination_reason = f"reset_failed: {e}"
        ep.success = False
        return ep
    belief = BeliefState(task=task.public(), **(belief_kwargs or {}))
    belief.reset(obs)
    if memory is not None:
        belief.memory_hints = memory.hints_for(task.public())
    actor.begin_episode(belief)
    reason = "truncated:max_steps"
    terminated = False
    for step in range(task.max_steps):
        candidates = belief.augment_candidates(env.available_actions())
        prompt = belief.prompt() if record_prompts else None
        dec = actor.decide(belief, candidates)  # raw skill actions reach the env and are rejected there
        result = env.step(dec.action)
        diff = compute_diff(obs, result.observation)
        last = result.terminated or result.truncated or step == task.max_steps - 1
        tr = Transition(
            episode_id=ep.episode_id, step=step, task_id=task.task_id, env_id=env.env_id,
            env_version=getattr(env, "version", ""), family=task.family, split=task.split, goal=task.goal,
            observation=obs, belief_prompt=prompt, action=dec.action.to_str(), action_type=dec.action.type.value,
            candidates=dec.candidates, chosen_index=dec.chosen_index, policy_logprob=dec.policy_logprob,
            behavior_logprob=dec.behavior_logprob, selection=dec.selection, wm_prediction=dec.wm_prediction,
            next_observation=result.observation, diff_text=diff.to_text(),
            events=diff.events(result.observation), action_valid=result.action_valid, error=result.error,
            reward=result.reward, terminated=result.terminated,
            truncated=result.truncated or (last and not result.terminated),
            checkpoints=dict(checkpoints or {}), provenance=prov,
        )
        ep.transitions.append(tr)
        if not result.action_valid:
            ep.num_invalid += 1
        if step_callback is not None:
            step_callback(belief, dec, result)
        belief.update(dec.action, result)
        if record_prompts and (result.terminated or result.truncated or step == task.max_steps - 1):
            tr.next_belief_prompt = belief.prompt()
        obs = result.observation
        if result.terminated:
            terminated = True
            break
        if result.truncated:
            reason = "truncated:" + (result.info.get("guard") or "budget")
            break
    ver = env.verify_outcome()
    ep.success = bool(ver.success)
    ep.verification = ver.model_dump()
    ep.termination_reason = ("terminated:" + ("success" if ver.success else "failure")) if terminated else reason
    ep.num_steps = len(ep.transitions)
    ep.wall_time = time.monotonic() - t0
    if ep.transitions:
        ep.transitions[-1].termination_reason = ep.termination_reason
        ep.transitions[-1].verification = ep.verification
        ep.transitions[-1].progress = ver.progress
    for tr in ep.transitions:
        tr.episode_success = ep.success
    if memory is not None:
        memory.add_episode(ep)
    return ep

