"""Episodes: prompt rendering, batched lock-step rollouts, expert demonstrations, records.

Every step stores what policy-gradient training needs to be exact: the rendered prompt, the
prefill, the sampled token ids and their behaviour log-probs, and the policy version that produced
them. Rewards come only from the environment's hidden verifier.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from agentrs.econ.core import REGISTRY, Task

SCHEMA_VERSION = "econ-exp-v1"
TOOL_PREFILL = '{"tool": "'
MAX_RESULT_CHARS = 900
MAX_PROMPT_CHARS = 11000


@dataclass
class Step:
    prompt: str
    prefill: str
    action: str                      # full action text sent to the environment (prefill + generated)
    token_ids: list = field(default_factory=list)
    behavior_logprobs: list = field(default_factory=list)
    result: str = ""
    valid: bool = True
    error: Optional[str] = None
    elapsed_s: float = 0.0
    finished_generation: bool = True
    t_wall: float = 0.0              # wall-clock time the action was executed (real elapsed time between steps)


@dataclass
class Episode:
    task_id: str
    domain: str
    family: str
    split: str
    seed: int
    goal: str
    provenance: str                  # agent_policy | expert_demonstration | evaluation
    policy_version: str
    steps: list = field(default_factory=list)
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    verification: dict = field(default_factory=dict)
    reward: float = 0.0
    wall_s: float = 0.0
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    group: Optional[str] = None
    schema: str = SCHEMA_VERSION

    @property
    def n_invalid(self) -> int:
        return sum(not s.valid for s in self.steps)

    def to_record(self, with_prompts: bool = True) -> dict:
        d = asdict(self)
        if not with_prompts:
            for s in d["steps"]:
                s["prompt"] = None
        return d


def make_env(task: Task):
    return REGISTRY[f"{task.domain}/{task.family}"].make_env(task)


def render(env, task: Task, history: list[tuple[str, str]]) -> tuple[list[dict], str]:
    """Chat messages + assistant prefill for the current state."""
    if getattr(env, "action_format", "json") == "miniweb":
        return [{"role": "system", "content": env.system_prompt()}, {"role": "user", "content": env.state_text()}], ""
    head = f"TASK: {task.goal}\n\nSTART: {env.initial_observation()}\n"
    steps = [f"STEP {i + 1}: ACTION {a}\nRESULT: {r[:MAX_RESULT_CHARS]}" for i, (a, r) in enumerate(history)]
    tail = f"\n\nYou have used {len(history)} of {task.max_steps} steps. Reply with your next action as one JSON line."
    budget = MAX_PROMPT_CHARS - len(head) - len(tail) - len(env.system_prompt())
    kept = []
    for s in reversed(steps):
        if sum(len(x) for x in kept) + len(s) > budget:
            kept.append(f"[{len(steps) - len(kept)} earlier steps omitted]")
            break
        kept.append(s)
    body = "\n".join(reversed(kept))
    user = head + ("\nHISTORY:\n" + body if body else "") + tail
    return [{"role": "system", "content": env.system_prompt()}, {"role": "user", "content": user}], TOOL_PREFILL


def finish(ep: Episode, env, t0: float) -> Episode:
    ok, details = env.verify()
    ep.success = bool(ok)
    ep.verification = details
    ep.terminated = bool(env.done)
    ep.truncated = not env.done
    ep.reward = (1.0 if ok else 0.0)
    ep.wall_s = time.time() - t0
    return ep


def run_expert(task: Task, oracle: Optional[Callable] = None) -> Episode:
    """Execute the expert (scripted oracle) in the real environment and record the trajectory."""
    env = make_env(task)
    fam = REGISTRY[f"{task.domain}/{task.family}"]
    ep = Episode(task.id, task.domain, task.family, task.split, task.seed, task.goal, "expert_demonstration", "expert")
    t0 = time.time()
    hist: list[tuple[str, str]] = []
    plan = None if hasattr(env, "expert_action") else list((oracle or fam.oracle)(task))
    for i in range(task.max_steps):
        msgs, prefill = render(env, task, hist)
        if plan is not None:
            if i >= len(plan):
                break
            action = plan[i]
        else:
            action = env.expert_action()
        t1 = time.time()
        out = env.step(action)
        ep.steps.append(Step(prompt="", prefill=prefill, action=action, result=out.observation, valid=out.valid,
                             error=out.error, elapsed_s=time.time() - t1, t_wall=t1))
        ep.steps[-1].messages = msgs  # type: ignore[attr-defined]
        hist.append((action, out.observation))
        if env.done:
            break
    return finish(ep, env, t0)


def rollout_batch(tasks: list[Task], act_fn: Callable, policy_version: str, provenance: str = "agent_policy",
                  groups: Optional[list[str]] = None, on_step: Optional[Callable] = None) -> list[Episode]:
    """Run all tasks in lock-step. ``act_fn(list[(messages, prefill)]) -> list[dict(text, ids, logprobs, finished, prompt)]``
    is called once per step with every still-active episode, so generation is batched across agents."""
    envs, eps, hists = [], [], []
    t0 = time.time()
    for k, t in enumerate(tasks):
        try:
            env = make_env(t)
        except Exception as e:  # a broken environment instance must not kill the batch
            env = None
            err = f"env setup failed: {type(e).__name__}: {e}"
        ep = Episode(t.id, t.domain, t.family, t.split, t.seed, t.goal, provenance, policy_version,
                     group=groups[k] if groups else None)
        if env is None:
            ep.verification = {"error": err}
        envs.append(env)
        eps.append(ep)
        hists.append([])
    active = [i for i, e in enumerate(envs) if e is not None]
    for step in range(max(t.max_steps for t in tasks)):
        active = [i for i in active if not envs[i].done and len(eps[i].steps) < tasks[i].max_steps]
        if not active:
            break
        reqs = [render(envs[i], tasks[i], hists[i]) for i in active]
        if getattr(act_fn, "wants_ctx", False):  # e.g. world-model-assisted selection needs the episode so far
            outs = act_fn(reqs, [(tasks[i], eps[i]) for i in active])
        else:
            outs = act_fn(reqs)
        for i, (msgs, prefill), o in zip(active, reqs, outs):
            action = prefill + o["text"]
            t1 = time.time()
            try:
                res = envs[i].step(action)
            except Exception as e:  # environment bug: record and end this episode
                eps[i].verification = {"error": f"step crashed: {type(e).__name__}: {e}"}
                envs[i].done = True
                continue
            eps[i].steps.append(Step(prompt=o["prompt"], prefill=prefill, action=action, token_ids=o["ids"],
                                     behavior_logprobs=o["logprobs"], result=res.observation, valid=res.valid,
                                     error=res.error, elapsed_s=time.time() - t1, finished_generation=o["finished"],
                                     t_wall=t1))
            hists[i].append((action, res.observation))
        if on_step:
            on_step(step, len(active))
    for i, ep in enumerate(eps):
        if envs[i] is not None and "error" not in ep.verification:
            finish(ep, envs[i], t0)
        else:
            ep.wall_s = time.time() - t0
    return eps
