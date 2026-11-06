"""Running agents on task sets and aggregating verified outcomes."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import torch

from agentrs.agent.runtime import Actor, run_episode
from agentrs.data.schemas import EpisodeRecord, Provenance, Source
from agentrs.environments.base import Environment, TaskSpec
from agentrs.evaluation.metrics import episode_metrics
from agentrs.security.permissions import GuardedEnvironment, PermissionPolicy


def make_env(env_id: str = "miniweb", guarded: bool = True, policy: Optional[PermissionPolicy] = None,
             **kw) -> Environment:
    if env_id == "miniweb":
        from agentrs.environments.miniweb.core import MiniWebEnv

        env: Environment = MiniWebEnv(**kw)
    elif env_id == "localweb":
        from agentrs.environments.web.playwright_env import LocalWebEnv

        env = LocalWebEnv(**kw)
    elif env_id == "content":
        from agentrs.environments.content.env import ContentEnv

        env = ContentEnv(**kw)
    elif env_id.startswith("browsergym"):
        from agentrs.environments.browsergym_adapter import BrowserGymEnv

        env = BrowserGymEnv(**kw)
    else:
        raise ValueError(f"unknown env {env_id}")
    if guarded:
        env = GuardedEnvironment(env, policy or PermissionPolicy(step_timeout_seconds=0 if env_id in ("miniweb", "content") else 30))
    return env


def run_tasks(env: Environment, tasks: Sequence[TaskSpec], actor: Actor, provenance: Optional[Provenance] = None,
              checkpoints: Optional[dict] = None, repeats: int = 1, memory=None,
              compute_models: Sequence = ()) -> list[EpisodeRecord]:
    eps = []
    for r in range(repeats):
        for t in tasks:
            flops0 = sum(getattr(m, "flops_counter", 0) for m in compute_models)
            ep = run_episode(env, t, actor, provenance=provenance or Provenance(source=Source.AGENT,
                                                                                source_id=getattr(actor, "name", "")),
                             checkpoints=checkpoints, memory=memory)
            ep.metadata["difficulty"] = t.difficulty
            ep.metadata["layout"] = t.layout
            ep.metadata["repeat"] = r
            ep.metadata["inference_flops"] = sum(getattr(m, "flops_counter", 0) for m in compute_models) - flops0
            eps.append(ep)
    return eps


@torch.no_grad()
def evaluate_actor(env: Environment, tasks: Sequence[TaskSpec], actor: Actor, repeats: int = 1,
                   compute_models: Sequence = (), memory=None) -> tuple[dict, list[EpisodeRecord]]:
    t0 = time.time()
    eps = run_tasks(env, tasks, actor, repeats=repeats, compute_models=compute_models, memory=memory)
    m = episode_metrics(eps)
    m["eval_wall_time_s"] = time.time() - t0
    m["inference_flops"] = float(sum(e.metadata.get("inference_flops", 0) for e in eps))
    m["inference_flops_per_success"] = m["inference_flops"] / max(1, sum(bool(e.success) for e in eps))
    return m, eps

