"""Agent-facing API of Crucible (``crucible-api/1``). See docs/AGENT_API.md.

* `WorldServer` (native Python API) and its public types: `ActorIdentity`, `Observation`, `ActionResult`,
  `EvaluationResult`, `ApiError` and subclasses.
* `render`: deterministic text rendering for LLM agents and `parse_action`.
* `CrucibleEnv` (Gymnasium), `CrucibleParallelEnv` (PettingZoo), `make_app` (FastAPI: OpenEnv + MCP + /v1),
  `MCPHandler` (stdio/HTTP MCP) are imported lazily so the core API has no optional dependencies.
"""
from __future__ import annotations

from .local import (API_VERSION, ActionResult, ActorIdentity, ApiError, AuthError, ConflictError,
                    EvaluationResult, ForbiddenError, NotFoundError, Observation, WorldServer)
from .render import parse_action, render_observation, render_tools

_LAZY = {"CrucibleEnv": ".gym_env", "CrucibleParallelEnv": ".pettingzoo_env", "make_app": ".http_server",
         "serve": ".http_server", "MCPHandler": ".mcp_server"}

__all__ = ["API_VERSION", "WorldServer", "ActorIdentity", "Observation", "ActionResult", "EvaluationResult",
           "ApiError", "AuthError", "ConflictError", "ForbiddenError", "NotFoundError", "parse_action",
           "render_observation", "render_tools", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(name)
