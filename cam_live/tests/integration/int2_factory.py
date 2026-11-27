"""Agent factory for the loop_service test: records what ``runtime.loop_service`` passes to it."""
from __future__ import annotations

from typing import Any, Dict

SEEN: Dict[str, Any] = {}


def build(cfg: Any = None, run_dir: str = "", clock: Any = None, task: Any = None, verifier: Any = None) -> Any:
    from cam_live.agent import build_agent
    SEEN.update(cfg=cfg, run_dir=run_dir, clock=clock, task=task, verifier=verifier)
    agent = build_agent(cfg, run_dir, clock, task, verifier, durable=False, use_backbone=False, planner=False)
    SEEN["agent"] = agent
    return agent
