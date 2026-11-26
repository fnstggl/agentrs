"""The lifetime harness API (pre-v4 signature): a thin wrapper over the canonical loop `runtime.run_lifetime`.

    observe -> agent.act -> adapter.execute -> adapter.advance_time(cost + think | wait + think)
            -> verified reward (operator-side evaluator) -> agent.observe(Transition) -> periodic agent.update

Everything the harness did - contract `EventRecord`s per decision, the optional JSONL transcript, atomic
checkpoints (operator + evaluator + agent + not yet consumed batch + loop counters + transcript byte offset) every
`checkpoint_every_days`, ``resume=True`` in this or another process, per-phase compute accounting, verified
success days - is done by `runtime.run_lifetime` (the only act -> execute -> advance loop of the package); this
module maps the old keyword arguments onto it and re-exports `LifetimeResult`.

Behaviour changes of v4 (plan 2026-11-26 §2): the tool results of an action (ExecutionResult.observations) are now
delivered in the next observation batch (merged by event seq with the advance's observations; the old harness
dropped them), and every record carries extra["versions"] / ["schema"] / ["reward_mode"] (cam-event/4).
Reward mode: `reward_mode` if given, else the agent's cfg.reward_mode, else "legacy" when an evaluator is given
(the verified increments are delivered to the agent, as before), else "observation_only".

Separation: operator-side code. The agent gets only the adapter's records and - in reward_mode "legacy" - scalar
verified increments; `IndependentGoalResult.details` and evaluator objects stay here (`LifetimeResult.final`).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..runtime import DAY, CheckpointSpec, LifetimeResult, legacy_default_mode  # noqa: F401 (re-exported)
from ..runtime import run_lifetime as _run_lifetime
from .agent_protocol import Transition  # noqa: F401 (re-exported)


def run_lifetime(agent, adapter, evaluator=None, *, days: Optional[float] = None,
                 max_decisions: Optional[int] = None, think_s: float = 600.0, update_every: int = 0,
                 seeds: Optional[Sequence[int]] = None, stop_on_success: bool = False,
                 transcript: Optional[str] = None, operator=None, checkpoint_dir: Optional[str] = None,
                 checkpoint_every_days: Optional[float] = None, resume: bool = False,
                 model_version: Optional[Callable[[], int]] = None,
                 on_step: Optional[Callable[[int, Transition], None]] = None,
                 stop_after_decisions: Optional[int] = None, reward_mode: Optional[str] = None, **kw) -> LifetimeResult:
    """Run `agent` in `adapter` until every environment is done, `days` simulated days have passed, `max_decisions`
    rounds were made, or (``stop_on_success``) every environment reached verified success; see
    `runtime.run_lifetime` (extra keyword arguments - labels, label_store, learner, require_cam_eval, ... - are passed
    through)."""
    ck = None
    if checkpoint_dir or checkpoint_every_days or resume or operator is not None:
        ck = CheckpointSpec(checkpoint_dir, every_days=checkpoint_every_days, operator=operator, resume=resume)
    return _run_lifetime(agent, adapter, evaluator=evaluator, reward_mode=reward_mode or
                         legacy_default_mode(agent, evaluator), days=days, max_decisions=max_decisions,
                         think_s=think_s, update_every=update_every, seeds=seeds, stop_on_success=stop_on_success,
                         transcript=transcript, checkpoint=ck, model_version=model_version, on_step=on_step,
                         stop_after_decisions=stop_after_decisions, reward_source="auto", flush_reason=None, **kw)
