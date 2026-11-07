"""Acting with delayed outcomes: decide → forecast (recorded before execution) → execute → hand the
created environment object to the sensor network, which follows it like any other action."""

from __future__ import annotations

from typing import Optional

from agentrs.agent.belief import BeliefState
from agentrs.environments.base import TaskSpec
from agentrs.sensors.observation_network import ObservationSensorNetwork
from agentrs.socialweblab.agent_tasks import SocialWebLabContentEnv, own_context, space_for
from agentrs.temporal.delayed_outcome_ledger import ActionRecord


def act_with_delayed_outcome(env: SocialWebLabContentEnv, task: TaskSpec, actor, net: ObservationSensorNetwork,
                             source_id: str = "swl-feed", group_prefix: str = "forum", allow_decline: bool = True) -> dict:
    obs = env.reset(task)
    belief = BeliefState(task=task.public())
    belief.reset(obs)
    actor.begin_episode(belief)
    cands = env.available_actions()
    if not allow_decline:
        cands = [c for c in cands if c.type.value != "finish"]
    dec = actor.decide(belief, cands)
    a = dec.action
    space = space_for(a.name or "") or task.private["space"]
    lab = env.lab
    sem = dec.semantic.model_dump() if getattr(dec, "semantic", None) is not None else None
    if sem is not None:
        sem["execution_status"] = "executed"
    rec = ActionRecord(
        source_id="self", source_type="self", actor_type="agent", actor_ref=task.private["account"],
        env_id="socialweblab", env_version="1", time_base=lab.time_base, action_time=lab.now(), observed_time=lab.now(),
        goal=task.goal, context_text=own_context(lab, env.account_uid(), space),
        context={"env": "socialweblab/forum", "space": space, "task_id": task.task_id, "correct_space": task.private["space"]},
        action_str=a.to_str(), action={"operation": a.name, "content": a.text, "parameters": {"space": space}},
        semantic=sem, policy_checkpoint=(sem or {}).get("policy_checkpoint") or getattr(actor, "name", "actor"),
        provenance={"selection": dec.selection}, permission={"allowlisted_operation": True},
        split=task.split, group=f"{group_prefix}:{space}")
    aid = net.register_self_action(rec)
    res = env.step(a)
    item: Optional[int] = res.info.get("created_item_id")
    if item is not None:
        net.link_self_action(aid, source_id, str(item))
    else:
        net.ledger.set_eligibility(aid, False, f"not executed: {res.error}")
    return {"action_id": aid, "item_id": item, "error": res.error, "selection": dec.selection, "space": space,
            "content": a.text, "wm": getattr(dec, "wm_prediction", None)}
