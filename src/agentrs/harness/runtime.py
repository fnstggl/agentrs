"""Agent runtime: the main loop of one agent over one task.

    task -> requirements -> authorised identity (or provisioning request) -> observe (via gateway)
    -> decide (shared inference, pinned checkpoint) -> forecast consequences (temporal world model,
    stored before execution) -> execute (via gateway) -> record immediate outcome -> delayed outcomes
    are followed up by the outcome tracker -> verification -> experience for training

The runtime is model-agnostic: decisions come from the inference gateway; environment specifics
live in a ``TaskAdapter``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from agentrs.harness.gateway import ToolGateway
from agentrs.harness.inference import DecideRequest, InferenceClient
from agentrs.harness.provisioning import AccountProvisioningOrchestrator
from agentrs.harness.tasks import TaskRegistry
from agentrs.harness.types import ActionRequest
from agentrs.harness.verification import VerificationService


class TaskAdapter:
    environment = "abstract"

    async def observe(self, rt: "AgentRuntime", task) -> dict: ...
    def candidates(self, task) -> list[str]: ...
    def to_action(self, rt: "AgentRuntime", task, op: str, content: str) -> ActionRequest: ...
    def context(self, rt: "AgentRuntime", task, action: ActionRequest) -> dict: ...
    def evidence(self, rt: "AgentRuntime", task, action: ActionRequest, result) -> dict: ...


class SocialWebLabHelpAdapter(TaskAdapter):
    """'Get a helpful answer within a day' in SocialWebLab. The observation and prompt have exactly
    the format the content policy was trained on (``SocialWebLabContentEnv``)."""

    environment = "socialweblab"

    def __init__(self, lab_now, time_base: str):
        self.lab_now, self.time_base = lab_now, time_base

    async def observe(self, rt, task) -> dict:
        from agentrs.agent.belief import BeliefState
        from agentrs.environments.base import ElementState, Observation, PublicTask
        from agentrs.socialweblab.agent_tasks import FORUM_SPACES

        acc = rt.gateway.accounts.find_for(rt.agent_id, "socialweblab", ["create_post"])
        r = await rt.gateway.observe(rt.agent_id, task.id, "socialweblab", "read_user", {"handle": acc.external_id})
        n = int((r.observation.get("data") or {}).get("followers", 0)) if r.status == "executed" else 0
        els = [ElementState(eid="g1", role="heading", name="Ask the community for help"),
               ElementState(eid="g2", role="text", name="Spaces: " + ", ".join(FORUM_SPACES)),
               ElementState(eid="g3", role="text", name=f"Your followers: {n}")]
        obs = Observation(env_id="socialweblab_content", url="/new", title="New post", elements=els, step=0)
        b = BeliefState(task=PublicTask(task_id=task.id.split("#")[0], env_id="socialweblab_content",
                                        family="ask_for_help", goal=task.goal, max_steps=2, category="content"))
        b.reset(obs)
        return {"prompt": b.prompt(), "followers": n, "account": acc, "read_status": r.status}

    def candidates(self, task) -> list[str]:
        from agentrs.actions.schema import Action
        from agentrs.socialweblab.agent_tasks import FORUM_SPACES, op_for

        return [Action.submit(op_for(s), "").to_str() for s in FORUM_SPACES]

    def to_action(self, rt, task, op: str, content: str) -> ActionRequest:
        from agentrs.socialweblab.agent_tasks import space_for

        space = space_for(op) or "linux"
        title = (content.split("?")[0][:100] + ("?" if "?" in content[:101] else "")).strip() or content[:60] or "Question"
        return ActionRequest(agent_id=rt.agent_id, task_id=task.id, provider="socialweblab", operation="create_post",
                             params={"space": space, "title": title, "body": content or "(empty)",
                                     "client_ref": uuid.uuid4().hex})

    def context(self, rt, task, action) -> dict:
        from agentrs.data.observed_actor_action import context_text
        from agentrs.socialweblab.apps import SPACES

        space = action.params["space"]
        ctx = {"env": f"socialweblab/{SPACES[space][0]}", "space": space, "topic": ", ".join(SPACES[space][1][:5]),
               "author_followers": rt._obs.get("followers", 0)}
        return {"context_text": context_text(ctx, self.lab_now(), self.time_base), "time_base": self.time_base,
                "action_time": self.lab_now()}

    def evidence(self, rt, task, action, result) -> dict:
        return {"item_id": int(result.provider_object_id) if result.provider_object_id else None,
                "posted_at": self.lab_now(), "space": action.params["space"]}


class AgentRuntime:
    def __init__(self, agent_id: str, gateway: ToolGateway, inference: InferenceClient, tasks: TaskRegistry,
                 provisioning: AccountProvisioningOrchestrator, verification: VerificationService,
                 adapters: dict[str, TaskAdapter], world_model=None, sample: bool = True):
        self.agent_id, self.gateway, self.inference, self.tasks = agent_id, gateway, inference, tasks
        self.provisioning, self.verification, self.adapters = provisioning, verification, adapters
        self.world_model, self.sample = world_model, sample
        self.log: list[dict] = []
        self._obs: dict = {}

    async def ensure_identity(self, task) -> tuple[bool, str]:
        for cap in task.required_capabilities:
            provider, c = cap.split(":", 1)
            if self.gateway.accounts.find_for(self.agent_id, provider, [c]) is not None:
                continue
            req = self.provisioning.request(provider, [c2.split(":", 1)[1] for c2 in task.required_capabilities
                                                       if c2.startswith(provider + ":")],
                                            f"task {task.id}: {task.goal[:120]}", self.agent_id)
            if req.state == "CREDENTIALS_CONNECTED":
                req = await self.provisioning.verify(req.id)
            if req.state != "ACTIVE":
                return False, f"{provider}: {req.state} ({req.decision_reason})"
        return True, ""

    async def run_task(self, task, checkpoint: Optional[str] = None) -> dict:
        t0 = time.time()
        ad = self.adapters[task.environment]
        ok, why = await self.ensure_identity(task)
        if not ok:
            self.tasks.set_state(task.id, "waiting_human" if "PENDING" in why else "failed", {"reason": why})
            return {"task": task.id, "status": "blocked", "reason": why}
        self._obs = await ad.observe(self, task)
        cands = ad.candidates(task)
        resp = await self.inference.decide(DecideRequest(agent_id=self.agent_id, task_id=task.id, checkpoint=checkpoint,
                                                         prompt=self._obs["prompt"], op_candidates=cands,
                                                         sample=self.sample))
        d = resp.drafts[0]
        from agentrs.actions.schema import Action

        op = Action.parse(cands[d.op_index]).name
        action = ad.to_action(self, task, op, d.content)
        action.model_checkpoint = resp.checkpoint
        from agentrs.content.policy import op_text

        action.generation = {"context": self._obs["prompt"], "op_candidates": [op_text(Action.parse(c)) for c in cands],
                             "op_index": d.op_index, "op_logprob": d.op_logprob, "generation_prompt": d.gen_prompt,
                             "token_ids": d.token_ids, "token_logprobs": d.token_logprobs, "content": d.content,
                             "policy_checkpoint": resp.checkpoint,
                             "source": "policy" if self.sample else "policy_greedy",
                             "sampling_temperature": 1.0 if self.sample else None}
        ctx = ad.context(self, task, action)
        if self.world_model is not None:  # forecast BEFORE execution; stored with the action
            from agentrs.temporal.temporal_world_model import TemporalQuery

            q = TemporalQuery(context=ctx["context_text"], action=f'submit({op}, "{d.content}")', source_type="self",
                              actor_type="agent", goal=task.goal)
            ctx["forecast"] = self.world_model.predict(q, horizons=[3600.0, 86400.0]).to_dict()
        res = await self.gateway.submit(action, context=ctx)
        ev = ad.evidence(self, task, action, res) if res.status == "executed" else {"item_id": None}
        ev.update(status=res.status, reason=res.reason, batch_size=resp.batch_size)
        if res.status == "executed":
            self.tasks.set_state(task.id, "waiting_outcome")
            self.verification.record_attempt(task, self.agent_id, action.action_id, action.action_id, resp.checkpoint, ev)
        else:
            self.tasks.set_state(task.id, "failed" if res.status != "pending_approval" else "waiting_approval",
                                 {"reason": res.reason})
        rec = {"task": task.id, "status": res.status, "checkpoint": resp.checkpoint, "op": op, "content": d.content,
               "wall_s": time.time() - t0, "batch_size": resp.batch_size}
        self.log.append(rec)
        return rec

    async def loop(self, max_tasks: int, environments: Optional[list[str]] = None, split: Optional[str] = None,
                   checkpoint: Optional[str] = None) -> list[dict]:
        out = []
        for _ in range(max_tasks):
            agent = self.gateway.agents.get(self.agent_id)
            if agent is None or agent.status != "active" or self.gateway.policy.stopped():
                break
            t = self.tasks.claim(self.agent_id, environments, split)
            if t is None:
                break
            out.append(await self.run_task(t, checkpoint))
        return out
