"""Participants of a self-play civilisation behind the `crucible.experience` learner protocol.

Every participant is driven by `crucible.experience.loop.run_learners` through the universal interaction contract:
``act(observation, tools) -> action | None`` then ``observe(reward, result, next_observation, done)`` and
``update()``. None of them holds a handle on the world, the environment, the evaluator or another participant:
what they know they received as their own observations and action results.

* `LearnerAgent`    one identity of a neural `PolicyGroup` (current learner, historical checkpoint, independent
                    or held-out policy: the group decides whether it learns); its training signal is its own
                    `Objective` over its own observations; the environment's reward is checked to be 0.0 and is
                    never used;
* `SemanticAgent`   a compact semantic generalist (`crucible.semantic.participants.SemanticParticipant`, class 4,
                    workstream D) acting with the same tools; its model's outputs are recorded in the world
                    (`semantic_inference`) when the world has that module.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .knowledge import Knowledge
from .objectives import Objective, _cash
from .policy import PolicyGroup, state_vector


def _read_only(obs: dict | None, tool: str | None) -> bool:
    if not tool or not obs:
        return False
    for t in obs.get("tools") or []:
        if t.get("name") == tool:
            return bool((t.get("annotations") or {}).get("readOnlyHint"))
    return False


class LearnerAgent:
    kind = "learner"

    def __init__(self, principal: str, group: PolicyGroup, objective: Objective, *, role: str = "current",
                 chooser=None):
        self.principal, self.group, self.objective, self.role = principal, group, objective, role
        self.chooser = chooser                # a scripted demonstrator (demonstration runs only; labelled)
        self.demos: list[dict] = []
        self._decisions: list[dict] = []
        self.kn = Knowledge(principal)
        self.h = None
        self.prev_obs: dict | None = None
        self.prev_total: int | None = None
        self.signal_ema = 0.0
        self.last_action: dict | None = None
        self.last_rec: dict | None = None
        self.nonzero_env_rewards = 0
        self.history: list[dict] = []         # its own decisions and their results (for analysis by the operator)
        self.signals: list[tuple[int, float, float]] = []
        if principal not in group.identities:
            group.identities.append(principal)

    # ------------------------------------------------------------------ protocol
    def info(self) -> dict:
        return {"name": f"multiagent:{self.group.gid}", "version": self.group.version,
                "params_hash": self.group.hash, "algo": self.group.algo, "arch": self.group.arch,
                "role": self.role}

    def act(self, obs: dict, tools: list[dict], explore: bool = True) -> dict | None:
        self.kn.ingest_observation(obs)
        x = state_vector(obs, self.kn, self.prev_total, obs.get("last_result"), self.signal_ema)
        if self.chooser is not None:
            self.chooser.see(obs, self.kn)
        action, rec, h = self.group.decide(self.principal, obs, self.kn, x, self.h, explore, chooser=self.chooser)
        self.h = h
        self.last_action, self.last_rec = action, rec
        if self.chooser is not None:
            self.demos.append(rec)
        else:
            self.group.record(rec)
            if len(self.history) % 4 == 0 and len(self._decisions) < 120:
                self._decisions.append(rec)     # situations kept for behavioural probes (operator side)
        self.prev_obs = obs
        self.prev_total = _cash(obs)
        return action

    def observe(self, reward: float, result: dict | None, next_obs: dict, done: bool) -> None:
        if float(reward) != 0.0:
            self.nonzero_env_rewards += 1          # never used; counted for the shared-reward-channel audit
        act = self.last_action
        ro = _read_only(self.prev_obs, act.get("tool") if act else None)
        # its own action's result first: what that result names for the first time is what the action brought
        # into being (its own things), even when the next observation's events mention it too
        self.kn.ingest_result(act, result, ro)
        self.kn.ingest_observation(next_obs)
        meta = (self.last_rec or {}).get("meta") or {}
        if act and result is not None and result.get("ok") and meta.get("fn"):
            self.kn.note_value("target", meta["fn"], self.kn.t, mine=False)
            self.kn.note_value("fn", meta["fn"], self.kn.t, mine=False)
        if self.chooser is not None:
            self.chooser.result(act, result, self.kn)
        ext, intr = self.objective.signal(self.prev_obs, next_obs, act, result, self.kn, ro)
        sig = ext + intr
        self.group.add_signal(self.principal, sig)
        self.signal_ema = 0.9 * self.signal_ema + 0.1 * sig
        t = int(next_obs.get("t", 0))
        self.signals.append((t, float(ext), float(intr)))
        self.history.append({"t": t, "tool": act.get("tool") if act else None,
                             "ok": None if result is None else bool(result.get("ok")),
                             "error": None if result is None or result.get("ok") else
                             ((result.get("error") or {}).get("code") if isinstance(result.get("error"), dict)
                              else str(result.get("error"))[:40]),
                             "read_only": ro, "version": self.group.version, "ext": float(ext),
                             "intr": float(intr)})

    def update(self) -> dict:
        return self.group.maybe_update()

    def reset(self, agent: str) -> None:
        self.h = None


class SemanticAgent:
    """A semantic generalist participant (see the module docstring). It decides every `every`-th step (its model
    calls are the expensive part of a civilisation step) and waits otherwise."""
    kind = "semantic"

    def __init__(self, principal: str, participant: Any, *, every: int = 4, seed: int = 0):
        self.principal, self.p, self.every = principal, participant, max(1, int(every))
        self.p.reset(principal, np.random.default_rng([int(seed), 0x5E]))
        self.k = 0
        self.last_action: dict | None = None
        self.history: list[dict] = []
        self.role = "semantic"
        self.nonzero_env_rewards = 0

    def info(self) -> dict:
        svc = getattr(getattr(self.p, "group", None), "service", None)
        mh = svc.model_hash() if svc is not None and hasattr(svc, "model_hash") else None
        return {"name": f"semantic:{getattr(svc, 'model_id', 'model')}", "version": "5.0", "params_hash": mh,
                "role": "semantic"}

    def act(self, obs: dict, tools: list[dict], explore: bool = True) -> dict | None:
        self.k += 1
        self.last_action = None
        if (self.k - 1) % self.every:
            return None
        o4 = {"time": obs.get("t", 0), "principal": obs.get("agent"),
              "balances": {k.rsplit(".", 1)[-1]: v for k, v in (obs.get("numbers") or {}).items()
                           if k.startswith("bank.balances.")},
              "resources": obs.get("resources") or {}, "objective": obs.get("objective"),
              "events": [{"seq": e.get("seq"), "t": e.get("t"), "type": e.get("type"), "actor": e.get("actor"),
                          "payload": e.get("payload")} for e in (obs.get("events") or [])[-30:]]}
        act = self.p.act(o4, tools)
        self.last_action = act
        return act

    def observe(self, reward: float, result: dict | None, next_obs: dict, done: bool) -> None:
        if float(reward) != 0.0:
            self.nonzero_env_rewards += 1
        if self.last_action is not None and result is not None:
            self.p.observe_result(self.last_action, result)
            own = (getattr(self.p, "history", None) or [{}])[-1]       # the participant's own record of the call
            err = result.get("error")
            self.history.append({"t": int(next_obs.get("t", 0)), "tool": self.last_action.get("tool"),
                                 "ok": bool(result.get("ok")), "read_only": bool(own.get("read_only")),
                                 "error": None if result.get("ok") else
                                 ((err or {}).get("code") if isinstance(err, dict) else str(err)[:40])})
