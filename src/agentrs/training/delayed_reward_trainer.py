"""Policy learning from delayed, verified rewards.

Rewards of our own content actions mature long after the action (e.g. "a helpful answer within one
day"). By then the policy has usually been updated, so each matured reward is applied to the
*stored* action — exact token ids, behaviour log-probabilities and the generating checkpoint
recorded in the ledger — with the clipped importance ratio pi_now / mu_behaviour (PPO), a staleness
limit, group-relative advantages (GRPO, per task) and an exact KL penalty to the reference policy.

Eligibility (never relaxed):
* only ``source_type == "self"`` actions — a third party's outcome is never a policy-gradient
  reward for our policy (it trains the world model instead);
* only content sampled from the policy at temperature 1 with stored behaviour log-probs;
* only *matured* rewards (pending ones wait; censored ones are dropped, never counted as 0);
* each reward is consumed once (training-clock record in the ledger).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from agentrs.content.policy import ContentGeneratingPolicy
from agentrs.content.training import ContentPolicyRLTrainer, ContentRLConfig, ContentRLStep
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.reward_maturation import semi_markov_return


class DelayedRewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = "help@86400"
    rl: ContentRLConfig = Field(default_factory=ContentRLConfig)
    max_staleness: int = 6  # policy versions between behaviour and update
    gamma: float = 1.0  # semi-Markov discount per tau of delay between action and reward
    tau_s: float = 86400.0
    trainer_name: str = "policy-delayed-rl"


# verifier(action row) -> (status, value, maturity_time); status in pending | matured | censored
Verifier = Callable[[dict], tuple[str, Optional[float], Optional[float]]]


def record_matured_rewards(ledger: DelayedOutcomeLedger, verifier: Verifier, objective: str,
                           source_types: Sequence[str] = ("self",)) -> dict:
    stats: dict[str, int] = defaultdict(int)
    for a in ledger.actions():
        if a["source_type"] not in source_types:
            continue
        if any(r["objective"] == objective for r in ledger.rewards(a["action_id"])):
            stats["already"] += 1
            continue
        status, value, t = verifier(a)
        stats[status] += 1
        if status == "matured":
            ledger.record_reward(a["action_id"], "delayed", float(value), env_time=t, maturity_time=t,
                                 checkpoint=a["policy_checkpoint"], objective=objective,
                                 idem=f"{a['action_id']}:{objective}")
    return dict(stats)


class DelayedRewardTrainer:
    def __init__(self, cp: ContentGeneratingPolicy, ledger: DelayedOutcomeLedger, cfg: DelayedRewardConfig,
                 ref: Optional[ContentGeneratingPolicy] = None, seed: int = 0):
        self.cp, self.ledger, self.cfg = cp, ledger, cfg
        self.rl = ContentPolicyRLTrainer(cp, cfg.rl, ref=ref, seed=seed)
        self.rng = random.Random(seed)
        self.version_of: dict[str, int] = {cp.checkpoint_id: 0}

    def _consumed(self) -> set[str]:
        rows = self.ledger._q("SELECT action_id FROM ingest WHERE trainer=?", (self.cfg.trainer_name,))
        return {r["action_id"] for r in rows}

    def collect(self) -> tuple[list[ContentRLStep], dict]:
        consumed = self._consumed()
        rejected: dict[str, int] = defaultdict(int)
        rows = []
        cur_v = self.rl.version
        for a in self.ledger.actions():
            if a["action_id"] in consumed:
                continue
            rew = [r for r in self.ledger.rewards(a["action_id"]) if r["objective"] == self.cfg.objective]
            if not rew:
                continue
            if a["source_type"] != "self":
                rejected["third_party_outcome_not_a_policy_reward"] += 1
                continue
            sa = a["semantic"]
            if not sa or not sa.get("token_ids") or sa.get("token_logprobs") is None or sa.get("op_logprob") is None:
                rejected["missing_behaviour_logprobs"] += 1
                continue
            if sa.get("source") != "policy" or sa.get("sampling_temperature") not in (None, 1.0):
                rejected["not_sampled_from_policy"] += 1
                continue
            bv = self.version_of.get(sa.get("policy_checkpoint"), None)
            if bv is None or cur_v - bv > self.cfg.max_staleness:
                rejected["too_stale_or_unknown_checkpoint"] += 1
                continue
            r = rew[0]
            R = semi_markov_return([(r["maturity_time"], r["value"])], a["action_time"], self.cfg.gamma, self.cfg.tau_s)
            rows.append((a, sa, R))
        groups: dict[str, list[float]] = defaultdict(list)
        for a, _, R in rows:
            groups[(a["context"] or {}).get("task_id", a["action_id"])].append(R)
        allR = [R for *_, R in rows]
        steps = []
        for a, sa, R in rows:
            g = groups[(a["context"] or {}).get("task_id", a["action_id"])]
            if self.cfg.rl.advantage == "group" and len(g) > 1:
                A = (R - float(np.mean(g))) / (float(np.std(g)) + 1e-6)
            else:
                A = (R - float(np.mean(allR))) / (float(np.std(allR)) + 1e-6) if len(allR) > 1 else 0.0
            steps.append(ContentRLStep(prompt=sa["context"], op_cands=sa["op_candidates"], op_index=sa["op_index"],
                                       op_logprob_old=sa["op_logprob"], gen_prompt=sa["generation_prompt"],
                                       token_ids=sa["token_ids"], token_logprobs_old=sa["token_logprobs"],
                                       advantage=float(A), checkpoint=sa.get("policy_checkpoint")))
            steps[-1].action_id = a["action_id"]  # type: ignore[attr-defined]
        return steps, {"n_matured": len(rows), "mean_reward": float(np.mean(allR)) if allR else None, **rejected}

    def update(self) -> dict:
        steps, info = self.collect()
        if not steps:
            return {"updated": False, **info}
        acc: dict[str, list[float]] = defaultdict(list)
        self.cp.policy.train()
        for _ in range(self.cfg.rl.epochs):
            self.rng.shuffle(steps)
            for i in range(0, len(steps), self.cfg.rl.minibatch_size):
                mb = steps[i:i + self.cfg.rl.minibatch_size]
                if all(abs(s.advantage) < 1e-9 for s in mb):
                    continue
                loss, st = self.rl.loss(mb)
                upd = self.rl.trainer.step(loss)
                for k, v in {**st, **upd}.items():
                    acc[k].append(v)
        self.cp.policy.eval()
        self.rl.version += 1
        self.cp.checkpoint_id = f"policy-delayed-rl-v{self.rl.version}"
        self.version_of[self.cp.checkpoint_id] = self.rl.version
        self.ledger.mark_ingested([s.action_id for s in steps], self.cfg.trainer_name, self.rl.trainer.global_step)
        return {"updated": bool(acc), "n_steps": len(steps), "checkpoint": self.cp.checkpoint_id, **info,
                **{k: float(np.nanmean(v)) for k, v in acc.items()}}
