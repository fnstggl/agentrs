"""Training data pipeline and learner for experience gathered through the harness.

* World model: matured observations of executed actions (exact content, elapsed time, censoring)
  → ``TemporalExample`` batches → progressive updates of the temporal world model.
* Policy: verified task rewards recorded in the ledger → ``ContentRLStep`` batches with the stored
  behaviour log-probs and checkpoint ids → clipped importance-ratio RL (``DelayedRewardTrainer``).
* Imitation: only from demonstrations with provenance (``demonstrations``), kept separate.

The learner always trains a **copy** of the production weights (live inference workers keep their
fixed checkpoint), writes a candidate checkpoint, and promotion happens only through held-out
evaluation (``promote_if_better``) — never because a training loss went down.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable, Optional

from agentrs.checkpoints.registry import CheckpointRegistry, PromotionGate
from agentrs.content.policy import ContentGeneratingPolicy
from agentrs.training.common import param_change, param_snapshot, save_checkpoint
from agentrs.training.delayed_reward_trainer import DelayedRewardConfig, DelayedRewardTrainer


class HarnessLearner:
    def __init__(self, workdir: str | Path, ledger, production: ContentGeneratingPolicy, registry: CheckpointRegistry,
                 rl_cfg: Optional[DelayedRewardConfig] = None, reference: Optional[ContentGeneratingPolicy] = None):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger, self.production, self.registry = ledger, production, registry
        self.rl_cfg = rl_cfg or DelayedRewardConfig(objective="task:socialweblab_help_within")
        self.reference = reference or ContentGeneratingPolicy(copy.deepcopy(production.policy), production.cfg, "reference")
        self.trainer_state: Optional[DelayedRewardTrainer] = None
        self.version = 0

    def _known_checkpoints(self) -> dict[str, int]:
        ids = {r["policy_checkpoint"] for r in self.ledger._q("SELECT DISTINCT policy_checkpoint FROM actions")}
        return {c: 0 for c in ids if c}

    def training_batch(self) -> dict[str, Any]:
        """Inspect the RL batch the next update would use (for tests / dashboards)."""
        cand = ContentGeneratingPolicy(copy.deepcopy(self.production.policy), self.production.cfg, self.production.checkpoint_id)
        t = DelayedRewardTrainer(cand, self.ledger, self.rl_cfg, ref=self.reference)
        t.version_of.update(self._known_checkpoints())
        steps, info = t.collect()
        return {"steps": steps, "info": info}

    def train_candidate(self) -> dict[str, Any]:
        cand = ContentGeneratingPolicy(copy.deepcopy(self.production.policy), self.production.cfg,
                                       self.production.checkpoint_id)
        t = DelayedRewardTrainer(cand, self.ledger, self.rl_cfg, ref=self.reference)
        t.version_of.update(self._known_checkpoints())
        snap = param_snapshot(cand.policy)
        upd = t.update()
        if not upd.get("updated"):
            return {"candidate": None, **{k: v for k, v in upd.items() if not isinstance(v, dict)}}
        self.version += 1
        cid_local = f"harness-policy-v{self.version}"
        cand.checkpoint_id = cid_local
        path = save_checkpoint(self.dir / cid_local, cand.policy, t.rl.trainer, extra={"checkpoint_id": cid_local,
                                                                                      "parent": self.production.checkpoint_id})
        prod = self.registry.production("policy")
        rid = self.registry.register("policy", str(path), parent=prod.id if prod else None,
                                     metrics={"n_steps": upd.get("n_steps"), "mean_reward": upd.get("mean_reward"),
                                              "param_change": param_change(cand.policy, snap)},
                                     notes=cid_local)
        return {"candidate": cand, "registry_id": rid, "checkpoint_id": cid_local, "param_change": param_change(cand.policy, snap),
                **{k: v for k, v in upd.items() if isinstance(v, (int, float, str, bool, type(None)))}}

    def promote_if_better(self, candidate: dict, evaluate: Callable[[str], dict], backend,
                          gate: Optional[PromotionGate] = None) -> dict[str, Any]:
        """``evaluate(checkpoint_id) -> {"success_rate": ...}`` on held-out tasks (same conditions for both)."""
        gate = gate or PromotionGate(metric="success_rate", min_improvement=0.0)
        backend.add(candidate["checkpoint_id"], candidate["candidate"])
        t0 = time.time()
        m_cand = evaluate(candidate["checkpoint_id"])
        m_prod = evaluate(self.production.checkpoint_id)
        ok, why = gate.decide(m_cand, m_prod)
        self.registry.update_metrics(candidate["registry_id"], {"heldout": m_cand, "production_heldout": m_prod})
        if ok:
            self.registry.promote(candidate["registry_id"], detail=why)
            backend.production = candidate["checkpoint_id"]
            self.production = candidate["candidate"]
        else:
            self.registry.reject(candidate["registry_id"], reason=why)
        return {"promoted": ok, "reason": why, "candidate": m_cand, "production": m_prod, "eval_wall_s": time.time() - t0}
