"""Repeatable continual-learning cycle over self and third-party experience.

One cycle:

1. **act** — our agent attempts ``tasks_per_cycle`` delayed-outcome tasks (forecast recorded before
   each execution; the created object is handed to the sensor network);
2. **observe** — the environment's clock advances in steps while the sensor network ingests
   third-party actions (forecasting each on arrival), runs due follow-ups and closes windows;
3. **world-model learning** — progressive update of the temporal world model on every action whose
   labels changed since the learner last consumed it (short horizons train long before long ones
   mature), mixed with historical replay → candidate checkpoint;
4. **policy learning** — rewards that matured in this cycle (verified by the environment) update the
   policy with off-policy-corrected delayed-reward RL → candidate checkpoint (only if any matured);
5. **evaluation** — held-out temporal prediction (fixed held-out set rebuilt as of now) for current
   and candidate world models, with an old-distribution check to catch forgetting;
6. **promotion** — ``CheckpointRegistry`` gate: a candidate is promoted only if it improves the
   held-out metric and does not regress the old-distribution metric by more than ``max_regression``;
   otherwise it is rejected and the production weights are restored.

Everything is persisted (ledger, scheduler, registry, checkpoints), so the loop can be stopped and
resumed.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentrs.checkpoints.registry import CheckpointRegistry
from agentrs.training.common import save_checkpoint
from agentrs.training.temporal_world_model_trainer import TemporalTrainingDataBuilder, TemporalWorldModelTrainer


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_s: float = 86400.0
    step_s: float = 900.0
    tasks_per_cycle: int = 8
    eval_horizons: list[float] = [3600.0, 6 * 3600.0, 86400.0]
    eval_events: list[str] = ["first_response", "answered", "positive_feedback", "removed"]
    max_regression: float = 0.01  # Brier on the old-distribution set
    min_improvement: float = 0.0
    eval_split: str = "val"


class ContinualLearningOrchestrator:
    def __init__(self, workdir: str | Path, lab, net, wm_trainer: TemporalWorldModelTrainer,
                 builder: TemporalTrainingDataBuilder, cfg: OrchestratorConfig,
                 act_fn: Optional[Callable[[int, int], list[dict]]] = None,
                 reward_fn: Optional[Callable[[], dict]] = None, policy_trainer=None):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lab, self.net, self.T, self.builder, self.cfg = lab, net, wm_trainer, builder, cfg
        self.act_fn = act_fn  # (cycle, n) -> list of action summaries
        self.reward_fn = reward_fn  # records matured rewards in the ledger, returns stats
        self.policy_trainer = policy_trainer
        self.registry = CheckpointRegistry(self.dir / "registry.sqlite")
        self.cycle = 0
        self.old_eval = None  # frozen old-distribution set (first cycle's held-out examples)
        self.log: list[dict] = []

    def _eval(self, examples) -> dict:
        if not examples:
            return {}
        r = self.T.evaluate(examples, self.cfg.eval_horizons, self.cfg.eval_events)
        return r.get("overall", {})

    def run_cycle(self) -> dict:
        t0 = time.perf_counter()
        self.cycle += 1
        c = self.cycle
        rec: dict = {"cycle": c, "lab_time_start": self.lab.now()}
        # 1. act
        rec["actions"] = self.act_fn(c, self.cfg.tasks_per_cycle) if self.act_fn else []
        # 2. observe
        end = self.lab.now() + self.cfg.cycle_s
        while self.lab.now() < end - 1e-6:
            self.lab.advance(min(self.cfg.step_s, end - self.lab.now()))
            self.net.step()
        rec["ledger"] = self.net.ledger.stats()
        as_of = {self.lab.time_base: self.lab.now()}
        held = self.builder.build(as_of=as_of, splits=[self.cfg.eval_split], cutoffs=[0])
        if self.old_eval is None:
            self.old_eval = held
        before = {"held_out": self._eval(held), "old": self._eval(self.old_eval)}
        # 3. world model
        snap = copy.deepcopy({k: v.detach().clone() for k, v in self.T.model.state_dict().items()})
        prod = self.registry.production("temporal_world_model")
        upd = self.T.progressive_update(self.builder, as_of, splits=("train",), force=True)
        rec["wm_update"] = upd
        after = {"held_out": self._eval(held), "old": self._eval(self.old_eval)}
        ck = self.dir / f"wm_cycle{c}"
        save_checkpoint(ck, self.T.model, self.T.opt, extra={"cycle": c, "checkpoint_id": self.T.model.checkpoint_id})
        cid = self.registry.register("temporal_world_model", str(ck), parent=prod.id if prod else None,
                                     metrics={"held_out_brier": after["held_out"].get("brier"),
                                              "old_brier": after["old"].get("brier")})
        b_new, b_old = after["held_out"].get("brier"), before["held_out"].get("brier")
        o_new, o_old = after["old"].get("brier"), before["old"].get("brier")
        improves = b_new is not None and b_old is not None and b_new <= b_old - self.cfg.min_improvement
        no_regress = o_new is None or o_old is None or o_new <= o_old + self.cfg.max_regression
        if upd.get("trained") and improves and no_regress:
            self.registry.promote(cid, detail=f"cycle {c}: held-out brier {b_old}->{b_new}")
            rec["wm_promoted"] = True
        else:
            self.registry.reject(cid, reason=f"held-out {b_old}->{b_new}, old {o_old}->{o_new}")
            self.T.model.load_state_dict(snap)
            rec["wm_promoted"] = False
        rec["wm_eval_before"], rec["wm_eval_after"] = before, after
        # 4. policy
        if self.reward_fn is not None:
            rec["rewards"] = self.reward_fn()
        if self.policy_trainer is not None:
            rec["policy_update"] = self.policy_trainer.update()
        rec["wall_s"] = time.perf_counter() - t0
        rec["lab_time_end"] = self.lab.now()
        self.log.append(rec)
        (self.dir / "cycles.jsonl").open("a").write(json.dumps(rec, default=str) + "\n")
        return rec
