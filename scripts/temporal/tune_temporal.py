"""Validation-only tuning of the temporal world model's optimisation (steps, learning rate, weight
decay). Uses corpus 0's TRAINING window only: actions in its last 1.5 days are the validation set,
earlier ones the tuning-train set. The test window and held-out environments are never touched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVENTS, HORIZONS, RUNS, Timer, make_wm, save_json, threads  # noqa: E402

from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.socialweblab.engine import LAB_EPOCH  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.temporal_evaluation import evaluate_temporal, model_predict_fn, summarize  # noqa: E402
from agentrs.training.common import OptimConfig  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=100)
    a = ap.parse_args()
    threads()
    T = Timer()
    L = DelayedOutcomeLedger(RUNS / "corpus_s0" / "net" / "ledger.sqlite")
    b = TemporalTrainingDataBuilder(L, lab_temporal_spec())
    cut = LAB_EPOCH + 7.5 * 86400
    ids_tr = [r["action_id"] for r in L._q("SELECT action_id FROM actions WHERE split='train' AND action_time<?", (cut,))]
    ids_va = [r["action_id"] for r in L._q("SELECT action_id FROM actions WHERE split='train' AND action_time>=?", (cut,))]
    tr = b.build(action_ids=ids_tr, cutoffs="observations", max_cutoffs=2)
    va = b.build(action_ids=ids_va, cutoffs=[0])
    m = make_wm(0)
    cfg = TemporalTrainConfig(steps=a.eval_every, batch_size=16, seed=0,
                              optim=OptimConfig(lr=a.lr, weight_decay=a.wd, total_steps=a.steps, schedule="cosine",
                                                warmup_steps=30))
    tt = TemporalWorldModelTrainer(m, cfg, name="tune")
    curve = []
    for k in range(a.steps // a.eval_every):
        tt.train(tr, steps=a.eval_every)
        s = summarize(evaluate_temporal(model_predict_fn(m), va, HORIZONS, EVENTS), EVENTS)
        curve.append({"step": (k + 1) * a.eval_every, "val_brier": s["mean_brier"], "val_auroc": s["mean_auroc"],
                      "val_ece": (s["overall"] or {}).get("ece")})
        print(curve[-1], flush=True)
    save_json(f"tune_lr{a.lr}_wd{a.wd}.json", {"lr": a.lr, "wd": a.wd, "n_train": len(tr), "n_val": len(va),
                                               "curve": curve, "wall_s": T.s()})


if __name__ == "__main__":
    main()
