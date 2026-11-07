"""Experiments C and D on the observational corpora.

C — self experience only vs self + permitted third-party observations (same model, same number
    of gradient steps). Arms: self_only, third_only, self_plus_third, self_plus_third_matched (third-
    party subsampled to the size of the self set: same volume as self_only, different composition).
    Evaluation: our own later posts (self_test), third parties' later posts (third_test), and unseen
    environments / action types (test_env: held-out forum spaces and the whole code-review app,
    i.e. ``submit_patch`` actions never seen in training).
D — volume vs diversity of observational data at a fixed example budget and fixed steps:
    repetitive (all examples from the two most frequent strata — more data, less variety), uniform,
    stratified, novelty (farthest-point on hashed n-grams), uncertainty (ensemble disagreement of a
    model trained on a small seed set), and all data (reference).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVENTS, HORIZONS, RUNS, TUNED, Timer, make_wm, save_json, threads  # noqa: E402

from agentrs.sensors.information_sampling import coverage, select  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.temporal_evaluation import evaluate_temporal, marginal_predict_fn, model_predict_fn, summarize  # noqa: E402
from agentrs.training.common import OptimConfig  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)


def ev(fn, sets):
    return {k: summarize(evaluate_temporal(fn, v, HORIZONS, EVENTS, ["responses", "views"]), EVENTS) for k, v in sets.items()
            if v}


def train_eval(train, sets, steps, seed, name):
    m = make_wm(seed)
    cfg = TemporalTrainConfig(steps=steps, batch_size=16, seed=seed,
                              optim=OptimConfig(lr=TUNED["lr"], weight_decay=TUNED["wd"], total_steps=steps,
                                                schedule="cosine", warmup_steps=30))
    info = TemporalWorldModelTrainer(m, cfg, name=name).train(train)
    return m, {"n_train": len(train), "train": {k: info.get(k) for k in ("last_loss", "seconds")},
               "eval": ev(model_predict_fn(m), sets)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["C", "D"], required=True)
    ap.add_argument("--corpus", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=TUNED["steps"])
    ap.add_argument("--budget", type=int, default=150)
    a = ap.parse_args()
    threads()
    T = Timer()
    L = DelayedOutcomeLedger(RUNS / f"corpus_s{a.corpus}" / "net" / "ledger.sqlite")
    b = TemporalTrainingDataBuilder(L, lab_temporal_spec())
    rng = random.Random(a.seed)
    res = {"exp": a.exp, "corpus": a.corpus, "seed": a.seed, "steps": a.steps, "arms": {}}
    out = f"exp{a.exp}_c{a.corpus}_s{a.seed}.json"
    if a.exp == "C":
        self_tr = b.build(splits=["train"], source_types=["self"], cutoffs="observations", max_cutoffs=2)
        third_tr = b.build(splits=["train"], source_types=["third_party"], cutoffs="observations", max_cutoffs=2)
        sets = {"self_test": b.build(splits=["test"], source_types=["self"], cutoffs=[0]),
                "third_test": b.build(splits=["test"], source_types=["third_party"], cutoffs=[0]),
                "test_env": b.build(splits=["test_env"], cutoffs=[0])}
        res["sizes"] = {"self_train": len(self_tr), "third_train": len(third_tr), **{k: len(v) for k, v in sets.items()}}
        res["arms"]["marginal_km_self"] = {"eval": ev(marginal_predict_fn([e for e in self_tr if e.cutoff_elapsed_s == 0],
                                                                        EVENTS, ["responses", "views"]), sets)}
        res["arms"]["marginal_km_all"] = {"eval": ev(marginal_predict_fn(
            [e for e in self_tr + third_tr if e.cutoff_elapsed_s == 0], EVENTS, ["responses", "views"]), sets)}
        arms = {"self_only": self_tr, "third_only": third_tr, "self_plus_third": self_tr + third_tr,
                "self_plus_third_matched": self_tr + rng.sample(third_tr, min(len(third_tr), len(self_tr)))}
        for name, data in arms.items():
            _, r = train_eval(data, sets, a.steps, a.seed, name)
            res["arms"][name] = r
            print(name, len(data), {k: round(v["mean_brier"], 4) for k, v in r["eval"].items()}, flush=True)
            save_json(out, res)
    else:
        pool = b.build(splits=["train"], source_types=["third_party"], cutoffs=[0])
        sets = {"test": b.build(splits=["test"], cutoffs=[0]), "test_env": b.build(splits=["test_env"], cutoffs=[0])}
        cov = Counter(e.group for e in pool)
        top2 = [g for g, _ in cov.most_common(2)]
        res["pool"] = {"n": len(pool), "coverage": coverage(pool), "top2": top2}
        sel = {"repetitive_volume": [i for i, e in enumerate(pool) if e.group in top2]}
        for s in ("uniform", "stratified", "novelty"):
            sel[s] = select(pool, a.budget, s, seed=a.seed)
        seed_idx = select(pool, 40, "uniform", seed=a.seed + 100)
        m0, _ = train_eval([pool[i] for i in seed_idx], {}, 150, a.seed, "seed-model")
        unc = select([pool[i] for i in range(len(pool))], a.budget, "uncertainty", seed=a.seed, model=m0)
        sel["uncertainty"] = sorted(set(unc) | set(seed_idx))[: a.budget + 40]
        sel["all"] = list(range(len(pool)))
        res["arms"]["marginal_km"] = {"eval": ev(marginal_predict_fn(pool, EVENTS, ["responses", "views"]), sets)}
        for name, idx in sel.items():
            data = [pool[i] for i in idx]
            _, r = train_eval(data, sets, a.steps, a.seed, name)
            r["n_groups"] = len({e.group for e in data})
            res["arms"][name] = r
            print(name, len(data), r["n_groups"], {k: round(v["mean_brier"], 4) for k, v in r["eval"].items()}, flush=True)
            save_json(out, res)
    res["wall_s"] = T.s()
    save_json(out, res)


if __name__ == "__main__":
    main()
