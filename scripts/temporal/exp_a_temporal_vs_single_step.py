"""Experiment A: single-step vs temporal world model (held-out outcome prediction).

All learned models share the Pythia-14M trunk, data and number of gradient steps:
  temporal       continuous-time hazard / count / state heads (ours)
  no_content     same, but the action's generated content is hidden (operation only)
  discrete       one head per (event, trained horizon), no time representation; other horizons
                 interpolated in log time
  single_step    temporal model trained only on what is observable within 5 minutes of each action
                 (next-observation learning); later horizons by persistence
  marginal_km    censoring-aware Kaplan-Meier lookup table (no model)
Evaluation: forecasts at the action time (no later information) for actions in a later time window
(test) and in held-out spaces/app (test_env), at the observed profile horizons and at durations
between them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVENTS, HELD_HORIZONS, HORIZONS, RESULTS, RUNS, TUNED, Timer, make_wm, save_json, strip_content, threads  # noqa: E402

from agentrs.evaluation.temporal_baselines import DiscreteHorizonModel, cap_example, persistence_predict_fn  # noqa: E402
from agentrs.models.backbone import BackboneConfig  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger  # noqa: E402
from agentrs.temporal.temporal_evaluation import (  # noqa: E402
    evaluate_temporal, marginal_predict_fn, model_predict_fn, summarize)
from agentrs.training.common import GradientTrainer, OptimConfig  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)

PROFILE = [60.0, 300.0, 900.0, 3600.0, 21600.0, 86400.0, 259200.0, 604800.0]


def load(corpus: Path):
    L = DelayedOutcomeLedger(corpus / "net" / "ledger.sqlite")
    b = TemporalTrainingDataBuilder(L, lab_temporal_spec())
    train_upd = b.build(splits=["train"], cutoffs="observations", max_cutoffs=2)
    train0 = [e for e in train_upd if e.cutoff_elapsed_s == 0]
    test = b.build(splits=["test"], cutoffs=[0])
    test_env = b.build(splits=["test_env"], cutoffs=[0])
    return train_upd, train0, test, test_env


def evaluate(fn, sets, counts=("responses", "views")):
    out = {}
    for name, exs in sets.items():
        r_obs = evaluate_temporal(fn, exs, HORIZONS, EVENTS, counts)
        r_mid = evaluate_temporal(fn, exs, HELD_HORIZONS, EVENTS, ())
        out[name] = {"observed_horizons": summarize(r_obs, EVENTS), "between_horizons": summarize(r_mid, EVENTS),
                     "detail": r_obs["events"], "detail_between": r_mid["events"], "counts": r_obs.get("counts")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--arms", default="temporal,no_content,discrete,single_step,marginal_km")
    a = ap.parse_args()
    threads()
    train_upd, train0, test, test_env = load(RUNS / f"corpus_s{a.corpus}")
    sets = {"test": test, "test_env": test_env}
    res = {"corpus": a.corpus, "seed": a.seed, "steps": a.steps, "n_train_actions": len(train0),
           "n_train_examples": len(train_upd), "n_test": len(test), "n_test_env": len(test_env), "arms": {}}
    prev = RESULTS / f"expA_c{a.corpus}_s{a.seed}.json"
    if prev.exists():  # keep arms already computed (re-running a subset merges)
        res["arms"] = json.loads(prev.read_text()).get("arms", {})
    cfg = TemporalTrainConfig(steps=a.steps, batch_size=16, seed=a.seed,
                              optim=OptimConfig(lr=a.lr, total_steps=a.steps, schedule="cosine", warmup_steps=30))
    tuned_cfg = TemporalTrainConfig(steps=TUNED["steps"], batch_size=16, seed=a.seed, optim=OptimConfig(
        lr=TUNED["lr"], weight_decay=TUNED["wd"], total_steps=TUNED["steps"], schedule="cosine", warmup_steps=30))
    base_cfg = cfg
    for arm_name in a.arms.split(","):
        T0 = Timer()
        arm = arm_name.removesuffix("_tuned")
        cfg = tuned_cfg if arm_name.endswith("_tuned") else base_cfg
        if arm == "marginal_km":
            fn = marginal_predict_fn(train0, EVENTS, ["responses", "views"])
            info = {}
        elif arm == "discrete":
            m = DiscreteHorizonModel(BackboneConfig(name="EleutherAI/pythia-14m", max_prompt_tokens=320, seed=a.seed),
                                     EVENTS, PROFILE)
            opt = GradientTrainer(m, cfg.optim)
            import random

            rng = random.Random(a.seed)
            for _ in range(a.steps):
                loss, _ = m.loss(rng.sample(train0, 16))
                opt.step(loss)
            m.eval()
            fn = m.predict_fn()
            info = {"last_loss": float(loss.detach())}
        else:
            m = make_wm(a.seed)
            T = TemporalWorldModelTrainer(m, cfg, name=arm)
            if arm == "temporal":
                data = train_upd
            elif arm == "no_content":
                data = [strip_content(e) for e in train_upd]
                sets_nc = {k: [strip_content(e) for e in v] for k, v in sets.items()}
            elif arm == "single_step":
                data = [cap_example(e, 300.0) for e in train0]
            info = T.train(data)
            fn = model_predict_fn(m)
            if arm == "single_step":
                fn = persistence_predict_fn(fn, 300.0)
        ev = evaluate(fn, sets_nc if arm == "no_content" else sets)
        res["arms"][arm_name] = {"train": info, "eval": ev, "wall_s": T0.s(),
                                 "optim": cfg.optim.model_dump(), "steps": cfg.steps}
        print(arm_name, T0.s(), {k: (round(v["observed_horizons"]["mean_brier"], 4), round(v["observed_horizons"]["mean_auroc"] or 0, 3),
                                round(v["between_horizons"]["mean_brier"], 4)) for k, v in ev.items()}, flush=True)
        save_json(f"expA_c{a.corpus}_s{a.seed}.json", res)


if __name__ == "__main__":
    main()
