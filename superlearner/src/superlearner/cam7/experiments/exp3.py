"""Experiment 3: parametric transfer. Does experience on some mechanism families, kept only in the weights, reduce the
experience needed on unfamiliar ones? Crucible's own `protocols.transfer`, unchanged:

per learner seed: train on the six training families for SOURCE_BUDGET steps and save; then for every target category
(`within`: new seeds of the source families; `compositional`, `structural`, `cross_impl`: the validation families that
declare that relation), (i) zero-shot: the frozen source-trained checkpoint against the untrained one, fresh context, on
the same tasks; (ii) adaptation: two lifetimes on the same target training stream with the same evaluation tasks and the
same ADAPT_BUDGET, one from the source-trained checkpoint, one from scratch (target-only control). Reported: areas
under both adaptation curves and steps to the goal threshold, paired over (seed, task). Nothing else (trajectories,
written notes, retrieval, posterior state) carries over: the learner contract's `reset_context` and `freeze` hold.

Evaluation tasks come from VAL offset 80,000 (untouched; declared as C-4 in the ledger before the run).

    python -m superlearner.cam7.experiments.exp3 run --learner cam7 --seeds 0   (then --seeds 0,1,2 to pool)
"""
from __future__ import annotations

import argparse
import json
import os

from .train import TRAIN, make_learner

RUNS = "/workspace/agentrs/runs/cam7/exp3"
SOURCE_BUDGET = 240_000
ADAPT_BUDGET = 120_000
EVAL_EVERY = 30_000
OFFSET = 80_000


def run(a) -> dict:
    from crucible.research import protocols
    from crucible.research.learner import EnvSpec
    spec = EnvSpec()
    over = json.loads(a.overrides)
    scale = over.pop("scale", "small")
    targets = protocols.transfer_targets(TRAIN, n_eval=a.n_eval, offset=OFFSET, workers=8, skip_failures=True)
    factory = lambda seed: make_learner(a.learner, spec, seed, "cuda", dict(over), scale)
    # one work directory per learner: each seed writes seed_<s>/ inside it, so seeds can run in separate processes
    # and a final call with all seeds pools them (finished runs are recognised and not repeated)
    wd = os.path.join(RUNS, a.name)
    os.makedirs(wd, exist_ok=True)
    seeds = tuple(int(x) for x in a.seeds.split(","))
    res = protocols.transfer(factory, TRAIN, targets, source_budget=a.source_budget, adapt_budget=a.adapt_budget,
                             eval_every=EVAL_EVERY, seeds=seeds, spec=spec, n_envs=32, update_every=8,
                             workdir=wd, label=f"exp3_{a.name}")
    protocols.save_result(res, os.path.join(wd, f"transfer_{'_'.join(map(str, seeds))}.json"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--learner", default="cam7")
    ap.add_argument("--name", default="cam7_small")
    ap.add_argument("--overrides", default='{"scale": "small"}')
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--source-budget", type=int, default=SOURCE_BUDGET)
    ap.add_argument("--adapt-budget", type=int, default=ADAPT_BUDGET)
    run(ap.parse_args())
