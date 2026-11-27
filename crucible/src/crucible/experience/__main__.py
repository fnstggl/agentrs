"""Command line of the learner interface.

    PYTHONPATH=src python -m crucible.experience run --tier 1 --name mech:scm_additive --seed 0 --steps 300 \\
        --reward-mode observation_only --store /tmp/exp --checkpoint /tmp/ref.pt --out /tmp/run.json
    PYTHONPATH=src python -m crucible.experience tiers

`run` drives the `ReferenceLearner` (or, with ``--learner tool-value``, the statistical `ToolValueLearner`) through
`run_learner` on one environment, stores every interaction in an
`ExperienceStore`, updates the learner, saves its checkpoint and writes a summary whose digests (experience store,
learner parameters, final world state) identify the run: the same command gives the same digests. `run` is a
training/development path: the learner shares the world's process.

`evaluate` is the evaluation path (W.2, OO.4): the world, the learner and the evaluator in three processes
(`crucible.experience.remote.evaluate`); the learner holds only the participant pipe and the world gets a sealed seed
unless ``--seed`` names one:

    PYTHONPATH=src python -m crucible.experience evaluate --tier 1 --name pe_rule_induction \
        --learner crucible.experience.remote:reference_learner --learner-kw '{"steps": 200}' --out /tmp/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _run(a) -> dict:
    from .loop import run_learner
    from .reference import ReferenceLearner
    from .registry import make
    from .store import ExperienceStore
    kw = {}
    if a.tier == 1 and not a.name.startswith("rapid:"):
        kw["validate"] = not a.unvalidated
    if a.tier in (0, 2, 3) and a.agents:
        kw["agents"] = a.agents.split(",")
    env = make(a.tier, a.name, a.seed, a.reward_mode, **kw)
    if a.learner == "tool-value":
        from .statistical import ToolValueLearner
        learner = ToolValueLearner(seed=a.learner_seed, reward_mode=env.reward_mode)
    else:
        learner = ReferenceLearner(seed=a.learner_seed, reward_mode=env.reward_mode)
    if a.resume and a.checkpoint and os.path.exists(a.checkpoint):
        learner.load(a.checkpoint)
    store = ExperienceStore(a.store) if a.store else None
    before = learner.params_hash()
    s = run_learner(env, learner, a.steps, store=store, update_every=a.update_every,
                    think_s=a.think_s or None)
    if store is not None:
        store.close()
    if a.checkpoint:
        learner.save(a.checkpoint)
    out = {"command": " ".join(sys.argv), "env": env.describe(), "steps": s["steps"], "actions": s["actions"],
           "world_changes": s["world_changes"], "updates": s["updates"], "reward_sum": s["reward_sum"],
           "nonzero_rewards": s["nonzero_rewards"], "schema_failures": s["schema_failures"], "kinds": s["kinds"],
           "params_hash_before": before, "params_hash_after": learner.params_hash(), "params": learner.params(),
           "env_digest": env.digest(), "store_digest": store.digest() if store is not None else None,
           "store_records": len(store) if store is not None else 0}
    env.close()
    return out


def main(argv=None) -> int:
    from ..hosting import engage
    engage("experience runner")                # Part 16: no real-world bridge in a process that hosts worlds
    p = argparse.ArgumentParser(prog="python -m crucible.experience")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the reference learner on one environment")
    r.add_argument("--tier", type=int, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--reward-mode", default="observation_only")
    r.add_argument("--steps", type=int, default=200)
    r.add_argument("--learner-seed", type=int, default=0)
    r.add_argument("--learner", choices=("reference", "tool-value"), default="reference",
                   help="the neural ReferenceLearner or the statistical ToolValueLearner")
    r.add_argument("--update-every", type=int, default=32)
    r.add_argument("--think-s", type=int, default=0)
    r.add_argument("--agents", default="")
    r.add_argument("--unvalidated", action="store_true", help="Tier 1: an unvalidated training task")
    r.add_argument("--store")
    r.add_argument("--checkpoint")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--out")
    sub.add_parser("tiers", help="which tiers are available in this checkout")
    e = sub.add_parser("evaluate", help="evaluate a learner out of process (world, learner, evaluator apart)")
    e.add_argument("--tier", type=int, required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--seed", type=int, default=None, help="default: a sealed seed only the evaluator knows")
    e.add_argument("--reward-mode", default="observation_only")
    e.add_argument("--agents", default="")
    e.add_argument("--learner", default="crucible.experience.remote:reference_learner",
                   help="module:function or file.py:function, called as fn(participant, **learner_kw)")
    e.add_argument("--learner-kw", default="{}", help="JSON object")
    e.add_argument("--timeout", type=float, default=3600.0)
    e.add_argument("--out")
    a = p.parse_args(argv)
    if a.cmd == "evaluate":
        from .remote import evaluate
        ref = {"tier": a.tier, "name": a.name, "reward_mode": a.reward_mode}
        if a.seed is not None:
            ref["seed"] = a.seed
        if a.agents:
            ref["kw"] = {"agents": a.agents.split(",")}
        out = evaluate(ref, a.learner, json.loads(a.learner_kw), timeout=a.timeout)
        text = json.dumps(out, indent=1, sort_keys=True, default=str)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(text)
        print(text)
        return 0 if out["learner_status"] == "ok" else 1
    if a.cmd == "tiers":
        from .registry import tier2_domains, tiers_available
        print(json.dumps({"tiers": tiers_available(), "domains": sorted(tier2_domains())}, indent=1))
        return 0
    out = _run(a)
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
