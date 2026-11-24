"""Development evaluation: frozen learners (fresh context) on the development sets, per-family scores.

    python -m superlearner.cam7.experiments.deveval out.json name=learner[:json overrides][@checkpoint] ...

Development sets only (the CAM 6.1 decisive evaluation sets, inspected many times).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .train import HELDOUT, TRAIN, make_learner


def main(out: str, items: list[str], n_eval: int = 8, offset: int = 0, sets: str = "familiar,heldout") -> None:
    import torch
    from crucible.research import protocols
    from crucible.research.learner import EnvSpec
    from crucible.research.tasks import eval_tasks
    spec = EnvSpec()
    fams = {"familiar": TRAIN, "heldout": HELDOUT}
    tasks = {k: eval_tasks(fams[k], n_eval, offset=offset, workers=4, skip_failures=True) for k in sets.split(",")}
    res = {}
    for item in items:
        name, rest = item.split("=", 1)
        ckpt = None
        if "@" in rest:
            rest, ckpt = rest.split("@", 1)
        learner, over = (rest.split(":", 1) + ["{}"])[:2]
        over = json.loads(over)
        scale = over.pop("scale", "base")
        torch.manual_seed(0)
        L = make_learner(learner, spec, 0, "cuda", over, scale)
        if ckpt:
            L.load(ckpt)
        L.freeze(True)
        row = {}
        for k, ts in tasks.items():
            ev = protocols.evaluate(L, ts, spec=spec, n_envs=32, explore=False)
            recs = ev["records"]
            by = {}
            for r in recs:
                by.setdefault(r["family"], []).append(r["score"])
            row[k] = {"mean": float(np.mean([r["score"] for r in recs])),
                      "by_family": {f: float(np.mean(v)) for f, v in by.items()},
                      "scores": [r["score"] for r in recs]}
        res[name] = row
        print(name, {k: round(v["mean"], 3) for k, v in row.items()},
              {k: {f: round(x, 2) for f, x in v["by_family"].items()} for k, v in row.items()}, flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
