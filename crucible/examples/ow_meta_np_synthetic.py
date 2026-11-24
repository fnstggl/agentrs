"""What can the `meta_np` correction learn beyond its regression? Three synthetic families of worlds, full network size.

    python examples/ow_meta_np_synthetic.py        # writes docs/experiments/ow_meta_np_synthetic.json

Each row is the held-out error of the corrected prediction divided by the regression's (below 1 is better), for the last
4 arms of an episode given its first 12. Uses a GPU if one is visible.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402

from crucible.research import learners  # noqa: E402
from crucible.research.learner import EnvSpec  # noqa: E402
from crucible.research.learners.meta_np import to_z  # noqa: E402

CHECKPOINTS = (1000, 2000, 4000, 8000)


def _ep(x, r):
    return {"x": x.astype(np.float32), "x0": np.zeros(x.shape[1], np.float32), "r": r.astype(np.float32),
            "c": np.full(len(x), 0.2, np.float32)}


def shared_curve(g):
    x = g.uniform(-1, 1, (16, int(g.integers(2, 5))))
    return _ep(x, 1.0 + 0.5 * np.cos(4.0 * x).mean(1))


def per_world_saturation(g):
    x = g.uniform(-1, 1, (16, int(g.integers(2, 5))))
    return _ep(x, 1.0 + (g.uniform(-0.2, 0.2, x.shape[1]) * np.tanh(5.0 * x)).sum(1))


def pure_product(g):
    x = g.uniform(-1, 1, (16, int(g.integers(2, 5))))
    a, b = g.choice(x.shape[1], 2, replace=False)
    return _ep(x, 1.0 + g.choice([-1.0, 1.0]) * 0.8 * x[:, a] * x[:, b])


def error(lr, held, a) -> float:
    dev, out = lr.device, []
    lr.net.eval()
    t = lambda v: torch.as_tensor(np.asarray(v, np.float32), device=dev)[None]
    for e in held:
        r, c = to_z(e["r"], 0.01), to_z(e["c"], 0.01)
        before = np.vstack([e["x0"][None], e["x"][:-1]])
        yc = np.stack([r[:12], c[:12], to_z(e["r"][:12] - e["c"][:12], 0.01)], 1)
        with torch.no_grad():
            mu, _ = lr.net(t(e["x"][:12]), t(yc), t(e["x"][12:]), t(e["x0"]),
                           torch.ones(1, e["x"].shape[1], dtype=torch.bool, device=dev),
                           torch.ones(1, 12, dtype=torch.bool, device=dev), t(before[:12]), t(before[12:]), a, 0.0)
        out.append(float(((mu[0, :, 0].cpu().numpy() - r[12:]) ** 2).mean()))
    lr.net.train()
    return float(np.mean(out))


def main() -> None:
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"what": __doc__.strip().splitlines()[0], "episodes": 3000, "held_out_episodes": 40, "device": device, "rows": {}}
    for name, f in (("shared_curve", shared_curve), ("per_world_saturation", per_world_saturation),
                    ("pure_product", pure_product)):
        g = np.random.default_rng(5)
        lr = learners.make("meta_np", EnvSpec(), seed=0, device=device, min_episodes=8, updates_per_episode=0.0)
        lr._buffer = [f(g) for _ in range(3000)]
        held = [f(g) for _ in range(40)]
        base = error(lr, held, 0.0)
        ratio = {}
        for k in range(1, CHECKPOINTS[-1] + 1):
            lr._step()
            if k in CHECKPOINTS:
                ratio[str(k)] = error(lr, held, 1.0) / base
        res["rows"][name] = {"regression_mse": base, "corrected_over_regression": ratio}
        print(name, round(base, 4), {k: round(v, 3) for k, v in ratio.items()}, flush=True)
    with open(os.path.join(HERE, "..", "docs", "experiments", "ow_meta_np_synthetic.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
