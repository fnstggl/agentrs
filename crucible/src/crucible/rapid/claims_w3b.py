"""Identification procedures and intervention tests of the families redesigned in wave 3b (evaluator side).

Identification answers a family's experimental question from public actions and observations only (`PublicView`);
the truth comes from the hidden configuration. Nothing in a world imports this module.
"""
from __future__ import annotations

import numpy as np

from .identify import PublicView


def _lever(env, nid: str) -> str:
    cfg = env._config
    return cfg["nodes"][env._engine.idx[nid]]["lever"]["name"]


def identify_gate_threshold(env) -> dict:
    """Question: above which setting of the second lever does the first lever start to move sales? Public: with
    the first lever alternating between its ends every tick, sweep the second lever upward in steps and measure
    the sales difference between the high and low ticks at each step; the threshold is where that difference
    jumps."""
    c = env._config["claim"]
    A, B = _lever(env, c["lever"]), _lever(env, c["gate_lever"])
    view = PublicView(env)
    levels = np.linspace(0.0, 10.0, 21)
    diff = []
    for b in levels:
        hi_, lo_ = [], []
        for k in range(max(1, min(3, (view.horizon - 2) // (2 * len(levels))))):
            if view.done:
                break
            r_hi = view.step({"kind": "levers", "values": {A: 10.0, B: float(b)}})
            r_lo = view.step({"kind": "levers", "values": {A: 0.0, B: float(b)}})
            hi_.append(list(r_hi["observation"]["numbers"].values())[0])
            lo_.append(list(r_lo["observation"]["numbers"].values())[0])
        diff.append(float(np.mean(hi_) - np.mean(lo_)) if hi_ else 0.0)
    d = np.asarray(diff)
    k = int(np.argmax(np.diff(d)))
    est = float(0.5 * (levels[k] + levels[k + 1]))
    truth = float(c["theta"])
    return {"question": "the hidden threshold of the second lever that opens the pathway", "estimate": est,
            "truth": truth, "ok": abs(est - truth) <= 0.6}
