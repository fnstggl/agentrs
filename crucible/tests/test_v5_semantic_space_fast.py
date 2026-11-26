"""The cheap LM feature backend of the shared space (`crucible.semantic.space.lm_fast`): same outputs as `lm`, its
valence head distilled from Qwen3-0.6B (`distill.py`), measured against `lm` on held-out sentences (CPU)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from crucible.semantic import finegrained as FG  # noqa: E402
from crucible.semantic.space import lm, lm_fast  # noqa: E402


def test_fast_backend_matches_the_qwen_features_on_held_out_sentences():
    import torch
    torch.set_num_threads(1)
    assert os.path.exists(lm_fast.HEAD), "valence head missing: run python -m crucible.semantic.space.distill"
    pairs = FG.TEST
    texts = [s for p in pairs for s in p[:3]]
    lm.analyse(texts[:1])
    lm_fast.analyse(texts[:1])
    t0 = time.time()
    q = lm.analyse(texts)
    tq = time.time() - t0
    t0 = time.time()
    f = lm_fast.analyse(texts)
    tf = time.time() - t0
    assert set(q[0]) == set(f[0])
    agree = moved = 0
    for j, (_, _, _, kind) in enumerate(pairs):
        if kind in ("negation", "polarity"):
            dq = q[3 * j + 2]["valence"] - q[3 * j]["valence"]
            df = f[3 * j + 2]["valence"] - f[3 * j]["valence"]
            moved += 1
            agree += int(np.sign(dq) == np.sign(df))
    assert agree >= moved - 1, (agree, moved)
    cq = np.array([r["coherence_novel"] for r in q])
    cf = np.array([r["coherence_novel"] for r in f])
    assert np.corrcoef(cq, cf)[0, 1] > 0.6
    assert tf * 3 < tq, (tf, tq)
