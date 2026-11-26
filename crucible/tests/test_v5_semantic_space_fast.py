"""The cheaper LM feature candidate of the shared space (`crucible.semantic.space.lm_fast`: SmolLM2-135M coherence,
valence head distilled from Qwen3-0.6B by `distill.py`) and its adoption rule.

The space adopts the candidate only with measured equivalence (`lm_fast.ADOPTION` on the committed held-out report
`space/maps/valence_head_report.json`); the report of 2026-11-26 fails it (valence Pearson 0.75), so the space keeps
Qwen3-0.6B. These tests pin that rule and re-measure what the candidate does preserve on the held-out TEST
fine-grained sentences (negation and polarity direction, coherence ranking).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from crucible.semantic import finegrained as FG  # noqa: E402
from crucible.semantic.space import compute, lm, lm_fast  # noqa: E402


def test_the_space_uses_the_candidate_only_when_its_report_shows_equivalence():
    assert os.path.exists(lm_fast.HEAD) and os.path.exists(lm_fast.REPORT)
    with open(lm_fast.REPORT) as fh:
        rep = json.load(fh)
    assert rep["n_heldout_texts"] >= 1000
    assert (compute.LM is lm_fast) == lm_fast.equivalent(rep)
    assert rep["equivalent"] == lm_fast.equivalent(rep)


def test_candidate_keeps_valence_direction_and_coherence_ranking_on_held_out_sentences():
    import torch
    torch.set_num_threads(1)
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
    assert moved >= 8 and agree >= moved - 1, (agree, moved)
    cq = np.array([r["coherence_novel"] for r in q])
    cf = np.array([r["coherence_novel"] for r in f])
    assert np.corrcoef(cq, cf)[0, 1] > 0.6
    if lm.device() == "cpu":            # the cost it would save: one CPU thread
        assert tf * 3 < tq, (tf, tq)
    else:
        assert tf < tq, (tf, tq)
