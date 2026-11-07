"""Choosing which observed actions to train on when there are more than the learner can use.

Strategies (all return indices into the candidate list, within a budget):

* ``uniform``     — random sample (volume baseline: mirrors whatever the sources produce most);
* ``stratified``  — equal share per stratum (environment / app / action type);
* ``novelty``     — greedy farthest-point selection on hashed n-gram vectors of context + action
                    (diverse content, not near-duplicates);
* ``uncertainty`` — highest ensemble disagreement (epistemic) under the current model, mixed with
                    stratification so one noisy stratum cannot take the whole budget;
* ``learnable_error`` — needs matured labels: prediction error minus the model's own aleatoric
                    estimate (Brier − p(1−p)), so inherently random outcomes are not chased.

Raw surprise alone is never used: an unpredictable coin flip has high error forever.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from agentrs.data.temporal_transition import TemporalExample

DIM = 512


def hashed_vector(text: str, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    toks = text.lower().split()
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
        v[h % dim] += 1.0 if (h >> 9) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def coverage(examples: Sequence[TemporalExample]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for e in examples:
        c[e.group or "?"] += 1
    return dict(c)


def _stratified(groups: Sequence[str], budget: int, rng: random.Random, order: list[int] | None = None) -> list[int]:
    by: dict[str, list[int]] = defaultdict(list)
    for i in (order if order is not None else range(len(groups))):
        by[groups[i]].append(i)
    if order is None:
        for v in by.values():
            rng.shuffle(v)
    keys = sorted(by)
    out, k = [], 0
    while len(out) < budget and any(by.values()):
        g = keys[k % len(keys)]
        if by[g]:
            out.append(by[g].pop(0))
        k += 1
    return out


def select(examples: Sequence[TemporalExample], budget: int, strategy: str, seed: int = 0, model=None,
           predictions: dict | None = None) -> list[int]:
    n = len(examples)
    if budget >= n or strategy == "all":
        return list(range(n))
    rng = random.Random(seed)
    groups = [e.group or "?" for e in examples]
    if strategy == "uniform":
        return sorted(rng.sample(range(n), budget))
    if strategy == "stratified":
        return sorted(_stratified(groups, budget, rng))
    if strategy == "novelty":
        X = np.stack([hashed_vector(e.initial_state + " " + e.complete_action) for e in examples])
        first = rng.randrange(n)
        chosen = [first]
        d = 1.0 - X @ X[first]
        while len(chosen) < budget:
            j = int(np.argmax(d))
            chosen.append(j)
            d = np.minimum(d, 1.0 - X @ X[j])
            d[chosen] = -1.0
        return sorted(chosen)
    if strategy in ("uncertainty", "learnable_error"):
        score = np.asarray(_scores(examples, strategy, model, predictions))
        order = list(np.argsort(-score))
        # half by score, half stratified over the score-ranked remainder (keeps every stratum)
        top = order[: (budget + 1) // 2]
        rest = [i for i in order if i not in set(top)]
        more = _stratified(groups, budget - len(top), rng, order=rest)
        return sorted(set(top) | set(more))
    raise ValueError(f"unknown strategy {strategy}")


def _scores(examples, strategy, model, predictions) -> list[float]:
    from agentrs.temporal.temporal_evaluation import event_truth
    from agentrs.temporal.temporal_world_model import TemporalQuery

    if predictions is None:
        if model is None:
            raise ValueError(f"{strategy} sampling needs a model")
        hs = [3600.0, 86400.0]
        preds = model.predict_batch([TemporalQuery.from_example(e) for e in examples], hs)
    else:
        preds = predictions["preds"]
        hs = predictions["horizons"]
    out = []
    for e, p in zip(examples, preds):
        if strategy == "uncertainty":
            out.append(float(np.mean([hp.epistemic for hp in p.horizons])))
            continue
        errs = []
        for j, h in enumerate(hs):
            for name, pr in p.horizons[j].event_prob.items():
                y = event_truth(e, name, h)
                if y is None:
                    continue
                errs.append((pr - y) ** 2 - pr * (1 - pr))
        out.append(float(np.mean(errs)) if errs else 0.0)
    return out
