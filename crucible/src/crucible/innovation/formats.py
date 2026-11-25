"""Problem formats beyond the 4.0 need registry, with the same interface as `crucible.economy.needs.Need`
(name, title, doc, keywords, generate(rng, size) -> (input, hidden), score(input, hidden, output) -> [0, 1]).

They live here, not in the process-global `NEEDS`, so that worlds and tests that enumerate `NEEDS` are unchanged.

* `route.batch`: many shortest-path queries in one request (a composite technology: a service for it is naturally
  built on top of an existing `route.shortest` service and multiplies the load on it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Format:
    name: str
    title: str
    doc: str
    keywords: tuple[str, ...]
    generate: Callable[[np.random.Generator, float], tuple[dict, Any]]
    score: Callable[[dict, Any, Any], float]
    inputs: dict
    outputs: dict


FORMATS: dict[str, Format] = {}


def _gen_batch(rng, size):
    from ..economy.needs import NEEDS
    sp = NEEDS["route.shortest"]
    k = 3 + int(round(3 * size))
    qs, hs = [], []
    for _ in range(k):
        i, h = sp.generate(rng, size)
        qs.append(i)
        hs.append(h)
    return {"queries": qs}, hs


def _score_batch(inp, hidden, out):
    from ..economy.needs import NEEDS
    sp = NEEDS["route.shortest"]
    try:
        paths = out["paths"]
    except (TypeError, KeyError):
        return 0.0
    if not isinstance(paths, list) or len(paths) != len(inp["queries"]):
        return 0.0
    return float(np.mean([sp.score(q, h, {"path": p}) for q, h, p in zip(inp["queries"], hidden, paths)]))


FORMATS["route.batch"] = Format(
    "route.batch", "Batch shortest-path planning",
    "Input: {\"queries\": [route.shortest inputs]}.\nOutput: {\"paths\": [one path per query]}.\n"
    "Score: mean route.shortest score over the queries.",
    ("batch", "routes", "fleet", "planning", "dispatch", "logistics"), _gen_batch, _score_batch,
    {"type": "object", "properties": {"queries": {"type": "array"}}, "required": ["queries"],
     "additionalProperties": True},
    {"type": "object", "properties": {"paths": {"type": "array"}}, "required": ["paths"], "additionalProperties": True})


def lookup(name: str):
    """A format by name: the 4.0 need registry first, then the formats defined here."""
    from ..economy.needs import NEEDS
    if name in NEEDS:
        return NEEDS[name]
    if name in FORMATS:
        return FORMATS[name]
    raise KeyError(name)
