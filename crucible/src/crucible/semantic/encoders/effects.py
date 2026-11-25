"""Gate E.3: do controlled artifact changes change participant behaviour through learned representations?

The measurement is a controlled experiment. Two worlds are built from the same spec and seed; they differ in exactly
one thing - the bytes of one content artifact (same title, tags, creator, duration and publication time). The
population runs, and each individual's behaviour towards that item (watch seconds, views, shares...) is recorded in
both worlds. Because everything else is identical, a difference can only come from what the participants' decision
models read from the artifact, i.e. its encoded representation. The test is a paired sign-flip permutation test on
per-individual differences (the same individuals exist in both worlds), two-sided.

The social internet and population (workstreams B and C) provide the worlds. They register a probe:

    from crucible.semantic.encoders import effects
    effects.register_probe("video_feed", make_world, run, modality="video")
        # make_world(artifact_bytes: bytes, mime: str, seed: int) -> world   (the item is published in it)
        # run(world) -> np.ndarray of per-individual outcomes for that item (same order in both worlds)

`measure_probe(name, base, changed)` runs one probe on one artifact pair; `controlled_pairs(modality)` gives artifact
pairs with a genuine content change from the validation suites. Nothing here reads a quality score: the only input
that differs between the arms is the artifact's bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Probe:
    name: str
    make_world: Callable[[bytes, str, int], Any]
    run: Callable[[Any], np.ndarray]
    modality: str
    mime: str
    doc: str = ""


PROBES: dict[str, Probe] = {}
MIMES = {"video": "video/mp4", "audio": "audio/wav", "image": "image/png", "text": "text/plain",
         "code": "application/json"}


def register_probe(name: str, make_world, run, *, modality: str, mime: str | None = None, doc: str = "") -> None:
    PROBES[name] = Probe(name, make_world, run, modality, mime or MIMES[modality], doc)


def paired_permutation_pvalue(diff: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for mean(diff) == 0 (with the +1 correction)."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, n_perm, 2000):
        k = min(2000, n_perm - s)
        signs = rng.choice((-1.0, 1.0), size=(k, len(d)))
        hits += int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))
    return (hits + 1) / (n_perm + 1)


def two_sample_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    pool = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        hits += abs(p[:len(a)].mean() - p[len(a):].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def measure_probe(name: str, base: bytes, changed: bytes, *, seed: int = 0, n_perm: int = 20000) -> dict:
    """Run probe `name` with the two artifacts; returns the effect, its p-value and the representation distance."""
    from . import encode_bytes
    from .validation import cosine_distance
    pr = PROBES[name]
    out_a = np.asarray(pr.run(pr.make_world(base, pr.mime, seed)), dtype=np.float64)
    out_b = np.asarray(pr.run(pr.make_world(changed, pr.mime, seed)), dtype=np.float64)
    if out_a.shape != out_b.shape:
        raise ValueError("the two arms must report outcomes for the same individuals")
    diff = out_b - out_a
    return {"probe": name, "modality": pr.modality, "n_individuals": int(len(diff)),
            "individuals_changed": int(np.sum(diff != 0)), "mean_outcome_base": float(out_a.mean()),
            "mean_outcome_changed": float(out_b.mean()), "effect": float(diff.mean()),
            "artifact_change_behavior_pvalue": paired_permutation_pvalue(diff, n_perm=n_perm, seed=seed),
            "embedding_distance": cosine_distance(encode_bytes(base, pr.modality), encode_bytes(changed, pr.modality))}


def controlled_pairs(modality: str) -> list[tuple[str, bytes, bytes]]:
    """(label, base, changed) artifact pairs with a genuine content change, from the validation suites."""
    from .validation import SUITES
    s = SUITES[modality]()
    return [(f"{a} -> {b} ({g})", s.items[a], s.items[b]) for a, b, g in s.semantic if "!" in b]


def available() -> list[str]:
    """Names of registered probes (after `crucible.load_all_modules()` has imported every package's modules)."""
    from ... import load_all_modules
    load_all_modules()
    return sorted(PROBES)
