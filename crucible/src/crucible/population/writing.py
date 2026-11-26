"""How population members write (forum posts, replies, social posts, articles): from a real text corpus.

A writer draws sentences from real news abstracts of the topic it writes about (MIND-small, `space_asset`), the
way a person retells what they read. Skill shapes the text, not any score attached to it: skilled writers keep to
one subject and write longer, coherent pieces from consecutive sentences of a source; less skilled writers drift to
other subjects and write shorter. Whatever a reader later makes of the piece comes from its bytes (the shared
space's representation and `semantics.item_features`). Fidelity: plausible, unvalidated (FIDELITY_REGISTER).
"""
from __future__ import annotations

import functools
import re

import numpy as np

from . import semantics

_SPLIT = re.compile(r"(?<=[.!?])\s+")


@functools.lru_cache(maxsize=1)
def sentences() -> list[list[str]]:
    """Per topic, the corpus abstracts split into sentences (at least 5 words each)."""
    out = []
    for absq in semantics.corpus():
        ss = []
        for a in absq:
            ss += [s.strip() for s in _SPLIT.split(a) if len(s.split()) >= 5]
        out.append(ss)
    return out


def compose(rng: np.random.Generator, topic: int, n_sentences: int, skill: float = 0.5) -> tuple[str, str]:
    """(title, body) of a piece about `topic` (an index of `semantics.topic_names()`)."""
    S_ = sentences()
    T = len(S_)
    topic = int(topic) % T
    own = S_[topic] or S_[0]
    n_sentences = max(1, int(n_sentences))
    start = int(rng.integers(0, len(own)))
    body = []
    for j in range(n_sentences):
        if rng.random() < 0.45 * (1.0 - float(skill)):
            other = S_[int(rng.integers(0, T))] or own
            body.append(other[int(rng.integers(0, len(other)))])
        elif rng.random() < 0.6 + 0.35 * float(skill):
            body.append(own[(start + j) % len(own)])
        else:
            body.append(own[int(rng.integers(0, len(own)))])
    words = re.sub(r"[^\w\s'-]", "", body[0]).split()
    title = " ".join(words[:int(rng.integers(5, 10))]).strip() or semantics.topic_names()[topic]
    return title[:1].upper() + title[1:], " ".join(body)
