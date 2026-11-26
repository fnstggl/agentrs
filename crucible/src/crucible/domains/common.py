"""Helpers shared by the workstream I4 domain modules (education, information, software ecosystems).

* `make_seat`: the participant seat a learner or the scripted reference occupies (a person with a checking account
  and an endowment), created at genesis so that every configuration has the same seat.
* `independent`: independence of two principals (`goals.metrics.is_related`, the rule every v5 evaluator uses).
* `SlateClicks`: open weights for the items of a slate (search results) a participant looks through: a hand-set
  logit of absolute features computed from recorded world state only (the bag-of-words cosine between what the
  participant is looking for and the result, the participant's own history, the item's recorded impressions and
  opens, its age). Fidelity record `learn.slate_open`: plausible, unvalidated. It does NOT use a fitted model:
  `news_click@1` (MIND news impressions) is outside its card's scope for lesson search results, and the
  within-slate rank mapping of its text feature erased absolute relevance (audit E S6; workstream E, wave 3).
* `bow_cos`: that cosine.

Nothing here reads evaluator data; all functions are deterministic given their inputs and the rng passed in.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..services.search import tokenize

HOUR, DAY = 3600, 86400


def make_seat(ctx, handle: str, endowment: int, name: str | None = None, kind: str = "agent") -> str:
    """Create the seat principal (idempotent): identity, checking account, endowment from the genesis account."""
    pid = f"p:{handle}"
    if ctx.state.has("principals", pid):
        return pid
    idm = ctx.world.modules["identity"]
    bank = ctx.world.modules["bank"]
    idm.create_person(ctx, handle, name or handle, kind, {"role": "seat"})
    acct = bank.open_account(ctx, pid, "checking")
    if endowment:
        bank.move(ctx, "acct_genesis", acct, int(endowment), "endowment", f"endowment for {pid}")
    return pid


def make_person(ctx, handle: str, name: str, endowment: int = 0, kind: str = "npc", role: str = "participant") -> str:
    idm = ctx.world.modules["identity"]
    bank = ctx.world.modules["bank"]
    pid = idm.create_person(ctx, handle, name, kind, {"role": role})
    acct = bank.open_account(ctx, pid, "checking")
    if endowment:
        bank.move(ctx, "acct_genesis", acct, int(endowment), "endowment", f"savings of {pid}")
    return pid


def make_org(ctx, handle: str, name: str, kind: str = "company") -> tuple[str, str]:
    """An organisation with a founding person (who acts for it); returns (org, founder)."""
    idm = ctx.world.modules["identity"]
    bank = ctx.world.modules["bank"]
    founder = idm.create_person(ctx, f"{handle}-lead"[:31], f"{name} lead", "npc", {"role": "publisher"})
    bank.open_account(ctx, founder, "checking")
    oid = idm.create_org(ctx, handle, name, founder, kind, {})
    bank.open_account(ctx, oid, "operating")
    return oid, founder


def independent(world, a: str, b: str) -> bool:
    from ..goals.metrics import is_related
    return not is_related(world, a, b)


def bow_cos(a: str, b: str) -> float:
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    return dot / math.sqrt(sum(v * v for v in ca.values()) * sum(v * v for v in cb.values()))


class SlateClicks:
    """Open weights for a slate of results: a plausible, unvalidated logit of absolute features (fidelity record
    `learn.slate_open`; no fitted model, see the module docstring). Row-wise: an item's weight depends only on its
    own features, never on the other items' ranks."""

    MODEL = "learn.slate_open@1"
    PRIOR = 0.04                    # smoothing prior of the per-item open rate (ctr_features)
    FEATURES = ("text_cos", "cat_share", "subcat_share", "log_hist_len", "pop_ctr_24h", "pop_log_clicks_per_1k",
                "age_log_hours")
    COEF = (3.0, 1.0, 0.5, 0.0, 4.0, 0.3, -0.15)
    BIAS = -3.2

    @classmethod
    def features(cls, cos: list[float], cat_share: list[float], subcat_share: list[float], hist_len: int,
                 ctr_24h: list[float], clicks_per_1k: list[float], age_hours: list[float]) -> np.ndarray:
        n = len(cos)
        cols = {"text_cos": np.asarray(cos, float), "cat_share": np.asarray(cat_share, float),
                "subcat_share": np.asarray(subcat_share, float), "log_hist_len": np.full(n, math.log1p(hist_len)),
                "pop_ctr_24h": np.asarray(ctr_24h, float),
                "pop_log_clicks_per_1k": np.log1p(np.asarray(clicks_per_1k, float)),
                "age_log_hours": np.log1p(np.maximum(0.0, np.asarray(age_hours, float)))}
        return np.stack([cols[f] for f in cls.FEATURES], axis=1) if n else np.zeros((0, len(cls.FEATURES)))

    @classmethod
    def prior(cls) -> float:
        return cls.PRIOR

    @classmethod
    def probabilities(cls, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        z = cls.BIAS + np.asarray(X, dtype=float) @ np.asarray(cls.COEF)
        return 1.0 / (1.0 + np.exp(-z))


def ctr_features(stats: dict | None, now: int, active_24h: int, prior: float) -> tuple[float, float]:
    """(pop_ctr_24h, clicks per 1,000 active individuals) from an item's hourly impression/open buckets."""
    imp = clk = 0
    for h, (i, c) in (stats or {}).items():
        if int(h) * HOUR >= now - DAY:
            imp += i
            clk += c
    return (clk + 10 * prior) / (imp + 10), 1000.0 * clk / max(1, active_24h)


def bump_hourly(stats: dict | None, now: int, impressions: int, opens: int) -> dict:
    """Return a new hourly-bucket dict with this hour incremented and buckets older than 24 h dropped."""
    hour = now // HOUR
    out = {k: list(v) for k, v in (stats or {}).items() if int(k) >= hour - 24}
    cur = out.get(str(hour), [0, 0])
    out[str(hour)] = [cur[0] + int(impressions), cur[1] + int(opens)]
    return out
