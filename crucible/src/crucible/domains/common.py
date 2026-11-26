"""Helpers shared by the workstream I4 domain modules (education, information, software ecosystems).

* `make_seat`: the participant seat a learner or the scripted reference occupies (a person with a checking account
  and an endowment), created at genesis so that every configuration has the same seat.
* `independent`: independence of two principals (`goals.metrics.is_related`, the rule every v5 evaluator uses).
* `SlateClicks`: which items of a slate (search results) a participant opens, from the calibrated `news_click@1`
  model of workstream E (MIND-small news impressions). Features are computed from recorded world state only (the
  participant's own history, the item's recorded impressions and opens, its age); text affinity is the
  bag-of-words cosine between what the participant is looking for and the result, quantile-mapped through the
  model card's `feature_quantiles` by its within-slate rank (the card's rule for a different encoder).
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
    """Open/skip decisions for a slate of results through the calibrated `news_click@1` model (workstream E)."""

    MODEL = "news_click@1"
    _cache: dict = {}

    @classmethod
    def model(cls):
        m = cls._cache.get(cls.MODEL)
        if m is None:
            from ..calibration import load
            m = cls._cache[cls.MODEL] = load(cls.MODEL)
        return m

    @classmethod
    def features(cls, cos: list[float], cat_share: list[float], subcat_share: list[float], hist_len: int,
                 ctr_24h: list[float], clicks_per_1k: list[float], age_hours: list[float]) -> np.ndarray:
        m = cls.model()
        n = len(cos)
        q = m.card["feature_quantiles"]
        levels, tc = np.array(q["levels"]), np.array(q["text_cos"])
        c = np.asarray(cos, dtype=float)
        order = np.argsort(np.argsort(c, kind="stable"), kind="stable")
        text_cos = np.interp((order + 0.5) / max(n, 1), levels, tc)
        sd = float(c.std())
        z = (c - c.mean()) / sd if sd > 1e-9 else np.zeros(n)
        cols = {"text_cos": text_cos, "text_cos_z": z, "cat_share": np.asarray(cat_share, float),
                "subcat_share": np.asarray(subcat_share, float), "log_hist_len": np.full(n, math.log1p(hist_len)),
                "pop_ctr_24h": np.asarray(ctr_24h, float), "pop_log_clicks_per_1k": np.log1p(np.asarray(clicks_per_1k, float)),
                "age_log_hours": np.log1p(np.maximum(0.0, np.asarray(age_hours, float))),
                "slate_size_log": np.full(n, math.log1p(n))}
        return np.stack([cols[f] for f in m.features], axis=1)

    @classmethod
    def prior(cls) -> float:
        return float(cls.model().card.get("prior_rate", 0.04))

    @classmethod
    def probabilities(cls, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return np.asarray(cls.model().predict(X), dtype=float)


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
