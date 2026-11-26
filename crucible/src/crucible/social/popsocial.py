"""The population's own contributions to the social internet (kernel module `pop_social`, host `pop_social`).

Members of the one population (`crucible.population`) create content through the same path as everyone else
(`content.publish`: catalogue, search index, distribution and exact audience accounting), as an activity that
competes for their attention and time with reading, browsing, searching and the communities:

* `social.create` - write a piece about what one cares about (the nearest corpus topic to one's interests, with
  personal variation) in one's own words (`population.writing`, from real news text; skill shapes length and
  coherence), store it as one's artifact and publish it on the text platform that suits its length (`posts` for
  short pieces, `blog` for articles). Its utility is what the creator observes: the platform's published,
  delayed and noisy analytics of its earlier pieces (views gained since it last looked) against the time the
  writing took - so the creators whose pieces find an audience keep creating and the others stop (habit learning
  in the population's tick). No piece carries a quality, relevance or virality figure.

Decision provenance: the decision to write and the choice of topic and platform are recorded (`pop_social.created`
batch through `population.record_decisions`, model `pop-create@1`).

Fidelity: plausible, unvalidated (FIDELITY_REGISTER: population creator model).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import CrucibleError, ProtocolError
from ..core.kernel import Ctx, Module, register_module
from ..substrate import artifacts, attention, columns, ids
from .. import population
from ..population import semantics, writing
from . import audience as AUD
from . import content as C

CREATE_MODEL = {"model_id": "pop-create", "model_version": "1", "fidelity_id": "population-creator-model"}
DEFAULTS = {"prior": -1.2, "platforms": {"short": "posts", "long": "blog"}, "minutes": [10.0, 45.0],
            "topic_noise": 0.08}


@register_module
class PopSocial(Module):
    name = "pop_social"
    host = "pop_social"
    version = "1"
    depends = ("pop", "space", "content", "aud")

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"pop_social: unknown config key {k!r}")
            self.cfg[k] = v

    def install(self, world):
        n = columns.rows(world, "pop")
        columns.declare(world, "pop", n, {"soc_items": ("int32", (), 0), "soc_seen_views": ("int64", (), 0),
                                          "soc_last_item": ("int32", (), -1)})
        population.register_activity(world, "social.create", self.act_create, owner="pop_social",
                                     prior=float(self.cfg["prior"]), goals=(3, 5))

    # ------------------------------------------------------------------ the activity
    def _views_seen(self, world, i: int) -> int:
        """Views of the individual's items as the platforms publish them to their creator (delayed, noisy)."""
        p = ids.pid(i)
        tot = 0
        for r in world.state.find_ids("aud_item_info", "creator", p):
            tot += int(AUD.audience(world, int(r), principal=p)["views"])
        return tot

    def act_create(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        w = ctx.world
        idx = np.asarray(idx, np.int64)
        n = idx.size
        rng = ctx.rng("pop_social.create")
        coords = population.interest_coords(w, idx)
        T = len(semantics.topic_names())
        tc = semantics.topic_coords(coords.shape[1])
        noise = float(self.cfg["topic_noise"])
        topic = np.argmax(coords @ tc.T + rng.gumbel(size=(n, T)) * noise, axis=1)
        skill = columns.col(w, "pop", "skills")[idx, 1].astype(np.float64)
        left = columns.col(w, "pop", "attention_left_min")[idx].astype(np.float64)
        lo, hi = (float(x) for x in self.cfg["minutes"])
        minutes = np.minimum(left, lo + (hi - lo) * rng.random(n) * (0.5 + skill))
        long_piece = rng.random(n) < 0.15 + 0.6 * skill
        n_sent = np.where(long_piece, 4 + rng.integers(0, 8, n) * (0.5 + skill), 1 + rng.integers(0, 3, n))
        made = np.zeros(n, bool)
        u = np.zeros(n)
        seen_new = columns.col(w, "pop", "soc_seen_views")[idx].astype(np.int64).copy()
        last_item = columns.col(w, "pop", "soc_last_item")[idx].astype(np.int64).copy()
        nitems = columns.col(w, "pop", "soc_items")[idx].astype(np.int64).copy()
        for j in range(n):
            i = int(idx[j])
            if minutes[j] < lo * 0.5:
                u[j] = -0.05
                continue
            # what the creator learns from the platforms about its earlier pieces
            views = self._views_seen(w, i) if nitems[j] else 0
            gained = max(0, views - int(seen_new[j]))
            seen_new[j] = views
            title, body = writing.compose(rng, int(topic[j]), int(n_sent[j]), float(skill[j]))
            plat = self.cfg["platforms"]["long" if long_piece[j] else "short"]
            kind = "article" if plat == "blog" else "post"
            p = ids.ensure_principal(ctx, i)
            data = f"{title}\n\n{body}\n".encode("utf-8")
            try:
                ref = artifacts.put(ctx, data, "text/plain", owner=p, meta={"title": title[:80]}, kind="text")
                cid = C.module(w).publish(ctx, p, ref, kind, plat, title)
            except CrucibleError:
                u[j] = -0.1
                continue
            rec = ctx.state.tables["content"][cid]
            if rec.get("aud_item") is not None:
                last_item[j] = int(rec["aud_item"])
            made[j] = True
            nitems[j] += 1
            # the creator's experienced value: the audience its earlier pieces found, against the effort
            u[j] = 0.5 * math.log1p(gained) / math.log(30.0) - 0.004 * minutes[j]
        spend = np.flatnonzero(made & (minutes > 0))
        if spend.size:
            attention.spend(ctx, idx[spend], minutes[spend], activity="social.create")
        population.record_decisions(ctx, "pop_social.created", idx, {
            "soc_items": nitems.astype(np.int32), "soc_seen_views": seen_new, "soc_last_item": last_item.astype(np.int32)},
            **CREATE_MODEL, extra={"t": int(now), "created": int(made.sum()),
                                   "topics": [int(t) for t in topic[made]]})
        return u
