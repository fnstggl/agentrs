"""Gate E.3 probe world: individuals of the real population (`crucible.population`, module `pop`) read a small
catalogue of artifacts and choose what to open through the shared space.

Kernel module `space_read` (host `space_read`; no tools): its config holds the catalogue (artifact bytes, base64)
and which entry is the *target* whose bytes the experiment changes. At genesis the artifacts enter the artifact store
and their representations are recorded (`space.item_vectors(ctx, ...)`). It registers the population activity
`space_read`: each due individual sees the whole catalogue and opens at most one item, by a multinomial logit with
an outside option whose item scores are

    s_ij = a + b * cos(interest_i, item_j) + log(comprehension_j)

where (a, b) is the MIND-fitted click calibration of the space (`space.click_calibration()`) and comprehension the
space's intrinsic-feature index (`space.comprehension`). Choice noise is Gumbel from `world.rng_for` keyed by time
and individual, so two worlds that differ only in the target's bytes face identical noise (common random numbers):
any behavioural difference is caused by the representation. Opens are recorded per individual in the column table
`space_read` (`opens`, `target_opens`) by `columns.batch` events.

Interests: `crucible.population.interests(world, idx)` when the population exposes it (ONE_WORLD_CONTRACTS 1);
otherwise the space's interest prior drawn deterministically per world (never stored; a test fixture only).

``encoder = "random"`` is the negative control: item vectors are replaced by unit vectors seeded by the artifact's
hash (no learned structure); everything else is identical.
"""
from __future__ import annotations

import base64
import hashlib
import math

import numpy as np

from ...core.kernel import SYSTEM, Module, applier, register_module
from ...substrate import artifacts, columns

TABLE = "space_read"


def _interests(world, idx: np.ndarray) -> np.ndarray:
    try:
        from ... import population as P
        fn = getattr(P, "interests", None)
    except ImportError:
        fn = None
    if fn is not None:
        return np.asarray(fn(world, idx), dtype=np.float32)
    from . import interest_prior
    cache = world.runtime.get("space_read.interests")
    if cache is None:
        from ...population.core import n_individuals
        cache = world.runtime["space_read.interests"] = interest_prior(
            world, world.rng_for("space_read", "interests"), n_individuals(world))
    return cache[np.asarray(idx, dtype=np.int64)]


@register_module
class SpaceRead(Module):
    name = "space_read"
    host = "space_read"
    version = "1"

    def install(self, world):
        from ...population.core import n_individuals, register_activity
        self.items = [base64.b64decode(x["b64"]) for x in self.config.get("catalogue", [])]
        self.mimes = [x.get("mime", "application/octet-stream") for x in self.config.get("catalogue", [])]
        self.target = int(self.config.get("target", 0))
        self.encoder = self.config.get("encoder", "space")
        world.state.declare_table("space_read_items")
        columns.declare(world, TABLE, n_individuals(world), {"opens": ("i4", (), 0), "target_opens": ("i4", (), 0)})
        register_activity(world, "space_read", self.act, owner=self.name, prior=float(self.config.get("prior", 2.0)))

    def genesis(self, ctx):
        from . import item_features, item_vectors
        refs = [artifacts.put(ctx, d, m, owner=SYSTEM) for d, m in zip(self.items, self.mimes)]
        for i, r in enumerate(refs):
            ctx.emit("space_read.item", {"i": i, "ref": r}, vis="internal")
        item_vectors(ctx, refs)
        for r in refs:
            item_features(ctx, r)

    @applier("space_read.item")
    def ap_item(self, ev):
        self.world.state.put("space_read_items", str(ev.payload["i"]), {"ref": ev.payload["ref"]})

    def refs(self, world) -> list[str]:
        return [world.state.get("space_read_items", str(i))["ref"] for i in range(len(self.items))]

    def vectors(self, ctx, refs: list[str]) -> np.ndarray:
        from . import DIM, item_vectors
        if self.encoder == "random":
            out = np.zeros((len(refs), DIM), dtype=np.float32)
            for i, r in enumerate(refs):
                g = np.random.default_rng(int(hashlib.sha256(r.encode()).hexdigest()[:16], 16))
                v = g.standard_normal(DIM)
                out[i] = v / np.linalg.norm(v)
            return out
        return np.asarray(item_vectors(ctx, refs), dtype=np.float32)

    def act(self, ctx, idx, now):
        from . import click_calibration, comprehension, item_features
        idx = np.asarray(idx, dtype=np.int64)
        refs = self.refs(ctx.world)
        V = self.vectors(ctx, refs)
        comp = np.array([max(1e-6, comprehension(item_features(ctx, r))) for r in refs])
        cal = click_calibration()
        I = _interests(ctx.world, idx)
        s = cal["intercept"] + cal["slope_cos"] * (I @ V.T) + np.log(comp)[None, :]
        noise = np.stack([ctx.world.rng_for("space_read", int(now), int(i)).gumbel(size=len(refs) + 1) for i in idx])
        u = np.concatenate([np.zeros((len(idx), 1)), s], axis=1) + noise
        ch = np.argmax(u, axis=1) - 1                         # -1 = outside option
        opened = ch >= 0
        if opened.any():
            sel = idx[opened]
            columns.batch(ctx, TABLE, "space_read.open", sel,
                          {"opens": np.ones(len(sel), np.int32),
                           "target_opens": (ch[opened] == self.target).astype(np.int32)}, mode="add")
        util = np.where(opened, s[np.arange(len(idx)), np.maximum(ch, 0)] - cal["intercept"], 0.0)
        return util.astype(np.float64)


def world_spec(catalogue: list[tuple[bytes, str]], target: int, seed: int, *, n: int = 1000, encoder: str = "space",
               tick_s: int = 3600):
    """A population world (the `pop` module of `crucible.population`, lightweight class) whose only activity is the
    probe's reading activity (no generated web sites: nothing else competes for attention or adds noise)."""
    from ...core.kernel import WorldSpec
    cat = [{"b64": base64.b64encode(d).decode("ascii"), "mime": m} for d, m in catalogue]
    mods = [{"name": m, "config": {}} for m in ("identity", "bank", "admin", "artifacts", "cols", "space")]
    mods.append({"name": "pop", "config": {"profile": "development", "n": int(n), "tick_s": int(tick_s),
                                           "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}})
    mods.append({"name": "space_read", "config": {"catalogue": cat, "target": int(target), "encoder": encoder}})
    return WorldSpec("space_probe", int(seed), mods, {"cost_model": "deterministic"})


def run(spec, days: float = 2.0):
    """Create the world, run it for `days`, return (world, per-individual target opens, per-individual opens)."""
    from ...core.kernel import World
    w = World.create(spec)
    w.advance_time(w.now + int(days * 86400))
    return w, np.array(columns.col(w, TABLE, "target_opens")), np.array(columns.col(w, TABLE, "opens"))


def representation_response(world_a, world_b, target: int) -> dict:
    """Per-individual change of the target's affinity between the two worlds' recorded representations (what the
    learned representation predicts each individual's response to be)."""
    mod_a, mod_b = world_a.modules["space_read"], world_b.modules["space_read"]
    ra, rb = mod_a.refs(world_a)[target], mod_b.refs(world_b)[target]
    from . import item_vector
    from ...population.core import n_individuals
    idx = np.arange(n_individuals(world_a))
    I = _interests(world_a, idx)
    va, vb = item_vector(world_a, ra), item_vector(world_b, rb)
    return {"delta_cos": I @ vb - I @ va, "cos_items": float(va @ vb) / max(1e-12, math.sqrt(float(va @ va) * float(vb @ vb)))}
