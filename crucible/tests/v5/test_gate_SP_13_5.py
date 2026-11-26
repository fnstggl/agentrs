"""SP-13.5: each artifact is encoded once, not per viewer, and representations are reused across participants and
worlds (MASTER_SPEC 13.5), measured in an integrated social world: a civilisation (`crucible.civ`, development
population) whose individuals publish and consume content through the social internet's feeds.

With a fresh machine-wide cache directory, world 1 runs one simulated day: every representation its participants
and platforms use comes from the shared space (`crucible.semantic.space`). We count the space's model runs per
distinct artifact (`encoder_runs_per_artifact` = computed representations / distinct artifacts recorded in the
world) next to the number of consumption decisions (views) and impressions that used those representations. Then
world 2 - the same civilisation generated again in the same process, as a counterfactual experiment would - runs
the same day: every representation it records that world 1 computed must come from the cache, not from a model
(`cross_world_representation_reuse` = such representations; `world2_model_runs` must be 0 for them).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 3000 --ram-mb 12000 --timeout 3600 \
        --label "D: SP-13.5" -- python -m pytest -q tests/v5/test_gate_SP_13_5.py
"""
import os
import tempfile

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _keys(world):
    return {k for k, _ in world.state.scan("space_items")} if "space_items" in world.state.tables else set()


def _consumption(world) -> dict:
    from crucible.substrate import columns
    return {q: int(columns.col(world, "aud_items", col).sum()) for q, col in (("impressions", "imp"),
                                                                              ("views", "views"))}


def test_SP_13_5_encode_once_and_reuse_across_worlds():
    import crucible
    from crucible import civ
    from crucible.semantic import space
    from crucible.semantic.space import store
    crucible.load_all_modules()
    saved = os.environ.get("CRUCIBLE_CACHE")
    tmp = tempfile.mkdtemp(prefix="sp135-cache-")
    os.environ["CRUCIBLE_CACHE"] = tmp
    try:
        def computed():
            return sum(space.stats()["computed"].values())

        c0 = computed()
        w1 = civ.generate("balanced", 4, population="development")
        w1.advance_time(w1.now + 86400)
        c1 = computed()
        k1 = _keys(w1)
        cons1 = _consumption(w1)
        cache = store.default_cache()
        h0 = dict(cache.stats)
        w2 = civ.generate("balanced", 4, population="development")
        w2.advance_time(w2.now + 86400)
        c2 = computed()
        k2 = _keys(w2)
        hits = (cache.stats["memory_hits"] - h0["memory_hits"]) + (cache.stats["disk_hits"] - h0["disk_hits"])
        same = k1 & k2
        runs1 = c1 - c0
        m = {"encoder_runs_per_artifact": round(runs1 / max(1, len(k1)), 4),
             "artifacts_represented_world1": len(k1), "model_runs_world1": runs1,
             "impressions_world1": cons1["impressions"], "views_world1": cons1["views"],
             "cross_world_representation_reuse": int(min(hits, len(same))) if c2 - c1 == 0 else
             int(max(0, len(same) - (c2 - c1))),
             "shared_artifacts": len(same), "model_runs_world2": c2 - c1, "cache_hits_world2": hits,
             "digests_equal": w1.digest() == w2.digest()}
        w1.close()
        w2.close()
    finally:
        if saved is None:
            os.environ.pop("CRUCIBLE_CACHE", None)
        else:
            os.environ["CRUCIBLE_CACHE"] = saved
    record("SP-13.5", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_13_5.py",
           notes="Fresh cache directory; world 2 is the same civilisation generated again in the same process.")
    assert m["artifacts_represented_world1"] > 0 and m["views_world1"] > 0
    assert m["encoder_runs_per_artifact"] <= 1.0
    assert m["cross_world_representation_reuse"] >= 1 and m["model_runs_world2"] == 0
