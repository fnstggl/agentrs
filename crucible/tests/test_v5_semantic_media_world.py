"""The `media` kernel module (creator tools) and `embed.compare` in a world with the substrate artifact store.

Needs `crucible.substrate.artifacts` (workstream A, INTERFACE_CONTRACTS 1.1). Without it these tests fail (they do not
skip): the media tools store their outputs as artifacts.
"""
from __future__ import annotations

import copy

import pytest

from crucible import load_all_modules
from crucible.core.kernel import World, WorldSpec

load_all_modules()

MODS = ("identity", "bank", "admin", "artifacts", "embed", "media")


def _world(seed=3):
    w = World.create(WorldSpec("d1-media", seed, [{"name": n} for n in MODS], {}))
    alice = w.system_call("admin.join", {"handle": "alice", "endowment": 0})["principal"]
    bob = w.system_call("admin.join", {"handle": "bob", "endowment": 0})["principal"]
    return w, alice, bob


def _ok(w, actor, tool, args):
    r = w.execute(actor, {"tool": tool, "args": args})
    assert r.ok, (tool, r.error)
    return r.result


def test_creator_tools_make_real_artifacts_and_the_world_replays_and_regenerates():
    from crucible.substrate import artifacts as art
    from crucible.semantic import media as M
    w, alice, bob = _world()
    t1 = _ok(w, alice, "media.render_text", {"text": "Spring sale: everything half price", "width": 480,
                                             "height": 360, "font_size": 40})
    t2 = _ok(w, alice, "media.draw", {"shapes": [{"kind": "circle", "xy": [0.3, 0.3, 0.7, 0.7], "fill": "#cc0000"}],
                                      "width": 480, "height": 360})
    t3 = _ok(w, alice, "media.render_html", {"html": "<h1>Opening hours</h1><p>Monday to Friday, 9 to 5</p>",
                                             "width": 480})
    sp = _ok(w, alice, "media.speak", {"text": "Everything is half price this weekend.", "voice": "slt"})
    mel = _ok(w, alice, "media.synth", {"kind": "melody", "notes": ["C4", "E4", "G4", "C4+E4+G4"], "bpm": 120})
    assert t1["mime"] == "image/png" and (t1["width"], t1["height"]) == (480, 360)
    assert sp["mime"] == "audio/wav" and sp["duration_s"] > 1.0 and mel["duration_s"] == pytest.approx(2.0, abs=0.05)
    vid = _ok(w, alice, "media.make_video", {"images": [t1["ref"], t2["ref"]], "audio": sp["ref"],
                                             "seconds_per_image": 1.5, "width": 480, "height": 360})
    assert vid["mime"] == "video/mp4" and vid["has_audio"] and abs(vid["duration_s"] - 3.0) < 0.2
    info = art.info(w, vid["ref"])
    assert info["owner"] == alice and set(info["parents"]) == {t1["ref"], t2["ref"], sp["ref"]}
    assert M.probe(art.get(w, vid["ref"]))["has_video"]
    small = _ok(w, alice, "media.transcode", {"ref": vid["ref"], "format": "webm", "width": 240, "crf": 40})
    assert small["mime"] == "video/webm" and small["width"] == 240
    pr = _ok(w, alice, "media.probe", {"ref": small["ref"]})
    assert pr["has_audio"] and pr["width"] == 240
    edited = _ok(w, alice, "media.edit_image", {"ref": t3["ref"], "ops": [{"op": "scale", "f": 0.5},
                                                                           {"op": "encode", "fmt": "JPEG",
                                                                            "codec_quality": 70}]})
    assert edited["mime"] == "image/jpeg" and edited["width"] == 240
    # another actor cannot use what alice has not published; bad inputs are refused, not crashes
    r = w.execute(bob, {"tool": "media.edit_image", "args": {"ref": t1["ref"], "ops": [{"op": "flip"}]}})
    assert not r.ok and r.error["code"] == "not_found"
    r = w.execute(alice, {"tool": "media.edit_image", "args": {"ref": t1["ref"], "ops": [{"op": "resize"}]}})
    assert not r.ok and r.error["code"] == "invalid_action"
    # replay (appliers) and regeneration (handlers with recorded outcomes) reproduce the world exactly
    evs = copy.deepcopy([e.to_dict() for e in w.events_range(0)])
    assert w.check_invariants() == []
    rp = World.replay(w.spec, evs)
    assert (rp.digest(), rp.last_hash) == (w.digest(), w.last_hash)
    g = World.regenerate(w.spec, evs)
    assert (g.last_hash, g.digest()) == (w.last_hash, w.digest())


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_embed_compare_reads_published_content_through_the_encoders():
    w, alice, bob = _world(5)
    a = _ok(w, alice, "media.render_text", {"text": "Flood warning for the coast this weekend", "width": 448,
                                            "height": 448, "font_size": 36})
    b = _ok(w, alice, "media.edit_image", {"ref": a["ref"], "ops": [{"op": "encode", "fmt": "JPEG",
                                                                     "codec_quality": 60}]})
    c = _ok(w, alice, "media.draw", {"shapes": [{"kind": "rect", "xy": [0, 0.6, 1, 1], "fill": "#228822"},
                                                {"kind": "circle", "xy": [0.6, 0.1, 0.8, 0.3], "fill": "#ffdd00"}],
                                     "background": "#88ccff", "width": 448, "height": 448})
    r = w.execute(bob, {"tool": "embed.compare", "args": {"a": a["ref"], "b": b["ref"], "modality": "image"}})
    assert not r.ok and r.error["code"] == "not_found"
    for x in (a, b, c):
        _ok(w, alice, "artifact.publish", {"ref": x["ref"]})
    same = _ok(w, bob, "embed.compare", {"a": a["ref"], "b": b["ref"], "modality": "image"})["cosine"]
    diff = _ok(w, bob, "embed.compare", {"a": a["ref"], "b": c["ref"], "modality": "image"})["cosine"]
    assert same > 0.95 and diff < same - 0.1
    assert w.state.count("embeddings") == 3 and w.check_invariants() == []
    evs = copy.deepcopy([e.to_dict() for e in w.events_range(0)])
    g = World.regenerate(w.spec, evs)
    assert g.last_hash == w.last_hash


def test_media_module_passes_admission():
    from crucible.worldgen.admission import admit
    rep = admit(["artifacts", "media"], steps=150)
    assert rep.passed, (rep.checks, rep.errors)
