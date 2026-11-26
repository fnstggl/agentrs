"""Regression tests (workstream C, wave 3b): participant media never reach a native tool or decoder outside the jail
(Gate W.1 / SP-16b), and coordinated fake engagement with a believable footprint is not organic (W.4).

CPU only, no population (the C1 social world of the red-team probes)."""
from __future__ import annotations

import io
import time

import numpy as np
import pytest

from crucible.sandbox import isolation

pytestmark = pytest.mark.skipif(not isolation.available()[0], reason="the media-tool jail cannot be applied here")


def _png(w=40, h=30, color=(200, 30, 30)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def test_social_image_decoding_runs_in_the_jail():
    from crucible.redteam.v5.network import decodes
    from crucible.social import formats, render
    png = _png()
    with decodes() as d:
        facts = formats.image_info(png)
        im = render.open_image(png)
        out = render.process_image(png, [{"op": "grayscale"}])
    assert d.n == 0                                    # no Pillow decode of participant bytes in this process
    assert facts["width"] == 40 and facts["height"] == 30 and facts["mode"] == "RGB"
    assert im.size == (40, 30) and im.getpixel((1, 1)) == (200, 30, 30)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(formats.FormatError):
        formats.image_info(png[:30] + b"\0" * 40)
    with pytest.raises(render.MediaError):
        render.open_image(b"\x89PNG\r\n\x1a\n" + b"junk" * 10)


def test_studio_tools_start_ffmpeg_and_sox_only_through_the_jail():
    from crucible.redteam.v5.network import launches, media_launches
    from crucible.social import formats, render
    render._HASH.clear()
    with launches() as rows:
        wav = render.wav(render.synth("tone", seconds=0.4))
        fx = render.sox_effects(wav, [{"effect": "gain", "db": -3}])
        vid = render.assemble_video([_png(64, 48)], fx, seconds_per_image=0.5, width=64, height=48, fps=5)
        sl = render.slides_video([_png(64, 48), _png(64, 48, (0, 0, 200))], [0.6, 0.4], fx, fps=5)
        render.tool_hash()
        rows = list(rows)
    ls = media_launches(rows)
    assert ls and all(x["jailed"] for x in ls), [x for x in ls if not x["jailed"]]
    assert formats.inspect(vid, "video", "video/mp4")["duration_s"] > 0.3
    assert abs(formats.inspect(sl, "video", "video/mp4")["duration_s"] - 1.0) < 0.3


def test_ogg_audio_is_probed_in_the_jail():
    from crucible.semantic import media as MD
    from crucible.social import formats, render
    ogg = MD.transcode(render.wav(render.synth("tone", seconds=1.0)), "opus", audio_bitrate="24k")
    f = formats.inspect(ogg, "audio", "audio/ogg")
    assert f["audio_format"] == "ogg" and abs(f["duration_s"] - 1.0) < 0.1
    with pytest.raises(formats.FormatError):
        formats.inspect(b"OggS" + b"\0" * 64, "audio", "audio/ogg")


def test_sybil_ring_with_a_fake_footprint_is_not_organic():
    from crucible.redteam import v5
    names = ("v5.social.sybils_with_a_fake_footprint", "v5.social.sybil_ring_engagement",
             "v5.social.engagement_paid_by_the_target", "v5.social.own_organisations_inflate_followers")
    ps = [p for p in v5.probes(only=["social"]) if p.name in names]
    assert len(ps) == len(names)
    t0 = time.time()
    rep = v5.run(ps)
    res = {r["name"]: r for r in rep["probes"]}
    for n in names:
        assert res[n]["result"] == "held", (n, res[n])
    ev = res["v5.social.sybils_with_a_fake_footprint"]["evidence"]["organic_company"]
    assert ev["organic"].get("follow") == 3 and ev["engagers"].get("cluster") == 6
    assert time.time() - t0 < 600


def _c1_world(people):
    import crucible
    from crucible.core.kernel import World, WorldSpec
    from crucible.social.modules import social_internet_modules
    crucible.load_all_modules()
    w = World.create(WorldSpec("soc-sec", 3, social_internet_modules(), {}))
    P = {h: w.system_call("admin.join", {"handle": h, "endowment": 10_000, "password": f"pw-{h}"})["principal"]
         for h in people}
    return w, P


def _act(w, p, tool, **args):
    r = w.execute(p, {"tool": tool, "args": args})
    assert r.ok, (tool, r.error)
    return r.result


def test_a_ring_is_excluded_until_its_work_draws_independent_attention():
    """Four agents who follow each other (a reciprocal ring), post a little and follow a creator are a coordinated
    cluster; once an independent participant follows one of them, the cluster is validated and counts as organic."""
    from crucible.social import integrity as I
    names = ["star", "r0", "r1", "r2", "r3", "indie"]
    w, P = _c1_world(names)
    ring = [P[f"r{k}"] for k in range(4)]
    for k, r in enumerate(ring):
        ref = _act(w, r, "studio.write", format="text", title=f"note {k}", body=f"my notes about birds number {k} " * 4)
        _act(w, r, "posts.publish", ref=ref["ref"], title=f"note {k}")
        _act(w, r, "social.follow", target=P["star"])
        nxt = ring[(k + 1) % 4]
        _act(w, r, "social.follow", target=nxt)
        _act(w, nxt, "social.follow", target=r)
    oc = I.organic_counts(w, P["star"])
    assert oc["organic"].get("follow", 0) == 0 and oc["engagers"].get("cluster") == 4, oc
    _act(w, P["indie"], "social.follow", target=ring[0])          # independent attention to one member
    oc2 = I.organic_counts(w, P["star"])
    assert oc2["organic"].get("follow", 0) == 4, oc2
