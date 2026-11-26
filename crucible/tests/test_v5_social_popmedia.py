"""Unit tests (workstream C, wave 3b): what population creators make besides text (`crucible.social.popmedia`), the
medium model (`crucible.social.popcreate`) and the random-exposure insertion (`crucible.social.contentopen`).

CPU only; no world (the renderers are pure functions of their inputs)."""
from __future__ import annotations

import io
import re

import numpy as np
import pytest

from crucible.sandbox import isolation

needs_jail = pytest.mark.skipif(not isolation.available()[0], reason="the media-tool jail cannot be applied here")


def _piece(n=6, topic=76, seed=1):
    from crucible.population import writing
    title, body = writing.compose(np.random.default_rng(seed), topic, n, 0.8)
    return title, [s for s in re.split(r"(?<=[.!?])\s+", body) if s], body


def test_style_is_stable_per_creator_and_json_native():
    import json
    from crucible.social import popmedia as PM
    a, b = PM.style_of(12345), PM.style_of(12345)
    assert a == b and json.loads(json.dumps(a)) == a
    styles = {json.dumps(PM.style_of(s), sort_keys=True) for s in range(40)}
    assert len(styles) > 20                              # creators differ in style


@needs_jail
def test_image_card_is_a_decodable_poster_of_the_title():
    from crucible.social import formats, popmedia as PM
    title, sents, _ = _piece()
    img = PM.image_card(title, sents[0], PM.style_of(3))
    f = formats.inspect(img, "image", "image/png")
    assert f["width"] == 448 and f["height"] == 448
    assert img != PM.image_card(title, sents[0], PM.style_of(4))


@needs_jail
def test_narration_is_speech_of_the_text():
    from crucible.social import formats, popmedia as PM
    title, _, body = _piece(4)
    a = PM.narration(f"{title}. {body}", PM.style_of(5))
    f = formats.inspect(a, "audio", "audio/ogg")
    words = len(f"{title} {body}".split())
    # flite speaks about 2-3.5 words per second; plus the jingle
    assert words / 4.0 < f["duration_s"] < words / 1.5 + 5


@needs_jail
def test_slideshow_has_slides_narration_and_subtitles():
    from crucible.semantic import media as MD
    from crucible.social import formats, popmedia as PM
    title, sents, _ = _piece(9)
    v = PM.slideshow(title, sents, PM.style_of(6), min_s=62.0)
    f = formats.inspect(v, "video", "video/mp4")
    assert f["duration_s"] >= 61.5 and f["has_audio"]
    subs = MD.extract_subtitles(v)
    assert title.split()[0] in subs and sents[-1].split()[-1].strip(".") in subs
    short = PM.slideshow(title, sents[:2], PM.style_of(6), max_s=170.0)
    assert formats.inspect(short, "video", "video/mp4")["duration_s"] <= 170.0
    with pytest.raises(MD.MediaError):
        PM.slideshow(title, sents, PM.style_of(6), max_s=5.0)


def test_medium_priors_follow_taste_and_ability():
    from crucible.social import popcreate as PC
    skills = np.full((2, 8), 0.5)
    prefs = np.full((2, 8), 0.5)
    skills[1, 4] = 0.98                                  # a very technical member
    pri = PC._priors(skills, prefs)
    tech = PC.MEDIUM_NAMES.index("tech")
    assert pri[1, tech] > pri[0, tech] + 1.0
    assert PC.medium_of("pics", "image") == PC.MEDIUM_NAMES.index("image")
    assert PC.medium_of("letters", "newsletter") == PC.MEDIUM_NAMES.index("article")
    assert PC.medium_of("kb", "thread") == PC.MEDIUM_NAMES.index("answer")


def test_random_insertion_places_one_uniform_recent_item_after_the_inbox():
    from crucible.social import audience as AUD
    from crucible.social import contentopen as CO
    cand = np.arange(100, 200, dtype=np.int64)
    V, k = 2000, 8
    feed = np.tile(np.arange(k, dtype=np.int64), (V, 1))
    src = np.full((V, k), 5, np.int8)
    gen = np.zeros((V, k), np.uint8)
    n_in = np.full(V, 2)
    CO.insert_into(feed, src, gen, n_in, cand, np.random.default_rng(1), 0.1)
    ins = src == AUD.SOURCE_CODE["random"]
    rows = ins.any(axis=1)
    assert 0.07 < rows.mean() < 0.13                      # about `rate` of the feeds
    assert ins.sum(axis=1).max() == 1                    # one slot at most
    assert not ins[:, :2].any()                          # never in the share inbox slots
    assert np.all((feed[ins] >= 100) & (feed[ins] < 200))
    picked = feed[ins]
    assert len(np.unique(picked)) > 60                   # spread over the candidates (uniform draw)
