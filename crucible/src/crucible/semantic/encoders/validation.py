"""Controlled artifact suites for validating the encoders (Gate E): for each modality, base artifacts with genuinely
different content, nuisance variants of each base that keep its content (re-encoding, resizing, noise, trimming,
reformatting, paraphrase), and semantic variants that change it.

Every artifact is generated locally and deterministically with the media tools (rendered text and HTML, drawn
scenes, flite speech, synthesised tones, melodies and noise, FFmpeg-assembled videos, source trees), so the suites
need no dataset and no network. `measure` reports, per modality, the mean cosine distance of semantic pairs and of
nuisance pairs, their ratio (the separation ratio), the probability that a semantic pair is farther apart than a
nuisance pair (AUC), and the same per variant kind and semantic group.

    from crucible.semantic.encoders import validation as V
    res = V.run()                      # all modalities: {modality: {...}}
    res["audio"]["separation_ratio"]
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .. import media as M


@dataclass
class Suite:
    modality: str
    items: dict[str, bytes] = field(default_factory=dict)
    nuisance: list[tuple[str, str, str]] = field(default_factory=list)      # (base, variant, kind)
    semantic: list[tuple[str, str, str]] = field(default_factory=list)      # (a, b, group)

    def add_base(self, name: str, data: bytes) -> None:
        self.items[name] = data

    def add_nuisance(self, base: str, kind: str, data: bytes) -> None:
        n = f"{base}~{kind}"
        self.items[n] = data
        self.nuisance.append((base, n, kind))

    def bases(self) -> list[str]:
        return [n for n in self.items if "~" not in n and "!" not in n]

    def cross_pairs(self, names: Sequence[str], group: str = "different_content") -> None:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.semantic.append((names[i], names[j], group))


# ============================================================================================== text
TEXTS = {
    "transport": ("The city council voted on Tuesday to expand the public bus network, adding three new routes to the "
                  "northern suburbs next spring.",
                  "On Tuesday the city council approved an expansion of public bus services, with three additional "
                  "routes serving the northern suburbs from next spring.",
                  "The city council voted on Tuesday to close the public swimming pool and sell the land to a hotel "
                  "developer next spring."),
    "cooking": ("Preheat the oven to 180 degrees, whisk the eggs with the sugar, fold in the flour and bake the sponge "
                "cake for 25 minutes.",
                "Heat the oven to 180 degrees. Beat the eggs and sugar together, gently fold in the flour, then bake "
                "the sponge for about 25 minutes.",
                "Bring a large pot of salted water to the boil, cook the spaghetti for nine minutes and toss it with "
                "garlic, olive oil and chilli."),
    "sports": ("The striker scored twice in the second half, securing the championship title for the club after a "
               "tense final.",
               "Two second-half goals from the striker clinched the championship for the club at the end of a nervy "
               "final.",
               "The marathon was cancelled an hour before the start because of the heat, and runners were offered a "
               "refund of their entry fee."),
    "battery": ("Researchers report that the new battery chemistry keeps 90 percent of its capacity after two thousand "
                "charge cycles.",
                "According to the researchers, the new battery chemistry still holds 90 percent of its original "
                "capacity after 2,000 charging cycles.",
                "Researchers report that a new coating makes rooftop solar panels produce twelve percent more "
                "electricity on cloudy days."),
    "retail": ("Quarterly revenue fell by eight percent as the retailer struggled with rising shipping costs and "
               "weaker consumer demand.",
               "The retailer's revenue for the quarter dropped eight percent, hurt by higher shipping costs and softer "
               "demand from shoppers.",
               "The retailer will open forty new stores next year and hire two thousand staff after record holiday "
               "sales."),
    "health": ("Physiotherapists say regular stretching and a short daily walk can reduce lower back pain.",
               "A short walk every day and regular stretching help ease lower back pain, according to "
               "physiotherapists.",
               "Doctors warn that children who sleep less than eight hours a night struggle to concentrate at school."),
    "museum": ("The museum's new exhibition shows impressionist paintings on loan from collections across Europe.",
               "Impressionist paintings borrowed from European collections are on display in the museum's latest "
               "exhibition.",
               "The museum will close for two years while its leaking roof is replaced, and its sculptures will go "
               "into storage."),
    "weather": ("Heavy rain is expected over the weekend, and flood warnings have been issued for low-lying coastal "
                "areas.",
                "Forecasters expect heavy rain this weekend and have put flood warnings in place for low-lying areas "
                "along the coast.",
                "A long heatwave is expected over the weekend, and residents are asked to save water because of the "
                "drought."),
}


def _typos(s: str, n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    w = s.split(" ")
    cand = [i for i, x in enumerate(w) if len(x) > 5 and x.isalpha()]
    for i in rng.choice(cand, size=min(n, len(cand)), replace=False):
        x = w[i]
        k = int(rng.integers(1, len(x) - 2))
        w[i] = x[:k] + x[k + 1] + x[k] + x[k + 2:]
    return " ".join(w)


def _reformat(s: str) -> str:
    words = s.rstrip(".").split(" ")
    mid = len(words) // 2
    return "  " + " ".join(words[:mid]) + "\n\n   " + "  ".join(words[mid:]) + "  \n"


def text_suite() -> Suite:
    s = Suite("text")
    names = []
    for i, (name, (base, para, changed)) in enumerate(TEXTS.items()):
        s.add_base(name, base.encode())
        names.append(name)
        s.add_nuisance(name, "paraphrase", para.encode())
        s.add_nuisance(name, "whitespace_punctuation", _reformat(base).encode())
        s.add_nuisance(name, "typos", _typos(base, 2, seed=i).encode())
        s.add_nuisance(name, "html_wrapped", (f"<!doctype html><html><body><article><p>{base}</p></article></body>"
                                              f"</html>").encode())
        s.items[f"{name}!changed"] = changed.encode()
        s.semantic.append((name, f"{name}!changed", "same_opening_different_event"))
    s.cross_pairs(names, "different_topic")
    return s


# ============================================================================================== images
def _scenes() -> dict[str, bytes]:
    W = 448
    sc = {}
    sc["day_landscape"] = M.draw_scene({"background": "#87ceeb", "shapes": [
        {"kind": "circle", "xy": [0.68, 0.08, 0.88, 0.28], "fill": "#ffd700"},
        {"kind": "rect", "xy": [0, 0.68, 1, 1], "fill": "#3a9d23"},
        {"kind": "triangle", "xy": [0.1, 0.7, 0.2, 0.42, 0.3, 0.7], "fill": "#1e5e1e"},
        {"kind": "triangle", "xy": [0.3, 0.7, 0.4, 0.38, 0.5, 0.7], "fill": "#1e5e1e"},
        {"kind": "rect", "xy": [0.18, 0.7, 0.22, 0.78], "fill": "#6b3e1e"},
        {"kind": "rect", "xy": [0.38, 0.7, 0.42, 0.78], "fill": "#6b3e1e"}]}, width=W, height=W)
    sc["night_city"] = M.draw_scene({"background": "#0b1030", "shapes": [
        {"kind": "circle", "xy": [0.75, 0.06, 0.9, 0.21], "fill": "#f0f0e0"},
        *[{"kind": "rect", "xy": [0.05 + 0.18 * i, 0.35 + 0.08 * (i % 3), 0.2 + 0.18 * i, 1], "fill": "#3c3c4a"}
          for i in range(5)],
        *[{"kind": "rect", "xy": [0.08 + 0.18 * i + 0.05 * k, 0.45 + 0.1 * j, 0.11 + 0.18 * i + 0.05 * k,
                                  0.5 + 0.1 * j], "fill": "#ffe066"}
          for i in range(5) for j in range(4) for k in range(2) if (i + j + k) % 3]]}, width=W, height=W)
    sc["red_apple"] = M.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "circle", "xy": [0.22, 0.28, 0.78, 0.86], "fill": "#d01818"},
        {"kind": "line", "xy": [0.5, 0.3, 0.53, 0.14], "fill": "#5a3210", "width": 9},
        {"kind": "circle", "xy": [0.54, 0.12, 0.72, 0.22], "fill": "#2e9c2e"}]}, width=W, height=W)
    sc["smiley_face"] = M.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "circle", "xy": [0.12, 0.12, 0.88, 0.88], "fill": "#ffd21f", "outline": "#000000", "width": 4},
        {"kind": "circle", "xy": [0.32, 0.32, 0.4, 0.44], "fill": "#000000"},
        {"kind": "circle", "xy": [0.6, 0.32, 0.68, 0.44], "fill": "#000000"},
        {"kind": "line", "xy": [0.3, 0.62, 0.4, 0.7, 0.5, 0.72, 0.6, 0.7, 0.7, 0.62], "fill": "#000000", "width": 8}]},
        width=W, height=W)
    sc["bar_chart"] = M.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "line", "xy": [0.1, 0.9, 0.92, 0.9], "fill": "#000000", "width": 3},
        {"kind": "line", "xy": [0.1, 0.1, 0.1, 0.9], "fill": "#000000", "width": 3},
        *[{"kind": "rect", "xy": [0.16 + 0.15 * i, 0.9 - h, 0.26 + 0.15 * i, 0.9], "fill": c}
          for i, (h, c) in enumerate([(0.3, "#4472c4"), (0.55, "#ed7d31"), (0.42, "#a5a5a5"), (0.7, "#ffc000"),
                                      (0.62, "#5b9bd5")])],
        {"kind": "text", "xy": [0.3, 0.02], "text": "Sales by region", "size": 26, "fill": "#000000"}]},
        width=W, height=W)
    sc["sale_poster"] = M.render_text_image("BIG SALE\n50% OFF\nEVERYTHING", width=W, height=W, font_size=64,
                                            color="#d00000", background="#ffe600", align="center", bold=True)
    sc["text_page"] = M.render_text_image(TEXTS["transport"][0] + " " + TEXTS["weather"][0], width=W, height=W,
                                          font_size=22, color="#111111", background="#ffffff")
    sc["sailboat"] = M.draw_scene({"background": "#bfe6ff", "shapes": [
        {"kind": "rect", "xy": [0, 0.6, 1, 1], "fill": "#1f4e9c"},
        {"kind": "polygon", "xy": [0.3, 0.6, 0.7, 0.6, 0.62, 0.68, 0.38, 0.68], "fill": "#7a4a1d"},
        {"kind": "triangle", "xy": [0.5, 0.18, 0.5, 0.58, 0.68, 0.58], "fill": "#ffffff"},
        {"kind": "triangle", "xy": [0.48, 0.25, 0.48, 0.58, 0.36, 0.58], "fill": "#f4f4f4"}]}, width=W, height=W)
    # semantic changes of a base: same layout, different content
    sc["day_landscape!night"] = M.draw_scene({"background": "#0a0f2a", "shapes": [
        {"kind": "circle", "xy": [0.7, 0.08, 0.85, 0.23], "fill": "#f2f2e6"},
        {"kind": "rect", "xy": [0, 0.68, 1, 1], "fill": "#0e2a12"},
        {"kind": "triangle", "xy": [0.1, 0.7, 0.2, 0.42, 0.3, 0.7], "fill": "#061a08"},
        {"kind": "triangle", "xy": [0.3, 0.7, 0.4, 0.38, 0.5, 0.7], "fill": "#061a08"}]}, width=W, height=W)
    sc["red_apple!banana"] = M.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "polygon", "xy": [0.15, 0.35, 0.3, 0.6, 0.5, 0.7, 0.7, 0.66, 0.86, 0.5, 0.82, 0.62, 0.66, 0.78,
                                   0.46, 0.8, 0.26, 0.7, 0.14, 0.5], "fill": "#f4d03f", "outline": "#8a6d00"},
        {"kind": "line", "xy": [0.15, 0.35, 0.12, 0.3], "fill": "#4a3a10", "width": 6}]}, width=W, height=W)
    sc["sale_poster!closed"] = M.render_text_image("CLOSED\nFOR\nRENOVATION", width=W, height=W, font_size=64,
                                                   color="#ffffff", background="#202020", align="center", bold=True)
    sc["bar_chart!pie"] = M.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "polygon", "xy": [0.5, 0.5] + [v for a in np.linspace(0, 2.2, 12)
                                                 for v in (0.5 + 0.35 * math.cos(a), 0.5 + 0.35 * math.sin(a))],
         "fill": "#4472c4"},
        {"kind": "polygon", "xy": [0.5, 0.5] + [v for a in np.linspace(2.2, 4.2, 12)
                                                 for v in (0.5 + 0.35 * math.cos(a), 0.5 + 0.35 * math.sin(a))],
         "fill": "#ed7d31"},
        {"kind": "polygon", "xy": [0.5, 0.5] + [v for a in np.linspace(4.2, 2 * math.pi, 12)
                                                 for v in (0.5 + 0.35 * math.cos(a), 0.5 + 0.35 * math.sin(a))],
         "fill": "#a5a5a5"}]}, width=W, height=W)
    sc["sailboat!car"] = M.draw_scene({"background": "#bfe6ff", "shapes": [
        {"kind": "rect", "xy": [0, 0.62, 1, 1], "fill": "#555555"},
        {"kind": "rect", "xy": [0, 0.78, 1, 0.8], "fill": "#ffffff"},
        {"kind": "rect", "xy": [0.25, 0.45, 0.75, 0.62], "fill": "#c0392b"},
        {"kind": "polygon", "xy": [0.35, 0.45, 0.42, 0.34, 0.6, 0.34, 0.66, 0.45], "fill": "#c0392b"},
        {"kind": "polygon", "xy": [0.4, 0.44, 0.44, 0.37, 0.58, 0.37, 0.62, 0.44], "fill": "#cde7f7"},
        {"kind": "circle", "xy": [0.3, 0.56, 0.42, 0.68], "fill": "#111111"},
        {"kind": "circle", "xy": [0.58, 0.56, 0.7, 0.68], "fill": "#111111"}]}, width=W, height=W)
    return sc


IMAGE_NUISANCE = {
    "jpeg_q60": [{"op": "encode", "fmt": "JPEG", "codec_quality": 60}],
    "half_size": [{"op": "scale", "f": 0.5}],
    "noise_sigma6": [{"op": "noise", "sigma": 6, "seed": 3}],
    "crop_4pct": [{"op": "crop", "box": [0.04, 0.04, 0.96, 0.96]}],
    "brighter_10pct": [{"op": "brightness", "f": 1.1}],
    "webp_q70_resized": [{"op": "resize", "w": 300, "h": 300}, {"op": "encode", "fmt": "WEBP", "codec_quality": 70}],
}


def image_suite() -> Suite:
    s = Suite("image")
    sc = _scenes()
    names = [n for n in sc if "!" not in n]
    for n in names:
        s.add_base(n, sc[n])
        for kind, ops in IMAGE_NUISANCE.items():
            s.add_nuisance(n, kind, M.process_image(sc[n], ops))
    for n, data in sc.items():
        if "!" in n:
            s.items[n] = data
            s.semantic.append((n.split("!")[0], n, "same_layout_different_content"))
    s.cross_pairs(names, "different_scene")
    return s


# ============================================================================================== audio
SR = 16000
SPEECH = {
    "speech_weather": ("The weather tomorrow will be sunny with light winds in the afternoon.", "slt"),
    "speech_finance": ("Shares in the bank fell sharply after it reported a large loss for the third quarter.", "kal"),
    "speech_cooking": ("Chop the onions finely and fry them gently in butter until they turn golden.", "awb"),
}


def _beeps(seconds: float, freq: float = 1000.0, on: float = 0.2, off: float = 0.2) -> np.ndarray:
    x = M.synth_tone(freq, seconds, sr=SR, amp=0.5)
    t = np.arange(len(x)) / SR
    return (x * ((t % (on + off)) < on)).astype(np.float32)


def _audio_signals() -> dict[str, np.ndarray]:
    sig = {}
    for name, (text, voice) in SPEECH.items():
        sig[name] = M.decode_audio(M.synth_speech(text, voice=voice), SR)
    sig["tone_440"] = M.synth_tone(440.0, 5.0, sr=SR, amp=0.5)
    sig["beeps_1k"] = _beeps(5.0)
    sig["melody_drums"] = M.synth_melody(["C4", "E4", "G4", "C5", "B4", "G4", "E4", "D4"] * 2, bpm=150, sr=SR,
                                         drums=True)
    sig["organ_chords"] = M.synth_melody([("C3", "E3", "G3"), ("A2", "C3", "E3"), ("F2", "A2", "C3"),
                                          ("G2", "B2", "D3")], bpm=48, beats=1.0, sr=SR,
                                         timbre=(1.0, 0.8, 0.6, 0.5, 0.4, 0.3), decay=0.3)
    sig["pink_noise"] = M.synth_noise(5.0, color="pink", sr=SR, seed=1)
    # semantic changes of a base
    sig["speech_weather!other_words"] = M.decode_audio(M.synth_speech(
        "Please remember to return your library books before the end of the month.", voice="slt"), SR)
    sig["melody_drums!other_song"] = M.synth_melody(["A3", None, "A3", "C4", "D4", None, "F4", "E4"] * 2, bpm=96,
                                                    sr=SR, timbre=(1.0, 0.2, 0.05), decay=6.0, drums=True)
    sig["tone_440!speech"] = M.decode_audio(M.synth_speech("Four hundred and forty hertz is the note A above middle "
                                                           "C.", voice="rms"), SR)
    return sig


def _audio_nuisance(x: np.ndarray) -> dict[str, bytes]:
    wav = M.wav_bytes(x, SR)
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(len(x)).astype(np.float32) * (10 ** (-35 / 20)) * max(1e-6, float(np.max(np.abs(x))))
    return {"mp3_64k": M.transcode(wav, "mp3", audio_bitrate="64k"),
            "ogg_vorbis": M.transcode(wav, "ogg"),
            "resampled_22k": M.transcode(wav, "wav", sample_rate=22050),
            "gain_minus6db": M.wav_bytes(x * 0.5, SR),
            "noise_minus35db": M.wav_bytes(x + noise, SR),
            "trim_start_5pct": M.wav_bytes(x[int(0.05 * len(x)):], SR)}


def audio_suite() -> Suite:
    s = Suite("audio")
    sig = _audio_signals()
    names = [n for n in sig if "!" not in n]
    for n in names:
        s.add_base(n, M.wav_bytes(sig[n], SR))
        for kind, data in _audio_nuisance(sig[n]).items():
            s.add_nuisance(n, kind, data)
    groups = {"speech_weather!other_words": "same_voice_other_words", "melody_drums!other_song": "other_song",
              "tone_440!speech": "tone_replaced_by_speech"}
    for n, x in sig.items():
        if "!" in n:
            s.items[n] = M.wav_bytes(x, SR)
            s.semantic.append((n.split("!")[0], n, groups[n]))
    s.cross_pairs(names, "different_sound")
    return s


# ============================================================================================== video
def _slide(text: str, bg: str = "#ffffff", fg: str = "#111111") -> bytes:
    return M.render_text_image(text, width=480, height=360, font_size=40, color=fg, background=bg, align="center")


def video_suite() -> Suite:
    s = Suite("video")
    sc = _scenes()
    sig = _audio_signals()
    wav = {k: M.wav_bytes(v, SR) for k, v in sig.items()}
    news_speech = M.synth_speech("The city council has voted to expand the bus network with three new routes to the "
                                 "northern suburbs.", voice="slt")
    cook_speech = M.synth_speech("Today we are baking a sponge cake. Whisk the eggs with sugar and fold in the flour.",
                                 voice="awb")
    match_speech = M.synth_speech("What a goal! The striker scores again and the home team takes the lead.", voice="kal")
    crowd = M.wav_bytes(M.mix(M.decode_audio(match_speech, SR), M.synth_noise(6.0, color="pink", sr=SR, seed=5),
                              gains=(1.0, 0.35)), SR)
    football = M.draw_scene({"background": "#2e8b2e", "shapes": [
        {"kind": "rect", "xy": [0.05, 0.1, 0.95, 0.9], "outline": "#ffffff", "width": 4},
        {"kind": "line", "xy": [0.5, 0.1, 0.5, 0.9], "fill": "#ffffff", "width": 4},
        {"kind": "circle", "xy": [0.42, 0.42, 0.58, 0.58], "outline": "#ffffff", "width": 4},
        {"kind": "circle", "xy": [0.62, 0.55, 0.67, 0.61], "fill": "#ffffff"}]}, width=480, height=360)
    shapes = [M.draw_scene({"background": bg, "shapes": [
        {"kind": "circle", "xy": [0.1 + 0.1 * k, 0.2, 0.4 + 0.1 * k, 0.5], "fill": c1},
        {"kind": "rect", "xy": [0.5, 0.4 - 0.05 * k, 0.9, 0.8], "fill": c2},
        {"kind": "triangle", "xy": [0.2, 0.9, 0.35, 0.6, 0.5, 0.9], "fill": c3}]}, width=480, height=360)
        for k, (bg, c1, c2, c3) in enumerate([("#2b0a3d", "#ff3cac", "#2bd2ff", "#fffb7d"),
                                              ("#0a3d2b", "#ffb13c", "#d22bff", "#7dfff1"),
                                              ("#3d0a0a", "#3cff9e", "#ff2b5e", "#7d9bff")])]
    bases = {
        "news_report": ([_slide("CITY NEWS\nBus network expands"), sc["night_city"], _slide("3 new routes\nfrom spring")],
                        news_speech),
        "cooking_show": ([_slide("Baking a\nsponge cake", "#fff4e0"), sc["red_apple"], _slide("180 degrees\n25 minutes",
                                                                                              "#fff4e0")], cook_speech),
        "music_video": (shapes, wav["melody_drums"]),
        "nature_film": ([sc["day_landscape"], sc["sailboat"], sc["day_landscape"]], wav["pink_noise"]),
        "chart_slideshow": ([sc["bar_chart"], sc["bar_chart!pie"], _slide("Q3 results\nrevenue +4%")], None),
        "football_match": ([football, _slide("HOME 2 - 1 AWAY", "#0b3d0b", "#ffffff"), football], crowd),
    }
    vids = {n: M.assemble_video(imgs, au, seconds_per_image=1.6, size=(480, 360), crf=26)
            for n, (imgs, au) in bases.items()}
    for n, v in vids.items():
        s.add_base(n, v)
        s.add_nuisance(n, "reencoded_crf36", M.transcode(v, "mp4", crf=36))
        s.add_nuisance(n, "half_size", M.transcode(v, "mp4", width=240, crf=26))
        s.add_nuisance(n, "trim_start_0.4s", M.transcode(v, "mp4", start_s=0.4, crf=26))
        s.add_nuisance(n, "webm_vp9", M.transcode(v, "webm", crf=36))
    # semantic changes: the same pictures with another soundtrack, the same soundtrack with other pictures
    s.items["news_report!music_audio"] = M.assemble_video(bases["news_report"][0], wav["melody_drums"],
                                                          seconds_per_image=1.6, size=(480, 360), crf=26)
    s.semantic.append(("news_report", "news_report!music_audio", "same_pictures_other_soundtrack"))
    s.items["music_video!chart_pictures"] = M.assemble_video(bases["chart_slideshow"][0], wav["melody_drums"],
                                                             seconds_per_image=1.6, size=(480, 360), crf=26)
    s.semantic.append(("music_video", "music_video!chart_pictures", "same_soundtrack_other_pictures"))
    s.cross_pairs(list(bases), "different_video")
    return s


# ============================================================================================== code
CODE_BASES = {
    "sorting": {"sorting.py": '''"""Sorting algorithms for lists of comparable items."""


def quicksort(items):
    """Return a new list with the items in ascending order (quicksort)."""
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    smaller = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    larger = [x for x in items if x > pivot]
    return quicksort(smaller) + equal + quicksort(larger)


def merge_sort(items):
    """Stable merge sort."""
    if len(items) <= 1:
        return list(items)
    middle = len(items) // 2
    left, right = merge_sort(items[:middle]), merge_sort(items[middle:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        # take from the left run on ties to keep the sort stable
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    return merged + left[i:] + right[j:]
'''},
    "weather_service": {"crucible.json": json.dumps({"entry": "service.py", "description": "Weather forecasts and "
                                                     "severe weather alerts for cities.", "endpoints": {
                                                         "forecast": {"handler": "forecast", "params": {}},
                                                         "alerts": {"handler": "alerts", "params": {}}}}, indent=1),
                        "service.py": '''"""A weather forecast service."""
import crucible_sdk

FORECASTS = {"london": "rain", "madrid": "sunny", "oslo": "snow"}


def forecast(args, caller):
    """Forecast for a city."""
    city = args.get("city", "").lower()
    return {"city": city, "forecast": FORECASTS.get(city, "unknown")}


def alerts(args, caller):
    """Severe weather alerts (storms, floods) for a city."""
    city = args.get("city", "").lower()
    warnings = [w for w in crucible_sdk.call("db.query", collection="alerts") if w.get("city") == city]
    return {"city": city, "alerts": warnings}
'''},
    "todo_app": {"index.html": "<!doctype html><html><head><title>Todo list</title></head><body><h1>My todo list</h1>"
                               "<ul id='todos'></ul><input id='new'><button id='add'>Add task</button>"
                               "<script src='app.js'></script></body></html>",
                 "app.js": '''// Todo list: add, complete and remove tasks in the browser
const todos = [];

function addTodo(title) {
  todos.push({ title: title, done: false });
  renderTodos();
}

function toggleTodo(index) {
  todos[index].done = !todos[index].done;
  renderTodos();
}

function removeTodo(index) {
  todos.splice(index, 1);
  renderTodos();
}

function renderTodos() {
  const list = document.getElementById("todos");
  list.innerHTML = "";
  todos.forEach((t, i) => {
    const li = document.createElement("li");
    li.textContent = (t.done ? "[x] " : "[ ] ") + t.title;
    li.onclick = () => toggleTodo(i);
    list.appendChild(li);
  });
}
'''},
    "csv_report": {"report.py": '''"""Summarise a CSV file of expenses by category and write a text report."""
import csv
from collections import defaultdict


def totals_by_category(path):
    """Sum the amount column for each category."""
    totals = defaultdict(float)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            totals[row["category"]] += float(row["amount"])
    return dict(totals)


def write_report(totals, out_path):
    """Write one line per category, largest first."""
    with open(out_path, "w") as out:
        for category, amount in sorted(totals.items(), key=lambda kv: -kv[1]):
            out.write(f"{category}: {amount:.2f}\\n")
'''},
    "linear_algebra": {"matrix.py": '''"""Small dense matrix helpers."""


def transpose(m):
    """Swap rows and columns."""
    return [list(r) for r in zip(*m)]


def matmul(a, b):
    """Matrix product of a (n x k) and b (k x m)."""
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def determinant(m):
    """Determinant by Laplace expansion along the first row."""
    if len(m) == 1:
        return m[0][0]
    total = 0
    for j, v in enumerate(m[0]):
        minor = [r[:j] + r[j + 1:] for r in m[1:]]
        total += (-1) ** j * v * determinant(minor)
    return total
'''},
    "checksum_tool": {"crc.c": '''/* crc32: print the CRC-32 checksum of a file */
#include <stdio.h>
#include <stdint.h>

static uint32_t crc_table[256];

/* build the lookup table for the reflected polynomial */
static void make_table(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
}

uint32_t crc32_file(FILE *f) {
    uint32_t crc = 0xFFFFFFFFu;
    int ch;
    while ((ch = fgetc(f)) != EOF) crc = crc_table[(crc ^ (uint32_t)ch) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    make_table();
    printf("%08x\\n", crc32_file(f));
    fclose(f);
    return 0;
}
'''},
}

CODE_RENAMES = {"sorting": [("smaller", "lo"), ("larger", "hi"), ("merged", "out"), ("middle", "mid")],
                "weather_service": [("warnings", "found"), ("city", "place")],
                "todo_app": [("list", "ul"), ("li", "item")],
                "csv_report": [("totals", "sums"), ("row", "record"), ("out", "dst")],
                "linear_algebra": [("bt", "cols"), ("minor", "sub"), ("total", "acc")],
                "checksum_tool": [("crc", "acc"), ("ch", "byte")]}

CODE_CHANGED = {"sorting": {"searching.py": '''"""Searching sorted lists."""


def binary_search(items, target):
    """Index of target in the sorted list, or -1 when it is absent."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''},
                "csv_report": {"report.py": '''"""Send a monthly newsletter email to every subscriber."""
import smtplib
from email.message import EmailMessage


def send_newsletter(subscribers, subject, body, server="localhost"):
    """Email the newsletter to each subscriber address."""
    with smtplib.SMTP(server) as smtp:
        for address in subscribers:
            msg = EmailMessage()
            msg["To"], msg["Subject"] = address, subject
            msg.set_content(body)
            smtp.send_message(msg)
'''}}


def _rename(files: dict[str, str], pairs) -> dict[str, str]:
    import re
    out = {}
    for p, t in files.items():
        for a, b in pairs:
            t = re.sub(rf"\b{re.escape(a)}\b", b, t)
        out[p] = t
    return out


def _reformat_code(files: dict[str, str]) -> dict[str, str]:
    out = {}
    for p, t in files.items():
        if p.endswith(".py"):
            t = t.replace("\n\n\n", "\n\n\n\n").replace("    ", "  ")      # consistent 2-space indentation
        elif p.endswith((".js", ".c")):
            t = t.replace("  ", "    ").replace(") {", ")\n{")
        out[p] = t
    return out


def _files_json(files: dict[str, str]) -> bytes:
    return json.dumps({"files": files}, sort_keys=True).encode()


def _files_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files):
            zi = zipfile.ZipInfo("project/" + p, date_time=(2020, 1, 1, 0, 0, 0))
            zf.writestr(zi, files[p])
    return buf.getvalue()


def code_suite() -> Suite:
    s = Suite("code")
    for n, files in CODE_BASES.items():
        s.add_base(n, _files_json(files))
        s.add_nuisance(n, "renamed_locals", _files_json(_rename(files, CODE_RENAMES[n])))
        s.add_nuisance(n, "reformatted", _files_json(_reformat_code(files)))
        s.add_nuisance(n, "zip_packaged", _files_zip(files))
    for n, files in CODE_CHANGED.items():
        s.items[f"{n}!changed"] = _files_json(files)
        s.semantic.append((n, f"{n}!changed", "same_project_different_function"))
    s.cross_pairs(list(CODE_BASES), "different_program")
    return s


SUITES: dict[str, Callable[[], Suite]] = {"text": text_suite, "image": image_suite, "audio": audio_suite,
                                          "video": video_suite, "code": code_suite}


# ============================================================================================== measurement
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(1.0 - a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def measure(suite: Suite, vectors: dict[str, np.ndarray]) -> dict:
    nd = [(k, cosine_distance(vectors[a], vectors[b])) for a, b, k in suite.nuisance]
    sd = [(g, cosine_distance(vectors[a], vectors[b])) for a, b, g in suite.semantic]
    nv = np.array([d for _, d in nd])
    sv = np.array([d for _, d in sd])
    auc = float(np.mean(sv[:, None] > nv[None, :]) + 0.5 * np.mean(sv[:, None] == nv[None, :]))
    by_kind = {k: round(float(np.mean([d for kk, d in nd if kk == k])), 5) for k in dict.fromkeys(k for k, _ in nd)}
    by_group = {g: round(float(np.mean([d for gg, d in sd if gg == g])), 5) for g in dict.fromkeys(g for g, _ in sd)}
    mean_n = float(nv.mean())
    return {"modality": suite.modality, "n_items": len(suite.items), "n_nuisance_pairs": len(nd),
            "n_semantic_pairs": len(sd), "mean_nuisance_distance": round(mean_n, 5),
            "mean_semantic_distance": round(float(sv.mean()), 5),
            "separation_ratio": round(float(sv.mean()) / max(mean_n, 1e-9), 4),
            "median_ratio": round(float(np.median(sv)) / max(float(np.median(nv)), 1e-9), 4),
            "auc_semantic_farther": round(auc, 4),
            "max_nuisance_distance": round(float(nv.max()), 5), "min_semantic_distance": round(float(sv.min()), 5),
            "nuisance_by_kind": by_kind, "semantic_by_group": by_group,
            "group_ratios": {g: round(v / max(mean_n, 1e-9), 3) for g, v in by_group.items()}}


def run(modalities: Sequence[str] = ("text", "image", "audio", "video", "code"),
        encode_fn: Callable[[list[bytes], str], np.ndarray] | None = None) -> dict:
    """Build each suite, encode every artifact (batched), and measure separation."""
    from . import encode_bytes_many
    enc = encode_fn or (lambda datas, m: encode_bytes_many(datas, m))
    out = {}
    for m in modalities:
        suite = SUITES[m]()
        names = list(suite.items)
        mat = enc([suite.items[n] for n in names], m)
        out[m] = measure(suite, dict(zip(names, mat)))
    return out
