"""Fitting and measuring the image alignment map (run once; the map is committed under `space/maps/`).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --gpu-mb 3000 --label "D: space fit" -- \
        python -m crucible.semantic.space fit --report <out.json>

Data: MIND news titles and abstracts (real text, `corpus.news()`), split deterministically by news id hash into
FIT (ridge regression pairs), DEV (choice of the gap correction and of the ridge/concept blend) and TEST (reported
only). Images for DEV/TEST:

* rendered news titles - text in images (posters, thumbnails, screenshots): the title drawn with varied size, colour
  and background (`media.render_text_image`); the caption is the title itself;
* drawn scenes with captions: the 13 scenes of the Gate E image suite (DEV) and 8 other drawn scenes (TEST);
* the three natural photographs shipped with matplotlib and scikit-learn (TEST).

Retrieval: each image's mapped topic vector against the MiniLM vectors of all captions of its set (R@1, R@5, MRR);
CLIP's own image-text retrieval on the same sets is reported as the reference.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from .. import media as MD
from ..encoders.image import IMAGE
from ..encoders.text import TEXT
from . import align as A
from . import corpus as C

N_FIT, N_DEV, N_TEST = 24000, 200, 300


def _bucket(nid: str) -> int:
    return int(hashlib.sha256(nid.encode()).hexdigest()[:8], 16) % 100


def splits() -> dict[str, list[str]]:
    """News ids per split (FIT: buckets 0-79, DEV 80-89, TEST 90-99), sorted, capped."""
    ids = sorted(C.news())
    out = {"fit": [], "dev": [], "test": []}
    for nid in ids:
        b = _bucket(nid)
        out["fit" if b < 80 else "dev" if b < 90 else "test"].append(nid)
    out["fit"] = out["fit"][:N_FIT]
    return out


def rendered_titles(ids: list[str], seed: int) -> tuple[list[bytes], list[str]]:
    rng = np.random.default_rng(seed)
    news = C.news()
    palette = [("#111111", "#ffffff"), ("#ffffff", "#1d3557"), ("#d00000", "#ffe600"), ("#003049", "#eae2b7"),
               ("#ffffff", "#2a9d8f"), ("#222222", "#f4a261")]
    imgs, caps = [], []
    for nid in ids:
        title = news[nid][2]
        fg, bg = palette[int(rng.integers(0, len(palette)))]
        imgs.append(MD.render_text_image(title, width=448, height=448, font_size=int(rng.integers(30, 48)), color=fg,
                                         background=bg, align="center", bold=bool(rng.integers(0, 2))))
        caps.append(title)
    return imgs, caps


SCENE_CAPTIONS_DEV = {
    "day_landscape": "a sunny day in the countryside with green trees and grass",
    "night_city": "a city skyline at night with lit windows and the moon",
    "red_apple": "a red apple", "smiley_face": "a yellow smiley face",
    "bar_chart": "a bar chart of sales by region", "sale_poster": "a big sale poster: 50 percent off everything",
    "text_page": "a page of text about new bus routes and the weather forecast", "sailboat": "a sailboat on the sea",
    "day_landscape!night": "a dark landscape with trees at night under the moon", "red_apple!banana": "a yellow banana",
    "sale_poster!closed": "a sign saying the shop is closed for renovation", "bar_chart!pie": "a pie chart",
    "sailboat!car": "a red car on a road",
}


def test_scenes() -> dict[str, tuple[bytes, str]]:
    W = 448
    d = {}
    d["house"] = (MD.draw_scene({"background": "#cfefff", "shapes": [
        {"kind": "rect", "xy": [0, 0.75, 1, 1], "fill": "#4caf50"},
        {"kind": "rect", "xy": [0.25, 0.45, 0.75, 0.78], "fill": "#e9c46a"},
        {"kind": "triangle", "xy": [0.2, 0.46, 0.5, 0.2, 0.8, 0.46], "fill": "#b23a48"},
        {"kind": "rect", "xy": [0.45, 0.6, 0.55, 0.78], "fill": "#6d4c41"},
        {"kind": "rect", "xy": [0.3, 0.52, 0.4, 0.6], "fill": "#ffffff"}]}, width=W, height=W), "a small house with a red roof")
    d["pizza"] = (MD.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "circle", "xy": [0.1, 0.1, 0.9, 0.9], "fill": "#e0a040"},
        {"kind": "circle", "xy": [0.15, 0.15, 0.85, 0.85], "fill": "#d63a1c"},
        *[{"kind": "circle", "xy": [x, y, x + 0.1, y + 0.1], "fill": "#8b1a1a"}
          for x, y in [(0.3, 0.3), (0.55, 0.28), (0.4, 0.55), (0.62, 0.58), (0.25, 0.6), (0.5, 0.42)]]]},
        width=W, height=W), "a pizza with pepperoni")
    d["traffic_light"] = (MD.draw_scene({"background": "#dddddd", "shapes": [
        {"kind": "rect", "xy": [0.38, 0.1, 0.62, 0.8], "fill": "#222222"},
        {"kind": "circle", "xy": [0.42, 0.14, 0.58, 0.32], "fill": "#ff2020"},
        {"kind": "circle", "xy": [0.42, 0.37, 0.58, 0.55], "fill": "#553300"},
        {"kind": "circle", "xy": [0.42, 0.6, 0.58, 0.78], "fill": "#003300"},
        {"kind": "rect", "xy": [0.48, 0.8, 0.52, 1], "fill": "#444444"}]}, width=W, height=W), "a traffic light")
    d["line_graph"] = (MD.draw_scene({"background": "#ffffff", "shapes": [
        {"kind": "line", "xy": [0.1, 0.9, 0.92, 0.9], "fill": "#000000", "width": 3},
        {"kind": "line", "xy": [0.1, 0.1, 0.1, 0.9], "fill": "#000000", "width": 3},
        {"kind": "line", "xy": [0.12, 0.8, 0.3, 0.6, 0.45, 0.65, 0.6, 0.4, 0.75, 0.35, 0.9, 0.15], "fill": "#1f77b4",
         "width": 6}]}, width=W, height=W), "a line graph going up")
    d["christmas_tree"] = (MD.draw_scene({"background": "#0b2545", "shapes": [
        {"kind": "triangle", "xy": [0.2, 0.75, 0.5, 0.15, 0.8, 0.75], "fill": "#1b7f3b"},
        {"kind": "rect", "xy": [0.45, 0.75, 0.55, 0.9], "fill": "#6b3e1e"},
        *[{"kind": "circle", "xy": [x, y, x + 0.05, y + 0.05], "fill": c}
          for (x, y), c in zip([(0.4, 0.5), (0.55, 0.6), (0.47, 0.35), (0.33, 0.65), (0.6, 0.45)],
                               ["#ff0000", "#ffd700", "#ff0000", "#ffd700", "#00bfff"])],
        {"kind": "circle", "xy": [0.46, 0.1, 0.54, 0.18], "fill": "#ffd700"}]}, width=W, height=W),
        "a decorated christmas tree")
    d["warning"] = (MD.render_text_image("DANGER\nHIGH VOLTAGE\nKEEP OUT", width=W, height=W, font_size=56,
                                         color="#000000", background="#ffcc00", align="center", bold=True),
                    "a danger sign warning of high voltage")
    d["code_screen"] = (MD.render_text_image("def add(a, b):\n    return a + b\n\nprint(add(2, 3))", width=W, height=W,
                                             font_size=30, color="#a6e22e", background="#272822", family="mono"),
                        "a screenshot of python source code")
    d["menu"] = (MD.render_text_image("MENU\nTomato soup  4.50\nPasta with pesto  9.00\nApple pie  3.50",
                                      width=W, height=W, font_size=30, color="#3e2723", background="#fff8e1",
                                      align="center"), "a restaurant menu with prices")
    import matplotlib
    import sklearn.datasets as sd
    root_m = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data")
    root_s = os.path.join(os.path.dirname(sd.__file__), "images")
    for name, path, cap in (("grace_hopper", os.path.join(root_m, "grace_hopper.jpg"),
                             "a portrait photo of a woman in a navy uniform"),
                            ("china", os.path.join(root_s, "china.jpg"), "a chinese temple with a pagoda roof"),
                            ("flower", os.path.join(root_s, "flower.jpg"), "a close-up photo of a flower")):
        with open(path, "rb") as fh:
            d[name] = (fh.read(), cap)
    return d


def retrieval(q: np.ndarray, keys: np.ndarray) -> dict:
    """Query i should retrieve key i."""
    s = A._norm_rows(q) @ A._norm_rows(keys).T
    ranks = np.array([int(np.sum(s[i] > s[i, i])) for i in range(len(s))])
    return {"n": int(len(s)), "r1": round(float(np.mean(ranks == 0)), 4), "r5": round(float(np.mean(ranks < 5)), 4),
            "mrr": round(float(np.mean(1.0 / (ranks + 1))), 4),
            "mean_matched_cos": round(float(np.mean(np.diag(s))), 4),
            "mean_unmatched_cos": round(float((s.sum() - np.trace(s)) / max(1, s.size - len(s))), 4)}


def run(report: str | None = None, save: bool = True) -> dict:
    t0 = time.time()
    sp = splits()
    news = C.news()
    fit_texts = [C.news_text(news[n]) for n in sp["fit"]]
    fit_titles = [news[n][2] for n in sp["fit"]]
    texts = fit_texts + fit_titles
    mini = TEXT.embed_texts(texts)
    clip = A.clip_text(texts)
    maps = A.fit_ridge(clip, mini)
    # held-out text->text alignment (titles of DEV news)
    dev_ids = sp["dev"][:2000]
    dt = [news[n][2] for n in dev_ids]
    dm, dc = TEXT.embed_texts(dt), A.clip_text(dt)
    pred = A._norm_rows((dc - maps["mx"]) @ maps["W"] + maps["my"])
    text_align = {"heldout_titles": len(dt), "mean_cos_mapped_vs_minilm": round(float(np.mean(np.sum(pred * dm, 1))), 4),
                  "retrieval_mapped_clip_text_to_minilm": retrieval(pred[:1000], dm[:1000])}
    # images
    dev_imgs, dev_caps = rendered_titles(sp["dev"][:N_DEV], seed=1)
    test_imgs, test_caps = rendered_titles(sp["test"][:N_TEST], seed=2)
    from ..encoders.validation import _scenes
    sc = _scenes()
    dev_sc = [sc[k] for k in SCENE_CAPTIONS_DEV]
    dev_sc_caps = list(SCENE_CAPTIONS_DEV.values())
    ts = test_scenes()
    test_sc = [v[0] for v in ts.values()]
    test_sc_caps = [v[1] for v in ts.values()]
    sets = {"dev_titles": (dev_imgs, dev_caps), "dev_scenes": (dev_sc, dev_sc_caps),
            "test_titles": (test_imgs, test_caps), "test_scenes": (test_sc, test_sc_caps)}
    feats = {k: IMAGE.embed([*v[0]]) for k, v in sets.items()}
    # image mean for the modality-gap correction: DEV images only
    image_mean = np.concatenate([feats["dev_titles"], feats["dev_scenes"]]).mean(0)
    A._CACHE.pop("ridge", None)
    res = {"text_alignment": text_align, "fit_pairs": len(texts), "images": {}}
    options = {}
    for k, (imgs, caps) in sets.items():
        capv = TEXT.embed_texts(caps)
        capc = A.clip_text(caps)
        X = feats[k]
        gap = A._norm_rows(X - image_mean + maps["mx"])
        r_gap = A._norm_rows((gap - maps["mx"]) @ maps["W"] + maps["my"])
        r_raw = A._norm_rows((X - maps["mx"]) @ maps["W"] + maps["my"])
        conc, _ = A.concepts_apply(X)
        opts = {"clip_native_reference": retrieval(X, capc), "ridge_raw": retrieval(r_raw, capv),
                "ridge_gap": retrieval(r_gap, capv), "concepts": retrieval(conc, capv)}
        for wr in (0.25, 0.5, 0.75):
            opts[f"blend_ridge{wr}"] = retrieval(A._norm_rows(wr * r_gap + (1 - wr) * conc), capv)
        res["images"][k] = opts
        options[k] = {o: v["mrr"] for o, v in opts.items()}
    # choose on DEV (mean MRR over the two DEV sets; the CLIP reference is not a candidate)
    cands = [o for o in options["dev_titles"] if o != "clip_native_reference"]
    dev_score = {o: round((options["dev_titles"][o] + options["dev_scenes"][o]) / 2, 4) for o in cands}
    best = max(cands, key=lambda o: dev_score[o])
    res["dev_mean_mrr"] = dev_score
    res["chosen"] = best
    res["test_of_chosen"] = {k: res["images"][k][best] for k in ("test_titles", "test_scenes")}
    res["seconds"] = round(time.time() - t0, 1)
    if save:
        meta = {"fit_pairs": len(texts), "fit": "MIND train+dev news, ids bucket 0-79 (sha256), title+abstract and "
                "title", "lambda": A.RIDGE_LAMBDA, "image_mean": "DEV rendered titles + Gate E suite scenes",
                "chosen_on_dev": best}
        res["ridge_sha256"] = A.save_ridge(maps, image_mean, meta)
    if report:
        with open(report, "w") as fh:
            json.dump(res, fh, indent=1)
    return res
