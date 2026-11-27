"""The replaceable grounding interface: the non-neural reference grounder on CPU, the rules every model-backed
grounder follows (pinned, frozen, no remote code, not in a credential-holding process), and gpu-marked tests of
the integrated models."""
import io
import json
import os
import re

import pytest

from cam_live import contracts as c
from cam_live.harness import grounding as G
from cam_live.harness.grounding import base, goclick, maiui, vocaela
from cam_live.harness.grounding.reference import ReferenceGrounder, candidate_labels

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOXES = {"Export": (200, 300, 350, 344), "Archive": (400, 300, 550, 344), "Delete account": (900, 600, 1080, 644),
         "Search": (1000, 60, 1110, 100), "Settings": (80, 620, 200, 664)}


def ui_png(boxes=BOXES, circle=True):
    from PIL import Image, ImageDraw, ImageFont
    if not os.path.exists(FONT):
        pytest.skip("DejaVu Sans is not installed")
    font = ImageFont.truetype(FONT, 15)
    img = Image.new("RGB", (1280, 800), (244, 245, 247))
    d = ImageDraw.Draw(img)
    for label, (x1, y1, x2, y2) in boxes.items():
        d.rounded_rectangle((x1, y1, x2, y2), 5, fill=(42, 110, 220))
        d.text(((x1 + x2) // 2, (y1 + y2) // 2), label, fill="white", font=font, anchor="mm")
    if circle:
        d.ellipse((600, 100, 660, 160), fill=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def inside(r, box):
    return box[0] <= r["x"] <= box[2] and box[1] <= r["y"] <= box[3]


# ---------------------------------------------------------------------------------------------------------- CPU
def test_reference_grounder_finds_literal_labels_and_plain_colours():
    g = ReferenceGrounder()
    assert isinstance(g, c.VisualGrounder) and g.baseline and "baseline" in g.info()["note"]
    png = ui_png()
    for query, label in (('"Export" button', "Export"), ("the Archive button", "Archive"),
                         ("Delete account", "Delete account"), ('click the "Settings" link', "Settings")):
        r = g.ground(png, query)
        assert inside(r, BOXES[label]), (query, r)
        assert r["method"] == "text-template" and 0.68 <= r["confidence"] <= 1.0 and r["grounder"] == "reference-template"
        assert isinstance(r["x"], int) and isinstance(r["y"], int) and r["latency_s"] > 0
    red = g.ground(png, "the red circle")
    assert red["method"] == "colour-blob" and 600 <= red["x"] <= 660 and 100 <= red["y"] <= 160
    info = g.info()
    assert info["params"] == 0 and info["calls"] == 5 and info["mean_latency_s"] > 0 and info["available"]


def test_reference_grounder_fails_honestly_on_what_it_cannot_see():
    g = ReferenceGrounder()
    png = ui_png()
    for query in ('the "Unsubscribe" link', "the green tile", "the purple icon"):
        with pytest.raises(G.GroundingError):
            g.ground(png, query)
    with pytest.raises(G.GroundingError):
        g.ground(png, "   ")
    with pytest.raises(G.GroundingError):
        g.ground(b"not a png", "Export")
    assert candidate_labels('"Edit" button in the row of "invoice-1837"')[:2] == ["Edit", "invoice-1837"]
    assert candidate_labels("click the Save draft button") == ["Save draft", "draft", "Save"]


def test_registry_pins_revisions_and_states_licences():
    assert set(G.REGISTRY) == {"goclick-base", "mai-ui-2b", "vocaela-2-500m", "reference-template"}
    for name in ("goclick-base", "mai-ui-2b", "vocaela-2-500m"):
        meta = G.REGISTRY[name]
        assert re.fullmatch(r"[0-9a-f]{40}", meta["revision"]) and "/" in meta["repo"]
    assert G.REGISTRY["goclick-base"]["license"] == "MIT" and not G.REGISTRY["goclick-base"]["needs_gpu_lock"]
    assert G.REGISTRY["mai-ui-2b"]["license"] == "Apache-2.0" and G.REGISTRY["mai-ui-2b"]["needs_gpu_lock"]
    assert G.REGISTRY["vocaela-2-500m"]["noncommercial"] and "NC" in G.REGISTRY["vocaela-2-500m"]["license"]
    assert G.REGISTRY["reference-template"]["baseline"]
    assert G.DEFAULT_ORDER == ("goclick-base", "mai-ui-2b")              # never the non-commercial model or the baseline
    avail = G.available_grounders()
    assert avail["reference-template"]["available"] and set(avail) == set(G.REGISTRY)
    for name, cls in (("goclick-base", G.GoClickGrounder), ("mai-ui-2b", G.MaiUIGrounder), ("vocaela-2-500m", G.VocaelaGrounder)):
        info = cls().info()
        assert info["revision"] == G.REGISTRY[name]["revision"] and info["trust_remote_code"] is False
        assert info["frozen"] is True and info["loaded"] is False       # lazily loaded: nothing happened yet
    assert isinstance(G.load_grounder("reference-template"), ReferenceGrounder)
    assert isinstance(G.load_grounder("goclick-base"), G.IsolatedGrounder)   # the default path is a separate process
    with pytest.raises(KeyError):
        G.load_grounder("some-other-model")
    with pytest.raises(G.GrounderUnavailable):
        G.IsolatedGrounder("reference-template")


def test_models_are_not_loaded_in_a_credential_holding_process(monkeypatch):
    png = ui_png()
    g = G.GoClickGrounder()                                               # in-process loading was not requested
    with pytest.raises(G.GrounderLoadRefused, match="allow_inprocess"):
        g.ground(png, "Export button")
    monkeypatch.setenv("CAM_LIVE_HOLDS_CREDENTIALS", "1")
    assert "CAM_LIVE_HOLDS_CREDENTIALS" in base.credential_holder()
    for cls in (G.GoClickGrounder, G.MaiUIGrounder, G.VocaelaGrounder):
        with pytest.raises(G.GrounderLoadRefused, match="not loading model weights"):
            cls(allow_inprocess=True, accept_noncommercial=True).ground(png, "Export button")
    monkeypatch.delenv("CAM_LIVE_HOLDS_CREDENTIALS")
    assert base.credential_holder() == ""
    v = G.VocaelaGrounder(allow_inprocess=True)                           # licence not accepted
    ok, why = v.available()
    assert not ok and ("non" in why.lower() or "not in the local cache" in why)
    with pytest.raises(G.GrounderUnavailable):
        v.ground(png, "Export button")
    assert not g.loaded and not v.loaded


def test_missing_or_unpinned_snapshots_are_unavailable_not_downloaded(monkeypatch, tmp_path):
    with pytest.raises(G.GrounderUnavailable, match="not in the local cache"):
        base.snapshot_dir("HongxinLi/GoClick-Base", "0" * 40)             # another revision is not accepted
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    g = G.GoClickGrounder(allow_inprocess=True)
    assert g.available()[0] is False
    with pytest.raises(G.GrounderUnavailable):
        g.ground(ui_png(), "Export button")


def test_goclick_checkpoint_keys_map_onto_the_native_florence2_layout():
    import torch
    t = torch.zeros
    original = {
        "language_model.model.shared.weight": t(5, 4), "language_model.lm_head.weight": t(5, 4),
        "language_model.final_logits_bias": t(1, 5),
        "language_model.model.encoder.layers.0.self_attn.q_proj.weight": t(4, 4),
        "vision_tower.convs.1.proj.weight": t(8, 4, 3, 3), "vision_tower.convs.1.norm.weight": t(4),
        "vision_tower.blocks.2.5.spatial_block.conv1.fn.dw.weight": t(4, 1, 3, 3),
        "vision_tower.blocks.2.5.spatial_block.window_attn.fn.qkv.weight": t(12, 4),
        "vision_tower.blocks.2.5.spatial_block.window_attn.norm.bias": t(4),
        "vision_tower.blocks.2.5.spatial_block.ffn.fn.net.fc1.weight": t(16, 4),
        "vision_tower.blocks.2.5.spatial_block.ffn.norm.weight": t(4),
        "vision_tower.blocks.0.0.channel_block.channel_attn.fn.proj.bias": t(4),
        "vision_tower.blocks.0.0.channel_block.channel_attn.norm.weight": t(4),
        "vision_tower.blocks.0.0.channel_block.conv2.fn.dw.bias": t(4),
        "image_projection": torch.arange(6.0).reshape(3, 2), "image_proj_norm.weight": t(2),
        "image_pos_embed.row_embeddings.weight": t(50, 2), "visual_temporal_embed.pos_idx_to_embed": t(100, 3),
    }
    out = goclick.convert_state_dict(original)
    assert set(out) == {
        "model.language_model.shared.weight", "lm_head.weight",
        "model.language_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_tower.convs.1.conv.weight", "model.vision_tower.convs.1.norm.weight",
        "model.vision_tower.blocks.2.5.spatial_block.conv1.weight",
        "model.vision_tower.blocks.2.5.spatial_block.window_attn.qkv.weight",
        "model.vision_tower.blocks.2.5.spatial_block.norm1.bias",
        "model.vision_tower.blocks.2.5.spatial_block.ffn.fc1.weight",
        "model.vision_tower.blocks.2.5.spatial_block.norm2.weight",
        "model.vision_tower.blocks.0.0.channel_block.channel_attn.proj.bias",
        "model.vision_tower.blocks.0.0.channel_block.norm1.weight",
        "model.vision_tower.blocks.0.0.channel_block.conv2.bias",
        "model.multi_modal_projector.image_projection.weight", "model.multi_modal_projector.image_proj_norm.weight",
        "model.multi_modal_projector.image_position_embed.row_embeddings.weight",
        "model.multi_modal_projector.visual_temporal_embed.pos_idx_to_embed"}
    w = out["model.multi_modal_projector.image_projection.weight"]
    assert tuple(w.shape) == (2, 3) and w[1, 2].item() == 5.0             # x @ P  ==  Linear(weight = P^T)
    assert len(goclick.EXTRA_TOKENS) == 1024 and goclick.EXTRA_TOKENS[4] == "<loc_0>"


def test_model_answers_are_parsed_strictly():
    assert maiui.parse_answer('<grounding_think>the blue one</grounding_think>\n<answer>\n{"coordinate": [512, 300]}\n</answer>') == (512.0, 300.0)
    assert maiui.parse_answer('<answer>{"coordinate": [1, 2, 3]}</answer>') is None
    assert maiui.parse_answer("I think you should run rm -rf /") is None
    assert vocaela.parse_action('<Action>[{"action": "click", "coordinate": [0.25, 0.5]}]</Action>') == (0.25, 0.5)
    assert vocaela.parse_action('<Action>[{"action": "type", "text": "hello"}]</Action>') is None
    assert vocaela.parse_action('[{"action":"click","coordinate":[0.1,0.9]},{"action":"type","text":"x"}]') == (0.1, 0.9)
    r = base.point_result(10.4, 20.6, 100, 50, 1.7, "n", "r", raw="x" * 1000, action="type")
    assert (r["x"], r["y"], r["confidence"]) == (10, 21, 1.0) and len(r["raw"]) == 400
    for x, y in ((100, 10), (-1, 10), (float("nan"), 1)):
        with pytest.raises(G.GroundingError):
            base.point_result(x, y, 100, 50, 0.5, "n", "r")


# ---------------------------------------------------------------------------------------------------------- GPU
def _require(name):
    ok, why = {"goclick-base": G.GoClickGrounder, "mai-ui-2b": G.MaiUIGrounder,
               "vocaela-2-500m": G.VocaelaGrounder}[name](accept_noncommercial=True).available()
    if not ok:
        pytest.skip(why)
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _hits(g, png):
    out = {}
    for label, box in BOXES.items():
        try:
            r = g.ground(png, f"{label} button")
            out[label] = inside(r, box)
            assert 0.0 <= r["confidence"] <= 1.0 and r["latency_s"] > 0
        except G.GroundingError:
            out[label] = False
    return out


@pytest.mark.gpu
def test_goclick_base_grounds_labelled_buttons_in_an_isolated_process(monkeypatch):
    """The runtime path: the model lives in a child process whose environment carries none of ours."""
    _require("goclick-base")
    monkeypatch.setenv("CAM_FAKE_DISCORD_TOKEN", "tok-must-not-reach-the-model-process")
    g = G.load_grounder("goclick-base")
    try:
        assert isinstance(g, c.VisualGrounder) and isinstance(g, G.IsolatedGrounder)
        hits = _hits(g, ui_png())
        assert sum(hits.values()) >= 4, hits
        info = g.info()
        assert info["isolated"] and info["pid"] != os.getpid() and info["loaded"] and info["frozen"]
        assert info["revision"] == goclick.REVISION and info["trust_remote_code"] is False
        assert 200e6 < info["params"] < 300e6 and info["gpu_peak_mb"] < 4096       # small: no shared lock needed
        env = open(f"/proc/{info['pid']}/environ", "rb").read().split(b"\0")
        names = {e.split(b"=", 1)[0].decode() for e in env if e}
        assert b"tok-must-not-reach-the-model-process" not in b"\0".join(env)
        assert names <= {"PATH", "HOME", "LANG", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONPATH",
                         "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM", "CAM_LIVE_GROUNDER_WORKER",
                         "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES"}
        assert dict(e.split(b"=", 1) for e in env if e)[b"HF_HUB_OFFLINE"] == b"1"
        with pytest.raises(G.GroundingError):
            g.ground(b"\x89PNG broken", "Export button")
        assert g.ground(ui_png(), "Export button")["x"] > 0                         # the process survived the bad input
    finally:
        g.close()


@pytest.mark.gpu
def test_goclick_base_in_process_is_frozen_and_releases_memory():
    _require("goclick-base")
    import torch
    before = torch.cuda.memory_allocated()
    g = G.GoClickGrounder(allow_inprocess=True)
    assert not g.loaded
    r = g.ground(ui_png(), "Delete account button")
    assert inside(r, BOXES["Delete account"]) and g.loaded
    assert all(not p.requires_grad for p in g._model.parameters()) and not g._model.training
    assert g._model.lm_head.weight.data_ptr() != g._model.model.language_model.shared.weight.data_ptr()
    assert g.info()["gpu_mem_mb"] > 100
    g.close()
    assert not g.loaded and torch.cuda.memory_allocated() <= before + 2**20


@pytest.mark.gpu
def test_vocaela_runs_for_evaluation_only():
    _require("vocaela-2-500m")
    g = G.load_grounder("vocaela-2-500m", accept_noncommercial=True)
    try:
        hits = _hits(g, ui_png())
        info = g.info()
        assert info["noncommercial"] and info["license"] == "CC BY-NC-SA 4.0" and info["gpu_peak_mb"] < 4096
        assert sum(hits.values()) >= 3, hits
    finally:
        g.close()
    refused = G.load_grounder("vocaela-2-500m")                            # without accepting the licence terms
    try:
        with pytest.raises(G.GrounderUnavailable):
            refused.ground(ui_png(), "Export button")
    finally:
        refused.close()


@pytest.mark.gpu
def test_mai_ui_2b_grounds_under_the_shared_gpu_lock(gpu_heavy):
    _require("mai-ui-2b")
    g = G.load_grounder("mai-ui-2b")
    try:
        assert g.heavy
        hits = _hits(g, ui_png())
        info = g.info()
        assert info["revision"] == maiui.REVISION and info["license"] == "Apache-2.0" and info["needs_gpu_lock"]
        assert sum(hits.values()) >= 4, hits
        assert info["gpu_peak_mb"] < 24 * 1024
    finally:
        g.close()                                                          # the child exits: its memory is freed
