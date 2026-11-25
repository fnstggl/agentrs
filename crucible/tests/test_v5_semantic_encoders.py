"""Content encoders (crucible.semantic.encoders): the contract API, the embedding cache, recording in worlds (replay,
regeneration, snapshots), agreement with the reference models, CPU execution, the kernel module's admission, and the
audits (quality fields, stub vectors) and statistics used by Gate E.

Run through the slot scheduler (models load on CPU or the slot's GPU):
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 3000 --label "ws-D1: encoder tests" -- \
        python -m pytest -q tests/test_v5_semantic_encoders.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from crucible.core import schema as S
from crucible.core.kernel import MODULES, Module, World, WorldSpec, endpoint, register_module
from crucible.semantic import encoders as E, media as M
from crucible.semantic.cache import EmbeddingCache
from crucible.semantic.encoders import audit as A, effects as EF, models as MD, module as _embed_module  # noqa: F401
from crucible.services import admin as _admin, bank as _bank, identity as _identity  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


@pytest.fixture(autouse=True, scope="module")
def private_cache(tmp_path_factory):
    """A fresh $CRUCIBLE_CACHE for this module, so nothing comes from an earlier run's disk cache."""
    d = tmp_path_factory.mktemp("cache")
    old = os.environ.get("CRUCIBLE_CACHE")
    os.environ["CRUCIBLE_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)
    else:
        os.environ["CRUCIBLE_CACHE"] = old


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================================================== contract API
def test_cache_key_has_the_contract_shape_and_changes_with_content():
    r1, r2 = E.ref_of(b"one"), E.ref_of(b"two")
    k1 = E.cache_key(r1, "text")
    enc = E.encoder("text")
    assert k1 == f"{r1}|{enc.id}@{enc.version()}|{enc.preprocessing_hash()}"
    assert E.cache_key(r2, "text") != k1 and E.cache_key(r1, "image") != k1
    with pytest.raises(E.EncodingError):
        E.cache_key("sha256:xyz", "text")
    with pytest.raises(E.EncodingError):
        E.cache_key(r1, "smell")


def test_encoder_info_reports_real_models():
    for m in E.MODALITIES:
        info = E.encoder_info(m)
        assert {"id", "version", "dim", "params", "preprocessing_hash"} <= set(info)
        assert info["params"] >= 1_000_000, m
        assert info["dim"] == E.encoder(m).dim
        for role, md in info["models"].items():
            assert len(md["weights_sha256"]) == 64 and md["revision"] == MD.SPECS[role].revision


def test_vectors_have_the_declared_shape_and_are_deterministic():
    data = b"A quiet street market sells fresh bread every Saturday morning."
    v1 = E.encode_bytes(data, "text")
    assert v1.dtype == np.float32 and v1.shape == (384,) and abs(np.linalg.norm(v1) - 1) < 1e-4
    c = EmbeddingCache(disk=False)
    v2 = E.encode_bytes(data, "text", cache=c)                 # recomputed by the model, not cached
    assert np.allclose(v1, v2, atol=1e-5)
    with pytest.raises(ValueError):
        v1[0] = 1.0                                            # vectors handed out are read-only


def test_text_encoder_agrees_with_an_independent_reference_computation():
    """MiniLM mean pooling computed directly with transformers (separate model instance, standard tokenisation)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = MD.snapshot_dir("text")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    texts = ["The striker scored twice in the second half.", "Bake the sponge cake for 25 minutes."]
    with torch.inference_mode():
        b = tok(texts, padding=True, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        ref = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()
    ours = E.encode_bytes_many([t.encode() for t in texts], "text", cache=EmbeddingCache(disk=False))
    for a, b_ in zip(ours, ref):
        assert cos(a, b_) > 0.9999


def test_html_is_read_as_its_visible_text():
    plain = b"Flood warnings have been issued for the coast."
    html = b"<!doctype html><html><body><p>Flood warnings have been <b>issued</b> for the coast.</p>" \
           b"<script>var x = 'ignore me';</script></body></html>"
    assert cos(E.encode_bytes(plain, "text"), E.encode_bytes(html, "text")) > 0.999


def test_long_text_uses_every_window():
    """A document longer than one 256-token window is not truncated: changing only its end changes the vector."""
    para = " ".join(f"Sentence number {i} talks about the harbour, the boats and the fish market." for i in range(60))
    a = E.encode_bytes(para.encode(), "text")
    b = E.encode_bytes((para + " The final section is about a volcano erupting on the island.").encode(), "text")
    assert cos(a, b) < 0.9999


def test_audio_analysis_hears_speech_and_music():
    sr = 16000
    speech = M.decode_audio(M.synth_speech("Please turn left at the next junction.", voice="slt"), sr)
    music = M.synth_melody(["C4", "E4", "G4", "C5"] * 3, bpm=140, sr=sr, drums=True)
    an = E.encoder("audio").analyze([speech, music])
    assert an[0]["speech"] > 0.5 and "left" in an[0]["transcript"].lower()
    assert an[1]["speech"] < 0.5 and an[1]["transcript"] == ""
    assert an[1]["labels"][0][0] in ("Music", "Musical instrument", "Ringtone", "Video game music", "Electronic music")


def test_code_inputs_in_every_packaging_read_the_same_tree():
    import io
    import zipfile
    files = {"pkg/util.py": "def slugify(title):\n    \"\"\"Make a URL slug.\"\"\"\n    return title.lower()\n"}
    js = json.dumps({"files": files}).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg/util.py", files["pkg/util.py"])
    from crucible.semantic.encoders import code as C
    assert C.read_tree(js) == C.read_tree(buf.getvalue()) == files
    assert list(C.read_tree(files["pkg/util.py"].encode())) == ["main.py"]
    v_js, v_zip = E.encode_bytes(js, "code"), E.encode_bytes(buf.getvalue(), "code")
    assert np.allclose(v_js, v_zip, atol=1e-6)


def test_unreadable_artifacts_raise_encoding_errors():
    with pytest.raises(E.EncodingError):
        E.encode_bytes(b"definitely not an image", "image")
    with pytest.raises(E.EncodingError):
        E.encode_bytes(b"not audio", "audio")
    png = M.render_text_image("still image", width=64, height=64)
    with pytest.raises(E.EncodingError):
        E.encode_bytes(png, "video")


def test_cpu_execution_agrees_with_this_process():
    """The encoders run on CPU (a separate process forced to CPU) and agree with this process's device."""
    texts = [b"Shares in the bank fell after a loss.", b"A red apple on a white table."]
    img = M.draw_scene({"background": "#ffffff", "shapes": [{"kind": "circle", "xy": [0.2, 0.2, 0.8, 0.8],
                                                             "fill": "#cc0000"}]}, width=256, height=256)
    here_t = E.encode_bytes_many(texts, "text", cache=EmbeddingCache(disk=False))
    here_i = E.encode_bytes(img, "image", cache=EmbeddingCache(disk=False))
    prog = textwrap.dedent("""
        import base64, json, sys
        from crucible.semantic import encoders as E
        from crucible.semantic.cache import EmbeddingCache
        d = json.loads(sys.stdin.read())
        t = E.encode_bytes_many([base64.b64decode(x) for x in d["t"]], "text", cache=EmbeddingCache(disk=False))
        i = E.encode_bytes(base64.b64decode(d["i"]), "image", cache=EmbeddingCache(disk=False))
        print(json.dumps({"t": t.tolist(), "i": i.tolist(), "dev": E.models.load("text").device}))
    """)
    env = dict(os.environ, CRUCIBLE_ENCODER_DEVICE="cpu", CRUCIBLE_EMBED_DISK_CACHE="0", PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    inp = json.dumps({"t": [base64.b64encode(t).decode() for t in texts], "i": base64.b64encode(img).decode()})
    out = subprocess.run([sys.executable, "-c", prog], input=inp.encode(), capture_output=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr.decode()[-2000:]
    res = json.loads(out.stdout.decode().strip().splitlines()[-1])
    assert res["dev"] == "cpu"
    for a, b in zip(here_t, res["t"]):
        assert cos(a, np.array(b)) > 0.9995
    assert cos(here_i, np.array(res["i"])) > 0.9995


# ============================================================================================== cache
def test_cache_hits_misses_and_verification(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    key = E.cache_key(E.ref_of(b"x"), "text")
    assert c.get(key) is None and c.stats["misses"] == 1
    v = np.arange(384, dtype=np.float32) / 384
    c.put(key, v)
    assert np.array_equal(c.get(key), v) and c.stats["memory_hits"] == 1
    c2 = EmbeddingCache(str(tmp_path))                          # another process / world on the same machine
    assert np.array_equal(c2.get(key), v) and c2.stats["disk_hits"] == 1
    # a tampered file is detected, removed and treated as a miss
    p = c.path(key)
    raw = bytearray(open(p, "rb").read())
    raw[-3] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    c3 = EmbeddingCache(str(tmp_path))
    assert c3.get(key) is None and c3.stats["corrupt"] == 1 and not os.path.exists(p)
    # an entry copied under another key (a stale vector for changed content) is refused
    other = E.cache_key(E.ref_of(b"y"), "text")
    c.put(key, v)
    os.makedirs(os.path.dirname(c.path(other)), exist_ok=True)
    open(c.path(other), "wb").write(open(c.path(key), "rb").read())
    c4 = EmbeddingCache(str(tmp_path))
    assert c4.get(other) is None and c4.stats["corrupt"] == 1
    with pytest.raises(ValueError):
        c.put(key, np.array([np.nan], dtype=np.float32))


# ============================================================================================== worlds
@register_module
class _EncodeProbe(Module):
    """Test harness module: an endpoint that reads a text through the world's encoder (as a participant would)."""
    name = "_d1probe"
    host = "d1probe"

    @endpoint("read", S.obj({"text": S.s_str(maxLength=2000), "modality": S.s_enum(["text", "code"])}, ["text"]),
              doc="Encode a text the caller provides and return the first coordinates of its vector.")
    def ep_read(self, ctx, text, modality="text"):
        data = text.encode()
        v = E.encode(ctx.world, E.ref_of(data), modality, ctx=ctx, data=data)
        return {"head": [round(float(x), 6) for x in v[:3]]}


def _world():
    spec = WorldSpec("d1", 7, [{"name": n} for n in ("identity", "bank", "admin", "embed", "_d1probe")], {})
    w = World.create(spec)
    p = w.system_call("admin.join", {"handle": "reader", "endowment": 0})["principal"]
    return w, p


def _log(w):
    import copy
    return copy.deepcopy([e.to_dict() for e in w.events_range(0)])


def test_world_records_each_embedding_once_and_replays_and_regenerates_exactly():
    w, p = _world()
    texts = ["The ferry to the island leaves every hour.", "Our bakery opens at seven.",
             "The ferry to the island leaves every hour."]
    heads = [w.execute(p, {"tool": "d1probe.read", "args": {"text": t}}) for t in texts]
    assert all(r.ok for r in heads)
    assert heads[0].result == heads[2].result
    rec = [e for e in w.events_range(0) if e.type == "embed.recorded"]
    assert len(rec) == 2                                       # the repeated text was read from the world record
    assert w.state.count("embeddings") == 2 and w.check_invariants() == []
    ext = [e for e in w.events_range(0) if e.type == "ext"]
    assert len(ext) == 2 and all(o["kind"].startswith("infer.embed.") for e in ext for o in e.payload["outcomes"])
    evs = _log(w)
    r = World.replay(w.spec, evs)
    assert (r.digest(), r.last_hash) == (w.digest(), w.last_hash)
    # regeneration re-executes the commands; the recorded vectors are used, so no model runs and the log is identical
    computed = E.stats()["vectors_computed"]["text"]
    from crucible.semantic import cache as CA
    CA.default_cache().clear_memory()
    old = os.environ.get("CRUCIBLE_EMBED_DISK_CACHE")
    os.environ["CRUCIBLE_EMBED_DISK_CACHE"] = "0"
    os.environ["CRUCIBLE_CACHE"] = os.environ["CRUCIBLE_CACHE"] + "-regen"   # a new, empty cache
    try:
        g = World.regenerate(w.spec, evs)
    finally:
        os.environ["CRUCIBLE_CACHE"] = os.environ["CRUCIBLE_CACHE"][:-len("-regen")]
        if old is None:
            os.environ.pop("CRUCIBLE_EMBED_DISK_CACHE", None)
        else:
            os.environ["CRUCIBLE_EMBED_DISK_CACHE"] = old
    assert (g.last_hash, g.digest()) == (w.last_hash, w.digest())
    assert E.stats()["vectors_computed"]["text"] == computed
    # snapshots and forks carry the vectors
    s = World.from_blob(w.snapshot_blob())
    key = E.cache_key(E.ref_of(texts[1].encode()), "text")
    assert np.array_equal(E.encode(s, E.ref_of(texts[1].encode()), "text"), E.encode(w, E.ref_of(texts[1].encode()),
                                                                                     "text"))
    assert s.state.get("embeddings", key)["sha256"] == w.state.get("embeddings", key)["sha256"]


def test_world_vector_wins_over_a_different_cache_entry():
    """Once recorded in a world, a vector is fixed for that world even if the machine's cache later holds another
    value for the key (a different device or library version)."""
    w, p = _world()
    text = "Wind turbines were installed on the hill last year."
    assert w.execute(p, {"tool": "d1probe.read", "args": {"text": text}}).ok
    ref = E.ref_of(text.encode())
    key = E.cache_key(ref, "text")
    recorded = np.array(E.encode(w, ref, "text"))
    from crucible.semantic import cache as CA
    CA.default_cache().put(key, -recorded)
    assert np.array_equal(E.encode(w, ref, "text"), recorded)
    assert np.array_equal(E.encode(None, ref, "text"), -recorded)    # outside the world: the cache's value


def test_embed_module_passes_admission():
    from crucible.worldgen.admission import admit
    rep = admit(["embed"], steps=120)
    assert rep.passed, (rep.checks, rep.errors)


def test_embed_encoders_tool_lists_every_modality():
    w, p = _world()
    r = w.execute(p, {"tool": "embed.encoders", "args": {}})
    assert r.ok and set(r.result["encoders"]) == set(E.MODALITIES)
    assert r.result["encoders"]["video"]["dim"] == 3208


# ============================================================================================== audits and statistics
def test_quality_audit_finds_planted_reads_and_fields(tmp_path):
    root = tmp_path / "crucible"
    (root / "social").mkdir(parents=True)
    (root / "mechanisms").mkdir()
    (root / "social" / "feed.py").write_text(textwrap.dedent("""
        def rank(items):
            return sorted(items, key=lambda it: it["quality"] + it.get("virality_score", 0))

        def publish(ctx, title, creator_quality=1.0):
            return {"title": title}
    """))
    (root / "mechanisms" / "firm.py").write_text("def step(f):\n    return f['quality'] * 2\n")
    rep = A.quality_fields(str(root), include_endpoints=False)
    fields = sorted(v["field"] for v in rep["violations"])
    assert fields == ["quality", "virality_score"]
    assert [a["field"] for a in rep["creator_assignable_fields"]] == ["creator_quality"]
    assert rep["out_of_scope_files"] == ["mechanisms/firm.py"]


def test_quality_audit_finds_creator_settable_endpoint_fields():
    @register_module
    class _Planted(Module):
        name = "_planted_quality"
        host = "planted"

        @endpoint("post", S.obj({"title": S.s_str(), "virality": S.s_num()}), doc="planted")
        def ep_post(self, ctx, title="", virality=0.0):
            return {}
    try:
        found = A.endpoint_fields()
        assert {"module": "_planted_quality", "endpoint": "planted.post", "field": "virality",
                "path": "virality"} in found
    finally:
        MODULES.pop("_planted_quality", None)


def test_stub_check_flags_fake_vectors():
    good = {"a": np.array([0.1, -0.2, 0.3], dtype=np.float32), "b": np.array([0.3, 0.1, -0.2], dtype=np.float32)}
    assert A.stub_check(good, {"a": "1", "b": "2"})["stub_vectors_detected"] == 0
    bad = {"zero": np.zeros(3), "const": np.full(3, 0.5), "nan": np.array([np.nan, 1, 2]),
           "x": np.array([1.0, 2.0, 3.0]), "y": np.array([1.0, 2.0, 3.0])}
    r = A.stub_check(bad, {"zero": "1", "const": "2", "nan": "3", "x": "4", "y": "5"})
    assert r["stub_vectors_detected"] == 4


def test_paired_permutation_test_detects_real_effects_only():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 3, 400)
    assert EF.paired_permutation_pvalue((base + rng.normal(0.8, 1, 400)) - base, n_perm=4000) < 0.001
    ps = [EF.paired_permutation_pvalue(rng.normal(0, 1, 200), n_perm=999, seed=s) for s in range(40)]
    assert 0.2 < float(np.mean(ps)) < 0.8                      # roughly uniform under the null
    assert EF.paired_permutation_pvalue(np.zeros(50)) == 1.0
    assert EF.two_sample_permutation_pvalue(rng.normal(0, 1, 100), rng.normal(1, 1, 100), n_perm=2000) < 0.001
