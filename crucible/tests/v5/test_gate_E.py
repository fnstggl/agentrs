"""Gate E - Multimodal content understanding (MASTER_SPEC Part 17; workstream D1).

One test per requirement; each measures the requirement on the real encoders and records evidence:

* E.1  pretrained encoders process text, image, audio and video (and code): parameter counts, model forward passes
       during this run, agreement with independent reference computations, no stub vectors.
* E.2  separation: controlled artifact suites (rendered text and HTML, drawn scenes, flite speech, tones, melodies,
       noise, FFmpeg-assembled videos, source trees) with nuisance variants (re-encoding, resizing, noise, trimming,
       paraphrase, reformatting) and semantic variants; ratio of mean semantic to mean nuisance cosine distance.
* E.3  controlled artifact changes and participant behaviour: runs every registered participant probe
       (`crucible.semantic.encoders.effects`); fails, with the measurement recorded, while no probe exists.
* E.4  cache invalidation: hits, disk persistence, recomputation after a content change, new keys for a new encoder
       version or preprocessing, tamper detection, and per-world records.
* E.5  static audit: no creator-assignable quality/relevance/virality field and no read of one by participants.
* E.n1 real model execution: forward passes counted on the models, and outputs that change when the weights change.

Run (from crucible/):
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 3000 --label "ws-D1: gate E" -- \
        python -m pytest -q tests/v5/test_gate_E.py
CPU-only also works (--gpu-mb 0), more slowly.
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.core.kernel import World, WorldSpec
from crucible.semantic import encoders as E, media as M
from crucible.semantic.cache import EmbeddingCache
from crucible.semantic.encoders import audit as A, effects as EF, models as MD, validation as V
from crucible.services import admin as _admin, bank as _bank, identity as _identity  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
CMD = ('PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 3000 --label "gate E" -- '
       'python -m pytest -q tests/v5/test_gate_E.py')
REQUIRED = ("text", "image", "audio", "video")


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def fresh_cache(tmp_path_factory):
    """A new, empty $CRUCIBLE_CACHE: every vector in this run is computed by the models."""
    d = tmp_path_factory.mktemp("gateE_cache")
    old = os.environ.get("CRUCIBLE_CACHE")
    os.environ["CRUCIBLE_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)
    else:
        os.environ["CRUCIBLE_CACHE"] = old


@pytest.fixture(scope="module")
def encoded(fresh_cache):
    out = {}
    for m in E.MODALITIES:
        suite = V.SUITES[m]()
        names = list(suite.items)
        before = E.stats()["vectors_computed"][m]
        t0 = time.time()
        mat = E.encode_bytes_many([suite.items[n] for n in names], m)
        out[m] = {"suite": suite, "vectors": dict(zip(names, mat)), "seconds": time.time() - t0,
                  "computed": E.stats()["vectors_computed"][m] - before}
    return out


# ============================================================================================== reference checks
def _ref_text(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = MD.snapshot_dir("text")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    with torch.inference_mode():
        b = tok(texts, padding=True, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()


def _ref_image(images: list[bytes]) -> np.ndarray:
    """CLIP features with the stock CLIPProcessor (resize + centre crop; the suite images are square)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    path = MD.snapshot_dir("image")
    proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
    model = CLIPModel.from_pretrained(path, local_files_only=True).eval()
    with torch.inference_mode():
        px = proc(images=[M.open_image(b) for b in images], return_tensors="pt")["pixel_values"]
        pooled = model.vision_model(pixel_values=px).pooler_output
        return torch.nn.functional.normalize(model.visual_projection(pooled), dim=-1).numpy()


def _ref_ast(waves: list[np.ndarray]) -> np.ndarray:
    import warnings

    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from crucible.semantic.encoders.audio import normalise_loudness
    path = MD.snapshot_dir("audio")
    fe = ASTFeatureExtractor.from_pretrained(path, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(path, local_files_only=True).eval()
    with warnings.catch_warnings(), torch.inference_mode():
        warnings.simplefilter("ignore")
        iv = fe([normalise_loudness(w) for w in waves], sampling_rate=16000, return_tensors="pt")["input_values"]
        out = model.audio_spectrogram_transformer(input_values=iv).pooler_output
        return torch.nn.functional.normalize(out, dim=-1).numpy()


# ============================================================================================== E.1
def test_E1_pretrained_encoders_process_text_image_audio_and_video(encoded):
    per, problems = {}, []
    st = E.stats()
    fwd = st["forward_passes"]
    for m in E.MODALITIES:
        info = E.encoder_info(m)
        enc = E.encoder(m)
        stub = A.stub_check(encoded[m]["vectors"], {n: n.split("~")[0] for n in encoded[m]["vectors"]})
        ok = (info["params"] >= 1_000_000 and encoded[m]["computed"] == len(encoded[m]["vectors"])
              and all(fwd.get(r, 0) > 0 for r in enc.roles) and stub["stub_vectors_detected"] == 0)
        per[m] = {"params": info["params"], "computed": encoded[m]["computed"], "stub": stub["stub_vectors_detected"],
                  "ok": ok, "seconds": round(encoded[m]["seconds"], 1)}
    # agreement with independent computations of the same pretrained models
    tv = encoded["text"]["vectors"]
    ref_t = _ref_text([V.TEXTS["transport"][0], V.TEXTS["cooking"][0]])
    agree_text = min(cos(tv["transport"], ref_t[0]), cos(tv["cooking"], ref_t[1]))
    iv = encoded["image"]["vectors"]
    imgs = [encoded["image"]["suite"].items[n] for n in ("day_landscape", "red_apple", "sale_poster")]
    ref_i = _ref_image(imgs)
    agree_image = min(cos(iv[n], r) for n, r in zip(("day_landscape", "red_apple", "sale_poster"), ref_i))
    av = encoded["audio"]["vectors"]
    waves = [M.decode_audio(encoded["audio"]["suite"].items[n], 16000) for n in ("speech_weather", "tone_440")]
    ref_a = _ref_ast(waves)
    agree_audio = min(cos(av[n][:768], r) for n, r in zip(("speech_weather", "tone_440"), ref_a))
    an = E.encoder("audio").analyze(waves)
    transcript_ok = "sunny" in an[0]["transcript"].lower() and an[1]["transcript"] == ""
    # a still-image video: its visual segments are the CLIP vector of that image (cross-modal consistency)
    still = encoded["image"]["suite"].items["red_apple"]
    vid = M.assemble_video([still], None, seconds_per_image=2.0, size=(448, 448), crf=18)
    vv = E.encode_bytes(vid, "video")
    seg = vv[:512] / np.linalg.norm(vv[:512])
    agree_video = cos(seg, iv["red_apple"])
    checks = {"text": agree_text > 0.999, "image": agree_image > 0.99, "audio": agree_audio > 0.999 and transcript_ok,
              "video": agree_video > 0.95, "code": True}
    for m in E.MODALITIES:
        per[m]["reference_ok"] = checks[m]
        if not (per[m]["ok"] and checks[m]):
            problems.append(m)
    real = sum(1 for m in REQUIRED if per[m]["ok"] and per[m]["reference_ok"])
    metrics = {"real_encoder_modalities": real,
               "real_encoder_modalities_incl_code": real + int(per["code"]["ok"]),
               "stub_vectors_detected": int(sum(per[m]["stub"] for m in E.MODALITIES)),
               "min_encoder_params": int(min(per[m]["params"] for m in REQUIRED)),
               "reference_cosine_text": round(agree_text, 6), "reference_cosine_image": round(agree_image, 6),
               "reference_cosine_audio_ast": round(agree_audio, 6), "video_still_vs_image_cosine": round(agree_video, 6),
               "whisper_transcript_ok": transcript_ok, "device": MD.device(),
               "vectors_encoded": int(sum(per[m]["computed"] for m in E.MODALITIES))}
    for m in E.MODALITIES:
        metrics[f"params_{m}"] = int(per[m]["params"])
        metrics[f"dim_{m}"] = int(E.encoder(m).dim)
        metrics[f"seconds_{m}"] = per[m]["seconds"]
    record("E.1", metrics, command=CMD, notes=(
        "Encoders: " + "; ".join(f"{m}={E.encoder(m).id}@{E.encoder(m).version()}" for m in E.MODALITIES) +
        ". Every vector in this run was computed by the models (fresh cache). Reference checks recompute with "
        "separately loaded transformers models (text mean pooling, stock CLIPProcessor, AST pooled output) and a "
        "still-image video against the image vector."))
    assert not problems, per
    assert metrics["real_encoder_modalities"] >= 4 and metrics["stub_vectors_detected"] == 0
    assert metrics["min_encoder_params"] >= 1_000_000


# ============================================================================================== E.2
def test_E2_semantic_changes_separate_more_than_nuisance_changes(encoded):
    res = {m: V.measure(encoded[m]["suite"], encoded[m]["vectors"]) for m in E.MODALITIES}
    metrics = {"min_modality_separation_ratio": round(min(r["separation_ratio"] for r in res.values()), 4),
               "min_required_modality_separation_ratio": round(min(res[m]["separation_ratio"] for m in REQUIRED), 4)}
    for m, r in res.items():
        metrics[f"separation_ratio_{m}"] = r["separation_ratio"]
        metrics[f"auc_{m}"] = r["auc_semantic_farther"]
        metrics[f"mean_nuisance_distance_{m}"] = r["mean_nuisance_distance"]
        metrics[f"mean_semantic_distance_{m}"] = r["mean_semantic_distance"]
        metrics[f"pairs_{m}"] = f"{r['n_semantic_pairs']} semantic / {r['n_nuisance_pairs']} nuisance"
        metrics[f"weakest_semantic_group_ratio_{m}"] = min(r["group_ratios"].values())
    notes = "; ".join(f"{m}: nuisance {r['nuisance_by_kind']}, semantic {r['semantic_by_group']}"
                      for m, r in res.items())
    record("E.2", metrics, command=CMD, notes="Separation ratio = mean cosine distance of semantic pairs / mean "
           "cosine distance of nuisance pairs, per modality; groups: " + notes)
    assert metrics["min_modality_separation_ratio"] >= 1.5, res


# ============================================================================================== E.3
def test_E3_controlled_artifact_changes_affect_participant_behaviour():
    probes = EF.available()
    results = []
    for name in probes:
        pr = EF.PROBES[name]
        pairs = EF.controlled_pairs(pr.modality)[:3]
        for label, base, changed in pairs:
            r = EF.measure_probe(name, base, changed)
            r["pair"] = label
            results.append(r)
    if not results:
        record("E.3", {"artifact_change_behavior_pvalue": 1.0, "probes_available": 0, "pairs_measured": 0},
               command=CMD, notes="No participant probe is registered: the population and social internet "
               "(workstreams B and C) have not registered a world in which participants consume content "
               "(crucible.semantic.encoders.effects.register_probe). Nothing was measured; p is reported as 1.0.")
        pytest.fail("E.3 cannot be measured yet: no participant probe is registered "
                    "(crucible.semantic.encoders.effects.register_probe)")
    p_min = min(r["artifact_change_behavior_pvalue"] for r in results)
    p_bonf = min(1.0, p_min * len(results))
    best = min(results, key=lambda r: r["artifact_change_behavior_pvalue"])
    record("E.3", {"artifact_change_behavior_pvalue": p_bonf, "raw_min_pvalue": p_min, "probes_available": len(probes),
                   "pairs_measured": len(results), "best_probe": best["probe"], "best_pair": best["pair"],
                   "best_effect": best["effect"], "best_individuals": best["n_individuals"],
                   "best_embedding_distance": round(best["embedding_distance"], 5)},
           command=CMD, notes="Paired sign-flip permutation test per (probe, artifact pair); the reported p-value is "
           "Bonferroni-adjusted over all pairs measured.")
    assert p_bonf <= 0.01, results


# ============================================================================================== E.4
def _edit(m: str, suite: V.Suite) -> tuple[bytes, bytes]:
    """(original, edited content) for one modality."""
    if m == "text":
        base = suite.items["transport"]
        return base, base + b" The first buses will run on the new routes in April."
    if m == "image":
        base = suite.items["red_apple"]
        return base, M.process_image(base, [{"op": "text", "text": "ORGANIC", "xy": [0.3, 0.88], "size": 36}])
    if m == "audio":
        x = M.decode_audio(suite.items["melody_drums"], 16000)
        return suite.items["melody_drums"], M.wav_bytes(np.concatenate([x[:len(x) // 2],
                                                                         M.synth_tone(660.0, len(x) / 32000)]), 16000)
    if m == "video":
        return suite.items["news_report"], suite.items["news_report!music_audio"]
    files = dict(V.CODE_BASES["linear_algebra"])
    files["matrix.py"] += '\n\ndef trace(m):\n    """Sum of the diagonal."""\n    return sum(m[i][i] for i in range(len(m)))\n'
    return suite.items["linear_algebra"], V._files_json(files)


def test_E4_embedding_caches_invalidate_when_content_changes(encoded, tmp_path):
    failures, hits, recomputes, log = [], 0, 0, {}
    for m in E.MODALITIES:
        root = str(tmp_path / m)
        c = EmbeddingCache(root)
        orig, edited = _edit(m, encoded[m]["suite"])
        n0 = E.stats()["vectors_computed"][m]
        v0 = E.encode_bytes(orig, m, cache=c)
        if E.stats()["vectors_computed"][m] != n0 + 1:
            failures.append(f"{m}: first encoding did not run the model")
        v1 = E.encode_bytes(orig, m, cache=c)                          # memory hit
        c2 = EmbeddingCache(root)
        v2 = E.encode_bytes(orig, m, cache=c2)                         # disk hit in a new cache (another process)
        if E.stats()["vectors_computed"][m] != n0 + 1 or not (np.array_equal(v0, v1) and np.array_equal(v0, v2)):
            failures.append(f"{m}: unchanged content was recomputed or changed")
        hits += c.stats["memory_hits"] + c2.stats["disk_hits"]
        v3 = E.encode_bytes(edited, m, cache=c2)                       # changed content: new key, recomputed
        if E.stats()["vectors_computed"][m] != n0 + 2:
            failures.append(f"{m}: changed content was not recomputed")
        else:
            recomputes += 1
        if np.array_equal(v3, v0) or E.cache_key(E.ref_of(edited), m) == E.cache_key(E.ref_of(orig), m):
            failures.append(f"{m}: changed content returned the old representation")
        if not np.array_equal(E.encode_bytes(orig, m, cache=c2), v0):
            failures.append(f"{m}: the original's entry was disturbed by the change")
        # a new encoder version or different preprocessing is a new representation: new key, cache miss
        enc = E.encoder(m)
        ref = E.ref_of(orig)
        for variant in ("version", "preprocessing"):
            e2 = copy.copy(enc)
            if variant == "preprocessing":
                e2.preprocessing = {**enc.preprocessing, "changed_for_test": True}
                k2 = f"{ref}|{e2.id}@{e2.version()}|{e2.preprocessing_hash()}"
            else:
                k2 = f"{ref}|{e2.id}@{e2.version()}.next|{e2.preprocessing_hash()}"
            if k2 == E.cache_key(ref, m) or c2.get(k2) is not None:
                failures.append(f"{m}: a new {variant} did not invalidate the cached vector")
        # a tampered disk entry is detected and the vector recomputed
        p = c.path(E.cache_key(ref, m))
        raw = bytearray(open(p, "rb").read())
        raw[-1] ^= 0x55
        open(p, "wb").write(bytes(raw))
        c3 = EmbeddingCache(root)
        v4 = E.encode_bytes(orig, m, cache=c3)
        if c3.stats["corrupt"] != 1 or E.stats()["vectors_computed"][m] != n0 + 3 or cos(v4, v0) < 0.9999:
            failures.append(f"{m}: tampered entry not detected/recomputed")
        log[m] = {"edited_cosine": round(cos(v0, v3), 4)}
    # per-world records: an updated artifact gets its own record; the old one is kept unchanged
    spec = WorldSpec("gateE4", 1, [{"name": n} for n in ("identity", "bank", "admin", "embed")], {})
    w = World.create(spec)
    a, b = b"Version one of the article.", b"Version two of the article, rewritten."
    out = {}

    def run(ctx):
        out["a"] = np.array(E.encode(ctx.world, E.ref_of(a), "text", ctx=ctx, data=a))
        out["b"] = np.array(E.encode(ctx.world, E.ref_of(b), "text", ctx=ctx, data=b))
        out["a2"] = np.array(E.encode(ctx.world, E.ref_of(a), "text", ctx=ctx, data=a))
    w._run_system("gateE4.encode", run)
    recs = [it for e in w.events_range(0) if e.type == "embed.recorded" for it in e.payload["items"]]
    if len(recs) != 2 or not np.array_equal(out["a"], out["a2"]) or np.array_equal(out["a"], out["b"]):
        failures.append("world: records were not one per content version")
    metrics = {"cache_invalidation_failures": len(failures), "cache_hits": int(hits),
               "recomputes_after_change": int(recomputes), "modalities_checked": len(E.MODALITIES),
               "world_records": len(recs)}
    record("E.4", metrics, command=CMD, notes=f"failures: {failures}; edited-vs-original cosine: {log}")
    assert not failures, failures
    assert hits >= 1 and recomputes >= 1


# ============================================================================================== E.5
def test_E5_no_creator_assigned_quality_scores():
    rep = A.quality_fields()
    in_scope_reads = sum(1 for r in rep["violations"] if r["participant_scope"])
    metrics = {"creator_assigned_quality_fields_read": rep["creator_assigned_quality_fields_read"],
               "creator_assignable_fields": rep["creator_assignable_count"],
               "participant_scope_reads": in_scope_reads, "files_scanned": rep["files_scanned"],
               "quality_like_reads_anywhere": rep["reads"], "exempt_lines": len(rep["exempt"]),
               "out_of_scope_files": len(rep["out_of_scope_files"])}
    record("E.5", metrics, command=CMD, notes=(
        "Static audit (crucible.semantic.encoders.audit.quality_fields) over src/crucible: tool input schemas of "
        "every registered module and public functions of content/participant packages for quality/virality/"
        "relevance-like fields, and every read of such a field. Out-of-scope reads (not content scores): "
        f"{rep['out_of_scope_files']}. Exempt lines: {rep['exempt']}. Encoders take only artifact bytes; "
        "no metadata argument exists through which a score could reach them."))
    assert rep["creator_assigned_quality_fields_read"] == 0, rep["violations"]
    assert rep["creator_assignable_count"] == 0, rep["creator_assignable_fields"]


# ============================================================================================== E.n1
def _perturbed(role: str, fn):
    """Run fn() with one weight matrix of the model scaled, then restore it exactly."""
    import torch
    ld = MD.load(role)
    name = {"text": "encoder.layer.5.output.dense.weight", "image": "visual_projection.weight",
            "audio": "audio_spectrogram_transformer.layers.11.mlp.fc2.weight"}[role]
    target = dict(ld.model.named_parameters())[name]
    saved = target.detach().clone()
    try:
        with torch.no_grad():
            target.mul_(0.5).add_(torch.flip(saved, dims=[0]) * 0.5)
        return fn()
    finally:
        with torch.no_grad():
            target.copy_(saved)


def test_En1_real_model_execution_not_stub_vectors(encoded, tmp_path):
    suites = {m: encoded[m]["suite"] for m in E.MODALITIES}
    samples = {"text": suites["text"].items["sports"], "image": suites["image"].items["sailboat"],
               "audio": suites["audio"].items["speech_finance"], "video": suites["video"].items["music_video"],
               "code": suites["code"].items["sorting"]}
    f0 = sum(MD.forward_passes().values())
    vecs = {m: E.encode_bytes(samples[m], m, cache=EmbeddingCache(disk=False)) for m in E.MODALITIES}
    forward = sum(MD.forward_passes().values()) - f0
    # the outputs depend on the weights: change one weight matrix -> the vector changes; restore -> it comes back
    dep = {}
    for m, role in (("text", "text"), ("image", "image"), ("audio", "audio"), ("video", "image"), ("code", "text")):
        changed = _perturbed(role, lambda: E.encode_bytes(samples[m], m, cache=EmbeddingCache(disk=False)))
        back = E.encode_bytes(samples[m], m, cache=EmbeddingCache(disk=False))
        dep[m] = {"cos_perturbed": round(cos(changed, vecs[m]), 6), "restored": cos(back, vecs[m]) > 0.99999}
    stub = A.stub_check({f"{m}:{n}": v for m in E.MODALITIES for n, v in encoded[m]["vectors"].items()},
                        {f"{m}:{n}": f"{m}:{n.split('~')[0]}" for m in E.MODALITIES for n in encoded[m]["vectors"]})
    weight_dependent = sum(1 for d in dep.values() if d["cos_perturbed"] < 0.9999 and d["restored"])
    metrics = {"stub_vectors_detected": stub["stub_vectors_detected"], "model_forward_passes": int(forward),
               "weight_dependent_modalities": weight_dependent, "vectors_checked": stub["checked"]}
    for m, d in dep.items():
        metrics[f"perturbed_weights_cosine_{m}"] = d["cos_perturbed"]
    record("E.n1", metrics, command=CMD, notes=(
        "Forward passes counted by hooks on the models while encoding one artifact per modality without any cache. "
        "Weight dependence: one weight matrix of the model is changed and the artifact re-encoded (the vector "
        "changes), then restored (the vector returns). Stub check over every suite vector: finite, non-zero, "
        f"non-constant, distinct for different content. Problems: {stub['problems']}"))
    assert stub["stub_vectors_detected"] == 0 and forward >= 4
    assert weight_dependent == len(E.MODALITIES), dep
