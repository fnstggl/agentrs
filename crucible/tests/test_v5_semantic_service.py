"""Crucible 5.0 workstream D2: the shared small-model inference service and its recording in kernel worlds.

These tests run a real small model (SmolLM2-135M-Instruct from the local Hugging Face cache) on the CPU: batched,
prefix-cached and option-scoring paths are checked against plain one-request-at-a-time computation, and recorded
inference is checked to replay and regenerate a world without running the model again.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from crucible.semantic import service as SV  # noqa: E402

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYS = {"role": "system", "content": "You are a terse assistant in a simulated town. Answer in a few words. " * 6}


@pytest.fixture(scope="module")
def svc():
    try:
        SV.local_revision(SMALL)
    except FileNotFoundError:   # a test that cannot run fails (docs/v5/INTERFACE_CONTRACTS.md 0.7)
        pytest.fail(f"{SMALL} is not in the local Hugging Face cache: download it once with "
                    "huggingface_hub.snapshot_download (about 270 MB)")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    s = SV.SemanticService(SMALL, device="cpu", max_batch=4, max_new_tokens=12)
    yield s
    s.close()


def _reqs(qs, prefix=True):
    return [{"messages": [SYS, {"role": "user", "content": q}], **({"prefix_messages": 1} if prefix else {})}
            for q in qs]


QS = ["What colour is grass?", "Name a large animal that lives in the sea, please.", "2+2=?",
      "Say the word hello backwards, letter by letter, and nothing else."]


def _one_by_one(svc, reqs):
    """Reference: plain Hugging Face greedy generation, one request at a time, no cache tricks."""
    w = svc.weights
    out = []
    for r in reqs:
        text = svc._template(r["messages"], True)
        ids = w.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        g = w.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=w.pad_id)
        out.append(w.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return out


def test_batched_left_padded_generation_matches_unbatched(svc):
    svc.prefix_cache = False
    try:
        got = [o["text"] for o in svc.generate(None, _reqs(QS, prefix=False))]
    finally:
        svc.prefix_cache = True
    assert got == _one_by_one(svc, _reqs(QS))
    assert svc.stats["padding_tokens"] > 0          # the prompts really had different lengths


def test_prefix_cache_reuses_shared_prefix_and_gives_identical_text(svc):
    svc.weights.prefixes.clear()
    h0, m0 = svc.stats["prefix_hits"], svc.stats["prefix_misses"]
    with_cache = svc.generate(None, _reqs(QS))
    assert all(o["prefix_tokens"] > 30 for o in with_cache)
    assert svc.stats["prefix_misses"] - m0 == 1 and svc.stats["prefix_hits"] - h0 == len(QS) - 1
    again = svc.generate(None, _reqs(QS))
    assert svc.stats["prefix_hits"] - h0 == 2 * len(QS) - 1                     # the second call reuses it
    assert [o["text"] for o in with_cache] == [o["text"] for o in again] == _one_by_one(svc, _reqs(QS))
    m = svc.measure_prefix_cache(_reqs(QS), calls=3)
    assert m["identical_outputs"] == m["outputs"] == 3 * len(QS) and m["prefix_tokens"] > 30


def test_prefix_cache_is_exact_for_option_scoring(svc):
    """Prefix-first evaluation makes the cache exact by construction: identical log probabilities on/off."""
    reqs = [{"messages": [SYS, {"role": "user", "content": q}], "choices": ["yes", "no", "maybe"],
             "normalize": "sum", "prefix_messages": 1} for q in QS]
    m = svc.measure_prefix_cache(reqs, kind="choose", calls=3)
    assert m["identical_outputs"] == m["outputs"] == 3 * len(QS)


def test_continuous_batching_serves_several_callers_in_shared_steps(svc):
    import threading
    from crucible.semantic.batching import ContinuousBatcher
    ref = [o["text"] for o in svc.generate(None, _reqs(QS, prefix=False))]
    b = ContinuousBatcher(svc, max_batch=4)
    futs = {}

    def caller(name, qs):
        for q in qs:
            futs[(name, q)] = b.submit(_reqs([q], prefix=False)[0], caller=name)
    ts = [threading.Thread(target=caller, args=(f"p{i}", [q])) for i, q in enumerate(QS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    got = [futs[(f"p{i}", q)].result(timeout=120)["text"] for i, q in enumerate(QS)]
    b.close()
    assert got == ref                                     # float32 CPU: same greedy tokens as batched generate
    assert b.stats["completed"] == len(QS) and b.stats["mixed_caller_steps"] > 0
    assert b.stats["steps"] < sum(len(svc.weights.tok(t)["input_ids"]) + 1 for t in got)


def test_routing_escalates_hard_capabilities_and_policy_needs_a_slot(monkeypatch):
    from crucible.semantic import routing as R
    monkeypatch.setenv("CRUCIBLE_DATA", "/nonexistent")
    assert R.route("programming") == R.route("negotiation") == R.route("strategic_decisions") == "Qwen/Qwen3-1.7B"
    assert R.route("tool_choice") == R.route("communication") == "Qwen/Qwen3-0.6B"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert R.device_for("Qwen/Qwen3-1.7B") == ("cpu", "int8-dynamic")
    assert R.device_for("Qwen/Qwen3-0.6B") == ("cpu", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CRUCIBLE_SLOT_GPU_MB", "500")         # too small a grant: stay on CPU
    assert R.device_for("Qwen/Qwen3-0.6B")[0] == "cpu"


def test_generation_is_bounded(svc):
    out = svc.generate(None, [{"messages": [SYS, {"role": "user", "content": "Count from one to one hundred."}],
                               "max_new_tokens": 5}, {"messages": [SYS, {"role": "user", "content": "Count to 50."}],
                                                       "max_new_tokens": 10_000}])
    assert out[0]["gen_tokens"] <= 5 and out[1]["gen_tokens"] <= svc.max_new_tokens
    assert out[0]["finish"] in ("length", "stop")


def test_sampling_is_seeded_per_request(svc):
    r = {"messages": [SYS, {"role": "user", "content": "Invent a name for a bakery."}], "temperature": 1.0}
    a = svc.generate(None, [dict(r, seed=7), dict(r, seed=7), dict(r, seed=8)])
    assert a[0]["text"] == a[1]["text"]
    texts = {svc.generate(None, [dict(r, seed=s)])[0]["text"] for s in range(5)}
    assert len(texts) > 1


def test_choose_matches_direct_log_probability(svc):
    msgs = [SYS, {"role": "user", "content": "Which tool sends money? Reply with the tool name."}]
    choices = ["bank.transfer", "mail.send", "docs.create"]
    out = svc.choose(None, [{"messages": msgs, "choices": choices, "normalize": "sum", "prefix_messages": 1}])[0]
    w = svc.weights
    prompt = w.tok(svc._template(msgs, True), add_special_tokens=False)["input_ids"]
    ref = []
    for c in choices:
        opt = w.tok(c, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt + opt])
        with torch.no_grad():
            lp = torch.log_softmax(w.model(ids).logits.float(), -1)[0]
        ref.append(sum(float(lp[len(prompt) - 1 + j, t]) for j, t in enumerate(opt)))
    assert out["logprobs"] == pytest.approx(ref, abs=2e-3)
    assert out["choice"] == max(range(3), key=lambda i: ref[i])


def test_one_weight_copy_per_model_and_revision(svc):
    other = SV.SemanticService(SMALL, device="cpu")
    try:
        assert other.weights is svc.weights
        assert SV.loaded_copies(SMALL) == 1
        info = other.info()
        assert info["revision"] == SV.local_revision(SMALL) and len(info["weights_hash"]) == 64
        assert info["params"] > 100_000_000
    finally:
        other.close()
    assert SV.loaded_copies(SMALL) == 1          # svc still holds the weights


def test_weights_hash_is_over_parameter_bytes():
    m = torch.nn.Linear(3, 2)
    h1 = SV.weights_hash(m)
    assert h1 == SV.weights_hash(m.state_dict())
    with torch.no_grad():
        m.weight[0, 0] += 1e-3
    assert SV.weights_hash(m) != h1


def test_int8_dynamic_quantization_runs_on_cpu():
    q = SV.SemanticService(SMALL, device="cpu", quantization="int8-dynamic", max_new_tokens=6)
    try:
        out = q.generate(None, _reqs(QS[:2]))
        assert all(isinstance(o["text"], str) and o["gen_tokens"] >= 1 for o in out)
        assert q.info()["quantization"] == "int8-dynamic" and q.weights is not None
        assert SV.loaded_copies(SMALL) == 2      # a different numeric variant is a different weight copy
    finally:
        q.close()


def test_statistics_helpers():
    assert SV.binom_sf(0, 10, 0.3) == 1.0
    assert SV.binom_sf(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert SV.poisson_binom_sf(2, [0.5, 0.5]) == pytest.approx(0.25)
    lo, hi = SV.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21


# ---------------------------------------------------------------------------------------------- recording in a world
def _world():
    import crucible
    from crucible.core.kernel import World
    from crucible.worldgen import generator as G
    crucible.load_all_modules()
    return World.create(G.generate({"family": "tiny", "extra_modules": ["semantic_inference"]}, 0))


def test_recorded_inference_replays_and_regenerates_without_the_model(svc):
    from crucible.core.kernel import World
    from crucible.semantic.recording import record_inference
    w = _world()
    p = w.system_call("admin.join", {"handle": "alice", "endowment": 1000})["principal"]
    out = record_inference(w, svc, "generate", _reqs(QS[:2]), [p, p], "test")
    ch = record_inference(w, svc, "choose", [{"messages": [SYS, {"role": "user", "content": "Pick one."}],
                                              "choices": ["red", "blue"]}], [p], "test")
    assert len(out) == 2 and out[0]["text"] and ch[0]["choice"] in (0, 1)
    evs = [e for e in w.events_range(0) if e.type == "semantic.inference"]
    assert len(evs) == 2 and evs[0].vis == "internal" and evs[0].payload["model_hash"] == svc.model_hash()
    assert w.state.get("semantic_inference", svc.model_hash())["requests"] == 3
    log = [e.to_dict() for e in w.events_range(0)]
    # no prompt text in the log: only digests
    assert QS[0] not in str(log)
    calls = []
    orig = svc.run
    svc.run = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        rg = World.regenerate(w.spec, log)
        rp = World.replay(w.spec, log)
    finally:
        svc.run = orig
    assert calls == []                          # the model never ran during regeneration
    assert rg.last_hash == w.last_hash and rp.digest() == w.digest()


def test_in_world_actors_cannot_use_or_see_the_recording_tool(svc):
    w = _world()
    p = w.system_call("admin.join", {"handle": "bob", "endowment": 1000})["principal"]
    assert not [t for t in w.available_tools(p) if "semantic" in t["name"]]
    r = w.execute(p, {"tool": "admin.semantic_infer", "args": {"model_id": SMALL, "model_hash": "x" * 64,
                                                               "kind": "generate", "requests": [{"digest": "d"}]}})
    assert not r.ok and r.error["code"] == "forbidden"


def test_semantic_inference_module_passes_admission():
    from crucible.worldgen.admission import admit
    rep = admit(["semantic_inference"], steps=120)
    assert rep.passed, rep.errors
