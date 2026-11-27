"""SP-13.4: the shared semantic inference service and its nine features (MASTER_SPEC 13.4), measured on real models.

Each feature counts only when a measurement shows it working:

1. batching - batched left-padded greedy generation equals one-request-at-a-time generation, and the cross-caller
   continuous batcher serves several callers in shared decode steps;
2. quantization - an int8 dynamically quantized copy runs on CPU (its own weights hash) and its option choices are
   compared with the float32 model's (agreement reported);
3. short contexts - prompts beyond the context bound are cut (head and tail kept, flagged), and the prompts semantic
   participants build in a civilisation stay within the bound;
4. retrieval of relevant state - over a participant's full tool interface the service shortlist contains the
   service of the right tool (recall on the dev items of `generalists.bench`), and memory retrieval returns the
   remembered fact that answers a question from among unrelated memories;
5. prefix caching - identical outputs with the cache on and off, and a speedup on a long shared prefix;
6. bounded generation - a request asking for 500 tokens stops at the service's bound;
7. model versioning - revision and weights hash identify the weights; recorded inference carries the hash;
8. CPU execution - the model runs on the CPU;
9. selective GPU execution - the device policy puts a model on the GPU only with a sufficient slot grant, and in
   this (GPU-granted) slot the routed model actually runs on CUDA.

Plus: no network-dependent call in the default simulation (an audit hook counts every socket connect to a
non-loopback address during a civilisation day with semantic participants; the semantic models load with
``local_files_only``).

Run through the slot scheduler with a GPU grant:
    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 4000 --ram-mb 12000 --timeout 3600 \
        --label "D: SP-13.4" -- python -m pytest -q tests/v5/test_gate_SP_13_4.py
"""
import os
import sys
import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYS = {"role": "system", "content": "You are a resident of a small simulated town. Answer briefly and precisely. " * 4}
QS = ["What colour is grass?", "Name a large animal that lives in the sea.", "What is 2+2?",
      "Which season comes after winter?", "What do bees make?", "Name a fruit that is yellow."]


def _reqs(qs, prefix=True, **kw):
    return [{"messages": [SYS, {"role": "user", "content": q}], **({"prefix_messages": 1} if prefix else {}), **kw}
            for q in qs]


@pytest.fixture(scope="module")
def cpu_svc():
    import torch
    from crucible.semantic.service import SemanticService
    torch.set_num_threads(1)
    s = SemanticService(SMALL, device="cpu", max_batch=6, max_new_tokens=16, max_context=256)
    yield s


def _batching(svc) -> dict:
    from crucible.semantic.batching import ContinuousBatcher
    svc.prefix_cache = False
    try:
        batched = [o["text"] for o in svc.generate(None, _reqs(QS, prefix=False))]
    finally:
        svc.prefix_cache = True
    single = [svc.generate(None, [r])[0]["text"] for r in _reqs(QS, prefix=False)]
    b = ContinuousBatcher(svc, max_batch=4)
    futs = [b.submit(r, caller=f"caller{i}") for i, r in enumerate(_reqs(QS, prefix=False))]
    cont = [f.result(timeout=300)["text"] for f in futs]
    st = dict(b.stats)
    b.close()
    return {"batched_equals_single": sum(x == y for x, y in zip(batched, single)), "n": len(QS),
            "continuous_equals_batched": sum(x == y for x, y in zip(cont, batched)),
            "mixed_caller_steps": int(st["mixed_caller_steps"]), "max_live_rows": int(st["max_live"]),
            "ok": batched == single and cont == batched and st["mixed_caller_steps"] > 0}


def _quantization(svc) -> dict:
    from crucible.semantic.service import SemanticService
    q = SemanticService(SMALL, device="cpu", max_batch=6, quantization="int8-dynamic")
    reqs = [{"messages": [SYS, {"role": "user", "content": f"{x} Answer yes or no."}], "choices": ["yes", "no"],
             "normalize": "sum"} for x in ("Is the sky blue on a clear day?", "Is fire cold?", "Do fish swim?",
                                           "Can a cat fly?", "Is ice frozen water?", "Is the sun dark?")]
    a = [o["choice"] for o in svc.choose(None, reqs)]
    b = [o["choice"] for o in q.choose(None, reqs)]
    out = {"int8_runs_on": q.device, "int8_weights_hash": q.model_hash()[:16], "fp32_weights_hash":
           svc.model_hash()[:16], "int8_choice_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 3)}
    out["ok"] = q.device == "cpu" and q.model_hash() != svc.model_hash() and len(b) == len(reqs)
    q.close()
    return out


def _short_contexts(svc) -> dict:
    long = "word " * 2000
    o = svc.generate(None, [{"messages": [SYS, {"role": "user", "content": long + " What is the last word?"}],
                             "max_new_tokens": 4}])[0]
    return {"long_prompt_tokens_kept": int(o["prompt_tokens"]), "max_context": svc.max_context,
            "truncated_flag": bool(o.get("truncated")),
            "ok": bool(o.get("truncated")) and int(o["prompt_tokens"]) <= svc.max_context}


def _participant_prompts() -> dict:
    """Prompt sizes semantic participants build in a civilisation (persona, situation, memory, choices)."""
    from crucible.semantic.generalists import bench, mind as M
    from transformers import AutoTokenizer
    items = bench.build(0, 1, ("dev",))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    sizes = []
    for it in items:
        a = bench.agent_of(it)
        a.intent = "Next I will do the first step of my goal."
        hosts = sorted({t["name"].split(".", 1)[0] for t in a.tools})[:10]
        for req in (M.intent_request(a, it["t"], None), M.service_request(a, it["t"], hosts,
                                                                        it["state"]["summaries"], None)):
            text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
            sizes.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    return {"participant_prompt_tokens_median": int(np.median(sizes)), "participant_prompt_tokens_max": int(max(sizes)),
            "items": items}


def _retrieval(items) -> dict:
    """Service recall where the participants use it: the shortlist of services retrieved for each participant's own
    next-step phrase (`mind.decide` with the routed models, as in the world) contains the service of a right tool;
    beside it the recall for the goal sentence alone (no model)."""
    from crucible.semantic.generalists import bench, brain as B, catalog, mind as M
    hit, n = 0, 0
    for it in items:
        a = bench.agent_of(it)
        hs = catalog.rank_hosts([it["goal"]], [a.tools], M.Config().hosts, [it["goal"]],
                                it["state"]["summaries"])[0]
        hit += bool({x.split(".", 1)[0] for x in it["accept"]} & set(hs))
        n += 1
    live = bench.evaluate(items, B.LMBrain())["splits"]["dev"]
    facts = ["The bakery on Elm street closes at 6 pm.", "Ana paid 2500 cents for the logo on Tuesday.",
             "The forum board 'help' is for questions.", "My server instance runs on provider cloud1.",
             "Ben's birthday is in March.", "The museum is closed on Mondays.", "The repository textutils has tests.",
             "Gus asked about cheap hosting.", "The train to the coast leaves at 9."]
    a = M.Agent(i=0, principal="p0", account=None, persona={"about": ""})
    for k, f in enumerate(facts * 3):
        a.memory.append({"t": k, "kind": "obs", "text": f if k < len(facts) else f"Unrelated note {k}: weather mild."})
    q = catalog.embed(["How much did Ana pay for the logo?"])[0]
    got = a.relevant_memory(q, k=6)
    return {"host_recall_dev": live["host_recall"], "host_recall_dev_goal_only": round(hit / max(1, n), 3),
            "dev_items": n, "memory_retrieval_found_fact": "2500 cents" in got,
            "ok": live["host_recall"] >= 0.8 and "2500 cents" in got}


def _prefix_cache(svc) -> dict:
    big = {"role": "system", "content": ("You are a participant in a simulated economy with banks, mail, documents, "
                                         "forums and code. " * 40)}
    reqs = [{"messages": [big, {"role": "user", "content": f"Is {x} a colour? Answer yes or no."}],
             "choices": ["yes", "no"], "normalize": "sum", "prefix_messages": 1}
            for x in ("red", "table", "green", "seven", "blue", "music")]
    m = svc.measure_prefix_cache(reqs, kind="choose", calls=4)
    g = svc.measure_prefix_cache(_reqs(QS), kind="generate", calls=2)
    return {"choose_speedup": m["speedup"], "choose_identical": f"{m['identical_outputs']}/{m['outputs']}",
            "generate_identical": f"{g['identical_outputs']}/{g['outputs']}", "generate_speedup": g["speedup"],
            "ok": m["identical_outputs"] == m["outputs"] and g["identical_outputs"] == g["outputs"]
            and (m["speedup"] or 0) > 1.0}


def _bounded(svc) -> dict:
    o = svc.generate(None, [{"messages": [SYS, {"role": "user", "content": "Count from one to five hundred."}],
                             "max_new_tokens": 500}])[0]
    return {"asked_tokens": 500, "service_bound": svc.max_new_tokens, "generated": int(o["gen_tokens"]),
            "ok": int(o["gen_tokens"]) <= svc.max_new_tokens}


def _versioning(svc) -> dict:
    import crucible
    from crucible.core.kernel import World, WorldSpec
    crucible.load_all_modules()
    info = svc.info()
    w = World.create(WorldSpec("sp134", 0, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                            {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                            {"name": "semantic_inference", "config": {}}]))
    from crucible.semantic.recording import record_inference
    record_inference(w, svc, "generate", _reqs(QS[:2]))
    rec = w.state.tables.get("semantic_inference", {})
    return {"revision": info["revision"][:12], "weights_hash": info["weights_hash"][:16],
            "recorded_model_hashes": sorted(rec), "ok": bool(info["revision"]) and info["weights_hash"] in rec}


def _selective_gpu() -> dict:
    from crucible.semantic import routing as R
    from crucible.semantic.service import _LOADED
    import torch
    # the policy for a model not yet on the GPU in this process (a model already placed there is shared, whatever
    # the grant: the retrieval check above may have placed the routed models)
    on_gpu = {k[0] for k in list(_LOADED) if str(k[2]).startswith("cuda")}
    probe = next(m for m in sorted(R.PARAMS, key=R.PARAMS.get) if m not in on_gpu)
    saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CRUCIBLE_SLOT_GPU_MB")}
    try:
        os.environ["CRUCIBLE_SLOT_GPU_MB"] = "100"
        small_grant = R.device_for(probe)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    granted = R.granted_gpu_mb()
    dev, _ = R.device_for("Qwen/Qwen3-0.6B")
    ran = None
    if dev == "cuda":
        s = R.service_for_model("Qwen/Qwen3-0.6B", max_batch=4, max_new_tokens=8)
        s.generate(None, _reqs(QS[:2]))
        ran = s.device
    return {"cuda_available": bool(torch.cuda.is_available()), "slot_grant_mb": granted, "probe_model": probe,
            "device_with_100mb_grant": small_grant[0], "device_in_this_slot": dev, "ran_on": ran,
            "ok": small_grant[0] == "cpu" and dev == "cuda" and ran == "cuda"}


def _network_calls() -> dict:
    """Socket connects to non-loopback addresses during a civilisation day with semantic participants."""
    from crucible.semantic import generalists as G
    calls = []

    def hook(event, args):
        if event == "socket.connect":
            addr = args[1] if len(args) > 1 else None
            host = addr[0] if isinstance(addr, tuple) and addr else str(addr)
            if host and not (str(host).startswith("127.") or host in ("::1", "localhost") or str(host).startswith("/")):
                calls.append(str(host))
    sys.addaudithook(hook)
    w = G.civ_world("balanced", 0, population="development", config={"steps": 2})
    w.advance_time(w.now + 86400)
    acts = sum(int(r["counts"].get("calls", 0)) for r in G.members(w))
    w.close()
    return {"network_calls": len(calls), "participant_calls": acts, "ok": len(calls) == 0 and acts > 0}


def test_SP_13_4_inference_service_features(cpu_svc):
    feats = {"batching": _batching(cpu_svc), "quantization": _quantization(cpu_svc),
             "short_contexts": _short_contexts(cpu_svc)}
    pp = _participant_prompts()
    feats["short_contexts"].update({k: v for k, v in pp.items() if k != "items"})
    feats["short_contexts"]["ok"] = (feats["short_contexts"]["ok"] and pp["participant_prompt_tokens_max"] <= 4096
                                     and pp["participant_prompt_tokens_median"] > 100)
    feats["retrieval"] = _retrieval(pp["items"])
    feats["prefix_caching"] = _prefix_cache(cpu_svc)
    feats["bounded_generation"] = _bounded(cpu_svc)
    feats["versioning"] = _versioning(cpu_svc)
    feats["cpu_execution"] = {"device": cpu_svc.device, "ok": cpu_svc.device == "cpu"
                              and cpu_svc.usage()["requests"] > 0}
    feats["selective_gpu"] = _selective_gpu()
    net = _network_calls()
    m = {"inference_service_features": sum(bool(f["ok"]) for f in feats.values()),
         "network_calls_in_default_simulation": net["network_calls"],
         "participant_calls_in_network_probe": net["participant_calls"]}
    for name, f in feats.items():
        m[f"feature_{name}"] = bool(f["ok"])
        for k, v in f.items():
            if k != "ok" and isinstance(v, (int, float, str, bool)):
                m[f"{name}.{k}"] = v
    record("SP-13.4", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_13_4.py",
           notes="Every feature is a measurement on real models (SmolLM2-135M-Instruct on CPU for exactness checks; "
                 "Qwen3-0.6B routed onto the GPU of this slot for selective execution; class-4 participants in a "
                 "civilisation for prompt sizes, retrieval and the network probe).")
    assert m["inference_service_features"] == 9, {k: f for k, f in feats.items() if not f["ok"]}
    assert m["network_calls_in_default_simulation"] == 0
