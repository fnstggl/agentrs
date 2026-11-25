"""Gate H: compact semantic generalists (MASTER_SPEC Part 17, Gate H; workstream D2).

One module-scoped run: eight semantic participants with their own goals share one Qwen3-0.6B behind the batched
`SemanticService` in a 4.0 open-economy world (`open_digital`), acting only through world tools for a fixed number of
rounds. Every test below measures one requirement on that run (or on the held-out evaluation sets built from world
situations) and records evidence. Nothing is asserted about thresholds: the evidence checker compares the measured
metrics with the manifest's criteria, so a shortfall is recorded honestly rather than hidden.

Run through the slot scheduler (GPU about 5 GB):
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 6000 --label "ws-D2: gate H" -- \
        python -m crucible.acceptance.v5 run --gate H
"""
import json
import os
import random
import time

import pytest

from crucible.acceptance.v5.evidence import record

MODEL = os.environ.get("CRUCIBLE_SEMANTIC_MODEL", "Qwen/Qwen3-0.6B")
ROUNDS = int(os.environ.get("CRUCIBLE_GATE_H_ROUNDS", "12"))
SMALLER = ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct")
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate H"

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]


@pytest.fixture(scope="module")
def service():
    from crucible.semantic.service import SemanticService
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    s = SemanticService(MODEL, device="auto", max_batch=16, max_new_tokens=640)
    s.info()
    yield s


@pytest.fixture(scope="module")
def society(service):
    from crucible.semantic.scenarios import build_society, run_society, verify
    t0 = time.time()
    soc = build_society(service, family="open_digital", seed=0)
    runner = run_society(soc, rounds=ROUNDS, dt=600, seed=0)
    wall = time.time() - t0
    return {"soc": soc, "runner": runner, "wall_s": wall, "verify": verify(soc.world, soc)}


@pytest.fixture(scope="module")
def sets():
    from crucible.semantic.capbench.run import load_sets
    return load_sets(0, ("tool_choice", "content_understanding"))


def _mc_eval(service, items, method):
    """Predictions for multiple-choice items with one of the participants' selection methods."""
    from crucible.semantic.capbench.models import _relabel
    from crucible.semantic.participants import SemanticGroup, SemanticParticipant
    if method == "label":
        reqs = [_relabel(i) for i in items]
    else:
        reqs = [{"messages": i["messages"], "choices": i["options"], "prefix_messages": i.get("prefix_messages"),
                 "normalize": "mean" if method == "think" else method} for i in items]
        if method == "think":
            for r in reqs:
                r["think_tokens"] = SemanticParticipant.think_tokens
    picks, _ = SemanticGroup(service).select(reqs, [""] * len(reqs))   # unrecorded: no world
    return picks


def test_H_1_eight_independent_participants(society):
    soc = society["soc"]
    acted = [m for m in soc.members if m.policy.stats["actions"] > 0]
    principals = {m.principal for m in acted}
    record("H.1", {"semantic_participants": len(principals), "participants_admitted": len(soc.members),
                   "world_family": soc.world.spec.family, "rounds": ROUNDS,
                   "actions_total": sum(m.policy.stats["actions"] for m in soc.members),
                   "world_modules": len(soc.world.modules), "tools_in_world": len(soc.world.available_tools(
                       soc.members[0].principal))},
           command=CMD, notes="Participants acting through ordinary tools of a 4.0 open_digital world (the v5 "
                              "standard population profile is not merged yet); each has its own principal, policy "
                              "object, goal, memory and history.")


def test_H_2_shared_model_weights(society, service):
    from crucible.semantic.service import loaded_copies
    soc = society["soc"]
    ws = {id(m.policy.group.service.weights) for m in soc.members}
    ptrs = {m.policy.group.service.weights.model.get_input_embeddings().weight.data_ptr() for m in soc.members}
    record("H.2", {"model_weight_copies": loaded_copies(MODEL), "participants_sharing_weights":
                   sum(1 for m in soc.members if m.policy.group.service.weights is service.weights),
                   "distinct_weight_objects": len(ws), "distinct_embedding_storage": len(ptrs),
                   "weights_hash": service.model_hash(), "model_id": MODEL, "revision": service.revision},
           command=CMD)


def test_H_3_separate_goals_and_histories(society):
    soc = society["soc"]
    goals = {m.policy.goal for m in soc.members}
    hist = {m.policy.history_digest() for m in soc.members}
    record("H.3", {"distinct_goals": len(goals), "distinct_histories": len(hist),
                   "participants": len(soc.members),
                   "history_lengths": json.dumps([len(m.policy.history) for m in soc.members]),
                   "memory_items": json.dumps([len(m.policy.memory) for m in soc.members])}, command=CMD)


def test_H_4_tool_selection_on_held_out_situations(service, sets):
    from crucible.semantic.participants import SemanticParticipant
    from crucible.semantic.service import poisson_binom_sf, wilson
    items = sets["tool_choice"]
    test = [i for i in items if i["split"] == "test"]
    dev = [i for i in items if i["split"] == "dev"]
    method = SemanticParticipant.choice_method
    preds = _mc_eval(service, test, method)
    k = sum(int(p == i["answer"]) for p, i in zip(preds, test))
    n = len(test)
    chance = [i["chance"] for i in test]
    rnd = random.Random(0)
    rand_k = sum(int(rnd.randrange(len(i["options"])) == i["answer"]) for i in test)
    lex_k = sum(int(i["lexical_choice"] == i["answer"]) for i in test)
    lo, hi = wilson(k, n)
    dev_acc = sum(int(p == i["answer"]) for p, i in zip(_mc_eval(service, dev, method), dev)) / max(1, len(dev))
    record("H.4", {"tool_selection_accuracy": round(k / n, 4), "tool_selection_vs_random_pvalue":
                   float(f"{poisson_binom_sf(k, chance):.3g}"), "n_situations": n, "ci95_lo": round(lo, 4),
                   "ci95_hi": round(hi, 4), "random_expected_accuracy": round(sum(chance) / n, 4),
                   "random_policy_accuracy": round(rand_k / n, 4), "lexical_bm25_accuracy": round(lex_k / n, 4),
                   "options_per_situation": len(test[0]["options"]), "held_out_templates": len({i["template"] for i in test}),
                   "dev_accuracy": round(dev_acc, 4), "scoring": method, "model_id": MODEL},
           command=CMD, notes="Held-out set: tool-choice situations staged in open_digital worlds from templates never "
                              "used for development; options are the correct tool, the most lexically similar tools, "
                              "random tools, world.find_tools and wait, rendered exactly as participants see them.")


def test_H_5_language_understanding(service, sets):
    from crucible.semantic.service import poisson_binom_sf, wilson
    items = sets["content_understanding"]
    test = [i for i in items if i["split"] == "test"]
    dev = [i for i in items if i["split"] == "dev"]
    dev_acc = {m: sum(int(p == i["answer"]) for p, i in zip(_mc_eval(service, dev, m), dev)) / len(dev)
               for m in ("mean", "sum", "label", "think")}
    method = max(dev_acc, key=lambda m: dev_acc[m])          # chosen on dev only
    preds = _mc_eval(service, test, method)
    hits = [int(p == i["answer"]) for p, i in zip(preds, test)]
    k, n = sum(hits), len(test)
    held = [h for h, i in zip(hits, test) if i["held_out_template"]]
    lo, hi = wilson(k, n)
    record("H.5", {"comprehension_accuracy": round(k / n, 4), "comprehension_vs_chance_pvalue":
                   float(f"{poisson_binom_sf(k, [i['chance'] for i in test]):.3g}"), "n_questions": n,
                   "chance": round(sum(i["chance"] for i in test) / n, 4), "ci95_lo": round(lo, 4),
                   "ci95_hi": round(hi, 4), "held_out_genre_accuracy": round(sum(held) / max(1, len(held)), 4),
                   "scoring": method, "dev_accuracy_by_scoring": json.dumps(dev_acc), "model_id": MODEL}, command=CMD,
           notes="Questions about documents created with docs.create and read with docs.get in a world; distractors "
                 "come from the same document (ruled-out causes, original times, other suppliers).")


def test_H_6_consequential_actions(society):
    from crucible.semantic.participants import consequential_actions
    soc = society["soc"]
    ca = consequential_actions(soc.world, [m.principal for m in soc.members])
    by_tool = {}
    for c in ca:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
    record("H.6", {"consequential_actions_executed": len(ca), "participants_with_consequential_actions":
                   len({c["actor"] for c in ca}), "by_tool": json.dumps(by_tool, sort_keys=True),
                   "ok_actions": sum(m.policy.stats["ok_actions"] for m in soc.members),
                   "rejected_actions": sum(m.policy.stats["actions"] - m.policy.stats["ok_actions"] for m in soc.members)},
           command=CMD, notes="Counted from the world's event log: successful non-read-only commands of the "
                              "participants whose transaction changed module state.")


def test_H_7_verified_productive_or_collaborative_result(society):
    v = society["verify"]
    ok = [h for h, r in v.items() if r.get("verified")]
    record("H.7", {"verified_productive_results": len(ok), "goals": len(v), "verified_goals": ",".join(ok),
                   "collaborative_verified": int(bool(v.get("dev", {}).get("verified"))),
                   "program_verified": int(bool(v.get("eli", {}).get("verified"))),
                   "details": json.dumps(v, sort_keys=True)[:3000]}, command=CMD,
           notes="Each goal is verified by an evaluator from world state only (documents, ledger postings, delivered "
                 "mail, repository files executed against hidden tests in the sandbox, forum posts, instances).")


def test_H_8_memory_and_inference_measurements(society, service):
    import torch
    from crucible.semantic.service import SemanticService, peak_vram_mb
    u = service.usage()
    soc = society["soc"]
    reqs = [m.policy.args_request({"name": "mail.send", "description": "Send a private message.",
                                   "inputSchema": {"type": "object", "properties": {"to": {"type": "array"},
                                                                                     "subject": {"type": "string"},
                                                                                     "body": {"type": "string"}},
                                                   "required": ["to", "subject", "body"]}}) for m in soc.members]
    pc = service.measure_prefix_cache(reqs)
    cpu = SemanticService(MODEL, device="cpu", max_batch=8, max_new_tokens=24)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cpu.generate(None, [dict(r, max_new_tokens=4) for r in reqs[:1]])
    t0 = time.time()
    out = cpu.generate(None, [dict(r, max_new_tokens=24) for r in reqs])
    cpu_s = time.time() - t0
    cpu_tok = sum(o["gen_tokens"] for o in out)
    cpu.close()
    record("H.8", {"peak_vram_mb_measured": peak_vram_mb(service.device) is not None,
                   "tokens_per_second_measured": u["tokens_per_s"] is not None,
                   "peak_vram_mb": peak_vram_mb(service.device) or 0.0, "gpu_tokens_per_s": u["tokens_per_s"] or 0.0,
                   "prompt_tokens": u["prompt_tokens"], "gen_tokens": u["gen_tokens"], "batches": u["batches"],
                   "rows": u["rows"], "mean_batch_rows": round(u["rows"] / max(1, u["batches"]), 2),
                   "prefix_hits": u["prefix_hits"], "prefix_tokens_reused": u["prefix_tokens_reused"],
                   "prefix_cache_speedup": pc["speedup"] or 0.0, "prefix_cache_identical_outputs": pc["identical_outputs"],
                   "cpu_tokens_per_s": round(cpu_tok / cpu_s, 2), "cpu_threads": torch.get_num_threads(),
                   "cpu_batch_s": round(cpu_s, 2), "society_wall_s": round(society["wall_s"], 1),
                   "device": service.device, "model_params": service.info()["params"]}, command=CMD)


def test_H_n1_smallest_effective_models(service, sets):
    """The default model's size, and smaller alternatives benchmarked on the same held-out sets (the full comparison
    over eight capabilities is SP-3.2b)."""
    from crucible.semantic.service import SemanticService
    res = {}
    for mid in SMALLER:
        s = SemanticService(mid, device="auto", max_batch=16)
        for cap in ("tool_choice", "content_understanding"):
            test = [i for i in sets[cap] if i["split"] == "test"]
            preds = _mc_eval(s, test, "mean")
            res[f"{mid.split('/')[-1]}:{cap}"] = round(sum(int(p == i["answer"]) for p, i in zip(preds, test)) / len(test), 4)
        res[f"{mid.split('/')[-1]}:params"] = s.info()["params"]
        s.close()
    record("H.n1", {"semantic_model_params": service.info()["params"], "smaller_alternatives_benchmarked":
                    len(SMALLER), "default_model": MODEL, **res}, command=CMD)
