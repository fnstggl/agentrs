"""Gate H: compact semantic generalists (MASTER_SPEC Part 17, Gate H; Part 3.2 class 4; workstream D).

One module-scoped run: a STANDARD civilisation (`crucible.civ`, population profile "standard": 10,000 lightweight,
256 adaptive, 32 strategic and 8 semantic individuals) whose class-4 individuals are bound to the semantic
participant runtime (`crucible.semantic.generalists`, kernel module `pop_semantic`). Nobody assigns them anything:
each forms its own goals from its own state, chooses services and tools over its whole tool interface and writes
the arguments with the shared small models (Qwen3-0.6B by default, routed per capability by
`crucible.semantic.routing`), all inference recorded in the world. The run lasts `CRUCIBLE_GATE_H_DAYS` simulated
days (default 3). Every requirement is measured on that run by evaluator-side code (`generalists.verify`), or - for
H.4 and H.5 - on held-out item sets through the participants' own decision code and models.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 6000 --ram-mb 24000 --timeout 14400 \
        --label "D: gate H" -- python -m pytest -q tests/v5/test_gate_H.py
"""
import hashlib
import json
import os
import resource
import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record

DAYS = float(os.environ.get("CRUCIBLE_GATE_H_DAYS", "3"))
SMALLER = ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct")
ORDINARY = ("situation_interpretation", "next_step", "tool_choice", "communication", "content_understanding")
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_H.py"

pytestmark = [pytest.mark.slow, pytest.mark.timeout(14400)]


def _artifact(name: str, obj) -> str:
    """A JSON artifact beside the evidence (official runs) or in the unofficial evidence directory."""
    if os.environ.get("CRUCIBLE_V5_RUN_ID"):
        from crucible.acceptance.v5.manifest import docs_dir
        d = os.path.join(docs_dir(), "evidence", "gate_H")
    else:
        d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-unofficial-evidence", "H")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return path


@pytest.fixture(scope="module")
def society():
    import torch
    import crucible
    from crucible.semantic import generalists as G, routing
    from crucible.semantic.generalists import verify
    crucible.load_all_modules()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    w = G.civ_world("balanced", 0, population="standard", overrides={"params": {"log_retain": 20_000_000}})
    gen_s = time.time() - t0
    t1 = time.time()
    w.advance_time(w.now + int(DAYS * 86400))
    run_s = time.time() - t1
    brain = w.runtime.get("semp.brain")
    services = dict(routing._SERVICES)
    out = {"world": w, "summary": verify.summary(w), "actions": verify.actions(w), "gen_s": gen_s, "run_s": run_s,
           "services": services, "rows": G.members(w), "brain": brain,
           "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
           # what the standard run itself used, taken before any held-out evaluation (H.4, H.5, H.n1) sends requests
           # through the same shared services: decisions per capability and each service's counters
           "run_decisions_by_capability": {c: int(n) for c, n in sorted(brain.requests.items()) if n},
           "run_usage": {m: sv.usage() for m, sv in services.items()}}
    out["rows_artifact"] = _artifact("H_society_rows.json", {"rows": out["rows"], "summary": out["summary"]})
    yield out
    w.close()


def _mc_eval(service, items, method):
    """Predictions for multiple-choice capbench items with one scoring method (as SP-3.2b scores them)."""
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
    picks, _ = SemanticGroup(service).select(reqs, [""] * len(reqs))
    return picks


def test_H_1_eight_independent_participants_in_a_standard_world(society):
    s, w = society["summary"], society["world"]
    from crucible.substrate import columns
    cls = columns.col(w, "pop", "class")
    m = {"semantic_participants": s["acting"], "participants_bound": s["bound"]["bound"],
         "population_individuals": int(cls.shape[0]), "class4_rows": int((cls == 4).sum()),
         "population_profile": w.spec.params.get("civ_population"), "world_family": w.spec.family,
         "simulated_days": DAYS, "calls": s["calls"], "ok_calls": s["ok_calls"], "tools_used": s["tools_used"],
         "tools_available": len(w.available_tools(society["rows"][0]["principal"]))}
    record("H.1", m, command=CMD, artifacts=[society["rows_artifact"]],
           notes="Class-4 rows of the one population of a standard civilisation, bound to the semantic participant "
                 "runtime; acting = at least one successful call its own model chose. Every participant's state "
                 "(goals, memory, history, decision steps) is in the artifact.")
    assert m["population_profile"] == "standard" and m["class4_rows"] >= 8
    assert m["semantic_participants"] >= 8


def test_H_2_shared_model_weights(society):
    from crucible.semantic import service as SV
    services = society["services"]
    copies = {m: SV.loaded_copies(m) for m in services}
    served = [r for r in society["rows"] if int(r["counts"].get("activations", 0)) > 0]
    m = {"model_weight_copies": max(copies.values()) if copies else 0, "models_loaded": len(copies),
         "participants_sharing_weights": len(served),
         "distinct_weight_objects": len({id(s.weights) for s in services.values()}),
         "weights_hashes": ",".join(f"{k}={s.model_hash()[:12]}" for k, s in services.items())}
    record("H.2", m, command=CMD, notes="One process-wide service per routed model (crucible.semantic.routing); every "
                                        "participant's decisions are served by it; copies counted per model.")
    assert m["model_weight_copies"] == 1 and m["participants_sharing_weights"] >= 8


def test_H_3_separate_goals_and_histories(society):
    rows = society["rows"]
    firsts = [r["goals"][0]["text"] for r in rows if r["goals"]]
    hist = {hashlib.sha256(json.dumps([[h["t"], h["tool"], h["args"], h["ok"]] for h in r["history"]],
                                      sort_keys=True).encode()).hexdigest() for r in rows if r["history"]}
    m = {"distinct_goals": len(set(firsts)), "participants_with_goals": len(firsts),
         "goals_total": sum(len(r["goals"]) for r in rows), "distinct_histories": len(hist),
         "history_lengths": json.dumps([len(r["history"]) for r in rows]),
         "memory_items": json.dumps([len(r["memory"]) for r in rows]),
         "example_goals": json.dumps(firsts[:8])[:1500]}
    record("H.3", m, command=CMD, notes="Goals are written by each participant itself; histories and memories are "
                                        "rows of the world state (semp_state), persisted with the world.")
    assert m["distinct_goals"] >= 8 and m["distinct_histories"] >= 8


def test_H_4_tool_selection_on_held_out_situations(society):
    from crucible.semantic.generalists import bench
    from crucible.semantic.generalists import brain as B
    items = bench.build(0, 6, ("test",))
    run = society["brain"]
    # the participants' own decision code and routing (same routes and reasoning budgets), in a brain of its own so
    # the held-out items do not count as decisions of the standard run (H.n1)
    res = bench.evaluate(items, B.LMBrain(run.routes, think=run.think))
    t = res["splits"]["test"]
    art = _artifact("H4_items.json", res["items"])
    m = {"tool_selection_accuracy": t["accuracy"], "tool_selection_vs_random_pvalue": t["p_vs_random"],
         "items": t["n"], "ci95": json.dumps(t["ci95"]), "random_choice_accuracy": t["chance_mean"],
         "retrieval_only_accuracy": t["retrieval_only_accuracy"], "service_recall": t["host_recall"],
         "held_out_templates": len(t["by_template"]), "by_template": json.dumps(t["by_template"])}
    record("H.4", m, command=CMD, artifacts=[art], notes="Held-out templates (generalists.bench, split test): situations of a real "
                                        "class-4 participant in a civilisation, decided by the participants' own "
                                        "decision code over ALL tools available to them (several hundred).")
    assert m["tool_selection_accuracy"] >= 0.6 and m["tool_selection_vs_random_pvalue"] <= 0.01


def test_H_5_language_understanding(society):
    """Reading comprehension of documents read in a world (capbench content_understanding), the scoring method
    chosen on dev items, measured on held-out test items that a no-understanding shortcut (pick the option
    mentioned last in the text) gets wrong; plus the relevance of in-world replies to requests."""
    from crucible.semantic import routing
    from crucible.semantic.capbench.run import load_sets
    from crucible.semantic.service import poisson_binom_sf, wilson
    items = load_sets(0, ("content_understanding",))["content_understanding"]

    def last_mentioned(it):
        text = it["messages"][-1]["content"].lower()
        return int(np.argmax([text.rfind(o.lower()) for o in it["options"]]))
    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    ctrl = [i for i in test if last_mentioned(i) != i["answer"]]
    svc = routing.service("content_understanding")
    dev_acc = {meth: float(np.mean([p == i["answer"] for p, i in zip(_mc_eval(svc, dev, meth), dev)]))
               for meth in ("mean", "sum", "think")}
    method = max(sorted(dev_acc), key=lambda k: dev_acc[k])
    picks = _mc_eval(svc, ctrl, method)
    k = sum(int(p == i["answer"]) for p, i in zip(picks, ctrl))
    n = len(ctrl)
    lo, hi = wilson(k, n)
    coll = [c for c in society["summary"]["collaborative"] if c["kind"] == "answered_request"]
    m = {"comprehension_accuracy": round(k / max(1, n), 4),
         "comprehension_vs_chance_pvalue": float(f"{poisson_binom_sf(k, [i['chance'] for i in ctrl]):.3g}"),
         "items_shortcut_controlled": n, "items_test": len(test), "ci95": json.dumps([round(lo, 4), round(hi, 4)]),
         "shortcut_rule_accuracy_on_test": round(float(np.mean([last_mentioned(i) == i["answer"] for i in test])), 4),
         "scoring": method, "dev_accuracy_by_scoring": json.dumps(dev_acc),
         "in_world_answered_requests": len(coll),
         "in_world_reply_relevance_mean": round(float(np.mean([c["relevance"] for c in coll])), 4) if coll else 0.0,
         "model": svc.model_id}
    record("H.5", m, command=CMD, notes="Scored only on held-out items where the 'last mentioned option' rule fails "
                                        "(audit D skeptic S3), with the participants' routed model.")
    assert m["comprehension_accuracy"] >= 0.6 and m["comprehension_vs_chance_pvalue"] <= 0.01


def test_H_6_consequential_world_actions(society):
    s, acts = society["summary"], society["actions"]
    # evaluator side (generalists.verify): junk content (a tool or service name, prompt scaffolding, the goal sentence
    # restated, under two letters) does not count, and a repeated call with the same content (same participant, tool
    # and written content, or the same arguments when it wrote nothing) counts once (audit D skeptic)
    good = [a for a in acts if a["consequential"] and not a["junk"]]
    m = {"consequential_actions_executed": s["consequential"], "consequential_calls_raw": s["consequential_raw"],
         "consequential_junk": s["consequential_junk"], "consequential_repeats": s["consequential_repeats"],
         "ok_calls": s["ok_calls"], "calls": s["calls"],
         "rejected_share": round(s["rejected_calls"] / max(1, s["calls"]), 4),
         "by_tool": json.dumps(s["consequential_by_tool"])[:1500],
         "participants_with_consequential_actions": len({a["principal"] for a in good})}
    record("H.6", m, command=CMD, notes="Consequential = succeeded, not read-only, and emitted an event another "
                                        "principal or the public can see (checked in the event log by the "
                                        "evaluator); calls whose written content (event payload text fields, "
                                        "stored text artifacts) is junk are excluded and calls repeating the same "
                                        "content count once.")
    assert m["consequential_actions_executed"] >= 8


def test_H_7_verified_productive_or_collaborative_result(society):
    s = society["summary"]
    m = {"verified_productive_results": s["productive_verified"] + s["collaborative_verified"],
         "productive_verified": s["productive_verified"], "collaborative_verified": s["collaborative_verified"],
         "published_items": len(s["productive"]),
         "independent_viewers_max": max([p["independent_viewers"] for p in s["productive"]] or [0]),
         "collaborations": json.dumps(s["collaborative"])[:1500],
         "productive": json.dumps(s["productive"])[:1500]}
    record("H.7", m, command=CMD, notes="Productive: content a participant published that >= 3 independent population "
                                        "individuals viewed (exact audience sets). Collaborative: an on-topic reply "
                                        "to another participant's request, a co-edited document, or a payment "
                                        "between participants who talked.")
    assert m["verified_productive_results"] >= 1


def test_H_8_memory_and_inference_measurements(society):
    use = society["run_usage"]                    # the standard run's own requests (before H.4/H.5 evaluations)
    rows = society["rows"]
    state_bytes = sum(len(json.dumps(r, default=str)) for r in rows)
    tps = [u["tokens_per_s"] for u in use.values() if u.get("tokens_per_s")]
    vram = [u["peak_vram_mb"] for u in use.values() if u.get("peak_vram_mb") is not None]
    m = {"peak_vram_mb_measured": bool(vram), "tokens_per_second_measured": bool(tps),
         "peak_vram_mb": max(vram) if vram else 0.0, "tokens_per_second": max(tps) if tps else 0.0,
         "peak_host_rss_mb": round(society["maxrss_mb"], 1), "participant_state_bytes": state_bytes,
         "state_bytes_per_participant": round(state_bytes / max(1, len(rows))),
         "activations": sum(int(r["counts"].get("activations", 0)) for r in rows),
         "run_wall_s": round(society["run_s"], 1), "world_generation_s": round(society["gen_s"], 1),
         "requests": sum(u["requests"] for u in use.values()),
         "mean_latency_s_per_request": json.dumps({k: u.get("mean_latency_s_per_request") for k, u in use.items()}),
         "prefix_hits": sum(u["prefix_hits"] for u in use.values()),
         "devices": json.dumps({k: u["device"] for k, u in use.items()})}
    record("H.8", m, command=CMD)
    assert m["peak_vram_mb_measured"] and m["tokens_per_second_measured"]


def test_H_n1_smallest_effective_models(society):
    from crucible.semantic import routing
    from crucible.semantic.generalists import bench, brain as B
    items = bench.build(0, 3, ("test",))
    acc = {}
    for model in SMALLER:
        res = bench.evaluate(items, B.LMBrain({c: model for c in routing.TABLE}))
        acc[model] = res["splits"]["test"]["accuracy"]
    default = routing.DEFAULT
    brain = society["brain"]
    esc = [c for c in routing.capabilities() if routing.route(c) != default]
    # what the standard run actually used: decisions served per capability and requests served per model (every
    # class-4 decision goes through one of these process-wide services)
    by_cap = dict(society["run_decisions_by_capability"])      # the run's decisions only (not H.4's bench items)
    req = {mdl: int(u["requests"]) for mdl, u in society["run_usage"].items()}
    total = max(1, sum(by_cap.values()))
    # the size that counts is the largest model serving the decisions every participant makes in every activation
    # (its goal and next step, the service and tool it uses, the values it chooses and writes, what it reads):
    # with D-36/D-37 that was Qwen3-1.7B (situation interpretation and tool choice); capabilities only some
    # decisions need (programming, negotiation, strategy, incentives) may escalate, and their share is asserted small
    ordinary = [c for c in ORDINARY if c in by_cap]
    ord_params = max([routing.PARAMS.get(brain.model_for(c), 0) for c in ordinary] or [0])
    above = sum(n for c, n in by_cap.items() if routing.PARAMS.get(brain.model_for(c), 0) > 1_100_000_000)
    largest = max([routing.PARAMS.get(mdl, 0) for mdl, n in req.items() if n] or [0])
    m = {"semantic_model_params": ord_params, "default_model": default,
         "default_model_params": routing.PARAMS[default],
         "smaller_alternatives_benchmarked": len(acc), "smaller_alternative_accuracy": json.dumps(acc),
         "models_used": ",".join(sorted(society["services"])), "escalated_capabilities": ",".join(esc),
         "escalation_model_params": max([routing.PARAMS.get(routing.route(c), 0) for c in esc] or [0]),
         "ordinary_capabilities_served": ",".join(ordinary),
         "ordinary_models": json.dumps({c: brain.model_for(c) for c in ordinary}),
         "think_tokens": json.dumps({c: brain.think_tokens(c) for c in by_cap if brain.think_tokens(c)}),
         "decisions_by_capability": json.dumps(by_cap), "requests_by_model": json.dumps(req),
         "decision_share_above_1_1b_params": round(above / total, 4),
         "largest_model_params_used": largest}
    record("H.n1", m, command=CMD, notes="semantic_model_params is the largest model serving the class-4 decisions "
                                         "of every activation (situation interpretation, tool choice, writing, "
                                         "content understanding) in the standard run; routing uses the smallest "
                                         "model effective per capability (D-36 SP-3.2b held-out templates; D-37/"
                                         "D-38 in-situ class-4 bench, DEV templates). Capabilities some decisions "
                                         "need (programming, negotiation, strategy, incentives) may escalate to "
                                         "the escalation model; decisions per capability, requests per model, the "
                                         "share of decisions above 1.1B parameters and the largest model used are "
                                         "listed.")
    assert m["semantic_model_params"] <= 1_100_000_000 and m["smaller_alternatives_benchmarked"] >= 2
    assert ordinary and m["decision_share_above_1_1b_params"] <= 0.2
