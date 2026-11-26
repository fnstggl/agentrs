"""SP-3.2b: the semantic-model choice is benchmarked (MASTER_SPEC Part 3.2 class 4; workstream D2).

Candidates, smallest to largest: a hashed n-gram network trained from scratch (about 2M parameters), a task-specific
encoder (pretrained all-MiniLM-L6-v2, 22.7M, with heads trained per capability), SmolLM2-135M-Instruct,
SmolLM2-360M-Instruct, Qwen3-0.6B and Qwen3-1.7B. The eight capabilities are evaluated on test sets built from world
situations (`crucible.semantic.capbench.tasks`); test items come from held-out templates where a capability has
templates. The report holds accuracy or pass rate with Wilson 95% intervals, p-values against chance, memory and
latency, and the smallest effective model per capability.

Run through the slot scheduler (GPU about 9 GB for Qwen3-1.7B):
    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 9000 --label "ws-D2: SP-3.2b" -- \
        python -m crucible.acceptance.v5 run --requirement SP-3.2b
"""
import json
import os

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def test_SP_3_2b_semantic_model_benchmark():
    from crucible.semantic.capbench import CAPABILITIES
    from crucible.acceptance.v5.manifest import docs_dir
    from crucible.semantic.capbench.run import DEFAULT_LMS, primary, run
    if os.environ.get("CRUCIBLE_V5_RUN_ID"):     # an official run files the report next to its evidence record
        out = os.path.join(docs_dir(), "evidence", "gate_SP", "SP-3.2b_report.json")
    else:                                        # an ordinary run keeps it out of docs/v5/evidence (see evidence.py)
        out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-unofficial-evidence", "SP",
                           "SP-3.2b_report.json")
    rep = run(DEFAULT_LMS, CAPABILITIES, seed=0, device="auto", encoders=True, out=out)
    sizes = sorted({r["model"]["params"] for r in rep["results"]})
    measured = [c for c in CAPABILITIES
                if sum(1 for r in rep["results"] if primary(c, r["capabilities"].get(c, {}))["value"] is not None) >= 4]
    m = {"model_sizes_compared": len(sizes), "capabilities_benchmarked": len(measured),
         "candidates": ",".join(r["model"]["name"] for r in rep["results"]),
         "params": ",".join(str(r["model"]["params"]) for r in rep["results"]), "report_wall_s": rep["wall_s"]}
    for cap, rec in rep["recommendation"].items():
        m[f"recommended:{cap}"] = rec.get("recommended") or "none"
    for cap, rec in rep["recommendation_held_out"].items():
        m[f"recommended_held_out:{cap}"] = rec.get("recommended") or "none"
    for r in rep["results"]:
        nm = r["model"]["name"].split("/")[-1]
        for cap in CAPABILITIES:
            p = primary(cap, r["capabilities"].get(cap, {}))
            if p["value"] is not None:
                m[f"{nm}:{cap}"] = p["value"]
        if r.get("peak_vram_mb") is not None:
            m[f"{nm}:peak_vram_mb"] = r["peak_vram_mb"]
        if r.get("mc_ms_per_item") is not None:
            m[f"{nm}:mc_ms_per_item"] = r["mc_ms_per_item"]
        if r.get("gen_tokens_per_s") is not None:
            m[f"{nm}:gen_tokens_per_s"] = r["gen_tokens_per_s"]
    record("SP-3.2b", m, command="PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-3.2b",
           artifacts=[out],
           notes="Full report (per-capability intervals, p-values, held-out-template breakdown, dev-chosen scoring, "
                 f"examples) in the artifact. Recommendation: " + json.dumps(rep["recommendation"])[:2500])
    assert m["model_sizes_compared"] >= 4 and m["capabilities_benchmarked"] >= 8, m
