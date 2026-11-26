"""SP-4.2b: expensive generative models are not required for ordinary simulation actions (MASTER_SPEC 4.2).

A civilisation (`crucible.civ`, development population) runs one simulated day with every class active, including
its semantic participants (class 4, which do use generation for their own decisions). We count every generation
request any shared model served in this process (`SemanticService.stats`) and every generation request the class-4
runtime made (`LMBrain.calls`), next to the ordinary decisions of the day: the population's recorded decision
batches (browsing, consumption, purchases, work, social actions) and the rows they decided.
``ordinary_actions_requiring_generation`` = generation requests not made by class-4 participants (must be 0).
Content the population writes comes from real corpora (`population.writing`), not from a generator.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 6000 --ram-mb 12000 --timeout 7200 \
        --label "D: SP-4.2b" -- python -m pytest -q tests/v5/test_gate_SP_4_2b.py
"""
import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


class _Decisions:
    """World listener: decision batches recorded on the population table (cols.batch with decision provenance)."""

    def __init__(self):
        self.batches, self.rows, self.models = 0, 0, {}

    def __call__(self, ev):
        d = (ev.payload.get("extra") or {}).get("decision") if ev.type == "cols.batch" else None
        if isinstance(d, dict):
            self.batches += 1
            self.rows += int(ev.payload.get("n", 0) or 0)
            mid = d.get("model_id", "?")
            self.models[mid] = self.models.get(mid, 0) + 1


def test_SP_4_2b_ordinary_actions_need_no_generation():
    import crucible
    from crucible.semantic import generalists as G, service as SV
    crucible.load_all_modules()
    w = G.civ_world("balanced", 3, population="development")
    dec = _Decisions()
    w.listeners.append(dec)
    before = sum(s.stats["generate_calls"] for s in list(SV._SERVICES_ALL))
    w.advance_time(w.now + 86400)
    total = sum(s.stats["generate_calls"] for s in list(SV._SERVICES_ALL)) - before
    brain = w.runtime.get("semp.brain")
    class4 = int(brain.calls["generate"]) if brain is not None else 0
    acts = sum(int(r["counts"].get("calls", 0)) for r in G.members(w))
    m = {"ordinary_actions_requiring_generation": int(total - class4), "generation_requests_total": int(total),
         "generation_requests_class4": class4, "class4_tool_calls": acts,
         "ordinary_decision_batches": dec.batches, "ordinary_decision_rows": dec.rows,
         "decision_models": len(dec.models)}
    record("SP-4.2b", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_2b.py")
    w.close()
    assert m["ordinary_decision_batches"] > 0 and m["class4_tool_calls"] > 0
    assert m["ordinary_actions_requiring_generation"] == 0
