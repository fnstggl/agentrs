from cam_live import contracts as C
from cam_live.diagnostics.collect import LabCollector, collect_family, uniform_policy, verified_return
from cam_live.diagnostics.memstore import MemoryStore
from cam_live.diagnostics.world import LabConfig, LabWorld
from cam_live.learning_api import matured_outcomes


def test_collects_complete_honest_records():
    store = MemoryStore()
    assert isinstance(store, C.ExperienceStore)
    lt = LabCollector(store, max_context=24).run(LabWorld(LabConfig(seed=5)), uniform_policy(1), 30, seed=1)
    assert len(lt.records) == 30 and store.count() == 30
    acted = [r for r in lt.records if r.decision.chosen.operation == "lab.act"]
    assert acted and all(not r.pending_queries for r in lt.records)          # everything matured at the end
    for r in lt.records:
        assert abs(sum(r.decision.probabilities) - 1) < 1e-9 and r.provenance == "diagnostic"
        assert len(r.context_item_refs) <= 24 and r.context_item_refs[0].startswith("goal:")
        assert len(store.items(r.context_item_refs)) == len(r.context_item_refs)
        assert all(i.ts <= r.decision.ts for i in store.items(r.context_item_refs))   # no future information in H_t
        assert r.post_item_refs and r.goal_text
    pairs = [p for r in acted for p in matured_outcomes(r)]
    assert len(pairs) == len(acted) and all(o.verified for _, o in pairs)
    total = sum(o.value for _, o in pairs)
    assert total <= lt.verified.value + 1e-9 and lt.verified.signature and 0 <= verified_return(lt) <= 1


def test_family_worlds_differ_and_sequence_numbers_are_monotone():
    store = MemoryStore()
    lts = collect_family(store, 3, 8, seed=2)
    assert len({tuple(l.world._truth()["w"]) for l in lts}) == 3
    seqs = [r.seq for l in lts for r in l.records]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
