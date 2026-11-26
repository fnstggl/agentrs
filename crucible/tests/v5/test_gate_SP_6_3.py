"""SP-6.3 - general artifact-and-capability registry: a participant discovers a previously nonexistent service,
tests it and adopts it if it solves a genuine problem (workstream F+G; D-18).

Measured in the creation arms of `_creation_kit` (the one population; LABELLED SCRIPTED creators). Individuals of
the population work on a procedurally generated task category (`usefulness` family `allocate`, a hidden rule drawn
for this world) that no format registry documents and no firm serves at genesis. They do it by hand until a
service appears that (a) their own search finds - the capability declaration or the creator's reply to their
public request - (b) whose *declared input schema accepts their own input* (interface match), and which (c) on a
trial on their own input produces work their employer accepts (world physics; they never see the verifier).
Only then do they adopt it and keep paying per call. In the arm where the same creator ships an input-ignoring
program with the same interface, the same individuals discover and try it and do not adopt it.

Metric: novel_services_discovered_tested_adopted - novel-category services (created after genesis, for a task not
in the predefined published formats) that independent individuals discovered, tested on their own input and
adopted, with adoption higher than for the input-ignoring counterpart.
"""
import pytest

from _creation_kit import arms
from crucible.acceptance.v5.evidence import record
from crucible.innovation import measure as M

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def test_SP_6_3_novel_service_discovered_tested_and_adopted_if_useful():
    A = arms()
    f, g = A["factual"], A["garbage"]
    w = f["world"]
    fams = w.modules["work"].fams
    novel_needs = {x["need"] for x in fams if x["need"].startswith("gen.")}
    st = w.state
    docs = [d["title"] for _, d in st.scan("documents")] if "documents" in st.tables else []
    undocumented = all(not any(n in t for t in docs) for n in novel_needs)
    rows = {}
    for p in M.agent_products(w):
        provs = M._provider_ids(w, p)
        relied_needs = {r["step_need"] for _, r in st.scan("work_exp") if r["prov"] in provs}
        if not relied_needs & novel_needs:
            continue
        o = f["outcome"]["products"][p["id"]]
        og = g["outcome"]["products"].get(p["id"], {"triers": [], "kept": [], "discoverers": 0})
        rows[p["id"]] = {"created_day": p["created_t"] // 86400, "discoverers": o["discoverers"],
                         "tested": len(o["triers"]), "adopted": len(o["kept"]), "adopters_ever": o["adopters"],
                         "garbage_discoverers": og["discoverers"], "garbage_tested": len(og["triers"]),
                         "garbage_adopted": len(og["kept"]), "creator_kind": p["creator_kind"]}
    ok = [pid for pid, r in rows.items() if r["discoverers"] >= 1 and r["tested"] >= 1 and r["adopted"] >= 1
          and r["adopted"] / max(1, r["tested"]) > r["garbage_adopted"] / max(1, r["garbage_tested"])]
    record("SP-6.3", {"novel_services_discovered_tested_adopted": len(ok), "novel_task_categories": len(novel_needs),
                      "novel_categories_undocumented": bool(undocumented), "creators_scripted": True},
           notes=f"novel categories {sorted(novel_needs)}; per service {rows}. Not covered here: subjective "
                 "usefulness from diverse grounded responses (the semantic participants of CORE-SEM/CORE-POP).")
    assert ok and undocumented
