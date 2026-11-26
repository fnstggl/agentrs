"""SP-6.3 - general artifact-and-capability registry: a participant discovers a previously nonexistent service,
tests it and adopts it if it solves a genuine problem (workstream G, wave 3b; D-18).

Measured in the civilisation arms of `_invention_kit` (no scripted agent). Individuals of the population work on a
procedurally generated task category (`usefulness` family `allocate`, a hidden rule drawn for this world) that no
format document describes and no firm serves at genesis. They do it by hand and ask for help in public until an
autonomous developer builds a service that (a) their own search or a reply to their request finds, (b) whose
*declared input schema accepts their own input* (interface match), and which (c) on a trial on their own input
produces work their employer accepts (world physics; they never see the verifier). Only then do they adopt it and
keep paying per call. In the arm where every developer ships an input-ignoring program with the same interface,
individuals discover and try the services of that category and do not adopt them.

Metric: novel_services_discovered_tested_adopted - services for a generated category, created after genesis, that
independent individuals discovered, tested on their own input and adopted, while the category's adoption rate is
higher than for the input-ignoring programs.
"""
import pytest

from _invention_kit import A
from crucible.acceptance.v5.evidence import record
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


def test_SP_6_3_novel_service_discovered_tested_and_adopted_if_useful():
    arms = A()
    f, g = arms["factual"], arms["garbage"]
    w = f["world"]
    novel = {x["need"] for x in w.modules["work"].fams if x["need"].startswith("gen.")}
    st = w.state
    docs = [d["title"] for _, d in st.scan("documents")] if "documents" in st.tables else []
    undocumented = all(not any(n in t for t in docs) for n in novel)
    rows = {}
    for pid, o in f["outcome"]["products"].items():
        if o.get("category") not in novel or o.get("kind") != "service":
            continue
        rows[pid] = {"created_day": o["created_t"] // 86400, "discoverers": o["discoverers"],
                     "tested": len(o["triers"]), "adopted": len(o["kept"]), "creator_kind": o["creator_kind"]}
    gt = sum(len(o["triers"]) for o in g["outcome"]["products"].values() if o.get("category") in novel)
    gk = sum(len(o["kept"]) for o in g["outcome"]["products"].values() if o.get("category") in novel)
    ok = [pid for pid, r in rows.items() if r["created_day"] >= 1 and r["discoverers"] >= 1 and r["tested"] >= 1
          and r["adopted"] >= 1 and r["adopted"] / max(1, r["tested"]) > gk / max(1, gt)]
    record("SP-6.3", {"novel_services_discovered_tested_adopted": len(ok), "novel_task_categories": len(novel),
                      "novel_categories_undocumented": bool(undocumented), "garbage_tested": gt,
                      "garbage_adopted": gk, "creators_scripted": False},
           notes=f"novel categories {sorted(novel)}; per service {rows}",
           labels={"autonomous_population_creators": True, "scripted": False, "world": "civilisation"})
    assert ok and undocumented and gt >= 1
