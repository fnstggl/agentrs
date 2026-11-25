"""SP-6.3 - general artifact-and-capability registry: a participant discovers a previously nonexistent service,
tests it and adopts it if it solves a genuine problem (workstream G).

World `crucible.innovation.scenarios.market` (seed 11, broken=True, tester=True): two agents create services for the
same public problem format after the world began - one correct, one that returns invalid routes. 24 need-driven
participants find them through caps.search, execute each on inputs they generate themselves (caps.try through the
real cloud/sandbox path), score the outputs with their own problem's verifier and adopt only what passes their
threshold. An independent tester runs its own suite (caps.verify, fresh seeds) against both declarations.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


@pytest.fixture(scope="module")
def market():
    r = SC.market(seed=11, participants=24, days=4.0, broken=True, tester=True, contrast=False)
    yield r
    SC.close(r)


def test_SP_6_3_novel_service_discovered_tested_and_adopted_if_useful(market):
    w = market["world"]
    st = w.state
    first_update = M.first_update_t(w)
    rows = {}
    for p in M.agent_products(w):
        if p["kind"] != "service":
            continue
        decls = set(p["declarations"])
        disc = set(M.discoverers(w, p))
        tested = {t["by"] for _, t in st.scan("cap_trials") if t["decl"] in decls and t["independent"]}
        adopted = {a["adopter"] for _, a in st.scan("cap_adoptions") if a["decl"] in decls and a["independent"]}
        used = {a["adopter"] for _, a in st.scan("cap_adoptions") if a["decl"] in decls and a["uses_ok"] > 0}
        full = disc & tested & adopted & used
        scores = [d["mean"] for _, d in st.scan("adopt_decisions") if d["decl"] in decls]
        rows[p["host"]] = {"novel": p["created_t"] > 0, "discovered": len(disc), "tested": len(tested),
                           "adopted": len(adopted), "used_after_adoption": len(used), "full_path": len(full),
                           "participant_mean_test_score": round(sum(scores) / len(scores), 4) if scores else None,
                           "independent_verification": (market.get("verifications") or {}).get(p["host"], {}).get("mean_score")}
    good = [h for h, r in rows.items() if r["novel"] and r["full_path"] >= 1]
    broken_adopters = rows.get("malroute", {}).get("adopted", -1)
    m = {"novel_services_discovered_tested_adopted": len(good),
         "participants_full_path_best": max((r["full_path"] for r in rows.values()), default=0),
         "useless_service_adopters": broken_adopters,
         "useless_service_testers": rows.get("malroute", {}).get("tested", 0),
         "independent_verification_good": rows.get("anaroute", {}).get("independent_verification") or 0.0,
         "independent_verification_useless": rows.get("malroute", {}).get("independent_verification") or 0.0,
         "needs_first_update_t": first_update or 0}
    evidence.record("SP-6.3", m, notes="Services created by scripted reference agents after genesis; participants "
                                       "are rule-based need-driven adopters; usefulness is measured objectively by "
                                       "executing the service on independently generated tests (each participant's "
                                       "own inputs scored by its own verifier, and an independent tester's suite via "
                                       f"caps.verify). Per service: {rows}")
    assert m["novel_services_discovered_tested_adopted"] >= 1
    assert broken_adopters == 0 and m["useless_service_testers"] >= 1          # tested, found useless, not adopted
    assert m["independent_verification_good"] > m["independent_verification_useless"]
