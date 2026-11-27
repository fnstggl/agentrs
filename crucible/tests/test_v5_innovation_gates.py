"""The Gate J/L and SP-6.3b/6.4b/6.5b measurement pipeline (`crucible.innovation.gates`) on a small ECON creation world
with the population's developers (`civexp` arms, kind "creation"): every measure computes from the arms, the
counterfactual arms are summarised and closed, nothing is scripted, and the integrity metrics hold. The gate tests run
the same pipeline on the integrated civilisation."""
import os

import pytest

import crucible

crucible.load_all_modules()
from crucible.innovation import civexp as CX  # noqa: E402
from crucible.innovation import gates as G  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


@pytest.fixture(scope="module")
def A():
    return G.arms("creation", 5, 8, "balanced", 300)


def test_counterfactual_arms_are_summarised_and_closed(A):
    assert list(A) == list(CX.ARMS)
    assert A["factual"]["world"] is not None and A["factual"]["world"].log_base == 0
    for arm in CX.ARMS[1:]:
        assert A[arm]["world"] is None
        assert set(A[arm]["summary"]) == {"relying", "bonuses", "rival_actions"}
        assert A[arm]["summary"]["bonuses"]["bonuses"] == 0
    # what the summary keeps equals what the open factual world gives
    w = A["factual"]["world"]
    for need, n in A["factual"]["summary"]["relying"].items():
        assert n == G.relying_on_creations(w, need)


def test_gate_measures_compute_and_integrity_holds(A):
    J = G.j_measure(A)
    assert J["J.1"]["agent_created_executable_products"] >= 1 and J["J.1"]["creators_scripted"] == 0
    assert J["J.1"]["creators_population"] == J["J.1"]["agent_created_executable_products"]
    L = G.l_measure(A)
    assert L["L.n1"]["runtime_bonuses"] == 0 and L["L.4"]["scripted_stages"] == 0
    s3 = G.sp63b_measure(A, os.path.join(os.path.dirname(G.__file__), ".."))
    assert s3["metrics"]["single_judge_usefulness_paths"] == 0
    assert s3["metrics"]["grounded_adoptions"] == s3["metrics"]["capability_adoptions"]
    s4 = G.sp64b_measure(A)
    assert s4["metrics"]["designer_supplied_demand_paths"] == 0
    assert set(s4["missing"]) | {k for k in G.LISTED if s4["found"][k]} == set(G.LISTED)
    w = A["factual"]["world"]
    assert G.sandbox_coverage(w)["agent_programs_outside_sandbox"] == 0
    assert not w.check_invariants()
