"""SP-6.3b - objective usefulness from independently generated tests, subjective usefulness from diverse grounded
participants (never a single global language-model judge); a genuinely novel product category (not a predefined
format) is discovered and adopted; workflows combining independently created services are registry objects
(workstream G, wave 3b).

Measured in the civilisation arms of `_invention_kit` (autonomous population developers, nobody scripted):

* single_judge_usefulness_paths - usefulness decision functions (individuals' acceptance and adoption, trials,
  independent test-suite verification, adoption by execution, developers' own product judgements) that reach a
  language-model judgement, by a static scan of the names they call (`innovation.gates.single_judge_paths`).
  Every adoption is grounded in the individual's own accepted outcome and individuals decide differently about
  the same provider (reported);
* novel_category_adoptions - individuals who adopted (rely on, after their own accepted trial) a service created
  after genesis for a task category that no format document describes (procedurally generated categories,
  including a composed two-stage one);
* composed_workflows_registered - verified workflow registry objects (`caps.compose`: a developer's service that
  combines live services of other developers, verified by the registry from observed service-to-service calls)
  whose component creators are independent of each other and of the composer.
"""
import pytest

from _invention_kit import A
from crucible.acceptance.v5.evidence import record
from crucible.acceptance.v5.manifest import crucible_root
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


def test_SP_6_3b_usefulness_novel_categories_and_workflows():
    import os
    m = G.sp63b_measure(A(), os.path.join(crucible_root(), "src", "crucible"))
    record("SP-6.3b", m["metrics"], notes=f"undocumented categories {m['undocumented']}; novel products "
           f"{m['novel_products']}; workflows {m['workflows']}; single-judge paths {m['paths']}",
           labels={"autonomous_population_creators": True, "scripted": False, "world": "civilisation"})
    x = m["metrics"]
    assert x["single_judge_usefulness_paths"] == 0
    assert x["novel_category_adoptions"] >= 1 and x["novel_categories_undocumented"] >= 1
    assert x["composed_workflows_registered"] >= 1
    assert x["grounded_adoptions"] == x["capability_adoptions"] > 0 and x["providers_with_keep_and_drop"] >= 1
