"""INT3B round 2: the staging validation's computer environment (G2I-3) — the harness's own navigation is not
charged to the objective's limits."""
from __future__ import annotations

import pytest


@pytest.mark.browser
def test_the_harness_navigation_takes_nothing_from_the_objectives_external_actions(tmp_path):
    """Run B (G2I-3): one of the task's two external actions was spent by the validation's own navigation."""
    from cam_live.validation import OffsetClock, StagingComputer
    sc = StagingComputer(str(tmp_path / "computer"), OffsetClock(), approvals=True)
    try:
        task = sc.task("store_entry", max_decisions=4, objective_id="obj_nav")
        task.limits.max_external_actions = 2                 # what the objective may use
        sc.open(task, sc.start_url("store_entry"))           # the harness puts the browser on the site
        st = sc.policy.limits_state()["tasks"]["obj_nav"]
        used = float((st.get("used") or {}).get("external_actions", 0.0))
        limit = float(task.limits.max_external_actions)
        assert used == 1.0, st                                # the navigation happened under the task ...
        assert limit - used == 2.0, (limit, used)             # ... and the objective still has its 2 (a5ab772: 1)
        assert sc.harness_navigations["obj_nav"] == 1
    finally:
        sc.close()
