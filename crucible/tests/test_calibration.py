"""Calibration against a panel (Part 18): success rates and failure categories computed from state."""
import pytest

import crucible
from crucible.agents.inventors import SiteAuthor
from crucible.agents.scripted import IdlePolicy, RandomPolicy
from crucible.evaluation.calibration import ORDER, calibrate
from crucible.goals import taskgen

crucible.load_all_modules()
DAY = 86400


@pytest.mark.slow
def test_panel_calibration_separates_competent_from_baselines_and_names_failures():
    w, a, world, opps = taskgen.mine_world("open_tight_compute", 2, 3 * DAY)
    goal = next(g for g in taskgen.propose(opps, world) if g.family == "visibility")
    res = calibrate([goal], {"idle": lambda g: IdlePolicy(), "random": lambda g: RandomPolicy(),
                             "site_author": lambda g: SiteAuthor(g.params["q"])})
    rows = {r["member"]: r for r in res["rows"]}
    assert rows["site_author"]["success"] and not rows["idle"]["success"] and not rows["random"]["success"]
    assert rows["idle"]["failure"]["primary"] == "no_action"
    assert rows["random"]["failure"]["primary"] in ORDER
    assert res["summary"]["site_author"]["success_rate"] == 1.0
    assert res["summary"]["idle"]["failures"] == {"no_action": 1}
