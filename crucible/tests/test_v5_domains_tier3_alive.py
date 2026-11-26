"""Education and employment are alive in an integrated civilisation (INT.json: both were inert in every profile).

The civilisation generator composes `learn` (education) and `firms`/`labor`/`inst` (employment) with the one
population; the population's own `study`, `teach` and `job_search` activities (registered by domains.learning and
domains.firms) make them work without any agent: individuals take placement tests and study documents, some write
lessons, the Board assesses, individuals enter the labour market, apply, are offered jobs, and employed individuals do
measured work. Slow (the population's text path runs on every lesson): about 3 CPU minutes.
"""
import re
from collections import Counter

import pytest

import crucible
from crucible import civ
from crucible.domains import base

crucible.load_all_modules()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_education_and_employment_are_alive_in_a_generated_civilisation():
    w = civ.generate("balanced", 1)
    since = w.seq
    w.advance_time(w.now + 5 * 86400)
    ev = Counter(e.type for e in w.events_range(since, w.seq))
    kinds = base.population_decisions(w, since)["activity_kinds"]
    st = w.state
    assert kinds.get("learn.study", 0) > 10 and kinds.get("learn.placed", 0) > 50 and kinds.get("learn.teach", 0) > 5
    assert ev["learn.sessions"] > 10 and ev["learn.assessed"] > 50 and ev["learn.authored"] > 5
    assert kinds.get("firms.jobsearch", 0) > 30 and ev["labor.application"] > 20 and ev["labor.offer"] > 0
    pop_work = [r for _, r in st.scan("labor_perf") if re.fullmatch(r"p\d+", r["worker"])]
    assert pop_work                                      # population individuals employed by firms did measured work
    assert ev.get("proc.failed", 0) == 0 and w.check_invariants() == []
    w.close()
