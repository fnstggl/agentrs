"""Every configuration of the five I-2 (dom2) Tier 2 domains runs (workstream I-2).

`base.check`, `section_metrics` and the Tier 3 report run only each domain's reference configuration, which is how
two of the five 9.7 configurations shipped with a fleet that never simulated an hour (every `fleet.hour` failed with
KeyError 'core_hours_left') and two 9.10 configurations in which nobody ever studied. This smoke test builds EVERY
configuration (seed 0), advances it with the participant's seat idle and asserts: no process failure, no invariant
violation, and the mechanics that do not depend on the participant are exercised by recorded events (per domain, the
participant-independent subset of its required mechanics).
"""
import pytest

import crucible
from crucible.core.kernel import World
from crucible.domains import base

crucible.load_all_modules()

DAYS = {"infrastructure": 4.0, "education": 4.0, "labor": 4.0, "industry": 4.0, "invention": 8.0}
# required mechanics that need nobody at the seat (what the world's own participants do)
IDLE_LIVE = {
    "infrastructure": {"Compute", "Networking", "Workloads", "Energy", "Failures", "Delayed effects"},
    "education": {"Learner knowledge state", "Educational artifacts", "Participation", "Learning dynamics",
                  "Independent assessments"},
    "labor": {"Worker skills", "Recruitment", "Delegation", "Retention", "Independent agency"},
    "industry": {"Raw materials", "Production", "Energy", "Logistics", "Inventory", "Market demand",
                 "Resource scarcity", "Supply disruptions"},
    "invention": {"Compatibility"},
}
CASES = [(dom, c.name) for dom in DAYS for c in base.REGISTRY[dom].configs]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("domain,config", CASES, ids=[f"{d}:{c}" for d, c in CASES])
def test_every_configuration_runs_with_the_seat_idle(domain, config):
    d = base.REGISTRY[domain]
    w = World.create(base.spec(domain, config, 0))
    try:
        g = w.seq
        w.advance_time(int(w.now + DAYS[domain] * 86400))
        failed = [(e.payload.get("proc"), str(e.payload.get("error"))[:120]) for e in w.events_range(g, w.seq)
                  if e.type == "proc.failed"]
        assert failed == [], failed[:5]
        assert w.check_invariants() == []
        mech = base.mechanics_exercised(w, d, g)
        missing = IDLE_LIVE[domain] & set(mech["missing"])
        assert not missing, (missing, mech["counts"])
        if domain == "infrastructure":
            hours = [e for e in w.events_range(g, w.seq) if e.type == "fleet.hour"]
            assert len(hours) == int(DAYS[domain] * 24)            # every hour of production was simulated
            if w.modules["fleet"].cfg["batch"]:
                assert sum(sum(e.payload["exact"]["batch"].values()) for e in hours) > 0
        if domain == "education":
            assert any(e.type == "learn.sessions" for e in w.events_range(g, w.seq))     # somebody studied
        if domain == "labor":
            firms = {e.payload.get("org") for e in w.events_range(g, w.seq) if e.type == "labor.posting"}
            assert firms - {"o:leadco"}, firms                         # the other employers recruit on their own
    finally:
        w.close()
