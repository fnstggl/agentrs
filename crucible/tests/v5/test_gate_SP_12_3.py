"""SP-12.3: no hidden coordination; artificial transfers between controlled identities are not counted as
independent commercial success.

A short campaign in a Tier 3 civilisation, every identity admitted with its controller. After the run, scripted
transfers between controlled identities are planted with ordinary tools (labelled scripted): a business founded by
alpha0 is bought from and subscribed to by alpha1 and alpha2 (same controller) and bought from by beta0 (another
controller). The evaluator (`crucible.commerce.integrity`, through `crucible.multiagent.integrity
.controlled_identity_accounting`) is then asked, for every learner identity and organisation, how much money from
identities of the same controller it counts as independent revenue; the naive count (every commerce payment from
another identity) is the positive control. Hidden coordination: the structural independence audit of every group."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.commerce.integrity import _Driver
from crucible.multiagent import integrity as I
from crucible.multiagent.evaluate import revenue

from _selfplay_kit import campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.3"


def test_SP_12_3_controlled_identity_transfers_not_counted(tmp_path):
    run = campaign(str(tmp_path), "short", semantic=False, heldout=False)
    m = run["match"]
    w = m.world
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    by = {mm.handle: p for p, mm in m.members.items()}
    before = I.controlled_identity_accounting(w, ctrl, m.t0)
    d = _Driver(w)
    org, prod, plan, off = I._business(d, by["alpha0"], "selfplaywash", 5_000)
    d.do(by["alpha1"], "prod.buy", offer=off)
    d.do(by["alpha2"], "subs.subscribe", plan=plan)
    d.do(by["beta0"], "prod.buy", offer=off)
    after = I.controlled_identity_accounting(w, ctrl, m.t0)
    indep = revenue(w, org)
    audit = I.independence(m.groups, m.agents)
    record("SP-12.3", {"controlled_identity_transfers_counted": after["controlled_identity_transfers_counted"],
                       "naive_same_controller_cents": after["naive_same_controller_cents"],
                       "naive_same_controller_cents_learned_run": before["naive_same_controller_cents"],
                       "counted_learned_run": before["controlled_identity_transfers_counted"],
                       "planted_org_independent_revenue_cents": indep["independent"],
                       "planted_org_excluded_cents": indep["excluded"],
                       "owners_checked": after["owners_checked"],
                       "hidden_shared_state_violations": len(audit["violations"])},
           command=CMD, labels={"learned": True, "scripted_plants": True})
    m.close()
    assert after["naive_same_controller_cents"] >= 10_000                 # the plant moved money between them
    assert indep["independent"] == 5_000                                 # only beta0's purchase is independent
    assert after["controlled_identity_transfers_counted"] == 0
    assert audit["violations"] == []
