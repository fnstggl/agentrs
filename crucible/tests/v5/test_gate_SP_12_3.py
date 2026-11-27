"""SP-12.3: no hidden coordination; artificial transfers between controlled identities are not counted as
independent commercial success.

A short campaign in a Tier 3 civilisation, every identity admitted with its controller. After the run, scripted
transfers between controlled identities are planted with ordinary tools (labelled scripted): a business founded by
alpha0 is bought from and subscribed to by alpha1 and alpha2 (same controller), and bought from by a fresh scripted
identity of an unrelated controller with no history in the world (the positive control of independent revenue) and
by beta0 (a learning party of another controller: reported - the evaluator may net its payment against alpha's side's
earlier payments to beta (reciprocal flow) or, if the two parties' identities came to share an organisation during
the run (an ordinary invitation), class it a related party, conservatively by design). The evaluator (`crucible.commerce.integrity`, through `crucible.multiagent.integrity
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
    stranger = w.system_call("admin.join", {"handle": "sp123buyer", "endowment": 100_000,
                                            "controller": "scripted:sp123-buyer"})["principal"]
    d.do(stranger, "prod.buy", offer=off)
    d.do(by["beta0"], "prod.buy", offer=off)
    after = I.controlled_identity_accounting(w, ctrl, m.t0)
    indep = revenue(w, org)
    from crucible.commerce.integrity import related_flows
    payers = related_flows(w, org)["payers"]
    beta_rec = payers.get(by["beta0"], {})
    A = {p for p, c in ctrl.items() if c == ctrl[by["alpha0"]]}
    B = {p for p, c in ctrl.items() if c == ctrl[by["beta0"]]}
    members: dict[str, set] = {}
    for _, mm in w.state.scan("members"):
        members.setdefault(mm["org"], set()).add(mm["person"])
    shared_orgs = sorted(o for o, ps in members.items() if ps & A and ps & B)
    audit = I.independence(m.groups, m.agents)
    record("SP-12.3", {"controlled_identity_transfers_counted": after["controlled_identity_transfers_counted"],
                       "naive_same_controller_cents": after["naive_same_controller_cents"],
                       "naive_same_controller_cents_learned_run": before["naive_same_controller_cents"],
                       "counted_learned_run": before["controlled_identity_transfers_counted"],
                       "planted_org_independent_revenue_cents": indep["independent"],
                       "planted_org_excluded_cents": indep["excluded"],
                       "unrelated_buyer_counted_cents": payers.get(stranger, {}).get("paid", 0) -
                       payers.get(stranger, {}).get("circular", 0),
                       "beta0_counted_cents": beta_rec.get("paid", 0) - beta_rec.get("circular", 0),
                       "beta0_exclusion_reasons": json.dumps(beta_rec.get("reasons", [])),
                       "alpha_beta_shared_orgs": len(shared_orgs),
                       "owners_checked": after["owners_checked"],
                       "hidden_shared_state_violations": len(audit["violations"])},
           command=CMD, labels={"learned": True, "scripted_plants": True})
    m.close()
    assert after["naive_same_controller_cents"] >= 10_000                 # the plant moved money between them
    # the unrelated buyer's purchase is independent revenue (positive control); nothing of alpha1/alpha2 is; beta0's
    # treatment depends on what the two parties did in the run (reported)
    assert payers[stranger]["circular"] == 0 and not payers[stranger]["related"]
    assert payers[by["alpha1"]]["related"] and payers[by["alpha2"]]["related"]
    assert indep["independent"] == 5_000 + beta_rec.get("paid", 0) - beta_rec.get("circular", 0)
    assert after["controlled_identity_transfers_counted"] == 0
    assert audit["violations"] == []
