"""SP-12.0: learning agents found competing businesses, collaborate on software, trade, negotiate, join
organisations, develop competing technologies, learn from independent experiences, create services used by other
agents, form cooperative ventures, compete for scarce resources and respond to changing strategies, in a Tier 3
civilisation (Part 12 introduction).

The Gate Q campaign (`_selfplay_kit.campaign`, full size). Each capability is read from the world's state and event
log by `crucible.multiagent.analysis.capabilities`, counting only interactions in which a *learning* party (the
parties of the learning policy groups alpha, beta, gamma, delta) takes part; "learn from independent experiences"
from the registry (>= 2 groups updated on disjoint identities' experience); "respond to changing strategies" from a
counterfactual: the world and every participant are forked, the competitor group beta's policy is replaced by the
held-out family in one branch, and alpha's decisions must differ from the unchanged branch while two unchanged
branches agree exactly (`crucible.multiagent.campaign.strategy_switch`)."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import analysis as AN
from crucible.multiagent import campaign as C

from _selfplay_kit import campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.0"
LEARNING = ("alpha", "beta", "gamma", "delta")


def test_SP_12_0_selfplay_capabilities_in_a_civilisation(tmp_path):
    run = campaign(str(tmp_path), "full")
    m = run["match"]
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    ev = C.learning_evidence(run)
    inter = AN.interactions(m.world, ctrl, m.t0)
    sw = C.strategy_switch(m, responder="alpha", switched="beta", replacement=run["heldout"], hours=12, seed=7)
    gl = {g: {"changed_updates": ev[g]["updates_in_run"], "identities": ev[g]["identities"]} for g in LEARNING}
    caps = AN.capabilities(inter, ventures=AN.ventures_of(m.world, ctrl, m.t0), groups_learning=gl,
                           strategy_response=sw, parties={f"ctrl:{g}" for g in LEARNING})
    ok = sorted(k for k, v in caps.items() if v["ok"])
    record("SP-12.0", {"selfplay_capability_kinds": len(ok), "capabilities": json.dumps(ok),
                       "missing": json.dumps(sorted(set(AN.CAPABILITIES) - set(ok))),
                       "evidence": json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ok"} for k, v in
                                               caps.items()}, default=str)[:4000],
                       "strategy_switch_divergent_decisions": sw["switch_divergent_decisions"],
                       "strategy_replicate_divergent_decisions": sw["replicate_divergent_decisions"],
                       "simulated_hours": int((m.world.now - m.t0) / 3600)},
           command=CMD, labels={"learned": True, "scripted_demonstrations_for_pretraining": True},
           notes="capabilities exercised by learning parties in the civilisation (policies pre-trained by behaviour "
                 "cloning on scripted demonstrations in other worlds, then trained by self-play here)")
    m.close()
    assert sw["replicate_divergent_decisions"] == 0
    assert len(ok) >= 10, sorted(set(AN.CAPABILITIES) - set(ok))
