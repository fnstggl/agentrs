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
branches agree exactly (`crucible.multiagent.campaign.strategy_switch`).

Where the capabilities come from: the same civilisation is also played by the policies frozen as they were after
behaviour-cloning pre-training (`_selfplay_kit.frozen_control`); the world-interaction capabilities of both runs are
reported side by side, with those the learning run added (the criterion is on the learning run)."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import analysis as AN
from crucible.multiagent import campaign as C

from _selfplay_kit import campaign, frozen_control

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.0"
LEARNING = ("alpha", "beta", "gamma", "delta")


WORLD_KINDS = tuple(k for k in AN.CAPABILITIES if k not in ("learn_from_independent_experiences",
                                                            "respond_to_changing_strategies"))


def _caps(m, groups_learning, strategy_response):
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    inter = AN.interactions(m.world, ctrl, m.t0)
    return AN.capabilities(inter, ventures=AN.ventures_of(m.world, ctrl, m.t0), groups_learning=groups_learning,
                           strategy_response=strategy_response, parties={f"ctrl:{g}" for g in LEARNING})


def test_SP_12_0_selfplay_capabilities_in_a_civilisation(tmp_path):
    run = campaign(str(tmp_path), "full")
    m = run["match"]
    ev = C.learning_evidence(run)
    sw = C.strategy_switch(m, responder="alpha", switched="beta", replacement=run["heldout"], hours=12, seed=7)
    gl = {g: {"changed_updates": ev[g]["updates_in_run"], "identities": ev[g]["identities"]} for g in LEARNING}
    caps = _caps(m, gl, sw)
    ok = sorted(k for k, v in caps.items() if v["ok"])
    hours = int((m.world.now - m.t0) / 3600)
    m.close()
    # the control: the same civilisation played by the frozen pre-trained policies (no self-play learning)
    fz = frozen_control(run)
    fm = fz["match"]
    fcaps = _caps(fm, {g: {"changed_updates": 0, "identities": list(fz["groups"][g].identities)} for g in LEARNING},
                  None)
    fm.close()
    world_ok = sorted(k for k in WORLD_KINDS if caps[k]["ok"])
    frozen_ok = sorted(k for k in WORLD_KINDS if fcaps[k]["ok"])
    added, lost = sorted(set(world_ok) - set(frozen_ok)), sorted(set(frozen_ok) - set(world_ok))
    record("SP-12.0", {"selfplay_capability_kinds": len(ok), "capabilities": json.dumps(ok),
                       "missing": json.dumps(sorted(set(AN.CAPABILITIES) - set(ok))),
                       "evidence": json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ok"} for k, v in
                                               caps.items()}, default=str)[:4000],
                       "world_capability_kinds_learning_run": len(world_ok),
                       "world_capability_kinds_frozen_pretrained": len(frozen_ok),
                       "capabilities_frozen_pretrained": json.dumps(frozen_ok),
                       "capabilities_added_by_selfplay_learning": json.dumps(added),
                       "capabilities_lost_with_selfplay_learning": json.dumps(lost),
                       "capability_source": "self-play learning added some" if added else
                       "behaviour cloning of scripted demonstrations (learning added none)",
                       "frozen_control_evidence": json.dumps({k: {kk: vv for kk, vv in fcaps[k].items() if kk != "ok"}
                                                              for k in WORLD_KINDS}, default=str)[:3000],
                       "strategy_switch_divergent_decisions": sw["switch_divergent_decisions"],
                       "strategy_replicate_divergent_decisions": sw["replicate_divergent_decisions"],
                       "strategy_switch_host_tv": sw["host_tv_switch_vs_control"],
                       "simulated_hours": hours},
           command=CMD, labels=dict(C.EVIDENCE_LABELS, capabilities_from_selfplay_learning=bool(added)),
           notes="capabilities exercised by learning parties in the civilisation: policies pre-trained by behaviour "
                 "cloning on scripted demonstrations in other worlds, then trained by self-play here. The frozen "
                 "control plays the same civilisation with the pre-trained policies and no learning; "
                 "capability_source says whether learning added any world-interaction capability to what "
                 "pre-training brought. create_services_used_by_others counts only what a learning party provides "
                 "(it is the paid seller, the called service's owner or the depended-on package's owner); "
                 "compete_for_scarce_resources only binding scarcity (a compute request refused for capacity another "
                 "party held)")
    assert sw["replicate_divergent_decisions"] == 0
    assert len(ok) >= 10, sorted(set(AN.CAPABILITIES) - set(ok))
