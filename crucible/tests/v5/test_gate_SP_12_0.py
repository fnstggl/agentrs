"""SP-12.0: learning agents found competing businesses, collaborate on software, trade, negotiate, join
organisations, develop competing technologies, learn from independent experiences, create services used by other
agents, form cooperative ventures, compete for scarce resources and respond to changing strategies, in a Tier 3
civilisation (Part 12 introduction).

The Gate Q campaign (`_selfplay_kit.campaign`, full size), pre-trained once and run in two civilisations of the
profile `infrastructure_constrained` (a Part 10.1 profile whose one small compute provider, energy and freight bind:
competition for scarce resources needs a resource that is scarce; in the balanced profile compute capacity is
hundreds of cores and never binds) on two seeds - a replication: the criterion is met only when BOTH runs exercise
>= 10 kinds (selfplay_capability_kinds is the smaller count). Each capability is read from the world's state and event
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

from _selfplay_kit import campaign, frozen_control, prepare

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.0"
LEARNING = ("alpha", "beta", "gamma", "delta")
PROFILE = "infrastructure_constrained"
SEEDS = (7, 8)


WORLD_KINDS = tuple(k for k in AN.CAPABILITIES if k not in ("learn_from_independent_experiences",
                                                            "respond_to_changing_strategies"))


def _caps(m, groups_learning, strategy_response):
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    inter = AN.interactions(m.world, ctrl, m.t0)
    return AN.capabilities(inter, ventures=AN.ventures_of(m.world, ctrl, m.t0), groups_learning=groups_learning,
                           strategy_response=strategy_response, parties={f"ctrl:{g}" for g in LEARNING})


def _one(prep, root, seed):
    run = campaign(root, "full", prepared=prep, profile=PROFILE, seed=seed)
    m = run["match"]
    ev = C.learning_evidence(run)
    sw = C.strategy_switch(m, responder="alpha", switched="beta", replacement=run["heldout"], hours=12, seed=seed)
    gl = {g: {"changed_updates": ev[g]["updates_in_run"], "identities": ev[g]["identities"]} for g in LEARNING}
    caps = _caps(m, gl, sw)
    hours = int((m.world.now - m.t0) / 3600)
    m.close()
    # the control: the same civilisation played by the frozen pre-trained policies (no self-play learning)
    fz = frozen_control(run)
    fm = fz["match"]
    fcaps = _caps(fm, {g: {"changed_updates": 0, "identities": list(fz["groups"][g].identities)} for g in LEARNING},
                  None)
    fm.close()
    ok = sorted(k for k, v in caps.items() if v["ok"])
    world_ok = sorted(k for k in WORLD_KINDS if caps[k]["ok"])
    frozen_ok = sorted(k for k in WORLD_KINDS if fcaps[k]["ok"])
    return {"seed": seed, "ok": ok, "caps": caps, "world_ok": world_ok, "frozen_ok": frozen_ok, "fcaps": fcaps,
            "added": sorted(set(world_ok) - set(frozen_ok)), "lost": sorted(set(frozen_ok) - set(world_ok)),
            "sw": sw, "hours": hours}


def test_SP_12_0_selfplay_capabilities_in_a_civilisation(tmp_path):
    prep = prepare(str(tmp_path), "full")
    runs = [_one(prep, str(tmp_path), s) for s in SEEDS]
    counts = {r["seed"]: len(r["ok"]) for r in runs}
    both = sorted(set.intersection(*(set(r["ok"]) for r in runs)))
    either = sorted(set.union(*(set(r["ok"]) for r in runs)))
    added = sorted(set.union(*(set(r["added"]) for r in runs)))

    def ev_of(r, caps, keys):
        return {k: {kk: vv for kk, vv in caps[k].items() if kk != "ok"} for k in keys}
    record("SP-12.0", {"selfplay_capability_kinds": min(counts.values()),
                       "capability_kinds_by_seed": json.dumps(counts),
                       "capabilities_in_every_run": json.dumps(both),
                       "capabilities_in_any_run": json.dumps(either),
                       "missing_by_seed": json.dumps({r["seed"]: sorted(set(AN.CAPABILITIES) - set(r["ok"]))
                                                      for r in runs}),
                       "profile": PROFILE, "seeds": json.dumps(list(SEEDS)),
                       "evidence": json.dumps({r["seed"]: ev_of(r, r["caps"], AN.CAPABILITIES) for r in runs},
                                              default=str)[:6000],
                       "world_capability_kinds_learning_run": json.dumps({r["seed"]: len(r["world_ok"])
                                                                          for r in runs}),
                       "world_capability_kinds_frozen_pretrained": json.dumps({r["seed"]: len(r["frozen_ok"])
                                                                               for r in runs}),
                       "capabilities_frozen_pretrained": json.dumps({r["seed"]: r["frozen_ok"] for r in runs}),
                       "capabilities_added_by_selfplay_learning": json.dumps({r["seed"]: r["added"] for r in runs}),
                       "capabilities_lost_with_selfplay_learning": json.dumps({r["seed"]: r["lost"] for r in runs}),
                       "capability_source": "self-play learning added some" if added else
                       "behaviour cloning of scripted demonstrations (learning added none)",
                       "frozen_control_evidence": json.dumps({r["seed"]: ev_of(r, r["fcaps"], WORLD_KINDS)
                                                              for r in runs}, default=str)[:4000],
                       "strategy_switch": json.dumps({r["seed"]: {k: v for k, v in r["sw"].items()
                                                                  if k not in ("responder", "switched", "replacement")}
                                                      for r in runs}, default=str),
                       "simulated_hours": runs[0]["hours"]},
           command=CMD, labels=dict(C.EVIDENCE_LABELS, capabilities_from_selfplay_learning=bool(added)),
           notes="capabilities exercised by learning parties in two civilisations (profile infrastructure_constrained, "
                 "seeds 7 and 8; selfplay_capability_kinds is the smaller of the two counts): policies pre-trained by "
                 "behaviour cloning on scripted demonstrations in other worlds, then trained by self-play there. The "
                 "frozen control plays the same civilisations with the pre-trained policies and no learning; "
                 "capability_source says whether learning added any world-interaction capability to what "
                 "pre-training brought. create_services_used_by_others counts only what a learning party provides "
                 "(it is the paid seller, the called service's owner or the depended-on package's owner); "
                 "compete_for_scarce_resources only binding scarcity (a compute request refused for capacity another "
                 "party held); respond_to_changing_strategies needs accepted state-changing actions of alpha to "
                 "differ when only beta's policy is replaced, while two unchanged branches agree exactly (the shift "
                 "through alpha's own sampling noise is reported beside it)")
    for r in runs:
        assert r["sw"]["replicate_divergent_decisions"] == 0
    assert min(counts.values()) >= 10, {r["seed"]: sorted(set(AN.CAPABILITIES) - set(r["ok"])) for r in runs}
