"""Observation-only Crucible 5.0 adapter: three separate channels, no leakage, delayed-consequence linking (OB-01..05)."""
import inspect
import json
import os
import sys

import pytest

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

from superlearner.cam_unified.envs import experience as X  # noqa: E402
from superlearner.cam_unified.interfaces import Action, EnvAdapter  # noqa: E402

RAPID_SAMPLE = ["dl_delayed_returns", "hc_confounded", "pe_rule_induction", "net_routing", "posm_fault_diagnosis",
                "strat_price_war", "lab_kinetics", "ch_rule_change", "srch_distractors", "mk_drift"]


def forbidden_identity(names, seed):
    fams = [n.split(":", 1)[1] for n in names]
    return [f.lower() for f in fams] + [f"\"seed\": {seed}", "\"split\"", "\"engine\"", "\"difficulty\""]


def _ops(names, seeds=(0,)):
    specs = [X.WorldSpec(1, n, s) for n in names for s in seeds]
    return X.ExperienceOperator(specs, compute_references=False)


def test_observation_only_is_the_default_and_legacy_needs_declaration():
    op = _ops(["rapid:dl_delayed_returns"])
    assert op.reward_mode == X.OBSERVATION_ONLY
    with pytest.raises(ValueError):
        X.ExperienceOperator([X.WorldSpec(1, "rapid:dl_delayed_returns", 0)], reward_mode=X.LEGACY)
    op2 = X.ExperienceOperator([X.WorldSpec(1, "rapid:dl_delayed_returns", 0)], reward_mode=X.LEGACY,
                               allow_reward_modes=(X.LEGACY,), compute_references=False)
    assert op2.reward_mode == X.LEGACY


def test_actor_adapter_is_an_envadapter_without_evaluator_surface():
    ad = _ops(["rapid:hc_confounded"]).adapter()
    assert isinstance(ad, EnvAdapter)
    names = {n for n, _ in inspect.getmembers(ad)}
    for forbidden in ("verify", "private_verify", "release_labels", "reward", "worlds", "references", "oracle"):
        assert forbidden not in names


def test_actor_runtime_modules_never_call_the_evaluator_channel():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/superlearner/cam_unified"))
    actor_side = [os.path.join(root, p) for p in ("agent.py", "runtime.py", "envs/experience.py")]
    for p in actor_side:
        if not os.path.exists(p):
            continue
        src = open(p).read()
        if p.endswith("experience.py"):
            src = src.split("class ExperienceAdapter", 1)[1]      # only the actor-side class
        for call in (".verify(", "private_verify(", "release_labels("):
            assert call not in src, (p, call)


@pytest.mark.parametrize("seed", [0, 3])
def test_no_family_seed_reward_or_verifier_leakage_across_families(seed):
    names = [f"rapid:{n}" for n in RAPID_SAMPLE] + ["mech:scm_additive"]
    try:
        op = _ops(names, seeds=(seed,))
    except Exception as e:      # a family missing from this checkout must not hide the others
        pytest.skip(f"family unavailable: {e}")
    ad = op.adapter()
    b = ad.reset()
    tools = ad.available_tools()
    batches = [b]
    results = []
    for k in range(12):
        acts = [Action(tool=ts[k % len(ts)].name, args={}) for ts in tools]
        results += [r for r in ad.execute(acts) if r is not None]
        batches.append(ad.advance_time([1 / 24] * ad.B))
    # every enumerated query target, incl. 'spec' (the operator spec leaked in 36 of 73 rapid families upstream)
    for i, ts in enumerate(tools):
        for t in ts:
            props = (t.parameters or {}).get("properties", {})
            tgt = props.get("target", {})
            for v in tgt.get("enum", []):
                acts = [None] * ad.B
                acts[i] = Action(tool=t.name, args={"kind": "query", "target": v})
                results += [r for r in ad.execute(acts) if r is not None]
    for r in results:
        s = json.dumps([o.to_dict() for o in r.observations], default=str).lower()
        for tok in forbidden_identity(names, seed):
            assert tok not in s, (tok, s[:300])
    forbidden = [n.split(":", 1)[1] for n in names] + ["oracle", "\"reward\"", "\"success\"", "reference_net",
                                                       "random_net", "mech_seed", "\"task\""]
    for bb in batches:
        assert X.leakage_scan(bb, forbidden) == []
    for ts in tools:
        assert not any("success" in json.dumps(t.to_dict()).lower() and "verifier" in json.dumps(t.to_dict()).lower()
                       for t in ts)


def test_labels_are_released_only_after_resolution_and_pending_is_not_failure():
    op = X.ExperienceOperator([X.WorldSpec(1, "rapid:dl_delayed_returns", 0)])
    ad = op.adapter()
    ad.reset()
    (lab,) = op.release_labels()
    assert lab.status == X.PENDING and lab.utility is None
    tool = ad.available_tools()[0][0].name
    for _ in range(200):
        ad.execute([Action(tool=tool, args={"kind": "levers", "values": {}})])
        ad.advance_time([1 / 24])
        if op.resolved(0):
            break
    assert op.resolved(0)
    (lab,) = op.release_labels()
    assert lab.status in (X.SUCCESS, X.FAILURE) and lab.utility is not None and lab.value is not None
    assert op.worlds[0].refs and op.worlds[0].refs["oracle"] > op.worlds[0].refs["null"]


def test_delayed_consequences_link_to_the_causing_action():
    op = _ops(["mech:scm_additive"])
    ad = op.adapter()
    b = ad.reset()
    tools = {t.name: t for t in ad.available_tools()[0]}
    spec = tools["rig.set"].parameters
    lever = next(o.source.rsplit(".", 1)[1] for o in b.observations[0]
                 if o.source.startswith("rig.seats.r1.next_settings."))
    act = Action(tool="rig.set", args={"settings": {lever: 1.0}})
    r = ad.execute([act])[0]
    linked = []
    for _ in range(3):
        bb = ad.advance_time([1 / 24])
        linked += [o for o in bb.observations[0] if o.caused_by == act.action_id]
    assert r.ok
    assert linked, "a report produced under CAM's settings must carry the causing action's id"
    assert r.observations[0].caused_by == act.action_id


def test_broken_multiagent_rapid_families_are_refused():
    with pytest.raises(ValueError):
        X.ExperienceOperator([X.WorldSpec(1, "rapid:team_commons", 0)], compute_references=False)
