"""Adaptive execution-modality selection (spec 12A.7; requirement CU-31)."""
import json
import os
import random

import pytest

from cam_live.contracts import (ActionResult, ActionSpec, CandidateEvaluation, Cost, FailureClass, Modality,
                                Observation, ResultStatus, SimClock)
from cam_live.harness.modality import (DEFAULT_PRIORS, KINDS, ExecContextFeatures, ModalitySelector, Realisation,
                                       default_kind, intent_key)

SPEC_ORDER = ["skill", "structured_browser", "api", "visual_light", "visual_heavy"]
EVIDENCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evidence", "ES"))


def realisations(kinds=SPEC_ORDER):
    mod = {"skill": Modality.SKILL, "structured_browser": Modality.STRUCTURED_BROWSER, "api": Modality.API,
           "code": Modality.CODE, "visual_light": Modality.VISUAL, "visual_heavy": Modality.VISUAL}
    return [Realisation(k, ActionSpec(action_id=f"a_{k}", operation=f"{k}.do", modality=mod[k],
                                      arguments={"grounder_tier": "heavy"} if k == "visual_heavy" else {})) for k in kinds]


def result(ok, seconds, compute=0.0, ops=1, cls=FailureClass.WRONG_SITE_ASSUMPTION, status=None):
    return ActionResult(action_id="x", status=status or (ResultStatus.OK if ok else ResultStatus.FAILED),
                        failure_class=FailureClass.NONE if ok else cls, started=0.0, ended=seconds,
                        cost=Cost(seconds=seconds, compute_s=compute, primitive_ops=ops), verified=True if ok else None)


def test_prior_reproduces_the_specification_ordering_and_is_weak():
    sel = ModalitySelector(seed=0)
    costs = [sel.estimate(k).expected_cost for k in SPEC_ORDER]
    assert costs == sorted(costs) and len(set(costs)) == 5
    choice = sel.choose("intent-1", realisations(), rule="greedy")
    assert choice.chosen_kind == "skill" and choice.probabilities == [1.0, 0, 0, 0, 0]
    # it is only a prior: a handful of observations overturn it
    for _ in range(6):
        sel.update("skill", None, result(False, 2.0))
        sel.update("api", None, result(True, 0.2))
    assert sel.choose("intent-2", realisations(), rule="greedy").chosen_kind == "api"
    assert sel.estimate("skill").p_success < 0.3 and sel.estimate("api").p_success > 0.9


def test_context_features_are_generic():
    obs = Observation(obs_id="o", ts=0.0, kind="page", data={"url": "https://shop.example/cart", "has_a11y": True,
                                                             "has_canvas": False, "n_interactive": 14})
    a = ActionSpec(action_id="a", operation="browser.click", modality=Modality.STRUCTURED_BROWSER,
                   arguments={"role": "button", "ref": "e12"})
    f = ExecContextFeatures.of(a, obs)
    assert f.tool == "click" and f.element_role == "button" and len(f.host_hash) == 12 and "shop" not in f.host_hash
    assert dict(f.props) == {"has_a11y": "1", "has_canvas": "0", "interactive": "10-49"}
    keys = f.keys()
    assert keys[0] == "*" and keys[1] == "t=click" and keys[2] == f"h={f.host_hash}" and len(keys) == 5
    assert "ref" not in json.dumps(keys) and "e12" not in json.dumps(keys)        # nothing task- or instance-specific
    assert ExecContextFeatures.of(ActionSpec(action_id="b", operation="http.get", modality=Modality.API,
                                             arguments={"url": "https://shop.example/api"})).host_hash == f.host_hash
    assert [default_kind(r.action) for r in realisations(KINDS)] == list(KINDS)
    with pytest.raises(ValueError):
        default_kind(ActionSpec(action_id="w", operation="wait", modality=Modality.INTERNAL))


# true behaviour of a simulated mix: (kind, context) -> (success probability, latency, compute, ops)
A = ExecContextFeatures(tool="click", host_hash="aaaaaaaaaaaa", element_role="button", props=(("has_a11y", "1"),))
B = ExecContextFeatures(tool="click", host_hash="bbbbbbbbbbbb", element_role="", props=(("has_canvas", "1"),))
TRUTH = {
    ("structured_browser", "A"): (0.95, 0.4, 0.0, 1), ("structured_browser", "B"): (0.10, 0.6, 0.0, 1),
    ("visual_light", "A"): (0.70, 1.2, 0.3, 2), ("visual_light", "B"): (0.80, 1.2, 0.3, 2),
    ("visual_heavy", "A"): (0.90, 3.0, 2.5, 2), ("visual_heavy", "B"): (0.92, 3.0, 2.5, 2),
    ("api", "A"): (0.60, 0.3, 0.0, 1), ("api", "B"): (0.05, 0.3, 0.0, 1),
}


def simulate(sel, n, rng, rule="thompson"):
    kinds = ["structured_browser", "api", "visual_light", "visual_heavy"]
    picks = {"A": [], "B": []}
    for i in range(n):
        name, ctx = ("A", A) if rng.random() < 0.5 else ("B", B)
        choice = sel.choose(f"intent-{i}", realisations(kinds), ctx, rule=rule)
        assert abs(sum(choice.probabilities) - 1.0) < 1e-9 and choice.probabilities[choice.chosen_index] > 0
        p, lat, comp, ops = TRUTH[(choice.chosen_kind, name)]
        ok = rng.random() < p
        sel.update_choice(choice, ctx, result(ok, max(0.01, rng.gauss(lat, 0.1 * lat)), comp, ops))
        picks[name].append(choice.chosen_kind)
    return picks


def test_estimates_calibrate_on_a_simulated_mix_and_selection_shifts(tmp_path):
    rng = random.Random(7)
    sel = ModalitySelector(seed=7, clock=SimClock())
    kinds = ["structured_browser", "api", "visual_light", "visual_heavy"]
    # before any experience the prior ordering decides, in both contexts
    assert sel.choose("i", realisations(kinds), A, rule="greedy").chosen_kind == "structured_browser"
    assert sel.choose("i", realisations(kinds), B, rule="greedy").chosen_kind == "structured_browser"
    # calibration data: every modality tried in every context (uniform exploration), 150 attempts per cell
    for (kind, name), (p, lat, comp, ops) in TRUTH.items():
        for _ in range(150):
            sel.update(kind, A if name == "A" else B, result(rng.random() < p, max(0.01, rng.gauss(lat, 0.1 * lat)), comp, ops))
    errs = []
    for (kind, name), (p, lat, comp, ops) in TRUTH.items():
        e = sel.estimate(kind, A if name == "A" else B)
        errs.append(abs(e.p_success - p))
        assert abs(e.p_success - p) < 0.09, (kind, name, e.p_success, p)          # ~2.5 sd of a 150-sample proportion
        assert abs(e.latency_s - lat) < 0.1 * lat and abs(e.compute_s - comp) < 0.05 and abs(e.primitive_ops - ops) < 0.05
        assert e.n_context == 150 and e.p_sd < 0.05
    assert sum(errs) / len(errs) < 0.04
    # the same kind has different estimates in different contexts (it is not one global ranking)
    assert sel.estimate("structured_browser", A).p_success - sel.estimate("structured_browser", B).p_success > 0.7
    # selection has shifted accordingly: structured control where it works, visual grounding where it does not,
    # and never the expensive model where a cheap one succeeds
    assert sel.choose("j", realisations(kinds), A, rule="greedy").chosen_kind == "structured_browser"
    assert sel.choose("j", realisations(kinds), B, rule="greedy").chosen_kind == "visual_light"
    picks = simulate(sel, 300, rng)
    assert picks["A"].count("structured_browser") / len(picks["A"]) > 0.9
    assert picks["B"].count("visual_light") / len(picks["B"]) > 0.8
    assert picks["A"].count("visual_heavy") / len(picks["A"]) < 0.03
    # an unseen context inherits the general estimate but with more uncertainty
    new = ExecContextFeatures(tool="click", host_hash="cccccccccccc")
    assert sel.estimate("visual_light", new).p_sd > sel.estimate("visual_light", A).p_sd
    out = {"simulated_mix": {f"{k}|{n}": {"p_success": v[0], "latency_s": v[1], "compute_s": v[2], "primitive_ops": v[3]}
                             for (k, n), v in TRUTH.items()},
           "attempts_per_cell": 150, "mean_abs_error_p_success": sum(errs) / len(errs), "max_abs_error_p_success": max(errs),
           "estimates": {f"{k}|{n}": sel.estimate(k, A if n == "A" else B).to_dict() for (k, n) in TRUTH},
           "greedy_choice_before_experience": {"A": "structured_browser", "B": "structured_browser"},
           "greedy_choice_after_experience": {"A": "structured_browser", "B": "visual_light"},
           "thompson_share_after_300_more_choices": {
               "A": {k: picks["A"].count(k) / len(picks["A"]) for k in kinds},
               "B": {k: picks["B"].count(k) / len(picks["B"]) for k in kinds}},
           "environment": "simulation (seeded); no real executor involved",
           "command": "cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/harness/skills/test_es_modality.py"
                      "::test_estimates_calibrate_on_a_simulated_mix_and_selection_shifts -q --rootdir=cam_live"}
    # written to a temporary file; the seeded simulation must reproduce the committed record exactly
    # (CAM_LIVE_WRITE_EVIDENCE=1 regenerates the committed file instead)
    target = os.path.join(EVIDENCE, "modality_calibration.json")
    with open(tmp_path / "modality_calibration.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if os.environ.get("CAM_LIVE_WRITE_EVIDENCE") == "1":
        with open(target, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    with open(target) as f:
        committed = json.load(f)
    assert json.loads(json.dumps(out, sort_keys=True)) == committed


def test_selection_learns_online_with_exploration_from_the_prior_alone():
    rng = random.Random(3)
    sel = ModalitySelector(seed=3, clock=SimClock())
    picks = simulate(sel, 600, rng)
    late_a, late_b = picks["A"][-100:], picks["B"][-100:]
    assert late_a.count("structured_browser") / len(late_a) > 0.8
    assert late_b.count("visual_light") / len(late_b) > 0.6
    assert late_b.count("structured_browser") / len(late_b) < 0.1 and late_b.count("api") / len(late_b) < 0.1
    # every recorded choice carries the probability of each option and of the one taken
    for c in sel.history[-50:]:
        assert len(c.probabilities) == len(c.options) == 4 and c.rule == "thompson"
        assert {"kind", "p_success", "latency_s", "compute_s", "primitive_ops", "recovery_s", "expected_cost"} <= set(c.options[0])


@pytest.mark.parametrize("status,cls", [(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION),
                                        (ResultStatus.FAILED, FailureClass.MISSING_PERMISSION),
                                        (ResultStatus.FAILED, FailureClass.PLATFORM_RESTRICTION),
                                        (ResultStatus.NEEDS_APPROVAL, FailureClass.NONE)])
def test_a_refusal_never_causes_a_modality_switch(status, cls):
    sel = ModalitySelector(seed=0, clock=SimClock())
    intent = ActionSpec(action_id="i", operation="post_message", target="channel 7", modality=Modality.INTERNAL)
    options = realisations(["structured_browser", "api", "visual_light", "visual_heavy"])
    first = sel.choose(intent, options, A, rule="greedy")
    assert first.chosen_kind == "structured_browser" and not first.blocked
    before = sel.estimate("structured_browser", A)
    assert sel.update_choice(first, A, result(False, 0.3, cls=cls, status=status)) == "blocked"
    # the refusal removes the intent, not just the modality: no realisation is offered any more
    again = sel.choose(intent, options, A, rule="greedy")
    assert again.blocked and again.chosen_index == -1 and again.options == [] and again.chosen_kind == ""
    assert "no alternative modality" in again.reason
    for kind in ("api", "visual_light", "visual_heavy"):
        only = sel.choose(intent, realisations([kind]), A)
        assert only.blocked and only.chosen_index == -1
    assert sel.blocked(intent)["failure_class"] == cls and sel.blocked(intent_key(intent))["kind"] == "structured_browser"
    # and it is not evidence about the modality's reliability
    after = sel.estimate("structured_browser", A)
    assert (after.p_success, after.n, after.latency_s) == (before.p_success, before.n, before.latency_s)
    # other intents are unaffected; an ordinary failure does lead to the alternatives being considered
    other = ActionSpec(action_id="o", operation="read_page", target="page 3", modality=Modality.INTERNAL)
    c = sel.choose(other, options, A, rule="greedy")
    assert not c.blocked
    assert sel.update_choice(c, A, result(False, 0.3, cls=FailureClass.PAGE_CHANGED)) == "failure"
    assert not sel.choose(other, options, A).blocked
    # the block survives persistence and is lifted only when the authorization service reports a grant
    restored = ModalitySelector(seed=0, clock=SimClock())
    restored.load_state_dict(json.loads(json.dumps(sel.state_dict())))
    assert restored.choose(intent, options, A).blocked
    restored.authorization_changed(intent)
    assert not restored.choose(intent, options, A).blocked


def test_estimates_are_exposed_as_quantities_and_persist(tmp_path):
    sel = ModalitySelector(seed=1, clock=SimClock())
    for _ in range(20):
        sel.update("api", A, result(True, 0.25, ops=1))
        sel.update("api", A, result(False, 0.5, ops=1), recovery_s=1.5)
    q = {x.name: x.value for x in sel.quantities(A)}
    assert len(q) == 8 * len(KINDS)
    assert abs(q["exec.modality.api.p_success"] - 0.5) < 0.05 and q["exec.modality.api.n"] == 40.0
    assert abs(q["exec.modality.api.latency_s"] - 0.375) < 0.02
    assert abs(q["exec.modality.api.recovery_s"] - 2.0) < 0.1          # failed attempt (0.5 s) + reported recovery (1.5 s)
    e = sel.estimate("api", A)
    expected = (e.latency_s + 1.0 * e.compute_s + 0.05 * e.primitive_ops + (1 - e.p_success) * e.recovery_s) / e.p_success
    assert q["exec.modality.api.expected_cost"] == pytest.approx(expected)
    # the same numbers reach the decision-maker's candidate evaluations
    ev = [CandidateEvaluation(action=ActionSpec(action_id="c", operation="http.get", modality=Modality.API,
                                                arguments={"url": "https://x.example/a"})),
          CandidateEvaluation(action=ActionSpec(action_id="w", operation="wait", modality=Modality.INTERNAL))]
    sel.annotate(ev)
    assert set(ev[0].extras) == {"exec_p_success", "exec_p_sd", "exec_latency_s", "exec_compute_s", "exec_primitive_ops",
                                 "exec_recovery_s", "exec_expected_cost"} and ev[1].extras == {}
    path = str(tmp_path / "modality.json")
    sel.save(path)
    again = ModalitySelector(seed=1, clock=SimClock())
    again.load(path)
    assert again.estimate("api", A).to_dict() == sel.estimate("api", A).to_dict()
    # cancelled attempts teach nothing
    assert sel.update("api", A, result(False, 0.1, status=ResultStatus.CANCELLED)) == "ignored"
    assert set(DEFAULT_PRIORS) == set(KINDS)
