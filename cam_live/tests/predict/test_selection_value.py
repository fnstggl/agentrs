"""Selection that uses what the agent predicts (change request S-5, spec 5.1 / 6 / 8): the ``thompson_policy`` rule.

Regression tests for the diagnosis in ``docs/S/DIAGNOSIS.md``: under the old default rule (``policy``: a softmax of
an untrained policy head) the choice ignored the world model's predicted goal values and was indistinguishable from
uniform. ``thompson_policy`` is linear Thompson sampling over the world model's joint posterior samples and the
policy's centred log-probability, weighted by a Bayesian calibration on verified outcomes."""
import dataclasses
import math

import pytest
import torch

from cam_live.contracts import ActionSpec, CandidateEvaluation, Decision
from cam_live.predict import policy as P


def _evals(values, logits, *, sd=0.05, n_samples=32, seed=0, feasible=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (v, lg) in enumerate(zip(values, logits)):
        ex = {"q_goal": float(v), "q_goal_epistemic_std": sd, "net_cost": 0.0, "policy_logit": float(lg)}
        for s in range(n_samples):
            ex[f"ts_{s}"] = float(v + sd * torch.randn((), generator=g))
        out.append(CandidateEvaluation(action=ActionSpec(action_id=f"a{i}", operation=f"op{i % 3}"), value=float(lg),
                                       p_success=0.5, uncertainty=sd, extras=ex,
                                       feasible=True if feasible is None else feasible[i]))
    return out


class _Heads:
    def __init__(self):
        from cam_live.predict.heads import PredictConfig
        self.cfg = PredictConfig()
        self.policy = P.CandidatePolicy(P.PolicyConfig(d_model=8, hidden=8))


def _dm(seed=0):
    return P.DecisionMaker(model=None, heads=_Heads(), seed=seed)


def test_rule_is_registered_and_selectable_by_configuration():
    assert P.RULE_THOMPSON_POLICY in P.RULES and P.RULES[-1] == P.RULE_RANDOMIZED


def test_untrained_policy_follows_the_world_models_predictions_with_exact_propensities():
    """A uniform (untrained) policy and clearly separated predicted values: the old rule ``policy`` is uniform, the new
    rule puts almost all mass on the candidate the agent itself predicts best, and samples from exactly the vector it
    records."""
    values = [0.0, 0.0, 1.0, 5.0, 0.0, 2.0]
    evs = _evals(values, [0.0] * 6)
    dm = _dm()
    p_old, _ = dm.probabilities(evs, P.RULE_POLICY)
    assert float((p_old - 1 / 6).abs().max()) < 1e-12                       # what S measured: uniform
    p, sc = dm.probabilities(evs, P.RULE_THOMPSON_POLICY)
    eps = dm.cfg.eps_floor
    assert abs(float(p.sum()) - 1.0) < 1e-12 and float(p.min()) >= eps / 6 - 1e-15
    assert int(p.argmax()) == 3 and float(p[3]) > 0.9
    assert sc["selection"]["policy_share_tv"] < 1e-9                         # c == 0: the policy has no say
    g = torch.Generator().manual_seed(3)
    picks = torch.tensor([dm.select(evs, P.RULE_THOMPSON_POLICY, ts=0.0, generator=g).chosen_index
                          for _ in range(600)])
    # each select redraws the calibration weights, but with c == 0 and beta_q > 0 the distribution is the same
    freq = torch.bincount(picks, minlength=6).double() / 600
    assert float((freq - p).abs().max()) < 4 * math.sqrt(0.25 / 600)
    dec = dm.select(evs, P.RULE_THOMPSON_POLICY, ts=0.0)
    sel = dec.diagnostics["selection"]
    assert sel["rule"] == P.RULE_THOMPSON_POLICY and sel["calibration"]["n"] == 0
    assert len(sel["value_only_probabilities"]) == 6 and sel["coef_positive_share"]["beta_q"] > 0.9
    assert dec.exploration == P.RULE_THOMPSON_POLICY and dec.propensity == dec.probabilities[dec.chosen_index]


def test_uncertain_predictions_explore_in_proportion_to_the_posterior():
    """Thompson: two candidates with overlapping posteriors share the mass by their probability of being best."""
    evs = _evals([1.0, 1.1, -3.0], [0.0] * 3, sd=0.5, n_samples=32, seed=4)
    p, _ = _dm().probabilities(evs, P.RULE_THOMPSON_POLICY)
    assert 0.15 < float(p[0]) < 0.85 and 0.15 < float(p[1]) < 0.85 and float(p[2]) < 0.05


def test_infeasible_candidates_get_exactly_zero_and_ties_are_not_broken_by_position():
    evs = _evals([3.0, 3.0, 9.0], [0.0] * 3, sd=0.0, feasible=[True, True, False])
    p, _ = _dm().probabilities(evs, P.RULE_THOMPSON_POLICY)
    assert float(p[2]) == 0.0 and abs(float(p[0]) - float(p[1])) < 1e-12


def _feed(cal, n, fn, seed=0):
    g = torch.Generator().manual_seed(seed)
    for k in range(n):
        z = float(torch.randn((), generator=g))
        c = float(torch.randn((), generator=g))
        cal.observe(f"d{k}", z, c, fn(z, c, float(torch.randn((), generator=g))))


def test_policy_influence_grows_only_when_its_preferences_predict_verified_outcomes():
    """The weight of the policy is learned from verified outcomes: a policy whose preference predicts the verified
    progress beyond the world model gains influence on the selection; one whose preference is noise or harmful does
    not (the weight is clipped at zero, never inverted)."""
    values = [1.0, 2.0, 1.0, 2.0]             # the world model cannot tell candidates 1 and 3 apart
    logits = [0.0, 0.0, 3.0, 3.0]             # the policy prefers 2 and 3: orthogonal to the world model's ranking
    evs = _evals(values, logits, sd=0.05)
    shares = {}
    for name, fn in (("prior", None), ("predictive", lambda z, c, e: 2 + z + 1.5 * c + 0.3 * e),
                     ("noise", lambda z, c, e: 2 + z + 0.3 * e), ("harmful", lambda z, c, e: 2 + z - 1.5 * c + 0.3 * e)):
        dm = _dm(seed=1)
        if fn is not None:
            _feed(dm.calibration, 200, fn)
        p, sc = dm.probabilities(evs, P.RULE_THOMPSON_POLICY)
        shares[name] = (float(p[3]), sc["selection"]["policy_share_tv"], sc["selection"]["calibration"],
                        float(p[1]))
    vu = lambda k, b: shares[k][2][b] * shares[k][2]["scale"]                 # noqa: E731  verified units
    assert shares["prior"][2]["n"] == 0 and shares["prior"][0] < shares["predictive"][0]
    assert abs(vu("predictive", "beta_pi") - 1.5) < 0.2 and abs(vu("predictive", "beta_q") - 1.0) < 0.2
    assert shares["predictive"][0] > 0.9 and shares["predictive"][3] < 0.05
    assert abs(vu("noise", "beta_pi")) < 0.2 and shares["noise"][0] < shares["predictive"][0] - 0.2
    assert vu("harmful", "beta_pi") < -1.0 and shares["harmful"][1] < 1e-12            # lambda = 0: value only
    assert 0.3 < shares["harmful"][0] < 0.7 and 0.3 < shares["harmful"][3] < 0.7      # 1 vs 3: the posterior's tie
    assert shares["predictive"][1] > 0.3 > shares["noise"][1]
    # the policy's preference only counts beyond the world model's ranking: a policy that repeats it has no say
    dm = _dm(seed=1)
    _feed(dm.calibration, 200, lambda z, c, e: 2 + z + 1.5 * c + 0.3 * e)
    copy = _evals(values, [0.0, 3.0, 0.0, 3.0], sd=0.05)
    p, sc = dm.probabilities(copy, P.RULE_THOMPSON_POLICY)
    assert sc["selection"]["policy_share_tv"] < 1e-9
    # a world model whose ranking says nothing about the verified outcome loses its weight (beta_q -> 0): ranking,
    # not the units of the prediction, is what is calibrated
    cal = P.SelectionCalibration()
    _feed(cal, 300, lambda z, c, e: 2 + 0.3 * e, seed=2)
    assert abs(cal.summary()["beta_q"]) < 0.1 and cal.predict(1.0) == pytest.approx(cal.predict(-1.0), abs=0.3)


def test_calibration_reads_the_recorded_decision_and_round_trips():
    evs = _evals([0.5, 2.0, 1.0], [0.0, 1.0, -1.0])
    dec = Decision(decision_id="dec_1", ts=0.0, objective_id="o", chosen=evs[1].action, candidates=evs,
                   probabilities=[0.2, 0.6, 0.2], chosen_index=1)
    z, c = P.SelectionCalibration.features(dec)
    lp = torch.log_softmax(torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64), -1)
    q = torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64)
    zs = (q - q.mean()) / q.std(unbiased=False)
    assert abs(z - float(zs[1])) < 1e-12
    cc = lp - lp.mean()
    perp = cc - (cc * zs).sum() / (zs * zs).sum() * zs            # the preference beyond the world model's ranking
    assert abs(c - float(perp[1] / max(1.0, float(perp.pow(2).mean().sqrt())))) < 1e-12
    # units drop out: the same predictions x 1000 give the same feature
    big = dataclasses.replace(dec, candidates=[dataclasses.replace(e, extras={**e.extras, "q_goal": 1000 * e.extras[
        "q_goal"]}) for e in evs])
    assert abs(P.SelectionCalibration.features(big)[0] - z) < 1e-9
    no_goal = dataclasses.replace(dec, candidates=[dataclasses.replace(e, extras={}) for e in evs])
    assert P.SelectionCalibration.features(no_goal) is None
    cal = P.SelectionCalibration()
    assert cal.observe_decision("e1", dec, 3.0) and not cal.observe_decision("e1", dec, 3.0)   # once per decision
    _feed(cal, 20, lambda g, c, e: g + c)
    other = P.SelectionCalibration()
    other.load_state_dict(cal.state_dict())
    assert other.posterior() == cal.posterior() and len(other) == len(cal) == 21
    assert not other.observe("e1", 1.0, 0.0, 1.0)


def test_shared_calibration_object_on_the_policy_head():
    pol = P.CandidatePolicy(P.PolicyConfig(d_model=8, hidden=8))
    a, b = P.selection_calibration(pol), P.selection_calibration(pol)
    assert a is b and "selection_calibration" not in dict(pol.named_parameters())
    assert all("selection" not in k for k in pol.state_dict())               # checkpoints are unchanged


def test_real_debug_model_choices_track_its_own_predictions(lab_rig):
    """On the debug-scale model after a few honestly collected LabWorld lifetimes: the new rule's chosen candidates
    have a predicted goal value close to the best predicted one, the old rule's do not."""
    rig = lab_rig(n_lifetimes=2, n_decisions=10)
    dm = P.DecisionMaker(rig.model, heads=rig.heads, clock=rig.clock, seed=2)
    evs = rig.evaluate()
    sc = dm.scores(evs)
    q = sc["q"][0]
    pol, _ = dm.probabilities(evs, P.RULE_POLICY)
    new, _ = dm.probabilities(evs, P.RULE_THOMPSON_POLICY)
    rel = lambda p: float((p * (q - q.min())).sum() / (q.max() - q.min()).clamp_min(1e-12))   # noqa: E731
    print("relative predicted value of the choice: thompson_policy", rel(new), "policy", rel(pol))
    assert rel(new) > 0.6 and rel(new) > rel(pol) + 0.25
