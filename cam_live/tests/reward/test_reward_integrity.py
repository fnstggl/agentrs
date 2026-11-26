"""Reward-integrity guards (spec 7.2, 7.8, 16.6; MR-09, MR-11, MR-06)."""
import hashlib
import hmac
import json

import pytest
import torch

from cam_live.contracts import (GoalStatus, OutcomeObservation, OutcomeQuery, OutcomeStatus, PredictionRecord,
                                Distribution, Trust, VerifiedOutcome)
from cam_live.reward import integrity as I
from cam_live.reward import meta as M
from cam_live.reward.bandit_lifetime import BanditLifetime

DT = torch.float64


class Verifier:
    """Stands in for the independent verifier process: holds a key the learner never sees."""

    def __init__(self, key=b"verifier-only-key"):
        self._key = key

    def _mac(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def sign(self, v: VerifiedOutcome) -> VerifiedOutcome:
        v.signature = ""
        v.signature = self._mac(v.to_json())
        return v

    def check_signature(self, v: VerifiedOutcome) -> bool:
        sig, v.signature = v.signature, ""
        try:
            return hmac.compare_digest(sig, self._mac(v.to_json()))
        finally:
            v.signature = sig

    def attest(self, o: OutcomeObservation) -> None:
        self.attested = getattr(self, "attested", set()) | {self._mac(o.to_json())}

    def check_measurement(self, o: OutcomeObservation) -> bool:
        return self._mac(o.to_json()) in getattr(self, "attested", set())


def _vo(value=12.0, at=100.0, objective="obj1", refs=("ev1",)):
    return VerifiedOutcome(objective_id=objective, status=GoalStatus.PENDING, value=value, target=50.0,
                           evaluated_at=at, evidence_refs=list(refs))


def _obs(i=0, **kw):
    d = dict(query_id=f"q{i}", status=OutcomeStatus.OBSERVED, value=3.0, observed_at=10.0 + i, verified=True,
             source="verifier", dedup_key=f"k{i}")
    d.update(kw)
    return OutcomeObservation(**d)


# ----------------------------------------------------------------------------------------- psi isolation
def test_no_learner_optimizer_may_hold_reward_parameters():
    P = BanditLifetime()
    inner = torch.optim.SGD(P.policy.parameters(), lr=0.1)
    I.assert_optimizers_disjoint([inner], P.reward)
    I.assert_reward_optimizer_exclusive(torch.optim.Adam(P.reward.parameters()), P.reward)
    leaky = torch.optim.SGD(list(P.policy.parameters()) + [next(P.reward.parameters())], lr=0.1)
    assert I.optimizer_overlap(leaky, P.reward) == [0]
    with pytest.raises(I.IntegrityError):
        I.assert_optimizers_disjoint([inner, leaky], P.reward)
    with pytest.raises(I.IntegrityError):
        I.assert_reward_optimizer_exclusive(leaky, P.reward)
    with pytest.raises(I.IntegrityError):
        I.assert_reward_optimizer_exclusive(torch.optim.Adam(list(P.reward.parameters())[:-1]), P.reward)


def test_inner_loss_cannot_update_psi_and_the_guard_catches_a_leak():
    P = BanditLifetime()
    g = torch.Generator().manual_seed(0)
    batch = P.batch(torch.randint(P.n_ctx, (10,), generator=g), torch.randint(P.n_arms, (10,), generator=g))
    theta = dict(P.policy.named_parameters())
    opt = torch.optim.SGD(P.policy.parameters(), lr=0.1)
    # the learner's real (first-order) step: reward detached -> allowed
    with I.PsiGuard(P.reward, "policy step"):
        loss = -(P.logp_fn(theta, batch) * P.reward_fn(P.psi(), batch).detach()).mean()
        I.assert_no_grad_path(loss, P.reward, "policy loss")
        opt.zero_grad(); loss.backward(); opt.step()
    # a leak: the same loss with the reward left in the graph
    leaky = -(P.logp_fn(theta, batch) * P.reward_fn(P.psi(), batch)).mean()
    assert I.depends_on(leaky, P.reward)
    with pytest.raises(I.IntegrityError):
        I.assert_no_grad_path(leaky, P.reward, "policy loss")
    with pytest.raises(I.IntegrityError, match="gradient reached"):
        with I.PsiGuard(P.reward, "policy step"):
            (-(P.logp_fn(theta, batch) * P.reward_fn(P.psi(), batch)).mean()).backward()
    for p in P.reward.parameters():
        p.grad = None
    # a direct write to psi ("raise my own reward") is caught too
    with pytest.raises(I.IntegrityError, match="changed"):
        with I.PsiGuard(P.reward, "policy step"):
            with torch.no_grad():
                next(P.reward.parameters()).add_(1.0)
    # world-model / evaluator style losses do not depend on psi at all
    world_loss = (P.policy(P.x) ** 2).mean()
    I.assert_no_grad_path(world_loss, P.reward, "world loss")
    assert not I.depends_on(world_loss, P.reward)


def test_the_functional_inner_update_leaves_psi_bit_identical():
    P = BanditLifetime()
    g = torch.Generator().manual_seed(0)
    sup = [P.batch(torch.randint(P.n_ctx, (10,), generator=g), torch.randint(P.n_arms, (10,), generator=g))]
    mg = M.MetaGradient(M.policy_gradient_inner_loss(P.logp_fn, P.reward_fn), P.logp_fn, inner_lr=0.5)
    with I.PsiGuard(P.reward, "inner update"):
        for mode in M.MODES:
            mg.inner_update(P.theta(), P.psi(), sup, K=4, mode=mode)


# ------------------------------------------------------------------------------------- scale / magnitude
def test_inflating_reward_magnitude_cannot_change_the_induced_policy_update():
    P = BanditLifetime()
    g = torch.Generator().manual_seed(0)
    sup = [P.batch(torch.randint(P.n_ctx, (10,), generator=g), torch.randint(P.n_arms, (10,), generator=g))
           for _ in range(4)]
    inflated = lambda psi, b: 1000.0 * P.reward_fn(psi, b) + 7.0

    def theta_K(reward_fn, normalise):
        mg = M.MetaGradient(M.policy_gradient_inner_loss(P.logp_fn, reward_fn, normalise=normalise), P.logp_fn,
                            inner_lr=0.5)
        return M.flat(mg.inner_update(P.theta(), P.psi(), sup, K=4, create_graph=False).detached())
    a, b = theta_K(P.reward_fn, True), theta_K(inflated, True)
    assert float((a - b).abs().max()) < 1e-7                         # identical update
    c, d = theta_K(P.reward_fn, False), theta_K(inflated, False)
    assert float((c - d).abs().max()) > 1.0                          # without the guard it would dominate
    r = torch.randn(50, dtype=DT, generator=g)
    z = I.scale_normalise(r)
    assert torch.allclose(z, I.scale_normalise(1e4 * r - 3.0), atol=1e-6)
    assert float(z.mean().abs()) < 1e-12 and float((z ** 2).mean()) == pytest.approx(1.0, abs=1e-6)
    grp = torch.arange(50) % 2
    zg = I.scale_normalise(r, grp)
    assert torch.allclose(zg[grp == 0], I.scale_normalise(r[grp == 0]))
    assert float(I.bound_reward(torch.tensor([1e9, -1e9, 0.01], dtype=DT), 2.0).abs().max()) <= 2.0
    one = I.scale_normalise(torch.tensor([3.0], dtype=DT, requires_grad=True))
    assert float(one.detach()) == 0.0 and bool(torch.isfinite(torch.autograd.grad(one.sum(), one, allow_unused=True)[0]))


# ---------------------------------------------------------------------------------- outcome screening
def test_duplicate_outcomes_are_counted_once():
    outs = [_obs(1), _obs(1), _obs(2), _obs(1, value=99.0), _obs(3, dedup_key="")]
    kept, dropped = I.dedup_outcomes(outs + [outs[-1]])
    assert [o.dedup_key for o in kept] == ["k1", "k2", ""] and len(dropped) == 3


def test_only_authenticated_verifier_measurements_pass_screening():
    v = Verifier()
    good = _obs(1)
    v.attest(good)
    assert I.screen_observation(good, check_measurement=v.check_measurement) == (True, "")
    cases = {
        "imagined": (good, dict(imagined=True)),
        "self_authored": (_obs(2, source="prediction"), {}),
        "not_verified": (_obs(3, verified=False), {}),
        "status_pending": (_obs(4, status=OutcomeStatus.PENDING), {}),
        "status_unobservable": (_obs(5, status=OutcomeStatus.UNOBSERVABLE), {}),
        "no_finite_value": (_obs(6, value=None), {}),
        "bad_signature": (_obs(7), {}),                              # verified=True set by the learner itself
    }
    for reason, (o, kw) in cases.items():
        assert I.screen_observation(o, check_measurement=v.check_measurement, **kw) == (False, reason), reason
    assert I.screen_observation(good) == (False, "no_measurement_authentication")       # fail closed
    tampered = _obs(1, value=1e6)
    assert I.screen_observation(tampered, check_measurement=v.check_measurement) == (False, "bad_signature")


def test_objective_outcomes_need_a_valid_signature_and_an_independent_objective():
    v = Verifier()
    signed = v.sign(_vo())
    assert I.screen_verified(signed, v.check_signature) == (True, "")
    assert I.screen_verified(_vo(), v.check_signature) == (False, "unsigned")
    forged = _vo(value=1e9); forged.signature = "f" * 64             # the learner fabricates a success
    assert I.screen_verified(forged, v.check_signature) == (False, "bad_signature")
    edited = v.sign(_vo()); edited.value = 49.0                      # a signed record edited afterwards
    assert I.screen_verified(edited, v.check_signature) == (False, "bad_signature")
    assert I.screen_verified(signed, v.check_signature, objective_origin=Trust.SELF) == (False, "self_authored_objective")
    assert I.screen_verified(signed, v.check_signature, objective_origin=Trust.UNTRUSTED)[0] is False
    assert I.screen_verified(signed, lambda _: 1 / 0) == (False, "bad_signature")       # a failing check is a no


def test_stale_predictions_are_not_prospective():
    q = OutcomeQuery(query_id="q1", metric="m", issued_at=5.0, horizon_s=10.0)
    mk = lambda made: PredictionRecord(prediction_id="p", query=q, dist=Distribution("normal"), made_at=made)
    o = _obs(1, query_id="q1", observed_at=15.0)
    assert I.is_prospective(mk(6.0), o)
    assert not I.is_prospective(mk(15.0), o) and not I.is_prospective(mk(20.0), o)       # written after the fact
    assert not I.is_prospective(mk(6.0), _obs(2, query_id="other", observed_at=15.0))


# ------------------------------------------------------------------------------------------- ledger
def test_ledger_admits_signed_outcomes_once_and_rejects_fabrications(tmp_path):
    v = Verifier()
    path = str(tmp_path / "ledger" / "outcomes.jsonl")
    led = I.OutcomeLedger(v.check_signature, v.check_measurement, path=path)
    assert led.head == I.OutcomeLedger.GENESIS
    a = led.admit_verified(v.sign(_vo(12.0, 100.0)), group="exp:E1", exp_ids=["e1", "e2"])
    assert a.accepted and a.entry.utility == 12.0 and a.entry.group == "exp:E1" and a.entry.kind == "objective"
    again = led.admit_verified(v.sign(_vo(12.0, 100.0)), group="exp:E1")
    assert not again.accepted and again.reason == "duplicate" and len(led) == 1          # counted once
    forged = _vo(1e9, 101.0); forged.signature = "0" * 64
    assert led.admit_verified(forged).reason == "bad_signature"
    assert led.admit_verified(v.sign(_vo(5.0, 102.0, refs=("ev2",))), objective_origin=Trust.SELF).reason \
        == "self_authored_objective"
    m = _obs(1); v.attest(m)
    assert led.admit_measurement(m, objective_id="obj1", group="exp:E2").accepted
    assert led.admit_measurement(m).reason == "duplicate"
    assert led.admit_measurement(_obs(2)).reason == "bad_signature"
    assert led.admit_measurement(m, imagined=True).reason == "imagined"
    assert len(led) == 2 and led.independent_groups() == 2 and led.verify_chain() == (True, -1)
    assert [r for r, _ in led.rejected] == ["duplicate", "bad_signature", "self_authored_objective", "duplicate",
                                            "bad_signature", "imagined"]
    # persistence: the reloaded ledger has the same head and still refuses the duplicate
    re = I.OutcomeLedger(v.check_signature, v.check_measurement, path=path)
    assert re.head == led.head and len(re) == 2 and not re.admit_measurement(m).accepted
    no_meas = I.OutcomeLedger(v.check_signature)
    assert no_meas.admit_measurement(m).reason == "no_measurement_authentication"


def test_ledger_tampering_is_detected(tmp_path):
    v = Verifier()
    path = str(tmp_path / "outcomes.jsonl")
    led = I.OutcomeLedger(v.check_signature, path=path)
    for i in range(4):
        assert led.admit_verified(v.sign(_vo(float(i), 100.0 + i, refs=(f"ev{i}",)))).accepted
    head = led.head
    lines = open(path).read().splitlines()
    # (1) edit a utility in place
    d = json.loads(lines[1]); d["utility"] = 1e9
    open(path, "w").write("\n".join([lines[0], json.dumps(d, sort_keys=True)] + lines[2:]) + "\n")
    with pytest.raises(I.IntegrityError, match="entry 1"):
        I.OutcomeLedger(v.check_signature, path=path)
    # (2) delete an entry
    open(path, "w").write("\n".join([lines[0]] + lines[2:]) + "\n")
    with pytest.raises(I.IntegrityError):
        I.OutcomeLedger(v.check_signature, path=path)
    # (3) reorder
    open(path, "w").write("\n".join([lines[1], lines[0]] + lines[2:]) + "\n")
    with pytest.raises(I.IntegrityError):
        I.OutcomeLedger(v.check_signature, path=path)
    # (4) truncation changes the head recorded by an earlier meta-update
    open(path, "w").write("\n".join(lines[:3]) + "\n")
    assert I.OutcomeLedger(v.check_signature, path=path).head != head
    # in memory
    e = led._entries[2]
    led._entries[2] = I.LedgerEntry(**{**e.__dict__, "utility": 77.0})
    assert led.verify_chain() == (False, 2)


def test_fabricated_outcomes_never_reach_the_outer_objective():
    """End to end: utilities come from the ledger; anything it rejected is absent, and a row flagged unverified
    makes the outer estimate refuse."""
    v = Verifier()
    led = I.OutcomeLedger(v.check_signature)
    real = [v.sign(_vo(float(i), 100.0 + i, refs=(f"ev{i}",))) for i in range(6)]
    fake = _vo(1e6, 500.0, refs=("made-up",)); fake.signature = "a" * 64
    for o in real + [fake]:
        led.admit_verified(o)
    assert [e.utility for e in led.entries()] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    P = BanditLifetime()
    g = torch.Generator().manual_seed(0)
    mg = M.MetaGradient(M.policy_gradient_inner_loss(P.logp_fn, P.reward_fn), P.logp_fn, inner_lr=0.5)
    sup = [P.batch(torch.randint(P.n_ctx, (8,), generator=g), torch.randint(P.n_arms, (8,), generator=g))]
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=1)
    batch, _ = P.sample(tr.detached(), 7, g)
    U = torch.tensor([e.utility for e in led.entries()] + [1e6], dtype=DT)
    flags = torch.tensor([True] * 6 + [False])
    res = mg.meta_gradient(tr, M.QueryData(batch=batch, utility=U, verified=flags))
    assert not res.supported and res.reason == "unverified_outcome_in_query" and res.grads == {}


# ------------------------------------------------------------------------------- self-generated goals
def test_repeated_and_trivial_self_generated_goals_are_discounted():
    d = I.SelfGoalDiscount(decay=0.5, trivial_above=0.9)
    w = [d.weight("Reduce prediction loss on metric 17 below 0.30", 0.4) for _ in range(5)]
    assert w == [1.0, 0.5, 0.25, 0.125, 0.0625]
    assert d.weight("reduce   prediction loss on METRIC 99 below 0.05!", 0.4) == 0.03125   # same goal, new numbers
    assert d.weight("Learn what the second form field accepts", 0.4) == 1.0                # a different goal
    assert d.weight("Observe that the page still exists", 0.999) == pytest.approx(0.01)   # already certain: trivial
    assert d.weight("Another certain thing", 1.0) == 0.0
    assert d.weight("Peek only", 0.2, register=False) == 1.0 and d.weight("Peek only", 0.2) == 1.0
    total = sum(d.weight("the same easy thing again", 0.5) for _ in range(1000))
    assert total <= 2.0                                              # unbounded repetition earns a bounded amount
    twin = I.SelfGoalDiscount(); twin.load_state_dict(d.state_dict())
    assert twin.weight("Learn what the second form field accepts", 0.4) == 0.5
