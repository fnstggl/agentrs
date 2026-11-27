"""Regression tests for the defects of the independent verification G-pr (docs/verification/G_pr.md, D1-D4, D6-D8,
D12; docs/C/FIXES.md). Each fails on the code the verifier audited."""
import dataclasses
import itertools
import math

import pytest
import torch

from cam_live.contracts import Cost, GoalStatus, TaskSpec, Trust
from cam_live.diagnostics.collect import LabCollector, verified_return
from cam_live.diagnostics.world import LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.reward import diagnose as dg
from cam_live.reward.integrity import IntegrityError, OutcomeLedger
from cam_live.reward.meta import ESTIMATOR_IMPORTANCE
from cam_live.reward.metalearner import MetaLearner, _Episode
from cam_live.reward.selfgoals import GoalFreeBudget, LearningGoalCandidate, SelfGoalSelector

D = torch.float64


# ============================================================================== D1: return-to-go baseline, unbiased
def _episode(U, group, prefix=None, held=None):
    n = len(U)
    return _Episode(0, [], "s", "q", None, [f"e{i}" for i in range(n)], list(U), [True] * n, list(group),
                    list(held or [False] * n), set(), [], indep=[f"g{i}" for i in range(n)],
                    prefix_base=list(prefix or [0.0] * n))


def _rtg(m, disc):
    return [sum(disc ** (j - t) * m[j] for j in range(t, len(m))) for t in range(len(m))]


def test_return_to_go_baseline_keeps_the_outer_policy_gradient_unbiased_exactly():
    """Exact expectation over every action sequence of a 4-row lifetime (2 actions per row, the measured outcome of
    row t depends on its own action): E[sum_t 1/N grad log pi(a_t) (U_t - b_t)] equals grad (1/N) sum_t E[m_t] with
    MetaLearner's baselines (return-to-go chain, prefix fallback), and is biased with the leave-group-out baseline
    over independent-outcome groups that the audited code used."""
    T, disc = 4, 0.9
    w = torch.tensor([0.3, -0.4], dtype=D)                       # shared softmax policy over 2 actions
    pi = torch.softmax(w, 0)
    mu = [torch.tensor([1.0, 3.0], dtype=D), torch.tensor([2.0, -1.0], dtype=D), torch.tensor([0.5, 4.0], dtype=D),
          torch.tensor([-2.0, 1.5], dtype=D)]
    score = lambda a: torch.eye(2, dtype=D)[a] - pi                # grad_w log pi(a)
    true = sum(((torch.diag(pi) - torch.outer(pi, pi)) @ mu[t]) for t in range(T)) / T
    est_new, est_old = torch.zeros(2, dtype=D), torch.zeros(2, dtype=D)
    for seq in itertools.product((0, 1), repeat=T):
        p = math.prod(float(pi[a]) for a in seq)
        m = [float(mu[t][a]) for t, a in enumerate(seq)]
        U = _rtg(m, disc)
        rows = [(0, t) for t in range(T)]
        new = MetaLearner.outer_baselines([_episode(U, ["rtg:life=0:held=0"] * T, prefix=[0.7] * T)], rows)
        old = {(0, t): sum(U[j] for j in range(T) if j != t) / (T - 1) for t in range(T)}   # leave-own-group-out
        for t, a in enumerate(seq):
            est_new += p * score(a) * (U[t] - new[(0, t)]) / T
            est_old += p * score(a) * (U[t] - old[(0, t)]) / T
    assert torch.allclose(est_new, true, atol=1e-12, rtol=0)
    assert float((est_old - true).norm() / true.norm()) > 0.1    # the audited baseline is biased


def test_meta_learner_baseline_of_a_row_does_not_depend_on_its_own_measured_outcome():
    """The verifier's scenario (one lifetime, authenticated measurements, return-to-go): every row whose utility
    contains row a's outcome shares row a's baseline group, and the baseline actually used for row a is unchanged
    when row a's measured outcome changes."""
    ec = dg.ExperimentConfig(min_independent=2)
    auth = dg.LabMeasurementAuth()
    model, store, clock, ml = dg.make_stack(7, ec, mix=0.0, check_measurement=auth)
    ml.new_lifetime()
    dg.lab_lifetime(ml, store, clock, ec, seed=4242, auth=auth, n_decisions=24, update_every=8)
    clock.advance(5 * 86400.0)
    eps = [e for e in ml._collect()[0] if e.ibq is not None]
    beta = ml.inner.cfg.beta
    checked = 0
    for k, e in enumerate(eps):
        mea = [sum(x.utility for x in ml.ledger.by_exp(i) if x.kind == "measurement") for i in e.exp_ids]
        ts = [store.get(i).decision.ts for i in e.exp_ids]
        rtg = lambda m: [sum(math.exp(-beta * max(0.0, ts[j] - ts[i])) * m[j] for j in range(i, len(m)))
                         for i in range(len(m))]
        assert max(abs(u - v) for u, v in zip(rtg(mea), e.utility)) < 1e-9
        assert len(set(e.group)) == 1 and len(set(e.indep)) > 1  # one chain, several independent outcome groups
        rows = [(0, a) for a in range(len(e.exp_ids)) if e.has_u[a] and not e.held[a]]
        base = MetaLearner.outer_baselines([e], rows)
        for a in range(len(mea)):
            if not mea[a] or (0, a) not in base:
                continue
            m2 = list(mea)
            m2[a] += 1.0
            e2 = dataclasses.replace(e, utility=rtg(m2))
            assert MetaLearner.outer_baselines([e2], rows)[(0, a)] == base[(0, a)]
            checked += 1
    assert checked >= 2


# ================================================================== D2 / D3 on the real MetaLearner path
def test_signed_outcome_utility_and_credit_cannot_be_chosen_by_the_caller():
    ec = dg.ExperimentConfig(min_independent=2)
    model, store, clock, ml = dg.make_stack(0, ec, mix=0.0)
    ml.new_lifetime()
    lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=5150)
    verifier = LabVerifier(world)
    good = lt.verified
    r = ml.admit_objective(good, utility=1e6)
    assert not r.accepted and r.reason == "utility_mismatch" and r.entry.kind == "refused"
    r = ml.admit_objective(good, exp_ids=["exp_that_does_not_exist"])
    assert not r.accepted and r.reason == "exp_ids_not_decisions_of_objective"
    ok = ml.admit_objective(good, utility=verified_return(lt))
    assert ok.accepted and ok.entry.utility == verified_return(lt)
    assert set(ok.entry.exp_ids) == {x.exp_id for x in lt.records}
    clock.advance(60.0)
    later = verifier.verify(lab_task(world, objective_id=good.objective_id))
    r = ml.admit_objective(later, utility=1e6)
    assert not r.accepted and not any(e.utility >= 1e6 for e in ml.ledger.entries())
    assert ml.ledger.verify_chain() == (True, -1)


def test_verifier_signed_outcome_of_a_self_generated_goal_never_supervises_by_default():
    ec = dg.ExperimentConfig(min_independent=2)
    model, store, clock, ml = dg.make_stack(0, ec, mix=0.0)
    world = LabWorld(LabConfig(seed=3, **ec.lab), clock=clock)
    verifier = LabVerifier(world)
    own = TaskSpec(objective_id="self_goal_1", objective="reach 1 adopter (self-generated)", success={"target": 1},
                   limits=lab_task(world).limits, origin=Trust.SELF)
    v = verifier.verify(own)
    assert verifier.check_signature(v)
    assert ml.admit_objective(v).reason == "objective_origin_unknown"         # never registered: excluded
    assert ml.admit_objective(v, task=own).reason == "self_authored_objective"   # registered from its TaskSpec
    assert ml.admit_objective(v, objective_origin=Trust.OPERATOR).reason == "self_authored_objective"
    op = lab_task(world, objective_id="operator_goal")
    assert ml.admit_objective(verifier.verify(op), task=op).accepted
    with pytest.raises(ValueError):
        ml.admit_objective(verifier.verify(op), task=own)
    sd = ml.state_dict()                                           # origins travel with the checkpoint
    assert sd["objective_origins"] == {"self_goal_1": Trust.SELF, "operator_goal": Trust.OPERATOR}
    twin = MetaLearner(model, store, ml.cfg, policy=ml.policy, reward=ml.reward, clock=clock, meta_cfg=ml.mc,
                       inner_cfg=ml.inner.cfg, ledger=OutcomeLedger(verifier.check_signature))
    twin.load_state_dict(dict(sd, ledger_head=OutcomeLedger.GENESIS))
    assert twin.ledger.origin_of("self_goal_1") == Trust.SELF and twin.ledger.origin_of("operator_goal") == "operator"


def test_truncated_ledger_file_stops_meta_learning(tmp_path):
    """D8: a ledger file cut at an entry boundary has a consistent chain but no longer reaches its anchor."""
    ec = dg.ExperimentConfig(min_independent=2)
    path = str(tmp_path / "ledger.jsonl")
    model, store, clock, ml = dg.make_stack(1, ec, mix=0.0)
    ml.ledger = OutcomeLedger(ml.ledger.check_signature, path=path)          # a ledger with a file
    for i in range(3):
        ml.new_lifetime()
        lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=300 + i)
        assert ml.admit_objective(lt.verified, utility=verified_return(lt)).accepted
    lines = open(path).read().splitlines()
    open(path, "w").write("\n".join(lines[:1]) + "\n")
    cut = OutcomeLedger(ml.ledger.check_signature, path=path)
    assert cut.verify_chain() == (False, 1) and len(cut) == 1
    ml.ledger = cut
    with pytest.raises(IntegrityError):
        ml.meta_update()


# ======================================================================== D7: the cold-start shadow update
def test_cold_start_shadow_update_never_uses_the_on_policy_estimator_by_tolerance():
    """mix 1 live, the differentiated update uses the learned reward: the shadow policy did not act. Even with a
    tolerance that every propensity meets, the estimator is importance weighting."""
    ec = dg.ExperimentConfig(min_independent=3)
    model, store, clock, ml = dg.make_stack(5, ec, mix=1.0, on_policy_tol=1.0, min_ess=1.0)
    for i in range(4):
        ml.new_lifetime()
        lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=50 + i)
        assert ml.admit_objective(lt.verified).accepted
    rec = ml.meta_update()
    assert rec.applied and rec.estimator == ESTIMATOR_IMPORTANCE


def test_on_policy_only_while_psi_is_the_one_the_live_update_used():
    """Learned reward live (mix 0): rows collected after an update made with the current psi are on-policy; after
    a meta-update has changed psi, rows of an update made with the old psi are scored by importance weighting."""
    ec = dg.ExperimentConfig(min_independent=2)
    model, store, clock, ml = dg.make_stack(6, ec, mix=0.0, on_policy_tol=1.0, min_ess=1.0)
    for i in range(3):
        ml.new_lifetime()
        lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=60 + i)
        assert ml.admit_objective(lt.verified).accepted
    first = ml.meta_update()
    assert first.applied and first.estimator == "on_policy_query"
    ev = ml.events[-1]                                           # made before the meta-update, its rows come later
    for i in range(3):
        ml.new_lifetime()
        lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=70 + i)
        assert ml.admit_objective(lt.verified).accepted
    assert ev.reward_version != ml.reward.version and ev.query_ids
    second = ml.meta_update()
    assert second.applied and second.estimator in (ESTIMATOR_IMPORTANCE, "mixed_on_policy_and_importance")


# =============================================================== D12: no "learned" label before promotion
def test_reward_is_labelled_bootstrap_until_a_meta_update_was_applied_and_promoted():
    ec = dg.ExperimentConfig(min_independent=2)
    model, store, clock, ml = dg.make_stack(9, ec, mix=0.0)          # R_psi alone drives the policy, untrained
    ml.new_lifetime()
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=900)
    assert all(r.reward_version.startswith("bootstrap:mix=0.000:") for r in lt.records)
    assert ml.internal_rewards([])["label"] == "bootstrap"
    ml.mixer.note_meta_update()
    ml.mixer.min_independent = 2
    assert ml.mixer.update([0.2, 0.3, 0.25]).action == "lower" and ml.mixer.promoted
    ml.new_lifetime()
    lt2, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=901)
    assert all(r.reward_version.startswith("learned:mix=0.000:") for r in lt2.records)


def test_an_applied_meta_update_is_counted_for_the_label():
    ec = dg.ExperimentConfig(min_independent=2)
    model, store, clock, ml = dg.make_stack(10, ec, mix=0.0)
    for i in range(3):
        ml.new_lifetime()
        lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=910 + i)
        ml.admit_objective(lt.verified)
    assert ml.mixer.meta_updates_applied == 0
    rec = ml.meta_update()
    assert rec.applied and ml.mixer.meta_updates_applied == 1 and ml.mixer.label == "bootstrap"


# ============================================================ D4: the goal-free budget counts pending goals
def _cands(n=5):
    return [LearningGoalCandidate(key=f"k{i}", description=f"learn about channel {chr(97 + i)}", epistemic=1.0,
                                  achievable_progress=0.5, est_cost=Cost(seconds=10.0)) for i in range(n)]


def test_goal_free_budget_is_enforced_including_pending_goals():
    sel = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=2))
    assert len(sel.select(_cands(), k=5)) == 2                     # one call never exceeds the budget
    assert sel.select(_cands(), k=1) == [] and sel.budget.committed
    sel2 = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=2))
    picked = [g for _ in range(6) for g in sel2.select(_cands(), k=1)]   # earlier goals still pending
    assert len(picked) == 2 and sel2.budget.n_reserved == 2
    sel2.abandon(picked[0])                                          # not executed: the reservation comes back
    assert sel2.budget.n_reserved == 1 and len(sel2.select(_cands(), k=3)) == 1
    for g in [picked[1]]:
        sel2.record_result(g, Cost(seconds=12.0), epistemic_after=0.5, realised_progress=0.1)
    assert sel2.budget.n_decisions == 1 and sel2.budget.n_reserved == 1 and sel2.budget.spent.seconds == 12.0
    assert sel2.select(_cands(), k=3) == []                          # 1 spent + 1 pending = 2
    twin = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=2)); twin.load_state_dict(sel2.state_dict())
    assert twin.budget.n_reserved == 1 and twin.select(_cands(), k=3) == []   # restored reservations still count
    secs = SelfGoalSelector(GoalFreeBudget(seconds=25.0, decisions=100))
    assert len(secs.select(_cands(), k=5)) == 2                      # estimated seconds are reserved too
