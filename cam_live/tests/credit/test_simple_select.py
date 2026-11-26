"""Simple attribution statistics and the evidence-based estimator selector (spec 9.4; RC-08)."""
import json
import math

import pytest
import torch

from cam_live.contracts import ExperimentRecord, OutcomeObservation, OutcomeStatus, SimClock
from cam_live.credit.cocoa import HindsightModel, arm_contrast
from cam_live.credit.links import OutcomeLinker
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.credit.rudder import ReturnDecomposer
from cam_live.credit.select import EstimatorSelector, ExperimentEvidence
from cam_live.credit.simple import (SimpleAttribution, arm_difference, experiment_effects, student_t_cdf,
                                    student_t_ppf)


# ------------------------------------------------------------------------------------------------ statistics
def test_student_t_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1.0, 2.5, 7.0, 30.0, 400.0):
        for p in (0.6, 0.9, 0.975, 0.995):
            assert student_t_ppf(p, df) == pytest.approx(stats.t.ppf(p, df), rel=1e-8)
        for t in (-3.0, -0.4, 0.0, 1.7):
            assert student_t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), abs=1e-10)


def test_arm_difference_is_welch_and_covers_the_true_effect():
    stats = pytest.importorskip("scipy.stats")
    g = torch.Generator().manual_seed(0)
    a = (2.0 + 1.5 * torch.randn(23, generator=g)).tolist()
    b = (1.2 + 0.4 * torch.randn(41, generator=g)).tolist()
    d = arm_difference(a, b, confidence=0.9, randomized=True)
    ref = stats.ttest_ind(a, b, equal_var=False)
    ci = ref.confidence_interval(0.9)
    assert d.diff == pytest.approx(sum(a) / 23 - sum(b) / 41)
    assert (d.ci_low, d.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-7) and d.df == pytest.approx(ref.df)
    assert arm_difference([1.0], [2.0, 3.0]) is None                 # no interval from a single unit
    # coverage of the 95 % interval over repeated randomised experiments with a known effect
    cover = sig = 0
    for i in range(400):
        x = 0.5 + torch.randn(30, generator=g)
        y = torch.randn(30, generator=g) * 2.0
        r = arm_difference(x.tolist(), y.tolist(), randomized=True)
        cover += r.ci_low <= 0.5 <= r.ci_high
        sig += r.significant
    assert 0.92 <= cover / 400 <= 0.98, cover / 400


def test_experiment_effects_need_real_randomisation(tmp_path, make_record):
    st = SqliteExperienceStore(str(tmp_path / "e.sqlite"), durable=False)
    linker = OutcomeLinker(st, clock=SimClock(1e6))
    g = torch.Generator().manual_seed(1)
    with st.batch():
        for i in range(60):
            arm = "B" if i % 2 else "A"
            rec = make_record(i, experiment_id="x1", arm=arm, n_cand=2, chosen=i % 2, probs=[0.5, 0.5])
            st.append(rec)
            val = float(torch.poisson(torch.tensor(6.0 if arm == "B" else 2.0), generator=g))
            linker.reconcile(outcomes=[OutcomeObservation(rec.pending_queries[0].query_id, OutcomeStatus.OBSERVED, val,
                                                          observed_at=rec.decision.ts + 50.0)])
    # without a registered randomised experiment the difference is reported but is not experimental evidence
    (d,) = experiment_effects(st, "x1", control="A")
    assert not d.randomized and d.n_treatment == d.n_control == 30
    st.put_experiment(ExperimentRecord("x1", 0.0, "does B change the count", "B > A", arms=["A", "B"],
                                       assignment="randomized", comparison="A"))
    (d,) = experiment_effects(st, "x1")
    assert d.randomized and d.treatment == "B" and d.control == "A" and d.significant and 2.5 < d.diff < 5.5
    # the simple attribution reproduces the measured difference when every outcome is linked by its declared query
    SimpleAttribution(st).apply()
    sel = EstimatorSelector(min_experiments=1)
    (ev,) = sel.add_from_store(st, "x1", ["simple"])
    assert ev.estimates["simple"] == pytest.approx(d.diff) and ev.randomized


# ------------------------------------------------------------------------------------------------ selector logic
def _evidence(i, diff, se, estimates, randomized=True, est_se=None):
    return ExperimentEvidence(f"x{i}", "B", "A", diff, se, diff - 1.96 * se, diff + 1.96 * se, 50, 50, randomized,
                              dict(estimates), dict(est_se or {}))


def test_selector_starts_simple_and_promotes_only_on_evidence(tmp_path):
    log = str(tmp_path / "selection.jsonl")
    sel = EstimatorSelector(log_path=log, clock=SimClock(100.0), min_experiments=5)
    r0 = sel.select()
    assert r0.default_after == "simple" and not r0.changed and "no randomised experiment yet" in " ".join(r0.reasons)
    effects = [1.0, -0.8, 0.5, 2.0, 0.0, 1.2, -1.5]
    g = torch.Generator().manual_seed(0)
    for i, e in enumerate(effects):
        measured = e + 0.1 * float(torch.randn(1, generator=g))
        good = e + 0.05 * float(torch.randn(1, generator=g))
        sel.add_evidence(_evidence(i, measured, 0.1, {"simple": measured * 0.2, "good": good, "latest": 0.0,
                                                      "overclaim": 3.0 * e}, est_se={"good": 0.05}))
        rec = sel.select()
        if i < 4:                                  # fewer than five randomised experiments: nothing is promoted
            assert rec.default_after == "simple" and any("only" in r for r in rec.reasons)
    assert sel.default == "good" and rec.default_after == "good"
    sc = rec.scores
    assert sc["good"]["passes"] and not sc["latest"]["passes"] and not sc["overclaim"]["passes"] and not sc["simple"]["passes"]
    assert sc["latest"]["sign_agreement"] < 0.8 and sc["overclaim"]["mean_sq_z"] > 3.0
    changed = [r for r in sel.history if r.changed]
    assert len(changed) == 1 and changed[0].n_randomized_experiments == 5
    # the selection and its evidence are logged
    lines = [json.loads(l) for l in open(log)]
    assert len(lines) == len(sel.history) == 8
    assert lines[-1]["selection"]["default_after"] == "good" and len(lines[-1]["evidence"]) == 7
    assert lines[-1]["selection"]["gates"]["min_experiments"] == 5
    # non-randomised comparisons never count
    sel2 = EstimatorSelector(min_experiments=2)
    for i, e in enumerate(effects):
        sel2.add_evidence(_evidence(i, e, 0.1, {"good": e, "simple": 0.0}, randomized=False))
    rec2 = sel2.select()
    assert rec2.default_after == "simple" and rec2.n_randomized_experiments == 0 and len(rec2.excluded_non_randomized) == 7
    # later evidence contradicts the promoted estimator: the default falls back to the simple attribution
    for i in range(10, 22):
        sel.add_evidence(_evidence(i, 1.0, 0.05, {"good": -1.0, "simple": 0.2}))
    back = sel.select()
    assert back.default_before == "good" and back.default_after == "simple" and back.changed
    # state round trip
    sel3 = EstimatorSelector()
    sel3.load_state_dict(json.loads(json.dumps(sel.state_dict())))
    assert sel3.default == "simple" and len(sel3.evidence) == len(sel.evidence)


def test_an_accurate_estimator_is_not_promoted_when_the_default_is_already_right():
    sel = EstimatorSelector(min_experiments=3)
    for i, e in enumerate([1.0, -1.0, 0.7, 1.4]):
        sel.add_evidence(_evidence(i, e, 0.1, {"simple": e, "rudder": e + 0.02}))
    rec = sel.select()
    assert rec.default_after == "simple" and rec.scores["rudder"]["passes"]
    assert any("rudder not promoted" in r and "common experiments" in r for r in rec.reasons)


# ------------------------------------------------------------------------------------------------ RUDDER and COCOA exercised
T, N_CTX, N_PER = 5, 6, 200
EFFECT = [0.0, 0.5, 1.0, 1.5, -1.0, 2.0]
EXP_STEP = 1


def _world(seed=0):
    """Six contexts, each a randomised two-arm experiment at decision 1 of a five-decision trajectory; the arm's
    effect arrives with the delayed final outcome, together with the effect of a later non-randomised decision."""
    g = torch.Generator().manual_seed(seed)
    n = N_CTX * N_PER
    ctx = torch.arange(N_CTX).repeat_interleave(N_PER)
    act = (torch.rand(n, T, generator=g) < 0.3).long()                    # policy: action 1 with probability 0.3
    act[:, EXP_STEP] = (torch.rand(n, generator=g) < 0.5).long()          # the experiment: 50 / 50
    G = torch.tensor(EFFECT)[ctx] * act[:, EXP_STEP] + 0.3 * act[:, 3] + 0.3 * torch.randn(n, generator=g)
    x = torch.zeros(n, T, N_CTX + T + 1)
    x[torch.arange(n), :, ctx] = 1.0
    x[:, torch.arange(T), N_CTX + torch.arange(T)] = 1.0
    x[:, :, -1] = act.float() * 2 - 1
    probs = torch.tensor([0.7, 0.3]).expand(n, T, 2).clone()
    probs[:, EXP_STEP] = 0.5
    return ctx, act, G.double(), x, probs


def _outcome_features(G):
    centers = torch.linspace(-2.0, 3.5, 12)
    return torch.cat([G.float()[:, None], torch.exp(-0.5 * ((G.float()[:, None] - centers) / 0.5) ** 2)], -1)


def test_rudder_and_cocoa_are_exercised_and_compared_with_randomised_differences(tmp_path):
    torch.manual_seed(0)
    ctx, act, G, x, probs = _world()
    n = x.shape[0]
    mask, matured = torch.ones(n, T, dtype=torch.bool), torch.ones(n, dtype=torch.bool)
    gen = torch.Generator().manual_seed(1)
    # RUDDER: return decomposition
    dec = ReturnDecomposer(x.shape[-1], d_model=32)
    dec.fit_scale(G, matured)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400)
    for _ in range(400):
        i = torch.randint(0, n, (128,), generator=gen)
        loss, _ = dec.loss(x[i], mask[i], G[i], matured[i])
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    red = dec.redistribute(x, mask, G, matured)
    assert float(red.conservation_error(G).max()) < 1e-9
    # COCOA: hindsight model over the two logged candidates of every decision
    b = x[:, :, :-1].reshape(n * T, -1)
    u = _outcome_features(G).repeat_interleave(T, 0)
    taken, logp = act.reshape(-1), probs.reshape(n * T, 2).log()
    cand, cmask = torch.eye(2)[None].expand(n * T, -1, -1), torch.ones(n * T, 2, dtype=torch.bool)
    hm = HindsightModel(b.shape[1], u.shape[1], 2, d_hidden=64, d_key=16, n_members=2, n_elapsed=0)
    opt = torch.optim.Adam(hm.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    for _ in range(500):
        i = torch.randint(0, n * T, (512,), generator=gen)
        loss, _ = hm.loss(b[i], u[i], cand[i], cmask[i], taken[i], logp[i], generator=gen)
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    rows = torch.arange(n) * T + EXP_STEP
    cc = hm.contribution(b[rows], u[rows], cand[rows], cmask[rows], logp[rows])
    assert cc.status == ["supported"] * n

    def evidence(simple_credit):
        """Measured arm differences per context and each estimator's estimate of them."""
        out = []
        for c in range(N_CTX):
            sel_c = ctx == c
            arm_b, arm_a = sel_c & (act[:, EXP_STEP] == 1), sel_c & (act[:, EXP_STEP] == 0)
            measured = arm_difference(G[arm_b].tolist(), G[arm_a].tolist(), randomized=True, treatment_name="B",
                                      control_name="A", experiment_id=f"x{c}")
            est, se = {}, {}
            for name, credit in (("simple", simple_credit), ("rudder", red.diff[:, EXP_STEP]),
                                 ("latest", torch.zeros(n, dtype=torch.float64))):   # 'the last action did it'
                d = arm_difference(credit[arm_b].tolist(), credit[arm_a].tolist())
                est[name], se[name] = d.diff, d.se
            est["cocoa"], se["cocoa"], n_sup = arm_contrast(cc.w[sel_c], G[sel_c], treatment=1, control=0)
            assert n_sup == N_PER
            out.append(ExperimentEvidence.from_difference(measured, est, se))
        return out

    # (i) every outcome is linked to the experiment decision by evidence: the simple attribution is already right
    log = str(tmp_path / "selection.jsonl")
    clear = EstimatorSelector(log_path=log, clock=SimClock(1.0))
    for ev in evidence(simple_credit=G):
        clear.add_evidence(ev)
    r1 = clear.select()
    assert r1.default_after == "simple"
    assert r1.scores["rudder"]["passes"] and r1.scores["cocoa"]["passes"], json.dumps(r1.scores)
    assert not r1.scores["latest"]["passes"]
    # (ii) the outcome can only be attributed to the trajectory as a whole (one shared attribution window): the
    # equal split under-states every effect, and a learned estimator validated against the experiments is promoted
    ambiguous = EstimatorSelector(log_path=log, clock=SimClock(2.0))
    evs = evidence(simple_credit=G / T)
    for ev in evs:
        ambiguous.add_evidence(ev)
    r2 = ambiguous.select()
    assert not r2.scores["simple"]["passes"] and not r2.scores["latest"]["passes"]
    assert r2.scores["rudder"]["passes"] and r2.scores["cocoa"]["passes"], r2.scores
    assert r2.default_after in ("rudder", "cocoa") and r2.changed
    truth = torch.tensor(EFFECT, dtype=torch.float64)
    err = {k: float((torch.tensor([e.estimates[k] for e in evs]) - truth).abs().max()) for k in ("rudder", "cocoa")}
    assert err["rudder"] < 0.2 and err["cocoa"] < 0.35, err
    assert len(open(log).readlines()) == 2
    print("SELECTION", json.dumps({"default_clear": r1.default_after, "default_ambiguous": r2.default_after,
                                   "max_abs_err_vs_true_effect": err,
                                   "scores": {k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in s.items()
                                                  if m in ("n", "rmse", "mean_sq_z", "sign_agreement", "passes")}
                                              for k, s in r2.scores.items()}}))
