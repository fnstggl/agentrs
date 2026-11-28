"""IMP (amendment 2026-11-28 item 2): multi-step imagination in decision-making.

Implementation correctness of the pieces that make imagined consequences part of action selection:
the imagined preference (what imagination adds beyond the one-step ranking), the imagination calibration on
verified multi-step windows (no systematic weight until imagined differences predict verified progress), the rule
``thompson_policy`` with and without imagination (exact probabilities; the ablation draws the same numbers), the
multi-step look-ahead of ``LatentPlanner.plan`` on a world whose dynamics the transition head has learned (an
enabling action that pays off only one or two steps later), the value-accuracy horizon criterion, and the feed of
verified windows from a store."""
import math
import random
from types import SimpleNamespace

import pytest
import torch

from cam_live.contracts import ActionSpec, Cost
from cam_live.predict import policy as P
from cam_live.predict.imagination import HorizonController, value_consistency_error
from cam_live.predict.planner import ImaginedEvaluation, LatentPlanner
from cam_live.predict.world import LatentTransition

F64 = torch.float64


# ---------------------------------------------------------------------------------------------- imagined preference
def test_imagined_preference_is_in_one_step_units_and_zero_without_differences():
    mask = torch.tensor([[True, True, True, False]])
    q = torch.tensor([[1.0, 2.0, 3.0, 50.0]], dtype=F64)
    # every candidate continues equally well: imagination adds nothing
    same = torch.full((5, 1, 4), 7.0, dtype=F64)
    ms, mm = P.imagined_preference(same, q, mask)
    assert torch.allclose(ms, torch.zeros_like(ms)) and torch.allclose(mm, torch.zeros_like(mm))
    # a continuation advantage counts in units of the one-step spread over the feasible candidates; with weight 1
    # the rule's score ranks by the standardised long-term value (one-step + continuation)
    c = torch.zeros(3, 1, 4, dtype=F64)
    c[:, 0, 0] = 3.0
    c[:, 0, 3] = 1e9                                                              # infeasible: ignored
    _, mm = P.imagined_preference(c, q, mask)
    sd = float(q[0, :3].std(unbiased=False))
    assert torch.allclose(mm[0, :3], torch.tensor([2.0, -1.0, -1.0], dtype=F64) / sd, atol=1e-12)
    assert mm[0, 3] == 0.0
    zq = (q[0, :3] - q[0, :3].mean()) / sd
    long = q[0, :3] + c[0, 0, :3]
    assert torch.equal((zq + mm[0, :3]).argsort(), long.argsort())
    # winsorised like the other features of the rule
    big = torch.zeros(1, 1, 4, dtype=F64)
    big[0, 0, 0] = 1e6
    ms, mm = P.imagined_preference(big, q, mask)
    assert float(ms.abs().max()) <= P.Z_CLIP and float(mm.abs().max()) <= P.Z_CLIP


# ---------------------------------------------------------------------------------------------- calibration
def _fit(cal, h, beta_im, n=200, seed=0):
    rng = random.Random(seed)
    for i in range(n):
        z, m = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 1.0 * z + beta_im * m + rng.gauss(0, 0.3)
        cal.observe(f"d{h}_{i}", h, z, m, y)


def test_imagination_weight_is_learned_from_verified_windows_per_horizon():
    cal = P.ImaginationCalibration()
    g = torch.Generator().manual_seed(0)
    prior = P.policy_weight(cal.sample(1, 4000, g))
    assert float(prior.mean()) < 0.5 and 0.3 < float((prior > 0).double().mean()) < 0.7     # exploratory prior only
    _fit(cal, 1, beta_im=0.0)                       # imagined differences do not predict verified progress
    _fit(cal, 2, beta_im=1.5, seed=1)               # ... they do, at horizon 2
    lam1 = P.policy_weight(cal.sample(1, 4000, g))
    lam2 = P.policy_weight(cal.sample(2, 4000, g))
    s1, s2 = cal.summary(1), cal.summary(2)
    # coefficients are in units of the verified scale; their ratio is the imagination weight of the rule
    assert abs(s1["beta_im"]) < 0.1 and s2["beta_im"] / s2["beta_q"] == pytest.approx(1.5, abs=0.15)
    assert float(lam1.mean()) < 0.1 and float(lam2.mean()) > 1.0
    assert cal.summary(3)["n"] == 0 and cal.summary(3)["beta_im"] == 0.0           # a new horizon starts at the prior
    assert not cal.observe("d1_0", 1, 0.0, 0.0, 5.0)                                # each decision once
    st = cal.state_dict()
    cal2 = P.ImaginationCalibration()
    cal2.load_state_dict(st)
    assert cal2.summary(2) == s2 and len(cal2) == len(cal)


# ---------------------------------------------------------------------------------------------- the rule
def _evals(q, cont, feasible=None, n_ts=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (qi, ci) in enumerate(zip(q, cont)):
        ex = {"q_goal": qi, "net_cost": 0.0, "q_goal_epistemic_std": 0.05, "policy_logit": 0.0,
              P.IM_MEAN: float(sum(ci) / len(ci)), P.IM_HORIZON: 1.0}
        ex.update({f"ts_{s}": qi + 0.05 * float(torch.randn(1, generator=g)) for s in range(n_ts)})
        ex.update({f"{P.IM_SAMPLE}{p}": float(v) for p, v in enumerate(ci)})
        a = ActionSpec(action_id=f"a{i}", operation=f"op{i}", arguments={}, est_cost=Cost())
        out.append(SimpleNamespace(action=a, extras=ex, feasible=True if feasible is None else feasible[i],
                                   value=0.0, p_success=0.5, uncertainty=0.05, voi=0.0, predictions=[], cost=Cost()))
    return out


class _Heads:
    def __init__(self, select=True):
        from cam_live.predict.heads import PredictConfig
        self.cfg = PredictConfig(imagination_select=select)
        self.policy = torch.nn.Linear(1, 1)
        self.transition = torch.nn.Linear(1, 1)


def test_rule_uses_imagination_only_through_the_learned_weight_and_records_both_distributions():
    # candidate 0 is best now; candidate 2 is worse now but imagined to enable much more afterwards
    q = [3.0, 2.0, 1.0]
    cont = [[0.0, 0.1, -0.1], [0.0, 0.0, 0.0], [6.0, 5.5, 6.5]]
    heads = _Heads(select=True)
    dm = P.DecisionMaker(None, heads=heads, seed=0)
    p_prior, sc = dm.probabilities(_evals(q, cont), mode=P.RULE_THOMPSON_POLICY)
    im = sc["selection"]["imagination"]
    assert im["enabled"] and im["calibration"]["n"] == 0
    assert abs(sum(im["with_probabilities"]) - 1) < 1e-12 and abs(sum(im["without_probabilities"]) - 1) < 1e-12
    # trusted imagination (verified windows where the imagined preference predicted progress) moves the choice
    _fit(dm.imagination, 1, beta_im=2.0)
    p_trust, sc = dm.probabilities(_evals(q, cont), mode=P.RULE_THOMPSON_POLICY)
    im_t = sc["selection"]["imagination"]
    assert im_t["weight"]["mean"] > 1.0 and im_t["share_tv"] > 0.5
    assert float(p_trust[2]) > 0.5 > float(p_prior[2])
    # discredited imagination has no say however large the imagined differences
    heads2 = _Heads(select=True)
    dm2 = P.DecisionMaker(None, heads=heads2, seed=0)
    _fit(dm2.imagination, 1, beta_im=0.0)
    for c in dm2.imagination.by_h.values():                    # verified windows contradicting imagination
        c.obs = [(z, m, y - 0.8 * m) for z, m, y in c.obs]
        c._post = None
    p_bad, sc = dm2.probabilities(_evals(q, cont), mode=P.RULE_THOMPSON_POLICY)
    assert sc["selection"]["imagination"]["weight"]["mean"] < 0.05 and float(p_bad[2]) < 0.05
    # the ablation (imagination weight 0) draws the same numbers and uses the distribution without imagination
    heads3 = _Heads(select=False)
    dm3 = P.DecisionMaker(None, heads=heads3, seed=0)
    _fit(dm3.imagination, 1, beta_im=2.0)
    p_off, sc = dm3.probabilities(_evals(q, cont), mode=P.RULE_THOMPSON_POLICY)
    im_o = sc["selection"]["imagination"]
    assert not im_o["enabled"]
    floor = dm3.cfg.eps_floor
    assert torch.allclose(p_off, P.apply_floor(torch.tensor([im_o["without_probabilities"]], dtype=F64),
                                               torch.tensor([[True] * 3]), floor)[0])
    assert im_o["share_tv"] == pytest.approx(im_t["share_tv"], abs=1e-12)         # same draws in both arms
    assert float(p_off[2]) < 0.05
    # infeasible candidates keep exactly zero, whatever imagination says
    p_inf, _ = dm.probabilities(_evals(q, cont, feasible=[True, True, False]), mode=P.RULE_THOMPSON_POLICY)
    assert float(p_inf[2]) == 0.0 and abs(float(p_inf.sum()) - 1) < 1e-12
    # without imagination numbers on every candidate the rule is unchanged
    ev = _evals(q, cont)
    del ev[1].extras[P.IM_MEAN]
    _, sc = dm.probabilities(ev, mode=P.RULE_THOMPSON_POLICY)
    assert "imagination" not in sc["selection"]


# ---------------------------------------------------------------------------------------------- planner
D = 6
KEY1, KEY2, DOOR, PLAIN = 0, 1, 2, 3


class _ToyModel:
    """A latent world with an enabling chain: KEY1 sets belief dim 0, KEY2 (only after KEY1) sets dim 1, the DOOR
    pays 10 x dim 1 (clipped to [0, 1]), PLAIN pays 1 always. decision_latent concatenates the belief into the
    action embedding's free dims (z = e_a + b); the transition head is a real LatentTransition trained on samples
    of these dynamics."""

    def __init__(self):
        self.e_a = torch.zeros(1, 4, D)
        for i in range(4):
            self.e_a[0, i, 2 + i] = 1.0
        self.e_g = torch.zeros(1, D)

    def decision_latent(self, b, e_a, e_g):
        return e_a + b.unsqueeze(1)

    @staticmethod
    def step_true(b, a):
        nb = b.clone()
        if a == KEY1:
            nb[0] = 1.0
        if a == KEY2 and b[0] > 0.5:
            nb[1] = 1.0
        return nb


def _toy_value(z):                                                             # [R, N, D] -> [R, N] net goal values
    b = z.clone()
    for i in range(4):
        b[:, i, 2 + i] -= 1.0
    v = torch.zeros(z.shape[0], 4, dtype=F64)
    v[:, DOOR] = 10.0 * b[:, DOOR, 1].clamp(0.0, 1.0).double()
    v[:, PLAIN] = 1.0
    return v


class _ToyPredictor:
    def __init__(self, model, b):
        z = model.decision_latent(b, model.e_a, model.e_g)
        self.last = {"b": b, "e_a": model.e_a, "e_g": model.e_g, "z": z, "world": {"toy": True}}

    def goal_values(self, z, member=None):
        return _toy_value(z)


def _trained_transition(model):
    torch.manual_seed(0)
    tr = LatentTransition(D, hidden=64, n_members=3, d_time=8, seed=0)
    opt = torch.optim.Adam(tr.parameters(), lr=3e-3)
    states = [torch.tensor(v, dtype=torch.float32) for v in ([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                                                             [1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])]
    from cam_live.predict.world import transition_nll
    for _ in range(1500):
        bs, zs, ns = [], [], []
        for _ in range(64):
            b = random.choice(states)
            a = random.randrange(4)
            bs.append(b)
            zs.append(model.decision_latent(b[None], model.e_a[:, a:a + 1], model.e_g)[0, 0])
            ns.append(_ToyModel.step_true(b, a))
        b, z, nb = torch.stack(bs), torch.stack(zs), torch.stack(ns)
        loss, _ = transition_nll(tr(z, torch.full((64,), 3600.0), b), nb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tr


@pytest.fixture(scope="module")
def toy():
    model = _ToyModel()
    tr = _trained_transition(model)
    from cam_live.predict.heads import PredictConfig
    heads = SimpleNamespace(cfg=PredictConfig(imagination_particles=3, max_imagination_horizon=3),
                            transition=tr, world=SimpleNamespace(cfg=SimpleNamespace(n_members=1)),
                            evaluator=_ToyEvaluator(), policy=None)
    return model, heads


class _ToyEvaluator:
    def __call__(self, b, e_g, hz, scale=None):
        return SimpleNamespace(p_success=torch.full((b.shape[0],), 0.5))


def _plan(model, heads, horizon, b0):
    ctl = HorizonController(max_horizon=3, threshold=0.5, min_count=1, criterion="value")
    ctl.update_value([0.0] * horizon + [float("inf")] * (3 - horizon))
    planner = LatentPlanner.__new__(LatentPlanner)
    planner.model, planner.heads, planner.cfg, planner.controller = model, heads, heads.cfg, ctl
    planner.calibration = P.ImaginationCalibration()
    cands = [ActionSpec(action_id=f"t{i}", operation=n, arguments={}, est_cost=Cost(seconds=3600.0))
             for i, n in enumerate(("key1", "key2", "door", "plain"))]
    evs = [SimpleNamespace(feasible=True, extras={"net_cost": 0.0}) for _ in cands]
    return planner.plan([], "", cands, now=0.0, deadline=None, predictor=_ToyPredictor(model, b0), evaluations=evs)


def test_planner_looks_ahead_through_learned_dynamics_and_needs_the_horizon_for_longer_chains(toy):
    model, heads = toy
    start = torch.zeros(1, D)
    h1 = _plan(model, heads, 1, start)
    h2 = _plan(model, heads, 2, start)
    assert all(isinstance(p, ImaginedEvaluation) and p.imagined and p.value_source == "q_goal" for p in h1 + h2)
    assert {p.horizon_steps for p in h1} == {1} and {p.horizon_steps for p in h2} == {2}
    # one imagined transition: nothing is enabled yet from the start (two keys are needed): PLAIN continues best
    c1 = [p.cont_mean for p in h1]
    assert all(abs(c - 1.0) < 0.2 for c in c1) and all(p.next_index == PLAIN for p in h1)
    # two imagined transitions: KEY1 -> KEY2 -> DOOR is found; only the chain that starts with KEY1 reaches it
    c2 = [p.cont_mean for p in h2]
    assert c2[KEY1] == max(c2) and c2[KEY1] > 8.0 and max(c2[KEY2], c2[DOOR], c2[PLAIN]) < 3.0
    assert h2[KEY1].next_index == KEY2 and h2[KEY1].next_share == 1.0
    # after KEY1 (belief dim 0 set) one transition suffices: KEY2 now enables the DOOR
    after = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    h1b = _plan(model, heads, 1, after)
    c = [p.cont_mean for p in h1b]
    assert c[KEY2] > 8.0 and c[KEY2] == max(c) and h1b[KEY2].next_index == DOOR
    # the planner's preference reaches the rule: with a calibration that trusts it, the enabling action is chosen
    # although its one-step value is the lowest
    q = [0.0, 0.0, 0.0, 1.0]
    ev = _evals(q, [list(p.cont_samples) for p in h2])
    for e, p in zip(ev, h2):
        e.extras.update(p.extras())
    dm = P.DecisionMaker(None, heads=_Heads(select=True), seed=0)
    _fit(dm.imagination, 2, beta_im=2.0)
    pr, sc = dm.probabilities(ev, mode=P.RULE_THOMPSON_POLICY)
    assert float(pr[KEY1]) > 0.8 and sc["selection"]["imagination"]["horizon"] == 2
    dm_off = P.DecisionMaker(None, heads=_Heads(select=False), seed=0)
    _fit(dm_off.imagination, 2, beta_im=2.0)
    pr_off, _ = dm_off.probabilities(ev, mode=P.RULE_THOMPSON_POLICY)
    assert float(pr_off[PLAIN]) > 0.8 and float(pr_off[KEY1]) < 0.05
    # information value: non-negative, zero where all particles agree on a candidate that cannot beat the incumbent
    assert all(p.info_value >= 0.0 for p in h2)


def test_deadline_ends_the_imagined_sequence(toy):
    model, heads = toy
    ctl = HorizonController(max_horizon=3, threshold=0.5, min_count=1, criterion="value")
    ctl.update_value([0.0, 0.0, float("inf")])
    planner = LatentPlanner.__new__(LatentPlanner)
    planner.model, planner.heads, planner.cfg, planner.controller = model, heads, heads.cfg, ctl
    planner.calibration = P.ImaginationCalibration()
    cands = [ActionSpec(action_id=f"t{i}", operation=n, arguments={}, est_cost=Cost(seconds=3600.0))
             for i, n in enumerate(("key1", "key2", "door", "plain"))]
    evs = [SimpleNamespace(feasible=True, extras={"net_cost": 0.0}) for _ in cands]
    # 1.5 actions of time left: the first action fits, a continuation that ends after the deadline adds nothing
    out = planner.plan([], "", cands, now=0.0, deadline=1.5 * 3600.0, predictor=_ToyPredictor(model, torch.zeros(1, D)),
                       evaluations=evs)
    assert all(p.cont_mean == 0.0 for p in out)


# ---------------------------------------------------------------------------------------------- horizon criterion
def test_value_criterion_horizon_and_old_controller_states():
    c = HorizonController(max_horizon=3, threshold=0.5, min_count=2, criterion="value")
    assert c.allowed() == 1
    c.update(torch.tensor([2.0, 2.0, 2.0], dtype=F64))                # the belief itself is imagined badly ...
    c.update_value([0.1, 0.2, 0.9])                                    # ... the values the decision needs are fine
    assert c.allowed() == 1                                            # one measurement is not enough
    c.update_value([0.1, 0.3, float("nan")])
    assert c.allowed() == 2 and c.state()["allowed_by"] == {"latent": 1, "value": 2}
    c.update_value([0.1, 5.0, 5.0])                                    # the error rises: the horizon shrinks
    c.update_value([0.1, 5.0, 5.0])
    assert c.allowed() == 1
    # a controller state written before the value criterion existed loads (value measurements start empty)
    old = {"allowed": 1, "err": [1.4, 1.5, 1.6, 1.7], "count": [9, 9, 9, 9], "threshold": 0.5}
    n = HorizonController(max_horizon=3, criterion="value")
    n.load_state(old)
    assert n.err == [1.4, 1.5, 1.6] and n.value_count == [0, 0, 0] and n.allowed() == 1
    with pytest.raises(ValueError):
        HorizonController(criterion="vibes")


def test_value_consistency_error_is_zero_for_exact_dynamics_and_relative_to_candidate_spread():
    model = _ToyModel()

    class Exact(torch.nn.Module):
        def forward(self, z, dt, b):
            a = int((z[0] - b[0])[2:].argmax())
            nb = _ToyModel.step_true(b[0], a)[None]
            return SimpleNamespace(mean=nb)

    class Frozen(torch.nn.Module):
        def forward(self, z, dt, b):
            return SimpleNamespace(mean=b)

    acts = [KEY1, KEY2, DOOR]
    bs = [torch.zeros(D)]
    for a in acts:
        bs.append(_ToyModel.step_true(bs[-1], a))
    b_seq = torch.stack(bs)[None]
    e_a = torch.stack([model.e_a[0, a] for a in acts])[None]
    q_fn = lambda b, t: _toy_value(model.decision_latent(b, model.e_a, model.e_g))[0]     # noqa: E731
    num, den, per = value_consistency_error(Exact(), model.decision_latent, q_fn, b_seq, e_a, model.e_g,
                                            torch.ones(1, 3), 2)
    assert float(num.sum()) == 0.0 and float(den[0]) > 0
    num_f, den_f, per_f = value_consistency_error(Frozen(), model.decision_latent, q_fn, b_seq, e_a, model.e_g,
                                                  torch.ones(1, 3), 2)
    assert torch.allclose(num_f, per_f) and float(num_f[1]) > 0      # not imagining == the persistence baseline
    assert float(per[1]) == float(per_f[1])


# ---------------------------------------------------------------------------------------------- verified windows
class _Store:
    def __init__(self, recs):
        self.recs = recs

    def trajectory(self, objective_id):
        return [r for r in self.recs if r.objective_id == objective_id]


def _rec(exp_id, obj, progress, horizon=1, cont=(0.0, 1.0), goal_q=True):
    from cam_live.contracts import OutcomeKind, OutcomeQuery
    cands = []
    for i, c in enumerate(cont):
        ex = {"q_goal": float(i), "net_cost": 0.0, P.IM_MEAN: c, P.IM_HORIZON: float(horizon), f"{P.IM_SAMPLE}0": c}
        cands.append(SimpleNamespace(feasible=True, extras=ex, value=0.0))
    diag = {} if progress is None else {"reward_signals": {"verified_progress_delta": progress}}
    dec = SimpleNamespace(candidates=cands, chosen_index=0, diagnostics=diag)
    q = OutcomeQuery("q" + exp_id, "m", OutcomeKind.COUNT, goal_relevant=goal_q)
    res = SimpleNamespace(expected_outcomes=[q])
    return SimpleNamespace(exp_id=exp_id, objective_id=obj, decision=dec, result=res, pending_queries=[],
                           post_item_refs=["x"])


def test_verified_windows_feed_the_calibration_once_and_only_when_final():
    planner = LatentPlanner.__new__(LatentPlanner)
    planner.calibration = P.ImaginationCalibration()
    recs = [_rec("e1", "o1", 1.0), _rec("e2", "o1", None), _rec("e3", "o1", 2.0, goal_q=False)]
    recs[2].decision.diagnostics = {}                                  # nothing attributable: contributes 0
    store = _Store(recs)
    out = planner.refresh_calibration(store, [recs])
    assert out["added"] == 0 and out["pending"] == 3                   # e1's window needs e2, still pending
    recs[1].decision.diagnostics = {"reward_signals": {"verified_progress_delta": 3.0}}
    out = planner.refresh_calibration(store, [recs])
    cal = planner.calibration.get(1)
    assert out["added"] == 2 and dict(zip(cal.keys, [o[2] for o in cal.obs])) == {"e1": 4.0, "e2": 3.0}
    # e3's window runs past the end of o1: final once a later objective is seen
    recs.append(_rec("f1", "o2", None))
    out = planner.refresh_calibration(store, [recs[3:]])
    assert out["added"] == 1 and "e3" in cal.keys and cal.obs[cal.keys.index("e3")][2] == 0.0
    assert planner.refresh_calibration(store, [recs])["added"] == 0 and len(cal) == 3      # never twice
    # features come from what the decision recorded: z of the chosen candidate, its imagined preference, horizon
    f = P.ImaginationCalibration.features(recs[0].decision)
    assert f is not None and f[2] == 1 and math.isfinite(f[0]) and math.isfinite(f[1])
