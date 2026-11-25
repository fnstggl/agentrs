"""Support/query fast-adaptation loss over k in {0..64} (strict separation, per-episode support, padding, gradients
through adaptation, training reduces it), the contract-module adapter (queries conditioned on their actions through
the world model's prior), J_meta credit across trials and support sizes with config / stage gating of the LP term,
and the multi-trial episode harness (nothing an earlier trial revealed is lost)."""
import copy
import math

import pytest
import torch
import torch.nn as nn

from learn_doubles import KalmanDouble, regression_sequence
from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import Action, FastState, ObsKind, TokenKind, TokenSet
from superlearner.cam_unified.learn.diagnostics import DelayedRewardAdapter, ProbeBanditAdapter
from superlearner.cam_unified.learn.explore import hindsight_learning_progress
from superlearner.cam_unified.learn.meta import (SUPPORT_SIZES, MetaRLObjective, SupportQueryLoss, TrialRunner,
                                                 belief_callables_from_modules, support_belief, tree_where)
from superlearner.cam_unified.learn.ppo import ActorCriticObjective
from superlearner.cam_unified.learn.trainer import CombinedTrainer
from superlearner.cam_unified.learn.trajectory import Trajectory
from superlearner.cam_unified.testing import DoubleOutcomeHead, rand_sequence, tiny_config, zero_fast_state


def _run(loss_fn, kd, steps, **kw):
    return loss_fn(steps, kd.init, kd.update, kd.log_prob, **kw)


def test_support_sizes_and_meta_loss_decreases_on_toy_task():
    torch.manual_seed(0)
    assert SUPPORT_SIZES == (0, 1, 2, 4, 8, 16, 32, 64)
    kd = KalmanDouble(4, init_W=0.3)
    L = SupportQueryLoss(max_query=16)
    steps, _ = regression_sequence(32, 66, seed=100)
    with torch.no_grad():
        before = _run(L, kd, steps)
    assert sorted(before["per_k"]) == list(SUPPORT_SIZES) and before["skipped_k"] == []
    opt = torch.optim.Adam(kd.parameters(), lr=0.05)
    for i in range(80):
        tr, _ = regression_sequence(32, 66, seed=i)
        out = _run(L, kd, tr)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    with torch.no_grad():
        after = _run(L, kd, steps)
    assert after["loss"].item() < before["loss"].item() - 0.5, (before["per_k"], after["per_k"])
    pk = after["per_k"]
    assert pk[64] < pk[8] < pk[1] < pk[0]          # adaptation helps more with more support
    assert 0.03 < math.exp(kd.log_noise.item()) < 0.5    # moved from e^1 towards the true noise variance 0.09


def test_support_query_strict_separation():
    kd = KalmanDouble(4)
    L = SupportQueryLoss(ks=(0, 2, 4))
    steps, _ = regression_sequence(8, 10, seed=1)
    base = L(steps, kd.init, kd.update, kd.log_prob)
    # perturb a query outcome (step 6): only its own query term changes, for every k
    s2 = list(steps)
    st = copy.copy(steps[6])
    st.outcomes = copy.copy(steps[6].outcomes)
    st.outcomes.y = steps[6].outcomes.y + 5.0
    s2[6] = st
    pert = L(s2, kd.init, kd.update, kd.log_prob)
    for k in (0, 2, 4):
        diff = (base["per_query"][k] - pert["per_query"][k]).abs().sum(0)
        assert diff[6] > 0 and diff[[j for j in range(10) if j != 6]].abs().sum() == 0
    # perturb a support outcome of k = 4 (step 1): every query term of k >= 2 changes, k = 0 does not
    s3 = list(steps)
    st = copy.copy(steps[1])
    st.outcomes = copy.copy(steps[1].outcomes)
    st.outcomes.y = steps[1].outcomes.y + 5.0
    s3[1] = st
    p3 = L(s3, kd.init, kd.update, kd.log_prob)
    assert torch.equal(p3["per_query"][0][:, 2:], base["per_query"][0][:, 2:])
    assert (p3["per_query"][4][:, 4:] - base["per_query"][4][:, 4:]).abs().sum() > 0
    # the same event in support and query is refused
    s4 = list(steps)
    s4[5] = copy.copy(steps[5])
    s4[5].event_ids = steps[1].event_ids
    with pytest.raises(ValueError):
        L(s4, kd.init, kd.update, kd.log_prob)


def test_every_episode_gets_its_own_support_and_unusable_sizes_are_reported():
    kd = KalmanDouble(4)
    steps, _ = regression_sequence(4, 12, seed=2, done_at=5)       # two episodes per row: steps 0..5 and 6..11
    L = SupportQueryLoss(ks=(0, 2, 8))
    out = L(steps, kd.init, kd.update, kd.log_prob)
    assert out["n_query"][0] == 4 * 12 and out["n_query"][2] == 4 * (4 + 4)   # both episodes scored
    assert out["n_episodes"] == {0: 8, 2: 8} and out["coverage"][0] == 1.0
    assert out["skipped_k"] == [8]                                              # no episode has 9 steps
    with pytest.raises(ValueError):
        SupportQueryLoss(ks=(0, 8), on_skip="raise")(steps, kd.init, kd.update, kd.log_prob)
    # episode 2 is adapted from a fresh belief on its own first k steps: identical to scoring it alone
    for t in range(6, 12):
        steps[t].event_ids = [f"x{b}_{t}" for b in range(4)]
    alone = SupportQueryLoss(ks=(0, 2))(steps[6:], kd.init, kd.update, kd.log_prob)
    for k in (0, 2):
        assert torch.allclose(out["per_query"][k][:, 6:], alone["per_query"][k], atol=1e-6)
    # and episode 1's outcomes never reach episode 2's scores
    s2 = list(steps)
    for t in range(0, 6):
        s2[t] = copy.copy(steps[t])
        s2[t].outcomes = copy.copy(steps[t].outcomes)
        s2[t].outcomes.y = steps[t].outcomes.y + 3.0
    p2 = L(s2, kd.init, kd.update, kd.log_prob)
    assert torch.equal(p2["per_query"][2][:, 6:], out["per_query"][2][:, 6:])
    assert not torch.equal(p2["per_query"][2][:, :6], out["per_query"][2][:, :6])
    g = torch.autograd.grad(out["per_k_tensor"][2], [kd.W, kd.log_noise, kd.log_prior], retain_graph=True)
    assert all(x.abs().sum() > 0 for x in g)                              # through the Kalman update itself


def test_padded_rows_are_neither_support_nor_query():
    kd = KalmanDouble(4)
    steps, _ = regression_sequence(3, 8, seed=5)
    steps[3].done[1] = True                         # row 1 ended after step 3; steps 4..7 are right padding
    for t in range(4, 8):
        steps[t].obs.mask[1] = False
        steps[t].outcomes.y_mask[1] = False
        steps[t].outcomes.y[1] = float("nan")      # garbage in padding must not matter
    out = SupportQueryLoss(ks=(0, 2))(steps, kd.init, kd.update, kd.log_prob)
    assert out["n_query"][0] == 8 + 4 + 8 and out["n_query"][2] == 6 + 2 + 6
    assert torch.isfinite(out["loss"]) and out["per_query"][0][1, 4:].abs().sum() == 0
    g = torch.autograd.grad(out["loss"], [kd.W, kd.log_noise])
    assert all(torch.isfinite(x).all() for x in g)
    full = SupportQueryLoss(ks=(0, 2))(steps, kd.init, kd.update, kd.log_prob,
                                       mask=torch.ones(3, 8, dtype=torch.bool))
    assert full["n_query"][0] == 8 + 8 + 8 - 4      # an explicit mask overrides (padded outcomes are unobserved)


# ----------------------------------------------------------------------------------------------- contract modules

class _ParticleBeliefDouble(nn.Module):
    """Contract-shaped belief module (BeliefModuleP): reweights particles by the outcome head's likelihood."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.particles = nn.Parameter(torch.randn(cfg.n_particles, cfg.d_particle) * 0.5)
        self.tok = nn.Linear(cfg.d_particle, cfg.d_model)

    def init_fast(self, B, device=None):
        fs = zero_fast_state(B, self.cfg, device=device)
        fs.particles = self.particles.unsqueeze(0).expand(B, -1, -1)
        return fs

    def tokens(self, fast):
        return TokenSet.from_dense(self.tok(fast.particles) * fast.logw.exp().unsqueeze(-1), TokenKind.BELIEF)

    def update(self, fast, Z_prev, act, outcomes, outcome_head):
        d = outcome_head.dist(outcome_head.features(Z_prev, outcomes), outcomes, cond=fast.particles)
        lp = d.log_prob(outcomes.y.unsqueeze(1).expand(-1, fast.particles.shape[1], -1))
        lp = (lp * outcomes.y_mask.unsqueeze(1)).sum(-1)
        out = FastState(fast.particles, torch.log_softmax(fast.logw + lp, -1), fast.pscale, fast.bayes_mu, fast.bayes_P,
                        fast.bayes_noise, fast.bayes_n, fast.surprise, fast.steps + 1, fast.channels)
        return out, {"loglik": lp}


class _WorldModelDouble(nn.Module):
    """WorldModelP.prior-shaped double: Z_hat = Z + tanh(W [mean action token; mean belief token; dt])."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.lin = nn.Linear(2 * D + 1, D)
        self.calls = 0

    def prior(self, Z, act, belief, dt, sample=True):
        self.calls += 1
        a = (act.x * act.mask.unsqueeze(-1)).sum(1) / act.mask.sum(1, keepdim=True).clamp_min(1)
        bt = belief.x.mean(1)
        h = torch.tanh(self.lin(torch.cat([a, bt, torch.as_tensor(dt).float().view(-1, 1)], -1)))
        return Z + h.unsqueeze(1), {}


def _modules(seed=3):
    torch.manual_seed(seed)
    cfg = tiny_config()
    seq = rand_sequence(2, 6, cfg, Q=3, seed=4)
    Z = [torch.randn(2, cfg.n_state, cfg.d_model) for _ in range(6)]
    return cfg, seq, Z, DoubleOutcomeHead(cfg), _ParticleBeliefDouble(cfg), _WorldModelDouble(cfg)


def test_contract_adapter_conditions_queries_on_their_actions_through_the_world_model():
    cfg, seq, Z, head, bel, wm = _modules()
    L = SupportQueryLoss(ks=(1, 3))
    with pytest.raises(ValueError):                                     # action-blind scoring must be explicit
        belief_callables_from_modules(bel, head, Z)
    init, upd, lp = belief_callables_from_modules(bel, head, Z, steps=seq, world_model=wm)
    out = L(seq, init, upd, lp)
    assert torch.isfinite(out["loss"])
    gp, gw = torch.autograd.grad(out["loss"], [bel.particles, wm.lin.weight])
    assert gp.abs().sum() > 0 and gw.abs().sum() > 0                    # through the update and the open-loop prior
    assert wm.calls <= 2 * 6                                            # memoised rollout: O(T) per support size
    # changing the action of query step 4 changes the scores of queries 4 and 5 (open loop), not of 3
    seq2 = copy.deepcopy(seq)
    seq2.steps[4].act.x = seq.steps[4].act.x + 1.0
    i2, u2, l2 = belief_callables_from_modules(bel, head, Z, steps=seq2, world_model=wm)
    o2 = L(seq2, i2, u2, l2)
    d = (o2["per_query"][3] - out["per_query"][3]).abs().sum(0)
    assert d[3] == 0 and d[4] > 0 and d[5] > 0
    # a latent state that already absorbed query-time observations is never used; the support-end state is
    Z2 = list(Z)
    Z2[5] = Z[5] + 10.0
    i3, u3, l3 = belief_callables_from_modules(bel, head, Z2, steps=seq, world_model=wm)
    assert torch.equal(L(seq, i3, u3, l3)["per_query"][3], out["per_query"][3])
    Z3 = list(Z)
    Z3[3] = Z[3] + 10.0
    i4, u4, l4 = belief_callables_from_modules(bel, head, Z3, steps=seq, world_model=wm)
    assert not torch.equal(L(seq, i4, u4, l4)["per_query"][3], out["per_query"][3])
    # call-order independence: query 5 scored first or after query 4 gives the same value
    b = init(2, None)
    for t in range(3):
        b = upd(b, seq.steps[t], t)
    late = lp(b, seq.steps[5], 5)
    b2 = init(2, None)
    for t in range(3):
        b2 = upd(b2, seq.steps[t], t)
    lp(b2, seq.steps[4], 4)
    assert torch.allclose(lp(b2, seq.steps[5], 5), late)
    assert torch.allclose(late, out["per_query"][3][:, 5])


def test_contract_adapter_action_condition_and_explicit_action_blind_mode():
    cfg, seq, Z, head, bel, wm = _modules()
    L = SupportQueryLoss(ks=(1, 3))
    cond = nn.Linear(cfg.d_model, cfg.d_model)

    def action_condition(Zs, act):
        return Zs + cond(act.x.mean(1)).unsqueeze(1)
    init, upd, lp = belief_callables_from_modules(bel, head, Z, action_condition=action_condition)
    out = L(seq, init, upd, lp)
    seq2 = copy.deepcopy(seq)
    seq2.steps[4].act.x = seq.steps[4].act.x + 1.0
    d = (L(seq2, init, upd, lp)["per_query"][3] - out["per_query"][3]).abs().sum(0)
    assert d[4] > 0 and d[3] == 0 and d[5] == 0                        # only the query's own action
    ib, ub, lb = belief_callables_from_modules(bel, head, Z, allow_action_blind=True)
    blind = L(seq, ib, ub, lb)
    assert torch.equal(L(seq2, ib, ub, lb)["per_query"][3], blind["per_query"][3])


def test_hindsight_lp_through_the_adapter_scores_the_prior_belief_from_the_prior_state():
    """B_t^- must be scored from Z_t (before o_t), B_t^+ from Z_{t+1}: perturbing Z_{t+1} moves only the B^+ side."""
    cfg, seq, Z, head, bel, wm = _modules()
    init, upd, lp = belief_callables_from_modules(bel, head, Z, steps=seq, world_model=wm)
    fasts = [bel.init_fast(2)]
    for t in range(5):
        fasts.append(bel.update(fasts[-1], Z[t], seq.steps[t].act, seq.steps[t].outcomes, head)[0])
    beliefs = [support_belief(f, t) for t, f in enumerate(fasts)]
    r = hindsight_learning_progress(beliefs, seq.steps, upd, lp, horizon=1)
    b_minus = lp(beliefs[2], seq.steps[3], 3)
    Z2 = list(Z)
    Z2[3] = Z[3] + 10.0
    _, upd2, lp2 = belief_callables_from_modules(bel, head, Z2, steps=seq, world_model=wm)
    assert torch.equal(lp2(beliefs[2], seq.steps[3], 3), b_minus)        # B_2^- ignores Z_3
    r2 = hindsight_learning_progress(beliefs, seq.steps, upd2, lp2, horizon=1)
    assert not torch.equal(r["lp"][:, 2], r2["lp"][:, 2])                # B_2^+ = Update(B_2^-, o_2) uses Z_3
    assert torch.equal(r["lp"][:, 0], r2["lp"][:, 0])


def test_tree_where_selects_rows_of_nested_beliefs():
    a = (torch.zeros(3, 2), {"P": torch.zeros(3, 2, 2)}, "tag")
    b = (torch.ones(3, 2), {"P": torch.ones(3, 2, 2)}, "other")
    m = torch.tensor([True, False, True])
    out = tree_where(m, a, b)
    assert out[0][:, 0].tolist() == [0.0, 1.0, 0.0] and out[1]["P"][1].sum() == 4 and out[2] == "tag"


# ----------------------------------------------------------------------------------------------- J_meta

def _traj4(intrinsic=None):
    reward = torch.tensor([[-30.0, 0.0, 100.0, 100.0]])
    return Trajectory(reward=reward, dt=torch.zeros(1, 4), done=torch.tensor([[False, False, False, True]]),
                      behavior_logp=torch.zeros(1, 4), model_version=torch.zeros(1, 4, dtype=torch.long),
                      value=torch.zeros(1, 5), cost=torch.tensor([[30.0, 0.0, 0.0, 0.0]]),
                      trial=torch.tensor([[0, 0, 1, 1]]), intrinsic=intrinsic)


def test_j_meta_credits_early_exploration_with_later_trial_competence():
    # two trials of 2 decisions; trial 0 decision 0 is a costly probe (-30), trial 1 earns 100 per decision
    traj = _traj4()
    ac = ActorCriticObjective(gae_lambda=1.0, normalize_advantages=False)
    meta = MetaRLObjective(ac, trial_weights=[0.5, 1.0], cost_coef=0.1)
    shaped, prep = meta.prepare(traj, 0)
    assert shaped.reward.tolist() == [[-15.0 - 3.0, 0.0, 100.0, 100.0]]
    assert prep.adv[0, 0].item() == pytest.approx(-18.0 + 200.0)      # the probe is credited with trial 1
    assert meta.j_meta(traj).item() == pytest.approx(182.0)
    assert meta.per_trial_return(traj).tolist() == [[-30.0, 200.0]]
    d = meta.loss(shaped, prep, torch.zeros(1, 4, requires_grad=True), torch.zeros(1, 4), torch.zeros(1, 5))
    assert "meta" in d and "actor" not in d and MetaRLObjective.policy_key == "meta"
    # if the trial boundary were (wrongly) terminal, the probe would get no credit
    traj_cut = Trajectory(reward=traj.reward, dt=torch.zeros(1, 4), done=torch.tensor([[False, True, False, True]]),
                          behavior_logp=torch.zeros(1, 4), model_version=torch.zeros(1, 4, dtype=torch.long),
                          value=torch.zeros(1, 5), cost=traj.cost, trial=traj.trial)
    _, prep_cut = meta.prepare(traj_cut, 0)
    assert prep_cut.adv[0, 0].item() == pytest.approx(-18.0)


def test_j_meta_over_support_sizes_weights_competence_after_k_decisions():
    """J_meta = sum_k w_k J(U(D_{1:k})) with J = return of the `window` decisions after k support decisions."""
    T = 8
    reward = torch.tensor([[-30.0] + [100.0] * (T - 1)])
    traj = Trajectory(reward=reward, dt=torch.zeros(1, T), done=torch.tensor([[False] * (T - 1) + [True]]),
                      behavior_logp=torch.zeros(1, T), model_version=torch.zeros(1, T, dtype=torch.long),
                      value=torch.zeros(1, T + 1))
    meta = MetaRLObjective(ActorCriticObjective(gae_lambda=1.0, kappa=0.0, normalize_advantages=False),
                           support_weights={1: 1.0, 2: 0.5, 4: 0.25}, competence_window=2, use_intrinsic=False)
    n = torch.arange(T)
    want = torch.tensor([0.0, 1.0, 1.5, 0.5, 0.25, 0.25, 0.0, 0.0])      # W(n) = sum of w_k with k <= n < k + 2
    assert torch.allclose(meta.support_weight(n), want)
    explicit = sum(w * reward[0, k:k + 2].sum() for k, w in {1: 1.0, 2: 0.5, 4: 0.25}.items())
    assert meta.j_meta(traj).item() == pytest.approx(float(explicit))
    _, prep = meta.prepare(traj, 0)
    assert prep.adv[0, 0].item() == pytest.approx(float(explicit))       # decision 0 credited with later competence
    comp = meta.competence_by_support(traj)
    assert comp[1] == pytest.approx(200.0) and comp[4] == pytest.approx(200.0) and comp["n"][4] == 1


def test_lp_term_is_gated_by_the_ablation_switch_the_stage_flag_and_scaled_by_lp_coef():
    intr = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    traj = _traj4(intrinsic=intr)
    cfg = make_config("dev", lp_coef=7.0)
    m_off = MetaRLObjective.from_config(make_config("dev", use_learned_exploration=False, lp_coef=7.0))
    m_on = MetaRLObjective.from_config(cfg)
    assert m_off.shaped_reward(traj)[0, 0].item() == pytest.approx(-30.0)
    assert m_on.shaped_reward(traj)[0, 0].item() == pytest.approx(-30.0 + 14.0)
    lin = nn.Linear(2, 2)
    tr = CombinedTrainer(cfg, core=lin, stage="actor_critic", amp=False)
    m_stage = MetaRLObjective.from_config(cfg, trainer=tr)
    assert not m_stage.intrinsic_enabled()                              # stage 4: no LP reward yet
    assert m_stage.prepare(traj, 0)[1].info["lp_in_reward"] == 0
    tr.set_stage("explore_meta")
    assert m_stage.intrinsic_enabled() and m_stage.shaped_reward(traj)[0, 0].item() == pytest.approx(-16.0)
    # in stage 4 the trainer does not optimise the J_meta objective (stage gating through the "meta" key)
    tr.set_stage("actor_critic")
    x = lin(torch.ones(1, 2)).sum()
    info = tr.step({"meta": x, "value": x * 0.0 + lin.weight.sum()})
    assert info.disabled == ["meta"]


# ----------------------------------------------------------------------------------------------- harness

class _Scripted:
    """Probe while the paying arm is unknown; afterwards pull the signalled arm (remembered across trials)."""

    def __init__(self):
        self.version = 7
        self.shown = []

    def initial_state(self, B):
        return {"arm": [None] * B}

    def act(self, state, obs, tools, budget, goals, active, greedy=False):
        self.shown.append([list(o) for o in obs])
        acts = []
        for b in range(len(obs)):
            for o in obs[b]:
                if o.source == "probe" and o.value == 1.0:
                    state["arm"][b] = int(o.entity.split("_")[1])
            acts.append(Action("probe") if state["arm"][b] is None else Action("pull", {"arm": state["arm"][b]}))
        B = len(obs)
        return acts, torch.full((B,), -0.5), torch.zeros(B), state, {"dl": budget.decisions_left.clone()}


def test_trial_runner_multi_trial_episode_compaction_and_provenance():
    ad = ProbeBanditAdapter(3, horizon=(1, 4))
    seeds = [10, 11, 12]
    runner = TrialRunner(ad, ad.evaluator, think_days=0.5)
    res = runner.run(_Scripted(), seeds, n_trials=2)
    tr = res.traj
    H = [ad.envs[b].horizon for b in range(3)]
    assert tr.lengths().tolist() == [2 * h for h in H]                 # both trials, left-aligned
    for b in range(3):
        L = 2 * H[b]
        assert tr.done[b, :L].tolist() == [False] * (L - 1) + [True]  # only the last decision is terminal
        assert tr.trial[b, :L].tolist() == [0] * H[b] + [1] * H[b]
        assert res.actions[b][:H[b]][0] == "probe"
        assert all(a == "pull" for a in res.actions[b][H[b]:])         # the arm is remembered across trials
        assert res.per_trial_return[b, 1].item() == pytest.approx(100.0 * H[b])
        dl = tr.extras["dl"][b, :L].tolist()
        assert dl == list(range(H[b], 0, -1)) * 2                       # budget token input per trial
    assert torch.allclose(tr.dt[tr.mask], torch.full((int(tr.mask.sum()),), 1 / 24 + 0.5))
    assert torch.all(tr.model_version[tr.mask] == 7) and torch.all(tr.behavior_logp[tr.mask] == -0.5)
    assert torch.allclose(tr.reward.sum(1), res.per_trial_return.sum(1))
    assert tr.cost[tr.mask].max().item() == 30.0


def test_last_decision_of_a_trial_reaches_the_policy_in_the_next_trial():
    """One-decision trials: the probe's report (produced by trial 1's only decision) must reach trial 2."""
    ad = ProbeBanditAdapter(4, horizon=1)
    pol = _Scripted()
    res = TrialRunner(ad, ad.evaluator).run(pol, [1, 2, 3, 4], n_trials=2)
    assert all(a == ["probe", "pull"] for a in res.actions)
    assert res.per_trial_return[:, 1].tolist() == [100.0] * 4         # exploits what trial 1 revealed
    first_of_trial2 = pol.shown[1]
    for b in range(4):
        srcs = [o.source for o in first_of_trial2[b]]
        assert srcs.count("probe") == 6 and srcs[-1] == "operator"     # previous outcome, then the reset view
    # a delayed payoff released by the last clock movement of a trial is delivered as well
    ad2 = DelayedRewardAdapter(2, horizon=1, delay_days=0.25, decision_days=0.25)

    class Invest:
        version = 0

        def __init__(self):
            self.shown, self.ids = [], []

        def initial_state(self, B):
            return None

        def act(self, state, obs, tools, budget, goals, active, greedy=False):
            self.shown.append([list(o) for o in obs])
            acts = [Action("invest") for _ in obs]
            self.ids.append([a.action_id for a in acts])
            return acts, torch.zeros(len(obs)), torch.zeros(len(obs)), None, {}
    inv = Invest()
    TrialRunner(ad2, ad2.evaluator).run(inv, [0, 1], n_trials=2)
    for b in range(2):
        ev = [o for o in inv.shown[1][b] if o.kind == ObsKind.EVENT]
        assert len(ev) == 1 and ev[0].caused_by == inv.ids[0][b] and ev[0].value == 150.0


def test_trial_runner_needs_no_adapter_batch_attribute():
    class Wrapped:
        """An EnvAdapter that only implements the contract (no `.B`)."""

        def __init__(self, inner):
            self._a = inner

        def __getattr__(self, name):
            if name == "B":
                raise AttributeError(name)
            return getattr(self._a, name)
    inner = ProbeBanditAdapter(3, horizon=2)
    res = TrialRunner(Wrapped(inner), inner.evaluator).run(_Scripted(), [5, 6, 7])
    assert res.traj.B == 3 and res.traj.lengths().tolist() == [2, 2, 2]
    with pytest.raises(ValueError):
        TrialRunner(Wrapped(inner), inner.evaluator).run(_Scripted(), [5, 6])
