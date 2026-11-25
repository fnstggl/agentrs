"""Support/query fast-adaptation loss over k in {0..64} (strict separation, gradients through adaptation, training
reduces it), the contract-module adapter, J_meta credit across trials, and the multi-trial episode harness."""
import math

import pytest
import torch
import torch.nn as nn

from learn_doubles import KalmanDouble, regression_sequence
from superlearner.cam_unified.interfaces import Action, FastState
from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
from superlearner.cam_unified.learn.meta import (SUPPORT_SIZES, MetaRLObjective, SupportQueryLoss, TrialRunner,
                                                 belief_callables_from_modules)
from superlearner.cam_unified.learn.ppo import ActorCriticObjective
from superlearner.cam_unified.learn.trajectory import Trajectory
from superlearner.cam_unified.testing import DoubleOutcomeHead, rand_sequence, tiny_config, zero_fast_state


def _run(loss_fn, kd, steps):
    return loss_fn(steps, kd.init, kd.update, kd.log_prob)


def test_support_sizes_and_meta_loss_decreases_on_toy_task():
    torch.manual_seed(0)
    assert SUPPORT_SIZES == (0, 1, 2, 4, 8, 16, 32, 64)
    kd = KalmanDouble(4, init_W=0.3)
    L = SupportQueryLoss(max_query=16)
    steps, _ = regression_sequence(32, 66, seed=100)
    with torch.no_grad():
        before = _run(L, kd, steps)
    assert sorted(before["per_k"]) == list(SUPPORT_SIZES)
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
    import copy
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


def test_queries_stay_inside_the_support_episode_and_gradients_flow_through_adaptation():
    kd = KalmanDouble(4)
    steps, _ = regression_sequence(4, 12, seed=2, done_at=5)       # episode ends after step 5
    out = SupportQueryLoss(ks=(0, 2, 8))(steps, kd.init, kd.update, kd.log_prob)
    assert out["n_query"][0] == 4 * 6 and out["n_query"][2] == 4 * 4     # queries 0..5 / 2..5 only
    assert 8 not in out["per_k"]                                          # a support crossing the boundary is unused
    g = torch.autograd.grad(out["per_k_tensor"][2], [kd.W, kd.log_noise, kd.log_prior], retain_graph=True)
    assert all(x.abs().sum() > 0 for x in g)                              # through the Kalman update itself
    g0 = torch.autograd.grad(out["per_k_tensor"][0], [kd.W, kd.log_noise, kd.log_prior], allow_unused=True)
    assert all(x is not None for x in g0)


class _ParticleBeliefDouble(nn.Module):
    """Contract-shaped belief module (BeliefModuleP): reweights particles by the outcome head's likelihood."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.particles = nn.Parameter(torch.randn(cfg.n_particles, cfg.d_particle) * 0.5)

    def init_fast(self, B, device=None):
        fs = zero_fast_state(B, self.cfg, device=device)
        fs.particles = self.particles.unsqueeze(0).expand(B, -1, -1)
        return fs

    def update(self, fast, Z_prev, act, outcomes, outcome_head):
        d = outcome_head.dist(outcome_head.features(Z_prev, outcomes), outcomes, cond=fast.particles)
        lp = d.log_prob(outcomes.y.unsqueeze(1).expand(-1, fast.particles.shape[1], -1))
        lp = (lp * outcomes.y_mask.unsqueeze(1)).sum(-1)
        out = FastState(fast.particles, torch.log_softmax(fast.logw + lp, -1), fast.pscale, fast.bayes_mu, fast.bayes_P,
                        fast.bayes_noise, fast.bayes_n, fast.surprise, fast.steps + 1, fast.channels)
        return out, {"loglik": lp}


def test_belief_callables_from_contract_modules_score_queries_from_the_support_state():
    torch.manual_seed(3)
    cfg = tiny_config()
    seq = rand_sequence(2, 6, cfg, Q=3, seed=4)
    Z = [torch.randn(2, cfg.n_state, cfg.d_model) for _ in range(6)]
    head, bel = DoubleOutcomeHead(cfg), _ParticleBeliefDouble(cfg)
    init, upd, lp = belief_callables_from_modules(bel, head, Z)
    L = SupportQueryLoss(ks=(1, 3))
    out = L(seq, init, upd, lp)
    assert torch.isfinite(out["loss"])
    (g,) = torch.autograd.grad(out["loss"], [bel.particles])
    assert g.abs().sum() > 0
    Z2 = list(Z)
    Z2[5] = Z[5] + 10.0                       # a latent state that already absorbed query-time observations
    init2, upd2, lp2 = belief_callables_from_modules(bel, head, Z2)
    out2 = L(seq, init2, upd2, lp2)
    assert torch.equal(out["per_query"][3], out2["per_query"][3])      # queries scored from Z at the support end
    Z3 = list(Z)
    Z3[3] = Z[3] + 10.0                       # the support-end state does matter
    init3, upd3, lp3 = belief_callables_from_modules(bel, head, Z3)
    assert not torch.equal(out["per_query"][3], L(seq, init3, upd3, lp3)["per_query"][3])


def test_j_meta_credits_early_exploration_with_later_trial_competence():
    # two trials of 2 decisions; trial 0 decision 0 is a costly probe (-30), trial 1 earns 100 per decision
    reward = torch.tensor([[-30.0, 0.0, 100.0, 100.0]])
    traj = Trajectory(reward=reward, dt=torch.zeros(1, 4), done=torch.tensor([[False, False, False, True]]),
                      behavior_logp=torch.zeros(1, 4), model_version=torch.zeros(1, 4, dtype=torch.long),
                      value=torch.zeros(1, 5), cost=torch.tensor([[30.0, 0.0, 0.0, 0.0]]),
                      trial=torch.tensor([[0, 0, 1, 1]]))
    ac = ActorCriticObjective(gae_lambda=1.0, normalize_advantages=False)
    meta = MetaRLObjective(ac, trial_weights=[0.5, 1.0], cost_coef=0.1)
    shaped, prep = meta.prepare(traj, 0)
    assert shaped.reward.tolist() == [[-15.0 - 3.0, 0.0, 100.0, 100.0]]
    assert prep.adv[0, 0].item() == pytest.approx(-18.0 + 200.0)      # the probe is credited with trial 1
    assert meta.j_meta(traj).item() == pytest.approx(182.0)
    assert meta.per_trial_return(traj).tolist() == [[-30.0, 200.0]]
    # if the trial boundary were (wrongly) terminal, the probe would get no credit
    traj_cut = Trajectory(reward=reward, dt=torch.zeros(1, 4), done=torch.tensor([[False, True, False, True]]),
                          behavior_logp=torch.zeros(1, 4), model_version=torch.zeros(1, 4, dtype=torch.long),
                          value=torch.zeros(1, 5), cost=traj.cost, trial=traj.trial)
    _, prep_cut = meta.prepare(traj_cut, 0)
    assert prep_cut.adv[0, 0].item() == pytest.approx(-18.0)


class _Scripted:
    """Probe first in trial 0; afterwards pull the signalled arm (remembered across trials)."""

    def __init__(self):
        self.version = 7

    def initial_state(self, B):
        return {"arm": [None] * B}

    def act(self, state, obs, tools, budget, goals, active, greedy=False):
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
