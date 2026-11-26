"""The CAM-scale meta-gradient machinery agrees with the exact tabular meta-gradient (IR-03, IR-04, IR-06)."""
import torch
import torch.nn as nn

from superlearner.cam_unified.credit import exact as X
from superlearner.cam_unified.credit import meta_exact as M
from superlearner.cam_unified.selfeval.metagrad import InnerBatch, MetaRewardLearner, QueryBatch

torch.set_num_threads(4)


class TabPolicy(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(S, A, dtype=torch.float64))

    def forward(self, s):
        return self.theta[s]


class TabReward(nn.Module):
    def __init__(self, psi):
        super().__init__()
        self.psi = nn.Parameter(psi.clone())

    def forward(self, t, s, a):
        return self.psi[t, s, a]


def _batch(mdp, b):
    T = mdp.T
    t = torch.arange(T)[None, :].expand_as(b.s)
    return dict(policy_inputs={"s": b.s}, actions=b.a, mask=torch.ones_like(b.s, dtype=torch.bool),
                reward_inputs={"t": t, "s": b.s, "a": b.a})


def test_meta_gradient_matches_exact_tabular_reference():
    mdp = X.key_to_door(n_distract=4, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    psi0 = 0.1 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    K, alpha = 2, 1.0
    g_exact, J = M.exact_meta_gradient(mdp, psi0, torch.zeros(mdp.S, mdp.A, dtype=X.DT), K, alpha)
    gen = torch.Generator().manual_seed(1)
    est = []
    for _ in range(24):
        pol, rew = TabPolicy(mdp.S, mdp.A), TabReward(psi0)
        # the class's inner_update minimises a per-step mean; scale the inner lr so one step equals the exact
        # inner update theta + alpha * grad J_int (sum over T steps)
        mrl = MetaRewardLearner(pol, rew, inner_lr=alpha * mdp.T, K=K, normalize_inner_adv=False)
        # inner batches are sampled from the *current* fast policy at each inner step (on-policy), as in meta_exact
        fast = mrl.start()
        for k in range(K):
            b = X.sample(mdp, fast["theta"].detach(), 256, gen)
            fast, _ = mrl.inner_step(fast, InnerBatch(**_batch(mdp, b)))
        q = X.sample(mdp, fast["theta"].detach(), 512, gen)
        Gx = q.r.sum(1)
        qb = QueryBatch(policy_inputs={"s": q.s}, actions=q.a, mask=torch.ones_like(q.s, dtype=torch.bool),
                        utility=Gx)
        # inner-data score term: any constant baseline is unbiased; ~E[J_query] keeps its variance low
        g = mrl.meta_gradient(fast, qb, score_baseline=J)[0]
        est.append(-g * mdp.T)       # outer_loss is a per-step mean of -surrogate
    g_mean = torch.stack(est).mean(0)
    cos = torch.nn.functional.cosine_similarity(g_mean.flatten(), g_exact.flatten(), 0)
    assert float(cos) > 0.9, float(cos)


def test_inner_update_api_connects_reward_parameters_to_query_loss_and_steps_psi():
    torch.manual_seed(0)
    N, T, A, F = 32, 5, 3, 4
    pol = nn.Sequential(nn.Linear(F, 16), nn.Tanh(), nn.Linear(16, A))
    rew = nn.Linear(F + A, 1)

    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = rew

        def forward(self, x, a):
            return self.l(torch.cat([x, torch.nn.functional.one_hot(a, A).float()], -1)).squeeze(-1)

    class P(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = pol

        def forward(self, x):
            return self.net(x)

    x = torch.randn(N, T, F)
    a = torch.randint(0, A, (N, T))
    m = torch.ones(N, T, dtype=torch.bool)
    p_mod, r_mod = P(), R()
    mrl = MetaRewardLearner(p_mod, r_mod, inner_lr=0.5, K=3, fast_param_filter=lambda n: n.startswith("net.2"))
    assert mrl.fast_names == ["net.2.weight", "net.2.bias"]
    fast = mrl.inner_update([InnerBatch({"x": x}, a, m, {"x": x, "a": a})])
    q = QueryBatch({"x": x}, a, m, utility=torch.randn(N))
    g = mrl.meta_gradient(fast, q, retain_graph=True)
    assert sum(float(t.abs().sum()) for t in g) > 0
    before = [p.detach().clone() for p in r_mod.parameters()]
    st = mrl.outer_step(fast, q)
    assert st["meta_grad_norm"] > 0
    assert any(float((b - p).abs().sum()) > 0 for b, p in zip(before, r_mod.parameters()))
    # detaching the reward inside the inner loop removes the pathway
    mrl2 = MetaRewardLearner(p_mod, r_mod, inner_lr=0.5, K=1)
    fast2 = mrl2._params()
    fast2 = {n: p.detach().requires_grad_(True) for n, p in fast2.items()}
    g2 = torch.autograd.grad(mrl2.outer_loss(fast2, q), list(r_mod.parameters()), allow_unused=True)
    assert all(t is None for t in g2)
