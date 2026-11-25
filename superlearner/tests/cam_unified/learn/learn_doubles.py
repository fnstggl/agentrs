"""Test-only doubles for workstream D: tiny differentiable beliefs (Kalman, particle, arm-hypothesis), a small recurrent
diagnostic policy for the probe bandit, and the training / evaluation loop that drives it with the production
machinery (TrialRunner, ActorCriticObjective / MetaRLObjective, CombinedTrainer, ValueHead, BudgetEncoder)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import (Action, FastState, Likelihood, ObsKind, OutcomeBatch, StepBatch,
                                                 TokenKind, TokenSet)
from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
from superlearner.cam_unified.learn.explore import (categorical_decomposition, combine_signal_frac,
                                                    exploration_reward, hindsight_learning_progress)
from superlearner.cam_unified.learn.heads import BudgetEncoder, BudgetState, ValueHead
from superlearner.cam_unified.learn.math_assist import CandidateValueModel, MathAssistEncoder
from superlearner.cam_unified.learn.meta import MetaRLObjective, TrialRunner
from superlearner.cam_unified.learn.ppo import ActorCriticObjective
from superlearner.cam_unified.learn.trainer import CombinedTrainer


# ================================================================================================ beliefs

class KalmanDouble(nn.Module):
    """Differentiable Kalman belief over a per-task weight vector theta in R^F (tests only).
    Learned: prior log-variance, noise log-variance, feature map W (phi = W x). Belief = (mu [B,F], P [B,F,F])."""

    def __init__(self, F_: int = 4, init_W: float = 0.3, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.F = F_
        self.log_prior = nn.Parameter(torch.tensor(0.0))
        self.log_noise = nn.Parameter(torch.tensor(1.0))
        self.W = nn.Parameter(init_W * torch.eye(F_) + 0.05 * torch.randn(F_, F_, generator=g))

    def init(self, B, device=None):
        P = torch.exp(self.log_prior) * torch.eye(self.F, device=device).expand(B, self.F, self.F)
        return (torch.zeros(B, self.F, device=device), P)

    def phi(self, step):
        return step.act.x[:, 0, :self.F] @ self.W.t()

    def update(self, belief, step, t):
        mu, P = belief
        phi = self.phi(step)
        y = step.outcomes.y[:, 0]
        m = step.outcomes.y_mask[:, 0].float()
        s = torch.einsum("bf,bfg,bg->b", phi, P, phi) + torch.exp(self.log_noise)
        K = torch.einsum("bfg,bg->bf", P, phi) / s.unsqueeze(-1)
        innov = y - (phi * mu).sum(-1)
        mu = mu + m.unsqueeze(-1) * K * innov.unsqueeze(-1)
        P = P - m[:, None, None] * torch.einsum("bf,bg->bfg", K, torch.einsum("bfg,bg->bf", P, phi))
        return (mu, P)

    def log_prob(self, belief, step, t):
        mu, P = belief
        phi = self.phi(step)
        mean = (phi * mu).sum(-1)
        var = torch.einsum("bf,bfg,bg->b", phi, P, phi) + torch.exp(self.log_noise)
        y = step.outcomes.y[:, 0]
        lp = -0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)
        return lp * step.outcomes.y_mask[:, 0].float()


def regression_sequence(B: int, T: int, F_: int = 4, D: int = 16, theta_sd: float = 2.0, noise: float = 0.3,
                        seed: int = 0, done_at=None):
    """Toy few-shot task as contract StepBatches: y_t = theta . x_t + eps with a per-row hidden theta."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(B, F_, generator=g) * theta_sd
    steps = []
    for t in range(T):
        x = torch.randn(B, F_, generator=g)
        y = (theta * x).sum(-1) + noise * torch.randn(B, generator=g)
        ax = torch.zeros(B, 1, D)
        ax[:, 0, :F_] = x
        q = TokenSet.from_dense(torch.zeros(B, 1, D), TokenKind.QUERY)
        oc = OutcomeBatch(q, torch.full((B, 1), int(Likelihood.GAUSSIAN)), torch.zeros(B, 1, dtype=torch.long),
                          y.unsqueeze(1), torch.zeros(B, 1, dtype=torch.bool), torch.ones(B, 1, dtype=torch.bool),
                          torch.zeros(B, 1, dtype=torch.long))
        done = torch.zeros(B, dtype=torch.bool)
        if done_at is not None and t == done_at:
            done[:] = True
        steps.append(StepBatch(obs=TokenSet.empty(B, D), act=TokenSet.from_dense(ax, TokenKind.ACTION), outcomes=oc,
                               dt=torch.ones(B), reward=torch.zeros(B), done=done, behavior_logp=torch.zeros(B),
                               model_version=torch.zeros(B, dtype=torch.long),
                               event_ids=[f"e{b}_{t}" for b in range(B)]))
    return steps, theta


@dataclass
class ScalarStep:
    x: torch.Tensor          # [B] feature (0 = pure-noise channel)
    y: torch.Tensor          # [B] observed outcome
    noise_sd: torch.Tensor   # [B] observation noise sd of this channel


class ParticleThetaBelief:
    """K fixed hypotheses theta_k with weights; y = theta x + N(0, sd^2). ``naive_shift`` > 0 adds a (bad) learned-
    style update that moves every hypothesis by a fraction of the innovation even when it carries no information."""

    def __init__(self, thetas: torch.Tensor, naive_shift: float = 0.0):
        self.thetas = thetas          # [K]
        self.naive_shift = naive_shift

    def init(self, B):
        K = self.thetas.shape[0]
        return (self.thetas.expand(B, K).clone(), torch.full((B, K), -math.log(K)))

    def _ll(self, th, s: ScalarStep):
        mean = th * s.x.unsqueeze(1)
        sd = s.noise_sd.unsqueeze(1)
        return -0.5 * (math.log(2 * math.pi) + 2 * torch.log(sd) + ((s.y.unsqueeze(1) - mean) / sd) ** 2)

    def update(self, belief, s: ScalarStep, t):
        th, logw = belief
        logw = torch.log_softmax(logw + self._ll(th, s), -1)
        if self.naive_shift:
            mean = (logw.exp() * th).sum(-1, keepdim=True) * s.x.unsqueeze(1)
            th = th + self.naive_shift * (s.y.unsqueeze(1) - mean)
        return (th, logw)

    def log_prob(self, belief, s: ScalarStep, t):
        th, logw = belief
        return torch.logsumexp(logw + self._ll(th, s), -1)

    def moments(self, belief, s: ScalarStep):
        th, logw = belief
        mean = (th * s.x.unsqueeze(1)).unsqueeze(-1)
        var = (s.noise_sd ** 2).unsqueeze(1).expand_as(th).unsqueeze(-1)
        return mean, var, logw


class ArmBelief:
    """Exact categorical belief over the paying arm of `ProbeBanditAdapter` (hypothesis h = paying arm), built only
    from agent-side observations. log-likelihoods of a decision's observations under each hypothesis:
    probe signals (one-hot of h, with flip prob eps), pull payout (pay iff arm == h), watch (uniform 1/100)."""

    def __init__(self, n_arms: int, pay: float = 100.0, eps: float = 1e-3):
        self.n, self.pay, self.eps = n_arms, pay, eps

    def init(self, B):
        return torch.full((B, self.n), -math.log(self.n))

    def loglik(self, obs: list) -> torch.Tensor:
        """[n_arms] log p(observations | h) for one environment's observation list."""
        ll = torch.zeros(self.n)
        h = torch.arange(self.n)
        for o in obs:
            if o.kind != ObsKind.NUMERIC or o.value is None:
                continue
            if o.source == "probe":
                i = int(o.entity.split("_")[1])
                match = (h == i) == (o.value > 0.5)
                ll += torch.where(match, math.log(1 - self.eps), math.log(self.eps))
            elif o.source == "pull":
                i = int(o.entity.split("_")[1])
                paid = o.value > 0.5 * self.pay
                match = (h == i) == paid
                ll += torch.where(match, math.log(1 - self.eps), math.log(self.eps))
            elif o.source == "watch":
                ll += -math.log(100.0)
        return ll

    def outcome_probs(self, obs: list) -> torch.Tensor | None:
        """Per-hypothesis distribution over the decision's discrete outcome (for the epistemic/aleatoric split):
        probe -> which arm signals (n_arms classes); pull -> paid / not paid (2); watch -> 100 readings."""
        srcs = {o.source for o in obs if o.kind == ObsKind.NUMERIC}
        h = torch.arange(self.n)
        if "probe" in srcs:
            return F.one_hot(h, self.n).float().unsqueeze(1)                          # [K,1,C]
        if "pull" in srcs:
            i = next(int(o.entity.split("_")[1]) for o in obs if o.source == "pull")
            p_paid = (h == i).float()
            return torch.stack([1 - p_paid, p_paid], -1).unsqueeze(1)
        if "watch" in srcs:
            return torch.full((self.n, 1, 100), 0.01)
        return None


def arm_learning_progress(res, belief: ArmBelief, horizon: int = 8, coef: float = 1.0, corrected: bool = True,
                          enabled: bool = True) -> dict:
    """Hindsight LP of every decision of a `RunResult` from the probe bandit, under the exact arm belief.
    Returns {"lp", "signal_frac", "intrinsic"} [B, L] (compacted like the trajectory)."""
    traj = res.traj
    B, L = traj.B, traj.T
    K = belief.n
    LL = torch.zeros(B, L, K)
    probs = {}
    for b in range(B):
        for t, obs in enumerate(res.observations[b]):
            LL[b, t] = belief.loglik(obs)
            probs[(b, t)] = belief.outcome_probs(obs)
    steps = [LL[:, t] for t in range(L)]
    beliefs = [belief.init(B)]
    for t in range(L - 1):
        beliefs.append(torch.log_softmax(beliefs[-1] + steps[t], -1))
    upd = lambda bel, ll, t: torch.log_softmax(bel + ll, -1)                 # noqa: E731
    lpf = lambda bel, ll, t: torch.logsumexp(bel + ll, -1)                   # noqa: E731
    ep = traj.trial if traj.trial is not None else None
    r = hindsight_learning_progress(beliefs, steps, upd, lpf, horizon, traj.mask, ep)
    rho = torch.zeros(B, L)
    for (b, t), p in probs.items():
        if p is None or not bool(traj.mask[b, t]):
            continue
        d = categorical_decomposition(p.unsqueeze(0), beliefs[t][b:b + 1])
        rho[b, t] = combine_signal_frac(d["signal_frac"])[0]
    intr = exploration_reward(r["lp"], rho if corrected else None, coef, enabled=enabled) * traj.mask.float()
    return {"lp": r["lp"], "signal_frac": rho, "intrinsic": intr}


# ================================================================================================ probe agent

class ProbeAgent(nn.Module):
    """Small recurrent (GRU) policy for the probe bandit, implementing `meta.EpisodePolicy` (tests only).

    Inputs per decision: featurised observations delivered after the previous decision (probe signals and pull
    payouts by arm), the previous action, and the BUDGET token (`BudgetEncoder`). Heads: categorical policy over
    [probe, pull_0..pull_{n-1}(, watch)] and `ValueHead` over the projected recurrent state with the BUDGET token.
    With ``math_assist`` the CAM 7.0 per-candidate scores (`CandidateValueModel`) enter through `MathAssistEncoder`."""

    def __init__(self, n_arms: int = 6, D: int = 64, noise_tv: bool = False, math_assist: bool = False,
                 seed: int = 0, pay: float = 100.0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_arms, self.D, self.pay = n_arms, D, pay
        self.names = ["probe"] + [f"pull:{i}" for i in range(n_arms)] + (["watch"] if noise_tv else [])
        self.A = len(self.names)
        self.in_dim = 2 * n_arms + self.A + 1
        self.math_assist = math_assist
        self.core_in = nn.Linear(self.in_dim, D)
        self.budget = BudgetEncoder(D)
        self.assist = MathAssistEncoder(D, enabled=True) if math_assist else None
        self.gru = nn.GRUCell(3 * D if math_assist else 2 * D, D)
        self.proj = nn.Linear(D, D)
        self.value = ValueHead(D, n_heads=4)
        self.pi = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, self.A))
        self.version = 0

    def core_params(self):
        mods = [self.core_in, self.gru, self.proj] + ([self.assist] if self.assist is not None else [])
        return [p for m in mods for p in m.parameters()]

    def head_params(self):
        return [p for m in (self.budget, self.value, self.pi) for p in m.parameters()]

    # ---- episode policy
    def initial_state(self, B):
        st = {"h": torch.zeros(B, self.D), "prev": torch.full((B,), -1, dtype=torch.long)}
        if self.math_assist:
            st["cvm"] = CandidateValueModel(B, self.A, prior_mean=[-30.0] + [0.0] * (self.A - 1), b0=100.0 ** 2)
        return st

    def featurize(self, obs: list, prev: torch.Tensor) -> torch.Tensor:
        B = len(obs)
        x = torch.zeros(B, self.in_dim)
        n = self.n_arms
        for b in range(B):
            for o in obs[b]:
                if o.kind != ObsKind.NUMERIC or not (o.entity or "").startswith("arm_"):
                    continue
                i = int(o.entity.split("_")[1])
                if o.source == "probe":
                    x[b, i] = float(o.value)
                elif o.source == "pull":
                    x[b, n + i] = float(o.value) / self.pay
        valid = prev >= 0
        x[valid, 2 * n + prev[valid]] = 1.0
        x[:, -1] = (~valid).float()
        return x

    def _assist_update(self, st, obs, active):
        cvm = st["cvm"]
        prev = st["prev"]
        ok = (prev >= 0) & active
        if not bool(ok.any()):
            return
        out = torch.zeros(len(obs), dtype=torch.float64)
        for b in range(len(obs)):
            if prev[b] < 0:
                continue
            name = self.names[int(prev[b])]
            val = sum(float(o.value) for o in obs[b] if o.kind == ObsKind.NUMERIC and o.unit == "cents")
            out[b] = val - (30.0 if name == "probe" else 0.0)
        cvm.update(prev.clamp_min(0), out, ok)

    def _step(self, x, bt, h, scores=None):
        parts = [F.silu(self.core_in(x)), bt]
        if self.assist is not None:
            parts.append(self.assist(scores).x.mean(1))
        return self.gru(torch.cat(parts, -1), h)

    def _heads(self, h, btok: TokenSet):
        logits = self.pi(h)
        v = self.value(self.proj(h).unsqueeze(1), extra=[btok])
        return logits, v

    @torch.no_grad()
    def act(self, state, obs, tools, budget, goals, active, greedy=False):
        x = self.featurize(obs, state["prev"])
        btok = self.budget(budget)
        scores = None
        if self.math_assist:
            self._assist_update(state, obs, active)
            scores = state["cvm"].scores()
        h_new = self._step(x, btok.x[:, 0], state["h"], scores)
        h = torch.where(active.unsqueeze(1), h_new, state["h"])
        logits, v = self._heads(h, btok)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(a)
        actions = []
        for b, ai in enumerate(a.tolist()):
            nm = self.names[ai]
            if nm.startswith("pull:"):
                actions.append(Action("pull", {"arm": int(nm.split(":")[1])}, logp=float(logp[b])))
            else:
                actions.append(Action(nm, {}, logp=float(logp[b])))
        rec = {"x": x, "budget": budget.stack(), "a": a}
        if scores is not None:
            rec["scores"] = scores
        new = {"h": h, "prev": torch.where(active, a, state["prev"])}
        if self.math_assist:
            new["cvm"] = state["cvm"]
        return actions, logp, v.value, new, rec

    def recompute(self, traj):
        x, bud, a = traj.extras["x"], traj.extras["budget"], traj.extras["a"].long()
        scores = traj.extras.get("scores")
        B, T = a.shape
        h = torch.zeros(B, self.D)
        hs, bts = [], []
        for t in range(T):
            b6 = bud[:, t]
            bs = BudgetState(*(b6[:, i] for i in range(6)))
            btok = self.budget(bs)
            h_new = self._step(x[:, t], btok.x[:, 0], h, None if scores is None else scores[:, t])
            h = torch.where(traj.mask[:, t].unsqueeze(1), h_new, h)
            hs.append(h)
            bts.append(btok)
        H = torch.stack(hs, 1)                                         # [B,T,D]
        logits = self.pi(H)
        logp_all = torch.log_softmax(logits.float(), -1)
        logp = logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        btx = torch.stack([b.x[:, 0] for b in bts], 1).reshape(B * T, 1, self.D)
        btok = TokenSet.from_dense(btx, TokenKind.BUDGET)
        v = self.value(self.proj(H).reshape(B * T, 1, self.D), extra=[btok])
        vl = v.logits.view(B, T, -1)
        vl = torch.cat([vl, torch.zeros(B, 1, vl.shape[-1])], 1)         # terminal bootstrap = 0
        return {"logp": logp, "entropy": ent, "value_logits": vl, "value_head": self.value}


# ================================================================================================ training loop

def train_probe_agent(agent: ProbeAgent, iters: int = 200, B: int = 128, horizon=(1, 6), n_trials: int = 1,
                      trial_weights=None, epochs: int = 4, lr: float = 2e-3, entropy_coef: float = 0.01,
                      seed0: int = 0, noise_tv: bool = False, log_every: int = 0, grad_probe_every: int = 25,
                      stage="actor_critic", lp_coef: float = 0.0, lp_horizon: int = 8,
                      lp_anneal: float | None = None) -> dict:
    """``lp_anneal``: fraction of the iterations over which the LP coefficient decays linearly to 0 (LP as an
    initial training signal); None keeps it constant."""
    """Train with the production machinery; returns the trainer, per-iteration stats and timing."""
    cfg = make_config("dev", lr=lr, grad_clip=1.0)
    adapter = ProbeBanditAdapter(B, n_arms=agent.n_arms, horizon=horizon, noise_tv=noise_tv)
    runner = TrialRunner(adapter, adapter.evaluator)
    ac = ActorCriticObjective(kappa=cfg.kappa, gae_lambda=0.95, clip=0.2, entropy_coef=entropy_coef,
                              target_kl=0.05, max_fresh_kl=0.05)
    meta = MetaRLObjective(ac, trial_weights=trial_weights, use_intrinsic=lp_coef > 0)
    belief = ArmBelief(agent.n_arms, agent.pay)
    trainer = CombinedTrainer(cfg, core=agent.core_params(), heads=agent.head_params(), stage=stage, amp=False,
                              grad_probe_every=grad_probe_every)
    hist = []
    t0 = time.time()
    for it in range(iters):
        seeds = list(range(seed0 + it * B, seed0 + (it + 1) * B))
        res = runner.run(agent, seeds, n_trials=n_trials)
        lp_mean = 0.0
        coef = lp_coef if lp_anneal is None else lp_coef * max(0.0, 1.0 - it / max(1.0, lp_anneal * iters))
        if lp_coef > 0:
            lpr = arm_learning_progress(res, belief, lp_horizon, coef)
            res.traj.intrinsic = lpr["intrinsic"]
            lp_mean = float(lpr["lp"][res.traj.mask].mean())
        shaped, prep = meta.prepare(res.traj, agent.version)
        last = None
        for _ in range(epochs):
            r = agent.recompute(res.traj)
            d = meta.loss(shaped, prep, r["logp"], r["entropy"], value_logits=r["value_logits"],
                          value_head=r["value_head"])
            last = trainer.step({"actor": d["actor"], "value": d["value"]})
            d["update"] = last
            if d["stop"]:
                break
        agent.version += 1
        probe_rate = sum(sum(a == "probe" for a in row) for row in res.actions) / max(1, int(res.traj.mask.sum()))
        hist.append({"it": it, "return": float(res.per_trial_return.sum(1).mean()), "kl": d["kl"],
                     "entropy": d["entropy"], "update": last, "lp": lp_mean, "probe_rate": probe_rate})
        if log_every and it % log_every == 0:
            print(it, round(hist[-1]["return"], 1), round(d["entropy"], 3), round(d["kl"], 4), round(lp_mean, 3),
                  round(probe_rate, 3))
    return {"trainer": trainer, "hist": hist, "seconds": time.time() - t0}


def evaluate(policy, n_arms: int = 6, horizons=range(1, 7), n_per: int = 200, seed0: int = 1_000_000,
             greedy: bool = True, noise_tv: bool = False) -> dict:
    """Mean verified return and probe-first rate per horizon on held-out seeds."""
    out = {}
    for H in horizons:
        adapter = ProbeBanditAdapter(n_per, n_arms=n_arms, horizon=H, noise_tv=noise_tv)
        runner = TrialRunner(adapter, adapter.evaluator)
        res = runner.run(policy, list(range(seed0 + 1000 * H, seed0 + 1000 * H + n_per)), greedy=greedy)
        first = [a[0] if a else "" for a in res.actions]
        out[H] = {"return": float(res.per_trial_return.sum(1).mean()),
                  "probe_first": sum(f == "probe" for f in first) / n_per,
                  "probes": sum(sum(x == "probe" for x in a) for a in res.actions) / n_per}
    return out
