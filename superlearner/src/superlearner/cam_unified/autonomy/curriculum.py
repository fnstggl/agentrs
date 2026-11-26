"""Learned autonomous curriculum C_chi and learning-gain estimation (meta-reward spec 8.3, 9.3, 9.4; manifest AU-03,
KV-02).

    g_self ~ C_chi(b_t, lifetime_state, uncertainty, available_resources)

`CurriculumPolicy` scores candidate self-goals (from `GoalCompiler.candidates`, i.e. instantiated over what the agent
currently observes) with a learned network over CAM's pooled latent state, a lifetime-state vector and each
candidate's descriptor (+ optional goal-token embedding from the shared goal encoder). It returns a categorical
distribution with exact log-probabilities, so it is trained by policy gradient.

Its training signal is the *measured learning gain* of the experience that pursuing the goal produced (KV-02):

    learning_gain(D_g) = J_query(U(theta, D_g)) - J_query(theta_control)

measured on independent training-query tasks that were NOT disclosed during the goal-free period, with a matched
control learner (same compute, no D_g, or D_random). Completing a self-goal earns nothing by itself (trivial goals
would otherwise win) and nothing rewards difficulty per se (impossible goals would). Because measuring the gain is
expensive, `LearningGainPredictor` regresses measured gains from (latent, goal descriptor, outcome of the attempt)
and the curriculum is trained on predicted gains between measurements, the predictor being corrected whenever a
real measurement arrives (and it reports its calibration on held-out measurements).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

N_GOAL_FEATS = 5
N_LIFE_FEATS = 6


@dataclass
class CurriculumChoice:
    index: Tensor        # [B] chosen candidate
    logp: Tensor         # [B] log-probability of the choice (behaviour probability)
    probs: Tensor        # [B, C]
    entropy: Tensor      # [B]


class CurriculumPolicy(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, n_goal_feats: int = N_GOAL_FEATS,
                 n_life_feats: int = N_LIFE_FEATS, d_goal_emb: int = 0):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(d_model + n_life_feats, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.goal = nn.Sequential(nn.Linear(n_goal_feats + d_goal_emb, d_hidden), nn.SiLU(),
                                  nn.Linear(d_hidden, d_hidden))
        self.score = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))
        self.d_goal_emb = d_goal_emb

    def logits(self, b: Tensor, life: Tensor, goal_feats: Tensor, mask: Tensor,
               goal_emb: Optional[Tensor] = None) -> Tensor:
        """b [B, D], life [B, Fl], goal_feats [B, C, Fg], mask [B, C] -> [B, C]."""
        s = self.state(torch.cat([b.float(), life.float()], -1))
        gx = goal_feats.float() if goal_emb is None else torch.cat([goal_feats.float(), goal_emb.float()], -1)
        g = self.goal(gx)
        x = torch.cat([s[:, None, :].expand_as(g), g], -1)
        return self.score(x).squeeze(-1).masked_fill(~mask.bool(), float("-inf"))

    def choose(self, b, life, goal_feats, mask, goal_emb=None, greedy: bool = False,
               generator: Optional[torch.Generator] = None) -> CurriculumChoice:
        lg = self.logits(b, life, goal_feats, mask, goal_emb)
        p = torch.softmax(lg, -1)
        idx = p.argmax(-1) if greedy else torch.multinomial(p.detach().cpu(), 1, generator=generator).squeeze(-1).to(p.device)
        logp = torch.log_softmax(lg, -1).gather(-1, idx[:, None]).squeeze(-1)
        ent = -(p * torch.log_softmax(lg, -1).masked_fill(~mask.bool(), 0)).sum(-1)
        return CurriculumChoice(idx, logp, p, ent)

    def loss(self, b, life, goal_feats, mask, chosen: Tensor, gain: Tensor, baseline: Optional[Tensor] = None,
             entropy_coef: float = 1e-2, goal_emb=None) -> Tensor:
        """REINFORCE on (predicted or measured) learning gain of the chosen goal."""
        lg = self.logits(b, life, goal_feats, mask, goal_emb)
        logp = torch.log_softmax(lg, -1)
        lp = logp.gather(-1, chosen.long()[:, None]).squeeze(-1)
        adv = gain.float() - (baseline.float() if baseline is not None else gain.float().mean())
        p = logp.exp()
        ent = -(p * logp.masked_fill(~mask.bool(), 0)).sum(-1)
        return -(lp * adv.detach()).mean() - entropy_coef * ent.mean()


class LearningGainPredictor(nn.Module):
    """Predicts the measured learning gain of the experience produced by pursuing a goal."""

    def __init__(self, d_model: int, d_hidden: int = 256, n_goal_feats: int = N_GOAL_FEATS, n_outcome: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model + n_goal_feats + n_outcome, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 2))

    def forward(self, b: Tensor, goal_feats: Tensor, outcome: Tensor) -> tuple:
        o = self.net(torch.cat([b.float(), goal_feats.float(), outcome.float()], -1))
        return o[..., 0], F.softplus(o[..., 1]) + 1e-3          # mean, std

    def loss(self, b, goal_feats, outcome, measured: Tensor) -> Tensor:
        mu, sd = self(b, goal_feats, outcome)
        return (0.5 * ((measured.float() - mu) / sd) ** 2 + torch.log(sd)).mean()


def measured_learning_gain(j_after: Tensor, j_control: Tensor) -> Tensor:
    """learning_gain(D) = J_query(U(theta, D)) - J_query(theta_control): both evaluated on the same independent query
    tasks with matched compute; per-task differences [n] (the mean is the estimate, its SE the uncertainty)."""
    return j_after.float() - j_control.float()


# ================================================================================================ measured gain

@dataclass
class GainMeasurement:
    """One measured learning gain (KV-02): the gain of the experience a goal produced, on independent query tasks,
    against a matched-compute control learner."""
    gain: float                  # mean over query tasks of J_after - J_control
    se: float                    # standard error over query tasks
    per_task: list
    n_query: int
    j_after: float
    j_control: float


def measure_learning_gain(learner, data, query_tasks: Sequence, *, update_fn: Callable, evaluate_fn: Callable,
                          control_data=None) -> GainMeasurement:
    """learning_gain(D_g) = J_query(U(theta, D_g)) - J_query(U(theta, D_control)).

    `learner` is cloned (deepcopy) twice and never modified; update_fn(clone, data) applies the learner's own
    update rule; control_data (default: no data) receives the same update_fn, so compute is matched when it holds
    as many examples as `data` (e.g. experience from a random goal); evaluate_fn(clone, task) -> J on one independent
    query task (tasks NOT disclosed during the goal-free period)."""
    a = copy.deepcopy(learner)
    update_fn(a, data)
    c = copy.deepcopy(learner)
    if control_data is not None:
        update_fn(c, control_data)
    ja = torch.tensor([float(evaluate_fn(a, t)) for t in query_tasks], dtype=torch.float64)
    jc = torch.tensor([float(evaluate_fn(c, t)) for t in query_tasks], dtype=torch.float64)
    d = measured_learning_gain(ja, jc).double()
    n = d.numel()
    se = float(d.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
    return GainMeasurement(float(d.mean()), se, d.tolist(), n, float(ja.mean()), float(jc.mean()))


class CurriculumTrainer:
    """C_chi training on measured learning gain (meta-reward spec 9.3-9.4; AU-03, KV-02).

    Per round: `choose` a candidate goal (behaviour log-probability recorded); pursue it (caller); then either a
    measured gain arrives (`record`: the LearningGainPredictor takes NLL steps on it; every `holdout_every`-th
    measurement is kept back as a held-out calibration example instead) or the predictor's mean stands in for it
    (`signal`). `policy_step` is REINFORCE on that gain with a running-mean baseline and an entropy bonus. Goal
    completion and difficulty earn nothing by themselves: the only signal is (measured or predicted) learning gain."""

    def __init__(self, policy: CurriculumPolicy, predictor: Optional[LearningGainPredictor] = None, *,
                 lr: float = 3e-3, pred_lr: float = 3e-3, entropy_coef: float = 1e-2, baseline_momentum: float = 0.1,
                 pred_steps: int = 4, holdout_every: int = 0):
        self.policy, self.predictor = policy, predictor
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.pred_opt = torch.optim.Adam(predictor.parameters(), lr=pred_lr) if predictor is not None else None
        self.entropy_coef = entropy_coef
        self.mom = baseline_momentum
        self.baseline: Optional[float] = None
        self.pred_steps = pred_steps
        self.holdout_every = int(holdout_every)
        self.train_set: list = []
        self.held_out: list = []
        self.n_measured = 0
        self.log: list = []

    def choose(self, b, life, goal_feats, mask, goal_emb=None, generator=None) -> CurriculumChoice:
        with torch.no_grad():
            return self.policy.choose(b, life, goal_feats, mask, goal_emb, generator=generator)

    def record(self, b: Tensor, goal_feat: Tensor, outcome: Tensor, measured: Tensor) -> dict:
        """Add measured gains (rows: [B, D], [B, Fg], [B, n_outcome], [B]); train the predictor on the train set."""
        out = {}
        for i in range(measured.shape[0]):
            self.n_measured += 1
            ex = (b[i].detach(), goal_feat[i].detach(), outcome[i].detach(), measured[i].detach().float())
            if self.holdout_every and self.n_measured % self.holdout_every == 0:
                self.held_out.append(ex)
            else:
                self.train_set.append(ex)
        if self.predictor is not None and self.train_set:
            bb, gg, oo, yy = (torch.stack(z) for z in zip(*self.train_set))
            for _ in range(self.pred_steps):
                L = self.predictor.loss(bb, gg, oo, yy)
                self.pred_opt.zero_grad(set_to_none=True)
                L.backward()
                self.pred_opt.step()
            out["pred_nll"] = float(L.detach())
        return out

    def signal(self, b: Tensor, goal_feat: Tensor, outcome: Tensor, measured: Optional[Tensor] = None) -> Tensor:
        """The gain the policy is trained on: measured when available, else the predictor's mean (detached)."""
        if measured is not None:
            return measured.detach().float()
        if self.predictor is None:
            raise ValueError("no measurement and no predictor")
        with torch.no_grad():
            mu, _ = self.predictor(b, goal_feat, outcome)
        return mu.detach().float()

    def policy_step(self, b, life, goal_feats, mask, chosen: Tensor, gain: Tensor, goal_emb=None) -> dict:
        g = gain.detach().float()
        base = self.baseline if self.baseline is not None else float(g.mean())
        L = self.policy.loss(b, life, goal_feats, mask, chosen, g, baseline=torch.full_like(g, base),
                             entropy_coef=self.entropy_coef, goal_emb=goal_emb)
        self.opt.zero_grad(set_to_none=True)
        L.backward()
        self.opt.step()
        m = float(g.mean())
        self.baseline = m if self.baseline is None else (1 - self.mom) * self.baseline + self.mom * m
        st = {"loss": float(L.detach()), "gain_mean": m, "baseline": self.baseline}
        self.log.append(st)
        return st

    def calibration(self) -> dict:
        """Predictor calibration on held-out measurements: Gaussian NLL, mean |z|, +-1 sd coverage, correlation."""
        if self.predictor is None or not self.held_out:
            return {"n": 0}
        bb, gg, oo, yy = (torch.stack(z) for z in zip(*self.held_out))
        with torch.no_grad():
            mu, sd = self.predictor(bb, gg, oo)
        z = (yy - mu) / sd
        nll = float((0.5 * z ** 2 + torch.log(sd) + 0.5 * math.log(2 * math.pi)).mean())
        corr = float(torch.corrcoef(torch.stack([mu, yy]))[0, 1]) if yy.numel() > 2 and float(yy.std()) > 0 else \
            float("nan")
        out = {"n": int(yy.numel()), "nll": nll, "mean_abs_z": float(z.abs().mean()),
               "coverage_1sd": float((z.abs() <= 1).float().mean()), "corr": corr,
               "rmse": float(((mu - yy) ** 2).mean().sqrt())}
        if self.train_set:                    # reference: a constant Gaussian fitted to the training measurements
            yt = torch.stack([e[3] for e in self.train_set])
            m0, s0 = float(yt.mean()), max(float(yt.std()), 1e-6) if yt.numel() > 1 else 1.0
            out["nll_constant"] = float((0.5 * ((yy - m0) / s0) ** 2 + math.log(s0) + 0.5 * math.log(2 * math.pi)).mean())
            out["rmse_constant"] = float(((yy - m0) ** 2).mean().sqrt())
        return out
