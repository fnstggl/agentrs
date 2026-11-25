"""Baselines for the controlled experiments (spec Part 19; Experiment 1 "strong Bayesian, recurrent meta-RL,
attention-based, frozen semantic and existing CAM baselines").

Predictive baselines on the identical support of `predict.SupportEpisode` (each returns its own predictive of the
next-tick reward of every query candidate):

* `NIGPredictor(meta=False)` - exact Bayesian regression (normal-inverse-gamma, Student-t predictive; the math of
  D's `learn.math_assist`, ported from CAM 7.0) of the reward on per-lever features [d, d^2, e, e^2] (d = offset
  of the lever from its resting setting, e = the same offset of an exponential moving average of the setting, so
  delayed responses have something to load on) plus an intercept, with a generic prior: spec baseline 4 and Crucible
  `model_based`'s fast layer without its network features.
* `NIGPredictor(meta=True)` - the same regression with its prior (means, precisions per feature type, noise prior)
  meta-learned on training-family episodes by minimising the support/query log predictive (ALPaCA; Crucible
  `model_based`'s across-episode level): the strong Bayesian baseline.
* `SeqPredictor("gru")` - recurrent meta-learner: a GRU reads the support steps (settings, offsets, gauges, reward and
  whether it is revealed) and predicts the reward of a candidate setting from its state (the predictive half of
  RL^2 / Crucible `ppo_gru`; the acting half is run on decisions through Crucible's learner, see `decision_learner`).
* `SeqPredictor("attn")` - attention meta-learner: a causal transformer over the support steps, the candidate as a
  query token attending to them (a transformer neural process; the predictive core of Crucible `meta_np`).

All four see exactly the numeric information the support contains (condition D: hidden gauge columns zero and
masked, unrevealed rewards marked missing) and are trained only on training-family episodes from the training seed
range (never on evaluation tasks).

Decision baselines (agent-selected experiments, conditions B/D, Experiment 2) are Crucible's numeric learners behind
the architecture-neutral contract (`crucible.research.learners`: `model_based`, `ppo_gru`, `meta_np`, scripted
`probe_exploit`, `random`, `hold`) and CAM 7.0 (`superlearner.cam7.agent.Cam7Learner`, explore rules `kg` /
`thompson`: the existing mathematical knowledge-gradient and Thompson-sampling baselines), trained on the same
training families and seed ranges, run through `run_numeric_episodes` on the same validated tasks.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..learn import math_assist as MA

N_TYPES = 4          # d, d^2, e, e^2
L_MAX, G_MAX = 12, 16


# ============================================================================ features

def lever_features(X: np.ndarray, x0: np.ndarray, ema: float = 0.5) -> tuple:
    """X [T, L] normalised settings in force at each tick -> (Phi [T, 1 + 4L], EMA state after each tick [T, L])."""
    T, L = X.shape
    e = np.zeros((T, L))
    cur = x0.copy()
    for t in range(T):
        cur = (1 - ema) * cur + ema * X[t]
        e[t] = cur
    d, de = X - x0, e - x0
    phi = np.concatenate([np.ones((T, 1)), d, d * d, de, de * de], 1)
    return phi, e


def query_features(Xq: np.ndarray, e_prev: np.ndarray, x0: np.ndarray, ema: float = 0.5) -> np.ndarray:
    eq = (1 - ema) * e_prev[None] + ema * Xq
    d, de = Xq - x0, eq - x0
    return np.concatenate([np.ones((len(Xq), 1)), d, d * d, de, de * de], 1)


def _type_index(L: int) -> np.ndarray:
    """Feature -> prior type (0 intercept, 1 d, 2 d^2, 3 e, 4 e^2)."""
    return np.concatenate([[0], np.repeat(np.arange(1, N_TYPES + 1), L)])


# ============================================================================ NIG regression

class NIGPredictor:
    """Exact NIG Bayesian regression of the reward (see the module docstring). Prior per feature type: mean mu_t and
    precision lam_t (w ~ N(mu, s2 / lam)), noise s2 ~ InvGamma(a0, b0)."""

    def __init__(self, meta: bool = False, ema: float = 0.5):
        self.meta, self.ema = bool(meta), float(ema)
        self.name = "nig_meta" if meta else "nig_exact"
        # generic prior (reward units: net per tick over the public money scale): intercept sd 1, linear sd 0.1,
        # quadratic 0.05, noise sd 0.02 (Crucible model_based's generic prior values)
        s2 = 0.02 ** 2
        sd = np.array([1.0, 0.10, 0.05, 0.10, 0.05])
        self.mu = torch.zeros(N_TYPES + 1, dtype=torch.float64)
        self.log_lam = torch.tensor(np.log(s2 / sd ** 2), dtype=torch.float64)
        self.log_a0 = torch.tensor(math.log(2.0), dtype=torch.float64)
        self.log_b0 = torch.tensor(math.log(s2), dtype=torch.float64)
        self.fit_info: dict = {}

    def _prior(self, L: int):
        ti = torch.as_tensor(_type_index(L))
        return (self.mu[ti].unsqueeze(0), self.log_lam.exp()[ti].unsqueeze(0), self.log_a0.exp().reshape(1),
                self.log_b0.exp().reshape(1))

    def _post(self, phi: np.ndarray, y: np.ndarray, w: np.ndarray, L: int):
        mu0, lam0, a0, b0 = self._prior(L)
        p = torch.as_tensor(phi, dtype=torch.float64).unsqueeze(0)
        yy = torch.as_tensor(y, dtype=torch.float64).reshape(1, -1, 1)
        ww = torch.as_tensor(w, dtype=torch.float64).reshape(1, -1)
        return MA.fit(p, yy, ww, mu0, lam0, a0, b0)

    def predict(self, ep, k: int) -> tuple:
        L = ep.X.shape[1]
        phi, e = lever_features(ep.X, ep.x0, self.ema)
        rev = ep.revealed(k)
        if k == 0:
            phi_s, y_s, w_s = np.zeros((1, phi.shape[1])), np.zeros(1), np.zeros(1)
            e_prev = ep.x0
        else:
            phi_s, y_s, w_s = phi[:k], ep.Y[:k], rev.astype(np.float64)
            e_prev = e[k - 1]
        post = self._post(phi_s, y_s, w_s, L)
        pq = query_features(ep.queries[k]["X"], e_prev, ep.x0, self.ema)
        loc, s2, dof = MA.predict(post, torch.as_tensor(pq, dtype=torch.float64).unsqueeze(0))
        return (loc[0, :, 0].numpy(), s2[0, :, 0].numpy(), np.full(len(pq), float(dof[0, 0])))

    def predict_with_notes(self, ep, k: int) -> tuple:
        """Condition C numeric reference: the predecessor notes' numbers added as support rows (resting settings with
        one lever moved, the reported net after the move, at a resting EMA of the moved setting)."""
        L = ep.X.shape[1]
        phi, e = lever_features(ep.X, ep.x0, self.ema)
        scale = float(ep.goal.meta["reward_scale_cents"])
        rows, ys = [], []
        for j, v, base, after in ep.notes:
            x = ep.x0.copy()
            x[j] = 0.0 if ep.hi[j] <= ep.lo[j] else 2.0 * (v - ep.lo[j]) / (ep.hi[j] - ep.lo[j]) - 1.0
            # two ticks at the moved setting: EMA = x0 + 0.75 (x - x0)
            ee = ep.x0 + 0.75 * (x - ep.x0)
            rows.append(query_features(x[None], (ee - self.ema * x) / (1 - self.ema), ep.x0, self.ema)[0])
            ys.append(after / scale)
        if k > 0:
            rows += list(phi[:k][ep.revealed(k)])
            ys += list(ep.Y[:k][ep.revealed(k)])
        phi_s = np.array(rows) if rows else np.zeros((1, phi.shape[1]))
        y_s = np.array(ys) if ys else np.zeros(1)
        w_s = np.ones(len(rows)) if rows else np.zeros(1)
        post = self._post(phi_s, y_s, w_s, L)
        e_prev = e[k - 1] if k > 0 else ep.x0
        pq = query_features(ep.queries[k]["X"], e_prev, ep.x0, self.ema)
        loc, s2, dof = MA.predict(post, torch.as_tensor(pq, dtype=torch.float64).unsqueeze(0))
        return (loc[0, :, 0].numpy(), s2[0, :, 0].numpy(), np.full(len(pq), float(dof[0, 0])))

    def fit_meta(self, episodes: Sequence, steps: int = 300, lr: float = 0.05, seed: int = 0,
                 ks: Sequence[int] = (0, 1, 2, 4, 8, 16, 32)) -> dict:
        """Meta-learn the prior on training episodes: minimise the Student-t NLL of the next-step reward y_k (the
        reward of the design action actually taken at k) given the revealed support < k, over k in `ks`."""
        if not self.meta:
            raise ValueError("fit_meta on the generic-prior baseline")
        params = [self.mu.requires_grad_(), self.log_lam.requires_grad_(), self.log_a0.requires_grad_(),
                  self.log_b0.requires_grad_()]
        opt = torch.optim.Adam(params, lr=lr)
        rng = np.random.default_rng(seed)
        data = []
        for ep in episodes:
            phi, e = lever_features(ep.X, ep.x0, self.ema)
            data.append((ep, phi))
        hist = []
        t0 = time.time()
        for it in range(steps):
            batch = [data[i] for i in rng.choice(len(data), size=min(32, len(data)), replace=False)]
            loss = 0.0
            n = 0
            for ep, phi in batch:
                L = ep.X.shape[1]
                for k in ks:
                    if k >= len(ep.Y):
                        continue
                    rev = ep.revealed(k).astype(np.float64) if k else np.zeros(1)
                    ps = phi[:k] if k else np.zeros((1, phi.shape[1]))
                    ys = ep.Y[:k] if k else np.zeros(1)
                    post = self._post(ps, ys, rev, L)
                    loc, s2, dof = MA.predict(post, torch.as_tensor(phi[k:k + 1], dtype=torch.float64).unsqueeze(0))
                    nll = MA.t_nll(torch.tensor([[[float(ep.Y[k])]]], dtype=torch.float64), loc, s2, dof)
                    loss = loss + nll.sum()
                    n += 1
            loss = loss / max(1, n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.log_a0.clamp_(math.log(1.01), math.log(1e4))
                self.log_lam.clamp_(-20.0, 20.0)
            hist.append(float(loss))
        for p in params:
            p.requires_grad_(False)
        self.fit_info = {"steps": steps, "episodes": len(episodes), "loss_first": hist[0], "loss_last": hist[-1],
                         "wall_s": time.time() - t0, "mu": self.mu.tolist(), "lam": self.log_lam.exp().tolist(),
                         "a0": float(self.log_a0.exp()), "b0": float(self.log_b0.exp())}
        return self.fit_info

    def state_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "log_lam": self.log_lam.tolist(), "log_a0": float(self.log_a0),
                "log_b0": float(self.log_b0), "meta": self.meta, "ema": self.ema}

    def load_state_dict(self, st: dict) -> None:
        self.mu = torch.tensor(st["mu"], dtype=torch.float64)
        self.log_lam = torch.tensor(st["log_lam"], dtype=torch.float64)
        self.log_a0 = torch.tensor(st["log_a0"], dtype=torch.float64)
        self.log_b0 = torch.tensor(st["log_b0"], dtype=torch.float64)


# ============================================================================ sequence meta-learners

def step_inputs(ep, k_max: Optional[int] = None) -> np.ndarray:
    """[T, F] per-step inputs: settings (padded to L_MAX), lever mask, offsets, gauges (padded to G_MAX), gauge mask,
    and the reward *as it arrives*: row t carries the reward of step t - delay (condition D; delay 0 = its own) and a
    flag whether one arrived, so a causal pass over rows < k knows exactly the rewards revealed at decision k."""
    T = len(ep.Y) if k_max is None else min(int(k_max), len(ep.Y))
    L, Gn = ep.X.shape[1], ep.G.shape[1]
    out = np.zeros((T, 2 * L_MAX + L_MAX + 2 * G_MAX + 2), dtype=np.float32)
    for t in range(T):
        out[t, :L] = ep.X[t]
        out[t, L_MAX:L_MAX + L] = 1.0
        out[t, 2 * L_MAX:2 * L_MAX + L] = ep.X[t] - ep.x0
        out[t, 3 * L_MAX:3 * L_MAX + Gn] = ep.G[t]
        out[t, 3 * L_MAX + G_MAX:3 * L_MAX + G_MAX + Gn] = ep.gmask
        src = t - ep.delay
        out[t, -2] = float(ep.Y[src]) if src >= 0 else 0.0
        out[t, -1] = 1.0 if src >= 0 else 0.0
    return out


def query_inputs(Xq: np.ndarray, x0: np.ndarray) -> np.ndarray:
    Q, L = Xq.shape
    out = np.zeros((Q, 3 * L_MAX), dtype=np.float32)
    out[:, :L] = Xq
    out[:, L_MAX:L_MAX + L] = 1.0
    out[:, 2 * L_MAX:2 * L_MAX + L] = Xq - x0
    return out


class _SeqNet(nn.Module):
    def __init__(self, kind: str, d_in: int, d_q: int, hidden: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(d_in, hidden)
        self.qin = nn.Linear(d_q, hidden)
        self.start = nn.Parameter(torch.zeros(hidden))
        if kind == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        else:
            enc = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, 0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.pos = nn.Parameter(torch.randn(256, hidden) * 0.02)
            self.att = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def contexts(self, steps: torch.Tensor) -> torch.Tensor:
        """steps [B, T, F] -> contexts c [B, T+1, H]: c[:, k] summarises steps < k (c[:, 0] = learned start)."""
        B, T, _ = steps.shape
        h = self.inp(steps)
        if self.kind == "gru":
            out, _ = self.rnn(h)
            return torch.cat([self.start.expand(B, 1, -1), out], 1)
        h = h + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=steps.device), 1)
        enc = self.enc(h, mask=mask)
        return torch.cat([self.start.expand(B, 1, -1), enc], 1)

    def predict(self, c: torch.Tensor, q: torch.Tensor, k: torch.Tensor, enc_all: Optional[torch.Tensor] = None):
        """c [N, H] context rows; q [N, Dq] query inputs -> (mean [N], log var [N])."""
        qh = self.qin(q)
        o = self.head(torch.cat([c, qh], -1))
        return o[:, 0], o[:, 1].clamp(-12, 6)


class SeqPredictor:
    """Recurrent ("gru") or attention ("attn") meta-learned predictor (see the module docstring)."""

    def __init__(self, kind: str = "gru", hidden: int = 128, device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.kind = kind
        self.name = f"seq_{kind}"
        self.device = device
        self.net = _SeqNet(kind, 2 * L_MAX + L_MAX + 2 * G_MAX + 2, 3 * L_MAX, hidden).to(device)
        self.fit_info: dict = {}

    def _batch(self, eps: Sequence):
        T = max(len(e.Y) for e in eps)
        S = np.zeros((len(eps), T, self.net.inp.in_features), dtype=np.float32)
        Qx = np.zeros((len(eps), T, 3 * L_MAX), dtype=np.float32)
        Y = np.zeros((len(eps), T), dtype=np.float32)
        M = np.zeros((len(eps), T), dtype=np.float32)
        for i, e in enumerate(eps):
            n = len(e.Y)
            S[i, :n] = step_inputs(e)
            Qx[i, :n] = query_inputs(e.X, e.x0)
            Y[i, :n] = e.Y
            M[i, :n] = 1.0
        f = lambda a: torch.as_tensor(a, device=self.device)   # noqa: E731
        return f(S), f(Qx), f(Y), f(M)

    def fit(self, episodes: Sequence, steps: int = 1500, lr: float = 1e-3, batch: int = 32, seed: int = 0) -> dict:
        """Next-step NLL on training episodes: the reward of the design action at k from the context of steps < k
        (with delayed rewards arriving as they would, see `step_inputs`)."""
        rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        hist = []
        t0 = time.time()
        self.net.train()
        for it in range(steps):
            idx = rng.choice(len(episodes), size=min(batch, len(episodes)), replace=False)
            S, Qx, Y, M = self._batch([episodes[i] for i in idx])
            c = self.net.contexts(S)[:, :-1]                         # context before each step
            B, T, H = c.shape
            mean, lv = self.net.predict(c.reshape(B * T, H), Qx.reshape(B * T, -1), None)
            nll = 0.5 * (lv + (Y.reshape(-1) - mean) ** 2 / lv.exp() + math.log(2 * math.pi))
            loss = (nll * M.reshape(-1)).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            hist.append(float(loss))
        self.net.eval()
        self.fit_info = {"steps": steps, "episodes": len(episodes), "loss_first": float(np.mean(hist[:20])),
                         "loss_last": float(np.mean(hist[-20:])), "wall_s": time.time() - t0,
                         "params": sum(p.numel() for p in self.net.parameters())}
        return self.fit_info

    @torch.no_grad()
    def predict(self, ep, k: int) -> tuple:
        S = step_inputs(ep, k_max=max(k, 1))
        if k == 0:
            S = S[:0]
        St = torch.as_tensor(S, device=self.device).unsqueeze(0)
        if k == 0:
            c = self.net.start.unsqueeze(0)
        else:
            c = self.net.contexts(St)[:, k]
        Qx = torch.as_tensor(query_inputs(ep.queries[k]["X"], ep.x0), device=self.device)
        mean, lv = self.net.predict(c.expand(len(Qx), -1), Qx, None)
        return mean.cpu().numpy().astype(np.float64), lv.exp().cpu().numpy().astype(np.float64), None

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd)


# ============================================================================ numeric decision learners (Crucible)

def decision_learner(name: str, seed: int = 0, device: str = "cpu", **kw):
    """A Crucible numeric learner behind `research.learner.Learner` (spec EnvSpec of the MechWorld operator), or CAM 7.0
    (``cam7``, ``cam7_thompson``: Cam7Learner small scale with the KG or the Thompson exploration rule)."""
    from crucible.research.learner import EnvSpec
    spec = EnvSpec(max_levers=32, max_gauges=32)
    if name.startswith("cam7"):
        from superlearner.cam7.agent import Cam7Learner
        rule = "thompson" if name == "cam7_thompson" else kw.pop("explore_rule", "kg")
        return Cam7Learner(spec, seed=seed, device=device, scale="small", explore_rule=rule, **kw)
    from crucible.research import learners
    return learners.make(name, spec, seed=seed, device=device, **kw)


def run_numeric_episodes(learner, tasks: Sequence, *, learn: bool = False, explore: bool = False,
                         delay: int = 0, hide_frac: float = 0.0, seed: int = 0) -> list:
    """One episode per task with a Crucible learner on MechVecEnv (the same mechanisms the MechWorld adapter wraps;
    parity tested by G). Returns per-task dicts with per-tick nets. ``delay`` / ``hide_frac``: condition D at the
    numeric interface (hidden gauge columns zeroed with their mask, reward/revenue/cost scalars of each tick shown
    `delay` ticks late; the learner's training reward is delayed the same way)."""
    from crucible.research.envs import MechVecEnv
    from crucible.research.learner import EnvSpec
    from .mechrun import ensure_registered
    ensure_registered()                       # catalogue families must be registered in this process
    spec = EnvSpec(max_levers=32, max_gauges=32)
    env = MechVecEnv(list(tasks), spec)
    obs = env.reset()
    learner.reset_context(env.n)
    if learner.frozen and learn:
        raise ValueError("learn=True with a frozen learner")
    rng = np.random.default_rng(seed)
    hid = np.zeros((env.n, spec.max_gauges), dtype=bool)
    if hide_frac > 0:
        for i in range(env.n):
            ng = int(env._gmask[i].sum())
            k = int(math.floor(hide_frac * ng))
            hid[i, rng.choice(ng, size=k, replace=False)] = True
    sc = spec.scalars
    hist_sc = []
    nets = [[] for _ in range(env.n)]
    rew_hist = []

    def view(o, t):
        o = o.copy()
        if hide_frac > 0:
            g = o[:, spec.gauges]
            g[hid] = 0.0
            o[:, spec.gauges] = g
            gm = o[:, spec.gauge_mask]
            gm[hid] = 0.0
            o[:, spec.gauge_mask] = gm
        if delay > 0:
            past = hist_sc[t - 1 - delay] if t - 1 - delay >= 0 else np.zeros_like(o[:, sc][:, :3])
            s = o[:, sc]
            s[:, :3] = past
            o[:, sc] = s
        return o
    t = 0
    v = view(obs, 0)
    while not env.done.all():
        act = learner.act(v, explore=explore)
        nxt, rew, done, info = env.step(act)
        hist_sc.append(nxt[:, sc][:, :3].copy())
        rew_hist.append(rew.copy())
        t += 1
        nv = view(nxt, t)
        r_obs = rew_hist[t - 1 - delay] if delay > 0 and t - 1 - delay >= 0 else (rew if delay == 0
                                                                                  else np.zeros_like(rew))
        if learn and not learner.frozen:
            learner.observe(v, act, r_obs, done, nv)
        for i in range(env.n):
            if not (len(nets[i]) >= int(env._h[i])):
                nets[i].append(float(info["net_cents"][i]))
        v = nv
    recs = env.drain()
    out = []
    for i, task in enumerate(env.tasks):
        n = nets[i][: int(env._h[i])]
        out.append({"task": task.id, "family": task.family, "nets": n,
                    "score": task.score(float(np.mean(n))) if task.validated else None,
                    "success": task.success(float(np.sum(n))) if task.validated else None})
    return out
