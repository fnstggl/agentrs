"""`model_based`: learn a predictive model of the world at hand, act by planning with it.

Two levels, kept apart exactly as the learner contract separates them.

(i) Inside an episode (fast adaptation, no weights change). Revenue and cost of a tick are regressed on features
    of the settings in force::

        y_t  =  bias  +  sum over levers j of  w_j . psi(d_j, e_j, s_j)  +  noise

    where d_j is the lever's offset from where it rested at the start of the episode, e_j the same offset of an
    exponential moving average of the lever (so slow responses - delays, stocks - have something to load on) and
    s_j the resting position itself. psi is `[d, d^2, e, e^2]` followed by the outputs of a small network. The
    weights w have a Gaussian prior and the regression is exact Bayesian linear regression (one for revenue, one
    for cost; the noise variance of each is estimated inside the episode from consecutive ticks at unchanged
    settings, starting from a prior value). Every `dwell` ticks the learner draws one sample of the weights from
    the posterior (Thompson sampling, with a temperature that falls to zero at `explore_until` of the horizon),
    and searches a grid of candidate settings for every lever inside a trust region around the current settings
    for the candidate that maximises the sampled model's steady-state net. The model is additive over levers, so
    the coordinate search is exact; every candidate evaluated is counted in `usage().search_evals`. If the
    model's one-step-ahead errors suddenly become several times larger than they were (a regime change), the
    posterior is reset to the prior and exploration restarts.

(ii) Across episodes (parametric learning: `observe` + `update`). The feature network, the prior mean and
    precision of the weights and the prior of the noise model are trained on the stream of training episodes by
    gradient descent on the negative log posterior-predictive density of held-out ticks: the regression is run
    on the first n ticks of a stored episode (n random) and scored on the ticks that follow. This is a
    meta-learned Bayesian last layer (ALPaCA: Harrison, Sharma and Pavone 2018) with one weight block per lever
    and everything else shared across levers, so nothing learned is tied to a slot. What it buys is a faster and
    better calibrated identification inside the next unfamiliar world: the prior says which way costs and
    revenues usually move and how noisy a tick is before the first experiment.

An untrained learner (no `update` yet) is level (i) alone with a generic prior: zero mean, a fixed scale for the
linear and quadratic terms, and the (random) network features under a prior scale so small that they contribute
next to nothing. `freeze(True)` stops (ii) and nothing else.

`explore=False` does not switch off the experiments inside an episode (without them nothing can be identified);
it makes the episode reproducible: the Thompson draws of slot i then come from a generator re-seeded at
`reset_context`.
"""
from __future__ import annotations

import time

import numpy as np

from ..learner import EnvSpec, Learner
from .util import cpu_adam

MB_DEFAULTS = {
    # acting
    "dwell": 2, "grid": 9, "trust": 0.3, "temp": 0.5, "explore_until": 0.4, "ema": 0.5, "min_funds": 0.15,
    "surprise": 4.0, "surprise_window": 8,
    # the generic prior of an untrained learner
    "prior_sd": 0.10, "prior_sd_quad": 0.05, "prior_sd_feat": 0.02, "noise_sd_rev": 0.02, "noise_sd_cost": 0.005,
    "noise_n0": 4.0,
    # the meta-learned part
    "feat_dim": 6, "feat_hidden": 32, "lr": 5e-3, "batch_episodes": 24, "buffer_episodes": 1024,
    "min_episodes": 48, "updates_per_episode": 1.0, "contexts": 3, "query": 24, "max_ticks": 256,
}
N_BASE = 4                       # d, d^2, e, e^2
BIAS_SD = 1.0


def _torch():
    import torch
    return torch


def _build_net(p: dict, seed: int):
    torch = _torch()
    nn = torch.nn

    class Prior(nn.Module):
        """Everything (ii) learns. `mu`, `log_lam`: prior mean and log precision of a lever's weight block, per
        target (0 revenue, 1 cost); `log_s2`, `log_n0`: prior noise variance and its pseudo-count per target."""

        def __init__(self):
            super().__init__()
            m, h = int(p["feat_dim"]), int(p["feat_hidden"])
            self.m = m
            self.mlp = nn.Sequential(nn.Linear(3, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, m))
            f = N_BASE + m
            sd = torch.full((2, f), float(p["prior_sd"]))
            sd[:, 1] = float(p["prior_sd_quad"])
            sd[:, 3] = float(p["prior_sd_quad"])
            # The network features of an untrained learner are random and their weights have a small prior
            # scale: next to nothing in the regression, but alive (were they exactly zero, the predictive
            # density would not depend on the network to first order and training could never leave that point).
            sd[:, N_BASE:] = float(p["prior_sd_feat"])
            self.mu = nn.Parameter(torch.zeros(2, f))
            self.log_lam = nn.Parameter(-2.0 * torch.log(sd))
            self.log_s2 = nn.Parameter(2.0 * torch.log(torch.tensor([float(p["noise_sd_rev"]),
                                                                    float(p["noise_sd_cost"])])))
            self.log_n0 = nn.Parameter(torch.log(torch.full((2,), float(p["noise_n0"]))))

        def feats(self, d, e, s):
            z = self.mlp(torch.stack([d, e, s], dim=-1))
            return torch.cat([torch.stack([d, d * d, e, e * e], dim=-1), z], dim=-1)

    g = torch.Generator().manual_seed(int(seed))
    net = Prior()
    with torch.no_grad():                               # seeded initialisation, independent of global state
        for name, q in net.named_parameters():
            if name.startswith("mlp"):
                bound = 1.0 / np.sqrt(q.shape[-1]) if q.dim() > 1 else 0.1
                q.copy_((torch.rand(q.shape, generator=g) * 2 - 1) * bound)
    return net.double()


class _NP:
    """The prior as numpy arrays: what acting uses (no torch call per tick)."""

    def __init__(self, net):
        sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
        self.w = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in (0, 2, 4)]
        self.mu, self.lam = sd["mu"], np.exp(np.clip(sd["log_lam"], -12.0, 16.0))
        self.s2 = np.exp(np.clip(sd["log_s2"], -18.0, 2.0))
        self.n0 = np.exp(np.clip(sd["log_n0"], -3.0, 6.0))
        self.f = self.mu.shape[1]

    def feats(self, d: np.ndarray, e: np.ndarray, s: np.ndarray) -> np.ndarray:
        h = np.stack([d, e, s], axis=-1)
        h = np.tanh(h @ self.w[0][0].T + self.w[0][1])
        h = np.tanh(h @ self.w[1][0].T + self.w[1][1])
        return np.concatenate([np.stack([d, d * d, e, e * e], axis=-1), h @ self.w[2][0].T + self.w[2][1]], axis=-1)


class _MBSlot:
    def __init__(self, owner: "ModelBasedLearner", rng: np.random.Generator):
        self.o, self.rng = owner, rng
        self.t = 0
        self.h = 120
        self.safe = False

    # ---- regression state
    def _clear(self) -> None:
        dim = self.dim
        self.G = np.zeros((dim, dim))
        self.cr, self.cc = np.zeros(dim), np.zeros(dim)
        self.resid: list[float] = []
        self.mean_net: np.ndarray | None = None
        self.t_reset = self.t

    def _phi(self, x: np.ndarray) -> np.ndarray:
        m = self.o._np
        f = m.feats(x - self.x0, self.ema - self.x0, self.x0)          # [L, F]
        return np.concatenate([f.ravel(), [1.0]])

    def step(self, obs: np.ndarray) -> np.ndarray:
        o, p, sp = self.o, self.o.p, self.o.spec
        sc = obs[sp.scalars]
        rev, cost, funds, clock = float(sc[1]), float(sc[2]), float(sc[3]), float(sc[4])
        if self.t == 0:
            self.levers = np.flatnonzero(obs[sp.lever_mask] > 0.5)
            self.x0 = np.asarray(obs[sp.last_action], dtype=np.float64)[self.levers].copy()
            self.ema, self.target = self.x0.copy(), self.x0.copy()
            self.dim = len(self.levers) * o._np.f + 1
            self.ss, self.n_held, self.prev = np.zeros(2), 0, None
            self.since, self.resets, self.clock0 = 0, 0, clock
            self._clear()
        elif clock >= 1.0 - 1e-9:
            pass                                    # the episode is over (a batch keeps calling until all are)
        else:
            if self.t == 1 and clock > self.clock0:        # ticks left when this context began
                self.h = max(2, int(round((1.0 - self.clock0) / (clock - self.clock0))))
            x = np.asarray(obs[sp.last_action], dtype=np.float64)[self.levers]
            a = float(p["ema"])
            self.ema = (1.0 - a) * self.ema + a * x
            phi = self._phi(x)
            if self.mean_net is not None:
                self.resid.append(float(rev - cost - phi @ self.mean_net))
            if self.prev is not None and np.allclose(x, self.prev[0], atol=1e-6):
                self.ss += 0.5 * np.array([(rev - self.prev[1]) ** 2, (cost - self.prev[2]) ** 2])
                self.n_held += 1
            self.G += np.outer(phi, phi)
            self.cr += phi * rev
            self.cc += phi * cost
            self.prev = (x, rev, cost)
            self.since += 1
            if not self.safe and funds < p["min_funds"]:
                self.safe, self.target = True, self.x0.copy()
            if not self.safe and self.since >= int(p["dwell"]):
                self._decide(x)
        out = np.zeros(sp.act_dim, dtype=np.float32)
        out[self.levers] = self.target
        self.t += 1
        return out

    def _decide(self, x: np.ndarray) -> None:
        o, p, m = self.o, self.o.p, self.o._np
        self.since = 0
        w = int(p["surprise_window"])
        if len(self.resid) >= 3 * w:
            r2 = np.square(self.resid)
            if r2[-w:].mean() > float(p["surprise"]) ** 2 * max(r2[w:-w].mean(), 1e-8) and self.h - self.t > 4 * w:
                self._clear()                                   # the world changed: forget, explore again
                self.resets += 1
        s2 = (m.n0 * m.s2 + self.ss) / (m.n0 + self.n_held)
        s2 = np.maximum(s2, 1e-8)
        n_l, f = len(self.levers), m.f
        tau = float(p["temp"]) * max(0.0, 1.0 - (self.t - self.t_reset) /
                                     max(1.0, float(p["explore_until"]) * (self.h - self.t_reset)))
        ws = []
        for k, c in ((0, self.cr), (1, self.cc)):
            lam = np.concatenate([np.tile(m.lam[k], n_l), [1.0 / BIAS_SD ** 2]])
            mu = np.concatenate([np.tile(m.mu[k], n_l), [0.0]])
            A = self.G / s2[k]
            A[np.diag_indices_from(A)] += lam
            L = np.linalg.cholesky(A)
            mean = np.linalg.solve(L.T, np.linalg.solve(L, lam * mu + c / s2[k]))
            draw = mean + tau * np.linalg.solve(L.T, self.rng.standard_normal(self.dim)) if tau > 0 else mean
            ws.append((mean, draw))
        self.mean_net = ws[0][0] - ws[1][0]
        wnet = (ws[0][1] - ws[1][1])[:-1].reshape(n_l, f)
        g = int(p["grid"])
        cand = np.clip(x[:, None] + np.linspace(-float(p["trust"]), float(p["trust"]), g)[None, :], -1.0, 1.0)
        d = cand - self.x0[:, None]
        val = np.einsum("lgf,lf->lg", m.feats(d, d, np.broadcast_to(self.x0[:, None], d.shape)), wnet)
        self.target = cand[np.arange(n_l), np.argmax(val, axis=1)]
        o._usage.search_evals += int(n_l * g)


class ModelBasedLearner(Learner):
    name, kind = "model_based", "model_based"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", **kw):
        super().__init__(spec, seed, "cpu")
        bad = sorted(set(kw) - set(MB_DEFAULTS))
        if bad:
            raise TypeError(f"model_based has no option(s) {bad}")
        self.p = dict(MB_DEFAULTS, **kw)
        torch = _torch()
        self.net = _build_net(self.p, seed)
        self.opt = cpu_adam(self.net.parameters(), lr=float(self.p["lr"]))
        self._np = _NP(self.net)
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self._slots: list[_MBSlot] | None = None
        self._explore_rng = np.random.default_rng([self.seed, 15485863])
        self._train_rng = np.random.default_rng([self.seed, 32452843])
        self._tg = torch.Generator().manual_seed(self.seed + 1)
        self._buffer: list[dict] = []          # finished training episodes (long-term experience)
        self._open: list[dict | None] = []     # the episode each slot is in the middle of
        self._owed = 0.0
        self._live: np.ndarray | None = None
        self.last_loss: float | None = None

    # ---- fast adaptation ---------------------------------------------------------------------------
    def _new(self, i: int) -> _MBSlot:
        return _MBSlot(self, np.random.default_rng([self.seed, 104729, i]))

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._slots is None or len(self._slots) != n or which is None:
            self._slots = [self._new(i) for i in range(n)]
            self._open = [None] * n
            self._live = np.ones(n, dtype=bool)
        else:
            w = np.asarray(which)
            for i in (np.flatnonzero(w) if w.dtype == bool else w):
                self._slots[int(i)] = self._new(int(i))
                self._open[int(i)] = None
                self._live[int(i)] = True

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        c0 = time.process_time()
        n = obs.shape[0]
        if self._slots is None or len(self._slots) != n:
            self.reset_context(n)
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        for i, s in enumerate(self._slots):
            if explore and not self.frozen and s.t == 0:
                s.rng = self._explore_rng               # training: a stream that never repeats
            out[i] = s.step(obs[i])
        self._usage.infer_cpu_s += time.process_time() - c0
        return out

    # ---- parametric learning -----------------------------------------------------------------------
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen:
            return
        c0 = time.process_time()
        sp = self.spec
        n = obs.shape[0]
        if self._live is None or len(self._live) != n:
            self._open, self._live = [None] * n, np.ones(n, dtype=bool)
        for i in range(n):
            if not self._live[i]:
                continue
            ep = self._open[i]
            if ep is None:
                ep = self._open[i] = {"x0": np.asarray(obs[i, sp.last_action], dtype=np.float32).copy(),
                                      "mask": np.asarray(obs[i, sp.lever_mask] > 0.5), "x": [], "rev": [],
                                      "cost": []}
            if len(ep["x"]) < int(self.p["max_ticks"]):
                ep["x"].append(np.asarray(next_obs[i, sp.last_action], dtype=np.float32).copy())
                ep["rev"].append(float(next_obs[i, sp.scalars][1]))
                ep["cost"].append(float(next_obs[i, sp.scalars][2]))
            self._usage.extra["observed_steps"] = self._usage.extra.get("observed_steps", 0) + 1
            if done[i]:
                self._live[i] = False
                self._open[i] = None
                if len(ep["x"]) >= 16:
                    self._buffer.append({"x0": ep["x0"], "mask": ep["mask"], "x": np.stack(ep["x"]),
                                         "rev": np.asarray(ep["rev"], dtype=np.float32),
                                         "cost": np.asarray(ep["cost"], dtype=np.float32)})
                    del self._buffer[:-int(self.p["buffer_episodes"])]
                    self._owed += float(self.p["updates_per_episode"])
        self._usage.train_cpu_s += time.process_time() - c0

    def update(self) -> dict:
        if self.frozen or self._owed < 1.0 or len(self._buffer) < int(self.p["min_episodes"]):
            return {"updates": 0}
        c0 = time.process_time()
        k = 0
        while self._owed >= 1.0:
            self._owed -= 1.0
            self.last_loss = self._step()
            k += 1
        self._np = _NP(self.net)
        self._usage.updates += k
        self._usage.train_cpu_s += time.process_time() - c0
        return {"updates": k, "loss": self.last_loss}

    def _batch(self):
        torch = _torch()
        p = self.p
        idx = self._train_rng.integers(0, len(self._buffer), size=int(p["batch_episodes"]))
        eps = [self._buffer[int(i)] for i in idx]
        b, t_max = len(eps), max(len(e["x"]) for e in eps)
        # only as many lever slots as the widest world of the batch has (the regression is cubic in their number)
        n_l = max(1, max(int(np.flatnonzero(e["mask"]).max()) + 1 if e["mask"].any() else 1 for e in eps))
        x = np.zeros((b, t_max, n_l))
        y = np.zeros((2, b, t_max))
        tm = np.zeros((b, t_max), dtype=bool)
        x0 = np.zeros((b, n_l))
        lm = np.zeros((b, n_l), dtype=bool)
        for i, e in enumerate(eps):
            n = len(e["x"])
            x[i, :n], y[0, i, :n], y[1, i, :n], tm[i, :n] = e["x"][:, :n_l], e["rev"], e["cost"], True
            x[i, n:] = e["x"][-1, :n_l]
            x0[i], lm[i] = e["x0"][:n_l], e["mask"][:n_l]
        ema = np.zeros_like(x)
        cur, a = x0.copy(), float(p["ema"])
        for t in range(t_max):
            cur = (1.0 - a) * cur + a * x[:, t]
            ema[:, t] = cur
        held = np.zeros((b, t_max), dtype=bool)
        held[:, 1:] = np.all(np.abs(x[:, 1:] - x[:, :-1]) < 1e-6, axis=2) & tm[:, 1:]
        dy2 = np.zeros((2, b, t_max))
        dy2[:, :, 1:] = 0.5 * (y[:, :, 1:] - y[:, :, :-1]) ** 2 * held[None, :, 1:]
        lens = tm.sum(1)
        ctx = np.stack([self._train_rng.integers(2, np.maximum(3, lens - 4)) for _ in range(int(p["contexts"]))])
        as_t = lambda v: torch.as_tensor(v, dtype=torch.float64)          # noqa: E731
        return (as_t(x - x0[:, None, :]), as_t(ema - x0[:, None, :]), as_t(x0), torch.as_tensor(lm),
                as_t(y), torch.as_tensor(tm), as_t(np.cumsum(dy2, axis=2)), as_t(np.cumsum(held, axis=1)),
                torch.as_tensor(ctx))

    def _loss(self, batch):
        torch = _torch()
        d, e, x0, lm, y, tm, css, cheld, ctx = batch
        net, q = self.net, int(self.p["query"])
        b, t_max, n_l = d.shape
        f = net.feats(d, e, x0[:, None, :].expand(-1, t_max, -1)) * lm[:, None, :, None]
        phi = torch.cat([f.reshape(b, t_max, -1), torch.ones(b, t_max, 1, dtype=f.dtype)], dim=2)
        dim = phi.shape[2]
        lam_all = torch.exp(net.log_lam.clamp(-12.0, 16.0))
        s2_0, n0 = torch.exp(net.log_s2.clamp(-18.0, 2.0)), torch.exp(net.log_n0.clamp(-3.0, 6.0))
        ar = torch.arange(t_max)
        total, count = phi.new_zeros(()), 0
        for c in ctx:                                           # one context length per episode
            cm = ((ar[None, :] < c[:, None]) & tm).to(phi.dtype)
            qi = (c[:, None] + torch.arange(q)[None, :]).clamp(max=t_max - 1)
            qm = torch.gather(tm, 1, qi) & ((c[:, None] + torch.arange(q)[None, :]) < t_max)
            pc = phi * cm[:, :, None]
            G = pc.transpose(1, 2) @ pc
            pq = torch.gather(phi, 1, qi[:, :, None].expand(-1, -1, dim))
            last = (c - 1).clamp(min=0)[:, None]
            for k in (0, 1):
                s2 = (n0[k] * s2_0[k] + torch.gather(css[k], 1, last)[:, 0]) / (n0[k] + torch.gather(cheld, 1, last)[:, 0])
                s2 = s2.clamp(min=1e-8)
                lam = torch.cat([lam_all[k].repeat(n_l), lam_all.new_full((1,), 1.0 / BIAS_SD ** 2)])
                mu = torch.cat([net.mu[k].repeat(n_l), net.mu.new_zeros(1)])
                A = G / s2[:, None, None] + torch.diag(lam)[None]
                rhs = (lam * mu)[None, :] + (pc * y[k][:, :, None]).sum(1) / s2[:, None]
                L = torch.linalg.cholesky(A)
                mean = torch.cholesky_solve(rhs[:, :, None], L)[:, :, 0]
                z = torch.linalg.solve_triangular(L, pq.transpose(1, 2), upper=False)
                var = s2[:, None] + (z * z).sum(1)
                err = torch.gather(y[k], 1, qi) - (pq * mean[:, None, :]).sum(2)
                nll = 0.5 * (torch.log(var) + err * err / var)
                total = total + (nll * qm).sum()
                count += int(qm.sum())
        return total / max(1, count)

    def _step(self) -> float:
        torch = _torch()
        loss = self._loss(self._batch())
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach())

    def heldout_nll(self, episodes: list[dict] | None = None, batches: int = 4) -> float:
        """Mean predictive negative log-likelihood on stored episodes (a diagnostic; changes nothing)."""
        torch = _torch()
        keep, state = self._buffer, self._train_rng.bit_generator.state
        if episodes is not None:
            self._buffer = episodes
        try:
            self._train_rng = np.random.default_rng(12345)
            with torch.no_grad():
                return float(np.mean([float(self._loss(self._batch())) for _ in range(batches)]))
        finally:
            self._buffer = keep
            self._train_rng = np.random.default_rng()
            self._train_rng.bit_generator.state = state

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch = _torch()
        torch.save({"learner": self.name, "params": self.p, "seed": self.seed, "net": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "updates": int(self._usage.updates)}, path)

    def load(self, path: str) -> None:
        """Restore the prior network and its optimiser. The network shape follows the checkpoint; every other
        option (how to act, how to train) stays as this learner was constructed."""
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        shape = ("feat_dim", "feat_hidden")
        if any(d["params"][k] != self.p[k] for k in shape):
            self.p.update({k: d["params"][k] for k in shape})
            self.net = _build_net(self.p, self.seed)
            self.opt = cpu_adam(self.net.parameters(), lr=float(self.p["lr"]))
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self._np = _NP(self.net)
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self._usage.updates = int(d.get("updates", 0))
        self._slots = None
