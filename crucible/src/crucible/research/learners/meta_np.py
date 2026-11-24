"""`meta_np`: a neural process meta-trained across worlds, acting by Bayesian optimisation inside each new one.

The problem a learner faces in a hidden-mechanism task is an experimental-design problem: settings go in, money comes
out, nothing says what any lever does, and every experiment costs money. Across many worlds there is structure worth
learning (responses saturate, some levers only cost, effects are gated by other levers, a good region is usually
interior) even though no lever means the same thing in two worlds. This learner keeps the two levels apart exactly
as the learner contract separates them.

(i) Inside an episode (fast adaptation, no weights change). The episode is a sequence of *arms*: settings held for
    `dwell` ticks, measured on the later ticks of the hold. Given the arms so far, the network predicts revenue and
    cost, with uncertainty, for any candidate settings. The learner picks the next arm by an upper confidence bound
    that shrinks as the episode runs out, then commits: from `explore_until` of the horizon it holds the best settings
    it has actually measured, moving to an untried one only if a pessimistic prediction beats that measurement. A
    sustained drop in what the held settings earn (a regime change) discards the old arms and restarts the search.

(ii) Across episodes (parametric learning: `observe` + `update`). The arms of finished training episodes are stored,
    and the network is trained by maximum likelihood to predict held-out arms of an episode from the others. This is
    a transformer neural process over a table with one row per arm and one column per lever. Attention runs along rows
    and along columns and there is no positional encoding, so the network is equivariant to the order of levers and
    of arms, and nothing it learns can be tied to a lever slot or a name. Its output is a *correction* to an exact
    Bayesian regression on each lever's move and squared move, fitted to the episode's arms. The correction starts at
    zero and the regression's own settings are fixed, so an untrained learner is exactly that regression and the
    difference between trained and untrained is what the correction learned. Decisions use the corrected mean with
    the regression's spread, because the learned spread did not clearly improve decisions on development tasks.

Measured outcome: this learner FAILED its positive-control criteria. See `docs/experiments/ow_meta_np_result.json`.

Inputs are the public surface only: the settings in force, the operator's own revenue and cost per tick, the clock
and the balance. Gauges are not used. An untrained learner runs the same procedure with the correction at zero, which
is what makes the trained-minus-untrained comparison a measure of what the weights acquired.
"""
from __future__ import annotations

import time

import numpy as np

from ..learner import EnvSpec, Learner

META_DEFAULTS = {
    # acting
    "dwell": 2, "max_arms": 64, "n_global": 48, "n_local": 128, "beta": 1.0,
    "explore_until": 0.45, "kappa": 0.75, "min_funds": 0.25, "drop": 0.5, "drop_patience": 2,
    "r0": 0.15, "r_min": 0.04, "r_max": 0.8, "r_grow": 1.4, "r_shrink": 0.7, "r_patience": 2,
    "loss_floor": 0.01, "loss_share": 0.5, "corr_mean": 1.0, "corr_sd": 0.0,
    # network
    "d": 64, "heads": 4, "enc_blocks": 2, "dec_blocks": 2, "y_scale": 0.01,
    "ridge_noise": 0.15, "ridge_tau": 1.0, "ridge_tau0": 5.0,
    # training
    "lr": 3e-4, "batch": 32, "query": 16, "buffer": 12000, "min_episodes": 64, "updates_per_episode": 16.0,
    "random_frac": 0.35, "train_beta": 1.5, "grad_clip": 1.0, "gpu_fraction": 0.05,
}
TOL = 1e-3
COORD_STEPS = np.array([-1.0, -0.6, -0.3, 0.3, 0.6, 1.0])


def _torch():
    import torch
    return torch


def to_z(y, s: float):
    """Symmetric log: worlds differ by orders of magnitude in what a tick earns."""
    return np.sign(y) * np.log1p(np.abs(y) / s)


def from_z(z, s: float):
    return np.sign(z) * s * np.expm1(np.minimum(np.abs(z), 12.0))


def _build_net(p: dict, seed: int):
    torch = _torch()
    nn = torch.nn
    d, heads = int(p["d"]), int(p["heads"])

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.na, self.nb = nn.LayerNorm(d), nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, q, k, mask):
            q = q + self.att(self.na(q), self.na(k), self.na(k), key_padding_mask=mask, need_weights=False)[0]
            return q + self.ff(self.nb(q))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_c = nn.Sequential(nn.Linear(5 + 3, d), nn.GELU(), nn.Linear(d, d))
            self.cell_q = nn.Sequential(nn.Linear(5 + 4, d), nn.GELU(), nn.Linear(d, d))
            self.enc_col = nn.ModuleList(Attn() for _ in range(int(p["enc_blocks"])))
            self.enc_row = nn.ModuleList(Attn() for _ in range(int(p["enc_blocks"])))
            self.dec_col = nn.ModuleList(Attn() for _ in range(int(p["dec_blocks"])))
            self.dec_row = nn.ModuleList(Attn() for _ in range(int(p["dec_blocks"])))
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 4))
            nn.init.zeros_(self.head[-1].weight)         # untrained, the prediction is exactly the regression below
            nn.init.zeros_(self.head[-1].bias)
            self.log_noise = nn.Parameter(torch.tensor(float(np.log(p["ridge_noise"]))), requires_grad=False)
            self.log_tau = nn.Parameter(torch.tensor([float(np.log(p["ridge_tau"]))] * 2), requires_grad=False)
            self.tau0 = float(p["ridge_tau0"])

        @staticmethod
        def _feat(x, x0, prev):
            return torch.stack([x, x - x0, x0, (x - x0).abs(), x - prev], dim=-1)   # `prev`: settings just before

        def encode(self, xc, yc, x0, lm, cm, pc):
            b, c, l = xc.shape
            x0e = x0[:, None, :].expand(-1, c, -1)
            h = self.cell_c(torch.cat([self._feat(xc, x0e, pc), yc[:, :, None, :].expand(-1, -1, l, -1)], dim=-1))
            for col, row in zip(self.enc_col, self.enc_row):
                u = h.permute(0, 2, 1, 3).reshape(b * l, c, -1)
                u = col(u, u, (~cm).repeat_interleave(l, dim=0))
                h = u.reshape(b, l, c, -1).permute(0, 2, 1, 3)
                v = h.reshape(b * c, l, -1)
                h = row(v, v, (~lm).repeat_interleave(c, dim=0)).reshape(b, c, l, -1)
            return h

        def ridge(self, xc, yc, xq, x0, lm, cm):
            """Bayesian regression of (revenue, cost) on each lever's move and squared move, from this episode's arms."""
            b, l = xc.shape[0], xc.shape[2]
            w = lm.double()
            def phi(x):
                dx = (x.double() - x0.double()[:, None, :]) * w[:, None, :]
                return torch.cat([torch.ones_like(dx[..., :1]), dx, dx ** 2], dim=-1)
            pc, pq = phi(xc) * cm.double()[..., None], phi(xq)
            noise = (2.0 * self.log_noise.double()).exp()
            tau = (2.0 * self.log_tau.double()).exp()
            prior = torch.cat([torch.full((1,), self.tau0 ** 2, dtype=torch.float64, device=xc.device),
                               tau[0].repeat(l), tau[1].repeat(l)])
            A = pc.transpose(1, 2) @ pc / noise + torch.diag(1.0 / prior)[None].expand(b, -1, -1)
            L = torch.linalg.cholesky(A)
            y = yc[..., :2].double() * cm.double()[..., None]
            coef = torch.cholesky_solve(pc.transpose(1, 2) @ y / noise, L)
            v = torch.linalg.solve_triangular(L, pq.transpose(1, 2), upper=False)
            var = noise + (v ** 2).sum(1)
            return (pq @ coef).float(), 0.5 * var.log().float()[..., None].expand(-1, -1, 2)

        def decode(self, h, xq, x0, lm, cm, pm, pls, pq):
            b, c, l = h.shape[:3]
            q = xq.shape[1]
            x0e = x0[:, None, :].expand(-1, q, -1)
            prior = torch.cat([pm / 5.0, pls], dim=-1)[:, :, None, :].expand(-1, -1, l, -1)
            z = self.cell_q(torch.cat([self._feat(xq, x0e, pq), prior], dim=-1))
            ctx = h.permute(0, 2, 1, 3).reshape(b * l, c, -1)
            cmask = (~cm).repeat_interleave(l, dim=0)
            lmask = (~lm).repeat_interleave(q, dim=0)
            for col, row in zip(self.dec_col, self.dec_row):
                u = z.permute(0, 2, 1, 3).reshape(b * l, q, -1)
                z = col(u, ctx, cmask).reshape(b, l, q, -1).permute(0, 2, 1, 3)
                v = z.reshape(b * q, l, -1)
                z = row(v, v, lmask).reshape(b, q, l, -1)
            w = lm[:, None, :, None].to(z.dtype)
            out = self.head((z * w).sum(2) / w.sum(2).clamp(min=1.0))
            return out[..., [0, 2]], out[..., [1, 3]]

        def forward(self, xc, yc, xq, x0, lm, cm, pc, pq, a=1.0, b=1.0):
            """`a` and `b` scale the learned correction to the mean and to the log spread (1 = as trained)."""
            pm, pls = self.ridge(xc, yc, xq, x0, lm, cm)
            dm, dls = self.decode(self.encode(xc, yc, x0, lm, cm, pc), xq, x0, lm, cm, pm.detach(), pls.detach(), pq)
            if self.training:                            # the spread is learned so the mean is fitted by likelihood
                return pm + dm, (pls + dls).clamp(-5.0, 2.0)
            return pm + a * dm, (pls + b * dls).clamp(-5.0, 2.0)

    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))                 # seeded initialisation, independent of global state afterwards
    net = Net()
    torch.random.set_rng_state(state)
    return net


class _Slot:
    """One environment's episode: the arms measured so far and what to do next."""

    def __init__(self, owner: "MetaNPLearner", rng: np.random.Generator):
        self.o, self.rng = owner, rng
        self.t, self.h, self.safe, self.random_mode = 0, 120, False, False
        self.X, self.R, self.C, self.N = [], [], [], []
        self.buf_r, self.buf_c = [], []
        self.since, self.t_reset, self.drops = 0, 0, 0
        self.ref_net: float | None = None
        self.radius, self.fails, self.spent = float(owner.p["r0"]), 0, 0.0
        self.base_net: float | None = None       # what the first settings measured earned
        self.pending: tuple | None = None        # (predicted net, predicted spread) of the arm being tried
        self.cal_sum, self.cal_n = 0.0, 0        # how wrong this episode's predictions have been, in spreads
        self.levers = self.x0 = self.target = None
        self.over = False

    def ingest(self, obs: np.ndarray) -> bool:
        """Read one observation. True if a new decision is due."""
        o, sp = self.o, self.o.spec
        if self.over:
            return False
        sc = obs[sp.scalars]
        rev, cost, funds, clock = float(sc[1]), float(sc[2]), float(sc[3]), float(sc[4])
        if self.t == 0:
            self.levers = np.flatnonzero(obs[sp.lever_mask] > 0.5)
            self.x0 = np.asarray(obs[sp.last_action], dtype=np.float64)[self.levers].copy()
            self.target, self.clock0 = self.x0.copy(), clock
            self.t = 1
            return False
        if self.t == 1 and clock > self.clock0:
            self.h = max(2, int(round((1.0 - self.clock0) / (clock - self.clock0))))
        self.t += 1
        self.buf_r.append(rev)
        self.buf_c.append(cost)
        self.since += 1
        if clock >= 1.0 - 1e-9:
            self.close_arm(obs)
            self.over = True
            return False
        if not self.safe and funds < float(o.p["min_funds"]):
            self.safe, self.target = True, self.x0.copy()
        if self.since >= int(o.p["dwell"]):
            self.close_arm(obs)
            return not self.safe
        return False

    def close_arm(self, obs: np.ndarray) -> None:
        if not self.buf_r:
            return
        sp = self.o.spec
        k = max(1, len(self.buf_r) // 2)
        r, c = float(np.mean(self.buf_r[-k:])), float(np.mean(self.buf_c[-k:]))
        x = np.asarray(obs[sp.last_action], dtype=np.float64)[self.levers]
        self.buf_r, self.buf_c, self.since = [], [], 0
        if self.X and np.max(np.abs(self.X[-1] - x)) < TOL:          # the same settings, measured again
            old = self.R[-1] - self.C[-1]
            if self.N[-1] >= 2 and self.ref_net is not None and self._dropped(old, r - c):
                self.drops += 1
                if self.drops >= int(self.o.p["drop_patience"]):     # what these settings earn has changed
                    self.X, self.R, self.C, self.N = [], [], [], []
                    self.t_reset, self.drops, self.ref_net = self.t, 0, None
                    self.radius, self.fails, self.spent, self.base_net = float(self.o.p["r0"]), 0, 0.0, None
                    self.pending, self.cal_sum, self.cal_n = None, 0.0, 0
            else:
                self.drops = 0
            if self.X:
                n = min(self.N[-1], 3)                               # recent measurements count most
                self.R[-1] = (n * self.R[-1] + r) / (n + 1)
                self.C[-1] = (n * self.C[-1] + c) / (n + 1)
                self.N[-1] += 1
                self.ref_net = self.R[-1] - self.C[-1]
                return
        else:
            self.drops = 0
        self._score_arm(r - c)
        self.X.append(x.copy())
        self.R.append(r)
        self.C.append(c)
        self.N.append(1)
        self.ref_net = r - c
        lim = int(self.o.p["max_arms"])
        if len(self.X) > lim:                                        # keep the best and the most recent arms
            net = np.asarray(self.R) - np.asarray(self.C)
            keep = sorted(set(np.argsort(net)[-lim // 2:].tolist()) | set(range(len(net) - lim // 2, len(net))))
            self.X, self.R, self.C, self.N = [self.X[i] for i in keep], [self.R[i] for i in keep], \
                [self.C[i] for i in keep], [self.N[i] for i in keep]

    def _score_arm(self, net: float) -> None:
        """Trust region and exploration account, updated when a new arm has been measured."""
        p = self.o.p
        if self.pending is not None:
            pred, spread = self.pending
            self.cal_sum += min((net - pred) ** 2 / max(spread ** 2, 1e-8), 100.0)
            self.cal_n += 1
            self.pending = None
        if not self.X:
            self.base_net = net
            return
        best = float(np.max(self.net()))
        self.spent += max(0.0, best - net) * int(p["dwell"])
        if net > best + 1e-6:
            self.radius, self.fails = min(float(p["r_max"]), self.radius * float(p["r_grow"])), 0
        else:
            self.fails += 1
            if self.fails >= int(p["r_patience"]):
                self.radius, self.fails = max(float(p["r_min"]), self.radius * float(p["r_shrink"])), 0

    def calibration(self) -> float:
        """Factor that turns the model's spread into this episode's observed error (1 before any evidence)."""
        return float(np.sqrt(np.clip((1.0 + self.cal_sum) / (1.0 + self.cal_n), 1.0 / 16.0, 16.0)))

    def may_explore(self) -> bool:
        """Experiments may lose a small fixed amount, plus a share of the gain already demonstrated and still to collect."""
        p = self.o.p
        if self.phase() >= 1.0:
            return False
        gain = max(0.0, float(np.max(self.net())) - (self.base_net if self.base_net is not None else 0.0))
        return self.spent <= float(p["loss_floor"]) * self.h + float(p["loss_share"]) * gain * max(0, self.h - self.t)

    def _dropped(self, old: float, new: float) -> bool:
        return new < old - max(float(self.o.p["drop"]) * abs(old), 0.01)

    def net(self) -> np.ndarray:
        return np.asarray(self.R) - np.asarray(self.C)

    def phase(self) -> float:
        """0 at the start of the current search, 1 at the point where it must commit."""
        span = max(1.0, float(self.o.p["explore_until"]) * (self.h - self.t_reset))
        return (self.t - self.t_reset) / span

    def candidates(self) -> tuple[np.ndarray, int]:
        """Settings to consider: the first `k` lie in the trust region around the best arm, the rest anywhere."""
        p, rng, n_l = self.o.p, self.rng, len(self.levers)
        best, r = self.X[int(np.argmax(self.net()))], self.radius
        n = int(p["n_local"])
        mask = rng.random((n, n_l)) < rng.uniform(0.15, 1.0, (n, 1))
        local = best + rng.normal(0.0, r, (n, n_l)) * mask
        coord = np.repeat(best[None, :], 4 * n_l, axis=0)
        for j in range(n_l):
            coord[4 * j:4 * j + 4, j] += np.array([-r, -r / 2, r / 2, r])
        near = np.clip(np.concatenate([local, coord]), -1.0, 1.0)
        far = rng.uniform(-1.0, 1.0, (int(p["n_global"]), n_l))
        return np.concatenate([near, far]), len(near)

    def coord_grid(self, x: np.ndarray) -> np.ndarray:
        """Each lever moved alone around `x`: shape (levers * len(COORD_STEPS), levers)."""
        n_l, g = len(self.levers), len(COORD_STEPS)
        out = np.repeat(x[None, :], n_l * g, axis=0)
        for j in range(n_l):
            out[j * g:(j + 1) * g, j] += COORD_STEPS * self.radius
        return np.clip(out, -1.0, 1.0)

    def random_arm(self) -> np.ndarray:
        rng, n_l, u = self.rng, len(self.levers), self.rng.random()
        if u < 0.5:
            return rng.uniform(-1.0, 1.0, n_l)
        if u < 0.75 or not self.X:
            return np.clip(self.x0 + rng.normal(0.0, 0.3, n_l), -1.0, 1.0)
        return np.clip(self.X[int(np.argmax(self.net()))] + rng.normal(0.0, 0.2, n_l), -1.0, 1.0)


class MetaNPLearner(Learner):
    name, kind = "meta_np", "model_based"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", **kw):
        super().__init__(spec, seed, device)
        bad = sorted(set(kw) - set(META_DEFAULTS))
        if bad:
            raise TypeError(f"meta_np has no option(s) {bad}")
        self.p = dict(META_DEFAULTS, **kw)
        torch = _torch()
        if str(device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(float(self.p["gpu_fraction"]), torch.device(device).index or 0)
        self.net = _build_net(self.p, seed).to(device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.p["lr"]), weight_decay=1e-5)
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self._slots: list[_Slot] | None = None
        self._explore_rng = np.random.default_rng([self.seed, 15485863])
        self._train_rng = np.random.default_rng([self.seed, 32452843])
        self._buffer: list[dict] = []
        self._owed = 0.0
        self.last_stats: dict | None = None

    # ---- fast adaptation ---------------------------------------------------------------------------
    def _new(self, i: int) -> _Slot:
        return _Slot(self, np.random.default_rng([self.seed, 104729, i]))

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._slots is None or len(self._slots) != n or which is None:
            self._slots = [self._new(i) for i in range(n)]
        else:
            w = np.asarray(which)
            for i in (np.flatnonzero(w) if w.dtype == bool else w):
                self._slots[int(i)] = self._new(int(i))

    def _tick_time(self, t0: float, c0: float, field: str) -> None:
        u = self._usage
        if str(self.device).startswith("cuda"):
            setattr(u, f"{field}_gpu_s", getattr(u, f"{field}_gpu_s") + time.perf_counter() - t0)
        setattr(u, f"{field}_cpu_s", getattr(u, f"{field}_cpu_s") + time.process_time() - c0)

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        t0, c0 = time.perf_counter(), time.process_time()
        n = obs.shape[0]
        if self._slots is None or len(self._slots) != n:
            self.reset_context(n)
        training = bool(explore and not self.frozen)
        due: list[_Slot] = []
        for i, s in enumerate(self._slots):
            if s.t == 0 and training:
                s.rng = self._explore_rng               # training: a stream that never repeats
                s.random_mode = bool(s.rng.random() < float(self.p["random_frac"]))
            if s.ingest(obs[i]):
                if s.random_mode:
                    s.target = s.random_arm()
                else:
                    due.append(s)
        if due:
            self._decide(due, training)
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        for i, s in enumerate(self._slots):
            if s.levers is not None:
                out[i, s.levers] = s.target
        self._tick_time(t0, c0, "infer")
        return out

    def _pad(self, rows: list[np.ndarray], width: int, fill: float = 0.0) -> np.ndarray:
        out = np.full((len(rows), rows[0].shape[0], width), fill, dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
        return out

    def _tensors(self, slots: list[_Slot], cand: list[np.ndarray]):
        torch = _torch()
        b, s_y = len(slots), float(self.p["y_scale"])
        lmax = max(len(s.levers) for s in slots)
        cmax = max(len(s.X) for s in slots)
        qmax = max(len(c) for c in cand)
        xc = np.zeros((b, cmax, lmax), np.float32)
        yc = np.zeros((b, cmax, 3), np.float32)
        xq = np.zeros((b, qmax, lmax), np.float32)
        x0 = np.zeros((b, lmax), np.float32)
        lm = np.zeros((b, lmax), bool)
        cm = np.zeros((b, cmax), bool)
        pc, pq = np.zeros((b, cmax, lmax), np.float32), np.zeros((b, qmax, lmax), np.float32)
        for i, (s, c) in enumerate(zip(slots, cand)):
            n, l = len(s.X), len(s.levers)
            arms = np.stack(s.X)
            xc[i, :n, :l] = arms
            pc[i, :n, :l] = np.vstack([s.x0[None, :], arms[:-1]])
            pq[i, : len(c), :l] = arms[-1]                   # a candidate would follow the settings now in force
            r, co = np.asarray(s.R), np.asarray(s.C)
            yc[i, :n] = np.stack([to_z(r, s_y), to_z(co, s_y), to_z(r - co, s_y)], axis=1)
            xq[i, : len(c), :l] = c
            x0[i, :l], lm[i, :l], cm[i, :n] = s.x0, True, True
        dev = self.device
        return [torch.as_tensor(v, device=dev) for v in (xc, yc, xq, x0, lm, cm, pc, pq)]

    def _predict(self, slots: list[_Slot], cand: list[np.ndarray]):
        """Predicted net income per tick of each candidate *relative to the best arm so far*, and its spread.
        Errors shared by all settings of a world cancel in the difference, and the difference is what a decision needs."""
        torch = _torch()
        s_y = float(self.p["y_scale"])
        with_best = [np.concatenate([c, s.X[int(np.argmax(s.net()))][None, :]]) for s, c in zip(slots, cand)]
        xc, yc, xq, x0, lm, cm, pc, pq = self._tensors(slots, with_best)
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            mu, ls = self.net(xc, yc, xq, x0, lm, cm, pc, pq, float(self.p["corr_mean"]), float(self.p["corr_sd"]))
        self.net.train(was)
        mu, sd = mu.cpu().numpy(), np.exp(ls.cpu().numpy())
        out = []
        for i, c in enumerate(cand):
            q = len(c)
            zr, zc = mu[i, :q + 1, 0], mu[i, :q + 1, 1]
            m = from_z(zr, s_y) - from_z(zc, s_y)
            gr, gc = s_y * np.exp(np.minimum(np.abs(zr), 12.0)), s_y * np.exp(np.minimum(np.abs(zc), 12.0))
            v = np.sqrt((gr * sd[i, :q + 1, 0]) ** 2 + (gc * sd[i, :q + 1, 1]) ** 2)
            out.append((m[:q] - m[q], v[:q]))
            self._usage.search_evals += q
        return out

    def _value(self, s: _Slot, d, v, explore: bool, training: bool, k: int | None = None) -> np.ndarray:
        """What a candidate is worth over the best arm: optimistic while searching, pessimistic once committed."""
        p = self.p
        spread = s.calibration() * v
        lcb = d - float(p["kappa"]) * spread
        if not explore:
            return lcb
        beta = float(p["train_beta"] if training else p["beta"]) * max(0.0, 1.0 - s.phase())
        acq = d + beta * spread
        if k is not None:                                # far from the best arm only if confidently better
            acq[k:] = np.where(lcb[k:] > self._margin(s), acq[k:], -np.inf)
        return acq

    @staticmethod
    def _margin(s: _Slot) -> float:
        return 0.02 * abs(float(np.max(s.net())))

    def _decide(self, slots: list[_Slot], training: bool) -> None:
        ready = [s for s in slots if s.X]
        for s in slots:
            if not s.X:
                s.target = s.x0.copy()
        if not ready:
            return
        explore = [s.may_explore() for s in ready]

        made = [s.candidates() for s in ready]           # 1. a broad look
        first = self._predict(ready, [c for c, _ in made])
        tops, top_val, top_pred = [], [], []
        for s, (c, k), (d, v), ex in zip(ready, made, first, explore):
            val = self._value(s, d, v, ex, training, k)
            j = int(np.argmax(val))
            tops.append(c[j].copy())
            top_val.append(float(val[j]))
            top_pred.append((float(d[j]), float(v[j])))

        grids = [s.coord_grid(x) for s, x in zip(ready, tops)]       # 2. each lever on its own
        second = self._predict(ready, grids)
        joints, grid_vals = [], []
        g = len(COORD_STEPS)
        for s, x, a, grid, (d, v), ex in zip(ready, tops, top_val, grids, second, explore):
            val = self._value(s, d, v, ex, training)
            grid_vals.append(val)
            n_l = len(s.levers)
            per = val.reshape(n_l, g)
            bj = per.argmax(1)
            gain = per[np.arange(n_l), bj] - a
            order = np.argsort(-gain)
            moved = x.copy()
            rows = []
            for j in order:                              # 3. the best single moves, combined
                if gain[j] <= 0:
                    break
                moved[j] = grid[j * g + bj[j], j]
                rows.append(moved.copy())
            joints.append(np.stack(rows) if rows else x[None, :].copy())
        third = self._predict(ready, joints)

        for i, (s, x, a, grid, gv, jo, (d3, v3), ex) in enumerate(zip(ready, tops, top_val, grids, grid_vals, joints,
                                                                      third, explore)):
            jv = self._value(s, d3, v3, ex, training)
            every = np.concatenate([x[None, :], grid, jo])
            vals = np.concatenate([[a], gv, jv])
            ds = np.concatenate([[top_pred[i][0]], second[i][0], d3])
            vs = np.concatenate([[top_pred[i][1]], second[i][1], v3])
            j = int(np.argmax(vals))
            net = s.net()
            ib = int(np.argmax(net))
            if vals[j] > (0.0 if ex else self._margin(s)):
                s.target = every[j].copy()
                s.pending = (float(net[ib] + ds[j]), float(vs[j]))
            else:
                s.target, s.pending = s.X[ib].copy(), None

    # ---- parametric learning -----------------------------------------------------------------------
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen or self._slots is None:
            return
        c0 = time.process_time()
        live = sum(1 for s in self._slots if not getattr(s, "stored", False))
        self._usage.extra["observed_steps"] = self._usage.extra.get("observed_steps", 0) + live
        for i in np.flatnonzero(done):
            s = self._slots[int(i)]
            if s.levers is not None and not getattr(s, "stored", False):
                s.ingest(next_obs[int(i)])
                s.stored = True
                if len(s.X) >= 3:
                    self._buffer.append({"x": np.stack(s.X).astype(np.float32), "x0": s.x0.astype(np.float32),
                                         "r": np.asarray(s.R, np.float32), "c": np.asarray(s.C, np.float32)})
                    del self._buffer[:-int(self.p["buffer"])]
                    self._owed += float(self.p["updates_per_episode"])
        self._usage.train_cpu_s += time.process_time() - c0

    def update(self) -> dict:
        if self.frozen or self._owed < 1.0 or len(self._buffer) < int(self.p["min_episodes"]):
            return {"updates": 0}
        t0, c0 = time.perf_counter(), time.process_time()
        k, loss = 0, None
        while self._owed >= 1.0:
            self._owed -= 1.0
            loss = self._step()
            k += 1
        self._usage.updates += k
        self.last_stats = {"nll": round(loss, 4), "buffer": len(self._buffer)}
        self._tick_time(t0, c0, "train")
        return {"updates": k, "loss": loss}

    def _batch(self):
        torch = _torch()
        p, rng, s_y = self.p, self._train_rng, float(self.p["y_scale"])
        eps = [self._buffer[int(i)] for i in rng.integers(0, len(self._buffer), int(p["batch"]))]
        b, q = len(eps), int(p["query"])
        lmax = max(e["x"].shape[1] for e in eps)
        picks = []
        for e in eps:
            n = len(e["x"])
            c = int(rng.integers(1, n))                  # context size
            order = np.arange(n) if rng.random() < 0.5 else rng.permutation(n)
            picks.append((order[:c], order[c:][:q]))
        cmax = max(len(a) for a, _ in picks)
        qmax = max(len(t) for _, t in picks)
        xc, yc = np.zeros((b, cmax, lmax), np.float32), np.zeros((b, cmax, 3), np.float32)
        xq, yq = np.zeros((b, qmax, lmax), np.float32), np.zeros((b, qmax, 2), np.float32)
        x0, lm = np.zeros((b, lmax), np.float32), np.zeros((b, lmax), bool)
        cm, qm = np.zeros((b, cmax), bool), np.zeros((b, qmax), bool)
        pc, pq = np.zeros((b, cmax, lmax), np.float32), np.zeros((b, qmax, lmax), np.float32)
        for i, (e, (a, t)) in enumerate(zip(eps, picks)):
            l = e["x"].shape[1]
            before = np.vstack([e["x0"][None, :], e["x"][:-1]])
            pc[i, : len(a), :l], pq[i, : len(t), :l] = before[a], before[t]
            r, co = to_z(e["r"], s_y), to_z(e["c"], s_y)
            xc[i, : len(a), :l] = e["x"][a]
            yc[i, : len(a)] = np.stack([r[a], co[a], to_z(e["r"][a] - e["c"][a], s_y)], axis=1)
            xq[i, : len(t), :l] = e["x"][t]
            yq[i, : len(t)] = np.stack([r[t], co[t]], axis=1)
            x0[i, :l], lm[i, :l], cm[i, : len(a)], qm[i, : len(t)] = e["x0"], True, True, True
        return [torch.as_tensor(v, device=self.device) for v in (xc, yc, xq, yq, x0, lm, cm, qm, pc, pq)]

    def _step(self) -> float:
        torch = _torch()
        xc, yc, xq, yq, x0, lm, cm, qm, pc, pq = self._batch()
        mu, ls = self.net(xc, yc, xq, x0, lm, cm, pc, pq)
        nll = (0.5 * ((yq - mu) / ls.exp()) ** 2 + ls).sum(-1)
        loss = (nll * qm).sum() / qm.sum().clamp(min=1)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.p["grad_clip"]))
        self.opt.step()
        return float(loss.detach())

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch = _torch()
        cpu = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save({"learner": self.name, "params": self.p, "seed": self.seed, "net": cpu,
                    "updates": int(self._usage.updates)}, path)

    def load(self, path: str) -> None:
        """Restore the network. Its shape follows the checkpoint; how to act stays as this learner was built."""
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        shape = ("d", "heads", "enc_blocks", "dec_blocks")
        if any(d["params"][k] != self.p[k] for k in shape):
            self.p.update({k: d["params"][k] for k in shape})
            self.net = _build_net(self.p, self.seed).to(self.device)
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.p["lr"]), weight_decay=1e-5)
        self.net.load_state_dict(d["net"])
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self._usage.updates = int(d.get("updates", 0))
        self._slots = None
