"""Learners for the arena, all behind `research.learner.Learner` and all strictly independent.

Two adaptive learners that share no idea, so that no self-play result rests on one algorithm:

`SPSALearner` ("spsa")   simultaneous-perturbation stochastic approximation on the individual's own objective.
                         Its policy is a settings vector theta; in every match it plays theta + c*Delta in one
                         world and theta - c*Delta in another (or in alternating stretches of one match), and
                         moves theta along the measured difference. No network, no model, no gradient of anything
                         but its own measured income. With `segments > 1` it also updates inside a match.
`PGLearner` ("pg")       a small neural policy (Gaussian, observation-conditioned) with its own value network,
                         trained by clipped policy gradient (PPO) with generalised advantage estimation on its own
                         rollouts only. It decides every `hold` ticks and keeps a fading trace of its own
                         observations as recurrent state, so it can react inside a match to what its gauges show
                         (including, where the mechanism shows it, a rival's lagged price).

Independence is structural: a learner sees its own observation and its own reward, owns its parameters, optimiser,
normalisers, random generator and buffer, and has no handle on any other learner. `tests/test_selfplay.py` checks
that no array or tensor is shared between two individuals and that a buffer only ever holds its owner's
observations.

Scripted operators and fixtures:

`ClimberLearner` ("climber")  a competent scripted operator that uses the public interface only: coordinate
                              search on its own measured income, one lever at a time, keep what helped.
`FixedLearner` ("fixed")      plays one constant settings vector (used for operator-side profiles, see
                              `selfplay.profiles`; such individuals are marked `privileged`).
`hold`, `random`              `research.baselines`.

`make_learner(spec, algo, seed, **options)` builds any of them, and any learner of the `research.learners`
registry by its name (for example `ppo_gru`) when that package is importable.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from ..research.baselines import HoldLearner, RandomLearner
from ..research.learner import EnvSpec, Learner

OWN = ("spsa", "pg", "climber", "fixed", "hold", "random")


# ====================================================================== stochastic approximation
class SPSALearner(Learner):
    """Simultaneous-perturbation stochastic approximation (Spall 1992) on the individual's own reward.

    theta (one number per lever, in [-1, 1]) starts at the facility's public default settings, read from the
    first observation. With n parallel matches, slots 2j and 2j+1 form a pair that plays theta + c_k Delta_j and
    theta - c_k Delta_j (Delta_j a random sign per lever); an odd slot plays theta itself. With n = 1 the two
    signs are played in consecutive matches. The gradient estimate is the mean over pairs of
    (R+ - R-) / (2 c_k) Delta_j, R being the mean reward per tick, and theta moves by an Adam-normalised step of
    size a_k (so that the step does not depend on the money scale of a world). Gains decay as in Spall:
    a_k = a / (1 + k / A)^alpha, c_k = max(c_min, c / (1 + k / A)^gamma).

    `segments = S > 1` splits every match into S equal stretches by the public clock; the signs of a pair swap
    from stretch to stretch and theta is updated after every two stretches, so the learner also adapts inside a
    match. A perturbation whose effect arrives later than the stretch is then credited to the wrong sign: use it
    only where effects are fast.

    Frozen, or with `explore=False`, it plays theta.
    """
    name, kind = "spsa", "stochastic_approximation"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", a: float = 0.06, c: float = 0.10,
                 A: float = 30.0, alpha: float = 0.602, gamma: float = 0.101, c_min: float = 0.03,
                 a_min: float = 0.004, segments: int = 1, beta1: float = 0.7, beta2: float = 0.98):
        super().__init__(spec, seed, device)
        self.a, self.c, self.A, self.alpha, self.gamma = float(a), float(c), float(A), float(alpha), float(gamma)
        self.c_min, self.a_min, self.segments = float(c_min), float(a_min), max(1, int(segments))
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.rng = np.random.default_rng([int(seed), 0x5B5A])
        L = spec.act_dim
        self.theta: np.ndarray | None = None        # long-term state ("weights")
        self.mask = np.zeros(L, dtype=bool)
        self.k = 0                                  # updates done
        self.m, self.v = np.zeros(L), np.zeros(L)
        self.grad_log: list[float] = []             # |gradient estimate| per update (kept short)
        # episodic context
        self._n = 0
        self._delta: np.ndarray | None = None       # [pairs, L]
        self._sign = np.zeros(0)                    # [n] in {+1, -1, 0}
        self._rp = np.zeros(0)                      # reward collected under + and under -, per pair
        self._rm = np.zeros(0)
        self._np = np.zeros(0)
        self._nm = np.zeros(0)
        self._stretch = 0
        self._seq_plus: tuple | None = None         # n == 1: the + half waiting for its - half
        self._explored = False
        self._usage.params = L

    # ---- gains -------------------------------------------------------------------------------------
    def gains(self) -> tuple[float, float]:
        x = 1.0 + self.k / self.A
        return max(self.a_min, self.a / x ** self.alpha), max(self.c_min, self.c / x ** self.gamma)

    # ---- context -----------------------------------------------------------------------------------
    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        self._n = int(n)
        self._stretch = 0
        self._explored = False
        self._draw()

    def _draw(self) -> None:
        n, L = self._n, self.spec.act_dim
        pairs = max(1, n // 2)
        self._delta = self.rng.choice([-1.0, 1.0], size=(pairs, L))
        if n >= 2:
            self._sign = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
            if n % 2:
                self._sign[-1] = 0.0
        else:
            if self._seq_plus is None:
                self._sign = np.array([1.0])
            else:                                   # second half of a sequential pair: same direction, other sign
                self._delta = self._seq_plus[0].copy()
                self._sign = np.array([-1.0])
        self._rp, self._rm = np.zeros(pairs), np.zeros(pairs)
        self._np, self._nm = np.zeros(pairs), np.zeros(pairs)

    def _init_theta(self, obs: np.ndarray) -> None:
        sp = self.spec
        self.mask = obs[0, sp.lever_mask] > 0.5
        self.theta = np.where(self.mask, obs[:, sp.last_action].mean(axis=0), 0.0).astype(np.float64)
        self._usage.params = int(self.mask.sum())

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        n = obs.shape[0]
        if self.theta is None:
            self._init_theta(obs)
        if self._delta is None or self._n != n:
            self.reset_context(n)
        out = np.repeat(self.theta[None, :], n, axis=0)
        self._explored = bool(explore) and not self.frozen
        if self._explored:
            if self.segments > 1:                   # which stretch of the match this is, from the public clock
                stretch = min(self.segments - 1, int(float(obs[0, self.spec.scalars][4]) * self.segments + 1e-6))
                if stretch != self._stretch:
                    if stretch % 2 == 0:
                        self._step()                # a pair of stretches is complete
                        self._draw()
                        out = np.repeat(self.theta[None, :], n, axis=0)
                    self._stretch = stretch
            _, c = self.gains()
            flip = -1.0 if (self.segments > 1 and self._stretch % 2 == 1) else 1.0
            pair = np.minimum(np.arange(n) // 2, len(self._delta) - 1)
            out = out + (flip * self._sign)[:, None] * c * self._delta[pair]
        out = np.where(self.mask[None, :], np.clip(out, -1.0, 1.0), 0.0)
        self._usage.infer_cpu_s += time.perf_counter() - t0
        return out.astype(np.float32)

    # ---- learning ----------------------------------------------------------------------------------
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen or not self._explored:
            return
        flip = -1.0 if (self.segments > 1 and self._stretch % 2 == 1) else 1.0
        sign = flip * self._sign
        n = len(sign)
        pair = np.minimum(np.arange(n) // 2, len(self._rp) - 1)
        r = np.asarray(reward, dtype=np.float64)
        plus, minus = sign > 0, sign < 0
        np.add.at(self._rp, pair[plus], r[plus])
        np.add.at(self._np, pair[plus], 1.0)
        np.add.at(self._rm, pair[minus], r[minus])
        np.add.at(self._nm, pair[minus], 1.0)

    def _step(self) -> int:
        """One stochastic-approximation step from the reward collected since the last draw."""
        if self.frozen or self.theta is None:
            return 0
        if self._n == 1 and self.segments == 1:     # sequential pair over two matches
            if self._sign[0] > 0:
                if self._np[0] > 0:
                    self._seq_plus = (self._delta.copy(), self._rp[0] / self._np[0])
                return 0
            if self._seq_plus is None or self._nm[0] == 0:
                return 0
            diff = np.array([self._seq_plus[1] - self._rm[0] / self._nm[0]])
            delta = self._seq_plus[0]
            self._seq_plus = None
        else:
            ok = (self._np > 0) & (self._nm > 0)
            if not ok.any():
                return 0
            diff = (self._rp[ok] / self._np[ok]) - (self._rm[ok] / self._nm[ok])
            delta = self._delta[ok]
        t0 = time.perf_counter()
        a, c = self.gains()
        g = (diff[:, None] / (2.0 * c) * delta).mean(axis=0) * self.mask
        self.k += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        mh, vh = self.m / (1 - self.beta1 ** self.k), self.v / (1 - self.beta2 ** self.k)
        self.theta = np.where(self.mask, np.clip(self.theta + a * mh / (np.sqrt(vh) + 1e-12), -1.0, 1.0), 0.0)
        self.grad_log = (self.grad_log + [float(np.abs(g).sum())])[-200:]
        self._usage.updates += 1
        self._usage.train_cpu_s += time.perf_counter() - t0
        return 1

    def update(self) -> dict:
        done = self._step()
        self._rp[:] = self._rm[:] = self._np[:] = self._nm[:] = 0.0
        return {"updates": done, "k": self.k}

    # ---- the policy, for analysis --------------------------------------------------------------------
    def policy_vector(self) -> np.ndarray | None:
        return None if self.theta is None else self.theta.copy()

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "seed": self.seed, "k": self.k,
                       "theta": None if self.theta is None else self.theta.tolist(), "mask": self.mask.tolist(),
                       "m": self.m.tolist(), "v": self.v.tolist(), "rng": self.rng.bit_generator.state,
                       "updates": self._usage.updates}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.k = int(d["k"])
        self.theta = None if d["theta"] is None else np.asarray(d["theta"], dtype=np.float64)
        self.mask = np.asarray(d["mask"], dtype=bool)
        self.m, self.v = np.asarray(d["m"], dtype=np.float64), np.asarray(d["v"], dtype=np.float64)
        self.rng.bit_generator.state = d["rng"]
        self._usage.updates = int(d.get("updates", 0))
        self._seq_plus = None


# ====================================================================== neural policy gradient
class PGLearner(Learner):
    """A neural policy trained by clipped policy gradient on its own rollouts (PPO-clip, GAE; Schulman et al.).

    Policy: mean = anchor + MLP(z), a state-independent learned log-std per lever, actions clipped to [-1, 1].
    `anchor` is the facility's public default setting of each lever, read from the first observation of a match,
    and the last layer starts at zero: an untrained policy holds the defaults and explores around them. z is the
    individual's own observation (normalised by its own running statistics) joined with a fading trace of it
    (`trace`: the recurrent state, cleared at the start of a match). The critic is a separate MLP on the same
    input; nothing is shared with any other learner.

    It decides every `hold` ticks and repeats the action in between; the reward of a decision is what its own
    objective paid over those ticks. `update()` is called after a batch of complete matches and runs `epochs`
    passes of the clipped surrogate over that batch, then discards it (on-policy: the rollout buffer never holds
    anything but the current batch of this individual's experience).
    """
    name, kind = "pg", "model_free"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", hidden: int = 32, lr: float = 3e-3,
                 vf_lr: float = 1e-2, gamma: float = 0.95, lam: float = 0.9, clip: float = 0.2, epochs: int = 6,
                 hold: int = 6, sigma0: float = 0.18, sigma_min: float = 0.04, trace: float = 0.7,
                 max_kl: float = 0.03, ent: float = 0.0):
        super().__init__(spec, seed, "cpu")
        import torch
        self._torch = torch
        self.hp = {"hidden": int(hidden), "lr": float(lr), "vf_lr": float(vf_lr), "gamma": float(gamma),
                   "lam": float(lam), "clip": float(clip), "epochs": int(epochs), "hold": max(1, int(hold)),
                   "sigma0": float(sigma0), "sigma_min": float(sigma_min), "trace": float(trace),
                   "max_kl": float(max_kl), "ent": float(ent)}
        self.gen = torch.Generator().manual_seed(int(seed) * 7919 + 13)
        D, L, H = 2 * spec.obs_dim, spec.act_dim, int(hidden)

        def lin(i, o, scale):
            layer = torch.nn.Linear(i, o)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(o, i, generator=self.gen) * (scale / np.sqrt(i)))
                layer.bias.zero_()
            return layer

        self.pi = torch.nn.Sequential(lin(D, H, 1.0), torch.nn.Tanh(), lin(H, H, 1.0), torch.nn.Tanh(), lin(H, L, 0.0))
        self.vf = torch.nn.Sequential(lin(D, H, 1.0), torch.nn.Tanh(), lin(H, H, 1.0), torch.nn.Tanh(), lin(H, 1, 0.0))
        self.log_std = torch.nn.Parameter(torch.full((L,), float(np.log(sigma0))))
        self.opt_pi = torch.optim.Adam(list(self.pi.parameters()) + [self.log_std], lr=float(lr))
        self.opt_vf = torch.optim.Adam(self.vf.parameters(), lr=float(vf_lr))
        # own running statistics (long-term state)
        self.o_mean, self.o_var, self.o_count = np.zeros(spec.obs_dim), np.ones(spec.obs_dim), 1e-4
        self.r_var, self.r_count = 1.0, 1e-4
        self.iters = 0
        self._usage.params = int(sum(p.numel() for p in self.pi.parameters()) + L
                                 + sum(p.numel() for p in self.vf.parameters()))
        # episodic context (recurrent state) and the rollout buffer
        self._n = 0
        self._anchor: np.ndarray | None = None
        self._tr: np.ndarray | None = None
        self._age: np.ndarray | None = None
        self._cur: np.ndarray | None = None
        self._pending: dict | None = None
        self._explored = False
        self.buffer: list[dict] = []                 # one entry per decision: this individual's experience only
        self.last_stats: dict = {}

    # ---- context -----------------------------------------------------------------------------------
    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        self._n = int(n)
        self._anchor = None
        self._tr = np.zeros((n, self.spec.obs_dim))
        self._age = np.zeros(n, dtype=np.int64)
        self._cur = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        self._pending = None

    def _features(self, obs: np.ndarray, learn: bool) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float64)
        if learn:                                    # own running mean and variance of own observations
            b = o.shape[0]
            tot = self.o_count + b
            delta = o.mean(axis=0) - self.o_mean
            self.o_mean = self.o_mean + delta * b / tot
            self.o_var = (self.o_var * self.o_count + o.var(axis=0) * b + delta ** 2 * self.o_count * b / tot) / tot
            self.o_count = tot
        z = np.clip((o - self.o_mean) / np.sqrt(self.o_var + 1e-6), -8.0, 8.0)
        tr = self.hp["trace"]
        self._tr = tr * self._tr + (1.0 - tr) * z
        return np.concatenate([z, self._tr], axis=1).astype(np.float32)

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        torch, sp = self._torch, self.spec
        n = obs.shape[0]
        if self._tr is None or self._n != n:
            self.reset_context(n)
        if self._anchor is None:
            self._anchor = np.asarray(obs[:, sp.last_action], dtype=np.float32).copy()
        learn = bool(explore) and not self.frozen
        self._explored = learn
        z = self._features(obs, learn)
        decide = self._age % self.hp["hold"] == 0
        if decide.all() or decide.any():
            mask = np.asarray(obs[:, sp.lever_mask], dtype=np.float32)
            with torch.no_grad():
                zt = torch.from_numpy(z)
                mean = torch.from_numpy(self._anchor) + self.pi(zt)
                std = self.log_std.exp().clamp_min(self.hp["sigma_min"])
                if explore:
                    raw = mean + std * torch.randn(mean.shape, generator=self.gen)
                else:
                    raw = mean
                logp = (-(0.5 * ((raw - mean) / std) ** 2 + std.log() + 0.9189385) * torch.from_numpy(mask)).sum(1)
                val = self.vf(zt).squeeze(1)
            raw_np = raw.numpy()
            new = np.clip(raw_np, -1.0, 1.0) * mask
            self._cur = np.where(decide[:, None], new, self._cur).astype(np.float32)
            if learn:
                if self._pending is not None:
                    self._close(z, val.numpy(), np.zeros(n, dtype=bool))
                self._pending = {"z": z.copy(), "a": raw_np.copy(), "logp": logp.numpy().copy(),
                                 "v": val.numpy().copy(), "mask": mask.copy(), "anchor": self._anchor.copy(),
                                 "r": np.zeros(n), "k": 0}
        self._age += 1
        self._usage.infer_cpu_s += time.perf_counter() - t0
        return self._cur.copy()

    # ---- experience --------------------------------------------------------------------------------
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen or not self._explored or self._pending is None:
            return
        p = self._pending
        p["r"] += (self.hp["gamma"] ** p["k"]) * np.asarray(reward, dtype=np.float64)
        p["k"] += 1
        if np.asarray(done).all():
            self._close(None, np.zeros(len(p["r"])), np.ones(len(p["r"]), dtype=bool))

    def _close(self, z_next, v_next, done) -> None:
        p = self._pending
        self._pending = None
        if p["k"] == 0:                              # a decision it was never told the outcome of is not experience
            return
        p["v_next"], p["done"] = np.asarray(v_next, dtype=np.float64).copy(), np.asarray(done, dtype=bool).copy()
        self.buffer.append(p)

    # ---- learning ----------------------------------------------------------------------------------
    def update(self) -> dict:
        if self.frozen or not self.buffer:
            return {"updates": 0}
        t0 = time.perf_counter()
        torch, hp = self._torch, self.hp
        if self._pending is not None:               # a match cut short: bootstrap from the last value
            self._close(None, self._pending["v"], np.zeros(len(self._pending["r"]), dtype=bool))
        buf = self.buffer
        T, n = len(buf), len(buf[0]["r"])
        r = np.stack([b["r"] for b in buf])                       # [T, n] reward per decision
        self.r_var = (self.r_var * self.r_count + r.var() * r.size) / (self.r_count + r.size)
        self.r_count += r.size
        r = r / np.sqrt(self.r_var + 1e-12)
        v = np.stack([b["v"] for b in buf])
        v_next = np.stack([b["v_next"] for b in buf])
        done = np.stack([b["done"] for b in buf])
        gk = np.stack([np.full(n, hp["gamma"] ** b["k"]) for b in buf])
        adv = np.zeros((T, n))
        last = np.zeros(n)
        for t in range(T - 1, -1, -1):
            nd = 1.0 - done[t]
            delta = r[t] + gk[t] * v_next[t] * nd - v[t]
            last = delta + gk[t] * hp["lam"] * nd * last
            adv[t] = last
        ret = adv + v
        z = torch.from_numpy(np.concatenate([b["z"] for b in buf]))
        a = torch.from_numpy(np.concatenate([b["a"] for b in buf]))
        mask = torch.from_numpy(np.concatenate([b["mask"] for b in buf]))
        anchor = torch.from_numpy(np.concatenate([b["anchor"] for b in buf]))
        logp0 = torch.from_numpy(np.concatenate([b["logp"] for b in buf]).astype(np.float32))
        adv_t = torch.from_numpy(adv.reshape(-1).astype(np.float32))
        ret_t = torch.from_numpy(ret.reshape(-1).astype(np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        steps, kl = 0, 0.0
        before = self.parameters_flat()
        for _ in range(hp["epochs"]):
            mean = anchor + self.pi(z)
            std = self.log_std.exp().clamp_min(hp["sigma_min"])
            logp = (-(0.5 * ((a - mean) / std) ** 2 + std.log() + 0.9189385) * mask).sum(1)
            ratio = (logp - logp0).exp()
            surr = torch.min(ratio * adv_t, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * adv_t).mean()
            ent = ((std.log() + 1.4189385) * mask).sum(1).mean()
            loss = -surr - hp["ent"] * ent
            self.opt_pi.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.pi.parameters()) + [self.log_std], 1.0)
            self.opt_pi.step()
            vloss = ((self.vf(z).squeeze(1) - ret_t) ** 2).mean()
            self.opt_vf.zero_grad()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 1.0)
            self.opt_vf.step()
            steps += 1
            with torch.no_grad():
                mean2 = anchor + self.pi(z)
                std2 = self.log_std.exp().clamp_min(hp["sigma_min"])
                lp2 = (-(0.5 * ((a - mean2) / std2) ** 2 + std2.log() + 0.9189385) * mask).sum(1)
                kl = float((logp0 - lp2).mean())
            if kl > hp["max_kl"]:
                break
        with torch.no_grad():
            self.log_std.clamp_(min=float(np.log(hp["sigma_min"])), max=0.0)
        self.buffer = []
        self.iters += 1
        # 5.0 (audit K 3.1): an optimiser step that leaves the policy unchanged (zero advantage, e.g. no observed
        # income at all) is not a policy update; only steps of a pass that changed the policy are counted
        changed = bool(np.any(self.parameters_flat() != before))
        counted = steps if changed else 0
        self._usage.updates += counted
        self._usage.train_cpu_s += time.perf_counter() - t0
        self.last_stats = {"kl": kl, "vloss": float(vloss.detach()), "sigma": float(self.log_std.detach().exp().mean()),
                           "decisions": T * n, "optimizer_steps": steps, "policy_changed": changed}
        return {"updates": counted, **self.last_stats}

    def freeze(self, frozen: bool = True) -> None:
        super().freeze(frozen)
        if frozen:
            self.buffer, self._pending = [], None

    # ---- the policy, for analysis --------------------------------------------------------------------
    def parameters_flat(self) -> np.ndarray:
        with self._torch.no_grad():
            return np.concatenate([p.detach().numpy().ravel() for p in self.pi.parameters()]
                                  + [self.log_std.detach().numpy().ravel()])

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        d = {"learner": self.name, "hp": self.hp, "pi": self.pi.state_dict(), "vf": self.vf.state_dict(),
             "log_std": self.log_std.detach().clone(), "o_mean": self.o_mean, "o_var": self.o_var,
             "o_count": self.o_count, "r_var": self.r_var, "r_count": self.r_count, "iters": self.iters,
             "gen": self.gen.get_state(), "updates": self._usage.updates}
        if not self.frozen:                          # a frozen copy never trains again: weights only
            d["opt_pi"], d["opt_vf"] = self.opt_pi.state_dict(), self.opt_vf.state_dict()
        self._torch.save(d, path)

    def load(self, path: str) -> None:
        torch = self._torch
        d = torch.load(path, weights_only=False)
        self.pi.load_state_dict(d["pi"])
        self.vf.load_state_dict(d["vf"])
        with torch.no_grad():
            self.log_std.copy_(d["log_std"])
        if "opt_pi" in d:
            self.opt_pi.load_state_dict(d["opt_pi"])
            self.opt_vf.load_state_dict(d["opt_vf"])
        self.o_mean, self.o_var, self.o_count = np.array(d["o_mean"]), np.array(d["o_var"]), float(d["o_count"])
        self.r_var, self.r_count, self.iters = float(d["r_var"]), float(d["r_count"]), int(d["iters"])
        self.gen.set_state(d["gen"])
        self._usage.updates = int(d.get("updates", 0))
        self.buffer, self._pending = [], None


# ====================================================================== scripted operators
class ClimberLearner(Learner):
    """A competent scripted operator that uses nothing but the public interface: coordinate search on its own
    measured net income, which it reads from its own observation (observed revenue minus observed cost; the 4.0
    reward slot is zero outside legacy mode), so it adapts inside any match,
    learning or not. Every `block` ticks it either keeps the last trial (if the block paid more than the block
    before it, by more than `margin` of the larger magnitude) or takes it back, then tries the next lever one
    step up or down. A lever whose last trial failed is tried in the other direction next time, and its step
    shrinks after two failures in a row. With `persist`, the settings it had settled on are where it starts the
    next match (its only long-term memory; not written while frozen)."""
    name, kind = "climber", "scripted"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", block: int = 6, step: float = 0.3,
                 shrink: float = 0.6, margin: float = 0.02, persist: bool = True, min_step: float = 0.04):
        super().__init__(spec, seed, device)
        self.block, self.step0, self.shrink = max(1, int(block)), float(step), float(shrink)
        self.margin, self.persist, self.min_step = float(margin), bool(persist), float(min_step)
        self.rng = np.random.default_rng([int(seed), 0xC11B])
        self.memory: np.ndarray | None = None        # long-term: settings per slot carried between matches
        self._x = self._base = self._acc = self._step = self._dir = self._fails = None
        self._lever = self._trial = None
        self._age = 0
        self._n = 0

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        L = self.spec.act_dim
        self._n = int(n)
        self._x = None
        self._base = np.full(n, np.nan)
        self._acc = np.zeros(n)
        self._step = np.full((n, L), self.step0)
        self._dir = self.rng.choice([-1.0, 1.0], size=(n, L))
        self._fails = np.zeros((n, L), dtype=np.int64)
        self._lever = np.full(n, -1, dtype=np.int64)
        self._trial = np.zeros((n, L))               # the change under trial (0: none)
        self._age = 0

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        sp = self.spec
        n = obs.shape[0]
        if self._base is None or self._n != n:
            self.reset_context(n)
        mask = obs[:, sp.lever_mask] > 0.5
        if self._x is None:
            start = np.asarray(obs[:, sp.last_action], dtype=np.float64)
            if self.persist and self.memory is not None and self.memory.shape == start.shape:
                start = self.memory.copy()
            self._x = np.where(mask, start, 0.0)
        else:
            # own net of the last tick, from its own observed revenue and cost (5.0, Clarification 2: the reward slot
            # is 0 outside legacy mode; observed revenue minus cost equals the 4.0 reward slot to float rounding)
            sc = np.asarray(obs[:, sp.scalars], dtype=np.float64)
            self._acc += sc[:, 1] - sc[:, 2]
        if self._age > 0 and self._age % self.block == 0:
            mean = self._acc / self.block
            for i in range(n):
                levers = np.flatnonzero(mask[i])
                if len(levers) == 0:
                    continue
                j = int(self._lever[i])
                if j >= 0 and self._trial[i].any():
                    tol = self.margin * max(abs(mean[i]), abs(self._base[i]), 1e-9)
                    if mean[i] > self._base[i] + tol:            # keep it; try further the same way later
                        self._base[i] = mean[i]
                        self._fails[i, j] = 0
                    else:                                        # take it back; other direction next time
                        self._x[i] -= self._trial[i]
                        self._dir[i, j] *= -1.0
                        self._fails[i, j] += 1
                        if self._fails[i, j] % 2 == 0:
                            self._step[i, j] = max(self.min_step, self._step[i, j] * self.shrink)
                else:
                    self._base[i] = mean[i]
                self._trial[i] = 0.0
                j = int(levers[(int(np.searchsorted(levers, j)) + 1) % len(levers)]) if j >= 0 else int(levers[0])
                self._lever[i] = j
                new = float(np.clip(self._x[i, j] + self._dir[i, j] * self._step[i, j], -1.0, 1.0))
                if new == self._x[i, j]:                         # at the edge: go the other way
                    self._dir[i, j] *= -1.0
                    new = float(np.clip(self._x[i, j] + self._dir[i, j] * self._step[i, j], -1.0, 1.0))
                self._trial[i, j] = new - self._x[i, j]
                self._x[i, j] = new
            self._acc[:] = 0.0
            if self.persist and not self.frozen:                 # what it has settled on, without the open trial
                self.memory = (self._x - self._trial).copy()
        self._age += 1
        return self._x.astype(np.float32)

    def policy_vector(self) -> np.ndarray | None:
        return None if self.memory is None else self.memory.mean(axis=0)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "seed": self.seed,
                       "memory": None if self.memory is None else self.memory.tolist()}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.memory = None if d.get("memory") is None else np.asarray(d["memory"], dtype=np.float64)


class FixedLearner(Learner):
    """Plays one constant settings vector (in [-1, 1] per lever). `action=None` holds the defaults."""
    name, kind = "fixed", "fixed"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", action: Any = None, label: str = ""):
        super().__init__(spec, seed, device)
        self.action = None if action is None else np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action is not None and len(self.action) != spec.act_dim:
            a = np.zeros(spec.act_dim, dtype=np.float32)
            a[:len(self.action)] = self.action[:spec.act_dim]
            self.action = a
        self.label = str(label)

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if self.action is None:
            return np.asarray(obs[:, self.spec.last_action], dtype=np.float32).copy()
        return np.repeat(self.action[None, :], obs.shape[0], axis=0) * (obs[:, self.spec.lever_mask] > 0.5)

    def policy_vector(self) -> np.ndarray | None:
        return None if self.action is None else self.action.astype(np.float64)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "label": self.label,
                       "action": None if self.action is None else self.action.tolist()}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.label = d.get("label", "")
        self.action = None if d.get("action") is None else np.asarray(d["action"], dtype=np.float32)


# ====================================================================== construction
def make_learner(spec: EnvSpec | None = None, algo: str = "spsa", seed: int = 0, device: str = "cpu",
                 **options: Any) -> Learner:
    """Build a learner by name: one of this module's (`spsa`, `pg`, `climber`, `fixed`, `hold`, `random`) or any
    name of the `research.learners` registry (for example `ppo_gru`, `hill_climb`, `probe_exploit`)."""
    spec = spec or EnvSpec()
    own = {"spsa": SPSALearner, "pg": PGLearner, "climber": ClimberLearner, "fixed": FixedLearner,
           "hold": HoldLearner, "random": RandomLearner}
    if algo in own:
        return own[algo](spec, int(seed), device, **options)
    from ..research import learners as registry
    if algo not in registry.available():
        raise ValueError(f"unknown learner {algo!r}; own: {sorted(own)}; registry: {registry.available()}")
    return registry.make(algo, spec, seed=int(seed), device=device, **options)


def learner_config(algo: str, seed: int = 0, **options: Any) -> dict:
    """The JSON description from which `make_learner` rebuilds a learner (stored with every individual)."""
    return {"algo": str(algo), "seed": int(seed), **options}


def policy_summary(learner: Learner) -> list[float] | None:
    """A learner's current policy as a settings vector where it has one (`spsa`, `fixed`); None otherwise."""
    fn = getattr(learner, "policy_vector", None)
    v = fn() if fn is not None else None
    return None if v is None else [float(x) for x in v]
