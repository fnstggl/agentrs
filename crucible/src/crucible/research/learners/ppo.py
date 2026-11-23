"""`ppo_gru`: a model-free recurrent learner (recurrent PPO).

Algorithm. Proximal policy optimisation with the clipped surrogate objective and generalised advantage estimation
(Schulman et al. 2017), with a recurrent policy trained by backpropagation through whole episodes: the recurrent
state is never reset inside an episode, and during an update it is recomputed from the first tick with the
current weights (no stale stored states). PPO is used because it is the most widely reproduced on-policy
actor-critic, it is the algorithm of the RL^2 line of meta-reinforcement learning this learner belongs to (a
recurrent policy trained across many tasks, so that what the weights encode is a procedure for a new task), and
it needs no replay of off-policy recurrent states.

What can be learned. Lever and gauge slots mean something different in every world, so the weights cannot store
"raise lever 3". What they can store is a procedure: perturb, watch what net does, move the levers that helped,
stop when it stops helping. The action of a lever is therefore its *offset from where it rested at the start of
the episode* (mean plus Gaussian noise whose scale the network also outputs): an untrained network (zero output
layer) holds the starting settings and dithers around them, and the dithering is the experiment whose outcome
the recurrent state can integrate.

Input. Per lever: the setting in force, its resting value, the offset, the change since the last tick, that
change multiplied by the change of reward, revenue and cost (the raw material of a finite-difference estimate),
and exponentially decayed running sums of those products divided by the running sum of squared changes (a noisy
slope of reward, and of cost, with respect to the lever, and how much the lever has been moved). The policy head
also reads the lever's input directly (a skip connection around the recurrent layers).
Per gauge: the symmetric-log reading and its change. Global: reward, revenue, cost and their changes, balance,
clock, lever count.

Two architectures over the same input, same algorithm (`arch`):

    "set"   one GRU shared by all lever slots (each slot keeps its own state), one shared by all gauge slots, a
            global GRU over their pooled states, and a second shared lever GRU that also reads the global state;
            the policy head reads a lever's own state and the global state. Permuting the slots permutes the
            actions: nothing is tied to a slot.                                                  (`ppo_gru`)
    "flat"  one GRU over the concatenated, padded features; one output per slot.               (`ppo_gru_flat`)

`explore=False` and the within-episode noise. The dithering is how this learner experiments inside an episode,
so with `eval_sample=True` (the default) `act(..., explore=False)` still samples, with the noise scale times
`eval_temp`, from a per-slot generator re-seeded at `reset_context` (an evaluation is reproducible).
`eval_sample=False` gives the mean action.
"""
from __future__ import annotations

import time

import numpy as np

from ..learner import EnvSpec, Learner
from .util import cpu_adam

PPO_DEFAULTS = {
    "arch": "set", "hidden": 32, "lr": 1e-3, "gamma": 0.995, "lam": 0.95, "clip": 0.2, "epochs": 5,
    "minibatches": 4, "rollout_steps": 4096, "ent_coef": 0.003, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "init_log_std": -1.6, "min_log_std": -4.0, "max_log_std": -0.5, "eval_sample": True, "eval_temp": 1.0,
    "max_ticks": 512,
}
F_LEVER, F_GAUGE, F_GLOBAL = 10, 2, 10
K_PROD = 20.0
TRACE = 0.97                    # decay of the per-lever running sums of (change of setting) x (change of money)


def _torch():
    import torch
    return torch


def _build(spec: EnvSpec, p: dict, seed: int):
    torch = _torch()
    nn = torch.nn
    n_l, n_g, d = spec.max_levers, spec.max_gauges, int(p["hidden"])

    class SetNet(nn.Module):
        """lever GRU (shared over slots) -> pooled -> global GRU -> second lever GRU that also reads the global
        state -> per-lever policy head. Every recurrent layer runs over the whole sequence in one call."""

        def __init__(self):
            super().__init__()
            dg = d // 2
            self.dg = dg
            self.lever_in = nn.Linear(F_LEVER + F_GLOBAL, d)
            self.gru_l1 = nn.GRU(d, d)
            self.gauge_in = nn.Linear(F_GAUGE + F_GLOBAL, dg)
            self.gru_g = nn.GRU(dg, dg)
            self.gru_c = nn.GRU(2 * d + dg + F_GLOBAL, d)
            self.gru_l2 = nn.GRU(2 * d, d)
            self.pi = nn.Sequential(nn.Linear(2 * d + F_LEVER, d), nn.Tanh(), nn.Linear(d, 2))
            self.v = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1))

        def initial(self, b: int, nl: int = n_l, ng: int = n_g):
            return (torch.zeros(1, b * nl, d), torch.zeros(1, b * ng, self.dg), torch.zeros(1, b, d),
                    torch.zeros(1, b * nl, d))

        def slots(self, state, b: int):
            """Views of the state with the environment as the leading dimension (for per-environment resets)."""
            return [s.view(b, -1) for s in state]

        def forward(self, lf, gf, glob, lmask, gmask, state):
            # the number of slots is whatever the input carries (an update drops slots no world of the batch has)
            t, b, n_l, n_g = lf.shape[0], lf.shape[1], lf.shape[2], gf.shape[2]
            g_l = glob[:, :, None, :].expand(-1, -1, n_l, -1)
            x = torch.tanh(self.lever_in(torch.cat([lf, g_l], dim=3))).reshape(t, b * n_l, d)
            h1, s1 = self.gru_l1(x, state[0])
            h1 = h1.reshape(t, b, n_l, d) * lmask[None, :, :, None]
            y = torch.tanh(self.gauge_in(torch.cat([gf, glob[:, :, None, :].expand(-1, -1, n_g, -1)], dim=3)))
            hg, sg = self.gru_g(y.reshape(t, b * n_g, self.dg), state[1])
            hg = hg.reshape(t, b, n_g, self.dg) * gmask[None, :, :, None]
            n_real = lmask.sum(1).clamp(min=1.0)[None, :, None]
            neg = 1e4 * (1.0 - lmask)[None, :, :, None]
            pooled = torch.cat([h1.sum(2) / n_real, (h1 - neg).max(2).values,
                                hg.sum(2) / gmask.sum(1).clamp(min=1.0)[None, :, None], glob], dim=2)
            hc, sc = self.gru_c(pooled, state[2])
            z = torch.cat([h1, hc[:, :, None, :].expand(-1, -1, n_l, -1)], dim=3).reshape(t, b * n_l, 2 * d)
            h2, s2 = self.gru_l2(z, state[3])
            h2 = h2.reshape(t, b, n_l, d) * lmask[None, :, :, None]
            out = self.pi(torch.cat([h2, hc[:, :, None, :].expand(-1, -1, n_l, -1), lf], dim=3))
            val = self.v(torch.cat([hc, h2.sum(2) / n_real, (h2 - neg).max(2).values], dim=2))[:, :, 0]
            return out[..., 0], out[..., 1], val, (s1, sg, sc, s2)

    class FlatNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = 2 * d
            self.h = h
            n_in = n_l * (F_LEVER + 1) + n_g * (F_GAUGE + 1) + F_GLOBAL
            self.inp = nn.Linear(n_in, h)
            self.gru = nn.GRU(h, h)
            self.pi = nn.Sequential(nn.Linear(h + n_in, h), nn.Tanh(), nn.Linear(h, 2 * n_l))
            self.v = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

        def initial(self, b: int, nl: int = n_l, ng: int = n_g):
            return (torch.zeros(1, b, self.h),)

        def slots(self, state, b: int):
            return [s.view(b, -1) for s in state]

        def forward(self, lf, gf, glob, lmask, gmask, state):
            t, b = lf.shape[0], lf.shape[1]
            x = torch.cat([lf.reshape(t, b, -1), lmask[None].expand(t, -1, -1), gf.reshape(t, b, -1),
                           gmask[None].expand(t, -1, -1), glob], dim=2)
            hc, s = self.gru(torch.tanh(self.inp(x)), state[0])
            out = self.pi(torch.cat([hc, x], dim=2)).reshape(t, b, n_l, 2)
            return out[..., 0], out[..., 1], self.v(hc)[:, :, 0], (s,)

    gen_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))                        # seeded initialisation; the global stream is restored
    try:
        net = SetNet() if p["arch"] == "set" else FlatNet()
        with torch.no_grad():
            net.pi[-1].weight.zero_()
            net.pi[-1].bias.zero_()
            net.pi[-1].bias.view(-1, 2)[:, 1].fill_(float(p["init_log_std"]))
    finally:
        torch.random.set_rng_state(gen_state)
    return net


class PPOGRULearner(Learner):
    name, kind = "ppo_gru", "model_free"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", **kw):
        super().__init__(spec, seed, "cpu")
        bad = sorted(set(kw) - set(PPO_DEFAULTS))
        if bad:
            raise TypeError(f"ppo_gru has no option(s) {bad}")
        self.p = dict(PPO_DEFAULTS, **kw)
        if self.p["arch"] not in ("set", "flat"):
            raise ValueError("arch must be 'set' or 'flat'")
        torch = _torch()
        self.net = _build(spec, self.p, seed)
        self.opt = cpu_adam(self.net.parameters(), lr=float(self.p["lr"]), eps=1e-5)
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self.ret_var = 1.0                          # running variance of returns (part of the long-term state)
        self._train_rng = np.random.default_rng([self.seed, 86028121])
        self._perm_rng = np.random.default_rng([self.seed, 49979687])
        self._n = 0
        self._done_eps: list[dict] = []             # finished episodes waiting for the next update
        self._done_steps = 0
        self.last_stats: dict = {}

    # ---- context -----------------------------------------------------------------------------------
    def _alloc(self, n: int) -> None:
        sp = self.spec
        self._n = n
        self._state = self.net.initial(n)
        self._x0 = np.zeros((n, sp.max_levers), np.float32)
        self._prev_a = np.zeros((n, sp.max_levers), np.float32)
        self._prev_g = np.zeros((n, sp.max_gauges), np.float32)
        self._prev_m = np.zeros((n, 3), np.float32)
        self._trace = np.zeros((n, sp.max_levers, 3), np.float32)      # running sums: da*dr, da*dcost, da*da
        self._t = np.zeros(n, np.int64)
        self._rngs = [np.random.default_rng([self.seed, 2147483647, i]) for i in range(n)]
        self._traj: list[dict | None] = [None] * n
        self._live = np.ones(n, dtype=bool)
        self._pending = None

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._n != n or which is None:
            self._alloc(n)
            return
        w = np.asarray(which)
        idx = np.flatnonzero(w) if w.dtype == bool else w.astype(np.int64)
        torch = _torch()
        for view in self.net.slots(self._state, n):             # zero the recurrent state of those environments
            view[torch.as_tensor(idx, dtype=torch.long)] = 0.0
        for i in idx:
            i = int(i)
            self._t[i] = 0
            self._rngs[i] = np.random.default_rng([self.seed, 2147483647, i])
            self._traj[i] = None
            self._live[i] = True

    def _features(self, obs: np.ndarray):
        sp = self.spec
        a = obs[:, sp.last_action]
        g = obs[:, sp.gauges]
        lmask, gmask = obs[:, sp.lever_mask], obs[:, sp.gauge_mask]
        sc = obs[:, sp.scalars]
        first = self._t == 0
        if first.any():
            self._x0[first], self._prev_a[first] = a[first], a[first]
            self._prev_g[first], self._prev_m[first] = g[first], 0.0
            self._trace[first] = 0.0
        da = a - self._prev_a
        dm = sc[:, :3] - self._prev_m                                   # change of reward, revenue, cost
        tr = self._trace
        tr *= TRACE
        tr[:, :, 0] += da * dm[:, 0:1]
        tr[:, :, 1] += da * dm[:, 2:3]
        tr[:, :, 2] += da * da
        den = tr[:, :, 2] + 1e-3
        lf = np.stack([a, self._x0, a - self._x0, da, K_PROD * da * dm[:, 0:1], K_PROD * da * dm[:, 1:2],
                       K_PROD * da * dm[:, 2:3], np.clip(5.0 * tr[:, :, 0] / den, -5.0, 5.0),
                       np.clip(5.0 * tr[:, :, 1] / den, -5.0, 5.0), np.minimum(tr[:, :, 2], 3.0)],
                      axis=2) * lmask[:, :, None]
        gf = np.stack([0.25 * g, np.clip(g - self._prev_g, -3.0, 3.0)], axis=2) * gmask[:, :, None]
        glob = np.concatenate([5.0 * sc[:, :3], 5.0 * dm, np.clip(sc[:, 3:4], 0.0, 3.0), sc[:, 4:5],
                               first[:, None].astype(np.float32), lmask.sum(1, keepdims=True) / sp.max_levers],
                              axis=1)
        self._prev_a, self._prev_g, self._prev_m = a.copy(), g.copy(), sc[:, :3].copy()
        return (lf.astype(np.float32), gf.astype(np.float32), glob.astype(np.float32),
                lmask.astype(np.float32), gmask.astype(np.float32))

    # ---- acting ------------------------------------------------------------------------------------
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        c0 = time.process_time()
        torch = _torch()
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        if self._n != n:
            self._alloc(n)
        lf, gf, glob, lmask, gmask = self._features(obs)
        with torch.no_grad():
            mu, ls, val, self._state = self.net(torch.from_numpy(lf)[None], torch.from_numpy(gf)[None],
                                                torch.from_numpy(glob)[None], torch.from_numpy(lmask),
                                                torch.from_numpy(gmask), self._state)
        mu, val = mu[0].numpy(), val[0]
        ls = np.clip(ls[0].numpy(), float(self.p["min_log_std"]), float(self.p["max_log_std"]))
        training = explore and not self.frozen
        if training:
            eps = self._train_rng.standard_normal(mu.shape).astype(np.float32)
        elif self.p["eval_sample"]:
            eps = float(self.p["eval_temp"]) * np.stack(
                [r.standard_normal(mu.shape[1]) for r in self._rngs]).astype(np.float32)
        else:
            eps = np.zeros_like(mu)
        u = (mu + np.exp(ls) * eps) * lmask
        action = np.clip(self._x0 + u, -1.0, 1.0).astype(np.float32)
        if not self.frozen:
            logp = (-0.5 * ((u - mu) / np.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)) * lmask
            self._pending = {"lf": lf, "gf": gf, "glob": glob, "lmask": lmask, "gmask": gmask, "u": u,
                             "logp": logp.sum(1), "val": val.numpy().copy()}
        self._t += 1
        self._usage.infer_cpu_s += time.process_time() - c0
        return action

    # ---- parametric learning -----------------------------------------------------------------------
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen or self._pending is None:
            return
        c0 = time.process_time()
        pd = self._pending
        self._pending = None
        for i in range(self._n):
            if not self._live[i]:
                continue
            tr = self._traj[i]
            if tr is None:
                tr = self._traj[i] = {"lmask": pd["lmask"][i].copy(), "gmask": pd["gmask"][i].copy(), "lf": [],
                                      "gf": [], "glob": [], "u": [], "logp": [], "val": [], "rew": []}
            if len(tr["rew"]) < int(self.p["max_ticks"]):
                for k in ("lf", "gf", "glob", "u"):
                    tr[k].append(pd[k][i].copy())
                tr["logp"].append(float(pd["logp"][i]))
                tr["val"].append(float(pd["val"][i]))
                tr["rew"].append(float(reward[i]))
            self._usage.extra["observed_steps"] = self._usage.extra.get("observed_steps", 0) + 1
            if done[i]:
                self._live[i] = False
                self._traj[i] = None
                ep = {k: np.asarray(v, dtype=np.float32) for k, v in tr.items()}
                self._done_eps.append(ep)
                self._done_steps += len(ep["rew"])
        self._usage.train_cpu_s += time.process_time() - c0

    def update(self) -> dict:
        if self.frozen or self._done_steps < int(self.p["rollout_steps"]):
            return {"updates": 0}
        c0 = time.process_time()
        eps, self._done_eps, self._done_steps = self._done_eps, [], 0
        k = self._ppo(eps)
        self._usage.updates += k
        self._usage.train_cpu_s += time.process_time() - c0
        return dict(self.last_stats, updates=k)

    def _ppo(self, eps: list[dict]) -> int:
        torch = _torch()
        p = self.p
        gamma, lam = float(p["gamma"]), float(p["lam"])
        # returns and advantages per episode (an episode ends at its horizon: no bootstrap)
        raw = []
        for ep in eps:
            r, v = ep["rew"], ep["val"] * np.sqrt(self.ret_var)
            adv = np.zeros_like(r)
            last = 0.0
            for t in range(len(r) - 1, -1, -1):
                nv = v[t + 1] if t + 1 < len(r) else 0.0
                last = r[t] + gamma * nv - v[t] + gamma * lam * last
                adv[t] = last
            ep["adv"], ep["ret"] = adv, adv + v
            raw.append(ep["ret"])
        allret = np.concatenate(raw)
        self.ret_var = float(0.9 * self.ret_var + 0.1 * max(1e-6, float(np.var(allret)))) \
            if self._usage.updates else float(max(1e-6, np.var(allret)))
        scale = float(np.sqrt(self.ret_var))
        alladv = np.concatenate([ep["adv"] for ep in eps])
        a_mean, a_std = float(alladv.mean()), float(alladv.std() + 1e-8)
        n_mb = max(1, min(int(p["minibatches"]), len(eps)))
        steps, stats = 0, {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "kl": 0.0}
        lo, hi = float(p["min_log_std"]), float(p["max_log_std"])
        for _ in range(int(p["epochs"])):
            order = self._perm_rng.permutation(len(eps))
            for mb in np.array_split(order, n_mb):
                batch = [eps[int(i)] for i in mb]
                b, t_max = len(batch), max(len(e["rew"]) for e in batch)

                def pad(key, shape=()):
                    out = np.zeros((t_max, b) + shape, np.float32)
                    for j, e in enumerate(batch):
                        out[:len(e["rew"]), j] = e[key]
                    return torch.from_numpy(out)
                lf = pad("lf", batch[0]["lf"].shape[1:])
                gf = pad("gf", batch[0]["gf"].shape[1:])
                glob = pad("glob", batch[0]["glob"].shape[1:])
                u = pad("u", batch[0]["u"].shape[1:])
                old_lp, adv, ret = pad("logp"), pad("adv"), pad("ret")
                tmask = torch.zeros(t_max, b)
                for j, e in enumerate(batch):
                    tmask[:len(e["rew"]), j] = 1.0
                lmask = torch.from_numpy(np.stack([e["lmask"] for e in batch]))
                gmask = torch.from_numpy(np.stack([e["gmask"] for e in batch]))
                nl, ng = lmask.shape[1], gmask.shape[1]
                if p["arch"] == "set":                          # slots that exist in no world of the minibatch
                    nl = max(1, int(lmask.any(0).nonzero().max()) + 1) if bool(lmask.any()) else 1
                    ng = max(1, int(gmask.any(0).nonzero().max()) + 1) if bool(gmask.any()) else 1
                    lf, u, lmask, gf, gmask = lf[:, :, :nl], u[:, :, :nl], lmask[:, :nl], gf[:, :, :ng], gmask[:, :ng]
                mu, ls, val, _ = self.net(lf, gf, glob, lmask, gmask, self.net.initial(b, nl, ng))
                ls = ls.clamp(lo, hi)
                lp = ((-0.5 * ((u - mu) / ls.exp()) ** 2 - ls - 0.5 * float(np.log(2 * np.pi))) * lmask[None]).sum(2)
                ent = ((0.5 + 0.5 * float(np.log(2 * np.pi)) + ls) * lmask[None]).sum(2)
                ratio = (lp - old_lp).clamp(-20.0, 20.0).exp()
                a = (adv - a_mean) / a_std
                surr = torch.min(ratio * a, ratio.clamp(1.0 - float(p["clip"]), 1.0 + float(p["clip"])) * a)
                n_valid = tmask.sum().clamp(min=1.0)
                pi_loss = -(surr * tmask).sum() / n_valid
                v_loss = (((val - ret / scale) ** 2) * tmask).sum() / n_valid
                entropy = (ent * tmask).sum() / n_valid
                loss = pi_loss + float(p["vf_coef"]) * v_loss - float(p["ent_coef"]) * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(p["max_grad_norm"]))
                self.opt.step()
                steps += 1
                with torch.no_grad():
                    stats["pi_loss"] += float(pi_loss)
                    stats["v_loss"] += float(v_loss)
                    stats["entropy"] += float(entropy)
                    stats["clip_frac"] += float((((ratio - 1.0).abs() > float(p["clip"])).float() * tmask).sum() / n_valid)
                    stats["kl"] += float(((old_lp - lp) * tmask).sum() / n_valid)
        self.last_stats = {k: v / max(1, steps) for k, v in stats.items()}
        self.last_stats.update(episodes=len(eps), mean_return=float(np.mean([e["rew"].sum() for e in eps])),
                               ret_std=scale)
        return steps

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch = _torch()
        torch.save({"learner": self.name, "params": self.p, "seed": self.seed, "net": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "ret_var": self.ret_var, "updates": int(self._usage.updates)},
                   path)

    def load(self, path: str) -> None:
        """Restore weights, optimiser state and return statistics. The network shape (`arch`, `hidden`) follows
        the checkpoint; every other option stays as this learner was constructed."""
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        if any(d["params"][k] != self.p[k] for k in ("arch", "hidden")):
            self.p.update({k: d["params"][k] for k in ("arch", "hidden")})
            self.net = _build(self.spec, self.p, self.seed)
            self.opt = cpu_adam(self.net.parameters(), lr=float(self.p["lr"]), eps=1e-5)
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self.ret_var = float(d.get("ret_var", 1.0))
        self._usage.params = int(sum(q.numel() for q in self.net.parameters()))
        self._usage.updates = int(d.get("updates", 0))
        self._n = 0
