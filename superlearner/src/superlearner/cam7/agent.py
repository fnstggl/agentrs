"""CAM 7.0 as a Crucible learner (the architecture-neutral contract in `crucible.research.learner`).

Fast adaptation happens inside `act` (the history of the current episode and the posterior it implies; cleared by
`reset_context`). Parametric learning happens inside `update` from experience delivered by `observe`:

    L = L_meta + world_coef L_world + [L_policy + value_coef L_value - entropy_coef H]

L_meta: support/query negative log predictive density (rows < k build the posterior, later rows are scored; k drawn from
meta_ks and from later cuts). It trains the backbone's lever context, the feature networks and the priors.
L_world: auxiliary predictions (next gauge change, net of the next 8 ticks) from pass 1.
L_policy / L_value: clipped importance-weighted actor-critic over the candidate decisions actually taken, on actual
outcomes, with generalised advantages over decisions; the Bayesian quantities enter the policy as fixed inputs.

Replay keeps whole episodes (successful and unsuccessful) with the decisions taken, their candidate sets, behaviour
log-probabilities, model version and the rule that generated them; `save` writes it next to the weights.
"""
from __future__ import annotations

import io
import json
import math
import os
import time

import numpy as np
import torch

from crucible.research.learner import EnvSpec, Learner

from .config import SCALES, Cam7Config
from .model import Cam7Model, History, N_CAND_FLAGS

T_MAX = 256


def _identities(n: int, k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, k, dim))
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


class _Episode:
    """One environment's fast state (numpy, CPU)."""

    def __init__(self, spec: EnvSpec, cfg: Cam7Config, rng: np.random.Generator):
        L, G = spec.max_levers, spec.max_gauges
        self.x = np.zeros((T_MAX, L), np.float32)
        self.ema = np.zeros((T_MAX, L), np.float32)
        self.out = np.zeros((T_MAX, 2), np.float32)
        self.gauges = np.zeros((T_MAX, G), np.float32)
        self.tau = np.zeros(T_MAX, np.float32)
        self.funds = np.zeros(T_MAX, np.float32)
        self.pred = np.zeros((T_MAX, 2), np.float32)
        self.cfg = cfg
        self.rows = 0
        self.t = 0
        self.h = 120
        self.since = 0
        self.safe = False
        self.keep_from = 0
        self.resets = 0
        self.pending = np.zeros(2, np.float32)
        self.decisions: list[dict] = []
        self.fixed_k = 0

    def start(self, obs: np.ndarray, spec: EnvSpec, rng: np.random.Generator) -> None:
        self.lid = _identities(1, spec.max_levers, self.cfg.id_dim, rng)[0]
        self.gid = _identities(1, spec.max_gauges, self.cfg.id_dim, rng)[0]
        self.lmask = (obs[spec.lever_mask] > 0.5).astype(np.float32)
        self.gmask = (obs[spec.gauge_mask] > 0.5).astype(np.float32)
        self.x0 = np.asarray(obs[spec.last_action], np.float32) * self.lmask
        self.target = self.x0.copy()
        self.clock0 = float(obs[spec.scalars][4])
        self.ema_now = self.x0.copy()

    def ingest(self, obs: np.ndarray, spec: EnvSpec, ema: float) -> None:
        if self.rows >= T_MAX:
            return
        sc = obs[spec.scalars]
        x = np.asarray(obs[spec.last_action], np.float32) * self.lmask
        self.ema_now = (1 - ema) * self.ema_now + ema * x
        r = self.rows
        self.x[r], self.ema[r] = x, self.ema_now
        self.out[r] = (sc[1], sc[2])
        self.gauges[r] = obs[spec.gauges] * self.gmask
        self.tau[r], self.funds[r] = sc[4], sc[3]
        self.pred[r] = self.pending
        if self.rows == 0 and sc[4] > self.clock0:
            self.h = max(2, int(round((1.0 - self.clock0) / (sc[4] - self.clock0))))
        self.rows += 1

    def record(self) -> dict:
        n = self.rows
        return {"x": self.x[:n].copy(), "ema": self.ema[:n].copy(), "out": self.out[:n].copy(),
                "gauges": self.gauges[:n].copy(), "tau": self.tau[:n].copy(), "funds": self.funds[:n].copy(),
                "pred": self.pred[:n].copy(), "x0": self.x0.copy(), "lmask": self.lmask.copy(),
                "gmask": self.gmask.copy(), "lid": self.lid, "gid": self.gid, "decisions": self.decisions,
                "h": self.h}


def history_batch(eps: list, cfg: Cam7Config, device, rng: np.random.Generator, live: bool = False) -> History:
    """Stack episodes (live `_Episode`s or replay records) into a padded History on `device`."""
    T = max(1, max((e.rows if live else len(e["x"])) for e in eps))
    T = min(T_MAX, T)

    def arr(name, shape_tail, dtype=np.float32):
        out = np.zeros((len(eps), T, *shape_tail), dtype)
        for i, e in enumerate(eps):
            v = getattr(e, name)[:T] if live else e[name][:T]
            n = e.rows if live else len(e[name])
            out[i, :min(n, T)] = v[:min(n, T)]
        return out

    L = len(eps[0].x0 if live else eps[0]["x0"])
    G = len(eps[0].gmask if live else eps[0]["gmask"])
    g = lambda e, k: getattr(e, k) if live else e[k]
    lid = np.stack([g(e, "lid") for e in eps])       # the identities the episode was acted with
    gid = np.stack([g(e, "gid") for e in eps])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return History(x=t(arr("x", (L,))), ema=t(arr("ema", (L,))), out=t(arr("out", (2,))), gauges=t(arr("gauges", (G,))),
                   tau=t(arr("tau", ())), funds=t(arr("funds", ())), pred=t(arr("pred", (2,))),
                   x0=t(np.stack([g(e, "x0") for e in eps])), lmask=t(np.stack([g(e, "lmask") for e in eps])),
                   gmask=t(np.stack([g(e, "gmask") for e in eps])), lid=t(lid), gid=t(gid))


class Cam7Learner(Learner):
    name, kind = "cam7", "model_based"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cuda", cfg: Cam7Config | None = None,
                 scale: str = "base", **kw):
        super().__init__(spec, seed, device)
        self.cfg = (cfg or SCALES[scale]()).with_(**kw) if kw else (cfg or SCALES[scale]())
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = Cam7Model(self.cfg, spec.max_levers, spec.max_gauges).to(device)
        self.opt = torch.optim.AdamW(self.model.param_groups(self.cfg.lr, self.cfg.prior_lr_mult), weight_decay=1e-4)
        self.replay: list[dict] = []
        self.version = 0
        self.new_episodes = 0
        self.meta_debt = self.rl_debt = 0.0
        self.ret_stats = np.array([0.0, 1.0, 1e-4])        # running mean, var, count of decision returns
        self.slots: list[_Episode] = []
        self.eval_gen = None
        self.stats: dict = {}
        self._usage.params = sum(p.numel() for p in self.model.parameters())

    # ----------------------------------------------------------------------------------------- context
    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if which is None or len(self.slots) != n:
            self.slots = [_Episode(self.spec, self.cfg, self.rng) for _ in range(n)]
        else:
            for i in np.flatnonzero(np.asarray(which)):
                self.slots[i] = _Episode(self.spec, self.cfg, self.rng)
        # frozen evaluation (explore=False) is reproducible: identities, Thompson draws and random candidates come
        # from generators re-seeded here
        self.eval_gen = torch.Generator(device=self.device)
        self.eval_gen.manual_seed(self.seed)
        self.ctx_rng = np.random.default_rng([self.seed, 1])

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        t0, c0 = time.perf_counter(), time.process_time()
        sp, cfg = self.spec, self.cfg
        if len(self.slots) != len(obs):
            self.reset_context(len(obs))
        deciders = []
        for i, (s, o) in enumerate(zip(self.slots, obs)):
            clock = float(o[sp.scalars][4])
            if s.t == 0:
                s.start(o, sp, self.rng if explore else self.ctx_rng)
            elif clock < 1.0 - 1e-9:
                s.ingest(o, sp, cfg.ema)
                s.since += 1
                if not s.safe and float(o[sp.scalars][3]) < cfg.min_funds:
                    s.safe, s.target = True, s.x0.copy()
                if not s.safe and s.since >= cfg.dwell:
                    deciders.append(i)
            s.t += 1
        if deciders:
            self._decide(deciders, explore)
        out = np.zeros((len(obs), sp.act_dim), np.float32)
        for i, s in enumerate(self.slots):
            out[i] = s.target
        if self.device.startswith("cuda"):
            self._usage.infer_gpu_s += time.perf_counter() - t0
        else:
            self._usage.infer_cpu_s += time.process_time() - c0
        return out

    def _change_point(self, s: _Episode) -> None:
        """The model_based rule: if recent squared prediction errors are surprise^2 times the earlier ones, the world
        changed: drop evidence before the recent window (it describes a world that no longer exists)."""
        w, n = self.cfg.surprise_window, s.rows
        has = s.pred[s.keep_from:n, 1] > 0
        if has.sum() < 3 * w or s.h - s.t <= 4 * w:
            return
        net = s.out[s.keep_from:n, 0] - s.out[s.keep_from:n, 1]
        r2 = ((net - s.pred[s.keep_from:n, 0]) ** 2)[has]
        if r2[-w:].mean() > self.cfg.surprise ** 2 * max(r2[w:-w].mean(), 1e-12):
            s.keep_from, s.resets = n - w, s.resets + 1

    @torch.no_grad()
    def _decide(self, idx: list[int], explore: bool) -> None:
        cfg, m = self.cfg, self.model
        m.eval()
        slots = [self.slots[i] for i in idx]
        for s in slots:
            s.since = 0
            self._change_point(s)
        H = history_batch(slots, cfg, self.device, self.rng, live=True)
        cut = torch.tensor([s.rows for s in slots], device=self.device)
        T = H.x.shape[1]
        keep = (torch.arange(T, device=self.device)[None] >= torch.tensor([s.keep_from for s in slots],
                                                                           device=self.device)[:, None]).float()
        innov = torch.tensor([self._innov_rms(s) for s in slots], device=self.device, dtype=torch.float32)
        n_rem = torch.tensor([max(0, (s.h - s.t) // cfg.dwell - 1) for s in slots], device=self.device)
        pre = m.prepare(H, cut, keep)
        gen = None if explore else self.eval_gen
        n_th = cfg.n_thompson if cfg.explore_rule != "greedy" else 0
        cand, valid, flags = m.propose(H, cut, pre["post"], pre["ctx"], pre["x_now"], max(1, n_th), cfg.n_random, gen,
                                       None if explore else self.ctx_rng)
        out = m.decide(H, cut, cand, valid, flags, n_rem, keep, innov, pre=pre)
        logits = out["logits"] / max(cfg.policy_temp, 1e-6)
        logp = torch.log_softmax(logits, -1)
        B, C = valid.shape
        rule = cfg.explore_rule
        if rule == "thompson":
            choice = torch.full((B,), 2, device=self.device)
        elif rule == "random":
            r = torch.rand(B, C, device=self.device, generator=gen) * valid
            choice = r.argmax(-1)
        elif rule == "fixed_design":
            choice = torch.tensor([self._fixed_design(s, cand.shape[2], C) for s in slots], device=self.device)
            choice = torch.where(valid.gather(1, choice[:, None]).squeeze(1), choice, torch.zeros_like(choice))
        elif explore:
            choice = torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
        else:
            choice = logits.argmax(-1)
        self._usage.search_evals += int(valid.sum()) + int(B * cfg.grid * self.spec.max_levers * (1 + n_th))
        mean, post = out["mean"], out["post"]
        # the prediction for the ticks this decision covers: mean net per tick, and its predictive sd
        sd_pred = (post.s2.sum(-1) + 0.0).sqrt()
        ch = choice.cpu().numpy()
        cand_np, valid_np, flags_np = cand.cpu().numpy(), valid.cpu().numpy(), flags.cpu().numpy()
        lp = logp.gather(1, choice[:, None]).squeeze(1).cpu().numpy()
        val = out["value"].cpu().numpy()
        mu = mean.gather(1, choice[:, None]).squeeze(1).cpu().numpy()
        sdp = sd_pred.cpu().numpy()
        nr = n_rem.cpu().numpy()
        innov_np = innov.cpu().numpy()
        sd_np = out["sd"].cpu().numpy()
        for k, s in enumerate(slots):
            s.target = cand_np[k, ch[k]] * s.lmask
            s.pending = np.array([mu[k], sdp[k]], np.float32)
            if not self.frozen:
                s.decisions.append({"row": s.rows, "cand": cand_np[k].astype(np.float16), "valid": valid_np[k],
                                    "flags": flags_np[k].astype(np.uint8), "choice": int(ch[k]), "logp": float(lp[k]),
                                    "value": float(val[k]), "n_rem": int(nr[k]), "keep_from": int(s.keep_from),
                                    "innov": float(innov_np[k]), "sd": float(sd_np[k]), "version": self.version,
                                    "rule": rule})

    def _innov_rms(self, s: _Episode) -> float:
        n = s.rows
        lo = max(s.keep_from, n - self.cfg.surprise_window)
        p = s.pred[lo:n]
        ok = p[:, 1] > 0
        if not ok.any():
            return 0.0
        net = s.out[lo:n, 0] - s.out[lo:n, 1]
        z = (net[ok] - p[ok, 0]) / np.maximum(p[ok, 1], 1e-9)
        return float(min(8.0, np.sqrt(np.mean(z * z))))

    def _fixed_design(self, s: _Episode, L: int, C: int) -> int:
        """A fixed experimental design: cycle through the levers, each to the low then the high edge of the trust
        region (candidate order: hold, optima, L low probes, L high probes), then exploit the posterior mean optimum."""
        n_opt = 1 + max(1, self.cfg.n_thompson)
        levers = np.flatnonzero(s.lmask)
        k = s.fixed_k
        s.fixed_k += 1
        if k >= 2 * len(levers):
            return 1
        j = levers[k // 2]
        return 1 + n_opt + (j if k % 2 == 0 else L + j)

    # ----------------------------------------------------------------------------------------- experience
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen:
            return
        for i in np.flatnonzero(np.asarray(done)):
            s = self.slots[i]
            if s.rows < T_MAX:
                s.ingest(next_obs[i], self.spec, self.cfg.ema)
            if s.rows >= 8:
                rec = s.record()
                rec["version"] = self.version
                self.replay.append(rec)
                self.new_episodes += 1
            s.decisions = []
        if len(self.replay) > self.cfg.replay_episodes:
            del self.replay[:len(self.replay) - self.cfg.replay_episodes]

    def update(self) -> dict:
        cfg = self.cfg
        if self.frozen or len(self.replay) < cfg.min_episodes or self.new_episodes == 0:
            return {"updates": 0}
        t0 = time.perf_counter()
        self.meta_debt += cfg.meta_steps_per_episode * self.new_episodes
        self.rl_debt += cfg.rl_steps_per_episode * self.new_episodes
        self.new_episodes = 0
        n = 0
        info: dict = {}
        self.model.train()
        while self.meta_debt >= 1 or self.rl_debt >= 1:
            do_rl = self.rl_debt >= 1 and (cfg.use_residual or cfg.explore_rule == "learned_only")
            loss, inf = self._meta_step()
            if do_rl:
                rl_loss, rinf = self._rl_step()
                if rl_loss is not None:
                    loss = loss + rl_loss
                    inf.update(rinf)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            inf["grad_norm"] = float(gn)
            for k, v in inf.items():
                info[k] = 0.9 * info.get(k, v) + 0.1 * v if np.isfinite(v) else info.get(k, v)
            self.meta_debt = max(0.0, self.meta_debt - 1)
            if do_rl:
                self.rl_debt = max(0.0, self.rl_debt - 1)
            elif self.rl_debt >= 1 and not (cfg.use_residual or cfg.explore_rule == "learned_only"):
                self.rl_debt = 0.0
            n += 1
        self.version += 1
        self._usage.updates += n
        self._usage.train_gpu_s += time.perf_counter() - t0
        self.stats = {**info, "forget": float(self.model.forget), "beta": float(self.model.beta)}
        return {"updates": n, **self.stats}

    def _sample_cuts(self, eps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        cuts, qm = [], []
        T = min(T_MAX, max(len(e["x"]) for e in eps))
        for e in eps:
            n = len(e["x"])
            if self.rng.random() < 0.75:
                k = int(self.rng.choice(cfg.meta_ks))
            else:
                k = int(self.rng.integers(0, max(1, n - 8)))
            k = min(k, n - 4)
            q = np.zeros(T, bool)
            q[k:min(n, k + cfg.meta_query)] = True
            cuts.append(k)
            qm.append(q)
        return np.array(cuts), np.stack(qm)

    def _meta_step(self):
        cfg = self.cfg
        eps = [self.replay[int(i)] for i in self.rng.integers(len(self.replay), size=cfg.meta_batch)]
        H = history_batch(eps, cfg, self.device, self.rng)
        cuts, qm = self._sample_cuts(eps)
        cut = torch.from_numpy(cuts).to(self.device)
        qmask = torch.from_numpy(qm).to(self.device)
        meta, info, (lever_h, gauge_h, glob_h) = self.model.meta_loss(H, cut, qmask)
        world = self.model.world_loss(H, cut, gauge_h, glob_h)
        info["world"] = float(world.detach())
        return meta + cfg.world_coef * world, info

    def _rl_step(self):
        """Clipped importance-weighted actor-critic on decisions from the most recent episodes."""
        cfg = self.cfg
        recent = [e for e in self.replay[-cfg.recent_episodes:] if e["decisions"]]
        if not recent:
            return None, {}
        rows = []                                   # (episode, decision index, advantage, return)
        for e in recent:
            dec = e["decisions"]
            net = e["out"][:, 0] - e["out"][:, 1]
            rew = []
            for j, d in enumerate(dec):
                end = dec[j + 1]["row"] if j + 1 < len(dec) else len(net)
                scale = max(1e-3, float(np.abs(e["out"][:d["row"]]).sum(-1).mean())) if d["row"] > 0 else 1.0
                rew.append(float(net[d["row"]:end].sum()) / scale / max(1, cfg.dwell))
            vals = np.array([d["value"] for d in dec] + [0.0])
            mu, var = self.ret_stats[0], self.ret_stats[1]
            vals_raw = vals * math.sqrt(var) + mu
            vals_raw[-1] = 0.0
            adv = np.zeros(len(dec))
            g = 0.0
            for j in reversed(range(len(dec))):
                delta = rew[j] + 0.98 * vals_raw[j + 1] - vals_raw[j]
                g = delta + 0.98 * cfg.gae_lambda * g
                adv[j] = g
            ret = adv + vals_raw[:-1]
            for j in range(len(dec)):
                rows.append((e, j, adv[j], ret[j]))
        pick = self.rng.choice(len(rows), size=min(cfg.rl_batch, len(rows)), replace=False)
        batch = [rows[int(i)] for i in pick]
        rets = np.array([b[3] for b in batch])
        # running normalisation of returns for the value head
        c = self.ret_stats[2] + len(rets)
        m_new = self.ret_stats[0] + (rets.mean() - self.ret_stats[0]) * len(rets) / c
        v_new = (self.ret_stats[1] * self.ret_stats[2] + rets.var() * len(rets)) / c
        self.ret_stats = np.array([m_new, max(v_new, 1e-6), min(c, 1e5)])
        eps = [b[0] for b in batch]
        decs = [b[0]["decisions"][b[1]] for b in batch]
        H = history_batch(eps, cfg, self.device, self.rng)
        dev = self.device
        cut = torch.tensor([d["row"] for d in decs], device=dev)
        T = H.x.shape[1]
        keep = (torch.arange(T, device=dev)[None] >= torch.tensor([d["keep_from"] for d in decs], device=dev)[:, None]).float()
        Cmax = max(len(d["valid"]) for d in decs)
        L = H.x.shape[2]
        cand = np.zeros((len(decs), Cmax, L), np.float32)
        valid = np.zeros((len(decs), Cmax), bool)
        flags = np.zeros((len(decs), Cmax, N_CAND_FLAGS), np.float32)
        for i, d in enumerate(decs):
            c = len(d["valid"])
            cand[i, :c], valid[i, :c], flags[i, :c] = d["cand"], d["valid"], d["flags"]
        t = lambda a: torch.from_numpy(a).to(dev)
        n_rem = torch.tensor([d["n_rem"] for d in decs], device=dev)
        innov = torch.tensor([d["innov"] for d in decs], device=dev, dtype=torch.float32)
        out = self.model.decide(H, cut, t(cand), t(valid), t(flags), n_rem, keep, innov)
        logp_all = torch.log_softmax(out["logits"] / max(cfg.policy_temp, 1e-6), -1)
        choice = torch.tensor([d["choice"] for d in decs], device=dev)
        logp = logp_all.gather(1, choice[:, None]).squeeze(1)
        old = torch.tensor([d["logp"] for d in decs], device=dev)
        adv = torch.tensor([b[2] for b in batch], device=dev, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        ratio = torch.exp((logp - old).clamp(-10, 10))
        pol = -torch.min(ratio * adv, ratio.clamp(1 - cfg.rl_clip, 1 + cfg.rl_clip) * adv).mean()
        p = logp_all.exp()
        ent = -(p * torch.where(t(valid), logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
        target = torch.tensor((rets - self.ret_stats[0]) / math.sqrt(self.ret_stats[1]), device=dev, dtype=torch.float32)
        vloss = torch.nn.functional.smooth_l1_loss(out["value"], target)
        loss = pol + cfg.value_coef * vloss - cfg.entropy_coef * ent
        return loss, {"policy": float(pol), "value_loss": float(vloss), "entropy": float(ent),
                      "ratio_dev": float((ratio - 1).abs().mean())}

    # ----------------------------------------------------------------------------------------- checkpoints
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "cfg": self.cfg.to_dict(),
                    "version": self.version, "ret_stats": self.ret_stats, "usage_updates": self._usage.updates,
                    "rng": self.rng.bit_generator.state}, path)
        # persistent experience beside the weights (versioned episode records, not episodic context)
        buf = io.BytesIO()
        np.save(buf, np.array(self.replay, dtype=object), allow_pickle=True)
        with open(path + ".replay.npy", "wb") as fh:
            fh.write(buf.getvalue())

    def load(self, path: str) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.version, self.ret_stats = st["version"], np.asarray(st["ret_stats"])
        self.rng.bit_generator.state = st["rng"]
        rp = path + ".replay.npy"
        if os.path.exists(rp):
            self.replay = list(np.load(rp, allow_pickle=True))


def make(spec=None, seed: int = 0, device: str = "cuda", scale: str = "base", **kw) -> Cam7Learner:
    return Cam7Learner(spec or EnvSpec(), seed, device, None, scale, **kw)
