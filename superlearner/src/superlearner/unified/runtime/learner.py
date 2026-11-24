"""CAM 6.1 behind Crucible's architecture-neutral ``Learner`` contract.

Crucible runs and scores it exactly as it does every baseline. The learner receives only the public observation
vector. Task records, private references and hidden mechanism state are never passed in; family labels may be given
to the replay buffer for balancing, and they never enter the network.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from crucible.research.learner import EnvSpec, Learner, Usage

from ..config import SCALES, UnifiedConfig
from ..dist import symlog
from ..encoders.programs import DEFAULT, encode, execute, features, parse, structured_candidates
from ..interfaces import ActionCandidateBatch, BeliefState, GoalBatch, ObservationBatch
from ..learning.replay import EpisodeReplay
from ..learning.trainer import UnifiedTrainer
from ..models.unified import ObsLayout, UnifiedModel


class UnifiedLearner(Learner):
    name = "cam61"
    kind = "model_based"

    def __init__(self, spec: EnvSpec | None = None, seed: int = 0, device: str = "cuda", cfg: UnifiedConfig | None = None,
                 scale: str = "small", tick_seconds: float = 3600.0, replay_capacity: int = 1500):
        spec = spec or EnvSpec()
        super().__init__(spec, seed, device)
        self.cfg = cfg or SCALES[scale]()
        torch.manual_seed(seed)
        self.layout = ObsLayout(spec.max_gauges, spec.max_levers, 5)
        self.lang = DEFAULT
        if self.lang.n_levers != spec.max_levers:
            raise ValueError("the program language must cover the environment's levers")
        self.model = UnifiedModel(self.cfg, self.layout, 4, self.lang).to(device)
        self.trainer = UnifiedTrainer(self.model, self.cfg, tick_seconds)
        self.replay = EpisodeReplay(replay_capacity, seed=seed, n_ids=self.layout.n_ids)
        self.rng = np.random.default_rng(seed)
        self.tick_seconds = tick_seconds
        self.gamma_tick = self.cfg.gamma_day ** (tick_seconds / self.cfg.seconds_per_day)
        self.ids = torch.arange(self.layout.n_ids, device=device)
        self.n = 0
        self.labels: list[str] = []
        self.pending = 0
        self.greedy_eval = True
        self.proposer = None          # optional callable(rng, lever_mask, k) -> list of token arrays
        self.metrics: dict = {}
        self._usage = Usage(params=sum(self.model.n_params().values()))

    # ------------------------------------------------------------------ context
    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if n != self.n or which is None:
            self.n = n
            # Owned copies: the initial state is an expanded view of parameters and must never be written in place.
            self.state = self.model.initial_belief(n, self.device).map(lambda x: x.detach().clone())
            self.prev_action = torch.zeros(n, self.cfg.action_dim, device=self.device)
            self.prev_dt = torch.zeros(n, device=self.device)
            self.prev_values = np.zeros((n, self.layout.n_ids), dtype=np.float32)
            self.queue: list[list[np.ndarray]] = [[] for _ in range(n)]
            self.fresh = np.ones(n, dtype=bool)
            self.ticks = np.zeros(n, dtype=np.int64)
            self.acc = np.zeros(n, dtype=np.float64)
            self.since_r = np.zeros(n, dtype=np.float64)
            self.since_n = np.zeros(n, dtype=np.int64)
            L = self.spec.max_levers
            self.x0 = np.zeros((n, L), dtype=np.float32)
            self.track_a = np.zeros((n, L), dtype=np.float32)
            self.track_m = np.zeros((n, 3), dtype=np.float32)
            self.trace = np.zeros((n, L, 3), dtype=np.float32)
            self.r_n, self.r_mean, self.r_m2 = np.zeros(n), np.zeros(n), np.zeros(n)
            self.adapt_A, self.adapt_b = self.model.adapt.empty(n, self.device)
            self.tick_buf: list[dict] = [{"x": [], "r": [], "sd": []} for _ in range(n)]
            self.event_buf: list[list[torch.Tensor]] = [[] for _ in range(n)]
            self.episodes: list[dict] = [self._new_ep() for _ in range(n)]
            if len(self.labels) != n:
                self.labels = ["default"] * n
            return
        idx = np.flatnonzero(np.asarray(which))
        if len(idx) == 0:
            return
        t = torch.as_tensor(idx, device=self.device)
        fresh = self.model.initial_belief(len(idx), self.device)
        with torch.no_grad():
            for f in ("h", "z", "memory", "read", "u", "prior_logits", "post_logits"):
                getattr(self.state, f)[t] = getattr(fresh, f).detach()
        self.prev_action[t] = 0.0
        self.prev_dt[t] = 0.0
        self.prev_values[idx] = 0.0
        self.adapt_A[t] = 0.0
        self.adapt_b[t] = 0.0
        for i in idx:
            self.queue[i] = []
            self.fresh[i] = True
            self.ticks[i] = 0
            self.acc[i] = 0.0
            self.since_r[i] = 0.0
            self.since_n[i] = 0
            self.trace[i] = 0.0
            self.r_n[i], self.r_mean[i], self.r_m2[i] = 0.0, 0.0, 0.0
            self.tick_buf[i] = {"x": [], "r": [], "sd": []}
            self.event_buf[i] = []
            self.episodes[i] = self._new_ep()

    @staticmethod
    def _new_ep() -> dict:
        return {k: [] for k in ("values", "deltas", "present", "extra", "goal", "cand_tokens", "cand_features", "cand_mask", "cand_dur", "cand_bonus", "rscale",
                                "action", "behavior_log_prob", "reward", "dt", "cont", "first")}

    def set_labels(self, labels: list[str]) -> None:
        self.labels = list(labels)

    # ------------------------------------------------------------------ observation
    def _track(self, obs: np.ndarray) -> None:
        """Per-lever intervention evidence, the same public statistic Crucible's baselines read: decayed sums of
        (change of setting) x (change of money), divided by the decayed squared change. Updated on every tick."""
        sp = self.spec
        a, sc = obs[:, sp.last_action], obs[:, sp.scalars][:, :3]
        first = self.fresh
        if first.any():
            self.x0[first], self.track_a[first] = a[first], a[first]
            self.track_m[first], self.trace[first] = 0.0, 0.0
            self.r_n[first], self.r_mean[first], self.r_m2[first] = 0.0, 0.0, 0.0
        da, dm = a - self.track_a, sc - self.track_m
        self.trace *= 0.97
        self.trace[:, :, 0] += da * dm[:, 0:1]
        self.trace[:, :, 1] += da * dm[:, 2:3]
        self.trace[:, :, 2] += da * da
        self.track_a, self.track_m = a.copy(), sc.copy()
        live = ~first                                   # the first observation carries no reward yet
        r = sc[:, 0].astype(np.float64)
        self.r_n[live] += 1
        d = r - self.r_mean
        self.r_mean[live] += d[live] / self.r_n[live]
        self.r_m2[live] += (d * (r - self.r_mean))[live]
        if self.cfg.use_adaptive and live.any():
            # Every tick is one (settings in force, reward) pair for the within-episode regression.
            dev, ad = self.device, self.model.adapt
            phi = ad.phi(torch.as_tensor(a, device=dev), torch.as_tensor(self.x0, device=dev),
                         torch.as_tensor(obs[:, sp.lever_mask], device=dev))
            self.adapt_A, self.adapt_b = ad.add(self.adapt_A, self.adapt_b, phi,
                                                torch.as_tensor(sc[:, 0], device=dev, dtype=phi.dtype),
                                                torch.as_tensor(live, device=dev))
        if not self.frozen:
            sd = self._reward_sd(np.arange(self.n))
            for i in np.flatnonzero(live):
                buf = self.tick_buf[i]
                buf["x"].append(a[i].copy()); buf["r"].append(float(sc[i, 0])); buf["sd"].append(float(sd[i]))

    @torch.no_grad()
    def _adaptive(self, obs: np.ndarray, idx: np.ndarray, finals: np.ndarray | None):
        """Predictions of the within-episode regression, in units of this episode's reward spread.
        Returns per-candidate features ``[m, K, 3]``, per-lever features ``[m, L, 2]`` and the sampled gains ``[m, K]``."""
        sp, dev, ad = self.spec, self.device, self.model.adapt
        m, L = len(idx), sp.max_levers
        K = 0 if finals is None else finals.shape[1]
        if not self.cfg.use_adaptive:
            return np.zeros((m, K, 3), np.float32), np.zeros((m, L, 2), np.float32), torch.zeros(m, K, device=dev)
        t = torch.as_tensor(idx, device=dev)
        A, b = self.adapt_A[t], self.adapt_b[t]
        cur = torch.as_tensor(obs[idx][:, sp.last_action], device=dev)
        x0 = torch.as_tensor(self.x0[idx], device=dev)
        mask = torch.as_tensor(obs[idx][:, sp.lever_mask], device=dev)
        s = torch.as_tensor(self._reward_sd(idx), device=dev, dtype=cur.dtype).unsqueeze(1)
        feats = lambda q: ad.phi(q, x0.unsqueeze(1).expand_as(q), mask.unsqueeze(1).expand_as(q))
        eye = torch.eye(L, device=dev) * 0.2
        q = torch.cat([cur.unsqueeze(1), (cur.unsqueeze(1) + eye).clamp(-1, 1), (cur.unsqueeze(1) - eye).clamp(-1, 1)], 1)
        if K:
            q = torch.cat([q, torch.as_tensor(finals, device=dev)], 1)
        phi = feats(q)
        mu, var = ad.predict(A, b, phi)
        w = ad.sample_weights(A, b)
        ts = (phi * w.unsqueeze(1)).sum(-1)
        slope = ((mu[:, 1:1 + L] - mu[:, 1 + L:1 + 2 * L]) / s).clamp(-5, 5) / 5 * mask
        unc = (0.5 * (var[:, 1:1 + L] + var[:, 1 + L:1 + 2 * L])).sqrt().clamp(max=5) / 5 * mask
        lever = torch.stack([slope, unc], -1).cpu().numpy()
        if not K:
            return np.zeros((m, 0, 3), np.float32), lever, torch.zeros(m, 0, device=dev)
        c0 = 1 + 2 * L
        gain = (mu[:, c0:] - mu[:, :1]) / s
        tg = (ts[:, c0:] - ts[:, :1]) / s
        cand = torch.stack([gain.clamp(-5, 5) / 5, var[:, c0:].sqrt().clamp(max=5) / 5, tg.clamp(-5, 5) / 5], -1)
        return cand.cpu().numpy(), lever, tg

    def _reward_sd(self, idx: np.ndarray) -> np.ndarray:
        """This episode's own reward variability so far: the public unit for how large an effect is."""
        return np.sqrt(self.r_m2[idx] / np.maximum(self.r_n[idx] - 1, 1)) + 1e-3

    def _evidence(self, idx: np.ndarray) -> np.ndarray:
        """``[len(idx), n_levers, 2]``: estimated slope of reward and of cost per lever, roughly in [-1, 1]."""
        tr = self.trace[idx]
        den = (tr[:, :, 2] + 1e-3) * self._reward_sd(idx)[:, None]
        return np.stack([np.clip(tr[:, :, 0] / den, -5, 5) / 5, np.clip(tr[:, :, 1] / den, -5, 5) / 5], -1).astype(np.float32)

    def _extra(self, obs: np.ndarray, idx: np.ndarray, lever_adapt: np.ndarray | None = None) -> np.ndarray:
        sp = self.spec
        out = np.zeros((len(idx), self.layout.n_ids, 6), dtype=np.float32)
        lo = self.layout.lever_id0
        ev = self._evidence(idx)
        mask = obs[idx][:, sp.lever_mask]
        out[:, lo:lo + sp.max_levers, 0] = (obs[idx][:, sp.last_action] - self.x0[idx]) * mask
        out[:, lo:lo + sp.max_levers, 1] = ev[:, :, 0] * mask
        out[:, lo:lo + sp.max_levers, 2] = ev[:, :, 1] * mask
        out[:, lo:lo + sp.max_levers, 3] = np.minimum(self.trace[idx][:, :, 2], 3.0) / 3.0 * mask
        if not self.cfg.use_hand_evidence:
            out[:, :, 1:4] = 0.0
        if lever_adapt is not None:
            out[:, lo:lo + sp.max_levers, 4:6] = lever_adapt
        return out

    def _tokens(self, obs: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sp = self.spec
        o = obs[idx]
        sc = o[:, sp.scalars].copy()
        # The reward reading summarises every tick since the last decision, so a long program loses no reward signal.
        mean_r = np.where(self.since_n[idx] > 0, self.since_r[idx] / np.maximum(self.since_n[idx], 1), sc[:, 0])
        sc[:, 0] = mean_r
        sc[:, :3] = np.sign(sc[:, :3]) * np.log1p(np.abs(sc[:, :3]))
        vals = np.concatenate([o[:, sp.gauges], o[:, sp.last_action], sc], 1).astype(np.float32)
        pres = np.concatenate([o[:, sp.gauge_mask] > 0.5, o[:, sp.lever_mask] > 0.5, np.ones_like(sc, dtype=bool)], 1)
        deltas = np.where(self.fresh[idx][:, None], 0.0, vals - self.prev_values[idx]).astype(np.float32)
        return vals, deltas, pres

    def _goal(self, obs: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Public goal: maximise net income. The numeric target is withheld because it encodes the private reference."""
        sp = self.spec
        o = obs[idx]
        g = np.zeros((len(idx), 4), dtype=np.float32)
        g[:, 0] = 1.0
        g[:, 1] = o[:, sp.lever_mask].mean(1)
        g[:, 2] = o[:, sp.gauge_mask].mean(1)
        return g

    def _propose(self, u: torch.Tensor, goal: torch.Tensor, obs: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, ...]:
        cfg, lang, sp = self.cfg, self.lang, self.spec
        k = cfg.candidates
        n_gen = int(k * cfg.generator_fraction) if cfg.use_generator else 0
        lever_mask = obs[idx][:, sp.lever_mask]
        settings = obs[idx][:, sp.last_action]
        evidence = self._evidence(idx)
        toks = np.zeros((len(idx), k, lang.max_len), dtype=np.int64)
        if n_gen:
            uu, gg = u.repeat_interleave(n_gen, 0), goal.repeat_interleave(n_gen, 0)
            lm = torch.as_tensor(lever_mask, device=self.device).repeat_interleave(n_gen, 0)
            gen, _ = self.model.generate_candidates(uu, gg, lm)
            toks[:, k - n_gen:] = gen.reshape(len(idx), n_gen, -1).cpu().numpy()
        fixed = k - n_gen
        for r in range(len(idx)):
            prop = self.proposer(self.rng, lever_mask[r], fixed) if self.proposer else \
                structured_candidates(self.rng, lever_mask[r], fixed, lang)
            toks[r, :fixed] = np.stack(prop[:fixed])
        fea = np.zeros((len(idx), k, lang.n_features + 3), dtype=np.float32)
        dur = np.zeros((len(idx), k), dtype=np.float32)
        scheds: list[list] = [[None] * k for _ in range(len(idx))]
        for r in range(len(idx)):
            for c in range(k):
                st = parse(toks[r, c], lang)
                f, ticks = features(st, settings[r], lever_mask[r], lang, evidence[r])
                fea[r, c, :lang.n_features], dur[r, c] = f, ticks
                if ticks:
                    scheds[r][c] = st
        mask = dur > 0
        n = lang.n_levers
        if not cfg.use_hand_evidence:
            fea[:, :, 2 * n + 4:2 * n + 6] = 0.0
        cand, _, tg = self._adaptive(obs, idx, fea[:, :, n:2 * n])
        fea[:, :, lang.n_features:] = cand * mask[..., None]
        self._last_sampled_gain = tg
        return toks, fea, mask, dur, scheds

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        out = obs[:, self.spec.last_action].copy()
        self._track(obs)
        # Counted here, not in observe(), so frozen evaluation sees exactly the inputs training saw.
        last_r = obs[:, self.spec.scalars][:, 0]
        for i in range(self.n):
            if not self.fresh[i]:
                self.since_r[i] += float(last_r[i])
                self.since_n[i] += 1
        need = np.array([i for i in range(self.n) if not self.queue[i]], dtype=np.int64)
        if len(need):
            self._decide(obs, need, explore)
        for i in range(self.n):
            if self.queue[i]:
                out[i] = self.queue[i].pop(0)
        self._usage.infer_gpu_s += time.perf_counter() - t0
        return out

    def _plan_bonus(self, u: torch.Tensor, e: torch.Tensor, mask: torch.Tensor, explore: bool) -> torch.Tensor:
        """Decision-time use of the consequence model, only after it passes the held-out check. One ensemble member
        per decision is followed (Thompson-style), so disagreement produces experiments rather than a fixed choice."""
        c = self.cfg
        bonus = torch.zeros_like(mask, dtype=u.dtype)
        ar2 = self.trainer.adapt_r2
        if c.use_adaptive and c.use_adaptive_planner and ar2 == ar2 and ar2 >= c.adapt_min_r2:
            # Thompson sampling from the within-episode posterior: uncertain candidates get tried in proportion to
            # the chance that they are better, which is how this learner chooses experiments.
            bonus = bonus + (c.adapt_beta * self._last_sampled_gain).clamp(-10, 10)
        r2 = self.trainer.effect_r2
        if c.use_effect_planner and r2 == r2 and r2 >= c.plan_min_r2:
            pred = self.model.effect_predictions(u, e)
            pick = torch.randint(pred.shape[0], (u.shape[0],), device=u.device)
            one = pred[pick, torch.arange(u.shape[0], device=u.device)]
            bonus = bonus + (c.plan_beta * (one - one[:, :1])).clamp(-10, 10)
        return bonus * mask

    def _event_belief(self, o: ObservationBatch, idx: np.ndarray, t: torch.Tensor, goal: torch.Tensor) -> BeliefState:
        """Append this decision's event to each episode's window and read the belief at its newest position. The window
        and positions match training, where replay windows are at most ``event_window`` decisions of one episode."""
        m, W = self.model, self.cfg.event_window
        _, pooled = m.encode_observation(o)
        ev = m.events.event(pooled, self.prev_action[t], self.prev_dt[t])
        lens = []
        for r, i in enumerate(idx):
            buf = self.event_buf[i]
            buf.append(ev[r])
            if len(buf) > W:
                del buf[0]
            lens.append(len(buf))
        win = torch.zeros(len(idx), max(lens), ev.shape[-1], device=ev.device)
        valid = torch.zeros(len(idx), max(lens), dtype=torch.bool, device=ev.device)
        for r, i in enumerate(idx):
            win[r, :lens[r]] = torch.stack(self.event_buf[i])
            valid[r, :lens[r]] = True
        u_all = m.events(win, valid, goal)
        u = u_all[torch.arange(len(idx), device=ev.device), torch.as_tensor(lens, device=ev.device) - 1]
        self.state.u[t] = u
        return BeliefState(u, u.new_zeros(len(idx), 1, 1), u.new_zeros(len(idx), 1, 1), u.new_zeros(len(idx), 1), u,
                           u.new_zeros(len(idx), 1, 1), None)

    def _decide(self, obs: np.ndarray, idx: np.ndarray, explore: bool) -> None:
        device, cfg = self.device, self.cfg
        vals, deltas, pres = self._tokens(obs, idx)
        _, lever_adapt, _ = self._adaptive(obs, idx, None)
        extra = self._extra(obs, idx, lever_adapt)
        goal_np = self._goal(obs, idx)
        t = torch.as_tensor(idx, device=device)
        o = ObservationBatch(torch.as_tensor(vals, device=device), torch.as_tensor(deltas, device=device),
                             torch.as_tensor(pres, device=device), self.ids.expand(len(idx), -1),
                             extra=torch.as_tensor(extra, device=device))
        goal = self.model.encode_goal(GoalBatch(torch.as_tensor(goal_np, device=device)))
        if cfg.core == "attention":
            belief = self._event_belief(o, idx, t, goal)
        else:
            prev = self.state.map(lambda x: x[t])
            belief = self.model.update_belief(prev, self.prev_action[t], o, self.prev_dt[t], goal, sample=explore)
            for f in ("h", "z", "memory", "read", "u", "prior_logits", "post_logits"):
                getattr(self.state, f)[t] = getattr(belief, f)
        toks, fea, mask, dur, scheds = self._propose(belief.u, goal, obs, idx)
        ct, cf = torch.as_tensor(toks, device=device), torch.as_tensor(fea, device=device)
        cm = torch.as_tensor(mask, device=device)
        e = self.model.action_encoder(ct, cf)
        bonus = self._plan_bonus(belief.u, e, cm, explore)
        logp = self.model.score_candidates(belief.u, goal, e, cm, cf, bonus)
        temp = 1.0 if explore else cfg.eval_temperature
        if temp <= 0:
            choice = logp.argmax(-1)
        else:
            choice = torch.multinomial(torch.softmax(logp.clamp(min=-1e4) / temp, -1), 1).squeeze(1)
        lp = logp.gather(1, choice.unsqueeze(1)).squeeze(1)
        self.prev_action[t] = e[torch.arange(len(idx), device=device), choice]
        choice_np = choice.cpu().numpy()
        settings = obs[idx][:, self.spec.last_action]
        lever_mask = obs[idx][:, self.spec.lever_mask]
        for r, i in enumerate(idx):
            sched = execute(scheds[r][int(choice_np[r])], settings[r], lever_mask[r], self.lang)
            self.queue[i] = list(sched)
            self.prev_dt[int(i)] = float(len(sched))
            self.prev_values[i] = vals[r]
            self.since_r[i], self.since_n[i] = 0.0, 0
            if not self.frozen:
                ep = self.episodes[i]
                ep["values"].append(vals[r]); ep["deltas"].append(deltas[r]); ep["present"].append(pres[r])
                ep["extra"].append(extra[r].astype(np.float16)); ep["goal"].append(goal_np[r]); ep["cand_tokens"].append(toks[r].astype(np.int16))
                ep["cand_features"].append(fea[r].astype(np.float16)); ep["cand_mask"].append(mask[r])
                ep["cand_dur"].append(dur[r]); ep["cand_bonus"].append(bonus[r].cpu().numpy().astype(np.float16))
                ep["rscale"].append(float(self._reward_sd(np.array([i]))[0] * cfg.reward_scale)); ep["action"].append(int(choice_np[r]))
                ep["behavior_log_prob"].append(float(lp[r])); ep["first"].append(float(self.fresh[i]))
            self.fresh[i] = False

    # ------------------------------------------------------------------ learning
    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen:
            return
        for i in range(self.n):
            ep = self.episodes[i]
            if len(ep["action"]) == len(ep["reward"]):       # slot already closed
                continue
            self.acc[i] += (self.gamma_tick ** self.ticks[i]) * float(reward[i]) * self.cfg.reward_scale
            self.ticks[i] += 1
            if done[i] or not self.queue[i]:
                ep["reward"].append(float(self.acc[i]))
                ep["dt"].append(float(self.ticks[i]))
                ep["cont"].append(0.0 if done[i] else 1.0)
                self.pending += int(1)
                self.acc[i], self.ticks[i] = 0.0, 0
                if done[i]:
                    self.queue[i] = []
                    self._close(i, next_obs)

    def _close(self, i: int, next_obs: np.ndarray) -> None:
        """Add the terminal row (its state is observed; no decision is taken there) and store the episode."""
        ep = self.episodes[i]
        idx = np.array([i])
        self.since_r[i] += float(next_obs[i, self.spec.scalars][0])
        self.since_n[i] += 1
        vals, deltas, pres = self._tokens(next_obs, idx)
        ep["values"].append(vals[0]); ep["deltas"].append(deltas[0]); ep["present"].append(pres[0])
        ep["extra"].append(self._extra(next_obs, idx)[0].astype(np.float16))
        ep["goal"].append(self._goal(next_obs, idx)[0])
        ep["cand_tokens"].append(np.zeros_like(ep["cand_tokens"][-1]))
        ep["cand_features"].append(np.zeros_like(ep["cand_features"][-1]))
        m = np.zeros_like(ep["cand_mask"][-1]); m[0] = True
        ep["cand_mask"].append(m)
        ep["cand_dur"].append(np.zeros_like(ep["cand_dur"][-1]))
        ep["cand_bonus"].append(np.zeros_like(ep["cand_bonus"][-1]))
        ep["rscale"].append(ep["rscale"][-1])
        buf = self.tick_buf[i]
        sp = self.spec
        buf["x"].append(next_obs[i, sp.last_action].copy()); buf["r"].append(float(next_obs[i, sp.scalars][0]))
        buf["sd"].append(buf["sd"][-1] if buf["sd"] else 1e-3)
        ep["tick_x"], ep["tick_r"], ep["tick_sd"] = np.stack(buf["x"]), np.array(buf["r"]), np.array(buf["sd"])
        ep["x0"], ep["lmask"] = self.x0[i].copy(), next_obs[i, sp.lever_mask].copy()
        self.tick_buf[i] = {"x": [], "r": [], "sd": []}
        ep["action"].append(0); ep["behavior_log_prob"].append(0.0)
        ep["reward"].append(0.0); ep["dt"].append(0.0); ep["cont"].append(0.0); ep["first"].append(0.0)
        self.replay.add(ep, self.labels[i])
        self.episodes[i] = self._new_ep()

    def update(self) -> dict:
        if self.frozen or len(self.replay) < 2:
            return {"updates": 0}
        n = int(self.pending * self.cfg.updates_per_step)
        if n == 0:
            return {"updates": 0}
        self.pending -= int(n / self.cfg.updates_per_step)
        t0 = time.perf_counter()
        info: dict = {}
        for _ in range(n):
            info = self.trainer.update(self.replay.sample(self.cfg.batch_size, self.cfg.seq_len))
            if self.cfg.use_adaptive:
                info.update(self.trainer.update_adapt(self.replay.sample_episodes(self.cfg.adapt_batch,
                                                                                  task_scale=self.cfg.adapt_task_scale)))
            if self.trainer.steps % 50 == 0:
                info.update(self.trainer.heldout_prediction(self.replay.sample_holdout(self.cfg.batch_size, self.cfg.seq_len)))
                if self.cfg.use_adaptive:
                    info.update(self.trainer.heldout_adapt(self.replay.sample_episodes(16, holdout=True)))
        self._usage.updates += n
        self._usage.train_gpu_s += time.perf_counter() - t0
        self.metrics = info
        return {"updates": n, **info}

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save({"trainer": self.trainer.state_dict(), "config": self.cfg.to_dict(), "seed": self.seed}, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.trainer.load_state_dict(st["trainer"])
