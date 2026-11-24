"""The learning update with explicit gradient ownership.

    world loss     -> encoders, recurrent core, memory, prediction heads
    critic loss    -> value and Q heads (belief detached unless the ``shared_policy_gradients`` ablation is on)
    actor loss     -> candidate scorer (same rule)
    generator loss -> program decoder

The actor objective is the V-trace policy gradient on replayed actions (corrected for the behaviour policy with its
stored log-probability). When held-out reward prediction clears ``imagination_min_r2`` it is augmented by a one-step
all-candidate expected advantage from the world model, penalised by ensemble disagreement. Executed programs are never
differentiated through: they only ever contribute a log-probability multiplied by a stopped advantage.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import UnifiedConfig
from ..dist import balanced_kl, categorical_entropy, categorical_kl, symlog, twohot_loss, twohot_mean
from ..interfaces import ActionCandidateBatch, BeliefState, ObservationBatch, ReplaySequence
from ..models.unified import EFFECT_SCALE, UnifiedModel
from .returns import ReturnNormalizer, duration_discount, vtrace


def _mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _stack(states: list[BeliefState]) -> BeliefState:
    return BeliefState(*(torch.stack([getattr(s, f) for s in states], 1)
                         for f in ("h", "z", "memory", "read", "u", "prior_logits", "post_logits")))


def _cut_gradient(s: BeliefState, rows: Tensor) -> BeliefState:
    def f(x: Tensor) -> Tensor:
        m = rows.reshape(-1, *([1] * (x.dim() - 1)))
        return torch.where(m, x.detach(), x)
    return s.map(f)


class UnifiedTrainer:
    def __init__(self, model: UnifiedModel, cfg: UnifiedConfig, tick_seconds: float = 3600.0):
        self.model, self.cfg, self.tick_seconds = model, cfg, tick_seconds
        g = model.param_groups()
        wd = cfg.weight_decay
        self.opt = {"world": torch.optim.AdamW(g["world"], lr=cfg.lr_world, weight_decay=wd),
                    "actor": torch.optim.AdamW(g["actor"], lr=cfg.lr_actor, weight_decay=wd),
                    "critic": torch.optim.AdamW(g["critic"], lr=cfg.lr_critic, weight_decay=wd),
                    "generator": torch.optim.AdamW(g["generator"], lr=cfg.generator_lr, weight_decay=wd)}
        self.groups = g
        self.norm = ReturnNormalizer(cfg.return_norm_rate, cfg.return_norm_limit)
        self.steps = 0
        self.heldout_r2 = float("nan")
        self.effect_r2 = float("nan")
        self.adapt_r2 = float("nan")
        self.opt["adapt"] = torch.optim.Adam(g["adapt"], lr=cfg.adapt_lr)
        self.train_seconds = 0.0

    # ------------------------------------------------------------------ belief unroll
    def _action_inputs(self, b: ReplaySequence) -> tuple[Tensor, Tensor, Tensor]:
        """Embedding of the chosen program at each step, and the previous action and elapsed time at each step."""
        B, T = b.action.shape
        idx = b.action.reshape(B, T, 1, 1)
        tok = b.cand_tokens.gather(2, idx.expand(-1, -1, 1, b.cand_tokens.shape[-1])).squeeze(2)
        fea = b.cand_features.gather(2, idx.expand(-1, -1, 1, b.cand_features.shape[-1])).squeeze(2)
        chosen = self.model.action_encoder(tok, fea)
        prev = torch.cat([torch.zeros_like(chosen[:, :1]), chosen[:, :-1]], 1)
        prev_dt = torch.cat([torch.zeros_like(b.dt[:, :1]), b.dt[:, :-1]], 1)
        keep = 1.0 - b.first
        return chosen, prev * keep.unsqueeze(-1), prev_dt * keep

    def _encode(self, b: ReplaySequence, start: int, end: int, target: bool = False) -> Tensor:
        B = b.values.shape[0]
        o = ObservationBatch(*(x[:, start:end].flatten(0, 1) for x in (b.values, b.deltas, b.present, b.ids)),
                             extra=b.extra[:, start:end].flatten(0, 1))
        enc = self.model.obs_target if target else self.model.obs_encoder
        _, pooled = enc(o)
        return pooled.reshape(B, end - start, -1)

    def unroll(self, b: ReplaySequence, burn: int):
        """Unroll every step. Rows that begin mid-episode use the first ``burn`` steps only to rebuild state: their
        gradient is cut at ``burn`` and ``training_mask`` removes those steps from the losses. Rows that begin at the
        episode start have a true initial state, so all of their steps train (including the learned initial state)."""
        m, T = self.model, b.action.shape[1]
        goal = m.goal_encoder.net(b.goal[:, 0])
        chosen, prev_a, prev_dt = self._action_inputs(b)
        if self.cfg.core == "attention":
            return self._unroll_events(b, goal, chosen, prev_a, prev_dt)
        state = m.rssm.initial(b.action.shape[0], b.action.device)
        emb = self._encode(b, 0, T)
        started = b.first[:, 0].bool()
        out = []
        for t in range(T):
            if t == burn and burn > 0:
                state = _cut_gradient(state, ~started)
            state = m.rssm.reset_where(state, b.first[:, t])
            state = m.rssm.observe(state, prev_a[:, t], prev_dt[:, t], emb[:, t], goal)
            out.append(state)
        return _stack(out), goal, chosen, emb

    def _unroll_events(self, b: ReplaySequence, goal: Tensor, chosen: Tensor, prev_a: Tensor, prev_dt: Tensor):
        """Attention core: every step of the window in one causal pass. Replay windows never span two episodes."""
        m, c = self.model, self.cfg
        B, T = b.action.shape
        emb = self._encode(b, 0, T)
        u = m.events(m.events.event(emb, prev_a, prev_dt), b.mask > 0, goal)
        z = torch.zeros(B, T, c.stoch_vars, c.stoch_classes, device=u.device)
        return (BeliefState(u, z, torch.zeros(B, T, 1, c.memory_dim, device=u.device),
                            torch.zeros(B, T, c.memory_dim, device=u.device), u, z, z), goal, chosen, emb)

    @staticmethod
    def training_mask(b: ReplaySequence, burn: int) -> Tensor:
        """1 where a step may contribute to a loss."""
        T = b.action.shape[1]
        after = (torch.arange(T, device=b.mask.device) >= burn).float().expand(b.mask.shape[0], -1)
        started = b.first[:, 0].unsqueeze(1)
        return b.mask * torch.maximum(after, started)

    # ------------------------------------------------------------------ losses
    def world_loss(self, b: ReplaySequence, s: BeliefState, chosen: Tensor, burn: int, goal: Tensor):
        m, c = self.model, self.cfg
        mask = b.mask[:, burn:]
        first = b.first[:, burn:]
        prev_rew = torch.cat([torch.zeros_like(b.reward[:, :1]), b.reward[:, :-1]], 1)[:, burn:]
        prev_cont = torch.cat([torch.ones_like(b.cont[:, :1]), b.cont[:, :-1]], 1)[:, burn:]
        trans = mask * (1 - first)

        # Observation prediction: readings in their transformed (symlog) units
        pred = m.decode_observation(s.u)
        pres = b.present[:, burn:].float() * mask.unsqueeze(-1)
        l_num = ((pred - b.values[:, burn:]) ** 2 * pres).sum() / pres.sum().clamp(min=1.0)

        # Latent prediction: the pre-observation state predicts a slow encoder of the observation
        with torch.no_grad():
            tgt = F.normalize(self._encode(b, burn, b.action.shape[1], target=True), dim=-1)
        before = s.h if c.core == "rssm" else torch.cat([torch.zeros_like(s.u[:, :1]), s.u[:, :-1]], 1)
        pl = F.normalize(m.latent_pred(before), dim=-1)
        l_lat = _mean(2 - 2 * (pl * tgt).sum(-1), trans)

        l_rew = _mean(twohot_loss(m.reward_head(s.u), prev_rew, m.bins), trans)
        # Each ensemble head sees its own bootstrap of the transitions.
        l_ens = s.u.new_zeros(())
        for head in m.reward_ensemble:
            boot = (torch.rand_like(trans) < 0.8).float() * trans
            l_ens = l_ens + _mean(twohot_loss(head(s.u.detach()), prev_rew, m.bins), boot)
        l_con = _mean(F.binary_cross_entropy_with_logits(m.continue_head(s.u).squeeze(-1), prev_cont, reduction="none"),
                      mask)

        dyn, rep = balanced_kl(s.post_logits, s.prior_logits, c.free_nats)
        l_dyn, l_rep = _mean(dyn, mask), _mean(rep, mask)

        # Multi-horizon: cumulative reward of the next k decisions after taking the chosen action
        l_multi = s.u.new_zeros(())
        rew = b.reward[:, burn:]
        cum = rew.cumsum(1)
        for head, k in zip(m.multi_head, c.multi_horizons):
            T = rew.shape[1]
            if T <= k:
                continue
            tgt_k = cum[:, k - 1:] - torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-k]], 1)
            valid = mask[:, k - 1:] * mask[:, :T - k + 1]
            logits = head(torch.cat([s.u[:, :T - k + 1], chosen[:, :T - k + 1]], -1))
            l_multi = l_multi + _mean(twohot_loss(logits, tgt_k, m.bins), valid)

        # Open-loop consistency: roll the prior forward from a posterior, compare with later posteriors
        l_over = s.u.new_zeros(())
        T = s.u.shape[1]
        if c.overshoot > 0 and c.core == "rssm" and T > c.overshoot + 1:
            at = int(torch.randint(0, T - c.overshoot, (1,)).item())
            cur = BeliefState(s.h[:, at], s.z[:, at], s.memory[:, at], s.read[:, at], s.u[:, at], s.prior_logits[:, at],
                              s.post_logits[:, at])
            for k in range(1, c.overshoot + 1):
                t = at + k
                alive = mask[:, t] * (1 - first[:, at + 1:t + 1].max(1).values)
                cur = m.rssm.imagine(cur, chosen[:, t - 1], b.dt[:, burn + t - 1], goal)
                kl = categorical_kl(s.post_logits[:, t].detach(), cur.prior_logits)
                l_over = l_over + _mean(kl.clamp(min=c.free_nats), alive) / c.overshoot

        # Consequence of the chosen program: reward per tick after it minus reward per tick before it
        delta, dmask = self.effect_targets(b, burn)
        l_eff = s.u.new_zeros(())
        for head in m.effect_heads:
            boot = (torch.rand_like(dmask) < 0.8).float() * dmask
            l_eff = l_eff + _mean(twohot_loss(head(torch.cat([s.u, chosen], -1)), delta * EFFECT_SCALE, m.bins), boot)

        total = l_num + l_lat + l_rew + l_ens + l_con + l_multi + c.beta_dyn * l_dyn + c.beta_rep * l_rep \
            + c.overshoot_weight * l_over + c.effect_weight * l_eff
        return total, {"wm/obs": l_num, "wm/latent": l_lat, "wm/reward": l_rew, "wm/continue": l_con, "wm/dyn": l_dyn,
                       "wm/rep": l_rep, "wm/multi": l_multi, "wm/overshoot": l_over, "wm/effect": l_eff,
                       "wm/post_ent": _mean(categorical_entropy(s.post_logits), mask),
                       "wm/raw_kl": _mean(categorical_kl(s.post_logits, s.prior_logits), mask)}

    @staticmethod
    def effect_targets(b: ReplaySequence, burn: int) -> tuple[Tensor, Tensor]:
        """``(delta, mask)`` aligned with decisions. Needs a preceding decision in the same episode and a real outcome."""
        dt, rew, mask, first = b.dt[:, burn:], b.reward[:, burn:], b.mask[:, burn:], b.first[:, burn:]
        rate = rew / dt.clamp(min=1.0)
        before = torch.cat([torch.zeros_like(rate[:, :1]), rate[:, :-1]], 1)
        had = torch.cat([torch.zeros_like(dt[:, :1]), (dt[:, :-1] > 0).float() * mask[:, :-1]], 1)
        delta = (rate - before) / b.rscale[:, burn:].clamp(min=1e-3)
        return delta.clamp(-20, 20), mask * (1 - first) * (dt > 0).float() * had

    def control_loss(self, b: ReplaySequence, s: BeliefState, goal: Tensor, burn: int):
        m, c = self.model, self.cfg
        u = s.u if c.shared_policy_gradients else s.u.detach()
        g = goal if c.shared_policy_gradients else goal.detach()
        B, T = u.shape[:2]
        mask = b.mask[:, burn:]
        first = b.first[:, burn:]
        prev_cont = torch.cat([torch.ones_like(b.cont[:, :1]), b.cont[:, :-1]], 1)[:, burn:]
        act_mask = mask * (1 - (1 - first) * (1 - prev_cont))   # a terminal row has no decision

        toks, feats = b.cand_tokens[:, burn:], b.cand_features[:, burn:]
        K = toks.shape[2]
        cmask = b.cand_mask[:, burn:]
        with torch.no_grad():
            e = m.action_encoder(toks.flatten(0, 1), feats.flatten(0, 1))
        uf, gf = u.flatten(0, 1), g.unsqueeze(1).expand(-1, T, -1).flatten(0, 1)
        logp = m.score_candidates(uf, gf, e, cmask.flatten(0, 1), feats.flatten(0, 1),
                                  b.cand_bonus[:, burn:].flatten(0, 1)).reshape(B, T, K)
        act = b.action[:, burn:]
        logp_a = logp.gather(2, act.unsqueeze(-1)).squeeze(-1)

        rew, disc_dt, cont = b.reward[:, burn:], b.dt[:, burn:], b.cont[:, burn:]
        gam = duration_discount(disc_dt * self.tick_seconds, c.gamma_day, c.seconds_per_day)
        with torch.no_grad():
            v = m.estimate_value(u.detach(), g.detach().unsqueeze(1).expand(-1, T, -1), target=True)
            ratio = (logp_a.detach() - b.behavior_log_prob[:, burn:]).clamp(-10, 10) * act_mask
            vs, rho = vtrace(rew[:, :-1], v, gam[:, :-1], cont[:, :-1], ratio[:, :-1], c.lam)
            step_mask = act_mask[:, :-1]
            pg_target = rew[:, :-1] + gam[:, :-1] * cont[:, :-1] * vs[:, 1:]
            adv = pg_target - v[:, :-1]
            fam_scale = torch.ones(B, device=u.device)
            for fam in set(b.family):
                rows = torch.tensor([x == fam for x in b.family], device=u.device)
                r_vals = vs[:, :-1][rows][step_mask[rows] > 0]
                if r_vals.numel() >= 4:
                    fam_scale[rows] = self.norm.update(r_vals, fam)
            scale = fam_scale.unsqueeze(1)

        # Critic
        v_logits = m.value(torch.cat([u.detach() if not c.shared_policy_gradients else u,
                                      g.unsqueeze(1).expand(-1, T, -1)], -1))
        l_value = _mean(twohot_loss(v_logits[:, :-1], vs[:, :-1].detach(), m.bins), step_mask)
        idx = act.reshape(B, T, 1, 1).expand(-1, -1, 1, e.shape[-1])
        e_seq = e.reshape(B, T, K, -1)
        ea = e_seq.gather(2, idx).squeeze(2)
        q_logits = m.q_logits(u[:, :-1], ea[:, :-1], g.unsqueeze(1).expand(-1, T - 1, -1))
        q2_logits = m.q2(torch.cat([u[:, :-1], ea[:, :-1], g.unsqueeze(1).expand(-1, T - 1, -1)], -1))
        l_q = _mean(twohot_loss(q_logits, pg_target.detach(), m.bins) + twohot_loss(q2_logits, pg_target.detach(), m.bins),
                    step_mask)

        # Actor: V-trace policy gradient plus entropy
        probs = logp.exp()
        ent = -(probs * logp.clamp(min=-30)).sum(-1)
        q_spread = u.new_zeros(())
        if c.actor_objective == "expected":
            # grad J = E_s sum_a grad pi(a|s) Q(s,a). Every proposed candidate contributes, so no sampled-action
            # variance and no behaviour correction for the action; replayed states are the usual off-policy approximation.
            with torch.no_grad():
                qa = m.q_pessimistic(uf.unsqueeze(1).expand(-1, K, -1), e.reshape(B * T, K, -1),
                                     gf.unsqueeze(1).expand(-1, K, -1)).reshape(B, T, K)
                base = (probs.detach() * qa).sum(-1, keepdim=True)
                adv_all = (qa - base) / scale.unsqueeze(-1)
                q_spread = _mean(qa.std(-1), act_mask)
            l_pg = _mean(-(probs * adv_all).sum(-1), act_mask)
        elif c.actor_objective == "vtrace":
            l_pg = _mean(-(rho * adv / scale).detach() * logp_a[:, :-1], step_mask)
        else:
            raise ValueError(f"unknown actor objective {c.actor_objective}")
        l_ent = -c.entropy_coef * _mean(ent, act_mask)

        l_imag, imag_info = u.new_zeros(()), {}
        if (c.use_imagination and c.core == "rssm" and self.heldout_r2 == self.heldout_r2
                and self.heldout_r2 >= c.imagination_min_r2):
            l_imag, imag_info = self._imagined_policy_loss(s, g, e_seq, cmask, b, burn, logp, act_mask, scale)
        l_actor = l_pg + l_ent + c.imagination_weight * l_imag
        info = {"ac/value": l_value, "ac/q": l_q, "ac/pg": l_pg, "ac/entropy": _mean(ent, act_mask),
                "ac/adv_abs": _mean(adv.abs(), step_mask), "ac/rho": _mean(rho, step_mask), "ac/imag": l_imag,
                "ac/v_mean": _mean(v[:, :-1], step_mask), "ac/q_spread": q_spread, **imag_info}
        return l_actor, l_value + l_q, info, u, g, adv, step_mask, scale

    def _imagined_policy_loss(self, s, g, e_seq, cmask, b, burn, logp, act_mask, scale):
        """All-candidate one-step lookahead: candidates are valid in the real state they were proposed in."""
        m, c = self.model, self.cfg
        B, T, K = e_seq.shape[:3]
        rows = act_mask.bool()
        with torch.no_grad():
            u0, n_rows = s.u[rows], int(rows.sum())
            if n_rows == 0:
                return s.u.new_zeros(()), {}
            pick = lambda x: x[rows].repeat_interleave(K, 0)
            prev = BeliefState(pick(s.h), pick(s.z), pick(s.memory), pick(s.read), pick(s.u), pick(s.prior_logits),
                               pick(s.post_logits))
            gr = g.unsqueeze(1).expand(-1, T, -1)[rows]
            dur = b.cand_dur[:, burn:][rows].flatten()
            nxt, out = m.predict_future(prev, e_seq[rows].flatten(0, 1), dur, gr.repeat_interleave(K, 0))
            gam = duration_discount(dur * self.tick_seconds, c.gamma_day, c.seconds_per_day)
            v_next = m.estimate_value(nxt.u, gr.repeat_interleave(K, 0), target=True)
            pred = out["reward"] - c.kappa * out["reward_std"] + gam * out["continue"] * v_next
            pred = pred.reshape(n_rows, K)
            base = m.estimate_value(u0, gr, target=True).unsqueeze(-1)
            sc = scale.expand(B, T)[rows].unsqueeze(-1)
            adv = (pred - base) / sc
        p = logp[rows].exp()
        return -(p * adv).sum(-1).mean(), {"imag/adv_spread": adv.std(-1).mean(), "imag/unc": out["reward_std"].mean()}

    def generator_loss(self, b: ReplaySequence, u: Tensor, g: Tensor, adv: Tensor, mask: Tensor, scale: Tensor, burn: int):
        """Advantage-weighted program likelihood (AWR): an offline method; it does not replace online RL."""
        m, c = self.model, self.cfg
        rows = mask.bool()
        if not rows.any():
            return u.new_zeros(()), {}
        B, T = b.action.shape[0], u.shape[1]
        act = b.action[:, burn:][:, :-1]
        idx = act.reshape(B, T - 1, 1, 1).expand(-1, -1, 1, b.cand_tokens.shape[-1])
        tok = b.cand_tokens[:, burn:][:, :-1].gather(2, idx).squeeze(2)[rows]
        lever = b.values.new_ones(tok.shape[0], m.lang.n_levers)
        lever = b.present[:, burn:][:, :-1][rows][:, m.layout.lever_id0:m.layout.lever_id0 + m.lang.n_levers].float()
        w = ((adv / scale)[rows] / c.awr_temperature).clamp(max=10).exp().clamp(max=c.awr_max_weight)
        gu = u[:, :-1][rows].detach()
        gg = g.unsqueeze(1).expand(-1, T - 1, -1)[rows].detach()
        lp = m.generator.log_prob(gu, gg, tok, lever)
        return -(w.detach() * lp).sum() / w.sum().clamp(min=1e-6), {"gen/logp": lp.mean(), "gen/weight": w.mean()}

    # ------------------------------------------------------------------ one gradient step
    def update(self, batch: ReplaySequence) -> dict:
        t0 = time.perf_counter()
        c, m = self.cfg, self.model
        dev = next(m.parameters()).device
        b = batch.to(dev)
        burn = 0 if c.core == "attention" else min(c.burn_in, b.action.shape[1] - 2)
        s, goal, chosen, _ = self.unroll(b, burn)
        b.mask = self.training_mask(b, burn)     # the losses below index from step 0 and rely on this mask
        burn = 0
        info: dict = {}

        wm, wi = self.world_loss(b, s, chosen, burn, goal)
        info.update(wi)
        actor, critic, ci, u, g, adv, step_mask, scale = self.control_loss(b, s, goal, burn)
        info.update(ci)

        for o in self.opt.values():
            o.zero_grad(set_to_none=True)
        if c.shared_policy_gradients:
            (wm * float(c.use_world_model_loss) + actor + critic).backward()
        else:
            if c.use_world_model_loss:
                wm.backward(retain_graph=False)
            (actor + critic).backward()
        for name in ("world", "actor", "critic"):
            info[f"grad/{name}"] = torch.nn.utils.clip_grad_norm_(self.groups[name], c.grad_clip)
            if name != "world" or c.use_world_model_loss or c.shared_policy_gradients:
                self.opt[name].step()

        if c.use_generator and c.train_generator:
            lg, gi = self.generator_loss(b, u.detach(), g.detach(), adv.detach(), step_mask, scale.detach(), burn)
            info.update(gi)
            self.opt["generator"].zero_grad(set_to_none=True)
            if lg.requires_grad:
                lg.backward()
                info["grad/generator"] = torch.nn.utils.clip_grad_norm_(self.groups["generator"], c.grad_clip)
                self.opt["generator"].step()
            info["gen/loss"] = lg

        m.update_targets()
        self.steps += 1
        self.train_seconds += time.perf_counter() - t0
        return {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in info.items()}

    # ------------------------------------------------------------------ within-episode regression
    def update_adapt(self, episodes: dict | None) -> dict:
        """Meta-train the regression's features and priors: predictive likelihood of each reward from earlier pairs."""
        if episodes is None:
            return {}
        dev = next(self.model.parameters()).device
        ep = {k: v.to(dev) for k, v in episodes.items()}
        loss, info = self.model.adapt.prefix_nll(**ep)
        self.opt["adapt"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.groups["adapt"], self.cfg.grad_clip)
        self.opt["adapt"].step()
        return {k: float(v) for k, v in info.items()}

    @torch.no_grad()
    def heldout_adapt(self, episodes: dict | None) -> dict:
        """Held-out episodes only. Sets the gate for using the regression at decision time."""
        if episodes is None:
            return {}
        dev = next(self.model.parameters()).device
        _, info = self.model.adapt.prefix_nll(**{k: v.to(dev) for k, v in episodes.items()})
        self.adapt_r2 = float(info["adapt/r2_vs_mean"])
        return {"heldout/adapt_r2": self.adapt_r2, "heldout/adapt_nll": float(info["adapt/nll"]),
                "heldout/adapt_nll_mean_only": float(info["adapt/nll_mean_only"])}

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def heldout_prediction(self, batch: ReplaySequence | None) -> dict:
        """Reward-prediction quality on held-out episodes, one step ahead from the prior. Sets the imagination gate."""
        if batch is None:
            return {}
        c, m = self.cfg, self.model
        dev = next(m.parameters()).device
        b = batch.to(dev)
        burn = min(c.burn_in, b.action.shape[1] - 2)
        if c.core == "attention":
            burn = 0
        s, goal, chosen, _ = self.unroll(b, burn)
        b.mask = self.training_mask(b, burn)
        burn = 0
        T = s.u.shape[1]
        rows = (b.mask[:, burn:][:, 1:] * (1 - b.first[:, burn:][:, 1:])).bool()
        if rows.sum() < 8:
            return {}
        r2, corr = float("nan"), float("nan")
        if c.core == "rssm":         # the one-step forecast exists only for the RSSM's prior
            prev = BeliefState(s.h[:, :-1].flatten(0, 1), s.z[:, :-1].flatten(0, 1), s.memory[:, :-1].flatten(0, 1),
                               s.read[:, :-1].flatten(0, 1), s.u[:, :-1].flatten(0, 1),
                               s.prior_logits[:, :-1].flatten(0, 1), s.post_logits[:, :-1].flatten(0, 1))
            g = goal.unsqueeze(1).expand(-1, T - 1, -1).flatten(0, 1)
            nxt, out = m.predict_future(prev, chosen[:, :-1].flatten(0, 1), b.dt[:, burn:][:, :-1].flatten(), g)
            pred = out["reward"].reshape(rows.shape)[rows]
            true = b.reward[:, burn:][:, :-1][rows]
            ss_res = ((symlog(true) - symlog(pred)) ** 2).sum()
            ss_tot = ((symlog(true) - symlog(true).mean()) ** 2).sum().clamp(min=1e-8)
            r2 = float(1 - ss_res / ss_tot)
            err = (true - pred).abs()
            unc = out["reward_std"].reshape(rows.shape)[rows]
            if err.numel() > 2 and unc.std() > 0:
                corr = float(torch.corrcoef(torch.stack([err, unc]))[0, 1])
            self.heldout_r2 = r2
        # Held-out test of the consequence model on programs that really changed something
        delta, dmask = self.effect_targets(b, burn)
        idx = b.action.reshape(*b.action.shape, 1, 1).expand(-1, -1, 1, b.cand_features.shape[-1])
        moved = b.cand_features.gather(2, idx).squeeze(2)[..., m.lang.move_index] > 1e-4
        use = (dmask > 0) & moved
        eff_r2 = float("nan")
        if use.sum() >= 16:
            p = torch.stack([twohot_mean(h(torch.cat([s.u, chosen], -1)), m.bins) for h in m.effect_heads]).mean(0) / EFFECT_SCALE
            t, q = delta[use], p[use]
            eff_r2 = float(1 - ((t - q) ** 2).sum() / ((t - t.mean()) ** 2).sum().clamp(min=1e-8))
            self.effect_r2 = eff_r2
        return {"heldout/reward_r2": r2, "heldout/unc_error_corr": corr, "heldout/effect_r2": eff_r2}

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "opt": {k: o.state_dict() for k, o in self.opt.items()},
                "norm": self.norm.state_dict(), "steps": self.steps, "heldout_r2": self.heldout_r2,
                "effect_r2": self.effect_r2, "adapt_r2": self.adapt_r2}

    def load_state_dict(self, st: dict) -> None:
        self.model.load_state_dict(st["model"])
        for k, o in self.opt.items():
            o.load_state_dict(st["opt"][k])
        self.norm.load_state_dict(st["norm"])
        self.steps, self.heldout_r2 = int(st["steps"]), float(st["heldout_r2"])
        self.effect_r2 = float(st.get("effect_r2", float("nan")))
        self.adapt_r2 = float(st.get("adapt_r2", float("nan")))
