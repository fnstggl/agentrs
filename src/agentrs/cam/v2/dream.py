"""Dream learner: actor and critic updates from bounded imagined trajectories of the learned RSSM.

``imagine``: start from *real* posterior states sampled from replay (stop-gradient), let CAM's current
actor choose a complete structured decision (option termination, option, grammar-masked tokens),
advance the RSSM prior, and read the predicted contract reward, constraint-violation probability,
continuation and clock duration of the action from the world model. Repeat for up to ``horizon``
actions. The world model is frozen here (no gradients), so neither actor nor critic can change it.

``evaluate``: semi-Markov lambda-returns (discount gamma ** predicted dt, weighted by predicted
continuation), bootstrapped with the target imagination critic at the last trusted state.

``update``: the imagination critic regresses to the returns; the actor receives score-function
(REINFORCE) gradients, (R - V) * grad log pi(decision), with DreamerV3-style percentile return
normalisation and an entropy bonus — a valid estimator for discrete structured actions (no
reparameterisation). Adaptive mode: each trajectory stops at the first imagined step whose ensemble
disagreement exceeds a threshold calibrated on held-out real transitions, and the actor's
imagination loss is scaled by a reliability score measured against real outcomes (0 = ignored).

Imagined transitions are never written to replay and imagined successes are never counted.
"""
from __future__ import annotations

import numpy as np
import torch

from agentrs.cam.v2.learner import _grammar
from agentrs.cam.v2.model import CAM2


class Dreamer:
    def __init__(self, m: CAM2, cfg: dict):
        self.m, self.c = m, cfg
        self.scale = 1.0
        self.u_thr = float("inf") if cfg.get("mode", "fixed") == "fixed" else cfg.get("u_thr_init", 0.0)
        self.reliability = 0.0 if cfg.get("mode") == "adaptive" else 1.0
        self.max_h = cfg.get("horizon", 1)
        self.stats_hist: list = []

    @torch.no_grad()
    def _act(self, f, g, cur, first, fams):
        m = self.m
        logits, beta = m.actor_dists(f, g, cur)
        pb = torch.sigmoid(beta)
        term = torch.bernoulli(pb).long()
        new = first | term.bool()
        o_s = torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1)
        opt = torch.where(new, o_s, cur)
        toks, ids, _ = m.generate_action(f, g, opt, [_grammar(x) for x in fams])
        return term, new, opt, toks, ids

    def imagine(self, h, z, g, fams, prev_opt, first, lam, horizon: int) -> dict:
        """Roll out the actor in the world model from real states (h, z): lists over steps."""
        m = self.m
        dev = m.device
        n = h.shape[0]
        traj = {"feat": [], "term": [], "new": [], "opt": [], "prev_opt": [], "first": [], "toks": [], "ids": [],
                "rew": [], "cont": [], "disc": [], "u": [], "viol": [], "succ": []}
        cur, fst = prev_opt, first
        lam_t = torch.tensor([lam.get(f, 0.0) for f in fams], device=dev)
        gamma = self.c.get("gamma", 0.99)
        for _ in range(horizon):
            f = m.feat(h, z)
            term, new, opt, toks, ids = self._act(f, g, cur, fst, fams)
            with torch.no_grad():
                a_e = m.wm_act_enc(m.ids(toks, m.cfg.max_act + 2))
                u = m.ens_pred(f, a_e).var(0).mean(-1)
                h, pr = m.img_step(h, z, a_e)
                z = m.sample_z(pr)
                H = m.heads(m.feat(h, z), g)
                viol = torch.sigmoid(H["viol"])
                dt = torch.expm1(H["dt"].clamp(-3, 5)).clamp(min=0.0)
            for k, v in (("feat", f), ("term", term), ("new", new), ("opt", opt), ("prev_opt", cur), ("first", fst),
                         ("toks", toks), ("ids", ids), ("rew", H["rew"] - lam_t * viol), ("cont", torch.sigmoid(H["cont"])),
                         ("disc", gamma ** dt), ("u", u), ("viol", viol), ("succ", torch.sigmoid(H["succ"]))):
                traj[k].append(v)
            cur, fst = opt, torch.zeros_like(fst)
        traj["feat"].append(m.feat(h, z))
        return traj

    def returns(self, traj, g, L: torch.Tensor):
        """Lambda-returns over steps k < L (per trajectory), bootstrapped at state L with target V_img."""
        m = self.m
        lam = self.c.get("lambda", 0.95)
        Hn = len(traj["rew"])
        with torch.no_grad():
            V = torch.stack([m.value_img(f, g, target=True) for f in traj["feat"]])  # (H+1, n)
            R = torch.zeros_like(V)
            R[Hn] = V[Hn]
            for k in reversed(range(Hn)):
                boot = torch.where(L == k + 1, V[k + 1], (1 - lam) * V[k + 1] + lam * R[k + 1])
                R[k] = traj["rew"][k] + traj["disc"][k] * traj["cont"][k] * boot
                R[k] = torch.where(k >= L, V[k], R[k])
        return R, V

    def update(self, m_opt_actor, m_opt_critic, h, z, g, fams, prev_opt, first, lam, clip: float = 1.0) -> dict:
        m, c = self.m, self.c
        dev = m.device
        H = self.max_h
        traj = self.imagine(h, z, g, fams, prev_opt, first, lam, H)
        n = h.shape[0]
        u = torch.stack(traj["u"])  # (H, n)
        if self.c.get("mode") == "adaptive":
            bad = (u > self.u_thr).float()
            first_bad = torch.where(bad.any(0), bad.argmax(0), torch.full((n,), H, device=dev))
            L = first_bad.clamp(max=H)
        else:
            L = torch.full((n,), H, device=dev)
        R, _ = self.returns(traj, g, L)
        # percentile return normalisation (DreamerV3)
        live = torch.stack([(k < L) for k in range(H)])  # (H, n)
        if live.any():
            rv = R[:H][live]
            rng_ = float(torch.quantile(rv, 0.95) - torch.quantile(rv, 0.05))
            self.scale = 0.99 * self.scale + 0.01 * max(1.0, rng_)
        w = torch.ones(H, n, device=dev)
        for k in range(1, H):
            w[k] = w[k - 1] * traj["cont"][k - 1]
        w = w * live.float()
        # critic on imagined returns
        Lc = torch.zeros((), device=dev)
        Lpi = torch.zeros((), device=dev)
        ent_tot = torch.zeros((), device=dev)
        kappa = c.get("kappa", 1.0) * (self.reliability if c.get("mode") == "adaptive" else 1.0)
        from agentrs.cam.v2.learner import decision_logprob  # noqa: WPS433

        class _S:  # minimal step view for decision_logprob
            __slots__ = ("prev_opt", "opt", "first", "term_sample", "new_opt", "act_tokens", "act_ids")

        for k in range(H):
            if not live[k].any():
                continue
            f = traj["feat"][k].detach()
            vk = m.value_img(f, g)
            Lc = Lc + (w[k] * (vk - R[k]) ** 2).sum() / w.sum().clamp(min=1)
            steps = []
            for i in range(n):
                s = _S()
                s.prev_opt, s.opt, s.first = int(traj["prev_opt"][k][i]), int(traj["opt"][k][i]), bool(traj["first"][k][i])
                s.term_sample, s.new_opt = int(traj["term"][k][i]), bool(traj["new"][k][i])
                s.act_tokens, s.act_ids = traj["toks"][k][i], traj["ids"][k][i]
                steps.append(s)
            lp, ent = decision_logprob(m, f, g, steps, [_grammar(x) for x in fams])
            adv = ((R[k] - vk) / self.scale).detach()
            Lpi = Lpi - (w[k] * (adv * lp + c.get("ent_coef", 0.003) * ent)).sum() / w.sum().clamp(min=1)
            ent_tot = ent_tot + (w[k] * ent).sum() / w.sum().clamp(min=1)
        out = {"dream_len_mean": float(L.float().mean()), "dream_u_mean": float(u.mean()),
               "dream_return_mean": float(R[0].mean()), "dream_scale": self.scale, "dream_kappa": kappa,
               "dream_pred_viol": float(torch.stack(traj["viol"]).mean()), "dream_pred_succ": float(torch.stack(traj["succ"]).mean()),
               "imagined_transitions": int(live.sum())}
        if not live.any():  # nothing trusted to imagine (e.g. adaptive mode before calibration)
            out.update({"dream_critic_loss": 0.0, "dream_actor_loss": 0.0})
            return out
        m_opt_critic.zero_grad(set_to_none=True)
        Lc.backward()
        torch.nn.utils.clip_grad_norm_([p for gp in m_opt_critic.param_groups for p in gp["params"]], clip)
        m_opt_critic.step()
        if kappa > 0:
            m_opt_actor.zero_grad(set_to_none=True)
            (kappa * Lpi).backward()
            out["dream_actor_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
                [p for gp in m_opt_actor.param_groups for p in gp["params"]], clip))
            m_opt_actor.step()
        out.update({"dream_critic_loss": float(Lc), "dream_actor_loss": float(Lpi)})
        return out

    def set_reliability(self, cal: dict) -> None:
        """Adaptive mode: reliability from real held-out prediction errors; disagreement threshold from
        the calibration of ensemble disagreement against actual one-step errors."""
        if self.c.get("mode") != "adaptive" or not cal:
            return
        def num(x, bad):  # missing / NaN / inf measurements count as unreliable
            return bad if x is None or not np.isfinite(x) else float(x)

        e1 = num(cal.get("ret_abs_err_k1"), 1e9)
        ece = num(cal.get("viol_ece_k1"), 1.0)
        rank = num(cal.get("u_err_spearman"), 0.0)
        self.reliability = float(np.clip(1 - e1 / self.c.get("err_ref", 0.1), 0, 1) * np.clip(1 - ece / 0.2, 0, 1))
        if rank < 0.1:  # disagreement does not track real error: do not trust it to extend rollouts
            self.u_thr = num(cal.get("u_q50"), 0.0)
        else:
            self.u_thr = num(cal.get(f"u_q{int(self.c.get('u_quantile', 0.8) * 100)}"), num(cal.get("u_q50"), 0.0))
        if self.reliability <= 0:
            self.u_thr = 0.0  # unreliable model: imagine nothing
        # effective horizon: largest k whose real open-loop return error stays within err_max
        ok = [k for k in (1, 2, 4, 8) if k <= self.c.get("horizon", 8) and num(cal.get(f"ret_abs_err_k{k}"), 1e9) <= self.c.get("err_max", 0.1)]
        self.max_h = max(ok) if ok else 1
