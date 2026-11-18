"""CAM-v2 off-policy learning from replayed real experience.

World model (own optimizer): posterior unroll of complete replayed episodes; KL-balanced prior /
posterior objective with free bits (DreamerV3-style), reconstruction of observable features and
verified status, contract-reward / continuation / clock-duration / event / violation / success
heads, eventual-outcome (viability) head, and an ensemble of next-embedding predictors (uncertainty).

Critics (own optimizer): twin Q(s, g, a) on stop-gradient states, regressed to semi-Markov n-step
targets (discount gamma ** dt per action) whose n-step tail is cut where the current policy's
probability of the recorded decision falls below ``c_min`` times the behaviour probability
(off-policy safeguard) and bootstrapped with V; V by expectile regression on min target-Q (IQL; it
never queries Q on actions outside the data). Polyak-averaged target critics.

Actor (own optimizer), one of:
  * ``awr``  — advantage-weighted regression of the full structured decision (option termination,
               option, grammar-masked action tokens) on replayed decisions: exp(beta (Q - V)) log pi;
  * ``sac``  — discrete soft actor-critic for families whose complete action set is enumerable
               (product: 33, bizdev: 231 actions; masked probabilities over every valid action and an
               auto-tuned temperature with a calibrated target entropy); other families use ``awr``.
Hindsight views (relabelled goals) are used only by these off-policy losses, never as on-policy data.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from agentrs.cam.envs import ENVS
from agentrs.cam.v2.data import View
from agentrs.cam.v2.model import CAM2

ENUMERABLE = {"product", "bizdev"}
OUTCOME = {"success": 0, "violation": 1, "failure": 2}


def enumerate_actions(grammar) -> list[list[str]]:
    out = []
    for typ, slots in grammar.spec.items():
        for combo in itertools.product(*slots) if slots else [()]:
            out.append([typ, *combo, "END"])
    return out


def _fam_masks(m: CAM2, fams: list[str]):
    fm = torch.zeros(len(fams), m.cfg.n_feat)
    em = torch.zeros(len(fams), m.cfg.n_events)
    for i, f in enumerate(fams):
        fm[i, :ENVS[f].n_features] = 1
        em[i, :len(ENVS[f].events)] = 1
    return fm.to(m.device), em.to(m.device)


def observe(m: CAM2, views: list[View], gen=None) -> dict:
    """Posterior unroll over complete (possibly hindsight-truncated) episodes: states t = 0..T (T =
    number of steps; state T is the terminal observation). Returns flat rows and per-row tensors."""
    dev = m.device
    rows, obs, prev, el, stv, sem = [], [], [], [], [], []
    for i, v in enumerate(views):
        st = v.steps
        T = len(st)
        for t in range(T + 1):
            if t < T:
                s = st[t]
                obs.append(s.obs), prev.append(s.prev_act), el.append(s.elapsed_prev), stv.append(s.status), sem.append(s.sem)
            else:
                e = v.ep
                nxt = e.steps[t] if t < len(e.steps) else None
                if nxt is not None:
                    obs.append(nxt.obs), prev.append(nxt.prev_act), el.append(nxt.elapsed_prev), stv.append(nxt.status)
                    sem.append(nxt.sem)
                else:
                    obs.append(e.final_obs or st[-1].obs), prev.append(st[-1].act_tokens), el.append(e.final_elapsed)
                    stv.append(st[-1].status_after)
                    sem.append(e.final_sem if e.final_sem is not None else st[-1].sem)
            rows.append((i, t))
    pos = {r: k for k, r in enumerate(rows)}
    obs_in = torch.tensor(np.stack(sem), device=dev) if m.semantic else m.pad_ids(obs, m.cfg.max_obs)
    e_all = m.encode_obs(obs_in, torch.tensor(el, device=dev, dtype=torch.float32),
                         torch.tensor(np.stack(stv), device=dev, dtype=torch.float32))
    a_prev = m.wm_act_enc(m.ids(prev, m.cfg.max_act + 2))
    B = len(views)
    Tm = max(len(v.steps) for v in views) + 1
    h, z = m.initial(B)
    feats = [None] * len(rows)
    priors = [None] * len(rows)
    posts = [None] * len(rows)
    for t in range(Tm):
        act = [i for i in range(B) if t <= len(views[i].steps)]
        ks = [pos[(i, t)] for i in act]
        ai = torch.tensor(act, device=dev)
        kt = torch.tensor(ks, device=dev)
        h_new, pr = m.img_step(h[ai], z[ai], a_prev[kt])
        po = m.post_logits(h_new, e_all[kt])
        z_new = m.sample_z(po, gen=gen)
        h = h.index_copy(0, ai, h_new)
        z = z.index_copy(0, ai, z_new)
        f = m.feat(h_new, z_new)
        for j, k in enumerate(ks):
            feats[k], priors[k], posts[k] = f[j], pr[j], po[j]
    if m.semantic:
        goal = m.encode_goal(torch.tensor(np.stack([v.goal_sem for v in views]), device=dev))
    else:
        goal = m.encode_goal(m.ids([[f"FAM_{v.ep.family}"] + v.goal for v in views], 48))
    return {"rows": rows, "pos": pos, "feat": torch.stack(feats), "prior": torch.stack(priors),
            "post": torch.stack(posts), "e": e_all, "goal": goal}


def wm_loss(m: CAM2, views: list[View], ob: dict, rng: random.Random, free_bits: float = 1.0) -> dict:
    """World-model objective on real (non-relabelled) episodes; archive burn-in prefixes excluded."""
    dev = m.device
    rows, pos = ob["rows"], ob["pos"]
    keep = [k for k, (i, t) in enumerate(rows) if not views[i].relabeled and t >= views[i].ep.n_prefix]
    kk = torch.tensor(keep, device=dev)
    dyn = m.kl(ob["post"][kk].detach(), ob["prior"][kk])
    rep = m.kl(ob["post"][kk], ob["prior"][kk].detach())
    L_kl = 0.5 * dyn.clamp(min=free_bits).mean() + 0.1 * rep.clamp(min=free_bits).mean()
    gi = torch.tensor([rows[k][0] for k in keep], device=dev)
    H = m.heads(ob["feat"][kk], ob["goal"][gi])
    fams = [views[rows[k][0]].ep.family for k in keep]
    fm, em = _fam_masks(m, fams)
    ft, sv = [], []
    for k in keep:
        i, t = rows[k]
        st = views[i].steps
        ft.append(st[t].feat_obs if t < len(st) else st[-1].feat_after)
        sv.append(st[t].status if t < len(st) else st[-1].status_after)
    ft = torch.tensor(np.stack(ft), device=dev)
    sv = torch.tensor(np.stack(sv), device=dev)
    L_rec = (((H["feat"] - ft) ** 2) * fm).sum() / fm.sum() + ((H["status"] - sv) ** 2).mean()
    # transition targets at the state reached by each step (t >= 1)
    tr = [j for j, k in enumerate(keep) if rows[k][1] >= 1]
    tj = torch.tensor(tr, device=dev)
    src = [views[rows[keep[j]][0]].steps[rows[keep[j]][1] - 1] for j in tr]
    r_t = torch.tensor([s.reward(0.0) for s in src], device=dev, dtype=torch.float32)
    c_t = torch.tensor([0.0 if s.done else 1.0 for s in src], device=dev)
    dt_t = torch.tensor([math.log1p(max(0.0, s.dt)) for s in src], device=dev, dtype=torch.float32)
    ev_t = torch.tensor(np.stack([s.events for s in src]), device=dev)
    vi_t = torch.tensor([s.viol for s in src], device=dev, dtype=torch.float32)
    su_t = torch.tensor([float(s.r_succ > 0 or (s.done and views[rows[keep[j]][0]].ep.success)) for j, s in zip(tr, src)],
                        device=dev)
    L_rew = F.mse_loss(H["rew"][tj], r_t)
    L_cont = F.binary_cross_entropy_with_logits(H["cont"][tj], c_t)
    L_dt = F.mse_loss(H["dt"][tj], dt_t)
    L_ev = (F.binary_cross_entropy_with_logits(H["ev"][tj], ev_t, reduction="none") * em[tj]).sum() / em[tj].sum().clamp(min=1)
    L_viol = F.binary_cross_entropy_with_logits(H["viol"][tj], vi_t)
    L_succ = F.binary_cross_entropy_with_logits(H["succ"][tj], su_t)
    # eventual outcome (viability) at decision states
    dec = [j for j, k in enumerate(keep) if rows[k][1] < len(views[rows[k][0]].steps)]
    dj = torch.tensor(dec, device=dev)
    oc = torch.tensor([OUTCOME[views[rows[keep[j]][0]].ep.outcome or "failure"] for j in dec], device=dev)
    L_out = F.cross_entropy(H["outcome"][dj], oc)
    # ensemble of next-embedding predictors (uncertainty), on stop-gradient inputs/targets
    a_ids = m.ids([views[rows[keep[j]][0]].steps[rows[keep[j]][1]].act_tokens for j in dec], m.cfg.max_act + 2)
    a_e = m.wm_act_enc(a_ids).detach()
    nxt = torch.tensor([ob["pos"][(rows[keep[j]][0], rows[keep[j]][1] + 1)] for j in dec], device=dev)
    pred = m.ens_pred(ob["feat"][kk][dj].detach(), a_e)
    tgt = ob["e"][nxt].detach()
    boot = (torch.rand(pred.shape[0], pred.shape[1], 1, device=dev) < 0.8).float()
    L_ens = (((pred - tgt[None]) ** 2) * boot).sum() / (boot.sum() * pred.shape[-1]).clamp(min=1)
    total = L_kl + L_rec + L_rew + L_cont + L_dt + L_ev + L_viol + L_succ + 0.5 * L_out + L_ens
    return {"total": total, "kl_dyn": dyn.mean(), "kl_rep": rep.mean(), "recon": L_rec, "rew": L_rew, "cont": L_cont,
            "dt": L_dt, "events": L_ev, "viol": L_viol, "succ": L_succ, "outcome": L_out, "ens": L_ens}


def decision_logprob(m: CAM2, feat, g, steps, grammars):
    """Log-prob (and entropy) of the recorded factored decisions under the current actor."""
    dev = m.device
    prev_opt = torch.tensor([s.prev_opt for s in steps], device=dev)
    opt = torch.tensor([s.opt for s in steps], device=dev)
    logits, beta = m.actor_dists(feat, g, prev_opt)
    first = torch.tensor([s.first for s in steps], device=dev)
    term = torch.tensor([s.term_sample for s in steps], device=dev).float()
    new = torch.tensor([s.new_opt for s in steps], device=dev)
    t_lp = torch.where(first, torch.zeros_like(beta), term * F.logsigmoid(beta) + (1 - term) * F.logsigmoid(-beta))
    olp = torch.log_softmax(logits.float(), -1)
    o_lp = torch.where(new, olp.gather(-1, opt[:, None]).squeeze(-1), torch.zeros_like(beta))
    allowed = []
    for s, gr in zip(steps, grammars):
        toks = s.act_tokens
        allowed.append([m.allowed_ids(gr, toks[:k]) for k in range(len(toks))])
    a_lp, a_ent = m.action_logprob(feat, g, opt, [s.act_ids for s in steps], allowed)
    o_ent = -(olp.exp() * olp).sum(-1)
    return t_lp + o_lp + a_lp, a_ent + 0.1 * o_ent


class OffPolicy:
    def __init__(self, m: CAM2, cfg: dict):
        self.m, self.c = m, cfg
        g = m.groups()
        self.opt = {"wm": torch.optim.AdamW(g["wm"], lr=cfg.get("wm_lr", 3e-4), weight_decay=0.0),
                    "actor": torch.optim.AdamW(g["actor"], lr=cfg.get("actor_lr", 1e-4), weight_decay=0.0),
                    "critic": torch.optim.AdamW(g["critic"], lr=cfg.get("critic_lr", 3e-4), weight_decay=0.0)}
        a0 = math.log(cfg.get("sac_alpha_init", 0.02))  # temperature on the scale of the contract rewards (~0.01-1)
        self.log_alpha = {f: torch.full((), a0, device=m.device, requires_grad=True) for f in ENUMERABLE}
        self.alpha_opt = torch.optim.Adam(list(self.log_alpha.values()), lr=3e-4)
        self._enum: dict = {}
        self.steps = 0
        self.adv_scale = 1.0   # running std of advantages (AWR weights are scale-free in the reward units)

    def state_dict(self) -> dict:
        return {"opt": {k: o.state_dict() for k, o in self.opt.items()}, "alpha_opt": self.alpha_opt.state_dict(),
                "log_alpha": {k: float(v.detach()) for k, v in self.log_alpha.items()}, "steps": self.steps,
                "adv_scale": self.adv_scale}

    def load_state_dict(self, d: dict) -> None:
        for k, o in self.opt.items():
            o.load_state_dict(d["opt"][k])
        self.alpha_opt.load_state_dict(d["alpha_opt"])
        for k, v in d["log_alpha"].items():
            self.log_alpha[k].data.fill_(v)
        self.steps = d.get("steps", 0)
        self.adv_scale = d.get("adv_scale", 1.0)

    def _step(self, name: str, loss: torch.Tensor) -> float:
        o = self.opt[name]
        o.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for gp in o.param_groups for p in gp["params"]], self.c.get("grad_clip", 1.0))
        o.step()
        return float(gn)

    # ------------------------------------------------------------------ targets
    def _targets(self, views, ob, f_det, lam, cur_lp, n_max, c_min):
        """Semi-Markov n-step targets with truncation on the current/behaviour probability ratio."""
        m = self.m
        dev = m.device
        rows, pos = ob["rows"], ob["pos"]
        with torch.no_grad():
            V_all = m.value(f_det, ob["goal"][torch.tensor([i for i, _ in rows], device=dev)])
        ys, idx = [], []
        for i, v in enumerate(views):
            r = v.rewards(lam.get(v.ep.family, 0.0))
            d = v.dones()
            st = v.steps
            T = len(st)
            for t in range(T):
                if not st[t].learn:
                    continue
                y, D, k = 0.0, 1.0, 0
                while True:
                    y += D * r[t + k]
                    D *= st[t + k].disc
                    if d[t + k]:
                        D = 0.0
                        break
                    k += 1
                    if k >= n_max or t + k >= T:
                        break
                    ratio = math.exp(min(20.0, cur_lp.get((i, t + k), st[t + k].logp_b) - st[t + k].logp_b))
                    if ratio < c_min:
                        break
                if D > 0:
                    y += D * float(V_all[pos[(i, t + k)]])
                ys.append(y)
                idx.append((i, t))
        return idx, torch.tensor(ys, device=dev, dtype=torch.float32)

    def _enum_actions(self, fam):
        if fam not in self._enum:
            gr = ENVS[fam]().grammar()
            acts = enumerate_actions(gr)
            m = self.m
            ids = [[m.tok[t] for t in a] for a in acts]
            allowed = [[m.allowed_ids(gr, a[:k]) for k in range(len(a))] for a in acts]
            self._enum[fam] = (acts, ids, allowed, m.ids([["<bos>"] + a for a in acts], m.cfg.max_act + 2))
        return self._enum[fam]

    # ------------------------------------------------------------------ one update
    def update(self, views: list[View], lam: dict, rng: random.Random, train_wm: bool = True) -> dict:
        m, c = self.m, self.c
        m.train()
        dev = m.device
        out = {}
        ob = observe(m, views)
        if train_wm:
            W = wm_loss(m, views, ob, rng, c.get("free_bits", 1.0))
            out["wm_grad_norm"] = self._step("wm", W["total"])
            out.update({f"wm_{k}": float(v) for k, v in W.items() if k != "total"})
        f_det = ob["feat"].detach()
        g_det = ob["goal"].detach()
        rows, pos = ob["rows"], ob["pos"]
        # decisions (learnable steps)
        dec = [(i, t) for i, v in enumerate(views) for t in range(len(v.steps)) if v.steps[t].learn]
        if not dec:
            return out
        k_dec = torch.tensor([pos[x] for x in dec], device=dev)
        gi = torch.tensor([i for i, _ in dec], device=dev)
        fd, gd = f_det[k_dec], g_det[gi]
        steps = [views[i].steps[t] for i, t in dec]
        grams = [_grammar(views[i].ep.family) for i, _ in dec]
        lp, ent = decision_logprob(m, fd, gd, steps, grams)
        cur_lp = {x: float(v) for x, v in zip(dec, lp.detach())}
        idx, y = self._targets(views, ob, f_det, lam, cur_lp, c.get("n_step", 3), c.get("c_min", 0.3))
        assert idx == dec
        act_ids = m.ids([["<bos>"] + s.act_tokens for s in steps], m.cfg.max_act + 2)
        # ---- critics
        fams = [views[i].ep.family for i, _ in dec]
        sac_rows = [j for j, f in enumerate(fams) if c.get("actor") == "sac" and f in ENUMERABLE]
        q1, q2 = m.q_values(fd, gd, act_ids)
        y_used = y.clone()
        if sac_rows:
            y_used = self._sac_targets(views, dec, ob, f_det, g_det, lam, y_used, sac_rows)
        L_q = F.mse_loss(q1, y_used) + F.mse_loss(q2, y_used)
        with torch.no_grad():
            tq1, tq2 = m.q_values(fd, gd, act_ids, target=True)
            tq = torch.min(tq1, tq2)
        V = m.value(fd, gd)
        tau = c.get("expectile", 0.7)
        u = tq - V
        L_v = (torch.abs(tau - (u < 0).float()) * u ** 2).mean()
        # V_img grounded on real targets too (used by the dream learner)
        L_vimg = F.mse_loss(m.value_img(fd, gd), y)
        out["critic_grad_norm"] = self._step("critic", L_q + L_v + c.get("vimg_real_coef", 0.5) * L_vimg)
        m.update_targets(c.get("polyak", 0.01))
        # ---- actor
        with torch.no_grad():
            A = tq - m.value(fd, gd)
            if c.get("awr_norm", True):
                self.adv_scale = 0.99 * self.adv_scale + 0.01 * float(A.std().clamp(min=1e-4)) if self.steps else float(A.std().clamp(min=1e-4))
                An = A / self.adv_scale
            else:
                An = A
        beta = c.get("awr_beta", 1.0)
        w = torch.exp(beta * An).clamp(max=c.get("awr_max_w", 20.0))
        awr_rows = [j for j in range(len(dec)) if j not in set(sac_rows)]
        L_pi = torch.zeros((), device=dev)
        if awr_rows:
            aj = torch.tensor(awr_rows, device=dev)
            L_pi = -(w[aj] * lp[aj]).mean() - c.get("ent_coef", 0.003) * ent[aj].mean()
        if sac_rows:
            L_pi = L_pi + self._sac_actor(dec, views, fd, gd, sac_rows, out)
        out["actor_grad_norm"] = self._step("actor", L_pi)
        self.steps += 1
        out.update({"q_loss": float(L_q), "v_loss": float(L_v), "vimg_real_loss": float(L_vimg), "actor_loss": float(L_pi),
                    "awr_w_mean": float(w.mean()), "awr_w_max": float(w.max()), "adv_mean": float(A.mean()), "adv_scale": self.adv_scale, "q_mean": float(q1.mean()),
                    "target_mean": float(y.mean()), "entropy": float(ent.mean()), "decisions": len(dec),
                    "logp_change_vs_behaviour": float(np.mean([cur_lp[x] - views[x[0]].steps[x[1]].logp_b for x in dec]))})
        m.eval()
        return out

    # ------------------------------------------------------------------ discrete SAC (enumerable families)
    def _policy_all(self, fam, feat, g, opt):
        """log pi(a|s) for every valid action (masked), batched over states x actions."""
        acts, ids, allowed, _ = self._enum_actions(fam)
        n, A = feat.shape[0], len(acts)
        F_ = feat.repeat_interleave(A, 0)
        G_ = g.repeat_interleave(A, 0)
        O_ = opt.repeat_interleave(A, 0)
        lp, _ = self.m.action_logprob(F_, G_, O_, ids * n, allowed * n)
        return lp.view(n, A)

    def _q_all(self, fam, feat, g, target):
        _, _, _, aid = self._enum_actions(fam)
        n, A = feat.shape[0], aid.shape[0]
        q1, q2 = self.m.q_values(feat.repeat_interleave(A, 0), g.repeat_interleave(A, 0), aid.repeat(n, 1), target=target)
        return torch.min(q1, q2).view(n, A)

    def _sac_targets(self, views, dec, ob, f_det, g_det, lam, y, sac_rows):
        m = self.m
        dev = m.device
        y = y.clone()
        for fam in ENUMERABLE:
            js = [j for j in sac_rows if views[dec[j][0]].ep.family == fam]
            if not js:
                continue
            alpha = float(self.log_alpha[fam].exp())
            nxt, live = [], []
            for j in js:
                i, t = dec[j]
                nxt.append(ob["pos"][(i, t + 1)])
            kn = torch.tensor(nxt, device=dev)
            gi = torch.tensor([dec[j][0] for j in js], device=dev)
            opt = torch.tensor([views[dec[j][0]].steps[dec[j][1]].opt for j in js], device=dev)
            with torch.no_grad():
                lpn = self._policy_all(fam, f_det[kn], g_det[gi], opt)
                qn = self._q_all(fam, f_det[kn], g_det[gi], target=True)
                vsoft = (lpn.exp() * (qn - alpha * lpn)).sum(-1)
            for jj, j in enumerate(js):
                i, t = dec[j]
                v = views[i]
                r = v.rewards(lam.get(fam, 0.0))[t]
                d = v.dones()[t]
                y[j] = r + (0.0 if d else v.steps[t].disc * float(vsoft[jj]))
        return y

    def _sac_actor(self, dec, views, fd, gd, sac_rows, out):
        m = self.m
        dev = m.device
        L = torch.zeros((), device=dev)
        for fam in ENUMERABLE:
            js = [j for j in sac_rows if views[dec[j][0]].ep.family == fam]
            if not js:
                continue
            jt = torch.tensor(js, device=dev)
            opt = torch.tensor([views[dec[j][0]].steps[dec[j][1]].opt for j in js], device=dev)
            lpa = self._policy_all(fam, fd[jt], gd[jt], opt)
            with torch.no_grad():
                qa = self._q_all(fam, fd[jt], gd[jt], target=False)
            alpha = self.log_alpha[fam].exp().detach()
            p = lpa.exp()
            L = L + (p * (alpha * lpa - qa)).sum(-1).mean()
            H = -(p * lpa).sum(-1).detach().mean()
            A = lpa.shape[1]
            target_H = self.c.get("sac_target_entropy_frac", 0.5) * math.log(A)
            la = -(self.log_alpha[fam] * (target_H - H))
            self.alpha_opt.zero_grad(set_to_none=True)
            la.backward()
            self.alpha_opt.step()
            out[f"sac_alpha_{fam}"] = float(self.log_alpha[fam].exp())
            out[f"sac_entropy_{fam}"] = float(H)
        return L


_GRAMMARS: dict = {}


def _grammar(fam):
    if fam not in _GRAMMARS:
        _GRAMMARS[fam] = ENVS[fam]().grammar()
    return _GRAMMARS[fam]


def ppo_update(off: "OffPolicy", eps, lam: dict, cfg: dict, rng: random.Random) -> dict:
    """On-policy PPO for the CAM-v2 actor and value head (benchmark learner inside the same architecture).
    The actor's inputs (stop-gradient RSSM features and goal encoding) were stored at collection time, so
    probability ratios are exact; semi-Markov GAE (discount gamma ** dt per action); archive burn-in prefixes
    are excluded. The world model is trained separately (``wm_step``) from replay."""
    m = off.m
    dev = m.device
    m.train()
    data = []
    for e in eps:
        st = [s for s in e.steps if s.learn]
        if not st or st[0].feat is None:
            continue
        data.append((e, st))
    if not data:
        return {}
    with torch.no_grad():
        for e, st in data:
            f = torch.tensor(np.stack([s.feat for s in st]), device=dev).float()
            g = torch.tensor(e.goal_feat, device=dev).float()[None].expand(len(st), -1)
            v = m.value(f, g).cpu().numpy()
            nxt_v, nxt_a = 0.0, 0.0
            for k in reversed(range(len(st))):
                s = st[k]
                d = 0.0 if s.done else s.disc
                delta = s.reward(lam.get(e.family, 0.0)) + d * nxt_v - v[k]
                nxt_a = delta + d * cfg.get("lam", 0.95) * nxt_a
                s._adv, s._ret = nxt_a, nxt_a + v[k]
                nxt_v = v[k]
    advs = np.array([s._adv for _, st in data for s in st])
    mu, sd = advs.mean(), advs.std() + 1e-8
    stats = {k: 0.0 for k in ("pg", "entropy", "value", "clipfrac", "kl")}
    n_mb, stop = 0, False
    for _ in range(cfg.get("epochs", 3)):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s0 in range(0, len(order), cfg.get("mb_episodes", 64)):
            mb = [data[i] for i in order[s0:s0 + cfg.get("mb_episodes", 64)]]
            rows = [(e, s) for e, st in mb for s in st]
            f = torch.tensor(np.stack([s.feat for _, s in rows]), device=dev).float()
            g = torch.tensor(np.stack([e.goal_feat for e, _ in rows]), device=dev).float()
            lp, ent = decision_logprob(m, f, g, [s for _, s in rows], [_grammar(e.family) for e, _ in rows])
            old = torch.tensor([s.logp_b for _, s in rows], device=dev)
            A = torch.tensor([(s._adv - mu) / sd for _, s in rows], device=dev, dtype=torch.float32)
            R = torch.tensor([s._ret for _, s in rows], device=dev, dtype=torch.float32)
            ratio = torch.exp(lp - old)
            if n_mb == 0:
                stats["mismatch"] = float((lp.detach() - old).abs().mean())
            clip = cfg.get("clip", 0.2)
            L_pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean() - cfg.get("ent_coef", 0.01) * ent.mean()
            L_v = F.mse_loss(m.value(f, g), R)
            for name, loss in (("actor", L_pg), ("critic", L_v)):
                off.opt[name].zero_grad(set_to_none=True)
            (L_pg + 0.5 * L_v).backward()
            for name in ("actor", "critic"):
                torch.nn.utils.clip_grad_norm_([p for gp in off.opt[name].param_groups for p in gp["params"]], cfg.get("grad_clip", 1.0))
                off.opt[name].step()
            with torch.no_grad():
                kl = float((old - lp.detach()).mean())
            stats["pg"] += float(L_pg)
            stats["entropy"] += float(ent.mean())
            stats["value"] += float(L_v)
            stats["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
            stats["kl"] += kl
            n_mb += 1
            if kl > cfg.get("target_kl", 0.1):
                stop = True
                break
        if stop:
            break
    for k in ("pg", "entropy", "value", "clipfrac", "kl"):
        stats[k] /= max(1, n_mb)
    stats["updates"] = n_mb
    stats["early_stop"] = float(stop)
    m.eval()
    return stats


def wm_step(off: "OffPolicy", views, rng: random.Random) -> dict:
    """World-model-only update on replayed real episodes (used with the PPO learner)."""
    m = off.m
    m.train()
    ob = observe(m, views)
    W = wm_loss(m, views, ob, rng, off.c.get("free_bits", 1.0))
    gn = off._step("wm", W["total"])
    m.eval()
    return {"wm_grad_norm": gn, **{f"wm_{k}": float(v) for k, v in W.items() if k != "total"}}
