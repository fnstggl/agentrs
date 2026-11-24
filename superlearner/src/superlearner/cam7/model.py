"""CAM 7.0: the Unified Adaptive Experience Learner (one shared network, explicit fast Bayesian adaptation).

Tensor interfaces (B environments, T history rows, L = 12 lever slots, G = 16 gauge slots, K = 2 outcomes, C candidates):

    History           x, ema [B,T,L]   settings in force during each tick and their moving average (scaled [-1, 1])
                      out [B,T,K]      revenue and cost of each tick (the task's public money scale)
                      gauges [B,T,G]   symlog readings;   tau [B,T] tick / horizon;   funds [B,T]
                      pred [B,T,2]     the net the learner predicted for the row before seeing it (mean, sd; 0 = none)
                      x0 [B,L] defaults, lmask [B,L], gmask [B,G], lid [B,L,I], gid [B,G,I] per-episode identities
    cut c [B]         rows < c are the visible history H

One decision:

    tokens(H)                         lever tokens (L), gauge tokens (G), one global token, W recent event tokens
    h = Backbone_theta(tokens)        pass 1: the context of every lever
    phi_t = F_theta(h, x_t, e_t, t)   adaptive features of every row (fixed [d, d^2, e, e^2] + learned + interaction)
    posterior = NIG(phi, y, omega)    exact fast Bayesian update (posterior.py), one per outcome
    z = Backbone_theta(tokens, posterior tokens)    pass 2: the same network reads means, uncertainty, innovations
    score(a) = beta * [Q(a) + n_rem KG(a)] / sd  +  r_theta(z, a)      over candidate interventions a
    V(z)                              value of the integrated state

Slow parameters theta: transformer, feature networks, priors (mu0, Lambda0, a0, b0, forgetting), heads.
Fast state: the history and the posterior it implies; nothing about a world is carried in theta except through training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import posterior as P
from .config import Cam7Config

K_OUT = 2            # revenue, cost
N_FIXED = 4          # d, d^2, e, e^2
N_GLOBAL = 3         # bias, tau, tau^2
N_POST_LEVER = 8     # 2 outcomes x 2 probe directions x (mean, sd)
N_POST_GLOBAL = 4
N_CAND_FLAGS = 5     # hold, posterior-mean optimum, Thompson optimum, single-lever probe, random
N_CAND = 11 + N_CAND_FLAGS


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def small_init(m: nn.Sequential, std: float = 0.01) -> nn.Sequential:
    last = m[-1]
    nn.init.normal_(last.weight, std=std)
    nn.init.zeros_(last.bias)
    return m


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


@dataclass
class History:
    x: Tensor
    ema: Tensor
    out: Tensor
    gauges: Tensor
    tau: Tensor
    funds: Tensor
    pred: Tensor
    x0: Tensor
    lmask: Tensor
    gmask: Tensor
    lid: Tensor
    gid: Tensor

    def index(self, idx: Tensor) -> "History":
        return History(*[getattr(self, f)[idx] for f in self.__dataclass_fields__])

    def to(self, device) -> "History":
        return History(*[getattr(self, f).to(device) for f in self.__dataclass_fields__])


def _rows(t: Tensor, r: Tensor) -> Tensor:
    """t [B, T, ...] gathered at row r [B] (clamped at 0) -> [B, ...]."""
    r = r.clamp_min(0)
    idx = r.view(-1, *([1] * (t.dim() - 1))).expand(-1, 1, *t.shape[2:])
    return t.gather(1, idx).squeeze(1)


class Cam7Model(nn.Module):
    def __init__(self, cfg: Cam7Config, n_levers: int = 12, n_gauges: int = 16):
        super().__init__()
        self.cfg = cfg
        self.L, self.G = n_levers, n_gauges
        d, I = cfg.d_model, cfg.id_dim
        self.F = (N_FIXED if cfg.use_fixed_features else 0) + \
            ((cfg.n_learned + cfg.n_interact) if cfg.use_learned_features else 0)
        self.D = self.L * self.F + N_GLOBAL
        # ---- tokens
        self.lever_in = nn.Linear(4 + I + N_POST_LEVER + 1, d)
        self.gauge_in = nn.Linear(3 + I, d)
        self.global_in = nn.Linear(6 + N_POST_GLOBAL + 1, d)
        self.event_in = nn.Linear(2 * I + 8, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        # ---- adaptive features
        c = 32
        self.ctx = nn.Linear(d, c)
        # interaction partners: each lever contributes A(h_k) d_k to the others' summaries (linear in the move, so
        # d_j * u_j spans pairwise products; the per-episode identities inside h_k make any pair representable)
        self.q_net = nn.Linear(c, cfg.n_interact)
        self.g_net = small_init(mlp(c + 6 + cfg.n_interact, cfg.feat_hidden, cfg.n_learned), 0.1)
        # ---- priors of the fast layer (shared across lever slots: nothing is tied to a slot)
        nf = N_FIXED if cfg.use_fixed_features else 0
        nl = (cfg.n_learned + cfg.n_interact) if cfg.use_learned_features else 0
        s2 = torch.tensor([0.02 ** 2, 0.005 ** 2])
        lam_fixed = torch.stack([s2 / 0.10 ** 2, s2 / 0.05 ** 2, s2 / 0.10 ** 2, s2 / 0.05 ** 2], 1)  # [K, 4]
        lam = torch.cat([lam_fixed[:, :nf], (s2 / 0.10 ** 2)[:, None] * cfg.learned_prec_init * torch.ones(K_OUT, nl)], 1)
        self.mu0 = nn.Parameter(torch.zeros(K_OUT, self.F))
        self.log_lam = nn.Parameter(lam.log())
        self.log_lam_time = nn.Parameter(torch.full((K_OUT, 2), math.log(1e-2)))
        self.a0_raw = nn.Parameter(torch.full((K_OUT,), math.log(math.e - 1.0)))      # a0 = 1 + softplus = 2
        self.log_s0 = nn.Parameter(0.5 * s2.log())
        self.forget_logit = nn.Parameter(torch.tensor(math.log(cfg.forget_init / (1 - cfg.forget_init))))
        # ---- decision and value heads
        self.z_proj = nn.Linear(d, 64)
        self.move_proj = nn.Linear(d, 32)
        self.residual = small_init(mlp(64 + 32 + N_CAND, 128, 1))
        self.beta = nn.Parameter(torch.ones(()))
        self.value = small_init(mlp(d, 128, 1))
        # ---- world-model auxiliaries (representation learning): next gauge change, net over the next 8 ticks
        self.gauge_head = small_init(mlp(d, 64, 1))
        self.ahead_head = small_init(mlp(d, 64, 1))

    # ------------------------------------------------------------------------------------------- priors
    PRIOR_NAMES = ("mu0", "log_lam", "log_lam_time", "a0_raw", "log_s0", "forget_logit")

    def param_groups(self, lr: float, prior_lr_mult: float = 10.0) -> list[dict]:
        """Priors are a handful of numbers with large effects on adaptation; they learn at a higher rate."""
        pri = [p for n, p in self.named_parameters() if n in self.PRIOR_NAMES]
        rest = [p for n, p in self.named_parameters() if n not in self.PRIOR_NAMES]
        return [{"params": rest, "lr": lr}, {"params": pri, "lr": lr * prior_lr_mult, "weight_decay": 0.0}]

    def prior(self, B: int):
        lam_lever = self.log_lam.exp()                                               # [K, F]
        lam = torch.cat([lam_lever.repeat(1, self.L), torch.full((K_OUT, 1), 1e-6, device=lam_lever.device),
                         self.log_lam_time.exp()], 1)                                # [K, D]
        mu0 = torch.cat([self.mu0.repeat(1, self.L), torch.zeros(K_OUT, N_GLOBAL, device=lam.device)], 1)
        a0 = 1.0 + nn.functional.softplus(self.a0_raw)
        b0 = a0 * (2 * self.log_s0).exp()
        return mu0, lam, a0, b0

    @property
    def forget(self) -> Tensor:
        return torch.sigmoid(self.forget_logit)

    # ------------------------------------------------------------------------------------------- tokens
    def _scale(self, H: History, cut: Tensor) -> Tensor:
        """Running money scale from visible rows only: mean |revenue| + |cost| per tick (no future information)."""
        T = H.out.shape[1]
        vis = (torch.arange(T, device=cut.device)[None, :] < cut[:, None]).float()
        tot = (H.out.abs().sum(-1) * vis).sum(1) / vis.sum(1).clamp_min(1.0)
        return tot.clamp_min(1e-3)

    def tokens(self, H: History, cut: Tensor, post_lever: Tensor | None, post_global: Tensor | None):
        cfg = self.cfg
        B, T, L = H.x.shape
        dev = H.x.device
        r = cut - 1
        has = (cut > 0).float()[:, None]
        s = self._scale(H, cut)
        x_now = torch.where(cut[:, None] > 0, _rows(H.x, r), H.x0)
        e_now = torch.where(cut[:, None] > 0, _rows(H.ema, r), H.x0)
        flag = torch.zeros(B, 1, device=dev) if post_lever is None else torch.ones(B, 1, device=dev)
        pl = torch.zeros(B, L, N_POST_LEVER, device=dev) if post_lever is None else post_lever.float()
        lever = torch.stack([x_now, x_now - H.x0, e_now - H.x0, H.x0], -1)
        lever = torch.cat([lever, H.lid, pl, flag[:, None, :].expand(B, L, 1)], -1)
        g_now = _rows(H.gauges, r) * has
        g_prev = _rows(H.gauges, r - 1) * (cut > 1).float()[:, None]
        g_first = H.gauges[:, 0] * has
        gauge = torch.cat([torch.stack([g_now, g_now - g_prev, g_now - g_first], -1), H.gid], -1)
        out_now = _rows(H.out, r) * has
        tau_now = torch.where(cut > 0, _rows(H.tau, r), torch.zeros_like(s))
        funds_now = torch.where(cut > 0, _rows(H.funds, r), torch.ones_like(s))
        pg = torch.zeros(B, N_POST_GLOBAL, device=dev) if post_global is None else post_global.float()
        glob = torch.cat([tau_now[:, None], 1 - tau_now[:, None], funds_now[:, None], symlog(out_now / s[:, None]),
                          s.log()[:, None], pg, flag], -1)
        # recent event tokens: rows cut-W .. cut-1
        W = cfg.event_window
        idx = cut[:, None] - W + torch.arange(W, device=dev)[None, :]                      # [B, W]
        ev_ok = idx >= 0
        ix = idx.clamp_min(0)
        take = lambda t: t.gather(1, ix.view(B, W, *([1] * (t.dim() - 2))).expand(B, W, *t.shape[2:]))
        prev_ix = (idx - 1).clamp_min(0)
        take_prev = lambda t: t.gather(1, prev_ix.view(B, W, *([1] * (t.dim() - 2))).expand(B, W, *t.shape[2:]))
        first = (idx == 0).float()
        xr, xp = take(H.x), take_prev(H.x)
        xp = torch.where(first[..., None] > 0, H.x0[:, None, :].expand_as(xp), xp)
        dx = (xr - xp) * H.lmask[:, None, :]
        gr, gp = take(H.gauges), take_prev(H.gauges)
        dg = (gr - gp) * (1 - first[..., None]) * H.gmask[:, None, :]
        o = take(H.out) / s[:, None, None]
        net = o[..., 0] - o[..., 1]
        netp = (take_prev(H.out)[..., 0] - take_prev(H.out)[..., 1]) / s[:, None]
        pr = take(H.pred)
        has_pred = (pr[..., 1] > 0).float()
        innov = torch.where(has_pred > 0, ((o[..., 0] - o[..., 1]) * s[:, None] - pr[..., 0]) / pr[..., 1].clamp_min(1e-9),
                            torch.zeros_like(net)).clamp(-8, 8)
        ev = torch.cat([torch.einsum("bwl,bli->bwi", dx, H.lid),
                        torch.einsum("bwg,bgi->bwi", dg, H.gid),
                        torch.stack([dx.abs().sum(-1), symlog(o[..., 0]), symlog(o[..., 1]),
                                     symlog((net - netp) * (1 - first)), innov / 4,
                                     has_pred * torch.log(pr[..., 1].clamp_min(1e-9) / s[:, None]).clamp(-10, 5) / 5,
                                     take(H.tau), (cut[:, None] - 1 - idx).float() / W], -1)], -1)
        tok = torch.cat([self.lever_in(lever) + self.type_emb[0], self.gauge_in(gauge) + self.type_emb[1],
                         (self.global_in(glob) + self.type_emb[2])[:, None, :], self.event_in(ev) + self.type_emb[3]], 1)
        pad = torch.cat([H.lmask < 0.5, H.gmask < 0.5, torch.zeros(B, 1, dtype=torch.bool, device=dev), ~ev_ok], 1)
        return tok, pad

    def encode(self, H: History, cut: Tensor, post_lever=None, post_global=None):
        tok, pad = self.tokens(H, cut, post_lever, post_global)
        hs = self.out_norm(self.backbone(tok, src_key_padding_mask=pad))
        L, G = self.L, self.G
        return hs[:, :L], hs[:, L:L + G], hs[:, L + G]

    # ------------------------------------------------------------------------------------------- features
    def features(self, ctx: Tensor, d: Tensor, e: Tensor, x0: Tensor, lmask: Tensor, tau: Tensor) -> Tensor:
        """ctx [B, L, c]; d, e [B, R, L] offsets from the defaults; x0, lmask [B, L]; tau [B, R]. -> phi [B, R, D]."""
        cfg = self.cfg
        B, R, L = d.shape
        parts = []
        if cfg.use_fixed_features:
            parts.append(torch.stack([d, d * d, e, e * e], -1))
        if cfg.use_learned_features:
            c = ctx[:, None].expand(B, R, L, ctx.shape[-1])
            q = self.q_net(ctx)[:, None] * d[..., None] * lmask[:, None, :, None]          # [B, R, L, n_interact]
            u = q.sum(2, keepdim=True) - q                                              # the other levers' moves
            g_in = torch.cat([c, d[..., None], e[..., None], (d * d)[..., None], (e * e)[..., None],
                              x0[:, None, :, None].expand(B, R, L, 1), (d * e)[..., None], u], -1)
            parts.append(torch.tanh(self.g_net(g_in)))
            parts.append(d[..., None] * u)                                              # pairwise products
        per = torch.cat(parts, -1) * lmask[:, None, :, None]                    # [B, R, L, F]
        glob = torch.stack([torch.ones_like(tau), tau, tau * tau], -1)
        return torch.cat([per.reshape(B, R, L * self.F), glob], -1)

    def context(self, lever_h: Tensor) -> Tensor:
        if not self.cfg.use_context_features:
            return torch.zeros(*lever_h.shape[:2], self.ctx.out_features, device=lever_h.device)
        return self.ctx(lever_h)

    def history_features(self, H: History, ctx: Tensor) -> Tensor:
        return self.features(ctx, H.x - H.x0[:, None], H.ema - H.x0[:, None], H.x0, H.lmask, H.tau)

    def omega(self, H: History, cut: Tensor, keep: Tensor | None = None) -> Tensor:
        """Row weights: visible rows only, forgetting rho^(age); `keep` [B, T] optionally cuts rows (change points)."""
        T = H.x.shape[1]
        ar = torch.arange(T, device=cut.device)[None, :]
        vis = (ar < cut[:, None]).float()
        age = (cut[:, None] - 1 - ar).clamp_min(0).float()
        w = vis * torch.exp(age * torch.log(self.forget))
        return w if keep is None else w * keep

    def fit(self, H: History, cut: Tensor, ctx: Tensor, keep: Tensor | None = None):
        phi = self.history_features(H, ctx)
        mu0, lam, a0, b0 = self.prior(H.x.shape[0])
        post = P.fit(phi, H.out, self.omega(H, cut, keep), mu0, lam, a0, b0)
        return post, phi

    # ------------------------------------------------------------------------------------------- learning signal
    def meta_loss(self, H: History, cut: Tensor, qmask: Tensor) -> tuple[Tensor, dict, tuple]:
        """Support/query: the posterior from rows < cut predicts the outcomes of later rows (qmask [B, T]), which are
        not visible to it. Mean Student-t NLL per query outcome; gradients reach features, priors and pass 1."""
        lever_h, gauge_h, glob_h = self.encode(H, cut)
        post, phi = self.fit(H, cut, self.context(lever_h))
        loc, scale2, dof = P.predict(post, phi)
        nll = P.t_nll(H.out, loc, scale2, dof)                                  # [B, T, K]
        qm = qmask.to(nll.dtype)
        loss = (nll.sum(-1) * qm).sum() / qm.sum().clamp_min(1.0) / K_OUT
        with torch.no_grad():
            y = (H.out[..., 0] - H.out[..., 1]).double()
            yh = loc[..., 0] - loc[..., 1]
            info = {"meta_nll": float(loss), "query_rows": float(qm.sum())}
            # R^2 of the net against each task's own query mean (how much of the within-task variation is explained)
            mq = (y * qm).sum(1, keepdim=True) / qm.sum(1, keepdim=True).clamp_min(1)
            ss_res = (((y - yh) ** 2) * qm).sum(1)
            ss_tot = (((y - mq) ** 2) * qm).sum(1)
            ok = (qm.sum(1) >= 4) & (ss_tot > 0)
            info["meta_r2"] = float((1 - ss_res[ok] / ss_tot[ok]).clamp(-5, 1).mean()) if ok.any() else float("nan")
        return loss.float(), info, (lever_h, gauge_h, glob_h)

    def world_loss(self, H: History, cut: Tensor, gauge_h: Tensor, glob_h: Tensor) -> Tensor:
        """Auxiliary prediction from pass 1: each gauge's next change, and the mean net of the next 8 ticks relative to
        the last one (in the running money scale). Only rows that exist are scored."""
        B, T, G = H.gauges.shape
        nxt_ok = (cut < T) & (cut > 0)
        g_next = _rows(H.gauges, cut.clamp_max(T - 1)) - _rows(H.gauges, cut - 1)
        pg = self.gauge_head(gauge_h).squeeze(-1)
        m = H.gmask * nxt_ok[:, None].float()
        lg = (nn.functional.smooth_l1_loss(pg, g_next, reduction="none") * m).sum() / m.sum().clamp_min(1)
        s = self._scale(H, cut)
        net = (H.out[..., 0] - H.out[..., 1]) / s[:, None]
        ar = torch.arange(T, device=cut.device)[None, :]
        win = ((ar >= cut[:, None]) & (ar < cut[:, None] + 8)).float()
        ok = (win.sum(1) >= 8) & (cut > 0)
        ahead = (net * win).sum(1) / win.sum(1).clamp_min(1) - _rows(net[..., None], cut - 1).squeeze(-1)
        pa = self.ahead_head(glob_h).squeeze(-1)
        la = (nn.functional.smooth_l1_loss(pa, ahead.clamp(-20, 20), reduction="none") * ok).sum() / ok.sum().clamp_min(1)
        return lg + la

    # ------------------------------------------------------------------------------------------- the decision
    def candidate_features(self, ctx, cand: Tensor, H: History, cut: Tensor) -> Tensor:
        """Steady-state features of holding each candidate setting: d = e = a - x0 at the current time."""
        tau_now = torch.where(cut > 0, _rows(H.tau, cut - 1), torch.zeros(cand.shape[0], device=cand.device))
        d = (cand - H.x0[:, None]) * H.lmask[:, None]
        return self.features(ctx, d, d, H.x0, H.lmask, tau_now[:, None].expand(cand.shape[:2]))

    def posterior_tokens(self, post: P.Posterior, ctx: Tensor, H: History, cut: Tensor, x_now: Tensor,
                         innov_rms: Tensor):
        """Per lever: the posterior mean and sd of each outcome's change if the lever moved by -trust / +trust from
        the current settings (in units of that outcome's noise sd). Global: log noise sd relative to the money scale,
        effective rows / 64, recent innovation RMS."""
        B, L = x_now.shape
        tr = self.cfg.trust
        probes = x_now[:, None, :].repeat(1, 1 + 2 * L, 1)
        eye = torch.eye(L, device=x_now.device)
        probes[:, 1:1 + L] = (x_now[:, None] - tr * eye).clamp(-1, 1)
        probes[:, 1 + L:] = (x_now[:, None] + tr * eye).clamp(-1, 1)
        phi = self.candidate_features(ctx, probes, H, cut)
        loc, cov = P.mean_cov(post, phi)                                          # [B, Q, K], [B, K, Q, Q]
        sd_k = post.s2.sqrt()                                                     # [B, K]
        dmean = (loc[:, 1:] - loc[:, :1]) / sd_k[:, None, :]                      # [B, 2L, K]
        var = cov.diagonal(dim1=-2, dim2=-1)                                      # [B, K, Q]
        dvar = var[..., 1:] + var[..., :1] - 2 * cov[..., 0, 1:]                  # [B, K, 2L]
        dsd = dvar.clamp_min(0).sqrt().transpose(1, 2) / sd_k[:, None, :]
        lo, hi = slice(0, L), slice(L, 2 * L)
        lever = torch.cat([dmean[:, lo], dmean[:, hi], dsd[:, lo], dsd[:, hi]], -1)   # [B, L, 8]
        lever = symlog(lever) * H.lmask[..., None]
        s = self._scale(H, cut)
        glob = torch.stack([(sd_k[:, 0] / s).log().clamp(-12, 4) / 4, (sd_k[:, 1] / s).log().clamp(-12, 4) / 4,
                            post.n[:, 0].float() / 64, innov_rms], -1)
        return lever.float(), glob.float()

    def prepare(self, H: History, cut: Tensor, keep: Tensor | None = None, grad_bayes: bool = False) -> dict:
        """Pass 1 and the fast posterior. With grad_bayes False the posterior is a fixed input to what follows (the
        deliberate gradient boundary: only the support/query objective shapes the Bayesian layer)."""
        lever_h, gauge_h, glob_h = self.encode(H, cut)
        ctx = self.context(lever_h)
        ctx_b = ctx if grad_bayes else ctx.detach()
        with torch.set_grad_enabled(torch.is_grad_enabled() and grad_bayes):
            post, _ = self.fit(H, cut, ctx_b, keep)
        x_now = torch.where(cut[:, None] > 0, _rows(H.x, cut - 1), H.x0)
        return {"lever_h": lever_h, "glob_h": glob_h, "ctx": ctx_b, "post": post, "x_now": x_now,
                "grad_bayes": grad_bayes}

    def decide(self, H: History, cut: Tensor, cand: Tensor, cand_valid: Tensor, cand_flags: Tensor,
               n_rem: Tensor, keep: Tensor | None = None, innov_rms: Tensor | None = None,
               rule: str | None = None, grad_bayes: bool = False, pre: dict | None = None) -> dict:
        """Score candidate interventions. cand [B, C, L]; cand_valid [B, C]; cand_flags [B, C, 5]; n_rem [B] decisions
        left after this one. Returns logits, the Bayesian quantities, value and the posterior."""
        cfg = self.cfg
        rule = rule or cfg.explore_rule
        B, C, L = cand.shape
        pre = pre or self.prepare(H, cut, keep, grad_bayes)
        lever_h, glob_h, ctx_b, post, x_now = pre["lever_h"], pre["glob_h"], pre["ctx"], pre["post"], pre["x_now"]
        grad_bayes = pre["grad_bayes"]
        if innov_rms is None:
            innov_rms = torch.zeros(B, device=cut.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and grad_bayes):
            phi_c = self.candidate_features(ctx_b, cand, H, cut)
            loc, cov = P.mean_cov(post, phi_c)                                    # [B, C, K], [B, K, C, C]
            mean = loc[..., 0] - loc[..., 1]
            cov_net = cov[:, 0] + cov[:, 1]
            obs_var = (post.s2.sum(-1) / max(1, cfg.dwell))[:, None].expand(B, C)
            kg = P.knowledge_gradient(mean, cov_net, obs_var, cand_valid, cfg.kg_points)
            eig = P.info_gain(cov_net, obs_var)
            sd = post.s2.sum(-1).sqrt()                                           # [B]
            hold = mean[:, :1]
        if cfg.use_posterior_tokens:
            pl, pg = self.posterior_tokens(post, ctx_b, H, cut, x_now, innov_rms)
            lever_z, _, glob_z = self.encode(H, cut, pl.detach(), pg.detach())
        else:
            lever_z, glob_z = lever_h, glob_h
        if not cfg.rl_into_backbone:
            lever_z, glob_z = lever_z.detach(), glob_z.detach()
        z_mean =((mean - hold) / sd[:, None]).float().detach()
        z_sd = (cov_net.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt() / sd[:, None]).float().detach()
        z_kg = (kg / sd[:, None]).float().detach()
        nr = n_rem.float()[:, None].expand(B, C)
        if rule == "kg":
            base = z_mean + nr * z_kg
        elif rule == "eig":
            base = z_mean + cfg.eig_weight * eig.float().detach()
        elif rule in ("greedy", "thompson", "random", "fixed_design"):
            base = z_mean
        elif rule == "learned_only":
            base = torch.zeros_like(z_mean)
        else:
            raise ValueError(rule)
        moved = (cand - x_now[:, None]).abs() * H.lmask[:, None]                  # [B, C, L]
        mpool = torch.einsum("bcl,bld->bcd", moved / moved.sum(-1, keepdim=True).clamp_min(1e-6), lever_z)
        feats = torch.stack(
            [z_mean.clamp(-50, 50) / 10, symlog(z_sd), symlog(z_kg), symlog(nr * z_kg), eig.float().detach().clamp(0, 10),
             moved.sum(-1) / 2, (moved > 1e-6).float().mean(-1), nr / 64, symlog(z_mean), (z_mean > 0).float(),
             torch.log1p(nr)], -1)
        feats = torch.cat([feats, cand_flags.float()], -1)
        zin = self.z_proj(glob_z)[:, None].expand(B, C, -1)
        r = self.residual(torch.cat([zin, self.move_proj(mpool), feats], -1)).squeeze(-1)
        score = (self.beta * base if rule != "learned_only" else 0.0) + (r if cfg.use_residual or rule == "learned_only"
                                                                          else 0.0 * r)
        logits = torch.where(cand_valid, score, torch.full_like(score, -1e9))
        return {"logits": logits, "base": base, "residual": r, "mean": mean, "kg": kg, "eig": eig, "sd": sd,
                "value": self.value(glob_z).squeeze(-1), "post": post, "ctx": ctx_b, "x_now": x_now}

    # ------------------------------------------------------------------------------------------- proposals
    @torch.no_grad()
    def propose(self, H: History, cut: Tensor, post: P.Posterior, ctx: Tensor, x_now: Tensor, n_thompson: int,
                n_random: int, gen: torch.Generator | None = None, np_rng=None):
        """Candidate interventions: hold; the posterior-mean optimum and n_thompson posterior-sample optima found
        lever by lever on a grid inside the trust region (other levers at their current settings; exact for the
        additive part, then the joint setting is evaluated with the full non-additive model); every single-lever
        move to the edge of the trust region; n_random random moves of 1-3 levers."""
        cfg = self.cfg
        B, L = x_now.shape
        dev = x_now.device
        g = cfg.grid
        offs = torch.linspace(-cfg.trust, cfg.trust, g, device=dev)
        grid = x_now[:, None, None, :].repeat(1, L, g, 1)                          # [B, L, g, L]
        eye = torch.eye(L, device=dev)
        grid = grid + offs[None, None, :, None] * eye[None, :, None, :]
        grid = grid.clamp(-1, 1).reshape(B, L * g, L)
        phi = self.candidate_features(ctx, grid, H, cut).double()                  # [B, Lg, D]
        ws = [post.mu[:, None]]                                                    # [B, 1, K, D]
        if n_thompson > 0:
            ws.append(P.sample_weights(post, n_thompson, gen, np_rng))
        W = torch.cat(ws, 1)                                                       # [B, S, K, D]
        val = torch.einsum("bqd,bskd->bsqk", phi, W)
        val = (val[..., 0] - val[..., 1]).view(B, W.shape[1], L, g)                # net per grid point
        val = torch.where(H.lmask[:, None, :, None] > 0.5, val, val[..., g // 2: g // 2 + 1])
        best = val.argmax(-1)                                                      # [B, S, L]
        opt = (x_now[:, None, :] + offs[best]).clamp(-1, 1) * H.lmask[:, None] + x_now[:, None] * (1 - H.lmask[:, None])
        probes = torch.cat([(x_now[:, None] - cfg.trust * eye).clamp(-1, 1), (x_now[:, None] + cfg.trust * eye).clamp(-1, 1)], 1)
        rnd = []
        for _ in range(n_random):
            k = torch.randint(1, 4, (B, 1), device=dev, generator=gen)
            pri = torch.rand(B, L, device=dev, generator=gen) + (1 - H.lmask) * 10
            sel = (pri.argsort(-1).argsort(-1) < k).float() * H.lmask
            step = (torch.rand(B, L, device=dev, generator=gen) * 2 - 1) * cfg.trust
            rnd.append((x_now + sel * step).clamp(-1, 1))
        cands = [x_now[:, None], opt] + [probes] + ([torch.stack(rnd, 1)] if rnd else [])
        cand = torch.cat(cands, 1)
        C = cand.shape[1]
        flags = torch.zeros(B, C, N_CAND_FLAGS, device=dev)
        flags[:, 0, 0] = 1
        flags[:, 1, 1] = 1
        flags[:, 2:1 + W.shape[1], 2] = 1
        flags[:, 1 + W.shape[1]:1 + W.shape[1] + 2 * L, 3] = 1
        flags[:, 1 + W.shape[1] + 2 * L:, 4] = 1
        lever_slot = torch.cat([torch.ones(B, 1 + W.shape[1], device=dev), H.lmask.repeat(1, 2)] +
                               ([torch.ones(B, n_random, device=dev)] if n_random else []), 1)
        valid = lever_slot > 0.5
        # a probe that cannot move (lever already at the bound) duplicates hold: drop it
        same = (cand - x_now[:, None]).abs().amax(-1) < 1e-6
        valid = valid & ~(same & (torch.arange(C, device=dev)[None] > 0))
        valid[:, 0] = True
        return cand, valid, flags
