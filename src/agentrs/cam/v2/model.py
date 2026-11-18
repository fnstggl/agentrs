"""CAM-v2: Causal Agency Model with a recurrent stochastic state-space world model (RSSM), an
off-policy actor and twin goal-conditioned critics. All parameters are trained from scratch (no Qwen).

Three parameter groups with separate optimizers and hard gradient boundaries:

* World model (``wm``): token embedding, observation / goal / action encoders, RSSM
  (deterministic GRU state ``h`` + stochastic categorical state ``z``; prior p(z|h) for imagination,
  posterior q(z|h, obs) for real experience), prediction heads on [h, z, goal]: observable features,
  task status (progress, time left, resources, milestones), contract reward, continuation, clock
  duration of the action, events, constraint violation, verified success, eventual outcome
  (success / constraint violation / other failure = viability) and an ensemble of next-embedding
  predictors whose disagreement measures epistemic uncertainty. The RSSM state itself does not
  depend on the goal, so hindsight goals can be evaluated on the same states.
* Actor (``actor``): own token embedding; option (strategy) choice and termination, and the
  autoregressive grammar-masked action decoder. Input: stop-gradient [h, z] and goal.
* Critics (``critic``): own action encoder; twin Q(s, g, a), expectile value V(s, g), imagination
  value V_img(s, g); Polyak-averaged target copies of the Q critics and V_img.

Because actor and critics read stop-gradient world-model features, policy / value optimisation can
never change the world model to manufacture favourable predictions.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from agentrs.cam.envs.base import N_STATUS
from agentrs.cam.model import SeqEncoder, build_vocab, mlp, time_feats

WM_STATUS_IDX = [1, 2, 3, 4, 6, 7]  # goal-independent status entries the world model observes


class CAM2Config(BaseModel):
    d: int = 256
    deter: int = 512
    stoch: int = 16
    classes: int = 16
    hidden: int = 512
    layers: int = 3
    heads: int = 4
    n_options: int = 8
    max_obs: int = 128
    max_act: int = 8
    n_feat: int = 24
    n_events: int = 8
    n_status: int = N_STATUS
    ens: int = 4
    unimix: float = 0.01
    obs_source: str = "tokens"   # "tokens" (structured, learned from scratch) | "semantic" (frozen pretrained LM vectors)
    sem_dim: int = 0             # width of the frozen semantic vectors (semantic mode)
    vocab: list = []


class CAM2(nn.Module):
    def __init__(self, cfg: CAM2Config):
        super().__init__()
        if not cfg.vocab:
            cfg = cfg.model_copy(update={"vocab": build_vocab()})
        self.cfg = cfg
        self.vocab = list(cfg.vocab)
        self.tok = {t: i for i, t in enumerate(self.vocab)}
        V, d, H, S = len(self.vocab), cfg.d, cfg.hidden, cfg.stoch * cfg.classes
        self.F = cfg.deter + S
        F_ = self.F
        # ---------------- world model
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.semantic = cfg.obs_source == "semantic"
        if self.semantic:  # trainable semantic bridge from the frozen pretrained LM into CAM (part of the world model)
            assert cfg.sem_dim > 0, "semantic mode needs sem_dim"
            self.sem_obs = nn.Sequential(nn.LayerNorm(cfg.sem_dim), nn.Linear(cfg.sem_dim, H), nn.GELU(), nn.Linear(H, d))
            self.sem_goal = nn.Sequential(nn.LayerNorm(cfg.sem_dim), nn.Linear(cfg.sem_dim, H), nn.GELU(), nn.Linear(H, d))
            self.obs_enc = self.goal_enc = None
        else:
            self.obs_enc = SeqEncoder(self.emb, d, cfg.layers, cfg.heads, cfg.max_obs)
            self.goal_enc = SeqEncoder(self.emb, d, 1, cfg.heads, 48)
        self.wm_act_enc = SeqEncoder(self.emb, d, 1, cfg.heads, cfg.max_act + 2)
        self.time = nn.Linear(4, 32)
        self.status_in = nn.Linear(len(WM_STATUS_IDX), 32)
        self.embed = mlp(d + 64, H, d, n=1)
        self.img_in = nn.Sequential(nn.Linear(S + d, H), nn.LayerNorm(H), nn.SiLU())
        self.gru = nn.GRUCell(H, cfg.deter)
        self.prior_net = mlp(cfg.deter, H, S, n=1)
        self.post_net = mlp(cfg.deter + d, H, S, n=1)
        hin = F_ + d
        self.h_feat = mlp(hin, H, cfg.n_feat, n=1)
        self.h_status = mlp(hin, H, cfg.n_status, n=1)
        self.h_rew = mlp(hin, H, 1, n=1)
        self.h_cont = mlp(hin, H, 1, n=1)
        self.h_dt = mlp(hin, H, 1, n=1)
        self.h_ev = mlp(hin, H, cfg.n_events, n=1)
        self.h_viol = mlp(hin, H, 1, n=1)
        self.h_succ = mlp(hin, H, 1, n=1)
        self.h_outcome = mlp(hin, H, 3, n=1)          # eventual: success / constraint violation / other failure
        self.ens = nn.ModuleList([mlp(F_ + d, H, d, n=1) for _ in range(cfg.ens)])
        # ---------------- actor
        self.act_emb = nn.Embedding(V, d, padding_idx=0)
        K = max(1, cfg.n_options)
        self.opt_emb = nn.Embedding(K, d)
        self.opt_pi = mlp(F_ + d, 256, K, n=1)
        self.opt_beta = mlp(F_ + d, 256, 1, n=1)
        self.dec_init = nn.Linear(F_ + 2 * d, 512)
        self.dec_cell = nn.GRUCell(d, 512)
        self.dec_out = nn.Linear(512, d)
        # ---------------- critics
        self.crit_emb = nn.Embedding(V, d, padding_idx=0)
        self.crit_act_enc = SeqEncoder(self.crit_emb, d, 1, cfg.heads, cfg.max_act + 2)
        self.q1 = mlp(F_ + 2 * d, 512, 1, n=2)
        self.q2 = mlp(F_ + 2 * d, 512, 1, n=2)
        self.v = mlp(F_ + d, 512, 1, n=2)
        self.v_img = mlp(F_ + d, 512, 1, n=2)
        for e in (self.emb, self.act_emb, self.crit_emb):
            nn.init.normal_(e.weight, std=0.02)
            with torch.no_grad():
                e.weight[0].zero_()
        self.target = nn.ModuleDict({"crit_act_enc": copy.deepcopy(self.crit_act_enc), "q1": copy.deepcopy(self.q1),
                                     "q2": copy.deepcopy(self.q2), "v_img": copy.deepcopy(self.v_img)})
        for p in self.target.parameters():
            p.requires_grad_(False)
        self._mask_cache: dict = {}
        self._allowed: dict = {}

    # ------------------------------------------------------------------ parameter groups
    def groups(self) -> dict[str, list[nn.Parameter]]:
        bridge = [self.sem_obs, self.sem_goal] if self.semantic else [self.obs_enc, self.goal_enc]
        wm = [self.emb, *bridge, self.wm_act_enc, self.time, self.status_in, self.embed, self.img_in,
              self.gru, self.prior_net, self.post_net, self.h_feat, self.h_status, self.h_rew, self.h_cont, self.h_dt,
              self.h_ev, self.h_viol, self.h_succ, self.h_outcome, self.ens]
        actor = [self.act_emb, self.opt_emb, self.opt_pi, self.opt_beta, self.dec_init, self.dec_cell, self.dec_out]
        critic = [self.crit_emb, self.crit_act_enc, self.q1, self.q2, self.v, self.v_img]
        out = {}
        seen = set()
        for k, mods in (("wm", wm), ("actor", actor), ("critic", critic)):
            ps = []
            for m in mods:
                for p in m.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        ps.append(p)
            out[k] = ps
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        assert not rest, "every trainable parameter must belong to exactly one group"
        return out

    def n_params(self) -> dict:
        g = self.groups()
        out = {k: sum(p.numel() for p in v) for k, v in g.items()}
        out["target_copies_not_trained"] = sum(p.numel() for p in self.target.parameters())
        out["trainable"] = sum(out[k] for k in g)
        if self.semantic:
            out["semantic_bridge"] = sum(p.numel() for m in (self.sem_obs, self.sem_goal) for p in m.parameters())
        return out

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for k in ("crit_act_enc", "q1", "q2", "v_img"):
            for pt, p in zip(self.target[k].parameters(), getattr(self, k).parameters()):
                pt.lerp_(p, tau)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return self.emb.weight.device

    def ids(self, seqs: list[list[str]], max_len: int) -> torch.Tensor:
        max_len = max(1, min(max_len, max(len(s) for s in seqs)))
        out = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ii = [self.tok[t] for t in s[:max_len]]
            out[i, :len(ii)] = torch.tensor(ii, dtype=torch.long)
        return out.to(self.device)

    def pad_ids(self, rows: list[list[int]], max_len: int) -> torch.Tensor:
        L = max(1, min(max_len, max(len(r) for r in rows)))
        out = torch.zeros(len(rows), L, dtype=torch.long)
        for i, r in enumerate(rows):
            r = r[:L]
            out[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        return out.to(self.device)

    # ------------------------------------------------------------------ world model
    def encode_goal(self, goal):
        """goal: token ids (structured mode) or frozen semantic vectors (semantic mode)."""
        return self.sem_goal(goal.float()) if self.semantic else self.goal_enc(goal)

    def encode_obs(self, obs, elapsed_h, status):
        """obs: token ids (structured mode) or frozen semantic vectors (semantic mode)."""
        o = self.sem_obs(obs.float()) if self.semantic else self.obs_enc(obs)
        st = status[:, WM_STATUS_IDX]
        return self.embed(torch.cat([o, self.time(time_feats(elapsed_h)), self.status_in(st)], -1))

    def _dist(self, logits):
        C = self.cfg.classes
        p = F.softmax(logits.view(*logits.shape[:-1], self.cfg.stoch, C).float(), -1)
        return (1 - self.cfg.unimix) * p + self.cfg.unimix / C

    def sample_z(self, logits, mode: bool = False, gen: Optional[torch.Generator] = None):
        p = self._dist(logits)
        if mode:
            idx = p.argmax(-1)
        else:
            idx = torch.multinomial(p.view(-1, p.shape[-1]), 1, generator=gen).view(p.shape[:-1])
        oh = F.one_hot(idx, self.cfg.classes).float()
        oh = oh + p - p.detach()  # straight-through
        return oh.flatten(-2)

    def initial(self, n: int):
        dev = self.device
        return torch.zeros(n, self.cfg.deter, device=dev), torch.zeros(n, self.cfg.stoch * self.cfg.classes, device=dev)

    def img_step(self, h, z, act_e):
        """Prior transition: h' = GRU(h, [z, action]); returns h' and the prior logits of z'."""
        h2 = self.gru(self.img_in(torch.cat([z, act_e], -1)), h)
        return h2, self.prior_net(h2)

    def post_logits(self, h, e):
        return self.post_net(torch.cat([h, e], -1))

    def feat(self, h, z):
        return torch.cat([h, z], -1)

    def heads(self, feat, g) -> dict:
        x = torch.cat([feat, g], -1)
        return {"feat": self.h_feat(x), "status": self.h_status(x), "rew": self.h_rew(x).squeeze(-1),
                "cont": self.h_cont(x).squeeze(-1), "dt": self.h_dt(x).squeeze(-1), "ev": self.h_ev(x),
                "viol": self.h_viol(x).squeeze(-1), "succ": self.h_succ(x).squeeze(-1), "outcome": self.h_outcome(x)}

    def ens_pred(self, feat, act_e):
        x = torch.cat([feat, act_e], -1)
        return torch.stack([m(x) for m in self.ens], 0)  # (E, B, d): predicted next observation embedding

    def kl(self, post_logits, prior_logits):
        q, p = self._dist(post_logits), self._dist(prior_logits)
        return (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).sum(-1)

    # ------------------------------------------------------------------ actor
    def actor_dists(self, feat, g, cur_opt):
        logits = self.opt_pi(torch.cat([feat, g], -1))
        beta = self.opt_beta(torch.cat([feat, self.opt_emb(cur_opt)], -1)).squeeze(-1)
        return logits, beta

    def _dec_logits(self, h):
        return self.dec_out(h) @ self.act_emb.weight.T

    def _mask_row(self, key, allowed) -> torch.Tensor:
        if key is None:
            key, allowed = ("done",), [self.tok["<pad>"]]
        row = self._mask_cache.get(key)
        if row is None:
            row = torch.full((len(self.vocab),), float("-inf"))
            row[list(allowed)] = 0.0
            self._mask_cache[key] = row
        return row

    def allowed_ids(self, grammar, prefix_tokens) -> list[int]:
        key = (grammar.name, tuple(prefix_tokens))
        al = self._allowed.get(key)
        if al is None:
            al = [self.tok[t] for t in grammar.allowed(list(prefix_tokens))]
            self._allowed[key] = al
        return al

    def generate_action(self, feat, g, opt, grammars, greedy=False, gen=None):
        """Sample complete grammar-valid actions token by token; the mask is part of the distribution."""
        n = feat.shape[0]
        h = torch.tanh(self.dec_init(torch.cat([feat, g, self.opt_emb(opt)], -1)))
        prev = torch.full((n,), self.tok["<bos>"], dtype=torch.long, device=self.device)
        toks, ids, lps = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
        done = [False] * n
        for _ in range(self.cfg.max_act):
            h = self.dec_cell(self.act_emb(prev), h)
            logits = self._dec_logits(h).float()
            rows = []
            for i in range(n):
                if done[i]:
                    rows.append(self._mask_row(None, ()))
                    continue
                al = self.allowed_ids(grammars[i], toks[i])
                rows.append(self._mask_row((grammars[i].name, tuple(toks[i])), al))
            logp = F.log_softmax(logits + torch.stack(rows).to(self.device, non_blocking=True), -1)
            nxt = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1, generator=gen).squeeze(-1)
            ch = logp.gather(-1, nxt[:, None]).squeeze(-1)
            for i in range(n):
                if done[i]:
                    continue
                t = self.vocab[int(nxt[i])]
                toks[i].append(t)
                ids[i].append(int(nxt[i]))
                lps[i].append(float(ch[i]))
                if t == "END":
                    done[i] = True
            if all(done):
                break
            prev = nxt
        return toks, ids, lps

    def action_logprob(self, feat, g, opt, ids: list[list[int]], allowed: list[list[list[int]]]):
        """Differentiable log-prob (and entropy) of given action token ids under the same grammar masks."""
        n = feat.shape[0]
        L = max(len(x) for x in ids)
        h = torch.tanh(self.dec_init(torch.cat([feat, g, self.opt_emb(opt)], -1)))
        prev = torch.full((n,), self.tok["<bos>"], dtype=torch.long, device=self.device)
        tot = torch.zeros(n, device=self.device)
        ent = torch.zeros(n, device=self.device)
        tgt = torch.zeros(n, L, dtype=torch.long)
        mask = torch.full((n, L, len(self.vocab)), float("-inf"))
        valid = torch.zeros(n, L, dtype=torch.bool)
        for i in range(n):
            for k, (t, al) in enumerate(zip(ids[i], allowed[i])):
                tgt[i, k] = t
                mask[i, k, al] = 0.0
                valid[i, k] = True
        tgt, mask, valid = tgt.to(self.device), mask.to(self.device), valid.to(self.device)
        for k in range(L):
            h = self.dec_cell(self.act_emb(prev), h)
            logits = self._dec_logits(h).float() + torch.where(valid[:, k:k + 1], mask[:, k], torch.zeros_like(mask[:, k]))
            logp = F.log_softmax(logits, -1)
            lp = logp.gather(-1, tgt[:, k:k + 1]).squeeze(-1)
            p = logp.exp()
            e = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
            tot = tot + torch.where(valid[:, k], lp, torch.zeros_like(lp))
            ent = ent + torch.where(valid[:, k], e, torch.zeros_like(e))
            prev = tgt[:, k]
        return tot, ent

    # ------------------------------------------------------------------ critics
    def q_values(self, feat, g, act_ids, target: bool = False):
        enc = self.target["crit_act_enc"] if target else self.crit_act_enc
        q1 = self.target["q1"] if target else self.q1
        q2 = self.target["q2"] if target else self.q2
        x = torch.cat([feat, g, enc(act_ids)], -1)
        return q1(x).squeeze(-1), q2(x).squeeze(-1)

    def value(self, feat, g):
        return self.v(torch.cat([feat, g], -1)).squeeze(-1)

    def value_img(self, feat, g, target: bool = False):
        net = self.target["v_img"] if target else self.v_img
        return net(torch.cat([feat, g], -1)).squeeze(-1)
