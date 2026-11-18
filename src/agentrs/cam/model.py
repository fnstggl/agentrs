"""The Causal Agency Model: a new, fully trainable neural core (no Qwen, no LoRA).

Components (all parameters trainable, trained from scratch):
  * vocabulary + token embedding shared by every encoder and the action decoder (tied output)
  * ObsEncoder        — transformer over observation / goal / action token sequences
  * BeliefModel       — recurrent belief: b_t = GRU([obs_t, prev action, goal, elapsed time], b_{t-1})
  * Dynamics          — action-conditioned, horizon- and elapsed-time-conditioned consequence model:
                        next latent belief, observable feature distribution (Gaussian), event
                        probabilities, verified-progress distribution, termination and success
  * GoalValue         — expected return, P(eventual verified success), expected remaining steps (Gaussian)
  * Strategy (options)— learned skills: option choice π(ω|b,g) and termination β(b,ω) (subgoal done /
                        abandoned), a learned option embedding that conditions the action decoder
  * ActionDecoder     — autoregressive generator of the complete executable action (tool + all
                        arguments) under the environment's action grammar; the grammar mask is part
                        of the sampling distribution and is re-applied when re-scoring for training
  * Aux reconstruction — current observable features from the belief (anti-collapse)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from agentrs.cam.envs import ENVS, NBINS

SPECIALS = ["<pad>", "<bos>", "END", "FAM_negotiation", "FAM_product", "FAM_bizdev"]


def build_vocab() -> list[str]:
    toks = set(f"N{i}" for i in range(NBINS))
    for E in ENVS.values():
        toks |= set(E.keywords)
        e = E()
        e.reset(0)
        toks |= e.grammar().tokens()
    toks -= set(SPECIALS)
    return SPECIALS + sorted(toks)


class CAMConfig(BaseModel):
    d: int = 256
    belief: int = 512
    layers: int = 3
    heads: int = 4
    n_options: int = 8
    max_obs: int = 128
    max_act: int = 8
    n_feat: int = 24
    n_events: int = 8
    horizons: list = [1, 3, 0]      # +1 action, within +3 actions, end of task
    use_options: bool = True
    vocab: list = []


def time_feats(hours: torch.Tensor) -> torch.Tensor:
    x = torch.log1p(hours.clamp(min=0)) / 6.0
    return torch.stack([x, torch.sin(math.pi * x), torch.cos(math.pi * x), x * x], -1)


class SeqEncoder(nn.Module):
    def __init__(self, emb: nn.Embedding, d: int, layers: int, heads: int, max_len: int):
        super().__init__()
        self.emb = emb
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # ids (B, L), 0 = pad
        mask = ids != 0
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.tf(x, src_key_padding_mask=~mask)
        m = mask.float().unsqueeze(-1)
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1))


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class CAM(nn.Module):
    def __init__(self, cfg: CAMConfig):
        super().__init__()
        if not cfg.vocab:
            cfg = cfg.model_copy(update={"vocab": build_vocab()})
        self.cfg = cfg
        self.vocab = list(cfg.vocab)
        self.tok = {t: i for i, t in enumerate(self.vocab)}
        V, d, B = len(self.vocab), cfg.d, cfg.belief
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.obs_enc = SeqEncoder(self.emb, d, cfg.layers, cfg.heads, cfg.max_obs)
        self.goal_enc = SeqEncoder(self.emb, d, 1, cfg.heads, 48)
        self.act_enc = SeqEncoder(self.emb, d, 1, cfg.heads, cfg.max_act + 2)
        self.time = nn.Linear(4, 32)
        self.belief_cell = nn.GRUCell(3 * d + 32, B)
        self.belief_norm = nn.LayerNorm(B)
        # dynamics
        self.h_emb = nn.Embedding(len(cfg.horizons), 32)
        self.dyn = mlp(B + d + 32 + 32, 1024, 1024, n=2)
        self.dyn_z = nn.Linear(1024, B)
        self.dyn_feat = nn.Linear(1024, 2 * cfg.n_feat)
        self.dyn_ev = nn.Linear(1024, cfg.n_events)
        self.dyn_prog = nn.Linear(1024, 2)
        self.dyn_term = nn.Linear(1024, 1)
        self.dyn_succ = nn.Linear(1024, 1)
        # goal-value
        self.value = mlp(B + d, 512, 4, n=2)  # return, success logit, remaining-steps mu, logvar
        # strategy / skills (options)
        K = cfg.n_options if cfg.use_options else 1
        self.opt_emb = nn.Embedding(K, d)
        self.opt_pi = mlp(B + d, 256, K, n=1)
        self.opt_beta = mlp(B + d, 256, 1, n=1)
        # action decoder
        self.dec_init = nn.Linear(B + 2 * d, 512)
        self.dec_cell = nn.GRUCell(d, 512)
        self.dec_out = nn.Linear(512, d)
        # aux reconstruction of current observable features
        self.recon = mlp(B, 256, cfg.n_feat, n=1)
        nn.init.normal_(self.emb.weight, std=0.02)   # small tied embeddings: smooth initial policy over actions
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self._mask_cache: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return self.emb.weight.device

    def ids(self, seqs: list[list[str]], max_len: int) -> torch.Tensor:
        max_len = max(1, min(max_len, max(len(s) for s in seqs)))  # trim padding to the longest sequence
        out = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ii = [self.tok[t] for t in s[:max_len]]
            out[i, :len(ii)] = torch.tensor(ii, dtype=torch.long)
        return out.to(self.device)

    def n_params(self) -> dict:
        groups = {"embedding": self.emb, "obs_encoder": self.obs_enc, "goal_encoder": self.goal_enc,
                  "action_encoder": self.act_enc, "belief": nn.ModuleList([self.time, self.belief_cell, self.belief_norm]),
                  "dynamics": nn.ModuleList([self.h_emb, self.dyn, self.dyn_z, self.dyn_feat, self.dyn_ev, self.dyn_prog,
                                             self.dyn_term, self.dyn_succ]),
                  "goal_value": self.value, "strategy_options": nn.ModuleList([self.opt_emb, self.opt_pi, self.opt_beta]),
                  "action_decoder": nn.ModuleList([self.dec_init, self.dec_cell, self.dec_out]), "aux_recon": self.recon}
        seen, out = set(), {}
        for k, m in groups.items():
            n = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    n += p.numel()
            out[k] = n
        out["total"] = sum(p.numel() for p in self.parameters())
        out["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out

    # ------------------------------------------------------------------ belief
    def encode_goal(self, goal_ids):
        return self.goal_enc(goal_ids)

    def update_belief(self, b_prev, obs_ids, prev_act_ids, goal_e, elapsed_h):
        o = self.obs_enc(obs_ids)
        a = self.act_enc(prev_act_ids)
        x = torch.cat([o, a, goal_e, self.time(time_feats(elapsed_h))], -1)
        return self.belief_norm(self.belief_cell(x, b_prev))

    # ------------------------------------------------------------------ dynamics
    def predict_effects(self, b, act_ids, elapsed_h, horizon_idx):
        a = self.act_enc(act_ids)
        x = torch.cat([b, a, self.time(time_feats(elapsed_h)), self.h_emb(horizon_idx)], -1)
        h = self.dyn(x)
        fm, flv = self.dyn_feat(h).chunk(2, -1)  # Gaussian over observable features
        pm, plv = self.dyn_prog(h).chunk(2, -1)
        return {"z": self.dyn_z(h), "feat_mu": fm, "feat_logvar": flv.clamp(-5, 4), "events": self.dyn_ev(h),
                "prog_mu": pm.squeeze(-1), "prog_logvar": plv.squeeze(-1).clamp(-4, 4),
                "term": self.dyn_term(h).squeeze(-1), "success": self.dyn_succ(h).squeeze(-1)}

    # ------------------------------------------------------------------ value / strategy
    def estimate_value(self, b, goal_e):
        v = self.value(torch.cat([b, goal_e], -1))
        return {"ret": v[:, 0], "succ_logit": v[:, 1], "steps_mu": v[:, 2], "steps_logvar": v[:, 3].clamp(-6, 4)}

    def option_dists(self, b, goal_e, cur_opt):
        logits = self.opt_pi(torch.cat([b, goal_e], -1))
        beta = self.opt_beta(torch.cat([b, self.opt_emb(cur_opt)], -1)).squeeze(-1)
        return logits, beta

    # ------------------------------------------------------------------ action decoding
    def _dec_logits(self, h):
        return self.dec_out(h) @ self.emb.weight.T

    def generate_action(self, b, goal_e, opt, grammars, greedy=False, gen: Optional[torch.Generator] = None):
        """Sample complete actions token by token under each environment's grammar. Returns token
        lists, ids, per-token log-probs under the *masked* sampling distribution and the allowed-id
        sets (so that training re-scores exactly the same distribution)."""
        n = b.shape[0]
        h = torch.tanh(self.dec_init(torch.cat([b, goal_e, self.opt_emb(opt)], -1)))
        prev = torch.full((n,), self.tok["<bos>"], dtype=torch.long, device=self.device)
        toks = [[] for _ in range(n)]
        ids = [[] for _ in range(n)]
        lps = [[] for _ in range(n)]
        allowed_sets = [[] for _ in range(n)]
        done = [False] * n
        for _ in range(self.cfg.max_act):
            h = self.dec_cell(self.emb(prev), h)
            logits = self._dec_logits(h).float()
            rows = []
            for i in range(n):
                if done[i]:
                    rows.append(self._mask_row(None, ()))
                    continue
                key = (grammars[i].name, tuple(toks[i]))
                al = self._allowed_cache.get(key)
                if al is None:
                    al = [self.tok[t] for t in grammars[i].allowed(toks[i])]
                    self._allowed_cache[key] = al
                allowed_sets[i].append(al)
                rows.append(self._mask_row(key, al))
            mask = torch.stack(rows).to(self.device, non_blocking=True)
            logp = F.log_softmax(logits + mask, -1)
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
        return toks, ids, lps, allowed_sets

    @property
    def _allowed_cache(self) -> dict:
        if not hasattr(self, "_allowed"):
            self._allowed = {}
        return self._allowed

    def _mask_row(self, key, allowed) -> torch.Tensor:
        """CPU mask row (0 for allowed tokens, -inf elsewhere), cached per grammar state."""
        if key is None:
            key = ("done",)
            allowed = [self.tok["<pad>"]]
        row = self._mask_cache.get(key)
        if row is None:
            row = torch.full((len(self.vocab),), float("-inf"))
            row[list(allowed)] = 0.0
            self._mask_cache[key] = row
        return row

    def action_logprob(self, b, goal_e, opt, ids: list[list[int]], allowed: list[list[list[int]]]):
        """Differentiable log-prob of given action token ids under the same grammar masks; also the
        per-step entropy of the masked distributions."""
        n = b.shape[0]
        L = max(len(x) for x in ids)
        h = torch.tanh(self.dec_init(torch.cat([b, goal_e, self.opt_emb(opt)], -1)))
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
            h = self.dec_cell(self.emb(prev), h)
            logits = self._dec_logits(h).float() + torch.where(valid[:, k:k + 1], mask[:, k], torch.zeros_like(mask[:, k]))
            logp = F.log_softmax(logits, -1)
            lp = logp.gather(-1, tgt[:, k:k + 1]).squeeze(-1)
            p = logp.exp()
            e = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
            tot = tot + torch.where(valid[:, k], lp, torch.zeros_like(lp))
            ent = ent + torch.where(valid[:, k], e, torch.zeros_like(e))
            prev = tgt[:, k]
        return tot, ent
