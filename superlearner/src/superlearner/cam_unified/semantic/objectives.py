"""Semantic learning objectives (spec 2.3) as real losses over the workstream-A components.

* MaskedSemanticObjective  - masked semantic-token prediction: observation tokens are replaced by a learned mask
  vector, the core (any UnifiedCoreP: forward(Z, ctx) -> (Z, ctx_out)) contextualises the set, a predictor maps the
  outputs at the masked positions to the (stop-gradient) original tokens; InfoNCE against all masked targets of the
  batch (no collapse to a constant prediction).
* ToolConsequenceObjective - tool-call consequence prediction: from state features and the ACTION tokens predict the
  embedding of the observed result tokens (TOOL_RESULT / EXEC_RESULT / ERROR); InfoNCE over the batch.
* EntityGroundingObjective - contrastive mention <-> entity matching with multiple positives: an observation token
  whose content mentions an entity's persistent id (or is about it) must score that entity's token above the other
  entities of the same episode. `mention_labels` extracts the supervision from the records (string match of ids).
* CodeExecutionObjective   - code <-> execution matching: symmetric InfoNCE between code representations (ACTION /
  CODE tokens) and the representations of their real sandbox execution results.
* evidence_retrieval_loss  - supervised evidence pointer: -log pi(evidence = labelled observations | R, tool, obs).
* behaviour_cloning_loss   - -log pi(actions) of demonstration actions under a declared condition name (e.g.
  "demo-init:example_inductor"); refuses an undeclared condition.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import ObsKind, TokenKind, TokenSet
from .layers import RMSNorm, SwiGLU
from .tokenizer import parse_obs_ref


def info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.1, symmetric: bool = False):
    """pred, target [M, D] (row i of pred matches row i of target) -> (loss, top-1 accuracy)."""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    logits = p @ t.t() / temperature
    labels = torch.arange(p.shape[0], device=p.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(1) == labels).float().mean()
    return loss, acc


class MaskedSemanticObjective(nn.Module):
    def __init__(self, d: int, mask_frac: float = 0.15, temperature: float = 0.1):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pred = nn.Sequential(RMSNorm(d), SwiGLU(d), nn.Linear(d, d))
        self.mask_frac = mask_frac
        self.temperature = temperature

    def forward(self, core_fn: Callable[[TokenSet], torch.Tensor], tokens: TokenSet,
                generator: Optional[torch.Generator] = None) -> tuple:
        """core_fn(ctx TokenSet) -> ctx_out [B, N, D]. Returns (loss, {"acc", "n_masked"})."""
        eligible = tokens.mask & ((tokens.kind == TokenKind.OBS_SEM) | (tokens.kind == TokenKind.OBS_NUM))
        r = torch.rand(tokens.mask.shape, generator=generator, device=tokens.x.device if generator is None else
                       generator.device).to(tokens.x.device)
        sel = eligible & (r < self.mask_frac)
        for b in range(tokens.B):                       # at least one masked token per row that has any
            if bool(eligible[b].any()) and not bool(sel[b].any()):
                j = int(torch.nonzero(eligible[b])[0])
                sel[b, j] = True
        if not bool(sel.any()):
            z = tokens.x.sum() * 0.0
            return z, {"acc": 0.0, "n_masked": 0}
        x = torch.where(sel.unsqueeze(-1), self.mask_emb.to(tokens.x.dtype).expand_as(tokens.x), tokens.x)
        out = core_fn(TokenSet(x, tokens.mask, tokens.kind, tokens.time, tokens.entity, tokens.refs, tokens.entity2))
        pred = self.pred(out[sel])
        target = tokens.x[sel].detach()
        loss, acc = info_nce(pred, target, self.temperature)
        return loss, {"acc": float(acc), "n_masked": int(sel.sum())}


def pool_rows(ts: TokenSet) -> torch.Tensor:
    """Masked mean of each row's tokens -> [B, D] (zero for empty rows)."""
    m = ts.mask.unsqueeze(-1).to(ts.x.dtype)
    return (ts.x * m).sum(1) / m.sum(1).clamp_min(1.0)


class ToolConsequenceObjective(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, temperature: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(RMSNorm(2 * d), nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.temperature = temperature

    def predict(self, state: torch.Tensor, action: TokenSet) -> torch.Tensor:
        """state [B, D] (e.g. pooled R), action TokenSet [B, A, D] -> predicted result embedding [B, D]."""
        q = (self.q + state.unsqueeze(1)).to(action.x.dtype)
        pad = ~action.mask
        pad = pad & action.mask.any(1, keepdim=True)          # rows without action tokens attend to padding (zeros)
        a, _ = self.att(q, action.x, action.x, key_padding_mask=pad)
        return self.mlp(torch.cat([state, a.squeeze(1)], -1))

    def forward(self, state: torch.Tensor, action: TokenSet, result: TokenSet) -> tuple:
        pred = self.predict(state, action)
        target = pool_rows(result).detach()
        loss, acc = info_nce(pred, target, self.temperature)
        cos = F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
        return loss, {"acc": float(acc), "cos": float(cos)}


def mention_labels(tokens: TokenSet, obs_lists: list, registries: list) -> list:
    """Per row: list of (token position, set of entity slots the token's observation refers to): its own entity /
    target plus every registered entity whose persistent id occurs in the observation's content."""
    out = []
    for b in range(tokens.B):
        reg = registries[b]
        ids = sorted(reg.slot, key=len, reverse=True)
        rows = []
        for j, ref in enumerate(tokens.refs[b] if tokens.refs is not None else []):
            i = parse_obs_ref(ref)
            if i is None or not bool(tokens.mask[b, j]) or int(tokens.kind[b, j]) not in (TokenKind.OBS_SEM,
                                                                                         TokenKind.RELATION,
                                                                                         TokenKind.EVENT):
                continue
            o = obs_lists[b][i]
            pos = {reg.slot[e] for e in ids if e in (o.content or "")}
            for e in (o.entity, o.target):
                if e is not None and e in reg.slot:
                    pos.add(reg.slot[e])
            if pos:
                rows.append((j, pos))
        out.append(rows)
    return out


def slot_mean_keys(tokens: TokenSet, n_slots: int) -> tuple:
    """Entity keys from observation tokens bound to each slot (mean): ([B, n_slots, D], present [B, n_slots]).
    Stand-in input when no entity encoder output is supplied; pass workstream B's ENTITY tokens when available."""
    B, N, D = tokens.x.shape
    ent = tokens.entity.clamp(min=0)
    valid = (tokens.entity >= 0) & tokens.mask
    keys = torch.zeros(B, n_slots, D, device=tokens.x.device, dtype=tokens.x.dtype)
    cnt = torch.zeros(B, n_slots, device=tokens.x.device)
    keys = keys.scatter_add(1, ent.unsqueeze(-1).expand(B, N, D), tokens.x * valid.unsqueeze(-1).to(tokens.x.dtype))
    cnt = cnt.scatter_add(1, ent, valid.float())
    return keys / cnt.clamp_min(1.0).unsqueeze(-1).to(keys.dtype), cnt > 0


class EntityGroundingObjective(nn.Module):
    def __init__(self, d: int, temperature: float = 0.1):
        super().__init__()
        self.pq = nn.Sequential(RMSNorm(d), nn.Linear(d, d))
        self.pk = nn.Sequential(RMSNorm(d), nn.Linear(d, d))
        self.temperature = temperature

    def forward(self, mentions: TokenSet, entity_keys: torch.Tensor, entity_present: torch.Tensor,
                labels: list) -> tuple:
        """mentions [B, N, D]; entity_keys [B, E, D] (index = entity slot); entity_present [B, E] bool;
        labels from `mention_labels`. Multi-positive InfoNCE over the entities of the same episode."""
        losses, hits, n = [], 0, 0
        for b, rows in enumerate(labels):
            if not rows:
                continue
            pres = entity_present[b]
            if int(pres.sum()) < 2:
                continue
            j = torch.tensor([r[0] for r in rows], device=mentions.x.device)
            q = F.normalize(self.pq(mentions.x[b, j]).float(), dim=-1)                 # [m, D]
            k = F.normalize(self.pk(entity_keys[b]).float(), dim=-1)                   # [E, D]
            s = (q @ k.t()) / self.temperature
            s = s.masked_fill(~pres.unsqueeze(0), -torch.inf)
            pos = torch.zeros_like(s, dtype=torch.bool)
            for a, (_, slots) in enumerate(rows):
                for sl in slots:
                    if sl < pos.shape[1] and bool(pres[sl]):
                        pos[a, sl] = True
            keep = pos.any(1)
            if not bool(keep.any()):
                continue
            s, pos = s[keep], pos[keep]
            lse_all = torch.logsumexp(s, 1)
            lse_pos = torch.logsumexp(s.masked_fill(~pos, -torch.inf), 1)
            losses.append((lse_all - lse_pos).mean())
            hits += int(pos.gather(1, s.argmax(1, keepdim=True)).sum())
            n += int(keep.sum())
        if not losses:
            z = mentions.x.sum() * 0.0
            return z, {"acc": 0.0, "n": 0}
        return torch.stack(losses).mean(), {"acc": hits / max(n, 1), "n": n}


class CodeExecutionObjective(nn.Module):
    def __init__(self, d: int, temperature: float = 0.1):
        super().__init__()
        self.pc = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.pe = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.temperature = temperature

    def forward(self, code: TokenSet, execution: TokenSet) -> tuple:
        """code [M, Nc, D] (one program per row: ACTION or CODE-observation tokens) and execution [M, Ne, D] (tokens of
        that program's EXEC_RESULT / ERROR observations) -> symmetric InfoNCE (loss, {"acc"})."""
        loss, acc = info_nce(self.pc(pool_rows(code)), self.pe(pool_rows(execution)), self.temperature, symmetric=True)
        return loss, {"acc": float(acc)}


def evidence_retrieval_loss(gen, R, goal, belief, tools, obs, tool_lists, obs_lists, tool_names: list,
                            evidence: list) -> tuple:
    lp, ent = gen.evidence_log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_names, evidence)
    return -lp.mean(), {"logp": lp.detach(), "entropy": ent.detach()}


def behaviour_cloning_loss(gen, R, goal, belief, tools, obs, tool_lists, obs_lists, actions: list, condition: str,
                           evidence: Optional[list] = None) -> tuple:
    """-mean log pi(demo actions). `condition` must name the declared demonstration condition (spec: BC is a
    reported experimental condition, never silent). Rows with None actions are excluded."""
    if not condition or not isinstance(condition, str):
        raise ValueError("behaviour cloning requires a declared condition name")
    lp, _ = gen.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions, evidence)
    total = sum(lp.values())
    rows = [b for b, a in enumerate(actions) if a is not None]
    if not rows:
        return total.sum() * 0.0, {"n": 0, "condition": condition}
    sel = total[torch.tensor(rows, device=total.device)]
    return -sel.mean(), {"n": len(rows), "condition": condition, "logp": sel.detach()}
