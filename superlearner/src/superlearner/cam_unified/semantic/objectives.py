"""Semantic learning objectives (spec 2.3) as real losses over the workstream-A components.

* MaskedSemanticObjective  - masked semantic prediction at the observation level: whole observations (every token
  that shares the observation's ref - its Q pooled semantic tokens and its OBS_NUM token) are replaced by a learned
  mask vector (+ the index of the token inside its observation; time/entity/kind stay in the TokenSet fields), so the
  target can only be inferred from the *other* observations; the core (any function ctx TokenSet -> ctx_out
  [B, N, D], e.g. UnifiedCoreP.forward) contextualises the set, a predictor maps the outputs at the masked positions
  to the (stop-gradient) original tokens; InfoNCE against all masked targets of the batch (no collapse to a constant
  prediction). `obs_acc` reports whether the best-scoring target belongs to the right observation.
* ToolConsequenceObjective - tool-call consequence prediction: from state features and the ACTION tokens predict the
  (projected) embedding of the observed result tokens (TOOL_RESULT / EXEC_RESULT / ERROR); InfoNCE over the batch.
* EntityGroundingObjective - contrastive mention <-> entity matching with multiple positives: an observation token
  whose content mentions an entity's persistent id (or is about it) must score that entity's token above the other
  entities of the same episode. `mention_labels` extracts the supervision from the records (string match of ids).
* CodeExecutionObjective   - code <-> execution matching: symmetric InfoNCE between code representations (ACTION /
  CODE tokens) and the representations of their real sandbox execution results.
* evidence_retrieval_loss  - supervised evidence pointer: -log pi(evidence = labelled observations | R, tool, obs).
* behaviour_cloning_loss   - -log pi(actions) of demonstration actions under a declared experimental condition
  ("demo-init:<source>", which must be listed in the run's declared conditions and must not name a privileged
  source such as a builder / reference solver / oracle); demonstration actions must come from a non-privileged
  source (ActionSource.BASELINE, e.g. the ExampleInductor; SCRIPTED builder transcripts are refused).
"""
from __future__ import annotations

import re
from typing import Callable, Collection, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import ActionSource, TokenKind, TokenSet
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


MASKABLE_KINDS = (TokenKind.OBS_SEM, TokenKind.OBS_NUM, TokenKind.RELATION, TokenKind.EVENT)


class MaskedSemanticObjective(nn.Module):
    def __init__(self, d: int, mask_frac: float = 0.15, temperature: float = 0.1, max_group: int = 16):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.group_pos = nn.Embedding(max_group, d)      # which token of its observation a masked token was
        nn.init.normal_(self.group_pos.weight, std=0.02)
        self.max_group = max_group
        self.pred = nn.Sequential(RMSNorm(d), SwiGLU(d), nn.Linear(d, d))
        self.mask_frac = mask_frac
        self.temperature = temperature

    @staticmethod
    def group_index(tokens: TokenSet) -> torch.Tensor:
        """[B, N] index of each token inside its run of tokens with the same ref (its observation)."""
        out = torch.zeros(tokens.mask.shape, dtype=torch.long)
        if tokens.refs is not None:
            for b, row in enumerate(tokens.refs):
                prev, k = object(), 0
                for j, r in enumerate(row):
                    k = k + 1 if (r is not None and r == prev) else 0
                    out[b, j] = k
                    prev = r
        return out.to(tokens.x.device)

    @staticmethod
    def observation_ids(tokens: TokenSet) -> torch.Tensor:
        """[B, N] batch-unique id of each eligible token's observation (its ref), -1 elsewhere."""
        eligible = (tokens.mask & torch.isin(tokens.kind, torch.tensor(MASKABLE_KINDS, device=tokens.kind.device)))
        el = eligible.tolist()
        ids = torch.full(tokens.mask.shape, -1, dtype=torch.long)
        seen: dict = {}
        for b in range(tokens.B):
            for j in range(tokens.N):
                if not el[b][j]:
                    continue
                ref = tokens.refs[b][j] if tokens.refs is not None else None
                key = (b, ref if ref is not None else ("tok", j))
                ids[b, j] = seen.setdefault(key, len(seen))
        return ids.to(tokens.x.device)

    def forward(self, core_fn: Callable[[TokenSet], torch.Tensor], tokens: TokenSet,
                generator: Optional[torch.Generator] = None) -> tuple:
        """core_fn(ctx TokenSet) -> ctx_out [B, N, D]. Masks whole observations (each with probability mask_frac, at
        least one per row that has any) and predicts all their tokens. Returns (loss, {"acc", "obs_acc", "n_masked",
        "n_masked_obs"})."""
        oid = self.observation_ids(tokens)
        n_obs = int(oid.max()) + 1 if bool((oid >= 0).any()) else 0
        if n_obs == 0:
            z = tokens.x.sum() * 0.0
            return z, {"acc": 0.0, "obs_acc": 0.0, "n_masked": 0, "n_masked_obs": 0}
        rdev = generator.device if generator is not None else tokens.x.device
        r = torch.rand(n_obs, generator=generator, device=rdev).to(tokens.x.device)
        pick = r < self.mask_frac
        row_of = torch.full((n_obs,), -1, dtype=torch.long, device=tokens.x.device)
        bidx = torch.arange(tokens.B, device=tokens.x.device).unsqueeze(1).expand_as(oid)
        row_of[oid[oid >= 0]] = bidx[oid >= 0]
        for b in range(tokens.B):                       # at least one masked observation per row that has any
            mine = torch.nonzero(row_of == b).flatten()
            if len(mine) and not bool(pick[mine].any()):
                pick[mine[0]] = True
        sel = (oid >= 0) & pick[oid.clamp(min=0)]
        gpos = self.group_pos(self.group_index(tokens).clamp(max=self.max_group - 1))
        masked_in = (self.mask_emb + gpos).to(tokens.x.dtype)
        x = torch.where(sel.unsqueeze(-1), masked_in, tokens.x)
        out = core_fn(TokenSet(x, tokens.mask, tokens.kind, tokens.time, tokens.entity, tokens.refs, tokens.entity2))
        pred = self.pred(out[sel])
        target = tokens.x[sel].detach()
        loss, acc = info_nce(pred, target, self.temperature)
        with torch.no_grad():
            o = oid[sel]
            best = (F.normalize(pred.float(), dim=-1) @ F.normalize(target.float(), dim=-1).t()).argmax(1)
            obs_acc = float((o[best] == o).float().mean())
        return loss, {"acc": float(acc), "obs_acc": obs_acc, "n_masked": int(sel.sum()),
                      "n_masked_obs": int(pick.sum())}


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
        self.tgt = nn.Sequential(RMSNorm(d), nn.Linear(d, d))   # target projection (CPC-style W_k), trained jointly
        self.temperature = temperature

    def predict(self, state: torch.Tensor, action: TokenSet) -> torch.Tensor:
        """state [B, D] (e.g. pooled R), action TokenSet [B, A, D] -> predicted result embedding [B, D]."""
        q = (self.q + state.unsqueeze(1)).to(action.x.dtype)
        pad = ~action.mask
        pad = pad & action.mask.any(1, keepdim=True)          # rows without action tokens attend to padding (zeros)
        a, _ = self.att(q, action.x, action.x, key_padding_mask=pad)
        return self.mlp(torch.cat([state, a.squeeze(1)], -1))

    def forward(self, state: torch.Tensor, action: TokenSet, result: TokenSet) -> tuple:
        """InfoNCE between the prediction and the projected result tokens; the negatives (other rows' results) prevent
        collapse, so gradients may also flow into the result tokens' encoder."""
        pred = self.predict(state, action)
        target = self.tgt(pool_rows(result))
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


CONDITION_RE = re.compile(r"demo-init:[a-z0-9][a-z0-9_.-]*")
PRIVILEGED_SOURCES = ("builder", "builders", "reference", "oracle", "privileged", "evaluator", "solver", "hidden",
                      "truth", "spec", "ground")
DEMO_ACTION_SOURCES = (ActionSource.BASELINE,)


def check_demo_condition(condition: str, declared: Collection[str]) -> str:
    """Validate a behaviour-cloning condition: "demo-init:<source>", listed in the run's declared conditions, and
    not naming a privileged source (architecture §11 decision 2). Returns the source name."""
    if not isinstance(condition, str) or not CONDITION_RE.fullmatch(condition):
        raise ValueError(f"behaviour cloning condition {condition!r} must look like 'demo-init:<source>'")
    if declared is None or condition not in set(declared):
        raise ValueError(f"behaviour cloning condition {condition!r} is not among the declared conditions "
                         f"{sorted(declared or [])}")
    src = condition.split(":", 1)[1]
    bad = [w for w in re.split(r"[_.-]", src) if w in PRIVILEGED_SOURCES]
    if bad:
        raise ValueError(f"demonstration source {src!r} looks privileged ({bad}); builder / reference-solver "
                         f"transcripts are never training data")
    return src


def behaviour_cloning_loss(gen, R, goal, belief, tools, obs, tool_lists, obs_lists, actions: list, condition: str,
                           declared: Collection[str], evidence: Optional[list] = None) -> tuple:
    """-mean log pi(demo actions) under the declared condition `condition` (spec: BC is a reported experimental
    condition, never silent; see check_demo_condition). Every demo action must have a non-privileged source
    (DEMO_ACTION_SOURCES). Rows with None actions are excluded."""
    check_demo_condition(condition, declared)
    for a in actions:
        if a is not None and a.source not in DEMO_ACTION_SOURCES:
            raise ValueError(f"demonstration action of tool {a.tool!r} has source {a.source.value!r}; only "
                             f"{[s.value for s in DEMO_ACTION_SOURCES]} demonstrations may be cloned")
    lp, _ = gen.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions, evidence)
    total = sum(lp.values())
    rows = [b for b, a in enumerate(actions) if a is not None]
    if not rows:
        return total.sum() * 0.0, {"n": 0, "condition": condition}
    sel = total[torch.tensor(rows, device=total.device)]
    return -sel.mean(), {"n": len(rows), "condition": condition, "logp": sel.detach()}
