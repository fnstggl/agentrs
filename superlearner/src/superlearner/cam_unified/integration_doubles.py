"""Minimal but real doubles for the integration tests of workstream I. TESTS ONLY: production code never imports this
module (checked by tests/cam_unified/integration/test_integration.py::test_production_code_never_imports_doubles).

They implement the contract protocols that `testing.py` does not cover (tokenizer, entity encoder, reasoner, beliefs,
Bayes last layer, world model, retriever, action generator, value / option / exploration heads, archive, replay,
trainer, trace recorder) plus a pure-python EnvAdapter (`ToyShopEnv`) and its GoalEvaluator. Each double does a small
genuine computation with gradients so the integration tests fail if a path is disconnected. They are not the
workstreams' implementations and do not count towards the completion manifest.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CamConfig
from .interfaces import (Action, EventRecord, ExecutionResult, FastState, GoalSpec, IndependentGoalResult,
                         Observation, ObservationBatch, ObsKind, OutcomeBatch, Provenance, TokenKind, TokenSet,
                         ToolSchema)
from .outcome_spec import stable_hash

OBS_KINDS = list(ObsKind)


def _fourier(v: torch.Tensor, n: int) -> torch.Tensor:
    """v [...] -> [..., 2n+1] Fourier features of symlog(v)/4."""
    s = torch.sign(v) * torch.log1p(v.abs()) / 4.0
    f = torch.pow(2.0, torch.arange(n, dtype=s.dtype, device=s.device))
    ang = s.unsqueeze(-1) * f
    return torch.cat([s.unsqueeze(-1), torch.sin(ang), torch.cos(ang)], -1)


class HashText(nn.Module):
    """Hashed unigram+bigram bag of words -> D (EmbeddingBag mean)."""

    def __init__(self, D: int, buckets: int = 2048):
        super().__init__()
        self.buckets = buckets
        self.emb = nn.EmbeddingBag(buckets, D, mode="mean")
        nn.init.normal_(self.emb.weight, std=0.3)

    def ids(self, text: str) -> list:
        w = re.findall(r"[a-z0-9_.]+", (text or "").lower())[:256]
        grams = w + [a + " " + b for a, b in zip(w, w[1:])]
        return [stable_hash(g) % self.buckets for g in grams] or [0]

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        dev = self.emb.weight.device
        ids, offs = [], []
        for t in texts:
            offs.append(len(ids))
            ids.extend(self.ids(t))
        if not texts:
            return self.emb.weight.new_zeros(0, self.emb.embedding_dim)
        return self.emb(torch.tensor(ids, device=dev), torch.tensor(offs, device=dev))


def _mean_tokens(ts: TokenSet) -> torch.Tensor:
    if ts.N == 0:
        return ts.x.new_zeros(ts.B, ts.x.shape[2])
    m = ts.mask.to(ts.x.dtype).unsqueeze(-1)
    return (ts.x * m).sum(1) / m.sum(1).clamp(min=1.0)


def _pack(rows: list, D: int, dev, B: int) -> TokenSet:
    """rows: [B] list of (x [D], kind, time, entity, entity2, ref) -> padded TokenSet."""
    N = max([len(r) for r in rows] + [0])
    x = torch.zeros(B, N, D, device=dev)
    mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
    kind = torch.zeros(B, N, dtype=torch.long, device=dev)
    time = torch.zeros(B, N, device=dev)
    ent = torch.full((B, N), -1, dtype=torch.long, device=dev)
    ent2 = torch.full((B, N), -1, dtype=torch.long, device=dev)
    refs = [[None] * N for _ in range(B)]
    bi, ni, vecs = [], [], []
    for b, r in enumerate(rows):
        for n, (v, k, t, e, e2, ref) in enumerate(r):
            bi.append(b)
            ni.append(n)
            vecs.append(v)
            mask[b, n] = True
            kind[b, n] = int(k)
            time[b, n] = float(t)
            ent[b, n] = int(e)
            ent2[b, n] = int(e2)
            refs[b][n] = ref
    if vecs:
        x = x.index_put((torch.tensor(bi, device=dev), torch.tensor(ni, device=dev)), torch.stack(vecs))
    return TokenSet(x, mask, kind, time, ent, refs, ent2)


# ============================================================================ semantic side (A)

class DoubleTokenizer(nn.Module):
    """ObservationTokenizer double: one token per observation (text: hashed bag of words; numeric: Fourier features of
    symlog(value) + unit text + metric text), TOOL / GOAL / ACTION tokens from hashed text and JSON."""

    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.text = HashText(D)
        self.num = nn.Linear(2 * cfg.n_fourier + 1, D)
        self.kind_emb = nn.Embedding(len(OBS_KINDS), D)
        self.tool_proj = nn.Linear(D, D)
        self.goal_proj = nn.Linear(D, D)
        self.act_proj = nn.Linear(D, D)
        self.tokens_encoded = 0

    def observations(self, obs_lists, registries, now) -> TokenSet:
        B, D = len(obs_lists), self.cfg.d_model
        dev = self.kind_emb.weight.device
        flat, meta = [], []
        for b, obs in enumerate(obs_lists):
            for i, o in enumerate(obs):
                flat.append(o)
                meta.append((b, i))
        texts, nums, kinds = [], [], []
        for o in flat:
            m = o.meta if isinstance(o.meta, dict) else {}
            if o.kind == ObsKind.NUMERIC:
                texts.append(f"{m.get('metric', '')} {o.source} {o.entity_type or ''} unit {o.unit or ''}")
            else:
                texts.append(f"{o.kind.value} {m.get('type', '')} {o.relation or ''} {o.source} {o.content[:2000]}")
            nums.append(float(o.value) if (o.kind == ObsKind.NUMERIC and o.value is not None) else 0.0)
            kinds.append(OBS_KINDS.index(o.kind))
        self.tokens_encoded += len(texts)
        rows = [[] for _ in range(B)]
        if flat:
            tx = self.text(texts) + self.kind_emb(torch.tensor(kinds, device=dev))
            nv = torch.tensor(nums, device=dev)
            isnum = torch.tensor([o.kind == ObsKind.NUMERIC for o in flat], device=dev).unsqueeze(-1)
            tx = tx + torch.where(isnum, self.num(_fourier(nv, self.cfg.n_fourier)), torch.zeros_like(tx))
            for v, o, (b, i) in zip(tx, flat, meta):
                e = registries[b].bind(o.entity, t=o.t, etype=o.entity_type)
                e2 = registries[b].bind(o.target, t=o.t) if o.target else -1
                if o.kind == ObsKind.NUMERIC:
                    k = TokenKind.OBS_NUM
                elif o.kind == ObsKind.RELATION:
                    k = TokenKind.RELATION
                elif o.kind in (ObsKind.EVENT, ObsKind.TRANSACTION, ObsKind.MESSAGE):
                    k = TokenKind.EVENT
                else:
                    k = TokenKind.OBS_SEM
                rows[b].append((v, k, o.t - now[b], e, e2, f"obs:{i}"))
        return _pack(rows, D, dev, B)

    def tools(self, tool_lists) -> TokenSet:
        B, D = len(tool_lists), self.cfg.d_model
        dev = self.kind_emb.weight.device
        flat = [(b, t) for b, tl in enumerate(tool_lists) for t in tl]
        rows = [[] for _ in range(B)]
        if flat:
            texts = [f"{t.name} {t.description} " + " ".join(sorted(t.parameters.get("properties", {})))
                     for _, t in flat]
            self.tokens_encoded += len(texts)
            x = self.tool_proj(self.text(texts))
            for v, (b, t) in zip(x, flat):
                rows[b].append((v, TokenKind.TOOL, 0.0, -1, -1, f"tool:{t.name}"))
        return _pack(rows, D, dev, B)

    def goal(self, goals) -> TokenSet:
        B, D = len(goals), self.cfg.d_model
        dev = self.kind_emb.weight.device
        texts = [g.text if g is not None else "" for g in goals]
        base = self.goal_proj(self.text(texts))
        rows = []
        for b, g in enumerate(goals):
            r = [(base[b], TokenKind.GOAL, 0.0, -1, -1, f"goal:{g.goal_id if g else ''}")]
            for k, v in sorted((g.metrics if g else {}).items()):
                vec = self.text([k])[0] + self.num(_fourier(torch.tensor(float(v), device=dev), self.cfg.n_fourier))
                r.append((vec, TokenKind.GOAL, 0.0, -1, -1, f"goal_metric:{k}"))
            rows.append(r)
        return _pack(rows, D, dev, B)

    def action(self, actions) -> TokenSet:
        B, D = len(actions), self.cfg.d_model
        dev = self.kind_emb.weight.device
        texts = [f"{a.tool} {json.dumps(a.args, sort_keys=True, default=str)}" if a is not None else ""
                 for a in actions]
        x = self.act_proj(self.text(texts))
        rows = [[(x[b], TokenKind.ACTION, 0.0, -1, -1, f"act:{a.action_id}")] if a is not None else []
                for b, a in enumerate(actions)]
        ts = _pack(rows, D, dev, B)
        if ts.N == 0:                                  # keep one (masked) slot so downstream shapes are stable
            ts = _pack([[(x[b], TokenKind.ACTION, 0.0, -1, -1, None)] for b in range(B)], D, dev, B)
            ts.mask[:] = False
        return ts


class DoubleActionGenerator(nn.Module):
    """ActionGeneratorP double: pointer over TOOL tokens; schema-walking typed heads (number/integer Gaussian, enum and
    boolean categorical); string arguments from an evidence pointer over the observation tokens (entity id or text).
    log_prob recomputes exactly what sample used (marginalising the pointer over equal string values)."""

    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.h = nn.Sequential(nn.Linear(3 * D, D), nn.Tanh())
        self.tool_q = nn.Linear(D, D)
        self.prop = HashText(D)
        self.num_head = nn.Linear(2 * D, 2)
        self.cat_q = nn.Linear(2 * D, D)
        self.bool_head = nn.Linear(2 * D, 1)
        self.ev_q = nn.Linear(2 * D, D)

    def _ctx(self, R, goal, belief):
        return self.h(torch.cat([R.float().mean(1), _mean_tokens(goal).float(), _mean_tokens(belief).float()], -1))

    @staticmethod
    def _value_of(o: Observation) -> str:
        return o.entity if o.entity else (o.content or "")[:40]

    def _heads(self, h, tools, obs, tool_lists, obs_lists, b, temperature):
        """Per-row distributions: tool logits; per chosen tool the argument head closures."""
        D = self.cfg.d_model
        sc = (tools.x[b].float() @ self.tool_q(h[b])) / math.sqrt(D) / temperature
        n_t = len(tool_lists[b])
        sc = sc[:n_t]
        return sc

    def _arg_dists(self, h_b, schema: dict, obs_x, obs_ok, temperature):
        props = schema.get("properties", {})
        out = []
        for name in sorted(props):
            p = props[name]
            pe = self.prop([f"{name} {p.get('type', '')} {p.get('description', '')}"])[0]
            hc = torch.cat([h_b, pe], -1)
            typ = p.get("type")
            if "enum" in p:
                vals = list(p["enum"])
                lv = (self.prop([str(v) for v in vals]) @ self.cat_q(hc)) / math.sqrt(len(pe)) / temperature
                out.append((name, "enum", torch.distributions.Categorical(logits=lv), vals))
            elif typ in ("number", "integer"):
                mu, ls = self.num_head(hc).unbind(-1)
                lo, hi = p.get("minimum", 0.0), p.get("maximum", 100.0)
                mid, half = (lo + hi) / 2.0, max(1e-3, (hi - lo) / 2.0)
                dist = torch.distributions.Normal(mid + half * torch.tanh(mu),
                                                  half * 0.3 * torch.exp(ls.clamp(-4, 1)) * temperature + 1e-3)
                out.append((name, typ, dist, None))
            elif typ == "boolean":
                out.append((name, "boolean", torch.distributions.Bernoulli(logits=self.bool_head(hc)[0] / temperature),
                            None))
            else:
                if obs_x is None or not bool(obs_ok.any()):
                    out.append((name, "const", None, ""))
                    continue
                lv = (obs_x @ self.ev_q(hc)) / math.sqrt(len(pe)) / temperature
                lv = torch.where(obs_ok, lv, torch.full_like(lv, -1e9))
                out.append((name, "pointer", torch.distributions.Categorical(logits=lv), None))
        return out

    def _obs_index(self, obs: TokenSet, b: int, n_obs: int):
        idx = []
        if obs.refs is not None:
            for n, r in enumerate(obs.refs[b]):
                if isinstance(r, str) and r.startswith("obs:") and bool(obs.mask[b, n]) and int(r[4:]) < n_obs:
                    idx.append((n, int(r[4:])))
        return idx

    def sample(self, R, goal, belief, tools, obs, tool_lists, obs_lists, temperature: float = 1.0,
               greedy: bool = False) -> list:
        h = self._ctx(R, goal, belief)
        out = []
        for b in range(R.shape[0]):
            if not tool_lists[b]:
                out.append(Action("noop", {}, logp=0.0, logp_parts={"tool": 0.0, "args": 0.0}))
                continue
            sc = self._heads(h, tools, obs, tool_lists, obs_lists, b, temperature)
            td = torch.distributions.Categorical(logits=sc)
            ti = int(sc.argmax()) if greedy else int(td.sample())
            tool = tool_lists[b][ti]
            lp_tool = float(td.log_prob(torch.tensor(ti, device=sc.device)))
            pos = self._obs_index(obs, b, len(obs_lists[b]))
            obs_x = obs.x[b, [p for p, _ in pos]].float() if pos else None
            obs_ok = torch.ones(len(pos), dtype=torch.bool, device=h.device) if pos else None
            args, lp_args, ents = {}, 0.0, []
            for name, typ, dist, extra in self._arg_dists(h[b], tool.parameters, obs_x, obs_ok, temperature):
                if typ == "enum":
                    k = int(dist.probs.argmax()) if greedy else int(dist.sample())
                    args[name] = extra[k]
                    lp_args += float(dist.log_prob(torch.tensor(k, device=h.device)))
                elif typ in ("number", "integer"):
                    v = dist.mean if greedy else dist.sample()
                    if typ == "integer":
                        v = torch.round(v)
                    args[name] = int(v) if typ == "integer" else round(float(v), 4)
                    lp_args += float(dist.log_prob(torch.tensor(float(args[name]), device=h.device)))
                elif typ == "boolean":
                    v = (dist.probs > 0.5).float() if greedy else dist.sample()
                    args[name] = bool(v > 0.5)
                    lp_args += float(dist.log_prob(v))
                elif typ == "pointer":
                    k = int(dist.probs.argmax()) if greedy else int(dist.sample())
                    o = obs_lists[b][pos[k][1]]
                    args[name] = self._value_of(o)
                    if o.entity:
                        ents.append(o.entity)
                    same = torch.tensor([self._value_of(obs_lists[b][i]) == args[name] for _, i in pos],
                                        device=h.device)
                    lp_args += float(torch.logsumexp(torch.where(same, dist.logits, torch.full_like(dist.logits,
                                                                                                    -1e9)), 0))
                else:
                    args[name] = extra
            out.append(Action(tool.name, args, entities=ents, logp=lp_tool + lp_args,
                              logp_parts={"tool": lp_tool, "args": lp_args}))
        return out

    def log_prob(self, R, goal, belief, tools, obs, tool_lists, obs_lists, actions) -> tuple:
        h = self._ctx(R, goal, belief)
        lps, ents = [], []
        for b, a in enumerate(actions):
            names = [t.name for t in tool_lists[b]]
            if a is None or a.tool not in names:
                lps.append(h.new_zeros(()))
                ents.append(h.new_zeros(()))
                continue
            sc = self._heads(h, tools, obs, tool_lists, obs_lists, b, 1.0)
            td = torch.distributions.Categorical(logits=sc)
            ti = names.index(a.tool)
            lp = td.log_prob(torch.tensor(ti, device=h.device))
            ent = td.entropy()
            pos = self._obs_index(obs, b, len(obs_lists[b]))
            obs_x = obs.x[b, [p for p, _ in pos]].float() if pos else None
            obs_ok = torch.ones(len(pos), dtype=torch.bool, device=h.device) if pos else None
            for name, typ, dist, extra in self._arg_dists(h[b], tool_lists[b][ti].parameters, obs_x, obs_ok, 1.0):
                v = a.args.get(name)
                if typ == "enum":
                    lp = lp + dist.log_prob(torch.tensor(extra.index(v), device=h.device))
                    ent = ent + dist.entropy()
                elif typ in ("number", "integer"):
                    lp = lp + dist.log_prob(torch.tensor(float(v), device=h.device))
                    ent = ent + dist.entropy()
                elif typ == "boolean":
                    lp = lp + dist.log_prob(torch.tensor(1.0 if v else 0.0, device=h.device))
                    ent = ent + dist.entropy()
                elif typ == "pointer":
                    same = torch.tensor([self._value_of(obs_lists[b][i]) == v for _, i in pos], device=h.device)
                    lp = lp + torch.logsumexp(torch.where(same, dist.logits, torch.full_like(dist.logits, -1e9)), 0)
                    ent = ent + dist.entropy()
            lps.append(lp)
            ents.append(ent)
        return torch.stack(lps), torch.stack(ents)


# ============================================================================ latent side (B)

class DoubleEntityEncoder(nn.Module):
    """EntityEncoderP double: ENTITY token per bound slot (scatter-mean of its observation tokens -> MLP) and a
    RELATION token per observed relation (MLP of e_i, e_j and the relation observation)."""

    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.ent = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.rel = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, obs: TokenSet) -> TokenSet:
        B, D = obs.B, self.cfg.d_model
        dev = obs.x.device
        e2 = obs._e2()
        rows = []
        for b in range(B):
            r = []
            slots = sorted({int(s) for s, m in zip(obs.entity[b].tolist(), obs.mask[b].tolist()) if m and s >= 0})
            emb = {}
            for s in slots:
                sel = obs.mask[b] & (obs.entity[b] == s)
                v = self.ent(obs.x[b][sel].float().mean(0))
                emb[s] = v
                r.append((v, TokenKind.ENTITY, float(obs.time[b][sel].max()), s, -1, f"entity_slot:{s}"))
            for n in range(obs.N):
                if obs.mask[b, n] and int(obs.kind[b, n]) == TokenKind.RELATION and int(e2[b, n]) >= 0:
                    i, j = int(obs.entity[b, n]), int(e2[b, n])
                    ei, ej = emb.get(i, obs.x.new_zeros(D).float()), emb.get(j, obs.x.new_zeros(D).float())
                    r.append((self.rel(torch.cat([ei, ej, obs.x[b, n].float()])), TokenKind.RELATION,
                              float(obs.time[b, n]), i, j, f"relation:{n}"))
            rows.append(r)
        return _pack(rows, D, dev, B)


class DoubleReasoner(nn.Module):
    """ReasonerP double: R_{j+1} = Layer(R_j + iter_emb_j, ctx) for J iterations; PonderNet halting
    p_j = lambda_j prod_{i<j}(1 - lambda_i); output mixture sum_j p_j R_j; ponder = E[iterations]."""

    def __init__(self, cfg: CamConfig, max_iters: int = 4):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.max_iters = max_iters
        self.layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 2 * D, batch_first=True, norm_first=True, dropout=0.0)
        self.iter_emb = nn.Embedding(cfg.max_reason_iters + 1, D)
        self.halt = nn.Linear(D, 1)
        self.kind = nn.Embedding(TokenKind.N_KINDS, D)

    def forward(self, Z, ctx: TokenSet, n_iters: Optional[int] = None):
        J = n_iters or self.max_iters
        B, S, D = Z.shape
        pad = torch.cat([torch.zeros(B, S, dtype=torch.bool, device=Z.device), ~ctx.mask], 1)
        cx = ctx.x + self.kind(ctx.kind)
        R, outs, lam = Z, [], []
        for j in range(J):
            h = torch.cat([R + self.iter_emb.weight[j], cx.to(R.dtype)], 1)
            R = self.layer(h, src_key_padding_mask=pad)[:, :S]
            outs.append(R)
            lam.append(torch.sigmoid(self.halt(R.mean(1)).squeeze(-1)) if j < J - 1 else torch.ones(B, device=Z.device))
        lam = torch.stack(lam, 1).float()
        surv = torch.cumprod(torch.cat([torch.ones(B, 1, device=Z.device), 1 - lam[:, :-1]], 1), 1)
        p = lam * surv
        Rm = (torch.stack(outs, 1).float() * p.view(B, J, 1, 1)).sum(1).to(Z.dtype)
        iters = (p * torch.arange(1, J + 1, device=Z.device).float()).sum(1)
        return Rm, {"halt_p": p, "iters": iters, "ponder": iters}


class DoubleBeliefs(nn.Module):
    """BeliefModuleP double: K particles; log w += log p(o | Z, a, m^k) from the particle-conditioned outcome head;
    learned proposal m <- m + g(m, outcome summary) + sigma * eps; systematic resampling when ESS < ess_frac * K;
    running surprise z-score with uncertainty expansion when z > misspec_z."""

    def __init__(self, cfg: CamConfig, n_channels: int = 16):
        super().__init__()
        D, Dp = cfg.d_model, cfg.d_particle
        self.cfg = cfg
        self.C = n_channels
        self.proj = nn.Linear(Dp + 1, D)
        self.summary = nn.Linear(D, D)
        self.g = nn.Sequential(nn.Linear(Dp + D, Dp), nn.Tanh())
        self.m0 = nn.Parameter(torch.randn(cfg.n_particles, Dp) * 0.3)

    def init_fast(self, B: int, device=None) -> FastState:
        cfg = self.cfg
        dev = device or self.m0.device
        K, Dp, F_, C, W = cfg.n_particles, cfg.d_particle, cfg.bayes_feat, self.C, cfg.misspec_window
        return FastState(self.m0.detach().unsqueeze(0).expand(B, K, Dp).clone().to(dev),
                         torch.full((B, K), -math.log(K), device=dev), torch.full((B, K, Dp), 0.05, device=dev),
                         torch.zeros(B, C, F_, device=dev, dtype=torch.float64),
                         torch.eye(F_, device=dev, dtype=torch.float64).expand(B, C, F_, F_).clone()
                         * cfg.bayes_prior_var_init,
                         torch.full((B, C), cfg.bayes_obs_var_init, device=dev, dtype=torch.float64),
                         torch.zeros(B, C, device=dev), torch.zeros(B, W, device=dev),
                         torch.zeros(B, dtype=torch.long, device=dev), [{} for _ in range(B)])

    def tokens(self, fast: FastState) -> TokenSet:
        w = torch.softmax(fast.logw.float(), -1)
        pt = self.proj(torch.cat([fast.particles.float(), fast.logw.float().unsqueeze(-1)], -1))   # [B,K,D]
        summ = self.summary((w.unsqueeze(-1) * pt).sum(1, keepdim=True))
        x = torch.cat([pt, summ], 1)
        return TokenSet.from_dense(x, TokenKind.BELIEF)

    def update(self, fast: FastState, Z_prev, act: TokenSet, outcomes: OutcomeBatch, outcome_head):
        cfg = self.cfg
        B, K, Dp = fast.particles.shape
        Zc = torch.cat([Z_prev, act.x.to(Z_prev.dtype) * act.mask.unsqueeze(-1)], 1)
        feat = outcome_head.features(Zc, outcomes)
        d = outcome_head.dist(feat, outcomes, cond=fast.particles)
        y = outcomes.y.unsqueeze(1).expand(B, K, -1)
        lp = d.log_prob(y).float()
        ll = torch.where(outcomes.y_mask.unsqueeze(1), lp, torch.zeros_like(lp)).sum(-1)            # [B,K]
        logw = torch.log_softmax(fast.logw.float() + ll, -1)
        mix = torch.logsumexp(torch.log_softmax(fast.logw.float(), -1) + ll, -1)                    # [B]
        osum = (outcomes.q.x.float() * outcomes.y_mask.unsqueeze(-1)).sum(1) / outcomes.y_mask.sum(1, keepdim=True
                                                                                                    ).clamp(min=1)
        step = self.g(torch.cat([fast.particles.float(), osum.unsqueeze(1).expand(B, K, -1)], -1))
        particles = fast.particles.float() + 0.1 * step + fast.pscale.float() * torch.randn_like(fast.particles.float())
        pscale = fast.pscale.float()
        w = logw.exp()
        ess = 1.0 / (w * w).sum(-1)
        resampled = ess < cfg.ess_frac * K
        if bool(resampled.any()):
            u = (torch.rand(B, 1, device=w.device) + torch.arange(K, device=w.device).float()) / K
            idx = torch.searchsorted(torch.cumsum(w, -1).detach().contiguous(), u.contiguous()).clamp(max=K - 1)
            rs = particles.gather(1, idx.unsqueeze(-1).expand(-1, -1, Dp))
            particles = torch.where(resampled.view(B, 1, 1), rs + 0.02 * torch.randn_like(rs), particles)
            logw = torch.where(resampled.view(B, 1), torch.full_like(logw, -math.log(K)), logw)
        surprise = torch.cat([fast.surprise[:, 1:], (-mix).detach().unsqueeze(1).float()], 1)
        n = (fast.steps + 1).clamp(max=surprise.shape[1]).float()
        hist = surprise[:, :-1]
        mu = hist.sum(1) / (n - 1).clamp(min=1)
        sd = ((hist - mu.unsqueeze(1)) ** 2).sum(1).div((n - 1).clamp(min=1)).sqrt().clamp(min=1e-3)
        z = torch.where(n > 2, (surprise[:, -1] - mu) / sd, torch.zeros_like(mu))
        expanded = z > cfg.misspec_z
        pscale = torch.where(expanded.view(B, 1, 1), pscale * 2.0, pscale)
        new = FastState(particles, logw, pscale, fast.bayes_mu, fast.bayes_P, fast.bayes_noise, fast.bayes_n, surprise,
                        fast.steps + 1, [dict(c) for c in fast.channels])
        return new, {"ess": ess.detach(), "loglik": ll, "surprise_z": z.detach(), "resampled": resampled,
                     "expanded": expanded}


class DoubleBayes(nn.Module):
    """BayesLastLayerP double: phi = tanh(W out_feat); per channel residual ~ phi^T mu_c; Joseph-form Kalman update
    in float64; predictive variance phi^T P phi + noise."""

    def __init__(self, cfg: CamConfig):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Linear(cfg.d_model, cfg.bayes_feat)

    def features(self, out_feat, outcomes):
        return torch.tanh(self.w(out_feat.float()))

    def predict(self, fast: FastState, phi, outcomes):
        B, Q, F_ = phi.shape
        ch = outcomes.channel.clamp(min=0)
        mu = fast.bayes_mu.gather(1, ch.unsqueeze(-1).expand(B, Q, F_))                         # [B,Q,F]
        P = fast.bayes_P.gather(1, ch.view(B, Q, 1, 1).expand(B, Q, F_, F_))
        ph = phi.double()
        mean = (ph * mu).sum(-1)
        var = torch.einsum("bqf,bqfg,bqg->bq", ph, P, ph) + fast.bayes_noise.gather(1, ch)
        ok = outcomes.channel >= 0
        return (torch.where(ok, mean, torch.zeros_like(mean)).float(),
                torch.where(ok, var, torch.zeros_like(var)).float())

    def update(self, fast: FastState, phi, outcomes):
        mu, P, n = fast.bayes_mu.clone(), fast.bayes_P.clone(), fast.bayes_n.clone()
        noise = fast.bayes_noise
        ph = phi.detach().double()
        B, Q, F_ = ph.shape
        I = torch.eye(F_, dtype=torch.float64, device=ph.device)
        for b in range(B):
            for q in range(Q):
                c = int(outcomes.channel[b, q])
                if c < 0 or not bool(outcomes.y_mask[b, q]):
                    continue
                f = ph[b, q]
                Pc, r = P[b, c], noise[b, c]
                s = r + f @ Pc @ f
                K = (Pc @ f) / s
                mu[b, c] = mu[b, c] + K * (float(outcomes.y[b, q]) - f @ mu[b, c])
                A = I - torch.outer(K, f)
                P[b, c] = A @ Pc @ A.T + r * torch.outer(K, K)
                n[b, c] += 1
        return FastState(fast.particles, fast.logw, fast.pscale, mu, P, fast.bayes_noise, n, fast.surprise, fast.steps,
                         [dict(c) for c in fast.channels])


# ============================================================================ world model (C)

class DoubleWorldModel(nn.Module):
    """WorldModelP double using the SHARED core: Z_hat = core(Z, [act; belief; TIME(dt); NOISE(z)]) with a categorical
    stochastic latent (straight-through). loss(seq) filters the sequence with the core and trains one-step latent
    prediction (against the detached filtered next state) plus the outcome NLL of the prior latent."""

    def __init__(self, cfg: CamConfig, core, outcome_head):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.core = core
        self.outcome_head = outcome_head
        self.logits = nn.Linear(2 * D, cfg.stoch_dim)
        self.noise = nn.Linear(cfg.stoch_dim, D)
        self.time = nn.Linear(1, D)

    def prior(self, Z, act: TokenSet, belief: TokenSet, dt, sample: bool = True):
        B = Z.shape[0]
        G, C = self.cfg.stoch_groups, self.cfg.stoch_classes
        lg = self.logits(torch.cat([Z.float().mean(1), _mean_tokens(act).float()], -1)).view(B, G, C)
        probs = torch.softmax(lg, -1)
        if sample:
            idx = torch.multinomial(probs.view(B * G, C).detach(), 1).view(B, G)
            hard = F.one_hot(idx, C).float()
            z = hard + probs - probs.detach()
        else:
            z = probs
        z = z.view(B, G * C)
        dt = torch.as_tensor(dt, device=Z.device, dtype=torch.float).view(B, 1)
        tt = TokenSet.from_dense(self.time(torch.log1p(dt.clamp(min=0) * 24)).unsqueeze(1), TokenKind.TIME)
        nt = TokenSet.from_dense(self.noise(z).unsqueeze(1), TokenKind.NOISE)
        ctx = TokenSet.cat(act, belief, tt, nt)
        ctx = TokenSet(ctx.x.to(Z.dtype), ctx.mask, ctx.kind, ctx.time, ctx.entity, ctx.refs, ctx.entity2)
        Z_hat, _ = self.core(Z, ctx)
        return Z_hat, {"logits": lg, "z": z}

    def loss(self, seq) -> dict:
        steps = seq.steps
        B = steps[0].obs.B
        Z = self.core.init_state(B, device=steps[0].obs.x.device)
        Zs, prev_act = [], None
        for st in steps:
            Z, _ = self.core(Z, TokenSet.cat(st.obs, prev_act, st.goal))
            Zs.append(Z)
            prev_act = st.act
        lat, out_nll, n = 0.0, 0.0, 0
        empty = TokenSet.empty(B, self.cfg.d_model, device=Z.device)
        for t in range(len(steps)):
            st = steps[t]
            Z_hat, _ = self.prior(Zs[t], st.act, empty, st.dt, sample=True)
            if t + 1 < len(steps):
                lat = lat + ((Z_hat - Zs[t + 1].detach()) ** 2).mean()
            feat = self.outcome_head.features(Z_hat, st.outcomes)
            lp = self.outcome_head.dist(feat, st.outcomes).log_prob(st.outcomes.y)
            m = st.outcomes.y_mask.float()
            out_nll = out_nll - (lp * m).sum() / m.sum().clamp(min=1)
            n += 1
        lat, out_nll = lat / max(1, n - 1), out_nll / max(1, n)
        return {"world": lat + out_nll, "latent": lat.detach() if torch.is_tensor(lat) else torch.tensor(lat),
                "outcome": out_nll.detach()}


# ============================================================================ memory (E)

class DoubleRetriever(nn.Module):
    """RetrieverP double: bank of archived events (fixed hashed features -> fixed random key projection); learned
    query q = W [mean Z; mean G; mean B]; cosine top-k; MEMORY tokens = learned projection of the stored features.
    loss(seq, Zs): InfoNCE that step t's query retrieves step t-1's event among the batch's events."""

    def __init__(self, cfg: CamConfig, feat_dim: int = 64):
        super().__init__()
        D, Kd = cfg.d_model, cfg.mem_key_dim
        self.cfg = cfg
        self.text = HashText(feat_dim)
        self.text.requires_grad_(False)
        self.register_buffer("key_proj", torch.randn(feat_dim, Kd) / math.sqrt(feat_dim))
        self.q = nn.Linear(3 * D, Kd)
        self.mem = nn.Linear(feat_dim, D)
        self.ev = nn.Linear(D, feat_dim)
        self.ids: list = []
        self.feats: Optional[torch.Tensor] = None
        self.n_retrieve = 0

    def _event_text(self, r: EventRecord) -> str:
        a = r.action
        res = r.result
        return (f"{a.tool if a else ''} {json.dumps(a.args, sort_keys=True, default=str) if a else ''} "
                f"ok {bool(res.ok) if res else ''} {res.error or '' if res else ''} reward {round(r.reward, 2)}")

    def add(self, records: Sequence[EventRecord]) -> None:
        if not records:
            return
        with torch.no_grad():
            f = self.text([self._event_text(r) for r in records])
        self.feats = f if self.feats is None else torch.cat([self.feats, f], 0)
        self.ids.extend(r.event_id for r in records)

    def reset_index(self) -> None:
        self.ids, self.feats = [], None

    def query(self, Z, goal: TokenSet, belief: TokenSet):
        return self.q(torch.cat([Z.float().mean(1), _mean_tokens(goal).float(), _mean_tokens(belief).float()], -1))

    def retrieve(self, q, k: int, filters=None):
        self.n_retrieve += 1
        B, D = q.shape[0], self.cfg.d_model
        dev = q.device
        if self.feats is None or not self.ids:
            return TokenSet.empty(B, D, device=dev), [[] for _ in range(B)], torch.zeros(B, 0, device=dev)
        keys = F.normalize(self.feats.to(dev) @ self.key_proj, dim=-1)
        sc = F.normalize(q.float(), dim=-1) @ keys.T                                             # [B, n]
        kk = min(k, sc.shape[1])
        top = sc.topk(kk, -1)
        x = self.mem(self.feats.to(dev)[top.indices])                                            # [B,kk,D]
        ts = TokenSet(x, torch.ones(B, kk, dtype=torch.bool, device=dev),
                      torch.full((B, kk), int(TokenKind.MEMORY), dtype=torch.long, device=dev),
                      torch.zeros(B, kk, device=dev), torch.full((B, kk), -1, dtype=torch.long, device=dev),
                      [[f"mem:{self.ids[int(i)]}" for i in top.indices[b]] for b in range(B)])
        return ts, [[self.ids[int(i)] for i in top.indices[b]] for b in range(B)], top.values

    def loss(self, seq, Zs):
        """Query at step t should match the hashed features of the event at step t-1 (contrastive over the batch)."""
        steps = seq.steps
        if len(steps) < 2:
            return Zs[0].new_zeros(()).float()
        tot, n = 0.0, 0
        for t in range(1, len(steps)):
            q = F.normalize(self.query(Zs[t], steps[t].goal, TokenSet.empty(Zs[t].shape[0], self.cfg.d_model,
                                                                              Zs[t].device)), dim=-1)
            target = F.normalize(self.ev(_mean_tokens(steps[t - 1].act).float()) @ self.key_proj, dim=-1)
            logits = q @ target.T / 0.1
            tot = tot + F.cross_entropy(logits, torch.arange(q.shape[0], device=q.device))
            n += 1
        return tot / n


class DoubleArchive:
    """ArchiveP double: in-memory, append-only, JSON round-trip on append (records must be serialisable), rejects
    non-REAL provenance."""

    def __init__(self):
        self.rows: list = []
        self.flushes = 0

    def append(self, records: list) -> None:
        for r in records:
            if r.provenance != Provenance.REAL:
                raise ValueError(f"archive rejects {r.provenance.value} records")
        self.rows.extend(EventRecord.from_dict(json.loads(r.to_json())) for r in records)

    def get(self, event_ids: list) -> list:
        want = set(event_ids)
        return [r for r in self.rows if r.event_id in want]

    def scan(self, chunk: int = 1024, **filters):
        sel = [r for r in self.rows if all(getattr(r, k) == v for k, v in filters.items())]
        for i in range(0, len(sel), chunk):
            yield sel[i: i + chunk]

    def count(self) -> int:
        return len(self.rows)

    def flush(self) -> None:
        self.flushes += 1


class DoubleReplay:
    """Replay double: uniform random windows of consecutive records from one episode of the archive."""

    def __init__(self, archive: DoubleArchive, seed: int = 0):
        self.archive = archive
        self.rng = random.Random(seed)
        self.notified = 0

    def add(self, records: list) -> None:
        self.notified += len(records)

    def sample(self, n_seqs: int, seq_len: int) -> list:
        eps = defaultdict(list)
        for r in self.archive.rows:
            eps[(r.env_id, r.episode_id)].append(r)
        eps = [sorted(v, key=lambda r: r.step) for v in eps.values() if len(v) >= 2]
        if not eps:
            return []
        picks = self.rng.sample(range(len(eps)), min(n_seqs, len(eps)))
        picks += [self.rng.randrange(len(eps)) for _ in range(n_seqs - len(picks))]
        out = []
        for i in picks:
            e = eps[i]
            L = min(seq_len, len(e))
            s = self.rng.randrange(0, len(e) - L + 1)
            out.append(e[s: s + L])
        return out


# ============================================================================ learning (D)

class DoubleValueHead(nn.Module):
    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh(), nn.Linear(D, 1))

    def forward(self, R, goal: TokenSet):
        return self.net(torch.cat([R.float().mean(1), _mean_tokens(goal).float()], -1)).squeeze(-1)


class DoubleOptionPolicy(nn.Module):
    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.pi = nn.Linear(2 * D, cfg.n_options)
        self.emb = nn.Embedding(cfg.n_options, D)
        self.beta = nn.Linear(2 * D, 1)

    def forward(self, R, goal: TokenSet):
        return self.pi(torch.cat([R.float().mean(1), _mean_tokens(goal).float()], -1))

    def termination(self, R, option):
        return torch.sigmoid(self.beta(torch.cat([R.float().mean(1), self.emb(option)], -1)).squeeze(-1) - 1.0)

    def embed(self, option):
        return self.emb(option)


class DoubleExplorationHead(nn.Module):
    def __init__(self, cfg: CamConfig):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Linear(3 * D, D)

    def forward(self, R, goal: TokenSet, belief: TokenSet) -> TokenSet:
        x = self.net(torch.cat([R.float().mean(1), _mean_tokens(goal).float(), _mean_tokens(belief).float()], -1))
        return TokenSet.from_dense(x.unsqueeze(1), TokenKind.BUDGET)


class DoubleTrainer:
    """TrainerP double: AdamW over the model; L_total = sum_k w_k L_k (cfg.loss_weights, default 1); per-objective
    gradient norm into the shared core via autograd.grad; per-component gradient norms; clip; step."""

    def __init__(self, model: nn.Module, cfg: CamConfig, lr: float = 1e-3):
        self.cfg = cfg
        self.model = model
        self.optimizers = {"all": torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                                    weight_decay=cfg.weight_decay)}
        self.history: list = []

    def reset_optimizer(self) -> None:
        for o in self.optimizers.values():
            o.state.clear()

    def step(self, losses: dict, model) -> dict:
        core_params = [p for p in model.core.parameters() if p.requires_grad]
        mets = {}
        total = 0.0
        for k, v in losses.items():
            w = float(self.cfg.loss_weights.get(k, 1.0))
            g = torch.autograd.grad(w * v, core_params, retain_graph=True, allow_unused=True)
            mets[f"grad_core/{k}"] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)
                                                      or torch.tensor(0.0)))
            total = total + w * v
        opt = self.optimizers["all"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        for name, mod in model.named_children():
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            if gs:
                mets[f"gradnorm/{name}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs)))
        before = [p.detach().clone() for p in core_params]
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], self.cfg.grad_clip)
        opt.step()
        mets["param_delta/core"] = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum()
                                                        for p, b in zip(core_params, before))))
        mets["loss_total"] = float(total.detach())
        self.history.append(mets)
        return mets


class ListTraceRecorder:
    """Trace recorder double (JSON round-trip on record; update links)."""

    def __init__(self):
        self.traces: list = []
        self.by_id: dict = {}

    def record(self, traces: list) -> None:
        for t in traces:
            d = json.loads(t.to_json())
            self.traces.append(d)
            self.by_id[d["decision_id"]] = d

    def link_updates(self, decision_ids: list, update_id: str) -> None:
        for d in decision_ids:
            if d in self.by_id:
                self.by_id[d]["update_refs"].append(update_id)


# ============================================================================ environment (pure python EnvAdapter)

TOY_TOOLS = [
    ToolSchema("set_price", "Set the unit price of one of your products in cents.",
               {"type": "object", "properties": {"product": {"type": "string"},
                                                 "price": {"type": "number", "minimum": 50, "maximum": 2000}},
                "required": ["product", "price"]}),
    ToolSchema("restock", "Buy more units of a product from the supplier (costs cents).",
               {"type": "object", "properties": {"product": {"type": "string"},
                                                 "count": {"type": "integer", "minimum": 1, "maximum": 40}},
                "required": ["product", "count"]}, duration_hint=0.02),
    ToolSchema("advertise", "Pay for an advertisement on a channel.",
               {"type": "object", "properties": {"channel": {"type": "string", "enum": ["forum", "market", "radio"]},
                                                 "budget": {"type": "integer", "minimum": 0, "maximum": 3000},
                                                 "repeat": {"type": "boolean"}},
                "required": ["channel", "budget"]}),
    ToolSchema("wait", "Let simulated time pass.",
               {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 0, "maximum": 86400}},
                "required": ["seconds"]}),
]


class ToyShopEnv:
    """Pure-python EnvAdapter: B independent shops selling products with hidden demand elasticities. Observations are
    numeric readings (stock in "count", balance and price in "cents"), sale TRANSACTION events with seq cursors,
    messages; tools are typed (string / number / integer / enum / boolean). The clock moves only in advance_time."""

    def __init__(self, B: int = 2, horizon_days: float = 5.0, seed: int = 0, n_products: int = 2):
        self.B = B
        self.horizon = horizon_days
        self.seed = seed
        self.n_products = n_products
        self.env_ids = [f"shop{b}" for b in range(B)]
        self._events: list = [[] for _ in range(B)]
        self._observe_cursor = [0] * B
        self.reset()

    def reset(self, seeds=None) -> ObservationBatch:
        seeds = list(seeds) if seeds is not None else [self.seed * 1000 + b for b in range(self.B)]
        self.rng = [random.Random(s) for s in seeds]
        self.t = [0.0] * self.B
        self.seq = [0] * self.B
        self._events = [[] for _ in range(self.B)]
        self._hidden = []        # evaluator-only mechanism parameters
        self.state = []
        for b in range(self.B):
            r = self.rng[b]
            prods = {f"prod_{r.randrange(100, 999)}_{i}": {"price": 500.0, "stock": 10} for i in range(self.n_products)}
            self._hidden.append({p: {"elasticity_secret": r.uniform(0.5, 2.5), "base_rate": r.uniform(0.5, 2.0)}
                                 for p in prods})
            self.state.append({"products": prods, "balance": 20000.0, "revenue": 0.0, "ads": 0.0})
        obs = []
        for b in range(self.B):
            o = [Observation(ObsKind.TEXT, "You run a small shop. Products sell faster when cheaper or advertised.",
                             t=0.0, source="brief")]
            o += self._readings(b)
            obs.append(o)
        self._observe_cursor = [self.seq[b] for b in range(self.B)]
        return ObservationBatch(list(self.env_ids), obs, list(self.t), [False] * self.B)

    def cursors(self) -> list:
        return [self.seq[b] for b in range(self.B)]

    def _emit(self, b: int, o: Observation) -> Observation:
        self.seq[b] += 1
        o.seq = self.seq[b]
        self._events[b].append(o)
        return o

    def _readings(self, b: int) -> list:
        st, t = self.state[b], self.t[b]
        out = [self._emit(b, Observation(ObsKind.NUMERIC, value=st["balance"], unit="cents", entity="me",
                                         entity_type="account", t=t, source="bank", meta={"metric": "balance"}))]
        for p, d in st["products"].items():
            out.append(self._emit(b, Observation(ObsKind.NUMERIC, value=float(d["stock"]), unit="count", entity=p,
                                                 entity_type="product", t=t, source="inventory",
                                                 meta={"metric": "stock"})))
            out.append(self._emit(b, Observation(ObsKind.NUMERIC, value=d["price"], unit="cents", entity=p,
                                                 entity_type="product", t=t, source="catalog",
                                                 meta={"metric": "price"})))
        return out

    def observe(self) -> ObservationBatch:
        obs = []
        for b in range(self.B):
            obs.append([o for o in self._events[b] if o.seq > self._observe_cursor[b]])
            self._observe_cursor[b] = self.seq[b]
        return ObservationBatch(list(self.env_ids), obs, list(self.t), self._done())

    def available_tools(self) -> list:
        return [list(TOY_TOOLS) for _ in range(self.B)]

    def _done(self) -> list:
        return [t >= self.horizon for t in self.t]

    def execute(self, actions) -> list:
        out = []
        for b, a in enumerate(actions):
            if a is None:
                out.append(None)
                continue
            out.append(self._exec(b, a))
        return out

    def _err(self, b, code, msg, cost=0.0):
        t = self.t[b]
        return ExecutionResult(False, [Observation(ObsKind.ERROR, f"{code}: {msg}", t=t, source="shop")],
                               error=f"{code}: {msg}", cost=cost, duration=1 / 86400, started_t=t, finished_t=t,
                               meta={"error_code": code})

    def _exec(self, b: int, a: Action) -> ExecutionResult:
        st, t = self.state[b], self.t[b]
        names = [x.name for x in TOY_TOOLS]
        if a.tool not in names:
            return self._err(b, "unknown_tool", a.tool)
        args = a.args
        if a.tool in ("set_price", "restock"):
            p = args.get("product")
            if p not in st["products"]:
                return self._err(b, "not_found", f"no product {p!r}")
        if a.tool == "set_price":
            price = float(args.get("price", -1))
            if not (50 <= price <= 2000):
                return self._err(b, "invalid_args", "price out of range")
            st["products"][args["product"]]["price"] = price
            res = {"product": args["product"], "price": price}
            cost, dur = 0.0, 60 / 86400
        elif a.tool == "restock":
            n = int(args.get("count", 0))
            cost = 120.0 * n
            if n < 1 or n > 40:
                return self._err(b, "invalid_args", "count out of range")
            if cost > st["balance"]:
                return self._err(b, "insufficient_funds", "balance too low")
            st["balance"] -= cost
            st["products"][args["product"]]["stock"] += n
            res, dur = {"product": args["product"], "stock": st["products"][args["product"]]["stock"]}, 0.02
        elif a.tool == "advertise":
            if args.get("channel") not in ("forum", "market", "radio"):
                return self._err(b, "invalid_args", "unknown channel")
            cost = float(max(0, int(args.get("budget", 0))))
            if cost > st["balance"]:
                return self._err(b, "insufficient_funds", "balance too low")
            st["balance"] -= cost
            st["ads"] += cost / 1000.0
            res, dur = {"channel": args["channel"], "budget": cost}, 300 / 86400
        else:   # wait
            s = int(args.get("seconds", 0))
            if not (0 <= s <= 86400):
                return self._err(b, "invalid_args", "seconds out of range")
            res, cost, dur = {"waited_s": s}, 0.0, s / 86400
        obs = [Observation(ObsKind.TOOL_RESULT, json.dumps(res), t=t, source=a.tool)]
        return ExecutionResult(True, obs, cost=cost, duration=dur, started_t=t, finished_t=t)

    def advance_time(self, duration) -> ObservationBatch:
        obs = []
        for b in range(self.B):
            d = max(0.0, float(duration[b]))
            if d <= 0:
                obs.append([])
                continue
            st, r = self.state[b], self.rng[b]
            new = []
            hours = max(1, int(round(d * 24)))
            for h in range(hours):
                tt = self.t[b] + d * (h + 1) / hours
                for p, dd in st["products"].items():
                    hid = self._hidden[b][p]
                    lam = hid["base_rate"] * math.exp(-hid["elasticity_secret"] * dd["price"] / 1000.0) \
                        * (1 + st["ads"]) * (d / hours) * 24
                    k = min(dd["stock"], self._poisson(r, lam))
                    if k > 0:
                        dd["stock"] -= k
                        amt = k * dd["price"]
                        st["balance"] += amt
                        st["revenue"] += amt
                        new.append(self._emit(b, Observation(ObsKind.TRANSACTION, f"sold {k} x {p} for {amt:.0f} cents",
                                                             entity=p, target="customer", t=tt, source="market",
                                                             meta={"type": "sale"})))
                st["ads"] *= 0.9
            if r.random() < 0.3:
                new.append(self._emit(b, Observation(ObsKind.MESSAGE, "A customer asks whether you restock soon.",
                                                     entity="cust_1", target="me", t=self.t[b] + d, source="inbox",
                                                     meta={"type": "message"})))
            self.t[b] += d
            new += self._readings(b)
            obs.append(new)
        return ObservationBatch(list(self.env_ids), obs, list(self.t), self._done())

    @staticmethod
    def _poisson(r: random.Random, lam: float) -> int:
        L, k, p = math.exp(-min(lam, 30.0)), 0, 1.0
        while True:
            p *= r.random()
            if p <= L:
                return k
            k += 1

    def collect_events(self, since) -> tuple:
        out, cur = [], []
        for b in range(self.B):
            s = 0 if since is None or since[b] is None else int(since[b])
            out.append([o for o in self._events[b] if o.seq > s])
            cur.append(self.seq[b])
        return out, cur

    def goals(self) -> list:
        return [GoalSpec(f"goal_{self.env_ids[b]}", "Earn 5000 cents of sales revenue by pricing, restocking and "
                         "advertising your products.", {"revenue_cents": 5000}, self.horizon, family="toy_shop")
                for b in range(self.B)]

    def close(self) -> None:
        pass


class ToyShopEvaluator:
    """GoalEvaluator: reads the hidden state; only `score` becomes a verified return increment for the learner."""

    def __init__(self, env: ToyShopEnv, target_cents: float = 5000.0):
        self.env = env
        self.target = target_cents

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        st = self.env.state[env_index]
        rev = st["revenue"]
        return IndependentGoalResult(f"goal_{self.env.env_ids[env_index]}", rev >= self.target,
                                     min(1.0, rev / self.target), {"revenue_cents": rev}, self.env.t[env_index],
                                     "toy_shop_evaluator", {"hidden": dict(self.env._hidden[env_index])})


# ============================================================================ assembly helper for tests

def double_components(cfg: CamConfig, reasoner_iters: int = 3) -> dict:
    """Instances of every component (core and outcome head from testing.py's reference doubles)."""
    from .testing import DoubleCore, DoubleOutcomeHead
    core = DoubleCore(cfg)
    head = DoubleOutcomeHead(cfg)
    return {"tokenizer": DoubleTokenizer(cfg), "core": core, "entity_encoder": DoubleEntityEncoder(cfg),
            "reasoner": DoubleReasoner(cfg, reasoner_iters), "outcome_head": head, "beliefs": DoubleBeliefs(cfg),
            "bayes": DoubleBayes(cfg), "world_model": DoubleWorldModel(cfg, core, head),
            "retriever": DoubleRetriever(cfg), "action_generator": DoubleActionGenerator(cfg),
            "value_head": DoubleValueHead(cfg), "option_policy": DoubleOptionPolicy(cfg),
            "exploration_head": DoubleExplorationHead(cfg)}


DOUBLE_PATHS = {
    "tokenizer": f"{__name__}:DoubleTokenizer", "entity_encoder": f"{__name__}:DoubleEntityEncoder",
    "reasoner": f"{__name__}:DoubleReasoner", "beliefs": f"{__name__}:DoubleBeliefs", "bayes": f"{__name__}:DoubleBayes",
    "world_model": f"{__name__}:DoubleWorldModel", "retriever": f"{__name__}:DoubleRetriever",
    "action_generator": f"{__name__}:DoubleActionGenerator", "value_head": f"{__name__}:DoubleValueHead",
    "option_policy": f"{__name__}:DoubleOptionPolicy", "exploration_head": f"{__name__}:DoubleExplorationHead",
    "core": "superlearner.cam_unified.testing:DoubleCore", "outcome_head": "superlearner.cam_unified.testing:DoubleOutcomeHead",
}
