"""Time-conditioned world model

    P(outcome at elapsed time h | information at the action (or at a later cutoff), complete action, h)

``TemporalWorldModel`` extends ``ContentConditionedWorldModel`` (same LM trunk; the immediate
next-observation diff head, event heads and content-outcome heads keep working unchanged) with
time-conditioned ensemble heads on the trunk's pooled state:

* one pooled forward per (context, action) serves *every* horizon — horizons only enter through a
  continuous log-time embedding (``TimeEncoder``) in small MLP heads, so asking about 50 horizons
  costs almost nothing extra, and durations between trained horizons are well defined;
* **events** (first response, positive feedback, resolution, removal, edit, ...): mixture cure model
  with a piecewise-linear hazard in log time (``event_time_model``) → P(event by h), P(ever),
  time-to-event quantiles; interval / right censoring and competing risks in the likelihood;
* **counts** (responses, reactions, views): monotone cumulative mean + negative binomial on
  increments between observations;
* **states** (visible / hidden / removed): categorical at h (not monotone: hidden posts can return);
* **continuous** outcomes (net score): Gaussian at h with learned variance;
* every head is an ensemble of ``n_heads`` members trained on bootstrap masks; member disagreement
  is reported as epistemic uncertainty next to each probability.

Updated conditional forecasts: the prompt includes the events observed between the action and the
information cutoff ``c`` (and the state at ``c``); events that have not happened by ``c`` are trained
with the left-truncated likelihood log P(label | T > c) and reported as P(T <= h | T > c).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from agentrs.content.world_model import (
    ContentConditionedWorldModel,
    ContentWorldModelConfig,
)
from agentrs.data.temporal_transition import TemporalExample
from agentrs.models.world_model import bootstrap_mask
from agentrs.temporal.event_time_model import (
    EXACT,
    INTERVAL,
    NONE,
    RIGHT,
    TAU_MAX,
    event_log_likelihood,
    negbin_log_prob,
    piecewise_linear_integral,
    time_quantile,
)
from agentrs.temporal.time_encoding import (
    TimeEncoder,
    describe_duration,
    from_log_time,
    log_time,
)

_KIND = {"none": NONE, "exact": EXACT, "interval": INTERVAL, "right_censored": RIGHT}


class TemporalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[str] = Field(default_factory=lambda: ["first_response", "positive_feedback", "resolved", "removed",
                                                       "edited"])
    counts: list[str] = Field(default_factory=lambda: ["responses", "reactions", "views"])
    states: list[str] = Field(default_factory=lambda: ["visible", "hidden", "removed"])
    continuous: list[str] = Field(default_factory=lambda: ["net_score"])


class TemporalWorldModelConfig(ContentWorldModelConfig):
    model_config = ConfigDict(extra="forbid")

    spec: TemporalSpec = Field(default_factory=TemporalSpec)
    d_latent: int = 128
    d_time: int = 32
    n_freq: int = 8
    grid_size: int = 24
    temporal_hidden: int = 128
    event_weight: float = 1.0
    count_weight: float = 0.5
    state_weight: float = 0.5
    continuous_weight: float = 0.25
    text_weight: float = 0.0  # optional generative state text at a horizon
    weighting: str = "per_action"  # per_action: every action weighs the same however often observed
    hazard_bias: float = 0.0
    count_bias: float = 1.0
    max_history_items: int = 8


@dataclass
class TemporalQuery:
    """Everything the model may condition on. Nothing after ``cutoff_elapsed_s`` may appear here."""

    context: str
    action: str
    source_type: str = "self"  # self (our executed intervention) | third_party (observed) | demo
    actor_type: str = "agent"
    goal: str | None = None  # None: unknown, never invented
    cutoff_elapsed_s: float = 0.0
    history: list[tuple[float, str]] = field(default_factory=list)  # (s after action, observed event)
    known_now: dict[str, Any] = field(default_factory=dict)  # counts / state / occurred events at cutoff

    @classmethod
    def from_example(cls, ex: TemporalExample) -> TemporalQuery:
        return cls(context=ex.initial_state, action=ex.complete_action, source_type=ex.source_type,
                   actor_type=ex.actor_type, goal=ex.goal, cutoff_elapsed_s=ex.cutoff_elapsed_s,
                   history=list(ex.history), known_now=dict(ex.known_now))


@dataclass
class HorizonPrediction:
    horizon_s: float
    event_prob: dict[str, float]  # P(event by h) (conditional on the cutoff information)
    event_std: dict[str, float]  # ensemble std (epistemic)
    count_mean: dict[str, float]  # expected cumulative count at h
    count_std: dict[str, float]
    state_probs: dict[str, float]
    continuous: dict[str, tuple[float, float, float]]  # name -> (mean, aleatoric std, epistemic std)
    expected_reward: float | None = None

    @property
    def epistemic(self) -> float:
        v = list(self.event_std.values())
        return float(sum(v) / len(v)) if v else 0.0

    def to_dict(self) -> dict:
        return {"horizon_s": self.horizon_s, "horizon": describe_duration(self.horizon_s),
                "event_prob": self.event_prob, "event_std": self.event_std, "count_mean": self.count_mean,
                "count_std": self.count_std, "state_probs": self.state_probs,
                "continuous": {k: list(v) for k, v in self.continuous.items()}, "expected_reward": self.expected_reward}


@dataclass
class TemporalPrediction:
    horizons: list[HorizonPrediction]
    event_ever: dict[str, float]  # P(event ever happens)
    median_time_s: dict[str, float]  # median time to event given that it happens (inf if beyond range)
    checkpoint: str
    cutoff_elapsed_s: float
    source_type: str
    meta: dict[str, Any] = field(default_factory=dict)

    def at(self, h: float) -> HorizonPrediction:
        return min(self.horizons, key=lambda p: abs(p.horizon_s - h))

    def to_dict(self) -> dict:
        return {"horizons": [p.to_dict() for p in self.horizons], "event_ever": self.event_ever,
                "median_time_s": {k: (None if math.isinf(v) else v) for k, v in self.median_time_s.items()},
                "checkpoint": self.checkpoint, "cutoff_elapsed_s": self.cutoff_elapsed_s,
                "source_type": self.source_type, "meta": self.meta}


class EnsembleMLP(nn.Module):
    """n_heads independent 2-layer MLPs evaluated in one einsum; input (B, d) or (B, T, d)."""

    def __init__(self, d_in: int, hidden: int, n_out: int, n_heads: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_heads, d_in, hidden) / d_in ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_heads, hidden))
        self.w2 = nn.Parameter(torch.randn(n_heads, hidden, n_out) / hidden ** 0.5 * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_heads, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            z = F.gelu(torch.einsum("bd,hdk->hbk", x, self.w1) + self.b1[:, None])
            return torch.einsum("hbk,hko->hbo", z, self.w2) + self.b2[:, None]
        z = F.gelu(torch.einsum("btd,hdk->hbtk", x, self.w1) + self.b1[:, None, None])
        return torch.einsum("hbtk,hko->hbto", z, self.w2) + self.b2[:, None, None]


class TemporalWorldModel(ContentConditionedWorldModel):
    component = "temporal_world_model"

    def __init__(self, cfg: TemporalWorldModelConfig):
        super().__init__(cfg)
        s = cfg.spec
        self.nE, self.nC, self.nS, self.nK = len(s.events), len(s.counts), len(s.states), len(s.continuous)
        d = self.backbone.hidden_size
        self.t_norm = nn.LayerNorm(d)
        self.t_proj = nn.Linear(d, cfg.d_latent)
        self.time_enc = TimeEncoder(cfg.n_freq, cfg.d_time)
        self.t_static = EnsembleMLP(cfg.d_latent, cfg.temporal_hidden, self.nE + self.nC, cfg.n_heads)
        self.n_dyn = self.nE + self.nC + self.nS + 2 * self.nK
        self.t_dyn = EnsembleMLP(cfg.d_latent + cfg.d_time, cfg.temporal_hidden, self.n_dyn, cfg.n_heads)
        self.register_buffer("grid", torch.linspace(0.0, TAU_MAX, cfg.grid_size), persistent=False)
        self.grid_dt = TAU_MAX / (cfg.grid_size - 1)
        self.checkpoint_id = "init"

    # ------------------------------------------------------------------ prompts
    def query_prompt(self, q: TemporalQuery, action_text: str | None = None) -> str:
        lines = [f"SOURCE: {q.source_type} | ACTOR: {q.actor_type}", f"GOAL: {q.goal if q.goal else 'unknown'}",
                 "CONTEXT:", q.context.strip(), f"ACTION: {action_text if action_text is not None else q.action}"]
        if q.cutoff_elapsed_s > 0 or q.history or q.known_now:
            lines.append(f"SINCE ACTION (information cutoff +{describe_duration(q.cutoff_elapsed_s)}):")
            for dt, ev in list(q.history)[-self.cfg.max_history_items:]:
                lines.append(f"- +{describe_duration(dt)} {ev}")
            if q.known_now:
                lines.append("NOW: " + " ".join(f"{k}={v}" for k, v in sorted(q.known_now.items())
                                                if not isinstance(v, (list, dict))))
        lines.append("FORECAST:")
        return "\n".join(lines)

    def _prompt_group(self, q: TemporalQuery) -> list[str]:
        from agentrs.actions.schema import try_parse_action

        a = try_parse_action(q.action)
        texts = self.encoder.chunked_action_texts(a) if a is not None else [q.action]
        if len(texts) <= 1:
            return [self.query_prompt(q)]
        return [self.query_prompt(q, t) for t in texts]

    def query_hidden(self, queries: Sequence[TemporalQuery]) -> torch.Tensor:
        groups = [self._prompt_group(q) for q in queries]
        if all(len(g) == 1 for g in groups):
            return self.backbone.pooled([g[0] for g in groups])
        return self.content_hidden(groups)

    # ------------------------------------------------------------------ heads
    def temporal_outputs(self, h: torch.Tensor, query_tau: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """h (B, d); query_tau (B, Q) extra log-times for state / continuous heads."""
        B = h.shape[0]
        z = self.t_proj(self.t_norm(h))
        stat = self.t_static(z)  # (H, B, E + C)
        G = self.grid.shape[0]
        taus = self.grid.to(h.device).unsqueeze(0).expand(B, G)
        if query_tau is not None and query_tau.shape[1] > 0:
            taus = torch.cat([taus, query_tau.to(h.device)], 1)
        T = taus.shape[1]
        phi = self.time_enc(taus)
        dyn = self.t_dyn(torch.cat([z.unsqueeze(1).expand(B, T, z.shape[-1]), phi], -1))  # (H, B, T, K)
        E, C, S, K = self.nE, self.nC, self.nS, self.nK
        hz = torch.exp((dyn[:, :, :G, :E] + self.cfg.hazard_bias).clamp(-12.0, 8.0)).permute(0, 1, 3, 2)
        cr = torch.exp((dyn[:, :, :G, E:E + C] + self.cfg.count_bias).clamp(-12.0, 10.0)).permute(0, 1, 3, 2)
        q = dyn[:, :, G:]
        return {"cure_logit": stat[..., :E], "count_log_r": stat[..., E:E + C].clamp(-6.0, 6.0),
                "hazard": hz, "count_rate": cr, "state_logits": q[..., E + C:E + C + S],
                "cont": q[..., E + C + S:].reshape(*q.shape[:-1], K, 2)}

    # ------------------------------------------------------------------ training
    def temporal_loss(self, examples: Sequence[TemporalExample]) -> tuple[torch.Tensor, dict]:
        cfg, spec = self.cfg, self.cfg.spec
        dev = self.backbone.device
        B = len(examples)
        queries = [TemporalQuery.from_example(ex) for ex in examples]
        h = self.query_hidden(queries)
        # state / continuous query times
        sq = [[(s.h_s, spec.states.index(s.state)) for s in ex.states if s.state in spec.states and
               s.h_s >= ex.cutoff_elapsed_s] for ex in examples]
        cq = [[(k, o.h_s, o.value) for k, name in enumerate(spec.continuous) for o in ex.continuous.get(name, [])
               if o.h_s >= ex.cutoff_elapsed_s] for ex in examples]
        nq = max([len(a) + len(b) for a, b in zip(sq, cq)] + [0])
        qtau = torch.zeros(B, nq)
        for b in range(B):
            ts = [t for t, _ in sq[b]] + [t for _, t, _ in cq[b]]
            if ts:
                qtau[b, :len(ts)] = log_time(ts)
        out = self.temporal_outputs(h, qtau if nq else None)
        H = out["hazard"].shape[0]
        # ---- events
        ev_kind = torch.zeros(B, self.nE, dtype=torch.long)
        ev_tau = torch.zeros(B, self.nE, 2)
        trunc_tau = torch.zeros(B, self.nE)  # left truncation at the cutoff (event not yet happened)
        for b, ex in enumerate(examples):
            c = ex.cutoff_elapsed_s
            for e, name in enumerate(spec.events):
                lab = ex.events.get(name)
                if lab is None or lab.kind == "none":
                    continue
                tr = ex.truncation_s.get(name, c) if c > 0 else 0.0
                if lab.b_s < tr - 1e-6 or (lab.kind == "right_censored" and lab.b_s <= tr):
                    continue  # nothing new after what was known at the cutoff
                ev_kind[b, e] = _KIND[lab.kind]
                a_s = lab.b_s if lab.kind in ("exact", "right_censored") else max(lab.a_s, tr)
                ev_tau[b, e] = log_time([a_s, lab.b_s])
                trunc_tau[b, e] = float(log_time(tr))
        ev_kind, ev_tau, trunc_tau = ev_kind.to(dev), ev_tau.to(dev), trunc_tau.to(dev)
        Lam, lam = piecewise_linear_integral(out["hazard"], ev_tau.unsqueeze(0).expand(H, -1, -1, -1), self.grid_dt)
        ll_ev = event_log_likelihood(out["cure_logit"], Lam, lam, ev_kind.unsqueeze(0).expand(H, -1, -1))
        Lc, _ = piecewise_linear_integral(out["hazard"], trunc_tau.unsqueeze(0).unsqueeze(-1).expand(H, -1, -1, 1),
                                          self.grid_dt)
        log_surv_c = torch.logaddexp(F.logsigmoid(-out["cure_logit"]), F.logsigmoid(out["cure_logit"]) - Lc[..., 0])
        ll_ev = torch.where(ev_kind.unsqueeze(0) != NONE, ll_ev - log_surv_c, torch.zeros_like(ll_ev))
        n_ev = (ev_kind != NONE).float().sum(-1)  # (B,)
        # ---- counts (increments)
        incs = [[(c, i.a_s, i.b_s, i.delta) for c, name in enumerate(spec.counts) for i in ex.counts.get(name, [])
                 if i.a_s >= ex.cutoff_elapsed_s - 1e-6] for ex in examples]
        M = max([sum(1 for x in v if x[0] == c) for v in incs for c in range(self.nC)] + [0])
        ll_ct = torch.zeros(H, B, device=dev)
        n_ct = torch.zeros(B, device=dev)
        if M:
            ab = torch.zeros(B, self.nC, M, 2)
            kk = torch.zeros(B, self.nC, M)
            mk = torch.zeros(B, self.nC, M)
            fill = [[0] * self.nC for _ in range(B)]
            for b, v in enumerate(incs):
                for c, a_s, b_s, d in v:
                    j = fill[b][c]
                    ab[b, c, j] = log_time([a_s, b_s])
                    kk[b, c, j] = float(d)
                    mk[b, c, j] = 1.0
                    fill[b][c] += 1
            ab, kk, mk = ab.to(dev), kk.to(dev), mk.to(dev)
            taus = ab.reshape(B, self.nC, 2 * M).unsqueeze(0).expand(H, -1, -1, -1)
            Lc2, _ = piecewise_linear_integral(out["count_rate"], taus, self.grid_dt)
            Lc2 = Lc2.reshape(H, B, self.nC, M, 2)
            mean = (Lc2[..., 1] - Lc2[..., 0]).clamp(min=1e-4)
            r = torch.exp(out["count_log_r"]).unsqueeze(-1).expand_as(mean)
            lp = negbin_log_prob(kk.unsqueeze(0).expand_as(mean), mean, r) * mk
            ll_ct = lp.sum((-1, -2))
            n_ct = mk.sum((-1, -2))
        # ---- states / continuous
        ll_st = torch.zeros(H, B, device=dev)
        n_st = torch.zeros(B, device=dev)
        ll_co = torch.zeros(H, B, device=dev)
        n_co = torch.zeros(B, device=dev)
        if nq:
            logp_states = F.log_softmax(out["state_logits"], -1)  # (H, B, Q, S)
            cont = out["cont"]
            for b in range(B):
                for j, (_, si) in enumerate(sq[b]):
                    ll_st[:, b] = ll_st[:, b] + logp_states[:, b, j, si]
                    n_st[b] += 1
                off = len(sq[b])
                for j, (k, _, v) in enumerate(cq[b]):
                    mu, logv = cont[:, b, off + j, k, 0], cont[:, b, off + j, k, 1].clamp(-6, 6)
                    ll_co[:, b] = ll_co[:, b] - 0.5 * (logv + (v - mu) ** 2 / torch.exp(logv) + math.log(2 * math.pi))
                    n_co[b] += 1
        # ---- aggregate
        if cfg.weighting == "per_action":
            norm = lambda n: 1.0 / n.clamp(min=1.0)
        else:
            norm = lambda n: torch.ones_like(n)
        total = (cfg.event_weight * ll_ev.sum(-1) * norm(n_ev) + cfg.count_weight * ll_ct * norm(n_ct)
                 + cfg.state_weight * ll_st * norm(n_st) + cfg.continuous_weight * ll_co * norm(n_co))
        w = torch.tensor([ex.weight for ex in examples], device=dev)
        has = ((n_ev + n_ct + n_st + n_co) > 0).float()
        bm = bootstrap_mask([ex.example_id for ex in examples], cfg.n_heads, cfg.bootstrap_prob, dev) * (w * has)
        if cfg.weighting == "per_action":
            loss = -(total * bm).sum() / bm.sum().clamp(min=1e-8)
        else:
            n_all = (n_ev + n_ct + n_st + n_co).clamp(min=1.0)
            loss = -(total * bm).sum() / (bm * n_all).sum().clamp(min=1e-8)
        with torch.no_grad():
            stats = {"event_nll": float(-(ll_ev.sum(-1) * bm).sum() / (bm * n_ev).sum().clamp(min=1)),
                     "count_nll": float(-(ll_ct * bm).sum() / (bm * n_ct).sum().clamp(min=1)),
                     "state_nll": float(-(ll_st * bm).sum() / (bm * n_st).sum().clamp(min=1))}
        stats.update({
                      "n_event_labels": float(n_ev.sum()), "n_count_labels": float(n_ct.sum()),
                      "n_state_labels": float(n_st.sum())})
        if cfg.text_weight > 0:
            tp, tt = [], []
            for q, ex in zip(queries, examples):
                for hs, txt in ex.text_targets:
                    if hs >= ex.cutoff_elapsed_s:
                        tp.append(self.query_prompt(q).rsplit("FORECAST:", 1)[0] + f"STATE AFTER {describe_duration(hs)}:")
                        tt.append(" " + txt + "\nEND\n")
            if tp:
                nll, cnt = self.backbone.token_nll(tp, tt)
                tl = (nll / cnt).mean()
                loss = loss + cfg.text_weight * tl
                stats["text_nll_per_token"] = float(tl.detach())
        return loss, stats

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict(self, belief_state: str | TemporalQuery, action: str | None = None,
                horizons: Sequence[float] = (60.0, 3600.0, 86400.0), relevant_context: dict | None = None,
                objective: str | None = None) -> TemporalPrediction:
        if isinstance(belief_state, TemporalQuery):
            q = belief_state
        else:
            ctx = dict(relevant_context or {})
            q = TemporalQuery(context=belief_state, action=action or "", source_type=ctx.pop("source_type", "self"),
                              actor_type=ctx.pop("actor_type", "agent"), goal=ctx.pop("goal", None),
                              cutoff_elapsed_s=float(ctx.pop("cutoff_elapsed_s", 0.0)),
                              history=list(ctx.pop("history", [])), known_now=dict(ctx.pop("known_now", {})))
        return self.predict_batch([q], horizons, objective=objective)[0]

    @torch.no_grad()
    def predict_batch(self, queries: Sequence[TemporalQuery], horizons: Sequence[float],
                      objective: str | None = None, chunk: int = 64) -> list[TemporalPrediction]:
        res: list[TemporalPrediction] = []
        for i in range(0, len(queries), chunk):
            res += self._predict_chunk(list(queries[i:i + chunk]), list(horizons), objective)
        return res

    def _predict_chunk(self, queries, horizons, objective):
        spec = self.cfg.spec
        B, Q = len(queries), len(horizons)
        h = self.query_hidden(queries)
        qtau = log_time(horizons).unsqueeze(0).expand(B, Q)
        out = self.temporal_outputs(h, qtau)
        H = out["hazard"].shape[0]
        cut = torch.tensor([float(log_time(q.cutoff_elapsed_s)) for q in queries], device=h.device)
        tq = torch.cat([cut.unsqueeze(1), qtau.to(h.device)], 1)  # (B, 1 + Q)
        Lh, _ = piecewise_linear_integral(out["hazard"], tq.unsqueeze(0).unsqueeze(2).expand(H, B, self.nE, Q + 1),
                                          self.grid_dt)
        pi = torch.sigmoid(out["cure_logit"]).unsqueeze(-1)  # (H, B, E, 1)
        Fq = pi * (-torch.expm1(-Lh))
        Fc, Fh = Fq[..., :1], Fq[..., 1:]
        Fcond = ((Fh - Fc) / (1 - Fc).clamp(min=1e-6)).clamp(0, 1)
        Lcnt, _ = piecewise_linear_integral(out["count_rate"], tq.unsqueeze(0).unsqueeze(2).expand(H, B, self.nC, Q + 1),
                                            self.grid_dt)
        inc = (Lcnt[..., 1:] - Lcnt[..., :1]).clamp(min=0)
        sp = torch.softmax(out["state_logits"], -1)  # (H, B, Q, S)
        cont = out["cont"]
        Lgrid = out["hazard"].mean(0)  # (B, E, G) mean rates for quantiles
        seg = 0.5 * (Lgrid[..., 1:] + Lgrid[..., :-1]) * self.grid_dt
        cumgrid = torch.cat([torch.zeros_like(Lgrid[..., :1]), torch.cumsum(seg, -1)], -1)
        grid = self.grid.tolist()
        preds = []
        for b, q in enumerate(queries):
            occurred = set(q.known_now.get("occurred", []) or [])
            hps = []
            for j, hs in enumerate(horizons):
                evp, evs = {}, {}
                for e, name in enumerate(spec.events):
                    if name in occurred:
                        evp[name], evs[name] = 1.0, 0.0
                        continue
                    v = Fcond[:, b, e, j] if q.cutoff_elapsed_s > 0 else Fh[:, b, e, j]
                    if hs < q.cutoff_elapsed_s:
                        v = torch.zeros_like(v)
                    evp[name], evs[name] = float(v.mean()), float(v.std()) if H > 1 else 0.0
                cm, cs = {}, {}
                for c, name in enumerate(spec.counts):
                    base = float(q.known_now.get(name, 0) or 0)
                    v = inc[:, b, c, j] if hs >= q.cutoff_elapsed_s else torch.zeros_like(inc[:, b, c, j])
                    cm[name], cs[name] = base + float(v.mean()), float(v.std()) if H > 1 else 0.0
                stp = {name: float(sp[:, b, j, s].mean()) for s, name in enumerate(spec.states)}
                co = {}
                for k, name in enumerate(spec.continuous):
                    mu = cont[:, b, j, k, 0]
                    al = torch.exp(0.5 * cont[:, b, j, k, 1].clamp(-6, 6)).mean()
                    co[name] = (float(mu.mean()), float(al), float(mu.std()) if H > 1 else 0.0)
                hp = HorizonPrediction(horizon_s=float(hs), event_prob=evp, event_std=evs, count_mean=cm, count_std=cs,
                                       state_probs=stp, continuous=co)
                if objective is not None:
                    hp.expected_reward = evp.get(objective)
                hps.append(hp)
            ever = {name: float(torch.sigmoid(out["cure_logit"][:, b, e]).mean()) for e, name in enumerate(spec.events)}
            med = {}
            for e, name in enumerate(spec.events):
                t = time_quantile(ever[name], cumgrid[b, e].tolist(), grid, 0.5)
                med[name] = float("inf") if math.isinf(t) else float(from_log_time(t))
            preds.append(TemporalPrediction(horizons=hps, event_ever=ever, median_time_s=med,
                                            checkpoint=self.checkpoint_id, cutoff_elapsed_s=q.cutoff_elapsed_s,
                                            source_type=q.source_type,
                                            meta={"n_heads": int(H), "model": self.backbone.cfg.name}))
        return preds

    @torch.no_grad()
    def sample_state_text(self, q: TemporalQuery, horizon_s: float, greedy: bool = True) -> str:
        p = self.query_prompt(q).rsplit("FORECAST:", 1)[0] + f"STATE AFTER {describe_duration(horizon_s)}:"
        out = self.backbone.generate_batch([p], max_new_tokens=self.cfg.max_diff_tokens, greedy=greedy, stop_text="\nEND")
        return out[0][0].split("\nEND")[0].strip()
