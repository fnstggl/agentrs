"""Causal evidence provenance (spec 6.5) and intervention-sensitive representations (spec 13.4, W03).

* ``CausalLedger`` (harness / actor side, plain records): records per event the intervention assignment and its
  known randomisation probability (CAM's own sampled actions have an exactly known assignment probability
  exp(behaviour log-prob); a planner's deterministic argmax has probability 1), the simultaneous interventions
  (other actions of the same environment whose active window overlaps), delays since the causing action for delayed
  / attributed outcomes, missingness and evidence reliability, and a confounding flag. Everything is derived from
  information the agent actually has (its own actions and observations) -- never from evaluator-only state.
* ``provenance_features(record)`` -> the fixed feature vector ``PROV_FEATURES`` for one EventRecord.
* ``CausalProvenance`` (nn.Module) appends those features to EVENT tokens: x <- x + MLP(features) on tokens of kind
  EVENT (found through ``TokenSet.refs``), leaving every other token unchanged.
* ``InterventionContrastive`` (nn.Module, W03): states whose observed responses to the *same* intervention are
  similar are pulled together (soft InfoNCE targets from response similarity), and similar-looking states with
  different responses to the same intervention are pushed apart (hard-negative margin). Only observed
  intervention consequences are used; no hidden causal labels.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import (ActionSource, EventRecord, EvidenceKind, ObsKind, Provenance, TokenKind, TokenSet)
from ._util import MLP, AttnPool

PROV_FEATURES = ("rand_p", "rand_known", "log_n_simultaneous", "log_delay_days", "delay_known", "missing_frac",
                 "result_missing", "ev_controlled", "ev_observational", "ev_inferred", "confounding_possible",
                 "counterfactual", "reliability", "reliability_known", "attributed")


def _assignment_probability(ev: EventRecord) -> Optional[float]:
    """Known probability with which the executed intervention was assigned, if any."""
    if ev.randomization_p is not None:
        return float(ev.randomization_p)
    a = ev.action
    if a is None:
        return None
    if a.proposal and a.proposal.get("deterministic"):
        return 1.0
    lp = ev.behavior_logp if ev.behavior_logp is not None else a.logp
    if lp is not None and a.source in (ActionSource.CAM_DECODER, ActionSource.FROZEN_PROPOSAL):
        return float(math.exp(min(0.0, lp)))
    return None


class CausalLedger:
    """Per-environment intervention ledger that fills the causal fields of EventRecords as they are produced.

    ``effect_window_days``: an intervention counts as active (a possible simultaneous cause) from its start until
    its execution finishes plus this window."""

    def __init__(self, effect_window_days: float = 0.0):
        self.window = float(effect_window_days)
        self.active: Dict[str, List[tuple]] = {}          # env -> [(action_id, event_id, start, end)]
        self.event_t: Dict[str, float] = {}               # event id -> t_sim
        self.action_event: Dict[str, str] = {}            # action id -> event id

    def _interval(self, ev: EventRecord) -> tuple:
        start = ev.t_sim
        end = start
        if ev.result is not None:
            start = ev.result.started_t or start
            end = max(start, ev.result.finished_t or start, start + (ev.result.duration or 0.0))
        return start, end + self.window

    def record(self, ev: EventRecord) -> EventRecord:
        """Fill randomization_p, simultaneous, attributed_to and extra['causal'] (in place; returns ev)."""
        env = ev.env_id
        acts = self.active.setdefault(env, [])
        t = ev.t_sim
        self.event_t[ev.event_id] = t
        info: dict = {}
        if ev.action is not None:
            start, end = self._interval(ev)
            others = [aid for (aid, eid, s, e) in acts if s <= start <= e and aid != ev.action.action_id]
            if not ev.simultaneous:
                ev.simultaneous = others
            acts.append((ev.action.action_id, ev.event_id, start, end))
            self.action_event[ev.action.action_id] = ev.event_id
            # prune long-finished interventions
            self.active[env] = [x for x in acts if x[3] >= start - 1e-9]
            p = _assignment_probability(ev)
            if ev.randomization_p is None and p is not None:
                ev.randomization_p = p
            if ev.result is not None:
                info["exec_delay_days"] = max(0.0, (ev.result.finished_t or start) - start)
        # delayed consequences: observations attributed to an earlier action
        delays = []
        for o in ev.observations + (ev.result.observations if ev.result is not None else []):
            cause = o.caused_by
            if cause is None:
                continue
            cause_ev = self.action_event.get(cause, cause)
            if cause_ev in self.event_t:
                delays.append(max(0.0, o.t - self.event_t[cause_ev]))
                if ev.action is None and ev.attributed_to is None:
                    ev.attributed_to = cause_ev
        if delays:
            info["delays_days"] = delays
        if ev.attributed_to is not None and ev.attributed_to in self.event_t and not delays:
            info["delays_days"] = [max(0.0, t - self.event_t[ev.attributed_to])]
        num = [o for o in ev.observations + (ev.result.observations if ev.result is not None else [])
               if o.kind == ObsKind.NUMERIC]
        miss = [o for o in num if o.value is None or o.meta.get("missing")]
        info["missing_frac"] = len(miss) / max(1, len(num)) if num else 0.0
        info["result_missing"] = ev.action is not None and ev.result is None
        rel = [float(o.meta["reliability"]) for o in ev.observations if "reliability" in o.meta]
        if rel:
            info["reliability"] = sum(rel) / len(rel)
        randomised = ev.randomization_p is not None and ev.randomization_p < 1.0
        info["confounding_possible"] = bool(ev.evidence == EvidenceKind.OBSERVATIONAL or len(ev.simultaneous) > 0
                                            or (ev.action is not None and not randomised))
        ev.extra = dict(ev.extra, causal=info)
        return ev


def provenance_features(ev: EventRecord) -> List[float]:
    """Feature vector (``PROV_FEATURES`` order) of one EventRecord (uses extra['causal'] when recorded)."""
    c = ev.extra.get("causal", {}) if isinstance(ev.extra, dict) else {}
    p = ev.randomization_p if ev.randomization_p is not None else _assignment_probability(ev)
    delays = c.get("delays_days") or ([c["exec_delay_days"]] if "exec_delay_days" in c else [])
    d = sum(delays) / len(delays) if delays else 0.0
    rel = c.get("reliability")
    ev_kind = ev.evidence
    if "confounding_possible" in c:
        conf = bool(c["confounding_possible"])
    else:
        conf = ev_kind == EvidenceKind.OBSERVATIONAL or len(ev.simultaneous) > 0 or p is None or p >= 1.0
    return [float(p) if p is not None else 0.0, 1.0 if p is not None else 0.0,
            math.log1p(len(ev.simultaneous)), math.log1p(max(0.0, d)), 1.0 if delays else 0.0,
            float(c.get("missing_frac", 0.0)), 1.0 if c.get("result_missing", ev.action is not None and
                                                              ev.result is None) else 0.0,
            1.0 if ev_kind == EvidenceKind.CONTROLLED else 0.0, 1.0 if ev_kind == EvidenceKind.OBSERVATIONAL else 0.0,
            1.0 if ev_kind == EvidenceKind.INFERRED else 0.0, 1.0 if conf else 0.0,
            1.0 if ev.provenance == Provenance.COUNTERFACTUAL else 0.0,
            float(rel) if rel is not None else 0.0, 1.0 if rel is not None else 0.0,
            1.0 if ev.attributed_to is not None else 0.0]


class CausalProvenance(nn.Module):
    """Appends provenance features to EVENT tokens: x_event <- x_event + MLP(features). [B,N,D] -> [B,N,D]."""

    def __init__(self, d_model: int, n_features: int = len(PROV_FEATURES)):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(nn.Linear(n_features, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, tokens: TokenSet, feats: torch.Tensor) -> TokenSet:
        """feats [B,N,F] aligned with tokens (rows of non-EVENT tokens are ignored)."""
        is_ev = (tokens.kind == int(TokenKind.EVENT)) & tokens.mask.bool()
        add = self.net(feats.to(tokens.x.dtype)) * is_ev.unsqueeze(-1).to(tokens.x.dtype)
        return TokenSet(tokens.x + add, tokens.mask, tokens.kind, tokens.time, tokens.entity, tokens.refs,
                        tokens.entity2)

    def features_for(self, tokens: TokenSet, records: Dict[str, EventRecord]) -> torch.Tensor:
        """[B,N,F] features looked up through tokens.refs (event ids) for EVENT tokens; zeros elsewhere."""
        B, N = tokens.mask.shape
        out = torch.zeros(B, N, self.n_features)
        if tokens.refs is None:
            return out.to(tokens.x.device)
        for b in range(B):
            for i in range(N):
                if int(tokens.kind[b, i]) != int(TokenKind.EVENT):
                    continue
                ref = tokens.refs[b][i] if i < len(tokens.refs[b]) else None
                if ref in records:
                    out[b, i] = torch.tensor(provenance_features(records[ref]))
        return out.to(tokens.x.device)

    def attach(self, tokens: TokenSet, records: Dict[str, EventRecord]) -> TokenSet:
        return self(tokens, self.features_for(tokens, records))


class InterventionContrastive(nn.Module):
    """W03: intervention-sensitive contrastive objective on latent states.

    embed(Z [N,S,D] | [N,D]) -> unit embeddings [N,E].
    loss(Z, intervention [N] ids, response [N,F], response_mask [N,F], obs_similarity [N,N] optional) -> dict
      soft: for anchor i and every j != i that received the same intervention, the target distribution is
            softmax_j(-d_ij / tau_r) of the observed response distance d_ij (mean squared difference over jointly
            observed response channels; tau_r = 0.1 x the mean candidate distance unless given); prediction
            softmax_j(e_i.e_j / tau).
      hard: pairs with the same intervention, response distance > tau_r and observation similarity at or above the
            ``similar_obs`` quantile of the batch (all such pairs when no similarity is given) are pushed below
            cosine ``margin``.
    """

    def __init__(self, d_model: int, d_embed: int = 64, n_heads: int = 4, temperature: float = 0.1,
                 response_temperature: Optional[float] = None, margin: float = 0.3, hard_weight: float = 1.0,
                 similar_obs: float = 0.5, response_temperature_frac: float = 0.1):
        super().__init__()
        self.tau_frac = response_temperature_frac
        self.pool = AttnPool(d_model, n_heads)
        self.proj = MLP(d_model, 2 * d_model, d_embed)
        self.tau, self.tau_r = temperature, response_temperature
        self.margin, self.hard_weight, self.similar_obs = margin, hard_weight, similar_obs

    def embed(self, Z: torch.Tensor) -> torch.Tensor:
        h = self.pool(Z.float())[:, 0] if Z.dim() == 3 else Z.float()
        return F.normalize(self.proj(h), dim=-1)

    @staticmethod
    def response_distance(resp: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        """Pairwise mean squared response difference over jointly observed channels -> (d [N,N], valid [N,N])."""
        resp = resp.float()
        m = torch.ones_like(resp, dtype=torch.bool) if mask is None else mask.bool()
        mf = m.float()
        both = mf @ mf.T                                                             # jointly observed channels
        r = resp * mf
        sq = (r ** 2) @ mf.T + mf @ (r ** 2).T - 2 * r @ r.T
        return sq / both.clamp(min=1.0), both > 0

    def loss(self, Z: torch.Tensor, intervention, response: torch.Tensor, response_mask: Optional[torch.Tensor] = None,
             obs_similarity: Optional[torch.Tensor] = None) -> dict:
        dev = response.device
        e = self.embed(Z)
        N = e.shape[0]
        if isinstance(intervention, (list, tuple)) and intervention and isinstance(intervention[0], str):
            ids: dict = {}
            iv = torch.tensor([ids.setdefault(x, len(ids)) for x in intervention], device=dev)
        else:
            iv = torch.as_tensor(intervention, device=dev)
        same = (iv.unsqueeze(0) == iv.unsqueeze(1)) & ~torch.eye(N, dtype=torch.bool, device=dev)
        d, both = self.response_distance(response, response_mask)
        cand = same & both
        tau_r = self.tau_r
        if tau_r is None:                        # scale-free default: a fraction of the mean response distance
            vals = d[cand]
            tau_r = self.tau_frac * float(vals.mean()) if vals.numel() else 1.0
            tau_r = max(tau_r, 1e-6)
        sim = e @ e.T
        logits = (sim / self.tau).masked_fill(~cand, -1e4)
        target = torch.softmax((-d / tau_r).masked_fill(~cand, -1e4), 1)
        anchors = cand.sum(1) >= 2
        logp = torch.log_softmax(logits, 1)
        soft = -(target * logp).sum(1)
        soft = soft[anchors].mean() if bool(anchors.any()) else e.sum() * 0
        hard_pairs = cand & (d > tau_r)
        if obs_similarity is not None:           # "similar-looking": above the batch quantile of observation similarity
            osim = obs_similarity.to(dev).float()
            off = ~torch.eye(N, dtype=torch.bool, device=dev)
            thr = torch.quantile(osim[off], self.similar_obs) if bool(off.any()) else osim.max()
            hard_pairs = hard_pairs & (osim >= thr)
        hard = (F.relu(sim - self.margin) ** 2)[hard_pairs].mean() if bool(hard_pairs.any()) else e.sum() * 0
        return {"loss": soft + self.hard_weight * hard, "soft": soft.detach(), "hard": hard.detach(),
                "n_anchors": int(anchors.sum()), "n_hard": int(hard_pairs.sum()), "tau_r": tau_r}
