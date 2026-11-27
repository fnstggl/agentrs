"""HashedEventEncoder: a learned, semantics-free encoder of archived ``EventRecord`` s into one event token each.

It lets the memory hierarchy encode raw archived events without any other workstream's modules: feature hashing of
discrete structure (tool names, argument keys/short values, observation kinds, units, entity types, sources, error
codes) and, separately, of the words of text content into one shared ``EmbeddingBag``, plus scale-free numeric
features (rewards, costs, durations, outcome flags, observed numeric values as sign*log1p Fourier features). The three
fields are RMS-normalised separately and summed (so a rare content word is not drowned by numeric variation), then
projected.

Entity *ids* are deliberately not hashed (they are random per world and carry no meaning across worlds): observation
``entity`` / ``target`` fields are never read, and action argument values equal to an entity id of the event (its
``entity_links`` or its observations' entities) are replaced by an ``<entity>`` placeholder token. Non-finite numbers
(NaN / inf in rewards, costs, observed values or numeric arguments) are mapped to finite values (NaN -> 0, +-inf ->
+-1e30) plus a non-finite count feature, so one bad reading cannot turn the event token into NaN. The hash is CRC32
(stable across processes, unlike Python's ``hash``). The integrated system may pass the core's own event-token encoder
to ``Retriever(event_encoder=...)`` instead; this module is the default and the semantic-free baseline.

Actor view (reward integrity; plan 2026-11-26 §1 Memory "featurize REBUILD: ActorView only"). Memory event tokens are
actor inputs, so by default (``view="actor"``) the encoder reads ONLY what the acting agent itself saw or produced:
``actor_view(record)`` = the observations available before the action, its own action, the execution result the
environment returned to it (ok, error, cost, duration, result observations), its own predictions' behaviour
log-probability, its entity links and the step. It never reads ``EventRecord.reward`` (the evaluator's return
increment, or a verified label stored there), ``cost`` / ``done`` of the record (harness-side copies; the actor-visible
cost is ``result.cost``), ``extra`` (labels such as success / verified_outcome / family), ``training_provenance``,
``goal_id`` or ids. Changing any of those leaves the event token bitwise identical (tested with canaries).
``view="legacy"`` is the pre-2026-11-26 encoder that also reads reward / cost / done: a declared baseline only, refused
when ``reward_mode`` is "observation_only".
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Sequence

import torch
import torch.nn as nn

from ._nn import RMSNorm, signed_log_fourier

_WORD = re.compile(r"[a-z0-9_]+")


def _h(token: str, n: int) -> int:
    return zlib.crc32(token.encode("utf-8", "replace")) % n


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


_BIG = 1e30


def _finite(v) -> tuple:
    """-> (finite float, 1 if the input was NaN/inf else 0)."""
    v = float(v)
    if math.isfinite(v):
        return v, 0
    return (0.0 if math.isnan(v) else math.copysign(_BIG, v)), 1


ACTOR_VIEW_FIELDS = ("observations", "action", "result", "behavior_logp", "entity_links", "step")
VIEWS = ("actor", "legacy")


def actor_view(rec) -> dict:
    """The actor-visible part of an archived event (EventRecord or its dict form): ACTOR_VIEW_FIELDS only. Nothing an
    evaluator, a label source or the harness attached (reward, record cost/done, extra, training provenance) is in
    it."""
    return {k: _get(rec, k) for k in ACTOR_VIEW_FIELDS}


def _entity_ids(rec) -> set:
    """Entity ids an event mentions (entity links + observation entity / target), for masking argument values."""
    out = set()
    for e in _get(rec, "entity_links") or []:
        e = e.get("id", e.get("entity")) if isinstance(e, dict) else e
        if e is not None:
            out.add(str(e))
    res = _get(rec, "result")
    obs = list(_get(rec, "observations") or []) + (list(_get(res, "observations") or []) if res is not None else [])
    for o in obs:
        for f in ("entity", "target"):
            v = _get(o, f)
            if v is not None:
                out.add(str(v))
    return out


class HashedEventEncoder(nn.Module):
    """records -> event tokens [n, d_model].

    Args:
        d_model: output width.
        n_buckets: hash buckets of the discrete-feature embedding bag.
        max_words: words of text content hashed per observation.
        n_fourier: Fourier frequencies of the numeric encodings.
        d_emb: width of the bag embedding and the numeric projection (default ``min(d_model, 128)``), projected to
            ``d_model`` by the output layer.
        view: "actor" (default; reads `actor_view(record)` only) or "legacy" (declared baseline; also reads the
            record's reward / cost / done).
        reward_mode: the run's reward mode; "observation_only" refuses ``view="legacy"``.
    """

    N_SCALARS = 10

    def __init__(self, d_model: int, n_buckets: int = 1 << 14, max_words: int = 48, n_fourier: int = 6,
                 d_emb: int | None = None, view: str = "actor", reward_mode: str | None = None):
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        if view == "legacy" and reward_mode == "observation_only":
            raise ValueError("the legacy event encoder reads the evaluator reward; it is a declared baseline and "
                             "cannot be used with reward_mode='observation_only'")
        self.view = view
        self.d = d_model
        e = int(d_emb or min(d_model, 128))
        self.d_emb = e
        self.n_buckets = int(n_buckets)
        self.max_words = int(max_words)
        self.n_fourier = int(n_fourier)
        self.emb = nn.EmbeddingBag(self.n_buckets, e, mode="mean")
        n_num = self.N_SCALARS + 3 * (2 * n_fourier + 1)
        self.num = nn.Linear(n_num, e)
        self.norm = RMSNorm(e)
        self.out = nn.Linear(e, d_model)
        self.norm_s, self.norm_w, self.norm_n = RMSNorm(e), RMSNorm(e), RMSNorm(e)    # per-field normalisation
        nn.init.normal_(self.emb.weight, std=0.5)

    # ------------------------------------------------------------------ discrete features
    def tokens(self, rec) -> list:
        if self.view == "actor":
            rec = actor_view(rec)
        toks = []
        act = _get(rec, "action")
        if act is not None:
            toks.append("tool:" + str(_get(act, "tool")))
            ents = None
            for k, v in sorted((_get(act, "args") or {}).items()):
                toks.append("arg:" + str(k))
                if isinstance(v, str):
                    ents = _entity_ids(rec) if ents is None else ents
                    if v in ents:                          # an entity id: random per world, never hashed
                        toks.append(f"argv:{k}=<entity>")
                        continue
                if isinstance(v, (str, bool)) or (isinstance(v, int) and abs(v) < 1000):
                    sv = str(v)
                    if len(sv) <= 32:
                        toks.append(f"argv:{k}={sv.lower()}")
                    else:
                        toks.extend("aw:" + w for w in _WORD.findall(sv.lower())[: self.max_words // 2])
        res = _get(rec, "result")
        if res is not None:
            toks.append("ok:" + str(bool(_get(res, "ok"))))
            err = _get(res, "error")
            if err:
                toks.append("err:" + str(err)[:48].lower())
            for o in _get(res, "observations") or []:
                toks.extend(self._obs_tokens(o, "r"))
        for o in _get(rec, "observations") or []:
            toks.extend(self._obs_tokens(o, "o"))
        return toks or ["<empty>"]

    def _obs_tokens(self, o, pre: str) -> list:
        kind = _get(o, "kind")
        kind = getattr(kind, "value", kind)
        t = [f"{pre}kind:{kind}"]
        for f in ("unit", "entity_type", "source", "relation"):
            v = _get(o, f)
            if v:
                t.append(f"{pre}{f}:{str(v).lower()[:48]}")
        c = _get(o, "content") or ""
        if c:
            t.extend(f"{pre}w:{w}" for w in _WORD.findall(c.lower())[: self.max_words])
        return t

    # ------------------------------------------------------------------ numeric features
    def numeric(self, rec) -> list:
        if self.view == "actor":
            rec = actor_view(rec)
        res = _get(rec, "result")
        ok = 0.0 if res is None else (1.0 if _get(res, "ok") else -1.0)
        obs = list(_get(rec, "observations") or []) + (list(_get(res, "observations") or []) if res is not None else [])
        bad = 0

        def fin(v):
            nonlocal bad
            x, b = _finite(v)
            bad += b
            return x
        vals = [fin(_get(o, "value")) for o in obs
                if isinstance(_get(o, "value"), (int, float)) and not isinstance(_get(o, "value"), bool)]
        act = _get(rec, "action")
        argn = [fin(v) for v in ((_get(act, "args") or {}).values() if act is not None else [])
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        blp = _get(rec, "behavior_logp")
        if self.view == "actor":
            # slot 0 carries the actor-visible spend of the action (result.cost); no reward, no done flag
            reward = fin(0.0 if res is None else (_get(res, "cost", 0.0) or 0.0))
            cost, done = 0.0, 0.0
        else:
            reward, cost = fin(_get(rec, "reward", 0.0) or 0.0), fin(_get(rec, "cost", 0.0) or 0.0)
            done = 1.0 if _get(rec, "done") else 0.0
        dur = 0.0 if res is None else fin(_get(res, "duration", 0.0) or 0.0)
        blp = 0.0 if blp is None else max(-20.0, fin(blp)) / 10.0
        clip = lambda x: max(-1e4, min(1e4, x))      # noqa: E731  (raw scale kept; extreme values bounded)
        return [clip(reward), clip(cost),
                done, ok, 0.0 if act is None else 1.0, math.log1p(len(obs)),
                min(2.0, blp) if blp > 0 else blp, math.log1p(abs(dur)), math.log1p(len(vals)), math.log1p(bad)], \
            vals, argn

    def featurize(self, records: Sequence, device=None) -> tuple:
        """-> (struct idx, struct offsets, word idx, word offsets, has-words [n], numeric features [n, F])."""
        n_f = self.n_fourier
        idx_s, offs_s, idx_w, offs_w, has_w, scal = [], [], [], [], [], []
        vals, vseg, argn, aseg = [], [], [], []
        for i, r in enumerate(records):
            offs_s.append(len(idx_s))
            offs_w.append(len(idx_w))
            nw = 0
            for t in self.tokens(r):
                if t[:3] in _WORD_FIELDS:
                    idx_w.append(_h(t, self.n_buckets))
                    nw += 1
                else:
                    idx_s.append(_h(t, self.n_buckets))
            if len(idx_s) == offs_s[-1]:
                idx_s.append(_h("<empty>", self.n_buckets))
            has_w.append(1.0 if nw else 0.0)
            s, v, a = self.numeric(r)
            scal.append(s)
            vals.extend(v)
            vseg.extend([i] * len(v))
            argn.extend(a)
            aseg.extend([i] * len(a))
        n = len(records)
        sc = torch.tensor(scal, dtype=torch.float32)
        num = torch.cat([sc, _seg_mean(vals, vseg, n, n_f), _seg_mean(argn, aseg, n, n_f),
                         signed_log_fourier(sc[:, 0], n_f)], -1)
        t = lambda x: torch.tensor(x, dtype=torch.long, device=device)      # noqa: E731
        return (t(idx_s), t(offs_s), t(idx_w), t(offs_w), torch.tensor(has_w, device=device), num.to(device))

    def forward(self, records: Sequence) -> torch.Tensor:
        """records: list[EventRecord | dict] -> [n, d_model] float32. Each feature field (structure, content words,
        numbers) is normalised on its own before they are combined, so no field drowns the others."""
        dev = self.emb.weight.device
        if len(records) == 0:
            return torch.zeros(0, self.d, device=dev)
        idx_s, offs_s, idx_w, offs_w, has_w, num = self.featurize(records, dev)
        h = self.norm_s(self.emb(idx_s, offs_s)) + has_w.unsqueeze(1) * self.norm_w(self.emb(idx_w, offs_w)) + \
            self.norm_n(self.num(num))
        return self.out(self.norm(h))


_WORD_FIELDS = ("ow:", "rw:", "aw:")         # content words (observation / result text, long argument text)


def _seg_mean(values: list, seg: list, n: int, n_f: int) -> torch.Tensor:
    """Mean signed-log Fourier features of each record's values ([n, 2*n_f+1]; zeros where a record has none)."""
    out = torch.zeros(n, 2 * n_f + 1)
    if values:
        f = signed_log_fourier(torch.tensor(values, dtype=torch.float32), n_f)
        idx = torch.tensor(seg, dtype=torch.long)
        out.index_add_(0, idx, f)
        out /= torch.bincount(idx, minlength=n).clamp(min=1).unsqueeze(1).float()
    return out
