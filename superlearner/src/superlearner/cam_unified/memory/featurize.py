"""HashedEventEncoder: a learned, semantics-free encoder of archived ``EventRecord`` s into one event token each.

It lets the memory hierarchy encode raw archived events without any other workstream's modules: feature hashing of
discrete content (tool names, argument keys/short values, observation kinds, units, entity types, sources, words of
text content, error codes) into an ``EmbeddingBag`` plus scale-free numeric features (rewards, costs, durations,
outcome flags, observed numeric values as sign*log1p Fourier features), followed by a learned projection.

Entity *ids* are deliberately not hashed (they are random per world and carry no meaning across worlds): observation
``entity`` / ``target`` fields are never read, and action argument values equal to an entity id of the event (its
``entity_links`` or its observations' entities) are replaced by an ``<entity>`` placeholder token. Non-finite numbers
(NaN / inf in rewards, costs, observed values or numeric arguments) are mapped to finite values (NaN -> 0, +-inf ->
+-1e30) plus a non-finite count feature, so one bad reading cannot turn the event token into NaN. The hash is CRC32
(stable across processes, unlike Python's ``hash``). The integrated system may pass the core's own event-token encoder
to ``Retriever(event_encoder=...)`` instead; this module is the default and the semantic-free baseline.
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
    """

    N_SCALARS = 10

    def __init__(self, d_model: int, n_buckets: int = 1 << 14, max_words: int = 48, n_fourier: int = 6,
                 d_emb: int | None = None):
        super().__init__()
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
        nn.init.normal_(self.emb.weight, std=0.5)

    # ------------------------------------------------------------------ discrete features
    def tokens(self, rec) -> list:
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
        reward, cost = fin(_get(rec, "reward", 0.0) or 0.0), fin(_get(rec, "cost", 0.0) or 0.0)
        dur = 0.0 if res is None else fin(_get(res, "duration", 0.0) or 0.0)
        blp = 0.0 if blp is None else max(-20.0, fin(blp)) / 10.0
        clip = lambda x: max(-1e4, min(1e4, x))      # noqa: E731  (raw scale kept; extreme values bounded)
        return [clip(reward), clip(cost),
                1.0 if _get(rec, "done") else 0.0, ok, 0.0 if act is None else 1.0, math.log1p(len(obs)),
                min(2.0, blp) if blp > 0 else blp, math.log1p(abs(dur)), math.log1p(len(vals)), math.log1p(bad)], \
            vals, argn

    def featurize(self, records: Sequence, device=None) -> tuple:
        idx, offs, scal, vfeat, afeat, rfeat = [], [], [], [], [], []
        n = 2 * self.n_fourier + 1
        for r in records:
            offs.append(len(idx))
            idx.extend(_h(t, self.n_buckets) for t in self.tokens(r))
            s, vals, argn = self.numeric(r)
            scal.append(s)
            vfeat.append(signed_log_fourier(torch.tensor(vals), self.n_fourier).mean(0) if vals else torch.zeros(n))
            afeat.append(signed_log_fourier(torch.tensor(argn), self.n_fourier).mean(0) if argn else torch.zeros(n))
            rfeat.append(signed_log_fourier(torch.tensor([s[0]]), self.n_fourier)[0])
        num = torch.cat([torch.tensor(scal, dtype=torch.float32), torch.stack(vfeat).float(), torch.stack(afeat).float(),
                         torch.stack(rfeat).float()], -1)
        return (torch.tensor(idx, dtype=torch.long, device=device), torch.tensor(offs, dtype=torch.long, device=device),
                num.to(device))

    def forward(self, records: Sequence) -> torch.Tensor:
        """records: list[EventRecord | dict] -> [n, d_model] float32."""
        dev = self.emb.weight.device
        if len(records) == 0:
            return torch.zeros(0, self.d, device=dev)
        idx, offs, num = self.featurize(records, dev)
        h = self.emb(idx, offs) + self.num(num)
        return self.out(self.norm(h))
