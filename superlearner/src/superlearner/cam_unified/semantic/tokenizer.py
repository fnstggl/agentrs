"""ObservationTokenizer: records -> CAM tokens (spec Part 2.1/2.2, architecture §2).

Implements `interfaces.ObservationTokenizer` as an nn.Module whose only trainable parts are the attention-pooling
queries, the shared key/value projection from the semantic model's width to d_model, the numeric/unit/descriptor
projections and a few embeddings. The pretrained encoder is held by reference (not a submodule), so its weights never
enter this module's state_dict.

Token construction (D = cfg.d_model, H = encoder.sem_dim, Q = cfg.sem_tokens_per_obs):

* semantic kinds (TEXT, TOOL_SCHEMA, TOOL_RESULT, HTML, A11Y, CODE, FILE, EXEC_RESULT, ERROR, IMAGE, MESSAGE,
  TRANSACTION, EVENT, GOAL, CONSTRAINT, RELATION): the rendered text (a one-line header with kind/source/entity type
  + the original content) -> cached token states [T, H] (all chunks) -> Q learned queries attention-pool them
  (segment softmax, no padding) -> [Q, D]; + ObsKind, EvidenceKind and within-observation position embeddings.
  TokenKind: RELATION -> RELATION, EVENT -> EVENT, all others -> OBS_SEM.
* NUMERIC (and any observation that carries a `value`): one OBS_NUM token = MLP(symlog value, sign/zero/integer
  flags, n_fourier sin/cos features) + W_u * embed(unit string) + W_d * embed(descriptor "source metric entity_type");
  the value is never unit-converted.
* entity binding: `EntityRegistry.bind` gives the slot; the token gets slot_gain * code[slot] from a fixed random
  orthonormal table drawn per registry (i.e. per episode seed), so identity binds within an episode but carries no
  cross-world meaning; RELATION / MESSAGE / TRANSACTION targets add slot_gain * W_role code[target] and fill entity2.
* time = obs.t - now (days); refs = "obs:<index>[#seq:<seq>][@<artifact_ref>]" so every token points back to its
  record (evidence pointer, traces, exact retrieval of the artifact).
* non-finite values (nan, +inf, -inf) get dedicated flag features (the other numeric features see 0) and are counted
  in `last_stats["nonfinite_values"]` / `nonfinite_total`; they never produce NaN tokens.
* working budget: at most `max_obs_tokens` (default cfg.obs_token_budget) observation tokens per row. Observations are
  taken newest first (by t, then arrival order) and the walk stops at the first one that does not fit, so the kept set
  is always a contiguous most-recent suffix; the drop is reported in `last_stats` and the cumulative `dropped_total`
  counters - never silent.

`tools()` gives one TOOL token per tool from "name + description + JSON schema" text, so an unseen (or renamed) tool
needs no new parameters. `goal()` gives GOAL tokens (text + one per numeric metric target + deadline). `action()` gives
ACTION tokens (the tool token of the action's tool + pooled tokens of its arguments JSON / code); the tool's schema is
resolved from the row's tool list when given, else by the ToolSchema.key in the action's decode record, else by a
name that maps to exactly one registered schema - an ambiguous or unknown tool raises (or, with allow_unknown=True,
is encoded from its name alone and counted in `unknown_tools`). The schema registry is a bounded LRU
(`tool_registry_size`) that `clear_tool_registry()` resets, e.g. per episode.
"""
from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..config import CamConfig
from ..interfaces import (OBS_SCHEMA_VERSION, Action, EvidenceKind, GoalSpec, Observation, ObsKind, TokenKind, TokenSet,
                          ToolSchema)
from .layers import NumericFeatures, RMSNorm, SegmentAttentionPool

OBS_KINDS = list(ObsKind)
EV_KINDS = list(EvidenceKind)
SECOND_ENTITY_KINDS = (ObsKind.RELATION, ObsKind.MESSAGE, ObsKind.TRANSACTION)


def obs_ref(i: int, o: Observation) -> str:
    r = f"obs:{i}"
    if o.seq is not None:
        r += f"#seq:{o.seq}"
    if o.artifact_ref:
        r += f"@{o.artifact_ref}"
    return r


def parse_obs_ref(ref) -> Optional[int]:
    """Observation index from a token ref produced by `obs_ref` (None for other tokens)."""
    if not isinstance(ref, str) or not ref.startswith("obs:"):
        return None
    body = ref[4:]
    for sep in ("#", "@"):
        body = body.split(sep, 1)[0]
    try:
        return int(body)
    except ValueError:
        return None


def _clip(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    h = (2 * max_chars) // 3
    return text[:h] + f"\n...[{len(text) - max_chars} chars omitted]...\n" + text[-(max_chars - h):]


def render_observation(o: Observation, max_chars: Optional[int] = None) -> str:
    """Deterministic text of an observation (header + content). Used for encoding and for verbatim evidence."""
    head = f"[{o.kind.value}"
    if o.source:
        head += f" from {o.source}"
    if o.entity_type:
        head += f" about a {o.entity_type}"
    if o.kind == ObsKind.RELATION and o.relation:
        head += f"; relation: {o.relation}"
    head += "]"
    if o.kind == ObsKind.IMAGE and o.artifact_ref:
        head += f" (image artifact {o.artifact_ref}; no pixels are encoded)"
    body = o.content or ""
    if o.value is not None:
        metric = o.meta.get("metric") if isinstance(o.meta, dict) else None
        num = f"{metric + ' = ' if metric else 'value = '}{o.value!r}{(' ' + o.unit) if o.unit else ''}"
        body = f"{body}\n{num}" if body else num
    return head + ("\n" + _clip(body, max_chars) if body else "")


def numeric_descriptor(o: Observation) -> str:
    metric = o.meta.get("metric") if isinstance(o.meta, dict) else None
    parts = [p for p in (o.source, metric, o.entity_type) if p]
    return " ".join(parts) or "measurement"


def tool_text(t: ToolSchema) -> str:
    return (f"tool {t.name}\n{t.description}\narguments: "
            f"{json.dumps(t.parameters, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
            + (f"\nreturns: {t.returns}" if t.returns else ""))


class ObservationTokenizer(nn.Module):
    def __init__(self, cfg: CamConfig, encoder, *, max_obs_tokens: Optional[int] = None, max_goal_tokens: int = 8,
                 n_action_arg_tokens: int = 3, max_obs_chars: Optional[int] = 200_000, autocast: bool = True,
                 tool_registry_size: int = 8192):
        super().__init__()
        object.__setattr__(self, "enc", encoder)          # held by reference: never part of this state_dict
        D, H, Q = cfg.d_model, encoder.sem_dim, cfg.sem_tokens_per_obs
        self.cfg = cfg
        self.D, self.Q = D, Q
        # observation share of the per-step working budget (config v3: cfg.max_obs_tokens, 0 = max_ctx_tokens // 2), so
        # goal, belief, memory and action tokens keep room in the core's context
        self.max_obs_tokens = int(max_obs_tokens or cfg.obs_token_budget)
        self.max_goal_tokens = int(max_goal_tokens)
        self.n_action_arg_tokens = int(n_action_arg_tokens)
        self.max_obs_chars = max_obs_chars
        self.autocast = autocast
        self.pool = SegmentAttentionPool(H, D, cfg.n_heads)
        # pooling queries start diverse (std 1) so the Q tokens of one observation attend differently from step 0
        self.q_obs = nn.Parameter(torch.randn(Q, D))
        self.q_tool = nn.Parameter(torch.randn(1, D))
        self.q_goal = nn.Parameter(torch.randn(Q, D))
        self.q_act = nn.Parameter(torch.randn(self.n_action_arg_tokens, D))
        # additive type embeddings start small: the (RMS-normalised) content must dominate the initial tokens
        self.obs_kind = nn.Embedding(len(OBS_KINDS), D)
        self.ev_kind = nn.Embedding(len(EV_KINDS), D)
        self.pos = nn.Embedding(max(Q, self.n_action_arg_tokens) + 1, D)
        self.role_emb = nn.Embedding(4, D)               # 0 obs, 1 tool, 2 goal, 3 action
        for e in (self.obs_kind, self.ev_kind, self.pos, self.role_emb):
            nn.init.normal_(e.weight, std=0.02)
        self.content_norm = RMSNorm(D)
        self.numf = NumericFeatures(cfg.n_fourier)
        self.num_mlp = nn.Sequential(nn.Linear(self.numf.dim, D), nn.SiLU(), nn.Linear(D, D))
        self.unit_proj = nn.Sequential(RMSNorm(H), nn.Linear(H, D))
        self.desc_proj = nn.Sequential(RMSNorm(H), nn.Linear(H, D))
        self.norm = RMSNorm(D)
        self.slot_gain = nn.Parameter(torch.tensor(0.5))
        self.target_role = nn.Linear(D, D, bias=False)
        nn.init.orthogonal_(self.target_role.weight)
        self._codes = weakref.WeakKeyDictionary()          # EntityRegistry -> [max_entities, D] codes
        self._tool_by_key: OrderedDict = OrderedDict()     # ToolSchema.key -> ToolSchema (LRU)
        self._keys_by_name: dict = {}                       # name -> {key: None} (insertion-ordered)
        self.tool_registry_size = int(tool_registry_size)
        self.last_stats: dict = {}
        self.dropped_total = {"obs": 0, "tokens": 0, "goal_tokens": 0}
        self.nonfinite_total = 0
        self.unknown_tools = 0
        self.to(encoder.device)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return self.q_obs.device

    def _ac(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast and self.device.type == "cuda")

    def slot_codes(self, registry) -> torch.Tensor:
        """Fixed random orthonormal code table of one registry (one episode): [registry.max, D]."""
        t = self._codes.get(registry)
        if t is None:
            st = repr(registry.rng.getstate()).encode()
            seed = int(hashlib.sha256(st).hexdigest()[:15], 16)
            g = torch.Generator().manual_seed(seed)
            n = registry.max
            a = torch.randn(max(n, self.D), self.D, generator=g)
            qm, _ = torch.linalg.qr(a)                     # orthonormal columns
            t = (qm[:n] if n <= self.D else torch.randn(n, self.D, generator=g) / self.D ** 0.5)
            t = (t / t.norm(dim=1, keepdim=True).clamp_min(1e-6) * self.D ** 0.5).to(self.device)
            self._codes[registry] = t
        return t

    def _pool(self, texts: list, queries: torch.Tensor) -> torch.Tensor:
        """Attention-pool the cached states of `texts` with a query set [Q, D] -> [len(texts), Q, D]."""
        if not texts:
            return torch.zeros(0, queries.shape[0], self.D, device=self.device)
        states = self.enc.token_states(texts)
        seg = torch.cat([torch.full((s.shape[0],), i, dtype=torch.long, device=s.device) for i, s in enumerate(states)])
        flat = self.enc.standardize_states(torch.cat(states, 0))      # fixed per-dimension standardisation
        q = queries.unsqueeze(0).expand(len(texts), -1, -1)
        return self.content_norm(self.pool(flat, seg, q))

    def _numeric(self, values: list, units: list, descs: list) -> torch.Tensor:
        n_bad = sum(1 for v in values if not math.isfinite(v))
        if n_bad:
            self.nonfinite_total += n_bad
            self.last_stats["nonfinite_values"] = self.last_stats.get("nonfinite_values", 0) + n_bad
        v = torch.tensor(values, dtype=torch.float64, device=self.device)
        x = self.num_mlp(self.numf(v))
        x = x + self.unit_proj(self.enc.embed([u or "unitless" for u in units]).to(self.device))
        x = x + self.desc_proj(self.enc.embed(descs).to(self.device))
        return self.content_norm(x)

    @staticmethod
    def _semantic(o: Observation) -> bool:
        return o.kind != ObsKind.NUMERIC or o.value is None

    def _n_tokens(self, o: Observation) -> int:
        return (self.Q if self._semantic(o) else 0) + (1 if o.value is not None else 0)

    def _pack(self, rows: list, B: int) -> TokenSet:
        """rows[b] = list of (x [D], kind, time, entity, entity2, ref) -> padded TokenSet."""
        N = max([len(r) for r in rows] + [0])
        dev = self.device
        x = torch.zeros(B, N, self.D, device=dev)
        mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        kind = torch.zeros(B, N, dtype=torch.long, device=dev)
        tim = torch.zeros(B, N, device=dev)
        ent = torch.full((B, N), -1, dtype=torch.long, device=dev)
        ent2 = torch.full((B, N), -1, dtype=torch.long, device=dev)
        refs = [[None] * N for _ in range(B)]
        for b, r in enumerate(rows):
            if not r:
                continue
            x[b, : len(r)] = torch.stack([t[0] for t in r]).float()
            mask[b, : len(r)] = True
            kind[b, : len(r)] = torch.tensor([int(t[1]) for t in r], device=dev)
            tim[b, : len(r)] = torch.tensor([float(t[2]) for t in r], device=dev)
            ent[b, : len(r)] = torch.tensor([int(t[3]) for t in r], device=dev)
            ent2[b, : len(r)] = torch.tensor([int(t[4]) for t in r], device=dev)
            refs[b][: len(r)] = [t[5] for t in r]
        return TokenSet(x, mask, kind, tim, ent, refs, ent2)

    # ------------------------------------------------------------------ observations
    def observations(self, obs: list, registries: Optional[list], now: Sequence[float]) -> TokenSet:
        ts, _ = self.observations_with_stats(obs, registries, now)
        return ts

    def observations_with_stats(self, obs: list, registries: Optional[list], now: Sequence[float]):
        """[B] list[Observation] -> (TokenSet [B, N, D], stats {kept_obs, dropped_obs, dropped_tokens, n_tokens})."""
        B = len(obs)
        for lst in obs:
            for o in lst or []:
                if o.schema_version > OBS_SCHEMA_VERSION:
                    raise ValueError(f"observation schema version {o.schema_version} is newer than the tokenizer's "
                                     f"{OBS_SCHEMA_VERSION}")
        kept: list = []
        stats = {"kept_obs": [], "dropped_obs": [], "dropped_tokens": [], "n_tokens": [], "nonfinite_values": 0}
        self.last_stats = stats
        for b in range(B):
            lst = obs[b] or []
            order = sorted(range(len(lst)), key=lambda i: (lst[i].t, i), reverse=True)
            budget, keep = self.max_obs_tokens, []
            dropped_tok = 0
            full = False
            for i in order:                       # newest first; stop at the first observation that does not fit
                n = self._n_tokens(lst[i])
                if not full and n <= budget:
                    keep.append(i)
                    budget -= n
                else:
                    full = True
                    dropped_tok += n
            keep.sort()
            kept.append(keep)
            stats["kept_obs"].append(len(keep))
            stats["dropped_obs"].append(len(lst) - len(keep))
            stats["dropped_tokens"].append(dropped_tok)
            stats["n_tokens"].append(self.max_obs_tokens - budget)
            self.dropped_total["obs"] += len(lst) - len(keep)
            self.dropped_total["tokens"] += dropped_tok
        sem_items, num_items = [], []                        # (b, i)
        for b in range(B):
            for i in kept[b]:
                o = obs[b][i]
                if self._semantic(o):
                    sem_items.append((b, i))
                if o.value is not None:
                    num_items.append((b, i))
        sem_at = {k: j for j, k in enumerate(sem_items)}
        num_at = {k: j for j, k in enumerate(num_items)}
        # one entry per output token: (row, source index into cat(sem tokens, num tokens), obs kind, evidence kind,
        # position, code index or -1, target code index or -1, TokenKind, time, slot, slot2, ref)
        meta = []
        code_tables, code_off = [], []
        off = 0
        for b in range(B):
            reg = registries[b] if registries is not None else None
            if reg is not None:
                code_tables.append(self.slot_codes(reg))
            code_off.append(off)
            off += reg.max if reg is not None else 0
            Ns = len(sem_items)
            for i in kept[b]:
                o = obs[b][i]
                slot = reg.bind(o.entity, o.t, o.entity_type) if reg is not None else -1
                slot2 = (reg.bind(o.target, o.t) if (reg is not None and o.kind in SECOND_ENTITY_KINDS and o.target)
                         else -1)
                ok, ek = OBS_KINDS.index(o.kind), EV_KINDS.index(o.evidence)
                c1 = code_off[b] + slot if slot >= 0 else -1
                c2 = code_off[b] + slot2 if slot2 >= 0 else -1
                dt = float(o.t) - float(now[b])
                ref = obs_ref(i, o)
                if (b, i) in sem_at:
                    kind = (TokenKind.RELATION if o.kind == ObsKind.RELATION else
                            TokenKind.EVENT if o.kind == ObsKind.EVENT else TokenKind.OBS_SEM)
                    for j in range(self.Q):
                        meta.append((b, sem_at[(b, i)] * self.Q + j, ok, ek, j, c1, c2, kind, dt, slot, slot2, ref))
                if (b, i) in num_at:
                    meta.append((b, Ns * self.Q + num_at[(b, i)], ok, ek, self.Q, c1, c2, TokenKind.OBS_NUM, dt,
                                 slot, slot2, ref))
        dev = self.device
        with self._ac():
            sem = self._pool([render_observation(obs[b][i], self.max_obs_chars) for b, i in sem_items], self.q_obs)
            parts = [sem.reshape(-1, self.D)]
            if num_items:
                parts.append(self._numeric([float(obs[b][i].value) for b, i in num_items],
                                           [obs[b][i].unit for b, i in num_items],
                                           [numeric_descriptor(obs[b][i]) for b, i in num_items]))
            if not meta:
                self.last_stats = stats
                return self._pack_vec(B, None, None, [], [], [], [], [], []), stats
            src = torch.cat([p.float() for p in parts], 0)
            col = list(zip(*meta))
            t = lambda v: torch.tensor(v, device=dev)
            x = src[t(col[1])] + self.obs_kind(t(col[2])) + self.ev_kind(t(col[3])) + self.pos(t(col[4]))
            x = self.norm(x).float()
            if code_tables:
                table = torch.cat(code_tables, 0)
                c1, c2 = t(col[5]), t(col[6])
                v1 = (c1 >= 0).float().unsqueeze(1)
                v2 = (c2 >= 0).float().unsqueeze(1)
                x = x + self.slot_gain * table[c1.clamp(min=0)] * v1
                x = x + self.slot_gain * self.target_role(table[c2.clamp(min=0)]).float() * v2
        self.last_stats = stats
        return self._pack_vec(B, list(col[0]), x, list(col[7]), list(col[8]), list(col[9]), list(col[10]),
                              list(col[11]), [m[0] for m in meta]), stats

    def _pack_vec(self, B: int, row_of, x, kind, tim, ent, ent2, refs, rows_dummy) -> TokenSet:
        """Vectorised packing of per-token tensors into a padded TokenSet (tokens of a row keep their order)."""
        dev = self.device
        if not refs:
            return TokenSet(torch.zeros(B, 0, self.D, device=dev), torch.zeros(B, 0, dtype=torch.bool, device=dev),
                            torch.zeros(B, 0, dtype=torch.long, device=dev), torch.zeros(B, 0, device=dev),
                            torch.full((B, 0), -1, dtype=torch.long, device=dev), [[] for _ in range(B)],
                            torch.full((B, 0), -1, dtype=torch.long, device=dev))
        counts = [0] * B
        col = []
        for r in row_of:
            col.append(counts[r])
            counts[r] += 1
        N = max(counts)
        rr = torch.tensor(row_of, device=dev)
        cc = torch.tensor(col, device=dev)
        X = torch.zeros(B, N, self.D, device=dev, dtype=x.dtype)
        X = X.index_put((rr, cc), x)
        mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        mask[rr, cc] = True
        K = torch.zeros(B, N, dtype=torch.long, device=dev)
        K[rr, cc] = torch.tensor([int(k) for k in kind], device=dev)
        Tm = torch.zeros(B, N, device=dev)
        Tm[rr, cc] = torch.tensor(tim, dtype=torch.float32, device=dev)
        E = torch.full((B, N), -1, dtype=torch.long, device=dev)
        E[rr, cc] = torch.tensor(ent, device=dev)
        E2 = torch.full((B, N), -1, dtype=torch.long, device=dev)
        E2[rr, cc] = torch.tensor(ent2, device=dev)
        R = [[None] * N for _ in range(B)]
        for r, c, ref in zip(row_of, col, refs):
            R[r][c] = ref
        return TokenSet(X, mask, K, Tm, E, R, E2)

    # ------------------------------------------------------------------ tools / goals / actions
    def register_tools(self, tools: Sequence[ToolSchema]) -> None:
        """Remember the schemas (by ToolSchema.key, bounded LRU) so ACTION tokens can be built for their actions."""
        for t in tools:
            k = t.key
            if k in self._tool_by_key:
                self._tool_by_key.move_to_end(k)
            else:
                self._tool_by_key[k] = t
                self._keys_by_name.setdefault(t.name, {})[k] = None
            while len(self._tool_by_key) > self.tool_registry_size:
                old_k, old_t = self._tool_by_key.popitem(last=False)
                names = self._keys_by_name.get(old_t.name, {})
                names.pop(old_k, None)
                if not names:
                    self._keys_by_name.pop(old_t.name, None)

    def clear_tool_registry(self) -> None:
        self._tool_by_key.clear()
        self._keys_by_name.clear()

    def resolve_tool(self, a: Action, row_tools: Optional[Sequence[ToolSchema]] = None,
                     allow_unknown: bool = False) -> ToolSchema:
        """The schema an action was taken with: from the row's tool list (by name, and by the decode record's key when
        present), else by that key in the registry, else by a name with exactly one registered schema."""
        rec = (a.logp_parts or {}).get("decode") if isinstance(a.logp_parts, dict) else None
        key = rec.get("key") if isinstance(rec, dict) else None
        if row_tools is not None:
            for t in row_tools:
                if t.name == a.tool and (key is None or t.key == key):
                    return t
        # not in the row's list (e.g. the generator's built-in wait / noop, which are registered when their TOOL
        # tokens are built): the registry, exactly by the record's key, else by a uniquely registered name
        if key is not None and key in self._tool_by_key:
            return self._tool_by_key[key]
        if key is None:
            keys = list(self._keys_by_name.get(a.tool, {}))
            if len(keys) == 1:
                return self._tool_by_key[keys[0]]
            if len(keys) > 1:
                raise ValueError(f"tool name {a.tool!r} maps to {len(keys)} registered schemas; pass the row's tool "
                                 f"list or an action with a decode record")
        self.unknown_tools += 1
        if allow_unknown:
            return ToolSchema(a.tool, "", {})
        raise ValueError(f"unknown tool {a.tool!r} (key {key}): not in the row's tool list / registry")

    def tool_vectors(self, tools: list) -> torch.Tensor:
        """[n] ToolSchema -> TOOL token vectors [n, D]."""
        self.register_tools(tools)
        with self._ac():
            p = self._pool([tool_text(t) for t in tools], self.q_tool)[:, 0]
            return self.norm(p + self.role_emb.weight[1]).float()

    def tools(self, tools: list) -> TokenSet:
        """[B] list[ToolSchema] -> TOOL tokens [B, T, D] (one per tool; refs = ToolSchema.key)."""
        B = len(tools)
        flat = [t for lst in tools for t in lst]
        vecs = self.tool_vectors(flat) if flat else None
        rows, k = [], 0
        for lst in tools:
            r = []
            for t in lst:
                r.append((vecs[k], TokenKind.TOOL, 0.0, -1, -1, t.key))
                k += 1
            rows.append(r)
        return self._pack(rows, B)

    def goal(self, goals: list) -> TokenSet:
        """[B] GoalSpec|None -> GOAL tokens: Qg text tokens + one per numeric metric target (+ deadline), at most
        max_goal_tokens; metrics beyond the budget are counted in dropped_total['goal_tokens']."""
        B = len(goals)
        Qg = min(self.Q, self.max_goal_tokens)
        texts, nums = [], []                      # (b,), (b, value, unit, desc)
        for b, g in enumerate(goals):
            if g is None:
                continue
            texts.append(b)
            items = [(float(v), str(k), f"goal target {k}") for k, v in (g.metrics or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if g.deadline_days is not None:
                items.append((float(g.deadline_days), "days", "goal deadline"))
            room = self.max_goal_tokens - Qg
            self.dropped_total["goal_tokens"] += max(0, len(items) - room)
            for v, u, d in items[:room]:
                nums.append((b, v, u, d))
        rows: list = [[] for _ in range(B)]
        with self._ac():
            tx = self._pool([goals[b].text for b in texts], self.q_goal)
            nx = self._numeric([n[1] for n in nums], [n[2] for n in nums], [n[3] for n in nums]) if nums else None
            for j, b in enumerate(texts):
                for q in range(Qg):
                    rows[b].append((self.norm(tx[j, q] + self.role_emb.weight[2] + self.pos.weight[q]),
                                    TokenKind.GOAL, 0.0, -1, -1, goals[b].goal_id))
            for j, (b, v, u, d) in enumerate(nums):
                rows[b].append((self.norm(nx[j] + self.role_emb.weight[2] + self.pos.weight[self.Q]),
                                TokenKind.GOAL, 0.0, -1, -1, goals[b].goal_id))
        return self._pack(rows, B)

    def action(self, actions: list, tool_lists: Optional[list] = None, allow_unknown: bool = False) -> TokenSet:
        """[B] Action|None -> ACTION tokens: [tool token of the action's tool] + n_action_arg_tokens pooled tokens of
        its arguments JSON (+ code when not inside args); an action with no arguments has the tool token only.
        tool_lists[b] (optional) is the tool list row b's action was chosen from (see resolve_tool)."""
        B = len(actions)
        tools, texts, idx = [], [], []
        for b, a in enumerate(actions):
            if a is None:
                continue
            t = self.resolve_tool(a, tool_lists[b] if tool_lists is not None else None, allow_unknown)
            tools.append((b, t))
            txt = json.dumps(a.args, sort_keys=True, ensure_ascii=False, default=str) if a.args else ""
            if a.code and (not a.args or a.code not in json.dumps(a.args, ensure_ascii=False)):
                txt = (txt + "\ncode:\n" + a.code) if txt else a.code
            if txt:
                texts.append(txt)
                idx.append(b)
        rows: list = [[] for _ in range(B)]
        with self._ac():
            tv = self.tool_vectors([t for _, t in tools]) if tools else None
            av = self._pool(texts, self.q_act) if texts else None
            for j, (b, t) in enumerate(tools):
                rows[b].append((self.norm(tv[j] + self.role_emb.weight[3]), TokenKind.ACTION, 0.0, -1, -1,
                                actions[b].action_id))
            for j, b in enumerate(idx):
                for q in range(self.n_action_arg_tokens):
                    rows[b].append((self.norm(av[j, q] + self.role_emb.weight[3] + self.pos.weight[q]),
                                    TokenKind.ACTION, 0.0, -1, -1, actions[b].action_id))
        return self._pack(rows, B)
