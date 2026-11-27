"""cam_live.nn.model — :class:`CamLiveModel`, the assembled neural model (implements ``nn.api.LatentModelAPI``).

    frozen pretrained backbone (+ LoRA adapters + latent bridge)         ``backbone``     groups: backbone_frozen,
                                                                                            semantic_adapters
    experience tokenizer, experiential core, action / goal / query
    encoders, decision trunk, feature map, learned fast prior            ``exp``          group: experiential
    heads registered by the other workstreams                            ``heads[name]``  groups: heads / reward

Everything downstream of the backbone is float32 (double backward for meta-gradients needs it).
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..config import CamLiveConfig, CoreConfig
from ..contracts import CONTRACT_VERSION, ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Trust
from .api import (GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD, GROUPS,
                  KIND_INDEX, CoreOutput, CoreState, LatentModelAPI, TokenBatch)
from .backbone import _Backbone, build_backbone
from .core import ExperientialCore
from .decision import (ActionEncoder, DecisionTrunk, FeatureMap, GoalEncoder, QueryEncoder, level_ids, modality_ids)
from .fast import FastBeliefState, FastPrior
from .presets import NNConfig, preset_nn, resolve_core
from .tokens import ExperienceTokenizer, _cost_quantities

_TENSOR_KEYS = ("pad_mask", "has_text", "time", "kind_ids", "trust_ids", "values", "value_mask", "name_ids",
                "unit_ids", "qkind", "exposure")


class ExperientialNet(nn.Module):
    """All parameters of the ``experiential`` group. Depends on the backbone only through ``d_sem`` / ``n_taps``
    (so its size can be measured without loading the backbone)."""

    def __init__(self, core: CoreConfig, nn_cfg: NNConfig, d_sem: int, n_taps: int):
        super().__init__()
        d = core.d_model
        small = d < 128
        self.tokenizer = ExperienceTokenizer(
            d, d_sem, n_taps, max_values=core.max_values, value_dim=nn_cfg.value_dim,
            n_buckets=nn_cfg.value_buckets or (256 if small else 2048),
            unit_buckets=nn_cfg.unit_buckets or (64 if small else 256), sem_rank=nn_cfg.sem_rank)
        self.core = ExperientialCore(d, core.n_layers, core.n_heads, core.d_ff, core.n_slots,
                                     state_every=nn_cfg.state_every, dropout=core.dropout, gate_bias=nn_cfg.gate_bias)
        self.action_enc = ActionEncoder(d)
        self.goal_enc = GoalEncoder(d)
        self.query_enc = QueryEncoder(d, self.tokenizer.value_dim)
        self.trunk = DecisionTrunk(d, nn_cfg.trunk_ff or d)
        self.feature_map = FeatureMap(d, nn_cfg.feature_ff or max(8, d // 2), core.n_features)
        self.fast_prior = FastPrior(core.n_features, prior_var=nn_cfg.fast_prior_var, noise=nn_cfg.fast_noise,
                                    noise_mode=nn_cfg.fast_noise_mode)

    def breakdown(self) -> Dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())
        return {"tokenizer": count(self.tokenizer), "core": count(self.core),
                "encoders": count(self.action_enc) + count(self.goal_enc) + count(self.query_enc),
                "trunk": count(self.trunk), "feature_map": count(self.feature_map),
                "fast_prior": count(self.fast_prior)}


def build_experiential(core: CoreConfig, nn_cfg: Optional[NNConfig] = None, d_sem: int = 7680, n_taps: int = 3,
                       seed: int = 0) -> ExperientialNet:
    """The experiential network alone, deterministically initialised from ``seed`` (global RNG left untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1_000_003 + int(seed))
        return ExperientialNet(resolve_core(core), nn_cfg or NNConfig(**preset_nn(core)), d_sem, n_taps)


class CamLiveModel(LatentModelAPI):
    """See module docstring. ``backbone`` may be passed in to share one loaded backbone between models."""

    def __init__(self, cfg: CamLiveConfig, backbone: Optional[_Backbone] = None):
        super().__init__()
        self.cfg = cfg
        self.nn_cfg = NNConfig.from_cfg(cfg)
        self.core_cfg = resolve_core(cfg.core)
        dev = cfg.resolved_device()
        if backbone is None:
            backbone = build_backbone(cfg.backbone, self.nn_cfg, self.core_cfg.d_model, dev, cfg.seed)
        elif backbone.d_latent != self.core_cfg.d_model:
            raise ValueError("the shared backbone's bridge was built for a different d_model")
        self.backbone = backbone
        self.exp = build_experiential(self.core_cfg, self.nn_cfg, backbone.d_sem, backbone.n_taps, cfg.seed)
        self.exp.to(backbone.device)
        self.heads = nn.ModuleDict()
        self._head_group: Dict[str, str] = {}
        self.d_model = self.core_cfg.d_model
        self.n_features = self.core_cfg.n_features
        self.d_sem = backbone.d_sem

    @property
    def device(self) -> torch.device:
        return self.exp.core.slot_init.device

    # ---- semantic backbone -------------------------------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str], *, with_grad: bool = False) -> Tensor:
        return self.backbone.embed(texts, with_grad=with_grad)

    def generate(self, prompts: Sequence[str], *, latent: Optional[Tensor] = None, max_new_tokens: int = 256,
                 temperature: float = 0.8, num_return_sequences: int = 1, seed: Optional[int] = None
                 ) -> List[List[str]]:
        return self.backbone.generate(prompts, latent=latent, max_new_tokens=max_new_tokens, temperature=temperature,
                                      num_return_sequences=num_return_sequences, seed=seed)

    def generation_logprob(self, prompts: Sequence[str], completions: Sequence[str], *,
                           latent: Optional[Tensor] = None) -> Tensor:
        return self.backbone.generation_logprob(prompts, completions, latent=latent)

    def mark_adapters_updated(self) -> None:
        """Invalidate cached embeddings after an adapter update (also detected automatically)."""
        self.backbone.mark_adapters_updated()

    # ---- experience encoding ---------------------------------------------------------------------------------------------
    def _semantic(self, batch: Dict[str, Any], with_grad: bool, grad_last: Optional[int] = None) -> Tensor:
        """[B, T, d_sem] embeddings of the item texts of a collated batch (zeros where an item has no text)."""
        pad, has_text, texts = batch["pad_mask"], batch["has_text"], batch["texts"]
        B, T = pad.shape
        sem = torch.zeros(B, T, self.d_sem, device=self.device)
        where = [(b, t) for b in range(B) for t in range(T) if bool(pad[b, t]) and bool(has_text[b, t])]
        if not where:
            return sem
        n_real = pad.sum(1).tolist()
        parts = [(where, with_grad)]
        if with_grad and grad_last is not None:
            new = [(b, t) for b, t in where if t >= n_real[b] - grad_last]
            old = [(b, t) for b, t in where if t < n_real[b] - grad_last]
            parts = [(old, False), (new, True)]
        for idx, grad in parts:
            if not idx:
                continue
            emb = self.embed_texts([texts[b][t] for b, t in idx], with_grad=grad)
            bi = torch.tensor([b for b, _ in idx], device=self.device)
            ti = torch.tensor([t for _, t in idx], device=self.device)
            sem = sem.index_put((bi, ti), emb.to(sem.dtype))
        return sem

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if k in _TENSOR_KEYS else v) for k, v in batch.items()}

    def tokenize(self, contexts: Sequence[Sequence[ExperienceItem]], *, with_grad: bool = False,
                 grad_last: Optional[int] = None) -> TokenBatch:
        """``with_grad=True`` builds a graph into the semantic adapters; ``grad_last=k`` restricts that graph to the
        newest ``k`` items of each context (older items use cached, detached embeddings) to bound memory."""
        batch = self._to_device(self.exp.tokenizer.collate(contexts, self.core_cfg.max_context))
        x = self.exp.tokenizer.encode(self._semantic(batch, with_grad, grad_last), batch)
        return TokenBatch(x=x, pad_mask=batch["pad_mask"], time=batch["time"], kind_ids=batch["kind_ids"],
                          values=batch["values"], value_mask=batch["value_mask"], refs=batch["refs"])

    def core_forward(self, tokens: TokenBatch, state: Optional[CoreState] = None) -> CoreOutput:
        return self.exp.core(tokens, state)

    def init_state(self, batch: int) -> CoreState:
        return self.exp.core.init_state(batch, self.device)

    def encode_actions(self, actions: Sequence[Sequence[ActionSpec]], *, with_grad: bool = False
                       ) -> Tuple[Tensor, Tensor]:
        V = self.core_cfg.max_values
        ctxs = [[ExperienceItem(kind=ItemKind.ACTION, text=a.render(), ref=a.action_id, trust=Trust.SELF,
                                quantities=_cost_quantities(a.est_cost, "est_cost")[:V]) for a in row]
                for row in actions]
        batch = self._to_device(self.exp.tokenizer.collate(ctxs))
        B, N = batch["pad_mask"].shape
        x = self.exp.tokenizer.encode(self._semantic(batch, with_grad), batch, with_time=False)
        mods = [[a.modality for a in row] + [""] * (N - len(row)) for row in actions]
        lvls = [[a.level for a in row] + [""] * (N - len(row)) for row in actions]
        timeout = torch.tensor([[float(a.timeout_s) for a in row] + [0.0] * (N - len(row)) for row in actions],
                               dtype=torch.float64, device=self.device).view(B, N)
        steps = torch.tensor([[float(len(a.steps)) for a in row] + [0.0] * (N - len(row)) for row in actions],
                             dtype=torch.float32, device=self.device).view(B, N)
        e_a = self.exp.action_enc(x, modality_ids([m for r in mods for m in r], self.device).view(B, N),
                                  level_ids([m for r in lvls for m in r], self.device).view(B, N), timeout, steps)
        mask = batch["pad_mask"]
        return e_a * mask.unsqueeze(-1).to(e_a.dtype), mask

    def encode_goals(self, goals: Sequence[str], *, with_grad: bool = False) -> Tensor:
        ctxs = [[ExperienceItem(kind=ItemKind.GOAL, text=g, trust=Trust.OPERATOR)] for g in goals]
        batch = self._to_device(self.exp.tokenizer.collate(ctxs))
        x = self.exp.tokenizer.encode(self._semantic(batch, with_grad), batch, with_time=False)
        return self.exp.goal_enc(x[:, 0])

    def encode_queries(self, queries: Sequence[Sequence[OutcomeQuery]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = len(queries)
        Q = max(1, max((len(r) for r in queries), default=1))
        dev = self.device
        q_mask = torch.zeros(B, Q, dtype=torch.bool, device=dev)
        kind_ids = torch.zeros(B, Q, dtype=torch.long, device=dev)
        horizon = torch.zeros(B, Q, dtype=torch.float64, device=dev)
        exposure = torch.ones(B, Q, dtype=torch.float64, device=dev)
        relevant = torch.zeros(B, Q, dtype=torch.bool, device=dev)
        texts, names, where = [], [], []
        for b, row in enumerate(queries):
            for j, q in enumerate(row):
                if q.kind not in KIND_INDEX:
                    raise ValueError(f"unknown outcome kind {q.kind!r}")
                q_mask[b, j] = True
                kind_ids[b, j] = KIND_INDEX[q.kind]
                horizon[b, j], exposure[b, j], relevant[b, j] = float(q.horizon_s), float(q.exposure), bool(q.goal_relevant)
                text = f"{q.metric}: {q.description}" if q.description else q.metric
                if q.kind == OutcomeKind.CATEGORICAL and q.categories:
                    text += " categories: " + " | ".join(q.categories)
                texts.append(text)
                names.append(q.metric)
                where.append((b, j))
        tok = self.exp.tokenizer
        sem = torch.zeros(B, Q, self.d_sem, device=dev)
        name_emb = torch.zeros(B, Q, tok.value_dim, device=dev)
        if where:
            bi = torch.tensor([b for b, _ in where], device=dev)
            ji = torch.tensor([j for _, j in where], device=dev)
            sem = sem.index_put((bi, ji), self.embed_texts(texts))
            name_emb = name_emb.index_put((bi, ji), tok.name_embedding(names))
        e_q = self.exp.query_enc(tok.content(sem, q_mask), kind_ids, name_emb, horizon, exposure, relevant)
        return e_q * q_mask.unsqueeze(-1).to(e_q.dtype), q_mask, kind_ids, horizon

    def decision_latent(self, b: Tensor, e_a: Tensor, e_g: Tensor) -> Tensor:
        return self.exp.trunk(b, e_a, e_g)

    def features(self, z: Tensor, e_q: Tensor) -> Tensor:
        return self.exp.feature_map(z, e_q)

    def new_fast_state(self, **kw: Any) -> FastBeliefState:
        """A fresh :class:`FastBeliefState` at the current learned prior (forgetting / change detection from the
        ``nn`` config section unless overridden)."""
        kw.setdefault("forgetting", self.nn_cfg.fast_forgetting)
        kw.setdefault("change_h", self.nn_cfg.fast_change_h)
        return FastBeliefState(self.exp.fast_prior, **kw)

    # ---- heads and parameter groups ----------------------------------------------------------------------------------------
    def register_head(self, name: str, module: nn.Module, group: str = GROUP_HEADS) -> nn.Module:
        if group not in (GROUP_HEADS, GROUP_REWARD, GROUP_EXPERIENTIAL):
            raise ValueError(f"heads can be registered under {GROUP_HEADS!r}, {GROUP_REWARD!r} or "
                             f"{GROUP_EXPERIENTIAL!r}, not {group!r}")
        if not name or "." in name:
            raise ValueError("head names must be non-empty and contain no '.'")
        if name in self.heads:
            if self.heads[name] is module and self._head_group[name] == group:
                return module
            raise ValueError(f"a different head is already registered as {name!r}")
        self.heads[name] = module.to(self.device)
        self._head_group[name] = group
        return module

    def head(self, name: str) -> nn.Module:
        return self.heads[name]

    def head_names(self, group: Optional[str] = None) -> List[str]:
        return [n for n in self.heads if group is None or self._head_group[n] == group]

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups: Dict[str, List[nn.Parameter]] = {g: [] for g in GROUPS}
        seen: set = set()

        def add(group: str, params: Any) -> None:
            for p in params:
                if id(p) not in seen:
                    seen.add(id(p))
                    groups[group].append(p)

        add(GROUP_BACKBONE_FROZEN, self.backbone.frozen_parameters())
        add(GROUP_ADAPTERS, self.backbone.adapter_parameters())
        add(GROUP_EXPERIENTIAL, self.exp.parameters())
        for name, mod in self.heads.items():
            add(self._head_group[name], mod.parameters())
        return groups

    def _named_groups(self) -> List[Tuple[str, nn.Parameter, str]]:
        gid = {id(p): g for g, ps in self.param_groups().items() for p in ps}
        return [(n, p, gid[id(p)]) for n, p in self.named_parameters()]

    def count_parameters(self) -> Dict[str, Any]:
        groups = {g: sum(p.numel() for p in ps) for g, ps in self.param_groups().items()}
        heads = {n: sum(p.numel() for p in m.parameters()) for n, m in self.heads.items()}
        lora = sum(p.numel() for p in self.backbone.lora_parameters())
        total = groups[GROUP_EXPERIENTIAL] + groups[GROUP_ADAPTERS] + groups[GROUP_REWARD] + groups[GROUP_HEADS]
        return {"frozen": groups[GROUP_BACKBONE_FROZEN], "experiential": groups[GROUP_EXPERIENTIAL],
                "adapters": groups[GROUP_ADAPTERS], "adapters_lora": lora,
                "adapters_bridge": groups[GROUP_ADAPTERS] - lora, "reward": groups[GROUP_REWARD],
                "heads": heads, "head_groups": dict(self._head_group), "heads_total": groups[GROUP_HEADS],
                "total_persistent_trainable": total, "by_group": groups,
                "experiential_breakdown": self.exp.breakdown(), "scale": self.core_cfg.scale,
                "backbone": self.backbone.name, "backbone_pretrained": self.backbone.pretrained,
                "fast_state_numbers_per_kind": self.n_features + self.n_features ** 2}

    def group_hashes(self) -> Dict[str, str]:
        """sha256 of every trainable group's tensors (name-sorted). The frozen backbone is identified by name,
        revision and parameter count instead of hashing billions of weights that cannot change."""
        hs = {g: hashlib.sha256() for g in GROUPS if g != GROUP_BACKBONE_FROZEN}
        for name, p, g in sorted(self._named_groups(), key=lambda x: x[0]):
            if g == GROUP_BACKBONE_FROZEN:
                continue
            hs[g].update(name.encode())
            hs[g].update(memoryview(p.detach().to("cpu", torch.float32).contiguous().numpy()).cast("B"))
        out = {g: h.hexdigest() for g, h in hs.items()}
        ident = f"{self.backbone.name}@{self.backbone.revision}#{self.backbone.n_frozen}"
        out[GROUP_BACKBONE_FROZEN] = hashlib.sha256(ident.encode()).hexdigest()
        return out

    # ---- persistence -----------------------------------------------------------------------------------------------------------
    def _trainable_named(self) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {n: p for n, p, g in self._named_groups() if g != GROUP_BACKBONE_FROZEN}
        non_persistent = {f"{mn}.{bn}" if mn else bn for mn, m in self.named_modules()
                          for bn in m._non_persistent_buffers_set}
        for n, buf in self.named_buffers():
            if not n.startswith("backbone.lm.") and n not in non_persistent:
                out[n] = buf
        return out

    def trainable_state(self) -> Dict[str, Tensor]:
        return {n: t.detach().to("cpu").clone() for n, t in self._trainable_named().items()}

    def load_trainable_state(self, state: Dict[str, Tensor], *, strict: bool = True) -> Dict[str, List[str]]:
        own = self._trainable_named()
        missing = sorted(k for k in own if k not in state)
        unexpected = sorted(k for k in state if k not in own)
        mismatched = sorted(k for k in own if k in state and tuple(state[k].shape) != tuple(own[k].shape))
        if strict and (missing or unexpected or mismatched):
            raise RuntimeError(f"load_trainable_state(strict=True): missing={missing[:5]} ({len(missing)}), "
                               f"unexpected={unexpected[:5]} ({len(unexpected)}), "
                               f"shape mismatch={mismatched[:5]} ({len(mismatched)})")
        with torch.no_grad():
            for k, t in own.items():
                if k in state and k not in mismatched:
                    t.copy_(state[k].to(t.device, t.dtype))
        self.backbone.mark_adapters_updated()
        return {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}

    def arch_signature(self) -> Dict[str, Any]:
        c = self.core_cfg
        return {"contract_version": CONTRACT_VERSION, "scale": c.scale,
                "core": {k: getattr(c, k) for k in ("d_model", "n_layers", "n_heads", "d_ff", "n_slots",
                                                    "n_features", "max_context", "max_values")},
                "nn": dataclasses.asdict(self.nn_cfg), "d_sem": self.d_sem, "backbone": self.backbone.info(),
                "heads": {n: {"group": self._head_group[n], "params": sum(p.numel() for p in m.parameters())}
                          for n, m in self.heads.items()}}

    def group_norms(self) -> Dict[str, float]:
        """L2 norm of every trainable group's parameters (for tracking parameter-norm changes, spec 2.3)."""
        out = {}
        for g, ps in self.param_groups().items():
            if g != GROUP_BACKBONE_FROZEN:
                sq = [p.detach().to(torch.float32).pow(2).sum() for p in ps]
                out[g] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
        return out

    def group_delta_norms(self, previous: Dict[str, Tensor]) -> Dict[str, float]:
        """L2 norm of the change of each trainable group relative to an earlier :meth:`trainable_state`
        (parameters absent from ``previous``, e.g. a head registered since, are skipped)."""
        sq: Dict[str, float] = {g: 0.0 for g in GROUPS if g != GROUP_BACKBONE_FROZEN}
        for n, p, g in self._named_groups():
            if g != GROUP_BACKBONE_FROZEN and n in previous and previous[n].shape == p.shape:
                sq[g] += float((p.detach().to("cpu", torch.float32) - previous[n].to(torch.float32)).pow(2).sum())
        return {g: v ** 0.5 for g, v in sq.items()}

    # ---- optimisation helpers ----------------------------------------------------------------------------------------------------
    def grad_norms(self) -> Dict[str, float]:
        """L2 norm of the current gradients per parameter group."""
        out = {}
        for g, ps in self.param_groups().items():
            sq = [p.grad.detach().to(torch.float32).pow(2).sum() for p in ps if p.grad is not None]
            out[g] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
        return out

    def clip_gradients(self, max_norms: Dict[str, float]) -> Dict[str, float]:
        """Clip each listed group to its own maximum norm (never one global clip); returns the pre-clip norms."""
        groups = self.param_groups()
        out = {}
        for g, mx in max_norms.items():
            ps = [p for p in groups[g] if p.grad is not None]
            out[g] = float(torch.nn.utils.clip_grad_norm_(ps, mx)) if ps else 0.0
        return out


def build_model(cfg: CamLiveConfig) -> CamLiveModel:
    return CamLiveModel(cfg)
