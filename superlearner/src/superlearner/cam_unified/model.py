"""CamUnifiedModel: the assembled CAM Unified intelligence (spec Parts 1.2, 12, 20 A22; architecture §1).

One decision implements the step equation

    O_t   = Tokenize(new observations)  (+ ENTITY / RELATION tokens from the entity encoder)
    M_t   = Retrieve(Q(Z_{t-1}, G_t, B_t))                                      (use_memory)
    Z_t   = F_theta(Z_{t-1}, [O_t; A_{t-1}; TIME(dt_{t-1}); B_t; option_{t-1}; M_t; G_t])   (shared backbone)
    R_t   = Reason_theta(Z_t, [B_t; option_{t-1}; M_t; G_t])                    (use_reasoning; R_t = Z_t otherwise)
    o_t   ~ pi_option(R_t, G_t) with termination beta(R_t, o_{t-1})             (option policy)
    b_t   = BUDGET(remaining days / funds / decisions)                           (use_learned_exploration)
    a_t   ~ pi(a | R_t, [G_t; o_t; b_t], B_t, tools, O_t)                       (CAM action generator), or
    a_t   = argmax over CAM-generated candidates of imagined, belief-aware returns (use_imagination)
    Y_hat = OutcomeHead(Prior(Z_t, A_t, B_t, dt)) mixed over belief particles, combined with the Kalman/NIG last
            layer on bound continuous outcomes (mean = head + phi^T mu_c, variance = the Kalman predictive variance)

and `observe_outcomes` performs the belief (particle) and Bayes-last-layer updates from the realised outcomes.

The same context assembly is used when acting (`encode_step`), when recomputing a decision for the policy objective
(`policy_evaluate`) and when filtering replayed sequences for the world / adaptation / memory objectives
(`filter_sequence`: beliefs re-filtered with the stored outcomes, memory tokens rebuilt from the stored retrieved ids,
options from the stored option ids), so the core is trained on the context it acts on.

Every component is injected (dependency injection against the contract protocols) or built by `build_model` from a
dotted-path registry, so the classes of workstreams A-G are wired in by path once their branches merge:
`DEFAULT_COMPONENT_PATHS` lists exactly what is expected. This module never imports test doubles.

Shapes: B environments, S state tokens, N context tokens, D = d_model, K particles, Q outcome queries, T tools.
"""
from __future__ import annotations

import copy
import functools
import importlib
import inspect
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import torch
import torch.nn as nn

from .config import CamConfig
from .interfaces import (Action, ActionSource, EntityRegistry, FastState, ImaginedStep, ImaginedTrajectory,
                         OutcomeBatch, TokenKind, TokenSet)
from .outcome_spec import (CONTINUOUS, CORE_INDEX, OutcomeQueryEncoder, OutcomeTracker, bind_channels,
                           build_outcome_batch, lookup_channels, make_room, make_text_encoder, select_outcomes,
                           with_targets)

N_CONTINUOUS_CORE = 3          # cost, duration, reward: the continuous core outcome keys (Kalman channels)


# ============================================================================ errors and small helpers

class ShapeError(ValueError):
    """A tensor crossing a component boundary violates the contract shape."""


class ComponentUnavailable(ImportError):
    """A component's dotted path cannot be imported because its module does not exist yet (its workstream has not
    merged), or a required component was explicitly left out."""


class ComponentImportError(ImportError):
    """A component's module exists but importing it failed (a bug or a missing dependency inside it). Never
    swallowed by `build_model`."""


def check_tokens(ts: TokenSet, B: int, D: int, where: str) -> TokenSet:
    """Assert the TokenSet contract: x [B,N,D] float; mask/kind/entity [B,N]; time [B,N]; refs [B][N] when given."""
    if not isinstance(ts, TokenSet):
        raise ShapeError(f"{where}: expected TokenSet, got {type(ts).__name__}")
    x = ts.x
    if x.dim() != 3 or x.shape[0] != B or x.shape[2] != D:
        raise ShapeError(f"{where}: x {tuple(x.shape)} != [B={B}, N, D={D}]")
    N = x.shape[1]
    for name in ("mask", "kind", "time", "entity"):
        t = getattr(ts, name)
        if tuple(t.shape) != (B, N):
            raise ShapeError(f"{where}: {name} {tuple(t.shape)} != [{B}, {N}]")
    if ts.mask.dtype != torch.bool:
        raise ShapeError(f"{where}: mask dtype {ts.mask.dtype} != bool")
    if ts.entity2 is not None and tuple(ts.entity2.shape) != (B, N):
        raise ShapeError(f"{where}: entity2 {tuple(ts.entity2.shape)} != [{B}, {N}]")
    if ts.refs is not None and (len(ts.refs) != B or any(len(r) != N for r in ts.refs)):
        raise ShapeError(f"{where}: refs must be [B={B}][N={N}]")
    return ts


def check_shape(t: torch.Tensor, shape: tuple, where: str) -> torch.Tensor:
    if tuple(t.shape) != tuple(shape):
        raise ShapeError(f"{where}: {tuple(t.shape)} != {tuple(shape)}")
    return t


def dist_mean(d) -> torch.Tensor:
    m = d.mean
    return m() if callable(m) else m


def dist_var(d) -> torch.Tensor:
    v = d.variance
    return v() if callable(v) else v


def dist_log_prob(d, y: torch.Tensor, censored: Optional[torch.Tensor] = None) -> torch.Tensor:
    if censored is not None and bool(censored.any()):
        try:
            return d.log_prob(y, censored=censored)
        except TypeError:
            pass
    return d.log_prob(y)


def dist_sample(d, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if generator is not None:
        try:
            return d.sample(generator=generator)
        except TypeError:
            pass
    return d.sample()


def gen_on(generator: Optional[torch.Generator], device) -> Optional[torch.Generator]:
    """The generator if it lives on `device` (torch requires generator and tensor on one device), else None."""
    if generator is None:
        return None
    return generator if torch.device(generator.device).type == torch.device(device).type else None


def randn_like_gen(t: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is not None and torch.device(generator.device).type == "cpu":
        return torch.randn(t.shape, generator=generator).to(t.device, t.dtype)
    return torch.randn_like(t)


def masked_token(x: torch.Tensor, kind: int, valid: Optional[torch.Tensor] = None) -> TokenSet:
    """[B,D] -> one-token TokenSet [B,1,D] of `kind` (mask = valid)."""
    B = x.shape[0]
    dev = x.device
    mask = torch.ones(B, 1, dtype=torch.bool, device=dev) if valid is None else valid.view(B, 1).to(torch.bool)
    return TokenSet(x.unsqueeze(1), mask, torch.full((B, 1), int(kind), dtype=torch.long, device=dev),
                    torch.zeros(B, 1, device=dev), torch.full((B, 1), -1, dtype=torch.long, device=dev),
                    [[None] for _ in range(B)])


def as_kind(ts: TokenSet, kind: int) -> TokenSet:
    return TokenSet(ts.x, ts.mask, torch.full_like(ts.kind, int(kind)), ts.time, ts.entity, ts.refs, ts.entity2)


def with_mask(ts: TokenSet, mask: torch.Tensor) -> TokenSet:
    return TokenSet(ts.x, ts.mask & mask.view(-1, 1).to(torch.bool), ts.kind, ts.time, ts.entity, ts.refs,
                    ts.entity2)


def detach_tokens(ts: Optional[TokenSet]) -> Optional[TokenSet]:
    if ts is None:
        return None
    return TokenSet(ts.x.detach().float(), ts.mask, ts.kind, ts.time.detach(), ts.entity,
                    None if ts.refs is None else [list(r) for r in ts.refs],
                    None if ts.entity2 is None else ts.entity2)


def cat_rows(sets: Sequence[TokenSet]) -> TokenSet:
    """Concatenate TokenSets along the batch dimension, right-padding N (masked)."""
    N = max(s.N for s in sets)
    D = sets[0].x.shape[2]
    dev = sets[0].x.device

    def pad(t, fill, n):
        if t.shape[1] == n:
            return t
        shape = (t.shape[0], n - t.shape[1]) + tuple(t.shape[2:])
        return torch.cat([t, torch.full(shape, fill, dtype=t.dtype, device=t.device)], 1)

    x = torch.cat([pad(s.x, 0.0, N) for s in sets], 0)
    refs = []
    for s in sets:
        rr = s.refs if s.refs is not None else [[None] * s.N for _ in range(s.B)]
        refs.extend(list(r) + [None] * (N - len(r)) for r in rr)
    e2 = None
    if any(s.entity2 is not None for s in sets):
        e2 = torch.cat([pad(s._e2(), -1, N) for s in sets], 0)
    out = TokenSet(x, torch.cat([pad(s.mask, False, N) for s in sets], 0),
                   torch.cat([pad(s.kind, int(TokenKind.PAD), N) for s in sets], 0),
                   torch.cat([pad(s.time, 0.0, N) for s in sets], 0),
                   torch.cat([pad(s.entity, -1, N) for s in sets], 0), refs, e2)
    assert out.x.shape[2] == D and out.x.device == dev
    return out


KIND_PRIORITY = {int(TokenKind.GOAL): 9, int(TokenKind.OPTION): 9, int(TokenKind.BUDGET): 9,
                 int(TokenKind.TIME): 9, int(TokenKind.ACTION): 8, int(TokenKind.BELIEF): 8,
                 int(TokenKind.EVENT): 6, int(TokenKind.OBS_NUM): 5, int(TokenKind.OBS_SEM): 5,
                 int(TokenKind.ENTITY): 4, int(TokenKind.RELATION): 4, int(TokenKind.MEMORY): 3}


def fit_budget(ts: TokenSet, budget: int, dropped: Optional[list] = None) -> TokenSet:
    """Keep at most `budget` context tokens per row (the per-step working-memory budget, not a lifetime cap):
    real tokens by kind priority (goal/option/budget/time > action/belief > events > observations > entities >
    memory), then recency (token time closest to now). Order of the kept tokens is preserved. `dropped` ([B] dicts)
    receives the number of real tokens left out per kind name (the archive keeps every raw observation)."""
    if ts.N <= budget:
        return ts
    pri = torch.zeros_like(ts.time)
    for k, p in KIND_PRIORITY.items():
        pri = torch.where(ts.kind == k, torch.full_like(pri, float(p)), pri)
    score = pri * 1e4 + ts.time.clamp(-999.0, 0.0)
    score = torch.where(ts.mask, score, torch.full_like(score, -float("inf")))
    idx = score.topk(budget, dim=1).indices.sort(dim=1).values           # [B, budget]
    g = lambda t: t.gather(1, idx)
    if dropped is not None:
        keep = torch.zeros_like(ts.mask)
        keep.scatter_(1, idx, True)
        lost = ts.mask & ~keep
        for b in range(ts.B):
            if bool(lost[b].any()):
                for k in ts.kind[b][lost[b]].tolist():
                    name = TokenKind(int(k)).name
                    dropped[b][name] = dropped[b].get(name, 0) + 1
    x = ts.x.gather(1, idx.unsqueeze(-1).expand(-1, -1, ts.x.shape[2]))
    refs = None if ts.refs is None else [[ts.refs[b][int(i)] for i in idx[b]] for b in range(ts.B)]
    return TokenSet(x, g(ts.mask), g(ts.kind), g(ts.time), g(ts.entity), refs,
                    None if ts.entity2 is None else g(ts.entity2))


def belief_summary(logw: Optional[torch.Tensor], top: int = 3) -> list:
    """[B,K] log-weights -> per-env {"ess", "entropy", "top_w"} (DecisionTrace.belief_before/after)."""
    if logw is None:
        return []
    w = torch.softmax(logw.float(), -1)
    ess = 1.0 / (w * w).sum(-1)
    ent = -(w * torch.log(w.clamp_min(1e-30))).sum(-1)
    tw = w.topk(min(top, w.shape[-1]), -1).values
    return [{"ess": float(ess[b]), "entropy": float(ent[b]), "top_w": [float(v) for v in tw[b]]}
            for b in range(w.shape[0])]


FAST_TENSORS = ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n", "surprise", "steps")


def fast_cat(states: Sequence[FastState]) -> FastState:
    return FastState(*(torch.cat([getattr(s, n) for s in states], 0) for n in FAST_TENSORS),
                     [dict(c) for s in states for c in s.channels])


def fast_scatter(full: FastState, rows: Sequence[int], sub: FastState) -> None:
    """Write sub-batch rows back into `full` (in place; tensors are detached state between decisions)."""
    idx = torch.tensor(list(rows), dtype=torch.long, device=full.logw.device)
    for n in FAST_TENSORS:
        dst = getattr(full, n)
        src = getattr(sub, n).detach().to(dst.dtype)
        if src.shape[1:] != dst.shape[1:]:
            raise ShapeError(f"FastState.{n}: sub-batch {tuple(src.shape)} vs state {tuple(dst.shape)}")
        dst[idx] = src
    for j, r in enumerate(rows):
        full.channels[r] = dict(sub.channels[j])


def fast_detach(f: FastState) -> FastState:
    return FastState(*(getattr(f, n).detach() for n in FAST_TENSORS), [dict(c) for c in f.channels])


def fast_where(mask: torch.Tensor, a: FastState, b: FastState) -> FastState:
    """Row-wise select: rows with mask [B] True from `a`, others from `b` (differentiable)."""
    out = []
    for n in FAST_TENSORS:
        ta, tb = getattr(a, n), getattr(b, n)
        m = mask.view(-1, *([1] * (ta.dim() - 1)))
        out.append(torch.where(m, ta.to(tb.dtype), tb))
    ch = [dict(a.channels[i]) if bool(mask[i]) else dict(b.channels[i]) for i in range(mask.shape[0])]
    return FastState(*out, ch)


# ============================================================================ budget state (exploration input)

@dataclass
class BudgetState:
    """Remaining resources per environment (NaN = unknown), all [B] float. Field-for-field the same as workstream D's
    `learn.heads.BudgetState` (duck-typed: D's BudgetEncoder only calls `stack()`)."""
    days_left: torch.Tensor
    funds_left: torch.Tensor
    decisions_left: torch.Tensor
    days_total: torch.Tensor
    funds_total: torch.Tensor
    decisions_total: torch.Tensor

    FIELDS = ("days_left", "funds_left", "decisions_left", "days_total", "funds_total", "decisions_total")

    @staticmethod
    def make(B: int, device=None, **kw) -> "BudgetState":
        vals = {}
        for f in BudgetState.FIELDS:
            v = kw.get(f)
            vals[f] = (torch.full((B,), float("nan"), device=device) if v is None else
                       torch.as_tensor(v, dtype=torch.float32, device=device).reshape(-1).expand(B).clone())
        return BudgetState(**vals)

    @staticmethod
    def from_rows(rows: Sequence[dict], device=None) -> "BudgetState":
        nan = float("nan")
        return BudgetState(*(torch.tensor([float(r.get(f, nan) if r.get(f) is not None else nan) for r in rows],
                                          dtype=torch.float32, device=device) for f in BudgetState.FIELDS))

    def stack(self) -> torch.Tensor:
        return torch.stack([getattr(self, f) for f in self.FIELDS], -1)

    def select(self, idx) -> "BudgetState":
        idx = torch.as_tensor(list(idx) if not torch.is_tensor(idx) else idx, dtype=torch.long,
                              device=self.days_left.device)
        return BudgetState(*(getattr(self, f)[idx] for f in self.FIELDS))

    def to(self, device) -> "BudgetState":
        return BudgetState(*(getattr(self, f).to(device) for f in self.FIELDS))

    @staticmethod
    def cat(items: Sequence["BudgetState"]) -> "BudgetState":
        return BudgetState(*(torch.cat([getattr(i, f) for i in items], 0) for f in BudgetState.FIELDS))

    def row(self, b: int) -> dict:
        return {f: float(getattr(self, f)[b]) for f in self.FIELDS}


# ============================================================================ protocols not (yet) in interfaces.py

class ValueHeadP(Protocol):                   # workstream D: learn/heads.py:ValueHead
    def forward(self, R, goal: TokenSet) -> Any: ...      # -> V [B], or an object with .value [B] (+ .logits)
    # optional: loss(logits, target) -> [B] (distributional value loss; MSE on .value otherwise)


class OptionPolicyP(Protocol):                # workstream D: learn/heads.py:OptionModule
    def step(self, R, goal: TokenSet, prev, steps_prev, greedy: bool = False, generator=None) -> Any: ...
    # -> object with option [B] long, new [B] bool, logp [B], entropy [B], steps [B] long
    def log_prob(self, R, goal: TokenSet, prev, steps_prev, option, new) -> tuple: ...   # (logp [B], entropy [B])
    def token(self, option, steps=None) -> TokenSet: ...                                # OPTION [B,1,D], masked if -1


class ExplorationHeadP(Protocol):             # workstream D: learn/heads.py:BudgetEncoder
    def forward(self, b: BudgetState) -> TokenSet: ...    # -> BUDGET token [B,1,D]


class PlannerP(Protocol):                     # workstream C (adapter needed: world/imagination.py:Imagination)
    def plan(self, model: "CamUnifiedModel", req: "PlanRequest") -> "PlanResult": ...


# Optional component methods used when present (see integration notes):
#   RetrieverP.memory_tokens(ids [B][k], now [B]) -> TokenSet   differentiable MEMORY tokens of stored retrieved ids
#   RetrieverP.loss(seq, Zs) -> scalar                           the retriever's own sequence objective
#   WorldModelP.time_tokens(dt [B], mask [B]) -> TokenSet        TIME(dt_{t-1}) context token (C has it)
#   WorldModelP.predict_duration(Z, act) -> [B] days             learned duration when no hint exists (C has it)
#   WorldModelP.loss(seq, beliefs=[T] TokenSet, memory=[T] TokenSet)   acting-consistent filter context (C has it)
#   BayesLastLayerP.assign_channels / log_prob(fast, phi, ob, head_mean) / combine / prior_var() / noise()  (B has)


# ============================================================================ factory registry

PKG = "superlearner.cam_unified"
DEFAULT_COMPONENT_PATHS = {
    "semantic": f"{PKG}.semantic.encoder:QwenSemantic",                 # A (not an nn.Module; counted separately)
    "tokenizer": f"{PKG}.semantic.tokenizer:ObservationTokenizer",      # A (cfg, encoder)
    "action_generator": f"{PKG}.semantic.actions:ActionGenerator",      # A (cfg, encoder, tokenizer)
    "core": f"{PKG}.core.backbone:UnifiedCore",                         # B (cfg)
    "entity_encoder": f"{PKG}.core.entities:EntityEncoder",             # B (cfg)
    "reasoner": f"{PKG}.core.reasoning:Reasoner",                       # B (cfg, core)
    "beliefs": f"{PKG}.core.beliefs:BeliefModule",                      # B (cfg, n_channels)
    "bayes": f"{PKG}.core.bayes:BayesLastLayer",                        # B (cfg)
    "outcome_head": f"{PKG}.world.outcomes:OutcomeHead",                # C (cfg)
    "world_model": f"{PKG}.world.dynamics:WorldModel",                  # C (cfg, core, outcome_head)
    "planner": f"{PKG}.world.planner:Planner",                          # C: needs a PlannerP adapter (C's
    # world/planner.py has Sampling/Beam/PolicyRolloutPlanner over world/imagination.py:Imagination, a different call
    # shape); never auto-loaded: the built-in LatentImaginationPlanner is used unless components={'planner': ...}
    "retriever": f"{PKG}.memory.retrieval:Retriever",                   # E (cfg, index)
    "value_head": f"{PKG}.learn.heads:ValueHead",                       # D (d_model) -> ValueOutput(.value, .logits)
    "option_policy": f"{PKG}.learn.heads:OptionModule",                 # D (d_model, n_options, max_steps)
    "exploration_head": f"{PKG}.learn.heads:BudgetEncoder",             # D (d_model) forward(BudgetState)
}
# build order = dependency order; constructor parameters named like an earlier component receive it
BUILD_ORDER = ("semantic", "tokenizer", "core", "entity_encoder", "reasoner", "outcome_head", "beliefs", "bayes",
               "world_model", "retriever", "action_generator", "value_head", "option_policy", "exploration_head",
               "planner")
ALIASES = {"semantic_encoder": "semantic", "encoder": "semantic", "sem": "semantic", "backbone": "core",
           "outcomes": "outcome_head", "dynamics": "world_model", "obs_tokenizer": "tokenizer"}
# optional components loaded from their default path when the module exists (absent while the workstream has not
# merged, reported in build_report; an import error *inside* an existing module is raised); the planner is never
# auto-loaded: its `plan(model, PlanRequest)` interface must be confirmed when C merges.
# The pretrained semantic model is a *dependency*, not a component of its own: it is auto-loaded only when another
# component that build_model constructs (from a dotted path or a class) takes it as a constructor parameter (A's
# ObservationTokenizer(cfg, encoder), ActionGenerator(cfg, encoder, tokenizer)). Assemblies whose tokenizer and
# action generator are given as instances never load a 0.6-4 B parameter model they would not use.
AUTO_OPTIONAL = ("semantic", "entity_encoder")
SEMANTIC_PARAMS = frozenset({"semantic", "semantic_encoder", "encoder", "sem"})


def cfg_params(cfg: CamConfig) -> dict:
    """Constructor parameters filled from the config by name (D's heads take sizes, not a CamConfig)."""
    return {"d_model": cfg.d_model, "n_options": cfg.n_options, "max_steps": cfg.option_max_steps}


def required_components(cfg: CamConfig) -> set:
    req = {"tokenizer", "core", "outcome_head", "world_model", "action_generator", "value_head"}
    if cfg.use_reasoning:
        req.add("reasoner")
    if cfg.use_fast_adaptation:
        req |= {"beliefs", "bayes"}
    if cfg.use_memory:
        req.add("retriever")
    if cfg.n_options > 0:
        req.add("option_policy")
    if cfg.use_learned_exploration:
        req.add("exploration_head")
    return req


def resolve_path(path: str):
    """'package.module:Attr' -> the attribute (class or factory). ComponentUnavailable when the module (or one of its
    packages) does not exist; ComponentImportError when it exists but fails to import."""
    if ":" not in path:
        raise ValueError(f"dotted path must look like 'pkg.mod:Attr', got {path!r}")
    mod, attr = path.split(":", 1)
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing and (mod == missing or mod.startswith(missing + ".")):
            raise ComponentUnavailable(f"module {mod!r} for {path!r} does not exist (not merged): {e}") from e
        raise ComponentImportError(f"importing {mod!r} for {path!r} failed inside the module: {e}") from e
    except ImportError as e:
        raise ComponentImportError(f"importing {mod!r} for {path!r} failed: {e}") from e
    try:
        return getattr(m, attr)
    except AttributeError as e:
        raise ComponentImportError(f"{mod!r} exists but has no attribute {attr!r}") from e


def _instantiate(name: str, spec, cfg: CamConfig, built: dict, kwargs: dict):
    """spec: an instance (used as is), a class / factory (called with constructor injection), or a dotted path."""
    if isinstance(spec, str):
        spec = resolve_path(spec)
    if not (inspect.isclass(spec) or inspect.isfunction(spec) or inspect.ismethod(spec)):
        return spec                                   # an already-built instance (nn.Module or plain object)
    sig = inspect.signature(spec)
    args = {}
    from_cfg = cfg_params(cfg)
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        key = ALIASES.get(pname, pname)
        if pname in kwargs:
            args[pname] = kwargs[pname]
        elif pname in ("cfg", "config"):
            args[pname] = cfg
        elif key in built and built[key] is not None:
            args[pname] = built[key]
        elif pname in from_cfg:
            args[pname] = from_cfg[pname]
        elif p.default is p.empty:
            raise ComponentUnavailable(f"{name}: cannot satisfy constructor parameter {pname!r} of {spec!r}")
    return spec(**args)


def available_components(paths: Optional[dict] = None) -> dict:
    """{name: True | 'error message'} for the default (or given) dotted paths: which workstreams have merged."""
    out = {}
    for name, path in {**DEFAULT_COMPONENT_PATHS, **(paths or {})}.items():
        try:
            resolve_path(path)
            out[name] = True
        except (ComponentUnavailable, ComponentImportError) as e:
            out[name] = str(e)
    return out


def _constructor_params(spec) -> set:
    """Parameter names of a component's constructor/factory (empty for an instance or an unresolvable path)."""
    try:
        target = resolve_path(spec) if isinstance(spec, str) else spec
    except (ComponentUnavailable, ComponentImportError):
        return set()
    if not (inspect.isclass(target) or inspect.isfunction(target) or inspect.ismethod(target)):
        return set()
    return _params_of(target)


def semantic_needed(cfg: CamConfig, components: dict, paths: dict) -> bool:
    """True when a component build_model will construct (not an instance) takes the semantic encoder."""
    need = required_components(cfg)
    for name in BUILD_ORDER:
        if name == "semantic":
            continue
        if name in components:
            spec = components[name]
        elif name in need or name in AUTO_OPTIONAL:
            spec = paths.get(name)
        else:
            continue
        if spec is None:
            continue
        if _constructor_params(spec) & SEMANTIC_PARAMS:
            return True
    return False


def build_model(cfg: CamConfig, components: Optional[dict] = None, paths: Optional[dict] = None,
                component_kwargs: Optional[dict] = None, **model_kwargs) -> "CamUnifiedModel":
    """Assemble CamUnifiedModel. `components` maps a name to an instance, class/factory or dotted path (overrides
    `paths`, which overrides DEFAULT_COMPONENT_PATHS). A component set to None is left out (allowed only when it is
    optional for `cfg`). The belief module's Kalman channel count is sized to the outcome-key budget
    (N_CONTINUOUS_CORE + max_numeric_outcomes) when its constructor takes `n_channels` and none is given.
    The semantic encoder is auto-loaded (default path, or `paths["semantic"]`) only when a component built here
    takes it (see SEMANTIC_PARAMS); pass components={"semantic": obj} to inject one.
    Raises ComponentUnavailable listing every required component that cannot be built; import errors inside existing
    modules propagate (ComponentImportError)."""
    components = dict(components or {})
    paths = {**DEFAULT_COMPONENT_PATHS, **(paths or {})}
    component_kwargs = {k: dict(v) for k, v in (component_kwargs or {}).items()}
    n_ch = N_CONTINUOUS_CORE + int(model_kwargs.get("max_numeric_outcomes", 32))
    component_kwargs.setdefault("beliefs", {})
    need = required_components(cfg)
    built: dict = {}
    report: dict = {}
    missing = []
    sem_auto = "semantic" not in components and semantic_needed(cfg, components, paths)
    for name in BUILD_ORDER:
        if name in components:
            spec = components[name]
            if spec is None:
                built[name] = None
                report[name] = "absent: set to None"
                continue
        elif name == "semantic" and not sem_auto:
            built[name] = None
            report[name] = "absent: no component built here takes a semantic encoder"
            continue
        elif name in need or name in AUTO_OPTIONAL:
            spec = paths.get(name)
        else:
            built[name] = None
            report[name] = "absent: not required by the config" if name != "planner" else \
                "built-in LatentImaginationPlanner (pass components={'planner': ...} to override)"
            continue
        kw = component_kwargs.get(name, {})
        try:
            if name == "beliefs" and "n_channels" not in kw:
                target = resolve_path(spec) if isinstance(spec, str) else spec
                if inspect.isclass(target) and "n_channels" in inspect.signature(target).parameters:
                    kw = {**kw, "n_channels": n_ch}
            built[name] = _instantiate(name, spec, cfg, built, kw)
            report[name] = f"built from {spec if isinstance(spec, str) else type(built[name]).__name__}"
        except ComponentUnavailable as e:
            built[name] = None
            report[name] = f"absent: {e}"
            if name in need:
                missing.append(f"{name}: {e}")
    absent = sorted(n for n in need if built.get(n) is None and not any(m.startswith(n + ":") for m in missing))
    missing += [f"{n}: explicitly set to None but required by the config" for n in absent]
    if missing:
        raise ComponentUnavailable("CAM Unified components unavailable:\n  " + "\n  ".join(missing))
    model = CamUnifiedModel(cfg, **{k: v for k, v in built.items() if v is not None}, **model_kwargs)
    model.build_report = report
    return model


# ============================================================================ state and decision records

@dataclass
class CamState:
    """Per-environment working state (batched). Z [B,S,D]; fast FastState (None without fast adaptation);
    fast_prev: the belief before the last update (hindsight learning progress); option [B] long (-1 = none);
    option_steps [B] long (decisions the active option has run, including the last); last_now [B] float (time of
    the previous decision, -1 = none)."""
    Z: torch.Tensor
    fast: Optional[FastState]
    registries: list
    trackers: list
    option: torch.Tensor
    option_steps: torch.Tensor
    last_actions: list
    last_now: torch.Tensor
    fast_prev: Optional[FastState] = None

    @property
    def B(self) -> int:
        return self.Z.shape[0]

    def select(self, rows: Sequence[int]) -> "CamState":
        idx = torch.tensor(list(rows), dtype=torch.long, device=self.Z.device)
        return CamState(self.Z[idx], None if self.fast is None else self.fast.select(idx),
                        [self.registries[r] for r in rows], [self.trackers[r] for r in rows],
                        self.option[idx].clone(), self.option_steps[idx].clone(),
                        [self.last_actions[r] for r in rows], self.last_now[idx].clone(),
                        None if self.fast_prev is None else self.fast_prev.select(idx))

    def scatter(self, rows: Sequence[int], sub: "CamState") -> None:
        idx = torch.tensor(list(rows), dtype=torch.long, device=self.Z.device)
        self.Z[idx] = sub.Z.detach().to(self.Z.dtype)
        self.option[idx] = sub.option
        self.option_steps[idx] = sub.option_steps
        self.last_now[idx] = sub.last_now.to(self.last_now.dtype)
        if self.fast is not None and sub.fast is not None:
            fast_scatter(self.fast, rows, sub.fast)
        if sub.fast_prev is not None:
            if self.fast_prev is None:
                self.fast_prev = self.fast.clone()
            fast_scatter(self.fast_prev, rows, sub.fast_prev)
        for j, r in enumerate(rows):
            self.registries[r] = sub.registries[j]
            self.trackers[r] = sub.trackers[j]
            self.last_actions[r] = sub.last_actions[j]


@dataclass
class StepInputs:
    """Everything needed to recompute a decision's forward pass with gradients (policy objective on fresh data): the
    stored detached Z_{t-1} (truncated BPTT of one decision), a FastState snapshot, an entity-registry snapshot taken
    before tokenisation (identical slot binding), raw observations / tools / goals / previous actions, the retrieved
    memory (tokens for fallback, ids to rebuild them with gradients), dt_{t-1}, the budget and the option decision."""
    Z_prev: torch.Tensor
    fast: Optional[FastState]
    registries: list
    obs_lists: list
    now: list
    last_actions: list
    goals: list
    tools: list
    memory: Optional[TokenSet]
    memory_ids: list
    dt_prev: torch.Tensor            # [B] days since the previous decision (-1 = first decision of the episode)
    budget: Optional[BudgetState]
    option_prev: torch.Tensor        # [B] long
    option_steps_prev: torch.Tensor  # [B] long
    option: torch.Tensor             # [B] long
    option_new: torch.Tensor         # [B] bool
    option_steps: torch.Tensor       # [B] long

    @property
    def B(self) -> int:
        return self.Z_prev.shape[0]

    TENSORS = ("dt_prev", "option_prev", "option_steps_prev", "option", "option_new", "option_steps")

    def select(self, js: Sequence[int]) -> "StepInputs":
        js = list(js)
        idx = torch.tensor(js, dtype=torch.long, device=self.Z_prev.device)
        return StepInputs(self.Z_prev[idx], None if self.fast is None else self.fast.select(idx),
                          [self.registries[j] for j in js], [self.obs_lists[j] for j in js],
                          [self.now[j] for j in js], [self.last_actions[j] for j in js], [self.goals[j] for j in js],
                          [self.tools[j] for j in js], None if self.memory is None else self.memory.select(idx),
                          [self.memory_ids[j] for j in js], self.dt_prev[idx],
                          None if self.budget is None else self.budget.select(idx),
                          *(getattr(self, n)[idx] for n in self.TENSORS[1:]))

    @staticmethod
    def cat(items: Sequence["StepInputs"]) -> "StepInputs":
        mem = None
        if any(i.memory is not None for i in items):
            D = items[0].Z_prev.shape[2]
            dev = items[0].Z_prev.device
            mem = cat_rows([i.memory if i.memory is not None else TokenSet.empty(i.B, D, dev) for i in items])
        fasts = [i.fast for i in items]
        bud = None
        if all(i.budget is not None for i in items):
            bud = BudgetState.cat([i.budget for i in items])
        return StepInputs(torch.cat([i.Z_prev for i in items], 0), None if fasts[0] is None else fast_cat(fasts),
                          [r for i in items for r in i.registries], [o for i in items for o in i.obs_lists],
                          [n for i in items for n in i.now], [a for i in items for a in i.last_actions],
                          [g for i in items for g in i.goals], [t for i in items for t in i.tools], mem,
                          [m for i in items for m in i.memory_ids],
                          torch.cat([i.dt_prev for i in items], 0), bud,
                          *(torch.cat([getattr(i, n) for i in items], 0) for n in StepInputs.TENSORS[1:]))


@dataclass
class StepEncoding:
    obs: TokenSet           # O_t (+ entity/relation tokens) [B,No,D]
    goal: TokenSet          # G_t [B,Ng,D]
    belief: TokenSet        # B_t [B,K+1,D] (empty without fast adaptation)
    memory: Optional[TokenSet]
    option_tok: Optional[TokenSet]   # OPTION token of the previous option (context of F and Reason)
    ctx: TokenSet           # budget-fitted context for F_theta
    Z: torch.Tensor         # Z_t [B,S,D]
    R: torch.Tensor         # R_t [B,S,D]
    rinfo: dict
    dropped: list           # [B] {token kind: n} left out by the context budget


@dataclass
class PlanRequest:
    """Inputs of a planner call (all rows are real environments). R/Z [B,S,D]; goal_in = [G; option; budget]."""
    Z: torch.Tensor
    R: torch.Tensor
    fast: Optional[FastState]
    goal_in: TokenSet
    goal: TokenSet
    belief: TokenSet
    memory: Optional[TokenSet]
    tools: TokenSet
    obs: TokenSet
    tool_lists: list
    obs_lists: list
    prospective: OutcomeBatch
    dt: torch.Tensor
    n_candidates: int
    horizon: int
    n_samples: int
    generator: Optional[torch.Generator] = None
    candidates: Optional[list] = None    # [B*n] pre-drawn CAM candidates (row b*n+i); None = the planner samples them


@dataclass
class PlanResult:
    actions: list                               # [B] chosen Action (source CAM_PLANNER, proposal filled)
    candidates: list                            # [B] list[{"action", "score", "logp", "source"}]
    imagined: list                              # [B] list[{"candidate", "horizon", "value", ...}]
    scores: torch.Tensor                        # [B, n_candidates]
    trajectory: Optional[ImaginedTrajectory] = None


@dataclass
class Decision:
    actions: list
    inputs: StepInputs
    Z: torch.Tensor                 # Z_t [B,S,D] (detached)
    R: torch.Tensor
    Z_hat: torch.Tensor             # Prior(Z_t, A_t, B_t, dt) [B,S,D]
    act_tokens: TokenSet            # selected actions' tokens [B,Na,D]
    ctx: TokenSet
    prospective_specs: list         # [B] list[OutcomeSpec]
    pred_mean: torch.Tensor         # [B,Q]
    pred_var: torch.Tensor          # [B,Q]
    predicted: list                 # [B] {key: mean}
    uncertainty: list               # [B] {key: var}
    value: torch.Tensor             # [B]
    logp: torch.Tensor              # [B] behaviour log-probability recorded with the action (see `behavior`)
    dt: torch.Tensor                # [B] expected days until the next decision (think time + expected duration)
    reasoning: list                 # [B] {"iters", "halt_p"}
    retrieved: list                 # [B] [{"id", "score"}]
    candidates: list
    imagined: list
    belief_before: list
    behavior: list                  # [B] behaviour-policy provenance {"policy", "logp_kind", "selection", ...}
    dropped: list                   # [B] {"ctx_tokens": {kind: n}} budget drops of this decision
    trajectory: Optional[ImaginedTrajectory] = None
    timings: dict = field(default_factory=dict)


@dataclass
class FilterResult:
    """Acting-consistent filter over a replayed sequence: Zs [T] of [B,S,D]; beliefs [T] TokenSet (B_t with the
    previous option token, as passed to the world model); memory [T] TokenSet or None."""
    Zs: list
    beliefs: list
    memory: list
    n_belief_updates: int = 0
    memory_rebuilt: bool = False


@dataclass
class ReplayAux:
    """Per-step side information of a tokenised replay sequence (from the archived records): [T][B] lists."""
    retrieved: list          # stored retrieved memory ids
    now: list                # decision times (days)
    option_prev: list        # option active before the decision (-1 = none)
    option_steps_prev: list


# ============================================================================ the model

class acting_mode:
    """Context manager: put `module` in eval mode (the acting computation) and restore its mode afterwards."""

    def __init__(self, module):
        self.module = module
        self.was = None

    def __enter__(self):
        if isinstance(self.module, nn.Module):
            self.was = self.module.training
            self.module.eval()
        return self.module

    def __exit__(self, *exc):
        if self.was is not None:
            self.module.train(self.was)
        return False


def _acting(fn):
    """Method decorator: run in acting (eval) mode whatever mode the caller left the model in."""

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with acting_mode(self):
            return fn(self, *a, **k)
    return wrapper


DURATION_ARG = re.compile(r"(?:^|_)(seconds|secs|minutes|mins|hours|days)$")
DURATION_DAYS = {"seconds": 1 / 86400, "secs": 1 / 86400, "minutes": 1 / 1440, "mins": 1 / 1440, "hours": 1 / 24,
                 "days": 1.0}


class CamUnifiedModel(nn.Module):
    """The single trainable CAM Unified network: injected components + outcome query encoder.

    Required: tokenizer, core, outcome_head, world_model, action_generator, value_head; reasoner with use_reasoning;
    beliefs + bayes with use_fast_adaptation; retriever with use_memory; option_policy with n_options > 0;
    exploration_head with use_learned_exploration. Optional: semantic (counted separately as pretrained),
    entity_encoder, planner (else `LatentImaginationPlanner`).
    Extra knobs (constructor kwargs): imag_candidates / imag_horizon / imag_samples (default from cfg), unc_coef
    (uncertainty penalty), n_error_classes, money_unit, max_numeric_outcomes, max_event_types, amp (bf16 autocast;
    default on for CUDA), reason_iters (fixed J; None = adaptive), reward_in_beliefs (whether the realised verified
    reward increment is an observed outcome for the belief / Kalman updates; design decision, default True: the
    evaluator's scalar score increment is the only evaluator output that reaches the learner, as a verified return,
    and conditioning beliefs on it is standard meta-RL; False keeps it as a world-model target only)."""

    def __init__(self, cfg: CamConfig, *, tokenizer, core, outcome_head, world_model, action_generator, value_head,
                 semantic=None, entity_encoder=None, reasoner=None, beliefs=None, bayes=None, retriever=None,
                 option_policy=None, exploration_head=None, planner=None, query_encoder=None,
                 imag_candidates: Optional[int] = None, imag_horizon: Optional[int] = None,
                 imag_samples: Optional[int] = None, unc_coef: float = 0.1, n_error_classes: int = 16,
                 money_unit: str = "cents", max_numeric_outcomes: int = 32, max_event_types: int = 16,
                 amp: Optional[bool] = None, reason_iters: Optional[int] = None, reward_in_beliefs: bool = True):
        super().__init__()
        self.cfg = cfg
        self._version = 0
        self.weights_epoch = 0
        self.build_report: dict = {}
        need = required_components(cfg)
        given = dict(tokenizer=tokenizer, core=core, outcome_head=outcome_head, world_model=world_model,
                     action_generator=action_generator, value_head=value_head, reasoner=reasoner, beliefs=beliefs,
                     bayes=bayes, retriever=retriever, option_policy=option_policy, exploration_head=exploration_head)
        lacking = sorted(n for n in need if given.get(n) is None)
        if lacking:
            raise ComponentUnavailable(f"CamUnifiedModel: config requires {lacking}")
        self.semantic = semantic
        self.tokenizer = tokenizer
        self.core = core
        self.entity_encoder = entity_encoder
        self.reasoner = reasoner
        self.beliefs = beliefs
        self.bayes = bayes
        self.outcome_head = outcome_head
        self.world_model = world_model
        self.retriever = retriever
        self.action_generator = action_generator
        self.value_head = value_head
        self.option_policy = option_policy if cfg.n_options > 0 else None
        self.exploration_head = exploration_head
        self.planner = planner if planner is not None else LatentImaginationPlanner()
        self.query_encoder = query_encoder if query_encoder is not None else OutcomeQueryEncoder(
            cfg.d_model, make_text_encoder(tokenizer, cfg.d_model, lambda: self.weights_epoch))
        self.imag_candidates = imag_candidates or cfg.imag_branches
        self.imag_horizon = imag_horizon or cfg.imag_horizons[0]
        self.imag_samples = imag_samples or cfg.imag_samples
        self.unc_coef = unc_coef
        self.n_error_classes = n_error_classes
        self.money_unit = money_unit
        self.max_numeric_outcomes = max_numeric_outcomes
        self.max_event_types = max_event_types
        self.amp = amp
        self.reason_iters = reason_iters
        self.reward_in_beliefs = reward_in_beliefs
        self.think_days = cfg.think_s / 86400.0
        self.calls: dict = {}         # component call counters and budget-drop counters (observability)
        self._retrieve_now = retriever is not None and "now" in _params_of(retriever.retrieve)
        self._wm_loss_ctx = {n for n in ("beliefs", "memory") if n in _params_of(world_model.loss)}
        self._bayes_allocates = bayes is not None and hasattr(bayes, "assign_channels")
        self.register_load_state_dict_post_hook(lambda module, incompatible: module.bump_weights_epoch())

    # ---------------------------------------------------------------- bookkeeping
    @property
    def version(self) -> int:
        """Number of learner updates. Assigning it (checkpoint loads reset it) also bumps the weights epoch."""
        return self._version

    @version.setter
    def version(self, v: int) -> None:
        self._version = int(v)
        self.bump_weights_epoch()

    def bump_weights_epoch(self) -> None:
        """Weights changed (optimiser step, load, version assignment): invalidate weight-dependent caches."""
        self.weights_epoch += 1
        te = getattr(self.query_encoder, "text_encoder", None)
        if hasattr(te, "clear"):
            te.clear()

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def _count(self, name: str, n: int = 1) -> None:
        self.calls[name] = self.calls.get(name, 0) + n

    def _autocast(self):
        dev = self.device
        on = self.amp if self.amp is not None else dev.type == "cuda"
        return torch.autocast(dev.type, dtype=torch.bfloat16) if on else nullcontext()

    @staticmethod
    def _is_semantic(obj) -> bool:
        return all(hasattr(obj, a) for a in ("hidden", "embed", "sem_dim"))

    def _semantic_objects(self) -> list:
        """Pretrained semantic encoders: nn.Modules and plain objects (A's QwenSemantic is deliberately not a
        Module) reachable from the model or its components."""
        out = []
        cands = [self.semantic] + [m for m in self.modules() if m is not self]
        for m in list(self.modules()):
            for a in ("semantic", "encoder", "sem", "semantic_encoder"):
                v = m.__dict__.get(a)
                if v is not None and not isinstance(v, nn.Module):
                    cands.append(v)
        for c in cands:
            if c is not None and self._is_semantic(c) and all(c is not o for o in out):
                out.append(c)
        return out

    @staticmethod
    def _semantic_params(obj) -> tuple:
        """(frozen pretrained count, trainable tensors) of one semantic encoder."""
        if isinstance(obj, nn.Module):
            ps = list(obj.parameters())
            return sum(p.numel() for p in ps if not p.requires_grad), [p for p in ps if p.requires_grad]
        lora = list(obj.lora_parameters()) if callable(getattr(obj, "lora_parameters", None)) else []
        if callable(getattr(obj, "num_parameters", None)):
            base = int(obj.num_parameters()) - sum(p.numel() for p in lora)
        elif isinstance(getattr(obj, "n_pretrained", None), int):
            base = int(obj.n_pretrained)
        else:
            lm = getattr(obj, "lm", None) or getattr(obj, "model", None)
            base = sum(p.numel() for p in lm.parameters() if not p.requires_grad) if isinstance(lm, nn.Module) else 0
        return base, lora

    def count_parameters(self) -> dict:
        """Unique parameters per component (shared modules, e.g. the backbone inside the world model, are counted
        once, at the first component in BUILD_ORDER that holds them). The pretrained semantic model (any object
        implementing SemanticEncoder, nn.Module or not) is reported separately, with its trainable adapters (LoRA),
        and excluded from the CAM totals."""
        seen = set()
        sem_base = sem_train = 0
        for obj in self._semantic_objects():
            base, train = self._semantic_params(obj)
            sem_base += base
            if isinstance(obj, nn.Module):
                for p in obj.parameters():
                    seen.add(id(p))
            for p in train:
                if id(p) not in seen or isinstance(obj, nn.Module):
                    sem_train += p.numel()
                seen.add(id(p))
        comps = {}
        for name in BUILD_ORDER + ("query_encoder",):
            mod = getattr(self, name, None)
            if not isinstance(mod, nn.Module):
                continue
            tot = tr = 0
            for p in mod.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                tot += p.numel()
                tr += p.numel() if p.requires_grad else 0
            comps[name] = {"total": tot, "trainable": tr}
        rest = [p for p in self.parameters() if id(p) not in seen]
        if rest:
            comps["other"] = {"total": sum(p.numel() for p in rest),
                              "trainable": sum(p.numel() for p in rest if p.requires_grad)}
        return {"components": comps,
                "cam_total": sum(c["total"] for c in comps.values()),
                "cam_trainable": sum(c["trainable"] for c in comps.values()),
                "semantic_pretrained": {"total": sem_base + sem_train, "trainable": sem_train, "frozen": sem_base}}

    def frozen_semantic_keys(self) -> tuple:
        """State-dict keys of frozen pretrained semantic parameters held as nn.Modules (excluded from weight files;
        re-loaded from cfg.sem_model)."""
        sem = [o for o in self._semantic_objects() if isinstance(o, nn.Module)]
        ids = {id(p) for o in sem for p in o.parameters() if not p.requires_grad}
        return tuple(n for n, p in self.named_parameters(remove_duplicate=False) if id(p) in ids)

    # ---------------------------------------------------------------- state
    @torch.no_grad()
    def init_state(self, B: int, device=None, seed: int = 0) -> CamState:
        dev = device or self.device
        Z = self.core.init_state(B, device=dev)
        check_shape(Z, (B, self.cfg.n_state, self.cfg.d_model), "core.init_state")
        fast = self.init_fast(B, dev)
        regs = [EntityRegistry(self.cfg.max_entities, seed=seed * 100003 + b) for b in range(B)]
        trackers = [self.new_tracker() for _ in range(B)]
        return CamState(Z.detach().float(), fast, regs, trackers, torch.full((B,), -1, dtype=torch.long, device=dev),
                        torch.zeros(B, dtype=torch.long, device=dev), [None] * B, torch.full((B,), -1.0, device=dev))

    @torch.no_grad()
    def init_fast(self, B: int, device=None) -> Optional[FastState]:
        if not self.cfg.use_fast_adaptation:
            return None
        fast = fast_detach(self.beliefs.init_fast(B, device=device or self.device))
        check_shape(fast.logw, (B, self.cfg.n_particles), "beliefs.init_fast logw")
        return fast

    def new_tracker(self) -> OutcomeTracker:
        return OutcomeTracker(self.money_unit, self.n_error_classes, self.max_numeric_outcomes, self.max_event_types)

    def empty_tokens(self, B: int) -> TokenSet:
        return TokenSet.empty(B, self.cfg.d_model, device=self.device)

    @property
    def n_channels(self) -> int:
        """Kalman channel slots per environment (0 without fast adaptation)."""
        if not self.cfg.use_fast_adaptation:
            return 0
        return int(self.beliefs.init_fast(1, device=self.device).bayes_mu.shape[1])

    # ---------------------------------------------------------------- component adapters
    @staticmethod
    def _value(out) -> torch.Tensor:
        return (out.value if hasattr(out, "value") else out).float()

    def _exploration_token(self, budget: BudgetState, enc: StepEncoding) -> TokenSet:
        names = _params_of(self.exploration_head.forward)
        avail = {"b": budget, "budget": budget, "R": enc.R, "goal": enc.goal, "belief": enc.belief}
        kw = {n: avail[n] for n in names if n in avail}
        return self.exploration_head(**kw)

    def _time_tokens(self, dt_prev: torch.Tensor) -> Optional[TokenSet]:
        """TIME(dt_{t-1}) context token from the world model's time encoder (masked on an episode's first step)."""
        if not hasattr(self.world_model, "time_tokens"):
            return None
        valid = dt_prev >= 0
        ts = self.world_model.time_tokens(dt_prev.clamp(min=0).float(), valid)
        check_tokens(ts, dt_prev.shape[0], self.cfg.d_model, "world_model.time_tokens")
        return with_mask(as_kind(ts, TokenKind.TIME), valid)

    def _option_token(self, option: torch.Tensor, steps: torch.Tensor) -> Optional[TokenSet]:
        if self.option_policy is None:
            return None
        ts = self.option_policy.token(option, steps)
        check_tokens(ts, option.shape[0], self.cfg.d_model, "option_policy.token")
        return with_mask(as_kind(ts, TokenKind.OPTION), option >= 0)

    def memory_tokens(self, ids: list, now: Optional[list] = None) -> Optional[TokenSet]:
        """Differentiable MEMORY tokens of stored retrieved ids (None when the retriever cannot rebuild them)."""
        if self.retriever is None or not hasattr(self.retriever, "memory_tokens"):
            return None
        M = self.retriever.memory_tokens(ids, now)
        check_tokens(M, len(ids), self.cfg.d_model, "retriever.memory_tokens")
        return as_kind(M, TokenKind.MEMORY)

    # ---------------------------------------------------------------- the step equation
    def _goal_and_belief(self, goals: list, fast: Optional[FastState], B: int) -> tuple:
        G = check_tokens(self.tokenizer.goal(goals), B, self.cfg.d_model, "tokenizer.goal")
        if self.cfg.use_fast_adaptation and fast is not None:
            self._count("beliefs.tokens")
            Bt = check_tokens(self.beliefs.tokens(fast), B, self.cfg.d_model, "beliefs.tokens")
        else:
            Bt = self.empty_tokens(B)
        return G, Bt

    def retrieve(self, Z_prev: torch.Tensor, G: TokenSet, Bt: TokenSet, filters: Optional[list] = None,
                 now: Optional[list] = None) -> tuple:
        """M_t = Retrieve(Q(Z_{t-1}, G_t, B_t)) -> (TokenSet MEMORY [B,<=k,D], ids [B][k], scores)."""
        B = Z_prev.shape[0]
        self._count("retriever.retrieve")
        q = self.retriever.query(Z_prev, G, Bt)
        if q.dim() != 2 or q.shape[0] != B:
            raise ShapeError(f"retriever.query: {tuple(q.shape)} != [B={B}, key_dim]")
        if self._retrieve_now:
            M, ids, scores = self.retriever.retrieve(q, self.cfg.mem_top_k, filters, now=now)
        else:
            M, ids, scores = self.retriever.retrieve(q, self.cfg.mem_top_k, filters)
        check_tokens(M, B, self.cfg.d_model, "retriever.retrieve")
        return as_kind(M, TokenKind.MEMORY), ids, scores

    def context(self, O: TokenSet, A_prev: Optional[TokenSet], TIME: Optional[TokenSet], Bt: TokenSet,
                opt_tok: Optional[TokenSet], M: Optional[TokenSet], G: TokenSet,
                dropped: Optional[list] = None) -> TokenSet:
        """The core's working context [O; A_{t-1}; TIME; B; option; M; G], fitted to cfg.max_ctx_tokens. Used for
        acting, policy recomputation and replay filtering alike."""
        ctx = TokenSet.cat(O, A_prev, TIME, Bt, opt_tok, M if self.cfg.use_memory else None, G)
        return fit_budget(ctx, self.cfg.max_ctx_tokens, dropped)

    def encode_step(self, inp: StepInputs, pre: Optional[tuple] = None, memory: Optional[TokenSet] = None
                    ) -> StepEncoding:
        """Z_t = F(Z_{t-1}, [O_t; A_{t-1}; TIME; B_t; option_{t-1}; M_t; G_t]) and R_t = Reason(Z_t, [B; option; M;
        G]). `inp.registries` are bound in place by the tokenizer (pass copies to recompute). `memory` overrides the
        stored memory tokens (rebuilt with gradients)."""
        cfg, B, D = self.cfg, inp.B, self.cfg.d_model
        O = check_tokens(self.tokenizer.observations(inp.obs_lists, inp.registries, inp.now), B, D,
                         "tokenizer.observations")
        if self.entity_encoder is not None:
            E = check_tokens(self.entity_encoder(O), B, D, "entity_encoder")
            O = TokenSet.cat(O, E)
        A_prev = check_tokens(self.tokenizer.action(inp.last_actions), B, D, "tokenizer.action")
        TIME = self._time_tokens(inp.dt_prev.to(inp.Z_prev.device))
        G, Bt = pre if pre is not None else self._goal_and_belief(inp.goals, inp.fast, B)
        opt_tok = self._option_token(inp.option_prev, inp.option_steps_prev)
        M = (memory if memory is not None else inp.memory) if cfg.use_memory else None
        dropped = [dict() for _ in range(B)]
        ctx = self.context(O, A_prev, TIME, Bt, opt_tok, M, G, dropped)
        self._count("core")
        Z, ctx_out = self.core(inp.Z_prev, ctx)
        check_shape(Z, (B, cfg.n_state, D), "core.forward Z")
        if cfg.use_reasoning:
            rctx = TokenSet.cat(Bt, opt_tok, M, G)
            self._count("reasoner")
            R, rinfo = self.reasoner(Z, rctx, self.reason_iters)
            check_shape(R, (B, cfg.n_state, D), "reasoner R")
        else:
            R, rinfo = Z, {"iters": torch.zeros(B, device=Z.device), "halt_p": None, "ponder": None}
        return StepEncoding(O, G, Bt, M, opt_tok, ctx, Z, R, rinfo, dropped)

    def _goal_inputs(self, enc: StepEncoding, option_tok: Optional[TokenSet], budget: Optional[BudgetState]
                     ) -> TokenSet:
        """[G_t; OPTION(o_t); BUDGET(b_t)]: the action generator's / value's conditioning."""
        parts = [enc.goal, option_tok]
        if self.cfg.use_learned_exploration and self.exploration_head is not None:
            B = enc.R.shape[0]
            bud = budget if budget is not None else BudgetState.make(B, device=enc.R.device)
            self._count("exploration_head")
            tok = check_tokens(self._exploration_token(bud.to(enc.R.device), enc), B, self.cfg.d_model,
                               "exploration_head")
            parts.append(as_kind(tok, TokenKind.BUDGET))
        return TokenSet.cat(*parts)

    def prospective_batch(self, state: CamState) -> tuple:
        """Outcome queries of the next action; channels are looked up only (prospective keys never allocate)."""
        specs = [tr.prospective() for tr in state.trackers]
        chans = [lookup_channels(state.fast, b, specs[b]) for b in range(state.B)]
        return specs, build_outcome_batch(specs, self.query_encoder, state.registries, chans, self.device)

    # ---------------------------------------------------------------- outcome predictions (particles + Kalman)
    def _particle_dist(self, feat: torch.Tensor, ob: OutcomeBatch, fast: Optional[FastState]):
        if self.cfg.use_fast_adaptation and fast is not None:
            return self.outcome_head.dist(feat, ob, cond=fast.particles)
        return self.outcome_head.dist(feat, ob)

    def base_moments(self, Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch) -> tuple:
        """Outcome-head predictive mean/var [B,Q] from the action-conditioned latent Z_hat: the mixture over belief
        particles (weights w) with fast adaptation, the unconditioned head otherwise. Also returns the features."""
        B, Q = ob.lik.shape
        feat = self.outcome_head.features(Z_hat, ob)
        check_shape(feat, (B, Q, self.cfg.d_model), "outcome_head.features")
        d = self._particle_dist(feat, ob, fast)
        if self.cfg.use_fast_adaptation and fast is not None:
            mk, vk = dist_mean(d).float(), dist_var(d).float()
            check_shape(mk, (B, self.cfg.n_particles, Q), "outcome_head.dist(cond) mean")
            w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
            mean = (w * mk).sum(1)
            var = (w * (vk + mk * mk)).sum(1) - mean * mean
        else:
            mean, var = dist_mean(d).float(), dist_var(d).float()
            check_shape(mean, (B, Q), "outcome_head.dist mean")
        return mean, var.clamp_min(0.0), feat

    @staticmethod
    def _continuous(ob: OutcomeBatch) -> torch.Tensor:
        c = torch.zeros_like(ob.y_mask)
        for k in CONTINUOUS:
            c = c | (ob.lik == k)
        return c

    def _kalman_on(self) -> bool:
        return self.cfg.use_fast_adaptation and self.bayes is not None

    def channels_of(self, fast: FastState, ob: OutcomeBatch) -> OutcomeBatch:
        """The same queries with `channel` looked up in fast.channels by key (continuous keys only)."""
        B, Q = ob.lik.shape
        cont = self._continuous(ob).cpu()
        ch = torch.full((B, Q), -1, dtype=torch.long)
        for b in range(B):
            table = fast.channels[b]
            keys = ob.keys[b] if ob.keys is not None else []
            for q, k in enumerate(keys):
                if k is not None and bool(cont[b, q]):
                    ch[b, q] = int(table.get(k, -1))
        return OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y, ob.censored, ob.y_mask, ch.to(ob.lik.device), ob.keys)

    def kalman_bound(self, ob: OutcomeBatch) -> torch.Tensor:
        return self._continuous(ob) & (ob.channel >= 0) & ob.q.mask

    def kalman_predict(self, fast: FastState, feat: torch.Tensor, ob: OutcomeBatch) -> tuple:
        """Residual correction of the Kalman/NIG last layer: (corr_mean [B,Q], corr_var [B,Q], phi [B,Q,F],
        bound [B,Q])."""
        B, Q = ob.lik.shape
        phi = self.bayes.features(feat, ob)
        cm, cv = self.bayes.predict(fast, phi, ob)
        check_shape(cm, (B, Q), "bayes.predict mean")
        return cm.float(), cv.float().clamp_min(0.0), phi, self.kalman_bound(ob)

    def combine(self, head_mean, head_var, corr_mean, corr_var, bound) -> tuple:
        """Bound continuous queries: (head + correction mean, the Kalman predictive variance); others: the head.
        (B's BayesLastLayer.combine when present: same semantics.)"""
        if hasattr(self.bayes, "combine"):
            m, v = self.bayes.combine(head_mean, head_var, corr_mean, corr_var, bound)
            return m.float(), v.float()
        return torch.where(bound, head_mean + corr_mean, head_mean), torch.where(bound, corr_var, head_var)

    def kalman_log_prob(self, fast: FastState, phi: torch.Tensor, ob: OutcomeBatch, head_mean: torch.Tensor
                        ) -> torch.Tensor:
        """log p(y | head mean + Kalman residual predictive) [B,Q] (0 where unbound/unobserved). Differentiable in
        phi, head_mean and the Kalman statistics (hence the learned prior / noise)."""
        if hasattr(self.bayes, "log_prob"):
            return self.bayes.log_prob(fast, phi, ob, head_mean).float()
        m, v = self.bayes.predict(fast, phi, ob)
        v = v.float().clamp_min(1e-8)
        r = ob.y.float() - head_mean.float()
        lp = -0.5 * (math.log(2 * math.pi) + torch.log(v) + (r - m.float()) ** 2 / v)
        return torch.where(self.kalman_bound(ob) & ob.y_mask, lp, torch.zeros_like(lp))

    def residual_batch(self, ob: OutcomeBatch, base_mean: torch.Tensor) -> OutcomeBatch:
        """Targets r = y - head mean on observed continuous queries (the Kalman layer's regression target)."""
        cont = self._continuous(ob) & ob.y_mask
        return with_targets(ob, torch.where(cont, ob.y.float() - base_mean.float(), torch.zeros_like(ob.y.float())),
                            cont)

    def kalman_update(self, fast: FastState, feat: torch.Tensor, ob: OutcomeBatch, base_mean: torch.Tensor
                      ) -> FastState:
        """Kalman/NIG update with the residuals of `ob` against `base_mean` (differentiable closed form when the
        Bayes layer is). The layer allocates slots for new observed keys when it can (`assign_channels`)."""
        resid = self.residual_batch(ob, base_mean)
        phi = self.bayes.features(feat, resid)
        return self.bayes.update(fast, phi, resid)

    def predict_outcomes(self, Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch) -> tuple:
        """Predictive mean/var [B,Q]: base_moments combined, on continuous outcomes with a bound channel, with the
        Kalman/NIG last layer (mean head + phi^T mu_c, variance = Kalman predictive variance)."""
        mean, var, feat = self.base_moments(Z_hat, fast, ob)
        if self._kalman_on() and fast is not None:
            cm, cv, _, bound = self.kalman_predict(fast, feat, ob)
            mean, var = self.combine(mean, var, cm, cv, bound)
        return mean, var, feat

    def belief_view(self, ob: OutcomeBatch) -> OutcomeBatch:
        """The outcomes the belief / Kalman updates condition on (the realised reward is masked unless
        reward_in_beliefs)."""
        if self.reward_in_beliefs or ob.keys is None:
            return ob
        m = ob.y_mask.clone()
        for b, keys in enumerate(ob.keys):
            for q, k in enumerate(keys):
                if k == "reward":
                    m[b, q] = False
        return OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y, ob.censored, m, ob.channel, ob.keys)

    def mixture_ll(self, fast: FastState, feat: torch.Tensor, ob: OutcomeBatch, use: torch.Tensor) -> torch.Tensor:
        """log sum_k w_k prod_{q in use} p(y_q | feat, m_k) [B] (0 where nothing is used)."""
        B, Q = ob.lik.shape
        d = self._particle_dist(feat, ob, fast)
        K = fast.logw.shape[1]
        y = ob.y.unsqueeze(1).expand(B, K, Q)
        cens = None if ob.censored is None else ob.censored.unsqueeze(1).expand(B, K, Q)
        lp = dist_log_prob(d, y, cens).float()
        ll = torch.where(use.unsqueeze(1), lp, torch.zeros_like(lp)).sum(-1)              # [B,K]
        return torch.logsumexp(torch.log_softmax(fast.logw.float(), -1) + ll, -1)

    def mixture_loglik(self, fast: FastState, Z_hat: torch.Tensor, ob: OutcomeBatch) -> torch.Tensor:
        """Particle-mixture log p(y | B) per env [B] divided by the number of observed outcomes (per-outcome nats)."""
        feat = self.outcome_head.features(Z_hat, ob)
        return self.mixture_ll(fast, feat, ob, ob.y_mask) / ob.y_mask.float().sum(-1).clamp_min(1.0)

    # ---------------------------------------------------------------- deciding
    @torch.no_grad()
    @_acting
    def decide(self, state: CamState, obs_lists: list, now: list, tools: list, goals: list,
               budget: Optional[BudgetState] = None, greedy: bool = False,
               generator: Optional[torch.Generator] = None, retrieval_filters: Optional[list] = None) -> Decision:
        """One decision for every row of `state` (all rows must be live environments), computed in acting (eval)
        mode. Mutates state.Z, option, last_now and the registries (entity binding); beliefs change only in
        observe_outcomes."""
        cfg = self.cfg
        B, D = state.B, cfg.d_model
        dev = state.Z.device
        if not (len(obs_lists) == len(now) == len(tools) == len(goals) == B):
            raise ShapeError("decide: obs/now/tools/goals must all have B entries")
        regs_snapshot = copy.deepcopy(state.registries)
        now_t = torch.tensor([float(x) for x in now], device=dev)
        dt_prev = torch.where(state.last_now >= 0, (now_t - state.last_now).clamp(min=0), torch.full_like(now_t, -1))
        with self._autocast():
            G, Bt = self._goal_and_belief(goals, state.fast, B)
            M, ids, scores = (None, [[] for _ in range(B)], None)
            if cfg.use_memory:
                M, ids, scores = self.retrieve(state.Z, G, Bt, retrieval_filters, list(now))
            inp = StepInputs(state.Z.detach(), None if state.fast is None else state.fast.clone(), regs_snapshot,
                             obs_lists, list(now), list(state.last_actions), list(goals), list(tools),
                             detach_tokens(M), [list(i) for i in ids], dt_prev, budget, state.option.clone(),
                             state.option_steps.clone(), state.option.clone(),
                             torch.zeros(B, dtype=torch.bool, device=dev), state.option_steps.clone())
            inp.registries = state.registries          # live registries bind new entities; snapshot kept below
            enc = self.encode_step(inp, pre=(G, Bt))
            inp.registries = regs_snapshot
            for d in enc.dropped:
                for k, n in d.items():
                    self._count(f"ctx_dropped/{k}", n)
            opt_logp = torch.zeros(B, device=dev)
            opt_tok = None
            if self.option_policy is not None:
                self._count("option_policy.step")
                ost = self.option_policy.step(enc.R, enc.goal, state.option, state.option_steps, greedy,
                                              gen_on(generator, dev))
                check_shape(ost.option, (B,), "option_policy.step option")
                inp.option, inp.option_new, inp.option_steps = ost.option.long(), ost.new.bool(), ost.steps.long()
                opt_logp = ost.logp.float()
                opt_tok = self._option_token(inp.option, inp.option_steps)
                state.option, state.option_steps = inp.option.clone(), inp.option_steps.clone()
            goal_in = self._goal_inputs(enc, opt_tok, budget)
            T = check_tokens(self.tokenizer.tools(tools), B, D, "tokenizer.tools")
            specs, prosp = self.prospective_batch(state)
            dt0 = torch.full((B,), self.think_days, device=dev)
            plan = None
            if cfg.use_imagination:
                self._count("planner")
                req = PlanRequest(enc.Z, enc.R, state.fast, goal_in, enc.goal, enc.belief, enc.memory, T, enc.obs,
                                  list(tools), list(obs_lists), prosp, dt0, self.imag_candidates, self.imag_horizon,
                                  self.imag_samples, generator)
                plan = self.planner.plan(self, req)
                actions = plan.actions
            else:
                self._count("action_generator.sample")
                actions = self.action_generator.sample(enc.R, goal_in, enc.belief, T, enc.obs, list(tools),
                                                       list(obs_lists), temperature=1.0, greedy=greedy)
            if len(actions) != B:
                raise ShapeError(f"action generator/planner returned {len(actions)} actions for B={B}")
            logp = torch.zeros(B, device=dev)
            behavior = []
            for b, a in enumerate(actions):
                if a.logp is None:
                    raise ShapeError("every CAM action must carry its behaviour log-probability (Action.logp)")
                if self.option_policy is not None:
                    a.option_id = int(state.option[b])
                    a.logp_parts["option"] = float(opt_logp[b])
                    a.logp = float(a.logp) + float(opt_logp[b])
                logp[b] = float(a.logp)
                behavior.append(self.behavior_of(a, greedy))
            A_tok = check_tokens(self.tokenizer.action(actions), B, D, "tokenizer.action(selected)")
            dt = self.expected_dt(enc.Z, A_tok, actions, tools)
            self._count("world_model.prior")
            Z_hat, _ = self.world_model.prior(enc.Z, A_tok, enc.belief, dt, sample=False)
            check_shape(Z_hat, (B, cfg.n_state, D), "world_model.prior Z_hat")
            mean, var, _ = self.predict_outcomes(Z_hat, state.fast, prosp)
            value = self._value(self.value_head(enc.R, enc.goal))
            check_shape(value, (B,), "value_head")
        state.Z = enc.Z.detach().float()
        state.last_actions = list(actions)
        state.last_now = now_t
        predicted, uncertainty = [], []
        for b in range(B):
            predicted.append({s.key: float(mean[b, q]) for q, s in enumerate(specs[b])})
            uncertainty.append({s.key: float(var[b, q]) for q, s in enumerate(specs[b])})
        halt = enc.rinfo.get("halt_p")
        iters = enc.rinfo.get("iters")
        reasoning = [{"iters": float(iters[b]) if iters is not None else 0.0,
                      "halt_p": [] if halt is None else [float(v) for v in halt[b].float()]} for b in range(B)]
        retrieved = [[{"id": str(i), "score": float(scores[b][j]) if scores is not None else 0.0}
                      for j, i in enumerate(ids[b])] for b in range(B)]
        if plan is None:
            cands = [[{"action": a.to_dict(), "score": None, "logp": a.logp, "source": a.source.value}]
                     for a in actions]
            imagined, traj = [[] for _ in range(B)], None
        else:
            cands, imagined, traj = plan.candidates, plan.imagined, plan.trajectory
        return Decision(actions, inp, enc.Z.detach().float(), enc.R.detach().float(), Z_hat.detach().float(),
                        detach_tokens(A_tok), detach_tokens(enc.ctx), specs, mean, var, predicted, uncertainty,
                        value, logp, dt, reasoning, retrieved, cands, imagined,
                        belief_summary(None if state.fast is None else state.fast.logw) or [{} for _ in range(B)],
                        behavior, [{"ctx_tokens": dict(d)} for d in enc.dropped], traj)

    @staticmethod
    def behavior_of(a: Action, greedy: bool) -> dict:
        """Behaviour-policy provenance of an executed CAM action (spec 12.3). For a planner choice the recorded logp
        is the *generator's* proposal log-probability of the chosen candidate (+ option), not the probability of the
        argmax selection; the policy objective routes such decisions to an estimator that needs no behaviour ratio."""
        if a.source == ActionSource.CAM_PLANNER:
            p = a.proposal or {}
            return {"policy": "planner_" + str(p.get("selection", "argmax")), "logp_kind": "generator_proposal",
                    "selection": p.get("selection", "argmax"), "n_candidates": p.get("n")}
        return {"policy": "decoder_greedy" if greedy else "decoder_sample",
                "logp_kind": "deterministic" if greedy else "exact", "selection": None, "n_candidates": 1}

    def _arg_duration(self, a: Action) -> Optional[float]:
        """Days implied by duration-typed arguments (e.g. a wait tool's `seconds`), capped at cfg.max_wait_s."""
        best = None
        for k, v in (a.args or {}).items():
            m = DURATION_ARG.search(str(k).lower())
            if m and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                d = min(max(0.0, float(v) * DURATION_DAYS[m.group(1)]), self.cfg.max_wait_s / 86400.0)
                best = d if best is None else max(best, d)
        return best

    def expected_dt(self, Z: torch.Tensor, A_tok: TokenSet, actions: list, tools: list) -> torch.Tensor:
        """Expected days until the next decision [B]: think time + the action's expected duration (its own
        expected_duration, else duration-typed arguments, else the tool's duration_hint, else the world model's
        learned duration when it has one, else 0)."""
        B = len(actions)
        out, need = [], []
        for b, (a, tl) in enumerate(zip(actions, tools)):
            d = a.expected_duration
            if d is None:
                d = self._arg_duration(a)
            if d is None:
                hint = [t.duration_hint for t in tl if t.name == a.tool and t.duration_hint is not None]
                d = float(hint[0]) if hint else None
            if d is None:
                need.append(b)
            out.append(0.0 if d is None else float(d))
        dt = torch.tensor(out, device=Z.device) + self.think_days
        if need and hasattr(self.world_model, "predict_duration"):
            idx = torch.tensor(need, dtype=torch.long, device=Z.device)
            pd = self.world_model.predict_duration(Z[idx], A_tok.select(idx)).float().detach().clamp(min=0)
            dt[idx] = dt[idx] + pd
        return dt

    # ---------------------------------------------------------------- outcomes -> beliefs
    @torch.no_grad()
    @_acting
    def observe_outcomes(self, state: CamState, decision: Decision, results: list, next_obs: list,
                         rewards: Sequence[float]) -> dict:
        """Realise the outcomes of `decision` (OutcomeBatch with y), compute the hindsight learning progress of the
        previous belief update, then update the particle beliefs and the Kalman/NIG last layer (use_fast_adaptation).
        Kalman channels: slots are freed for the realised continuous keys by least-recently-realised eviction
        (`make_room`), then allocated by the Bayes layer (B's `assign_channels`) or by the model; evictions and keys
        that do not fit are counted. Returns {"outcomes", "specs", "belief_info", "lp" [B] or None, "belief_after",
        "loglik" [B], "channels" [B] {"evicted", "unbound"}, "dropped" [B]}."""
        cfg = self.cfg
        B = state.B
        specs = [state.trackers[b].realize(decision.inputs.obs_lists[b], decision.actions[b], results[b], next_obs[b],
                                           float(rewards[b])) for b in range(B)]
        dropped = [dict(state.trackers[b].last_dropped) for b in range(B)]
        ch_stats = [{"evicted": 0, "unbound": 0} for _ in range(B)]
        if self._kalman_on() and state.fast is not None:
            pv, nv = self._bayes_prior()
            chans = []
            for b in range(B):
                if self._bayes_allocates:
                    make_room(state.fast, b, specs[b], state.trackers[b].last_seen, pv, nv, ch_stats[b])
                    chans.append(lookup_channels(state.fast, b, specs[b]))
                else:
                    chans.append(bind_channels(state.fast, b, specs[b], state.trackers[b].last_seen, pv, nv,
                                               ch_stats[b]))
        else:
            chans = [[-1] * len(s) for s in specs]
        for d in dropped:
            for k, n in d.items():
                self._count(f"outcome_dropped/{k}", n)
        with self._autocast():
            ob = build_outcome_batch(specs, self.query_encoder, state.registries, chans, self.device)
            out = {"outcomes": ob, "specs": specs, "belief_info": {}, "lp": None, "loglik": None,
                   "channels": ch_stats, "dropped": dropped}
            if cfg.use_fast_adaptation and state.fast is not None:
                obb = self.belief_view(ob)
                ll_now = self.mixture_loglik(state.fast, decision.Z_hat, obb)
                out["loglik"] = ll_now
                if cfg.use_learned_exploration and state.fast_prev is not None:
                    # LP_{t-1} = log p(o_t | B_t) - log p(o_t | B_{t-1}): o_t is disjoint from the support o_{t-1}
                    out["lp"] = (ll_now - self.mixture_loglik(state.fast_prev, decision.Z_hat, obb)).float()
                before = state.fast.clone()
                # Kalman target: residual of y against the same base prediction the residual is added to
                base_mean, _, feat = self.base_moments(decision.Z_hat, before, obb)
                self._count("beliefs.update")
                fast1, binfo = self.beliefs.update(state.fast, decision.Z, decision.act_tokens, obb, self.outcome_head)
                check_shape(fast1.logw, (B, cfg.n_particles), "beliefs.update logw")
                self._count("bayes.update")
                fast2 = self.kalman_update(fast1, feat, obb, base_mean)
                state.fast_prev = before
                state.fast = fast_detach(fast2)
                out["belief_info"] = {k: v for k, v in (binfo or {}).items() if k != "feat"}
                for b in range(B):
                    table = state.fast.channels[b]
                    ch_stats[b]["unbound"] = sum(1 for s in specs[b] if s.observed and int(s.lik) in CONTINUOUS
                                                 and (s.key != "reward" or self.reward_in_beliefs)
                                                 and s.key not in table)
        for st in ch_stats:
            self._count("channels.evicted", st["evicted"])
            self._count("channels.unbound", st["unbound"])
        out["belief_after"] = belief_summary(None if state.fast is None else state.fast.logw) or [{} for _ in range(B)]
        return out

    def _bayes_prior(self) -> tuple:
        """(prior variance [F] or scalar, noise) a freed slot is reset to: the Bayes layer's learned values when it
        exposes them, the config initial values otherwise."""
        pv, nv = self.cfg.bayes_prior_var_init, self.cfg.bayes_obs_var_init
        if callable(getattr(self.bayes, "prior_var", None)):
            pv = self.bayes.prior_var().detach().double()
        if callable(getattr(self.bayes, "noise", None)):
            nv = float(self.bayes.noise().detach())
        return pv, nv

    # ---------------------------------------------------------------- training-time recomputation
    def policy_evaluate(self, inp: StepInputs, actions: list) -> dict:
        """Differentiable recomputation of stored decisions: total behaviour log-probability of the executed actions
        (generator + option parts), entropy, value V(R, G) (+ distributional logits when the head has them) and the
        reasoner's ponder cost, all [B]. The step is recomputed with the reasoner in its acting (eval) mode, so the
        recomputed log-probability is that of the policy that acted (B's reasoner halts hard when acting and mixes
        iterations in training); with adaptive reasoning a second, train-mode reasoner pass on the same inputs
        provides the differentiable ponder cost. Memory tokens are rebuilt from the stored ids when the retriever can
        (so the memory content projection is trained by the decision objectives); otherwise the stored tokens are
        used."""
        regs = copy.deepcopy(inp.registries)
        live = StepInputs(**{**inp.__dict__, "registries": regs})
        B = inp.B
        cfg = self.cfg
        with self._autocast():
            mem = None
            if cfg.use_memory and any(inp.memory_ids):
                mem = self.memory_tokens(inp.memory_ids, inp.now)
            train_reasoner = self.reasoner is not None and self.reasoner.training
            with acting_mode(self):          # every module as when acting (B's reasoner: hard halting, not the mixture)
                enc = self.encode_step(live, memory=mem)
            halting = None
            if (cfg.use_reasoning and cfg.adaptive_reasoning and self.reason_iters is None and train_reasoner):
                # the halting policy's own differentiable compute cost (train-mode pass on the same inputs)
                _, rinfo_t = self.reasoner(enc.Z, TokenSet.cat(enc.belief, enc.option_tok, enc.memory, enc.goal),
                                           None)
                halting = rinfo_t
            opt_tok = None
            opt_lp = torch.zeros(B, device=enc.R.device)
            opt_ent = torch.zeros(B, device=enc.R.device)
            if self.option_policy is not None:
                with acting_mode(self):
                    opt_lp, opt_ent = self.option_policy.log_prob(enc.R, enc.goal, inp.option_prev,
                                                                  inp.option_steps_prev, inp.option, inp.option_new)
                    opt_tok = self._option_token(inp.option, inp.option_steps)
            with acting_mode(self):
                goal_in = self._goal_inputs(enc, opt_tok, inp.budget)
                T = self.tokenizer.tools(inp.tools)
                logp, ent = self.action_generator.log_prob(enc.R, goal_in, enc.belief, T, enc.obs, inp.tools,
                                                           inp.obs_lists, actions)
                vout = self.value_head(enc.R, enc.goal)
        check_shape(logp, (B,), "action_generator.log_prob")
        ponder = (halting if halting is not None else enc.rinfo).get("ponder")
        if ponder is not None and not ponder.requires_grad:
            ponder = None
        logits = getattr(vout, "logits", None)
        return {"logp": logp.float() + opt_lp.float(), "entropy": ent.float() + opt_ent.float(),
                "value": self._value(vout), "value_logits": None if logits is None else logits.float(),
                "ponder": None if ponder is None else ponder.float(), "memory_rebuilt": mem is not None}

    @torch.no_grad()
    def policy_probe(self, inp: StepInputs, actions: list) -> dict:
        """The policy at fixed stored decisions (probe contexts) under the *current* weights, computed exactly as
        when acting: per row the log-probability parts of the recorded actions (tool / evidence / typed-head / token
        parts when the generator exposes `log_prob_parts`), the entropies of the corresponding distributions (the
        tool entropy is the entropy of the full tool-pointer distribution over the row's available tools), the
        option log-probability and the value V(R, G). Weight changes that change decisions show up here on a context
        that stays fixed (spec 18.2 'changes in action distributions')."""
        regs = copy.deepcopy(inp.registries)
        live = StepInputs(**{**inp.__dict__, "registries": regs})
        B = inp.B
        with acting_mode(self), self._autocast():
            enc = self.encode_step(live)
            opt_tok, opt_lp = None, torch.zeros(B, device=enc.R.device)
            if self.option_policy is not None:
                opt_lp, _ = self.option_policy.log_prob(enc.R, enc.goal, inp.option_prev, inp.option_steps_prev,
                                                        inp.option, inp.option_new)
                opt_tok = self._option_token(inp.option, inp.option_steps)
            goal_in = self._goal_inputs(enc, opt_tok, inp.budget)
            T = self.tokenizer.tools(inp.tools)
            ag = self.action_generator
            if hasattr(ag, "log_prob_parts"):
                lp, en = ag.log_prob_parts(enc.R, goal_in, enc.belief, T, enc.obs, inp.tools, inp.obs_lists, actions)
            else:
                lp0, en0 = ag.log_prob(enc.R, goal_in, enc.belief, T, enc.obs, inp.tools, inp.obs_lists, actions)
                lp, en = {"total": lp0}, {"total": en0}
            value = self._value(self.value_head(enc.R, enc.goal))
        rows = []
        for b in range(B):
            rows.append({"logp": {k: float(v[b]) for k, v in lp.items()},
                         "logp_total": float(sum(float(v[b]) for v in lp.values())),
                         "entropy": {k: float(v[b]) for k, v in en.items()}, "option_logp": float(opt_lp[b]),
                         "value": float(value[b]), "reason_iters": float(enc.rinfo["iters"][b])
                         if enc.rinfo.get("iters") is not None else None})
        return {"rows": rows, "version": self.version}

    def filter_sequence(self, seq, aux: Optional[ReplayAux] = None) -> FilterResult:
        """Run F_theta over a tokenised SequenceBatch with the acting context: Z_t = F(Z_{t-1}, [O_t; A_{t-1};
        TIME(dt_{t-1}); B_t; option_{t-1}; M_t; G_t]), where B_t is re-filtered along the sequence by the belief module
        from the stored outcomes (B_t has seen o_{<t}), M_t is rebuilt from the stored retrieved ids (when the
        retriever can) and option_{t-1} comes from the stored option decisions. Rows whose previous step ended an
        episode restart from the initial state and a fresh belief."""
        cfg = self.cfg
        steps = seq.steps
        B = steps[0].obs.B
        dev = steps[0].obs.x.device
        Z0 = self.core.init_state(B, device=dev).float()
        Z = Z0
        fast = self.beliefs.init_fast(B, device=dev) if cfg.use_fast_adaptation else None
        prev, prev_done = None, torch.zeros(B, dtype=torch.bool, device=dev)
        res = FilterResult([], [], [])
        with self._autocast():
            for t, st in enumerate(steps):
                if bool(prev_done.any()):
                    Z = torch.where(prev_done.view(B, 1, 1), Z0.to(Z.dtype), Z)
                    if fast is not None:
                        fast = fast_where(prev_done, self.beliefs.init_fast(B, device=dev), fast)
                alive = ~prev_done
                Bt = self.beliefs.tokens(fast) if fast is not None else self.empty_tokens(B)
                opt_tok = None
                if self.option_policy is not None and aux is not None:
                    o = torch.tensor(aux.option_prev[t], dtype=torch.long, device=dev)
                    s = torch.tensor(aux.option_steps_prev[t], dtype=torch.long, device=dev)
                    opt_tok = self._option_token(torch.where(alive, o, torch.full_like(o, -1)), s)
                M = None
                if cfg.use_memory and aux is not None and any(aux.retrieved[t]):
                    M = self.memory_tokens(aux.retrieved[t], aux.now[t])
                    res.memory_rebuilt = res.memory_rebuilt or M is not None
                A_prev = None if prev is None else with_mask(prev.act, alive)
                dtp = (torch.full((B,), -1.0, device=dev) if prev is None else
                       torch.where(alive, prev.dt.float().to(dev), torch.full((B,), -1.0, device=dev)))
                TIME = self._time_tokens(dtp)
                ctx = self.context(st.obs, A_prev, TIME, Bt, opt_tok, M, st.goal)
                Z, _ = self.core(Z, ctx)
                Z = Z.float()
                res.Zs.append(Z)
                res.beliefs.append(TokenSet.cat(Bt, opt_tok))
                res.memory.append(M if M is not None else self.empty_tokens(B))
                if fast is not None and bool(st.outcomes.y_mask.any()):
                    fast, _ = self.beliefs.update(fast, Z, st.act, self.belief_view(st.outcomes), self.outcome_head)
                    res.n_belief_updates += 1
                prev, prev_done = st, st.done.to(torch.bool).to(dev)
        return res

    def world_loss(self, seq, filt: Optional[FilterResult]) -> dict:
        """WorldModelP.loss with the acting-consistent belief / memory context when the world model accepts it."""
        kw = {}
        if filt is not None:
            if "beliefs" in self._wm_loss_ctx:
                kw["beliefs"] = filt.beliefs
            if "memory" in self._wm_loss_ctx and self.cfg.use_memory:
                kw["memory"] = filt.memory
        return self.world_model.loss(seq, **kw)

    def adapt_loss(self, seq, filt: FilterResult, k: int) -> tuple:
        """L_adapt = -log p(Q | B_S, query actions) (spec 12.2). The support belief B_S is inferred from the first k
        steps by the belief module (particles) *and* the Kalman/NIG last layer (residuals of the observed continuous
        outcomes against the particle-mixture head mean, differentiable closed form); the later steps' observed
        outcomes (queries, disjoint from the support) are then scored under B_S without further updates: outcomes on
        a bound Kalman channel by the combined head + Kalman predictive (`kalman_log_prob`), all others by the
        particle mixture. Gradients reach the core (via Z), outcome head, world-model prior, belief module, the
        Kalman feature map Phi_theta and the Bayes layer's learned prior / noise. Returns (loss, metrics)."""
        steps = seq.steps
        T = len(steps)
        B = steps[0].obs.B
        Zs = filt.Zs
        dev = Zs[0].device
        fast = self.beliefs.init_fast(B, device=dev)
        fast.channels = [{} for _ in range(B)]
        kal = self._kalman_on()
        n_kal = n_mix = 0
        with self._autocast():
            for t in range(k):
                st = steps[t]
                ob = self.belief_view(st.outcomes)
                if not bool(ob.y_mask.any()):
                    continue
                Bt = self.beliefs.tokens(fast)
                Z_hat, _ = self.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
                ob = self.channels_of(fast, ob)
                base_mean, _, feat = self.base_moments(Z_hat, fast, ob)
                fast_n, _ = self.beliefs.update(fast, Zs[t], st.act, ob, self.outcome_head)
                if kal:
                    fast_n = self.kalman_update(fast_n, feat, ob, base_mean)
                fast = fast_n
            Bt = self.beliefs.tokens(fast)
            nll, n = Zs[0].new_zeros(()).float(), 0
            for t in range(k, T):
                st = steps[t]
                valid = st.outcomes.y_mask.any(-1) & ~self._padded(st)
                if not bool(valid.any()):
                    continue
                Z_hat, _ = self.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
                ob = self.channels_of(fast, st.outcomes)
                base_mean, _, feat = self.base_moments(Z_hat, fast, ob)
                kb = self.kalman_bound(ob) & ob.y_mask if kal else torch.zeros_like(ob.y_mask)
                ll = self.mixture_ll(fast, feat, ob, ob.y_mask & ~kb)
                if bool(kb.any()):
                    phi = self.bayes.features(feat, ob)
                    lpk = self.kalman_log_prob(fast, phi, ob, base_mean)
                    ll = ll + torch.where(kb, lpk, torch.zeros_like(lpk)).sum(-1)
                n_kal += int(kb.sum())
                n_mix += int((ob.y_mask & ~kb).sum())
                ll = ll / ob.y_mask.float().sum(-1).clamp_min(1.0)
                nll = nll - (ll * valid.float()).sum() / valid.float().sum()
                n += 1
        loss = nll / max(1, n)
        return loss, {"adapt_k": k, "adapt_T": T, "adapt_queries": n, "adapt_nll": float(loss.detach()),
                      "adapt_kalman_outcomes": n_kal, "adapt_mixture_outcomes": n_mix}

    @staticmethod
    def _padded(st) -> torch.Tensor:
        return ~st.obs.mask.any(-1) & ~st.act.mask.any(-1)


def _params_of(fn) -> set:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


# ============================================================================ built-in planner (protocol-only)

class LatentImaginationPlanner:
    """Belief-aware multistep imagination over CAM-generated candidates (spec 7.1-7.2), composed only from the
    contract protocols: n candidate first actions from the ActionGenerator; for each, `n_samples` stochastic futures
    of `horizon` decisions: Z_hat = WorldModel.prior(Z, EncAction(a), B, dt) (stochastic latent), outcomes sampled from
    the particle-conditioned outcome head combined with the Kalman/NIG last layer on bound continuous outcomes, an
    *independent clone* of the FastState updated with the imagined outcomes (BeliefModule.update and the Kalman
    update), the reasoner re-run on the imagined latent, and the next action sampled from the policy conditioned on
    the revised belief. Return = sum_h gamma_h * reward_hat_h + gamma_H * V(R_H) (gamma = exp(-kappa dt)) minus
    unc_coef * (particle disagreement on reward); the real action is the argmax candidate (recorded as
    selection="argmax" with one-hot probs; its logp stays the generator's proposal log-probability). Nothing here
    touches the environment, the archive or the real FastState. Workstream C's planner replaces it via the registry."""

    def plan(self, model: CamUnifiedModel, req: PlanRequest) -> PlanResult:
        cfg = model.cfg
        B, n, H, ns = req.R.shape[0], req.n_candidates, req.horizon, req.n_samples
        dev = req.R.device
        rep = torch.arange(B, device=dev).repeat_interleave(n)                           # candidate rows [B*n]
        ag = model.action_generator
        if req.candidates is not None:
            if len(req.candidates) != B * n:
                raise ShapeError(f"PlanRequest.candidates: {len(req.candidates)} actions for B*n={B * n}")
            cands = list(req.candidates)
        else:
            model._count("action_generator.sample")
            cands = ag.sample(req.R[rep], req.goal_in.select(rep), req.belief.select(rep), req.tools.select(rep),
                              req.obs.select(rep), [req.tool_lists[int(i)] for i in rep],
                              [req.obs_lists[int(i)] for i in rep], temperature=1.0, greedy=False)
        rows = rep.repeat_interleave(ns)                                                  # rollout rows [B*n*ns]
        cand_of_row = torch.arange(B * n, device=dev).repeat_interleave(ns)
        Bb = rows.shape[0]
        Z = req.Z[rows]
        fast = req.fast.select(rows) if (cfg.use_fast_adaptation and req.fast is not None) else None
        kal = model._kalman_on() and fast is not None
        goal_in, goal = req.goal_in.select(rows), req.goal.select(rows)
        tools_t, obs_t = req.tools.select(rows), req.obs.select(rows)
        tool_lists = [req.tool_lists[int(i)] for i in rows]
        obs_lists = [req.obs_lists[int(i)] for i in rows]
        mem = None if req.memory is None else req.memory.select(rows)
        prosp = select_outcomes(req.prospective, rows.tolist())
        r_idx = CORE_INDEX["reward"]
        acts = [cands[int(c)] for c in cand_of_row]
        ret = torch.zeros(Bb, device=dev)
        pen = torch.zeros(Bb, device=dev)
        disc = torch.ones(Bb, device=dev)
        steps, first_out = [], None
        R = req.R[rows]
        for h in range(H):
            act_tok = model.tokenizer.action(acts)
            dt = model.expected_dt(Z, act_tok, acts, tool_lists)
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            model._count("world_model.prior")
            Z_hat, _ = model.world_model.prior(Z, act_tok, belief, dt, sample=True)
            feat = model.outcome_head.features(Z_hat, prosp)
            if fast is not None:
                d = model.outcome_head.dist(feat, prosp, cond=fast.particles)
                mk = dist_mean(d).float()                                               # [Bb,K,Q]
                w = torch.softmax(fast.logw.float(), -1)
                k = torch.multinomial(w.cpu(), 1, generator=req.generator).view(Bb).to(dev)
                ys = dist_sample(d, gen_on(req.generator, dev)).float()[torch.arange(Bb, device=dev), k]   # [Bb,Q]
                base = (w.unsqueeze(-1) * mk).sum(1)
                var = (w.unsqueeze(-1) * (dist_var(d).float() + mk * mk)).sum(1) - base * base
                disagree = ((w * (mk[:, :, r_idx] - base[:, r_idx:r_idx + 1]) ** 2).sum(1)).sqrt()
                mean = base
                if kal:
                    prosp_c = model.channels_of(fast, prosp)
                    cm, cv, _, bound = model.kalman_predict(fast, feat, prosp_c)
                    mean, var = model.combine(base, var.clamp_min(0), cm, cv, bound)
                    eps = randn_like_gen(mean, req.generator)
                    ys = torch.where(bound, mean + var.clamp_min(0).sqrt() * eps, ys)
                    model._count("imagination.bayes.predict")
            else:
                d = model.outcome_head.dist(feat, prosp)
                mean, var = dist_mean(d).float(), dist_var(d).float()
                ys = dist_sample(d, gen_on(req.generator, dev)).float()
                disagree = var[:, r_idx].clamp_min(0).sqrt()
            ret = ret + disc * mean[:, r_idx]
            pen = pen + disc * disagree
            disc = disc * torch.exp(-cfg.kappa * dt)
            if h == 0:
                first_out = mean
            imagined = with_targets(prosp, ys, prosp.q.mask.clone())
            if fast is not None:
                model._count("imagination.beliefs.update")
                fast_n, _ = model.beliefs.update(fast, Z, act_tok, model.belief_view(imagined), model.outcome_head)
                if kal:
                    model._count("imagination.bayes.update")
                    fast_n = model.kalman_update(fast_n, feat, model.belief_view(model.channels_of(fast, imagined)),
                                                 base)
                fast = fast_n
            Z = Z_hat
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            if cfg.use_reasoning:
                R, _ = model.reasoner(Z, TokenSet.cat(belief, mem, goal), model.reason_iters)
            else:
                R = Z
            v = model._value(model.value_head(R, goal))
            if fast is not None:
                wn = torch.softmax(fast.logw.float(), -1)
                ess = 1.0 / (wn * wn).sum(-1)
            else:
                ess = torch.ones(Bb, device=dev)
            steps.append(ImaginedStep(Z.detach(), list(acts), act_tok, mean.detach(), var.detach(), v.detach(),
                                      mean[:, r_idx].detach(), ess, disagree.detach()))
            if h + 1 < H:
                acts = ag.sample(R, goal_in, belief, tools_t, obs_t, tool_lists, obs_lists, temperature=1.0)
        ret = ret + disc * steps[-1].value
        score_row = ret - model.unc_coef * pen
        scores = score_row.view(B, n, ns).mean(-1)                                       # [B,n]
        choice = scores.argmax(-1)
        actions, cand_out, imag_out = [], [], []
        for b in range(B):
            c = int(choice[b])
            a = copy.deepcopy(cands[b * n + c])
            a.source = ActionSource.CAM_PLANNER
            a.logp_parts["select"] = 0.0                       # deterministic given the imagined scores
            a.proposal = {"n": n, "selection": "argmax", "probs": [1.0 if i == c else 0.0 for i in range(n)],
                          "chosen": c, "scores": [float(s) for s in scores[b]],
                          "generator_logp": [float(cands[b * n + i].logp) for i in range(n)]}
            actions.append(a)
            cand_out.append([{"action": cands[b * n + i].to_dict(), "score": float(scores[b, i]),
                              "logp": cands[b * n + i].logp, "source": ActionSource.CAM_PLANNER.value}
                             for i in range(n)])
            items = []
            for i in range(n):
                sl = slice((b * n + i) * ns, (b * n + i + 1) * ns)
                keys = req.prospective.keys[b] if req.prospective.keys else []
                items.append({"candidate": i, "horizon": H, "value": float(scores[b, i]),
                              "return": float(ret[sl].mean()), "model_error": float(pen[sl].mean()),
                              "outcomes": {k: float(first_out[sl, q].mean()) for q, k in enumerate(keys[:8])
                                           if k is not None},
                              "next_tools": [steps[1].action[j].tool for j in range(sl.start, sl.stop)]
                              if H > 1 else []})
            imag_out.append(items)
        traj = ImaginedTrajectory(steps, ret.detach(), rows.detach(), cand_of_row.remainder(n).detach())
        return PlanResult(actions, cand_out, imag_out, scores.detach(), traj)
