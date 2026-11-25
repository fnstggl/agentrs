"""CamUnifiedModel: the assembled CAM Unified intelligence (spec Parts 1.2, 12, 20 A22; architecture §1).

One decision implements the step equation

    O_t   = Tokenize(new observations)  (+ ENTITY / RELATION tokens from the entity encoder)
    M_t   = Retrieve(Q(Z_{t-1}, G_t, B_t))                                  (use_memory)
    Z_t   = F_theta(Z_{t-1}, [O_t; A_{t-1}; B_t; M_t; G_t; option_{t-1}])    (shared backbone)
    R_t   = Reason_theta(Z_t, [B_t; M_t; G_t; option])                      (use_reasoning; R_t = Z_t otherwise)
    o_t   ~ pi_option(R_t, G_t) with termination beta(R_t, o_{t-1})         (option policy, when present)
    b_t   = ExploreBudget(R_t, G_t, B_t)                                    (use_learned_exploration)
    a_t   ~ pi(a | R_t, [G_t; o_t; b_t], B_t, tools, O_t)                   (CAM action generator), or
    a_t   = argmax over CAM-generated candidates of imagined, belief-aware returns (use_imagination)
    Y_hat = OutcomeHead(Prior(Z_t, A_t, B_t, dt))  mixed over belief particles (+ Kalman residual)

and `observe_outcomes` performs the belief (particle) and Bayes-last-layer updates from the realised outcomes.

Every component is injected (dependency injection against the contract protocols) or built by `build_model` from a
dotted-path registry, so the classes of workstreams A-G are wired in by path once their branches merge:
`DEFAULT_COMPONENT_PATHS` lists exactly what is expected. This module never imports test doubles.

Shapes: B environments, S state tokens, N context tokens, D = d_model, K particles, Q outcome queries, T tools.
"""
from __future__ import annotations

import copy
import importlib
import inspect
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import torch
import torch.nn as nn

from .config import CamConfig
from .interfaces import (Action, ActionSource, EntityRegistry, FastState, ImaginedStep, ImaginedTrajectory,
                         OutcomeBatch, TokenKind, TokenSet)
from .outcome_spec import (CONTINUOUS, CORE_INDEX, OutcomeQueryEncoder, OutcomeTracker, bind_channels,
                           build_outcome_batch, make_text_encoder, select_outcomes, with_targets)


# ============================================================================ errors and small helpers

class ShapeError(ValueError):
    """A tensor crossing a component boundary violates the contract shape."""


class ComponentUnavailable(ImportError):
    """A component's dotted path cannot be imported (its workstream has not merged yet) or constructed."""


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
                 int(TokenKind.ACTION): 8, int(TokenKind.BELIEF): 8, int(TokenKind.EVENT): 6,
                 int(TokenKind.OBS_NUM): 5, int(TokenKind.OBS_SEM): 5, int(TokenKind.ENTITY): 4,
                 int(TokenKind.RELATION): 4, int(TokenKind.MEMORY): 3}


def fit_budget(ts: TokenSet, budget: int) -> TokenSet:
    """Keep at most `budget` context tokens per row (the per-step working-memory budget, not a lifetime cap):
    real tokens by kind priority (goal/option/budget > action/belief > events > observations > entities > memory),
    then recency (token time closest to now). Order of the kept tokens is preserved."""
    if ts.N <= budget:
        return ts
    pri = torch.zeros_like(ts.time)
    for k, p in KIND_PRIORITY.items():
        pri = torch.where(ts.kind == k, torch.full_like(pri, float(p)), pri)
    score = pri * 1e4 + ts.time.clamp(-999.0, 0.0)
    score = torch.where(ts.mask, score, torch.full_like(score, -float("inf")))
    idx = score.topk(budget, dim=1).indices.sort(dim=1).values           # [B, budget]
    g = lambda t: t.gather(1, idx)
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


def fast_cat(states: Sequence[FastState]) -> FastState:
    names = ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n", "surprise", "steps")
    return FastState(*(torch.cat([getattr(s, n) for s in states], 0) for n in names),
                     [dict(c) for s in states for c in s.channels])


def fast_scatter(full: FastState, rows: Sequence[int], sub: FastState) -> None:
    """Write sub-batch rows back into `full` (in place; tensors are detached state between decisions)."""
    idx = torch.tensor(list(rows), dtype=torch.long, device=full.logw.device)
    for n in ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n", "surprise", "steps"):
        dst = getattr(full, n)
        src = getattr(sub, n).detach().to(dst.dtype)
        if src.shape[1:] != dst.shape[1:]:
            raise ShapeError(f"FastState.{n}: sub-batch {tuple(src.shape)} vs state {tuple(dst.shape)}")
        dst[idx] = src
    for j, r in enumerate(rows):
        full.channels[r] = dict(sub.channels[j])


def fast_detach(f: FastState) -> FastState:
    return FastState(*(t.detach() for t in (f.particles, f.logw, f.pscale, f.bayes_mu, f.bayes_P, f.bayes_noise,
                                            f.bayes_n, f.surprise, f.steps)), [dict(c) for c in f.channels])


# ============================================================================ protocols not (yet) in interfaces.py

class ValueHeadP(Protocol):                   # workstream D: cam_unified/learn/heads.py
    def forward(self, R, goal: TokenSet) -> Any: ...                     # -> V [B]


class OptionPolicyP(Protocol):                # workstream D: cam_unified/learn/heads.py
    def forward(self, R, goal: TokenSet) -> Any: ...                     # -> option logits [B, n_options]
    def termination(self, R, option) -> Any: ...                         # option [B] long -> beta [B] in (0,1)
    def embed(self, option) -> Any: ...                                  # option [B] long -> [B, D]


class ExplorationHeadP(Protocol):             # workstream D: cam_unified/learn/heads.py / explore.py
    def forward(self, R, goal: TokenSet, belief: TokenSet) -> TokenSet: ...   # -> BUDGET token [B,1,D]


class PlannerP(Protocol):                     # workstream C: cam_unified/world/planner.py
    def plan(self, model: "CamUnifiedModel", req: "PlanRequest") -> "PlanResult": ...


# ============================================================================ factory registry

PKG = "superlearner.cam_unified"
DEFAULT_COMPONENT_PATHS = {
    "semantic": f"{PKG}.semantic.encoder:QwenSemantic",                 # A
    "tokenizer": f"{PKG}.semantic.tokenizer:ObservationTokenizer",      # A
    "action_generator": f"{PKG}.semantic.actions:ActionGenerator",      # A
    "core": f"{PKG}.core.backbone:UnifiedCore",                         # B
    "entity_encoder": f"{PKG}.core.entities:EntityEncoder",             # B
    "reasoner": f"{PKG}.core.reasoning:LatentReasoner",                 # B
    "beliefs": f"{PKG}.core.beliefs:BeliefModule",                      # B
    "bayes": f"{PKG}.core.bayes:BayesLastLayer",                        # B
    "outcome_head": f"{PKG}.world.outcomes:OutcomeHead",                # C
    "world_model": f"{PKG}.world.dynamics:WorldModel",                  # C
    "planner": f"{PKG}.world.planner:Planner",                          # C (optional; built-in imagination otherwise)
    "retriever": f"{PKG}.memory.retrieval:Retriever",                   # E
    "value_head": f"{PKG}.learn.heads:ValueHead",                       # D
    "option_policy": f"{PKG}.learn.heads:OptionPolicy",                 # D
    "exploration_head": f"{PKG}.learn.heads:ExplorationHead",           # D
}
# build order = dependency order; constructor parameters named like an earlier component receive it
BUILD_ORDER = ("semantic", "tokenizer", "core", "entity_encoder", "reasoner", "outcome_head", "beliefs", "bayes",
               "world_model", "retriever", "action_generator", "value_head", "option_policy", "exploration_head",
               "planner")
ALIASES = {"semantic_encoder": "semantic", "encoder": "semantic", "sem": "semantic", "backbone": "core",
           "outcomes": "outcome_head", "dynamics": "world_model", "obs_tokenizer": "tokenizer"}
# optional components loaded from their default path when it resolves (silently absent otherwise, see build_report);
# the planner is never auto-loaded: its `plan(model, PlanRequest)` interface must be confirmed when C merges
AUTO_OPTIONAL = ("semantic", "entity_encoder", "option_policy", "exploration_head")


def required_components(cfg: CamConfig) -> set:
    req = {"tokenizer", "core", "outcome_head", "world_model", "action_generator", "value_head"}
    if cfg.use_reasoning:
        req.add("reasoner")
    if cfg.use_fast_adaptation:
        req |= {"beliefs", "bayes"}
    if cfg.use_memory:
        req.add("retriever")
    return req


def resolve_path(path: str):
    """'package.module:Attr' -> the attribute (class or factory)."""
    if ":" not in path:
        raise ValueError(f"dotted path must look like 'pkg.mod:Attr', got {path!r}")
    mod, attr = path.split(":", 1)
    try:
        m = importlib.import_module(mod)
    except ImportError as e:
        raise ComponentUnavailable(f"cannot import {mod!r} for {path!r}: {e}") from e
    try:
        return getattr(m, attr)
    except AttributeError as e:
        raise ComponentUnavailable(f"{mod!r} has no attribute {attr!r}") from e


def _instantiate(name: str, spec, cfg: CamConfig, built: dict, kwargs: dict):
    """spec: an instance (used as is), a class / factory (called with constructor injection), or a dotted path."""
    if isinstance(spec, str):
        spec = resolve_path(spec)
    if not (inspect.isclass(spec) or inspect.isfunction(spec) or inspect.ismethod(spec)):
        return spec                                   # an already-built instance (nn.Module or plain object)
    sig = inspect.signature(spec)
    args = {}
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
        except ComponentUnavailable as e:
            out[name] = str(e)
    return out


def build_model(cfg: CamConfig, components: Optional[dict] = None, paths: Optional[dict] = None,
                component_kwargs: Optional[dict] = None, **model_kwargs) -> "CamUnifiedModel":
    """Assemble CamUnifiedModel. `components` maps a name to an instance, class/factory or dotted path (overrides
    `paths`, which overrides DEFAULT_COMPONENT_PATHS). A component set to None is left out (allowed only when it is
    optional for `cfg`). Raises ComponentUnavailable listing every required component that cannot be built."""
    components = dict(components or {})
    paths = {**DEFAULT_COMPONENT_PATHS, **(paths or {})}
    component_kwargs = component_kwargs or {}
    need = required_components(cfg)
    built: dict = {}
    report: dict = {}
    missing = []
    for name in BUILD_ORDER:
        if name in components:
            spec = components[name]
            if spec is None:
                built[name] = None
                report[name] = "absent: set to None"
                continue
        elif name in need or name in AUTO_OPTIONAL:
            spec = paths.get(name)
        else:
            built[name] = None
            report[name] = "absent: not required by the config" if name != "planner" else \
                "built-in LatentImaginationPlanner (pass components={'planner': ...} to override)"
            continue
        try:
            built[name] = _instantiate(name, spec, cfg, built, component_kwargs.get(name, {}))
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
    fast_prev: the belief before the last update (hindsight learning progress); option / option_steps [B] long."""
    Z: torch.Tensor
    fast: Optional[FastState]
    registries: list
    trackers: list
    option: torch.Tensor
    option_steps: torch.Tensor
    last_actions: list
    fast_prev: Optional[FastState] = None

    @property
    def B(self) -> int:
        return self.Z.shape[0]

    def select(self, rows: Sequence[int]) -> "CamState":
        idx = torch.tensor(list(rows), dtype=torch.long, device=self.Z.device)
        return CamState(self.Z[idx], None if self.fast is None else self.fast.select(idx),
                        [self.registries[r] for r in rows], [self.trackers[r] for r in rows],
                        self.option[idx].clone(), self.option_steps[idx].clone(),
                        [self.last_actions[r] for r in rows],
                        None if self.fast_prev is None else self.fast_prev.select(idx))

    def scatter(self, rows: Sequence[int], sub: "CamState") -> None:
        idx = torch.tensor(list(rows), dtype=torch.long, device=self.Z.device)
        self.Z[idx] = sub.Z.detach().to(self.Z.dtype)
        self.option[idx] = sub.option
        self.option_steps[idx] = sub.option_steps
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
    """Everything needed to recompute a decision's forward pass with gradients (PPO on fresh data): the stored
    detached Z_{t-1} (truncated BPTT of one decision), a FastState snapshot, an entity-registry snapshot taken
    before tokenisation (identical slot binding), raw observations / tools / goals / previous actions, the retrieved
    memory tokens (selection is not differentiable) and the option decisions taken."""
    Z_prev: torch.Tensor
    fast: Optional[FastState]
    registries: list
    obs_lists: list
    now: list
    last_actions: list
    goals: list
    tools: list
    memory: Optional[TokenSet]
    option_prev: torch.Tensor
    option: torch.Tensor
    option_switched: torch.Tensor
    option_evaluated: torch.Tensor
    option_terminated: torch.Tensor

    @property
    def B(self) -> int:
        return self.Z_prev.shape[0]

    @staticmethod
    def cat(items: Sequence["StepInputs"]) -> "StepInputs":
        mem = None
        if any(i.memory is not None for i in items):
            D = items[0].Z_prev.shape[2]
            dev = items[0].Z_prev.device
            mem = cat_rows([i.memory if i.memory is not None else TokenSet.empty(i.B, D, dev) for i in items])
        fasts = [i.fast for i in items]
        return StepInputs(torch.cat([i.Z_prev for i in items], 0), None if fasts[0] is None else fast_cat(fasts),
                          [r for i in items for r in i.registries], [o for i in items for o in i.obs_lists],
                          [n for i in items for n in i.now], [a for i in items for a in i.last_actions],
                          [g for i in items for g in i.goals], [t for i in items for t in i.tools], mem,
                          *(torch.cat([getattr(i, n) for i in items], 0) for n in
                            ("option_prev", "option", "option_switched", "option_evaluated", "option_terminated")))


@dataclass
class StepEncoding:
    obs: TokenSet           # O_t (+ entity/relation tokens) [B,No,D]
    goal: TokenSet          # G_t [B,Ng,D]
    belief: TokenSet        # B_t [B,K+1,D] (empty without fast adaptation)
    memory: Optional[TokenSet]
    option_tok: Optional[TokenSet]
    ctx: TokenSet           # budget-fitted context for F_theta
    Z: torch.Tensor         # Z_t [B,S,D]
    R: torch.Tensor         # R_t [B,S,D]
    rinfo: dict


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
    logp: torch.Tensor              # [B] total behaviour log-probability (action + option parts)
    dt: torch.Tensor                # [B] expected days until the next decision
    reasoning: list                 # [B] {"iters", "halt_p"}
    retrieved: list                 # [B] [{"id", "score"}]
    candidates: list
    imagined: list
    belief_before: list
    trajectory: Optional[ImaginedTrajectory] = None
    timings: dict = field(default_factory=dict)


# ============================================================================ the model

class CamUnifiedModel(nn.Module):
    """The single trainable CAM Unified network: injected components + outcome query encoder.

    Required: tokenizer, core, outcome_head, world_model, action_generator, value_head; reasoner with use_reasoning;
    beliefs + bayes with use_fast_adaptation; retriever with use_memory. Optional: semantic (counted separately as
    pretrained), entity_encoder, option_policy, exploration_head, planner (else `LatentImaginationPlanner`).
    Extra knobs (constructor kwargs): imag_candidates / imag_horizon / imag_samples (default from cfg), unc_coef
    (uncertainty penalty), n_error_classes, money_unit, max_numeric_outcomes, max_event_types, amp (bf16 autocast;
    default on for CUDA), reason_iters (fixed J; None = adaptive)."""

    def __init__(self, cfg: CamConfig, *, tokenizer, core, outcome_head, world_model, action_generator, value_head,
                 semantic=None, entity_encoder=None, reasoner=None, beliefs=None, bayes=None, retriever=None,
                 option_policy=None, exploration_head=None, planner=None, query_encoder=None,
                 imag_candidates: Optional[int] = None, imag_horizon: Optional[int] = None,
                 imag_samples: Optional[int] = None, unc_coef: float = 0.1, n_error_classes: int = 16,
                 money_unit: str = "cents", max_numeric_outcomes: int = 32, max_event_types: int = 16,
                 amp: Optional[bool] = None, reason_iters: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.version = 0
        self.build_report: dict = {}
        need = required_components(cfg)
        given = dict(tokenizer=tokenizer, core=core, outcome_head=outcome_head, world_model=world_model,
                     action_generator=action_generator, value_head=value_head, reasoner=reasoner, beliefs=beliefs,
                     bayes=bayes, retriever=retriever)
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
        self.option_policy = option_policy
        self.exploration_head = exploration_head
        self.planner = planner if planner is not None else LatentImaginationPlanner()
        self.query_encoder = query_encoder if query_encoder is not None else OutcomeQueryEncoder(
            cfg.d_model, make_text_encoder(tokenizer, cfg.d_model, lambda: self.version))
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
        self.think_days = cfg.think_s / 86400.0
        self.calls: dict = {}         # component call counters (observability; ablation tests)

    # ---------------------------------------------------------------- bookkeeping
    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _autocast(self):
        dev = self.device
        on = self.amp if self.amp is not None else dev.type == "cuda"
        return torch.autocast(dev.type, dtype=torch.bfloat16) if on else nullcontext()

    def _semantic_modules(self) -> list:
        mods = []
        if isinstance(self.semantic, nn.Module):
            mods.append(self.semantic)
        for m in self.modules():
            if m is not self and all(hasattr(m, a) for a in ("hidden", "embed", "sem_dim")) and m not in mods:
                mods.append(m)
        return mods

    def count_parameters(self) -> dict:
        """Unique parameters per component (shared modules, e.g. the backbone inside the world model, are counted
        once, at the first component in BUILD_ORDER that holds them). The pretrained semantic model (any submodule
        implementing SemanticEncoder) is reported separately and excluded from the CAM totals."""
        seen = set()
        sem_total = sem_train = 0
        for m in self._semantic_modules():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    sem_total += p.numel()
                    sem_train += p.numel() if p.requires_grad else 0
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
                "semantic_pretrained": {"total": sem_total, "trainable": sem_train}}

    # ---------------------------------------------------------------- state
    def init_state(self, B: int, device=None, seed: int = 0) -> CamState:
        dev = device or self.device
        Z = self.core.init_state(B, device=dev)
        check_shape(Z, (B, self.cfg.n_state, self.cfg.d_model), "core.init_state")
        fast = None
        if self.cfg.use_fast_adaptation:
            fast = self.beliefs.init_fast(B, device=dev)
            check_shape(fast.logw, (B, self.cfg.n_particles), "beliefs.init_fast logw")
        regs = [EntityRegistry(self.cfg.max_entities, seed=seed * 100003 + b) for b in range(B)]
        trackers = [self.new_tracker() for _ in range(B)]
        return CamState(Z.detach(), fast, regs, trackers, torch.full((B,), -1, dtype=torch.long, device=dev),
                        torch.zeros(B, dtype=torch.long, device=dev), [None] * B)

    def new_tracker(self) -> OutcomeTracker:
        return OutcomeTracker(self.money_unit, self.n_error_classes, self.max_numeric_outcomes, self.max_event_types)

    def empty_tokens(self, B: int) -> TokenSet:
        return TokenSet.empty(B, self.cfg.d_model, device=self.device)

    # ---------------------------------------------------------------- the step equation
    def _goal_and_belief(self, goals: list, fast: Optional[FastState], B: int) -> tuple:
        G = check_tokens(self.tokenizer.goal(goals), B, self.cfg.d_model, "tokenizer.goal")
        if self.cfg.use_fast_adaptation and fast is not None:
            self._count("beliefs.tokens")
            Bt = check_tokens(self.beliefs.tokens(fast), B, self.cfg.d_model, "beliefs.tokens")
        else:
            Bt = self.empty_tokens(B)
        return G, Bt

    def retrieve(self, Z_prev: torch.Tensor, G: TokenSet, Bt: TokenSet, filters: Optional[list] = None) -> tuple:
        """M_t = Retrieve(Q(Z_{t-1}, G_t, B_t)) -> (TokenSet MEMORY [B,<=k,D], ids [B][k], scores)."""
        B = Z_prev.shape[0]
        self._count("retriever.retrieve")
        q = self.retriever.query(Z_prev, G, Bt)
        if q.dim() != 2 or q.shape[0] != B:
            raise ShapeError(f"retriever.query: {tuple(q.shape)} != [B={B}, key_dim]")
        M, ids, scores = self.retriever.retrieve(q, self.cfg.mem_top_k, filters)
        check_tokens(M, B, self.cfg.d_model, "retriever.retrieve")
        return as_kind(M, TokenKind.MEMORY), ids, scores

    def encode_step(self, inp: StepInputs, pre: Optional[tuple] = None) -> StepEncoding:
        """Z_t = F(Z_{t-1}, [O_t; A_{t-1}; B_t; M_t; G_t; option_{t-1}]) and R_t = Reason(Z_t, [B_t; M_t; G_t; option]).
        `inp.registries` are bound in place by the tokenizer (pass copies to recompute)."""
        cfg, B, D = self.cfg, inp.B, self.cfg.d_model
        O = check_tokens(self.tokenizer.observations(inp.obs_lists, inp.registries, inp.now), B, D,
                         "tokenizer.observations")
        if self.entity_encoder is not None:
            E = check_tokens(self.entity_encoder(O), B, D, "entity_encoder")
            O = TokenSet.cat(O, E)
        A_prev = check_tokens(self.tokenizer.action(inp.last_actions), B, D, "tokenizer.action")
        G, Bt = pre if pre is not None else self._goal_and_belief(inp.goals, inp.fast, B)
        opt_tok = None
        if self.option_policy is not None:
            emb = self.option_policy.embed(inp.option_prev.clamp(min=0))
            check_shape(emb, (B, D), "option_policy.embed")
            opt_tok = masked_token(emb, TokenKind.OPTION, inp.option_prev >= 0)
        M = inp.memory if cfg.use_memory else None
        ctx = TokenSet.cat(O, A_prev, Bt, M, G, opt_tok)
        ctx = fit_budget(ctx, cfg.max_ctx_tokens)
        self._count("core")
        Z, ctx_out = self.core(inp.Z_prev, ctx)
        check_shape(Z, (B, cfg.n_state, D), "core.forward Z")
        if cfg.use_reasoning:
            rctx = TokenSet.cat(Bt, M, G, opt_tok)
            self._count("reasoner")
            R, rinfo = self.reasoner(Z, rctx, self.reason_iters)
            check_shape(R, (B, cfg.n_state, D), "reasoner R")
        else:
            R, rinfo = Z, {"iters": torch.zeros(B, device=Z.device), "halt_p": None, "ponder": None}
        return StepEncoding(O, G, Bt, M, opt_tok, ctx, Z, R, rinfo)

    def _select_option(self, R, G, option_prev: torch.Tensor, option_steps: torch.Tensor, greedy: bool,
                       gen: Optional[torch.Generator]) -> tuple:
        """Option termination beta(R, o_{t-1}) and re-selection pi_option(R, G); an option is forced to end after
        `option_max_steps` decisions. Returns (option [B], logp [B], switched [B], evaluated [B], terminated [B],
        OPTION token of the active option)."""
        B = R.shape[0]
        dev = R.device
        need = (option_prev < 0) | (option_steps >= self.cfg.option_max_steps)
        beta = self.option_policy.termination(R, option_prev.clamp(min=0)).float().clamp(1e-6, 1 - 1e-6)
        check_shape(beta, (B,), "option_policy.termination")
        evaluated = ~need
        if greedy:
            term = beta > 0.5
        else:
            term = torch.rand(B, generator=gen, device="cpu").to(dev) < beta
        term = term & evaluated
        switched = need | term
        logits = self.option_policy(R, G).float()
        check_shape(logits, (B, self.cfg.n_options), "option_policy logits")
        lsm = torch.log_softmax(logits, -1)
        new = lsm.argmax(-1) if greedy else torch.multinomial(lsm.exp().cpu(), 1, generator=gen).view(B).to(dev)
        option = torch.where(switched, new, option_prev)
        logp = torch.where(switched, lsm.gather(1, option.clamp(min=0).view(B, 1)).view(B), torch.zeros(B, device=dev))
        logp = logp + torch.where(evaluated, torch.where(term, beta.log(), (1 - beta).log()), torch.zeros_like(beta))
        tok = masked_token(self.option_policy.embed(option), TokenKind.OPTION)
        return option, logp, switched, evaluated, term, tok

    def _option_logp(self, R, G, inp: StepInputs) -> torch.Tensor:
        """Recompute the stored option decisions' log-probability under current parameters (differentiable)."""
        B = R.shape[0]
        beta = self.option_policy.termination(R, inp.option_prev.clamp(min=0)).float().clamp(1e-6, 1 - 1e-6)
        lsm = torch.log_softmax(self.option_policy(R, G).float(), -1)
        lp = torch.where(inp.option_switched, lsm.gather(1, inp.option.clamp(min=0).view(B, 1)).view(B),
                         torch.zeros(B, device=R.device))
        return lp + torch.where(inp.option_evaluated, torch.where(inp.option_terminated, beta.log(), (1 - beta).log()),
                                torch.zeros_like(beta))

    def _goal_inputs(self, enc: StepEncoding, option_tok: Optional[TokenSet]) -> TokenSet:
        parts = [enc.goal, option_tok]
        if self.cfg.use_learned_exploration and self.exploration_head is not None:
            self._count("exploration_head")
            bud = self.exploration_head(enc.R, enc.goal, enc.belief)
            check_tokens(bud, enc.R.shape[0], self.cfg.d_model, "exploration_head")
            parts.append(as_kind(bud, TokenKind.BUDGET))
        return TokenSet.cat(*parts)

    def prospective_batch(self, state: CamState) -> tuple:
        specs = [tr.prospective() for tr in state.trackers]
        chans = [bind_channels(state.fast, b, specs[b], state.trackers[b].last_seen, self.cfg.bayes_prior_var_init,
                               self.cfg.bayes_obs_var_init) for b in range(state.B)]
        return specs, build_outcome_batch(specs, self.query_encoder, state.registries, chans, self.device)

    def base_moments(self, Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch) -> tuple:
        """Outcome-head predictive mean/var [B,Q] from the action-conditioned latent Z_hat: the mixture over belief
        particles (weights w) with fast adaptation, the unconditioned head otherwise. Also returns the features."""
        B, Q = ob.lik.shape
        feat = self.outcome_head.features(Z_hat, ob)
        check_shape(feat, (B, Q, self.cfg.d_model), "outcome_head.features")
        if self.cfg.use_fast_adaptation and fast is not None:
            d = self.outcome_head.dist(feat, ob, cond=fast.particles)
            mk, vk = dist_mean(d).float(), dist_var(d).float()
            check_shape(mk, (B, self.cfg.n_particles, Q), "outcome_head.dist(cond) mean")
            w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
            mean = (w * mk).sum(1)
            var = (w * (vk + mk * mk)).sum(1) - mean * mean
        else:
            d = self.outcome_head.dist(feat, ob)
            mean, var = dist_mean(d).float(), dist_var(d).float()
            check_shape(mean, (B, Q), "outcome_head.dist mean")
        return mean, var.clamp_min(0.0), feat

    def predict_outcomes(self, Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch) -> tuple:
        """Predictive mean/var [B,Q]: base_moments plus, on continuous outcomes with a channel, the Kalman/NIG
        last layer's residual prediction (mean phi^T mu_c, variance phi^T P_c phi + noise)."""
        mean, var, feat = self.base_moments(Z_hat, fast, ob)
        if self.cfg.use_fast_adaptation and fast is not None:
            B, Q = ob.lik.shape
            phi = self.bayes.features(feat, ob)
            bm, bv = self.bayes.predict(fast, phi, ob)
            check_shape(bm, (B, Q), "bayes.predict mean")
            cont = self._continuous(ob)
            mean = mean + torch.where(cont, bm.float(), torch.zeros_like(mean))
            var = var + torch.where(cont, bv.float().clamp_min(0.0), torch.zeros_like(var))
        return mean, var, feat

    @staticmethod
    def _continuous(ob: OutcomeBatch) -> torch.Tensor:
        c = torch.zeros_like(ob.y_mask)
        for k in CONTINUOUS:
            c = c | (ob.lik == k)
        return c & (ob.channel >= 0)

    def mixture_loglik(self, fast: FastState, Z_hat: torch.Tensor, ob: OutcomeBatch) -> torch.Tensor:
        """log p(y | B) per env [B]: log sum_k w_k prod_q p(y_q | Z_hat, m_k), divided by the number of observed
        outcomes (per-outcome nats). Differentiable in the particles, weights, outcome head and Z_hat."""
        B, Q = ob.lik.shape
        feat = self.outcome_head.features(Z_hat, ob)
        d = self.outcome_head.dist(feat, ob, cond=fast.particles)
        K = fast.logw.shape[1]
        y = ob.y.unsqueeze(1).expand(B, K, Q)
        cens = None if ob.censored is None else ob.censored.unsqueeze(1).expand(B, K, Q)
        lp = dist_log_prob(d, y, cens).float()
        m = ob.y_mask.unsqueeze(1).float()
        ll = (torch.where(m > 0, lp, torch.zeros_like(lp))).sum(-1)                   # [B,K]
        logw = torch.log_softmax(fast.logw.float(), -1)
        return torch.logsumexp(logw + ll, -1) / ob.y_mask.float().sum(-1).clamp_min(1.0)

    @torch.no_grad()
    def decide(self, state: CamState, obs_lists: list, now: list, tools: list, goals: list,
               greedy: bool = False, generator: Optional[torch.Generator] = None,
               retrieval_filters: Optional[list] = None) -> Decision:
        """One decision for every row of `state` (all rows must be live environments). Mutates state.Z, option,
        registries (entity binding) and the fast state's channel table; beliefs change only in observe_outcomes."""
        cfg = self.cfg
        B, D = state.B, cfg.d_model
        dev = state.Z.device
        if not (len(obs_lists) == len(now) == len(tools) == len(goals) == B):
            raise ShapeError("decide: obs/now/tools/goals must all have B entries")
        regs_snapshot = copy.deepcopy(state.registries)
        with self._autocast():
            G, Bt = self._goal_and_belief(goals, state.fast, B)
            M, ids, scores = (None, [[] for _ in range(B)], None)
            if cfg.use_memory:
                M, ids, scores = self.retrieve(state.Z, G, Bt, retrieval_filters)
            inp = StepInputs(state.Z.detach(), None if state.fast is None else state.fast.clone(), regs_snapshot,
                             obs_lists, list(now), list(state.last_actions), list(goals), list(tools),
                             detach_tokens(M), state.option.clone(), state.option.clone(),
                             torch.zeros(B, dtype=torch.bool, device=dev), torch.zeros(B, dtype=torch.bool, device=dev),
                             torch.zeros(B, dtype=torch.bool, device=dev))
            inp.registries = state.registries          # live registries bind new entities; snapshot kept below
            enc = self.encode_step(inp, pre=(G, Bt))
            inp.registries = regs_snapshot
            opt_logp = torch.zeros(B, device=dev)
            opt_tok = None
            if self.option_policy is not None:
                option, opt_logp, sw, ev, term, opt_tok = self._select_option(
                    enc.R, enc.goal, state.option, state.option_steps, greedy, generator)
                inp.option, inp.option_switched, inp.option_evaluated, inp.option_terminated = option, sw, ev, term
                state.option_steps = torch.where(sw, torch.ones_like(state.option_steps), state.option_steps + 1)
                state.option = option
            goal_in = self._goal_inputs(enc, opt_tok)
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
            for b, a in enumerate(actions):
                if a.logp is None:
                    raise ShapeError("every CAM action must carry its behaviour log-probability (Action.logp)")
                if self.option_policy is not None:
                    a.option_id = int(state.option[b])
                    a.logp_parts["option"] = float(opt_logp[b])
                    a.logp = float(a.logp) + float(opt_logp[b])
                logp[b] = float(a.logp)
            dt = torch.tensor([self.think_days + self._expected_duration(a, tl) for a, tl in zip(actions, tools)],
                              device=dev)
            A_tok = check_tokens(self.tokenizer.action(actions), B, D, "tokenizer.action(selected)")
            self._count("world_model.prior")
            Z_hat, _ = self.world_model.prior(enc.Z, A_tok, enc.belief, dt, sample=False)
            check_shape(Z_hat, (B, cfg.n_state, D), "world_model.prior Z_hat")
            mean, var, _ = self.predict_outcomes(Z_hat, state.fast, prosp)
            value = self.value_head(enc.R, enc.goal).float()
            check_shape(value, (B,), "value_head")
        state.Z = enc.Z.detach().float()
        state.last_actions = list(actions)
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
                        traj)

    def _expected_duration(self, a: Action, tools: list) -> float:
        if a.expected_duration is not None:
            return float(a.expected_duration)
        for t in tools:
            if t.name == a.tool and t.duration_hint is not None:
                return float(t.duration_hint)
        return 0.0

    # ---------------------------------------------------------------- outcomes -> beliefs
    @torch.no_grad()
    def observe_outcomes(self, state: CamState, decision: Decision, results: list, next_obs: list,
                         rewards: Sequence[float]) -> dict:
        """Realise the outcomes of `decision` (OutcomeBatch with y), compute the hindsight learning progress of the
        previous belief update, then update the particle beliefs and the Kalman/NIG last layer (use_fast_adaptation).
        Returns {"outcomes", "specs", "belief_info", "lp" [B] or None, "belief_after", "loglik" [B]}."""
        cfg = self.cfg
        B = state.B
        specs = [state.trackers[b].realize(decision.inputs.obs_lists[b], decision.actions[b], results[b], next_obs[b],
                                           float(rewards[b])) for b in range(B)]
        chans = [bind_channels(state.fast, b, specs[b], state.trackers[b].last_seen, cfg.bayes_prior_var_init,
                               cfg.bayes_obs_var_init) for b in range(B)]
        with self._autocast():
            ob = build_outcome_batch(specs, self.query_encoder, state.registries, chans, self.device)
            out = {"outcomes": ob, "specs": specs, "belief_info": {}, "lp": None, "loglik": None}
            if cfg.use_fast_adaptation and state.fast is not None:
                ll_now = self.mixture_loglik(state.fast, decision.Z_hat, ob)
                out["loglik"] = ll_now
                if cfg.use_learned_exploration and state.fast_prev is not None:
                    # LP_{t-1} = log p(o_t | B_t) - log p(o_t | B_{t-1}): o_t is disjoint from the support o_{t-1}
                    out["lp"] = (ll_now - self.mixture_loglik(state.fast_prev, decision.Z_hat, ob)).float()
                before = state.fast.clone()
                # Kalman target: residual of y against the same base prediction the residual is added to
                base_mean, _, feat = self.base_moments(decision.Z_hat, before, ob)
                self._count("beliefs.update")
                fast1, binfo = self.beliefs.update(state.fast, decision.Z, decision.act_tokens, ob, self.outcome_head)
                check_shape(fast1.logw, (B, cfg.n_particles), "beliefs.update logw")
                cont = self._continuous(ob) & ob.y_mask
                resid = with_targets(ob, torch.where(cont, ob.y - base_mean, torch.zeros_like(ob.y)), cont)
                self._count("bayes.update")
                phi = self.bayes.features(feat, resid)
                fast2 = self.bayes.update(fast1, phi, resid)
                fast2.channels = [dict(c) for c in state.fast.channels]
                state.fast_prev = before
                state.fast = fast_detach(fast2)
                out["belief_info"] = {k: v for k, v in (binfo or {}).items()}
        out["belief_after"] = belief_summary(None if state.fast is None else state.fast.logw) or [{} for _ in range(B)]
        return out

    # ---------------------------------------------------------------- training-time recomputation
    def policy_evaluate(self, inp: StepInputs, actions: list) -> dict:
        """Differentiable recomputation of stored decisions: total behaviour log-probability of the executed actions
        (generator + option parts), entropy, value V(R, G) and the reasoner's ponder cost. All [B]."""
        regs = copy.deepcopy(inp.registries)
        live = StepInputs(**{**inp.__dict__, "registries": regs})
        with self._autocast():
            enc = self.encode_step(live)
            opt_tok, opt_lp = None, torch.zeros(inp.B, device=enc.R.device)
            if self.option_policy is not None:
                opt_lp = self._option_logp(enc.R, enc.goal, inp)
                opt_tok = masked_token(self.option_policy.embed(inp.option), TokenKind.OPTION)
            goal_in = self._goal_inputs(enc, opt_tok)
            T = self.tokenizer.tools(inp.tools)
            logp, ent = self.action_generator.log_prob(enc.R, goal_in, enc.belief, T, enc.obs, inp.tools,
                                                       inp.obs_lists, actions)
            value = self.value_head(enc.R, enc.goal)
        B = inp.B
        check_shape(logp, (B,), "action_generator.log_prob")
        ponder = enc.rinfo.get("ponder")
        return {"logp": logp.float() + opt_lp.float(), "entropy": ent.float(), "value": value.float(),
                "ponder": None if ponder is None else ponder.float()}

    def filter_sequence(self, seq) -> list:
        """Run F_theta over a tokenised SequenceBatch: Z_t = F(Z_{t-1}, [obs_t; act_{t-1}; goal_t]) -> [T] of [B,S,D].
        Rows whose previous step ended an episode restart from the initial state."""
        steps = seq.steps
        B = steps[0].obs.B
        dev = steps[0].obs.x.device
        Z0 = self.core.init_state(B, device=dev)
        Z = Z0
        prev_act, prev_done = None, torch.zeros(B, dtype=torch.bool, device=dev)
        Zs = []
        with self._autocast():
            for st in steps:
                Z = torch.where(prev_done.view(B, 1, 1), Z0, Z)
                ctx = fit_budget(TokenSet.cat(st.obs, prev_act, st.goal), self.cfg.max_ctx_tokens)
                Z, _ = self.core(Z, ctx)
                Zs.append(Z)
                prev_act, prev_done = st.act, st.done.to(torch.bool)
        return Zs

    def adapt_loss(self, seq, Zs: list, k: int, channel_tables: Optional[list] = None) -> tuple:
        """L_adapt = -log p(Q | B_S, query actions) (spec 12.2): beliefs inferred from the first k steps (support)
        by the belief module and the Kalman layer, then the later steps' observed outcomes (queries, disjoint from
        the support) scored under the support belief without further updates. Returns (loss, metrics)."""
        steps = seq.steps
        T = len(steps)
        B = steps[0].obs.B
        dev = Zs[0].device
        fast = self.beliefs.init_fast(B, device=dev)
        if channel_tables is not None:
            fast.channels = [dict(c) for c in channel_tables]
        with self._autocast():
            for t in range(k):
                st = steps[t]
                fast, _ = self.beliefs.update(fast, Zs[t], st.act, st.outcomes, self.outcome_head)
            Bt = self.beliefs.tokens(fast)
            nll, n = Zs[0].new_zeros(()).float(), 0
            for t in range(k, T):
                st = steps[t]
                valid = st.outcomes.y_mask.any(-1) & ~self._padded(st)
                if not bool(valid.any()):
                    continue
                Z_hat, _ = self.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
                ll = self.mixture_loglik(fast, Z_hat, st.outcomes)
                nll = nll - (ll * valid.float()).sum() / valid.float().sum()
                n += 1
        loss = nll / max(1, n)
        return loss, {"adapt_k": k, "adapt_queries": n, "adapt_nll": float(loss.detach())}

    @staticmethod
    def _padded(st) -> torch.Tensor:
        return ~st.obs.mask.any(-1) & ~st.act.mask.any(-1)


# ============================================================================ built-in planner (protocol-only)

class LatentImaginationPlanner:
    """Belief-aware multistep imagination over CAM-generated candidates (spec 7.1-7.2), composed only from the
    contract protocols: n candidate first actions from the ActionGenerator; for each, `n_samples` stochastic futures
    of `horizon` decisions: Z_hat = WorldModel.prior(Z, EncAction(a), B, dt) (stochastic latent), outcomes sampled from
    the particle-conditioned outcome head, an *independent clone* of the FastState updated with the imagined outcomes
    (BeliefModule.update), the reasoner re-run on the imagined latent, and the next action sampled from the policy
    conditioned on the revised belief. Return = sum_h gamma_h * reward_hat_h + gamma_H * V(R_H) (gamma = exp(-kappa dt))
    minus unc_coef * (particle disagreement on reward); the real action is the best candidate. Nothing here touches
    the environment, the archive or the real FastState. Workstream C's planner replaces it via the registry."""

    def plan(self, model: CamUnifiedModel, req: PlanRequest) -> PlanResult:
        cfg = model.cfg
        B, n, H, ns = req.R.shape[0], req.n_candidates, req.horizon, req.n_samples
        dev = req.R.device
        rep = torch.arange(B, device=dev).repeat_interleave(n)                           # candidate rows [B*n]
        ag = model.action_generator
        model._count("action_generator.sample")
        cands = ag.sample(req.R[rep], req.goal_in.select(rep), req.belief.select(rep), req.tools.select(rep),
                          req.obs.select(rep), [req.tool_lists[int(i)] for i in rep],
                          [req.obs_lists[int(i)] for i in rep], temperature=1.0, greedy=False)
        rows = rep.repeat_interleave(ns)                                                  # rollout rows [B*n*ns]
        cand_of_row = torch.arange(B * n, device=dev).repeat_interleave(ns)
        Bb = rows.shape[0]
        Z = req.Z[rows]
        fast = req.fast.select(rows) if (cfg.use_fast_adaptation and req.fast is not None) else None
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
            dt = torch.tensor([model.think_days + model._expected_duration(a, tl) for a, tl in zip(acts, tool_lists)],
                              device=dev)
            act_tok = model.tokenizer.action(acts)
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            model._count("world_model.prior")
            Z_hat, _ = model.world_model.prior(Z, act_tok, belief, dt, sample=True)
            feat = model.outcome_head.features(Z_hat, prosp)
            if fast is not None:
                d = model.outcome_head.dist(feat, prosp, cond=fast.particles)
                mk = dist_mean(d).float()                                               # [Bb,K,Q]
                w = torch.softmax(fast.logw.float(), -1)
                k = torch.multinomial(w.cpu(), 1, generator=req.generator).view(Bb).to(dev)
                ys = dist_sample(d, req.generator).float()[torch.arange(Bb, device=dev), k]   # [Bb,Q]
                mean = (w.unsqueeze(-1) * mk).sum(1)
                var = (w.unsqueeze(-1) * (dist_var(d).float() + mk * mk)).sum(1) - mean * mean
                disagree = ((w * (mk[:, :, r_idx] - mean[:, r_idx:r_idx + 1]) ** 2).sum(1)).sqrt()
            else:
                d = model.outcome_head.dist(feat, prosp)
                mean, var = dist_mean(d).float(), dist_var(d).float()
                ys = dist_sample(d, req.generator).float()
                disagree = var[:, r_idx].clamp_min(0).sqrt()
            ret = ret + disc * mean[:, r_idx]
            pen = pen + disc * disagree
            disc = disc * torch.exp(-cfg.kappa * dt)
            if h == 0:
                first_out = mean
            imagined = with_targets(prosp, ys, prosp.q.mask.clone())
            if fast is not None:
                model._count("imagination.beliefs.update")
                fast, _ = model.beliefs.update(fast, Z, act_tok, imagined, model.outcome_head)
            Z = Z_hat
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            if cfg.use_reasoning:
                R, _ = model.reasoner(Z, TokenSet.cat(belief, mem, goal), model.reason_iters)
            else:
                R = Z
            v = model.value_head(R, goal).float()
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
        probs = torch.softmax(scores, -1)
        actions, cand_out, imag_out = [], [], []
        for b in range(B):
            c = int(choice[b])
            a = copy.deepcopy(cands[b * n + c])
            a.source = ActionSource.CAM_PLANNER
            a.proposal = {"n": n, "probs": [float(p) for p in probs[b]], "chosen": c,
                          "scores": [float(s) for s in scores[b]]}
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
                              "outcomes": {k: float(first_out[sl, q].mean()) for q, k in enumerate(keys[:8])},
                              "next_tools": [steps[1].action[j].tool for j in range(sl.start, sl.stop)]
                              if H > 1 else []})
            imag_out.append(items)
        traj = ImaginedTrajectory(steps, ret.detach(), rows.detach(), cand_of_row.remainder(n).detach())
        return PlanResult(actions, cand_out, imag_out, scores.detach(), traj)
