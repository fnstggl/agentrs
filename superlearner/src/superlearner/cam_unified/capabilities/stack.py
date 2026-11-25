"""The real assembled CAM Unified stack for capability diagnostics (workstream Y).

`build_stack` assembles `CamUnifiedModel` through `model.build_model` from the **real** workstream classes only:
QwenSemantic (Qwen3-0.6B, `cfg.sem_test_model`, frozen encoder path), A's ObservationTokenizer and ActionGenerator,
B's UnifiedCore / EntityEncoder / Reasoner / BeliefModule / BayesLastLayer, C's OutcomeHead / WorldModel, E's
Retriever (with a MemoryIndex when memory is on), D's ValueHead / OptionModule / BudgetEncoder, and wraps it in
`CamUnifiedAgent` with D's CombinedTrainer and, when asked, E's EventArchive / ReplaySampler / MemoryIndex /
Consolidator and F's TraceRecorder. No test double is imported here (production code never imports
`cam_unified.testing` or `integration_doubles`).

The semantic model is loaded once per (model name, device) and process (`load_semantic`): the pretrained weights are
not part of any CAM state, so several stacks in one process share it (each stack has its own trainable tokenizer
pooling/projections, action generator and LoRA tensors are frozen unless the caller trains them).

`capability_config` is the small configuration of the diagnostics (same capabilities as dev, tiny widths), the
analogue of `testing.tiny_config` without importing it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..config import CamConfig, make_config

_SEMANTIC: dict = {}


def capability_config(**over) -> CamConfig:
    """Tiny configuration with every capability switched on (override any field)."""
    base = dict(n_layers=2, d_model=64, n_heads=4, n_state=8, d_particle=32, n_particles=8, bayes_feat=8,
                stoch_groups=8, stoch_classes=8, mem_key_dim=32, reason_layers=1, max_ctx_tokens=96, n_prefix=4,
                max_entities=32, n_options=0, imag_branches=4, imag_samples=2, imag_horizons=(2,), mem_top_k=4,
                sem_tokens_per_obs=2, mem_segment=8, mem_segment_tokens=2)
    base.update(over)
    return make_config("dev", **base)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_semantic(cfg: CamConfig, device=None, model_name: Optional[str] = None):
    """The process-wide QwenSemantic for (model, device); Qwen3-0.6B (`cfg.sem_test_model`) by default."""
    from ..semantic.encoder import QwenSemantic
    dev = str(device or default_device())
    name = model_name or cfg.sem_test_model
    key = (name, dev)
    if key not in _SEMANTIC:
        _SEMANTIC[key] = QwenSemantic(cfg, model_name=name, device=dev)
    return _SEMANTIC[key]


class SyncedReplay:
    """E's ReplaySampler as the agent's ReplayP: the agent notifies newly archived records through `add` (E's
    sampler has no `add`; its membership tables are brought up to date by `sync()`), so the sampler is synced
    lazily before the next draw. Everything else is delegated."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.dirty = False
        self.n_synced = 0

    def add(self, records) -> None:
        self.dirty = True

    def sample(self, n: int, length: int) -> list:
        if self.dirty:
            self.n_synced += self.sampler.sync()
            self.dirty = False
        return self.sampler.sample(n, length)

    def __getattr__(self, name):
        return getattr(self.sampler, name)


class ServingConsolidator:
    """E's Consolidator as the agent calls it (`step()` in `CamUnifiedAgent.update`), plus making the encoded version
    searchable. `Consolidator.step()` writes segment keys under the retriever's key_version, but MemoryIndex serves
    only its `current_version`, which is set by `Consolidator.ensure_current()` / `reencode_step()` - and the agent
    calls neither, so in the assembled agent every retrieval returned no memory (measured: 0 of all answer decisions
    retrieved anything with 52 indexed segments; interface change request filed). This adapter calls
    `ensure_current()` after each step and `reencode_step()` while a migration is pending; everything else is
    delegated."""

    def __init__(self, consolidator):
        self.consolidator = consolidator
        self.n_reencode_calls = 0

    def step(self, *a, **k) -> dict:
        out = self.consolidator.step(*a, **k)
        c = self.consolidator
        c.ensure_current()
        cur = c.index.current_version
        if cur is not None and cur != c.retriever.key_version:
            out["reencode"] = c.reencode_step()
            self.n_reencode_calls += 1
        out["served_version"] = c.index.current_version
        return out

    def __getattr__(self, name):
        return getattr(self.consolidator, name)


class TargetSyncTrainer:
    """D's CombinedTrainer plus the post-step EMA update of the world model's latent-target encoder.

    `WorldModel.update_target()` must follow every optimiser step (world/dynamics.py: "call after each optimiser
    step"), but `CamUnifiedAgent.update()` never calls it, so in the assembled agent the EMA target backbone stays
    at its initial random weights forever (interface change request filed). This adapter performs the documented
    update after each step; everything else is delegated."""

    def __init__(self, trainer, model, decay: Optional[float] = None):
        self.trainer = trainer
        self.model = model
        self.decay = decay
        self.n_target_updates = 0

    def step(self, losses: dict, *a, **k):
        out = self.trainer.step(losses, *a, **k)
        wm = getattr(self.model, "world_model", None)
        if wm is not None and hasattr(wm, "update_target"):
            wm.update_target(self.decay)
            self.n_target_updates += 1
        return out

    def __getattr__(self, name):
        return getattr(self.trainer, name)


@dataclass
class Stack:
    """model + agent + trainer (+ memory collaborators) of one assembled CAM."""
    cfg: CamConfig
    model: object
    agent: object
    trainer: object
    semantic: object
    archive: object = None
    replay: object = None
    index: object = None
    consolidator: object = None
    recorder: object = None
    root: Optional[str] = None
    info: dict = field(default_factory=dict)

    def real_component_report(self) -> dict:
        """{component: class path} of every built component (proof that no double is in the stack)."""
        out = {}
        for name in ("semantic", "tokenizer", "core", "entity_encoder", "reasoner", "beliefs", "bayes", "outcome_head",
                     "world_model", "retriever", "action_generator", "value_head", "option_policy",
                     "exploration_head", "planner"):
            obj = getattr(self.model, name, None)
            out[name] = None if obj is None else f"{type(obj).__module__}.{type(obj).__qualname__}"
        return out


def make_trainer(model, cfg: CamConfig, stage="continual", train_semantic: bool = False, **kw):
    """D's CombinedTrainer over the whole model: core group = the shared backbone, heads = every other trainable
    CAM parameter; the semantic LoRA tensors (registered in the action generator) form the semantic group only when
    `train_semantic`, and are frozen otherwise (the diagnostics train CAM's own parameters)."""
    from ..learn.trainer import CombinedTrainer
    core = list(model.core.parameters())
    core_ids = {id(p) for p in core}
    lora_ids = {id(p) for p in getattr(model.action_generator, "lora", [])}
    heads, sem = [], []
    seen = set()
    for p in model.parameters():
        if id(p) in seen or id(p) in core_ids or not p.requires_grad:
            continue
        seen.add(id(p))
        (sem if id(p) in lora_ids else heads).append(p)
    if not train_semantic:
        for p in sem:
            p.requires_grad_(False)
        sem = []
    kw.setdefault("grad_probe_every", 0)
    kw.setdefault("amp", model.device.type == "cuda")
    return CombinedTrainer(cfg, core=core, heads=heads, semantic=sem or None, stage=stage, **kw)


def build_stack(cfg: Optional[CamConfig] = None, *, device=None, seed: int = 0, include_builtins: bool = False,
                grammar_opts=None, memory_root: Optional[str] = None, trace_root: Optional[str] = None,
                trainer_stage="continual", train_semantic: bool = False, model_kwargs: Optional[dict] = None,
                agent_kwargs: Optional[dict] = None, trainer_kwargs: Optional[dict] = None,
                components: Optional[dict] = None, reasoner_kwargs: Optional[dict] = None,
                exact_rows: bool = False, target_decay: Optional[float] = None) -> Stack:
    """Assemble the real CAM stack. `memory_root` (a directory) attaches E's archive + replay + index +
    consolidator (needed for replay-trained objectives and for cfg.use_memory); without it the agent trains on its
    fresh on-policy buffer only. `components` may override individual real components (e.g. a differently
    configured real Reasoner); they are still real classes. `exact_rows=False` batches the action generator's learner
    path (its documented fast mode): the diagnostics' tools have no string arguments, so the batched tool pointer
    equals the row-wise one up to float rounding (PPO ratios at lag 0 are reported as policy/logp_maxdev_lag0)."""
    from ..agent import CamUnifiedAgent
    from ..model import build_model
    from ..semantic.actions import ActionGenerator
    from ..semantic.grammar import GrammarOptions
    from ..semantic.tokenizer import ObservationTokenizer

    cfg = cfg or capability_config()
    dev = torch.device(device) if device is not None else default_device()
    enc = load_semantic(cfg, dev)
    torch.manual_seed(seed)
    tok = ObservationTokenizer(cfg, enc).to(dev)
    opts = grammar_opts or GrammarOptions(max_string_tokens=24, max_action_tokens=96, max_key_tokens=16)
    gen = ActionGenerator(cfg, enc, tok, opts=opts, include_builtins=include_builtins, exact_rows=exact_rows)
    comps = {"semantic": enc, "tokenizer": tok, "action_generator": gen}
    ckw = {}
    if reasoner_kwargs:
        ckw["reasoner"] = dict(reasoner_kwargs)
    index = archive = replay = consolidator = recorder = None
    if cfg.use_memory and memory_root is None:
        raise ValueError("cfg.use_memory needs memory_root (E's MemoryIndex / EventArchive live on disk)")
    comps.update(components or {})
    model = build_model(cfg, components=comps, component_kwargs=ckw, **(model_kwargs or {})).to(dev)
    if memory_root is not None:
        from ..memory.archive import EventArchive
        from ..memory.consolidate import Consolidator
        from ..memory.index import MemoryIndex
        from ..memory.replay import ReplaySampler
        os.makedirs(memory_root, exist_ok=True)
        archive = EventArchive(os.path.join(memory_root, "archive"), cfg, fsync=False)
        replay = SyncedReplay(ReplaySampler(archive, seed=seed))
        if cfg.use_memory:
            index = MemoryIndex(os.path.join(memory_root, "index"), cfg.mem_key_dim, cfg.mem_segment_tokens,
                                cfg.d_model, device=dev, fsync=False)
            model.retriever.attach_index(index)
            consolidator = ServingConsolidator(Consolidator(archive, index, model.retriever))
    if trace_root is not None:
        from ..observe.trace import TraceRecorder
        recorder = TraceRecorder(trace_root)
    trainer = TargetSyncTrainer(make_trainer(model, cfg, trainer_stage, train_semantic, **(trainer_kwargs or {})),
                                model, target_decay)
    akw = dict(archive=archive, recorder=recorder, replay=replay, trainer=trainer, index=index,
               consolidator=consolidator, seed=seed)
    akw.update(agent_kwargs or {})
    agent = CamUnifiedAgent(model, **akw)
    return Stack(cfg, model, agent, trainer, enc, archive, replay, index, consolidator, recorder, memory_root,
                 {"device": str(dev), "build_report": dict(model.build_report)})
