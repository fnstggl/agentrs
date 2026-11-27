"""The canonical CAM Unified factory (redesign plan 2026-11-26 §2; owner W1):

    build_agent(scale, *, env_kind, sem_model, device, run_dir, overrides) -> System

`scale` is a DOM preset ("dom", "dom_dev", "dom_150m": config.make_preset) or a base scale ("dev", "main",
"superlearner", ...: config.make_config); `overrides` are CamConfig fields applied on top. The System holds the
full real stack, in this order:

    QwenSemantic (frozen; LoRA only when cfg.sem_lora)      pretrained semantic model (counted separately)
    MemoryIndex                                             <run_dir>/memory_index (used only with use_memory)
    CamUnifiedModel via model.build_model                   every default (real) component; build_report checked:
                                                            no test double may be part of a System
    CombinedTrainer                                         core backbone / CAM heads / semantic-side groups
    ArtifactStore + EventArchive                            <run_dir>/artifacts, <run_dir>/archive
    ReplaySampler (through BranchView), Consolidator        persistent replay; memory segments
    TraceRecorder                                           <run_dir>/traces
    CamUnifiedAgent + CheckpointOps                         the acting agent with its collaborators injected
    LabelStore                                              <run_dir>/labels (learner side; the agent never holds it)
    RewardModel                                             selfeval.reward_model (W8; the seam stub returns 0)
    learner (runtime.AgentLearner)                          the LearnerHook the canonical loop drives (W7's
                                                            IntegratedLearner replaces it behind the same protocol)

`env_kind` ("openworld", "rapid", "mech", "experience", "integrity", ...) is recorded in the System and the run
description; `System.make_env(...)` hands it to W2's `envs.factory.make_env` once that is merged. Every runner,
experiment and capability driver builds through here (stage1.system.build_system is a wrapper).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .config import PRESETS, CamConfig, make_config, make_preset

ENV_KINDS = ("openworld", "ow", "rapid", "mech", "experience", "integrity", "toy", "mixed")
DOUBLE_MODULES = ("integration_doubles", ".testing", "tests.")


class OperatorLabels:
    """LabelSource over an operator that releases `OutcomeLabel`s (envs.experience.ExperienceOperator.release_labels:
    every resolved episode on every call, PENDING ones included): take_labels() returns each resolved label once as
    an envs.labels.LabelRecord. Learner-side assembly (never handed to the agent); W2's own take_labels replaces it."""

    def __init__(self, operator, evaluator: str = ""):
        self.operator = operator
        self.evaluator = evaluator
        self._seen: set = set()
        self.version = getattr(operator, "version", None)

    def episode_of(self, env_id: str):
        fn = getattr(self.operator, "episode_of", None)
        return fn(env_id) if callable(fn) else None

    def take_labels(self) -> list:
        from .envs.labels import from_outcome_label
        out = []
        for lab in self.operator.release_labels():
            rec = from_outcome_label(lab, evaluator=self.evaluator)
            if rec is None:                              # pending: not a label
                continue
            key = (rec.env_id, rec.episode, rec.goal_id, rec.kind)
            if key in self._seen:
                continue
            self._seen.add(key)
            out.append(rec)
        return out


class ConcatLabels:
    """Several LabelSources as one (learner side)."""

    def __init__(self, sources):
        self.sources = [s for s in sources if s is not None]
        vs = [getattr(s, "version", None) for s in self.sources]
        self.version = None if all(v is None for v in vs) else "+".join(str(v) for v in vs)

    def take_labels(self) -> list:
        from .runtime import take_labels
        return [r for s in self.sources for r in take_labels(s)]

    def episode_of(self, env_id: str):
        from .runtime import label_episode_of
        for s in self.sources:
            e = label_episode_of(s, env_id)
            if e is not None:
                return e
        return None


def make_cam_config(scale: str, overrides: Optional[dict] = None) -> CamConfig:
    """CamConfig of a preset or base scale with `overrides` applied."""
    over = dict(overrides or {})
    if scale in PRESETS:
        return make_preset(scale, **over)
    return make_config(scale, **over)


@dataclass
class SystemConfig:
    """What a system is made of (beyond the CamConfig scale + overrides)."""
    scale: str = "dev"
    overrides: dict = field(default_factory=dict)      # make_cam_config(scale, overrides)
    sem_model: Optional[str] = None                    # None = cfg.sem_model; tests: cfg.sem_test_model
    device: str = "cuda"
    seed: int = 0
    actor_id: str = "cam0"
    env_kind: Optional[str] = None
    # learner
    seq_len: int = 8                                   # replay sequence length (world / memory objectives)
    n_seqs: int = 2
    adapt_seq_len: int = 17                            # support/query sequences (k in {0..16} < T)
    adapt_n_seqs: int = 1
    fresh_max_steps: int = 64
    trainer_stage: str = "continual"                   # D's stage 7: every objective enabled
    grad_probe_every: int = 1                          # per-objective backbone gradient norms every update
    amp: bool = True
    lr_warmup: int = 0
    # imagination (planner) at matched, bounded compute: None = cfg values
    imag_candidates: Optional[int] = None
    imag_horizon: Optional[int] = None
    imag_samples: Optional[int] = None
    greedy: bool = False
    # durability: fsync the archive's WAL after every append (E's actor setting). Events written after the last
    # checkpoint then survive a crash as an explicitly excluded abandoned branch instead of vanishing silently.
    archive_autoflush: bool = True
    # exact continuation: save the semantic model's content-hash caches (hidden states / pooled embeddings) with
    # each checkpoint. A's encoder output depends on the batch a text is encoded in (bf16, padding), so states
    # recomputed after a restart differ from the cached ones the interrupted process used
    checkpoint_semantic_cache: bool = True
    # per-objective magnitude cap (CamUnifiedAgent.loss_cap; None = off): see the agent's constructor
    loss_cap: Optional[float] = 10.0
    # deterministic GPU kernels (torch.use_deterministic_algorithms + cuBLAS workspace config): measured without
    # it, recomputing the same decoder log-probability twice differed by up to 0.23 nats (cuBLAS algorithm choice)
    deterministic: bool = True
    # memory encoders: migrate the index to the trained encoders every k updates (CamUnifiedAgent.memory_sync_every)
    memory_sync_every: Optional[int] = 4

    def cam_config(self) -> CamConfig:
        return make_cam_config(self.scale, self.overrides)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class System:
    cfg: CamConfig
    sc: SystemConfig
    run_dir: str
    semantic: object
    model: object
    trainer: object
    archive: object
    artifacts: object
    view: object
    index: object
    replay: object
    consolidator: object
    recorder: object
    agent: object
    ops: object
    # v4 (canonical factory)
    label_store: object = None
    reward_model: object = None
    learner: object = None
    env_kind: Optional[str] = None
    build_s: float = 0.0

    def paths(self) -> dict:
        return {k: os.path.join(self.run_dir, k) for k in ("archive", "artifacts", "memory_index", "traces",
                                                           "labels")}

    def learner_for(self, agent):
        """The LearnerHook for a wrapper around this system's agent (e.g. stage1.HarnessAgent, whose update() adds
        its own accounting): the system learner when it drives `agent` itself, else an AgentLearner over `agent`
        with the same reward model."""
        from .runtime import AgentLearner
        if self.learner is not None and getattr(self.learner, "agent", None) is agent:
            return self.learner
        return AgentLearner(agent, reward_model=self.reward_model)

    def make_env(self, *args, **kw):
        """W2's envs.factory.make_env(kind, ...) -> (adapter, operator, LabelSource) for this system's env_kind."""
        try:
            from .envs.factory import make_env
        except ImportError as e:                         # pragma: no cover - until W2 merges
            raise ImportError("envs.factory.make_env (W2) is not available in this checkout") from e
        return make_env(self.env_kind, *args, **kw)


def set_deterministic(on: bool) -> None:
    """Deterministic kernels. The cuBLAS workspace setting must precede the first cuBLAS handle (set it in the
    environment of the process, as the CLI does, for it to take effect after CUDA was initialised)."""
    if not on:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_semantic(cfg: CamConfig, model_name: Optional[str], device: str):
    from .semantic.encoder import QwenSemantic
    return QwenSemantic(cfg, model_name=model_name or cfg.sem_model, device=device, lora=bool(cfg.sem_lora))


def make_index(cfg: CamConfig, root: str, device: str):
    from .memory.index import MemoryIndex
    return MemoryIndex(root, cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model, device=device)


def make_trainer(cfg: CamConfig, model, sc: SystemConfig, log_path: Optional[str]):
    """D's CombinedTrainer with the three parameter groups: the shared UnifiedCore (whose per-objective gradient
    contributions are probed), every other CAM parameter (heads), and the semantic-side trainables (the decoder
    LoRA tensors and the latent-to-language bridge) at cfg.lr_sem."""
    from .learn.trainer import CombinedTrainer
    core_ids = {id(p) for p in model.core.parameters()}
    ag = model.action_generator
    sem_ids = {id(p) for p in getattr(ag, "lora", [])} | {id(p) for p in ag.bridge.parameters()} \
        if hasattr(ag, "bridge") else {id(p) for p in getattr(ag, "lora", [])}
    params = [p for p in model.parameters() if p.requires_grad]
    heads = [p for p in params if id(p) not in core_ids and id(p) not in sem_ids]
    semantic = [p for p in params if id(p) in sem_ids]
    return CombinedTrainer(cfg, core=model.core, heads=heads, semantic=semantic, stage=sc.trainer_stage,
                           amp=sc.amp, grad_probe_every=sc.grad_probe_every, log_path=log_path,
                           lr_warmup=sc.lr_warmup)


def component_classes(model) -> dict:
    """{component: module.qualname} of every component built into the model."""
    out = {}
    for name in ("semantic", "tokenizer", "core", "entity_encoder", "reasoner", "beliefs", "bayes", "outcome_head",
                 "world_model", "retriever", "action_generator", "value_head", "option_policy", "exploration_head",
                 "planner", "query_encoder"):
        obj = getattr(model, name, None)
        out[name] = None if obj is None else f"{type(obj).__module__}.{type(obj).__qualname__}"
    return out


def _is_double(path: str) -> bool:
    mod = path.split(" ")[-1]
    head = mod.split(".")[0]
    return any(d in mod for d in DOUBLE_MODULES) or head.startswith(("test_", "conftest"))


def assert_no_doubles(model) -> dict:
    """Raise when a test double (testing.py / integration_doubles.py / a test module) is part of the model."""
    cls = component_classes(model)
    bad = {k: v for k, v in cls.items() if v and _is_double(v)}
    bad.update({f"report:{k}": v for k, v in (getattr(model, "build_report", {}) or {}).items()
                if isinstance(v, str) and any(d in v for d in DOUBLE_MODULES)})
    if bad:
        raise ValueError(f"a System is built from real components only; test doubles found: {bad}")
    return cls


def build_agent(scale: str = "dom_dev", *, env_kind: Optional[str] = None, sem_model: Optional[str] = None,
                device: Optional[str] = None, run_dir: Optional[str] = None, overrides: Optional[dict] = None,
                seed: Optional[int] = None, semantic=None, system: Optional[SystemConfig | dict] = None,
                exclude_events: Optional[set] = None, components: Optional[dict] = None,
                model_kwargs: Optional[dict] = None, agent_kwargs: Optional[dict] = None,
                learner_factory=None) -> System:
    """The full real stack (see the module docstring). `system`: a SystemConfig (or its fields as a dict) for the
    learner / durability settings; explicit `scale`, `overrides`, `sem_model`, `device`, `seed`, `env_kind` win over
    it. `semantic` reuses an already loaded QwenSemantic; `components` overrides individual real components (they
    must still be real classes). `run_dir` None = a fresh temporary directory. `learner_factory(system)` -> the
    LearnerHook; default: W7's `learn.integrated.IntegratedLearner.from_system(system)` when it exists, else
    runtime.AgentLearner (the agent's own update)."""
    from .agent import CamUnifiedAgent
    from .envs.labels import LabelStore
    from .memory.archive import EventArchive
    from .memory.artifacts import ArtifactStore
    from .memory.checkpoint import CheckpointOps
    from .memory.consolidate import Consolidator
    from .memory.replay import ReplaySampler
    from .model import build_model
    from .observe.trace import TraceRecorder
    from .runtime import AgentLearner
    from .selfeval.reward_model import RewardModel
    from .stage1.branch import BranchView
    t0 = time.time()
    if isinstance(system, dict):
        system = SystemConfig(**system)
    sc = system if system is not None else SystemConfig()
    kw: dict = {"scale": scale}
    if seed is not None:
        kw["seed"] = int(seed)
    if overrides is not None:
        kw["overrides"] = dict(overrides)
    if sem_model is not None:
        kw["sem_model"] = sem_model
    if device is not None:
        kw["device"] = str(device)
    if env_kind is not None:
        kw["env_kind"] = env_kind
    sc = SystemConfig(**{**sc.to_dict(), **kw})
    if sc.env_kind is not None and sc.env_kind not in ENV_KINDS:
        raise ValueError(f"env_kind {sc.env_kind!r}; known: {ENV_KINDS}")
    run_dir = os.path.abspath(run_dir) if run_dir else tempfile.mkdtemp(prefix="cam_system_")
    os.makedirs(run_dir, exist_ok=True)
    cfg = sc.cam_config()
    set_deterministic(sc.deterministic)
    seed_everything(sc.seed)
    dev = sc.device if (sc.device != "cuda" or torch.cuda.is_available()) else "cpu"
    sem = semantic if semantic is not None else make_semantic(cfg, sc.sem_model, dev)
    index = make_index(cfg, os.path.join(run_dir, "memory_index"), dev)   # on disk; attached only with use_memory
    comps = {"semantic": sem, **dict(components or {})}
    ckw = {"retriever": {"index": index}}
    mkw = dict(imag_candidates=sc.imag_candidates, imag_horizon=sc.imag_horizon, imag_samples=sc.imag_samples)
    mkw.update(model_kwargs or {})
    model = build_model(cfg, components=comps, component_kwargs=ckw, **mkw)
    assert_no_doubles(model)
    model.to(dev)
    model.eval()
    trainer = make_trainer(cfg, model, sc, os.path.join(run_dir, "learner", "trainer.jsonl"))
    artifacts = ArtifactStore(os.path.join(run_dir, "artifacts"))
    archive = EventArchive(os.path.join(run_dir, "archive"), cfg, artifacts=artifacts, autoflush=sc.archive_autoflush)
    view = BranchView(archive, exclude_events or set())
    replay = ReplaySampler(view, seed=sc.seed)
    consolidator = Consolidator(view, index, model.retriever) if model.retriever is not None else None
    recorder = TraceRecorder(os.path.join(run_dir, "traces"))
    akw = dict(archive=archive, recorder=recorder, replay=replay, trainer=trainer, index=index,
               consolidator=consolidator, actor_id=sc.actor_id, seed=sc.seed, seq_len=sc.seq_len, n_seqs=sc.n_seqs,
               adapt_seq_len=sc.adapt_seq_len, adapt_n_seqs=sc.adapt_n_seqs, greedy=sc.greedy,
               fresh_max_steps=sc.fresh_max_steps, loss_cap=sc.loss_cap, memory_sync_every=sc.memory_sync_every)
    akw.update(agent_kwargs or {})
    agent = CamUnifiedAgent(model, **akw)
    agent.exclude_events = view.exclude
    ops = CheckpointOps(agent)
    label_store = LabelStore(os.path.join(run_dir, "labels"))
    reward_model = RewardModel(cfg).to(dev)
    if hasattr(model, "attach_reward_model"):          # Phi enters the evaluator's Q; psi version stamped (verifier F3)
        model.attach_reward_model(reward_model)
    system = System(cfg, sc, run_dir, sem, model, trainer, archive, artifacts, view, index, replay, consolidator,
                    recorder, agent, ops, label_store=label_store, reward_model=reward_model, learner=None,
                    env_kind=sc.env_kind)
    if learner_factory is None:
        from .learn import integrated
        il = getattr(integrated, "IntegratedLearner", None)
        learner_factory = il.from_system if il is not None and hasattr(il, "from_system") else \
            (lambda s: AgentLearner(s.agent, reward_model=s.reward_model))
    system.learner = learner_factory(system)
    system.build_s = time.time() - t0
    return system


def describe(system: System) -> dict:
    """Run-description block: configs, build report, component classes, parameter counts (spec 1.1)."""
    m = system.model
    counts = m.count_parameters()
    ag = m.action_generator
    rm = system.reward_model
    return {"cam_config": json.loads(json.dumps(system.cfg.to_dict(), default=list)),
            "system_config": system.sc.to_dict(), "env_kind": system.env_kind,
            "build_report": dict(m.build_report), "components": component_classes(m), "parameters": counts,
            "action_generator_parts": ag.count_parameters() if hasattr(ag, "count_parameters") else None,
            "semantic_model": getattr(system.semantic, "model_name", None),
            "planner": type(m.planner).__name__,
            "imagination": {"candidates": m.imag_candidates, "horizon": m.imag_horizon, "samples": m.imag_samples},
            "trainer": {"stage": system.trainer.stage.name,
                        "groups": {g: sum(p.numel() for p in ps) for g, ps in system.trainer.groups.items()}},
            "reward_model": None if rm is None else {"class": f"{type(rm).__module__}.{type(rm).__qualname__}",
                                                     "stub": bool(getattr(rm, "stub", False)),
                                                     "version": getattr(rm, "version", None)},
            "learner": type(system.learner).__name__ if system.learner is not None else None,
            "build_s": system.build_s}
