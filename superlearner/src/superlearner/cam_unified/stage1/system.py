"""Assembly of the complete CAM Unified system from the real workstream classes (no test doubles).

`build_system(run_dir, SystemConfig)` constructs, in this order:

    QwenSemantic (A)                          pretrained semantic model (not an nn.Module; counted separately)
    MemoryIndex (E)                           <run_dir>/memory_index   version-partitioned keys / memory values
    CamUnifiedModel via model.build_model     tokenizer + action generator (A), core / entities / reasoner /
                                              beliefs / Kalman layer (B), outcome head / world model (C), retriever (E,
                                              attached to the MemoryIndex), value / option / budget heads (D);
                                              planner = the built-in multistep belief-aware LatentImaginationPlanner
    CombinedTrainer (D)                       three optimiser groups: core backbone / CAM heads / semantic (LoRA + bridge)
    ArtifactStore + EventArchive (E)          <run_dir>/artifacts, <run_dir>/archive (verbatim long originals)
    ReplaySampler (E)                         stratified persistent replay over the archive (through `BranchView`)
    Consolidator (E)                          archive -> memory segments -> keys in the MemoryIndex
    TraceRecorder (F)                         <run_dir>/traces (one DecisionTrace per decision, update-ref patches)
    CamUnifiedAgent                           the agent around the model, with every collaborator above injected
    CheckpointOps (E)                         weights-only / full-lifetime checkpoints over the agent

Every number that defines the run (the CamConfig, the system config, the build report of every component) is
written to <run_dir>/run.json by the runner. Nothing here imports `testing` or `integration_doubles`.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from ..agent import CamUnifiedAgent
from ..config import CamConfig, make_config
from ..model import CamUnifiedModel, build_model
from .branch import BranchView


@dataclass
class SystemConfig:
    """What a Stage-1 system is made of (beyond the CamConfig scale + overrides)."""
    scale: str = "dev"
    overrides: dict = field(default_factory=dict)      # make_config(scale, **overrides)
    sem_model: Optional[str] = None                    # None = cfg.sem_model (Qwen3-1.7B); tests: cfg.sem_test_model
    device: str = "cuda"
    seed: int = 0
    actor_id: str = "cam0"
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

    def cam_config(self) -> CamConfig:
        return make_config(self.scale, **dict(self.overrides))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class System:
    cfg: CamConfig
    sc: SystemConfig
    run_dir: str
    semantic: object
    model: CamUnifiedModel
    trainer: object
    archive: object
    artifacts: object
    view: BranchView
    index: object
    replay: object
    consolidator: object
    recorder: object
    agent: CamUnifiedAgent
    ops: object

    def paths(self) -> dict:
        return {k: os.path.join(self.run_dir, k) for k in ("archive", "artifacts", "memory_index", "traces")}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_semantic(cfg: CamConfig, model_name: Optional[str], device: str):
    from ..semantic.encoder import QwenSemantic
    return QwenSemantic(cfg, model_name=model_name or cfg.sem_model, device=device)


def make_index(cfg: CamConfig, root: str, device: str):
    from ..memory.index import MemoryIndex
    return MemoryIndex(root, cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model, device=device)


def make_trainer(cfg: CamConfig, model: CamUnifiedModel, sc: SystemConfig, log_path: Optional[str]):
    """D's CombinedTrainer with the three parameter groups: the shared UnifiedCore (whose per-objective gradient
    contributions are probed), every other CAM parameter (heads), and the semantic-side trainables (the decoder
    LoRA tensors and the latent-to-language bridge) at cfg.lr_sem."""
    from ..learn.trainer import CombinedTrainer
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


def build_system(run_dir: str, sc: SystemConfig, semantic=None, exclude_events: Optional[set] = None) -> System:
    """The full real stack over `run_dir` (created). `semantic` reuses an already loaded QwenSemantic (tests)."""
    from ..memory.archive import EventArchive
    from ..memory.artifacts import ArtifactStore
    from ..memory.checkpoint import CheckpointOps
    from ..memory.consolidate import Consolidator
    from ..memory.replay import ReplaySampler
    from ..observe.trace import TraceRecorder
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cfg = sc.cam_config()
    seed_everything(sc.seed)
    dev = sc.device if (sc.device != "cuda" or torch.cuda.is_available()) else "cpu"
    sem = semantic if semantic is not None else make_semantic(cfg, sc.sem_model, dev)
    index = make_index(cfg, os.path.join(run_dir, "memory_index"), dev)
    model = build_model(cfg, components={"semantic": sem}, component_kwargs={"retriever": {"index": index}},
                        imag_candidates=sc.imag_candidates, imag_horizon=sc.imag_horizon,
                        imag_samples=sc.imag_samples)
    model.to(dev)
    model.eval()
    trainer = make_trainer(cfg, model, sc, os.path.join(run_dir, "learner", "trainer.jsonl"))
    artifacts = ArtifactStore(os.path.join(run_dir, "artifacts"))
    archive = EventArchive(os.path.join(run_dir, "archive"), cfg, artifacts=artifacts, autoflush=sc.archive_autoflush)
    view = BranchView(archive, exclude_events or set())
    replay = ReplaySampler(view, seed=sc.seed)
    consolidator = Consolidator(view, index, model.retriever)
    recorder = TraceRecorder(os.path.join(run_dir, "traces"))
    agent = CamUnifiedAgent(model, archive=archive, recorder=recorder, replay=replay, trainer=trainer, index=index,
                            consolidator=consolidator, actor_id=sc.actor_id, seed=sc.seed, seq_len=sc.seq_len,
                            n_seqs=sc.n_seqs, adapt_seq_len=sc.adapt_seq_len, adapt_n_seqs=sc.adapt_n_seqs,
                            greedy=sc.greedy, fresh_max_steps=sc.fresh_max_steps)
    agent.exclude_events = view.exclude
    ops = CheckpointOps(agent)
    return System(cfg, sc, run_dir, sem, model, trainer, archive, artifacts, view, index, replay, consolidator,
                  recorder, agent, ops)


def describe(system: System) -> dict:
    """Run-description block: configs, build report, parameter counts per component (spec 1.1)."""
    m = system.model
    counts = m.count_parameters()
    ag = m.action_generator
    return {"cam_config": json.loads(json.dumps(system.cfg.to_dict(), default=list)),
            "system_config": system.sc.to_dict(), "build_report": dict(m.build_report), "parameters": counts,
            "action_generator_parts": ag.count_parameters() if hasattr(ag, "count_parameters") else None,
            "semantic_model": getattr(system.semantic, "model_name", None),
            "planner": type(m.planner).__name__,
            "imagination": {"candidates": m.imag_candidates, "horizon": m.imag_horizon, "samples": m.imag_samples},
            "trainer": {"stage": system.trainer.stage.name,
                        "groups": {g: sum(p.numel() for p in ps) for g, ps in system.trainer.groups.items()}}}
