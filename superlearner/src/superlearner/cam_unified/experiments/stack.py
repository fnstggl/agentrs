"""The integrated CAM Unified stack used by every controlled experiment (real components only).

`build_stack` assembles `model.build_model` over the default dotted-path registry (B core / entities / reasoner /
beliefs / Kalman layer, C outcome head + world model, D value / option / budget heads, E retriever) plus workstream
A's semantic tokenizer and action generator over a frozen Qwen3 encoder, then wraps it in `agent.CamUnifiedAgent`
with D's `CombinedTrainer` and E's archive / replay / memory index / consolidator, and E's `CheckpointOps` for the
checkpoint semantics of spec 11.5 / 13.7.

Declared configuration choices for the numeric mechanism experiments (recorded in `cfg.declared_conditions` and in
every run's ``config.json``):

* ``sem_model = Qwen/Qwen3-0.6B`` (the brief allows the test model for numeric-mechanism experiments; the encoder is
  frozen and cached in Stages 1-3, architecture decision 9).
* decoder LoRA off (``QwenSemantic(lora=False)``): MechWorld tools take one typed number each, so no action has a
  free-string hole and the decoder path (the only LoRA consumer while the encoder is frozen) is never run; keeping
  PEFT attached only adds its per-call adapter toggling to every encoder call.
* ``ActionGenerator(exact_rows=False)``: batched learner-path log-probabilities (the row-exact mode evaluates every
  row separately, which dominated decision time at B=16); typed-head-only actions have no bf16 decoder drift, and the
  agent records `policy/logp_maxdev_lag0` so any acting/learning mismatch stays visible.
* reasoning at a fixed J = 4 iterations (``CamUnifiedModel(reason_iters=4)``, one of cfg.reason_iters): the
  adaptive-halting training pass unrolls all 16 iterations for every recomputed decision (+5 GB and ~1 s per update
  at B=8); fixed J keeps the latent reasoner in every decision at a quarter of the cost.
* imagination settings at dev size (``imag_branches=4, imag_samples=2, imag_horizons=(4,)``); training rollouts act
  with the decoder (``use_imagination=False`` while acting, the on-policy data PPO needs); evaluation reports both
  decoder and imagination acting so inference-time search is separated from learning (spec 19).

Two adapters work around defects found while integrating, and two settings remove pure overhead (reported as
interface change requests; production components are not modified here):

* `SafeEntityEncoder` - B's `EntityEncoder.forward` fails (``amax`` over zero tokens) when no row of a step has an
  observation token, which happens in replay windows where every sequence holds a decision that delivered nothing
  (e.g. a ``noop`` between two ticks). The adapter returns no entity tokens for such a step and delegates otherwise.
* `SyncingReplay` - `CamUnifiedAgent` notifies its replay with ``add(records)``; E's `ReplaySampler` has no ``add``
  and only sees new archive events after ``sync()``, so without this adapter replay never returns a sequence and the
  world / adaptation / memory objectives silently never run.
* `MemoTokenizer` - A's `ObservationTokenizer.slot_codes` caches the episode slot-code table per registry *object*
  and rebuilds it (a QR decomposition of a [max(n, D), D] matrix) for every new registry; replay tokenisation and
  the outcome-query text encoder create fresh registries on every call (about 2-3 s of every update at dev scale).
  The table is a deterministic function of (registry RNG state, registry size, D), so memoising it under that key
  returns the identical tensor (tested bit-for-bit).
* Tool registry persistence - replayed actions are re-encoded through A's tokenizer, which finds an action's schema
  only in its process-local registry (`ObservationTokenizer.resolve_tool`); after a restart (or for another run's
  replay, condition D) every archived action of an unseen tool raises "unknown tool". `save_checkpoint` writes the
  registry (``tools.json``) and `load_condition` re-registers it for the conditions that restore experience; the
  registry bound is raised to 200k schemas (default 8192 would evict the tools of old episodes within one lifetime).
* ``torch.distributions`` argument validation is switched off in experiment processes
  (`Distribution.set_default_validate_args(False)`): the outcome head builds thousands of distribution objects per
  update and every validation is a device synchronisation; it never changes a value.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..agent import CamUnifiedAgent
from ..config import CamConfig, make_config
from ..interfaces import TokenSet
from ..model import DEFAULT_COMPONENT_PATHS, build_model, resolve_path

DEFAULT_SEM = "Qwen/Qwen3-0.6B"
RUNS_ROOT = os.environ.get("CAM_X_RUNS", "/workspace/agentrs/runs/cam_unified/wave2/X")
GPU_LOCK = "/workspace/agentrs/runs/cam_unified/gpu_heavy.lock"
DECLARED = ("sem_model:Qwen3-0.6B(numeric-mechanism experiments)", "decoder_lora:off(no string holes)",
            "action_generator:exact_rows=False")


MODEL_KW = {"reason_iters": 4}


def experiment_config(**over) -> CamConfig:
    """Dev-scale configuration of the X experiments (see the module docstring for every deviation from defaults)."""
    base = dict(sem_model=DEFAULT_SEM, imag_branches=4, imag_samples=2, imag_horizons=(4,), mem_top_k=8,
                use_imagination=False, declared_conditions=DECLARED)
    base.update(over)
    return make_config(base.pop("scale", "dev"), **base)


# ============================================================================ adapters (see module docstring)

class SafeEntityEncoder(nn.Module):
    """Delegates to B's EntityEncoder except for a step with no valid observation token in any row, for which it
    returns an empty TokenSet (there are no entities to encode)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, obs_tokens: TokenSet, type_tokens=None) -> TokenSet:
        if obs_tokens.N == 0 or not bool(obs_tokens.mask.any()):
            return TokenSet.empty(obs_tokens.B, obs_tokens.x.shape[2], device=obs_tokens.x.device,
                                  dtype=obs_tokens.x.dtype)
        if type_tokens is None:
            return self.inner(obs_tokens)
        return self.inner(obs_tokens, type_tokens)


def memo_tokenizer_class():
    """`ObservationTokenizer` with slot-code tables memoised by (registry RNG state, size) - see the module
    docstring. Defined lazily so that importing this module does not import the semantic package."""
    import hashlib
    from collections import OrderedDict
    from ..semantic.tokenizer import ObservationTokenizer

    class MemoTokenizer(ObservationTokenizer):
        _memo_size = 4096

        def slot_codes(self, registry) -> torch.Tensor:
            t = self._codes.get(registry)
            if t is not None:
                return t
            memo = self.__dict__.setdefault("_code_memo", OrderedDict())
            key = (hashlib.sha256(repr(registry.rng.getstate()).encode()).hexdigest(), int(registry.max),
                   str(self.device))
            t = memo.get(key)
            if t is None:
                t = super().slot_codes(registry)
                memo[key] = t
                if len(memo) > self._memo_size:
                    memo.popitem(last=False)
            else:
                memo.move_to_end(key)
                self._codes[registry] = t
            return t

    return MemoTokenizer


class SyncingReplay:
    """E's ReplaySampler behind the agent's ReplayP: ``add`` marks new archive events, ``sample`` syncs the sampler's
    membership tables first. Checkpoint state is the sampler's own (``state_dict`` / ``load_state_dict``)."""

    def __init__(self, sampler):
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "dirty", True)
        object.__setattr__(self, "syncs", 0)

    def add(self, records) -> None:
        object.__setattr__(self, "dirty", True)

    def sample(self, n: int, length: int) -> list:
        if self.dirty:
            self.sampler.sync()
            object.__setattr__(self, "dirty", False)
            object.__setattr__(self, "syncs", self.syncs + 1)
        return self.sampler.sample(n, length)

    def state_dict(self) -> dict:
        return self.sampler.state_dict()

    def load_state_dict(self, st: dict) -> None:
        self.sampler.load_state_dict(st)
        object.__setattr__(self, "dirty", True)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "sampler"), name)


# ============================================================================ the stack

@dataclass
class CamStack:
    cfg: CamConfig
    sem: object
    model: nn.Module
    agent: CamUnifiedAgent
    trainer: object = None
    archive: object = None
    replay: object = None
    index: object = None
    consolidator: object = None
    ops: object = None
    run_dir: Optional[str] = None
    info: dict = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.agent.device

    def param_counts(self) -> dict:
        return self.model.count_parameters()


def _param_groups(model: nn.Module) -> tuple:
    """(core, heads, semantic) parameter lists for D's CombinedTrainer: the shared backbone, the decoder bridge (+ any
    LoRA) at lr_sem, everything else trainable as heads. Disjoint by construction."""
    core = [p for p in model.core.parameters() if p.requires_grad]
    gen = model.action_generator
    sem_ids = {id(p) for p in gen.bridge.parameters()} | {id(p) for p in getattr(gen, "lora", [])}
    core_ids = {id(p) for p in core}
    seen, heads, sem = set(), [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in core_ids or id(p) in seen:
            continue
        seen.add(id(p))
        (sem if id(p) in sem_ids else heads).append(p)
    return core, heads, sem


def build_stack(cfg: Optional[CamConfig] = None, *, run_dir: Optional[str] = None, seed: int = 0, device=None,
                experience: bool = True, train: bool = True, sem=None, agent_kw: Optional[dict] = None,
                trainer_kw: Optional[dict] = None, recorder=None, model_kw: Optional[dict] = None) -> CamStack:
    """Assemble the real stack. ``experience``: attach an archive / replay / memory index / consolidator under
    ``run_dir`` (required then); ``train``: attach D's CombinedTrainer (stage "continual": every objective on).
    ``sem``: reuse an already loaded QwenSemantic (it is frozen and stateless apart from its content-hash cache)."""
    from ..learn.trainer import CombinedTrainer
    from ..memory.checkpoint import CheckpointOps
    from ..semantic.actions import ActionGenerator
    from ..semantic.encoder import QwenSemantic
    from ..semantic.grammar import GrammarOptions
    cfg = cfg or experiment_config()
    torch.distributions.Distribution.set_default_validate_args(False)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    if sem is None:
        sem = QwenSemantic(cfg, model_name=cfg.sem_model, lora=False, device=str(dev))
    torch.manual_seed(int(seed))
    tok = memo_tokenizer_class()(cfg, sem, tool_registry_size=200_000)
    gen = ActionGenerator(cfg, sem, tok, opts=GrammarOptions(max_string_tokens=64, max_action_tokens=256),
                          exact_rows=False)
    ent = SafeEntityEncoder(resolve_path(DEFAULT_COMPONENT_PATHS["entity_encoder"])(cfg))
    mkw = dict(MODEL_KW)
    mkw.update(model_kw or {})
    model = build_model(cfg, components={"semantic": sem, "tokenizer": tok, "action_generator": gen,
                                         "entity_encoder": ent}, **mkw).to(dev)
    archive = replay = index = cons = None
    if experience:
        if run_dir is None:
            raise ValueError("experience=True needs a run_dir")
        from ..memory.archive import EventArchive
        from ..memory.consolidate import Consolidator
        from ..memory.index import MemoryIndex
        from ..memory.replay import ReplaySampler
        os.makedirs(run_dir, exist_ok=True)
        archive = EventArchive(os.path.join(run_dir, "archive"), cfg, fsync=False)
        replay = SyncingReplay(ReplaySampler(archive, seed=int(seed)))
        if cfg.use_memory and model.retriever is not None:
            index = MemoryIndex(os.path.join(run_dir, "index"), cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model,
                                device=dev, fsync=False)
            model.retriever.attach_index(index)
            cons = Consolidator(archive, index, model.retriever)
    trainer = None
    if train:
        core, heads, semp = _param_groups(model)
        kw = dict(stage="continual", amp=dev.type == "cuda", grad_probe_every=50)
        kw.update(trainer_kw or {})
        trainer = CombinedTrainer(cfg, core=core, heads=heads, semantic=semp or None, **kw)
    akw = dict(seq_len=8, n_seqs=2, adapt_seq_len=33, adapt_n_seqs=2, fresh_max_steps=64)
    akw.update(agent_kw or {})
    agent = CamUnifiedAgent(model, archive=archive, recorder=recorder, replay=replay, trainer=trainer, index=index,
                            consolidator=cons, seed=int(seed), **akw)
    ops = CheckpointOps(agent)
    st = CamStack(cfg, sem, model, agent, trainer, archive, replay, index, cons, ops, run_dir,
                  {"build_s": time.time() - t0, "seed": int(seed), "device": str(dev)})
    return st


def rewrap_replay(stack: CamStack) -> None:
    """After a CheckpointOps restore that created a raw ReplaySampler, put the syncing adapter back."""
    a = stack.agent
    if a.replay is not None and not isinstance(a.replay, SyncingReplay):
        a.replay = SyncingReplay(a.replay)
    stack.replay, stack.archive, stack.index, stack.consolidator = a.replay, a.archive, a.index, a.consolidator


def save_checkpoint(stack: CamStack, path: str, full: bool = True) -> dict:
    """``<path>/weights.safetensors`` (weights-only file) and, with ``full``, ``<path>/lifetime`` (E's full-lifetime
    directory: weights, optimiser, fast/latent state, RNG, counters, archive / index / replay / consolidator
    pointers)."""
    os.makedirs(path, exist_ok=True)
    out = {"weights": stack.ops.save_weights(os.path.join(path, "weights.safetensors"))}
    if full:
        if stack.archive is not None:
            stack.archive.flush()
        out["lifetime"] = stack.ops.save_full_lifetime(os.path.join(path, "lifetime"))
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(stack.cfg.to_dict(), fh, indent=1, default=list)
    tools = [t.to_dict() for t in stack.model.tokenizer._tool_by_key.values()]
    with open(os.path.join(path, "tools.json"), "w") as fh:
        json.dump(tools, fh)
    out["tools"] = len(tools)
    return out


def restore_tools(stack: CamStack, ckpt: str) -> int:
    """Re-register the tool schemas a checkpoint's lifetime used (see the module docstring)."""
    from ..interfaces import ToolSchema
    p = os.path.join(ckpt, "tools.json")
    if not os.path.exists(p):
        return 0
    with open(p) as fh:
        tools = [ToolSchema.from_dict(d) for d in json.load(fh)]
    stack.model.tokenizer.register_tools(tools)
    return len(tools)


def load_condition(stack: CamStack, ckpt: Optional[str], condition: str) -> dict:
    """Spec 13.7 transfer conditions on this (fresh-process) stack: "A" fresh (nothing loaded), "B" weights only
    (``load_weights_only``: optimiser, fast state, experience access, version and counters reset), "C" weights +
    optimiser state, "D" weights + optimiser + historical replay / experience access, "F" the full lifetime
    (everything, incl. fast/latent state, RNG and counters)."""
    c = condition.upper()
    ops = stack.ops
    if c == "A":
        r = ops.load_transfer_condition(None, "A")
    elif c == "B":
        r = ops.load_weights_only(os.path.join(ckpt, "weights.safetensors"))
        r = {"restored": ["weights"], "condition": "B", "digest": r.get("digest")}
    elif c in ("C", "D"):
        r = ops.load_transfer_condition(os.path.join(ckpt, "lifetime"), c)
    elif c == "F":
        r = ops.load_full_lifetime(os.path.join(ckpt, "lifetime"))
        r["condition"] = "F"
    else:
        raise ValueError(f"unknown condition {condition!r}")
    if c in ("D", "F"):
        r["tools_restored"] = restore_tools(stack, ckpt)
    rewrap_replay(stack)
    return r


def state_report(stack: CamStack) -> dict:
    """What a condition actually restored (used by tests and written next to results): optimiser moments, experience
    handles, model version."""
    a = stack.agent
    n_mom = sum(len(o.state) for o in a.optimizers.values())
    return {"optimizer_state_tensors": n_mom, "archive": a.archive is not None, "replay": a.replay is not None,
            "index": a.index is not None, "consolidator": a.consolidator is not None, "version": int(a.version),
            "archive_count": a.archive.count() if a.archive is not None else 0}
