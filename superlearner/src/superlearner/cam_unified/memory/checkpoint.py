"""Checkpoint semantics (implements ``CheckpointOpsP``; spec 11.5 and 13.7, architecture §8).

Agent attribute contract (duck-typed; only ``model`` and ``config`` are required):

    model            nn.Module holding every learned parameter / buffer of CAM (core, heads, retriever, LoRA, ...)
    config           CamConfig (or a JSON-serialisable dict) - the fixed model configuration
    weights_exclude  optional tuple of state-dict key prefixes NOT written to weight files (e.g. the frozen pretrained
                     semantic base, which is re-loaded from ``config.sem_model``; trainable LoRA weights are kept)
    optimizers       optional dict name -> torch.optim.Optimizer;  make_optimizers() -> dict  (fresh ones)
    schedulers       optional dict name -> LR scheduler;           make_schedulers() -> dict
    fast_state       optional FastState (per-environment beliefs; never weights);  init_fast_state() -> FastState
    latent_state     optional Z_t tensor;                                          init_latent_state() -> tensor
    registries       optional per-environment EntityRegistry list (episode-local; saved/restored and reset together
                     with the fast state, so restored beliefs / latent state keep referring to the same slots)
    archive, index, replay, consolidator   optional EventArchive / MemoryIndex / ReplaySampler / Consolidator handles
    (retriever)      any module of ``model`` with ``attach_index`` / ``key_version`` (e.g. ``Retriever``) gets the
                     restored index attached; its readable versions are pinned in the index by full-lifetime saves
    rng              optional torch.Generator, numpy Generator, random.Random, or a dict of them
    version          optional int model version (number of learner updates)
    counters         optional JSON-serialisable dict (steps, episodes, ...)

Files:

* weights-only (``save_weights``): ONE safetensors file. Tensors = ``model.state_dict()`` minus ``weights_exclude``
  (parameters and buffers only). Metadata = {"format", "config" (JSON), "param_digest", "excluded_prefixes"}. Nothing
  else: no optimiser moments, replay, RNG, version, counters, fast/latent state, context or environment state.
* full lifetime (``save_full_lifetime``): a directory written as ``<path>.tmp`` then renamed: ``weights.safetensors``,
  ``optim.pt`` (optimiser + scheduler state), ``state.pt`` (fast state, latent state, entity registries, RNG states),
  ``lifetime.json`` (version, counters, config, archive / index / replay pointers and sampler state, consolidator
  pointer, file digests). The archive itself is never copied into a checkpoint (it is append-only and verified by
  pointer: count + chunk digests); ``EventArchive.backup`` makes verified copies. The memory index is not copied
  either: the save pins (``MemoryIndex.pin``) the embedding versions the checkpoint's retriever can read, so a later
  migration cannot drop them, and a restore reconciles the index with the restored retriever (serves the checkpoint's
  version, or re-encodes from raw events to the restored target version; see ``load_parts``).

``load_parts`` / ``load_transfer_condition`` restore chosen parts of a full-lifetime checkpoint, giving spec 13.7's
transfer conditions: A fresh model+optimiser, B pretrained weights only, C weights + optimiser state, D weights +
optimiser + historical replay (experience access).

Architecture identity and migration (plan 2026-11-26 §3 W9). Weight files also carry ``arch_hash`` (sha256 of the
sorted (key, dtype, shape) list of the saved tensors: the exact parameter layout), ``code_commit`` and
``code_dirty`` (the git commit of the code that wrote the file, and whether ``superlearner/src`` had uncommitted
changes). ``load_weights(path, allow_new=prefixes, allow_dropped=prefixes, allow_config=fields)`` loads a file whose
layout differs from the model's only by EXPLICITLY allowed differences: tensors the model has and the file lacks
whose key starts with an ``allow_new`` prefix keep the model's construction-time initialisation (from the config
seed) and are listed in ``report["migrated"]``; file tensors the model no longer has are dropped only under an
``allow_dropped`` prefix; architecture-config fields may differ only when named in ``allow_config``. Anything else
(missing / unexpected keys, a shape change, a config difference) raises ``CheckpointMismatch`` before the model is
touched, so adding a module never silently invalidates (or silently half-loads) earlier checkpoints.

Weights-only semantics (``load_weights_only``). Only parameters and buffers cross; everything a lifetime accumulates
is reset to what a freshly built agent has: optimiser moments and schedulers, fast state, latent state, entity
registries, the working state of every environment row (outcome trackers, options, the previous action A_{t-1},
time of the last decision), pending post-action observations, the on-policy fresh buffer, experience access
(archive, memory index, replay, consolidator, retriever index), learner diagnostics (loss magnitudes, update and
memory-sync logs), the trainer's counters / stage / loss weights / history, the agent's version and counters, the
tokenizer's tool registry and the semantic model's content caches, and the agent RNGs (re-seeded from ``seed``, or
the agent's own construction seed). ``seed=`` given explicitly also re-seeds the process-global python / numpy /
torch (CPU and CUDA) generators, so two processes that load the same file with the same seed act identically.
Experience-derived normalisation statistics kept as module buffers (e.g. ``state_norm.{count,mean,var}`` of the
outcome head and world model) ARE part of the weights by decision: they define the function the weights compute
(inputs are standardised with them), they are not addressable experience, and a transfer condition that reset them
would evaluate a different function from the one trained. A condition that wants them reset says so explicitly
(``reset_buffers=prefixes``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from typing import Optional, Sequence

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..interfaces import EntityRegistry, FastState
from ._fs import fsync_dir, fsync_file, sha256_file
from .errors import ArchiveCorrupted, CheckpointMismatch

WEIGHTS_FORMAT = "cam_unified.weights.v1"
LIFETIME_FORMAT = "cam_unified.lifetime.v1"
WEIGHTS_METADATA_KEYS = frozenset({"format", "config", "param_digest", "excluded_prefixes", "arch_hash", "code_commit",
                                   "code_dirty"})
# configuration fields that determine parameter shapes / semantics of the weights
ARCH_FIELDS = ("n_layers", "d_model", "n_heads", "n_state", "ffn_mult", "sem_model", "sem_tokens_per_obs",
               "sem_lora_rank", "sem_lora_alpha", "sem_lora_targets", "n_prefix", "n_fourier", "max_entities",
               "reason_layers", "max_reason_iters", "n_particles", "d_particle", "bayes_feat", "stoch_groups",
               "stoch_classes", "mem_key_dim", "mem_segment", "mem_segment_tokens", "n_options")


# ============================================================================ config helpers
def config_to_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
    elif dataclasses.is_dataclass(cfg):
        d = dataclasses.asdict(cfg)
    else:
        d = dict(cfg)
    return json.loads(json.dumps(d, default=list))


def config_from_dict(d: dict):
    """Rebuild a ``CamConfig`` from its JSON form (lists back to tuples where the field default is a tuple)."""
    from ..config import CamConfig
    kw = {}
    for f in dataclasses.fields(CamConfig):
        if f.name in d:
            v = d[f.name]
            default = f.default if f.default is not dataclasses.MISSING else None
            if isinstance(default, tuple) and isinstance(v, list):
                v = tuple(v)
            kw[f.name] = v
    return CamConfig(**kw)


# configuration fields added by the DOM rebuild (plan 2026-11-26 §2) that change the parameter layout. A file written
# before a field existed is compared as if it had the field's CamConfig default (so pre-DOM files still load into a
# pre-DOM-shaped model, and a DOM-shaped model refuses them unless the difference is explicitly allowed).
DOM_ARCH_FIELDS = ("state_gate", "sem_xattn_every", "sem_xattn_dim", "dec_xattn_every", "n_intent", "value_channel",
                   "d_satellite", "wm_transition_blocks", "wm_ema_target", "use_memory")


def _arch(d: dict) -> dict:
    out = {k: d.get(k) for k in ARCH_FIELDS if k in d}
    defaults = _config_defaults()
    for k in DOM_ARCH_FIELDS:
        if k in d:
            out[k] = d[k]
        elif k in defaults:
            out[k] = defaults[k]
    return out


def _config_defaults() -> dict:
    from ..config import CamConfig
    out = {}
    for f in dataclasses.fields(CamConfig):
        if f.default is not dataclasses.MISSING:
            out[f.name] = json.loads(json.dumps(f.default, default=list))
    return out


def arch_hash_of(sd: dict) -> str:
    """sha256 of the sorted (key, dtype, shape) list of a state dict: the exact parameter layout (values excluded)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}\n".encode())
    return h.hexdigest()


def model_arch_hash(model: torch.nn.Module, exclude_prefixes=()) -> str:
    """``arch_hash_of`` the model's state dict minus ``exclude_prefixes`` (what ``write_weights`` would record)."""
    return arch_hash_of({k: v for k, v in model.state_dict().items()
                         if not any(k.startswith(p) for p in exclude_prefixes)})


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def code_version(repo: Optional[str] = None) -> dict:
    """{"commit": HEAD sha of the repository holding this code (None when unknown), "dirty": True when
    superlearner/src has uncommitted changes (None when unknown)}."""
    repo = repo or _REPO
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=20)
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        st = subprocess.run(["git", "status", "--porcelain", "--", "superlearner/src"], cwd=repo, capture_output=True,
                            text=True, timeout=20)
        return {"commit": head.stdout.strip() or None,
                "dirty": None if st.returncode != 0 else bool(st.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _digest(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


# ============================================================================ weight files
def inspect_weights(path: str) -> dict:
    """What a weights file contains, read without materialising tensors: {"tensors": {name: (dtype, shape)},
    "metadata": {...}, "bytes": int}."""
    out = {}
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            sl = f.get_slice(k)
            out[k] = (str(sl.get_dtype()), tuple(sl.get_shape()))
    return {"tensors": out, "metadata": meta, "bytes": os.path.getsize(path)}


def read_weights(path: str) -> tuple:
    """-> (state_dict, config dict, metadata). Verifies format and digest."""
    info = inspect_weights(path)
    meta = info["metadata"]
    if meta.get("format") != WEIGHTS_FORMAT:
        raise CheckpointMismatch(f"{path} is not a CAM Unified weights file (format={meta.get('format')})")
    sd = load_file(path, device="cpu")
    if _digest(sd) != meta.get("param_digest"):
        raise CheckpointMismatch(f"{path}: tensor digest does not match its metadata")
    if meta.get("arch_hash") and arch_hash_of(sd) != meta["arch_hash"]:
        raise CheckpointMismatch(f"{path}: tensor layout does not match its recorded arch_hash")
    return sd, json.loads(meta.get("config", "{}")), meta


def write_weights(model: torch.nn.Module, config, path: str, exclude_prefixes=()) -> dict:
    sd = {}
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in exclude_prefixes):
            continue
        sd[k] = v.detach().cpu().contiguous().clone()
    cv = code_version()
    meta = {"format": WEIGHTS_FORMAT, "config": json.dumps(config_to_dict(config), sort_keys=True),
            "param_digest": _digest(sd), "excluded_prefixes": json.dumps(list(exclude_prefixes)),
            "arch_hash": arch_hash_of(sd), "code_commit": cv["commit"] or "", "code_dirty": json.dumps(cv["dirty"])}
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp-{os.path.basename(path)}-{os.getpid()}")
    save_file(sd, tmp, metadata=meta)
    fsync_file(tmp)
    os.replace(tmp, path)
    fsync_dir(d)
    return {"path": path, "tensors": len(sd), "bytes": os.path.getsize(path), "digest": meta["param_digest"],
            "arch_hash": meta["arch_hash"], "code_commit": cv["commit"], "code_dirty": cv["dirty"],
            "sha256": sha256_file(path)}


# ============================================================================ RNG
def _rng_capture(agent_rng) -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

    def one(g):
        if isinstance(g, torch.Generator):
            return ("torch", g.get_state())
        if isinstance(g, np.random.Generator):
            return ("numpy", g.bit_generator.state)
        if isinstance(g, random.Random):
            return ("python", g.getstate())
        raise TypeError(f"unsupported rng {type(g).__name__}")
    if agent_rng is not None:
        st["agent"] = {k: one(v) for k, v in agent_rng.items()} if isinstance(agent_rng, dict) else one(agent_rng)
    return st


def _rng_restore(st: dict, agent_rng) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])

    def one(g, s):
        kind, val = s
        if kind == "torch":
            g.set_state(val)
        elif kind == "numpy":
            g.bit_generator.state = val
        else:
            g.setstate(val)
    if agent_rng is not None and "agent" in st:
        if isinstance(agent_rng, dict):
            for k, g in agent_rng.items():
                one(g, st["agent"][k])
        else:
            one(agent_rng, st["agent"])


def _fast_to_dict(fs: Optional[FastState]):
    if fs is None:
        return None
    return {f.name: (getattr(fs, f.name).detach().cpu().clone() if isinstance(getattr(fs, f.name), torch.Tensor)
                     else json.loads(json.dumps(getattr(fs, f.name)))) for f in dataclasses.fields(FastState)}


def _fast_from_dict(d, device=None):
    if d is None:
        return None
    return FastState(**{k: (v.to(device) if isinstance(v, torch.Tensor) and device is not None else v)
                        for k, v in d.items()})


def _registry_to_dict(r) -> dict:
    if not isinstance(r, EntityRegistry):
        raise TypeError(f"registries must be EntityRegistry objects, got {type(r).__name__}")
    st = r.rng.getstate()
    return {"max": int(r.max), "slot": dict(r.slot), "etype": dict(r.etype), "last_seen": dict(r.last_seen),
            "free": list(r.free), "rng": [st[0], list(st[1]), st[2]]}


def _registry_from_dict(d: dict) -> EntityRegistry:
    r = EntityRegistry(int(d["max"]))
    r.slot, r.etype, r.last_seen = dict(d["slot"]), dict(d["etype"]), dict(d["last_seen"])
    r.free = list(d["free"])
    r.rng.setstate((d["rng"][0], tuple(d["rng"][1]), d["rng"][2]))
    return r


def _model_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _trainer_snapshot(t) -> Optional[dict]:
    """The learner bookkeeping a trainer was constructed with (CombinedTrainer attribute names)."""
    if t is None:
        return None
    snap = {}
    for k in ("n_updates", "n_skipped", "stage"):
        if hasattr(t, k):
            snap[k] = getattr(t, k)
    if isinstance(getattr(t, "weights", None), dict):
        snap["weights"] = dict(t.weights)
    return snap


def _trainer_restore(t, snap: Optional[dict]) -> None:
    """Back to the constructed bookkeeping: counters 0, initial stage and loss weights, history cleared."""
    snap = snap or {}
    if hasattr(t, "n_updates"):
        t.n_updates = 0
    if hasattr(t, "n_skipped"):
        t.n_skipped = 0
    if "stage" in snap:
        t.stage = snap["stage"]
    if "weights" in snap and isinstance(getattr(t, "weights", None), dict):
        t.weights = dict(snap["weights"])
    h = getattr(t, "history", None)
    if hasattr(h, "clear"):
        h.clear()
    if hasattr(t, "_cos_idx"):
        t._cos_idx = None


def _clear_semantic_caches(sem) -> bool:
    """Drop the semantic model's content caches (encoder states / pooled embeddings) if it has any."""
    if callable(getattr(sem, "clear_caches", None)):
        sem.clear_caches()
        return True
    hit = False
    for name in ("_cache", "_ecache"):
        c = getattr(sem, name, None)
        if hasattr(c, "clear"):
            c.clear()
            hit = True
    if hit and hasattr(sem, "_cache_bytes"):
        sem._cache_bytes = 0
    return hit


def _seed_one(g, s: int) -> None:
    if isinstance(g, torch.Generator):
        g.manual_seed(int(s))
    elif isinstance(g, np.random.Generator):
        g.bit_generator.state = type(g.bit_generator)(int(s)).state
    elif isinstance(g, random.Random):
        g.seed(int(s))
    else:
        raise TypeError(f"unsupported rng {type(g).__name__}")


def _reseed(agent, seed: Optional[int]) -> str:
    """Re-seed the agent's RNGs from `seed` (default: the agent's construction seed, else 0); an explicit seed
    also re-seeds the process-global python / numpy / torch / CUDA generators."""
    base = int(seed if seed is not None else (getattr(agent, "seed", 0) or 0))
    py, tg = getattr(agent, "py_rng", None), getattr(agent, "torch_rng", None)
    if isinstance(py, random.Random):
        py.seed(base)
    if isinstance(tg, torch.Generator):
        tg.manual_seed(base)
    if py is None and tg is None:
        rng = getattr(agent, "rng", None)
        if isinstance(rng, dict):
            for i, k in enumerate(sorted(rng)):
                _seed_one(rng[k], base + 1 + i)
        elif rng is not None:
            _seed_one(rng, base)
    if seed is not None:
        random.seed(base)
        np.random.seed(base % 2 ** 32)
        torch.manual_seed(base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base)
        return "rng(agent+global)"
    return "rng(agent)"


class _quiet_scheduler_warning:
    """Silence torch's "scheduler.step() before optimizer.step()" warning for the constructor-equivalent step."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        return self

    def __exit__(self, *a):
        return self._cm.__exit__(*a)


# ============================================================================ the operations
class CheckpointOps:
    """The seven checkpoint operations of ``CheckpointOpsP`` (plus partial loads for transfer conditions) over an
    agent object following the attribute contract of the module docstring."""

    def __init__(self, agent):
        if not hasattr(agent, "model") or not hasattr(agent, "config"):
            raise TypeError("agent must expose .model and .config")
        self.agent = agent
        # the learner state a freshly built agent starts with (restored by weights-only loads of a used object)
        self._trainer_init = _trainer_snapshot(getattr(agent, "trainer", None))

    @property
    def exclude(self) -> tuple:
        return tuple(getattr(self.agent, "weights_exclude", ()) or ())

    def _retrievers(self) -> list:
        return [m for m in self.agent.model.modules()
                if hasattr(m, "attach_index") and hasattr(m, "detach_index") and hasattr(m, "compatible_versions")]

    # ------------------------------------------------------------------ weights
    def save_weights(self, path: str) -> dict:
        """Weights-only file: parameters + buffers + fixed configuration, nothing else."""
        return write_weights(self.agent.model, self.agent.config, path, self.exclude)

    def _load_weights_into_model(self, path: str, allow_new: Sequence[str] = (), allow_dropped: Sequence[str] = (),
                                 allow_config: Sequence[str] = ()) -> dict:
        """Validate the file against the model (configuration, key set, shapes, dtypes) and load it. Every check
        runs before the model is modified. See the module docstring for the migration rules."""
        sd, cfg_d, meta = read_weights(path)
        allow_new, allow_dropped, allow_config = tuple(allow_new), tuple(allow_dropped), set(allow_config)
        mine = _arch(config_to_dict(self.agent.config))
        theirs = _arch(cfg_d)
        diff = {k: (theirs.get(k), mine.get(k)) for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
        config_changes = {k: v for k, v in diff.items() if k in allow_config}
        diff = {k: v for k, v in diff.items() if k not in allow_config}
        if diff:
            raise CheckpointMismatch(f"architecture configuration differs (file, agent): {diff}")
        excl = tuple(json.loads(meta.get("excluded_prefixes", "[]")))
        msd = self.agent.model.state_dict()
        expected = {k for k in msd if not any(k.startswith(p) for p in excl)}
        missing = sorted(expected - set(sd))
        unexpected = sorted(set(sd) - expected)
        migrated = [k for k in missing if any(k.startswith(p) for p in allow_new)]
        dropped = [k for k in unexpected if any(k.startswith(p) for p in allow_dropped)]
        bad_missing = [k for k in missing if k not in set(migrated)]
        bad_unexpected = [k for k in unexpected if k not in set(dropped)]
        if bad_missing or bad_unexpected:
            raise CheckpointMismatch(
                f"weights file keys differ: missing {bad_missing[:5]} ({len(bad_missing)}), unexpected "
                f"{bad_unexpected[:5]} ({len(bad_unexpected)}); a deliberate architecture change loads with "
                f"load_weights(path, allow_new=(prefix, ...), allow_dropped=(prefix, ...))")
        shapes = [(k, tuple(sd[k].shape), tuple(msd[k].shape)) for k in sorted(set(sd) & expected)
                  if tuple(sd[k].shape) != tuple(msd[k].shape)]
        if shapes:
            raise CheckpointMismatch(f"tensor shapes differ (key, file, model): {shapes[:5]} ({len(shapes)})")
        dtypes = [(k, str(sd[k].dtype), str(msd[k].dtype)) for k in sorted(set(sd) & expected)
                  if sd[k].dtype != msd[k].dtype and not (sd[k].is_floating_point() and msd[k].is_floating_point())]
        if dtypes:
            raise CheckpointMismatch(f"tensor dtypes differ (key, file, model): {dtypes[:5]}")
        load = {k: v for k, v in sd.items() if k in expected}
        res = self.agent.model.load_state_dict(load, strict=False)
        if res.unexpected_keys:
            raise CheckpointMismatch(f"unexpected keys {res.unexpected_keys[:5]}")
        return {"tensors": len(load), "digest": meta["param_digest"], "config": cfg_d,
                "arch_hash": meta.get("arch_hash") or arch_hash_of(sd),
                "model_arch_hash": model_arch_hash(self.agent.model, excl),
                "code_commit": meta.get("code_commit") or None, "migrated": migrated, "dropped": dropped,
                "config_changes": config_changes, "sha256": sha256_file(path)}

    def load_weights(self, path: str, allow_new: Sequence[str] = (), allow_dropped: Sequence[str] = (),
                     allow_config: Sequence[str] = ()) -> dict:
        """Load parameters/buffers into the model with an explicit migration (module docstring); nothing else is
        touched. Returns {"tensors", "digest", "arch_hash" (file), "model_arch_hash", "code_commit", "migrated"
        (model keys kept at their initialisation), "dropped" (file keys ignored), "config_changes", "sha256"}."""
        return self._load_weights_into_model(path, allow_new, allow_dropped, allow_config)

    def load_weights_only(self, path: str, seed: Optional[int] = None, allow_new: Sequence[str] = (),
                          allow_dropped: Sequence[str] = (), allow_config: Sequence[str] = (),
                          reset_buffers: Sequence[str] = (), attach: Optional[dict] = None) -> dict:
        """Load parameters/buffers only, then reset everything a lifetime accumulates (module docstring) so nothing
        of the source lifetime other than its weights reaches the agent. `seed`: re-seeds the agent RNGs and, when
        given, the process-global generators. `attach`: fresh experience stores of the NEW lifetime to attach after
        the reset ({"archive": ..., "replay": ..., "index": ..., "consolidator": ...}; never the source's)."""
        info = self._load_weights_into_model(path, allow_new, allow_dropped, allow_config)
        if reset_buffers:
            info["reset_buffers"] = self._reset_buffers(reset_buffers)
        info["reset"] = self.reset_learning_state(seed=seed)
        if attach:
            for k, v in attach.items():
                if k not in ("archive", "index", "replay", "consolidator"):
                    raise ValueError(f"attach: unknown experience store {k!r}")
                setattr(self.agent, k, v)
                if k == "index" and v is not None:
                    for m in self._retrievers():
                        m.attach_index(v)
            info["attached"] = sorted(k for k, v in attach.items() if v is not None)
        return info

    def _reset_buffers(self, prefixes: Sequence[str]) -> list:
        """Reinitialise running-statistics buffers under `prefixes` (count/mean -> 0, var -> 1, other -> 0)."""
        out = []
        for name, buf in self.agent.model.named_buffers():
            if any(name.startswith(p) for p in prefixes) and buf.is_floating_point():
                with torch.no_grad():
                    buf.fill_(1.0 if name.rsplit(".", 1)[-1] in ("var", "running_var") else 0.0)
                out.append(name)
        return out

    def reset_learning_state(self, seed: Optional[int] = None) -> dict:
        """Everything a lifetime accumulated except the weights (weights-only semantics, module docstring). Each
        item is applied through the agent's own method when it has one, else to the documented attributes; the
        result lists what was reset."""
        a = self.agent
        done = []
        # experience access first, so that nothing below can flush source-lifetime records into a store
        self.reset_experience_access()
        done.append("experience_access")
        # re-seed before the working state is rebuilt (its entity registries draw their seeds from the agent RNG, so
        # the rebuilt state is a function of `seed` only) and again after it, so that a used agent ends in exactly
        # the RNG state of a freshly built agent that loaded the same file with the same seed
        done.append(_reseed(a, seed))
        done += self._reset_context()
        self.reset_fast_state()
        self.reset_optimizer()
        done += ["fast_state", "optimizer"]
        if hasattr(a, "version"):
            a.version = 0
        if hasattr(a, "counters"):
            a.counters = {}
        done += ["version", "counters"]
        done += self._reset_diagnostics()
        m = a.model
        tok = getattr(m, "tokenizer", None)
        if callable(getattr(tok, "clear_tool_registry", None)):
            tok.clear_tool_registry()
            done.append("tool_registry")
        sem = getattr(m, "semantic", None)
        if sem is not None and _clear_semantic_caches(sem):
            done.append("semantic_caches")
        _reseed(a, seed)          # last: the rebuilt working state / registries consumed agent RNG draws
        return {"reset": done, "seed": seed}

    def _reset_context(self) -> list:
        """Per-environment context of the source lifetime: pending post-action observations (dropped, never flushed
        into a store), the fresh buffer, and the working state of every row (Z, entity registries, outcome trackers,
        options, the previous action A_{t-1}, the last decision time)."""
        a = self.agent
        for name in ("_post", "_pending", "_last_fresh", "_last_eid"):
            if hasattr(a, name):
                cur = getattr(a, name)
                setattr(a, name, {} if isinstance(cur, dict) else None)
        if hasattr(a, "fresh"):
            a.fresh = []
        out = ["pending_observations", "fresh_buffer"]
        if callable(getattr(a, "reset_working_state", None)):
            a.reset_working_state()
            out.append("working_state")
        return out

    def _reset_diagnostics(self) -> list:
        """Learner bookkeeping: loss magnitudes, update / memory-sync logs, provenance counters, timings, decision
        counts, the trainer's counters / stage / weights / history (or the holder's own reset_state())."""
        a = self.agent
        for name, empty in (("loss_mag", dict), ("update_log", list), ("memory_sync_log", list),
                            ("exclude_events", set)):
            if hasattr(a, name):
                setattr(a, name, empty())
        if isinstance(getattr(a, "provenance_stats", None), dict):
            a.provenance_stats = {k: 0 for k in a.provenance_stats}
        if isinstance(getattr(a, "timing", None), dict):
            a.timing = {k: 0.0 for k in a.timing}
        if hasattr(a, "n_decisions"):
            a.n_decisions = 0
        if isinstance(getattr(a, "_lp_valid", None), list):
            a._lp_valid = [False] * len(a._lp_valid)
        out = ["learner_diagnostics"]
        for holder in ("trainer", "learner", "reward_model"):
            obj = getattr(a, holder, None)
            if obj is None:
                continue
            if callable(getattr(obj, "reset_state", None)):
                obj.reset_state()
                out.append(holder)
            elif holder == "trainer":
                _trainer_restore(obj, self._trainer_init)
                out.append("trainer")
        calls = getattr(a.model, "calls", None)
        if isinstance(calls, dict):
            calls.clear()
        return out

    # ------------------------------------------------------------------ full lifetime
    def save_full_lifetime(self, path: str, pin_memory: bool = True) -> dict:
        """Write a full-lifetime checkpoint directory (see the module docstring). ``pin_memory`` pins the memory
        versions this checkpoint's retriever can read, so restoring it never needs a full re-encode."""
        a = self.agent
        path = os.path.abspath(path)
        tmp = f"{path}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        t0 = time.time()
        write_weights(a.model, a.config, os.path.join(tmp, "weights.safetensors"), self.exclude)
        opt = {"optimizers": {k: o.state_dict() for k, o in (getattr(a, "optimizers", None) or {}).items()},
               "schedulers": {k: s.state_dict() for k, s in (getattr(a, "schedulers", None) or {}).items()}}
        torch.save(opt, os.path.join(tmp, "optim.pt"))
        lat = getattr(a, "latent_state", None)
        regs = getattr(a, "registries", None)
        state = {"fast_state": _fast_to_dict(getattr(a, "fast_state", None)),
                 "latent_state": None if lat is None else lat.detach().cpu().clone(),
                 "registries": None if regs is None else [_registry_to_dict(r) for r in regs],
                 "rng": _rng_capture(getattr(a, "rng", None))}
        torch.save(state, os.path.join(tmp, "state.pt"))
        life = {"format": LIFETIME_FORMAT, "saved": time.time(), "version": int(getattr(a, "version", 0) or 0),
                "counters": json.loads(json.dumps(getattr(a, "counters", {}) or {})),
                "config": config_to_dict(a.config), "archive": None, "index": None, "replay": None,
                "consolidator": None}
        arc = getattr(a, "archive", None)
        if arc is not None:
            arc.flush()
            life["archive"] = arc.pointer()
        idx = getattr(a, "index", None)
        pin_name = f"ckpt:{path}"
        pinned = []
        if idx is not None:
            if pin_memory:
                have = set(idx.versions())
                pinned = sorted({v for r in self._retrievers() for v in r.compatible_versions()} & have)
                if pinned:      # merged with the old pin until the new checkpoint is in place
                    idx.pin(pin_name + "#saving", pinned)
            life["index"] = idx.pointer()
            life["index"]["pinned"] = pinned
            life["index"]["retriever_versions"] = sorted({v for r in self._retrievers()
                                                          for v in r.compatible_versions()})
        rep = getattr(a, "replay", None)
        if rep is not None:
            life["replay"] = rep.state_dict()
        con = getattr(a, "consolidator", None)
        if con is not None:
            life["consolidator"] = con.pointer()
        life["files"] = {f: sha256_file(os.path.join(tmp, f)) for f in ("weights.safetensors", "optim.pt", "state.pt")}
        with open(os.path.join(tmp, "lifetime.json"), "w") as fh:
            json.dump(life, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        for f in os.listdir(tmp):
            fsync_file(os.path.join(tmp, f))
        fsync_dir(tmp)
        prev = None
        if os.path.exists(path):
            prev = f"{path}.prev-{int(time.time() * 1e3)}"
            os.replace(path, prev)
        os.replace(tmp, path)
        fsync_dir(os.path.dirname(path))
        if idx is not None and pin_memory:
            if pinned:
                idx.pin(pin_name, pinned)
            else:
                idx.unpin(pin_name)
            idx.unpin(pin_name + "#saving")
        if prev is not None:
            shutil.rmtree(prev, ignore_errors=True)
        return {"path": path, "seconds": time.time() - t0, "pinned_versions": pinned,
                "bytes": sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))}

    def _read_lifetime(self, path: str) -> dict:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            prevs = sorted(p for p in os.listdir(os.path.dirname(path)) if p.startswith(os.path.basename(path) + ".prev-"))
            if not prevs:
                raise FileNotFoundError(path)
            path = os.path.join(os.path.dirname(path), prevs[-1])     # crash between the two renames
        with open(os.path.join(path, "lifetime.json")) as fh:
            life = json.load(fh)
        if life.get("format") != LIFETIME_FORMAT:
            raise CheckpointMismatch(f"{path} is not a CAM Unified lifetime checkpoint")
        for f, h in life["files"].items():
            if sha256_file(os.path.join(path, f)) != h:
                raise CheckpointMismatch(f"{path}/{f} does not match its digest")
        life["_dir"] = path
        return life

    def load_full_lifetime(self, path: str, memory_reencode_budget: Optional[int] = None) -> dict:
        """Restore everything: weights, optimisers/schedulers, RNG, version, counters, fast and latent state, entity
        registries, archive (verified by pointer), index (reconciled with the restored retriever), replay cursors/RNG
        and consolidator state."""
        return self.load_parts(path, weights=True, optimizer=True, experience=True, fast_state=True, rng=True,
                               counters=True, memory_reencode_budget=memory_reencode_budget)

    def load_parts(self, path: str, weights: bool = True, optimizer: bool = False, experience: bool = False,
                   fast_state: bool = False, rng: bool = False, counters: bool = False,
                   memory_reencode_budget: Optional[int] = None) -> dict:
        """Restore selected parts of a full-lifetime checkpoint; everything not restored is reset to fresh.

        With ``experience`` the memory index is reconciled with the restored retriever: if the index serves a
        version the retriever cannot read (it migrated after the checkpoint), the checkpoint's pinned version is
        served again when it covers every segment, else segments are re-encoded from raw events to the restored
        target version (``memory_reencode_budget`` segments now, None = all); until that completes ``retrieve``
        returns empty memory instead of raising. The outcome is ``result["memory"]``."""
        a = self.agent
        life = self._read_lifetime(path)
        d = life["_dir"]
        restored = []
        if weights:
            self._load_weights_into_model(os.path.join(d, "weights.safetensors"))
            restored.append("weights")
        if optimizer:
            opt = torch.load(os.path.join(d, "optim.pt"), map_location="cpu", weights_only=False)
            self.reset_optimizer()
            for k, o in (getattr(a, "optimizers", None) or {}).items():
                if k in opt["optimizers"]:
                    o.load_state_dict(opt["optimizers"][k])
            for k, s in (getattr(a, "schedulers", None) or {}).items():
                if k in opt["schedulers"]:
                    s.load_state_dict(opt["schedulers"][k])
            restored.append("optimizer")
        else:
            self.reset_optimizer()
        st = torch.load(os.path.join(d, "state.pt"), map_location="cpu", weights_only=False) \
            if (fast_state or rng) else None
        if fast_state:
            a.fast_state = _fast_from_dict(st["fast_state"], _model_device(a.model))
            if hasattr(a, "latent_state"):
                lat = st["latent_state"]
                a.latent_state = None if lat is None else lat.to(_model_device(a.model))
            if hasattr(a, "registries") or st.get("registries") is not None:
                regs = st.get("registries")
                a.registries = [] if regs is None else [_registry_from_dict(r) for r in regs]
            restored.append("fast_state")
        else:
            self._reset_context()
            self.reset_fast_state()
        if rng:
            _rng_restore(st["rng"], getattr(a, "rng", None))
            restored.append("rng")
        if counters:
            if hasattr(a, "version"):
                a.version = int(life["version"])
            if hasattr(a, "counters"):
                a.counters = dict(life["counters"])
            restored.append("counters")
        else:
            if hasattr(a, "version"):
                a.version = 0
            if hasattr(a, "counters"):
                a.counters = {}
            self._reset_diagnostics()
        memory = None
        if experience:
            memory = self._attach_experience(life, memory_reencode_budget)
            restored.append("experience")
        else:
            self.reset_experience_access()
        return {"restored": restored, "version": life["version"], "dir": d, "memory": memory}

    def load_transfer_condition(self, path: Optional[str], condition: str) -> dict:
        """Spec 13.7: "A" fresh model and optimiser (no load; weights re-initialised by ``agent.reinit()`` when
        available, else left as constructed), "B" pretrained weights only, "C" weights + optimiser state,
        "D" weights + optimiser + historical replay / experience access."""
        c = condition.upper()
        if c == "A":
            if callable(getattr(self.agent, "reinit", None)):
                self.agent.reinit()
            r = self.reset_learning_state()
            return {"restored": [], "condition": "A", "reset": r["reset"]}
        if path is None:
            raise ValueError(f"condition {c} needs a checkpoint")
        parts = {"B": {}, "C": {"optimizer": True}, "D": {"optimizer": True, "experience": True}}
        if c not in parts:
            raise ValueError("condition must be one of A, B, C, D")
        if c == "B" and os.path.isfile(path):
            r = self.load_weights_only(path)
            return {"restored": ["weights"], "condition": "B", **{k: v for k, v in r.items() if k != "config"}}
        r = self.load_parts(path, weights=True, **parts[c])
        r["condition"] = c
        return r

    def _attach_experience(self, life: dict, budget: Optional[int] = None) -> dict:
        a = self.agent
        p = life.get("archive")
        if p is not None:
            arc = getattr(a, "archive", None)
            if arc is None:
                from .archive import EventArchive
                arc = EventArchive(p["root"], chunk_rows=p.get("chunk_rows"), cold_dir=p.get("cold_dir"))
            have = {c["n"]: (c["rows"], c["sha256"]) for c in arc.chunks()}
            for n, rows, sha in p["chunks"]:
                if have.get(n, (None, None))[0] != rows:
                    raise ArchiveCorrupted(f"archive {p['root']} lost chunk {n} recorded by the checkpoint")
            if arc.count() < p["count"]:
                raise ArchiveCorrupted(f"archive {p['root']} has {arc.count()} events; the checkpoint saw {p['count']}")
            a.archive = arc
        p = life.get("index")
        if p is not None:
            idx = getattr(a, "index", None)
            if idx is None:
                from .index import MemoryIndex
                idx = MemoryIndex(p["root"], p["key_dim"], p["value_tokens"], p["value_dim"], p["shard_rows"])
            a.index = idx
            for m in a.model.modules():
                if hasattr(m, "attach_index") and hasattr(m, "detach_index"):
                    m.attach_index(idx)
        p = life.get("replay")
        if p is not None:
            rep = getattr(a, "replay", None)
            if rep is None and getattr(a, "archive", None) is not None:
                from .replay import ReplaySampler
                rep = ReplaySampler(a.archive, path=p["path"], mix=p["mix"], recent_window=p["recent_window"],
                                    rare_z=p["rare_z"], novelty_z=p["novelty_z"], alpha_under=p["alpha_under"])
            if rep is not None:
                rep.load_state_dict(p)
                a.replay = rep
        p = life.get("consolidator")
        if p is not None:
            con = getattr(a, "consolidator", None)
            if con is None:
                con = self._make_consolidator(p)
                a.consolidator = con
            con.load_state_dict(p["state"] if "state" in p else p)
        return self._reconcile_memory(life, budget)

    def _make_consolidator(self, p: Optional[dict] = None):
        """Rebuild a Consolidator over the restored archive / index / retriever (fails explicitly if one is missing)."""
        a = self.agent
        rets = self._retrievers()
        if getattr(a, "archive", None) is None or getattr(a, "index", None) is None or not rets:
            raise CheckpointMismatch("the checkpoint holds consolidation state, but the agent has no archive, memory "
                                     "index or retriever module to rebuild the consolidator")
        from .consolidate import Consolidator
        p = p or {}
        return Consolidator(a.archive, a.index, rets[0], segment_len=p.get("segment_len"), fanout=p.get("fanout"),
                            batch_segments=p.get("batch_segments", 64))

    def _reconcile_memory(self, life: dict, budget: Optional[int]) -> dict:
        """Make the index serve a version the restored retriever can read (see ``load_parts``)."""
        a = self.agent
        idx = getattr(a, "index", None)
        rets = self._retrievers()
        if idx is None or not rets:
            return {"status": "no-memory"}
        ret = rets[0]
        compat = ret.compatible_versions()
        cur = idx.current_version
        if cur is None or cur in compat:
            return {"status": "served", "version": cur}
        want = (life.get("index") or {}).get("current_version")
        for v in (want, ret.key_version):                  # prefer what the checkpoint served
            if v is not None and v in compat and v in idx.versions() and idx.coverage(v) >= 1.0:
                idx.set_current(v)
                return {"status": "restored-version", "version": v, "replaced": cur}
        if getattr(a, "archive", None) is None:
            raise CheckpointMismatch(f"memory index serves {cur}, which the restored retriever cannot read, and no "
                                     "archive is attached to re-encode it from raw events")
        con = getattr(a, "consolidator", None)
        if con is None or con.retriever is not ret:
            con = self._make_consolidator((life.get("consolidator") or {}))
            if getattr(a, "consolidator", None) is None:
                a.consolidator = con
        r = con.reencode_step(max_items=budget)
        if idx.current_version in compat:
            return {"status": "reencoded", "version": idx.current_version, "replaced": cur,
                    "reencoded": r["reencoded"], "dropped": r["dropped"]}
        ret.await_version(ret.key_version)
        return {"status": "reencode-pending", "version": ret.key_version, "served": cur, "coverage": r["coverage"],
                "reencoded": r["reencoded"]}

    # ------------------------------------------------------------------ resets
    def reset_fast_state(self) -> None:
        """Fresh per-environment adaptive state: fast beliefs, latent state, entity registries (emptied)."""
        a = self.agent
        init = getattr(a, "init_fast_state", None)
        a.fast_state = init() if callable(init) else None
        if hasattr(a, "latent_state"):
            init = getattr(a, "init_latent_state", None)
            a.latent_state = init() if callable(init) else None
        if hasattr(a, "registries"):
            a.registries = []

    def reset_optimizer(self) -> dict:
        """Fresh optimisers (no moments, initial learning rates) and schedulers (back at epoch 0).

        Prefers the agent's ``make_optimizers`` / ``make_schedulers``. Without them, moments are cleared in place,
        each group's lr is set back to its ``initial_lr`` (recorded by LR schedulers) and every scheduler is re-run
        from epoch 0 as at construction. Groups whose initial lr is unknown (no scheduler, no factory) keep their
        current lr and are listed in the result (``AdaptiveGroupLR`` never leaves its multiplier in ``lr``)."""
        a = self.agent
        info = {"factory": False, "unknown_initial_lr": []}
        mk = getattr(a, "make_optimizers", None)
        if callable(mk):
            a.optimizers = mk()
            info["factory"] = True
        else:
            for name, o in (getattr(a, "optimizers", None) or {}).items():
                o.state.clear()
                for i, g in enumerate(o.param_groups):
                    if "initial_lr" in g:
                        g["lr"] = g["initial_lr"]
                    else:
                        info["unknown_initial_lr"].append((name, i))
        mk = getattr(a, "make_schedulers", None)
        if callable(mk):
            a.schedulers = mk()
        elif getattr(a, "schedulers", None):
            if info["factory"]:
                raise CheckpointMismatch("the agent re-creates its optimisers (make_optimizers) but not its schedulers "
                                         "(make_schedulers); the old schedulers would keep driving the old optimisers")
            for s in a.schedulers.values():
                if not isinstance(s, torch.optim.lr_scheduler.LRScheduler):
                    raise CheckpointMismatch(f"cannot reset scheduler {type(s).__name__} in place; provide "
                                             "make_schedulers()")
                for g in s.optimizer.param_groups:
                    g["lr"] = g["initial_lr"]
                s.last_epoch = -1
                s._step_count = 0                 # exactly what the constructor's initial step starts from
                with _quiet_scheduler_warning():
                    s.step()
        return info

    def reset_experience_access(self) -> None:
        """Detach archive, memory index, replay and consolidator (the data on disk is untouched), and everything else
        the agent's own ``reset_experience_access`` drops (CamUnifiedAgent: the on-policy fresh buffer)."""
        a = self.agent
        own = getattr(a, "reset_experience_access", None)
        if callable(own):
            own()
        for name in ("archive", "index", "replay", "consolidator"):
            if hasattr(a, name):
                setattr(a, name, None)
        for m in a.model.modules():
            if hasattr(m, "detach_index") and hasattr(m, "attach_index"):
                m.detach_index()
