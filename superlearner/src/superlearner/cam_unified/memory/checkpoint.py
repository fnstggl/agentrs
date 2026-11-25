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
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import time
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..interfaces import EntityRegistry, FastState
from ._fs import fsync_dir, fsync_file, sha256_file
from .errors import ArchiveCorrupted, CheckpointMismatch

WEIGHTS_FORMAT = "cam_unified.weights.v1"
LIFETIME_FORMAT = "cam_unified.lifetime.v1"
WEIGHTS_METADATA_KEYS = frozenset({"format", "config", "param_digest", "excluded_prefixes"})
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


def _arch(d: dict) -> dict:
    return {k: d.get(k) for k in ARCH_FIELDS if k in d}


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
    return sd, json.loads(meta.get("config", "{}")), meta


def write_weights(model: torch.nn.Module, config, path: str, exclude_prefixes=()) -> dict:
    sd = {}
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in exclude_prefixes):
            continue
        sd[k] = v.detach().cpu().contiguous().clone()
    meta = {"format": WEIGHTS_FORMAT, "config": json.dumps(config_to_dict(config), sort_keys=True),
            "param_digest": _digest(sd), "excluded_prefixes": json.dumps(list(exclude_prefixes))}
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp-{os.path.basename(path)}-{os.getpid()}")
    save_file(sd, tmp, metadata=meta)
    fsync_file(tmp)
    os.replace(tmp, path)
    fsync_dir(d)
    return {"path": path, "tensors": len(sd), "bytes": os.path.getsize(path), "digest": meta["param_digest"]}


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

    def _load_weights_into_model(self, path: str) -> dict:
        sd, cfg_d, meta = read_weights(path)
        mine = _arch(config_to_dict(self.agent.config))
        theirs = _arch(cfg_d)
        diff = {k: (theirs.get(k), mine.get(k)) for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
        if diff:
            raise CheckpointMismatch(f"architecture configuration differs (file, agent): {diff}")
        excl = tuple(json.loads(meta.get("excluded_prefixes", "[]")))
        model_keys = set(self.agent.model.state_dict())
        expected = {k for k in model_keys if not any(k.startswith(p) for p in excl)}
        if set(sd) != expected:
            raise CheckpointMismatch(f"weights file keys differ: missing {sorted(expected - set(sd))[:5]}, "
                                     f"unexpected {sorted(set(sd) - expected)[:5]}")
        res = self.agent.model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise CheckpointMismatch(f"unexpected keys {res.unexpected_keys[:5]}")
        return {"tensors": len(sd), "digest": meta["param_digest"], "config": cfg_d}

    def load_weights_only(self, path: str) -> dict:
        """Load parameters/buffers only; then reset fast state, optimiser and experience access, version and counters,
        so nothing of the source lifetime other than its weights reaches the agent."""
        info = self._load_weights_into_model(path)
        self.reset_fast_state()
        self.reset_optimizer()
        self.reset_experience_access()
        if hasattr(self.agent, "version"):
            self.agent.version = 0
        if hasattr(self.agent, "counters"):
            self.agent.counters = {}
        return info

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
            self.reset_fast_state()
            self.reset_optimizer()
            self.reset_experience_access()
            if hasattr(self.agent, "version"):
                self.agent.version = 0
            return {"restored": [], "condition": "A"}
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
        """Detach archive, memory index, replay and consolidator (the data on disk is untouched)."""
        a = self.agent
        for name in ("archive", "index", "replay", "consolidator"):
            if hasattr(a, name):
                setattr(a, name, None)
        for m in a.model.modules():
            if hasattr(m, "detach_index") and hasattr(m, "attach_index"):
                m.detach_index()
