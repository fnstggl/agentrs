"""cam_live.credit.checkpoint — atomic checkpoints, tolerant loading, weights-only export, promotion and rollback
(spec 10.3, 10.4, 15.2; manifest CL-05, CL-06).

Layout of a checkpoint root::

    <root>/ckpt_000007_ab12cd34/      one complete checkpoint (the directory appears by a single rename)
        MANIFEST.json                 written last: file list with sha256, architecture signature, versions
        weights/<group>.safetensors   trainable tensors by parameter group (``nn.api.GROUPS``)
        optim/<name>.pt               optimizer state keyed by *parameter name*
        fast_state.pt  lifetime.pt    fast adaptive state and lifetime context
        rng.pt                        python / numpy / torch (CPU and CUDA) generator states
        state.json                    current task, replay sampler indices, pending-outcome pointer and the
                                      pending outcome queries themselves, experiment records, schedules,
                                      authorization-config hash, model / reward versions
    <root>/LATEST  <root>/PROMOTED    pointer files (a convenience: validity is decided by scanning)
    <root>/promotions.jsonl           promotion / refusal / rollback log with the metrics that decided them

Guarantees:

* **Atomic.** Everything is written into ``.tmp-*``, fsynced, then renamed. A process killed at any point leaves
  either the previous checkpoints only, or the previous checkpoints plus the complete new one; :func:`recover`
  removes debris and repairs the pointers. The previous checkpoint is always kept.
* **Nothing lives only on the GPU.** Every tensor is moved to the CPU and written to disk.
* **Tolerant, reported loading.** Weights and optimizer state are matched by parameter name. Parameters that are
  new in the model (an added head) keep their initial values and are *listed* in the load report; nothing is
  skipped silently and nothing breaks because a module was added.
* **Weights-only export** contains neural parameters and fixed configuration and nothing else (no optimizer, no
  fast state, no experience, no task, no RNG): the artefact used for transfer evaluation in a fresh process.
* **Promotion** needs independent validation and retention metrics supplied by the caller; rollback returns to the
  last promoted checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ..contracts import Clock, WallClock
from ._fs import atomic_write_bytes, fsync_dir, fsync_file, fsync_tree, sha256_file

FORMAT_VERSION = 1
EXPORT_FORMAT = "cam_live.weights_only.v1"
MANIFEST = "MANIFEST.json"
GROUP_SEP = "::"
BUFFER_GROUP = "buffers"
FROZEN_GROUP = "backbone_frozen"
#: test seam: called with the name of each durability step (a subprocess test makes it kill the process there)
FAULT_HOOK: Optional[Callable[[str], None]] = None


def _fault(point: str) -> None:
    if FAULT_HOOK is not None:
        FAULT_HOOK(point)


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatch(CheckpointError):
    """Strict loading found missing / unexpected / mismatched tensors. ``report`` has the details."""

    def __init__(self, msg: str, report: "LoadReport"):
        super().__init__(msg)
        self.report = report


# ------------------------------------------------------------------------------------------------ state
@dataclass
class CheckpointState:
    """Everything a restart needs (spec 15.2). Tensors may live on any device; they are saved from the CPU."""
    weights: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)      # group -> parameter name -> tensor
    optimizers: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # name -> ``optimizer_state(...)``
    fast_state: Any = None                   # fast adaptive state m_t (nested tensors / numbers)
    lifetime_context: Any = None             # recurrent lifetime summary of the reward generator
    current_task: Optional[Dict[str, Any]] = None
    replay_state: Dict[str, Any] = field(default_factory=dict)               # replay sampler indices
    pending_pointer: Dict[str, Any] = field(default_factory=dict)            # where the pending-outcome queue lives
    schedules: Dict[str, Any] = field(default_factory=dict)                  # learning-schedule counters
    rng: Dict[str, Any] = field(default_factory=dict)
    auth_config_hash: str = ""
    model_version: str = ""
    reward_version: str = ""
    arch_signature: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)                      # JSON-native
    aux_state: Any = None                    # nested tensors / numbers owned by loss providers (e.g. retention anchors)
    experiments: List[Dict[str, Any]] = field(default_factory=list)          # experiment records (ExperimentRecord)
    pending_outcomes: List[Dict[str, Any]] = field(default_factory=list)     # unresolved outcome queries (OutcomeQuery)


@dataclass
class CheckpointInfo:
    checkpoint_id: str
    seq: int
    path: str
    created_at: float = 0.0
    parent: str = ""
    notes: str = ""
    model_version: str = ""
    reward_version: str = ""
    group_hashes: Dict[str, str] = field(default_factory=dict)
    rollback_of: str = ""


@dataclass
class LoadReport:
    loaded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)          # in the model, not in the checkpoint (e.g. new heads)
    unexpected: List[str] = field(default_factory=list)       # in the checkpoint, not in the model
    mismatched: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)   # name, saved, model

    @property
    def clean(self) -> bool:
        return not (self.missing or self.unexpected or self.mismatched)

    def summary(self) -> str:
        return (f"loaded {len(self.loaded)}, missing {len(self.missing)} {self.missing[:4]}, unexpected "
                f"{len(self.unexpected)} {self.unexpected[:4]}, mismatched {len(self.mismatched)} {self.mismatched[:2]}")


# ------------------------------------------------------------------------------------------------ weights
def tensor_hash(tensors: Dict[str, Tensor]) -> str:
    """sha256 over names, dtypes, shapes and raw bytes (order independent)."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def collect_weights(model: nn.Module, groups: Optional[Dict[str, Union[nn.Module, Iterable[nn.Parameter]]]] = None,
                    *, include_buffers: bool = True) -> Dict[str, Dict[str, Tensor]]:
    """Trainable tensors of ``model`` by group, on the CPU.

    * ``groups`` given: group name -> module or parameters (names are taken from ``model.named_parameters()``).
    * otherwise, a model with ``param_groups()`` (``nn.api.LatentModelAPI``) is grouped by it; the frozen backbone
      group and parameters without ``requires_grad`` are left out. A model with ``trainable_state()`` supplies the
      set of persistent tensors; those that are not parameters land in the ``buffers`` group.
    * otherwise everything trainable is one group ``"model"``.
    Persistent buffers of plain modules are saved in the ``buffers`` group when ``include_buffers``."""
    names = {id(p): n for n, p in model.named_parameters()}
    out: Dict[str, Dict[str, Tensor]] = {}

    def put(group: str, name: str, t: Tensor) -> None:
        out.setdefault(group, {})[name] = t.detach().cpu().contiguous().clone()

    if groups is None and hasattr(model, "param_groups"):
        raw = model.param_groups()
        groups = {g: ps for g, ps in raw.items() if g != FROZEN_GROUP}
    if groups is None:
        groups = {"model": list(model.parameters())}
    seen = set()
    for g, src in groups.items():
        params = src.parameters() if isinstance(src, nn.Module) else src
        for p in params:
            if id(p) in names and p.requires_grad and id(p) not in seen:
                put(g, names[id(p)], p)
                seen.add(id(p))
    grouped = {n for d in out.values() for n in d}
    if hasattr(model, "trainable_state"):
        for n, t in model.trainable_state().items():
            if n not in grouped:
                put(BUFFER_GROUP, n, t)
    elif include_buffers:
        persistent = set(model.state_dict().keys())
        for n, t in model.named_buffers():
            if n in persistent:
                put(BUFFER_GROUP, n, t)
    return out


def load_weights(model: nn.Module, weights: Dict[str, Dict[str, Tensor]], *, strict: bool = False) -> LoadReport:
    """Load grouped tensors into ``model`` by name. Tolerant by default: tensors the model has but the checkpoint
    lacks (newly added heads) keep their current values and are reported in ``missing``; tensors the model no longer
    has are reported in ``unexpected``; shape changes are reported in ``mismatched`` and not loaded. With
    ``strict=True`` any of those raises :class:`CheckpointMismatch`. A non-clean load also emits a warning."""
    flat: Dict[str, Tensor] = {}
    for d in weights.values():
        flat.update(d)
    target: Dict[str, Tensor] = {n: p for n, p in model.named_parameters() if p.requires_grad}
    persistent = set(model.state_dict().keys())
    buffers = {n: b for n, b in model.named_buffers() if n in persistent}
    rep = LoadReport()
    with torch.no_grad():
        for n, t in flat.items():
            dst = target.get(n, buffers.get(n))
            if dst is None:
                rep.unexpected.append(n)
            elif tuple(dst.shape) != tuple(t.shape):
                rep.mismatched.append((n, tuple(t.shape), tuple(dst.shape)))
            else:
                dst.copy_(t.to(device=dst.device, dtype=dst.dtype))
                rep.loaded.append(n)
    saved_buffers = BUFFER_GROUP in weights
    rep.missing = sorted(n for n in target if n not in flat) + \
        (sorted(n for n in buffers if n not in flat) if saved_buffers else [])
    if not rep.clean:
        if strict:
            raise CheckpointMismatch("checkpoint does not match the model: " + rep.summary(), rep)
        warnings.warn("tolerant checkpoint load: " + rep.summary(), stacklevel=2)
    return rep


# ------------------------------------------------------------------------------------------------ optimizers
def optimizer_state(optimizer: torch.optim.Optimizer, model: Union[nn.Module, Dict[str, nn.Parameter]]) -> Dict[str, Any]:
    """Optimizer state keyed by parameter *name*, so it survives a change of the parameter list."""
    named = dict(model.named_parameters()) if isinstance(model, nn.Module) else dict(model)
    names = {id(p): n for n, p in named.items()}
    groups, state = [], {}
    for g in optimizer.param_groups:
        hyper = {k: v for k, v in g.items() if k != "params" and isinstance(v, (int, float, bool, str, tuple, list, type(None)))}
        groups.append({"hyper": hyper, "params": [names.get(id(p), "") for p in g["params"]]})
        for p in g["params"]:
            if id(p) in names and p in optimizer.state:
                state[names[id(p)]] = {k: (v.detach().cpu().clone() if isinstance(v, Tensor) else v)
                                       for k, v in optimizer.state[p].items()}
    return {"class": type(optimizer).__name__, "groups": groups, "state": state}


def load_optimizer_state(optimizer: torch.optim.Optimizer, model: Union[nn.Module, Dict[str, nn.Parameter]],
                         saved: Dict[str, Any]) -> Dict[str, List[str]]:
    """Restore per-parameter optimizer state by name. Parameters without saved state (new heads) start fresh and
    are reported under ``fresh``; saved state of parameters that no longer exist under ``dropped``; hyperparameters
    are restored when the group structure is unchanged, otherwise the optimizer's own are kept (``hyper``)."""
    named = dict(model.named_parameters()) if isinstance(model, nn.Module) else dict(model)
    names = {id(p): n for n, p in named.items()}
    rep: Dict[str, List[str]] = {"restored": [], "fresh": [], "dropped": [], "mismatched": [], "hyper": []}
    if saved.get("class") and saved["class"] != type(optimizer).__name__:
        rep["hyper"].append(f"optimizer class changed {saved['class']} -> {type(optimizer).__name__}: state not restored")
        rep["fresh"] = [names.get(id(p), "") for g in optimizer.param_groups for p in g["params"]]
        return rep
    st = saved.get("state", {})
    known = {n for g in saved.get("groups", []) for n in g.get("params", [])}
    rep["unstepped"] = []
    present = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            n = names.get(id(p), "")
            present.add(n)
            s = st.get(n)
            if s is None:
                # no saved state: the parameter is new, or it had never been stepped when the checkpoint was taken.
                # Either way it must start clean (a rollback must not keep moments accumulated afterwards).
                optimizer.state.pop(p, None)
                (rep["unstepped"] if n in known else rep["fresh"]).append(n)
                continue
            if any(isinstance(v, Tensor) and v.dim() > 0 and tuple(v.shape) != tuple(p.shape) for v in s.values()):
                rep["mismatched"].append(n)
                continue
            optimizer.state[p] = {k: (v.clone().to(p.device) if isinstance(v, Tensor) else v) for k, v in s.items()}
            rep["restored"].append(n)
    rep["dropped"] = sorted(n for n in st if n not in present)
    sg = saved.get("groups", [])
    if len(sg) == len(optimizer.param_groups):
        for g, s in zip(optimizer.param_groups, sg):
            for k, v in s["hyper"].items():
                if k in g:
                    g[k] = tuple(v) if isinstance(g[k], tuple) and isinstance(v, list) else v
        rep["hyper"].append("restored")
    else:
        rep["hyper"].append(f"group structure changed ({len(sg)} -> {len(optimizer.param_groups)}): hyperparameters kept")
    if rep["fresh"] or rep["dropped"] or rep["mismatched"]:
        warnings.warn(f"tolerant optimizer load: fresh {rep['fresh'][:4]}, dropped {rep['dropped'][:4]}, "
                      f"mismatched {rep['mismatched'][:4]}", stacklevel=2)
    return rep


# ------------------------------------------------------------------------------------------------ RNG
def capture_rng(generators: Optional[Dict[str, torch.Generator]] = None) -> Dict[str, Any]:
    """States of python ``random``, numpy's global generator, torch CPU and every CUDA device, plus named
    ``torch.Generator`` objects."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    out: Dict[str, Any] = {
        "python": list(random.getstate()[1]) + [random.getstate()[2]],
        "python_version": random.getstate()[0],
        "numpy": {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached": float(cached)},
        "torch": torch.get_rng_state(),
        "cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "generators": {k: g.get_state() for k, g in (generators or {}).items()},
    }
    return out


def restore_rng(state: Dict[str, Any], generators: Optional[Dict[str, torch.Generator]] = None) -> None:
    if not state:
        return
    py = state["python"]
    random.setstate((state.get("python_version", 3), tuple(py[:-1]), py[-1]))
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    for k, g in (generators or {}).items():
        if k in state.get("generators", {}):
            g.set_state(state["generators"][k])


# ------------------------------------------------------------------------------------------------ files
def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj.copy())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_pt(obj: Any, path: str) -> None:
    with open(path, "wb") as fh:
        torch.save(_to_cpu(obj), fh)
        fh.flush()
        os.fsync(fh.fileno())


def _load_pt(path: str, map_location: Any = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=True)


def _write_json(obj: Any, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True).encode())
        fh.flush()
        os.fsync(fh.fileno())


def _seq_of(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_manifest(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(path, MANIFEST)) as fh:
            m = json.load(fh)
        return m if m.get("format") == FORMAT_VERSION and "files" in m else None
    except (OSError, ValueError):
        return None


def verify_checkpoint(path: str) -> List[str]:
    """Problems of a checkpoint directory ([] = complete and intact): manifest present, every file present with
    the recorded size and sha256."""
    m = _read_manifest(path)
    if m is None:
        return ["manifest missing or unreadable"]
    bad = []
    for rel, meta in m["files"].items():
        p = os.path.join(path, rel)
        if not os.path.isfile(p):
            bad.append(f"{rel}: missing")
        elif os.path.getsize(p) != meta["bytes"]:
            bad.append(f"{rel}: size {os.path.getsize(p)} != {meta['bytes']}")
        elif sha256_file(p) != meta["sha256"]:
            bad.append(f"{rel}: sha256 mismatch")
    return bad


def strip_optimizer_state(path: str) -> int:
    """Remove a checkpoint's optimizer state, keeping it a complete, verifiable checkpoint of everything else: the
    manifest is rewritten first (atomically) without the ``optim/`` files and with ``optimizer_state_pruned``, then
    the files are deleted (a crash in between leaves extra files, never a damaged checkpoint). Returns the bytes
    freed (0 when there was no optimizer state)."""
    m = _read_manifest(path)
    if m is None:
        return 0
    gone = {k: v for k, v in m["files"].items() if k.startswith("optim/")}
    if not gone:
        return 0
    m = dict(m)
    m["files"] = {k: v for k, v in m["files"].items() if not k.startswith("optim/")}
    m["optimizer_state_pruned"] = sorted(m.get("optimizers", []))
    m["optimizers"] = []
    atomic_write_bytes(os.path.join(path, MANIFEST), json.dumps(m, indent=1, sort_keys=True).encode())
    freed = 0
    for rel, meta in gone.items():
        f = os.path.join(path, rel)
        if os.path.isfile(f):
            freed += int(meta.get("bytes", 0))
            os.remove(f)
    if os.path.isdir(os.path.join(path, "optim")):
        fsync_dir(os.path.join(path, "optim"))
    return freed


def _info(path: str, m: Dict[str, Any]) -> CheckpointInfo:
    return CheckpointInfo(m["checkpoint_id"], int(m["seq"]), path, m.get("created_at", 0.0), m.get("parent", ""),
                          m.get("notes", ""), m.get("model_version", ""), m.get("reward_version", ""),
                          m.get("group_hashes", {}), m.get("rollback_of", ""))


def list_checkpoints(root: str) -> List[CheckpointInfo]:
    """Complete checkpoints (a readable manifest) under ``root``, oldest first. Contents are not re-hashed here."""
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.startswith("ckpt_") and os.path.isdir(p):
            m = _read_manifest(p)
            if m is not None:
                out.append(_info(p, m))
    return sorted(out, key=lambda i: i.seq)


def latest_valid(root: str, *, verify: bool = True) -> Optional[CheckpointInfo]:
    """The newest checkpoint that verifies; older ones are tried when the newest is damaged."""
    for info in reversed(list_checkpoints(root)):
        if not verify or not verify_checkpoint(info.path):
            return info
    return None


def recover(root: str) -> Dict[str, Any]:
    """Clean up after a crash: remove unfinished ``.tmp-*`` directories of dead processes and ``ckpt_*`` directories
    without a manifest, point LATEST at the newest intact checkpoint, and repair PROMOTED from the promotion log."""
    rep: Dict[str, Any] = {"removed": [], "corrupt": [], "latest": None, "promoted": None}
    if not os.path.isdir(root):
        return rep
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.startswith(".tmp-") and os.path.isdir(p):
            try:
                pid = int(name.split("-")[1])
            except (IndexError, ValueError):
                pid = -1
            if pid == os.getpid() or not _pid_alive(pid):
                shutil.rmtree(p, ignore_errors=True)
                rep["removed"].append(name)
        elif name.startswith(".trash-") and os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            rep["removed"].append(name)
        elif name.startswith("ckpt_") and os.path.isdir(p) and _read_manifest(p) is None:
            os.rename(p, os.path.join(root, "corrupt-" + name))        # kept for inspection, never loaded
            rep["corrupt"].append({"checkpoint_id": name, "problems": ["manifest missing or unreadable"]})
    for info in reversed(list_checkpoints(root)):
        bad = verify_checkpoint(info.path)
        if not bad:
            rep["latest"] = info.checkpoint_id
            break
        rep["corrupt"].append({"checkpoint_id": info.checkpoint_id, "problems": bad})
    if rep["latest"]:
        atomic_write_bytes(os.path.join(root, "LATEST"), rep["latest"].encode())
    existing = {i.checkpoint_id for i in list_checkpoints(root)}
    promoted = None
    for ev in _read_log(root):
        if ev.get("event") == "promote" and ev.get("promoted") and ev.get("checkpoint_id") in existing:
            promoted = ev["checkpoint_id"]
    if promoted:
        atomic_write_bytes(os.path.join(root, "PROMOTED"), promoted.encode())
        rep["promoted"] = promoted
    return rep


def _read_log(root: str) -> List[Dict[str, Any]]:
    p = os.path.join(root, "promotions.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                      # a torn last line after a crash
    return out


def _append_log(root: str, event: Dict[str, Any]) -> None:
    with open(os.path.join(root, "promotions.jsonl"), "a") as fh:
        fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------------------------------------ save / load
def _write_checkpoint(root: str, state: CheckpointState, *, seq: int, name: Optional[str], parent: str, notes: str,
                      clock: Optional[Clock], rollback_of: str) -> CheckpointInfo:
    """Write one checkpoint directory under ``root`` (temp directory -> fsync -> rename). ``name`` fixes the
    directory name; otherwise it is ``ckpt_<seq>_<content digest>``."""
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(os.path.join(tmp, "weights"))
    os.makedirs(os.path.join(tmp, "optim"))
    group_hashes, n_params = {}, {}
    for g, tensors in state.weights.items():
        if not tensors:
            continue
        cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        p = os.path.join(tmp, "weights", f"{g}.safetensors")
        save_file(cpu, p, metadata={"group": g})
        fsync_file(p)
        group_hashes[g] = tensor_hash(cpu)
        n_params[g] = int(sum(v.numel() for v in cpu.values()))
    _fault("weights")
    for oname, st in state.optimizers.items():
        _save_pt(st, os.path.join(tmp, "optim", f"{oname}.pt"))
    _save_pt({"fast_state": state.fast_state}, os.path.join(tmp, "fast_state.pt"))
    _save_pt({"lifetime_context": state.lifetime_context}, os.path.join(tmp, "lifetime.pt"))
    _save_pt(state.rng, os.path.join(tmp, "rng.pt"))
    if state.aux_state is not None:
        _save_pt({"aux_state": state.aux_state}, os.path.join(tmp, "aux.pt"))
    _write_json({"current_task": state.current_task, "replay_state": state.replay_state,
                 "pending_pointer": state.pending_pointer, "schedules": state.schedules,
                 "auth_config_hash": state.auth_config_hash, "model_version": state.model_version,
                 "reward_version": state.reward_version, "arch_signature": state.arch_signature,
                 "extra": state.extra, "experiments": state.experiments,
                 "pending_outcomes": state.pending_outcomes}, os.path.join(tmp, "state.json"))
    _fault("files")
    files = {}
    for d, _dirs, fs in os.walk(tmp):
        for f in fs:
            p = os.path.join(d, f)
            files[os.path.relpath(p, tmp)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_id = name or f"ckpt_{seq:06d}_{digest}"
    _write_json({"format": FORMAT_VERSION, "checkpoint_id": ckpt_id, "seq": seq,
                 "created_at": (clock or WallClock()).now(), "parent": parent, "notes": notes,
                 "rollback_of": rollback_of, "files": files, "group_hashes": group_hashes, "n_params": n_params,
                 "arch_signature": state.arch_signature, "model_version": state.model_version,
                 "reward_version": state.reward_version, "auth_config_hash": state.auth_config_hash,
                 "optimizers": sorted(state.optimizers)}, os.path.join(tmp, MANIFEST))
    fsync_tree(tmp)
    _fault("before_rename")
    final = os.path.join(root, ckpt_id)
    os.rename(tmp, final)
    fsync_dir(root)
    _fault("after_rename")
    atomic_write_bytes(os.path.join(root, "LATEST"), ckpt_id.encode())
    _fault("after_latest")
    return _info(final, _read_manifest(final))


def _next_seq(root: str) -> int:
    names = os.listdir(root) if os.path.isdir(root) else []
    return max([_seq_of(n) for n in names if n.startswith("ckpt_")] + [0]) + 1


def save_checkpoint(root: str, state: CheckpointState, *, parent: str = "", notes: str = "",
                    clock: Optional[Clock] = None, rollback_of: str = "") -> CheckpointInfo:
    """Write ``state`` as a new checkpoint ``<root>/ckpt_<seq>_<digest>`` atomically and return its info."""
    return _write_checkpoint(root, state, seq=_next_seq(root), name=None, parent=parent, notes=notes, clock=clock,
                             rollback_of=rollback_of)


def save_checkpoint_at(path: str, state: CheckpointState, *, parent: str = "", notes: str = "",
                       clock: Optional[Clock] = None) -> CheckpointInfo:
    """Write ``state`` atomically as the checkpoint directory ``path`` (the caller chooses the name, e.g. the
    runtime's ``checkpoints/ckpt_00000007``). An existing complete checkpoint is never overwritten."""
    path = os.path.abspath(path.rstrip("/"))
    if os.path.exists(path):
        raise CheckpointError(f"{path} already exists; checkpoints are immutable")
    root, name = os.path.dirname(path), os.path.basename(path)
    seq = _seq_of(name) if name.startswith("ckpt_") and _seq_of(name) > 0 else _next_seq(root)
    return _write_checkpoint(root, state, seq=seq, name=name, parent=parent, notes=notes, clock=clock, rollback_of="")


def _resolve(path_or_root: str, checkpoint_id: Optional[str], verify: bool) -> str:
    if os.path.isfile(os.path.join(path_or_root, MANIFEST)):
        path = path_or_root
    elif checkpoint_id:
        path = os.path.join(path_or_root, checkpoint_id)
    else:
        info = latest_valid(path_or_root, verify=verify)
        if info is None:
            raise CheckpointError(f"no intact checkpoint under {path_or_root}")
        return info.path
    if verify:
        bad = verify_checkpoint(path)
        if bad:
            raise CheckpointError(f"checkpoint {path} is damaged: {bad[:3]}")
    return path


def load_checkpoint(path_or_root: str, checkpoint_id: Optional[str] = None, *, map_location: Any = "cpu",
                    verify: bool = True) -> CheckpointState:
    """Load a checkpoint: a checkpoint directory, or a root (then ``checkpoint_id`` or the newest intact one)."""
    path = _resolve(path_or_root, checkpoint_id, verify)
    m = _read_manifest(path)
    if m is None:
        raise CheckpointError(f"{path} is not a complete checkpoint")
    with open(os.path.join(path, "state.json")) as fh:
        js = json.load(fh)
    st = CheckpointState(**{k: js[k] for k in ("current_task", "replay_state", "pending_pointer", "schedules",
                                               "auth_config_hash", "model_version", "reward_version",
                                               "arch_signature", "extra")})
    st.experiments = list(js.get("experiments") or [])            # absent in checkpoints written before DX
    st.pending_outcomes = list(js.get("pending_outcomes") or [])
    wdir = os.path.join(path, "weights")
    for f in sorted(os.listdir(wdir)):
        if f.endswith(".safetensors"):
            st.weights[f[:-len(".safetensors")]] = load_file(os.path.join(wdir, f), device=str(map_location))
    odir = os.path.join(path, "optim")
    for f in sorted(os.listdir(odir)) if os.path.isdir(odir) else []:
        if f.endswith(".pt"):
            st.optimizers[f[:-3]] = _load_pt(os.path.join(odir, f), map_location)
    st.fast_state = _load_pt(os.path.join(path, "fast_state.pt"), map_location)["fast_state"]
    st.lifetime_context = _load_pt(os.path.join(path, "lifetime.pt"), map_location)["lifetime_context"]
    st.rng = _load_pt(os.path.join(path, "rng.pt"), "cpu")
    if os.path.exists(os.path.join(path, "aux.pt")):
        st.aux_state = _load_pt(os.path.join(path, "aux.pt"), map_location)["aux_state"]
    st.extra = dict(st.extra or {})
    st.extra["checkpoint_id"] = m["checkpoint_id"]
    return st


def signature_differences(saved: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Keys whose values differ between two architecture signatures (reported on load, never ignored silently)."""
    return sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))


def capture_state(model: Optional[nn.Module] = None, optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None, *,
                  groups: Optional[Dict[str, Any]] = None, store: Any = None,
                  generators: Optional[Dict[str, torch.Generator]] = None, **fields: Any) -> CheckpointState:
    """Assemble a :class:`CheckpointState` from live objects: grouped weights and architecture signature of
    ``model``, name-keyed optimizer states, RNG states, and the replay sampler indices / pending-outcome pointer of
    an experience ``store``. Remaining fields (fast_state, lifetime_context, current_task, schedules,
    auth_config_hash, model_version, reward_version, extra) are passed by keyword."""
    st = CheckpointState(**fields)
    if model is not None:
        st.weights = collect_weights(model, groups)
        if not st.arch_signature and hasattr(model, "arch_signature"):
            st.arch_signature = dict(model.arch_signature())
        for name, opt in (optimizers or {}).items():
            st.optimizers[name] = optimizer_state(opt, model)
    if store is not None:
        st.replay_state = store.state_dict()
        st.pending_pointer = store.pending_pointer()
        if hasattr(store, "experiments"):
            st.experiments = [json.loads(json.dumps(e.to_dict(), default=str)) for e in store.experiments()]
        if hasattr(store, "pending_all"):
            st.pending_outcomes = [json.loads(json.dumps(q.to_dict(), default=str)) for q in store.pending_all()]
    st.rng = capture_rng(generators)
    return st


def _restore_store_records(state: CheckpointState, store: Any) -> Dict[str, Any]:
    """Experiment records of the checkpoint that the store lacks are written back (a record the store holds is
    never replaced: experience is not rolled back); pending queries are checked against the store."""
    from ..contracts import ExperimentRecord
    rep: Dict[str, Any] = {"experiments_in_checkpoint": len(state.experiments), "experiments_restored": [],
                           "pending_in_checkpoint": len(state.pending_outcomes), "pending_missing_in_store": [],
                           "pending_resolved_since": 0}
    if hasattr(store, "experiment") and hasattr(store, "put_experiment"):
        for d in state.experiments:
            if store.experiment(d.get("experiment_id", "")) is None:
                store.put_experiment(ExperimentRecord.from_dict(d))
                rep["experiments_restored"].append(d.get("experiment_id", ""))
    if hasattr(store, "pending_all") and hasattr(store, "query"):
        open_now = {q.query_id for q in store.pending_all()}
        for d in state.pending_outcomes:
            qid = d.get("query_id", "")
            if qid in open_now:
                continue
            if store.query(qid) is None:
                rep["pending_missing_in_store"].append(qid)
            else:
                rep["pending_resolved_since"] += 1
    return rep


def restore_state(state: CheckpointState, model: Optional[nn.Module] = None,
                  optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None, *, strict: bool = False, store: Any = None,
                  generators: Optional[Dict[str, torch.Generator]] = None, rng: bool = True,
                  expected_auth_hash: Optional[str] = None) -> Dict[str, Any]:
    """Apply a loaded state to live objects and return a report of everything that did not match exactly."""
    rep: Dict[str, Any] = {}
    if model is not None:
        if hasattr(model, "arch_signature"):
            rep["arch_signature_differences"] = signature_differences(state.arch_signature, dict(model.arch_signature()))
        rep["weights"] = load_weights(model, state.weights, strict=strict)
        rep["optimizers"] = {}
        for name, opt in (optimizers or {}).items():
            if name in state.optimizers:
                rep["optimizers"][name] = load_optimizer_state(opt, model, state.optimizers[name])
            else:
                rep["optimizers"][name] = {"restored": [], "fresh": ["<all>"], "dropped": [], "mismatched": [],
                                           "hyper": ["no saved state for this optimizer"]}
    if store is not None and state.replay_state:
        rep["replay"] = store.load_state_dict(state.replay_state)
    if store is not None and (state.experiments or state.pending_outcomes):
        rep["store_records"] = _restore_store_records(state, store)
    if rng:
        restore_rng(state.rng, generators)
    if expected_auth_hash is not None:
        rep["auth_config_changed"] = expected_auth_hash != state.auth_config_hash
    return rep


# ------------------------------------------------------------------------------------------------ weights-only
def export_weights_only(state: CheckpointState, out_dir: str, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write a transfer artefact: ``weights.safetensors`` (keys ``<group>::<name>``) and ``config.json`` with the
    architecture signature and fixed configuration. Nothing else from the checkpoint is exported."""
    parent = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(tmp)
    flat = {f"{g}{GROUP_SEP}{n}": t.detach().cpu().contiguous() for g, d in state.weights.items() for n, t in d.items()}
    wp = os.path.join(tmp, "weights.safetensors")
    save_file(flat, wp, metadata={"format": EXPORT_FORMAT})
    fsync_file(wp)
    cfg = {"format": EXPORT_FORMAT, "arch_signature": state.arch_signature, "model_version": state.model_version,
           "reward_version": state.reward_version, "config": config or {},
           "group_hashes": {g: tensor_hash(d) for g, d in state.weights.items() if d},
           "n_tensors": len(flat), "weights_sha256": sha256_file(wp)}
    _write_json(cfg, os.path.join(tmp, "config.json"))
    fsync_tree(tmp)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    fsync_dir(parent)
    return cfg


def load_weights_only(path: str, *, map_location: Any = "cpu") -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    """Read a weights-only export: (group -> name -> tensor, config). Verifies the recorded hash."""
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    wp = os.path.join(path, "weights.safetensors")
    if cfg.get("format") != EXPORT_FORMAT or sha256_file(wp) != cfg.get("weights_sha256"):
        raise CheckpointError(f"{path} is not an intact weights-only export")
    out: Dict[str, Dict[str, Tensor]] = {}
    for k, t in load_file(wp, device=str(map_location)).items():
        g, n = k.split(GROUP_SEP, 1)
        out.setdefault(g, {})[n] = t
    return out, cfg


def export_weights(checkpoint_path: str, out_dir: str, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Weights-only export of a checkpoint on disk (a checkpoint directory, or a root: its newest intact one)."""
    return export_weights_only(load_checkpoint(checkpoint_path), out_dir, config=config)


def export_keys(path: str) -> List[str]:
    """Tensor names inside a weights-only export (without loading the tensors)."""
    with safe_open(os.path.join(path, "weights.safetensors"), framework="pt") as f:
        return sorted(f.keys())


# ------------------------------------------------------------------------------------------------ manager
@dataclass
class PromotionGates:
    """A checkpoint is promoted only on independent evidence. All metrics are oriented higher = better.

    primary               validation metric that must improve by ``min_validation_gain`` over the promoted
                          checkpoint ('' = no single primary metric)
    max_validation_drop   tolerated drop of any other validation metric
    max_retention_drop    tolerated drop of any retention metric (measured forgetting)
    min_validation        absolute floors, also applied to the first promotion
    """
    primary: str = ""
    min_validation_gain: float = 0.0
    max_validation_drop: float = 0.0
    max_retention_drop: float = 0.02
    min_validation: Dict[str, float] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    promoted: bool
    checkpoint_id: str
    baseline_id: str = ""
    reasons: List[str] = field(default_factory=list)
    validation: Dict[str, float] = field(default_factory=dict)
    retention: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    #: what held-out evidence existed (``promote_learner``: held-out records per provider and whether any provider
    #: produced a held-out loss); ``{}`` when the caller supplied its own validation metrics
    held_out: Dict[str, Any] = field(default_factory=dict)


#: provenance of a metric that can justify a promotion (independent of the agent's own reward)
INDEPENDENT_PROVENANCE = ("heldout_validation", "retention", "verified_outcome", "independent_evaluator",
                          "randomized_experiment", "declared_validation", "declared_retention")
#: provenance of the agent's own reward signals: never evidence for a promotion
INTERNAL_PROVENANCE = ("internal_reward", "intrinsic_reward", "reward_model", "meta_reward", "agent_reward",
                       "self_assessment")


def _internal(name: str) -> bool:
    """Name heuristic, used only for metrics whose provenance was not declared (fail-closed safety net)."""
    n = name.lower()
    return "intrinsic" in n or "internal_reward" in n or n in ("reward", "r_psi")


def metric_value(v: Any) -> float:
    """A metric is a number, a ``(value, provenance)`` pair or ``{"value": v, "provenance": p}``."""
    if isinstance(v, dict):
        return float(v["value"])
    if isinstance(v, (tuple, list)):
        return float(v[0])
    return float(v)


def metric_provenance(name: str, v: Any, declared: Optional[Dict[str, str]], default: str) -> str:
    if isinstance(v, dict) and v.get("provenance"):
        return str(v["provenance"])
    if isinstance(v, (tuple, list)) and len(v) > 1 and v[1]:
        return str(v[1])
    if declared and name in declared:
        return str(declared[name])
    return "internal_reward" if _internal(name) else default


def provider_metric_provenance(provider: Any) -> str:
    """Provenance of a provider's held-out loss: ``provider.metric_provenance`` if it declares one (a provider whose
    target is the agent's own reward declares "internal_reward"), else held-out validation on measured outcomes."""
    return str(getattr(provider, "metric_provenance", "") or "heldout_validation")


class CheckpointManager:
    """Checkpoint root with retention, promotion gated by validation and retention metrics, and rollback."""

    def __init__(self, root: str, *, keep_last: int = 3, keep_promoted: int = 5, gates: Optional[PromotionGates] = None,
                 clock: Optional[Clock] = None, keep_optim: int = 1):
        self.root = os.path.abspath(root)
        self.keep_last, self.keep_promoted = max(2, int(keep_last)), max(1, int(keep_promoted))
        #: kept checkpoints that keep their optimizer state: the newest ``keep_optim`` (cam_live.storage's retention
        #: policy: optimizer state only in the newest checkpoint of a group; weights in every kept one)
        self.keep_optim = max(0, int(keep_optim))
        self.gates = gates or PromotionGates()
        self.clock = clock or WallClock()
        os.makedirs(self.root, exist_ok=True)
        self.recovery = recover(self.root)

    # ---- basic operations --------------------------------------------------------------------------------
    def save(self, state: CheckpointState, *, notes: str = "", parent: Optional[str] = None) -> CheckpointInfo:
        if parent is None:
            last = self.latest(verify=False)
            parent = last.checkpoint_id if last else ""
        info = save_checkpoint(self.root, state, parent=parent, notes=notes, clock=self.clock)
        self.prune()
        return info

    def list(self) -> List[CheckpointInfo]:
        return list_checkpoints(self.root)

    def latest(self, *, verify: bool = True) -> Optional[CheckpointInfo]:
        return latest_valid(self.root, verify=verify)

    def get(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        for i in self.list():
            if i.checkpoint_id == checkpoint_id:
                return i
        return None

    def load(self, checkpoint_id: Optional[str] = None, *, map_location: Any = "cpu", verify: bool = True
             ) -> CheckpointState:
        return load_checkpoint(self.root, checkpoint_id, map_location=map_location, verify=verify)

    # ---- promotion ---------------------------------------------------------------------------------------
    def promotion_log(self) -> List[Dict[str, Any]]:
        return _read_log(self.root)

    def promoted_ids(self) -> List[str]:
        existing = {i.checkpoint_id for i in self.list()}
        out = []
        for ev in self.promotion_log():
            if ev.get("event") == "promote" and ev.get("promoted") and ev["checkpoint_id"] in existing:
                out.append(ev["checkpoint_id"])
        return out

    def promoted(self) -> Optional[CheckpointInfo]:
        ids = self.promoted_ids()
        return self.get(ids[-1]) if ids else None

    def _baseline_metrics(self, checkpoint_id: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        for ev in reversed(self.promotion_log()):
            if ev.get("event") == "promote" and ev.get("promoted") and ev.get("checkpoint_id") == checkpoint_id:
                return ev.get("validation", {}), ev.get("retention", {})
        return {}, {}

    def promote(self, checkpoint_id: str, *, validation: Dict[str, Any], retention: Dict[str, Any],
                notes: str = "", provenance: Optional[Dict[str, str]] = None,
                held_out: Optional[Dict[str, Any]] = None) -> PromotionDecision:
        """Decide whether ``checkpoint_id`` becomes the promoted checkpoint.

        ``validation``: independent validation metrics of this checkpoint; ``retention``: its performance on
        previously acquired capabilities. Both are required and must be finite. A metric is a number or carries its
        provenance (``(value, provenance)``, ``{"value", "provenance"}``, or ``provenance[name]``); only metrics whose
        provenance is in :data:`INDEPENDENT_PROVENANCE` count. The agent's own reward (:data:`INTERNAL_PROVENANCE`)
        and unknown provenances are ignored, whatever the metric is called. A plain number is the caller's
        declaration ("declared_validation" / "declared_retention"), except that an undeclared metric named like the
        internal reward is ignored as well. The decision, the numbers and each metric's provenance are logged.
        ``held_out`` (from :func:`promote_learner`) states the held-out evidence behind ``validation``; when it says
        there is none (``evidence`` False) the refusal names it explicitly (fail closed, as the slow-step guard)."""
        g = self.gates
        prov_used: Dict[str, str] = {}
        ignored: List[str] = []
        val: Dict[str, float] = {}
        ret: Dict[str, float] = {}
        for src, dst, default in ((validation, val, "declared_validation"), (retention, ret, "declared_retention")):
            for k, v in src.items():
                pv = metric_provenance(k, v, provenance, default)
                if pv not in INDEPENDENT_PROVENANCE:
                    ignored.append(k)
                    prov_used[k] = pv
                    continue
                dst[k] = metric_value(v)
                prov_used[k] = pv
        ignored = sorted(ignored)
        base = self.promoted()
        dec = PromotionDecision(False, checkpoint_id, base.checkpoint_id if base else "", validation=val, retention=ret,
                                ignored=ignored, held_out=dict(held_out or {}))
        if held_out is not None and not held_out.get("evidence", True):
            dec.reasons.append("no held-out evidence: " + str(held_out.get("why") or "no provider has a held-out loss"))
        info = self.get(checkpoint_id)
        if info is None:
            dec.reasons.append("checkpoint not found")
        elif verify_checkpoint(info.path):
            dec.reasons.append("checkpoint is damaged")
        if not val:
            dec.reasons.append("no independent validation metric supplied")
        if not ret:
            dec.reasons.append("no retention metric supplied")
        for k, v in {**val, **ret}.items():
            if not math.isfinite(v):
                dec.reasons.append(f"metric {k} is not finite")
        if g.primary and val and g.primary not in val:
            dec.reasons.append(f"primary validation metric {g.primary} not measured")
        for k, floor in g.min_validation.items():
            if k not in val:
                dec.reasons.append(f"validation metric {k} not measured")
            elif val[k] < floor:
                dec.reasons.append(f"validation {k} {val[k]:.6g} below the floor {floor:.6g}")
        if base is not None and not dec.reasons:
            bval, bret = self._baseline_metrics(base.checkpoint_id)
            for k, old in bret.items():
                if k not in ret:
                    dec.reasons.append(f"retention metric {k} not measured")
                    continue
                dec.deltas["retention." + k] = ret[k] - old
                if ret[k] < old - g.max_retention_drop:
                    dec.reasons.append(f"retention {k} dropped {old - ret[k]:.6g} (> {g.max_retention_drop:.6g}): forgetting")
            for k, old in bval.items():
                if k not in val:
                    dec.reasons.append(f"validation metric {k} not measured")
                    continue
                dec.deltas["validation." + k] = val[k] - old
                if k == g.primary:
                    if val[k] < old + g.min_validation_gain:
                        dec.reasons.append(f"primary validation {k} gained {val[k] - old:.6g} (< {g.min_validation_gain:.6g})")
                elif val[k] < old - g.max_validation_drop:
                    dec.reasons.append(f"validation {k} dropped {old - val[k]:.6g} (> {g.max_validation_drop:.6g})")
        dec.promoted = not dec.reasons
        _append_log(self.root, {"event": "promote", "ts": self.clock.now(), "checkpoint_id": checkpoint_id,
                                "promoted": dec.promoted, "baseline_id": dec.baseline_id, "validation": val,
                                "retention": ret, "deltas": dec.deltas, "reasons": dec.reasons, "ignored": ignored,
                                "provenance": prov_used, "held_out": dec.held_out, "notes": notes})
        if dec.promoted:
            atomic_write_bytes(os.path.join(self.root, "PROMOTED"), checkpoint_id.encode())
            self.prune()
        return dec

    def rollback(self, reason: str = "") -> Tuple[CheckpointInfo, CheckpointState]:
        """Return to the last promoted checkpoint: its content becomes a new latest checkpoint (so a restart
        resumes from it) while the rejected checkpoints stay on disk for comparison until pruned."""
        base = self.promoted()
        if base is None:
            raise CheckpointError("no promoted checkpoint to roll back to")
        bad = verify_checkpoint(base.path)
        if bad:
            raise CheckpointError(f"promoted checkpoint {base.checkpoint_id} is damaged: {bad[:3]}")
        latest = self.latest(verify=False)
        state = load_checkpoint(base.path)
        state.extra = {k: v for k, v in state.extra.items() if k != "checkpoint_id"}
        info = save_checkpoint(self.root, state, parent=base.checkpoint_id, notes=f"rollback: {reason}", clock=self.clock,
                               rollback_of=latest.checkpoint_id if latest else "")
        _append_log(self.root, {"event": "rollback", "ts": self.clock.now(), "to": base.checkpoint_id,
                                "from": latest.checkpoint_id if latest else "", "new_checkpoint": info.checkpoint_id,
                                "reason": reason})
        self.prune()
        return info, state

    # ---- retention of checkpoints on disk ------------------------------------------------------------------
    def prune(self) -> List[str]:
        """Delete old checkpoints, keeping the newest ``keep_last``, the newest ``keep_promoted`` promoted ones and
        the rollback targets. Returns the removed ids."""
        infos = self.list()
        keep = {i.checkpoint_id for i in infos[-self.keep_last:]}
        keep |= set(self.promoted_ids()[-self.keep_promoted:])
        removed = []
        for i in infos:
            if i.checkpoint_id not in keep:
                trash = os.path.join(self.root, f".trash-{i.checkpoint_id}")
                os.rename(i.path, trash)
                shutil.rmtree(trash, ignore_errors=True)
                removed.append(i.checkpoint_id)
        if removed:
            fsync_dir(self.root)
        kept = [i for i in infos if i.checkpoint_id in keep]
        for i in kept[:max(0, len(kept) - self.keep_optim)]:
            strip_optimizer_state(i.path)
        return removed


# ------------------------------------------------------------------------------------------------ complete learner state
#: TaskSpec fields that make up the authorization configuration as the learner receives it (``runtime.tasks.
#: learner_view``): what it may use, where, how much, under which data permissions and stop conditions, in which mode
AUTH_FIELDS = ("resources", "interfaces", "limits", "data_permissions", "stop_conditions", "mode", "origin")


def auth_config_hash(task: Any) -> str:
    """``"sha256:<hex>"`` of the authorization-relevant fields of a task (a ``TaskSpec`` or its dict; ``""`` for
    None), for ``capture_learner(auth_config_hash=...)`` and ``restore_learner(expected_auth_hash=...)`` (CL-10). The
    authorization service's own records (operator authorization, credentials) are never visible to the learner and
    are not part of it."""
    if task is None:
        return ""
    d = task.to_dict() if hasattr(task, "to_dict") else dict(task)
    part = {k: d.get(k) for k in AUTH_FIELDS}
    return "sha256:" + hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()


def capture_learner(learner: Any, *, extra_optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                    generators: Optional[Dict[str, torch.Generator]] = None, lifetime_context: Any = None,
                    current_task: Optional[Dict[str, Any]] = None, auth_config_hash: str = "", reward_version: str = "",
                    extra: Optional[Dict[str, Any]] = None) -> CheckpointState:
    """Everything a restart of the learning process needs, from a ``credit.learner.ContinualLearner`` on the real
    model: all trainable groups (experiential, heads, semantic adapters, reward), the learner's optimizers (plus
    ``extra_optimizers``, e.g. the reward workstream's meta optimizer), the fast belief state, the lifetime context,
    replay sampler indices and pending-outcome pointer, learning schedules and counters, RNG states, versions, the
    architecture signature, the configuration, the credit selector and the tensor state of loss providers."""
    opts = dict(learner.optimizers)
    for k, o in (extra_optimizers or {}).items():
        if k in opts:
            raise ValueError(f"optimizer name {k!r} is already used by the learner")
        opts[k] = o
    credit = getattr(learner, "credit", None)
    if credit is not None and getattr(credit, "evaluates_experiments", False):
        try:                                   # the caller's experiment registry is in the store, hence checkpointed
            credit.record_experiments()
        except Exception as e:  # noqa: BLE001 — a checkpoint is never lost over the copy; the store keeps the last one
            warnings.warn(f"experiment records not copied before the checkpoint: {type(e).__name__}: {e}")
    fast = learner.fast_state.state_dict() if getattr(learner, "fast_state", None) is not None else None
    aux = {"providers": {p.name: p.state_dict() for p in learner.providers if hasattr(p, "state_dict")}}
    ex = dict(extra or {})
    ex["config"] = learner.cfg.to_dict()
    if getattr(learner, "credit", None) is not None:
        ex["credit"] = learner.credit.state_dict()
    return capture_state(learner.model, opts, store=learner.store, generators=generators, fast_state=fast,
                         lifetime_context=lifetime_context, current_task=current_task, schedules=learner.state_dict(),
                         auth_config_hash=auth_config_hash, model_version=learner.model_version,
                         reward_version=reward_version, extra=json.loads(json.dumps(ex, default=str)), aux_state=aux)


def restore_learner(learner: Any, state: CheckpointState, *, strict: bool = False,
                    extra_optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                    generators: Optional[Dict[str, torch.Generator]] = None, rng: bool = True,
                    expected_auth_hash: Optional[str] = None) -> Dict[str, Any]:
    """Apply a loaded checkpoint to a freshly built learner (same configuration, same providers registered).
    Returns the report of everything that did not match exactly. RNG states are restored last."""
    learner._sync_optimizers()
    opts = dict(learner.optimizers)
    opts.update(extra_optimizers or {})
    rep = restore_state(state, learner.model, opts, strict=strict, store=learner.store, generators=generators,
                        rng=False, expected_auth_hash=expected_auth_hash)
    if hasattr(learner.model, "mark_adapters_updated"):
        learner.model.mark_adapters_updated()
    rep["learner"] = learner.load_state_dict(state.schedules or {})
    if getattr(learner, "fast_state", None) is not None and state.fast_state is not None:
        learner.fast_state.load_state_dict(state.fast_state)
        rep["fast_state"] = "restored"
    saved = (state.aux_state or {}).get("providers", {}) if isinstance(state.aux_state, dict) else {}
    rep["provider_state"] = {"restored": [], "missing": []}
    for p in learner.providers:
        if hasattr(p, "load_state_dict"):
            if p.name in saved:
                p.load_state_dict(saved[p.name])
                rep["provider_state"]["restored"].append(p.name)
            else:
                rep["provider_state"]["missing"].append(p.name)
    if getattr(learner, "credit", None) is not None and "credit" in (state.extra or {}):
        learner.credit.load_state_dict(state.extra["credit"])
    if rng:
        restore_rng(state.rng, generators)
    return rep


def save_learner(target: Union[str, "CheckpointManager"], learner: Any, *, notes: str = "", **capture_kw: Any
                 ) -> CheckpointInfo:
    """Atomic checkpoint of the complete learner state into a checkpoint root (path or manager)."""
    state = capture_learner(learner, **capture_kw)
    if isinstance(target, CheckpointManager):
        return target.save(state, notes=notes)
    return save_checkpoint(target, state, notes=notes, clock=getattr(learner, "clock", None))


def load_learner(source: Union[str, "CheckpointManager"], learner: Any, *, checkpoint_id: Optional[str] = None,
                 strict: bool = False, map_location: Any = "cpu", **restore_kw: Any) -> Dict[str, Any]:
    """Resume: load a checkpoint (default: the newest intact one under the root) into ``learner``."""
    root = source.root if isinstance(source, CheckpointManager) else source
    state = load_checkpoint(root, checkpoint_id, map_location=map_location)
    rep = restore_learner(learner, state, strict=strict, **restore_kw)
    rep["checkpoint_id"] = state.extra.get("checkpoint_id", "")
    return rep


def promote_learner(manager: "CheckpointManager", learner: Any, *, retention: Dict[str, float],
                    validation: Optional[Dict[str, float]] = None, notes: str = "", rollback: bool = True,
                    **capture_kw: Any) -> Tuple[PromotionDecision, CheckpointInfo, Optional[Dict[str, Any]]]:
    """Checkpoint the learner and ask for promotion on independent evidence: ``validation`` (default: the negated
    held-out losses of ``learner.validate()``) and ``retention`` (``retention.retention_metrics`` of a measured
    forgetting report). When promotion is refused and ``rollback`` is set, the learner is returned to the last
    promoted checkpoint (weights, optimizers, schedules, fast state); experience is never rolled back.
    Returns (decision, the checkpoint that was judged, rollback report or None)."""
    provenance: Dict[str, str] = {}
    held_out: Optional[Dict[str, Any]] = None
    if validation is None:
        validation = {}
        names = {p.name: p for p in getattr(learner, "providers", [])}
        losses = learner.validate()
        for k, v in losses.items():
            validation[f"heldout/neg_{k}"] = -float(v)
            provenance[f"heldout/neg_{k}"] = provider_metric_provenance(names.get(k))
        store = getattr(learner, "store", None)
        counts: Dict[str, int] = {}
        if store is not None and hasattr(store, "count"):
            for name in names:
                try:
                    counts[name] = int(store.count(loss="val:" + name))
                except Exception:  # noqa: BLE001 — a store without eligibility counts: reported as unknown
                    counts[name] = -1
        n_rec = int(store.count()) if store is not None and hasattr(store, "count") else -1
        held_out = {"evidence": bool(losses), "validation_records": counts, "providers_with_loss": sorted(losses),
                    "records": n_rec}
        if not losses:
            held_out["why"] = (f"no provider has a held-out loss ({sum(c for c in counts.values() if c > 0)} held-out "
                               f"provider-records among {n_rec} records)")
    info = save_learner(manager, learner, notes=notes, **capture_kw)
    dec = manager.promote(info.checkpoint_id, validation=validation, retention=retention, notes=notes,
                          provenance=provenance, held_out=held_out)
    rb = None
    if dec.promoted and hasattr(learner, "set_slow_reference"):
        learner.set_slow_reference(info.checkpoint_id)       # the slow-update drift bound starts again here
    if not dec.promoted and rollback and manager.promoted() is not None:
        _info, state = manager.rollback("; ".join(dec.reasons)[:500])
        rb = restore_learner(learner, state)
        rb["rolled_back_to"] = dec.baseline_id
        # a promoted checkpoint that is not the newest keeps no optimizer state (retention policy): the moments of
        # the refused trajectory do not belong to the restored weights, so the learner's optimizers start fresh
        reset = []
        for name, opt in getattr(learner, "optimizers", {}).items():
            if name not in state.optimizers:
                opt.state.clear()
                reset.append(name)
        rb["optimizer_state_reset"] = reset
        if hasattr(learner, "set_slow_reference"):
            learner.set_slow_reference(dec.baseline_id)
    return dec, info, rb


# ------------------------------------------------------------------------------------------------ weights-only transfer
@dataclass
class ExportedModel:
    model: nn.Module
    fast_state: Any                      # a fresh fast belief state at the exported learned prior (no evidence)
    report: LoadReport
    config: Dict[str, Any]


def export_learner_weights(learner: Any, out_dir: str) -> Dict[str, Any]:
    """Weights-only export of the live learner's model with the configuration needed to rebuild it."""
    state = CheckpointState(weights=collect_weights(learner.model), arch_signature=dict(learner.model.arch_signature()),
                            model_version=learner.model_version)
    return export_weights_only(state, out_dir, config=learner.cfg.to_dict())


def load_exported_model(export_dir: str, *, cfg: Any = None, register_heads: Optional[Callable[[nn.Module], Any]] = None,
                        strict: bool = True, device: Optional[str] = None) -> ExportedModel:
    """Build a *fresh* model from configuration plus a weights-only export: no experience store, no optimizer, no
    task, and fast beliefs reset to the exported learned prior.

    ``cfg`` defaults to the configuration stored in the export. ``register_heads(model)`` must register the same
    heads the exporting process had (head modules are code, not data); with ``strict`` any tensor that does not
    find its place raises :class:`CheckpointMismatch`."""
    from ..config import CamLiveConfig
    from ..nn.model import build_model
    weights, meta = load_weights_only(export_dir)
    if cfg is None:
        if not meta.get("config"):
            raise CheckpointError(f"{export_dir} carries no configuration; pass cfg")
        cfg = CamLiveConfig.from_dict(meta["config"])
    if device is not None:
        cfg.device = device
    model = build_model(cfg)
    if register_heads is not None:
        register_heads(model)
    rep = load_weights(model, weights, strict=strict)
    if hasattr(model, "mark_adapters_updated"):
        model.mark_adapters_updated()
    diff = signature_differences(meta.get("arch_signature", {}),
                                 json.loads(json.dumps(dict(model.arch_signature()), default=str)))
    if strict and diff:
        raise CheckpointMismatch(f"architecture signature differs in {diff}", rep)
    fast = model.new_fast_state() if hasattr(model, "new_fast_state") else None
    return ExportedModel(model, fast, rep, meta)
