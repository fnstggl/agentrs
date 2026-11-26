"""Evaluation protocols: how learners and worlds become measurements.

The question every protocol here serves: does learning system A acquire the ability to succeed in unfamiliar worlds
faster than learning system B, per unit of experience and of compute - and how much of what it learned is in its
weights? The answer is always a set of curves and paired differences with intervals, never one number.

A *learner* is anything implementing `research.learner.Learner`; a *learner factory* is a callable
`factory(seed) -> Learner` (see `from_registry`). A *task* is a `research.tasks.Task`: a hidden mechanism with a
public objective and a private reference, so that every episode has a normalised `score` (0 = left the defaults
alone, 1 = the reference policy) and a `success` flag (public objective reached).

Protocols (each returns a JSON-serialisable dict whose `"protocol"` names it; `save_result` writes it):

    evaluate         frozen evaluation: one episode per task, weights frozen, context reset, paired by task
    baseline         frozen evaluation of several learners on identical tasks, paired statistics
                     (`evaluate_many` for learners as they are, `combine_evaluations` for results already made)
    lifetime         a learning curve: training interaction alternating with frozen evaluations, every point
                     with its full cost, checkpointed and resumable
    timescales       fast adaptation, parametric learning and meta-learning, measured separately
    internalisation  the eight-step parametric internalisation protocol, with retention
    transfer         zero-shot and adaptation transfer over the four categories, transfer efficiency
    compare          two or more learners under equal budgets, paired and corrected, with a failure diagnosis

What the harness guarantees, and how:

* **Frozen means frozen.** `evaluate` calls `freeze(True)`, never calls `observe` or `update`, restores the flag
  afterwards, and hashes the learner's checkpoint before and after (`learner_hash`). A learner whose long-term
  state changed during evaluation is reported (`frozen.unchanged = False`), and makes an internalisation result
  invalid. The hash covers everything `save` writes except entries stored under a key named in `VOLATILE_KEYS`
  (accounting and random-generator state); a learner may instead define `state_hash() -> str`.
* **Fresh means fresh.** Protocols that claim generalisation check the identity of every training task actually
  consumed against the evaluation tasks, by task id *and* by a fingerprint of the hidden mechanism (two task ids
  can name the same mechanism), and refuse an overlap (`ProtocolViolation`).
* **Paired means paired.** Learners are compared on identical tasks in identical order; training streams are
  consumed in a schedule that does not depend on the learner, so equal environment steps mean equal tasks.
* **Costs are counted here**, by `research.accounting`, not taken from the learner's word.

Training interaction. Tasks are drawn from a `TaskSource` and run in batches of exactly `n_envs` tasks with equal
horizon (tasks with another horizon wait in their own bucket), so every `observe` call carries `n_envs` live
transitions and the batch size never changes. `update()` is called after every `update_every` ticks; the learner
decides whether anything is pending. Frozen evaluations happen only at batch boundaries, where no training episode
is in flight, because the contract gives a learner one context and evaluation resets it.

The within-episode profile. Every episode record carries `segment_scores`: the task's normalisation applied to the
mean net of each of `segments` successive stretches of the episode, `adaptation = last - first`, `slope` (least
squares over the segments) and `rise = mean - first` (how early the improvement comes).

Failure diagnosis (`diagnose_episode`). What an episode record can establish, tested in this order:

    succeeded        the public objective was reached
    never_touched    the settings never differed from where they started (within 1e-5 of the lever range)
    went_broke       it moved levers, and in at least `broke_frac` (0.1) of the ticks it could not pay: the tick ran
                     at fallback settings or idle
    found_and_lost   some earlier segment ran at or above the objective rate (`good`, the score rate the public
                     objective needs), the last one did not, and the drop is at least `margin` (0.1); the record
                     says whether the learner's own settings moved after the peak segment (by more than
                     `shift_tol`, 0.05 of a lever's range, from where they stood at its end) or were held (then
                     the world changed under them)
    found_late       the last segment ran at or above the objective rate, but the episode as a whole fell short
    never_found      no segment reached the objective rate
    not_determinable the task has no private reference (unvalidated), so nothing can be said about quality

"No better than the scripted experimenter" is a statement about a learner, not an episode: it is the paired
difference to a named reference learner on the same tasks (`compare(..., reference=...)`).

Reward mode (Crucible 5.0, Clarification 2): every entry point takes ``reward_mode`` (default ``observation_only``),
passed to the environments it builds. A learner that reads the environment's reward receives 0.0 in that mode and
must derive its own signal from the observations; the 4.0 protocol results are reproduced with an explicitly named
``reward_mode="legacy_shaped_reward"`` (OO.8). The operator-side episode records the protocols score are computed
on the evaluator side and are the same in every mode.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import shutil
import tempfile
import time
import zipfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..core.canonical import digest
from . import stats as S
from .accounting import COST_COLUMNS, Cost, Meter, charge_episodes, cost_row, params_of
from .envs import MechVecEnv, money_scale
from .learner import EnvSpec
from .tasks import GOAL_FRACTION, Task

SCHEMA = "crucible.research.protocols/1"
VOLATILE_KEYS = frozenset({"usage", "_usage", "rng", "rng_state", "random_state", "generator"})
CATEGORIES = ("within", "compositional", "structural", "cross_impl")
FAILURE_CLASSES = ("never_touched", "went_broke", "found_and_lost", "found_late", "never_found", "not_determinable")
_TOL = 1e-5
N_BOOT = 2000


class ProtocolViolation(ValueError):
    """The conditions a protocol's claim rests on do not hold (overlapping tasks, unequal task sets, ...)."""


# ====================================================================== small helpers
def _js(x: Any) -> Any:
    """Plain JSON: numpy out, non-finite floats to None, keys to str."""
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return _js(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def save_result(result: dict, path: str) -> str:
    """Write a protocol result as JSON (atomically). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(_js(result), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)
    return path


def load_result(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _f(xs: Iterable) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in xs], dtype=float)


def versions() -> dict:
    """What a result was measured under. A change of the task definition (`tasks.TASK_VERSION`) changes money
    scales, endowments or references of some tasks, so results across versions are not paired, and a `lifetime`
    run is never resumed across one."""
    from ..mechanisms.base import CONTRACT_VERSION
    from .tasks import TASK_VERSION
    return {"task": TASK_VERSION, "mech_contract": CONTRACT_VERSION, "protocols": SCHEMA}


def _ident(learner) -> dict:
    return {"name": str(getattr(learner, "name", type(learner).__name__)), "kind": str(getattr(learner, "kind", "")),
            "class": type(learner).__name__, "seed": int(getattr(learner, "seed", 0)), "params": params_of(learner)}


def from_registry(name: str, spec: EnvSpec | None = None, device: str = "cpu", **kw) -> Callable[[int], Any]:
    """A learner factory `seed -> Learner` resolved through `crucible.research.learners.make`."""
    spec = spec or EnvSpec()

    def factory(seed: int):
        from . import learners as registry
        return registry.make(name, spec, int(seed), device, **kw)
    factory.__name__ = f"registry:{name}"
    return factory


def _workdir(workdir: str | None, *parts: str) -> tuple[str, bool]:
    """(directory, temporary?)"""
    if workdir is None:
        return tempfile.mkdtemp(prefix="crucible-protocol-"), True
    d = os.path.join(workdir, *parts)
    os.makedirs(d, exist_ok=True)
    return d, False


# ====================================================================== task identity
def mech_fingerprint(task: Task) -> str:
    """A digest of the task's hidden mechanism (not of its noise seed): two tasks with the same fingerprint pose
    the same causal problem, whatever their ids."""
    return digest(task.mechanism()["config"])


def task_identities(tasks: Iterable[Task]) -> tuple[set[str], set[str]]:
    ids, fps = set(), set()
    for t in tasks:
        ids.add(t.id)
        fps.add(mech_fingerprint(t))
    return ids, fps


def overlap(train_tasks: Iterable[Task], eval_tasks: Iterable[Task]) -> list[str]:
    """Ids of training tasks that are also evaluation tasks, by id or by mechanism."""
    ids, fps = task_identities(eval_tasks)
    out = []
    for t in train_tasks:
        if t.id in ids or mech_fingerprint(t) in fps:
            out.append(t.id)
    return sorted(set(out))


class _Watch:
    """Sees every training task actually consumed: keeps a running digest of their ids and checks each against
    the evaluation sets it guards."""

    def __init__(self, eval_sets: Mapping[str, Sequence[Task]] | None = None, strict: bool = False):
        self.strict = strict
        self.sets = {}
        for name, ts in (eval_sets or {}).items():
            self.sets[name] = task_identities(ts)
        self.overlaps: dict[str, list[str]] = {name: [] for name in self.sets}
        self.n = 0
        self.chain = ""
        self.families: dict[str, int] = {}
        self._fp: dict[str, str] = {}

    def see(self, task: Task) -> None:
        self.n += 1
        self.chain = digest([self.chain, task.id])
        self.families[task.family] = self.families.get(task.family, 0) + 1
        if not self.sets:
            return
        fp = self._fp.get(task.id)
        if fp is None:
            fp = self._fp[task.id] = mech_fingerprint(task)
        for name, (ids, fps) in self.sets.items():
            if task.id in ids or fp in fps:
                if task.id not in self.overlaps[name]:
                    self.overlaps[name].append(task.id)
                if self.strict:
                    how = "id" if task.id in ids else "hidden mechanism"
                    raise ProtocolViolation(f"training task {task.id} is also in evaluation set {name!r} (same {how}): "
                                            "evaluation tasks must never have been seen in training")

    def state(self) -> dict:
        return {"n": self.n, "chain": self.chain, "families": dict(self.families),
                "overlaps": {k: list(v) for k, v in self.overlaps.items()}}

    def restore(self, st: dict) -> None:
        self.n, self.chain, self.families = int(st["n"]), st["chain"], dict(st["families"])
        for k, v in st.get("overlaps", {}).items():
            self.overlaps[k] = list(v)

    def report(self) -> dict:
        return {name: {"ok": not ov, "overlap": sorted(ov), "checked_training_tasks": self.n,
                       "by": "task id and mechanism fingerprint"} for name, ov in self.overlaps.items()}

    def summary(self) -> dict:
        return {"n": self.n, "digest": self.chain, "families": dict(sorted(self.families.items()))}


# ====================================================================== checkpoint hashing
_UNPARSED = object()


def _parse_checkpoint(path: str, raw: bytes) -> Any:
    head = raw.lstrip()[:1]
    if head in (b"{", b"["):
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _UNPARSED
    if raw[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
        if names and all(n.endswith(".npy") for n in names):
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        if any(n.endswith("data.pkl") for n in names):
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        return _UNPARSED
    if raw[:1] == b"\x80":
        try:
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:                                   # noqa: BLE001 - not a torch file
            import pickle
            return pickle.loads(raw)
    return _UNPARSED


def _stable(x: Any, depth: int = 0) -> Any:
    """A canonical, hashable view of a parsed checkpoint with the volatile keys removed."""
    if depth > 32:
        raise ValueError("checkpoint nested too deeply")
    if isinstance(x, Mapping):
        return {str(k): _stable(v, depth + 1) for k, v in x.items() if str(k) not in VOLATILE_KEYS}
    if isinstance(x, (list, tuple)):
        return [_stable(v, depth + 1) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_stable(v, depth + 1) for v in x), key=repr)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else repr(float(x))
    if isinstance(x, (str, int, bool, bytes, np.generic)) or x is None:
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):          # a torch tensor
        return x.detach().cpu().numpy()
    if hasattr(x, "__dict__"):
        return {"__type__": type(x).__name__, "state": _stable(vars(x), depth + 1)}
    return repr(x)


def checkpoint_digest(path: str) -> dict:
    """Digest of a saved checkpoint: `{"hash", "method", "bytes"}`. `path` is what was passed to `Learner.save`
    (a file, a directory, or a prefix the learner extended). JSON, .npy/.npz and torch checkpoints are hashed by
    content (`method="structured"`: arrays by dtype, shape and bytes; entries under `VOLATILE_KEYS` ignored);
    anything else by its raw bytes."""
    files = _saved_files(path)
    if not files:
        raise FileNotFoundError(f"Learner.save wrote nothing at {path}")
    parts, methods, size = [], set(), 0
    base = os.path.dirname(os.path.abspath(path))
    for p in files:
        with open(p, "rb") as fh:
            raw = fh.read()
        size += len(raw)
        try:
            obj = _parse_checkpoint(p, raw)
            if obj is _UNPARSED:
                raise ValueError("unknown format")
            h, m = digest(_stable(obj)), "structured"
        except Exception:                                   # noqa: BLE001 - fall back to the bytes themselves
            h, m = hashlib.blake2b(raw, digest_size=16).hexdigest(), "bytes"
        methods.add(m)
        parts.append([os.path.relpath(p, base) if len(files) > 1 else "", h])
    return {"hash": parts[0][1] if len(parts) == 1 else digest(parts),
            "method": "+".join(sorted(methods)), "bytes": size}


def _saved_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs)
    d, stem = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith(stem) and
                  os.path.isfile(os.path.join(d, f)))


def learner_hash(learner, workdir: str | None = None) -> dict:
    """Digest of a learner's long-term state: its own `state_hash()` if it defines one, otherwise the digest of
    a checkpoint saved to a temporary file."""
    fn = getattr(learner, "state_hash", None)
    if callable(fn):
        return {"hash": str(fn()), "method": "state_hash", "bytes": None}
    d = tempfile.mkdtemp(prefix="crucible-hash-", dir=workdir)
    try:
        p = os.path.join(d, "ckpt")
        learner.save(p)
        return checkpoint_digest(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ====================================================================== one batch of episodes
def _run_batch(learner, tasks: Sequence[Task], spec: EnvSpec, meter: Meter, *, learn: bool, explore: bool,
               segments: int, reset_every: int | None = None, update_every: int = 1,
               reward_mode: str | None = None) -> list[dict]:
    """One episode of every task (all with the same horizon), in parallel. Returns the operator-side episode
    records in slot order, each extended with the within-episode profile and the activity measures the failure
    diagnosis needs. `reward_mode`: that of `MechVecEnv` (observation_only unless named, Clarification 2)."""
    env = MechVecEnv(tasks, spec, reward_mode=reward_mode)
    n, h = env.n, int(tasks[0].horizon)
    if any(int(t.horizon) != h for t in tasks):
        raise ValueError("a batch runs tasks of one horizon")
    k = max(1, min(int(segments), h))
    seg_of = (np.arange(h) * k) // h
    obs = env.reset()
    learner.reset_context(n)
    mask = obs[:, spec.lever_mask] > 0.5
    start = obs[:, spec.last_action].astype(np.float64)
    prev = start.copy()
    nets = np.zeros((h, n))
    ret = np.zeros(n)
    moved, changes = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    first_move = np.full(n, -1, dtype=np.int64)
    max_move = np.zeros(n)
    traj = np.zeros((h, n, spec.max_levers))          # the settings asked for after each tick (masked levers 0)
    for t in range(h):
        if reset_every and t and t % int(reset_every) == 0:
            learner.reset_context(n)
        act = np.asarray(meter.act(obs, explore), dtype=np.float32)
        nxt, rew, done, info = env.step(act)
        if learn:
            meter.observe(obs, act, rew, done, nxt)
            if (t + 1) % update_every == 0:
                meter.update()
        cur = nxt[:, spec.last_action].astype(np.float64)
        dev = np.where(mask, np.abs(cur - start), 0.0).max(axis=1)
        chg = np.where(mask, np.abs(cur - prev), 0.0).max(axis=1) > _TOL
        away = dev > _TOL
        first_move[(first_move < 0) & away] = t
        moved += away
        changes += chg
        max_move = np.maximum(max_move, dev)
        traj[t] = np.where(mask, cur, 0.0)
        nets[t] = info["net_cents"]
        ret += rew
        obs, prev = nxt, cur
    if learn and h % update_every:
        meter.update()
    if not env.done.all():
        raise RuntimeError("an episode did not end at its horizon")
    recs = env.drain()
    if [r["task"] for r in recs] != [t.id for t in tasks]:
        raise RuntimeError("episode records out of slot order")
    ends = [int(np.flatnonzero(seg_of == j)[-1]) for j in range(k)]
    # how far the settings ever went, after the end of segment j, from where they stood at that end
    drift = [np.abs(traj[e + 1:] - traj[e]).max(axis=(0, 2)) if e + 1 < h else np.zeros(n) for e in ends[:-1]]
    for i, (task, r) in enumerate(zip(tasks, recs)):
        seg = [task.score(float(nets[seg_of == j, i].mean())) for j in range(k)]
        r["segment_scores"] = seg
        r["adaptation"] = None if seg[0] is None or seg[-1] is None else float(seg[-1] - seg[0])
        r["slope"] = None if seg[0] is None else S.slope(seg)
        r["rise"] = None if seg[0] is None else float(np.mean(seg) - seg[0])
        r["return"] = float(ret[i])
        r["money_scale"] = money_scale(task)
        r["ticks_moved"] = int(moved[i])
        r["n_changes"] = int(changes[i])
        r["first_move_tick"] = None if first_move[i] < 0 else int(first_move[i])
        r["max_move"] = float(max_move[i])
        r["drift_after_segment"] = [float(d[i]) for d in drift]
        r["validated"] = bool(task.validated)
    charge_episodes(meter.cost, recs, env.sim_seconds)
    return recs


def _horizon_batches(tasks: Sequence[Task], n_envs: int) -> list[list[int]]:
    by: dict[int, list[int]] = {}
    for i, t in enumerate(tasks):
        by.setdefault(int(t.horizon), []).append(i)
    out = []
    for h in sorted(by):
        idx = by[h]
        out += [idx[j:j + n_envs] for j in range(0, len(idx), n_envs)]
    return out


# ====================================================================== summaries
def summarise(records: Sequence[dict], seed: int = 0) -> dict:
    """Aggregates over tasks, each with its uncertainty: mean score (bootstrap interval), interquartile mean,
    success probability (Wilson interval), and the within-episode profile."""
    sc = _f(r["score"] for r in records)
    succ = [bool(r["success"]) for r in records if r["success"] is not None]
    k = max((len(r.get("segment_scores") or []) for r in records), default=0)
    seg = np.array([[np.nan if v is None else v for v in (r.get("segment_scores") or [None] * k)] +
                    [np.nan] * (k - len(r.get("segment_scores") or [None] * k)) for r in records], dtype=float) \
        if records and k else np.zeros((len(records), 0))
    return {"n": len(records), "n_scored": int(np.isfinite(sc).sum()),
            "score": S.mean_ci(sc, n_boot=N_BOOT, seed=seed), "iqm": S.iqm_ci(sc, n_boot=N_BOOT, seed=seed),
            "success": S.proportion_ci(int(sum(succ)), len(succ)),
            "adaptation": S.mean_ci(_f(r.get("adaptation") for r in records), n_boot=N_BOOT, seed=seed),
            "slope": S.mean_ci(_f(r.get("slope") for r in records), n_boot=N_BOOT, seed=seed),
            "rise": S.mean_ci(_f(r.get("rise") for r in records), n_boot=N_BOOT, seed=seed),
            "segments": [S.mean_ci(seg[:, j], n_boot=N_BOOT, seed=seed + j) for j in range(seg.shape[1])],
            "first_half": S.mean_ci(_f(r.get("score_first_half") for r in records), n_boot=N_BOOT, seed=seed),
            "second_half": S.mean_ci(_f(r.get("score_second_half") for r in records), n_boot=N_BOOT, seed=seed)}


def _group(records: Sequence[dict], key: str, seed: int = 0) -> dict:
    """Aggregates per value of a record field. For `transfer`, episodes on a training family are reported as
    `within`: an evaluation task of a training family is, by the definition in `mechanisms.base`, that family at
    seeds not used for training (whether they really were unseen is what the `disjoint` checks establish)."""
    by: dict[str, list[dict]] = {}
    for r in records:
        g = str(r.get(key) or "unspecified")
        by.setdefault("within" if key == "transfer" and g == "base" else g, []).append(r)
    return {g: summarise(rs, seed) for g, rs in sorted(by.items())}


# ====================================================================== 1. frozen evaluation
def evaluate(learner, tasks: Sequence[Task], *, spec: EnvSpec | None = None, n_envs: int = 16, segments: int = 4,
             explore: bool = False, reset_every: int | None = None, verify_frozen: bool = True, label: str = "",
             cost: Cost | None = None, boot_seed: int = 0, reward_mode: str | None = None) -> dict:
    """Frozen evaluation of `learner` on `tasks`: weights frozen, context reset at the start of every episode,
    one episode per task, records in the order of `tasks` (so two learners evaluated on the same list are paired
    by position and by `record["task"]`).

    `reset_every=k` additionally clears the learner's context every k ticks inside the episode (the ablation that
    isolates what context carries). `verify_frozen` hashes the learner's long-term state before and after.
    The learner's frozen flag is restored afterwards. `cost`, if given, also receives what this evaluation
    consumed (the result always carries its own)."""
    tasks = list(tasks)
    if not tasks:
        raise ValueError("no tasks to evaluate")
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation tasks must be distinct: results are paired by task id")
    spec = spec or learner.spec
    was = bool(getattr(learner, "frozen", False))
    learner.freeze(True)
    local = Cost()
    meter = Meter(learner, local)
    out: list[dict | None] = [None] * len(tasks)
    t0 = time.perf_counter()
    try:
        before = learner_hash(learner) if verify_frozen else None
        for idx in _horizon_batches(tasks, max(1, int(n_envs))):
            recs = _run_batch(learner, [tasks[i] for i in idx], spec, meter, learn=False, explore=explore,
                              segments=segments, reset_every=reset_every, reward_mode=reward_mode)
            for i, r in zip(idx, recs):
                out[i] = r
        meter.close()
        after = learner_hash(learner) if verify_frozen else None
    finally:
        meter.close()
        learner.freeze(was)
    if cost is not None:
        cost.add(local)
    frozen = {"verified": bool(verify_frozen), "unchanged": None, "hash_before": None, "hash_after": None,
              "method": None, "flag_was": was}
    if verify_frozen:
        frozen.update(unchanged=before["hash"] == after["hash"], hash_before=before["hash"],
                      hash_after=after["hash"], method=before["method"])
    return {"protocol": "evaluate", "schema": SCHEMA, "versions": versions(), "label": label,
            "learner": _ident(learner), "n_tasks": len(tasks), "segments": int(segments), "reset_every": reset_every, "explore": bool(explore),
            "tasks": ids, "records": out, "aggregate": summarise(out, boot_seed),
            "by_family": _group(out, "family", boot_seed), "by_transfer": _group(out, "transfer", boot_seed),
            "frozen": frozen, "cost": cost_row(local), "wall_s": time.perf_counter() - t0, "sealed": False}


def _scores(ev: dict) -> np.ndarray:
    return _f(r["score"] for r in ev["records"])


def _succ(ev: dict) -> np.ndarray:
    return _f(None if r["success"] is None else float(r["success"]) for r in ev["records"])


def _compact(ev: dict) -> dict:
    """An evaluation without its per-episode records: aggregates plus the per-task vectors that pairing needs."""
    recs = ev["records"]
    return {"n": ev["n_tasks"], "score": ev["aggregate"]["score"], "iqm": ev["aggregate"]["iqm"],
            "success": ev["aggregate"]["success"], "adaptation": ev["aggregate"]["adaptation"],
            "segments": ev["aggregate"]["segments"], "scores": [r["score"] for r in recs],
            "successes": [r["success"] for r in recs], "segment_scores": [r["segment_scores"] for r in recs],
            "by_family": {g: {"n": a["n"], "score": a["score"], "success": a["success"]}
                          for g, a in ev["by_family"].items()},
            "by_transfer": {g: {"n": a["n"], "score": a["score"], "success": a["success"]}
                            for g, a in ev["by_transfer"].items()},
            "frozen_unchanged": ev["frozen"]["unchanged"]}


# ====================================================================== failure diagnosis from records
def diagnose_episode(rec: Mapping, good: float = GOAL_FRACTION, broke_frac: float = 0.1, margin: float = 0.1,
                     shift_tol: float = 0.05) -> dict:
    """What one episode record establishes about why the objective was or was not reached (see the module
    docstring for the classes and their order). Returns `{"class", "evidence", "statement"}`."""
    ticks = max(1, int(rec["ticks"]))
    seg = rec.get("segment_scores") or []
    ev: dict[str, Any] = {"score": rec.get("score"), "segment_scores": list(seg),
                          "ticks_moved": rec.get("ticks_moved"), "n_changes": rec.get("n_changes"),
                          "ticks_fallback": rec.get("ticks_fallback"), "ticks_idle": rec.get("ticks_idle"),
                          "funds_end_cents": rec.get("funds_end_cents")}
    if rec.get("success"):
        return {"class": "succeeded", "evidence": ev, "statement": "the public objective was reached"}
    if rec.get("score") is None or not seg or any(s is None for s in seg):
        return {"class": "not_determinable", "evidence": ev,
                "statement": "the task has no private reference: the quality of the episode cannot be judged"}
    if int(rec.get("ticks_moved") or 0) == 0:
        return {"class": "never_touched", "evidence": ev,
                "statement": "no lever was ever moved from its starting setting"}
    broke = int(rec.get("ticks_fallback") or 0) + int(rec.get("ticks_idle") or 0)
    if broke / ticks >= broke_frac:
        return {"class": "went_broke", "evidence": ev,
                "statement": f"moved levers and could not pay in {broke} of {ticks} ticks "
                             f"({rec.get('ticks_fallback')} at fallback settings, {rec.get('ticks_idle')} idle); "
                             f"balance at the end {rec.get('funds_end_cents')} cents"}
    is_good = [s >= good for s in seg]
    last = seg[-1]
    if any(is_good[:-1]) and not is_good[-1]:
        peak = int(np.argmax(seg[:-1]))
        if seg[peak] - last >= margin:
            drifts = rec.get("drift_after_segment") or []
            after = float(drifts[peak]) if peak < len(drifts) else 0.0
            ev.update(peak_segment=peak, peak_score=seg[peak], last_score=last, settings_drift_after_peak=after)
            how = ("after it changed its own settings" if after > shift_tol else
                   "while it held its settings (the world changed under them)")
            return {"class": "found_and_lost", "evidence": ev, "sub": "own_change" if after > shift_tol else "held",
                    "statement": f"ran at {seg[peak]:.2f} of the reference in segment {peak + 1} of {len(seg)} and "
                                 f"at {last:.2f} in the last, {how}"}
    if is_good[-1]:
        first = int(np.argmax(is_good))
        ev.update(first_good_segment=first)
        return {"class": "found_late", "evidence": ev,
                "statement": f"ran at {last:.2f} of the reference in the last segment (objective rate {good:.2f}), "
                             f"first reached in segment {first + 1} of {len(seg)}: too late for the episode total"}
    ev.update(best_segment_score=max(seg), max_move=rec.get("max_move"))
    return {"class": "never_found", "evidence": ev,
            "statement": f"moved levers on {rec.get('ticks_moved')} of {ticks} ticks ({rec.get('n_changes')} changes) "
                         f"and never ran at the objective rate: best segment {max(seg):.2f} of the reference"}


def diagnose_records(records: Sequence[Mapping], **kw) -> dict:
    """Failure classes over a set of episode records, with the share of each and one example statement."""
    out = {"n": len(records), "n_failed": 0, "classes": {}, "examples": {}}
    for r in records:
        d = diagnose_episode(r, **kw)
        c = d["class"]
        if c == "succeeded":
            continue
        out["n_failed"] += 1
        out["classes"][c] = out["classes"].get(c, 0) + 1
        out["examples"].setdefault(c, {"task": r.get("task"), "statement": d["statement"]})
    nf = max(1, out["n_failed"])
    out["shares"] = {c: v / nf for c, v in out["classes"].items()}
    out["statements"] = [f"{v} of {out['n_failed']} failed episodes: {c} (e.g. {out['examples'][c]['task']}: "
                         f"{out['examples'][c]['statement']})" for c, v in
                         sorted(out["classes"].items(), key=lambda kv: -kv[1])]
    return out


def kernel_diagnosis(learner, task: Task, spec: EnvSpec | None = None, reward_mode: str | None = None) -> dict:
    """Run `learner` (frozen) on `task` inside a kernel world through `rig.*` tool calls (`research.bridge`) and
    classify the resulting trajectory with `research.diagnostics.diagnose`. This is where the event-log
    diagnosis applies to a numeric learner: every action is an ordinary tool call in a replayable log."""
    from . import diagnostics as D
    from .bridge import KernelMechEnv
    spec = spec or learner.spec
    was = bool(getattr(learner, "frozen", False))
    learner.freeze(True)
    try:
        env = KernelMechEnv(task, spec, reward_mode=reward_mode)
        obs = env.reset()
        learner.reset_context(1)
        while not env.done.all():
            obs, _, _, _ = env.step(learner.act(obs, explore=False))
    finally:
        learner.freeze(was)
    s = env.summary()
    # a validated task has a reference policy that reaches the objective: the task is known to be feasible
    refs = [{"name": "validity reference policy", "competent": True, "success": True, "progress": 1.0}] \
        if task.validated and task.reference_net is not None else None
    d = D.diagnose(env.world, env.actor, goal={"success": bool(s["success"]), "progress": s["score"]}, references=refs)
    return {"task": task.id, "score": s["score"], "success": s["success"], "tool_calls": s["tool_calls"],
            "primary": d["primary"], "primary_reason": d["primary_reason"],
            "findings": [{"kind": f["kind"], "status": f["status"], "detail": f["detail"]} for f in d["findings"]],
            "classes": {c: v["n"] for c, v in d["classes"].items()}, "cash": d["resources"].get("cash"),
            "difficulty": d["difficulty"]["status"],
            "not_determinable": d["not_determinable"], "world_digest": s["world_digest"]}


# ====================================================================== training interaction
def task_stream(families: Sequence[str], start: int = 0) -> Callable[[int], Task]:
    """An index-addressable endless stream of unvalidated training tasks: item i is family `i % F`, seed
    `start + i // F` (the same order as `tasks.train_stream`). Being addressable by index, it can be resumed
    without rebuilding the tasks already consumed."""
    from .tasks import TRAIN_SEED_MAX, make_task
    fams = list(families)
    if not fams:
        raise ValueError("no families")

    def at(i: int) -> Task:
        seed = int(start) + i // len(fams)
        if seed > TRAIN_SEED_MAX:
            raise RuntimeError("training seed range exhausted")
        return make_task(fams[i % len(fams)], seed, validate=False)
    at.families = tuple(fams)          # type: ignore[attr-defined]
    at.start = int(start)              # type: ignore[attr-defined]
    return at


class TaskSource:
    """A deterministic, resumable view of a training-task source, handing out batches of `n_envs` tasks with
    equal horizon. The source is a sequence of tasks (cycled without end: a fixed training set), a callable
    `index -> Task` (an endless stream, see `task_stream`), or any other iterable (consumed once; resuming has to
    re-iterate what was consumed). `state()` is all that is needed to continue: how many tasks were drawn and
    which drawn tasks still wait in a bucket."""

    def __init__(self, source: Any, n_envs: int, consumed: int = 0, pending: Mapping[str, Sequence[int]] | None = None):
        self.n = max(1, int(n_envs))
        self.consumed = 0
        self.exhausted = False
        self._buckets: dict[int, list[tuple[int, Task]]] = {}
        self._seq = self._fn = self._it = None
        if isinstance(source, TaskSource):
            raise TypeError("pass the underlying source, not a TaskSource")
        if callable(source) and not isinstance(source, (list, tuple)):
            self._fn = source
        elif isinstance(source, Sequence):
            if len(source) == 0:
                raise ValueError("empty training set")
            self._seq = list(source)
        else:
            self._it = iter(source)
        want = {int(i) for idx in (pending or {}).values() for i in idx}
        if self._it is not None:
            for i in range(int(consumed)):
                t = next(self._it)
                if i in want:
                    self._buckets.setdefault(int(t.horizon), []).append((i, t))
        else:
            for i in sorted(want):
                t = self._at(i)
                self._buckets.setdefault(int(t.horizon), []).append((i, t))
        self.consumed = int(consumed)

    def _at(self, i: int) -> Task:
        return self._seq[i % len(self._seq)] if self._seq is not None else self._fn(i)

    def _next(self) -> Task | None:
        if self._it is not None:
            try:
                return next(self._it)
            except StopIteration:
                self.exhausted = True
                return None
        return self._at(self.consumed)

    def batch(self) -> list[Task] | None:
        """The next `n_envs` tasks of one horizon; a smaller final batch when a finite source runs dry; None
        when nothing is left."""
        while not self.exhausted:
            t = self._next()
            if t is None:
                break
            b = self._buckets.setdefault(int(t.horizon), [])
            b.append((self.consumed, t))
            self.consumed += 1
            if len(b) >= self.n:
                del self._buckets[int(t.horizon)]
                return [x for _, x in b]
        for h in sorted(self._buckets):
            b = self._buckets.pop(h)
            if b:
                return [x for _, x in b]
        return None

    def state(self) -> dict:
        return {"consumed": self.consumed,
                "pending": {str(h): [i for i, _ in b] for h, b in sorted(self._buckets.items()) if b}}


def _train(learner, source: TaskSource, until_steps: int, cost: Cost, *, spec: EnvSpec, n_envs: int,
           update_every: int, explore: bool, segments: int, watch: _Watch | None = None,
           reward_mode: str | None = None) -> dict:
    """Training interaction until `cost.env_steps >= until_steps` (whole batches) or the source runs dry.
    Returns a summary of the training episodes run."""
    learner.freeze(False)
    meter = Meter(learner, cost)
    rets, scores, succ, n = [], [], [], 0
    try:
        while cost.env_steps < until_steps:
            batch = source.batch()
            if batch is None:
                break
            if watch is not None:
                for t in batch:
                    watch.see(t)
            recs = _run_batch(learner, batch, spec, meter, learn=True, explore=explore, segments=segments,
                              update_every=update_every, reward_mode=reward_mode)
            n += len(recs)
            for r in recs:
                rets.append(r["return"] / max(1, r["ticks"]))
                if r["score"] is not None:
                    scores.append(r["score"])
                if r["success"] is not None:
                    succ.append(bool(r["success"]))
    finally:
        meter.close()
    return {"episodes": n, "reward_per_tick": float(np.mean(rets)) if rets else None,
            "score": float(np.mean(scores)) if scores else None, "n_scored": len(scores),
            "success_rate": float(np.mean(succ)) if succ else None, "exhausted": bool(source.exhausted)}


# ====================================================================== 2. lifetime / learning curve
def _eval_sets(eval_sets: Any) -> dict[str, list[Task]]:
    if isinstance(eval_sets, Mapping):
        out = {str(k): list(v) for k, v in eval_sets.items()}
    else:
        out = {"eval": list(eval_sets)}
    if not out or any(not v for v in out.values()):
        raise ValueError("every evaluation set needs at least one task")
    return out


def _state_path(d: str) -> str:
    return os.path.join(d, "lifetime_state.json")


def _point_row(i: int, train: Cost, ev_cost: Cost, wall: float) -> dict:
    row = {"i": i}
    row.update(cost_row(train))
    row.update({"train_wall_s": train.train_wall_s, "infer_wall_s": train.infer_wall_s,
                "updates_returned": train.updates_returned, "updates_reported": train.updates_reported,
                "eval_env_steps": ev_cost.env_steps, "eval_infer_cpu_s": ev_cost.infer_cpu_s,
                "eval_env_sim_s": ev_cost.env_sim_s, "eval_infer_tokens": ev_cost.infer_tokens,
                "eval_search_evals": ev_cost.search_evals, "wall_s": wall})
    return row


def lifetime(learner, train_tasks: Any, budget_env_steps: int, eval_every: int, eval_sets: Any, *,
             spec: EnvSpec | None = None, n_envs: int = 16, update_every: int = 1, explore: bool = True,
             segments: int = 4, checkpoint_dir: str | None = None, resume: bool = True,
             thresholds: Sequence[float] = (GOAL_FRACTION,), sustain: int = 2, budgets: Sequence[int] | None = None,
             require_disjoint: bool = False, verify_frozen: bool = True, label: str = "",
             on_point: Callable[[dict], None] | None = None, max_points: int | None = None,
             reward_mode: str | None = None) -> dict:
    """A learning curve. Alternates training interaction on `train_tasks` (see `TaskSource`) with frozen
    evaluations on the fixed task sets `eval_sets` (`{name: [Task]}` or one list), every `eval_every` environment
    steps up to `budget_env_steps` (the first point is the untrained learner; points fall on batch boundaries, so
    their step counts are multiples of whole episodes and are recorded exactly).

    Every point carries the cumulative cost of training up to it (`research.accounting`), the evaluation
    aggregates and per-task vectors, the summary of the training episodes since the previous point, and the
    checkpoint saved at it. With `checkpoint_dir`, the run is resumable: state is committed after every point
    and a later call with the same arguments continues from the last one (the learner is restored with `load`;
    the budget may be raised). `require_disjoint` refuses a training task that is also an evaluation task;
    otherwise overlaps are only recorded (`disjoint`), because evaluating frozen on the training set is a
    legitimate diagnostic. `max_points` stops early (the result says `complete: False`).

    Derived measures (`derived[set]`), each separately and with an interval over tasks: area under the curve
    against steps and against compute, performance at fixed experience budgets, steps and time to reach each
    threshold (held for `sustain` consecutive points), for the score and for the success probability."""
    spec = spec or learner.spec
    sets = _eval_sets(eval_sets)
    budget_env_steps, eval_every = int(budget_env_steps), max(1, int(eval_every))
    ckpt_dir, temporary = _workdir(checkpoint_dir)
    cfg = {"learner": _ident(learner)["name"], "class": type(learner).__name__, "seed": int(getattr(learner, "seed", 0)),
           "n_envs": int(n_envs), "update_every": int(update_every), "explore": bool(explore),
           "eval_every": eval_every, "segments": int(segments), "label": label, "versions": versions(),
           "spec": [spec.max_levers, spec.max_gauges, spec.horizon],
           "eval_sets": {k: digest([t.id for t in v]) for k, v in sets.items()}}
    watch = _Watch(sets, strict=require_disjoint)
    train_cost, eval_cost = Cost(), Cost()
    state = None
    if resume and not temporary and os.path.exists(_state_path(ckpt_dir)):
        with open(_state_path(ckpt_dir)) as fh:
            state = json.load(fh)
        if state["config"] != _js(cfg):
            raise ValueError(f"{ckpt_dir} holds the state of a different run (learner, seed, schedule, evaluation "
                             "sets or task version differ); use another checkpoint_dir or resume=False")
    t_session = time.perf_counter()
    if state is not None:
        learner.load(state["points"][-1]["checkpoint"])
        source = TaskSource(train_tasks, n_envs, state["stream"]["consumed"], state["stream"]["pending"])
        source.exhausted = bool(state.get("exhausted", False))
        train_cost, eval_cost = Cost.from_dict(state["cost"]["train"]), Cost.from_dict(state["cost"]["eval"])
        watch.restore(state["watch"])
        if state["points"][-1]["env_steps"] < budget_env_steps and not source.exhausted:
            state["resumed"].append(int(state["points"][-1]["i"]))
        wall0 = float(state["wall_s"])
    else:
        source = TaskSource(train_tasks, n_envs)
        state = {"schema": SCHEMA, "config": _js(cfg), "points": [], "resumed": [], "wall_s": 0.0, "last_eval": {}}
        wall0 = 0.0

    def commit(train_summary: dict | None) -> dict:
        i = len(state["points"])
        before = learner_hash(learner) if verify_frozen else None
        evs = {name: evaluate(learner, ts, spec=spec, n_envs=n_envs, segments=segments, verify_frozen=False,
                              label=name, cost=eval_cost, boot_seed=i, reward_mode=reward_mode)
               for name, ts in sets.items()}
        path = os.path.join(ckpt_dir, f"point_{i:04d}.ckpt")
        learner.save(path)
        saved = checkpoint_digest(path)
        after = learner_hash(learner) if verify_frozen else None
        point = _point_row(i, train_cost, eval_cost, wall0 + time.perf_counter() - t_session)
        point.update({"train": train_summary, "eval": {name: _compact(ev) for name, ev in evs.items()},
                      "checkpoint": path, "checkpoint_hash": saved["hash"], "checkpoint_bytes": saved["bytes"],
                      "weights_hash": after["hash"] if after else saved["hash"],
                      "frozen_unchanged": None if before is None else before["hash"] == after["hash"]})
        state["points"].append(_js(point))
        state["last_eval"] = _js(evs)
        state["stream"] = source.state()
        state["exhausted"] = bool(source.exhausted)
        state["cost"] = {"train": train_cost.to_dict(), "eval": eval_cost.to_dict()}
        state["watch"] = watch.state()
        state["wall_s"] = wall0 + time.perf_counter() - t_session
        if not temporary:
            tmp = _state_path(ckpt_dir) + f".{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(state, fh)
            os.replace(tmp, _state_path(ckpt_dir))
        if on_point is not None:
            on_point(state["points"][-1])
        return point

    try:
        if not state["points"]:
            commit(None)
        while train_cost.env_steps < budget_env_steps and not source.exhausted:
            if max_points is not None and len(state["points"]) >= max_points:
                break
            target = min(budget_env_steps, (train_cost.env_steps // eval_every + 1) * eval_every)
            summary = _train(learner, source, target, train_cost, spec=spec, n_envs=n_envs,
                             update_every=update_every, explore=explore, segments=segments, watch=watch,
                             reward_mode=reward_mode)
            if summary["episodes"] == 0:
                break
            commit(summary)
        complete = train_cost.env_steps >= budget_env_steps or source.exhausted
        result = _lifetime_result(state, learner, budget_env_steps, thresholds, sustain, budgets, complete,
                                  watch, train_cost, eval_cost, label)
    finally:
        if temporary:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
    if temporary:
        for p in result["points"]:
            p["checkpoint"] = None
    return result


def _lifetime_result(state: dict, learner, budget: int, thresholds, sustain, budgets, complete: bool,
                     watch: _Watch, train_cost: Cost, eval_cost: Cost, label: str) -> dict:
    points = state["points"]
    if budgets is None:
        budgets = sorted({int(round(budget * f)) for f in (0.25, 0.5, 1.0)})
    derived = {name: derive_curve(points, name, budget, thresholds, sustain, budgets)
               for name in points[0]["eval"]}
    return _js({"protocol": "lifetime", "schema": SCHEMA, "versions": state["config"]["versions"], "label": label,
                "learner": _ident(learner),
                "config": dict(state["config"], budget_env_steps=int(budget), thresholds=list(thresholds),
                               sustain=int(sustain), budgets=list(budgets)),
                "points": points, "final": state["last_eval"], "derived": derived,
                "cost": {"train": cost_row(train_cost), "eval": cost_row(eval_cost)},
                "complete": bool(complete), "exhausted": bool(state.get("exhausted", False)),
                "resumed": list(state["resumed"]), "disjoint": watch.report(), "train_tasks": watch.summary(),
                "frozen_unchanged": all(p["frozen_unchanged"] is not False for p in points)})


def curve_matrix(points: Sequence[Mapping], set_name: str, metric: str = "score") -> tuple[np.ndarray, np.ndarray]:
    """(env steps per point, values [points, tasks]) of one evaluation set; metric is `score` or `success`."""
    key = "scores" if metric == "score" else "successes"
    x = np.array([p["env_steps"] for p in points], dtype=float)
    y = np.array([_f(None if v is None else float(v) for v in p["eval"][set_name][key]) for p in points])
    return x, y


def _threshold(x: np.ndarray, y: np.ndarray, thr: float, sustain: int, others: Mapping[str, np.ndarray],
               seed: int = 0, n_boot: int = 500) -> dict:
    """Steps to reach `thr` on the mean curve (sustained), with a bootstrap interval over tasks and the other
    cumulative quantities (compute, wall time) read off at the crossing."""
    ok = np.isfinite(y).all(axis=0)
    y = y[:, ok]
    out = {"threshold": float(thr), "sustain": int(sustain), "steps": None, "lo": None, "hi": None,
           "reached_frac": None, "n": int(y.shape[1])}
    if y.shape[1] == 0:
        return out
    sustain = max(1, min(int(sustain), len(x)))
    v = S.steps_to_threshold(x, y.mean(axis=1), thr, sustain)
    out["steps"] = v
    if v is not None:
        for k, arr in others.items():
            out[k] = float(np.interp(v, x, arr))
    rng = np.random.default_rng(seed)
    got = []
    for _ in range(n_boot):
        m = y[:, rng.integers(0, y.shape[1], y.shape[1])].mean(axis=1)
        got.append(S.steps_to_threshold(x, m, thr, sustain))
    hit = [g for g in got if g is not None]
    out["reached_frac"] = len(hit) / len(got)
    if len(hit) >= 0.5 * len(got):
        out["lo"], out["hi"] = float(np.quantile(hit, 0.025)), float(np.quantile(hit, 0.975))
    return out


def derive_curve(points: Sequence[Mapping], set_name: str, budget: float, thresholds: Sequence[float] = (0.5,),
                 sustain: int = 2, budgets: Sequence[float] = ()) -> dict:
    """The derived measures of one evaluation set's learning curve; see `lifetime`."""
    xc = np.array([p["compute_s"] for p in points], dtype=float)
    xw = np.array([p["wall_s"] for p in points], dtype=float)
    out: dict[str, Any] = {"budget_env_steps": float(budget), "compute_s_total": float(xc[-1]) if len(xc) else 0.0}
    for metric in ("score", "success"):
        x, y = curve_matrix(points, set_name, metric)
        nt = y.shape[1]
        cols = [j for j in range(nt) if np.isfinite(y[:, j]).all()]
        d: dict[str, Any] = {"n_tasks": len(cols)}
        if not cols:
            out[metric] = d
            continue
        a_steps = [S.auc(x, y[:, j], x_max=float(budget)) for j in cols]
        d["auc_steps"] = dict(S.mean_ci(a_steps, n_boot=N_BOOT), x_max=float(budget),
                              per_task=[float(v) for v in a_steps])
        if xc[-1] > 0:
            a_comp = [S.auc(xc, y[:, j], x_max=float(xc[-1])) for j in cols]
            d["auc_compute"] = dict(S.mean_ci(a_comp, n_boot=N_BOOT), x_max=float(xc[-1]),
                                    per_task=[float(v) for v in a_comp])
        else:
            d["auc_compute"] = {"n": 0, "mean": None, "lo": None, "hi": None, "low_n": True, "x_max": 0.0,
                                "note": "no measurable learner compute"}
        d["at_budget"] = {}
        for b in budgets:
            vals = [S.at_budget(x, y[:, j], float(b)) if b <= x[-1] else None for j in cols]
            d["at_budget"][str(int(b))] = dict(S.mean_ci(_f(vals), n_boot=N_BOOT), reached=bool(b <= x[-1]))
        d["final"] = S.mean_ci(y[-1, cols], n_boot=N_BOOT)
        if metric == "score":
            d["to_threshold"] = {str(t): _threshold(x, y[:, cols], float(t), sustain, {"compute_s": xc, "wall_s": xw})
                                 for t in thresholds}
        else:
            d["to_threshold"] = {str(t): _threshold(x, y[:, cols], float(t), sustain, {"compute_s": xc, "wall_s": xw})
                                 for t in (0.5,)}
        out[metric] = d
    last = points[-1]
    out["simulated_resources"] = {k: last[k] for k in ("sim_cost_cents", "sim_revenue_cents", "sim_net_cents",
                                                        "ticks_fallback", "ticks_idle")}
    return out


def aggregate_lifetimes(results: Sequence[Mapping], level: float = 0.95) -> dict:
    """Several runs of one learner (different seeds) on the same schedule: the mean curve of each evaluation set
    with a bootstrap band over runs (flagged `low_n` below five runs), and each derived measure over runs."""
    if not results:
        raise ValueError("no runs")
    names = list(results[0]["points"][0]["eval"])
    grid = sorted({float(p["env_steps"]) for r in results for p in r["points"]})
    out: dict[str, Any] = {"protocol": "lifetime_aggregate", "schema": SCHEMA, "versions": results[0].get("versions"),
                           "learner": results[0]["learner"]["name"],
                           "seeds": [r["learner"]["seed"] for r in results], "n_runs": len(results), "sets": {}}
    for name in names:
        s: dict[str, Any] = {}
        for metric, key in (("score", "score"), ("success", "success")):
            curves = []
            for r in results:
                x = [p["env_steps"] for p in r["points"]]
                y = [p["eval"][name][key]["mean" if metric == "score" else "p"] for p in r["points"]]
                curves.append((x, _f(y)))
            band = S.curve_band(curves, grid, level)
            band["low_n"] = len(results) < S.MIN_N
            s[metric] = band
        for q in ("auc_steps", "auc_compute", "final"):
            s[q] = {m: S.mean_ci(_f((r["derived"][name][m].get(q) or {}).get("mean") for r in results), n_boot=N_BOOT)
                    for m in ("score", "success")}
        out["sets"][name] = s
    cols = ("env_steps", "updates", "train_cpu_s", "infer_cpu_s", "train_gpu_s", "infer_gpu_s", "env_sim_s",
            "infer_tokens", "search_evals", "compute_s", "sim_cost_cents", "sim_net_cents")
    out["cost"] = {c: S.mean_ci([r["cost"]["train"][c] for r in results], n_boot=N_BOOT) for c in cols}
    return _js(out)


# ====================================================================== 3. the three timescales
def _paired(a: Sequence, b: Sequence, seed: int = 0) -> dict:
    return S.paired_diff(_f(a), _f(b), n_boot=N_BOOT, seed=seed)


def _decide(pd: Mapping, p_adj: float | None, alpha: float, min_effect: float = 0.0) -> bool:
    return bool(pd.get("n") and pd["mean_diff"] is not None and pd["lo"] is not None and pd["lo"] > 0
                and pd["mean_diff"] > min_effect and p_adj is not None and p_adj < alpha)


def _pair_block(ev_intact: dict, ev_reset: dict) -> dict:
    """Fast adaptation at one checkpoint: the within-episode profile with context intact, and what the context
    carries (intact minus the copy whose context is cleared every k ticks, paired by task)."""
    return {"score_intact": ev_intact["aggregate"]["score"], "score_context_reset": ev_reset["aggregate"]["score"],
            "success_intact": ev_intact["aggregate"]["success"], "success_context_reset": ev_reset["aggregate"]["success"],
            "segments_intact": ev_intact["aggregate"]["segments"], "segments_context_reset": ev_reset["aggregate"]["segments"],
            "late_minus_early": ev_intact["aggregate"]["adaptation"], "slope": ev_intact["aggregate"]["slope"],
            "late_minus_early_context_reset": ev_reset["aggregate"]["adaptation"],
            "context": _paired(_scores(ev_intact), _scores(ev_reset)),
            "context_adaptation": _paired([r["adaptation"] for r in ev_intact["records"]],
                                          [r["adaptation"] for r in ev_reset["records"]]),
            "scores_intact": [r["score"] for r in ev_intact["records"]],
            "scores_context_reset": [r["score"] for r in ev_reset["records"]],
            "adaptation_intact": [r["adaptation"] for r in ev_intact["records"]],
            "slope_intact": [r["slope"] for r in ev_intact["records"]],
            "frozen_unchanged": [ev_intact["frozen"]["unchanged"], ev_reset["frozen"]["unchanged"]]}


def timescales(learner_factory: Callable[[int], Any], train_tasks: Any, eval_tasks: Sequence[Task],
               budget_env_steps: int, *, seed: int = 0, k_reset: int = 1, spec: EnvSpec | None = None,
               n_envs: int = 16, update_every: int = 1, segments: int = 4, alpha: float = 0.05,
               min_effect: float = 0.0, workdir: str | None = None, label: str = "",
               reward_mode: str | None = None) -> dict:
    """The three timescales of learning, measured separately on the same fresh tasks.

    Two checkpoints of one learner are evaluated frozen: the untrained one and the one after `budget_env_steps`
    of training interaction on `train_tasks`. Each is evaluated twice, paired by task: with its context intact
    through the episode, and as a copy whose context is cleared every `k_reset` ticks.

    (a) fast adaptation   what happens inside an episode with frozen weights: the late-minus-early segment
                          score with context intact, and `context` = score(intact) - score(context cleared every k
                          ticks), which isolates what the context carries from what the weights (or the world's
                          own dynamics) give; `context_adaptation` is the same difference taken on late-minus-
                          early (the within-episode improvement that needs the context);
    (b) parametric        frozen score with fresh context after training minus before (`gain`); it splits exactly
                          into the gain that survives without context (`without_context`) and the gain that works
                          through the context (c);
    (c) meta-learning     the change in within-episode adaptation from the untrained to the trained checkpoint:
                          `context_gain` = context(trained) - context(untrained), and the change of the
                          adaptation profile itself: `rise_gain` (rise = mean segment score minus the first
                          segment's, the area of the profile above its starting level: larger when the same
                          climb happens earlier), the first segment in which the trained profile is already at
                          the untrained profile's final level, and the late-minus-early and slope changes (which
                          a learner that adapts *faster* need not increase: it may finish its climb in the first
                          segment).

    `attribution` says which of the three is present: the paired interval excludes zero and the Holm-adjusted
    sign-flip p-value (over the three tests) is below `alpha`. Training tasks are checked against the evaluation
    tasks as they are consumed; an overlap raises `ProtocolViolation`."""
    eval_tasks = list(eval_tasks)
    if isinstance(train_tasks, Sequence):
        ov = overlap(train_tasks, eval_tasks)
        if ov:
            raise ProtocolViolation(f"training and evaluation tasks overlap: {ov[:5]}")
    d, temporary = _workdir(workdir)
    try:
        learner = learner_factory(seed)
        spec = spec or learner.spec
        kw = dict(spec=spec, n_envs=n_envs, segments=segments, reward_mode=reward_mode)
        p0 = os.path.join(d, "untrained.ckpt")
        learner.save(p0)
        h0 = checkpoint_digest(p0)
        u_int = evaluate(learner, eval_tasks, label="untrained/intact", **kw)
        u_rst = evaluate(learner, eval_tasks, reset_every=k_reset, label="untrained/context_reset", **kw)
        watch = _Watch({"eval": eval_tasks}, strict=True)
        cost = Cost()
        tr = _train(learner, TaskSource(train_tasks, n_envs), int(budget_env_steps), cost, spec=spec, n_envs=n_envs,
                    update_every=update_every, explore=True, segments=segments, watch=watch, reward_mode=reward_mode)
        p1 = os.path.join(d, "trained.ckpt")
        learner.save(p1)
        h1 = checkpoint_digest(p1)
        t_int = evaluate(learner, eval_tasks, label="trained/intact", **kw)
        t_rst = evaluate(learner, eval_tasks, reset_every=k_reset, label="trained/context_reset", **kw)
    finally:
        if temporary:
            shutil.rmtree(d, ignore_errors=True)
    fast = {"untrained": _pair_block(u_int, u_rst), "trained": _pair_block(t_int, t_rst)}
    gain = _paired(_scores(t_int), _scores(u_int))
    direct = _paired(_scores(t_rst), _scores(u_rst))
    ctx_u, ctx_t = _scores(u_int) - _scores(u_rst), _scores(t_int) - _scores(t_rst)
    meta_ctx = _paired(ctx_t, ctx_u)
    meta = {"context_gain": meta_ctx,
            "late_minus_early_gain": _paired([r["adaptation"] for r in t_int["records"]],
                                             [r["adaptation"] for r in u_int["records"]]),
            "slope_gain": _paired([r["slope"] for r in t_int["records"]], [r["slope"] for r in u_int["records"]]),
            "rise_gain": _paired([r["rise"] for r in t_int["records"]], [r["rise"] for r in u_int["records"]]),
            "curves": {"untrained": u_int["aggregate"]["segments"], "trained": t_int["aggregate"]["segments"]}}
    cu = [c["mean"] for c in meta["curves"]["untrained"]]
    ct = [c["mean"] for c in meta["curves"]["trained"]]
    if cu and ct and all(v is not None for v in cu + ct):
        hit = [j for j, v in enumerate(ct) if v >= cu[-1]]
        meta["segments_to_untrained_final"] = {"untrained_final": cu[-1], "segment": hit[0] + 1 if hit else None,
                                               "of": len(ct)}
    ps = [fast["trained"]["context"]["p_value"], gain["p_value"], meta_ctx["p_value"]]
    adj = S.holm([1.0 if p is None else p for p in ps])
    attribution = {"fast": _decide(fast["trained"]["context"], adj[0], alpha, min_effect),
                   "parametric": _decide(gain, adj[1], alpha, min_effect),
                   "meta": _decide(meta_ctx, adj[2], alpha, min_effect)}
    return _js({"protocol": "timescales", "schema": SCHEMA, "versions": versions(), "label": label,
                "learner": _ident(learner),
                "k_reset": int(k_reset), "segments": int(segments), "alpha": alpha, "min_effect": min_effect,
                "n_tasks": len(eval_tasks), "tasks": [t.id for t in eval_tasks],
                "budget_env_steps": int(budget_env_steps), "training": dict(tr, **cost_row(cost)),
                "checkpoints": {"untrained": h0["hash"], "trained": h1["hash"],
                                "weights_changed_by_training": h0["hash"] != h1["hash"]},
                "disjoint": watch.report()["eval"], "train_tasks": watch.summary(),
                "fast": fast,
                "parametric": {"gain": gain, "without_context": direct,
                               "success_gain": _paired(_succ(t_int), _succ(u_int)),
                               "score_untrained": u_int["aggregate"]["score"], "score_trained": t_int["aggregate"]["score"],
                               "success_untrained": u_int["aggregate"]["success"],
                               "success_trained": t_int["aggregate"]["success"]},
                "meta": meta, "p_holm": {"fast": adj[0], "parametric": adj[1], "meta": adj[2]},
                "attribution": attribution,
                "frozen_unchanged": all(e["frozen"]["unchanged"] is not False for e in (u_int, u_rst, t_int, t_rst))})


# ====================================================================== 4. parametric internalisation
def _verdict(pd: Mapping, alpha: float) -> str:
    if not pd.get("n") or pd["mean_diff"] is None:
        return "no_data"
    if pd["lo"] > 0 and pd["p_value"] < alpha:
        return "gain"
    if pd["hi"] < 0 and pd["p_value"] < alpha:
        return "loss"
    return "no_difference_detected"


def internalisation(learner_factory: Callable[[int], Any], train_tasks: Any, eval_tasks: Sequence[Task],
                    budget_env_steps: int, *, seed: int = 0, spec: EnvSpec | None = None, n_envs: int = 16,
                    update_every: int = 1, segments: int = 4, alpha: float = 0.05, reload: bool = True,
                    repeat_check: bool = True, retention: Mapping | None = None, workdir: str | None = None,
                    label: str = "", reward_mode: str | None = None) -> dict:
    """The parametric internalisation protocol: how much of what a learner acquired is in its weights.

    Eight steps, each recorded in `steps`:

      1. evaluate the original weights (frozen, fresh context) on `eval_tasks`;
      2. allow learning interactions on `train_tasks` for `budget_env_steps` environment steps;
      3. update weights (the learner's `update`, interleaved with step 2 as the contract prescribes);
      4. save the checkpoint;
      5. clear task-specific context: with `reload` (default) the subject of the evaluation is a *new* learner
         from the factory restored from that checkpoint, so nothing but the checkpoint survives;
      6. disable retrieval of training trajectories (`freeze(True)`);
      7. evaluate on `eval_tasks` - fresh related tasks never seen in training;
      8. compare with the original checkpoint, paired by task.

    Verified, not assumed: every training task consumed is checked against the evaluation tasks by id and by
    mechanism fingerprint (an overlap raises `ProtocolViolation`); the checkpoint is hashed before and after each
    frozen evaluation (a change makes the result `valid: False` with the violation named); with `repeat_check`
    the final evaluation is run twice and must reproduce itself (a frozen learner with a cleared context has no
    way to differ, unless it is stochastic or keeps state outside its checkpoint - reported, not judged).

    The result is reported whatever the outcome: `verdict` is `internalised`, `no_gain_detected` or `regressed`
    (or `invalid`). `retention={"train_tasks": ..., "budget_env_steps": int, "eval_tasks": optional}` continues
    training on a second family and re-evaluates the first: how much of the first gain is kept."""
    eval_tasks = list(eval_tasks)
    guard = {"eval": eval_tasks}
    if retention and retention.get("eval_tasks"):
        guard["second_eval"] = list(retention["eval_tasks"])
    for name, src in (("train_tasks", train_tasks), ("retention.train_tasks", (retention or {}).get("train_tasks"))):
        if isinstance(src, Sequence):
            for gname, gtasks in guard.items():
                ov = overlap(src, gtasks)
                if ov:
                    raise ProtocolViolation(f"{name} and evaluation set {gname!r} overlap (by id or hidden "
                                            f"mechanism): {ov[:5]}")
    d, temporary = _workdir(workdir)
    violations: list[str] = []
    warnings: list[str] = []
    steps: list[dict] = []
    try:
        learner = learner_factory(seed)
        spec = spec or learner.spec
        kw = dict(spec=spec, n_envs=n_envs, segments=segments, reward_mode=reward_mode)
        # 1 ---------------------------------------------------------------------------------------
        p0 = os.path.join(d, "original.ckpt")
        learner.save(p0)
        h0 = checkpoint_digest(p0)
        e0 = evaluate(learner, eval_tasks, label="original", **kw)
        if e0["frozen"]["unchanged"] is False:
            violations.append("the long-term state changed during the frozen evaluation of the original weights")
        steps.append({"step": 1, "name": "evaluate_original_weights", "checkpoint_hash": h0["hash"],
                      "score": e0["aggregate"]["score"], "success": e0["aggregate"]["success"],
                      "frozen_unchanged": e0["frozen"]["unchanged"], "cost": e0["cost"]})
        # 2, 3 ------------------------------------------------------------------------------------
        watch = _Watch(guard, strict=True)
        cost = Cost()
        tr = _train(learner, TaskSource(train_tasks, n_envs), int(budget_env_steps), cost, spec=spec, n_envs=n_envs,
                    update_every=update_every, explore=True, segments=segments, watch=watch, reward_mode=reward_mode)
        steps.append({"step": 2, "name": "learning_interactions", "env_steps": cost.env_steps,
                      "episodes": cost.episodes, "infer_cpu_s": cost.infer_cpu_s, "infer_gpu_s": cost.infer_gpu_s,
                      "infer_tokens": cost.infer_tokens, "search_evals": cost.search_evals,
                      "env_sim_s": cost.env_sim_s, "sim_cost_cents": cost.sim_cost_cents,
                      "sim_net_cents": cost.sim_net_cents, "training_episodes": tr,
                      "training_tasks": watch.summary()})
        steps.append({"step": 3, "name": "update_weights", "updates": cost.updates,
                      "updates_returned": cost.updates_returned, "updates_reported": cost.updates_reported,
                      "train_cpu_s": cost.train_cpu_s, "train_gpu_s": cost.train_gpu_s,
                      "interleaved_with_step_2": True, "update_every_ticks": int(update_every)})
        # 4 ---------------------------------------------------------------------------------------
        p1 = os.path.join(d, "trained.ckpt")
        learner.save(p1)
        h1 = checkpoint_digest(p1)
        steps.append({"step": 4, "name": "save_checkpoint", "path": None if temporary else p1,
                      "checkpoint_hash": h1["hash"], "bytes": h1["bytes"], "hash_method": h1["method"],
                      "weights_changed_by_training": h1["hash"] != h0["hash"]})
        # 5 ---------------------------------------------------------------------------------------
        if reload:
            subject = learner_factory(seed)
            subject.load(p1)
            hs = learner_hash(subject)
            same = hs["hash"] == h1["hash"] if hs["method"] != "state_hash" else None
            if same is False:
                warnings.append("a new learner restored from the checkpoint does not save back to the same state: "
                                "the checkpoint does not round-trip")
            how = "a new learner from the factory, restored from the saved checkpoint"
        else:
            subject, same, how = learner, None, "reset_context on the trained learner"
        subject.reset_context(n_envs)
        steps.append({"step": 5, "name": "clear_task_context", "method": how, "checkpoint_round_trips": same})
        # 6 ---------------------------------------------------------------------------------------
        subject.freeze(True)
        if not getattr(subject, "frozen", False):
            violations.append("freeze(True) did not set the learner's frozen flag")
        steps.append({"step": 6, "name": "disable_retrieval", "call": "freeze(True)",
                      "frozen_flag": bool(getattr(subject, "frozen", False))})
        # 7 ---------------------------------------------------------------------------------------
        e1 = evaluate(subject, eval_tasks, label="trained", **kw)
        if e1["frozen"]["unchanged"] is False:
            violations.append("the long-term state changed during the frozen evaluation of the trained weights")
        disjoint = watch.report()["eval"]
        if not disjoint["ok"]:
            violations.append("evaluation tasks were seen in training")
        repeat = None
        if repeat_check:
            e1b = evaluate(subject, eval_tasks, label="trained/repeat", verify_frozen=False, **kw)
            diff = np.abs(_scores(e1) - _scores(e1b))
            repeat = {"identical": bool(np.nanmax(diff) == 0.0) if np.isfinite(diff).any() else None,
                      "max_abs_score_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else None}
            if repeat["identical"] is False:
                warnings.append("the frozen evaluation did not reproduce itself: the learner is stochastic at "
                                "evaluation or keeps state outside its checkpoint")
        steps.append({"step": 7, "name": "evaluate_on_fresh_tasks", "n_tasks": len(eval_tasks),
                      "score": e1["aggregate"]["score"], "success": e1["aggregate"]["success"],
                      "frozen_unchanged": e1["frozen"]["unchanged"], "hash_before": e1["frozen"]["hash_before"],
                      "hash_after": e1["frozen"]["hash_after"], "disjoint_from_training": disjoint,
                      "repeat": repeat, "cost": e1["cost"]})
        # 8 ---------------------------------------------------------------------------------------
        gain = _paired(_scores(e1), _scores(e0))
        sgain = _paired(_succ(e1), _succ(e0))
        steps.append({"step": 8, "name": "compare_with_original_paired_by_task", "score_gain": gain,
                      "success_gain": sgain})
        v = _verdict(gain, alpha)
        verdict = "invalid" if violations else {"gain": "internalised", "loss": "regressed"}.get(v, "no_gain_detected")
        out = {"protocol": "internalisation", "schema": SCHEMA, "versions": versions(), "label": label,
               "learner": _ident(learner),
               "seed": int(seed), "alpha": alpha, "n_tasks": len(eval_tasks), "tasks": [t.id for t in eval_tasks],
               "budget_env_steps": int(budget_env_steps), "steps": steps,
               "update_count": cost.updates, "env_interactions": cost.env_steps,
               "compute": {"train_cpu_s": cost.train_cpu_s, "train_gpu_s": cost.train_gpu_s,
                           "infer_cpu_s": cost.infer_cpu_s, "infer_gpu_s": cost.infer_gpu_s,
                           "env_sim_s": cost.env_sim_s, "infer_tokens": cost.infer_tokens,
                           "search_evals": cost.search_evals, "compute_s": cost.compute_s,
                           "eval_infer_cpu_s": e0["cost"]["infer_cpu_s"] + e1["cost"]["infer_cpu_s"]},
               "simulated_resources": {"sim_cost_cents": cost.sim_cost_cents, "sim_net_cents": cost.sim_net_cents,
                                       "ticks_fallback": cost.ticks_fallback, "ticks_idle": cost.ticks_idle},
               "original": _compact(e0), "trained": _compact(e1),
               "transfer_performance": {"score": e1["aggregate"]["score"], "iqm": e1["aggregate"]["iqm"],
                                        "success": e1["aggregate"]["success"], "by_family": e1["by_family"],
                                        "by_transfer": e1["by_transfer"]},
               "gain": gain, "success_gain": sgain, "disjoint": disjoint,
               "frozen_unchanged": {"original": e0["frozen"]["unchanged"], "trained": e1["frozen"]["unchanged"]},
               "repeat": repeat, "violations": violations, "warnings": warnings, "valid": not violations,
               "verdict": verdict, "retention": None}
        # retention -------------------------------------------------------------------------------
        if retention:
            out["retention"] = _retention(learner_factory, learner, subject, retention, eval_tasks, e0, e1, watch,
                                          dict(kw, update_every=update_every), seed, reload, d, alpha, violations)
            out["valid"] = not violations
            if violations:
                out["verdict"] = "invalid"
    finally:
        if temporary:
            shutil.rmtree(d, ignore_errors=True)
    return _js(out)


def _retention(factory, learner, subject1, retention: Mapping, eval_tasks, e0, e1, watch: _Watch, kw: dict,
               seed: int, reload: bool, d: str, alpha: float, violations: list[str]) -> dict:
    update_every = kw.pop("update_every")
    second_eval = list(retention.get("eval_tasks") or [])
    b_before = evaluate(subject1, second_eval, label="second/before", **kw) if second_eval else None
    cost = Cost()
    tr = _train(learner, TaskSource(retention["train_tasks"], kw["n_envs"]), int(retention["budget_env_steps"]), cost,
                spec=kw["spec"], n_envs=kw["n_envs"], update_every=update_every, explore=True,
                segments=kw["segments"], watch=watch, reward_mode=kw.get("reward_mode"))
    p2 = os.path.join(d, "after_second.ckpt")
    learner.save(p2)
    if reload:
        subject = factory(seed)
        subject.load(p2)
    else:
        subject = learner
    subject.reset_context(kw["n_envs"])
    subject.freeze(True)
    e2 = evaluate(subject, eval_tasks, label="first/after_second_training", **kw)
    if e2["frozen"]["unchanged"] is False:
        violations.append("the long-term state changed during the frozen evaluation after the second training")
    s0, s1, s2 = _scores(e0), _scores(e1), _scores(e2)
    first_gain = _paired(s1, s0)
    learned = _verdict(first_gain, alpha) == "gain"
    kept = S.ratio_ci(s2 - s0, s1 - s0, n_boot=N_BOOT) if learned else \
        {"defined": False, "ratio": None, "lo": None, "hi": None, "n": int(len(s0)),
         "reason": "nothing was gained on the first family, so there is nothing to retain"}
    out = {"second_training": dict(tr, **cost_row(cost)), "second_training_tasks": watch.summary(),
           "first_family": {"original": e0["aggregate"]["score"], "after_first_training": e1["aggregate"]["score"],
                            "after_second_training": e2["aggregate"]["score"],
                            "success_after_second_training": e2["aggregate"]["success"],
                            "scores_after_second_training": [r["score"] for r in e2["records"]]},
           "kept_fraction": kept, "forgetting": _paired(s1, s2), "forgetting_verdict": _verdict(_paired(s1, s2), alpha),
           "frozen_unchanged": e2["frozen"]["unchanged"], "disjoint": watch.report()}
    if b_before is not None:
        b_after = evaluate(subject, second_eval, label="second/after", **kw)
        out["second_family"] = {"before": b_before["aggregate"]["score"], "after": b_after["aggregate"]["score"],
                                "gain": _paired(_scores(b_after), _scores(b_before)),
                                "success_after": b_after["aggregate"]["success"]}
    return out


# ====================================================================== 5. transfer
def _as_source(x: Any) -> Any:
    """Family names become an index-addressable training stream; anything else is already a task source."""
    if isinstance(x, (list, tuple)) and x and all(isinstance(v, str) for v in x):
        return task_stream(list(x))
    return x


def _source_desc(src: Any) -> Any:
    """What identifies a re-iterable training source (None for a one-shot iterator: it cannot be recognised)."""
    if callable(src) and hasattr(src, "families"):
        return {"families": list(src.families), "start": int(getattr(src, "start", 0))}
    if isinstance(src, Sequence):
        return {"tasks": digest([t.id for t in src])}
    return None


def transfer_targets(source_families: Sequence[str], n_eval: int = 8, split: str = "val", adapt_start: int = 500_000,
                     seats: int | None = 1, **build_kw) -> dict:
    """Targets for `transfer` from the family registry: `within` is new (validation-range) seeds of the source
    families; the other three categories are the families of `split` that declare that relation to the training
    set. Adaptation streams start at seed `adapt_start` of the training range, away from the source stream."""
    from .tasks import MECH_FAMILIES, eval_tasks, families_by
    if split != "val":
        raise ValueError("public targets come from the validation split; the test split is reached through "
                         "research.sealed")
    out = {"within": {"category": "within", "families": list(source_families),
                      "eval": eval_tasks(list(source_families), n_eval, **build_kw),
                      "train": task_stream(list(source_families), adapt_start)}}
    for cat in CATEGORIES[1:]:
        fams = [f for f in families_by(split, transfer=cat, seats=seats) if f in MECH_FAMILIES]
        if fams:
            out[cat] = {"category": cat, "families": fams, "eval": eval_tasks(fams, n_eval, **build_kw),
                        "train": task_stream(fams, adapt_start)}
    return out


def _auc_tasks(res: Mapping, set_name: str, x_max: float, floor: float) -> np.ndarray:
    x, y = curve_matrix(res["points"], set_name)
    return np.array([S.auc(x, y[:, j] - floor, x_max=x_max) if np.isfinite(y[:, j]).all() else np.nan
                     for j in range(y.shape[1])])


def _steps_pair(xa, ya, xb, yb, thr: float, sustain: int, seed: int = 0, n_boot: int = 500) -> dict:
    """Steps to threshold of two arms (pretrained a, scratch b) on the same (seed, task) columns, with the ratio
    b / a (above 1: the pretrained learner got there in fewer steps) and a joint bootstrap over columns."""
    ok = np.isfinite(ya).all(axis=0) & np.isfinite(yb).all(axis=0)
    ya, yb = ya[:, ok], yb[:, ok]
    sustain_a, sustain_b = max(1, min(sustain, len(xa))), max(1, min(sustain, len(xb)))
    sa = S.steps_to_threshold(xa, ya.mean(axis=1), thr, sustain_a) if ya.size else None
    sb = S.steps_to_threshold(xb, yb.mean(axis=1), thr, sustain_b) if yb.size else None
    out = {"threshold": float(thr), "sustain": int(sustain), "pretrained": sa, "scratch": sb, "ratio": None,
           "lo": None, "hi": None, "saved_steps": None, "note": None, "n": int(ok.sum())}
    if sa is None and sb is None:
        out["note"] = "neither arm reached the threshold within the budget"
    elif sa is None:
        out["note"] = "only the learner trained from scratch reached the threshold: the ratio is below " \
                      f"{sb / float(xa[-1]):.3g}"
        out["ratio_upper_bound"] = sb / float(xa[-1])
    elif sb is None:
        out["note"] = "only the pretrained learner reached the threshold" + \
                      ("" if sa > 0 else " (without any target interaction)")
        if sa > 0:
            out["ratio_lower_bound"] = float(xb[-1]) / sa
    else:
        out["saved_steps"] = sb - sa
        if sa > 0:
            out["ratio"] = sb / sa
        else:
            out["note"] = "the pretrained learner met the threshold without any target interaction: ratio unbounded"
            out["unbounded"] = sb > 0
    if out["ratio"] is not None and ya.shape[1] > 1:
        rng = np.random.default_rng(seed)
        rs = []
        for _ in range(n_boot):
            c = rng.integers(0, ya.shape[1], ya.shape[1])
            a = S.steps_to_threshold(xa, ya[:, c].mean(axis=1), thr, sustain_a)
            b = S.steps_to_threshold(xb, yb[:, c].mean(axis=1), thr, sustain_b)
            if a and b is not None:
                rs.append(b / a)
        out["boot_defined_frac"] = len(rs) / n_boot
        if len(rs) >= 0.5 * n_boot:
            out["lo"], out["hi"] = float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))
    return out


def _mean_curve(x: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    cis = [S.mean_ci(y[i], n_boot=1000, seed=seed + i) for i in range(len(x))]
    return {"x": x.tolist(), "mean": [c["mean"] for c in cis], "lo": [c["lo"] for c in cis],
            "hi": [c["hi"] for c in cis], "n": [c["n"] for c in cis]}


def _wilson_curve(x: np.ndarray, y: np.ndarray) -> dict:
    """A success-probability curve: at every point the share of successful episodes (columns of `y`, 0/1) with its
    Wilson interval - an interval that stays honest when every episode failed or every one succeeded."""
    cis = [S.proportion_ci(int(np.nansum(y[i])), int(np.isfinite(y[i]).sum())) for i in range(len(x))]
    return {"x": np.asarray(x, float).tolist(), "mean": [c["p"] for c in cis], "lo": [c["lo"] for c in cis],
            "hi": [c["hi"] for c in cis], "n": [c["n"] for c in cis], "interval": "wilson"}


def transfer(learner_factory: Callable[[int], Any], source: Any, targets: Mapping[str, Mapping], *,
             source_budget: int, adapt_budget: int, eval_every: int, seeds: Sequence[int] = (0,),
             spec: EnvSpec | None = None, n_envs: int = 16, update_every: int = 1, segments: int = 4,
             threshold: float = GOAL_FRACTION, sustain: int = 1, floor: float = 0.0, alpha: float = 0.05,
             workdir: str | None = None, label: str = "", reward_mode: str | None = None) -> dict:
    """Transfer from source families to targets in the four categories (`CATEGORIES`).

    `source`: family names or any task source (`TaskSource`). `targets`: `{name: {"category": one of CATEGORIES,
    "eval": [Task], "train": family names / task source / None}}` (see `transfer_targets`). For every learner
    seed: train a learner on the source for `source_budget` steps and save it; then per target

    * zero-shot: frozen evaluation of the source-trained checkpoint on the target's evaluation tasks, against
      the untrained learner of the same seed on the same tasks (paired);
    * adaptation (when the target has training tasks): two `lifetime` runs with the same seed, the same
      target training stream, the same evaluation tasks and the same `adapt_budget` - one starting from the
      source-trained checkpoint, one from scratch.

    Transfer efficiency, pooled over (seed, task) pairs: the difference and the ratio of the areas under the two
    adaptation curves (areas above `floor`; the ratio is defined only while the scratch area is reliably
    positive), and the ratio of steps to `threshold` (scratch / pretrained). Above 1 means the source helped.
    `negative_transfer` is True when the pretrained arm is reliably *worse*: its area is below the scratch area
    with the paired interval excluding zero (`negative_zero_shot` likewise for the zero-shot comparison).
    No target evaluation task may appear in source or target training: verified as tasks are consumed.
    With `workdir`, an interrupted call continues: finished source trainings are recognised by their
    fingerprint and checkpoint hash, and the adaptation runs resume from their last point."""
    for name, t in targets.items():
        if t.get("category") not in CATEGORIES:
            raise ValueError(f"target {name!r}: category must be one of {CATEGORIES}")
    guard = {name: list(t["eval"]) for name, t in targets.items()}
    src = _as_source(source)
    root, temporary = _workdir(workdir)
    per: dict[str, dict] = {name: {"zs_src": [], "zs_new": [], "A": [], "B": []} for name in targets}
    source_runs = []
    learner = None
    try:
        for seed in seeds:
            sd = os.path.join(root, f"seed_{int(seed)}")
            os.makedirs(sd, exist_ok=True)
            learner = learner_factory(seed)
            sp = spec or learner.spec
            ck, meta_path = os.path.join(sd, "source.ckpt"), os.path.join(sd, "source_state.json")
            fingerprint = _js({"learner": _ident(learner)["name"], "seed": int(seed), "budget": int(source_budget),
                               "n_envs": int(n_envs), "update_every": int(update_every), "versions": versions(),
                               "source": _source_desc(src),
                               "guard": {k: digest([t.id for t in v]) for k, v in guard.items()}})
            run = None
            if not temporary and fingerprint["source"] is not None and os.path.exists(meta_path) and _saved_files(ck):
                with open(meta_path) as fh:
                    meta = json.load(fh)
                if meta["fingerprint"] == fingerprint and meta["run"]["checkpoint_hash"] == checkpoint_digest(ck)["hash"]:
                    run = dict(meta["run"], reused=True)       # the source training of an interrupted run
            if run is None:
                watch = _Watch(guard, strict=True)
                cost = Cost()
                tr = _train(learner, TaskSource(src, n_envs), int(source_budget), cost, spec=sp, n_envs=n_envs,
                            update_every=update_every, explore=True, segments=segments, watch=watch,
                            reward_mode=reward_mode)
                learner.save(ck)
                run = _js({"seed": int(seed), "training": dict(tr, **cost_row(cost)), "reused": False,
                           "training_tasks": watch.summary(), "checkpoint_hash": checkpoint_digest(ck)["hash"]})
                if not temporary:
                    with open(meta_path + ".tmp", "w") as fh:
                        json.dump({"fingerprint": fingerprint, "run": run}, fh)
                    os.replace(meta_path + ".tmp", meta_path)
            source_runs.append(run)
            for name, t in targets.items():
                ev_tasks = guard[name]
                pre = learner_factory(seed)
                pre.load(ck)
                new = learner_factory(seed)
                per[name]["zs_src"].append(evaluate(pre, ev_tasks, spec=sp, n_envs=n_envs, segments=segments,
                                                    label=f"{name}/zero_shot/source_trained", reward_mode=reward_mode))
                per[name]["zs_new"].append(evaluate(new, ev_tasks, spec=sp, n_envs=n_envs, segments=segments,
                                                    label=f"{name}/zero_shot/untrained", reward_mode=reward_mode))
                if t.get("train") is None:
                    continue
                kw = dict(spec=sp, n_envs=n_envs, update_every=update_every, segments=segments,
                          thresholds=(threshold,), sustain=sustain, require_disjoint=True, reward_mode=reward_mode)
                a = lifetime(pre, _as_source(t["train"]), adapt_budget, eval_every, {"target": ev_tasks},
                             checkpoint_dir=None if temporary else os.path.join(sd, name, "pretrained"),
                             label=f"{name}/pretrained", **kw)
                b = lifetime(new, _as_source(t["train"]), adapt_budget, eval_every, {"target": ev_tasks},
                             checkpoint_dir=None if temporary else os.path.join(sd, name, "scratch"),
                             label=f"{name}/scratch", **kw)
                if a["train_tasks"]["digest"] != b["train_tasks"]["digest"]:
                    raise ProtocolViolation(f"target {name!r}: the two arms did not train on the same tasks")
                per[name]["A"].append(a)
                per[name]["B"].append(b)
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)
    out_t = {}
    for name, t in targets.items():
        p = per[name]
        zs_s = np.concatenate([_scores(e) for e in p["zs_src"]])
        zs_n = np.concatenate([_scores(e) for e in p["zs_new"]])
        zs_gain = _paired(zs_s, zs_n)
        rec_s = [r for e in p["zs_src"] for r in e["records"]]
        rec_n = [r for e in p["zs_new"] for r in e["records"]]
        o: dict[str, Any] = {"category": t["category"], "families": sorted({x.family for x in guard[name]}),
                             "n_tasks": len(guard[name]), "n_pairs": int(len(zs_s)),
                             "zero_shot": {"source_trained": summarise(rec_s), "untrained": summarise(rec_n),
                                           "gain": zs_gain, "success_gain": _paired(
                                               np.concatenate([_succ(e) for e in p["zs_src"]]),
                                               np.concatenate([_succ(e) for e in p["zs_new"]])),
                                           "by_family": _group(rec_s, "family"),
                                           "scores_source_trained": zs_s.tolist(), "scores_untrained": zs_n.tolist()},
                             "negative_zero_shot": _verdict(zs_gain, alpha) == "loss", "adaptation": None,
                             "negative_transfer": None}
        if p["A"]:
            auc_a = np.concatenate([_auc_tasks(r, "target", float(adapt_budget), floor) for r in p["A"]])
            auc_b = np.concatenate([_auc_tasks(r, "target", float(adapt_budget), floor) for r in p["B"]])
            diff = _paired(auc_a, auc_b)
            xa, _ = curve_matrix(p["A"][0]["points"], "target")
            ya = np.concatenate([curve_matrix(r["points"], "target")[1] for r in p["A"]], axis=1)
            yb = np.concatenate([curve_matrix(r["points"], "target")[1] for r in p["B"]], axis=1)
            sa = np.concatenate([curve_matrix(r["points"], "target", "success")[1] for r in p["A"]], axis=1)
            sb = np.concatenate([curve_matrix(r["points"], "target", "success")[1] for r in p["B"]], axis=1)
            neg = _verdict(diff, alpha) == "loss"
            o["adaptation"] = {
                "budget_env_steps": int(adapt_budget), "eval_every": int(eval_every), "floor": float(floor),
                "auc": {"pretrained": S.mean_ci(auc_a, n_boot=N_BOOT), "scratch": S.mean_ci(auc_b, n_boot=N_BOOT),
                        "diff": diff, "ratio": S.ratio_ci(auc_a, auc_b, n_boot=N_BOOT)},
                "steps_to_threshold": _steps_pair(xa, ya, xa, yb, float(threshold), int(sustain)),
                "final": {"pretrained": S.mean_ci(ya[-1], n_boot=N_BOOT), "scratch": S.mean_ci(yb[-1], n_boot=N_BOOT),
                          "diff": _paired(ya[-1], yb[-1])},
                "curves": {"pretrained": _mean_curve(xa, ya), "scratch": _mean_curve(xa, yb),
                           "pretrained_success": _wilson_curve(xa, sa), "scratch_success": _wilson_curve(xa, sb)},
                "cost": {"pretrained": [r["cost"]["train"] for r in p["A"]],
                         "scratch": [r["cost"]["train"] for r in p["B"]]},
                "same_training_tasks": True}
            o["negative_transfer"] = bool(neg)
            o["evidence"] = (f"area under the adaptation curve: pretrained {diff['mean_diff']:+.3f} against scratch "
                             f"[{diff['lo']:+.3f}, {diff['hi']:+.3f}], p={diff['p_value']:.3g}")
        out_t[name] = o
    return _js({"protocol": "transfer", "schema": SCHEMA, "versions": versions(), "label": label,
                "learner": _ident(learner) if learner is not None else None, "seeds": [int(s) for s in seeds],
                "source": {"families": list(getattr(src, "families", ())) or sorted(
                               {f for r in source_runs for f in r["training_tasks"]["families"]}),
                           "budget_env_steps": int(source_budget), "runs": source_runs},
                "threshold": float(threshold), "sustain": int(sustain), "alpha": alpha, "targets": out_t,
                "categories": {c: [n for n, t in targets.items() if t["category"] == c] for c in CATEGORIES},
                "negative_transfer": sorted(n for n, o in out_t.items() if o["negative_transfer"]),
                "negative_zero_shot": sorted(n for n, o in out_t.items() if o["negative_zero_shot"])})


# ====================================================================== 6. comparison
def _per_task_mean(mats: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over runs (seeds) of per-task vectors: the unit of analysis stays the task."""
    return np.nanmean(np.vstack(mats), axis=0)


def _pairwise(values: Mapping[str, np.ndarray], quantity: str, set_name: str, reference: str | None) -> list[dict]:
    names = list(values)
    pairs = [(a, reference) for a in names if a != reference] if reference in values else \
        list(itertools.combinations(names, 2))
    rows = []
    for a, b in pairs:
        pd = _paired(values[a], values[b])
        rows.append(dict(pd, a=a, b=b, set=set_name, quantity=quantity))
    return rows


def _holm_rows(rows: list[dict], alpha: float) -> None:
    ps = [1.0 if r["p_value"] is None else r["p_value"] for r in rows]
    for r, p in zip(rows, S.holm(ps) if rows else []):
        r["p_holm"] = p
        r["significant"] = bool(r["n"] and r["lo"] is not None and p < alpha and (r["lo"] > 0 or r["hi"] < 0))
        r["direction"] = (r["a"] if r["mean_diff"] > 0 else r["b"]) if r["significant"] else None
        r["family_size"] = len(rows)


def evaluate_many(subjects: Mapping[str, Sequence[Any]], tasks: Sequence[Task], *, reference: str | None = None,
                  spec: EnvSpec | None = None, n_envs: int = 16, segments: int = 4, alpha: float = 0.05,
                  label: str = "", reward_mode: str | None = None) -> dict:
    """Frozen evaluation of several learners on identical tasks: `{name: [learner, ...]}` (one learner per seed,
    as they are - trained or not), then `combine_evaluations`."""
    tasks = list(tasks)
    evs = {}
    for name, learners in subjects.items():
        learners = list(learners)
        if not learners:
            raise ValueError(f"no learner for {name!r}")
        evs[name] = [evaluate(lr, tasks, spec=spec, n_envs=n_envs, segments=segments, label=f"{name}/{k}", boot_seed=k,
                              reward_mode=reward_mode) for k, lr in enumerate(learners)]
    return combine_evaluations(evs, reference=reference, alpha=alpha, label=label)


def combine_evaluations(evaluations: Mapping[str, Sequence[Mapping]], *, reference: str | None = None,
                        alpha: float = 0.05, label: str = "") -> dict:
    """The frozen baseline table from `evaluate` results: `{learner name: [one result per seed]}`, all on the
    same tasks in the same order (else `ProtocolViolation`). Per learner: the per-task mean over its seeds with
    intervals over tasks, the success probability, the within-episode profile, per family and per transfer
    kind, its evaluation cost and the failure diagnosis of its episodes. Between learners: paired differences on
    score and success (to `reference` if named, otherwise all pairs), all adjusted together with Holm's method."""
    per, values, succ = {}, {}, {}
    ids, vers = None, None
    for name, evs in evaluations.items():
        evs = list(evs)
        for e in evs:
            if ids is None:
                ids, vers = e["tasks"], e.get("versions")
            if e["tasks"] != ids:
                raise ProtocolViolation("the learners were not evaluated on identical tasks in identical order")
            if e.get("versions") != vers:
                raise ProtocolViolation("the evaluations were made under different task or contract versions")
        recs = [r for e in evs for r in e["records"]]
        values[name] = _per_task_mean([_scores(e) for e in evs])
        succ[name] = _per_task_mean([_succ(e) for e in evs])
        agg = summarise(recs)
        per[name] = {"learner": evs[0]["learner"], "seeds": [e["learner"]["seed"] for e in evs],
                     "score": S.mean_ci(values[name], n_boot=N_BOOT), "iqm": S.iqm_ci(values[name], n_boot=N_BOOT),
                     "success": agg["success"], "adaptation": agg["adaptation"], "segments": agg["segments"],
                     "by_family": _group(recs, "family"), "by_transfer": _group(recs, "transfer"),
                     "scores": values[name].tolist(), "success_rate_per_task": succ[name].tolist(),
                     "records": recs, "cost": cost_row(_sum_costs([e["cost"] for e in evs])),
                     "frozen_unchanged": all(e["frozen"]["unchanged"] is not False for e in evs),
                     "diagnosis": diagnose_records(recs)}
    rows = []
    if len(values) > 1:
        rows = _pairwise(values, "score", "tasks", reference) + _pairwise(succ, "success", "tasks", reference)
        _holm_rows(rows, alpha)
        if reference in values:
            for r in rows:
                if r["quantity"] == "score" and r["b"] == reference:
                    v = per[r["a"]]["diagnosis"]["vs_reference"] = _vs_reference(r, reference)
                    per[r["a"]]["diagnosis"]["statements"].append(v["statement"])
    first = next(iter(per.values()))["records"] if per else []
    return _js({"protocol": "baseline", "schema": SCHEMA, "versions": vers, "label": label,
                "learners": list(evaluations), "reference": reference, "alpha": alpha, "n_tasks": len(ids or []),
                "tasks": list(ids or []), "families": sorted({r["family"] for r in first}), "per_learner": per,
                "pairwise": rows, "correction": "holm", "unit": "task (mean over seeds)", "sealed": False})


def baseline(learners: Mapping[str, Callable[[int], Any]], tasks: Sequence[Task], *, seeds: Sequence[int] = (0,),
             reference: str | None = None, **kw) -> dict:
    """The frozen baseline table: every learner as its factory builds it (untrained), one per seed, evaluated on
    the same tasks. See `evaluate_many`."""
    out = evaluate_many({name: [f(int(s)) for s in seeds] for name, f in learners.items()}, tasks,
                        reference=reference, **kw)
    out["seeds"] = [int(s) for s in seeds]
    return out


def _vs_reference(row: Mapping, reference: str) -> dict:
    """Whether a learner did better than the reference, read off the corrected pairwise row (a = the learner,
    b = the reference): the same decision the pairwise table shows, never a second, uncorrected test."""
    if row.get("significant"):
        verdict = "better than the reference" if row["mean_diff"] > 0 else "worse than the reference"
    else:
        verdict = "no better than the reference (difference not distinguishable from zero after correction)"
    return {"reference": reference, "paired": {k: row.get(k) for k in ("n", "mean_diff", "lo", "hi", "p_value",
                                                                       "p_holm", "frac_better", "low_n")},
            "verdict": verdict,
            "statement": f"against {reference} on the same tasks: {row['mean_diff']:+.3f} [{row['lo']:+.3f}, "
                         f"{row['hi']:+.3f}], p={row['p_value']:.3g} (Holm {row['p_holm']:.3g}): {verdict}"}


def _sum_costs(rows: Sequence[Mapping]) -> Cost:
    c = Cost()
    for r in rows:
        c.add(Cost.from_dict(dict(r)))
    return c


def compare(learners: Mapping[str, Callable[[int], Any]], train_tasks: Any, eval_sets: Any, budget_env_steps: int,
            eval_every: int, *, seeds: Sequence[int] = (0,), reference: str | None = None,
            spec: EnvSpec | None = None, n_envs: int = 16, update_every: int = 1, segments: int = 4,
            thresholds: Sequence[float] = (GOAL_FRACTION,), sustain: int = 2, alpha: float = 0.05,
            compute_budget_s: float | None = None, kernel_diagnose: int = 0, workdir: str | None = None,
            label: str = "", keep_runs: bool = False, reward_mode: str | None = None) -> dict:
    """Two or more learners under equal budgets: one `lifetime` per (learner, seed) with identical training
    tasks, evaluation sets, seeds and schedule, then `compare_runs`. `train_tasks` must be re-iterable (a list of
    tasks, family names, or an index-addressable stream), since every run consumes it from the start."""
    if len(learners) < 2:
        raise ValueError("a comparison needs at least two learners")
    sets = _eval_sets(eval_sets)
    src = _as_source(train_tasks)
    if not (callable(src) or isinstance(src, Sequence)):
        raise ValueError("train_tasks must be a list of tasks, family names or an index-addressable stream: every "
                         "run consumes it from the start")
    runs: dict[str, list[dict]] = {}
    subjects: dict[str, Any] = {}
    for name, factory in learners.items():
        runs[name] = []
        for s in seeds:
            learner = factory(s)
            runs[name].append(lifetime(
                learner, src, budget_env_steps, eval_every, sets, spec=spec, n_envs=n_envs,
                update_every=update_every, segments=segments, thresholds=thresholds, sustain=sustain,
                checkpoint_dir=None if workdir is None else os.path.join(workdir, name, f"seed_{int(s)}"),
                label=f"{name}/seed{s}", reward_mode=reward_mode))
            subjects[name] = learner
    out = compare_runs(runs, reference=reference, alpha=alpha, compute_budget_s=compute_budget_s, label=label)
    if kernel_diagnose:
        for name, learner in subjects.items():
            for set_name, ts in sets.items():
                failed = [r["task"] for r in runs[name][-1]["final"][set_name]["records"] if not r["success"]]
                by_id = {t.id: t for t in ts}
                out["diagnosis"][name][set_name]["kernel"] = _js(
                    [kernel_diagnosis(learner, by_id[i], spec, reward_mode) for i in failed[:int(kernel_diagnose)]])
    if keep_runs:
        out["runs"] = runs
    return out


def compare_runs(runs: Mapping[str, Sequence[Mapping]], *, reference: str | None = None, alpha: float = 0.05,
                 compute_budget_s: float | None = None, label: str = "") -> dict:
    """Paired statistics over `lifetime` results: `{learner name: [one result per seed]}`.

    Checks first that the runs are comparable: identical evaluation tasks (else `ProtocolViolation`), identical
    training tasks, step grid and seeds (recorded in `checks`; `equal_budgets` is False if any differs).

    Quantities compared, per evaluation set, on per-task means over seeds: the final frozen score and success at
    the common step budget, the area under the learning curve against steps, and - where learner compute is
    measurable - the score at a common compute budget (`compute_budget_s`, default the smallest total any
    learner used, so that no learner is extrapolated). All p-values of the comparison form one family and are
    adjusted with Holm's method. With `reference`, each learner is compared to it; otherwise all pairs.

    `diagnosis[learner][set]` says what the final evaluation's episode records establish about its failures
    (`diagnose_records`), and against the reference whether it did any better."""
    names = list(runs)
    if len(names) < 2:
        raise ValueError("a comparison needs at least two learners")
    first = runs[names[0]][0]
    set_names = list(first["points"][0]["eval"])
    checks = {"same_eval_tasks": True, "same_train_tasks": True, "same_step_grid": True, "same_seeds": True,
              "same_versions": True, "all_complete": True, "frozen_unchanged": True}
    seeds0 = [r["learner"]["seed"] for r in runs[names[0]]]
    for name in names:
        if [r["learner"]["seed"] for r in runs[name]] != seeds0:
            checks["same_seeds"] = False
        for r in runs[name]:
            if r["config"]["eval_sets"] != first["config"]["eval_sets"]:
                checks["same_eval_tasks"] = False
            if r.get("versions") != first.get("versions"):
                checks["same_versions"] = False
            if r["train_tasks"]["digest"] != first["train_tasks"]["digest"]:
                checks["same_train_tasks"] = False
            if [p["env_steps"] for p in r["points"]] != [p["env_steps"] for p in first["points"]]:
                checks["same_step_grid"] = False
            checks["all_complete"] &= bool(r["complete"])
            checks["frozen_unchanged"] &= bool(r["frozen_unchanged"])
    if not checks["same_eval_tasks"]:
        raise ProtocolViolation("the runs were not evaluated on identical task sets: they cannot be paired")
    if not checks["same_versions"]:
        raise ProtocolViolation("the runs were measured under different task or contract versions: the same task "
                                "id does not name the same problem in both, so they cannot be paired")
    checks["equal_budgets"] = bool(checks["same_train_tasks"] and checks["same_step_grid"] and checks["same_seeds"])
    budget = float(first["config"]["budget_env_steps"])
    totals = {n: float(np.mean([r["points"][-1]["compute_s"] for r in runs[n]])) for n in names}
    measurable = all(v > 1e-6 for v in totals.values())
    c_budget = float(compute_budget_s) if compute_budget_s is not None else min(
        min(r["points"][-1]["compute_s"] for r in runs[n]) for n in names)
    per: dict[str, Any] = {}
    rows: list[dict] = []
    eq_compute: dict[str, Any] = {"measurable": bool(measurable), "budget_s": c_budget, "totals_s": totals,
                                  "per_learner": {}}
    diag: dict[str, Any] = {n: {} for n in names}
    for name in names:
        per[name] = {"learner": runs[name][0]["learner"], "final": {}, "auc_steps": {}, "auc_success_steps": {},
                     "curves": {}, "to_threshold": {},
                     "cost": {c: float(np.mean([r["cost"]["train"][c] for r in runs[name]])) for c in COST_COLUMNS}}
        eq_compute["per_learner"][name] = {}
    for set_name in set_names:
        fin, fin_s, auc_s, auc_ss, at_c = {}, {}, {}, {}, {}
        for name in names:
            rs = runs[name]
            mats = [curve_matrix(r["points"], set_name) for r in rs]
            smats = [curve_matrix(r["points"], set_name, "success") for r in rs]
            fin[name] = _per_task_mean([y[-1] for _, y in mats])
            fin_s[name] = _per_task_mean([y[-1] for _, y in smats])
            auc_s[name] = _per_task_mean([[S.auc(x, y[:, j], x_max=budget) for j in range(y.shape[1])] for x, y in mats])
            auc_ss[name] = _per_task_mean([[S.auc(x, y[:, j], x_max=budget) for j in range(y.shape[1])] for x, y in smats])
            seed_means = [float(np.nanmean(y[-1])) for _, y in mats]
            per[name]["final"][set_name] = {
                "score": S.mean_ci(fin[name], n_boot=N_BOOT), "iqm": S.iqm_ci(fin[name], n_boot=N_BOOT),
                "success": S.mean_ci(fin_s[name], n_boot=N_BOOT),
                "success_pooled": S.proportion_ci(int(np.nansum([y[-1] for _, y in smats])),
                                                  int(np.isfinite(np.array([y[-1] for _, y in smats])).sum())),
                "seed_sd": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
                "per_seed": seed_means}
            per[name]["auc_steps"][set_name] = S.mean_ci(auc_s[name], n_boot=N_BOOT)
            per[name]["auc_success_steps"][set_name] = S.mean_ci(auc_ss[name], n_boot=N_BOOT)
            x = mats[0][0]
            ybar = np.nanmean(np.stack([y for _, y in mats]), axis=0)
            per[name]["curves"][set_name] = {"score": _mean_curve(x, ybar),
                                             "success": _wilson_curve(x, np.concatenate([y for _, y in smats], axis=1)),
                                             "compute_s": np.mean([[p["compute_s"] for p in r["points"]]
                                                                   for r in rs], axis=0).tolist()}
            per[name]["to_threshold"][set_name] = rs[0]["derived"][set_name]["score"]["to_threshold"] \
                if len(rs) == 1 else {k: {"steps": S.mean_ci(_f(r["derived"][set_name]["score"]["to_threshold"][k]["steps"]
                                                                for r in rs), n_boot=N_BOOT),
                                          "reached_runs": int(sum(r["derived"][set_name]["score"]["to_threshold"][k]["steps"]
                                                                  is not None for r in rs))}
                                      for k in rs[0]["derived"][set_name]["score"]["to_threshold"]}
            if measurable:
                vals, reached, steps_at = [], True, []
                for r, (x_, y) in zip(rs, mats):
                    xc = np.array([p["compute_s"] for p in r["points"]], dtype=float)
                    reached &= bool(xc[-1] >= c_budget - 1e-12)
                    vals.append([float(np.interp(c_budget, xc, y[:, j])) for j in range(y.shape[1])])
                    steps_at.append(float(np.interp(c_budget, xc, x_)))
                at_c[name] = _per_task_mean(vals)
                eq_compute["per_learner"][name][set_name] = dict(
                    S.mean_ci(at_c[name], n_boot=N_BOOT), reached=reached, env_steps_at_budget=float(np.mean(steps_at)))
            final_recs = [rec for r in rs for rec in r["final"][set_name]["records"]]
            diag[name][set_name] = diagnose_records(final_recs)
        rows += _pairwise(fin, "final_score@equal_steps", set_name, reference)
        rows += _pairwise(fin_s, "final_success@equal_steps", set_name, reference)
        rows += _pairwise(auc_s, "auc_score_vs_steps", set_name, reference)
        rows += _pairwise(auc_ss, "auc_success_vs_steps", set_name, reference)
        if measurable:
            rows += _pairwise(at_c, "score@equal_compute", set_name, reference)
    _holm_rows(rows, alpha)
    if reference in names:
        for r in rows:
            if r["quantity"] == "final_score@equal_steps" and r["b"] == reference:
                v = diag[r["a"]][r["set"]]["vs_reference"] = _vs_reference(r, reference)
                diag[r["a"]][r["set"]]["statements"].append(v["statement"])
    return _js({"protocol": "compare", "schema": SCHEMA, "versions": first.get("versions"), "label": label,
                "learners": names, "seeds": seeds0,
                "reference": reference, "alpha": alpha, "budget_env_steps": budget, "sets": set_names,
                "checks": checks, "per_learner": per, "pairwise": rows, "equal_compute": eq_compute,
                "diagnosis": diag, "correction": "holm", "unit": "task (mean over seeds)"})
