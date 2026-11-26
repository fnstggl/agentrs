"""The canonical environment factory of CAM Unified (redesign plan 2026-11-26 §3, W2).

    make_env(kind, specs | tasks, n_envs, reward_mode, split) -> EnvSet(adapter, operator, labels)

* ``adapter``  - the actor side (`interfaces.EnvAdapter`): the only object a runner hands to the agent. Built with
  ``deliver_results=True``: every executed action's direct result is delivered (``caused_by`` = the action id) at
  the head of the next `ObservationBatch`, whichever loop drives it; ``adapter.delivers_results`` is True so a loop
  that also merges `ExecutionResult.observations` must not add them again (`merge_results` does it idempotently).
* ``operator`` - the operator side (checkpoints, clock bookkeeping): `ExperienceOperator`, `OpenWorldOperator`, or
  for process-isolated OpenWorld the checkpoint-only `RemoteOperator`. Never handed to the learner's agent.
* ``labels``   - the `interfaces.LabelSource` (``take_labels() -> [labels.LabelRecord]``, each label once):
  `ExperienceOperator` itself, `OpenWorldEvaluator` (wrapping its `RewardGate`) or its `RemoteEvaluator` proxy.
  Drain it into the trainer-only `labels.LabelStore` (`labels.drain`); the agent never holds it.

Kinds: ``"openworld"`` (Crucible OpenWorld kernel worlds; `tasks` = GoalContract-like objects or names, see
`resolve_task`; process-isolated by default), ``"rapid"`` / ``"mech"`` (Crucible 5.0 Tier 1 through
`experience.ExperienceOperator`), ``"integrity"`` (the CAM-side reward-integrity worlds of `integrity.py`),
``"kernel"`` (Tier 0 world families through the experience interface), ``"experience"`` (mixed `WorldSpec`s) and
``"capability"`` (one diagnostic env of `capabilities.envs` by name - chain, slot, probe, cue, delayfx, interact,
switch - with `n_envs` rows, switched to observable payouts and result delivery; its `ScoreEvaluator` is the gated
reward / label source; diagnostics have no held-out families, so the split guard does not apply).

Reward mode (``reward_mode``): ``"observation_only"`` is the default and the only mode accepted unless the caller
declares a baseline condition in `allow_reward_modes` (``"legacy"`` / ``"legacy_shaped_reward"`` = the shaped
evaluator reward, ``"verified"`` / ``"verified_terminal_outcome"`` = 1.0 once at verified success).

Split guard (``split``): ``"train"`` (default) is a training stream - every world is checked before it is created
and a validation / test family, a non-training seed (above Crucible's ``TRAIN_SEED_MAX``), a (family, seed) pair a
task protocol declares held out, or a task labelled val/test raises `worldgen.HeldOutFamilyError`. Rapid families
use Crucible's own declared split (`crucible.rapid.families`); mechanism, OpenWorld and kernel families use
`worldgen.SplitGuard` with every protocol's pair declarations. ``"val"`` / ``"test"`` are evaluation streams: any
family is allowed, and their labels carry ``meta["stream"] = "eval"`` so a training `LabelStore` refuses them.
"""
from __future__ import annotations

import functools
from typing import Any, Optional, Sequence

from .experience import LEGACY, OBSERVATION_ONLY, VERIFIED, ExperienceOperator, WorldSpec
from .labels import EVAL_STREAM, TRAIN_STREAM
from .ow_convert import is_delivered_result, result_key

KINDS = ("openworld", "rapid", "mech", "integrity", "kernel", "experience", "capability")
SPLITS = ("train", "val", "test")
MODE_ALIASES = {"legacy": LEGACY, LEGACY: LEGACY, "verified": VERIFIED, VERIFIED: VERIFIED,
                OBSERVATION_ONLY: OBSERVATION_ONLY, "observation-only": OBSERVATION_ONLY}
TIER = {"rapid": 1, "mech": 1, "integrity": -1, "kernel": 0}


class EnvSet(tuple):
    """``(adapter, operator, labels)`` - unpacks as the plan's triple - with the stream's bookkeeping as attributes
    (kind, split, reward_mode, n_envs, guarded: worlds checked by the split guard). Operator side only."""

    def __new__(cls, adapter, operator, labels, *, kind: str, split: str, reward_mode: str, guarded: int):
        self = super().__new__(cls, (adapter, operator, labels))
        self.kind, self.split, self.reward_mode, self.guarded = kind, split, reward_mode, int(guarded)
        self.n_envs = len(getattr(adapter, "env_ids", None) or []) or int(getattr(adapter, "n", 0) or 0)
        return self

    @property
    def adapter(self):
        return self[0]

    @property
    def operator(self):
        return self[1]

    @property
    def labels(self):
        return self[2]

    def close(self) -> None:
        self[0].close()
        close = getattr(self[1], "close", None)
        if callable(close) and self.kind == "openworld":
            close()


def reward_mode_of(mode: Optional[str]) -> str:
    m = OBSERVATION_ONLY if mode is None else MODE_ALIASES.get(str(mode))
    if m is None:
        raise ValueError(f"unknown reward_mode {mode!r}; one of {sorted(MODE_ALIASES)}")
    return m


# ============================================================================================ split guard
@functools.lru_cache(maxsize=1)
def protocol_pairs() -> dict:
    """{(world family, seed): split} declared by every task protocol (most restrictive split per pair)."""
    from .protocols import all_protocols
    rank = {"test": 3, "heldout": 3, "val": 2, "train": 1, "dev": 0}
    out: dict = {}
    for p in all_protocols().values():
        for k, sp in p.pair_declarations().items():
            if k not in out or rank.get(sp, 3) > rank.get(out[k], 3):
                out[k] = sp
    return out


def default_guard():
    """`worldgen.SplitGuard` with every protocol's (family, seed) declarations."""
    from .worldgen import SplitGuard
    return SplitGuard(extra_pairs=protocol_pairs())


def rapid_split(family: str) -> Optional[str]:
    from crucible.rapid.families import FAMILIES
    f = FAMILIES.get(family)
    return None if f is None else f.split


def guard_spec(spec: WorldSpec, guard) -> None:
    """Refuse a held-out world spec in a training stream (raises `worldgen.HeldOutFamilyError`)."""
    from .worldgen import TRAINABLE_SPLITS, HeldOutFamilyError
    kind, _, fam = spec.name.partition(":") if ":" in spec.name else ("", "", spec.name)
    if kind == "integrity":                        # CAM-side diagnostic worlds (Exp F): no held-out families
        return
    if kind == "rapid":
        sp = rapid_split(fam)
        if sp is None:
            raise HeldOutFamilyError(f"rapid family {fam!r} is unknown; refusing it in a training stream")
        if sp not in TRAINABLE_SPLITS:
            raise HeldOutFamilyError(f"rapid family {fam!r} is a {sp} family; it must never enter a training stream")
        if int(spec.seed) > guard.seed_limit:
            raise HeldOutFamilyError(f"rapid family {fam!r} seed {spec.seed} is outside the training seed range "
                                     f"(0..{guard.seed_limit})")
        guard.checked += 1
        return
    guard.check(fam, int(spec.seed), what=f"{kind or 'kernel'} family")


def guard_task(task: Any, guard) -> None:
    guard.check_task(task)


# ============================================================================================ specs and tasks
def resolve_spec(kind: str, s: Any, i: int = 0) -> WorldSpec:
    """A WorldSpec from a WorldSpec, ``"<kind>:<family>[#seed]"`` or a bare ``"<family>[#seed]"`` of `kind`."""
    if isinstance(s, WorldSpec):
        return s
    if not isinstance(s, str):
        raise TypeError(f"world spec {s!r}: give a WorldSpec or a name string")
    name, _, seed = s.partition("#")
    if ":" not in name:
        if kind not in TIER:
            raise ValueError(f"spec {s!r} needs a '<kind>:' prefix for kind {kind!r}")
        name = name if kind == "kernel" else f"{kind}:{name}"
    k = name.split(":", 1)[0] if ":" in name else "kernel"
    return WorldSpec(TIER.get(k, 0), name, int(seed) if seed else 0)


def expand(items: Sequence[Any], n_envs: Optional[int]) -> list:
    """`n_envs` items cycling through `items`; repeated WorldSpecs get consecutive seeds (distinct worlds)."""
    items = list(items)
    if not items:
        raise ValueError("no environments given")
    n = len(items) if n_envs is None else int(n_envs)
    out = []
    for i in range(n):
        x = items[i % len(items)]
        k = i // len(items)
        if k and isinstance(x, WorldSpec):
            x = WorldSpec(x.tier, x.name, x.seed + k, x.kw)
        out.append(x)
    return out


def resolve_task(t: Any, i: int = 0) -> Any:
    """An OpenWorld task: a GoalContract-like object as is; ``"open_economy[#seed]"`` (ow_tiny open economy, 8
    days; default seed 100000 + i, the CAM development range); ``"<protocol>:<split>:<k>"`` (task k of a protocol's
    declared split, with its seat and participants)."""
    if not isinstance(t, str):
        return t
    name, _, seed = t.partition("#")
    if name == "open_economy":
        from .protocols.ow1 import open_economy_task
        return open_economy_task(seed=int(seed) if seed else 100_000 + i)
    parts = name.split(":")
    if len(parts) == 3:
        from .protocols import load
        p = load(parts[0])
        fam, s = p.splits[parts[1]][int(parts[2])]
        return p.make_task(fam, s)
    raise ValueError(f"unknown OpenWorld task name {t!r}")


# ============================================================================================ factory
def make_env(kind: str, specs: Optional[Sequence[Any]] = None, *, tasks: Optional[Sequence[Any]] = None,
             n_envs: Optional[int] = None, reward_mode: str = OBSERVATION_ONLY, split: str = "train",
             allow_reward_modes: Sequence[str] = (OBSERVATION_ONLY,), isolated: bool = True,
             deliver_results: bool = True, guard=None, seed: int = 0, salt: Optional[str] = None,
             evaluator_kw: Optional[dict] = None, compute_references: bool = True, **kw) -> EnvSet:
    """See the module docstring. `specs` (experience kinds) or `tasks` (openworld; `specs` is accepted as an alias).
    `isolated` (openworld): worlds, operator and evaluator in a spawned process (learner runs); False keeps them in
    this process (tests, scripted controls). `guard`: a `worldgen.SplitGuard` (default: `default_guard()`).
    Other keyword arguments go to `open_world` / `open_world_isolated` (openworld) or the `WorldSpec` builder."""
    if kind not in KINDS:
        raise ValueError(f"unknown env kind {kind!r}; one of {KINDS}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    mode = reward_mode_of(reward_mode)
    allowed = {reward_mode_of(m) for m in allow_reward_modes}
    if mode not in allowed:
        raise ValueError(f"reward_mode {mode!r} is not allowed: declare it in allow_reward_modes for a baseline "
                         f"condition; CAM runs use {OBSERVATION_ONLY!r}")
    stream = TRAIN_STREAM if split == "train" else EVAL_STREAM
    if kind == "capability":
        return _capability(specs if specs is not None else tasks, n_envs, mode, split, seed, kw)
    g = guard if guard is not None else (default_guard() if split == "train" else None)
    if kind == "openworld":
        items = list(tasks if tasks is not None else (specs or []))
        resolved = [resolve_task(t, i) for i, t in enumerate(items)]
        if n_envs is not None and len(resolved) not in (1, int(n_envs)):
            resolved = expand(resolved, n_envs)
        if split == "train":
            for t in resolved:
                guard_task(t, g)
        ev_kw = {"reward_mode": mode, "stream": stream, **dict(evaluator_kw or {})}
        if isolated:
            from .openworld import open_world_isolated
            ad, op, ev = open_world_isolated(resolved, n_envs, evaluator=ev_kw, seed=seed, reward_mode=mode,
                                             deliver_results=deliver_results, strip_goal_meta=True, **kw)
        else:
            from .openworld import open_world
            from .openworld_eval import OpenWorldEvaluator
            ad, op = open_world(resolved, n_envs, seed=seed, reward_mode=mode, deliver_results=deliver_results,
                                strip_goal_meta=True, **kw)
            ev = OpenWorldEvaluator(op, **ev_kw)
        return EnvSet(ad, op, ev, kind=kind, split=split, reward_mode=mode, guarded=len(resolved) if g else 0)
    items = list(specs if specs is not None else (tasks or []))
    ws = expand([resolve_spec(kind, s, i) for i, s in enumerate(items)], n_envs)
    if kind in ("rapid", "mech", "integrity", "kernel"):
        for w in ws:
            k = w.name.split(":", 1)[0] if ":" in w.name else "kernel"
            if k != kind:
                raise ValueError(f"spec {w.name!r} is not of kind {kind!r} (use kind='experience' for a mix)")
    if split == "train":
        for w in ws:
            guard_spec(w, g)
    if kw:
        raise TypeError(f"unexpected arguments for kind {kind!r}: {sorted(kw)}")
    op = ExperienceOperator(ws, reward_mode=mode, allow_reward_modes=tuple(allowed),
                            compute_references=compute_references, salt=salt, stream=stream)
    return EnvSet(op.adapter(deliver_results=deliver_results), op, op, kind=kind, split=split, reward_mode=mode,
                  guarded=len(ws) if g else 0)


def _capability(specs, n_envs, mode, split, seed, kw) -> EnvSet:
    from ..capabilities import envs as CE
    classes = {c.name: c for c in (CE.ChainEnv, CE.MechanismEnv, CE.ProbeEnv, CE.DelayedCueEnv, CE.DelayEffectEnv,
                                   CE.InteractionEnv, CE.SwitchEnv)}
    names = [specs] if isinstance(specs, str) else list(specs or [])
    if len(names) != 1 or names[0] not in classes:
        raise ValueError(f"capability kind takes one env name of {sorted(classes)}, got {specs!r}")
    env = CE.observation_only(classes[names[0]](int(n_envs or 1), seed=seed, **kw))
    ev = CE.ScoreEvaluator(env, reward_mode=mode)
    return EnvSet(env, env, ev, kind="capability", split=split, reward_mode=mode, guarded=0)


# ============================================================================================ loop helper
def merge_results(results: Sequence[Any], batch) -> list:
    """[B] next-decision observation lists: the batch's observations, preceded by each row's direct-result
    observations unless the adapter already delivered them (idempotent: a delivered copy - `ow_convert.
    delivered_copy` - or an identical observation is never added twice). For loops that merge results themselves."""
    out = []
    for b, olist in enumerate(batch.observations):
        olist = list(olist)
        have = {result_key(o) for o in olist if is_delivered_result(o) or o.caused_by}
        r = results[b] if b < len(results) else None
        extra = [o for o in (r.observations if r is not None else []) if result_key(o) not in have]
        out.append(extra + olist)
    return out
