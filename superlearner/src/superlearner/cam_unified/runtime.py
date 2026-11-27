"""The canonical CAM Unified runtime: ONE act -> execute -> advance loop (redesign plan 2026-11-26 §2; owner W1).

    tools   = env.available_tools()
    actions = agent.act(obs, tools)            authorised: every action carries a cam_eval stamp when the DOM
                                               decision path is on (`dom_decision_path(cfg)`), else ActionNotAuthorised
    results = env.execute(actions)
    nxt     = env.advance_time(duration + think)                      SMDP clock (think charged to idle live rows)
    next_obs = merge_by_seq(results[b].observations, nxt.observations[b] [, collect_events])   tool results delivered
    agent.observe(Transition(records, next_obs, rewards, dones, dt))  rewards is None unless reward_mode == "legacy"
    learner.on_step(transition)
    labels.take_labels() -> label_store.append                       learner side only (the agent never holds them)
    every update_every rounds: learner.step(); every cfg.meta_every updates: learner.meta_step()
    checkpoints (operator + evaluator + agent + learner + loop state; every_days and/or every_updates), transcript

`run_lifetime` is the only function of the package that contains this loop (checked by
tests/cam_unified/integration/test_single_loop.py). `envs.harness.run_lifetime`, `run_steps` (below),
`experiments.mechrun.run_batch`, `capabilities.loop.train_agent`, the Stage-1 runner and `experiments/train.py`
are wrappers over it.

Reward channels (`reward_mode`; None = the agent's cfg.reward_mode, else "observation_only"):
* "observation_only" (default of every DOM path): the evaluator's verified increments stay operator-side
  (`LifetimeResult.returns`, `rounds[*]["rewards"]`); `Transition.rewards is None`, `EventRecord.reward == 0.0` on
  every record the agent receives, and verified outcomes reach the learner only as labels in the LabelStore.
* "sparse_verified": a labelled baseline condition whose actor reward is the verified terminal label (the learner
  joins it from the LabelStore); per-step delivery as in observation_only.
* "legacy": the pre-seam behaviour - the verified increment is delivered to the agent (Transition.rewards, records).
  The compatibility wrappers (`envs.harness.run_lifetime`, `run_steps`, `mechrun.run_batch`) default to it for
  agents without a config, so pre-seam callers and their tests see the same rewards.

Legacy reward source (`reward_source`): "take_rewards" (evaluator.take_rewards() after the advance; the harness),
"score_delta" (increments of evaluator.evaluate_goal(b).score; run_steps), a callable(next ObservationBatch) -> [B],
or "auto" (take_rewards when the evaluator has it, else score_delta).

Versions (cam-event/4): every EventRecord of the loop carries extra["versions"] = event_versions(policy, reward_model,
evaluator, env, code), extra["schema"] = "cam-event/4" and extra["reward_mode"]; CamUnifiedAgent copies them into the
records it archives.

Semi-Markov duration: Transition.dt and EventRecord.extra["dt_days"] are the simulated time that actually elapsed
(next observation time minus this one); extra["dt_requested_days"] is the duration the decision asked for (result
duration + think, or think alone for an idle live row, 0 for a done row).

Checkpoints and resume (`CheckpointSpec`): written atomically under `dir/ckpt_<n>` (operator.pkl, evaluator.pkl,
agent/, learner.pt, loop.pkl) with `dir/LATEST`; `resume=True` continues from the latest one in this or another
process (the world store must be durable) after truncating the transcript to the checkpoint's byte offset. Resume
needs the operator (environment state); update-triggered checkpoints without an operator save agent + learner + loop
state only.
"""
from __future__ import annotations

import functools
import json
import os
import pickle
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from .interfaces import (EVENT_SCHEMA, EnvAdapter, EventRecord, EvidenceKind, GoalEvaluator, ObservationBatch, ObsKind,
                         Provenance, event_versions, new_id)

FROM_START = -1                        # event cursor sentinel: nothing delivered yet (exclusive since)
SEED_STRIDE = 1_000_003
DAY = 86400.0
REWARD_MODES = ("legacy", "observation_only", "sparse_verified")
CAM_EVAL_KEY = "cam_eval"              # Action.proposal["cam_eval"]: CAM's pre-execution evaluation of the action
LIFETIME_FORMAT = "cam-unified-lifetime/1"


# ============================================================================ records of a run

@dataclass
class ComputeAccount:
    """Spec Part 22: GPU time, CPU time, inference tokens, environment steps, decisions, simulated time, memory.

    gpu_span_s: CUDA-event span of act/observe/update (includes GPU idle time while the CPU launches work: an upper
    bound, close to wall time for small models; only with gpu_timing). gpu_busy_s: summed CUDA kernel time from
    torch.profiler (only with profile_gpu=True; None otherwise)."""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    act_s: float = 0.0
    env_s: float = 0.0
    observe_s: float = 0.0
    update_s: float = 0.0
    gpu_span_s: float = 0.0
    gpu_busy_s: Optional[float] = None
    gpu_kernels: Optional[int] = None
    env_steps: int = 0                  # environment-step slots (B per loop iteration)
    decisions: int = 0                  # executed (non-None) actions
    sim_days: float = 0.0               # simulated time requested (durations + think), summed over environments
    updates: int = 0
    episodes: int = 0
    peak_gpu_mem_bytes: int = 0
    max_rss_bytes: int = 0
    inference_tokens: Optional[int] = None    # semantic tokens encoded + generated, when the components count them
    archive_records: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    """run_steps' report (a view of the LifetimeResult)."""
    steps: list = field(default_factory=list)       # per round: rewards, ok flags, tools, sim time
    updates: list = field(default_factory=list)     # learner.step() metrics
    returns: list = field(default_factory=list)     # [B] accumulated verified return (operator side)
    goal_scores: list = field(default_factory=list)  # [B] last evaluator score (scalar only)
    episode_seeds: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)     # every budget that left data out (agent.drop_report())
    compute: ComputeAccount = field(default_factory=ComputeAccount)


@dataclass
class LifetimeResult:
    steps: int                         # rounds run (one decision per live environment per round)
    t_days: list                       # [B] simulated time at the end (days)
    elapsed_days: list                 # [B] simulated days lived in the current episode
    returns: list                      # [B] total verified return (operator record; never given to the agent
                                       # unless reward_mode == "legacy")
    final: list                        # [B] IndependentGoalResult dicts (operator record, includes details)
    success_day: list                  # [B] simulated day of verified success (None = not achieved)
    accounting: dict
    updates: list = field(default_factory=list)
    transcript: Optional[str] = None
    checkpoints: list = field(default_factory=list)
    resumed_from: Optional[str] = None
    # v4 (canonical loop)
    reward_mode: str = "observation_only"
    episodes: int = 1
    episode_seeds: list = field(default_factory=list)
    goal_scores: list = field(default_factory=list)  # [B] last evaluate_goal score (reward_source "score_delta")
    compute: dict = field(default_factory=dict)      # ComputeAccount.to_dict()
    rounds: list = field(default_factory=list)       # per-round rows (keep_rounds=True)
    meta_updates: list = field(default_factory=list)
    labels_taken: int = 0              # labels released by the LabelSource
    labels_appended: int = 0           # new labels written to the LabelStore (duplicates are refused by the store)
    versions: dict = field(default_factory=dict)     # extra["versions"] of the last round
    authorisation: dict = field(default_factory=dict)  # {"required", "checked", "source"}
    dropped: dict = field(default_factory=dict)      # agent.drop_report() at the end
    interrupted: bool = False          # stop_after_decisions ended the call (nothing was closed or flushed)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckpointSpec:
    """Where and when the canonical loop checkpoints. every_days: when the least advanced environment has lived
    another `every_days` simulated days; every_updates: after every k-th successful learner update. `operator`
    (checkpoint(path) / restore(path)) is required for resume and for day-based checkpoints."""
    dir: Optional[str]
    every_days: Optional[float] = None
    every_updates: Optional[int] = None
    operator: Any = None
    resume: bool = False
    ops: Any = None                    # memory.checkpoint.CheckpointOps: weights + trainer + experience of an agent
                                       # without save/load (System.ops); full-lifetime save / load


@dataclass
class RoundInfo:
    """What `on_round` receives after every round (operator-side view: `rewards` are the verified increments even
    when they were not delivered to the agent)."""
    index: int                         # 0-based round index of this call / lifetime
    step: int                          # 1-based step counter (as on_step)
    episode: int
    transition: Any                    # the Transition the agent observed
    actions: list
    results: list
    rewards: list                      # operator-side verified increments
    next_obs: ObservationBatch
    dt_requested: list
    update: Optional[dict] = None      # learner.step() output when an update ran this round
    meta: Optional[dict] = None
    row: dict = field(default_factory=dict)


class ActionNotAuthorised(RuntimeError):
    """An action reached the execution boundary without CAM's pre-execution evaluation stamp while the DOM decision
    path is on (plan §2 step 3)."""


# ============================================================================ helpers (merge, seeds, versions)

def merge_by_seq(seen_seq: int, *lists) -> tuple:
    """Concatenate observation lists, dropping environment events already delivered: seq <= seen_seq (an earlier
    delivery), or a seq that an earlier list of this merge already delivered (the same event from advance_time and
    collect_events). Several observations of ONE event share its seq inside one list (an adapter expands an event
    into its numbers, e.g. MechWorld's tick readings) and are all kept; observations without a seq (tool results,
    summaries) are always kept. -> (observations, new seen seq). Moved here from dist/actor.py (plan §2 step 5);
    dist/actor's copy deduplicated within a list as well and so dropped all but the first number of such events."""
    out, top, here = [], seen_seq, set()
    for lst in lists:
        this = set()
        for o in lst or []:
            if o.seq is not None:
                if o.seq <= seen_seq or o.seq in here:
                    continue
                this.add(o.seq)
                top = max(top, o.seq)
            out.append(o)
        here |= this
    return out, top


_merge = merge_by_seq                  # pre-v4 private name


def drop_delivered(res_obs, *delivered) -> list:
    """The direct-result observations of `ExecutionResult.observations` that the adapter has NOT already delivered in
    the next batch (`delivered`: that row's next observations / events). W2's adapters deliver each direct result at
    the head of the next ObservationBatch (`envs.ow_convert.delivered_copy`, same `result_key`); merging the
    ExecutionResult copy as well gave the agent every result twice (W2 integration finding, wave 3). Keyed by content,
    so it holds whether or not the adapter declares `delivers_results`."""
    if not res_obs:
        return []
    from .envs.ow_convert import is_delivered_result, result_key
    have = {result_key(o) for lst in delivered for o in (lst or []) if is_delivered_result(o)}
    if not have:
        return list(res_obs)
    return [o for o in res_obs if result_key(o) not in have]


def _max_seq(obs) -> int:
    return max([o.seq for o in obs if o.seq is not None] or [FROM_START])


def _initial_cursors(env, ob: ObservationBatch) -> list:
    if hasattr(env, "cursors") and callable(env.cursors):
        return list(env.cursors())
    return [_max_seq(obs) for obs in ob.observations]


def derived_seeds(base: Optional[Sequence[int]], episode: int, B: int) -> Optional[list]:
    """Seeds of episode `episode`: the given seeds for episode 0, distinct derived seeds afterwards."""
    if episode == 0:
        return None if base is None else list(base)
    base = list(base) if base is not None else list(range(B))
    return [int(s) + SEED_STRIDE * episode for s in base]


def _tokens_used(agent) -> Optional[int]:
    tot, any_ = 0, False
    model = getattr(agent, "model", None)
    for comp in (getattr(model, "tokenizer", None), getattr(model, "action_generator", None),
                 getattr(model, "semantic", None)):
        for attr in ("tokens_encoded", "tokens_generated"):
            v = getattr(comp, attr, None)
            if isinstance(v, int):
                tot += v
                any_ = True
    return tot if any_ else None


@functools.lru_cache(maxsize=1)
def code_version() -> Optional[str]:
    """The git commit of this checkout (EventRecord.extra["versions"]["code"]); None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def agent_config(agent):
    """The CamConfig behind an agent (CamUnifiedAgent.config, HarnessAgent.agent.config), or None."""
    for obj in (agent, getattr(agent, "agent", None)):
        cfg = getattr(obj, "config", None) if obj is not None else None
        if cfg is not None and hasattr(cfg, "reward_mode"):
            return cfg
    return None


def agent_reward_mode(agent) -> Optional[str]:
    cfg = agent_config(agent)
    return None if cfg is None else str(cfg.reward_mode)


def legacy_default_mode(agent, evaluator) -> str:
    """The reward mode of the pre-v4 compatibility wrappers: the agent's configured mode, else "legacy" when an
    evaluator supplies rewards (what those callers always got), else "observation_only"."""
    return agent_reward_mode(agent) or ("legacy" if evaluator is not None else "observation_only")


def dom_decision_path(cfg) -> bool:
    """True when the config asks for the DOM decision bottleneck (intent tokens or K > 1 evaluated candidates):
    then every executed action must carry CAM's pre-execution evaluation stamp."""
    if cfg is None:
        return False
    return int(getattr(cfg, "n_intent", 0) or 0) > 0 or int(getattr(cfg, "n_candidates", 1) or 1) > 1


def cam_eval_stamp(action) -> Optional[dict]:
    """CAM's pre-execution evaluation stamp of an action: Action.proposal["cam_eval"] (v4 reserved key; W5's
    decision.select.attach_stamp writes q, k (= chosen), n_candidates, gen_logp, sel_logp, sel_logp_all, probs,
    vetoed, tau, greedy, policy_version, psi_version, versions, candidates and an HMAC digest), or the legacy
    logp_parts["cam_eval"]. None when absent or empty. Validity is checked by `authorise` (verify_stamp)."""
    if action is None:
        return None
    p = action.proposal if isinstance(action.proposal, dict) else {}
    st = p.get(CAM_EVAL_KEY)
    if st is None:
        st = (action.logp_parts or {}).get(CAM_EVAL_KEY)
    return st if isinstance(st, dict) and st else None


def authorise(actions: Sequence, step: int) -> int:
    """Refuse execution of any action without a VALID cam_eval stamp: present, and its HMAC digest matching the
    action (W5 `decision.select.verify_stamp`: forged, tampered or stale-content stamps are refused, not only missing
    ones; actor processes that execute each other's decisions share `decision.select.set_stamp_key`).
    -> number of actions checked."""
    from .decision.select import verify_stamp
    n = 0
    for b, a in enumerate(actions):
        if a is None:
            continue
        n += 1
        if cam_eval_stamp(a) is None:
            raise ActionNotAuthorised(
                f"round {step}, row {b}: action {a.tool!r} (source {getattr(a.source, 'value', a.source)}) has no "
                f"{CAM_EVAL_KEY} stamp; with the DOM decision path on, only actions CAM evaluated before execution "
                "are executed (pass require_cam_eval=False only for a declared legacy decision path)")
        ok, why = verify_stamp(a)
        if not ok:
            raise ActionNotAuthorised(f"round {step}, row {b}: action {a.tool!r}: invalid {CAM_EVAL_KEY} stamp ({why})")
    return n


def policy_version(agent, model_version: Optional[Callable[[], int]] = None) -> int:
    if model_version is not None:
        return int(model_version())
    for attr in ("model_version", "version"):
        v = getattr(agent, attr, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return 0


def take_labels(source) -> list:
    """LabelSource.take_labels() -> [envs.labels.LabelRecord] (dicts are converted). Operators that only release
    OutcomeLabels (ExperienceOperator before W2's take_labels) are wrapped with build.OperatorLabels."""
    from .envs.labels import LabelRecord
    if not hasattr(source, "take_labels"):
        raise TypeError(f"{type(source).__name__} is not a LabelSource (no take_labels; wrap an operator with "
                        "build.OperatorLabels)")
    out = []
    for it in source.take_labels() or []:
        if isinstance(it, LabelRecord):
            out.append(it)
        elif isinstance(it, dict):
            out.append(LabelRecord(**it))
        else:
            raise TypeError(f"label of type {type(it).__name__} from {type(source).__name__}")
    return out


def label_episode_of(source, env_id) -> Optional[int]:
    """The label source's episode number for an env id (`episode_of`, W2 operators / evaluators), else None."""
    fn = getattr(source, "episode_of", None) if source is not None else None
    if not callable(fn):
        return None
    try:
        e = fn(env_id)
    except Exception:            # an operator that cannot answer leaves label_join's archive-order fallback
        return None
    return None if e is None else int(e)


def _held_by(agent, objs: Sequence) -> list:
    """Names of agent attributes (one container level deep, also through a HarnessAgent's .agent) that are one of
    `objs` - the acting agent must never hold the label source, the label store or the evaluator."""
    ids = {id(o): type(o).__name__ for o in objs if o is not None}
    if not ids:
        return []
    bad = []
    for owner_name, owner in (("agent", agent), ("agent.agent", getattr(agent, "agent", None))):
        if owner is None or not hasattr(owner, "__dict__"):
            continue
        for k, v in vars(owner).items():
            vals = list(v.values()) if isinstance(v, dict) else list(v) if isinstance(v, (list, tuple)) else [v]
            for x in [v] + vals:
                if id(x) in ids:
                    bad.append(f"{owner_name}.{k} -> {ids[id(x)]}")
    return sorted(set(bad))


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _self_ids(batch) -> list:
    out = []
    for b, olist in enumerate(batch.observations):
        sid = next((o.entity for o in olist if o.kind is ObsKind.TEXT and o.meta.get("self")), None)
        out.append(sid or batch.env_ids[b])
    return out


def _observer(agent) -> Callable:
    """agent.observe as a function of the Transition: `observe(transition)` (AgentP, CamUnifiedAgent) or, for a
    pre-v4 agent whose observe needs (results, rewards, next_obs), that call with the records' results."""
    import inspect
    fn = agent.observe
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    except (TypeError, ValueError):
        return fn
    if len(ps) >= 3:
        return lambda tr: fn([None if r is None else r.result for r in tr.records], tr.rewards, tr.next_obs)
    return fn


def _begin(agent, env_ids: list, goals: list) -> None:
    """Start an episode on the agent: `begin(env_ids, goals)` (harness agents) or `reset(env_ids)` +
    `set_goals(goals)` (CamUnifiedAgent)."""
    if hasattr(agent, "begin"):
        agent.begin(list(env_ids), list(goals))
        return
    if hasattr(agent, "reset"):
        agent.reset(list(env_ids))
    if hasattr(agent, "set_goals"):
        agent.set_goals(list(goals))


class ConcatEnv:
    """Several EnvAdapters as one batch (e.g. 16 rapid worlds + 2 OpenWorld worlds in one canonical loop): rows are
    the parts' rows in order; each part keeps its own clock, tools, events and goals."""

    def __init__(self, parts: Sequence):
        self.parts = list(parts)
        self.sizes = [int(getattr(p, "B", 0) or getattr(p, "n", 0) or 0) for p in self.parts]
        if not all(self.sizes):
            raise ValueError("ConcatEnv needs parts with a known batch size (attribute B or n)")
        self.B = sum(self.sizes)
        self.version = "+".join(str(getattr(p, "version", type(p).__name__)) for p in self.parts)

    @property
    def delivers_results(self) -> bool:
        """True when any part re-delivers direct results in its next batch (W2; the loop drops the duplicate)."""
        return any(bool(getattr(p, "delivers_results", False)) for p in self.parts)

    def _split(self, xs) -> list:
        out, i = [], 0
        for n in self.sizes:
            out.append(None if xs is None else list(xs[i:i + n]))
            i += n
        return out

    @staticmethod
    def _cat(batches: list) -> ObservationBatch:
        return ObservationBatch([e for b in batches for e in b.env_ids], [o for b in batches for o in b.observations],
                                [t for b in batches for t in b.t], [d for b in batches for d in b.done])

    def reset(self, seeds=None) -> ObservationBatch:
        return self._cat([p.reset(s) for p, s in zip(self.parts, self._split(seeds))])

    def observe(self) -> ObservationBatch:
        return self._cat([p.observe() for p in self.parts])

    def available_tools(self) -> list:
        return [t for p in self.parts for t in p.available_tools()]

    def execute(self, actions) -> list:
        return [r for p, a in zip(self.parts, self._split(actions)) for r in p.execute(a)]

    def advance_time(self, duration) -> ObservationBatch:
        return self._cat([p.advance_time(d) for p, d in zip(self.parts, self._split(duration))])

    def collect_events(self, since) -> tuple:
        got = [p.collect_events(c) for p, c in zip(self.parts, self._split(since))]
        return [e for g in got for e in g[0]], [c for g in got for c in g[1]]

    def goals(self) -> list:
        return [g for p in self.parts for g in p.goals()]

    def close(self) -> None:
        for p in self.parts:
            if hasattr(p, "close"):
                p.close()


class GuardedEnv:
    """Opt-in wrapper for real-world runs: actions are executed one row at a time and an exception raised by the
    environment for one action (e.g. an upstream engine that indexes a lever the decoded arguments omit) becomes that
    row's ExecutionResult(ok=False, error="env_exception: ...", ERROR observation) instead of ending the lifetime.
    Every such event is counted and kept (`errors`) so it is reported, never hidden; everything else is delegated."""

    def __init__(self, env):
        self.env = env
        self.errors: list = []

    def __getattr__(self, k):
        return getattr(self.env, k)

    def execute(self, actions) -> list:
        from .interfaces import ExecutionResult, Observation
        out = []
        for b, a in enumerate(actions):
            if a is None:
                out.append(None)
                continue
            row = [None] * len(actions)
            row[b] = a
            try:
                out.append(self.env.execute(row)[b])
            except Exception as e:  # noqa: BLE001 - reported per row (self.errors), the world continues
                msg = f"env_exception: {type(e).__name__}: {str(e)[:200]}"
                self.errors.append({"row": b, "tool": a.tool, "error": msg})
                out.append(ExecutionResult(ok=False, observations=[Observation(ObsKind.ERROR, msg, source=a.tool,
                                                                               caused_by=a.action_id)],
                                           error=msg, meta={"guarded": True}))
        return out


# ============================================================================ the learner side

class AgentLearner:
    """LearnerHook over the agent's own `update()` (the pre-W7 learner: CamUnifiedAgent.update samples replay,
    assembles the objectives of learn/integrated.py and steps the injected trainer). `reward_model`: the actor reward
    model in force (selfeval.reward_model.RewardModel; the W8 stub returns 0), recorded in extra["versions"]. W7's
    IntegratedLearner replaces this class behind the same protocol."""

    def __init__(self, agent, reward_model=None):
        self.agent = agent
        self.reward_model = reward_model
        self.n_steps = 0
        self.n_updates = 0

    def on_step(self, transition) -> None:
        self.n_steps += 1

    def step(self) -> dict:
        out = self.agent.update()
        out = out if isinstance(out, dict) else {"updated": False, "result": out}
        if out.get("updated"):
            self.n_updates += 1
        return out

    def meta_step(self) -> dict:
        return {"updated": False, "reason": "AgentLearner has no meta learner (W8's MetaRewardTrainer plugs in "
                                            "through the IntegratedLearner)"}

    def state_dict(self) -> Optional[dict]:
        return None                    # the agent's own save/load carries the trainer state

    def load_state_dict(self, d) -> None:
        return None


# ============================================================================ accounting

class _Meter:
    PHASES = ("act", "execute", "advance", "events", "observe", "learner", "labels", "update", "meta", "tools",
              "checkpoint", "reset")

    def __init__(self, device=None, gpu_timing: bool = False):
        self.wall = {k: 0.0 for k in self.PHASES}
        self.counts = {"decisions": 0, "actions": 0, "waits": 0, "rejected": 0, "skipped": 0}
        self.by_tool: dict = {}
        self.errors: dict = {}
        self.cpu0 = time.process_time()
        self.t0 = time.perf_counter()
        self.dev = device
        self.cuda = bool(gpu_timing and device is not None and getattr(device, "type", None) == "cuda")
        self.gpu_span = 0.0

    def timed(self, key: str, fn, *a, gpu: bool = False, **kw):
        t = time.perf_counter()
        if self.cuda and gpu:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            torch.cuda.synchronize(self.dev)
            self.gpu_span += s.elapsed_time(e) / 1000.0
        else:
            out = fn(*a, **kw)
        self.wall[key] += time.perf_counter() - t
        return out

    def summary(self, agent) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        out = {"wall_s": time.perf_counter() - self.t0, "cpu_s": time.process_time() - self.cpu0,
               "max_rss_mb": ru.ru_maxrss / 1024.0, "phase_wall_s": dict(self.wall), **self.counts,
               "by_tool": dict(sorted(self.by_tool.items(), key=lambda kv: -kv[1])), "errors": dict(self.errors)}
        if hasattr(agent, "usage"):
            try:
                out["agent"] = agent.usage()
            except Exception as e:  # noqa: BLE001 - accounting must not kill a lifetime
                out["agent"] = {"error": repr(e)}
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out["gpu_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        except Exception:  # noqa: BLE001
            pass
        return out


# ============================================================================ THE loop

def run_lifetime(agent, env: EnvAdapter, *, evaluator: Optional[GoalEvaluator] = None, labels=None, label_store=None,
                 reward_mode: Optional[str] = None, learner=None, days: Optional[float] = None,
                 max_decisions: Optional[int] = None, update_every: Optional[int] = None,
                 think_s: Optional[float] = None, checkpoint: Optional[CheckpointSpec] = None,
                 on_step: Optional[Callable] = None, seeds: Optional[Sequence[int]] = None,
                 episode_seeds: Optional[Callable[[int], Optional[Sequence[int]]]] = None,
                 reset_on_done: bool = False, stop_on_success: bool = False,
                 stop_after_decisions: Optional[int] = None, transcript: Optional[str] = None,
                 model_version: Optional[Callable[[], int]] = None, collect_events: bool = False,
                 reward_source="auto", require_cam_eval: Optional[bool] = None,
                 on_round: Optional[Callable[[RoundInfo], None]] = None,
                 on_reset: Optional[Callable[[int, ObservationBatch], None]] = None, keep_rounds: bool = False,
                 flush_reason: Optional[str] = "end_of_run", final_results: bool = True, gpu_timing: bool = False,
                 profile_gpu: bool = False, env_version=None) -> LifetimeResult:
    """Run `agent` in `env` (see the module docstring for the loop and its channels).

    Stops when every environment is done (task horizon, or `days` simulated days into the episode) - unless
    `reset_on_done`, which starts a new episode with `episode_seeds(e)` / derived seeds - after `max_decisions`
    rounds, when every environment reached verified success (`stop_on_success`), or after `stop_after_decisions`
    rounds of this call (an interruption: nothing is flushed or closed). `update_every` (None = the agent's
    cfg.update_every, else 0 = never): learner.step() every k rounds; `learner` None = AgentLearner(agent).
    `on_step(step, transition)` after every round (1-based step), `on_round(RoundInfo)` likewise with the operator-side
    view, `on_reset(episode, batch)` after every env.reset."""
    from .envs.agent_protocol import Transition
    cfg = agent_config(agent)
    mode = reward_mode if reward_mode is not None else (agent_reward_mode(agent) or "observation_only")
    if mode not in REWARD_MODES:
        raise ValueError(f"reward_mode {mode!r}; known: {REWARD_MODES}")
    deliver = mode == "legacy"
    require = dom_decision_path(cfg) if require_cam_eval is None else bool(require_cam_eval)
    auth_source = "config" if require_cam_eval is None else "explicit"
    if think_s is None:
        think_s = float(cfg.think_s) if cfg is not None else 600.0
    if update_every is None:
        update_every = int(getattr(cfg, "update_every", 0) or 0) if cfg is not None else 0
    meta_every = int(getattr(cfg, "meta_every", 0) or 0) if cfg is not None else 0
    if learner is None and hasattr(agent, "update"):
        rm = None
        if not deliver:
            from .selfeval.reward_model import RewardModel
            rm = RewardModel(cfg)
        learner = AgentLearner(agent, reward_model=rm)
    if update_every and learner is None:
        raise ValueError("update_every needs a learner (the agent has no update())")
    if labels is not None and label_store is None:
        raise ValueError("labels need a label_store: verified labels reach the learner only through the LabelStore")
    ck = checkpoint
    operator = ck.operator if ck is not None else None
    if ck is not None and (ck.every_days or ck.resume) and (operator is None or not ck.dir):
        raise ValueError("checkpoints and resume need `operator` and `checkpoint_dir`")
    if ck is not None and ck.every_updates and not ck.dir:
        raise ValueError("update checkpoints need a checkpoint dir")
    held = _held_by(agent, [labels, label_store, evaluator, operator])
    if held:
        raise ValueError(f"the acting agent holds operator / learner-side objects: {held}")
    if callable(reward_source):
        rsrc = reward_source
    elif reward_source == "auto":
        rsrc = ("take_rewards" if hasattr(evaluator, "take_rewards") else
                "score_delta" if hasattr(evaluator, "evaluate_goal") else None) if evaluator is not None else None
    elif reward_source in ("take_rewards", "score_delta", None):
        rsrc = reward_source if evaluator is not None else None
    else:
        raise ValueError(f"reward_source {reward_source!r}")
    dev = getattr(agent, "device", None)
    if not isinstance(dev, torch.device):
        dev = None
    meter = _Meter(dev, gpu_timing)
    observe = _observer(agent)
    acc = ComputeAccount()
    if meter.cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    prof = None
    if meter.cuda and profile_gpu:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.__enter__()
    env_v = env_version if env_version is not None else getattr(env, "version", None)
    eval_v = getattr(labels, "version", None) if labels is not None else getattr(evaluator, "version", None)
    code_v = code_version()
    latest = os.path.join(ck.dir, "LATEST") if ck is not None and ck.dir else None

    ckpts: list = []
    resumed_from = None
    episode_seeds_used: list = []
    rows: list = []
    updates: list = []
    meta_updates: list = []
    n_updates = 0
    labels_taken = labels_appended = 0
    scores = None
    cursors = None
    st = None

    def start_episode(e: int, B_hint: int):
        t = time.perf_counter()
        sd = episode_seeds(e) if episode_seeds is not None else derived_seeds(seeds, e, B_hint)
        episode_seeds_used.append(None if sd is None else list(sd))
        ob = env.reset(sd)
        _begin(agent, list(ob.env_ids), env.goals())
        meter.wall["reset"] += time.perf_counter() - t
        acc.episodes += 1
        sc = [float(evaluator.evaluate_goal(b).score) for b in range(len(ob.env_ids))] \
            if rsrc == "score_delta" else None
        cur = _initial_cursors(env, ob) if collect_events else None
        seen_ = [_max_seq(obs) for obs in ob.observations]
        if on_reset is not None:
            on_reset(e, ob)
        return ob, sc, cur, seen_

    if ck is not None and ck.resume and latest and os.path.exists(latest):
        with open(latest) as f:
            d = os.path.join(ck.dir, f.read().strip())
        operator.restore(os.path.join(d, "operator.pkl"))
        with open(os.path.join(d, "loop.pkl"), "rb") as f:
            st = pickle.load(f)
        if evaluator is not None and os.path.exists(os.path.join(d, "evaluator.pkl")):
            with open(os.path.join(d, "evaluator.pkl"), "rb") as f:
                evaluator.load_state(f.read())
        agent_restored = False
        if hasattr(agent, "load") and os.path.exists(os.path.join(d, "agent")):
            agent.load(os.path.join(d, "agent"))
            agent_restored = True
        elif ck.ops is not None and os.path.exists(os.path.join(d, "agent")):
            ck.ops.load_full_lifetime(os.path.join(d, "agent"))
            agent_restored = True
        elif hasattr(agent, "begin"):
            agent.begin(st["env_ids"], st["goals"])
        lp = os.path.join(d, "learner.pt")
        if learner is not None and os.path.exists(lp) and hasattr(learner, "load_state_dict"):
            learner.load_state_dict(torch.load(lp, map_location="cpu", weights_only=False))
        batch, step, returns = st["batch"], st["step"], st["returns"]
        t_start, episode_id, actor_ids = st["t_start"], st["episode_id"], st["actor_ids"]
        next_ckpt, success_day, updates = st["next_ckpt"], st["success_day"], st["updates"]
        ckpts = list(st.get("ckpts", []))
        episode = int(st.get("episode", 0))
        n_updates = int(st.get("n_updates", sum(bool(u.get("updated")) for u in updates if isinstance(u, dict))))
        meta_updates = list(st.get("meta_updates", []))
        scores, cursors = st.get("scores"), st.get("cursors")
        seen = st.get("seen") or [_max_seq(obs) for obs in batch.observations]
        episode_seeds_used = list(st.get("episode_seeds", []))
        resumed_from = d
        meter.counts["agent_restored"] = int(agent_restored)
    else:
        episode = 0
        batch, scores, cursors, seen = start_episode(0, int(getattr(env, "B", 1) or 1))
        step = 0
        returns = [0.0] * len(batch.env_ids)
        t_start = list(batch.t)
        episode_id = new_id("ep_")
        actor_ids = _self_ids(batch)
        next_ckpt = ck.every_days if ck is not None else None
        success_day = [None] * len(batch.env_ids)
    B = len(batch.env_ids)
    tw = None
    if transcript:
        if resumed_from is not None:
            # the interrupted run may have written steps after its last checkpoint: they are an abandoned branch
            off = st.get("transcript_offset")
            if off is None:
                raise ValueError("the checkpoint has no transcript offset (it was written without a transcript)")
            size = os.path.getsize(transcript) if os.path.exists(transcript) else -1
            if size < off:
                raise ValueError(f"transcript {transcript} is shorter ({size} B) than at the checkpoint ({off} B)")
            with open(transcript, "r+b") as fh:
                fh.truncate(off)
            tw = open(transcript, "a")
        else:
            tw = open(transcript, "w")             # a transcript is one lifetime
            tw.write(json.dumps({"header": {"format": LIFETIME_FORMAT, "env_ids": batch.env_ids, "think_s": think_s,
                                            "episode_id": episode_id, "reward_mode": mode,
                                            "event_schema": EVENT_SCHEMA}}) + "\n")
    calls_this = 0
    n_checked = 0
    interrupted = False
    versions: dict = {}

    def save_checkpoint() -> None:
        t0 = time.perf_counter()
        os.makedirs(ck.dir, exist_ok=True)
        d = os.path.join(ck.dir, f"ckpt_{len(ckpts):04d}")
        os.makedirs(d, exist_ok=True)
        if operator is not None:
            operator.checkpoint(os.path.join(d, "operator.pkl"))
        if evaluator is not None and hasattr(evaluator, "state_dict"):
            _write_atomic(os.path.join(d, "evaluator.pkl"), evaluator.state_dict())
        # the learner's state first: IntegratedLearner.state_dict aborts an open meta window and restores the slow
        # weights, which must happen before the model weights are written (W7 change request)
        ls = learner.state_dict() if learner is not None and hasattr(learner, "state_dict") else None
        if hasattr(agent, "save"):
            agent.save(os.path.join(d, "agent"))
        elif ck.ops is not None:
            ck.ops.save_full_lifetime(os.path.join(d, "agent"))
        if ls is not None:
            tmp = os.path.join(d, "learner.pt.tmp")
            torch.save(ls, tmp)
            os.replace(tmp, os.path.join(d, "learner.pt"))
        ckpts.append(d)
        off = None
        if tw:
            tw.flush()
            os.fsync(tw.fileno())
            off = tw.tell()
        _write_atomic(os.path.join(d, "loop.pkl"), pickle.dumps({
            "transcript_offset": off, "batch": batch, "step": step, "returns": returns, "t_start": t_start,
            "episode_id": episode_id, "actor_ids": actor_ids, "next_ckpt": next_ckpt, "success_day": success_day,
            "updates": updates, "env_ids": list(batch.env_ids), "goals": env.goals(), "ckpts": ckpts,
            "episode": episode, "n_updates": n_updates, "meta_updates": meta_updates, "scores": scores,
            "cursors": cursors, "seen": seen, "episode_seeds": episode_seeds_used, "reward_mode": mode}))
        _write_atomic(latest, os.path.basename(d).encode())
        meter.wall["checkpoint"] += time.perf_counter() - t0

    try:
        while True:
            elapsed = [batch.t[b] - t_start[b] for b in range(B)]
            done = [bool(batch.done[b]) or (days is not None and elapsed[b] >= days) for b in range(B)]
            if all(done):
                if not (reset_on_done and (max_decisions is None or step < max_decisions) and calls_this > 0
                        and (stop_after_decisions is None or calls_this < stop_after_decisions)):
                    break
                episode += 1
                batch, scores, cursors, seen = start_episode(episode, B)
                B = len(batch.env_ids)
                t_start = list(batch.t)
                episode_id = new_id("ep_")
                actor_ids = _self_ids(batch)
                if rows:
                    rows[-1]["reset_delivered"] = [[o.seq for o in obs if o.seq is not None]
                                                   for obs in batch.observations]
                if all(bool(x) for x in batch.done):          # a new episode that is over at reset: nothing to do
                    break
                continue
            if max_decisions is not None and step >= max_decisions:
                break
            if stop_on_success and evaluator is not None and all(s is not None for s in success_day):
                break
            if stop_after_decisions is not None and calls_this >= stop_after_decisions:
                interrupted = True
                break
            # the label episode each row's decision belongs to (taken before execution: a reset during the advance
            # starts the next one); stamped as extra["label_episode"] for label_join (W8 / W7 change request)
            lab_ep = [label_episode_of(labels, batch.env_ids[b]) for b in range(B)] if labels is not None else [None] * B
            tools = meter.timed("tools", env.available_tools)
            actions = list(meter.timed("act", agent.act, batch, tools, gpu=True))
            actions = [None if done[b] else actions[b] for b in range(B)]
            if require:
                n_checked += authorise(actions, step)
            results = meter.timed("execute", env.execute, actions)
            dts = []
            for b in range(B):
                if done[b] or actions[b] is None:
                    dts.append(0.0 if done[b] else think_s / DAY)
                    meter.counts["skipped"] += int(not done[b] and actions[b] is None)
                    continue
                r = results[b]
                dts.append(float(r.duration) + think_s / DAY)
                tool = actions[b].tool
                meter.by_tool[tool] = meter.by_tool.get(tool, 0) + 1
                if r.meta.get("harness") and tool == "wait":
                    meter.counts["waits"] += 1
                else:
                    meter.counts["actions"] += 1
                if not r.ok:
                    meter.counts["rejected"] += 1
                    meter.errors[str(r.error)] = meter.errors.get(str(r.error), 0) + 1
            nxt = meter.timed("advance", env.advance_time, dts)
            events = None
            if collect_events:
                events, cursors = meter.timed("events", env.collect_events, cursors)
            # operator-side verified increments (delivered to the agent only in reward_mode "legacy")
            if rsrc == "take_rewards":
                inc = [float(x) for x in evaluator.take_rewards()]
            elif rsrc == "score_delta":
                new_scores = [float(evaluator.evaluate_goal(b).score) for b in range(B)]
                inc = [ns - s for ns, s in zip(new_scores, scores)]
                scores = new_scores
            elif callable(rsrc):
                inc = [float(x) for x in rsrc(nxt)]
            else:
                inc = [0.0] * B
            nxt_obs = []
            for b in range(B):
                ev_b = events[b] if events is not None else None
                res_obs = drop_delivered(results[b].observations if results[b] is not None else [],
                                         nxt.observations[b], ev_b)
                merged, seen[b] = merge_by_seq(seen[b], res_obs, nxt.observations[b], ev_b)
                nxt_obs.append(merged)
            next_batch = ObservationBatch(list(nxt.env_ids), nxt_obs, list(nxt.t), list(nxt.done))
            mv = policy_version(agent, model_version)
            rm = getattr(learner, "reward_model", None) if learner is not None else None
            versions = event_versions(policy=mv, reward_model=None if deliver else getattr(rm, "version", None),
                                      evaluator=eval_v, env=env_v, code=code_v)
            elapsed_dt = [float(nxt.t[b]) - float(batch.t[b]) for b in range(B)]    # the semi-Markov duration
            records = []
            for b in range(B):
                a, r = actions[b], results[b]
                returns[b] += inc[b]
                records.append(EventRecord(
                    event_id=a.action_id if a is not None else new_id("e_"), env_id=batch.env_ids[b],
                    actor_id=actor_ids[b], episode_id=episode_id, step=step, model_version=mv, t_sim=batch.t[b],
                    observations=batch.observations[b], action=a, result=r,
                    reward=float(inc[b]) if deliver else 0.0,
                    cost=float(r.cost) if r is not None else 0.0, done=bool(nxt.done[b]),
                    behavior_logp=None if a is None else a.logp, provenance=Provenance.REAL,
                    evidence=EvidenceKind.CONTROLLED,
                    extra={"dt_days": elapsed_dt[b], "dt_requested_days": dts[b], "skipped": a is None,
                           "versions": dict(versions), "schema": EVENT_SCHEMA, "reward_mode": mode,
                           **({"label_episode": lab_ep[b]} if lab_ep[b] is not None else {})}))
                if evaluator is not None and success_day[b] is None:
                    stt = getattr(evaluator, "success_t", None)
                    if stt is not None and stt[b] is not None:
                        success_day[b] = float(stt[b]) / DAY - t_start[b]   # days since the episode began
            tr = Transition(records, next_batch, list(inc) if deliver else None, list(nxt.done), elapsed_dt)
            meter.timed("observe", observe, tr, gpu=True)
            if learner is not None and hasattr(learner, "on_step"):
                meter.timed("learner", learner.on_step, tr)
            if labels is not None:
                t0 = time.perf_counter()
                got = take_labels(labels)
                labels_taken += len(got)
                for rec in got:
                    labels_appended += int(bool(label_store.append(rec)))
                meter.wall["labels"] += time.perf_counter() - t0
            if tw:
                for rec in records:
                    tw.write(rec.to_json() + "\n")
            meter.counts["decisions"] += 1
            acc.env_steps += B
            acc.decisions += sum(a is not None for a in actions)
            acc.sim_days += float(sum(dts))
            step += 1
            calls_this += 1
            um = meta = None
            if update_every and step % update_every == 0:
                um = meter.timed("update", learner.step, gpu=True)
                updates.append(um)
                if isinstance(um, dict) and um.get("updated"):
                    n_updates += 1
                    acc.updates += 1
                    if meta_every and n_updates % meta_every == 0 and hasattr(learner, "meta_step"):
                        meta = meter.timed("meta", learner.meta_step, gpu=True)
                        meta_updates.append(meta)
            row = {"step": step - 1, "rewards": list(inc), "ok": [None if r is None else bool(r.ok) for r in results],
                   "tools": [None if a is None else a.tool for a in actions], "t": list(nxt.t),
                   "sources": [None if a is None else a.source.value for a in actions],
                   "delivered": [[o.seq for o in obs if o.seq is not None] for obs in nxt_obs]}
            if um is not None:
                row["update"] = um.get("update_id") if isinstance(um, dict) else None
            if keep_rounds:
                rows.append(row)
            if on_step is not None:
                on_step(step, tr)
            if on_round is not None:
                on_round(RoundInfo(calls_this - 1, step, episode, tr, actions, results, list(inc), next_batch, dts,
                                   um, meta, row))
            batch = next_batch
            due = False
            if ck is not None and ck.every_days and min(batch.t[b] - t_start[b] for b in range(B)) >= next_ckpt:
                next_ckpt += ck.every_days
                due = True
            if ck is not None and ck.every_updates and isinstance(um, dict) and um.get("updated") \
                    and n_updates % int(ck.every_updates) == 0:
                due = True
            if due:
                save_checkpoint()
    finally:
        if tw:
            tw.close()
    if flush_reason and not interrupted and hasattr(agent, "flush"):
        agent.flush(flush_reason)
    final = []
    if final_results:
        for b in range(B):
            final.append(evaluator.evaluate_goal(b).to_dict() if evaluator is not None
                         and hasattr(evaluator, "evaluate_goal") else {})
    if prof is not None:
        torch.cuda.synchronize(dev)
        prof.__exit__(None, None, None)
        kern = [e for e in prof.events() if getattr(e, "device_type", None) is not None
                and str(e.device_type).endswith("CUDA")]
        acc.gpu_busy_s = sum(float(getattr(e, "device_time", 0.0) or 0.0) for e in kern) / 1e6
        acc.gpu_kernels = len(kern)
    w = meter.wall
    acc.wall_s = time.perf_counter() - meter.t0
    acc.cpu_s = time.process_time() - meter.cpu0
    acc.act_s, acc.observe_s, acc.update_s = w["act"], w["observe"], w["update"] + w["meta"]
    acc.env_s = w["tools"] + w["execute"] + w["advance"] + w["events"] + w["reset"]
    acc.gpu_span_s = meter.gpu_span
    acc.max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if meter.cuda:
        acc.peak_gpu_mem_bytes = int(torch.cuda.max_memory_allocated(dev))
    acc.inference_tokens = _tokens_used(agent)
    arch = getattr(agent, "archive", None)
    acc.archive_records = arch.count() if arch is not None and hasattr(arch, "count") else None
    dropped = agent.drop_report() if hasattr(agent, "drop_report") else {}
    return LifetimeResult(steps=step, t_days=list(batch.t), elapsed_days=[batch.t[b] - t_start[b] for b in range(B)],
                          returns=returns, final=final, success_day=success_day, accounting=meter.summary(agent),
                          updates=updates, transcript=transcript, checkpoints=ckpts, resumed_from=resumed_from,
                          reward_mode=mode, episodes=episode + 1, episode_seeds=episode_seeds_used,
                          goal_scores=list(scores) if scores is not None else [], compute=acc.to_dict(), rounds=rows,
                          meta_updates=meta_updates, labels_taken=labels_taken, labels_appended=labels_appended,
                          versions=dict(versions),
                          authorisation={"required": require, "checked": n_checked, "source": auth_source},
                          dropped=dropped, interrupted=interrupted)


# ============================================================================ compatibility wrapper

def run_steps(agent, env: EnvAdapter, n: int, evaluator: Optional[GoalEvaluator] = None,
              update_every: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
              episode_seeds: Optional[Callable[[int], Optional[Sequence[int]]]] = None,
              think_s: Optional[float] = None, reset_on_done: bool = True, profile_gpu: bool = False,
              on_step: Optional[Callable[[int, dict], None]] = None, reward_mode: Optional[str] = None,
              **kw) -> RunReport:
    """`n` decision rounds of `agent` on `env` through run_lifetime: score-delta rewards (evaluate_goal), events
    collected and merged by seq, a new episode with derived seeds whenever every row is done (`reset_on_done`),
    agent.update() every `update_every` rounds, CUDA-event timing; `on_step(i, row)` per round (0-based)."""
    res = run_lifetime(agent, env, evaluator=evaluator, reward_mode=reward_mode or legacy_default_mode(agent, evaluator),
                       max_decisions=int(n), update_every=int(update_every or 0), think_s=think_s, seeds=seeds,
                       episode_seeds=episode_seeds, reset_on_done=reset_on_done, collect_events=True,
                       reward_source="score_delta", keep_rounds=True, final_results=False, gpu_timing=True,
                       profile_gpu=profile_gpu, on_round=None if on_step is None else
                       (lambda info: on_step(info.index, info.row)), **kw)
    return RunReport(steps=res.rounds, updates=res.updates, returns=res.returns, goal_scores=res.goal_scores,
                     episode_seeds=res.episode_seeds, dropped=res.dropped, compute=ComputeAccount(**res.compute))
