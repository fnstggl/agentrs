"""The integrated CAM Unified model on the actor side of the distributed stack (spec 14.1/14.3, manifest A16/F02).

`CamActorPolicy` is the actor's versioned copy of the shared intelligence: the full `CamUnifiedModel` built by
`model.build_model` from the real components of workstreams A-E (Qwen semantic encoder + tokenizer + action generator,
backbone, entities, reasoner, beliefs, Bayes last layer, outcome head, world model, value / option / exploration
heads) wrapped in the integration's `CamUnifiedAgent` (per-environment latent state Z, FastState, entity registries,
outcome trackers, options, budgets). The agent's own `act` / `observe` make every decision; this class only connects it
to the actor process:

* **records.** The agent's EventRecords (decision records and the world-event records of post-action observations)
  and DecisionTraces are captured (instead of an archive / recorder in the actor process) and handed to the actor,
  which stamps routing facts (state version, determinism, incarnation) and writes them to the durable queue.
* **exact learner inputs.** For every decision the inputs of the forward pass (`model.StepInputs`: Z_{t-1}, the
  FastState snapshot, the entity-registry snapshot taken before tokenisation, previous action, goal, budget, option
  decisions, retrieved memory ids) are serialised into the run's content-addressed `ArtifactStore`
  (``<run_dir>/artifacts``) and referenced from the record (``extra["inputs_ref"]``); the decision's tool list is
  stored once per distinct list (``extra["tools_ref"]``). Observations travel in the record itself. With them the
  central learner (`cam_learner.CamLearnerModel`) recomputes the decision's behaviour log-probability exactly under the
  behaviour weights (tested: lag-0 deviation < 1e-3 nats), or under newer weights for V-trace.
* **weights.** `load_weights` swaps the published weights-only state dict in place (aliased parameter names, e.g. the
  backbone shared by the world model and the reasoner, are published once; `unique_state_dict` /
  `load_unique_state_dict`), sets the model version (stamped on every record as ``model_version``) and marks the LoRA
  update for the semantic encoder's cache bookkeeping.
* **latent reconstruction.** `rebuild_state` reconstructs every live environment's Z from the initial state over its
  last ``rebuild_window`` decisions under the *current* weights, from the stored raw inputs (observations, previous
  actions, goals, belief snapshots), so a large representation change never leaves a stale Z in use (spec 14.3).

`build_cam_model` / `cam_config` give the actor and the learner the identical architecture. The default distributed
configuration declares its departures from the full default config (kept in every run's spec): Qwen3-0.6B as the
semantic model (``sem_model``), ``max_arg_tokens`` = 32 (the untrained decoder otherwise emits strings up to 1,536
tokens), imagination off (planner decisions have no tractable behaviour probability, and imagined candidates multiply
decoder calls) and memory retrieval off (no per-actor episodic index is wired into the actor processes yet).
"""
from __future__ import annotations

import collections
import copy
import hashlib
import io
import json
import os
import time
from dataclasses import replace
from typing import Optional

import torch

from ..config import CamConfig, make_config
from ..interfaces import (Action, EntityRegistry, FastState, GoalSpec, ObservationBatch, TokenSet, ToolSchema)

# the departures of the distributed runs from the full default config (spec Part 22: declared, never silent)
DIST_DEFAULTS = {"sem_model": "Qwen/Qwen3-0.6B", "max_arg_tokens": 32, "use_imagination": False, "use_memory": False}
INPUTS_FORMAT = "cam-step-inputs/1"
FAST_TENSORS = ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n", "surprise", "steps")
BUDGET_FIELDS = ("days_left", "funds_left", "decisions_left", "days_total", "funds_total", "decisions_total")


# ============================================================================ model construction (actor = learner)

def cam_config(scale: str = "dev", **over) -> CamConfig:
    """The distributed runs' config: `make_config(scale)` + `DIST_DEFAULTS` + ``over`` (lists become tuples)."""
    kw = dict(DIST_DEFAULTS)
    kw.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in over.items()})
    return make_config(scale, **kw)


def set_deterministic_attention(on: bool = True) -> None:
    """Process-wide: scaled-dot-product attention uses the math kernel only (no flash / memory-efficient / cuDNN
    kernels). Those kernels are chosen per call from the inputs' layout and alignment, which depend on the allocator's
    state, so two processes computing the same attention can take different kernels (different bf16 rounding); the
    math kernel is the same everywhere. Used by actors and the learner so behaviour log-probabilities recompute
    exactly across processes."""
    torch.backends.cuda.enable_flash_sdp(not on)
    torch.backends.cuda.enable_mem_efficient_sdp(not on)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(not on)


def build_cam_model(cfg: CamConfig, device: Optional[str] = None, seed: int = 0, batch_invariant: bool = True,
                    deterministic_attention: bool = True, **model_kwargs):
    """The full CamUnifiedModel from the default component paths (the real classes of workstreams A-E), on
    ``device`` (default cuda when available), in acting (eval) mode.

    batch_invariant: the semantic encoder runs every text chunk in its own forward pass (``max_batch_tokens=1``), so
    a text's hidden states do not depend on which other texts share its batch. In bf16 a padded batch changes the
    states by ~1e-3, which moves a long string action's log-probability by up to ~0.5 nats; with batch-invariant
    encoding the learner, in another process, recomputes the actor's behaviour log-probability to < 1e-3 nats
    (tested). Cost: more, smaller encoder calls (each text is encoded once per process: content-hash cache).
    deterministic_attention: `set_deterministic_attention` (process-wide)."""
    from ..model import build_model
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if deterministic_attention:
        set_deterministic_attention(True)
    torch.manual_seed(int(seed))
    sem_kw = {"device": dev}
    if batch_invariant:
        sem_kw["max_batch_tokens"] = 1
    model = build_model(cfg, component_kwargs={"semantic": sem_kw}, **model_kwargs).to(dev)
    model.eval()
    return model


def alias_map(module: torch.nn.Module) -> dict:
    """{alias name: canonical name} for state-dict entries that share storage with an earlier entry (shared
    modules: the backbone inside the world model / reasoner)."""
    seen, out = {}, {}
    for k, v in module.state_dict(keep_vars=True).items():
        t = v.detach()
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
        if key in seen:
            out[k] = seen[key]
        else:
            seen[key] = k
    return out


def unique_state_dict(module: torch.nn.Module) -> dict:
    """The module's state dict with every shared tensor once (under its first name); tensors only."""
    al = alias_map(module)
    return {k: v for k, v in module.state_dict().items() if k not in al}


def load_unique_state_dict(module: torch.nn.Module, sd: dict, aliases: Optional[dict] = None) -> None:
    """Load a `unique_state_dict` in place. The only keys allowed to be missing are this module's own aliases (they
    are the same tensors as their canonical names, so loading those updates them); anything else is an error."""
    al = alias_map(module) if aliases is None else aliases
    res = module.load_state_dict(sd, strict=False)
    missing = set(res.missing_keys) - set(al)
    if missing or res.unexpected_keys:
        raise RuntimeError(f"weights do not match the model: missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(res.unexpected_keys)[:5]}")


# ============================================================================ step-input serialisation

def _registry_dict(r: EntityRegistry) -> dict:
    st = r.rng.getstate()
    return {"max": int(r.max), "slot": dict(r.slot), "etype": dict(r.etype), "last_seen": dict(r.last_seen),
            "free": list(r.free), "rng": [int(st[0]), [int(x) for x in st[1]], st[2]]}


def _registry_from(d: dict) -> EntityRegistry:
    r = EntityRegistry(int(d["max"]), seed=0)
    r.slot = {str(k): int(v) for k, v in d["slot"].items()}
    r.etype = {str(k): str(v) for k, v in d["etype"].items()}
    r.last_seen = {str(k): float(v) for k, v in d["last_seen"].items()}
    r.free = [int(x) for x in d["free"]]
    st = d["rng"]
    r.rng.setstate((int(st[0]), tuple(int(x) for x in st[1]), st[2]))
    return r


def _goal_dict(g) -> Optional[dict]:
    if g is None:
        return None
    return {"goal_id": g.goal_id, "text": g.text, "metrics": dict(g.metrics or {}), "deadline_days": g.deadline_days}


def _goal_from(d: Optional[dict]):
    if d is None:
        return None
    return GoalSpec(d["goal_id"], d["text"], dict(d.get("metrics") or {}), d.get("deadline_days"))


def _plain(x):
    """JSON-normalised copy (tuples -> lists, enums -> values) so torch.load(weights_only=True) can read it."""
    return json.loads(json.dumps(x, default=str))


def pack_step_inputs(inp, j: int, tools_ref: str, last_tool: Optional[dict]) -> dict:
    """Row j of a `model.StepInputs` -> a weights_only-loadable dict (observations and tools are stored elsewhere:
    the record and the tool-list artifact)."""
    sub = inp.select([j])
    fast = None
    if sub.fast is not None:
        fast = {n: getattr(sub.fast, n).detach().cpu().clone() for n in FAST_TENSORS}
        fast["channels"] = _plain(sub.fast.channels)
    mem = None
    if sub.memory is not None:
        m = sub.memory
        mem = {"x": m.x.detach().float().cpu(), "mask": m.mask.cpu(), "kind": m.kind.cpu(), "time": m.time.cpu(),
               "entity": m.entity.cpu(), "refs": _plain(m.refs), "entity2": None if m.entity2 is None
               else m.entity2.cpu()}
    bud = None
    if sub.budget is not None:
        bud = {f: getattr(sub.budget, f).detach().cpu() for f in BUDGET_FIELDS}
    la = sub.last_actions[0]
    return {"format": INPUTS_FORMAT, "Z_prev": sub.Z_prev.detach().float().cpu().clone(), "fast": fast,
            "registries": [_registry_dict(r) for r in sub.registries], "now": [float(x) for x in sub.now],
            "last_action": None if la is None else _plain(la.to_dict()), "last_tool": last_tool,
            "goal": _goal_dict(sub.goals[0]), "tools_ref": tools_ref, "memory": mem,
            "memory_ids": _plain(sub.memory_ids), "dt_prev": sub.dt_prev.detach().float().cpu(), "budget": bud,
            "option_prev": sub.option_prev.cpu(), "option_steps_prev": sub.option_steps_prev.cpu(),
            "option": sub.option.cpu(), "option_new": sub.option_new.cpu(), "option_steps": sub.option_steps.cpu()}


def inputs_bytes(d: dict) -> bytes:
    bio = io.BytesIO()
    torch.save(d, bio)
    return bio.getvalue()


def inputs_from_bytes(b: bytes) -> dict:
    d = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
    if d.get("format") != INPUTS_FORMAT:
        raise ValueError(f"not a {INPUTS_FORMAT} payload")
    return d


def unpack_step_inputs(d: dict, observations: list, tools: list, device, Z_prev: Optional[torch.Tensor] = None):
    """A packed row + the record's observations + the decision's tool list -> one-row `model.StepInputs` on
    ``device`` (fresh registry objects: callers may bind into them)."""
    from ..model import BudgetState, StepInputs
    dev = torch.device(device)
    fast = None
    if d["fast"] is not None:
        f = d["fast"]
        fast = FastState(*(f[n].to(dev) for n in FAST_TENSORS), [dict(c) for c in f["channels"]])
    mem = None
    if d["memory"] is not None:
        m = d["memory"]
        mem = TokenSet(m["x"].to(dev), m["mask"].to(dev), m["kind"].to(dev), m["time"].to(dev), m["entity"].to(dev),
                       m["refs"], None if m["entity2"] is None else m["entity2"].to(dev))
    bud = None
    if d["budget"] is not None:
        bud = BudgetState(*(d["budget"][f].to(dev) for f in BUDGET_FIELDS))
    la = Action.from_dict(d["last_action"]) if d["last_action"] is not None else None
    Z = d["Z_prev"] if Z_prev is None else Z_prev
    return StepInputs(Z.to(dev), fast, [_registry_from(r) for r in d["registries"]], [list(observations)],
                      list(d["now"]), [la], [_goal_from(d["goal"])], [list(tools)], mem, [list(x) for x in d["memory_ids"]],
                      d["dt_prev"].to(dev), bud, d["option_prev"].to(dev), d["option_steps_prev"].to(dev),
                      d["option"].to(dev), d["option_new"].to(dev), d["option_steps"].to(dev))


def tools_payload(tools: list) -> bytes:
    """Key order is preserved (never sorted): the decoder prompt renders a tool's JSON schema in its own key order,
    so a reordered schema is a different prompt (the action generator then refuses the replay)."""
    return json.dumps([t.to_dict() for t in tools], separators=(",", ":"), ensure_ascii=False).encode()


def tools_from_payload(b: bytes) -> list:
    return [ToolSchema.from_dict(x) for x in json.loads(b.decode())]


def artifacts_dir(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), "artifacts")


class InputsStore:
    """The run's content-addressed store of decision inputs and tool lists (workstream E's `ArtifactStore`:
    atomic writes, digest-verified reads), with a bounded in-process cache of decoded tool lists."""

    def __init__(self, root: str, fsync: bool = True, codec: str = "gzip", level: int = 1):
        from ..memory.artifacts import ArtifactStore
        self.store = ArtifactStore(root, codec=codec, level=level, fsync=fsync)
        self._tools: collections.OrderedDict = collections.OrderedDict()
        self._tools_digest: dict = {}
        self.stats = collections.Counter()

    def put_tools(self, tools: list) -> str:
        raw = tools_payload(tools)
        h = hashlib.sha256(raw).hexdigest()
        ref = self._tools_digest.get(h)
        if ref is None:
            ref = self.store.put(raw)
            self._tools_digest[h] = ref
            self.stats["tool_lists"] += 1
        return ref

    def put_inputs(self, d: dict) -> str:
        b = inputs_bytes(d)
        self.stats["inputs"] += 1
        self.stats["inputs_bytes"] += len(b)
        return self.store.put(b)

    def tools(self, ref: str) -> list:
        t = self._tools.get(ref)
        if t is None:
            t = tools_from_payload(self.store.get(ref))
            self._tools[ref] = t
            while len(self._tools) > 64:
                self._tools.popitem(last=False)
        else:
            self._tools.move_to_end(ref)
        return t

    def inputs(self, ref: str) -> dict:
        return inputs_from_bytes(self.store.get(ref))


# ============================================================================ the actor's policy

class _Capture:
    """Stands in for the archive / recorder of the actor-side agent: collects what the agent archives / records so
    the actor process sends it through the durable queue (the archive lives with the learner)."""

    def __init__(self):
        self.records: list = []
        self.traces: list = []

    def append(self, records: list) -> None:          # ArchiveP.append
        self.records.extend(records)

    def count(self) -> int:
        return 0

    def record(self, traces: list) -> None:           # TraceRecorderP.record
        self.traces.extend(traces)

    def drain(self) -> tuple:
        r, t = self.records, self.traces
        self.records, self.traces = [], []
        return r, t


class CamActorPolicy:
    """Implements the actor's record-producing policy protocol (`actor.RecordPolicyP`) over the integrated agent.

    Args: seed (sampling and registry seeds), actor_id, run_dir (the shared run directory: artifacts), cfg (a dict of
    `cam_config` overrides, incl. ``scale``), device, greedy (deterministic behaviour: logp 0), rebuild_window
    (decisions replayed per environment by `rebuild_state`), fsync (artifact durability)."""

    produces_records = True

    def __init__(self, seed: int = 0, actor_id: str = "actor0", run_dir: Optional[str] = None,
                 cfg: Optional[dict] = None, device: Optional[str] = None, greedy: bool = False,
                 rebuild_window: int = 64, fsync: bool = True, model_kwargs: Optional[dict] = None):
        from ..agent import CamUnifiedAgent
        if run_dir is None:
            raise ValueError("CamActorPolicy needs the run directory (decision inputs are stored there)")
        over = dict(cfg or {})
        scale = over.pop("scale", "dev")
        self.cfg = cam_config(scale, **over)
        self.cfg_overrides = {"scale": scale, **over}
        t0 = time.time()
        self.model = build_cam_model(self.cfg, device=device, seed=seed, **(model_kwargs or {}))
        self.build_s = time.time() - t0
        self.aliases = alias_map(self.model)
        self.capture = _Capture()
        self.agent = CamUnifiedAgent(self.model, archive=self.capture, recorder=self.capture, actor_id=actor_id,
                                     seed=int(seed), greedy=bool(greedy))
        self.actor_id = actor_id
        self.greedy = bool(greedy)
        self.store = InputsStore(artifacts_dir(run_dir), fsync=fsync)
        self.think_days = self.cfg.think_s / 86400.0
        self.rebuild_window = int(rebuild_window)
        self._windows: list = []              # per env row: deque of (packed inputs, observations, tools ref)
        self._tools_ref: list = []
        self.counters = collections.Counter()
        self.timing = collections.Counter()

    # ---------------------------------------------------------------- weights
    def load_weights(self, state_dict: dict, version: int) -> None:
        with torch.no_grad():
            load_unique_state_dict(self.model, state_dict, self.aliases)
        self.model.version = int(version)               # bumps the weights epoch (query-embedding caches)
        enc = self.model.semantic
        if enc is not None and hasattr(enc, "mark_updated"):
            enc.mark_updated()
        self.model.eval()

    @property
    def version(self) -> int:
        return int(self.model.version)

    # ---------------------------------------------------------------- episodes
    def reset(self, obs: ObservationBatch, goals: list, tools: list) -> None:
        self.agent.flush("reset")                       # (end_episode already did; leftovers go out with the next drain)
        self.agent.state = None                         # every episode is a new batch of worlds (new env ids)
        self.agent.reset(list(obs.env_ids))
        self.agent.set_goals(list(goals))
        B = len(obs.env_ids)
        self._windows = [collections.deque(maxlen=max(1, self.rebuild_window)) for _ in range(B)]
        self._tools_ref = [None] * B

    @property
    def episode_ids(self) -> list:
        return list(self.agent.episode_ids)

    def act_batch(self, obs: ObservationBatch, tools: list, active: list) -> list:
        """One decision per active environment (inactive rows: done, are skipped)."""
        t0 = time.perf_counter()
        done = [bool(d) or not a for d, a in zip(obs.done, active)]
        ob = ObservationBatch(list(obs.env_ids), [list(o) for o in obs.observations], list(obs.t), done)
        self._tools_ref = [self.store.put_tools(list(tl)) for tl in tools]
        acts = self.agent.act(ob, [list(t) for t in tools])
        self.timing["act_s"] += time.perf_counter() - t0
        return acts

    def observe_batch(self, results: list, rewards: list, next_obs: ObservationBatch) -> list:
        """Consequences of the last `act_batch` -> [(EventRecord, DecisionTrace | None)] in archive order; every
        decision record carries ``extra["inputs_ref"]`` / ``extra["tools_ref"]`` (the exact learner inputs)."""
        t0 = time.perf_counter()
        pend = self.agent._pending
        self.agent.observe(results, rewards, next_obs)
        t1 = time.perf_counter()
        self.timing["observe_s"] += t1 - t0
        records, traces = self.capture.drain()
        fresh = self.agent.fresh[-1] if (pend is not None and self.agent.fresh) else None
        by_decision = {}
        if fresh is not None:
            for j, did in enumerate(fresh.decision_ids):
                by_decision[did] = j
        tr_by_event = {t.event_id: t for t in traces}
        out = []
        for rec in records:
            tr = None
            if rec.action is not None:
                did = rec.extra.get("decision_id")
                j = by_decision.get(did)
                if j is None:
                    raise RuntimeError(f"decision {did} has no step inputs")
                row = fresh.rows[j]
                inp = fresh.inputs
                la = inp.last_actions[j]
                last_tool = None
                if la is not None:
                    try:
                        last_tool = self.model.tokenizer.resolve_tool(la, None).to_dict()
                    except ValueError:
                        last_tool = None
                packed = pack_step_inputs(inp, j, self._tools_ref[row], last_tool)
                rec.extra["inputs_ref"] = self.store.put_inputs(packed)
                rec.extra["tools_ref"] = self._tools_ref[row]
                self._windows[row].append((packed, list(rec.observations), self._tools_ref[row]))
                tr = tr_by_event.get(rec.event_id)
                if tr is not None:
                    tr.actor_id = self.actor_id
                    tr.episode_id = rec.episode_id
                    vals = (tr.outcome or {}).get("values") or {}
                    for k, v in vals.items():          # realised outcome values next to the predictions (dashboard)
                        tr.outcome.setdefault(k, v)
                self.counters["decisions"] += 1
            out.append((rec, tr))
        # the actor never trains: drop the agent's on-policy buffer (the learner gets the data through the queue)
        self.agent.fresh = []
        self.agent._last_fresh = {}
        self.timing["pack_s"] += time.perf_counter() - t1
        return out

    def end_episode(self, reason: str = "end_of_run") -> list:
        """World-event records of observations delivered after the last decision of still-open environments."""
        self.agent.flush(reason)
        recs, _ = self.capture.drain()
        return recs

    # ---------------------------------------------------------------- latent reconstruction (spec 14.3)
    @torch.no_grad()
    def rebuild_state(self, history: list) -> None:
        """Rebuild Z of every environment from the initial state over its last ``rebuild_window`` decisions with the
        current weights (raw stored inputs only; the stale Z is never used)."""
        from ..model import StepInputs, acting_mode
        st = self.agent.state
        if st is None:
            return
        t0 = time.perf_counter()
        m = self.model
        dev = m.device
        rows = [b for b in range(len(self._windows)) if self._windows[b]]
        Z = {b: m.core.init_state(1, device=dev).float() for b in rows}
        L = max((len(self._windows[b]) for b in rows), default=0)
        with acting_mode(m), m._autocast():
            for i in range(L):
                live = [b for b in rows if len(self._windows[b]) > i]
                inps = []
                for b in live:
                    packed, obs, tref = self._windows[b][i]
                    inps.append(unpack_step_inputs(packed, obs, self.store.tools(tref), dev, Z_prev=Z[b]))
                    self.model.tokenizer.register_tools(self.store.tools(tref))
                enc = m.encode_step(StepInputs.cat(inps))
                for k, b in enumerate(live):
                    Z[b] = enc.Z[k:k + 1].detach().float()
        for b in rows:
            st.Z[b] = Z[b][0].to(st.Z.dtype)
        self.counters["rebuilds"] += 1
        self.counters["rebuild_decisions"] += sum(len(self._windows[b]) for b in rows)
        self.timing["rebuild_s"] += time.perf_counter() - t0

    # ---------------------------------------------------------------- accounting
    def compute_counters(self) -> dict:
        """Cumulative semantic-model token counts of this process (encoder prefill, decoder sampled tokens)."""
        enc = self.model.semantic
        ag = self.model.action_generator
        es = getattr(enc, "stats", {}) or {}
        gs = getattr(ag, "stats", {}) or {}
        return {"tokens_prompt": int(es.get("encoded_tokens", 0)), "tokens_generated": int(gs.get("sampled_tokens", 0)),
                "qwen_decoder_steps": int(gs.get("qwen_steps", 0)), "encoder_cache_hits": int(es.get("hits", 0)),
                "encoder_cache_misses": int(es.get("misses", 0))}

    def describe(self) -> dict:
        return {"cfg": self.cfg_overrides, "params": self.model.count_parameters(), "build_s": self.build_s,
                "greedy": self.greedy, "rebuild_window": self.rebuild_window,
                "build_report": dict(getattr(self.model, "build_report", {}))}


def make_cam_policy(seed: int = 0, actor_id: str = "actor0", run_dir: Optional[str] = None, **kw) -> CamActorPolicy:
    """Actor policy factory (`ActorSpec.policy_factory`); the actor passes its id and run directory."""
    return CamActorPolicy(seed=seed, actor_id=actor_id, run_dir=run_dir, **kw)
