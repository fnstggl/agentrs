"""`HarnessAgent`: the integrated CamUnifiedAgent as workstream G's lifetime-harness `AgentP`, with durable save/load.

    act(obs, tools)      -> CamUnifiedAgent.act: one CAM decision per live environment (decoder or planner source)
    observe(Transition)  -> CamUnifiedAgent.observe: outcomes -> beliefs / Kalman layer, EventRecords -> archive,
                            DecisionTraces -> F's TraceRecorder, fresh on-policy buffer
    update()             -> CamUnifiedAgent.update: one learner step (D's CombinedTrainer) from archived real events
                            (E's replay) + the fresh buffer; per-component parameter deltas, the core's parameter hash
                            before / after, per-objective backbone gradient norms; the fixed policy probes
    save(path) / load(path)  full-lifetime checkpoint (E's CheckpointOps: weights, optimiser, fast / latent state,
                            entity registries, RNG, version, counters, archive pointer, replay sampler state,
                            consolidator state, pinned memory versions) + a snapshot of the memory index + the
                            agent's remaining working state (outcome trackers, options, last actions, the previous
                            belief, the fresh buffer, budgets, the tokenizer's tool registry, trainer counters, the
                            global torch / CUDA / numpy / python RNG states), so a NEW process continues the lifetime
                            where the checkpoint left it. Events archived after the checkpoint by the interrupted
                            process are identified (`branch.abandoned_after`), excluded from replay and memory
                            consolidation (`BranchView`) and their traces are marked ``abandoned_branch``.

Run-directory files written for the evidence summaries and F's dashboard (`observe.dashboard.build_dashboard`):
``decisions.jsonl`` (one row per decision: tool, source, ok / error code, reward, tokens, wall times, model version),
``learner/updates.jsonl`` (one row per update), ``probes/probes.jsonl`` (policy at fixed probe contexts per version),
``ckpt/v<version>/meta.json`` + ``ckpt/LATEST``, ``status/<actor>.json``, ``series.jsonl``, ``compute/*.json``,
``resume.jsonl``. Log files are truncated to their checkpoint offsets on load (like the harness transcript), so they
hold one lifetime with every step exactly once.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch

from ..interfaces import GoalSpec
from ..model import StepInputs
from .branch import abandoned_after
from .system import System, make_index

LOGS = ("decisions.jsonl", "learner/updates.jsonl", "probes/probes.jsonl", "learner/trainer.jsonl", "series.jsonl")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if torch.is_tensor(x):
        return _jsonable(x.detach().cpu().tolist())
    if isinstance(x, (np.floating, np.integer)):
        return _jsonable(x.item())
    return str(x)


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(_jsonable(row), separators=(",", ":")) + "\n")


def _append_rows(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(_jsonable(r), separators=(",", ":")) + "\n")


def _atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def component_of(name: str) -> str:
    return name.split(".", 1)[0]


def param_snapshot(model) -> dict:
    """{name: detached clone} of every unique parameter (shared modules once)."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def tensor_hash(tensors: list) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def core_hash(model) -> str:
    """sha256 of the shared UnifiedCore's parameters (float32, name order)."""
    return tensor_hash([p for _, p in sorted(model.core.named_parameters(), key=lambda kv: kv[0])])


def param_deltas(model, before: dict) -> dict:
    """Per component: L2 norm of the parameter change, relative change, number of tensors changed / total."""
    acc: dict = {}
    for n, p in model.named_parameters():
        b = before.get(n)
        if b is None:
            continue
        c = component_of(n)
        a = acc.setdefault(c, {"d2": 0.0, "n2": 0.0, "changed": 0, "tensors": 0, "numel": 0})
        d = (p.detach().float() - b.float())
        a["d2"] += float((d * d).sum())
        a["n2"] += float((b.float() * b.float()).sum())
        a["changed"] += int(not torch.equal(p.detach(), b))
        a["tensors"] += 1
        a["numel"] += p.numel()
    return {c: {"l2_delta": math.sqrt(v["d2"]), "rel_delta": math.sqrt(v["d2"]) / max(math.sqrt(v["n2"]), 1e-12),
                "tensors_changed": v["changed"], "tensors": v["tensors"], "numel": v["numel"]}
            for c, v in sorted(acc.items())}


def snapshot_index(src_root: str, dst: str) -> dict:
    """Consistent copy of a MemoryIndex directory: shard files copied (bytes beyond the committed row counts are
    truncated by MemoryIndex on open), the SQLite database through the online backup API (includes its WAL)."""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    n = 0
    for dirpath, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.startswith("index.sqlite"):
                continue
            shutil.copy2(os.path.join(dirpath, f), os.path.join(out, f))
            n += 1
    src_db = os.path.join(src_root, "index.sqlite")
    if os.path.exists(src_db):
        s = sqlite3.connect(src_db)
        d = sqlite3.connect(os.path.join(dst, "index.sqlite"))
        with d:
            s.backup(d)
        d.close()
        s.close()
    return {"files": n}


def rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"].cpu())
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


class HarnessAgent:
    """See the module docstring."""

    name = "cam_unified"

    def __init__(self, system: System, *, probe_rows: int = 4, track_params: bool = True):
        self.sys = system
        self.agent = system.agent
        self.run_dir = system.run_dir
        self.probe_rows = int(probe_rows)
        self.track_params = track_params
        self.t_start = time.time()
        self._act: Optional[dict] = None
        self._before: Optional[dict] = None
        self._probe: Optional[dict] = None
        self.n_updates = 0
        self.peak_gpu_mb = 0.0
        self.resume_info: Optional[dict] = None
        from ..observe.accounting import ComputeMeter
        self.meter = ComputeMeter(f"{system.sc.actor_id}", role="actor+learner")
        self._load_probe_set()

    # ------------------------------------------------------------------ AgentP
    @property
    def model_version(self) -> int:
        return int(self.agent.version)

    def begin(self, env_ids: list, goals: list) -> None:
        self.agent.reset(list(env_ids))
        self.agent.set_goals(list(goals))
        if self.track_params and self._before is None:
            self._before = param_snapshot(self.sys.model)
        self._status("running")

    def _counters(self) -> dict:
        ag = self.sys.model.action_generator
        st = getattr(ag, "stats", {}) or {}
        es = getattr(self.sys.semantic, "stats", {}) or {}
        return {"sampled_tokens": int(st.get("sampled_tokens", 0)), "qwen_steps": int(st.get("qwen_steps", 0)),
                "actions_generated": int(st.get("actions", 0)), "encoded_tokens": int(es.get("encoded_tokens", 0)),
                "decisions_heads": int(st.get("decisions", 0)), "schema_violations": int(st.get("schema_violations", 0))}

    def _peak_reset(self) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()

    def _peak(self) -> Optional[float]:
        """Peak GPU memory (MB) since the last reset; also folded into the lifetime peak `peak_gpu_mb`."""
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return None
        v = torch.cuda.max_memory_allocated() / 2 ** 20
        self.peak_gpu_mb = max(self.peak_gpu_mb, v)
        return v

    def act(self, obs, tools) -> list:
        c0 = self._counters()
        self._peak_reset()
        t0 = time.perf_counter()
        with self.meter.gpu():
            actions = self.agent.act(obs, tools)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        c1 = self._counters()
        self._act = {"act_s": dt, **{f"d_{k}": c1[k] - c0[k] for k in c0},
                     "n_tools": [len(t) for t in tools], "peak_gpu_mb": self._peak(),
                     "valid": [self._schema_check(a, t) for a, t in zip(actions, tools)]}
        return actions

    def _schema_check(self, action, tools) -> Optional[dict]:
        """Validate the executed arguments against the schema the action was generated for (the environment's tool
        list at decision time, or the generator's built-ins) with Crucible's own JSON-schema validator."""
        if action is None:
            return None
        from crucible.core import schema as S
        by = {t.name: t for t in tools}
        for t in getattr(self.sys.model.action_generator, "builtins", []) or []:
            by.setdefault(t.name, t)
        t = by.get(action.tool)
        if t is None:
            return {"valid": False, "errors": [f"unknown tool {action.tool}"]}
        errs = S.validate(t.parameters or {"type": "object"}, dict(action.args or {}))
        return {"valid": not errs, "errors": list(errs)[:4]}

    def observe(self, transition) -> None:
        t0 = time.perf_counter()
        with self.meter.gpu():
            records = self.agent.observe(transition)
        dt = time.perf_counter() - t0
        a = self._act or {}
        n_dec = 0
        valid = [v for v in (a.get("valid") or []) if v is not None]
        for r in records:
            if r.action is None:
                continue
            v = valid[n_dec] if n_dec < len(valid) else None
            n_dec += 1
            act = r.action
            row = {"step": r.step, "event_id": r.event_id, "env_id": r.env_id, "episode_id": r.episode_id,
                   "t_sim": r.t_sim, "dt_days": r.extra.get("dt"), "model_version": r.model_version,
                   "tool": act.tool, "source": act.source.value, "ok": bool(r.result.ok) if r.result else None,
                   "error": None if r.result is None else r.result.error, "reward": r.reward, "cost": r.cost,
                   "behavior_logp": r.behavior_logp, "behavior": r.extra.get("behavior", {}).get("policy"),
                   "n_tokens_chosen": (act.logp_parts or {}).get("n_tokens"),
                   "n_decisions_chosen": (act.logp_parts or {}).get("n_decisions"),
                   "candidates": (act.proposal or {}).get("n"),
                   "decoded_tokens_total": a.get("d_sampled_tokens"), "qwen_steps": a.get("d_qwen_steps"),
                   "actions_generated": a.get("d_actions_generated"), "encoded_tokens": a.get("d_encoded_tokens"),
                   "act_s": a.get("act_s"), "observe_s": dt, "n_tools": (a.get("n_tools") or [None])[0],
                   "peak_gpu_mb_act": a.get("peak_gpu_mb"),
                   "schema_valid": None if v is None else v["valid"],
                   "schema_errors": None if v is None else v["errors"],
                   "reasoning_iters": r.extra.get("reasoning_iters"), "retrieved": len(r.extra.get("retrieved") or []),
                   "args_chars": len(json.dumps(act.args, default=str)),
                   "args": json.dumps(act.args, sort_keys=True, default=str)[:400], "t_wall": time.time()}
            _append(os.path.join(self.run_dir, "decisions.jsonl"), row)
        self.meter.add(decisions=n_dec, env_steps=1, records=len(records),
                       sim_days=float(sum(transition.dt)) if getattr(transition, "dt", None) else 0.0,
                       tokens_generated=int(a.get("d_sampled_tokens") or 0),
                       tokens_prompt=int(a.get("d_encoded_tokens") or 0))
        _append(os.path.join(self.run_dir, "series.jsonl"),
                {"t_rel": time.time() - self.t_start, "decisions": self.agent.n_decisions,
                 "archive": self.agent.archive.count() if self.agent.archive is not None else None,
                 "version": self.agent.version})
        self._act = None

    def update(self) -> dict:
        m = self.sys.model
        if self._probe is None:
            self._make_probe_set()
        if self._probe is not None and not self._probe.get("evaluated_at_v0") and self.agent.version == 0:
            self._log_probe("before_first_update")
            self._probe["evaluated_at_v0"] = True
        before = self._before if self.track_params else None
        h0 = core_hash(m) if self.track_params else None
        self._peak_reset()
        t0 = time.perf_counter()
        with self.meter.gpu():
            out = self.agent.update()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak = self._peak()
        if not out.get("updated"):
            return {"updated": False, "reason": out.get("reason"), "version": self.agent.version}
        self.n_updates += 1
        deltas = param_deltas(m, before) if before is not None else None
        h1 = core_hash(m) if self.track_params else None
        if self.track_params:
            self._before = param_snapshot(m)
        tr = {k[len("trainer/"):]: v for k, v in out.items() if k.startswith("trainer/")}
        bb = tr.get("backbone") or {}
        row = {"update_id": out["update_id"], "version": out["version"], "trained": True, "t": time.time(),
               "t_rel": time.time() - self.t_start, "update_s": wall, "peak_gpu_mb_update": peak,
               "n_decisions": len(out.get("consumed_decisions") or []),
               "by_actor": {self.agent.actor_id: len(out.get("consumed_decisions") or [])},
               "metrics": {"loss": tr.get("total"), **{k: v for k, v in (out.get("losses") or {}).items()}},
               "losses": out.get("losses"), "scaled_losses": out.get("scaled_losses"),
               "balance": {k.split("/", 1)[1]: v for k, v in out.items() if k.startswith("balance/")},
               "weighted": tr.get("weighted"), "total": tr.get("total"),
               "grad_norm": tr.get("grad_norm"), "group_grad_norm": tr.get("group_grad_norm"),
               "clipped": tr.get("clipped"), "backbone_grad_norm": bb.get("norm"), "backbone_share": bb.get("share"),
               "backbone_cosine": bb.get("cosine"), "dominant": bb.get("dominant"), "stage": tr.get("stage"),
               "param_delta": deltas, "core_hash_before": h0, "core_hash_after": h1,
               "consumed_events": len(out.get("consumed_events") or []),
               "replay_events": len(out.get("replay_events") or []),
               "replay_event_ids": list(out.get("replay_events") or [])[:64],
               "consumed_event_ids": list(out.get("consumed_events") or [])[:64],
               "marked_events_total": self.agent.provenance_stats["marked_events"],
               "consolidated": out.get("consolidated"), "memory_linked": len(out.get("memory_linked") or []),
               "adapt_T": out.get("adapt_seq_T"), "adapt_k": out.get("adapt/adapt_k"),
               "policy": {k.split("/", 1)[1]: v for k, v in out.items() if k.startswith("policy/")},
               "world": {k.split("/", 1)[1]: v for k, v in out.items() if k.startswith("world/")},
               "memory": {k.split("/", 1)[1]: v for k, v in out.items() if k.startswith("memory/")},
               "skipped": {k: v for k, v in out.items() if k.startswith("skipped/")}}
        _append(os.path.join(self.run_dir, "learner", "updates.jsonl"), row)
        self.meter.add(updates=1)
        if self._probe is not None:
            self._log_probe("after_update")
        self._status("running")
        return {"updated": True, "update_id": out["update_id"], "version": out["version"],
                "losses": {k: float(v) for k, v in (out.get("losses") or {}).items()}}

    def usage(self) -> dict:
        c = self._counters()
        snap = self.meter.snapshot()
        return {**c, "decisions": self.agent.n_decisions, "updates": self.n_updates, "gpu_s": snap.get("gpu_s"),
                "peak_gpu_mb": self.peak_gpu_mb,
                "timing": dict(self.agent.timing), "version": self.agent.version}

    # ------------------------------------------------------------------ probes (fixed contexts)
    def _probe_file(self) -> str:
        return os.path.join(self.run_dir, "probes", "probe_set.pt")

    def _load_probe_set(self) -> None:
        p = self._probe_file()
        if os.path.exists(p):
            self._probe = torch.load(p, map_location=self.sys.model.device, weights_only=False)

    def _make_probe_set(self) -> None:
        """The first `probe_rows` fresh decisions become the fixed probe contexts (inputs + executed actions)."""
        rows = []
        for st in self.agent.fresh:
            for j in range(len(st.rows)):
                if st.inputs is not None and len(rows) < self.probe_rows:
                    rows.append((st, j))
        if not rows:
            return
        inp = StepInputs.cat([st.inputs.select([j]) for st, j in rows])
        acts = [copy.deepcopy(st.actions[j]) for st, j in rows]
        self._probe = {"inputs": inp, "actions": acts, "event_ids": [st.event_ids[j] if st.event_ids else None
                                                                     for st, j in rows],
                       "created_at_version": self.agent.version}
        os.makedirs(os.path.dirname(self._probe_file()), exist_ok=True)
        torch.save(self._probe, self._probe_file())

    def probe(self) -> dict:
        p = self._probe
        out = self.sys.model.policy_probe(p["inputs"], p["actions"])
        out["event_ids"] = p["event_ids"]
        out["tools"] = [a.tool for a in p["actions"]]
        return out

    def _log_probe(self, when: str) -> None:
        r = self.probe()
        _append(os.path.join(self.run_dir, "probes", "probes.jsonl"),
                {"when": when, "version": r["version"], "rows": r["rows"], "tools": r["tools"],
                 "event_ids": r["event_ids"], "t": time.time()})

    # ------------------------------------------------------------------ dashboard status
    def _status(self, state: str) -> None:
        a = self.agent
        _atomic_json(os.path.join(self.run_dir, "status", f"{a.actor_id}.json"),
                     {"actor_id": a.actor_id, "state": state, "version": a.version, "env_ids": list(a.env_ids),
                      "goals": [g.text for g in a.goals], "episodes": 1, "decisions": a.n_decisions,
                      "records": a.archive.count() if a.archive is not None else None, "t": time.time(),
                      "t_start": self.t_start})
        _atomic_json(os.path.join(self.run_dir, "status", "learner.json"),
                     {"counts": {"updates": a.version}, "archive_count": a.archive.count() if a.archive else None,
                      "t": time.time()})
        os.makedirs(os.path.join(self.run_dir, "compute"), exist_ok=True)
        self.meter.write(os.path.join(self.run_dir, "compute", f"{self.meter.name}.{os.getpid()}.json"))

    # ------------------------------------------------------------------ checkpoints
    def _extra_state(self) -> dict:
        a = self.agent
        st = a.state
        tok = self.sys.model.tokenizer
        tr = self.sys.trainer
        offsets = {}
        for f in LOGS:
            p = os.path.join(self.run_dir, f)
            offsets[f] = os.path.getsize(p) if os.path.exists(p) else 0
        if a._pending is not None:
            raise RuntimeError("save between act() and observe(): the decision's consequences are not recorded yet")
        return {
            "format": "cam_unified.stage1.agent_state.v1",
            "env_ids": list(a.env_ids), "goals": [g.to_dict() for g in a.goals], "episode_ids": list(a.episode_ids),
            "steps": list(a.steps), "n_decisions": a.n_decisions,
            "cam_state": {"trackers": st.trackers, "option": st.option.cpu(), "option_steps": st.option_steps.cpu(),
                          "last_actions": st.last_actions, "last_now": st.last_now.cpu(),
                          "fast_prev": None if st.fast_prev is None else st.fast_prev},
            "fresh": a.fresh, "last_fresh": a._last_fresh, "post": a._post, "last_eid": a._last_eid,
            "loss_mag": dict(a.loss_mag), "memory_sync_log": list(a.memory_sync_log), "lp_valid": list(a._lp_valid), "t0": list(a._t0), "funds": a._funds, "funds0": list(a._funds0),
            "ep_decisions": list(a._ep_decisions), "adapt_k_hist": dict(a.adapt_k_hist), "dropped": dict(a.dropped),
            "update_log": a.update_log, "timing": dict(a.timing), "provenance_stats": dict(a.provenance_stats),
            "exclude_events": sorted(self.sys.view.exclude), "exclude_gseq": sorted(self.sys.view.exclude_gseq),
            "tool_registry": (list(tok._tool_by_key.items()), {k: list(v) for k, v in tok._keys_by_name.items()})
            if hasattr(tok, "_tool_by_key") else None,
            "model_calls": dict(self.sys.model.calls),
            "trainer": {"n_updates": tr.n_updates, "n_skipped": tr.n_skipped, "stage": tr.stage.index,
                        "run_id": tr.run_id, "weights": dict(tr.weights)},
            "semantic_updates": getattr(self.sys.semantic, "_updates", None),
            "harness_agent": {"n_updates": self.n_updates, "t_start": self.t_start},
            "log_offsets": offsets, "rng": rng_state(), "saved_pid": os.getpid(), "saved_t": time.time()}

    def save(self, path: str) -> dict:
        t0 = time.time()
        os.makedirs(path, exist_ok=True)
        life = self.sys.ops.save_full_lifetime(os.path.join(path, "lifetime"))
        snapshot_index(self.sys.index.root, os.path.join(path, "memory_index"))
        extra = self._extra_state()
        tmp = os.path.join(path, f"agent_state.pt.tmp-{os.getpid()}")
        torch.save(extra, tmp)
        os.replace(tmp, os.path.join(path, "agent_state.pt"))
        sem_bytes = self._save_semantic_cache(path) if self.sys.sc.checkpoint_semantic_cache else None
        v = self.agent.version
        n_params = sum(p.numel() for p in self.sys.model.parameters())
        meta = {"version": v, "t": time.time(), "validated": True, "validation": {"reason": "full-lifetime save"},
                "dir": os.path.abspath(path), "n_params": n_params, "bytes": life.get("bytes"),
                "sha256": core_hash(self.sys.model), "archive_count": self.agent.archive.count(),
                "decisions": self.agent.n_decisions, "save_s": time.time() - t0, "semantic_cache_bytes": sem_bytes}
        _atomic_json(os.path.join(self.run_dir, "ckpt", f"v{v}", "meta.json"), meta)
        _atomic_json(os.path.join(self.run_dir, "ckpt", "LATEST"), meta)
        self._status("running")
        return meta

    def _save_semantic_cache(self, path: str) -> Optional[int]:
        """The semantic model's content-hash caches as of the checkpoint (see SystemConfig.checkpoint_semantic_cache).
        Keys carry the encoder version, so a restored entry is only ever served to the same encoder version."""
        sem = self.sys.semantic
        if not hasattr(sem, "_cache"):
            return None
        blob = {"version": sem.version, "cache": [(k, v.detach().cpu()) for k, v in sem._cache.items()],
                "ecache": [(k, v.detach().cpu()) for k, v in getattr(sem, "_ecache", {}).items()],
                "bytes": int(getattr(sem, "_cache_bytes", 0))}
        tmp = os.path.join(path, f"semantic_cache.pt.tmp-{os.getpid()}")
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(path, "semantic_cache.pt"))
        return blob["bytes"]

    def _load_semantic_cache(self, path: str) -> Optional[dict]:
        p = os.path.join(path, "semantic_cache.pt")
        sem = self.sys.semantic
        if not os.path.exists(p) or not hasattr(sem, "_cache"):
            return None
        from collections import OrderedDict
        blob = torch.load(p, map_location="cpu", weights_only=False)
        dev = getattr(sem, "device", None)
        sem._cache = OrderedDict((k, v.to(dev)) for k, v in blob["cache"])
        sem._cache_bytes = int(blob["bytes"])
        if hasattr(sem, "_ecache"):
            sem._ecache = OrderedDict((k, v.to(dev)) for k, v in blob["ecache"])
        return {"entries": len(sem._cache), "bytes": sem._cache_bytes, "version_match": blob["version"] == sem.version}

    def load(self, path: str) -> dict:
        """Continue the lifetime saved at `path` (possibly written by another process)."""
        t0 = time.time()
        a, sysm = self.agent, self.sys
        extra = torch.load(os.path.join(path, "agent_state.pt"), map_location=sysm.model.device, weights_only=False)
        with open(os.path.join(path, "lifetime", "lifetime.json")) as fh:
            life = json.load(fh)
        # 1. working rows for the saved environments (E's loader assigns latent / fast state into them)
        a.goals = [GoalSpec(**g) for g in extra["goals"]]
        a.state = None
        a.env_ids = []
        a.reset(extra["env_ids"])
        a.goals = [GoalSpec(**g) for g in extra["goals"]]
        # 2. memory index as of the checkpoint (segments the interrupted process consolidated later are dropped)
        snap = os.path.join(path, "memory_index")
        if os.path.isdir(snap):
            root = sysm.index.root
            sysm.index.close()
            shutil.rmtree(root, ignore_errors=True)
            shutil.copytree(snap, root)
            idx = make_index(sysm.cfg, root, str(sysm.model.device))
            sysm.index = idx
            a.index = idx
            sysm.model.retriever.attach_index(idx)
            from ..memory.consolidate import Consolidator
            sysm.consolidator = Consolidator(sysm.view, idx, sysm.model.retriever)
            a.consolidator = sysm.consolidator
        # 3. the abandoned branch: archived after the checkpoint by the interrupted process
        sysm.view.exclude.update(extra.get("exclude_events", []))
        sysm.view.exclude_gseq.update(int(g) for g in extra.get("exclude_gseq", []))
        ptr = (life.get("archive") or {}).get("count", 0)
        abandoned = abandoned_after(a.archive, ptr)
        sysm.view.add(abandoned)
        a.exclude_events = sysm.view.exclude
        # 4. E's full-lifetime restore (weights, optimiser, RNG of the agent, fast / latent state, registries,
        #    version, counters, archive verification, index attachment, replay sampler, consolidator)
        res = sysm.ops.load_full_lifetime(os.path.join(path, "lifetime"))
        # 5. the remaining working state
        st = a.state
        cs = extra["cam_state"]
        dev = st.Z.device
        st.trackers = cs["trackers"]
        st.option = cs["option"].to(dev)
        st.option_steps = cs["option_steps"].to(dev)
        st.last_actions = cs["last_actions"]
        st.last_now = cs["last_now"].to(dev)
        st.fast_prev = cs["fast_prev"]
        a.episode_ids, a.steps, a.n_decisions = list(extra["episode_ids"]), list(extra["steps"]), extra["n_decisions"]
        a.fresh, a._last_fresh, a._post, a._last_eid = extra["fresh"], extra["last_fresh"], extra["post"], \
            extra["last_eid"]
        a._lp_valid, a._t0, a._funds, a._funds0 = list(extra["lp_valid"]), list(extra["t0"]), extra["funds"], \
            list(extra["funds0"])
        a._ep_decisions = list(extra["ep_decisions"])
        a.adapt_k_hist = Counter({int(k): int(v) for k, v in extra["adapt_k_hist"].items()})
        a.dropped, a.update_log, a.timing = dict(extra["dropped"]), list(extra["update_log"]), dict(extra["timing"])
        a.provenance_stats = dict(extra["provenance_stats"])
        a.loss_mag = dict(extra.get("loss_mag") or {})
        a.memory_sync_log = list(extra.get("memory_sync_log") or [])
        tok = sysm.model.tokenizer
        if extra.get("tool_registry") is not None and hasattr(tok, "_tool_by_key"):
            from collections import OrderedDict
            items, by_name = extra["tool_registry"]
            tok._tool_by_key = OrderedDict(items)
            tok._keys_by_name = {k: {kk: None for kk in v} for k, v in by_name.items()}
        sysm.model.calls = dict(extra["model_calls"])
        trs = extra["trainer"]
        tr = sysm.trainer
        tr.n_updates, tr.n_skipped, tr.run_id = int(trs["n_updates"]), int(trs["n_skipped"]), trs["run_id"]
        tr.stage = tr._resolve(int(trs["stage"]))
        tr.weights.update(trs["weights"])
        if extra.get("semantic_updates") is not None and hasattr(sysm.semantic, "_updates"):
            sysm.semantic._updates = int(extra["semantic_updates"])
        sem_cache = self._load_semantic_cache(path)
        self.n_updates = int(extra["harness_agent"]["n_updates"])
        self.t_start = float(extra["harness_agent"]["t_start"])
        self._before = param_snapshot(sysm.model) if self.track_params else None
        self._load_probe_set()
        # 6. logs back to their checkpoint offsets (the interrupted run's later rows are an abandoned branch; its
        #    decision rows are kept aside to compare with the resumed lifetime's decisions)
        abandoned_rows = []
        for f, off in extra["log_offsets"].items():
            p = os.path.join(self.run_dir, f)
            if os.path.exists(p) and os.path.getsize(p) > off:
                with open(p, "r+b") as fh:
                    if f == "decisions.jsonl":
                        fh.seek(off)
                        abandoned_rows = [json.loads(x) for x in fh.read().decode().splitlines() if x.strip()]
                    fh.truncate(off)
        if abandoned_rows:
            _append_rows(os.path.join(self.run_dir, "abandoned", f"decisions_{int(time.time())}.jsonl"),
                         abandoned_rows)
        marked = 0
        dec = [r for r in (a.archive.get([e for e, _ in abandoned]) if abandoned else []) if r.action is not None]
        for r in dec:
            sysm.recorder.fill_outcome({"abandoned_branch": True, "resumed_from": os.path.abspath(path)},
                                       event_id=r.event_id, env_id=r.env_id)
            marked += 1
        sysm.recorder.flush()
        self.resume_info = {"checkpoint": os.path.abspath(path), "pid": os.getpid(), "saved_pid": extra["saved_pid"],
                            "version": a.version, "archive_count_checkpoint": ptr,
                            "archive_count_now": a.archive.count(), "abandoned_events": len(abandoned),
                            "abandoned_decisions": len(dec), "abandoned_traces_marked": marked,
                            "abandoned_event_ids": [e for e, _ in abandoned],
                            "abandoned_decision_rows": abandoned_rows,
                            "optimizer_steps": _optimizer_steps(tr.opt), "trainer_n_updates": tr.n_updates,
                            "replay_cursor": sysm.replay.cursor, "replay_counts": dict(sysm.replay.stratum_sizes()),
                            "consolidator_cursor": sysm.consolidator.state.get("cursor"),
                            "memory": res.get("memory"), "n_decisions": a.n_decisions, "steps": list(a.steps),
                            "semantic_cache": sem_cache,
                            "load_s": time.time() - t0}
        _append(os.path.join(self.run_dir, "resume.jsonl"), self.resume_info)
        # 7. global RNG streams last: nothing above may consume randomness after they are restored
        set_rng_state(extra["rng"])
        self._status("running")
        return self.resume_info


def _optimizer_steps(opt) -> Optional[float]:
    """AdamW step counter (max over parameters with state)."""
    best = None
    for s in opt.state.values():
        step = s.get("step")
        if step is not None:
            v = float(step)
            best = v if best is None else max(best, v)
    return best
