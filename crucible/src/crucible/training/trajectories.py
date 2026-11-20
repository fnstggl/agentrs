"""Export agent trajectories and world event logs for offline RL, imitation and world-model training.

Formats
* **JSONL** (`crucible-traj/1`): one record per agent step::

      {"format", "world_id", "spec_hash", "seq_before", "seq_after", "actor", "policy", "policy_kind",
       "policy_version", "step", "t", "observation", "tools", "n_tools", "action", "tool", "args", "result",
       "ok", "reward", "events", "wall_ms"}

  `tools` (names of the tools available at that step) and the `seq_*` fields are filled by `RecordingRunner`;
  for plain `Runner` trajectories they are null (the Runner only records the number of tools).
  World event logs export as `crucible-events/1` records (`Event.to_dict()` plus the format tag).
* **NPZ** (columnar, `numpy.savez_compressed`, loadable with ``allow_pickle=False``): scalar columns (step, t,
  reward, ok, seq, wall_ms, n_tools, n_events), categorical columns as integer codes plus a vocabulary
  (actor, policy, tool), and the nested fields (observation, args, result, tools, events) as UTF-8 JSON blobs
  addressed by offsets. `load_npz` restores exactly the JSONL records.

`ImitationDataset` turns step records into (rendered observation text, target action JSON) pairs for behaviour
cloning / SFT of language-model or sequence policies.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable, Iterator

import numpy as np

from ..agents.base import Runner, Step, Trajectory, _compact

TRAJ_FORMAT = "crucible-traj/1"
EVENTS_FORMAT = "crucible-events/1"
_SCALARS = ("step", "t", "seq_before", "seq_after", "n_tools", "reward", "wall_ms")
_CATS = ("actor", "policy", "policy_kind", "policy_version", "tool", "world_id", "spec_hash")
_BLOBS = ("observation", "tools", "args", "result", "events", "action")


def _dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


# ====================================================================== recording runner
class RecordingRunner(Runner):
    """A `Runner` that additionally records, per step, the names of the available tools, the world sequence
    number before / after the action, and the policy kind / version. Same scheduling as `Runner`."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.extra: dict[str, list[dict]] = {actor: [] for actor, _ in self.agents}

    def round(self) -> None:
        order = self.rng.permutation(len(self.agents))
        for i in order:
            actor, pol = self.agents[i]
            t0 = time.perf_counter()
            seq0 = self.world.seq
            obs = self.observe(actor)
            tools = self.tools(actor)
            action = pol.act(obs, tools)
            res, evs = None, []
            if action is not None:
                r = self.world.execute(actor, action)
                res, evs = r.to_dict(), r.events
                pol.observe_result(action, res)
            reward = 0.0
            obj = self.objectives.get(actor)
            if obj is not None:
                reward = obj.step_reward(self.world, actor)
            if self.record:
                self.trajs[actor].steps.append(Step(self.world.now, actor, pol.name, _compact(obs), len(tools), action,
                                                    res, evs, reward, (time.perf_counter() - t0) * 1000))
                self.extra[actor].append({"tools": [t["name"] for t in tools], "seq_before": seq0,
                                          "seq_after": self.world.seq, "policy_kind": getattr(pol, "kind", None),
                                          "policy_version": str(getattr(pol, "version", ""))})
        self.world.advance_time(self.world.now + self.dt)


# ====================================================================== records
def step_records(trajs, extra: dict | None = None, policy_version: str | None = None) -> list[dict]:
    """Flatten trajectories (a `Trajectory`, a list of them, a Runner's `trajs` dict, or a `Runner`) into step
    records. `extra` is `RecordingRunner.extra` (taken automatically from a RecordingRunner)."""
    if isinstance(trajs, Runner):
        extra = getattr(trajs, "extra", None) if extra is None else extra
        trajs = trajs.trajs
    if isinstance(trajs, Trajectory):
        trajs = [trajs]
    elif isinstance(trajs, dict):
        trajs = [trajs[k] for k in sorted(trajs)]
    out = []
    for tr in trajs:
        ex = (extra or {}).get(tr.actor) or []
        for i, s in enumerate(tr.steps):
            e = ex[i] if i < len(ex) else {}
            a = s.action if isinstance(s.action, dict) else None
            out.append({"format": TRAJ_FORMAT, "world_id": tr.world_id, "spec_hash": tr.world_spec_hash,
                        "seq_before": e.get("seq_before"), "seq_after": e.get("seq_after"), "actor": s.actor,
                        "policy": s.policy, "policy_kind": e.get("policy_kind"),
                        "policy_version": policy_version if policy_version is not None else e.get("policy_version"),
                        "step": i, "t": int(s.t), "observation": _jsonable(s.obs), "tools": e.get("tools"),
                        "n_tools": int(s.n_tools), "action": _jsonable(a), "tool": a.get("tool") if a else None,
                        "args": _jsonable(a.get("args")) if a else None, "result": _jsonable(s.result),
                        "ok": None if s.result is None else bool(s.result.get("ok")), "reward": float(s.reward),
                        "events": list(s.events), "wall_ms": round(float(s.wall_ms), 3)})
    return out


def _jsonable(x):
    if x is None:
        return None
    return json.loads(_dumps(x))


def event_records(world, from_seq: int = 0, to_seq: int | None = None, visible_to: str | None = None) -> list[dict]:
    """World event log records (operator view, or only what `visible_to` may see)."""
    from ..analytics import world_events
    ps = world.principals_of(visible_to) if visible_to else None
    out = []
    for ev in world_events(world, from_seq, to_seq):
        if ps is not None and not world.visible(ev, ps):
            continue
        d = ev.to_dict()
        d["format"] = EVENTS_FORMAT
        d["world_id"] = world.world_id
        out.append(d)
    return out


# ====================================================================== JSONL
def write_jsonl(records: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(_dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def export_jsonl(trajs, path: str, **kw) -> int:
    """Trajectories -> JSONL (one record per step). Returns the number of records."""
    return write_jsonl(step_records(trajs, **kw), path)


def export_events_jsonl(world, path: str, **kw) -> int:
    return write_jsonl(event_records(world, **kw), path)


# ====================================================================== NPZ (columnar)
def _blob(values: list) -> tuple[np.ndarray, np.ndarray]:
    parts = [_dumps(v).encode() for v in values]
    offs = np.zeros(len(parts) + 1, np.int64)
    if parts:
        offs[1:] = np.cumsum([len(p) for p in parts])
    return np.frombuffer(b"".join(parts), dtype=np.uint8).copy(), offs


def _unblob(data: np.ndarray, offs: np.ndarray) -> list:
    b = data.tobytes()
    return [json.loads(b[offs[i]:offs[i + 1]].decode()) for i in range(len(offs) - 1)]


def save_npz(records: list[dict], path: str, kind: str = "traj") -> dict:
    """Columnar export of step records (kind "traj") or event records (kind "events")."""
    cols: dict[str, np.ndarray] = {}
    n = len(records)
    meta = {"format": TRAJ_FORMAT if kind == "traj" else EVENTS_FORMAT, "n": n, "kind": kind, "vocab": {}}
    if kind == "traj":
        for k in _SCALARS:
            vals = [r.get(k) for r in records]
            if k in ("reward", "wall_ms"):
                cols[k] = np.array([np.nan if v is None else v for v in vals], np.float64)
            else:
                cols[k] = np.array([-1 if v is None else v for v in vals], np.int64)
        cols["ok"] = np.array([-1 if r.get("ok") is None else int(r["ok"]) for r in records], np.int8)
        cols["n_events"] = np.array([len(r.get("events") or []) for r in records], np.int32)
        cat_keys, blob_keys = _CATS, _BLOBS
    else:
        cols["seq"] = np.array([r["seq"] for r in records], np.int64)
        cols["t"] = np.array([r["t"] for r in records], np.int64)
        cols["cmd"] = np.array([-1 if r.get("cmd") is None else r["cmd"] for r in records], np.int64)
        cat_keys, blob_keys = ("type", "actor", "world_id"), ("payload", "vis", "hash")
    for k in cat_keys:
        vals = [r.get(k) for r in records]
        vocab = sorted({v for v in vals if v is not None})
        idx = {v: i for i, v in enumerate(vocab)}
        cols[f"{k}_code"] = np.array([idx[v] if v is not None else -1 for v in vals], np.int32)
        meta["vocab"][k] = vocab
    for k in blob_keys:
        data, offs = _blob([r.get(k) for r in records])
        cols[f"{k}_data"] = data
        cols[f"{k}_offsets"] = offs
    cols["meta_json"] = np.frombuffer(_dumps(meta).encode(), dtype=np.uint8).copy()
    np.savez_compressed(path, **cols)
    return meta


def load_npz(path: str) -> list[dict]:
    """Inverse of `save_npz` (records equal to the exported JSONL records)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(z["meta_json"].tobytes().decode())
    n = meta["n"]
    kind = meta["kind"]
    recs: list[dict] = [{} for _ in range(n)]
    if kind == "traj":
        for k in _SCALARS:
            col = z[k]
            for i in range(n):
                v = col[i]
                if k in ("reward", "wall_ms"):
                    recs[i][k] = None if np.isnan(v) else float(v)
                else:
                    recs[i][k] = None if (v == -1 and k in ("seq_before", "seq_after")) else int(v)
        ok = z["ok"]
        for i in range(n):
            recs[i]["ok"] = None if ok[i] == -1 else bool(ok[i])
            recs[i]["format"] = meta["format"]
        cat_keys, blob_keys = _CATS, _BLOBS
    else:
        for i in range(n):
            recs[i]["seq"] = int(z["seq"][i])
            recs[i]["t"] = int(z["t"][i])
            recs[i]["cmd"] = None if z["cmd"][i] == -1 else int(z["cmd"][i])
            recs[i]["format"] = meta["format"]
        cat_keys, blob_keys = ("type", "actor", "world_id"), ("payload", "vis", "hash")
    for k in cat_keys:
        vocab = meta["vocab"][k]
        codes = z[f"{k}_code"]
        for i in range(n):
            recs[i][k] = None if codes[i] == -1 else vocab[int(codes[i])]
    for k in blob_keys:
        vals = _unblob(z[f"{k}_data"], z[f"{k}_offsets"])
        for i in range(n):
            recs[i][k] = vals[i]
    return recs


def export_npz(trajs, path: str, **kw) -> dict:
    return save_npz(step_records(trajs, **kw), path, "traj")


def export_events_npz(world, path: str, **kw) -> dict:
    return save_npz(event_records(world, **kw), path, "events")


def columns(path: str) -> dict[str, np.ndarray]:
    """The raw numeric columns of an NPZ export (for vectorised offline-RL pipelines)."""
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if not k.endswith(("_data", "_offsets")) and k != "meta_json"}


# ====================================================================== imitation
def render_step_observation(rec: dict, max_events: int = 20, max_tools: int = 200) -> str:
    """Deterministic text for one step's observation (same renderer as the agent API) plus the tool names."""
    from ..api.render import render_observation
    obs = dict(rec.get("observation") or {})
    summary = {k: v for k, v in obs.items() if k not in ("time", "principal", "events", "events_dropped")}
    o = {"time": obs.get("time", rec.get("t", 0)), "principal": obs.get("principal", rec.get("actor")),
         "summary": summary, "events": obs.get("events", []), "dropped": obs.get("events_dropped", 0),
         "last_result": None, "tools_changed": False, "tools": None}
    text = render_observation(o, max_events=max_events)
    tools = rec.get("tools")
    if tools:
        text += "\ntools: " + ", ".join(tools[:max_tools]) + (" ..." if len(tools) > max_tools else "")
    else:
        text += f"\ntools: {rec.get('n_tools', '?')} available"
    return text


class ImitationDataset:
    """(rendered observation text, target action JSON) pairs from step records.

    `source` is a list of records, a JSONL / NPZ path, or trajectories. Steps where the policy waited are
    excluded unless `include_waits` (target "null"); `only_ok` keeps only actions the world accepted."""

    def __init__(self, source, include_waits: bool = False, only_ok: bool = False, policies=None,
                 render=render_step_observation):
        if isinstance(source, str):
            recs = load_npz(source) if source.endswith(".npz") else read_jsonl(source)
        elif isinstance(source, list) and (not source or isinstance(source[0], dict)):
            recs = source
        else:
            recs = step_records(source)
        pol = set(policies) if policies else None
        self.records = [r for r in recs if (include_waits or r.get("action") is not None)
                        and (not only_ok or r.get("ok")) and (pol is None or r.get("policy") in pol)]
        self.render = render

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[str, str]:
        r = self.records[i]
        a = r.get("action")
        target = "null" if a is None else _dumps({"tool": a.get("tool"), "args": a.get("args", {})})
        return self.render(r), target

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for i in range(len(self)):
            yield self[i]

    def to_jsonl(self, path: str) -> int:
        return write_jsonl(({"prompt": p, "completion": c, "actor": r["actor"], "policy": r["policy"],
                             "t": r["t"]} for (p, c), r in zip(self, self.records)), path)
