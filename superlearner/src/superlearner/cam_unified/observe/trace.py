"""Machine-readable decision traces (spec 18.1; manifest A20 with the dashboard).

`TraceRecorder` stores one `DecisionTrace` per consequential decision as JSON lines under a ``traces/`` directory and
links them to the rest of the system after the fact, without rewriting history:

* ``record(trace)``                         — the decision as it was made (context refs, retrieved memory, candidates,
                                               predictions, uncertainty, imagined alternatives, selected action, belief
                                               before/after, policy version, immediate outcome);
* ``fill_outcome(event_id=..., outcome=...)`` — consequences observed later (delayed outcomes, attributed by the
                                               action's event id); successive fills merge, and list-valued keys append;
* ``add_update_refs(event_ids, update_id)`` — the learner update(s) that consumed the decision's event.

Each of these appends a patch line; `iter_traces` / `load` merge patches into the base records, so the files are
append-only and several processes may record into one directory (each recorder instance writes its own
``part-<pid>-<n>.jsonl`` files). `to_parquet` exports flattened traces (nested fields as JSON strings) with pyarrow.

`retrospective_summary` builds readable per-lifetime summaries from recorded actions, predictions and outcomes
only. Every summary is labelled as an interpretation of the record, not access to the model's internal reasoning
(spec 18.1), and every statement it makes is a statistic computed from the trace.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Iterable, Iterator, Optional

from ..interfaces import DecisionTrace

INTERPRETATION_LABEL = ("Interpretation generated from the recorded actions, predictions and outcomes; "
                        "it is not direct access to the model's internal (numerical) reasoning.")


class TraceRecorder:
    def __init__(self, root: str, rows_per_file: int = 50_000, fsync: bool = False, name: Optional[str] = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.rows_per_file = rows_per_file
        self.fsync = fsync
        self.name = name or f"{os.getpid()}"
        self._fh = None
        self._rows = 0
        self._part = 0
        self.counts = Counter()

    # ------------------------------------------------------------ writing
    def _out(self):
        if self._fh is None or self._rows >= self.rows_per_file:
            if self._fh is not None:
                self._fh.close()
            while True:
                self._part += 1
                p = os.path.join(self.root, f"part-{self.name}-{self._part:05d}.jsonl")
                if not os.path.exists(p):
                    break
            self._fh = open(p, "a")
            self._rows = 0
        return self._fh

    def _write(self, obj: dict) -> None:
        fh = self._out()
        fh.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
        self._rows += 1
        self.counts[obj["type"]] += 1

    def record(self, trace: DecisionTrace) -> None:
        wire = trace.__dict__.get("_wire")       # exact JSON as received from the queue (no re-serialisation)
        if wire is not None:
            fh = self._out()
            fh.write('{"type":"trace","d":' + wire + "}\n")
            self._rows += 1
            self.counts["trace"] += 1
            return
        self._write({"type": "trace", "d": trace.to_dict()})

    def fill_outcome(self, outcome: dict, event_id: Optional[str] = None, decision_id: Optional[str] = None,
                     t: Optional[float] = None) -> None:
        if event_id is None and decision_id is None:
            raise ValueError("fill_outcome needs event_id or decision_id")
        self._write({"type": "outcome", "event_id": event_id, "decision_id": decision_id, "outcome": outcome,
                     "t_wall": t if t is not None else time.time()})

    def set_field(self, field: str, value, event_id: Optional[str] = None,
                  decision_id: Optional[str] = None) -> None:
        """Late value of a trace field (e.g. ``belief_after`` computed once the outcome arrived)."""
        if field not in DecisionTrace.__dataclass_fields__:
            raise KeyError(field)
        self._write({"type": "field", "event_id": event_id, "decision_id": decision_id, "field": field,
                     "value": value})

    def add_update_refs(self, event_ids: Iterable[str], update_id: str) -> None:
        self._write({"type": "update", "event_ids": list(event_ids), "update_id": update_id})

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------ reading
    def _lines(self) -> Iterator[dict]:
        self.flush()
        for fn in sorted(os.listdir(self.root)):
            if fn.startswith("part-") and fn.endswith(".jsonl"):
                with open(os.path.join(self.root, fn)) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue          # a line being written by a live process

    def iter_traces(self, env_id: Optional[str] = None) -> Iterator[DecisionTrace]:
        return iter(load_traces(self.root, env_id=env_id, flush=self.flush))

    def load(self, env_id: Optional[str] = None) -> list:
        return load_traces(self.root, env_id=env_id, flush=self.flush)

    def to_parquet(self, path: str, env_id: Optional[str] = None) -> str:
        return traces_to_parquet(self.load(env_id), path)


def _merge_outcome(dst: dict, new: dict) -> None:
    for k, v in new.items():
        if isinstance(v, list) and isinstance(dst.get(k), list):
            dst[k] = dst[k] + v
        else:
            dst[k] = v


def load_traces(root: str, env_id: Optional[str] = None, flush=None) -> list:
    """Merge base traces and patches from every part file under ``root``; ordered by (env_id, t_sim)."""
    if flush is not None:
        flush()
    if not os.path.isdir(root):
        return []
    base: dict[str, dict] = {}
    by_event: dict[str, str] = {}
    patches: list = []
    for fn in sorted(os.listdir(root)):
        if not (fn.startswith("part-") and fn.endswith(".jsonl")):
            continue
        with open(os.path.join(root, fn)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("type") == "trace":
                    d = o["d"]
                    base[d["decision_id"]] = d
                    by_event[d["event_id"]] = d["decision_id"]
                else:
                    patches.append(o)
    for o in patches:
        typ = o.get("type")
        if typ == "update":
            for eid in o["event_ids"]:
                did = by_event.get(eid)
                if did is not None and o["update_id"] not in base[did]["update_refs"]:
                    base[did]["update_refs"].append(o["update_id"])
            continue
        did = o.get("decision_id") or by_event.get(o.get("event_id"))
        if did is None or did not in base:
            continue
        if typ == "outcome":
            _merge_outcome(base[did].setdefault("outcome", {}), o["outcome"])
        elif typ == "field":
            base[did][o["field"]] = o["value"]
    out = [DecisionTrace(**d) for d in base.values() if env_id is None or d["env_id"] == env_id]
    out.sort(key=lambda t: (t.env_id, t.t_sim, t.decision_id))
    return out


def traces_to_parquet(traces: list, path: str) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    scalar = ("decision_id", "event_id", "env_id", "policy_version", "t_sim", "objective", "result_ref", "source")
    cols: dict = {k: [] for k in scalar}
    nested = [f for f in DecisionTrace.__dataclass_fields__ if f not in scalar]
    for k in nested:
        cols[k] = []
    for t in traces:
        d = t.to_dict()
        for k in scalar:
            cols[k].append(d[k])
        for k in nested:
            cols[k].append(json.dumps(d[k], default=str))
    table = pa.table(cols)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table, path)
    return path


# ============================================================================ retrospective summaries

def _action_label(sel: Optional[dict]) -> str:
    if not sel:
        return "(no action)"
    args = sel.get("args") or {}
    a = ",".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{sel.get('tool')}({a})" if a else str(sel.get("tool"))


def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def prediction_errors(traces: list) -> list:
    """[(trace, key, predicted, observed, z or None)] for every outcome key present in both ``predicted`` and
    ``outcome`` of a trace (numeric values only)."""
    out = []
    for t in traces:
        for k, p in (t.predicted or {}).items():
            pv, ov = _num(p), _num((t.outcome or {}).get(k))
            if pv is None or ov is None:
                continue
            u = _num((t.uncertainty or {}).get(k))
            z = (ov - pv) / u if u and u > 0 else None
            out.append((t, k, pv, ov, z))
    return out


def retrospective_summary(traces: list, window: int = 0) -> dict:
    """Readable, trace-grounded summary of one lifetime (all traces of one env_id, in time order).

    Returns {"label", "n_decisions", "versions", "phases": [{...}], "largest_misses": [...], "statements": [str]}.
    ``window`` decisions per phase (default: a quarter of the lifetime, at least 1)."""
    traces = sorted(traces, key=lambda t: (t.t_sim, t.decision_id))
    n = len(traces)
    res = {"label": INTERPRETATION_LABEL, "n_decisions": n, "versions": [], "phases": [], "largest_misses": [],
           "statements": []}
    if n == 0:
        res["statements"].append("No decisions were recorded for this lifetime.")
        return res
    w = window or max(1, math.ceil(n / 4))
    versions = []
    for t in traces:
        if not versions or versions[-1] != t.policy_version:
            versions.append(t.policy_version)
    res["versions"] = versions
    for i in range(0, n, w):
        ph = traces[i:i + w]
        acts = Counter(_action_label(t.selected) for t in ph)
        oks = [t.outcome.get("ok") for t in ph if isinstance(t.outcome, dict) and "ok" in t.outcome]
        rews = [_num(t.outcome.get("reward")) for t in ph if isinstance(t.outcome, dict)]
        rews = [r for r in rews if r is not None]
        res["phases"].append({
            "from_t": ph[0].t_sim, "to_t": ph[-1].t_sim, "n": len(ph),
            "versions": sorted({t.policy_version for t in ph}),
            "top_actions": acts.most_common(3),
            "ok_rate": (sum(1 for o in oks if o) / len(oks)) if oks else None,
            "mean_reward": (sum(rews) / len(rews)) if rews else None,
            "updates_consumed": len({u for t in ph for u in t.update_refs})})
    errs = prediction_errors(traces)
    errs.sort(key=lambda e: -abs(e[3] - e[2]))
    res["largest_misses"] = [{"decision_id": t.decision_id, "t_sim": t.t_sim, "key": k, "predicted": p,
                              "observed": o, "z": z, "action": _action_label(t.selected)}
                             for t, k, p, o, z in errs[:5]]
    st = res["statements"]
    st.append(f"{n} decisions from t={traces[0].t_sim:.3f} to t={traces[-1].t_sim:.3f} simulated days under "
              f"policy versions {versions[0]}..{versions[-1]} ({len(versions)} distinct, in order of use).")
    first, last = res["phases"][0], res["phases"][-1]
    st.append(f"Most frequent actions in the first {first['n']} decisions: "
              + ", ".join(f"{a} x{c}" for a, c in first["top_actions"]) + ".")
    if len(res["phases"]) > 1:
        st.append(f"Most frequent actions in the last {last['n']} decisions: "
                  + ", ".join(f"{a} x{c}" for a, c in last["top_actions"]) + ".")
        if first["mean_reward"] is not None and last["mean_reward"] is not None:
            st.append(f"Mean recorded reward changed from {first['mean_reward']:.3f} (first phase) to "
                      f"{last['mean_reward']:.3f} (last phase).")
        if first["ok_rate"] is not None and last["ok_rate"] is not None:
            st.append(f"Execution success rate changed from {first['ok_rate']:.2f} to {last['ok_rate']:.2f}.")
    if res["largest_misses"]:
        m = res["largest_misses"][0]
        st.append(f"Largest recorded prediction miss: '{m['key']}' after {m['action']} at t={m['t_sim']:.3f}: "
                  f"predicted {m['predicted']:.3f}, observed {m['observed']:.3f}.")
    nupd = len({u for t in traces for u in t.update_refs})
    st.append(f"Experience from this lifetime was consumed by {nupd} learner updates.")
    return res


def lifetimes(traces: list) -> dict:
    """env_id -> traces in time order."""
    d = defaultdict(list)
    for t in traces:
        d[t.env_id].append(t)
    for v in d.values():
        v.sort(key=lambda t: (t.t_sim, t.decision_id))
    return dict(d)
