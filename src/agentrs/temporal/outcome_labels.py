"""Turn ledger rows (observations + events) into censoring-aware temporal training examples.

Rules (times are seconds after the action, in the action's time base):

* An event occurrence with an environment-reported time is an **exact** label; an occurrence only
  detected by polling is **interval**-censored between the last observation that showed it absent
  and the first that showed it present.
* No occurrence yet → **right-censored** at the last time it was known absent, with a status:
  ``pending`` (window still open), ``closed_negative`` (outcome window closed while observed),
  ``censored`` (observation stopped early: job failures, revoked source), or ``precluded`` (a
  competing terminal event such as removal happened first — cause-specific censoring at its time).
  None of these is ever turned into a definite failure label.
* Counts become **increments** between consecutive complete observations (items start at 0), so
  repeated observation never double counts; a decrease (deleted replies) is recorded as a revision
  and contributes a zero increment.
* **Information cutoffs**: an example at cutoff ``c`` sees only events and observations whose
  *observation* time is <= action_time + c (never the environment time of something observed later),
  so later events cannot leak into an earlier forecast.
* **Evaluation metadata**: per event, how long its status was followed whatever it turned out to be
  (``followed_s``) and, for events the source times only by polling, the status-check times
  (``checks_s``). Whether an event is polled is decided per *source* (``polled_events``), from all
  of its actions, so it never depends on one example's own outcome.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence

from pydantic import BaseModel, Field

from agentrs.data.temporal_transition import (
    ContinuousObs,
    CountIncrement,
    EventLabel,
    StateObs,
    TemporalExample,
)
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.temporal_world_model import TemporalSpec


class OutcomeMapping(BaseModel):
    # ledger event type -> outcome event name (several types may map to one outcome)
    event_types: dict[str, str] = Field(default_factory=lambda: {
        "response_created": "first_response", "answer_posted": "answered", "reaction_up": "positive_feedback",
        "item_resolved": "resolved", "item_removed": "removed", "item_hidden": "removed", "item_deleted": "removed",
        "item_edited": "edited"})
    competing: list[str] = Field(default_factory=lambda: ["removed"])
    only_by_others: list[str] = Field(default_factory=lambda: ["first_response", "answered", "positive_feedback"])
    only_by_original: list[str] = Field(default_factory=lambda: ["edited"])
    counts_start_at_zero: bool = True


def default_event_text(ev: dict) -> str:
    who = "original actor" if ev.get("by_original_actor") else "another actor"
    p = ev.get("payload") or {}
    extra = f": {p['summary']}" if p.get("summary") else ""
    return f"{ev['type']} by {who}{extra}"


def _mapped(ev: dict, mapping: OutcomeMapping) -> str | None:
    name = mapping.event_types.get(ev["type"])
    if name is None:
        return None
    orig = ev.get("by_original_actor")
    if name in mapping.only_by_others and orig:
        return None
    if name in mapping.only_by_original and orig is not None and not orig:
        return None
    return name


def polled_events(ledger: DelayedOutcomeLedger, source_id: str, mapping: OutcomeMapping | None = None,
                  ) -> frozenset[str]:
    """Outcome events this source times only by polling: some occurrence, on any of the source's
    actions, came with no exact time (no event-feed record with an environment time and no timestamp
    on the observed item), i.e. it would be an interval label. Cached per ledger state."""
    mapping = mapping or OutcomeMapping()
    marks = ledger._q("SELECT (SELECT MAX(rowid) FROM observations) AS o, (SELECT MAX(rowid) FROM events) AS e")[0]
    cache = ledger.__dict__.setdefault("_polled_cache", {})
    key = (source_id, marks["o"], marks["e"], mapping.model_dump_json())
    if key in cache:
        return cache[key]
    first_ev: dict[tuple[str, str], bool] = {}  # (action, outcome) -> first record has an environment time
    for e in ledger._q("SELECT e.action_id, e.type, e.env_event_time, e.observed_time, e.by_original_actor "
                       "FROM events e JOIN actions a ON a.action_id = e.action_id WHERE a.source_id = ? "
                       "ORDER BY COALESCE(e.env_event_time, e.observed_time), e.observed_time", (source_id,)):
        name = _mapped(dict(e), mapping)
        if name is not None:
            first_ev.setdefault((e["action_id"], name), e["env_event_time"] is not None)
    polled = {name for (_, name), exact in first_ev.items() if not exact}
    seen: set[tuple[str, str]] = set()
    for o in ledger._q("SELECT o.action_id, o.payload_json FROM observations o JOIN actions a "
                       "ON a.action_id = o.action_id WHERE a.source_id = ? AND o.completeness != 'missing' "
                       "ORDER BY o.actual_time", (source_id,)):
        p = json.loads(o["payload_json"] or "{}")
        for name, occ in (p.get("occurred") or {}).items():
            k = (o["action_id"], name)
            if not occ or k in seen or k in first_ev:
                continue
            seen.add(k)
            if (p.get("occurred_at") or {}).get(name) is None:
                polled.add(name)
    cache[key] = frozenset(polled)
    return cache[key]


def build_examples(ledger: DelayedOutcomeLedger, action_id: str, spec: TemporalSpec,
                   mapping: OutcomeMapping | None = None, as_of: float | None = None,
                   cutoffs: Sequence[float] | str = (0.0,), event_text: Callable[[dict], str] = default_event_text,
                   max_cutoffs: int = 3, polled: Collection[str] | None = None) -> list[TemporalExample]:
    """``cutoffs``: seconds after the action, or ``"observations"`` for one updated forecast after
    each of the first ``max_cutoffs`` observations (plus the initial one). ``polled``: events the
    source times only by polling (default: inferred with ``polled_events``)."""
    mapping = mapping or OutcomeMapping()
    a = ledger.action(action_id)
    if a is None or not a["training_eligible"] or a["status"] == "purged":
        return []
    if polled is None:
        polled = polled_events(ledger, a["source_id"], mapping)
    t0 = a["action_time"]
    T = as_of if as_of is not None else float("inf")
    obs = [o for o in ledger.observations(action_id, observed_before=T) if o["completeness"] != "missing"]
    evs = ledger.events(action_id, observed_before=T)
    closed = a["status"] == "closed" and (a["closed_time"] or 0) <= T
    if cutoffs == "observations":
        cs = [0.0] + [o["elapsed"] for o in obs[:max_cutoffs] if o["elapsed"] > 0]
    else:
        cs = [float(c) for c in cutoffs]
    labels = _event_labels(spec, mapping, t0, obs, evs, closed, a)
    followed, checks = _follow_up(spec, mapping, obs, labels, polled)
    counts = _count_increments(spec, mapping, obs)
    states = [StateObs(h_s=o["elapsed"], state=o["payload"]["state"]) for o in obs
              if o["payload"].get("state") in spec.states]
    cont = {k: [ContinuousObs(h_s=o["elapsed"], value=float(o["payload"]["continuous"][k])) for o in obs
                if k in (o["payload"].get("continuous") or {})] for k in spec.continuous}
    out = []
    for c in sorted(set(cs)):
        cut_time = t0 + c
        if cut_time > T:
            continue
        hist_ev = [e for e in evs if e["observed_time"] <= cut_time + 1e-9]
        hist_obs = [o for o in obs if o["actual_time"] <= cut_time + 1e-9]
        occurred = sorted({n for n in (_mapped(e, mapping) for e in hist_ev) if n} |
                          {k for o in hist_obs for k, v in (o["payload"].get("occurred") or {}).items() if v})
        known = {}
        if hist_obs:
            last = hist_obs[-1]["payload"]
            known.update({k: v for k, v in (last.get("counts") or {}).items()})
            if last.get("state"):
                known["state"] = last["state"]
        if c > 0:
            known["occurred"] = occurred
        ev_labels, trunc = {}, {}
        for name, lab in labels.items():
            if name in occurred and c > 0:
                continue  # already known at this cutoff
            ev_labels[name] = lab
            if c > 0:
                absent = [o["elapsed"] for o in hist_obs if (o["payload"].get("occurred") or {}).get(name) is False]
                trunc[name] = max([0.0] + absent)
        ex = TemporalExample(
            example_id=f"{action_id}@{c:.0f}", action_id=action_id, initial_state=a["context_text"] or "",
            complete_action=a["action_str"], source_type=a["source_type"], actor_type=a["actor_type"], goal=a["goal"],
            cutoff_elapsed_s=c, history=[((e["env_event_time"] or e["observed_time"]) - t0, event_text(e))
                                         for e in hist_ev],
            known_now=known if c > 0 else {}, truncation_s=trunc, events=ev_labels,
            followed_s={n: v for n, v in followed.items() if n in ev_labels},
            checks_s={n: v for n, v in checks.items() if n in ev_labels},
            counts={k: [i for i in v if i.a_s >= (max([o["elapsed"] for o in hist_obs] + [0.0]) if c > 0 else 0) - 1e-6]
                    for k, v in counts.items()},
            states=[s for s in states if s.h_s > c or c == 0], continuous=cont,
            censoring_metadata={n: lab.status for n, lab in labels.items()},
            prediction_checkpoint=a["policy_checkpoint"], time_base=a["time_base"], env_id=a["env_id"] or "",
            split=a["split"] or "train", group=a["grp"] or "")
        out.append(ex)
    return out


def _event_labels(spec, mapping, t0, obs, evs, closed, a) -> dict[str, EventLabel]:
    first: dict[str, tuple[str, float, float]] = {}
    for e in evs:
        name = _mapped(e, mapping)
        if name is None or name not in spec.events or name in first:
            continue
        if e["env_event_time"] is not None:
            t = e["env_event_time"] - t0
            first[name] = ("exact", t, t)
        else:
            b = e["observed_time"] - t0
            prev = [o["elapsed"] for o in obs if o["elapsed"] < b and (o["payload"].get("occurred") or {}).get(name) is False]
            first[name] = ("interval", max([0.0] + prev), b)
    for name in spec.events:
        if name in first:
            continue
        for o in obs:  # polling-only detection
            occ = (o["payload"].get("occurred") or {}).get(name)
            if occ:
                at = (o["payload"].get("occurred_at") or {}).get(name)
                if at is not None:
                    first[name] = ("exact", at - t0, at - t0)
                else:
                    prev = [p["elapsed"] for p in obs if p["elapsed"] < o["elapsed"]
                            and (p["payload"].get("occurred") or {}).get(name) is False]
                    first[name] = ("interval", max([0.0] + prev), o["elapsed"])
                break
    any_seen = max([o["elapsed"] for o in obs] + [0.0])
    window = (a["window_end"] - t0) if a["window_end"] else None
    comp = {n: first[n][2] for n in mapping.competing if n in first}
    labels = {}
    for name in spec.events:
        if name in first:
            kind, s, t = first[name]
            # an occurrence after a competing terminal event cannot be attributed (e.g. reply to a removed post)
            prior = [r for n, r in comp.items() if n != name and r < s]
            if not prior:
                labels[name] = EventLabel(kind=kind, a_s=s, b_s=t, status="occurred")
                continue
        prec = [r for n, r in comp.items() if n != name]
        if prec:
            labels[name] = EventLabel(kind="right_censored", a_s=min(prec), b_s=min(prec), status="precluded")
            continue
        keyed = [o for o in obs if name in (o["payload"].get("occurred") or {})]
        # last time known absent: explicit "not occurred" observations; sources without that field
        # (complete event feeds) are known absent up to their last observation
        last_seen = max([o["elapsed"] for o in keyed if o["payload"]["occurred"][name] is False] + [0.0]) if keyed \
            else any_seen
        if last_seen <= 0:
            labels[name] = EventLabel(kind="none", status="unknown")
            continue
        if closed and window is not None and last_seen >= window - 1e-6:
            status = "closed_negative"
        elif closed:
            status = "censored"
        else:
            status = "pending"
        labels[name] = EventLabel(kind="right_censored", a_s=last_seen, b_s=last_seen, status=status)
    return labels


def _follow_up(spec, mapping, obs, labels, polled) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Per event: how long its status was followed, regardless of the outcome (the last observation
    that reported it — or the last observation at all for sources whose payload has no per-event
    field — capped at the time of a competing terminal event), and for polled events the times of
    the observations that reported its status."""
    any_seen = max([o["elapsed"] for o in obs] + [0.0])
    comp = {n: lab.b_s for n, lab in labels.items() if n in mapping.competing and lab.status == "occurred"}
    followed, checks = {}, {}
    for name in spec.events:
        keyed = [o["elapsed"] for o in obs if name in (o["payload"].get("occurred") or {})]
        until = max(keyed) if keyed else any_seen
        others = [r for n, r in comp.items() if n != name]
        followed[name] = min([until] + others)
        if name in polled:
            checks[name] = sorted(set(keyed))
    return followed, checks


def _count_increments(spec, mapping, obs) -> dict[str, list[CountIncrement]]:
    out: dict[str, list[CountIncrement]] = {k: [] for k in spec.counts}
    for k in spec.counts:
        prev_t, prev_v = (0.0, 0.0) if mapping.counts_start_at_zero else (None, None)
        for o in obs:
            if o["completeness"] != "complete":
                continue
            v = (o["payload"].get("counts") or {}).get(k)
            if v is None:
                continue
            v = float(v)
            if prev_t is not None and o["elapsed"] > prev_t:
                out[k].append(CountIncrement(a_s=prev_t, b_s=o["elapsed"], delta=max(0.0, v - prev_v)))
            prev_t, prev_v = o["elapsed"], max(v, prev_v if prev_v is not None else v)
    return out
