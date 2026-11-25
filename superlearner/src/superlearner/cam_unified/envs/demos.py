"""Demonstrations for the declared "demo-init" condition (architecture decision 2): transcripts of the
**non-privileged** ExampleInductor recorded through the same `OpenWorldAdapter` records a learned agent receives,
as contract `EventRecord`s with ``action.source = SCRIPTED`` and a demo tag in ``extra``.

Rules enforced here, not by convention:

* privileged controls (Crucible's `Builder`, `Employer`, `Composer`, `BountyHunter`, `SchedulePolicy`: they use
  hidden reference solvers or private task fields) are refused (`PrivilegedDemoError`) - their transcripts are
  feasibility evidence, never training data;
* the worlds must be training families (`worldgen.SplitGuard`): demonstrations never come from validation or test
  worlds;
* every record carries ``extra = {"demo_condition": "demo-init", "demonstrator": <name>, "privileged": False,
  "verified_return_episode": ...}`` so a training run can include or exclude them explicitly and report the
  demo-init arm beside the no-demo arm; `load_demos` re-checks the tags.

The sink is a JSONL file and/or any callable taking a list of EventRecords (e.g. workstream E's archive
``ArchiveP.append``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from ..interfaces import ActionSource, EventRecord, Provenance
from .agent_protocol import PolicyAgent, is_privileged
from .harness import run_lifetime
from .openworld import open_world
from .openworld_eval import OpenWorldEvaluator
from .worldgen import SplitGuard

DEMO_CONDITION = "demo-init"


class PrivilegedDemoError(RuntimeError):
    """A privileged control was about to produce training demonstrations."""


def _default_policy():
    from crucible.usefulness.agents import ExampleInductor
    return ExampleInductor()


def record_demos(tasks: Sequence[Any], out_path: Optional[str] = None, *,
                 policy_factory: Callable[[], Any] = _default_policy, think_s: float = 600.0,
                 sink: Optional[Callable[[list], None]] = None, guard: Optional[SplitGuard] = None,
                 days: Optional[float] = None, seed: int = 0) -> dict:
    """Run the demonstrator once per task (one OpenWorld env per task, in parallel worlds) and record every decision.
    Returns {"episodes", "records", "successes", "returns", "path"}."""
    probe = policy_factory()
    if is_privileged(probe):
        raise PrivilegedDemoError(f"{type(probe).__name__} is a privileged control: its transcripts are never "
                                  f"training data")
    g = guard or SplitGuard()
    for t in tasks:
        g.check_task(t)
    ad, op = open_world(list(tasks), seed=seed)
    ev = OpenWorldEvaluator(op)
    agent = PolicyAgent(lambda b: policy_factory(), source=ActionSource.SCRIPTED)
    name = getattr(probe, "name", type(probe).__name__)
    buf: list[list] = [[] for _ in tasks]

    def on_step(k, tr):
        for b, rec in enumerate(tr.records):
            if rec.action is None and rec.extra.get("skipped") and tr.dones[b]:
                continue
            rec.extra.update(demo_condition=DEMO_CONDITION, demonstrator=name, privileged=False)
            buf[b].append(rec)
    try:
        res = run_lifetime(agent, ad, ev, think_s=think_s, days=days, on_step=on_step)
    finally:
        ad.close()
    n = 0
    fh = open(out_path, "a") if out_path else None
    try:
        for b, recs in enumerate(buf):
            for r in recs:
                r.extra["verified_return_episode"] = res.returns[b]
                r.extra["episode_success"] = bool(res.final[b]["achieved"])
            if fh:
                for r in recs:
                    fh.write(r.to_json() + "\n")
            if sink is not None and recs:
                sink(recs)
            n += len(recs)
    finally:
        if fh:
            fh.close()
    return {"episodes": len(tasks), "records": n, "successes": [bool(f["achieved"]) for f in res.final],
            "returns": res.returns, "success_day": res.success_day, "path": out_path, "demonstrator": name}


def load_demos(path: str) -> Iterator[EventRecord]:
    """Stream demonstration records, refusing anything not tagged as a non-privileged demo-init transcript."""
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = EventRecord.from_dict(json.loads(line))
            ex = r.extra or {}
            if ex.get("demo_condition") != DEMO_CONDITION or ex.get("privileged") is not False:
                raise PrivilegedDemoError(f"record {r.event_id} is not a non-privileged {DEMO_CONDITION} record")
            if r.provenance is not Provenance.REAL or (r.action is not None and r.action.source is not ActionSource.SCRIPTED):
                raise PrivilegedDemoError(f"record {r.event_id} has provenance {r.provenance} / source "
                                          f"{r.action.source if r.action else None}")
            yield r
