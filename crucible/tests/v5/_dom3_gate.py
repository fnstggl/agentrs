"""Shared measurement of the per-section gates SP-9.11 .. SP-9.15 (workstream I-3).

The five shared metrics come from the framework (`crucible.domains.base.section_metrics`: integrated reference runs
through `crucible.experience.make(2, ...)` in the default mode), run on EVERY configuration of the domain (three
training, one compositional, one structural evaluation configuration); `required_mechanics_exercised` /
`_missing` are then taken over the intersection of the configurations (a mechanic counts only when it is exercised
in every configuration's run), which is stricter than the framework's union. Every run must be clean (no failed
process, no invariant violation).

The participant is needed: each domain names its PARTICIPANT_MECHANICS (measured only on the participant's own
services, exchanges, initiatives or venture); the same world (first configuration, same seed) with an idle seat - the
null baseline - must exercise none of them and must fail the objective. The other mechanics are the autonomous world
the participant works in (publishers and queries, providers and congestion, cascading damage)."""
from __future__ import annotations

import importlib

from crucible.domains import base


def section(domain: str, seed: int) -> tuple[dict, list[dict], str]:
    import crucible.domains as D
    D.load_all()
    d = base.REGISTRY[domain]
    m, runs = base.section_metrics(domain, configs=[c.name for c in d.configs], seeds=[seed])
    per = {r["config"]: {k for k, v in r["mechanics"]["counts"].items() if v > 0} for r in runs}
    inter = [k for k in d.required_mechanics if all(k in per[c] for c in per)]
    m["required_mechanics_exercised"] = len(inter)
    m["required_mechanics_missing"] = len(d.required_mechanics) - len(inter)
    m["configurations_run"] = len(per)
    m["clean_runs"] = sum(1 for r in runs if r["proc_failed"] == 0 and r["invariant_violations"] == 0)
    # the participant is needed: its mechanics under the null baseline (idle seat) on the first configuration
    part = tuple(getattr(importlib.import_module(d.evaluate.__module__), "PARTICIPANT_MECHANICS", ()))
    null = base.cached_run(domain, d.configs[0].name, seed, "null")
    null_ex = {k for k, v in null["mechanics"]["counts"].items() if v > 0}
    m["participant_mechanics_listed"] = len(part)
    m["participant_mechanics_exercised_by_null"] = sum(1 for k in part if k in null_ex)
    m["null_mechanics_exercised"] = len(null_ex)
    m["null_baseline_fails"] = not null["success"]
    notes = " | ".join(f"{r['config']}: success={r['success']} counts={r['mechanics']['counts']} "
                       f"details={str((r['outcome'] or {}).get('details'))[:600]}" for r in runs)
    notes = f"null ({d.configs[0].name}): counts={null['mechanics']['counts']} | " + notes
    return m, runs, notes


def assert_participant_needed(m: dict, notes: str) -> None:
    """The gates' participant check (see the module docstring)."""
    assert m["participant_mechanics_listed"] >= 1, m
    assert m["participant_mechanics_exercised_by_null"] == 0, (m, notes[:2000])
    assert m["null_baseline_fails"], (m, notes[:2000])
