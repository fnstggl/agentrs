"""Shared measurement of the per-section gates SP-9.11 .. SP-9.15 (workstream I-3).

The five shared metrics come from the framework (`crucible.domains.base.section_metrics`: integrated reference runs
through `crucible.experience.make(2, ...)` in the default mode), run on EVERY configuration of the domain (three
training, one compositional, one structural evaluation configuration); `required_mechanics_exercised` /
`_missing` are then taken over the intersection of the configurations (a mechanic counts only when it is exercised
in every configuration's run), which is stricter than the framework's union. Every run must be clean (no failed
process, no invariant violation)."""
from __future__ import annotations

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
    notes = " | ".join(f"{r['config']}: success={r['success']} counts={r['mechanics']['counts']} "
                       f"details={str((r['outcome'] or {}).get('details'))[:600]}" for r in runs)
    return m, runs, notes
