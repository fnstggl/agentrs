"""Crucible 5.0 red team (workstream O): probes against every merged 5.0 module, the Gate W categories they
measure, the ratchet of findings known to be open, and the measurement of Gate W.

    from crucible.redteam.v5 import probes, run, gate_metrics
    report = run(probes())                  # each probe in its own fresh world (v5_world_factory)
    metrics = gate_metrics(report)          # {"W.1": {...}, ..., "W.9": {...}}

The probe contract is the 4.0 one (`crucible.redteam.framework`): an operator sets up, the attacker acts only
through what an in-world participant can do (tool calls as principals it controls, code it runs in the sandbox,
content it uploads, waiting), and an independent evaluator reads authoritative state for the verdict. An integrity
post-check (invariants, ledger, conservation, exact replay) runs after every probe.
"""
from __future__ import annotations

from typing import Iterable

from .base import GATES, V5Probe  # noqa: F401
from .worlds import v5_spec, v5_world_factory  # noqa: F401

MODULES = ("substrate", "sandbox", "network", "isolation", "crossworld", "private", "population", "social", "commerce",
           "innovation", "experience", "rapid")


def probes(include_slow: bool = True, include_sandbox: bool = True, only: Iterable[str] = ()) -> list:
    """The 5.0 probe catalogue (names are unique; `only` restricts to some of MODULES)."""
    import importlib
    out = []
    for m in (only or MODULES):
        mod = importlib.import_module(f"{__name__}.{m}")
        out.extend(mod.probes())
    if not include_slow:
        out = [p for p in out if not p.slow]
    if not include_sandbox:
        out = [p for p in out if not p.needs_sandbox]
    names = [p.name for p in out]
    assert len(names) == len(set(names)), "duplicate probe names"
    return out


def run(ps: Iterable | None = None, progress=None) -> dict:
    """Run probes (default: the whole 5.0 catalogue) with the 5.0 world factory; returns the red-team report."""
    from ..framework import run_probes
    return run_probes(v5_world_factory, list(ps) if ps is not None else probes(), progress=progress)


def __getattr__(name):          # lazy: the ratchet and the metrics live in suite.py
    if name in ("KNOWN_OPEN", "MUST_HOLD", "gate_metrics", "legacy_gates", "FIXED", "ratchet", "catalogue"):
        from . import suite
        return getattr(suite, name)
    raise AttributeError(name)
