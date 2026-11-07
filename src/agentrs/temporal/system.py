"""Wiring for the observational-learning loop on SocialWebLab (used by tests, experiments and the
continual orchestrator): lab → public source → sensor network → ledger/scheduler → forecasts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agentrs.sensors.observation_network import (
    ForecastService,
    NetworkConfig,
    ObservationSensorNetwork,
)
from agentrs.sensors.socialweblab_source import (
    DirectLabClient,
    SocialWebLabSource,
    socialweblab_declaration,
)
from agentrs.sensors.source_registry import SourceRegistry


def lab_source(lab, source_id: str = "swl-feed", client=None, rpm: float = 600.0) -> SocialWebLabSource:
    decl = socialweblab_declaration(source_id, lab.time_base, rpm=rpm)
    return SocialWebLabSource(decl, client or DirectLabClient(lab), pseudonym_key=f"key-{lab.lab_id}",
                              env_key_prefix=f"swl-{lab.lab_id}")


def build_network(workdir: str | Path, lab, model=None, cfg: NetworkConfig | None = None,
                  split_fn: Callable | None = None, client=None, source_id: str = "swl-feed",
                  forecast_batch: int = 64) -> ObservationSensorNetwork:
    reg = SourceRegistry(Path(workdir) / "sources.json")
    src = lab_source(lab, source_id, client)
    reg.register(src, coverage_tags=["socialweblab", "forum", "helpdesk", "codereview"])
    fc = None
    if model is not None:
        from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger

        ledger = DelayedOutcomeLedger(Path(workdir) / "ledger.sqlite")
        fc = ForecastService(model, ledger, batch=forecast_batch)
        return ObservationSensorNetwork(workdir, reg, fc, cfg, split_fn=split_fn, ledger=ledger)
    return ObservationSensorNetwork(workdir, reg, None, cfg, split_fn=split_fn)


def run_lab(lab, net: ObservationSensorNetwork, duration_s: float, step_s: float = 600.0,
            on_step: Callable[[float], None] | None = None) -> dict:
    """Advance the lab's virtual clock in steps, letting the sensor network collect after each."""
    t_end = lab.now() + duration_s
    steps = 0
    while lab.now() < t_end - 1e-6:
        lab.advance(min(step_s, t_end - lab.now()))
        net.step()
        if on_step:
            on_step(lab.now())
        steps += 1
    return {"steps": steps, **net.stats}
