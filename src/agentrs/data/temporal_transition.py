"""Time-conditioned training examples built from the delayed-outcome ledger.

One example = one action (ours or a third party's) seen with the information available at an
*information cutoff* (0 s for the initial forecast, later for updated conditional forecasts), and
every outcome label that has matured since. Labels carry their censoring status, so the losses can
mask what is unknown instead of guessing it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentrs.temporal.time_encoding import describe_duration

LabelKind = Literal["exact", "interval", "right_censored", "none"]
LabelStatus = Literal["occurred", "pending", "censored", "precluded", "closed_negative", "unknown"]


class EventLabel(BaseModel):
    """Time-to-event label, times in seconds after the action.

    exact:          event at a_s == b_s (environment-reported event time)
    interval:       event happened in (a_s, b_s] (seen absent at a_s, present at b_s)
    right_censored: no event up to b_s (status says why: pending / censored / precluded / closed)
    """

    kind: LabelKind
    a_s: float = 0.0
    b_s: float = 0.0
    status: LabelStatus = "unknown"


class CountIncrement(BaseModel):
    a_s: float
    b_s: float
    delta: float  # new items in (a_s, b_s]; never a cumulative total


class StateObs(BaseModel):
    h_s: float
    state: str


class ContinuousObs(BaseModel):
    h_s: float
    value: float


class TemporalExample(BaseModel):
    example_id: str
    action_id: str
    initial_state: str  # context available at the action (belief prompt or observed context)
    complete_action: str  # canonical action string including the exact content
    source_type: str = "self"  # self | third_party | demo
    actor_type: str = "agent"
    goal: str | None = None  # None = unknown (third-party goals are not invented)
    cutoff_elapsed_s: float = 0.0  # information cutoff, seconds after the action
    history: list[tuple[float, str]] = Field(default_factory=list)  # (s after action, event) up to the cutoff
    known_now: dict[str, Any] = Field(default_factory=dict)  # observed state at the cutoff
    # per event: last time (s after action) the event was known NOT to have happened, as of the
    # cutoff (left truncation of the likelihood). Defaults to the cutoff itself.
    truncation_s: dict[str, float] = Field(default_factory=dict)
    events: dict[str, EventLabel] = Field(default_factory=dict)
    # Evaluation metadata (not model inputs). Per event: the time up to which the event's status was
    # followed whatever it turned out to be (last status check, capped at a competing terminal
    # event), and — only for events this source times by polling — the status-check times. Scoring
    # uses them so that whether an example is scored at a horizon never depends on its outcome.
    followed_s: dict[str, float] = Field(default_factory=dict)
    checks_s: dict[str, list[float]] = Field(default_factory=dict)
    counts: dict[str, list[CountIncrement]] = Field(default_factory=dict)
    states: list[StateObs] = Field(default_factory=list)
    continuous: dict[str, list[ContinuousObs]] = Field(default_factory=dict)
    text_targets: list[tuple[float, str]] = Field(default_factory=list)  # (h, observed state text)
    censoring_metadata: dict[str, Any] = Field(default_factory=dict)
    prediction_checkpoint: str | None = None
    time_base: str = "real"
    env_id: str = ""
    split: str = "train"
    group: str = ""  # stratum (environment / action type / actor) for sampling and evaluation
    weight: float = 1.0

    def n_labels(self) -> int:
        return (sum(1 for e in self.events.values() if e.kind != "none") + sum(len(v) for v in self.counts.values())
                + len(self.states) + sum(len(v) for v in self.continuous.values()))

    def flat_rows(self) -> list[dict[str, Any]]:
        """Per-horizon view {initial_state, complete_action, horizon_seconds, target_*, target_mask, ...}."""
        rows = []
        base = {"initial_state": self.initial_state, "complete_action": self.complete_action,
                "source_type": self.source_type, "prediction_checkpoint": self.prediction_checkpoint,
                "censoring_metadata": self.censoring_metadata}
        for name, lab in self.events.items():
            if lab.kind == "none":
                continue
            rows.append({**base, "horizon_seconds": lab.b_s, "horizon": describe_duration(lab.b_s),
                         "target_event": name, "target_outcome": lab.status, "target_mask": 1,
                         "label_kind": lab.kind, "interval_start_s": lab.a_s})
        for name, incs in self.counts.items():
            for inc in incs:
                rows.append({**base, "horizon_seconds": inc.b_s, "horizon": describe_duration(inc.b_s),
                             "target_event": None, "target_outcome": {name: inc.delta}, "target_mask": 1,
                             "label_kind": "increment", "interval_start_s": inc.a_s})
        for s in self.states:
            rows.append({**base, "horizon_seconds": s.h_s, "horizon": describe_duration(s.h_s),
                         "target_observation": {"state": s.state}, "target_mask": 1, "label_kind": "state"})
        return rows
