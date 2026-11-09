"""Shared outcome definition for live public social data (one world model across platforms).

Events (time-to-event, censoring-aware): first_response (someone replied / commented),
positive_feedback (at least one like / upvote beyond the author's own), engagement_10 and
engagement_100 (reactions + responses + shares reached 10 / 100), removed (deleted / killed).
Counts (increments between observations): reactions, responses, shares.
"""

from __future__ import annotations

from agentrs.temporal.horizon_config import HorizonConfig, HorizonProfile, parse_duration
from agentrs.temporal.temporal_world_model import TemporalSpec

EVENTS = ["first_response", "positive_feedback", "engagement_10", "engagement_100", "removed"]
COUNTS = ["reactions", "responses", "shares"]


def live_spec() -> TemporalSpec:
    return TemporalSpec(events=list(EVENTS), counts=list(COUNTS), states=["visible", "removed"], continuous=[])


LIVE_HORIZONS = [parse_duration(x) for x in ("10m", "1h", "6h", "1d", "3d")]


def live_horizon_config(horizons: list | None = None, window: str | float = "4d") -> HorizonConfig:
    hs = [parse_duration(h) for h in horizons] if horizons else list(LIVE_HORIZONS)
    prof = HorizonProfile(name="live_post", horizons=hs, window_s=parse_duration(window))
    return HorizonConfig(profiles={"live_post": prof}, operation_profile={"create_post": "live_post"},
                         default_profile="live_post")


def payload(reactions: int, responses: int, shares: int, removed: bool, first_response_at=None) -> dict:
    total = reactions + responses + shares
    p = {"counts": {"reactions": reactions, "responses": responses, "shares": shares},
         "state": "removed" if removed else "visible",
         "occurred": {"positive_feedback": reactions > 0, "engagement_10": total >= 10, "engagement_100": total >= 100,
                      "removed": removed},
         "occurred_at": {}}
    if not removed:  # replies of removed items are not visible: unknown, not absent
        p["occurred"]["first_response"] = responses > 0
    if first_response_at is not None:
        p["occurred_at"]["first_response"] = first_response_at
    return p
