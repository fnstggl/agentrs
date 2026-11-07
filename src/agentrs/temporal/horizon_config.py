"""Multi-resolution prediction horizons and per-action-type observation profiles.

The defaults are a starting point, not a universal observation schedule: each action type gets a
profile with only the horizons that are informative for it (a click needs seconds, a forum post
needs days), and any other duration can always be queried from the temporal world model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agentrs.temporal.time_encoding import (
    DAY,
    HOUR,
    MINUTE,
    describe_duration,
    parse_duration,
)

DEFAULT_TIERS: dict[str, list[float]] = {
    "immediate": [0.0, 1.0, 5.0],
    "short": [10.0, 30.0, MINUTE, 5 * MINUTE],
    "intermediate": [15 * MINUTE, HOUR, 6 * HOUR],
    "long": [DAY, 3 * DAY, 7 * DAY],
    "extended": [14 * DAY, 30 * DAY],
}

ALL_DEFAULT_HORIZONS: list[float] = sorted({h for hs in DEFAULT_TIERS.values() for h in hs})


class HorizonProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    horizons: list[float]  # seconds after the action at which to observe / evaluate
    window_s: float  # outcome window: after this the action's outcomes are considered final

    def labels(self) -> list[str]:
        return [describe_duration(h) for h in self.horizons]


def _p(name: str, hs: list, window) -> HorizonProfile:
    return HorizonProfile(name=name, horizons=[parse_duration(h) for h in hs], window_s=parse_duration(window))


DEFAULT_PROFILES: dict[str, HorizonProfile] = {
    "ui": _p("ui", [0, 1, 5], "30s"),
    "api_call": _p("api_call", [0, 5, 30, "1m", "5m"], "15m"),
    "code_run": _p("code_run", [0, 30, "5m", "15m", "1h", "6h"], "1d"),
    "email": _p("email", ["5m", "1h", "6h", "1d", "3d"], "7d"),
    "forum_post": _p("forum_post", ["1m", "5m", "15m", "1h", "6h", "1d", "3d", "7d"], "14d"),
    "forum_reply": _p("forum_reply", ["5m", "1h", "6h", "1d", "3d"], "7d"),
    "ticket": _p("ticket", ["15m", "1h", "6h", "1d", "3d"], "7d"),
    "code_review": _p("code_review", ["5m", "15m", "1h", "6h", "1d", "3d"], "7d"),
    "research": _p("research", ["1h", "6h", "1d", "3d", "7d", "14d"], "30d"),
}

OPERATION_PROFILE = {
    "click": "ui", "type": "ui", "select": "ui", "scroll": "ui", "navigate": "ui", "wait": "ui", "read": "ui",
    "create_post": "forum_post", "post_reply": "forum_reply", "reply": "forum_reply",
    "open_ticket": "ticket", "submit_patch": "code_review", "send_email": "email", "submit_code": "code_run",
    "apply_config": "api_call", "write_document": "email",
}


class HorizonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, HorizonProfile] = Field(default_factory=lambda: dict(DEFAULT_PROFILES))
    operation_profile: dict[str, str] = Field(default_factory=lambda: dict(OPERATION_PROFILE))
    default_profile: str = "forum_post"

    def profile_for(self, operation: str | None) -> HorizonProfile:
        name = self.operation_profile.get(operation or "", self.default_profile)
        return self.profiles[name]

    def horizons_for(self, operation: str | None) -> list[float]:
        return list(self.profile_for(operation).horizons)
