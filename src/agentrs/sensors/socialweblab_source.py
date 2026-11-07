"""Observation source for SocialWebLab's public interface.

Uses only what any user of the application can see: the public activity feed
(``/api/events``) and public item pages (``/api/items/<id>``), through a client that is either the
HTTP API (``HttpLabClient``) or a direct read of the same public views (``DirectLabClient``, for
fast accelerated-time experiments; identical data). Author identities are pseudonymised; reply
texts are not stored (only their kind), following data minimisation.
"""

from __future__ import annotations

from typing import Any

from agentrs.data.observed_actor_action import (
    ObservedAction,
    ObservedActorAction,
    pseudonym,
)
from agentrs.sensors.permissions import redact_pii
from agentrs.sensors.source_adapter import (
    FollowUp,
    RawEvent,
    SourceAdapter,
    SourceDeclaration,
    SourceError,
)
from agentrs.socialweblab.apps import SPACES

FOLLOW_UP_TYPES = {"response_created": "public_replies", "answer_posted": "public_replies",
                   "reaction_up": "public_reactions", "reaction_down": "public_reactions", "item_edited": "public_posts",
                   "item_hidden": "public_moderation", "item_removed": "public_moderation",
                   "item_restored": "public_moderation", "item_deleted": "public_posts", "item_resolved": "public_posts"}


def socialweblab_declaration(source_id: str, time_base: str, rpm: float = 600.0, **kw) -> SourceDeclaration:
    return SourceDeclaration(
        source_id=source_id, name="SocialWebLab public activity feed", access_method="self_hosted_feed",
        operator="this project (self-hosted, simulated users only)", simulated=True,
        allowed_data_categories=["public_posts", "public_replies", "public_reactions", "public_counts",
                                 "public_moderation", "public_profiles_followers"],
        training_authorized=True, authorization_ref=None, retention_days=365, redistribution_allowed=False,
        event_types=sorted(FOLLOW_UP_TYPES) + ["item_created"], min_poll_interval_s=10.0, backfill_allowed=True,
        deletion_policy="redact", time_base=time_base,
        collection_limits={"requests_per_minute": rpm, "max_follow_ups_per_action": 16}, **kw)


class DirectLabClient:
    """Reads the lab's *public* views directly (same data as the HTTP API, no privileged state)."""

    def __init__(self, lab):
        self.lab = lab

    def events(self, since: int, limit: int = 200) -> dict:
        return {"events": self.lab.public_events(since, limit), "now": self.lab.now()}

    def item(self, item_id: int) -> dict:
        return self.lab.public_item(int(item_id))

    def now(self) -> float:
        return self.lab.now()


class HttpLabClient:
    """Adapter over ``socialweblab.server.LabClient`` (HTTP or in-process transport) with an observer key."""

    def __init__(self, lab_client, now_fn=None):
        self.c = lab_client
        self._now = now_fn
        self._last_now: float | None = None

    @staticmethod
    def _unwrap(res):
        status, body = res if isinstance(res, tuple) else (200, res)
        if status != 200:
            retry = None
            if isinstance(body, dict):
                retry = body.get("retry_after")
            raise SourceError(f"HTTP {status}: {body}", status, retry)
        return body

    def events(self, since: int, limit: int = 200) -> dict:
        body = self._unwrap(self.c.events(since, limit))
        self._last_now = body.get("now", self._last_now)
        return body

    def item(self, item_id: int) -> dict:
        body = self._unwrap(self.c.item(int(item_id)))
        self._last_now = body.get("now", self._last_now)
        return body

    def now(self) -> float:
        if self._now:
            return self._now()
        if self._last_now is None:
            self.events(10 ** 12, 1)
        return float(self._last_now)


def payload_from_item(it: dict) -> dict:
    author = it["author"]
    others = [r for r in it.get("responses", []) if r["author"] != author]
    answers = [r for r in others if r["kind"] == "answer"]
    state = {"deleted": "removed"}.get(it["state"], it["state"])
    c = it["counts"]
    occurred = {"positive_feedback": c["reactions"] > 0, "resolved": it["status"] == "resolved",
                "removed": state != "visible", "edited": it["version"] > 1}
    if state == "visible" or others:  # responses of removed items are not shown: unknown, not absent
        occurred["first_response"] = bool(others)
        occurred["answered"] = bool(answers)
    return {"counts": {"responses": c["responses"], "reactions": c["reactions"], "views": c["views"]},
            "state": state, "continuous": {"net_score": c["reactions"] - c.get("downs", 0)},
            "occurred": occurred,
            "occurred_at": {k: v for k, v in (("first_response", min((r["created_at"] for r in others), default=None)),
                                              ("answered", min((r["created_at"] for r in answers), default=None)))
                            if v is not None}}


class SocialWebLabSource(SourceAdapter):
    def __init__(self, declaration: SourceDeclaration, client, pseudonym_key: str, env_key_prefix: str = "swl"):
        self.declaration = declaration
        self.client = client
        self.key = pseudonym_key
        self.env_key_prefix = env_key_prefix
        self.requests = 0

    def now(self) -> float:
        return self.client.now()

    def env_key(self, object_ref: Any) -> str:
        return f"{self.env_key_prefix}:{object_ref}"

    def poll(self, cursor):
        self.requests += 1
        body = self.client.events(int(cursor or 0), 500)
        evs = body["events"]
        raws = [RawEvent(provider_event_id=str(e["id"]), type=e["type"], at=e["at"], object_ref=str(e["item_id"]),
                         data=e) for e in evs]
        return raws, (evs[-1]["id"] if evs else int(cursor or 0))

    def fetch_state(self, object_ref: str):
        self.requests += 1
        it = self.client.item(int(object_ref))
        return payload_from_item(it), "complete"

    def normalize(self, raw: RawEvent):
        e = raw.data
        p = e.get("payload", {})
        if raw.type == "item_created":
            space = p.get("space", "")
            app = p.get("app", SPACES.get(space, ("forum",))[0])
            content = redact_pii((p.get("title") or "") + "\n" + (p.get("body") or ""))
            return [ObservedActorAction(
                source_id=self.declaration.source_id, provider_object_id=str(e["item_id"]),
                actor_type="simulated_user" if self.declaration.simulated else "unknown",
                actor_ref=pseudonym(str(p.get("author", "")), self.key), action_time=e["at"], action_time_known=True,
                observation_time=self.now(), time_base=self.declaration.time_base,
                context_before={"env": f"socialweblab/{app}", "space": space,
                                "topic": ", ".join(SPACES.get(space, ("", []))[1][:5]),
                                "author_followers": p.get("author_followers", 0)},
                context_complete=False, action=ObservedAction(operation=p.get("operation", "create_post"), content=content),
                goal=None, goal_known=False,
                provenance={"source": self.declaration.source_id, "provider_event_id": raw.provider_event_id,
                            "simulated": self.declaration.simulated},
                permission_metadata={"categories": ["public_posts", "public_profiles_followers"]},
                missing=["goal", "author_history", "feed_ranking", "audience_exposure"],
                env_id="socialweblab", env_version="1", group=f"{app}:{space}")]
        if raw.type in FOLLOW_UP_TYPES:
            return [FollowUp(object_ref=str(e["item_id"]), type=raw.type, env_event_time=e["at"],
                             actor_ref=pseudonym(str(e.get("actor", "")), self.key),
                             by_original_actor=p.get("by_author") if raw.type == "response_created" else (
                                 True if raw.type in ("item_edited", "item_deleted", "item_resolved") else
                                 (False if raw.type.startswith("reaction") or raw.type == "answer_posted" else None)),
                             provider_event_id=raw.provider_event_id,
                             payload={"summary": p.get("kind") or p.get("state") or ""},
                             categories=[FOLLOW_UP_TYPES[raw.type]])]
        return []


def lab_temporal_spec():
    from agentrs.temporal.temporal_world_model import TemporalSpec

    return TemporalSpec(events=["first_response", "answered", "positive_feedback", "resolved", "removed", "edited"],
                        counts=["responses", "reactions", "views"], states=["visible", "hidden", "removed"],
                        continuous=["net_score"])
