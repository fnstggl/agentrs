"""Hacker News source via the official public API (https://github.com/HackerNews/API).

New stories from ``/v0/newstories.json``; follow-ups read ``/v0/item/<id>.json`` (score,
descendants, dead / deleted). Context: story kind (Ask / Show / link), link domain, author karma
(``/v0/user/<id>.json``, public), time. The author id is pseudonymised before storage.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from agentrs.data.observed_actor_action import ObservedAction, ObservedActorAction, pseudonym
from agentrs.live.spec import audience_label, payload
from agentrs.sensors.source_adapter import FollowUp, RawEvent, SourceAdapter, SourceDeclaration, SourceError

API = "https://hacker-news.firebaseio.com/v0"
_TAG = re.compile(r"<[^>]+>")


def hn_declaration(source_id: str = "hackernews") -> SourceDeclaration:
    return SourceDeclaration(
        source_id=source_id, name="Hacker News official API", access_method="authorized_api", operator="Y Combinator",
        allowed_data_categories=["public_posts", "public_counts", "public_profiles_followers"],
        training_authorized=True,
        authorization_ref="operator decision: public official API; operator to confirm terms permit model training",
        retention_days=365, redistribution_allowed=False, event_types=["item_created"], min_poll_interval_s=30,
        backfill_allowed=False, deletion_policy="redact", time_base="real",
        collection_limits={"requests_per_minute": 300, "max_follow_ups_per_action": 8})


class HackerNewsSource(SourceAdapter):
    def __init__(self, declaration: Optional[SourceDeclaration] = None, client: Optional[httpx.Client] = None,
                 pseudonym_key: str = "hn-key", max_age_s: float = 900.0):
        self.declaration = declaration or hn_declaration()
        self.client = client or httpx.Client(timeout=20, headers={"User-Agent": "agentrs-research-observer/0.1"})
        self.key = pseudonym_key
        self.requests = 0
        self._seen_max = 0
        self._karma: dict[str, int] = {}
        self.max_age_s = max_age_s  # never ingest stories older than this (no backfill after downtime)
        self._deletes: list[str] = []

    def now(self) -> float:
        return time.time()

    def _get(self, path: str) -> Any:
        self.requests += 1
        try:
            r = self.client.get(f"{API}{path}")
        except httpx.HTTPError as e:
            raise SourceError(f"HN {path}: {type(e).__name__}") from e
        if r.status_code != 200:
            raise SourceError(f"HN {path}: HTTP {r.status_code}", r.status_code)
        return r.json()

    def env_key(self, object_ref: Any) -> str:
        return f"hn:{object_ref}"

    def poll(self, cursor):
        """Only stories that appeared since the previous poll (never a backfill of old items)."""
        out = [RawEvent(provider_event_id=f"del:{r}", type="item_deleted", at=time.time(), object_ref=r)
               for r in self._deletes]
        self._deletes.clear()
        ids = self._get("/newstories.json") or []
        floor = int(cursor or 0)
        if floor == 0:  # first poll: start from now (ids already listed are history, not live)
            return out, max(ids) if ids else 0
        new = sorted(i for i in ids if i > floor)
        for i in new[:100]:
            it = self._get(f"/item/{i}.json")
            if not it or it.get("type") != "story" or it.get("deleted") or it.get("dead"):
                continue
            if time.time() - float(it.get("time", 0)) > self.max_age_s:
                continue
            out.append(RawEvent(provider_event_id=str(i), type="item_created", at=float(it.get("time", time.time())),
                                object_ref=str(i), data=it))
        return out, max(new) if new else floor

    def _author_karma(self, by: str) -> Optional[int]:
        if by not in self._karma:
            try:
                u = self._get(f"/user/{by}.json") or {}
                self._karma[by] = int(u.get("karma", 0))
            except SourceError:
                return None
        return self._karma[by]

    def normalize(self, raw: RawEvent):
        if raw.type == "item_deleted":
            return [FollowUp(object_ref=raw.object_ref, type="item_deleted", env_event_time=raw.at, actor_ref=None,
                             by_original_actor=True, provider_event_id=raw.provider_event_id,
                             categories=["public_posts"])]
        it = raw.data
        title = html.unescape(it.get("title", ""))
        text = html.unescape(_TAG.sub(" ", it.get("text") or "")).strip()
        url = it.get("url") or ""
        domain = urlsplit(url).hostname or ""
        kind = "ask" if title.lower().startswith("ask hn") else "show" if title.lower().startswith("show hn") else (
            "link" if url else "text")
        content = title + ("\n" + domain if domain else "") + ("\n" + text[:1500] if text else "")
        ctx = {"env": "hackernews", "space": kind, "domain": domain or "none"}
        karma = self._author_karma(it.get("by", "")) if it.get("by") else None
        if karma is not None:
            ctx["author_karma"] = karma
            ctx["audience"] = audience_label(karma, "karma")
        return [ObservedActorAction(
            source_id=self.declaration.source_id, provider_object_id=raw.object_ref, actor_type="human",
            actor_ref=pseudonym(str(it.get("by", "")), self.key), action_time=raw.at, action_time_known=True,
            observation_time=time.time(), time_base="real", context_before=ctx, context_complete=False,
            action=ObservedAction(operation="create_post", content=content), goal=None, goal_known=False,
            provenance={"source": "hackernews", "item": raw.object_ref},
            permission_metadata={"categories": ["public_posts", "public_profiles_followers"]},
            missing=["goal", "ranking_algorithm", "audience_exposure"], env_id="hackernews", group=f"hn:{kind}")]

    def fetch_state(self, object_ref: str):
        it = self._get(f"/item/{int(object_ref)}.json")
        if it is None:
            raise SourceError("item not found", 404)
        if it.get("deleted"):  # deleted by its author: propagate (the stored text is redacted)
            self._deletes.append(str(object_ref))
        removed = bool(it.get("dead") or it.get("deleted"))
        score = max(0, int(it.get("score", 1)) - 1)  # the submitter's own point is not feedback
        p = payload(reactions=score, responses=int(it.get("descendants", 0) or 0), shares=0, removed=removed)
        p["counts"].pop("shares")
        p["raw"] = {"score": it.get("score"), "descendants": it.get("descendants")}
        return p, "complete"
