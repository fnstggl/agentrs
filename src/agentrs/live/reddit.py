"""Reddit source via the official Data API — gated off unless the operator holds Reddit's permission.

Reddit's terms require registered, approved Data API access, and using Reddit content to train AI
models needs Reddit's separate, explicit (written) permission; public visibility of a post is not
that permission. This adapter therefore refuses to start unless the operator has set:

* ``AGENTRS_REDDIT_AUTHORIZATION_REF`` – the reference of Reddit's approval covering collection
  *and* model training for this use;
* ``AGENTRS_REDDIT_TRAINING_AUTHORIZED=true``;
* ``AGENTRS_SECRET_REDDIT_CLIENT_ID`` / ``AGENTRS_SECRET_REDDIT_CLIENT_SECRET`` (in the secret
  manager / environment, never in a chat or the repository);
* ``AGENTRS_REDDIT_USER_AGENT`` (Reddit's required ``<platform>:<app id>:<version> (by /u/<user>)``);
* ``AGENTRS_REDDIT_SUBREDDITS`` – comma-separated subreddits whose rules allow this research.

Intake: ``/r/<subs>/new`` (OAuth, app-only token). Follow-ups: ``/api/info`` (100 posts per call):
score, comments, upvote ratio, removal. No scraping and no headless browser: those would bypass the
API terms. This adapter has **not** been run against Reddit (no approved credentials were available;
reddit.com answered 403 from the build sandbox).
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

from agentrs.data.observed_actor_action import ObservedAction, ObservedActorAction, pseudonym
from agentrs.live.spec import payload
from agentrs.sensors.source_adapter import FollowUp, RawEvent, SourceAdapter, SourceDeclaration, SourceError

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"


def reddit_declaration(authorization_ref: str, training_authorized: bool, source_id: str = "reddit") -> SourceDeclaration:
    return SourceDeclaration(
        source_id=source_id, name="Reddit Data API (approved)", access_method="authorized_api", operator="Reddit, Inc.",
        allowed_data_categories=["public_posts", "public_counts"], training_authorized=training_authorized,
        authorization_ref=authorization_ref or None, retention_days=30, redistribution_allowed=False,
        event_types=["item_created", "item_deleted"], min_poll_interval_s=60, backfill_allowed=False,
        deletion_policy="purge", time_base="real",
        collection_limits={"requests_per_minute": 60, "max_follow_ups_per_action": 8})


class RedditSource(SourceAdapter):
    def __init__(self, declaration: SourceDeclaration, client_id: str, client_secret: str, user_agent: str,
                 subreddits: list[str], pseudonym_key: str, client: Optional[httpx.Client] = None,
                 max_age_s: float = 900.0):
        if not (declaration.training_authorized and declaration.authorization_ref):
            raise PermissionError("reddit: collection for model training needs Reddit's explicit authorisation "
                                  "(AGENTRS_REDDIT_AUTHORIZATION_REF and AGENTRS_REDDIT_TRAINING_AUTHORIZED=true)")
        if not subreddits:
            raise ValueError("reddit: name the subreddits to observe")
        self.declaration = declaration
        self.cid, self.secret, self.ua = client_id, client_secret, user_agent
        self.subs = subreddits
        self.key = pseudonym_key
        self.client = client or httpx.Client(timeout=20)
        self.max_age_s = max_age_s
        self._token: tuple[str, float] | None = None
        self._deletes: list[str] = []
        self.requests = 0

    def now(self) -> float:
        return time.time()

    def env_key(self, object_ref: Any) -> str:
        return f"reddit:{object_ref}"

    def _auth(self) -> str:
        if self._token and self._token[1] > time.time() + 60:
            return self._token[0]
        r = self.client.post(TOKEN_URL, data={"grant_type": "client_credentials"}, auth=(self.cid, self.secret),
                             headers={"User-Agent": self.ua})
        self.requests += 1
        if r.status_code != 200:
            raise SourceError(f"reddit token: HTTP {r.status_code}", r.status_code)
        js = r.json()
        self._token = (js["access_token"], time.time() + float(js.get("expires_in", 3600)))
        return self._token[0]

    def _get(self, path: str, params: dict) -> dict:
        r = self.client.get(f"{API}{path}", params={**params, "raw_json": 1},
                            headers={"Authorization": f"Bearer {self._auth()}", "User-Agent": self.ua})
        self.requests += 1
        if r.status_code == 429:
            raise SourceError("reddit rate limited", 429, float(r.headers.get("x-ratelimit-reset", 60)))
        if r.status_code != 200:
            raise SourceError(f"reddit {path}: HTTP {r.status_code}", r.status_code)
        return r.json()

    def poll(self, cursor):
        out = [RawEvent(provider_event_id=f"del:{r}", type="item_deleted", at=time.time(), object_ref=r)
               for r in self._deletes]
        self._deletes.clear()
        js = self._get(f"/r/{'+'.join(self.subs)}/new", {"limit": 100})
        floor = float(cursor or 0)
        posts = [c["data"] for c in js.get("data", {}).get("children", [])]
        newest = max([p.get("created_utc", 0) for p in posts] + [floor])
        if floor == 0:
            return out, newest  # start live: what is already listed is history
        for p in sorted(posts, key=lambda p: p.get("created_utc", 0)):
            if p.get("created_utc", 0) <= floor or time.time() - p.get("created_utc", 0) > self.max_age_s:
                continue
            if p.get("over_18"):
                continue
            out.append(RawEvent(provider_event_id=p["name"], type="item_created", at=float(p["created_utc"]),
                                object_ref=p["name"], data=p))
        return out, newest

    def normalize(self, raw: RawEvent):
        if raw.type == "item_deleted":
            return [FollowUp(object_ref=raw.object_ref, type="item_deleted", env_event_time=raw.at, actor_ref=None,
                             by_original_actor=True, provider_event_id=raw.provider_event_id,
                             categories=["public_posts"])]
        p = raw.data
        content = p.get("title", "") + ("\n" + p.get("domain", "") if not p.get("is_self") else "") + \
            ("\n" + (p.get("selftext") or "")[:1500] if p.get("selftext") else "")
        ctx = {"env": "reddit", "space": f"r/{p.get('subreddit', '')}", "kind": "self" if p.get("is_self") else "link",
               "flair": p.get("link_flair_text") or "none"}
        return [ObservedActorAction(
            source_id=self.declaration.source_id, provider_object_id=raw.object_ref, actor_type="unknown",
            actor_ref=pseudonym(str(p.get("author", "")), self.key), action_time=raw.at, action_time_known=True,
            observation_time=time.time(), time_base="real", context_before=ctx, context_complete=False,
            action=ObservedAction(operation="create_post", content=content), goal=None, goal_known=False,
            provenance={"source": "reddit-data-api", "authorization": self.declaration.authorization_ref},
            permission_metadata={"categories": ["public_posts"]},
            missing=["goal", "ranking_algorithm", "audience_exposure"], env_id="reddit",
            group=f"reddit:{p.get('subreddit', '')}")]

    def fetch_state(self, object_ref: str):
        js = self._get("/api/info", {"id": object_ref})
        items = [c["data"] for c in js.get("data", {}).get("children", [])]
        if not items:
            raise SourceError("not found", 404)
        p = items[0]
        if p.get("author") == "[deleted]" and p.get("selftext") in ("[deleted]", "") and not p.get("removed_by_category"):
            self._deletes.append(object_ref)
        removed = bool(p.get("removed_by_category"))
        up = max(0, int(p.get("score", 1)) - 1)
        out = payload(reactions=up, responses=int(p.get("num_comments", 0)), shares=0, removed=removed)
        out["counts"].pop("shares")
        return out, "complete"


def reddit_source_from_env(pseudonym_key: str) -> RedditSource:
    ref = os.environ.get("AGENTRS_REDDIT_AUTHORIZATION_REF", "")
    ok = os.environ.get("AGENTRS_REDDIT_TRAINING_AUTHORIZED", "").lower() == "true"
    if not (ref and ok):
        raise PermissionError(
            "reddit source refused: set AGENTRS_REDDIT_AUTHORIZATION_REF (Reddit's approval for Data API access "
            "and AI-model training) and AGENTRS_REDDIT_TRAINING_AUTHORIZED=true; public visibility is not permission")
    cid, sec = os.environ.get("AGENTRS_SECRET_REDDIT_CLIENT_ID"), os.environ.get("AGENTRS_SECRET_REDDIT_CLIENT_SECRET")
    ua = os.environ.get("AGENTRS_REDDIT_USER_AGENT")
    if not (cid and sec and ua):
        raise PermissionError("reddit source: client id / secret / user agent are not configured")
    subs = [s.strip() for s in os.environ.get("AGENTRS_REDDIT_SUBREDDITS", "").split(",") if s.strip()]
    return RedditSource(reddit_declaration(ref, ok), cid, sec, ua, subs, pseudonym_key)
