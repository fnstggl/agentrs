"""Bluesky source: new public posts from Jetstream (live, sampled), outcomes from the public AppView.

* Intake: a background thread holds one Jetstream WebSocket (``app.bsky.feed.post`` commits only)
  and keeps a random sample of new top-level posts (``target_per_min``; the keep probability adapts
  to the stream's rate and is recorded with every post). Nothing older than the connection is read:
  no backfill.
* Context: author follower / post counts via ``app.bsky.actor.getProfiles`` (25 per call). Authors
  carrying the ``!no-unauthenticated`` self-label (they asked not to be shown to logged-out viewers)
  are skipped.
* Follow-ups: ``app.bsky.feed.getPosts`` (25 posts per call, prefetched in batches) gives like /
  repost / quote / reply counts. A post that disappeared, or a Jetstream ``delete`` for a tracked post,
  or an account deactivation / deletion, becomes an ``item_deleted`` follow-up so the network redacts
  the stored text (deletion propagation).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
import threading
import time
from collections import deque
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from agentrs.data.observed_actor_action import ObservedAction, ObservedActorAction, pseudonym
from agentrs.live.spec import audience_label, payload
from agentrs.sensors.source_adapter import FollowUp, RawEvent, SourceAdapter, SourceDeclaration, SourceError

APPVIEW = "https://public.api.bsky.app/xrpc"
JETSTREAM = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
OPT_OUT_LABELS = {"!no-unauthenticated"}
EMBED_KIND = {"app.bsky.embed.images": "images", "app.bsky.embed.external": "link", "app.bsky.embed.record": "quote",
              "app.bsky.embed.recordWithMedia": "quote+media", "app.bsky.embed.video": "video"}


def bluesky_declaration(source_id: str = "bluesky") -> SourceDeclaration:
    return SourceDeclaration(
        source_id=source_id, name="Bluesky Jetstream + public AppView", access_method="authorized_api",
        operator="Bluesky Social PBC (public AT Protocol endpoints)",
        allowed_data_categories=["public_posts", "public_counts", "public_profiles_followers"],
        training_authorized=True,
        authorization_ref="operator decision: public AT Protocol endpoints; operator to confirm terms permit model training",
        retention_days=30, redistribution_allowed=False, event_types=["item_created", "item_deleted"],
        min_poll_interval_s=5, backfill_allowed=False, deletion_policy="redact", time_base="real",
        collection_limits={"requests_per_minute": 300, "max_follow_ups_per_action": 8})


def ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=ca) if ca and os.path.exists(ca) else ssl.create_default_context()


def post_uri(did: str, rkey: str) -> str:
    return f"at://{did}/app.bsky.feed.post/{rkey}"


class JetstreamSampler:
    """Background WebSocket consumer. ``handle(msg)`` is separated from the socket for testing."""

    def __init__(self, target_per_min: float = 20.0, langs: Optional[list[str]] = None, url: str = JETSTREAM,
                 seed: int = 0, max_buffer: int = 5000):
        self.target = target_per_min
        self.langs = set(langs) if langs else None
        self.url = url
        self.rng = random.Random(seed)
        self.posts: deque = deque(maxlen=max_buffer)
        self.deletes: deque = deque(maxlen=max_buffer)
        self.lock = threading.Lock()
        self.tracked: dict[str, str] = {}  # uri -> did (posts we follow), for deletion propagation
        self.rate_per_min = 0.0  # EMA of eligible posts per minute
        self._win_n, self._win_t = 0, None
        self.stats = {"messages": 0, "eligible": 0, "kept": 0, "deletes": 0, "reconnects": 0, "errors": 0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected_at: Optional[float] = None

    def keep_prob(self) -> float:
        return 1.0 if self.rate_per_min <= 0 else min(1.0, self.target / self.rate_per_min)

    def handle(self, msg: dict, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.stats["messages"] += 1
        kind = msg.get("kind")
        did = msg.get("did", "")
        if kind == "account" and not (msg.get("account") or {}).get("active", True):
            with self.lock:
                for uri, d in list(self.tracked.items()):
                    if d == did:
                        self.deletes.append((uri, msg.get("time_us", 0) / 1e6, "account " +
                                             str((msg.get("account") or {}).get("status", "inactive"))))
            return
        if kind != "commit":
            return
        c = msg.get("commit") or {}
        if c.get("collection") != "app.bsky.feed.post":
            return
        uri = post_uri(did, c.get("rkey", ""))
        if c.get("operation") == "delete":
            with self.lock:
                if uri in self.tracked:
                    self.deletes.append((uri, msg.get("time_us", 0) / 1e6, "deleted by author"))
                    self.stats["deletes"] += 1
            return
        if c.get("operation") != "create":
            return
        rec = c.get("record") or {}
        if rec.get("reply") or not (rec.get("text") or "").strip():
            return  # top-level posts with text only
        if self.langs and not (set(rec.get("langs") or []) & self.langs):
            return
        self.stats["eligible"] += 1
        if self._win_t is None:
            self._win_t = now
        self._win_n += 1
        if now - self._win_t >= 30:
            r = self._win_n / ((now - self._win_t) / 60)
            self.rate_per_min = r if self.rate_per_min == 0 else 0.7 * self.rate_per_min + 0.3 * r
            self._win_n, self._win_t = 0, now
        if self.rate_per_min == 0:  # first 30 s: estimate the rate before keeping anything
            return
        p = self.keep_prob()
        if self.rng.random() >= p:
            return
        self.stats["kept"] += 1
        with self.lock:
            self.posts.append({"uri": uri, "did": did, "cid": c.get("cid"), "time": msg.get("time_us", 0) / 1e6,
                               "record": rec, "keep_prob": p})

    def drain(self) -> tuple[list[dict], list[tuple]]:
        with self.lock:
            p, d = list(self.posts), list(self.deletes)
            self.posts.clear()
            self.deletes.clear()
        return p, d

    # ------------------------------------------------------------------ socket
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="jetstream")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def _run(self) -> None:
        import websockets

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, ssl=ssl_context(), max_size=2 ** 21, open_timeout=30,
                                              ping_interval=20) as ws:
                    self.connected_at = time.time()
                    backoff = 1.0
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        except asyncio.TimeoutError:
                            break
                        try:
                            self.handle(json.loads(raw))
                        except (ValueError, TypeError, AttributeError):
                            self.stats["errors"] += 1
            except Exception:  # noqa: BLE001 - reconnect on any network failure
                self.stats["errors"] += 1
            if self._stop.is_set():
                break
            self.stats["reconnects"] += 1
            await asyncio.sleep(backoff)
            backoff = min(60.0, backoff * 2)


class BlueskySource(SourceAdapter):
    def __init__(self, declaration: Optional[SourceDeclaration] = None, sampler: Optional[JetstreamSampler] = None,
                 client: Optional[httpx.Client] = None, pseudonym_key: str = "bsky-key", cache_s: float = 120.0):
        self.declaration = declaration or bluesky_declaration()
        self.sampler = sampler or JetstreamSampler()
        self.client = client or httpx.Client(timeout=20, headers={"User-Agent": "agentrs-research-observer/0.1"})
        self.key = pseudonym_key
        self.cache_s = cache_s
        self._cache: dict[str, tuple[float, Optional[dict]]] = {}
        self.requests = 0
        self.stats = {"skipped_opt_out": 0, "skipped_no_profile": 0}

    def now(self) -> float:
        return time.time()

    def env_key(self, object_ref: Any) -> str:
        return f"bsky:{object_ref}"

    def _get(self, method: str, params: list[tuple[str, str]]) -> dict:
        self.requests += 1
        try:
            r = self.client.get(f"{APPVIEW}/{method}", params=params)
        except httpx.HTTPError as e:
            raise SourceError(f"bluesky {method}: {type(e).__name__}") from e
        if r.status_code == 429:
            raise SourceError("bluesky rate limited", 429, float(r.headers.get("retry-after", 60)))
        if r.status_code != 200:
            raise SourceError(f"bluesky {method}: HTTP {r.status_code}", r.status_code)
        return r.json()

    def profiles(self, dids: list[str]) -> dict[str, dict]:
        out = {}
        uniq = list(dict.fromkeys(dids))
        for i in range(0, len(uniq), 25):
            js = self._get("app.bsky.actor.getProfiles", [("actors", d) for d in uniq[i:i + 25]])
            for p in js.get("profiles", []):
                out[p["did"]] = p
        return out

    def prefetch(self, uris: list[str]) -> int:
        """Batch the current state of many posts (25 per request) into a short-lived cache."""
        now = time.time()
        need = [u for u in dict.fromkeys(uris) if not (u in self._cache and now - self._cache[u][0] < self.cache_s)]
        for i in range(0, len(need), 25):
            chunk = need[i:i + 25]
            js = self._get("app.bsky.feed.getPosts", [("uris", u) for u in chunk])
            got = {p["uri"]: p for p in js.get("posts", [])}
            for u in chunk:
                self._cache[u] = (now, got.get(u))
        for u in [u for u, (t, _) in self._cache.items() if now - t > 4 * self.cache_s]:
            self._cache.pop(u, None)
        return len(need)

    def poll(self, cursor):
        self.sampler.start()
        posts, deletes = self.sampler.drain()
        events: list[RawEvent] = []
        for uri, at, why in deletes:
            events.append(RawEvent(provider_event_id=f"del:{uri}", type="item_deleted", at=at, object_ref=uri,
                                   data={"reason": why}))
        if posts:
            try:
                profs = self.profiles([p["did"] for p in posts])
            except SourceError:
                profs = {}
            for p in posts:
                prof = profs.get(p["did"])
                if prof is None:
                    self.stats["skipped_no_profile"] += 1
                    continue
                if {lab.get("val") for lab in prof.get("labels", [])} & OPT_OUT_LABELS:
                    self.stats["skipped_opt_out"] += 1
                    continue
                events.append(RawEvent(provider_event_id=p["uri"], type="item_created", at=p["time"],
                                       object_ref=p["uri"], data={**p, "profile": prof}))
        return events, int(cursor or 0) + len(events)

    def normalize(self, raw: RawEvent):
        if raw.type == "item_deleted":
            self.sampler.tracked.pop(raw.object_ref, None)
            return [FollowUp(object_ref=raw.object_ref, type="item_deleted", env_event_time=raw.at, actor_ref=None,
                             by_original_actor=True, provider_event_id=raw.provider_event_id,
                             payload={"summary": raw.data.get("reason", "")}, categories=["public_posts"])]
        d = raw.data
        rec, prof = d["record"], d["profile"]
        emb = rec.get("embed") or {}
        kind = EMBED_KIND.get(emb.get("$type", ""), "text")
        domain = ""
        if kind == "link":
            domain = urlsplit((emb.get("external") or {}).get("uri", "")).hostname or ""
        tags = [f.get("tag") for fc in rec.get("facets") or [] for f in fc.get("features", []) if f.get("tag")]
        created = (prof.get("createdAt") or "")[:4]
        ctx = {"env": "bluesky", "space": kind, "author_followers": int(prof.get("followersCount", 0)),
               "author_posts": int(prof.get("postsCount", 0)), "author_following": int(prof.get("followsCount", 0)),
               "langs": ",".join(rec.get("langs") or []) or "unknown",
               "audience": audience_label(int(prof.get("followersCount", 0)), "followers")}
        if domain:
            ctx["domain"] = domain
        if created:
            ctx["account_since"] = created
        if tags:
            ctx["hashtags"] = len(tags)
        text = rec.get("text", "")
        if emb.get("images"):
            text += "\n[" + "; ".join((i.get("alt") or "image")[:200] for i in emb["images"][:4]) + "]"
        if kind == "link":
            ext = emb.get("external") or {}
            text += f"\n[link: {(ext.get('title') or '')[:200]}]"
        self.sampler.tracked[raw.object_ref] = d["did"]
        return [ObservedActorAction(
            source_id=self.declaration.source_id, provider_object_id=raw.object_ref, actor_type="unknown",
            actor_ref=pseudonym(d["did"], self.key), action_time=raw.at, action_time_known=True,
            observation_time=time.time(), time_base="real", context_before=ctx, context_complete=False,
            action=ObservedAction(operation="create_post", content=text),
            goal=None, goal_known=False,
            provenance={"source": "bluesky-jetstream", "keep_prob": round(d.get("keep_prob", 1.0), 5)},
            permission_metadata={"categories": ["public_posts", "public_profiles_followers"]},
            missing=["goal", "ranking_algorithm", "audience_exposure"], env_id="bluesky", group=f"bsky:{kind}")]

    def fetch_state(self, object_ref: str):
        hit = self._cache.get(object_ref)
        if hit is None or time.time() - hit[0] > self.cache_s:
            self.prefetch([object_ref])
            hit = self._cache[object_ref]
        p = hit[1]
        if p is None:  # gone: deleted by the author or taken down; the stored text is redacted
            with self.sampler.lock:
                if object_ref in self.sampler.tracked:
                    self.sampler.deletes.append((object_ref, time.time(), "post no longer available"))
            return payload(0, 0, 0, removed=True), "complete"
        out = payload(reactions=int(p.get("likeCount", 0)), responses=int(p.get("replyCount", 0)),
                      shares=int(p.get("repostCount", 0)) + int(p.get("quoteCount", 0)), removed=False)
        return out, "complete"
