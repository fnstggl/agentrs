"""Local simulator of the Bluesky PDS / AppView XRPC subset used by ``BlueskyConnector``.

Implements (``/xrpc/<nsid>``): ``com.atproto.server.createSession`` / ``refreshSession``,
``com.atproto.repo.createRecord`` / ``putRecord`` / ``getRecord``, ``app.bsky.actor.getProfile``,
``app.bsky.feed.getPostThread`` / ``getAuthorFeed``, ``app.bsky.notification.listNotifications``.

Auth: ``Authorization: Bearer <accessJwt>`` (HS256 JWTs with ``scope`` ``com.atproto.access`` /
``com.atproto.refresh``; refresh tokens rotate). Errors: ``{"error": ..., "message": ...}`` with 400
(``InvalidRequest``, ``ExpiredToken``, ``NotFound``, ``RecordNotFound``), 401
(``AuthenticationRequired`` / ``InvalidToken``) and 429 (``RateLimitExceeded``). Rate-limit headers
``ratelimit-limit``, ``ratelimit-remaining``, ``ratelimit-reset`` (epoch s), ``ratelimit-policy``
(``<limit>;w=<window>``). Repo writes cost points (create 3, update 2) against an hourly budget.

Other users' activity is driven from tests: ``simulate_reply``, ``simulate_like``,
``simulate_repost``, ``simulate_mention``, ``simulate_takedown``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Optional

import jwt
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    iso_z,
    read_params,
)

B32_SORT = "234567abcdefghijklmnopqrstuvwxyz"
TID_EPOCH_US = 1_767_225_600_000_000  # 2026-01-01T00:00:00Z — base of deterministic TIDs


class XrpcError(Exception):
    def __init__(self, status: int, error: str, message: str = "", headers: Optional[dict] = None):
        self.status, self.error, self.message, self.headers = status, error, message, headers or {}


def _tid(n: int) -> str:
    v = ((TID_EPOCH_US + n * 1000) << 10) | 0  # timestamp micros << 10 | clock id
    s = ""
    for _ in range(13):
        s = B32_SORT[v & 31] + s
        v >>= 5
    return s


def _cid(obj: Any) -> str:
    h = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).digest()
    return "bafyrei" + base64.b32encode(h).decode().lower().rstrip("=")[:52]


def _did(handle: str) -> str:
    return "did:plc:" + base64.b32encode(hashlib.sha256(handle.encode()).digest()).decode().lower()[:24]


class BlueskyState:
    def __init__(self, clock: Optional[SimClock] = None, service_did: str = "did:web:bsky.sim",
                 access_ttl: float = 2 * 3600, refresh_ttl: float = 90 * 86400):
        self.clock = clock or SimClock()
        self.service_did = service_did
        self.access_ttl, self.refresh_ttl = access_ttl, refresh_ttl
        self._secret = "simulated-pds-signing-secret-not-a-real-key-0001"
        self.users: dict[str, dict] = {}  # did -> user
        self.by_handle: dict[str, str] = {}
        self.records: dict[str, dict] = {}  # at-uri -> record entry
        self.children: dict[str, list[str]] = {}  # parent uri -> reply uris
        self.likes: dict[str, list[tuple[str, float]]] = {}
        self.reposts: dict[str, list[tuple[str, float]]] = {}
        self.notifications: dict[str, list[dict]] = {}
        self.active_refresh: set[str] = set()
        self.access_generation = 0  # access tokens of an older generation are expired
        # bucket -> (limit, window seconds); real values per Bluesky's published PDS limits
        self.limits = {"global": (3000, 300), "createSession": (30, 300), "write_points": (5000, 3600)}
        self.limiter = RateLimiter(self.clock)
        self._n = 0
        self.requests: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ setup / simulation hooks
    def add_user(self, handle: str, app_password: Optional[str] = None, display_name: str = "",
                 description: str = "") -> str:
        did = _did(handle)
        self.users[did] = {"did": did, "handle": handle, "app_passwords": [app_password] if app_password else [],
                           "displayName": display_name or handle.split(".")[0], "description": description,
                           "createdAt": iso_z(self.clock.now())}
        self.by_handle[handle] = did
        self.notifications.setdefault(did, [])
        return did

    def resolve(self, actor: str) -> Optional[str]:
        if actor.startswith("did:"):
            return actor if actor in self.users else None
        return self.by_handle.get(actor)

    def _next(self) -> int:
        self._n += 1
        return self._n

    def _write(self, did: str, collection: str, value: dict, rkey: Optional[str] = None) -> dict:
        rkey = rkey or _tid(self._next())
        uri = f"at://{did}/{collection}/{rkey}"
        cid = _cid({"uri": uri, "value": value})
        prev = self.records.get(uri)
        self.records[uri] = {"uri": uri, "cid": cid, "did": did, "collection": collection, "rkey": rkey,
                             "value": value, "indexedAt": iso_z(self.clock.now()), "t": self.clock.now(),
                             "removed": False}
        if collection == "app.bsky.feed.post" and prev is None:
            self._index_post(self.records[uri])
        return self.records[uri]

    def _notify(self, did: str, reason: str, rec: dict, subject: Optional[str] = None) -> None:
        if did not in self.users or did == rec["did"]:
            return
        self.notifications[did].append({"uri": rec["uri"], "cid": rec["cid"], "author": self.profile_basic(rec["did"]),
                                        "reason": reason, "reasonSubject": subject, "record": rec["value"],
                                        "isRead": False, "indexedAt": rec["indexedAt"], "labels": []})

    def _index_post(self, rec: dict) -> None:
        v = rec["value"]
        parent = ((v.get("reply") or {}).get("parent") or {}).get("uri")
        notified = set()
        if parent:
            self.children.setdefault(parent, []).append(rec["uri"])
            p = self.records.get(parent)
            if p:
                self._notify(p["did"], "reply", rec, parent)
                notified.add(p["did"])
        mentioned = {f["did"] for fc in v.get("facets") or [] for f in fc.get("features", [])
                     if f.get("$type") == "app.bsky.richtext.facet#mention" and f.get("did")}
        for h in re.findall(r"@([a-zA-Z0-9.-]+\.[a-zA-Z]+)", v.get("text", "")):
            if h in self.by_handle:
                mentioned.add(self.by_handle[h])
        for did in mentioned - notified:
            self._notify(did, "mention", rec)

    def post(self, by: str, text: str, reply_to: Optional[str] = None) -> str:
        did = self.resolve(by) or self.add_user(by)
        value: dict[str, Any] = {"$type": "app.bsky.feed.post", "text": text, "createdAt": iso_z(self.clock.now())}
        if reply_to:
            parent = self.records[reply_to]
            root = (parent["value"].get("reply") or {}).get("root") or {"uri": parent["uri"], "cid": parent["cid"]}
            value["reply"] = {"root": root, "parent": {"uri": parent["uri"], "cid": parent["cid"]}}
        return self._write(did, "app.bsky.feed.post", value)["uri"]

    def simulate_reply(self, post_uri: str, text: str, by: str = "carol.bsky.sim") -> str:
        return self.post(by, text, reply_to=post_uri)

    def simulate_mention(self, text: str, by: str = "dave.bsky.sim") -> str:
        return self.post(by, text)

    def simulate_like(self, post_uri: str, by: str = "erin.bsky.sim") -> None:
        did = self.resolve(by) or self.add_user(by)
        self.likes.setdefault(post_uri, []).append((did, self.clock.now()))
        p = self.records[post_uri]
        rec = self._write(did, "app.bsky.feed.like", {"$type": "app.bsky.feed.like", "createdAt": iso_z(self.clock.now()),
                                                       "subject": {"uri": post_uri, "cid": p["cid"]}})
        self._notify(p["did"], "like", rec, post_uri)

    def simulate_repost(self, post_uri: str, by: str = "frank.bsky.sim") -> None:
        did = self.resolve(by) or self.add_user(by)
        self.reposts.setdefault(post_uri, []).append((did, self.clock.now()))
        p = self.records[post_uri]
        rec = self._write(did, "app.bsky.feed.repost", {"$type": "app.bsky.feed.repost", "createdAt": iso_z(self.clock.now()),
                                                         "subject": {"uri": post_uri, "cid": p["cid"]}})
        self._notify(p["did"], "repost", rec, post_uri)

    def simulate_takedown(self, post_uri: str) -> None:
        """Moderation takedown: the AppView stops serving the post."""
        self.records[post_uri]["removed"] = True

    def expire_access_tokens(self) -> None:
        self.access_generation += 1

    # ------------------------------------------------------------------ views
    def self_labels(self, did: str) -> list[dict]:
        prof = self.records.get(f"at://{did}/app.bsky.actor.profile/self")
        if not prof:
            return []
        vals = ((prof["value"].get("labels") or {}).get("values")) or []
        return [{"src": did, "uri": prof["uri"], "cid": prof["cid"], "val": v.get("val"), "cts": prof["indexedAt"]}
                for v in vals]

    def profile_basic(self, did: str) -> dict:
        u = self.users[did]
        prof = self.records.get(f"at://{did}/app.bsky.actor.profile/self")
        name = (prof or {}).get("value", {}).get("displayName") or u["displayName"]
        return {"did": did, "handle": u["handle"], "displayName": name, "labels": self.self_labels(did),
                "createdAt": u["createdAt"]}

    def visible_children(self, uri: str) -> list[dict]:
        return [self.records[c] for c in self.children.get(uri, []) if not self.records[c]["removed"]]

    def post_view(self, rec: dict) -> dict:
        return {"uri": rec["uri"], "cid": rec["cid"], "author": self.profile_basic(rec["did"]), "record": rec["value"],
                "replyCount": len(self.visible_children(rec["uri"])), "repostCount": len(self.reposts.get(rec["uri"], [])),
                "likeCount": len(self.likes.get(rec["uri"], [])), "quoteCount": 0, "indexedAt": rec["indexedAt"],
                "labels": []}

    def thread(self, rec: dict, depth: int, parent_height: int) -> dict:
        node: dict[str, Any] = {"$type": "app.bsky.feed.defs#threadViewPost", "post": self.post_view(rec)}
        if depth > 0:
            node["replies"] = [self.thread(c, depth - 1, 0) for c in self.visible_children(rec["uri"])]
        parent_uri = ((rec["value"].get("reply") or {}).get("parent") or {}).get("uri")
        if parent_uri and parent_height > 0:
            p = self.records.get(parent_uri)
            if p is None or p["removed"]:
                node["parent"] = {"$type": "app.bsky.feed.defs#notFoundPost", "uri": parent_uri, "notFound": True}
            else:
                node["parent"] = self.thread(p, 0, parent_height - 1)
                node["parent"].pop("replies", None)
        return node

    # ------------------------------------------------------------------ tokens
    def _issue(self, did: str) -> dict:
        now = self.clock.now()
        n = self._next()
        acc = jwt.encode({"scope": "com.atproto.access", "sub": did, "aud": self.service_did, "iat": int(now),
                          "exp": int(now + self.access_ttl), "jti": f"a{n}", "gen": self.access_generation}, self._secret, "HS256")
        jti = f"r{n}"
        ref = jwt.encode({"scope": "com.atproto.refresh", "sub": did, "aud": self.service_did, "iat": int(now),
                          "exp": int(now + self.refresh_ttl), "jti": jti}, self._secret, "HS256")
        self.active_refresh.add(jti)
        u = self.users[did]
        return {"accessJwt": acc, "refreshJwt": ref, "handle": u["handle"], "did": did, "active": True,
                "didDoc": {"id": did, "alsoKnownAs": [f"at://{u['handle']}"]}}

    def check_token(self, auth: Optional[str], scope: str) -> dict:
        if not auth or not auth.startswith("Bearer "):
            raise XrpcError(401, "AuthenticationRequired", "Authentication Required")
        try:
            claims = jwt.decode(auth[7:], self._secret, algorithms=["HS256"], audience=self.service_did,
                                options={"verify_exp": False, "verify_iat": False, "verify_nbf": False})
        except jwt.PyJWTError:
            raise XrpcError(401, "InvalidToken", "Token could not be verified") from None
        if claims.get("scope") != scope:
            raise XrpcError(400, "InvalidToken", "Bad token scope")
        now = self.clock.now()
        if claims["exp"] <= now or (scope == "com.atproto.access" and claims.get("gen", 0) < self.access_generation):
            raise XrpcError(400, "ExpiredToken", "Token has expired")
        if scope == "com.atproto.refresh" and claims["jti"] not in self.active_refresh:
            raise XrpcError(400, "ExpiredToken", "Token has been revoked")
        if claims["sub"] not in self.users:
            raise XrpcError(401, "InvalidToken", "Unknown account")
        return claims


def create_app(state: Optional[BlueskyState] = None) -> FastAPI:
    st = state or BlueskyState()
    app = FastAPI(title="Simulated Bluesky XRPC (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="bluesky")

    @app.exception_handler(XrpcError)
    async def _xrpc_err(request: Request, e: XrpcError):
        return JSONResponse({"error": e.error, "message": e.message}, status_code=e.status, headers=e.headers)

    def _rl_headers(rs, key) -> dict:
        lim, win = st.limits[key]
        return {"ratelimit-limit": str(rs.limit), "ratelimit-remaining": str(rs.remaining),
                "ratelimit-reset": str(int(rs.reset)), "ratelimit-policy": f"{lim};w={int(win)}"}

    def _limit(key: str, principal: str, response: Response, cost: int = 1) -> None:
        lim, win = st.limits[key]
        rs = st.limiter.hit(f"{key}:{principal}", lim, win, cost)
        h = _rl_headers(rs, key)
        if not rs.allowed:
            raise XrpcError(429, "RateLimitExceeded", "Rate Limit Exceeded", h)
        response.headers.update(h)

    async def global_limit(request: Request, response: Response) -> None:
        st.requests.append((request.method, request.url.path))
        principal = request.headers.get("authorization", "")[-16:] or (request.client.host if request.client else "-")
        _limit("global", principal, response)

    def authed(request: Request) -> dict:
        return st.check_token(request.headers.get("authorization"), "com.atproto.access")

    G = [Depends(global_limit)]

    @app.post("/xrpc/com.atproto.server.createSession", dependencies=G)
    async def create_session(request: Request, response: Response):
        body = await read_params(request)
        ident, pw = body.get("identifier"), body.get("password")
        if not ident or not pw:
            raise XrpcError(400, "InvalidRequest", "Input must have the properties \"identifier\" and \"password\"")
        _limit("createSession", str(ident), response)
        did = st.resolve(str(ident))
        if not did or pw not in st.users[did]["app_passwords"]:
            raise XrpcError(401, "AuthenticationRequired", "Invalid identifier or password")
        return st._issue(did)

    @app.post("/xrpc/com.atproto.server.refreshSession", dependencies=G)
    async def refresh_session(request: Request):
        claims = st.check_token(request.headers.get("authorization"), "com.atproto.refresh")
        st.active_refresh.discard(claims["jti"])  # rotation: the used refresh token is revoked
        return st._issue(claims["sub"])

    @app.post("/xrpc/com.atproto.server.deleteSession", dependencies=G)
    async def delete_session(request: Request):
        claims = st.check_token(request.headers.get("authorization"), "com.atproto.refresh")
        st.active_refresh.discard(claims["jti"])
        st.access_generation += 1  # simplification: ends every outstanding access token of the PDS
        return Response(status_code=200)

    @app.get("/xrpc/com.atproto.server.getSession", dependencies=G)
    async def get_session(claims: dict = Depends(authed)):
        u = st.users[claims["sub"]]
        return {"handle": u["handle"], "did": u["did"], "active": True}

    @app.get("/xrpc/app.bsky.actor.getProfile", dependencies=G)
    async def get_profile(actor: str = "", claims: dict = Depends(authed)):
        did = st.resolve(actor) if actor else None
        if not did:
            raise XrpcError(400, "InvalidRequest", "Profile not found")
        u = st.users[did]
        prof = st.records.get(f"at://{did}/app.bsky.actor.profile/self")
        pv = (prof or {}).get("value", {})
        posts = [r for r in st.records.values() if r["did"] == did and r["collection"] == "app.bsky.feed.post"]
        return {"did": did, "handle": u["handle"], "displayName": pv.get("displayName") or u["displayName"],
                "description": pv.get("description", u["description"]), "followersCount": 0, "followsCount": 0,
                "postsCount": len(posts), "labels": st.self_labels(did), "indexedAt": u["createdAt"],
                "createdAt": u["createdAt"]}

    def _repo_did(body: dict, claims: dict) -> str:
        did = st.resolve(str(body.get("repo", "")))
        if did != claims["sub"]:
            raise XrpcError(400, "InvalidRequest", "Input/repo must be the authenticated account")
        return did

    def _validate(collection: str, rec: Any) -> None:
        if not isinstance(rec, dict):
            raise XrpcError(400, "InvalidRequest", "Input/record must be an object")
        if rec.get("$type") not in (None, collection):
            raise XrpcError(400, "InvalidRequest", f"Invalid $type: expected {collection}")
        if collection == "app.bsky.feed.post":
            if not isinstance(rec.get("text"), str):
                raise XrpcError(400, "InvalidRequest", 'Invalid app.bsky.feed.post record: Record must have the property "text"')
            if len(rec["text"]) > 300:
                raise XrpcError(400, "InvalidRequest",
                                "Invalid app.bsky.feed.post record: Record/text must not be longer than 300 graphemes")
            if not rec.get("createdAt"):
                raise XrpcError(400, "InvalidRequest",
                                'Invalid app.bsky.feed.post record: Record must have the property "createdAt"')
        elif collection not in ("app.bsky.actor.profile", "app.bsky.feed.like", "app.bsky.feed.repost"):
            raise XrpcError(400, "InvalidRequest", f"Unsupported collection in simulator: {collection}")

    @app.post("/xrpc/com.atproto.repo.createRecord", dependencies=G)
    async def create_record(request: Request, response: Response, claims: dict = Depends(authed)):
        body = await read_params(request)
        did = _repo_did(body, claims)
        coll = body.get("collection", "")
        rec = body.get("record")
        _validate(coll, rec)
        _limit("write_points", did, response, cost=3)
        rec = {"$type": coll, **rec}
        if body.get("rkey") and f"at://{did}/{coll}/{body['rkey']}" in st.records:
            raise XrpcError(400, "InvalidRequest", "Record already exists")
        out = st._write(did, coll, rec, body.get("rkey"))
        return {"uri": out["uri"], "cid": out["cid"], "commit": {"cid": _cid(out["uri"] + "c"), "rev": _tid(st._n)},
                "validationStatus": "valid"}

    @app.post("/xrpc/com.atproto.repo.putRecord", dependencies=G)
    async def put_record(request: Request, response: Response, claims: dict = Depends(authed)):
        body = await read_params(request)
        did = _repo_did(body, claims)
        coll, rkey = body.get("collection", ""), body.get("rkey")
        if not rkey:
            raise XrpcError(400, "InvalidRequest", 'Input must have the property "rkey"')
        if coll == "app.bsky.actor.profile" and rkey != "self":
            raise XrpcError(400, "InvalidRequest", "Profile record key must be 'self'")
        rec = body.get("record")
        _validate(coll, rec)
        uri = f"at://{did}/{coll}/{rkey}"
        if body.get("swapRecord") and st.records.get(uri, {}).get("cid") != body["swapRecord"]:
            raise XrpcError(400, "InvalidSwap", "Record was at a different cid")
        _limit("write_points", did, response, cost=2 if uri in st.records else 3)
        out = st._write(did, coll, {"$type": coll, **rec}, rkey)
        return {"uri": out["uri"], "cid": out["cid"], "validationStatus": "valid"}

    @app.get("/xrpc/com.atproto.repo.getRecord", dependencies=G)
    async def get_record(repo: str = "", collection: str = "", rkey: str = ""):
        did = st.resolve(repo)
        uri = f"at://{did}/{collection}/{rkey}"
        rec = st.records.get(uri)
        if not did or rec is None:
            raise XrpcError(400, "RecordNotFound", f"Could not locate record: {uri}")
        return {"uri": uri, "cid": rec["cid"], "value": rec["value"]}

    @app.get("/xrpc/app.bsky.feed.getPostThread", dependencies=G)
    async def get_post_thread(uri: str = "", depth: int = 6, parentHeight: int = 80, claims: dict = Depends(authed)):
        rec = st.records.get(uri)
        if rec is None or rec["removed"] or rec["collection"] != "app.bsky.feed.post":
            raise XrpcError(400, "NotFound", f"Post not found: {uri}")
        return {"thread": st.thread(rec, max(0, min(depth, 1000)), max(0, min(parentHeight, 1000)))}

    @app.get("/xrpc/app.bsky.feed.getAuthorFeed", dependencies=G)
    async def get_author_feed(actor: str = "", limit: int = 50, cursor: str = "", filter: str = "posts_with_replies",
                              claims: dict = Depends(authed)):
        did = st.resolve(actor)
        if not did:
            raise XrpcError(400, "InvalidRequest", "Profile not found")
        posts = sorted((r for r in st.records.values() if r["did"] == did and r["collection"] == "app.bsky.feed.post"
                        and not r["removed"]), key=lambda r: r["rkey"], reverse=True)
        if filter == "posts_no_replies":
            posts = [p for p in posts if not p["value"].get("reply")]
        if cursor:
            posts = [p for p in posts if p["rkey"] < cursor]
        page = posts[:max(1, min(limit, 100))]
        feed = []
        for p in page:
            item: dict[str, Any] = {"post": st.post_view(p)}
            rp = p["value"].get("reply")
            if rp:
                par = st.records.get(rp["parent"]["uri"])
                roo = st.records.get(rp["root"]["uri"])
                item["reply"] = {k: (st.post_view(v) if v and not v["removed"] else
                                     {"$type": "app.bsky.feed.defs#notFoundPost", "uri": rp[k]["uri"], "notFound": True})
                                 for k, v in (("root", roo), ("parent", par))}
            feed.append(item)
        out: dict[str, Any] = {"feed": feed}
        if len(posts) > len(page):
            out["cursor"] = page[-1]["rkey"]
        return out

    @app.get("/xrpc/app.bsky.notification.listNotifications", dependencies=G)
    async def list_notifications(limit: int = 50, cursor: str = "", claims: dict = Depends(authed)):
        items = list(reversed(st.notifications.get(claims["sub"], [])))
        start = int(cursor) if cursor.isdigit() else 0
        page = items[start:start + max(1, min(limit, 100))]
        out: dict[str, Any] = {"notifications": page, "priority": False}
        if start + len(page) < len(items):
            out["cursor"] = str(start + len(page))
        return out

    @app.api_route("/xrpc/{nsid}", methods=["GET", "POST"])
    async def unknown(nsid: str):
        raise XrpcError(501, "MethodNotImplemented", f"Method Not Implemented: {nsid}")

    return app
