"""Local simulator of the X API v2 subset used by ``XConnector``.

Implements (under ``/2``): ``GET /users/me``, ``POST /tweets`` (``text``, optional
``reply.in_reply_to_tweet_id``), ``GET /tweets/:id`` (``tweet.fields`` e.g. ``public_metrics,
created_at,author_id,conversation_id,in_reply_to_user_id,entities,referenced_tweets``),
``DELETE /tweets/:id``, ``POST /users/:id/likes``, ``GET /tweets/search/recent`` (only
``conversation_id:<id>`` queries; 403 unless ``search_enabled`` — it needs a paid access tier on
the real API) and ``POST /oauth2/token`` (``refresh_token`` grant for public clients).

Auth: OAuth 2.0 user-context bearer tokens with scopes (``tweet.read``, ``tweet.write``,
``users.read``, ``like.write``, ``offline.access``); an app-only bearer gets the real API's 403
``Unsupported Authentication`` problem. Errors follow the v2 problem format (``title``, ``detail``,
``type``, ``status``). Identical text from the same user -> 403 "You are not allowed to create a
Tweet with duplicate content.". A missing / deleted tweet is HTTP 200 with an ``errors`` array (as the
real API does). ``restrict_unsummoned_replies`` (default on) reproduces the API's reply restriction:
replying is only allowed when the author of the post replied to mentioned the replier or the
replier authored the conversation. Rate-limit headers ``x-rate-limit-limit/remaining/reset``.

Other users: ``simulate_reply``, ``simulate_mention``, ``simulate_like``, ``simulate_retweet``,
``simulate_quote``, ``simulate_removal``.
"""

from __future__ import annotations

import re
import secrets as _secrets
from typing import Any, Optional

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    iso_z,
    read_params,
)

PROBLEM = "https://api.twitter.com/2/problems/"


class XError(Exception):
    def __init__(self, status: int, body: dict, headers: Optional[dict] = None):
        self.status, self.body, self.headers = status, body, headers or {}


def _problem(status: int, title: str, detail: str, kind: str = "about:blank") -> XError:
    return XError(status, {"title": title, "detail": detail, "type": kind, "status": status})


class XState:
    def __init__(self, clock: Optional[SimClock] = None, restrict_unsummoned_replies: bool = True,
                 search_enabled: bool = True, access_ttl: float = 7200):
        self.clock = clock or SimClock()
        self.restrict_unsummoned_replies = restrict_unsummoned_replies
        self.search_enabled = search_enabled
        self.access_ttl = access_ttl
        self.users: dict[str, dict] = {}
        self.by_username: dict[str, str] = {}
        self.tweets: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.refresh: dict[str, dict] = {}
        self.app_only_tokens: set[str] = set()
        self.clients: set[str] = set()
        # endpoint -> (limit, window seconds) per user. Real limits depend on the access tier.
        self.limits = {"POST /2/tweets": (100, 900), "DELETE /2/tweets": (50, 900), "GET /2/tweets": (900, 900),
                       "GET /2/users/me": (75, 900), "POST /2/likes": (50, 900), "GET /2/search": (60, 900)}
        self.limiter = RateLimiter(self.clock)
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return str(1_850_000_000_000_000_000 + self._n)

    def add_user(self, username: str, name: str = "") -> str:
        if username in self.by_username:
            return self.by_username[username]
        uid = str(1_700_000_000 + len(self.users) + 1)
        self.users[uid] = {"id": uid, "name": name or username, "username": username,
                           "created_at": iso_z(self.clock.now())}
        self.by_username[username] = uid
        return uid

    def issue_token(self, username: str, scopes: str = "tweet.read tweet.write users.read like.write offline.access",
                    client_id: str = "sim-client") -> dict:
        uid = self.add_user(username)
        self.clients.add(client_id)
        return self._token(uid, set(scopes.split()), client_id)

    def issue_app_only_token(self) -> str:
        tok = "AAAAAAAAAAAAAAAAAAAAA" + _secrets.token_urlsafe(24)
        self.app_only_tokens.add(tok)
        return tok

    def _token(self, uid: str, scopes: set[str], client_id: str) -> dict:
        tok = _secrets.token_urlsafe(32)
        self.tokens[tok] = {"user_id": uid, "scopes": scopes, "expires_at": self.clock.now() + self.access_ttl,
                            "client_id": client_id}
        out = {"token_type": "bearer", "expires_in": int(self.access_ttl), "access_token": tok,
               "scope": " ".join(sorted(scopes))}
        if "offline.access" in scopes:
            rt = _secrets.token_urlsafe(32)
            self.refresh[rt] = {"user_id": uid, "scopes": scopes, "client_id": client_id}
            out["refresh_token"] = rt
        return out

    def expire_tokens(self) -> None:
        for t in self.tokens.values():
            t["expires_at"] = self.clock.now() - 1

    # ------------------------------------------------------------------ tweets
    def create(self, uid: str, text: str, reply_to: Optional[str] = None, quote_of: Optional[str] = None) -> dict:
        tid = self._id()
        parent = self.tweets.get(reply_to) if reply_to else None
        mentions = []
        for m in re.finditer(r"@(\w{1,15})", text):
            if m.group(1) in self.by_username:
                mentions.append({"start": m.start(), "end": m.end(), "username": m.group(1),
                                 "id": self.by_username[m.group(1)]})
        refs = []
        if parent:
            refs.append({"type": "replied_to", "id": reply_to})
        if quote_of:
            refs.append({"type": "quoted", "id": quote_of})
        t = {"id": tid, "text": text, "author_id": uid, "created_at": iso_z(self.clock.now()), "t": self.clock.now(),
             "conversation_id": parent["conversation_id"] if parent else tid,
             "in_reply_to_user_id": parent["author_id"] if parent else None, "referenced_tweets": refs,
             "mentions": mentions, "likes": set(), "retweets": set(), "quotes": set(), "removed": False}
        self.tweets[tid] = t
        if quote_of:
            self.tweets[quote_of]["quotes"].add(tid)
        return t

    def replies(self, tid: str) -> list[dict]:
        return [t for t in self.tweets.values() if not t["removed"]
                and any(r["type"] == "replied_to" and r["id"] == tid for r in t["referenced_tweets"])]

    def view(self, t: dict, fields: set[str]) -> dict:
        out: dict[str, Any] = {"id": t["id"], "text": t["text"], "edit_history_tweet_ids": [t["id"]]}
        if "public_metrics" in fields:
            out["public_metrics"] = {"retweet_count": len(t["retweets"]), "reply_count": len(self.replies(t["id"])),
                                     "like_count": len(t["likes"]), "quote_count": len(t["quotes"]),
                                     "bookmark_count": 0, "impression_count": 10 * (1 + len(t["likes"]))}
        for f in ("created_at", "author_id", "conversation_id"):
            if f in fields:
                out[f] = t[f]
        if "in_reply_to_user_id" in fields and t["in_reply_to_user_id"]:
            out["in_reply_to_user_id"] = t["in_reply_to_user_id"]
        if "referenced_tweets" in fields and t["referenced_tweets"]:
            out["referenced_tweets"] = t["referenced_tweets"]
        if "entities" in fields and t["mentions"]:
            out["entities"] = {"mentions": t["mentions"]}
        return out

    def simulate_reply(self, tweet_id: str, text: str, by: str = "carol") -> str:
        return self.create(self.add_user(by), text, reply_to=tweet_id)["id"]

    def simulate_mention(self, text: str, by: str = "dave") -> str:
        return self.create(self.add_user(by), text)["id"]

    def simulate_like(self, tweet_id: str, by: str = "erin") -> None:
        self.tweets[tweet_id]["likes"].add(self.add_user(by))

    def simulate_retweet(self, tweet_id: str, by: str = "frank") -> None:
        self.tweets[tweet_id]["retweets"].add(self.add_user(by))

    def simulate_quote(self, tweet_id: str, text: str, by: str = "gina") -> str:
        return self.create(self.add_user(by), text, quote_of=tweet_id)["id"]

    def simulate_removal(self, tweet_id: str) -> None:
        self.tweets[tweet_id]["removed"] = True


def create_app(state: Optional[XState] = None) -> FastAPI:
    st = state or XState()
    app = FastAPI(title="Simulated X API v2 (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="x")

    @app.exception_handler(XError)
    async def _err(request: Request, e: XError):
        return JSONResponse(e.body, status_code=e.status, headers=e.headers)

    def _rl(bucket: str, uid: str, response: Response) -> None:
        lim, win = st.limits[bucket]
        rs = st.limiter.hit(f"{bucket}:{uid}", lim, win)
        h = {"x-rate-limit-limit": str(rs.limit), "x-rate-limit-remaining": str(rs.remaining),
             "x-rate-limit-reset": str(int(rs.reset))}
        if not rs.allowed:
            raise XError(429, {"title": "Too Many Requests", "detail": "Too Many Requests", "type": "about:blank",
                               "status": 429}, h)
        response.headers.update(h)

    def user_ctx(*scopes: str):
        async def dep(request: Request) -> dict:
            auth = request.headers.get("authorization", "")
            tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if tok in st.app_only_tokens:
                raise _problem(403, "Unsupported Authentication",
                               "Authenticating with OAuth 2.0 Application-Only is forbidden for this endpoint.  "
                               "Supported authentication types are [OAuth 1.0a User Context, OAuth 2.0 User Context].",
                               PROBLEM + "unsupported-authentication")
            info = st.tokens.get(tok)
            if info is None or info["expires_at"] <= st.clock.now():
                raise XError(401, {"title": "Unauthorized", "type": "about:blank", "status": 401, "detail": "Unauthorized"})
            missing = [s for s in scopes if s not in info["scopes"]]
            if missing:
                raise _problem(403, "Forbidden", f"Missing required OAuth 2.0 scopes: {', '.join(missing)}")
            return info
        return dep

    @app.post("/2/oauth2/token")
    async def token(request: Request):
        p = await read_params(request)
        if p.get("grant_type") != "refresh_token":
            raise XError(400, {"error": "invalid_request", "error_description": "Value passed for the grant type was invalid."})
        r = st.refresh.pop(str(p.get("refresh_token", "")), None)
        if r is None or r["client_id"] != p.get("client_id"):
            raise XError(400, {"error": "invalid_request", "error_description": "Value passed for the token was invalid."})
        return st._token(r["user_id"], r["scopes"], r["client_id"])

    @app.get("/2/users/me")
    async def me(response: Response, info: dict = Depends(user_ctx("users.read", "tweet.read"))):
        _rl("GET /2/users/me", info["user_id"], response)
        u = st.users[info["user_id"]]
        return {"data": {"id": u["id"], "name": u["name"], "username": u["username"]}}

    @app.post("/2/tweets", status_code=201)
    async def post_tweet(request: Request, response: Response,
                         info: dict = Depends(user_ctx("tweet.read", "tweet.write", "users.read"))):
        _rl("POST /2/tweets", info["user_id"], response)
        try:
            p = await request.json()
        except ValueError:
            raise _problem(400, "Invalid Request", "One or more parameters to your request was invalid.",
                           PROBLEM + "invalid-request") from None
        text = p.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _problem(400, "Invalid Request", "One or more parameters to your request was invalid.",
                           PROBLEM + "invalid-request")
        if len(text) > 280:
            raise XError(400, {"errors": [{"parameters": {"text": [text[:20] + "..."]},
                                           "message": "text: Tweet text is too long"}],
                               "title": "Invalid Request", "detail": "One or more parameters to your request was invalid.",
                               "type": PROBLEM + "invalid-request"})
        uid = info["user_id"]
        if any(t["author_id"] == uid and t["text"] == text and not t["removed"] for t in st.tweets.values()):
            raise _problem(403, "Forbidden", "You are not allowed to create a Tweet with duplicate content.")
        reply_to = (p.get("reply") or {}).get("in_reply_to_tweet_id")
        if reply_to:
            parent = st.tweets.get(str(reply_to))
            if parent is None or parent["removed"]:
                raise _problem(403, "Forbidden", "You attempted to reply to a Tweet that is deleted or not visible to you.")
            conv_root = st.tweets.get(parent["conversation_id"])
            summoned = (parent["author_id"] == uid or any(m["id"] == uid for m in parent["mentions"])
                        or (conv_root is not None and conv_root["author_id"] == uid))
            if st.restrict_unsummoned_replies and not summoned:
                raise _problem(403, "Forbidden", "Reply to this conversation is not allowed because you have not been "
                                                 "mentioned or otherwise engaged by the author of the post you are replying to.")
        t = st.create(uid, text, reply_to=str(reply_to) if reply_to else None)
        return {"data": {"id": t["id"], "text": t["text"], "edit_history_tweet_ids": [t["id"]]}}

    @app.get("/2/tweets/search/recent")
    async def search(query: str, request: Request, response: Response, max_results: int = 10,
                     info: dict = Depends(user_ctx("tweet.read"))):
        if not st.search_enabled:
            raise _problem(403, "Client Forbidden", "When authenticating requests to the Twitter API v2 endpoints, you "
                                                    "must use keys and tokens from a Twitter developer App that is "
                                                    "attached to a Project.", PROBLEM + "client-forbidden")
        _rl("GET /2/search", info["user_id"], response)
        m = re.fullmatch(r"conversation_id:(\d+)", query.strip())
        if not m:
            raise _problem(400, "Invalid Request", "simulator only supports conversation_id:<id> queries",
                           PROBLEM + "invalid-request")
        fields = set(filter(None, request.query_params.get("tweet.fields", "").split(",")))
        items = sorted((t for t in st.tweets.values() if t["conversation_id"] == m.group(1) and t["id"] != m.group(1)
                        and not t["removed"]), key=lambda t: int(t["id"]), reverse=True)[:max(10, min(max_results, 100))]
        data = [st.view(t, fields | {"author_id", "created_at", "conversation_id", "in_reply_to_user_id",
                                     "referenced_tweets"}) for t in items]
        return {"data": data, "meta": {"result_count": len(data)}} if data else {"meta": {"result_count": 0}}

    @app.get("/2/tweets/{tid}")
    async def get_tweet(tid: str, request: Request, response: Response, info: dict = Depends(user_ctx("tweet.read"))):
        _rl("GET /2/tweets", info["user_id"], response)
        fields = set(filter(None, request.query_params.get("tweet.fields", "").split(",")))
        t = st.tweets.get(tid)
        if t is None or t["removed"]:
            return {"errors": [{"value": tid, "detail": f"Could not find tweet with id: [{tid}].",
                                "title": "Not Found Error", "resource_type": "tweet", "parameter": "id",
                                "resource_id": tid, "type": PROBLEM + "resource-not-found"}]}
        return {"data": st.view(t, fields)}

    @app.delete("/2/tweets/{tid}")
    async def delete_tweet(tid: str, response: Response, info: dict = Depends(user_ctx("tweet.read", "tweet.write"))):
        _rl("DELETE /2/tweets", info["user_id"], response)
        t = st.tweets.get(tid)
        if t is None or t["removed"]:
            return {"data": {"deleted": False}}
        if t["author_id"] != info["user_id"]:
            raise _problem(403, "Forbidden", "You are not authorized to delete this Tweet.")
        t["removed"] = True
        return {"data": {"deleted": True}}

    @app.post("/2/users/{uid}/likes")
    async def like(uid: str, request: Request, response: Response,
                   info: dict = Depends(user_ctx("tweet.read", "users.read", "like.write"))):
        if uid != info["user_id"]:
            raise _problem(403, "Forbidden", "You are not permitted to perform this action.")
        _rl("POST /2/likes", uid, response)
        p = await request.json()
        t = st.tweets.get(str(p.get("tweet_id", "")))
        if t is None or t["removed"]:
            raise _problem(400, "Invalid Request", "The tweet_id could not be found.", PROBLEM + "invalid-request")
        t["likes"].add(uid)
        return {"data": {"liked": True}}

    return app
