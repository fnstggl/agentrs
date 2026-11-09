"""Local simulator of the Instagram Graph API (professional accounts) subset used by ``InstagramConnector``.

Implements (under ``/v<major>.<minor>``): ``GET /{ig-user-id}?fields=account_type,username,...``,
``POST /{ig-user-id}/media`` (``image_url`` + ``caption`` -> container id), ``GET /{container-id}?
fields=status_code``, ``POST /{ig-user-id}/media_publish`` (``creation_id`` -> media id),
``GET /{media-id}?fields=like_count,comments_count,timestamp,...``, ``GET /{media-id}/comments``,
``GET /{ig-user-id}/content_publishing_limit``.

Auth: ``access_token`` query/form parameter or ``Authorization: Bearer``/``OAuth`` header; tokens
carry permissions (``instagram_basic``, ``instagram_content_publish``,
``instagram_manage_comments``). Errors use the Graph format ``{"error": {"message", "type",
"code", "error_subcode", "fbtrace_id"}}``: 190 invalid token, 10 missing permission or
non-professional account, 100/33 unknown object, 9/2207042 publishing quota, 80002 too many calls
(with ``X-Business-Use-Case-Usage``). ``account_type`` ``PERSONAL`` accounts cannot publish.

Other users: ``simulate_comment``, ``simulate_like``, ``simulate_removal``.
"""

from __future__ import annotations

import json
import re
import secrets as _secrets
from datetime import datetime, timezone, UTC
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    read_params,
)

PROFESSIONAL = {"BUSINESS", "MEDIA_CREATOR"}


class GraphError(Exception):
    def __init__(self, status: int, message: str, code: int, subcode: Optional[int] = None,
                 kind: str = "OAuthException", headers: Optional[dict] = None):
        self.status, self.message, self.code, self.subcode, self.kind = status, message, code, subcode, kind
        self.headers = headers or {}


def _ts(t: float) -> str:
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S+0000")


class InstagramState:
    def __init__(self, clock: Optional[SimClock] = None, publish_quota: int = 100, calls_per_hour: int = 200):
        self.clock = clock or SimClock()
        self.publish_quota = publish_quota
        self.calls_per_hour = calls_per_hour
        self.users: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.containers: dict[str, dict] = {}
        self.media: dict[str, dict] = {}
        self.published: dict[str, list[float]] = {}
        self.limiter = RateLimiter(self.clock)
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return str(17_900_000_000_000_000 + self._n)

    def add_account(self, username: str, account_type: str = "BUSINESS",
                    permissions: tuple[str, ...] = ("instagram_basic", "instagram_content_publish",
                                                    "instagram_manage_comments", "pages_show_list")) -> tuple[str, str]:
        """Create an account and a token for it; returns (ig_user_id, access_token)."""
        uid = str(17_841_400_000_000_000 + len(self.users) + 1)
        self.users[uid] = {"id": uid, "username": username, "account_type": account_type, "name": username.title(),
                           "media_count": 0, "followers_count": 0, "follows_count": 0}
        tok = "EAAsim" + _secrets.token_urlsafe(32)
        self.tokens[tok] = {"users": {uid}, "permissions": set(permissions), "expires_at": self.clock.now() + 60 * 86400}
        self.published[uid] = []
        return uid, tok

    def simulate_comment(self, media_id: str, text: str, by: str = "carol") -> str:
        cid = self._id()
        self.media[media_id]["comments"].append({"id": cid, "text": text, "username": by, "t": self.clock.now(),
                                                 "timestamp": _ts(self.clock.now()), "like_count": 0, "hidden": False})
        return cid

    def simulate_like(self, media_id: str, n: int = 1) -> None:
        self.media[media_id]["likes"] += n

    def simulate_removal(self, media_id: str) -> None:
        self.media[media_id]["removed"] = True


def create_app(state: Optional[InstagramState] = None) -> FastAPI:
    st = state or InstagramState()
    app = FastAPI(title="Simulated Instagram Graph API (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="instagram")

    @app.exception_handler(GraphError)
    async def _err(request: Request, e: GraphError):
        err: dict[str, Any] = {"message": e.message, "type": e.kind, "code": e.code,
                               "fbtrace_id": "SIM" + _secrets.token_hex(6)}
        if e.subcode:
            err["error_subcode"] = e.subcode
        return JSONResponse({"error": err}, status_code=e.status, headers=e.headers)

    async def _auth(request: Request, params: dict) -> dict:
        tok = params.get("access_token")
        auth = request.headers.get("authorization", "")
        if not tok and auth.split(" ", 1)[0].lower() in ("bearer", "oauth"):
            tok = auth.split(" ", 1)[1].strip()
        info = st.tokens.get(str(tok or ""))
        if info is None:
            raise GraphError(400, "Invalid OAuth access token - Cannot parse access token", 190)
        if info["expires_at"] <= st.clock.now():
            raise GraphError(400, "Error validating access token: Session has expired.", 190, 463)
        return info

    def _usage(uid: str) -> dict:
        rs = st.limiter.hit(f"bucu:{uid}", st.calls_per_hour, 3600)
        pct = int(100 * rs.used / max(1, rs.limit))
        regain = 0 if rs.allowed else int((rs.reset - st.clock.now()) // 60) + 1
        h = {"X-Business-Use-Case-Usage": json.dumps({uid: [{"type": "instagram", "call_count": pct, "total_cputime": 1,
                                                              "total_time": 1, "estimated_time_to_regain_access": regain}]}),
             "X-App-Usage": json.dumps({"call_count": min(pct, 100), "total_cputime": 1, "total_time": 1})}
        if not rs.allowed:
            raise GraphError(400, "There have been too many calls to this Instagram account. Wait a bit and try again.",
                             80002, kind="OAuthException", headers=h)
        return h

    def _need(info: dict, perm: str) -> None:
        if perm not in info["permissions"]:
            raise GraphError(403, "(#10) Application does not have permission for this action", 10)

    def _unsupported(node: str) -> GraphError:
        return GraphError(400, f"Unsupported get request. Object with ID '{node}' does not exist, cannot be loaded due "
                               "to missing permissions, or does not support this operation.", 100, 33, "GraphMethodException")

    def _owner(info: dict, uid: str) -> dict:
        if uid not in st.users or uid not in info["users"]:
            raise _unsupported(uid)
        return st.users[uid]

    def _fields(params: dict) -> list[str]:
        return [f.strip() for f in str(params.get("fields", "")).split(",") if f.strip()]

    def _quota_used(uid: str) -> int:
        now = st.clock.now()
        st.published[uid] = [t for t in st.published[uid] if now - t < 86400]
        return len(st.published[uid])

    def _media_view(m: dict, fields: list[str]) -> dict:
        allv = {"id": m["id"], "caption": m["caption"], "media_type": "IMAGE", "media_url": m["media_url"],
                "permalink": m["permalink"], "timestamp": m["timestamp"], "like_count": m["likes"],
                "comments_count": len([c for c in m["comments"] if not c["hidden"]]),
                "username": st.users[m["owner"]]["username"], "owner": {"id": m["owner"]}}
        return {"id": m["id"], **{f: allv[f] for f in fields if f in allv}}

    @app.api_route("/{version}/{node}", methods=["GET"])
    async def get_node(version: str, node: str, request: Request):
        if not re.fullmatch(r"v\d+\.\d+", version):
            raise GraphError(400, "Unknown path components", 2500)
        params = await read_params(request)
        info = await _auth(request, params)
        fields = _fields(params)
        if node in st.users:
            u = _owner(info, node)
            h = _usage(node)
            _need(info, "instagram_basic")
            view = {"id": u["id"], **{f: u[f] for f in fields if f in u}}
            return JSONResponse(view, headers=h)
        if node in st.containers:
            c = st.containers[node]
            h = _usage(_owner(info, c["owner"])["id"])
            view = {"id": node, "status_code": c["status_code"], "status": c["status"]}
            return JSONResponse({"id": node, **{f: view[f] for f in fields if f in view}}, headers=h)
        m = st.media.get(node)
        if m and not m["removed"]:
            h = _usage(_owner(info, m["owner"])["id"])
            _need(info, "instagram_basic")
            return JSONResponse(_media_view(m, fields), headers=h)
        raise _unsupported(node)

    @app.api_route("/{version}/{node}/{edge}", methods=["GET", "POST"])
    async def edge(version: str, node: str, edge: str, request: Request):
        if not re.fullmatch(r"v\d+\.\d+", version):
            raise GraphError(400, "Unknown path components", 2500)
        params = await read_params(request)
        info = await _auth(request, params)
        if request.method == "GET" and edge == "comments" and node in st.media:
            m = st.media[node]
            if m["removed"]:
                raise _unsupported(node)
            h = _usage(_owner(info, m["owner"])["id"])
            _need(info, "instagram_manage_comments")
            fields = _fields(params) or ["id", "text", "timestamp"]
            data = [{"id": c["id"], **{f: c[f] for f in fields if f in c and f != "id"}}
                    for c in m["comments"] if not c["hidden"]]
            return JSONResponse({"data": data, "paging": {"cursors": {"before": "QVFI", "after": "QVFJ"}}}, headers=h)
        u = _owner(info, node)
        h = _usage(node)
        if request.method == "GET" and edge == "content_publishing_limit":
            _need(info, "instagram_content_publish")
            return JSONResponse({"data": [{"config": {"quota_total": st.publish_quota, "quota_duration": 86400},
                                           "quota_usage": _quota_used(node)}]}, headers=h)
        if request.method == "POST" and edge in ("media", "media_publish"):
            _need(info, "instagram_content_publish")
            if u["account_type"] not in PROFESSIONAL:
                raise GraphError(403, "(#10) The Content Publishing API is only available to Instagram professional "
                                      "(Business or Creator) accounts", 10)
            if edge == "media":
                url = str(params.get("image_url") or "")
                caption = str(params.get("caption") or "")
                if not url:
                    raise GraphError(400, "(#100) The parameter image_url is required", 100)
                if urlsplit(url).scheme != "https":
                    raise GraphError(400, "(#100) Param image_url must be a valid https URL", 100)
                if len(caption) > 2200 or caption.count("#") > 30 or len(re.findall(r"@\w", caption)) > 20:
                    raise GraphError(400, "(#100) The caption exceeds the limits (2200 characters, 30 hashtags, 20 @ tags)", 100)
                ok = urlsplit(url).path.lower().endswith((".jpg", ".jpeg"))
                cid = st._id()
                st.containers[cid] = {"owner": node, "image_url": url, "caption": caption, "created": st.clock.now(),
                                      "status_code": "FINISHED" if ok else "ERROR",
                                      "status": "Finished: Media has been uploaded" if ok else
                                      "Error: Unsupported image format (only JPEG is supported)"}
                return JSONResponse({"id": cid}, headers=h)
            cid = str(params.get("creation_id") or "")
            c = st.containers.get(cid)
            if c is None or c["owner"] != node:
                raise GraphError(400, "(#100) Invalid parameter creation_id", 100)
            if c["status_code"] != "FINISHED":
                raise GraphError(400, "The media is not ready for publishing, please wait for a moment", 9007, 2207027)
            if _quota_used(node) >= st.publish_quota:
                raise GraphError(400, "(#9) Application request limit reached: the account reached the maximum number "
                                      "of posts published via the API in 24 hours", 9, 2207042)
            mid = st._id()
            now = st.clock.now()
            st.media[mid] = {"id": mid, "owner": node, "caption": c["caption"], "media_url": c["image_url"],
                             "permalink": f"https://www.instagram.com/p/SIM{mid[-8:]}/", "timestamp": _ts(now), "t": now,
                             "likes": 0, "comments": [], "removed": False}
            c["status_code"], c["status"] = "PUBLISHED", "Published"
            st.published[node].append(now)
            u["media_count"] += 1
            return JSONResponse({"id": mid}, headers=h)
        raise GraphError(400, f"(#100) Tried accessing nonexisting field ({edge}) on node type (IGUser)", 100)

    return app
