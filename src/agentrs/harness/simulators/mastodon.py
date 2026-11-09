"""Local simulator of the Mastodon REST + OAuth subset used by ``MastodonConnector``.

Implements: ``POST /api/v1/apps``; ``GET /oauth/authorize`` (a simulated consent page — a code is
issued only with ``simulate_human_approval=1``, which stands for the operator clicking "Authorize";
``simulate_human_approval=0`` denies); ``POST /oauth/token`` (``authorization_code`` with PKCE S256,
``refresh_token``, ``client_credentials``); ``POST /oauth/revoke``;
``GET /api/v1/accounts/verify_credentials``, ``GET /api/v1/accounts/:id``, ``GET /api/v1/instance``
(incl. ``rules``), ``POST /api/v1/statuses`` (honours ``Idempotency-Key`` for one hour),
``GET /api/v1/statuses/:id``, ``GET /api/v1/statuses/:id/context``, ``GET /api/v1/notifications``,
``GET /api/v1/timelines/home``.

Scopes are enforced (``read`` covers ``read:*``; 403 "This action is outside the authorized
scopes"). Rate limits: ``X-RateLimit-Limit`` / ``-Remaining`` / ``-Reset`` (ISO 8601), 300 requests
per 5 minutes per token and 300 statuses per 3 hours per account by default; 429 when exceeded.

Other users: ``simulate_reply``, ``simulate_mention``, ``simulate_favourite``, ``simulate_reblog``,
``simulate_post``, ``simulate_removal`` (moderation).
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
import secrets as _secrets
from typing import Any, Optional
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    iso_z,
    read_params,
)

OOB = "urn:ietf:wg:oauth:2.0:oob"
INVALID_GRANT = ("The provided authorization grant is invalid, expired, revoked, does not match the redirection URI "
                 "used in the authorization request, or was issued to another client.")


class MastoError(Exception):
    def __init__(self, status: int, body: dict, headers: Optional[dict] = None):
        self.status, self.body, self.headers = status, body, headers or {}


def _scope_ok(granted: set[str], needed: str) -> bool:
    return needed in granted or needed.split(":")[0] in granted


class MastodonState:
    def __init__(self, domain: str = "mastodon.sim", clock: Optional[SimClock] = None, token_ttl: Optional[float] = None,
                 issue_refresh_tokens: bool = True):
        self.domain = domain
        self.clock = clock or SimClock()
        self.token_ttl = token_ttl  # None: non-expiring tokens (Mastodon's historical default)
        self.issue_refresh_tokens = issue_refresh_tokens
        self.apps: dict[str, dict] = {}
        self.codes: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.refresh_tokens: dict[str, str] = {}  # refresh -> access token
        self.accounts: dict[str, dict] = {}
        self.by_username: dict[str, str] = {}
        self.statuses: dict[str, dict] = {}
        self.notifications: dict[str, list[dict]] = {}
        self.follows: dict[str, set[str]] = {}
        self.idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self.rules = [{"id": "1", "text": "Automated accounts must set the bot flag and name their operator.", "hint": ""},
                      {"id": "2", "text": "No unsolicited replies or mass mentions from bots.", "hint": ""},
                      {"id": "3", "text": "Simulated instance: nothing posted here is public.", "hint": ""}]
        self.limits = {"requests": (300, 300), "statuses": (300, 3 * 3600)}
        self.limiter = RateLimiter(self.clock)
        self._n = 0
        self.max_characters = 500

    def _id(self) -> str:
        self._n += 1
        return str(113_000_000_000_000_000 + self._n)

    # ------------------------------------------------------------------ setup
    def add_account(self, username: str, display_name: str = "", bot: bool = False) -> str:
        if username in self.by_username:
            return self.by_username[username]
        aid = self._id()
        self.accounts[aid] = {"id": aid, "username": username, "acct": username, "display_name": display_name or username,
                              "locked": False, "bot": bot, "discoverable": True, "group": False,
                              "created_at": iso_z(self.clock.now()), "note": "", "url": f"https://{self.domain}/@{username}",
                              "avatar": "", "header": "", "followers_count": 0, "following_count": 0,
                              "statuses_count": 0, "last_status_at": None, "emojis": [], "fields": []}
        self.by_username[username] = aid
        self.notifications[aid] = []
        self.follows[aid] = set()
        return aid

    def follow(self, follower: str, followed: str) -> None:
        self.follows[self.by_username[follower]].add(self.by_username[followed])

    def issue_token(self, account: str, scopes: str = "read write", client_id: str = "") -> str:
        """Directly mint a user token (tests that do not exercise the OAuth flow)."""
        aid = self.by_username.get(account, account)
        return self._token(client_id, aid, set(scopes.split()))["access_token"]

    def _token(self, client_id: str, account_id: Optional[str], scopes: set[str]) -> dict:
        tok = "sim_mst_" + _secrets.token_urlsafe(24)
        now = self.clock.now()
        self.tokens[tok] = {"client_id": client_id, "account_id": account_id, "scopes": scopes, "created_at": now,
                            "expires_at": now + self.token_ttl if self.token_ttl else None, "revoked": False}
        out = {"access_token": tok, "token_type": "Bearer", "scope": " ".join(sorted(scopes)), "created_at": int(now)}
        if self.token_ttl:
            out["expires_in"] = int(self.token_ttl)
        if self.issue_refresh_tokens and account_id:
            rt = "sim_mst_r_" + _secrets.token_urlsafe(24)
            self.refresh_tokens[rt] = tok
            self.tokens[tok]["refresh_token"] = rt
            out["refresh_token"] = rt
        return out

    # ------------------------------------------------------------------ statuses
    def _content(self, text: str) -> tuple[str, list[dict]]:
        mentions, parts = [], []
        for chunk in re.split(r"(@[A-Za-z0-9_]+(?:@[A-Za-z0-9.-]+)?)", text):
            m = re.fullmatch(r"@([A-Za-z0-9_]+)(?:@([A-Za-z0-9.-]+))?", chunk)
            if m and (m.group(2) in (None, self.domain)) and m.group(1) in self.by_username:
                a = self.accounts[self.by_username[m.group(1)]]
                mentions.append({"id": a["id"], "username": a["username"], "acct": a["acct"], "url": a["url"]})
                parts.append(f'<span class="h-card"><a href="{a["url"]}" class="u-url mention">@<span>'
                             f'{a["username"]}</span></a></span>')
            else:
                parts.append(html.escape(chunk).replace("\n", "<br />"))
        return "<p>" + "".join(parts) + "</p>", mentions

    def create_status(self, account_id: str, text: str, in_reply_to_id: Optional[str] = None, visibility: str = "public",
                      spoiler_text: str = "", language: Optional[str] = None, application: Optional[dict] = None) -> dict:
        sid = self._id()
        a = self.accounts[account_id]
        content, mentions = self._content(text)
        parent = self.statuses.get(in_reply_to_id) if in_reply_to_id else None
        st = {"id": sid, "created_at": iso_z(self.clock.now()), "t": self.clock.now(), "in_reply_to_id": in_reply_to_id,
              "in_reply_to_account_id": parent["account_id"] if parent else None, "sensitive": bool(spoiler_text),
              "spoiler_text": spoiler_text, "visibility": visibility, "language": language or "en",
              "uri": f"https://{self.domain}/users/{a['username']}/statuses/{sid}",
              "url": f"https://{self.domain}/@{a['username']}/{sid}", "content": content, "text": text,
              "mentions": mentions, "account_id": account_id, "favourited_by": [], "reblogged_by": [],
              "application": application, "removed": False, "edited_at": None}
        self.statuses[sid] = st
        a["statuses_count"] += 1
        a["last_status_at"] = st["created_at"][:10]
        notified = set()
        for m in mentions:
            if m["id"] != account_id:
                self._notify(m["id"], "mention", account_id, sid)
                notified.add(m["id"])
        return st

    def _notify(self, to: str, kind: str, from_account: str, status_id: Optional[str]) -> None:
        self.notifications[to].append({"id": self._id(), "type": kind, "created_at": iso_z(self.clock.now()),
                                       "account_id": from_account, "status_id": status_id})

    def replies(self, sid: str) -> list[dict]:
        return [s for s in self.statuses.values() if s["in_reply_to_id"] == sid and not s["removed"]]

    def status_view(self, st: dict) -> dict:
        out = {k: v for k, v in st.items() if k not in ("t", "text", "account_id", "favourited_by", "reblogged_by",
                                                         "removed")}
        out.update(account=self.accounts[st["account_id"]], replies_count=len(self.replies(st["id"])),
                   reblogs_count=len(st["reblogged_by"]), favourites_count=len(st["favourited_by"]),
                   reblog=None, media_attachments=[], tags=[], emojis=[], card=None, poll=None)
        return out

    # ------------------------------------------------------------------ other users
    def simulate_post(self, by: str, text: str) -> str:
        return self.create_status(self.add_account(by), text)["id"]

    def simulate_reply(self, status_id: str, text: str, by: str = "carol") -> str:
        parent = self.statuses[status_id]
        author = self.accounts[parent["account_id"]]["username"]
        body = text if f"@{author}" in text else f"@{author} {text}"
        return self.create_status(self.add_account(by), body, in_reply_to_id=status_id)["id"]

    def simulate_mention(self, text: str, by: str = "dave") -> str:
        return self.create_status(self.add_account(by), text)["id"]

    def simulate_favourite(self, status_id: str, by: str = "erin") -> None:
        aid = self.add_account(by)
        st = self.statuses[status_id]
        st["favourited_by"].append(aid)
        self._notify(st["account_id"], "favourite", aid, status_id)

    def simulate_reblog(self, status_id: str, by: str = "frank") -> None:
        aid = self.add_account(by)
        st = self.statuses[status_id]
        st["reblogged_by"].append(aid)
        self._notify(st["account_id"], "reblog", aid, status_id)

    def simulate_removal(self, status_id: str) -> None:
        self.statuses[status_id]["removed"] = True


def create_app(state: Optional[MastodonState] = None) -> FastAPI:
    st = state or MastodonState()
    app = FastAPI(title="Simulated Mastodon (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="mastodon")

    @app.exception_handler(MastoError)
    async def _err(request: Request, e: MastoError):
        return JSONResponse(e.body, status_code=e.status, headers=e.headers)

    def _rl(key: str, principal: str, response: Optional[Response]) -> None:
        lim, win = st.limits[key]
        rs = st.limiter.hit(f"{key}:{principal}", lim, win)
        h = {"X-RateLimit-Limit": str(rs.limit), "X-RateLimit-Remaining": str(rs.remaining),
             "X-RateLimit-Reset": iso_z(rs.reset)}
        if not rs.allowed:
            raise MastoError(429, {"error": "Too many requests"}, h)
        if response is not None:
            response.headers.update(h)

    def token_info(request: Request) -> Optional[dict]:
        auth = request.headers.get("authorization", "")
        if not auth:
            return None
        if not auth.lower().startswith("bearer "):
            raise MastoError(401, {"error": "The access token is invalid"})
        tok = st.tokens.get(auth[7:].strip())
        if tok is None:
            raise MastoError(401, {"error": "The access token is invalid"})
        if tok["revoked"]:
            raise MastoError(401, {"error": "The access token was revoked"})
        if tok["expires_at"] and tok["expires_at"] <= st.clock.now():
            raise MastoError(401, {"error": "The access token expired"})
        return tok

    def need(scope: Optional[str], user: bool = True):
        async def dep(request: Request, response: Response) -> Optional[dict]:
            tok = token_info(request)
            principal = request.headers.get("authorization", "")[-12:] or (request.client.host if request.client else "-")
            _rl("requests", principal, response)
            if scope is None:
                return tok
            if tok is None:
                raise MastoError(401, {"error": "The access token is invalid"})
            if user and not tok["account_id"]:
                raise MastoError(422, {"error": "This method requires an authenticated user"})
            if not _scope_ok(tok["scopes"], scope):
                raise MastoError(403, {"error": "This action is outside the authorized scopes"})
            return tok
        return dep

    # ------------------------------------------------------------------ OAuth
    @app.post("/api/v1/apps")
    async def create_app_(request: Request):
        p = await read_params(request)
        if not p.get("client_name"):
            raise MastoError(422, {"error": "Validation failed: Application name can't be blank"})
        uris = p.get("redirect_uris") or OOB
        uris = uris if isinstance(uris, list) else str(uris).split()
        scopes = str(p.get("scopes") or "read").split()
        cid, secret = _secrets.token_urlsafe(32), _secrets.token_urlsafe(32)
        aid = st._id()
        st.apps[cid] = {"id": aid, "name": p["client_name"], "website": p.get("website"), "redirect_uris": uris,
                        "scopes": scopes, "client_secret": secret}
        return {"id": aid, "name": p["client_name"], "website": p.get("website"), "scopes": scopes,
                "redirect_uri": "\n".join(uris), "redirect_uris": uris, "client_id": cid, "client_secret": secret,
                "client_secret_expires_at": 0, "vapid_key": "BSIMULATEDVAPIDKEY="}

    def _app_scopes_ok(app_: dict, scopes: set[str]) -> bool:
        return all(any(s == a or s.split(":")[0] == a for a in app_["scopes"]) for s in scopes)

    @app.get("/oauth/authorize")
    async def authorize(request: Request):
        q = dict(request.query_params)
        app_ = st.apps.get(q.get("client_id", ""))
        if app_ is None or q.get("response_type") != "code":
            return HTMLResponse("<h1>The client is not authorized to perform this request (simulated)</h1>", 400)
        redirect = q.get("redirect_uri", "")
        if redirect not in app_["redirect_uris"]:
            return HTMLResponse("<h1>The redirect uri included is not valid. (simulated)</h1>", 400)
        scopes = set((q.get("scope") or "read").split())

        def back(**params) -> Response:
            if q.get("state"):
                params["state"] = q["state"]
            sep = "&" if "?" in redirect else "?"
            return RedirectResponse(f"{redirect}{sep}{urlencode(params)}", status_code=302)

        if not _app_scopes_ok(app_, scopes):
            return back(error="invalid_scope", error_description="The requested scope is invalid, unknown, or malformed.")
        method = q.get("code_challenge_method")
        if q.get("code_challenge") and method != "S256":
            return back(error="invalid_request", error_description="Code challenge method is not supported")
        approval = q.get("simulate_human_approval")
        if approval is None:  # what a browser would show the signed-in operator
            body = (f"<html><body><h1>Authorization required</h1><p><b>{html.escape(app_['name'])}</b> would like "
                    f"permission to access your account: {html.escape(' '.join(sorted(scopes)))}</p>"
                    "<p>SIMULATED consent page. A human operator must approve; in tests append "
                    "<code>simulate_human_approval=1</code> (approve) or <code>=0</code> (deny).</p></body></html>")
            return HTMLResponse(body, 200)
        if approval != "1":
            return back(error="access_denied", error_description="The resource owner or authorization server denied the request.")
        username = q.get("simulate_account") or next(iter(st.by_username), None)
        if username not in st.by_username:
            return HTMLResponse("<h1>No such account (simulated)</h1>", 400)
        code = _secrets.token_urlsafe(32)
        st.codes[code] = {"client_id": q["client_id"], "redirect_uri": redirect, "scopes": scopes,
                          "challenge": q.get("code_challenge"), "account_id": st.by_username[username],
                          "expires_at": st.clock.now() + 600, "used": False}
        if redirect == OOB:
            return HTMLResponse(f"<html><body><p>Authorization code:</p><input class='oauth-code' value='{code}'>"
                                "</body></html>")
        return back(code=code)

    def _client(request: Request, p: dict) -> dict:
        cid, secret = p.get("client_id"), p.get("client_secret")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                cid, secret = base64.b64decode(auth[6:]).decode().split(":", 1)
            except (ValueError, UnicodeDecodeError):
                cid = secret = None
        app_ = st.apps.get(cid or "")
        if app_ is None or not _secrets.compare_digest(str(secret or ""), app_["client_secret"]):
            raise MastoError(401, {"error": "invalid_client", "error_description": (
                "Client authentication failed due to unknown client, no client authentication included, or "
                "unsupported authentication method.")})
        return {"client_id": cid, **app_}

    @app.post("/oauth/token")
    async def token(request: Request):
        p = await read_params(request)
        client = _client(request, p)
        grant = p.get("grant_type")
        if grant == "authorization_code":
            c = st.codes.get(p.get("code", ""))
            if (c is None or c["used"] or c["client_id"] != client["client_id"] or c["redirect_uri"] != p.get("redirect_uri")
                    or c["expires_at"] <= st.clock.now()):
                raise MastoError(400, {"error": "invalid_grant", "error_description": INVALID_GRANT})
            if c["challenge"]:
                v = p.get("code_verifier") or ""
                calc = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).decode().rstrip("=")
                if not v or not _secrets.compare_digest(calc, c["challenge"]):
                    raise MastoError(400, {"error": "invalid_grant", "error_description": INVALID_GRANT})
            c["used"] = True
            return st._token(client["client_id"], c["account_id"], c["scopes"])
        if grant == "refresh_token":
            old = st.refresh_tokens.pop(p.get("refresh_token", ""), None)
            if old is None or st.tokens[old]["client_id"] != client["client_id"]:
                raise MastoError(400, {"error": "invalid_grant", "error_description": INVALID_GRANT})
            info = st.tokens[old]
            info["revoked"] = True
            return st._token(client["client_id"], info["account_id"], info["scopes"])
        if grant == "client_credentials":
            scopes = set((p.get("scope") or "read").split())
            return st._token(client["client_id"], None, scopes)
        raise MastoError(400, {"error": "unsupported_grant_type",
                               "error_description": "The authorization grant type is not supported by the authorization server."})

    @app.post("/oauth/revoke")
    async def revoke(request: Request):
        p = await read_params(request)
        client = _client(request, p)
        tok = p.get("token", "")
        if tok in st.refresh_tokens:
            tok = st.refresh_tokens.pop(tok)
        info = st.tokens.get(tok)
        if info and info["client_id"] == client["client_id"]:
            info["revoked"] = True
            st.refresh_tokens.pop(info.get("refresh_token", ""), None)
        return {}  # RFC 7009: 200 whether or not the token existed

    # ------------------------------------------------------------------ REST
    @app.get("/api/v1/instance")
    async def instance(_: Any = Depends(need(None))):
        return {"uri": st.domain, "title": "Simulated Mastodon", "short_description": "agentrs local simulator",
                "description": "SIMULATED instance for connector tests", "email": "admin@" + st.domain,
                "version": "4.3.0 (compatible; agentrs-simulator)", "urls": {"streaming_api": f"wss://{st.domain}"},
                "stats": {"user_count": len(st.accounts), "status_count": len(st.statuses), "domain_count": 1},
                "languages": ["en"], "registrations": False, "approval_required": True, "invites_enabled": False,
                "configuration": {"statuses": {"max_characters": st.max_characters, "max_media_attachments": 4,
                                               "characters_reserved_per_url": 23}},
                "contact_account": None, "rules": st.rules}

    @app.get("/api/v1/accounts/verify_credentials")
    async def verify(tok: dict = Depends(need("read:accounts"))):
        a = dict(st.accounts[tok["account_id"]])
        a["source"] = {"privacy": "public", "sensitive": False, "language": "en", "note": a["note"], "fields": []}
        return a

    @app.get("/api/v1/accounts/{account_id}")
    async def account(account_id: str, _: Any = Depends(need(None))):
        if account_id not in st.accounts:
            raise MastoError(404, {"error": "Record not found"})
        return st.accounts[account_id]

    @app.post("/api/v1/statuses")
    async def post_status(request: Request, response: Response, tok: dict = Depends(need("write:statuses"))):
        p = await read_params(request)
        aid = tok["account_id"]
        key = request.headers.get("idempotency-key")
        if key and (aid, key) in st.idempotency:
            sid, t = st.idempotency[(aid, key)]
            if st.clock.now() - t < 3600:
                return st.status_view(st.statuses[sid])
        text = str(p.get("status") or "")
        if not text.strip() and not p.get("media_ids"):
            raise MastoError(422, {"error": "Validation failed: Text can't be blank"})
        if len(text) > st.max_characters:
            raise MastoError(422, {"error": f"Validation failed: Text character limit of {st.max_characters} exceeded"})
        reply_to = p.get("in_reply_to_id")
        if reply_to and (reply_to not in st.statuses or st.statuses[reply_to]["removed"]):
            raise MastoError(404, {"error": "Record not found"})
        vis = p.get("visibility") or "public"
        if vis not in ("public", "unlisted", "private", "direct"):
            raise MastoError(422, {"error": "Validation failed: Visibility is not included in the list"})
        _rl("statuses", aid, response)
        app_ = st.apps.get(tok["client_id"])
        s = st.create_status(aid, text, reply_to, vis, str(p.get("spoiler_text") or ""), p.get("language"),
                             {"name": app_["name"], "website": app_.get("website")} if app_ else None)
        if key:
            st.idempotency[(aid, key)] = (s["id"], st.clock.now())
        return st.status_view(s)

    def _visible(sid: str, tok: Optional[dict]) -> dict:
        s = st.statuses.get(sid)
        if s is None or s["removed"]:
            raise MastoError(404, {"error": "Record not found"})
        if s["visibility"] in ("private", "direct") and not (tok and tok["account_id"] == s["account_id"]):
            raise MastoError(404, {"error": "Record not found"})
        return s

    @app.get("/api/v1/statuses/{sid}")
    async def get_status(sid: str, tok: Optional[dict] = Depends(need(None))):
        return st.status_view(_visible(sid, tok))

    @app.get("/api/v1/statuses/{sid}/context")
    async def context(sid: str, tok: Optional[dict] = Depends(need(None))):
        s = _visible(sid, tok)
        anc, cur = [], s
        while cur["in_reply_to_id"] and cur["in_reply_to_id"] in st.statuses:
            cur = st.statuses[cur["in_reply_to_id"]]
            if not cur["removed"]:
                anc.insert(0, st.status_view(cur))
        desc, stack = [], [s["id"]]
        while stack:
            for r in sorted(st.replies(stack.pop(0)), key=lambda x: x["id"]):
                desc.append(st.status_view(r))
                stack.append(r["id"])
        return {"ancestors": anc, "descendants": desc}

    def _page(items: list[dict], q: dict, response: Response, path: str) -> list[dict]:
        limit = max(1, min(int(q.get("limit", 20)), 40))
        if q.get("max_id"):
            items = [i for i in items if int(i["id"]) < int(q["max_id"])]
        if q.get("since_id"):
            items = [i for i in items if int(i["id"]) > int(q["since_id"])]
        if q.get("min_id"):
            items = sorted([i for i in items if int(i["id"]) > int(q["min_id"])], key=lambda i: int(i["id"]))[:limit]
            items = sorted(items, key=lambda i: int(i["id"]), reverse=True)
        page = items[:limit]
        if page:
            base = f"https://{st.domain}{path}"
            response.headers["Link"] = (f'<{base}?max_id={page[-1]["id"]}>; rel="next", '
                                        f'<{base}?min_id={page[0]["id"]}>; rel="prev"')
        return page

    @app.get("/api/v1/notifications")
    async def notifications(request: Request, response: Response, tok: dict = Depends(need("read:notifications"))):
        q = request.query_params
        types = q.getlist("types[]") or q.getlist("types")
        excl = q.getlist("exclude_types[]") or q.getlist("exclude_types")
        items = []
        for n in reversed(st.notifications[tok["account_id"]]):
            if (types and n["type"] not in types) or n["type"] in excl:
                continue
            s = st.statuses.get(n["status_id"]) if n["status_id"] else None
            items.append({"id": n["id"], "type": n["type"], "created_at": n["created_at"],
                          "account": st.accounts[n["account_id"]],
                          "status": st.status_view(s) if s and not s["removed"] else None})
        return _page(items, dict(q), response, "/api/v1/notifications")

    @app.get("/api/v1/timelines/home")
    async def home(request: Request, response: Response, tok: dict = Depends(need("read:statuses"))):
        aid = tok["account_id"]
        who = st.follows[aid] | {aid}
        items = [st.status_view(s) for s in sorted(st.statuses.values(), key=lambda s: int(s["id"]), reverse=True)
                 if s["account_id"] in who and not s["removed"] and s["visibility"] != "direct"]
        return _page(items, dict(request.query_params), response, "/api/v1/timelines/home")

    return app
