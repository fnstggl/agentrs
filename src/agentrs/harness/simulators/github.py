"""Local simulator of the GitHub REST API subset used by ``GitHubConnector`` (GitHub App auth).

Auth:
* App JWT (RS256, ``iss`` = app id, ``exp`` at most 10 minutes ahead) for ``GET /app`` and
  ``POST /app/installations/{id}/access_tokens`` (optionally down-scoped with ``repositories`` /
  ``permissions``; 422 if more than the installation has). Installation tokens (``ghs_...``) expire
  after one hour and carry the installation's permissions; ``DELETE /installation/token`` revokes.
* Installation token (``Authorization: Bearer|token ghs_...``) for the REST endpoints.

REST: ``GET /installation/repositories``, ``GET /repos/{o}/{r}``, ``GET .../issues`` (+ ``/{n}``),
``GET .../contents/{path}?ref=``, ``PUT .../contents/{path}`` (sha required to update; 409 on sha
mismatch or protected branch; ``.github/workflows/*`` needs the ``workflows`` permission),
``GET .../git/ref/heads/{b}``, ``POST .../git/refs``, ``POST .../pulls``, ``GET .../pulls/{n}``,
``GET .../pulls/{n}/reviews``, ``GET .../commits/{ref}/check-runs``,
``POST .../actions/workflows/{id}/dispatches``, ``GET /rate_limit``.

Enforced: installation permissions (``Resource not accessible by integration`` 403) and repository
selection (404 for repositories outside it — GitHub does not reveal their existence). Rate limits:
``x-ratelimit-limit/remaining/used/reset/resource``; primary limit exhaustion returns 403 with
``x-ratelimit-remaining: 0`` (as GitHub does).

Simulated others / CI: ``run_ci(repo, ref, conclusion)`` creates a check run and delivers a
``check_run`` webhook signed with ``X-Hub-Signature-256``; ``simulate_review``, ``simulate_merge``,
``simulate_close``, ``simulate_issue``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets as _secrets
import uuid
from typing import Any, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from agentrs.harness.simulators._common import (
    RateLimiter,
    SimClock,
    SimulatorMarker,
    WebhookTarget,
    deliver_webhook,
    iso_z,
    jdump,
    read_params,
)

API = "https://api.github.com"
DOCS = "https://docs.github.com/rest"
LEVEL = {"none": 0, "read": 1, "write": 2, "admin": 3}


class GHError(Exception):
    def __init__(self, status: int, message: str, headers: Optional[dict] = None, errors: Optional[list] = None,
                 docs: str = DOCS):
        self.status, self.message, self.headers, self.errors, self.docs = status, message, headers or {}, errors, docs


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def generate_app_key() -> tuple[str, str]:
    """(private_key_pem, public_key_pem) for a simulated GitHub App."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                             serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


class GitHubState:
    def __init__(self, clock: Optional[SimClock] = None, webhook_target: WebhookTarget = None,
                 webhook_secret: Optional[str] = None):
        self.clock = clock or SimClock()
        self.apps: dict[int, dict] = {}
        self.installations: dict[int, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.repos: dict[str, dict] = {}
        self.limits = {"core": (5000, 3600), "app": (5000, 3600)}
        self.limiter = RateLimiter(self.clock)
        self.webhook_target = webhook_target
        self.webhook_secret = webhook_secret
        self.deliveries: list[dict] = []
        self._n = 0
        self.requests: list[tuple[str, str]] = []

    def _id(self) -> int:
        self._n += 1
        return 1000 + self._n

    # ------------------------------------------------------------------ setup
    def create_app(self, app_id: int = 424242, slug: str = "agentrs-sim") -> str:
        """Register a GitHub App and return its private key PEM (what the operator downloads)."""
        priv, pub = generate_app_key()
        self.register_app(app_id, pub, slug)
        return priv

    def register_app(self, app_id: int, public_key_pem: str, slug: str = "agentrs-sim") -> None:
        self.apps[int(app_id)] = {"id": int(app_id), "slug": slug, "name": slug, "public_key": public_key_pem,
                                  "owner": {"login": "acme", "id": 1, "type": "Organization"}}

    def install(self, app_id: int, installation_id: int, repositories: list[str], permissions: dict[str, str],
                account: str = "acme", selection: str = "selected") -> None:
        self.installations[int(installation_id)] = {
            "id": int(installation_id), "app_id": int(app_id), "account": {"login": account, "id": 1, "type": "Organization"},
            "permissions": {"metadata": "read", **permissions}, "repository_selection": selection,
            "repositories": list(repositories)}

    def add_repo(self, full_name: str, files: Optional[dict[str, str]] = None, default_branch: str = "main",
                 workflows: Optional[dict[str, dict]] = None, private: bool = True,
                 protected: tuple[str, ...] = ("main",)) -> dict:
        owner, name = full_name.split("/")
        tree = {p: c.encode() for p, c in (files or {"README.md": f"# {name}\n"}).items()}
        repo = {"id": self._id(), "full_name": full_name, "owner": owner, "name": name, "private": private,
                "default_branch": default_branch, "branches": {}, "commits": {}, "issues": {}, "next_number": 1,
                "check_runs": [], "workflows": {}, "workflow_runs": [], "protected": set(protected),
                "created_at": iso_z(self.clock.now(), ms=False)}
        self.repos[full_name] = repo
        sha = self._commit(repo, tree, [], "Initial commit")
        repo["branches"][default_branch] = sha
        for fname, wf in (workflows or {}).items():
            wid = wf.get("id") or self._id()
            repo["workflows"][wid] = {"id": wid, "name": wf.get("name", fname), "path": f".github/workflows/{fname}",
                                      "state": "active", "dispatch": wf.get("dispatch", True)}
        return repo

    def _commit(self, repo: dict, tree: dict[str, bytes], parents: list[str], message: str, author: str = "sim") -> str:
        th = hashlib.sha1(b"".join(p.encode() + blob_sha(c).encode() for p, c in sorted(tree.items()))).hexdigest()
        sha = hashlib.sha1(f"{parents}|{th}|{message}|{self._id()}".encode()).hexdigest()
        repo["commits"][sha] = {"sha": sha, "tree": dict(tree), "tree_sha": th, "parents": list(parents),
                                "message": message, "date": iso_z(self.clock.now(), ms=False), "author": author}
        return sha

    def resolve(self, repo: dict, ref: str) -> Optional[str]:
        if ref in repo["branches"]:
            return repo["branches"][ref]
        if ref.startswith("refs/heads/") and ref[11:] in repo["branches"]:
            return repo["branches"][ref[11:]]
        if ref in repo["commits"]:
            return ref
        return None

    # ------------------------------------------------------------------ webhooks / CI / other users
    def _deliver(self, event: str, payload: dict, installation_id: Optional[int] = None) -> Optional[dict]:
        if installation_id is not None:
            payload = {**payload, "installation": {"id": installation_id, "node_id": f"MDIz{installation_id}"}}
        body = jdump(payload)
        app_id = next(iter(self.apps), 0)
        headers = {"Content-Type": "application/json", "User-Agent": "GitHub-Hookshot/simulated",
                   "X-GitHub-Event": event, "X-GitHub-Delivery": str(uuid.uuid5(uuid.NAMESPACE_URL, f"gh-{self._id()}")),
                   "X-GitHub-Hook-ID": "1", "X-GitHub-Hook-Installation-Target-Type": "integration",
                   "X-GitHub-Hook-Installation-Target-ID": str(app_id), "X-Agentrs-Simulator": "1"}
        if self.webhook_secret:
            key = self.webhook_secret.encode()
            headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(key, body, hashlib.sha256).hexdigest()
            headers["X-Hub-Signature"] = "sha1=" + hmac.new(key, body, hashlib.sha1).hexdigest()
        return deliver_webhook(self.webhook_target, headers, body, self.deliveries)

    def _installation_for(self, full_name: str) -> Optional[int]:
        for i in self.installations.values():
            if i["repository_selection"] == "all" or full_name in i["repositories"]:
                return i["id"]
        return None

    def repo_view(self, repo: dict) -> dict:
        return {"id": repo["id"], "node_id": f"R_{repo['id']}", "name": repo["name"], "full_name": repo["full_name"],
                "private": repo["private"], "owner": {"login": repo["owner"], "id": 1, "type": "Organization"},
                "html_url": f"https://github.com/{repo['full_name']}", "url": f"{API}/repos/{repo['full_name']}",
                "default_branch": repo["default_branch"], "visibility": "private" if repo["private"] else "public",
                "archived": False, "disabled": False, "created_at": repo["created_at"],
                "pushed_at": iso_z(self.clock.now(), ms=False)}

    def run_ci(self, repo_name: str, ref: str, conclusion: Optional[str] = "success", name: str = "ci/simulated") -> dict:
        """Simulated CI: a check run for ``ref``'s head commit, completed if ``conclusion`` is given."""
        repo = self.repos[repo_name]
        if ref.startswith("pull/"):
            ref = repo["issues"][int(ref.split("/")[1])]["head_ref"]
        sha = self.resolve(repo, ref)
        if sha is None:
            raise KeyError(f"unknown ref {ref}")
        now = iso_z(self.clock.now(), ms=False)
        cr = {"id": self._id(), "head_sha": sha, "node_id": f"CR_{self._n}", "external_id": "", "name": name,
              "status": "completed" if conclusion else "in_progress", "conclusion": conclusion, "started_at": now,
              "completed_at": now if conclusion else None,
              "output": {"title": f"Simulated CI: {conclusion or 'running'}", "summary": "agentrs simulator", "text": None},
              "app": {"id": 15368, "slug": "simulated-ci", "name": "Simulated CI"},
              "check_suite": {"id": self._id()}, "html_url": f"https://github.com/{repo_name}/runs/{self._n}",
              "details_url": f"https://ci.example/{self._n}",
              # like GitHub: open pull requests in this repository whose head is this commit
              "pull_requests": [{"number": n, "head": {"ref": i.get("head_ref"), "sha": sha}}
                                for n, i in repo["issues"].items() if i.get("head_ref") and i.get("state") == "open"
                                and self.resolve(repo, i["head_ref"]) == sha]}
        repo["check_runs"].append(cr)
        self._deliver("check_run", {"action": "completed" if conclusion else "created", "check_run": cr,
                                    "repository": self.repo_view(repo), "sender": {"login": "simulated-ci[bot]"}},
                      self._installation_for(repo_name))
        return cr

    def complete_check_run(self, repo_name: str, check_run_id: int, conclusion: str) -> dict:
        repo = self.repos[repo_name]
        cr = next(c for c in repo["check_runs"] if c["id"] == check_run_id)
        cr.update(status="completed", conclusion=conclusion, completed_at=iso_z(self.clock.now(), ms=False))
        self._deliver("check_run", {"action": "completed", "check_run": cr, "repository": self.repo_view(repo),
                                    "sender": {"login": "simulated-ci[bot]"}}, self._installation_for(repo_name))
        return cr

    def simulate_issue(self, repo_name: str, title: str, body: str = "", by: str = "reporter") -> int:
        repo = self.repos[repo_name]
        n = repo["next_number"]
        repo["next_number"] += 1
        now = iso_z(self.clock.now(), ms=False)
        repo["issues"][n] = {"number": n, "kind": "issue", "title": title, "body": body, "user": by, "state": "open",
                             "created_at": now, "updated_at": now, "closed_at": None, "comments": 0}
        return n

    def simulate_review(self, repo_name: str, number: int, state: str = "APPROVED", body: str = "",
                        by: str = "maintainer") -> dict:
        repo = self.repos[repo_name]
        pr = repo["issues"][number]
        rv = {"id": self._id(), "node_id": f"PRR_{self._n}", "user": {"login": by, "type": "User"}, "body": body,
              "state": state, "submitted_at": iso_z(self.clock.now(), ms=False),
              "commit_id": repo["branches"].get(pr["head_ref"]), "author_association": "MEMBER"}
        pr["reviews"].append(rv)
        self._deliver("pull_request_review", {"action": "submitted", "review": rv, "pull_request": self.pr_view(repo, pr),
                                              "repository": self.repo_view(repo), "sender": {"login": by}},
                      self._installation_for(repo_name))
        return rv

    def simulate_merge(self, repo_name: str, number: int, by: str = "maintainer") -> None:
        repo = self.repos[repo_name]
        pr = repo["issues"][number]
        head = repo["branches"][pr["head_ref"]]
        base = repo["branches"][pr["base_ref"]]
        sha = self._commit(repo, repo["commits"][head]["tree"], [base, head], f"Merge pull request #{number}", by)
        repo["branches"][pr["base_ref"]] = sha
        now = iso_z(self.clock.now(), ms=False)
        pr.update(state="closed", merged=True, merged_at=now, closed_at=now, updated_at=now, merge_commit_sha=sha,
                  merged_by=by)
        self._deliver("pull_request", {"action": "closed", "number": number, "pull_request": self.pr_view(repo, pr),
                                       "repository": self.repo_view(repo), "sender": {"login": by}},
                      self._installation_for(repo_name))

    def simulate_close(self, repo_name: str, number: int, by: str = "maintainer") -> None:
        repo = self.repos[repo_name]
        now = iso_z(self.clock.now(), ms=False)
        repo["issues"][number].update(state="closed", closed_at=now, updated_at=now)

    def expire_tokens(self) -> None:
        for t in self.tokens.values():
            t["expires_at"] = min(t["expires_at"], self.clock.now() - 1)

    # ------------------------------------------------------------------ views
    def issue_view(self, repo: dict, it: dict) -> dict:
        url = f"{API}/repos/{repo['full_name']}/issues/{it['number']}"
        out = {"id": 9_000_000 + it["number"], "node_id": f"I_{it['number']}", "number": it["number"],
               "title": it["title"], "body": it["body"], "user": {"login": it["user"]}, "labels": [],
               "state": it["state"], "state_reason": None, "comments": it.get("comments", 0),
               "created_at": it["created_at"], "updated_at": it["updated_at"], "closed_at": it["closed_at"],
               "url": url, "html_url": f"https://github.com/{repo['full_name']}/issues/{it['number']}"}
        if it["kind"] == "pr":
            out["pull_request"] = {"url": f"{API}/repos/{repo['full_name']}/pulls/{it['number']}",
                                   "html_url": f"https://github.com/{repo['full_name']}/pull/{it['number']}",
                                   "merged_at": it.get("merged_at")}
        return out

    def pr_view(self, repo: dict, pr: dict) -> dict:
        head_sha = repo["branches"].get(pr["head_ref"], pr.get("head_sha"))
        base_sha = repo["branches"].get(pr["base_ref"])
        fn = repo["full_name"]
        return {"url": f"{API}/repos/{fn}/pulls/{pr['number']}", "id": 8_000_000 + pr["number"],
                "node_id": f"PR_{pr['number']}", "html_url": f"https://github.com/{fn}/pull/{pr['number']}",
                "number": pr["number"], "state": pr["state"], "locked": False, "title": pr["title"],
                "user": {"login": pr["user"], "type": "Bot" if pr["user"].endswith("[bot]") else "User"},
                "body": pr["body"], "created_at": pr["created_at"], "updated_at": pr["updated_at"],
                "closed_at": pr["closed_at"], "merged_at": pr.get("merged_at"),
                "merge_commit_sha": pr.get("merge_commit_sha"), "draft": pr.get("draft", False),
                "head": {"label": f"{repo['owner']}:{pr['head_ref']}", "ref": pr["head_ref"], "sha": head_sha},
                "base": {"label": f"{repo['owner']}:{pr['base_ref']}", "ref": pr["base_ref"], "sha": base_sha},
                "merged": pr.get("merged", False), "mergeable": True, "mergeable_state": "clean",
                "merged_by": {"login": pr["merged_by"]} if pr.get("merged_by") else None,
                "comments": 0, "review_comments": 0, "commits": 1, "additions": 0, "deletions": 0, "changed_files": 1}


def create_app(state: Optional[GitHubState] = None) -> FastAPI:
    st = state or GitHubState()
    app = FastAPI(title="Simulated GitHub REST API (agentrs)")
    app.state.sim = st
    app.add_middleware(SimulatorMarker, provider="github")

    @app.exception_handler(GHError)
    async def _err(request: Request, e: GHError):
        body: dict[str, Any] = {"message": e.message, "documentation_url": e.docs, "status": str(e.status)}
        if e.errors:
            body["errors"] = e.errors
        return JSONResponse(body, status_code=e.status, headers=e.headers)

    def _rl(bucket: str, principal: str, response: Response) -> None:
        lim, win = st.limits[bucket]
        rs = st.limiter.hit(f"{bucket}:{principal}", lim, win)
        h = {"x-ratelimit-limit": str(rs.limit), "x-ratelimit-remaining": str(rs.remaining),
             "x-ratelimit-used": str(rs.used), "x-ratelimit-reset": str(int(rs.reset)), "x-ratelimit-resource": "core",
             "x-github-api-version-selected": "2022-11-28", "x-github-request-id": f"SIM:{st._n:08X}"}
        if not rs.allowed:
            h["x-ratelimit-remaining"] = "0"
            raise GHError(403, f"API rate limit exceeded for {principal}.", h,
                          docs="https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api")
        response.headers.update(h)

    async def app_auth(request: Request, response: Response) -> dict:
        st.requests.append((request.method, request.url.path))
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise GHError(401, "A JSON web token could not be decoded")
        tok = auth[7:].strip()
        try:
            iss = jwt.decode(tok, options={"verify_signature": False}).get("iss")
            app_ = st.apps[int(iss)]
            claims = jwt.decode(tok, app_["public_key"], algorithms=["RS256"],
                                options={"verify_exp": False, "verify_iat": False, "verify_nbf": False, "require": ["iat", "exp", "iss"]})
        except (jwt.PyJWTError, KeyError, ValueError, TypeError):
            raise GHError(401, "A JSON web token could not be decoded") from None
        now = st.clock.now()
        if claims["exp"] > now + 600 + 30:
            raise GHError(401, "'Expiration time' claim ('exp') is too far in the future")
        if claims["exp"] <= now:
            raise GHError(401, "'Expiration time' claim ('exp') must be a numeric value representing the future time "
                               "at which the assertion expires")
        if claims["iat"] > now + 60:
            raise GHError(401, "'Issued at' claim ('iat') must be an Integer representing the time that the assertion "
                               "was issued")
        _rl("app", f"app {app_['id']}", response)
        return app_

    async def inst_auth(request: Request, response: Response) -> dict:
        st.requests.append((request.method, request.url.path))
        auth = request.headers.get("authorization", "")
        parts = auth.split(" ", 1)
        info = st.tokens.get(parts[1].strip()) if len(parts) == 2 and parts[0].lower() in ("bearer", "token") else None
        if info is None or info["revoked"] or info["expires_at"] <= st.clock.now():
            raise GHError(401, "Bad credentials")
        _rl("core", f"installation ID {info['installation_id']}", response)
        return info

    def repo_for(info: dict, owner: str, name: str, perm: str, level: str = "read") -> dict:
        fn = f"{owner}/{name}"
        repo = st.repos.get(fn)
        inst = st.installations[info["installation_id"]]
        visible = repo is not None and (inst["repository_selection"] == "all" or fn in inst["repositories"])
        if visible and info["repositories"] is not None and name not in info["repositories"]:
            visible = False
        if not visible:
            raise GHError(404, "Not Found")
        if LEVEL.get(info["permissions"].get(perm, "none"), 0) < LEVEL[level]:
            raise GHError(403, "Resource not accessible by integration")
        return repo

    # ------------------------------------------------------------------ app endpoints
    @app.get("/app")
    async def get_app(app_: dict = Depends(app_auth)):
        return {"id": app_["id"], "slug": app_["slug"], "name": app_["name"], "owner": app_["owner"],
                "permissions": {}, "events": ["check_run", "pull_request", "pull_request_review"],
                "installations_count": sum(1 for i in st.installations.values() if i["app_id"] == app_["id"])}

    @app.post("/app/installations/{installation_id}/access_tokens", status_code=201)
    async def access_token(installation_id: int, request: Request, app_: dict = Depends(app_auth)):
        inst = st.installations.get(installation_id)
        if inst is None or inst["app_id"] != app_["id"]:
            raise GHError(404, "Not Found")
        p = await read_params(request)
        perms = dict(inst["permissions"])
        if p.get("permissions"):
            for k, v in p["permissions"].items():
                if LEVEL.get(perms.get(k, "none"), 0) < LEVEL.get(v, 99):
                    raise GHError(422, "The permissions requested are not granted to this installation.")
            perms = {"metadata": "read", **p["permissions"]}
        repos = None
        if p.get("repositories"):
            names = [str(r) for r in p["repositories"]]
            allowed = {fn.split("/")[1] for fn in inst["repositories"]} if inst["repository_selection"] == "selected" \
                else {r["name"] for r in st.repos.values() if r["owner"] == inst["account"]["login"]}
            if not set(names) <= allowed:
                raise GHError(422, "There is at least one repository that does not exist or is not accessible to the "
                                   "parent installation.")
            repos = names
        tok = "ghs_" + _secrets.token_urlsafe(27).replace("-", "a").replace("_", "b")[:36]
        exp = st.clock.now() + 3600
        st.tokens[tok] = {"installation_id": installation_id, "permissions": perms, "repositories": repos,
                          "expires_at": exp, "revoked": False}
        out: dict[str, Any] = {"token": tok, "expires_at": iso_z(exp, ms=False), "permissions": perms,
                               "repository_selection": "selected" if repos else inst["repository_selection"]}
        if repos:
            out["repositories"] = [st.repo_view(st.repos[f"{inst['account']['login']}/{n}"]) for n in repos
                                   if f"{inst['account']['login']}/{n}" in st.repos]
        return out

    @app.delete("/installation/token", status_code=204)
    async def revoke_token(request: Request, info: dict = Depends(inst_auth)):
        info["revoked"] = True
        return Response(status_code=204)

    @app.get("/rate_limit")
    async def rate_limit(request: Request):
        auth = request.headers.get("authorization", "")
        info = st.tokens.get(auth.split(" ", 1)[-1].strip())
        if info is None:
            raise GHError(401, "Bad credentials")
        lim, win = st.limits["core"]
        rs = st.limiter.peek(f"core:installation ID {info['installation_id']}", lim, win)
        core = {"limit": rs.limit, "used": rs.used, "remaining": rs.remaining, "reset": int(rs.reset)}
        return {"resources": {"core": core}, "rate": core}

    # ------------------------------------------------------------------ installation endpoints
    @app.get("/installation/repositories")
    async def inst_repos(info: dict = Depends(inst_auth)):
        inst = st.installations[info["installation_id"]]
        names = inst["repositories"] if inst["repository_selection"] == "selected" else \
            [fn for fn, r in st.repos.items() if r["owner"] == inst["account"]["login"]]
        if info["repositories"] is not None:
            names = [fn for fn in names if fn.split("/")[1] in info["repositories"]]
        repos = [st.repo_view(st.repos[fn]) for fn in names if fn in st.repos]
        return {"total_count": len(repos), "repository_selection": inst["repository_selection"], "repositories": repos}

    @app.get("/repos/{owner}/{name}")
    async def get_repo(owner: str, name: str, info: dict = Depends(inst_auth)):
        return st.repo_view(repo_for(info, owner, name, "metadata"))

    @app.get("/repos/{owner}/{name}/issues")
    async def list_issues(owner: str, name: str, response: Response, state: str = "open", per_page: int = 30,
                          page: int = 1, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "issues")
        items = [i for n, i in sorted(repo["issues"].items(), reverse=True) if state == "all" or i["state"] == state]
        per_page = max(1, min(per_page, 100))
        chunk = items[(page - 1) * per_page: page * per_page]
        if page * per_page < len(items):
            response.headers["Link"] = (f'<{API}/repos/{owner}/{name}/issues?state={state}&per_page={per_page}'
                                        f'&page={page + 1}>; rel="next"')
        return [st.issue_view(repo, i) for i in chunk]

    @app.get("/repos/{owner}/{name}/issues/{number}")
    async def get_issue(owner: str, name: str, number: int, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "issues")
        if number not in repo["issues"]:
            raise GHError(404, "Not Found")
        return st.issue_view(repo, repo["issues"][number])

    def _file_view(repo: dict, path: str, data: bytes, ref: str) -> dict:
        fn = repo["full_name"]
        b64 = base64.b64encode(data).decode()
        sha = blob_sha(data)
        return {"type": "file", "encoding": "base64", "size": len(data), "name": path.split("/")[-1], "path": path,
                "content": "\n".join(b64[i:i + 60] for i in range(0, len(b64), 60)) + "\n", "sha": sha,
                "url": f"{API}/repos/{fn}/contents/{path}?ref={ref}",
                "html_url": f"https://github.com/{fn}/blob/{ref}/{path}", "git_url": f"{API}/repos/{fn}/git/blobs/{sha}",
                "download_url": f"https://raw.githubusercontent.com/{fn}/{ref}/{path}"}

    @app.get("/repos/{owner}/{name}/contents/{path:path}")
    async def get_contents(owner: str, name: str, path: str, ref: Optional[str] = None, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "contents")
        ref = ref or repo["default_branch"]
        sha = st.resolve(repo, ref)
        if sha is None:
            raise GHError(404, f"No commit found for the ref {ref}")
        tree = repo["commits"][sha]["tree"]
        path = path.strip("/")
        if path in tree:
            return _file_view(repo, path, tree[path], ref)
        prefix = path + "/" if path else ""
        entries: dict[str, dict] = {}
        for p, data in tree.items():
            if p.startswith(prefix):
                rest = p[len(prefix):]
                head = rest.split("/")[0]
                full = prefix + head
                if "/" in rest:
                    entries[head] = {"type": "dir", "name": head, "path": full, "sha": None, "size": 0}
                else:
                    entries[head] = {"type": "file", "name": head, "path": full, "sha": blob_sha(data), "size": len(data)}
        if not entries:
            raise GHError(404, "Not Found")
        return sorted(entries.values(), key=lambda e: e["name"])

    @app.put("/repos/{owner}/{name}/contents/{path:path}")
    async def put_contents(owner: str, name: str, path: str, request: Request, response: Response,
                           info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "contents", "write")
        path = path.strip("/")
        if path.startswith(".github/workflows/") and LEVEL.get(info["permissions"].get("workflows", "none"), 0) < 2:
            raise GHError(403, f"Resource not accessible by integration (creating or updating workflow `{path}` "
                               "requires the `workflows` permission)")
        p = await read_params(request)
        for f in ("message", "content"):
            if f not in p or p[f] is None or (f == "message" and not p[f]):
                raise GHError(422, f'Invalid request.\n\n"{f}" wasn\'t supplied.')
        try:
            data = base64.b64decode(str(p.get("content", "")), validate=False)
        except ValueError:
            raise GHError(422, "content is not valid Base64") from None
        branch = p.get("branch") or repo["default_branch"]
        if branch not in repo["branches"]:
            raise GHError(404, f"Branch {branch} not found")
        if branch in repo["protected"]:
            raise GHError(409, "Repository rule violations found\n\nChanges must be made through a pull request.\n\n")
        head = repo["branches"][branch]
        tree = dict(repo["commits"][head]["tree"])
        existed = path in tree
        if existed:
            if not p.get("sha"):
                raise GHError(422, 'Invalid request.\n\n"sha" wasn\'t supplied.')
            if p["sha"] != blob_sha(tree[path]):
                raise GHError(409, f"{path} does not match {p['sha']}")
        tree[path] = data
        sha = st._commit(repo, tree, [head], str(p["message"]), author="agentrs-sim[bot]")
        repo["branches"][branch] = sha
        c = repo["commits"][sha]
        person = {"name": "agentrs-sim[bot]", "email": "bot@users.noreply.github.com", "date": c["date"]}
        response.status_code = 200 if existed else 201
        return {"content": _file_view(repo, path, data, branch),
                "commit": {"sha": sha, "node_id": f"C_{sha[:10]}", "url": f"{API}/repos/{repo['full_name']}/git/commits/{sha}",
                           "html_url": f"https://github.com/{repo['full_name']}/commit/{sha}", "author": person,
                           "committer": person, "message": c["message"], "tree": {"sha": c["tree_sha"]},
                           "parents": [{"sha": head}], "verification": {"verified": True, "reason": "valid"}}}

    def _ref_view(repo: dict, branch: str) -> dict:
        sha = repo["branches"][branch]
        fn = repo["full_name"]
        return {"ref": f"refs/heads/{branch}", "node_id": f"REF_{branch}", "url": f"{API}/repos/{fn}/git/refs/heads/{branch}",
                "object": {"sha": sha, "type": "commit", "url": f"{API}/repos/{fn}/git/commits/{sha}"}}

    @app.get("/repos/{owner}/{name}/git/ref/heads/{branch:path}")
    async def get_ref(owner: str, name: str, branch: str, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "contents")
        if branch not in repo["branches"]:
            raise GHError(404, "Not Found")
        return _ref_view(repo, branch)

    @app.post("/repos/{owner}/{name}/git/refs", status_code=201)
    async def create_ref(owner: str, name: str, request: Request, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "contents", "write")
        p = await read_params(request)
        ref, sha = str(p.get("ref", "")), str(p.get("sha", ""))
        if not ref.startswith("refs/") or ref.count("/") < 2:
            raise GHError(422, "Reference name must start with 'refs/' and have at least two slashes.")
        if sha not in repo["commits"]:
            raise GHError(422, "Object does not exist")
        if not ref.startswith("refs/heads/"):
            raise GHError(422, "Only branch references are supported by the simulator")
        branch = ref[11:]
        if branch in repo["branches"]:
            raise GHError(422, "Reference already exists")
        repo["branches"][branch] = sha
        return _ref_view(repo, branch)

    @app.post("/repos/{owner}/{name}/pulls", status_code=201)
    async def create_pull(owner: str, name: str, request: Request, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "pull_requests", "write")
        p = await read_params(request)
        head = str(p.get("head", "")).split(":")[-1]
        base = str(p.get("base", ""))
        if not p.get("title"):
            raise GHError(422, "Validation Failed", errors=[{"resource": "PullRequest", "field": "title", "code": "missing_field"}])
        if head not in repo["branches"]:
            raise GHError(422, "Validation Failed", errors=[{"resource": "PullRequest", "field": "head", "code": "invalid"}])
        if base not in repo["branches"]:
            raise GHError(422, "Validation Failed", errors=[{"resource": "PullRequest", "field": "base", "code": "invalid"}])
        if repo["branches"][head] == repo["branches"][base]:
            raise GHError(422, "Validation Failed", errors=[{"resource": "PullRequest", "code": "custom",
                                                             "message": f"No commits between {base} and {head}"}])
        for it in repo["issues"].values():
            if it["kind"] == "pr" and it["state"] == "open" and it["head_ref"] == head and it["base_ref"] == base:
                raise GHError(422, "Validation Failed", errors=[{"resource": "PullRequest", "code": "custom",
                                                                 "message": f"A pull request already exists for {owner}:{head}."}])
        n = repo["next_number"]
        repo["next_number"] += 1
        now = iso_z(st.clock.now(), ms=False)
        app_slug = st.apps[st.installations[info["installation_id"]]["app_id"]]["slug"]
        pr = {"number": n, "kind": "pr", "title": str(p["title"]), "body": p.get("body") or "", "user": f"{app_slug}[bot]",
              "state": "open", "created_at": now, "updated_at": now, "closed_at": None, "head_ref": head,
              "base_ref": base, "draft": bool(p.get("draft", False)), "merged": False, "reviews": []}
        repo["issues"][n] = pr
        return st.pr_view(repo, pr)

    def _pr(repo: dict, number: int) -> dict:
        it = repo["issues"].get(number)
        if it is None or it["kind"] != "pr":
            raise GHError(404, "Not Found")
        return it

    @app.get("/repos/{owner}/{name}/pulls/{number}")
    async def get_pull(owner: str, name: str, number: int, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "pull_requests")
        return st.pr_view(repo, _pr(repo, number))

    @app.get("/repos/{owner}/{name}/pulls/{number}/reviews")
    async def get_reviews(owner: str, name: str, number: int, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "pull_requests")
        return _pr(repo, number)["reviews"]

    @app.get("/repos/{owner}/{name}/commits/{ref:path}/check-runs")
    async def check_runs(owner: str, name: str, ref: str, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "checks")
        sha = st.resolve(repo, ref)
        if sha is None:
            raise GHError(422, f"No commit found for SHA: {ref}")
        runs = [c for c in repo["check_runs"] if c["head_sha"] == sha]
        return {"total_count": len(runs), "check_runs": runs}

    @app.post("/repos/{owner}/{name}/actions/workflows/{workflow_id}/dispatches", status_code=204)
    async def dispatch(owner: str, name: str, workflow_id: str, request: Request, info: dict = Depends(inst_auth)):
        repo = repo_for(info, owner, name, "actions", "write")
        wf = next((w for w in repo["workflows"].values()
                   if str(w["id"]) == workflow_id or w["path"].split("/")[-1] == workflow_id), None)
        if wf is None:
            raise GHError(404, "Not Found")
        p = await read_params(request)
        ref = str(p.get("ref", ""))
        sha = st.resolve(repo, ref)
        if not ref or sha is None or ref in repo["commits"]:
            raise GHError(422, f"No ref found for: {ref}")
        if not wf["dispatch"]:
            raise GHError(422, "Workflow does not have 'workflow_dispatch' trigger")
        inputs = p.get("inputs") or {}
        if not isinstance(inputs, dict) or len(inputs) > 10:
            raise GHError(422, "Invalid inputs")
        repo["workflow_runs"].append({"id": st._id(), "workflow_id": wf["id"], "head_branch": ref, "head_sha": sha,
                                      "event": "workflow_dispatch", "status": "queued", "inputs": inputs,
                                      "created_at": iso_z(st.clock.now(), ms=False)})
        return Response(status_code=204)

    return app
