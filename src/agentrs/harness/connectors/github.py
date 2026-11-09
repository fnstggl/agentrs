"""GitHub connector: GitHub App installation on specific, authorised repositories.

Credentials: ``app_private_key`` (secret PEM of the GitHub App). Account fields: ``app_id``,
``installation_id``, ``repositories`` (``owner/name`` list — operations on anything else are
refused), optional ``permissions`` (down-scopes every installation token) and ``branch_prefix``
(default ``agentrs/``; the connector only writes to branches with that prefix and never to the
repository's default branch or to ``.github/``).

Installation tokens are minted on demand (RS256 App JWT -> ``POST /app/installations/{id}/
access_tokens``, down-scoped to the account's repositories), cached in memory until shortly before
they expire and re-minted on 401; they are not persisted (they live one hour).
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional
from urllib.parse import quote

import jwt

from agentrs.harness.connectors._common import (
    HarnessConnector,
    ProviderHTTPError,
    outcome_payload,
    parse_time,
)
from agentrs.harness.connectors.base import ConnectorError, Subscription
from agentrs.harness.types import ActionRequest, OperationSpec, Tier

_R = dict(tier=Tier.OBSERVE, mutates=False)
_W = dict(tier=Tier.REVERSIBLE, mutates=True)
API_VERSION = "2022-11-28"
OK_CONCLUSIONS = {"success", "neutral", "skipped"}


class GitHubConnector(HarnessConnector):
    provider = "github"
    operations = {
        "list_repositories": OperationSpec(name="list_repositories", capability="read_repository", **_R),
        "read_repository": OperationSpec(name="read_repository", params=["repo"], capability="read_repository", **_R),
        "read_file": OperationSpec(name="read_file", params=["repo", "path"], optional_params=["ref"],
                                   capability="read_repository", **_R),
        "read_pull_request": OperationSpec(name="read_pull_request", params=["repo", "number"],
                                           capability="read_repository", **_R),
        "list_issues": OperationSpec(name="list_issues", params=["repo"], optional_params=["state", "per_page", "page"],
                                     capability="read_issues", **_R),
        "read_issue": OperationSpec(name="read_issue", params=["repo", "number"], capability="read_issues", **_R),
        "read_checks": OperationSpec(name="read_checks", params=["repo", "ref"], capability="read_checks", **_R),
        "create_branch": OperationSpec(name="create_branch", params=["repo", "branch"], optional_params=["from_ref"],
                                       capability="create_branch", description="POST /repos/{o}/{r}/git/refs", **_W),
        "commit_file": OperationSpec(name="commit_file", params=["repo", "branch", "path", "content", "message"],
                                     optional_params=["sha"], content_param="content", capability="commit",
                                     description="PUT /repos/{o}/{r}/contents/{path} on an agent branch", **_W),
        "create_pull_request": OperationSpec(name="create_pull_request", params=["repo", "head", "title"],
                                             optional_params=["base", "body", "draft"], content_param="body",
                                             capability="create_pull_request", description="POST /repos/{o}/{r}/pulls",
                                             **_W),
        "dispatch_workflow": OperationSpec(name="dispatch_workflow", tier=Tier.REVERSIBLE, mutates=True,
                                           params=["repo", "workflow", "ref"], optional_params=["inputs"],
                                           capability="run_workflow", reversible=False,
                                           description="POST /repos/{o}/{r}/actions/workflows/{id}/dispatches"),
    }
    outcome_profile = {"create_pull_request": "code_review", "commit_file": "code_run", "dispatch_workflow": "code_run"}

    def __init__(self, ctx):
        super().__init__(ctx)
        a = ctx.account or {}
        self.app_id = a.get("app_id")
        self.installation_id = a.get("installation_id")
        self.repositories = [str(r) for r in a.get("repositories", [])]
        self.branch_prefix = a.get("branch_prefix", "agentrs/")
        self._inst_token: Optional[str] = None
        self._inst_exp = 0.0
        self._inst_perms: dict[str, str] = {}

    # ------------------------------------------------------------------ auth
    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION, "User-Agent": "agentrs-harness"}

    def app_jwt(self) -> str:
        if not self.app_id:
            raise ConnectorError("github: account has no app_id", 400)
        now = int(time.time())
        return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": str(self.app_id)},
                          self._secret("app_private_key"), algorithm="RS256")

    async def _mint(self) -> dict[str, Any]:
        if not self.installation_id:
            raise ConnectorError("github: account has no installation_id", 400)
        body: dict[str, Any] = {}
        if self.repositories:
            body["repositories"] = sorted({r.split("/", 1)[1] for r in self.repositories})
        if (self.ctx.account or {}).get("permissions"):
            body["permissions"] = dict(self.ctx.account["permissions"])
        r = await self._call("POST", f"/app/installations/{self.installation_id}/access_tokens",
                             headers=self._headers(self.app_jwt()), json=body)
        if r.status_code >= 400:
            raise self._gh_error("POST", "/app/installations/{id}/access_tokens", r)
        d = r.json()
        self._inst_token = d["token"]
        self._inst_exp = parse_time(d.get("expires_at")) or (time.time() + 3000)
        self._inst_perms = d.get("permissions", {})
        return d

    async def _token(self) -> str:
        if not self._inst_token or self._inst_exp - time.time() < 300:
            await self._mint()
        return str(self._inst_token)

    def _gh_error(self, method: str, path: str, r) -> ProviderHTTPError:
        e = self._error(method, path, r)
        if r.status_code in (403, 429) and r.headers.get("x-ratelimit-remaining") == "0":
            reset = r.headers.get("x-ratelimit-reset")
            retry = max(0.0, float(reset) - time.time()) if reset and reset.isdigit() else e.retry_after
            return ProviderHTTPError(f"github: rate limit exhausted ({e})", 429, e.body, "rate_limited", retry)
        if isinstance(e.body, dict) and e.body.get("message"):
            e.code = str(e.body["message"])
        return e

    async def _gh(self, method: str, path: str, **kw) -> Any:
        for attempt in range(2):
            r = await self._call(method, path, headers=self._headers(await self._token()), **kw)
            if r.status_code == 401 and attempt == 0:
                self._inst_token = None  # expired / revoked: mint a new one once
                continue
            if r.status_code >= 400:
                raise self._gh_error(method, path, r)
            return r.json() if r.content else {}
        raise ConnectorError("github: authentication failed", 401)

    # ------------------------------------------------------------------ guards
    def _repo(self, repo: str) -> str:
        if repo.count("/") != 1:
            raise ConnectorError(f"github: repo must be 'owner/name', got {repo!r}", 400)
        if repo not in self.repositories:
            raise ConnectorError(f"github: repository {repo} is not authorised for this account", 403)
        return repo

    def _branch_ok(self, branch: str) -> Optional[str]:
        if not branch or branch.startswith("refs/"):
            return "branch must be a plain branch name"
        if self.branch_prefix and not branch.startswith(self.branch_prefix):
            return f"agents may only write to branches starting with {self.branch_prefix!r}"
        return None

    async def _guard_write_branch(self, repo: str, branch: str) -> None:
        why = self._branch_ok(branch)
        if why:
            raise ConnectorError(f"github: {why}", 403)
        info = await self._gh("GET", f"/repos/{repo}")
        if branch == info.get("default_branch"):
            raise ConnectorError("github: writing to the default branch is refused (open a pull request)", 403)

    def _extra_policy(self, action: ActionRequest, entries: list[dict]) -> tuple[bool, str]:
        p = action.params
        repo = str(p.get("repo", ""))
        if repo not in self.repositories:
            return False, f"repository {repo!r} is not in this account's authorised repositories"
        branch = {"create_branch": p.get("branch"), "commit_file": p.get("branch"),
                  "create_pull_request": p.get("head"), "dispatch_workflow": p.get("ref")}.get(action.operation)
        if branch is not None:
            why = self._branch_ok(str(branch))
            if why:
                return False, why
            if str(branch) in ("main", "master", "trunk", "develop") or str(branch) == p.get("base"):
                return False, "writes to default / protected branches are refused"
        if action.operation == "commit_file" and str(p.get("path", "")).lstrip("/").startswith(".github/"):
            return False, "agents may not modify .github/ (workflows, CODEOWNERS, repository settings)"
        return True, ""

    # ------------------------------------------------------------------ interface
    async def validate_credentials(self) -> dict[str, Any]:
        r = await self._call("GET", "/app", headers=self._headers(self.app_jwt()))
        if r.status_code >= 400:
            raise self._gh_error("GET", "/app", r)
        app = r.json()
        tok = await self._mint()
        repos = await self._gh("GET", "/installation/repositories")
        names = [x["full_name"] for x in repos.get("repositories", [])]
        return {"ok": True, "app_slug": app.get("slug"), "app_id": app.get("id"), "installation_id": self.installation_id,
                "permissions": tok.get("permissions", {}), "token_expires_at": tok.get("expires_at"),
                "repositories": names, "missing_repositories": sorted(set(self.repositories) - set(names))}

    async def op_list_repositories(self) -> dict[str, Any]:
        r = await self._gh("GET", "/installation/repositories")
        return {"repositories": [{"full_name": x["full_name"], "private": x.get("private"),
                                  "default_branch": x.get("default_branch")} for x in r.get("repositories", [])]}

    async def op_read_repository(self, repo: str) -> dict[str, Any]:
        return {"repository": await self._gh("GET", f"/repos/{self._repo(repo)}")}

    async def op_read_file(self, repo: str, path: str, ref: Optional[str] = None) -> dict[str, Any]:
        params = {"ref": ref} if ref else {}
        d = await self._gh("GET", f"/repos/{self._repo(repo)}/contents/{quote(path.strip('/'))}", params=params)
        if isinstance(d, list):
            return {"type": "dir", "path": path, "entries": [{"name": e["name"], "type": e["type"], "path": e["path"]}
                                                             for e in d]}
        text = base64.b64decode(d.get("content", "")).decode("utf-8", errors="replace")
        return {"type": "file", "path": d["path"], "sha": d["sha"], "size": d.get("size"), "text": text, "ref": ref}

    async def op_read_pull_request(self, repo: str, number: int) -> dict[str, Any]:
        return {"pull_request": await self._gh("GET", f"/repos/{self._repo(repo)}/pulls/{int(number)}")}

    async def op_list_issues(self, repo: str, state: str = "open", per_page: int = 30, page: int = 1) -> dict[str, Any]:
        items = await self._gh("GET", f"/repos/{self._repo(repo)}/issues",
                               params={"state": state, "per_page": int(per_page), "page": int(page)})
        return {"issues": items}

    async def op_read_issue(self, repo: str, number: int) -> dict[str, Any]:
        return {"issue": await self._gh("GET", f"/repos/{self._repo(repo)}/issues/{int(number)}")}

    async def op_read_checks(self, repo: str, ref: str) -> dict[str, Any]:
        return await self._gh("GET", f"/repos/{self._repo(repo)}/commits/{quote(ref)}/check-runs")

    async def op_create_branch(self, repo: str, branch: str, from_ref: Optional[str] = None) -> dict[str, Any]:
        repo = self._repo(repo)
        await self._guard_write_branch(repo, branch)
        if from_ref is None:
            from_ref = (await self._gh("GET", f"/repos/{repo}"))["default_branch"]
        base = await self._gh("GET", f"/repos/{repo}/git/ref/heads/{quote(from_ref)}")
        sha = base["object"]["sha"]
        r = await self._gh("POST", f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": sha})
        return {"object_id": f"{repo}@{branch}", "ref": r["ref"], "sha": r["object"]["sha"], "from_ref": from_ref}

    async def op_commit_file(self, repo: str, branch: str, path: str, content: str, message: str,
                             sha: Optional[str] = None) -> dict[str, Any]:
        repo = self._repo(repo)
        path = path.strip("/")
        if path.startswith(".github/"):
            raise ConnectorError("github: agents may not modify .github/", 403)
        await self._guard_write_branch(repo, branch)
        if sha is None:
            try:
                cur = await self._gh("GET", f"/repos/{repo}/contents/{quote(path)}", params={"ref": branch})
                sha = cur.get("sha") if isinstance(cur, dict) else None
            except ProviderHTTPError as e:
                if e.status != 404:
                    raise
        body: dict[str, Any] = {"message": message, "content": base64.b64encode(content.encode()).decode(),
                                "branch": branch}
        if sha:
            body["sha"] = sha
        r = await self._gh("PUT", f"/repos/{repo}/contents/{quote(path)}", json=body)
        c = r["commit"]
        return {"object_id": f"{repo}@{c['sha']}", "commit_sha": c["sha"], "path": path, "branch": branch,
                "content_sha": r["content"]["sha"], "created": sha is None}

    async def op_create_pull_request(self, repo: str, head: str, title: str, base: Optional[str] = None,
                                     body: str = "", draft: bool = False) -> dict[str, Any]:
        repo = self._repo(repo)
        why = self._branch_ok(head)
        if why:
            raise ConnectorError(f"github: {why}", 403)
        if base is None:
            base = (await self._gh("GET", f"/repos/{repo}"))["default_branch"]
        pr = await self._gh("POST", f"/repos/{repo}/pulls",
                            json={"title": title, "head": head, "base": base, "body": body, "draft": bool(draft)})
        return {"object_id": f"{repo}#{pr['number']}", "number": pr["number"], "html_url": pr.get("html_url"),
                "head_sha": (pr.get("head") or {}).get("sha"), "state": pr.get("state")}

    async def op_dispatch_workflow(self, repo: str, workflow: str, ref: str,
                                   inputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        repo = self._repo(repo)
        why = self._branch_ok(ref)
        if why:
            raise ConnectorError(f"github: workflows may only be dispatched on agent branches ({why})", 403)
        await self._gh("POST", f"/repos/{repo}/actions/workflows/{quote(str(workflow))}/dispatches",
                       json={"ref": ref, "inputs": inputs or {}})
        return {"object_id": f"{repo}@{ref}", "dispatched": True, "workflow": workflow, "ref": ref}

    # ------------------------------------------------------------------ outcomes
    async def _ci(self, repo: str, ref: str) -> tuple[dict, dict, dict, dict]:
        try:
            runs = (await self._gh("GET", f"/repos/{repo}/commits/{quote(ref)}/check-runs")).get("check_runs", [])
        except ProviderHTTPError as e:
            if e.status in (403, 404, 422):
                return {}, {}, {}, {"checks_error": e.status}
            raise
        completed = bool(runs) and all(r.get("status") == "completed" for r in runs)
        passed = completed and all(r.get("conclusion") in OK_CONCLUSIONS for r in runs)
        t_done = max((parse_time(r.get("completed_at")) or 0 for r in runs), default=None) if completed else None
        counts = {"check_runs": len(runs), "check_failures": sum(1 for r in runs if r.get("status") == "completed"
                                                                 and r.get("conclusion") not in OK_CONCLUSIONS)}
        occurred = {"ci_completed": completed, "ci_passed": passed}
        at = {"ci_completed": t_done, "ci_passed": t_done if passed else None}
        raw = {"check_runs": [{"name": r.get("name"), "status": r.get("status"), "conclusion": r.get("conclusion")}
                              for r in runs]}
        return counts, occurred, at, raw

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        if "#" in object_ref:
            repo, num = object_ref.split("#", 1)
            repo = self._repo(repo)
            try:
                pr = await self._gh("GET", f"/repos/{repo}/pulls/{int(num)}")
            except ProviderHTTPError as e:
                if e.status == 404:
                    return outcome_payload({}, "removed", {"removed": True}, raw={"ref": object_ref})
                raise
            reviews = await self._gh("GET", f"/repos/{repo}/pulls/{int(num)}/reviews")
            counts, occurred, at, raw = await self._ci(repo, (pr.get("head") or {}).get("sha") or "")
            state = "merged" if pr.get("merged") else pr.get("state", "open")
            by_state: dict[str, list[float]] = {}
            for r in reviews:
                by_state.setdefault(r.get("state", ""), []).append(parse_time(r.get("submitted_at")) or 0.0)
            all_t = [t for ts in by_state.values() for t in ts]
            counts.update(reviews=len(reviews), approvals=len(by_state.get("APPROVED", [])),
                          changes_requested=len(by_state.get("CHANGES_REQUESTED", [])))
            occurred.update(reviewed=bool(reviews), approved=bool(by_state.get("APPROVED")),
                            changes_requested=bool(by_state.get("CHANGES_REQUESTED")), merged=state == "merged",
                            closed=state != "open", removed=False)
            at.update(reviewed=min(all_t) if all_t else None, approved=min(by_state.get("APPROVED", []), default=None),
                      changes_requested=min(by_state.get("CHANGES_REQUESTED", []), default=None),
                      merged=parse_time(pr.get("merged_at")), closed=parse_time(pr.get("closed_at")))
            raw.update(number=pr.get("number"), head_sha=(pr.get("head") or {}).get("sha"), html_url=pr.get("html_url"))
            return outcome_payload(counts, state, occurred, at, {}, raw)
        if "@" in object_ref:
            repo, ref = object_ref.split("@", 1)
            counts, occurred, at, raw = await self._ci(self._repo(repo), ref)
            return outcome_payload(counts, "visible", occurred, at, {}, {**raw, "ref": ref})
        raise ConnectorError(f"github: unknown object reference {object_ref!r} (use owner/repo#N or owner/repo@ref)", 400)

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="webhook", event_types=event_types or ["check_run", "pull_request", "pull_request_review"],
                            detail={"signature_header": "X-Hub-Signature-256",
                                    "verify": "agentrs.harness.webhooks.verify_github_signature",
                                    "setup": "a human sets the App's webhook URL and secret in the GitHub App settings",
                                    "fallback": "poll fetch_outcome(owner/repo#N)"})

    async def get_rate_limits(self) -> dict[str, Any]:
        out = await super().get_rate_limits()
        try:
            out["provider"] = (await self._gh("GET", "/rate_limit")).get("resources", {})
        except ConnectorError as e:
            out["provider_error"] = str(e)
        return out

    async def revoke(self) -> dict[str, Any]:
        if not self._inst_token:
            return {"revoked_at_provider": False, "reason": "no installation token is cached; the App private key and "
                                                            "installation are managed by a human in GitHub settings"}
        r = await self._call("DELETE", "/installation/token", headers=self._headers(self._inst_token))
        self._inst_token = None
        return {"revoked_at_provider": r.status_code == 204, "scope": "installation_token",
                "reason": "installation token revoked; to remove all access, a human uninstalls the App or rotates its "
                          "private key in GitHub settings"}
