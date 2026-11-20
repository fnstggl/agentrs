"""Code hosting and execution (host `code`).

Repositories hold content-addressed blobs and a linear commit history. Programs in a repository can be executed
in the sandbox (`code.run`, `code.test`) and deployed as services (`cloud.deploy`). A program run by an actor may
call back into the world as that actor through `crucible_sdk.call(tool, **args)` - this is how agents build their
own tools and automations. Execution consumes compute: a small free daily allowance per principal, then a charge
per CPU-second from the actor's primary account.

Dependencies: a file `crucible.json` may list `"deps": ["owner/repo"]`; their files are mounted read-only under
`deps/<repo>/` if the dependency is public and open-licensed, or the caller holds a `code.read` grant on it.
"""
from __future__ import annotations

import hashlib
import json
import re

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound, ServiceUnavailable
from ..core.kernel import Ctx, Module, applier, endpoint, register_module
from .identity import ident

NAME = r"[a-z][a-z0-9_-]{0,39}"
PATH = r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,119}"
MAX_FILE = 64 * 1024
MAX_REPO = 2 * 1024 * 1024
MAX_FILES = 200


def blob_id(content: str) -> str:
    return "b" + hashlib.blake2b(content.encode(), digest_size=12).hexdigest()


@register_module
class Code(Module):
    name = "code"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("repos", index=("owner",))
        st.declare_table("commits", index=("repo",))
        st.declare_table("blobs")
        st.declare_table("compute_use")
        world.hooks["code"] = self

    # ------------------------------------------------------------------ internal API
    def repo(self, state, rid: str):
        r = state.get("repos", rid)
        if r is None:
            raise NotFound(f"no repository {rid}")
        return r

    def can_read(self, ctx: Ctx, rid: str, actor: str | None = None) -> bool:
        r = self.repo(ctx.state, rid)
        actor = actor or ctx.actor
        if r["visibility"] == "public":
            return True
        return ident(ctx.world).allowed(ctx, actor, r["owner"], "code.read", scope=rid)

    def tree(self, state, rid: str, ref: str | None = None) -> dict[str, str]:
        r = self.repo(state, rid)
        cid = ref or r["head"]
        if cid is None:
            return {}
        c = state.get("commits", cid)
        if c is None or c["repo"] != rid:
            raise NotFound(f"no commit {ref} in {rid}")
        return dict(c["tree"])

    def files(self, state, rid: str, ref: str | None = None) -> dict[str, str]:
        return {p: state.tables["blobs"][b]["content"] for p, b in sorted(self.tree(state, rid, ref).items())}

    def files_with_deps(self, ctx: Ctx, rid: str, ref: str | None, actor: str) -> dict[str, str]:
        files = self.files(ctx.state, rid, ref)
        manifest = _manifest(files)
        for dep in manifest.get("deps", [])[:10]:
            dep_ref = None
            if "@" in dep:
                dep, dep_ref = dep.split("@", 1)
            r = ctx.state.get("repos", dep)
            if r is None:
                raise NotFound(f"dependency {dep} not found")
            if not ((r["visibility"] == "public" and r["license"] == "open") or
                    ident(ctx.world).allowed(ctx, actor, r["owner"], "code.read", scope=dep)):
                raise Forbidden(f"no licence to use {dep}")
            for p, c in self.files(ctx.state, dep, dep_ref).items():
                files[f"deps/{dep.split('/', 1)[1].replace('-', '_')}/{p}"] = c
        return files

    def commit_files(self, ctx: Ctx, rid: str, changes: dict[str, str | None], message: str, author: str) -> str:
        r = self.repo(ctx.state, rid)
        tree = self.tree(ctx.state, rid)
        new_blobs = {}
        for path, content in sorted(changes.items()):
            if not re.fullmatch(PATH, path) or ".." in path.split("/"):
                raise InvalidAction(f"bad path {path!r}")
            if content is None:
                tree.pop(path, None)
                continue
            if len(content.encode()) > MAX_FILE:
                raise InvalidAction(f"{path} exceeds {MAX_FILE} bytes")
            b = blob_id(content)
            tree[path] = b
            if not ctx.state.has("blobs", b):
                new_blobs[b] = content
        if len(tree) > MAX_FILES:
            raise InvalidAction(f"repository limit of {MAX_FILES} files")
        size = 0
        for path, b in tree.items():
            size += len((new_blobs.get(b) if b in new_blobs else ctx.state.tables["blobs"][b]["content"]).encode())
        if size > MAX_REPO:
            raise InvalidAction("repository size limit exceeded")
        cid = "c" + hashlib.blake2b(json.dumps([rid, r["head"], sorted(tree.items()), message, author, ctx.now]).encode(),
                                    digest_size=10).hexdigest()
        for b, content in sorted(new_blobs.items()):
            ctx.emit("code.blob", {"id": b, "content": content}, vis="internal")
        vis = "public" if r["visibility"] == "public" else (r["owner"], author)
        ctx.emit("code.commit", {"id": cid, "repo": rid, "parent": r["head"], "tree": tree, "message": message[:500],
                                 "author": author, "size": size}, vis=vis)
        return cid

    def create_repo(self, ctx: Ctx, owner: str, name: str, visibility: str = "private", description: str = "",
                    license: str = "closed") -> str:
        handle = owner.split(":", 1)[1]
        rid = f"{handle}/{name}"
        if ctx.state.has("repos", rid):
            raise Conflict(f"repository {rid} exists")
        ctx.emit("code.repo", {"id": rid, "owner": owner, "visibility": visibility, "description": description[:500],
                               "license": license}, vis="public" if visibility == "public" else (owner,))
        if visibility == "public":
            self._index(ctx, rid, description)
        return rid

    def _index(self, ctx, rid, description):
        search = ctx.world.modules.get("search")
        if search is not None:
            search.index(ctx, f"repo:{rid}", "repo", rid, description or rid, self.repo(ctx.state, rid)["owner"], ref=rid)

    # ------------------------------------------------------------------ compute accounting
    def charge_compute(self, ctx: Ctx, payer: str, cpu_ms: float, label: str) -> int:
        """Free daily allowance, then a per-CPU-second charge to the payer's primary account."""
        p = ctx.world.spec.params
        free = float(p.get("free_cpu_ms_per_day", 30_000))
        price = float(p.get("dev_compute_cents_per_cpu_s", 2.0))
        day = ctx.now // 86400
        use = ctx.state.get("compute_use", payer)
        used = use["ms"] if use and use["day"] == day else 0.0
        billable = max(0.0, used + cpu_ms - max(free, used))
        cents = int(round(billable / 1000.0 * price))
        ctx.emit("code.compute", {"principal": payer, "day": day, "ms": round(used + cpu_ms, 3)}, vis=(payer,))
        if cents > 0:
            bank = ctx.world.modules["bank"]
            acct = bank.primary_account(ctx.state, payer)
            if acct is None:
                raise Forbidden("compute beyond the free allowance needs a bank account")
            bank.move(ctx, acct, "acct_fees", cents, "compute", f"sandbox compute: {label}")
        return cents

    def sandbox_pool(self, world):
        pool = world.runtime.get("sandbox_pool")
        if pool is None:
            from ..sandbox.runner import SandboxConfig, WorkerPool, capabilities
            caps = capabilities()
            if caps["landlock_abi"] < 1 or not caps["seccomp"]:
                if not world.spec.params.get("allow_weak_sandbox"):
                    raise ServiceUnavailable("untrusted code execution is disabled: Landlock/seccomp unavailable "
                                             f"on this host ({caps}); see docs/SECURITY.md")
            p = world.spec.params
            cfg = SandboxConfig(mem_mb=int(p.get("sandbox_mem_mb", 1024)), job_wall_s=float(p.get("sandbox_job_s", 10)),
                                allow_weak=bool(p.get("allow_weak_sandbox", False)),
                                count_lines=p.get("cost_model") == "deterministic")
            pool = WorkerPool(int(p.get("sandbox_max_workers", 24)), cfg)
            world.runtime["sandbox_pool"] = pool
        return pool

    def run_program(self, ctx: Ctx, rid: str, ref: str | None, target: str, input, principal: str,
                    timeout: float | None = None, world_access: bool = True) -> dict:
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        if not self.can_read(ctx, rid, principal):
            raise Forbidden(f"no read access to {rid}")
        files = self.files_with_deps(ctx, rid, ref, principal)
        pool = self.sandbox_pool(ctx.world)

        def make():
            return Worker(files, pool.cfg, "program").start()
        out = run_in_ctx(ctx, "code.run", make, {"kind": "call", "target": target, "args": input,
                                                 "caller": principal}, principal, timeout=timeout, close_after=True,
                         world_access=world_access)
        return out

    def run_batch(self, ctx: Ctx, rid: str, ref: str | None, target: str, inputs: list, principal: str,
                  timeout: float | None = None, world_access: bool = False, warmup: bool = True) -> list[dict]:
        """Run `target` on several inputs in ONE sandbox worker (after an unrecorded-for-scoring warm-up call, so
        CPU budgets measure the algorithm rather than process start-up). Each run is recorded."""
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        if not self.can_read(ctx, rid, principal):
            raise Forbidden(f"no read access to {rid}")
        files = self.files_with_deps(ctx, rid, ref, principal)
        pool = self.sandbox_pool(ctx.world)
        holder: dict = {}

        def worker():
            if "w" not in holder:
                holder["w"] = Worker(files, pool.cfg, "batch").start()
            return holder["w"]
        outs = []
        try:
            seq = ([inputs[0]] if warmup and inputs else []) + list(inputs)
            for i, inp in enumerate(seq):
                o = run_in_ctx(ctx, "code.batch", worker, {"kind": "call", "target": target, "args": inp,
                                                           "caller": principal}, principal, timeout=timeout,
                               world_access=world_access)
                if not (warmup and i == 0):
                    outs.append(o)
        finally:
            if "w" in holder:
                holder["w"].close()
        return outs

    # ------------------------------------------------------------------ endpoints
    @endpoint("create_repo", S.obj({"name": S.s_str(pattern=NAME, maxLength=40),
                                    "owner": S.s_str("org handle (default: you)", maxLength=40),
                                    "visibility": S.s_enum(["private", "public"]),
                                    "license": S.s_enum(["closed", "open"], "open = anyone may depend on it"),
                                    "description": S.s_str(maxLength=500)}, ["name"]), cost_s=30)
    def ep_create_repo(self, ctx, name, owner=None, visibility="private", license="closed", description=""):
        owner = ident(ctx.world).resolve_handle(ctx.state, owner) if owner else ctx.actor
        if owner != ctx.actor:
            ident(ctx.world).authorize(ctx, owner, "code.write")
        if len(ctx.state.find_ids("repos", "owner", owner)) >= 50:
            raise Conflict("repository limit reached")
        return {"repo": self.create_repo(ctx, owner, name, visibility, description, license)}

    @endpoint("commit", S.obj({"repo": S.s_str(maxLength=81),
                               "files": {"type": "object", "additionalProperties": {"type": ["string", "null"],
                                                                                    "maxLength": MAX_FILE},
                                         "description": "path -> new content (null deletes the file)"},
                               "message": S.s_str(maxLength=500)}, ["repo", "files"]), cost_s=60)
    def ep_commit(self, ctx, repo, files, message=""):
        """Write files to a repository as a new commit (requires code.write)."""
        r = self.repo(ctx.state, repo)
        ident(ctx.world).authorize(ctx, r["owner"], "code.write", scope=repo)
        if not files:
            raise InvalidAction("no changes")
        cid = self.commit_files(ctx, repo, files, message, ctx.actor)
        return {"commit": cid, "files": len(self.tree(ctx.state, repo))}

    @endpoint("ls", S.obj({"repo": S.s_str(maxLength=81), "ref": S.s_str(maxLength=40)}, ["repo"]), read_only=True)
    def ep_ls(self, ctx, repo, ref=None):
        if not self.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        r = self.repo(ctx.state, repo)
        tree = self.tree(ctx.state, repo, ref)
        return {"repo": repo, "head": r["head"], "owner": r["owner"], "visibility": r["visibility"],
                "license": r["license"], "description": r["description"],
                "files": [{"path": p, "bytes": ctx.state.tables["blobs"][b]["size"]} for p, b in sorted(tree.items())]}

    @endpoint("read", S.obj({"repo": S.s_str(maxLength=81), "path": S.s_str(maxLength=120),
                             "ref": S.s_str(maxLength=40)}, ["repo", "path"]), read_only=True)
    def ep_read(self, ctx, repo, path, ref=None):
        if not self.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        tree = self.tree(ctx.state, repo, ref)
        if path not in tree:
            raise NotFound(f"{path} not in {repo}")
        return {"path": path, "content": ctx.state.tables["blobs"][tree[path]]["content"]}

    @endpoint("log", S.obj({"repo": S.s_str(maxLength=81), "limit": S.s_int(minimum=1, maximum=100)}, ["repo"]),
              read_only=True)
    def ep_log(self, ctx, repo, limit=20):
        if not self.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        out, cid = [], self.repo(ctx.state, repo)["head"]
        while cid and len(out) < limit:
            c = ctx.state.tables["commits"][cid]
            out.append({"commit": cid, "author": c["author"], "message": c["message"], "t": c["t"], "files": len(c["tree"])})
            cid = c["parent"]
        return out

    @endpoint("list", S.obj({"owner": S.s_str(maxLength=40)}), read_only=True)
    def ep_list(self, ctx, owner=None):
        """Repositories of an owner (public ones, plus private ones you can read). Default: yours."""
        owner = ident(ctx.world).resolve_handle(ctx.state, owner) if owner else ctx.actor
        out = []
        for rid in ctx.state.find_ids("repos", "owner", owner):
            if self.can_read(ctx, rid):
                r = ctx.state.tables["repos"][rid]
                out.append({"repo": rid, "visibility": r["visibility"], "license": r["license"],
                            "description": r["description"], "head": r["head"]})
        return out

    @endpoint("fork", S.obj({"repo": S.s_str(maxLength=81), "name": S.s_str(pattern=NAME, maxLength=40),
                             "owner": S.s_str(maxLength=40)}, ["repo", "name"]), cost_s=30)
    def ep_fork(self, ctx, repo, name, owner=None):
        """Copy a repository you may use (public+open, or licensed to you) into a new private repository."""
        r = self.repo(ctx.state, repo)
        owner = ident(ctx.world).resolve_handle(ctx.state, owner) if owner else ctx.actor
        if owner != ctx.actor:
            ident(ctx.world).authorize(ctx, owner, "code.write")
        if not ((r["visibility"] == "public" and r["license"] == "open") or
                ident(ctx.world).allowed(ctx, ctx.actor, r["owner"], "code.read", scope=repo)):
            raise Forbidden("no licence to copy this repository")
        rid = self.create_repo(ctx, owner, name, "private", f"fork of {repo}", "closed")
        files = self.files(ctx.state, repo)
        if files:
            self.commit_files(ctx, rid, files, f"fork of {repo}@{r['head']}", ctx.actor)
        return {"repo": rid}

    @endpoint("set_visibility", S.obj({"repo": S.s_str(maxLength=81), "visibility": S.s_enum(["private", "public"]),
                                       "license": S.s_enum(["closed", "open"])}, ["repo", "visibility"]))
    def ep_set_visibility(self, ctx, repo, visibility, license=None):
        r = self.repo(ctx.state, repo)
        ident(ctx.world).authorize(ctx, r["owner"], "org.admin" if r["owner"].startswith("o:") else "code.write")
        ctx.emit("code.visibility", {"id": repo, "visibility": visibility, "license": license or r["license"]},
                 vis="public")
        if visibility == "public":
            self._index(ctx, repo, r["description"])
        return {"ok": True}

    @endpoint("run", S.obj({"repo": S.s_str(maxLength=81),
                            "target": S.s_str("module:function, e.g. tools/price.py:main; called as f(input, caller)",
                                              maxLength=160),
                            "input": {"description": "JSON input passed to the function"},
                            "ref": S.s_str(maxLength=40), "timeout_s": S.s_num(minimum=0.1, maximum=30)},
                           ["repo", "target"]), cost_s=30)
    def ep_run(self, ctx, repo, target, input=None, ref=None, timeout_s=10.0):
        """Execute a function from a repository in the sandbox. The program can act in the world as you through
        `import crucible_sdk; crucible_sdk.call("tool.name", **args)`. Compute is charged after a free allowance."""
        out = self.run_program(ctx, repo, ref, target, input, ctx.actor, timeout=timeout_s)
        cents = self.charge_compute(ctx, ctx.actor, out["cpu_ms"], f"{repo}:{target}")
        return {"ok": out["ok"], "result": out["result"], "error": out["error"], "stdout": out["stdout"],
                "cpu_ms": out["cpu_ms"], "world_calls": len(out["rpcs"]), "charged_cents": cents}

    @endpoint("test", S.obj({"repo": S.s_str(maxLength=81), "ref": S.s_str(maxLength=40)}, ["repo"]), cost_s=60)
    def ep_test(self, ctx, repo, ref=None):
        """Run every test_* function in tests/test_*.py of the repository in the sandbox (no world access)."""
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        if not self.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        files = self.files_with_deps(ctx, repo, ref, ctx.actor)
        pool = self.sandbox_pool(ctx.world)
        out = run_in_ctx(ctx, "code.test", lambda: Worker(files, pool.cfg, "tests").start(),
                         {"kind": "tests", "pattern": "tests"}, ctx.actor, timeout=30, close_after=True)
        cents = self.charge_compute(ctx, ctx.actor, out["cpu_ms"], f"tests {repo}")
        return {"ok": out["ok"], "result": out["result"], "error": out["error"], "charged_cents": cents}

    # ------------------------------------------------------------------ appliers
    @applier("code.repo")
    def ap_repo(self, ev):
        p = ev.payload
        self.world.state.insert("repos", p["id"], {"owner": p["owner"], "visibility": p["visibility"],
                                                  "description": p["description"], "license": p["license"],
                                                  "head": None, "created": ev.t, "commits": 0})

    @applier("code.blob")
    def ap_blob(self, ev):
        if not self.world.state.has("blobs", ev.payload["id"]):
            c = ev.payload["content"]
            self.world.state.insert("blobs", ev.payload["id"], {"content": c, "size": len(c.encode())})

    @applier("code.commit")
    def ap_commit(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("commits", p["id"], {"repo": p["repo"], "parent": p["parent"], "tree": p["tree"],
                                       "message": p["message"], "author": p["author"], "t": ev.t, "size": p["size"]})
        r = st.tables["repos"][p["repo"]]
        st.patch("repos", p["repo"], head=p["id"], commits=r["commits"] + 1, size=p["size"])

    @applier("code.visibility")
    def ap_visibility(self, ev):
        self.world.state.patch("repos", ev.payload["id"], visibility=ev.payload["visibility"],
                               license=ev.payload["license"])

    @applier("code.compute")
    def ap_compute(self, ev):
        p = ev.payload
        self.world.state.put("compute_use", p["principal"], {"day": p["day"], "ms": p["ms"]})

    def check_invariants(self, world):
        st = world.state
        errs = []
        for rid, r in st.tables["repos"].items():
            if r["head"] is not None and r["head"] not in st.tables["commits"]:
                errs.append(f"{rid}: head missing")
        for cid, c in st.tables["commits"].items():
            for p, b in c["tree"].items():
                if b not in st.tables["blobs"]:
                    errs.append(f"{cid}: blob {b} missing")
        return errs

    def close(self):
        pool = self.world.runtime.pop("sandbox_pool", None) if self.world else None
        if pool is not None:
            pool.close()


def _manifest(files: dict[str, str]) -> dict:
    raw = files.get("crucible.json")
    if not raw:
        return {}
    try:
        m = json.loads(raw)
        return m if isinstance(m, dict) else {}
    except json.JSONDecodeError:
        return {}
