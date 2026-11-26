"""Software ecosystem (module `pkg`, host `pkg`; INTERFACE_CONTRACTS section 7, MASTER_SPEC Parts 6.5 and 9.2).

Packages extend 4.0 code hosting (`code`: repositories, commits, forks, `crucible.json` deps mounted under
`deps/<repo>/`) and the 4.0 issue tracker (`issues`):

* hosting and discovery: `pkg.register` binds a package name to a public, open-licensed repository and opens a public
  issue project for it; `pkg.search` / `pkg.info` find packages by their documentation and API (results carry
  verified-adoption counts; stars are shown separately and never count as adoption);
* releases: `pkg.release` pins a semantic version to a commit, runs the repository's tests in the sandbox, extracts
  its public API (function names and arities, by importing it in the sandbox), compares it with the previous
  release (breaking changes, semver conformance) and publishes its documentation (README.md);
* dependency graph: `pkg.depend` resolves a version constraint and writes the lock into the dependent repository's
  `crucible.json` (`deps` entry `owner/repo@commit`, `packages` constraints); `pkg.upgrade` re-resolves only if the
  dependent's own tests still pass with the new version (compatibility);
* adoption by execution only, measured by the parent (D-20): `pkg.exec` (one call) and `pkg.ci` (the dependent's
  tests) run the dependent's code in the sandbox with its locked dependencies, and then once more per dependency
  with that dependency *ablated* (its module replaced by a stub whose every attribute is an object that raises when
  used). A package is adopted by a dependent only when the dependent's successful outcome depends on it: the run
  with the package ablated fails or returns a different result. Both outcomes are observed by the parent; the
  in-sandbox call tracer's counts are recorded as self-reported and never decide adoption (they can be rewritten by
  the traced code). `pkg.deploy` records a dependent's deployment with its locked versions;
* collaboration: `pkg.pr_open` (a change proposed from another repository, tested in the sandbox), `pkg.pr_diff`
  (reading the change is recorded), `pkg.pr_review` (reviews count as independent only from principals unrelated to
  the author who read the diff), `pkg.pr_merge` (maintainer; needs an independent approval and passing tests);
  `pkg.report` files issues; `pkg.fork` forks with lineage;
* failures: `pkg.advisory` reports a defect with a reproduction repository. It is accepted as *reported* (visible to
  the reporter and the maintainer) only if the reproduction's tests fail against the affected release AND their
  outcome changes when the package is ablated (the reproduction exercises the package). It becomes *verified*
  (public, an issue is filed, dependents learn of it) only when a later release of the package makes the same
  reproduction pass: the defect is then demonstrably in the package and fixed in that release (`fixed_in`; the
  versions where it reproduced are `affected`). A reproduction that never passes on any release of the package
  (e.g. `assert False`, or tests of the reporter's own code) never verifies, so advisories cannot be used to brand
  a competitor's package.
* lineage: `forked_from` comes only from the code host's fork record (`code.fork`), never from a description.

All outcomes come from recorded sandbox executions (ctx.reserve/fill through `sandbox.bridge.run_in_ctx`).
"""
from __future__ import annotations

import json
import re

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, register_module
from ..services.identity import ident

NAME_RE = r"[a-z][a-z0-9_-]{1,39}"
VER_RE = r"(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})"
CONSTRAINT_RE = r"\*|(\^|~|>=|=)?" + VER_RE

TRACE_PY = r'''
import importlib, os, sys
_C = {}
def _prof(frame, event, arg):
    if event == "call":
        f = frame.f_code.co_filename
        i = f.find("/deps/")
        if i >= 0:
            n = f[i + 6:].split("/", 1)[0]
            _C[n] = _C.get(n, 0) + 1
def _fn(target):
    mod, _, fn = target.partition(":")
    mod = mod[:-3] if mod.endswith(".py") else mod
    return getattr(importlib.import_module(mod.replace("/", ".")), fn)
def call(args, caller):
    fn = _fn(args["target"])
    _C.clear()
    sys.setprofile(_prof)
    try:
        out = fn(args.get("input"), caller)
    finally:
        sys.setprofile(None)
    return {"result": out, "calls": dict(_C)}
def tests(args, caller):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    files = sorted(f for f in os.listdir(root) if f.startswith("test_") and f.endswith(".py")) if os.path.isdir(root) else []
    res = []
    _C.clear()
    for f in files:
        name = "tests." + f[:-3]
        try:
            m = importlib.import_module(name)
        except BaseException as e:
            res.append({"test": name, "ok": False, "error": ("import: %s: %s" % (type(e).__name__, e))[:300]})
            continue
        for t in sorted(dir(m)):
            if t.startswith("test_") and callable(getattr(m, t)):
                sys.setprofile(_prof)
                try:
                    getattr(m, t)()
                    res.append({"test": name + "." + t, "ok": True})
                except BaseException as e:
                    res.append({"test": name + "." + t, "ok": False, "error": ("%s: %s" % (type(e).__name__, e))[:300]})
                finally:
                    sys.setprofile(None)
    return {"passed": sum(r["ok"] for r in res), "failed": sum(not r["ok"] for r in res), "results": res[:50],
            "calls": dict(_C)}
def api(args, caller):
    import inspect
    m = importlib.import_module(args["module"])
    out = {}
    for name in sorted(dir(m)):
        if name.startswith("_"):
            continue
        obj = getattr(m, name)
        if inspect.isfunction(obj) and obj.__module__ == m.__name__:
            ps = [p for p in inspect.signature(obj).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            out[name] = [sum(1 for p in ps if p.default is p.empty), len(ps)]
    return out
'''


def parse_version(v: str) -> tuple[int, int, int]:
    m = re.fullmatch(VER_RE, v)
    if not m:
        raise InvalidAction(f"bad version {v!r} (MAJOR.MINOR.PATCH)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def satisfies(version: str, constraint: str) -> bool:
    if constraint == "*":
        return True
    v = parse_version(version)
    for op in (">=", "^", "~", "="):
        if constraint.startswith(op):
            c = parse_version(constraint[len(op):])
            break
    else:
        op, c = "=", parse_version(constraint)
    if op == "=":
        return v == c
    if op == ">=":
        return v >= c
    if op == "^":
        return v >= c and (v[0] == c[0] if c[0] > 0 else v[:2] == c[:2])
    return v >= c and v[:2] == c[:2]          # ~


def import_dir(repo: str) -> str:
    """Directory under which a dependency's files are mounted for dependents (`deps/<repo name>/`)."""
    return repo.split("/", 1)[1].replace("-", "_")


@register_module
class Packages(Module):
    name = "pkg"
    depends = ("identity", "bank", "code", "issues")

    def install(self, world):
        st = world.state
        st.declare_table("pkg_packages", index=("owner", "repo", "forked_from"))
        st.declare_table("pkg_releases", index=("package",))
        st.declare_table("pkg_deps", index=("repo", "package", "status"))
        st.declare_table("pkg_execs", index=("repo", "by", "kind"))
        st.declare_table("pkg_adoptions", index=("package", "repo"))
        st.declare_table("pkg_prs", index=("repo", "author", "status"))
        st.declare_table("pkg_reviews", index=("pr", "reviewer"))
        st.declare_table("pkg_reads", index=("pr",))
        st.declare_table("pkg_advisories", index=("package", "status"))
        st.declare_table("pkg_stars", index=("package",))
        st.declare_table("pkg_seen", index=("package", "by"))
        st.declare_table("pkg_deploys", index=("repo", "host"))
        st.declare_table("pkg_compat", index=("package", "repo"))
        world.hooks["pkg"] = self

    # ================================================================== helpers
    def package(self, st, name: str):
        p = st.get("pkg_packages", name)
        if p is None:
            raise NotFound(f"no package {name}")
        return p

    def releases(self, st, name: str, include_yanked: bool = False) -> list[dict]:
        rs = [dict(st.tables["pkg_releases"][r]) for r in st.find_ids("pkg_releases", "package", name)]
        rs = [r for r in rs if include_yanked or not r["yanked"]]
        return sorted(rs, key=lambda r: parse_version(r["version"]))

    def resolve(self, st, name: str, constraint: str) -> dict:
        ok = [r for r in self.releases(st, name) if satisfies(r["version"], constraint)]
        if not ok:
            raise NotFound(f"no release of {name} satisfies {constraint}")
        return ok[-1]

    def _maintainer(self, ctx, p):
        ident(ctx.world).authorize(ctx, p["owner"], "code.write", scope=p["repo"])

    def _repo_vis(self, ctx, r):
        """Facts about a dependent repository are public only if the repository is."""
        return "public" if r["visibility"] == "public" else tuple(sorted({r["owner"], ctx.actor}))

    def _independent(self, world, a: str, b: str) -> bool:
        from ..goals.metrics import is_related
        return not is_related(world, a, b)

    def _manifest(self, files: dict) -> dict:
        raw = files.get("crucible.json")
        if not raw:
            return {}
        try:
            m = json.loads(raw)
        except json.JSONDecodeError:
            raise InvalidAction("crucible.json is not valid JSON")
        return m if isinstance(m, dict) else {}

    def _run(self, ctx, files: dict, job: dict, principal: str, label: str, world_access=False, timeout=20.0) -> dict:
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        code = ctx.world.modules["code"]
        pool = code.sandbox_pool(ctx.world)
        files = dict(files)
        files["_pkgtrace.py"] = TRACE_PY
        return run_in_ctx(ctx, label, lambda: Worker(files, pool.cfg, label).start(), job, principal,
                          timeout=timeout, close_after=True, world_access=world_access)

    ABLATED = ("class _Ablated:\n"
               "    def __init__(self, n):\n        self._n = n\n"
               "    def __call__(self, *a, **k):\n        raise RuntimeError('dependency ablated: ' + self._n)\n"
               "    def __getattr__(self, n):\n        return _Ablated(self._n + '.' + n)\n"
               "    def __iter__(self):\n        raise RuntimeError('dependency ablated: ' + self._n)\n"
               "    def __bool__(self):\n        raise RuntimeError('dependency ablated: ' + self._n)\n"
               "def __getattr__(name):\n    return _Ablated(name)\n")

    def _ablate(self, st, files: dict, pkg_name: str) -> dict:
        """The same tree with one mounted dependency replaced by an ablation stub (parent-side adoption test)."""
        p = st.tables["pkg_packages"][pkg_name]
        d = f"deps/{import_dir(p['repo'])}/"
        out = {k: v for k, v in files.items() if not k.startswith(d)}
        out[f"{d}{p['module']}.py"] = self.ABLATED
        out[f"{d}__init__.py"] = ""
        return out

    def _exercised(self, ctx, repo: str, files: dict, job: dict, principal: str, base: dict, label: str) -> list[str]:
        """Packages whose ablation changes a successful outcome (both outcomes observed by the parent)."""
        from ..core.canonical import digest as _dg
        st = ctx.state
        out = []
        base_key = _dg(self._outcome(base))
        for _, (name, _) in sorted(self.closure(st, repo).items()):
            o2 = self._run(ctx, self._ablate(st, files, name), job, principal, label + ".ablate")
            if _dg(self._outcome(o2)) != base_key:
                out.append(name)
        return out

    @staticmethod
    def _outcome(o: dict):
        """What the parent compares: success and the returned value (for tests: which tests passed)."""
        if not o.get("ok"):
            return ["failed"]
        r = o.get("result")
        if isinstance(r, dict) and "results" in r and "passed" in r:
            # which tests passed, and how the failing ones failed (the exception type)
            return ["tests", sorted([x.get("test"), bool(x.get("ok")), str(x.get("error") or "").split(":", 1)[0]]
                                    for x in r["results"])]
        if isinstance(r, dict) and "result" in r:
            return ["ok", r["result"]]
        return ["ok", r]

    def _files_with(self, ctx, repo: str, ref: str | None, actor: str, swap: dict | None = None) -> dict:
        """Files of `repo` with its locked dependencies mounted; `swap` = {dep_repo: commit} replaces locks."""
        code = ctx.world.modules["code"]
        files = code.files(ctx.state, repo, ref)
        swap = dict(swap or {})
        out = dict(files)
        mounted: set[str] = set()
        # breadth-first over the locked dependencies of dependencies (a flat deps/ namespace, like site-packages);
        # the first lock seen for a repository wins, and a swapped version replaces the lock wherever it appears
        queue = [(dep, 1) for dep in list(self._manifest(files).get("deps", []))[:10]]
        queue += [(f"{r}@{c}", 1) for r, c in sorted(swap.items())]
        while queue:
            dep, depth = queue.pop(0)
            dep_repo, _, dep_ref = dep.partition("@")
            if dep_repo in mounted or dep_repo == repo or depth > 4:
                continue
            if dep_repo in swap:
                dep_ref = swap[dep_repo]
            r = ctx.state.get("repos", dep_repo)
            if r is None:
                raise NotFound(f"dependency {dep_repo} not found")
            if not ((r["visibility"] == "public" and r["license"] == "open") or
                    code._licensed(ctx, actor, r["owner"], dep_repo)):
                raise Forbidden(f"no licence to use {dep_repo}")
            dfiles = code.files(ctx.state, dep_repo, dep_ref or None)
            mount = f"deps/{import_dir(dep_repo)}/"
            # the mount point belongs to the package manager: a dependent's own files there are not mounted
            # (a shadow copy of a dependency cannot pass for the dependency)
            for k in [k for k in out if k.startswith(mount)]:
                del out[k]
            for p, c in dfiles.items():
                out[f"{mount}{p}"] = c
            mounted.add(dep_repo)
            queue += [(d, depth + 1) for d in list(self._manifest(dfiles).get("deps", []))[:10]]
        return out

    def closure(self, st, repo: str) -> dict[str, str]:
        """Packages whose code a repository runs with: its locked packages and, transitively, theirs.
        Returns {import dir: (package name, locked version)}."""
        out: dict[str, tuple] = {}
        todo = [repo]
        seen = set()
        while todo:
            r = todo.pop(0)
            if r in seen:
                continue
            seen.add(r)
            for name, d in sorted(self._locked(st, r).items()):
                p = st.tables["pkg_packages"][name]
                out.setdefault(import_dir(p["repo"]), (name, d["version"]))
                todo.append(p["repo"])
        return out

    def _locked(self, st, repo: str) -> dict[str, dict]:
        """package -> active dependency record of a dependent repo."""
        return {d["package"]: dict(d) for d in st.find("pkg_deps", "repo", repo) if d["status"] == "active"}

    def _calls_by_package(self, st, repo: str, calls: dict) -> dict[str, int]:
        out = {}
        for d, (name, _) in sorted(self.closure(st, repo).items()):
            n = int(calls.get(d, 0))
            if n:
                out[name] = n
        return out

    def _ci(self, ctx, files: dict, principal: str, label="pkg.ci") -> dict:
        out = self._run(ctx, files, {"kind": "call", "target": "_pkgtrace:tests", "args": {}, "caller": principal},
                        principal, label)
        res = out["result"] if out["ok"] and isinstance(out["result"], dict) else \
            {"passed": 0, "failed": 1, "results": [{"test": "runner", "ok": False,
                                                    "error": str((out.get("error") or {}).get("message", ""))[:300]}],
             "calls": {}}
        return {**res, "cpu_ms": out["cpu_ms"]}

    def _api(self, ctx, files: dict, module: str, principal: str) -> dict:
        out = self._run(ctx, files, {"kind": "call", "target": "_pkgtrace:api", "args": {"module": module},
                                     "caller": principal}, principal, "pkg.api")
        if not out["ok"] or not isinstance(out["result"], dict):
            raise InvalidAction(f"the package module {module} does not import: "
                                f"{(out.get('error') or {}).get('message', '')[:300]}")
        return out["result"]

    # ================================================================== hosting and discovery
    @endpoint("register", S.obj({"name": S.s_str(pattern=NAME_RE, maxLength=40),
                                 "repo": S.s_str("public, open-licensed repository", maxLength=81),
                                 "description": S.s_str(maxLength=1000),
                                 "module": S.s_str("python module of the package inside the repo (default lib)",
                                                   pattern=r"[a-z_][a-z0-9_]{0,40}", maxLength=41)},
                                ["name", "repo", "description"]), cost_s=30)
    def ep_register(self, ctx, name, repo, description, module="lib"):
        """Register a package name for a public, open-licensed repository you maintain; opens a public issue
        tracker for it. Release versions with pkg.release."""
        st = ctx.state
        code = ctx.world.modules["code"]
        r = code.repo(st, repo)
        ident(ctx.world).authorize(ctx, r["owner"], "code.write", scope=repo)
        if r["visibility"] != "public" or r["license"] != "open":
            raise InvalidAction("packages are published from public, open-licensed repositories")
        if st.has("pkg_packages", name):
            raise Conflict(f"package {name} exists")
        if st.find_ids("pkg_packages", "repo", repo):
            raise Conflict(f"{repo} is already a package")
        forked = None
        parent = code.fork_parent(st, repo) if hasattr(code, "fork_parent") else r.get("forked_from")
        if parent:
            for other_name, other in st.scan("pkg_packages"):
                if other["repo"] == parent:
                    forked = other_name
        owner_handle = r["owner"].split(":", 1)[1]
        proj = ctx.call("issues.create_project", {"name": f"pkg-{name}", "repo": repo, "public": True,
                                                  "description": f"Issues of package {name}",
                                                  **({"owner": owner_handle} if r["owner"] != ctx.actor else {})})
        ctx.emit("pkg.registered", {"id": name, "repo": repo, "owner": r["owner"], "description": description,
                                    "module": module, "project": proj["project"], "forked_from": forked, "by": ctx.actor},
                 vis="public")
        self._index(ctx, name, description, None)
        return {"package": name, "project": proj["project"], "import": f"deps.{import_dir(repo)}.{module}",
                "forked_from": forked}

    def _index(self, ctx, name, description, rel):
        from ..services.search import index
        st = ctx.state
        p = st.get("pkg_packages", name)
        api = " ".join(sorted((rel or {}).get("api") or {}))
        docs = (rel or {}).get("docs") or ""
        index(ctx, f"pkg:{name}", "package", f"{name} {rel['version'] if rel else ''}".strip(),
              f"{description}\nAPI: {api}\n{docs[:3000]}", p["owner"] if p else ctx.actor, ref=name)

    def summary(self, world, name: str) -> dict:
        st = world.state
        p = st.tables["pkg_packages"][name]
        rels = self.releases(st, name)
        latest = rels[-1] if rels else None
        adopters = [a for a in st.find("pkg_adoptions", "package", name) if a["owner"] != p["owner"]]
        issues = [i for i in st.find("iss_issues", "project", p["project"])] if "iss_issues" in st.tables else []
        advs = [a for a in st.find("pkg_advisories", "package", name) if a["verified"]]
        return {"package": name, "repo": p["repo"], "owner": p["owner"], "description": p["description"],
                "import": f"deps.{import_dir(p['repo'])}.{p['module']}", "latest": latest["version"] if latest else None,
                "releases": len(rels), "last_release_t": latest["t"] if latest else None,
                "tests_latest": latest["tests"] if latest else None,
                "api": latest["api"] if latest else {}, "docs": bool(latest and latest["docs"]),
                "verified_adopters": len(adopters), "dependents": len({d["repo"] for d in st.find("pkg_deps", "package", name)
                                                                     if d["status"] == "active"}),
                "open_issues": sum(1 for i in issues if i["status"] != "closed"),
                "open_advisories": sum(1 for a in advs if not a["fixed_in"]),
                "forked_from": p["forked_from"], "stars": p["stars"]}

    @endpoint("search", S.obj({"query": S.s_str("what you need: words, function names", maxLength=300),
                               "limit": S.s_int(minimum=1, maximum=30)}, ["query"]), read_only=True)
    def ep_search(self, ctx, query, limit=10):
        """Find packages by documentation and API. Each result carries verified adoption (dependents whose own
        code executed it), release, test, issue and advisory statistics; stars are listed but are not adoption."""
        st = ctx.state
        out = []
        search = ctx.world.modules.get("search")
        names = []
        q = query.strip().lower()
        if st.has("pkg_packages", q):
            names.append(q)
        if search is not None:
            sp, org, _ = search.search(ctx.world, query, "package", limit * 2)
            for r in sp + org:
                if r.get("ref") and st.has("pkg_packages", r["ref"]) and r["ref"] not in names:
                    names.append(r["ref"])
        for n in names[:limit]:
            out.append(self.summary(ctx.world, n))
        new = [n for n in names[:limit] if not st.has("pkg_seen", f"{n}|{ctx.actor}")]
        if new:
            ctx.emit("pkg.seen", {"by": ctx.actor, "packages": new}, vis=(ctx.actor,))
        return out

    @endpoint("info", S.obj({"package": S.s_str(maxLength=40)}, ["package"]), read_only=True)
    def ep_info(self, ctx, package):
        """A package: every release (version, commit, tests, API, compatibility with the previous release,
        documentation), verified adopters, advisories and fork lineage."""
        st = ctx.state
        self.package(st, package)
        rels = self.releases(st, package, include_yanked=True)
        advs = [dict(a) for a in st.find("pkg_advisories", "package", package) if a["verified"]]
        if not st.has("pkg_seen", f"{package}|{ctx.actor}"):
            ctx.emit("pkg.seen", {"by": ctx.actor, "packages": [package]}, vis=(ctx.actor,))
        return {**self.summary(ctx.world, package),
                "versions": [{k: r[k] for k in ("version", "commit", "tests", "api", "compat", "yanked", "t", "notes")}
                             | {"docs": r["docs"][:2000]} for r in rels],
                "advisories": [{k: a[k] for k in ("id", "kind", "severity", "affected", "fixed_in", "description", "t")}
                               for a in advs],
                "forks": sorted(st.find_ids("pkg_packages", "forked_from", package))}

    @endpoint("star", S.obj({"package": S.s_str(maxLength=40)}, ["package"]))
    def ep_star(self, ctx, package):
        """Star a package (a public bookmark; not evidence of adoption)."""
        self.package(ctx.state, package)
        if ctx.state.has("pkg_stars", f"{package}|{ctx.actor}"):
            raise Conflict("already starred")
        ctx.emit("pkg.star", {"package": package, "by": ctx.actor}, vis="public")
        return {"ok": True}

    # ================================================================== releases
    @endpoint("release", S.obj({"package": S.s_str(maxLength=40), "version": S.s_str(pattern=VER_RE, maxLength=14),
                                "notes": S.s_str(maxLength=2000)}, ["package", "version"]), cost_s=120)
    def ep_release(self, ctx, package, version, notes=""):
        """Release the repository head as a version: its tests run in the sandbox, its public API is extracted and
        compared with the previous release (breaking changes, semver), its README.md becomes the documentation.
        Open advisories are re-checked against the new release."""
        st = ctx.state
        p = self.package(st, package)
        self._maintainer(ctx, p)
        code = ctx.world.modules["code"]
        repo = code.repo(st, p["repo"])
        if repo["head"] is None:
            raise InvalidAction("the repository has no commits")
        prev = self.releases(st, package, include_yanked=True)
        v = parse_version(version)
        if prev and v <= parse_version(prev[-1]["version"]):
            raise Conflict(f"version must be greater than {prev[-1]['version']}")
        files = self._files_with(ctx, p["repo"], repo["head"], ctx.actor)
        tests = self._ci(ctx, files, ctx.actor, "pkg.release_tests")
        api = self._api(ctx, files, p["module"], ctx.actor)
        compat = {"previous": None, "removed": [], "changed": [], "added": sorted(api), "breaking": False,
                  "semver_ok": True}
        live_prev = [r for r in prev if not r["yanked"]]
        if live_prev:
            pa = live_prev[-1]["api"]
            removed = sorted(k for k in pa if k not in api)
            changed = sorted(k for k in pa if k in api and (api[k][0] > pa[k][0] or api[k][1] < pa[k][0]))
            breaking = bool(removed or changed)
            pv = parse_version(live_prev[-1]["version"])
            compat = {"previous": live_prev[-1]["version"], "removed": removed, "changed": changed,
                      "added": sorted(k for k in api if k not in pa), "breaking": breaking,
                      "semver_ok": (not breaking) or v[0] > pv[0] or (pv[0] == 0 and v[1] > pv[1])}
        docs = files.get("README.md", "") or "\n".join(c for f, c in sorted(files.items())
                                                       if f.startswith("docs/") and f.endswith(".md"))
        rid = f"{package}@{version}"
        ctx.emit("pkg.released", {"id": rid, "package": package, "version": version, "commit": repo["head"],
                                  "by": ctx.actor, "notes": notes[:2000],
                                  "tests": {"passed": tests["passed"], "failed": tests["failed"]},
                                  "api": api, "compat": compat, "docs": docs[:8000]}, vis="public")
        self._index(ctx, package, p["description"], {"version": version, "api": api, "docs": docs})
        fixed = []
        for aid in st.find_ids("pkg_advisories", "package", package):
            a = st.tables["pkg_advisories"][aid]
            res = self._repro(ctx, a["repro_repo"], a["repro_commit"], p["repo"], repo["head"], a["by"])
            fixes = res["failed"] == 0 and res["passed"] > 0
            if a["verified"] and not res["failed"]:
                continue                                    # still fixed
            ctx.emit("pkg.advisory_check", {"id": aid, "version": version, "reproduces": res["failed"] > 0,
                                            "tests": {"passed": res["passed"], "failed": res["failed"]}},
                     vis="public" if (a["verified"] or fixes) else (a["by"], p["owner"]))
            if fixes and not a["verified"]:
                fixed.append(aid)
                self._publish_advisory(ctx, aid)
        return {"release": rid, "commit": repo["head"], "tests": {"passed": tests["passed"], "failed": tests["failed"]},
                "api": api, "compat": compat, "fixes_advisories": fixed}

    @endpoint("yank", S.obj({"package": S.s_str(maxLength=40), "version": S.s_str(pattern=VER_RE, maxLength=14),
                             "reason": S.s_str(maxLength=300)}, ["package", "version"]))
    def ep_yank(self, ctx, package, version, reason=""):
        """Withdraw a release (it is no longer resolved for new dependents; existing locks keep working)."""
        st = ctx.state
        p = self.package(st, package)
        self._maintainer(ctx, p)
        rid = f"{package}@{version}"
        if not st.has("pkg_releases", rid):
            raise NotFound("no release " + rid)
        ctx.emit("pkg.yanked", {"id": rid, "reason": reason[:300]}, vis="public")
        return {"ok": True}

    # ================================================================== dependencies
    def _write_lock(self, ctx, repo, package, constraint, rel, message) -> str:
        st = ctx.state
        code = ctx.world.modules["code"]
        files = code.files(st, repo, None)
        m = self._manifest(files)
        p = self.package(st, package)
        # a flat lock: the package and, transitively, what its release itself locks (so that code hosting and
        # cloud deployments mount every dependency the code imports)
        lock: dict[str, str] = {}
        todo = [(p["repo"], rel["commit"])]
        while todo and len(lock) <= 10:
            r, c = todo.pop(0)
            if r in lock or r == repo:
                continue
            lock[r] = c
            for d in self._manifest(code.files(st, r, c)).get("deps", [])[:10]:
                dr, _, dc = d.partition("@")
                todo.append((dr, dc or None))
        deps = [d for d in m.get("deps", []) if d.partition("@")[0] not in lock]
        deps += [f"{r}@{c}" if c else r for r, c in lock.items()]
        if len(deps) > 10:
            raise InvalidAction("at most 10 dependencies (including transitive ones)")
        pk = dict(m.get("packages") or {})
        pk[package] = {"constraint": constraint, "version": rel["version"]}
        m["deps"], m["packages"] = deps, dict(sorted(pk.items()))
        return code.commit_files(ctx, repo, {"crucible.json": json.dumps(m, indent=1, sort_keys=True)}, message,
                                 ctx.actor)

    @endpoint("depend", S.obj({"repo": S.s_str("your repository", maxLength=81), "package": S.s_str(maxLength=40),
                               "constraint": S.s_str("^1.2.0, ~1.2.0, =1.2.3, >=1.0.0 or *", pattern=CONSTRAINT_RE,
                                                     maxLength=16)}, ["repo", "package"]), cost_s=30)
    def ep_depend(self, ctx, repo, package, constraint="*"):
        """Add (or change) a dependency of your repository on a package: the newest release satisfying the
        constraint is locked into crucible.json and its files are mounted at deps/<repo>/ for your code. The
        package counts as adopted by your repository only once your code executes it (pkg.exec, pkg.ci)."""
        st = ctx.state
        code = ctx.world.modules["code"]
        r = code.repo(st, repo)
        ident(ctx.world).authorize(ctx, r["owner"], "code.write", scope=repo)
        p = self.package(st, package)
        if p["repo"] == repo:
            raise InvalidAction("a package cannot depend on itself")
        rel = self.resolve(st, package, constraint)
        cid = self._write_lock(ctx, repo, package, constraint, rel, f"depend on {package} {constraint} "
                                                                   f"(locked {rel['version']})")
        ctx.emit("pkg.dependency", {"id": f"{repo}|{package}", "repo": repo, "package": package,
                                    "constraint": constraint, "version": rel["version"], "commit": rel["commit"],
                                    "owner": r["owner"], "by": ctx.actor, "lock_commit": cid}, vis=self._repo_vis(ctx, r))
        return {"locked": rel["version"], "commit": cid,
                "import": f"deps.{import_dir(p['repo'])}.{p['module']}"}

    @endpoint("upgrade", S.obj({"repo": S.s_str(maxLength=81), "package": S.s_str(maxLength=40),
                                "constraint": S.s_str(pattern=CONSTRAINT_RE, maxLength=16)}, ["repo", "package"]),
              cost_s=60)
    def ep_upgrade(self, ctx, repo, package, constraint=None):
        """Move a dependency to the newest release satisfying its constraint (or a new one) - only if your
        repository's own tests pass with it in the sandbox (otherwise the incompatibility is recorded)."""
        st = ctx.state
        code = ctx.world.modules["code"]
        r = code.repo(st, repo)
        ident(ctx.world).authorize(ctx, r["owner"], "code.write", scope=repo)
        cur = self._locked(st, repo).get(package)
        if cur is None:
            raise NotFound(f"{repo} does not depend on {package}")
        constraint = constraint or cur["constraint"]
        rel = self.resolve(st, package, constraint)
        if rel["version"] == cur["version"]:
            return {"upgraded": False, "version": rel["version"], "reason": "already newest"}
        p = self.package(st, package)
        files = self._files_with(ctx, repo, None, ctx.actor, swap={p["repo"]: rel["commit"]})
        res = self._ci(ctx, files, ctx.actor)
        ok = res["failed"] == 0 and res["passed"] > 0
        ctx.emit("pkg.compat", {"id": f"{repo}|{package}|{rel['version']}", "repo": repo, "package": package,
                                "from": cur["version"], "to": rel["version"], "compatible": ok,
                                "tests": {"passed": res["passed"], "failed": res["failed"]}, "by": ctx.actor},
                 vis=self._repo_vis(ctx, r))
        if not ok:
            return {"upgraded": False, "version": cur["version"], "reason": "tests fail with " + rel["version"],
                    "tests": {"passed": res["passed"], "failed": res["failed"]}}
        cid = self._write_lock(ctx, repo, package, constraint, rel, f"upgrade {package} to {rel['version']}")
        ctx.emit("pkg.dependency", {"id": f"{repo}|{package}", "repo": repo, "package": package,
                                    "constraint": constraint, "version": rel["version"], "commit": rel["commit"],
                                    "owner": r["owner"], "by": ctx.actor, "lock_commit": cid}, vis=self._repo_vis(ctx, r))
        return {"upgraded": True, "version": rel["version"], "commit": cid}

    # ================================================================== execution (adoption evidence)
    def _record_exec(self, ctx, repo, commit, kind, ok, calls, extra=None, exercised=()):
        st = ctx.state
        r = st.tables["repos"][repo]
        by_pkg = self._calls_by_package(st, repo, calls or {})
        locked = {name: ver for name, ver in self.closure(st, repo).values()}
        eid = st.new_id("pexec")
        ctx.emit("pkg.executed", {"id": eid, "repo": repo, "commit": commit, "owner": r["owner"], "by": ctx.actor,
                                  "kind": kind, "ok": bool(ok), "self_reported_calls": by_pkg, "locked": locked,
                                  "exercised": sorted(exercised) if ok else [], **(extra or {})},
                 vis=tuple(sorted({r["owner"], ctx.actor})))
        return eid, by_pkg

    @endpoint("exec", S.obj({"repo": S.s_str(maxLength=81),
                             "target": S.s_str("module:function in your repo, called as f(input, caller)",
                                               maxLength=160),
                             "input": {"description": "JSON input"}, "ref": S.s_str(maxLength=40)},
                            ["repo", "target"]), cost_s=30)
    def ep_exec(self, ctx, repo, target, input=None, ref=None):
        """Run a function of a repository you can read in the sandbox with its locked dependencies, tracing how
        many calls entered each dependency's code (recorded; this is how adoption by execution is measured)."""
        st = ctx.state
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        r = code.repo(st, repo)
        commit = ref or r["head"]
        files = self._files_with(ctx, repo, commit, ctx.actor)
        job = {"kind": "call", "target": "_pkgtrace:call", "args": {"target": target, "input": input},
               "caller": ctx.actor}
        out = self._run(ctx, files, job, ctx.actor, "pkg.exec", world_access=True)
        code.charge_compute(ctx, ctx.actor, out["cpu_ms"], f"pkg.exec {repo}")
        res = out["result"] if out["ok"] and isinstance(out["result"], dict) else {}
        ex = self._exercised(ctx, repo, files, job, ctx.actor, out, "pkg.exec") if out["ok"] else []
        eid, by_pkg = self._record_exec(ctx, repo, commit, "exec", out["ok"], res.get("calls"), exercised=ex)
        return {"execution": eid, "ok": out["ok"], "result": res.get("result"), "error": out["error"],
                "exercised": ex, "cpu_ms": out["cpu_ms"]}

    @endpoint("ci", S.obj({"repo": S.s_str(maxLength=81), "ref": S.s_str(maxLength=40)}, ["repo"]), cost_s=60)
    def ep_ci(self, ctx, repo, ref=None):
        """Run the tests of a repository (tests/test_*.py) in the sandbox with its locked dependencies, tracing
        calls into each dependency (recorded)."""
        st = ctx.state
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        r = code.repo(st, repo)
        commit = ref or r["head"]
        files = self._files_with(ctx, repo, commit, ctx.actor)
        res = self._ci(ctx, files, ctx.actor)
        code.charge_compute(ctx, ctx.actor, res["cpu_ms"], f"pkg.ci {repo}")
        ok = res["failed"] == 0 and res["passed"] > 0
        ex = self._exercised(ctx, repo, files, {"kind": "call", "target": "_pkgtrace:tests", "args": {},
                                                "caller": ctx.actor}, ctx.actor,
                             {"ok": True, "result": {"results": res["results"], "passed": res["passed"]}},
                             "pkg.ci") if ok else []
        eid, by_pkg = self._record_exec(ctx, repo, commit, "ci", ok, res.get("calls"),
                                        {"tests": {"passed": res["passed"], "failed": res["failed"]}}, exercised=ex)
        return {"execution": eid, "ok": ok, "passed": res["passed"], "failed": res["failed"],
                "results": res["results"][:20], "exercised": ex}

    @endpoint("deploy", S.obj({"repo": S.s_str(maxLength=81), "host": S.s_str(maxLength=31),
                               "instance": S.s_str(maxLength=40)}, ["repo", "host", "instance"]), cost_s=120)
    def ep_deploy(self, ctx, repo, host, instance):
        """Deploy a repository as a service (cloud.deploy) and record the package versions it runs with."""
        st = ctx.state
        res = ctx.call("cloud.deploy", {"repo": repo, "host": host, "instance": instance})
        r = st.tables["repos"][repo]
        locked = {name: ver for name, ver in self.closure(st, repo).values()}
        ctx.emit("pkg.deployed", {"id": res["deployment"], "repo": repo, "host": host, "commit": r["head"],
                                  "status": res["status"], "packages": locked, "owner": r["owner"]},
                 vis=tuple(sorted({r["owner"], ctx.actor})))
        return {**res, "packages": locked}

    # ================================================================== collaboration
    @endpoint("pr_open", S.obj({"repo": S.s_str("target repository", maxLength=81),
                                "source_repo": S.s_str("repository holding the proposed tree (e.g. your fork)",
                                                       maxLength=81),
                                "title": S.s_str(minLength=1, maxLength=200), "body": S.s_str(maxLength=5000)},
                               ["repo", "source_repo", "title"]), cost_s=60)
    def ep_pr_open(self, ctx, repo, source_repo, title, body=""):
        """Propose the head of `source_repo` as the next state of `repo` (a pull request). The proposed tree is
        tested in the sandbox now; maintainers merge with pkg.pr_merge after an independent review."""
        st = ctx.state
        code = ctx.world.modules["code"]
        tgt = code.repo(st, repo)
        src = code.repo(st, source_repo)
        if not code.can_read(ctx, repo) or not code.can_read(ctx, source_repo):
            raise Forbidden("you must be able to read both repositories")
        ident(ctx.world).authorize(ctx, src["owner"], "code.write", scope=source_repo)
        base = code.files(st, repo, None)
        new = code.files(st, source_repo, None)
        changes = {p: c for p, c in sorted(new.items()) if base.get(p) != c}
        changes.update({p: None for p in sorted(base) if p not in new})
        if not changes:
            raise InvalidAction("no changes")
        res = self._ci(ctx, self._files_with(ctx, source_repo, None, ctx.actor), ctx.actor, "pkg.pr_tests")
        pid = st.new_id("pr")
        ctx.emit("pkg.pr", {"id": pid, "repo": repo, "source_repo": source_repo, "source_commit": src["head"],
                            "base_commit": tgt["head"], "author": ctx.actor, "title": title, "body": body[:5000],
                            "files": sorted(changes), "changes_digest": digest(changes),
                            "tests": {"passed": res["passed"], "failed": res["failed"]}}, vis=self._repo_vis(ctx, tgt))
        return {"pr": pid, "files": sorted(changes), "tests": {"passed": res["passed"], "failed": res["failed"]}}

    def _pr_changes(self, st, pr) -> dict:
        code = self.world.modules["code"]
        base = code.files(st, pr["repo"], pr["base_commit"])
        new = code.files(st, pr["source_repo"], pr["source_commit"])
        ch = {p: c for p, c in sorted(new.items()) if base.get(p) != c}
        ch.update({p: None for p in sorted(base) if p not in new})
        return ch

    @endpoint("pr_diff", S.obj({"pr": S.s_str(maxLength=40)}, ["pr"]), read_only=True)
    def ep_pr_diff(self, ctx, pr):
        """The files a pull request changes, with their new contents (your reading is recorded, so that your
        review can count)."""
        st = ctx.state
        p = st.get("pkg_prs", pr)
        code = ctx.world.modules["code"]
        if p is None or not code.can_read(ctx, p["repo"]):
            raise NotFound("no pull request " + pr)
        ch = self._pr_changes(st, p)
        if not st.has("pkg_reads", f"{pr}|{ctx.actor}"):
            ctx.emit("pkg.pr_read", {"id": f"{pr}|{ctx.actor}", "pr": pr, "by": ctx.actor}, vis=(ctx.actor,))
        return {"pr": pr, "repo": p["repo"], "title": p["title"], "body": p["body"],
                "changes": {k: (v[:20000] if v is not None else None) for k, v in ch.items()},
                "tests": p["tests"], "status": p["status"]}

    @endpoint("pr_review", S.obj({"pr": S.s_str(maxLength=40), "verdict": S.s_enum(["approve", "request_changes"]),
                                  "comment": S.s_str(maxLength=5000)}, ["pr", "verdict"]), cost_s=60)
    def ep_pr_review(self, ctx, pr, verdict, comment=""):
        """Review a pull request you have read (pkg.pr_diff). Reviews by the author or principals related to the
        author are recorded but never count as independent."""
        st = ctx.state
        p = st.get("pkg_prs", pr)
        if p is None or not ctx.world.modules["code"].can_read(ctx, p["repo"]):
            raise NotFound("no pull request " + pr)
        if p["status"] != "open":
            raise Conflict(f"pull request is {p['status']}")
        if ctx.actor == p["author"]:
            raise Forbidden("authors cannot review their own pull request")
        if not st.has("pkg_reads", f"{pr}|{ctx.actor}"):
            raise Forbidden("read the change first (pkg.pr_diff)")
        rid = st.new_id("prrev")
        ctx.emit("pkg.review", {"id": rid, "pr": pr, "reviewer": ctx.actor, "verdict": verdict, "comment": comment[:5000],
                                "independent": self._independent(ctx.world, ctx.actor, p["author"]),
                                "source_commit": p["source_commit"]},
                 vis=self._repo_vis(ctx, st.tables["repos"][p["repo"]]))
        return {"review": rid}

    @endpoint("pr_merge", S.obj({"pr": S.s_str(maxLength=40)}, ["pr"]), cost_s=60)
    def ep_pr_merge(self, ctx, pr):
        """Merge a pull request into the target repository (maintainers): needs an approving independent review of
        the proposed commit, passing tests, and an unchanged target since the request was opened."""
        st = ctx.state
        p = st.get("pkg_prs", pr)
        if p is None:
            raise NotFound("no pull request " + pr)
        if p["status"] != "open":
            raise Conflict(f"pull request is {p['status']}")
        code = ctx.world.modules["code"]
        tgt = code.repo(st, p["repo"])
        ident(ctx.world).authorize(ctx, tgt["owner"], "code.write", scope=p["repo"])
        revs = [st.tables["pkg_reviews"][r] for r in st.find_ids("pkg_reviews", "pr", pr)]
        if not any(r["verdict"] == "approve" and r["independent"] and r["source_commit"] == p["source_commit"]
                   for r in revs):
            raise Forbidden("needs an approving review by someone independent of the author")
        if any(r["verdict"] == "request_changes" and r["source_commit"] == p["source_commit"] for r in revs):
            raise Forbidden("changes were requested on this commit")
        if p["tests"]["failed"] or not p["tests"]["passed"]:
            raise Forbidden("the proposed change does not pass its tests")
        if tgt["head"] != p["base_commit"]:
            raise Conflict("the target changed since the pull request was opened; reopen it")
        cid = code.commit_files(ctx, p["repo"], self._pr_changes(st, p), f"merge {pr}: {p['title']}", ctx.actor)
        ctx.emit("pkg.merged", {"id": pr, "commit": cid, "by": ctx.actor}, vis=self._repo_vis(ctx, tgt))
        return {"merged": True, "commit": cid}

    @endpoint("fork", S.obj({"package": S.s_str(maxLength=40), "name": S.s_str(pattern=NAME_RE, maxLength=40)},
                            ["package", "name"]), cost_s=30)
    def ep_fork(self, ctx, package, name):
        """Fork a package's repository into a new public, open repository of yours (lineage is recorded when you
        register it as a package)."""
        st = ctx.state
        p = self.package(st, package)
        res = ctx.call("code.fork", {"repo": p["repo"], "name": name})
        ctx.call("code.set_visibility", {"repo": res["repo"], "visibility": "public", "license": "open"})
        ctx.emit("pkg.forked", {"package": package, "repo": res["repo"], "by": ctx.actor}, vis="public")
        return {"repo": res["repo"]}

    @endpoint("report", S.obj({"package": S.s_str(maxLength=40), "title": S.s_str(minLength=1, maxLength=200),
                               "body": S.s_str(maxLength=5000),
                               "labels": S.s_arr(S.s_str(pattern=r"[a-z0-9][a-z0-9:._-]{0,30}", maxLength=31),
                                                 maxItems=5)}, ["package", "title"]), cost_s=30)
    def ep_report(self, ctx, package, title, body="", labels=()):
        """File an issue on a package's public tracker."""
        p = self.package(ctx.state, package)
        return ctx.call("issues.create", {"project": p["project"], "title": title, "body": body,
                                          "labels": list(labels)})

    # ================================================================== failures
    def _repro(self, ctx, repro_repo, repro_commit, pkg_repo, pkg_commit, principal) -> dict:
        files = self._files_with(ctx, repro_repo, repro_commit, principal, swap={pkg_repo: pkg_commit})
        return self._ci(ctx, files, principal, "pkg.repro")

    @endpoint("advisory", S.obj({"package": S.s_str(maxLength=40), "version": S.s_str(pattern=VER_RE, maxLength=14),
                                 "kind": S.s_enum(["security", "reliability"]),
                                 "severity": S.s_enum(["low", "medium", "high", "critical"]),
                                 "description": S.s_str(maxLength=4000),
                                 "repro_repo": S.s_str("your repository whose tests demonstrate the defect (they "
                                                       "must fail against the affected version)", maxLength=81)},
                                ["package", "version", "kind", "severity", "description", "repro_repo"]), cost_s=120)
    def ep_advisory(self, ctx, package, version, kind, severity, description, repro_repo):
        """Report a security or reliability defect with a reproduction repository whose tests fail against the
        affected release. The report is kept private (you and the maintainer) and checked against every later
        release; it becomes verified and public only when a release makes the reproduction pass."""
        st = ctx.state
        p = self.package(st, package)
        rid = f"{package}@{version}"
        rel = st.get("pkg_releases", rid)
        if rel is None:
            raise NotFound("no release " + rid)
        code = ctx.world.modules["code"]
        rr = code.repo(st, repro_repo)
        ident(ctx.world).authorize(ctx, rr["owner"], "code.write", scope=repro_repo)
        res, exercises = self._repro_check(ctx, repro_repo, rr["head"], p["repo"], rel["commit"], package, ctx.actor)
        if res["failed"] == 0:
            raise InvalidAction("the reproduction's tests pass against the affected release")
        if not exercises:
            raise InvalidAction("the reproduction does not exercise the package (its outcome is the same with the "
                                "package ablated)")
        later = [r for r in self.releases(st, package) if parse_version(r["version"]) > parse_version(version)]
        affected = [version]
        fixed_in = None
        for r in later:
            res2 = self._repro(ctx, repro_repo, rr["head"], p["repo"], r["commit"], ctx.actor)
            if res2["failed"] > 0:
                affected.append(r["version"])
            elif res2["passed"] > 0:
                fixed_in = r["version"]
                break
        verified = fixed_in is not None
        aid = st.new_id("adv")
        ctx.emit("pkg.advisory", {"id": aid, "package": package, "version": version, "kind": kind,
                                  "severity": severity, "description": description[:4000], "by": ctx.actor,
                                  "repro_repo": repro_repo, "repro_commit": rr["head"], "verified": verified,
                                  "affected": affected, "fixed_in": fixed_in,
                                  "tests": {"passed": res["passed"], "failed": res["failed"]}},
                 vis="public" if verified else (ctx.actor, p["owner"]))
        if verified:
            self._publish_advisory(ctx, aid)
        return {"advisory": aid, "verified": verified, "status": "verified" if verified else "reported",
                "affected": affected, "fixed_in": fixed_in, "tests": {"passed": res["passed"], "failed": res["failed"]}}

    def _repro_check(self, ctx, repro_repo, repro_commit, pkg_repo, pkg_commit, package, principal):
        """Run a reproduction against a release and with the package ablated: (result, exercises the package)."""
        st = ctx.state
        files = self._files_with(ctx, repro_repo, repro_commit, principal, swap={pkg_repo: pkg_commit})
        res = self._ci(ctx, files, principal, "pkg.repro")
        abl = self._run(ctx, self._ablate(st, files, package), {"kind": "call", "target": "_pkgtrace:tests",
                                                                 "args": {}, "caller": principal}, principal,
                        "pkg.repro.ablate")
        base = self._outcome({"ok": True, "result": {"results": res["results"], "passed": res["passed"]}})
        return res, base != self._outcome(abl)

    def _publish_advisory(self, ctx, aid: str) -> None:
        st = ctx.state
        a = st.tables["pkg_advisories"][aid]
        p = st.tables["pkg_packages"][a["package"]]
        if "issues" in ctx.world.modules:
            ctx.call("issues.create", {"project": p["project"], "title": f"[{a['kind']}] {a['description'][:150]}",
                                       "body": f"Advisory {aid} ({a['severity']}): affects {', '.join(a['affected'])}; "
                                               f"fixed in {a['fixed_in']}; reproduction {a['repro_repo']}",
                                       "labels": [a["kind"], a["severity"]]})

    # ================================================================== needs-graph interface
    def relations(self, world):
        """(src, rel, dst, source, owner_src, owner_dst) structural relations of the software ecosystem."""
        st = world.state
        out = []
        for name, p in st.scan("pkg_packages"):
            for r in self.releases(st, name):
                out.append((f"pkg:{name}@{r['version']}", "produces", f"pkg:{name}", "release", p["owner"], p["owner"]))
            if p["forked_from"] and st.has("pkg_packages", p["forked_from"]):
                mine, theirs = self.releases(st, name), self.releases(st, p["forked_from"])
                if mine and theirs and all(k in mine[-1]["api"] for k in theirs[-1]["api"]):
                    a, b = sorted([f"pkg:{name}", f"pkg:{p['forked_from']}"])
                    out.append((a, "substitutes", b, "fork with the same API", None, None))
        for _, d in st.scan("pkg_deps"):
            if d["status"] == "active":
                out.append((f"repo:{d['repo']}", "requires", f"pkg:{d['package']}", "dependency", d["owner"],
                            st.tables["pkg_packages"][d["package"]]["owner"]))
        for aid, a in st.scan("pkg_advisories"):
            if a["verified"]:
                out.append((f"adv:{aid}", "constrains", f"pkg:{a['package']}", "advisory", a["by"],
                            st.tables["pkg_packages"][a["package"]]["owner"]))
        return out

    def affects(self, world, adv_node: str, dependent_node: str) -> bool:
        st = world.state
        a = st.get("pkg_advisories", adv_node.split(":", 1)[1])
        if a is None or not a["verified"] or not dependent_node.startswith("repo:"):
            return False
        d = st.get("pkg_deps", f"{dependent_node.split(':', 1)[1]}|{a['package']}")
        return d is not None and d["status"] == "active" and d["version"] in a["affected"]

    def remedies(self, world, pkg_node: str, since_t: int) -> list[dict]:
        """Releases after `since_t` not affected by any verified advisory, and substitute packages."""
        st = world.state
        name = pkg_node.split(":", 1)[1]
        out = []
        advs = [a for a in st.find("pkg_advisories", "package", name) if a["verified"]]
        for r in self.releases(st, name):
            if r["t"] >= since_t and not any(r["version"] in a["affected"] for a in advs) and \
                    all(a["fixed_in"] for a in advs):
                out.append({"by": r["id"], "host": None, "provider": st.tables["pkg_packages"][name]["owner"],
                            "path": [{"src": f"pkg:{name}@{r['version']}", "rel": "produces", "dst": pkg_node,
                                      "source": "release"}]})
        # substitute packages: another package whose latest release offers every function of this one's API
        mine = self.releases(st, name)
        api = set((mine[-1]["api"] if mine else {}) or {})
        for other, op in sorted(st.scan("pkg_packages")):
            if other == name or not api:
                continue
            rs = [r for r in self.releases(st, other) if r["t"] >= since_t or True]
            if rs and api <= set(rs[-1]["api"] or {}) and not any(
                    rs[-1]["version"] in a["affected"] for a in st.find("pkg_advisories", "package", other)
                    if a["verified"]):
                out.append({"by": rs[-1]["id"], "host": None, "provider": op["owner"],
                            "path": [{"src": f"pkg:{other}", "rel": "substitutes", "dst": pkg_node,
                                      "source": "same API"}]})
        return out

    # ================================================================== appliers
    @applier("pkg.registered")
    def ap_registered(self, ev):
        p = dict(ev.payload)
        name = p.pop("id")
        p.update(t=ev.t, stars=0)
        self.world.state.insert("pkg_packages", name, p)

    @applier("pkg.released")
    def ap_released(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, yanked=False)
        self.world.state.insert("pkg_releases", p["id"], p)

    @applier("pkg.yanked")
    def ap_yanked(self, ev):
        self.world.state.patch("pkg_releases", ev.payload["id"], yanked=True, yank_reason=ev.payload["reason"])

    @applier("pkg.dependency")
    def ap_dependency(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, status="active")
        self.world.state.put("pkg_deps", p.pop("id"), p)

    @applier("pkg.compat")
    def ap_compat(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.put("pkg_compat", p.pop("id"), p)

    @applier("pkg.executed")
    def ap_executed(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        p["t"] = ev.t
        st.insert("pkg_execs", p["id"], p)
        if not p["ok"]:
            return
        for name in sorted(p.get("exercised") or []):
            k = f"{name}|{p['repo']}"
            a = st.tables["pkg_adoptions"].get(k)
            n = int((p.get("self_reported_calls") or {}).get(name, 0))
            if a is None:
                st.insert("pkg_adoptions", k, {"package": name, "repo": p["repo"], "owner": p["owner"],
                                               "first_t": ev.t, "last_t": ev.t, "executions": 1,
                                               "self_reported_calls": n, "versions": [p["locked"].get(name)]})
            else:
                vs = list(a["versions"])
                if p["locked"].get(name) not in vs:
                    vs.append(p["locked"].get(name))
                st.patch("pkg_adoptions", k, last_t=ev.t, executions=a["executions"] + 1,
                         self_reported_calls=a.get("self_reported_calls", 0) + n, versions=vs)

    @applier("pkg.deployed")
    def ap_deployed(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("pkg_deploys", p.pop("id"), p)

    @applier("pkg.pr")
    def ap_pr(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, status="open")
        self.world.state.insert("pkg_prs", p.pop("id"), p)

    @applier("pkg.pr_read")
    def ap_pr_read(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("pkg_reads", p.pop("id"), p)

    @applier("pkg.review")
    def ap_review(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("pkg_reviews", p.pop("id"), p)

    @applier("pkg.merged")
    def ap_merged(self, ev):
        self.world.state.patch("pkg_prs", ev.payload["id"], status="merged", merge_commit=ev.payload["commit"],
                               merged_by=ev.payload["by"], merged_t=ev.t)

    @applier("pkg.forked")
    def ap_forked(self, ev):
        st = self.world.state
        st.meta_set("pkg:forks", st.meta_get("pkg:forks", 0) + 1)

    @applier("pkg.advisory")
    def ap_advisory(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, status="fixed" if p["verified"] else "reported", checks=[])
        self.world.state.insert("pkg_advisories", p["id"], p)

    @applier("pkg.advisory_check")
    def ap_advisory_check(self, ev):
        st = self.world.state
        p = ev.payload
        a = st.tables["pkg_advisories"][p["id"]]
        ch = {"checks": list(a["checks"]) + [{"version": p["version"], "reproduces": p["reproduces"], "t": ev.t}]}
        if p["reproduces"]:
            ch["affected"] = list(a["affected"]) + [p["version"]]
        elif p["tests"]["passed"] > 0 and not a["fixed_in"]:
            # a release that makes the reproduction pass verifies the defect (it was in the package) and fixes it
            ch.update(fixed_in=p["version"], fixed_t=ev.t, status="fixed", verified=True)
        st.patch("pkg_advisories", p["id"], **ch)

    @applier("pkg.star")
    def ap_star(self, ev):
        st = self.world.state
        p = ev.payload
        st.insert("pkg_stars", f"{p['package']}|{p['by']}", {"package": p["package"], "by": p["by"], "t": ev.t})
        st.patch("pkg_packages", p["package"], stars=st.tables["pkg_packages"][p["package"]]["stars"] + 1)

    @applier("pkg.seen")
    def ap_seen(self, ev):
        st = self.world.state
        for n in ev.payload["packages"]:
            k = f"{n}|{ev.payload['by']}"
            if not st.has("pkg_seen", k):
                st.insert("pkg_seen", k, {"package": n, "by": ev.payload["by"], "t": ev.t})

    # ================================================================== invariants
    def check_invariants(self, world):
        errs = []
        st = world.state
        for k, a in st.scan("pkg_adoptions"):
            ex = [e for e in st.find("pkg_execs", "repo", a["repo"]) if e["ok"] and a["package"] in (e.get("exercised")
                                                                                                    or [])]
            if len(ex) != a["executions"]:
                errs.append(f"adoption {k}: {a['executions']} executions recorded, {len(ex)} traced")
        for name, p in st.scan("pkg_packages"):
            n = sum(1 for _ in st.find("pkg_stars", "package", name))
            if n != p["stars"]:
                errs.append(f"package {name}: star counter {p['stars']} != {n}")
        for pid, pr in st.scan("pkg_prs"):
            if pr["status"] == "merged" and not any(
                    r["verdict"] == "approve" and r["independent"] for r in st.find("pkg_reviews", "pr", pid)):
                errs.append(f"{pid} merged without an independent approval")
        return errs
