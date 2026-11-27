"""Evidence records for acceptance requirements.

A gate test measures something on the real system and calls `record(requirement, metrics, ...)`. The record states
what was measured (the metrics) and, computed here rather than asserted by the test, where it came from: the commit of
the checkout it ran in, the pytest node that produced it, every `crucible` (and test) module the process had imported
with its git blob id, `crucible` modules imported from outside the checkout, the environment (Python, CPU quota, GPU
visibility), checksums of the datasets the test declares, and the hashes of cited artifacts. It does not state whether
it passed: the checker evaluates the manifest's criteria on the metrics.

What else ran in the process is recorded too, so the checker can reject a measurement of code that is not the
committed code: the interpreter (executable, prefix, flags, `sys.path`), its start-up hooks (every `.pth` file of its
site directories with its SHA-256, and any loaded `sitecustomize` / `usercustomize`), modules imported from outside
both the checkout and the interpreter's own directories (`foreign_modules`), functions of checkout modules whose code
lives in such a foreign file (`patched_functions`: a patch, whatever the injection vector), PYTHON*/LD_* variables of
the process, and - in official runs, through an audit hook installed when this module is imported - every file
under the checkout the process opened (`files_read`, with git blob ids), so that a changed data or configuration file
makes the evidence stale just as a changed module does.

Evidence is filed only by an official run (`python -m crucible.acceptance.v5 run`): the runner clones the committed
tree into a fresh directory, marks the clone (a marker inside the clone's git directory, outside the work tree), runs
the requirement's tests there with `CRUCIBLE_V5_RUN_ID` and `CRUCIBLE_V5_EVIDENCE_OUT` set, and installs the records,
its own run record and the JUnit file into `docs/v5/evidence/`. Anywhere else (an ordinary test run, a working tree,
a run id without a matching clone marker) the record is written to a temporary directory and is never evidence.
"""
from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import site
import subprocess
import sys
import sysconfig
import types

from .manifest import crucible_root, docs_dir

SCHEMA = "crucible.v5.evidence/4"                  # /4: caller, phase, files_read_truncated (D-83); executed_implementation (D-83 4th addendum)
RUN_SCHEMA = "crucible.v5.run/2"
REVIEW_SCHEMA = "crucible.v5.review/3"
MARKER = "crucible_v5_run.json"                     # inside the clone's git dir: <clone>/.git/crucible_v5_run.json
EVIDENCE_DIR = "docs/v5/evidence"
RUNS_DIR = "docs/v5/evidence/official_runs"         # not `runs/`: that name is git-ignored repository-wide
REVIEWS_DIR = "docs/v5/reviews"
REPRO_DIR = "docs/v5/reviews/reproductions"
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_READ_BYTES = 256 * 1024 * 1024                  # files_read: larger files are listed without a blob id
ALLOWED_PY_ENV = ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTHONHASHSEED", "PYTHONUNBUFFERED",
                  "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONFAULTHANDLER")
STARTUP_HOOK_NAMES = ("sitecustomize", "usercustomize")

# ------------------------------------------------------------------------------------------------- files read
_OPENED: set[str] = set()
_OPENED_MAX = 200_000
_TRUNCATED = [False]                                 # more distinct files under the checkout than _OPENED_MAX
_HOOK_ROOT = [""]                                    # only paths under the checkout are kept (set at installation)


def _open_hook(event: str, args: tuple) -> None:
    if event != "open" or not args:
        return
    p = args[0]
    if isinstance(p, (str, bytes, os.PathLike)) and not isinstance(p, int):
        try:
            ap = os.path.abspath(os.fsdecode(p))
        except (TypeError, ValueError):
            return
        if _HOOK_ROOT[0] and not ap.startswith(_HOOK_ROOT[0]):
            return                                   # outside the checkout: never stale-relevant, never counted
        if ap in _OPENED:
            return
        if len(_OPENED) >= _OPENED_MAX:
            _TRUNCATED[0] = True                     # the record says so and the checker rejects it
            return
        _OPENED.add(ap)


if os.environ.get("CRUCIBLE_V5_RUN_ID"):            # only inside official runs: no cost anywhere else
    _HOOK_ROOT[0] = os.path.abspath(crucible_root()) + os.sep
    sys.addaudithook(_open_hook)


# ------------------------------------------------------------------------------------------- execution observer
# Wave 3b repair pass (D-83 fourth addendum, probe V9 "detect and sacrifice"): the block ablation proves that the
# ablated run ran into the implementation, not that the measured run did. A test that never calls its implementation
# in a normal run - recording values from elsewhere - but runs into it on purpose when it detects the ablation (the
# ablated tree's text is readable) left no record and looked sensitive. In official runs this module therefore
# observes, in the process that records, which functions of the checkout's src/ started while no module was being
# imported and outside the recording machinery (started, not returned: an honest test may exercise only paths that
# raise, e.g. an API refusing bad input); `record()` writes the count for the
# requirement's implementation files into the record (`executed_implementation`), and the ablation verdict accepts
# "no record, implementation blocked in the recording process" only when the reproduction executed its implementation
# in-process. A sys.monitoring tool of its own (never in an ablated run, whose hook holds the optimizer id); each code
# object is seen once and its events are then disabled, so the cost is one callback per function.
_EXEC: dict[str, set] = {}                          # realpath of a src file -> qualnames of functions that ran
_EXEC_STATE = {"tool": None, "src": ""}
OBSERVER_TOOL_NAME = "crucible-v5-observer"


def _observer_install(src_dir: str) -> None:
    mon = getattr(sys, "monitoring", None)
    if mon is None or any(mon.get_tool(t) == "crucible-v5-ablation" for t in range(6)):
        return
    tool = next((t for t in (5, 2) if mon.get_tool(t) is None), None)
    if tool is None:
        return
    src = os.path.realpath(src_dir) + os.sep
    machinery = os.sep + os.path.join("crucible", "acceptance", "v5", "evidence.py")
    seen: dict[int, object] = {}

    def started(code, offset):
        f = code.co_filename
        if not (code.co_flags & 1) or not f.startswith(src):
            return mon.DISABLE
        if id(code) in seen:
            return mon.DISABLE
        fr = sys._getframe(1)
        while fr is not None:
            c = fr.f_code
            if c.co_filename.startswith("<frozen importlib._bootstrap"):
                return None                          # import time: not the test executing its implementation
            if c.co_qualname in ("record", "_open_hook") and c.co_filename.endswith(machinery):
                return None
            fr = fr.f_back
        seen[id(code)] = code
        _EXEC.setdefault(f, set()).add(code.co_qualname)
        return mon.DISABLE
    mon.use_tool_id(tool, OBSERVER_TOOL_NAME)
    mon.register_callback(tool, mon.events.PY_START, started)
    mon.set_events(tool, mon.events.PY_START)
    _EXEC_STATE.update(tool=tool, src=src)


def executed_implementation(root: str, requirement: str) -> dict | None:
    """What the observer saw of the requirement's implementation files in this process: {"functions": n, "files":
    {relpath: n}}, or None when no observer ran (not an official run, or no free monitoring tool)."""
    if _EXEC_STATE["tool"] is None:
        return None
    try:
        m = json.load(open(os.path.join(docs_dir(root), "ACCEPTANCE_MANIFEST.json")))
        files = next(x for x in m["requirements"] if x["id"] == requirement).get("implementation_files") or []
    except (OSError, ValueError, StopIteration, KeyError):
        files = []
    out = {}
    for rel in files:
        n = len(_EXEC.get(os.path.realpath(os.path.join(root, rel)), ()))
        if n:
            out[rel] = n
    return {"functions": sum(out.values()), "files": out}


if os.environ.get("CRUCIBLE_V5_RUN_ID"):
    _observer_install(os.path.join(crucible_root(), "src"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_blob_id(data: bytes) -> str:
    """The id git gives a file with these bytes (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git(*args: str, cwd: str | None = None, strip: bool = True) -> str:
    out = subprocess.check_output(["git", *args], cwd=cwd or crucible_root(), stderr=subprocess.DEVNULL).decode()
    return out.strip() if strip else out


def tree_state(cwd: str | None = None) -> dict:
    """HEAD commit of the checkout and every uncommitted path (no exemptions)."""
    head = git("rev-parse", "HEAD", cwd=cwd)
    dirty = []
    for line in git("status", "--porcelain", "--untracked-files=all", cwd=cwd, strip=False).splitlines():
        if line.strip():
            dirty.append(line[3:].strip().strip('"'))
    return {"commit": head, "dirty_paths": dirty[:50], "clean": not dirty}


def gate_of(requirement: str) -> str:
    return "SP" if requirement.startswith("SP-") else requirement.split(".", 1)[0]


def evidence_relpath(requirement: str) -> str:
    return f"{EVIDENCE_DIR}/gate_{gate_of(requirement)}/{requirement}.json"


def identity() -> dict:
    """Who produced something: the declared agent name plus process facts. The session id is hashed, never stored."""
    sess = os.environ.get("CRUCIBLE_V5_SESSION") or os.environ.get("CLAUDE_CODE_SESSION_ID") or ""
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "")
    return {"agent": os.environ.get("CRUCIBLE_V5_AGENT") or os.environ.get("CRUCIBLE_V5_AUTHOR") or "",
            "session_sha256": hashlib.sha256(sess.encode()).hexdigest()[:16] if sess else "",
            "uid": os.getuid(), "user": user, "host": platform.node(), "pid": os.getpid(), "ppid": os.getppid()}


def environment() -> dict:
    try:
        from ...scale.slots import cgroup_cpus
        cpus = cgroup_cpus()
    except Exception:
        cpus = float(os.cpu_count() or 1)
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      stderr=subprocess.DEVNULL, timeout=10).decode().strip() or None
    except Exception:
        pass
    torch_cuda = None
    if "torch" in sys.modules:                       # never import torch just to ask
        try:
            torch_cuda = bool(sys.modules["torch"].cuda.is_available())
        except Exception:
            torch_cuda = None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_quota": cpus, "gpu": gpu,
            "cuda_visible_devices": vis, "gpu_visible": bool(gpu) and vis not in ("", "-1"),
            "torch_cuda_available": torch_cuda, "host": platform.node(),
            "historical_baseline": os.environ.get("CRUCIBLE_HISTORICAL_BASELINE") or None,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def trusted_dirs() -> list[str]:
    """The interpreter's own directories: its standard library and its site directories."""
    paths = sysconfig.get_paths()
    out = {os.path.realpath(paths[k]) for k in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(k)}
    try:
        out.update(os.path.realpath(d) for d in site.getsitepackages())
    except AttributeError:
        pass
    return sorted(out)


def _under(path: str, dirs) -> bool:
    return any(path == d or path.startswith(d + os.sep) for d in dirs)


def interpreter() -> dict:
    f = sys.flags
    return {"executable": os.path.realpath(sys.executable), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "version": sys.version.split()[0],
            "flags": {"no_user_site": bool(f.no_user_site), "ignore_environment": bool(f.ignore_environment),
                      "isolated": bool(f.isolated), "safe_path": bool(getattr(f, "safe_path", False)),
                      "dont_write_bytecode": bool(f.dont_write_bytecode)},
            "user_site_enabled": bool(site.ENABLE_USER_SITE), "path": list(sys.path)[:60]}


def site_hooks() -> dict:
    """Code that runs at interpreter start-up: the .pth files of the site directories actually processed, and the
    sitecustomize / usercustomize modules if one was loaded."""
    dirs = []
    try:
        dirs += site.getsitepackages()
    except AttributeError:
        pass
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    pth = {}
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if n.endswith(".pth"):
                p = os.path.join(d, n)
                try:
                    pth[os.path.realpath(p)] = sha256_file(p)
                except OSError:
                    pth[os.path.realpath(p)] = None
    cust = {}
    for name in STARTUP_HOOK_NAMES:
        m = sys.modules.get(name)
        cust[name] = os.path.realpath(m.__file__) if m is not None and getattr(m, "__file__", None) else None
    return {"pth": pth, "customize": cust}


def foreign_modules(root: str) -> list[str]:
    """Modules (any name) imported from files outside both the checkout and the interpreter's own directories."""
    root = os.path.realpath(root)
    trusted = trusted_dirs()
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        rf = os.path.realpath(f)
        if not _under(rf, [root]) and not _under(rf, trusted):
            out.append(f"{name}:{rf}")
    return sorted(out)


def _functions_of(obj):
    if isinstance(obj, types.FunctionType):
        yield "", obj
    elif isinstance(obj, (staticmethod, classmethod)):
        f = getattr(obj, "__func__", None)
        if isinstance(f, types.FunctionType):
            yield "", f
    elif isinstance(obj, property):
        for acc in (obj.fget, obj.fset, obj.fdel):
            if isinstance(acc, types.FunctionType):
                yield "", acc


_CODE_INDEX: dict[tuple, dict] = {}


def _code_index(path: str) -> dict | None:
    """(co_qualname, co_firstlineno) -> [code objects] of everything compiled from the file's current bytes (cached by
    path, size and mtime); None when the file cannot be read or compiled."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _CODE_INDEX:
        try:
            with open(path, "rb") as fh:
                top = compile(fh.read(), path, "exec", dont_inherit=True)
        except (OSError, SyntaxError, ValueError):
            _CODE_INDEX[key] = None
            return None
        idx: dict = {}
        todo = [top]
        while todo:
            c = todo.pop()
            idx.setdefault((c.co_qualname, c.co_firstlineno), []).append(c)
            todo.extend(x for x in c.co_consts if isinstance(x, types.CodeType))
        _CODE_INDEX[key] = idx
    return _CODE_INDEX[key]


_CODE_FIELDS = ("co_code", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_flags", "co_argcount",
                "co_posonlyargcount", "co_kwonlyargcount", "co_name", "co_qualname")


def _same_code(a, b) -> bool:
    if any(getattr(a, f) != getattr(b, f) for f in _CODE_FIELDS) or len(a.co_consts) != len(b.co_consts):
        return False
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, types.CodeType) or isinstance(y, types.CodeType):
            if not (isinstance(x, types.CodeType) and isinstance(y, types.CodeType) and _same_code(x, y)):
                return False
        elif type(x) is not type(y) or (x != y and not (x != x and y != y)):
            return False
    return True


def _plain_source_module(mod) -> bool:
    """Loaded by the standard source loader (not pytest's assertion-rewriting loader, whose code differs from a plain
    compile of the file by design)."""
    import importlib.machinery
    return type(getattr(mod, "__loader__", None)) is importlib.machinery.SourceFileLoader


def code_object_patches(root: str) -> list[str]:
    """Functions of modules under the checkout's `src/` whose code object claims a file of the checkout but is not
    the code compiled from that file (`fn.__code__ = fn.__code__.replace(...)`, `types.FunctionType` over altered or
    foreign globals): co_filename is kept by such a patch, so `patched_functions` (which judges by file) cannot see it
    (wave 3b verifier probe V4)."""
    root = os.path.realpath(root)
    src = os.path.join(root, "src")
    homes: dict[str, dict] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py") and _under(os.path.realpath(f), [src]) and _plain_source_module(mod):
            try:
                homes.setdefault(os.path.realpath(f), []).append(vars(mod))
            except TypeError:
                pass
    out = []
    seen: set[int] = set()

    def check(fn, where: str) -> None:
        if id(fn) in seen:
            return
        seen.add(id(fn))
        code = fn.__code__
        cf = code.co_filename
        if not cf or cf.startswith("<"):
            return
        rf = os.path.realpath(cf)
        if rf not in homes:
            return
        idx = _code_index(rf)
        if idx is None:
            return
        if not any(_same_code(code, c) for c in idx.get((code.co_qualname, code.co_firstlineno), [])):
            out.append(f"{where} <- code object not compiled from {os.path.relpath(rf, root)}")
        elif not any(fn.__globals__ is g for g in homes[rf]):
            out.append(f"{where} <- function of {os.path.relpath(rf, root)} rebuilt over other globals")

    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not _under(os.path.realpath(f), [root]):
            continue
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for attr, val in items:
            for _, fn in _functions_of(val):
                check(fn, f"{name}.{attr}")
            if isinstance(val, type) and getattr(val, "__module__", None) == name:
                try:
                    members = list(vars(val).items())
                except TypeError:
                    continue
                for mname, mval in members:
                    for _, fn in _functions_of(mval):
                        check(fn, f"{name}.{attr}.{mname}")
    return sorted(set(out))


_DEFS: dict[str, dict | None] = {}


def _scope_definitions(body: list, walrus: bool = True) -> tuple[dict[str, bool], set[str], dict[str, list]]:
    """For one scope's statements (descending into if/try/with/for/while blocks, not into functions or classes):
    {def name: decorated?}, every other name the scope binds (assignments, imports, deletes, classes, loop and with
    targets), and the class definitions {name: [body]}."""
    defs: dict[str, bool] = {}
    other: set[str] = set()
    classes: dict[str, list] = {}
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name in defs:
                other.add(n.name)                       # defined twice: which one is bound is the file's choice
            defs[n.name] = bool(n.decorator_list)
            continue
        if isinstance(n, ast.ClassDef):
            other.add(n.name)
            classes.setdefault(n.name, []).append(n.body)
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            other.update((a.asname or a.name).split(".")[0] for a in n.names)
            continue
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, ast.Delete):
            targets = n.targets
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While, getattr(ast, "TryStar", ast.Try))):
            targets = []
        else:                                           # match statements, walrus in expressions: everything stored
            targets = [n] if walrus or isinstance(n, getattr(ast, "Match", ())) else []
        for tg in targets:
            for t in ast.walk(tg):
                if isinstance(t, ast.Name) and isinstance(t.ctx, (ast.Store, ast.Del)):
                    other.add(t.id)
        if isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While, getattr(ast, "TryStar", ast.Try))):
            for field in ("body", "orelse", "finalbody"):
                stack.extend(getattr(n, field, []) or [])
            for h in getattr(n, "handlers", []) or []:
                if h.name:
                    other.add(h.name)
                stack.extend(h.body)
            for t in ast.walk(ast.Module(body=[x for x in (getattr(n, "target", None), *[i.optional_vars for i in
                                                     getattr(n, "items", []) or []]) if x is not None],
                                         type_ignores=[])):
                if isinstance(t, ast.Name):
                    other.add(t.id)
            test = getattr(n, "test", None)
            if test is not None:
                for t in ast.walk(test):
                    if isinstance(t, ast.NamedExpr):
                        other.add(t.target.id)
    return defs, other, classes


def _definitions(path: str) -> dict | None:
    """The functions a source file defines and never rebinds: {"": {name: decorated}, "Class": {...}} for the module
    scope and each top-level class body; None when the file cannot be parsed."""
    if path in _DEFS:
        return _DEFS[path]
    try:
        text = open(path, encoding="utf-8").read()
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        _DEFS[path] = None
        return None
    walrus = ":=" in text
    defs, other, classes = _scope_definitions(tree.body, walrus)
    if "global " in text:                               # `global x` in any function rebinds x at module scope
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                other.update(node.names)
    out = {"": {} if "globals()" in text else {k: v for k, v in defs.items() if k not in other}}
    for cname, bodies in classes.items():
        if len(bodies) != 1:
            continue
        cdefs, cother, _ = _scope_definitions(bodies[0], walrus)
        out[cname] = {k: v for k, v in cdefs.items() if k not in cother}
    _DEFS[path] = out
    return out


_WRAP_ATTRS = ("__func__", "fget", "__wrapped__", "func")


def _leaves(obj, depth: int = 0) -> list:
    """Every callable a decorated object wraps, following each wrapping attribute (a functools.partial carries both
    `func` and, after functools.wraps, `__wrapped__`: both are followed)."""
    inner = [getattr(obj, a, None) for a in _WRAP_ATTRS] if depth < 8 else []
    inner = [x for x in inner if x is not None and x is not obj]
    if not inner:
        return [obj]
    return [leaf for x in inner for leaf in _leaves(x, depth + 1)]


def rebound_definitions(root: str) -> list[str]:
    """Functions a module under the checkout's `src/` defines (at module level or in a top-level class body, and
    never rebinds in its own source) that are no longer bound to that function: deleted, replaced by an object that is
    not a function (`setattr(m, "leaks", int)`, functools.partial, a callable object) or by another function of the
    checkout (`m.leaks = m._reset`: an implementation function under another name). patched_functions and
    code_object_patches judge only the functions they find, by file and by code (wave 3b repair pass, probe V10).
    Decorated definitions are compared after unwrapping (__wrapped__, __func__, fget, func) and only rejected when
    they resolve to another function of the checkout or to a builtin."""
    root = os.path.realpath(root)
    src = os.path.join(root, "src")
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py") or not _plain_source_module(mod):
            continue
        rf = os.path.realpath(f)
        if not _under(rf, [src]):
            continue
        defs = _definitions(rf)
        if not defs:
            continue
        try:
            ns = vars(mod)
        except TypeError:
            continue
        scopes = [("", ns, defs.get("", {}))]
        for cname, cdefs in defs.items():
            cls = ns.get(cname) if cname else None
            if isinstance(cls, type) and cls.__module__ == name and cls.__qualname__ == cname:
                scopes.append((cname, vars(cls), cdefs))
        for cname, space, wanted in scopes:
            for fname, decorated in wanted.items():
                where = f"{name}.{cname + '.' if cname else ''}{fname}"
                qual = f"{cname}.{fname}" if cname else fname
                if fname not in space:
                    out.append(f"{where} <- deleted")
                    continue
                val = space[fname]
                for fn in (_leaves(val) if decorated else [val]):
                    if isinstance(fn, types.FunctionType):
                        c = fn.__code__
                        cf = os.path.realpath(c.co_filename)
                        if cf == rf:
                            if c.co_qualname != qual and (not decorated or "<locals>" not in c.co_qualname):
                                out.append(f"{where} <- bound to {c.co_qualname}, another function of the file")
                        elif _under(cf, [root]) and (not decorated or not _under(cf, [src])):
                            out.append(f"{where} <- bound to {c.co_qualname} of {os.path.relpath(cf, root)}")
                    elif not decorated:
                        out.append(f"{where} <- bound to a {type(val).__name__}, not the function the file defines")
                    elif isinstance(fn, (types.BuiltinFunctionType, type)):
                        out.append(f"{where} <- bound to the builtin {getattr(fn, '__name__', fn)!r}")
    return sorted(set(out))


def substituted_modules(root: str) -> list[str]:
    """sys.modules entries that do not name the module they hold, for the checkout's src/ packages (wave 3b repair
    pass, second round): a module of src/ registered under another name (`sys.modules["widgets_real"] = widgets`), and
    an entry under a src/ package's name that is not that package's own file (`sys.modules["widgets"] = fake`: an
    object without a file, or a file elsewhere). Namespace packages (no file, every path entry under src/) and the
    interpreter's `__main__` are what they claim."""
    root = os.path.realpath(root)
    src = os.path.join(root, "src")
    try:
        tops = {n[:-3] if n.endswith(".py") else n for n in os.listdir(src)
                if (n.endswith(".py") or os.path.isdir(os.path.join(src, n))) and not n.startswith((".", "_"))}
    except OSError:
        return []
    out = []
    for name, mod in list(sys.modules.items()):
        if name == "__main__" or mod is None:
            continue
        f = getattr(mod, "__file__", None)
        rf = os.path.realpath(f) if isinstance(f, str) else None
        if rf and _under(rf, [src]) and rf.endswith(".py"):
            rel = os.path.relpath(rf, src)[:-3].split(os.sep)
            expected = ".".join(rel[:-1] if rel[-1] == "__init__" else rel)
            if name != expected:
                out.append(f"sys.modules[{name!r}] <- holds {expected} ({os.path.relpath(rf, root)})")
        elif name.split(".")[0] in tops:
            paths = list(getattr(mod, "__path__", None) or [])
            if rf is None and paths and all(_under(os.path.realpath(p), [src]) for p in paths):
                continue                                # a namespace package of src/
            out.append(f"sys.modules[{name!r}] <- not the checkout's module ({type(mod).__name__}"
                       f"{', ' + rf if rf else ''})")
    return sorted(set(out))


_GENERATED: dict[tuple, bool] = {}


def _self_generated(path: str | None, fname: str) -> bool:
    """Does the module's own source compile code under this pseudo-filename (`compile(src, "<pcg64-seed>", "exec")`
    in crucible/mechanisms/sim.py, 4.0)? Then a function with that co_filename is the module's own generated code,
    not a patch (wave 3b repair pass: every record of a process importing crucible.mechanisms.sim was rejected)."""
    if not path or not path.endswith(".py"):
        return False
    key = (path, fname)
    if key not in _GENERATED:
        found = False
        try:
            import ast
            tree = ast.parse(open(path, encoding="utf-8").read())
            found = any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "compile"
                        and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == fname
                        for n in ast.walk(tree))
        except (OSError, SyntaxError, ValueError):
            found = False
        _GENERATED[key] = found
    return _GENERATED[key]


def patched_functions(root: str) -> list[str]:
    """Functions reachable from modules of the checkout (module attributes, and the methods of classes defined there)
    whose code is not implementation code: compiled from a real file outside the checkout and outside the interpreter;
    or - for the modules under `src/` - compiled from a file of the checkout outside `src/` (a test or a test helper
    module put it there), or compiled from a string (``exec``/``compile``) as a module-level function or an ordinary
    method (the dunder methods dataclasses and namedtuple generate are exempt). Any of these is a patch."""
    root = os.path.realpath(root)
    src = os.path.join(root, "src")
    trusted = trusted_dirs()
    ok: dict[str, bool] = {}

    def foreign(fn, owner_in_src: bool, member: str | None, owner_file: str | None = None) -> str | None:
        fname = fn.__code__.co_filename
        if not fname:
            return None
        if fname.startswith("<frozen "):        # the interpreter's frozen standard library (abc.abstractmethod)
            return None
        if fname.startswith("<"):
            dunder = member is not None and member.startswith("__") and member.endswith("__")
            if not owner_in_src or dunder or _self_generated(owner_file, fname):
                return None
            return f"code compiled from {fname} (exec/compile)"
        if fname not in ok:
            rf = os.path.realpath(fname)
            ok[fname] = _under(rf, [root]) or _under(rf, trusted)
        if not ok[fname]:
            return os.path.realpath(fname)
        if owner_in_src:
            rf = os.path.realpath(fname)
            if _under(rf, [root]) and not _under(rf, [src]):
                return f"{os.path.relpath(rf, root)} (test code)"
        return None

    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not _under(os.path.realpath(f), [root]):
            continue
        in_src = _under(os.path.realpath(f), [src])
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for attr, val in items:
            for _, fn in _functions_of(val):
                where = foreign(fn, in_src, None, os.path.realpath(f))
                if where:
                    out.append(f"{name}.{attr} <- {where}")
            if isinstance(val, type) and getattr(val, "__module__", None) == name:
                try:
                    members = list(vars(val).items())
                except TypeError:
                    continue
                for mname, mval in members:
                    for _, fn in _functions_of(mval):
                        where = foreign(fn, in_src, mname, os.path.realpath(f))
                        if where:
                            out.append(f"{name}.{attr}.{mname} <- {where}")
    return sorted(set(out))


def python_env() -> dict:
    """PYTHON* and dynamic-loader variables of this process (code-injection vectors when unexpected)."""
    return {k: v for k, v in sorted(os.environ.items())
            if (k.startswith("PYTHON") and k not in ALLOWED_PY_ENV) or k in ("LD_PRELOAD", "LD_AUDIT")}


def files_read(root: str) -> dict[str, str | None]:
    """Files under the checkout opened by this process since this module was imported (official runs only), with git
    blob ids; generated output (git directory, byte-code caches) is left out."""
    root = os.path.realpath(root)
    out: dict[str, str | None] = {}
    for p in sorted(_OPENED):
        rp = os.path.realpath(p)
        if not rp.startswith(root + os.sep) or not os.path.isfile(rp):
            continue
        rel = os.path.relpath(rp, root)
        parts = rel.split(os.sep)
        if ".git" in parts or "__pycache__" in parts:
            continue
        try:
            if os.path.getsize(rp) > MAX_READ_BYTES:
                out[rel] = None
                continue
            with open(rp, "rb") as fh:
                out[rel] = git_blob_id(fh.read())
        except OSError:
            continue
    return out


def imported_modules(root: str) -> tuple[dict[str, str], list[str]]:
    """(path relative to `root` -> git blob id) of every imported module file under `root`, and the `crucible`
    modules imported from anywhere else (which would mean the run measured code outside its checkout)."""
    root = os.path.realpath(root)
    inside, outside = {}, []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or f.endswith(".pyc"):
            continue
        rf = os.path.realpath(f)
        if rf.startswith(root + os.sep):
            try:
                with open(rf, "rb") as fh:
                    inside[os.path.relpath(rf, root)] = git_blob_id(fh.read())
            except OSError:
                pass
        elif name == "crucible" or name.startswith("crucible."):
            outside.append(f"{name}:{rf}")
    return dict(sorted(inside.items())), sorted(outside)


_DS_CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-dataset-sums.json")


def _file_sum(p: str, cache: dict) -> str:
    st = os.stat(p)
    key = f"{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        cache[key] = sha256_file(p)
    return cache[key]


def dataset_checksums(datasets: dict | None) -> dict:
    """name -> {path, files, bytes, sha256}; a directory's sha256 covers every file's relative path and content hash.
    Content hashes are cached per (path, size, mtime) in a temporary file so a 1 GB dataset is not re-read per test."""
    if not datasets:
        return {}
    try:
        cache = json.load(open(_DS_CACHE))
    except (OSError, ValueError):
        cache = {}
    out = {}
    for name, path in sorted(datasets.items()):
        if not os.path.exists(path):
            out[name] = {"path": path, "missing": True}
            continue
        if os.path.isfile(path):
            files = [("", path)]
        else:
            files = []
            for dp, dn, fn in os.walk(path):
                dn.sort()
                for f in sorted(fn):
                    full = os.path.join(dp, f)
                    files.append((os.path.relpath(full, path), full))
        h, total = hashlib.sha256(), 0
        for rel, full in files:
            h.update(f"{rel}\0{_file_sum(full, cache)}\n".encode())
            total += os.path.getsize(full)
        out[name] = {"path": path, "files": len(files), "bytes": total, "sha256": h.hexdigest()}
    try:
        tmp = f"{_DS_CACHE}.{os.getpid()}"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, _DS_CACHE)
    except OSError:
        pass
    return out


def official_run(root: str | None = None) -> dict | None:
    """The official run this process belongs to, or None. Requires the runner's environment AND the marker the
    runner wrote into the git directory of the clone this code runs from (at that clone's HEAD)."""
    rid, out = os.environ.get("CRUCIBLE_V5_RUN_ID"), os.environ.get("CRUCIBLE_V5_EVIDENCE_OUT")
    if not rid or not out:
        return None
    root = root or crucible_root()
    try:
        gd = git("rev-parse", "--absolute-git-dir", cwd=root)
        marker = json.load(open(os.path.join(gd, MARKER)))
        head = git("rev-parse", "HEAD", cwd=root)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    if marker.get("run_id") != rid or marker.get("commit") != head:
        return None
    return {"run_id": rid, "out": out, "commit": head}


def _check_metrics(metrics: dict) -> None:
    if not isinstance(metrics, dict) or not metrics:
        raise TypeError("metrics must be a non-empty dict")
    for k, v in metrics.items():
        if not isinstance(k, str):
            raise TypeError(f"metric name {k!r} must be a string")
        if not isinstance(v, (int, float, bool, str)) or (isinstance(v, float) and not math.isfinite(v)):
            raise TypeError(f"metric {k!r} must be a finite number, bool or string, not {v!r}")


def _manifest_binding(root: str, requirement: str) -> dict:
    try:
        m = json.load(open(os.path.join(docs_dir(root), "ACCEPTANCE_MANIFEST.json")))
        e = next(x for x in m["requirements"] if x["id"] == requirement)
    except (OSError, ValueError, StopIteration, KeyError):
        return {"requirement_sha256": None, "criteria_sha256": None, "criteria_version": None}
    return {"requirement_sha256": sha256_text(e["requirement"]), "criteria_sha256": sha256_text(canonical(e["criteria"])),
            "criteria_version": len(e.get("criteria_history", []))}


def record(requirement: str, metrics: dict, *, command: str | None = None, artifacts: list[str] = (),
           notes: str = "", datasets: dict | None = None, labels: dict | None = None, root: str | None = None) -> str:
    """Write the evidence record for `requirement`; returns its path.

    metrics: JSON numbers/bools/strings, finite. datasets: {name: path} whose checksums are recorded. labels: how the
    result was obtained, e.g. {"scripted": True} for scripted reference policies or {"learned": True}; the final report
    separates results by these labels. artifacts: files the record relies on (committed files of the checkout, or
    files written under the run's output directory, at most 5 MB each). `root` only says which checkout to describe;
    it never makes a record official."""
    _check_metrics(metrics)
    root = root or crucible_root()
    fr = sys._getframe(1)                             # who called record(): the checker binds it to a call site
    cf = os.path.realpath(fr.f_code.co_filename)
    caller = {"function": fr.f_code.co_name, "line": fr.f_lineno,
              "file": os.path.relpath(cf, os.path.realpath(root)) if cf.startswith(os.path.realpath(root) + os.sep)
              else cf}
    cur = os.environ.get("PYTEST_CURRENT_TEST", "")
    phase = cur.rsplit(" (", 1)[1].rstrip(")") if " (" in cur else None
    run = official_run(root)
    arts = []
    for a in artifacts:
        p = a if os.path.isabs(a) else os.path.join(root, a)
        size = os.path.getsize(p)
        rp = os.path.realpath(p)
        if run and rp.startswith(os.path.realpath(run["out"]) + os.sep):
            rel = "@run/" + os.path.relpath(rp, os.path.realpath(run["out"]))
        else:
            rel = os.path.relpath(p, root)
        arts.append({"path": rel, "sha256": sha256_file(p), "bytes": size, "too_large": size > MAX_ARTIFACT_BYTES})
    mods, outside = imported_modules(root)
    reads = files_read(root) if run else {}
    rec = {"schema": SCHEMA, "requirement": requirement, "gate": gate_of(requirement),
           "produced_by": cur.split(" (")[0] or command or "unknown", "caller": caller, "phase": phase,
           "command": command, "run_id": run["run_id"] if run else None, "official": bool(run),
           "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "tree": tree_state(root), "environment": environment(), "identity": identity(),
           "author": os.environ.get("CRUCIBLE_V5_AGENT") or os.environ.get("CRUCIBLE_V5_AUTHOR") or "",
           "binding": _manifest_binding(root, requirement),
           "modules": mods, "modules_outside_root": outside, "files_read": reads,
           "files_read_truncated": bool(_TRUNCATED[0]) if run else False,
           "interpreter": interpreter(), "site_hooks": site_hooks(), "foreign_modules": foreign_modules(root),
           "patched_functions": patched_functions(root) + code_object_patches(root) + rebound_definitions(root)
           + substituted_modules(root),
           "python_env": python_env(),
           "executed_implementation": executed_implementation(root, requirement) if run else None,
           "datasets": dataset_checksums(datasets),
           "metrics": metrics, "artifacts": arts, "labels": dict(labels or {}), "notes": notes,
           "junit": None, "run": None}
    if run:
        d = os.path.join(run["out"], "evidence", f"gate_{gate_of(requirement)}")
    else:                                        # an ordinary test run: measured, but never filed as evidence
        d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-unofficial-evidence", gate_of(requirement))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{requirement}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path
