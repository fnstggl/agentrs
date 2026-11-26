"""Static integrity scan of test functions (used by the checker for producing tests and by the Gate X.7 detector).

For each test function it reports:

* `no_assert`     - the body (nested functions excluded) has no assert, pytest.fail, pytest.raises or raise;
* `trivial_assert` - it asserts, but only on constants (`assert True`, `assert 1 == 1`);
* `early_return`  - a `return` in the test body;
* `swallow`       - an `except` handler whose body only passes / returns / continues;
* `skip`          - pytest.skip / importorskip / xfail calls, skip or xfail markers, unittest.skip;
* `mock`          - unittest.mock / mock / pytest-mock (MagicMock, Mock, patch, mocker);
* `patch`         - monkeypatch.setattr / setitem not undone (monkeypatch.undo()) before the record() call;
* `constant_record` - an evidence `record()` whose metrics are all literal constants (directly or through a name
  bound only to such a dict) without a following unconditional pytest.fail (an honest "nothing measured" record that
  then fails is reported as `constant_record_then_fail`, which is not a placeholder);
* `placeholder_text` - TODO / FIXME / XXX / placeholder / NotImplementedError in the function's source.

Findings that come from outside the test body apply to every test of the module, because fixtures and helpers act on
any test that uses them: `patch` (monkeypatch without undo, `setattr`/`delattr`, attribute, item or `sys.modules`
assignment on an imported local module - `crucible` or any module that is neither stdlib nor installed - in a fixture,
a helper or at module level), `swallow` (`contextlib.suppress`, or a broad `except` - bare, Exception, BaseException -
whose handler never raises) and `mock`. Constant metrics are also recognised through module-level constant dicts and
`dict(...)` calls with constant arguments.

`measured_asserts` counts asserts whose expression references a name, attribute, call or subscript.

`harness_findings(src)` scans a conftest.py (or a pytest plugin module) for what can change test outcomes or the
measured code without touching the test file: outcome-rewriting hooks (`pytest_pyfunc_call`, report and collection
hooks, session-finish hooks, a `pytest_runtest_makereport` that mutates or replaces the report), plugin registration,
`pytest_plugins`, `collect_ignore`, import hooks (`sys.meta_path` / `sys.path_hooks`), patches in fixtures or at
module level, and mocks.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import sysconfig

BLOCKING = ("no_assert", "trivial_assert", "early_return", "swallow", "skip", "mock", "patch", "constant_record",
            "placeholder_text")
BROAD_EXCEPTIONS = {"Exception", "BaseException"}
# hooks through which a conftest or plugin decides what runs, what is collected or what is reported
OUTCOME_HOOKS = {"pytest_pyfunc_call", "pytest_runtest_call", "pytest_runtest_protocol", "pytest_runtest_setup",
                 "pytest_runtest_teardown", "pytest_runtest_logreport", "pytest_report_teststatus",
                 "pytest_collectreport", "pytest_make_collect_report", "pytest_pycollect_makeitem",
                 "pytest_pycollect_makemodule", "pytest_collect_file", "pytest_ignore_collect", "pytest_itemcollected",
                 "pytest_deselected", "pytest_collection_modifyitems", "pytest_collection_finish",
                 "pytest_sessionfinish", "pytest_unconfigure", "pytest_terminal_summary", "pytest_exception_interact",
                 "pytest_internalerror", "pytest_fixture_setup", "pytest_fixture_post_finalizer",
                 "pytest_runtest_logfinish", "pytest_generate_tests", "pytest_keyboard_interrupt"}
PLACEHOLDER_WORDS = ("TODO", "FIXME", "XXX", "placeholder", "NotImplementedError")
SKIP_CALLS = {"skip", "importorskip", "xfail"}
SKIP_MARKS = {"skip", "skipif", "xfail"}
MOCK_NAMES = {"MagicMock", "Mock", "AsyncMock", "NonCallableMock", "create_autospec", "patch"}


def _walk_body(node):
    """Nodes of a function body, not descending into nested function or class definitions."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _dotted(n) -> str:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


PURE_BUILTINS = {"int", "float", "str", "bool", "len", "abs", "round", "min", "max", "sum", "sorted", "tuple", "list"}


def _is_const_expr(n) -> bool:
    if isinstance(n, ast.Constant):
        return True
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in PURE_BUILTINS:
        return all(_is_const_expr(a) for a in n.args) and all(_is_const_expr(k.value) for k in n.keywords)
    if isinstance(n, ast.UnaryOp):
        return _is_const_expr(n.operand)
    if isinstance(n, ast.BinOp):
        return _is_const_expr(n.left) and _is_const_expr(n.right)
    if isinstance(n, ast.Compare):
        return _is_const_expr(n.left) and all(_is_const_expr(c) for c in n.comparators)
    if isinstance(n, ast.BoolOp):
        return all(_is_const_expr(v) for v in n.values)
    if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_const_expr(e) for e in n.elts)
    return False


def _const_dict(n) -> bool:
    if isinstance(n, ast.Call) and _dotted(n.func) == "dict" and not n.args and n.keywords:
        return all(k.arg is not None and _is_const_expr(k.value) for k in n.keywords)
    return isinstance(n, ast.Dict) and bool(n.values) and all(k is not None and v is not None and _is_const_expr(v)
                                                              for k, v in zip(n.keys, n.values))


def _marker_names(dec) -> set[str]:
    d = _dotted(dec.func if isinstance(dec, ast.Call) else dec)
    out = set()
    if d.startswith("pytest.mark."):
        out.add(d.split(".")[2])
    if d in ("unittest.skip", "unittest.skipIf", "unittest.skipUnless", "skip", "skipIf"):
        out.add("skip")
    return out


_LOCAL_CACHE: dict[str, bool] = {}


def _installed_dirs() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    return tuple(os.path.realpath(paths[k]) for k in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(k))


def is_local_module(root: str) -> bool:
    """Is the top-level module `root` part of the code under test (crucible, the tests' helpers, the measured
    packages) rather than the standard library or an installed third-party package?"""
    if root in _LOCAL_CACHE:
        return _LOCAL_CACHE[root]
    local = True
    if root != "crucible" and (root in getattr(sys, "stdlib_module_names", ()) or root in sys.builtin_module_names):
        local = False
    elif root != "crucible":
        try:
            spec = importlib.util.find_spec(root)
        except (ImportError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None) if spec else None
        if origin and origin not in ("built-in", "frozen"):
            o = os.path.realpath(origin)
            local = not any(o.startswith(d + os.sep) for d in _installed_dirs())
        elif spec is not None:
            local = False
    _LOCAL_CACHE[root] = local
    return local


def imported_local_names(tree: ast.Module) -> set[str]:
    """Names bound by import statements (anywhere in the module) to local modules or objects from them."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if is_local_module(a.name.split(".")[0]):
                    out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level or is_local_module((n.module or "").split(".")[0]):
                out.update(a.asname or a.name for a in n.names if a.name != "*")
    return out


def _root_name(n) -> str | None:
    while isinstance(n, (ast.Attribute, ast.Subscript)):
        n = n.value
    return n.id if isinstance(n, ast.Name) else None


def _first_line(node, pred) -> int | None:
    lines = [n.lineno for n in _walk_body(node) if pred(n)]
    return min(lines) if lines else None


def patch_findings(node, local: set[str], *, until: int | None = None, cleanup_anywhere: bool = False) -> list[str]:
    """Patches of the code under test inside `node` (not descending into nested definitions) still in force at line
    `until` (default: the end of `node`). An attribute patch (`X.attr = ...`, `setattr(X, ...)` on an imported local
    module or object) counts unless the same attribute is assigned again (restored) before `until`; an item stored
    into an imported local registry counts unless the registry is popped / deleted / cleared again in `node` (a
    temporary negative control); monkeypatch calls count unless `monkeypatch.undo()` follows; `sys.modules` stores
    under a local module name and `builtins` stores always count."""
    module = isinstance(node, ast.Module)
    nodes = [x for top in node.body if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             for x in [top, *_walk_body(top)]] if module else list(_walk_body(node))
    end = until if until is not None else float("inf")
    attr: dict[str, list[int]] = {}
    items: dict[str, list[int]] = {}
    removals: dict[str, list[int]] = {}
    out, patch_lines, undo_lines = [], [], []
    for n in nodes:
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, ast.Delete):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and _root_name(t) in local:
                    removals.setdefault(_dotted(t.value), []).append(n.lineno)
        for t in targets:
            for sub in ast.walk(t):
                if not isinstance(getattr(sub, "ctx", None), ast.Store):
                    continue
                r = _root_name(sub)
                if isinstance(sub, ast.Attribute) and r in local:
                    attr.setdefault(_dotted(sub), []).append(n.lineno)
                elif isinstance(sub, ast.Subscript) and r in local:
                    items.setdefault(_dotted(sub.value), []).append(n.lineno)
                elif r == "builtins" and isinstance(sub, (ast.Attribute, ast.Subscript)):
                    out.append("patch")
                elif isinstance(sub, ast.Subscript) and _dotted(sub.value) == "sys.modules" and \
                        isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str) and \
                        sub.slice.value.split(".")[0] in local | {"crucible"}:
                    out.append("patch")
        if isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d in ("setattr", "delattr") and n.args and _root_name(n.args[0]) in local:
                name = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else "*"
                attr.setdefault(f"{_dotted(n.args[0])}.{name}", []).append(n.lineno)
            elif d in ("monkeypatch.setattr", "monkeypatch.setitem", "monkeypatch.delattr", "monkeypatch.delitem"):
                patch_lines.append(n.lineno)
            elif d == "monkeypatch.undo":
                undo_lines.append(n.lineno)
            elif isinstance(n.func, ast.Attribute) and n.func.attr in ("pop", "popitem", "clear", "__delitem__") and \
                    _root_name(n.func.value) in local:
                removals.setdefault(_dotted(n.func.value), []).append(n.lineno)
    for key, lines in attr.items():
        first = min(lines)
        if first <= end and not any(first < x < end for x in lines):
            out.append("patch")
    for key, lines in items.items():
        first = min(lines)
        if first <= end and not any(x > first and (until is None or cleanup_anywhere or x < end)
                                    for x in removals.get(key, [])):
            out.append("patch")
    for pl in patch_lines:
        if pl <= end and not any(u > pl for u in undo_lines):
            out.append("patch")
    return out


def _uses(handler: ast.ExceptHandler) -> bool:
    return bool(handler.name) and any(isinstance(x, ast.Name) and x.id == handler.name
                                      for b in handler.body for x in ast.walk(b))


def swallow_findings(node, *, helper: bool = False) -> list[str]:
    """Errors made to disappear inside `node` (not descending into nested definitions). In a test body: a handler
    that only passes / returns / continues, `contextlib.suppress`, or a broad handler (bare, Exception,
    BaseException) that neither re-raises nor uses the exception it caught. In a helper or fixture only the broad
    forms count (a helper may legitimately turn a narrow, expected error into a value)."""
    out = []
    for n in _walk_body(node) if not isinstance(node, ast.Module) else ast.walk(node):
        if isinstance(n, ast.Call) and _dotted(n.func) in ("contextlib.suppress", "suppress"):
            names = {_dotted(a).split(".")[-1] for a in n.args}
            if not helper or not n.args or names & BROAD_EXCEPTIONS:
                out.append("swallow")
        elif isinstance(n, ast.ExceptHandler):
            only = all(isinstance(b, (ast.Pass, ast.Return, ast.Continue)) or
                       (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)) for b in n.body)
            names = set()
            if n.type is None:
                names.add("BaseException")
            else:
                for t in ([n.type] if not isinstance(n.type, ast.Tuple) else n.type.elts):
                    names.add(_dotted(t).split(".")[-1])
            broad = bool(names & BROAD_EXCEPTIONS)
            reraises = any(isinstance(x, ast.Raise) or (isinstance(x, ast.Call) and _dotted(x.func) == "pytest.fail")
                           for b in n.body for x in ast.walk(b))
            if reraises:
                continue
            if (only and not helper) or (broad and not _uses(n)) or (only and broad):
                out.append("swallow")
    return out


def _is_fixture(fn) -> bool:
    return any(_dotted(d.func if isinstance(d, ast.Call) else d).split(".")[-1] == "fixture" for d in fn.decorator_list)


def _is_record_call(n) -> bool:
    if not isinstance(n, ast.Call):
        return False
    d = _dotted(n.func)
    return d in ("record", "evidence.record") or d.endswith(".evidence.record")


def _yield_line(fn) -> int | None:
    return _first_line(fn, lambda n: isinstance(n, (ast.Yield, ast.YieldFrom)))


def _referenced(fn) -> tuple[set[str], set[str]]:
    """(names called or referenced in `fn`'s body, including nested definitions; its argument names)."""
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    names |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
              and n.value.id == "self"}
    args = {a.arg for a in [*fn.args.args, *fn.args.kwonlyargs]}
    return names, args


def _outside_findings(fn, local: set[str], mock: bool, *, fixture: bool) -> list[str]:
    """Findings of a helper or fixture (and the definitions nested in it)."""
    out = []
    for sub in [fn, *[x for x in ast.walk(fn) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and x is not fn]]:
        until = _yield_line(sub) if fixture and sub is fn else None
        out += patch_findings(sub, local, until=until, cleanup_anywhere=fixture)
        out += swallow_findings(sub, helper=True)
        if any(isinstance(x, ast.Call) and _dotted(x.func).split(".")[-1] in MOCK_NAMES and
               (_dotted(x.func).startswith(("mock", "unittest")) or mock) for x in ast.walk(sub)):
            out.append("mock")
    if "mocker" in [a.arg for a in fn.args.args]:
        out.append("mock")
    return out


def module_flags(tree: ast.Module) -> dict:
    mock = False
    skip_module = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            if any(a.name in ("mock", "unittest.mock", "pytest_mock") for a in n.names):
                mock = True
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "") in ("mock", "unittest.mock", "pytest_mock"):
                mock = True
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets):
            for sub in ast.walk(n.value):
                if isinstance(sub, (ast.Attribute, ast.Call)) and _marker_names(sub) & SKIP_MARKS:
                    skip_module = True
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and _dotted(n.value.func) in (
                "pytest.skip", "pytest.importorskip"):
            skip_module = True
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _dotted(n.value.func) == "pytest.importorskip":
            skip_module = True
    local = imported_local_names(tree)
    consts: dict[str, bool] = {}
    for n in tree.body:                         # module-level constant metric dicts
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            consts[nm] = consts.get(nm, True) and _const_dict(n.value)
    for n in ast.walk(tree):                    # ... unless something mutates them
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
            consts[n.value.id] = False
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.attr in ("update", "setdefault", "__setitem__", "pop", "clear"):
            consts[n.func.value.id] = False
    for n in tree.body:                         # helpers whose every return is a constant metric dict
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test"):
            local_const: dict[str, bool] = {}
            for x in _walk_body(n):
                if isinstance(x, ast.Assign) and len(x.targets) == 1 and isinstance(x.targets[0], ast.Name):
                    local_const[x.targets[0].id] = local_const.get(x.targets[0].id, True) and _const_dict(x.value)
            rets = [x.value for x in _walk_body(n) if isinstance(x, ast.Return)]
            if rets and all(r is not None and (_const_dict(r) or (isinstance(r, ast.Name) and
                                                                  local_const.get(r.id, consts.get(r.id, False))))
                            for r in rets):
                consts[f"{n.name}()"] = True
    # what fixtures, helpers and classes of the module do acts on the tests that use them
    helpers: dict[str, tuple[list[str], set[str]]] = {}
    fixtures: dict[str, tuple[list[str], set[str], bool]] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names, args = _referenced(n)
            if _is_fixture(n):
                autouse = any(isinstance(d, ast.Call) and any(k.arg == "autouse" and isinstance(k.value, ast.Constant)
                                                              and k.value.value for k in d.keywords)
                              for d in n.decorator_list)
                fixtures[n.name] = (_outside_findings(n, local, mock, fixture=True), names | args, autouse)
            elif not n.name.startswith("test"):
                helpers[n.name] = (_outside_findings(n, local, mock, fixture=False), names)
        elif isinstance(n, ast.ClassDef):
            found, refs = [], set()
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("test"):
                    found += _outside_findings(m, local, mock, fixture=_is_fixture(m))
                    names, args = _referenced(m)
                    refs |= names
                    if _is_fixture(m):
                        fixtures[m.name] = (_outside_findings(m, local, mock, fixture=True), names | args, False)
            helpers[n.name] = (found, refs)
    return {"mock_import": mock, "skip_module": skip_module, "local": local, "consts": consts,
            "module_level": sorted(set(patch_findings(tree, local))), "helpers": helpers, "fixtures": fixtures}


def used_findings(fn, mflags: dict) -> list[str]:
    """Findings of the module-level code, of every autouse fixture and of the helpers, classes and fixtures a test
    uses (transitively)."""
    helpers, fixtures = mflags.get("helpers", {}), mflags.get("fixtures", {})
    names, args = _referenced(fn)
    todo = list(names | args) + [k for k, v in fixtures.items() if v[2]]
    seen, out = set(), list(mflags.get("module_level", []))
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen.add(k)
        if k in fixtures:
            out += fixtures[k][0]
            todo += list(fixtures[k][1])
        if k in helpers:
            out += helpers[k][0]
            todo += list(helpers[k][1])
    return out


def _const_call(n, mflags: dict) -> bool:
    """A call of a module helper whose every return is a constant metric dict, with constant arguments."""
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
        bool(mflags.get("consts", {}).get(f"{n.func.id}()")) and \
        all(_is_const_expr(a) for a in n.args) and all(_is_const_expr(k.value) for k in n.keywords)


def scan_function(fn, src_lines: list[str], mflags: dict) -> dict:
    findings: list[str] = []
    asserts = measured = 0
    raises = False
    const_names: dict[str, bool] = {}
    assigned: set[str] = set()
    for n in _walk_body(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            assigned.add(nm)
            const_names[nm] = const_names.get(nm, True) and (_const_dict(n.value) or _const_call(n.value, mflags))
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(getattr(n, "target", None), ast.Name):
            const_names[n.target.id] = False
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
            const_names[n.value.id] = False
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.attr in ("update", "setdefault", "__setitem__"):
            const_names[n.func.value.id] = False
    record_calls, fail_lines = [], []
    for n in _walk_body(fn):
        if isinstance(n, ast.Assert):
            asserts += 1
            if not _is_const_expr(n.test):
                measured += 1
        elif isinstance(n, ast.Raise):
            raises = True
        elif isinstance(n, ast.Return):
            findings.append("early_return")
        elif isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d in ("pytest.fail",):
                fail_lines.append(n.lineno)
                raises = True
            elif d in ("pytest.raises", "pytest.warns"):
                raises = True
                measured += 1
            elif d.startswith("pytest.") and d.split(".", 1)[1] in SKIP_CALLS:
                findings.append("skip")
            elif d.split(".")[-1] in MOCK_NAMES and (d.startswith("mock") or d.startswith("unittest") or
                                                     d in MOCK_NAMES or mflags.get("mock_import")):
                findings.append("mock")
            if _is_record_call(n):
                record_calls.append(n)
    args = [a.arg for a in fn.args.args]
    if "mocker" in args:
        findings.append("mock")
    if mflags.get("mock_import"):
        findings.append("mock")
    for dec in fn.decorator_list:
        if _marker_names(dec) & SKIP_MARKS:
            findings.append("skip")
    if mflags.get("skip_module"):
        findings.append("skip")
    local = mflags.get("local", set())
    first_record = min((rc.lineno for rc in record_calls), default=None)
    findings += patch_findings(fn, local, until=first_record) + swallow_findings(fn)
    for sub in ast.walk(fn):                     # helpers defined inside the test act on it too
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not fn:
            findings += patch_findings(sub, local) + swallow_findings(sub, helper=True)
    findings += used_findings(fn, mflags)
    mconsts = mflags.get("consts", {})
    for rc in record_calls:
        metrics = rc.args[1] if len(rc.args) > 1 else next((k.value for k in rc.keywords if k.arg == "metrics"), None)
        const = _const_dict(metrics) or _const_call(metrics, mflags) or (isinstance(metrics, ast.Name) and (
            const_names.get(metrics.id, False) if metrics.id in assigned else mconsts.get(metrics.id, False)))
        if const:
            if any(fl > rc.lineno for fl in fail_lines):
                findings.append("constant_record_then_fail")
            else:
                findings.append("constant_record")
    if asserts == 0 and not raises:
        findings.append("no_assert")
    elif asserts and measured == 0 and not raises:
        findings.append("trivial_assert")
    doc = fn.body[0] if (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
                         and isinstance(fn.body[0].value.value, str)) else None
    header = [fn.lineno] + [getattr(n, "end_lineno", fn.lineno) for n in
                            [*fn.args.args, *fn.args.kwonlyargs, fn.returns] if n is not None]
    start = (doc.end_lineno if doc is not None else max(header)) + 1   # the body: not the name, not the docstring
    seg = "\n".join(src_lines[start - 1:getattr(fn, "end_lineno", fn.lineno)])
    for w in PLACEHOLDER_WORDS:
        if w in seg:
            findings.append("placeholder_text")
            break
    return {"name": fn.name, "line": fn.lineno, "asserts": asserts, "measured_asserts": measured,
            "records": len(record_calls), "findings": sorted(set(findings))}


def scan_source(src: str) -> list[dict]:
    """Every test function (module level `test_*`, methods `test_*` of `Test*` classes) with its findings."""
    tree = ast.parse(src)
    lines = src.splitlines()
    mflags = module_flags(tree)
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            out.append({"class": None, **scan_function(n, lines, mflags)})
        elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
            cls_skip = any(_marker_names(d) & SKIP_MARKS for d in n.decorator_list)
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test"):
                    r = {"class": n.name, **scan_function(m, lines, mflags)}
                    if cls_skip:
                        r["findings"] = sorted(set(r["findings"]) | {"skip"})
                    out.append(r)
    return out


def blocking(findings: list[str]) -> list[str]:
    return [f for f in findings if f in BLOCKING]


def _makereport_mutates(fn) -> bool:
    """Does a pytest_runtest_makereport implementation change or replace the report (rather than only read it)?"""
    reports = {a.arg for a in fn.args.args if a.arg == "call"}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _dotted(n.value.func).endswith("get_result"):
            reports.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Yield):
            reports.update(t.id for t in n.targets if isinstance(t, ast.Name))       # the hookwrapper outcome
    for n in ast.walk(fn):
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)) and \
                _root_name(n) in reports:
            return True
        if isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d.split(".")[-1] in ("force_result", "force_exception", "_result", "__setattr__") or \
                    (d in ("setattr", "delattr", "object.__setattr__") and n.args and _root_name(n.args[0]) in reports):
                return True
        if isinstance(n, ast.Return) and n.value is not None:
            return True
    return False


def harness_findings(src: str) -> list[str]:
    """What a conftest.py or plugin module does that can change outcomes or the code under test (see module doc)."""
    tree = ast.parse(src)
    local = imported_local_names(tree)
    out: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name in OUTCOME_HOOKS:
                out.add(f"hook:{n.name}")
            elif n.name == "pytest_runtest_makereport" and _makereport_mutates(n):
                out.add("hook:pytest_runtest_makereport (changes the report)")
            if _is_fixture(n):
                for sub in [n, *[x for x in ast.walk(n) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 and x is not n]]:
                    if patch_findings(sub, local):
                        out.add(f"fixture_patch:{n.name}")
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ("pytest_plugins", "collect_ignore", "collect_ignore_glob"):
                    out.add(t.id)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d.endswith("pluginmanager.register") or d.endswith("pluginmanager.import_plugin") or \
                    d.endswith("config.pluginmanager.consider_module"):
                out.add("plugin_register")
            if any(x in d for x in ("meta_path.", "path_hooks.", "path_importer_cache.")):
                out.add("import_hook")
            if d.split(".")[-1] in MOCK_NAMES and (d.startswith(("mock", "unittest")) or d in MOCK_NAMES):
                out.add("mock")
        elif isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store) and \
                any(x in _dotted(n if isinstance(n, ast.Attribute) else n.value) for x in ("meta_path", "path_hooks")):
            out.add("import_hook")
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in ("exitstatus", "testsfailed")
                                               for t in n.targets):
            out.add("exit_status")
    if module_flags(tree)["mock_import"]:
        out.add("mock")
    if patch_findings(tree, local):
        out.add("module_patch")
    return sorted(out)
