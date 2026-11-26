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

`measured_asserts` counts asserts whose expression references a name, attribute, call or subscript.
"""
from __future__ import annotations

import ast

BLOCKING = ("no_assert", "trivial_assert", "early_return", "swallow", "skip", "mock", "patch", "constant_record",
            "placeholder_text")
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


def _is_const_expr(n) -> bool:
    if isinstance(n, ast.Constant):
        return True
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
    return isinstance(n, ast.Dict) and n.values and all(v is not None and _is_const_expr(v) for v in n.values)


def _marker_names(dec) -> set[str]:
    d = _dotted(dec.func if isinstance(dec, ast.Call) else dec)
    out = set()
    if d.startswith("pytest.mark."):
        out.add(d.split(".")[2])
    if d in ("unittest.skip", "unittest.skipIf", "unittest.skipUnless", "skip", "skipIf"):
        out.add("skip")
    return out


def _is_record_call(n) -> bool:
    if not isinstance(n, ast.Call):
        return False
    d = _dotted(n.func)
    return d in ("record", "evidence.record") or d.endswith(".evidence.record")


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
    return {"mock_import": mock, "skip_module": skip_module}


def scan_function(fn, src_lines: list[str], mflags: dict) -> dict:
    findings: list[str] = []
    asserts = measured = 0
    raises = False
    const_names: dict[str, bool] = {}
    for n in _walk_body(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            const_names[nm] = const_names.get(nm, True) and _const_dict(n.value)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(getattr(n, "target", None), ast.Name):
            const_names[n.target.id] = False
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
            const_names[n.value.id] = False
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.attr in ("update", "setdefault", "__setitem__"):
            const_names[n.func.value.id] = False
    record_calls, fail_lines, undo_lines, patch_lines = [], [], [], []
    for n in _walk_body(fn):
        if isinstance(n, ast.Assert):
            asserts += 1
            if not _is_const_expr(n.test):
                measured += 1
        elif isinstance(n, ast.Raise):
            raises = True
        elif isinstance(n, ast.Return):
            findings.append("early_return")
        elif isinstance(n, ast.ExceptHandler):
            if all(isinstance(b, (ast.Pass, ast.Return, ast.Continue)) or
                   (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)) for b in n.body):
                findings.append("swallow")
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
            elif d in ("monkeypatch.setattr", "monkeypatch.setitem", "monkeypatch.delattr"):
                patch_lines.append(n.lineno)
            elif d == "monkeypatch.undo":
                undo_lines.append(n.lineno)
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
    for pl in patch_lines:
        if not any(u > pl for u in undo_lines):
            findings.append("patch")
    for rc in record_calls:
        metrics = rc.args[1] if len(rc.args) > 1 else next((k.value for k in rc.keywords if k.arg == "metrics"), None)
        const = _const_dict(metrics) or (isinstance(metrics, ast.Name) and const_names.get(metrics.id, False))
        if const:
            if any(fl > rc.lineno for fl in fail_lines):
                findings.append("constant_record_then_fail")
            else:
                findings.append("constant_record")
    if asserts == 0 and not raises:
        findings.append("no_assert")
    elif asserts and measured == 0 and not raises:
        findings.append("trivial_assert")
    seg = "\n".join(src_lines[fn.lineno - 1:getattr(fn, "end_lineno", fn.lineno)])
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
