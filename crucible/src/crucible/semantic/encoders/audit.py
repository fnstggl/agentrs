"""Static audits for Gate E: no creator-assigned quality scores, and no stub vectors.

`quality_fields(root)` scans the source tree (MASTER_SPEC 4.3: "Do not allow the author to directly specify its own
quality, relevance or virality score"):

1. **Creator-assignable fields** - every tool input schema of every registered kernel module (anything an actor can
   send), and every parameter of the public functions of the content and participant packages, whose name is a
   quality-like score (`quality`, `virality`, `relevance`, `popularity`, `appeal`, `hotness`, ... with optional
   `creator_`/`author_`/`declared_`/`claimed_`/`self_` prefixes and `_score` suffixes).
2. **Reads** - every read of such a field anywhere in `src/crucible`: `x["quality"]`, `x.get("quality")`,
   `x.quality`, `"quality" in x`, keyword `quality=` in a call, and string keys of dict literals.

A violation is a read of a field that is creator-assignable (anywhere), or any read of a quality-like field inside the
content and participant packages (`PARTICIPANT_SCOPE`: the v5 social internet, population, semantic participants,
multi-agent, civilizations and domains, the 4.0 consumer population and the content services). Reads elsewhere (for
example the 4.0 hidden-mechanism engines, where `quality` is a costly production decision of a simulated firm, or
evaluator reports) are listed separately with their location, so the classification can be checked by a reviewer.
A line that uses such a name for something else (Pillow's JPEG `quality` compression level) carries the pragma
`# not-a-content-score: <reason>`; those lines are reported under `exempt` with the reason, never silently dropped.

`stub_check(vectors)` flags stub-like embeddings: non-finite, zero, constant, or identical for different content.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Iterable

import numpy as np

SCORE_NAMES = ("quality", "virality", "viral", "relevance", "relevancy", "popularity", "appeal", "hotness",
               "interestingness", "attractiveness", "catchiness", "trendiness", "goodness")
FORBIDDEN = re.compile(r"^(?:(?:creator|author|declared|claimed|self|stated|own|content|item|post|video|article)_)?"
                       r"(?:" + "|".join(SCORE_NAMES) + r")(?:_score|_rating|_level)?$", re.I)
PARTICIPANT_SCOPE = ("social/", "population/", "semantic/", "multiagent/", "civ/", "domains/", "experience/",
                     "economy/consumers.py", "economy/shoppers.py", "services/forum.py", "services/search.py",
                     "services/ads.py", "services/docs.py", "services/net.py", "services/mail.py",
                     "services/market.py", "actors/")
SKIP_DIRS = ("__pycache__",)
PRAGMA = "# not-a-content-score:"     # a line naming such a field for another meaning (with the reason after it)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def in_scope(rel: str) -> bool:
    return any(rel.startswith(p) if p.endswith("/") else rel == p for p in PARTICIPANT_SCOPE)


def _schema_fields(schema, path="") -> Iterable[tuple[str, str]]:
    if not isinstance(schema, dict):
        return
    for name, sub in (schema.get("properties") or {}).items():
        yield name, f"{path}.{name}" if path else name
        yield from _schema_fields(sub, f"{path}.{name}" if path else name)
    if isinstance(schema.get("items"), dict):
        yield from _schema_fields(schema["items"], path + "[]")
    if isinstance(schema.get("additionalProperties"), dict):
        yield from _schema_fields(schema["additionalProperties"], path + "{}")


def endpoint_fields() -> list[dict]:
    """Quality-like property names in the input schema of every endpoint of every registered module."""
    from ... import load_all_modules
    from ...core.kernel import MODULES
    load_all_modules()
    out = []
    for mname, cls in sorted(MODULES.items()):
        for attr in dir(cls):
            fn = getattr(cls, attr, None)
            spec = getattr(fn, "__crucible_endpoint__", None)
            if not spec:
                continue
            for name, path in _schema_fields(spec.get("params") or {}):
                if FORBIDDEN.match(name):
                    out.append({"module": mname, "endpoint": f"{cls.host or mname}.{spec['name']}", "field": name,
                                "path": path})
    return out


class _Reads(ast.NodeVisitor):
    def __init__(self):
        self.reads: list[tuple[int, str, str]] = []
        self.params: list[tuple[int, str, str]] = []

    def _hit(self, node, name, kind):
        if isinstance(name, str) and FORBIDDEN.match(name):
            self.reads.append((getattr(node, "lineno", 0), name, kind))

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant):
            self._hit(node, node.slice.value, "subscript" if isinstance(node.ctx, ast.Load) else "subscript_store")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self._hit(node, node.attr, "attribute")
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("get", "pop", "setdefault") and node.args and \
                isinstance(node.args[0], ast.Constant):
            self._hit(node, node.args[0].value, f".{f.attr}()")
        if isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2 and \
                isinstance(node.args[1], ast.Constant):
            self._hit(node, node.args[1].value, "getattr()")
        for kw in node.keywords:
            if kw.arg:
                self._hit(node, kw.arg, "keyword")
        self.generic_visit(node)

    def visit_Compare(self, node):
        if isinstance(node.left, ast.Constant) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
            self._hit(node, node.left.value, "in")
        self.generic_visit(node)

    def visit_Dict(self, node):
        for k in node.keys:
            if isinstance(k, ast.Constant):
                self._hit(k, k.value, "dict_key")
        self.generic_visit(node)

    def _fn(self, node):
        if not node.name.startswith("_"):
            for a in node.args.args + node.args.kwonlyargs:
                if FORBIDDEN.match(a.arg):
                    self.params.append((node.lineno, a.arg, node.name))
        self.generic_visit(node)

    visit_FunctionDef = _fn
    visit_AsyncFunctionDef = _fn


def scan_source(text: str) -> _Reads:
    v = _Reads()
    v.visit(ast.parse(text))
    return v


def quality_fields(root: str | None = None, *, include_endpoints: bool = True) -> dict:
    """Scan `src/crucible` (or `root`); returns the counts, the creator-assignable fields, every read and the
    violations. `creator_assigned_quality_fields_read` is the Gate E.5 metric."""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assignable = endpoint_fields() if include_endpoints else []
    reads, exempt, files = [], [], 0
    for d, dirs, fnames in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for fn in sorted(fnames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(d, fn)
            rel = _rel(p, root)
            if rel.startswith("semantic/encoders/audit.py"):
                continue          # this file names the fields it looks for
            files += 1
            try:
                text = open(p, encoding="utf-8").read()
                v = scan_source(text)
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue
            lines = text.splitlines()
            for line, name, kind in v.reads:
                src = lines[line - 1] if 0 < line <= len(lines) else ""
                if PRAGMA in src:
                    exempt.append({"file": rel, "line": line, "field": name,
                                   "reason": src.split(PRAGMA, 1)[1].strip()[:200]})
                    continue
                reads.append({"file": rel, "line": line, "field": name, "kind": kind, "participant_scope": in_scope(rel)})
            if in_scope(rel):
                for line, name, func in v.params:
                    assignable.append({"module": rel, "endpoint": f"function {func}()", "field": name, "path": name})
    names = {a["field"].lower() for a in assignable}
    violations = [r for r in reads if r["participant_scope"] or r["field"].lower() in names]
    other = [r for r in reads if r not in violations]
    return {"files_scanned": files, "creator_assignable_fields": assignable, "reads": len(reads),
            "violations": violations, "out_of_scope_reads": other, "exempt": exempt,
            "out_of_scope_files": sorted({r["file"] for r in other}),
            "creator_assigned_quality_fields_read": len(violations),
            "creator_assignable_count": len(assignable)}


# ============================================================================================== stub vectors
def stub_check(vectors: dict[str, np.ndarray], distinct_groups: dict[str, str] | None = None) -> dict:
    """Flags vectors that a real encoder would not produce: non-finite, (near) zero norm, constant entries, or two
    artifacts with different content (different `distinct_groups` labels) mapped to the same vector."""
    problems = []
    names = list(vectors)
    for n in names:
        v = np.asarray(vectors[n], dtype=np.float64)
        if not np.all(np.isfinite(v)):
            problems.append(f"{n}: non-finite")
        elif np.linalg.norm(v) < 1e-6:
            problems.append(f"{n}: zero vector")
        elif float(np.std(v)) < 1e-6:
            problems.append(f"{n}: constant vector")
    if distinct_groups:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if distinct_groups.get(a) != distinct_groups.get(b):
                    va, vb = np.asarray(vectors[a]), np.asarray(vectors[b])
                    if np.allclose(va, vb, atol=1e-6):
                        problems.append(f"{a} and {b}: identical vectors for different content")
    return {"stub_vectors_detected": len(problems), "problems": problems[:50], "checked": len(names)}
