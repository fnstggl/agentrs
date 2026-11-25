"""Code / software encoder: structured features of a source tree plus the text encoder over what the code declares.

Accepted inputs: one source file; a zip or (gzipped) tar archive; or JSON `{"files": {path: text}}` / `{path: text}`
(the shape of a `code` repository snapshot). At most 200 files and 2 MiB of text are read (archive members are size-
checked before extraction, so a compressed bomb cannot exhaust memory).

* Declaration summary -> text encoder ([384]): file paths, module docstrings, the names of classes, functions and
  methods (split into words: `parse_csv` / `parseCsv` -> "parse csv"), their docstrings, comments, imported modules
  and declared service endpoints/routes. Local variable names and formatting are not part of it, so renaming locals
  or reformatting does not change what the code is about.
* Structured features ([32], bounded): language shares, size, definition densities, comment and docstring ratios,
  tests, manifests, declared endpoints, branching per function, nesting depth, imports, async and I/O use, and a
  10-bucket signed hash of imported module names. Python is parsed with `ast`; other languages with regular
  expressions.

Output: concat(sqrt(0.75) * summary block, sqrt(0.25) * structure block): unit norm, float32 [416].
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import posixpath
import re
import tarfile
import zipfile
from typing import Sequence

import numpy as np

from .base import Encoder, block, register
from .text import TEXT

MAX_FILES, MAX_TOTAL, MAX_FILE = 200, 2 * 1024 * 1024, 256 * 1024
SUMMARY_CHARS = 6000
W_TEXT, W_STRUCT = math.sqrt(0.75), math.sqrt(0.25)

LANGS = {".py": "python", ".pyi": "python", ".js": "js", ".mjs": "js", ".ts": "js", ".tsx": "js", ".jsx": "js",
         ".html": "html", ".htm": "html", ".css": "css", ".json": "data", ".yaml": "data", ".yml": "data",
         ".toml": "data", ".md": "text", ".txt": "text", ".rst": "text", ".c": "native", ".h": "native",
         ".cc": "native", ".cpp": "native", ".hpp": "native", ".go": "native", ".rs": "native", ".java": "native",
         ".kt": "native", ".rb": "other", ".php": "other", ".sh": "other", ".sql": "other"}
LANG_ORDER = ("python", "js", "html", "css", "data", "text", "native", "other")


# ---------------------------------------------------------------------------------------------- reading the tree
def read_tree(data: bytes) -> dict[str, str]:
    """path -> text for the supported input shapes (bounded)."""
    files: dict[str, str] = {}
    total = 0

    def add(path: str, raw: bytes | str) -> None:
        nonlocal total
        if len(files) >= MAX_FILES or total >= MAX_TOTAL:
            return
        text = raw if isinstance(raw, str) else raw[:MAX_FILE].decode("utf-8", errors="replace")
        text = text[:MAX_FILE]
        path = posixpath.normpath("/" + str(path).replace("\\", "/")).lstrip("/")[:300]
        files[path] = text
        total += len(text)

    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for zi in sorted(zf.infolist(), key=lambda z: z.filename)[:MAX_FILES * 4]:
                if zi.is_dir() or zi.file_size > MAX_FILE or total + zi.file_size > MAX_TOTAL:
                    continue
                add(zi.filename, zf.read(zi))
        return files
    if data[:2] == b"\x1f\x8b" or (len(data) > 262 and data[257:262] == b"ustar"):
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                for m in sorted(tf.getmembers(), key=lambda m: m.name)[:MAX_FILES * 4]:
                    if not m.isfile() or m.size > MAX_FILE or total + m.size > MAX_TOTAL:
                        continue
                    fh = tf.extractfile(m)
                    if fh is not None:
                        add(m.name, fh.read(MAX_FILE))
            return files
        except tarfile.TarError:
            pass
    head = data[:1].strip() or data.lstrip()[:1]
    if head in (b"{",):
        try:
            j = json.loads(data.decode("utf-8", errors="replace"))
            src = j.get("files", j) if isinstance(j, dict) else None
            if isinstance(src, dict) and src and all(isinstance(v, str) for v in src.values()):
                for p in sorted(src):
                    add(p, src[p])
                return files
        except ValueError:
            pass
    add(guess_name(data), data)
    return files


def guess_name(data: bytes) -> str:
    s = data[:4096].decode("utf-8", errors="replace")
    if s.startswith("#!") and "python" in s.splitlines()[0]:
        return "main.py"
    if re.search(r"^\s*(def |class |import |from \S+ import )", s, re.M):
        return "main.py"
    if re.search(r"<html|<!doctype", s, re.I):
        return "index.html"
    if re.search(r"\b(function\s+\w+|const\s+\w+\s*=|=>|require\(|export\s+)", s):
        return "main.js"
    if re.search(r"#include\s*<|\bint\s+main\s*\(", s):
        return "main.c"
    if re.search(r"^\s*(fn |use |pub fn)", s, re.M):
        return "main.rs"
    if re.search(r"^\s*(package |func )", s, re.M):
        return "main.go"
    return "main.txt"


def lang_of(path: str) -> str:
    return LANGS.get(posixpath.splitext(path)[1].lower(), "other")


def words(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return " ".join(p for p in re.split(r"[_\W]+", s) if p).lower()


# ---------------------------------------------------------------------------------------------- per-language facts
class _PyFacts(ast.NodeVisitor):
    def __init__(self):
        self.defs, self.classes, self.docs, self.imports, self.routes = [], [], [], [], []
        self.branches = 0
        self.funcs = 0
        self.max_depth = 0
        self.async_ = 0
        self._depth = 0

    def generic_visit(self, node):
        nest = isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith))
        if nest:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
        if isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp, ast.ExceptHandler, ast.BoolOp, ast.comprehension)):
            self.branches += 1
        super().generic_visit(node)
        if nest:
            self._depth -= 1

    def _fn(self, node):
        self.funcs += 1
        self.defs.append(node.name)
        d = ast.get_docstring(node)
        if d:
            self.docs.append(d)
        for dec in node.decorator_list:
            for c in ast.walk(dec):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and len(c.value) < 80:
                    self.routes.append(c.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._fn(node)

    def visit_AsyncFunctionDef(self, node):
        self.async_ += 1
        self._fn(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        d = ast.get_docstring(node)
        if d:
            self.docs.append(d)
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports += [a.name for a in node.names]

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)


_RX = {
    "def": re.compile(r"\b(?:function\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
                      r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>|\bfn\s+(\w+)|\bfunc\s+(?:\([^)]*\)\s*)?(\w+)|"
                      r"\b(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^;{]*\)\s*\{)"),
    "class": re.compile(r"\b(?:class|struct|interface|trait|enum)\s+([A-Za-z_]\w*)"),
    "import": re.compile(r"(?:import\s+(?:[\w*{}\s,]+\s+from\s+)?['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]|"
                         r"#include\s*[<\"]([^>\"]+)[>\"]|^\s*use\s+([\w:]+)|^\s*import\s+([\w.]+))", re.M),
    "comment": re.compile(r"(?://([^\n]*)|/\*(.*?)\*/|^\s*#(?!include|!)([^\n]*))", re.S | re.M),
    "branch": re.compile(r"\b(if|for|while|case|catch|switch|elif|except)\b|&&|\|\||\?"),
    "route": re.compile(r"['\"]((?:GET|POST|PUT|DELETE)\s+/[^'\"]*|/[a-z0-9_/{}-]+)['\"]"),
}


def _comments_py(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") and not s.startswith("#!"):
            out.append(s[1:].strip())
    return out


def facts(files: dict[str, str]) -> dict:
    f = {"paths": sorted(files), "defs": [], "classes": [], "docs": [], "imports": [], "comments": [], "routes": [],
         "lines": 0, "comment_lines": 0, "funcs": 0, "branches": 0, "max_depth": 0, "async": 0,
         "langs": {k: 0 for k in LANG_ORDER}, "tests": 0, "manifest": 0, "io": 0, "module_docs": 0, "py_files": 0}
    for path in sorted(files):
        text = files[path]
        lang = lang_of(path)
        n_lines = text.count("\n") + 1
        f["lines"] += n_lines
        f["langs"][lang] += n_lines
        base = posixpath.basename(path).lower()
        if base.startswith("test") or "/tests/" in "/" + path or base.endswith(("_test.py", ".test.js", ".spec.js")):
            f["tests"] += 1
        if base in ("crucible.json", "package.json", "pyproject.toml", "setup.py", "cargo.toml", "go.mod",
                    "requirements.txt"):
            f["manifest"] += 1
            if base == "crucible.json":
                try:
                    man = json.loads(text)
                    f["routes"] += list((man.get("endpoints") or {}).keys())
                    f["routes"] += list(((man.get("web") or {}).get("routes") or {}).keys())
                    if man.get("description"):
                        f["docs"].append(str(man["description"])[:500])
                except (ValueError, AttributeError):
                    pass
        if re.search(r"\b(open|socket|requests|fetch|urllib|http|net\.fetch|crucible_sdk\.call)\b", text):
            f["io"] += 1
        if lang == "python":
            f["py_files"] += 1
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError, RecursionError):
                tree = None
            if tree is not None:
                v = _PyFacts()
                v.visit(tree)
                md = ast.get_docstring(tree)
                if md:
                    f["docs"].append(md)
                    f["module_docs"] += 1
                f["defs"] += v.defs
                f["classes"] += v.classes
                f["docs"] += v.docs
                f["imports"] += v.imports
                f["routes"] += v.routes
                f["funcs"] += v.funcs
                f["branches"] += v.branches
                f["max_depth"] = max(f["max_depth"], v.max_depth)
                f["async"] += v.async_
                cm = _comments_py(text)
                f["comments"] += cm
                f["comment_lines"] += len(cm)
                continue
        if lang in ("text",):
            f["docs"].append(text[:1500])
            continue
        if lang in ("data", "css"):
            continue
        for m in _RX["def"].finditer(text):
            name = next((g for g in m.groups() if g), None)
            if name and name not in ("if", "for", "while", "switch", "return"):
                f["defs"].append(name)
                f["funcs"] += 1
        f["classes"] += _RX["class"].findall(text)
        for m in _RX["import"].finditer(text):
            name = next((g for g in m.groups() if g), None)
            if name:
                f["imports"].append(name)
        for m in _RX["comment"].finditer(text):
            c = next((g for g in m.groups() if g), "")
            if c.strip():
                f["comments"].append(" ".join(c.split()))
                f["comment_lines"] += c.count("\n") + 1
        f["branches"] += len(_RX["branch"].findall(text))
        f["routes"] += _RX["route"].findall(text)[:50]
        depth = 0
        for ch in text:
            if ch == "{":
                depth += 1
                f["max_depth"] = max(f["max_depth"], depth)
            elif ch == "}":
                depth = max(0, depth - 1)
        f["async"] += len(re.findall(r"\basync\b", text))
        if lang == "html":
            from ..media import html_text
            title, body = html_text(text)
            f["docs"].append((title + " " + body)[:1500])
    return f


def summary(f: dict) -> str:
    def uniq(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    parts = []
    if f["docs"]:
        parts.append(" ".join(" ".join(d.split()) for d in f["docs"][:12]))
    if f["classes"]:
        parts.append("types: " + ", ".join(uniq(words(c) for c in f["classes"])[:40]))
    if f["defs"]:
        parts.append("functions: " + ", ".join(uniq(words(d) for d in f["defs"] if not d.startswith("__"))[:80]))
    if f["routes"]:
        parts.append("endpoints: " + ", ".join(uniq(str(r) for r in f["routes"])[:30]))
    if f["imports"]:
        parts.append("uses: " + ", ".join(uniq(words(i.split(".")[0]) for i in f["imports"])[:30]))
    if f["comments"]:
        parts.append(" ".join(f["comments"][:40]))
    parts.append("files: " + ", ".join(words(posixpath.splitext(p)[0]) for p in f["paths"][:30]))
    return " . ".join(p for p in parts if p)[:SUMMARY_CHARS]


def structure(f: dict) -> np.ndarray:
    lines = max(1, f["lines"])
    langs = np.array([f["langs"][k] for k in LANG_ORDER], dtype=np.float32) / lines
    funcs = max(1, f["funcs"])
    h = np.zeros(10, dtype=np.float32)
    for imp in set(i.split(".")[0] for i in f["imports"]):
        d = hashlib.sha256(imp.encode()).digest()
        h[d[0] % 10] += 1.0 if d[1] & 1 else -1.0
    if np.any(h):
        h = h / np.linalg.norm(h)
    v = np.concatenate([langs, np.array([
        math.log1p(len(f["paths"])) / math.log1p(MAX_FILES),
        math.log1p(f["lines"]) / math.log1p(100_000),
        math.tanh(100.0 * f["funcs"] / lines / 10.0),
        math.tanh(100.0 * len(f["classes"]) / lines / 3.0),
        min(1.0, f["comment_lines"] / lines * 3.0),
        min(1.0, (len(f["docs"])) / funcs),
        1.0 if f["tests"] else 0.0,
        1.0 if f["manifest"] else 0.0,
        math.tanh(len(f["routes"]) / 5.0),
        math.tanh(f["branches"] / funcs / 5.0),
        min(1.0, f["max_depth"] / 10.0),
        math.tanh(len(set(f["imports"])) / 20.0),
        math.tanh(f["async"] / 5.0),
        1.0 if f["io"] else 0.0,
    ], dtype=np.float32), h])
    return v.astype(np.float32)


class CodeEncoder(Encoder):
    modality = "code"
    id = "crucible.code.struct32+minilm-l6-v2"
    roles = ("text",)
    dim = 384 + 32
    batch_size = 64
    preprocessing = {"inputs": "file | zip | tar(.gz) | json files", "max_files": MAX_FILES, "max_total": MAX_TOTAL,
                     "max_file": MAX_FILE, "summary": "docstrings, type/function names (split), endpoints, imports, "
                     "comments, paths", "summary_chars": SUMMARY_CHARS, "struct_dims": 32,
                     "weights": {"summary": 0.75, "structure": 0.25}}

    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        fs = [facts(read_tree(b)) for b in items]
        tv = TEXT.embed_texts([summary(f) for f in fs])
        return np.stack([np.concatenate([block(t, W_TEXT), block(structure(f), W_STRUCT)])
                         for t, f in zip(tv, fs)]).astype(np.float32)


CODE = register(CodeEncoder())
