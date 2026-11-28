"""G2-integration / docs probe d01: do the `file:Symbol` citations of MODEL_ARCHITECTURE.md and MATH_NOTES.md
(and OPERATOR_RESOURCES.md) name symbols that exist in the code? Static (ast) check, nothing is imported.

usage (from /workspace/camlive):
  .venv/bin/python cam_live/docs/verification/repro/g2integration/docs/d01_citations.py [SRC_ROOT]
SRC_ROOT = a tree that contains cam_live/docs and cam_live/src/cam_live (default: the frozen copy of a5ab772 if
present, else this worktree). Prints HOLDS / BROKEN per citation and a summary line.
"""
import ast
import os
import re
import sys

FROZEN = "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2INTEGRATION/src"
ROOT = sys.argv[1] if len(sys.argv) > 1 else (FROZEN if os.path.isdir(FROZEN) else os.getcwd())
PKG = os.path.join(ROOT, "cam_live", "src", "cam_live")
DOCS = [os.path.join(ROOT, "cam_live", "docs", n) for n in ("MODEL_ARCHITECTURE.md", "MATH_NOTES.md",
                                                             "OPERATOR_RESOURCES.md")]
CITE = re.compile(r"`((?:[A-Za-z_]+/)*[A-Za-z_]+\.py):([A-Za-z_][\w.]*(?:\s*(?:/|,)\s*[A-Za-z_][\w.]*)*)")


def index(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(f"{node.name}.{b.name}")
                elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    names.add(f"{node.name}.{b.target.id}")
                elif isinstance(b, ast.Assign):
                    for t in b.targets:
                        if isinstance(t, ast.Name):
                            names.add(f"{node.name}.{t.id}")
    for b in tree.body:
        targets = b.targets if isinstance(b, ast.Assign) else [b.target] if isinstance(b, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
    return names


def resolve(rel):
    for cand in (os.path.join(PKG, rel), os.path.join(PKG, "reward", rel), os.path.join(PKG, "runtime", rel),
                 os.path.join(ROOT, "cam_live", rel)):
        if os.path.isfile(cand):
            return cand
    return None


seen, n_ok, n_bad = set(), 0, 0
cache = {}
for doc in DOCS:
    text = open(doc).read()
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in CITE.finditer(line):
            rel, syms = m.group(1), m.group(2)
            for sym in re.split(r"\s*[/,]\s*", syms):
                sym = sym.strip().rstrip(".")
                if not sym or (rel, sym) in seen:
                    continue
                seen.add((rel, sym))
                path = resolve(rel)
                if path is None:
                    print(f"BROKEN  {os.path.basename(doc)}:{lineno}  {rel}:{sym}  (file not found)")
                    n_bad += 1
                    continue
                names = cache.setdefault(path, index(path))
                ok = sym in names or sym.split(".")[-1] in names
                print(f"{'HOLDS ' if ok else 'BROKEN'}  {os.path.basename(doc)}:{lineno}  {rel}:{sym}")
                n_ok += ok
                n_bad += not ok
print(f"summary: {n_ok + n_bad} distinct citations, {n_ok} HOLD, {n_bad} BROKEN (root {ROOT})")
