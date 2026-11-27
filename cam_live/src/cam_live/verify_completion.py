"""cam_live.verify_completion — the executable completion checker (spec 0.3). MASTER-OWNED (MAN).

    python -m cam_live.verify_completion                          # check, print, exit 1 unless engineering-complete
    python -m cam_live.verify_completion --merge-status           # also write the merged manifest + docs/COMPLETION.md
    python -m cam_live.verify_completion --json out.json --quiet  # full machine-readable report
    python -m cam_live.verify_completion --only MR-07,RC-03       # a subset (never reports completion)

A requirement is **complete** only if ALL of the following hold (docs/COMPLETION.md states them with the run):

  1. its owning workstream's status file claims it ``implemented``;
  2. every named file exists and contains no placeholder (``raise NotImplementedError`` that is not an abstract method
     every subclass overrides, a to-do / fix-me marker in a comment or docstring, or a function whose whole body is
     ``pass`` / ``...`` outside a Protocol or abstract class, unless it overrides a library hook);
  3. it names acceptance tests (``file::test``); each exists, and its newest recorded non-skipped result in
     ``docs/evidence/test_results/*.xml`` is a pass that is *newer* than the last commit (or uncommitted edit) that
     touched the requirement's files or the test's own file — a result older than that is stale and does not count;
  4. at least one integration point resolves and is import-reachable from the entry points of the requirement's
     ``kind`` (runtime processes / console script / operator CLIs / configured factories; plus evaluation programs;
     plus diagnostic programs whose recorded command produced evidence) — a disconnected module cannot count;
  5. where the requirement needs executed evidence, an evidence file shows the run happened, did not fail, and is not
     older than the last change to the requirement's files;
  6. an independent verdict ``verified`` (workstream G, ``docs/verification/verdicts_<lens>.json``) is newer than the
     last change to the requirement's files. A verdict older than that means *needs re-verification*.

Requirements of kind ``none`` (documents, processes) need 1, 2 and 6 only. Each requirement lands in exactly one
group: complete, implemented-needs-reverification, partial, missing, failed. The five project statuses of spec 0.3
are computed separately (none implies the next) and reported both on their own and cumulatively.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import glob
import importlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))

GROUPS = ("complete", "implemented-needs-reverification", "partial", "missing", "failed")
LEVEL_NAMES = {1: "1_engineering_implementation_complete", 2: "2_controlled_experimental_validation_complete",
               3: "3_ready_for_authorized_live_deployment", 4: "4_live_experiment_executed",
               5: "5_scientific_hypothesis_supported"}
VERDICT_RANK = {"failed": 0, "rejected": 0, "unverified": 1, "partial": 2, "verified": 3}

#: entry points, relative to the package. "runtime": what a running system starts (plus operator CLIs and the
#: factories its task / config files name); "evaluation": evaluation programs; diagnostic programs are discovered
#: from recorded commands (see :func:`entry_points`).
RUNTIME_ENTRY_POINTS = {
    "agent": "integrated agent (default agent factory of cam-live run)",
    "validation": "cam-live validate --mode staging",
    "runtime.cli": "console script cam-live",
    "runtime.loop_service": "learner process started by cam-live run",
    "runtime.verifier_service": "independent verifier process",
    "runtime.dashboard": "operator dashboard (cam-live dashboard)",
    "harness.service": "computer-environment executor service",
    "harness.auth.service": "authorization service",
    "harness.auth.launcher": "launcher of the separated services",
    "harness.grounding.worker": "visual-grounding worker subprocess (spawned by harness.grounding.base)",
}
OPERATOR_ENTRY_POINTS = {
    "harness.auth.opctl": "operator CLI: approvals, stop, pause, resume",
    "harness.credentials": "operator CLI: credential entry",
    "harness.operator_requests": "operator CLI: answers to CAM's requests",
    "verify_completion": "operator CLI: cam-live verify-completion",
}
FACTORY_ENTRY_POINTS = {
    "harness.discord_membership": "verifier evidence-source factory (success.evidence_source of tasks/discord_100.yaml)",
}
EVALUATION_ENTRY_POINTS = {
    "eval.cli": "python -m cam_live.eval.cli (evaluation runs)",
    "eval.report": "python -m cam_live.eval.report (evidence digest)",
    "eval.transfer": "cam-live evaluate-transfer (fresh-process transfer)",
    "eval.extra": "python -m cam_live.eval.extra (integrity / meta / scaling commands)",
}
FORBIDDEN_IMPORTS = ("superlearner", "crucible", "agentrs")
_MARKER = re.compile(r"\bTODO\b|\bFIXME\b")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_MINUS_M = re.compile(r"-m\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)")
_FACTORY = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+):([A-Za-z_]\w*)\b")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_COMMIT_KEYS = ("commit", "code_commit", "git_head", "git_commit", "_code", "head")
_TIME_KEYS = ("finished_unix", "ended_at", "written_at_unix", "evaluated_at", "timestamp", "started_unix", "created_at",
              "started_at")


# =================================================================================================================
@dataclass
class Paths:
    """Where the checker looks. ``Paths()`` is the real project; tests build synthetic ones."""
    project: str = PROJECT
    package: str = "cam_live"

    def p(self, *a: str) -> str:
        return os.path.join(self.project, *a)

    @property
    def src(self) -> str:
        return self.p("src")

    @property
    def manifest(self) -> str:
        return self.p("docs", "CAM_LIVE_COMPLETION_MANIFEST.json")

    @property
    def status_dir(self) -> str:
        return self.p("docs", "status")

    @property
    def results_dir(self) -> str:
        return self.p("docs", "evidence", "test_results")

    @property
    def verification_dir(self) -> str:
        return self.p("docs", "verification")

    @property
    def evidence_dir(self) -> str:
        return self.p("docs", "evidence")

    @property
    def completion_md(self) -> str:
        return self.p("docs", "COMPLETION.md")

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.project)

    def abs(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if p.startswith("cam_live/") and not os.path.exists(self.p(p)) and os.path.basename(self.project) == "cam_live":
            p = p[len("cam_live/"):]
        return self.p(p)


_TEXT: Dict[str, Tuple[float, str, Any]] = {}


def _source(path: str) -> Tuple[str, Any]:
    """(text, parsed tree or a SyntaxError) of a Python file, cached by modification time."""
    mt = os.path.getmtime(path)
    hit = _TEXT.get(path)
    if hit is None or hit[0] != mt:
        text = open(path, encoding="utf-8", errors="replace").read()
        try:
            tree: Any = ast.parse(text)
        except SyntaxError as e:
            tree = e
        hit = (mt, text, tree)
        _TEXT[path] = hit
    return hit[1], hit[2]


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_time(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 1e9 else None
    if isinstance(v, str):
        try:
            d = _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    return None


def _iso(t: Optional[float]) -> str:
    if t is None:
        return "-"
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# =================================================================================================================
# git: when did a path last change, and what is not committed
class Git:
    def __init__(self, project: str):
        self.project = os.path.abspath(project)
        self.top = ""
        self.available = False
        try:
            self.top = self._run(["rev-parse", "--show-toplevel"], cwd=self.project).strip()
            self.available = bool(self.top)
        except (OSError, subprocess.CalledProcessError):
            self.available = False
        self.prefix = os.path.relpath(self.project, self.top) if self.available else ""
        if self.prefix == ".":
            self.prefix = ""
        self._times: Optional[Dict[str, float]] = None
        self._dirty: Optional[Dict[str, str]] = None
        self._shown: Dict[Tuple[str, str], Optional[str]] = {}
        self._hist: Dict[str, List[Tuple[str, float]]] = {}
        self._diffs: Dict[str, Optional[Set[str]]] = {}

    def _run(self, args: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
        p = subprocess.run(["git", *args], cwd=cwd or self.top, capture_output=True, text=True,
                           env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
        if check and p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, args, p.stdout, p.stderr)
        return p.stdout

    def _to_project(self, top_rel: str) -> Optional[str]:
        if not self.prefix:
            return top_rel
        pre = self.prefix.rstrip("/") + "/"
        return top_rel[len(pre):] if top_rel.startswith(pre) else None

    def file_times(self) -> Dict[str, float]:
        """project-relative path -> author time of the newest commit that changed it."""
        if self._times is None:
            self._times = {}
            if self.available:
                out = self._run(["log", "--format=%x00%at", "--name-only", "--no-renames", "--", self.prefix or "."])
                t = 0.0
                for line in out.splitlines():
                    if line.startswith("\x00"):
                        t = float(line[1:] or 0)
                    elif line.strip():
                        rel = self._to_project(line.strip())
                        if rel is not None and rel not in self._times:
                            self._times[rel] = t
        return self._times

    def dirty(self) -> Dict[str, str]:
        """project-relative path -> porcelain status code, for uncommitted changes and untracked files."""
        if self._dirty is None:
            self._dirty = {}
            if self.available:
                out = self._run(["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", self.prefix or "."])
                parts = out.split("\x00")
                i = 0
                while i < len(parts):
                    ent = parts[i]
                    i += 1
                    if len(ent) < 4:
                        continue
                    code, path = ent[:2], ent[3:]
                    if code[0] in "RC":
                        i += 1                          # the original path follows a rename / copy
                    rel = self._to_project(path)
                    if rel is not None:
                        self._dirty[rel] = code
        return self._dirty

    def history(self, rel: str) -> List[Tuple[str, float]]:
        if not self.available:
            return []
        if rel not in self._hist:
            path = os.path.join(self.prefix, rel) if self.prefix else rel
            out = self._run(["log", "--format=%H %at", "--", path], check=False)
            self._hist[rel] = [(ln.split()[0], float(ln.split()[1])) for ln in out.splitlines() if ln.strip()]
        return self._hist[rel]

    def show(self, sha: str, rel: str) -> Optional[str]:
        key = (sha, rel)
        if key not in self._shown:
            path = os.path.join(self.prefix, rel) if self.prefix else rel
            p = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=self.top, capture_output=True, text=True)
            self._shown[key] = p.stdout if p.returncode == 0 else None
        return self._shown[key]

    def changed_since(self, sha: str, rels: Sequence[str]) -> Optional[bool]:
        """Did any of ``rels`` change between ``sha`` and the working tree? None when ``sha`` is unknown."""
        if not self.available or not rels:
            return None
        if sha not in self._diffs:
            if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=self.top,
                              capture_output=True).returncode:
                self._diffs[sha] = None
            else:
                out = self._run(["diff", "--name-only", "--no-renames", sha, "--", self.prefix or "."], check=False)
                self._diffs[sha] = {r for r in (self._to_project(x.strip()) for x in out.splitlines() if x.strip()) if r}
        changed = self._diffs[sha]
        if changed is None:
            return None
        changed = changed | {k for k, c in self.dirty().items() if c == "??"}
        for r in rels:
            r = os.path.normpath(r)
            if r in changed or any(c.startswith(r.rstrip("/") + "/") for c in changed):
                return True
        return False

    def head(self) -> str:
        return self._run(["rev-parse", "HEAD"], check=False).strip() if self.available else ""

    def branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"], check=False).strip() if self.available else ""

    def ref(self, name: str) -> str:
        return self._run(["rev-parse", "--verify", "-q", name], check=False).strip() if self.available else ""

    def count(self, rng: str) -> Optional[int]:
        out = self._run(["rev-list", "--count", rng], check=False).strip()
        return int(out) if out.isdigit() else None

    def ignored(self) -> List[str]:
        if not self.available:
            return []
        out = self._run(["ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "--",
                         self.prefix or "."], check=False)
        return sorted(r for r in (self._to_project(x.strip()) for x in out.splitlines() if x.strip()) if r)


class ChangeTimes:
    """When did a path (file or directory) last change: newest commit touching it, or its mtime if uncommitted."""

    def __init__(self, paths: Paths, git: Git):
        self.P, self.git = paths, git
        self._cache: Dict[str, Tuple[float, str]] = {}

    def of(self, rel: str) -> Tuple[float, str]:
        """(time, how) — how is 'commit', 'uncommitted', 'untracked', 'mtime' or 'missing'."""
        rel = os.path.normpath(rel)
        if rel in self._cache:
            return self._cache[rel]
        full = self.P.abs(rel)
        times, dirty = self.git.file_times(), self.git.dirty()
        if os.path.isdir(full):
            best: Tuple[float, str] = (0.0, "commit")
            pre = rel.rstrip("/") + "/"
            for k, t in times.items():
                if k.startswith(pre) and t > best[0]:
                    best = (t, "commit")
            for k, code in dirty.items():
                if k.startswith(pre):
                    fk = self.P.abs(k)
                    t = os.path.getmtime(fk) if os.path.exists(fk) else time.time()
                    if t > best[0]:
                        best = (t, "untracked" if code == "??" else "uncommitted")
            res = best
        elif rel in dirty:
            res = (os.path.getmtime(full) if os.path.exists(full) else time.time(),
                   "untracked" if dirty[rel] == "??" else "uncommitted")
        elif rel in times and self.git.available:
            res = (times[rel], "commit")
        elif os.path.exists(full):
            res = (os.path.getmtime(full), "mtime")
        else:
            res = (0.0, "missing")
        self._cache[rel] = res
        return res

    def newest(self, rels: Iterable[str]) -> Tuple[float, str, str]:
        """(time, how, path) of the most recent change among ``rels``."""
        best = (0.0, "none", "")
        for r in rels:
            t, how = self.of(r)
            if t > best[0]:
                best = (t, how, r)
        return best


# =================================================================================================================
# manifest + status merge
def _aliases(owner: str, ws: str) -> bool:
    return ws == owner or ws.startswith(owner + "-")


def merge_status(manifest: Dict[str, Any], P: Paths) -> List[str]:
    """Fold workstream status files into the manifest (claims, files, tests, ...). Returns notes."""
    notes: List[str] = []
    by_id = {r["id"]: r for r in manifest["requirements"]}
    claims: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for path in sorted(glob.glob(os.path.join(P.status_dir, "*.json"))):
        try:
            st = _load_json(path)
        except Exception as e:  # noqa: BLE001
            notes.append(f"unreadable status file {os.path.basename(path)}: {e}")
            continue
        ws = st.get("workstream") or os.path.splitext(os.path.basename(path))[0]
        for rid, claim in (st.get("requirements") or {}).items():
            if rid not in by_id:
                notes.append(f"status file {os.path.basename(path)} claims unknown requirement {rid}")
                continue
            claims.setdefault(rid, []).append((ws, claim))
    order = {"not_started": 0, "partial": 1, "implemented": 2}
    for rid, req in by_id.items():
        decl = req.get("declared") or {}
        files, tests, ips = list(decl.get("files", [])), list(decl.get("tests", [])), list(decl.get("integration_points", []))
        cmds, ev, lim = list(decl.get("commands", [])), list(decl.get("evidence", [])), []
        owner_state: Optional[str] = None
        best: Optional[str] = None
        for ws, c in claims.get(rid, []):
            files += c.get("files") or []
            tests += c.get("tests") or []
            ips += c.get("integration_points") or []
            cmds += c.get("commands") or []
            ev += c.get("evidence") or []
            if c.get("limitations"):
                lim.append(f"[{ws}] {c['limitations']}")
            s = c.get("state", "not_started")
            s = s if s in order else "partial"
            if _aliases(req["workstream"], ws):
                owner_state = s if owner_state is None or order[s] < order[owner_state] else owner_state
            if best is None or order[s] > order[best]:
                best = s
        req["files"] = sorted(set(files))
        req["acceptance_tests"] = sorted(set(tests))
        req["integration_points"] = sorted(set(ips))
        req["commands"] = list(dict.fromkeys(cmds))
        req["evidence"] = sorted(set(ev))
        req["limitations"] = " | ".join(lim)
        req["claimed_state"] = owner_state if owner_state is not None else (best or "unclaimed")
        req["claimed_by"] = sorted({ws for ws, _ in claims.get(rid, [])})
        req["claim_source"] = ("owner" if owner_state is not None else "other workstream" if best else "none")
    return notes


# =================================================================================================================
# placeholders and standalone imports
def _py_files(full: str) -> List[str]:
    if os.path.isdir(full):
        return sorted(glob.glob(os.path.join(full, "**", "*.py"), recursive=True))
    return [full] if full.endswith(".py") and os.path.exists(full) else []


_PLACEHOLDERS: Dict[Tuple[str, float], List[str]] = {}


def placeholder_findings(path: str, P: Paths) -> List[str]:
    out: List[str] = []
    for f in _py_files(path):
        key = (os.path.abspath(f), os.path.getmtime(f))
        if key not in _PLACEHOLDERS:
            _PLACEHOLDERS[key] = _placeholders_in(f, P)
        out += _PLACEHOLDERS[key]
    return out


def _abstract_by_convention(tree: ast.AST) -> Set[int]:
    """Lines of ``raise NotImplementedError`` that are the whole body of a base-class method which every subclass
    defined in the same module overrides (an abstract method by convention, not a placeholder)."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    lines: Set[int] = set()
    for c in classes:
        subs = [k for k in classes if any(getattr(b, "id", getattr(b, "attr", "")) == c.name for b in k.bases)]
        if not subs:
            continue
        for fn in c.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
                continue
            exc = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
            if getattr(exc, "id", "") != "NotImplementedError":
                continue
            if all(any(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == fn.name for m in k.body)
                   for k in subs):
                lines.add(body[0].lineno)
    return lines


def _placeholders_in(f: str, P: Paths) -> List[str]:
    out: List[str] = []
    text, tree = _source(f)
    rel = P.rel(f)
    lines = text.splitlines()
    if isinstance(tree, SyntaxError):
        return [f"{rel}: syntax error {tree}"]
    exempt = _abstract_by_convention(tree)
    hits: Set[int] = set()
    try:                                     # markers in comments (string literals are data, e.g. a scanner's list)
        import io
        import tokenize
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and _MARKER.search(tok.string):
                hits.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) and \
                _MARKER.search(n.value.value):                   # ... and in docstrings
            hits.add(n.lineno)
        elif isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            if getattr(exc, "id", "") == "NotImplementedError" and n.lineno not in exempt:
                hits.add(n.lineno)
    for i in sorted(hits):
        out.append(f"{rel}:{i}: {lines[i - 1].strip()[:80] if 0 < i <= len(lines) else ''}")

    def abstract(cls: ast.ClassDef) -> bool:
        names = {getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases}
        return bool(names & {"Protocol", "ABC"}) or any("abc" in ast.dump(b) for b in cls.bases)

    imported: Dict[str, Tuple[str, str]] = {}         # local name -> (external module, attribute or "")
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and not n.level and n.module and n.module.split(".")[0] != P.package:
            for a in n.names:
                imported[a.asname or a.name] = (n.module, a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] != P.package:
                    imported[(a.asname or a.name).split(".")[0]] = (a.name.split(".")[0] if not a.asname else a.name, "")

    def library_hook(cls: Optional[ast.ClassDef], name: str) -> bool:
        """The method overrides a method of an imported library base class (a deliberate no-op override)."""
        for b in (cls.bases if cls is not None else []):
            dotted = ast.unparse(b) if hasattr(ast, "unparse") else ""
            head = dotted.split(".")[0]
            if head not in imported:
                continue
            mod, attr = imported[head]
            try:
                obj: Any = importlib.import_module(mod)
                for part in ([attr] if attr else []) + dotted.split(".")[1:]:
                    obj = getattr(obj, part) if hasattr(obj, part) else importlib.import_module(
                        f"{obj.__name__}.{part}")
                if callable(getattr(obj, name, None)):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def stmts(node: ast.AST) -> List[ast.AST]:
        out_: List[ast.AST] = []
        for name in ("body", "orelse", "finalbody", "handlers", "cases"):
            v = getattr(node, name, None)
            if isinstance(v, list):
                out_ += [x for x in v if isinstance(x, ast.AST)]
        return out_

    def walk(node: ast.AST, in_abstract: bool, cls: Optional[ast.ClassDef] = None) -> None:
        for ch in stmts(node):                                    # statements only: functions cannot hide in expressions
            if isinstance(ch, ast.ClassDef):
                walk(ch, in_abstract or abstract(ch), ch)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in ch.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                   and isinstance(b.value.value, str))]
                deco = {ast.dump(d) for d in ch.decorator_list}
                is_abs = in_abstract or any("abstractmethod" in d or "overload" in d for d in deco)
                empty = all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                        and b.value.value is Ellipsis) for b in body)
                if empty and not is_abs and not ch.name.startswith("__") and not library_hook(cls, ch.name):
                    out.append(f"{rel}:{ch.lineno}: empty function {ch.name}()")
                walk(ch, in_abstract)
            else:
                walk(ch, in_abstract, cls)
    walk(tree, False)
    return out


def standalone_findings(P: Optional[Paths] = None) -> List[str]:
    """Imports of CAM Unified / Crucible / AgentRS code anywhere in the package (spec 0.1: CAM Live is standalone)."""
    P = P or Paths()
    out: List[str] = []
    for f in sorted(glob.glob(os.path.join(P.src, P.package, "**", "*.py"), recursive=True)):
        _text, tree = _source(f)
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                    "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
                names = [str(node.args[0].value)]
            for n in names:
                if n.split(".")[0] in FORBIDDEN_IMPORTS:
                    out.append(f"{P.rel(f)}:{node.lineno}: imports {n}")
    return out


# =================================================================================================================
# import graph, entry points, reachability
@dataclass
class Graph:
    deps: Dict[str, Set[str]]                 # module -> modules it imports or loads dynamically
    programs: Set[str]                        # modules runnable with ``python -m``
    tools: Dict[str, str]                     # tool name -> module that registers it
    nodes_tools: Dict[str, Set[str]]          # tools/<script> -> package modules it imports

    def module_of(self, dotted: str) -> str:
        m = dotted
        while m and m not in self.deps:
            m = m.rpartition(".")[0]
        return m


def import_graph(P: Paths) -> Graph:
    pkg_root = os.path.join(P.src, P.package)
    deps: Dict[str, Set[str]] = {}
    programs: Set[str] = set()
    tools: Dict[str, str] = {}
    self_mod = f"{P.package}.verify_completion"
    files = sorted(glob.glob(os.path.join(pkg_root, "**", "*.py"), recursive=True))
    modules = set()
    for path in files:
        rel = os.path.relpath(path, P.src)[:-3].replace(os.sep, ".")
        modules.add(rel[:-9] if rel.endswith(".__init__") else rel)

    def known(name: str) -> bool:
        return name in modules

    for path in files:
        fn = os.path.basename(path)
        rel = os.path.relpath(path, P.src)[:-3].replace(os.sep, ".")
        mod = rel[:-9] if rel.endswith(".__init__") else rel
        pkg = mod if fn == "__init__.py" else mod.rpartition(".")[0]
        d: Set[str] = set()
        _text, tree = _source(path)
        if isinstance(tree, SyntaxError):
            deps[mod] = d
            continue
        if fn == "__main__.py":
            programs.add(mod.rpartition(".")[0])
            d.add(mod.rpartition(".")[0])
        docstrings = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                      and isinstance(n.value.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                d |= {a.name for a in node.names if a.name.split(".")[0] == P.package}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target.split(".")[0] == P.package:
                    d.add(target)
                    d |= {f"{target}.{a.name}" for a in node.names}
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                    isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators):
                    programs.add(mod)
            elif isinstance(node, ast.Call):
                fname = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if fname in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                        and str(node.args[0].value).split(".")[0] == P.package:
                    d.add(str(node.args[0].value))
                if "tool" in fname.lower() and fname not in ("import_module",):
                    cand = [a for a in node.args[:1] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                    cand += [k.value for k in node.keywords if k.arg == "name" and isinstance(k.value, ast.Constant)
                             and isinstance(k.value.value, str)]
                    for c in cand:
                        if "." in c.value and " " not in c.value:
                            tools.setdefault(c.value, mod)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(
                    "TOOL" in getattr(t, "id", "").upper() for t in node.targets):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        tools.setdefault(k.value, mod)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict) and \
                    "TOOL" in getattr(node.target, "id", "").upper():
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        tools.setdefault(k.value, mod)
            if mod != self_mod and isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                    len(node.value) < 200 and id(node) not in docstrings:
                s = node.value.strip()
                if known(s):                                         # a module named by string: subprocess -m, etc.
                    d.add(s)
                for m in _FACTORY.finditer(s):                       # "package.module:attr" factory references
                    if known(m.group(1)):
                        d.add(m.group(1))
        deps[mod] = {x for x in d if x != mod}
    nodes_tools: Dict[str, Set[str]] = {}
    for path in sorted(glob.glob(os.path.join(P.project, "tools", "*.py"))):
        d = set()
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                d |= {a.name for a in node.names if a.name.split(".")[0] == P.package}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module and \
                    node.module.split(".")[0] == P.package:
                d.add(node.module)
                d |= {f"{node.module}.{a.name}" for a in node.names}
        nodes_tools["tools/" + os.path.basename(path)] = d
    return Graph(deps, programs, tools, nodes_tools)


def console_scripts(P: Paths) -> Dict[str, str]:
    """[project.scripts] of pyproject.toml: script name -> 'module:function'."""
    path = P.p("pyproject.toml")
    if not os.path.exists(path):
        return {}
    try:
        import tomllib
        with open(path, "rb") as f:
            return dict((tomllib.load(f).get("project") or {}).get("scripts") or {})
    except Exception:  # noqa: BLE001
        return {}


def recorded_commands(P: Paths, manifest: Dict[str, Any]) -> List[str]:
    """Every command recorded in status files, the manifest and evidence JSON (``command`` / ``commands`` keys)."""
    out: List[str] = []

    def walk(o: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lstrip("_") in ("command", "commands", "measured_commands", "cmd"):
                    if isinstance(v, str):
                        out.append(v)
                    elif isinstance(v, list):
                        out.extend(x for x in v if isinstance(x, str))
                    elif isinstance(v, dict):
                        out.extend(x for x in v.values() if isinstance(x, str))
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o[:2000]:
                walk(v, depth + 1)
    for path in glob.glob(os.path.join(P.status_dir, "*.json")):
        try:
            walk(_load_json(path))
        except Exception:  # noqa: BLE001
            pass
    for r in manifest.get("requirements", []):
        out.extend(r.get("commands") or [])
    for path in glob.glob(os.path.join(P.evidence_dir, "**", "*.json"), recursive=True):
        if os.path.getsize(path) > 5_000_000:
            continue
        try:
            walk(_load_json(path))
        except Exception:  # noqa: BLE001
            pass
    return out


def configured_factories(P: Paths) -> Dict[str, str]:
    """'package.module:attr' references in task and config files -> the file that names them."""
    out: Dict[str, str] = {}
    for pat in ("tasks/**/*.yaml", "tasks/**/*.yml", "configs/**/*.yaml", "configs/**/*.yml"):
        for path in glob.glob(P.p(pat), recursive=True):
            for m in _FACTORY.finditer(open(path, encoding="utf-8", errors="replace").read()):
                if m.group(1).split(".")[0] == P.package:
                    out.setdefault(m.group(1), P.rel(path))
    return out


def entry_points(P: Paths, G: Graph, manifest: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """kind -> {module: why}. Kinds: runtime, operator, factory, evaluation, diagnostic, tool."""
    pk = P.package + "."
    eps: Dict[str, Dict[str, str]] = {k: {} for k in ("runtime", "operator", "factory", "evaluation", "diagnostic", "tool")}
    for table, kind in ((RUNTIME_ENTRY_POINTS, "runtime"), (OPERATOR_ENTRY_POINTS, "operator"),
                        (FACTORY_ENTRY_POINTS, "factory"), (EVALUATION_ENTRY_POINTS, "evaluation")):
        for m, why in table.items():
            if pk + m in G.deps:
                eps[kind][pk + m] = why
    for name, target in console_scripts(P).items():
        mod = target.split(":")[0]
        if mod in G.deps:
            eps["runtime"].setdefault(mod, f"console script {name}")
    for mod, where in configured_factories(P).items():
        if mod in G.deps:
            eps["factory"].setdefault(mod, f"factory named in {where}")
    for cmd in recorded_commands(P, manifest):
        for m in _MINUS_M.finditer(cmd):
            mod = m.group(1)
            if mod in G.programs and not any(mod in v for v in eps.values()):
                eps["diagnostic"][mod] = "program whose recorded command produced evidence"
    for node in G.nodes_tools:
        eps["tool"][node] = "operator / evidence tool script"
    return eps


def reachable(G: Graph, starts: Iterable[str]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(starts)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        if m in G.nodes_tools:
            nxt = list(G.nodes_tools[m])
        else:
            parts = m.split(".")
            nxt = [".".join(parts[:i]) for i in range(1, len(parts))] + list(G.deps.get(m, ()))
        for d in nxt:
            d = G.module_of(d) if d not in G.nodes_tools else d
            if d and d not in seen:
                stack.append(d)
    return seen


def reach_sets(G: Graph, eps: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    rt = list(eps["runtime"]) + list(eps["operator"]) + list(eps["factory"])
    ev = rt + list(eps["evaluation"])
    every = ev + list(eps["diagnostic"]) + list(eps["tool"])
    return {"runtime": reachable(G, rt), "evaluation": reachable(G, ev), "diagnostic": reachable(G, every)}


def _getattr_chain(obj: Any, attrs: Sequence[str]) -> Any:
    for a in attrs:
        if hasattr(obj, a):
            obj = getattr(obj, a)
            continue
        fields = getattr(obj, "__dataclass_fields__", None) or {}
        if a in fields or a in (getattr(obj, "__annotations__", None) or {}):
            return None                                  # a declared field of a record type
        raise AttributeError(f"{getattr(obj, '__name__', obj)!s} has no attribute {a!r}")
    return obj


def _subcommands(module: Any) -> Optional[Set[str]]:
    bp = getattr(module, "build_parser", None)
    if bp is None:
        return None
    try:
        parser = bp()
    except Exception:  # noqa: BLE001
        return None
    out: Set[str] = set()
    for act in getattr(parser, "_actions", []):
        if isinstance(act, argparse._SubParsersAction):  # noqa: SLF001
            out |= set(act.choices)
    return out


def resolve_integration_point(ip: str, P: Paths, G: Graph, reach: Dict[str, Set[str]],
                              scripts: Dict[str, str]) -> Dict[str, Any]:
    """{"ok": resolves, "module": ..., "reach": "runtime" | "evaluation" | "diagnostic" | "", "why": ...}"""
    s = ip.strip()
    res: Dict[str, Any] = {"ip": ip, "ok": False, "module": "", "reach": "", "why": "", "form": "symbol"}

    def reach_of(mod: str) -> str:
        for k in ("runtime", "evaluation", "diagnostic"):
            if mod in reach[k]:
                return k
        return ""
    first = s.split()[0] if s else ""
    if first in scripts:
        res["form"] = "console script"
        mod = scripts[first].split(":")[0]
        sub = next((t for t in s.split()[1:] if not t.startswith("-")), "")
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            res["why"] = f"{ip}: console script module {mod} does not import: {type(e).__name__}: {e}"
            return res
        if sub:
            subs = _subcommands(m)
            if subs is not None and sub not in subs:
                res["why"] = f"{ip}: {first} has no subcommand {sub!r}"
                return res
            if subs is None and not hasattr(m, "cmd_" + sub.replace("-", "_")):
                res["why"] = f"{ip}: {first} has no subcommand {sub!r}"
                return res
        res.update(ok=True, module=mod, reach=reach_of(mod))
        return res
    m = re.match(r"^(?:python3?|\S*/python3?)\s+-m\s+(\S+)", s)
    if m:
        res["form"] = "program"
        mod = m.group(1)
        if mod not in G.deps:
            res["why"] = f"{ip}: no module {mod}"
            return res
        if mod not in G.programs:
            res["why"] = f"{ip}: {mod} is not runnable as a program (no __main__ guard)"
            return res
        res.update(ok=True, module=mod, reach=reach_of(mod))
        return res
    if s.startswith("tool "):
        res["form"] = "tool"
        name = s.split()[1] if len(s.split()) > 1 else ""
        mod = G.tools.get(name, "")
        if not mod:
            res["why"] = f"{ip}: no tool named {name!r} is registered in the package"
            return res
        res.update(ok=True, module=mod, reach=reach_of(mod))
        return res
    mt = _IDENT.match(s)
    if not mt:
        res["why"] = f"{ip}: not a dotted name, console-script command, 'python -m' program or 'tool <name>'"
        return res
    target = mt.group(0)
    if target.split(".")[0] != P.package and G.module_of(f"{P.package}.{target}"):
        target = f"{P.package}.{target}"
    mod = G.module_of(target)
    if not mod:
        res["why"] = f"{ip}: no such module"
        return res
    try:
        obj = importlib.import_module(mod)
        _getattr_chain(obj, [a for a in target[len(mod):].split(".") if a])
    except Exception as e:  # noqa: BLE001
        res["why"] = f"{ip}: {type(e).__name__}: {e}"
        return res
    res.update(ok=True, module=mod, reach=reach_of(mod))
    if s[len(mt.group(0)):].strip():
        res["note"] = "trailing annotation ignored"
    return res


_KIND_REACH = {"runtime": ("runtime",), "evaluation": ("runtime", "evaluation"),
               "diagnostic": ("runtime", "evaluation", "diagnostic"), "none": ("runtime", "evaluation", "diagnostic")}


# =================================================================================================================
# recorded test results
@dataclass
class Result:
    outcome: str
    t: float
    file: str


def recorded_results(P: Paths) -> Dict[Tuple[str, str], List[Result]]:
    """(classname, name) -> every recorded result, with the start time of the run that produced it."""
    out: Dict[Tuple[str, str], List[Result]] = {}
    for path in sorted(glob.glob(os.path.join(P.results_dir, "*.xml"))):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for suite in suites:
            t = _parse_time(suite.get("timestamp")) or os.path.getmtime(path)
            for tc in suite.iter("testcase"):
                kinds = {c.tag for c in tc}
                oc = "failed" if kinds & {"failure", "error"} else "skipped" if "skipped" in kinds else "passed"
                out.setdefault((tc.get("classname", ""), tc.get("name", "")), []).append(
                    Result(oc, t, os.path.basename(path)))
    return out


def check_test(node_id: str, P: Paths, results: Dict[Tuple[str, str], List[Result]],
               changed: Tuple[float, str, str], ct: ChangeTimes) -> Dict[str, Any]:
    """Status of one named acceptance test: pass / fail (fresh or stale), skipped-only, no-result or bad name."""
    out: Dict[str, Any] = {"test": node_id, "status": "", "fresh": False, "result_time": None, "result_file": "",
                           "changed_at": None, "changed_by": ""}
    parts = node_id.split("::")
    path = parts[0][len("cam_live/"):] if parts[0].startswith("cam_live/") else parts[0]
    full = P.abs(path)
    if len(parts) < 2:
        out["status"] = "not-a-node"
        out["why"] = f"{node_id}: names a file or directory, not a test function (file::test)"
        return out
    if not os.path.isfile(full):
        out["status"] = "missing-file"
        out["why"] = f"{node_id}: test file missing"
        return out
    fn = parts[-1].split("[")[0]
    if not re.search(rf"def\s+{re.escape(fn)}\s*\(", open(full, encoding="utf-8", errors="replace").read()):
        out["status"] = "missing-function"
        out["why"] = f"{node_id}: test function not found"
        return out
    tchg = ct.of(path)
    when, how, who = changed
    if tchg[0] > when:
        when, how, who = tchg[0], tchg[1], path
    out["changed_at"], out["changed_by"] = when, f"{who} ({how})" if who else ""
    cls = path[:-3].replace("/", ".") + ("." + ".".join(parts[1:-1]) if len(parts) > 2 else "")
    exact = "[" in parts[-1]
    variants: Dict[str, Result] = {}
    any_skip = False
    for (c, n), rs in results.items():
        if not (c == cls or c.endswith("." + cls) or cls.endswith("." + c)):
            continue
        if not (n == parts[-1] if exact else n.split("[")[0] == fn):
            continue
        real = [r for r in rs if r.outcome != "skipped"]
        any_skip |= len(real) < len(rs)
        if real:
            variants[n] = max(real, key=lambda r: (r.t, r.outcome == "failed"))   # newest non-skipped; ties: failure
    if not variants:
        out["status"] = "skipped-only" if any_skip else "no-result"
        out["why"] = (f"{node_id}: only skipped results recorded (needs a gpu / privileged / browser / slow run)"
                      if any_skip else f"{node_id}: no recorded result")
        return out
    fails = [r for r in variants.values() if r.outcome == "failed"]
    rep = max(fails, key=lambda r: r.t) if fails else min(variants.values(), key=lambda r: r.t)
    out["result_time"], out["result_file"] = rep.t, rep.file
    out["fresh"] = rep.t > when
    if fails:
        out["status"] = "fail"
        out["why"] = (f"{node_id}: recorded FAILED ({rep.file}, {_iso(rep.t)})" if out["fresh"] else
                      f"{node_id}: recorded FAILED before the last change ({out['changed_by']}, {_iso(when)}): re-run")
    else:
        out["status"] = "pass"
        if not out["fresh"]:
            out["why"] = (f"{node_id}: newest pass ({rep.file}, {_iso(rep.t)}) predates the last change "
                          f"({out['changed_by']}, {_iso(when)}): stale")
    return out


# =================================================================================================================
# independent verdicts with the time they were issued
@dataclass
class Verdict:
    rid: str
    status: str
    by: str
    notes: str
    lens: str
    at: float
    at_source: str
    commit: str = ""
    supersedes: List[str] = field(default_factory=list)


def _entry_time(git: Git, P: Paths, rel: str, rid: str, entry: Dict[str, Any], current_head: Optional[Dict[str, Any]]
                ) -> Tuple[float, str]:
    for k in ("verified_at", "at", "timestamp", "date"):
        t = _parse_time(entry.get(k))
        if t:
            return t, f"field {k}"
    full = P.abs(rel)
    if not git.available or rel not in git.file_times():
        return os.path.getmtime(full), "file mtime (not committed)"
    if current_head is None or current_head.get(rid) != entry:
        return os.path.getmtime(full), "file mtime (uncommitted edit)"
    hist = git.history(rel)
    t, src = hist[0][1], f"commit {hist[0][0][:8]}"
    for sha, at in hist[1:]:
        txt = git.show(sha, rel)
        try:
            old = json.loads(txt) if txt else {}
        except ValueError:
            break
        if old.get(rid) != entry:
            break
        t, src = at, f"commit {sha[:8]}"
    return t, src


def load_verdicts(P: Paths, git: Git) -> Dict[str, List[Verdict]]:
    out: Dict[str, List[Verdict]] = {}
    lens_files = sorted(glob.glob(os.path.join(P.verification_dir, "verdicts_*.json")))
    seen: Set[Tuple[str, str]] = set()
    for path in lens_files:
        rel = P.rel(path)
        lens = os.path.basename(path)[len("verdicts_"):-len(".json")]
        try:
            cur = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        head_txt = git.show("HEAD", rel) if git.available and rel in git.file_times() else None
        try:
            head = json.loads(head_txt) if head_txt else None
        except ValueError:
            head = None
        for rid, v in (cur or {}).items():
            if not isinstance(v, dict):
                continue
            at, src = _entry_time(git, P, rel, rid, v, head)
            st = v.get("status", "unverified")
            out.setdefault(rid, []).append(Verdict(rid, st if st in VERDICT_RANK else "unverified",
                                                   v.get("by") or f"G-{lens}", v.get("notes", ""), lens, at, src,
                                                   str(v.get("commit", "") or ""), list(v.get("supersedes") or [])))
            seen.add((rid, lens))
    merged = os.path.join(P.verification_dir, "verdicts.json")
    if os.path.exists(merged):
        try:
            mv = _load_json(merged)
        except Exception:  # noqa: BLE001
            mv = {}
        for rid, v in (mv or {}).items():
            if rid in out or not isinstance(v, dict):
                continue                                     # lens files are the source; the merge only fills gaps
            st = v.get("status", "unverified")
            out.setdefault(rid, []).append(Verdict(rid, st if st in VERDICT_RANK else "unverified", v.get("by", "G"),
                                                   v.get("notes", ""), "merged", os.path.getmtime(merged),
                                                   "verdicts.json mtime"))
    return out


def judge_verdicts(vs: List[Verdict], P: Paths, git: Git, files: Sequence[str], changed: Tuple[float, str, str]
                   ) -> Dict[str, Any]:
    """state: fresh-verified | fresh-partial | fresh-failed | stale | none (+ the verdicts with their freshness)."""
    rows = []
    for v in vs:
        if v.commit:
            ch = git.changed_since(v.commit, files)
            fresh = ch is False
            basis = f"files {'changed' if ch else 'unchanged' if ch is False else 'unknown'} since verified commit {v.commit[:8]}"
        else:
            fresh = v.at > changed[0]
            basis = f"issued {_iso(v.at)} ({v.at_source}); last change {_iso(changed[0])} {changed[2]}"
        rows.append({"v": v, "fresh": fresh, "basis": basis})
    live = []
    for r in rows:                                           # a newer verdict may supersede an older one
        v = r["v"]
        newer = [o for o in rows if o is not r and o["v"].at > v.at and
                 (v.by in o["v"].supersedes or v.lens in o["v"].supersedes or "all" in o["v"].supersedes or
                  (o["v"].by == v.by and o["v"].lens != v.lens))]
        if not newer:
            live.append(r)
    out: Dict[str, Any] = {"verdicts": [{"status": r["v"].status, "by": r["v"].by, "lens": r["v"].lens,
                                         "fresh": r["fresh"], "basis": r["basis"]} for r in live]}
    if not live:
        out.update(state="none", status="unverified", by="", notes="")
        return out
    fresh = [r for r in live if r["fresh"]]
    stale = [r for r in live if not r["fresh"]]
    worst_f = min(fresh, key=lambda r: VERDICT_RANK[r["v"].status]) if fresh else None
    worst_all = min(live, key=lambda r: VERDICT_RANK[r["v"].status])
    if worst_f and VERDICT_RANK[worst_f["v"].status] == 0:
        state, pick = "fresh-failed", worst_f
    elif worst_f and worst_f["v"].status in ("partial", "unverified"):
        state, pick = "fresh-partial", worst_f
    elif stale:
        state, pick = "stale", worst_all
    else:
        state, pick = "fresh-verified", worst_f
    out.update(state=state, status=pick["v"].status, by="+".join(sorted({r["v"].by for r in live})),
               notes=pick["v"].notes)
    return out


# =================================================================================================================
# executed evidence
def _evidence_marker(d: Any) -> Tuple[Optional[bool], str]:
    """(True executed / False a failed run / None no marker, how)."""
    if not isinstance(d, dict):
        return None, ""
    if d.get("ok") is False:
        return False, "ok=false"
    if "exit_code" in d:
        return (d.get("exit_code") == 0), f"exit_code={d.get('exit_code')}"
    if d.get("executed") is True:
        return True, "executed=true"
    for k in ("command", "_command", "commands", "_commands"):
        if d.get(k):
            return True, f"recorded {k}"
    return None, ""


def _evidence_time(d: Dict[str, Any], rel: str, ct: ChangeTimes) -> Tuple[Optional[str], Optional[float], str]:
    for k in _COMMIT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and _SHA.match(v.strip()):
            return v.strip(), None, f"{k} {v.strip()[:8]}"
        if isinstance(v, dict):
            for kk in ("commit", "head", "sha"):
                vv = v.get(kk)
                if isinstance(vv, str) and _SHA.match(vv.strip()):
                    return vv.strip(), None, f"{k}.{kk} {vv.strip()[:8]}"
    for k in _TIME_KEYS:
        t = _parse_time(d.get(k))
        if t:
            return None, t, f"{k} {_iso(t)}"
    t, how = ct.of(rel)
    return None, t, f"file {how} {_iso(t)}"


def executed_index(P: Paths) -> Dict[str, List[str]]:
    """requirement id -> evidence JSON files that declare ``executed: true`` with that id in ``requirements``."""
    found: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(P.evidence_dir, "**", "*.json"), recursive=True):
        if os.path.getsize(path) > 20_000_000:
            continue
        try:
            d = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("executed") is True:
            for rid in d.get("requirements") or []:
                found.setdefault(str(rid), []).append(P.rel(path))
    return found


def check_evidence(req: Dict[str, Any], P: Paths, git: Git, ct: ChangeTimes, index: Dict[str, List[str]],
                   changed: Tuple[float, str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"needed": bool(req.get("needs_executed_evidence")), "missing": [], "runs": [], "state": ""}
    for e in req.get("evidence", []):
        if not os.path.exists(P.abs(e)):
            out["missing"].append(e)
    cands = list(dict.fromkeys(index.get(req["id"], []) + [e for e in req.get("evidence", []) if e.endswith(".json")]))
    for rel in cands:
        full = P.abs(rel)
        if not os.path.exists(full):
            continue
        try:
            d = _load_json(full)
        except Exception:  # noqa: BLE001
            continue
        ok, how = _evidence_marker(d)
        if ok is None:
            continue
        sha, t, basis = _evidence_time(d, rel, ct)
        if sha:
            ch = git.changed_since(sha, req.get("files", []))
            fresh = ch is False
            basis += f" (files {'changed' if ch else 'unchanged' if ch is False else 'unknown'} since)"
        else:
            fresh = (t or 0) > changed[0]
        out["runs"].append({"file": rel, "ok": ok, "marker": how, "fresh": fresh, "basis": basis})
    good = [r for r in out["runs"] if r["ok"]]
    out["state"] = ("fresh" if any(r["fresh"] for r in good) else "stale" if good else
                    "failed-run" if out["runs"] else "absent")
    return out


# =================================================================================================================
# one requirement
def check_requirement(req: Dict[str, Any], C: "Context") -> Dict[str, Any]:
    P = C.P
    kind = req.get("kind", "runtime")
    problems: List[str] = []
    stale: List[str] = []
    notes: List[str] = []
    state = req.get("claimed_state", "unclaimed")
    if state != "implemented":
        problems.append(f"claimed state is '{state}'" + (" (no status file claims it)" if state == "unclaimed" else ""))
    files = req.get("files", [])
    existing = [f for f in files if os.path.exists(P.abs(f))]
    if not files:
        problems.append("no implementation files named")
    for f in files:
        if f not in existing:
            problems.append(f"missing file {f}")
    placeholders: List[str] = []
    for f in existing:
        placeholders += placeholder_findings(P.abs(f), P)
    problems += placeholders
    changed = C.ct.newest(existing)
    # tests
    tests = [check_test(t, P, C.results, changed, C.ct) for t in req.get("acceptance_tests", [])]
    if not tests and kind != "none":
        problems.append("no acceptance test named")
    fresh_fail = [t for t in tests if t["status"] == "fail" and t["fresh"]]
    for t in tests:                                  # stale failures first: they are why such a requirement is partial
        if t["status"] == "fail" and not t["fresh"]:
            stale.insert(sum(1 for x in stale if "FAILED" in x), t["why"])
        elif t["status"] == "pass" and not t["fresh"]:
            stale.append(t["why"])
        elif t["status"] != "pass":
            problems.append(t["why"])
    n_fresh = sum(1 for t in tests if t["status"] == "pass" and t["fresh"])
    n_any = sum(1 for t in tests if t["status"] == "pass")
    # integration
    ips = [resolve_integration_point(ip, P, C.graph, C.reach, C.scripts) for ip in req.get("integration_points", [])]
    allowed = _KIND_REACH.get(kind, ("runtime",))
    ok_ip = [r for r in ips if r["ok"] and r["reach"] in allowed]
    if kind != "none":
        if not ips:
            problems.append("no runtime integration point named")
        elif not ok_ip:
            resolved = [r for r in ips if r["ok"]]
            if resolved:
                problems.append(f"no integration point is import-reachable from the {kind} entry points (" +
                                "; ".join(f"{r['ip']}: {r['reach'] or 'unreachable'}" for r in resolved[:3]) + ")")
    for r in ips:
        if not r["ok"]:
            problems.append(r["why"])
        elif r not in ok_ip and ok_ip:
            notes.append(f"{r['ip']}: reachable only as {r['reach'] or 'nothing'}")
    # evidence
    ev = check_evidence(req, P, C.git, C.ct, C.ev_index, changed)
    for e in ev["missing"]:
        problems.append(f"missing evidence file {e}")
    if ev["needed"]:
        if ev["state"] == "absent":
            problems.append("needs executed evidence: no evidence file shows the run (executed=true / recorded "
                            "command / exit_code)")
        elif ev["state"] == "failed-run":
            problems.append("executed evidence reports a failed run: " +
                            "; ".join(f"{r['file']} {r['marker']}" for r in ev["runs"]))
        elif ev["state"] == "stale":
            stale.append("executed evidence predates the last change: " +
                         "; ".join(f"{r['file']} ({r['basis']})" for r in ev["runs"] if r["ok"])[:300])
    # verification
    ver = judge_verdicts(C.verdicts.get(req["id"], []), P, C.git, existing, changed)
    # groups
    implemented = (state == "implemented" and bool(existing) and len(existing) == len(files) and not placeholders
                   and (kind == "none" or (bool(tests) and n_any == len(tests) and bool(ok_ip)))
                   and not (ev["needed"] and ev["state"] in ("absent", "failed-run"))
                   and not [p for p in problems if p.startswith("missing evidence")])
    all_fresh = all(t["status"] == "pass" and t["fresh"] for t in tests) and \
        not (ev["needed"] and ev["state"] != "fresh")
    if ver["state"] == "fresh-failed" or fresh_fail:
        group = "failed"
        reason = ("fresh failed verdict: " + ver["notes"][:160]) if ver["state"] == "fresh-failed" else \
            "a named acceptance test FAILED after the last change: " + fresh_fail[0]["test"]
    elif not existing:
        group, reason = "missing", ("no implementation file named" if not files else "none of the named files exists")
    elif implemented and all_fresh and ver["state"] == "fresh-verified":
        group, reason = "complete", ""
    elif implemented and ver["state"] != "fresh-partial":
        group = "implemented-needs-reverification"
        why = []
        if ver["state"] == "stale":
            why.append(f"verdict '{ver['status']}' predates the last change")
        elif ver["state"] == "none":
            why.append("never independently verified")
        if not all(t["status"] == "pass" and t["fresh"] for t in tests):
            why.append(f"{len(tests) - n_fresh} of {len(tests)} test results stale")
        if ev["needed"] and ev["state"] != "fresh":
            why.append("executed evidence stale")
        reason = "; ".join(why)
    else:
        group = "partial"
        reason = ("fresh verdict 'partial': " + ver["notes"][:120]) if ver["state"] == "fresh-partial" else \
            (problems[0] if problems else stale[0] if stale else "")
    req["engineering_status"] = "integrated" if (implemented and kind != "none") else \
        ("implemented" if implemented else state if state in ("partial", "not_started", "implemented") else "not_started")
    req["test_results"] = [f"{t['test']}: {t['status']}{'' if t['status'] != 'pass' else (' fresh' if t['fresh'] else ' stale')}"
                           + (f" ({t['result_file']}, {_iso(t['result_time'])})" if t.get("result_time") else "")
                           for t in tests]
    req["verification"] = {"status": ver["status"], "by": ver["by"], "notes": ver["notes"], "state": ver["state"]}
    req["group"] = group
    return {"id": req["id"], "workstream": req["workstream"], "title": req["title"], "kind": kind,
            "level": req.get("level", 1), "claimed_state": state, "claimed_by": req.get("claimed_by", []),
            "group": group, "reason": reason, "complete": group == "complete", "implemented": implemented,
            "engineering_ok": implemented and all_fresh, "verified": ver["state"] == "fresh-verified",
            "tests": {"named": len(tests), "pass_fresh": n_fresh, "pass_any": n_any,
                      "detail": [{k: v for k, v in t.items() if k != "why"} for t in tests]},
            "integration": [{k: v for k, v in r.items()} for r in ips], "evidence": ev, "verification": ver,
            "problems": problems, "stale": stale, "notes": notes}


# =================================================================================================================
@dataclass
class Context:
    P: Paths
    git: Git
    ct: ChangeTimes
    graph: Graph
    eps: Dict[str, Dict[str, str]]
    reach: Dict[str, Set[str]]
    scripts: Dict[str, str]
    results: Dict[Tuple[str, str], List[Result]]
    verdicts: Dict[str, List[Verdict]]
    ev_index: Dict[str, List[str]]


def _live_status(P: Paths) -> Dict[str, Any]:
    out: Dict[str, Any] = {"preflight_file": "", "preflight_ready": False, "blockers": [], "live_run_file": "",
                           "live_run_executed": False, "transfer_supported": False}
    pre = sorted(glob.glob(os.path.join(P.evidence_dir, "preflight*.json")), key=os.path.getmtime)
    if pre:
        try:
            d = _load_json(pre[-1])
            out["preflight_file"] = P.rel(pre[-1])
            if isinstance(d.get("live"), dict):                      # cam_live.runtime.preflight.preflight() format
                out["preflight_ready"] = bool(d["live"].get("ready"))
                out["blockers"] = list(d["live"].get("missing") or [])
            else:
                out["preflight_ready"] = bool(d.get("live_ready"))
                out["blockers"] = list(d.get("blockers") or [])
        except Exception as e:  # noqa: BLE001
            out["blockers"] = [f"unreadable preflight evidence: {e}"]
    else:
        out["blockers"] = ["no preflight evidence (docs/evidence/preflight*.json) recorded"]
    live = sorted(glob.glob(os.path.join(P.evidence_dir, "live_run*.json")), key=os.path.getmtime)
    if live:
        try:
            d = _load_json(live[-1])
            out["live_run_file"] = P.rel(live[-1])
            out["live_run_executed"] = bool(d.get("mode") == "live" and d.get("verified_outcome_signature_ok"))
            out["transfer_supported"] = bool(d.get("transfer_supported"))
        except Exception:  # noqa: BLE001
            pass
    return out


def durability(P: Paths, git: Git) -> Dict[str, Any]:
    """What is preserved off this instance and what is not (operator amendment of 2026-11-26)."""
    out: Dict[str, Any] = {"git": git.available}
    if not git.available:
        out["note"] = "not a git checkout: nothing about off-instance preservation can be established"
        return out
    br = git.branch()
    head = git.head()
    remote = git.ref(f"refs/remotes/origin/{br}")
    out.update(branch=br, head=head[:12], remote_ref=f"origin/{br}", remote_head=remote[:12] if remote else "")
    if remote:
        out["commits_not_on_remote"] = git.count(f"{remote}..{head}")
        out["remote_contains_head"] = out["commits_not_on_remote"] == 0
    else:
        out["commits_not_on_remote"] = None
        out["remote_contains_head"] = False
    d = git.dirty()
    out["uncommitted"] = sorted(k for k, c in d.items() if c != "??")
    out["untracked"] = sorted(k for k, c in d.items() if c == "??")
    out["ignored"] = git.ignored()
    out["note"] = ("The remote ref is the local remote-tracking ref (no network access): it shows the last push made "
                   "from this checkout. Uncommitted, untracked and git-ignored paths (run directories, checkpoints, "
                   "experience stores) have no off-instance copy from git; `cam-live doctor` reports whether the state "
                   "root is on a persistent volume.")
    return out


def coverage_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    ids = {r["id"] for r in manifest["requirements"]}
    cov = manifest.get("coverage") or {}
    unknown = sorted({i for items in cov.values() for it in items for i in it.get("ids", []) if i not in ids})
    uncovered = sorted(i for i in ids if not any(i in it.get("ids", []) for items in cov.values() for it in items))
    return {"sections": {k: len(v) for k, v in cov.items()}, "items": sum(len(v) for v in cov.values()),
            "unknown_ids": unknown, "ids_without_enumerated_spec_item": uncovered}


def build_context(P: Paths, manifest: Dict[str, Any]) -> Context:
    if P.src not in sys.path:
        sys.path.insert(0, P.src)
    git = Git(P.project)
    G = import_graph(P)
    eps = entry_points(P, G, manifest)
    return Context(P, git, ChangeTimes(P, git), G, eps, reach_sets(G, eps), console_scripts(P), recorded_results(P),
                   load_verdicts(P, git), executed_index(P))


def run(only: Optional[List[str]] = None, do_merge: bool = False, paths: Optional[Paths] = None,
        completion_md: Optional[str] = None) -> Dict[str, Any]:
    P = paths or Paths()
    manifest = _load_json(P.manifest)
    notes = merge_status(manifest, P)
    C = build_context(P, manifest)
    rows = [check_requirement(r, C) for r in manifest["requirements"] if not only or r["id"] in only]
    counts = {g: sum(1 for r in rows if r["group"] == g) for g in GROUPS}
    counts["total"] = len(rows)
    standalone = standalone_findings(P)
    live = _live_status(P)
    own: Dict[int, bool] = {}
    for lv in (1, 2, 3, 4):
        reqs = [r for r in rows if r["level"] == lv]
        own[lv] = bool(reqs) and all(r["complete"] for r in reqs) and not only
    own[1] = own[1] and not standalone
    own[3] = own[3] and live["preflight_ready"] and not live["blockers"]
    own[4] = own[4] and live["live_run_executed"]
    own[5] = bool(live["live_run_executed"] and live["transfer_supported"]) and not only
    cum: Dict[int, bool] = {}
    prev = True
    for lv in (1, 2, 3, 4, 5):
        prev = prev and own[lv]
        cum[lv] = prev
    levels = {LEVEL_NAMES[lv]: cum[lv] for lv in LEVEL_NAMES}
    level_detail = {LEVEL_NAMES[lv]: {"own_criteria_hold": own[lv], "cumulative": cum[lv],
                                      "requirements": sum(1 for r in rows if r["level"] == lv),
                                      "complete": sum(1 for r in rows if r["level"] == lv and r["complete"])}
                    for lv in LEVEL_NAMES}
    unreachable = sorted(m for m in C.graph.deps if m not in C.reach["diagnostic"] and not m.endswith("__main__"))
    diag_only = sorted(m for m in C.graph.deps if m in C.reach["diagnostic"] and m not in C.reach["evaluation"])
    eval_only = sorted(m for m in C.graph.deps if m in C.reach["evaluation"] and m not in C.reach["runtime"])
    rep = {"generated_at": _iso(time.time()), "head": C.git.head()[:12], "levels": levels, "level_detail": level_detail,
           "counts": counts, "rows": rows, "notes": notes, "live": live, "standalone_findings": standalone,
           "entry_points": {k: dict(sorted(v.items())) for k, v in C.eps.items()},
           "unreachable_modules": unreachable, "reachable_only_from_diagnostic_programs": diag_only,
           "reachable_only_from_evaluation_programs": eval_only, "coverage": coverage_report(manifest),
           "durability": durability(P, C.git),
           "dirty_requirement_files": sorted({f for r in manifest["requirements"] for f in r.get("files", [])
                                              if os.path.normpath(f) in C.git.dirty()})}
    manifest["status_levels"] = levels
    if do_merge and not only:
        tmp = P.manifest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, P.manifest)
    if completion_md and not only:
        with open(completion_md, "w") as f:
            f.write(render_completion(rep, manifest))
    return rep


# =================================================================================================================
def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_completion(rep: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    c = rep["counts"]
    L: List[str] = []
    L.append("# CAM Live — completion status (generated)")
    L.append("")
    L.append("Generated by the executable completion checker; do not edit by hand. Regenerate with:")
    L.append("")
    L.append("    cd /workspace/camlive/cam_live && ../.venv/bin/python tools/gen_manifest.py")
    L.append("    cd /workspace/camlive/cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.verify_completion "
             "--merge-status --json /tmp/completion.json --quiet")
    L.append("")
    L.append(f"Generated {rep['generated_at']} at commit `{rep['head']}`. Requirements: **{c['total']}** — "
             + ", ".join(f"{g}: **{c[g]}**" for g in GROUPS) + ".")
    if rep["dirty_requirement_files"]:
        L.append("")
        L.append("Uncommitted changes in requirement files at generation time (they count as changes made now): " +
                 ", ".join(f"`{f}`" for f in rep["dirty_requirement_files"][:20]) + ".")
    L.append("")
    L.append("## Status levels (spec 0.3)")
    L.append("")
    L.append("| Level | Own criteria hold | Cumulative (all lower levels too) | Requirements complete |")
    L.append("|---|---|---|---|")
    for name, d in rep["level_detail"].items():
        L.append(f"| {name} | {'yes' if d['own_criteria_hold'] else 'no'} | {'YES' if d['cumulative'] else 'no'} | "
                 f"{d['complete']} / {d['requirements']} |")
    L.append("")
    L.append("Each level is computed from its own criteria; none implies the next. Level 1: every level-1 requirement "
             "complete and no standalone-import finding. Level 2: every requirement whose claim is a controlled "
             "experiment (level 2) complete, i.e. with current executed evidence and a current verdict. Level 3: every "
             "level-3 requirement complete and the newest `docs/evidence/preflight*.json` reports live ready with "
             "nothing missing. Level 4: every level-4 requirement complete and `docs/evidence/live_run*.json` records a "
             "live run with a valid verifier signature. Level 5: the live-run record states the transfer hypothesis "
             "is supported. The cumulative column is the project status.")
    lv = rep["live"]
    L.append("")
    L.append(f"Preflight evidence: {lv['preflight_file'] or 'none'}; live ready: {lv['preflight_ready']}; blockers: "
             + (_md("; ".join(map(str, lv["blockers"][:8]))) or "none") + ".")
    L.append("")
    L.append("## Rules applied")
    L.append("")
    L.append("A requirement is **complete** only if: (1) its owning workstream's status file claims `implemented`; "
             "(2) every named file exists without placeholders; (3) every named acceptance test exists and its newest "
             "recorded non-skipped result in `docs/evidence/test_results/*.xml` passed **after** the last commit (or "
             "uncommitted edit) touching the requirement's files or the test file — older results are stale; "
             "(4) an integration point resolves and is import-reachable from the entry points of the requirement's "
             "kind; (5) where needed, executed evidence shows the run happened, did not fail and is not older than the "
             "files; (6) an independent `verified` verdict was issued after the last change to the files. Kind "
             "`none` (documents, processes) needs (1), (2) and (6).")
    L.append("")
    L.append("* **implemented-needs-reverification** — (1), (2), (4), (5) hold and every named test has passed, but "
             "a verdict or a test result or the evidence is older than the last change (or the requirement was never "
             "verified), and no current verdict says partial or failed.")
    L.append("* **partial** — some implementation exists, but a claim, test, integration point or evidence is "
             "missing, or a current verdict says partial.")
    L.append("* **missing** — no named implementation file exists.")
    L.append("* **failed** — a current verdict says failed, or a named test failed after the last change.")
    L.append("")
    L.append("Times: commit author times (`git log`), uncommitted edits by file mtime, test results by the JUnit "
             "`timestamp` of the run (its start), verdicts by the first commit since which the verdict entry in "
             "`docs/verification/verdicts_<lens>.json` is unchanged (or an explicit `verified_at` / `commit` field in "
             "the entry; a newer verdict replaces an older one of the same `by`, or of the lenses it lists in "
             "`supersedes`), evidence by its recorded commit or time. A skipped result never replaces a pass or fail.")
    L.append("")
    L.append("Kinds: `runtime` — reachable from the running system (" +
             ", ".join(f"`{m}`" for m in sorted(rep["entry_points"]["runtime"])) + "), the operator CLIs (" +
             ", ".join(f"`{m}`" for m in sorted(rep["entry_points"]["operator"])) + ") or configured factories (" +
             ", ".join(f"`{m}`" for m in sorted(rep["entry_points"]["factory"])) + "); `evaluation` — also from (" +
             ", ".join(f"`{m}`" for m in sorted(rep["entry_points"]["evaluation"])) + "); `diagnostic` — also from "
             "diagnostic programs whose recorded command produced evidence (" +
             ", ".join(f"`{m}`" for m in sorted(rep["entry_points"]["diagnostic"])) + ") and `tools/*.py`.")
    titles = {r["id"]: r for r in manifest["requirements"]}
    for g in GROUPS:
        rows = [r for r in rep["rows"] if r["group"] == g]
        L.append("")
        L.append(f"## {g} ({len(rows)})")
        L.append("")
        if not rows:
            L.append("None.")
            continue
        L.append("| Id | Ws | Requirement | Claim | Tests fresh/named | Verdict | Why |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            v = r["verification"]
            vtxt = f"{v['status']} ({v['state']})" if v["state"] != "none" else "none"
            title = r["title"] if len(r["title"]) <= 90 else r["title"][:87] + "..."
            L.append(f"| {r['id']} | {r['workstream']} | {_md(title)} | {r['claimed_state']} | "
                     f"{r['tests']['pass_fresh']}/{r['tests']['named']} | {_md(vtxt)} | {_md(r['reason'][:220])} |")
        if g in ("implemented-needs-reverification", "partial", "failed", "missing"):
            pass
    L.append("")
    L.append("## Findings")
    L.append("")
    L.append("Modules not import-reachable from any entry point: " +
             (", ".join(f"`{m}`" for m in rep["unreachable_modules"]) or "none") + ".")
    L.append("")
    L.append("Modules reachable only from diagnostic programs: " +
             (", ".join(f"`{m}`" for m in rep["reachable_only_from_diagnostic_programs"]) or "none") + ".")
    L.append("")
    L.append("Modules reachable only from evaluation programs: " +
             (", ".join(f"`{m}`" for m in rep["reachable_only_from_evaluation_programs"]) or "none") + ".")
    L.append("")
    L.append("Standalone check (no import of " + ", ".join(FORBIDDEN_IMPORTS) + "): " +
             (_md("; ".join(rep["standalone_findings"])) or "no finding") + ".")
    cov = rep["coverage"]
    L.append("")
    L.append(f"Spec coverage table: {cov['items']} enumerated items in {len(cov['sections'])} sections map to manifest "
             f"ids (unknown ids: {', '.join(cov['unknown_ids']) or 'none'}).")
    if rep["notes"]:
        L.append("")
        L.append("Merge notes: " + _md("; ".join(rep["notes"])))
    d = rep["durability"]
    L.append("")
    L.append("## Durability (what has an off-instance copy)")
    L.append("")
    if d.get("git"):
        L.append(f"Branch `{d['branch']}` at `{d['head']}`; `{d['remote_ref']}` at `{d['remote_head'] or 'absent'}`; "
                 f"commits not on the remote: {d['commits_not_on_remote']}.")
        L.append("")
        L.append(f"Uncommitted tracked files ({len(d['uncommitted'])}): " +
                 (", ".join(f"`{x}`" for x in d["uncommitted"][:30]) or "none") + ".")
        L.append("")
        L.append(f"Untracked files ({len(d['untracked'])}): " +
                 (", ".join(f"`{x}`" for x in d["untracked"][:30]) or "none") +
                 (" ..." if len(d["untracked"]) > 30 else "") + ".")
        L.append("")
        L.append("Git-ignored paths (never pushed): " + (", ".join(f"`{x}`" for x in d["ignored"][:30]) or "none") + ".")
        L.append("")
    L.append(d.get("note", ""))
    L.append("")
    L.append("## Per-requirement problems")
    L.append("")
    for r in rep["rows"]:
        if r["group"] == "complete":
            continue
        items = r["problems"][:6] + r["stale"][:3]
        if not items:
            continue
        L.append(f"* **{r['id']}** ({r['group']}): " + _md(" · ".join(x[:200] for x in items)))
    L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam-live verify-completion", description=__doc__.split("\n\n")[0])
    ap.add_argument("--merge-status", action="store_true",
                    help="write the merged manifest and regenerate docs/COMPLETION.md")
    ap.add_argument("--only", default="", help="comma-separated requirement ids")
    ap.add_argument("--json", default="", help="write the full report to this path")
    ap.add_argument("--completion-md", default="", help="write the completion table to this path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    P = Paths()
    md = a.completion_md or (P.completion_md if a.merge_status else None)
    rep = run([x.strip() for x in a.only.split(",") if x.strip()] or None, a.merge_status, P, md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)
    c = rep["counts"]
    print(f"requirements: {c['total']}  " + "  ".join(f"{g}: {c[g]}" for g in GROUPS))
    if not a.quiet:
        for g in GROUPS[1:]:
            for r in rep["rows"]:
                if r["group"] == g:
                    print(f"  [{g[:12]:12s}] {r['id']} ({r['workstream']}) {r['title'][:70]}")
                    for w in (r["problems"] + r["stale"])[:3]:
                        print(f"           - {w[:160]}")
        for n in rep["notes"]:
            print("note:", n)
        if rep["unreachable_modules"]:
            print("unreachable modules:", ", ".join(rep["unreachable_modules"]))
        for s in rep["standalone_findings"]:
            print("standalone finding:", s)
    print("status levels:")
    for k, d in rep["level_detail"].items():
        print(f"  {k}: {'YES' if d['cumulative'] else 'no'} (own criteria {'hold' if d['own_criteria_hold'] else 'do not hold'};"
              f" {d['complete']}/{d['requirements']} complete)")
    return 0 if rep["levels"][LEVEL_NAMES[1]] else 1


if __name__ == "__main__":
    sys.exit(main())
