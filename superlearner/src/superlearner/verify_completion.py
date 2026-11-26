"""Executable completion checker for CAM Unified (spec: CAM_UNIFIED_SPEC_META_REWARDS.md Part 0.4; hardened 2026-11-26
per plan CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md §3 W9 and the audit 'h' findings).

    python -m superlearner.verify_completion                 # check claims against recorded evidence
    python -m superlearner.verify_completion --run-tests     # also execute every referenced test, record JUnit XML
    python -m superlearner.verify_completion --json out.json # machine-readable report

It reads superlearner/docs/CAM_UNIFIED_COMPLETION.json and checks, per requirement, that the claimed status is backed
by evidence. It FAILS (exit code 1) when any of the following holds for a requirement:

  * a mandatory requirement is missing from the manifest (ids listed in REQUIRED_IDS / REQUIRED_PREFIXES),
  * a referenced implementation file does not exist, or a claim of IMPLEMENTED or above names no source file
    (``*/src/*.py``; a document or a test is not an implementation),
  * a claim of IMPLEMENTED or above has no resolvable acceptance test, or a referenced test has no valid PASS record:
    the newest JUnit record must be a pass written by the verify_completion pytest plugin (``-p
    superlearner.verify_completion``), which binds it to a commit; the record is rejected when its commit is not
    an ancestor of HEAD, when the test file or the requirement's source files had uncommitted changes when it ran,
    or when they changed between that commit and HEAD (or are modified in the working tree now),
  * a governance test (this checker's own tests, test_manifest.py) is cited as acceptance or integration evidence,
  * a claim of INTEGRATION_TESTED or above has no resolvable integration test with a valid PASS record that the
    plugin ran as an integration test (no test-double module imported: ``superlearner.cam_unified.testing`` /
    ``integration_doubles``) and in which the requirement's ``integration_entry`` actually executed; the entry
    must be a function under ``*/src/`` (never a test); ``verified_by`` must name a verification record
    ``superlearner/docs/evidence/verification/<id>.json`` {"requirement", "verifier", "workstream" (different from
    the implementing workstream), "commit" (an ancestor of HEAD), "commands", "measured"},
  * a claim of INTEGRATION_TESTED or above, or an empirical claim, has an evidence entry that is neither a resolvable
    repository path (optionally ``#anchor``), nor ``sha256:<64 hex>``, nor a resolvable test id,
  * a claim that names `gradient_tests` (mandatory gradient pathways) has a gradient test without a valid PASS,
  * an EMPIRICALLY_DEMONSTRATED / FAILED_EMPIRICALLY claim's ``experiment_result`` is missing, outside the repository,
    unreadable, or fails the experiment-protocol checks of ``check_experiment_result``: an exact verdict in
    VERDICTS that matches the claim AND is recomputed from the recorded primary interval and direction; a
    pre-registration committed before the first confirm-set row (and not modified after it); every raw file's
    sha256; each (learner, seed) evaluated on the confirm set exactly once; >= 3 seeds for a DEMONSTRATED verdict;
    reward_mode observation_only for CAM learners; the main-scale configuration for scale "dom" learners,
  * an implementation file of a claim at IMPLEMENTED or above is a stub (NotImplementedError bodies, 'placeholder'/
    'stub' markers) or production code imports the test doubles (`integration_doubles`, `testing`).

Global checks: register rows of CAM_UNIFIED_EXPERIMENTS.md against docs/experiments/U-*.md (every declared row has
its pre-registration file, present at the declaring commit; every pre-registration has a row), and a static
reward-integrity scan (actor-side modules never import evaluator / label-channel modules).

Engineering completion and scientific success are reported separately:

  engineering complete  = status in {INTEGRATION_TESTED, FAILED_EMPIRICALLY, EMPIRICALLY_DEMONSTRATED} and all checks pass
  scientific success    = EMPIRICALLY_DEMONSTRATED; FAILED_EMPIRICALLY is an honest negative result, not a defect.

The default invocation never claims the project is complete unless every requirement is engineering-complete.

As a pytest plugin (``-p superlearner.verify_completion``) this module records, on every test case of the JUnit XML,
the properties ``vc_commit`` (HEAD at session start), ``vc_dirty`` (JSON list of modified / untracked repository
paths under superlearner/ and crucible/ at session start), ``vc_test_sha256`` (the test file's digest), and, for
tests listed in ``--vc-integration FILE`` (one node id per line, optionally ``<node id>\\t<entry>[,<entry>]``),
``vc_integration`` ("clean" or the offending double modules) and ``vc_entry_hits`` (JSON {entry: calls}). An
integration test that loads a double, or whose module imports one, FAILS.
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "superlearner", "docs", "CAM_UNIFIED_COMPLETION.json")
RESULTS_DIR = os.path.join(ROOT, "superlearner", "docs", "evidence", "test_results")
REGISTER = os.path.join("superlearner", "docs", "CAM_UNIFIED_EXPERIMENTS.md")
PREREG_DIR = os.path.join("superlearner", "docs", "experiments")
VERIFICATION_DIR = os.path.join("superlearner", "docs", "evidence", "verification")

STATUSES = ["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY",
            "EMPIRICALLY_DEMONSTRATED"]
AT_LEAST_IMPLEMENTED = {"IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
AT_LEAST_INTEGRATED = {"INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
EMPIRICAL = {"FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
VERDICTS = ("DEMONSTRATED", "FAILED", "INCONCLUSIVE", "BLOCKED")
# claim -> experiment verdicts that support it (an inconclusive primary is not a demonstration: it is reported as a
# negative result)
CLAIM_VERDICTS = {"EMPIRICALLY_DEMONSTRATED": {"DEMONSTRATED"}, "FAILED_EMPIRICALLY": {"FAILED", "INCONCLUSIVE"}}
MIN_SEEDS_DEMONSTRATED = 3
MAIN_SCALE = {"n_layers": 32, "d_model": 1536, "n_heads": 24, "sem_model": "Qwen/Qwen3-4B"}
GOVERNANCE_TESTS = ("superlearner/tests/cam_unified/test_manifest.py",
                    "superlearner/tests/cam_unified/test_verify_completion.py")
DOUBLE_MODULES = ("superlearner.cam_unified.testing", "superlearner.cam_unified.integration_doubles")

# Every id below must exist in the manifest (the original A01-A22 acceptance list and the four requirement families
# added on 2026-11-26). A family prefix requires at least one entry and is checked for gaps in its numbering.
REQUIRED_IDS = [f"A{i:02d}" for i in range(1, 23)] + [f"DOM-{i:02d}" for i in range(1, 14)] + \
    [f"EX-{c}" for c in "ABCDEFGHIJK"] + ["X01", "X02", "X03", "X04", "VER-01", "RPT-01"]
REQUIRED_PREFIXES = ["OB-", "SE-", "IR-", "CR-", "KV-", "AU-", "CD-", "UR-"]

STUB_PATTERNS = [
    (re.compile(r"^\s*#.*\b(placeholder|stub implementation|not implemented yet)\b", re.I | re.M), "placeholder comment"),
    (re.compile(r"\breturn\s+[\'\"](placeholder|stub)[\'\"]", re.I), "placeholder return value"),
    (re.compile(r"^\s*#\s*(TODO|FIXME|STUB)\b", re.M), "TODO/STUB marker"),
]
DOUBLE_IMPORT = re.compile(r"^\s*(from\s+\S*(integration_doubles|cam_unified\.testing)\S*\s+import|import\s+\S*integration_doubles"
                           r"|from\s+\S*cam_unified\s+import\s+[^\n]*\b(integration_doubles|testing)\b)", re.M)

# static reward-integrity scan: modules on the acting path (their inputs are actor inputs) never import the
# evaluator / label-channel modules. ACTOR_SIDE_EXEMPT: modules under those paths that are not on the acting path.
ACTOR_SIDE = ("superlearner/src/superlearner/cam_unified/agent.py", "superlearner/src/superlearner/cam_unified/model.py",
              "superlearner/src/superlearner/cam_unified/core/", "superlearner/src/superlearner/cam_unified/semantic/",
              "superlearner/src/superlearner/cam_unified/world/", "superlearner/src/superlearner/cam_unified/decision/",
              "superlearner/src/superlearner/cam_unified/memory/featurize.py",
              "superlearner/src/superlearner/cam_unified/memory/retrieval.py",
              "superlearner/src/superlearner/cam_unified/memory/runtime.py")
ACTOR_SIDE_EXEMPT = {
    # path -> why it is not on the acting path. An exempt module must itself never be imported by a (non-exempt)
    # actor-side module; check_reward_integrity enforces that.
    "superlearner/src/superlearner/cam_unified/semantic/demo_init.py":
        "offline demonstration warm start (scores demonstrations with the evaluator); never imported by the runtime",
    "superlearner/src/superlearner/cam_unified/core/bench.py":
        "throughput / memory benchmark harness (W1): builds an ExperienceOperator to time real updates; imported "
        "only by operator-side drivers (stage1/dom_e2e), never by the acting path",
}
EVALUATOR_SIDE = re.compile(r"^\s*(from\s+[.\w]*(envs\.(labels|openworld_eval|integrity)|envs\s+import\s+[^\n]*"
                            r"\b(labels|openworld_eval)\b)|from\s+\.\.?(labels|openworld_eval)\s+import|import\s+[.\w]*"
                            r"envs\.(labels|openworld_eval))|\bExperienceOperator\b|\bprivate_verify\b|"
                            r"\btake_labels\s*\(", re.M)


# ----------------------------------------------------------------------------------------------- git helpers

def _git(root: str, *args, check: bool = False) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None if not check else ""
    return r.stdout


@lru_cache(maxsize=None)
def git_head(root: str = ROOT) -> Optional[str]:
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


@lru_cache(maxsize=4096)
def _is_ancestor(root: str, a: str, b: str) -> bool:
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=root, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@lru_cache(maxsize=4096)
def _changed_between(root: str, a: str, b: str, paths: tuple) -> tuple:
    out = _git(root, "diff", "--name-only", a, b, "--", *paths)
    return tuple(sorted(x for x in (out or "").splitlines() if x))


@lru_cache(maxsize=4096)
def _dirty_now(root: str, paths: tuple) -> tuple:
    out = _git(root, "status", "--porcelain", "--", *paths)
    return tuple(sorted(line[3:] for line in (out or "").splitlines() if line.strip()))


def clear_caches() -> None:
    """Forget cached git answers (HEAD and the working-tree state change between checks)."""
    for f in (git_head, _is_ancestor, _changed_between, _dirty_now, _commit_time, _exists_at,
              _commits_touching_after):
        f.cache_clear()


@lru_cache(maxsize=4096)
def _commit_time(root: str, c: str) -> Optional[float]:
    out = _git(root, "show", "-s", "--format=%ct", c)
    return float(out.strip()) if out and out.strip() else None


@lru_cache(maxsize=4096)
def _exists_at(root: str, c: str, path: str) -> bool:
    try:
        r = subprocess.run(["git", "cat-file", "-e", f"{c}:{path}"], cwd=root, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@lru_cache(maxsize=4096)
def _commits_touching_after(root: str, path: str, t: float) -> tuple:
    """Commits reachable from HEAD that changed `path` with a commit time after `t`."""
    out = _git(root, "log", "--format=%H %ct", "--", path)
    rows = []
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and float(parts[1]) > t:
            rows.append(parts[0])
    return tuple(rows)


def dirty_paths(root: str = ROOT) -> list:
    out = _git(root, "status", "--porcelain", "--", "superlearner", "crucible")
    return sorted(line[3:].strip() for line in (out or "").splitlines() if line.strip())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _inside(root: str, rel: str) -> Optional[str]:
    """Absolute path of `rel` when it resolves inside `root` (symlinks resolved), else None."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    full = os.path.realpath(os.path.join(root, rel))
    r = os.path.realpath(root)
    return full if full == r or full.startswith(r + os.sep) else None


# ----------------------------------------------------------------------------------------------- helpers

def load_manifest(path: str = MANIFEST) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _split_test_id(test_id: str) -> tuple[str, str] | None:
    if "::" not in test_id:
        return None
    path, rest = test_id.split("::", 1)
    name = rest.split("[", 1)[0].split("::")[-1]
    return path, name


def test_resolves(test_id: str, root: str = ROOT) -> bool:
    parts = _split_test_id(test_id)
    if parts is None:
        return False
    path, name = parts
    full = _inside(root, path)
    if full is None or not os.path.isfile(full):
        return False
    with open(full) as fh:
        src = fh.read()
    return re.search(rf"^\s*(async\s+)?def {re.escape(name)}\(", src, re.M) is not None


def _junit_key(classname: str, name: str, file_attr: str | None, root: str = ROOT) -> tuple[str, str]:
    """JUnit testcases carry classname 'superlearner.tests.cam_unified.core.test_x' (or with a class suffix)."""
    if file_attr:
        path = file_attr
    else:
        mod = classname.split(".")
        # drop a trailing class name (CamelCase) if present
        if mod and mod[-1][:1].isupper():
            mod = mod[:-1]
        path = "/".join(mod) + ".py"
    path = path.replace("\\", "/")
    if not path.startswith("superlearner/"):
        path = "superlearner/" + path if os.path.exists(os.path.join(root, "superlearner", path)) else path
    return path, name.split("[", 1)[0]


def load_results(results_dir: str = RESULTS_DIR, root: str = ROOT) -> dict[tuple[str, str], dict]:
    """Latest outcome per (test file, test function) across all JUnit XML files (newest file wins), with the
    plugin's binding properties. A parametrised test passes only if every recorded parametrisation in its newest
    file passed; its binding is that of its parametrisations (they ran in one session)."""
    out: dict[tuple[str, str], dict] = {}
    files = sorted(glob.glob(os.path.join(results_dir, "**", "*.xml"), recursive=True), key=os.path.getmtime)
    for f in files:
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        per_file: dict[tuple[str, str], dict] = {}
        for tc in tree.iter("testcase"):
            key = _junit_key(tc.get("classname", ""), tc.get("name", ""), tc.get("file"), root)
            failed = any(child.tag in ("failure", "error") for child in tc)
            skipped = any(child.tag == "skipped" for child in tc)
            props = {p.get("name"): p.get("value") for ps in tc.iter("properties") for p in ps.iter("property")}
            st = per_file.setdefault(key, {"passed": True, "skipped": False, "file": os.path.relpath(f, root),
                                           "props": {}, "entry_hits": {}, "integration": None})
            st["passed"] = st["passed"] and not failed and not skipped
            st["skipped"] = st["skipped"] or skipped
            for k in ("vc_commit", "vc_dirty", "vc_test_sha256"):
                if k in props:
                    st["props"][k] = props[k]
            if "vc_integration" in props:
                prev = st["integration"]
                st["integration"] = props["vc_integration"] if prev in (None, "clean") else prev
            if "vc_entry_hits" in props:
                try:
                    for e, n in json.loads(props["vc_entry_hits"]).items():
                        st["entry_hits"][e] = st["entry_hits"].get(e, 0) + int(n)
                except ValueError:
                    pass
        out.update(per_file)
    return out


def binding_problems(rec: dict, paths: list, root: str = ROOT) -> list:
    """Why a JUnit record does not bind to the current code of `paths` (test file + source files); [] = bound."""
    props = rec.get("props") or {}
    c = props.get("vc_commit")
    if not c:
        return ["not bound to a commit (recorded without -p superlearner.verify_completion)"]
    head = git_head(root)
    if head is None:
        return ["no git HEAD to bind the record to"]
    if c != head and not _is_ancestor(root, c, head):
        return [f"recorded at {c[:12]}, which is not an ancestor of HEAD {head[:12]}"]
    out = []
    try:
        dirty = set(json.loads(props.get("vc_dirty") or "[]"))
    except ValueError:
        dirty = {"<unreadable vc_dirty>"}
    bad = sorted(p for p in paths if p in dirty or any(p.startswith(d.rstrip("/") + "/") for d in dirty))
    if bad or "<unreadable vc_dirty>" in dirty or "<overflow>" in dirty:
        out.append(f"recorded with uncommitted changes to {bad[:3] or ['(unknown)']}")
    if c != head:
        ch = _changed_between(root, c, head, tuple(sorted(paths)))
        if ch:
            out.append(f"{list(ch)[:3]} changed between the recorded commit {c[:12]} and HEAD")
    now = _dirty_now(root, tuple(sorted(paths)))
    if now:
        out.append(f"{list(now)[:3]} modified in the working tree since the recorded run")
    return out


def test_passed(test_id: str, results: dict, paths: Optional[list] = None, root: str = ROOT,
                integration_entry: Optional[str] = None) -> tuple[bool, str]:
    """(ok, why). `paths`: source files the record must be bound to (the test file is always added).
    `integration_entry`: the record must be an integration record that hit this entry."""
    parts = _split_test_id(test_id)
    if parts is None:
        return False, "malformed test id"
    if parts[0] in GOVERNANCE_TESTS:
        return False, "a governance test is not evidence for a capability"
    rec = results.get(parts)
    if rec is None:
        return False, "never executed (no JUnit record)"
    if rec["skipped"]:
        return False, f"skipped ({rec['file']})"
    if not rec["passed"]:
        return False, f"failed ({rec['file']})"
    bp = binding_problems(rec, sorted({parts[0], *(paths or [])}), root)
    if bp:
        return False, f"{'; '.join(bp)} ({rec['file']})"
    if integration_entry is not None:
        if rec.get("integration") is None:
            return False, f"not run as an integration test by the plugin ({rec['file']})"
        if rec["integration"] != "clean":
            return False, f"test doubles loaded during the integration test: {rec['integration']} ({rec['file']})"
        if not rec.get("entry_hits", {}).get(integration_entry):
            return False, f"integration_entry {integration_entry} did not execute ({rec['file']})"
    return True, rec["file"]


def stub_findings(path: str, root: str = ROOT) -> list[str]:
    full = os.path.join(root, path)
    if not path.endswith(".py") or not os.path.isfile(full):
        return []
    with open(full) as fh:
        src = fh.read()
    found = []
    is_test = "/tests/" in path
    if not is_test and DOUBLE_IMPORT.search(src):
        found.append("production code imports test doubles")
    for pat, label in STUB_PATTERNS:
        if pat.search(src):
            found.append(label)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return found + [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                                                 and isinstance(b.value.value, str))]
            if not body:
                continue
            is_abstract = any((isinstance(d, ast.Name) and d.id == "abstractmethod") or
                              (isinstance(d, ast.Attribute) and d.attr == "abstractmethod") for d in node.decorator_list)
            if is_abstract:
                continue
            if len(body) == 1:
                b = body[0]
                if isinstance(b, ast.Raise) and b.exc is not None:
                    exc = b.exc.func if isinstance(b.exc, ast.Call) else b.exc
                    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                        found.append(f"{node.name}() raises NotImplementedError")
    return found


def _is_source(path: str) -> bool:
    return path.endswith(".py") and "/src/" in path and "/tests/" not in path


def _entry_problems(entry: str, root: str) -> list:
    mod_path, _, fn = entry.partition(":")
    f = _inside(root, mod_path)
    if f is None or not os.path.isfile(f):
        return [f"integration_entry file missing: {mod_path}"]
    if not _is_source(mod_path):
        return [f"integration_entry must be a function of the runtime under */src/ (not a test): {entry}"]
    if not fn:
        return [f"integration_entry names no function: {entry}"]
    if not re.search(rf"^\s*(async\s+)?def {re.escape(fn.split('.')[-1])}\(", open(f).read(), re.M):
        return [f"integration_entry function missing: {entry}"]
    return []


def evidence_ok(e, root: str = ROOT) -> bool:
    """A resolvable repository path (optionally #anchor), sha256:<64 hex>, or a resolvable test id."""
    if not isinstance(e, str):
        return False
    if re.fullmatch(r"sha256:[0-9a-f]{64}", e):
        return True
    if "::" in e:
        return test_resolves(e, root)
    p = e.split("#", 1)[0]
    full = _inside(root, p)
    return full is not None and os.path.exists(full)


def verification_problems(r: dict, root: str = ROOT) -> list:
    v = r.get("verified_by")
    if not v:
        return ["not independently verified (verified_by is empty)"]
    want_dir = VERIFICATION_DIR.replace(os.sep, "/") + "/"
    if not isinstance(v, str) or not v.replace(os.sep, "/").startswith(want_dir) or not v.endswith(".json"):
        return [f"verified_by must be a verification record {want_dir}<id>.json, got {v!r}"]
    f = _inside(root, v)
    if f is None or not os.path.isfile(f):
        return [f"verification record missing: {v}"]
    try:
        rec = json.load(open(f))
    except ValueError as e:
        return [f"verification record unreadable: {e}"]
    out = []
    for k in ("requirement", "verifier", "workstream", "commit", "commands", "measured"):
        if not rec.get(k):
            out.append(f"verification record {v} lacks {k!r}")
    if rec.get("requirement") and rec["requirement"] != r["id"]:
        out.append(f"verification record {v} is for {rec['requirement']}, not {r['id']}")
    if rec.get("workstream") and rec["workstream"] == r.get("workstream"):
        out.append(f"verification by the implementing workstream {rec['workstream']} is not independent")
    c = rec.get("commit")
    head = git_head(root)
    if c and head and c != head and not _is_ancestor(root, c, head):
        out.append(f"verification commit {str(c)[:12]} is not an ancestor of HEAD")
    return out


# ----------------------------------------------------------------------------------------------- experiments

def _rows(path: str) -> list:
    out = []
    with open(path) as fh:
        if path.endswith(".jsonl"):
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        else:
            d = json.load(fh)
            out = d if isinstance(d, list) else d.get("rows", []) if isinstance(d, dict) else []
    return [x for x in out if isinstance(x, dict)]


def recompute_verdict(primary: dict) -> Optional[str]:
    """The verdict the pre-registered decision rule gives on the recorded primary interval (stats.verdict)."""
    lo, hi, d = primary.get("lo"), primary.get("hi"), primary.get("direction", ">")
    m = float(primary.get("margin", 0.0) or 0.0)
    if lo is None or hi is None or d not in (">", "<"):
        return "INCONCLUSIVE" if d in (">", "<") else None
    if d == ">":
        return "DEMONSTRATED" if lo > m else ("FAILED" if hi <= m else "INCONCLUSIVE")
    return "DEMONSTRATED" if hi < m else ("FAILED" if lo >= m else "INCONCLUSIVE")


def check_experiment_result(path: str, claim: Optional[str] = None, root: str = ROOT) -> list:
    """Problems of one experiment result file (schema in the module docstring and plan 2026-11-26 §5)."""
    f = _inside(root, path)
    if f is None:
        return [f"experiment_result outside the repository: {path}"]
    if not os.path.isfile(f):
        return [f"experiment_result missing: {path}"]
    try:
        d = json.load(open(f))
    except ValueError as e:
        return [f"experiment_result unreadable: {e}"]
    if not isinstance(d, dict):
        return ["experiment_result is not a JSON object"]
    out = []
    verdict = d.get("verdict")
    if verdict not in VERDICTS:
        out.append(f"experiment_result verdict {verdict!r} is not one of {VERDICTS}")
    elif claim is not None and verdict not in CLAIM_VERDICTS.get(claim, set()):
        out.append(f"experiment_result verdict {verdict!r} does not support {claim}")
    for k in ("experiment_id", "prereg_file", "prereg_commit", "set", "code_commit", "learners", "seeds",
              "n_worlds", "primary", "raw_files", "compute"):
        if k not in d:
            out.append(f"experiment_result lacks {k!r}")
    if out and any("lacks" in p for p in out):
        return out
    if d["set"] != "confirm":
        out.append(f"experiment_result set is {d['set']!r}; empirical claims rest on the confirm set")
    prim = d["primary"] if isinstance(d["primary"], dict) else {}
    for k in ("name", "direction", "estimate", "lo", "hi", "method"):
        if k not in prim:
            out.append(f"primary lacks {k!r}")
    want = recompute_verdict(prim)
    if verdict in VERDICTS and verdict != "BLOCKED" and want is not None and want != verdict:
        out.append(f"verdict {verdict} disagrees with the primary interval [{prim.get('lo')}, {prim.get('hi')}] "
                   f"(direction {prim.get('direction')!r}, margin {prim.get('margin', 0)}): the rule gives {want}")
    seeds = d["seeds"] if isinstance(d["seeds"], list) else []
    if verdict == "DEMONSTRATED" and len(seeds) < MIN_SEEDS_DEMONSTRATED:
        out.append(f"a DEMONSTRATED verdict needs >= {MIN_SEEDS_DEMONSTRATED} training seeds, got {len(seeds)}")
    # learners
    for L in d["learners"] if isinstance(d["learners"], list) else []:
        tag = str(L.get("tag", "?"))
        for k in ("tag", "ckpt_sha256", "cfg_hash", "scale", "trainable_params", "semantic_model", "reward_mode"):
            if k not in L:
                out.append(f"learner {tag} lacks {k!r}")
        if (L.get("kind") == "cam" or tag.startswith("cam")) and L.get("reward_mode") != "observation_only":
            out.append(f"CAM learner {tag} ran with reward_mode {L.get('reward_mode')!r}, not observation_only")
        if L.get("scale") == "dom":
            cfg = L.get("cfg") or {}
            bad = {k: cfg.get(k) for k, v in MAIN_SCALE.items() if cfg.get(k, L.get(k)) != v}
            if bad:
                out.append(f"learner {tag} claims scale 'dom' but its configuration differs: {bad}")
            core = L.get("trainable_core_params")
            if not isinstance(core, (int, float)) or not 0.9e9 <= core <= 1.1e9:
                out.append(f"learner {tag} claims scale 'dom' with trainable core params {core!r} (need 0.9-1.1e9)")
    # raw files: hashes, confirm rows, exactly-once evaluation
    confirm_t, runs = [], defaultdict(set)
    for rf in d["raw_files"] if isinstance(d["raw_files"], list) else []:
        p = _inside(root, rf.get("path", ""))
        if p is None:
            out.append(f"raw file outside the repository: {rf.get('path')}")
            continue
        if not os.path.isfile(p):
            out.append(f"raw file missing: {rf.get('path')}")
            continue
        if _sha256(p) != rf.get("sha256"):
            out.append(f"raw file {rf.get('path')} does not match its recorded sha256")
            continue
        for row in _rows(p):
            if row.get("set") != "confirm":
                continue
            t = row.get("t_wall", row.get("timestamp"))
            if isinstance(t, (int, float)):
                confirm_t.append(float(t))
            if "learner" in row and "seed" in row:
                runs[(str(row["learner"]), str(row["seed"]))].add(str(row.get("run_id", "")))
    for (lr, s), ids in sorted(runs.items()):
        if len(ids) != 1:
            out.append(f"learner {lr} seed {s} was evaluated on the confirm set {len(ids)} times (runs {sorted(ids)})")
    if not confirm_t:
        out.append("no timestamped confirm-set rows in the raw files")
    # pre-registration before the first confirm row, unchanged afterwards
    pc, pf = str(d["prereg_commit"]), str(d["prereg_file"])
    head = git_head(root)
    if not _exists_at(root, pc, pf):
        out.append(f"pre-registration {pf} does not exist at commit {pc[:12]}")
    else:
        if head and pc != head and not _is_ancestor(root, pc, head):
            out.append(f"pre-registration commit {pc[:12]} is not an ancestor of HEAD")
        cc = str(d["code_commit"])
        if cc != pc and not _is_ancestor(root, pc, cc):
            out.append(f"pre-registration commit {pc[:12]} is not an ancestor of the code commit {cc[:12]}")
        ct = _commit_time(root, pc)
        if confirm_t and ct is not None:
            first = min(confirm_t)
            if ct >= first:
                out.append(f"pre-registration committed at {ct:.0f}, after the first confirm row at {first:.0f}")
            amended = _commits_touching_after(root, pf, first)
            if amended:
                out.append(f"pre-registration {pf} modified after the confirm evaluation started: {amended[:3]}")
    return out


def check_register(root: str = ROOT) -> list:
    """Register rows of CAM_UNIFIED_EXPERIMENTS.md against the docs/experiments/U-*.md pre-registrations."""
    reg = _inside(root, REGISTER)
    if reg is None or not os.path.isfile(reg):
        return []
    rows = {}
    for line in open(reg):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(r"U-[A-Z0-9]+(-[0-9A-Za-z]+)*", cells[0] or ""):
            rows[cells[0]] = cells
    out = []
    pdir = _inside(root, PREREG_DIR)
    files = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(pdir or "", "U-*.md"))
                   if not p.endswith("_results.md"))
    for pid in files:
        if pid not in rows:
            out.append(f"register: pre-registration {pid}.md has no row in {REGISTER}")
    for pid, cells in rows.items():
        status = cells[2].lower()
        if not status.startswith("declared"):
            if pid in files:
                out.append(f"register: {pid} is '{cells[2]}' but its pre-registration {pid}.md exists")
            continue
        rel = f"{PREREG_DIR}/{pid}.md".replace(os.sep, "/")
        if pid not in files:
            out.append(f"register: {pid} is '{cells[2]}' but {rel} does not exist")
            continue
        m = re.search(r"\b([0-9a-f]{7,40})\b", cells[3])
        if not m:
            out.append(f"register: {pid} is '{cells[2]}' without a declaring commit")
            continue
        c = _git(root, "rev-parse", m.group(1) + "^{commit}")
        if not c:
            out.append(f"register: {pid} declaring commit {m.group(1)} does not exist")
            continue
        c = c.strip()
        if not _exists_at(root, c, rel):
            out.append(f"register: {rel} does not exist at its declaring commit {c[:12]}")
        head = git_head(root)
        if head and c != head and not _is_ancestor(root, c, head):
            out.append(f"register: {pid} declaring commit {c[:12]} is not an ancestor of HEAD")
    return out


def _imports_module(src: str, importer_rel: str, target_rel: str) -> bool:
    """True when the source of `importer_rel` imports the package module `target_rel` (absolute, relative or as a
    name imported from its package)."""
    pkg_root = "superlearner/src/"
    if not target_rel.startswith(pkg_root) or not importer_rel.startswith(pkg_root):
        return False
    tgt = target_rel[len(pkg_root):-3].split("/")             # e.g. superlearner, cam_unified, core, bench
    imp = importer_rel[len(pkg_root):-3].split("/")
    mod, parent = tgt[-1], ".".join(tgt[:-1])
    pats = [rf"^\s*import\s+{re.escape('.'.join(tgt))}\b", rf"^\s*from\s+{re.escape('.'.join(tgt))}\s+import",
            rf"^\s*from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{mod}\b"]
    # relative forms: k dots lead from the importer's package up k - 1 levels to a base package of the target
    ip = imp[:-1]
    for k in range(1, len(ip) + 1):
        base = ip[:len(ip) - (k - 1)]
        if tgt[:len(base)] != base:
            continue
        rest = ".".join(tgt[len(base):-1])
        dots = "." * k
        pats += [rf"^\s*from\s+{re.escape(dots + (rest + '.' if rest else '') + mod)}\s+import",
                 rf"^\s*from\s+{re.escape(dots + rest)}\s+import\s+[^\n]*\b{mod}\b"]
    return any(re.search(p_, src, re.M) for p_ in pats)


def check_reward_integrity(root: str = ROOT) -> list:
    """Static scan: actor-side modules never import the evaluator / label channel, and never import an exempt
    module (ACTOR_SIDE_EXEMPT: operator-side harnesses that live under an actor-side package)."""
    out = []
    for spec in ACTOR_SIDE:
        full = os.path.join(root, spec)
        files = sorted(glob.glob(os.path.join(full, "**", "*.py"), recursive=True)) if spec.endswith("/") else \
            ([full] if os.path.isfile(full) else [])
        for f in files:
            rel = os.path.relpath(f, root).replace(os.sep, "/")
            if rel in ACTOR_SIDE_EXEMPT:
                continue
            src = open(f).read()
            for ex in sorted(ACTOR_SIDE_EXEMPT):
                if _imports_module(src, rel, ex):
                    out.append(f"reward integrity: actor-side module {rel} imports the exempt operator-side module "
                               f"{ex}")
            m = EVALUATOR_SIDE.search(src)
            if m:
                out.append(f"reward integrity: actor-side module {rel} references the evaluator / label channel "
                           f"({m.group(0).strip()[:60]!r})")
    return out


# ----------------------------------------------------------------------------------------------- checking

def check(manifest: dict, results: dict, root: str = ROOT, global_checks: bool = True) -> dict:
    clear_caches()
    reqs = manifest["requirements"]
    ids = [r["id"] for r in reqs]
    problems: dict[str, list[str]] = defaultdict(list)
    global_problems: list[str] = []

    if len(ids) != len(set(ids)):
        global_problems.append("duplicate requirement ids: " + ", ".join(k for k, v in Counter(ids).items() if v > 1))
    for rid in REQUIRED_IDS:
        if rid not in ids:
            global_problems.append(f"mandatory requirement {rid} missing from the manifest")
    for pre in REQUIRED_PREFIXES:
        if not any(i.startswith(pre) for i in ids):
            global_problems.append(f"no requirement with prefix {pre}")
    if global_checks:
        global_problems += check_register(root)
        global_problems += check_reward_integrity(root)
    allowed = set(manifest.get("allowed_statuses", STATUSES)) & set(STATUSES)

    for r in reqs:
        rid, st = r["id"], r["status"]
        p = problems[rid]
        if st not in allowed:
            p.append(f"unknown status {st}")
            continue
        if r.get("workstream") not in manifest.get("workstreams", {}):
            p.append(f"unknown workstream {r.get('workstream')}")
        for d in r.get("dependencies", []):
            if d not in ids:
                p.append(f"unknown dependency {d}")
        files = list(r.get("files", []))
        src_files = [f for f in files if _is_source(f)]
        if st in AT_LEAST_IMPLEMENTED:
            if not files:
                p.append("claimed implemented without implementation files")
            elif not src_files:
                p.append("claimed implemented without a source file under */src/ (documents and tests are not "
                         "implementations)")
            for f in files:
                full = _inside(root, f)
                if full is None or not os.path.exists(full):
                    p.append(f"missing file {f}")
                else:
                    for s in stub_findings(f, root):
                        p.append(f"{f}: {s}")
            acc = r.get("acceptance_tests", [])
            if not any(test_resolves(t, root) for t in acc):
                p.append("no resolvable acceptance test")
            for t in acc:
                if not test_resolves(t, root):
                    p.append(f"acceptance test does not resolve: {t}")
                    continue
                ok, why = test_passed(t, results, src_files, root)
                if not ok:
                    p.append(f"acceptance test {t}: {why}")
            for t in r.get("gradient_tests", []):
                ok, why = test_passed(t, results, src_files, root) if test_resolves(t, root) else \
                    (False, "does not resolve")
                if not ok:
                    p.append(f"gradient-pathway test {t}: {why}")
        if st in AT_LEAST_INTEGRATED:
            entry = r.get("integration_entry")
            if not entry:
                p.append("no integration_entry (function of the canonical runtime that exercises it)")
            else:
                p += _entry_problems(entry, root)
            integ = r.get("integration_tests", [])
            if not any(test_resolves(t, root) for t in integ):
                p.append("no resolvable integration test")
            good = 0
            for t in integ:
                if test_resolves(t, root):
                    ok, why = test_passed(t, results, src_files, root, integration_entry=entry or "")
                    good += ok
                    if not ok:
                        p.append(f"integration test {t}: {why}")
            if integ and not good:
                p.append("no integration test passed as a clean, commit-bound integration run")
            p += verification_problems(r, root)
            for e in r.get("evidence", []):
                if not evidence_ok(e, root):
                    p.append(f"evidence entry is not a resolvable path, sha256 or test id: {str(e)[:80]!r}")
        if st in EMPIRICAL:
            er = r.get("experiment_result")
            if not er:
                p.append("empirical claim without experiment_result")
            else:
                p += check_experiment_result(er, st, root)
            if not r.get("evidence"):
                p.append("empirical claim without evidence entries")

    by_status = Counter(r["status"] for r in reqs)
    eng_complete = [r["id"] for r in reqs if r["status"] in AT_LEAST_INTEGRATED and not problems[r["id"]]]
    incomplete = [r["id"] for r in reqs if r["id"] not in eng_complete]
    report = {
        "manifest": os.path.relpath(MANIFEST, ROOT),
        "head": git_head(root),
        "n_requirements": len(reqs),
        "by_status": dict(by_status),
        "global_problems": global_problems,
        "problems": {k: v for k, v in problems.items() if v},
        "engineering_complete": eng_complete,
        "engineering_incomplete": incomplete,
        "scientific": {
            "demonstrated": [r["id"] for r in reqs if r["status"] == "EMPIRICALLY_DEMONSTRATED" and not problems[r["id"]]],
            "failed": [r["id"] for r in reqs if r["status"] == "FAILED_EMPIRICALLY" and not problems[r["id"]]],
            "blocked": [r["id"] for r in reqs if r["status"] == "BLOCKED"],
        },
        "test_records": len(results),
    }
    report["claims_consistent"] = not global_problems and not report["problems"]
    report["project_complete"] = report["claims_consistent"] and not incomplete
    return report


def referenced_tests(manifest: dict, only: Optional[set] = None, root: str = ROOT) -> tuple:
    """(unit-level tests, {integration test id: [entries]}) referenced by claims of IMPLEMENTED or above."""
    unit, integ = set(), defaultdict(set)
    for r in manifest["requirements"]:
        if only and r["id"] not in only:
            continue
        if r["status"] not in AT_LEAST_IMPLEMENTED:
            continue
        for key in ("acceptance_tests", "gradient_tests"):
            unit |= {t for t in r.get(key, []) if test_resolves(t, root)}
        for t in r.get("integration_tests", []):
            if test_resolves(t, root):
                if r.get("integration_entry"):
                    integ[t].add(r["integration_entry"])
                else:
                    integ[t]
    unit -= set(integ)
    return sorted(unit), {t: sorted(v) for t, v in integ.items()}


def run_referenced_tests(manifest: dict, timeout: int = 7200, only: Optional[set] = None) -> int:
    """Execute every acceptance / integration / gradient test referenced by a claim of IMPLEMENTED or above with
    this module as a pytest plugin (commit-bound records) and record JUnit XML under docs/evidence/test_results/.
    Integration tests run in their own process, so that doubles imported by unit tests cannot mask or fake their
    cleanliness; slow and gpu tests are included (no marker deselection)."""
    import time
    unit, integ = referenced_tests(manifest, only)
    if not unit and not integ:
        print("no referenced tests to run")
        return 0
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def node(t):     # pytest node ids are relative to the superlearner project dir (rootdir)
        return t[len("superlearner/"):] if t.startswith("superlearner/") else t
    rc = 0
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "superlearner.verify_completion",
            "-o", "junit_family=xunit1"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "superlearner", "src"), os.path.join(ROOT, "crucible", "src"),
                                         env.get("PYTHONPATH", "")])
    if unit:
        out = os.path.join(RESULTS_DIR, f"verify_{stamp}_unit.xml")
        print(f"running {len(unit)} tests -> {os.path.relpath(out, ROOT)}")
        rc |= subprocess.call(base + [f"--junitxml={out}", *sorted(node(t) for t in unit)],
                              cwd=os.path.join(ROOT, "superlearner"), timeout=timeout, env=env)
    if integ:
        out = os.path.join(RESULTS_DIR, f"verify_{stamp}_integration.xml")
        lst = os.path.join(RESULTS_DIR, f".integration_{stamp}.txt")
        with open(lst, "w") as fh:
            for t, entries in sorted(integ.items()):
                fh.write(f"{node(t)}\t{','.join(entries)}\n")
        print(f"running {len(integ)} integration tests -> {os.path.relpath(out, ROOT)}")
        rc |= subprocess.call(base + [f"--junitxml={out}", f"--vc-integration={lst}",
                                      *sorted(node(t) for t in integ)],
                              cwd=os.path.join(ROOT, "superlearner"), timeout=timeout, env=env)
        os.remove(lst)
    return rc


# ----------------------------------------------------------------------------------------------- pytest plugin
# Loaded with `-p superlearner.verify_completion`. Hooks are inert unless pytest loads this module as a plugin.

_VC: dict = {}


def pytest_addoption(parser):
    parser.addoption("--vc-integration", default=None,
                     help="file of integration test node ids (optionally '<node id>\\t<entry>,<entry>'): those "
                          "tests fail when a test-double module is loaded, and record which entries executed")


def _resolve_entry(entry: str):
    """'superlearner/src/pkg/mod.py:func' or ':Class.method' -> code object (None when it cannot be resolved)."""
    import importlib
    mod_path, _, fn = entry.partition(":")
    rel = mod_path.replace("\\", "/")
    i = rel.find("/src/")
    if i < 0 or not fn:
        return None
    mod = rel[i + 5:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    try:
        obj = importlib.import_module(mod)
        for part in fn.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    obj = getattr(obj, "__func__", obj)
    return getattr(obj, "__code__", None)


def pytest_configure(config):
    head = git_head(ROOT)
    dirty = dirty_paths(ROOT)
    if len(dirty) > 500:
        dirty = dirty[:500] + ["<overflow>"]
    _VC.update(commit=head or "", dirty=json.dumps(dirty), integ={}, codes={}, hits=Counter())
    path = config.getoption("--vc-integration", default=None)
    if path:
        for line in open(path):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            nid, _, entries = line.partition("\t")
            _VC["integ"][nid.strip()] = [e for e in entries.split(",") if e]
        mon = getattr(sys, "monitoring", None)
        entries = sorted({e for es in _VC["integ"].values() for e in es})
        if entries and mon is not None:
            tool = None
            for tid in range(1, 6):
                if mon.get_tool(tid) is None:
                    tool = tid
                    break
            if tool is not None:
                mon.use_tool_id(tool, "verify_completion")
                _VC["tool"] = tool
                for e in entries:
                    code = _resolve_entry(e)
                    if code is None:
                        continue
                    _VC["codes"][code] = e
                    mon.set_local_events(tool, code, mon.events.PY_START)

                def _start(code, offset):
                    e = _VC["codes"].get(code)
                    if e is not None:
                        _VC["hits"][e] += 1
                mon.register_callback(tool, mon.events.PY_START, _start)


def _module_imports_doubles(item) -> bool:
    f = getattr(getattr(item, "module", None), "__file__", None)
    paths = [f] if f else []
    d = os.path.dirname(f) if f else None
    while d and os.path.basename(d) != "tests" and d != os.path.dirname(d):
        paths.append(os.path.join(d, "conftest.py"))
        d = os.path.dirname(d)
    for p in paths:
        if p and os.path.isfile(p) and DOUBLE_IMPORT.search(open(p).read()):
            return True
    return False


def _integration_entries(item):
    nid = item.nodeid
    for key in (nid, nid.split("[", 1)[0]):
        if key in _VC.get("integ", {}):
            return _VC["integ"][key]
    return None


def pytest_runtest_setup(item):
    if _integration_entries(item) is not None:
        item._vc_before = {m for m in DOUBLE_MODULES if m in sys.modules}
        item._vc_hits0 = Counter(_VC["hits"])


def _pytest_hookwrapper(fn):
    try:
        import pytest
        return pytest.hookimpl(hookwrapper=True)(fn)
    except ImportError:                         # the checker itself runs without pytest
        return fn


@_pytest_hookwrapper
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call":
        return
    props = item.user_properties
    props.append(("vc_commit", _VC.get("commit", "")))
    props.append(("vc_dirty", _VC.get("dirty", "[]")))
    f = getattr(getattr(item, "module", None), "__file__", None)
    if f and os.path.isfile(f):
        props.append(("vc_test_sha256", _sha256(f)))
    entries = _integration_entries(item)
    if entries is None:
        return
    after = {m for m in DOUBLE_MODULES if m in sys.modules}
    new = sorted(after - getattr(item, "_vc_before", set()))
    static = _module_imports_doubles(item)
    bad = new + (["<module or conftest imports a double>"] if static else [])
    props.append(("vc_integration", "clean" if not bad else ",".join(bad)))
    h0 = getattr(item, "_vc_hits0", Counter())
    props.append(("vc_entry_hits", json.dumps({e: _VC["hits"][e] - h0.get(e, 0) for e in entries})))
    if bad and rep.passed:
        rep.outcome = "failed"
        rep.longrepr = (f"verify_completion: integration test {item.nodeid} ran with test doubles {bad}; "
                        "integration evidence must use the real components only")


# ----------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-tests", action="store_true", help="execute referenced tests and record JUnit results first")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--only", help="comma-separated requirement ids whose tests --run-tests executes")
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    if a.run_tests:
        run_referenced_tests(m, only=set(a.only.split(",")) if a.only else None)
    rep = check(m, load_results())
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    if not a.quiet:
        print(f"requirements: {rep['n_requirements']}  by status: {rep['by_status']}  HEAD {rep['head']}")
        print(f"JUnit test records: {rep['test_records']}")
        for g in rep["global_problems"]:
            print(f"GLOBAL: {g}")
        for rid, ps in rep["problems"].items():
            for p in ps:
                print(f"{rid}: {p}")
        print(f"engineering complete: {len(rep['engineering_complete'])}/{rep['n_requirements']}")
        print(f"scientific: demonstrated={rep['scientific']['demonstrated']} failed={rep['scientific']['failed']} "
              f"blocked={rep['scientific']['blocked']}")
        print("CLAIMS CONSISTENT" if rep["claims_consistent"] else "CLAIMS INCONSISTENT WITH EVIDENCE")
        print("PROJECT COMPLETE" if rep["project_complete"] else "PROJECT NOT COMPLETE")
    return 0 if rep["project_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
