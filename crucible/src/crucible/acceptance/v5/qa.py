"""Gate X and recovery-preamble tooling of the independent QA workstream (P). Every function measures the real tree.

* `scan_acceptance_tests`  X.7  placeholder / skip / assertion-free / mock / constant-record detector over every gate
                                test, and the manifest ids no test records.
* `clean_checkout`         X.3  fresh clone of a committed tree, documented commands (REPRODUCE.md, block
                                `clean-checkout`), and the clean-checkout reproduction of every VERIFIED requirement.
* `defects`                X.6  the machine-readable defect register docs/v5/DEFECTS.md (one ```json block per defect),
                                open critical/high count, every audit critical finding registered.
* `integration_audit`      X.8  every workstream's package imported and exercised by an integrated world run.
* `claims_audit`           X.9  every numeric claim of the final report docs traceable to evidence.
* `documents`              X.5  required documents present with their required content.
* `cam_scan`               RP.1 no import of CAM / superlearner / agentrs learner code anywhere in crucible/.
* `audit_reports`          RP.2 audit reports with skeptic reviews for all 15 areas.
* `remote_state`           RP.3/RP.5 pushed integration head, unpushed milestones, old-branch anchor, worktrees.
* `secrets_scan`           RP.6 credential patterns in the tree and in every reachable commit.
* `artifacts`              RP.4 the important-artifact list docs/v5/ARTIFACTS.json: presence, hashes, durability.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

from . import manifest as M
from . import testscan

WORKSTREAM_PACKAGES = {   # WORKSTREAMS.md owned paths (one-world contracts: CORE-POP = B+C, ECON = F+G, CIV = J)
    "A": ["crucible.substrate"], "B": ["crucible.population"], "C": ["crucible.social"], "D": ["crucible.semantic"],
    "E": ["crucible.calibration"], "F": ["crucible.commerce"], "G": ["crucible.innovation"], "H": ["crucible.rapid"],
    "I": ["crucible.domains"], "J": ["crucible.civ"], "K": ["crucible.multiagent"], "L": ["crucible.experience"],
    "M": ["crucible.scale"], "N": ["crucible.research"], "O": ["crucible.redteam"], "P": ["crucible.acceptance"],
}
AUDIT_AREAS = ("A_J_substrate_civ", "B_population", "C_social", "D_semantic", "E_calibration", "F_commerce",
               "G_innovation", "H_rapid", "I_domains", "K_selfplay", "L_experience", "M_performance", "N_research",
               "O_security", "P_acceptance")
OLD_BRANCH, OLD_ANCHOR = "feat/crucible-5-openworld", "a881d18"
INTEGRATION_BRANCH = "feat/crucible-5-complete"


def _git(root: str, *args: str, check: bool = False) -> str:
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout


def _rel(root: str, p: str) -> str:
    return os.path.relpath(p, root)


# ------------------------------------------------------------------------------------------------- X.7
def _recorded_ids(src: str, ids: set[str]) -> set[str]:
    tree = ast.parse(src)
    out, dynamic = set(), False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and testscan._is_record_call(n) and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.add(a.value)
            else:
                dynamic = True
    if dynamic:                       # record(rid, ...) inside a loop or helper: the ids named in the file count
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in ids:
                out.add(n.value)
    return out & ids


def scan_acceptance_tests(root: str | None = None, files: list[str] | None = None) -> dict:
    root = root or M.crucible_root()
    files = files or sorted(_rel(root, p) for p in glob.glob(os.path.join(root, "tests", "v5", "test_gate_*.py")))
    ids = {e["id"] for e in M.load(root)["requirements"]}
    funcs, bad, recorded = [], [], set()
    for f in files:
        src = open(os.path.join(root, f), encoding="utf-8").read()
        for r in testscan.scan_source(src):
            r["file"] = f
            funcs.append(r)
            if testscan.blocking(r["findings"]):
                bad.append(r)
        recorded |= _recorded_ids(src, ids)
    unrecorded = sorted(ids - recorded)
    by_kind: dict[str, int] = {}
    for r in bad:
        for k in testscan.blocking(r["findings"]):
            by_kind[k] = by_kind.get(k, 0) + 1
    return {"metrics": {"acceptance_test_files_scanned": len(files), "acceptance_tests_scanned": len(funcs),
                        "placeholder_or_skipped_acceptance_tests": len(bad),
                        "manifest_ids_without_recording_test": len(unrecorded)},
            "by_kind": by_kind, "flagged": [f"{r['file']}::{r['name']} {testscan.blocking(r['findings'])}" for r in bad],
            "unrecorded_ids": unrecorded}


# ------------------------------------------------------------------------------------------------- X.3
def documented_commands(root: str | None = None) -> list[str]:
    """The commands of the ```clean-checkout block(s) of REPRODUCE.md, one per line."""
    text = open(os.path.join(M.docs_dir(root), "REPRODUCE.md"), encoding="utf-8").read()
    out = []
    for block in re.findall(r"```clean-checkout\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def clean_checkout(root: str | None = None, commit: str = "HEAD", scratch: str | None = None,
                   reproduce_verified: bool = True, agent: str = "qa-clean-checkout", timeout: int = 3600) -> dict:
    """Fresh clone at `commit`; run every documented command there; reproduce every VERIFIED requirement."""
    from .checker import verify_all
    root = root or M.crucible_root()
    base = scratch or os.environ.get("CRUCIBLE_V5_SCRATCH") or tempfile.gettempdir()
    work = tempfile.mkdtemp(prefix="x3-", dir=base)
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    prefix = os.path.relpath(os.path.realpath(root), os.path.realpath(top))
    sha = _git(root, "rev-parse", commit).strip()
    dest = os.path.join(work, "clone")
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", top, dest], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", sha], cwd=dest, check=True)
    croot = os.path.join(dest, prefix)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",) and not k.startswith("CRUCIBLE_V5_")}
    env.update(PYTHONPATH=os.path.join(croot, "src"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               PATH=os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", ""))
    cmds = documented_commands(croot)
    results = []
    for c in cmds:
        p = subprocess.run(["bash", "-c", c], cwd=croot, env=env, capture_output=True, text=True, timeout=timeout)
        results.append({"command": c, "rc": p.returncode, "tail": (p.stdout + p.stderr)[-600:]})
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=dest, capture_output=True,
                            text=True).stdout.splitlines()
    reproduced, failed = [], []
    if reproduce_verified:
        res = verify_all(root, reproduce=True, agent=agent, scratch=work)
        for r in res["rows"]:
            if "reproduction" in r:
                (reproduced if r["status"] == "VERIFIED" else failed).append(r["id"])
        verified = [r["id"] for r in res["rows"] if r["status"] == "VERIFIED"] + failed
    else:
        verified = []
    bad_cmds = [r for r in results if r["rc"] != 0 and not r["command"].startswith("! ")]
    return {"metrics": {"clean_checkout_runs": 1, "documented_commands_run": len(results),
                        "documented_command_failures": len(bad_cmds),
                        "clean_checkout_failures": len(bad_cmds) + len(failed) + (1 if status else 0),
                        "verified_requirements_not_reproduced": len(set(verified) - set(reproduced))},
            "commit": sha, "commands": results, "clone_status_after": status[:10], "reproduced": reproduced,
            "reproduction_failed": failed, "work": work}


# ------------------------------------------------------------------------------------------------- X.6
DEFECT_FIELDS = ("id", "severity", "status", "area", "summary", "source", "owner", "requirements")
SEVERITIES = ("critical", "high", "medium", "low")
DEFECT_STATUSES = ("open", "fixed", "duplicate", "not_a_defect")


def parse_defects(root: str | None = None) -> list[dict]:
    try:
        text = open(os.path.join(M.docs_dir(root), "DEFECTS.md"), encoding="utf-8").read()
    except OSError:
        return []
    return [json.loads(b) for b in re.findall(r"```json\n(.*?)```", text, re.S)]


def audit_dir(root: str | None = None) -> str:
    return os.path.join(M.docs_dir(root), "audit")


def defects(root: str | None = None, audits: str | None = None) -> dict:
    root = root or M.crucible_root()
    ds = parse_defects(root)
    present = os.path.exists(os.path.join(M.docs_dir(root), "DEFECTS.md"))
    malformed = []
    for d in ds:
        if any(f not in d for f in DEFECT_FIELDS) or d.get("severity") not in SEVERITIES or \
                d.get("status") not in DEFECT_STATUSES:
            malformed.append(d.get("id"))
        elif d["status"] == "fixed" and not d.get("fixed_by"):
            malformed.append(d.get("id"))
        elif d["status"] in ("duplicate", "not_a_defect") and not d.get("resolution"):
            malformed.append(d.get("id"))
    ids = [d.get("id") for d in ds]
    if len(ids) != len(set(ids)):
        malformed.append("duplicate ids")
    open_ch = [d["id"] for d in ds if d.get("status") == "open" and d.get("severity") in ("critical", "high")]
    reg = {(d.get("source") or {}).get("audit"): set() for d in ds}
    for d in ds:
        s = d.get("source") or {}
        if s.get("finding") is not None:
            reg.setdefault(s.get("audit"), set()).add(int(s["finding"]))
    unregistered, audits_read = [], 0
    adir = audits or audit_dir(root)
    for f in sorted(glob.glob(os.path.join(adir, "*.json"))):
        a = json.load(open(f))
        audits_read += 1
        area = os.path.basename(f)[:-5]
        for i, _ in enumerate(a.get("critical_findings", [])):
            if i not in reg.get(area, set()):
                unregistered.append(f"{area}#{i}")
    return {"metrics": {"defect_register_present": present, "defects_registered": len(ds),
                        "defects_malformed": len(malformed), "open_critical_high_defects": len(open_ch),
                        "audit_reports_read": audits_read, "audit_critical_findings_unregistered": len(unregistered)},
            "open_critical_high": open_ch, "malformed": malformed, "unregistered": unregistered[:30]}


# ------------------------------------------------------------------------------------------------- X.6 re-validation
REVALIDATION_VERDICTS = ("still_open", "fixed", "not_reproducible")
SINCE = re.compile(r"<!--\s*revalidate-since:\s*([0-9a-f]{7,40})\s*-->")


def defects_header_and_since(root: str | None = None) -> tuple[str, str | None]:
    text = open(os.path.join(M.docs_dir(root), "DEFECTS.md"), encoding="utf-8").read()
    i = text.find("\n### ")
    header = text if i < 0 else text[:i + 1]
    m = SINCE.search(header)
    return header, (m.group(1) if m else None)


def _owners(d: dict) -> list[str]:
    """The workstream letters (or 'orchestrator') named in a defect's owner field."""
    o = str(d.get("owner") or "")
    out = sorted(set(re.findall(r"(?<![\w-])([A-P])(?![\w-])", o)))
    if "orchestrator" in o.lower():
        out.append("orchestrator")
    if "all workstreams" in o.lower():
        out.append("all")
    return out or [o]


def _regression_nodes(spec) -> list[str]:
    """`regression_tests` as node ids: a list, or a comma-separated string whose '::name' items reuse the previous
    item's file."""
    items = spec if isinstance(spec, list) else [x.strip() for x in str(spec or "").split(",") if x.strip()]
    out, last = [], None
    for it in items:
        if it.startswith("::") and last:
            it = last + it
        if "::" in it:
            last = it.split("::", 1)[0]
        out.append(it)
    return out


def _test_exists(root: str, node: str) -> bool:
    f, _, name = node.partition("::")
    path = os.path.join(root, f)
    if not name or not f.endswith(".py") or not os.path.exists(path):
        return False
    name = name.split("::")[-1].split("[", 1)[0]
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree))


def _in_history(root: str, commit: str, since: str | None = None) -> bool:
    if not commit or subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
                                    capture_output=True).returncode != 0:
        return False
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root).returncode != 0:
        return False
    return since is None or subprocess.run(["git", "merge-base", "--is-ancestor", since, commit], cwd=root).returncode == 0


def defect_revalidation(root: str | None = None) -> dict:
    """The re-validation duty of the register (wave 3b, D-83): every OPEN defect carries, in `revalidations`, a
    re-validation by one of its owners made at a commit at or after the register's `revalidate-since` commit (the
    start of the current wave, an HTML comment in DEFECTS.md's header): its verdict (`still_open`, `fixed`,
    `not_reproducible`) and reason. A defect an owner reports fixed or not reproducible while it is still open is
    listed for closing. Every FIXED defect names what fixed it (`fixed_by`) and regression tests (`regression_tests`)
    that exist in the tree; a defect is never closed without both. Returns metrics plus, per owner, what is due."""
    root = root or M.crucible_root()
    ds = parse_defects(root)
    _, since = defects_header_and_since(root)
    due: dict[str, list[str]] = {}
    not_reval, to_close, bad_reval, fixed_no_test = [], [], [], []
    for d in ds:
        if d.get("status") == "open":
            revs = [r for r in d.get("revalidations") or [] if isinstance(r, dict)]
            owners = _owners(d)
            ok = [r for r in revs if r.get("verdict") in REVALIDATION_VERDICTS and (r.get("reason") or "").strip()
                  and _in_history(root, r.get("commit") or "", since) and
                  (r.get("by") in owners or "all" in owners or any(o in str(r.get("by")) for o in owners))]
            bad_reval += [f"{d['id']}: {r}" for r in revs if r not in ok][:3]
            if not ok:
                not_reval.append(d["id"])
                for o in owners:
                    due.setdefault(o, []).append(d["id"])
            elif ok[-1]["verdict"] != "still_open":
                to_close.append(f"{d['id']} ({ok[-1]['verdict']} per {ok[-1]['by']})")
        elif d.get("status") == "fixed":
            nodes = _regression_nodes(d.get("regression_tests"))
            if not d.get("fixed_by") or not nodes or not all(_test_exists(root, n) for n in nodes):
                fixed_no_test.append(d["id"])
    return {"metrics": {"defects_checked": len(ds), "open_defects_not_revalidated": len(not_reval),
                        "open_defects_reported_fixed_by_owner": len(to_close),
                        "fixed_defects_without_regression_test": len(fixed_no_test),
                        "revalidation_base_set": bool(since)},
            "since": since, "due_by_owner": {k: v for k, v in sorted(due.items())}, "not_revalidated": not_reval,
            "to_close": to_close, "fixed_without_regression_test": fixed_no_test, "invalid_revalidations": bad_reval}


def _rewrite_defects(root: str, ds: list[dict]) -> None:
    header, _ = defects_header_and_since(root)
    with open(os.path.join(M.docs_dir(root), "DEFECTS.md"), "w", encoding="utf-8") as fh:
        fh.write(render_defects(ds, header).rstrip() + "\n")


def revalidate_defect(root: str | None, defect: str, *, by: str, verdict: str, reason: str,
                      evidence: str = "") -> dict:
    """Record an owner's re-validation of an open defect at HEAD (DEFECTS.md is rewritten, entry order kept)."""
    root = root or M.crucible_root()
    if verdict not in REVALIDATION_VERDICTS:
        raise ValueError(f"verdict must be one of {REVALIDATION_VERDICTS}")
    if not reason.strip():
        raise ValueError("a re-validation states its reason")
    ds = parse_defects(root)
    d = next((x for x in ds if x.get("id") == defect), None)
    if d is None:
        raise KeyError(defect)
    head = _git(root, "rev-parse", "HEAD").strip()
    import datetime as _dt
    d.setdefault("revalidations", []).append({"by": by, "commit": head, "verdict": verdict, "reason": reason,
                                              "evidence": evidence,
                                              "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    _rewrite_defects(root, ds)
    return d


def close_defect(root: str | None, defect: str, *, fixed_by: str, regression_tests: list[str], note: str = "") -> dict:
    """Close a defect: only with what fixed it (a commit in HEAD's history) and regression tests that exist."""
    root = root or M.crucible_root()
    ds = parse_defects(root)
    d = next((x for x in ds if x.get("id") == defect), None)
    if d is None:
        raise KeyError(defect)
    sha = fixed_by.split()[0]
    if not _in_history(root, sha):
        raise ValueError(f"fixed_by must start with a commit in HEAD's history, got {fixed_by!r}")
    missing = [n for n in regression_tests if not _test_exists(root, n)]
    if not regression_tests or missing:
        raise ValueError(f"a defect is closed only with existing regression tests (missing: {missing or 'none given'})")
    d.update(status="fixed", fixed_by=fixed_by, regression_tests=list(regression_tests))
    if note:
        d["notes"] = (d.get("notes", "") + " " + note).strip()
    _rewrite_defects(root, ds)
    return d


def render_defects(entries: list[dict], header: str) -> str:
    out = [header.rstrip(), ""]
    for d in entries:
        out += [f"### {d['id']} ({d['severity']}, {d['status']}): {d['summary'][:110]}", "",
                "```json", json.dumps(d, indent=1, sort_keys=True), "```", ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------- X.8
INTEGRATION_SCRIPT = r'''
import collections, json, os, sys, time
import importlib
import crucible
crucible.load_all_modules()
for _p in crucible.V5_PACKAGES + ("scale", "research", "redteam", "acceptance"):
    try:
        importlib.import_module("crucible." + _p)
    except Exception:
        pass
out = {"errors": [], "worlds": []}
SRC = os.path.dirname(os.path.abspath(crucible.__file__)) + os.sep
executed = collections.Counter()           # distinct functions of each package executed after import
M = sys.monitoring
TOOL = next(i for i in (4, 3) if M.get_tool(i) is None)   # leaves the other one to the sandbox work counter
M.use_tool_id(TOOL, "crucible-qa")
def _start(code, offset):
    f = code.co_filename
    if f.startswith(SRC):
        executed[f[len(SRC):].split(os.sep, 1)[0].removesuffix(".py")] += 1
    return M.DISABLE
M.register_callback(TOOL, M.events.PY_START, _start)
M.set_events(TOOL, M.events.PY_START)
try:
    from crucible import civ
    profiles = sorted(getattr(civ, "PROFILES", {}) or [])
    if not profiles:
        raise RuntimeError("crucible.civ.PROFILES is empty or missing (no integrated civilization generator)")
    prof = profiles[0] if sys.argv[1] == "auto" else sys.argv[1]
    w = civ.generate(prof, 0, population="development")
    w.advance_time(w.now + int(float(sys.argv[2]) * 86400))
    by_pkg = collections.Counter()
    owner = {}
    for name, mod in w.modules.items():
        owner[name] = type(mod).__module__
    for e in w.log:
        mod = owner.get(getattr(e, "module", None) or (e.type.split(".", 1)[0] if "." in e.type else ""), "")
        if mod:
            by_pkg[".".join(mod.split(".")[:2])] += 1
    out["worlds"].append({"profile": prof, "events": len(w.log), "modules": sorted(set(owner.values())),
                          "events_by_package": dict(by_pkg)})
except Exception as exc:
    out["errors"].append(f"civ: {type(exc).__name__}: {exc}")
try:
    from crucible import experience
    for tier, name in ((1, sys.argv[3]), (3, sys.argv[4])):
        if not name:
            continue
        env = experience.make(tier, name, 0, reward_mode="observation_only")
        for a in env.agents():
            env.observe(a)
        env.advance(3600)
        out["worlds"].append({"tier": tier, "name": name, "ok": True})
except Exception as exc:
    out["errors"].append(f"experience: {type(exc).__name__}: {exc}")
M.set_events(TOOL, 0)
M.free_tool_id(TOOL)
out["imported"] = sorted({m.split(".")[1] for m in sys.modules if m.startswith("crucible.") and m.count(".") >= 1})
out["executed"] = dict(executed)
print("@@JSON@@" + json.dumps(out))
'''


def integration_audit(root: str | None = None, profile: str = "auto", days: float = 3.0,
                      tier1: str = "mech:scm_additive", tier3: str = "") -> dict:
    """Run an integrated civilization (crucible.civ.generate) plus the experience interface in a fresh process and
    check every workstream's package is imported and, for world packages, produced events."""
    root = root or M.crucible_root()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.path.join(root, "src"), OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", INTEGRATION_SCRIPT, profile, str(days), tier1, tier3 or ""],
                       cwd=root, env=env, capture_output=True, text=True, timeout=7200)
    res = json.loads(p.stdout.rsplit("@@JSON@@", 1)[1]) if "@@JSON@@" in p.stdout else {
        "errors": [f"integration probe crashed: {p.stderr[-800:]}"], "worlds": [], "imported": []}
    civ = next((w for w in res["worlds"] if "profile" in w), None)
    imported = set(res.get("imported", []))
    executed = res.get("executed", {})
    world_pkgs = {"A", "B", "C", "D", "F", "G", "I", "J"}          # these must produce events in the civilization
    verifiers = {"O", "P"}           # run against the integrated system rather than inside it: reachability only
    per = {}
    for ws, pkgs in WORKSTREAM_PACKAGES.items():
        names = [p_.split(".")[1] for p_ in pkgs]
        imp = all(n in imported for n in names)
        events = sum((civ or {}).get("events_by_package", {}).get(p_, 0) for p_ in pkgs)
        calls = sum(executed.get(n, 0) for n in names)
        if ws in verifiers:
            ok = imp and _targets_integrated_world(root, names[0])
        else:
            ok = imp and bool(civ) and calls > 0 and (ws not in world_pkgs or events > 0)
        per[ws] = {"imported": imp, "events": events, "functions_executed": calls, "connected": ok}
    disconnected = sorted(ws for ws, v in per.items() if not v["connected"])
    return {"metrics": {"workstreams_checked": len(per), "disconnected_workstreams": len(disconnected),
                        "integrated_world_runs": 1 if civ else 0},
            "per_workstream": per, "disconnected": disconnected, "errors": res["errors"], "worlds": res["worlds"]}


def _targets_integrated_world(root: str, pkg: str) -> bool:
    """A verification package is connected when its code drives the integrated generator or the Tier 3 interface."""
    d = os.path.join(root, "src", "crucible", pkg)
    for dp, _, fn in os.walk(d):
        for f in fn:
            if f.endswith(".py"):
                t = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
                if re.search(r"civ\.generate\(|make\(\s*3\s*,|tier3:", t):
                    return True
    return False


# ------------------------------------------------------------------------------------------------- X.9
TAG = re.compile(r"\[(ev|checker|manifest|run):([^\]]*)\]")
NUM = re.compile(r"(?<![\w.#/-])-?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?(?![\w-])")
STRUCTURAL = re.compile(r"((Part|Gate|Phase|Clarification|Crucible)\s+\d+(\.\d+)?|\b(D|BLK|R|DEF|M)-[\w-]*\d+\b|"
                        r"\bSP-\d+(\.\d+)*[a-z]?\b|\b(OO|RP|[A-X])\.n?\d+\b|\bM\d+\b|\bv\d+\b|"
                        r"\d{4}-\d{2}-\d{2}(T[\d:+.]+)?|#\d+|\b[0-9a-f]{7,40}\b|\b\d+(\.\d+){2,}\b|[\w./-]*\d[\w./-]*\.(py|md|json|xml|jsonl)\b)")


def _numbers(line: str) -> list[str]:
    clean = STRUCTURAL.sub(" ", TAG.sub(" ", re.sub(r"`[^`]*`", " ", re.sub(r"\([^)]*\.(json|md|xml|py)[^)]*\)", " ",
                                                                             line))))
    return [m.group(0) for m in NUM.finditer(clean)]


def claims_audit(root: str | None = None, paths: list[str] | None = None, rows: dict | None = None) -> dict:
    """Numeric claims of the final report documents. A number is traceable when its line carries a tag
    [ev:<id>#<metric>] whose evidence metric equals it, or [checker:...] / [manifest:...] / [run:...] (counts the
    report generator derives from the checker result, the manifest or a run record). Untagged numbers are claims
    without evidence."""
    root = root or M.crucible_root()
    paths = paths or ["docs/v5/FINAL_REPORT.md"]
    present = all(os.path.exists(os.path.join(root, p)) for p in paths)
    checked = without = mism = 0
    bad = []
    for p in paths:
        full = os.path.join(root, p)
        if not os.path.exists(full):
            continue
        in_code = False
        for k, line in enumerate(open(full, encoding="utf-8"), 1):
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.lstrip().startswith("#"):
                continue
            nums = _numbers(line)
            tags = TAG.findall(line)
            if not nums and not tags:
                continue
            checked += len(nums) + sum(1 for t in tags if t[0] == "ev")
            if nums and not tags:
                without += len(nums)
                bad.append(f"{p}:{k}: untagged {nums[:3]}")
                continue
            for kind, ref in tags:
                if kind != "ev":
                    continue
                rid, _, metric = ref.partition("#")
                ev_path = os.path.join(root, "docs", "v5", "evidence",
                                       "gate_" + ("SP" if rid.startswith("SP-") else rid.split(".")[0]), f"{rid}.json")
                try:
                    val = json.load(open(ev_path))["metrics"][metric]
                except (OSError, ValueError, KeyError):
                    mism += 1
                    bad.append(f"{p}:{k}: {rid}#{metric} has no evidence")
                    continue
                if not any(_same(n, val) for n in nums):
                    mism += 1
                    bad.append(f"{p}:{k}: {rid}#{metric}={val!r} not stated")
    return {"metrics": {"final_report_present": present, "report_claims_checked": checked,
                        "report_claims_without_evidence": without, "report_claims_mismatching_evidence": mism},
            "problems": bad[:40]}


def _same(text: str, val) -> bool:
    try:
        x = float(text.replace(",", "").rstrip("%"))
    except ValueError:
        return False
    if isinstance(val, bool):
        return False
    try:
        return abs(x - float(val)) <= max(1e-9, 5e-3 * abs(float(val)))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------- X.5 / SP-1.3
PART_1_3 = ("MASTER_SPEC.md", "ARCHITECTURE.md", "INTERFACE_CONTRACTS.md", "WORKSTREAMS.md", "DECISION_LOG.md",
            "PROGRESS.md", "BLOCKERS.md", "FIDELITY_REGISTER.md", "ACCEPTANCE_MANIFEST.json", "REPRODUCE.md")
BINDING = ("MASTER_SPEC_CLARIFICATIONS.md", "RECOVERY_PREAMBLE.md", "ONE_WORLD_CONTRACTS.md", "PROGRESS.json",
           "ACCEPTANCE_LEDGER.jsonl", "DEFECTS.md", "ARTIFACTS.json", "LEARNER_GUIDE.md")
CONTENT = {   # document -> phrases it must contain (required sections)
    "REPRODUCE.md": ["```clean-checkout", "--verify-all"],
    "BLOCKERS.md": ["## Open blockers"],
    "DEFECTS.md": ["```json"],
    "WORKSTREAMS.md": [INTEGRATION_BRANCH],
    "PROGRESS.md": ["PROGRESS.json"],
    "LEARNER_GUIDE.md": ["observation_only"],
    "DECISION_LOG.md": ["D-24"],
}
FINAL = ("FINAL_REPORT.md",)


def documents(root: str | None = None, final: bool = True) -> dict:
    root = root or M.crucible_root()
    d = M.docs_dir(root)
    req = list(PART_1_3) + list(BINDING) + (list(FINAL) if final else [])
    req += [f"components/{ws}.md" for ws in WORKSTREAM_PACKAGES if not glob.glob(os.path.join(d, "components",
                                                                                            f"{ws}*.md"))]
    req += [f"components/{os.path.basename(p)}" for p in sorted(glob.glob(os.path.join(d, "components", "*.md")))]
    req = list(dict.fromkeys(req))
    missing = [r for r in req if not os.path.exists(os.path.join(d, r))]
    lacking = []
    for doc, phrases in CONTENT.items():
        p = os.path.join(d, doc)
        if os.path.exists(p):
            text = open(p, encoding="utf-8").read()
            lacking += [f"{doc}: {ph!r}" for ph in phrases if ph not in text]
    stale = []
    for doc in ("WORKSTREAMS.md", "REPRODUCE.md", "PROGRESS.md", "ARCHITECTURE.md"):
        p = os.path.join(d, doc)
        if os.path.exists(p):
            for k, line in enumerate(open(p, encoding="utf-8"), 1):
                if re.search(r"(merges?|integrat\w*)\s+[^.]*into\s+`?" + re.escape(OLD_BRANCH), line):
                    stale.append(f"{doc}:{k}")
    return {"metrics": {"required_documents_checked": len(req), "missing_documents": len(missing),
                        "documents_missing_required_content": len(lacking),
                        "stale_integration_branch_references": len(stale)},
            "missing": missing, "lacking": lacking, "stale": stale}


# ------------------------------------------------------------------------------------------------- RP.1
CAM_IMPORT = re.compile(r"^\s*(?:from|import)\s+(agentrs\b|superlearner\b|cam\d*\b|cam_unified\b)|"
                        r"import_module\(\s*['\"](agentrs|superlearner)|__import__\(\s*['\"](agentrs|superlearner)")


def cam_scan(root: str | None = None) -> dict:
    """Static scan of every .py file under crucible/ (src, tests, examples, benchmarks)."""
    root = root or M.crucible_root()
    hits, n = [], 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x not in (".git", "__pycache__", "node_modules", ".venv")]
        for f in fn:
            if not f.endswith(".py"):
                continue
            n += 1
            p = os.path.join(dp, f)
            for k, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                if CAM_IMPORT.search(line.split("#", 1)[0]):
                    hits.append(f"{_rel(root, p)}:{k}: {line.strip()[:100]}")
    m = M.load(root)
    gates_cam = [e["id"] for e in m["requirements"] if re.search(r"\bCAM\b", e["requirement"]) and
                 not re.search(r"not CAM|\(not CAM\)|including CAM|independent of CAM|CAM's|without .*CAM|"
                               r"in place of CAM|replaces CAM|CAM integration\)|future CAM|CAM or any", e["requirement"])]
    return {"metrics": {"crucible_files_scanned": n, "cam_imports_in_crucible": len(hits),
                        "gates_depending_on_cam": len(gates_cam)}, "imports": hits, "gates": gates_cam}


# ------------------------------------------------------------------------------------------------- RP.2
def audit_reports(root: str | None = None, adir: str | None = None) -> dict:
    root = root or M.crucible_root()
    adir = adir or audit_dir(root)
    tracked = set(_git(root, "ls-files", "--", _rel(root, adir)).split())
    have, skeptic, committed, missing = 0, 0, 0, []
    for area in AUDIT_AREAS:
        md, js = os.path.join(adir, f"{area}.md"), os.path.join(adir, f"{area}.json")
        if not (os.path.exists(md) or os.path.exists(js)):
            missing.append(area)
            continue
        have += 1
        text = open(md, encoding="utf-8").read() if os.path.exists(md) else ""
        jd = json.load(open(js)) if os.path.exists(js) else {}
        if re.search(r"^#+\s*Skeptic", text, re.M | re.I) or jd.get("skeptic"):
            skeptic += 1
        if all(_rel(root, p) in tracked for p in (md, js) if os.path.exists(p)):
            committed += 1
    return {"metrics": {"audit_reports": have, "audit_reports_with_skeptic_review": skeptic,
                        "audited_areas_missing": len(missing), "audit_reports_committed": committed},
            "missing": missing}


# ------------------------------------------------------------------------------------------------- RP.3 / RP.5
def ls_remote(root: str, ref: str) -> str | None:
    p = subprocess.run(["git", "ls-remote", "origin", ref], cwd=root, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return None
    line = p.stdout.split()
    return line[0] if line else ""


def remote_state(root: str | None = None, milestones: list[dict] | None = None) -> dict:
    root = root or M.crucible_root()
    remote_head = ls_remote(root, f"refs/heads/{INTEGRATION_BRANCH}")
    local_head = _git(root, "rev-parse", INTEGRATION_BRANCH).strip()
    head = _git(root, "rev-parse", "HEAD").strip()
    unpushed = []
    for ms in milestones or []:
        c = ms.get("commit")
        if not c:
            continue
        if remote_head and _git(root, "cat-file", "-t", remote_head).strip() == "commit":
            ok = subprocess.run(["git", "merge-base", "--is-ancestor", c, remote_head], cwd=root).returncode == 0
        else:
            ok = False
        if not ok:
            unpushed.append(c[:9])
    anchor = _git(root, "rev-parse", OLD_ANCHOR).strip()
    old_local = _git(root, "rev-parse", OLD_BRANCH).strip()
    old_remote = ls_remote(root, f"refs/heads/{OLD_BRANCH}")
    # worktrees whose branch is merged into the integration branch but still have uncommitted changes
    dirty_merged, dirty_any = [], []
    first_parent = set(_git(root, "rev-list", "--first-parent", INTEGRATION_BRANCH).split())
    cur = {}
    for line in _git(root, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line.strip():
            if cur.get("worktree") and cur.get("branch") and os.path.isdir(cur["worktree"]):
                st = _git(cur["worktree"], "status", "--porcelain").strip()
                if st:
                    br = cur["branch"].replace("refs/heads/", "")
                    dirty_any.append(br)
                    tip = _git(root, "rev-parse", cur["branch"]).strip()
                    # merged = its own commits were merged (a branch still at an integration commit has merged nothing)
                    merged = tip not in first_parent and subprocess.run(
                        ["git", "merge-base", "--is-ancestor", tip, INTEGRATION_BRANCH], cwd=root).returncode == 0
                    if merged and br != INTEGRATION_BRANCH:
                        dirty_merged.append(br)
            cur = {}
            continue
        k, _, v = line.partition(" ")
        cur[k] = v
    return {"metrics": {"integration_head_pushed": bool(remote_head) and remote_head == local_head,
                        "unpushed_milestone_commits": len(unpushed),
                        "merged_worktrees_with_uncommitted_changes": len(dirty_merged),
                        "old_branch_tip_matches_anchor": bool(anchor) and old_local == anchor,
                        "old_branch_remote_matches_anchor": bool(anchor) and old_remote == anchor},
            "remote_head": remote_head, "local_head": local_head, "head": head, "unpushed": unpushed,
            "dirty_merged_worktrees": dirty_merged, "dirty_worktrees": dirty_any,
            "old_branch": {"local": old_local, "remote": old_remote, "anchor": anchor}}


# ------------------------------------------------------------------------------------------------- RP.6
SECRET = re.compile(r"(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|"
                    r"sk-(?:ant-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
                    r"xox[baprs]-[A-Za-z0-9-]{20,})")


def secrets_scan(root: str | None = None) -> dict:
    """Credential patterns in the working tree and in every commit reachable from any ref (git grep over history
    would be slow; `git log -p --all` is streamed once). Never prints a match, only where it is."""
    root = root or M.crucible_root()
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    commits = int(_git(top, "rev-list", "--all", "--count").strip() or 0)
    found = []
    p = subprocess.Popen(["git", "log", "--all", "-p", "--no-color", "--format=@@commit %H"], cwd=top,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cur = ""
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace")
        if line.startswith("@@commit "):
            cur = line.split()[1][:9]
        elif line.startswith("+") and SECRET.search(line):
            found.append(f"commit {cur}")
    p.wait()
    for dp, dn, fn in os.walk(top):
        dn[:] = [x for x in dn if x not in (".git", "__pycache__", "node_modules", ".venv")]
        for f in fn:
            fp = os.path.join(dp, f)
            try:
                if os.path.getsize(fp) > 5_000_000:
                    continue
                if SECRET.search(open(fp, encoding="utf-8", errors="ignore").read()):
                    found.append(f"file {os.path.relpath(fp, top)}")
            except OSError:
                continue
    urls = [u for u in _git(top, "remote", "-v").split() if "://" in u and "@" in u.split("://", 1)[1].split("/", 1)[0]]
    return {"metrics": {"commits_scanned": commits, "secrets_found": len(set(found)),
                        "remote_urls_with_credentials": len(set(urls))}, "where": sorted(set(found))[:20]}


def remote_forward_only(root: str | None = None) -> dict:
    """The pushed integration branch only moved forward: the reflog of its remote-tracking ref never records a
    forced update (and the remote head descends from every recorded value)."""
    root = root or M.crucible_root()
    ref = f"refs/remotes/origin/{INTEGRATION_BRANCH}"
    vals = _git(root, "reflog", "show", "--format=%H %gs", ref).splitlines()
    forced = [v for v in vals if "forced-update" in v]
    return {"metrics": {"non_fast_forward_remote_updates": len(forced), "remote_ref_updates_checked": len(vals)},
            "forced": forced[:5]}


# ------------------------------------------------------------------------------------------------- RP.4
ARTIFACTS_JSON = "docs/v5/ARTIFACTS.json"
DURABILITY = ("github", "public_source", "local_only")


def artifacts(root: str | None = None) -> dict:
    root = root or M.crucible_root()
    try:
        reg = json.load(open(os.path.join(root, ARTIFACTS_JSON)))
    except (OSError, ValueError):
        return {"metrics": {"required_artifacts_listed": 0, "required_artifacts_unpreserved": 1,
                            "artifact_hash_mismatches": 0, "artifacts_without_location": 1}, "problems": ["no list"]}
    tracked = set(_git(root, "ls-files").split())
    remote = ls_remote(root, f"refs/heads/{INTEGRATION_BRANCH}") or ""
    pushed_tree = set()
    if remote and _git(root, "cat-file", "-t", remote).strip() == "commit":
        pushed_tree = set(_git(root, "ls-tree", "-r", "--name-only", remote).split())
    unpreserved, mism, noloc, problems = [], [], [], []
    for a in reg["artifacts"]:
        if not a.get("location") or a.get("durability") not in DURABILITY:
            noloc.append(a.get("name"))
            continue
        p = a["path"] if os.path.isabs(a["path"]) else os.path.join(root, a["path"])
        if a.get("living") and not os.path.exists(p):
            problems.append(f"{a['name']}: missing at {a['path']}")
        elif os.path.isfile(p) and a.get("sha256") and not a.get("living"):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.getsize(p) < 2_000_000_000 else None
            if h and h != a["sha256"]:
                mism.append(a["name"])
        elif not os.path.exists(p) and a.get("durability") != "public_source":
            problems.append(f"{a['name']}: missing at {a['path']}")
        if a["durability"] == "local_only":
            unpreserved.append(a["name"])
        elif a["durability"] == "github":
            rel = a["path"]
            if rel not in tracked or (pushed_tree and rel not in pushed_tree) or not pushed_tree:
                unpreserved.append(a["name"])
                problems.append(f"{a['name']}: not in the pushed integration branch")
    return {"metrics": {"required_artifacts_listed": len(reg["artifacts"]),
                        "required_artifacts_unpreserved": len(unpreserved), "artifact_hash_mismatches": len(mism),
                        "artifacts_without_location": len(noloc)},
            "unpreserved": unpreserved, "mismatched": mism, "problems": problems[:20]}


# ------------------------------------------------------------------------------------------------- OO.13 / OO.15
LEARNER_REWARD = re.compile(r"\b(def|class)\s+\w*(intrinsic_reward|learned_reward|reward_model|curiosity|"
                            r"novelty_bonus|empowerment|cam_reward|internal_reward)\w*", re.I)


def learner_reward_scan(root: str | None = None) -> dict:
    """OO.13 (D-81): definitions of a learner's internal reward function (intrinsic/learned reward, curiosity, novelty
    bonus, empowerment, CAM reward) in Crucible itself - the environment and its library, `crucible/src` - and every
    `src` module that imports a file holding one. Every .py file under crucible/ is scanned; definitions in learner-side
    client code outside `src` (the external-learner example of OO.6 under examples/, test learners under tests/, written
    only against the public interface) are reported separately: a learner that computes its own reward from
    observations is what Clarification 2 asks for, as long as Crucible never contains or uses it. Environment-side
    legacy rewards are not learner rewards."""
    root = root or M.crucible_root()
    hits, n, files_with = [], 0, set()
    for dp, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x not in (".git", "__pycache__", "node_modules", ".venv")]
        for f in fn:
            if f.endswith(".py"):
                n += 1
                rel = _rel(root, os.path.join(dp, f))
                for k, line in enumerate(open(os.path.join(dp, f), encoding="utf-8", errors="replace"), 1):
                    if LEARNER_REWARD.search(line):
                        hits.append(f"{rel}:{k}: {line.strip()[:90]}")
                        files_with.add(rel)
    inside = [h for h in hits if h.startswith("src/")]
    client = [h for h in hits if not h.startswith("src/")]
    mods = {os.path.splitext(f)[0].replace(os.sep, ".").split("src.", 1)[-1] for f in files_with}
    stems = {os.path.splitext(os.path.basename(f))[0] for f in files_with if not f.startswith("src/")}
    uses = []                                  # src modules importing learner-side code that holds a reward function
    for dp, dn, fn in os.walk(os.path.join(root, "src")):
        dn[:] = [x for x in dn if x != "__pycache__"]
        for f in fn:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dp, f), encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for x in ast.walk(tree):
                names = [a.name for a in x.names] if isinstance(x, ast.Import) else \
                    [x.module or ""] if isinstance(x, ast.ImportFrom) else []
                if any(nm in mods or nm.split(".")[-1] in stems or nm.split(".")[0] in stems for nm in names):
                    uses.append(f"{_rel(root, os.path.join(dp, f))}:{x.lineno}")
    return {"metrics": {"crucible_files_scanned": n, "learner_reward_functions_in_crucible": len(inside) + len(uses),
                        "external_learner_reward_functions": len(client)},
            "hits": inside + uses, "client_hits": client}


def oo_independence(root: str | None = None) -> dict:
    """OO.15: OO.1-OO.8 whose committed evidence and review name different agents and sessions (and are VERIFIED)."""
    from .checker import verify_all
    root = root or M.crucible_root()
    res = verify_all(root)
    ok = []
    for r in res["rows"]:
        if r["id"] not in {f"OO.{i}" for i in range(1, 9)} or r["status"] != "VERIFIED":
            continue
        ev = json.load(open(os.path.join(root, r["evidence"])))
        rv = json.load(open(os.path.join(root, r["review"])))
        a, b = ev.get("identity") or {}, rv.get("reviewer") or {}
        if a.get("agent") and b.get("agent") and a["agent"] != b["agent"] and \
                (not a.get("session_sha256") or a.get("session_sha256") != b.get("session_sha256")):
            ok.append(r["id"])
    return {"metrics": {"oo_requirements_independently_verified": len(ok)}, "verified": ok}
