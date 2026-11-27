"""Gate A machinery: measurements of Crucible 4.0 baseline preservation on the real tree.

* A.1 `regression`: run the 4.0 test files of the recorded baseline (docs/v5/evidence/gate_A/regression_baseline.json
  and its JUnit files, 1,076 node ids at 777f28c) byte-identical to c07098b (restored in a fresh clone of HEAD) as the
  4.0 historical baseline (D-82) and require every baseline node id to pass by exact id.
* A.2 `definitions`: every 4.0 world family (74), mechanism family (25) and useful-work category family (10) is still
  registered with the same split (and engine/transfer/kind/relation), and generates byte-identical content on two seeds
  (hashes computed by `_v4probe.py` against an extracted c07098b tree and against this tree).
* A.3 `execution`: every 4.0 world family builds and advances on two seeds.
* A.4 `sealed`: the sealed store (~/.crucible or CRUCIBLE_SEALED_DIR) is unchanged against the committed baseline
  (docs/v5/baselines/sealed_store.json; registry and access log by SHA-256, salt and task files by size and mtime only:
  their contents are never read), its access log has no entry after v5 started, no v5 module names it, and a v5
  workload run under an audit hook opens nothing inside it.
* A.5 `prior_results`: the 4.0 acceptance records A-P and the deterministic demo records are regenerated in a fresh
  clone and compared with the committed records; and the 4.0 world trajectories (state digest and event count after
  the run) equal those of the c07098b tree.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from . import manifest as M

V4_COMMIT = "c07098b8726e2025487f2ef6e09f9863851ac416"
V5_START = _dt.datetime(2026, 11, 25, tzinfo=_dt.timezone.utc).timestamp()
BASELINE_RUN = "docs/v5/evidence/gate_A/regression_baseline.json"
SEALED_BASELINE = "docs/v5/baselines/sealed_store.json"
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v4probe.py")
TIMING_KEY = re.compile(r"(_s|_ms|_sec|_seconds)$|wall|elapsed|throughput|per_s(ec)?\b|per_second|recorded_at|^generated$")
SEALED_NAMES = ("sealed_salt", "sealed_access.jsonl", "sealed_sets.json", "sealed_tasks", "CRUCIBLE_SEALED_DIR",
                "research.sealed", "research import sealed", "open_set(")
# a name counts only as a whole token: `v5_sealed_salt` (the 5.0 evaluator's own sealed store, research/v5) or
# `CRUCIBLE_V5_SEALED_DIR` are other names, not the 4.0 store's
_SEALED_RE = re.compile("|".join(("(?<![A-Za-z0-9_])" if n[0].isalnum() or n[0] == "_" else "") + re.escape(n) +
                                 ("(?![A-Za-z0-9_])" if n[-1].isalnum() or n[-1] == "_" else "")
                                 for n in SEALED_NAMES))
_SEALED_IMPORT_RE = re.compile(r"from\s+\S*research\S*\s+import\s+.*\bsealed\b")


def sealed_reference(code: str) -> bool:
    """Whether a line of code (comments stripped) names the 4.0 sealed store or imports its module."""
    return bool(_SEALED_RE.search(code) or _SEALED_IMPORT_RE.search(code))


def _git(root: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout


def scratch_dir() -> str:
    d = os.environ.get("CRUCIBLE_V5_SCRATCH") or os.path.join(tempfile.gettempdir(), "crucible-v5-gateA")
    os.makedirs(d, exist_ok=True)
    return d


def v4_tree(root: str | None = None, dest: str | None = None) -> str:
    """The `crucible/` directory of c07098b, extracted with git archive (cached)."""
    root = root or M.crucible_root()
    dest = dest or os.path.join(scratch_dir(), f"v4full-{V4_COMMIT[:9]}")
    out = os.path.join(dest, "crucible")
    if os.path.exists(os.path.join(out, "pyproject.toml")):
        return out
    os.makedirs(dest, exist_ok=True)
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    arch = subprocess.run(["git", "archive", V4_COMMIT, "crucible/src", "crucible/tests", "crucible/examples",
                           "crucible/docs/experiments", "crucible/pyproject.toml"], cwd=top, capture_output=True,
                          check=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=arch, check=True)
    return out


def probe(src_root: str, *args: str, timeout: int = 3600, historical: bool = False) -> dict:
    """Run _v4probe.py against the tree at `src_root`; `historical`: as the Crucible 4.0 historical baseline (D-82)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CRUCIBLE_HISTORICAL_BASELINE")}
    env.update(PYTHONPATH=os.path.join(src_root, "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", **(HISTORICAL_ENV if historical else {}))
    p = subprocess.run([sys.executable, "-P", PROBE, *args], cwd=src_root, env=env, capture_output=True, text=True,
                       timeout=timeout)
    if "@@JSON@@" not in p.stdout:
        raise RuntimeError(f"probe {args} failed in {src_root}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.rsplit("@@JSON@@", 1)[1])


# ------------------------------------------------------------------------------------------------- A.1
def baseline_nodes(root: str | None = None) -> list[tuple[str, str]]:
    """(classname, name) of every test case of the recorded 4.0 baseline run."""
    root = root or M.crucible_root()
    b = json.load(open(os.path.join(root, BASELINE_RUN)))
    out = []
    for j in b["junit"]:
        for tc in ET.parse(os.path.join(root, j)).getroot().iter("testcase"):
            out.append((tc.get("classname") or "", tc.get("name") or ""))
    return out


def baseline_files(nodes) -> list[str]:
    return sorted({cl.split(".")[0] + "/" + cl.split(".")[1] + ".py" for cl, _ in nodes})


def v4_test_files_changed(root: str | None = None) -> list[str]:
    """4.0 test files (present at c07098b) that were modified or deleted since."""
    root = root or M.crucible_root()
    out = _git(root, "diff", "--name-only", "--relative", "--diff-filter=MDR", V4_COMMIT, "HEAD", "--", "tests").split()
    return sorted(out)


def _groups(root: str, files: list[str], n: int) -> list[list[str]]:
    sized = sorted(files, key=lambda f: -os.path.getsize(os.path.join(root, f)))
    groups = [[] for _ in range(max(1, n))]
    load = [0] * len(groups)
    for f in sized:
        i = load.index(min(load))
        groups[i].append(f)
        load[i] += os.path.getsize(os.path.join(root, f))
    return [g for g in groups if g]


# Resource classes of the whole-suite run through the slot scheduler (Gate X.2/X.4): (filename pattern, cpu, gpu MB,
# RAM MB, timeout s). The first matching pattern wins. Population/social/civ/semantic code runs the Qwen3-0.6B text
# path and needs the GPU; the gate tests carry their own heavy measurements. Overridable as JSON in
# $CRUCIBLE_GATE_X_RESOURCES (same row shape).
SUITE_RESOURCES = [
    (r"tests/v5/test_gate_(P|SP_3_3b|SP_13_6|SP_13_7b?|V|I)\.py$", 2, 9000, 28000, 30 * 3600),
    (r"tests/v5/test_gate_(H|SP_3_2b|SP_3_2d|SP_3_3c|SP_11_6c)\.py$", 1, 9000, 16000, 8 * 3600),
    (r"tests/v5/", 1, 3000, 12000, 6 * 3600),
    (r"tests/test_v5_(population|social|civ|semantic|econ|innovation|commerce|domains)", 1, 3000, 12000, 4 * 3600),
    (r"tests/test_agents_llm\.py$", 1, 6000, 8000, 3600),
    (r"", 1, 0, 6000, 4 * 3600),
]


def suite_resources(f: str) -> tuple[int, int, int, int]:
    rows = SUITE_RESOURCES
    if os.environ.get("CRUCIBLE_GATE_X_RESOURCES"):
        rows = [tuple(r) for r in json.loads(os.environ["CRUCIBLE_GATE_X_RESOURCES"])] + list(SUITE_RESOURCES)
    for pat, cpu, gpu, ram, tmo in rows:
        if re.search(pat, f):
            return int(cpu), int(gpu), int(ram), int(tmo)
    return 1, 0, 6000, 4 * 3600


def slot_prefix(label: str, cpu: int, gpu: int, ram: int, timeout: int) -> list[str]:
    """The slot-scheduler command prefix for one suite group (crucible.scale.slots of the checkout being run)."""
    return [sys.executable, "-m", "crucible.scale.slots", "--cpu", str(cpu), "--gpu-mb", str(gpu), "--ram-mb", str(ram),
            "--timeout", str(timeout), "--label", label, "--"]


def run_pytest_groups(root: str, files: list[str], out_dir: str, workers: int = 1, timeout: int = 4 * 3600,
                      extra_env: dict | None = None, isolate: tuple = (), plugins: tuple = (),
                      extra_path: tuple = (), via_slots: str | None = None) -> dict:
    """Run test files in `workers` concurrent pytest processes (one thread each); returns cases and exit codes.
    Files in `isolate` run in a pytest process of their own each (after the groups); `plugins` are loaded with -p in
    every process; `extra_path` entries follow the checkout's src on PYTHONPATH."""
    os.makedirs(out_dir, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_") and k not in (
        "PYTHONPATH", "CRUCIBLE_V5_RUN_ID", "CRUCIBLE_V5_EVIDENCE_OUT", "CRUCIBLE_HISTORICAL_BASELINE")}
    env.update(PYTHONPATH=os.pathsep.join([os.path.join(root, "src"), *extra_path]), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    plug = [x for p in plugins for x in ("-p", p)]
    if via_slots:
        # every resource class is a group of its own, submitted as one slot job per `workers` share: the scheduler
        # admits them as the machine's budget allows (GPU tests with a GPU, the rest without)
        by_class: dict[tuple, list[str]] = {}
        for f in files:
            if f not in isolate:
                by_class.setdefault(suite_resources(f), []).append(f)
        batches = [(g, True, c) for c, fs in sorted(by_class.items()) for g in _groups(root, fs, workers)]
    else:
        batches = [(g, True, None) for g in _groups(root, [f for f in files if f not in isolate], workers)]
    batches += [([f], False, suite_resources(f) if via_slots else None) for f in files if f in isolate]
    procs, pending = [], list(enumerate(batches))
    rcs, cases = [], []

    def start(i, g, res=None):
        junit = os.path.join(out_dir, f"group{i}.xml")
        log = open(os.path.join(out_dir, f"group{i}.log"), "w")
        pre = slot_prefix(f"{via_slots}: suite group {i} ({len(g)} files)", *res) if res else []
        return junit, log, subprocess.Popen(
            [*pre, sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *plug,
             f"--junitxml={junit}", *g], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    for i, (g, concurrent, res) in pending:
        if concurrent:
            procs.append(start(i, g, res))
    serial = [(i, g, res) for i, (g, concurrent, res) in pending if not concurrent]
    finished = []
    for junit, log, p in procs:
        try:
            rcs.append(p.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append("timeout")
        log.close()
        finished.append(junit)
    for i, g, res in serial:                    # isolated files: one process each, after the groups
        junit, log, p = start(i, g, res)
        try:
            rcs.append(p.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append("timeout")
        log.close()
        finished.append(junit)
    for junit in finished:
        if os.path.exists(junit):
            for tc in ET.parse(junit).getroot().iter("testcase"):
                outcome = "passed"
                for kind in ("failure", "error", "skipped"):
                    if tc.find(kind) is not None:
                        outcome = kind
                cases.append({"classname": tc.get("classname") or "", "name": tc.get("name") or "",
                              "outcome": outcome})
    return {"rcs": rcs, "cases": cases, "out_dir": out_dir}


V4_ISOLATED = ("tests/test_realworld_gate.py",)   # D-82: the v5 world-hosting latch (Part 16) forbids the real-world
#                                                  bridge in a process that hosted worlds; 4.0 ran it after them
V4_ALIAS_PLUGIN = "agentrs.cam2.envs.crucible_v4_names"   # D-81/D-82: 4.0 names of the relocated CAM adapters
HISTORICAL_ENV = {"CRUCIBLE_HISTORICAL_BASELINE": "4.0"}  # D-82: the 4.0 reward contract, selected explicitly (OO.8)


def v4_test_clone(root: str, dest: str) -> tuple[str, list[str], list[str]]:
    """A fresh clone of HEAD whose `tests/` files that exist at c07098b are restored to their c07098b bytes (the tree's
    copies of some carry v5 additions; Gate A.1 runs 4.0's own). Returns (the clone's crucible root, the restored
    files, the 4.0 test files whose bytes in the clone still differ from c07098b - empty unless the restore failed)."""
    croot = _plain_clone(root, dest)
    top = _git(croot, "rev-parse", "--show-toplevel").strip()
    prefix = os.path.relpath(os.path.realpath(croot), os.path.realpath(top))
    want = {}
    for ln in _git(top, "ls-tree", "-r", V4_COMMIT, "--", f"{prefix}/tests").splitlines():
        meta, path = ln.split("\t", 1)
        want[path] = meta.split()[2]

    def differing() -> list[str]:
        return sorted(p for p, b in want.items()
                      if not os.path.exists(os.path.join(top, p)) or _blob(os.path.join(top, p)) != b)
    restored = differing()
    if restored:
        subprocess.run(["git", "checkout", V4_COMMIT, "--", *restored], cwd=top, check=True)
    return croot, [os.path.relpath(p, prefix) for p in restored], [os.path.relpath(p, prefix) for p in differing()]


def _blob(path: str) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def regression(root: str | None = None, workers: int | None = None, out_dir: str | None = None) -> dict:
    """A.1 (D-82): the Crucible 4.0 regression suite as 4.0 wrote it - every 4.0 test file byte-identical to c07098b,
    restored in a fresh clone of HEAD - runs against the current implementation in processes started as the 4.0
    historical baseline (`CRUCIBLE_HISTORICAL_BASELINE=4.0`: the 4.0 reward contract for every constructor that names
    no mode); the 4.0 names of the CAM adapters moved to the AGENTRS side (D-81) resolve to the relocated modules;
    tests/test_realworld_gate.py runs in a process of its own (the v5 hosting latch). Every one of the 1,076 baseline
    node ids must pass by exact id. The tree's own copies of the 11 4.0 test files that carry v5 additions are v5 tests
    (X.2/X.4); `v4_test_files_modified` counts the 4.0 test files of this run that differ from c07098b, and
    `v4_test_files_modified_in_tree` reports the tree's."""
    root = root or M.crucible_root()
    workers = workers or int(os.environ.get("CRUCIBLE_GATE_A_WORKERS", "1"))
    nodes = baseline_nodes(root)
    files = baseline_files(nodes)
    work = out_dir or tempfile.mkdtemp(prefix="gateA1-", dir=scratch_dir())
    croot, restored, differ = v4_test_clone(root, os.path.join(work, "clone"))
    agentrs_src = os.path.join(_git(croot, "rev-parse", "--show-toplevel").strip(), "src")
    missing_files = [f for f in files if not os.path.exists(os.path.join(croot, f))]
    res = run_pytest_groups(croot, [f for f in files if f not in missing_files], os.path.join(work, "junit"), workers,
                            extra_env=HISTORICAL_ENV, isolate=V4_ISOLATED, plugins=(V4_ALIAS_PLUGIN,),
                            extra_path=(agentrs_src,))
    by = {(c["classname"], c["name"]): c["outcome"] for c in res["cases"]}
    not_run = [n for n in nodes if n not in by]
    not_passed = [n for n in nodes if n in by and by[n] != "passed"]
    outcomes = [c["outcome"] for c in res["cases"]]
    relocated = [n for n in nodes if n[0] == "tests.test_agents_llm" and n[1].startswith(("test_cam2", "test_sl_"))]
    return {"metrics": {
        "tests_total": len(res["cases"]), "tests_failed": outcomes.count("failure"),
        "tests_errors": outcomes.count("error"), "tests_skipped": outcomes.count("skipped"),
        "baseline_node_ids_checked": len(nodes), "baseline_node_ids_missing": len(not_run),
        "baseline_node_ids_not_passed": len(not_passed), "v4_test_files_modified": len(differ),
        "v4_test_files_run": len(files) - len(missing_files),
        "v4_test_files_modified_in_tree": len(restored), "v4_relocated_adapter_node_ids": len(relocated),
        "pytest_groups": len(res["rcs"]), "gpu_visible": _gpu_visible()},
        "not_run": not_run[:50], "not_passed": not_passed[:50], "changed_test_files": restored,
        "differ_after_restore": differ, "rcs": res["rcs"], "work": work}


def _gpu_visible() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    try:
        return bool(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------- A.2 / A.3
def definitions(root: str | None = None, seeds=(0, 1)) -> dict:
    root = root or M.crucible_root()
    s = [str(x) for x in seeds]
    old = probe(v4_tree(root), "definitions", "--seeds", *s)
    new = probe(root, "definitions", "--seeds", *s)
    changed, missing, checked = [], [], 0
    per_kind = {}
    for kind in ("world", "mech", "useful"):
        per_kind[kind] = len(old[kind])
        for name, d in old[kind].items():
            checked += 1
            if name not in new[kind]:
                missing.append(f"{kind}:{name}")
            elif new[kind][name] != d:
                changed.append(f"{kind}:{name}")
    errors = [f"{k}:{n}" for k in old for n, d in old[k].items() if any(str(v).startswith("error") for v in
                                                                         d["digests"].values())]
    return {"metrics": {"v4_definitions_checked": checked, "v4_definitions_changed": len(changed) + len(missing),
                        "v4_definitions_missing": len(missing), "v4_world_families_checked": per_kind["world"],
                        "v4_mechanism_families_checked": per_kind["mech"],
                        "v4_useful_work_families_checked": per_kind["useful"],
                        "v4_seeds_per_definition": len(seeds), "v4_definition_generation_errors": len(errors)},
            "changed": changed, "missing": missing, "errors": errors,
            "v5_added_world_families": sorted(set(new["world"]) - set(old["world"]))}


def execution(root: str | None = None, seeds=(0, 1), days: float = 2.0) -> dict:
    """A.3: every 4.0 world family builds and advances (default program-cost model)."""
    root = root or M.crucible_root()
    s = [str(x) for x in seeds]
    v4_names = sorted(probe(v4_tree(root), "definitions", "--seeds", "0")["world"])
    new = probe(root, "execute", "--seeds", *s, "--days", str(days))
    failed = [f"{n}/{sd}" for n in v4_names for sd, r in new.get(n, {}).items() if not r["ok"]]
    absent = [n for n in v4_names if n not in new]
    ran = [n for n in v4_names if n in new and all(r["ok"] for r in new[n].values())]
    return {"metrics": {"v4_families_run": len(ran), "v4_families_failed": len(set(f.split("/")[0] for f in failed))
                        + len(absent), "v4_families_total": len(v4_names), "v4_seeds_per_family": len(seeds)},
            "failed": failed, "absent": absent}


def trajectories(root: str | None = None, seeds=(0, 1), days: float = 2.0) -> dict:
    """A.5: in 4.0's deterministic mode, every 4.0 world family's trajectory (event count and final state digest
    after `days`) equals the one the c07098b tree produces. The c07098b tree is run twice to confirm it is itself
    deterministic; a family that is not is reported and counted as unreproducible."""
    root = root or M.crucible_root()
    s = [str(x) for x in seeds]
    args = ("execute", "--seeds", *s, "--days", str(days), "--deterministic")
    old1, old2 = probe(v4_tree(root), *args), probe(v4_tree(root), *args)
    new = probe(root, *args, historical=True)         # D-82: 4.0 as an explicitly selected historical baseline
    key = lambda r: (r.get("ok"), r.get("events"), r.get("state_digest"))
    nondet, mism, compared = [], [], 0
    for n in sorted(old1):
        for sd, r in old1[n].items():
            if key(r) != key(old2[n][sd]):
                nondet.append(f"{n}/{sd}")
                continue
            compared += 1
            q = (new.get(n) or {}).get(sd) or {}
            if key(q) != key(r):
                mism.append(f"{n}/{sd}")
    return {"metrics": {"v4_trajectories_compared": compared, "v4_trajectory_mismatches": len(mism),
                        "v4_nondeterministic_trajectories": len(nondet)},
            "mismatches": mism, "nondeterministic": nondet}


# ------------------------------------------------------------------------------------------------- A.4
def sealed_dir() -> str:
    return os.environ.get("CRUCIBLE_SEALED_DIR") or os.path.join(os.path.expanduser("~"), ".crucible")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def sealed_snapshot(d: str | None = None) -> dict:
    """Metadata of the sealed store. Registry and access log are hashed (they hold no task seeds); the salt and the
    task files are described by size and mtime only - their contents are never read."""
    d = d or sealed_dir()
    out = {}
    if not os.path.isdir(d):
        return out
    for dp, dn, fn in os.walk(d):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, d)
            st = os.stat(p)
            rec = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if rel in ("sealed_sets.json", "sealed_access.jsonl"):
                rec["sha256"] = _sha(p)
            out[rel] = rec
    return out


def capture_sealed_baseline(root: str | None = None) -> dict:
    root = root or M.crucible_root()
    snap = sealed_snapshot()
    rec = {"what": "metadata of the Crucible 4.0 sealed store, captured by workstream P; registry and access log by "
                   "SHA-256, salt and task files by size and mtime (never read)",
           "dir": "~/.crucible (or CRUCIBLE_SEALED_DIR)", "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "all_mtimes_before_v5_start": all(r["mtime_ns"] / 1e9 < V5_START for r in snap.values()), "files": snap}
    p = os.path.join(root, SEALED_BASELINE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec


def v5_source_files(root: str) -> list[str]:
    """Python files added to src/crucible since Crucible 4.0 (v5 code), excluding this Gate A machinery."""
    added = _git(root, "diff", "--name-only", "--relative", "--diff-filter=A", V4_COMMIT, "HEAD", "--", "src").split()
    here = {"src/crucible/acceptance/v5/baseline.py", "src/crucible/acceptance/v5/_v4probe.py",
            "src/crucible/acceptance/v5/requirements.py"}
    return sorted(f for f in added if f.endswith(".py") and f not in here and os.path.exists(os.path.join(root, f)))


AUDIT_SCRIPT = r'''
import os, sys, json
SEALED = os.path.realpath(sys.argv[1])
hits = []
def hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.realpath(os.fsdecode(args[0]))
        if p == SEALED or p.startswith(SEALED + os.sep):
            hits.append(p)
    elif ev in ("os.listdir", "os.scandir") and args and args[0] is not None:
        p = os.path.realpath(os.fsdecode(args[0]))
        if p == SEALED or p.startswith(SEALED + os.sep):
            hits.append(p)
sys.addaudithook(hook)
import importlib, pkgutil
import crucible
crucible.load_all_modules()
loaded = 0
for pkg in crucible.V5_PACKAGES:
    try:
        importlib.import_module("crucible." + pkg)
        loaded += 1
    except Exception as exc:
        print("import failed", pkg, exc, file=sys.stderr)
steps = 0
try:
    from crucible import experience
    tier, name = sys.argv[2].split(":", 1)
    env = experience.make(int(tier), name, 0, reward_mode="observation_only")
    for _ in range(int(sys.argv[3])):
        for agent in env.agents():
            env.observe(agent)
            env.tools(agent)
        env.advance(3600)
        steps += 1
except Exception as exc:
    print("workload failed", type(exc).__name__, exc, file=sys.stderr)
print("@@JSON@@" + json.dumps({"hits": sorted(set(hits)), "packages_loaded": loaded, "steps": steps}))
'''


def _denylist_strings(src: str) -> dict[int, list[str]]:
    """line -> dotted `crucible.` module paths that are elements of a literal collection of at least two such paths (a
    denylist of modules that must not be reachable, e.g. the red team's isolation probe): naming a module there is
    not a use of it. Imports, import_module/__import__ calls and the store's file and variable names still count."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out: dict[int, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
            mods = [e for e in n.elts if isinstance(e, ast.Constant) and isinstance(e.value, str) and
                    re.fullmatch(r"crucible(\.\w+)+", e.value)]
            if len(mods) >= 2 and len(mods) == len(n.elts):
                for e in mods:
                    out.setdefault(e.lineno, []).append(e.value)
    return out


def sealed(root: str | None = None, workload: str = "1:mech:scm_additive", steps: int = 48) -> dict:
    root = root or M.crucible_root()
    d = sealed_dir()
    base = json.load(open(os.path.join(root, SEALED_BASELINE)))
    snap = sealed_snapshot(d)
    changed = sorted(k for k in set(base["files"]) | set(snap) if base["files"].get(k) != snap.get(k))
    after = []
    log = os.path.join(d, "sealed_access.jsonl")
    if os.path.exists(log):
        for line in open(log):
            try:
                t = float(json.loads(line).get("t", 0))
            except ValueError:
                continue
            if t >= V5_START:
                after.append(t)
    reg = {}
    try:
        reg = json.load(open(os.path.join(d, "sealed_sets.json")))
    except (OSError, ValueError):
        pass
    over = [k for k, v in reg.items() if v.get("contaminated") or v.get("accesses", 0) > v.get("budget", 0)]
    static, denylisted = [], []
    for f in v5_source_files(root):
        txt = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        deny = _denylist_strings(txt)
        for k, line in enumerate(txt.splitlines(), 1):
            code = line.split("#", 1)[0]
            hit = sealed_reference(code)
            if hit and k in deny:                    # a module-path denylist entry names the module without using it
                rest = code
                for d in deny[k]:
                    rest = rest.replace(repr(d), "").replace(f'"{d}"', "")
                if not sealed_reference(rest):
                    denylisted.append(f"{f}:{k}")
                    continue
            if hit:
                static.append(f"{f}:{k}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.path.join(root, "src"), OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", AUDIT_SCRIPT, d, workload, str(steps)], cwd=root, env=env,
                       capture_output=True, text=True, timeout=1800)
    dyn = json.loads(p.stdout.rsplit("@@JSON@@", 1)[1]) if "@@JSON@@" in p.stdout else {"hits": ["<probe failed>"],
                                                                                         "packages_loaded": 0, "steps": 0}
    return {"metrics": {"sealed_sets_changed": len(changed), "sealed_files_checked": len(snap),
                        "sealed_access_entries_after_v5_start": len(after),
                        "sealed_sets_contaminated_or_over_budget": len(over),
                        "v5_files_scanned": len(v5_source_files(root)), "v5_static_references_to_sealed_store": len(static),
                        "v5_runtime_sealed_opens": len(dyn["hits"]), "v5_workload_steps": dyn["steps"],
                        "v5_denylist_mentions_of_sealed_module": len(denylisted),
                        "v5_packages_loaded_under_audit": dyn["packages_loaded"],
                        "sealed_leak_findings": len(static) + len(dyn["hits"]) + len(after) + len(over)},
            "changed": changed, "static": static[:20], "denylisted": denylisted[:20], "runtime_hits": dyn["hits"][:20],
            "probe_stderr": p.stderr[-1500:]}


# ------------------------------------------------------------------------------------------------- A.5
ACCEPTANCE_RECORDS = "docs/experiments/acceptance"
DEMOS = {   # deterministic 4.0 demo records with their reproduction command (from each script's docstring);
    # demo_social_org writes its record in place (inside the fresh clone), demo_lifetime takes --out/--store
    "docs/experiments/demo_social_org.json": ["examples/demo_social_org.py", "30"],
    "docs/experiments/demo_lifetime.json": ["examples/demo_lifetime.py", "--days", "30"],
}
PROVENANCE_KEYS = ("recorded_at", "failure", "logs", "commit")   # where/when a record was made, not what it measured
DURATION_TEXT = re.compile(r"\s+in\s+\d+(\.\d+)?s(\s+\(\d+:\d+:\d+\))?")
NOT_REPRODUCIBLE = {   # prior records without an executable reproduction, with the reason (reported, never hidden)
    "docs/experiments/demo_lifetime_v1_overscaling.json": "superseded record of the pre-fix v1 scaling (kept for history)",
    "docs/experiments/demo_llm_lifetime.json": "needs the local Qwen model on a GPU; reproduced only with a GPU slot",
    "docs/experiments/demo_information.json": "40-day demo (500 s); reproduced only by the slow schedule",
}


def prior_candidates(root: str | None = None) -> list[dict]:
    root = root or M.crucible_root()
    out = []
    d = os.path.join(root, ACCEPTANCE_RECORDS)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f == "summary.json":
            continue
        rec = json.load(open(os.path.join(d, f)))
        m = re.search(r"pytest -q '([^']+)'", rec.get("reproduce", ""))
        out.append({"record": f"{ACCEPTANCE_RECORDS}/{f}", "kind": "acceptance",
                    "node": m.group(1) if m else None, "expected": rec})
    for rel, cmd in DEMOS.items():
        out.append({"record": rel, "kind": "demo", "cmd": cmd, "expected": json.load(open(os.path.join(root, rel)))})
    return out


def _is_timing(key: str) -> bool:
    """Wall-clock quantities (they differ between executions by nature); everything else must reproduce."""
    return bool(TIMING_KEY.search(key.lower()))


def compare_records(expected, got, path="") -> list[str]:
    """Differences between two result records, ignoring timing fields; floats within 1e-6 relative."""
    out = []
    if isinstance(expected, dict) and isinstance(got, dict):
        for k, v in expected.items():
            if _is_timing(str(k)) or k in PROVENANCE_KEYS:
                continue
            if k not in got:
                out.append(f"{path}{k}: missing")
            else:
                out += compare_records(v, got[k], f"{path}{k}.")
    elif isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            out.append(f"{path}: length {len(expected)} vs {len(got)}")
        else:
            for i, (a, b) in enumerate(zip(expected, got)):
                out += compare_records(a, b, f"{path}{i}.")
    elif isinstance(expected, float) or isinstance(got, float):
        try:
            if abs(float(expected) - float(got)) > max(1e-9, 1e-6 * max(abs(float(expected)), abs(float(got)))):
                out.append(f"{path}: {expected!r} vs {got!r}")
        except (TypeError, ValueError):
            out.append(f"{path}: {expected!r} vs {got!r}")
    elif isinstance(expected, str) and isinstance(got, str):
        if DURATION_TEXT.sub("", expected) != DURATION_TEXT.sub("", got):   # "68 passed in 50.19s": the count counts
            out.append(f"{path}: {expected!r} vs {got!r}")
    elif expected != got:
        out.append(f"{path}: {expected!r} vs {got!r}")
    return out


def _plain_clone(root: str, dest: str) -> str:
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    prefix = os.path.relpath(os.path.realpath(root), os.path.realpath(top))
    head = _git(root, "rev-parse", "HEAD").strip()
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", top, dest], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", head], cwd=dest, check=True)
    return os.path.join(dest, prefix)


def _regenerate(tree: str, c: dict, work: str, env: dict, timeout: int) -> tuple[int, dict]:
    """Run one prior record's reproduction command in `tree` (a fresh clone, or the extracted 4.0 tree) and return
    (exit code, the regenerated record)."""
    env = dict(env, PYTHONPATH=os.path.join(tree, "src"))
    if c["kind"] == "acceptance":
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                            c["node"]], cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
        path = os.path.join(tree, c["record"])
    elif "demo_lifetime" in c["record"]:
        path = os.path.join(work, f"{os.path.basename(tree.rstrip(os.sep))}-{os.getpid()}-" + os.path.basename(c["record"]))
        db = path + ".db"
        p = subprocess.run([sys.executable, *c["cmd"], "--out", path, "--store", db], cwd=tree, env=env,
                           capture_output=True, text=True, timeout=timeout)
    else:
        path = os.path.join(tree, c["record"])
        p = subprocess.run([sys.executable, *c["cmd"]], cwd=tree, env=env, capture_output=True, text=True,
                           timeout=timeout)
    try:
        got = json.load(open(path)) if p.returncode == 0 or os.path.exists(path) else {}
    except ValueError:
        got = {}
    return p.returncode, got


def _diff(c: dict, exp: dict, got: dict) -> list[str]:
    if c["kind"] == "acceptance":
        return compare_records({"measured": exp.get("measured", {}), "outcome": exp.get("outcome")},
                               {"measured": got.get("measured", {}), "outcome": got.get("outcome")})
    keys = [k for k, v in exp.items() if not isinstance(v, (dict, list)) and not _is_timing(k) and k not in PROVENANCE_KEYS]
    return compare_records({k: exp[k] for k in keys}, {k: got.get(k) for k in keys})


def prior_results(root: str | None = None, only: list[str] | None = None, timeout: int = 3600) -> dict:
    """Regenerate prior 4.0 records in a fresh clone (the checkout itself is never written) and compare them with the
    committed records. A record the current tree does not reproduce is regenerated by the c07098b tree too: when 4.0
    itself no longer produces its committed record but the current tree produces exactly what 4.0 produces, the
    record counts as reproduced and is reported as `v4_self_irreproducible_records` (the committed number depends on
    conditions the record does not capture; v5 changed nothing about it). Anything else is a mismatch."""
    root = root or M.crucible_root()
    work = tempfile.mkdtemp(prefix="gateA5-", dir=scratch_dir())
    clone = v4_test_clone(root, os.path.join(work, "clone"))[0]     # D-82: 4.0's own test files, byte-identical
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_") and k not in (
        "PYTHONPATH", "CRUCIBLE_V5_RUN_ID", "CRUCIBLE_V5_EVIDENCE_OUT")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", CRUCIBLE_RECORD_ACCEPTANCE="1",
               **HISTORICAL_ENV)                     # D-82: regenerated as the explicitly selected 4.0 baseline
    results = []
    cands = [c for c in prior_candidates(root) if only is None or c["record"] in only]
    for c in cands:
        r = {"record": c["record"], "kind": c["kind"]}
        if c["kind"] == "acceptance" and not c["node"]:
            r.update(attempted=False, why="no reproduction command")
            results.append(r)
            continue
        rc, got = _regenerate(clone, c, work, env, timeout)
        diffs = _diff(c, c["expected"], got)
        r.update(attempted=True, rc=rc, mismatches=diffs[:20], n_mismatches=len(diffs), cls="reproduced")
        if rc != 0 or diffs:
            v4rc, v4got = _regenerate(v4_tree(root), c, work, env, timeout)
            v4_vs_record = _diff(c, c["expected"], v4got)
            v5_vs_v4 = _diff(c, v4got, got) if v4got else ["4.0 produced no record"]
            r.update(v4_rc=v4rc, v4_vs_record=v4_vs_record[:10], v5_vs_v4=v5_vs_v4[:10])
            r["cls"] = "v4_self_irreproducible" if (rc == 0 and v4rc == 0 and v4_vs_record and not v5_vs_v4) \
                else "mismatch"
        results.append(r)
    shutil.rmtree(work, ignore_errors=True)
    att = [r for r in results if r.get("attempted")]
    bad = [r for r in att if r["cls"] == "mismatch"]
    selfirr = [r["record"] for r in att if r["cls"] == "v4_self_irreproducible"]
    return {"metrics": {"reproduction_candidates": len(cands) + (0 if only else len(NOT_REPRODUCIBLE)),
                        "reproduced_results": len(att) - len(bad), "reproduction_mismatches": len(bad),
                        "reproduction_candidates_not_attempted": len(cands) - len(att),
                        "reproduced_exactly_as_recorded": sum(1 for r in att if r["cls"] == "reproduced"),
                        "v4_self_irreproducible_records": len(selfirr),
                        "documented_unreproducible_records": 0 if only else len(NOT_REPRODUCIBLE)},
            "results": results, "not_reproducible": NOT_REPRODUCIBLE, "v4_self_irreproducible": selfirr}
