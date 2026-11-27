"""Gate A machinery of the D-82 resolution: the Crucible 4.0 regression suite runs as 4.0 wrote it, as an explicitly
selected historical baseline. These tests check the pieces on the real tree; the full measurement is Gate A.1
(tests/v5/test_gate_A.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5 import baseline as B, checker, official
from crucible.experience import modes


def test_the_historical_baseline_is_selected_only_explicitly(monkeypatch):
    monkeypatch.delenv(modes.HISTORICAL_BASELINE_ENV, raising=False)
    assert modes.check_mode(None) == modes.OBSERVATION_ONLY and not modes.is_explicit_legacy(None)
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "4.0")
    assert modes.check_mode(None) == modes.LEGACY and modes.is_explicit_legacy(None)
    assert modes.check_mode(modes.OBSERVATION_ONLY) == modes.OBSERVATION_ONLY   # a named mode always wins
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "5.0")
    with pytest.raises(ValueError):
        modes.check_mode(None)
    # it never reaches a measured process, and a record made under it is rejected
    env, gone = official.scrubbed({"CRUCIBLE_HISTORICAL_BASELINE": "4.0", "PATH": "/bin"})
    assert "CRUCIBLE_HISTORICAL_BASELINE" in gone and "CRUCIBLE_HISTORICAL_BASELINE" not in env
    probs = checker.env_problems(type("C", (), {"site_hooks": {}})(), {
        "schema": checker.SCHEMA, "interpreter": {"flags": {"no_user_site": True}}, "site_hooks": {},
        "environment": {"historical_baseline": "4.0"}})
    assert any("historical baseline" in p for p in probs)


def test_the_sandbox_counts_executed_lines_only_in_a_4_0_baseline_process(monkeypatch):
    from crucible.sandbox.runner import SandboxConfig, run_program
    src = "def f(n, c):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"
    monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "4.0")
    a = run_program({"s.py": src}, "s:f", 1000, cfg=SandboxConfig(count_lines=True))
    b = run_program({"s.py": src}, "s:f", 3000, cfg=SandboxConfig(count_lines=True))
    assert a["ok"] and a["meter"]["cost_model"].startswith("crucible-4.0")
    assert a["work_lines"] >= 2000 and b["work_lines"] - a["work_lines"] == 2 * 2000   # 2 lines per iteration
    monkeypatch.delenv(modes.HISTORICAL_BASELINE_ENV)
    c = run_program({"s.py": src}, "s:f", 1000, cfg=SandboxConfig(count_lines=True))
    assert c["ok"] and "cost_model" not in c["meter"] and c["meter"]["billed_ns"] >= 0   # 5.0: parent-measured


def test_the_4_0_test_files_are_restored_byte_identical_in_a_fresh_clone(tmp_path):
    croot, restored, differ = B.v4_test_clone(B.M.crucible_root(), str(tmp_path / "clone"))
    assert differ == []
    assert set(restored) == set(B.v4_test_files_changed(B.M.crucible_root()))
    for f in restored:
        want = subprocess.run(["git", "rev-parse", f"{B.V4_COMMIT}:./{f}"], cwd=B.M.crucible_root(),
                              capture_output=True, text=True).stdout.strip()
        assert B._blob(os.path.join(croot, f)) == want
    nodes = B.baseline_nodes()
    assert len(nodes) == 1076 and all(os.path.exists(os.path.join(croot, f)) for f in B.baseline_files(nodes))


LINES_PROBE = r'''
import json, sys
from crucible.sandbox.runner import Worker, SandboxConfig
PROGRAMS = {
    "sorter.py": "def solve(args, caller):\n    recs = sorted(args['records'], key=lambda r: r['key'], reverse=bool(args.get('descending')))\n    return {'ids': [r['id'] for r in recs]}\n",
    "boom.py": "def solve(args, caller):\n    x = args['n'] * 2\n    raise ValueError('bad input %d' % x)\n",
    "lazy.py": "def solve(args, caller):\n    import heapq\n    h = list(args['xs'])\n    heapq.heapify(h)\n    return [heapq.heappop(h) for _ in range(len(h))]\n",
}
recs = [{"id": "r%d" % i, "key": (i * 7919) % 1000} for i in range(40)]
jobs = [("sorter:solve", {"records": recs[:n]}) for n in (0, 1, 7, 40)] + [("boom:solve", {"n": 3}),
        ("lazy:solve", {"xs": [5, 3, 9, 1, 7]}), ("lazy:solve", {"xs": []})]
out = []
for fresh in (False, True):
    w = Worker(PROGRAMS, SandboxConfig(count_lines=True), "probe").start()
    for target, args in jobs:
        r = w.run({"kind": "call", "target": target, "args": args, "caller": "p:x"})
        out.append([target, r.get("ok"), r.get("work_lines")])
        if fresh:                                   # a fresh worker for every job counts like a warm one
            w.close()
            w = Worker(PROGRAMS, SandboxConfig(count_lines=True), "probe").start()
    w.close()
print("@@" + json.dumps(out))
'''


def _lines(tree: str, historical: bool) -> list:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", modes.HISTORICAL_BASELINE_ENV)}
    env.update(PYTHONPATH=os.path.join(tree, "src"), OMP_NUM_THREADS="1",
               **({modes.HISTORICAL_BASELINE_ENV: "4.0"} if historical else {}))
    p = subprocess.run([sys.executable, "-c", LINES_PROBE], cwd=tree, env=env, capture_output=True, text=True,
                       timeout=600)
    assert "@@" in p.stdout, p.stderr[-2000:]
    return json.loads(p.stdout.rsplit("@@", 1)[1])


def test_a_4_0_baseline_process_counts_exactly_the_lines_crucible_4_0_counts():
    """The executed-line count of the 4.0 historical sandbox path equals c07098b's own count, request by request
    (sorting with a key function, an exception and its traceback, an import inside the call; warm and fresh
    workers). Before the wave-3b fix every request counted 17 lines more (v5's `json.dumps(result,
    allow_nan=False)` runs the pure-Python encoder set-up inside the counted region), which changed 4.0 worlds'
    compute bills and, through them, their trajectories (A.1 test_diagnostics, A.5 records)."""
    v4 = _lines(B.v4_tree(B.M.crucible_root()), historical=False)
    v5 = _lines(B.M.crucible_root(), historical=True)
    assert all(ok is not None and n is not None for _, ok, n in v4)
    assert [x[1] for x in v4] == [True] * 4 + [False, True, True] + [True] * 4 + [False, True, True]
    assert v5 == v4, (v5, v4)


def test_the_whole_suite_runner_submits_each_resource_class_through_the_slot_scheduler(tmp_path, monkeypatch):
    """Gate X.2/X.4's runner (`run_pytest_groups(via_slots=...)`) on a two-file mini suite, against a private slot
    ledger (so the smoke test never waits behind the machine's queue): every group is a slot job of its own resource
    class, the JUnit of each comes back, an isolated file runs in a process of its own, and the ledger records each
    job with the class's CPU, GPU and RAM request."""
    root = tmp_path / "tree"
    (root / "tests" / "v5").mkdir(parents=True)
    os.symlink(os.path.join(B.M.crucible_root(), "src"), root / "src")
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert sum(range(4)) == 6\n")
    (root / "tests" / "v5" / "test_gate_b.py").write_text("def test_b():\n    assert 'b'.upper() == 'B'\n")
    (root / "tests" / "test_iso.py").write_text("def test_iso():\n    assert True\n")
    ledger = tmp_path / "slots"
    monkeypatch.setenv("CRUCIBLE_SLOTS_DIR", str(ledger))
    monkeypatch.setenv("CRUCIBLE_GATE_X_RESOURCES", json.dumps([[r"tests/v5/", 1, 0, 2100, 600],
                                                                 [r"", 1, 0, 2000, 600]]))
    files = ["tests/test_a.py", "tests/v5/test_gate_b.py", "tests/test_iso.py"]
    res = B.run_pytest_groups(str(root), files, str(tmp_path / "out"), workers=1, timeout=900,
                              isolate=("tests/test_iso.py",), via_slots="P: smoke")
    assert res["rcs"] == [0, 0, 0], res
    assert sorted((c["name"], c["outcome"]) for c in res["cases"]) == [("test_a", "passed"), ("test_b", "passed"),
                                                                       ("test_iso", "passed")]
    jobs = [json.loads(ln) for ln in open(ledger / "usage.jsonl")]
    assert sorted((j["label"].split(" (")[0], j["ram_mb"]) for j in jobs) == [
        ("P: smoke: suite group 0", 2000), ("P: smoke: suite group 1", 2100), ("P: smoke: suite group 2", 2000)], jobs
    assert all(j["cpu"] == 1 and j["gpu_mb"] == 0 and j["rc"] == 0 for j in jobs)


def _fork_events(monkeypatch, historical: bool) -> list[str]:
    import crucible
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    if historical:
        monkeypatch.setenv(modes.HISTORICAL_BASELINE_ENV, "4.0")
    else:
        monkeypatch.delenv(modes.HISTORICAL_BASELINE_ENV, raising=False)
    crucible.load_all_modules()
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 20}}, 3))
    for h in ("ann", "bo"):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
    ex = lambda who, t, **k: w.execute(f"p:{who}", {"tool": t, "args": k})  # noqa: E731
    repo = ex("ann", "code.create_repo", name="lib", visibility="public", license="open").result["repo"]
    assert ex("ann", "code.commit", repo=repo, files={"lib.py": "X = 1\n"}).ok
    since = w.seq
    assert ex("bo", "code.fork", repo=repo, name="mine").ok
    return [e.type for e in w.log if e.seq > since]


def test_a_4_0_baseline_process_emits_only_4_0_events_from_4_0_tools(monkeypatch):
    """5.0 added a lineage event to 4.0's `code.fork` (`code.forked`, innovation). Every event advances the main
    sequence that keys the command RNG, so the extra event made every later random draw of a 4.0 world differ:
    demo_lifetime diverged on day 4 (10,683 events vs c07098b's 10,657). In a 4.0 historical-baseline process
    4.0's tools emit 4.0's events (D-82 addendum); a 5.0 process keeps the lineage record."""
    v5 = _fork_events(monkeypatch, historical=False)
    v4 = _fork_events(monkeypatch, historical=True)
    assert "code.forked" in v5 and "code.forked" not in v4
    assert [t for t in v5 if t != "code.forked"] == v4


def test_the_sealed_store_scan_matches_the_4_0_names_as_whole_tokens():
    """A.4's static scan names the 4.0 sealed store's files, variables and module. The 5.0 evaluator keeps its own
    sealed store (`v5_sealed_salt` in a directory its caller chooses, CRUCIBLE_V5_SEALED_DIR); a substring match made
    that a reference to the 4.0 store (research/v5/evaluator.py:78)."""
    hits = ['os.path.join(d, "sealed_salt")', 'os.environ["CRUCIBLE_SEALED_DIR"]', "from crucible.research import sealed",
            "import crucible.research.sealed as s", "S.open_set('x')", "p = root + '/sealed_tasks/a.json'",
            "json.load(open(os.path.join(d, 'sealed_sets.json')))"]
    misses = ['os.path.join(d, "v5_sealed_salt")', 'env["CRUCIBLE_V5_SEALED_DIR"] = d', "reopen_set(3)",
              "unsealed_tasks = []"]
    assert [B.sealed_reference(c) for c in hits] == [True] * len(hits)
    assert [B.sealed_reference(c) for c in misses] == [False] * len(misses)


def test_rp1_sees_gates_that_load_agentrs_code_by_name():
    """Wave 3b verifier: RP.1 ('no acceptance gate depends on CAM's completion or success') judged dependence from the
    requirement texts only, so Gate A.1 - whose child processes load the AGENTRS alias plugin and the relocated CAM
    adapters - counted as independent. The scan now follows what each gate runs (its producing test functions, the
    implementation functions they name, module constants), so A.1 is found and A.2, which shares the gate file and the
    implementation module, is not."""
    from crucible.acceptance.v5 import qa
    root = B.M.crucible_root()
    runs = qa.gates_running_agentrs_code(root)
    assert any("crucible_v4_names" in h for h in runs.get("A.1", [])), runs.get("A.1")
    assert "A.2" not in runs and "A.4" not in runs and "X.7" not in runs, sorted(runs)
    assert "A.1" in qa.cam_scan(root)["gates"]
