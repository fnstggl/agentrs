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
