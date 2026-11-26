"""Workstream O unit tests: parent-side sandbox metering (D-20), worker quiescence and quotas, per-tenant uids, the
tool jail, the world-hosting latch, evaluator-only metrics, the calibration import path and engagement integrity."""
import os
import subprocess
import sys
import time

import pytest

from crucible.sandbox import metering as M
from crucible.sandbox.runner import SandboxConfig, Worker, capabilities, run_program

caps = capabilities()
needs_sandbox = pytest.mark.skipif(caps["landlock_abi"] < 1 or not caps["seccomp"] or not caps["root"],
                                   reason=f"needs Landlock+seccomp and root: {caps}")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

LOOP = "def f(n, c):\n    s = 0\n    for i in range(n):\n        s += i\n    return s\n"
FORGE = ("import __main__, sys\ndef f(n, c):\n    s = 0\n    for i in range(n):\n        s += i\n"
         "    for k in ('_LINES', '_LN'):\n        x = getattr(__main__, k, None)\n"
         "        if isinstance(x, list):\n            x[0] = -10 ** 9\n    __main__._cpu_ms = lambda: -1e9\n"
         "    return s\n")


@needs_sandbox
def test_work_is_measured_by_the_parent_and_child_counters_are_ignored():
    cfg = SandboxConfig(count_lines=True)
    h = run_program({"m.py": LOOP}, "m:f", 300_000, cfg=cfg)
    f = run_program({"m.py": FORGE}, "m:f", 300_000, cfg=cfg)
    assert h["ok"] and f["ok"] and h["result"] == f["result"]
    assert f["work_lines"] >= 0.5 * h["work_lines"] > 0 and f["cpu_ms"] > 0
    assert h["meter"]["cpu_ns"] > 0 and h["meter"]["threads"] == 1 and "load_ns" in h["meter"]


@needs_sandbox
def test_deterministic_memo_bills_identical_computations_alike_per_lineage():
    a = run_program({"m.py": LOOP}, "m:f", 50_000, cfg=SandboxConfig(count_lines=True, meter_scope="L1"))
    b = run_program({"m.py": LOOP}, "m:f", 50_000, cfg=SandboxConfig(count_lines=True, meter_scope="L1"))
    c = run_program({"m.py": LOOP}, "m:f", 50_000, cfg=SandboxConfig(count_lines=True, meter_scope="L2"))
    assert a["work_lines"] == b["work_lines"] and b["meter"]["memo"] == "remembered"
    assert c["meter"]["memo"] == "first"                 # another lineage never reuses this lineage's measurement
    m = M.Memo()
    assert m.bill("k", 1000) == (1000, "first") and m.bill("k", 1500) == (1000, "remembered")
    assert m.bill("k", 10 ** 9)[1] == "exceeded" and m.bill("j", -5) == (0, "first")


@needs_sandbox
def test_threads_outliving_a_job_kill_the_worker():
    src = ("import threading\ndef f(a, c):\n    threading.Thread(target=lambda: [0 for _ in iter(int, 1)], "
           "daemon=True).start()\n    return 1\n")
    w = Worker({"t.py": src}).start()
    try:
        r = w.run({"kind": "call", "target": "t:f", "args": None})
        assert r["ok"] and r["meter"]["killed"] and not w.alive()
    finally:
        w.close()


@needs_sandbox
def test_work_after_a_forged_completion_is_billed_at_the_next_job():
    src = ("import __main__\ndef f(a, c):\n    m = __main__.msg\n"
           "    __main__._send({'op': 'done', 'id': m['id'], 'ok': True, 'result': 1, 'error': None, 'stdout': ''})\n"
           "    s = 0\n    for i in range(3_000_000):\n        s += i\n    return 2\n"
           "def g(a, c):\n    return 3\n")
    w = Worker({"t.py": src}).start()
    try:
        r1 = w.run({"kind": "call", "target": "t:f", "args": None})
        time.sleep(0.4)
        r2 = w.run({"kind": "call", "target": "t:g", "args": None})
        assert r1["result"] == 1
        assert not r2["ok"] and r2["error"]["type"] == "WorkerDied" and r2["meter"]["background_ns"] > 20_000_000
    finally:
        w.close()


@needs_sandbox
def test_working_directory_quota_and_entry_limit():
    big = run_program({"d.py": "def f(a, c):\n    for i in range(200):\n        open('f%d' % i, 'wb')"
                                ".write(b'x' * (1 << 20))\n    return 1\n"}, "d:f", None)
    many = run_program({"d.py": "def f(a, c):\n    for i in range(5000):\n        open('e%d' % i, 'w').close()\n"
                                 "    return 1\n"}, "d:f", None)
    assert big["error"]["type"] == "DiskQuota" and many["error"]["type"] == "DiskQuota"


@needs_sandbox
def test_tenants_run_as_their_own_uids_and_host_paths_are_scrubbed():
    src = "import os\ndef uid(a, c):\n    return os.getuid()\ndef boom(a, c):\n    raise ValueError(__file__)\n"
    u1 = run_program({"u.py": src}, "u:uid", None, cfg=SandboxConfig(tenant="world:a"))["result"]
    u2 = run_program({"u.py": src}, "u:uid", None, cfg=SandboxConfig(tenant="world:b"))["result"]
    assert u1 != u2 and 65534 not in (u1, u2) and 0 not in (u1, u2)
    err = run_program({"u.py": src}, "u:boom", None)["error"]
    assert "/var/tmp/crucible-sbx" not in str(err) and "/sandbox/u.py" in err["message"]


@needs_sandbox
def test_jailed_media_tool_cannot_read_host_files_or_connect(tmp_path):
    import shutil
    from crucible.sandbox import isolation
    if not shutil.which("ffmpeg"):
        pytest.fail("ffmpeg is required for the media jail test")
    d = str(tmp_path)
    for argv in (["ffmpeg", "-y", "-f", "data", "-i", "/etc/passwd", "-map", "0", "-c", "copy", "-f", "data", "o"],
                 ["ffmpeg", "-y", "-protocol_whitelist", "tcp,http", "-i", "http://127.0.0.1:9/x", "o.wav"]):
        cmd, cfg = isolation.tool_command(argv, d)
        try:
            assert subprocess.run(cmd, cwd=d, capture_output=True, timeout=60).returncode != 0
        finally:
            os.unlink(cfg)
    assert not os.path.exists(os.path.join(d, "o"))


def test_world_hosting_latch_refuses_the_real_world_bridge():
    code = ("from crucible.hosting import engage\nengage('test')\n"
            "from crucible.realworld import RealWorldBridge, BridgeRefused\n"
            "try:\n    RealWorldBridge()\n    print('OPEN')\nexcept BridgeRefused:\n    print('LATCHED')\n"
            "import subprocess, sys\n"
            "print(subprocess.run([sys.executable, '-c', 'import os; print(os.environ.get(\"CRUCIBLE_HOSTS_WORLDS\"))'],"
            " capture_output=True, text=True).stdout.strip())\n")
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_HOSTS_WORLDS"}
    env["PYTHONPATH"] = SRC
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.stdout.split() == ["LATCHED", "test"], out.stderr[-1000:]


def test_calibration_downloader_is_not_loaded_by_world_code():
    code = ("import sys, crucible\ncrucible.load_all_modules()\n"
            "print('crucible.calibration.datasets' in sys.modules)\n"
            "from crucible.calibration import datasets, acquire\nprint(callable(datasets) and callable(acquire))\n")
    env = dict(os.environ, PYTHONPATH=SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.stdout.split() == ["False", "True"], out.stderr[-1000:]


def test_evaluator_only_metrics_cannot_be_read_by_in_world_agreements():
    import crucible
    crucible.load_all_modules()
    from crucible.goals.metrics import EVALUATOR_ONLY_METRICS, check_predicate_scope
    from crucible.redteam.v5.worlds import v5_world_factory
    w = v5_world_factory("commerce")
    try:
        p = w.system_call("admin.join", {"handle": "orc", "endowment": 1000})["principal"]
        for name in ("mrr_cents", "arr_cents", "independent_revenue", "profit_cents"):
            assert name in EVALUATOR_ONLY_METRICS
            errs = check_predicate_scope(w, {"metric": name, "args": {"principal": p}, "op": ">=", "value": 1}, [p])
            assert errs and "evaluator" in errs[0]
        assert check_predicate_scope(w, {"metric": "cash", "args": {"principal": p}, "op": ">=", "value": 1}, [p]) == []
    finally:
        w.close()


def test_organic_engagement_excludes_related_and_funded_followers():
    import crucible
    crucible.load_all_modules()
    from crucible.redteam.v5.worlds import v5_world_factory
    from crucible.social import integrity
    w = v5_world_factory("social")
    try:
        J = lambda h, e=10_000: w.system_call("admin.join", {"handle": h, "endowment": e})["principal"]  # noqa: E731
        boss, fan, paid = J("boss", 100_000), J("fan"), J("paid", 0)
        X = lambda p, t, **a: w.execute(p, {"tool": t, "args": a})  # noqa: E731
        acct = w.modules["bank"].primary_account(w.state, boss)
        assert X(boss, "bank.transfer", from_account=acct, to=paid, amount=500).ok
        r = X(fan, "studio.write", format="text", title="mine", body="my own garden notes " * 5)
        assert r.ok and X(fan, "posts.publish", ref=r.result["ref"], title="garden").ok
        for p in (fan, paid):
            assert X(p, "social.follow", target=boss).ok
        lab = integrity.classify(w, boss)
        assert lab[fan] == "organic" and lab[paid] == "funded"
        assert integrity.organic_counts(w, boss)["organic"]["follow"] == 1
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- wave 3 (integrated)
@needs_sandbox
def test_module_level_work_is_billed_and_trusted_imports_are_preloaded():
    from crucible.sandbox.runner import preload_list
    assert preload_list({"a.py": "import numpy as np, json\nfrom collections import abc\nimport mymod\n"
                                 "x = 1; import re\n", "b.txt": "import math"}, ("numpy",)) == \
        ["numpy", "json", "collections"]
    at_import = "s = 0\nfor i in range(300000):\n    s += i % 7\ndef f(a, c):\n    return s\n"
    in_call = "def f(a, c):\n    s = 0\n    for i in range(300000):\n        s += i % 7\n    return s\n"
    imp = run_program({"m.py": at_import}, "m:f")
    call = run_program({"m.py": in_call}, "m:f")
    assert imp["ok"] and imp["result"] == call["result"]
    assert imp["cpu_ms"] >= 0.5 * call["cpu_ms"] > 0, (imp["cpu_ms"], call["cpu_ms"])
    npy = run_program({"m.py": "import numpy as np\ndef f(a, c):\n    return int(np.arange(4).sum())\n"}, "m:f")
    assert npy["ok"] and npy["result"] == 6 and npy["meter"]["load_ns"] < M.LOAD_FREE_NS   # preloaded, not billed


@needs_sandbox
def test_hostile_protocol_bytes_fail_the_job_not_the_parent():
    progs = {"nested": "import os\ndef f(a, c):\n    os.write(1, b'[' * 200000 + b']' * 200000 + b'\\n')\n",
             "bigint": "import os\ndef f(a, c):\n    os.write(1, b'{\"op\":\"done\",\"id\":' + b'9' * 5000 + b'}\\n')\n",
             "nan": "import os, time\ndef f(a, c):\n    os.write(1, b'{\"op\":\"done\",\"id\":2,\"ok\":true,\"result\":NaN,"
                    "\"stdout\":\"\"}\\n')\n    time.sleep(0.2)\n    return 1\n",
             "nan_result": "def f(a, c):\n    return float('nan')\n"}
    for name, src in progs.items():
        r = run_program({"m.py": src}, "m:f")
        assert not r["ok"] and r["result"] is None, (name, r)


@needs_sandbox
def test_a_program_that_kills_itself_is_billed_for_its_work():
    src = "import os\ndef f(a, c):\n    s = 0\n    for i in range(300000):\n        s += i % 7\n    os.kill(os.getpid(), 9)\n"
    r = run_program({"m.py": src}, "m:f")
    h = run_program({"m.py": "def f(a, c):\n    s = 0\n    for i in range(300000):\n        s += i % 7\n    return s\n"},
                    "m:f")
    assert not r["ok"] and r["cpu_ms"] >= 0.5 * h["cpu_ms"] > 0, (r["cpu_ms"], h["cpu_ms"])


@needs_sandbox
def test_pool_bills_a_worker_that_worked_and_exited_while_idle_without_failing_the_next_job():
    from crucible.sandbox.runner import WorkerPool
    src = ("import __main__, os\ndef early(a, c):\n    m = __main__.msg\n"
           "    __main__._send({'op': 'done', 'id': m['id'], 'ok': True, 'result': 1, 'error': None, 'stdout': ''})\n"
           "    s = 0\n    for i in range(2_000_000):\n        s += i\n    os.kill(os.getpid(), 9)\n"
           "def ping(a, c):\n    return 2\n")
    pool = WorkerPool(4)
    try:
        w = pool.get("k", lambda: {"s.py": src})
        r1 = w.run({"kind": "call", "target": "s:early", "args": None})
        time.sleep(0.4)
        w2 = pool.get("k", lambda: {"s.py": src})
        r2 = w2.run({"kind": "call", "target": "s:ping", "args": None})
        assert r1["ok"] and r2["ok"] and r2["result"] == 2, (r1, r2)
        assert r2["meter"]["background_ns"] > 20_000_000, r2["meter"]      # the predecessor's work, billed here
    finally:
        pool.close()
