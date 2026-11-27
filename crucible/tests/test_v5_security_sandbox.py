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


def _jail_ok():
    from crucible.sandbox import isolation
    return isolation.available()[0]


@pytest.mark.skipif(not _jail_ok(), reason="the tool jail needs root, Landlock and libseccomp")
def test_jailed_image_decoder_gives_identical_pixels_and_cannot_reach_host_or_network():
    import io
    import tempfile
    from PIL import Image
    from crucible.sandbox import isolation
    im = Image.new("RGB", (40, 30), (10, 200, 30))
    for x in range(40):
        im.putpixel((x, x % 30), (x * 5, 255 - x, 7))
    for fmt in ("PNG", "JPEG"):
        buf = io.BytesIO()
        im.save(buf, fmt)
        r = isolation.decode_image(buf.getvalue())
        ref = Image.open(io.BytesIO(buf.getvalue()))
        ref.load()
        assert r["ok"] and r["raw"] == ref.convert("RGB").tobytes() and r["size"] == [40, 30], r.get("error")
    bad = isolation.decode_image(b"garbage" * 20)
    assert not bad["ok"] and "/var/tmp" not in bad["error"]
    w = tempfile.mkdtemp(dir=isolation._base())
    code = ("import os, socket, sys\nout = []\n"
            "for name, fn in (('passwd', lambda: open('/etc/passwd').read()), ('socket', lambda: socket.socket()),\n"
            "                 ('unix', lambda: socket.socket(socket.AF_UNIX)), ('fork', lambda: os.fork()),\n"
            "                 ('write_tmp', lambda: open('/tmp/x-decoder-probe', 'w'))):\n"
            "    try:\n        fn()\n        out.append(name)\n    except BaseException:\n        pass\n"
            "open(os.path.join(sys.argv[1], 'r.txt'), 'w').write(','.join(out))\n")
    cmd, cfg = isolation.python_command(w, code)
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        assert open(os.path.join(w, "r.txt")).read() == ""            # every attempt refused
    finally:
        os.unlink(cfg)
        import shutil
        shutil.rmtree(w, ignore_errors=True)


@pytest.mark.skipif(not _jail_ok(), reason="the tool jail needs root, Landlock and libseccomp")
def test_isolation_run_is_a_jailed_drop_in_for_media_tool_call_sites():
    import shutil
    import tempfile
    from crucible.sandbox import isolation
    if not shutil.which("ffprobe"):
        pytest.fail("ffprobe is required on this host for the media tools")
    d = tempfile.mkdtemp(prefix="rt-run-")
    try:
        p = isolation.run(["ffprobe", "-v", "error", "-f", "data", "-i", "/etc/passwd"], cwd=d, timeout=20)
        assert p.returncode != 0                                     # host file unreadable in the jail
        ok = isolation.run(["ffprobe", "-version"], cwd=d, timeout=20)
        assert ok.returncode == 0 and b"ffprobe" in ok.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- wave 3b review (W.n1)
@needs_sandbox
def test_nested_jobs_share_the_enclosing_jobs_time_limit():
    """A program's world call that runs another sandboxed program used to get a fresh time limit while the caller's
    clock stood still (time in the broker is not the caller's): fan-outs of sleeping programs held the world for
    hours. A nested job now gets at most what is left of the enclosing job's limit, and none once it is spent."""
    inner_runs = []

    def rpc(tool, args):                      # the world call runs a nested sandboxed job, like code.run does
        t = time.monotonic()
        r = run_program({"z.py": "import time\ndef f(a, c):\n    time.sleep(5)\n    return 1\n"}, "z:f", timeout=30)
        inner_runs.append((round(time.monotonic() - t, 2), (r.get("error") or {}).get("type")))
        return True, r.get("ok")
    outer = "import crucible_sdk\ndef f(a, c):\n    return [crucible_sdk.call('x') for _ in range(3)]\n"
    t0 = time.monotonic()
    r = run_program({"o.py": outer}, "o:f", rpc=rpc, timeout=2.0)
    wall = time.monotonic() - t0
    assert r["ok"] and r["result"] == [False, False, False], r
    assert wall < 2 * 2.0 + 3.0, (wall, inner_runs)                     # not 3 x 5 s
    assert inner_runs[0][1] == "Timeout" and inner_runs[0][0] < 2.5, inner_runs
    assert all(e == "Timeout" for _, e in inner_runs), inner_runs


@needs_sandbox
def test_protocol_allowance_cannot_hide_work_in_padded_messages():
    work = "def f(a, c):\n    s = 0\n    for i in range(2_000_000):\n        s += i % 7\n    return s\n"
    pad = ("import crucible_sdk\n" + work.replace("def f(a, c):\n", "def f(a, c):\n    p = 'x' * 7_000_000\n"
           "    for k in range(20):\n        crucible_sdk.call('t', pad=p)\n"))
    cfg = SandboxConfig(count_lines=True, meter_scope="proto-test-%d" % os.getpid())
    h = run_program({"m.py": work}, "m:f", cfg=cfg)
    f = run_program({"m.py": pad}, "m:f", cfg=cfg, rpc=lambda tool, args: (True, None), timeout=60)
    assert h["ok"] and f["ok"] and f["result"] == h["result"], (h, f)
    assert f["work_lines"] >= 0.5 * h["work_lines"] > 0, (f["work_lines"], h["work_lines"])


def test_meter_memo_file_evicts_per_lineage(tmp_path):
    path = str(tmp_path / "memo.sqlite")
    a = M.DiskMemo(path, max_rows=500, scope_max=100, check_every=50)
    b = M.DiskMemo(path, max_rows=500, scope_max=100, check_every=50)
    mine = [M.digest(["victim", i]) for i in range(40)]
    for k in mine:
        assert a.first(k, 5, "victim") == (5, True)
    for i in range(3000):                                   # another lineage (another world) floods the shared file
        b.first(M.digest(["flood", i]), 1, "flood")
    assert all(M.DiskMemo(path).first(k, 9, "victim") == (5, False) for k in mine)
    import sqlite3
    n = dict(sqlite3.connect(path).execute("SELECT scope, count(*) FROM memo GROUP BY scope").fetchall())
    assert n["victim"] == 40 and n["flood"] <= 100 + 50, n
    m = M.Memo(max_entries=10)                              # the in-process memo is bounded per lineage too
    m.bill("v", 7, scope="victim")
    for i in range(100):
        m.bill(f"f{i}", 1, scope="flood")
    assert m.bill("v", 8, scope="victim") == (7, "remembered")


def test_jailed_browser_starts_below_a_root_only_tmpdir(tmp_path):
    """DEFECTS DEF-P-16: Playwright creates Chromium's profile under the caller's TMPDIR; below a root-only directory
    (an agent scratchpad of mode 700) the jail uid could not reach it and every jailed-browser test failed."""
    from crucible.sandbox import isolation
    ok, why = isolation.available()
    assert ok, why
    private = tmp_path / "private"
    (private / "tmp").mkdir(parents=True)
    os.chmod(private, 0o700)
    code = ("from playwright.sync_api import sync_playwright\nfrom crucible.sandbox import isolation\n"
            "exe, kw = isolation.browser_executable('rt-tmpdir-test')\n"
            "with sync_playwright() as pw:\n"
            "    b = pw.chromium.launch(headless=True, executable_path=exe, args=['--disable-gpu'], **kw)\n"
            "    p = b.new_page()\n    p.set_content('<title>jailed</title><p>ok</p>')\n    print(p.title())\n"
            "    b.close()\n")
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_HOSTS_WORLDS"}
    env.update(PYTHONPATH=SRC, TMPDIR=str(private / "tmp"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "jailed", r.stderr[-1500:]


def test_learner_host_runs_the_learner_jailed_with_only_its_participant_channel():
    from crucible.sandbox import isolation
    from crucible.sandbox.learnerhost import LearnerHost
    ok, why = isolation.available()
    assert ok, why
    with LearnerHost("crucible.redteam.v5.adversaries:make", {"name": "prober", "seed": 0}) as h:
        rep = h.call("probe")
        # its own uid (one per learner process, above the per-kind jail uids; wave-3b review)
        assert h.jail_uid not in (0, 65534) and h.jail_uid == h.uid
        assert isolation.UID_BASE + 0x10000 <= h.jail_uid < isolation.UID_BASE + 0x20000
    found, att = rep["found"], rep["attempts"]
    assert not {k for k in found if k in ("other_process_environ_read", "sealed_salt_read", "evaluator_memory_read",
                                          "gc:evaluator", "tmp_listed")}, found
    assert att["list_proc"] != "ok" and att["listdir /tmp"] != "ok", att


def test_a_4_0_open_world_with_agent_programs_is_reproducible_in_new_processes(tmp_path):
    """DEFECTS DEF-P-27: since D-20 a 4.0 open world whose agents run sandboxed programs billed each fresh process's
    own CPU measurement, so the same spec and seed gave different trajectories run to run (demo_lifetime: 74,280 vs
    78,900 events). With the machine-level meter memo (A/K3; evicted per lineage since wave 3b) two NEW processes
    give identical event hashes; without it (per-process memos) they diverge - the noise the memo removes."""
    code = (
        "import json, sys\nsys.path.insert(0, sys.argv[1])\nimport numpy as np\nimport crucible\n"
        "crucible.load_all_modules()\nfrom crucible.core.kernel import World\nfrom crucible.agents.base import Runner\n"
        "from crucible.worldgen.ow_families import ow_categories\nfrom crucible.usefulness.agents import ExampleInductor, "
        "GarbageSeller, Shopper\nfrom crucible.worldgen import generate\n"
        "spec = generate({'family': 'ow_tiny', 'overrides': {'n_consumers': 300}}, 1)\nw = World.create(spec)\n"
        "cat = next(c for c in ow_categories(spec) if c.family == 'transform')\n"
        "for h in ('ind', 'junk', 'buyer'):\n    w.system_call('admin.join', {'handle': h, 'endowment': 3_000_000})\n"
        "req, _ = cat.generate(np.random.default_rng(4242), 0.6)\n"
        "run = Runner(w, [('p:ind', ExampleInductor()), ('p:junk', GarbageSeller(cat.name)), "
        "('p:buyer', Shopper(cat.name, req, wait_rounds=20))], dt=3600)\nrun.run(30)\n"
        "jobs = sum(1 for e in w.log if e.type in ('cloud.usage', 'code.ran', 'code.run'))\n"
        "print(json.dumps({'seq': w.seq, 'hash': w.last_hash, 'jobs': jobs, 'sandbox_outcomes': sum(1 for e in w.log "
        "if e.type == 'ext.outcome' or 'cpu_ms' in json.dumps(e.payload, default=str))}))\n")

    def run(memo):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CRUCIBLE_METER_MEMO=memo)
        env.pop("CRUCIBLE_HOSTS_WORLDS", None)
        p = subprocess.run([sys.executable, "-c", code, SRC], capture_output=True, text=True, timeout=900, env=env)
        assert p.returncode == 0, p.stderr[-3000:]
        return json.loads(p.stdout.strip().splitlines()[-1])
    import json
    memo = str(tmp_path / "memo.sqlite")
    a, b = run(memo), run(memo)
    assert a["sandbox_outcomes"] > 10, ("the agents must run sandboxed programs for this test to mean anything", a)
    assert (a["seq"], a["hash"]) == (b["seq"], b["hash"]), (a, b)


def test_learner_host_runs_an_experience_remote_learner_function_jailed(tmp_path):
    """The experience package's learner contract (fn(participant, **kw)) over the participant pipe of
    `crucible.experience.remote.host`, inside the jail: the episode is ordinary, the adversary reaches nothing (no
    /proc, no ~/.crucible sealed files, no evaluator object)."""
    code = (
        "import json, sys\nsys.path.insert(0, sys.argv[1])\n"
        "def main():\n"
        "    from crucible.experience.remote import host\n"
        "    from crucible.sandbox.learnerhost import LearnerHost\n"
        "    part, ev = host({'tier': 1, 'name': 'pe_rule_induction', 'seed': 3, 'reward_mode': 'observation_only'})\n"
        "    try:\n"
        "        with LearnerHost(None, participant=part._conn, protocol='experience-remote') as h:\n"
        "            r = h.function('crucible.redteam.v5.adversaries:probe_participant', {})\n"
        "        v = ev.verify()\n"
        "    finally:\n"
        "        ev.close()\n"
        "    print(json.dumps({'report': r, 'verified': bool(v)}))\n"
        "if __name__ == '__main__':\n    main()\n")
    f = tmp_path / "run_hosted.py"
    f.write_text(code)
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop("CRUCIBLE_HOSTS_WORLDS", None)
    p = subprocess.run([sys.executable, str(f), SRC], capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    import json
    out = json.loads(p.stdout.strip().splitlines()[-1])
    rep = out["report"]
    assert out["verified"] and rep["acted"] is True and rep["uid"] not in (0, 65534), rep
    assert rep["found"] == {}, rep["found"]


def test_concurrent_jailed_browsers_of_one_kind_get_private_homes():
    """Two worlds' browsers of one kind share the jail uid; each launch gets its own HOME/TMPDIR/XDG directory (mode
    700) instead of the kind's shared home, so no writable state is shared between them (wave 3b review)."""
    from crucible.sandbox import isolation
    ok, why = isolation.available()
    assert ok, why
    code = ("import os, sys, json\nsys.path.insert(0, sys.argv[1])\nfrom playwright.sync_api import sync_playwright\n"
            "from crucible.sandbox import isolation\nexe, kw = isolation.browser_executable('rt-private-home')\n"
            "runs = os.path.join(isolation._base(), 'home-%d' % isolation.uid_for('rt-private-home'), 'runs')\n"
            "with sync_playwright() as pw:\n"
            "    a = pw.chromium.launch(headless=True, executable_path=exe, args=['--disable-gpu'], **kw)\n"
            "    b = pw.chromium.launch(headless=True, executable_path=exe, args=['--disable-gpu'], **kw)\n"
            "    live = sorted(x for x in os.listdir(runs) if os.path.exists('/proc/%s' % x.split('-')[1]))\n"
            "    st = [os.stat(os.path.join(runs, x)) for x in live]\n"
            "    print(json.dumps({'live': live, 'modes': [oct(s.st_mode & 0o777) for s in st], "
            "'owners': sorted({s.st_uid for s in st})}))\n"
            "    a.close(); b.close()\n")
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_HOSTS_WORLDS"}
    r = subprocess.run([sys.executable, "-c", code, SRC], capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    import json
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(out["live"]) >= 2 and set(out["modes"]) == {"0o700"}, out
    assert out["owners"] == [isolation.uid_for("rt-private-home")], out


IPC = ("import ctypes\ndef f(a, c):\n    libc = ctypes.CDLL(None, use_errno=True)\n    out = {}\n"
       "    for k, fn in (('shm', lambda: libc.shmget(0, 4096, 0o1600)), ('msg', lambda: libc.msgget(0, 0o1600)),\n"
       "                  ('sem', lambda: libc.semget(0, 1, 0o1600))):\n"
       "        i = fn()\n        out[k] = ctypes.get_errno() if i < 0 else 'created'\n"
       "        if i >= 0:\n            (libc.shmctl if k == 'shm' else libc.msgctl if k == 'msg' else libc.semctl)(i, 0, 0)\n"
       "    return out\n")


@needs_sandbox
def test_sandboxed_programs_cannot_create_kernel_ipc_objects():
    """Wave-3b review: System V IPC objects are named by a key (Landlock does not mediate them on this kernel) and
    outlive the program; two worlds' programs passed a canary through shared memory. The sandbox refuses them."""
    r = run_program({"i.py": IPC}, "i:f", None)
    assert r["ok"] and r["result"] == {"shm": 1, "msg": 1, "sem": 1}, r


@needs_sandbox
def test_jailed_learner_processes_have_own_uids_and_cannot_signal_each_other_or_share_ipc():
    """Wave-3b review: learner processes of one kind shared a uid; one SIGKILLed another through descriptor ownership
    (F_SETOWN + F_SETSIG on a pipe with O_ASYNC) and carried state to a fresh learner process through SysV IPC."""
    from crucible.sandbox.learnerhost import LearnerHost
    with LearnerHost(None) as v, LearnerHost(None) as a:
        assert v.jail_uid != a.jail_uid
        sig = a.function("crucible.redteam.v5.adversaries:signal_owner", {"victim": v.pid, "absent": 1})
        ipc = a.function("crucible.redteam.v5.adversaries:ipc_stash", {"key": 0x5E7E5700 + os.getpid() % 256,
                                                                       "canary": "x"})
        time.sleep(0.3)
        assert v.proc.poll() is None, sig
        assert sig["setown_victim"] == sig["setsig"] == sig["setown_absent_pid"] == "errno 1", sig
        assert all(ipc[k] == "errno 1" for k in ("shm", "msg", "sem", "mq")), ipc


SQL = ("import sqlite3\ndef f(a, c):\n    db = sqlite3.connect('t.db')\n    db.execute('pragma journal_mode=wal')\n"
       "    db.execute('create table t(x)')\n    db.executemany('insert into t values (?)', [(i,) for i in range(10)])\n"
       "    db.commit()\n    return db.execute('select sum(x) from t').fetchone()[0]\n")


@needs_sandbox
def test_advisory_locks_do_not_reach_other_tenants_and_sqlite_still_works():
    """Wave-3b review: flock() on read-only standard-library files and record locks on /dev/null carried bits
    between two tenants' programs. Locks are private to a program now (taking one succeeds without effect), so a
    long-lived worker's locks are invisible to another tenant, while sqlite3 (WAL mode too) keeps working."""
    from crucible.redteam.v5.crossworld import LOCK_HOLDER, LOCK_READER
    secret = 0b1010011
    w = Worker({"h.py": LOCK_HOLDER}, cfg=SandboxConfig(tenant="lock-holder")).start()
    try:
        held = w.run({"kind": "call", "target": "h:take", "args": {"bits": secret}})
        assert held["ok"] and w.alive(), held
        r = run_program({"r.py": LOCK_READER}, "r:read", None, cfg=SandboxConfig(tenant="lock-reader"))
        assert r["ok"] and r["result"]["bits"] == {"flock": 0, "devnull": 0, "rdlock": 0}, r
    finally:
        w.close()
    s = run_program({"s.py": SQL}, "s:f", None)
    assert s["ok"] and s["result"] == 45, s


@needs_sandbox
def test_file_watches_on_shared_files_are_refused():
    """Wave-3b review: a long-lived worker's inotify queue on standard-library files received another world's
    program's opens (16 bits crossed). Sandboxed programs cannot create watches."""
    src = ("import ctypes\ndef f(a, c):\n    libc = ctypes.CDLL(None, use_errno=True)\n"
           "    fd = libc.inotify_init1(0)\n    return ctypes.get_errno() if fd < 0 else 'created'\n")
    r = run_program({"w.py": src}, "w:f", None)
    assert r["ok"] and r["result"] == 1, r


@needs_sandbox
def test_memory_files_are_refused_and_learner_directories_have_a_quota():
    """Wave-3b review: memfd pages are not mapped, so RLIMIT_AS does not count them (640 MiB written under a 256 MiB
    cap in the sandbox; 512 MiB held by a learner process under a 512 MiB cap), and a learner process's private
    directory had no quota (RLIMIT_FSIZE bounds one file). memfd is refused; a watchdog stops the learner at its quota."""
    from crucible.sandbox.learnerhost import LearnerHost, LearnerHostError
    src = ("import ctypes\ndef f(a, c):\n    libc = ctypes.CDLL(None, use_errno=True)\n"
           "    fd = libc.syscall(319, b'm', 0)\n    return ctypes.get_errno() if fd < 0 else 'created'\n")
    r = run_program({"m.py": src}, "m:f", None, cfg=SandboxConfig(mem_mb=256))
    assert r["ok"] and r["result"] == 1, r
    with LearnerHost(None, mem_mb=512, disk_quota_mb=16) as h:
        assert h.function("crucible.redteam.v5.adversaries:memfd_hog", {"n": 2, "mb": 8})["bytes"] == 0
        with pytest.raises(LearnerHostError, match="quota"):
            h.function("crucible.redteam.v5.adversaries:disk_hog", {"files": 40, "mb": 2, "pause_s": 0.05}, timeout=60)


# ---------------------------------------------------------------------- repair pass after the wave-3b verifier
SELFATTR = r'''
import os, time, threading, ctypes, fcntl
def f(a, c):
    libc = ctypes.CDLL(None, use_errno=True)
    out = {}
    def t(name, fn):
        try:
            out[name] = ["ok", fn()]
        except OSError as e:
            out[name] = ["errno", e.errno]
    other = os.getppid()
    t("own_nice", lambda: (os.setpriority(os.PRIO_PROCESS, 0, 5), os.getpriority(os.PRIO_PROCESS, 0))[1])
    t("other_nice", lambda: os.getpriority(os.PRIO_PROCESS, other))
    t("user_nice", lambda: os.getpriority(os.PRIO_USER, 0))
    t("other_sched", lambda: os.sched_getscheduler(other))
    t("own_sched", lambda: os.sched_getscheduler(0))
    t("other_affinity", lambda: len(os.sched_getaffinity(other)))
    t("own_affinity", lambda: len(os.sched_getaffinity(0)) > 0)
    t("other_pgid", lambda: os.getpgid(other))
    t("other_sid", lambda: os.getsid(other))
    r = libc.syscall(252, 1, other)
    out["other_ioprio"] = ["ok", r] if r >= 0 else ["errno", ctypes.get_errno()]
    t("other_cpu_clock", lambda: time.clock_gettime_ns(((~other) << 3) | 2))
    t("own_cpu_clock", lambda: time.process_time_ns() > 0)
    t("thread_cpu_clock", lambda: time.clock_gettime_ns(time.pthread_getcpuclockid(threading.get_ident())) > 0)
    def dn():
        fd = os.open(os.path.join(os.path.dirname(os.__file__), "json"), os.O_RDONLY | os.O_DIRECTORY)
        return fcntl.fcntl(fd, fcntl.F_NOTIFY, fcntl.DN_ACCESS)
    t("dnotify", dn)
    word = ctypes.cast(libc.getpid, ctypes.c_void_p).value & ~3
    r = libc.syscall(202, ctypes.c_void_p(word), 1, 1, None, None, 0)              # shared FUTEX_WAKE
    out["shared_wake"] = ["ok", r] if r >= 0 else ["errno", ctypes.get_errno()]
    r = libc.syscall(202, ctypes.c_void_p(word), 1 | 128, 1, None, None, 0)        # private FUTEX_WAKE
    out["private_wake"] = ["ok", r] if r >= 0 else ["errno", ctypes.get_errno()]
    # ordinary threading (locks, conditions, join) needs only private futexes and the kernel's own exit wake-up
    box, lock, cond = [], threading.Lock(), threading.Condition()
    def work():
        with cond:
            box.append(1)
            cond.notify_all()
    th = [threading.Thread(target=work) for _ in range(4)]
    for x in th:
        x.start()
    for x in th:
        x.join(5)
    with lock:
        out["threads_joined"] = ["ok", len(box) == 4 and not any(x.is_alive() for x in th)]
    return out
'''


@needs_sandbox
def test_other_processes_attributes_clocks_notifications_and_shared_futexes_are_refused():
    """Repair pass after the wave-3b verifier: scheduling attributes of other pids were readable (a 16-bit secret
    crossed between two worlds in 12 s, and ESRCH was a pid oracle), F_NOTIFY watched other tenants' reads, a shared
    futex on libc's code woke another tenant's waiter, and other processes' CPU clocks were readable. All refused;
    the program's own attributes, CPU clocks, private futexes and ordinary threading still work."""
    r = run_program({"a.py": SELFATTR}, "a:f", None)
    assert r["ok"], r
    o = r["result"]
    for k in ("other_nice", "user_nice", "other_sched", "other_affinity", "other_pgid", "other_sid", "other_ioprio",
              "dnotify", "shared_wake"):
        assert o[k] == ["errno", 1], (k, o)
    assert o["other_cpu_clock"] == ["errno", 22], o
    assert o["own_nice"] == ["ok", 5] and o["own_sched"][0] == "ok" and o["own_affinity"] == ["ok", True], o
    assert o["own_cpu_clock"] == o["thread_cpu_clock"] == ["ok", True], o
    assert o["private_wake"] == ["ok", 0] and o["threads_joined"] == ["ok", True], o


@needs_sandbox
def test_jailed_learner_processes_see_no_other_process_and_never_share_a_uid():
    """Repair pass: the tool-mode jail refuses the same reads; a learner whose uid is forced to collide with
    another's cannot change that one's limits or nice value; concurrently started learner hosts claim distinct uids
    (the 16-bit hash collided with probability ~n^2/2^17) and release their claims."""
    from crucible.sandbox import learnerhost as LH
    hosts = [LH.LearnerHost(None) for _ in range(3)]
    try:
        uids = [h.jail_uid for h in hosts]
        assert len(set(uids)) == 3
        claims = set(os.listdir(LH._claim_dir()))
        assert {str(u) for u in uids} <= claims
        a, b = hosts[0], hosts[1]
        sc = b.function("crucible.redteam.v5.adversaries:attr_scan", {"lo": a.pid - 5, "hi": a.pid + 5,
                                                                        "absent": 4_000_000})
        assert sc["readable_pids"] == 0 and sc["oracle"]["absent"] == sc["oracle"]["parent"] == "errno 1", sc
        clk = b.function("crucible.redteam.v5.adversaries:cpuclock_read", {"pids": [a.pid]})
        assert clk[str(a.pid)] == "errno 22", clk
        real = LH._instance_uid
        LH._instance_uid = lambda kind: a.jail_uid
        try:
            c = LH.LearnerHost(None)
        finally:
            LH._instance_uid = real
        with c:
            assert c.jail_uid == a.jail_uid
            r = c.function("crucible.redteam.v5.adversaries:prlimit_other", {"victim": a.pid})
            assert r["prlimit"] == r["renice"] == "errno 1", r
        assert a.function("crucible.redteam.v5.adversaries:burn_cpu", {"ms": 1200})["burned_ms"] == 1200
    finally:
        for h in hosts:
            h.close()
    assert not ({str(u) for u in uids} & set(os.listdir(LH._claim_dir())))


@needs_sandbox
def test_disk_quotas_count_deleted_files_held_open():
    """Wave-3b verifier: the quotas read the directory listing, so files unlinked while held open were invisible (a
    learner held 8x its quota, a long-lived sandbox worker 2x across jobs). The parent now counts them (under the
    process's own fs identity: root has no CAP_SYS_PTRACE here); a deleted file kept alive by a mapping alone, whose
    size cannot be read, stops the process too. Files that are still named are counted once, as before."""
    from crucible.redteam.v5.crossworld import _attack_files
    from crucible.sandbox.learnerhost import LearnerHost, LearnerHostError
    files = _attack_files("hd.py", ("hidden_disk_hog", "mapped_deleted_hog"),
                          "def held(a, c):\n    return hidden_disk_hog(None, a['files'], a['mb'])\n"
                          "def mapped(a, c):\n    return mapped_deleted_hog(None, a['mb'], 1)\n")
    w = Worker(files, SandboxConfig(disk_quota_mb=8)).start()
    try:
        r = w.run({"kind": "call", "target": "hd:held", "args": {"files": 2, "mb": 6}})
        assert not r["ok"] and "deleted files held open" in str(r["meter"]["killed"]) and not w.alive(), r
    finally:
        w.close()
    w = Worker(files, SandboxConfig(disk_quota_mb=64)).start()
    try:
        r = w.run({"kind": "call", "target": "hd:mapped", "args": {"mb": 2}})
        assert not r["ok"] and "memory mapping" in str(r["meter"]["killed"]), r
    finally:
        w.close()
    ok = Worker({"k.py": "import os\nF = []\ndef f(a, c):\n    F.append(open('x', 'wb'))\n"
                         "    F[0].write(b'z' * (1 << 20))\n    F[0].flush()\n    return 1\n"}, SandboxConfig(disk_quota_mb=8)).start()
    try:
        assert ok.run({"kind": "call", "target": "k:f", "args": None})["ok"]
        b, n, unm = ok.disk_held()
        assert 2**20 <= b < 3 * 2**20 and n == 2 and not unm, (b, n, unm)    # the named file once, not twice
    finally:
        ok.close()
    with LearnerHost(None, disk_quota_mb=16) as h:
        with pytest.raises(LearnerHostError, match="quota"):
            h.function("crucible.redteam.v5.adversaries:hidden_disk_hog", {"files": 3, "mb": 8}, timeout=60)


def test_in_process_memo_is_bounded_process_wide_and_the_largest_lineage_pays():
    """Wave-3b verifier: the in-process memo was bounded per lineage only (50 lineages held 50x the bound)."""
    m = M.Memo(max_entries=1000)
    for sc in range(50):
        for i in range(1000):
            m.bill(f"k{i}", 5, scope=f"s{sc}")
    assert len(m) <= 1000 and sum(len(d) for d in m._d.values()) == len(m)
    v = M.Memo(max_entries=10)
    v.bill("v", 7, scope="victim")
    for i in range(100):
        v.bill(f"f{i}", 1, scope="flood")
    assert v.bill("v", 8, scope="victim") == (7, "remembered") and len(v) <= 10


LOCKSEM = r'''
import fcntl, os, threading
def f(a, c):
    out = {}
    x, y = open("l", "w"), open("l", "w")
    fcntl.flock(x, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        fcntl.flock(y, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out["second_flock"] = "granted"
    except OSError as e:
        out["second_flock"] = "errno %d" % e.errno
    lk = threading.Lock()
    lk.acquire()
    out["thread_lock_excludes"] = not lk.acquire(blocking=False)
    return out
'''


@needs_sandbox
def test_advisory_locks_are_no_ops_inside_the_sandbox_and_threads_still_exclude():
    """Documented semantic change (D-77 amendment): taking an advisory lock succeeds without effect, so two of a
    program's own descriptors do NOT exclude each other (a program that needs mutual exclusion between its own
    threads uses threading locks, which are unaffected; between processes it has none: no fork)."""
    r = run_program({"l.py": LOCKSEM}, "l:f", None)
    assert r["ok"] and r["result"] == {"second_flock": "granted", "thread_lock_excludes": True}, r
