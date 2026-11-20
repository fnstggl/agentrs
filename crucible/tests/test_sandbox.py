"""Sandbox isolation tests against the real Landlock/seccomp sandbox (skipped where unavailable)."""
import pytest

from crucible.sandbox.runner import SandboxConfig, SandboxUnavailable, Worker, capabilities, run_program, run_tests

caps = capabilities()
pytestmark = [pytest.mark.sandbox,
              pytest.mark.skipif(caps["landlock_abi"] < 1 or not caps["seccomp"] or not caps["root"],
                                 reason=f"needs Landlock+seccomp and root to drop privileges: {caps}")]

PROBE = '''
import os
def probe(args, caller):
    out = {}
    def t(name, fn):
        try:
            out[name] = ["allowed", str(fn())[:40]]
        except BaseException as e:
            out[name] = ["blocked", type(e).__name__]
    t("uid", lambda: os.getuid())
    t("read_root", lambda: open("/root/.bashrc").read())
    t("list_workspace", lambda: os.listdir("/workspace"))
    t("read_etc_environment", lambda: open("/etc/environment").read())
    t("socket", lambda: __import__("socket").socket())
    t("subprocess", lambda: __import__("subprocess").run(["/bin/true"]))
    t("fork", lambda: os.fork())
    t("multiprocessing", lambda: __import__("multiprocessing").Process(target=print).start())
    t("write_tmp", lambda: open("/tmp/crucible-escape", "w").write("x"))
    t("write_workdir", lambda: open("ok.txt", "w").write("x"))
    t("import_torch", lambda: __import__("torch"))
    t("numpy", lambda: __import__("numpy").arange(4).sum())
    t("env", lambda: sorted(os.environ))
    def thr():
        import threading
        r = []
        th = threading.Thread(target=lambda: r.append(1)); th.start(); th.join()
        return r
    t("thread", thr)
    return out
def hog(args, caller):
    return len(bytearray(3 * 1024 ** 3))
def spin(args, caller):
    while True:
        pass
def boom(args, caller):
    raise ValueError("nope")
def world(args, caller):
    import crucible_sdk
    return crucible_sdk.call("bank.accounts")
'''


def test_isolation_matrix():
    r = run_program({"p.py": PROBE}, "p:probe")
    assert r["ok"], r
    o = r["result"]
    assert o["uid"] == ["allowed", "65534"]
    for k in ("read_root", "list_workspace", "read_etc_environment", "socket", "subprocess", "fork",
              "multiprocessing", "write_tmp", "import_torch"):
        assert o[k][0] == "blocked", (k, o[k])
    assert o["write_workdir"][0] == "allowed" and o["numpy"] == ["allowed", "6"] and o["thread"][0] == "allowed"
    assert "CONTAINER_API_KEY" not in o["env"][1] and "OPEN_BUTTON_TOKEN" not in o["env"][1]


def test_limits_and_failures():
    r = run_program({"p.py": PROBE}, "p:hog")
    assert not r["ok"] and r["error"]["type"] == "MemoryError"
    r = run_program({"p.py": PROBE}, "p:spin", timeout=1.0)
    assert not r["ok"] and r["error"]["type"] == "Timeout"
    r = run_program({"p.py": PROBE}, "p:boom")
    assert not r["ok"] and r["error"]["type"] == "ValueError" and "nope" in r["error"]["message"]


def test_brokered_world_calls_only():
    calls = []
    r = run_program({"p.py": PROBE}, "p:world", rpc=lambda tool, args: (calls.append(tool) or True, {"ok": 1}))
    assert r["ok"] and calls == ["bank.accounts"] and r["rpc_calls"] == 1
    r = run_program({"p.py": PROBE}, "p:world")       # no broker -> refused, not a network call
    assert not r["ok"] and "forbidden" in r["error"]["message"] or not r["ok"]


def test_persistent_worker_and_tests_runner():
    w = Worker({"s.py": "def f(a, c):\n    return a * 2\n"}).start()
    try:
        assert [w.run({"kind": "call", "target": "s:f", "args": i})["result"] for i in range(5)] == [0, 2, 4, 6, 8]
    finally:
        w.close()
    r = run_tests({"tests/test_x.py": "def test_a():\n    assert 1\ndef test_b():\n    assert 0, 'b'\n"})
    assert r["result"]["passed"] == 1 and r["result"]["failed"] == 1


def test_deterministic_line_accounting():
    cfg = SandboxConfig(count_lines=True)
    a = run_program({"s.py": "def f(n, c):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"}, "s:f", 1000, cfg=cfg)
    b = run_program({"s.py": "def f(n, c):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"}, "s:f", 1000, cfg=cfg)
    assert a["work_lines"] == b["work_lines"] and a["work_lines"] >= 2000


def test_unsafe_paths_rejected():
    with pytest.raises(ValueError):
        Worker({"../escape.py": "x"}).start()
