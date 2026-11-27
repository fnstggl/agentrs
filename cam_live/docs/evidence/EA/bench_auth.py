"""Latency of the authorization path (CPU only). Usage:
    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/EA/bench_auth.py > cam_live/docs/evidence/EA/bench.json
"""
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
from cam_live import contracts as C  # noqa: E402
from cam_live.harness.auth import PolicyService, verify_audit_log  # noqa: E402
from cam_live.harness.auth.client import AuthClient  # noqa: E402
from cam_live.harness.auth.service import default_roles, serve  # noqa: E402

LOOP = "http://127.0.0.1:8765"


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median_ms": round(statistics.median(xs) * 1e3, 3), "p95_ms": round(xs[int(0.95 * (len(xs) - 1))] * 1e3, 3),
            "max_ms": round(xs[-1] * 1e3, 3)}


def timed(fn, n):
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        out.append(time.perf_counter() - t0)
    return stats(out)


def bench(policy, n, label):
    t = C.TaskSpec(f"bench-{label}", "bench", mode="staging", limits=C.Limits(max_decisions=10**9))
    res = {}
    acts = [C.ActionSpec(f"a{i}", "http.get", "api", arguments={"url": f"{LOOP}/p{i}"}) for i in range(n)]
    decs = []
    res["authorize_allow"] = timed(lambda i: decs.append(policy.authorize(acts[i], t, {})), n)
    res["check_grant_consume"] = timed(lambda i: policy.check_grant(acts[i], decs[i]), n)
    op = C.PrimitiveOp("net.request", "api", {"method": "GET", "url": LOOP + "/x"})
    res["check_primitive"] = timed(lambda i: policy.check_primitive(op, acts[i]), n)
    res["record"] = timed(lambda i: policy.record(acts[i], C.ActionResult(acts[i].action_id, "ok")), n)
    bad = [C.ActionSpec(f"b{i}", "http.get", "api", arguments={"url": f"https://denied{i}.example/"}) for i in range(n)]
    res["authorize_deny"] = timed(lambda i: policy.authorize(bad[i], t, {}), n)
    prog = C.ActionSpec("prog", "skill.run", "skill", arguments={"skill_id": "s"},
                        steps=[C.ActionSpec(f"s{j}", "browser.navigate", "structured_browser", arguments={"url": f"{LOOP}/s{j}"})
                               for j in range(20)])
    res["authorize_program_20_steps"] = timed(lambda i: policy.authorize(prog, t, {}), max(20, n // 10))
    return res


def main():
    n = int(os.environ.get("N", "300"))
    out = {"n": n, "host": os.uname().release}
    for fsync in (True, False):
        d = tempfile.mkdtemp(prefix="ea-bench-")
        s = PolicyService(d, fsync=fsync)
        # rate limits are irrelevant here: lift them through a policy copy
        import yaml
        from cam_live.harness.auth import DEFAULT_POLICY_PATH
        doc = yaml.safe_load(open(DEFAULT_POLICY_PATH))
        for k in ("default", "write", "primitive"):
            doc["rate_limits"][k] = {"count": 10**9, "window_s": 60}
        doc["limits"]["max_primitives_per_action"] = 10**6
        p = os.path.join(d, "bench_policy.yaml")
        yaml.safe_dump(doc, open(p, "w"))
        s.reload_policy("bench", p)
        key = "inproc_fsync" if fsync else "inproc_nofsync"
        out[key] = bench(s, n, key)
        t0 = time.perf_counter()
        v = verify_audit_log(s.audit.path, key=s.audit.key, head_path=s.audit.head_path)
        dt = time.perf_counter() - t0
        out[key]["audit_verify"] = {"entries": v.entries, "ok": v.ok, "seconds": round(dt, 4),
                                    "entries_per_s": round(v.entries / dt)}
        s.close()
    d = tempfile.mkdtemp(prefix="ea-bench-")
    sock = os.path.join(d, "sock", "auth.sock")
    server = serve(d, sock, None, roles=default_roles(single_uid=True), fsync=True)
    c = AuthClient(sock)
    t = C.TaskSpec("bench-rpc", "bench", mode="staging", limits=C.Limits(max_decisions=10**9))
    waits = [C.ActionSpec(f"w{i}", "wait", "internal") for i in range(n)]
    out["rpc_fsync"] = {"authorize_allow_internal": timed(lambda i: c.authorize(waits[i], t, {}), n),
                        "stopped": timed(lambda i: c.stopped(), n)}
    server.target.policy.close()
    server.stop()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
