"""Measure latency and observation sizes of the structured browser on the staging sites (workstream EB evidence).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/EB/measure_latency.py
"""
import json
import os
import statistics
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests", "harness", "env"))

from eb_support import Driver, GrantPolicy  # noqa: E402
from cam_live.contracts import TaskSpec  # noqa: E402
from cam_live.harness.browser import BrowserExecutor  # noqa: E402
from cam_live.harness.env import ComputerEnvironment  # noqa: E402
from cam_live.harness.internal import InternalExecutor  # noqa: E402
from cam_live.harness.service import RemoteEnvironment, serve  # noqa: E402
from cam_live.harness.staging import StagingSites  # noqa: E402

N = 15


def med(fn):
    xs = []
    for _ in range(N):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(xs), 1), round(max(xs), 1)


def main():
    run_dir = tempfile.mkdtemp(prefix="ebm")
    out = {"n": N, "unit": "ms (median, max)"}
    with StagingSites() as sites:
        policy = GrantPolicy()
        env = ComputerEnvironment(run_dir, policy)
        env.register(InternalExecutor(run_dir))
        env.register(BrowserExecutor(run_dir))
        d = Driver(env, policy, TaskSpec("obj_measure", "measure", mode="staging"))
        s = sites.urls["structured"]
        t = time.perf_counter()
        first = d.ok("browser.navigate", url=s + "/items")
        out["first_action_including_chromium_launch_ms"] = round((time.perf_counter() - t) * 1000, 1)
        out["navigate"] = med(lambda: d.ok("browser.navigate", url=s + "/items"))
        out["inspect_full_with_a11y"] = med(lambda: d.ok("browser.inspect"))
        out["inspect_diff_unchanged"] = med(lambda: d.ok("browser.inspect", full=False))
        out["fill"] = med(lambda: d.ok("browser.fill", role="textbox", name="Filter by name", text="abc"))
        out["scroll"] = med(lambda: d.ok("browser.scroll", dy=10))
        out["click_with_navigation"] = med(lambda: d.ok("browser.click", role="link", name="Items"))
        out["wait_schedule"] = med(lambda: d.ok("wait", seconds=1000))
        full = d.ok("browser.inspect").observations[0]
        same = d.ok("browser.inspect", full=False).observations[0]
        changed = d.ok("browser.fill", role="textbox", name="Filter by name", text="xyz").observations[0]
        out["observation_chars"] = {"full_text": len(full.text), "full_json": len(json.dumps(full.to_dict())),
                                    "diff_unchanged_text": len(same.text), "diff_after_fill_text": len(changed.text),
                                    "elements": len(full.data["elements"]), "quantities": len(full.quantities)}
        sock = os.path.join(run_dir, "sock", "exec.sock")
        server = serve(env, sock, group=None)
        remote = RemoteEnvironment(sock)
        rd = Driver(remote, policy, d.task)
        out["rpc_ping"] = med(remote.ping)
        out["rpc_inspect_full_with_a11y"] = med(lambda: rd.ok("browser.inspect"))
        remote.close()
        server.stop()
        rs = env.get_resource_state()
        out["totals"] = {"actions": rs["actions"], "primitive_ops": rs["primitive_ops"],
                         "primitive_checks": rs["primitive_checks"], "seconds": round(rs["seconds"], 2)}
        env.close()
    path = os.path.join(os.path.dirname(__file__), "latency.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
