"""Runtime process for the kill -9 tests: connects to a LabHost over RPC and runs LiveRuntime.

    python rt_child.py <socket> <run_dir> <max_iterations> <summary.json>

Environment: RT_KILL_AFTER_EXECUTE=<n>  SIGKILL itself right after the n-th action executed in the world, before
                                         the runtime sees the result (an action in flight);
             RT_KILL_BEFORE_SAVE=<n>     SIGKILL itself when the learner is asked for its n-th checkpoint
                                         (results journalled, nothing committed).
"""
import json
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from cam_live.contracts import TaskSpec  # noqa: E402
from cam_live.harness.rpc import RpcClient  # noqa: E402
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig  # noqa: E402
from rt_helpers import LabHost  # noqa: E402
from rt_lab_agent import LabAgent  # noqa: E402


class Env:
    interface_version = "1.0"

    def __init__(self, rpc):
        self._rpc, self.n_exec = rpc, 0
        self.kill_at = int(os.environ.get("RT_KILL_AFTER_EXECUTE", "0"))

    def __getattr__(self, name):
        if name in LabHost.ENV:
            return lambda *a, **k: self._rpc.call(name, *a, **k)
        raise AttributeError(name)

    def poll_events(self, cursor):
        ev, cur = self._rpc.call("poll_events", cursor)
        return ev, cur

    def execute(self, action, authorization):
        res = self._rpc.call("execute", action, authorization)
        self.n_exec += 1
        if self.kill_at and self.n_exec == self.kill_at:
            os.kill(os.getpid(), signal.SIGKILL)              # the world ran it; this process never learns the result
        return res


class Prefixed:
    def __init__(self, rpc, prefix):
        self._rpc, self._prefix = rpc, prefix

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._rpc.call(self._prefix + name, *a, **k)


class Clock:
    def __init__(self, rpc):
        self._rpc = rpc

    def now(self):
        return self._rpc.call("clock_now")

    def advance(self, dt):
        return self._rpc.call("clock_advance", dt)


class KillingAgent(LabAgent):
    def save(self, path):
        n = self.calls.get("save", 0) + 1
        if n == int(os.environ.get("RT_KILL_BEFORE_SAVE", "0")):
            os.kill(os.getpid(), signal.SIGKILL)
        return super().save(path)


def main():
    sock, run_dir, max_iter, summary = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    rpc = RpcClient(sock)
    clock = Clock(rpc)
    task = TaskSpec.from_dict(json.load(open(os.path.join(os.path.dirname(run_dir), "task.json"))))
    agent = KillingAgent(clock, seed=7, only_ops=["lab.act", "wait"], wait_seconds=86400.0, durable=True)
    rt = LiveRuntime(agent, Env(rpc), Prefixed(rpc, "p_"), Prefixed(rpc, "v_"), task, run_dir, clock,
                     RuntimeConfig(fsync=True))
    status = rt.run(max_iterations=max_iter)
    json.dump({"status": status, "results": agent.results, "reconciled": agent.reconciled,
               "queries": agent.queries, "experience": [e["action_id"] for e in agent.experience],
               "loads": agent.calls.get("load", 0)}, open(summary, "w"))
    rt.close()


if __name__ == "__main__":
    main()
