"""G-rt: runtime child process for the kill -9 probes (p02_crash.py starts it; not meant to be run by hand).

    crash_child.py <socket> <run_dir> <max_iterations> <summary.json>
    G_KILL=<point>:<n>   SIGKILL this process at the n-th occurrence of <point> (see hit() calls below)
"""
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import StandInAgent  # noqa: E402

from cam_live.contracts import RunFiles, TaskSpec  # noqa: E402
from cam_live.harness.rpc import RpcClient  # noqa: E402
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig  # noqa: E402

KP, KN = (os.environ.get("G_KILL") or ":0").split(":")
KN = int(KN)
COUNT = {}


def hit(point):
    COUNT[point] = COUNT.get(point, 0) + 1
    if point == KP and COUNT[point] == KN:
        os.kill(os.getpid(), signal.SIGKILL)


class Env:
    interface_version = "1.0"
    NAMES = ("observe", "discover_tools", "validate_action", "poll_events", "get_resource_state",
             "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure")

    def __init__(self, rpc):
        self._rpc = rpc

    def __getattr__(self, name):
        if name in self.NAMES:
            return lambda *a, **k: self._rpc.call(name, *a, **k)
        raise AttributeError(name)

    def execute(self, action, authorization):
        hit("before_execute")                      # WAL begin written, nothing sent to the world
        res = self._rpc.call("execute", action, authorization)
        hit("after_execute")                       # the world ran it; the runtime never sees the result
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


def main():
    sock, run_dir, max_iter, summary = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if os.environ.get("G_FSIZE"):                  # cheap "disk full": no file of this process may grow past N bytes
        import resource
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        n = int(os.environ["G_FSIZE"])
        resource.setrlimit(resource.RLIMIT_FSIZE, (n, n))
    rpc = RpcClient(sock)
    clock = Clock(rpc)
    task = TaskSpec.from_dict(json.load(open(os.path.join(os.path.dirname(run_dir), "task.json"))))
    script = json.loads(os.environ.get("G_SCRIPT", '["lab.act", "lab.probe", "wait"]'))
    agent = StandInAgent(clock, script=script, wait_seconds=float(os.environ.get("G_WAIT", "172800")), durable=True)
    cfg = RuntimeConfig(fsync=True, checkpoint_every_s=float(os.environ.get("G_CKPT_EVERY", "0")))
    rt = LiveRuntime(agent, Env(rpc), Prefixed(rpc, "p_"), Prefixed(rpc, "v_"), task, run_dir, clock, cfg)

    # ---- kill points inside the runtime ----
    o_dec = rt.log.decision
    rt.log.decision = lambda *a, **k: (o_dec(*a, **k), hit("after_decision_log"))[0]
    o_ev = rt.log.event

    def ev(type_, data=None, **k):
        if type_ == "action_result" and KP == "torn_event":
            COUNT["torn_event"] = COUNT.get("torn_event", 0) + 1
            if COUNT["torn_event"] == KN:
                rt.log.sync()
                with open(os.path.join(run_dir, RunFiles.EVENTS), "ab") as f:
                    f.write(b'{"rid": "ev-torn", "seq": 99999999, "type": "action_result", "data": {"status": "o')
                    f.flush()
                    os.fsync(f.fileno())
                os.kill(os.getpid(), signal.SIGKILL)
        rid = o_ev(type_, data, **k)
        hit("after_" + type_ + "_log")             # after_action_result_log, after_outcome_log, after_checkpoint_log ...
        return rid
    rt.log.event = ev
    o_wal = rt._wal_write

    def wal(entry):
        t, kind = entry.get("t"), entry.get("kind")
        if t == "end" and KP == "torn_wal_end":
            COUNT["torn_wal_end"] = COUNT.get("torn_wal_end", 0) + 1
            if COUNT["torn_wal_end"] == KN:
                with open(os.path.join(run_dir, "wal.jsonl"), "ab") as f:
                    f.write(b'{"t": "end", "kind": "execute", "action_id": "' + entry["action_id"].encode() + b'", "resu')
                    f.flush()
                    os.fsync(f.fileno())
                os.kill(os.getpid(), signal.SIGKILL)
        if t == "begin" and kind == "execute" and KP == "torn_wal_begin":
            COUNT["torn_wal_begin"] = COUNT.get("torn_wal_begin", 0) + 1
            if COUNT["torn_wal_begin"] == KN:      # the process dies while writing the begin entry; nothing executed
                with open(os.path.join(run_dir, "wal.jsonl"), "ab") as f:
                    f.write(b'{"t": "begin", "kind": "execute", "action_id": "' + entry["action_id"].encode() + b'", "deci')
                    f.flush()
                    os.fsync(f.fileno())
                os.kill(os.getpid(), signal.SIGKILL)
        if t == "end":
            hit("before_wal_end")                  # the result is known to the process, nowhere on disk
        o_wal(entry)
        if t == "begin" and kind == "execute":
            hit("after_wal_begin")
        if t == "end" and kind == "execute":
            hit("after_wal_end")                   # result journalled; not yet in events.jsonl / given to the learner
    rt._wal_write = wal
    o_reset = rt._wal_reset

    def reset():
        hit("before_wal_reset")                    # state.json committed, journal not yet emptied
        o_reset()
        hit("after_wal_reset")
    rt._wal_reset = reset

    o_sleep = rt.sleeper.sleep_until

    def sleep_until(t, interrupted):
        if rt._waiting is not None:
            hit("in_wait_sleep")                   # asleep inside a `wait` decision (state committed, journal emptied)
        else:
            hit("in_idle_sleep")
        return o_sleep(t, interrupted)
    rt.sleeper.sleep_until = sleep_until

    # ---- kill points inside the learner ----
    agent.hooks["record_result"] = lambda *a: hit("in_record_result")
    agent.hooks["learn"] = lambda *a: hit("in_learn")
    agent.hooks["reconcile"] = lambda *a: hit("in_reconcile")
    agent.hooks["meta_learn"] = lambda *a: hit("in_meta_learn")
    o_save = agent.save

    def save(path):
        if KP == "in_save_partial":
            COUNT["in_save_partial"] = COUNT.get("in_save_partial", 0) + 1
            if COUNT["in_save_partial"] == KN:
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "agent.json"), "w") as f:      # a half-written checkpoint file
                    f.write('{"id": "g-partial", "state": {"n_sel')
                    f.flush()
                    os.fsync(f.fileno())
                os.kill(os.getpid(), signal.SIGKILL)
        cid = o_save(path)
        hit("after_save")                          # checkpoint complete, state.json not yet pointing at it
        return cid
    agent.save = save

    status = rt.run(max_iterations=max_iter)
    json.dump({"status": status, "results": agent.results, "reconciled": agent.reconciled, "query_of": agent.query_of,
               "n_select": agent.n_select, "loads": agent.trace.count("load"), "version": agent.version,
               "counters": rt.counters}, open(summary, "w"))
    rt.close()


if __name__ == "__main__":
    main()
