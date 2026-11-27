"""Child process of the integration recovery tests.

    python int_child.py decide <agent_run_dir> <checkpoint> <env.pkl> <n> <out.json>
        a fresh process: build the agent, load the checkpoint, continue the saved environment for n decisions
    python int_child.py runtime <socket> <run_dir> <max_iterations> <summary.json>
        LiveRuntime + CamLiveAgent against a LabWorld host served over RPC (the world outlives this process).
        RT_KILL_AFTER_EXECUTE=<n>: SIGKILL this process right after the n-th action executed in the world.
"""
import json
import os
import pickle
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from cam_live.agent import CamLiveAgent  # noqa: E402
from cam_live.contracts import TaskSpec  # noqa: E402
from int_helpers import agent_config, drive  # noqa: E402


def warm(root, n_warm, n_after, out):
    """Build an agent, run ``n_warm`` decisions, save, keep a copy of the learner directory and of the environment
    as of the save, then continue ``n_after`` decisions (the uninterrupted agent)."""
    import shutil
    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
    from int_helpers import DriverState, lab_config
    clock = SimClock()
    world = LabWorld(lab_config(51), clock)
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(seed=5), os.path.join(root, "a"), clock=clock, verifier=verifier)
    agent.set_objective(lab_task(world, "obj_saveload"))
    state = DriverState()
    before = drive(agent, world, clock, verifier, state, int(n_warm))
    cid = agent.save(os.path.join(root, "ckpt"))
    shutil.copytree(os.path.join(root, "a"), os.path.join(root, "b"))
    with open(os.path.join(root, "env.pkl"), "wb") as f:
        pickle.dump((world, clock, verifier, state), f)
    after = drive(agent, world, clock, verifier, state, int(n_after))
    json.dump({"warm": before, "decisions": after, "checkpoint": cid}, open(out, "w"))
    agent.close()


def decide(run_dir, ckpt, env_pkl, n, out):
    with open(env_pkl, "rb") as f:
        world, clock, verifier, state = pickle.load(f)
    agent = CamLiveAgent(agent_config(seed=5), run_dir, clock=clock, verifier=verifier)
    agent.load(ckpt)
    res = drive(agent, world, clock, verifier, state, int(n))
    json.dump({"decisions": res, "loaded_checkpoint": agent.checkpoint_id}, open(out, "w"))
    agent.close()


class Remote:
    def __init__(self, rpc, prefix="", kill_at=0):
        self._rpc, self._prefix, self._kill_at, self.n_exec = rpc, prefix, kill_at, 0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._rpc.call(self._prefix + name, *a, **k)


class RemoteEnv(Remote):
    interface_version = "1.0"

    def execute(self, action, authorization):
        res = self._rpc.call("execute", action, authorization)
        self.n_exec += 1
        if self._kill_at and self.n_exec == self._kill_at:
            os.kill(os.getpid(), signal.SIGKILL)          # the world ran it; this process never sees the result
        return res


class RemoteClock(Remote):
    def now(self):
        return self._rpc.call("clock_now")

    def advance(self, dt):
        return self._rpc.call("clock_advance", dt)


def runtime(sock, run_dir, max_iter, summary):
    from cam_live.harness.auth import PolicyService
    from cam_live.harness.rpc import RpcClient
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    rpc = RpcClient(sock)
    clock = RemoteClock(rpc)
    task = TaskSpec.from_dict(json.load(open(os.path.join(os.path.dirname(run_dir), "task.json"))))
    verifier = Remote(rpc, "v_")
    agent = CamLiveAgent(agent_config(seed=7, durable=True), os.path.join(run_dir, "agent"), clock=clock,
                         verifier=verifier)
    policy = PolicyService(os.path.join(run_dir, "auth_state"), clock=clock)
    policy.register_task(task, "operator")
    env = RemoteEnv(rpc, kill_at=int(os.environ.get("RT_KILL_AFTER_EXECUTE", "0")))
    rt = LiveRuntime(agent, env, policy, verifier, task, run_dir, clock, RuntimeConfig(fsync=True, candidates_k=8))
    status = rt.run(max_iterations=int(max_iter))
    recs = agent.store.all()
    json.dump({"status": status, "records": [[r.decision.chosen.action_id, r.result.status if r.result else ""]
                                            for r in recs],
               "decisions": rt.counters["decisions"], "loaded": agent.checkpoint_id}, open(summary, "w"))
    rt.close()
    agent.close()


if __name__ == "__main__":
    {"warm": warm, "decide": decide, "runtime": runtime}[sys.argv[1]](*sys.argv[2:])
