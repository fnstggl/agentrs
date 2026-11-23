"""Scale layer of Crucible 4.0: many independent worlds at once, and measured performance.

* `crucible.scale.pool`    - `WorldPool`: process-parallel worlds run from declarative job specs, with isolation,
                             determinism, checkpoints, worker recycling and clean shutdown.
* `crucible.scale.rollout` - `RolloutFarm`: asynchronous rollout workers for learners that train while worlds run
                             (versioned policy parameters, backpressure, exact accounting).
* `crucible.scale.bench`   - the 4.0 benchmark suite (`python examples/bench_openworld.py`).

Importing this package imports neither numpy nor the kernel: worker processes import it before their BLAS thread
limits are set.
"""
from .pool import PoolBroken, Supervisor, WorldPool, ensure_registered, pid_budget, resolve, run_world_job, world_job
from .rollout import Rollout, RolloutFarm, WorldEnv, make_world_env

__all__ = ["WorldPool", "world_job", "run_world_job", "RolloutFarm", "Rollout", "WorldEnv", "make_world_env",
           "Supervisor", "PoolBroken", "resolve", "ensure_registered", "pid_budget"]
