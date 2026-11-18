"""Asynchronous self-play runtime: persistent logical workers sharing one dynamically batched policy.

* :class:`InferenceEngine` — one shared generation engine over the trainer's own policy object (no
  extra copy of Qwen in GPU memory). Workers submit requests independently; a scheduler groups the
  requests that are waiting into one batched forward pass (bounded queue = backpressure), returns the
  exact sampled token ids and their behaviour log-probs, and stamps every result with the served
  policy version.
* :class:`Collector` — N logical workers, each with its own task, isolated environment instance,
  history, budgets and timeouts. Environment steps run in a thread pool, so a worker running a test
  suite or waiting on a slow tool never blocks the others; workers pick up the next work unit as soon
  as they finish. Finished episodes are appended durably (fsync) as they complete.
* :mod:`agentrs.selfplay.bench` — end-to-end throughput benchmark (lock-step vs async, 1..32 workers).

The learning side is unchanged: episodes carry exact token ids, behaviour log-probs, sampling config
and policy version; the learner rejects samples from any other version (no stale off-policy updates).
"""
from agentrs.selfplay.engine import EngineStats, InferenceEngine  # noqa: F401
from agentrs.selfplay.workers import Collector, WorkUnit  # noqa: F401
