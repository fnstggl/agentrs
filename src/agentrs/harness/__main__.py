"""Process entry points (one per container in the Compose / Kubernetes deployment).

    python -m agentrs.harness migrate          # apply database migrations
    python -m agentrs.harness control          # control API + dashboard (uvicorn)
    python -m agentrs.harness inference        # shared inference gateway (mock or local policy)
    python -m agentrs.harness lab              # SocialWebLab sandbox (self-hosted test application)
    python -m agentrs.harness outcome-worker   # durable delayed-outcome follow-ups + verification
    python -m agentrs.harness agent-worker     # agent runtimes for queued sandbox tasks
All processes log JSON to stdout, expose /health where they serve HTTP, and stop gracefully on SIGTERM.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time


def _stop_event() -> asyncio.Event:
    ev = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, ev.set)
        except NotImplementedError:  # pragma: no cover
            pass
    return ev


def main() -> None:
    from agentrs.harness.monitoring import setup_logging

    setup_logging(os.environ.get("AGENTRS_LOG_LEVEL", "INFO"))
    ap = argparse.ArgumentParser(prog="agentrs.harness")
    ap.add_argument("cmd", choices=["migrate", "control", "inference", "lab", "outcome-worker", "agent-worker"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default=os.environ.get("AGENTRS_INFERENCE_BACKEND", "mock"))
    ap.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args()
    import uvicorn

    if a.cmd == "migrate":
        from agentrs.harness.db import migrate

        migrate(os.environ["AGENTRS_DATABASE_URL"])
        print("migrations applied")
    elif a.cmd == "control":
        from agentrs.harness.app import from_env
        from agentrs.harness.control_api import create_control_app

        h = from_env()
        uvicorn.run(create_control_app(h, os.environ["AGENTRS_OPERATOR_TOKEN"]), host=a.host, port=a.port,
                    timeout_graceful_shutdown=20)
    elif a.cmd == "inference":
        from agentrs.harness.inference import BatchingInferenceServer, MockBackend, create_inference_app

        if a.backend == "mock":
            be = MockBackend()
        else:
            from agentrs.content.policy import ContentGeneratingPolicy, ContentGenConfig
            from agentrs.training.common import load_model

            path = os.environ["AGENTRS_POLICY_CHECKPOINT"]
            from agentrs.harness.inference import LocalPolicyBackend

            be = LocalPolicyBackend({"production": ContentGeneratingPolicy(load_model(path), ContentGenConfig())},
                                    "production")
        srv = BatchingInferenceServer(be)
        uvicorn.run(create_inference_app(srv, os.environ.get("AGENTRS_OPERATOR_TOKEN")), host=a.host, port=a.port)
    elif a.cmd == "lab":
        from agentrs.socialweblab.engine import SocialWebLab
        from agentrs.socialweblab.server import SocialWebLabServer

        lab = SocialWebLab(os.environ.get("AGENTRS_LAB_DB", "/data/lab.db"), seed=int(os.environ.get("AGENTRS_LAB_SEED", 0)),
                           n_users=int(os.environ.get("AGENTRS_LAB_USERS", 60)), clock=os.environ.get("AGENTRS_LAB_CLOCK", "real"))
        srv = SocialWebLabServer(lab, host=a.host, port=a.port)
        srv.start()
        lab.start_worker()
        print(f"SocialWebLab (simulated users) at {srv.url}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            srv.stop()
            lab.close()
    elif a.cmd == "outcome-worker":
        asyncio.run(_outcome_worker(a.poll_s))
    elif a.cmd == "agent-worker":
        asyncio.run(_agent_worker(a.poll_s))


async def _outcome_worker(poll_s: float) -> None:
    from agentrs.harness.app import from_env

    h = from_env()
    stop = _stop_event()
    while not stop.is_set():
        fetchers = {}
        for acc in h.accounts.list():
            if acc.status == "active" and h.registry.enabled(acc.provider):
                c = h.gateway.connector_for(acc)
                fetchers[f"harness:{acc.provider}"] = c.fetch_outcome
        await h.outcomes.run_due(fetchers, {"real": time.time()})
        h.outcomes.close_windows({"real": time.time()})
        h.verification.run()
        try:
            await asyncio.wait_for(stop.wait(), poll_s)
        except asyncio.TimeoutError:
            pass


async def _agent_worker(poll_s: float) -> None:
    """Runs agents from config/agents.yaml on queued tasks whose environment has an adapter."""
    from agentrs.harness.app import from_env
    from agentrs.harness.inference import InferenceClient
    from agentrs.harness.runtime import AgentRuntime

    h = from_env()
    client = InferenceClient(base_url=os.environ.get("AGENTRS_INFERENCE_URL", "http://inference:8100"))
    adapters = {}  # environment adapters are registered by deployment-specific code (see scripts/harness/)
    stop = _stop_event()
    rts = [AgentRuntime(a.id, h.gateway, client, h.tasks, h.provisioning, h.verification, adapters)
           for a in h.agents.all() if a.status == "active"]
    while not stop.is_set():
        for rt in rts:
            if adapters:
                await rt.loop(1, environments=list(adapters))
        try:
            await asyncio.wait_for(stop.wait(), poll_s)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    main()
