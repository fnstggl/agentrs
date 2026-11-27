"""cam_live.runtime.verifier_service — the independent verifier as its own process (VerifyRPC).

    python -m cam_live.runtime.verifier_service --run-dir R --socket R/sock/verify.sock \\
        --task tasks/discord_100.yaml --source <module>:<factory>

The process holds the signing key (in ``R/verifier/``, mode 0700, created by this uid) and the hash-chained log
(``R/verifier.jsonl``). It judges only the task file the operator started it with: a TaskSpec sent by a client
merely names the objective, so success criteria can never be supplied by the learner. The learner's process uses
:class:`RemoteVerifier`, which can ask for outcomes and for signature checks but never sees the key.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence

from ..contracts import (OutcomeObservation, OutcomeQuery, RunFiles, TaskSpec, VerifiedOutcome, WallClock)
from ..harness import SERVICE_GROUP, SOCK_VERIFY, uid_of
from ..harness.rpc import RpcClient, RpcServer
from .tasks import evidence_source, load_task
from .verifier import SignedVerifier, build_verifier, source_from_spec

METHODS = ("verify", "measure", "check_signature", "check_measurement", "info")


class VerifierEndpoint:
    """The allowlisted surface exposed over the socket."""

    def __init__(self, verifier: SignedVerifier, source_name: str, source_live: bool):
        self._v, self._source_name, self._source_live = verifier, source_name, source_live
        self._lock = threading.Lock()

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        with self._lock:
            return self._v.verify(task)

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        with self._lock:
            return self._v.measure(list(queries))

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return self._v.check_signature(outcome)

    def check_measurement(self, obs: OutcomeObservation) -> bool:
        return bool(self._v.check_measurement(obs))

    def info(self) -> Dict[str, Any]:
        pinned = self._v.pinned_task
        return {"objective_id": pinned.objective_id if pinned else "", "source": self._source_name,
                "source_live": self._source_live, "log_entries": self._v.log.seq + 1,
                "verifier": type(self._v).__name__}


class RemoteVerifier:
    """Client side of VerifyRPC; implements ``contracts.OutcomeVerifier``."""

    def __init__(self, socket_path: str, timeout: float = 120.0):
        self._rpc = RpcClient(socket_path, timeout)

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        return self._rpc.call("verify", task)

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return self._rpc.call("measure", list(queries))

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return bool(self._rpc.call("check_signature", outcome))

    def check_measurement(self, obs: OutcomeObservation) -> bool:
        return bool(self._rpc.call("check_measurement", obs))

    def info(self) -> Dict[str, Any]:
        return self._rpc.call("info")

    def close(self) -> None:
        self._rpc.close()


def serve(run_dir: str, socket_path: str, task: TaskSpec, source: Any, *, clock: Any = None,
          allowed_uids: Optional[set] = None) -> RpcServer:
    """Build the verifier for ``task`` and start serving it (returns the running server)."""
    verifier = build_verifier(task, os.path.join(run_dir, "verifier"), clock or WallClock(), source,
                              log_path=os.path.join(run_dir, RunFiles.VERIFIER), pin=True)
    endpoint = VerifierEndpoint(verifier, getattr(source, "name", type(source).__name__),
                                bool(getattr(source, "live", False)))
    uids = allowed_uids if allowed_uids is not None else {uid_of("learner"), os.getuid()}
    return RpcServer(socket_path, endpoint, METHODS, allowed_uids=uids, group=SERVICE_GROUP, name="verify").start()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.runtime.verifier_service", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", default="")
    ap.add_argument("--task", default="", help="operator task file (default: <run-dir>/task.yaml)")
    ap.add_argument("--source", default="", help="evidence source: jsonl:<path> | json:<path> | <module>:<factory>; "
                                                 "default: success.evidence_source of the task, or for a task with "
                                                 "Discord resources cam_live.harness.discord_membership:"
                                                 "membership_source")
    a = ap.parse_args(argv)
    run_dir = os.path.abspath(a.run_dir)
    task_path = a.task or os.path.join(run_dir, "task.yaml")
    task = load_task(task_path, for_run=True)
    spec = a.source or evidence_source(task)
    if ":" not in spec:
        print(f"verifier: no usable evidence source ({spec!r}); give --source", file=sys.stderr)
        return 2
    source = source_from_spec(spec, task, run_dir)
    if task.mode == "live" and not getattr(source, "live", False):
        print(f"verifier: task mode is live but evidence source {spec!r} is not a live source", file=sys.stderr)
        return 2
    server = serve(run_dir, a.socket or os.path.join(run_dir, "sock", SOCK_VERIFY), task, source)
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    print(f"verifier: serving objective {task.objective_id} on {server.path} (uid {os.getuid()})", flush=True)
    done.wait()
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
