"""The trusted env hub: worlds and evaluators for learners that run somewhere else. PROTECTED.

    python -m superlearner.world.hub --socket runs/env.sock [--eval-tasks FILE] [--core JSON]

Every connection gets its own forked server process (`world.core.serve`). A learner may start tasks of the
``train`` and ``curriculum`` splits by naming them; held-out tasks (``--eval-tasks``: a JSON list, or the sealed
final list) can only be started by index, and the learner is never told their seeds. Rewards and success flags are
computed here. This is the only place a sandboxed candidate run can get experience or scores from.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from multiprocessing.connection import Listener

from .core import _serve_entry
from .tasks import TaskSpec

AUTHKEY = b"superlearner-env"


def serve_hub(socket_path: str, eval_tasks: list | None = None, cfg: dict | None = None, eval_cfg: dict | None = None,
              allow=("train", "curriculum"), ready=None, results_path: str | None = None, budget: dict | None = None) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    os.makedirs(os.path.dirname(os.path.abspath(socket_path)) or ".", exist_ok=True)
    ctx = mp.get_context("fork")
    shared = {"train_decisions": ctx.Value("q", 0), "train_episodes": ctx.Value("q", 0)}
    children = []
    with Listener(socket_path, family="AF_UNIX", authkey=AUTHKEY) as lst:
        os.chmod(socket_path, 0o666)
        if ready is not None:
            ready.set()
        while True:
            try:
                conn = lst.accept()
            except (OSError, EOFError):
                continue
            p = ctx.Process(target=_serve_entry, args=(conn, tuple(allow), eval_tasks, cfg, eval_cfg, results_path,
                                                       shared, budget), daemon=True)
            p.start()
            conn.close()
            children = [c for c in children if c.is_alive()] + [p]


def start_hub(socket_path: str, eval_tasks: list | None = None, cfg: dict | None = None, eval_cfg: dict | None = None,
              results_path: str | None = None, budget: dict | None = None):
    """Start the hub in a child process; returns the process once the socket is accepting."""
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    p = ctx.Process(target=serve_hub, args=(socket_path, eval_tasks, cfg, eval_cfg, ("train", "curriculum"), ready,
                                            results_path, budget))
    p.start()
    if not ready.wait(30):
        p.terminate()
        raise RuntimeError("env hub did not start")
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--eval-tasks")
    ap.add_argument("--core", default="{}")
    a = ap.parse_args(argv)
    tasks = None
    if a.eval_tasks:
        d = json.load(open(a.eval_tasks))
        tasks = [TaskSpec.from_dict(t) for t in (d["tasks"] if isinstance(d, dict) else d)]
    serve_hub(a.socket, tasks, json.loads(a.core))


if __name__ == "__main__":
    main()
