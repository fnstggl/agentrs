"""Operating modes. They are never conflated, and only Mode A is used for training.

* **Mode A - synthetic training (default).** Everything happens inside a world: the browser resolves only
  ``*.crucible`` names to the local front end and blocks every other request; sandboxed programs have no network
  (Landlock TCP rules + seccomp); no tool reaches outside the kernel. Nothing here can be configured to open the
  real internet for an autonomous agent.
* **Mode B - offline grounding.** Generators may read *local, read-only* reference material (e.g. a JSON corpus of
  website patterns or business processes obtained lawfully beforehand) to improve fidelity. `load_reference`
  refuses network locations and never writes.
* **Mode C - supervised real-world evaluation.** Disabled unless the operator sets ``CRUCIBLE_MODE_C_APPROVAL`` to
  an approval token issued for a named evaluation, and every side effect is confirmed by a human approval callback.
  `RealWorldGateway` is the only place such calls could pass; by default it refuses everything. No trained policy
  can enable it from inside a world (worlds have no access to the environment variables or this object).
"""
from __future__ import annotations

import enum
import json
import os
from typing import Any, Callable


class Mode(enum.Enum):
    SYNTHETIC = "A"
    GROUNDING = "B"
    SUPERVISED_REAL = "C"


class ModeError(PermissionError):
    pass


def load_reference(path: str) -> Any:
    """Mode B: read a local, read-only reference file (JSON or text). Network locations are refused."""
    if "://" in path or path.startswith("//"):
        raise ModeError("offline grounding reads local files only (no network locations)")
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise ModeError(f"no such reference file: {path}")
    with open(real, encoding="utf-8") as f:
        text = f.read()
    return json.loads(text) if real.endswith(".json") else text


class RealWorldGateway:
    """Mode C: the single choke point for any action with an effect outside the simulator. Refuses unless an
    approval token is configured *and* the human approval callback confirms this specific action."""

    def __init__(self, evaluation: str, approve: Callable[[dict], bool] | None = None):
        self.evaluation = evaluation
        self.approve = approve
        self.log: list[dict] = []

    def enabled(self) -> bool:
        tok = os.environ.get("CRUCIBLE_MODE_C_APPROVAL", "")
        return bool(tok) and tok.startswith(f"{self.evaluation}:") and self.approve is not None

    def request(self, action: dict) -> dict:
        rec = {"evaluation": self.evaluation, "action": action}
        if not self.enabled():
            rec["outcome"] = "refused: Mode C is not enabled for this evaluation"
            self.log.append(rec)
            raise ModeError(rec["outcome"])
        if not self.approve(action):
            rec["outcome"] = "refused by the human approver"
            self.log.append(rec)
            raise ModeError(rec["outcome"])
        rec["outcome"] = "approved (the caller performs the action; this gateway only gates it)"
        self.log.append(rec)
        return rec
