"""Shared setup for the G2-harness probes (independent re-verification at the code freeze 4d6044a).

Run every probe from /workspace/camlive:
    .venv/bin/python cam_live/docs/verification/repro/g2harness/<probe>.py
By default the probes import the code of this checkout (cam_live/src). To run them against an exact extract of the
freeze commit, set G2H_SRC to that tree's ``cam_live/src`` (``git archive 4d6044a | tar -x -C <dir>``).
Run directories go below G2H_SCRATCH (default: the verifier's scratch directory) or, where another uid must reach
them, below /tmp (removed at exit). Nothing outside this machine is contacted.
Exit status: 0 = every probed property HOLDS, 1 = at least one DEFECT reproduced. NOTE lines are informational.
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))          # cam_live/ of this checkout
SRC = os.environ.get("G2H_SRC") or os.path.join(ROOT, "src")
TESTS = os.path.join(os.path.dirname(SRC), "tests")
for p in (SRC, os.path.join(TESTS, "harness", "auth"), os.path.join(TESTS, "harness", "env")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRATCH = os.environ.get("G2H_SCRATCH",
                         "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2HARNESS/run")
os.makedirs(SCRATCH, exist_ok=True)

import cam_live  # noqa: E402
from cam_live import contracts as C  # noqa: E402

ALLOW, DENY, NEEDS = C.AuthVerdict.ALLOW, C.AuthVerdict.DENY, C.AuthVerdict.NEEDS_APPROVAL


def fresh_dir(tag: str, world: bool = False) -> str:
    if world:
        d = tempfile.mkdtemp(prefix=f"g2h_{tag}_", dir="/tmp")
        os.chmod(d, 0o755)
        return d
    return tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH)


def cleanup(d: str) -> None:
    shutil.rmtree(d, ignore_errors=True)


class Report:
    def __init__(self, name: str):
        self.name, self.defects, self.holds = name, [], []
        print(f"{name}: code under test = {os.path.dirname(cam_live.__file__)}")

    def hold(self, what: str, detail: str = "") -> None:
        self.holds.append(what)
        print(f"  HOLDS   {what}" + (f"  [{detail}]" if detail else ""))

    def defect(self, what: str, detail: str = "") -> None:
        self.defects.append(what)
        print(f"  DEFECT  {what}" + (f"  [{detail}]" if detail else ""))

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        (self.hold if ok else self.defect)(what, detail)
        return ok

    def note(self, what: str) -> None:
        print(f"  NOTE    {what}")

    def done(self) -> None:
        print(f"{self.name}: {len(self.holds)} properties hold, {len(self.defects)} defect(s) reproduced")
        sys.exit(1 if self.defects else 0)


def dec_str(d) -> str:
    return f"{d.verdict} {list(d.policy_ids)[:4]}"
