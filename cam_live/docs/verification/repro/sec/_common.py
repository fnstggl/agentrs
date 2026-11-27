"""Shared setup for the G-sec probes (independent verification, lens "sec").

Run every probe from /workspace/camlive:
    .venv/bin/python cam_live/docs/verification/repro/sec/<probe>.py
Probes use throw-away run directories below SCRATCH and never touch the source tree.
Exit status: 0 = the probed property HOLDS (or the probe is informational), 1 = DEFECT REPRODUCED.
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))          # cam_live/
SRC = os.path.join(ROOT, "src")
TESTS_AUTH = os.path.join(ROOT, "tests", "harness", "auth")
for p in (SRC, TESTS_AUTH):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRATCH = os.environ.get("G_SEC_SCRATCH",
                         "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/g_sec")
os.makedirs(SCRATCH, exist_ok=True)

from cam_live import contracts as C  # noqa: E402

ALLOW, DENY, NEEDS = C.AuthVerdict.ALLOW, C.AuthVerdict.DENY, C.AuthVerdict.NEEDS_APPROVAL


def fresh_dir(tag: str, world: bool = False) -> str:
    """A new empty directory. ``world=True`` puts it under /tmp with mode 0755 so other uids can traverse it."""
    if world:
        d = tempfile.mkdtemp(prefix=f"g_sec_{tag}_", dir="/tmp")
        os.chmod(d, 0o755)
        return d
    return tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH)


def cleanup(d: str) -> None:
    shutil.rmtree(d, ignore_errors=True)


class Report:
    """Collects findings; prints one line each; exit code 1 when any defect reproduced."""

    def __init__(self, name: str):
        self.name, self.defects, self.holds = name, [], []

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
