"""G2-runtime probe helpers (independent verifier; not CAM code). Run probes from /workspace/camlive:

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/<probe>.py

Every probe prints [HOLDS ] / [BROKEN] per check, writes <scratch>/<probe>.result.json and exits 1 when a property
under attack is broken. Scratch: $G2RT_SCRATCH (default: the G2RUNTIME scratch directory of this session).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
TESTS_RT = os.path.join(PROJECT, "tests", "runtime")
if TESTS_RT not in sys.path:
    sys.path.append(TESTS_RT)

SCRATCH = os.environ.get("G2RT_SCRATCH",
                         "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2RUNTIME/p")


def scratch(name: str) -> str:
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.mkdtemp(prefix=name + "_", dir=SCRATCH)


class Report:
    def __init__(self, name: str):
        self.name, self.rows = name, []

    def check(self, label: str, ok: bool, detail: Any = "") -> bool:
        self.rows.append({"check": label, "holds": bool(ok), "detail": detail})
        d = json.dumps(detail, default=str)
        print(f"[{'HOLDS ' if ok else 'BROKEN'}] {label}" + (f" :: {d[:700]}" if detail != "" else ""), flush=True)
        return ok

    def note(self, label: str, detail: Any = "") -> None:
        self.rows.append({"check": label, "holds": None, "detail": detail})
        print(f"[NOTE  ] {label} :: {json.dumps(detail, default=str)[:700]}", flush=True)

    def done(self) -> int:
        broken = [r for r in self.rows if r["holds"] is False]
        n = sum(1 for r in self.rows if r["holds"] is not None)
        os.makedirs(SCRATCH, exist_ok=True)
        out = os.path.join(SCRATCH, f"{self.name}.result.json")
        with open(out, "w") as f:
            json.dump({"probe": self.name, "broken": len(broken), "checks": n, "rows": self.rows}, f, indent=1,
                      default=str)
        print(f"== {self.name}: {len(broken)} broken of {n} checks; details: {out}", flush=True)
        return 1 if broken else 0
