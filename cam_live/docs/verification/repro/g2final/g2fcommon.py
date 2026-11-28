"""G2-final probe helpers (independent verifier; not CAM code). Run from /workspace/camlive:

    G2F_SRC=<git archive of fdab7d3>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/<probe>.py

``G2F_SRC`` selects the code under test (default: this tree's cam_live/src). Every probe prints [HOLDS ] / [BROKEN]
per check, writes <scratch>/<probe>.result.json and exits 1 when a property under attack is broken.
Scratch: $G2F_SCRATCH (default: the G2FINAL scratch directory of this session).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
TREE_PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.abspath(os.environ.get("G2F_SRC") or os.path.join(TREE_PROJECT, "src"))
PROJECT = os.path.dirname(SRC)                       # cam_live/ of the code under test (tasks/, tests/, configs/)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for _d in ("runtime",):
    _p = os.path.join(PROJECT, "tests", _d)
    if _p not in sys.path:
        sys.path.append(_p)
_rt = os.path.join(PROJECT, "docs", "verification", "repro", "rt")
if _rt not in sys.path:
    sys.path.append(_rt)

SCRATCH = os.environ.get("G2F_SCRATCH",
                         "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2FINAL/p")


def scratch(name: str) -> str:
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.mkdtemp(prefix=name + "_", dir=SCRATCH)


class Report:
    def __init__(self, name: str):
        self.name, self.rows = name, []
        print(f"== {name} on {SRC}", flush=True)

    def check(self, label: str, ok: bool, detail: Any = "") -> bool:
        self.rows.append({"check": label, "holds": bool(ok), "detail": detail})
        d = json.dumps(detail, default=str, ensure_ascii=False)
        print(f"[{'HOLDS ' if ok else 'BROKEN'}] {label}" + (f" :: {d[:700]}" if detail != "" else ""), flush=True)
        return ok

    def note(self, label: str, detail: Any = "") -> None:
        self.rows.append({"check": label, "holds": None, "detail": detail})
        print(f"[NOTE  ] {label} :: {json.dumps(detail, default=str, ensure_ascii=False)[:700]}", flush=True)

    def done(self) -> int:
        broken = [r for r in self.rows if r["holds"] is False]
        n = sum(1 for r in self.rows if r["holds"] is not None)
        os.makedirs(SCRATCH, exist_ok=True)
        out = os.path.join(SCRATCH, f"{self.name}.result.json")
        with open(out, "w") as f:
            json.dump({"probe": self.name, "src": SRC, "broken": len(broken), "checks": n, "rows": self.rows}, f,
                      indent=1, default=str, ensure_ascii=False)
        print(f"== {self.name}: {len(broken)} broken of {n} checks; details: {out}", flush=True)
        return 1 if broken else 0
