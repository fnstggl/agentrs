import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ACCEPTANCE_DIR = os.path.join(ROOT, "docs", "experiments", "acceptance")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item._crucible_call = rep


class _Acceptance:
    """What an acceptance test measured. A test states its claim, records the numbers it observed and any caveat;
    with CRUCIBLE_RECORD_ACCEPTANCE=1 (set by examples/run_acceptance.py) the record is written to
    docs/experiments/acceptance/<letter>.json together with the outcome pytest reported - never by the test itself,
    so a failing test is recorded as failing."""

    def __init__(self, nodeid: str):
        self.rec = {"test": nodeid, "claim": "", "measured": {}, "caveats": [], "logs": {},
                    "reproduce": f"cd crucible && python -m pytest -q '{nodeid}'"}

    def claim(self, text: str) -> None:
        self.rec["claim"] = " ".join(text.split())

    def measure(self, **kw) -> None:
        self.rec["measured"].update(kw)

    def caveat(self, text: str) -> None:
        self.rec["caveats"].append(" ".join(text.split()))

    def log(self, name: str, value) -> None:
        self.rec["logs"][name] = value


@pytest.fixture
def acceptance(request):
    a = _Acceptance(request.node.nodeid)
    t0 = time.time()
    yield a
    if os.environ.get("CRUCIBLE_RECORD_ACCEPTANCE") != "1":
        return
    rep = getattr(request.node, "_crucible_call", None)
    a.rec["outcome"] = rep.outcome if rep is not None else "unknown"
    if rep is not None and rep.failed:
        a.rec["failure"] = str(rep.longrepr)[-4000:]
    a.rec["wall_s"] = round(time.time() - t0, 1)
    a.rec["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(ACCEPTANCE_DIR, exist_ok=True)
    name = request.node.name.replace("test_", "", 1)
    with open(os.path.join(ACCEPTANCE_DIR, f"{name}.json"), "w") as fh:
        json.dump(a.rec, fh, indent=1, default=str)
