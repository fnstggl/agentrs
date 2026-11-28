"""G3-act probes of ST's agent-side changes (debug scale, CPU). A failing test = the property ST claims is violated.

Run: PYTHONPATH=<tree>/cam_live/src .venv/bin/python -m pytest <this file> -q --rootdir=<tree>/cam_live -p no:cacheprovider
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "integration"))

from cam_live.contracts import ActionSpec, Modality  # noqa: E402
from test_st_hotfix import _agent, _tools  # noqa: E402


def test_an_all_unauthorized_batch_is_not_offered(tmp_path):
    """ST hotfix claim: tools outside resources.authorized_tools are 'neither listed in the prompt nor offered'.
    agent._feasible drops unauthorized candidates only when at least one authorized candidate remains
    (``if unauth and len(unauth) < len(cands)``, agent.py:1490); a batch that is entirely unauthorized is kept."""
    agent = _agent(tmp_path)
    try:
        cands = [ActionSpec(action_id="x1", operation="http.request", modality=Modality.API,
                            arguments={"url": "https://x.example/"})]
        invalid = {}
        kept = agent._feasible(cands, _tools(), agent.clock.now(), invalid)
        assert not [a for a in kept if a.operation == "http.request"], (
            f"VIOLATED: an all-unauthorized batch is kept and offered: {[a.operation for a in kept]}, invalid={invalid}")
    finally:
        agent.close()
