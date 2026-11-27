"""G-rt: agent factory for CLI probes (`--agent-factory g_factory:build_agent`). A verification stand-in, not CAM."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import StandInAgent  # noqa: E402


def build_agent(cfg, run_dir, clock):
    sec = cfg.section("g_standin") if hasattr(cfg, "section") else {}
    return StandInAgent(clock, script=sec.get("script", ["lab.act", "lab.probe", "wait"]),
                        wait_seconds=float(sec.get("wait_seconds", 86400.0)), durable=True)
