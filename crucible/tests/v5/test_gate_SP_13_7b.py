"""SP-13.7b (MASTER_SPEC Part 13.7): hardware inspected, other tenants not interfered with, performance budgets set
before stress tests, no external compute purchased.

* budgets before stress tests: the machine-specific targets (docs/v5/performance/targets.json) are committed and
  adopted by DECISION_LOG entries citing each version's SHA-256 before this run, and the governor's stored budget
  (with the hardware inspection it was derived from) predates it;
* tenant interference: the governor never signals a process outside the job it started (adversarial probe: a sibling
  process survives the kill of a job that exceeds its RAM), and no governed job in the usage log ran over a limit
  without being stopped;
* external compute: no source file calls a cloud-compute or remote-inference API, and a world run under an audit
  hook opens no network connection to a non-loopback address."""
import ipaddress
import os
import re
import sys
import tempfile
import time

import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import govprobe, slots

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --gpu-mb 1500 --ram-mb 8000 --timeout 3600 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.7b")     # the network probe runs a population world
CLOUD = re.compile(r"\b(import\s+(boto3|botocore|runpod|modal|replicate|vastai|lambda_cloud)|from\s+(google\.cloud|azure|"
                   r"boto3|openai|anthropic)\s+import|api\.openai\.com|api\.anthropic\.com|ec2\.amazonaws|"
                   r"compute\.googleapis)")


def _network_probe() -> list[str]:
    outbound = []

    def hook(ev, args):
        if ev == "socket.connect" and len(args) > 1 and isinstance(args[1], tuple) and args[1]:
            host = str(args[1][0])
            try:
                if not ipaddress.ip_address(host).is_loopback:
                    outbound.append(host)
            except ValueError:
                if host not in ("localhost",):
                    outbound.append(host)
    sys.addaudithook(hook)
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World
    from crucible.population.worlds import population_world
    w = World.create(population_world("development", 5, n=200, sites=4))
    w.advance_time(6 * 3600)
    w.close()
    return outbound


def test_SP_13_7b_budgets_tenants_and_no_external_compute():
    ad = K.targets_adoption(K.ROOT)
    now = time.time()
    bpath = os.path.join(slots.slots_root(), "budget.json")
    import json
    bud = json.load(open(bpath)) if os.path.exists(bpath) else {}
    budgets_before = bool(ad["all_adopted"] and ad["current_committed"] and ad["first_commit_time"]
                          and ad["first_commit_time"] < now and bud.get("created", now) < now and bud.get("hardware"))
    probe = govprobe.run_all(tempfile.mkdtemp(prefix="sp137b-"), only=["foreign", "escape"])
    unstopped = [r for r in slots.usage(slots.slots_root()) if r.get("v") == 2
                 for v in r.get("violations") or () if v.get("action") not in ("killed", "repinned", "observed")]
    incidents = len(unstopped) + sum(0 if probe[k]["contained"] else 1 for k in probe)
    src = os.path.join(K.ROOT, "src")
    hits = []
    for dp, _, fn in os.walk(src):
        for f in fn:
            if f.endswith(".py"):
                t = open(os.path.join(dp, f), errors="replace").read()
                if CLOUD.search(t):
                    hits.append(os.path.relpath(os.path.join(dp, f), src))
    outbound = _network_probe()
    m = {"budgets_recorded_before_stress_tests": budgets_before, "tenant_interference_incidents": incidents,
         "external_compute_purchases": len(hits) + len(outbound), "cloud_api_files": len(hits),
         "outbound_connections": len(outbound), "hardware_inspected": bool(bud.get("hardware")),
         "governor_budget_cpus": int((bud.get("budget") or {}).get("cpus", 0))}
    evidence.record("SP-13.7b", m, command=CMD, notes=f"cloud API files {hits}; outbound {outbound[:20]}; "
                                                      f"unstopped {unstopped[:10]}; probes {probe}")
    assert m["budgets_recorded_before_stress_tests"] and m["hardware_inspected"]
    assert m["tenant_interference_incidents"] == 0 and m["external_compute_purchases"] == 0
