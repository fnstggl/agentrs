"""Run the red-team suite and write `docs/experiments/ow_redteam.json`.

    cd crucible && PYTHONPATH=src python examples/redteam_report.py [--fast] [--only SUBSTRING] [--out PATH]

Probes that deploy agent code need the real sandbox (Landlock + seccomp + privilege drop to `nobody`); where it
cannot start they are reported as `not_applicable`, never as held. `--fast` skips the slow probes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from crucible.redteam import all_probes, default_world_factory, run_probes, sandbox_available, table  # noqa: E402
from crucible.redteam.suite import KNOWN_OPEN  # noqa: E402
from crucible.redteam.toy import EXPECTED as TOY_EXPECTED, toy_probes, toy_world_factory  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip slow probes")
    ap.add_argument("--only", default="", help="run probes whose name contains this text")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "experiments", "ow_redteam.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    ok, why = sandbox_available()
    print(f"sandbox available: {ok} {why}", flush=True)
    probes = [p for p in all_probes(include_slow=not args.fast) if args.only in p.name]
    report = run_probes(default_world_factory, probes,
                        progress=lambda r: print(f"  [{r['wall_s']:7.1f}s] {r['result']:<19} {r['name']}", flush=True))
    sanity = run_probes(toy_world_factory, toy_probes())
    report["sanity_check"] = {
        "what": "the same framework run against deliberately vulnerable toys (redteam/toy.py); every toy must be flagged",
        "expected": TOY_EXPECTED, "observed": {r["name"]: r["result"] for r in sanity["probes"]},
        "passed": all(r["result"] == TOY_EXPECTED[r["name"]] for r in sanity["probes"])}
    found = {r["name"]: r["result"] for r in report["probes"] if r["name"] in {f["name"] for f in report["findings"]}}
    report["ratchet"] = {"known_open": KNOWN_OPEN,
                         "new_findings": {k: v for k, v in found.items() if k not in KNOWN_OPEN},
                         "fixed_since_listed": sorted(r["name"] for r in report["probes"]
                                                      if r["name"] in KNOWN_OPEN and r["result"] == "held")}
    report["environment"] = {"sandbox_available": ok, "sandbox_note": why, "wall_s": round(time.time() - t0, 1),
                             "world_factory": "crucible.redteam.default_world_factory (family `tiny`, or "
                                              "`open_digital` where a probe asks for the open world)"}
    report["how_to_read"] = ("held = the attack achieved nothing false; false_success = a goal or metric reported value "
                             "that was not earned; false_failure = earned value was not credited; money_created = "
                             "resources without a declared source; leak = hidden or private information reached the "
                             "attacker; integrity_break = authorisation, atomicity, settlement, availability or replay "
                             "was broken; legitimate_strategy = it worked and should count; not_applicable = could not "
                             "be run here (reason given); error = the probe itself failed.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, sort_keys=False, default=str)
    print()
    print(table(report))
    print()
    print("counts:", {k: v for k, v in report["counts"].items() if v})
    print("sanity check on vulnerable toys passed:", report["sanity_check"]["passed"])
    print("new findings (not in KNOWN_OPEN):", report["ratchet"]["new_findings"] or "none")
    print("wrote", args.out, f"({report['environment']['wall_s']} s)")
    return 0 if report["sanity_check"]["passed"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
