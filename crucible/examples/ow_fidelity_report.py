"""The fidelity register and the measurable quantities of a few finished worlds, written to one report.

Nothing here compares the simulation with reality: the repository holds no reference data, so every measure says so.
The register records, mechanism by mechanism, what kind of realism it bears on and that it is an uncalibrated hypothesis.

    PYTHONPATH=src python examples/ow_fidelity_report.py [--days 14] [--out docs/experiments/ow_fidelity.json]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation import fidelity as F  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "ow_fidelity.json"))
    a = ap.parse_args()
    worlds = []
    for fam, seed in (("ow_market", 1), ("ow_market", 2), ("ow_heldout", 1)):
        w = World.create(generate({"family": fam}, seed), world_id=f"fidelity-{fam}-{seed}")
        w.advance_time(a.days * DAY)
        worlds.append(w)
    rep = F.fidelity_report(worlds)
    reg = F.REGISTER
    out = {"days": a.days, "worlds": [w.world_id for w in worlds], "problems_in_register": F.validate_register(),
           "register_counts": dict(Counter(m.status for m in reg)), "n_mechanisms": len(reg),
           "register": [{"id": m.id, "modules": list(m.modules), "kinds": list(m.kinds), "status": m.status,
                         "rule": m.rule, "caveat": m.caveat} for m in reg], "report": rep}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(out["register_counts"], "mechanisms", len(reg), "->", a.out)
    for w in worlds:
        w.close()


if __name__ == "__main__":
    main()
