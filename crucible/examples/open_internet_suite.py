"""Part 21 evaluation suite on a digital world with an invention chain, factual vs four counterfactual conditions.

World `open_tight_compute` (6 service firms, a small cheap provider). Agents: `ada` = CapacityInventor (open
source, so imitators can fork it) and `bea` = PlacementBroker - scripted competent controls. In the
`invention_removed` condition both are admitted but idle. Each condition is run with the same seeds; the
measures (crucible.evaluation.open_internet) and their paired differences are written to
docs/experiments/open_internet_suite.json.

Usage: PYTHONPATH=crucible/src python crucible/examples/open_internet_suite.py [days] [seeds...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.inventors import CapacityInventor, PlacementBroker  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.open_internet import ablate  # noqa: E402
from crucible.worldgen import generate  # noqa: E402


def make_world(params, seed):
    return World.create(generate({"family": "open_tight_compute", "overrides": {"cheap_cores": 40, "params": params}},
                                 seed))


def make_agents(w, inventing):
    out = []
    for h, pol in (("ada", CapacityInventor(open_source=True)), ("bea", PlacementBroker(open_source=True))):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
        out.append((f"p:{h}", pol if inventing else IdlePolicy()))
    return out


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2]
    t0 = time.time()
    res = ablate(make_world, make_agents, days, seeds, warmup_days=1)
    res["config"] = {"family": "open_tight_compute", "days": days, "seeds": seeds, "warmup_days": 1,
                     "agents": {"ada": "CapacityInventor (scripted)", "bea": "PlacementBroker (scripted)"},
                     "wall_s": round(time.time() - t0, 1)}
    for r in res["runs"]:
        r.pop("census", None)
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_internet_suite.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps({"summary": {c: {k: v["mean"] for k, v in s.items()} for c, s in res["summary"].items()},
                      "second_order_established": res["second_order_established"],
                      "complexity_dependence": res["complexity_dependence"], "wall_s": res["config"]["wall_s"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
