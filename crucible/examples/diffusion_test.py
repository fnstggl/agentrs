"""Does information sharing change decisions? A constructed scenario with an ablation.

`open_digital` with 6 service firms: 2 early adopters (open to novelty, share evidence) and 4 "late majority"
firms that only try capabilities someone else has verified evidence for. A scripted inventor publishes an
autoscaler. With the sharing channels on (published evidence, forum posts with #evidence, mail to contacts) the late
majority can learn from the early adopters' verified trials; with `ablate_sharing` they cannot. Outcome: who
trialled and who adopted, per seed.

Usage: PYTHONPATH=crucible/src python crucible/examples/diffusion_test.py [days] [seeds...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

EARLY = {"risk": 0.8, "openness": 0.95, "share": 1.0, "sociability": 1.0, "honesty": 0.9, "attention": 5, "patience": 0.95}
LATE = {"risk": 0.4, "openness": 0.1, "share": 0.5, "sociability": 0.5, "honesty": 0.9, "attention": 3, "patience": 0.95}


def run(seed, days, ablate):
    traits = [EARLY, LATE, LATE, EARLY, LATE, LATE]
    w = World.create(generate({"family": "open_digital", "overrides": {
        "needs": ["route.shortest", "sort.records"], "firms_per_need": 3, "firm_traits": traits,
        "params": {"ablate_sharing": ablate}}}, seed))
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    Runner(w, [("p:ada", CapacityInventor())], dt=3600, record=False).run(24 * days)
    st = w.state
    firms = sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm" and m["role"] == "service")
    late = [f for f, t in zip(firms, [st.tables["minds"][f]["traits"] for f in firms]) if t["openness"] < 0.25]
    tried = {i["installer"] for _, i in st.scan("cap_installs")}
    adopted = {i["installer"] for _, i in st.scan("cap_installs") if i["status"] == "active"}
    return {"seed": seed, "ablate_sharing": ablate, "late_majority": late, "late_tried": sorted(tried & set(late)),
            "late_adopted": sorted(adopted & set(late)), "all_adopted": sorted(adopted),
            "public_outcomes": sum(1 for _, e in st.scan("cap_evidence") if e["kind"] == "outcome" and e["public"]),
            "beliefs_from_others": sum(1 for _, m in st.scan("minds") for b in m["beliefs"].values() if b.get("src") == "contact")}


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    rows = [run(s, days, a) for s in seeds for a in (False, True)]
    for r in rows:
        print(r, flush=True)
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "diffusion_test.json")
    with open(path, "w") as fh:
        json.dump({"days": days, "rows": rows}, fh, indent=1)
