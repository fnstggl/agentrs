"""Technology invention: executable technologies, constraints, adoption on the plants' own measurements (SP-9.6)."""
import crucible
from crucible.domains import base
from crucible.domains import technology_invention as T

crucible.load_all_modules()


def test_interface_constraints_and_noise_free_gain():
    w = base.world(T.NAME, "forming_standard", 1)
    ind = w.modules["industry"]
    for m in ind.phys["models"]:
        assert T.expected_efficiency(ind, m, 0.4, 0.5) == T.expected_efficiency(ind, m, 0.4, 0.5) > 0
    ns = {}
    exec(T.PATTERN_SEARCH, ns)
    h = [{"speed": 0.4, "temp": 0.5, "units": 100, "made": 60, "kwh": 100.0, "t": 0, "broke": False}]
    r = ns["recommend"]({"history": h})
    assert 0 <= r["speed"] <= 1 and 0 <= r["temp"] <= 1 and (r["speed"], r["temp"]) != (0.4, 0.5)
    w.close()
    broken = base.run(T.NAME, "forming_standard", 1, policy="broken_program", days=20, keep_env=True)
    st = broken["env"].world.state
    seat = broken["seats"][0]
    failed = [t for _, t in st.scan("tech_trials") if t["status"] == "failed" and t["inventor"] == seat]
    assert failed and all("range" in (t.get("why") or "") for t in failed)
    # the market's openctl (previous interface version) fails the world's interface check wherever it is tried
    vend = [t for _, t in st.scan("tech_trials") if t["inventor"] != seat]
    assert vend and all(t["status"] == "failed" and "interface" in (t.get("why") or "") for t in vend)
    assert broken["outcome"]["details"]["adopters"] == 0
    broken["env"].close()
