"""Gate SP-2.1 (MASTER_SPEC Part 2.1): three execution modes - controlled experimental environments, standalone
semantic worlds, integrated civilizations.

Metric (docs/v5/ACCEPTANCE_MANIFEST.json): execution_modes >= 3, where a mode counts only when the execution-mode
registry (crucible.substrate.modes) has a provider for it that is run here for real and passes its mode's check:

* Mode A - `crucible.rapid` (one training family per engine, built twice from a seed and driven by the same seeded
  random policy: identical outcomes, and responsive to actions) and the 4.0 hidden-mechanism families;
* Mode B - `crucible.domains` providers: each lists its worlds, builds one twice (identical hashes) and replays it;
* Mode C - `crucible.civ` providers: the world check PLUS genuine shared-state composition
  (`modes.composition`): the mandatory duplicate-fact check passes, >= 12 of the 17 Part 2.1 domains are engaged by
  another module or by the population after genesis, the one population reaches >= 6 domains, >= 8 shared kinds
  (population and money among them) are written by transactions another module initiated, >= 12 distinct
  (initiator, owner) pairs.

The test records the measured number honestly and does not assert the threshold (the acceptance checker compares
the recorded metric with the manifest criterion). It also measures the Mode C check on a negative control - every
shipped module loaded into one kernel world (audit J2) - which must NOT count as an integrated civilization.
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.substrate import modes

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_1.py"


def all_modules_world():
    import crucible
    from crucible.core.kernel import World, WorldSpec
    from crucible.substrate import facts
    crucible.load_all_modules()
    first = ["identity", "bank", "admin", "artifacts", "cols", "code", "search", "forum", "cloud", "net", "appdb",
             "popsites", "pop", "popweb"]
    excl = {"calendar", "crowd", "feed"}
    names = facts._with_deps(first + [m for m in facts.shipped_modules() if m not in excl and m not in first])
    return World.create(WorldSpec("sp21-all", 7, [{"name": m, "config": {}} for m in names],
                                  {"cost_model": "deterministic"}), world_id="sp21-all")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_SP_2_1_execution_modes():
    s = modes.status(run_checks=True)
    rows = []
    for m, info in s["modes"].items():
        for p in info["providers"]:
            rows.append(f"{m}:{p['name']}({p['package']})={'ok' if p['check']['ok'] else 'fail'}")
        # a mode is operational exactly when one of its providers passed a check that ran here
        assert info["operational"] == any(p["check"]["ok"] for p in info["providers"])
        if not info["providers"]:
            assert not info["operational"] and info["reason"]
        for p in info["providers"]:
            chk = p["check"]
            assert "wall_s" in chk and ("error" in chk or chk.get("n_envs", 0) > 0)
            if chk["ok"]:
                assert chk.get("deterministic") is True and chk.get("n_envs", 0) >= 1
            if m == "C" and "composition" in chk:
                assert chk["ok"] == (chk["deterministic"] and chk["replay_exact"] is not False
                                     and chk["composition"]["integrated"])
    n = s["execution_modes"]
    assert n == len(s["operational"]) == sum(1 for m in s["modes"].values() if m["operational"])
    metrics = {"execution_modes": n, "operational_modes": ",".join(s["operational"]) or "none",
               "providers_checked": len(rows), "providers": "; ".join(rows) or "none"}
    for m, info in s["modes"].items():
        metrics[f"mode_{m}_operational"] = bool(info["operational"])
        metrics[f"mode_{m}_providers"] = len(info["providers"])
        if not info["operational"]:
            metrics[f"mode_{m}_reason"] = info["reason"]
        for p in info["providers"]:
            chk = p["check"]
            if not chk["ok"] and chk.get("error"):
                metrics[f"mode_{m}_{p['name']}_error"] = chk["error"][:200]
    rapid = next(p for p in s["modes"]["A"]["providers"] if p["name"] == "rapid")["check"]
    metrics.update({"mode_A_rapid_ok": bool(rapid["ok"]), "mode_A_rapid_engines": rapid.get("engines", 0),
                    "mode_A_rapid_responsive_engines": rapid.get("responsive_engines", 0),
                    "mode_A_rapid_families": rapid.get("n_envs", 0)})
    # negative control for the Mode C check: everything loaded into one world, run for a day
    w = all_modules_world()
    w.advance_time(86400)
    comp = modes.composition(w)
    metrics.update({"control_all_modules_world_modules": len(w.modules),
                    "control_all_modules_world_integrated": bool(comp["integrated"]),
                    "control_all_modules_domains_engaged": len(comp["domains_engaged"]),
                    "control_all_modules_domains_not_engaged": ",".join(comp["domains_not_engaged"]),
                    "control_all_modules_population_domains": len(comp["population_domains"]),
                    "control_all_modules_cross_module_kinds": len(comp["cross_module_kinds"]),
                    "mode_C_thresholds": str(comp["thresholds"])})
    evidence.record("SP-2.1", metrics, command=CMD, notes=(
        "Mode A: crucible.rapid (registered by crucible.substrate.modes; one family per engine, seeded random policy, "
        "deterministic and responsive) and crucible.mechanisms (4.0 MechSim); Mode B: crucible.domains providers; "
        "Mode C: crucible.civ providers, checked for genuine shared-state composition (modes.composition). Negative "
        "control: every shipped module loaded into one world is refused by the Mode C composition check."))
    # the registry itself must find the Mode A providers operational on this machine, and the Mode C check must
    # refuse the disconnected all-module world
    assert s["modes"]["A"]["operational"] and rapid["ok"], s["modes"]["A"]
    assert not comp["integrated"] and len(w.modules) >= 85
