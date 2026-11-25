"""Gate SP-2.1 (MASTER_SPEC Part 2.1): three execution modes - controlled experimental environments, standalone
semantic worlds, integrated civilizations.

Metric (docs/v5/ACCEPTANCE_MANIFEST.json): execution_modes >= 3, where a mode counts only when the execution-mode
registry (crucible.substrate.modes) has a provider for it that is run here for real and passes: it lists environments,
builds one twice from the same seed, runs it with identical results and (for kernel worlds) exact replay.

The test records the measured number honestly; it does not assert the threshold (the acceptance checker compares
the recorded metric with the manifest criterion). Until the Tier 2 (crucible.domains) and Tier 3 (crucible.civ)
packages register providers, modes B and C are reported as not operational, with the reason.
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.substrate import modes

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_1.py"


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
    n = s["execution_modes"]
    assert n == len(s["operational"]) == sum(1 for m in s["modes"].values() if m["operational"])
    metrics = {"execution_modes": n, "operational_modes": ",".join(s["operational"]) or "none",
               "providers_checked": len(rows), "providers": "; ".join(rows) or "none"}
    for m, info in s["modes"].items():
        metrics[f"mode_{m}_operational"] = bool(info["operational"])
        metrics[f"mode_{m}_providers"] = len(info["providers"])
        if not info["operational"]:
            metrics[f"mode_{m}_reason"] = info["reason"]
    evidence.record("SP-2.1", metrics, command=CMD, notes=(
        "Mode A: crucible.mechanisms (4.0 hidden-mechanism families, kernel-free MechSim) plus any crucible.rapid "
        "provider; Mode B: crucible.domains providers; Mode C: crucible.civ providers. Each provider is checked by "
        "running it (two builds from one seed, identical results; kernel worlds also replayed)."))
    # the registry itself must find the built-in Mode A provider operational on this machine
    assert s["modes"]["A"]["operational"], s["modes"]["A"]
