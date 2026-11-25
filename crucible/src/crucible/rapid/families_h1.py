"""Workstream H1 families: executable program environments (`program_exec`) and scientific experiment environments
(`experiment_lab`) across Part 8.2 (experimental science), 8.3 (complex engineering) and 8.6 (delayed
consequences).

Every family draws its hidden mechanism from a seed (names, coefficients, which of a fixed set of schemes) under
`config["hidden"]`, the one place a participant's view never reaches (`ProgramExec.SECRET_KEYS`,
`ExperimentLab.SECRET_KEYS`). Its oracle, scripted public-information reference, identification procedure and
intervention tests are evaluator-side (`crucible.rapid.eval_program_exec`, `crucible.rapid.eval_experiment_lab`);
nothing here imports them.
"""
from __future__ import annotations

import numpy as np

from ..mechanisms.base import neutral_names
from .families import family

PE = "crucible.rapid.eval_program_exec:"
EL = "crucible.rapid.eval_experiment_lab:"
JOURNAL = ("File predictions with the journal: message to 'journal' with content {\"predictions\": [{\"set\": "
           "<condition exactly as listed>, \"value\": <number>}, ...]}. Only the last filing counts.")


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def hooks(prefix: str, stem: str, claims: tuple) -> dict:
    return {"claims": tuple(prefix + c for c in claims), "identify": f"{prefix}identify_{stem}",
            "reference": f"{prefix}reference_{stem}", "oracle": f"{prefix}oracle_{stem}"}


# ====================================================================== program_exec: 8.3 engineering
@family("pe_unstable_control", engine="program_exec", category="8.3", item="8.3.6", split="train",
        doc="An open-loop unstable plant evolves over ten inner steps per tick with a hidden actuator gain (sign and "
            "size), growth rate and bias. The manual lever holds one actuator value per tick, which cannot keep it "
            "in band; a controller program, called by the plant every inner step, can.",
        distinct_because="executable controller design: an agent-written program is executed inside the dynamics "
                         "at the inner rate of an unstable plant (feedback with a growth factor above one per "
                         "tick) and trips past a threshold; per-tick actions cannot stabilise it",
        axes={"horizon": 30, "observation_noise": 1.0, "resources": 1.0}, objective="maximize_net",
        **hooks(PE, "fast_control", ("unstable_open_loop", "inner_rate_control")))
def pe_unstable_control(rng, d):
    names = neutral_names(rng, 2)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    H = int(d["horizon"])
    steps = sorted(int(x) for x in rng.choice(np.arange(4, max(6, H - 2)), size=2, replace=False))
    hid = {"a": u(rng, 0.15, 0.25), "b": sign * u(rng, 0.2, 0.4), "w_sd": u(rng, 0.04, 0.07),
           "d": [[0, u(rng, -0.04, 0.04)]] + [[s, u(rng, 0.04, 0.09) * (1 if rng.random() < 0.5 else -1)]
                                              for s in steps]}
    umax = 1.0 * float(d["resources"])
    return {"engine": "program_exec", "scenario": "fast_control", "horizon": int(d["horizon"]), "inner": 10,
            "u_name": names[0], "y_name": names[1], "umax": umax, "band": 0.3, "trip": 3.0, "halt_ticks": 2,
            "price": 30.0, "energy_price": 4.0, "trip_cost": 600.0, "compute_price": 20.0,
            "sensor_sd": 0.08 * float(d["observation_noise"]), "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: every inner step the plant's state "
                                                          "is inside the band pays, more the closer to zero; "
                                                          "actuation, trips and compute cost money."},
            "brief": f"A fast plant with one actuator {names[0]} (range +-{umax:g}) and one sensor {names[1]}; ten "
                     "inner steps per tick; the load on it may change. The state should stay within +-0.3 of "
                     "zero; beyond +-3 the plant trips and halts for two ticks. The lever holds the actuator for a "
                     "whole tick; a deployed program is called by the plant at every inner step (query "
                     "'program_api')."}


# ====================================================================== program_exec: 8.2 science as code
@family("pe_rule_induction", engine="program_exec", category="8.2", item="8.2.1", split="train",
        doc="A hidden rule maps records to values (category offsets, a category-specific slope, a response that "
            "bends beyond a hidden threshold). Costly noisy probes inside a box; a submitted solver program is run "
            "on a validation suite (reported) and on a holdout suite beyond the box (scored).",
        distinct_because="the hypothesis is an executable program judged on hidden test suites at unobserved "
                         "conditions (extrapolation beyond the probe box); probing is the experiment, the "
                         "validation suite the only feedback",
        axes={"observation_noise": 1.0, "experiment_cost": 1.0, "resources": 60.0, "horizon": 20},
        objective="heldout_tests", **hooks(PE, "rule_induction", ("kink_at_threshold", "holdout_needs_the_law")))
def pe_rule_induction(rng, d):
    nm = neutral_names(rng, 7)
    fields, cats = nm[:3], nm[3:6]
    fa, fb, fc = fields
    box = {fa: [0.0, 10.0], fb: [0.0, 5.0]}

    def rec(alo, ahi, blo, bhi):
        return {fa: round(u(rng, alo, ahi), 4), fb: round(u(rng, blo, bhi), 4), fc: cats[int(rng.integers(3))]}
    hid = {"theta": u(rng, 3.0, 7.0), "k1": u(rng, 0.5, 1.5), "k2": u(rng, 0.1, 0.3), "p": u(rng, 1.4, 2.6),
           "off": [u(rng, -3.0, 3.0) for _ in range(3)], "slope": [u(rng, 0.3, 1.5) for _ in range(3)]}
    hid["validation"] = [rec(0, 10, 0, 5) for _ in range(30)]
    hid["holdout"] = [rec(7, 15, 2, 7) for _ in range(50)]
    cfg = {"engine": "program_exec", "scenario": "rule_induction", "horizon": int(d["horizon"]), "fields": fields,
           "categories": cats, "box": box, "probe_cost": float(d["experiment_cost"]), "submit_cost": 4.0,
           "budget": float(d["resources"]), "probe_sd": 0.25 * float(d["observation_noise"]), "tol": 0.4,
           "compute_price": 20.0, "value_name": nm[6], "hidden": hid, "endowment": 0}
    from .engines.program_exec import RuleInduction
    probe = RuleInduction(cfg)
    cfg["examples"] = [dict(r, value=round(probe.value(r), 4)) for r in (rec(0, 10, 0, 5) for _ in range(6))]
    cfg["objective"] = {"type": "heldout_tests",
                        "text": "Submit a program that computes the rule. It is scored on a holdout suite of "
                                f"records beyond the probe box ({fa} up to 15, {fb} up to 7); a value passes "
                                "within 0.4 of the rule. Only the last submission counts."}
    cfg["brief"] = (f"Records have numeric fields {fa} and {fb} and a category {fc} (one of {cats}). A hidden "
                    "rule gives each record a value. Probe the rule at records inside the box "
                    f"{fa} in [0, 10], {fb} in [0, 5] (query 'probe'); submit a solver program (kind 'program'); "
                    "each submission reports how many of a validation suite inside the box it passes. Every tick one "
                    f"record observed in the field arrives with its noisy value ({nm[6]}).")
    return cfg


# ====================================================================== experiment_lab: 8.2 experimental science
def _lab_common(d, budget_key="resources") -> dict:
    return {"engine": "experiment_lab", "horizon": int(d["horizon"]), "budget": float(d[budget_key]),
            "base_cost": float(d["experiment_cost"]), "endowment": 0}


@family("lab_kinetics", engine="experiment_lab", category="8.2", item="8.2.4", split="train",
        doc="Three species; load the first and set the temperature. The hidden scheme is sequential first-order or "
            "autocatalytic, with temperature factors per 10 degrees. Predict the second species at temperatures "
            "above the settable range from designed, sampled-in-time experiments.",
        distinct_because="unknown transformation dynamics: an ODE reaction network of hidden scheme integrated in "
                         "time, sampled at designed times by two instruments, extrapolated in temperature",
        axes={"horizon": 30, "observation_noise": 1.0, "experiment_cost": 1.0, "resources": 120.0},
        objective="predict_heldout", **hooks(EL, "kinetics", ("scheme_curvature", "mass_balance",
                                                               "rates_rise_with_temperature")))
def lab_kinetics(rng, d):
    nm = neutral_names(rng, 6)
    nT, nA, species, mon = nm[0], nm[1], nm[2:5], nm[5]
    noise = float(d["observation_noise"])
    scheme = "sequential" if rng.random() < 0.5 else "autocatalytic"
    k30 = [u(rng, 0.15, 0.4), u(rng, 0.05, 0.15)] if scheme == "sequential" else \
        [u(rng, 0.5, 1.2), u(rng, 0.05, 0.15)]
    hid = {"scheme": scheme, "k30": k30, "q": [u(rng, 1.5, 2.5), u(rng, 1.5, 2.5)],
           "seed_b": u(rng, 0.02, 0.06) if scheme == "autocatalytic" else 0.0}
    heldout = [{nT: T, nA: a0, "time": tt} for T in (45.0, 58.0) for a0 in (3.0, 4.0) for tt in (2, 4)]
    cfg = dict(_lab_common(d), scenario="kinetics",
               vars=[{"name": nT, "min": 20.0, "max": 50.0, "default": 30.0},
                     {"name": nA, "min": 0.5, "max": 2.0, "default": 1.0}],
               species=species, monitor_name=mon, monitor_time=3, max_time=12, max_replicates=4,
               instruments={"probe": {"cost": 1.0, "measures": [species[1]], "sd": round(0.02 * noise, 6)},
                            "chromatograph": {"cost": 4.0, "measures": list(species), "sd": round(0.006 * noise, 6)}},
               heldout=heldout, tol=0.01, hidden=hid)
    cfg["objective"] = {"type": "predict_heldout", "quantity": species[1],
                        "text": f"Predict the noise-free concentration of {species[1]} at these conditions "
                                f"(loads, and one temperature, beyond what you can set; 'time' is the sampling "
                                f"time; a prediction is judged with tolerance 0.01): {heldout}. "
                                + JOURNAL}
    cfg["brief"] = (f"A batch reactor: set temperature {nT} (20-50) and the initial load {nA} (0.5-2.0) of "
                    f"{species[0]}; species {species}. A trial samples fresh preparations at a chosen time. "
                    f"Instruments: 'probe' reads {species[1]}; 'chromatograph' reads all three. The gauge {mon} "
                    f"is the live reactor's {species[1]} at time 3 under the lever settings.")
    return cfg


@family("lab_drift_blanks", engine="experiment_lab", category="8.2", item="8.2.5", split="train",
        doc="A small dose-response effect on a meter whose baseline drifts between ticks and whose noise has heavy "
            "tails. Replicates in a tick share the baseline. Predict the drift-free effect at unobserved doses.",
        distinct_because="noisy repeated measurement: a random-walk baseline confounds readings taken at "
                         "different times and gross outliers corrupt means; only blanks interleaved in the same "
                         "tick and robust replication identify the saturating law",
        axes={"horizon": 30, "observation_noise": 1.0, "experiment_cost": 1.0, "resources": 160.0},
        objective="predict_heldout", **hooks(EL, "drift_blanks", ("baseline_drifts", "heavy_tails",
                                                                   "blanks_remove_drift")))
def lab_drift_blanks(rng, d):
    nm = neutral_names(rng, 3)
    nx, q, mon = nm
    noise = float(d["observation_noise"])
    hid = {"emax": u(rng, 0.8, 2.0), "ec50": u(rng, 1.5, 4.0), "n": u(rng, 1.0, 2.5), "drift_sd": u(rng, 0.08, 0.15),
           "base0": u(rng, -1.0, 1.0), "out_p": u(rng, 0.08, 0.15), "out_scale": u(rng, 1.5, 3.0)}
    heldout = [{nx: x} for x in (2.5, 5.0, 6.0, 8.0)]
    cfg = dict(_lab_common(d), scenario="drift_blanks",
               vars=[{"name": nx, "min": 0.0, "max": 4.0, "default": 0.0}], quantity=q, monitor_name=mon,
               max_replicates=6, instruments={"meter": {"cost": 1.0, "measures": [q], "sd": round(0.15 * noise, 6)}},
               heldout=heldout, tol=0.08, hidden=hid)
    cfg["objective"] = {"type": "predict_heldout", "quantity": q,
                        "text": f"Predict the drift-free effect on {q} (reading minus a blank's reading in the "
                                f"same tick, without noise) at these doses: {heldout}. " + JOURNAL}
    cfg["brief"] = (f"A meter reads {q} of a preparation at dose {nx} (0-4). Its baseline may move between "
                    f"ticks; preparations measured in the same tick share it. The gauge {mon} reads a blank "
                    "(dose 0) every tick.")
    return cfg
