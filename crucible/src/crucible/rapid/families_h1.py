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
FIELD_RECORDS = 4

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
    hid = {"theta": u(rng, 3.0, 7.0), "k1": u(rng, 0.5, 1.5), "k2": u(rng, 0.12, 0.3), "p": u(rng, 1.5, 2.5),
           "off": [u(rng, -3.0, 3.0) for _ in range(3)], "slope": [u(rng, 0.3, 1.5) for _ in range(3)]}
    hid["validation"] = [rec(0, 10, 0, 5) for _ in range(30)]
    hid["holdout"] = [rec(10.5, 15, 2, 7) for _ in range(50)]        # beyond the probe box in the first field
    cfg = {"engine": "program_exec", "scenario": "rule_induction", "horizon": int(d["horizon"]), "fields": fields,
           "categories": cats, "box": box, "probe_cost": float(d["experiment_cost"]), "submit_cost": 4.0,
           "budget": float(d["resources"]), "probe_sd": 0.25 * float(d["observation_noise"]), "tol": 0.3,
           "compute_price": 20.0, "value_name": nm[6], "field_records": FIELD_RECORDS, "hidden": hid,
           "endowment": 0}
    from .engines.program_exec import RuleInduction
    probe = RuleInduction(cfg)
    cfg["examples"] = [dict(r, value=round(probe.value(r), 4)) for r in (rec(0, 10, 0, 5) for _ in range(6))]
    cfg["objective"] = {"type": "heldout_tests",
                        "text": "Submit a program that computes the rule. It is scored on a holdout suite of "
                                f"records beyond the probe box ({fa} up to 15, {fb} up to 7); a value passes "
                                "within 0.3 of the rule. Only the last submission counts."}
    cfg["brief"] = (f"Records have numeric fields {fa} and {fb} and a category {fc} (one of {cats}). A hidden "
                    "rule gives each record a value. Probe the rule at records inside the box "
                    f"{fa} in [0, 10], {fb} in [0, 5] (query 'probe'); submit a solver program (kind 'program'); "
                    "each submission reports how many of a validation suite inside the box it passes. Every tick "
                    f"{FIELD_RECORDS} records observed in the field arrive with their noisy values ({nm[6]}_<i>).")
    return cfg


# ====================================================================== program_exec: more 8.3 and 8.6
@family("pe_deadtime_control", engine="program_exec", category="8.3", item="8.3.3", split="val",
        doc="A stable process with hidden gain, lag and dead time (inner steps), an unmeasured drifting load and a "
            "public setpoint schedule. A controller program runs every inner step; aggressive feedback oscillates "
            "through the dead time, so the delay has to be identified and compensated.",
        distinct_because="delayed controller response executed as a program at the inner rate: a first-order lag "
                         "with pure dead time and an autoregressive unmeasured load, setpoint tracking with "
                         "actuator-move costs (no instability, no trips)",
        axes={"horizon": 30, "observation_noise": 1.0, "resources": 1.0}, objective="maximize_net",
        **hooks(PE, "deadtime_control", ("dead_time_exact", "aggressive_gain_oscillates")))
def pe_deadtime_control(rng, d):
    names = neutral_names(rng, 3)
    H = int(d["horizon"])
    sched, t = [], 0
    while t < H:
        sched.append([t, round(u(rng, 3.0, 8.0), 2)])
        t += int(rng.integers(6, 10))
    hid = {"G": u(rng, 0.8, 1.6), "tau": u(rng, 3.0, 8.0), "L": int(rng.integers(2, 8)), "w_sd": u(rng, 0.02, 0.05),
           "phi": u(rng, 0.9, 0.97), "load_sd": u(rng, 0.02, 0.05)}
    umax = 10.0 * float(d["resources"])
    return {"engine": "program_exec", "scenario": "deadtime_control", "horizon": H, "inner": 10, "max_dead": 8,
            "u_name": names[0], "y_name": names[1], "r_name": names[2], "umax": umax, "setpoints": sched,
            "band": 1.0, "price": 20.0, "move_price": 3.0, "energy_price": 0.2, "compute_price": 20.0,
            "sensor_sd": 0.1 * float(d["observation_noise"]), "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: every inner step pays by how close "
                                                          "the process is to the setpoint; valve moves, valve "
                                                          "opening and compute cost money."},
            "brief": f"A process with one valve {names[0]} (0 to {umax:g}) and a sensor {names[1]}; ten inner steps "
                     f"per tick. The setpoint {names[2]} follows this schedule [[from tick, value], ...]: {sched}. "
                     "The lever holds the valve for a whole tick; a deployed program is called by the process at "
                     "every inner step (query 'program_api')."}


@family("pe_contention_sched", engine="program_exec", category="8.3", item="8.3.4", split="train",
        doc="A shared node with four slots; jobs of public types arrive with public sizes. Hidden resource classes "
            "group the types; jobs of one class running together slow each other sharply. A scheduler program is "
            "called at every inner step with a free slot.",
        distinct_because="shared resource contention: per-class congestion among concurrent jobs (non-monotonic "
                         "throughput in concurrency) behind a hidden type-to-class map; a program decides which "
                         "jobs share the node",
        axes={"horizon": 30, "action_complexity": 4, "resources": 4}, objective="maximize_net",
        **hooks(PE, "contention", ("same_class_interferes", "class_aware_beats_caps")))
def pe_contention_sched(rng, d):
    J = max(4, 2 * (int(d["action_complexity"]) // 2))
    names = neutral_names(rng, J + 1)
    cls = [int(x) for x in rng.permutation([k % 2 for k in range(J)])]
    hid = {"cls": cls, "kappa": u(rng, 1.5, 3.0)}
    return {"engine": "program_exec", "scenario": "contention", "horizon": int(d["horizon"]), "inner": 4,
            "slots": int(d["resources"]), "types": names[:J], "mix": [1.0 / J] * J, "arrival_p": 0.45,
            "size": [2.0, 6.0], "step_work": 1.0, "price": 100.0, "hold_cost": 2.0, "max_wait": 8,
            "drop_cost": 50.0, "cap_name": names[J], "compute_price": 20.0, "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: completed jobs pay; waiting and "
                                                          "dropped jobs and compute cost money."},
            "brief": f"A node with {int(d['resources'])} slots runs jobs of types {names[:J]} (equally common, about "
                     "0.45 arrivals per inner step, sizes 2-6 work units, four inner steps per tick; a job running "
                     "alone does one work unit per inner step). Jobs waiting "
                     "more than 8 ticks are dropped. The lever caps how many jobs the built-in oldest-first rule "
                     "runs at once; a deployed scheduler program chooses which jobs to start."}


@family("pe_stream_processor", engine="program_exec", category="8.3", item="8.3.2", split="train",
        doc="Sensors read one process; their gains and offsets drift as hidden random walks and some stick. A data "
            "processor program is run on every raw sample (observations carry only tick means) and paid by "
            "accuracy; a daily audit gives the truth for one sample; recalibration is costly maintenance.",
        distinct_because="equipment degradation handled by an executed data processor: drifting multiplicative "
                         "and additive sensor errors with failures, raw samples visible only to the program, "
                         "audits as sparse ground truth, maintenance resets",
        axes={"horizon": 30, "observation_noise": 1.0, "graph_size": 5}, objective="maximize_net",
        **hooks(PE, "stream", ("sensors_drift", "calibration_beats_averaging")))
def pe_stream_processor(rng, d):
    N = max(3, int(d["graph_size"]))
    names = neutral_names(rng, N)
    hid = {"g_sd": [u(rng, 0.002, 0.02) for _ in range(N)], "o_sd": [u(rng, 0.05, 0.4) for _ in range(N)],
           "hazard": [u(rng, 0.0, 0.04) for _ in range(N)], "phi": u(rng, 0.8, 0.95), "proc_sd": u(rng, 1.0, 2.0)}
    return {"engine": "program_exec", "scenario": "stream", "horizon": int(d["horizon"]), "sensor_names": names,
            "samples": 10, "mean": 50.0, "sensor_sd": 0.3 * float(d["observation_noise"]), "tol": 1.0,
            "price": 10.0, "recal_cost": 200.0, "compute_price": 20.0, "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: every sample's estimate pays by "
                                                          "closeness to the true value (within 1.0); recalibration "
                                                          "and compute cost money."},
            "brief": f"{N} sensors {names} read one process (around 50), ten samples per tick. A deployed processor "
                     "program turns each sample's raw readings into an estimate; without one the plain mean is "
                     "used. Each tick the laboratory audits the last sample of the previous tick. Recalibrating a "
                     "sensor (discrete 'recalibrate') costs 200 and resets it."}


@family("pe_router", engine="program_exec", category="8.3", item="8.3.1", split="val",
        doc="Parallel lines process jobs with hidden size-dependent service times (one fast for small jobs, one for "
            "large, one congestible). A router program is called for every arriving job; completions report service "
            "times, so the bottleneck has to be found from them.",
        distinct_because="unknown processing bottlenecks with per-job routing executed as a program: FIFO lines "
                         "whose service time is a hidden nonlinear function of job size, one with queue-length "
                         "congestion",
        axes={"horizon": 30, "observation_noise": 1.0, "graph_size": 3}, objective="maximize_net",
        **hooks(PE, "router", ("size_dependent_speed", "congested_line_slows")))
def pe_router(rng, d):
    M = max(3, int(d["graph_size"]))
    names = neutral_names(rng, M)
    proto = [(0.4, 1.7, 1.25), (2.6, 0.35, 0.9), (1.2, 0.9, 1.0)] + [(1.5, 0.8, 1.0)] * (M - 3)
    perm = [int(x) for x in rng.permutation(M)]
    jit = lambda x: x * u(rng, 0.85, 1.15)
    hid = {"alpha": [jit(proto[p][0]) for p in perm], "beta": [jit(proto[p][1]) for p in perm],
           "gamma": [jit(proto[p][2]) for p in perm], "congested": perm.index(2), "qstar": int(rng.integers(2, 4)),
           "cong": u(rng, 0.3, 0.6)}
    noise = float(d["observation_noise"])
    return {"engine": "program_exec", "scenario": "router", "horizon": int(d["horizon"]), "inner": 6,
            "line_names": names, "arrival_p": 0.7, "size": [0.5, 4.0], "price_base": 20.0, "price_size": 10.0,
            "hold_cost": 1.0, "max_wait": 30, "drop_cost": 40.0, "service_noise": 0.08 * noise,
            "compute_price": 20.0, "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: completed jobs pay 20 plus 10 per "
                                                          "unit of size; every job in the system costs 1 per inner "
                                                          "step; jobs older than 30 steps are dropped (40)."},
            "brief": f"Lines {names} each serve one job at a time, first come first served; jobs (sizes 0.5-4) "
                     "arrive at about 0.7 per inner step, six inner steps per tick. The built-in split ('allocate') "
                     "is even; a deployed router program places each arriving job. Completed jobs are reported "
                     "with their service times."}


@family("pe_fleet_policy", engine="program_exec", category="8.6", item="8.6.2", split="train",
        doc="A fleet whose machines wear at hidden rates; wear shows only as noisy vibration and makes failures "
            "steeply likely past a hidden level. Service takes a machine off for a tick and its benefit shows "
            "only later, as failures avoided; a policy program decides for the whole fleet each tick.",
        distinct_because="long-term maintenance effects: hidden per-machine wear stocks behind noisy indicators, a "
                         "threshold hazard, imperfect resets and a crew limit; costs now, benefits as absent "
                         "failures many ticks later",
        axes={"horizon": 40, "graph_size": 10, "resources": 2, "observation_noise": 1.0}, objective="maximize_net",
        **hooks(PE, "fleet", ("service_pays_later", "vibration_tracks_wear")))
def pe_fleet_policy(rng, d):
    M = max(4, int(d["graph_size"]))
    names = neutral_names(rng, M)
    hid = {"rate": [u(rng, 0.02, 0.07) for _ in range(M)], "sens": [u(rng, 0.7, 1.3) for _ in range(M)],
           "w0": u(rng, 0.35, 0.6), "k": u(rng, 0.8, 2.0), "p0": 0.002, "eff": u(rng, 0.7, 0.95),
           "w_init": [u(rng, 0.0, 0.5) for _ in range(M)]}
    return {"engine": "program_exec", "scenario": "fleet", "horizon": int(d["horizon"]), "machine_names": names,
            "crew": int(d["resources"]), "output": 100.0, "service_cost": 250.0, "repair_cost": 900.0,
            "repair_ticks": 4, "vib_sd": 0.15 * float(d["observation_noise"]), "compute_price": 20.0,
            "hidden": hid, "endowment": 200_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: every running machine produces 100 "
                                                          "per tick; service (250) and repairs (900) cost money."},
            "brief": f"Machines {names} each report a vibration reading every tick. A failed machine is down for 4 "
                     f"ticks and costs 900 to repair. Service (discrete 'service', or a deployed policy program; at "
                     f"most {int(d['resources'])} machines per tick) takes a machine off for one tick and costs 250."}


@family("pe_cascade_guard", engine="program_exec", category="8.3", item="8.3.5", split="train",
        doc="Units on a ring carry fluctuating demand; true capacities are their public ratings times hidden "
            "factors. A tripped unit's demand moves to its working neighbours and can trip them. A guard program "
            "sheds load every inner step.",
        distinct_because="cascading technical failures under an executed protection program: threshold trips "
                         "with load redistribution along a ring (a cascade) and capacities that differ from their "
                         "ratings",
        axes={"horizon": 30, "graph_size": 8, "observation_noise": 1.0}, objective="maximize_net",
        **hooks(PE, "cascade_guard", ("trips_cascade", "capacity_below_rating")))
def pe_cascade_guard(rng, d):
    N = max(4, int(d["graph_size"]))
    names = neutral_names(rng, N + 1)
    ratings = [round(u(rng, 80.0, 120.0), 2) for _ in range(N)]
    base = [round(r * u(rng, 0.62, 0.78), 2) for r in ratings]
    phase = [round(u(rng, 0.0, 6.283), 4) for _ in range(N)]
    hid = {"factor": [u(rng, 0.72, 1.0) for _ in range(N)]}
    return {"engine": "program_exec", "scenario": "cascade_guard", "horizon": int(d["horizon"]), "inner": 6,
            "unit_names": names[:N], "margin_name": names[N], "ratings": ratings, "base": base, "phase": phase,
            "amp": 0.25, "period": 20.0, "noise_sd": 4.0 * float(d["observation_noise"]), "price": 1.0,
            "trip_cost": 300.0, "restart_steps": 12, "compute_price": 20.0, "hidden": hid, "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise net income: served demand pays 1 per unit per "
                                                          "inner step; trips cost 300 each."},
            "brief": f"Units {names[:N]} on a ring, ratings {ratings}, contracted demands {base} varying by +-25% "
                     f"with period 20 inner steps (phases {phase}), six inner steps per tick. A tripped unit is out "
                     "for 12 inner steps and its demand moves to the nearest working units on either side. The "
                     f"lever {names[N]} sets the built-in protection (shed demand above margin x rating); a "
                     "deployed guard program decides shedding every inner step."}


@family("pe_rollout_targeting", engine="program_exec", category="8.6", item="8.6.3", split="train",
        doc="Users in groups with public features; exposure leads, after a hidden delay, to adoption (a "
            "subscription paid every tick) or permanent rejection. Groups below a hidden adoption share are not "
            "ready and mostly reject. A targeting program chooses whom to expose each tick.",
        distinct_because="delayed adoption with irreversible rejection: feature-dependent propensities, a "
                         "readiness threshold on the group's adopter share and word-of-mouth growth; returns "
                         "arrive as subscriptions ticks after the decision",
        axes={"horizon": 30, "graph_size": 60, "resources": 4}, objective="maximize_net",
        **hooks(PE, "rollout", ("adoption_is_delayed", "unready_groups_reject")))
def pe_rollout_targeting(rng, d):
    U, G = max(20, int(d["graph_size"])), 4
    names = neutral_names(rng, G)
    users = []
    for i in range(U):
        users.append({"group": names[i % G], "f1": round(float(rng.standard_normal()), 3),
                      "f2": round(float(rng.standard_normal()), 3)})
    sgn = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(2)]
    hid = {"w": [sgn[0] * u(rng, 0.8, 1.8), sgn[1] * u(rng, 0.8, 1.8)], "bias": [u(rng, -1.5, 0.5) for _ in range(G)],
           "tau": [u(rng, 0.05, 0.3) for _ in range(G)], "q": u(rng, 0.6, 0.9), "delay": u(rng, 2.0, 4.0),
           "org0": 0.002, "org1": u(rng, 0.03, 0.08), "seed": [u(rng, 0.0, 0.2) for _ in range(G)]}
    cfg = {"engine": "program_exec", "scenario": "rollout", "horizon": int(d["horizon"]), "users": users,
           "group_names": names, "capacity": int(d["resources"]), "fee": 15.0, "exposure_cost": 40.0,
           "compute_price": 20.0, "hidden": hid, "endowment": 100_000,
           "objective": {"type": "maximize_net", "text": "Maximise net income: every adopter pays 15 per tick; "
                                                         "each exposure costs 40."},
           "brief": f"{U} users in groups {names}, each with public features f1, f2. An exposed user decides after "
                    "a while: adopt (and pay every tick from then on) or reject for good. Up to "
                    f"{int(d['resources'])} exposures per tick (discrete 'expose_group', or a deployed targeting "
                    "program)."}
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
    cfg["objective"] = {"type": "predict_heldout", "scoring": "gaussian_tolerance", "quantity": species[1],
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
    cfg["objective"] = {"type": "predict_heldout", "scoring": "gaussian_tolerance", "quantity": q,
                        "text": f"Predict the drift-free effect on {q} (reading minus a blank's reading in the "
                                f"same tick, without noise) at these doses: {heldout}. " + JOURNAL}
    cfg["brief"] = (f"A meter reads {q} of a preparation at dose {nx} (0-4). Its baseline may move between "
                    f"ticks; preparations measured in the same tick share it. The gauge {mon} reads a blank "
                    "(dose 0) every tick.")
    return cfg
