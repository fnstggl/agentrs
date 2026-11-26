"""Gate V (MASTER_SPEC Part 17): performance and scalability, measured on the real system (workstream M).

The benchmark suite (crucible.scale.bench5) runs the integrated civilisation world (crucible.civ.generate, D-17) and
the population-only world for the development, standard and large profiles, 3 seeds x 30 simulated days each, as
governed jobs (crucible.scale.slots, declared grants in all seven dimensions). The machine-specific targets were
adopted by DECISION_LOG D-27 before any official benchmark (docs/v5/performance/targets.json) and bind the civ world.
Further workloads: the backend matrix (V.4), the event-driven comparison (V.5), the representation cache over three
worlds (V.6), the estimated-mode validation (V.9) and the 10M-identity capacity run (V.2).

Run it inside a governed job that can hold the suite (it nests its runs in the job's own ledger):
    PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 \\
        --label "M: gate V" -- python -m crucible.acceptance.v5 run --requirement V.1
"""
import math

import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import bench5, slots

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 -- "
       "python -m crucible.acceptance.v5 run --requirement V.<n>")
QUANT_KEYS = ("transitions_per_s", "decisions_per_s", "sim_days_per_wall_hour", "cpu_utilisation", "peak_ram_mb",
              "peak_vram_mb", "model_parameter_counts", "inference_token_counts", "event_storage_growth",
              "snapshot_size", "world_generation_cost")


def _profile_ok(runs, kind, prof):
    rs = [r for r in runs if r["world"] == kind and r["profile"] == prof]
    full = [r for r in rs if r["sim_days"] >= 30 - 1e-6 and all(k in r for k in QUANT_KEYS)
            and r["snapshot_size"]["fork_exact"]]
    acting = [r for r in full if all(c["acting"] for c in r["class_activity"].values())]
    return {"runs": len(rs), "full": len(full), "four_classes_acting": len(acting),
            "classes_acting_min": min((sum(c["acting"] for c in r["class_activity"].values()) for r in rs), default=0),
            "ok": len({r["seed"] for r in acting}) >= 3}


def test_V_1_profiles_benchmarked():
    s = K.suite()
    runs = s["runs"]
    per = {p: _profile_ok(runs, "civ", p) for p in bench5.PROFILES}
    pop = {p: _profile_ok(runs, "population", p) for p in bench5.PROFILES}
    m = {"profiles_benchmarked": sum(1 for v in per.values() if v["ok"]), "runs_completed": len(runs),
         "runs_failed": len(s["failed"]), "days_per_run": float(K.DAYS), "seeds": len(K.SEEDS)}
    for p, v in per.items():
        m[f"civ_{p}_runs_with_four_classes_acting"] = v["four_classes_acting"]
        m[f"civ_{p}_classes_acting_min"] = v["classes_acting_min"]
        m[f"population_{p}_runs_full"] = pop[p]["full"]
        m[f"population_{p}_classes_acting_min"] = pop[p]["classes_acting_min"]
    for key, a in s["aggregate"].items():
        m[f"{key.replace('/', '_')}_sim_days_per_cpu_hour"] = a["sim_days_per_cpu_hour"]["mean"]
    evidence.record("V.1", m, command=CMD.replace("<n>", "1"),
                    notes=f"failed runs: {s['failed']}; aggregate: {s['aggregate']}; reused from cache: {s['reused']}")
    assert m["profiles_benchmarked"] >= 3, f"civ profiles not fully benchmarked with four acting classes: {per}"


def test_V_2_ten_million_identities():
    c = K.capacity()
    r = c.get("result") or {}
    u = c.get("usage") or {}
    tg = next(t for t in bench5.targets()["targets"] if t["id"] == "T4")["value"]
    m = {"identities_in_scale_test": int(r.get("identities", 0)),
         "identities_with_decisions": int(r.get("identities_with_decisions", 0)),
         "qualifying_distinct_consumption_events": int(r.get("qualifying_distinct_consumption_events", 0)),
         "recount_mismatches": int(r.get("recount_mismatches", -1)), "wall_s": float(u.get("wall_s") or 0),
         "peak_rss_mb": float(u.get("peak_rss_mb") or 0), "disk_peak_mb": float(u.get("disk_peak_mb") or 0),
         "cpus": int(u.get("cpu") or 0), "estimated": False}
    m["capacity_target_met"] = (m["wall_s"] <= tg["wall_s"] and m["cpus"] <= tg["cpus"]
                                and m["peak_rss_mb"] <= tg["peak_rss_mb"] and m["disk_peak_mb"] <= tg["disk_mb"])
    evidence.record("V.2", m, command=CMD.replace("<n>", "2"),
                    notes=f"internet profile of the one population with the posts platform (social.cascade); "
                          f"progress {r.get('progress')}; usage violations {u.get('violations')}")
    assert m["identities_in_scale_test"] >= 10_000_000
    assert m["qualifying_distinct_consumption_events"] >= 10_000_000 and m["recount_mismatches"] == 0
    assert m["capacity_target_met"], f"T4 capacity target missed: {m}"


def test_V_3_parameter_counts_reported():
    s = K.suite()
    counts = {}
    for r in s["runs"]:
        for k, v in r["model_parameter_counts"].items():
            if int(v) > 0:
                counts[k.split(":")[0] + ":" + k.split(":")[1] if ":" in k else k] = int(v)
    bm = (K.backend_matrix().get("result") or {})
    if bm:
        counts["class2_adaptive_net(bench)"] = int(bm["adaptive"]["params"])
        for a, v in bm["strategic"].items():
            counts[f"class3_strategic_{a}(bench)"] = int(v["params"])
    m = {"model_parameter_counts_reported": len(counts), "models": ",".join(f"{k}={v}" for k, v in sorted(counts.items()))}
    evidence.record("V.3", m, command=CMD.replace("<n>", "3"))
    assert m["model_parameter_counts_reported"] >= 3 and all(v > 0 for v in counts.values())


def test_V_4_batched_inference():
    bm = K.backend_matrix()["result"]
    s = K.suite()
    sizes = []
    for r in s["runs"]:
        b = r.get("batching", {}).get("class2", {})
        if b.get("calls"):
            sizes += [max(1, int(round(b["mean_batch"])))] * int(b["calls"])
    from crucible.scale import backends
    sp = backends.batching_speedup(bm["adaptive"], sizes or None)
    in_world = sp.get("in_world", {})
    m = {"batched_inference_speedup": float(in_world.get("speedup") or 0.0),
         "micro_speedup_b256": float(sp["micro"].get("256", 0.0)), "in_world_batches": int(in_world.get("batches", 0)),
         "in_world_mean_batch": float(in_world.get("mean_batch", 0.0)),
         "strategic_mlp_speedup": float(bm["strategic"]["mlp"]["speedup"]),
         "strategic_gru_speedup": float(bm["strategic"]["gru"]["speedup"]),
         "encoder_batch_speedup": float(bm["encoder"]["speedup"])}
    best256 = [r for r in bm["adaptive"]["rows"] if r["B"] == 256 and r.get("backend") == "torch_cpu"]
    m["class2_batched_decisions_per_cpu_s"] = float(best256[0]["decisions_per_cpu_s"] or 0) if best256 else 0.0
    evidence.record("V.4", m, command=CMD.replace("<n>", "4"), notes=f"speedups {sp}; choice {bm['choice']}")
    assert m["in_world_batches"] > 0, "no in-world class-2 batches were measured"
    assert m["batched_inference_speedup"] >= 2.0 and m["micro_speedup_b256"] >= 2.0


def test_V_5_event_driven_computation():
    r = K.event_driven()["result"]
    m = {"event_driven_speedup": float(r["event_driven_speedup"]), "equal_outcomes": bool(r["equal_outcomes"]),
         "activation_ratio": float(r["activation_ratio"]), "activations": int(r["activations"]),
         "ticks": int(r["ticks"]), "wake_batches": int(r["wake_batches"]),
         "world_cpu_ratio": float(r["world_cpu_ratio"]), "world_cpu_per_activation_ms": float(r["world_cpu_per_activation_ms"])}
    evidence.record("V.5", m, command=CMD.replace("<n>", "5"), notes=str(r))
    assert m["equal_outcomes"] and m["activations"] > 0
    assert m["event_driven_speedup"] >= 2.0
    assert m["wake_batches"] > 0, "no activation was caused by a recorded trigger (population.wake)"


def test_V_6_content_representation_caching():
    r = K.cache_workload()["result"]
    m = {"embedding_cache_hit_rate": float(r["hit_rate"]), "requests": int(r["requests"]),
         "encoder_runs_per_artifact": float(r["encoder_runs_per_artifact"]),
         "cross_world_reused_items": int(r["cross_world_reused_items"]),
         "disk_hits_in_later_worlds": int(r["disk_hits_in_later_worlds"]), "worlds": int(r["worlds"])}
    evidence.record("V.6", m, command=CMD.replace("<n>", "6"), notes=str(r["per_world"]))
    assert m["requests"] > 0 and m["worlds"] == 3
    assert m["embedding_cache_hit_rate"] >= 0.9 and m["encoder_runs_per_artifact"] <= 1.0
    assert m["cross_world_reused_items"] >= 1 and m["disk_hits_in_later_worlds"] >= 1


def test_V_7_peak_cpu_ram_vram_measured():
    s = K.suite()
    us = [u for u in s["usage"] if u.get("rc") == 0]
    bm = K.backend_matrix()["usage"]
    peak_cpu = max((float(u.get("peak_cpu") or 0) for u in us), default=0.0)
    peak_ram = max((float(u.get("peak_rss_mb") or 0) for u in us), default=0.0)
    m = {"peak_cpu_measured": bool(us) and all(float(u.get("peak_cpu") or 0) > 0 for u in us),
         "peak_ram_measured": bool(us) and all(float(u.get("peak_rss_mb") or 0) > 0 for u in us),
         "peak_vram_measured": float(bm.get("peak_gpu_mb") or 0) > 0 and all("peak_vram_mb" in r for r in s["runs"]),
         "peak_cpu": peak_cpu, "peak_ram_mb": peak_ram, "peak_vram_mb_gpu_workload": float(bm.get("peak_gpu_mb") or 0),
         "peak_vram_mb_suite": max((float(r["peak_vram_mb"]) for r in s["runs"]), default=0.0),
         "jobs": len(us)}
    evidence.record("V.7", m, command=CMD.replace("<n>", "7"))
    assert m["peak_cpu_measured"] and m["peak_ram_measured"] and m["peak_vram_measured"]
    assert all(float(u["peak_rss_mb"]) <= float(u["ram_mb"]) for u in us if u.get("ram_mb"))
    assert all(float(u["peak_cpu"]) <= u["cpu"] * slots.CPU_TOLERANCE + 0.5 for u in us)


def test_V_8_compute_budget_compliance():
    K.suite()
    K.backend_matrix()
    recs = K.all_usage()
    viol = [v for r in recs for v in slots.budget_violations(r)]
    undeclared = [r["label"] for r in recs if not {"ram_mb", "disk_mb", "inference", "worlds"} <= set(r.get("declared") or [])]
    m = {"compute_budget_violations": len(viol), "governed_jobs": len(recs), "jobs_with_undeclared_dimensions": len(undeclared),
         "cpu_s_total": round(sum(float(r.get("cpu_s") or 0) for r in recs), 1)}
    evidence.record("V.8", m, command=CMD.replace("<n>", "8"), notes=f"violations: {viol[:20]}; undeclared: {undeclared}")
    assert m["governed_jobs"] >= 9 and not undeclared
    assert m["compute_budget_violations"] == 0


def test_V_9_approximation_errors_quantified():
    a = K.approximation()["result"]
    errs = a["errors"]
    finite = [q for q in K.approx.QUANTITIES if q in errs and math.isfinite(errs[q]["mean"])]
    m = {"approximation_error_metrics": len(finite), "estimated_mode": a["mode"], "n_fit": a["n_fit"],
         "n_target": a["n_target"], "target_seeds": len(a["target_seeds"])}
    for q in finite:
        m[f"error_{q}"] = float(errs[q]["mean"])
    evidence.record("V.9", m, command=CMD.replace("<n>", "9"), labels={"estimated": True},
                    notes=f"errors with 95% intervals over seeds: {errs}; limits: {a['limits']}")
    assert m["approximation_error_metrics"] >= 7 and m["target_seeds"] >= 3 and a["estimated"] is True


def test_V_n1_machine_targets_recorded_before_benchmark_and_met():
    ad = K.targets_adoption(K.ROOT)
    s = K.suite()
    started = min((r["started"] for r in s["runs"]), default=K.T_START)
    recorded_before = bool(ad["all_adopted"] and ad["current_adopted"] and ad["current_committed"]
                           and ad["first_commit_time"] is not None and ad["first_commit_time"] < started)
    checks = bench5.check_targets(s["aggregate"], bench5.targets(), world="civ")
    thr = [c for c in checks if c.get("group") == "throughput" or c.get("measured") is None]
    m = {"targets_recorded_before_benchmark": recorded_before,
         "throughput_targets_met": bool(thr) and all(c["met"] for c in thr),
         "targets_checked": len(checks), "targets_met": sum(1 for c in checks if c["met"]),
         "targets_versions": len(ad["versions"])}
    for c in checks:
        if isinstance(c.get("measured"), (int, float)):
            m[f"{c['id']}_{c['profile']}_measured"] = float(c["measured"])
    evidence.record("V.n1", m, command=CMD.replace("<n>", "n1"), notes=f"checks: {checks}; adoption: {ad}")
    assert recorded_before, ad
    assert m["throughput_targets_met"], [c for c in thr if not c["met"]]
