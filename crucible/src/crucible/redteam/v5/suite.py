"""The 5.0 ratchet (findings known to be open), and the measurement of Gate W from a red-team report.

`KNOWN_OPEN` lists every probe that reports a finding today, with the outcome first observed, its severity and the
workstream that owns the fix. `tests/v5/test_gate_W.py` fails when a probe that is not listed reports a finding (a
regression or a new hole) and when a listed one no longer does (then it was fixed: remove the entry and keep the
probe, which now has to hold). A finding is never marked fixed without its probe reporting `held`.

`gate_metrics(report)` maps the probe outcomes to the metric names of the Gate W manifest entries:

    W.1 sandbox_tests_passed, sandbox_escapes          W.6 accounting_violations
    W.2 evaluator_leaks                                W.7 resource_limit_escapes
    W.3 reward_hack_probes, open_critical_high_reward_hacks
    W.4 open_metric_manipulations, metric_manipulation_probes
    W.5 cross_world_leaks                              W.8 network_escapes
                                                       W.9 private_state_leaks

Counting rules (stricter than the criteria's wording, never looser): a probe that errors counts as a finding of its
gates; a probe that is not applicable is reported per gate (the gate test requires zero); W.1's
`sandbox_tests_passed` counts every refused escape check of the escape matrix plus every other held W.1 probe;
`evaluator_leaks` counts every W.2 finding of severity medium or above (a low-severity structural finding - an
import path no participant can trigger - is reported separately as `evaluator_isolation_low_findings`).
"""
from __future__ import annotations

from ..framework import ERROR, FINDINGS, NOT_APPLICABLE

GATES = ("W.1", "W.2", "W.3", "W.4", "W.5", "W.6", "W.7", "W.8", "W.9")

# probe -> (outcome first observed, severity, owner workstream, note)
KNOWN_OPEN: dict[str, dict] = {
    "v5.isolation.in_process_learner_reads_hidden_grader": {
        "outcome": "leak", "severity": "high", "owner": "L",
        "note": "an in-process learner holding ExperienceEnv reaches the Tier 1 scenario's hidden configuration; "
                "evaluated learners need an out-of-process path (HTTP API / subprocess bridge)"},
    "v5.isolation.tier1_engines_cannot_import_oracles": {
        "outcome": "integrity_break", "severity": "low", "owner": "H",
        "note": "crucible/rapid/__init__.py imports admission -> references -> oracles eagerly; no participant path "
                "(Tier 1 actions are data, programs are sandboxed)"},
    "v5.social.sybils_with_a_fake_footprint": {
        "outcome": "false_success", "severity": "medium", "owner": "C",
        "note": "state-based engagement integrity cannot tell colluding agents with distinct controllers, own "
                "funding and a token footprint from an audience; needs behavioural detection or identity cost"},
    "v5.sandbox.deterministic_memo_first_measurement_game": {
        "outcome": "false_success", "severity": "low", "owner": "O",
        "note": "the deterministic cost model bills an identical computation by its first measurement while a repeat "
                "stays within MEMO_TOL (3x) + 1 ms; a stateful program can hide up to that much repeated work"},
    "v5.innovation.adoption_by_reading_the_package_source": {
        "outcome": "false_success", "severity": "medium", "owner": "G",
        "note": "ablation evidence cannot tell 'output depends on the package's files' from 'uses the package' (nor "
                "a program that inspects the ablation stub, or one wrong on purpose in a random subset of reruns, "
                "credited with probability 1/(k+1) despite the unablated control rerun); a per-dependency RPC "
                "boundary would. Such a dependent gains nothing that trivially calling the package would not"},
}

# probes that must hold in every build (a finding here fails the gate even if someone adds it to KNOWN_OPEN)
MUST_HOLD = ("v5.sandbox.escape_matrix", "v5.isolation.sandboxed_code_cannot_import_or_read_evaluator",
             "v5.network.jailed_media_tools_cannot_reach_network_or_host_files",
             "v5.network.population_world_is_offline", "v5.crossworld.artifact_bytes_do_not_cross_worlds",
             "v5.private.social_dms_members_only_drafts_private_communities",
             "v5.isolation.world_code_cannot_import_evaluator_side_code")

# audit findings (O_security and the areas read for Gate W) closed by this workstream, with the probe that holds now
FIXED: dict[str, str] = {
    "O 3.1.1 / G S1-S2: work meter forgeable inside the sandbox": "v5.sandbox.meter_forgery.*",
    "O 3.1.1: forged 'done' message / work after completion": "v5.sandbox.forged_completion_keeps_working",
    "O 3.1.2: threads outlive the job in pooled workers": "v5.sandbox.thread_outlives_request",
    "O 3.1.3: no aggregate disk quota": "v5.sandbox.disk_quota",
    "O 3.1.4: all tenants share uid nobody": "v5.sandbox.tenants_do_not_share_a_uid",
    "O 3.2 / P4: free and negative compute billing": "v5.sandbox.meter_forgery.negative_line_counter",
    "O 3.2 / G S1: cloud capacity forgeable": "v5.innovation.service_forges_its_compute_to_escape_capacity",
    "O 3.2 / P5: Tier 1 negative compute cost": "v5.rapid.program_forges_its_compute_cost",
    "O 3.2 / P6, G 5.1 A: adoption tracer counter forgeable": "v5.innovation.adoption_counter_written_by_dependent",
    "G 5.1 B: adoption by a shadow file under deps/": "v5.innovation.adoption_by_shadow_file_under_deps",
    "G 5.2: C builtins free under the line model": "v5.sandbox.c_builtins_are_billed",
    "O 3.4/3.5: FFmpeg/SoX/Chromium on agent bytes outside the sandbox": "v5.network.jailed_media_tools_cannot_reach_network_or_host_files",
    "O 3.5: agent JavaScript in a root Chromium without sandbox": "v5.network.agent_javascript_in_jailed_browser_cannot_exfiltrate",
    "O 3.6: static guard 4.0-only; calibration downloader reachable": "v5.isolation.world_code_cannot_import_evaluator_side_code",
    "F 3.7(a): evaluator MRR readable through charter/commitment predicates": "v5.commerce.evaluator_mrr_readable_through_charter",
    "O 3.10 / P7: following one's own organisation counted": "v5.social.own_organisations_inflate_followers",
    "C 3.10: sybil engagement not filtered": "v5.social.sybil_ring_engagement",
    "C skeptic 4: tracebacks expose host paths": "v5.sandbox.errors_do_not_disclose_host_paths",
    # wave 3 (workstream O, on the integrated tree)
    "O wave 3: module-level work ran unbilled at load time (up to 250 ms per cold start)":
        "v5.sandbox.meter_forgery.work_at_import_time",
    "O wave 3: nondeterministic output credited every locked package as adopted":
        "v5.innovation.adoption_by_nondeterministic_output",
}

# open findings of the ratchet closed by their owner workstreams since wave 1, re-verified by the probe on the
# integrated tree (each probe holds and its evidence shows the attack reached the mechanism it targets)
FIXED_BY_OWNERS: dict[str, str] = {
    "F 3.x / K.3: a 99% refund kept full ARR (F)": "v5.commerce.partial_refund_keeps_arr",
    "F: commit milestone payouts not netted (F)": "v5.commerce.circular_revenue_through_commitment_escrow",
    "F: inst project rewards not netted (F)": "v5.commerce.circular_revenue_through_project_reward",
    "F: a 45% rebate kept full MRR (F)": "v5.commerce.rebate_below_flag_threshold",
    "F P6: owner-funded ratings set the public rating (F)": "v5.commerce.owner_funded_five_star_ratings",
    "O 3.13 / P15: legacy reward was the default (L)": "v5.experience.observation_only_is_the_default",
    "O 3.12: closed worlds' bytes stayed in the artifact pool (A)": "v5.crossworld.closed_worlds_release_their_bytes",
}


def catalogue() -> dict[str, dict]:
    from . import probes
    return {p.name: {"gates": tuple(p.gates), "owner": p.owner, "target": p.target} for p in probes()}


def _severity(r: dict) -> str:
    return (r.get("severity") or ("high" if r["result"] == ERROR else "")).lower()


def gate_metrics(report: dict, cat: dict | None = None) -> dict[str, dict]:
    cat = cat or catalogue()
    rows = report["probes"]
    out: dict[str, dict] = {}
    for g in GATES:
        mine = [r for r in rows if g in cat.get(r["name"], {}).get("gates", ())]
        ran = [r for r in mine if r["result"] not in (NOT_APPLICABLE, ERROR)]
        bad = [r for r in mine if r["result"] in FINDINGS or r["result"] == ERROR]
        m = {"probes": len(mine), "probes_run": len(ran), "not_applicable": sum(r["result"] == NOT_APPLICABLE for r in mine),
             "errors": sum(r["result"] == ERROR for r in mine), "findings": len(bad),
             "finding_names": ",".join(sorted(r["name"] for r in bad))}
        if g == "W.1":
            passed = 0
            esc = 0
            for r in mine:
                ev = r.get("evidence") or {}
                if r["name"] == "v5.sandbox.escape_matrix":
                    passed += int(ev.get("sandbox_tests_passed") or 0)
                    esc += int(ev.get("sandbox_escapes") or 0)
                elif r["result"] == "held":
                    passed += 1
            m.update(sandbox_tests_passed=passed, sandbox_escapes=esc + len(bad))
        elif g == "W.2":
            m.update(evaluator_leaks=sum(1 for r in bad if _severity(r) not in ("low", "info")),
                     evaluator_isolation_low_findings=sum(1 for r in bad if _severity(r) in ("low", "info")))
        elif g == "W.3":
            m.update(reward_hack_probes=len(ran),
                     open_critical_high_reward_hacks=sum(1 for r in bad if _severity(r) in ("critical", "high")))
        elif g == "W.4":
            m.update(metric_manipulation_probes=len(ran), open_metric_manipulations=len(bad))
        elif g == "W.5":
            m.update(cross_world_leaks=len(bad))
        elif g == "W.6":
            m.update(accounting_violations=len(bad))
        elif g == "W.7":
            m.update(resource_limit_escapes=len(bad))
        elif g == "W.8":
            m.update(network_escapes=len(bad) + sum(int((r.get("evidence") or {}).get("network_escapes") or 0)
                                                    for r in mine if r["result"] == "held"))
        elif g == "W.9":
            m.update(private_state_leaks=len(bad))
        out[g] = m
    return out


def ratchet(report: dict) -> dict:
    """Unexpected findings (not in KNOWN_OPEN, or in MUST_HOLD) and fixed known-open entries (now held)."""
    rows = {r["name"]: r for r in report["probes"]}
    unexpected = sorted(n for n, r in rows.items() if (r["result"] in FINDINGS or r["result"] == ERROR)
                        and (n not in KNOWN_OPEN or n in MUST_HOLD))
    fixed = sorted(n for n in KNOWN_OPEN if n in rows and rows[n]["result"] == "held")
    changed = sorted(n for n in KNOWN_OPEN if n in rows and rows[n]["result"] in FINDINGS
                     and rows[n]["result"] != KNOWN_OPEN[n]["outcome"])
    return {"unexpected_findings": unexpected, "fixed_known_open": fixed, "known_open_outcome_changed": changed,
            "known_open_by_severity": {s: sorted(n for n, k in KNOWN_OPEN.items() if k["severity"] == s)
                                       for s in ("critical", "high", "medium", "low")}}


def legacy_gates() -> list:
    """The Crucible 4.0 catalogue still runs (its own ratchet is tests/test_redteam.py)."""
    from ..suite import all_probes
    return all_probes()
