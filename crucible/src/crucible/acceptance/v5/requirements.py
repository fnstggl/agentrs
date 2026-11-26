"""Every mandatory Crucible 5.0 requirement, with the machine-checkable criteria its evidence must meet.

Gate requirements `<G>.<n>` are the "Required:" bullets of MASTER_SPEC.md Part 17, read from the spec at build time so
their wording cannot drift. `<G>.n<k>` are the gate's further mandatory sentences. `SP-...` are mandatory requirements
stated elsewhere in the spec. Clarification 1 (MASTER_SPEC_CLARIFICATIONS.md) replaces every CAM-specific
requirement with an architecture-independent one; those entries carry `clarified=True` and their new wording.

A criterion is (metric, op, value). The evidence for a requirement must report every metric; the checker evaluates
the comparison itself. Ops: >=, <=, >, <, ==, !=.
"""
from __future__ import annotations

# workstream letters (WORKSTREAMS.md): A core, B population, C social, D semantic, E calibration, F economy,
# G innovation, H rapid, I domains, J civilization, K multi-agent, L learner interface, M performance, N research,
# O security, P quality assurance

C = tuple  # a criterion

# SHA-256 of MASTER_SPEC.md as saved verbatim from the request (commit 4a7591e, unchanged through a881d18); the
# checker refuses to derive or verify anything against a different specification.
MASTER_SPEC_SHA256 = "7b35feb343ce7fdd9b50e4e485b3a10110c3e07e6a6ce6124ae0b66826c45690"

# Per-requirement reproduction tolerances {rid: {metric: {"abs": a, "rel": r}}} used when a review re-executes the
# tests. Default: exact for integers, booleans and strings; floats within max(1e-9, 1e-6 * |value|). Change-controlled.
TOLERANCES: dict = {}


def crit(*pairs):
    return [{"metric": m, "op": o, "value": v} for m, o, v in pairs]


# (id, workstream, depends on gates/ids, criteria, override text or None, clarified)
GATE_ITEMS = {
    "A": ("P", [], [
        # D-23: exact baseline node ids, zero skips (run with the GPU visible), 4.0 test files untouched
        crit(("tests_failed", "==", 0), ("tests_errors", "==", 0), ("tests_total", ">=", 1076), ("tests_skipped", "==", 0),
             ("baseline_node_ids_checked", ">=", 1076), ("baseline_node_ids_missing", "==", 0),
             ("baseline_node_ids_not_passed", "==", 0), ("v4_test_files_modified", "==", 0)),
        # D-23: all 109 4.0 definitions (74 world, 25 mechanism, 10 useful-work families) by generated content
        crit(("v4_definitions_changed", "==", 0), ("v4_definitions_checked", ">=", 109),
             ("v4_world_families_checked", ">=", 74), ("v4_mechanism_families_checked", ">=", 25),
             ("v4_useful_work_families_checked", ">=", 10), ("v4_seeds_per_definition", ">=", 2),
             ("v4_definition_generation_errors", "==", 0)),
        # D-23: every 4.0 world family on two seeds
        crit(("v4_families_failed", "==", 0), ("v4_families_run", ">=", 74), ("v4_seeds_per_family", ">=", 2)),
        # D-23: static and runtime isolation, access log, store metadata against the committed baseline
        crit(("sealed_leak_findings", "==", 0), ("sealed_sets_changed", "==", 0), ("sealed_files_checked", ">=", 3),
             ("sealed_access_entries_after_v5_start", "==", 0), ("v5_static_references_to_sealed_store", "==", 0),
             ("v5_runtime_sealed_opens", "==", 0), ("v5_files_scanned", ">=", 100), ("v5_workload_steps", ">=", 24)),
        # D-23: every 4.0 acceptance record A-P and the deterministic demos, plus 4.0 world trajectories
        crit(("reproduction_mismatches", "==", 0), ("reproduced_results", ">=", 18),
             ("reproduction_candidates_not_attempted", "==", 0), ("v4_trajectories_compared", ">=", 100),
             ("v4_trajectory_mismatches", "==", 0)),
    ]),
    "B": ("B", ["SP-2.3", "SP-3.1"], [
        crit(("distinct_participants", ">=", 10000)),
        crit(("learner_initiated_consumer_events", "==", 0), ("autonomous_sessions", ">=", 100000)),
        crit(("preference_effect_pvalue", "<=", 0.01), ("preference_effect_size", ">=", 0.05)),
        crit(("return_gap_useful_vs_unsatisfactory", ">=", 0.05), ("abandon_gap_unsatisfactory_vs_useful", ">=", 0.05),
             ("return_pvalue", "<=", 0.01), ("abandon_pvalue", "<=", 0.01)),
        crit(("simulated_days", ">=", 30), ("days_with_activity", ">=", 30), ("last_week_over_first_week_activity", ">=", 0.5)),
        crit(("intervention_effect_pvalue", "<=", 0.01), ("intervention_seeds", ">=", 3)),
    ]),
    "C": ("C", ["B", "E"], [
        crit(("published_text", ">=", 1), ("published_image", ">=", 1), ("published_audio", ">=", 1), ("published_video", ">=", 1)),
        crit(("discovery_channels_used", ">=", 3)),
        crit(("consumption_events", ">=", 1000), ("analytics_mismatches", "==", 0)),
        crit(("communities_formed", ">=", 3), ("communities_active_after_30_days", ">=", 1)),
        crit(("follow_effect_pvalue", "<=", 0.01), ("share_effect_pvalue", "<=", 0.01), ("subscription_effect_pvalue", "<=", 0.01)),
        crit(("browser_workflow_steps_passed", ">=", 5), ("browser_state_changes_verified", ">=", 1)),
    ]),
    "D": ("C", ["C"], [
        crit(("distinct_recommenders_end_to_end", ">=", 4), ("max_pairwise_exposure_overlap", "<=", 0.8)),
        crit(("recommender_exposure_difference_pvalue", "<=", 0.01), ("recommender_consumption_difference_pvalue", "<=", 0.01)),
        crit(("algorithm_changes_during_run", ">=", 1), ("post_change_exposure_shift_pvalue", "<=", 0.01)),
        crit(("hidden_parameter_probes", ">=", 5), ("hidden_parameter_leaks", "==", 0)),
        crit(("exposure_invariant_violations", "==", 0), ("attention_overspend_events", "==", 0)),
    ]),
    "E": ("D", [], [
        crit(("real_encoder_modalities", ">=", 4), ("stub_vectors_detected", "==", 0), ("min_encoder_params", ">=", 1000000)),
        crit(("min_modality_separation_ratio", ">=", 1.5)),
        crit(("artifact_change_behavior_pvalue", "<=", 0.01)),
        crit(("cache_invalidation_failures", "==", 0), ("cache_hits", ">=", 1), ("recomputes_after_change", ">=", 1)),
        crit(("creator_assigned_quality_fields_read", "==", 0)),
    ]),
    "F": ("E", [], [
        crit(("empirical_domains_fitted", ">=", 3)),
        crit(("content_consumption_fitted", "==", True), ("randomized_commercial_response_fitted", "==", True)),
        crit(("heldout_reports", ">=", 3)),
        crit(("baseline_comparisons", ">=", 3)),
        crit(("calibration_metric_reports", ">=", 3)),
        crit(("checkpoints_saved", ">=", 3), ("checkpoint_hash_mismatches", "==", 0)),
        crit(("datasets_with_checksums", ">=", 3), ("provenance_records", ">=", 3)),
        crit(("extrapolation_limit_sections", ">=", 3)),
    ]),
    "G": ("B", ["B"], [
        crit(("simultaneously_active_adaptive", ">=", 256)),
        crit(("distinct_personal_states", ">=", 256)),
        crit(("shared_weight_copies", "==", 1), ("max_inference_batch", ">=", 64)),
        crit(("independent_strategic_groups", ">=", 2), ("distinct_group_weight_hashes", ">=", 2)),
        crit(("weight_hash_changed", "==", True), ("optimizer_updates", ">=", 100), ("loss_decrease", ">", 0)),
        crit(("new_condition_improvement_ci_low", ">", 0)),
        crit(("compact_vs_language_comparison_rows", ">=", 2)),
    ]),
    "H": ("D", ["L"], [
        crit(("semantic_participants", ">=", 8)),
        crit(("model_weight_copies", "==", 1), ("participants_sharing_weights", ">=", 8)),
        crit(("distinct_goals", ">=", 8), ("distinct_histories", ">=", 8)),
        crit(("tool_selection_accuracy", ">=", 0.6), ("tool_selection_vs_random_pvalue", "<=", 0.01)),
        crit(("comprehension_accuracy", ">=", 0.6), ("comprehension_vs_chance_pvalue", "<=", 0.01)),
        crit(("consequential_actions_executed", ">=", 8)),
        crit(("verified_productive_results", ">=", 1)),
        crit(("peak_vram_mb_measured", "==", True), ("tokens_per_second_measured", "==", True)),
    ]),
    "I": ("C", ["C", "M"], [
        crit(("distinct_identities", ">=", 10000000)),
        crit(("exposure_recount_mismatches", "==", 0), ("exposures_checked", ">=", 1000000)),
        crit(("distinct_viewer_recount_mismatches", "==", 0), ("distinct_viewers_checked", ">=", 1000000)),
        crit(("attention_budget_violations", "==", 0)),
        crit(("forced_view_assignments", "==", 0), ("share_generations_observed", ">=", 3)),
        crit(("positive_control_unique_viewers", ">=", 1000000)),
        crit(("qualifying_distinct_consumption_events", ">=", 10000000)),
    ]),
    "J": ("G", ["C", "F"], [
        crit(("agent_created_executable_products", ">=", 2)),
        crit(("published_products", ">=", 2)),
        crit(("independent_discoverers", ">=", 10)),
        crit(("independent_users", ">=", 10)),
        crit(("measured_behavior_adoption_pvalue", "<=", 0.05)),
        crit(("independent_transactions", ">=", 1)),
        crit(("resource_consumption_delta", ">", 0)),
    ]),
    "K": ("F", [], [
        crit(("subscription_scenarios_passed", ">=", 3)),
        crit(("cancellation_scenarios_passed", ">=", 3)),
        crit(("refund_scenarios_passed", ">=", 3)),
        crit(("related_party_exclusion_errors", "==", 0), ("independent_customers_counted", ">=", 10)),
        crit(("mrr_arr_recompute_mismatches", "==", 0), ("arr_equals_12_mrr", "==", True)),
        crit(("profit_reconciliation_error_cents", "==", 0)),
        crit(("planted_circular_detected_fraction", "==", 1.0), ("false_positive_rate", "<=", 0.05)),
    ]),
    "L": ("G", ["G", "J"], [
        crit(("capability_effects_measured", ">=", 1)),
        crit(("emerged_needs", ">=", 1)),
        crit(("needs_addressed_by_new_services", ">=", 1)),
        crit(("independent_chains", ">=", 3), ("min_chain_stages", ">=", 3)),
        crit(("counterfactual_removals_confirming", ">=", 3)),
    ]),
    "M": ("H", [], [
        crit(("admitted_train_families", ">=", 60)),
        crit(("eval_families", ">=", 20)),
        crit(("capability_categories_covered", ">=", 10)),
        crit(("independent_engines", ">=", 6)),
        crit(("families_missing_fingerprint", "==", 0)),
        crit(("action_modalities", ">=", 5)),
        crit(("families_feasibility_validated", ">=", 80), ("feasibility_failures_admitted", "==", 0)),
        crit(("families_with_intervention_tests", ">=", 80), ("intervention_test_failures", "==", 0)),
    ]),
    "N": ("I", ["F", "G", "C"], [
        crit(("domains_implemented", ">=", 15)),
        crit(("min_train_configs_per_domain", ">=", 3)),
        crit(("domains_with_executable_mechanisms", ">=", 15), ("missing_required_mechanics", "==", 0)),
        crit(("domains_with_independent_evaluator", ">=", 15)),
        crit(("domains_with_successful_reference", ">=", 15)),
        crit(("min_seeds_per_config", ">=", 3)),
        crit(("domains_with_compositional_eval", ">=", 15), ("domains_with_structural_eval", ">=", 15)),
    ]),
    "O": ("J", ["N"], [
        crit(("subsystems_in_one_world", ">=", 17)),
        crit(("duplicate_shared_facts", "==", 0), ("identity_consistency_violations", "==", 0)),
        crit(("ledger_invariant_violations", "==", 0)),
        crit(("cross_system_event_types", ">=", 10)),
        crit(("cross_domain_effect_types", ">=", 6)),
        crit(("multi_stage_chains", ">=", 3), ("min_chain_stages", ">=", 3)),
        crit(("structurally_distinct_profiles", ">=", 3)),
    ]),
    "P": ("J", ["O"], [
        crit(("runs_30_days", ">=", 2), ("runs_90_days", ">=", 2), ("runs_180_days", ">=", 2), ("runs_365_days", ">=", 2),
             ("failed_runs", "==", 0)),
        crit(("unexplained_invariant_violations", "==", 0)),
        crit(("snapshot_resume_mismatches", "==", 0)),
        crit(("replay_mismatches", "==", 0)),
        crit(("event_accounting_mismatches", "==", 0)),
        crit(("min_late_over_early_economic_activity", ">=", 0.3), ("min_late_over_early_social_activity", ">=", 0.3)),
    ]),
    "Q": ("K", ["O"], [
        crit(("learned_policies_in_civilization", ">=", 2)),
        crit(("independent_policy_groups", ">=", 2)),
        crit(("cooperative_interactions", ">=", 1), ("competitive_interactions", ">=", 1)),
        crit(("shared_world_consequence_events", ">=", 1)),
        crit(("policy_versions_recorded", ">=", 4)),
        crit(("heldout_opponent_evaluations", ">=", 1)),
        crit(("weight_updates", ">=", 100), ("weight_hash_changed", "==", True)),
        crit(("collusion_checks_run", ">=", 1), ("planted_collusion_undetected", "==", 0)),
    ]),
    "R": ("L", [], [
        crit(("nonzero_native_reward_steps", "==", 0)),
        crit(("privileged_progress_fields", "==", 0)),
        crit(("hidden_evaluator_fields", "==", 0), ("canaries_planted", ">=", 5)),
        crit(("legitimate_fields_missing", "==", 0)),
        crit(("verifier_outcomes_recorded", ">=", 1)),
        crit(("shaped_baseline_runs", ">=", 1)),
        crit(("reference_learner_updates_observation_only", ">=", 1)),
    ]),
    "S": ("L", ["R"], [
        crit(("tier1_actions", ">=", 100), ("tier1_world_changes", ">=", 1)),
        crit(("tier2_actions", ">=", 100), ("tier2_world_changes", ">=", 1)),
        crit(("tier3_actions", ">=", 100), ("tier3_world_changes", ">=", 1)),
        crit(("experience_records", ">=", 1000), ("learner_parameter_updates", ">=", 1)),
        crit(("schema_validation_failures", "==", 0), ("world_types_one_interface", ">=", 3)),
        crit(("checkpoint_roundtrip_mismatches", "==", 0)),
        crit(("parallel_environments", ">=", 4)),
        crit(("reproducible_commands_verified", "==", True)),
    ]),
    "T": ("N", ["M"], [
        crit(("distinct_generated_structures", ">=", 50)),
        crit(("population_configurations_tested", ">=", 3)),
        crit(("objective_types", ">=", 5)),
        crit(("generated_worlds_validated", ">=", 50)),
        crit(("weakness_selection_pvalue", "<=", 0.05)),
        crit(("random_component_share", ">=", 0.2)),
        crit(("evaluation_partition_leaks", "==", 0)),
        crit(("targeted_weakness_improvement_pvalue", "<=", 0.05)),
    ]),
    "U": ("N", ["L", "M"], [
        crit(("executable_levels", ">=", 5)),
        crit(("level6_disabled_by_default", "==", True), ("level6_requires_authorization", "==", True)),
        crit(("fresh_process_weights_only_checks", ">=", 9)),
        crit(("positive_control_kinds_classified_correctly", ">=", 5)),
        crit(("matched_budget_modes", ">=", 2)),
        crit(("learning_curve_points", ">=", 7)),
        crit(("independent_evaluator_used", "==", True)),
        crit(("intervals_account_for_worlds_and_seeds", "==", True), ("learner_seeds", ">=", 3)),
    ]),
    "V": ("M", ["B", "C", "D"], [
        crit(("profiles_benchmarked", ">=", 3)),
        crit(("identities_in_scale_test", ">=", 10000000)),
        crit(("model_parameter_counts_reported", ">=", 3)),
        crit(("batched_inference_speedup", ">=", 2.0)),
        crit(("event_driven_speedup", ">=", 2.0)),
        crit(("embedding_cache_hit_rate", ">=", 0.9)),
        crit(("peak_cpu_measured", "==", True), ("peak_ram_measured", "==", True), ("peak_vram_measured", "==", True)),
        crit(("compute_budget_violations", "==", 0)),
        crit(("approximation_error_metrics", ">=", 7)),
    ]),
    "W": ("O", [], [
        crit(("sandbox_tests_passed", ">=", 10), ("sandbox_escapes", "==", 0)),
        crit(("evaluator_leaks", "==", 0)),
        crit(("reward_hack_probes", ">=", 20), ("open_critical_high_reward_hacks", "==", 0)),
        crit(("open_metric_manipulations", "==", 0), ("metric_manipulation_probes", ">=", 5)),
        crit(("cross_world_leaks", "==", 0)),
        crit(("accounting_violations", "==", 0)),
        crit(("resource_limit_escapes", "==", 0)),
        crit(("network_escapes", "==", 0)),
        crit(("private_state_leaks", "==", 0)),
    ]),
    "X": ("P", ["*"], [
        # D-23: every X criterion gets the denominator that makes its zero meaningful
        crit(("unverified_requirements", "==", 0), ("requirements_counted", ">=", 340),
             ("manifest_anchored", "==", True)),
        crit(("required_tests_not_executed", "==", 0), ("required_tests_collected", ">=", 2000),
             ("required_test_files_missing", "==", 0), ("required_tests_skipped", "==", 0)),
        crit(("clean_checkout_failures", "==", 0), ("clean_checkout_runs", ">=", 1),
             ("verified_requirements_not_reproduced", "==", 0), ("documented_commands_run", ">=", 3)),
        crit(("integration_branch_test_failures", "==", 0), ("integration_branch_tests_run", ">=", 2000),
             ("integration_branch_test_errors", "==", 0), ("integration_branch_tests_skipped", "==", 0)),
        crit(("missing_documents", "==", 0), ("required_documents_checked", ">=", 30),
             ("documents_missing_required_content", "==", 0)),
        crit(("open_critical_high_defects", "==", 0), ("defect_register_present", "==", True),
             ("defects_registered", ">=", 1), ("audit_critical_findings_unregistered", "==", 0),
             ("defects_malformed", "==", 0)),
        crit(("placeholder_or_skipped_acceptance_tests", "==", 0), ("acceptance_test_files_scanned", ">=", 39),
             ("acceptance_tests_scanned", ">=", 120), ("manifest_ids_without_recording_test", "==", 0)),
        crit(("disconnected_workstreams", "==", 0), ("workstreams_checked", ">=", 16),
             ("integrated_world_runs", ">=", 1)),
        crit(("report_claims_without_evidence", "==", 0), ("final_report_present", "==", True),
             ("report_claims_checked", ">=", 50), ("report_claims_mismatching_evidence", "==", 0)),
    ]),
}

# Clarification 1: architecture-independent wording for the CAM-specific gate items.
CLARIFIED = {
    "R.7": "A lightweight reference learner (not CAM) consumes experience and performs parameter updates through the "
           "observation-only interface.",
    "S.1": "An architecture-independent reference learner executes real actions in Tier 1.",
    "S.2": "An architecture-independent reference learner executes real actions in Tier 2.",
    "S.3": "An architecture-independent reference learner executes real actions in Tier 3.",
    "S.4": "Actual experience storage and parameter updates of a lightweight reference learner.",
    "S.5": "Observation and action schemas shared across Tier 1, 2 and 3 through one learner interface.",
    "S.6": "Checkpoint persistence of the reference learner.",
    "S.7": "Parallel environment support.",
    "S.8": "Reproducible commands.",
}
GATE_TITLE_OVERRIDE = {"S": "Architecture-independent learner integration (replaces Actual CAM integration, Clarification 1)"}

# mandatory sentences that follow a gate's bullets
GATE_NOTES = [
    ("C.n1", "C", "No direct counter mutation.", "C", crit(("direct_counter_writes", "==", 0), ("counter_write_probes_blocked", ">=", 3))),
    ("E.n1", "E", "Use real model execution, not stub vectors.", "D", crit(("stub_vectors_detected", "==", 0), ("model_forward_passes", ">=", 4))),
    ("F.n1", "F", "Synthetic-only training does not pass this gate.", "E", crit(("synthetic_only_domains", "==", 0))),
    ("G.n1", "G", "The gate must establish actual adaptation, not merely the presence of neural network classes.", "B",
     crit(("adaptation_effect_ci_low", ">", 0))),
    ("H.n1", "H", "The standard implementation should use the smallest effective models, targeting approximately 0.5-1B "
     "parameters or smaller.", "D", crit(("semantic_model_params", "<=", 1100000000), ("smaller_alternatives_benchmarked", ">=", 2))),
    ("I.n1", "I", "Separately demonstrate that large adoption can arise through ordinary world mechanisms without forced "
     "view assignment.", "C", crit(("organic_cascade_unique_viewers", ">=", 100000), ("forced_view_assignments", "==", 0))),
    ("I.n2", "I", "Do not claim empirically realistic ten-million-view viral dynamics unless validated evidence supports "
     "that claim.", "N", crit(("unsupported_realism_claims", "==", 0))),
    ("J.n1", "J", "Demonstrate more than one product category.", "G", crit(("product_categories", ">=", 2))),
    ("K.n1", "K", "Demonstrate a reference strategy achieving a feasible recurring-revenue objective in a generated world.",
     "F", crit(("reference_objective_met", "==", True), ("reference_arr_usd", ">=", 100000))),
    ("K.n2", "K", "Do not treat that reference strategy as evidence that a learner learned to achieve it.", "N",
     crit(("reference_results_labelled_scripted", "==", True))),
    ("L.n1", "L", "No special-case success bonuses.", "G", crit(("special_case_bonuses", "==", 0))),
    ("M.n1", "M", "A renamed family or parameter variant does not count.", "H",
     crit(("cosmetic_variants_counted", "==", 0), ("min_fingerprint_distance", ">", 0))),
    ("N.n1", "N", "Reference trajectories must not be exposed as required solution paths.", "I", crit(("reference_paths_exposed", "==", 0))),
    ("O.n1", "O", "Disconnected mini-environments behind a single dispatcher do not pass.", "J",
     crit(("modules_without_shared_state", "==", 0))),
    ("P.n1", "P", "Use multiple independent seeds.", "J", crit(("independent_seeds_per_horizon", ">=", 2))),
    ("Q.n1", "Q", "Numeric facility-only self-play does not pass.", "K", crit(("civilization_selfplay_runs", ">=", 1))),
    ("R.n1", "R", "Run explicit reward-leakage tests.", "L", crit(("reward_leakage_tests_passed", ">=", 5))),
    ("S.n1", "S", "This gate verifies integration, not the scientific success of any learning algorithm.", "N",
     crit(("integration_claimed_as_learning_success", "==", 0))),
    ("V.n1", "V", "Before major benchmarks, establish machine-specific throughput targets based on measured hardware; do not "
     "weaken them after seeing results without change control.", "M",
     crit(("targets_recorded_before_benchmark", "==", True), ("throughput_targets_met", "==", True))),
    ("W.n1", "W", "Conduct independent adversarial review.", "O", crit(("independent_adversarial_reviews", ">=", 1))),
    ("X.n1", "X", "The independent QA agent must run this gate after all implementation workstreams have been integrated.",
     "P", crit(("run_after_full_integration", "==", True), ("unmerged_workstream_branches", "==", 0))),
]

# mandatory requirements stated outside Part 17
SPEC_ITEMS = [
    ("SP-1.3", "Persistent engineering documents exist (MASTER_SPEC unchanged, ARCHITECTURE, INTERFACE_CONTRACTS, "
     "WORKSTREAMS, DECISION_LOG, PROGRESS, BLOCKERS, FIDELITY_REGISTER, ACCEPTANCE_MANIFEST, REPRODUCE).", "P", [],
     crit(("v5_documents_present", ">=", 10), ("master_spec_unchanged", "==", True),
          ("v5_documents_missing", "==", 0), ("master_spec_sha256_matches_anchor", "==", True),
          ("stale_integration_branch_references", "==", 0))),
    ("SP-1.5", "Milestone state is persisted (branch, commit, workstreams, failures, remaining requirements, next actions, "
     "compute).", "P", [], crit(("milestones_recorded", ">=", 1), ("ledger_valid", "==", True),
                                 ("milestones_missing_fields", "==", 0),
                                 ("latest_milestone_omits_unverified_requirements", "==", 0))),
    ("SP-2.1", "Three execution modes: controlled experimental environments, standalone semantic worlds, integrated "
     "civilizations.", "A", [], crit(("execution_modes", ">=", 3))),
    ("SP-2.3", "Shared world substrate: common typed primitives, no duplicate shared facts, cross-module transactional "
     "integrity, recorded neural inference replayable without rerunning it.", "A", [],
     crit(("shared_primitive_kinds", ">=", 19), ("cross_module_rollback_failures", "==", 0),
          ("inference_replay_without_rerun", "==", True))),
    ("SP-3.1", "Persistent individual identity with independently evolving state; individuals respond differently to the "
     "same situation and their preferences evolve through experience.", "B", ["SP-2.3"],
     crit(("identity_state_fields", ">=", 18), ("same_situation_response_heterogeneity", ">", 0),
          ("preference_drift_from_experience", ">", 0))),
    ("SP-3.2", "Four participant classes: vectorised behavioural, small adaptive neural, strategic neural, compact semantic.",
     "B", ["SP-3.1"], crit(("participant_classes", ">=", 4))),
    ("SP-3.2b", "Semantic-model choice benchmarked: task-specific encoders/policies, distilled tens-hundreds of millions, "
     "~0.5-0.6B and ~1B models, on the eight listed capabilities.", "D", [],
     crit(("model_sizes_compared", ">=", 4), ("capabilities_benchmarked", ">=", 8))),
    ("SP-3.3", "Population profiles Development, Standard, Large and Internet scale with the specified sizes.", "B", [],
     crit(("profiles_matching_spec", ">=", 4))),
    ("SP-3.4", "Independent learning tested separately: personal-state adaptation, shared-policy learning, independent "
     "policy groups, frozen historical participants, competing architectures; with weight hashes, losses and "
     "behaviour changes.", "B", ["G"], crit(("learning_kinds_tested", ">=", 5))),
    ("SP-4", "Interoperable platforms: short and long video, text posts, images, articles/blogs, forums, social networks, "
     "communities, messaging, search, creator subscriptions, newsletters, product discovery, developer communities, "
     "knowledge sharing.", "C", ["B"], crit(("platform_kinds", ">=", 15))),
    ("SP-4.1", "Autonomous browsing behaviours (13 listed) with limited attention and time.", "B", ["SP-4"],
     crit(("browsing_behaviours", ">=", 13))),
    ("SP-4.2", "Artifacts of 10 kinds can be created, edited, rendered, published, updated, removed and inspected in the "
     "sandbox.", "C", [], crit(("artifact_kinds", ">=", 10), ("content_tools", ">=", 8))),
    ("SP-4.5", "Recommendation mechanisms: chronological, interest retrieval, collaborative filtering, learned engagement, "
     "exploration, social graph, diversity-aware, reputation-sensitive, search-driven, freshness; delayed and noisy "
     "feedback.", "C", ["SP-4"], crit(("recommendation_mechanisms", ">=", 10))),
    ("SP-4.6", "Audience accounting distinguishes impressions, unique exposed, views, unique viewers, watch duration, "
     "completion, repeats, shares, subscribers, returning viewers; documented view threshold.", "C", ["SP-4"],
     crit(("audience_quantities_distinguished", ">=", 10), ("view_threshold_documented", "==", True))),
    ("SP-4.7", "Communities with persistent membership, governance, moderation, trust, entry/exit, inactivity and "
     "participant-founded subjects.", "C", ["SP-4"], crit(("community_features", ">=", 10))),
    ("SP-4.8", "Social-economic interoperability through ordinary mechanisms (no social-to-revenue bonus).", "C", ["F"],
     crit(("social_to_purchase_chains_verified", ">=", 1), ("hardcoded_social_revenue_bonuses", "==", 0))),
    ("SP-5.4", "Fidelity record for every important behavioural mechanism, with evidence population, platform, period "
     "and limits.", "E", [], crit(("important_mechanisms_without_record", "==", 0))),
    ("SP-5.5", "Prespecified held-out metrics; learned models compared with simple baselines and the better one kept.",
     "E", [], crit(("metrics_prespecified_before_fit", "==", True), ("worse_model_kept", "==", 0))),
    ("SP-6.1", "Businesses and products: formation, development, subscriptions, one-time purchases, licensing, "
     "marketplaces, SaaS, education/media/dev-tool/data/infrastructure/industrial products, contracts, refunds, "
     "cancellations, support, competition, bankruptcy.", "F", [], crit(("business_features", ">=", 19))),
    ("SP-6.3", "General artifact-and-capability registry: a participant discovers a previously nonexistent service, tests "
     "it and adopts it if useful.", "G", [], crit(("novel_services_discovered_tested_adopted", ">=", 1))),
    ("SP-6.4", "Emerging needs from reusable activity-resource-capability-constraint-need relations, not special-case "
     "triggers.", "G", [], crit(("reusable_need_relations", ">=", 5), ("special_case_need_triggers", "==", 0))),
    ("SP-6.5", "Software ecosystem: hosting, packages, dependency graphs, issues, PRs, review, tests, releases, docs, "
     "deployment, adoption by execution, maintenance, forks, compatibility, failures.", "G", [],
     crit(("software_features", ">=", 15), ("adoptions_by_execution", ">=", 1))),
    ("SP-6.6", "Adaptable institutions within stable constraints (cannot alter accounting, security, evaluation).", "F", [],
     crit(("institution_features", ">=", 10), ("core_law_modifications_blocked", ">=", 3))),
    ("SP-7.1", "Versioned registry of the 28 composable causal primitives.", "H", [], crit(("primitives_registered", ">=", 28))),
    ("SP-7.3", "Eight new engine kinds: discrete-event, graph dynamical, search/information, executable program, "
     "multi-agent strategic, scientific experiment, planning/resource, partially observed state machine.", "H", [],
     crit(("new_engine_kinds", ">=", 8))),
    ("SP-7.4", "World admission checks (10 listed) with evaluator-side oracles only.", "H", [],
     crit(("admission_checks", ">=", 10), ("oracle_exposure_findings", "==", 0))),
    ("SP-8.11", "Tier 1 action interface covers graph edits, discrete, sequential, tool discovery, experiment design, "
     "queries, communication, programs, allocation, coordination.", "H", [], crit(("action_types", ">=", 10))),
    ("SP-8.12", "Adjustable difficulty along nine axes; fast simulator and kernel agree where claimed.", "H", [],
     crit(("difficulty_axes", ">=", 9), ("sim_kernel_mismatches", "==", 0))),
    ("SP-10.1", "Civilization profiles (10 listed) that differ structurally.", "J", ["O"],
     crit(("civilization_profiles", ">=", 10), ("profiles_differing_only_in_parameters", "==", 0))),
    ("SP-10.3", "Multi-order chains include one with an agent-created artifact, one with social activity, one with "
     "technical or scientific activity, each confirmed by counterfactual removal.", "J", ["L"],
     crit(("chains_with_agent_artifact", ">=", 1), ("chains_with_social_activity", ">=", 1),
          ("chains_with_technical_activity", ">=", 1))),
    ("SP-11.1", "Universal interaction contract (observe, execute, observe_since) across tiers carrying the 13 listed "
     "observation kinds with exact numbers.", "L", [], crit(("observation_kinds", ">=", 13), ("tiers_supported", ">=", 3))),
    ("SP-11.2", "Open-ended tool interface with discovery and schemas; agent-created services add discoverable "
     "affordances.", "L", [], crit(("tool_categories", ">=", 13), ("affordances_from_agent_services", ">=", 1))),
    ("SP-11.3", "Reward modes legacy_shaped_reward, verified_terminal_outcome and observation_only.", "L", [],
     crit(("reward_modes", ">=", 3))),
    ("SP-11.4", "Self-generated experiments: prediction, measurable outcome, deadline, execution, result retrieval, no "
     "reward for easy subgoals.", "L", [], crit(("self_experiment_roundtrips", ">=", 1), ("self_goal_rewards_paid", "==", 0))),
    ("SP-11.5", "Long-horizon experience: durations, delayed events, asynchronous workflows, long-running programs, "
     "concurrent projects, commitments, provenance.", "L", [], crit(("long_horizon_features", ">=", 7))),
    ("SP-11.6", "Documented path for future learners (including CAM) to connect without environment changes, proven by an "
     "external adapter that uses only the public interface.", "L", [],
     crit(("external_adapter_env_changes", "==", 0), ("connection_guide_present", "==", True))),
    ("SP-12.1", "Self-play population mixes current, historical, independent, statistical, compact neural, semantic and "
     "held-out policies.", "K", ["Q"], crit(("population_member_kinds", ">=", 7))),
    ("SP-12.3", "No hidden coordination; controlled-identity transfers are not counted as independent success.", "K", [],
     crit(("controlled_identity_transfers_counted", "==", 0))),
    ("SP-13.7", "Resource scheduler enforcing CPU, RAM, GPU memory, GPU time, disk, inference concurrency and active "
     "worlds.", "M", [], crit(("scheduler_limits_enforced", ">=", 7), ("scheduler_limit_violations", "==", 0))),
    ("SP-13.8", "Performance report with the 12 listed quantities and a Crucible 4.0 comparison.", "M", ["V"],
     crit(("performance_quantities_reported", ">=", 12), ("v4_comparison_rows", ">=", 1))),
    ("SP-14.1", "Versioned training-distribution registry across tiers with partitions and a sampler; evaluation reuse "
     "prevented.", "N", [], crit(("registry_kinds", ">=", 9), ("evaluation_reuse_prevented", "==", True))),
    ("SP-14.2", "Default tier mixture 50/30/20 with all tiers available.", "N", [], crit(("default_mixture_matches", "==", True))),
    ("SP-14.4", "World proposal interface validated for feasibility, diversity, safety, resources, difficulty and "
     "usefulness; cannot alter the evaluator or sealed distribution.", "N", [],
     crit(("proposal_checks", ">=", 6), ("evaluator_tamper_attempts_blocked", ">=", 2))),
    ("SP-15.3", "Learning-efficiency curves at 1-64 interactions and censoring-aware acceleration N_q ratios.", "N", [],
     crit(("censoring_handled", "==", True), ("curve_budgets", ">=", 7))),
    ("SP-15.5", "Executable A/B/C/D training-diversity comparison with matched experience and compute budgets and the "
     "seven listed controls.", "N", ["U"], crit(("comparison_arms", ">=", 4), ("control_kinds", ">=", 7))),
    ("SP-15.6", "Positive-control transfer experiment on newly added nontrivial families detected by the evaluator.", "N",
     ["M", "U"], crit(("positive_control_transfer_detected", "==", True))),
    ("SP-16", "Security boundaries: default isolation from the public internet; real-world adapter disabled by default; "
     "consent, moderation and abuse prevention.", "O", [],
     crit(("realworld_enabled_by_default", "==", False), ("public_network_reachable_from_world", "==", False))),
    ("SP-18", "Final report contains the Part 18 completion table populated from verified evidence.", "P", ["X"],
     crit(("completion_table_rows", ">=", 18), ("rows_without_evidence", "==", 0), ("completion_table_rows", "==", 18),
          ("capabilities_matching_spec", "==", 18), ("statuses_outside_vocabulary", "==", 0),
          ("verified_rows_with_unverified_requirements", "==", 0), ("rows_verified", "==", 18))),
    ("SP-20", "All 21 final deliverables are present.", "P", ["X"],
     crit(("deliverables_present", ">=", 21), ("deliverables_verified", "==", 21),
          ("report_separation_categories", "==", 9))),
    ("SP-21", "`python -m crucible.acceptance.v5 --verify-all` validates the manifest and evidence and exits nonzero "
     "whenever a requirement is not verified.", "P", [], crit(("checker_rejects_fabricated_evidence", "==", True),
                                                              ("checker_nonzero_when_unverified", "==", True),
                                                              ("fabrication_probes_run", ">=", 14),
                                                              ("fabrication_probes_reaching_verified", "==", 0),
                                                              ("project_statuses_supported", "==", 3),
                                                              ("manifest_anchored_to_derivation", "==", True))),
]

# Clarification 2 (observation-only as the universal default) and Clarification 3 (recovery preamble):
# requirements the user added on 2026-11-26. (id, gate, text, workstream, dependencies, criteria, clarification)
AMENDMENT_ITEMS = [
    ("OO.1", "OO", "Observation-only mode is enabled by default across every environment family (Tier 1 rapid "
     "families, Tier 2 domains, Tier 3 civilization profiles, and the 4.0 environments exposed through crucible.experience).",
     "L", ["R"], crit(("tiers_covered", "==", 3), ("env_families_checked", ">=", 100),
                      ("families_not_observation_only_by_default", "==", 0)), 2),
    ("OO.2", "OO", "No native or privileged reward signals reach the learner: no intermediate reward, shaping, "
     "action-quality score or goal-progress value through observations, auxiliary fields, tool responses or event "
     "metadata; Gymnasium scalar reward is 0.0.", "L", ["R"],
     crit(("reward_leak_probe_kinds", ">=", 8), ("reward_leaks_found", "==", 0),
          ("nonzero_scalar_rewards_default_mode", "==", 0)), 2),
    ("OO.3", "OO", "Real observable consequences remain accessible: transactions, customer behaviour, technical "
     "measurements, social responses, resource changes, failures and delayed outcomes.", "L", ["R"],
     crit(("consequence_kinds_observable", ">=", 7), ("tiers_covered", "==", 3)), 2),
    ("OO.4", "OO", "Independent outcome verification remains functional through a separate evaluator-side interface "
     "unreachable by the learner.", "L", ["R"],
     crit(("tiers_covered", "==", 3), ("verifier_matches_ground_truth", "==", True),
          ("verifier_reachable_from_learner", "==", False)), 2),
    ("OO.5", "OO", "Agents can operate indefinitely within configured resource limits without an assigned objective.",
     "L", ["R", "P"], crit(("objectiveless_tiers", "==", 3), ("objectiveless_civ_sim_days", ">=", 90),
                            ("resource_limit_violations", "==", 0)), 2),
    ("OO.6", "OO", "An architecture-neutral external learner can supply its own training rewards and update its "
     "parameters with no environment change.", "L", ["R"],
     crit(("tiers_covered", "==", 3), ("external_reward_param_updates", ">=", 3), ("env_changes_required", "==", 0)), 2),
    ("OO.7", "OO", "Delayed outcomes remain accessible for retrospective learning, linked to the originating actions by "
     "observable provenance.", "L", ["R"],
     crit(("delayed_outcomes_retrieved", ">=", 1), ("max_retrieved_delay_sim_days", ">=", 30),
          ("provenance_links_to_action", "==", True)), 2),
    ("OO.8", "OO", "Historical reward-based experiments remain reproducible through explicitly selected alternative modes "
     "(legacy_shaped_reward, verified_terminal_outcome).", "L", ["R"],
     crit(("legacy_results_reproduced", ">=", 2), ("legacy_mode_requires_explicit_selection", "==", True)), 2),
    ("RP.1", "RP", "Crucible is independent of CAM: no import of CAM code, and no acceptance gate depends on CAM's "
     "completion or success.", "P", [],
     crit(("cam_imports_in_crucible", "==", 0), ("gates_depending_on_cam", "==", 0),
          ("crucible_files_scanned", ">=", 500)), 3),
    ("RP.2", "RP", "An independent audit classified every subsystem (RETAIN/REPAIR/EXTEND/REDESIGN/REBUILD/IMPLEMENT) "
     "on evidence before implementation resumed.", "P", [],
     crit(("audit_reports", ">=", 15), ("audit_reports_with_skeptic_review", ">=", 15), ("audited_areas_missing", "==", 0),
          ("audit_reports_committed", ">=", 15)), 3),
    ("RP.3", "RP", "The progress ledger is maintained with the exact remaining requirements and next steps; milestones "
     "are committed and pushed to GitHub; no milestone work exists only in an uncommitted worktree.", "P", [],
     crit(("milestones_recorded", ">=", 3), ("integration_head_pushed", "==", True),
          ("ledger_lists_remaining_requirements", "==", True), ("unpushed_milestone_commits", "==", 0),
          ("merged_worktrees_with_uncommitted_changes", "==", 0), ("uncommitted_milestone_work", "==", 0)), 3),
    ("RP.4", "RP", "Important fitted models, checkpoints and experimental artifacts are preserved in durable storage.",
     "P", [], crit(("required_artifacts_unpreserved", "==", 0), ("required_artifacts_listed", ">=", 20),
                   ("artifact_hash_mismatches", "==", 0), ("artifacts_without_location", "==", 0)), 3),
]

# ======================================================================================================================
# Requirements found missing by the recovery audit (docs/v5/audit/*.json "missing_requirements", 2026-11-26), added
# under decision D-22. Each is a mandatory statement of MASTER_SPEC.md or of a clarification that had no entry of its
# own; `spec` names where it is stated and `audit` which audit found it missing.
# ======================================================================================================================
_AU = "docs/v5/audit/{}.json missing_requirements"


def _sp(rid, text, ws, deps, criteria, spec, audit):
    return (rid, text, ws, deps, criteria, {"spec": f"MASTER_SPEC.md {spec}", "audit": _AU.format(audit)})


_MECHANICS = {   # Part 9 "Required mechanics" per domain, verbatim
    "9.1": ("Startup economy", ["Product creation", "Demand discovery", "Independent customers", "Subscription billing",
                                "Distribution", "Competition", "Retention", "Costs", "Profitability",
                                "Business operations"]),
    "9.2": ("Software ecosystem", ["Executable code", "Repositories", "Dependencies", "Issues", "Independent testing",
                                   "Developer preferences", "Documentation", "Adoption", "Compatibility",
                                   "Maintenance"]),
    "9.3": ("Content ecosystem", ["Actual content artifacts", "Learned content representations",
                                  "Autonomous consumption", "Recommendation systems", "Competing creators",
                                  "Social sharing", "Audience feedback", "Distinct verified viewers"]),
    "9.4": ("Community ecosystem", ["Social networks", "Persistent memberships", "Independent participants",
                                    "Shared activities", "Communication", "Moderation", "Governance", "Retention",
                                    "Community evolution"]),
    "9.5": ("Scientific research", ["Experimental apparatus", "Hidden laws", "Measurement noise", "Experimental costs",
                                    "Data collection", "Hypothesis testing", "Independent replication",
                                    "Novel-condition evaluation"]),
    "9.6": ("Technology invention", ["Executable inventions", "Technical constraints", "Independent evaluation",
                                     "Adoption", "Compatibility", "Economic consequences",
                                     "Downstream resource changes"]),
    "9.7": ("Engineering infrastructure", ["Compute", "Networking", "Workloads", "Capacity", "Energy", "Failures",
                                           "Technical optimization", "Delayed effects",
                                           "Deployment of new controllers and software"]),
    "9.8": ("Industrial economy", ["Raw materials", "Production", "Energy", "Logistics", "Inventory", "Equipment",
                                   "Labor", "Market demand", "Resource scarcity", "Supply disruptions"]),
    "9.9": ("Labor marketplace", ["Worker skills", "Preferences", "Compensation", "Recruitment", "Delegation",
                                  "Project dependencies", "Performance", "Retention", "Independent agency"]),
    "9.10": ("Education ecosystem", ["Learner knowledge state", "Educational artifacts", "Prerequisites",
                                     "Participation", "Learning dynamics", "Independent assessments",
                                     "Retention of acquired knowledge"]),
    "9.11": ("Information ecosystem", ["Search", "Documents", "Knowledge artifacts", "Queries", "Source provenance",
                                       "Information quality", "Updating facts", "Independent user satisfaction",
                                       "Verified retrieval performance"]),
    "9.12": ("Organizational ecosystem", ["Organizations", "Differing goals", "Contracts", "Negotiation",
                                          "Coordination", "Shared resources", "Governance", "Commitments",
                                          "Collective outcomes"]),
    "9.13": ("Digital marketplace", ["Two-sided participation", "Discovery", "Transactions", "Trust", "Quality",
                                     "Disputes", "Pricing", "Network effects", "Retention"]),
    "9.14": ("Crisis recovery", ["Cascading disruptions", "Scarce resources", "Prioritization", "Coordination",
                                 "Recovery", "Resilience", "Independent service-restoration metrics"]),
    "9.15": ("Research commercialization", ["Research", "Experimentation", "Technical development", "Product creation",
                                            "Financing", "Deployment", "Independent adoption",
                                            "Commercial sustainability"]),
}
_DOMAIN_EXTRA = {
    "9.3": (" Include difficulty levels scaling toward 10 million simulated unique viewers.",
            [("difficulty_levels", ">=", 3), ("max_difficulty_unique_viewers", ">=", 10000000)]),
    "9.10": (" Measure actual improvement on new test problems rather than time spent viewing content alone, with "
             "content controls (correct content > no content > wrong content).",
             [("improvement_on_new_problems_pvalue", "<=", 0.05), ("content_control_order_holds", "==", True)]),
    "9.15": (" This domain must genuinely connect scientific discovery and economic outcomes (confirmed by "
             "counterfactual removal of the discovery).",
             [("discovery_to_economic_outcome_confirmed", "==", True)]),
}

AUDIT_SPEC_ITEMS = []
for _k, (_name, _mech) in _MECHANICS.items():
    _extra_text, _extra_crit = _DOMAIN_EXTRA.get(_k, ("", []))
    AUDIT_SPEC_ITEMS.append(_sp(
        f"SP-{_k}", f"Tier 2 domain {_name}: every required mechanic is implemented and exercised by recorded events "
        f"in a generated configuration ({'; '.join(_mech)}), with autonomous participants of the shared population, "
        f"observation_only by default and reachable through crucible.experience.make(2, ...).{_extra_text}",
        "I", ["N.1"], crit(("required_mechanics_listed", "==", len(_mech)),
                           ("required_mechanics_exercised", "==", len(_mech)),
                           ("required_mechanics_missing", "==", 0), ("observation_only_default", "==", True),
                           ("reachable_through_experience", "==", True), *_extra_crit),
        f"Part {_k}", "I_domains"))

AUDIT_SPEC_ITEMS += [
    # ---------------------------------------------------------------- Part 0
    _sp("SP-0.9", "Crucible 4.0 validation and sealed evaluation families are not moved into any v5 training "
        "distribution, checked structurally (mechanism fingerprints of every v5 training family against the 4.0 "
        "val/test families), not by name.", "H", [],
        crit(("v4_heldout_families_fingerprinted", ">=", 18), ("v5_train_families_checked", ">=", 60),
             ("v5_train_families_matching_v4_heldout", "==", 0)), "Part 0 (contamination) and D-9", "P_acceptance"),
    _sp("SP-0.10", "Baseline Crucible 4.0 throughput, memory consumption and world-generation cost were established "
        "before v5 changes and are reported against v5.", "M", [],
        crit(("baseline_quantities_recorded", ">=", 3), ("baseline_measured_on_v4_commit", "==", True)),
        "Part 0 item 10", "P_acceptance"),
    # ---------------------------------------------------------------- Part 1
    _sp("SP-1.2", "Actual compute is coordinated through a resource scheduler; expensive GPU jobs are not started "
        "simultaneously outside it.", "M", [],
        crit(("scheduler_usage_log_present", "==", True), ("gpu_jobs_outside_scheduler_observed", "==", 0)),
        "Part 1.2", "M_performance"),
    _sp("SP-1.3b", "Every manifest entry lists its implementation files, tests and implementation commit, and every "
        "workstream report is merged into the manifest.", "P", [],
        crit(("entries_without_implementation_files", "==", 0), ("entries_without_commit", "==", 0),
             ("workstream_reports_unmerged", "==", 0)), "Part 1.3 / 1.4", "P_acceptance"),
    _sp("SP-1.5b", "Work of sub-agents that terminated early is reassigned and tracked to completion.", "P", [],
        crit(("interrupted_workstreams_recorded", ">=", 13), ("interrupted_workstreams_unassigned", "==", 0)),
        "Part 1.5", "P_acceptance"),
    # ---------------------------------------------------------------- Part 2
    _sp("SP-2.1b", "A technological invention can change economic outcomes, participant behaviour, resource demand "
        "and future technological opportunities (each confirmed by counterfactual removal in an integrated world).",
        "J", ["O"], crit(("invention_effect_kinds_confirmed", "==", 4)), "Part 2.1", "A_J_substrate_civ"),
    _sp("SP-2.2", "Compositional freedom through the 13 listed affordances (executable code, dynamic services, tool "
        "discovery, data processing, communication, market transactions, organization creation, product creation, "
        "research, experimentation, long-running workflows, agent-created tools, agent-created websites and services); "
        "no predetermined menu of strategies or task-specific moves.", "L", [],
        crit(("affordances_composable", ">=", 13), ("predetermined_strategy_menus", "==", 0)), "Part 2.2",
        "P_acceptance"),
    _sp("SP-2.3b", "All systems communicate through typed events and approved state interfaces (no cross-module state "
        "writes), and simulation is deterministic wherever possible (same spec and seed give the same event hashes in "
        "every v5 world kind).", "A", ["SP-2.3"],
        crit(("cross_module_state_writes", "==", 0), ("world_kinds_checked_for_determinism", ">=", 3),
             ("determinism_mismatches", "==", 0)), "Part 2.3", "A_J_substrate_civ"),
    # ---------------------------------------------------------------- Part 3
    _sp("SP-3.2c", "Class 2 models have 100k-2M parameters and class 3 models 2-20M (measured); strategic groups "
        "compete, cooperate, pursue unrelated objectives and adapt over their lifetime; they are not identical copies "
        "of one policy.", "B", ["SP-3.2"],
        crit(("class2_params_in_range", "==", True), ("class3_params_in_range", "==", True),
             ("strategic_group_behaviour_kinds", ">=", 4), ("identical_policy_copies_across_groups", "==", 0)),
        "Part 3.2", "P_acceptance"),
    _sp("SP-3.2d", "Class 4 semantic agents are not language-model wrappers around predetermined scripts: they have "
        "their own needs and goals and meaningful opportunities to act; the smallest model or combination of models "
        "meeting each capability is used (routing); checkpoint availability, licences and tool-use capability are "
        "verified.", "D", ["H"],
        crit(("scripted_semantic_agent_paths", "==", 0), ("semantic_agents_with_own_goals", ">=", 8),
             ("capabilities_routed_to_smallest_model", "==", True), ("models_with_licence_and_tool_use_verified", ">=", 2)),
        "Part 3.2 class 4", "D_semantic"),
    _sp("SP-3.3b", "Every population profile including Internet scale is benchmarked; large profiles use resource-aware "
        "sparse or lazy state and batched inference.", "M", ["SP-3.3"],
        crit(("profiles_benchmarked", "==", 4), ("large_profiles_with_sparse_state", ">=", 2)), "Part 3.3",
        "M_performance"),
    _sp("SP-3.3c", "Semantic participants have persistent identity and state in the standard profile (bound to rows "
        "of the shared population).", "D", ["SP-3.3"],
        crit(("semantic_participants_bound_to_population", ">=", 8)), "Part 3.1 / 3.3", "D_semantic"),
    # ---------------------------------------------------------------- Part 4
    _sp("SP-4.0", "All platforms share the civilization's participant population and world state: one population "
        "(D-15) acts on every platform, with no second copy of any per-person fact.", "B", ["SP-4"],
        crit(("populations_in_v5_world", "==", 1), ("platforms_used_by_population", ">=", 15),
             ("platforms_without_population_activity", "==", 0), ("duplicate_person_facts", "==", 0)),
        "Part 4 (introduction)", "P_acceptance"),
    _sp("SP-4.1b", "Tool-based reading (content.open) by participants is recorded as consumption with analytics like "
        "feed consumption.", "C", ["SP-4"], crit(("tool_reads_recorded_as_consumption", "==", True),
                                              ("analytics_mismatches_for_tool_reads", "==", 0)), "Part 4.1 / 4.6",
        "C_social"),
    _sp("SP-4.2b", "Expensive generative models are not required for ordinary simulation actions.", "D", [],
        crit(("ordinary_actions_requiring_generation", "==", 0)), "Part 4.2", "P_acceptance"),
    _sp("SP-4.3", "Content representations let participants respond to meaning, not tokens: gibberish, keyword-stuffed "
        "and token-optimised content does not beat informative content on equal topics; products and software have "
        "structured representations; semantic predictions are validated and never the sole definition of value; "
        "distributed items are re-encoded when content is updated.", "D", ["E"],
        crit(("adversarial_content_kinds_probed", ">=", 3), ("adversarial_content_wins_over_informative", "==", 0),
             ("structured_representation_kinds", ">=", 2), ("value_defined_solely_by_semantic_prediction", "==", 0),
             ("stale_encodings_after_update", "==", 0)), "Part 4.3", "C_social"),
    _sp("SP-4.4", "Participants choose content according to the 12 listed factors (interests, relevance, previous "
        "experience, attention, content characteristics, social relationships, creator familiarity, platform "
        "distribution, novelty, satisfaction, competing content, current needs), learned from empirical data where "
        "possible; continuing, sharing, subscribing and disengaging are actual simulated decisions.", "B", ["F"],
        crit(("choice_factors_used", ">=", 12), ("decision_kinds_simulated", ">=", 4),
             ("fitted_models_in_default_consumption", ">=", 2)), "Part 4.4", "C_social"),
    _sp("SP-4.5b", "Popularity of one item reduces the opportunities of competing content; feedback is delayed, "
        "analytics noisy, audiences change, platform algorithms evolve endogenously and competitors adapt.", "C",
        ["SP-4.5"], crit(("attention_competition_effect_pvalue", "<=", 0.01), ("dynamic_kinds_observed", ">=", 5)),
        "Part 4.5", "C_social"),
    _sp("SP-4.6b", "No artificial virality multipliers (static and dynamic check); every view traces to a recorded "
        "decision of an individual (forced views detected by decision provenance).", "C", ["SP-4.6"],
        crit(("virality_multipliers_found", "==", 0), ("views_without_decision_provenance", "==", 0)), "Part 4.6",
        "C_social"),
    _sp("SP-4.7b", "Community membership, inactivity and churn of the shared population; community success measured "
        "by sustained voluntary participation.", "C", ["SP-4.7"],
        crit(("communities_with_population_members", ">=", 3), ("churn_events_observed", ">=", 1),
             ("sustained_participation_days", ">=", 30)), "Part 4.7", "C_social"),
    _sp("SP-4.8b", "Population-level social-to-economy chains including infrastructure cost, retention and demand "
        "effects.", "C", ["SP-4.8"], crit(("population_social_economy_chains", ">=", 3),
                                        ("chain_effect_kinds", ">=", 3)), "Part 4.8", "C_social"),
    # ---------------------------------------------------------------- Part 5
    _sp("SP-5.0", "Fitted behavioural models drive Class 1 participant decisions in the running integrated world: "
        "every runtime use claimed in the fidelity register is exercised, and decision events carry the model id and "
        "version (decision provenance).", "E", ["F"],
        crit(("fitted_models_used_in_world", ">=", 5), ("claimed_runtime_uses_unverified", "==", 0),
             ("decision_events_without_model_provenance", "==", 0)), "Parts 3.2, 4.4, 5", "E_calibration"),
    _sp("SP-5.1", "Calibration data are acquired legitimately with licences recorded, contain no re-identifiable "
        "identifiers, are schema-validated, use time-based and user-level holdouts, handle missing data, estimate "
        "uncertainty, and distinguish observational from experimentally identified effects.", "E", [],
        crit(("datasets_with_licence_record", ">=", 4), ("reidentifiable_ids_in_calibration_data", "==", 0),
             ("models_with_uncertainty", ">=", 5), ("models_with_time_or_user_holdout", ">=", 5),
             ("effects_without_identification_label", "==", 0)), "Parts 5.1 / 5.2", "E_calibration"),
    _sp("SP-5.3", "Fitted models are compared with the rule-based simulator behaviour they replace and selected on "
        "held-out fit, long-horizon stability and computational efficiency; attention allocation, repeat consumption, "
        "purchase, retention, community participation and social diffusion are each fitted or the absence justified.",
        "E", [], crit(("rule_based_comparisons", ">=", 3), ("selection_reports_with_stability_and_cost", ">=", 3),
                      ("behaviour_domains_fitted_or_justified", ">=", 6)), "Part 5.3", "E_calibration"),
    _sp("SP-5.5b", "In-world validation: activity distributions, retention curves, heterogeneity and long-horizon "
        "drift of the integrated simulator against held-out data; in-world randomised experiments (random exposure, "
        "advertising holdout) against the KuaiRand random-exposure and Criteo intention-to-treat estimates; world "
        "use stays inside the model cards' limits.", "E", ["SP-5.0"],
        crit(("in_world_validation_reports", ">=", 4), ("in_world_randomised_experiments_matching", ">=", 2),
             ("out_of_card_scope_uses", "==", 0)), "Parts 5.2 / 5.5", "E_calibration"),
    # ---------------------------------------------------------------- Part 6
    _sp("SP-6.1b", "Products provide measurable usefulness, entertainment or educational value: revenue from "
        "independent population customers tracks measured value (useless products earn about nothing).", "F", [],
        crit(("useless_product_revenue_share", "<=", 0.05), ("value_revenue_relation_pvalue", "<=", 0.01)),
        "Part 6.1", "F_commerce"),
    _sp("SP-6.2", "Genuine recurring revenue: discounts, customer acquisition cost, churn, operating costs and "
        "related-party distinctions are modelled; rolling 30-day sales are never counted as recurring revenue; "
        "subscriptions must stay valid; one system of record for recurring revenue across creators and businesses and "
        "one product catalogue.", "F", ["K"],
        crit(("revenue_features", ">=", 6), ("rolling_sales_counted_as_recurring", "==", 0),
             ("recurring_revenue_systems_of_record", "==", 1), ("product_catalogues", "==", 1)), "Part 6.2",
        "F_commerce"),
    _sp("SP-6.3b", "Objective usefulness from independently generated tests and subjective usefulness from diverse "
        "grounded participants, never a single global language-model judge; a genuinely novel product category (not "
        "a predefined format) can be discovered and adopted; workflows combining independently created services are "
        "registry objects.", "G", ["SP-6.3"],
        crit(("single_judge_usefulness_paths", "==", 0), ("novel_category_adoptions", ">=", 1),
             ("composed_workflows_registered", ">=", 1)), "Part 6.3", "G_innovation"),
    _sp("SP-6.4b", "The example need mechanisms (workflow change, transport or market access, coordination needs, "
        "content creating communities and tools, education changing skill availability) arise from the one reusable "
        "relation graph; an agent-created novel combination of capabilities changes future civilisation state without "
        "designer-supplied demand.", "G", ["SP-6.4"],
        crit(("need_mechanism_examples", ">=", 5), ("designer_supplied_demand_paths", "==", 0)), "Part 6.4",
        "G_innovation"),
    _sp("SP-6.5b", "Adoption by execution resists forgery; stars are not adoption; independent autonomous developers "
        "adopt; maintenance continues over long horizons; agent programs run in the sandbox.", "G", ["SP-6.5"],
        crit(("forged_adoptions_counted", "==", 0), ("star_only_adoptions_counted", "==", 0),
             ("autonomous_developer_adoptions", ">=", 1), ("agent_programs_outside_sandbox", "==", 0)), "Part 6.5",
        "G_innovation"),
    _sp("SP-6.6b", "Revenue integrity holds under multi-agent collusion that uses authorised institutions and escrow "
        "instruments.", "F", ["K"], crit(("collusion_scenarios_tested", ">=", 3), ("collusive_revenue_counted", "==", 0)),
        "Parts 6.6 / 12.3", "F_commerce"),
    # ---------------------------------------------------------------- Part 7
    _sp("SP-7.0", "Objectives are generated independently of worlds (IndependentlyGenerateGoal(world, difficulty)), "
        "not hard-coded per family.", "H", [],
        crit(("independently_generated_objective_kinds", ">=", 5), ("families_with_world_specific_objective_code", "==", 0)),
        "Part 7 (introduction)", "H_rapid"),
    _sp("SP-7.2", "Structural composition across topologies, functional forms, temporal relations, observation "
        "channels, affordances, participants, networks, resources, institutions, initial conditions and historical "
        "trajectories; superficially similar worlds that obey fundamentally different causal rules, which a learner "
        "must disambiguate.", "H", [],
        crit(("composition_dimensions", ">=", 11), ("lookalike_world_pairs", ">=", 3),
             ("lookalike_pairs_confirmed_different_by_intervention", ">=", 3)), "Part 7.2", "H_rapid"),
    _sp("SP-7.5", "A machine-readable registry documents why each family is distinct, and every claimed structural "
        "difference is verified by a controlled intervention.", "H", ["M"],
        crit(("families_with_distinctness_record", ">=", 60), ("claimed_differences_unverified", "==", 0)),
        "Part 7.5", "H_rapid"),
    # ---------------------------------------------------------------- Part 8
    _sp("SP-8.0", "Each of the 60 named Tier 1 families of Parts 8.1-8.10 is implemented as an admitted environment.",
        "H", ["M"], crit(("named_families_admitted", "==", 60), ("named_families_missing", "==", 0)), "Parts 8.1-8.10",
        "H_rapid"),
    _sp("SP-8.0b", "Each Tier 1 category meets its behavioural requirement: 8.2 success evaluated on unobserved "
        "conditions; 8.4 discrete, continuous and programmatic actions; 8.5 reward-independent separation of useful "
        "discovery from random novelty; 8.8 policy populations and held-out opponent families; 8.9 genuine "
        "interdependence (independent agents cannot succeed); 8.10 previously successful constant strategies fail "
        "after the change.", "H", ["SP-8.0"], crit(("category_requirements_met", ">=", 6)), "Parts 8.2-8.10", "H_rapid"),
    _sp("SP-8.12b", "Fast CPU and batched simulation where valid, with a measured throughput target that learners "
        "use.", "H", ["SP-8.12"], crit(("batched_families", ">=", 20), ("batched_throughput_target_met", "==", True)),
        "Part 8.12", "H_rapid"),
    # ---------------------------------------------------------------- Part 9
    _sp("SP-9.0", "Every Tier 2 domain has autonomous participants and open-ended actions, and its configurations "
        "differ in world structure, not only in objective text (no task prompt over an unchanged environment); each "
        "domain composes into Tier 3.", "I", ["N"],
        crit(("domains_with_autonomous_participants", "==", 15), ("domains_with_open_ended_actions", "==", 15),
             ("configurations_differing_only_in_objective_text", "==", 0), ("domains_composable_into_tier3", "==", 15)),
        "Part 9 (introduction)", "I_domains"),
    _sp("SP-9.16", "Feasibility without guaranteed success in every domain (the null baseline fails; not every learner "
        "succeeds); difficulty emerges from complexity, scarcity and uncertainty, not arbitrary failure probabilities; "
        "each domain objective is probed for degenerate strategies.", "I", ["N"],
        crit(("domains_where_null_baseline_fails", "==", 15), ("arbitrary_failure_probability_mechanisms", "==", 0),
             ("domains_with_degenerate_strategy_probe", "==", 15)), "Part 9.16", "I_domains"),
    # ---------------------------------------------------------------- Part 10
    _sp("SP-10.0", "Standalone worlds are preserved for experiments and training is not exclusive to one permanent "
        "civilization; world forks, historical queries and independent replication are supported; participants, "
        "organizations, relationships, products, technologies and institutions persist and evolve.", "J", ["P"],
        crit(("fork_divergence_supported", "==", True), ("historical_query_kinds", ">=", 3),
             ("independent_replications", ">=", 3), ("entity_kinds_persisting_and_evolving", ">=", 6)),
        "Parts 10 / 10.4", "A_J_substrate_civ"),
    _sp("SP-10.2", "Cross-domain consequences arise from reusable interactions over shared state (couplings that can be "
        "removed for counterfactuals), not scripted ones.", "J", ["O"],
        crit(("reusable_couplings", ">=", 10), ("scripted_consequence_paths", "==", 0),
             ("couplings_removable_fraction", "==", 1.0)), "Part 10.2", "A_J_substrate_civ"),
    _sp("SP-10.3b", "Scripted narrative events do not count toward multi-order chains; chains occur in the integrated "
        "Tier 3 civilization and replicate across seeds; scripted reference policies are labelled as such.", "J",
        ["SP-10.3"], crit(("scripted_chain_stages", "==", 0), ("chains_replicated_across_seeds", ">=", 3),
                          ("unlabelled_scripted_results", "==", 0)), "Part 10.3", "G_innovation"),
    # ---------------------------------------------------------------- Part 11
    _sp("SP-11.1b", "Several learners act in one world through the interface (multi-agent driver) in every tier, with "
        "exact structured observations on the Gym, PettingZoo, HTTP and MCP surfaces and one unit system (seconds, "
        "cost_s, action provenance) across tiers.", "L", ["SP-11.1"],
        crit(("tiers_with_concurrent_learners", "==", 3), ("surfaces_with_exact_observations", ">=", 4),
             ("adapters_with_inconsistent_units", "==", 0)), "Part 11.1 / 11.6", "L_experience"),
    _sp("SP-11.4b", "The self-goal interface extends beyond the predefined 4.0 metrics.", "L", ["SP-11.4"],
        crit(("self_goal_metric_kinds_beyond_v4", ">=", 3)), "Part 11.4", "P_acceptance"),
    _sp("SP-11.6b", "Reports separate algorithmic limitations from environment failures.", "N", [],
        crit(("failure_reports_with_cause_classification", ">=", 1), ("unclassified_failures", "==", 0)), "Part 11.6",
        "L_experience"),
    _sp("SP-11.6c", "Environment capabilities are validated with scripted reference agents, small neural RL agents, "
        "statistical learners and lightweight language-model agents (Clarification 1), and a reference learner "
        "improves over a frozen copy of itself in at least one world per tier.", "L", ["S"],
        crit(("learner_kinds_validated", "==", 4), ("tiers_with_learning_improvement", "==", 3)),
        "Clarification 1 / Part 11.6", "L_experience"),
    _sp("SP-11.6d", "Compatibility with existing learner interfaces is preserved: the 4.0 Gymnasium and PettingZoo "
        "interfaces keep working (Clarification 1).", "L", [],
        crit(("legacy_interfaces_working", ">=", 2)), "Clarification 1", "P_acceptance"),
    # ---------------------------------------------------------------- Part 12
    _sp("SP-12.0", "Learning agents found competing businesses, collaborate on software, trade, negotiate, join "
        "organisations, develop competing technologies, create services used by other agents, form cooperative "
        "ventures, compete for scarce resources and respond to changing strategies in digital and integrated worlds.",
        "K", ["Q"], crit(("selfplay_capability_kinds", ">=", 10)), "Part 12 (introduction)", "K_selfplay"),
    _sp("SP-12.1b", "Self-play populations mix shared-training and independently updated participants with diverse "
        "learning algorithms, unrelated objectives and lifetime adaptation (not a homogeneous population).", "K",
        ["SP-12.1"], crit(("distinct_learning_algorithms", ">=", 3), ("participants_with_unrelated_objectives", ">=", 2),
                          ("independently_updated_participants", ">=", 2)), "Part 12.1", "K_selfplay"),
    _sp("SP-12.2", "Self-play records model hashes, participant identity, training-data provenance, experience, "
        "actions and outcome history, policy updates and population composition; cross-play between policy versions "
        "is reproducible; league or population-based evaluation runs.", "K", ["Q"],
        crit(("recorded_field_kinds", ">=", 6), ("crossplay_reproduction_mismatches", "==", 0),
             ("league_evaluations", ">=", 1)), "Part 12.2", "K_selfplay"),
    _sp("SP-12.3b", "No unauthorised access to private observations, no shared evaluator information and no "
        "collusive benchmark exploitation; cooperation emerges through permitted world interactions, not shared "
        "reward.", "K", ["SP-12.3"],
        crit(("private_observation_accesses", "==", 0), ("shared_reward_channels", "==", 0),
             ("collusion_probes_passed", ">=", 3)), "Part 12.3", "K_selfplay"),
    # ---------------------------------------------------------------- Part 13
    _sp("SP-13.1", "Compatible decisions are grouped for batched inference across every neural and semantic "
        "participant class.", "M", ["G"], crit(("participant_classes_batched", ">=", 3)), "Part 13.1", "M_performance"),
    _sp("SP-13.3", "Backends are benchmarked (NumPy, compiled kernels, Numba or equivalent, PyTorch CPU, quantized, "
        "batched GPU) and chosen by measured workload; high-frequency paths avoid excessive Python object creation.",
        "M", [], crit(("backends_benchmarked", ">=", 6), ("backend_choices_justified_by_measurement", "==", True)),
        "Part 13.3", "P_acceptance"),
    _sp("SP-13.4", "The semantic inference service provides batching, quantization, short contexts, retrieval of "
        "relevant state, prefix caching, bounded generation, model versioning, CPU execution and selective GPU "
        "execution; semantic generation only where necessary; no network-dependent API in the default simulation.",
        "D", ["H"], crit(("inference_service_features", "==", 9), ("network_calls_in_default_simulation", "==", 0)),
        "Part 13.4", "D_semantic"),
    _sp("SP-13.5", "Each artifact is encoded once, not per viewer, and representations are reused across participants "
        "and worlds (measured in an integrated social world).", "D", ["E.4"],
        crit(("encoder_runs_per_artifact", "<=", 1.0), ("cross_world_representation_reuse", ">=", 1)), "Part 13.5",
        "D_semantic"),
    _sp("SP-13.6", "The ten large-population techniques are investigated; low-resolution participants are never a "
        "source of fabricated outcomes; approximation limits are documented and approximate modes validated.", "M",
        ["V"], crit(("techniques_investigated", ">=", 10), ("approximate_modes_validated", ">=", 1),
                    ("outcomes_without_decision_events", "==", 0)), "Part 13.6 / Phase 5", "M_performance"),
    _sp("SP-13.7b", "Hardware is inspected, other compute tenants are not interfered with, performance budgets are set "
        "before stress tests and no external compute is purchased covertly.", "M", ["SP-13.7"],
        crit(("budgets_recorded_before_stress_tests", "==", True), ("tenant_interference_incidents", "==", 0),
             ("external_compute_purchases", "==", 0)), "Part 13.7", "M_performance"),
    _sp("SP-13.8b", "Meaningful diverse learner experience per unit of compute is measured and reported.", "M",
        ["SP-13.8"], crit(("experience_per_compute_reported", "==", True)), "Parts 13 / 13.8", "M_performance"),
    # ---------------------------------------------------------------- Part 14
    _sp("SP-14.0", "Training diversity is measured along all six dimensions (parameters, causal structures, "
        "compositions, objectives, participant policies, action spaces); seeds alone do not count.", "N", [],
        crit(("diversity_dimensions_measured", "==", 6)), "Part 14 (introduction)", "N_research"),
    _sp("SP-14.1b", "The sampler is compatible with external learner infrastructure, mixtures are balanced and "
        "configurable, and the distribution is versioned.", "N", ["SP-14.1"],
        crit(("external_sampler_round_trip", "==", True), ("distribution_versioned", "==", True)), "Part 14.1",
        "N_research"),
    _sp("SP-14.2b", "Compute and environment experience are measured separately; every tier is available from the "
        "start.", "N", ["SP-14.2"], crit(("compute_and_experience_reported_separately", "==", True),
                                       ("tiers_available_at_start", "==", 3)), "Part 14.2", "N_research"),
    _sp("SP-14.3", "Adaptive curriculum over the ten named weakness categories with learning-progress, regret, "
        "uncertainty and success measures and collapse prevention; the curriculum never accesses sealed evaluation "
        "results.", "N", ["T"],
        crit(("weakness_categories", ">=", 10), ("curriculum_measures", ">=", 4), ("sealed_result_accesses", "==", 0)),
        "Part 14.3", "N_research"),
    # ---------------------------------------------------------------- Part 15
    _sp("SP-15.0", "Evaluation distinguishes memorisation, pretrained competence, context adaptation, persistent state, "
        "neural weight learning, meta-learning and transferable efficiency.", "N", ["U"],
        crit(("explanations_distinguished", "==", 7)), "Part 15 (introduction)", "N_research"),
    _sp("SP-15.1", "No claim of real-world generalisation without executed, authorised Level 6 experiments.", "N", [],
        crit(("unsupported_real_world_claims", "==", 0)), "Part 15.1", "N_research"),
    _sp("SP-15.2", "Outcome metrics never use tool-call counts, reasoning length, posts, websites, agent-reported "
        "success, language-model judgments or unverified milestones.", "N", [],
        crit(("banned_proxy_kinds_checked", ">=", 7), ("banned_proxy_outcomes_used", "==", 0)), "Part 15.2",
        "N_research"),
    _sp("SP-15.3b", "Time to competence, total action cost, wall-clock, data usage, retention, transfer and sample "
        "efficiency are recorded.", "N", ["SP-15.3"], crit(("efficiency_quantities_recorded", ">=", 7)), "Part 15.3",
        "N_research"),
    _sp("SP-15.4", "Parametric internalisation controls: written memories, retrieval indexes and replay cleared; "
        "optimizer reset; architecture configuration immutable; within-episode adaptation distinguished.", "N", ["U"],
        crit(("internalisation_controls", ">=", 4)), "Part 15.4", "N_research"),
    _sp("SP-15.7", "Paired world comparisons, prespecified metrics, development and confirmation experiments "
        "separated, independent holdouts, transparent failure reporting; no winner selection on the sealed set (access "
        "budget audited).", "N", ["U"],
        crit(("confirmation_runs_with_prespecified_metrics", ">=", 1), ("dev_confirmation_separated", "==", True),
             ("sealed_access_budget_violations", "==", 0)), "Part 15.7", "P_acceptance"),
    _sp("SP-15.8", "Instrument integrity: the evaluator resists adversarial learners (process-memory stashes, files "
        "outside the checkpoint, generator calibration shortcuts).", "O", ["U"],
        crit(("evaluator_adversarial_controls_passed", ">=", 3)), "Part 15 / 16", "N_research"),
    # ---------------------------------------------------------------- Part 16
    _sp("SP-16b", "Participant consent, moderation, abuse prevention and integrity mechanisms exist in the world; no "
        "real-world spam, fabricated engagement, covert influence, surveillance or impersonation; agent-supplied media "
        "is processed outside the sandbox only under sandbox-equivalent restrictions; sandbox tenants are isolated; "
        "learners can be evaluated out of process.", "O", ["SP-16"],
        crit(("integrity_mechanisms", ">=", 4), ("realworld_abuse_paths", "==", 0),
             ("unrestricted_media_tool_runs", "==", 0), ("cross_tenant_isolation_failures", "==", 0),
             ("out_of_process_evaluation_mode", "==", True)), "Part 16", "O_security"),
    _sp("SP-16c", "Hidden-answer extraction is tested as its own category (Tier 1 holdouts, capability test splits, "
        "verification seeds, sealed sets); sandbox metering and adoption traces are measured outside the measured "
        "process (D-20); the 4.0 red-team ratchet stays part of Gate W.", "O", ["W"],
        crit(("hidden_answer_probe_kinds", ">=", 4), ("hidden_answers_extracted", "==", 0),
             ("agent_reported_metering_trusted", "==", 0), ("v4_redteam_tests_passing", "==", True)), "Part 16",
        "O_security"),
    # ---------------------------------------------------------------- Parts 17, 19, 20, 21
    _sp("SP-17.1", "Every world gate's capability is re-demonstrated inside an integrated Tier 3 civilization with the "
        "shared population (gates B, C, D, E, G, H, J, K, L, Q), not only in purpose-built partial worlds.", "J", ["O"],
        crit(("world_gates_redemonstrated_in_civilization", "==", 10), ("partial_world_only_gates", "==", 0)),
        "Part 17 (preamble) and Clarification 3", "P_acceptance"),
    _sp("SP-19.3", "The seven Phase 3 vertical slices were executed and recorded (with an architecture-independent "
        "learner in place of CAM, Clarification 1).", "J", [], crit(("vertical_slices_executed", "==", 7)),
        "Part 19 Phase 3", "P_acceptance"),
    _sp("SP-19.7", "A failed requirement is never removed from the manifest and acceptance criteria are never changed "
        "to make the project pass: every criteria change is recorded in the append-only ledger, cites a decision and "
        "is at least as strict.", "P", [],
        crit(("requirements_removed_from_ledger", "==", 0), ("criteria_changes_not_stricter", "==", 0),
             ("ledger_append_only_violations", "==", 0), ("ledger_entries_checked", ">=", 1000)), "Part 19 Phase 7",
        "P_acceptance"),
    _sp("SP-20b", "The final report separates what was implemented, verified, empirically calibrated, approximated, "
        "demonstrated by scripted reference policies, demonstrated by learned agents, not scientifically established, "
        "failed and blocked, and reports actual hardware and compute, model sizes, population sizes, world counts and "
        "test results, with no number that is not traceable to evidence.", "P", ["X"],
        crit(("report_separation_categories", "==", 9), ("report_actuals_sections", ">=", 5),
             ("report_numbers_without_evidence", "==", 0)), "Part 20", "P_acceptance"),
    _sp("SP-21b", "The release verification itself checks the full regression, the committed final tree and verified "
        "reproduction instructions (clean-checkout reproduction of every verified requirement).", "P", ["X"],
        crit(("verify_all_checks_regression", "==", True), ("verify_all_checks_committed_tree", "==", True),
             ("verify_all_reproduces_in_clean_checkout", "==", True)), "Part 21", "P_acceptance"),
]
SPEC_ITEMS = SPEC_ITEMS + AUDIT_SPEC_ITEMS

AMENDMENT_ITEMS = AMENDMENT_ITEMS + [
    ("OO.9", "OO", "Every public entry point defaults to reward_mode='observation_only': experience.registry.make, the "
     "experience CLI, Gymnasium CrucibleEnv, the PettingZoo environment, research.envs.MechVecEnv, "
     "research.bridge.KernelMechEnv, selfplay.arena, scale.rollout and the bench rollout, experience.reference."
     "ReferenceLearner, research protocols and positive controls, and every social, innovation, domain, civilization "
     "and multi-agent wrapper; legacy and verified modes only when explicitly selected and recorded.", "L", ["R"],
     crit(("entry_points_checked", ">=", 12), ("entry_points_not_observation_only_by_default", "==", 0),
          ("unrecorded_mode_selections", "==", 0)), 2, {"audit": _AU.format("P_acceptance")}),
    ("OO.10", "OO", "Observation-only operation with an externally assigned objective in every tier: the objective is "
     "delivered to the learner, its progress never is, and outcomes are verified evaluator-side only.", "L", ["R"],
     crit(("tiers_covered", "==", 3), ("objective_delivered_tiers", "==", 3),
          ("objective_progress_values_exposed", "==", 0)), 2, {"audit": _AU.format("P_acceptance")}),
    ("OO.11", "OO", "Self-generated experiments, hypotheses and learning objectives are supported in every tier "
     "without automatically rewarding their completion.", "L", ["R"],
     crit(("tiers_with_self_experiments", "==", 3), ("self_experiment_rewards_paid", "==", 0)), 2,
     {"audit": _AU.format("L_experience")}),
    ("OO.12", "OO", "Verified outcome labels reach the learner only through a separate, explicitly authorised channel "
     "for evaluation or meta-learning and never become handcrafted intermediate rewards.", "L", ["R"],
     crit(("label_channel_separate_from_observations", "==", True), ("labels_converted_to_step_rewards", "==", 0)), 2,
     {"audit": _AU.format("L_experience")}),
    ("OO.13", "OO", "No learner's internal reward function (CAM's or any other architecture's) is implemented inside "
     "Crucible.", "P", [],
     crit(("learner_reward_functions_in_crucible", "==", 0), ("crucible_files_scanned", ">=", 500)), 2,
     {"audit": _AU.format("P_acceptance")}),
    ("OO.14", "OO", "Leak channels are probed per tier by information-flow probes that try to recover verifier values "
     "(observation, auxiliary/info fields, tool responses, event metadata); the Gymnasium scalar reward 0.0 is "
     "documented as not a learning signal.", "L", ["R"],
     crit(("tiers_covered", "==", 3), ("leak_channels_probed_per_tier", ">=", 4), ("verifier_values_recovered", "==", 0),
          ("scalar_reward_documented", "==", True)), 2, {"audit": _AU.format("P_acceptance")}),
    ("OO.15", "OO", "The observation-only requirements OO.1-OO.8 were implemented by a dedicated implementation "
     "subagent and verified by an independent verification subagent (different agent identities on evidence and "
     "review).", "P", ["OO.1", "OO.2", "OO.3", "OO.4", "OO.5", "OO.6", "OO.7", "OO.8"],
     crit(("oo_requirements_independently_verified", "==", 8)), 2, {"audit": _AU.format("P_acceptance")}),
    ("OO.16", "OO", "No family, seed, split or evaluator identity reaches participant-visible data (Tier 1 holdout "
     "identity, Tier 2 configuration split, Tier 3 evaluation profile); the evaluator outcome is unreachable from the "
     "participant's objects.", "L", ["R"],
     crit(("tiers_covered", "==", 3), ("identity_probes", ">=", 3), ("split_identity_leaks", "==", 0)), 2,
     {"audit": _AU.format("H_rapid")}),
    ("RP.5", "RP", "The old branch feat/crucible-5-openworld is preserved as an immutable historical reference at the "
     "recovered state a881d18, locally and on GitHub; work continues on feat/crucible-5-complete.", "P", [],
     crit(("old_branch_tip_matches_anchor", "==", True), ("old_branch_remote_matches_anchor", "==", True)), 3,
     {"audit": _AU.format("P_acceptance")}),
    ("RP.6", "RP", "No force-push and no exposed secrets: no credential patterns in the tree or in any reachable "
     "commit, no credentials in remote URLs, and the pushed integration branch only ever moved forward.", "P", [],
     crit(("commits_scanned", ">=", 300), ("secrets_found", "==", 0), ("remote_urls_with_credentials", "==", 0),
          ("non_fast_forward_remote_updates", "==", 0)), 3, {"audit": _AU.format("P_acceptance")}),
]
