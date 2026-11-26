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
        crit(("tests_failed", "==", 0), ("tests_errors", "==", 0), ("tests_total", ">=", 1076), ("tests_skipped", "<=", 2)),
        crit(("v4_definitions_changed", "==", 0), ("v4_definitions_checked", ">=", 25)),
        crit(("v4_families_failed", "==", 0), ("v4_families_run", ">=", 16)),
        crit(("sealed_leak_findings", "==", 0), ("sealed_sets_changed", "==", 0)),
        crit(("reproduction_mismatches", "==", 0), ("reproduced_results", ">=", 5)),
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
        crit(("unverified_requirements", "==", 0)),
        crit(("required_tests_not_executed", "==", 0)),
        crit(("clean_checkout_failures", "==", 0)),
        crit(("integration_branch_test_failures", "==", 0)),
        crit(("missing_documents", "==", 0)),
        crit(("open_critical_high_defects", "==", 0)),
        crit(("placeholder_or_skipped_acceptance_tests", "==", 0)),
        crit(("disconnected_workstreams", "==", 0)),
        crit(("report_claims_without_evidence", "==", 0)),
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
     "P", crit(("run_after_full_integration", "==", True))),
]

# mandatory requirements stated outside Part 17
SPEC_ITEMS = [
    ("SP-1.3", "Persistent engineering documents exist (MASTER_SPEC unchanged, ARCHITECTURE, INTERFACE_CONTRACTS, "
     "WORKSTREAMS, DECISION_LOG, PROGRESS, BLOCKERS, FIDELITY_REGISTER, ACCEPTANCE_MANIFEST, REPRODUCE).", "P", [],
     crit(("v5_documents_present", ">=", 10), ("master_spec_unchanged", "==", True))),
    ("SP-1.5", "Milestone state is persisted (branch, commit, workstreams, failures, remaining requirements, next actions, "
     "compute).", "P", [], crit(("milestones_recorded", ">=", 1))),
    ("SP-2.1", "Three execution modes: controlled experimental environments, standalone semantic worlds, integrated "
     "civilizations.", "A", [], crit(("execution_modes", ">=", 3))),
    ("SP-2.3", "Shared world substrate: common typed primitives, no duplicate shared facts, cross-module transactional "
     "integrity, recorded neural inference replayable without rerunning it.", "A", [],
     crit(("shared_primitive_kinds", ">=", 17), ("cross_module_rollback_failures", "==", 0),
          ("inference_replay_without_rerun", "==", True))),
    ("SP-3.1", "Persistent individual identity with independently evolving state; individuals respond differently to the "
     "same situation and their preferences evolve through experience.", "B", ["SP-2.3"],
     crit(("identity_state_fields", ">=", 15), ("same_situation_response_heterogeneity", ">", 0),
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
     crit(("completion_table_rows", ">=", 18), ("rows_without_evidence", "==", 0))),
    ("SP-20", "All 21 final deliverables are present.", "P", ["X"], crit(("deliverables_present", ">=", 21))),
    ("SP-21", "`python -m crucible.acceptance.v5 --verify-all` validates the manifest and evidence and exits nonzero "
     "whenever a requirement is not verified.", "P", [], crit(("checker_rejects_fabricated_evidence", "==", True),
                                                              ("checker_nonzero_when_unverified", "==", True))),
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
     crit(("cam_imports_in_crucible", "==", 0), ("gates_depending_on_cam", "==", 0)), 3),
    ("RP.2", "RP", "An independent audit classified every subsystem (RETAIN/REPAIR/EXTEND/REDESIGN/REBUILD/IMPLEMENT) "
     "on evidence before implementation resumed.", "P", [],
     crit(("audit_reports", ">=", 15), ("audit_reports_with_skeptic_review", ">=", 15)), 3),
    ("RP.3", "RP", "The progress ledger is maintained with the exact remaining requirements and next steps; milestones "
     "are committed and pushed to GitHub; no milestone work exists only in an uncommitted worktree.", "P", [],
     crit(("milestones_recorded", ">=", 3), ("integration_head_pushed", "==", True),
          ("ledger_lists_remaining_requirements", "==", True)), 3),
    ("RP.4", "RP", "Important fitted models, checkpoints and experimental artifacts are preserved in durable storage.",
     "P", [], crit(("required_artifacts_unpreserved", "==", 0)), 3),
]
