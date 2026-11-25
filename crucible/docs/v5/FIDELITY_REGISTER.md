# Crucible 5.0 fidelity register

Every important behavioural or economic mechanism has a record here (Part 5.4). Categories:

* **empirically fitted** - parameters estimated from a named dataset, with held-out evaluation;
* **empirically validated** - not fitted to the data, but checked against it on prespecified metrics;
* **inspired by research** - form taken from published work, parameters not estimated;
* **plausible, unvalidated** - a reasoned assumption;
* **deliberate simplification** - known to omit something, on purpose;
* **known unrealistic** - kept for control or tractability, and labelled.

A neural network is not "calibrated" because it is a neural network, and fitting a historical dataset does not show
real-world predictive validity. Each record states the population, platform and period of its evidence and its limits.

| mechanism | module | category | evidence (dataset, period) | held-out result | limits |
|---|---|---|---|---|---|
| developer light-to-deep engagement (first star -> fork / issue / pull request by the same developer within 24 h) | `crucible.calibration.software` (`software_deep_engagement@1`; for the software ecosystem and population) | empirically fitted (observational) | GH Archive public GitHub events, 2024-10-07..10 UTC (Mon-Thu), 376,446 first stars by non-bot non-owner accounts | time-based test (83,304 stars): log loss 0.1269 vs 0.1439 base rate (bootstrap diff -0.017 [-0.019, -0.015]), AUC 0.737, ECE 0.005; new-user holdout 0.1351 vs 0.1475 | public events only; 24 h horizon; observational, not the effect of promotion; GitHub population and one week; stars are not usefulness; top decile over-predicts (0.169 vs 0.131) |
| response to advertising intervention (per-arm visit and conversion probabilities; intention-to-treat effect of campaign eligibility) | `crucible.calibration.commercial` (`commercial_response@1`; for population ad response and commerce) | empirically fitted (randomized experiment, identified conditional on features) | Criteo Uplift v2.1: pooled randomized incrementality tests, period and advertisers undisclosed, 13,979,592 users, sub-sampled non-uniformly by the publisher | user-level test (2.80M): pooled log loss visit+conversion 0.1143 vs 0.2092 constant-effect baseline (diff -0.095 [-0.0955, -0.0942]); per-arm ECE <= 0.0004; visit Qini 0.087 [0.081, 0.093]; standardised visit effect 0.0074 vs propensity-stratified 0.0077 [0.0071, 0.0083] | absolute rates and effect sizes not representative; anonymised features; one campaign period, no dynamics or saturation; raw arm difference biased by pooled-test composition (0.0103) |
| heterogeneity of advertising response (joint distribution of baseline propensity and effect; 64 response types) | `crucible.calibration.commercial` (`CommercialResponseModel.response_types/sample_types/respond/transport`) | empirically fitted and validated on held-out users | as above (types built on the validation split, validated on test) | type-level weighted correlation of predicted vs observed visit uplift 0.970 [0.948, 0.978], calibration slope 0.84 (honest-estimate table: 0.988, slope 1.01); effect concentrated in the top decile (0.066 predicted, 0.058 observed) | negative predicted effects not confirmed; transport() assumes constant per-type odds ratios; not a response to arbitrary content or prices |
| repository adoption growth (next-day stars given today's activity) with NB2 overdispersion | `crucible.calibration.software` (`software_repo_growth@1`) | empirically fitted (observational) | GH Archive, D1 -> D2 train, D2 -> D3 test (2024-10-08..10 UTC), repositories with a star, fork or creation that day | test (321,303 repo-days): mean Poisson deviance 0.536 vs 0.732 proportional rule (diff -0.196 [-0.213, -0.178]); NB2 alpha 0.848, simulated-vs-observed KS(log1p) 0.009; new-repository holdout 0.376 vs 0.525 | short horizon; units exclude inactive repositories; observational; weekday only; Spearman slightly below the proportional rule (0.441 vs 0.463) |

The 55 mechanisms registered by Crucible 4.0 (`evaluation/fidelity.py`, `docs/experiments/ow_fidelity.json`) keep
their 4.0 categories: 37 plausible, 18 known unrealistic, none calibrated. v5 records are added by the workstreams that
build each mechanism and checked by SP-5.4.
