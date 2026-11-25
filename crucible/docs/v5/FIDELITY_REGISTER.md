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
| repository adoption growth (next-day stars given today's activity) with NB2 overdispersion | `crucible.calibration.software` (`software_repo_growth@1`) | empirically fitted (observational) | GH Archive, D1 -> D2 train, D2 -> D3 test (2024-10-08..10 UTC), repositories with a star, fork or creation that day | test (321,303 repo-days): mean Poisson deviance 0.536 vs 0.732 proportional rule (diff -0.196 [-0.213, -0.178]); NB2 alpha 0.848, simulated-vs-observed KS(log1p) 0.009; new-repository holdout 0.376 vs 0.525 | short horizon; units exclude inactive repositories; observational; weekday only; Spearman slightly below the proportional rule (0.441 vs 0.463) |

The 55 mechanisms registered by Crucible 4.0 (`evaluation/fidelity.py`, `docs/experiments/ow_fidelity.json`) keep
their 4.0 categories: 37 plausible, 18 known unrealistic, none calibrated. v5 records are added by the workstreams that
build each mechanism and checked by SP-5.4.
