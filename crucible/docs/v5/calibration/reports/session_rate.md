# session_rate@1 held-out evaluation report

Domain: browsing_frequency. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 1 s.

## Estimand

individual rate of extra sessions (sessions - 1) per active day, heterogeneous across individuals

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | train users, active days 04-22..05-04 |  | 19003 |
| val | train users: history 04-22..04-30 predicts 05-01..05-04 |  | 18081 |
| test | test users (held out): history 04-22..05-04 predicts 05-05..05-08 (later time) |  | 3779 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `poisson_deviance`, lower is better).

| candidate | complexity | params | val poisson_deviance | test heldout_loglik_per_user | test ks_user_rate_distribution | test mae | test poisson_deviance |
|---|---|---|---|---|---|---|---|
| B0_global_mean | baseline: one rate for everybody | 1 | 4.1221 | -3.2865 | 0.1444 | 1.0519 | 3.7246 |
| B1_own_mean | baseline: individual's raw mean | 0 | 20.8139 | -3.2865 | 0.1444 | 1.0000 | 12.6748 |
| M1_gamma_poisson | gamma-Poisson heterogeneity (2 parameters) | 2 | 3.8287 | -2.5807 | 0.0799 | 0.9497 | 3.3015 |

## Baseline comparison

Selected on validation: **M1_gamma_poisson**; best simple baseline on test: **B0_global_mean**; kept: **M1_gamma_poisson**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test poisson_deviance: kept 3.3015, baseline 3.7246, difference -0.4231 (95% user-cluster bootstrap interval -0.6230 to -0.2398).

## Calibration

* count_calibration_error: 0.4312

## Further held-out checks

### fitted population distribution

```json
{
 "gamma_shape": 2.1861455758103303,
 "gamma_rate": 1.7103102908408632,
 "mean_rate": 1.278215764424551,
 "cv_rate": 0.6763328171238387,
 "train_loglik": -59667.84693135955
}
```

### held-out users without their own history

```json
{
 "nb_loglik_per_user": -2.580712296905414,
 "homogeneous_poisson_loglik_per_user": -3.2864633894701294,
 "ks_rate_distribution_nb": 0.0799153215136279,
 "ks_rate_distribution_poisson": 0.14439005027785132
}
```


## Identification

Observational description of between-individual heterogeneity; no causal claim.

## Extrapolation limits

Evidence: KuaiRand-Pure (27,285 anonymised Kuaishou users chosen by the KuaiRand authors (China, 2022); interactions only with the 7,583 videos of the random-exposure pool); sessions reconstructed from logged exposures with a 30-minute gap rule (2022-04-22..05-08). The simulator may draw each individual's propensity to start extra sessions on an active day from the fitted Gamma distribution (heterogeneity across individuals) and update it from observed behaviour with the posterior mean. It may NOT infer the absolute number of app sessions: only sessions that contain a logged pool-video exposure are seen, so true session counts are higher; nor session behaviour on other platforms, periods or populations. The gamma-Poisson form assumes a stable individual rate over two weeks.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| count | extra sessions (sessions - 1) the individual started on its active days so far |
| exposure | number of active days so far; predict() returns the posterior mean rate; draw_rates(n, rng) gives new individuals' rates |
