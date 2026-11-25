# return_daily@1 held-out evaluation report

Domain: browsing_frequency. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 10 s.

## Estimand

P(an individual is active on day d | activity history before d)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | train users, days 04-29..05-04 | 114678 | 19113 |
| val | validation users, days 05-01..05-04 | 16584 | 4146 |
| test | test users (held out), days 05-05..05-08 (later time) | 16104 | 4026 |
| test_time_only | train users, days 05-05..05-08 | 76452 | 19113 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.6125 | 0.5000 | 0.2130 | 0.0210 | 0.6174 | 0.6934 | 0.6724 |
| B1_own_rate | baseline rule: (active_days_7d + 1) / 9 | 0 | 0.6433 | 0.6480 | 0.2112 | 0.0728 | 0.6103 | 0.6934 | 0.6341 |
| M1_logistic | L2 logistic, 6 features | 7 | 0.5536 | 0.6938 | 0.1946 | 0.0428 | 0.5719 | 0.6934 | 0.7262 |
| M2_gbdt | GBDT, 119 trees | 10829 | 0.5429 | 0.7035 | 0.1899 | 0.0261 | 0.5617 | 0.6934 | 0.7183 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_own_rate**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test log_loss: kept 0.5617, baseline 0.6103, difference -0.0485 (95% user-cluster bootstrap interval -0.0549 to -0.0428).

## Calibration

* ece: 0.0261
* brier: 0.1899

## Further held-out checks

### test_time_only

```json
{
 "B0_constant": {
  "n": 76452,
  "log_loss": 0.6167042768191908,
  "brier": 0.2127203343110726,
  "ece": 0.02190481173372394,
  "auc": 0.5,
  "mean_pred": 0.6724131917194232,
  "mean_obs": 0.6943180034531471
 },
 "B1_own_rate": {
  "n": 76452,
  "log_loss": 0.6111794084096989,
  "brier": 0.21165414529442506,
  "ece": 0.07438508984577115,
  "auc": 0.6447760903649978,
  "mean_pred": 0.6369108867146852,
  "mean_obs": 0.6943180034531471
 },
 "M1_logistic": {
  "n": 76452,
  "log_loss": 0.573315364607966,
  "brier": 0.19496110665457855,
  "ece": 0.04234631497624849,
  "auc": 0.6896568923316856,
  "mean_pred": 0.7275743606918618,
  "mean_obs": 0.6943180034531471
 },
 "M2_gbdt": {
  "n": 76452,
  "log_loss": 0.5632799939469809,
  "brier": 0.1901300346492757,
  "ece": 0.026377197624514973,
  "auc": 0.7013418804150503,
  "mean_pred": 0.7194035473619056,
  "mean_obs": 0.6943180034531471
 }
}
```

### forward simulation of test users over 05-05..05-08 (20 draws)

```json
{
 "observed": {
  "active_days_distribution": [
   0.06135121708892201,
   0.11251862891207154,
   0.19026328862394437,
   0.2630402384500745,
   0.3728266269249876
  ],
  "retention_curve": [
   0.7434590553856609,
   0.7162759089364594,
   0.7308868501529052,
   0.745158002038736
  ]
 },
 "simulated": {
  "kept": {
   "active_days_distribution": [
    0.0383879781420765,
    0.09462245404868355,
    0.18340784898161946,
    0.28825136612021857,
    0.3953303527074019
   ],
   "retention_curve": [
    0.8091233435270132,
    0.7925416241930003,
    0.7504587155963304,
    0.7458885490995584
   ],
   "tv_distance_active_days": 0.047714853452558395,
   "retention_max_abs_err": 0.07626571525654091
  },
  "B0_constant": {
   "active_days_distribution": [
    0.01092896174863388,
    0.09483358171882762,
    0.2911823149528068,
    0.3979384003974168,
    0.20511674118231496
   ],
   "retention_curve": [
    0.672646958885491,
    0.6759768943255182,
    0.6736833163438669,
    0.6708120965001698
   ],
   "tv_distance_active_days": 0.23581718827620468,
   "retention_max_abs_err": 0.07434590553856613
  },
  "B1_own_rate": {
   "active_days_distribution": [
    0.09287133631395926,
    0.1550173869846001,
    0.20813462493790363,
    0.2547565822155986,
    0.2892200695479384
   ],
   "retention_curve": [
    0.6719503907577301,
    0.6879884471627592,
    0.6996772001359157,
    0.7006455997281686
   ],
   "tv_distance_active_days": 0.09189021361152509,
   "retention_max_abs_err": 0.07150866462793082
  }
 }
}
```

### per-user heterogeneity: distribution of active days 04-22..05-04 (all users)

```json
[
 170,
 375,
 781,
 1251,
 1874,
 2539,
 2932,
 3194,
 3102,
 2955,
 2535,
 2155,
 1788,
 1634
]
```


## Identification

Observational: return is predicted from the individual's own past activity (persistence and heterogeneity). Nothing here identifies what would make an individual return more often.

## Extrapolation limits

Evidence: KuaiRand-Pure (27,285 anonymised Kuaishou users chosen by the KuaiRand authors (China, 2022); interactions only with the 7,583 videos of the random-exposure pool); activity = at least one logged exposure that day in the standard or random-exposure log (Kuaishou app, 2022-04-22..05-08). The simulator may use the model for the daily probability that an individual comes back to a feed platform given its recent activity pattern, and the fitted relation between recent activity and return (habit persistence). It may NOT infer: absolute activity levels of other platforms (an active day here requires a logged pool-video exposure, and the random-insertion rate rose during the period, so detected activity rises over time); long-horizon churn beyond 17 days; effects of notifications, content quality or social ties (not inputs); populations other than this selected sample of Kuaishou users. The training days include the 2022-04-30..05-04 Labour Day holiday in China, the test days do not.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| active_days_7d | number of the previous 7 days on which the individual was active on the platform |
| days_since_last_active | days since the individual's last active day, capped at 14 |
| log_rows_7d | log1p(feed exposures in the previous 7 days); map by quantile (dataset logs only a sample of exposures) |
| log_sessions_7d | log1p(sessions in the previous 7 days; a session ends after 30 idle minutes); map by quantile |
| active_yesterday | 1 if the individual was active the previous day |
| weekend | 1 on Saturdays and Sundays |
