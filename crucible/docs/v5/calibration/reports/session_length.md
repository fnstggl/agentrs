# session_length@1 held-out evaluation report

Domain: attention_allocation. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit b1f2b73ce in 1234 s.

## Estimand

distribution of the minutes of logged play time an individual allocates to one feed session, given its earlier sessions and the time of day

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | train users, sessions starting 2022-04-22..05-02 | 270460 | 18809 |
| val | validation users, 2022-04-29..05-02 | 25739 | 3827 |
| test | test users, 2022-05-03..05-08 | 46864 | 3948 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `pinball`, lower is better).

| candidate | complexity | params | val pinball | test coverage_80 | test coverage_90 | test pinball | test pit_ks |
|---|---|---|---|---|---|---|---|
| B0_global | baseline: global quantiles | 19 | 0.0745 | 0.8233 | 0.9194 | 0.0750 | 0.0295 |
| B1_own_history | baseline: quantiles by own-history decile | 190 | 0.0719 | 0.8185 | 0.9231 | 0.0728 | 0.0428 |
| M1_quantile_linear | monotone quantile network, linear | 190 | 0.0718 | 0.8196 | 0.9121 | 0.0725 | 0.0425 |
| M2_quantile_mlp | monotone quantile network 2x32 | 2003 | 0.0714 | 0.8037 | 0.9086 | 0.0723 | 0.0305 |

## Baseline comparison

Selected on validation: **B0_global**; best simple baseline on test: **B1_own_history**; kept: **B1_own_history**. B0_global was selected on validation but the baseline B1_own_history is better on test; kept; rejected for long-horizon drift: ['B1_own_history', 'M1_quantile_linear', 'M2_quantile_mlp'].

Test pinball: kept 0.0728, baseline 0.0728, difference 0.0000 (95% user-cluster bootstrap interval 0.0000 to 0.0000).

## Calibration

* pit_ks: 0.0428
* coverage_80: 0.8185
* coverage_90: 0.9231

## Further held-out checks

### fit_history

```json
{
 "M1": {
  "epochs": 9,
  "val_pinball_by_epoch": [
   0.07246192544698715,
   0.07198217511177063,
   0.0719144344329834,
   0.07197551429271698,
   0.0718369260430336,
   0.07180129736661911,
   0.07181964814662933,
   0.07189228385686874,
   0.07191268354654312
  ]
 },
 "M2": {
  "epochs": 9,
  "val_pinball_by_epoch": [
   0.07190119475126266,
   0.07165852934122086,
   0.07164204865694046,
   0.07151059061288834,
   0.0716998428106308,
   0.07137487083673477,
   0.07150807231664658,
   0.07148325443267822,
   0.07147690653800964
  ]
 }
}
```


## Identification

Observational: individual differences in session length are associations with the individual's history.

## Extrapolation limits

KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, sessions of 2022-04-22..05-08 (30-minute idle gap). The simulator may use this model for how many minutes an individual spends watching in one feed session, given its own earlier sessions and the time of day. It may NOT infer: (1) total app time (only plays of the 7,583 pool videos are logged, so sessions are shorter than real app sessions); (2) other platforms or content kinds (articles, long video); (3) how session length responds to what is shown (no content inputs; use content_open and watch_time per item for that); (4) other populations or periods (one month including a holiday).

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| user_mean_log_minutes | user_mean_log_minutes: see the E4 plan, models.session_length.feature_definitions |
| user_log_sessions_before | user_log_sessions_before: see the E4 plan, models.session_length.feature_definitions |
| user_log_rows_per_session | user_log_rows_per_session: see the E4 plan, models.session_length.feature_definitions |
| active_days_7d | active_days_7d: see the E4 plan, models.session_length.feature_definitions |
| log_minutes_since_last_session | log_minutes_since_last_session: see the E4 plan, models.session_length.feature_definitions |
| sessions_earlier_today | sessions_earlier_today: see the E4 plan, models.session_length.feature_definitions |
| hour_sin | hour_sin: see the E4 plan, models.session_length.feature_definitions |
| hour_cos | hour_cos: see the E4 plan, models.session_length.feature_definitions |
| weekend | weekend: see the E4 plan, models.session_length.feature_definitions |
