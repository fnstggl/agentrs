# watch_duration@1 held-out evaluation report

Domain: watch_duration. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 44 s.

## Estimand

conditional distribution of y = log1p(play_time_s) of a feed exposure given consumed, duration and affinity features; 21 quantile levels 0.02..0.98

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | tab-1 rows of both logs, train users, standard 04-12..05-02 and random 04-22..05-02 | 765811 | 18709 |
| val | tab-1 rows of both logs, validation users, 04-29..05-02 | 46955 | 3712 |
| test | tab-1 rows of both logs, test users (held out), 05-03..05-08 | 99981 | 3875 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `pinball`, lower is better).

| candidate | complexity | params | val pinball | test completion_rate_abs_err | test coverage_80 | test coverage_90 | test pinball | test pit_ks | test view_rate_abs_err | test wasserstein_watch_s |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_unconditional | baseline: unconditional quantiles | 21 | 0.2697 | 0.0485 | 0.8350 | 0.9222 | 0.2592 | 0.1405 | 0.0834 | 7.7498 |
| B1_duration_bins | baseline: quantiles per duration decile x consumed | 420 | 0.1405 | 0.0055 | 0.8093 | 0.9081 | 0.1378 | 0.0549 | 0.0267 | 1.9979 |
| M1_quantile_linear | linear monotone quantiles | 252 | 0.1362 | 0.0078 | 0.7967 | 0.9165 | 0.1325 | 0.0314 | 0.0134 | 1.2279 |
| M2_quantile_mlp | 2 x 32 tanh monotone quantile network | 2133 | 0.1323 | 0.0111 | 0.7962 | 0.8956 | 0.1294 | 0.0109 | 0.0044 | 0.4832 |

## Baseline comparison

Selected on validation: **M2_quantile_mlp**; best simple baseline on test: **B1_duration_bins**; kept: **M2_quantile_mlp**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test pinball: kept 0.1294, baseline 0.1378, difference -0.0084 (95% user-cluster bootstrap interval -0.0090 to -0.0077).

## Calibration

* pit_ks: 0.0109
* coverage_80: 0.7962
* coverage_90: 0.8956

## Further held-out checks

### kept model by duration decile (test)

```json
[
 {
  "duration_decile": 0,
  "rows": 9232,
  "completion_obs": 0.1394064124783362,
  "completion_sim": 0.12315857885615251,
  "view_obs": 0.28650346620450606,
  "view_sim": 0.2719887348353553,
  "watch_q50_obs": 2.3925,
  "watch_q50_sim": 2.400072351387358,
  "watch_q90_obs": 11.597499999999998,
  "watch_q90_sim": 11.152952225257888
 },
 {
  "duration_decile": 1,
  "rows": 9499,
  "completion_obs": 0.1027476576481735,
  "completion_sim": 0.07969259922097063,
  "view_obs": 0.19044109906305928,
  "view_sim": 0.19002000210548478,
  "watch_q50_obs": 2.361,
  "watch_q50_sim": 2.3276831874270503,
  "watch_q90_obs": 16.631400000000003,
  "watch_q90_sim": 15.603068673866451
 },
 {
  "duration_decile": 2,
  "rows": 9419,
  "completion_obs": 0.0749548784372014,
  "completion_sim": 0.06083448349081644,
  "view_obs": 0.142477970060516,
  "view_sim": 0.1437519906571823,
  "watch_q50_obs": 2.323,
  "watch_q50_sim": 2.2783876693228824,
  "watch_q90_obs": 22.839400000000026,
  "watch_q90_sim": 21.3849208002155
 },
 {
  "duration_decile": 3,
  "rows": 9487,
  "completion_obs": 0.05238747760092759,
  "completion_sim": 0.042268367239380204,
  "view_obs": 0.10835880678823653,
  "view_sim": 0.11067776957942448,
  "watch_q50_obs": 2.31,
  "watch_q50_sim": 2.3390140096976637,
  "watch_q90_obs": 25.154999999999987,
  "watch_q90_sim": 25.575622529453135
 },
 {
  "duration_decile": 4,
  "rows": 10523,
  "completion_obs": 0.03639646488643923,
  "completion_sim": 0.025848142164781908,
  "view_obs": 0.08942316829801387,
  "view_sim": 0.08875795875700845,
  "watch_q50_obs": 2.297,
  "watch_q50_sim": 2.279965533812255,
  "watch_q90_obs": 24.98800000000004,
  "watch_q90_sim": 24.531491733859877
 },
 {
  "duration_decile": 5,
  "rows": 10881,
  "completion_obs": 0.030603804797353185,
  "completion_sim": 0.018288760224244097,
  "view_obs": 0.08583769874092455,
  "view_sim": 0.08776766841282971,
  "watch_q50_obs": 2.308,
  "watch_q50_sim": 2.2812396229299465,
  "watch_q90_obs": 22.824,
  "watch_q90_sim": 24.011646879494435
 },
 {
  "duration_decile": 6,
  "rows": 9777,
  "completion_obs": 0.024854249769868057,
  "completion_sim": 0.015342129487572876,
  "view_obs": 0.09348470901094405,
  "view_sim": 0.08387030786539838,
  "watch_q50_obs": 2.278,
  "watch_q50_sim": 2.341180873336634,
  "watch_q90_obs": 26.100799999999968,
  "watch_q90_sim": 22.020112061252554
 },
 {
  "duration_decile": 7,
  "rows": 10629,
  "completion_obs": 0.015711732053815035,
  "completion_sim": 0.010160880609652836,
  "view_obs": 0.07281964436917866,
  "view_sim": 0.07291372659704581,
  "watch_q50_obs": 2.214,
  "watch_q50_sim": 2.1862779255878,
  "watch_q90_obs": 17.61060000000005,
  "watch_q90_sim": 17.921599695745392
 },
 {
  "duration_decile": 8,
  "rows": 10173,
  "completion_obs": 0.01140273272387693,
  "completion_sim": 0.006487761722205839,
  "view_obs": 0.07146367836429765,
  "view_sim": 0.0739211638651332,
  "watch_q50_obs": 2.189,
  "watch_q50_sim": 2.172086088608528,
  "watch_q90_obs": 18.490200000000005,
  "watch_q90_sim": 17.608832016590682
 },
 {
  "duration_decile": 9,
  "rows": 10361,
  "completion_obs": 0.007721262426406718,
  "completion_sim": 0.002605926068912267,
  "view_obs": 0.07808126628703793,
  "view_sim": 0.06707846732940836,
  "watch_q50_obs": 2.234,
  "watch_q50_sim": 2.2086890143089524,
  "watch_q90_obs": 18.113,
  "watch_q90_sim": 15.930667909642533
 }
]
```

### kept model by subset (test)

```json
{
 "consumed": {
  "n": 21702,
  "pinball": 0.22428317079610793,
  "pit_ks": 0.025261467681748445,
  "coverage_80": 0.8035664915675974,
  "coverage_90": 0.9006082388719934,
  "wasserstein_watch_s": 1.8060013699715085,
  "completion_rate_abs_err": 0.05169569397342497,
  "view_rate_abs_err": 0.02757791035555842
 },
 "skipped": {
  "n": 78279,
  "pinball": 0.10304901343152695,
  "pit_ks": 0.012380299762387078,
  "coverage_80": 0.794133803446646,
  "coverage_90": 0.894147855746752,
  "wasserstein_watch_s": 0.027082704677606977,
  "completion_rate_abs_err": 0.0002741603838245374,
  "view_rate_abs_err": 0.0017389803286013742
 },
 "standard_log": {
  "n": 10229,
  "pinball": 0.17826281511318534,
  "pit_ks": 0.08354761218590245,
  "coverage_80": 0.7681102747091603,
  "coverage_90": 0.8741812493889921,
  "wasserstein_watch_s": 5.097384318265091,
  "completion_rate_abs_err": 0.05017824166573777,
  "view_rate_abs_err": 0.04290088908563726
 },
 "random_exposure_log": {
  "n": 89752,
  "pinball": 0.12379130001817072,
  "pit_ks": 0.011063820304840004,
  "coverage_80": 0.7993805151974329,
  "coverage_90": 0.8979855602103575,
  "wasserstein_watch_s": 0.2489364873898518,
  "completion_rate_abs_err": 0.006718548640412538,
  "view_rate_abs_err": 0.0051589297976519875
 }
}
```


## Identification

Observational association: watch time given features among exposures chosen by the recommender (standard log) and random insertions (random log), pooled. It is not the causal effect of changing an item's duration or an individual's affinity.

## Extrapolation limits

Evidence: KuaiRand-Pure (27,285 anonymised Kuaishou users chosen by the KuaiRand authors (China, 2022); interactions only with the 7,583 videos of the random-exposure pool); Kuaishou short-video app, single-column featured feed (tab 1); 2022-04-08 to 2022-05-08. The model gives the distribution of log1p(watch seconds) of one feed exposure given whether it was consumed, the item's duration and the affinity/popularity inputs. The simulator may draw watch time for autoplay short-video exposures with durations of 6 s to 20 min. It may NOT infer: watch time for long-form video, live streams, articles or audio; durations outside the data range; effects of interventions on watch time (all associations are observational); other populations or periods. Quantiles outside the 2%-98% levels are extrapolated linearly over one quantile gap, so extreme replays (play ratios above about 4) are not reproduced; the skip branch (consumed = 0) is dominated by the platform's autoplay, so it describes how long an autoplaying video runs before the individual swipes away.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| consumed | 1 if the choice model's draw was 'consume', 0 for a skip |
| item_log_duration_s | log1p(duration_s) of the content item |
| user_tag_share | (individual's past views in the item's topic + 5 s) / (past views + 5), s = population share of the topic; topic = the item's primary tag or the nearest topic centroid of its embedding |
| user_tag_lift | log(user_tag_share / s): relative interest in the item's topic (0 = average) |
| creator_familiarity | log1p(the individual's past views of this creator's items) |
| user_log_plays_7d | log1p(items the individual consumed (views) in the previous 7 simulated days). Scale: the dataset counts only plays of the 7,583 pool videos, so map by quantile (card feature_quantiles) rather than by raw value |
| user_click_rate | (consumed + 20 m) / (feed exposures + 20) over the individual's whole history before today, m = the card's prior_rate; scale-free |
| item_click_rate | (views + 20 m) / (impressions + 20) of the item on the platform before today (audience counters views / impressions); scale-free |
| item_log_pop_7d | log1p(1000 * views of the item in the previous 7 days / individuals active in those 7 days); per-capita, so independent of population size |
| item_log_age_days | log1p((now - published_at) / 86400); the data cover ages 0-30 days only |
| item_played_before | 1 if the individual consumed this item before today, else 0 (repeat consumption) |
| target | watch_s = expm1(sample); a view in the simulator needs watch_s >= min(30, 0.5 duration) |
