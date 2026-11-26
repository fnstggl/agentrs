# content_consumption@1 held-out evaluation report

Domain: content_selection. Dataset: kuairand_pure. Identification: randomized_exposure.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 35 s.

## Estimand

P(an individual consumes (keeps watching) a video shown in the featured feed | exposure, features), with exposure not selected on the individual's preferences

## Splits

| split | definition | rows | users |
|---|---|---|---|
| rnd_train | random-exposure tab-1 rows, train users, 2022-04-22..05-02 | 365861 | 17402 |
| rnd_val | random-exposure tab-1 rows, validation users, 2022-04-29..05-02 | 38266 | 3409 |
| rnd_test | random-exposure tab-1 rows, test users (held out), 2022-05-03..05-08 (later time) | 89752 | 3805 |
| rnd_transport | random-exposure tab-1 rows, test users, 2022-04-22..05-08 | 168454 | 3977 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.4714 | 0.5000 | 0.1519 | 0.0129 | 0.4816 | 0.1864 | 0.1735 |
| B1_popularity | baseline: popularity logistic (2 features) | 3 | 0.4679 | 0.5669 | 0.1506 | 0.0153 | 0.4779 | 0.1864 | 0.1713 |
| O_logistic | L2 logistic fitted on observational logs | 14 | 0.6938 | 0.6603 | 0.2478 | 0.3134 | 0.6892 | 0.1864 | 0.4999 |
| O_gbdt | GBDT fitted on observational logs | 30394 | 0.6442 | 0.6626 | 0.2281 | 0.2811 | 0.6458 | 0.1864 | 0.4675 |
| O_gbdt_recalibrated | observational GBDT + intercept -1.558 fitted on random exposure | 30394 | 0.4430 | 0.6626 | 0.1439 | 0.0133 | 0.4560 | 0.1864 | 0.1733 |
| R_logistic | L2 logistic fitted on random exposure | 14 | 0.4350 | 0.6864 | 0.1413 | 0.0093 | 0.4472 | 0.1864 | 0.1785 |
| R_gbdt | GBDT fitted on random exposure, 400 trees | 36400 | 0.4305 | 0.6983 | 0.1397 | 0.0063 | 0.4426 | 0.1864 | 0.1804 |

## Baseline comparison

Selected on validation: **R_gbdt**; best simple baseline on test: **B1_popularity**; kept: **R_gbdt**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test log_loss: kept 0.4426, baseline 0.4779, difference -0.0353 (95% user-cluster bootstrap interval -0.0386 to -0.0316).

## Calibration

* ece: 0.0063
* brier: 0.1397
* mean_pred: 0.1804
* mean_obs: 0.1864

## Further held-out checks

### observational vs randomized logistic coefficients (standardised features)

```json
{
 "user_log_plays_7d": {
  "observational": -0.0899697983124475,
  "randomized": -0.02209752502417342
 },
 "user_click_rate": {
  "observational": 0.6610478894127272,
  "randomized": 0.7189182162072454
 },
 "item_log_pop_7d": {
  "observational": 0.0327598749987642,
  "randomized": 0.20549811944847568
 },
 "item_click_rate": {
  "observational": 0.3526468657550909,
  "randomized": 0.02169204544704863
 },
 "item_log_age_days": {
  "observational": -0.08318198705218982,
  "randomized": 0.001852550002111424
 },
 "item_log_duration_s": {
  "observational": -0.0203262091356254,
  "randomized": -0.06815065152980163
 },
 "user_tag_share": {
  "observational": 0.04650985615599121,
  "randomized": 0.1302318743979539
 },
 "user_tag_lift": {
  "observational": 0.02393381198510859,
  "randomized": 0.15936950500955238
 },
 "creator_familiarity": {
  "observational": 0.013849583812904201,
  "randomized": 0.014357997987868337
 },
 "item_seen_before": {
  "observational": 0.0033310119843621326,
  "randomized": 0.016365169719776532
 },
 "item_played_before": {
  "observational": -0.009517251003400281,
  "randomized": 0.0
 },
 "hour_sin": {
  "observational": 0.04103795760217284,
  "randomized": 0.02082958188200388
 },
 "hour_cos": {
  "observational": -0.009896071229895491,
  "randomized": -0.035773961497567654
 },
 "_intercept": {
  "observational": 0.12315872970378104,
  "randomized": -1.7445375980923008
 }
}
```

### observational models on all random-exposure rows of test users (04-22..05-08)

```json
{
 "O_logistic": {
  "n": 168454,
  "log_loss": 0.6904949463261613,
  "brier": 0.24837773573283983,
  "ece": 0.31973348012057323,
  "auc": 0.6638592566054254,
  "mean_pred": 0.5005424843591191,
  "mean_obs": 0.18080900423854585
 },
 "O_gbdt": {
  "n": 168454,
  "log_loss": 0.6458086687334483,
  "brier": 0.22808868346844696,
  "ece": 0.2872575833104155,
  "auc": 0.6675828651085272,
  "mean_pred": 0.4680665875489614,
  "mean_obs": 0.18080900423854585
 },
 "O_gbdt_recalibrated": {
  "n": 168454,
  "log_loss": 0.44616723921977225,
  "brier": 0.14020415363881145,
  "ece": 0.008212732513706612,
  "auc": 0.6675828651085272,
  "mean_pred": 0.1739629794475453,
  "mean_obs": 0.18080900423854585
 },
 "R_logistic": {
  "n": 168454,
  "log_loss": 0.4377693286744478,
  "brier": 0.1377215856776085,
  "ece": 0.00599017311227799,
  "auc": 0.6907819037193755,
  "mean_pred": 0.17703724934026646,
  "mean_obs": 0.18080900423854585
 },
 "R_gbdt": {
  "n": 168454,
  "log_loss": 0.4328976771558554,
  "brier": 0.13609386481107077,
  "ece": 0.003504807772148958,
  "auc": 0.7034615393700293,
  "mean_pred": 0.1777871530098408,
  "mean_obs": 0.18080900423854585
 }
}
```

### repeat consumption (item consumed before by the same individual)

```json
{
 "standard": {
  "rows_repeat": 145,
  "consume_rate_repeat": 0.47586206896551725,
  "rows_first": 1036620,
  "consume_rate_first": 0.5287135112191546
 },
 "random": {
  "rows_repeat": 8,
  "consume_rate_repeat": 0.125,
  "rows_first": 1141385,
  "consume_rate_first": 0.180531547199236
 }
}
```

### reliability table of the kept model on test

```json
[
 {
  "lo": 0.0,
  "hi": 0.06666666666666667,
  "n": 13185,
  "mean_pred": 0.045899537086449435,
  "mean_obs": 0.053469852104664393
 },
 {
  "lo": 0.06666666666666667,
  "hi": 0.13333333333333333,
  "n": 22783,
  "mean_pred": 0.09982534260552106,
  "mean_obs": 0.1090725540973533
 },
 {
  "lo": 0.13333333333333333,
  "hi": 0.2,
  "n": 20529,
  "mean_pred": 0.16570212331124282,
  "mean_obs": 0.16786010034585222
 },
 {
  "lo": 0.2,
  "hi": 0.26666666666666666,
  "n": 14908,
  "mean_pred": 0.2312063960359151,
  "mean_obs": 0.2380601019586799
 },
 {
  "lo": 0.26666666666666666,
  "hi": 0.3333333333333333,
  "n": 9509,
  "mean_pred": 0.29691754279328164,
  "mean_obs": 0.3011883478809549
 },
 {
  "lo": 0.3333333333333333,
  "hi": 0.4,
  "n": 4975,
  "mean_pred": 0.3623273833713451,
  "mean_obs": 0.36844221105527636
 },
 {
  "lo": 0.4,
  "hi": 0.4666666666666667,
  "n": 2198,
  "mean_pred": 0.428673898504926,
  "mean_obs": 0.4335759781619654
 },
 {
  "lo": 0.4666666666666667,
  "hi": 0.5333333333333333,
  "n": 961,
  "mean_pred": 0.49466172363108096,
  "mean_obs": 0.5067637877211238
 },
 {
  "lo": 0.5333333333333333,
  "hi": 0.6,
  "n": 414,
  "mean_pred": 0.5627348909135212,
  "mean_obs": 0.5483091787439613
 },
 {
  "lo": 0.6,
  "hi": 0.6666666666666666,
  "n": 193,
  "mean_pred": 0.6285288541099975,
  "mean_obs": 0.5854922279792746
 },
 {
  "lo": 0.6666666666666666,
  "hi": 0.7333333333333333,
  "n": 73,
  "mean_pred": 0.6929866197252337,
  "mean_obs": 0.7123287671232876
 },
 {
  "lo": 0.7333333333333333,
  "hi": 0.8,
  "n": 21,
  "mean_pred": 0.7621359967510022,
  "mean_obs": 0.8095238095238095
 },
 {
  "lo": 0.8,
  "hi": 0.8666666666666667,
  "n": 3,
  "mean_pred": 0.8372513267696693,
  "mean_obs": 1.0
 }
]
```


## Identification

Randomised exposure: the evaluated (and, for R_* candidates, fitted) rows are videos the platform inserted into feeds uniformly at random from the pool, so whether an item was shown is independent of the individual's unobserved taste. Conditional consume probabilities are therefore not inflated by recommender selection. The feature effects remain associations among randomly shown items (the features themselves - popularity, affinity - were not randomised), so they are not causal effects of changing an item's popularity or an individual's interests.

## Extrapolation limits

Evidence: KuaiRand-Pure, 27,285 anonymised Kuaishou users chosen by the KuaiRand authors (China, 2022); interactions only with the 7,583 videos of the random-exposure pool; platform Kuaishou short-video app, single-column featured feed (tab 1); period 2022-04-08 to 2022-05-08. The simulator may use this model for the probability that an individual keeps watching an item its feed shows, as a function of the individual's activity, consume rate, topic affinity and creator familiarity and of the item's popularity, consume rate, age and duration. It may NOT infer: (1) behaviour on platforms other than an autoplay short-video feed (search, long-form video, articles: use news_click for article choice); (2) item ages beyond 30 days (all pool videos were uploaded 2022-04-09..11, so age is confounded with calendar time) or durations outside 6 s - 20 min; (3) responses of other populations or periods (one Chinese app, one month in 2022, a selected user sample, a holiday period 2022-04-30..05-04 inside the data); (4) the effect of social context, prices or advertising, which are not features; (5) quantities on another scale: activity counts cover only pool videos, so map scale-dependent inputs by quantile (feature_quantiles). Topic affinity is measured from one primary tag; with embedding-based affinity the mapping is by quantile, not by value. Repeat consumption is rare in the data (item_played_before is set on 0.007% of rows), so its coefficient is weakly identified.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| user_log_plays_7d | log1p(items the individual consumed (views) in the previous 7 simulated days). Scale: the dataset counts only plays of the 7,583 pool videos, so map by quantile (card feature_quantiles) rather than by raw value |
| user_click_rate | (consumed + 20 m) / (feed exposures + 20) over the individual's whole history before today, m = the card's prior_rate; scale-free |
| item_log_pop_7d | log1p(1000 * views of the item in the previous 7 days / individuals active in those 7 days); per-capita, so independent of population size |
| item_click_rate | (views + 20 m) / (impressions + 20) of the item on the platform before today (audience counters views / impressions); scale-free |
| item_log_age_days | log1p((now - published_at) / 86400); the data cover ages 0-30 days only |
| item_log_duration_s | log1p(duration_s) of the content item |
| user_tag_share | (individual's past views in the item's topic + 5 s) / (past views + 5), s = population share of the topic; topic = the item's primary tag or the nearest topic centroid of its embedding |
| user_tag_lift | log(user_tag_share / s): relative interest in the item's topic (0 = average) |
| creator_familiarity | log1p(the individual's past views of this creator's items) |
| item_seen_before | log1p(times the individual was exposed to this item before today) |
| item_played_before | 1 if the individual consumed this item before today, else 0 (repeat consumption) |
| hour_sin | sin(2 pi hour / 24) of the local simulated time |
| hour_cos | cos(2 pi hour / 24) of the local simulated time |
