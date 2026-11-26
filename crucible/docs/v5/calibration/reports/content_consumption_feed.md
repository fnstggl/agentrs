# content_consumption_feed@1 held-out evaluation report

Domain: content_selection. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 25 s.

## Estimand

P(consume | the production recommender exposed the video in the featured feed, features)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| obs_train | standard tab-1 rows, train users, 2022-04-12..05-02 | 399950 | 18117 |
| obs_val | standard tab-1 rows, validation users, 2022-04-29..05-02 | 8689 | 2490 |
| obs_test | standard tab-1 rows, test users (held out), 2022-05-03..05-08 (later time) | 10229 | 2688 |
| obs_test_time_only | standard tab-1 rows, train users, 2022-05-03..05-08 | 50099 | 12950 |
| rnd_transport | random-exposure tab-1 rows, test users, 2022-04-22..05-08 | 168454 | 3977 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.6917 | 0.5000 | 0.2517 | 0.0434 | 0.6965 | 0.4857 | 0.5290 |
| B1_popularity | baseline: popularity logistic (2 features) | 3 | 0.6573 | 0.6337 | 0.2364 | 0.0301 | 0.6650 | 0.4857 | 0.5157 |
| O_logistic | L2 logistic, 13 features | 14 | 0.6301 | 0.7075 | 0.2178 | 0.0178 | 0.6254 | 0.4857 | 0.5012 |
| O_gbdt | GBDT, 334 trees x 31 leaves | 30394 | 0.6277 | 0.7082 | 0.2171 | 0.0057 | 0.6238 | 0.4857 | 0.4833 |

## Baseline comparison

Selected on validation: **O_gbdt**; best simple baseline on test: **B1_popularity**; kept: **O_gbdt**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test log_loss: kept 0.6238, baseline 0.6650, difference -0.0412 (95% user-cluster bootstrap interval -0.0476 to -0.0338).

## Calibration

* ece: 0.0057
* brier: 0.2171
* mean_pred: 0.4833
* mean_obs: 0.4857

## Further held-out checks

### test_time_only (train users, later days)

```json
{
 "B0_constant": {
  "n": 50099,
  "log_loss": 0.6963316308376345,
  "brier": 0.25158995950999236,
  "ece": 0.041898157437746886,
  "auc": 0.5,
  "mean_pred": 0.5290336292036504,
  "mean_obs": 0.4871354717659035
 },
 "B1_popularity": {
  "n": 50099,
  "log_loss": 0.6655572816019341,
  "brier": 0.2366006337318547,
  "ece": 0.026753636646560674,
  "auc": 0.6333745517409465,
  "mean_pred": 0.5138891084124642,
  "mean_obs": 0.4871354717659035
 },
 "O_logistic": {
  "n": 50099,
  "log_loss": 0.6285124533767407,
  "brier": 0.21920523419788232,
  "ece": 0.01872840636706014,
  "auc": 0.7028711878554542,
  "mean_pred": 0.5011114837126861,
  "mean_obs": 0.4871354717659035
 },
 "O_gbdt": {
  "n": 50099,
  "log_loss": 0.6260428416510108,
  "brier": 0.21816689583339668,
  "ece": 0.006258974051521488,
  "auc": 0.7049304800081434,
  "mean_pred": 0.4839564359940941,
  "mean_obs": 0.4871354717659035
 }
}
```

### transport to random exposure (test users, 04-22..05-08)

```json
{
 "B0_constant": {
  "n": 168454,
  "log_loss": 0.7319467723091017,
  "brier": 0.26937749765689917,
  "ece": 0.34822462496510453,
  "auc": 0.5,
  "mean_pred": 0.5290336292036504,
  "mean_obs": 0.18080900423854585
 },
 "B1_popularity": {
  "n": 168454,
  "log_loss": 0.6959422353854723,
  "brier": 0.2511275250303478,
  "ece": 0.30932364565546255,
  "auc": 0.5281244110334034,
  "mean_pred": 0.4901326498940084,
  "mean_obs": 0.18080900423854585
 },
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
 }
}
```


## Identification

Observational: the exposures were chosen by Kuaishou's production recommender, which selects on preferences the features do not capture. The model therefore predicts engagement under that recommender and overstates the consume rate of exposures that are not so selected (see 'transport to random exposure'). Use it as a learned-engagement predictor (e.g. a recommender's scoring target), not as the individual's choice rule; the choice rule is content_consumption, identified under random exposure.

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
