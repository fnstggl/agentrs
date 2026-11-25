# news_click@1 held-out evaluation report

Domain: content_selection. Dataset: mind_small. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E1_evaluation_plan.json` (sha256 5ae089ed1f36, commit d6f476707); fit at commit d86f683cc in 210 s.

## Estimand

P(click | news shown in an impression, the user's click history, text, category, popularity, age)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | 50,000 train-file impressions of fit users, 2019-11-10..13 (seed 0) | 1810125 | 26282 |
| val | train-file impressions of validation users, 2019-11-14 | 243973 | 3995 |
| test | all dev-file impressions, 2019-11-15 (later day) | 2740998 | 50000 |
| test_new_users | dev impressions of users absent from the train file | 2403744 | 44057 |
| test_returning_users | dev impressions of train-file users | 337254 | 5943 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test impression_auc | test impressions | test log_loss | test mean_obs | test mean_pred | test mrr | test ndcg10 | test ndcg5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.1629 | 0.5000 | 0.0390 | 0.0007 | 0.5000 | 73152 | 0.1700 | 0.0406 | 0.0413 | 0.2178 | 0.2857 | 0.2230 |
| B1_popularity | baseline: popularity logistic (3 features) | 4 | 0.1500 | 0.7321 | 0.0370 | 0.0018 | 0.6477 | 73152 | 0.1539 | 0.0406 | 0.0389 | 0.3086 | 0.4005 | 0.3389 |
| M1_logistic | L2 logistic, 9 features | 10 | 0.1458 | 0.7613 | 0.0365 | 0.0025 | 0.6713 | 73152 | 0.1498 | 0.0406 | 0.0381 | 0.3325 | 0.4286 | 0.3687 |
| M2_gbdt | GBDT, 400 trees x 31 leaves | 36400 | 0.1425 | 0.7836 | 0.0361 | 0.0031 | 0.6934 | 73152 | 0.1465 | 0.0406 | 0.0377 | 0.3424 | 0.4396 | 0.3778 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_popularity**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better on the primary metric.

Test log_loss: kept 0.1465, baseline 0.1539, difference -0.0074 (95% user-cluster bootstrap interval -0.0076 to -0.0072).

## Calibration

* ece: 0.0031
* brier: 0.0361
* mean_pred: 0.0377
* mean_obs: 0.0406

## Further held-out checks

### test_new_users

```json
{
 "B0_constant": {
  "n": 2403744,
  "log_loss": 0.17011181457110608,
  "brier": 0.039027715104971675,
  "ece": 0.0006335623822734601,
  "auc": 0.5,
  "mean_pred": 0.0413159312202196,
  "mean_obs": 0.04068236883794614,
  "impressions": 64193,
  "impression_auc": 0.5,
  "mrr": 0.21753244634438867,
  "ndcg5": 0.2231308216691946,
  "ndcg10": 0.2856769728067088
 },
 "B1_popularity": {
  "n": 2403744,
  "log_loss": 0.15399535319495536,
  "brier": 0.03706918631769229,
  "ece": 0.0018013649999909908,
  "auc": 0.732162206629214,
  "mean_pred": 0.03889770392445476,
  "mean_obs": 0.04068236883794614,
  "impressions": 64193,
  "impression_auc": 0.6477674927327676,
  "mrr": 0.30827665387533953,
  "ndcg5": 0.33879770615429267,
  "ndcg10": 0.4003993927745303
 },
 "M1_logistic": {
  "n": 2403744,
  "log_loss": 0.14997216213170625,
  "brier": 0.036559719522908624,
  "ece": 0.002614196402279542,
  "auc": 0.7614550904716333,
  "mean_pred": 0.038115028002576906,
  "mean_obs": 0.04068236883794614,
  "impressions": 64193,
  "impression_auc": 0.6712473440707799,
  "mrr": 0.33219834786403407,
  "ndcg5": 0.36839544225477516,
  "ndcg10": 0.42851642036074733
 },
 "M2_gbdt": {
  "n": 2403744,
  "log_loss": 0.1466342054835973,
  "brier": 0.03612655787076928,
  "ece": 0.003199651964791752,
  "auc": 0.7835563911115461,
  "mean_pred": 0.037671218778469455,
  "mean_obs": 0.04068236883794614,
  "impressions": 64193,
  "impression_auc": 0.6932670743516196,
  "mrr": 0.34176614753867823,
  "ndcg5": 0.3772774543939881,
  "ndcg10": 0.43911591144000656
 }
}
```

### test_returning_users

```json
{
 "B0_constant": {
  "n": 337254,
  "log_loss": 0.1689250391782051,
  "brier": 0.03868146783604967,
  "ece": 0.0010109978465605782,
  "auc": 0.5,
  "mean_pred": 0.0413159312202196,
  "mean_obs": 0.04030493337365902,
  "impressions": 8959,
  "impression_auc": 0.5,
  "mrr": 0.21959343602013368,
  "ndcg5": 0.22205293939840484,
  "ndcg10": 0.28574921637195666
 },
 "B1_popularity": {
  "n": 337254,
  "log_loss": 0.15295875352451216,
  "brier": 0.03673958308992177,
  "ece": 0.0017851690876746421,
  "auc": 0.7313333123642466,
  "mean_pred": 0.03875854842703934,
  "mean_obs": 0.04030493337365902,
  "impressions": 8959,
  "impression_auc": 0.6474372989808557,
  "mrr": 0.3108635462308043,
  "ndcg5": 0.3400231361126232,
  "ndcg10": 0.4011611511398618
 },
 "M1_logistic": {
  "n": 337254,
  "log_loss": 0.14891605337272962,
  "brier": 0.03621983818182438,
  "ece": 0.00199470895002046,
  "auc": 0.7602435005529177,
  "mean_pred": 0.03835238547461786,
  "mean_obs": 0.04030493337365902,
  "impressions": 8959,
  "impression_auc": 0.6714102400350521,
  "mrr": 0.3345787847467358,
  "ndcg5": 0.3706328541075773,
  "ndcg10": 0.4293667784286492
 },
 "M2_gbdt": {
  "n": 337254,
  "log_loss": 0.14541229907318592,
  "brier": 0.03578881906661662,
  "ece": 0.002739849511997195,
  "auc": 0.7842144511937633,
  "mean_pred": 0.037860096468296564,
  "mean_obs": 0.04030493337365902,
  "impressions": 8959,
  "impression_auc": 0.6942330785978419,
  "mrr": 0.34654406005442706,
  "ndcg5": 0.38167097960329555,
  "ndcg10": 0.44302114590013
 }
}
```


## Identification

Observational: MIND records the production recommender's impressions; which news were shown depended on the user, so click probabilities are conditional on that selection and the feature effects are associations, not effects of exposure.

## Extrapolation limits

Evidence: MIND-small, 50,000 sampled Microsoft News users (English, US market), impressions 2019-11-09..15 chosen by the production recommender. The simulator may use this model for the probability that an individual opens an article shown in a slate of news/article recommendations, given text affinity to its history, topic shares, popularity, freshness and slate size. It may NOT infer: (1) clicks on video or social posts (use content_consumption); (2) behaviour of populations, languages or periods other than US Microsoft News in November 2019; (3) the causal effect of showing an item (exposure was chosen by a recommender; there is no randomised exposure in MIND); (4) dwell or reading time (not in the data); (5) raw affinity values from a different encoder (use text_cos_z, or quantile-map text_cos). Popularity counts come from a 50k-user sample and are used per capita. News first shown on 2019-11-09 have left-censored ages.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| text_cos | cosine between the item's text embedding and the mean embedding of the individual's consumed items (here: 128-d LSA of TF-IDF; with another encoder map by quantile, card feature_quantiles) |
| text_cos_z | text_cos standardised within the slate the individual is shown (scale-free; use as is) |
| cat_share | (individual's past consumption in the item's category + s) / (past consumption + 1), s = population share of the category |
| subcat_share | the same for the finer topic (subcategory) |
| log_hist_len | log1p(number of items in the individual's consumption history) |
| pop_ctr_24h | (clicks + 10 m) / (impressions + 10) of the item in the previous 24 simulated hours, m = card prior_rate |
| pop_log_clicks_per_1k | log1p(1000 * clicks of the item in the previous 24 h / individuals active in those 24 h) |
| age_log_hours | log1p(hours since the item was first shown on the platform) |
| slate_size_log | log1p(number of items in the slate / page the individual is shown) |
