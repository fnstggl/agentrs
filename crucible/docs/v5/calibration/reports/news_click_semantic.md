# news_click_semantic@1 held-out evaluation report

Domain: content_selection. Dataset: mind_small. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 7a9dba33a in 240 s.

## Estimand

P(click | news shown in an impression, the user's click history), with text affinity measured in the shared semantic encoder space (sentence-transformers/all-MiniLM-L6-v2) so that the simulator can compute it

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | as news_click@1 | 1810125 | 26282 |
| val | as news_click@1 | 243973 | 3995 |
| test | as news_click@1 (dev day) | 2740998 | 50000 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.1629 | 0.5000 | 0.0390 | 0.0007 | 0.1700 | 0.0406 | 0.0413 |
| B1_popularity | baseline: logistic on 3 features | 4 | 0.1525 | 0.7141 | 0.0374 | 0.0037 | 0.1565 | 0.0406 | 0.0427 |
| M1_logistic | L2 logistic, 7 features | 8 | 0.1469 | 0.7551 | 0.0367 | 0.0020 | 0.1511 | 0.0406 | 0.0425 |
| M2_gbdt | GBDT, 400 trees | 36400 | 0.1457 | 0.7613 | 0.0367 | 0.0014 | 0.1503 | 0.0406 | 0.0410 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_popularity**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better.

Test log_loss: kept 0.1503, baseline 0.1565, difference -0.0062 (95% user-cluster bootstrap interval -0.0064 to -0.0060).

## Calibration

* ece: 0.0014
* brier: 0.0367
* mean_pred: 0.0410
* mean_obs: 0.0406

## Further held-out checks

### reference_news_click_v1_test_log_loss

```json
0.14648386140288772
```

### ranking_test

```json
{
 "impressions": 73152,
 "impression_auc": 0.6576467840653093,
 "mrr": 0.3173598577122886,
 "ndcg5": 0.35320782220325864,
 "ndcg10": 0.4123297915402412
}
```


## Identification

Observational: the production recommender chose the impressions, so click probabilities are conditional on that selection; feature effects are associations.

## Extrapolation limits

MIND-small: 50,000 sampled Microsoft News users (US, English), impressions 2019-11-09..15 chosen by the production recommender. The simulator may use this model for the probability that an individual opens an article shown in a slate, given the cosine of the article's text and its consumption history in the shared MiniLM-L6 space, its category share, popularity per active individual, age and slate size. It may NOT infer: (1) clicks on video or social posts (content_open); (2) other populations, languages or periods; (3) the causal effect of showing an item (no randomised exposure); (4) dwell or reading time; (5) affinities from other encoders or projections that do not preserve MiniLM cosines. Licence: Microsoft Research terms, non-commercial; acquisition route unresolved (BLOCKERS B-1).

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| text_cos | text_cos: see the E4 plan, models.news_click_semantic.feature_definitions |
| text_cos_z | text_cos_z: see the E4 plan, models.news_click_semantic.feature_definitions |
| cat_share | cat_share: see the E4 plan, models.news_click_semantic.feature_definitions |
| log_hist_len | log_hist_len: see the E4 plan, models.news_click_semantic.feature_definitions |
| pop_log_clicks_per_1k | pop_log_clicks_per_1k: see the E4 plan, models.news_click_semantic.feature_definitions |
| age_log_hours | age_log_hours: see the E4 plan, models.news_click_semantic.feature_definitions |
| slate_size_log | slate_size_log: see the E4 plan, models.news_click_semantic.feature_definitions |
