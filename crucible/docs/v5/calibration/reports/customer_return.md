# customer_return@1 held-out evaluation report

Domain: product_retention. Dataset: online_retail_ii. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 096fdcb06 in 4 s.

## Estimand

P(a customer places another (non-cancelled) order within 90 days | it ordered today and its order history)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | train customers, days 2009-12-01..2011-03-31 | 14593 | 3291 |
| val | validation customers, days 2010-10-01..2011-03-31 | 1160 | 490 |
| test | test customers, days 2011-04-01..2011-09-10 | 1013 | 391 |
| test_time_only | train customers, days 2011-04-01..2011-09-10 (reported) | 4580 | 1858 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.6799 | 0.5000 | 0.1927 | 0.0490 | 0.5743 | 0.7443 | 0.6953 |
| B1_recency_frequency | baseline: logistic on 2 features | 3 | 0.5898 | 0.8168 | 0.1549 | 0.0891 | 0.4589 | 0.7443 | 0.7802 |
| M1_logistic | L2 logistic, 12 features | 13 | 0.5388 | 0.8334 | 0.1430 | 0.0494 | 0.4274 | 0.7443 | 0.7763 |
| M2_gbdt | GBDT, 119 trees | 10829 | 0.5324 | 0.8393 | 0.1397 | 0.0407 | 0.4179 | 0.7443 | 0.7659 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_recency_frequency**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better.

Test log_loss: kept 0.4179, baseline 0.4589, difference -0.0409 (95% user-cluster bootstrap interval -0.0576 to -0.0255).

## Calibration

* ece: 0.0407
* brier: 0.1397
* mean_pred: 0.7659
* mean_obs: 0.7443

## Further held-out checks

### test_time_only

```json
{
 "B0_constant": {
  "n": 4580,
  "log_loss": 0.5972230185986875,
  "brier": 0.2035390975237462,
  "ece": 0.021260507445544974,
  "auc": 0.5,
  "mean_pred": 0.6953333790173372,
  "mean_obs": 0.7165938864628821
 },
 "B1_recency_frequency": {
  "n": 4580,
  "log_loss": 0.4937701285850227,
  "brier": 0.16862515181111334,
  "ece": 0.09916863655504929,
  "auc": 0.8074162988293996,
  "mean_pred": 0.7772971160025238,
  "mean_obs": 0.7165938864628821
 },
 "M1_logistic": {
  "n": 4580,
  "log_loss": 0.46003549392594156,
  "brier": 0.1554395875298903,
  "ece": 0.05557276467160674,
  "auc": 0.8237364660768125,
  "mean_pred": 0.7697636527482291,
  "mean_obs": 0.7165938864628821
 },
 "M2_gbdt": {
  "n": 4580,
  "log_loss": 0.4545758746959949,
  "brier": 0.15332718612357749,
  "ece": 0.04619961680090436,
  "auc": 0.8252858661288308,
  "mean_pred": 0.7618995198249644,
  "mean_obs": 0.7165938864628821
 }
}
```


## Identification

Observational: customers' order histories and their returns share unobserved causes (business size, assortment fit); feature effects are associations.

## Extrapolation limits

UCI Online Retail II: 5,942 customers of one UK online gift-ware retailer (many of them wholesalers), 2009-12-01..2011-12-09. The simulator may use this model for the probability that a customer who bought from a shop today buys from the same shop again within 90 days, given its order history (frequency, recency, tenure, spend, basket breadth, returns). It may NOT infer: (1) retention for subscriptions, digital products or consumers of other shops (one retailer, mostly business customers); (2) responses to price, service quality or marketing (not features; observational); (3) horizons other than 90 days; (4) behaviour after 2011 or outside the UK-centred customer base; lines without a customer id (23%) are excluded, so guest buyers are not represented.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| log_orders_before | log_orders_before: see the E4 plan, models.customer_return.feature_definitions |
| log_tenure_days | log_tenure_days: see the E4 plan, models.customer_return.feature_definitions |
| log_recency_days | log_recency_days: see the E4 plan, models.customer_return.feature_definitions |
| log_mean_gap_days | log_mean_gap_days: see the E4 plan, models.customer_return.feature_definitions |
| log_spend_today | log_spend_today: see the E4 plan, models.customer_return.feature_definitions |
| log_spend_total | log_spend_total: see the E4 plan, models.customer_return.feature_definitions |
| log_lines_today | log_lines_today: see the E4 plan, models.customer_return.feature_definitions |
| log_products_today | log_products_today: see the E4 plan, models.customer_return.feature_definitions |
| cancel_share | cancel_share: see the E4 plan, models.customer_return.feature_definitions |
| uk | uk: see the E4 plan, models.customer_return.feature_definitions |
| month_sin | month_sin: see the E4 plan, models.customer_return.feature_definitions |
| month_cos | month_cos: see the E4 plan, models.customer_return.feature_definitions |
