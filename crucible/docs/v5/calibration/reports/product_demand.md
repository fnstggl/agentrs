# product_demand@1 held-out evaluation report

Domain: purchase_behavior. Dataset: online_retail_ii. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 096fdcb06 in 67 s.

## Estimand

weekly units sold of a product given its price relative to its own reference price and its recent sales (an association: prices were set by the retailer, not randomised)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | weeks starting 2009-11-30..2011-03-21 | 158397 |  |
| val | weeks 2011-03-28..2011-06-20 | 32965 |  |
| test | weeks 2011-06-27..2011-11-28 | 58394 |  |
| test_new_products | test weeks of products that are not eligible in training but have sales in at least 4 validation weeks (reported) | 9220 |  |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `mse`, lower is better).

| candidate | complexity | params | val mse | test bias | test mae | test mse | test price_coefficient |
|---|---|---|---|---|---|---|---|
| B0_product_mean | baseline: product training mean | 0 | 1.8554 | 0.3335 | 1.1372 | 2.0292 | 0.0000 |
| B1_recent_mean | baseline: mean of the previous 4 weeks | 0 | 1.4822 | 0.2229 | 0.7244 | 1.2751 | 0.0000 |
| M1_linear | ridge, 8 features | 9 | 1.1674 | 0.1449 | 0.8298 | 1.1608 | -0.5287 |
| M2_gbdt | GBDT, 400 trees | 36400 | 1.0715 | 0.1693 | 0.7451 | 1.0678 | -1.5144 |

## Baseline comparison

Selected on validation: **M1_linear**; best simple baseline on test: **B1_recent_mean**; kept: **M1_linear**. selected on validation and confirmed on test: no simple baseline is better; rejected for long-horizon drift: ['M2_gbdt'].

Test mse: kept 1.1608, baseline 1.2751, difference -0.1143 (95% user-cluster bootstrap interval -0.1482 to -0.0836).

## Calibration

* bias: 0.1449
* calibration_error: 0.1449

## Further held-out checks

### test_new_products

```json
{
 "B0_product_mean": {
  "mse": 1.3157897436184751,
  "mae": 0.847063654488133,
  "bias": 0.07107378941708596,
  "price_coefficient": 0.0,
  "n": 9220
 },
 "B1_recent_mean": {
  "mse": 1.254770309565896,
  "mae": 0.7707709501252717,
  "bias": 0.26723629119183523,
  "price_coefficient": 0.0,
  "n": 9220
 },
 "M1_linear": {
  "mse": 1.2060070611832667,
  "mae": 0.8277672737754249,
  "bias": -0.05423981122343783,
  "price_coefficient": -0.5286786997395442,
  "n": 9220
 },
 "M2_gbdt": {
  "mse": 1.0991230753241448,
  "mae": 0.7912161784656984,
  "bias": 0.13424653601747902,
  "price_coefficient": -0.8497111143815175,
  "n": 9220
 }
}
```

### price_response

```json
{
 "M1_linear_coefficient": -0.5286786997395442,
 "kept_mean_change_for_10pct_price_increase": -0.050388461930892664,
 "M1_coefficient_ci95_product_bootstrap_50": [
  -0.5889628640265955,
  -0.4753459132008673
 ]
}
```


## Identification

Observational: retailer-set prices; demand shocks that cause re-pricing (season, clearance) confound the price association.

## Extrapolation limits

UCI Online Retail II, one UK online gift-ware retailer (many wholesale customers), 2009-12..2011-12, 2561 products with sales in at least 20 training weeks. The simulator may use this model for the weekly demand of a durable consumer product in an online shop given its recent sales and its price relative to its own usual price. It may NOT infer: (1) a causal price elasticity: prices were set by the retailer (seasonal re-pricing, clearance, promotions), so the price coefficient is an association and may be biased in either direction; (2) demand for services, subscriptions or digital goods; (3) competition between shops or substitution between products (not modelled); (4) prices far from a product's usual price (log relative prices outside the training quantiles); (5) other markets or years. Quantity discounts are excluded from the price by using retail-size lines (<= 12 units).

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| log_rel_price | log_rel_price: see the E4 plan, models.product_demand.feature_definitions |
| log_units_prev_week | log_units_prev_week: see the E4 plan, models.product_demand.feature_definitions |
| log_units_prev_4w | log_units_prev_4w: see the E4 plan, models.product_demand.feature_definitions |
| log_customers_prev_4w | log_customers_prev_4w: see the E4 plan, models.product_demand.feature_definitions |
| product_log_mean_units | product_log_mean_units: see the E4 plan, models.product_demand.feature_definitions |
| week_sin | week_sin: see the E4 plan, models.product_demand.feature_definitions |
| week_cos | week_cos: see the E4 plan, models.product_demand.feature_definitions |
| weeks_since_first_sale_log | weeks_since_first_sale_log: see the E4 plan, models.product_demand.feature_definitions |
