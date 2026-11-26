# creator_return@1 held-out evaluation report

Domain: repeat_consumption. Dataset: kuairand_pure. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 096fdcb06 in 40 s.

## Estimand

P(an individual plays another video of a creator within the next 7 days | it played a video of that creator today), including re-exposure by the production recommender

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | train users, d = 2022-04-08..04-24 | 402027 | 18204 |
| val | validation users, d = 2022-04-18..04-24 | 13620 | 3079 |
| test | test users, d = 2022-04-25..05-01 | 15996 | 3127 |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.0231 | 0.5000 | 0.0021 | 0.0073 | 0.0187 | 0.0020 | 0.0093 |
| B1_familiarity | baseline: logistic on 2 features | 3 | 0.0230 | 0.5410 | 0.0020 | 0.0072 | 0.0185 | 0.0020 | 0.0092 |
| M1_logistic | L2 logistic, 10 features | 11 | 0.0235 | 0.4069 | 0.0021 | 0.0070 | 0.0190 | 0.0020 | 0.0090 |
| M2_gbdt | GBDT, 335 trees | 30485 | 0.0226 | 0.5060 | 0.0021 | 0.0050 | 0.0176 | 0.0020 | 0.0070 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_familiarity**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better.

Test log_loss: kept 0.0176, baseline 0.0185, difference -0.0009 (95% user-cluster bootstrap interval -0.0015 to -0.0002).

## Calibration

* ece: 0.0050
* brier: 0.0021
* mean_pred: 0.0070
* mean_obs: 0.0020

## Identification

Observational: the outcome mixes the recommender's re-exposure and the individual's choice; feature effects are associations.

## Extrapolation limits

KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08..05-08. The simulator may use this model for the probability that an individual who consumed a creator's item today consumes another item of the same creator within 7 days, given its familiarity, follow and like signals and both sides' activity. It may NOT infer: (1) the individual's preference net of exposure (re-exposure by the production recommender and by follow feeds is part of the outcome); (2) item-level repeat consumption (rare in the data); (3) other platforms, content kinds or periods; (4) horizons other than 7 days; plays count only pool videos.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| log_plays_author_before | log_plays_author_before: see the E4 plan, models.creator_return.feature_definitions |
| log_plays_author_today | log_plays_author_today: see the E4 plan, models.creator_return.feature_definitions |
| log_days_since_first_play | log_days_since_first_play: see the E4 plan, models.creator_return.feature_definitions |
| author_log_pop_7d | author_log_pop_7d: see the E4 plan, models.creator_return.feature_definitions |
| user_log_plays_7d | user_log_plays_7d: see the E4 plan, models.creator_return.feature_definitions |
| user_log_authors_7d | user_log_authors_7d: see the E4 plan, models.creator_return.feature_definitions |
| follows_author | follows_author: see the E4 plan, models.creator_return.feature_definitions |
| liked_author_today | liked_author_today: see the E4 plan, models.creator_return.feature_definitions |
| watch_fraction_today | watch_fraction_today: see the E4 plan, models.creator_return.feature_definitions |
| weekend | weekend: see the E4 plan, models.creator_return.feature_definitions |
