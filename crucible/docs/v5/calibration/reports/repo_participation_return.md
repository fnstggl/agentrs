# repo_participation_return@1 held-out evaluation report

Domain: community_participation. Dataset: gharchive_2024-10-07_96h. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 096fdcb06 in 45 s.

## Estimand

P(a developer who participates in a repository's discussion (opens or comments on an issue, opens or reviews a pull request) participates there again within 24 h)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | first events at hours 24-48 | 117830 |  |
| val | hours 48-60 | 44721 |  |
| test | hours 60-72 | 51783 |  |
| test_new_developers | test rows of developers with no event in hours 0-48 (reported) | 26127 |  |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `log_loss`, lower is better).

| candidate | complexity | params | val log_loss | test auc | test brier | test ece | test log_loss | test mean_obs | test mean_pred |
|---|---|---|---|---|---|---|---|---|---|
| B0_constant | baseline: constant | 1 | 0.6936 | 0.5000 | 0.2502 | 0.0444 | 0.6936 | 0.4583 | 0.5028 |
| B1_event_type | baseline: logistic on 4 features | 5 | 0.6623 | 0.6271 | 0.2369 | 0.0397 | 0.6668 | 0.4583 | 0.4981 |
| M1_logistic | L2 logistic, 16 features | 17 | 0.6568 | 0.6435 | 0.2341 | 0.0319 | 0.6607 | 0.4583 | 0.4897 |
| M2_gbdt | GBDT, 213 trees | 19383 | 0.6454 | 0.6658 | 0.2284 | 0.0295 | 0.6483 | 0.4583 | 0.4875 |

## Baseline comparison

Selected on validation: **M2_gbdt**; best simple baseline on test: **B1_event_type**; kept: **M2_gbdt**. selected on validation and confirmed on test: no simple baseline is better.

Test log_loss: kept 0.6483, baseline 0.6668, difference -0.0185 (95% user-cluster bootstrap interval -0.0223 to -0.0161).

## Calibration

* ece: 0.0295
* brier: 0.2284
* mean_pred: 0.4875
* mean_obs: 0.4583

## Further held-out checks

### test_new_developers

```json
{
 "B0_constant": {
  "n": 26127,
  "log_loss": 0.6940760028596525,
  "brier": 0.25046440640041157,
  "ece": 0.08556526881481635,
  "auc": 0.5,
  "mean_pred": 0.5027582109819232,
  "mean_obs": 0.41719294216710684
 },
 "B1_event_type": {
  "n": 26127,
  "log_loss": 0.6608331707312125,
  "brier": 0.2340028080133042,
  "ece": 0.06584530538819035,
  "auc": 0.6326899465760663,
  "mean_pred": 0.48303824755529723,
  "mean_obs": 0.41719294216710684
 },
 "M1_logistic": {
  "n": 26127,
  "log_loss": 0.6528322308449334,
  "brier": 0.23029017867120583,
  "ece": 0.043961709681172426,
  "auc": 0.6439793931952116,
  "mean_pred": 0.46026259119977236,
  "mean_obs": 0.41719294216710684
 },
 "M2_gbdt": {
  "n": 26127,
  "log_loss": 0.6441219190404747,
  "brier": 0.226284289983363,
  "ece": 0.038360782260351806,
  "auc": 0.6579347886992143,
  "mean_pred": 0.4553940004696694,
  "mean_obs": 0.41719294216710684
 }
}
```


## Identification

Observational: who participates where is self-selected; associations of activity with return are not effects of making a community more active.

## Extrapolation limits

GH Archive public GitHub events, 2024-10-07..10 UTC (Monday to Thursday): participation by non-bot, non-owner accounts in public issue and pull-request discussions. The simulator may use this model for the probability that an individual who has just taken part in a community or project discussion takes part again in the same community within a day, as a function of the community's recent activity and maintainer activity and of the individual's own activity. It may NOT infer: (1) participation in non-software communities, private spaces or other platforms; (2) horizons beyond 24 h or retention over weeks (the window has four days); (3) causal effects of maintainer responses or community size (observational: active repositories attract active developers); (4) the content or quality of contributions (only event types are seen); 'first participation' is first within the window, so earlier history before 2024-10-07 is unknown (left-censored).

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| first_is_issue_open | first_is_issue_open: see the E4 plan, models.repo_participation_return.feature_definitions |
| first_is_issue_comment | first_is_issue_comment: see the E4 plan, models.repo_participation_return.feature_definitions |
| first_is_pr_open | first_is_pr_open: see the E4 plan, models.repo_participation_return.feature_definitions |
| first_is_pr_review | first_is_pr_review: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_issue_events_24h | repo_issue_events_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_pr_events_24h | repo_pr_events_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_participants_24h | repo_participants_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_owner_events_24h | repo_owner_events_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_stars_24h | repo_stars_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_pushes_24h | repo_pushes_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| repo_org | repo_org: see the E4 plan, models.repo_participation_return.feature_definitions |
| actor_events_24h | actor_events_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| actor_participation_repos_24h | actor_participation_repos_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| actor_repo_events_24h | actor_repo_events_24h: see the E4 plan, models.repo_participation_return.feature_definitions |
| hour_sin | hour_sin: see the E4 plan, models.repo_participation_return.feature_definitions |
| hour_cos | hour_cos: see the E4 plan, models.repo_participation_return.feature_definitions |
