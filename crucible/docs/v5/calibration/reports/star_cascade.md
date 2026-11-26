# star_cascade@1 held-out evaluation report

Domain: social_diffusion. Dataset: gharchive_2024-10-07_96h. Identification: observational.
Evaluation plan: `docs/v5/calibration/plans/E4_new_domains_plan.json` (sha256 ba512b47d889, commit 8e191e2e7); fit at commit 7a9dba33a in 371 s.

## Estimand

expected number of non-bot stars a repository receives in the next hour given its recent star history (self-excitation of attention, which mixes contagion with outside referrals)

## Splits

| split | definition | rows | users |
|---|---|---|---|
| train | hours 24-59 | 3761897 |  |
| val | hours 60-71 | 1265596 |  |
| test | hours 72-95 (day 4) | 2533915 |  |

## Held-out evaluation

Every candidate on the validation and test splits (primary metric `poisson_deviance`, lower is better).

| candidate | complexity | params | val poisson_deviance | test count_calibration_error | test mean_obs | test mean_pred | test poisson_deviance |
|---|---|---|---|---|---|---|---|
| B0_persistence | baseline rule: stars_prev_24h / 24 | 0 | 0.2064 | 0.9358 | 0.0452 | 0.0799 | 0.2213 |
| B1_poisson_24h | baseline: Poisson GLM, 1 feature | 2 | 0.1793 | 0.6262 | 0.0452 | 0.0460 | 0.2023 |
| M1_poisson_glm | Poisson GLM, 13 features | 14 | 0.1709 | 0.2590 | 0.0452 | 0.0437 | 0.1938 |
| M2_gbdt_poisson | GBDT Poisson, 20 trees | 1820 | 8.144e+04 | 9.433e+05 | 0.0452 | 4.273e+04 | 8.546e+04 |

## Baseline comparison

Selected on validation: **M1_poisson_glm**; best simple baseline on test: **B1_poisson_24h**; kept: **M1_poisson_glm**. selected on validation and confirmed on test: no simple baseline is better.

Test poisson_deviance: kept 0.1938, baseline 0.2023, difference -0.0084 (95% user-cluster bootstrap interval -0.0093 to -0.0075).

## Calibration

* count_calibration_error: 0.2590
* mean_pred: 0.0437
* mean_obs: 0.0452

## Further held-out checks

### nb_dispersion

```json
{
 "B0_persistence": 0.0,
 "B1_poisson_24h": 0.5160492312669575,
 "M1_poisson_glm": 0.5679639967317647,
 "M2_gbdt_poisson": 0.9999998270941386
}
```


## Identification

Observational: the dependence of next-hour stars on recent stars mixes contagion, referral bursts and recommender (trending) feedback; it is a reduced-form self-excitation, not a causal diffusion effect.

## Extrapolation limits

GH Archive public GitHub events, 2024-10-07..10 UTC: hourly non-bot stars of repositories that received a star in the previous 24 hours. The simulator may use this model for the next-hour arrival rate of endorsements (stars, saves, upvotes) of an item given its own recent endorsement history and activity, i.e. self-exciting attention dynamics on a platform with a trending surface. It may NOT infer: (1) why attention arrives (observational: contagion through followers, outside referrals from news sites and GitHub's own trending page are not separated); (2) cascades of other content kinds or platforms; (3) horizons beyond a day or the long-term popularity of an item; (4) the effect of promotion; (5) items with no endorsement in the last 24 hours (outside the units). Weekday data only.

## Simulator feature mapping

| input | simulator quantity |
|---|---|
| stars_prev_1h | stars_prev_1h: see the E4 plan, models.star_cascade.feature_definitions |
| stars_prev_3h | stars_prev_3h: see the E4 plan, models.star_cascade.feature_definitions |
| stars_prev_24h | stars_prev_24h: see the E4 plan, models.star_cascade.feature_definitions |
| star_decay_2h | star_decay_2h: see the E4 plan, models.star_cascade.feature_definitions |
| star_decay_12h | star_decay_12h: see the E4 plan, models.star_cascade.feature_definitions |
| forks_prev_24h | forks_prev_24h: see the E4 plan, models.star_cascade.feature_definitions |
| events_prev_24h | events_prev_24h: see the E4 plan, models.star_cascade.feature_definitions |
| repo_org | repo_org: see the E4 plan, models.star_cascade.feature_definitions |
| created_in_window | created_in_window: see the E4 plan, models.star_cascade.feature_definitions |
| age_hours_log | age_hours_log: see the E4 plan, models.star_cascade.feature_definitions |
| global_stars_prev_1h | global_stars_prev_1h: see the E4 plan, models.star_cascade.feature_definitions |
| hour_sin | hour_sin: see the E4 plan, models.star_cascade.feature_definitions |
| hour_cos | hour_cos: see the E4 plan, models.star_cascade.feature_definitions |
