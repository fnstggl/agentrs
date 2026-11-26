# Runtime decision models: feature contracts for the simulator

Owner: workstream E. Consumer: the unified population and every subsystem that publishes opportunities to it
(DECISION_LOG D-15). The Contracts section is generated from `crucible.calibration.runtime.specs()` by
`PYTHONPATH=src python -m crucible.calibration.runtime`; its text is the code's.

## How to use a decision model

```python
from crucible.calibration import runtime
dm = runtime.get("content_open@1")             # fitted checkpoint loaded once per process, SHA-256 verified
res = dm.evaluate(inputs)                     # inputs: {name: 1-D array}, one row per individual (or slot)
p, flags = res.values, res.extrapolated       # calibrated outputs; rows outside the data's feature support
draws = dm.sample(inputs, ctx.rng("pop.content_open"))   # draws with the world's own RNG stream (determinism)
payload["model"] = res.provenance()           # JSON-native decision provenance for the recorded batch event
```

Rules (enforced by the code and `tests/test_v5_calibration_runtime.py`):

1. **Row-wise only.** Every input is turned into the fitted feature by a fixed formula of that row's inputs (the
   formula the fitting code used on the data). Outputs never depend on which other rows share the batch: no rank,
   quantile or z-score is ever computed across individuals here. A within-slate statistic (the news models'
   `text_cos_z`) is the individual's OWN slate, i.e. part of its decision context, and the caller computes it.
2. **Absolute inputs in simulator units.** Counts, durations, cosines, flags and money exactly as the input table says.
   Where the dataset measured a quantity on another scale (KuaiRand counts only plays of a 7,583-video pool; retail
   spend is GBP of 2010-2011), the contract exposes a fixed declared factor (`count_scale`, the world's currency
   factor) that a world sets once in its configuration and records; it is never re-estimated from the batch.
3. **Declared support.** `res.extrapolated` marks rows with a feature outside the range the data covered (the
   card's `feature_quantiles` min..max). `evaluate(..., strict=True)` refuses them. Values outside an input's hard
   domain raise `ContractError`.
4. **No silent fallback.** Unknown decision ids, missing inputs, failed checkpoint hashes and `offline_only` models
   raise. The old pattern of trying a model and silently keeping a plausible rule (`population.models.build`) must
   not be copied: a world that configures a decision model either uses it or fails.
5. **Provenance.** `res.provenance()` = {decision, model, version, fidelity (register id), served, rows,
   extrapolated}. `served` is `K` for the fitted checkpoint, or the contender the E3 review kept (`S` the
   simulator's rule, `B` the simple baseline): the runtime serves the reviewed choice and says so.
6. **Fidelity.** Every decision model has the register record `runtime:<id>`; a world that wires one must update the
   in-world records (population.*, feed.*, commerce.*) with an `opt-in:` or `default:` claim, and a `default:` claim
   needs an in-world probe (`fidelity.INWORLD_PROBES`) that observes the model being evaluated.

## What the population workstream needs to wire (wave 2)

| simulated decision | decision model | served | state the population must keep (beyond what it has) |
|---|---|---|---|
| keep watching / open a shown feed item (video-like) | `content_open@1` | K | per individual: views in the last 7 days, opens and impressions so far; per pair: topic views, creator views, item exposures; per item: views, opens, impressions (audience counters exist) |
| watch time of a shown item | `watch_time@1` | K | as above, plus the consumed draw |
| open an article in a slate | `article_open_semantic@1` | K | MiniLM cosine of item and consumption history (D-16 shared space must preserve it), own topic views, item opens in 24 h |
| active today | `daily_return@1` | K | per individual: active days in the last 7, days since last active, impressions and sessions in the last 7 days |
| sessions per active day | `extra_sessions@1` | S (C2 rule kept by the E3 review) | none beyond the day's activity |
| session length (attention allocation) | `session_minutes@1` | K (the E4 rule kept the own-history quantile baseline) | per individual: mean log minutes and count of earlier sessions, items per session, time since the last session |
| return to a creator | `creator_return@1` | K (weak: AUC 0.51) | per pair: views of the creator before / today, first-view day, follow and like flags, watch fraction |
| respond to an advertising campaign | `ad_response@1` | B (constant ITT effect; review compute rule) | per individual: a response type drawn once (kept for when K is served) and the campaign arm |
| buy again from a shop within 90 days | `customer_return@1` | K | per (customer, shop): order days, spend, lines, products, cancellations |
| weekly product demand and price response | `product_weekly_demand@1` | K | per product: weekly units, price, reference price, distinct buyers in 4 weeks |
| take part again in a community | `participation_return@1` | K | per community: 24 h counts of threads, contributions, participants, maintainer events, endorsements; per individual: 24 h activity |
| endorsement arrivals (diffusion) | `endorsement_rate@1` | K | per item: hourly endorsement history (1 h, 3 h, 24 h, decays 2 h and 12 h) |
| star -> contribution on a repository | `repo_deep_engagement@1` | K | code-hosting event counts of the repository and the developer in 24 h |
| next-day repository endorsements | `repo_star_growth@1` | K | daily repository activity counts |

## Extrapolation limits (read before wiring)

* **Populations and periods.** Every model is fitted on one platform and one period: Kuaishou (China, April-May
  2022), Microsoft News (US, November 2019), Criteo display advertising (period undisclosed, released 2018), public
  GitHub (four weekdays of October 2024), one UK online gift retailer (2009-2011). Behaviour of other populations,
  cultures, platforms or eras is an extrapolation; a world should treat the models as one empirically anchored
  population type, not as "humans".
* **Identification.** Only `content_open@1` (random exposure) and `ad_response@1` (randomised campaigns) identify
  responses to an intervention. Every other model is observational: its feature effects are associations and must
  not be read as the effect of changing that feature in the world (for example the price coefficient of
  `product_weekly_demand@1` is not a causal elasticity, and the self-excitation of `endorsement_rate@1` mixes
  contagion with outside referrals and trending feedback).
* **Content.** No model responds to what content is about except through the inputs named (topic shares, MiniLM
  cosine for articles). Quality, correctness, negation and sentiment are not inputs (and the encoders are weak on
  them, audit D). Ad response does not depend on ad content, price or relevance.
* **Scale mismatches.** KuaiRand counts cover pool videos only (declared `count_scale`); retail money is GBP of
  2010-2011 (declared currency factor); GitHub counts are per repository and hour of a very large platform. None of
  these may be rescaled from the batch.
* **Weak or replaced models.** `creator_return@1` barely discriminates (AUC 0.51; repeat consumption is rare in
  KuaiRand-Pure). For `session_rate@1`, `commercial_response@1` and `news_click@1` the E3 review keeps another
  contender (see `served`); `article_open@1` (LSA features) is offline only.
* **Feedback.** Models whose outputs feed their own inputs (daily return, session minutes, weekly demand,
  endorsement rate, repository growth) were checked by multi-step forward simulation on held-out windows of at most a
  week (endorsements: 24 hours; demand: 23 weeks); beyond that horizon drift is unmeasured.

## Contracts

## `ad_response@1` -> `commercial_response@1`

* use: response to a randomised advertising intervention: an individual carries a persistent latent response type (drawn once with draw_types) and, in each campaign period, visits / converts with its type's treated probabilities when it is in the campaign's treated group, else with its control probabilities
* output: arm_probabilities (columns p_visit, p_conversion for the row's arm)
* status: ready
* fidelity record: `runtime:ad_response@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `response_type` | latent | type index | [0, 63] int | the individual's persistent response type, an integer drawn once at creation with DecisionModel.draw_types(n, rng) and stored in the individual's state |
| `treated` | context | flag | [0, 1] | 1 if the individual is in the campaign's treated (eligible) group this period |

Extrapolation limits: Criteo Uplift v2.1 randomized incrementality tests (13,979,592 users, period and advertisers undisclosed, released 2018): intention-to-treat effects of campaign eligibility, identified by randomisation. Absolute rates and effect sizes are not representative (non-uniform sub-sampling); the response does NOT depend on ad content, price, relevance or frequency (not identified by the data) - a world that wants those must compose this with another fitted model and register the composition; negative predicted effects are not confirmed; transport() assumes constant per-type odds ratios.

## `article_open@1` -> `news_click@1`

* use: probability that an individual opens a news article shown in a slate (MIND-small)
* output: probability (P(open) per slot)
* status: offline_only - its text_cos feature is a cosine in MIND's own TF-IDF/LSA space, whose projection is not saved, and pop_ctr_24h counts whole-slate impressions; neither can be computed in a simulated world without a batch-relative mapping. Use article_open@2 (news_click@2, fitted on cosines of the shared semantic encoder) when it is registered.
* fidelity record: `runtime:article_open@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `text_cos_lsa` | pair | cosine | [-1, 1] | cosine of 128-d LSA (TF-IDF of title + abstract) vectors fitted on MIND news text |

Extrapolation limits: MIND-small, 50,000 Microsoft News users (US, English), 2019-11-09..15, production-recommender exposure (observational); no dwell time.

## `article_open_semantic@1` -> `news_click_semantic@1`

* use: content selection: probability that an individual opens an article shown in a slate, with text affinity measured in the shared semantic encoder space
* output: probability (P(open) per slot)
* status: ready
* fidelity record: `runtime:article_open_semantic@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `text_cos` | pair | cosine | [-1, 1] | cosine between the item's text embedding (crucible.semantic.encoders.text, MiniLM-L6, L2-normalised) and the L2-normalised mean embedding of the individual's consumed items; 0 without history; a shared-space projection must preserve these cosines |
| `text_cos_z` | slate | z-score | [-inf, inf] | text_cos standardised within the individual's OWN slate (mean and sd of text_cos over the items of that slate; 0 when sd = 0) - the individual's decision context, never other individuals' items |
| `own_topic_views` | pair | count | [0, inf] | the individual's past views in the item's topic |
| `own_views_total` | individual | count | [0, inf] | the individual's past views |
| `topic_population_share` | context | fraction | [0, 1] | share of all past views in the item's topic |
| `item_opens_24h` | item | count | [0, inf] | opens of the item in the previous 24 h |
| `active_individuals_24h` | context | count | [0, inf] | individuals active on the platform in the previous 24 h |
| `item_age_hours` | item | hours | [0, inf] | hours since the item was first shown |
| `slate_size` | slate | count | [1, inf] | items in the slate |

Extrapolation limits: MIND-small (US Microsoft News, 2019-11-09..15), production-recommender exposure (observational); news articles only; no dwell time; non-commercial licence, acquisition route unresolved (BLOCKERS B-1).

## `content_open@1` -> `content_consumption@1`

* use: individual decision: whether an individual shown an item in a short-video feed keeps watching it (consumes it beyond ~7 s). Fitted on randomly inserted videos, so it is the individual's response to an item it did not select, not the recommender's selection.
* output: probability (P(consume) per row)
* status: ready
* fidelity record: `runtime:content_open@1`
* parameters: {'count_scale': 1.0}

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `own_views_7d` | individual | count | [0, inf] | items the individual consumed (a view: watched beyond the platform's view threshold) on this platform in the previous 7 simulated days |
| `own_opens_before` | individual | count | [0, inf] | feed items the individual opened on this platform before today (whole history) |
| `own_impressions_before` | individual | count | [0, inf] | feed impressions (items reached in the individual's feed) before today (whole history) |
| `item_views_7d` | item | count | [0, inf] | views of the item in the previous 7 days, all individuals |
| `active_individuals_7d` | context | count | [0, inf] | distinct individuals with at least one feed impression on the platform in the previous 7 days |
| `item_opens_before` | item | count | [0, inf] | opens of the item before today, all individuals |
| `item_impressions_before` | item | count | [0, inf] | impressions of the item before today, all individuals |
| `item_age_days` | item | days | [0, inf] | days since the item was published (now - published_at) / 86400 |
| `item_duration_s` | item | seconds | [0, inf] | the content item's duration (artifact metadata measured from the bytes, never declared) |
| `own_topic_views` | pair | count | [0, inf] | the individual's past views (before today) of items whose primary topic equals this item's primary topic; primary topic = nearest topic centroid of the item's embedding in the shared semantic space (D-16) |
| `own_views_before` | individual | count | [0, inf] | the individual's past views before today, all topics |
| `topic_population_share` | context | fraction | [0, 1] | share of all past views on the platform that went to this item's primary topic (smoothed: (topic views + 1) / (views + topics)) |
| `own_creator_views` | pair | count | [0, inf] | the individual's past views (before today) of this item's creator |
| `own_item_exposures` | pair | count | [0, inf] | times this item was in the individual's feed before today |
| `own_item_consumed` | pair | flag | [0, 1] | 1 if the individual viewed this item before today |
| `local_hour` | context | hour | [0, 24] | local simulated hour of the decision, 0 <= h < 24 (floored) |

Extrapolation limits: KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08..05-08 (one app, one month including the Labour Day holiday). Autoplay short video (6 s - 20 min) only; not text, articles, long video or other cultures. 'Consumed' in the data means watching beyond about 7 s. Feature effects are associations among the shown items, not causal effects of popularity or interest. Counts in the data cover only plays of the 7,583 pool videos: own_views_7d is multiplied by the declared count_scale (default 1.0; a world whose feeds show a wider catalogue should declare the share it treats as comparable) - never rescaled from the batch. Item ages beyond 30 days are outside the data.

## `content_open_recommender_target@1` -> `content_consumption_feed@1`

* use: recommender scoring target only (a learned-engagement recommender ranks candidates by it); never an individual's choice rule: fitted on recommender-selected exposures, it predicts 0.47 where 0.18 is observed under random exposure
* output: probability (P(consume | recommender-selected exposure) per row)
* status: ready
* fidelity record: `runtime:content_open_recommender_target@1`
* parameters: {'count_scale': 1.0}

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `own_views_7d` | individual | count | [0, inf] | items the individual consumed (a view: watched beyond the platform's view threshold) on this platform in the previous 7 simulated days |
| `own_opens_before` | individual | count | [0, inf] | feed items the individual opened on this platform before today (whole history) |
| `own_impressions_before` | individual | count | [0, inf] | feed impressions (items reached in the individual's feed) before today (whole history) |
| `item_views_7d` | item | count | [0, inf] | views of the item in the previous 7 days, all individuals |
| `active_individuals_7d` | context | count | [0, inf] | distinct individuals with at least one feed impression on the platform in the previous 7 days |
| `item_opens_before` | item | count | [0, inf] | opens of the item before today, all individuals |
| `item_impressions_before` | item | count | [0, inf] | impressions of the item before today, all individuals |
| `item_age_days` | item | days | [0, inf] | days since the item was published (now - published_at) / 86400 |
| `item_duration_s` | item | seconds | [0, inf] | the content item's duration (artifact metadata measured from the bytes, never declared) |
| `own_topic_views` | pair | count | [0, inf] | the individual's past views (before today) of items whose primary topic equals this item's primary topic; primary topic = nearest topic centroid of the item's embedding in the shared semantic space (D-16) |
| `own_views_before` | individual | count | [0, inf] | the individual's past views before today, all topics |
| `topic_population_share` | context | fraction | [0, 1] | share of all past views on the platform that went to this item's primary topic (smoothed: (topic views + 1) / (views + topics)) |
| `own_creator_views` | pair | count | [0, inf] | the individual's past views (before today) of this item's creator |
| `own_item_exposures` | pair | count | [0, inf] | times this item was in the individual's feed before today |
| `own_item_consumed` | pair | flag | [0, 1] | 1 if the individual viewed this item before today |
| `local_hour` | context | hour | [0, 24] | local simulated hour of the decision, 0 <= h < 24 (floored) |

Extrapolation limits: KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08..05-08 (one app, one month including the Labour Day holiday). Autoplay short video (6 s - 20 min) only; not text, articles, long video or other cultures. 'Consumed' in the data means watching beyond about 7 s. Feature effects are associations among the shown items, not causal effects of popularity or interest. Counts in the data cover only plays of the 7,583 pool videos: own_views_7d is multiplied by the declared count_scale (default 1.0; a world whose feeds show a wider catalogue should declare the share it treats as comparable) - never rescaled from the batch. Item ages beyond 30 days are outside the data. Contains the production recommender's selection on unobserved preferences.

## `creator_return@1` -> `creator_return@1`

* use: repeat consumption: probability that an individual who consumed a creator's item today consumes another item of the same creator within 7 days (includes re-exposure by feeds)
* output: probability (P(another view of the creator within 7 days))
* status: ready
* fidelity record: `runtime:creator_return@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `own_creator_views_before` | pair | count | [0, inf] | the individual's views of the creator's items on earlier days |
| `own_creator_views_today` | pair | count | [1, inf] | views of the creator's items today (>= 1) |
| `days_since_first_creator_view` | pair | days | [0, inf] | days since the individual first viewed the creator |
| `creator_views_7d` | item | count | [0, inf] | views of the creator's items by everyone in the previous 7 days |
| `active_individuals_7d` | context | count | [0, inf] | distinct individuals active on the platform in the previous 7 days |
| `own_views_7d` | individual | count | [0, inf] | the individual's views in the previous 7 days |
| `own_creators_7d` | individual | count | [0, inf] | distinct creators the individual viewed in the previous 7 days |
| `follows_creator` | pair | flag | [0, 1] | 1 if the individual follows the creator |
| `liked_creator_today` | pair | flag | [0, 1] | 1 if the individual liked one of the creator's items today |
| `mean_watch_fraction_today` | pair | fraction | [0, inf] | mean watch time / duration over today's views of the creator (capped at 3) |
| `weekend` | context | flag | [0, 1] | 1 on Saturdays and Sundays |

Extrapolation limits: KuaiRand-Pure, 2022-04-08..05-08: repeat consumption of a creator within 7 days is rare (0.2% of units) because most creators have one pool video; discrimination is weak (held-out AUC about 0.5); the outcome mixes recommender re-exposure with choice.

## `customer_return@1` -> `customer_return@1`

* use: product retention: probability that a customer who ordered from a shop today orders again within 90 days, given its order history with that shop
* output: probability (P(another order within 90 days))
* status: ready
* fidelity record: `runtime:customer_return@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `orders_before` | pair | count | [0, inf] | days with an order from this shop before today |
| `tenure_days` | pair | days | [0, inf] | days since the customer's first order from the shop |
| `recency_days` | pair | days | [0, inf] | days since the previous order (ignored when orders_before = 0) |
| `mean_gap_days` | pair | days | [0, inf] | mean days between earlier orders (ignored when orders_before < 2) |
| `spend_today` | pair | currency | [0, inf] | money spent today with the shop, in GBP-equivalent units (the world's currency divided by its declared GBP exchange factor) |
| `spend_total` | pair | currency | [0, inf] | cumulative spend with the shop including today |
| `lines_today` | pair | count | [0, inf] | order lines today |
| `products_today` | pair | count | [0, inf] | distinct products ordered today |
| `cancel_share` | pair | fraction | [0, 1] | cancelled lines / all lines of the customer with the shop |
| `home_market` | pair | flag | [0, 1] | 1 if the customer is in the shop's home market |
| `month` | context | month | [1, 12] int | month of the year 1..12 |

Extrapolation limits: UCI Online Retail II: one UK online gift-ware retailer (many wholesalers), 2009-12..2011-12; observational; no price, service or marketing inputs; 90-day horizon only; spend is in GBP of 2010-2011, so a world must declare its currency factor (no rescaling from the batch).

## `daily_return@1` -> `return_daily@1`

* use: whether an individual is active on the platform on a simulated day, given its recent activity
* output: probability (P(active today) per individual)
* status: ready
* fidelity record: `runtime:daily_return@1`
* parameters: {'count_scale': 1.0}

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `active_days_7d` | individual | days | [0, 7] int | days among the previous 7 on which the individual had at least one feed impression on the platform |
| `days_since_last_active` | individual | days | [0, 14] | days since the last active day, capped at 14 (14 when never active) |
| `feed_exposures_7d` | individual | count | [0, inf] | feed impressions in the previous 7 days |
| `sessions_7d` | individual | count | [0, inf] | sessions in the previous 7 days (a session ends after 30 idle minutes) |
| `active_yesterday` | individual | flag | [0, 1] | 1 if active the previous day |
| `weekend` | context | flag | [0, 1] | 1 on Saturdays and Sundays of the simulated calendar |

Extrapolation limits: KuaiRand-Pure app activity of 27,285 users, 2022-04-22..05-08 (17 days, includes a holiday); an active day is a day with a logged pool-video exposure and the random-insertion rate rose during the period; no notification, content or social inputs, so it cannot respond to what the platform shows; observational. See the E3 review (reviews/return_daily.json) for its multi-day forward simulation drift.

## `endorsement_rate@1` -> `star_cascade@1`

* use: social diffusion: expected endorsements (stars, saves, upvotes) an item receives in the next hour given its own recent endorsement history (self-exciting attention); for items with at least one endorsement in the previous 24 h
* output: rate (expected endorsements next hour; sample() draws negative-binomial counts (card nb_alpha))
* status: ready
* fidelity record: `runtime:endorsement_rate@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `endorsements_prev_1h` | item | count | [0, inf] | endorsements in the previous hour |
| `endorsements_prev_3h` | item | count | [0, inf] | endorsements in the previous 3 hours |
| `endorsements_prev_24h` | item | count | [0, inf] | endorsements in the previous 24 hours (>= 1 for the model's units) |
| `endorsement_decay_2h` | item | count | [0, inf] | sum over earlier hours of endorsements x exp(-age / 2 h) |
| `endorsement_decay_12h` | item | count | [0, inf] | the same with a 12 h decay |
| `forks_prev_24h` | item | count | [0, inf] | copies / forks in the previous 24 h |
| `events_prev_24h` | item | count | [0, inf] | all public events on the item in the previous 24 h |
| `owner_org` | item | flag | [0, 1] | 1 if the owner is an organisation |
| `created_in_window` | item | flag | [0, 1] | 1 if the item was created in the last 4 days |
| `age_hours` | item | hours | [0, inf] | hours since creation (used when created_in_window) |
| `global_endorsements_prev_1h` | context | count | [0, inf] | all endorsements on the platform in the previous hour |
| `utc_hour` | context | hour | [0, 24] | hour of the world clock |

Extrapolation limits: GH Archive public GitHub stars, 2024-10-07..10 UTC; contagion, outside referrals and trending feedback are not separated; one-day horizon; weekday only.

## `extra_sessions@1` -> `session_rate@1`

* use: number of sessions beyond the first on an active day (gamma-Poisson individual rates)
* output: rate (posterior mean extra sessions per active day; sample() draws a count for one active day)
* status: ready
* fidelity record: `runtime:extra_sessions@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `extra_sessions_observed` | individual | count | [0, inf] | sum over the individual's observed active days of (sessions that day - 1) |
| `active_days_observed` | individual | days | [0, inf] | number of active days over which extra_sessions_observed was counted |

Extrapolation limits: KuaiRand-Pure sessions (30-minute gap) of 27,285 users, 2022-04-22..05-08; only sessions with a logged exposure are seen, so true counts are higher; held-out rates drift upward (count calibration error 0.43); assumes a stable individual rate over two weeks.

## `participation_return@1` -> `repo_participation_return@1`

* use: community participation: probability that an individual who just took part in a project or community discussion takes part there again within 24 h
* output: probability (P(another participation within 24 h))
* status: ready
* fidelity record: `runtime:participation_return@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `first_kind` | individual | category | [0, 4] int | kind of the participation, index into ['issue_open', 'issue_comment', 'pr_open', 'pr_review', 'other'] |
| `community_issue_events_24h` | item | count | [0, inf] | issue / question threads opened or commented in the community in the previous 24 h |
| `community_pr_events_24h` | item | count | [0, inf] | contribution proposals and reviews in the previous 24 h |
| `community_participants_24h` | item | count | [0, inf] | distinct participants in the previous 24 h |
| `community_owner_events_24h` | item | count | [0, inf] | events by the community's owner / maintainer in the previous 24 h |
| `community_endorsements_24h` | item | count | [0, inf] | endorsements (stars, upvotes) in the previous 24 h |
| `community_pushes_24h` | item | count | [0, inf] | content updates (pushes) in the previous 24 h |
| `community_org` | item | flag | [0, 1] | 1 if the community is run by an organisation |
| `own_events_24h` | individual | count | [0, inf] | the individual's events anywhere in the previous 24 h |
| `own_participation_communities_24h` | individual | count | [0, inf] | distinct communities the individual took part in in the previous 24 h |
| `own_community_events_24h` | pair | count | [0, inf] | the individual's events in this community in the previous 24 h |
| `utc_hour` | context | hour | [0, 24] | hour of the participation (world clock) |

Extrapolation limits: GH Archive public GitHub issues and pull requests, 2024-10-07..10 UTC; software communities only; 24 h horizon; observational; first participation is first within the 4-day window.

## `product_weekly_demand@1` -> `product_demand@1`

* use: purchase and price response: expected log1p(units) a product sells in a week given its price relative to its own usual price and its recent sales (an association, not a causal elasticity)
* output: log_units (E[log1p(units)]; sample() adds a residual with the card's residual_sd and returns units)
* status: ready
* fidelity record: `runtime:product_weekly_demand@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `price` | item | currency | [1e-09, inf] | the product's retail price this week |
| `reference_price` | item | currency | [1e-09, inf] | the product's usual price (median over its earlier weeks) |
| `units_prev_week` | item | count | [0, inf] | units sold the previous week |
| `mean_units_prev_4w` | item | count | [0, inf] | mean units over the previous 4 weeks |
| `customers_prev_4w` | item | count | [0, inf] | distinct buyers in the previous 4 weeks |
| `product_mean_log_units` | item | log1p(units) | [0, inf] | mean of log1p(weekly units) over the product's earlier weeks |
| `week_of_year` | context | week | [0, 52] | week of the year 0..51, week 0 = ISO week 49 |
| `weeks_since_first_sale` | item | weeks | [0, inf] | weeks since the product's first sale |

Extrapolation limits: UCI Online Retail II, one UK online gift retailer, 2009-11..2011-12; retailer-set prices (seasonal re-pricing, clearance) confound the price association; durable gift products only; no competition or substitution.

## `repo_deep_engagement@1` -> `software_deep_engagement@1`

* use: probability that a developer's first star on a repository is followed within 24 h by a fork, issue or pull request by the same developer on the same repository
* output: probability (P(deeper engagement within 24 h) per first star)
* status: ready
* fidelity record: `runtime:repo_deep_engagement@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `repo_stars_24h` | item | count | [0, inf] | stars the repository received in the 24 complete hours before the star's hour (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_forks_24h` | item | count | [0, inf] | forks of the repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_issue_events_24h` | item | count | [0, inf] | issues opened/closed/commented on the repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_pr_events_24h` | item | count | [0, inf] | pull requests and reviews on the repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_pushes_24h` | item | count | [0, inf] | pushes (commit batches) to the repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_releases_24h` | item | count | [0, inf] | releases published in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_events_24h` | item | count | [0, inf] | all public events on the repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_org` | item | flag | [0, 1] | 1 if the repository is owned by an organisation (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_created_in_window` | item | flag | [0, 1] | 1 if the repository was created (observed) before the star within the data window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_age_hours` | item | count | [0, inf] | hours since creation when repo_created_in_window, else 0 (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_events_24h` | individual | count | [0, inf] | all public events by the developer in the 24 complete hours before the star's hour (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_stars_24h` | individual | count | [0, inf] | stars given by the developer in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_forks_24h` | individual | count | [0, inf] | forks made by the developer in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_issue_events_24h` | individual | count | [0, inf] | issue events by the developer in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_pr_events_24h` | individual | count | [0, inf] | pull-request events by the developer in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_pushes_24h` | individual | count | [0, inf] | pushes by the developer in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `actor_repo_events_24h` | individual | count | [0, inf] | the developer's events on this repository in that window (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `utc_hour` | context | hour | [0, 24] | hour of the star (world clock), floored |

Extrapolation limits: GH Archive public GitHub events 2024-10-07..10 UTC (Mon-Thu); observational, not the effect of promotion; stars are not usefulness; 24 h horizon; top decile over-predicts (0.169 vs 0.131).

## `repo_star_growth@1` -> `software_repo_growth@1`

* use: expected stars a repository receives the next day given its activity today (NB2 for sampling)
* output: rate (expected next-day stars; sample() draws negative-binomial counts)
* status: ready
* fidelity record: `runtime:repo_star_growth@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `stars_today` | item | count | [0, inf] | stars received today (non-bot accounts) (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `forks_today` | item | count | [0, inf] | forks today (non-bot accounts) (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `issue_events_today` | item | count | [0, inf] | issue events today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `pr_events_today` | item | count | [0, inf] | pull-request events today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `pushes_today` | item | count | [0, inf] | pushes today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `releases_today` | item | count | [0, inf] | releases today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `events_today` | item | count | [0, inf] | all public events today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `stars_yesterday` | item | count | [0, inf] | stars received the previous day (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `created_today` | item | flag | [0, 1] | 1 if the repository was created today (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `created_yesterday` | item | flag | [0, 1] | 1 if the repository was created the previous day (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |
| `repo_org` | item | flag | [0, 1] | 1 if owned by an organisation (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, `issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases) |

Extrapolation limits: GH Archive, repositories with a star, fork or creation that day, 2024-10-08..10 UTC; observational; weekday only; short horizon; inactive repositories are outside the data.

## `session_minutes@1` -> `session_length@1`

* use: attention allocation: minutes an individual spends watching in one feed session, given its earlier sessions and the time of day (draw once per session; content-level choices stay per item)
* output: quantiles (quantiles of minutes watched in the session at the card's levels)
* status: ready
* fidelity record: `runtime:session_minutes@1`
* parameters: none

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `own_mean_log_session_minutes` | individual | log1p(minutes) | [0, inf] | mean of log1p(minutes watched) over the individual's sessions on earlier days (ignored when own_sessions_before = 0) |
| `own_sessions_before` | individual | count | [0, inf] | the individual's sessions on earlier days |
| `own_mean_items_per_session` | individual | count | [0, inf] | mean feed items reached per earlier session |
| `active_days_7d` | individual | days | [0, 7] int | active days among the previous 7 |
| `minutes_since_last_session` | individual | minutes | [0, inf] | minutes since the individual's previous session ended (10080 when none) |
| `sessions_earlier_today` | individual | count | [0, inf] | the individual's sessions earlier the same day |
| `local_hour` | context | hour | [0, 24] | local hour of the session start |
| `weekend` | context | flag | [0, 1] | 1 on Saturdays and Sundays |

Extrapolation limits: KuaiRand-Pure sessions (30-minute gap) of 27,285 Kuaishou users, 2022-04-22..05-08; only plays of the 7,583 pool videos are logged, so minutes are lower than real app time; no content inputs.

## `watch_time@1` -> `watch_duration@1`

* use: watch time of an item the individual was shown, given whether it consumed it (draw content_open@1 first); replaces an exponential abandonment hazard
* output: quantiles (quantiles of watch seconds at the card's quantile levels)
* status: ready
* fidelity record: `runtime:watch_time@1`
* parameters: {'count_scale': 1.0}

| input | kind | unit | domain | meaning |
|---|---|---|---|---|
| `consumed` | individual | flag | [0, 1] | 1 if the individual opened / kept watching the item (the content_open draw), 0 for a skip |
| `item_duration_s` | item | seconds | [0, inf] | the content item's duration (artifact metadata measured from the bytes, never declared) |
| `own_topic_views` | pair | count | [0, inf] | the individual's past views (before today) of items whose primary topic equals this item's primary topic; primary topic = nearest topic centroid of the item's embedding in the shared semantic space (D-16) |
| `own_views_before` | individual | count | [0, inf] | the individual's past views before today, all topics |
| `topic_population_share` | context | fraction | [0, 1] | share of all past views on the platform that went to this item's primary topic (smoothed: (topic views + 1) / (views + topics)) |
| `own_creator_views` | pair | count | [0, inf] | the individual's past views (before today) of this item's creator |
| `own_views_7d` | individual | count | [0, inf] | items the individual consumed (a view: watched beyond the platform's view threshold) on this platform in the previous 7 simulated days |
| `own_opens_before` | individual | count | [0, inf] | feed items the individual opened on this platform before today (whole history) |
| `own_impressions_before` | individual | count | [0, inf] | feed impressions (items reached in the individual's feed) before today (whole history) |
| `item_opens_before` | item | count | [0, inf] | opens of the item before today, all individuals |
| `item_impressions_before` | item | count | [0, inf] | impressions of the item before today, all individuals |
| `item_views_7d` | item | count | [0, inf] | views of the item in the previous 7 days, all individuals |
| `active_individuals_7d` | context | count | [0, inf] | distinct individuals with at least one feed impression on the platform in the previous 7 days |
| `item_age_days` | item | days | [0, inf] | days since the item was published (now - published_at) / 86400 |
| `own_item_consumed` | pair | flag | [0, 1] | 1 if the individual viewed this item before today |

Extrapolation limits: KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08..05-08 (one app, one month including the Labour Day holiday). Autoplay short video (6 s - 20 min) only; not text, articles, long video or other cultures. 'Consumed' in the data means watching beyond about 7 s. Feature effects are associations among the shown items, not causal effects of popularity or interest. Counts in the data cover only plays of the 7,583 pool videos: own_views_7d is multiplied by the declared count_scale (default 1.0; a world whose feeds show a wider catalogue should declare the share it treats as comparable) - never rescaled from the batch. Item ages beyond 30 days are outside the data. Watch times beyond about 4x the duration (replays) are not reproduced; on recommender-selected exposures alone it under-predicts long watches.
