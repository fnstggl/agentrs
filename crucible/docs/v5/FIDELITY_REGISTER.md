# Crucible 5.0 fidelity register

Every important behavioural or economic mechanism has its own record here (MASTER_SPEC Part 5.4). Categories:

* **empirically fitted** - parameters estimated from a named dataset, with held-out evaluation;
* **empirically validated** - not fitted to the data, but checked against it on prespecified metrics;
* **inspired by research** - form taken from published work, parameters not estimated;
* **plausible, unvalidated** - a reasoned assumption;
* **deliberate simplification** - known to omit something, on purpose;
* **known unrealistic** - kept for control or tractability, and labelled.

A neural network is not "calibrated" because it is a neural network, and fitting a historical dataset does not show
real-world predictive validity. Each record states the population, platform and period of its evidence and its limits.

How the register is read and checked (`crucible.calibration.fidelity`, gate test `tests/v5/test_gate_SP_5_4.py`):

* every table below whose header starts with `id` is part of the register (section headings and blank lines do not
  end it);
* a record covers exactly the mechanism whose `id` it carries. The important mechanisms are computed from the code
  (fitted-model cards, the 4.0 register, every v5 behaviour kernel module, every population decision model, every
  parameter group of the social consumption engine and every runtime decision model) and each needs its own record:
  one row for a module never vouches for the decisions inside it;
* the **in-world use** column says whether a simulated world actually consumes a fitted model. `none` makes no
  claim. `opt-in:` and `default:` claims must name the module (in backticks) and the model id; the checker verifies
  that the module is registered and its package references the model, and a `default:` claim additionally needs an
  in-world probe that observes the model being evaluated. An in-world mechanism may not be labelled empirically
  fitted while its in-world use is `none`.

**Status on 2026-11-26 (workstream E, recovery cycle).** The fitted models are genuine offline models with
reproducible held-out reports, and they are now exposed to the simulator through the runtime decision-model API
`crucible.calibration.runtime` (contracts in `docs/v5/calibration/RUNTIME.md`). **No default world consumes any fitted
model yet**: every population decision, the social consumption engine and commerce demand run on hand-set rules
(records below). Wiring the runtime models into the unified population is wave 2 (DECISION_LOG D-15). The only
in-world use is an opt-in feed configuration (`feed` config `calibrated`), which no shipped world sets.

## Fitted offline models

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| model:content_consumption@1 | keep watching a short video shown in the feed (randomly inserted exposures) | `crucible.calibration` (offline; runtime decision `content_open@1`) | empirically fitted | kuairand_pure random-exposure log: Kuaishou short-video featured feed (China), 27,285 selected users, 7,583 pool videos, 2022-04-22..05-08; randomised exposure | held-out users 2022-05-03..05-08: log loss 0.443 (95% CI 0.434-0.453) vs popularity baseline 0.478; AUC 0.698; ECE 0.006; models fitted on the observational log predict 0.47 where 0.18 is observed | autoplay short video only; item ages 0-30 days, durations 6 s-20 min; one app, one month (includes a holiday); feature effects are associations among randomly shown items, not causal effects of popularity or interest | none (runtime decision model `content_open@1` available; not consumed by any world) |
| model:content_consumption_feed@1 | engagement with recommender-selected exposures (scoring target) | `crucible.calibration` (offline; runtime decision `content_open_recommender_target@1`) | empirically fitted | kuairand_pure standard log: Kuaishou featured feed, 27,285 users, 2022-04-12..05-08; observational | held-out users 2022-05-03..05-08: log loss 0.624 (95% CI 0.617-0.632) vs popularity baseline 0.665; AUC 0.708; ECE 0.006 | contains the production recommender's selection on unobserved preferences (predicts 0.47 against 0.18 observed under random exposure); a recommender scoring target, never an individual's choice rule | none (runtime decision model available; the in-world learned-engagement recommender trains on in-world outcomes instead) |
| model:watch_duration@1 | watch time of a shown short video, given consumed or skipped | `crucible.calibration` (offline; runtime decision `watch_time@1`) | empirically fitted | kuairand_pure, both logs pooled: Kuaishou featured feed, 27,285 users, 2022-04-12..05-08; observational | held-out users 2022-05-03..05-08: mean pinball 0.129 (95% CI 0.127-0.131) vs duration-decile baseline 0.138; PIT KS 0.011; 90% coverage 0.896 | short-video autoplay only; replays beyond about 4x duration not reproduced; on recommender-selected exposures alone it under-predicts long watches (PIT KS 0.084) | none (runtime decision model `watch_time@1` available) |
| model:return_daily@1 | whether an individual is active on a day given recent activity | `crucible.calibration` (offline; runtime decision `daily_return@1`) | empirically fitted | kuairand_pure, both logs: Kuaishou app activity of 27,285 users, 2022-04-22..05-08; observational | held-out users 2022-05-05..05-08: log loss 0.562 (95% CI 0.551-0.571) vs own-rate rule 0.610; AUC 0.704; ECE 0.026; 4-day forward simulation retention-curve error 0.076 vs 0.074 and 0.072 for the baselines (see the E3 review) | an active day is a day with a logged pool-video exposure; the random-insertion rate rose during the period; 17 days; includes the Labour Day holiday; no notification, content or social inputs | none (runtime decision model `daily_return@1` available) |
| model:session_rate@1 | sessions beyond the first on an active day (gamma-Poisson heterogeneity) | `crucible.calibration` (offline; runtime decision `extra_sessions@1`) | empirically fitted | kuairand_pure, both logs: sessions (30-minute gap) of 27,285 users, 2022-04-22..05-08; observational | held-out users: Poisson deviance 3.30 (95% CI 3.15-3.50) vs one pooled rate 3.72; individual-rate distribution KS 0.080 vs 0.144 | only sessions with a logged exposure are seen, so true counts are higher; held-out rates drift upward (count calibration error 0.43, mean predicted 1.32 vs 1.75 observed) | none (runtime decision model `extra_sessions@1` available) |
| model:news_click@1 | open a news article shown in a slate | `crucible.calibration` (offline; runtime decision `article_open@1`, offline only) | empirically fitted | mind_small: Microsoft News (US, English) impressions of 50,000 sampled users, 2019-11-09..15; observational | dev day 2019-11-15 (88% of impressions from users absent from training): log loss 0.1465 (95% CI 0.1455-0.1475) vs popularity baseline 0.1539; impression AUC 0.693; ECE 0.003 | production-recommender exposure; US news, November 2019; text affinity is a cosine in MIND's own TF-IDF/LSA space, which cannot be computed in a world; no dwell time; acquired from an unofficial mirror (BLOCKERS B-1) | opt-in: `feed` news_click@1 through social/consumption.py CalibratedOpen when a platform's feed config sets `calibrated` (default off; no shipped world sets it). Known defect: the opt-in path maps three inputs by their rank within the session batch (see feed.calibrated_open) |
| model:commercial_response@1 | per-arm visit and conversion probabilities under randomised campaign eligibility; 64 response types | `crucible.calibration` (offline; runtime decision `ad_response@1`) | empirically fitted | criteo_uplift_v2.1 (Criteo Uplift v2.1): pooled randomized incrementality tests of Criteo display advertising, 13,979,592 users sub-sampled non-uniformly by the publisher; advertisers and period undisclosed (released 2018); randomized experiment | user-level test (2.80M): pooled log loss visit+conversion 0.1143 vs 0.2092 constant-arm baseline; per-arm ECE <= 0.0004; visit Qini 0.087 (T-logistic 0.089); type-level correlation of predicted vs observed uplift 0.970 | absolute rates and effect sizes not representative; anonymised features; one campaign period, no dynamics or saturation; response independent of ad content, price and relevance (not identified); licence CC BY-NC-SA 4.0 (derived checkpoint is non-commercial) | none (runtime decision model `ad_response@1` available) |
| model:software_deep_engagement@1 | first star on a repository followed within 24 h by a fork, issue or pull request by the same developer | `crucible.calibration` (offline; runtime decision `repo_deep_engagement@1`) | empirically fitted | gharchive_2024-10-07_96h: GH Archive public GitHub events, 2024-10-07..10 UTC (Mon-Thu), 376,446 first stars by non-bot non-owner accounts; observational | time-based test (83,304 stars): log loss 0.1269 vs 0.1439 base rate (bootstrap diff -0.017 [-0.019, -0.015]), AUC 0.737, ECE 0.005; new-user holdout 0.1351 vs 0.1475 | public events only; 24 h horizon; observational, not the effect of promotion; GitHub population and one week; stars are not usefulness; top decile over-predicts (0.169 vs 0.131) | none (runtime decision model `repo_deep_engagement@1` available) |
| model:software_repo_growth@1 | next-day stars of a repository given today's activity (NB2 overdispersion) | `crucible.calibration` (offline; runtime decision `repo_star_growth@1`) | empirically fitted | gharchive_2024-10-07_96h: GH Archive public GitHub events, D1 -> D2 train, D2 -> D3 test (2024-10-08..10 UTC), repositories with a star, fork or creation that day; observational | test (321,303 repo-days): mean Poisson deviance 0.536 vs 0.732 proportional rule; NB2 alpha 0.848, simulated-vs-observed KS(log1p) 0.009; new-repository holdout 0.376 vs 0.525 | short horizon; inactive repositories excluded; observational; weekday only; Spearman slightly below the proportional rule (0.441 vs 0.463) | none (runtime decision model `repo_star_growth@1` available) |

## Runtime decision models (crucible.calibration.runtime)

A runtime decision model is a fitted checkpoint plus a contract that turns simulator quantities into the model's
features with fixed row-wise formulas (no batch-relative mapping). Its own record states what the contract assumes.

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| runtime:content_open@1 | simulator contract of content_consumption@1 (keep watching a shown feed item) | `crucible.calibration.runtime` | empirically fitted | kuairand_pure random-exposure log, Kuaishou featured feed, 2022-04-22..05-08 (as model:content_consumption@1) | as model:content_consumption@1; the runtime path reproduces the checkpoint's held-out predictions exactly (tests/test_v5_calibration_runtime.py) | topics are nearest topic centroids of the shared semantic space instead of KuaiRand's primary tags; own view counts scaled by a declared count_scale (default 1.0) because the data count only pool-video plays | none |
| runtime:content_open_recommender_target@1 | simulator contract of content_consumption_feed@1 (recommender scoring target) | `crucible.calibration.runtime` | empirically fitted | kuairand_pure standard log, Kuaishou featured feed, 2022-04-12..05-08 (as model:content_consumption_feed@1) | as model:content_consumption_feed@1 | a scoring target for recommenders only; same topic and count assumptions as runtime:content_open@1 | none |
| runtime:watch_time@1 | simulator contract of watch_duration@1 (watch seconds given consumed or skipped) | `crucible.calibration.runtime` | empirically fitted | kuairand_pure, both logs, 2022-04-12..05-08 (as model:watch_duration@1) | as model:watch_duration@1 | outputs quantiles in seconds (expm1 of the fitted log1p quantiles); sampling interpolates between quantile levels; same topic and count assumptions as runtime:content_open@1 | none |
| runtime:daily_return@1 | simulator contract of return_daily@1 (active on a day) | `crucible.calibration.runtime` | empirically fitted | kuairand_pure, both logs, 2022-04-22..05-08 (as model:return_daily@1) | as model:return_daily@1 | feed exposures scaled by a declared count_scale (default 1.0); no content or social inputs | none |
| runtime:extra_sessions@1 | simulator contract of session_rate@1 (extra sessions on an active day) | `crucible.calibration.runtime` | empirically fitted | kuairand_pure sessions, 2022-04-22..05-08 (as model:session_rate@1) | as model:session_rate@1 | posterior predictive gamma-Poisson draws; individual rates assumed stable | none |
| runtime:article_open@1 | simulator contract of news_click@1 | `crucible.calibration.runtime` | empirically fitted | mind_small, Microsoft News, 2019-11-09..15 (as model:news_click@1) | as model:news_click@1 (held-out evaluation in the model's own feature space only) | status offline_only: the LSA text cosine and whole-slate impression counts cannot be computed in a world without a batch-relative mapping, so evaluate() refuses simulator inputs | none |
| runtime:ad_response@1 | simulator contract of commercial_response@1 (persistent response type x campaign arm) | `crucible.calibration.runtime` | empirically fitted | criteo_uplift_v2.1, randomized incrementality tests, released 2018 (as model:commercial_response@1) | as model:commercial_response@1; type-level uplift correlation 0.970 on held-out users | individuals draw a type once; the response is independent of ad content, price, relevance and frequency | none |
| runtime:repo_deep_engagement@1 | simulator contract of software_deep_engagement@1 | `crucible.calibration.runtime` | empirically fitted | gharchive_2024-10-07_96h, 2024-10-07..10 UTC (as model:software_deep_engagement@1) | as model:software_deep_engagement@1 | event counts of the simulated code-hosting ecosystem stand in for GitHub's public events | none |
| runtime:repo_star_growth@1 | simulator contract of software_repo_growth@1 | `crucible.calibration.runtime` | empirically fitted | gharchive_2024-10-07_96h, 2024-10-08..10 UTC (as model:software_repo_growth@1) | as model:software_repo_growth@1 | same as runtime:repo_deep_engagement@1; NB2 sampling | none |

## In-world mechanisms the fitted models are meant to ground

These records describe what actually runs in a simulated world today.

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| population.content_choice | which shown item an individual opens | `feed` (BUILTIN open rule, C2 engine), `pop` (decision `continue` of the web activity) | plausible, unvalidated | none: hand-set multinomial-logit coefficients (feed.behaviour.open, pop.decision.continue) | none in world; offline comparison with KuaiRand in the E3 review (reviews/content_consumption.json) | the fitted content_consumption@1 is not used; the in-world open rule is invariant to everything but affinity, familiarity, following, share source, seen, satisfaction and length | none |
| population.continue_watching | keep watching after opening | `feed` (bounce + exponential hazard) | plausible, unvalidated | none: hand-set (feed.behaviour.watch) | none in world; offline comparison in the E3 review (reviews/watch_duration.json) | the fitted models are not used | none |
| population.watch_duration | watch time of consumed items | `feed` (exponential abandonment hazard, truncated at duration) | inspired by research | survival models of dwell time (Liu, White & Dumais 2010); constant-hazard special case, parameters not estimated | none in world; offline comparison in the E3 review | watch_duration@1 is not used; no position or device effects | none |
| population.daily_return | whether and when an individual comes back | `pop` (rhythm: log-normal mean gap 7 h or 13 h by segment, session_start logit), `feed` (exponential return gap, 6 h mean) | plausible, unvalidated | none: hand-set (pop.decision.session_start, feed.behaviour.return) | none in world; offline comparison in the E3 review (reviews/return_daily.json) | return_daily@1 is not used; activity is not tied to observed history beyond satisfaction and habit | none |
| population.session_frequency | sessions per active day | `pop`, `feed` (as population.daily_return) | plausible, unvalidated | none: hand-set | none in world; offline comparison in the E3 review (reviews/session_rate.json) | session_rate@1 is not used; individual rates are latent log-normal draws unrelated to data | none |
| population.article_choice | which article in a slate an individual opens | `feed` (BUILTIN open rule by default) | plausible, unvalidated | none by default; with the opt-in configuration the open probability comes from news_click@1 (MIND-small, 2019-11-09..15) | none in world; offline comparison in the E3 review (reviews/news_click.json) | the default rule is hand-set; the opt-in path has the batch-rank defect recorded at feed.calibrated_open | opt-in: `feed` news_click@1 (CalibratedOpen; config `calibrated`, default off) |
| social.learned_engagement_score | the learned-engagement recommender's scoring target | `feed` (IRLS logistic on delayed, label-noisy in-world outcomes) | inspired by research | none: trained on in-world outcomes, not on data | none | content_consumption_feed@1 is not used as its target | none |
| advertising_intervention_response | how simulated individuals respond to advertising | `pop` (purchase decision), 4.0 `ads` | known unrealistic | none: no v5 population decision reads ad exposure, so advertising has no effect on simulated individuals | none in world | commercial_response@1 (Criteo, randomised) is fitted but not wired; ads change only what is shown | none |
| advertising_response_heterogeneity | heterogeneity of advertising response | `pop` | known unrealistic | none: there is no advertising response to be heterogeneous | none | see advertising_intervention_response | none |
| developer_light_to_deep_engagement | developers escalating from a star to contributions | `pkg`, `adopt` | known unrealistic | none: no simulated developer population; stars and contributions come only from agents and scripted adopters | none | software_deep_engagement@1 is fitted but not wired | none |
| repository_adoption_growth | how repository attention grows day to day | `pkg` | known unrealistic | none: no simulated star dynamics; only agents and scripted adopters act | none | software_repo_growth@1 is fitted but not wired | none |
| new_repository_adoption_curve | early adoption of new repositories | `pkg` | plausible, unvalidated | no fitted or validated mechanism. Measured reference (not fitted, no held-out evaluation): gharchive_2024-10-07_96h, the 205,095 repositories created on 2024-10-08 UTC: 0.57% received a non-owner non-bot star within 24 h, 0.74% within 48 h | none: descriptive reference only | until a simulator mechanism is checked against this curve, new-repository adoption in the simulator is an assumption; one creation day, 48 h horizon | none |

## Population decision models (crucible.population.models)

Every class 1 decision is a logistic model with hand-set coefficients. `population.models.build` tries to plug in
fitted models whose feature names equal the population's feature names; no registered model satisfies that, so
every decision below runs its plausible coefficients in every world (audit E S6).

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| pop.decision.session_start | start a session when activated | `pop` | plausible, unvalidated | none: coefficients set by hand (attention 2.2, satisfaction 0.9, habit 0.35, commitment -1.6, woken 1.5) | none in world; offline comparison with KuaiRand activity in the E3 review | the name `browse_session@1` it tries to load does not exist; daily_return@1 / extra_sessions@1 are the fitted alternatives (wave 2) | none |
| pop.decision.continue | read another page of the same site | `pop`, `popweb` | plausible, unvalidated | none: hand-set (page utility 1.6, attention 1.2, ...) | none | content_consumption@1's features are not population features, so the load is discarded | none |
| pop.decision.evaluate_product | open a product page when a site lists products | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | no fitted model | none |
| pop.decision.purchase | buy the evaluated product | `pop`, `popweb` | plausible, unvalidated | none: hand-set (price ratio 1.4, relevance 1.2, ...) | none in world; E4 fits purchase and price response on Online Retail II (offline) | `commercial_response@1` / `purchase@1` loads are discarded or absent; no response to advertising | none |
| pop.decision.abandon | leave a resource that no longer provides value | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | `churn@1` does not exist; E4 fits customer churn offline | none |
| pop.decision.recommend | recommend a resource to contacts | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | no fitted model | none |
| pop.decision.join_community | join a community after reading it | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | no fitted model; E4 fits community participation (GitHub) offline | none |
| pop.decision.follow_creator | follow the author of a post read | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | no fitted model | none |
| pop.decision.participate | reply in a discussion read | `pop`, `popweb` | plausible, unvalidated | none: hand-set | none | no fitted model; E4 fits return participation (GitHub) offline | none |

## Social consumption engine (crucible.social.consumption.BUILTIN)

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| feed.behaviour.open | open a feed item: logit of affinity, familiarity, following, share source, seen, satisfaction, length against the next 3 items and not opening | `feed` | plausible, unvalidated | form inspired by multinomial-logit choice; parameters set by hand | none in world; offline comparison on KuaiRand and MIND rows in the E3 review | text posts and articles only; affinity is a cosine in the lexical 32-d space by default | none |
| feed.behaviour.watch | bounce, then exponential abandonment hazard falling with affinity, familiarity, satisfaction and rising for repeats; scroll time per slot | `feed` | inspired by research | Weibull/hazard models of dwell time (Liu, White & Dumais 2010); exponential special case, parameters not estimated | none in world; offline comparison on KuaiRand in the E3 review | exponential, not Weibull; no position or device effects | none |
| feed.behaviour.engagement | like / share / follow after a qualifying view (logits; shares scaled by an individual log-normal propensity) | `feed` | plausible, unvalidated | none | none | share and follow rates are assumptions; no unfollow | none |
| feed.behaviour.continuation | continue the session (logit of satisfaction, view, minutes spent) | `feed` | plausible, unvalidated | none | none | no fatigue beyond minutes spent; E4 session_length (KuaiRand) is the fitted alternative | none |
| feed.behaviour.return | return gap (exponential, 6 h mean shrinking with satisfaction; wake-up after a share notification with probability 0.3) | `feed` | plausible, unvalidated | none | none in world; offline comparison in the E3 review | no daily rhythm beyond the attention reset | none |
| feed.behaviour.state | satisfaction, platform habit and taste drift (2 % toward consumed content), need weight | `feed` | plausible, unvalidated | none | none | one satisfaction scalar; drift without forgetting | none |
| feed.calibrated_open | opt-in article open from news_click@1 | `feed` (social/consumption.py CalibratedOpen) | empirically fitted | mind_small, Microsoft News, 2019-11-09..15 (news_click@1 card) | E's held-out AUC on MIND; in-world no validation | defect: text_cos, subcat_share and pop_ctr_24h are quantile-mapped by rank within the session batch, so opening probability is invariant to uniformly better content and depends on other viewers' slates; replaced by the runtime contract article_open@2 once registered | opt-in: `feed` news_click@1 (config `calibrated`; default off) |
| social.textrep | C2 content representation of text items (random indexing: hashed word vectors, 1+log tf, L2) | `aud` (social/textrep.py) | deliberate simplification | none (deterministic function of the artifact bytes) | n/a | lexical, not semantic; retired from distribution by D-16 (workstream D) | none |
| social.contact_graph | directed contact graph: 8 contacts in the community block and 4 long-range per individual, hash-computed | `crowd` (social/scale.py ContactGraph) | deliberate simplification | none | none | static; no degree heterogeneity or triadic closure | none |
| social.creator_writing | crowd creators' bag-of-topic-words posts | `crowd` (social/scale.py write_text) | known unrealistic | none | none | real artifacts but not natural prose | none |
| social.creator_analytics | creator analytics every 6 h, staged 6 h earlier, log-normal noise sd 0.05 | `aud` | plausible, unvalidated | measured in SP-4.5 against the exact recount (noise log-sd 0.053, median age 6 h) | n/a | delay and noise are configuration | none |
| social.recommenders | twelve recommendation algorithms fused by reciprocal rank | `feed` (social/recsys) | inspired by research | Hu, Koren & Volinsky 2008; Carbonell & Goldstein 1998; BM25; Thompson sampling; Cormack et al. 2009 | n/a (algorithms, not behaviour) | parameters by hand; pools of a few hundred items | none |
| social.cascade_runs | large-scale reach runs (positive control 10M, organic 2.5M) | `crowd`, `feed` (social/cascade.py) | known unrealistic | none: capacity and mechanism demonstrations | n/a | cascade size and shape are not claimed realistic | none |

## Commerce rules (F1)

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| patrons.subscription_choice | subscription buyer choice (value = wtp x expected quality - price + trial bonus) | `patrons` | plausible, unvalidated | none | none | one need per buyer; no switching while satisfied | none |
| patrons.usefulness_check | buyer usefulness check (own instances scored by the need's exact verifier) | `patrons` | deliberate simplification | none | none | needs with exact verifiers only | none |
| patrons.exchange_choice | goods sellers' exchange choice ((buyers + 1) x (1 - take rate), switch margin) | `patrons` | plausible, unvalidated | none | none | one exchange per seller, cost-plus prices | none |
| patrons.listing_choice | goods buyers' listing choice (cheapest acceptable on a trusted exchange) | `patrons` | plausible, unvalidated | none | none | trust from the public dispute record only | none |
| subs.dunning | dunning schedule (retries after 1, 3, 7 days) | `subs` | plausible, unvalidated | none | none | fixed schedule | none |
| subs.calendar | billing month 30 days, year 365 days | `subs` | deliberate simplification | none | none | no calendar | none |

## Kernel modules of the v5 behaviour packages

One record per kernel module. A module record states which behaviour the module contains (or that it contains none
and only provides tools whose use is decided by participants); the decisions inside a module have their own records
above.

| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use |
|---|---|---|---|---|---|---|---|
| module:pop | persistent individuals: activation rhythm, attention budgets, satisfaction, habits and the class 1 decisions | `pop` | plausible, unvalidated | none: every profile parameter (segment shares, gaps, budgets, incomes) is set by hand | none | the decisions are pop.decision.*; fitted models load but are discarded (feature names never match) | none |
| module:pop_adaptive | class 2 shared adaptive policies (GRU memory, online learning) choosing resource visits | `pop_adaptive` | plausible, unvalidated | none | none | learned from in-world outcomes, not from data; network size and learning rate by hand | none |
| module:pop_strategic | class 3 strategic entrepreneurs choosing topic, price and content volume of their sites | `pop_strategic` | plausible, unvalidated | none | none | policies learn from in-world revenue only | none |
| module:popsites | web sites operated by simulated organisations (generated applications) | `popsites` | deliberate simplification | none | none | generated site templates; content written from topic vocabularies | none |
| module:popweb | resource discovery, visits, use, return, abandonment and purchases of the population | `popweb` | plausible, unvalidated | none | none | uses pop.decision.continue / evaluate_product / purchase / abandon / recommend / join_community / follow_creator / participate | none |
| module:crowd | C2 stand-in population of the social scale path (topic-mixture interests, contact graph) | `crowd` | deliberate simplification | none | none | used only where `pop` is absent; retired from v5 Tier 2/3 worlds by D-15 | none |
| module:aud | content index and exact audience accounting (impressions, views, unique viewers) | `aud` | deliberate simplification | none: accounting, not behaviour; the view threshold min(30 s, 50 % of duration) is a platform convention chosen by hand | n/a | a lexical encoder by default (social.textrep) | none |
| module:feed | recommendation and the consumption engine for every platform | `feed` | plausible, unvalidated | none by default (feed.behaviour.*) | none | see feed.behaviour.* and feed.calibrated_open | opt-in: `feed` news_click@1 (config `calibrated`, default off) |
| module:content | the shared content catalogue (publish, update, remove, inspect) | `content` | deliberate simplification | none: a tool surface; behaviour comes from whoever publishes | n/a | update does not re-index the distributed representation (audit C 4.1) | none |
| module:studio | sandboxed creator tools for ten artifact formats | `studio` | deliberate simplification | none: tools | n/a | lightweight renderers (text cards, tones, slideshows) | none |
| module:media | media creator tools of the semantic package | `media` | deliberate simplification | none: tools | n/a | duplicates part of `studio` | none |
| module:embed | the world's record of pretrained content embeddings | `embed` | deliberate simplification | none: fixed pretrained encoders (MiniLM, CLIP, AST, Whisper) stand in for human perception | n/a | insensitive to negation and sentiment (audit D) | none |
| module:semantic_inference | operator-side recording of shared small-LM inference | `semantic_inference` | deliberate simplification | none: recording infrastructure | n/a | prompts are stored as digests only | none |
| module:social | follow graph, shares, profiles and timelines | `social` | deliberate simplification | none: a tool and batch surface; decisions come from participants | n/a | population batch APIs are not yet called by the population | none |
| module:social_ids | principal-to-node mapping for non-population principals | `social_ids` | deliberate simplification | none: bookkeeping, no behaviour | n/a | none beyond bookkeeping | none |
| module:social_rows | row counters of the social relation tables | `social_rows` | deliberate simplification | none: bookkeeping, no behaviour | n/a | none beyond bookkeeping | none |
| module:msg | private direct and group messaging | `msg` | deliberate simplification | none: tools | n/a | the population does not message | none |
| module:community | communities: membership spells, governance, moderation, events | `community` | plausible, unvalidated | none: in tests and gates communities are founded and joined by scripted reference participants | none | no population participation path yet; E4 fits GitHub return participation offline | none |
| module:creators | creator memberships (tiers paid from fans' own accounts) | `creators` | deliberate simplification | none: billing tools | n/a | a second recurring-revenue system beside `subs` (audit F) | none |
| module:letters | newsletters | `letters` | deliberate simplification | none: tools | n/a | no population readership model | none |
| module:launch | product launches linking to product pages; upvotes and comments | `launch` | deliberate simplification | none: tools | n/a | the population never discovers products here yet | none |
| module:posts | text post platform | `posts` | deliberate simplification | none: platform surface | n/a | consumption decisions are feed.behaviour.* | none |
| module:blog | article and blog platform | `blog` | deliberate simplification | none: platform surface | n/a | as module:posts | none |
| module:pics | image platform | `pics` | deliberate simplification | none: platform surface | n/a | images are not distributed with the default lexical encoder | none |
| module:pods | audio and podcast platform | `pods` | deliberate simplification | none: platform surface | n/a | as module:pics | none |
| module:shorts | short-video platform | `shorts` | deliberate simplification | none: platform surface | n/a | the KuaiRand models are its natural fitted behaviour but are not wired | none |
| module:tube | long-form video platform | `tube` | deliberate simplification | none: platform surface | n/a | no fitted long-video behaviour | none |
| module:dev | developer communities, packages and Q&A | `dev` | deliberate simplification | none: platform surface | n/a | no fitted developer behaviour in world | none |
| module:kb | knowledge sharing Q&A and educational products | `kb` | deliberate simplification | none: platform surface | n/a | none | none |
| module:patrons | independent customers with their own money (one need each, verifier-scored usefulness) | `patrons` | plausible, unvalidated | none | none | decisions are patrons.*; replaced by the unified population in v5 Tier 2/3 worlds (D-15) | none |
| module:subs | subscriptions, trials, dunning, cancellation, proration, refunds | `subs` | deliberate simplification | none: billing law, not behaviour | n/a | see subs.dunning, subs.calendar | none |
| module:products | products, offers, purchases, licences, measured use | `products` | deliberate simplification | none: commerce law and tools | n/a | seats are not enforced (audit F) | none |
| module:ventures | business formation, wind-down, insolvency and world-generated competitors | `ventures` | plausible, unvalidated | none: incumbents created at genesis never change price, quality or capacity | none | static competitors | none |
| module:xchg | participant-run marketplaces with take rates and disputes | `xchg` | deliberate simplification | none: market rules | n/a | none beyond its rules | none |
| module:inst | charters, decision procedures, offices, projects | `inst` | deliberate simplification | none: institutional law | n/a | evaluator metrics readable through charter conditions (audit F) | none |
| module:commit | escrowed commitments with negotiation and arbitration | `commit` | deliberate simplification | none: contract law | n/a | none beyond its rules | none |
| module:labor | labour market with simulated workers (reservation wages, private skills) | `labor` | plausible, unvalidated | none: task quality = skill match plus noise | none | work produces no artifact | none |
| module:adopt | need-driven capability adopters that test and adopt services | `adopt` | plausible, unvalidated | none | none | scripted test-and-adopt rule; no fitted developer behaviour | none |
| module:caps | typed capability registry | `caps` | deliberate simplification | none: registry tools | n/a | none | none |
| module:needs | needs graph derived from world state | `needs` | deliberate simplification | none: graph derivation rules | n/a | six relation types chosen by design | none |
| module:pkg | software packages, releases, verified adoption | `pkg` | deliberate simplification | none: hosting and release rules | n/a | stars and adoption come only from agents and adopters | none |
| module:infra | compute providers created at genesis | `infra` | deliberate simplification | none: configured providers | n/a | static providers | none |
| module:sci | experimental science domain (hidden laws, apparatus, publications) | `sci` | deliberate simplification | none: generated phenomena | n/a | laws are generated, not empirical | none |
| module:seat | participant seats of Tier 2 domain worlds | `seat` | deliberate simplification | none: bookkeeping | n/a | none beyond bookkeeping | none |

The 55 mechanisms registered by Crucible 4.0 (`evaluation/fidelity.py`, `docs/experiments/ow_fidelity.json`) keep
their 4.0 categories: 37 plausible, 18 known unrealistic, none calibrated. v5 records are added by the workstreams that
build each mechanism and checked by SP-5.4.
