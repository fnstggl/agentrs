# Crucible 5.0 fidelity register

Every important behavioural or economic mechanism has a record here (Part 5.4). Categories:

* **empirically fitted** - parameters estimated from a named dataset, with held-out evaluation;
* **empirically validated** - not fitted to the data, but checked against it on prespecified metrics;
* **inspired by research** - form taken from published work, parameters not estimated;
* **plausible, unvalidated** - a reasoned assumption;
* **deliberate simplification** - known to omit something, on purpose;
* **known unrealistic** - kept for control or tractability, and labelled.

A neural network is not "calibrated" because it is a neural network, and fitting a historical dataset does not show
real-world predictive validity. Each record states the population, platform and period of its evidence and its limits.

| mechanism | module | category | evidence (dataset, period) | held-out result | limits |
|---|---|---|---|---|---|
| C2 content representation of text items (random indexing: hashed word vectors, 1+log tf, L2) | social/textrep.py | deliberate simplification | none (deterministic function of the artifact bytes; Johnson-Lindenstrauss projection of the weighted bag of words) | n/a | lexical, not semantic: synonyms are not close; 32 dimensions add cosine noise of about 0.18 between unrelated texts; replaced by workstream D's encoders through `aud.encoder="semantic"` |
| C2 choice to open a feed item (logit utility of affinity, familiarity, following, share source, seen, satisfaction, length, against the next 3 items and not opening) | social/consumption.py `BUILTIN open_*` | plausible, unvalidated | form inspired by multinomial-logit choice among competing alternatives; parameters set by hand | none | not fitted; text posts and articles only; E1's content_consumption@1 (KuaiRand, autoplay short video) is not used because its card excludes text and articles |
| C2 reading/watch time (bounce, then exponential abandonment hazard falling with affinity, familiarity, satisfaction, rising for repeats; truncated by duration and attention left) | social/consumption.py `BUILTIN bounce_*, hazard_*` | inspired by research | survival (hazard) models of dwell time, e.g. Liu, White & Dumais 2010 (Weibull dwell time); constant-hazard special case, parameters not estimated | none | exponential (not Weibull) hazard; no position or device effects; E1's watch_duration@1 covers short video only |
| C2 like / share / follow decisions after a qualifying view (logits of affinity, completion, satisfaction, familiarity; shares scaled by an individual log-normal propensity) | social/consumption.py `BUILTIN like_*, share_*, follow_*` | plausible, unvalidated | none | none | share rate per view (about 2-4 % in the default configuration) and follow rate are assumptions; no unfollow |
| C2 session continuation (logit of satisfaction, a view, minutes spent) and return gap (exponential, mean shrinking with satisfaction; wake-up after a share notification with probability 0.3) | social/consumption.py | plausible, unvalidated | none | none | E1's session_rate@1 / return_daily@1 are not wired in yet; no daily rhythm beyond the attention reset |
| C2 satisfaction, platform habit and taste drift (satisfaction moves with watch fraction x (affinity - 0.2); habit by session value; interests drift 2 % toward consumed content) | social/consumption.py | plausible, unvalidated | none | none | single satisfaction scalar per individual; habit is a softmax over platforms; drift has no forgetting |
| C2 population interests (community topic mixtures from 24 topic vocabularies plus individual noise) | social/scale.py `crowd` | deliberate simplification | none | none | interests live in the lexical representation space; communities are contiguous index blocks; used only when workstream B's population is absent |
| C2 contact graph (directed, per individual 8 contacts in its community block and 4 uniform long-range contacts, computed by a counter-based hash, never stored) | social/scale.py `ContactGraph` | deliberate simplification | none | none | static; no degree heterogeneity or triadic closure; shares go to every contact |
| C2 creators' writing (posts and articles from one main and one secondary topic, Zipf-weighted words, focus drawn per post) | social/scale.py `write_text` | known unrealistic | none | none | bag-of-topic-words text, kept for tractability; real artifacts (bytes) but not natural prose; no creator learns from its analytics |
| C2 creator analytics (published every 6 h from values staged 6 h earlier, multiplicative log-normal noise sd 0.05, one draw per publication) | social/audience.py `aud.analytics` | plausible, unvalidated | measured in SP-4.5: noise log-sd 0.053 against the exact recount, median age 6 h | n/a | delay and noise are configuration, not estimated from any platform |
| C2 recommendation mechanisms (chronological, interest, implicit ALS, logistic engagement on delayed noisy labels, batched Thompson sampling, social, MMR, reputation, BM25 search, freshness, trending, random) | social/recsys/algorithms.py | inspired by research | Hu, Koren & Volinsky 2008 (implicit ALS); Carbonell & Goldstein 1998 (MMR); Robertson et al. (BM25); Thompson sampling; reciprocal-rank fusion (Cormack et al. 2009) | n/a (algorithms, not behaviour) | parameters chosen by hand; fusion weights per platform are configuration; no production-scale candidate retrieval (pools of up to a few hundred items) |
| C2 large-scale reach runs (positive control at 10M identities; organic run at 2.5M) | social/cascade.py | known unrealistic | none: capacity and mechanism demonstrations only | n/a | cascade size and shape are not claimed to be empirically realistic; the positive control is designed to cascade (favoured topic, seeded post, trending-weighted mixture, higher sharing) |

The 55 mechanisms registered by Crucible 4.0 (`evaluation/fidelity.py`, `docs/experiments/ow_fidelity.json`) keep
their 4.0 categories: 37 plausible, 18 known unrealistic, none calibrated. v5 records are added by the workstreams that
build each mechanism and checked by SP-5.4.
