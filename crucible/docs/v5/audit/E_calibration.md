# Audit E_calibration: empirical grounding (Part 5, Gate F, SP-5.4, SP-5.5)

Auditor: independent audit, worktree `feat/crucible-5-complete` at `a881d18`. No existing file was modified. Probe
scripts and logs: `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/E_calibration/`.

## Bottom line

* **The offline calibration is genuine.** The four datasets are real and match their publishers' checksums, which I
  checked against outside sources. The splits are prespecified and hold out both users and later time. Baselines,
  calibration metrics, cluster-bootstrap uncertainty and checkpoints are all there. I recomputed the held-out
  metrics exactly: two with my own code (Criteo on the full 2.8M-user test split; KuaiRand `session_rate` from the
  raw CSVs), and six more with the project's reproducers. The Criteo model keeps the randomised intention-to-treat
  (ITT) effect separate from the confounded comparison of exposed and unexposed users.
* **The fitted models are not used by the simulated world.** No population decision uses a fitted model, and none
  can: the plug-in requires the fitted feature names to equal population feature names, which never happens, and the
  mismatch is discarded silently. Of the nine checkpoints, only `news_click@1` can be reached from world code. That
  happens only when a feed config opts in, and the default is off. The one Tier 2 domain that uses it (`learn`) is
  not registered. `commercial_response@1` (Criteo), `watch_duration`, `return_daily`, `session_rate`,
  `content_consumption`, `content_consumption_feed` and both software models are never consumed. Every agent today
  learns against hand-set "plausible, unvalidated" behaviour.
* **The fidelity register is not honest about this.** It lists more than ten mechanisms as "empirically fitted ... via
  `calibration.load(...)`" in pop, social, commerce and software. At runtime none of them is. SP-5.4 fails:
  40 modules have no record, and the parser silently drops 17 rows that follow a blank line.
* **The one integrated model has a design artifact.** `CalibratedOpen` quantile-maps affinity by rank within the
  session batch. Opening probabilities are therefore exactly invariant to uniformly better content, and they depend on
  which other viewers share the batch.

## What I ran

| # | What | Result |
|---|---|---|
| p1 | `datasets.verify` and `validate_schema` for all four datasets (project code) | all files `ok`, no schema problems |
| p1b | own `sha256sum`/`md5sum`/`wc -l` on raw files | KuaiRand tar md5 `0820331067a3784d9691136f772b35a7`, sha256 `c814bf6f...`; MIND 4 TSVs; Criteo csv.gz `2716e1bf...`; row counts 156,965/73,152 impressions, 51,282/42,416 news; KuaiRand 1,141,112 / 295,497 / 1,186,059 log rows, 27,285 users, 7,583 videos |
| ext | outside cross-check (public APIs, no personal data sent) | Zenodo record 10439422 lists KuaiRand-Pure md5 `0820331067a3...` and 47,432,272 bytes (match). The Criteo-owned HF repo LFS oid is `2716e1bf...` (match). The `huyva/MIND-small` mirror LFS oids match all four files. I re-downloaded GH Archive `2024-10-07-0.json.gz`: sha256 `2fcde2e1...` matches the manifest, the file has 175,507 events, and the counts of all 16 event types equal the extracted `.npz` (0 mismatches) |
| p5 | **own recomputation, commercial_response@1, full test split (2,795,919 users)**: own split from the plan text (PCG64 seed 20261125, 60/20/20) and own log loss, AUC and ECE; checkpoint loaded via `registry.load` | primary (visit+conversion pooled log loss) **0.11425529** vs reported 0.11425529; visit log loss 0.1024365, AUC 0.9473957, conversion log loss 0.0118187 (all equal); my own constant-arm baseline 0.2091478 (= reported). ITT difference in means 0.01031 (se 0.00033), model-implied ATE 0.00738. Naive exposed-minus-unexposed among the treated is **0.378**, about 37x the ITT, so ignoring randomisation would be badly wrong; the report uses only ITT and a labelled Wald complier estimate. Observed uplift by predicted decile rises monotonically from decile 2 to decile 9 (0.058 observed vs 0.066 predicted in the top decile). In decile 0 the predicted effect is negative (-0.0059) but the observed effect is +0.0037 (documented in the card) |
| p6 | **own recomputation, session_rate@1** from raw KuaiRand CSVs (own 30-min sessions, own user hash split, own deviance) | test users 3,779; Poisson deviance **3.30146** = reported; B0 3.72459 = reported. Mean predicted rate is 1.32 against 1.75 observed: a 24% under-prediction on held-out days (the card discloses count calibration error 0.43) |
| p6b | own rebuild of the content_consumption `rnd_test` split | 89,752 rows, 3,805 users, click rate 0.18645, B0 log loss 0.4815997 (= reported) |
| p6c | own count on MIND dev | 2,740,998 impression items, click rate 0.040636 (= card `n`, `mean_obs`); 88.1% of dev users are absent from train |
| p9 | project reproducers (`reproduce_kuairand`, `reproduce_news`, `reproduce_software`) via the slot scheduler | content_consumption 0.4425515, content_consumption_feed 0.6238224, watch_duration 0.1293643, return_daily 0.5617287, software_deep_engagement 0.1268607 and software_repo_growth 0.5361287: abs diff 0.0; news_click 0.1464839 on 2,740,998 rows: abs diff 2.8e-17 (216 s on 1 CPU) |
| p3 | `population.models.build({})` with each card's features | all 9 decisions `plausible:*`. `content_consumption@1` loads, but none of its 13 features is in `population.models.FEATURES` except `hour_sin`/`hour_cos`. All 12 `commercial_response@1` features (f0..f11) are outside FEATURES. `browse_session@1`, `continue_watching@1`, `purchase@1` and `churn@1` do not exist |
| p8 | **integrated run**: `population_world('development', n=400)` advanced one simulated day, plus a default social world, with a spy on `calibration.load` | population world: `load` was called for content_consumption@1, commercial_response@1 and 4 absent names, and every result was discarded; all decisions `plausible, unvalidated`. Default social world: 0 load calls, `engine.calibrated == []` |
| p4 | `crucible.load_all_modules()` | 89 modules; `learn` (the Tier 2 education domain that uses `SlateClicks`/news_click) **not registered**; feed `DEFAULTS["calibrated"] == {}` |
| p7 | `CalibratedOpen.p_open` probe in the test's calibrated social world | affinity +0.3 for every slot: max \|dp\| = **0.0**; affinity x0.2: max \|dp\| = **0.0**. Viewer 0's eight probabilities change when scored alone (e.g. 0.1645 -> 0.1272) or when other viewers' items become more relevant (0.1765 -> 0.1470) |
| t1 | `pytest tests/v5/test_gate_F.py tests/v5/test_gate_SP_5_4.py tests/v5/test_gate_SP_5_5.py -m "not slow"` | 12 passed, **1 failed** (SP-5.4: `important_mechanisms_without_record=40`), 1 deselected (F.3 is `@slow`) |
| t2 | `pytest tests/test_v5_calibration_core.py tests/test_v5_social_calibrated.py` | 24 passed |
| p2 | `fidelity.sp54_metrics()` | 75 important, 35 complete, 40 missing (`module:feed`, `aud`, `content`, `creators`, `community`, `products`, `ventures`, `labor`, `pop_adaptive`, `pop_strategic`, `popweb`, ...); v5 rows parsed: **18 of 35** |

## Subsystems

### S1 Dataset acquisition, checksums, provenance, licences: REPAIR (S)
`src/crucible/calibration/datasets.py:57-206` (KuaiRand, MIND recipes), `criteo.py`, `gharchive.py`, manifests
`docs/v5/calibration/datasets/*.json`, `docs/v5/calibration/DATASETS.md`.
Sound and verified: SHA-256 on every file, safe tar extraction, schema and row-count validation, and provenance with
population, platform and period. The checksums match outside sources (see "ext" above).
Defects:
1. MIND-small comes from an unofficial mirror (`huyva/MIND-small`). That route skips the licence acceptance the
   official gated repository requires, and the MSR licence is research and non-commercial only. Matching row counts do
   not prove the files are identical to the official ones.
2. The GH Archive tables keep raw numeric GitHub actor ids (`gharchive.py:15`). Anyone can re-identify those through
   the public API (Part 5.1: "do not ... reconstruct individual real-world identities"). They should be salted-hashed
   at extraction.
3. GH Archive raw checksums are recorded on first download (trust on first use). I confirmed one hour independently.
4. The committed checkpoints derive from CC BY-NC-SA data (Criteo) and MSR non-commercial data (MIND). The repository
   has no licence notice for them.

### S2 Offline fitting and held-out evaluation pipelines: EXTEND (M)
`fit_e1.py`, `fitlib.py`, `kuairand.py`, `mind.py`, `metrics.py` (E1); `fit_commercial.py`, `fit_software.py`,
`scoring_e2.py`, `fitmeta_e2.py`, `gbdt_e2.py` (E2).
Correct on what exists:
* History features use strictly earlier days (`kuairand.py:107-190`, `PriorCounts.before`).
* Splits hold out users (hash bucket of 85 or more) and later days (`fit_e1.py:67-104`).
* There are simple baselines, cluster-bootstrap CIs, reliability tables, and the observational-versus-randomised
  contrast (observational models predict 0.47 where 0.18 is observed under random exposure).
* Plans were committed before the fits (`git merge-base --is-ancestor d6f4767 d86f683`; the plan sha256 `5ae089ed...`
  matches).

Missing or defective against the full spec:
1. There is no comparison with the simulator's **existing rule-based behaviour** (Part 5.3). The baselines are
   constants and popularity, never the `population.models` or `social.consumption.BUILTIN` rules that the fitted
   models are meant to replace.
2. Selection ignores "long-horizon stability and computational efficiency" (Part 5.3). The only forward simulation
   is four days, and there the kept `return_daily` model's retention-curve error (0.076) is *worse* than both
   baselines (0.074, 0.072).
3. `commercial_response@1` is selected by outcome log loss, not by uplift accuracy, and it ranks uplift worse than
   T-logistic (Qini 0.087 vs 0.089 for visits, 0.159 vs 0.200 for conversions).
4. Two parallel toolchains and two card schemas exist (`registry.py:9-31`). They work, but this is duplication.
5. Amendment 1 (`plans/E2_amendment_1.plan.json`) added `t_gbdt_reg` and `gbdt_poisson_reg` *after* run-1 test
   metrics had been computed, and both added candidates became the kept models. This is disclosed, and the conclusions
   would not change (run-1 `t_gbdt` 0.1191 also beats the baseline 0.2091), but the test split was used twice.

### S3 Model registry and checkpoints: RETAIN
`registry.py`, `predictors.py`, `portable.py`, `registry_e2.py`.
All nine models load, with SHA-256 verified on every load. Checkpoints use npz without pickle and are each under
5 MB. The F.6 tamper test passes. The portable predictions equal sklearn's (unit tests pass), and the project's
reproducers return abs diff 0.0.

### S4 Behavioural-domain coverage (Part 5.3): EXTEND (L)
Fitted today:
* content selection on Kuaishou short video and MIND news;
* watch duration;
* browsing frequency (two models);
* advertising ITT response;
* two short-horizon GitHub adoption associations.

Not modelled: attention allocation, repeat consumption (only a descriptive table: 145 repeat rows), purchase or price
response, product retention or churn, community participation, and social diffusion. RecSim NG is unused. No
sequence or hierarchical models are fitted beyond gamma-Poisson. Each domain covers one platform and roughly one week
to one month.

### S5 Commercial intervention-response model (Criteo): RETAIN (as an offline model)
`commercial.py`, `fit_commercial.py`, report `reports/commercial_response@1.json`.
Identification is handled correctly:
* The randomised assignment is the treatment.
* Exposure is never used as a feature or as a causal effect, except in a labelled Wald complier estimate.
* The composition bias of the pooled tests is diagnosed: raw ITT 0.0103 vs propensity-stratified 0.0077 and IPW
  0.0075.

I reproduced every headline number. The caveat is in S6: the simulator interface is a 64-type marginal table
(`sample_types`, `respond`, `transport`). A simulated individual's ad response would be independent of ad content,
price, relevance and frequency, so an agent could learn only "whether to target", and nothing calls the interface
anyway.

### S6 Integration of fitted models into world participants: REDESIGN (L)
* `population/models.py:100-162`: the decisions name `browse_session@1`, `content_consumption@1`,
  `continue_watching@1`, `commercial_response@1`, `purchase@1` and `churn@1`. A model is used only if *every* fitted
  feature name is in `FEATURES`, which no registered model satisfies. Failures are swallowed (`except Exception:
  continue`, line 155). This is structural, not a missing wire: it needs a per-model feature adapter that turns
  world state into the card's `feature_map` quantities, quantile maps that are validated, and a hard failure when a
  configured model cannot be used.
* `social/consumption.py:140-226`, `social/recsys/service.py:50-52,120`: `news_click@1` is used only when
  `feed.calibrated` names a platform, and nothing in `src/` sets it (only `tests/test_v5_social_calibrated.py`).
  `qmap` (line 179) maps `text_cos`, `pop_ctr_24h` and `subcat_share` by rank within the current cohort batch. In p7,
  uniformly more relevant content changes p_open by exactly 0, and one viewer's probabilities depend on others' slates.
  An agent can raise opens only relative to the batch, never absolutely: a simulator-specific shortcut, not transferable
  behaviour.
* `domains/learning.py:457-474`: `SlateClicks` multiplies the calibrated probability by a hand factor of 0.05 for
  "irrelevant" results and by trust. The module is not imported by `domains/modules.py:2` (only `seats`, `science`), so
  it never runs in an integrated world. It would also apply a news-click model to lesson search results, which the
  card does not cover.
* Never consumed by any world code: `commercial_response@1` (ads and commerce), `watch_duration@1` (watch time stays
  an exponential hazard, `consumption.py` BUILTIN), `return_daily@1` and `session_rate@1` (return gaps stay
  exponential), `content_consumption@1` (although `shorts` short-video platforms exist), `content_consumption_feed@1`
  (the recommender target) and both software models (the E2 report says "No simulator module uses the models yet").

### S7 Fidelity register and SP-5.4 machinery: REPAIR (M)
`docs/v5/FIDELITY_REGISTER.md`, `src/crucible/calibration/fidelity.py`.
1. SP-5.4 fails: 40 important modules have no record.
2. `v5_rows` stops at the first non-table line (`fidelity.py:43-61`). The blank line 35 of the register hides rows
   36-52 (three duplicate E2 rows and all 14 C2 social rows).
3. False claims:
   * Rows 23-27 and 29 say pop (B) or social (C) use content_consumption, watch_duration, return_daily, session_rate
     and content_consumption_feed. They do not (p3, p8). Row 45 even says session_rate and return_daily are "not wired
     in yet".
   * Rows 30-33 say "population ad response and commerce via `calibration.load("commercial_response@1")`" and
     "software ecosystem via ...". Nothing loads them.
   * Row 28 says article choice in pop (B) and social (C) uses news_click. Only an opt-in social config does.
4. Because records are matched to modules at module granularity, one false row ("pop ... empirically fitted") covers
   all nine population decisions. Those decisions actually run on hand-set coefficients that have no row of their own.
5. Nothing checks that a claimed `calibration.load` is reached at runtime.
6. `ARCHITECTURE.md:105` ("the world loads only the fitted checkpoints") also overstates integration.

### S8 Gate F / SP-5.5 audit and reproduction machinery: REPAIR (S)
`calibration/audit.py`, `reproduce.py`, `tests/v5/test_gate_F.py`, `test_gate_SP_5_5.py`.
The machinery measures the registry correctly and catches tampering, and its reproductions match mine. Weaknesses:
* F.3's reproduction test is `@pytest.mark.slow` and is not part of default runs.
* `sp55_metrics` treats an amendment written after test exposure as prespecification.
* No Gate F test checks that any participant uses a fitted model, or that its use stays inside the card's limits.

## Requirement verdicts

| id | verdict | reason / work needed | criteria weak |
|---|---|---|---|
| F.1 | RETAIN | 5 canonical domains qualify (content_selection, watch_duration, browsing_frequency, advertising_response, software_adoption); gate test passes. Work: official run from a clean tree plus review. Add a criterion that each counted domain drives an in-world decision | yes |
| F.2 | RETAIN | content_consumption is fitted on randomised exposure and beats its baseline (reproduced); commercial_response is a randomised ITT model (reproduced on the full test split). Work: official evidence. The criterion should also require in-world use of both | yes |
| F.3 | RETAIN | 9 held-out reports; I recomputed 2 independently and 6 via the reproducers, with zero differences. Work: run the `@slow` F.3 through slots officially | no |
| F.4 | EXTEND | 9 comparisons against constant or popularity baselines, all won with CIs. Part 5.3 also requires comparison against the existing rule-based simulator behaviour: score `population.models` / `consumption.BUILTIN` (feature-mapped) on the same held-out rows | yes |
| F.5 | RETAIN | ECE, Brier, PIT-KS, coverage and reliability tables on every model. Work: official run | no |
| F.6 | RETAIN | 9 checkpoints load with verified hashes; tamper test passes; 0 mismatches | no |
| F.7 | REPAIR | checksums verified against outside sources. Fix the MIND acquisition route: official gated source with recorded licence acceptance, or a documented authorisation. Hash GH Archive actor ids. Add a licence notice for NC-derived checkpoints | yes |
| F.8 | RETAIN | substantive extrapolation limits in every card and report. The criterion is only a 200-character minimum | yes |
| F.n1 | RETAIN | 0 synthetic-only domains; all four datasets are verified real | no |
| SP-5.4 | REPAIR | fails (40 modules missing). Fix the parser (multiple table blocks), add rows for the missing modules and per-decision rows for population, rewrite the false "via calibration.load" rows to their real status, and add a runtime check that each empirically fitted row's model is actually loaded and used | yes |
| SP-5.5 | REPAIR | passes, but the E2 kept candidates (`t_gbdt_reg`, `gbdt_poisson_reg`) were introduced after run-1 test metrics were seen. Record this as a protocol deviation the checker flags (amendment `written_after` a test computation), and have the reviewer decide, or confirm on an untouched holdout | yes |

## Missing requirements (spec items with no manifest entry)

1. Fitted models must actually drive participant decisions in integrated worlds: population, social, commerce,
   software (Part 5 objective, Part 4.4).
2. Fitted models must be compared against the existing rule-based simulator behaviour (Part 5.3).
3. Model selection must consider long-horizon stability and computational efficiency (Part 5.3).
4. In-world validation: simulate the integrated world with the calibrated mechanisms, then compare user-level activity
   distributions, retention curves, heterogeneity and long-horizon aggregate drift with held-out data (Part 5.5).
5. Randomised-data validation of in-world responses: run in-world random-exposure and ad-holdout experiments and
   compare them with KuaiRand random-exposure and Criteo ITT estimates (Part 5.2).
6. In-world use must stay inside each card's limits: no news-click model on lesson search results, no
   batch-relative quantile maps that erase absolute effects (Part 5.2, "do not assume historical engagement
   determines responses to unseen content").
7. Coverage of the prioritised behaviours that are still missing: attention allocation, repeat consumption, purchase,
   retention, community participation, social diffusion (Part 5.3).
8. Privacy: calibration data must not retain re-identifiable ids (Part 5.1).
9. Licence compliance and authorised acquisition for datasets and derived checkpoints (Part 5.1).
10. A fidelity-register truthfulness check: claimed runtime use is verified (Part 5.4).

## Transfer question

An agent learning in today's worlds learns hand-set behaviour: logits with made-up coefficients, exponential hazards
and fixed share rates. The empirically grounded models are present but disconnected. Where `news_click@1` is switched
on, the batch-rank quantile mapping teaches "be more relevant than the other items in this batch" rather than "be
relevant". Uniform improvements in content quality earn nothing, which is a simulator-specific shortcut. The Criteo
type table, if wired as designed, would make ad response independent of ad content and price. So Part 5 calibration
does not yet make learned capabilities more transferable to the real internet. The offline assets (data, fits,
reports) are trustworthy and reusable once integration is redesigned.

## Skeptic review

Reviewer: independent skeptic pass on the audit above. The worktree HEAD is `dd745b9`; `git diff a881d18 dd745b9` touches
nothing in `src/crucible/calibration`, `population`, `social`, `docs/v5/calibration` or `tests/v5/test_gate_F.py`, so
the audited code is unchanged. Probe scripts and logs:
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/E_calibration_skeptic/`. No repository file
other than this report and its JSON was modified; every tampering experiment ran on a scratch copy of the models
directory.

I tried to refute each RETAIN and each verdict that might be too generous. I also checked whether the harsh verdicts
hold up. **Result: 7 verdicts change (3 subsystems, 4 requirements); the rest are upheld, some with stronger evidence
or weaker criteria flagged.**

### What I ran

| # | Probe | Result |
|---|---|---|
| t1 | `pytest tests/v5/test_gate_F.py tests/v5/test_gate_SP_5_4.py tests/v5/test_gate_SP_5_5.py tests/test_v5_social_calibrated.py -m "not slow"` | 14 passed, 1 failed (SP-5.4, `important_mechanisms_without_record=40`), 1 deselected (F.3 is slow). Reproduces the auditor's result |
| r7, r8 | re-ran the auditor's `p7_calibrated_open.py` and `p8_world_usage.py` | identical output: p_open does not change under a uniform affinity shift or scale (max \|dp\| = 0.0); the population world discards all 6 `calibration.load` results; the default social world makes 0 load calls |
| s1 | **card tampering** on a scratch copy: swap two names in news_click's `features`, halve `feature_quantiles.text_cos`, set `prior_rate` from 0.04 to 0.4; point commercial_response's `selection.kept` at `t_logistic` | `registry.load` succeeds; `audit.examine` gives **hash_mismatches 0, qualifies True**. news_click mean p 0.039 -> 0.082 (max \|dp\| 0.64). On the 256 real Criteo rows stored in the checkpoint, the commercial uplift changes (corr 0.48 with the original). `gate_f_metrics` is unchanged (9 qualifying, 5 domains, 0 mismatches). Only SP-5.5's `selection_check` notices the `kept` swap. Nothing notices the quantile or prior change, which are exactly the parameters world code (`CalibratedOpen`, `SlateClicks`) reads |
| s2 | **fabricated or degraded evidence** on a scratch copy: every E1 report rewritten so the kept model *loses* to its baseline (x1.5), calibration set to `{"ece": NaN}`, limits replaced by 200 `x` characters; plus a fake `purchase_fake` model (session_rate's checkpoint relabelled `purchase_behavior`) | **F.1, F.3, F.4, F.5, F.6, F.8 and F.n1 criteria all still met.** Domains go from 5 to **6** (the fake counts as a new fitted domain); 7 losing comparisons count as baseline comparisons; 7 NaN calibrations count as calibration reports; filler counts as limits. SP-5.5 `worse_model_kept` stays 0 because it reads only cards, never reports. The fake fails only SP-5.5's plan check and, in a slow run, F.3's reproducer-set equality |
| s3 | **commercial type table** as wired by its card (`sample_types(rng, n)`, `respond(rng, types, treated)`) | `respond` takes only the type and a binary `treated`; `sample_types` takes only `rng, n`. The prespecified default ("model") table gives **18.4% of individuals a negative ad effect, down to -2.6 pp**. The "honest" validation-rate table has a minimum of -0.14 pp, so those negative responders are an artifact. Top-type effect 0.164 vs 0.141 honest; type-level calibration slope **0.84** on test (heterogeneity overstated by ~19%), yet the over-dispersed table is the default. `transport` re-bases the same table so the absolute ATE runs from **0.0025 to 0.039 (15x)** as the world's base rate goes from 0.01 to 0.6: an untested assumption sets advertising return on investment |
| s4 | **population plug-in contract**: add the fitted feature names to `FEATURES` in memory, so the name check passes | `purchase` ("buy the product evaluated") is now labelled "empirically fitted" with commercial_response. `prob` for 5 individuals returns **20 values** (4 per-arm columns flattened), and `sample` returns 20 decisions, silently misaligned with the individuals. `continue` ("read another page of the same site") would use KuaiRand's "kept watching a feed video > 7 s". The contract is unsound beyond the missing adapters |
| s5 | **what CalibratedOpen reads** (every column and membership lookup spied during `p_open`) | reads only `aud_items.{emb, imp, platform, published_at, views}`, `pop.aud_i_t` and `pop.feed_topic_views`. **Not read**: follow, share arrival, already seen, creator familiarity, satisfaction, duration. Turning calibration on removes the built-in odds multipliers for follow (x2.46), arrival by share (x3.32), already seen (x0.17) and familiar creator (x1.82), and with them the per-item effect of social diffusion and repeat avoidance on that platform |
| s6 | report vs card consistency for all 9 models | consistent today (kept and baseline test metrics equal). No checker enforces this (see s2) |
| s7 | **own recomputation of calibration metrics** (the auditor recomputed none) from the checkpoints on the prespecified KuaiRand test splits | content_consumption ECE 0.0063239, Brier 0.1396875; content_consumption_feed ECE 0.0056523; watch_duration coverage_80 0.796181 and coverage_90 0.895550; return_daily ECE 0.0261320. **All equal the reports.** Equal-mass ECE is 0.0148 for content_consumption_feed, 2.6x the reported equal-width figure. return_daily's kept model has ECE 0.0261 against the constant baseline's 0.0210 |
| s8 | **are card limits enforced?** Configure `news_click@1` on the `posts` (social posts) platform, which card limit (1) forbids | accepted and runs; no check anywhere. `ProtocolError` validates only the model name (`recsys/service.py:120-123`) |

### Verdict changes

| target | auditor | skeptic | why |
|---|---|---|---|
| S3 Model registry and checkpoints | RETAIN | **REPAIR (S)** | Integrity covers only the `.npz` (`registry.py:441-449`, `card_files`). Parameters that determine behaviour live in the unhashed `model_card.json`: `features` order, `feature_quantiles`, `prior_rate`, `selection.kept`, `response_types.usable`. s1 alters predictions and in-world inputs with 0 mismatches and `qualifies=True`. The F.6 tamper test flips one checkpoint byte only (`test_gate_F.py:170`). Fix: hash the card's behavioural fields, or move them into the checkpoint, and pin card plus checkpoint hashes in an index outside the model directory |
| S5 Commercial intervention-response model | RETAIN | **REDESIGN (L)** | The offline Criteo estimator is upheld (the auditor reproduced it on the full test split). But this subsystem *is* also the simulator-facing interface (`commercial.py:105-147`, card `simulator_mapping`), and that interface cannot express what an advertiser agent acts on. Response depends only on a latent type drawn independently of every individual attribute, plus a binary ITT flag: no creative, relevance, price, frequency or saturation. The default table invents an 18% negative-responder mass and overstates heterogeneity (slope 0.84). `transport` scales the absolute effect 15x with the base rate, an assumption nobody has validated. An agent can learn only "target or not". The absolute effect is not empirical either. The publisher sub-sampled the data non-uniformly "so that the original incrementality level cannot be deduced", and the card itself says absolute effect sizes are not representative. The world effect is therefore odds ratios from a deliberately distorted sample applied to a hand-set world base rate. Recovery preamble: "a fitted behavioral model does not establish realistic intervention responses" |
| S8 Gate F / SP-5.5 machinery | REPAIR (S) | **REPAIR (M)** (claim overturned) | The audit said the machinery "measures the registry correctly". It measures presence, not validity. `baseline_compared` checks that keys exist, not who wins (`audit.py:182`). Any finite or NaN calibration key counts (`audit.py:184`). Limits only need 200 characters (`audit.py:186`). Domains count from the card's self-declared label. s2 passes every fast F criterion with degraded reports and a fabricated sixth domain. The only check against outside data, the slow F.3, recomputes just the kept model's primary metric. It never checks baseline metrics, calibration metrics or report content |
| F.1 | RETAIN | **REPAIR** | Gate F is "empirical behavioural grounding" of the population (Part 5 title, objective "replace arbitrary behavioral assumptions"). The task defines RETAIN as integrated. No fitted domain drives any decision in an integrated world (r8: all 9 population decisions plausible; default social world 0 loads). The fast metric also counts a relabelled model as a new domain (s2). Work: wire at least 3 domains into default Tier 2 and Tier 3 worlds with adapters, and make the criterion count domains whose model drives a recorded in-world decision |
| F.2 | RETAIN | **REDESIGN** | Commercial response: see S5. Nothing calls it, and the interface needs redesign. Content consumption: the randomised-exposure model (`content_consumption@1`, the one whose identification supports causal use) is never used. The only exposure-response model reachable in the world is `news_click@1`. Its card says it "may NOT infer ... (3) the causal effect of showing an item (exposure was chosen by a recommender)". Yet the world uses it exactly as the response to recommender and agent exposure choices (Part 5.2: "do not assume historical engagement determines responses to ... interventions") |
| F.6 | RETAIN | **REPAIR** | s1: `checkpoint_hash_mismatches == 0` holds while the parameters that determine behaviour are altered. The criterion is self-referential: the expected hash sits in the card next to the file. criteria_weak = true |
| F.8 | RETAIN | **REPAIR** | The limits are written down but not enforced, and world code breaks them. CalibratedOpen applies news_click to exposure-response use (limit 3). `SlateClicks` applies it to lesson search in an education domain (limit 2: other populations). s8: any platform, including social posts, is accepted (limit 1). s2: 200 characters of filler meet the criterion. Work: machine-readable allowed-use and population scopes in each card, checked at `load` or by the consumer, plus a test that a forbidden use raises. criteria_weak = true |

### Upheld (with notes)

* **S1 datasets: REPAIR.** The MIND ungated mirror is confirmed (`datasets.py:144-190` records that the official route
  needs licence acceptance). The actor-id privacy point is lower severity than stated: the extracted tables live in
  `/workspace/data/crucible/gharchive_2024-10-07_96h`, outside the repository, and are never committed. GH Archive
  itself publishes those numeric ids.
* **S2 offline pipelines: EXTEND.** I checked for leakage in the MIND popularity features (`mind.py:111-123`, cumulative
  sums over hours `< h`) and the KuaiRand `PriorCounts` (strictly earlier days, safe for negative windows): none
  found. One minor issue: MIND's smoothing prior and category shares use all data, test included. Added defect:
  selection uses only the primary accuracy metric, while Part 5.3 also asks for calibration, long-horizon stability
  and cost. The kept models are worse calibrated than constant baselines for news_click (ECE 0.0031 vs 0.00068) and
  return_daily (0.0261 vs 0.0210, s7). Equal-width ECE puts 86% of news_click's test rows in one bin, so it barely
  measures calibration at a 4% base rate.
* **S4 coverage: EXTEND.** Upheld.
* **S6 integration: REDESIGN.** Upheld and strengthened. s4 shows the contract is wrong even when names match:
  outputs end up misaligned and semantics are mismatched. s5 shows that calibration removes social mechanics. s8 shows
  limits are not enforced. For 8 of the 9 models the integration is effectively IMPLEMENT. The existing tests would
  pass on this defective implementation: `test_v5_social_calibrated.py:49` asserts only `abs(r_cal - r_base) > 0.01`,
  and line 68 asserts a *within-batch* rank correlation > 0.2, which batch-relative quantile mapping satisfies by
  construction.
* **S7 fidelity register: REPAIR.** Upheld. Spot-checked rows 23, 24, 27, 31 and 32 (claimed `calibration.load` use)
  and `ARCHITECTURE.md:105`. `record_problems` accepts an "empirically fitted" row for any module whose evidence
  string names a dataset and a year. It never checks that the module loads a model.
* **F.3: RETAIN**, but criteria_weak becomes **true**. The numbers reproduce (the auditor's evidence plus s6
  consistency), but the manifest criterion `heldout_reports >= 3` counts fabricated reports (s2). The reproduction
  lives only in the slow-marked test and covers only the kept model's primary metric.
* **F.4: EXTEND.** Upheld. criteria_weak is confirmed by s2: 7 comparisons the kept model loses still count, and
  `test_F_4` computes `learned_models_beating_baseline` but never asserts on it.
* **F.5: RETAIN** (now with my own recomputation, s7, for 4 models), but criteria_weak becomes **true**: NaN counts
  (s2), and equal-width ECE is uninformative at low base rates. No calibration is measured for the in-world mechanism.
* **F.7: REPAIR.** Upheld (licence route); privacy severity lowered as above.
* **F.n1: RETAIN.** Upheld, because the auditor verified the datasets against publishers' checksums. criteria_weak
  becomes **true**: the "not synthetic" status is a self-declared manifest flag, linked to a checkpoint only by the
  card's claim (s2).
* **SP-5.4: REPAIR**, upheld (t1 reproduces 40). **SP-5.5: REPAIR**, upheld. Also: `selection_check` reads only the
  card. It would not notice a report that contradicts it, and "better" means the primary metric only.

### Additional findings

1. **The world uses the observational model and ignores the randomised one.** The randomised-exposure KuaiRand model
   is unused. The MIND model, which its card forbids for causal use, is the only exposure response that can be wired
   in.
2. **Calibration would delete social dynamics.** On a calibrated platform, opening ignores follows, shares, already
   seen items and familiarity (s5). Part 5 grounding therefore currently competes with Part 4.6 and 4.7 organic reach
   and diffusion rather than grounding them.
3. **Unprotected world-facing parameters.** The parameters world code reads (`feature_quantiles`, `prior_rate`) are
   outside every integrity check (s1).
4. **The fast Gate F suite is a presence check.** It passes on degraded or fabricated evidence (s2). Treat any
   unofficial Gate F "pass" as weak evidence until the slow reproduction runs officially and is extended to baselines,
   calibration and report content.
5. **Default ad-response table.** The prespecified default is the over-dispersed "model" table (slope 0.84, 18%
   spurious negative responders), not the honest table (slope 1.006). Because Criteo deliberately distorted the
   incrementality level, only relative structure is empirical. Any in-world advertising return on investment is set
   by `transport` and a hand-chosen base rate, and the fidelity register (rows 31, 32) should say so.

### Skeptic's transfer note

The auditor's conclusion stands and becomes sharper. In the one place calibration can be switched on, agents would
learn to beat the other items in a batch, not to be relevant. They would face a world without follow or share effects
on opening. Under the advertising interface as designed, the return on targeting would come from an untested
base-rate transport and an over-dispersed type table. None of this transfers to the open internet. The offline
assets remain trustworthy: I reproduced the calibration metrics in s7 and found no feature leakage. They should be
retained as data and fits, not as an integration design.
