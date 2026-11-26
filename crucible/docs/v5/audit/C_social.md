# Audit C_social: the social internet (MASTER_SPEC Part 4; Gates C, D, I; SP-4.x)

Auditor: independent audit, worktree `feat/crucible-5-complete` at `a881d18`, package `crucible/src/crucible/social`
(about 10,600 lines: C1 platforms/catalogue/studio/communities, C2 audience/recommenders/consumption/scale).
Nothing previously claimed was taken as evidence; every verdict below rests on code reading plus runs I made.
Probe scripts and logs: `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/C_social/`
(`probes/p1..p6*.py`, `*.log`, unofficial gate evidence under `tmp/crucible-v5-unofficial-evidence/`).

## 1. Verdict in one paragraph

The package contains two well-engineered halves that do not form one social internet. **C1** (catalogue, six
content sites, studio, follow graph, messaging, creator memberships, newsletters, launches, Q&A, communities,
browser pages) works through tools for hand-scripted principals. **C2** (exact audience accounting, twelve
recommenders, consumption engine, 10M-identity scale path) works for its own stand-in population `crowd`. They
share only one thin bridge (a public text post on a platform named in both). The autonomous population never uses
communities, messaging, newsletters, launches, Q&A, creator memberships, search, or the C1 follow/share state; B's
real population module (`pop`) cannot drive the C2 engine at all (feed install raises); and there are three
incompatible populations with three topic vocabularies and two interest spaces. The accounting core (views,
unique viewers, exposures, recount) is genuinely exact and scales to 10M identities, which I reproduced. But the
thing the audience rewards is not content: in a controlled probe a post made of **meaningless tokens chosen for
their hash vectors reached 3,213 of 4,000 people while an informative natural English article reached 330**;
images/audio/video either cannot be distributed (default encoder) or are distributed through a random projection
that maps a football text to "politics" and a cooking text to "cars". An agent learning here would learn to game a
32-dimensional lexical hash, not to make content people find useful.

## 2. What I ran

| Command / probe | Result |
|---|---|
| `pytest tests/test_v5_social_{audience,idset,recsys,calibrated}.py` (1 thread) | 18 passed (27 s) |
| `pytest tests/v5/test_gate_C.py -k "not C_6"` | 5 passed, **1 failed: C.3** (`ProtocolError: feed: the population module does not provide register_activity`) |
| `slots --cpu 2 -- pytest tests/v5/test_gate_C.py -k C_6` (Playwright Chromium) | 1 passed (9 browser steps, 5 state changes, headless twin identical) |
| `slots --cpu 1 -- pytest tests/v5/test_gate_D.py` | 5 passed (136 s). D.1: 12 mechanisms end to end, max pairwise exposure Jaccard 0.38; D.2 p=0; D.3 1 change, p=0; D.4 46 probes, 0 leaks; D.5 0 violations |
| `pytest tests/v5/test_gate_SP_4.py SP_4_2 SP_4_7 SP_4_8` | 4 passed (SP-4: 16/16 platform kinds; SP-4.2: 10 kinds, 24 tools; SP-4.7: 13 features; SP-4.8: 2 chains, 0 bonuses) |
| `pytest tests/v5/test_gate_SP_4_5.py SP_4_6` | 2 passed (10 mechanisms, 23 fitted parameter sets, analytics age 6 h, noise log-sd 0.053; 10 quantities) |
| `slots --cpu 1 -- pytest tests/v5/test_gate_I.py` at **10,000,000 identities** (1 thread) | 9 passed in 658 s; positive control I.1-I.7 all criteria met (4.11); organic run at 2.5M: top item 203,583 unique viewers (I.n1 met) |
| p1 `p1_integration.py`: C1 modules + crowd + aud + feed in one world | works for text posts; **image publish refused** with the default encoder; works with `encoder="semantic"` |
| p2 `p2_update_and_counters.py`: update staleness, counter-write attempts, forced views | **content.update leaves the distributed representation stale**; all tool-level writes refused; **module code can inject 500 views that the recount accepts as clean** |
| p3 `p3_shortcuts.py`: natural article vs keyword salads vs meaningless tokens, 3 seeds, forked worlds | natural 330 mean unique viewers; football keyword stuffing 1,540; meaningless hash-selected tokens **3,213 / 4,000** |
| p4 `p4_semantic_bridge.py`: topical alignment under each encoder | default: text aligns with its topic (cos 0.59-0.81); semantic mode: **text and images land on unrelated topics** (cos -0.37..0.19) |
| p5 `p5_cross_state.py`: cross-subsystem state | tool reading (5 x content.open) = 0 impressions/views; population 78 follows vs C1 follower count 0; 9 population shares vs 0 C1 share rows; 28 of 29 distributed items absent from the catalogue |
| p6 `p6_cascade_anatomy.py` at 500k | positive control: item 0 owns hour 1 because only 13 items exist; by hour 6 item 2 overtakes; top-item exposure sources: trending 57 %, share 33 %, interest 10 %. Organic: top item 23,598 viewers (4.7 %) in 6 h, 91 % via interest retrieval |

The consolidated regression (`docs/v5/evidence/integration/regression_consolidated_g*.xml`) contains none of the
`tests/v5` gate tests; no social gate had ever run on the integration branch before this audit.

## 3. Architecture as found (who calls what)

* `social/modules.py` imports every C1 and C2 module; `social_internet_modules()` lists the C1 world. Nothing
  outside `crucible/social` references `social_internet_modules`, `crowd`, `aud` or `feed` (grep of `src/crucible`):
  **no Tier 2 domain, no civilization (`civ/` is a 5-line stub), no worldgen family uses the social internet.**
* C1: `content.publish` (`content.py:255`) is the single entry of the catalogue; `_distribute` hands public items to
  `aud` through `integration.index_published` (`integration.py:31`). `integration.record_impressions` /
  `record_consumption` (`integration.py:60,84`) are **never called**; `content.open` (`content.py:621`) is not a
  consumption event.
* C2: `feed.install` (`recsys/service.py:175-190`) registers the browse activity with `crowd`, or with B's `pop`
  through `crucible.population.register_activity`, which does not exist (it is `population.core.register_activity`),
  so the `pop` branch raises. `consumption.Engine.deliver` imports `from ..population import wake`
  (`consumption.py:487`), which does not exist either.
* Population creators (`scale.publish_post`, `scale.py:348`) write directly to the artifact store and `aud`,
  bypassing the catalogue, search and the sites.
* Population batch APIs for C1 state - `graph.follow_batch/share_batch` (`graph.py:101,112`),
  `communities.join_batch/leave_batch/touch_batch` (`communities.py:148-158`) - are **never called**.
* Three populations: B's `pop` (16 topics, sign-hash text vectors, `population/text.py`), C2's `crowd` (24 topics,
  Gaussian random indexing, `social/textrep.py`, `social/vocab.py`), C1's scripted `ReferenceParticipant`s
  (8 topics, word sets, `social/reference.py`). Their interest representations are not comparable.

## 4. Subsystems

### 4.1 Platform registry, catalogue and content sites - REPAIR (M)
Paths: `platforms.py`, `content.py`, `sites.py`, `nodes.py`, `edges.py`, `formats.py`.
Evidence: SP-4 reproduced (16 platform kinds exercised by four scripted principals, one catalogue, one identity
space); C.1 reproduced (text, HTML, PNG, WAV, H.264 MP4 published and independently decoded by another principal).
Defects: (1) `content.update` (`content.py:351-375`) publishes a new artifact but never re-indexes the `aud` item:
probe p2 shows the distribution representation and terms stay those of the original (football) while the readable
bytes are now cooking (cos 0.14) - a bait-and-switch path and a violation of "recompute embeddings when content
changes"; (2) `content.open` is not recorded as consumption, so tool-using readers produce no analytics (p5);
(3) population items never enter the catalogue (p5: 28 of 29), so they are invisible to `search`, `<site>.latest`,
`trending`, comments; (4) `aud` defaults to platforms `["posts","articles"]` (`audience.py:110`) and "articles" is not
a C1 platform. Work: one publish path for everyone; re-index on update; record content reads through `aud`.

### 4.2 Creator studio and artifact formats - RETAIN (S)
Paths: `studio.py`, `render.py`, `formats.py`. Evidence reproduced: SP-4.2 (10 formats x 7 life-cycle steps, 24
tools; apps run and packages tested in the 4.0 sandbox), C.1 (ffprobe/Pillow/wave decode). Rendering is recorded so
replay never re-runs FFmpeg/SoX. Limits are honest and spec-compatible ("lightweight tools"): images are text cards
or vector scenes, audio is synthesised tones, video is a slideshow.

### 4.3 Social graph, messaging, creator memberships, newsletters, launches, Q&A - REPAIR (M)
Paths: `graph.py`, `messaging.py`, `creators.py`, `launch.py`, `qa.py`. Evidence: SP-4 and C.n1 reproduced; counters
recount exactly from events. Defects: this state is disjoint from C2's: population follows live in `aud` follower
sets (`audience.py` `fo` idsets), population shares in feed inboxes; p5 shows 78 population follows vs C1 follower
count 0 and 9 population shares vs 0 C1 share rows, so `content.inspect` reports `followers: 0` next to
`subscribers: 78`, and `<site>.trending` ignores population shares. Batch APIs exist but nothing calls them.
Work: one follow graph and one share log written by both population and agents.

### 4.4 Communities - EXTEND (L)
Path: `communities.py` (1,367 lines). Rich and correct as a tool surface (membership spells, governance by
owner/moderators/vote, moderation log, trust levels, events, merges, spin-offs, health = sustained participation);
SP-4.7 reproduced (13 features) and C.4 reproduced (16 formed, 14 active after 30 days). But every community was
founded, joined, posted in and left by scripted `ReferenceParticipant` code (`reference.py:178-205`, founders post
with probability 0.5 per day); the autonomous population has no path into communities (`join_batch` etc. unused).
"Communities form and persist" is therefore a scripted demo. Work: population-level membership/participation
decisions (join, lurk, post, churn) driven by experienced value, through the batch APIs.

### 4.5 Browser workflow - RETAIN (S)
Path: `web.py`. C.6 reproduced with Playwright Chromium confined to `*.crucible`: login, found a community, write
in the studio, post in the community, comment, follow, timeline, search; 5 state changes verified in world state;
the same commands headlessly give the identical event-hash chain. It operates on C1 state only.

### 4.6 Audience accounting - RETAIN (S, with one evaluator-side repair)
Paths: `audience.py`, `idset.py`. Code: impressions and consumption are recorded as `cols.batch` blobs; appliers
derive ten quantities; a view requires `watch_s >= min(30, 0.5*duration)` and a prior exposure
(`audience.py:291-336`); `recount` re-derives everything independently. Reproduced: SP-4.6 (10 quantities), D.5
(0 violations over 384 checks), and at 10M identities 23.6M exposures and 10.57M distinct viewers recounted with 0
mismatches. Tool-level counter writes all refused (p2: `cols.batch`, `aud.*`, extra `views` argument, set_mixture by
an agent, watch without exposure, non-population watcher). Weakness: the recount cannot tell a participant's decision
from scripted injection - p2 injects 500 impression+consumption rows from module code and the recount reports
0 mismatches and 0 "forced views" (`audience.py:820-832` only checks that an impression precedes the consumption).
Work: record the deciding process/actor in each batch and count only batches from the consumption engine or
participant tools as organic.

### 4.7 Content representation - REDESIGN (L)
Paths: `textrep.py`, `audience.py:465-484` (`Audience.encode`). The default encoder is a 32-d random projection of a
bag of words; population interests are mixtures of the centroids of a closed 576-word, 24-topic vocabulary
(`vocab.py`). Measured: cosine between documents sharing no word has sd 0.179 (p95 |cos| 0.35); "soccer manager" vs
"football coach" cos -0.22; an informative natural article on football pressing has cos -0.05 to the football
centroid. Probe p3 (3 seeds, identical forked worlds): natural article 330 mean unique viewers; football keyword
stuffing 1,540; **meaningless tokens (`zq...x`) chosen only because their hash vectors align with the mean
interest: 3,213 of 4,000.** Non-text kinds raise `InvalidAction` with the default encoder (p1: image publish
refused on a distributed platform). With `encoder="semantic"`, workstream D's vector is multiplied by a seeded
random Gaussian matrix (`audience.py:475`) into a space that has nothing to do with the interest space: p4 maps a
football text to "politics", cooking to "cars", programming images to "art". Work: one shared representation space
for content and individuals (learned or pretrained, multimodal), interests expressed in it, validation that
participants respond to meaning rather than tokens, and an adversarial "token-optimised content" test.

### 4.8 Recommendation mechanisms - EXTEND (M)
Paths: `recsys/__init__.py` (Mixture/RRF), `recsys/algorithms.py`, `recsys/training.py`, `recsys/features.py`,
`recsys/service.py`. Twelve genuinely different algorithms (following feed, cosine retrieval, implicit ALS, IRLS
logistic engagement on delayed label-noisy outcomes, batched Thompson sampling on delayed noisy analytics, social
inbox/likes, MMR, reputation from earned rates, BM25 search, freshness x velocity, trending, random), finite k slots
with reciprocal-rank fusion. Reproduced D.1-D.5 and SP-4.5. Limits: algorithm change is only a configured day
schedule or an operator override (`service.py:118-128`); platforms never adapt their own distribution to their own
outcomes; candidate pools are a few hundred items; every content-based mechanism inherits 4.7's representation.

### 4.9 Consumption and behaviour engine - EXTEND (L)
Path: `consumption.py`. Per slot: multinomial-logit open among competing items, bounce + exponential hazard watch
time, like/share/follow logits, continuation, satisfaction, habit and taste drift - all seeded per-individual draws,
attention-capped (D.5, I.4 reproduced). Value of content to a person is only `affinity` (4.7) plus social signals;
there is no notion of usefulness, correctness or need satisfaction. Parameters are hand-set ("plausible,
unvalidated", FIDELITY_REGISTER rows 41-46); the fitted `watch_duration@1`, `return_daily@1`,
`content_consumption_feed@1` are not wired; only `news_click@1` is, when a platform is configured `calibrated`.
Only text kinds are consumed. Work: calibrated defaults, multimodal kinds, value from what the artifact does for the
individual's current need.

### 4.10 Population integration - REDESIGN (L)
Paths: `scale.py` (`crowd`), `recsys/service.py:175-190`, `consumption.py:483-490`. The C2 engine runs only on
`crowd`, a stand-in written "because workstream B's population module is not merged" (`scale.py:3-5`); with B's
`pop` the feed module cannot install (C.3 failure reproduced). Even with the two import fixes, `pop`'s interests are
in a different vector space (`population/text.py:100-151`, sign hashing, 16 topics) from item embeddings, so
affinities would be noise. Work: one population, one interest/content space, B's activity scheduler calling the
social activities on every platform (feeds, communities, messaging, newsletters, launches, Q&A, search).

### 4.11 Scale path and cascade experiments - EXTEND (M)
Paths: `scale.py`, `cascade.py`, `experiments.py`. Reproduced at 10,000,000 identities (1 thread): world created in
28 s, 3 simulated hours in 179 s; item 0 reached 318,659 / 627,712 / 1,148,449 unique viewers at hours 1/2/3;
10,568,231 qualifying distinct consumption events; exposure and viewer recounts exact; 0 attention violations;
share generations up to 9. The contact graph is hash-computed and never stored, principals are materialised
lazily, sets switch to bitsets. Caveats from p6: the positive control's reach is mostly a genesis effect - in hour 1
the catalogue holds about 13 items and the feed shows 8, so ~76 % of everyone active views item 0; at 500k a
competing item overtook item 0 by hour 3; reach was delivered 57 % by the `trending` slot. The organic run's top
item is feed-retrieval-driven (91 % interest, 7 % share). Single text platform, stand-in population, no estimated
mode (none needed because the exact mode works).

### 4.12 Gate evidence harness - REPAIR (M)
Paths: `reference.py`, `tests/v5/test_gate_C.py`, `test_gate_SP_4*.py`, `social_helpers.py`. C.2, C.4, C.5,
SP-4, SP-4.7, SP-4.8 are measured on scripted Python drivers or on the test body calling tools in a fixed order,
not on the autonomous population. The C.5 effects are structural consequences of the driver's candidate order
(`reference.py:132`: share, newsletter, following first). Test bugs: C.2 counts population channels with
`AUD.batches(w, kind="impress")` and `b.get("sources")` (`test_gate_C.py:203`) - the kind is `imp` and there is no
`sources` field, so population channels are always 0; C.3 would hit `int(dict)` at `test_gate_C.py:238` (recount's
`mismatches` is a dict) even after the feed wiring is fixed. Gate I tests assert almost nothing themselves
(criteria are left to the checker).

### 4.13 Social-economic interoperability - EXTEND (L)
Paths: `launch.py`, `creators.py`. `launch.visit` returns the real `market.view` / `prod.view` page; buying happens
in the market/commerce modules; no money path in `social/` except a fan-paid membership (static scan reproduced).
But the only chain is the SP-4.8 test body calling tools in order; the population never discovers products through
content, never visits, tries, subscribes or reviews; infrastructure cost, retention and demand effects are not
exercised. No hard-coded social-to-revenue bonus exists (verified).

## 5. Requirement notes

Verdicts (detail and work items in `C_social.json`; criteria_weak marked *):
C.1=RETAIN, C.2=REPAIR*, C.3=REPAIR, C.4=EXTEND*, C.5=REPAIR*, C.6=RETAIN, C.n1=REPAIR*, D.1=RETAIN, D.2=RETAIN*,
D.3=RETAIN*, D.4=RETAIN, D.5=RETAIN, I.1=REPAIR, I.2=RETAIN, I.3=RETAIN, I.4=RETAIN, I.5=REPAIR*, I.6=RETAIN,
I.7=RETAIN, I.n1=EXTEND*, SP-4=REDESIGN*, SP-4.2=REPAIR, SP-4.5=RETAIN, SP-4.6=RETAIN, SP-4.7=EXTEND*,
SP-4.8=EXTEND*. RETAIN on a requirement means its criterion was reproduced in this audit on the mechanism it names;
it does not override the subsystem verdicts (e.g. the I.* capacity results ran on the C2 stand-in population).

* **C.3** fails in the integrated configuration (B's `pop`); on `crowd` (p5) it would pass: 69,716 consumption rows,
  0 recount mismatches. Tool reading is invisible to analytics.
* **C.n1 / I.5**: tool-level writes are blocked, but "forced view = consumption without earlier impression" does
  not detect module-level injection (p2). The criteria are too weak to establish "views arise from participant
  decisions".
* **D.2 / D.3 / C.5** p-value criteria with 10^5 rows are met by any change at all (p = 0.0); effect sizes and
  direction are not required.
* **I.6**: met (1,148,449 >= 1,000,000) by a designed positive control; the dynamics are a first-mover effect in an
  empty catalogue.
* **I.n1**: reproduced at 2,500,000 identities, 24 simulated hours: the top item (item 3, one of the first
  posts) reached 203,583 unique viewers (8.1 %), 0 forced views, 10 share generations, 0 recount mismatches;
  trajectory 33,630 (h2) -> 112,201 (h6) -> 203,583 (h24), i.e. early-item advantage under the "half newest, half
  most viewed" candidate pool (`consumption.py:68`). Whole Gate I file: 9 passed in 658 s on one thread.

## 6. Would a learner acquire transferable capability here?

Mostly no, for content and audience work. Observed shortcuts: (1) token optimisation against a 32-d hash space beats
writing (p3); (2) publish first into an empty catalogue (p6); (3) publish an attractive representation, then
`content.update` the bytes to anything (p2); (4) images/audio/video are either undistributable or a random lottery
(p1, p4). Social-graph effects (follow, share inbox) and finite attention are real mechanisms, and the accounting is
trustworthy, so lessons about scarcity, distribution channels and noisy delayed analytics can transfer; lessons
about what content succeeds cannot. No reward fields leak through social observations (grep of the package; creator
analytics are delayed noisy observable consequences, D.4 reproduced), and the social modules have no reward mode at
all, so the observation-only amendment is satisfied trivially here but untested in any social environment wrapper.

## 7. Missing requirements (spec text with no workstream-C manifest entry)

See `missing_requirements` in `C_social.json`: one shared population across all platforms; content representation
that responds to meaning (anti-token-gaming); re-encoding on update; tool reading recorded as consumption; population
participation in communities with churn; endogenous platform algorithm evolution and adaptive competitors; changing
audiences; no virality multipliers (static and dynamic check); population-level social-to-economy chains with
infrastructure cost and retention; social internet inside Tier 2/3 worlds; observation-only default for any social
environment wrapper.

## Skeptic review

Reviewer: independent skeptic. I worked in the same worktree. HEAD is now `dd745b9`, one commit past `a881d18`; that
commit changes only `acceptance/v5/manifest.py` and `requirements.py`, and nothing under `social/` changed. My aim was
to refute every RETAIN and every generous verdict above, and to test whether any harsh verdict was unjustified. Every
challenge rests on a probe I wrote and ran myself, on one thread (`OMP/OPENBLAS_NUM_THREADS=1`). Probes and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/C_social_skeptic/` (`sN_*.py`, `sN_*.log`).

### Verdict changes

| Target | Auditor | Skeptic | Probe | Why (short) |
|---|---|---|---|---|
| Subsystem: Audience accounting | RETAIN | **REPAIR** (S-M) | S2 | The verifier only works on the in-memory log window. After restoring a snapshot, `recount` reports 439 false mismatches, and with a short log window it reports 643. It returns 0 only when the full history is passed from a durable store, and no caller does that. There is no decision provenance (p2), and C1 reading is not accounted. |
| Subsystem: Browser workflow | RETAIN | **REPAIR** (S) | S6 | The social pages exist only when a test builds `make_social_web_app`. The deployable browser entry point (`python -m crucible.web.app`, which calls `make_web_app`) serves generic service-host consoles for `social.crucible` and the other social hosts. No runner or agent uses the browser mode for social work. |
| Subsystem: Consumption/behaviour engine | EXTEND | **REDESIGN** (L) | S7, S4 | Identical embeddings, 15 words vs 600 words: the blurb wins unique viewers (1,922 vs 1,208; 1,251 vs 532), shares and subscribers. The engine's value is `frac_read x (affinity - 0.2)`, which ignores what the reader gains. The session mechanics can stay; the value model must be rebuilt. |
| Requirement C.1 | RETAIN | **REPAIR** | S9 | C.1 was measured in a C1-only world. When pics, pods and shorts are distributed by `aud` with the default encoder, publishing images, audio and video is **refused** (`InvalidAction: no encoder for image/png`). It works only with `encoder="semantic"`, which is the random projection found in p4. |
| Requirement D.4 | RETAIN | **REPAIR** (criteria_weak) | S4, S3 | Public `aud.item` events, which `agents/base.py:96-103` delivers in every learner's observation, carry the platform's 32-d `emb` and `terms`. These match the encoder's output exactly (max diff 5e-7). From public data alone, an agent's post reached **7x** the audience of a natural article. D.4 scans only 16 configuration strings, so it cannot see this leak. |
| Requirement SP-4 | REDESIGN | **EXTEND** | S8 | This verdict is too harsh and contradicts the auditor's own subsystem verdicts (Social graph REPAIR, catalogue REPAIR). The population batch APIs work on population rows: 500 joins, 500 follows and 50 leaves, with clean invariants, visible in `community.view` and in profile follower counts. The missing piece is population behaviour on these platforms plus one state truth, not a new architecture. Population-space incompatibility stays under "Population integration", which remains REDESIGN. |
| criteria_weak on I.1, I.2, I.3, I.4, I.6, I.7 | false | **true** | S2, Gate I evidence | **I.1** counts table rows: only 3,628,555 of the 10M identities made any decision. **I.7** counts (identity, item) pairs, while spec 4.6 asks for 10M distinct identities with qualifying consumption for a 10M-viewer claim. **I.2/I.3** need no persistence or restore. **I.4** is measured only on the feed path, where the engine clamps attention by construction; C1 tools spend no attention. **I.6** can be met by a genesis broadcast (p6). |

### Probes and results

* **S1 `s1_pop_integration.py`** tests whether "Population integration = REDESIGN" is harsh. I aliased the two missing
  names at runtime only (`crucible.population.register_activity/wake -> population.core`). With that, the C2 engine
  installs and runs on B's `pop`: 1,042 rows and 30 reference participants over 3 days gave 1,982 consumption rows and
  0 recount mismatches. The install blocker is therefore shallow. The spaces, however, are unrelated. B's topic
  vector and C2's embedding of *the same words* have a mean cosine of 0.06. The true topic ranks 6.8 of 16 on average
  (chance is 8.5). Viewed items match the viewer's B topic 4.5 % of the time, which is about chance. The engine still
  "prefers" items (mean affinity viewed 0.116 vs exposed 0.027), but that affinity is noise. **REDESIGN upheld.**
  **C.3 = REPAIR upheld:** in S10, the official C.3 body with the aliases then fails only at the test's own
  `int(dict)` (`test_gate_C.py:238`).
* **S2 `s2_recount_persistence.py`** checks the accounting verifier. It produced 0 mismatches before restore.
  After `World.from_blob` and one more day it produced 439 (exposure 111, viewer 111), because `recount` reads
  `world.log` and `from_blob` sets `log_base = seq` (`kernel.py:958`). With `log_retain=2000` it produced 643
  mismatches. With a `MemoryStore` and `events=w.events_range(0)` it produced 0. On a 20-day run with the full log
  the result was 0, but `aud_batches` was by then retained only from day 6. Every caller (`cascade.py:131`,
  `test_gate_D.py:293`, `test_gate_SP_4_6.py`, `test_gate_C.py:236`) uses the window. Gate I avoids the problem only
  by raising `log_retain` to 10M (`cascade.py:53`).
* **S3 `s3_leaks_and_time.py`** checked the studio sandbox. Host reads of the simulator source, the manifest,
  `/etc/passwd` and `/proc/self/environ` all returned `PermissionError`, and the environment holds 7 variables (Landlock
  ABI 4 + seccomp): **studio RETAIN upheld**. A nit: `studio.run` error tracebacks reveal host paths
  (`/workspace/agentrs/.../sandbox/boot.py`, `/var/tmp/crucible-sbx/...`). Public `aud.item` payloads contain `emb`
  (32 floats) and `terms`.
* **S4 `s4_observable_shortcut.py`** asked whether the auditor's privileged token-gaming (p3) can be found from
  legitimate observations. It can. The agent used only public `aud.item` events, the public view counts from the
  `aud.item` tool, and an encoder oracle (publish a one-word post, read its public event). The views-weighted mean of
  public embeddings had cosine 0.85 and 0.70 with the hidden mean interest. A 120-token post of 12 unrelated
  vocabulary words picked this way ("derby blockchain hotel bass product users hammer ...") reached a mean of
  **2,287** unique viewers, against **320** for a natural football-coaching article; the privileged version reached
  3,277 (2 seeds, identical forks, 4,000 people). This strengthens **Content representation = REDESIGN** and
  overturns D.4.
* **S5 `s5_learned_recsys.py`** checked whether the learned recommenders learn. CF and engagement parameters changed
  8 and 7 times in 2 days. From the same day-2 state, views per impression were: random 0.299, trending 0.302,
  cf 0.360, engagement 0.388, chronological 0.395, interest 0.411. On 17,838 later impressions the engagement model's
  AUC was 0.589, against 0.634 for plain interest cosine. These are genuine learners, weaker than the hand-written
  cosine. **D.1, SP-4.5 RETAIN and Recommendation EXTEND upheld.**
* **S6 `s6_browser_entry.py`** compared the two web apps. With `make_web_app`, `social.crucible/` renders "Service
  host social.crucible ... endpoints 11 ... Add to my tools". With `make_social_web_app` it renders "Social network -
  Your timeline". `crucible.web.app.main` builds `make_web_app`. **C.6 RETAIN upheld** (the criterion is met by the
  test-built app); the subsystem is downgraded.
* **S7 `s7_engine_value.py`** compared a 15-word post with the same words repeated 40 times (identical embedding,
  max diff 0.0; 5 s vs 150 s; view thresholds 2.5 s vs 30 s). Over 2 seeds, unique viewers were 1,922/1,251 vs
  1,208/532, shares 53/12 vs 26/5, and subscribers 153/70 vs 133/29. Only total watch seconds favour the long piece.
  Satisfaction gain is `frac * (aff - 0.2)` (`consumption.py:379-381`), so a blurb fully read is worth as much as a
  substantial piece fully read.
* **S8 `s8_batch_apis.py`** exercised the never-called batch APIs. `communities.join_batch`, `touch_batch` and
  `leave_batch`, and `graph.follow_batch`, accept population rows as nodes (node i = p<i>). The results were 451
  members, 500 C1 followers, and 0 `aud` followers, so the two truths are confirmed. Invariants were clean and the
  calls took 6 ms. A minor defect: `community.health` counts members who only joined as `active_7d`/`active_30d`
  (451 active, of which 251 never visited); C.4 uses weekly contributors instead.
* **S9 `s9_media_publish_integrated.py`** published on posts, pics, pods and shorts with `aud` distributing all four.
  With the default encoder, posts was published and pics, pods and shorts were **refused**. With the semantic
  encoder, all four were published.
* **S10 `s10_c3_with_alias.py`** is described under S1.

### Upheld (attempted refutation failed or was not warranted)

Creator studio RETAIN (S3); Platform registry/catalogue REPAIR; Social graph REPAIR (S8); Communities EXTEND (S8:
the APIs exist and work, and nobody uses them); Content representation REDESIGN (strengthened by S4); Recommendation
EXTEND (S5); Population integration REDESIGN (S1, with the corrected characterisation); Scale path EXTEND; Gate harness
REPAIR (Clarification 1 permits scripted reference agents for capability validation; add the D.4 blind spot and C.3's
missing assertion to its defects); Social-economic EXTEND (the tool path exists; the population-level half is
effectively IMPLEMENT). For requirements: C.2, C.3, C.5, C.n1, I.1, I.5 REPAIR; C.4, I.n1, SP-4.7, SP-4.8 EXTEND;
C.6, D.1, D.2, D.3, D.5, I.2, I.3, I.4, I.6, I.7, SP-4.5, SP-4.6 RETAIN; and SP-4.2 REPAIR. The upheld RETAINs on
I.2, I.3 and D.5 hold only for single in-memory runs (S2).

### Additional findings

1. Every learner's observation carries the platform's content representation. Public `aud.item` events include `emb`
   and `terms` (`audience.py:200-203`, `vis="public"`). This hands the learner the hidden encoder's output and makes
   representation gaming learnable (S4).
2. Tool-based reading is attention-free. `content.open` (cost 2 s, read-only) and the C1 site, community and message
   reads spend no attention. Only `feed.*` charges attention (`recsys/service.py:299-304`). Finite attention is therefore
   enforced for the stand-in population but not for tool-using participants on 14 of the 16 platform kinds.
3. Gate I's I.1 evidence shows that 6.37M of the 10M identities never made a decision (3,628,555 did). No gate
   verifies 10M distinct identities with qualifying consumption, which spec 4.6 asks for before claiming 10M viewers.
4. `studio.run` and `studio.test` error tracebacks expose host filesystem paths to participants (minor, Gate W).

### Missing requirements (additions to the auditor's list)

* D.4 must probe representation leaks too: no observation, event or tool may carry per-item internal features
  (`emb`, `terms`). Add an adversarial "observation-derived gaming" test with an S4-style agent.
* Audience verification must survive snapshot/restore and log truncation: `recount` must run from the durable event
  store, and I.2, I.3 and D.5 must be checked on a resumed persistent run.
* C.1 must be measured in the integrated, distributed configuration with the default encoder.
* Spec 4.6: verify 10M *distinct identities* with qualifying consumption, or label the result as estimated.
* Finite attention for tool-using participants on every platform (reading, visiting, watching).
* The social browser pages must be reachable from the deployable browser entry point.
* Spec 4.4: a reader's gain must depend on what the content provides (information, usefulness, need), not only on
  topical cosine and fraction read.
