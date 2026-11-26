# Audit B_population: the autonomous simulated population (workstream B)

Auditor: independent audit subagent (recovery audit, Clarification 3). Worktree `feat/crucible-5-complete`
(audited code at a881d18; HEAD moved to dd745b9 during the audit, which only adds docs/manifest entries - no
population code changed). Scope: MASTER_SPEC Part 3 (3.1-3.4), Part 4.1, Gate B, Gate G, SP-3.1..SP-3.4, SP-4.1,
read against Clarification 1 (CAM independence), Clarification 2 (observation-only default) and the recovery
preamble (no previous completion claim counts without reproducible evidence).

Probe scripts and raw logs: `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/B_population/`
(each `probe_*.py` has a docstring with what it tests; `*.json` / `*.log` are the outputs quoted below).

## 1. Verdict in one paragraph

Workstream B built a real, deterministic, event-driven columnar population (`crucible/src/crucible/population`,
3,898 lines) that browses the 4.0 synthetic internet on its own: pages are fetched from the sites' own deployed code
through `net.fetch`, purchases are site checkouts paid from the individual's own bank account, every state change is a
recorded `cols.batch`, attention is a finite ledger, and nothing is initiated by a learner. At the standard profile
(10,296 individuals) it ran 30 simulated days in 2.0 h on one CPU (per-day wall cost rising 56 -> 400 s), with
17,400-21,200 sessions/day (575,862 in total) by 10,256 distinct individuals, zero attention violations and zero
learner-initiated events; the Gate B probe metrics (return/abandon gaps, preference effect, persistence) are met at
day 30 in this stand-alone world. Class-2 adaptive participants genuinely learn: with identical initial weights, the learning group's weight
hash changes, its held-out prediction error halves and its realised utility beats a frozen control by +0.11..+0.15
(95% CI excluding 0, 3/3 seeds), also after a controlled change of the sites. But the subsystem is NOT integrated:
nothing outside its own test file uses it, the social internet (C1/C2) cannot run on it (`crucible.population` exports
nothing, so `feed` raises at install), a duplicate population (`crowd`, C2) owns the same `pop` table, class-4 rows are
inert, the civilization (`civ`) and self-play (`multiagent`) packages are 5-line stubs, and **no fitted calibration
model is used by any population decision** although FIDELITY_REGISTER.md claims six are. Class 3 is a 4-head
categorical bandit over a scripted entrepreneur template with unstable A2C updates; it is not strategic learning. The
personal GRU memory shows no measurable benefit. No gate test for B, G, SP-3.1, SP-3.3, SP-3.4 or SP-4.1 exists
(`python -m crucible.acceptance.v5 run --requirement B.1` -> "No test files exist yet for this selection").

## 2. What I ran

| # | Command / probe | Result |
|---|---|---|
| 1 | `pytest -q tests/test_v5_population_core.py` | 11 passed in 51 s (replay/regeneration/resume digest equality, attention, bank purchases, wake, adaptive learn, policy-update records, entrepreneur deploys, admission) |
| 2 | `acceptance.v5 run --requirement B.1` | "No test files exist yet for this selection" - tests/v5/test_gate_B.py, test_gate_G.py, test_gate_SP_3_1/3/4, test_gate_SP_4_1 do not exist on any branch (checked `git log --all`) |
| 3 | `probe_calibration.py` | all 9 class-1 decisions use `plausible:*@1` hand-set coefficients; `browse_session@1`, `continue_watching@1`, `purchase@1`, `churn@1` do not exist; `content_consumption@1` and `commercial_response@1` load but their features (`user_log_plays_7d...`, `f0..f11`) match none of the population's features, so `models.build` silently keeps the plausible model |
| 4 | `probe_run.py development - 3 24 900` | 1,042 rows, ~25-30 s/sim-day; 13/13 Part 4.1 behaviours counted; 0 learner-initiated events; return gap rich-vs-bad 0.006 (p=0.84) on a 1-day window |
| 5 | `probe_scale.py standard 30 24` (slot scheduler, 1 CPU) | see section 4 (10,296 rows, 30 simulated days) |
| 6 | `probe_adaptive.py {0,1,2} 5` (slot scheduler) | learning vs frozen identical-init groups, controlled site switch at day 5 - see section 5 |
| 7 | `probe_adaptive_mem.py {0,1,2} 5` | learning with vs without personal GRU memory; class-1 baseline - see section 5 |
| 8 | `probe_frozen.py` | `freeze_copy` (adaptive and strategic) keeps frozen hashes fixed while sources keep learning; dev-scale strategic rewards all -0.012 (hosting cost only) |
| 9 | `probe_intervention.py` (fork at day 2, 3 seeds, 400 individuals) | redeploying the 2 most-visited rich sites as broken: abandon-rate gap +0.44..+0.55 (p 1e-22..1e-33); visit-share gap -0.04/-0.12/-0.06 (p 0.08 / 8e-9 / 0.007) |
| 10 | `probe_quality.py` (12 sites, 2 per behaviour) | mean utility rich 0.184, thin 0.110, offtopic 0.098, spam 0.035, flaky 0.001, broken -0.389; 2-day return rate rich 0.356, thin 0.20, broken 0.02 |
| 11 | `probe_agent.py` (two external learner agents deploy sites through ordinary tools) | both deployed and discovered (mostly via their forum post), `popweb.visits` works for the operator only (cross-read refused); BUT the agent's *thin* site got the same mean utility as the *rich* one (0.152 vs 0.147) and similar returns (18 vs 24); agent observations carry no reward field (only a forum vote `score`) |
| 12 | `probe_integration.py` | `crucible.population` exports neither `register_activity` nor `wake`; world `pop`+`aud`+`feed` fails at install: `ProtocolError: feed: the population module does not provide register_activity`; class-4 rows have `next_active_at = NEVER` and 0 sessions |
| 13 | `probe_profile.py` (cProfile, dev world, day 1 vs 5) | ~75% of wall time is sandboxed page execution (`sandbox/runner`, `net.fetch`); per-day cost grows with forum follows and entrepreneur sites |
| 14 | `probe_profiles.py large 10`, `internet 1` / `internet 8` (slot scheduler) | large (101,120 rows): create 1.1 s, first 10 sim-hours 40 s wall, 100,184 sessions, 1.04 GB; internet (10,001,120 rows): create 34-37 s, 5.3 GB; 8 sim-hours (00-08 h) ~490 s wall for 3.35M sessions, RSS rising to 11.8 GB, the 07-08 h hour alone 304 s for 0.9M sessions (per-session cost rising) - a simulated day would take hours on one CPU; no aggregate mode |
| 15 | `probe_stuffing.py` (genuine "rich" cooking site vs a word-salad of shuffled cooking lexicon words + filler) | word salad: utility 0.099, 1-day return 0.213, 158 abandons; genuine: 0.090, 0.184, 161 - the population cannot tell meaningless keyword text from articles |

## 3. Code walk-through (who calls what)

* `population/schema.py:53-113` - one columnar row per individual (all four classes), 18 Part-3.1 fields mapped in
  `FIELDS` (`schema.py:140-159`). `core.py:246-337` draws heterogeneous initial state from the world seed.
* `core.py:340-452` (`pop.tick`) - vectorised due-set, `models.Decision("session_start")`, activity choice by habit +
  goal affinity + Gumbel noise, dispatch to registered activities, one recorded session batch. Habits/satisfaction are
  learned EMAs of experienced utility. `core.py:349`: only classes 1-2 are ever activated; `core.py:333` sets classes 3
  and 4 to `NEVER`.
* `models.py:101-135` - nine logistic decisions with hand-set coefficients; `models.py:138-162` calibration plug-in
  requires every fitted feature name to be in `FEATURES` - never true (probe 3).
* `web.py` (`popweb`) - discovery (known resources, links, contacts' recommendations, forum mentions, search), visit
  pipeline over real page fetches (`web.py:193-222`), utility formula `web.py:362-370` (cosine of interest and the
  page's random-index text vector x richness from length and distinct-token share), purchases through the site's
  checkout and `bank.pay_invoice` (`web.py:431-454`), return/abandon/recommend/interest drift (`web.py:457-648`),
  forum reading/replying/following/joining (`web.py:748-865`), operator analytics `popweb.visits` (`web.py:897-914`).
* `sites.py` - generated site template with behaviours rich/thin/broken/flaky/spam/offtopic; `text.py:46-64` a
  closed 16-topic, 160-word lexicon from which both the individuals' interests and the generated sites are made.
* `adaptive.py` + `nets.py:61-83` - class 2: one `AdaptiveNet` (Linear encoder, GRUCell memory, 3-layer scorer;
  187,585 params) per policy group, personal memory column, used ONLY for the `web` activity's target choice
  (`web.py:274-283`); examples = (obs, memory, chosen-candidate features, realised utility) (`adaptive.py:344-361`);
  online MSE regression with Adam every 6 h (`adaptive.py:368-394`, loss `adaptive.py:112-123`).
* `strategic.py` - class 3: two groups (MLP 2.16M params REINFORCE/Adam; GRU 2.36M A2C/RMSprop), decision every 2
  days, 4 categorical heads (topic 16, price 5, content 3, promote 2) that parameterise ONE scripted "rich shop"
  template (`strategic.py:153-156`, `_carry_out` 389-423); reward = raw balance change / 1000 (`strategic.py:357`).
* `analysis.py` - evaluator-side metrics from the recorded log only (visit blobs, cols.batch events).
* Callers outside the package: none. `grep` finds the `pop` module used only by `tests/test_v5_population_core.py`.
  `social/recsys/service.py:183-186` and `social/consumption.py:487` try to use it and fail (probe 12).
  `social/scale.py` (`crowd`) is a second population implementation that declares the same `pop` table "because
  workstream B's population module is not merged" (`social/scale.py:3-7`); "A world lists crowd OR B's pop, never
  both". `civ/__init__.py` and `multiagent/__init__.py` are docstring-only stubs.

## 4. Gate B at the standard profile (probe 5)

Command: `python -m crucible.scale.slots --cpu 1 --gpu-mb 0 --label "audit: B_population gate B standard 30d" --timeout 16000 -- python probe_scale.py standard 30 24 scale_standard_30d.json 3,7,14,30`
(one CPU, 24 generated sites, tick 15 min, seed 11). Completed simulated days: **30**; wall seconds per day:
[56.4, 101.7, 176.4, 204.6, 204.6, 186.3, 198.0, 162.5, 191.3, 184.9, 191.3, 210.1, 204.6, 215.2, 227.7, 216.7, 233.2, 237.5, 245.7, 262.3, 262.7, 321.2, 286.5, 276.1, 280.9, 321.3, 385.7, 341.9, 334.0, 398.5] (RSS 3.01 GB at the end). Rows: 10296 ({'1': 10000, '2': 256, '3': 32, '4': 8}).

| day | distinct activated / visitors | visits | return gap useful-unsat. | abandon gap unsat.-useful | preference effect | same-situation SD | interest drift | last/first week sessions | attention violations | learner-initiated |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 10256 / 10020 | 41355 | 0.050 (p=3e-07) | 0.134 (p=2e-24) | 0.197 (p=5e-141) | 0.164 | 0.629 | nan | 0 | 0 |
| 7 | 10256 / 10251 | 93278 | 0.034 (p=3e-10) | 0.135 (p=4e-65) | 0.166 (p=0) | 0.181 | 0.679 | nan | 0 | 0 |
| 14 | 10256 / 10256 | 185052 | 0.138 (p=6e-78) | 0.118 (p=3e-55) | 0.376 (p=0) | 0.193 | 0.772 | 0.98 | 0 | 0 |
| 30 | 10256 / 10256 | 415728 | 0.114 (p=7e-119) | 0.096 (p=9e-80) | 0.345 (p=0) | 0.200 | 0.873 | 1.12 | 0 | 0 |

Sessions per day: {'0': 23697, '1': 18376, '2': 17587, '3': 17439, '4': 17416, '5': 17431, '6': 17489, '7': 17493, '8': 17754, '9': 17771, '10': 18161, '11': 18074, '12': 18497, '13': 18703, '14': 18707, '15': 18867, '16': 19286, '17': 19344, '18': 19675, '19': 19668, '20': 19700, '21': 19833, '22': 19932, '23': 20313, '24': 20410, '25': 20608, '26': 20866, '27': 20857, '28': 20650, '29': 21173, '30': 85}; total 575862. Days with activity: 30.
Part 4.1 behaviour counts at day 30: {'discover_websites': 118399, 'search_for_information': 89110, 'browse_feeds': 147254, 'visit_pages': 1100151, 'consume_media': 429258, 'follow_creators': 16022, 'join_communities': 6536, 'participate_in_discussions': 33534, 'evaluate_products': 27742, 'purchase_services': 11407, 'return_to_useful_resources': 200384, 'leave_resources': 73441, 'move_between_platforms': 253708}.
Class-2 (one shared group of 256): 84 updates, 1344 Adam
steps, 85 distinct weight hashes; max inference batch 71.
Class-3: strategic:A 56 updates, strategic:B 55 updates;
A mean reward first half 4.30 vs second half 9.12 (224 decisions); B mean reward first half 13.46 vs second half 40.91 (224 decisions); price-level histograms A [15, 217, 1, 5, 2],
B [6, 5, 1, 225, 3] (both groups collapse onto one price level); entrepreneur
balances min/median/max 551370/615116/1420234 cents (endowment 500,000). There is no frozen or
scripted control for class 3, so a reward trend cannot be attributed to learning rather than to the market growing.
Classes 3 and 4 have 0 browsing sessions (by design for 3, by omission for 4).

Reading: the B mechanics hold at the standard profile for 30 simulated days (distinct autonomous activity,
preference effects, return/abandon separation once a 7-day window is used, stable activity, no attention violations,
no learner-initiated events). This is unofficial evidence from a probe, not a gate test; the population is a
stand-alone world (24 generated sites + entrepreneur sites, one forum), not the integrated civilization.

## 5. Gate G / SP-3.4: do class-2 participants learn? (probes 6-8)

Design: 150 lightweight + 256 adaptive individuals, 10 hand-specified sites (5 rich, 2 broken, 2 thin, 1 spam).
Two adaptive groups with the SAME seed (identical initial weights, hash `ccad6896...` for seed 0): group 0 learns
online, group 1 is frozen. After 5 days the two most-visited rich sites are redeployed as broken and the two broken
sites as rich (a controlled new condition); 5 more days.

| seed | g0 hash changed / g1 unchanged | updates (Adam steps) | held-out MSE learned vs initial | phase-2 utility improvement (learning - frozen), 95% bootstrap CI | phase-2 share of visits to failing/thin/spam sites g0 vs g1 |
|---|---|---|---|---|---|
| 0 | yes / yes | 22 (352) | 0.036 vs 0.075 | +0.111 [0.079, 0.144] | 0.413 vs 0.491 |
| 1 | yes / yes | 22 (352) | 0.055 vs 0.075 | +0.114 [0.082, 0.146] | 0.441 vs 0.453 |
| 2 | yes / yes | 21 (336) | 0.042 vs 0.082 | +0.153 [0.119, 0.187] | 0.385 vs 0.535 |

This is genuine learning from the participants' own experience (gradient updates on their realised utilities;
held-out error falls; behaviour improves against an identical-init frozen control). Caveats that keep it below RETAIN:

* The improvement is already present before the switch (phase-1 utility 0.069/0.041/0.093 vs 0.032/0.007/0.006), so
  most of it is general learning ("prefer candidates with high personal satisfaction `known_sat`": chosen known
  resources have mean known_sat 0.15/0.13/0.17 vs 0.10/0.07/0.09 frozen), not adaptation specific to the change; the
  failing-site share difference after the switch is large in 2 of 3 seeds only.
* The frozen randomly-initialised network is a weak baseline. The class-1 heuristic on the same worlds reaches 0.10-0.13
  phase-2 utility vs 0.15-0.20 for learned adaptive participants (probe 7) - better, but the heuristic already encodes
  the same features (`web.py:257-258`).
* Personal state does not matter: learning WITH the GRU memory vs WITHOUT (probe 7): -0.018 [-0.051, 0.013],
  -0.051 [-0.080, -0.022], +0.028 [-0.004, 0.059]. 256 distinct memories exist (G.2) but they carry no demonstrated
  information. "Persistent personal-state adaptation" (Part 3.4) is not demonstrated.
* The network decides only which resource to visit in the `web` activity; session start, activity choice, reading
  depth, purchases, search, forum, recommending and leaving stay class-1 logistic rules for class-2 individuals.
* Batched inference: max batch 66 by day 7 and 71 by day 30 at the standard profile (one group; `adaptive_stats`), 40 with two groups; the
  64-row G.3 threshold is met only marginally. Per-update "loss decrease" is trivially satisfied (16 Adam steps on 32-90
  fresh examples: e.g. 0.0705 -> 0.0203), so the G.5 `loss_decrease` criterion proves nothing without a held-out loss.

Frozen historical participants (probe 8): `pop_adaptive.freeze_copy` and `pop_strategic.freeze_copy` create copies
whose hashes stay fixed for 2 more days while the sources change; frozen members keep acting (42 examples logged by
the frozen adaptive group). Independently updated groups: each strategic group trains only on its own members'
experience (`strategic.py:425-441`).

Class 3: at the development scale the strategic reward is a constant hosting cost (40/40 rewards = -0.012) - no
learning signal. At the standard scale rewards vary (A: mean 3.5, max 12.3; B: mean 6.6, max 37.1 by day 7) but the
A2C group's loss swings between +965 and -7,040 (unnormalised advantages, raw reward scale, 8 RMSprop steps on a
handful of examples), and the "strategy" is a choice of four parameters of a fixed script. Each member makes ~15
decisions in 30 days. Over the full 30-day standard run both groups collapse onto a single price level (A: 217/240
decisions at price index 1; B: 225/240 at index 3); group A's REINFORCE loss is exactly -0.0 for the last updates and
its 56 updates produced only 54 distinct weight hashes, i.e. the policy became deterministic and stopped changing (entropy
collapse), not a learned strategy. Both groups' mean rewards rise over the run (first vs second half: A 4.3 -> 9.1,
B 13.5 -> 40.9), including group A after its policy had stopped changing - so the rise reflects the market growing as
visitors discover the entrepreneur sites, not learning. There is no frozen or scripted control. There is no cooperation, no unrelated objective and no role other than entrepreneur (Part 3.2
lists ten).

## 6. Calibration actually used by the population

None. `FIDELITY_REGISTER.md:23-28` states that `pop (workstream B)` uses `content_consumption@1`, `watch_duration@1`,
`return_daily@1`, `session_rate@1`, `news_click@1`; `:31-32` that population ad response uses
`commercial_response@1`. Probe 3 shows every population decision runs `plausible:*@1`; `watch_duration`,
`return_daily`, `session_rate` and `news_click` are not referenced anywhere in `population/`. (`news_click@1` is used by
C2's `feed` through `consumption.CalibratedOpen`, which runs only on the `crowd` population.) The register is false for
the population and must be corrected; Gate F's "fitted against data" does not reach the population's behaviour.

## 7. Transfer vs simulator shortcuts (Part 3 objective)

* Content judgement is a closed-lexicon bag-of-words (`text.py:46-64`, random-index vectors) times a richness score
  `sqrt(min(1, chars/2000)) * min(1, 1.6 * distinct_share)` (`web.py:365-370`). A learner who discovers this can
  maximise visits with long, varied lists of the 160 topic words; text outside the lexicon is noise to every
  interest. Interests cannot point at genuinely new topics except by slow drift towards consumed vectors. No
  semantic encoder (Gate E, workstream D) is wired in (`text.py:16-18` only mentions a future switch).
* Probe 15 makes the shortcut concrete: a site whose articles are shuffled cooking-lexicon words with filler (no
  sentences, no meaning) earned slightly MORE utility and returns than a site with composed articles on the same
  topic (0.099 vs 0.090; 1-day return 0.213 vs 0.184). (The "genuine" generated articles are themselves template
  prose from the same lexicon, `sites.py:47-57`, so neither side has real meaning.) An agent that learns here would
  learn keyword coverage, not usefulness - this does not transfer to the open internet.
* The agent probe: a thin site and a rich site on the same topic get the same visitor utility (0.152 vs 0.147),
  because most visits end on a short home page - quality signals that would transfer (depth, accuracy, usefulness for
  a task) are not modelled.
* Utility, satisfaction and purchase decisions are hand formulas with hand coefficients (no data); purchases happen
  at willingness-to-pay `2500 * wtp_scale * (1 + 1.5 relevance)` cents (`web.py:387`), a constant independent of
  the product.
* No reward leaks to learners: all population events are `vis="internal"` (`substrate/columns.py:115`), `pop.me`
  serves only population members, `popweb.visits` only the site's operator (probe 11: cross-read refused). Operator
  analytics (visits, returns, purchases, revenue) are real observable consequences, compatible with Clarification 2.
* No direct counter writes: resource counters change only by recorded `mode="add"` batches whose visit log is stored
  as an evaluator-readable blob; invariant `visits == first + return` holds (`web.py:931-946`).

## 8. Subsystem verdicts

See `B_population.json` (`subsystems`). Summary:

| Subsystem | Verdict | Effort |
|---|---|---|
| Persistent individual state (schema, genesis, state evolution) | EXTEND | M |
| Activation scheduler, attention ledger, money/bank path, determinism/replay of learned weights | RETAIN | S |
| Class-1 decision models + calibration plug-in | REPAIR | L |
| Resource-visit engine popweb / popsites / text representation | EXTEND | L |
| Population <-> social internet integration (`pop` vs `crowd`, package API) | REDESIGN | L |
| Class-2 adaptive neural participants | EXTEND | M |
| Class-3 strategic neural participants | REDESIGN | XL |
| Class-4 semantic generalists as population members | IMPLEMENT | L |
| Population profiles and large-scale simulation | EXTEND | L |
| Evaluator-side population analysis | REPAIR | S |
| Gate tests B, G, SP-3.1, SP-3.3, SP-3.4, SP-4.1 | IMPLEMENT | M |

## 9. Requirement verdicts

| id | verdict | criteria weak | reason | work needed |
|---|---|---|---|---|
| B.1 | REPAIR | yes | Mechanism works at 10,296 individuals (standard profile, 30 simulated days, 10256 distinct activated individuals); no gate test, no official evidence; the population is not part of any Tier 2/3 world. | Write tests/v5/test_gate_B.py running the standard profile 30 days through the slot scheduler; count distinct individuals with autonomous activity (not declared rows); run it in the integrated civilization once pop is unified with crowd. |
| B.2 | REPAIR | no | 0 learner-initiated population events (analysis.learner_initiated), 575,862 sessions in 30 days (17.4k-21.2k/day) at 10,296 individuals; no gate test. | Gate test with the same metrics plus a learner actively using tools in the world, to show consumer events stay population-initiated while a learner acts. |
| B.3 | REPAIR | no | Preference effect (return rate above vs below the per-resource median interest match, CMH-stratified) 0.17-0.20 at day 3/7 and 0.345 at day 30 (p~0, 100,370 pairs over 54 resources); mechanically built into the hand utility formula; no gate test. | Gate test; also show effects of non-interest preferences (price sensitivity, loyalty, novelty) on the corresponding decisions with randomised exposure. |
| B.4 | REPAIR | no | Standard profile, 7-day window: day 30: return gap 0.1142 (p=6.8e-119), abandon gap 0.0960 (p=9.1e-80); criteria met in the probe from day 14 on, but at day 7 the return gap was only 0.034 (<0.05) and on the dev world with a 1-day window 0.006 (p=0.84): the effect needs about a week of history. Labels come from generated-site behaviours only; no gate test. | Gate test with a fixed protocol (window, evaluator-side usefulness labels that also cover agent-built resources); strengthen return-to-useful dynamics (bookmarks, subscriptions) so it holds at shorter horizons. |
| B.5 | REPAIR | no | Standard profile ran 30 simulated days with activity on 30 days, sessions/day 17416..23697, last-week/first-week sessions 1.12 (criteria met in the probe); wall 2.0 h on one CPU, per-day cost rising 56.4 -> 398.5 s. No gate test, no official evidence, not in an integrated civilization. | Gate test over >=30 simulated days at the standard profile (wall ~2 h on one CPU); reduce per-day cost growth. |
| B.6 | REPAIR | yes | Fork-based intervention (2 most-visited rich sites redeployed broken) at 400 individuals, 3 seeds: abandon-rate gap +0.44/+0.51/+0.55, p<1e-21; visit-share gap significant in 2/3 seeds. Not run at 10,000; no gate test. | Gate test at the standard profile with paired forks, >=3 seeds, pre-registered outcome and minimum effect size; include positive changes (new useful resource) as well as degradations. |
| G.1 | REPAIR | yes | Standard profile has 256 class-2 rows; 254-256 act within days; decisions are batched in 2-hour rounds with max batch 71 over 30 days, so 'simultaneously active' is not measured by any code. | Define and measure simultaneous activity (e.g. distinct adaptive individuals acting within one decision window / one day), gate test. |
| G.2 | EXTEND | yes | 256 distinct personal memories exist, but with-memory vs without-memory learning shows no benefit (-0.018, -0.051, +0.028 utility), so the personal state carries no demonstrated information. | Train memory with multi-step objectives; test persistence across sessions and that individual memory changes behaviour (memory-ablation arm must lose). |
| G.3 | REPAIR | no | One shared AdaptivePolicy object per group; max inference batch 71 over 30 days at the standard profile (40 with two groups). | Gate test; raise batch sizes (decision rounds per group) with margin above 64; measure throughput. |
| G.4 | REDESIGN | yes | Two strategic groups (MLP/REINFORCE, GRU/A2C) update independently (probe_frozen) and hashes differ, but the 'strategy' is 4 categorical parameters of a scripted template, A2C updates are unstable (loss 965 -> -7040) and no strategic improvement is shown. Over 30 days group A collapsed (loss -0.0, 56 updates -> 54 distinct hashes). | Redesign class 3 (tool-level actions, roles, stable algorithms) and require each group's hash to change from its own members' data, freeze-one-group independence, and measured strategy improvement. |
| G.5 | REPAIR | yes | Class-2 weight hashes change from experience (21-22 updates, 336-352 Adam steps in 10 days; frozen identical-init control unchanged); held-out MSE 0.036-0.055 vs 0.075-0.082. No gate test. | Gate test recording hashes, examples, held-out loss (not the per-batch training loss) and a perturbation-only negative control. |
| G.6 | REPAIR | yes | Learning vs frozen identical-init group after a controlled site switch: +0.111 [0.079,0.144], +0.114 [0.082,0.146], +0.153 [0.119,0.187]; but the gain is mostly general (present before the switch) and the failing-site share drops clearly in 2/3 seeds only. | Pre-registered new-condition protocol measuring change-specific adaptation (difference-in-differences vs pre-switch), against class-1 and memoryless baselines, in the integrated world; gate test. |
| G.7 | IMPLEMENT | yes | No comparison of the population's compact numerical policies with language-capable alternatives on population decisions exists (SP-3.2b compares models on semantic capability tasks only). | Run the same browsing/entrepreneur decisions with AdaptiveNet/strategic nets and with small LMs (SmolLM2/Qwen via crucible.semantic) on identical situations; report quality, cost and latency. |
| G.n1 | REPAIR | yes | Actual adaptation is shown at small scale (see G.6) but only for one decision type and with a weak frozen baseline. | Same as G.6 plus coverage of more decision types and of the strategic class. |
| SP-3.1 | EXTEND | yes | All 18 fields represented, heterogeneous and evolving; same-situation utility SD 0.16 (day 3) -> 0.20 (day 30), interest drift cosine 0.63 -> 0.87 with unvisited individuals unchanged; but employment and goals evolve by random draws, several fields only trivially (attention left, wake time). | Experience-driven dynamics for every field, gate test measuring evolution caused by experience (vs a no-experience control) per field. |
| SP-3.2 | IMPLEMENT | yes | Classes 1-3 exist; class-4 rows are inert (next_active_at=NEVER, 0 sessions); D2's semantic participants are not population members. | Drive class-4 rows with crucible.semantic participants using ordinary tools inside the population world; test that each class acts. |
| SP-3.3 | EXTEND | yes | Profile table matches Part 3.3; standard and large run; internet profile creates 10M rows (5.3 GB) but there is no aggregate/lazy mode, no labelled estimate, and no benchmark in the repository. | Benchmarks for all four profiles in the repository; lazy/sparse state and validated aggregate mode for the internet profile; test. |
| SP-3.4 | EXTEND | yes | Shared-policy learning shown; independent groups and frozen historical copies work (probe_frozen); personal-state adaptation not shown (memory ablation no effect); competing architectures exist but strategic learning is not shown. | Five separate tests each with weight hashes, training examples, held-out losses, behaviour change and evaluation outcome; plus negative controls (memory edit only, random perturbation). |
| SP-4.1 | EXTEND | yes | All 13 behaviours occur (>=10 events each; e.g. day 3 standard: 20k discoveries, 13k searches, 335 purchases, 17.9k leaves) with a finite attention ledger (0 violations) - but on a text-only web plus one forum; 'consume media' and 'move between platforms' are page reads and activity switches, not media or distinct platforms. | Run the population on the C1/C2 platforms (video, audio, images, subscriptions) and count platform switches across real platforms; gate test with meaningful minimum rates. |

## 10. Requirements in the spec with no manifest entry

* **Part 3.2 class 1** - Train and calibrate the class-1 behavioural models using appropriate empirical datasets wherever possible. Gate F checks that models were fitted, not that the population's decisions use them; no entry requires population decisions to run fitted models (they currently run none).
* **Part 3.2 class 2** - Adaptive participants must adapt their behaviour when products, environments or other participants change. G.6 tests one new condition; no entry covers changes of products and of other participants.
* **Part 3.2 class 3** - Strategic participants operate as entrepreneurs, researchers, engineers, developers, employees, organizations, investors, competitors, collaborators, community operators; learn strategies rather than execute predefined decision trees; some compete, some cooperate, some pursue unrelated objectives, some adapt throughout the lifetime; the civilization is not a collection of identical copies of one policy. Only G.4 (>=2 groups, distinct hashes) and Q.3 (one cooperative and one competitive interaction, workstream K) exist; role diversity, strategy learning vs scripted baseline and objective diversity have no entry.
* **Part 3.2 class 4** - Semantic generalists act as population members with ordinary world tools, not wrappers around scripts; smallest sufficient model per capability. H.* and SP-3.2b test semantic participants in separate worlds; nothing requires them to be members of the population.
* **Part 3.3** - Benchmark every profile; large profiles use resource-aware sparse or lazy state and batched inference; do not allocate a checkpoint per participant; internet-scale aggregate simulation must be validated and distinct-viewer claims need distinct identities or a labelled statistical estimate. V.1 (workstream M) benchmarks development/standard/large only; I.1/I.3 (workstream C) target the crowd module; no entry covers the internet profile of the population or validation of its aggregate mode.
* **Part 3.4** - Changing a participant's written memory is not sufficient evidence of neural learning; randomly perturbing parameters is not learning; record training examples and evaluation outcomes. No negative-control requirement (e.g. perturbation-only or memory-edit-only arms must fail the learning criteria).
* **Part 4 introduction / 4.1** - All platforms share the civilization's participant population; participants move between different platforms and prioritise opportunities by learned preferences and goals. No entry requires the social platforms (C1/C2) to run on the same population as browsing, commerce and the strategic classes; currently they cannot (pop vs crowd).
* **Part 3.1** - Preferences and decisions evolve through experience (for every listed field, not only interests). SP-3.1 only measures interest drift; employment/goals evolve by random draws and would pass 'evolving' checks.
* **Clarification 2 applied to the population** - Participant-internal utilities/satisfaction never reach a learner through observations, tool responses or event metadata. OO.2 is generic; a population-specific probe (pop.me, popweb.visits, forum content, event visibility) should be part of it.

## Skeptic review

Reviewer: skeptic subagent for area B_population. I tried to refute every RETAIN and every generous verdict above, and
checked the harsh verdicts too. Probe scripts and raw outputs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/B_population_skeptic/`. Each `probe_*.py` has a
docstring saying what it tests; each `*.json` / `*.log` holds its output. The code is unchanged since a881d18: HEAD
f394dec only adds docs and manifest entries. All runs used one CPU thread; the standard-profile run went through
`crucible.scale.slots`.

### S.1 What I ran

| # | Probe / command | Result |
|---|---|---|
| S1 | `probe_attention_binding.py normal` (dev: 400 individuals, 8 sites, 3 days) | No individual ever used >=95% of the daily attention budget (0.0 on all 3 days). Mean spent/budget 3.2%, 3.2%, 2.9%; p99 15-18%; max 35%. 1.66 min per visit, 1.9 min per individual per day. 0 violations. |
| S1' | same, arm `x20` (every budget x20 before the first tick) | Behaviour unchanged: sessions 2229 -> 2178, visits 1375 -> 1342, pages/visit 2.22 -> 2.24, minutes 2289 -> 2208, abandons 677 -> 666, and the 13 behaviour counts are within noise. |
| S1b | `probe_attention_binding_std.py std` (STANDARD profile, 10,296 rows, 24 sites, 2 days, slot scheduler, 203 s) | No individual-day was exhausted (0.0, 0.0). Mean spent/budget 4.0% / 4.2%; p99 19% / 21%; max 49% / 72%. 52,231 attention-minutes for 10,256 active individuals = **2.5 min of browsing per individual per day** against a median budget of 91 min. |
| S2 | `probe_cheap_product.py {1,2,3}` (3 sites with identical content and title: no products / $1 products / $300 products; 600 individuals, 3 days) | **My hypothesis was refuted.** The hard-coded purchase bonus (`web.py:398`) is not an exploitable shortcut at the purchase rates observed (12-20 purchases per 3 days). 1-day return rates for cheap vs no-shop: 0.16 vs 0.21, 0.17 vs 0.23, 0.22 vs 0.26 (cheap is not ahead). Nobody ever bought a $300 product: willingness to pay (`web.py:387`) does not depend on the product. |
| S3 | `probe_replay_learner.py` (test world plus an external learner who deploys a shop, posts, redeploys; odd-time snapshot at 1 d 05:13:07) | Replay digest equal, regenerate hash equal, resume-from-snapshot + same learner commands gives equal hash and digest. A log with one tampered population `cols.batch` is refused (`hash chain broken at seq 7476`). learner_initiated 0 of 1,548 population decision events while the learner acted. |
| S4 | `probe_text_judgement.py` (static; the population's own `text.py` and the `web.py:365-368` utility formula; cooking-interest reader) | A **tax-filing article with one appended line of cooking keywords scores u=0.743**, above a genuine English sourdough article (0.719). The same sourdough article with 11 lexicon words swapped for synonyms drops to relevance 0.243 (from 0.452) and u=0.439, **below a meaningless lexicon word salad (0.451)**. The generated "rich" prose (the evaluator's "useful" label) has distinct-token share 0.156 and richness 0.237, against 0.844 for the real article. Topic separability: at D=32 the mean absolute cosine between topics is 0.136 (max 0.451); at the internet profile's D=16, 16 topic pairs have cosine >0.3 (max 0.746). |
| S5 | `probe_integration_patch.py` (in-process monkeypatch only: `crucible.population.register_activity/wake = core.*`) | pop + aud + feed **creates and runs 1 day**: activities web/search/forum/browse, 44 `browse` sessions, 37 `feed.session` batches, 0 attention violations, no invariant errors. The ProtocolError the auditor reported is only a two-name export bug. |
| S5b | `probe_space_mismatch.py` (static) | The real blocker: pop's `interests` live in `population/text.py` space (sparse ternary, B's lexicon) while the social engine scores `cosine(pop.interests, item.emb)` with `social/textrep.py` embeddings (dense Gaussian, its own vocabulary; `recsys/features.py:65-80`, `consumption.py:432`). Both are 32-dimensional, so nothing raises. Across the two spaces a pure topic fan ranks its own topic first for **3 of 16 topics** (16 of 16 in pop's own space); the mean matching cosine is 0.047 across spaces vs 0.933 within pop's. |
| S6 | `probe_learner_metric_control.py` (positive control: an in-process probe tool lets a learner wake 20 members, 3 times) | The metric can fail: `learner_initiated` = 3 of 229 with nudges, 0 of 194 without. But it counts only the 3 wake batches. The sessions the learner forced (`core.py:381-394`: a woken individual's activity is overridden to the wake activity) are recorded inside `pop.tick` and are not attributed to the learner. |
| S7 | `probe_models_status.py` | Reproduced: 0 of 9 population decisions use a fitted model, all `plausible:*@1`. `models.py:155` silently swallows every loading exception (`except Exception: continue`). |
| T1 | `pytest -q -p no:cacheprovider -x tests/test_v5_population_core.py -k "attention or purchases or replay"` | 3 passed in 44.02 s (reproduced). |

### S.2 Verdicts overturned

1. **Activation scheduler, attention ledger, money/bank pathway, deterministic replay: RETAIN -> REPAIR (M).**
   * The determinism part survives a harder test than the repository's: learner commands in the log, an odd-time
     snapshot, and a tamper negative control (S3). That component on its own would be RETAIN.
   * The attention ledger does not limit behaviour, so Part 4.1 ("limited attention and time ... cannot perform
     unlimited actions") is not actually met:
     - The "0 attention violations" evidence holds by construction. `attention.spend` grants `min(request, left)`
       (`substrate/attention.py:54-77, 99`), so population code cannot violate it.
     - No individual-day came near the budget at either profile. Mean use was 3-4% (S1, S1b), and giving everyone
       20x more attention changed nothing (S1').
     - Individuals browse about 2.5 min per day, while the 18-minute `session_min` drawn at genesis is never used.
       A session occupies no simulated time, so attention never competes across activities or platforms.
   * "0 learner-initiated events" was measured in worlds without a learner. With a learner present it stays 0 (S3).
     But the metric attributes only the trigger, not the consumption a trigger forces (S6).
   * Work needed:
     - Make time and attention bind: calibrate budgets, dwell and session length to empirical time-use data, let
       sessions occupy simulated time, and draw every activity (web, forum, feed, messaging) from one budget.
     - Add a test in which attention binds for a measurable share of individual-days, and a counterfactual showing
       that a budget change moves behaviour.
     - Keep the replay machinery.

2. **Resource-visit engine (popweb/popsites/text): EXTEND -> REDESIGN for its content-judgement/utility model (L-XL).**
   * The visit mechanics are sound and can be kept: real page fetches, checkout through bank invoices, recorded
     batches, the visit log.
   * What the population counts as useful is a hand formula of closed-lexicon token overlap times length and
     distinct-token share (`web.py:365-368`, `text.py:46-64`). S4 shows this rewards keyword placement over meaning:
     - Off-topic text plus a keyword line beats a genuine article.
     - Synonyms halve relevance.
     - A word salad beats the synonym article.
   * The auditor's stuffing probe (one seed, 0.099 vs 0.090) points the same way.
   * Further shortcuts:
     - Every individual reads a global, population-shared perceived-content cache (`res.vec`, `web.py:240-244`), so
       it knows the content vector of a site it has only heard of.
     - Willingness to pay ignores the product (`web.py:387`).
     - A purchase adds a fixed utility bonus (`web.py:398`). S2 shows this bonus is not exploitable at today's rates.
   * Because both the judgement and the evaluator's "useful" labels come from the same generator, an agent trained
     here learns lexicon coverage, which does not transfer to the open internet. Swapping in an encoder (the
     `text.py:16-18` plan) is not enough. The utility model, the evaluator's usefulness labels (see B.4) and the
     population's interest space all have to be rebuilt around one learned, shared content representation, with
     anti-gaming tests.

### S.3 Verdicts examined and upheld

* **Population <-> social integration: REDESIGN upheld.** The auditor's evidence (a ProtocolError at install) was
  weaker than the verdict: S5 shows it is a two-name export fix. But S5b shows the deeper problem: the two subsystems
  have unrelated content spaces, so the obvious fix would run silently with random affinities. Gates D, I and SP-4.6
  run on `crowd` (`tests/v5/test_gate_D.py:3`, `test_gate_I.py:3`, `test_gate_SP_4_6.py:45`). One population and one
  shared content and interest space are needed.
* **Class-1 models REPAIR:** reproduced (S7). The silent exception swallowing should be removed so that a missing or
  mismatched fitted model fails loudly.
* **Class-2 EXTEND:** not overturned; the identical-init frozen control and the class-1 comparison are real evidence.
  One more defect: candidate features use the shared `res.vec` cache (privileged information, above).
* **Class-3 REDESIGN:** upheld. One more defect: an entrepreneur's site is always generated with behaviour "rich"
  (`strategic.py:155`), so its quality is assigned by the template, not produced by the participant. The policy-group,
  checkpoint and freeze infrastructure is reusable, which is why this is REDESIGN rather than REBUILD.
* **Class-4 IMPLEMENT:** upheld. Statically, `core.py:333` and `core.py:349` never activate class 3 or 4.
* **Gate tests IMPLEMENT:** upheld. `tests/v5/` has no `test_gate_B*`, `test_gate_G*`, `test_gate_SP_3_1/3_3/3_4` or
  `test_gate_SP_4_1`. `test_gate_SP_4.py` tests platforms with named actors, not the population.
* **Profiles EXTEND:** upheld, with one more defect: at the internet profile's D=16, the population's 16 interest
  topics are poorly separable (S4).
* **G.4 REDESIGN (harsh):** defensible only through criteria_weak. The literal criteria (>=2 groups, distinct hashes)
  are already met by the current groups; the REDESIGN reflects the spec's class-3 intent and should be kept, with the
  criteria strengthened rather than passed.

### S.4 Requirement changes

| id | auditor | skeptic | change |
|---|---|---|---|
| B.2 | REPAIR, weak=no | REPAIR, **weak=yes** | The metric is trivially 0 in learner-free worlds and counts only triggers, not forced consumption (S6). The gate must run with active learners and attribute consumption caused by learner-triggered wakes. |
| B.4 | REPAIR, weak=no | **REDESIGN, weak=yes** | Circular. "Useful" is the generator's behaviour label, and the population's utility formula is built on the same lexicon/length features the generator produces. It needs an evaluator-side usefulness label independent of the utility formula, which depends on the content-judgement redesign. |
| SP-4.1 | EXTEND | EXTEND (work added) | "Limited attention and time" is not met in substance (S1/S1b). Add a binding-attention criterion. |

### S.5 Missing requirements (in addition to section 10)

* **Part 4.1:** limited attention must actually constrain behaviour. No entry requires a binding share of
  individual-days, a budget counterfactual, or empirically plausible daily time use; "0 violations" holds by
  construction.
* **Part 4 / 4.3 applied to the population:** the population's interests and every platform's content
  representations must live in one shared space. No entry checks it, and today they do not (S5b).
* **Gate E / Part 4.3 applied to the population:** content judgement must be robust to keyword stuffing and
  off-topic keyword appendices. No anti-gaming criterion exists (S4).
