# Temporal world models, SocialWebLab and observational learning

This extends the existing agent (LM policy, action-conditioned world model, planner, RL, content
actions) with three capabilities, all integrated into the same training loop and all executed in
this repository's CPU sandbox:

* **A. multi-timescale consequence prediction** — a time-conditioned world model that predicts
  what an action leads to after any elapsed time, learns from outcomes as they mature, and handles
  outcomes that have not happened *yet*;
* **B. realistic web environments** — SocialWebLab, a self-hosted multi-user web application with a
  real database, HTTP API, HTML/JS frontend, background jobs and simulated users, driven by a real or
  accelerated clock and operated through real Chromium; plus one adapter interface over it, MiniWeb
  in Chromium, BrowserGym/MiniWoB++, and (not runnable here) WebArena / WorkArena;
* **C. observational learning from other actors** — an independently running sensor network that
  observes other actors' actions through authorised sources, forecasts their consequences before they
  happen, follows them up, and trains the same world model on what actually happened.

Compatibility map and the audit of what was reused: [`temporal_plan.md`](temporal_plan.md).
Measured results: section 9 below and `results/temporal/`.

## 1. Time

* `temporal/time_encoding.py` — elapsed time on a log scale, `tau(h) = log(1+h)/log(1+30 d)`, embedded
  with polynomial + Fourier features and a learned projection. Any duration has an embedding.
* `temporal/horizon_config.py` — default tiers (immediate 0/1/5 s … extended 14/30 d) and per-action
  profiles (a click: 0/1/5 s; a forum post: 1 m … 7 d with a 14-day outcome window; a ticket; a code
  review; …). Profiles decide what is *observed*; the model can be *asked* about any duration.
* `temporal/clocks.py` — three clocks are stored separately everywhere: environment time (when it
  happened, if the environment reports it), observation time (when a sensor saw it) and training time
  (wall clock, when a learner consumed it). Times carry a time base (`real` or `virtual:<lab id>`);
  the ledger refuses to mix bases, so simulated elapsed time is never presented as real time.

## 2. The temporal world model (`temporal/temporal_world_model.py`)

`TemporalWorldModel` subclasses the existing `ContentConditionedWorldModel` (same LM trunk; the
immediate next-observation diff head, event heads and content heads still work) and adds
time-conditioned ensemble heads on the trunk's pooled state:

```
predict(belief_state, action, horizons, relevant_context) -> TemporalPrediction
```

| outcome type | distribution | properties |
|---|---|---|
| events (first response, answered, positive feedback, resolved, removed, edited) | mixture **cure model** with a piecewise-linear hazard in log time | P(by h) monotone in h; P(ever) separate from "not yet"; median time-to-event; interval / right censoring; cause-specific competing risks |
| counts (responses, reactions, views) | monotone cumulative mean + **negative binomial on increments** | observing the same post at 1 h and 24 h never counts the first 10 replies twice |
| state (visible / hidden / removed) | categorical at h | not monotone (hidden posts can be restored) |
| continuous (net score) | Gaussian with learned variance | |

One trunk forward per (context, action) serves every horizon, so 8 horizons cost about the same as
one. Each head is an ensemble trained on bootstrap masks; member disagreement is reported next to
every probability (epistemic uncertainty). The input text contains the complete action including
the generated content, the actor/source type (our own executed intervention vs an observed third
party), the goal (or `unknown`), the context, and — for **updated conditional forecasts** — the events
observed between the action and the information cutoff plus the state at the cutoff.

Losses (`temporal/event_time_model.py`, `TemporalWorldModel.temporal_loss`): exact-time density,
interval-censored probability, right-censored survival; left truncation at the cutoff for updated
forecasts; NB increments; categorical and Gaussian likelihoods. Masks keep unknown horizons out of
the loss, and by default every action has the same total weight however many times it was observed.

## 3. Delayed outcomes

* **Ledger** (`temporal/delayed_outcome_ledger.py`, SQLite): actions (ours and others', complete
  content, generating checkpoint, permissions, time base, outcome window), **insert-only forecasts**
  (SQLite triggers abort updates; deletes only inside a logged governance purge), observations
  (intended vs actual time, completeness, the job that made them, idempotent), follow-up events
  (environment time vs observed time, by the original actor or not, deduplicated), rewards (with
  maturity time and checkpoint), training-clock ingestion records and a governance log.
* **Labels** (`temporal/outcome_labels.py`): exact / interval / right-censored with a status —
  `pending` (window still open), `closed_negative`, `censored` (observation stopped), `precluded`
  (removed first). Nothing pending is ever turned into a failure. An example at information cutoff
  `c` sees only what had been *observed* by `c`.
* **Scheduler** (`temporal/observation_scheduler.py`, SQLite): durable jobs with idempotency keys,
  deterministic jitter, leases with crash recovery, retries with exponential backoff, a "missing"
  observation after the last failure (censoring, not failure), per-source token budgets in the
  source's clock, event-driven coalesced checks, an adaptive policy that skips low-information
  checks and adds checks near uncertain predicted event times, cancellation and retention purging.
* **Rewards** (`temporal/reward_maturation.py`): `event_by_horizon` rewards mature when the event is
  observed or an observation after the horizon shows it did not happen; count streams become
  increments with explicit revision records; semi-Markov discounting by elapsed time.
* **Progressive learning** (`training/temporal_world_model_trainer.py`): the learner reads actions
  whose labels changed since it last consumed them, mixes them with historical replay and trains —
  a 1-minute outcome is used long before the 7-day outcome matures.

## 4. SocialWebLab (`socialweblab/`)

A self-hosted, persistent, multi-user web application with three apps on one data model:

* **forum** (questions/discussions in topic spaces, answers, comments, votes, accepted answers,
  moderation), **helpdesk** (tickets, staff answers in business hours, customer confirmation),
  **code review** (patches, an asynchronous CI job that posts a result, reviewer approval, merge);
* SQLite database (users, follows, spaces, items + versions, responses, reactions, views,
  notifications, sessions, job queue, analytics events); service layer with authorisation checks;
* background jobs — simulated users' browsing sessions (diurnal activity, feed ranking by recency,
  votes and followed authors), auto-moderation, moderator reviews, CI runs — executed by a worker
  thread in real time or by `advance(seconds)` under a deterministic virtual clock (a simulated week
  takes seconds; the clock lives in the database, so snapshots capture it);
* simulated users react only to what users can see: the text (detail, question form, politeness,
  shouting, clickbait, spam, topic match), the space, the author's public follower count and when the
  post reaches them. Content effects are causal by construction; audience and posting time change
  exposure and therefore confound naive content/outcome associations. Authors edit unanswered posts,
  accept helpful answers, sometimes delete posts; all activity is marked simulated;
* `server.py` — JSON API (session cookies / bearer tokens for users, read-only observer API keys,
  token-bucket rate limits in the lab's clock, CSRF origin checks), server-rendered HTML pages with a
  small JS client (asynchronous publishing and search, polling notification badge), and a secret
  control channel (clock, test accounts, observer keys, verification, snapshots) that the browser
  can never reach;
* `browser_env.py` — `SocialWebLabBrowserEnv(PlaywrightEnv)`: real login form, content macros
  (`submit(create_post, …)`, `submit(post_reply, …)`) executed as real UI interactions, verification
  from backend state only; `tasks.py` — browser workflow tasks (upvote via the async search page,
  reply, edit own post, create post) and a scripted teacher; `agent_tasks.py` — the delayed-outcome
  task "get a helpful answer within one day" (choose where to post and write the post; verified
  after 24 simulated hours from privileged answer-helpfulness truth).

`environments/realistic.py` — `RealisticWebEnvironmentAdapter` (`reset(task_id, seed)`, `observe`,
`available_actions`, `step`, `get_observable_events`, `verify_task`, `snapshot`, `restore`, `close`,
screenshots stored by hash) over SocialWebLab (real snapshot/restore), MiniWeb in Chromium (reset
only), BrowserGym/MiniWoB++ (reset by seed), WebArena / VisualWebArena / WorkArena (wrappers that
refuse with the exact missing requirement: the benchmarks' Docker-hosted sites or a ServiceNow
instance are not available in this sandbox).

## 5. Observing other actors (`sensors/`, `data/observed_actor_action.py`)

* `ObservedActorAction` — typed, versioned; unknown goals stay unknown, inferred actions carry a
  confidence, incomplete context is flagged, `missing` lists what was not observable.
* `SourceDeclaration` per source: access method, operator, allowed / prohibited data categories,
  training authorisation and its reference, rate limits, retention, redistribution, event types,
  polling interval, backfill, deletion policy, PII redaction, time base. `permissions.check_source`
  refuses sources without training authorisation, external sources without an authorisation
  reference, and declarations that allow prohibited categories; `check_record` refuses individual
  records outside the allowed categories. A live-web placeholder (e.g. for Reddit) refuses to run
  without explicit platform authorisation — public visibility is not treated as permission to train.
* Adapters: SocialWebLab public feed + public item pages (over HTTP with an observer key, or a direct
  read of the same public views), licensed JSONL datasets. Actor identities are pseudonymised with a
  keyed hash; reply texts are not stored (only their kind); PII is redacted.
* `ObservationSensorNetwork` — runs independently of the acting agent: polls sources (in parallel),
  normalises, checks permissions, deduplicates (same source, across sources, and our own actions),
  records each new action and **forecasts it immediately** through a bounded, batched
  `ForecastService` (one model copy), schedules follow-ups, attaches follow-up events, applies
  deletion policies, runs due observations, optionally records updated conditional forecasts, closes
  windows; revocation purges a source's data and cancels its jobs; retention purges old data.
  Our own actions enter through the same ledger (`source_type="self"`, forecast recorded before
  execution) and are followed through the same public interface.
* `information_sampling.py` — uniform / stratified / novelty / uncertainty / learnable-error
  selection; raw surprise alone is never used.

## 6. Shared learning, planning, RL

* **World model** — trained on self and third-party transitions in one model; source labels are part
  of the input, so the model can learn shared dynamics and source differences.
* **Imitation** — only from demonstrations with enough context (here: the scripted author). An
  observed third-party post with many replies is *not* imitated automatically.
* **Planner** (`planning/temporal_planner.py`) — samples K complete actions (where + what) from the
  policy, scores each at the goal's horizon (`P(answered within 1 day) − λ·uncertainty`), executes the
  best or declines; a count objective at a short horizon (views at 1 h) is available to study the
  proxy trap.
* **Delayed-reward RL** (`training/delayed_reward_trainer.py`) — matured, environment-verified rewards
  are applied to the stored action (exact token ids, behaviour log-probs, checkpoint) with the PPO
  clipped ratio against the behaviour policy, a staleness limit, group advantages and an exact KL to
  the imitation policy. Only our own policy-sampled actions are eligible; third-party outcomes are
  rejected as policy rewards; pending rewards wait, censored ones are dropped.
* **Continual cycle** (`training/continual_learning_orchestrator.py`) — act → observe (clock advances,
  sensors run) → progressive world-model update → policy update from matured rewards → held-out and
  old-distribution evaluation → promotion or rejection (with weight restore) in the checkpoint
  registry.

## 7. Tests

`tests/temporal` (hazard math, censoring, pending ≠ failure, real weight updates + reload, ledger
immutability and governance, labels and no-leakage, rewards, durable scheduler), `tests/sensors`
(test 9: unauthorised sources, prohibited categories, PII, rate limits, deletion, retention,
revocation, dedup, own actions; sampling), `tests/socialweblab` (API, auth, rate limits, control
channel, real Chromium workflows), `tests/environments` (test 6: real-browser workflow with async
updates, persistent state, independent verification, snapshot/restore; WebArena/WorkArena refusal),
`tests/end_to_end` (tests 1–5, 8, 10).

## 8. Running it

```bash
python -m agentrs observe-lab --out runs/obs --days 3 --scheduling event
```

```bash
python -m agentrs lab-serve --clock virtual --speedup 60
```

```bash
python -m agentrs temporal-train --ledger runs/obs/net/ledger.sqlite --train-split train --eval-split train --out runs/twm
```

Experiments: `scripts/temporal/` (`build_corpus.py`, `exp_a_*.py` … `exp_f_*.py`, `throughput.py`).
Compute profiles: `configs/temporal/{local_dev,single_gpu,multi_gpu}.yaml` (only `local_dev` executed).

## 9. Results

All numbers below were measured in this sandbox: 4 CPU cores, no GPU, Pythia-14M (14 M parameters)
for every learned model, SocialWebLab with 80 simulated users under the virtual clock. Raw files:
`results/temporal/*.json`; tables are regenerated by `python scripts/temporal/summarize.py` and
`python scripts/temporal/analyze_f.py`. Brier scores are averaged over 5 events (first response,
answered, positive feedback, resolved, removed) × 5 horizons (5 min, 1 h, 6 h, 1 d, 3 d); AUROC is
averaged over (event, horizon) cells where both classes occur (a lookup table scores 0.5). Only
outcomes known at a horizon are scored; pending outcomes are skipped.

**Corpora.** Three lab seeds × 14 simulated days (28–31 s of compute each): ≈1,080 observed actions
per corpus (≈950 by simulated users, 132 by our agent's test accounts), ≈7,100 scheduled follow-up
observations and ≈6,000 follow-up events. Splits by time — train: actions in days 0–9; test: days 9–11
(outcomes followed to day 14) — and by environment — two forum spaces and the whole code-review app
(`submit_patch` actions) never appear in training (`test_env`).

### A. Single-step vs temporal world model (3 corpora)

| model | test Brier ↓ | test AUROC ↑ | Brier at durations between observed horizons | unseen env Brier | unseen env AUROC |
|---|---|---|---|---|---|
| Kaplan–Meier lookup table (censoring-aware) | 0.155 ± 0.006 | 0.500 | 0.175 ± 0.004 | **0.158** ± 0.008 | 0.500 |
| single-step (next observation only; persistence after 5 min) | 0.311 ± 0.019 | 0.441 | 0.322 ± 0.027 | 0.381 | 0.510 |
| one head per trained horizon, no time representation | 0.168 ± 0.006 | 0.672 | 0.189 ± 0.010 | 0.171 | 0.634 |
| **temporal** (continuous time; 600 steps, lr 2e-4) | **0.153** ± 0.003 | **0.681** | **0.169** ± 0.006 | 0.167 | **0.648** |
| temporal, generated content hidden | 0.153 ± 0.009 | 0.652 | 0.168 ± 0.007 | 0.168 | 0.558 |
| temporal, validation-tuned (400 steps, lr 1e-4, wd 0.1) | 0.159 ± 0.002 | 0.661 | 0.172 ± 0.004 | 0.169 | 0.602 |

* A next-observation model cannot answer questions about later consequences (Brier ≈ 2× worse).
* Representing time continuously beats one head per horizon, especially between horizons.
* The temporal model **ranks** outcomes (AUROC 0.66–0.68) but is **not better calibrated than the
  lookup table** (Brier equal on test, worse on unseen environments). Reading the content adds
  ranking power on unseen environments (0.648 vs 0.558) but not on in-distribution Brier.
* Overfitting is real at this scale: validation curves inside the training window bottom out at
  300–400 steps; the tuned setting did not improve test scores.

### B. Observation scheduling (one lab run, 4 networks observing it concurrently, 10 simulated days)

| scheduling | observation requests | event labels exact / interval / right-censored | median detection latency | test Brier (marginal 0.169) |
|---|---|---|---|---|
| fixed (feed + polls at all 8 profile horizons) | 4,275 | 946 / 0 / 563 | 0.12 h | 0.143 |
| poll only (no event feed) | 4,275 | 599 / 341 / 569 | 0.52 h | 0.142 |
| event (feed + 2 polls + window end) | 4,727 | 946 / 0 / 563 | 0.12 h | 0.145 |
| adaptive (feed + forecast-driven polls) | 6,900 | 946 / 0 / 563 | 0.12 h | 0.140 |

BFOLLOWUP

* The event feed gives exact event times (no interval censoring) and 4× lower detection latency.
* **Negative result:** event-driven and adaptive scheduling cost *more* requests than fixed polling,
  because every reply or vote triggered a check (coalesced only per minute). Prediction quality was
  the same within noise (0.140–0.145; re-running the same configuration gives ±0.002).

### C. Self experience vs self + third-party observations (3 corpora, same steps)

| training data (examples) | our later posts: Brier / AUROC | third parties' later posts | unseen spaces + unseen app |
|---|---|---|---|
| lookup table, own posts | **0.105** / 0.500 | 0.188 / 0.500 | 0.171 / 0.500 |
| lookup table, all | 0.116 / 0.500 | 0.165 / 0.500 | **0.158** / 0.500 |
| own posts only (≈216) | 0.117 / 0.646 | 0.215 / 0.513 | 0.193 / 0.527 |
| own + third-party, same size (≈432) | 0.119 / **0.673** | 0.175 / 0.644 | 0.172 / 0.594 |
| third-party only (≈965) | 0.172 / 0.619 | 0.166 / 0.662 | 0.164 / **0.622** |
| own + all third-party (≈1,181) | 0.135 / 0.649 | **0.162** / **0.675** | 0.164 / **0.622** |

* Third-party observations improve generalisation to unseen environments and action types (AUROC
  0.53 → 0.62, Brier 0.193 → 0.164) — the observed-actions data teach dynamics our own few posts
  cannot.
* For our own posts, adding much more third-party data hurts calibration (0.117 → 0.135): a
  distribution shift between our accounts' posts and the population's. A same-size mix keeps
  calibration and improves ranking.
* No learned model beats the lookup table's Brier score here.

### D. Volume vs diversity at a fixed budget (2 corpora, 400 steps each)

| selection | examples | strata | test Brier | test AUROC | unseen env Brier |
|---|---|---|---|---|---|
| lookup table | — | — | 0.156 | 0.500 | 0.164 |
| repetitive (two most active strata) | 166 | 2 | 0.193 | 0.642 | 0.235 |
| uniform | 150 | 6 | 0.186 | 0.646 | 0.205 |
| stratified | 150 | 6 | 0.192 | 0.609 | **0.188** |
| novelty (farthest-point) | 150 | 6 | **0.180** | 0.642 | 0.243 |
| uncertainty (+40 seed examples) | 172 | 6 | 0.190 | 0.636 | 0.217 |
| all third-party data | 389 | 6 | 0.165 | 0.667 | 0.199 |

* Diversity matters most for transfer: repetitive data is worst on unseen environments (0.235);
  stratified selection is best there (0.188) at 150 examples. Novelty selection helps in-distribution
  but not on unseen environments; uncertainty selection is in between. Differences between selection
  strategies are within ≈0.01–0.03 with 2 corpora — suggestive, not conclusive. More data (all)
  still helps most in-distribution.

### E. Observational prediction vs randomized interventions (2 lab seeds)

Effects on P(answered within 24 h); brackets are bootstrap 95 % intervals over posts.

| estimate | content (detailed + polite) | audience (30 vs 0 followers) | timing (19:00 vs 04:00 UTC) |
|---|---|---|---|
| randomized, seed 20 (124 posts) | **0.28** [0.11, 0.45] | 0.14 [−0.03, 0.31] | 0.02 [−0.16, 0.20] |
| randomized, seed 21 (118 posts) | **0.20** [0.03, 0.37] | **0.30** [0.13, 0.45] | **0.36** [0.20, 0.51] |
| naive observational (third-party posts) | 0.31 / 0.28 | 0.14 / 0.19 (≥10 followers) | — |
| observational, follower-stratified | 0.27 / 0.24 | — | — |
| world model trained on observations (covariates in context) | 0.02 / 0.13 | 0.00 / 0.00 | 0.00 / 0.00 |
| world model without audience / time covariates | 0.27 / 0.11 | (not represented) | (not represented) |

* The lab has real confounding (detail correlates with follower count, r ≈ 0.2): the naive
  observational content estimate is 0.03–0.08 higher than the stratified one, and above the
  randomized estimate in seed 21.
* **The world model does not recover intervention effects.** It ignores the audience and timing
  covariates almost entirely (implied effects ≈ 0 where the randomized effects were up to 0.30–0.36)
  and its implied content effect varies 0.02–0.27 between runs. Its outputs are observational
  forecasts and are labelled as such; they must not be read as causal effects. (Experiment E used the
  untuned 500-step setting.)

### F. Verified task completion with delayed outcomes (3 seeds × 72 held-out tasks)

Task: get a helpful answer within one simulated day (verified by the environment 24 h after posting).
Policy: Pythia-14M; imitation of a mixed-quality scripted author (400 demos), then 16 rounds of
delayed-reward RL (384 posts; rewards matured 24 h after each post and were applied with the clipped
ratio to the stored behaviour log-probs of a 1–2 versions older policy). Every arm is evaluated in an
identical copy of the lab (snapshot restore), posting on the same schedule.

| agent | success (95 % CI) | paired vs RL policy |
|---|---|---|
| imitation only | 0.148 [0.102, 0.194] | −0.111 [−0.181, −0.046] |
| + delayed-reward RL (one policy sample) | 0.259 [0.204, 0.319] | — |
| planner, immediate (5-min) world model | 0.255 [0.199, 0.315] | −0.005 [−0.079, +0.074] |
| planner, temporal world model, own posts only | 0.218 [0.162, 0.273] | −0.042 [−0.116, +0.032] |
| **planner, temporal world model, own + third-party posts** | **0.315** [0.255, 0.380] | +0.056 [−0.023, +0.134] |
| planner, same model, objective = views after 1 h (proxy) | 0.310 [0.250, 0.370] | +0.051 [−0.028, +0.130] |

* **Delayed-reward RL works:** +11 points over imitation (significant), learned mainly by including
  concrete details (rate of posts with concrete details 0.51 → 0.86; the temporal-model planner picks them 0.95 of the time).
* **Third-party observations help planning:** the planner with own + third-party data beats the
  planner with own data only by **+9.7 points [+1.9, +17.1]** (paired, 216 tasks).
* Against the plain RL policy, model-based planning gives +5.6 points but the interval includes 0.
* The proxy objective (views after 1 h) chose nearly the same posts, so the proxy trap was not
  exercised here: after RL the policy rarely writes clickbait, and views and answers were correlated.
* No arm learned to post in the two unseen spaces (right-space rate ≈ 0 there): no transfer of the
  *where* decision.
* Compute per seed (1 CPU thread): 26 min total — imitation 9 min, RL 2.5 min, three world models
  10 min, evaluation 4.7 min; 384 own posts, ≈730 third-party actions observed, ≈1,800 temporal
  training examples.

### Throughput (4 CPU cores, Pythia-14M)

THROUGHPUT

### What this shows and what it does not

* Shown, at small scale: a time-conditioned world model with censoring-aware learning predicts
  delayed outcomes far better than a next-observation model and better than per-horizon heads; it is
  trained on real observed outcomes (ours and other actors'), learns progressively as outcomes
  mature, and its forecasts are stored before outcomes are known. Third-party observations improve
  generalisation to unseen environments and improve a planner's verified task success over a planner
  that only knows its own experience. Delayed-reward RL with off-policy correction improves verified
  task success.
* Not shown: better calibration than a lookup table; any advantage of event-driven / adaptive
  scheduling (it cost more here); recovery of causal (intervention) effects from observational data;
  a significant advantage of model-based planning over the RL policy; anything about large models,
  real websites or real users.

