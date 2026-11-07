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

RESULTS_PLACEHOLDER
