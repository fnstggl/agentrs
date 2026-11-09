# Live observational learning (real posts, real time, persistent weights)

`python -m agentrs.live run` is an always-on learner that watches **new** public posts as they are
published, predicts what will happen to each one, checks what actually happened at real time
intervals, and updates the world model's weights from the difference. It never posts, never logs
in, and never reads old posts (no backtesting: every post is forecast within seconds to minutes of
being published, before any outcome exists).

## What runs

```
Hacker News API ─┐                       ┌─► forecast (current weights)  ┐ stored immutably,
Bluesky Jetstream┼─► new post ─► ledger ─┤                               ├ before any outcome
(sampled)        ┘                       └─► forecast (baseline)         ┘
                                  │
          follow-up jobs at +10 min, 1 h, 6 h, 1 d, 3 d (durable queue, real clock)
                                  │
       public state (likes, replies, reposts/quotes, score, comments, removed / deleted)
                                  │
     ┌────────────────────────────┴───────────────────────────┐
     ▼                                                        ▼
  learn (every 2 min): censoring-aware labels of the     score: once the outcome at h is final,
  posts whose outcomes changed + replay → gradient       the stored forecasts are scored against
  steps on the same weights; new forecasts use them      it (model vs baseline, over time)
     │
  checkpoint every 10 min + hourly snapshots + on shutdown; restart resumes weights, optimiser,
  ledger, job queue and scores
```

| part | module |
|---|---|
| outcome definition shared by all platforms | `live/spec.py` |
| Hacker News source (official Firebase API) | `live/hackernews.py` |
| Bluesky source (Jetstream WebSocket + public AppView) | `live/bluesky.py` |
| Reddit source (official Data API) — **gated off** | `live/reddit.py` |
| real-time scoring + stratified baseline | `live/prequential.py` |
| service loop, checkpoints, status, goals | `live/service.py`, `python -m agentrs.live` |
| config / deployment | `config/live.yaml`, `deploy/live/docker-compose.yml`, `deploy/live/agentrs-live.service` |
| aggregate results snapshot | `scripts/live/snapshot.py` → `results/live/` |
| tests (mocked APIs + an end-to-end loop with second-scale horizons) | `tests/live/test_live.py` |

It reuses the existing temporal stack unchanged: `ObservationSensorNetwork` (ingestion, dedup,
pseudonymisation, PII redaction, deletion propagation), `DelayedOutcomeLedger` (insert-only
forecasts with information cutoffs), `TemporalObservationScheduler` (durable follow-ups),
`outcome_labels.build_examples` (censoring-aware labels) and `TemporalWorldModelTrainer`
(`progressive_update`).

### What is predicted

For every post and horizon (10 min, 1 h, 6 h, 1 day, 3 days):

* P(first reply / comment), P(at least one like / upvote from someone else), P(≥ 10 and ≥ 100 total
  engagements = likes + replies + reposts/quotes), P(removed);
* expected cumulative likes, replies and reposts/quotes.

The model sees the post text (plus image alt text / link title), the kind of post (text / link /
image / quote / video; Ask / Show / link on HN), link domain, author audience (Bluesky followers,
following, post count, account age; HN karma), language and time of week. Author identities are
replaced by keyed pseudonyms; the key never leaves the deployment.

### Why the scores are honest

* Forecasts are written to the ledger when the post is first seen, with the checkpoint id; the
  ledger refuses a forecast whose information cutoff is after its creation time.
* An outcome at horizon h is scored only after the follow-up at h has actually been made, so
  quick positives are not counted before the matching negatives exist.
* The baseline is not a straw man: per-stratum empirical rates (post kind × audience bucket,
  e.g. "Bluesky image posts by authors with 1k–10k followers"), shrunk to the kind and global
  rates, refitted only from outcomes that were already known when each baseline forecast was made.
* Learning and scoring use the same posts, but every post is scored with the forecast made
  *before* the weights ever saw its outcome (prequential evaluation).
* Deleted posts are redacted and leave scoring (a small survivorship effect, reported as
  `deletions_propagated`).

## What you need

| source | status | what is needed |
|---|---|---|
| Hacker News | **running** | nothing: official public API, no key. Confirm Y Combinator's terms allow your use for model training (the declaration records this as an operator decision). |
| Bluesky | **running** | nothing: public Jetstream + AppView endpoints, no account. Confirm Bluesky's developer guidelines for training use; the source skips authors with the `!no-unauthenticated` label and propagates deletions. |
| Reddit | **off** | 1) a Reddit account you own; 2) a registered app and **approved Data API access** for this use (Reddit's developer platform / Data API terms); 3) **Reddit's explicit written permission to use Reddit content to train models** — the Data API terms do not grant it by default and public visibility is not permission; 4) put the client id / secret in your secret manager as `AGENTRS_SECRET_REDDIT_CLIENT_ID` / `AGENTRS_SECRET_REDDIT_CLIENT_SECRET` (never in chat or git); 5) set `AGENTRS_REDDIT_AUTHORIZATION_REF`, `AGENTRS_REDDIT_TRAINING_AUTHORIZED=true`, `AGENTRS_REDDIT_USER_AGENT`, `AGENTRS_REDDIT_SUBREDDITS`, and add `reddit` to `sources`. The adapter (`/r/<subs>/new` + `/api/info`, 100 posts per follow-up request) is written and tested against a mock of the documented API; it has not been run against Reddit. |
| X, Threads, Instagram, LinkedIn, TikTok, YouTube | not built | each needs its own paid / approved API tier and training-use terms; same adapter interface. |

**No headless-browser scraping.** It would break the platforms' terms, get blocked, and would not
give you the data rights you need to train on the content anyway.

**A machine that stays on.** This build sandbox is temporary; anything running here stops when the
session ends. Run it on your own server, a home machine, or a small VM (CPU only, 4 cores, 2–4 GB
RAM, ~1–3 GB of disk per week at the default sampling rate):

```bash
docker compose -f deploy/live/docker-compose.yml up -d --build
```

```bash
docker compose -f deploy/live/docker-compose.yml exec live python -m agentrs.live status --workdir /data/live
```

or without Docker (systemd unit in `deploy/live/`):

```bash
pip install -e ".[live]"
```

```bash
python -m agentrs.live run --config config/live.yaml --workdir runs/live
```

SIGTERM / Ctrl-C saves the weights; restarting resumes from `checkpoints/latest` with the same
ledger, follow-up queue and scores. Old outcomes keep being collected for posts seen before the
restart (missed horizons are observed late and marked as such). Hourly weight snapshots are kept in
`checkpoints/snapshots/` (48 by default). Raw post text is kept 30 days (Bluesky, Reddit) / 365
days (HN) and then purged; scores and weights are kept.

## Goals ("run until it solves the task")

`targets` in `config/live.yaml` define what "solved" means, e.g. *Brier skill ≥ 0.10 over the
baseline on ≥ 5,000 Bluesky outcomes forecast in the last 24 h*. `status.json` reports progress on
each target; with `stop_when_targets_met: true` the service exits once all are met (otherwise it
keeps learning). The same mechanism is what the acting agent will use later: a task is a target
on verified outcomes, and the loop runs until the target is met or an operator stops it.

## How far this is from DreamerV3 / MuZero

What the repository has, and what those systems also have:

* a **learned world model of consequences** trained only from real outcomes, continuously, with
  persistent weights (this service) — the "model" part of model-based RL;
* forecasts **conditioned on what has already happened** (an updated forecast after 1 h of
  engagement), i.e. a one-step latent transition from the observed state;
* an **acting policy** (content generator) trained by delayed-reward RL on verified outcomes, a
  **planner** that scores candidate actions with the world model before acting, imagination
  rollouts, and the harness that would let it act on real platforms through approvals.

What is missing for a faithful DreamerV3 / MuZero loop:

1. **Self-generated actions in the real environment.** Both systems learn mostly from the
   consequences of *their own* actions. Watching other people's posts teaches the dynamics but
   cannot answer "what happens if *I* post X" (different account, audience and timing — a
   confounded, off-policy signal). The next phase is the agent posting from its own disclosed bot
   account, through the gateway, at low volume.
2. **A multi-step latent dynamics model** (state_t, action_t → state_t+1, reward) that can be
   rolled forward many steps for imagination training (Dreamer) or tree search (MuZero). Today the
   world model predicts outcomes of one action at many horizons; it does not model sequences of
   our actions (post → reply to comments → follow-up post).
3. **Policy and value learned inside imagination** (actor-critic on imagined trajectories) or
   **MCTS over candidate actions with a learned value** (MuZero). The existing planner samples and
   ranks candidates with the world model (best-of-N), which is MuZero-at-depth-1.
4. **Scale.** Pythia-14M is enough to learn audience / format / timing effects; reliably judging
   *content* quality needs a larger trunk (≥ 0.5–1 B parameters, with a GPU), which the code
   supports via `model:` in the config.

Recommended path: (a) let this learner run for weeks so the consequence model is good and
calibrated on real data; (b) use it as the reward model to rank candidate posts (best-of-N, then
MCTS over drafts); (c) post at low volume from one honest, disclosed account with operator approval
and feed those outcomes back with `source_type="self"` (the ledger already separates them); (d)
fine-tune the policy on those outcomes. "Lots of views" must be earned with posts people want —
platform rules forbid fake engagement, vote manipulation, multiple accounts and spam, and the
harness's policy engine enforces quotas, duplicate-content and disclosure checks.

## First live run (this build sandbox)

See `results/live/` for the aggregate snapshot written from the run started in the build sandbox
(no post text or user identifiers are committed).
