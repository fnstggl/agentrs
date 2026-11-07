# Temporal world models, SocialWebLab and observational learning — compatibility map

Audit of the existing system (what is reused, what is extended, what is new).

| existing component | file | how the new subsystems use it |
|---|---|---|
| LM backbone (HF causal LM, pooled states, exact token log-probs) | `models/backbone.py` | shared trunk of the temporal world model; one pooled forward per action serves every horizon |
| Ensemble heads + bootstrap masks | `models/world_model.py` | reused for epistemic uncertainty of every temporal head |
| Content-conditioned world model (complete action incl. content) | `content/world_model.py` | the temporal model reads the same `ACTION: submit(op, "content")` text; immediate heads stay as they are |
| `SemanticAction` (content, token ids, behaviour log-probs, checkpoint) | `content/semantic_action.py` | the action record of self actions and the `action` field of observed third-party actions |
| Content policy, content RL trainer (token-level clipped surrogate, stored behaviour log-probs) | `content/policy.py`, `content/training.py` | delayed-reward RL reuses it; matured rewards arrive later and the ratio corrects for the policy having moved |
| Content planner (drafts → WM → score → decline) | `content/planner.py` | the temporal planner scores drafts at a chosen objective and horizon |
| Playwright harness, DOM extraction, URL allowlist, control-channel pattern | `environments/web/*` | SocialWebLab's browser environment subclasses `PlaywrightEnv`; privileged control channel as in `server.py` |
| BrowserGym adapter (MiniWoB++) | `environments/browsergym_adapter.py` | wrapped by `RealisticWebEnvironmentAdapter` |
| Guarded environment, permission policy, approval callback | `security/permissions.py` | content operations in SocialWebLab go through the same allowlist / rate limits |
| GradientTrainer, checkpoints with optimizer state, param snapshots | `training/common.py` | all new trainers |
| Checkpoint registry, promotion gate, rollback | `checkpoints/registry.py` | the continual orchestrator promotes temporal world-model checkpoints |
| Experience schema (Parquet, provenance, imagined vs real) | `data/schemas.py`, `data/storage.py` | self transitions keep this schema; temporal outcomes live in the delayed-outcome ledger, keyed by action id |

New code:

* `temporal/` — time encoding, horizon config, three clocks (environment / observation / training) and
  real vs virtual clocks, the piecewise-hazard cure model for time-to-event with interval and right
  censoring and competing risks, the `TemporalWorldModel` (shared trunk, time-conditioned ensemble
  heads), losses, the SQLite `DelayedOutcomeLedger` (insert-only forecasts), the durable
  `TemporalObservationScheduler`, reward maturation (increments, revisions, semi-Markov discounting),
  temporal evaluation.
* `socialweblab/` — a self-hosted multi-user web application (forum, help desk, code review) with a
  SQLite database, HTTP API + HTML/JS frontend, background job queue driven by a real or virtual
  clock, simulated users acting through the application's request handlers, public event feed,
  privileged verifier channel.
* `sensors/` — source declarations and registry, permission checks, deduplication, normalisation to
  `ObservedActorAction`, the `ObservationSensorNetwork`, information-value sampling.
* `training/temporal_world_model_trainer.py`, `training/delayed_reward_trainer.py`,
  `training/continual_learning_orchestrator.py`.
* `environments/realistic.py` — `RealisticWebEnvironmentAdapter` over SocialWebLab (real snapshot /
  restore), MiniWeb-in-Chromium, BrowserGym/MiniWoB++, and WebArena / WorkArena (adapter only).

External frameworks (checked 2026-11-07): BrowserGym 0.14.3 (Apache-2.0, installed, MiniWoB++ runs
here); browsergym-webarena 0.14.3 and webarena-verified 1.2.3 (Apache-2.0 / unspecified) need the
WebArena sites as Docker containers — no Docker daemon and ≈17 GB of disk here, so they cannot run in
this sandbox; browsergym-workarena 0.5.3 needs a ServiceNow instance (external account, not
authorised); AgentLab 0.4.2 (Apache-2.0) is not needed for the comparisons run here; "OpenWebRL" is not
on PyPI and GitHub is not reachable from this sandbox, so it was not inspected.
