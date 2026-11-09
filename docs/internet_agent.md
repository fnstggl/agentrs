# Internet-agent harness

The execution infrastructure that lets many agents act on real internet services through
legitimate, operator-owned identities, observe the consequences, and feed verified experience into
the existing world-model and RL training. Implementation map: [`internet_agent_plan.md`](internet_agent_plan.md);
operator setup: [`operator_setup.md`](operator_setup.md).

Reasoning is general; **execution is authorised**: an agent can think about any task, but the tool
gateway only executes operations that a connector declares, the registry enables, the agent,
account and task are authorised for, the policy allows, and — where required — a human approved.

## Agent loop

```
task ─► requirements (provider:capability) ─► authorised account, or a provisioning request
     ─► observe (gateway reads) ─► decide (shared inference, pinned checkpoint)
     ─► forecast consequences (temporal world model; stored before execution)
     ─► gateway: schema → capability → account → policy / rate limits / content → provider policy
                 → approval tier → execute → audit log
     ─► immediate result ─► ledger + follow-up schedule (webhooks first, polling when necessary)
     ─► verification service (environment-side) ─► experience ─► learner (copy of the weights)
     ─► candidate checkpoint ─► held-out evaluation ─► promotion gate ─► served to new tasks
```

## Services (all in `src/agentrs/harness/`)

| service | module | notes |
|---|---|---|
| control-plane DB + migrations | `db.py`, `migrations/` | SQLAlchemy 2; Alembic (0001 schema, 0002 account connection settings); PostgreSQL in deployments |
| secret store / credential references | `secrets.py` | `secret://env|file|memory/<name>`; encrypted runtime file; redaction; exfiltration check |
| agent identities | `identity.AgentRegistry` | from `config/agents.yaml`; max tier and capabilities set by operators only |
| platform account registry | `identity.PlatformAccountRegistry` | every field of the brief; blocks overlapping accounts unless a distinct purpose is recorded; immediate revocation |
| provisioning orchestrator | `provisioning.py` | REQUESTED → … → ACTIVE / DENIED / FAILED; human setup steps; denials are final; automatic only for declared programmatic flows (our sandbox) |
| OAuth manager | `oauth.py` | authorization code + PKCE S256, single-use state, refresh, tokens only as secret refs |
| provider capability registry | `capabilities.py`, `config/platforms.yaml` | all providers disabled; fail closed |
| controlled egress | `net.py` | allowlists, per-provider domains, DNS + private/metadata blocking, timeouts, limits, redirect re-checks, egress log, corporate proxy only |
| connector framework | `connectors/base.py` | `capabilities / validate_credentials / observe / execute / subscribe / fetch_outcome / get_rate_limits / revoke`; `provider_policy` hook |
| connectors | `connectors/{bluesky,mastodon,github,discord,x,instagram,reddit,linkedin,generic_web,socialweblab}.py`, `email_service.py` | official APIs; Reddit / LinkedIn gated off by default |
| provider simulators | `simulators/` | local, faithful-to-documentation subsets of each API, responses marked simulated |
| permission + approval engine | `policy.py` | tiers 0–4, budgets, quotas (Redis), duplicate-content, blocked patterns, emergency stop |
| tool gateway | `gateway.py` | the only path to the outside; audit log; approvals; untrusted-content wrapping |
| isolated browser workers | `browser_pool.py` | one Chromium process + context per session; encrypted session state; per-request egress checks; budgets; screenshots / traces |
| email | `email_service.py` | SMTP/IMAP or local server; transactional vs external; threads; bounces; verification mail surfaced to the operator |
| durable jobs | `jobs.py` | leases (`FOR UPDATE SKIP LOCKED`), retries, idempotency |
| delayed outcomes | `outcomes.py` | existing ledger + durable follow-up scheduler; webhook events; async fetchers |
| task registry, practice generator | `tasks.py` | generator / solver / verifier separated |
| verification service | `verification.py` | environment-side verdicts; pending ≠ failure |
| inference gateway | `inference.py` | batched, bounded queue, pinned checkpoints; mock / local policy / OpenAI-compatible (vLLM, SGLang) |
| agent runtime | `runtime.py` | model-agnostic loop; environment adapters |
| learner | `learning.py` | training batches from the ledger; copy-of-weights training; candidates; gated promotion |
| control API + dashboard + metrics | `control_api.py`, `monitoring.py` | operator auth; agents, accounts, approvals, provisioning, learning, infrastructure; pause / revoke / disable / emergency stop; Prometheus; OpenTelemetry |
| processes | `__main__.py` | `migrate`, `control`, `inference`, `lab`, `outcome-worker`, `agent-worker` |
| deployment | `deploy/` | Dockerfile, Compose (+GPU override), Kubernetes (workers, egress NetworkPolicy, GPU inference, learner job) |

## What was tested where

Nothing in this repository has been run against a real external provider: no operator
credentials, accounts or approvals were available in the build sandbox, and no live request to any
platform was made. "Simulator" below means a local implementation of the provider's documented API
written from knowledge of that API (responses marked `X-Agentrs-Simulator: 1`) — not verified
against the live service; known uncertainties are listed in the connector tests' notes.

| required test | where it ran | result |
|---|---|---|
| 1 account lookup + capability checking | SocialWebLab (local, real HTTP) | pass |
| 2 provisioning that needs human setup; denial is final | registry + DB | pass |
| 3 OAuth / test-credential connection | Mastodon simulator (consent page, PKCE, single-use state) | pass |
| 4 browser session isolation | real Chromium, two accounts on SocialWebLab | pass |
| 5 real browser execution on a running local app | real Chromium via gateway → SocialWebLab; independently verified in the app DB | pass |
| 6 authorised API action | Bluesky simulator through the gateway (task pre-authorisation + approval queue) | pass |
| 7 third-party event ingestion | SocialWebLab public feed (sensor network) + signed GitHub webhooks via the control API | pass |
| 8 temporal follow-up after an action | SocialWebLab (minutes → days of virtual time) | pass |
| 9 verified task completion recorded | verification service (lab's privileged truth) → experience, task state, ledger reward | pass |
| 10 interaction data → valid training batch | ledger → `ContentRLStep` (exact token ids / behaviour log-probs) + `TemporalExample` | pass |
| 11 real optimizer step changes weights | learner's training copy; live weights untouched | pass |
| 12 candidate evaluation + promotion | held-out tasks in identical lab copies; registry + gate | pass |
| 13 revocation immediately prevents action | gateway, browser pool, emergency stop | pass |
| 14 provider without authorisation denied | Reddit / X / Bluesky disabled by default | pass |
| 15 invalid / restricted actions rejected before execution | unknown op, bad params, tier, task scope, secret exfiltration, spam patterns | pass |
| 16 multiple agents, shared inference | 4 agents, 1 model copy, batched requests, 1 shared sandbox account | pass |
| 17 restart without losing delayed-observation jobs | rebuilt harness from the same DB / outcome store / secret key | pass |
| 18 full autonomous sandbox practice loop with measured learning | script below (PostgreSQL + Redis + HTTP inference service) | ran; see results |

Also exercised locally: PostgreSQL 16 (migrations 0001→0002 on an existing database,
`FOR UPDATE SKIP LOCKED` job claims), Redis 7 (shared rate-limit windows), S3 API (moto server),
SMTP (aiosmtpd), the control API, inference server and lab as separate processes with health
checks and graceful SIGTERM shutdown, and `docker compose config` validation. **Not run:** the Docker
Compose stack itself (no Docker daemon in the sandbox), the Kubernetes manifests (nothing billable
provisioned), GPU / vLLM inference, Temporal.io (replaced by the SQL job queue and the durable
follow-up scheduler), and every live provider (build phases 11 and 12: no operator-approved
account or credential was available, so neither the read-only nor the write test could be run).

### Sandbox learning experiment (test 18; `scripts/harness/sandbox_learning_experiment.py`)

Setup: SocialWebLab (80 simulated users, virtual clock) behind its HTTP API; control plane on
PostgreSQL 16 + Redis 7; the Pythia-14M content policy served by the inference gateway over HTTP;
4 agents sharing one programmatically provisioned sandbox account; practice task "get a helpful
answer within one day" (success verified by the environment 24 simulated hours later); imitation of
a scripted author (400 demos), then 12 practice cycles (24 tasks each, agents acting in rounds over
simulated time), delayed-reward RL on a training copy after every cycle, held-out comparison and
gated promotion every 4 cycles, and a final comparison on 72 held-out test tasks in identical lab
copies. One seed; 3 CPU threads; 8.5 min wall time (imitation 4.2 min, practice + RL 3.8 min).

| | result |
|---|---|
| final held-out success, initial imitation checkpoint | 0.167 [0.083, 0.264] (n = 72) |
| final held-out success, final promoted checkpoint | 0.210 [0.113, 0.323] (n = 62; 10 posts refused as duplicate content) |
| promotion rounds | cycle 3 rejected (0.03 vs 0.07), cycle 7 promoted (0.11 vs 0.00), cycle 11 promoted (0.25 vs 0.13) |
| gateway decisions | 1,248 (1,181 executed, 67 denied — duplicate content, before the quota fix also provider 429s) |
| inference | 624 requests in 174 batches (≤ 4 per batch), one model copy |
| follow-ups / matured rewards | 1,855 observations, 557 matured rewards, 0 censored |
| outbound requests | 3,037, 0 denied by egress policy |

**Reading:** the complete loop runs — agents act through the gateway on a real application, rewards
mature a day later, the policy's weights are updated, candidates are evaluated on held-out tasks
and promoted or rejected — but this run shows **no measurable learning gain**: the final checkpoint's
held-out success is within the noise of the starting checkpoint (intervals overlap almost entirely),
and the promotion decisions rest on 20–32 tasks each, too few to separate small differences. The
earlier experiment F (same task, larger budget, 3 seeds) measured a significant RL gain (+11 points);
this harness run is a functional end-to-end test, not a better estimate.

Problems found while running it, and fixed: (1) all agents share one account, so the per-account
write quotas (30 / hour, 50 / day, wall clock) blocked practice after the first cycle — correct
behaviour, fixed for the sandbox only with per-provider quota overrides; (2) agents posting with the
lab clock frozen exhausted the lab's own per-session rate limit (HTTP 429, recorded by the gateway as
`rate_limited`) — the experiment now spreads tasks over simulated time; (3) async clients and the
inference batcher were bound to the first event loop — both now rebind per loop; (4) duplicate
content refusals reduce the evaluated sample of a low-diversity policy — reported, not hidden.

