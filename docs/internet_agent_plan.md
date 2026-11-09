# Internet-agent harness — implementation map

Audit of the existing system and how the execution harness (`src/agentrs/harness/`) plugs into it.

| existing component | file | role in the harness |
|---|---|---|
| Content-generating LM policy (op + content, exact token ids and behaviour log-probs) | `content/policy.py` | `LocalPolicyBackend` of the inference gateway; its outputs carry what delayed-reward RL needs |
| Temporal world model, delayed-outcome ledger, durable observation scheduler | `temporal/` | every executed real action is recorded in the ledger and followed up on its horizon profile |
| Sensor network, source declarations, permission checks | `sensors/` | third-party events from connectors / webhooks enter through `ConnectorSource` adapters |
| Delayed-reward RL, temporal world-model trainer, continual orchestrator, checkpoint registry | `training/`, `checkpoints/` | the learner process; candidate checkpoints, held-out evaluation, promotion, rollback |
| SocialWebLab (self-hosted multi-user app, Chromium env, delayed-outcome tasks) | `socialweblab/` | the default sandbox (tier 1) and the Reddit-like training environment |
| Playwright harness (DOM extraction, URL allowlist) | `environments/web/` | the isolated browser runtime and the generic website connector build on it |
| Guarded environment / approval callback pattern | `security/permissions.py` | generalised into the tiered permission engine and approval queue |

New (this phase):

| service | module | storage |
|---|---|---|
| control-plane data model + migrations | `harness/db.py`, `harness/migrations/` | PostgreSQL (SQLite for tests) |
| secret store / credential references | `harness/secrets.py` | env / encrypted file / cloud secret manager |
| agent identities, platform account registry | `harness/identity.py` | PostgreSQL |
| account provisioning orchestrator, OAuth manager | `harness/provisioning.py`, `harness/oauth.py` | PostgreSQL + secret store |
| provider capability registry (all disabled by default) | `harness/capabilities.py`, `config/platforms.yaml` | config |
| controlled egress (allowlists, SSRF / metadata protection, limits, logging) | `harness/net.py` | — |
| connector framework + connectors (Bluesky, Mastodon, GitHub, Discord, X, Reddit, Instagram, LinkedIn, generic web, SocialWebLab) and local provider simulators | `harness/connectors/`, `harness/simulators/` | — |
| permission + approval engine (tiers 0–4, quotas, budgets, duplicate detection, emergency stop) | `harness/policy.py` | PostgreSQL + Redis |
| tool gateway | `harness/gateway.py` | audit log in PostgreSQL |
| isolated browser workers | `harness/browser_pool.py` | encrypted session state in object storage |
| email service | `harness/email_service.py` | PostgreSQL + mailbox provider |
| task registry, practice-task generator / solver / verifier separation | `harness/tasks.py` | PostgreSQL |
| model inference gateway (batched, shared) | `harness/inference.py` | — |
| agent runtime | `harness/runtime.py` | — |
| training data pipeline / learner | `harness/learning.py` | ledger + checkpoints |
| control API + operator dashboard + metrics + emergency stop | `harness/control_api.py`, `harness/monitoring.py` | — |
| deployment | `deploy/` (Docker Compose, Dockerfiles, Kubernetes manifests) | — |

Substitutions (documented, not hidden): the durable workflow engine is the repository's own
SQL job queue + the temporal observation scheduler (leases, retries, idempotency, restart-safe)
instead of a Temporal.io server; the temporal outcome ledger stays SQLite (on a persistent volume).
