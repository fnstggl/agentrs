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

RESULTS_PLACEHOLDER
