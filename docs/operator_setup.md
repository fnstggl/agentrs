# Operator setup guide

How an operator connects real infrastructure and real platform identities to the harness. Nothing
in this guide is done by an agent; the steps marked **(you)** happen on the provider's official
interface. Never paste passwords, verification codes, private keys or API secrets into an AI
conversation, a ticket or a git commit — they go into your secret manager, and the harness only
stores `secret://…` references to them.

## 0. What is automated and what is not

| step | automated by the harness | needs you |
|---|---|---|
| deciding which permissions a task needs; finding an existing authorised account | yes | — |
| creating accounts on external platforms (sign-up, email / phone / identity verification, CAPTCHAs) | **never** | yes, on the official sign-up page |
| developer / app approvals (X, Meta, Reddit, LinkedIn developer programmes; GitHub App creation) | never | yes |
| registering an OAuth client where the provider offers an API for it (e.g. Mastodon `/api/v1/apps`) | yes | you approve the provisioning request |
| the OAuth consent click | never | yes, in your browser |
| token exchange, storage, refresh, revocation | yes | — |
| verifying a connected credential with a harmless read | yes | — |
| SocialWebLab sandbox test accounts (our own application) | yes | — |
| MFA / SMS / voice verification prompts | never | yes; the task pauses until you confirm |

A denied provisioning request is final for that agent, provider and capability set; the agent
cannot re-open it.

## 1. Domain and DNS

1. Choose a domain for the deployment, e.g. `agents.example.org`.
2. Create `A`/`AAAA` (or `CNAME`) records for the control plane, e.g. `control.agents.example.org`,
   pointing at your load balancer / ingress. Terminate TLS there (managed certificate).
3. Set `AGENTRS_PUBLIC_BASE_URL=https://control.agents.example.org`. OAuth callbacks are
   `${AGENTRS_PUBLIC_BASE_URL}/oauth/callback`; webhooks are `${AGENTRS_PUBLIC_BASE_URL}/webhooks/<provider>`.
4. For email: MX records from your mail provider, plus SPF (`v=spf1 include:<provider> -all`), DKIM
   (the provider's selector record) and a DMARC policy for the sending domain.

## 2. Email provider and mailbox

1. **(you)** Create a mailbox such as `agent@agents.example.org` with your provider (Google
   Workspace, Microsoft 365, Fastmail, …). The sender name should make clear it is an automated
   agent of your organisation.
2. Prefer OAuth2 (XOAUTH2) or an app password over the account password. Store it in the secret
   manager as `AGENTRS_SECRET_SMTP_PASSWORD` (and the user name as `AGENTRS_SECRET_SMTP_USERNAME`).
3. Register the account (`provider: email`, `external_id: agent@agents.example.org`) with
   `credential_refs: {"username": "secret://env/smtp_username", "password": "secret://env/smtp_password"}`
   and set SMTP/IMAP hosts in the account's `connection` settings.
4. Put your own domains in `EmailConnector.internal_domains` (transactional mail). Mail to anyone
   else is `send_external` (tier 3) and goes to the approval queue by default.
5. Verification mail (codes, "verify your email" links) arriving in the mailbox is shown to you as a
   notification; the harness does not follow those links or read codes to complete verification.

## 3. OAuth callback URLs

For every OAuth provider you enable, register exactly this redirect URI in the provider's developer
console: `https://control.agents.example.org/oauth/callback`. Start the flow from the dashboard
(the harness creates the state + PKCE challenge and shows you the authorisation link); you click
"Authorize" on the provider's page; the callback stores tokens in the runtime secret file
(encrypted with `AGENTRS_SECRET_KEY`).

## 4. Developer applications and credentials (per provider)

Every provider ships **disabled** in `config/platforms.yaml`, with every capability off and zero
rate limits. To enable one: read the provider's current developer terms and automation rules, get
the approvals they require, then set `enabled: true`, the capabilities you were approved for, real
`limits` from the provider's documentation, and `authorization_ref` (your approval / agreement id).
The gateway refuses any provider that is enabled without non-zero rate limits.

| provider | you create | secret names (secret manager) | account `connection` settings |
|---|---|---|---|
| Bluesky | the account (official sign-up), the 'bot' self-label, an app password | `AGENTRS_SECRET_BLUESKY_APP_PASSWORD` | `handle` |
| Mastodon | the account on an instance whose rules allow bots ("automated account" ticked) | OAuth tokens are obtained by the harness; client secret stored automatically | `username`, `base_url` (instance) |
| GitHub | a GitHub App with only the repository permissions you approve, installed on specific repositories | `AGENTRS_SECRET_GITHUB_APP_KEY` (PEM), `AGENTRS_SECRET_GITHUB_WEBHOOK_SECRET` | `app_id`, `installation_id`, `repositories` |
| Discord | an application + bot user, added only to servers whose admins agree | `AGENTRS_SECRET_DISCORD_BOT_TOKEN`; public key for interactions | `channel_ids`, `public_key` |
| X | approved developer access for the stated use; OAuth 2.0 user-context app | OAuth tokens via the harness; `AGENTRS_SECRET_X_CLIENT_SECRET` if confidential | `user_id` |
| Instagram | a professional (business / creator) account linked to an approved Meta app | `AGENTRS_SECRET_INSTAGRAM_ACCESS_TOKEN` | `ig_user_id` |
| Reddit | an operator-created account, approved API access, **explicit permission for any AI-training use** | OAuth tokens via the harness | — (`training_data_authorized` must be set explicitly in the config) |
| LinkedIn | specifically approved API products only | OAuth tokens via the harness | — |
| generic websites | a per-site permission profile in `config/permissions.yaml` (`web_sites`) confirming automation of that activity is allowed | site login is done by you once; the encrypted browser session is reused | — |

Register the account through the dashboard (provisioning request → "connect credentials" with
`secret://env/<name>` references); the harness then runs a non-destructive check and only then marks
the account active.

## 5. API tokens and account identifiers

Record for every account: provider, external identifier (handle, app/installation id, mailbox),
account type (bot, app installation, organisation mailbox), owner organisation, responsible
operator (a real person), bot-disclosure state, approved scopes, training-data permission, which
agents may use it, and when it must be re-authorised. One account may serve many agent workers when
the provider's terms allow it; register an additional account only for a genuinely distinct
approved purpose.

## 6. Approved permission scopes

Grant the smallest scopes that cover the tasks. Then align `config/agents.yaml` (which agent may
hold which `<provider>:<capability>`, and its maximum tier) and `config/permissions.yaml`
(approval tiers, operations that always need approval, forbidden operations, quotas).

## 7. Cloud infrastructure credentials

Nothing billable is provisioned by the harness or by the AI. To deploy `deploy/k8s/`:
a Kubernetes cluster, a managed PostgreSQL 16, a managed Redis 7, an S3 bucket, a NAT gateway with a
static egress IP for the worker node pool (the network policy only allows workers out through it),
and the External Secrets operator wired to your secret manager. Use a deployment identity with
least privilege; keep its credentials in your CI system, not in the repository.

## 8. GPU inference endpoint

Either deploy `deploy/k8s/gpu-inference.yaml` (vLLM, OpenAI-compatible) or point
`AGENTRS_INFERENCE_URL` at an existing endpoint and store its API key as
`AGENTRS_SECRET_INFERENCE_API_KEY`. Agents never load model weights; learners run as separate jobs
and only promoted checkpoints are served.

## 9. Database and object storage

`AGENTRS_DATABASE_URL=postgresql+psycopg://<user>@<host>:5432/<db>` (password via the secret
manager / IAM auth), `AGENTRS_REDIS_URL`, `AGENTRS_S3_BUCKET` (+ endpoint for MinIO). Run
`python -m agentrs.harness migrate` once per release (the Kubernetes manifest does this in an init
container).

## 10. Human verification and MFA

Keep MFA on every operator-owned account. Put recovery codes and verification phone numbers under
the organisation's control (not in the harness). When a provider asks for SMS / voice / MFA /
identity confirmation, the agent's task pauses with a human-verification approval in the dashboard;
complete it on the provider's site and mark the step done — the task then resumes. The harness has
no disposable-number rotation, CAPTCHA solving or automated recovery, by design.

## 11. Platform-specific developer approvals

Before enabling a provider, record in `authorization_ref` the approval that covers: automated
posting (if any), AI / bot disclosure, rate limits you will respect, and — separately — whether
data from the platform may be used to train models (`training_data_authorized`). Public visibility
of content is not treated as permission to train on it.

## 12. Running locally

```bash
cp .env.example .env
```

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Dashboard: `http://127.0.0.1:8000/login` (operator token from `.env`). Without Docker, the same
services run directly: `python -m agentrs.harness migrate|control|inference|lab|outcome-worker`.
