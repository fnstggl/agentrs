# CAM Live authorization layer — what is enforced by what

Workstream EA. This instance is an **unprivileged container**: root inside it has `CAP_SETUID`, `CAP_SETGID`,
`CAP_CHOWN`, `CAP_DAC_OVERRIDE`, `CAP_KILL` and a few others, but no user namespaces, no `CAP_NET_ADMIN`, no
`CAP_SYS_ADMIN`, no `CAP_LINUX_IMMUTABLE`, no cgroup or mount control. Measured here: Landlock ABI 1 is available;
`fs.protected_regular=2`, `protected_symlinks=1`, `protected_hardlinks=1`.

Every statement below is exercised by a test in `tests/harness/auth/` (privileged ones really create the users and
switch uid); the test is named in brackets.

## 1. Enforced by the kernel

| Property | Mechanism | Test |
|---|---|---|
| The learner uid (`camlearn`) and the sandbox uid (`camsbx`) cannot read the vault, the audit log, the authorization state, the grant / audit keys, the policy copy, the verifier's log and key, or the browser profile. Attempts fail with `EACCES`. | separate uids; directories `0700`, files `0600` owned by `camauth` / `camverify` / `cambrowser` | `test_ea_privileged::test_learner_and_sandbox_uids_cannot_read_vault_audit_or_verifier_files` |
| They cannot delete, rename or replace the audit log, the verifier log, the vault directory or the authorization socket. | run directory and `sock/` are **root-owned sticky** directories (`1771` / `1770`); `fs.protected_regular` also refuses `O_CREAT` opens of another uid's file there | same test |
| They cannot read the authorization service's environment or memory. | different uid; the service sets `PR_SET_DUMPABLE=0` | same test |
| A service cannot regain privileges. | uid change drops all capabilities (`CapEff=CapPrm=0`); `no_new_privs` is set before the service code runs (launcher trampoline), so setuid binaries do not elevate | `…::test_launcher_starts_supervises_and_stops_services_under_their_own_uids`, `…::test_service_uids_cannot_regain_privileges` |
| The authorization service can write only to `auth/`, `vault/`, `sock/` and the audit file, whatever a request makes it do. | **Landlock** self-restriction after start-up (`harness/auth/landlock.py`) | `test_ea_launcher::test_authorization_service_confines_its_own_writes_with_landlock` |
| The caller's identity on every RPC is the real peer uid. | `SO_PEERCRED` on the unix socket; the client checks the server's uid the same way | `test_ea_privileged::test_rpc_peer_uid_authorization_matrix` |
| A uid outside the `camlive` group cannot reach the socket at all. | socket `0660`, group `camlive`, inside `sock/` `1770` | same test (`nobody`) |
| Services do not inherit tokens from the launcher's environment (this container has several). | environment built from an allowlist | `test_ea_launcher::test_launch_auth_and_services_then_stop`, privileged launcher test |

**Not used by this workstream:** seccomp. The authorization service makes no use of it; generated code is confined
by the sandbox (workstream EC: uid + rlimits + Landlock + seccomp). The append-only inode flag for the audit log is
not available (`CAP_LINUX_IMMUTABLE` missing): tampering by the *authorization uid itself* is therefore only
detectable (hash chain + MAC + head file, section 3), not prevented.

## 2. Enforced only by the policy path (not by the kernel)

These hold **only if every executor calls the authorization service as the contracts require** (`check_grant`
before execution, `ExecContext.check` → `check_primitive` for every primitive that leaves the process, `record`
afterwards):

* **This machine's control plane.** No mode may reach the instance's own services: the policy denies, in every
  mode, every address of this machine or its link — loopback in every spelling (canonicalised to one host before
  any pattern is matched, so a deny entry for `127.0.0.1:P` also covers `localhost:P`, `localhost.:P`, `[::1]:P`,
  `0.0.0.0:P`, `127.1:P`, `2130706433:P`, `[::ffff:127.0.0.1]:P`), link-local and metadata addresses, private
  networks, the machine's own interface addresses and names `/etc/hosts` maps to them — unless the exact
  `host:port` is a registered local service (policy list, operator registration at run start, or in staging a
  port served only by the authorization service's own process) [`test_ea_fix_policy`]. This is policy-path
  enforcement, like everything in this section: there is no kernel egress filter. DNS is never queried at decision
  time, so a public name whose record points at this machine (DNS rebinding) is caught only if the executor reports
  the address it connected to (`peer_ip`); the browser executor does not yet (change request EA-10).
* **Per-destination network scoping.** Which hosts, paths, accounts and channels an action may reach is decided
  from the typed `ActionSpec` / `PrimitiveOp`. URLs are read as a WHATWG parser (Chromium) reads them — `\\` is a
  path separator, slashes after `scheme:` are skipped, user-info ends at the last `@`, the host is percent-decoded
  and IDNA-mapped before it is recognised as an IP address — and control characters are refused, so the policy and
  the browser agree on the host. Nothing in this container stops a process that already has network
  access from opening a socket to another host: there is no `CAP_NET_ADMIN`, so no netfilter rules, no network
  namespace per service, no egress proxy enforced by routing. The browser uid and the executor uid have ordinary
  outbound network access. Redirects are a concrete case: a permitted URL that redirects to a host outside the
  allowlist is caught only if the browser executor reports each request as a `net.request` primitive.
* **Side-effect class of browser interaction.** Whether a click submits a form is not visible in the typed action;
  it is bounded by ownership of the page's site (tracked from executor-reported navigation) and, precisely, only by
  per-request `net.request` checks.
* **Opaque payloads.** Program text, skill arguments and raw request bodies are scanned for refused targets and
  secret values (including base64 / hex / URL-encoded / reversed forms). Obfuscation that the scan does not know
  (string concatenation, compression, encryption) is not detected; the sandbox's lack of network is what actually
  stops such code.
* **Limits.** Every declared step of an action is charged at authorization (all steps must fit the remaining
  external-action budget and every rate window), and every external-effect primitive an executor reports through
  `check_primitive` is charged too (the declared step it realises pays for it; anything else is charged anew and
  refused once the budget or a window is used up). A program or a skill therefore never makes more external writes,
  external actions or payments than the remaining budget allows, whatever it carries [`test_ea_fix_limits`]. This
  holds as far as executors report what they send: an executor that does not call `check_primitive` for a request
  is not charged for it (and is not checked either). Money actually spent is also taken from `record`; a granted
  action that is never recorded is charged its estimate.
* **Prohibited classes (spec 12.6) — lexical classifiers are evadable, so approval is the default.** The
  detectors are structural and lexical: recipient-shaped values (addresses, numbers, mentions, handles, user ids)
  in a list under *any* field name, more than `max_mentions` individual mentions or a broadcast / role mention in
  any content field, identity-override fields also inside JSON- and form-encoded string bodies, membership
  operations, incentive fields, audience fields, data classes, a stored secret also when spaced out, re-cased,
  punctuated or split into parts, profiling words in generated code, and content patterns for join-for-join,
  reciprocal membership, member swaps, rewards conditional on engagement in either word order, claims to be human
  or not automated, platform-staff / authority claims and account phishing, and vulnerable groups in outbound text.
  **Every lexical classifier can be evaded by a sufficiently different phrasing**; these detectors reduce what
  slips through, they do not guarantee it. The guarantee is the approval rule: in live mode every publication or
  message to an external destination, every write and every interaction on an owned page needs the operator's
  approval unless the operator pre-authorized that destination in the signed live record (`preauthorized`), and a
  write to anything not owned is denied. Pre-authorizing a destination therefore means accepting that the
  detectors are the only check on what is posted there; an operator who does not accept that leaves
  `preauthorized` empty (or sets `publishing.require_approval: always`). Content that needs judgement always goes
  to the operator (`needs_approval`), never through silently.
* **Secrets after release.** Once a secret has been handed to the executor for a granted action and destination,
  what the executor process does with it is not controlled by the kernel. `scrub` removes known values and their
  common encodings from observations; an encoding it does not know would pass.
* **Grant key.** Grants are HMACs. The executor holds a copy of the key for offline checks, so the executor uid
  could compute a well-formed token; the service nevertheless accepts only nonces it issued itself.
* **Single-uid development mode** (`separated=False`, or the service users absent): every role is the same uid.
  All rules still run, but nothing in section 1 applies. It is for tests and diagnostics only.

## 2a. Hard platform constraints (fixed in code, not in the policy file)

`src/cam_live/harness/auth/platform_constraints.py` (HX, commits a1cdc0e, 863bdae; documented in its docstring and
`docs/HX.md`) adds `deny` findings that no policy file, task, live authorization record, approval, pre-authorization
or `owned` / `interactive` list can turn into anything else. It runs inside `Evaluator.node`, i.e. on `authorize`
(every program / skill step) and on `check_primitive` (every browser navigation and page write, HTTP hop, sandbox
connection, skill step, visual gesture):

* `X-HUMAN-VERIFICATION` — no CAPTCHA / human-verification / bot-challenge solving: no requests to solver services,
  no interaction with or answers to challenge endpoints, no solved-challenge tokens, no solver or evasion
  libraries, no `navigator.webdriver` override, no proxy / IP rotation. Challenges are also recognised
  structurally (answer-key substrings, accessible names of the element being filled or clicked, submitted form
  field names).
* `X-USER-ACCOUNT` — no automation of ordinary or owner Discord user accounts: the browser does not interact with
  Discord's client, login, invites, OAuth2 or user pages (public documentation stays readable; the developer
  portal is refused), raw HTTP never reaches user-only endpoints or uses a non-bot credential, generated code does
  not connect to Discord.
* `X-PLATFORM-ADAPTER` — Discord's API, Gateway and attachment CDN are reachable only through the official bot
  adapter (`discord.*`), which applies the data-use rules and the operator's permission scope; the bot credential
  is released only to an action that is, or contains, a `discord.*` operation.

Limits (HX, G2-harness): the lists are finite, generated code is judged lexically, the constraints cover Discord
(the one integrated platform), static subresources are judged per origin.

## 3. Tamper evidence (detective, not preventive)

The audit log is hash-chained (`hash = sha256(prev ‖ entry)`), each entry carries an HMAC under a key only the
authorization uid can read, and a head file records the last sequence number and hash. `verify_audit_log` detects
edited, deleted, inserted, duplicated, reordered and truncated entries and a chain rebuilt without the key
[`test_ea_audit`]. An attacker who *is* the authorization uid can rewrite log, head and MACs consistently; only an
anchor kept elsewhere (`expect_head=(seq, hash)`, e.g. recorded by the verifier or the operator) reveals that.
`verify_audit_log` requires the head file (or an `expect_head` anchor) by default, so a log emptied or truncated
together with the removal of its head fails; `require_head=False` checks chain consistency only.

**Emergency stop.** `emergency_stop`, `opctl stop` (online or offline) and `touch R/auth/STOP` have the same effect:
the first time the service sees a stop flag it voids every outstanding grant (bumps the revocation epoch), and
`reset_stop` voids whatever is still outstanding, so no grant issued before a stop is usable after the reset
[`test_ea_fix_policy::test_a_stop_set_by_any_means_voids_grants_for_good`]. A flag created and removed by hand while
the service made no call at all is not seen; reset with `opctl reset-stop`.

## 4. What a VM deployment would add

Recommended for live operation (spec 12A.1 prefers a VM):

1. **Kernel-enforced egress control**: a firewall / proxy outside the guest (or `nftables` inside a VM where
   `CAP_NET_ADMIN` is available) that pins each service uid to the destinations of the live authorization record,
   so that section 2's network scoping no longer depends on executors reporting requests.
2. **Network namespaces per service**: sandbox with no interface at all, browser behind the filtering proxy, the
   authorization service with no network.
3. **A separate kernel**: a container escape or kernel bug in the browser or sandbox no longer reaches the host or
   the other tenant of this machine; the GPU can be passed through to the learner VM only.
4. **Immutable audit storage**: append-only flag (`chattr +a`) or shipping the audit stream to a host-side
   collector, turning tamper evidence into tamper prevention.
5. **Snapshots and disk isolation**: the vault on a separate, encrypted virtual disk attached only to the
   authorization VM; rollback of a compromised executor without touching credentials or logs.
6. **Mount / PID namespaces and cgroup limits** per service (not creatable here), so that `/proc` of other services
   is invisible and memory / CPU caps are enforced by the kernel rather than by rlimits alone.

## 5. Trust assumptions

* Root in the container is the operator. Anything that runs as root is outside this model.
* The executor (`cambrowser` by default) is trusted to call the service as specified and to scrub what it returns.
  It is not trusted with the vault (it cannot read it) but it does receive individual secrets at the moment of use.
* The learner is untrusted: it may send any action, any task object and any context. It can pull the emergency
  brake; it cannot release it, approve anything, change the policy, read a secret or write the audit log.
* Environment content is untrusted data. It never reaches a decision except as the literal fields of a typed action.
