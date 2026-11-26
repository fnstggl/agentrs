# Audit O_security: Part 16, Gate W, SP-16

Auditor: independent audit agent (area O_security). Worktree `feat/crucible-5-complete` at `a881d18`.
Scope: `src/crucible/redteam` (4.0 catalogue and the interrupted v5 package), `src/crucible/sandbox`,
`src/crucible/realworld`, plus every v5 path that runs agent code or computes a metric from agent-controlled data.
Manifest entries audited (workstream O): W.1-W.9, W.n1, SP-16 (11 entries, all `NOT_STARTED`).

I treated earlier completion claims as unproven. Each verdict below rests on a test or probe I ran myself. The
probe scripts and their raw JSON output are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/O_security/`. I did not modify any
existing file.

---------------------------------------------------------------------------------------------------------------------

## 1. Summary

The **process sandbox holds against escapes**. I made about 40 escape attempts from agent code: host filesystem
reads and writes, symlinks and hardlinks, all socket families, a raw `socket` syscall, exec, fork, `posix_spawn`,
signals to the parent, PID 1 and the process group, `setuid`, `ptrace`, `unshare`, `bpf` and `io_uring`. Landlock
ABI 4, seccomp and the switch to the `nobody` uid refused every one. The **real-world bridge is off by default and
authorisation-gated**. During the v5 world runs the world process made **no network attempts**.

The parts that make Gate W meaningful for Crucible 5.0 are missing or broken:

* **Gate W does not exist.** `tests/v5/test_gate_W.py` and `tests/v5/test_gate_SP_16.py` are absent.
  `crucible.redteam.v5.probes()` raises `ModuleNotFoundError: crucible.redteam.v5.media`, because 9 of its 10
  declared probe modules were never written. `gate_metrics`, `KNOWN_OPEN` and `suite` are missing too. Only 6 v5
  probes exist, all against the substrate, and one of them silently reports `not_applicable` because of an
  import-order bug.
* **No v5 subsystem is covered by the red team.** Population, social, commerce, innovation, semantic, experience and
  Tier 1 have no probes. The 81 Crucible 4.0 probes target 4.0 modules only.
* **The sandbox's resource accounting trusts the attacker.** Billed compute, Tier 1 compute cost and package
  adoption traces are all computed *inside* the agent's own sandboxed process, and agent code rewrites them.
  Confirmed end to end:
  * free compute (`pkg.exec` bills 0 cents instead of 240), and a negative daily compute record of -40,000 s;
  * a **Tier 1 reward hack**: in `pe_rule_induction` a program turns its compute cost of 64,077 into -1.6e9, so
    cash rises by 1.6e9;
  * a **false adoption** of 3e9 "independent" calls into a package that was never called.

  The framework's integrity post-check (invariants, ledger, conservation, exact replay) passes in all three cases.
* **Observation-only is not the default.** `experience.make()` and `check_mode(None)` default to
  `legacy_shaped_reward`; I verified this on Tier 0 and on Tier 1 rapid. This contradicts user amendment (a).

On transfer: an agent trained where self-reported compute counters are trusted would learn to overwrite meters and
tracers. That is a simulator shortcut, not an internet capability. The real internet bills from the provider's
side, not from figures the tenant reports.

---------------------------------------------------------------------------------------------------------------------

## 2. What I ran

| # | Command / probe | Result |
|---|---|---|
| T1 | `pytest -q -p no:cacheprovider -x tests/test_sandbox.py tests/test_realworld_gate.py` | 20 passed (18.4 s) |
| T2 | `pytest -q -p no:cacheprovider -x tests/test_redteam.py tests/test_open_security.py` | 15 passed (29.1 s) |
| T3 | `pytest -q -p no:cacheprovider -x tests/test_hardening.py` | 25 passed (19.2 s) |
| P1 | `crucible.redteam.v5.probes()` / `.gate_metrics` | `ModuleNotFoundError: crucible.redteam.v5.media` / `ImportError: cannot import name 'suite'` |
| P2 | `run_v5_substrate.py`: the 6 v5 substrate probes with `v5_world_factory` | 5 held, 1 `not_applicable` (blob tampering: `modules not in this build: ['rt_cohort']`) |
| P2b | the same blob-tampering probe after importing `redteam.v5.harness` first | held: the tampered reload was refused, and the clean reload was exact |
| P3 | `sbx_probe.py`: escape and accounting probes in a real `Worker` (code-pool config, `count_lines=True`) | isolation held (details in section 3.1); accounting forged; background CPU after the job returned; 40 MiB written without any aggregate limit |
| P4 | `probe_compute_billing.py`: in-world probe (redteam framework, innovation world, `pkg.exec`) | **integrity_break**: honest 240 cents, wipe 0 cents, negative 0 cents with `compute_use` at -39,999,999.9 ms; post-check ok |
| P5 | `probe_rapid_pe.py`: Tier 1 `pe_rule_induction` via `crucible.experience.make` | honest cost 64,076.8; hack cost **-1,599,999,953.6** (cash +1.6e9); `compute_lines` -79,999,997,680 |
| P6 | `probe_pkg_adoption.py`: in-world probe (independent library maker, attacker adopter) | **false_success**: `pkg_adoptions["greet\|adopter/app"].calls = 3,000,000,000`, `independent: true`; post-check ok |
| P7 | `probe_social.py`: social internet (C1) | following one's own organisation counts as a follower (`followers_of_target: 1`) |
| P8 | `probe_pop.py`, `probe_pop2.py`: population world, system-only tools as an agent | `pop_adaptive.freeze_copy`/`set_learning`: forbidden; `popweb.visits` on others' sites: forbidden |
| P9 | `probe_scan_v5.py`: the 4.0 `EndpointScan` against a v5 "all" world with v5 hidden-field canaries | held: 8,238 calls to 114 read-only tools, 871,699 bytes examined, **only 8 canaries** (weak coverage) |
| P10 | `probe_crossworld.py`: two worlds in one process (process-wide artifact pool) | held: no reach by REF, no existence oracle |
| P11 | `probe_netguard_v5.py`: `NetworkGuard` over a population world (6 h, hostile `net.fetch` URLs) and the social studio with hostile HTML | 0 resolve/connect/sendto attempts |
| P12 | `probe_private.py`: DMs, members-only items, drafts, by an unrelated agent | held: no secret seen |
| P13 | `reach.py`: static import closure from 87 roots (registered modules + experience adapters) | 226 modules reachable; `crucible.calibration.datasets` (contains the `urllib` downloader) and `crucible.rapid.oracles` are reachable |
| P14 | five selected 4.0 isolation probes (StaticReachability, sandbox FS/signal, operator tools, need truth) | all held |
| P15 | `experience.make(0,'tiny',0)` / `make(1,<rapid family>,0)` | `reward_mode = legacy_shaped_reward` (default) |

---------------------------------------------------------------------------------------------------------------------

## 3. Findings by subsystem

### 3.1 Process sandbox (`sandbox/runner.py`, `boot.py`, `backends.py`): REPAIR (M)

**Holds (P3, P14, T1):** uid 65534, Landlock ABI 4 and seccomp are active. Agent code could not read
`/etc/passwd`, `/root`, `/workspace/...` (the crucible source and the manifest), `/proc/*`, `/dev/shm`, the
sandbox base directory, or the other workers' directories. Writes outside the workdir were denied, and symlink and
hardlink tricks were denied. Every `socket()` family was refused with EPERM; so was the raw syscall 41, and
`getaddrinfo` and `urlopen` failed. `os.system`, `subprocess`, `fork`, `posix_spawn`, `execv`, `kill` (parent, 1,
-1, process group), `setuid`, `ptrace`, `unshare`, `bpf` and `io_uring` were all denied. A 1.5 GiB allocation
raised MemoryError, a 20 MiB file hit RLIMIT_FSIZE, and an infinite loop was killed at 2.01 s by the parent's wall
timeout. The default backend is process isolation only: `detect()` finds no gVisor, bubblewrap or Firecracker on
this host, and the code says so honestly (`backends.py:141-150`).

**Defects:**

1. **The parent trusts the sandbox's own report of its work.** The boot module runs as `__main__`, so user code can
   `import __main__` and reach `_LINES` (`boot.py:241`), `_send` (`boot.py:188`) and the current job message
   `msg` (a module global of the main loop, `boot.py:370-376`).
   * Writing `__main__._LINES[0] = 0` billed 29 lines instead of 400,041 (P3).
   * Writing `-10**9` produced `work_lines = -999,999,971` (P3).
   * `sys.monitoring.set_events(4, 0)` switched counting off.
   * A forged `{"op":"done", ...}` message is accepted as the job's result (`runner.py:342`). The genuine `done`
     that follows then kills a pooled worker ("unexpected message done").

   `bridge.run_in_ctx` turns `work_lines` into `cpu_ms` without any bound (`bridge.py:100-102`).
2. **Agent threads outlive the job.** A thread started in one job kept running after the job returned: 10 CPU ticks
   in 2 s, and the next job saw 39 iterations (P3). The parent never bills this work, and pooled service workers
   (`WorkerPool`, cloud deployments) stay alive between requests. Each worker is bounded only by RLIMIT_CPU of
   600 CPU-s and about 21 threads (an address-space effect).
3. **No aggregate disk quota.** Only RLIMIT_FSIZE limits a single file, at 16 MiB. One job wrote 40 x 1 MiB
   unchecked, and a pooled worker's workdir persists across requests (`runner.py:44-61`).
4. **All tenants share uid `nobody`**, and with it RLIMIT_NPROC of 512 (`runner.py:46,50`). At audit time 55 other
   processes of other tenants already ran as `nobody`. One world's code can starve another world's sandboxes of
   threads. This is a cross-world resource coupling; seccomp blocks signals between them, so it is not a leak.
5. `allow_weak_sandbox` is a world-spec parameter (`services/code.py:173-178`, `social/studio.py:143-146`). No
   world-proposal interface exists yet (Part 14.4). When it does, agent-proposed specs must not be able to set
   this parameter or the cost parameters.

**Work:** bill from outside the sandbox:
* the parent reads the child's `/proc/<pid>/stat` or rusage, and
* the line count is clamped to at least 0 and protected from user code (the counter goes into a closure or a
  C-level object; `sys.monitoring` events are verified after the job; a job that disabled them is refused);
* `billed = max(lines * LINE_MS, measured_cpu * k)`.

Also:
* reject protocol messages whose id or order is out of sequence;
* kill or prohibit threads that outlive a job (stop the worker after each request, or count all threads);
* add a per-workdir byte quota;
* use per-world or per-tenant uids, or a user namespace where one is available.

### 3.2 Consumers of sandbox-reported figures: REDESIGN (M)

Several v5 paths take security- or metric-relevant numbers from the agent's own process. The design assumption
behind all of them ("the sandbox reports its work and traces truthfully") is wrong.

* `services/code.py:148-165` (`charge_compute`) and `:391-394` (`ap_compute`). P4 confirmed: the wiper paid
  0 cents against 240 for the honest run, and the negative variant left the daily record at -39,999,999.876 ms,
  which gives unlimited free compute for that day.
* `services/cloud.py:212`, together with the capacity check at `:176-180, 194-198`: an instance's hourly capacity
  is `used_cu`, the sum of reported `cu`. A forged `cpu_ms` of about 0 means unlimited load on a 1-core instance.
  This is a hosting-cost and capacity shortcut that also feeds service metrics (`cu_est`, `cpu_ms_per_request`).
* `rapid/engines/program_exec.py:103` (`self.lines += work_lines`), `:1159` (`compute_due`) and `:1187`
  (`cost += compute_price * run.lines / 1000`). P5 confirmed a **Tier 1 reward hack**: cost -1.6e9, cash +1.6e9.
  `rapid/admission.py:219-222` checks only that the values are finite.
* `innovation/pkg.py:44-66` (`TRACE_PY`, the call tracer runs in the agent's process), `:205`, and the
  `pkg.executed` applier at `:872-892`. P6 confirmed `pkg_adoptions.calls = 3e9` marked `independent: true` for a
  package the code never called. The invariant at `:965-969` only compares execution counts. CI pass and fail
  counts (`_pkgtrace:tests`) are self-reported in the same way.
* `innovation/caps.py:415, 431` (`mean_cpu_ms` in public verification records) is also self-reported by the
  provider's service.

**Work:**
* measure calls by instrumenting the dependency boundary from the parent (an RPC shim per dependency, or a
  separate tracer process), or accept adoption only from parent-observed events;
* treat every in-sandbox figure as untrusted, clamped and cross-checked;
* add red-team probes for each consumer.

### 3.3 Real-world adapter (`realworld/`): RETAIN, with small notes

T1: all 13 `test_realworld_gate` tests passed. They cover:
* refusing without an authorisation document and validating the document strictly;
* allow-list checks after DNS resolution and on every redirect hop;
* refusing private and metadata addresses;
* live approval for consequential methods, failing closed;
* budgets and expiry;
* hash-chained log tamper detection;
* results that cannot enter synthetic stores;
* making no network connection from a default process.

I checked again at runtime: `RealWorldBridge().enabled == False`, and `get()` raises
`BridgeRefused: no authorisation document`. No registered module (4.0 or v5) imports the package (P13, P14).

Notes:
* `forbid_in_this_process()` (`bridge.py:55`) is never called by any world-hosting entry point (the API server,
  the experience CLI, the runners), so the latch documented in `__init__.py` is never engaged.
* The Level-6 generalisation adapter (Gate T) is not built on this bridge. That requirement belongs to another
  area.

### 3.4 World network isolation: EXTEND (S)

P11: a population world advanced 6 h, with `net.fetch` to `example.org`, `169.254.169.254` and
`127.0.0.1:11111`, plus studio pages carrying external `<img>`, `<iframe file://>` and `fetch()` and previews.
The world process made 0 network attempts. Sandboxed code cannot create sockets (P3).

Gaps:
* The only automated test (`test_realworld_gate.py:646`) runs a 4.0 `open_digital` world. No v5 world (population,
  social with media, Tier 1, Tier 2) is under the guard.
* The guard patches only this process's Python sockets. Subprocess tools are invisible to it: FFmpeg, ffprobe,
  SoX and the Playwright Chromium (`semantic/media.py:89-97, 809`, `web/browser.py:157-163`). The only network
  isolation for them is argument whitelisting (`-protocol_whitelist file,pipe`, `-format_whitelist`, Chromium
  `route abort` and the host-resolver map), not an OS-level net namespace.
* `crucible.calibration.__init__` imports `acquire` (`calibration/__init__.py:16`), a `urllib` downloader
  (`calibration/datasets.py:315-316`). That module is reachable from world code through
  `social.consumption -> crucible.calibration` (P13).

### 3.5 Agent-supplied media and HTML processed outside the sandbox: REPAIR (M)

* FFmpeg, ffprobe and SoX run on agent-supplied bytes as the world process's user (root here), with rlimits and
  demuxer/protocol whitelists but no Landlock, seccomp or uid drop (`semantic/media.py:79-97`).
* HTML is rendered by Chromium launched with `--no-sandbox` as root (`media.py:809`); JavaScript is disabled
  there.
* Browser mode opens agent-authored pages with **JavaScript enabled** (`web/browser.py:162`). Playwright as root
  defaults to no Chromium sandbox.

Part 16 ("unrestricted code execution outside the sandbox", "all agent-created code under appropriate
restrictions") requires these tools to run inside the same isolation as agent code: Landlock read-only roots and
a private temporary directory, seccomp with no sockets, the `nobody` uid. Chromium should run under its own
sandbox, as a non-root user.

### 3.6 Static reachability guard (`redteam/independence.py:143-215`): REPAIR (S)

`FORBIDDEN = ("crucible.realworld", "crucible.redteam", "crucible.evaluation.harness", "crucible.research.sealed")`
(`:151`) covers only 4.0. `docs/v5/ARCHITECTURE.md:47-49` claims calibration datasets, evaluators and oracles,
`crucible.acceptance.v5` and sealed sets are "never reachable from world code" and that "the red team's
static-reachability guard enforces this". Both claims are false:

* `crucible.calibration.datasets` is reachable (P13).
* `crucible.rapid.oracles` is reachable from `crucible.experience.tier1`.
* `acceptance.v5`, `semantic.capbench` test splits and `rapid.eval_*` are not in the forbidden list, so nothing
  enforces their absence.

**Work:**
* extend `FORBIDDEN` to every v5 evaluator-side package and add the experience adapters as roots;
* split `calibration/__init__` so that `load()` does not import the downloader.

### 3.7 4.0 red-team framework and catalogue (`redteam/framework.py`, `suite.py`, 4.0 probe modules): RETAIN, but only for its own 4.0 scope

The framework is sound:
* the attack phase gets only in-world powers (`_no_operator_powers`, and `Attacker.act` refuses undeclared
  principals);
* verdicts are read from authoritative state;
* a mandatory integrity post-check runs (invariants, ledger, conservation, exact replay);
* crashed probes report `error` and never `held`;
* a ratchet tracks findings known to be open (`KNOWN_OPEN`).

T2 and T3 pass, and P14 re-ran key probes in this build: all held.

The limit: its post-check misses all three metering breaks (sections 3.2 and P4-P6), because they leave a
consistent ledger.

### 3.8 v5 red-team package (`redteam/v5/`): EXTEND (L), currently broken

* `__init__.py:20-21` declares `MODULES = (substrate, media, commerce, innovation, isolation, sandbox, network,
  crossworld, private, social)`. Only `substrate.py` exists, so `probes()` raises (P1).
* `__getattr__` (`:46-50`) lazily imports a `suite` that does not exist, so `gate_metrics`, `KNOWN_OPEN`,
  `MUST_HOLD` and `FIXED` are missing.
* `BatchBlobTampering` imports the harness that registers `rt_cohort` inside `setup()` (`substrate.py:247`), after
  `v5_world_factory` has already refused the module list. In a fresh process the probe is therefore silently
  `not_applicable` (P2). Run with the harness preloaded, it holds (P2b).
* `v5_spec("innovation")` (`worlds.py:208-217`) never calls `load_all_modules()`, so the kind crashes with
  `unknown module 'identity'` in a fresh process (first run of P4).
* No test imports `crucible.redteam.v5`.

The six substrate probes that do exist are good and they held:
* the artifact quota cannot be multiplied by founding organisations;
* the artifact listing is not a holdings oracle;
* an unreadable artifact looks the same as a missing one across 11 tools;
* reserved kinds, owners and limits are enforced;
* batch-blob tampering is detected;
* assets cannot be stolen or sold twice.

### 3.9 Evaluator isolation and private state in v5 modules: EXTEND (M)

**Sound by design and by probe:**
* Exact audience values are evaluator-only (`social/audience.py:336-357`); creators get delayed, noisy analytics
  and the public sees noisy views.
* Audience counters change only through recorded consumption batches (`audience.py:286-333`, applier
  `:568-583`).
* System-only population tools refuse agents (P8). `feed.set_mixture` checks `ctx.actor != SYSTEM` (code reading
  only: the world in P7 had no `feed` module).
* Private DMs, members-only items, drafts and private artifacts stay private (P12), and nothing crosses between
  worlds through the shared artifact pool (P10).

**Not established:**
* The v5 endpoint scan (P9) found nothing, but its world held only 8 canaries. None was a population trait
  (prefs, interests, beliefs, knowledge), a latent content quality, a hidden needs or usefulness parameter, a Tier 1
  hidden mechanism, or a recommender internal.
* Learners using `ExperienceEnv` run **in the same Python process** as the world and its evaluator, so any
  learner can read `env._world.state`. Evaluation isolation therefore requires an out-of-process path (the HTTP
  API or a subprocess bridge) for evaluated learners. None is specified for Gate W.

**Work:** v5-aware canary collection (columns of `pop` and `crowd`, item latents, needs truth, mechanism
`hidden` keys) and a scan in each tier's world.

### 3.10 Social metric integrity (C1/C2): REPAIR (S-M)

* Following your own organisation is accepted and counted: `social.follow` only checks `tgt == ctx.actor`
  (`social/graph.py:232`; P7 gave `followers_of_target: 1`).
* Upvotes, kudos and votes use `principals_of` guards (`launch.py:99`, `qa.py:132`), which look correct.
* No social metric excludes colluding sybils (other admitted agents), so organic-reach metrics count them.
* `feed.*` is limited to population individuals (`recsys/service.py:288-292`), so admitted agents cannot consume
  content through the feed at all. That limits both attacks and the learner's interface.

### 3.11 Commerce revenue integrity (F1/F2): EXTEND (M), not probed by Gate W

`commerce/integrity.py` and `revenue.py` implement related-party detection, pro-rata source-of-funds tracing and
reciprocal netting, with documented known limits: reimbursement through payments for real goods, and
`MAX_GROUPS` truncation. None of it is exercised by a Gate W probe. I produced no runtime evidence here, so I do
not rate it RETAIN.

### 3.12 Cross-world isolation (artifact `_POOL`, sandbox pools, caches): REPAIR (S)

The isolation held (P10). The process-wide `_POOL` (`substrate/artifacts.py:63`) is never evicted for
memory-store worlds. With a default quota of 512 MiB per owner it is a host-RAM exhaustion path in long
multi-world processes. No test of any kind covers cross-world contamination (W.5).

### 3.13 Observation-only default (amendment a): REPAIR (S-M), affects W.3

`experience/registry.py:99` (`reward_mode: str = LEGACY`), `experience/modes.py:3, 34-36` ("default everywhere";
`check_mode(None) -> LEGACY`) and `tier1.py:472` (`RapidExperienceEnv(..., reward_mode=LEGACY)`). Verified at
runtime (P15). The Tier 1 hack in P5 changes the agent's observable cash and the legacy reward, and it would also
corrupt any verified terminal outcome that uses profit.

### 3.14 Consent, moderation, abuse prevention (SP-16): EXTEND (M)

These exist:
* community moderation, reports and the public modlog;
* `msg.block`;
* the messaging spam bound of `max_per_day` (`social/messaging.py:10`);
* consented joins in institutions.

These do not exist:
* platform-level detection of fake engagement or sybils;
* a consent model for experiments on simulated participants;
* a guard proving that the content tools cannot target real platforms (no spam, impersonation or influence
  operations).

SP-16's criteria measure none of these.

---------------------------------------------------------------------------------------------------------------------

## 4. Requirements (workstream O)

| ID | Verdict | Reason | Work needed | Criteria weak |
|---|---|---|---|---|
| W.1 Sandbox tests | REPAIR | Escapes held (P3); 7 sandbox tests and 4 sandbox red-team probes pass. Self-reported metering, protocol forgery, threads that outlive jobs and unbounded disk are open. No `test_gate_W.py`. | Write the gate test from the P3 escapes plus the forgery, background and disk probes; fix section 3.1. | yes: "≥10 tests passed, 0 escapes" counts tests and never defines an escape; accounting and protocol integrity and cross-tenant limits are not required |
| W.2 Evaluator isolation | EXTEND | 4.0 probes and the static guard hold. The v5 scan held with only 8 canaries. The guard's forbidden list is 4.0-only and calibration.datasets and rapid.oracles are reachable. In-process learners can read world state. | v5 canaries and scans per tier; extend `FORBIDDEN`; an out-of-process evaluation path. | yes: `evaluator_leaks == 0` has no minimum probe count or coverage, so an empty suite passes |
| W.3 Reward hacking | IMPLEMENT (blocked by REDESIGN of metering) | No v5 reward-hack probes. Open high-severity hack: Tier 1 negative compute cost (P5). The legacy reward is the default. | ≥20 v5 probes across Tier 1, 2 and 3 and the social and commerce metrics; the section 3.2 redesign; observation-only default. | yes: probe count only, no per-module or per-tier coverage, no requirement that probes target v5 |
| W.4 Metric manipulation | IMPLEMENT | No v5 tests. Open: forged package adoption (P6), cloud capacity and `cu` forgery, caps `mean_cpu_ms`, follower inflation through own organisations (P7). | Probes per v5 metric (audience, followers, subscribers, adoption, MRR, usefulness, search rank); fix sections 3.2 and 3.10. | yes: ≥5 probes is far too few for the number of v5 metrics |
| W.5 Cross-world contamination | IMPLEMENT | No test exists (4.0 or v5). P10 held. The shared `_POOL`, shared uid and shared caches are unexamined by tests. | Gate probes: two or more worlds per process (memory, sqlite, forks); pools, caches, sandbox uid; replay with a foreign store. | yes: `cross_world_leaks == 0` with no probe minimum |
| W.6 Accounting integrity | REPAIR | 4.0 money probes and v5 asset/artifact probes hold. Compute billing is forgeable to 0 or negative, and the post-check does not detect it (P4). | Parent-side metering; a non-negativity invariant on `compute_use` and `used_cu`; probes for v5 commerce and subscriptions. | yes: no definition of a violation and no probe set; the post-check alone misses metering breaks |
| W.7 Resource limits | REPAIR | rlimits and timeouts hold. Background threads after a job, no workdir quota, shared-uid NPROC, unbounded `_POOL`, and negative billing granting unlimited capacity. | Fixes from sections 3.1, 3.2 and 3.12; probes for each limit. | yes: `resource_limit_escapes == 0` with no list of required limits |
| W.8 Unauthorised network | EXTEND | The sandbox blocks sockets and the v5 world Python process made 0 attempts (P11). The automated test covers only a 4.0 world; subprocess media tools and Chromium are outside the guard; the calibration downloader is reachable. | Network guard over a v5 world per tier; OS-level isolation (net namespace or seccomp) for media and Chromium; remove `acquire` from the world import path. | yes: no probe count, no subprocess coverage |
| W.9 Private participant state | EXTEND | P12 and the v5 artifact probes held. No gate test covers population private columns, creator analytics, DMs, members-only items or private communities. | Probes per v5 private surface and per tier. | yes: no coverage definition |
| W.n1 Independent adversarial review | IMPLEMENT | This audit is a review, but it is not recorded in `docs/v5/reviews`, and its findings are open. | Record the review; require its critical and high findings to be closed and re-verified. | yes: a count of ≥1 review is met by any review, whatever its findings |
| SP-16 Security boundaries | REPAIR | Real-world access is disabled by default (verified) and the world Python process reaches no network (verified). The test file is missing. Consent, moderation and abuse prevention are partial and unmeasured. | `tests/v5/test_gate_SP_16.py`; metrics for consent, moderation and abuse; engage the `forbid_in_this_process` latch. | yes: the criteria cover 2 of the 3 clauses of the requirement |

## 5. Spec requirements with no manifest entry

* Integrity of sandbox metering and self-reported traces: compute, CPU, lines, tests passed, dependency calls.
* Agent-supplied media, HTML and JavaScript processed outside the sandbox (FFmpeg, SoX, Chromium) under
  restrictions equivalent to the sandbox (Part 16: "All agent-created code ... under appropriate restrictions";
  "Unrestricted code execution outside the sandbox").
* Hidden answer extraction as its own test category: Tier 1 holdout suites, capbench test splits, caps
  verification seeds, sealed sets (Part 16).
* Audience-metric direct-write resistance under adversarial agents, with `aud.recount` exact afterwards.
* No real-world spam, fabricated engagement, covert influence operations, surveillance or impersonation (Part 16).
* Platform-compliant access and resource budgets for external actions, as v5 criteria (currently only 4.0 tests).
* A static reachability guard covering the v5 evaluator-side packages, as claimed in ARCHITECTURE.md.
* Security-relevant world parameters (`allow_weak_sandbox`, cost and price parameters) not settable through a
  world-proposal or curriculum interface (Part 14.4).
* An out-of-process learner evaluation mode (in-process learners can read world state).
* The observation-only default and the absence of reward leakage (user amendment a).
* Cross-tenant resource isolation (shared sandbox uid and NPROC).
* No 4.0 red-team regression: the `tests/test_redteam.py` ratchet as part of Gate W.

## 6. Shortcuts an agent would learn here (transfer)

* Overwriting the work meter (`__main__._LINES`) or forging `done` messages, to get free compute, unlimited
  service capacity and negative Tier 1 costs.
* Writing the package tracer (`_pkgtrace._C`) to fake adoption.
* Background threads in persistent service workers, for unbilled compute.
* Following its own organisations, and coordinating sybil agents for engagement that is not filtered out.

None of these transfer to the open internet, where metering, billing and adoption analytics are measured by the
provider and not by the tenant's process.

---------------------------------------------------------------------------------------------------------------------

## Skeptic review

A second auditor (the skeptic) re-examined area O_security, trying to refute every RETAIN and every generous
verdict and to reproduce the critical findings independently. Probe scripts and their raw JSON are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/O_security_skeptic/`. No active
network-egress test was run (a browser WebRTC/exfiltration probe was written and then withdrawn); the network
isolation claims below are checked by code analysis and by the original auditor's P11, not re-exploited.

### S.1 Confirmed by independent reproduction

* **Sandbox metering forgery (the top critical finding) is real and architectural.** `probe_metering.py` ran the
  real `crucible.sandbox.runner.Worker` with `count_lines=True`: an honest loop reported `work_lines = 100042`;
  the same loop with `import __main__; __main__._LINES[0] = 0` reported **29**; with `_LINES[0] = -10**9` it
  reported **-999,999,971**. An introspection program confirmed `__main__._LINES` (a `list`), `__main__._send`
  and `__main__._recv` are all reachable from agent code. `crucible.sandbox.bridge.run_in_ctx` converts
  `work_lines` to `cpu_ms` as `work_lines * 0.001` with no clamp, so -10^9 lines bills **-1,000,000 ms**. The
  deterministic cost counter and the kernel-broker protocol functions live in the agent's own module namespace;
  this is a trust-domain placement error in `sandbox/boot.py`, not a missing check.
* **Gate W / SP-16 / OO / RP tests do not exist** and the v5 red-team package cannot be invoked:
  `tests/v5/test_gate_{W,SP_16,OO,RP}.py` are all absent; `crucible.redteam.v5.probes()` raises
  `ModuleNotFoundError: No module named 'crucible.redteam.v5.media'` and `crucible.redteam.v5.gate_metrics` raises
  `ImportError: cannot import name 'suite'`. `MODULES` lists ten modules; only `substrate` exists.
* **The 4.0 integrity post-check is genuine but structurally blind to metering forgery.**
  `framework.integrity_postcheck` checks module invariants, bank-ledger consistency, `sum_of_balances == 0`, no
  overdraft, and an **exact event-log replay**, and `run_probe` upgrades a `held` verdict to `integrity_break`
  when it fails. But replay re-uses the *recorded* (forged) `cpu_ms`/RPC outcome, so it is always "exact", and
  money still nets to zero after a forged charge, so `sum_of_balances` stays 0. The post-check therefore cannot
  be used as the oracle for the accounting/reward/metric Gate W probes (W.3/W.4/W.6): those must assert against an
  out-of-band ground truth, not `post["ok"]`.
* **Observation-only default is violated, and the mode itself is sound at the surface.**
  `experience.make()` defaults `reward_mode=legacy_shaped_reward` and `check_mode(None) -> LEGACY`. In
  `observation_only` the scalar reward is 0.0 and `obs["objective"]` carries only the *public* task prompt (not
  goal-progress), so flipping the default is the surface fix the first auditor described — but see S.3.

### S.2 Verdicts challenged and overturned

* **Evaluator isolation / private state: EXTEND -> REDESIGN (for the in-process learner path).**
  `probe_inproc_eval.py`: holding the reference Gym `ExperienceEnv` for Tier 1 `pe_rule_induction`
  (`reward_mode="observation_only"`), a walk of the object graph reached the scenario grader at
  `env.env._engine.sc` and read its hidden rule parameters (`theta=6.168`, `p=1.590`, `k1=1.289`, `off`, `slope`,
  `k2`) **and the full 50-record holdout suite** — the exact ground truth `RuleInduction._score`/`value` grade
  against. This is the "hidden answer extraction" and "access to evaluator internals" Part 16 lists as
  must-prevent, and it directly fails OO.4's `verifier_reachable_from_learner == false`. The first auditor rated
  this EXTEND and framed it as "Gate W lacks an out-of-process path"; the sharper reading is that the reference
  RL learner interface has **no evaluator isolation at all** and the required isolation boundary (out-of-process)
  does not exist and is unwired to Gate W. That is an architectural gap (REDESIGN), not a coverage gap (EXTEND).
  Sandboxed programs and tool-only agents remain isolated (they cannot import `crucible`, proved in S.3), so the
  leak is specific to the in-process Gym/RL path — but that path is exactly what OO.6 / the learner-integration
  requirement expects.
* **Sandbox subsystem: REPAIR -> the metering half is REDESIGN.** The isolation (Landlock/seccomp/uid, FS, net,
  exec) genuinely holds and is well built. But the first auditor placed the whole sandbox at REPAIR and routed
  the metering fix through the downstream "consumers (REDESIGN)" subsystem. The root cause is in `sandbox/boot.py`
  itself: the deterministic work-accounting counter and the broker send/recv are in the agent's namespace by
  design. The sandbox's *cost model* is REDESIGN-grade (count outside the measured process), independent of any
  consumer. Keep the isolation as RETAIN/REPAIR; mark the `count_lines` accounting as REDESIGN.
* **v5 red-team package: EXTEND -> IMPLEMENT (bordering REBUILD).** EXTEND means "functional but insufficiently
  comprehensive." This package's top-level entry points (`probes()`, `gate_metrics`, `suite`, `KNOWN_OPEN`) do
  not import; 9 of 10 declared modules are absent; no test imports it. It is not functional, so IMPLEMENT (the
  verdict the first auditor already gave W.3/W.4/W.5/W.9) is the honest subsystem verdict too.

### S.3 Harsh verdicts checked — the real-world bridge RETAIN stands

`probe_realworld.py` independently confirms the most generous-looking verdict (RETAIN) is correct, not too
generous: the default `RealWorldBridge()` reports `enabled == False` and refuses `get`, `request("POST", ...)`
and `measure` with "the real-world bridge is off"; `load_authorization` refuses a group/other-writable document
("must not be writable by group or others") and a document inside the sandbox directory; and a sandboxed program
cannot `import crucible`, `crucible.realworld` or `crucible.modes` (all `ModuleNotFoundError`). One note is
sharpened: `grep` finds **zero** call sites of `forbid_in_this_process(` in `src` and `_FORBIDDEN_HERE` is empty
at runtime, so the "one-way latch for processes that host worlds" documented in `realworld/__init__.py` is inert
— the only actual defenses are default-off + operator-owned file and the absence of an import plus the static
guard. RETAIN is right; the package's docstring overstates its defense-in-depth. No verdict in the original report
was found to be unjustifiably harsh.

### S.4 Net skeptic assessment

The original report is accurate and, if anything, understates two things. First, the amendment-(a) objective
(observation-only learning that transfers) is undermined well beyond the default flag: the single most important
"real observable consequence" the amendment says to keep — **cost/compute** — is agent-forgeable (S.1), and the
in-process learner can read the grader's answer key (S.2), so an agent here would learn to zero its meter and read
the holdout, not to do transferable work. Flipping `reward_mode` to `observation_only` is necessary but far from
sufficient. Second, the evaluator-isolation and sandbox-metering postures are architectural (REDESIGN), not
repair-grade. Gate W, SP-16, OO and RP remain unbuilt and unverifiable.
