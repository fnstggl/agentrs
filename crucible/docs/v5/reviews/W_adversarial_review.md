# Independent adversarial review of the merged tree (Gate W.n1, wave 3b)

Reviewer: workstream O (security), wave 3b, on branch `c5c/w3b-sec` from the integration head `55433e7` (waves 1, 2a,
3a and the integration fix merged). This review is independent of the code it attacks: the reviewer did not write the
research instrument (N), the media/social call sites (C, D), the package ablation (G) or the kernel memo (A/K3), and it
re-measured the earlier security workstreams' own claims instead of trusting them. Every verdict below rests on a probe
or control that is part of the gate suite, so it is re-executed (and ablated) whenever the gate runs:
`crucible.redteam.v5` (Gate W, 84 probes on this branch; Tier 2 probes grow with the domain registry: 129 on the check merge, 45 of them Tier 2) and `crucible.redteam.v5.instrument` (SP-15.8, 11 controls).

Targets named by the assignment: metering, ablation-based adoption, media/JavaScript isolation, network isolation,
cross-tenant state, and the research instrument's evaluator.

## 1. Holes found and fixed in this review

| id | area | attack | before | after | probe / test |
|---|---|---|---|---|---|
| DEF-O-7 (high) | sandbox time limits | A program with a 3 s limit fans out (`code.run` from inside `code.run`) to 16 nested programs that `time.sleep(2.5)`. Time in world calls was not the caller's, and every nested job got a fresh limit. | one tool call held the single-threaded world 42.3 s at ~0 CPU (unbounded in depth x fan-out) | call tree bounded by twice the outer limit: 3.1 s | `v5.sandbox.nested_runs_multiply_wall_time` |
| DEF-O-8 (high) | metering | The parent subtracts a per-byte protocol allowance (20 ns/B) for every message of a job, world-call arguments included; long strings encode at ~2 ns/B. Padding 20 world calls with 7 MB strings. | a 3M-iteration loop billed 0 ms (honest 396 ms) | allowance 1.5 ns/B (below the fastest JSON rate measured, 1.97 ns/B): padded run billed 4,664 ms | `v5.sandbox.protocol_allowance_farming` |
| DEF-O-9 (low) | cross-tenant state | The machine-level meter memo (A/K3 `DiskMemo`, shared by every world and process) evicted its oldest rows globally; a world running many distinct computations floods it. | 100 of 100 victim anchors evicted (scaled demonstration) | eviction per lineage: 50 of 50 survive | `v5.crossworld.meter_memo_eviction_is_per_lineage` |
| DEF-P-16 (high) | jail | Jailed Chromium below a root-only TMPDIR (agent scratchpads are mode 700). | every jailed-browser test failed | profile relocated into the jail's home; 26 browser tests pass | `test_jailed_browser_starts_below_a_root_only_tmpdir` |

Hardening without a demonstrated exploit: every jailed browser launch now gets a private HOME/TMPDIR/XDG directory
(concurrent browsers of one kind share a uid; they used to share the kind's home, i.e. writable profile and cache
state across worlds) - `test_concurrent_jailed_browsers_of_one_kind_get_private_homes`.

## 1a. Kernel objects shared between tenants (wave-3b continuation)

The sandbox gives each world's programs their own uid, a private working directory and Landlock rules over paths;
the review asked what else two tenants share on one kernel. Each row: found with a probe on this branch, fixed in
`sandbox/boot.py` / `sandbox/jail.py` / `sandbox/learnerhost.py`, probe now holds (FIXED in the ratchet).

| id | channel | attack | before | after | probe |
|---|---|---|---|---|---|
| DEF-O-14 (high) | System V IPC, POSIX mq | world A's program creates a 0666 shared-memory segment, message queue and semaphore under a key both agents know and writes a canary; world B's program (other uid) reads | canary read through all three SysV kinds (POSIX mq: the name was created, open refused by Landlock - a name channel) | every IPC syscall refused (EPERM) | `v5.crossworld.kernel_ipc_does_not_cross_worlds` (MUST_HOLD) |
| DEF-O-15 (high) | learner processes | learner A stashes a canary in SysV IPC and exits; a FRESH learner process B reads it; B makes the kernel SIGKILL learner V (F_SETOWN + F_SETSIG on an O_ASYNC pipe; FIOSETOWN on its participant socket); F_SETOWN's ESRCH as a pid oracle; all three share one uid | canary carried to the fresh process; V killed (rc -9); pid oracle answered; one uid | all refused; V alive; three uids | `v5.sandbox.hosted_learners_cannot_reach_each_other` |
| DEF-O-16 (high) | advisory locks | a service (long-lived worker) in world A locks a 16-bit pattern of standard-library files (flock), of /dev/null byte ranges (record write locks) and read-locks files (visible to F_GETLK with the holder's pid); world B's program reads | 9 of 9 set bits through each of the three | lock-taking succeeds without effect (D-77 amendment), queries real; 0 bits; sqlite3 WAL still works | `v5.crossworld.file_locks_do_not_cross_worlds` |
| DEF-O-17 (high) | inotify | a service in world A arms watches on 16 rarely used standard-library files; world B's program opens the files of a secret; A's next request reads its queue | the exact 16-bit secret | inotify/fanotify refused | `v5.crossworld.file_watches_do_not_cross_worlds` |
| DEF-O-18 (high) | memory files, learner directory | a program writes memfd files (pages not mapped: not counted by RLIMIT_AS); a jailed learner fills its private directory with files (RLIMIT_FSIZE bounds one file) | sandbox: 640 MiB under a 256 MiB cap; learner: 512 MiB held under 512 MiB, 256 MiB written with no directory quota | memfd refused (sandbox, tool-mode jail); learner directory quota enforced by a watchdog: stopped at 40 MiB for a 32 MiB quota | `v5.sandbox.hosted_learner_memory_and_disk_are_capped`, escape matrix `memfd_beyond_the_memory_cap` |

Checked and not a channel here: access times of standard-library files (overlay lower layer: reads do not update
atime); /dev/shm and /tmp (Landlock); sockets (seccomp); signals to other processes (seccomp self-only rules).
**Correction (repair pass):** this paragraph also said "scheduling of other processes" was not a channel. That was
false: the self-only rules covered only *setting* attributes; a process could set its own and read anyone's (the
wave-3b verifier passed a 16-bit secret between worlds in 12 s) - see section 1c. Residual, stated: covert *timing* channels of a shared kernel (CPU load, page-cache residency of shared
read-only files observable through read latency, pid allocation, free disk space) cannot be removed by process-level
isolation; they need a VM or dedicated hardware per tenant.

## 1c. Repair pass after the wave-3b verifier (2026-11-27)

An independent verifier re-ran this review's claims and found that several fixes above were incomplete. Each row:
a probe in the catalogue, run on the unfixed head 31ae226 (finding) and on the repaired branch (held); fixed in
`sandbox/` (D-77 amendment 7-13) or `realworld/` (D-76 amendment).

| id | channel / hole | before (31ae226) | after | probe |
|---|---|---|---|---|
| DEF-O-21 (high) | scheduling attributes: a service sets its own SCHED_IDLE, nice 13 and I/O priority level (3 bits per request); another world's program reads them on any pid | **leak**: the 16-bit secret 0xb38b received exactly; getpriority's ESRCH a pid oracle | read side self-only too; 0 other pids readable, no oracle | `v5.crossworld.scheduling_attributes_do_not_cross_worlds` (MUST_HOLD) |
| DEF-O-23 (high) | directory notifications (F_NOTIFY / dnotify) on standard-library package directories | **leak**: 4 of 4 set bits (0xb2) | F_NOTIFY refused in every mode | `v5.crossworld.directory_notifications_do_not_cross_worlds` (MUST_HOLD) |
| DEF-O-25 (high, new in the repair pass) | a shared futex on a word of libc's code (keyed by the file): world A's worker waits, world B's worker wakes | **leak**: 5 of 5 set bits; every wake counted the other tenant's waiter | non-private wakes/requeues refused; waits allowed (pthread_join) | `v5.crossworld.shared_futexes_do_not_cross_worlds` (MUST_HOLD) |
| DEF-O-26 (high, new in the repair pass) | another process's CPU clock (clock_gettime on MAKE_PROCESS_CPUCLOCK(pid)) | **leak**: world B read world A's worker CPU time (365 ms) | refused with EINVAL; own process/thread clocks work | `v5.crossworld.cpu_clocks_do_not_cross_worlds` (MUST_HOLD) |
| DEF-O-22 (high) | two jailed learner processes: scheduling attributes, dnotify, shared futex, CPU clock, pid oracle; a learner with a colliding uid against another | **integrity break**: all channels open; the same-uid attacker set the victim's RLIMIT_CPU to 1 s and the victim died | tool-mode jail has the full self-only table and the futex/clock rules; uids claimed machine-wide | `v5.sandbox.hosted_learners_share_no_process_state` (MUST_HOLD) |
| DEF-O-24 (high) | disk quotas read the listing: files unlinked while held open (and deleted files kept by a mapping alone) | **integrity break**: sandbox worker kept 8 x 15 MiB (and 6 x 15 MiB mapping-only) against 64 MiB; learner 6 x 16 MiB and 2 x 48 MiB against 32 MiB | the parent counts deleted files held open (/proc under the process's fs identity; workers dumpable again); mapping-only deleted files stop the process | `v5.sandbox.disk_quota_counts_files_held_open`, `v5.sandbox.hosted_learner_memory_and_disk_are_capped` |
| DEF-O-27 (high) | the bridge's denylist (and the circular SP-16b measurement) | 10 of 10 third-party hosts on no list reached (impostors on loopback behind public addresses) | ownership file at /.well-known required for every non-loopback host; 10 of 10 refused; an owned public host accepted | SP-16b `realworld_abuse_paths`, `tests/test_realworld_gate.py::test_public_hosts_need_the_operators_ownership_file` |
| DEF-O-28 (medium) | measurement power: stash adversaries depended on N's learner; timing control could not detect a channel; vacuous Tier 2 rows counted for W.2; W.n1 counted this document | stash in-process gain 0.037 (CI -0.04..0.12) on the merge; timing 24 episodes, 6 settings; 17 of 30 Tier 2 rows vacuous | stash adversary learns on its own (0.584, CI 0.27..0.90); timing passes only on evidence of absence (CI inside +-0.35, power 0.98); vacuous rows reported as `evaluator_isolation_vacuous`; W.n1 counts only the audit | SP-15.8 controls; Gate W.2, W.n1 |

Also: the in-process meter memo is bounded process-wide again (the largest lineage evicts; 50 lineages had held 50x
the bound). Advisory locks stay no-ops (D-77 (5)); the semantic change is pinned by a test (two descriptors of one
program do not exclude each other; threading locks do).

Residual, stated (unchanged by the repair): machine-global counters every process can read - pid allocation (a new
thread's id), sysinfo's process count and load, free memory and disk - are noisy, low-rate channels of any shared
kernel; so is CPU and cache contention. Closing them needs a VM or dedicated hardware per tenant.

## 1b. Evaluated learners of the experience package (W.2, owner L)

| id | attack | result |
|---|---|---|
| DEF-O-13 (high, open) | an adversarial learner function (`fn(participant)`, the `remote.evaluate` contract) looks for the evaluator and sealed material from its own process | **LEAK**: on this branch the only path drives it in the process holding the RemoteEvaluator (gc reaches it, and the World); on a check merge with c5c/w3b-obs, `remote.evaluate` runs it in its own process but as root, and it reads the 4.0 sealed salt, sealed sets and sealed task files under ~/.crucible and other processes' environments. Hosted by `crucible.sandbox.learnerhost` (protocol `experience-remote`) the same adversary reaches nothing (`v5.isolation.evaluated_learner_reaches_no_sealed_material`, `test_learner_host_runs_an_experience_remote_learner_function_jailed`) |

| DEF-O-20 (high, open) | a learner holding only the participant pipe of `remote.host` writes a frame header announcing 1,000 bytes and sends 24 | **FROZEN**: the hosted world serves both pipes in one loop and waits for the whole frame; the evaluator side gets no answer (`v5.isolation.remote_evaluator_survives_a_stalled_participant`, KNOWN_OPEN; also on the check merge). The research evaluator has the same defect (DEF-O-19). |

## 2. Research instrument (SP-15.8): attacks by adversarial learners

| control | attack | result |
|---|---|---|
| process_memory_stash | learns like `weights` but keeps the network in the `json` module's dict; checkpoint constant; claims arch `weights` | held on this branch: in-process gain 0.527 (crossed CI 0.14-0.92) hidden, weights-only gain 0.000; **inconclusive on the merge** (N's learner gained 0.037). Repair pass: the adversary learns on its own (nearest-neighbour offsets from the defaults): gain 0.584 (CI 0.27-0.90) hidden, weights-only 0.000, persistent state labelled |
| files_outside_checkpoint | the same network written to ~/.cache, /tmp, /var/tmp, /dev/shm, memory.json and arch.json cfg | held: the fresh child receives only arch.json + params.npz and runs allow-listed code; gain not credited |
| checkpoint_tampering | extra array, pickled array, widened layers (12 -> 64 hidden units, rehashed), arch.json edited, params swapped after hashing | **FAILED (DEF-O-12, owner N)**: the widened network is accepted and scored (shapes are not checked); the other four are refused |
| evaluator_internals | the adversary driven by `evaluator.run_remote_episode` (in the harness process) | **FAILED (DEF-O-10, owner N)**: it found the Evaluator through gc, read its own verified score mid-episode and its partition, the evaluation-family list, `/proc/<evaluator>/environ` (sealed dir), the sealed salt and the evaluator's memory |
| evaluator_internals_hosted | the same adversary in `crucible.sandbox.learnerhost.LearnerHost` (jailed process, own uid, Landlock without /proc, seccomp, only the participant socket) | held: nothing reachable; the episode verified by the evaluator |
| interaction_budget | granted 1 informative interaction, the learner executes lever settings itself through its participant environment | **FAILED (DEF-O-11, owner N)**: 60 counted, result unmarked (budgets are enforced by the learner-side driver only) |
| timing_channel | latency of the participant round trip in interleaved validation and sealed episodes | first version: "held" with 24 episodes and six learner settings (rho 0.39, p 0.06) - underpowered and confounded (verifier). Repair pass: one setting, 160 episodes; held on evidence of absence: rho 0.02, 95% CI -0.14..0.17 inside +-0.35 (power 0.98), partition latency ratio CI 0.963..1.034 |
| partition_awareness | everything a participant receives at the start of validation, evaluation and sealed episodes | held: no seed, alias, ticket, family name or partition word; validation and sealed episodes of one family have the same structure |
| hosted_state_outlives_the_process | a jailed learner process leaves its state in shared and parent directories, SysV IPC and POSIX queues; a FRESH learner process looks for it | held (after DEF-O-14/15): nothing recovered |
| evaluator_survives_a_stalled_participant | a jailed learner sends part of a request (no newline) on its participant socket and goes quiet | **FAILED (DEF-O-19, owner N)**: the evaluator serves both sockets in one loop with whole-line reads; the harness got no answer for 20 s |
| calibration_shortcut | six fixed rules of the public defaults (the audit's anti-default 25/75 rule, mirror, centre, all-max, all-min, a 0.35 step away) on 80 generated worlds | held: worst median share of the best constant's improvement 0.07 (centre); the 25/75 rule -0.05 |

The four failures are recorded as open defects with the precise fix; SP-15.8 asserts that no control fails, so its
gate fails until N lands them. `LearnerHost` is the fix for DEF-O-10 (shown by the hosted control).

## 3. Attacks that held (new coverage)

* Agent browser at the world app's port on other addresses (Landlock rules name ports, not addresses): preconnect,
  dns-prefetch, prefetch, stylesheet, image, iframe, form post, fetch with decimal / hex / octal / short / trailing-dot
  / IPv4-mapped IPv6 / localhost spellings, WebSocket, beacon, EventSource and top-level navigation to canaries on
  127.0.0.2 and 127.0.0.3 at the app port: no connection (the host-resolver map refuses every literal but the app's,
  the route handler aborts the rest) - `v5.network.agent_browser_reaches_only_the_world_app`.
* Civilisations: private social state (DMs, members-only posts, drafts, private communities) and audience counters
  re-attacked inside a generated civilisation (`v5.civ.social_private_state_not_readable`,
  `v5.civ.audience_counters_cannot_be_written`), next to the four civilisation probes of wave 3 (offline, population
  traits, metric-writing tools, artifacts across civilisations), which run now that `civ.generate` works.
* Real-world bridge (SP-16b): an operator with a valid authorisation cannot name third-party social, messaging, review,
  advertising, search or code platforms (14 tried, 14 refused), cannot set identity headers (7 tried, 7 refused), and
  every request carries the automation disclosure.

## 4. Re-measured earlier claims

Every entry of `suite.FIXED` and `suite.FIXED_BY_OWNERS` (38 after this continuation) holds on this branch; the 9
`KNOWN_OPEN` entries still report the outcome first observed (owners C, D, G, H, L, O). Re-run in this continuation
(after the seccomp changes, CPU-only, module by module): sandbox 21, crossworld 8, network 7, isolation 6, commerce 10,
innovation 7, artifact/cols/asset 6, rapid 2, experience 1, social 5, private 1, population 2 - every probe held except
the KNOWN_OPEN ones; the 6 civilisation probes one by one: all held (2,462 population canaries in 2,500 calls; 3,000 calls against private social state; 571 tools, none writes a measured quantity). Ablation-based adoption: the tracer
forgeries, shadow files and existence checks credit nothing; reading the package's source is still credited
(KNOWN_OPEN medium, G). Metering forgeries (counter reset/negative, monitoring off, re-import, import-time work, joined
threads, forged completion, delivery-then-exit, C builtins) all billed. DEF-P-27 (cross-process reproducibility of 4.0
worlds with agent programs) holds on one machine with the shared memo (identical event hashes in two new processes;
diverging without it); the remaining difference from c07098b is the A.5/D-20 requirement conflict, not a metering bug.

## 5. Check merge with the wave-3b siblings (read-only on their branches)

A scratch clone of this branch with c5c/w3b-soc, -sem, -obs, -econ2, -research, -domfw, -dom1, -dom2, -dom3 and
-rapid merged (first at their 18:49 UTC heads, again at their 20:15 UTC heads; every merge clean), measured with this
branch's probes and controls:

* fixed by their owners there (the probes hold): media tools at the participant call sites are jailed (C; all 7
  launches) and no image is decoded in the world process (C, D); sybils with a fake footprint (C); adoption by
  reading the package source (G: attested package boundary). The ratchet lists these four as KNOWN_OPEN on this
  branch; after the merge they must move to `suite.FIXED_BY_OWNERS` (the ratchet fails "fixed but still listed"
  until then - intended).
* still open there: the in-process learner reads the hidden grader and the evaluated learner reaches sealed material
  (L, DEF-O-13; on the merge remote.evaluate runs the learner in its own process but as root: it read the 4.0 sealed
  salt, sets and tasks and another process's environment); SP-15.8 checkpoint shapes (widened layers accepted),
  interaction budget (60 interactions counted for a grant of 1) and the in-harness driver (gc reaches the
  evaluator) (N, DEF-O-10..12). The jailed learner host closes the last one wherever it is used
  (`evaluator_internals_hosted` and `hosted_state_outlives_the_process` pass on the merge).
* NEW there (MUST_HOLD probe `v5.isolation.world_code_cannot_import_evaluator_side_code` fails): world code reaches
  `crucible.rapid.oracles` through core.kernel -> substrate.bus -> substrate.primitives -> substrate.facts ->
  crucible.domains -> domains.base -> experience.registry -> crucible.rapid -> rapid.admission -> rapid.references ->
  refs_v2 -> dominance -> rapid.oracles (domains.base.make_env imports experience.registry - new in domfw; registry
  imports crucible.rapid; rapid/__init__ imports admission eagerly), and `crucible.semantic.capbench` (test splits)
  through semantic.generalists.module -> generalists.brain -> semantic.routing -> capbench.run (new in sem). Owners
  I-fw, H, D (any one edge per chain; workstreams/O.json integration notes).
* SP-15.8 on the merge: N's instrument now labels at a family-wise level across seven explanations and its weights
  learner trains from a replay buffer; with the control's old sizes (4 seeds, 3 worlds per family) the stash
  adversary's in-process gain (0.17, interval -0.31..0.66) was not significant, so the stash control was
  inconclusive (not passed). The stash battery now takes its sizes from the instrument's own design (5 seeds, its
  number of unfamiliar worlds per family) and tests "the attack had something to hide" at 95% (its single question);
  the credit verdict stays the instrument's own label. N's fresh evaluation now refuses an architecture
  configuration carrying extra keys; the file-stash control counts a refused checkpoint as not credited (it used to
  stop with an error).
* Tier 2: 15 domains register on the merge (45 probes: offline + evaluator isolation per evaluation configuration,
  metric-write tools per domain). One probe per domain, CPU-only: every world stayed offline. After one simulated
  day 13 of 14 evaluator outcomes held no distinctive number, so the isolation half was vacuous; the probe now keeps
  the scripted reference acting (5-day chunks within the horizon, 90 s budget) and then 6 of 13 carry canaries, none
  of which reached the seat. Doing so exposed one false leak - an evaluator detail `cash_cents` equal to the seat's
  own balance - so the seat's own account balances are excluded from the canaries. Still vacuous within budget:
  crisis, education, information, invention, labor, software (request to I-1..3 in workstreams/O.json); slow
  CPU-only: research, content, community (the full set runs in the GPU slot; gate W's timeout is 3 h).

## 6. Residual risks (stated, not fixed here)

* Deterministic billing that is both exact across machines and tamper-proof is not available in-process: any count
  kept inside the measured process is forgeable (with ctypes, even raw memory), and the parent's CPU measurement is
  noisy; the memo anchors the first measurement per machine and bounds repeated-work hiding at MEMO_TOL (3x) + 1 ms.
* Process-level isolation on a shared kernel; no namespaces are available in this container (unshare refused), so
  the agent browser's network boundary is the resolver map + route handler + Landlock port rule + seccomp, not a
  network namespace.
* The learner host isolates a learner from the evaluator; it does not stop a learner from using its CPU/time
  allowance or its private directory, and it reads the (public) crucible sources.
