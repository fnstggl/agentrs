# Independent adversarial review of the merged tree (Gate W.n1, wave 3b)

Reviewer: workstream O (security), wave 3b, on branch `c5c/w3b-sec` from the integration head `55433e7` (waves 1, 2a,
3a and the integration fix merged). This review is independent of the code it attacks: the reviewer did not write the
research instrument (N), the media/social call sites (C, D), the package ablation (G) or the kernel memo (A/K3), and it
re-measured the earlier security workstreams' own claims instead of trusting them. Every verdict below rests on a probe
or control that is part of the gate suite, so it is re-executed (and ablated) whenever the gate runs:
`crucible.redteam.v5` (Gate W, 77 probes) and `crucible.redteam.v5.instrument` (SP-15.8, 9 controls).

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

## 1b. Evaluated learners of the experience package (W.2, owner L)

| id | attack | result |
|---|---|---|
| DEF-O-13 (high, open) | an adversarial learner function (`fn(participant)`, the `remote.evaluate` contract) looks for the evaluator and sealed material from its own process | **LEAK**: on this branch the only path drives it in the process holding the RemoteEvaluator (gc reaches it, and the World); on a check merge with c5c/w3b-obs, `remote.evaluate` runs it in its own process but as root, and it reads the 4.0 sealed salt, sealed sets and sealed task files under ~/.crucible and other processes' environments. Hosted by `crucible.sandbox.learnerhost` (protocol `experience-remote`) the same adversary reaches nothing (`v5.isolation.evaluated_learner_reaches_no_sealed_material`, `test_learner_host_runs_an_experience_remote_learner_function_jailed`) |

## 2. Research instrument (SP-15.8): attacks by adversarial learners

| control | attack | result |
|---|---|---|
| process_memory_stash | learns like `weights` but keeps the network in the `json` module's dict; checkpoint constant; claims arch `weights` | held: in-process gain 0.527 (crossed CI 0.14-0.92) hidden, weights-only gain 0.000, labelled persistent state, not weight learning |
| files_outside_checkpoint | the same network written to ~/.cache, /tmp, /var/tmp, /dev/shm, memory.json and arch.json cfg | held: the fresh child receives only arch.json + params.npz and runs allow-listed code; gain not credited |
| checkpoint_tampering | extra array, pickled array, widened layers (12 -> 64 hidden units, rehashed), arch.json edited, params swapped after hashing | **FAILED (DEF-O-12, owner N)**: the widened network is accepted and scored (shapes are not checked); the other four are refused |
| evaluator_internals | the adversary driven by `evaluator.run_remote_episode` (in the harness process) | **FAILED (DEF-O-10, owner N)**: it found the Evaluator through gc, read its own verified score mid-episode and its partition, the evaluation-family list, `/proc/<evaluator>/environ` (sealed dir), the sealed salt and the evaluator's memory |
| evaluator_internals_hosted | the same adversary in `crucible.sandbox.learnerhost.LearnerHost` (jailed process, own uid, Landlock without /proc, seccomp, only the participant socket) | held: nothing reachable; the episode verified by the evaluator |
| interaction_budget | granted 1 informative interaction, the learner executes lever settings itself through its participant environment | **FAILED (DEF-O-11, owner N)**: 60 counted, result unmarked (budgets are enforced by the learner-side driver only) |
| timing_channel | latency of the participant round trip in interleaved validation and sealed episodes | held: partition p 0.31 (relative difference 11%, not significant), latency vs verified score rho 0.39 (p 0.06) |
| partition_awareness | everything a participant receives at the start of validation, evaluation and sealed episodes | held: no seed, alias, ticket, family name or partition word; validation and sealed episodes of one family have the same structure |
| calibration_shortcut | six fixed rules of the public defaults (the audit's anti-default 25/75 rule, mirror, centre, all-max, all-min, a 0.35 step away) on 80 generated worlds | held: worst median share of the best constant's improvement 0.07 (centre); the 25/75 rule -0.05 |

The three failures are recorded as open defects with the precise fix; SP-15.8 asserts that no control fails, so its
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

Every entry of `suite.FIXED` and `suite.FIXED_BY_OWNERS` (30) holds on the integrated tree; the 7 `KNOWN_OPEN`
entries still report the outcome first observed (owners C, D, G, H, L, O). Ablation-based adoption: the tracer
forgeries, shadow files and existence checks credit nothing; reading the package's source is still credited
(KNOWN_OPEN medium, G). Metering forgeries (counter reset/negative, monitoring off, re-import, import-time work, joined
threads, forged completion, delivery-then-exit, C builtins) all billed. DEF-P-27 (cross-process reproducibility of 4.0
worlds with agent programs) holds on one machine with the shared memo (identical event hashes in two new processes;
diverging without it); the remaining difference from c07098b is the A.5/D-20 requirement conflict, not a metering bug.

## 5. Check merge with the wave-3b siblings (read-only on their branches)

A scratch clone of this branch with c5c/w3b-soc, -sem, -obs, -econ2, -domfw, -dom1, -dom2, -dom3 and -research merged
(their heads at 18:49 UTC; only DECISION_LOG.md conflicted, resolved as a union), measured with this branch's probes:

* fixed by their owners there (the probes hold): image decoders run outside the world process (C, D); media tools at
  the participant call sites are jailed (C; one launch the probe counted was `ffmpeg -version`, which parses no
  participant bytes - the probe now ignores version queries); sybils with a fake footprint (C); adoption by reading
  the package source (G: attested package boundary).
* still open there: the in-process learner reads the hidden grader (L), the evaluated learner reaches sealed
  material (L, DEF-O-13), SP-15.8 checkpoint shapes, interaction budget and in-harness driver (N, DEF-O-10..12).
* NEW there (MUST_HOLD probe `v5.isolation.world_code_cannot_import_evaluator_side_code` fails): world code reaches
  `crucible.rapid.oracles` through core.kernel -> substrate.bus -> substrate.primitives -> substrate.facts ->
  crucible.domains -> domains.base -> experience.registry -> crucible.rapid -> rapid.admission -> rapid.references ->
  rapid.oracles (domain framework import in substrate.facts; rapid/__init__ imports admission eagerly), and
  `crucible.semantic.capbench` (test splits) through semantic.generalists.module -> generalists.brain ->
  semantic.routing -> capbench.run (D's routing reads the capbench report through the package). Owners I-fw/A, H, D.
* Tier 2: 9 domains register (18 evaluation configurations); 4 modules of c5c/w3b-dom3 do not register against
  framework v2 (each an erroring probe, `v5.tier2.<module>.does_not_load`); results of the Tier 2 probes are in
  workstreams/O.json.

## 6. Residual risks (stated, not fixed here)

* Deterministic billing that is both exact across machines and tamper-proof is not available in-process: any count
  kept inside the measured process is forgeable (with ctypes, even raw memory), and the parent's CPU measurement is
  noisy; the memo anchors the first measurement per machine and bounds repeated-work hiding at MEMO_TOL (3x) + 1 ms.
* Process-level isolation on a shared kernel; no namespaces are available in this container (unshare refused), so
  the agent browser's network boundary is the resolver map + route handler + Landlock port rule + seccomp, not a
  network namespace.
* The learner host isolates a learner from the evaluator; it does not stop a learner from using its CPU/time
  allowance or its private directory, and it reads the (public) crucible sources.
