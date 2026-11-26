# Crucible 5.0 decision log

Each decision has an id, the date, what was decided, why, and what it rules out. Acceptance criteria may change only
under a decision recorded here and cited in the manifest's `criteria_history`.

## D-1 Base and branch (2026-11-25)
Crucible 5.0 is built on `c07098b` (feat/crucible-4, also the default branch) in branch `feat/crucible-5-openworld`,
worktree `.worktrees/crucible-5`. It is the latest Crucible implementation: `feat/cam-7.0-unified` (b775cba) contains
it and changes only `superlearner/`.

## D-2 Independence from CAM (2026-11-25, Clarification 1)
Crucible does not import, merge or wait for CAM. A read-only detached checkout of b775cba (`.worktrees/cam7-ref`) is
used only to check interface compatibility. Gate S and R.7 are restated as architecture-independent learner
integration (manifest entries carry the clarification); SP-11.6 requires a documented, tested path for future learners.

## D-3 Columnar population state with batch events (2026-11-25)
Population-scale state lives in `StateStore` dense arrays; cohort decisions are written to content-addressed array
blobs and recorded as one `cols.batch` event carrying the blob's SHA-256. Why: 10M identities and tens of millions of
consumption decisions cannot be dictionary records or individual kernel events at about 40k events/s, yet every
decision must stay recorded, replayable and hash-chained. Rules out: multiplied counters, unrecorded aggregate updates.

## D-4 Content-addressed artifact store on the world store's blob table (2026-11-25)
Immutability and hashing are enforced by `crucible.substrate.artifacts`; bytes are kept out of event payloads (4.0 code
blobs in events would make media logs and snapshots grow without bound).

## D-5 Exact distinct-viewer accounting by bitsets; estimated mode only when labelled (2026-11-25)
Per-content viewer and exposure bitsets make unique counts exact and recountable from batch events. An estimated
population mode exists for statistics beyond the materialised population and is always labelled `estimated`.

## D-6 Default semantic model Qwen3-0.6B, chosen by benchmark (2026-11-25)
Qwen3-0.6B is local and within the 0.5-1B target. The final choice follows the SP-3.2b benchmark against smaller
specialised models and Qwen3-1.7B; if a smaller model meets a capability, it is used for that capability.

## D-7 Compute through slots (2026-11-25)
All heavy commands run through `crucible.scale.slots` with a budget of 9 CPUs, 24 GB of GPU memory and 8 jobs on this
machine (11.5-CPU quota, shared GPU). Usage is logged to `/tmp/crucible-slots/usage.jsonl` and summarised for 13.8.

## D-8 Evidence-checked acceptance (2026-11-25)
The manifest is generated from MASTER_SPEC Part 17 (170 gate items), the gates' further mandatory sentences (21) and
mandatory requirements elsewhere in the spec (50): 241 entries. `--verify-all` recomputes every verdict from evidence;
evidence is produced only on the integration branch from a clean tree; reviews must be by someone other than the
author and bind to the evidence's hash.

## D-9 New Tier 1 families never reuse 4.0 validation or sealed families (2026-11-25)
4.0's `val`/`test` mechanism families and the spent sealed set `ow-final` stay where they are. Tier 1 training growth
comes from new primitives, new engines and new compositions; the 20+ v5 evaluation families are new and partitioned.

## D-10 Semantic participant model sizes (2026-11-25, from the SP-3.2b benchmark of ws-D2)
The standard implementation uses Qwen3-0.6B for every semantic participant (H.n1 asks for 0.5-1B or smaller). The
benchmark found Qwen3-1.7B to be the smallest effective model for programming, negotiation and strategic decisions on
held-out templates; an optional escalation to it for those three capabilities is allowed and is reported as exceeding
the 0.5-1B target. Trained small heads win on seen templates only and are not used where they fail held-out ones.

## D-11 Ownership of the `product` fact (2026-11-25, open)
`primitives.OWNERS` maps `product` to 4.0 `market.listings`, while ws-F1 keeps v5 products in `products.products`
without declaring `shared_facts`, so `check_no_duplicates` cannot see the overlap. To resolve before Gate O: `product`
moves to `products.products`; a 4.0 listing becomes the kind `listing` (a sales offer that may reference a product).

## D-12 Integration merges (2026-11-25)
Workstream branches are merged with `--no-ff` (never squashed: SP-5.5 checks the order of plan and fit commits).
Only append-only documents (FIDELITY_REGISTER.md, REPRODUCE.md, pyproject extras) are resolved automatically by
keeping both sides; Python conflicts are resolved by hand. An automatic resolution of `commerce/modules.py` broke the
file once (fixed in a4d06b3), which is why.

## D-13 Domain framework owned by the orchestrator (2026-11-25)
`crucible.domains.base` was written by the orchestrator before the domain workstreams started, so twelve domains
built in parallel share one registration, split and reference-isolation scheme. Evaluation configurations use the
splits `v5eval_compositional` and `v5eval_structural`, distinct from 4.0's `val` and `test`.

## D-14 Recovery audit and the "one world" redesign (2026-11-26, Clarification 3)
Fifteen independent auditors (one per workstream) and fifteen skeptics re-examined the recovered implementation
(`docs/v5/audit/*.md|json`). Their common finding: most v5 subsystems work in isolation but are not one world. There
are at least six disjoint populations (4.0 consumers/shoppers, commerce `patrons`, social `crowd`, innovation adopters,
`pop`, semantic agents); the social engine was gated against a stand-in population and cannot run with `pop`; loading
all modules into one kernel world runs 30 days with every v5 commerce/social/innovation table empty; fitted models are
loaded then discarded; content is judged by lexical hash vectors; demand exists only for predefined formats with
hidden-truth verifiers. Decision: keep the kernel, substrate, artifact store, columns, audience accounting,
recommenders, studio, calibration fits, commerce ledgers and Tier 1 engines where audited sound, and REDESIGN the
integration layer around D-15 to D-21. Rules out: gating any v5 requirement on a stand-in population or a lean
single-purpose world.

## D-15 One population is the source of every simulated human decision (2026-11-26)
`crucible.population` (`pop`, columnar) is the only population in v5 Tier 2 and Tier 3 worlds. Subsystems publish
*opportunities* (content via platform feeds/search, product offers and subscriptions, services and packages,
communities, messages, jobs, courses, marketplace listings); one decision engine allocates each individual's finite
attention, time and money across them from the individual's own state, using fitted models where data exist
(Gate F) and recorded, fidelity-registered models elsewhere. Every decision is a recorded batch event carrying the
decision-model id and version (decision provenance). `crowd`, `patrons`, innovation adopters and 4.0 consumers remain
only for 4.0 regression families and their own historical tests.

## D-16 One shared semantic space (2026-11-26)
Content is represented only through pretrained encoders (text, image, audio, video, code), cached per
(artifact hash, encoder, version, derived preprocessing), projected into one shared interest space in which
individuals' interests also live (initialised from embeddings of real topic corpora, updated by experience).
Artifact-intrinsic features (e.g. language-model coherence/perplexity, information content, duration, production
features) are computed from the artifact, never declared by its creator. Lexical random indexing is retired from
distribution and choice.

## D-17 Tier 2 domains are configurations of the civilization generator (2026-11-26)
`crucible.civ` generates every v5 world: a Tier 3 civilization enables every subsystem; a Tier 2 domain configuration
enables a subset, chooses mechanism variants, couplings, topology and population, and adds the domain objective,
evaluator and hidden reference. The canonical domain contract is `crucible.domains.base.Domain`; `crucible.experience`
consumes it (the registry contract of INTERFACE_CONTRACTS section 9 is superseded).

## D-18 Usefulness through activities, not hidden truth (2026-11-26)
Participants perform activities (consume, work, build, sell, learn, research) whose outcomes depend on capabilities and
resources. Services and packages provide capabilities through typed interfaces; participants try them on their own
activity inputs and adopt them according to their own measured outcomes (time, cost, failures, downstream acceptance).
Needs arise from activity bottlenecks through one reusable relation graph. No participant reads a generator's hidden
truth.

## D-19 Observation-only everywhere; evaluator oracles are not participant tools (2026-11-26, Clarification 2)
`reward_mode` defaults to `observation_only` in every constructor, adapter and wrapper (`check_mode(None)` returns it).
In that mode any tool or field that exposes an evaluator-computed quantity (goal progress, verified quality, integrity
verdicts, related-party flags, evaluator metrics) is unavailable to participants. Leakage is tested by information-flow
probes that try to recover verifier values, not by key names.

## D-20 Metering outside the measured process (2026-11-26)
Sandbox CPU/memory/instruction metering and adoption tracing are measured by the parent (rusage/procfs, parent-side
tracing), clamped non-negative, and recorded as external events so replay stays exact. Nothing the agent process
reports is trusted for billing, capacity, program cost or adoption.

## D-21 Kernel and substrate repairs (2026-11-26)
Module `on_load` runs after the store is attached (cross-process resume); typed-bus subscribers draw from their own
derived RNG streams so observers never perturb a trajectory (required for counterfactual removal); batch blobs get
retention/compaction; the duplicate-fact check becomes mandatory over all tables (not opt-in), and D-11 is resolved
(`product` owned by `products`).
