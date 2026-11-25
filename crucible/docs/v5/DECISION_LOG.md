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
