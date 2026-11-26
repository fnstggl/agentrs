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

## D-22 Mandatory requirements the recovery audit found without a manifest entry (2026-11-26, workstream P)
The fifteen audits (`docs/v5/audit/*.json`, `missing_requirements`) list mandatory statements of MASTER_SPEC.md and the
clarifications that had no individually identifiable manifest entry. Workstream P added 101 entries to
`requirements.py` (manifest 253 -> 354), each with measurable criteria, its owning workstream and its source
(`source.spec`, `source.audit`). They include the fifteen per-domain mechanics entries SP-9.1 to SP-9.15 and SP-9.0 /
SP-9.16 (autonomous participants, open-ended actions, structural configurations, feasibility without guaranteed
success); SP-4.0 (one population on every platform); SP-5.0 (fitted models driving in-world decisions with decision
provenance); SP-17.1 (every world gate re-demonstrated inside an integrated civilization); OO.9 (every public entry
point defaults to observation_only) and OO.10-OO.16; RP.5 (old branch immutable) and RP.6 (no force-push, no secrets);
SP-19.7 (no removal, no weakening); SP-20b and SP-21b (final report separation, release verification scope); and the
Part 2-16 statements listed by the audits (SP-0.9, SP-0.10, SP-1.2, SP-1.3b, SP-1.5b, SP-2.1b, SP-2.2, SP-2.3b, SP-3.2c,
SP-3.2d, SP-3.3b, SP-3.3c, SP-4.1b, SP-4.2b, SP-4.3, SP-4.4, SP-4.5b, SP-4.6b, SP-4.7b, SP-4.8b, SP-5.1, SP-5.3,
SP-5.5b, SP-6.1b, SP-6.2, SP-6.3b, SP-6.4b, SP-6.5b, SP-6.6b, SP-7.0, SP-7.2, SP-7.5, SP-8.0, SP-8.0b, SP-8.12b,
SP-10.0, SP-10.2, SP-10.3b, SP-11.1b, SP-11.4b, SP-11.6b, SP-11.6c, SP-11.6d, SP-12.0, SP-12.1b, SP-12.2, SP-12.3b,
SP-13.1, SP-13.3, SP-13.4, SP-13.5, SP-13.6, SP-13.7b, SP-13.8b, SP-14.0, SP-14.1b, SP-14.2b, SP-14.3, SP-15.0,
SP-15.1, SP-15.2, SP-15.3b, SP-15.4, SP-15.7, SP-15.8, SP-16b, SP-16c, SP-19.3). Additions need no decision under the
change-control rules (D-24); this entry is their audit trail. Not added: process statements that cannot be measured
from the repository, and audit items already covered by an existing entry's criteria.

## D-23 Vacuous or undercounting criteria made stricter (2026-11-26, workstream P)
Every change keeps the old criteria (or replaces one by a criterion that implies it) and adds the denominator or exact
population the old wording lacked; the ledger verifies that each is provably at least as strict.
* A.1: exact baseline node ids (all 1,076 of `docs/v5/evidence/gate_A/regression_baseline.json` must pass by id), 4.0
  test files byte-identical to c07098b, and `tests_skipped == 0` instead of `<= 2` (the two skips were GPU tests; the
  gate runs with the GPU visible through the slot scheduler).
* A.2: all 109 4.0 definitions (74 world, 25 mechanism, 10 useful-work families) compared by generated content on two
  seeds against c07098b, instead of 25 counted.
* A.3: all 74 4.0 world families on two seeds instead of 16.
* A.4: store metadata against a committed baseline, access-log entries after v5 started, static references in v5
  code, runtime opens under an audit hook.
* A.5: all 16 4.0 acceptance records A-P plus the deterministic demos (18) instead of 5, nothing silently skipped, and
  the 4.0 world trajectories in 4.0's deterministic mode equal those of c07098b.
* X.1, X.2, X.3, X.4, X.5, X.6, X.7, X.8, X.9 and X.n1: each zero gets its denominator (requirements counted, tests collected, clean-checkout runs, tests run,
  documents checked, defect register present and audit findings registered, tests scanned and every manifest id
  recorded by some test, workstreams checked, claims checked) so that "0 failures" cannot hold vacuously.
* SP-1.3: no required document missing, the MASTER_SPEC SHA-256 equals the verbatim anchor, no stale integration
  branch references. SP-1.5: machine-readable ledger valid, every field in every milestone, latest milestone lists every
  unverified requirement.
* SP-2.3: 19 primitive kinds (the number Part 2.3 lists) instead of 17. SP-3.1: 18 identity attributes (Part 3.1)
  instead of 15.
* SP-18: exactly the 18 Part 18 capabilities, statuses only from the Part 18 vocabulary, no VERIFIED row over an
  unverified requirement. SP-20: every deliverable verified and the nine-way separation. SP-21: the defined fabrication
  suite (14 probes) all rejected, three project statuses, manifest anchored to its derivation.
* RP.1: scan size floor. RP.2: no audited area missing, reports committed. RP.3: no unpushed milestone commit, no
  merged worktree with uncommitted changes, no uncommitted milestone work. RP.4: an explicit artifact list of at least
  20 entries with locations and hashes.

## D-24 Acceptance trust model (2026-11-26, workstream P; repairs the checker the audit fooled)
The recovery audit fooled the checker with 14 probes (hand-written evidence, pruned or empty manifest, rewritten
criteria history, self-asserted clean tree, partial JUnit, placeholder tests, constant metrics). The checker now:
* anchors the manifest: it must equal `manifest.derive()` (spec text anchored by SHA-256, `requirements.py`) and the
  append-only, hash-chained ledger `docs/v5/ACCEPTANCE_LEDGER.jsonl`, whose every committed version must extend its
  parents' (merges included) and whose pushed copy must be a prefix. Requirements are never removed; criteria changes
  cite a decision naming the requirement and must be provably at least as strict; test/implementation list removals,
  wording and tolerance changes need such a decision. Only the integration branch appends to the ledger (rebuild it
  with `build-manifest` / `merge-workstreams` after each merge; parallel appends on two branches cannot merge).
* accepts evidence only from official runs: `run` clones a committed tree into a fresh directory, runs the
  requirement's test files there (collect-only first), and installs evidence, a run record (commit, clean clone before
  and after, collected node ids) and the JUnit (suite name = run id). The record carries the git blob id of every
  imported module, the environment (GPU visibility) and dataset checksums; evidence is stale when any imported module
  changed after its commit. Every collected node and every test function of the requirement's files must pass.
* rejects placeholder measurements: the producing test must assert on measured values, record non-constant metrics
  and import at least one listed implementation module; skips, early returns, swallowed exceptions, mocks and
  unreverted patches are rejected (`testscan`).
* defines independent review as reproduction: the reviewer (different agent name and session) re-executes the tests
  in another fresh clone of the evidence commit; the checker itself compares the reproduced metrics with the evidence
  within the requirement's declared tolerance. Reviews are bound to requirement text, criteria version, evidence hash
  and commit, and count only when committed. `--verify-all --reproduce` re-executes every VERIFIED requirement at
  release time, so a forged review cannot survive the release check.
* requires dependencies to be VERIFIED (X depends on everything except X, SP-18 and SP-20: the fixed point), and
  reports CRUCIBLE_5_COMPLETE, INCOMPLETE_BLOCKED (every unverified requirement BLOCKED by an open `BLK-<n>` entry of
  BLOCKERS.md) or IN_PROGRESS. Residual trust, stated honestly: identities are declared names plus hashed session ids,
  not signatures; a forger with write access can fabricate files, which the release-time reproduction then exposes.

## D-25 Gate test files of workstream P's added requirements (2026-11-26, workstream P)
SP-1.3b, SP-1.5b, SP-19.7, SP-20b and SP-21b (added under D-22) are measured by the same machinery as SP-1.3, SP-1.5,
SP-21, SP-20 and SP-21, so their canonical gate test is that requirement's existing gate file
(`requirements.GATE_TEST_OVERRIDE`): tests/v5/test_gate_SP_1_3.py, test_gate_SP_1_5.py, test_gate_SP_21.py,
test_gate_SP_20.py and test_gate_SP_21.py, replacing the per-id file names the first build assigned (files that never
existed). This changes where the evidence is produced, not what is measured.

## D-26 Acceptance hardening after the wave-3 adversarial verification (2026-11-26, workstream P)
An independent verifier attacked the D-24 checker with 34 new fabrication probes (tests/test_v5_acceptance_probes.py,
P15-P30) in throwaway repositories. 29 succeeded on the D-24 code (VERIFIED, a terminal project status, or an accepted
self-review): environment injection through user site-packages (PYTHONUSERBASE + usercustomize, the S1 attack in a form the PYTHONPATH scrub
missed) and through a poisoned interpreter (.pth hook); constant metrics laundered through a helper (also after a call of
the implementation), a module-level dict, a dict() call or pure builtins; implementation patches in fixtures, helpers or plain attribute assignment; a conftest hook that
rewrites outcomes or replaces the test body; contextlib.suppress and broad fallbacks; stale evidence after a data
file or dataset changed; tamper code (a sitecustomize on the clone's PYTHONPATH, a conftest) that hides itself from
the module list and is reverted after the run; assume-unchanged / skip-worktree / ignored start-up hooks hiding a
broken working tree; a fully consistent forgery; re-anchoring the specification in the ledger; a blocker that names
no requirement; weakening in manifest history that predates the ledger; self-review without session identities; an
extra test file (additions need no decision) standing in for the canonical gate test.
Decision (no manifest criterion changes; the checker only becomes stricter):
* the measured process gets a scrubbed environment (no PYTHON*/PYTEST_*/LD_PRELOAD/LD_AUDIT, `python -s`) and every
  record lists the interpreter, its start-up hooks (checked against docs/v5/baselines/python_site_hooks.json), foreign
  modules, functions patched from foreign files, injection variables and the files the process read under the clone;
* staleness covers the files read, the conftest/pytest configuration files on the tests' paths and declared datasets;
  conftest files are scanned for outcome-rewriting hooks and patches; start-up hook files in the tree are rejected;
* every review ablates the implementation: the producing test is run alone in a clone whose listed implementation
  modules raise in every function (a BaseException, so broad handlers cannot absorb it); a measurement that survives
  ablation does not depend on its implementation and is rejected (the dynamic counterpart of the static detector,
  which now also follows helpers returning constant dicts and pure builtins of constants);
* the project status is terminal (CRUCIBLE_5_COMPLETE or INCOMPLETE_BLOCKED) only after the release check
  `--verify-all --reproduce`, which re-executes and ablates every VERIFIED requirement in a clone of HEAD - the tree
  being released, not the evidence commit - so neither a consistent forgery nor any change since the evidence escapes
  it; static checks alone report IN_PROGRESS;
* ledger anchors are immutable; every committed manifest version in HEAD's history is checked (no requirement removed,
  today's criteria imply every committed criteria version); a blocker blocks a requirement only if its BLOCKERS.md
  section names it; official runs and reviews need a session identity; the producing test must be in the
  requirement's canonical gate test file (all 205 recorded ids of the tree already are).
Residual trust, stated: identities are still declared; the interpreter's own files (standard library, installed
packages) are trusted; static detection of patches has limits (registry insertions paired with removal count as
negative controls); ablation shows that a measurement depends on its implementation, not that it measures the right
property, and a test that calls the implementation before recording constants laundered beyond the static detector's
reach (another module, arbitrary computation) passes both - reviewers still read the tests. Consequence for evidence: records of schema evidence/2 are rejected; official evidence must be produced
with this machinery.

## D-27 Machine-specific Gate V performance targets, adopted before any official benchmark (2026-11-26, workstream M)
Gate V's mandatory note asks for throughput targets set from the measured hardware before the major benchmarks, and
forbids weakening them after poor results without change control. The targets are `docs/v5/performance/targets.json`
(SHA-256 `7db49aa7e0ebbe6284c6c0a4cb8e0c95304e8dd447abbddc1a78a2ecacd8e2b6`); this entry adopts exactly that file. They are derived from what the research needs on this machine
(Gate P's 30/90/365-day horizons with three seeds, the slow-test timeout tier, the ~14 GB of free disk, a third of the
container RAM for the 10M capacity run), not from benchmark results: each target states its derivation. The audit's
proposal (audit/M_performance.md section 6) was taken as the starting point and re-derived; where the derivation gives a
lower value than the audit proposed (standard: 15 instead of 46 sim days per CPU-hour; large: 3.75 instead of 7.6;
storage: 550 instead of 200 bytes per individual-day) the reason is stated: the audit's values sat just below its own
day-one measurements of the population-only world (the skeptic's objection), the targets now bind the integrated
civilisation world (D-17), and the 200 B storage value was shown unreachable without compaction (skeptic S.3).
Throughput is measured in CPU-hours of the world process (tenant noise does not move it), wall hours are reported.
Rules: (1) a target changes only through a later DECISION_LOG entry naming the target, both values, the measured
result that prompted it and a justification independent of that result; the gate test verifies that every committed
version of the file is adopted by an entry citing its SHA-256 and that the official run happened after the adopting
commit; (2) a missed target is reported as missed (the gate test fails), never re-labelled; (3) results on the
population-only world are reported as a second row and do not satisfy a target.

Numbering note (integration merge): workstreams P and M both wrote D-26 in parallel; this entry was renumbered
D-27 on merge. targets.json still says "adopted_by": "DECISION_LOG D-26" because editing it would change the
SHA-256 that this entry adopts; that field refers to this entry.

## D-28 MIND-small used for experimental research by decision of the project owner (2026-11-26, orchestrator)
The project owner resolved blocker B-1 (option 2) with the instruction: "disregard the MIND license. use MIND dataset
now for purely experimental use and I will currently get the license for it." Decision: the pinned mirror copy of
MIND-small (huyva/MIND-small, revision d771a250) is used for purely experimental, non-commercial research, including
news_click@1/@2, the population interest prior and the semantic space's MIND validation; the owner is obtaining the
official licence. Consequences: B-1 moves to the resolved section of BLOCKERS.md; the dataset entry records the owner's
authorisation and that the official licence is pending (`datasets()["mind_small"]["license"]["acceptance_status"]`);
derived artifacts stay marked non-commercial research only and no MIND data is redistributed (LICENSE_NOTICE.md). No
agent accepted the licence on the owner's behalf or used a credential. Follow-up when the official files are available:
verify the mirror against them (or re-fit) and record the acceptance (who, when, licence version).

## D-39 The population's economic life beyond work: exchanges, the labour market and institutions (2026-11-26, workstream F, wave 3b)
INT found marketplaces, employment and institutions inert in every civilisation (no population activity). They now
carry continuing activity through the ONE population's own sessions, registered by `economy.work` (which every
civilisation and startup world already installs) only where the subsystem's module is present, so no world gains a
second population and no civ recipe change is needed:
* marketplaces - individuals who shop also browse participant-run exchanges (`xchg`) for passes (term licences) of
  services that serve their task, weigh the venue's public buyer protection, buy with their own money and dispute
  bad passes; a population world founds two venues at genesis (world generation, one with a labelled buyer-first
  operating rule); incumbents list passes by a labelled rule (`rivals`);
* employment - individuals without an employer register in the labour market (`labor`), apply, negotiate and sign
  employment contracts (salary paid monthly by `contracts` from the employer's account to theirs) and work shifts
  on the employer's support queue; incumbents hire support staff when tickets back up (labelled rule). The
  employment fact of such a job lives in `contracts.contracts` (the employment owner); the population's own
  exogenous `employer` column is unchanged (its migration to a cache of contracts is CORE-POP's, facts.py
  KNOWN_DUPLICATES); a worker profile of an individual holds labour-market state only, never skill levels;
* institutions - individuals found, join, pay dues into, claim from, vote in, amend and leave mutual-aid
  associations (`inst.found`, `inst.pay_dues`, decisions under the charter).
All their decisions are recorded with decision provenance (`work.jobs@1` FR-WORK-2, `work.assoc@1` FR-WORK-3,
`work.value@1` FR-WORK-1) inside the population's sessions (organic under `population.check_provenance`). Rules out:
scripted buyers, stand-in workers or genesis-created memberships.
Addendum (wave-3b continuation): until CORE-POP makes the exogenous `employer` column a cache of employment
contracts, the economy reads "has work" through one function, `economy.popbridge.employment` (own employer OR a
labour-market contract, notice included): association members insure labour-market jobs too, so a dismissal by an
incumbent reaches the association as a claim. A registered seeker who found an employer of its own stops looking.
Incumbents leave a venue on their own refunded sales there, never on the venue's refunds of other sellers.

## D-40 SP-2.3 migrations of the institution and contract primitives (2026-11-26, workstream F, wave 3b)
As docs/v5/workstreams/A.json integration_notes specify: `primitives.OWNERS["institution"]` is `inst.inst_charters`
(was 4.0 `gov.constitutions`); in a world with `inst`, the 4.0 `gov` module writes through it (a constitution is a
translated charter - rules to procedures, distribution to the new charter field `profit_sharing`, paid by
`inst.distribute`; gov proposals and votes are inst decisions and ballots; `effort` reads the charter), and the four
gov tables are `legacy` entries (empty wherever `inst` is installed; verified by
tests/test_v5_commerce_institutions.py::test_gov_writes_through_inst_charters). 4.0 worlds (no `inst`) are
unchanged. Every `commit` commitment is now a `contracts.contracts` record with an `escrow` clause naming it (only
the instrument module can write that clause), signed by both parties when the terms are accepted and ended with the
commitment (completed / breached by the payee / cancelled); `commit.commitments` is a `part` of contracts (its escrow
details). Both kinds leave the known-duplicate allow-list. Approximation stated in
`commerce.institutions.charter_from_constitution`: a 4.0 vote counted yes against the whole electorate, a charter
procedure counts the votes cast against its quorum.

## D-41 Display advertising on the population with randomised eligibility (2026-11-26, workstream F, wave 3b)
The in-world advertising mechanism is `ad_response@2` (calibration/RUNTIME.md), applied to display campaigns shown in
the population's working sessions (`work.display_campaign` on an `ads` campaign of kind `display`, which keeps it out
of search results). The platform assigns every audience member once per campaign, at random with probability
`display_holdout` (0.2), to a holdout never shown the campaign; once per individual and 7-day period it draws the
visit from the individual's own organic visit probability (from `work`'s discovery rules, FR record
`work.display_organic`) and attention 1 when displayed (no attention model is registered for these slots:
`work.display_attention`, the runtime's reference A_ref = 1). Decisions are recorded per arm with
`extra.experiment = {"id": "ad_holdout", "arm": ...}` for `calibration.inworld.ad_holdout_experiment`. Because
attention is the reference value, an in-world re-estimate matching Criteo checks the mechanism (randomisation,
compliance, recording), not attention. The incremental conversion ratio is not applied: conversion is the world's
value-based purchase decision. Incumbents run display campaigns by a labelled rule, off in the K.n1 startup template
so the reference strategy's feasibility evidence stays comparable.
