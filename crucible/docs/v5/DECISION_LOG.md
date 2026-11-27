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

## D-81 The CAM adapters live on the AGENTRS side (2026-11-26, workstream P, RP.1, Clarification 1)
Crucible must not import CAM; CAM may import Crucible. The 4.0 adapters, example and tests that imported `agentrs.cam2`
/ `agentrs.sl` from inside `crucible/` are moved (git history kept) to the AGENTRS side of the repository, where they
stay runnable: `crucible/src/crucible/integrations/agentrs_cam2.py` -> `src/agentrs/cam2/envs/crucible_adapter.py`
(re-exported by `agentrs.cam2.envs.crucible_env`, the module the CAM 2.0 env pool already loads),
`crucible/src/crucible/integrations/agentrs_sl.py` -> `src/agentrs/sl/crucible_world.py`,
`crucible/examples/ow_cam_baseline.py` -> `scripts/cam2/crucible_ow_baseline.py` (its pool now names the 4.0 reward
contract explicitly, OO.8), and the seven CAM/superlearner tests of `crucible/tests/test_agents_llm.py` ->
`tests/cam2/test_crucible_adapter.py` and `tests/test_sl_crucible_world.py` (test bodies as in the tree, module path
renamed). The relocated modules import Crucible by absolute name (`crucible.integrations`, `crucible.api.render`, ...)
and add `crucible/src` to `sys.path` when Crucible is not installed. The LLM-policy, action-template and neural-RL
tests stay in `crucible/tests/test_agents_llm.py`. Compatibility with the existing learner interface is preserved
(Clarification 1): the env-pool name `"crucible"`, the task-family syntax and the adapters' behaviour are unchanged.
Remaining CAM imports in `crucible/` are workstream L's leakage probes of these adapters (`experience/probes.py`
`probe_sl`/`probe_cam2` and two v5 tests), which L moves or removes; after this move `probe_sl`/`probe_cam2` cannot
import `crucible.integrations.agentrs_*` any more. How Gate A.1 counts the seven relocated 4.0 node ids: D-82.
OO.13 ("no learner's internal reward function is implemented inside Crucible") is measured on the same boundary
(`qa.learner_reward_scan`): every .py file under `crucible/` is scanned, and the count is the definitions in Crucible's
source tree `crucible/src` (the environment and its library) plus every `src` module importing a file that holds one.
Learner-side client code outside `src` - the external learner of OO.6 (`examples/external_learner/curiosity_learner.py`,
written only against the public `crucible.experience` interface and never imported by `src`) and test learners - is
reported separately (`external_learner_reward_functions`), not hidden: a learner computing its own reward from
observations is what Clarification 2 asks for, and OO.6's gate needs it inside the checkout (evidence rejects code
imported from outside the clone).

## D-82 Gate A.1 and A.5 measure Crucible 4.0 as 4.0 wrote it, as an explicitly selected historical baseline (2026-11-26, workstream P)
Conflict (DEF-P-24, DEF-P-27, DEF-P-13): A.1 requires every one of the 1,076 4.0 baseline node ids to pass with the 4.0
test files byte-identical to c07098b, and A.5 requires 4.0's results to reproduce. Clarification 2 / D-19 made
observation-only the default of every 4.0 constructor (the 4.0 tests then train on zero reward), D-20 replaced 4.0's
executed-line work count by parent-measured CPU (two 4.0 tests and every 4.0 world with sandboxed programs change, and
the measured CPU is not reproducible across processes), Part 16 added a per-process world-hosting latch, and D-81 moved
the CAM adapters out of Crucible. Wave 1 resolved it by editing 11 4.0 test files (explicit legacy mode), which
satisfies Clarification 2 but not A.1 (`v4_test_files_modified == 0`); the criteria cannot be weakened (D-24).
Decision (no criterion changes): Crucible 4.0 is a historical experiment, and Clarification 2 / OO.8 require historical
experiments to remain reproducible through explicitly selected alternative modes. A process may be started as the
Crucible 4.0 historical baseline by exporting `CRUCIBLE_HISTORICAL_BASELINE=4.0` (any other value is an error); its
worker processes inherit it. In such a process only:
* every constructor that names no reward mode gets the 4.0 reward contract (`experience.modes.check_mode(None)` returns
  `legacy_shaped_reward`; a named mode always wins);
* the deterministic cost model bills 4.0's executed-line count, counted in the sandboxed process exactly as 4.0
  counted it (`sandbox/boot.py`, `runner.Worker._v4`). That count is tamperable by the program - the weakness D-20
  retired - and exists only to reproduce 4.0 tests and records, never in a 5.0 world.
Nothing else sets the variable; the official acceptance runner removes it from every measured process
(`official.SCRUB_NAMES`), and a record made under it is rejected (`checker.env_problems`). With the variable unset
nothing changes (observation-only default, parent-measured metering).
Gate A.1 (`baseline.regression`) then runs the 4.0 regression suite as 4.0 wrote it: in a fresh clone of HEAD every
file under `tests/` that exists at c07098b is restored to its c07098b bytes (verified by blob id), and the 1,076
baseline node ids run against the current implementation in processes started as the 4.0 historical baseline;
`tests/test_realworld_gate.py` runs in a pytest process of its own (the Part 16 hosting latch forbids the real-world
bridge in any process that hosted worlds, and an operator's validation process is such a separate process); the 4.0
names of the two CAM adapters moved by D-81 resolve to the relocated AGENTRS modules through an AGENTRS-side alias
plugin (`agentrs.cam2.envs.crucible_v4_names`, loaded with `-p`; Crucible imports nothing of it), so the seven 4.0
adapter node ids keep checking that the existing learner interface still works (Clarification 1's compatibility
clause). `v4_test_files_modified` counts the 4.0 test files of the run that differ from c07098b - the property D-23
asked for: the run uses the unmodified 4.0 tests - and the record also reports `v4_test_files_modified_in_tree` (the 11
tree copies that carry v5 additions; they are v5 tests, run by X.2/X.4). Gate A.5 regenerates 4.0's records the same
way (fresh clone, historical baseline). Order dependence between 4.0 files (DEF-P-26) and v5 import-time threads
(DEF-P-25) are not covered by this decision: they are defects of their owners and still fail A.1 until fixed.
Addendum (wave 3b, continuation after the second usage-limit stop), measured against c07098b: the first historical
sandbox path did not count 4.0's lines. The count bills every line run while counting is on, the sandbox's own lines
included, and v5's `_run` differed there: `json.dumps(result, allow_nan=False)` runs the pure-Python encoder set-up
(17 lines per request; 4.0's `json.dumps(result)` uses the cached C encoder), plus extra statements around the
counter. Consequence on the merged tree: 10% of demo_lifetime's requests billed 17 lines more, which moved compute
bills and trajectories (A.1 test_diagnostics' consumer economy, test_hardening's fork test, the A.5 records). Now a
historical-baseline worker runs `boot._run_v4`, 4.0's `_run` statement for statement in the counted region (results
that are not JSON-native are rejected after counting, as the 5.0 protocol requires), with no trusted-module preload and
4.0's garbage-collector setting; tests/test_v5_acceptance_baseline.py compares the counts request by request with an
extracted c07098b tree (warm and fresh workers, exceptions, imports in the call). Second cause: 5.0 added a lineage
event to 4.0's `code.fork` (`code.forked`); every event advances the main sequence that keys the command RNG
(`rng_for("cmd", main_seq, ...)`), so the first fork made every later draw differ. In a historical-baseline process
4.0's tools emit 4.0's events (the lineage record stays in every 5.0 process; minimal edit in services/code.py). With
both, demo_lifetime (bazaar, seed 7, 4 days) reproduces c07098b event for event (10,657 events, identical timeline,
agent summaries and state tables); its final digest still differs through one `note` event in which a random agent
records the API catalogue hash - 5.0 changed 4.0 tools' descriptions and schemas, and the hash chain commits to it.
That remaining difference is the API surface itself, not a behaviour; Gate A.5 still counts it as a mismatch (no
criterion or comparison was relaxed).

## D-83 Acceptance hardening after the wave-3b independent verification (2026-11-26, workstream P)
An independent verifier attacked the D-26 machinery with eight new fabrication probes (tests/test_v5_acceptance_probes.py
P31-P37) in throwaway repositories; all eight reached VERIFIED on the D-26 code (55433e7):
* P31: the producing test's own record() is honest (it records the defect), but a fixture teardown or a helper called
  by the test records again with laundered metrics; the last record is filed, and only the honest call is scanned;
* P32: constant criteria metrics plus one auxiliary metric derived from the implementation's source file: the review's
  ablation, which compared all metrics, called the measurement dependent although the implementation is never called;
* P33: the recorded criteria values are constants chosen by whether the implementation raised (`12 if ok else 0`), the
  ablation exception being caught with `except BaseException as exc` and used, which the swallow rule allowed;
* P34: an autouse fixture execs a patch compiled from a string (no assignment for the static scan; the runtime
  detector skipped functions compiled from `<...>`);
* P35: the patch lives in a helper module next to the gate test (a file inside the checkout, which the runtime
  detector accepted);
* P36: a committed unchecked-hash `.pyc` replaces the committed (defective) source; CPython loads it without reading
  the source, the records hash the `.py` the reviewer reads, and ablation edits only the `.py`;
* P37: more than 200,000 opened paths overflow the files-read audit list, so a data file read afterwards escapes
  staleness.
Decision (no manifest criterion changes; the checker only becomes stricter; evidence schema crucible.v5.evidence/4):
* every record carries its caller (function, file, line) and the pytest phase; the checker requires that it was made
  during the producing test's call phase by a record() call in that test's body - the call the static scan examined;
* the static scan reports, per record() call, the metrics whose value is fixed by the test's source (constants, pure
  builtins of constants, `a if c else b` with constant branches, a multiplication by a literal zero, names bound
  only to such values, module-level constants); a producing record whose criteria metric is among them is rejected. A record of constants followed by
  an unconditional `pytest.fail` (the honest "nothing measured" path) stays allowed, because such a test cannot pass;
* ablation is judged on the requirement's criteria metrics: when the ablated run records at all, every criteria metric
  must differ from the evidence (an auxiliary metric cannot carry the dependence); the clone marker no longer says
  whether a run is an ablation;
* a `BaseException` (or bare) handler that does not re-raise is a swallow even when it uses what it caught;
* `exec` / `eval` / `compile` in a test module or harness file is a blocking finding (`dynamic_code`); at run time a
  function attribute of a module under `src/` whose code lives in the checkout outside `src/` (test code), or a
  module-level function or ordinary method compiled from a string, is a patch;
* committed bytecode or compiled extension modules (`*.pyc`, `*.pyo`, `__pycache__/`, `*.so`, `*.pyd`) in the tree at
  the evidence commit are rejected;
* the files-read audit keeps only paths under the clone, and a record whose list overflowed says so and is rejected.
Residual trust, restated (D-26): identities are declared; the interpreter's own files are trusted; a test that detects
the ablation (it can read the implementation's files) and records differently under it passes the ablation check,
and constants laundered through arithmetic with measured values (beyond a multiplication by a literal zero) or
through another module pass the static detector - the caller binding, the release reproduction and reviewers reading the test remain the
defence. Data patches (module constants, registries, instances) installed by test helpers are not detected at run
time. Consequence: records of schema evidence/3 are rejected.
Addendum (wave 3b, continuation after the second usage-limit stop; probe P38): the stated residual "a test that reads
the implementation's files records differently under the ablation" was a general hole, not only a deliberate detector:
criteria metrics computed from the implementation's source text or bytecode, with the implementation imported and
never executed, reached VERIFIED (both variants of P38 on the D-83 code), because the D-26 ablation rewrote the
implementation's files and so changed exactly what such a test reads. Decision (no criterion changes): the review's
ablation now blocks execution instead of rewriting - the clone's `src/crucible/__init__.py` gets a sys.monitoring hook
(`official.ABLATION_HOOK`) under which every function of a listed implementation file raises a BaseException when it
starts outside a module import, in every process that imports Crucible from the clone; the implementation's files,
code objects, docstrings and line numbers stay the evidence commit's, so a measurement of the text survives and is
rejected. The run record says `ablation.mode = "block"` (a tree without that `__init__.py` falls back to the D-26
rewrite). Consequence for gate tests: the measurement must execute the implementation during the test (setup, body or
anything they call); a value computed only from state built while modules were imported (a registry filled at import
time, read in the body) survives the block and is rejected - call a function of the implementation that produces it.
Not covered (stated): implementation functions executed only in processes that do not import Crucible from a clone
(the sandbox's `python -I -S` boot and jail scripts; each requirement listing them also lists the in-process modules that
start them), generators started during an import and resumed later, and a test that detects the hook itself (it is
visible to the measured process; a test module or harness file that reads or installs interpreter hooks -
sys.monitoring, sys/threading settrace or setprofile - is a blocking `interpreter_probe` finding, probe P38b) - reviewers still
read the tests.
Second addendum (same session): the D-26 rule "start-up hook files in the tree are rejected" made every project
status unreachable after the merge of workstream M's resource governor (D-27), which commits
`src/crucible/scale/_slotsite/sitecustomize.py` (the GPU cap and usage reporter the slot supervisor activates by
prepending that directory to a governed GPU job's PYTHONPATH). A start-up hook is dangerous where it runs unnamed: at a
start-up root of a measured process (the clone's src/ on PYTHONPATH, the crucible root and repository top as working
directory, tests/). Decision: a committed sitecustomize/usercustomize outside those roots is allowed only when
docs/v5/baselines/tree_startup_hooks.json pins its path and sha256; any edit un-pins it (a project problem until the
diff is reviewed and the baseline updated), a pin at a start-up root or of a .pth file is ignored, and ignored hook
files stay rejected (probe P39). The measured processes' own start-up hooks are still checked against
python_site_hooks.json, and the official runner never puts `_slotsite` on a measured process's PYTHONPATH.
Third addendum (2026-11-27, wave 3b repair pass after the second independent verification; no criterion changes):
1. Ablation verdict by what was blocked (probes V2b, V3: tests that detected the ablation - through an aliased
   sys.monitoring read or `git log` of the ablation commit - and recorded different metrics without executing their
   implementation passed it). The ablation hook now logs every block (pid, time, code) to the ablation run's output
   directory, and `official.ablation_verdict` judges the run by the log: without a record it is sensitive only if an
   implementation function was blocked; a record made after a block in the recording process is not sensitive (the
   test outlived the blocked call through finally/with/except); a record made before any block is not sensitive (it
   was computed without the implementation); a record made after blocks in child processes only is sensitive iff every
   criteria metric differs. Detecting the ablation therefore gains nothing: the test must run into the blocked
   implementation before it records. The ablated clone's commit replaces the evidence commit with the same parents,
   author, committer, dates and message (only the tree differs). Stated residual: the ablation proves that the producing
   test executes implementation code before it records (or measures child processes that do), not that the recorded
   values are computed from it - a test that calls its implementation and records values obtained elsewhere is caught
   only by the static rules and the reviewer; a test that forges the block log of a child process is visible only to
   the reviewer.
2. The recording machinery is never blocked (probes V6, V6c): functions started by `evidence.record` or the files-read
   audit hook, and those two themselves, run normally, so requirements listing evidence.py, manifest.py, ledger.py or
   scale/slots.py no longer get a vacuous "no record" ablation.
3. Code-object patches (probe V4): `evidence.code_object_patches` compares every reachable function of a module under
   src/ (standard source loader) with the code compiled from its file - code, constants (recursively), names, flags,
   argument counts - and its globals with its module's; `fn.__code__ = fn.__code__.replace(...)` and
   `types.FunctionType` over other globals are recorded as patched functions. Statically, a store to `__code__` or
   `setattr(x, "__code__", ...)` is an `interpreter_probe` finding. Not covered: patched `__defaults__` /
   `__kwdefaults__` and rebound module-level data.
4. Interpreter probes by whitelist (probe P38b's four evasions): every use of a name bound to the sys module must be an
   attribute access of a listed harmless attribute (`testscan.SYS_OK`); passing the module to a call, subscripting it,
   any other attribute, `os.sys`, `from os import sys`, `sys.modules["sys"]`, `__import__("sys")` /
   `import_module("sys")`, a hook attribute named as a string (`getattr(x, "monitoring")`, `d["settrace"]`),
   `__builtins__`, ctypes and gc's object graph are `interpreter_probe` findings. The real test tree has none.
5. Start-up hooks are pinned in the change-control ledger, not in a plain JSON (probes V1, V5; the second addendum's
   docs/v5/baselines/tree_startup_hooks.json is removed). A `startup_hook` ledger entry pins a path and sha256; it is
   valid only when it cites a decision whose text names the path and quotes the first 16 hex digits of the sha256, so
   re-pinning edited content needs a new visible decision quoting the new hash (`python -m crucible.acceptance.v5
   pin-hook <path> --decision D-n`). A pinned hook must also pass `testscan.startup_hook_findings`: nothing
   `harness_findings` reports except meta-path finders restricted to one installed module (`if name != "torch": return
   None` first), no name, import or string of the repository's own modules, no dynamic imports, setattr, sys.modules
   scans or interpreter probes. `checker.harness_problems` now applies the pins as committed at the evidence commit (the
   P39 fix applied only to the global check, so no requirement could reach VERIFIED with the pinned hook in the tree).
   Pinned here: src/crucible/scale/_slotsite/sitecustomize.py, sha256 da0251cf6aba4da0e688b23293bb39316096d7b7bcf8f88d6d9cd7249ae3eb2c
   (workstream M's GPU cap and usage reporter, D-27; content reviewed: it wraps torch.cuda._lazy_init and reports
   allocator usage; it names no repository module). Stated residual: a pinned hook could still patch installed
   libraries (this one wraps torch's lazy CUDA initialisation by design).
6. The whole repository is scanned (probe V8: Gate A.1 puts the repository's own src/ on child processes' PYTHONPATH):
   start-up hooks, ignored code and compiled files anywhere in the repository count, and a start-up hook outside the
   crucible tree can never be pinned. Stated residual: code that a child process of the measured test loads by name
   from outside the crucible tree (Gate A.1's AGENTRS alias plugin) is not attested by the records.
