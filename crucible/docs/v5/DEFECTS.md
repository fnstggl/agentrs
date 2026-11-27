# Crucible 5.0 defect register

Machine-readable: every defect is one ```json block below with the fields id, severity (critical | high | medium |
low), status (open | fixed | duplicate | not_a_defect), area, summary, source (the audit file and finding index it came
from), owner and requirements; a fixed defect names what fixed it (`fixed_by`) and its regression tests, a duplicate or
not-a-defect its `resolution`. Gate X.6 counts open critical and high defects (`qa.defects`) and requires every
critical finding of every audit in docs/v5/audit/ to be registered here. Severity of an audit's critical findings is
"critical" unless re-triaged with a recorded reason.

Register opened by workstream P on 2026-11-26 from the recovery audit (docs/v5/audit/*.json `critical_findings`, the
B_population verdict) and P's own findings.

### DEF-A-1 (critical, open): Tier 3 does not exist: crucible/civ is a 5-line docstring (no generator, profiles, couplings, chains, Mode C p

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-1",
 "owner": "A, J",
 "requirements": [
  "SP-10"
 ],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Tier 3 does not exist: crucible/civ is a 5-line docstring (no generator, profiles, couplings, chains, Mode C provider, Gate O/P/SP-10 tests). All 13 J requirements are unmet."
}
```

### DEF-A-2 (critical, open): The real population (pop) cannot be combined with the recommender (feed): register_activity is not exported fr

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-2",
 "owner": "A, J",
 "requirements": [
  "C.3"
 ],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 1
 },
 "status": "open",
 "summary": "The real population (pop) cannot be combined with the recommender (feed): register_activity is not exported from crucible.population (social/recsys/service.py:179-187). Gate C.3 fails on this; the social gates C2/D/I were built on the stand-in crowd population, which refuses to coexist with pop."
}
```

### DEF-A-3 (critical, open): Loading all 87 modules into one world (shim) runs 30 days cleanly, but every v5 commerce/social/innovation/ins

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-3",
 "owner": "A, J",
 "requirements": [
  "O.n1"
 ],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Loading all 87 modules into one world (shim) runs 30 days cleanly, but every v5 commerce/social/innovation/institution/labour/science table stays empty: disconnected mini-environments behind one dispatcher (O.n1)."
}
```

### DEF-A-4 (critical, open): The duplicate-fact check is vacuous (opt-in; only assets declares). Real duplicates exist: relationships, comm

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-4",
 "owner": "A, J",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 3
 },
 "status": "open",
 "summary": "The duplicate-fact check is vacuous (opt-in; only assets declares). Real duplicates exist: relationships, communities, products (D-11 open), institutions, contracts, employment, and >=6 disjoint populations."
}
```

### DEF-A-5 (critical, open): Cross-process snapshot resume fails with pop_adaptive: kernel replay(base_blob) runs on_load before attaching 

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-5",
 "owner": "A, J",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Cross-process snapshot resume fails with pop_adaptive: kernel replay(base_blob) runs on_load before attaching the store (kernel.py:1003-1008, 965-966). In-process tests hide this."
}
```

### DEF-A-6 (critical, open): Subscriber notification advances the emitter's RNG (kernel.py:581), so typed validation or any coupling module

```json
{
 "area": "A_J_substrate_civ",
 "id": "DEF-A-6",
 "owner": "A, J",
 "requirements": [
  "SP-10.3"
 ],
 "severity": "critical",
 "source": {
  "audit": "A_J_substrate_civ",
  "file": "docs/v5/audit/A_J_substrate_civ.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Subscriber notification advances the emitter's RNG (kernel.py:581), so typed validation or any coupling module changes the trajectory, which confounds the counterfactual removal Gate O/SP-10.3 relies on. The 4.0 sandbox also makes run-to-run hashes differ."
}
```

### DEF-C-1 (critical, open): Audience rewards hash-vector alignment, not content: meaningless tokens reached 3,213/4,000 people vs 330 for 

```json
{
 "area": "C_social",
 "id": "DEF-C-1",
 "owner": "C (CORE-POP)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Audience rewards hash-vector alignment, not content: meaningless tokens reached 3,213/4,000 people vs 330 for an informative English article (p3)."
}
```

### DEF-C-2 (critical, open): Three disjoint populations; the C2 engine cannot run with B's pop (C.3 fails at feed install, service.py:184);

```json
{
 "area": "C_social",
 "id": "DEF-C-2",
 "owner": "C (CORE-POP)",
 "requirements": [
  "C.3"
 ],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Three disjoint populations; the C2 engine cannot run with B's pop (C.3 fails at feed install, service.py:184); autonomous population never uses communities, messaging, newsletters, launches, Q&A or search."
}
```

### DEF-C-3 (critical, open): Multimodal distribution broken: default encoder refuses non-text; semantic mode uses a random projection that 

```json
{
 "area": "C_social",
 "id": "DEF-C-3",
 "owner": "C (CORE-POP)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Multimodal distribution broken: default encoder refuses non-text; semantic mode uses a random projection that destroys topical alignment (p1, p4)."
}
```

### DEF-C-4 (critical, open): content.update leaves the distributed representation stale (bait-and-switch, p2).

```json
{
 "area": "C_social",
 "id": "DEF-C-4",
 "owner": "C (CORE-POP)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 3
 },
 "status": "open",
 "summary": "content.update leaves the distributed representation stale (bait-and-switch, p2)."
}
```

### DEF-C-5 (critical, open): C1 and C2 keep separate follower/share truths (78 vs 0 followers, p5); tool reading is never consumption.

```json
{
 "area": "C_social",
 "id": "DEF-C-5",
 "owner": "C (CORE-POP)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 4
 },
 "status": "open",
 "summary": "C1 and C2 keep separate follower/share truths (78 vs 0 followers, p5); tool reading is never consumption."
}
```

### DEF-C-6 (critical, open): Community, sharing/following and social-to-purchase evidence comes from scripted drivers; social internet is n

```json
{
 "area": "C_social",
 "id": "DEF-C-6",
 "owner": "C (CORE-POP)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "C_social",
  "file": "docs/v5/audit/C_social.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Community, sharing/following and social-to-purchase evidence comes from scripted drivers; social internet is not wired into any Tier 2 domain or civilization."
}
```

### DEF-D-1 (critical, open): E.3 unmet and unmeetable as designed: no probe registered; with aud.encoder='semantic' the MiniLM vector is ra

```json
{
 "area": "D_semantic",
 "id": "DEF-D-1",
 "owner": "D (CORE-SEM)",
 "requirements": [
  "E.3"
 ],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 0
 },
 "status": "open",
 "summary": "E.3 unmet and unmeetable as designed: no probe registered; with aud.encoder='semantic' the MiniLM vector is randomly projected into an unrelated interest space and audience choices stop following content (Spearman 0.026 vs 0.203 lexical)."
}
```

### DEF-D-2 (critical, open): Pretrained encoders drive no participant behaviour by default (aud default 'c2-random-indexing', a lexical bag

```json
{
 "area": "D_semantic",
 "id": "DEF-D-2",
 "owner": "D (CORE-SEM)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Pretrained encoders drive no participant behaviour by default (aud default 'c2-random-indexing', a lexical bag of words); text vectors are also blind to negation and sentiment (0.965 / 0.824 vs paraphrase 0.831)."
}
```

### DEF-D-3 (critical, open): E.5 fails on the merged tree (5 violations; likely false positives); the audit is a name regex that cannot est

```json
{
 "area": "D_semantic",
 "id": "DEF-D-3",
 "owner": "D (CORE-SEM)",
 "requirements": [
  "E.5"
 ],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 2
 },
 "status": "open",
 "summary": "E.5 fails on the merged tree (5 violations; likely false positives); the audit is a name regex that cannot establish absence of score proxies."
}
```

### DEF-D-4 (critical, open): Gate H reproduces but is weak: 74/129 calls rejected, 2/8 trivial goals met, no collaborative or productive re

```json
{
 "area": "D_semantic",
 "id": "DEF-D-4",
 "owner": "D (CORE-SEM)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 3
 },
 "status": "open",
 "summary": "Gate H reproduces but is weak: 74/129 calls rejected, 2/8 trivial goals met, no collaborative or productive result, 13/15 'consequential' actions are junk (echoed prompt fragments as arguments), in a 4.0 family."
}
```

### DEF-D-5 (critical, open): Semantic participants are unintegrated: nothing outside semantic/ uses them; population class-4 rows are inert

```json
{
 "area": "D_semantic",
 "id": "DEF-D-5",
 "owner": "D (CORE-SEM)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Semantic participants are unintegrated: nothing outside semantic/ uses them; population class-4 rows are inert (next_active_at=NEVER); no persistence, no self-goals; civ and multiagent are stubs."
}
```

### DEF-D-6 (critical, open): SP-3.2b finds 0.6B ineffective for programming, negotiation and strategy, yet D-10 standardizes 0.6B and no ro

```json
{
 "area": "D_semantic",
 "id": "DEF-D-6",
 "owner": "D (CORE-SEM)",
 "requirements": [
  "SP-3.2b"
 ],
 "severity": "critical",
 "source": {
  "audit": "D_semantic",
  "file": "docs/v5/audit/D_semantic.json",
  "finding": 5
 },
 "status": "open",
 "summary": "SP-3.2b finds 0.6B ineffective for programming, negotiation and strategy, yet D-10 standardizes 0.6B and no routing exists; prefix cache gives 0.986x and changes 3/8 bf16 outputs."
}
```

### DEF-E-1 (critical, open): No population decision uses a fitted model: models.build loads content_consumption@1 and commercial_response@1

```json
{
 "area": "E_calibration",
 "id": "DEF-E-1",
 "owner": "E",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 0
 },
 "status": "open",
 "summary": "No population decision uses a fitted model: models.build loads content_consumption@1 and commercial_response@1, then silently discards them (feature names never match FEATURES); all 9 decisions run 'plausible, unvalidated' in an integrated run."
}
```

### DEF-E-2 (critical, open): Only news_click@1 is reachable from world code, behind an opt-in feed config set only in tests; the 'learn' do

```json
{
 "area": "E_calibration",
 "id": "DEF-E-2",
 "owner": "E",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Only news_click@1 is reachable from world code, behind an opt-in feed config set only in tests; the 'learn' domain that uses it is not registered. Criteo, watch, return, session, feed and software models are never consumed."
}
```

### DEF-E-3 (critical, open): FIDELITY_REGISTER rows 23-33 claim fitted in-world use 'via calibration.load' that does not happen; SP-5.4 fai

```json
{
 "area": "E_calibration",
 "id": "DEF-E-3",
 "owner": "E",
 "requirements": [
  "SP-5.4"
 ],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 2
 },
 "status": "open",
 "summary": "FIDELITY_REGISTER rows 23-33 claim fitted in-world use 'via calibration.load' that does not happen; SP-5.4 fails (40 modules) and its parser drops 17 rows after a blank line."
}
```

### DEF-E-4 (critical, open): CalibratedOpen quantile-maps affinity by rank within the session batch: p_open is exactly invariant to uniform

```json
{
 "area": "E_calibration",
 "id": "DEF-E-4",
 "owner": "E",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 3
 },
 "status": "open",
 "summary": "CalibratedOpen quantile-maps affinity by rank within the session batch: p_open is exactly invariant to uniformly better content (max |dp| = 0) and depends on other viewers in the batch, a non-transferable simulator artifact."
}
```

### DEF-E-5 (critical, open): Offline calibration is genuine: checksums match Zenodo, HF and GH Archive; I reproduced the Criteo full-test p

```json
{
 "area": "E_calibration",
 "id": "DEF-E-5",
 "owner": "E",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Offline calibration is genuine: checksums match Zenodo, HF and GH Archive; I reproduced the Criteo full-test primary 0.114255 and session_rate deviance 3.3015 exactly; ITT (0.010) is correctly kept separate from the naive exposure contrast (0.378)."
}
```

### DEF-E-6 (critical, open): Protocol and licence: the E2 amendment added the eventually-kept candidates after test metrics were seen (disc

```json
{
 "area": "E_calibration",
 "id": "DEF-E-6",
 "owner": "E",
 "requirements": [
  "SP-5.5"
 ],
 "severity": "critical",
 "source": {
  "audit": "E_calibration",
  "file": "docs/v5/audit/E_calibration.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Protocol and licence: the E2 amendment added the eventually-kept candidates after test metrics were seen (disclosed; SP-5.5 passes); MIND came from an unofficial mirror that skips the licence gate; raw GitHub actor ids are kept."
}
```

### DEF-F-1 (critical, open): Qualifying ARR $100k is reached by one self-funded partner on a $100k annual plan with a 99% refund (net $1,00

```json
{
 "area": "F_commerce",
 "id": "DEF-F-1",
 "owner": "F (ECON)",
 "requirements": [
  "K.n1"
 ],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Qualifying ARR $100k is reached by one self-funded partner on a $100k annual plan with a 99% refund (net $1,000); K.n1 objective met on day 1 (only full refunds clear period_paid, subscriptions.py:298)."
}
```

### DEF-F-2 (critical, open): F2 escrow instruments (commit payouts contracts.py:256, inst project rewards institutions.py:1254) bypass circ

```json
{
 "area": "F_commerce",
 "id": "DEF-F-2",
 "owner": "F (ECON)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 1
 },
 "status": "open",
 "summary": "F2 escrow instruments (commit payouts contracts.py:256, inst project rewards institutions.py:1254) bypass circular-flow netting: payer not flagged, MRR counted; plain transfer is flagged."
}
```

### DEF-F-3 (critical, open): Evaluator-only integrity metrics (mrr_cents etc., registered with subject) are readable in-world through chart

```json
{
 "area": "F_commerce",
 "id": "DEF-F-3",
 "owner": "F (ECON)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Evaluator-only integrity metrics (mrr_cents etc., registered with subject) are readable in-world through charter conditions/commitment predicates: exact evaluator MRR recovered in 14 queries."
}
```

### DEF-F-4 (critical, open): MRR counts list price for payers below FLAG_SHARE 0.5 (45% cashback keeps full MRR) and for cancel-at-period-e

```json
{
 "area": "F_commerce",
 "id": "DEF-F-4",
 "owner": "F (ECON)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 3
 },
 "status": "open",
 "summary": "MRR counts list price for payers below FLAG_SHARE 0.5 (45% cashback keeps full MRR) and for cancel-at-period-end annual plans for up to a year."
}
```

### DEF-F-5 (critical, open): Commerce stack is used only by tests/reference/red-team: no domain/civilization installs it; the live 'revenue

```json
{
 "area": "F_commerce",
 "id": "DEF-F-5",
 "owner": "F (ECON)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Commerce stack is used only by tests/reference/red-team: no domain/civilization installs it; the live 'revenue' goal family uses rolling 30-day independent_revenue; 4.0 household-pool revenue is invisible to F1 metrics."
}
```

### DEF-F-6 (critical, open): Demand is a scripted single-need patrons population separate from the v5 population; no acquisition channels; 

```json
{
 "area": "F_commerce",
 "id": "DEF-F-6",
 "owner": "F (ECON)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "F_commerce",
  "file": "docs/v5/audit/F_commerce.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Demand is a scripted single-need patrons population separate from the v5 population; no acquisition channels; static incumbents; default experience reward is cash-delta legacy_shaped_reward."
}
```

### DEF-G-1 (critical, fixed): pkg.advisory is 'verified' iff any repro test fails (pkg.py:772): an assert-False repro that never touches vec

```json
{
 "area": "G_innovation",
 "fixed_by": "wave 2 (08ea124): an advisory is verified only by a release that makes its reproduction pass; wave 3b (c5c/w3b-econ2, D-42): a reproduction must exercise the package across the attested package boundary, so a test that never touches the package or only sniffs its source is refused",
 "id": "DEF-G-1",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_pkg.py::test_advisories_are_verified_only_by_a_release_that_fixes_them; red-team probe v5.innovation.advisory_by_sniffing_the_package_source (tests/v5/test_gate_W.py::test_W4_metric_manipulation)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 0
 },
 "status": "fixed",
 "summary": "pkg.advisory is 'verified' iff any repro test fails (pkg.py:772): an assert-False repro that never touches vecmath was verified critical, spawned an alternative need for its dependent, and can never be marked fixed."
}
```

### DEF-G-2 (critical, fixed): Adoption-by-execution is forgeable: dependent code overwrote _pkgtrace._C (pkg.py:46) or committed its own dep

```json
{
 "area": "G_innovation",
 "fixed_by": "wave 2: parent-side adoption by ablation, mounts owned by the package manager; wave 3b (D-42): the attested package boundary (measurement runs hold API proxies, each package runs in its own sandbox process, credit needs an exact control reproduction, a parent-observed call and a changed outcome)",
 "id": "DEF-G-2",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_pkg.py::test_forged_adoption_and_lineage_are_not_recorded, ::test_attested_boundary_credits_calls_not_reading_the_package; tests/v5/test_gate_SP_6_5b.py (six forgeries, 0 credited, honest control credited)",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 1
 },
 "status": "fixed",
 "summary": "Adoption-by-execution is forgeable: dependent code overwrote _pkgtrace._C (pkg.py:46) or committed its own deps/vecmath/ files; both recorded independent adoptions with no real calls; invariants stayed clean."
}
```

### DEF-G-3 (critical, fixed): Demand is closed to predefined formats with hidden-truth verifiers (adopt.py:78,143): a novel palette.extract 

```json
{
 "area": "G_innovation",
 "fixed_by": "wave 2: the population's work on procedurally generated, undocumented task categories with world-physics acceptance; wave 3b (D-43, D-44): composed task categories, autonomous population developers who serve them from public requests, and workflows of independently created services",
 "id": "DEF-G-3",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_devs.py; tests/v5/test_gate_SP_6_3.py; tests/v5/test_gate_SP_6_3b.py",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 2
 },
 "status": "fixed",
 "summary": "Demand is closed to predefined formats with hidden-truth verifiers (adopt.py:78,143): a novel palette.extract service got 24 'discoverers', 0 trials, 0 adopters. 6.3 novel categories / 6.4 novel combinations cannot be adopted."
}
```

### DEF-G-4 (critical, fixed): The deterministic cost model meters Python lines (sandbox/bridge.py:20,101): 149 ms of builtin work is metered

```json
{
 "area": "G_innovation",
 "fixed_by": "workstream O wave 3 (D-20): sandbox work is metered by the parent, C builtins are billed (probe v5.sandbox.c_builtins_are_billed, meter forgeries held); J.7 measures a pre-existing resource (the population's hand-work minutes saved) since wave 2, and reports developers' compute drawn from providers",
 "id": "DEF-G-4",
 "owner": "G (ECON)",
 "regression_tests": "tests/v5/test_gate_W.py::test_W3_reward_hacking; tests/v5/test_gate_J.py::test_J7_deployment_changes_world_resource_consumption",
 "requirements": [
  "J.7"
 ],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 3
 },
 "status": "fixed",
 "summary": "The deterministic cost model meters Python lines (sandbox/bridge.py:20,101): 149 ms of builtin work is metered 0.043 ms, a 1.7 ms loop 4.04 ms. J.7 and the capacity-driven needs rest on it; builtins give free compute."
}
```

### DEF-G-5 (critical, fixed): Gate L chains come from 4 hand-built lean worlds with host-keyed extractors; the software chain has 4 of 5 sta

```json
{
 "area": "G_innovation",
 "fixed_by": "wave 3b (c5c/w3b-econ2, D-43): Gate L is measured in the integrated civilisation (crucible.civ generator spec + the population's autonomous developers, innovation.inventworld.civ_spec) with counterfactual arms that differ only in the developers' config; chains are mined generically from the labelled-need history and the event log (no host-keyed extractor, no scripted stage); unofficial civilisation run of tests/v5/test_gate_L.py at 7bc3ddc (balanced seed 3, 30 days, 4 arms, G slot lane 2026-11-27): L.4 12 independent chains, minimum 5 stages, 0 scripted stages, all 12 with an agent artifact; L.5 11 of 12 chains confirmed by their removal arm (that count used the faulty confirmation of DEF-G-7; the repaired instrument is re-measured in the civilisation by the G lane); L.n1 0 runtime bonuses, 1553 income deposits verified. Replication: 4 seeds of small creation worlds (unofficial); the civilisation is measured on one seed (limitation recorded in workstreams/G.json). The 4.0-era alternative-seed checks of the lean worlds no longer exist (those worlds were removed).",
 "id": "DEF-G-5",
 "owner": "G (ECON)",
 "regression_tests": "tests/v5/test_gate_L.py (L.4 scripted_stages == 0, L.5 removal arms); tests/test_v5_innovation_gates.py; tests/test_v5_innovation_devs.py",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 4
 },
 "status": "fixed",
 "summary": "Gate L chains come from 4 hand-built lean worlds with host-keyed extractors; the software chain has 4 of 5 stages scripted; civ is a stub and no domain, population or civ uses innovation; 2 of 12 alternative-seed checks fail."
}
```

### DEF-G-6 (critical, fixed): Observation-only conflicts: the needs oracle reaches holders (adopters read nmod._held); 'independent' verdict

```json
{
 "area": "G_innovation",
 "fixed_by": "wave 2: needs events evaluator-only by default, no independence verdicts in caps tools or events, L.n1 counts every post-admission system credit; wave 3b: the pull-request review rule uses public identity facts (no related-party verdict in-world), L.n1 verifies population income deposits against the population's own deposit records",
 "id": "DEF-G-6",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_caps.py; tests/test_v5_innovation_needs.py; tests/test_v5_innovation_pkg.py::test_review_rule_uses_public_facts_not_the_evaluators_relatedness; tests/v5/test_gate_L.py::test_Ln1_no_special_case_success_bonuses",
 "requirements": [
  "L.n1"
 ],
 "severity": "critical",
 "source": {
  "audit": "G_innovation",
  "file": "docs/v5/audit/G_innovation.json",
  "finding": 5
 },
 "status": "fixed",
 "summary": "Observation-only conflicts: the needs oracle reaches holders (adopters read nmod._held); 'independent' verdicts appear in search results and caps.tried/adopted (a Sybil oracle). L.n1 misses system 'endowment' bonuses."
}
```

### DEF-G-7 (high, fixed): L.5 confirmed chains by construction or by name: every package chain went to ban_published whatever its task and was con

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: the removal arm is chosen by the chain's task (chain['need']); confirmation compares measured world outcomes of the removal arm (need never arises / persists at the end after factual relief, incumbents lose and respond, hand work on the step falls), never same-named artifacts",
 "id": "DEF-G-7",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_a_package_chain_is_confirmed_only_when_hand_work_does_not_fall_without_the_inventions; tests/test_v5_innovation_instruments.py::test_a_need_chain_needs_relief_in_the_factual_arm_and_persistence_without_the_invention; tests/test_v5_innovation_instruments.py::test_a_competition_chain_is_not_confirmed_when_incumbents_lose_and_respond_anyway; tests/test_v5_innovation_instruments.py::test_removal_arm_is_chosen_by_the_chains_task",
 "requirements": [
  "L.5"
 ],
 "severity": "high",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "L.5 confirmed chains by construction or by name: every package chain went to ban_published whatever its task and was confirmed when no same-named chain existed there (names carry developer indices); need chains were confirmed because the removal arm bans their own target, without checking that the factual arm relieved the need (2 of 8 package confirmations false in a creation probe; a gen.compose need confirmed although never relieved)."
}
```

### DEF-G-8 (high, fixed): The novel combination's demand was designer-supplied and undetected: inventworld planted the composed activity (COMPOSE 

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: composed activities come from a generic raw-feed world rule over every producer/consumer pair of the world's generated categories (inventworld.raw_feed_compositions; no pair is named); configured compositions count as designer-supplied demand (gates.configured_compositions) and their combinations are excluded from novel_combinations; developers plan a workflow from the public request text (devs.stages_of), never from the category; strict SP-6.4b classification reports needs from the builder's configured pipeline and pre-genesis community backlogs without counting them",
 "id": "DEF-G-8",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_composed_activities_follow_the_world_rule_and_a_hand_picked_pair_is_flagged; tests/test_v5_innovation_instruments.py::test_the_demand_detector_counts_a_configured_combination_in_a_world; tests/test_v5_innovation_instruments.py::test_a_developer_reads_the_stages_from_the_public_request_text_only",
 "requirements": [
  "SP-6.4b",
  "SP-6.3b"
 ],
 "severity": "high",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "The novel combination's demand was designer-supplied and undetected: inventworld planted the composed activity (COMPOSE = transform then allocate), developers composed only for usefulness.compose categories read from the category object, and designer_supplied_demand checked only data relations and literal workflow names (0 in both the planted and the compose=() world)."
}
```

### DEF-G-9 (medium, fixed): Chain mining found 'need relieved' before the need arose: relief was searched from the adoption onward, not from the nee

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: experiments.relief_t (relief only after both the adoption and the need's first appearance); _chain marks a chain unordered when relief does not follow the need",
 "id": "DEF-G-9",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_a_need_is_never_relieved_before_it_arose",
 "requirements": [
  "L.4"
 ],
 "severity": "medium",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "Chain mining found 'need relieved' before the need arose: relief was searched from the adoption onward, not from the need's first appearance, and the ordered check ignored the need stage (a 6-stage pack.knapsack chain with relief on day 9 and the need first labelled on day 17 led min_chain_stages)."
}
```

### DEF-G-10 (medium, fixed): J.3/J.4/J.6/J.7 took the minimum over categories that already pass J.n1 only (ok = passing or all), so they passed whene

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: minima over every category in which developers' services were used",
 "id": "DEF-G-10",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_j_minima_cover_every_category_not_only_the_passing_ones",
 "requirements": [
  "J.3",
  "J.4",
  "J.6",
  "J.7"
 ],
 "severity": "medium",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "J.3/J.4/J.6/J.7 took the minimum over categories that already pass J.n1 only (ok = passing or all), so they passed whenever J.n1 did (reported J.6 152 against 49 over all categories)."
}
```

### DEF-G-11 (medium, fixed): L.n1 exempted any acct_external credit of kind 'income' and memo 'income deposited' that coincided with a pop.deposit ba

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: an income credit is exempt only when it pairs one-to-one with the population module's own pop.deposit record of the same individual, time and exact amount",
 "id": "DEF-G-11",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_an_income_credit_booked_next_to_a_real_deposit_is_counted",
 "requirements": [
  "L.n1"
 ],
 "severity": "medium",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "L.n1 exempted any acct_external credit of kind 'income' and memo 'income deposited' that coincided with a pop.deposit batch, without comparing amounts: a planted 50,000-cent bonus per developer at every deposit (46 payments, 2,300,000 cents) reported special_case_bonuses 0."
}
```

### DEF-G-12 (medium, fixed): The SP-6.3b single-judge scan was name-based and same-module only: a call to the shared semantic space (space.text_vecto

```json
{
 "area": "G_innovation",
 "fixed_by": "eb517a1e: the scan follows calls across crucible modules and flags any call into the semantic space, social.audience, population.semantics, substrate.inference or a model library; negative controls inject each into every listed decision and through a helper module; the developer interest path (not a usefulness verdict) is reported separately",
 "id": "DEF-G-12",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_instruments.py::test_a_language_model_call_injected_into_a_participant_decision_is_flagged; tests/test_v5_innovation_instruments.py::test_the_scan_reaches_every_listed_decision; tests/test_v5_innovation_instruments.py::test_a_judgement_hidden_in_a_helper_of_another_module_is_flagged",
 "requirements": [
  "SP-6.3b"
 ],
 "severity": "medium",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "The SP-6.3b single-judge scan was name-based and same-module only: a call to the shared semantic space (space.text_vector, which runs Qwen3) or social.audience.encode injected into economy/work.py::_accepted was flagged on 0 paths, and no negative control existed."
}
```

### DEF-G-13 (high, fixed): Software collaboration was inert in the integrated world: devs never called pkg.pr_open, pr_review, pr_merge, fork, repo

```json
{
 "area": "G_innovation",
 "fixed_by": "8ce15b76: autonomous developers file issues, propose better approaches as pull requests from forks, maintainers read, review (independently), merge and release, dependents upgrade, unmerged or rejected proposals become fork packages, breaking releases get an issue and a reliability advisory (devs._contribute/_triage/_adopt_fork/_upgrade); SP-6.5 reports the civilisation's features next to the scripted scenario",
 "id": "DEF-G-13",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_devs.py::test_developers_collaborate_through_issues_pull_requests_review_and_forks; tests/v5/test_gate_SP_6_5.py",
 "requirements": [
  "SP-6.5"
 ],
 "severity": "high",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "Software collaboration was inert in the integrated world: devs never called pkg.pr_open, pr_review, pr_merge, fork, report, advisory or any issues tool, so SP-6.5's issue trackers, pull requests, independent review and forks came only from a scripted scenario."
}
```

### DEF-G-14 (low, fixed): devs._maintain replied to and resolved every open support ticket in the same session ('handled') with no change to the s

```json
{
 "area": "G_innovation",
 "fixed_by": "8ce15b76: tickets are acknowledged once and resolved only after a change shipped after the ticket was opened (new version, more cores, upgrade, workflow), with a note naming it",
 "id": "DEF-G-14",
 "owner": "G (ECON)",
 "regression_tests": "tests/test_v5_innovation_devs.py::test_developers_collaborate_through_issues_pull_requests_review_and_forks",
 "requirements": [],
 "severity": "low",
 "source": {
  "audit": "independent verifier of c5c/w3b-econ2 at f94ce3f1 (wave 3b repair pass)",
  "finding": null
 },
 "status": "fixed",
 "summary": "devs._maintain replied to and resolved every open support ticket in the same session ('handled') with no change to the service."
}
```

### DEF-H-1 (critical, open): Gate M below threshold on independent admission: 51/60 train and 18/20 eval (seeds 0-2); only 48/17 also pass 

```json
{
 "area": "H_rapid",
 "id": "DEF-H-1",
 "owner": "H",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Gate M below threshold on independent admission: 51/60 train and 18/20 eval (seeds 0-2); only 48/17 also pass held-out seeds 10-12. 6 of the 60 Part 8 items have no family; 4 more have none admitted."
}
```

### DEF-H-2 (critical, open): Fingerprint is not structural: 116/178 horizon/size/distractor variants have distance>0 to their own family (0

```json
{
 "area": "H_rapid",
 "id": "DEF-H-2",
 "owner": "H",
 "requirements": [
  "M.n1"
 ],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Fingerprint is not structural: 116/178 horizon/size/distractor variants have distance>0 to their own family (0.025-0.06 vs 0.063 closest real pair), so parameter variants would count. M.n1 gate test crashes (TypeError)."
}
```

### DEF-H-3 (critical, open): no_dominant_strategy tests only min/mid/max: one grid-searched constant lever setting gets >75% of oracle gain

```json
{
 "area": "H_rapid",
 "id": "DEF-H-3",
 "owner": "H",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 2
 },
 "status": "open",
 "summary": "no_dominant_strategy tests only min/mid/max: one grid-searched constant lever setting gets >75% of oracle gain in 17/31 admitted lever families (>=0.98 in 9, above the 'oracle' in 3). Hidden-causal/changing worlds reduce to static tuning."
}
```

### DEF-H-4 (critical, open): Intervention claims are not an admission check: 13 families fail their own controlled-intervention tests (29 f

```json
{
 "area": "H_rapid",
 "id": "DEF-H-4",
 "owner": "H",
 "requirements": [
  "M.8"
 ],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 3
 },
 "status": "open",
 "summary": "Intervention claims are not an admission check: 13 families fail their own controlled-intervention tests (29 failures), 9 of them admitted (cx_balancing_loop, plan_distribution, strat_bargain, team_assembly...). Gate M.8 fails."
}
```

### DEF-H-5 (critical, open): Participant 'spec' query reveals family name, seed, category and split='val' (memorisation key, eval-partition

```json
{
 "area": "H_rapid",
 "id": "DEF-H-5",
 "owner": "H",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Participant 'spec' query reveals family name, seed, category and split='val' (memorisation key, eval-partition leak). Tier 1 learner adapter defaults to legacy_shaped_reward; no objective-free rapid mode."
}
```

### DEF-H-6 (critical, open): Known failures: 3 stale tests (engine count, sandbox import) + real bug primitives.py:539 SelectionEffect (B,)

```json
{
 "area": "H_rapid",
 "id": "DEF-H-6",
 "owner": "H",
 "requirements": [
  "SP-7.3"
 ],
 "severity": "critical",
 "source": {
  "audit": "H_rapid",
  "file": "docs/v5/audit/H_rapid.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Known failures: 3 stale tests (engine count, sandbox import) + real bug primitives.py:539 SelectionEffect (B,) sd x (B,n) z; sd[:,None] fixes it (0 mismatches). SP-7.3 gate stale: records 4/8 though all 8 kinds have admitted families."
}
```

### DEF-I-1 (critical, open): 0 of 15 Tier 2 domains exist: crucible.domains REGISTRY empty, no d5_* families, no DOMAIN, no tests/v5/test_g

```json
{
 "area": "I_domains",
 "id": "DEF-I-1",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 0
 },
 "status": "open",
 "summary": "0 of 15 Tier 2 domains exist: crucible.domains REGISTRY empty, no d5_* families, no DOMAIN, no tests/v5/test_gate_N.py; 11 domains have no code; only kernel modules sci, fleet, firms, learn exist."
}
```

### DEF-I-2 (critical, open): Education world non-functional: learn.study raises KeyError('id') (learning.py:324 vs applier 675), every lear

```json
{
 "area": "I_domains",
 "id": "DEF-I-2",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Education world non-functional: learn.study raises KeyError('id') (learning.py:324 vs applier 675), every learn.tick fails, learners never study; with bug bypassed, wrong-key lessons beat correct ones on evaluator scores (2 seeds)."
}
```

### DEF-I-3 (critical, open): Framework/contract split: base.Domain vs experience.registry (module DOMAIN, dict configs, build(cfg,seed)); T

```json
{
 "area": "I_domains",
 "id": "DEF-I-3",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Framework/contract split: base.Domain vs experience.registry (module DOMAIN, dict configs, build(cfg,seed)); Tier 2 unreachable via experience.make (AttributeError); make() defaults to legacy_shaped_reward = evaluator score delta."
}
```

### DEF-I-4 (critical, open): Infrastructure: wages are 83% of default cost; fleet.staff technicians=0 cuts cost 81-84% with attainment >=0.

```json
{
 "area": "I_domains",
 "id": "DEF-I-4",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 3
 },
 "status": "open",
 "summary": "Infrastructure: wages are 83% of default cost; fleet.staff technicians=0 cuts cost 81-84% with attainment >=0.99997 over 60 days (one-call shortcut); fleet.experiment hands hidden true cpu_ms and hidden demand process to participant code."
}
```

### DEF-I-5 (critical, open): Science: real hidden-law/apparatus/rival/replicator mechanics work, but sci.replications and sci.paper crash (

```json
{
 "area": "I_domains",
 "id": "DEF-I-5",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Science: real hidden-law/apparatus/rival/replicator mechanics work, but sci.replications and sci.paper crash (KeyError 'id'); no novel-condition evaluator; replication test ignores relative noise and refuted the true law once."
}
```

### DEF-I-6 (critical, open): Admission passes sci/learn/fleet despite these crashes (ignores proc.failed); firms earn from an unconditional

```json
{
 "area": "I_domains",
 "id": "DEF-I-6",
 "owner": "I",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "I_domains",
  "file": "docs/v5/audit/I_domains.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Admission passes sci/learn/fleet despite these crashes (ignores proc.failed); firms earn from an unconditional external buyer; fleet/learn/firms not registered by crucible.load_all_modules()."
}
```

### DEF-K-1 (critical, open): Gate Q has no implementation: multiagent/ is a 5-line stub, test_gate_Q.py / test_gate_SP_12_1.py / test_gate_

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-1",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Gate Q has no implementation: multiagent/ is a 5-line stub, test_gate_Q.py / test_gate_SP_12_1.py / test_gate_SP_12_3.py do not exist, civ_available()=False, domains()={}; checker exits 'no test files exist yet'."
}
```

### DEF-K-2 (critical, open): Observation-only default violated: check_mode(None), SelfPlayEnv, KernelSelfPlayEnv, KernelExperienceEnv, regi

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-2",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Observation-only default violated: check_mode(None), SelfPlayEnv, KernelSelfPlayEnv, KernelExperienceEnv, registry.make, PettingZoo/Gym default to legacy_shaped_reward; selfplay learners cannot learn in observation_only (pg 72 'updates', weights unchanged)."
}
```

### DEF-K-3 (critical, open): Reward leak: in observation_only SelfPlayEnv still returns info['score'] and play() feeds it to Objective('sco

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-3",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Reward leak: in observation_only SelfPlayEnv still returns info['score'] and play() feeds it to Objective('score'); a PG learner reaches legacy performance (eval 7629 vs 7579) in 'observation_only'. The leakage battery ignores info."
}
```

### DEF-K-4 (critical, open): The 4.0 'joint' cooperative objective hands each learner its partner's private per-tick net (recovered to 6.8e

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-4",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 3
 },
 "status": "open",
 "summary": "The 4.0 'joint' cooperative objective hands each learner its partner's private per-tick net (recovered to 6.8e-09): cooperation is manufactured by reward sharing, not emergent through permitted interactions (12.3)."
}
```

### DEF-K-5 (critical, open): Controlled-identity revenue counted: multi-agent harnesses admit agents without a controller; a same-learner i

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-5",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Controlled-identity revenue counted: multi-agent harnesses admit agents without a controller; a same-learner identity's subscription counts as 2000c MRR / 1 independent customer; with a shared controller commerce.integrity excludes it."
}
```

### DEF-K-6 (critical, open): Only in-world learned groups are pop_strategic A/B: real hash-chained updates and frozen copies, but a 4-head 

```json
{
 "area": "K_selfplay",
 "id": "DEF-K-6",
 "owner": "K",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "K_selfplay",
  "file": "docs/v5/audit/K_selfplay.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Only in-world learned groups are pop_strategic A/B: real hash-chained updates and frozen copies, but a 4-head template choice with no cooperative actions, no measured competition and no held-out/cross-play evaluation."
}
```

### DEF-L-1 (critical, open): observation_only is not the default anywhere: make(), all adapters, MechVecEnv, KernelMechEnv, SelfPlay envs, 

```json
{
 "area": "L_experience",
 "id": "DEF-L-1",
 "owner": "L",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 0
 },
 "status": "open",
 "summary": "observation_only is not the default anywhere: make(), all adapters, MechVecEnv, KernelMechEnv, SelfPlay envs, Gym, PettingZoo, WorldEnv, CAM2 adapter and ReferenceLearner default to legacy; default MechVecEnv paid reward 30/30 steps"
}
```

### DEF-L-2 (critical, open): Privileged progress leaks via tools in observation_only: goals.check_now over independent_revenue let a learne

```json
{
 "area": "L_experience",
 "id": "DEF-L-2",
 "owner": "L",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Privileged progress leaks via tools in observation_only: goals.check_now over independent_revenue let a learner recover the verifier's exact progress (0.4) in 21 read-only calls; use.report returns verified_quality; the battery sees 0"
}
```

### DEF-L-3 (critical, open): Gate R leakage tests prove only mode-invariance and key-name absence; world-tool oracles pass them by construc

```json
{
 "area": "L_experience",
 "id": "DEF-L-3",
 "owner": "L",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Gate R leakage tests prove only mode-invariance and key-name absence; world-tool oracles pass them by construction; no coverage of the rapid adapter, Tier 2/3, v5 module worlds or the sl adapter (which pays a native -0.01 penalty)"
}
```

### DEF-L-4 (critical, open): RapidExperienceEnv (v5 Tier 1) broken for multi-agent: execute(a1) returns a0's result, observe_since leaks pr

```json
{
 "area": "L_experience",
 "id": "DEF-L-4",
 "owner": "L",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 3
 },
 "status": "open",
 "summary": "RapidExperienceEnv (v5 Tier 1) broken for multi-agent: execute(a1) returns a0's result, observe_since leaks private a0->a2 events to a1, events duplicated, t in ticks, cost_s 0, no causes, world_changes never counted"
}
```

### DEF-L-5 (critical, open): Tier 2 branch of make() is incompatible with workstream I's domains.base.Domain (AttributeError with a conform

```json
{
 "area": "L_experience",
 "id": "DEF-L-5",
 "owner": "L",
 "requirements": [
  "S.2",
  "S.3",
  "S.5",
  "SP-11.1"
 ],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Tier 2 branch of make() is incompatible with workstream I's domains.base.Domain (AttributeError with a conforming domain); no domain registered and civ is a stub, so S.2, S.3, S.5 and SP-11.1 fail and would keep failing after I lands"
}
```

### DEF-L-6 (critical, open): Positive: reference learner genuinely learns in Tier 1 observation_only (score -0.29 -> 0.75 vs frozen -0.45);

```json
{
 "area": "L_experience",
 "id": "DEF-L-6",
 "owner": "L",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "L_experience",
  "file": "docs/v5/audit/L_experience.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Positive: reference learner genuinely learns in Tier 1 observation_only (score -0.29 -> 0.75 vs frozen -0.45); store, exact checkpoints, reproducible CLI, exp module and long-horizon provenance reproduced"
}
```

### DEF-M-1 (critical, open): Workstream M never ran: no gate V / SP-13.7 / SP-13.8 tests, no v5 benchmark, no recorded throughput targets; 

```json
{
 "area": "M_performance",
 "id": "DEF-M-1",
 "owner": "M",
 "requirements": [
  "SP-13.7",
  "SP-13.8"
 ],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Workstream M never ran: no gate V / SP-13.7 / SP-13.8 tests, no v5 benchmark, no recorded throughput targets; all 12 M requirements unmet."
}
```

### DEF-M-2 (critical, open): crucible.scale.slots enforces nothing: 3 of 7 dimensions tracked; a 100 MB GPU reservation allocated 1,024 MB;

```json
{
 "area": "M_performance",
 "id": "DEF-M-2",
 "owner": "M",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 1
 },
 "status": "open",
 "summary": "crucible.scale.slots enforces nothing: 3 of 7 dimensions tracked; a 100 MB GPU reservation allocated 1,024 MB; affinity 48 CPUs, no rlimits; 13/219 logged jobs exceeded reserved CPU; no FIFO; zombies keep slots; simulation never consults it."
}
```

### DEF-M-3 (critical, open): Storage growth blocks long/large runs: about 1.83 KB per individual-day (large: 184 MB per sim day), uncompres

```json
{
 "area": "M_performance",
 "id": "DEF-M-3",
 "owner": "M",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Storage growth blocks long/large runs: about 1.83 KB per individual-day (large: 184 MB per sim day), uncompressed ZIP_STORED blobs, whole-row rewrites; a 30-day 10M run would need about 550 GB against 21 GB free disk."
}
```

### DEF-M-4 (critical, open): No approximate large-population mode and no high-fidelity comparison (V.9); the 'estimated population mode' cl

```json
{
 "area": "M_performance",
 "id": "DEF-M-4",
 "owner": "M",
 "requirements": [
  "V.9"
 ],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 3
 },
 "status": "open",
 "summary": "No approximate large-population mode and no high-fidelity comparison (V.9); the 'estimated population mode' claimed in ARCHITECTURE.md and DECISION_LOG D-5 does not exist in code."
}
```

### DEF-M-5 (critical, open): Amendment (a) violated in the owned path: crucible.scale.rollout.WorldEnv defaults to legacy_shaped_reward (ro

```json
{
 "area": "M_performance",
 "id": "DEF-M-5",
 "owner": "M",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 4
 },
 "status": "open",
 "summary": "Amendment (a) violated in the owned path: crucible.scale.rollout.WorldEnv defaults to legacy_shaped_reward (rollout.py:128); probe returned reward 12345.0 for an exogenous deposit with no action."
}
```

### DEF-M-6 (critical, open): Integrated population cost is dominated by work unrelated to decisions: event-driven only 2.18x faster than de

```json
{
 "area": "M_performance",
 "id": "DEF-M-6",
 "owner": "M",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "M_performance",
  "file": "docs/v5/audit/M_performance.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Integrated population cost is dominated by work unrelated to decisions: event-driven only 2.18x faster than dense (28.6x fewer activations); sandbox process per program call costs more CPU than the world; quadratic set(own) in popweb forum (web.py:775); class-4 semantic rows never act; 10M exists only on the simplified crowd path or as B pop creation."
}
```

### DEF-N-1 (critical, open): Workstream N's v5 scope is essentially unimplemented: no cross-tier registry or sampler, 50/30/20 mixture, v5 

```json
{
 "area": "N_research",
 "id": "DEF-N-1",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Workstream N's v5 scope is essentially unimplemented: no cross-tier registry or sampler, 50/30/20 mixture, v5 curriculum, proposal interface, six-level framework, A/B/C/D config or new-family positive controls; no N gate tests exist; all 25 requirements NOT_STARTED."
}
```

### DEF-N-2 (critical, open): The weights-only internalisation protocol is in-process and certified a learner whose checkpoint never changed

```json
{
 "area": "N_research",
 "id": "DEF-N-2",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 1
 },
 "status": "open",
 "summary": "The weights-only internalisation protocol is in-process and certified a learner whose checkpoint never changed (state kept in process memory) as 'internalised', valid, with no warning (p2)."
}
```

### DEF-N-3 (critical, open): Comparison intervals average learner seeds away (unit 'task (mean over seeds)'); 94% false-positive rate for i

```json
{
 "area": "N_research",
 "id": "DEF-N-3",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Comparison intervals average learner seeds away (unit 'task (mean over seeds)'); 94% false-positive rate for identical learners under seed-level variation (p5)."
}
```

### DEF-N-4 (critical, open): The research instrument always runs legacy shaped reward (MechVecEnv default, no reward_mode in the protocols)

```json
{
 "area": "N_research",
 "id": "DEF-N-4",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 3
 },
 "status": "open",
 "summary": "The research instrument always runs legacy shaped reward (MechVecEnv default, no reward_mode in the protocols); its positive controls fail or are misclassified under the mandated observation_only default (p4, p7)."
}
```

### DEF-N-5 (critical, open): The only adaptive curriculum (GoalSampler) moves away from measured weaknesses, has no random share, and sampl

```json
{
 "area": "N_research",
 "id": "DEF-N-5",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 4
 },
 "status": "open",
 "summary": "The only adaptive curriculum (GoalSampler) moves away from measured weaknesses, has no random share, and samples held-out test worlds labelled train (p3)."
}
```

### DEF-N-6 (critical, open): compose.py calibration plants a simulator shortcut (optimum at 25%/75% opposite the published default) that a 

```json
{
 "area": "N_research",
 "id": "DEF-N-6",
 "owner": "N",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "N_research",
  "file": "docs/v5/audit/N_research.json",
  "finding": 5
 },
 "status": "open",
 "summary": "compose.py calibration plants a simulator shortcut (optimum at 25%/75% opposite the published default) that a fixed rule exploits for a median 0.82 of the oracle's improvement, undetected by admission (p9); v5 partitions are mutable and the sealed salt is readable by learner code (p8)."
}
```

### DEF-O-1 (critical, open): Sandbox metering is self-reported: agent code rewrote boot's __main__._LINES (or forged a done message) so pkg

```json
{
 "area": "O_security",
 "id": "DEF-O-1",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 0
 },
 "status": "open",
 "summary": "Sandbox metering is self-reported: agent code rewrote boot's __main__._LINES (or forged a done message) so pkg.exec billed 0c vs 240c and compute_use reached -40,000 s; integrity post-check still passed (bridge.py:100, code.py:148)."
}
```

### DEF-O-2 (critical, open): Tier 1 reward hack: pe_* program_exec families bill cost from sandbox-reported lines; a program setting _LINES

```json
{
 "area": "O_security",
 "id": "DEF-O-2",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 1
 },
 "status": "open",
 "summary": "Tier 1 reward hack: pe_* program_exec families bill cost from sandbox-reported lines; a program setting _LINES negative turned cost 64,077 into -1.6e9 (cash +1.6e9) in pe_rule_induction (program_exec.py:103,1187)."
}
```

### DEF-O-3 (critical, open): Adoption-by-execution forgeable: pkg tracer runs in the agent's process; writing _pkgtrace._C recorded 3e9 'in

```json
{
 "area": "O_security",
 "id": "DEF-O-3",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 2
 },
 "status": "open",
 "summary": "Adoption-by-execution forgeable: pkg tracer runs in the agent's process; writing _pkgtrace._C recorded 3e9 'independent' calls into a never-called package; invariants pass (innovation/pkg.py:44-66,872-892)."
}
```

### DEF-O-4 (critical, open): Gate W does not exist: no test_gate_W.py or test_gate_SP_16.py; redteam.v5.probes() raises ModuleNotFoundError

```json
{
 "area": "O_security",
 "id": "DEF-O-4",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 3
 },
 "status": "open",
 "summary": "Gate W does not exist: no test_gate_W.py or test_gate_SP_16.py; redteam.v5.probes() raises ModuleNotFoundError (9/10 modules missing), suite/gate_metrics missing; 6 substrate probes only, 1 silently N/A."
}
```

### DEF-O-5 (critical, open): No v5 module (population, social, commerce, innovation, semantic, experience, Tier 1/2/3) has red-team coverag

```json
{
 "area": "O_security",
 "id": "DEF-O-5",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 4
 },
 "status": "open",
 "summary": "No v5 module (population, social, commerce, innovation, semantic, experience, Tier 1/2/3) has red-team coverage; static guard forbidden list is 4.0-only (calibration.datasets with urllib acquire reachable from world code)."
}
```

### DEF-O-6 (critical, open): reward_mode defaults to legacy_shaped_reward (experience/registry.py:99, modes.py:34), violating the observati

```json
{
 "area": "O_security",
 "id": "DEF-O-6",
 "owner": "O",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "O_security",
  "file": "docs/v5/audit/O_security.json",
  "finding": 5
 },
 "status": "open",
 "summary": "reward_mode defaults to legacy_shaped_reward (experience/registry.py:99, modes.py:34), violating the observation-only amendment; sandbox threads keep running after jobs return (unbilled CPU)."
}
```

### DEF-P-1 (critical, fixed): --verify-all exits 0 with CRUCIBLE_5_COMPLETE when requirements are deleted from the manifest (empty manifest 

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w1-qa: manifest anchored to derive() and the ledger; empty or pruned manifests never complete",
 "id": "DEF-P-1",
 "owner": "P",
 "regression_tests": "tests/test_v5_acceptance_checker.py::test_P2_an_empty_manifest_never_completes, ::test_P3_P13_deleting_a_requirement_is_detected",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 0
 },
 "status": "fixed",
 "summary": "--verify-all exits 0 with CRUCIBLE_5_COMPLETE when requirements are deleted from the manifest (empty manifest is ok=True); manifest edits are exempt from the dirty-tree check (evidence.py:26, checker.py:207)."
}
```

### DEF-P-2 (critical, fixed): Hand-written evidence+JUnit+review with no test run is VERIFIED; the official run+review pipeline accepts a ga

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w1-qa: evidence only from official runs in fresh clones; reviews re-execute and reproduce; placeholder and constant-metric tests rejected",
 "id": "DEF-P-2",
 "owner": "P",
 "regression_tests": "tests/test_v5_acceptance_checker.py::test_P1_handwritten_evidence_junit_and_review_are_rejected, ::test_P12_a_test_that_records_constants_and_asserts_nothing_fails",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 1
 },
 "status": "fixed",
 "summary": "Hand-written evidence+JUnit+review with no test run is VERIFIED; the official run+review pipeline accepts a gate test that records constants; reviewer identity is a free-text string; nothing re-executes evidence."
}
```

### DEF-P-3 (critical, fixed): Criteria can be weakened silently by rewriting criteria_history[0]; tests/implementation lists are not change-

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w1-qa: append-only hash-chained ledger anchored in git; criteria changes provably stricter with a decision; test and implementation lists change-controlled",
 "id": "DEF-P-3",
 "owner": "P",
 "regression_tests": "tests/test_v5_acceptance_checker.py::test_P4_weakening_criteria_by_rewriting_history_is_detected, ::test_P4b_rewriting_the_ledger_itself_is_detected, ::test_P10_swapping_the_test_list_is_detected",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 2
 },
 "status": "fixed",
 "summary": "Criteria can be weakened silently by rewriting criteria_history[0]; tests/implementation lists are not change-controlled; the checker never compares the manifest with requirements.py/derive()."
}
```

### DEF-P-4 (critical, open): No requirement can be verified: 0/253 entries list implementation files, 132 name missing tests (all of Gates 

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-4",
 "owner": "P",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 3
 },
 "status": "open",
 "summary": "No requirement can be verified: 0/253 entries list implementation files, 132 name missing tests (all of Gates A,B,G,N-Q,T-X,OO,RP); no P gate test, no evidence, no reviews, no final report."
}
```

### DEF-P-5 (critical, open): reward_mode defaults to legacy_shaped_reward at every public entry point (make, Gym, PettingZoo, MechVecEnv, K

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-5",
 "owner": "P",
 "requirements": [],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 4
 },
 "status": "open",
 "summary": "reward_mode defaults to legacy_shaped_reward at every public entry point (make, Gym, PettingZoo, MechVecEnv, KernelMechEnv), contradicting the observation-only amendment; OO/RP entries only added at dd745b9."
}
```

### DEF-P-6 (critical, open): Full regression is not green: 1664 tests, 5 failed (a 4.0 test broken by the v5 media_error code; cxs_selected

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-6",
 "notes": "c5c/w1-qa fixed the 4.0 discovery failure (media_error now has a meaning in core/discovery.py; the 4.0 test is unchanged). The rapid failures belong to workstream H.",
 "owner": "P",
 "requirements": [
  "A.1",
  "X.4"
 ],
 "severity": "critical",
 "source": {
  "audit": "P_acceptance",
  "file": "docs/v5/audit/P_acceptance.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Full regression is not green: 1664 tests, 5 failed (a 4.0 test broken by the v5 media_error code; cxs_selected_layers batch bug at rapid/primitives.py:539; 3 stale Tier 1 contract tests), 2 GPU skips; A.1/X.4 fail."
}
```

### DEF-B-1 (critical, open): Population is not integrated: nothing outside its own tests uses it, the social internet cannot run on it (cru

```json
{
 "area": "B_population",
 "id": "DEF-B-1",
 "owner": "B (CORE-POP)",
 "requirements": [
  "SP-4.0",
  "B.1"
 ],
 "severity": "critical",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 0,
  "section": "critical_findings (also docs/v5/audit/B_population.md, 1. Verdict)"
 },
 "status": "open",
 "summary": "Population is not integrated: nothing outside its own tests uses it, the social internet cannot run on it (crucible.population exported nothing), a duplicate population (crowd) owns the same pop table, class-4 rows are inert."
}
```

### DEF-B-2 (critical, open): No fitted calibration model is used by any population decision although FIDELITY_REGISTER.md claimed six were 

```json
{
 "area": "B_population",
 "id": "DEF-B-2",
 "owner": "B, E",
 "requirements": [
  "SP-5.0",
  "SP-5.4"
 ],
 "severity": "critical",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 1,
  "section": "critical_findings (also docs/v5/audit/B_population.md, 1. Verdict)"
 },
 "status": "open",
 "summary": "No fitted calibration model is used by any population decision although FIDELITY_REGISTER.md claimed six were (an unsupported claim)."
}
```

### DEF-B-3 (high, open): Class 3 is a 4-head categorical bandit over a scripted entrepreneur template with unstable A2C updates; it is 

```json
{
 "area": "B_population",
 "id": "DEF-B-3",
 "owner": "B, K",
 "requirements": [
  "SP-3.2c",
  "G.4"
 ],
 "severity": "high",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 4,
  "section": "critical_findings (also docs/v5/audit/B_population.md, 1. Verdict)"
 },
 "status": "open",
 "summary": "Class 3 is a 4-head categorical bandit over a scripted entrepreneur template with unstable A2C updates; it is not strategic learning."
}
```

### DEF-B-4 (high, open): No gate test exists for B, G, SP-3.1, SP-3.3, SP-3.4 or SP-4.1.

```json
{
 "area": "B_population",
 "id": "DEF-B-4",
 "owner": "B",
 "requirements": [
  "B.1",
  "SP-3.1",
  "SP-3.3",
  "SP-3.4",
  "SP-4.1"
 ],
 "severity": "high",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 2,
  "section": "critical_findings (also docs/v5/audit/B_population.md, 1. Verdict)"
 },
 "status": "open",
 "summary": "No gate test exists for B, G, SP-3.1, SP-3.3, SP-3.4 or SP-4.1."
}
```

### DEF-P-7 (high, open): 13 gate tests assert nothing on measured values (test_F_4_baseline_comparisons, test_F_5_calibration_metrics, 

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-7",
 "owner": "D, E",
 "requirements": [
  "H.1",
  "F.4",
  "F.5",
  "F.n1",
  "SP-3.2b"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "13 gate tests assert nothing on measured values (test_F_4_baseline_comparisons, test_F_5_calibration_metrics, test_F_n1_no_synthetic_only_domain, test_H_1_eight_independent_participants, test_H_2_shared_model_weights, test_H_3_separate_goals_and_histories, test_H_4_tool_selection_on_held_out_situations, test_H_5_language_understanding, test_H_6_consequential_actions, test_H_7_verified_productive_or_collaborative_result, test_H_8_memory_and_inference_measurements, test_H_n1_smallest_effective_models, test_SP_3_2b_semantic_model_benchmark); the hardened checker refuses their evidence until they assert."
}
```

### DEF-P-8 (high, open): RP.1: 23 imports of CAM learner code (agentrs.cam2 / agentrs.sl) inside crucible/: the 4.0 adapters src/crucib

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-8",
 "owner": "L, orchestrator",
 "requirements": [
  "RP.1",
  "A.1",
  "SP-11.6d"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "RP.1: 23 imports of CAM learner code (agentrs.cam2 / agentrs.sl) inside crucible/: the 4.0 adapters src/crucible/integrations/agentrs_cam2.py and agentrs_sl.py, examples/ow_cam_baseline.py, the 4.0 test tests/test_agents_llm.py and the v5 test tests/test_v5_experience_core.py. Conflicts with A.1 (4.0 tests unchanged) and Clarification 1 (compatibility); needs an orchestrator decision (e.g. move the adapters and their tests to the agentrs side)."
}
```

### DEF-P-9 (high, open): The recovery audit (docs/v5/audit/, 29 files) exists only as uncommitted files in the c5-complete worktree; RP

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-9",
 "owner": "orchestrator",
 "requirements": [
  "RP.2",
  "RP.3",
  "X.6"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "The recovery audit (docs/v5/audit/, 29 files) exists only as uncommitted files in the c5-complete worktree; RP.2 and X.6 cannot be verified and RP.3 (no milestone work only in an uncommitted worktree) is violated until it is committed."
}
```

### DEF-P-10 (medium, open): Audit reports lack a skeptic review section for A_J_substrate_civ, M_performance, P_acceptance and B_populatio

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-10",
 "owner": "orchestrator",
 "requirements": [
  "RP.2"
 ],
 "severity": "medium",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "Audit reports lack a skeptic review section for A_J_substrate_civ, M_performance, P_acceptance and B_population, and B_population.json (referenced by B_population.md) does not exist."
}
```

### DEF-P-12 (high, open): Worktrees of merged wave-1 branches still hold uncommitted changes (worktree-wf_62a0cf0b-bb3-3 = F1, worktree-

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-12",
 "owner": "orchestrator",
 "requirements": [
  "RP.3"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "Worktrees of merged wave-1 branches still hold uncommitted changes (worktree-wf_62a0cf0b-bb3-3 = F1, worktree-wf_62a0cf0b-bb3-6 = L): work that exists only uncommitted (RP.3)."
}
```

### DEF-P-13 (high, open): Gate A.5, settled by measurement (wave 3): the committed 4.0 record docs/experiments/demo_lifetime.json (bazaa

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-13",
 "owner": "orchestrator decision (L for D-19, O for D-20, P)",
 "requirements": [
  "A.5"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "Gate A.5, settled by measurement (wave 3): the committed 4.0 record docs/experiments/demo_lifetime.json (bazaar, seed 7, 30 days; 73,617 events, final digest dc8a6794) is stale in 4.0 itself: c07098b regenerates 78,353 events, digest 9b08db59, identically in two runs, and the commit that last wrote the record (3df5faf) regenerates 73,617 events but digest 2b679e64, so even there the digest does not reproduce. Independently, v5 changes 4.0's own trajectory (bisected over the first-parent merges): 7ce3db0 (kernel, wave 1) equals c07098b; 1c3d69d (observation-only L, D-19) keeps 78,353 events but changes the final state digest (6fa80946, unchanged through 2d0f4ce); 33b5f2b (security O, D-20 parent-measured metering) changes the trajectory to 74,735 events; the wave-2a merges give 74,280 events, digest 753ff574, at 87a6c54. Classification: the record is v4-self-irreproducible, and the merged tree is a genuine mismatch against c07098b caused by D-19 (state) and D-20 (billing by measured CPU time instead of executed lines), i.e. a requirement conflict between A.5 and D-19/D-20 like A.1's (DEF-P-24); A.5 cannot pass without a decision. Full Gate A.5 on 87a6c54 (21 min, 1 CPU): 148 of 148 4.0 world trajectories in 4.0's deterministic mode equal c07098b's (0 nondeterministic); of 18 attempted records 10 reproduce exactly, 1 is v4-self-irreproducible, 7 mismatch (acceptance records A, B, G, H, I, M and demo_lifetime), and demo_lifetime gave 78,900 events in that run against 74,280 in another fresh process at the same commit: see DEF-P-27."
}
```

### DEF-P-11 (medium, open): 204 of the manifest's requirements are recorded by no gate test.

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-11",
 "owner": "all workstreams",
 "requirements": [
  "X.7"
 ],
 "severity": "medium",
 "source": {
  "audit": "workstream P (c5c/w1-qa)",
  "finding": null
 },
 "status": "open",
 "summary": "204 of the manifest's requirements are recorded by no gate test."
}
```

### DEF-P-14 (critical, fixed): The S1 environment injection still worked: the runner scrubbed only PYTHONPATH, so a usercustomize/.pth in a u

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: the measured process gets no PYTHON*/PYTEST_*/LD_PRELOAD/LD_AUDIT variable and runs with python -s and PYTHONNOUSERSITE=1",
 "id": "DEF-P-14",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P15_S1_pythonpath_sitecustomize_injection_is_inert",
  "tests/test_v5_acceptance_probes.py::test_P15b_user_site_usercustomize_injection_is_inert"
 ],
 "requirements": [
  "SP-21",
  "SP-21b"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "The S1 environment injection still worked: the runner scrubbed only PYTHONPATH, so a usercustomize/.pth in a user site named by PYTHONUSERBASE (user site-packages are enabled in /venv/main) repaired a committed defect in the measured process; evidence and review were VERIFIED."
}
```

### DEF-P-15 (critical, fixed): Evidence did not show what else ran in the measured process: a producer and a colluding reviewer using an inte

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: records carry the interpreter, its start-up hooks (checked against docs/v5/baselines/python_site_hooks.json), foreign modules, patched functions and injection variables; checker.env_problems rejects them",
 "id": "DEF-P-15",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P15c_a_poisoned_interpreter_is_detected_in_the_records",
  "tests/test_v5_acceptance_probes.py::test_P15d_a_function_patched_from_outside_the_tree_is_detected"
 ],
 "requirements": [
  "SP-21"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "Evidence did not show what else ran in the measured process: a producer and a colluding reviewer using an interpreter with a .pth start-up hook that patches the defect away reached VERIFIED; functions patched from foreign files were invisible."
}
```

### DEF-P-17 (critical, fixed): Measurements that do not depend on the implementation passed the official pipeline: constant metrics laundered

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: every review ablates the implementation (each function of the listed implementation modules raises a BaseException) and the measurement must not survive; the release check ablates again at HEAD; the static detector follows module-level constants and dict() calls",
 "id": "DEF-P-17",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P16_constant_metrics_that_evade_the_static_scan_are_caught_by_ablation",
  "tests/test_v5_acceptance_probes.py::test_P16d_an_honest_measurement_survives_ablation"
 ],
 "requirements": [
  "SP-21",
  "X.7"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "Measurements that do not depend on the implementation passed the official pipeline: constant metrics laundered through a helper function, a module-level dict or a dict() call evaded the static detector and reproduced exactly in the review."
}
```

### DEF-P-18 (critical, fixed): Patches of the implementation outside the test body (autouse fixture, helper with setattr, plain attribute ass

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: testscan follows fixtures, helpers and classes a test uses (patches count unless restored before record()); testscan.harness_findings scans the conftest files on the tests' paths; checker.harness_problems rejects them",
 "id": "DEF-P-18",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P17_patching_the_implementation_outside_the_test_body_is_rejected",
  "tests/test_v5_acceptance_probes.py::test_P17d_a_conftest_fixture_patch_is_rejected",
  "tests/test_v5_acceptance_probes.py::test_P18_a_conftest_hook_that_rewrites_outcomes_is_rejected"
 ],
 "requirements": [
  "SP-21",
  "X.7"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "Patches of the implementation outside the test body (autouse fixture, helper with setattr, plain attribute assignment, conftest fixture) and committed conftest hooks that rewrite outcomes (a mutating pytest_runtest_makereport, pytest_pyfunc_call replacing the body) made a defective implementation VERIFIED; the JUnit and exit code come from pytest, so every reproduction agreed."
}
```

### DEF-P-19 (high, fixed): A broken implementation hidden by contextlib.suppress or by a broad except with a fallback value reached VERIF

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: suppress and broad handlers that neither re-raise nor use the exception are 'swallow'",
 "id": "DEF-P-19",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P19_swallowing_a_broken_implementation_is_rejected"
 ],
 "requirements": [
  "SP-21",
  "X.7"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "A broken implementation hidden by contextlib.suppress or by a broad except with a fallback value reached VERIFIED (only pass/return/continue handlers were detected)."
}
```

### DEF-P-20 (critical, fixed): Stale evidence stayed VERIFIED: staleness covered imported modules only (not data files the implementation rea

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: files_read (audit hook loaded with -p before collection), harness files and dataset checksums are part of staleness; start-up hook files in the tree are rejected; the release check reproduces and ablates at HEAD",
 "id": "DEF-P-20",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P20_a_changed_data_file_makes_evidence_stale",
  "tests/test_v5_acceptance_probes.py::test_P20b_the_release_check_reproduces_at_head_not_at_the_evidence_commit",
  "tests/test_v5_acceptance_probes.py::test_P21_tamper_code_that_hides_itself_and_is_reverted_is_detected",
  "tests/test_v5_acceptance_probes.py::test_P28_a_changed_dataset_makes_evidence_stale"
 ],
 "requirements": [
  "SP-21",
  "SP-21b"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "Stale evidence stayed VERIFIED: staleness covered imported modules only (not data files the implementation reads, conftest/pytest configuration or declared datasets); tamper code that hides itself from the module list and is reverted after the run was invisible; and --reproduce re-ran the evidence commit, so it could never see a change made since."
}
```

### DEF-P-21 (high, fixed): A broken working tree completed: assume-unchanged / skip-worktree flags hide edits of tracked files and an ign

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: checker.hidden_changes (index flags, ignored code files in src/ and tests/, start-up hooks anywhere) are project-level problems",
 "id": "DEF-P-21",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P22_index_flags_that_hide_working_tree_changes_block_completion",
  "tests/test_v5_acceptance_probes.py::test_P22b_an_ignored_startup_hook_in_the_tree_blocks_completion"
 ],
 "requirements": [
  "SP-21b"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "A broken working tree completed: assume-unchanged / skip-worktree flags hide edits of tracked files and an ignored sitecustomize hides from git status, which was the only working-tree check."
}
```

### DEF-P-22 (critical, fixed): A fully consistent forgery (genuine records re-stamped onto a defective commit, every hash fixed) gave CRUCIBL

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: a terminal project status (CRUCIBLE_5_COMPLETE, INCOMPLETE_BLOCKED) requires the release check --verify-all --reproduce; static checks alone report IN_PROGRESS",
 "id": "DEF-P-22",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P23_a_fully_consistent_forgery_cannot_complete_without_the_release_reproduction",
  "tests/test_v5_acceptance_probes.py::test_P23b_the_honest_project_completes_only_with_the_release_reproduction"
 ],
 "requirements": [
  "SP-21",
  "SP-21b"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "A fully consistent forgery (genuine records re-stamped onto a defective commit, every hash fixed) gave CRUCIBLE_5_COMPLETE on --verify-all: the stated residual was not enforced, because the release reproduction was optional."
}
```

### DEF-P-23 (high, fixed): Change-control and independence gaps: a review counted with no session identity on either side (self-review un

```json
{
 "area": "P_acceptance",
 "fixed_by": "c5c/w3-qa, decision D-26: session identities required by run/review and checker; anchors immutable; blockers must name the requirement; manifest.history_problems; the producing test must be in the canonical gate test file",
 "id": "DEF-P-23",
 "owner": "P",
 "regression_tests": [
  "tests/test_v5_acceptance_probes.py::test_P24_self_review_without_session_identities_is_rejected",
  "tests/test_v5_acceptance_probes.py::test_P25_re_anchoring_the_specification_in_the_ledger_is_detected",
  "tests/test_v5_acceptance_probes.py::test_P26_a_blocker_that_does_not_name_the_requirement_does_not_block_it",
  "tests/test_v5_acceptance_probes.py::test_P27_weakening_in_manifest_history_before_the_ledger_existed_is_detected",
  "tests/test_v5_acceptance_probes.py::test_P30_an_extra_test_file_cannot_stand_in_for_the_canonical_gate_test"
 ],
 "requirements": [
  "SP-21",
  "SP-19.7"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "fixed",
 "summary": "Change-control and independence gaps: a review counted with no session identity on either side (self-review under a second name); a second ledger anchor silently re-anchored an edited MASTER_SPEC.md; a blocker that names no requirement gave INCOMPLETE_BLOCKED; weakening committed in manifest history before the ledger existed went unnoticed; an extra test file (additions need no decision) could stand in for the canonical gate test."
}
```

### DEF-P-16 (high, open): 19 4.0 regression tests fail only because TMPDIR lies below a root-only directory (agent scratchpads such as /

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-16",
 "owner": "O (sandbox jail) / orchestrator (how the regression is run)",
 "requirements": [
  "A.1",
  "X.2",
  "X.4"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "open",
 "summary": "19 4.0 regression tests fail only because TMPDIR lies below a root-only directory (agent scratchpads such as /tmp/claude-0, mode 700): the v5 jailed Chromium (crucible.sandbox.jail drops to its own uid; its Playwright profile directory is created under TMPDIR) cannot create its profile (5 tests: test_web x2, test_internet x2, test_internet_adversarial x1), and the 4.0 PostgreSQL fixture tests/pg_fixture.py runs initdb as the postgres user, which cannot enter the directory (14 test_storage[postgres] tests). Measured in Gate A.1 run 1 (TMPDIR in the scratchpad) and present in the orchestrator's M5 failed_tests. At c07098b the browser tests launch Chromium unjailed and are not affected; the postgres tests fail the same way at c07098b. Workaround: run regressions with TMPDIR unset (REPRODUCE.md). Fix for the jail: give the Playwright driver a jail-traversable temporary directory instead of the inherited TMPDIR."
}
```

### DEF-P-24 (high, open): Gate A.1 conflicts with Clarification 2 / D-19 / D-20 as merged: 11 4.0 test files differ from c07098b (test_a

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-24",
 "owner": "orchestrator decision (P, L, O)",
 "requirements": [
  "A.1",
  "OO.8",
  "SP-13.7"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "open",
 "summary": "Gate A.1 conflicts with Clarification 2 / D-19 / D-20 as merged: 11 4.0 test files differ from c07098b (test_acceptance, test_agents_llm, test_api, test_hardening, test_learners, test_llm_eval, test_realworld_gate, test_research_envs, test_research_protocols, test_sandbox, test_selfplay): the observation-only default made wave-1 L pass reward_mode='legacy_shaped_reward' explicitly in 4.0 tests, D-20 (parent-measured metering) replaced test_sandbox::test_deterministic_line_accounting by test_deterministic_work_accounting (so 1 baseline node id no longer exists), and test_realworld_gate gained an autouse fixture that patches the bridge's hosting latch. A.1's D-23 criteria (v4_test_files_modified == 0, every baseline node id passes) cannot hold together with the observation-only default, and the criteria cannot be weakened (D-24). Needs a decision: either restore the 4.0 test files byte-identical and make 4.0 entry points select the legacy reward without changing the 4.0 tests (which contradicts 'observation_only is the default everywhere'), or record A.1 as BLOCKED by this conflict in BLOCKERS.md (naming A.1)."
}
```

### DEF-P-25 (high, open): crucible.load_all_modules() now imports huggingface_hub at import time in every process (semantic/space/__init

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-25",
 "owner": "D (CORE-SEM) / INT",
 "requirements": [
  "A.1",
  "SP-13.7"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "open",
 "summary": "crucible.load_all_modules() now imports huggingface_hub at import time in every process (semantic/space/__init__.py _version() -> semantic/space/lm.identity() -> semantic/service.local_revision()), which starts a native thread: 4.0 pool workers are no longer single-threaded and tests/test_scale.py::test_workers_are_recycled_after_a_fixed_number_of_jobs fails (threads == 2; passes at c07098b). Import-time model-identity resolution also adds start-up cost to every 4.0 process; resolve it lazily."
}
```

### DEF-B-5 (critical, open): Class-2 learning is real (identical-init learning vs frozen: utility +0.11..+0.15, CI>0 in 3/3 seeds; held-out

```json
{
 "area": "B_population",
 "id": "DEF-B-5",
 "notes": "registered by workstream P (wave 3): the B_population audit JSON was committed after the register was opened; whether the wave-2a population/semantic merge resolved it is for CORE-POP/CORE-SEM to show with gate evidence",
 "owner": "B (CORE-POP)",
 "requirements": [
  "B.3",
  "SP-3.2"
 ],
 "severity": "critical",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 3
 },
 "status": "open",
 "summary": "Class-2 learning is real (identical-init learning vs frozen: utility +0.11..+0.15, CI>0 in 3/3 seeds; held-out MSE halves) but limited to one decision, and personal GRU memory gives no measurable benefit."
}
```

### DEF-B-6 (critical, open): Content judgement is a closed 160-word bag-of-words x length/distinct-token score: a meaningless word-salad of

```json
{
 "area": "B_population",
 "id": "DEF-B-6",
 "notes": "registered by workstream P (wave 3): the B_population audit JSON was committed after the register was opened; whether the wave-2a population/semantic merge resolved it is for CORE-POP/CORE-SEM to show with gate evidence",
 "owner": "B, D (CORE-SEM)",
 "requirements": [
  "SP-4.0",
  "D.1"
 ],
 "severity": "critical",
 "source": {
  "audit": "B_population",
  "file": "docs/v5/audit/B_population.json",
  "finding": 5
 },
 "status": "open",
 "summary": "Content judgement is a closed 160-word bag-of-words x length/distinct-token score: a meaningless word-salad of topic words scored as well as a 'rich' article site (utility 0.099 vs 0.090, 1-day return 0.21 vs 0.18) and an agent's thin site equalled its rich site (0.152 vs 0.147)."
}
```

### DEF-P-26 (high, open): 4.0 test tests/test_ow_worlds.py::test_end_to_end_only_working_software_earns_a_living is order-dependent on t

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-26",
 "owner": "O (sandbox metering, D-20) / A",
 "requirements": [
  "A.1"
 ],
 "severity": "high",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "open",
 "summary": "4.0 test tests/test_ow_worlds.py::test_end_to_end_only_working_software_earns_a_living is order-dependent on the merged tree 87a6c54: it passes alone but fails when tests/test_learners.py ran earlier in the same process (the junk service gets long_term_adoption 1.0 instead of 0.0); at c07098b the same pair passes (71 passed). Suspected cause: process-wide state introduced since 4.0, most likely the D-20 deterministic-cost memo (crucible.sandbox.metering.MEMO, keyed by world lineage, which bills the first parent-side measurement of a computation for every later identical one in the process) or measured-CPU billing replacing the executed-line count; not yet confirmed."
}
```

### DEF-P-27 (critical, open): Since the D-20 merge (33b5f2b), 4.0 open worlds whose agents run sandboxed programs are no longer reproducible

```json
{
 "area": "P_acceptance",
 "id": "DEF-P-27",
 "owner": "O (D-20 metering) / orchestrator decision",
 "requirements": [
  "A.5",
  "A.1",
  "SP-13.7"
 ],
 "severity": "critical",
 "source": {
  "audit": "workstream P wave 3 (c5c/w3-qa): adversarial verification",
  "finding": null
 },
 "status": "open",
 "summary": "Since the D-20 merge (33b5f2b), 4.0 open worlds whose agents run sandboxed programs are no longer reproducible across processes: the 'deterministic' cost model bills the first parent-measured CPU time of a computation (crucible.sandbox.metering.MEMO, per process), so each fresh process bills differently. Measured: demo_lifetime at the same commit 87a6c54 gives 74,280 events in one fresh process and 78,900 in another (4.0: 78,353 in every run; 2d0f4ce, just before the D-20 merge: 78,353); Gate A.5 finds 6 of the 16 4.0 acceptance records changed by small amounts (e.g. A_real_software_creation requests_metered 15,629 vs 15,458; M_reproducibility events 7,362 vs 7,367). 4.0's reproducibility claim (record M) no longer holds in the merged tree. Needs a decision: keep a deterministic, measurement-free work count (4.0's executed-line model or a parent-side instruction count) for the deterministic cost model and use parent-measured CPU only for billing real hosts, or accept and document the loss under change control."
}
```
