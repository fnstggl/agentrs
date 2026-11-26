# Audit: Tier 2 domains (workstream I, MASTER_SPEC Part 9, Gate N)

Auditor scope: `crucible/src/crucible/domains/` (the orchestrator's framework `base.py`, plus partial kernel modules left by
the interrupted agents I1-I4), how it connects to `crucible.experience`, `crucible.substrate.modes` and the acceptance
manifest. The worktree was at a881d18 when the audit began; HEAD moved to dd745b9 during the audit (that commit only adds
Clarifications 2-3 and the OO/RP manifest entries, with no change to domain code). I modified no existing file. Every
probe script and log is in `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/I_domains/`.

## 1. Bottom line

**No Tier 2 domain exists.** `crucible.domains.load_all()` returns an empty `REGISTRY`, and no `d5_*` world family is
registered. No module defines a `DOMAIN`, and `tests/v5/test_gate_N.py` does not exist.
`python -m crucible.acceptance.v5 run --requirement N.1` prints "no test files exist yet for this selection". The
interrupted agents left four **kernel modules** (world mechanics) and no domain definitions. A domain definition
needs configurations, objectives, an evaluator, a reference policy and mechanics probes. The four modules:

| Module (file) | Kernel name | Intended domain | Runs? | Key defect found at runtime |
|---|---|---|---|---|
| `science.py` + `lawlang.py` | `sci` | 9.5 (I1) | yes | `sci.replications` and `sci.paper` crash with KeyError 'id'. The replication chi-square ignores relative noise, so it refutes the true law. |
| `infra_world.py` | `fleet` | 9.7 (I2) | yes | Firing all technicians cuts cost by 81-84% with no reliability loss. The experiment tool feeds participant code hidden true CPU costs. |
| `firms.py` + `_kit.py` | `firms` | 9.9 / 9.12 (I3) | yes | Revenue comes from an unconditional external buyer. Unknown skills are silently dropped. |
| `learning.py` + `concepts.py` + `common.py` | `learn` | 9.10 (I4) | **no** | `learn.study` raises KeyError 'id', so learners never study. With the crash bypassed, wrong-answer lessons beat correct ones. |

The other 11 domains have no code in `crucible.domains`: startup, software, content, community, invention, industry,
information, marketplace, crisis, commercialization, and organizational (beyond `firms`). Several have usable building
blocks in sibling packages (section 5).

## 2. Framework (`domains/base.py`, `__init__.py`, `modes.py`, `modules.py`, `seats.py`)

What it does:
* `Domain` / `DomainConfig` dataclasses.
* `validate()` checks for at least 3 train configs, one compositional and one structural eval config, and distinct
  `structure` digests.
* `register()` turns each config into a `d5_<domain>_<config>` family with splits `train`, `v5eval_compositional` and
  `v5eval_structural`.
* `run_reference()` drives a policy with the 4.0 Runner and calls `evaluate`.
* Mode B provider (`modes.py`).

Tests: `tests/test_v5_domains_framework.py` passes 7 tests. All of them use a fake probe domain built on the 4.0 `tiny`
family.

Defects (verified):
1. **Contract split, so Tier 2 cannot be reached through the learner interface.** INTERFACE_CONTRACTS section 9 and
   `experience/registry.py` expect a module-level `DOMAIN` with `configs: {split: [...]}`, `build(config, seed) -> WorldSpec`
   and `objective(world, difficulty)` (`registry.py:40-56, 85, 92, 125, 173`).
   `base.py` (decision D-13) uses something different:
   * `configs` is a tuple of `DomainConfig`;
   * `build(config, rng, overrides)` returns a recipe dict;
   * `objective(world, principal, difficulty)`;
   * registration goes through `base.REGISTRY`, not a module attribute.

   Probe `p06_contract.py`: after `base.register(probe)`, `experience.registry.domains()` returns `[]`, and
   `tiers_available()` gives `{2: False}`. Pushing the same `base.Domain` through the registry's own code path raises
   `AttributeError: 'tuple' object has no attribute 'values'`. Even a finished domain written to `base.py` would be
   invisible to `crucible.experience.make(2, ...)`.
2. **`validate()` cannot tell structure from numbers.** It compares `digest(c.structure)` (`base.py:109`). The
   framework's own passing test uses structures `{"x": 1}`, `{"x": 2}` and `{"x": 3}`, which differ only by a number.
   That is exactly what the docstring forbids ("changing only numbers is not a new configuration"). A real check would
   compare module lists, couplings, topologies and participant types of the *built* worlds.
3. **Reward-mode default.** `experience.make()` defaults to `reward_mode=legacy_shaped_reward`
   (`registry.py:99`; `modes.check_mode(None)` gives legacy). `_DomainObjective.step_reward` is the change in the
   evaluator's score (`registry.py:145-170`). That is evaluator-derived shaping and violates Clarification 2, which
   requires observation-only by default. The framework itself has no `reward_mode` notion.
4. **Module registration gap.** `domains/modules.py` imports only `seats` and `science`. After
   `crucible.load_all_modules()`, the module table reads `{'seat': True, 'sci': True, 'fleet': False, 'learn': False,
   'firms': False}`. A stored spec naming `fleet`, `learn` or `firms` therefore cannot be created or replayed in a
   fresh process unless something calls `crucible.domains.load_all()`.
5. **Three incompatible "seat" mechanisms:**
   * `seats.Seats` (genesis);
   * `common.make_seat` (used by `learn`);
   * `fleet`'s `on_join` hook, which fires only on `admin.join`.

   Probe `p04d_fleet_seat.py`: a seat created by the framework's `seat` module is never made fleet operator
   (`you_operate=False` in both module orders).
6. `run_reference` labels results `scripted_reference: True` but does not check anything about hiding. No mechanism
   verifies N.n1 (references are never importable from participant code and never named in tools or observations).

Verdict: **REPAIR** (S-M). Unify on one contract, add a structural-difference check on built worlds, add
`reward_mode=observation_only` with `verify()` separation, register every domain module in `modules.py`, use one seat
mechanism, and add a reference-isolation scan.

## 3. The four existing kernel modules

### 3.1 Scientific research (`science.py`, `lawlang.py`): hidden laws, apparatus, rival and replication labs

Mechanics are real:
* hidden laws drawn from 4 structurally different families (`lawlang.py:213-244`), compiled by a safe expression
  language;
* paid, capacity-limited, queued apparatus with hidden gain, offset and drift, plus Gaussian and relative noise;
* certified standards for calibration;
* dataset artifacts;
* publications whose model is executable;
* replication reports with relatedness and independence recorded;
* autonomous rival labs (AIC polynomial fits) and replication labs (calibrate, then measure a grid).

Probe `p02_science.py`, 100 simulated days, 0.5 s:
* the hidden law text and the gain never appear in `sci.phenomena` or `sci.apparatus`;
* a participant grid run on the true law gives a fit p of 0.099, against 0.0 for a constant model;
* the rivals published 9 papers and replicators filed 22 reports;
* replay and regeneration are equal; invariants are clean; NPC failures 0.

Defects:
* **`sci.replications` and `sci.paper` crash with KeyError 'id'** (`science.py:446-448` reads `r["id"]`, but the
  applier at `science.py:733` pops `id` before inserting the row). The paper view and the replication list, which are
  core tools, are unusable.
* **The replication verdict refutes the true law.** On paper_000010 (the exact hidden law), one independent replicator
  returned "refuted" (p = 2.8e-05) and one "consistent". The chi-square test uses only the stated sd, while the
  readings also contain `noise_rel * |truth|` (`science.py` `pr_complete`). The in-world institution therefore
  mislabels truth. That may be realism ("stated precision is wrong"), but it must be documented and must never be used
  as the evaluator.
* There is **no novel-condition evaluation** (spec 9.5) and no evaluator-side "demonstrate validity" measure, such as
  prediction error of the claimed model on held-out conditions outside the measured ranges.
* The law space is narrow: 4 forms, always exactly 2 variables (`law["uses"] = [x, z]`).
* Rival labs never produce a consistent paper (9 of 9 refuted), so competition is weak.
* `sci.publish` returns a world-computed fit (chi2, p, rel_err). That is legitimate as a statistics service on the
  participant's own data, but the OO audit should confirm it is not treated as reward.

Verdict for the module: **REPAIR** (M). The domain wrapper is **IMPLEMENT**.

### 3.2 Engineering infrastructure (`infra_world.py`): the `fleet` module

Mechanics are rich:
* machines with on/off/boot/fail/repair/order/retire states;
* M/M/1 latency against SLOs;
* network routing using the 4.0 flow physics;
* PUE and thermal effects, breakers, wear-driven hazard, technicians, procurement lead time;
* tariffs (flat, time-of-use, spot), batch jobs;
* tenants who move traffic according to the reliability they experience, and unannounced releases that change CPU cost;
* a controller deployed from a repository through sandboxed code, taking effect in the maintenance window.

Evaluator-side `operator_costs()` and `reliability()` exist (`infra_world.py:1842-1887`). Admission passes, and 10
simulated days take 0.3 s.

Defects (probes `p04*.py`):
* **Degenerate cost structure, a one-call shortcut.** With the defaults (`technicians: 2`, `tech_wage_cents_day: 20000`,
  `fleet_factor 2.3-2.8`, `infra_world.py:76-79`), technician wages are about 83% of all cost and energy is under 2%.
  Over 60 days, `fleet.staff technicians=0` (`infra_world.py:1521`) cuts total cost by 83.8% (seed 5) and 80.8%
  (seed 9), with attainment 0.99997 and 1.0. The fleet is so over-provisioned that failures that are never repaired do
  not matter. Retiring machines is refused while they are all on. "Substantially reduce infrastructure costs while
  preserving reliability" is solved without any of the technical optimization the domain exists to teach.
* **Hidden data reaches participant code.** `fleet.experiment` runs the candidate controller on observations built by
  `_sim_obs`, which include the hidden true `cpu_ms` (`infra_world.py:1381, 1449`); production observations carry only
  noisy telemetry. It also samples the *hidden* demand process. That is a privileged "perfect twin" oracle that does
  not exist on the real internet.
* Following demand with the incumbent controller cut energy 53% but raised SLA credits, so the total did not fall over
  10 days. The tradeoff is real, but nothing shows that a feasible reference succeeds.
* The default world is one site with 10 machines, so scale is tiny. The framework seat is not wired (section 2,
  item 5).

Verdict: **REPAIR** (M). Recalibrate the cost structure and the reliability consequences of understaffing, remove the
hidden-truth leak or make the experiment use only past recorded traffic, and wire the seat.

### 3.3 Labour and organisations (`firms.py`, `_kit.py`)

Autonomous firms with private goals (profit, growth, mission, rival, client):
* they recruit through `labor`, run internal projects, retain staff, negotiate `commit` offers and vote in `inst`;
* `_kit.py` provides `make_brief` (pipeline, chain, wide and diamond task graphs with a critical path),
  `ToolPolicy` (tool-only scripted references) and `TeamOps`.

Probe `p05_firms.py`, 30 days:
* 3 firms, 14 hires, 66 tasks, 51 sales;
* no process failures, invariants clean;
* replay and regeneration equal.

Defects:
* **Revenue is unconditional.** Finished internal work is paid from `acct_external` at hours × rate × quality
  (`firms.py:281-296`), whatever the demand. Firms' cash doubled (5.0M to 10.4M in 30 days) from an infinite buyer.
* Skills outside `commerce.labor.SKILLS` are silently dropped (`firms.py:158`; my config with "backend"/"frontend"
  silently became the default skills).
* There is no labour or organisational domain definition, no evaluator for "independently evaluated project" or
  "shared measurable objective", and no reference that uses `TeamOps`.

Verdict: **EXTEND** (M).

### 3.4 Education (`learning.py`, `concepts.py`, `common.py`): the `learn` module

The design is sound in outline:
* executable concepts with a generator, parser, solver and typical misconception (`concepts.py`);
* per-learner mastery, half-life and misconception strength, kept private;
* prerequisites, retention decay;
* placement and periodic assessments on fresh problems that exclude every published problem;
* an evaluator-side `final_scores()`;
* NPC teachers; calibrated slate clicks for discovery.

Runtime (probes `p03*.py`):
* **The world does not work.** `learn.study` returns `internal error: KeyError: 'id'`. `_study` reads `r["id"]`
  (`learning.py:324, 398, 412`), but the applier at `learning.py:675` pops `id`. Every `learn.tick` therefore fails
  (39 `proc.failed` events in 10 days), and **no learner ever studies**: 42-day final scores are identical with and
  without the participant's lessons (0.054 / 0.050).
* With the crash bypassed by an in-memory patch only (`p03c_learning_patched.py`), on 2 seeds × 42 days × 30 learners:

  | Lesson type | Seed 3 | Seed 4 | Notes |
  |---|---|---|---|
  | No lesson | 0.054 | 0.056 | |
  | Correct lessons | 0.097 | 0.110 | |
  | **Wrong-key lessons** | **0.117** | **0.122** | |
  | Prose only | 0.054 | 0.056 | read 205 times, no gain |

  Lessons whose stated answers are wrong produce **more** measured learning than correct ones. For most concepts the
  "buggy" answers really are wrong at levels 2-3 (`p03d_buggy_rate.py`). So the dynamics (trust, motivation and
  misconception terms) reward wrong content: an agent optimizing learner outcomes here learns to publish errors.
* Learners read only the exact lines `Example: <problem> => <answer>` and `Practice: ...`; prose is ignored.
  `learn.check` tells authors exactly which lines will be read, which is a format oracle. The skill a learner acquires
  here (emit this format) does not transfer to real teaching.
* The learning-rate constants claim to be "documented in FIDELITY_REGISTER.md rows education.*"; there are no such
  rows.
* The content is limited to 8-13 arithmetic concepts.

Verdict: **REDESIGN** (L). Fix the crash, then rebuild the learner model so that worse content cannot raise measured
learning. Add positive and negative controls (correct > none > wrong), ground the parameters (for example in published
knowledge-tracing fits), and let artifacts be read through representations rather than one magic line format.

### 3.5 Admission does not catch these bugs

Probe `p07_admission.py`: `worldgen.admission.admit` passes `sci`, `learn` and `fleet` on all 8 checks. It counts
`internal error` only on randomly generated tool calls (`admission.py:73-78`); process failures (`proc.failed`) are not
counted, and random arguments never reach real run or paper ids. Passing admission is no evidence that these modules
work.

## 4. Domain-by-domain status (spec Part 9)

Columns: exists = a `DOMAIN` is defined; reg = registered; mech = real mechanisms; auto = autonomous participants;
open = open-ended actions; eval = independent evaluator; cfg = 3 training configs plus compositional and structural
eval configs; ref = a feasible reference that succeeds; proc = procedural instances.

| # | Domain | exists | reg | mech | auto | open | eval | cfg | ref | proc | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.1 | startup economy | no | no | building block | building block | yes (4.0 tools) | no | no | building block | partial | `commerce.reference.startup_world` + `ReferenceSaaS`: ARR $33.2k in 45 days with 300 customers; idle strategy $0 (`p08`) |
| 9.2 | software ecosystem | no | no | building block | building block | yes | no | no | no | partial | `innovation.worlds.innovation_spec` builds and runs 3 days, 0 failures (`p09`) |
| 9.3 | content | no | no | building blocks (`content`, `creators`, recsys, `aud`) | ? | ? | no | no | no | ? | C1/C2 were interrupted; no scaling toward 10M viewers was shown |
| 9.4 | community | no | no | building block (`community`, 1367 lines) | ? | ? | no | no | no | ? | |
| 9.5 | scientific research | no | no | **yes** (`sci`) | **yes** | partial | no (no novel-condition eval) | no | no | partial (4 law forms) | 2 tool crashes |
| 9.6 | technology invention | no | no | partial (`caps`, `adopt`) | partial | yes | no | no | no | no | |
| 9.7 | engineering infrastructure | no | no | **yes** (`fleet`) | yes (tenants, utility) | yes (controllers) | helpers only | no | no (shortcut dominates) | yes (`make_world`) | |
| 9.8 | industrial economy | no | no | 4.0 `industry` / `freight` / `open_industry` | 4.0 | ? | no | no | no | ? | |
| 9.9 | labor marketplace | no | no | yes (`labor` + `firms`) | yes | yes | no | no | no | partial (`make_brief`) | |
| 9.10 | education | no | no | **broken** (`learn`) | broken | partial | helper (`final_scores`) | no | no | partial | wrong lessons beat correct ones |
| 9.11 | information | no | no | 4.0 `search` only | no | ? | no | no | no | no | |
| 9.12 | organizational | no | no | partial (`firms` + `inst` / `commit`) | yes | yes | no | no | no | no | |
| 9.13 | digital marketplace | no | no | building block (`xchg`) | ? | yes | no | no | no | no | |
| 9.14 | crisis recovery | no | no | 4.0 `shocks` only | no | ? | no | no | no | no | |
| 9.15 | research commercialization | no | no | no | no | no | no | no | no | no | must connect `sci` with commerce |

Summary: 0/15 exist, 0/15 registered, 0/15 have evaluators, configs or references.

## 5. Shortcuts and transfer risks (for an intelligent learner)

* fleet: fire every technician (one call) gives -84% cost with no reliability loss; a perfect-twin experiment oracle
  over the hidden demand process; hidden true CPU costs handed to experimental controllers.
* learn: publish wrong answer keys to "improve learning"; only rigidly formatted lines count; a format-checking oracle
  tool.
* firms: an unconditional infinite external buyer (revenue without customers).
* sci: the in-world replication verdict is miscalibrated; a learner can come to treat refutations as noise.
* The framework's default legacy shaped reward is the change in the evaluator's score, so the evaluator leaks into
  training by default.

## 6. Manifest requirements (workstream I)

All eight are **IMPLEMENT**; nothing currently produces any of the metrics. Criteria are weak for N.1-N.7. Measuring:
* **N.1** by "registered" counts a `DOMAIN` object.
* **N.2** with `validate()` accepts number-only structures.
* **N.3** uses `mechanics()` self-reported by the domain author.
* **N.4** only counts evaluators.
* **N.5** lacks multiple seeds, eval configs, a null baseline that must fail and a shortcut check.
* **N.6** counts seeds, not structural variation.
* **N.7** does not show that eval configs are unseen in training.

Section 7 of the JSON lists the stronger criteria needed. Missing requirements (no SP-9.x entries exist at all; the
manifest has SP-1..8, 10..16, 18, 20, 21):
* per-domain required mechanics (15 lists);
* autonomous participants and open-ended actions per domain;
* "not a task prompt over an unchanged environment";
* 9.3 difficulty levels toward 10M unique viewers;
* 9.10 improvement on new test problems, with content controls;
* 9.15 a science-to-economy causal link;
* 9.16 difficulty from complexity, scarcity and uncertainty rather than failure dice, and feasible without being
  guaranteed;
* per-domain observation-only default, operation without an objective, and evaluator isolation;
* a degenerate-strategy red team per objective;
* reachability through `experience.make(2, ...)`.

## 7. Commands run

* `pytest -q -p no:cacheprovider -x tests/test_v5_domains_framework.py`: 7 passed (fake probe domain only).
* `python -m crucible.acceptance.v5 run --requirement N.1`: "no test files exist yet for this selection".
* `p01_registry.py`: registry `[]`, no `d5_*` families.
* `p02_science.py`: science runs; 2 endpoint crashes; true law refuted once.
* `p03_learning.py` / `p03b`: `learn.study` KeyError; 0 study sessions in any variant.
* `p03c_learning_patched.py` (in-memory bypass): wrong-key lessons beat correct ones on 2 seeds.
* `p03d_buggy_rate.py`: the buggy answers are genuinely wrong at levels 2-3.
* `p04_fleet.py`, `p04b`, `p04c_fleet_levers.py`, `p04d_fleet_seat.py`: fleet runs; technicians=0 shortcut; experiment
  receives hidden cpu_ms; seat not wired.
* `p05_firms.py`: firms run; external buyer; replay and regeneration equal.
* `p06_contract.py`: base/experience contract mismatch; legacy reward is the default.
* `p07_admission.py`: admission passes all three modules despite their crashes.
* `p08_startup_block.py`: commerce startup reference reaches ARR $33.2k in 45 days; idle reaches $0.
* `p09_blocks.py`: `innovation_spec` runs; `community`, `content`, `creators`, `xchg`, `subs`, `pkg`, `industry`,
  `freight` and `shocks` modules are registered.

## 8. Work needed (in order)

1. Choose one domain contract (fix `base.py` or `experience/registry.py`). Register every domain module in
   `domains/modules.py` and use one seat mechanism. Make the default observation-only, with `verify()` on the
   evaluator side.
2. Fix the `sci` and `learn` KeyError bugs. Add `proc.failed` counting and targeted call paths to admission.
3. Redesign the education learner dynamics, with controls correct > none > wrong. Recalibrate fleet costs so that
   understaffing has consequences, and remove the hidden-truth leak. Replace the firms' external buyer with real demand.
4. For each of the 15 domains:
   * a `DOMAIN` with 3 or more structurally different training configs (different modules, couplings, topologies and
     participant types) plus compositional and structural eval configs;
   * a procedural `build` function;
   * a public objective, with operation also possible without one;
   * an independent evaluator (ground truth, unreachable from the learner);
   * an evaluator-side scripted reference that succeeds on several seeds while an idle baseline fails;
   * mechanics probes that are interventions (removing a mechanic changes outcomes), not self-reports.
5. Write `tests/v5/test_gate_N.py` with evidence recording. Include a degenerate-strategy red team per objective and a
   reference-isolation scan (N.n1).
6. Build the 11 missing domains on the existing building blocks: commerce (`subs`, `patrons`, `xchg`, `inst`,
   `commit`), innovation (`pkg`, `adopt`, `caps`), social (`community`, `content`, `creators`, recsys), economy
   (`industry`, `freight`), and 4.0 `search` and `shocks`. Build 9.15 as a real `sci` to `products` / `subs` chain.

Overall effort: **XL**.

## Skeptic review

Reviewer role: skeptic for area I_domains (workstream I). I tried to refute each generous verdict and checked whether the
harsh verdicts are justified. Worktree HEAD is dd745b9 (no domain-code change since a881d18). I modified no existing file
except this appended section and the `skeptic` key of `I_domains.json`. Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/I_domains_skeptic/`. Every probe ran
single-threaded (`OMP_NUM_THREADS=1`) and finished in under 40 s.

### S.0 Summary of challenges

| Subsystem | Auditor | Skeptic | Change |
|---|---|---|---|
| Domain framework | REPAIR | **REPAIR (upheld)** | Runtime confirmation of the reward leak. The "module registration gap" is only partly real. Three new defects. |
| Science (`sci`) | REPAIR | **REPAIR (upheld)** | The replication defect is worse than reported: the true law is refuted 61% of the time. No leaks; isolation holds. |
| Infrastructure (`fleet`) | REPAIR | **REPAIR (upheld, evidence corrected)** | Zero technicians is *not* free beyond about 150 days. One technician is the durable shortcut (-41%). The cpu_ms leak is now exfiltrated at runtime. The seat works on the learner path. |
| Firms + kit | EXTEND | **REPAIR (overturned)** | Revenue is a zero-elasticity money faucet: rate ×100 gives sales ×100, exactly. |
| Education (`learn`) | REDESIGN | **REDESIGN (upheld, rationale overturned)** | "Wrong lessons beat correct ones" is a **grader/concept-library bug**, not inverted learner dynamics. With misconceptions made strictly wrong: wrong 0.019 < none 0.056 < correct 0.107 on 3/3 seeds. The REDESIGN rests on the artifact model instead. |
| Tier 2 definitions (15) | IMPLEMENT | IMPLEMENT (upheld) | The recovery branches hold no extra domain code. |
| Building blocks | EXTEND | EXTEND (upheld) | Caveat: the startup reference is privileged (reads `world.spec.params`, deploys the world's own best solver). |
| Admission | REPAIR | REPAIR (upheld) | Also: the money-conservation check passes the firms faucet. |
| N.n1 criteria | criteria_weak=false | **criteria_weak=true** | `reference_paths_exposed == 0` is satisfied vacuously, and nothing requires references to be observation-limited. |

New cross-cutting findings:

* **Zero tests.** The four kernel modules total about 5,000 lines (`science.py`, `infra_world.py`, `learning.py`,
  `firms.py` plus helpers), and **not one test exercises them**. `grep -rl crucible.domains tests` finds only
  `test_v5_domains_framework.py`, which uses a fake `tiny` domain, plus gate tests that only call
  `experience.registry.domains()`.
* **Not wired in.** Nothing outside `crucible/domains/` uses `sci`, `fleet`, `learn` or `firms`: no world family, no
  `civ` profile, no experience path. They are isolated modules that only hand-built WorldSpecs can reach. None of their
  constants has a FIDELITY_REGISTER row; the register has 0 rows for any domain module.

### S.1 Framework: REPAIR upheld, with runtime evidence and corrections

Probe `s5_framework_contract.py` puts a throwaway domain written to the *registry's* contract (module-level `DOMAIN`,
dict configs, `build(config, seed)`) on `crucible.domains.__path__` at runtime, from the scratch directory. No repository
file is touched.

* `experience.make(2, "zz_skeptic_probe:a", 0)` works end to end. So the experience side of the contract is functional,
  and REPAIR, not REDESIGN, is right for the framework: align `base.Domain` with `registry.py:85-128` (or the reverse).
* **Default reward leak, now measured at runtime.** With the default mode (`legacy_shaped_reward`, `registry.py:99`), one
  `bank.transfer` produced reward **-0.15**, which is exactly the evaluator's score change (0.50 to 0.35). In
  `observation_only` the reward is 0.0, and the observation has no `score`, `success` or `details` key. The OO machinery
  works when selected; the default is wrong.
* **`tiers_available()` flips to `{2: True}`** as soon as *any* module defines a `DOMAIN` attribute
  (`registry.py:70`), here a 20-line SimpleNamespace. Gate S tests that branch on `domains()` would count it.
* **The learner never occupies the domain's seat.** `make()` joins a fresh `p:agent` through `admin.join`.
  `participant_principal` and the `seat` module are never consulted (`inspect.getsource(R.make)`).
* **References are omniscient by contract.** `base.run_reference` calls `d.reference(w, who, obj)` with the live World
  (`base.py:140`), and INTERFACE_CONTRACTS section 9 has the same shape. A reference that may read hidden state does not
  show feasibility for an observation-limited learner. That undermines N.5 as specified ("Reference trajectories
  establish that the environment supports meaningful activity"). `_kit.ToolPolicy` (tool-only) exists but is not
  required.
* `validate()` accepted five configurations that differ only in `n_sites` (1..5), with `problems == []`.
* **Partial overturn of the auditor's item 4 (module registration).** `api.local.ensure_modules` (`api/local.py:175`)
  imports `fleet`, `learn` and `firms` on demand from the `domains` package, and `KernelExperienceEnv` calls it. Probe
  `s6_module_registration.py`:
  * (a) a raw `World.create` with `fleet` after `load_all_modules()` raises `ProtocolError unknown module 'fleet'`;
  * (b) `KernelExperienceEnv` on the same spec works.

  The gap therefore affects raw create/replay in a fresh process, not the learner path. It is still a defect, but minor.

### S.2 Science (`sci`): REPAIR upheld, with a stronger replication finding

Probe `s1_science.py`: 4 law families × 3 seeds, 3 NPC replication labs. The participant publishes:
* the **true** law;
* the true form with every constant ×1.10;
* the true form with every constant ×1.25;
* the rival labs' own AIC polynomial fitted to the same data.

| Model | noise_rel 0.02: consistent / refuted | noise_rel 0.0: consistent / refuted |
|---|---|---|
| true law | **14 / 22 (61% refuted)** | 33 / 3 |
| constants +10% | 0 / 36 | 0 / 36 |
| constants +25% | 0 / 36 | 0 / 36 |
| AIC polynomial | 0 / 36 | 0 / 36 |

* The in-world replication institution has **no false positives (0/108)** but refutes the true law in **most**
  independent replications once relative noise exists. It is not "once", as the auditor reported. The cause is confirmed:
  `fit_stats` uses only `stated_sd` (`science.py:265`), while readings carry `noise_rel*|truth|` (`science.py:515`); the
  verdict is at `science.py:480`.
* The participant's own fit p for the true law ranged from 5e-05 to 0.30, so the published fit is also a weak validity
  signal. A learner here would learn that refutation is uninformative.
* **Leak scan (`s1b_science_leak.py`).** The participant never publishes the law; 3 rivals and 2 replicators are active
  for 60 days. I ran 24 read-only calls across 31 read-only tools (including `world.describe`, `world.help` and
  `world.api`), plus `observe()` and every visible event. None contains the law text, any law constant, the gain
  (1.0817), the offset, the drift or noise_rel. **Hidden-law isolation holds.** This is the one RETAIN-quality property
  I could confirm.
* When the author replicates its own paper on another apparatus, the report is correctly labelled `independent: False`
  (`is_related(x, x)` is True). Sybil labs are not possible through `id.create_org`: an org is related to its founder.
* Both crashes reproduce: `sci.replications` and `sci.paper` raise KeyError 'id'.

### S.3 Infrastructure (`fleet`): REPAIR upheld, evidence corrected

Probe `s2_fleet_horizon.py`: 180 days, seeds 5 and 9, default config with wear on. The 30-day buckets read
attainment / machines failed:

| Strategy (seed 5) | total cost | d60 | d120 | d150 | d180 |
|---|---|---|---|---|---|
| baseline (2 techs) | 8.72M | 0.99998 / 0 | 0.99998 / 0 | 0.99998 / 0 | 0.99999 / 0 |
| technicians=0 | 1.37M | 0.99996 / 2 | 0.99973 / 5 | 0.98442 / 9 | **0.22752 / 10** |
| technicians=0 + follow-demand 1.3 | 1.55M | 0.99624 / 1 | 0.99521 / 3 | 0.99533 / 5 | 0.53887 / 10 |

Seed 9 shows the same pattern: technicians=0 falls to attainment 0.406 with 12 of 12 machines failed by day 180.

* **Correction to the auditor.** "Firing all technicians cuts cost 84% with no reliability loss" is true only for horizons
  under about 120 days. Wear-driven failures accumulate and the fleet collapses between day 150 and day 180. The
  delayed-effects mechanic works. An evaluator whose window is shorter than the wear timescale would accept the
  shortcut, so the domain's evaluator horizon must exceed it. That requirement is missing from the manifest.
* **The durable shortcut is one technician** (`s2b_fleet_staff1.py`): 5.14M vs 8.72M (seed 5) and 5.52M vs 9.10M
  (seed 9), about **-41% over 180 days**, with attainment unchanged and 0 failed machines at every checkpoint. Technical
  optimization (energy is under 2% of cost) still cannot compete with a staffing call. Recalibration is still required.
* **The hidden-data leak is exfiltrated at runtime** (`s2c_fleet_exfil.py`). A candidate controller encodes one digit of
  `obs.services[svc0].cpu_ms` as machines-on. Eight 1-hour `fleet.experiment` calls recovered the hidden true cpu_ms
  **7.4192 exactly** (`infra_world.py:1381, 1449`). Production telemetry shows 7.4795 (about ±0.8%), so the cpu_ms leak
  itself has modest value. The larger privilege is that the twin samples the hidden demand process (`simulate`,
  `infra_world.py:1376`).
* **Partial overturn of "seat not wired".** On the learner path (`KernelExperienceEnv`, `admin.join` with handle
  `agent`), the agent *is* the operator: `fleet.staff` succeeds (`s6`, b2). A second learner in the same world gets
  `lacks fleet.operate` (b3). So the seat works for exactly one learner whose handle matches the config, while the
  framework `seat` module path does not. Multi-agent Tier 2 operation of `fleet` is not supported.

### S.4 Firms: EXTEND overturned to REPAIR

Probe `s3_firms_buyer.py` runs the same orgworld with 2 firms for 30 days at the firms' hourly rate ×1, ×10 and ×100:

| Rate | sales rows | sales (cents) | firm inflows by kind |
|---|---|---|---|
| ×1 | 48 | 10,447,769 | transfer (capital) 10.0M, external_sale 10.4M |
| ×10 | 52 | 104,303,354 | external_sale 104.3M |
| ×100 | 52 | 1,043,033,540 | external_sale 1,043.0M |

Revenue scales **exactly linearly with price**. There is no buyer, no demand curve, no competition and no budget; 100% of
the firms' earned inflows come from `acct_external` (`firms.py:280-296`). A firm at ×100 earns $10.4M a month from 4-6
employees.

This is not "functional but insufficiently comprehensive" (EXTEND). It is a defect in the economic core that makes every
firm unconditionally solvent: solvency, dismissal and band-raise logic never bind. That is **REPAIR**. The labour,
negotiation and governance logic stays as it is, and the faucet is replaced by contracted client demand with budgets.

My config's skill "code" was silently dropped (SKILLS has "software"), which confirms the auditor's silent-drop finding.

### S.5 Education (`learn`): REDESIGN upheld, but the auditor's main evidence is misattributed

My probe `s4_learning_controls.py` uses the same in-memory KeyError bypass, 3 seeds × 42 days × 30 learners, and adds a
**strictly wrong** control: every stated answer is verified wrong by the solver.

| Variant | Mean, seeds 3/4/5 | Mean misconception |
|---|---|---|
| none | 0.056 | 0.00 |
| correct | 0.107 | 0.00 |
| **wrong (100% wrong keys)** | **0.164** | 0.58-0.66 |
| buggy (auditor's) | 0.127 | 0.74-0.76 |
| buggy_only (strictly wrong) | 0.143 | 0.79-0.82 |

The inversion is even stronger with strictly wrong content. Per concept, wrong lessons push frac_simplify to 0.42-0.67,
against 0.02-0.03 for correct lessons.

**Mechanism** (`s4b_learning_buggyfix.py`, `s4d_grader.log`):
* Wrong content raises misconception strength (`learning.py:356-381`, e.g. lines 359 and 379).
* A learner with a misconception answers `con.buggy(p)` (`learning.py:97-98`).
* For several concepts the misconception answer is **graded correct**:

  | Concept | Misconception answer graded correct (L1 / L2 / L3) |
  |---|---|
  | frac_simplify | 100% / 100% / 100% |
  | div | 100% / 0% / 0% |
  | sub | 83% / 34% / 0% |
  | add | 49% / 14% / 5% |
  | gcd | 45% / 18% / 7% |

* The root causes are in the grader and generators:
  * `_norm_frac` normalises with `Fraction`, so for "simplify 12/18" the answers `12/18` and even `24/36` are graded
    correct (`concepts.py:28-33, 93-95, 206-208`);
  * level-1 `div` problems are always exact, so the "ignore the remainder" misconception equals the solution
    (`concepts.py:375`).

**Control.** With every misconception answer made strictly wrong in memory, the ordering is correct on all three seeds:
**wrong 0.015/0.018/0.023 < none 0.047/0.054/0.067 < correct 0.097/0.106/0.118**. The learner dynamics are directionally
sound. The auditor's work item "rebuild the learner model so that worse content cannot raise measured learning" is the
wrong fix. The right fix is:
* form-aware grading;
* generators where misconceptions are genuinely wrong;
* a permanent control test (correct > none > wrong) in the gate.

The REDESIGN verdict still holds, for reasons that do not depend on that bug:
* Learners parse only a fixed line grammar (`Example:` / `Practice:` / `Next:`); prose is never read.
* `learn.check` tells authors exactly which lines count.
* Discovery always uses the fixed query `"<Title> lesson <name>"` (`learning.py:448`).

An agent here learns a file format and a search keyword, not teaching. The forgetting constants are also ungrounded:
with no study, adult **addition** accuracy falls to 0.12-0.17 within 7 weeks (per-concept "none" column), and no
FIDELITY_REGISTER row exists for any of the constants that `learning.py:14, 53` claims are documented.

The unpatched crash reproduces (`s4c_learning_unpatched.log`): 167 `proc.failed` (`learn.tick`, KeyError 'id') in 42
days and **0 sessions** in every variant, including the curriculum document. The world is non-functional as shipped.

### S.6 Building blocks: EXTEND upheld, with a caveat on the 9.1 reference

`commerce/reference.py` `ReferenceSaaS.setup` reads `world.spec.params["need"]` (line 88). It then deploys
`service_repo_files(need, best_tier(need), ...)` (line 95), which is the world's own top-tier reference solver source
(`worlds/digital_civilization.py:25-30`), while the incumbents run weaker tiers. The docstring says the reference "reads
only what its tools return". In fact it uses world-generation internals for both demand discovery and product quality.
Its ARR success (p08) therefore shows that *the world's best solver sells*, not that a tool-limited participant can
discover demand and build a product. Each startup world also has exactly one need, so "demand discovery" is trivial.
This must be fixed before it becomes the 9.1 N.5 reference.

### S.7 Admission and requirements

* **Admission: REPAIR upheld.** Confirmed in code: `admission.py:66-78` counts only `internal error` results of random
  calls. Also, the money-conservation check (`total == 0`) passes the firms faucet, because `acct_external` goes
  negative.
* **N.1-N.7: IMPLEMENT upheld; criteria_weak=true upheld.** I add to N.5: a reference that receives the World object
  (or reads `spec.params`) must not count. Require tool-only references (`_kit.ToolPolicy`) whose inputs are restricted
  to participant observations, checked dynamically.
* **N.n1: criteria_weak false to true.** `reference_paths_exposed == 0` is satisfied vacuously by a scanner that finds
  nothing, or when no references exist. It needs:
  * a positive control: a planted reference exposed through a tool, observation, doc or importable module must be
    detected;
  * coverage of `world.describe` / `world.api` / `world.help` output and sandbox import paths.

**Missing requirements** (in addition to the auditor's list):
* **(a) Evaluator horizon.** Each domain's evaluation window must exceed its slowest delayed effect, as shown by fleet
  wear collapsing at about day 150.
* **(b) Grader validity.** Every executable grader or misconception generator must be validated: misconception answers
  must grade wrong, and form-sensitive tasks must reject unsimplified forms.
* **(c) Learner seat.** A Tier 2 learner must occupy the domain's participant seat, and more than one learner per world
  must be supported.
* **(d) Kernel-module tests.** Unit and integration tests must exist for every domain kernel module; there are currently
  0.
* **(e) Price-responsive demand.** No exogenous, price-insensitive revenue source (a zero-elasticity buyer) may fund
  in-world participants.

### S.8 Commands run (skeptic)

All commands ran from `/workspace/agentrs/.worktrees/c5-complete/crucible` with `PYTHONPATH=src`, `OMP_NUM_THREADS=1` and
`/venv/main/bin/python`. Results:

* `pytest -q -p no:cacheprovider -x tests/test_v5_domains_framework.py`: 7 passed.
* `python -m crucible.acceptance.v5 run --requirement N.1`: "no test files exist yet for this selection".
* `s1_science.py`: replication confusion table above; both endpoint crashes reproduced.
* `s1b_science_leak.py`: 0 hidden-value hits in 24 read-only calls, the observation and the visible events.
* `s2_fleet_horizon.py 180`: technicians=0 collapses by day 180 (attainment 0.23 / 0.41); baseline stays at 0.99999+.
* `s2b_fleet_staff1.py`: one technician costs 41% less over 180 days with no reliability loss.
* `s2c_fleet_exfil.py`: hidden cpu_ms 7.4192 recovered exactly through `fleet.experiment` results.
* `s3_firms_buyer.py`: sales scale linearly with price (×1 / ×10 / ×100).
* `s4_learning_controls.py 3,4,5`: strict-wrong 0.164 > correct 0.107 > none 0.056.
* `s4b_learning_buggyfix.py`: misconception grading table above; with strict misconceptions, wrong 0.019 < none 0.056
  < correct 0.107.
* `s4c_learning_unpatched.py`: 167 proc.failed and 0 sessions.
* `s4d` (inline): `grade('12/18', simplify 12/18)` is True.
* `s5_framework_contract.py`: legacy reward -0.15 equals the evaluator delta; observation_only gives 0.0 with no
  evaluator keys; `validate([])` passes numbers-only configs; the reference gets the World.
* `s6_module_registration.py`: raw World.create fails while KernelExperienceEnv works; the agent is operator and a
  second learner is refused.
* `git diff feat/crucible-5-complete...recovery/c5-I{1..4}-*`: no extra domain files on the recovery branches.
