# SG — autonomous skill acquisition (amendment item 6) and goal-free learning (item 7)

Workstream SG of the core-intelligence amendment of 2026-11-28 (`docs/amendments/2026-11-28_autonomy_and_core_intelligence.md`).
Owned: `src/cam_live/harness/skills.py`, `src/cam_live/reward/selfgoals.py`, `tests/harness/skills/**`,
`tests/reward/test_sg_*.py`, this file, `docs/status/SG.json`, `docs/change_requests/SG.md`. Agent-side hooks:
`docs/change_requests/SG.md` (for ST). Start: HEAD `d1d06dc`. Commits `cam live [SG]`.

Scale words: **controlled** = the real structured browser (Playwright/Chromium) on the local staging sites through the
real authorization service, actions issued by a script in exactly the form of the agent's grounded candidates (a
mechanism measurement, not a measurement of CAM's choices); **real configuration** = Qwen3-4B + 50m core, open-ended
generation on, the real runtime (`runtime.loop.LiveRuntime`) on the staging computer, one GPU-lock hold per run.

## 1. Starting state (measured, not assumed)

* **Item 6.** In every run before SG, skill induction never produced a validated skill (INTEGRATION.md §8, KF-X03,
  KF-X18): debug-scale staging runs 9 of 9 induce attempts failed validation; the real-configuration X4 runs on
  48600c4 (`docs/evidence/S/v3/old_snapshot_48600c4/x4_skills_s{0,1}.json`) chose `skill.induce` 3 times, each
  rejected — `step 0 (browser.fill) failed: observation obs_… is outdated for tab t4: inspect the page again` (twice)
  and `step 2 (browser.wait_for) failed: the awaited state was not reached within 10.0 s`. Skills against primitives
  was therefore never evaluated on browser tasks (CU-33 unclaimed).
* **Item 7.** `GoalFreeMode.candidates` set `transfer=1.0` for every candidate (AE-04 partial, KF-E21, KF-V10);
  `usefulness()` existed but nothing in the runtime fed it.

## 2. Diagnosis of item 6 (reproduced before any change)

Probe `scratchpad/SG/probe_induce.py` (real browser, staging entry book, actions in the agent's grounded form
`{ref, obs_id, tab, ...}`, then `skill.induce` over their action ids), on the code of HEAD `d1d06dc`:

| traces | result at `d1d06dc` | cause |
|---|---|---|
| 1 | `step 0 (browser.fill) failed: observation obs_… is outdated for tab t1: inspect the page again` | the replayed step carries the source observation's id; the browser refuses element refs of an outdated observation |
| 2 | `step 0 (browser.fill) failed: malformed element ref 'e1-629f6fbe'` | the refs differed between the traces (`e1` / `e5`), so the inducer made `ref` a *parameter*; validation's fresh value was a random string |

Both are architectural defects of the inducer, not empirical limits: an element reference is a handle valid only inside
the observation it was read from, and the inducer treated it as a constant or as data. Further defects found on the way
(each reproduced, each fixed, each pinned by a test):

* postconditions were derived from the observation's own bookkeeping (`diff/elements_changed/# changes by +2`,
  `diff/document_changed is False beforehand`) — properties of how the observation was produced relative to the
  previous one, not of the world;
* a single example cannot separate an effect from a size that depends on the arguments (`text_len changes by +25`
  failed every validation with a fresh, longer name — also when two examples had equally long names);
* with a single trace no parameter could ever be found (no variation, no declared inputs), and a value chosen from a
  select's options was captured *positionally* from the page's element table;
* without a probe the skill had no start state, so delta postconditions were undecidable and the first step's
  precondition could not be checked;
* `skill.run` was protected like the library's management operations, so skills could not be composed;
* found while measuring (each reproduced on the real browser, fixed, pinned): an element's label was "captured" from
  an equal table-header text of an earlier page, so a shorter procedure could not be formed; the radio clicked in each
  trace became a free-text parameter whose fresh value named no element; the page outline count (+1 when a dialog
  opens) was a postcondition that failed when the dialog was already open; and — a false success — "the argument
  appears in the resulting state" held because the typed title was still the value of the text field while the site
  had refused the in-page commit (its rate limit), so a run reported verified success that stored nothing.

## 3. What was built (all in `src/cam_live/harness/skills.py`)

| capability | mechanism |
|---|---|
| re-executable steps | `stable_arguments`: an observation-bound `ref` (+ `obs_id` / `ref@obs`) is replaced by the element's role and name (and `nth` among equally named elements) when the tool accepts those locators; the executor indexes every executed action's observations by id (`_note_observations`, `trace_of`), so the element a buffered step addressed is known. A step whose observation is unknown is refused at induction ("cannot be re-executed outside that observation") instead of failing later. A tab handle used throughout a trace is dropped (the procedure acts in the active tab). |
| parameters | `_Abstraction.flows_into_result` (one trace): a value absent from the content before the procedure and present after it (the text it stored) is an input; a choice among an element's options gets the options as JSON-schema `enum`, and a varying element choice (the radio clicked) gets the observed names of elements of that role; integer-like strings keep their shape in fresh arguments. Positions in a page's control table are never captured, and an element's role / name is never "captured" from equal text elsewhere. |
| conditions | observation bookkeeping (`diff` structure, `state_hash`, `doc`, `focused`, `scroll`, ...) and a page's presentation (outline, accessibility tree, control table, forms, links) are excluded; a count changing by exactly one is a postcondition, any other size and any size-like key (`text_len`, `...bytes`) is *tentative*: validation with fresh arguments keeps it only if every validation run confirmed it (`postconditions_refuted` records the rest), and refuses the skill when nothing observable is left. Text conditions are evaluated on the *content* of the state — never on what is typed into a control — and a choice among displayed options gives none. The precondition of an element-addressing procedure is the element it starts on (`element_present`), not one URL. Without a probe the latest observed state is the start state, and after the last step the environment's side-effect-free `observe()` is taken once more (an effect that appears after the step returned is seen; a text not yet visible at induction is a tentative condition). |
| validation | unchanged rule (fresh arguments, every step ok, postconditions hold); a failure *at* a step that only observes and whose result nothing uses is retried without that step; a failure at an acting step, or the absence of the starting element in the current state, is retried from the first later step the current state supports (the reusable portion of a noisy own trace; conditions derived against the removed prefix become tentative); at most two retries, each logged. |
| learned applicability | `ApplicabilityModel`: Bayesian probit regression of P(success \| state) over generic state features (`state_features`: observation kind, host / path segments, title words, element role:name, quantity names), factorised Gaussian posterior updated by assumed-density filtering (Graepel et al. 2010), prior N(0, 0.25) per feature. Trained by every attributable run and every validation run. A run is refused (`not_applicable`, nothing executed, not counted against reliability) only when P(p < 0.15) >= 0.75; thin evidence keeps the skill being tried. `skill.list` reports P(success here). |
| composition | `skill.run` (only) may be a step of a skill: the called skill must be validated and fixed, a skill cannot contain itself, nesting depth <= 3, the composite's scopes include the called skill's scopes; every primitive is still authorized one by one. |
| discovery | tool `skill.discover`: segments of the executed successful actions that recur (same operations and argument names, non-overlapping, >= 2 times), change something in the world and are not yet a skill are induced from their occurrences (aligned: what varied becomes a parameter), validated and registered. |
| efficiency from experience | `SkillExecutor.refine` (run by `skill.discover`): after `refine_after` = 3 successful runs since the last examination, the procedure without each observation-only, unreferenced step is validated with fresh arguments; the first that passes becomes the next version. |

Nothing here is site- or task-specific: the rules use the observation format (`data.elements` with `ref` / `role` /
`name` / `options`), tool schemas and side-effect declarations, and the executed traces.

## 4. Item 7: learned expected transferable competence (`src/cam_live/reward/selfgoals.py`)

`TransferEstimator` replaces the constant. Evidence: every outcome observed during a later *independently assigned*
objective is scored by the goal-free posterior (the fast layer that absorbed the goal-free observations) and by the
reference prior that posterior started from (`GoalFreeMode.reference`, a clone taken before any goal-free update,
checkpointed), on the same data: `gain = NLL_reference - NLL_goal_free` (nats) — a within-sample counterfactual of
"with versus without the goal-free knowledge". A soft success `(1 + clip(gain / 0.25, -1, 1)) / 2` enters a Beta
posterior; a candidate's estimate pools the evidence points with weights `max(exp((cos(phi, phi_i) - 1) / 0.1),
1 for the same operation|metric key, 1/2 for the same operation)`. Without evidence the estimate is the prior 1/2. The
estimate multiplies the candidate's selection score (uncertainty x achievable progress x transfer x discount x gate /
cost), so goal-free learning shifts towards what was worth something later. The agent hands assigned-objective
outcomes over through change request SG-2 (features from the predictor's cached commitment; no extra model call).
Old checkpoints load: the estimator starts empty and the restored fast state becomes the reference
(`restore_notes`).

## 5. Evidence

### 5.1 Implementation correctness (tests)

| file | what it pins | result |
|---|---|---|
| `tests/harness/skills/test_sg_skills.py` (10 unit + 5 `browser`) | locator rewriting (refs, `ref@obs`, duplicates -> `nth`, unknown observation / no locator -> none); grounded traces -> role/name steps, enum domain, observation bookkeeping excluded, size changes tentative; one-trace parameter from what flows into the result; unknown observation refused at induction; applicability learns state-dependence and gates only when confident; step removal renumbers captures and refuses needed steps; recurring procedures found and induced; composition rules; **browser**: own grounded traces -> validated skill reused with new arguments (3 ops per run, induce cost = its validation run), single-trace skill; reuse on an unseen path, precondition refusal elsewhere (applicability told, reliability not), composite navigate + skill run from another site; `skill.discover` registers the recurring procedure and two refinements take it from 5 to 4 to 3 primitive operations per run; the reusable portion of a noisy own trace; layout B: a choice parameter over the observed radios and no false success; observations the environment shows by itself are indexed | pass (JUnit `docs/evidence/test_results/SG_fix.xml`: 180 passed, 0 failed, 138 s, on `git archive` of HEAD `3daecde`) |
| `tests/harness/skills/test_es_skills.py` (ES) | updated for composition (`skill.induce` / `skill.discover` stay protected; `skill.run` composable) and the new tool | pass |
| `tests/reward/test_sg_transfer.py` (4) | transfer estimate = prior 1/2 without evidence; same mechanism later -> all candidates > 0.7; re-drawn mechanism later -> mean < 1/2 and below the same-mechanism estimate; the estimate changes which goal is selected (score ratio = transfer ratio); persistence; old checkpoints | pass |
| `tests/reward/test_sg_agent_transfer.py` (1) | through the agent API: outcomes of an assigned LabWorld objective after a goal-free period become transfer evidence, none during the goal-free period, survive save / load | pass on HEAD (ST integrated SG-1..3 in `dde7d2d`) |

### 5.2 Integration into the actual runtime

* The skill executor is the `skill` modality of `harness.env.ComputerEnvironment` (`harness.factories.all_executors`):
  every executed action of the runtime reaches `observe_execution` (its observations indexed by id), `skill.run` /
  `skill.induce` / `skill.discover` are tools the agent's candidate generators offer (`agent._skill_candidates` offers
  `skill.induce` over its own verified-progress traces and `skill.run` for validated skills; the schema walker and the
  backbone see `skill.discover` like any tool), every step goes through `ExecContext.check` (authorization service)
  and the environment's own dispatch. Nothing in agent.py had to change for the skill pathway except SG-1
  (composition through traces that contain `skill.run`).
* The transfer evidence needs SG-2 (agent hands over assigned-objective outcomes) and SG-3 (reference prior on
  restore). ST integrated SG-1..3 into agent.py in `dde7d2d` (with IMP, PS-1, RW-1/2; ST's agent tests 33 passed).
  The agent-level test `tests/reward/test_sg_agent_transfer.py` passes on HEAD `0b48cfd`: after a goal-free LabWorld
  period, outcomes of the assigned objective become transfer evidence (none during the goal-free period) and survive
  a restart with the reference prior restored. (SG's own run of `tests/integration/test_agent_runtime.py` +
  `test_int2_checkpoints.py` on its patched copy was stopped under the master's CPU rule before it finished.)
* The real-configuration runs (§6) ran the real runtime (`runtime.loop.LiveRuntime` through
  `eval.oe_runner.run_objective`) on the persistent staging computer, from `git archive` of HEAD `9656770` (ST's
  integrated agent with every amendment hook applied so far, the queued GPU lock) with `harness/skills.py` of
  `1da315a`.
* A debug-scale runtime run (`scratchpad/SG/runs/debug_skills.json`: tiny random backbone on CPU, 3 objectives
  "store two more entries", 15 decisions each, real browser and authorization service) found an integration defect:
  CAM chose `skill.discover` by itself twice; the second call found a recurring procedure (two clicks, 3 occurrences)
  and induction refused every occurrence with "observation ... is not known here" — the grounded clicks named
  elements of the observations the runtime takes of the environment every cycle (`observe()`), which the skill
  executor never saw (only action results). Fixed in `1da315a` (the executor indexes `observe()` too; test). No
  objective was achieved at debug scale (the tiny backbone's policy filled the form in 2 of 45 decisions), so no
  verified-progress trace arose there.

### 5.3 Controlled mechanism measurement (CU-33 comparison; `docs/evidence/SG/skills_browser_measure.{py,json}`)

Real structured browser on the staging sites, real authorization service, skills induced from two own grounded
traces (refs + observation ids, as the agent issues them), 10 tasks per arm, final skill code `07303fe`, actions
paced at 2 s so that the site rate limit of the authorization service (60 operations / 60 s, in-page requests
included) is not the main thing measured:

| task | arm | success | decisions / task | primitive ops / task | latency / task (mean) |
|---|---|---|---|---|---|
| store an entry (entry book, layout A) | primitive only | 10 / 10 | 4 | 4 | 0.73 s |
| | acquired skill | 10 / 10 | **1** | 3 | 0.64 s |
| record an item (3-page wizard) | primitive only | 10 / 10 | 6 | 6 | 1.70 s |
| | acquired skill | 8 / 10 | **1** | 5.4 | 1.35 s |
| store an entry, layout B (dialog, in-page commit) | primitive only | 10 / 10 | 5 | 5 | 0.93 s |
| | acquired skill | — | — | — | no skill registered |

Acquisition: one `skill.induce` decision over the two source traces (8 resp. 12 source decisions); its validation run
cost 3 resp. 6 primitive operations. The two failed wizard runs were stopped by the authorization service's rate limit
after 3 operations (a refusal, correctly not retried by the skill; the harness then waited); none failed by itself.
Reuse on an unfamiliar path (the same form at `/app/`, never seen by the skill): ok. After the interface changed to
layout B every run of the layout-A skill was refused by its precondition before anything executed; acquiring a
layout-B skill from two own traces there was **refused by validation**: the commit's effect (an in-page update) is not
in any observation the procedure makes, and the only text condition left would have been the title still typed into
the field — which, before the content rule, produced runs reported as verified that stored nothing (the false
success described in §2). The gain is in *decisions*: one decision of the real configuration costs ~20 s of backbone
generation and evaluation (X4: 389 s for 20 decisions), a browser operation ~0.2 s, so 4 -> 1 and 6 -> 1 decisions
dominate; primitive operations and harness latency hardly change (the skill executes the same operations, minus the
look at the page).

Efficiency from experience (browser test `test_discover_registers_a_recurring_procedure_and_refines_it_with_experience`):
a procedure executed three times with a needless look at the page and a needless read became a skill by
`skill.discover` (5 primitive operations per run); after 3 successful runs each, two refinements validated shorter
versions: 5 -> 4 -> 3 operations per run (versions 1, 2, 3 on disk).

### 5.4 Goal-free transfer diagnostic (`docs/evidence/SG/transfer_labworld.{py,json}`; debug model, LabWorld)

Per seed: 24 self-chosen goal-free probes, then 12 outcomes of a later objective in the same world (mechanism
persists) or in a re-drawn world. gain = NLL_reference - NLL_goal_free on the same later outcome.

| later objective | mean gain (nats) | seeds with positive mean gain | transfer estimate after (mean) |
|---|---|---|---|
| same mechanism | +1.11 | 5 / 5 | 0.80 |
| re-drawn mechanism | -0.57 | 1 / 5 | 0.46 |

Before any later evidence every candidate's estimate is 0.5 (the constant 1.0 is gone). This is a diagnostic of the
estimator on the debug model: it shows the estimate measures what it should (goal-free knowledge helps predict a later
objective exactly when what was learned still holds), not that goal-free learning helps CAM on real objectives.

### 5.5 Goal-free ablation, debug scale (`docs/evidence/SG/goalfree_ablation_lab.{py,json}`; LabWorld, agent API)

Pairs with the same world seed and agent seed: arm "goalfree" spends 8 decisions without an objective (goal-free mode
chooses its own learning goals), then gets the objective "reach 30 adopters" for 16 decisions; arm "none" gets the
objective at once. Measured: verified adopters gained during the objective (after 12 h of drain).

* **First design (one world for both phases; `goalfree_ablation_lab_shared_world.json`, 10 pairs)**: goal-free arm
  9.2 adopters on average vs 20.4 without; paired difference -11.2 (bootstrap 95 % interval [-19.8, -3.5]),
  positive in 2 of 10 pairs. Diagnosis: a design
  confound and an architectural weakness. The goal-free period consumed the world the objective then had to work in
  (its `wait`s used up the objective's time, its interventions the shared budget), and goal-free mode spent its 8
  decisions mostly on `wait` (47 of 80) and the static gauge (`lab.noise`, the noisy TV; 22 of 80), probing a
  channel twice and intervening 9 times in 80 decisions: in a period this short the repeat discount and the noisy-TV gate have not yet
  learned that these teach nothing, and "does a wait execute" is genuinely uncertain at the start.
* **Second design (the goal-free period on its own instance of the same world, the objective on a fresh one)**:
  goal-free arm 17.3 adopters vs 21.9 without; paired difference
  -4.6 (bootstrap 95 % interval [-10.6, +1.9]), positive in 3 of 10 pairs — **no improvement**, a slight harm that the
  interval does not separate from zero. The goal-free periods were again mostly `wait` (45 of 80 decisions) and the
  static gauge (20), with 4 probes and 11 interventions, and the objectives' outcomes produced 5-14 transfer evidence
  points per pair. Diagnosis: with 8 decisions the period yields almost no informative experience (the discount and
  the noisy-TV gate need repetitions to learn what teaches nothing, and the learned transfer estimate needs later
  objectives before it can redirect goal choice), while the experience it does yield is trained into the shared
  weights; an empirical result about short goal-free periods at debug scale, not evidence about the full model.

## 6. Real configuration (Qwen3-4B + 50m core, open-ended generation on)

Runs: `scratchpad/SG/sg_runner.py` — one agent, several objectives inside ONE hold of the queued GPU lock (<= 820 s),
the real runtime on the persistent staging computer (real browser, real authorization service, signing verifier
reading the sites' own state), tree = `git archive` of HEAD `9656770` with `harness/skills.py` of `1da315a`, seed 0,
`nice -n 10`, 4 threads. Summaries: `docs/evidence/SG/real_summary.json` (from `analyse_real.py`), raw result files
copied to `docs/evidence/SG/real/`.

### 6.1 Skills arm (`real/real_skills.json`; one hold, 826 s, peak 11.3 GB, after 32 min in the lock queue)

| objective | verified | decisions | chosen actions that failed | skill decisions |
|---|---|---|---|---|
| store two more entries | 0 of 2 | 12 | 8 of 12 (backbone actions naming a tab `"A"` that does not exist; downloads refused as a private address) | 0 |
| store two more entries | 0 of 2 | 12 | 2 of 12 | 0 |
| record one more item | 3 -> 3 of 4 | 8 | 1 (tab `"items"`) | 1 `skill.discover` (nothing recurring yet) |

**No skill was acquired in the real configuration**, because there was nothing to acquire from: in 32 decisions the
agent never produced verified progress (no entry stored, no item recorded), so no verified-progress trace arose and
the agent offered no `skill.induce`; its one `skill.discover` found no procedure that had recurred. Diagnosis: the
limit is the action choice of the real configuration on this interface (backbone-written browser actions with
invented tab ids `"A"`, `"items"`, CSS selectors given an element ref; grounded fills chosen without the submit), not
the skill pathway — every skill-pathway defect that would have blocked acquisition from such a trace is fixed and
tested on exactly this interface (§5.1, §5.3). This is an empirical negative result for item 6 in the real
configuration, at 32 decisions; the X4 runs on 48600c4 reached verified progress on this task in some objectives
with 20 decisions per objective (`docs/evidence/S/v3/old_snapshot_48600c4/`), so more decisions per objective would
be the next test.

### 6.2 Debug scale through the same runtime (`real/debug_skills.json`, `real/debug_skills2.json`; CPU, tiny backbone)

2 x 3 objectives, 15 decisions each: none achieved. CAM chose `skill.discover` by itself 2 resp. 5 times; in the first
run one call found a recurring two-click procedure (3 occurrences) and **induction refused it — an integration
defect**: the clicks named elements of the observations the runtime takes of the environment every cycle, which the
skill executor did not index. Fixed in `1da315a` (pinned by a test); the second run (with the fix) found nothing
recurring.

### 6.3 Goal-free arm (`real/real_goalfree.json`; one hold, 710 s, after 55 min in the lock queue)

Plan: a goal-free period of 8 decisions on the same staging computer (no success criterion, nothing checked), then
"store two more entries" and "record one more item". Comparison arm: the skills arm's first objective (the same
objective from a fresh agent, same seed, no goal-free period) and its record objective.

| | goal-free arm | without goal-free period (6.1) |
|---|---|---|
| goal-free period | 24 learning goals proposed, 6 resolved (2 achieved), 8 decisions | — |
| store two more entries | 0 of 2 in 12 decisions | 0 of 2 in 12 decisions |
| record one more item | 3 -> 3 of 4 in 12 decisions | 3 -> 3 of 4 in 8 decisions |
| transfer evidence from the objectives' outcomes | 21 outcomes scored (SG-2 in the real runtime) | — |

What goal-free mode chose to learn: every resolved goal was a prediction of the environment's *own execution
statistics* (`observed.exec.modality.visual_heavy.n`, `...recovery_s`, `...expected_cost`) after an action — the only
declared measurements with epistemic uncertainty and achievable progress on this computer; the sites' content offered
it nothing measurable before an objective names a metric. The measured transfer of that knowledge to the objectives'
outcomes (`staging.entries.count`, `staging.items.count` after each action) was small: per key mean gains from
-0.03 nats (`browser.scroll`, 0 of 5 positive) to +0.10 nats (`internal.inspect`, 3 of 3), estimates 0.46-0.62 (the
prior is 0.50). **Usefulness: not measurable here** — neither arm achieved either objective (floor), so the
comparison says nothing about improvement; the pathway (goal-free goals -> later assigned outcomes -> measured gains ->
learned estimates that now weight goal selection) ran in the real configuration.


## 7. Negative results, limitations, diagnosis

* **No skill transfers to a different interface, and on layout B none could be acquired.** A skill addresses
  elements by role and name; on the entry book's layout B (other labels, a dialog) the layout-A skill is refused by its
  precondition before anything executes. Acquiring a layout-B skill from two own traces was refused by validation:
  the effect of its last step (an in-page update after an asynchronous request) is in no observation the procedure
  makes, so nothing observable confirms what it does (diagnosis: an *empirical limit of the observation*, not of the
  inducer — the environment's own `observe()` after the last step showed it in some debug runs and not in others (timing); a procedure would need an
  explicit wait for the effect, which only a trace that contains such a wait gives). Transfer to unfamiliar
  interfaces has to come from the neural policy (which chooses the primitive actions there) or from re-acquisition.
* **What goal-free mode finds to learn is limited by what is measured.** On the staging computer every goal it chose
  was a prediction of the environment's own execution statistics (the only declared measurements with epistemic
  uncertainty); in LabWorld's short periods it chose `wait` and the noisy gauge most of the time. Its candidates
  come from the agent's candidate generators and the questions the environment declares; nothing in it seeks out
  the content of an unfamiliar site before an objective names a metric. The learned transfer estimate is the
  mechanism that can redirect it once later objectives have shown which knowledge paid off — in one real goal-free
  period it had no such evidence yet (it accumulates across periods).
* **Real configuration: no improvement could be measured for either item** — the agent achieved none of its five
  objectives in the two holds (floor), so neither skills (nothing to acquire) nor goal-free experience (no success to
  improve on) had a measurable effect there. One seed per arm, 32 resp. 32 decisions.
* **Postconditions are inferred heuristically.** What counts as an effect (a count changing by one, a parameter
  appearing in the result) versus a hypothesis (any other size, anything derived against a removed prefix) versus not
  a condition at all (observation bookkeeping, page presentation) is a rule over the observation format; validation
  with fresh arguments is the judge, so a wrong rule shows up as a rejected skill (logged in `skills/rejected.jsonl`)
  or as a weaker postcondition — the latter is the risk (a skill that "works" with a postcondition that only checks
  that its text appeared).
* **Validation executes the procedure in the world.** On the staging sites that is an extra stored entry per
  validation; in a live environment every validation run is a real action under the normal authorization (no extra
  permission), which is what spec 12A.8 asks ("test the procedure in an authorized environment") but costs.
* **Refinement and discovery are actions CAM must choose** (`skill.discover`); nothing runs them automatically.
* **Applicability is learned per skill from few runs**; in the runs it never reached a refusal (too few
  failures in one state). Its effect on behaviour is shown in the controlled tests only.
* **Transfer estimate on the real configuration**: the features come from the predictor's commitment under the
  *objective's* goal text while goal-free mode learned under an empty goal; a mismatch shows up as low measured
  transfer, honestly, but it is a limitation of using the cached features (no extra model call).
* **The agent hooks are not applied** (agent.py belongs to ST); the integration evidence uses a patched copy
  (`git archive` of SG's commit + `docs/change_requests/SG.md`).
* **Statistics**: two real-configuration holds (skills arm, goal-free arm; the primitive-only and second
  no-goal-free arms were not run: the shared GPU lock queue gave one hold per ~50 min, and the master asked to queue
  no further holds); they show the pathways ran, not an effect size.
* Shared worktree: other workstreams' uncommitted changes to the authorization service made two Discord live-check
  tests of `tests/harness/skills` fail in the working tree; with SG's files on `git archive HEAD` (no foreign
  changes) every test passes. SG's test runs therefore use that isolated tree.

## 8. The four evidence levels per capability

(a) implementation correctness, (b) integration into the actual runtime, (c) causal influence on behaviour,
(d) measured improvement — or a negative result with its diagnosis.

| capability | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| 6.1 acquisition from CAM's own executed traces | unit + browser tests (§5.1) | `skill` modality of the real environment; the agent's own `skill.induce` over verified-progress traces; real runs §6 | real configuration: none (no verified-progress trace in 32 decisions, so nothing to induce: §6.1); debug scale: CAM chose `skill.discover` by itself 7 times in 90 decisions, one found a recurring procedure whose induction exposed the `observe()` indexing defect (fixed) | controlled: before SG 0 of every induce attempt validated (X4: 3 of 3 rejected); now own grounded traces -> validated skills on both staging tasks |
| 6.2 validation (independent, by execution) | tests: fresh arguments, tentative conditions, reusable portion, no false success | every registration goes through it in the runtime | a refused candidate is never offered (`skill.run` only for validated skills) | layout B: refused instead of a false success (the effect is not observable at the end) |
| 6.3 learned applicability | `ApplicabilityModel` tests (learns state dependence; refuses only when P(p<0.15) >= 0.75) | trained by every runtime run, validation run and precondition failure; reported by `skill.list` | a confident "not applicable" stops a run before anything executes (tested); not exercised in the real runs (no skill existed) | not measured as an improvement (the real runs are too short to reach a refusal) |
| 6.4 efficiency from experience | browser test: 5 -> 4 -> 3 primitive operations over three versions | `skill.discover` is an ordinary tool of the runtime | refinement changes what a run executes (tested) | controlled: decisions per task 4 -> 1 (store) and 6 -> 1 (wizard); real configuration §6: not reached (no skill acquired) |
| 6.5 composition | tests: rules, composite navigate + skill run from another site (4 primitives) | SG-1 (agent traces may contain `skill.run`); executor nesting through the environment's dispatch | a composite runs where its part alone is refused (precondition) | not measured in the real configuration |
| 6.6 reuse in unfamiliar situations | tests: unseen path (`/app/`), unseen arguments (fresh, enum) | as 6.1 | the skill runs on a page it never saw; refused on a different interface | negative: no transfer to a different interface (layout B); re-acquisition there was refused by validation (asynchronous effect) |
| 6.7 procedural vs neural | — | the policy chooses `skill.run` among all candidates like any action | real runs: no skill existed, so `skill.run` was never offered; the policy's own `skill.discover` choices: 1 of 32 (real), 7 of 90 (debug) | not measurable here (no skill in the real configuration); controlled: the procedural gain is 4 -> 1 / 6 -> 1 decisions, which is not neural learning — it does not change what the policy prefers |
| 7.1 gaps, informative experiences, unfamiliar mechanisms | existing `GoalFreeMode` tests (uncertainty x achievable progress, noisy-TV gate, budget) | the runtime's goal-free mode (no success criterion); real run §6 | real: 24 goals proposed, 6 resolved, all about the environment's own execution statistics; LabWorld: the chosen goal-free actions were mostly `wait` (92 of 160) and the noisy gauge (42 of 160) | — |
| 7.2 learned transfer estimate (was 1.0) | `test_sg_transfer.py` (4) | SG-2 / SG-3 hooks (ST); patched copy: agent-level test passes on HEAD (`tests/reward/test_sg_agent_transfer.py`) | the estimate scales the selection score (test: the higher-transfer goal is selected; score ratio = transfer ratio) | LabWorld diagnostic: same mechanism +1.11 nats, 5/5 seeds, estimate 0.80; re-drawn -0.57 nats, 1/5, estimate 0.46; real run §6: 21 later outcomes scored, per-key mean gains -0.03..+0.10 nats, estimates 0.46-0.62 (small transfer) |
| 7.3 usefulness on later unfamiliar objectives (ablation) | — | `CUFREE` period before assigned objectives in the same runtime | the goal-free period changes the later objective's behaviour and result (LabWorld pairs differ; real: same floor) | negative: LabWorld second design paired difference -4.6 adopters [-10.6, +1.9], 3 of 10 positive (first, confounded design -11.2 [-19.8, -3.5]); real: not measurable (neither arm achieved anything) |

## 9. Reproduce

* tests: `PYTHONPATH=<git archive of HEAD>/cam_live/src .venv/bin/python -m pytest <tree>/cam_live/tests/harness/skills
  <tree>/cam_live/tests/reward/test_sg_transfer.py <tree>/cam_live/tests/reward/test_sg_agent_transfer.py
  <tree>/cam_live/tests/reward/test_reward_goalfree.py <tree>/cam_live/tests/reward/test_reward_selfgoals.py
  -m "browser or not browser" --rootdir=<tree>/cam_live` (JUnit `docs/evidence/test_results/SG_fix.xml`)
* controlled measurement: `PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/SG/skills_browser_measure.py --out OUT.json --n 10`
* transfer diagnostic: `... cam_live/docs/evidence/SG/transfer_labworld.py --seeds 0,1,2,3,4 --out OUT.json`
* goal-free ablation: `... cam_live/docs/evidence/SG/goalfree_ablation_lab.py --seeds 0,...,9 --free 8 --decisions 16 --out OUT.json`
* real configuration: `docs/evidence/SG/real/sg_runner.py` (copied run script; `real_queue.sh` / `real_queue2.sh` show
  the exact plans), analysed with `docs/evidence/SG/analyse_real.py`.
