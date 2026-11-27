# Workstream B: the autonomous population

Package `crucible/src/crucible/population/`, registered by `crucible.load_all_modules()` through
`population/modules.py`. It implements MASTER_SPEC Part 3 (persistent individuals, the four participant classes, the
profiles, independent learning) and Part 4.1 / 4.4 (browsing with limited attention, content choice), and is the ONE
population of every v5 world (D-15, ONE_WORLD_CONTRACTS section 1). Decisions: D-30 (fitted activation and content
choice), D-31 (class 3 redesign), D-32 (class 2 activity decisions and memory through time).

## What exists

| file | what it is |
|---|---|
| `core.py` | Kernel module `pop`: the columnar table `pop` (one row per individual, all classes), genesis, the activation tick `pop.tick` (day decisions, session start, session length, activity choice, dispatch to registered activities, learning and scheduling batches), `pop.daily`, the public API (`register_activity`, `register_class_runtime` with the class-3/4 binding contract, `wake`, `features`, `spendable`, `session_budget`, `class_members`). |
| `activation.py` | The day's decisions from workstream E's runtime models: `daily_return@1` (active today) and `extra_sessions@1` (sessions per active day), the 8-day history ring they read, the burn-in history at genesis, session scheduling and the session-length rule. |
| `api.py` | `record_decisions` (decision provenance), `experience`, `pay`, interests in the shared space, and the evaluator-side `check_provenance`. |
| `web.py` | Kernel module `popweb`: discovery, visits within the session's time (up to five per session), the article slate decision (`article_open_semantic@1`), purchases through the sites' own checkouts, repurchase intentions (`customer_return@1`), return and abandonment, search, forum. `resource_scores` is the class-1 resource choice. |
| `sites.py` | Kernel module `popsites`: generated sites (rich / thin / offtopic retold from the real corpus; spam keyword lists; broken; flaky). |
| `adaptive.py`, `nets.py` | Kernel module `pop_adaptive`: class 2 - one network per policy group (~0.24M parameters), personal GRU memory per individual, the daily memory step at each member's first activation of the day (`review_day`, active day or not), activity and resource decisions, online learning with sequence (BPTT) training of the memory over the member's own steps, frozen copies. |
| `strategic.py` | Kernel module `pop_strategic`: class 3 - ventures built with ordinary tools by four policy groups (REINFORCE, frozen twin, A2C, PPO; objectives profit / reach; 2.1-2.4M parameters each); `behaviour(world)` measures competition, cooperation, objectives and lifetime adaptation. |
| `models.py` | Plausible class-1 decisions (logistic, hand-set) and the list of fitted runtime decisions (`FITTED`). |
| `choice.py` | Evaluator side: which of the twelve Part 4.4 content-choice factors the decisions respond to (controlled perturbation of each factor's input on real individuals' rows). |
| `analysis.py` | Evaluator side: visit log, return/abandon, preference effects, state fields, browsing behaviours, platform activity (SP-4.0), attention use. |
| `experiments.py` | Class-2 adaptation experiments (identical-init learning vs frozen twins, controlled new condition, held-out loss) and the compact vs language-model comparison (G.7). |
| `fidelity_probes.py` | In-world probes for the register's `default:` claims of the fitted models (registered by workstream E). |
| `profiles.py`, `schema.py`, `semantics.py`, `text.py`, `writing.py`, `worlds.py` | Profiles (Part 3.3), columns and the Part 3.1 field map, the shared-space helpers, page parsing, writing from the real corpus, population world specs. |

## Provenance

Every population decision batch carries `extra.decision` (model id, version, fidelity register id) and is recorded inside
the population's own session (`pop.session`, which covers the whole activated cohort and lists every decision batch
of the session), so `population.check_provenance` and `calibration.inworld.audit` classify it. The fitted models are
used only through `crucible.calibration.runtime`; nothing loads a checkpoint directly.

## Gate tests

`tests/v5/test_gate_B.py` (B.1-B.6, strict standard-profile world with the social internet, 30 days, an active learner,
three intervention forks), `test_gate_G.py` (G.1-G.7, G.n1), `test_gate_SP_3_1.py`, `_3_2.py`, `_3_2c.py`, `_3_3.py`,
`_3_4.py`, `_4_0.py` (civilisation; also checks the Gate B/G mechanisms in the civ world), `_4_1.py`, `_4_4.py`. Every
test asserts the manifest's criteria on the measured values.

G.6 and G.n1 measure different things: G.6 forks each seed's world at the controlled change and stops the learning
group's learning in the fork only (`experiments.adaptation(counterfactual=True)`, D-32 addendum 2), so the paired
per-member difference after the change is what learning under the new condition did; G.n1 compares the learning group
with its identical-init frozen twin after the change (general plus change-specific learning) and reports the activity
decisions and class 3's policy change next to it. Both use the controlled condition "category" (D-32 addendum 3):
rich guides and poor blogs before the switch, two guides failing and every blog rewritten rich after it, so the
change reverses a regularity the network can perceive (the site's category) instead of only per-site facts that a
reader's own satisfaction already tracks for any policy (condition "sites", whose G.6 run failed). SP-3.2 counts class 4 by its members' own tool actions during
activations of the bound runtime, not by the activations.

## Performance notes (wave 3b)

* Class 2: the day review is one batched forward pass per policy group over the members waking in the same decision
  round (`adaptive_window_s`); activity and resource decisions batch the members due in the round.
* Class 3: members decide on their group's common grid (multiples of `decide_every_s`), one forward pass per group.
* `popweb` visit batches are delta-only: columns no row changes are not rewritten and only changed rows are recorded
  (the visit log blob and the resource counters keep every visit); `money_deposit` skips a deposit that would change
  nothing (deposited earlier the same day, nothing accrued).
* A page the population fetches for the first time is represented with ONE call of the shared space for its text and
  its slate's headlines (it was two).
* `popweb.seen` (the resource table's entry-page record) is delta-only: a page served from its cached record rewrites
  nothing. A web session spends its attention in ONE ledger batch at its end (within the session the session budget
  binds; class 2's observation subtracts the minutes still pending).
* Measured (population-only development world, 1,500 individuals, warm text cache, one CPU thread, days 2-4, no
  profiler; `perfcmp.py` of the wave-3b continuation): 55433e7 4.01 ms CPU per web visit (1,270-1,450 visits a day,
  664 simulated days per CPU-hour); this branch 1.50 ms per web visit (5,660-6,840 visits a day: sessions now spend
  real session budgets over up to five visits) and 377 simulated days per CPU-hour; 4184cce (wave 3b before the R6/R7
  work) 1.45 ms per web visit, 382 simulated days per CPU-hour. The R6/R7 work cut the population's batches a day
  from 3,840-4,040 to 2,250-2,370 and its events from 12,800-13,450 to 9,510-9,770, but CPU per visit is unchanged
  within the noise of this heavily loaded machine (load average 20-28 on about 11 CPUs during the runs).
