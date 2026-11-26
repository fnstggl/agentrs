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

## Performance notes (wave 3b)

* Class 2: the day review is one batched forward pass per policy group over the members waking in the same decision
  round (`adaptive_window_s`); activity and resource decisions batch the members due in the round.
* Class 3: members decide on their group's common grid (multiples of `decide_every_s`), one forward pass per group.
* `popweb` visit batches are delta-only: columns no row changes are not rewritten and only changed rows are recorded
  (the visit log blob and the resource counters keep every visit); `money_deposit` skips a deposit that would change
  nothing (deposited earlier the same day, nothing accrued).

