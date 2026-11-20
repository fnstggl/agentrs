# Goals, rewards and scientific evaluation

Code: `crucible/src/crucible/goals/` (metrics, contracts, curriculum, self-goals) and
`crucible/src/crucible/evaluation/` (harness, audit, counterfactual, emergence). Tests: `tests/test_goals.py`,
`tests/test_tooling.py`.

## 1. Four separate things

| | what | who can see it |
|---|---|---|
| training reward | `RewardContract.step_reward`: potential-based shaping on a verified progress metric + a one-time success bonus | the learner |
| world-state consequences | balances, customers, services, contracts, reputation ... | partially, through observations |
| scientific metrics | `RunResult`: success, time to success, final progress, progress AUC, actions, rejections, compute, money spent, integrity findings, wall time, policy usage | the evaluator |
| integrity constraints | `evaluation/audit.py`: self-dealing, circular flows, operator-tool attempts, shell organisations, breached contracts | the evaluator |

## 2. Goal contracts (`goals/contracts.py`)

A goal states an outcome (a predicate over metrics), initial conditions (world family + seed, endowment), budgets
and a horizon, optionally "sustained for k daily checks" — never actions or strategies. Implemented families
(4 difficulty levels each; parameters scale targets, sustain periods, horizons and endowments):

| family | outcome | top level |
|---|---|---|
| `revenue` | independent revenue over a 30-day window ≥ target, sustained | $100,000 / 30 days sustained for 14 checks within 180 days |
| `recruit` | ≥ N independent simulated people as members for ≥ T days, no unpaid employment | 10 people for 30 days |
| `service` | ≥ k successful requests in 7 days with availability ≥ 0.95 | 200,000 requests |
| `solvency` | independent weekly revenue at every daily check for D days | 90 days |
| (lab) | engineering-lab improvement over the baseline controller | see engineering lab docs |

Generated goals differ in world family (mechanisms), seed (population, firms, prices), targets, horizons and
endowments. Every generated goal passes `GoalContract.validate`, and success that is already true at the start
is never counted (`initially_true` blocks it; tested).

**Independent metrics.** `independent_revenue` counts only money from counterparties unrelated to the principal
(not itself, its organisations, their members, or organisations sharing a member, or their service principals),
nets inflows against outflows per counterparty (so circular payments cancel), excludes escrow until released,
and counts the household sector as independent demand. `independent_members` counts only simulated persons who
are not agent-controlled, with tenure, and without breached employment contracts.

## 3. Reward-hacking checks (tested in `tests/test_goals.py`)

| exploit | outcome |
|---|---|
| paying your own organisation / between your organisations / invoices between related orgs | 0 revenue; flagged `self_dealing` |
| circular payments with an unrelated accomplice | nets to 0 |
| fake customers | agents cannot create persons; organisations they create are related parties |
| rating spam | only paying customers may rate, once each; non-customers get `forbidden` |
| inactivity | idle policy: progress 0, reward 0 |
| early termination | no terminal reward without verified success |
| cycling to farm shaping | potential returns to its value: total shaping reward of a cycle is exactly 0 |
| pre-existing success | not counted |
| operator tools | rejected; counted in the audit |
| manipulating a judge | there is no LLM judge anywhere |
| destroying others' resources | not possible without authorisation; goals use absolute, not relative, metrics |

In the tests, the random and idle policies were run on the level-0 revenue goal and never made progress (0.0); broader random-policy sweeps are in EXPERIMENTS.md.

## 4. Autocurriculum and self-generated goals

`GoalSampler` chooses situations (goal family × level × world family × seed) by learning progress, posterior
uncertainty and novelty, filters arms outside a useful success band, and reserves a fraction of samples for the
hardest (frontier) arms. `goals.propose` lets an agent register private predicates over its own observable
metrics; outcomes are told only to the agent and never change independent success.

## 5. Partitions and independence

* **Train / val / test world families** (WORLD_GENERATION.md): test families hold out whole problem formats and
  mechanism settings.
* **Locked test**: `locked_test_goals(ack, purpose)` requires the exact acknowledgement string and appends every
  access to an access log.
* The evaluator's state (goal verifiers, sustain counters) lives outside the world; agents cannot read or modify
  it; evaluator-only measurements (e.g. fork-based counterfactuals) run in forks and never touch the canonical
  world.
* **Paired evaluation**: `paired(a, b)` compares candidates on identical goal instances (same world seed).
  Architecture-discovery candidates must be compared with independent seeds before claiming an improvement.

## 6. Counterfactuals and emergence

`evaluation/counterfactual.py` forks a world twice, applies a recorded intervention to one fork (remove a service,
suspend a principal, change a parameter) and compares outcomes; under the deterministic cost model the two forks
differ only by the intervention. `evaluation/emergence.py` measures tool reuse (direct calls and nested program
calls into agent-built hosts), diffusion curves, market concentration (HHI), specialisation, strategy diversity
and artifact dependency graphs; novelty is counted as useful only when its counterfactual removal reduces an
outcome.
