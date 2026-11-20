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

## 7. Observability and causal debugging



Code: `crucible/src/crucible/cli.py`, `analytics/` (`trace.py`, `metrics_export.py`), `dashboard/app.py` (+
`templates/`), `training/` (`trajectories.py`, `league.py`; see `SELF_PLAY.md`), `evaluation/counterfactual.py`,
`evaluation/emergence.py`. Tests: `crucible/tests/test_tooling.py`.

### 1. The `crucible` command

Installed as a console script (`crucible = "crucible.cli:main"`); without installation use
`PYTHONPATH=crucible/src python -m crucible.cli ...`. Every command takes `--store URL` (`memory://`,
`sqlite:///path.db`, `postgresql://...`; default `$CRUCIBLE_STORE` or `sqlite:///./crucible-data/crucible.db`, the
directory is created) and `--json` for machine-readable output.

| command | what it does |
|---|---|
| `world create --family F --seed S [--id ID] [--deterministic] [--param k=v]` | generate and persist a world |
| `world list` / `world inspect ID` | worlds with events, snapshots, parents / time, actors, orgs, balances, services, recent events, invariants |
| `world run ID --days N [--snapshot-every-days K]` | advance a persistent world hour by hour; snapshots periodically and at the end; Ctrl-C finishes the step, snapshots and exits (exit 0, `"interrupted": true`); the next run resumes via `World.load` (latest snapshot + replay) |
| `world snapshot ID`, `world fork SNAPSHOT|ID [--id F]` | immutable snapshot / copy-on-snapshot fork (lineage recorded) |
| `world replay ID [--verify]` | replay the full log (following fork lineage) from genesis; `--verify` checks the hash chain and that the replayed digest and final hash equal a fresh snapshot+tail load |
| `world verify ID [--no-regen]` | invariants + regeneration (re-execute recorded commands and clock moves through the handlers; identical hashes and digest) |
| `actor join ID HANDLE [--endowment C]` | recorded `admin.join`; prints the bearer token once |
| `actor run ID HANDLE --policy SPEC --days N [--trajectory f.jsonl|f.npz]` | run a scripted policy for an actor in the persisted world (flushes continuously, holds the writer lease, snapshots at the end) |
| `experiment run --goal G --world F --level L --seeds a,b --policies p1,p2 [--max-days D] [--out r.json]` | `evaluation.harness.run_goal` per (seed, policy): summaries with bootstrap CIs, paired differences vs the first policy, per-run reward curves |
| `selfplay run ...` | league of policies in shared worlds (`SELF_PLAY.md`) |
| `evaluate --policy SPEC [--split val] [--world-families ...]` | the validation (or train) goal suite; the locked test split is not reachable from the CLI |
| `trace ID ENTITY` / `trace ID '#SEQ'` | why-explanations and timelines / causal chain of one event |
| `report RESULTS.json` | markdown tables for experiment, league, evaluation or counterfactual JSON |
| `counterfactual ID --undeploy HOST [--close-listings HOST] [--suspend P] [--set-param k=v] [--agent H=SPEC] --days N` | `fork_compare` from the CLI |
| `emergence ID` | emergence metrics report |
| `dashboard [ID] [--port 8050] [--experiments f...] [--trajectories f...] [--operator]` | read-only dashboard on 127.0.0.1 |
| `serve [--port 8000] [--world W]` | the agent API HTTP server (`api.http_server`) |

Policy specs: `name[:key=value...]`, e.g. `builder:need=route.shortest:undercut=0.8:advertise=false`.
Scripted policies keep their plan in memory: a new `actor run` restarts the script on the persisted world.

### 2. Metrics export

`analytics.metrics_export.world_metrics(world, window_s=DAY)` -> JSON snapshot; `prometheus_text(world)` -> text
exposition (`crucible_*` families labelled with `world`): time, seq, principals by kind, persons by kind,
organisations by status, money by sector (households, genesis/external/fees system accounts, escrow, persons,
firms, incumbent firms, providers, lenders, services) and in circulation, transactions per day / volume / by kind
over the window, deployments by status and services live, instances and rented cores, provider capacity and
outages, requests per host by outcome and service revenue over the window, customers per host and unserved
consumers, tickets, contracts, claims and listings by status, and kernel counters (commands, rejected, events,
process runs) with an events-per-second rate measured between two calls. The dashboard serves it at
`/w/<id>/metrics`.

### 3. Causal debugging (`analytics.trace`)

* `timeline(world, entity, since=0)` - events mentioning the entity or its aliases (an organisation's accounts,
  hosts, service principals, deployments, instances, listings; a person's owned organisations), bookkeeping
  events excluded unless `include_noise`.
* `explain(world, entity)` returns `{entity, kind, headline, facts, findings, series, text}`:
  * **organisation** - status and distress; revenue vs costs (7-day windows, trend, cost breakdown); claims
    against it, breached contracts, unpaid fees and invoices; instance suspensions and terminations (failed
    compute bills), stopped deployments, provider outages; customers lost per host (from the consumer market's
    hourly slot summaries), consumer-request failure rates, rival entries, support tickets. For a bankrupt company
    the headline lists the most severe preceding causes.
  * **host** - customers over time and market share/rank, price vs rivals, verified ratings, ad campaigns and
    impressions, organic and sponsored search appearances (from `consumers.hour`), reliability; and every
    customer-loss episode with what preceded it (failure / overload / unavailability counts in the prior 48 h,
    price changes, provider outages, new rivals).
  * **account** - flows by counterparty; **person** - organisations, tools used, rejected actions by error code.
* `causal_chain(world, seq)` - the transaction that emitted the event (command with actor/tool/args, or scheduled
  process run), who scheduled that process (periodic jobs stop there), and recursively the creation events of
  entities referenced in the payload or the command's arguments.

Example (tiny world seed 3; in a fork the scripted Builder's company account is emptied right after the hourly
consumer market paid it, then 2 days pass):

```
o:alice-labs is active: cash $0.00, revenue $35.08 in the last 7 days ...; risks: instance inst_000003 suspended:
payment failed: acct_000018 has $0.00 available, needs $2.00; ...; alice-shorte: 98% of consumer requests failed
(errors 0, unavailable 4337, overload 0)
alice-shorte.crucible (route.shortest, operated by o:alice-labs): 116 customers now, peak 155 on day 1; market
share 30% (rank 2/2); 2 customer-loss episode(s), latest day 3: 97% of requests failed (errors 0, unavailable
3781, overload 0)
```

### 4. Dashboard

`dashboard.app.make_app(store=..., worlds=[...], world=<live World>, experiments=[...], trajectories=[...],
operator=False)`: overview, actors, organisations, economy (balance distribution, Gini, transactions with
filters), communication (counts by type and visibility), services and infrastructure, event stream (filters: type
prefix, actor, entity, since, bookkeeping), trajectories (returns, tool histograms), lineage (ancestry, forks,
snapshots), evaluation (invariants, digest, metrics; emergence on demand), trace / explain / causal chain,
experiments (reports, reward curves, league ratings and score histories); JSON under `/api/...`.

Guarantees: GET only (other methods: 405); worlds are loaded by the dashboard itself (no writer lease) behind a
read-only store proxy; new events written by another process are applied incrementally with hash-chain checks.
Private communication (mail, docs, calendar, support messages, issues, contract notes, and commands to those
services, plus recorded program RPCs) is shown as counts and shapes only; payloads appear only with
`--operator`. The CLI binds to loopback unless `--allow-remote`.

### 5. Training data (`training.trajectories`)

* `RecordingRunner` (a `Runner`) additionally records available tool names, seq before/after, policy kind/version.
* `export_jsonl` / `export_npz` (records `crucible-traj/1`: observation, tools, action, tool, args, result, ok,
  reward, events, timing, policy id/version, world id, spec hash, seq) and `event_records` /
  `export_events_jsonl` / `export_events_npz` for world logs (optionally only what one actor may see).
* NPZ: numeric columns + categorical codes with vocabularies + JSON blobs with offsets; `load_npz` restores the
  JSONL records exactly; `columns()` gives the raw arrays.
* `ImitationDataset(source)` -> (rendered observation text incl. tool names, target action JSON) pairs;
  `to_jsonl` writes prompt/completion files.

### 6. Counterfactuals (`evaluation.counterfactual`)

`fork_compare(world, intervention, horizon_s, agents=None, check_determinism=False)` forks the world twice from
one snapshot blob (in memory; the source world is untouched), applies the intervention to one fork as recorded
system calls (`undeploy_host`, `close_listings`, `suspend_principal`, `system_call`, `set_params` - recorded as an
internal note, `combine`, `custom`), advances both on the same schedule (optionally with fresh deterministic
scripted agents from the same factories and seed), and returns baseline / treatment / delta for customers, shares
and HHI per need, independent and sales revenue per organisation, requests per host, household spending, unserved
consumers, plus the requests served by services that depended on the removed host.

**Exactness.** Use `params.cost_model = "deterministic"` (`crucible world create --deterministic`): sandboxed and
trusted code then report executed-line counts instead of measured CPU, so capacity/overload and therefore the
future are a function of the snapshot alone; `check_determinism=True` verifies that a second baseline fork reaches
the identical digest (`exact` reports the cost model). Note that random draws are keyed by event sequence numbers,
so after an intervention the treatment fork draws *different* random numbers from the baseline (no common random
numbers after the intervention point): small deltas in unrelated markets are sampling noise; compare over several
seeds/horizons before reading them.

Measured (tiny world seed 3, scripted Builder `alice-shorte` live for 1.25 days, undeployed in one fork):
over 12 h `o:alice-labs` revenue $54.52 -> $0, customers 218 -> 169, incumbent `shorte1` 113 -> 158
(+45), `o:shorteco1` +$2.00; over 24 h alice $123.84 -> $0, customers 257 -> 169, `shorte1` 117 -> 200 (+83),
`o:shorteco1` +$19.28. Removing the incumbent a Composer resells (`shorte1`) reduces the requests the composer
(`svc:bob-mesh`) serves.

### 7. Emergence metrics (`evaluation.emergence`)

`tool_reuse` (distinct direct and nested callers of agent-deployed hosts; nested calls attributed through the
ledger for priced calls - the calling service principal is the transaction's `by` - and through the outer
request's recorded RPC list for unpriced calls), `diffusion_curves` (customers per host per day + peak, t10, t50),
`market_concentration` (HHI per need over time and now), `specialization` (roles and offering categories per
organisation; specialists, category entropy), `strategy_diversity` (per-agent tool-category histograms ->
Jensen-Shannon leader clustering: clusters, entropy, mean pairwise divergence), `dependency_graph` (repo depends_on
repo, host deploys repo, host calls host, principal uses host, org operates host; longest service chain),
`useful_novelty` (an agent artifact counts only if removing it in a counterfactual fork reduces households'
served requests, served consumers or its dependents' served requests), `emergence_report` (all but novelty).

Limitations: nested unpriced calls made from inside a scheduled process (e.g. a composer serving simulated
consumers) are counted for the call graph only when priced; analytics scan the event log (use `since` on long
worlds); explanations are heuristics over recorded facts, not proofs of causation - use `fork_compare` to test a
causal claim.
