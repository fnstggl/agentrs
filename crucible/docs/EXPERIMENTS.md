# Experiments and demonstrations (all measured)

Every number here was produced by the scripts named in each section on this machine; raw outputs are in
`docs/experiments/` and `docs/benchmarks/`. Runs are single seeds or a few seeds unless stated; they demonstrate
properties of the environment, not of any learner.

## 1. Reproduction commands

```bash
cd /path/to/agentrs
export PYTHONPATH=crucible/src
python -m pytest -q crucible/tests                                        # full test suite
python crucible/examples/strategy_matrix.py --out crucible/docs/experiments/strategy_matrix.json      # §2
python crucible/examples/strategy_matrix.py --goals outperform --out crucible/docs/experiments/strategy_matrix_outperform_v2.json
python crucible/examples/demo_lifetime.py --days 30 --out crucible/docs/experiments/demo_lifetime.json  # §3
python -m crucible.bench.run --out crucible/docs/benchmarks/bench.json                                 # SCALING.md
```

## 2. Solvability, multiple strategies, exploit resistance (`examples/strategy_matrix.py`)

108 lifetimes: goal families {revenue, service, outperform} at level 0 × train world families {bazaar,
cheap_compute_closed_markets, expensive_compute_open_distribution} × seeds {1, 2} × 6 policies; one simulated hour
per agent step; horizon capped at 21 simulated days; evaluator = `GoalVerifier` on authoritative state.
Wall time 535 s on 8 processes.

| goal (level 0) | Builder | Composer | Employer | BountyHunter | Random | Idle |
|---|---|---|---|---|---|---|
| revenue: ≥ $1,000 independent revenue / 30 days | **6/6** (mean day 5.8) | 0/6 (progress 0.34) | 0/6 | 0/6 | 0/6 | 0/6 |
| service: ≥ 500 successful requests / 7 days, ≥ 95 % availability | **6/6** (day 1.3) | **6/6** (day 5.0) | 0/6 | 0/6 | 0/6 | 0/6 |
| outperform incumbents on a hidden battery (v1 generator) | 2/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |

Findings:
* The world is solvable, and by **two substantially different strategies** under the same objective: writing a
  better service (Builder) and building a service on top of other firms' APIs, paying them per call (Composer). In
  all three structurally different families the service goal was met by both.
* Random and idle policies never succeeded and never made progress (0.00) on any goal; integrity audits flagged
  the random policy for small circular transfers (`circular_flows`, 18 runs) — reported, not rewarded.
* The Employer strategy (hire a simulated developer, assign the work through the issue tracker) did not produce
  revenue within 21 days in any run: hiring takes days, low-skill hires write weak code slowly, salaries start
  after 30 days. It is a legitimate but slower path.
* The BountyHunter earns bounty rewards (personal income) but does not satisfy organisation-revenue goals.
* **Generator bug found by this experiment:** the v1 `outperform` generator produced impossible goals in 4 of 6
  worlds (incumbents already at the best implementation, or the best possible gain below the margin). Generated
  goals now pass an executable feasibility check against reference implementations (`needs.tier_quality`).

**§2b. Outperform, feasibility-checked generator (`strategy_matrix_outperform_v2.json`, 36 runs, 234 s):**
Builder **5/6** (day 1–2), Composer 0/6 (returning the best incumbent answer cannot beat the best incumbent — as
expected), Employer/BountyHunter/Random/Idle 0/6 (progress 0.00). The one Builder failure was a world where no
need had any headroom (both incumbents already at the best tier); the generator then still emitted a lowered
bar. It now rejects such draws (`GoalInfeasible`) and redraws the world seed deterministically
(`params.redraws`).

## 3. A persistent lifetime (`examples/demo_lifetime.py`, family `bazaar`, seed 7, 30 simulated days)

Canonical world on SQLite with snapshots every 5 days; agents: Builder (writes and operates a service, offers a
code licence), BountyHunter, Random, Employer; a Composer joins on day 10 and builds on services it finds.
6,000 consumers, 8 incumbent firms, deterministic cost model.

**v1 (recorded failure, `demo_lifetime_v1_overscaling.json`).** The Builder chose the knapsack market (weakest
incumbent ratings), reached ~300 customers and ~$1,000/week of independent revenue, but its maintenance rule
("scale up on any overload") kept growing its instance from 1 to 8 cores (~$384/day of compute against
~$150/day of revenue). On day 20.75 an hourly compute bill failed (`payment failed: acct_000060 has $15.41
available, needs $16.00`), the instance was suspended, terminated on day 23.75, the service stopped and its
customers returned to incumbents: revenue 0 by day 28. Nothing in the world script anticipated this; it followed
from billing, capacity and demand mechanics — and it is fully explained by the event log.

**v2–v4 (`demo_lifetime.json` is the v4 run).** With a profitability check before scaling (otherwise reprice):
* the Builder earned **$1,368.63** independent revenue over the last 30 days and kept its service live;
* an NPC incumbent (knapsaco2) **bought a licence to the Builder's code and redeployed it** in place of its own
  implementation (diffusion of an agent invention through the market). This exposed a security bug — agent code
  adopted by a "trusted" incumbent ran in-process — fixed by provenance-based trust (SECURITY.md);
* the Composer's program made 3 brokered calls into the Builder's service (agent→agent tool reuse);
* persistence: reload from the store **equal**, full replay from genesis **equal**, and — in v4, after two kernel
  fixes found by this demo (scheduled-process outcomes keyed by transaction start seq; canonical argument key
  order) — **regeneration of the whole 30-day lifetime from recorded commands reproduces the identical event log**
  (73,617 events, identical hash chain, 34 s). In v3 regeneration diverged at seq 16,178 (a random agent's rejected
  `contracts.propose`: the same two validation errors listed in a different order because args read back from
  sorted JSON had a different key order) — the demo's own divergence report pinpointed it;
* causal debugging (`crucible trace canonical o:bob-labs`): "bob-knapsa lost customers: 459 on day 1 → 103 on day
  29; 58 % of consumer requests failed (overload 16,519); costs exceeded revenue over the last 7 days ($293.77 vs
  $229.14; compute $1,420 total)" — the profitability rule refused to scale, raised prices instead, and overload
  plus higher prices drove customers away;
* invariants empty;
* counterfactual (two forks, deterministic cost model, Builder's service removed in one): Builder's 5-day revenue
  $93.15 (control) vs $0 (removed); its customers moved to incumbents; the canonical world was unchanged.

## 4. Engineering laboratory (`tests/test_engineering_lab.py`, 16-node world, SLA 34.1 ms)

| controller in production | mean delay | improvement vs paired baseline | dropped | SLA penalties |
|---|---|---|---|---|
| baseline shortest-path (day 0) | 27.5 ms | 0 | 0.8 % | $667 |
| congestion-aware greedy (agent code, 23 h) | 8.6 ms | +0.69 | 0 % | $0 |
| invalid controller (23 h) | 156 ms | −4.59 | 62 % | $6,756 |

The agent's paid experiment predicted +0.79; family worlds (seed 11): +0.79 in production (train family), +0.63
(held-out scale-free family).

## 5. Headless ≡ browser (`tests/test_web.py`)

A Chromium session logs in, creates an organisation, registers it, opens an account, transfers money, triggers a
rejected overdraft, posts and searches; the same 7 tool calls through `World.execute` on a second world give
identical command sequences, identical state digests and identical event hash chains.

## 6. Reproducibility

* Replay from genesis is digest-exact in every test and demo.
* Regeneration from recorded commands (handlers re-executed, external outcomes injected) reproduces the event log
  exactly — including sandboxed agent code, consumer demand executions and random agents, in both cost models
  (tests and `examples` checks). Recorded outcomes are keyed by transaction start seq so scheduled processes are
  injected too (an earlier gap, fixed). The 30-day lifetime demo's in-process regeneration check: see §3 (v3).
* Under the deterministic cost model, forks from the same snapshot evolve bit-identically.

## 7. Two learning agents in one world: competition and cooperation (`examples/multiagent_learning.py`)

`tiny` family (2,500 consumers, needs `route.shortest` and `pack.knapsack`, one incumbent each, deterministic
cost model), 40 simulated days, seeds 1–3, actions every simulated hour. Learners (`agents/learners.py`) start
from a fixed scripted launch (company, best-tier service, listing) and then learn one daily decision with a
softmax-linear policy trained online by REINFORCE from consequences they observe (their own `cloud.status`
revenue, public market prices). Raw output: `docs/experiments/multiagent_learning.json`.

| condition | joint independent revenue, last 10 days ($/day, seeds 1/2/3) | mean | notes |
|---|---|---|---|
| competition: two PricingLearners, same market | 190.5 / 271.7 / 233.7 | 232.0 | policy weights moved (‖W‖₁ 6.1–9.0); final prices 8/6, 5/5, 3/5 cents |
| control: two fixed-price Builders | 285.6 / 270.2 / 227.4 | 261.1 | — |
| cooperation: two AdLearners, joint campaign (public good) | 376.2 / 206.0 / 255.0 | 279.1 | contribution rate (last 10 days) 1.00 / 0.30 / 0.00 |
| cooperation + unconditional cost-sharing contract | 380.7 / 244.9 / 236.5 | 287.4 | contribution rate 1.00 / 0.25 / 0.00 |

Findings (n = 3; none of the differences is statistically resolved):
* Multi-agent learning through real world mechanics runs and is measurable: competitors' prices and the public
  good contributions respond to experienced consequences; all worlds' invariants held.
* The pricing learners did **not** beat fixed-price controls (−11 % joint revenue): learning increased price
  variance without finding better prices within 40 days.
* In the public-goods game the seed where both learners kept contributing had the highest joint revenue; in the
  other seeds they free-rode. The unconditional cost-sharing contract did not change contributions — as theory
  predicts, since unconditional transfers net out and do not change marginal incentives. A contract conditioned
  on contribution (a deliverable predicate on ad spend) is the natural next experiment; the clause type exists.

## 8. Counterfactuals and self-play (`tests/test_tooling.py`, `crucible counterfactual`, `crucible selfplay run`)

Counterfactual (tiny world, seed 3, deterministic cost model; a Builder's service `alice-shorte` live for 1.25
days, then removed in one of two forks; a second control fork reached the identical digest, so the comparison is
exact):

| horizon | o:alice-labs revenue (control → removed) | alice customers | incumbent shorte1 customers | o:shorteco1 revenue |
|---|---|---|---|---|
| 12 h | $54.52 → $0 | 218 → 169 | 113 → 158 | +$2.00 |
| 24 h | $123.84 → $0 | 257 → 169 | 117 → 200 | +$19.28 |

Removing the incumbent `shorte1` instead reduces the requests served by a Composer (`svc:bob-mesh`) that resells
it — a dependency on another firm's artifact measured causally. Note: after an intervention the forks draw
different random numbers (draws are keyed by event sequence numbers), so small deltas in unrelated markets are
noise.

Self-play league (2 simulated days, both seatings, paired worlds): Builder(undercut 0.9, with ads) beat
Builder(undercut 0.6, no ads) 2–0; mean revenue $102.90 vs $36.15, customers 177 vs 84, market share 50 % vs 24 %;
Elo 1023 vs 977. See SELF_PLAY.md.

## 9. A general-purpose frozen-LLM agent with a broad objective (`examples/demo_llm_lifetime.py`)

Qwen3-4B (frozen, local, greedy), family `bazaar` seed 11, one decision per simulated hour for 5 simulated days
(120 decisions), endowment $30,000, objective given verbatim and nothing else: *"Establish and maintain a
sustainable source of value in this digital economy."* Raw records: `docs/experiments/demo_llm_lifetime.json`
(+ `.jsonl` trajectory with every action and result).

| measure | value |
|---|---|
| tool calls / waits / unparseable | 105 / 15 / 0 |
| accepted calls | 103 (1 `unknown_tool`, 1 `invalid_action`) |
| distinct tools used | 4 — `market.browse` ×96, `bank.balance` ×6, `bank.accounts` ×2, `market.listing_update` ×1 (a hallucinated tool named after an event type it saw) |
| organisations / repositories / deployments created | 0 / 0 / 0 |
| cash at end | unchanged ($30,000) |
| prompt / generated tokens | 433,938 / 3,253 (≈3.6k prompt tokens per decision); 292 s generation |
| world integrity | invariants empty; replay digest-equal |

The world worked; the agent did not. Unlike Qwen3-1.7B (RL_INTEGRATION.md: degenerate repetition of rejected
calls), the 4B model produced valid calls but stayed in passive information gathering — it never founded a
company, wrote code or offered anything. This is the expected starting point for experience-driven learning and
a concrete baseline for CAM-style controllers; it is not a statement about larger models or better prompting.
