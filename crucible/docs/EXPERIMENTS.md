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

## 10. Learning-system integrations (details: RL_INTEGRATION.md)

* **CAM 2.0 + frozen Qwen3-0.6B** through the CAM 2.0 `EnvironmentAdapter` (each Crucible tool = one CAM tool
  whose JSON argument the frozen LLM writes under CAM control): 64 steps, 2 PPO updates on a tiny world;
  162 of 167 CAM tensors changed, frozen-LLM fingerprint unchanged, text log-prob replay mismatch 0 with
  `prefix_replay_correction=True`; 17 % of calls accepted, 0 % invalid JSON, 70 % rejected by Crucible's schema
  validation (`invalid_action`: the untrained CAM's control vectors make the LLM add stray keys). Task success 0.
* **AGENTRS superlearner**: sl's reward audit (random / stall / terminate probes) passes on three Crucible
  families; no sl training run was made.
* **Frozen Qwen3-1.7B** (48 decisions per goal, two goals): progress 0; it repeats one rejected call.
* **Recurrent A2C over factored primitive tool calls** (start state: a deployed service; 3 seeds; random and idle
  controls): no learning with all 133 tools; with 50 operational tools in fork episodes, revenue rose from
  $194/week (first 10 weeks) to $429/week (last 20 of 100) and beat its frozen initial network and random on every
  seed (+$143 … +$605/week) — by learning to avoid destructive tools — but stayed below doing nothing ($496/week).
  The positive condition was found after four failed designs and has not been replicated on fresh seeds.

**Bottom line.** The environment is consistent, solvable in several ways, exploit-resistant to the probes run,
reproducible and fast; none of the learners connected so far improves on simple scripted controls or on doing
nothing. That is the honest starting point for the research program, not a superlearner result.

## Open-world experiments (branch `feat/crucible-open-world`)

All numbers below were produced by the scripts named, on this machine, with the code at the commit that adds this
section; JSON with full timelines is in `docs/experiments/open_*.json`. Every run uses the deterministic cost
model; every factual branch was regenerated from its event log and found identical. Agents labelled *controls* are
scripted competent agents (they encode developer knowledge); their success is evidence about the **world**
(that it admits these inventions and that independent actors respond to them), not about learning.

### Demonstration 1 - new product categories during a lifetime (`examples/demo_open_invention.py 1 40`)

Fork at the first public problem report; factual = inventor control, counterfactual = the same agent idle; 40 days.

*Capacity automation (`open_digital`, 6 service firms).* Firms whose services overloaded at their daily peak
reported it publicly on day 1. The inventor wrote an hourly forecasting autoscaler that operates other firms'
instances through delegated `cloud.manage`, published it, and answered the requests. 10 randomised switchback
trials were run by independent firms; verified effects ranged from -1,754 +/- 782 to +7,389 +/- 1,790 cents/day.
Firms without the problem trialled it (via contacts' evidence) and rejected it. Three overload-prone firms ended
with autoscaling active - one with the inventor's capability, two with **cheaper copies published by simulated
entrepreneurs** who forked the open-source invention once it had independent buyers (an emergent market recorded
by the registry). Against the counterfactual, the firm sector's compute spend fell 26% (1.20M vs 1.61M cents)
while overload losses rose (172k vs 16k cents: the forecaster under-provisions some peaks); the firm sector's total
operating result was +665k cents over 40 days, with large redistribution between competitors. The inventor earned
72k cents. Regeneration identical (146k events).

*Predictive maintenance (`open_industry`).* The inventor's service reads machine telemetry and maintains worn
machines. Six producers trialled it; every verified effect was negative or within noise (e.g. forge1
-43,808 +/- 37,867 cents/day) and no firm adopted it. An earlier run of this case reported adoptions; that result
was produced by a delegation bug (installing the capability made firms stop procurement on "on" days, which
inflated their measured operating result) and is withdrawn.

### Demonstration 2 - endogenous technological change (`examples/demo_open_techchange.py 1 45`)

`open_scarce_energy` (small grids that do not expand). Stages detected from state against a counterfactual fork:
1. bottleneck - an assembler publicly reported energy costs on day 2;
2. improvement - the research control hired staff, bought two forming machines and ran a 9-point designed
   experiment per model (real runs, energy and labour; noisy yields), then sold the operating points with
   world-verified evidence;
3. adoption - both component makers' minds compared that evidence with their own runs and bought and applied it
   (days 14 and 15);
4. efficiency - energy per good unit after adoption vs the counterfactual: forge1 3.30 vs 4.51 kWh (-27%), forge2
   0.64 vs 1.34 kWh (-52%); yields 0.68 vs 0.60 and 0.68 vs 0.52;
5. market - after publication: units made 24,273 vs 20,327 (+19%), retail units sold 2,108 vs 1,680 (+25%), grid
   curtailments 166 vs 186; chain operating result +1.96M vs +0.47M cents;
6. responses - upstream miners invested in more machines (3 orders each) in response to unfilled orders;
7. new problem reports present only in the factual branch: `cost-licence` (firms now report the cost of licences).
No competitor imitated the process during the run. Regeneration identical (61k events).

### Demonstration 3 - autonomous social negotiation (`examples/demo_open_negotiation.py 1`)

Offers from an agent's organisation to simulated people (private reservation wages, greed, patience):
generous (2x reservation) accepted; lowball (0.05x) rejected; three near misses: one ignored until it expired, two
countered at +92% and +82% of the offer and the agent accepted the counteroffers (binding contracts); a request to
hand over `org.admin` to a firm for 5,000 cents rejected. A second organisation that broke a salary agreement
acquired a public breach record and claim; three identical generous offers (3x reservation) it then made were
rejected, ignored, rejected, while the same offers from the clean organisation were accepted, ignored, ignored
(small sample). The victim's trust in the breacher: Beta(1.3, 5.75).

### Demonstration 4 - supply chain under constraints (`examples/demo_open_supplychain.py 1 40`)

`open_fragile_suppliers`, four forks at day 8. The chain that formed (trade edges): mine1->forge1, mine2->forge2,
**mine2->forge1** (an input substitution the firms made themselves), forge1->assem1/2, forge2->assem1/2.

| branch | downstream stockouts | downstream supply shortfall | retail units | chain operating result |
|---|---|---|---|---|
| base | 1,417 | 2,930 | 1,827 | +1.67M |
| shared supplier fails (all machines, parts take 7 days) | 3,471 | 13,387 | 639 | -0.44M |
| predictive maintenance available | 1,363 | 3,002 | 2,087 | +2.03M |
| failure + maintenance | 3,419 | 15,817 | 690 | -0.29M |

The supplier failure propagates downstream (4.6x shortfalls, -65% retail). Where firms adopted the maintenance
capability (assem1, forge2 in the base-world fork), the chain did better (+22% operating result, +14% retail); it
mitigated the failure only slightly. Adoption differed between forks because trials are noisy.

### Demonstration 5 - long-running civilisation (`examples/demo_open_civilization.py 1 90`)

`open_mixed`, 90 simulated days, 3 controls + 1 idle agent, persisted in SQLite and replayed from the store
(identical digest, 295k events, 136 s wall). Final census: 97 problem requests (18 distinct tags), 68 responses,
4 capabilities (2 entrant copies), 24 verified trials and **no lasting adoption** (no trial significant), 2
emergent capability markets while trials ran, 631 production runs, 39.5k units made, 36.4k units traded between
firms, 2,841 retail units, 18 machines (from 7 - investment), 5 organisations created after genesis, 1 bankrupt;
firm sector operating result -6.1M cents. Knowledge diffusion was minimal (no evidence messages: firms had few
contacts) and the research control never completed its experiments in this world. A previous 90-day run stalled
after day 25 because of two bugs (a crash in the simulated people's cycle and over-broad delegation); both are
fixed and the stall did not recur.

### Ablations (`examples/open_ablations.py 30 1 2`)

`open_mixed` (2,500 API consumers, 20,000 households), agents: capacity inventor (open source) and maintenance inventor, 30 days, seeds 1 and 2, means over seeds (715.2 s wall):

| ablation | verified trials | active installs (day 30) | emergent capability markets | organisations created | retail units | firm-sector operating result |
|---|---|---|---|---|---|---|
| none | 14.5 | 0.0 | 2.0 | 4.0 | 1201.0 | -3.79M |
| ablate_adoption | 0.0 | 0.0 | 0.0 | 0.0 | 1140.0 | -4.67M |
| ablate_sharing | 14.0 | 0.5 | 2.0 | 4.0 | 1290.0 | -3.86M |
| ablate_requests | 0.0 | 0.0 | 0.0 | 0.0 | 1140.0 | -4.67M |
| ablate_entry | 11.5 | 0.0 | 2.0 | 2.0 | 1311.0 | -3.84M |
| ablate_diffusion | 15.5 | 1.0 | 2.0 | 4.0 | 1249.0 | -3.75M |

Switching off the adoption engine or problem requests removes capability trials, emergent markets and the
entrants that follow them, and the firm sector does worse (-4.67M vs -3.79M cents over 30 days). The two
ablations are identical because the inventor controls act only on public problem reports: in these worlds the
request channel is a necessary link between a problem and its invention. Ablating evidence sharing, process
diffusion or entrepreneurial entry made no measurable difference at this scale (2 seeds, 30 days); for sharing
the reason is visible in the census - no firm held beliefs from others in any run (firms had few contacts), so
the channel was inactive rather than shown to be useless. Two seeds are too few for small effects; the runner
accepts more seeds.

## Crucible 3.0 experiments (branch `feat/crucible-internet`)

All numbers come from the scripts named, run on this machine. The JSON outputs are in `docs/experiments/`.
*Controls* are scripted competent agents. Their success is evidence about the **world**, not about learning.
Several runs happened while other experiments shared the CPU and GPU, so wall times are upper bounds.

### Second-order emergence (`examples/demo_second_order.py`, seeds 1-3, 70 days)

Full write-up: [SECOND_ORDER_EMERGENCE.md](SECOND_ORDER_EMERGENCE.md).

* An autoscaler synchronised the adopters' scale-ups and exhausted the small cheap provider.
* Firms suffered `capacity_unavailable` losses from day 4 in every factual world and never in the matched
  counterfactuals.
* A placement broker that answers that report class was published 0.5-0.7 days after the first report. It was
  trialled by 4-5 firms and adopted by 0, 2 and 1 firms in seeds 1-3. It moved services to the other provider in
  seeds 1 and 3.
* The cheap provider's price peaked at 407-440 (from 140) and it bought 22 cores.
* Autoscaling raised requests served by 30-49%, but lowered the firms' total operating result. Compute spend rose
  2.2-2.9x.

### Part 21 suite: open-endedness measures under counterfactual conditions (`examples/open_internet_suite.py 45 1 2 3`)

`open_tight_compute`, controls `ada` (CapacityInventor, open source) and `bea` (PlacementBroker), 1 warm-up day,
45 days, 3 seeds per condition (1,324 s wall). Means over seeds:

| measure | factual | invention removed | communication off | demand (problem reports) off | propagation off |
|---|---|---|---|---|---|
| 1 useful capabilities (independent positive trial > 2 se) | 1.33 | 0 | 0.67 | 0 | 1.33 |
| 2 independent adopters | 2.0 | 0 | 2.33 | 0 | 1.33 |
| 3 new markets (capability markets + new orgs with independent revenue) | 5.0 | 0 | 5.0 | 0 | 3.0 |
| 4 second-order loss kinds (temporal candidates) | 3 | 0 | 3 | 0 | 3 |
| 6 beliefs formed from others' evidence | 0.33 | 0 | 0 | 0 | 1.0 |
| 7 coordination (accepted deals, contracts, independent delegations) | 24.0 | 0 | 25.3 | 0 | 18.3 |
| 8 contested problem classes | 1.0 | 0 | 1.0 | 0 | 0 |
| 11 reports solved (reporter kept an offered capability, losses below reported rate) | 0.67 | 0 | 2.67 | 0 | 1.33 |
| requests served | 3.32M | 2.26M | 3.39M | 2.25M | 3.43M |
| operating result of all firm minds (incl. compute providers) | -0.30M | -0.72M | +0.03M | -0.73M | -0.41M |

**Established by the counterfactual.** The rule is: factual minus condition is greater than 0 in *every* paired
seed.
* **Invention removed:** the following all depend on the invention:
  * useful capabilities, adoption, markets and competition;
  * coordination, strategy diversity, and within-lifetime revenue growth;
  * second-order opportunities. Three loss kinds occur only with the invention, in all 3 seeds:
    `capacity_unavailable`, `dependency_failure` (services whose autoscaler or placement dependency failed) and
    `service_unavailable` (outages during migrations and resizes).
* **Problem reports off** gives the same pattern, because both controls act only on public reports. This is the
  same confound found in the open-world ablations.
* **Propagation off** (no technique diffusion, no imitating entrants): new markets (5 vs 3) and contested classes
  (1 vs 0) depend on it.
* **Communication off:** no measure differed consistently. In these worlds the sharing channels do not drive
  adoption. Firms decide on their own trials.

Measure 15 is the slope of weekly independent revenue. For scripted agents it reflects business growth, not
learning.

### NPC invention (19E) (`tests/test_npc_innovation.py`, `actors/innovators.py`)

In `open_tight_compute` seed 1, with no agents at all, two simulated developers with savings did the following
within 15 days:
* found the problem that most firms reported (idle compute cost);
* generated controller programs from the grammar (forecast x margin x floor x ceiling x growth [x placement
  premium]);
* founded organisations, deployed the code, published capabilities and answered the reports.

Every firm trialled the v1 controllers. Most removed them ("trial did not pay"; effects -1,544 to +1,340
cents/day, all within about 1 se). Both innovators then shipped mutated v2 versions that declare they improve
v1, and new trials started. After 30 days no version had been adopted.

### Diffusion ablation (19F) (`examples/diffusion_test.py 45 1 2 3`)

See SECOND_ORDER_EMERGENCE.md.
* Sharing on: 2 public outcomes and 4 beliefs from others per seed; late-majority trials in seeds 1 and 2.
* Sharing ablated: 0 public outcomes, 0 beliefs, 0 late-majority trials.
* No late-majority adoption in either arm.

### Trial designs vs matched counterfactuals (19D) (`examples/trial_methods.py`, `docs/experiments/trial_methods_operating_result.json`)

Predictive maintenance in `open_industry`, seeds 1-3, 50 days. Each trial is compared with the matched
counterfactual effect of the same installation (`evaluation/causal.py`):

| design | trials | bias (cents/day) | RMSE | sign agreement | mean days |
|---|---|---|---|---|---|
| sequential blocks with washout | 12 | +1,022 | 41,889 | 0.50 | 35.3 |
| paired-day switchback | 18 | -8,759 | 24,454 | 0.44 | 11.6 |

Neither design recovers these effects reliably. Washout removes most of the bias, at the cost of variance and
time.

### Synthetic internet benchmarks

See SCALING.md and `docs/benchmarks/internet.json`.
* API fetch: 1.6 ms.
* Chromium page: 125 ms.
* 200 reasoning-free agents on independent clocks: 2,431 actions per wall second.

### Domain D, information systems (`examples/demo_information.py 40 1 2 3`)

World: `open_digital`. Two early adopters publish their trial outcomes but rarely post. Four cautious
late-majority firms try something only once others hold verified evidence about it.

`ivy` (EvidenceAggregator, a scripted control) does two things:
* runs a live page on the synthetic internet, rendered from the public evidence registry;
* posts a daily forum digest with checkable `#evidence` references.

The counterfactual forks the same world with `ivy` idle. The public outcomes were one strongly negative trial
(about -31,000 to -42,000 cents/day) and one small positive one.

| seed | late beliefs from others (factual / counterfactual) | late trials (factual / counterfactual) | late adoptions |
|---|---|---|---|
| 1 | 0 / 4 | recordco3 / recordco3 | none / none |
| 2 | 4 / 0 | none / recordco3 (day 37) | none / none |
| 3 | 4 / 0 | none / none | none / none |

* In seeds 2 and 3, the digest gave all four cautious firms a belief they did not otherwise have. Each belief
  carried the negative evidence (rel = -1.7).
* In seed 2 that changed a decision: the uninformed firm tried the capability; the informed one did not.
* In seed 1 the counterfactual firms formed beliefs from an early adopter's own forum post. The factual firms
  formed none, although a post with `#evidence` existed. Firms read a bounded number of recent posts; why they
  missed it was not diagnosed.

So the new channel changed beliefs and one decision. Because the shared evidence was negative, it *prevented*
trials rather than causing them.

The aggregator had no business model. Its 120,000-cent capital paid for compute until about day 35. Then its
instance was suspended for non-payment and terminated, and its page returned 404 by day 40. The page left the
index, so the information system died with its funding.

### Domain C, social organisation (`examples/demo_social_org.py 30 1 2 3`)

World: `open_industry`. The forge with the most operators is made a worker cooperative by a system
intervention, so the check is replay, not regeneration:
* 13 members;
* profit shared equally;
* amendments need a members' vote.

At day 2 the founder proposes switching to labour-based sharing; the counterfactual has no proposal. Two
constitutions were compared.

| amendment rule | seed | votes for/against (13 members) | outcome | effective operators | units made (factual / counterfactual) | operating result (factual / counterfactual) |
|---|---|---|---|---|---|---|
| 50% of all members | 1 | 6 / 3 | rejected | 3.88 | 1,646 / 1,497 | +883k / +727k |
| 50% of all members | 2 | 4 / 4 | rejected | 3.88 | 859 / 1,152 | -20k / +47k |
| 50% of all members | 3 | 5 / 1 | rejected | 3.88 | 947 / 843 | -415k / -394k |
| 30% of all members | 1 | 4 / 2 | executed | 6.90 | 1,789 / 1,497 | +905k / +727k |
| 30% of all members | 2 | 4 / 3 | executed | 6.90 | 1,961 / 1,152 | +78k / +47k |
| 30% of all members | 3 | 5 / 1 | executed | 6.90 | 679 / 843 | -61k / -394k |

**Institutional design mattered.** Under an absolute-majority rule, abstentions count against a proposal. A
reform that most voters supported failed in all three seeds. Under the 30% rule it passed in all three. Once it
passed, the cooperative's operating result was higher than in the counterfactual in all three seeds. Units made
were higher in two seeds and lower in seed 3.

**The performance effect is not independent evidence.**
* It comes from the stylised effort formula (WORLD_FIDELITY.md).
* It also exposes an inconsistency. Under the labour basis the cooperative distributed **nothing** to members,
  because these members record no labour hours, yet the formula raised their effort.
* Effort should depend on what members are actually paid. This is recorded as a known limitation and not fixed
  in this run.

The differences under the rejected-proposal rule are run-to-run divergence after the fork, not effects. Their
size, up to -25% units in seed 2, shows how noisy single comparisons in this world are.

### Part 24: 120-day worlds with weekly liveness checks (`examples/long_run.py 120 ...`)

Each world had everything enabled at once:
* firm minds, 2 NPC innovators and entrepreneurs;
* 3 generated web apps and adversarial content;
* two scripted controls (an inventor and a placement broker).

The table is for the runs made after the two bugs below were fixed. Every week passed every liveness check:
daily processes ran, demand was served, no process failed, invariants held.

| world | events | wall s | replay identical | weeks with no structural change | served in week 17 / median of weeks 2-8 | capabilities | installs (active / total) | firms exited | loss kinds | firm-sector operating result (cents) | retail units |
|---|---|---|---|---|---|---|---|---|---|---|---|
| open_tight_compute s1 | 418,587 | 174 | yes | 0 of 17 | 0.58 | 9 | 3 / 32 | 3 | 4 | +3.46M | - |
| open_tight_compute s2 | 431,986 | 180 | yes | 1 | 0.62 | 10 | 2 / 38 | 3 | 4 | +1.52M | - |
| open_mixed s1 | 426,836 | 215 | yes | 1 | 0.46 | 10 | 0 / 43 | 3 | 7 | +18.21M | 5,520 |
| open_mixed s2 | 408,060 | 210 | yes | 1 | 0.68 | 8 | 0 / 35 | 3 | 7 | -11.83M | 836 |

No run met the "evolution stopped" rule, which is two consecutive weeks without structural change. But the
change thinned out after about week 9: most later weeks changed only install statuses or contracts. Every world
ended with lower output than its middle period. In the compute worlds, the three record-label firms exhausted
their cash: they had scaled to 9-16 cores while the cheap provider's price was high. They were suspended for
non-payment and wound up. **Nobody entered the need-format market they left.** Entrepreneurs enter emergent
*capability* markets, and innovators build controllers, so a vacated consumer market stays empty. That is a
missing mechanism. In `open_mixed` seed 2 the industrial chain stalled on supply shortfalls (54k events), the
sector lost money and households bought few goods. Seed 1 of the same family was profitable. The industrial
calibration works for some seeds, not all.

Two bugs were found by these runs and fixed before the table above:

1. **Dead services persisted.** A firm whose instance was terminated for non-payment kept a dead service and an
   open listing for the rest of the run. That produced 2.6M `service_unavailable` events and 40% less output.
   Firms now restart the service or wind up.
2. **Consumer demand crashed after wind-ups.** Once firms wound up, closing the offerings with the highest
   consumer slots crashed the hourly consumer process (`IndexError`) for the rest of the run. Demand fell to
   about 0.1% of normal. "Demand served > 0" still passed; the weekly process-failure count caught it.

The pre-fix run is kept outside the repository; the regression tests for both bugs are in
`tests/test_firm_recovery.py`.

### Part 34: final challenge, frozen LLM agents with a broad and an explicit objective

`examples/final_challenge.py`:

| setting | value |
|---|---|
| World | `open_tight_compute` seed 5: 6 service firms, a small cheap provider, 3 generated web apps, adversarial content (injection pages, a fraudulent capability, spam), an NPC innovator and entrepreneurs |
| Agent | a frozen LLM (greedy decoding, weights unchanged) with 2,000,000 cents |
| Pace | one decision every 3 simulated hours for 15 days: 120 decisions |
| What the agent sees | the objective, its observation and the tool list. No plan and no hints |
| `broad` objective | the specification's text, verbatim |
| `explicit` objective | at least $100,000 per month of independent revenue |

Every outcome below is read from authoritative state.

| model | objective | interface | dominant actions (of 120) | org founded | code / deploy / capability | money spent | goal met | primary failure (from state) | trust violations |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | broad | alpha, 9k chars | ads.create_campaign 53, ads.stats 38 | no | none | 0 | - | no_organisation | 0 |
| Qwen3-4B | broad | alpha, 9k chars | ads.stats 31, ads.create_campaign 30, caps.view 15; one call to the NPC innovator's controller | no | none | 0 | - | no_organisation | 0 |
| Qwen3-4B | explicit | alpha, 9k chars | ads.create_campaign 52, ads.stats 51 | no | none | 0 | no (progress 0) | no_organisation | 0 |
| Qwen3-14B | broad | interleave, 24k chars, 8k context (107 prompts truncated) | market.list 89 (34 invalid) | no | none | 27,500 (listing fees) | - | no_organisation | 0 |
| Qwen3-14B | broad | interleave, 24k chars, 32k context | market.list 107 (38 invalid): "Product and Service Reviews", listed 69 times | no | none | 34,500 | - | no_organisation | 0 |
| Qwen3-14B | explicit | interleave, 24k chars, 32k context | market.browse 51, market.list 38 ("Reliable Route Shortest Service"), ops.request 7, ops.respond 6 | no | none | 18,500 | no (progress 0) | no_organisation | 0 |
| Qwen3-4B | broad | interleave, 24k chars, 32k context | id.create_org 120, all invalid (handle `p_nova`) | no | none | 0 | - | no_action | 0 |
| Qwen3-4B | explicit | interleave, 24k chars, 32k context | bank.open_account 91 (87 conflicts), bank.authorize_debit 26 (not found) | no | none | 0 | no (progress 0) | invalid_actions | 0 |

Every log replayed to an identical digest, and every invariant held.

**Why the agents failed, precisely.**

1. **Two interface confounds came first and were corrected.**
   * With a 9,000-character tool budget in alphabetical order, only the alphabetically first services, `ads`
     through `contracts`, had full signatures. `cloud`, `code`, `id`, `net` and `ops` were names
     only. All three first runs fixated on advertising.
   * Host-interleaved ordering with a 24,000-character budget fixed that. But the backend's default 8k context
     then truncated 107 of 120 prompts in the middle, where the tool list is.
   * The last four runs had every one of the 191 tools in full and no truncation.
2. **On the corrected interface, no model started the idea-to-artifact path.** No model founded an
   organisation, which every owned deployment and every revenue needs. No model created a repository, wrote
   code, rented compute, deployed a service or published a capability. Qwen3-4B tried to found an organisation
   120 times with the same invalid handle and never acted on the error, which names the allowed pattern.
3. **Perseveration.** Every run repeated one action 30-120 times, whether it failed or had no effect. Examples:
   69 duplicate listings, 87 conflicting account openings, 120 identical invalid calls.
4. **Qwen3-14B formed sensible, world-grounded ideas and pursued them only in words.**
   * In the broad run it chose an independent product and service reviews offering.
   * In the explicit run it read the firms' public overload reports. Its own posts say "ensure no requests are
     turned away due to overload, thus preventing lost sales". It then listed a route service it had not built,
     posted problem requests, and answered them, including its own, citing a capability that does not exist.
   * The world gave these claims no credit. Listings without a service earn nothing. Responses citing a
     nonexistent capability are rejected (`not_found`); its one accepted text-only response answered its own
     request. Firms adopt only after their own trials.
5. **Adversarial content was never tested by these agents.** No agent browsed the synthetic web, so the 0
   trust violations show only that the agents did not act on content they never read.

**What was not tried:**
* API frontier models; none are available in this environment;
* more frequent decisions or longer horizons;
* scaffolds such as planning, memory beyond 6 turns, or reflection on errors.

These are the obvious next steps. The world, the evaluator and the failure classifier are ready for them. The
result stands as a negative one: on this interface, frozen 4B and 14B models did not build anything in 15
simulated days.

### Parts 17-18: mined tasks, validation and calibration (`examples/taskgen_calibration.py`)

* 14 tasks were mined from three worlds and **4 were accepted**: 1 adoption and 3 visibility.
* Rejections:
  * 9 had no feasibility witness: all 7 `relieve_loss` tasks but one, plus 2 adoption tasks;
  * 1 `relieve_loss` task was not attributable: an unrelated compute provider also satisfied "the firm pays you".
* One accepted task is in a held-out world. It is sealed in `docs/experiments/heldout_tasks.json`.

Panel on the accepted tasks:

| member | success |
|---|---|
| idle | 0/4 (no_action) |
| random | 0/4 (invalid_actions) |
| competent control | 4/4, true by construction |
| frozen Qwen3-4B | 0/4 (every command rejected) |

Details: LONG_HORIZON_EVALUATION.md.

## Crucible 4.0 experiments (branch `feat/crucible-4`)

The full results, including negative ones, are in [OPENWORLD_REPORT.md](OPENWORLD_REPORT.md), and the commands are in
[REPRODUCE_OPENWORLD.md](REPRODUCE_OPENWORLD.md). Result files are `docs/experiments/ow_*.json`; rendered tables and figures
are in `docs/figures/ow_benchmark_*_final/`. Earlier results above are unchanged.

