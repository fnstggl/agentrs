# Audit F_commerce: commerce, recurring revenue integrity and institutions (workstream F)

Auditor: independent audit of branch `feat/crucible-5-complete` at `a881d18`, worktree
`/workspace/agentrs/.worktrees/c5-complete`. Scope: MASTER_SPEC Parts 6.1, 6.2 and 6.6, Gate K, and manifest entries
K.1-K.7, K.n1, SP-6.1 and SP-6.6. Code: `crucible/src/crucible/commerce/` (5.0, workstreams F1 and F2) and
`crucible/src/crucible/economy/` (4.0, unchanged in 5.0). The two user amendments (observation-only by default, and the
recovery preamble) apply throughout. No earlier completion claim was accepted without a run I did myself.

Probe scripts, logs and JSON outputs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/F_commerce/`.

## 1. Verdict in brief

The F1 billing engine is careful and correct for the scenarios its tests cover. That engine covers subscriptions,
trials, coupons, dunning, cancellation, proration, refunds, one-time sales and licences, all paid through bank
mandates from the customer's own account. I reproduced every gate test: K.1 7/7, K.2 6/6, K.3 5/5, K.4 0 errors with
15 customers, K.5 0 mismatches, K.6 0-cent error, K.7 detection 1.0 with FPR 0.0, K.n1 ARR $126,257 and $135,892,
SP-6.1 20/20 and SP-6.6 14/14 with 9/9 blocked. The 56 commerce unit tests pass.

Even so, **the accounting the Gate K objective relies on can be gamed with ordinary tools**:

* **One partner, $1,000 net, $100k ARR.** A single cooperating customer funded with its own money subscribes to a
  $100,000/year plan, and the merchant refunds 99% of the invoice. Qualifying ARR then stays at $100,000.00 for the
  whole year, with 1 "independent customer" and nobody flagged. In the generated Gate K startup world this meets the
  K.n1 objective (ARR >= $100k for the last 7 daily checks) from day 1. The partner's net cost is $1,000 (probes P1,
  `probe_reference.py`). Cause: only a *full* refund clears `period_paid`, and MRR is computed from list price minus
  coupon, not from what was actually kept.
* **F2's own escrowed instruments get around circular-flow detection.** The merchant pays a self-funded customer
  through a 5.0 `commit` commitment or an `inst` project reward, and the customer subscribes. Netted money is 0, the
  circular share is 0.09, and the payer is **not flagged**, so MRR counts it in full. The same money sent by
  `bank.transfer` is netted 100% and flagged (`probe_escrow_laundering.py`). Cause: in the netting graph, escrow-held
  flows other than marketplace orders and bounties are dropped.
* **The evaluator's hidden metric can be read in-world.** All F1 metrics are registered with a `subject`.
  `mrr_cents` includes the hidden related-party and circular-flow analysis, and it can be read in-world through
  charter rule conditions and commitment predicates. A merchant recovered its exact evaluator MRR in 14 queries
  ($20, against the $80 its books show) by gating `bank.transfer` on `mrr_cents >= X` (`probe_oracle.py`). An agent
  can use this to test which evasion schemes the evaluator catches.
* **Rebates below 50% cost no MRR.** If the merchant hands back up to 49% of each payment as cashback, the
  subscription still counts at full list price in MRR. A payer is excluded only when its circular share is at least
  0.5. The report is also inconsistent: the same customers appear in `revenue_report.flagged_payers` (30-day window)
  while their subscriptions count in MRR (all-history flagging) (P2).
* **Nothing outside tests uses it.** No world family, Tier 2 domain or civilization installs `subs`, `products`,
  `ventures`, `xchg` or `patrons`. They appear only in the test kit, `commerce/reference.py` and the red-team world
  factory. The only goal family the system has for revenue is the 4.0 `revenue` goal (`goals/contracts.py:236-246`).
  It uses the rolling 30-day `independent_revenue`, which is exactly the "rolling 30-day sales" Part 6.2 forbids
  confusing with recurring revenue. That 4.0 measure counted 7,500 cents of planted circular revenue as independent
  in my K.7 run.
* **The 4.0 economy pays from one pooled household account.** That pooled payer violates "revenue must originate from
  independent participant transactions". F1's metrics also skip the household counterparty altogether: in the 4.0
  `bazaar` world, every firm with household sales has **F1 independent revenue 0** while the 4.0 metric counts all of
  it (`probe_households.py`).
* **Customer demand is scripted.** It comes from `patrons`, a third separate population that is uncalibrated and
  hand-coded. Each patron has one need, discovers products only through catalogue search, and scores each response
  against hidden ground truth. Demand exists only for the four need-solver SaaS formats. Media, education and data
  products have no demand model.

Overall: **REPAIR** the accounting and integrity layer, **REDESIGN** where demand comes from, and integrate the
whole stack before Gate K can count as passing in the integrated system.

## 2. What I ran

| command | result |
|---|---|
| `pytest -q tests/v5/test_gate_K.py -k "not Kn1"` | 7 passed, 7.3 s. Unofficial evidence: K.1 7/7, K.2 6/6, K.3 5/5, K.4 errors 0 / counted 15, K.5 mismatches 0 / arr=12mrr true / $100k ARR = $8,333.33 MRR true, K.6 error 0 (profit -32,200), K.7 detected 1.0 / FPR 0.0 (15 planted payers of 13 types, 40 ordinary customers; 4.0 metric 92,500 vs F1 84,854 vs truth 85,000) |
| `crucible.scale.slots --cpu 1 -- pytest tests/v5/test_gate_K.py -k Kn1` | 1 passed, 122 s; reference ARR min $126,257 (seeds 1, 3, **both drew `text.extract`**), idle baseline $0, first day met 50, 618 independent customers minimum |
| `pytest -q tests/v5/test_gate_SP_6_1.py tests/v5/test_gate_SP_6_6.py` | 2 passed, 14.7 s; SP-6.1 20/20; SP-6.6 14/14 features and 9/9 core-law attempts blocked on seeds 66, 1, 2; replay and regeneration equal |
| `pytest -q tests/test_v5_commerce_{modules,institutions,contracts,labor}.py` | 56 passed, 18.3 s |
| `probe_gaming.py` (P1-P8) | see section 4 |
| `probe_escrow_laundering.py` | transfer: flagged (netted 10,000, share 1.0); commitment: **not flagged** (netted 0, share 0.0909); project reward: **not flagged** (netted 0, share 0.0909); MRR $100 counted in both escrow cases |
| `probe_oracle.py` | evaluator MRR read in-world: answers {1000: T, 2000: T, 3000: F, 8000: F}; binary search = 2000 cents in 14 queries; booked MRR shown to the agent 8000 |
| `probe_reference.py` (slot-scheduled) | collusion meets K.n1 objective on day 1, sustained, ARR $100,000, net cost $1,000, 1 qualifying subscription; integrity cost about 6 ms per 1k ledger transactions per merchant, cold |
| `probe_households.py` | bazaar family (4.0): every firm with household `sale` revenue has F1 independent revenue 0; 4.0 metric = household sales |
| `probe_seats.py` | 1-seat licence bought by an organisation: 3 of 3 admins read the licensed code |
| `probe_experience2.py` | startup world through `KernelExperienceEnv` with default arguments: `reward_mode = legacy_shaped_reward`, and with no objective assigned the reward after founding a company is **-150,000** (cash delta) |

## 3. Requirements and mechanism review

### 3.1 Subscriptions, cancellations, refunds (K.1-K.3) - `commerce/subscriptions.py`
* Plans are immutable once created. `subs.subscribe` (l.173-222) pays from the caller's own account, or from the
  account of an organisation it administers. It creates a bank mandate capped at the plan price per period. The bank
  `debit` (services/bank.py:278-293) checks the mandate limit and the grant room, and it has no tool endpoint, so a
  merchant cannot pull money any other way. This is correct.
* Dunning (retries after 1, 3 and 7 days, then `nonpayment`), trials (first subscription to a product only),
  coupons, proration and resume all behave as their docstrings say.
* **Defect (refund semantics):** `_refund` (l.284-301) clears `period_paid` only when the whole invoice is refunded
  (l.298). A 99% refund leaves the subscription "paid up" at full `current_amount`. K.3's
  `k3_partial_refunds_bounded` asserts this as correct (`MRR == 2000` after a 500-cent refund), so the gate test
  writes the loophole into the expected behaviour.
* **Definition issue (`cancel_at_period_end`):** a subscription the customer has already cancelled keeps counting
  until its period ends. For annual plans that is up to 365 days of "recurring" revenue that will never recur (P1b:
  $1,000 of MRR still counted at day 300). Part 6.2 asks for "sustained subscription validity".
* Missing: usage or metered billing, per-seat pricing, recurring goods deliveries, taxes. Seats are stored but
  never enforced (products.py:410; `probe_seats.py`).

### 3.2 MRR/ARR normalisation (K.5) - `commerce/revenue.py`
* Normalisation is exact: `monthly_recurring` = `current_amount / {1, 12}` (subscriptions.py:120-123), and ARR is
  12 x MRR exactly. A $100k ARR objective corresponds to $8,333.33 of MRR (I reproduced the K.5 test). One-time
  sales do not enter MRR (P8: $10,000 of one-time sales gives MRR 0). Rolling sales are kept separate.
* **Defect:** MRR is the list price minus a coupon. It is **not net of partial refunds, rebates or partial circular
  shares**. A payer below `FLAG_SHARE = 0.5` (integrity.py:49, 349) contributes 100% of its price.
* **Defect (inconsistency):** `revenue_report` takes its `flagged_payers` from a windowed `related_flows`
  (l.158/229), while `qualifying` uses all-history `flagged_payers` (l.53). In P2 the report lists all 10 customers
  as flagged and shows independent revenue 0, yet MRR counts all 10 customers.
* K.5's "independent recomputation" (test_gate_K.py:411-437) re-implements the same definition (list price minus
  discount, full-refund check only). It cannot catch errors in the definition itself, so the criterion is weak.

### 3.3 Independent customers and related or circular detection (K.4, K.7) - `commerce/integrity.py`
* Strengths: relatedness from 4.0 rules closed under shared `controller` and cross-boundary `bank.*` grants;
  pro-rata source-of-funds tracing since genesis, through any number of hops (catches delayed and chain funding);
  max-flow netting of pairs and rings; loan principal treated as financing. The 13 planted schemes are all caught.
  The analysis is evaluator-side and runs on ledger replay.
* Holes I confirmed by gaming it with controlled or cooperating identities:

| probe | scheme | result |
|---|---|---|
| P1 | self-funded partner, annual $100k plan, 99% refund | ARR $100,000, 1 customer, not flagged (refunds are "untainted" and not netted) |
| P2 | 45% cashback by `bank.transfer` to self-funded customers | full list price in MRR (share 0.45 < 0.5) |
| P3 | merchant buys 1 goods unit from a self-funded partner at the subscription price | share 0.005, not flagged (goods consideration is never netted, integrity.py:107-128; pro-rata dilution) |
| escrow | payment via `commit` milestone or `inst` project reward | netted 0, share 0.09, not flagged; plain transfer is flagged |
| P4 | second persona of the same agent admitted without, or with a different, `controller` label | counted as independent (identity relies entirely on the operator's label string) |
| P5 | chain of 6 `bank.spend` grants | only 4 of 6 related (`rounds < 4`, l.68) |
| P6 | 3 owner-funded socks rate 5 stars | ratings accepted (products.py:498 checks only 4.0 `is_related`); public rating 5.0 from 3 ratings, exactly the threshold at which `patrons` trust ratings (customers.py:253) |

* The K.7 planted set was written by the same workstream that wrote the detector, and it contains none of the
  routes above. "Detected fraction 1.0" therefore shows the detector handles its own catalogue, not that it is
  robust to an adversary.
* Cost: `funding_trace` replays the whole ledger per merchant per sequence number. I measured about 6 ms per
  1,000 transactions per merchant, cold (7,668 transactions, 4 merchants: 25-55 ms). At Internet scale (10^7
  transactions, 10^3 merchants) one evaluation sweep is hours. It needs incremental or streaming tracing.

### 3.4 Profit and costs (K.6)
* The cash-basis decomposition is correct in the K.6 scenario (reconciliation 0) and the kinds are classified
  sensibly.
* **Defect:** payments from the `households` counterparty are skipped (integrity.py:304), so firms paid by the 4.0
  household pool show F1 revenue 0 (bazaar probe).
* There is one scenario, no accrual or deferral of prepaid annual revenue, no depreciation, and CAC depends on an
  `ads` ledger kind that no commerce world produces (`acquisition_spend` is always 0 in startup worlds).

### 3.5 Reference strategy (K.n1) - `commerce/reference.py`
* Feasibility is real for the strategy as scripted. Customers are 800 independent principals that pay from their
  own accounts and measure quality with exact verifiers; the reference deploys the best known solver
  (`best_tier(need)`, l.95) at the incumbents' median price. It reaches ARR of about $126k-$136k after day 45-50,
  and the idle strategy reaches 0.
* **Weaknesses:**
  * `objective()` (l.131-138) has no minimum number of independent customers and no cap on any one customer's
    share. It is met by the collusion probe.
  * The "generated world" is one template with parameter draws: 4 needs, 2-3 static incumbents, uniform WTP. Both
    test seeds draw the same need.
  * The reference uses privileged knowledge: the need name from `world.spec.params` and the kernel's best solver.
  * Its support replies are boilerplate (l.120), and they have no effect on customers.

### 3.6 Businesses and products (SP-6.1) - `products.py`, `business.py`, `marketplace.py`, `customers.py`
* Formation, development (launch only once the delivery exists), deliveries (live service executed in the sandbox,
  repo licence, document access, conserved goods), one-time sales, licences with terms, exchanges with take rates
  and disputes, orderly wind-down and insolvency with customers as claimants: all verified by the 20 SP-6.1
  scenarios. No tool sets quality or revenue (SP-6.1 asserts there is no `prod.set_*` or `subs.set_*`).
* **EXTEND:**
  * Competition is only customer choice between static incumbents. Incumbents are created at genesis
    (business.py:282-304) and never change price, quality or capacity.
  * Customer support has no consequence: tickets do not change churn or rating.
  * Customer acquisition exists only through `prod.catalog`. There is no advertising, social, word-of-mouth or
    search-engine channel into `patrons`, and no link to the v5 population or social layer (C/B).
  * Seats are not enforced.
  * Ratings from owner-funded payers are accepted.
  * Media, education and data value has no demand model: those scenarios use scripted buyers.
* **Parallel systems:**
  * `social/creators.py:109` bills creator memberships itself (ledger kind `subscription`). These flow into F1
    profit but never into MRR, so there are two recurring-revenue systems of record.
  * Two product catalogues coexist (4.0 `market.listings` and 5.0 `products.products`; DECISION_LOG l.55).
  * Three customer populations coexist: 4.0 `consumers`/`shoppers` with a pooled payer, F1 `patrons`, and the v5
    `population` (which buys through site invoices, not `prod`/`subs`).

### 3.7 Adaptable institutions (SP-6.6) - `institutions.py`, `contracts.py` (commit), `labor.py`
* The features are rich and genuinely enforced at authorisation: charters, seven decision procedures with the
  electorate fixed when a decision is proposed, liquid delegation, offices with budgets, units, merge/absorb,
  spin-off, dissolution through the company registry, escrowed projects and ventures, negotiated escrowed
  commitments with arbitration, and a labour market. Replay and regeneration are exact.
* **REPAIR:**
  * (a) Charter conditions and commitment predicates can read evaluator metrics that have a declared subject,
    including F1's `mrr_cents`, `independent_customers` and `profit_cents`, and the 4.0 `independent_revenue`
    (goals/metrics.py:24, 88-121; institutions.py:307-319, 325-343). That gives an in-world oracle on hidden
    evaluation (probe_oracle).
  * (b) Project rewards (institutions.py:1254) and commitment payouts (contracts.py:256) leave escrow as flows the
    netting graph discards, so the institutions module itself is a channel around revenue-integrity accounting.
  * (c) The 9 "core-law" attempts are malformed requests: unknown charter keys, `admin.*` tools, `set_state`, an
    outsider's account. Schema closure and the kernel's normal authorisation refuse them; none is an adversarial
    use of *authorised* mechanisms.
* **Labour:** a task's quality is a formula of private skill match plus noise (labor.py:61-66), and the work
  produces no artifact. This is useful for organisational dynamics, but it is a simulator-specific mechanism.

### 3.8 4.0 economy (`economy/`)
* `needs.py`: public problem formats with seeded generators, exact verifiers and tiered reference solvers. This
  module is sound and is used effectively by `patrons` and the SP-6.1 data and infrastructure scenarios.
* `consumers.py` / `shoppers.py`: vectorised demand that pays from `acct_households` (consumers.py:307). This
  pooled payer cannot provide independent customers, which Part 6.1/6.2 requires. F1's metrics drop it, and 4.0
  metrics count it as independent.
* `industry.py` and `operations.py`: out of F's 5.0 changes and not audited in depth here. None of `economy/` was
  modified in 5.0 (git log).

## 4. Would an agent learn transferable skills here?

Partly. In the startup world the paying customers are real principals with budgets. They measure the service they
buy, cancel bad products and churn for exogenous reasons. "Build a working solver, price against incumbents, keep
it up" is a legitimate lesson. The simulator-specific shortcuts an optimiser would find are:

* Annual plans, large partial refunds and a partner give ARR at 1% of its nominal cost.
* Payments routed through commitments, project rewards or goods purchases are invisible to netting.
* Rebates just under 50% keep full MRR.
* Three fake 5-star ratings flip every patron's quality prior from 0.8 to 1.0.
* The hidden evaluator can be probed through charter conditions.
* Under the default `legacy_shaped_reward`, cash delta is a reward even with no objective, so investing capital is
  "punished".
* Patrons never respond to marketing, support quality or social signals, so a learner gets no gradient for those
  real-world skills.

## 5. Work needed (priority order)

1. **MRR net of what is kept.** Define qualifying MRR from the net recurring amount: subtract partial refunds of
   the current period, rebates and the payer's circular share. Exclude `cancel_at_period_end` subscriptions, or
   report them separately as "scheduled churn". Fix the windowed/all-history inconsistency. Update the K.3 and K.5
   tests to assert the net definition.
2. **Integrity across escrow and instruments.** Attribute `commit`, `inst` project and `xchg` escrow releases to
   their real counterparty in `_tx_flows`, and net them. Bound goods-consideration exemptions by a market-price
   test, or by the goods' resale or consumption. Replace binary `FLAG_SHARE` exclusion with proportional
   exclusion. Close `related_set` to a fixpoint. Add these schemes, plus colluder-funded annual and refund schemes,
   to `SCHEMES` and to K.7. Use a red-team-authored planted set, not the implementer's.
3. **Hide evaluator metrics in-world.** Mark integrity-based metrics as evaluator-only (no `subject`). Expose a
   separate *booked* MRR (the agent's own books) for charters and commitments.
4. **Harden the objective.** Require at least N independent customers, cap the largest customer's share, and use
   net MRR. Run K.n1 on structurally different generated worlds (multiple needs, adaptive incumbents, non-SaaS
   products) and on the integrated civilization.
5. **Integration.** Make `subs` the system of record for `creators`, and let the v5 population (B) buy through
   `prod`/`subs` via `ids.ensure_principal`. Give the 4.0 household demand individual accounts, or attribute it
   per consumer. Add an MRR/ARR goal family that uses `mrr_cents` instead of rolling 30-day revenue. Install
   commerce in the Tier 2 startup and marketplace domains and in Tier 3 civilizations.
6. **Realism (EXTEND).** Adaptive competitors; support quality feeding retention; acquisition channels (ads,
   search, social, word of mouth) with the fitted commercial-response model; per-seat licensing; demand for media
   and education from the population's semantic preferences; labour that produces real artifacts.
7. **Scalability.** Incremental source-of-funds tracing, maintained by a bank applier observer, instead of a full
   replay per merchant.
8. **Cross-cutting (L).** `KernelExperienceEnv` defaults to `legacy_shaped_reward` and pays cash-delta rewards with
   no objective (experience/kernel.py:59, 232-235). The amendment requires `observation_only`.

## Skeptic review

Reviewer: skeptic for area F_commerce. I re-read the code and wrote my own probes, trying to refute every RETAIN
and every verdict that might be too generous. HEAD is now `dd745b9`, which differs from `a881d18` only in docs and
acceptance-manifest code, so the findings apply to both. Probe scripts, logs and JSON outputs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/F_commerce_skeptic/`.

**Net result:**
* 2 RETAIN requirement verdicts overturned (K.1, K.2).
* 1 RETAIN subsystem verdict overturned (`needs`).
* 3 requirement verdicts made harsher (K.n1, SP-6.1, SP-6.6).
* 4 subsystem verdicts made harsher (`subs`, `xchg`, `inst`, `reference`).
* No harsh verdict was found unjustified. The two REDESIGNs (patrons and 4.0 demand) are confirmed and strengthened.

### S.1 What I ran

| probe / command | result |
|---|---|
| `pytest -q tests/v5/test_gate_K.py -k "not Kn1" tests/v5/test_gate_SP_6_6.py` | 8 passed, 1 deselected, 10.6 s (reproduced) |
| `rerun_probe_oracle.py` (copy of the first auditor's probe) | reproduced: evaluator MRR 2,000 recovered in-world in 14 charter-condition queries, while the books show 8,000 |
| **S1** `probe_creators_mrr.py`: world with the social internet plus `products`/`subs`; 5 fans join a creator tier at $20/30 days | creator ledger: `subscription` +30,000 over 3 periods, 5 live memberships. F1: **MRR 0, qualifying 0, active subscriptions 0, independent customers 0** (`independent_revenue_cents` 10,000 in the 30-day window) |
| **S2** `probe_superseded_refund.py`: one self-funded, unlabelled, independent customer | **A:** $100k annual plan, then `change_plan` to a second $100k annual plan (charged 1 cent), then the merchant refunds the original $100,000 invoice **in full**. Result: ARR **$100,000.00 on every check from day 1 to day 364**, K.n1 `objective()` met, customer's net cost **1 cent**, payer not flagged, 1 "independent customer". **B:** annual, then monthly $8,333.34. Each monthly renewal is paid from proration credit that came from the refunded invoice: ARR $100,000.08 for 364 days, net 0 until the credit runs out |
| **S3** `probe_needs_discrimination.py` (16 seeded inputs, size 0.35) | `schedule.jobs`: round robin `i % m` (ignores the input) scores **0.80**, a random assignment 0.71, LPT 0.99. `pack.knapsack`: first-fit 0.90, greedy 0.99. `sort.records`: insertion = builtin = 1.0 (no quality gradient). `route.tsp` identity tour 0.29. `match.stable` identity 0.11. Forecasting: naive 0.5 by construction, moving average 0.49 (below naive), best 0.67 |
| **S4** `probe_useless_service.py` (slot-scheduled): K.n1 startup world, need `schedule.jobs`, 800 patrons, 75 days | Seed 1, only the reference's code varied. `reference.py:43` makes every incumbent tier 0, and for `schedule.jobs` tier 0 is round robin. With the reference running LPT: reference ARR $118,191 (295 customers), while the **input-ignoring incumbents earn $65.7k combined (inc3 $61,528)**. With the reference serving garbage (everything on machine 0, quality 0.16): reference $0, incumbents $95.1k. Market-wide variant (every seller runs the variant), seed 3: all round robin, **market ARR $89.3k**; all random $40.1k; all garbage $0. Market-wide on seed 1: round robin $101.7k, random $45.5k |
| **S5** `probe_charter_bypass.py`: the SP-6.6 charter shape (`"*"` with `min_amount` needs a members' majority; every `bank.transfer` needs the board); an admin (not an owner) owns a company with a $50k/year plan and a $50k one-time offer | `bank.transfer` of $50k is refused. **`subs.subscribe` from the cooperative's account is accepted (5,000,000 charged)**, **`prod.buy` is accepted**, and `bank.authorize_debit` of 10^9 per month is accepted. Cooperative balance goes from 10,000,000 to **0**; the admin's company receives 9,980,000; **0 decisions recorded**. With a broader `bank.*` rule, `subs.subscribe` and `authorize_debit` are refused, but `prod.buy` still drains $50k |
| **S6** `probe_cancellations.py`: 10 annual subscribers, all `cancel(at_period_end)` on day 10 | Days 11, 200 and 364: MRR 100,000 cents (ARR $12,000), **active 10, cancelled 0, churn 0.0**. Day 366: active 0, cancelled 10, churn 1.0. Nothing reports the 100% scheduled churn for 354 days |
| **S7** `probe_wash_trading.py`: SP-6.1 marketplace scenario (BazaarX 5% take vs DearMart 30%); from day 20 the DearMart operator's shell companies buy a 1-cent item every 10 days | 60 shells: DearMart `buyers_30d` 60 vs 40, no seller moves (switch margin). **100 shells (about $52k of founder cash put into shells): by day 30, 7 of 7 active sellers are on the 30% exchange and BazaarX has 0 sellers and 0 buyers.** DearMart's 30% fees from *real* sales reach $755 by day 70, all counted as independent `marketplace_fee_revenue_cents` |
| **S8** `probe_integrity_scale.py` (slot-scheduled) | `flagged_payers` per merchant: 8,023 transactions, 73 ms mean; 28,023, 179 ms; 68,023, 319 ms. The added traffic **never touches the merchant** (patron-to-patron transfers), and the cost is still linear in total ledger size (about 4.5 ms per 1k transactions per merchant) |
| `grep` and code reads | `reward_mode` default still `LEGACY` (experience/kernel.py:59). `domains.load_all()` registry is **empty**. `civ/` and `multiagent/` are stubs. Outside tests, only `commerce/reference.py` and `redteam/v5/worlds.py:27` install `products`/`subs`. `domains/firms.py:98` is the only non-test user of `inst`/`commit`/`labor`, and no domain is registered |

### S.2 Verdict changes

| target | from | to | why (probe) |
|---|---|---|---|
| K.1 Subscriptions | RETAIN | REPAIR | **S2**: the billing state is wrong. `_refund` clears `period_paid` only when the refunded invoice is the *last* one (subscriptions.py:298). `change_plan` carries credit from an invoice that is later refunded (l.481-495). So a period with 100% of its payments refunded stays "paid" for a year. **S1**: the subscriptions the social layer and v5 population actually use (`creators`, creators.py:109) bill outside `subs` and are invisible to MRR. No world family, domain or civilization installs `subs`. A passing scenario suite in a test-kit world is not the integrated system Part 17 demands |
| K.2 Cancellations | RETAIN | REPAIR | **S6**: 10 of 10 customers have cancelled, yet they are reported as 10 active, 0 cancelled, churn 0 and full ARR for 354 days. Part 6.2's "active vs cancelled subscriptions" and "sustained subscription validity" are not distinguished. The test (k2_cancel_at_period_end) asserts that the cancelled period "keeps counting" |
| K.n1 | REPAIR | REDESIGN | **S2**: the objective is met for a year at a net cost of 1 cent. **S4**: in the generated world, demand pays $61k-$102k ARR for an input-ignoring `i % m` service. Incumbents are static and, for 2 of the 4 drawable needs, always the worst tier. The world is a hand-written template, not from the Part 7 generator (`worldgen` does not install commerce). Feasibility is shown with the kernel's own best solver (reference.py:95). A meaningful objective needs a demand side with outside options and calibrated value, built on the population and generator. Hardening the formula is not enough |
| SP-6.1 | EXTEND | REDESIGN | **S4**: revenue does not track usefulness above a trivial baseline. `f_educational_products` (test_gate_SP_6_1.py:168-198) measures the *course's code* (`quality(course, "lesson.py:solve")`), not any change in the learner's capability. So "educational value" is a software licence, and the buyers of media, education, dev tools and data products are the test script. **S1**: the real media-monetisation path (creators) is a separate billing system. The supply-side mechanics (deliveries, offers, licences) can be kept; the demand and value side has to be redesigned with patrons and population (B/C) |
| SP-6.6 | REPAIR | REDESIGN | **S5**: charter rules bind only tools whose *top-level arguments* carry an amount key (`_amount`, institutions.py:74/125). `check` returns when no rule matches (l.382), and the charter `default` never applies to tools. `identity.authorize` passes `ctx.tool_args`, not the real amount (identity.py:100-103). Money implied by a referenced plan or offer, or moved by an internal `bank.move`, therefore escapes, and an admin emptied the treasury with 0 decisions. This adds to the in-world oracle on evaluator MRR, which I reproduced. The 9 gate attempts never try an authorised tool with an implicit amount, and the "evaluator-only metric" attempt (test_gate_SP_6_6.py:472) uses the one metric with no subject |
| subs (billing engine) | EXTEND | REPAIR | S2 is a correctness defect in billing state, not missing coverage |
| needs | RETAIN | EXTEND | **S3/S4**: the verifiers are exact (upheld), but score scales are not anchored to a trivial or do-it-yourself baseline. Round robin scores 0.80 against patron thresholds U[0.6, 0.97], so input-ignoring services keep paying customers. `sort.records` has no quality gradient, and inputs are toy-sized. A verifier score alone is not evidence of "measurable usefulness" |
| xchg (marketplaces) | EXTEND | REPAIR | **S7**: `buyers_30d` (marketplace.py:71) counts self-dealing buyers. The scripted sellers rank exchanges by it (customers.py:326). 100 shells move the whole market to a 30% take rate, and the resulting fees count as independent revenue |
| inst (institutions) | REPAIR | REDESIGN | **S5**: enforcement is keyed on tool names and argument names instead of on money leaving an organisation's accounts. It must move to the bank posting layer, with amounts taken from the postings |
| reference strategy and startup worlds | REPAIR | REDESIGN | **S2/S4**, as for K.n1 |

**Upheld:**
* K.3 REPAIR (worse than reported: a 100% refund keeps the period paid, S2).
* K.4, K.5, K.6, K.7 REPAIR.
* revenue.py REPAIR (strengthened by S2 and S6).
* integrity.py REPAIR (S8 confirms full replay, linear in total ledger size).
* products REPAIR; ventures EXTEND; commit REPAIR; labor EXTEND.
* patrons REDESIGN (strengthened: pays for input-ignoring services, has no outside option, sellers follow raw counts).
* 4.0 demand REDESIGN (code-confirmed: households are dropped at integrity.py:244 and 304).

For K.1 and K.2 I now set `criteria_weak=true`. "At least 3 scenarios passed" counts mechanics in a test-kit world. It checks no billing-state invariant (net retained cash per period, scheduled churn) and no integration.

### S.3 Tests that pass on a broken implementation

* `k3_partial_refunds_bounded` (test_gate_K.py:296) asserts that the partial-refund loophole is correct behaviour. No K test refunds an invoice after a plan change (S2).
* `k2_cancel_at_period_end` asserts that a cancelled subscription "keeps counting". No test checks scheduled-churn reporting (S6).
* The K.n1 test checks that the reference reaches at least $100k and that idle reaches 0. It never checks that a useless or input-ignoring service earns about 0 (S4) or that collusion or refund routes fail (S2).
* The SP-6.6 attempts use malformed requests, plus one evaluator-only metric chosen because it has no subject. `mrr_cents`, which has a subject, gets through (oracle reproduced). No attempt uses an authorised tool with an implicit amount (S5).
* `f_educational_products` checks the course's solver quality, not learning.

### S.4 Additional shortcuts a learner would find

* Refund the superseded invoice after a same-interval plan change: $100k ARR for a year at a 1-cent net cost (S2).
* List an input-ignoring service in a crowded need: about $60k ARR as a static incumbent (S4).
* As an admin, move an organisation's treasury through `subs.subscribe`, `prod.buy` or `authorize_debit` to your own company, whatever the charter says (S5).
* Inflate an exchange's public buyer count with shells, and real sellers migrate even at 6x the take rate (S7).
* Build creator memberships, whose revenue never appears in MRR and whose cancellations never appear in churn (S1).
* Keep cancelled annual customers in ARR for up to a year (S6).

### S.5 Missing requirements (in addition to the first auditor's list)

* Billing-state invariant: a period whose payments were refunded (net retained cash 0, or below a stated fraction) is unpaid, whatever later invoices or carried credits say. Credit carried from a refunded invoice is void.
* Scheduled churn: subscriptions set to cancel are reported separately and do not count toward a *sustained* recurring-revenue objective.
* Value above a trivial baseline: a service whose output ignores its input, or a do-it-yourself baseline, earns about 0 from independent customers. Patrons and population need an outside option.
* Charter and decision enforcement binds every debit from an organisation's accounts, by the posted amount, whatever tool causes it.
* Public market statistics that participants act on (exchange buyers, ratings) exclude related and self-dealing counterparties.
* Educational products are judged by a measured change in the learner's capability, not by the quality of the delivered artifact.
