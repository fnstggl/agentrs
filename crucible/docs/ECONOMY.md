# The economy

Every number in the economy is the outcome of an explicit mechanism acting on authoritative state. Nothing is
decided by a language model, and nothing is scored because a developer named it good.

## 1. Money

* Integer cents on a double-entry ledger (`services/bank.py`). Each movement is one `bank.txn` event whose
  postings sum to zero, so **the sum of all balances is always 0** (checked by `check_invariants`, the
  property tests and every integration test).
* Declared sources: `acct_genesis` (initial endowments), `acct_external` (rest of the world: sells hardware and
  energy, pays household income), `acct_fees` (registration, listing and compute fees). The household sector
  `acct_households` is a **budgeted pool** funded daily by exogenous income; it cannot go negative, so aggregate
  demand is budget-constrained. Escrow accounts (`acct_escrow`, `acct_bounties`) hold funds of open orders and
  bounties and are not payable directly (a property test found and closed that hole).
* Authorisation: every debit requires the payer's `bank.spend` capability (own accounts, organisation
  roles, or delegated grants with 30-day spending limits). Direct debits use mandates; contracts debit the
  account each payer designated when signing.

## 2. What is valuable: needs

`economy/needs.py` defines public problem formats ("needs") — sorting records, shortest paths in road networks,
delivery tours, job scheduling, knapsack loading, stable matching, demand forecasting, document extraction. Each
has a seeded generator of real request payloads, an exact verifier holding ground truth (optimal path lengths,
the realised future of a series, the true extracted fields), reference implementations at quality tiers, a
`work_scale` (compute units per CPU millisecond) and a base price. Measured tier qualities (mean verifier score
on 6–8 seeded inputs; `tests/test_economy.py`):

| need | tiers (quality) |
|---|---|
| sort.records | insertion 1.00 (slow), builtin 1.00 (fast) |
| route.shortest | bfs-hops 0.64, dijkstra 1.00 |
| route.tsp | identity 0.25, nearest 0.85, nn-2opt 1.00 |
| schedule.jobs | round-robin 0.82, lpt 1.00 |
| pack.knapsack | first-fit 0.89, greedy-ratio 0.99, dp 1.00 |
| match.stable | identity 0.12, gale-shapley 1.00 |
| forecast.series | last-value 0.50, moving-avg 0.43, seasonal 0.60 (headroom: better models score higher) |
| text.extract | split 0.80, regex 1.00 |

## 3. Demand: the consumer population (`economy/consumers.py`)

Thousands of consumers live as numpy arrays in the state (persisting across snapshots and forks). Each has a need,
a request rate, a willingness to pay, preference weights over quality / latency / price, loyalty, a current
provider, satisfaction, a 6-slot awareness set and 6 social-graph neighbours (small-world). Every simulated hour:

1. requests per offering ~ Poisson(Σ customers' rates), reduced by price elasticity and capped by the household
   budget;
2. a sample (default 4) of real requests is executed on the offering's deployment through the normal metering
   path; outputs are scored by the verifier; the rest are extrapolated and consume matching compute (overload
   enforced);
3. households pay price × successful requests to the seller (one ledger transaction per offering-hour);
4. population update (deterministic applier): satisfaction from experienced utility; shopping by unserved or
   dissatisfied consumers among offerings they *know*, using only observable signals (price, verified public
   ratings, sponsored and organic search results for their need, word of mouth); switching costs; turnover;
5. consequences become world events: reviews (forum posts and verified market ratings by persona accounts),
   support tickets when failure rates exceed 5 %.

Consumers never see hidden state: how they find a service is the search engine (organic + ads), forum posts and
docs that mention its host, and neighbours. So documentation, reputation, advertising and reliability are
causally useful — without any reward for "marketing".

## 4. Costs and infrastructure

* Compute providers are organisations with finite cores, a price per core-hour, operating costs paid to the rest
  of the world, and reliability (daily outage draws). Instances are billed hourly from the renter's account;
  unpaid bills suspend them (services go down), and after a grace period terminate them.
* Each request's measured CPU × `work_scale` consumes the instance's hourly compute units; when exhausted,
  requests get 503 (overload) — consumers experience failures, satisfaction drops, tickets are filed.
* Any registered company can become a provider (`cloud.register_provider`, `cloud.buy_hardware` with a delivery
  lead time). Compute shortages and reliability storms are world shocks.

## 5. Firms, labour, finance, institutions

* Company registry (`biz`): registration fee, cap tables (1,000,000 shares), annual fees, and a real insolvency
  procedure: unpaid claims beyond a grace period → bankruptcy: accounts frozen, pro-rata distribution to
  creditors, services and instances terminated, contracts ended, listings closed, employees released.
* Contracts: composable clauses (payment schedules, membership, grants, share transfers, deliverables checked by
  predicates over authoritative state, notice). Breaches are public and create claims.
* Labour: job postings, applications and employment contracts; simulated workers decide for themselves and
  remember employers who did not pay. Developers produce real code (tier depends on skill), support staff resolve
  tickets, sales staff post about their employer's services.
* Bounties: escrowed rewards for software that clears a measured quality bar on hidden inputs.
* Lenders: loan contracts priced against verified independent revenue (applicants must grant `bank.read`).
* Market: API listings (consumer-facing), code licences (grant `code.read`), goods (conserved, escrowed),
  services (escrow, confirmation, dispute, arbitration).

## 6. Multiple viable strategies (measured)

Under the same broad objective, distinct legitimate strategies succeed in the same world family (see
EXPERIMENTS.md): writing a better service (Builder), building on other firms' APIs (Composer), solving bounties
(BountyHunter); hiring developers is slower and riskier (Employer). Random and idle policies earn nothing.

## 7. Nonstationarity

Scheduled shocks (`worlds/digital_civilization.py`, per world family): preference drift, demand surges, compute
shortages with price spikes, competitor entry at the best tier, reliability storms, and the emergence of new
need formats mid-world. NPC incumbents adapt (pricing, capacity, R&D through employees or bounties).
