# Second-order emergence

Status: **implemented and measured for one mechanism family (compute capacity), 3 seeds, scripted competent
controls.** Not yet demonstrated with learned or LLM inventors, and not yet in other domains.

The question: can an invention create a problem that did not exist before it, and can another actor discover
and solve that new problem? The test is a matched counterfactual world that never gets the first invention.

## Mechanism (what the world implements; nothing below is scripted as an outcome)

| Link | Implemented by | Where |
|---|---|---|
| Demand has daily cycles that exceed a firm's fixed allocation | ops demand model | `economy/operations.py` |
| Overload is a loss event that is publicly reportable | `ops.loss_event("capacity_overload")`, firm minds post `ops_requests` | `actors/minds.py` |
| An autoscaler is a program that resizes a service every hour | `CapacityInventor` app code (`AUTOSCALE`) | `agents/inventors.py` |
| Resizing needs free cores at the service's provider | `cloud.resize` refuses when a provider is full and records a `capacity_unavailable` loss that survives rollback (`CrucibleError(persist=...)`) | `services/cloud.py` |
| Providers react to utilisation by changing price and ordering hardware | provider routine | `services/cloud.py`, provider minds |
| Firms report the new loss kind with the same machinery | `LOSS_TEXT` / `SEARCH_WORDS` include `capacity_unavailable` | `actors/minds.py` |
| A second actor answers *that* report class | `PlacementBroker`: rents at a provider with free cores, redeploys, releases the old instance, and sizes | `agents/inventors.py` |
| Adoption is decided only by the adopter's own trial | sequential randomised blocks with washout for stateful capabilities | `services/capabilities.py` |

The shortage comes from synchronisation. Every autoscaler sees the same daily cycle, so adopters all grow at the
same hour. Without autoscalers, firms keep their fixed allocations and the small cheap provider (`o:cloud1`, 40
cores) is never full. Nothing in the world mentions placement until the broker publishes it. The broker is
present in *both* worlds; it acts only when a public `capacity_unavailable` report exists.

## Experiment

`examples/demo_second_order.py [seed] [days] [cheap_cores]`, family `open_tight_compute`, 6 service firms,
`cheap_cores=40`, 70 days after the fork. The world forks at the first public problem report (day 1).
* **Factual:** `ada` = CapacityInventor and `bea` = PlacementBroker.
* **Counterfactual:** `ada` idle and `bea` = PlacementBroker.

Every stage is read from authoritative state. Raw output: `docs/experiments/second_order_{1,2,3}.json`. Each
factual world regenerated from its log to the identical hash.

| Stage (factual) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 1 first overload report (day) | 1.98 | 1.98 | 1.98 |
| 2 autoscaler published (day) | 2.62 | 2.62 | 2.62 |
| imitating entrants (open source forks of the autoscaler) | 2 | 2 | 2 |
| 5 first `capacity_unavailable` loss (day) | 4 | 4 | 4 |
| 6 first public report of it (day) | 5.98 | 21.98 | 21.98 |
| 7 placement broker published (day) | 6.71 | 22.46 | 22.46 |
| 8 placement trials (firms) | 4 | 5 | 4 |
| placement adopted (active at end) | 0 (2 trials still running) | 2 | 1 |
| 9 migrations to the other provider by `svc:placer` | 1 | 0 | 1 |
| 10 cheap-provider hardware orders (cores) | 10 + 12 | 10 + 12 | 10 + 12 |
| autoscaler adopted at end (firms) | 2 | 0 | 1 |

| Counterfactual (no autoscaler) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| `capacity_unavailable` losses | 0 | 0 | 0 |
| public reports of it | none | none | none |
| placement published | no | no | no |
| migrations | 0 | 0 | 0 |
| provider hardware orders | none | none | none |

Totals over 70 days (cents; requests):

| | factual s1 | cf s1 | factual s2 | cf s2 | factual s3 | cf s3 |
|---|---|---|---|---|---|---|
| requests served | 4.91M | 3.29M | 4.78M | 3.43M | 5.07M | 3.89M |
| overload events | 0.94M | 2.87M | 0.89M | 3.12M | 0.65M | 2.57M |
| `capacity_unavailable` events | 535 | 0 | 450 | 0 | 401 | 0 |
| firm revenue | 13.6M | 12.3M | 10.8M | 9.8M | 12.0M | 11.1M |
| firm compute spend | 7.15M | 2.47M | 6.96M | 2.69M | 6.55M | 2.93M |
| firm operating result | 5.49M | 9.87M | 2.35M | 7.06M | 5.42M | 8.19M |

## What the results show, and what they do not

1. **The new problem is caused by the invention.** All three seeds show it. `capacity_unavailable` losses start
   on day 4 in every factual world and never occur in the matched counterfactuals. Those counterfactuals have the
   same firms, the same demand and the same broker.
2. **A different actor discovered and answered it.** The broker acted only on the public report of the new loss
   kind. It published 0.5–0.7 days after that report. It was then trialled by 4–5 firms, adopted by 2 firms in
   seed 2 and 1 firm in seed 3, and moved one service to the second provider in seeds 1 and 3.
3. **A third-order market reaction appeared that was not designed as a stage.**
   * The cheap provider's price rose from 140 to a peak of 407–440 cents per core-hour (days 21–31). In the
     counterfactual it fell to 75.
   * The provider ordered 22 cores of hardware.
   * The pricier provider (105) became the *cheaper* one, which is what made migration worthwhile.
   * As adopters dropped autoscalers whose trials did not pay, the cheap provider's price fell back (119–161 by
     day 70).
4. **The invention was not good for the firms overall in this world.** Autoscaling served 30–49% more requests
   and raised revenue 7–11%. But compute spend rose 2.2–2.9x, partly because of the price spike the adopters
   caused, so total firm operating result was lower in every factual world. Most firms' own trials said "did
   not pay" and they removed the autoscaler. This is consistent with their trial evidence, not a bug.
5. **Confound: placement adoption is not purely about placement.** The broker also sizes the service every hour,
   so it subsumes autoscaling. In seed 2 two firms adopted it although it never had to migrate them. A cleaner
   test would offer a migration-only service; not done yet.
6. **Weak points.**
   * The inventors are scripted competent controls that react to report classes. They are not learned agents.
   * The discovery lag (days 6 vs 22) depends on when a firm with `share` traits posts the report.
   * Three seeds of one family is not a distribution.

## Related ablations (same machinery)

* **Diffusion is not inert** (`examples/diffusion_test.py`, `docs/experiments/diffusion_test.json`, 45 days, seeds
  1–3).
  * With the sharing channels on, 2 public outcomes and 4 beliefs formed from others' evidence per seed; a
    late-majority firm trialled in seeds 1 and 2.
  * With `ablate_sharing`, the counts were 0, 0 and no late-majority trials.
  * No late-majority firm *adopted* in either arm: their own trials did not pay. So sharing changes who
    *tries*, and verified own outcomes decide who *keeps*.
* **Trial-design validation** (`examples/trial_methods.py`, `docs/experiments/trial_methods_operating_result.json`).
  For predictive maintenance in `open_industry`, neither design reliably recovers the matched-counterfactual
  effect:
  * sequential blocks with washout: sign agreement 0.50, bias +1,022/day, RMSE 41,889;
  * paired-day switchback: sign agreement 0.44, bias -8,759/day, RMSE 24,454.

  Washout removes most of the bias but needs about 35 days and leaves high variance. Daily industrial operating
  results are dominated by lumpy sales. This is a limitation of in-world experimentation that firms in the world
  face too, and it is documented rather than tuned away.
