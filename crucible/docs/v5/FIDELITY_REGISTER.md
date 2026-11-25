# Crucible 5.0 fidelity register

Every important behavioural or economic mechanism has a record here (Part 5.4). Categories:

* **empirically fitted** - parameters estimated from a named dataset, with held-out evaluation;
* **empirically validated** - not fitted to the data, but checked against it on prespecified metrics;
* **inspired by research** - form taken from published work, parameters not estimated;
* **plausible, unvalidated** - a reasoned assumption;
* **deliberate simplification** - known to omit something, on purpose;
* **known unrealistic** - kept for control or tractability, and labelled.

A neural network is not "calibrated" because it is a neural network, and fitting a historical dataset does not show
real-world predictive validity. Each record states the population, platform and period of its evidence and its limits.

| mechanism | module | category | evidence (dataset, period) | held-out result | limits |
|---|---|---|---|---|---|
| subscription buyer choice (value = wtp x expected quality - price + trial bonus) | commerce.customers (`patrons`) | plausible, unvalidated | none | none | one need per buyer; no switching while satisfied (F1) |
| buyer usefulness check (own instances scored by the need's exact verifier) | commerce.customers (`patrons`) | deliberate simplification | none | none | needs with exact verifiers only (F1) |
| goods sellers' exchange choice ((buyers + 1) x (1 - take rate), switch margin) | commerce.customers (`patrons`) | plausible, unvalidated | none | none | one exchange per seller, cost-plus prices (F1) |
| goods buyers' listing choice (cheapest acceptable on a trusted exchange) | commerce.customers (`patrons`) | plausible, unvalidated | none | none | trust from public dispute record only (F1) |
| dunning schedule (retries after 1, 3, 7 days) | commerce.subscriptions (`subs`) | plausible, unvalidated | none | none | fixed schedule (F1) |
| billing month 30 days, year 365 days | commerce.subscriptions (`subs`) | deliberate simplification | none | none | no calendar (F1) |

The 55 mechanisms registered by Crucible 4.0 (`evaluation/fidelity.py`, `docs/experiments/ow_fidelity.json`) keep
their 4.0 categories: 37 plausible, 18 known unrealistic, none calibrated. v5 records are added by the workstreams that
build each mechanism and checked by SP-5.4.
