# Level 6 methodology: authorised real-world objectives (MASTER_SPEC Part 15.1)

Level 6 asks whether something learned in Crucible's simulations carries over to an authorised real-world
objective. Crucible ships the adapter (`crucible.research.v5.levels.Level6Adapter`) and this methodology; it ships
**no Level 6 result**. No real-world experiment has been executed in this repository, and nothing here supports a
claim of real-world generalisation (SP-15.1 scans the documentation for such claims).

## Gating

* The adapter is disabled by default. It refuses to construct unless `enable=True` is passed **and**
  `CRUCIBLE_LEVEL6=1` is set **and** an operator authorisation document is given.
* The document is the one of `crucible.realworld.authorization` (owner-only file, exact acknowledgement, base URLs
  the operator owns or is authorised to test, methods, request/byte/rate/wall-time limits, an expiry within 90 days,
  whether consequential actions need live approval). `crucible.realworld.RealWorldBridge` enforces it on every
  request and redirect, logs every request, response, refusal and measurement in a hash-chained log, and refuses
  to run in a process that hosts synthetic worlds.
* Learners act only through the bridge (HTTP requests it checks). They never receive credentials.

## Protocol

1. Preregister before any request: the objective, the measurement function, the competence threshold, the number of
   experiments and the analysis (as for simulation confirmation runs, `evaluator.preregister`).
2. Pick the learner checkpoint by development results on simulation partitions only; never by looking at real
   outcomes.
3. For each experiment: a fresh process loads the checkpoint (weights-only where the claim is about weights,
   `fresh.weights_only_eval` rules), runs the learner through `Level6Adapter.run`, and the outcome is computed by the
   experimenter's measurement function through the bridge's read-only reader (`RealWorldBridge.measure`). Nothing
   the learner reports is an input to the outcome.
4. Holdouts: real-world holdout evaluations are budgeted by the document (`holdouts`); a request beyond the budget is
   refused and logged.
5. Controls: the same objective with an untrained learner and with a scripted reference, in the same window, so the
   comparison is paired in time.
6. Analysis: paired differences with intervals over experiments; report every experiment, including failures and
   refusals, from the log.

## Claims

`Level6Adapter.claim()` returns no claim until at least `MIN_EXPERIMENTS_FOR_CLAIM` measured, logged experiments
exist; even then the claim is scoped to the authorised systems of the document. A claim of general real-world
competence is outside what this protocol can support.

## What is tested in this repository

Gate U.2 checks that the adapter is disabled by default, refuses without a valid document (no document, expired,
wrong acknowledgement), and - in a fresh process, against a local test service on 127.0.0.1 that the document lists
explicitly - that an authorised objective runs through the bridge and is measured by the experimenter's function.
That run is an interoperability check, not a real-world experiment.
