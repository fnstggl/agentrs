# Change requests of workstream S (scientific evaluation)

## S-1 Runtime: observe until the deadline after the decision budget is exhausted (owner: F / INT2, `runtime/loop.py`)

**What.** `LiveRuntime` finishes with `exhausted` as soon as `limits.max_decisions` is used up, even when the
objective is still pending and its delayed consequences (adopters arriving hours later) have not been measured.
The agent then never receives the verifier's final verdict of that objective, so the objective is never closed in
its store: no RUDDER redistribution, no evaluator target, no meta-learning utility, no resolution of the
objective-success forecasts.

**Why.** Every evaluation lifetime with a decision budget (spec 16.9 "minimum new experience") needs the outcome of
the experience it consumed. Workaround in `eval.runner.close_out`: after the runtime stops, the evaluation advances
to the deadline and delivers the verifier's signed measurements of the due queries and its signed final verdict
through the agent's own `reconcile` / `update_beliefs` / `internal_rewards` / `learn` / `meta_learn` — the same for
every condition. A runtime option would make this the runtime's job.

**Proposed diff (sketch).** `RuntimeConfig.drain_after_exhaustion: bool = False`; when set and the exhausted limit is
`decisions`, enter a `draining` status in which `step()` performs steps 2–5 and 13–16 (no proposal, no execution)
and sleeps to the next due outcome until the verifier reports a terminal status or the deadline passes; then
`_finish(EXHAUSTED, ...)`.

## S-2 Weights-only export: the outcome scaler is not part of it (owner: D, `credit/checkpoint.py`; B, `predict/heads.py`)

**What.** `PredictHeads.scaler` (`KindNormaliser` statistics) lives in the checkpoint's `extra`, not in the
weights, so `export_weights` drops it and a transfer process starts with a fresh scaler while the world-model heads
were trained against the old normalisation.

**Why.** Spec 16.8 keeps "the trained neural parameters and necessary fixed configuration". Whether the scaler is a
learned quantity of the experience (then it must be exported as part of the model) or a per-objective statistic
(then resetting it is right) should be decided by its owner; today the transfer evaluation measures the
experienced weights *with* a reset scaler. Recorded as a limitation in `docs/S/RESULTS.md`.

## S-3 Authorization: `network: "none"` is treated as a network request (owner: EA, `harness/auth/policy.py`)

**What.** In staging mode `sandbox.python` with `{"code": ..., "network": "none"}` is **denied** with "network access
for generated code is not permitted in staging mode; network access requested without naming the hosts", while the
same call without the `network` argument is allowed. `"none"` is the first value of the tool's own enum, so the
schema walker (and a model that fills every field) asks for exactly this. Probe:
`eval/openended.py` environment, `PolicyService.authorize` on `sandbox.python {"code": "print(1)", "network": "none"}`
→ deny; without `network` → allow (measured 2026-11-27 18:20 UTC, HEAD 8e666ad).

**Why.** Every substantive (open-ended) family writes and runs programs; a correct, maximally restrictive request
is refused and costs a decision. **Fix:** only `network == "scoped"` (or any value other than "none") is a network
request.

## S-4 Authorization: `sandbox.list_files` without `path` is denied (owner: EA)

`sandbox.list_files {}` → deny ("names no file destination in arguments.path / target"); the tool schema makes
`path` optional (default: the workspace root). Either default the destination to the workspace root or make `path`
required in the schema.
