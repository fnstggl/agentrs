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

## S-5 Selection ignores the world model's predictions; the internal reward pays zero-value actions (owner: INT2 / B / C)

**What (measured, `docs/S/DIAGNOSIS.md`, debug scale, 10 seeds).** With the default rule `policy` the chosen
candidate's true value is 0.12 of the best; the candidate ranked first by the agent's *own* registered goal
prediction is 0.30–0.39 of the best. Shadow evaluation on the recorded contexts: behaviour 1.42 expected adopters per
decision (uniform 1.43), greedy-on-predicted-goal-mean 5.05 (E) / 6.83 (D). 64 % of decisions and 70–73 % of the
positive internal reward go to operations with no direct effect (probe, noisy gauge, wait); Spearman(internal return,
verified progress) = -0.17 [-0.32, -0.05] for E.

**Why it matters.** Learning in the world model / evaluator cannot improve behaviour if selection is driven by a
policy trained only on an internal reward that is anti-correlated with verified progress. Candidates for the
owners: (a) make the policy's training signal include the evaluator's / world model's goal value (e.g. advantage
from predicted goal-relevant outcomes, or verified-outcome policy gradient), (b) a selection rule that combines the
policy with the predicted goal value, (c) check the bootstrap reward's exploration term (it rewards uncertainty
reduction on the noisy gauge and probes regardless of decision relevance). Re-measure with
`python -m cam_live.eval.cli analyse` (shadow + action-quality sections) after a fix.

## S-6 BLOCKER for substantive evaluation: open-ended generation does not reach the objective's actions (owner: INT2 / B)

**Measured** (HEAD 8e666ad, real Qwen3-4B + 50m, substantive settings `use_backbone=True`,
`proposer_max_new_tokens` 512 / 1024, `proposer_k=4`; evidence `docs/evidence/S/v3/`):

* SW objective ("write solution.py defining solve(s) ..."), 12 decisions under the runtime
  (`probe_E_s99_sw_rle.json`): the agent never wrote a file. 11 of 12 chosen actions were schema-walker
  candidates (`internal.inspect`, `sandbox.list_files`, `wait`, `sandbox.job_status`); the backbone contributed
  1–2 of 10 candidates per decision, all of them `sandbox.list_files {"path": "solution.py"}`,
  `internal.inspect {"what": "actions"}` or `wait`; selection probabilities were ~uniform (0.09–0.12 each).
  Verifier: pass fraction 0.0 throughout (solution.py missing).
* Proposal probe, same objective, 3 rounds (`proposal_probe_sw.json`): 0 valid backbone actions. Qwen writes one
  composite `{"operation": "program", "arguments": {"steps": [...], "expected": ...}}` of inspection steps; it is
  rejected (`invalid_arguments`: unexpected properties `steps` / `expected` for the `program` tool) or truncated
  (`not_json`). No round contains `sandbox.write_file` with program text. Prompt: 12.6 k characters for 16 tools.
* MECH objective (query a local service, write model.py), 1024 tokens (`proposal_probe_mech_1024.json`): sensible
  experiments are written (`http.get .../measure?x=-3&y=3` at the corners of the domain), but in 2 of 3 rounds
  every one is rejected for a harmless extra property (`"method": "GET"`); 4 valid in the third round.

**Consequence.** With the current proposer and selection, substantive objectives that require writing software
cannot be attempted by the agent at all, and backbone-written candidates are a small, rarely chosen minority.
Requested (owners decide how): (a) the `program` tool's schema and the prompt must agree (the model is invited to
compose `program` but its natural argument form is refused); (b) tolerate / strip unknown argument properties that
do not change meaning instead of rejecting the whole action (or show `additionalProperties: false` in the prompt);
(c) make sure artifact-producing tools (`sandbox.write_file`) are reachable — e.g. ask for several single actions,
not one plan; (d) the selection rule should not dilute k backbone proposals among 20 walker candidates with a
uniform untrained policy (see S-5). Re-measure with `python -m cam_live.eval.proposal_probe` and
`python -m cam_live.eval.oe_runner sequence ...` (both in `cam_live/eval/`).

**S-6 update (X0, 2026-11-27 18:45 UTC, snapshot 737996e = HEAD with INT2's c5b1cb5 prompt rotation).** Real E seed
0, SW:rle, 15 decisions (lock-hold budget reached, 810 s, ~51 s per decision of which ~48 s inference):
**0 backbone-written candidates in all 15 decisions** (0 of 9 candidates each time), every chosen action from the
schema walker (`sandbox.list_files` x7 — all denied, S-4 —, `internal.inspect` x5, `wait` x2,
`sandbox.job_status`); verifier pass fraction 0.0 (no solution.py). Evidence:
`runs/cam_live/eval/v3/results/x0/E_s0.json` (copied to `docs/evidence/S/v3/` when X0 completes).
