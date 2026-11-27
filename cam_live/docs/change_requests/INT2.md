# Change requests from the integration engineer, round 2 (INT2)

Each has a local workaround in INT2's own files; none blocks the integrated agent.

## INT2-1 (reward, `reward/assemble.py`, owner C): `verified_progress_delta` is never available in a runtime run

**What / why.** `record_signals` takes `verified_progress_delta` from verifier-trust OUTCOME items that are in
`post_item_refs` (H_{t+1}) but not in the context. In `runtime.loop.LiveRuntime` the verifier measures a decision's
goal-relevant queries in step 4 of a *later* cycle, after `record_result` has stored the record with its
`post_item_refs` (immutable since D-14). So the source is empty for every decision of a live or staging run, and
the bootstrap's `verified` term (weight 2.0) is always 0 — the one bootstrap component that is grounded in the
verifier never contributes. Measured: 0 of all records in the M-int staging runs carried a non-zero value.

**Proposed diff** (`record_signals`, after the post-item computation):
```diff
     if rec.post_item_refs:
         new = [i for i in post_items if i.ref not in ctx_refs and i.kind == ItemKind.OUTCOME
                and i.trust == Trust.VERIFIER]
         out["verified_progress_delta"] = float(sum(q.value for i in new for q in i.quantities))
+    goal = [(q, o) for q, o in matured_outcomes(rec) if q.goal_relevant and o.verified and o.value is not None]
+    if goal and not out.get("verified_progress_delta"):
+        prev = _previous_verified_level(rec, store, goal[0][0].metric)     # last verified reading before rec
+        if prev is not None:
+            out["verified_progress_delta"] = float(goal[0][1].value) - prev
```
(`_previous_verified_level`: the latest verified matured outcome of the same metric among the objective's earlier
records, by `seq`.) The ledger still decides what supervises the outer objective; this only feeds the labelled
bootstrap term.

**Workaround (in `agent.py`).** When the verifier's signed reading of the objective metric after a decision arrives,
the agent writes `decision.diagnostics["reward_signals"] = {"verified_progress_delta": value - previous signed
reading}` once (an added annotation: allowed by D-14), which `record_signals` reads through its stored-signal path.

## INT2-2 (credit, `credit/learner.py`, owner D): `holdout_experiments=True` stops learning in a single objective

**What / why.** D-14 holds out every *lifetime* that contains a decision of a registered randomised experiment. The
agent runs its randomised comparisons (spec 5.1, 8, 11.1) inside the objective it is learning from; with the default
a single experiment decision makes the whole objective validation data and no frequent update trains on it again.
In a single long live objective that is all of the agent's experience.

**Proposed diff.** Hold out the experiment's own decisions (and its arms' outcome windows), not the lifetime:
```diff
-        if self.settings.holdout_experiments:
-            if rec.decision.experiment_id or (rec.objective_id and rec.objective_id in self.experiment_objectives()):
-                return True
+        if self.settings.holdout_experiments and rec.decision.experiment_id:
+            return True
```
or keep the lifetime rule behind a setting that the integrated agent can turn off.

**Workaround (in `agent.py`).** The agent passes `holdout_experiments=False` to `LearnerSettings` unless the
configuration sets the key; experiment decisions stay identifiable by `decision.experiment_id`.

## INT2-3 (harness, `harness/browser.py`, owner EB): `browser.download` by element on a non-link

**What / why.** `browser.download {ref, obs_id}` on an element that is not a link (a select, a text field) resolves a
URL from the element (measured: `url:group/` for a `<select name=group>`), which the authorization service then
denies as `private_name`. The structural check accepts the action, so it costs a decision and a denial. Expected:
`validate` refuses a download by element when the element has no `href` and is not a download control.

**Proposed diff** (`BrowserExecutor.validate`): when `op == "browser.download"` and no `url` is given, require the
referenced element's role to be `link` (or `button` inside a form whose action is a download), else
`"download needs a link element or a url"`.

**Workaround.** None needed: the agent learns from the denial (it is an ordinary failed result).

## INT2-4 (runtime, `runtime/loop.py`, INT2 itself — noted for review): learner queries the agent resolves itself

`LiveRuntime._add_learner_queries` queues every registered prediction of the chosen action that the environment did
not cover, including `action.succeeded` and `observed.<metric>`, which the agent resolves from the result and from
later observations; the environment answers them `unobservable`. Harmless (the store keeps the agent's observed
value), but it costs queue work. Not changed in this round: the contract has no field by which a learner marks a
query as self-resolved; a `categories=["self"]` convention would do.

## INT2-5 (tests, owners B and D): two test modules named `test_providers.py`

**What / why.** `tests/predict/test_providers.py` and `tests/credit/test_providers.py` share a basename and the test
directories have no `__init__.py`, so collecting the whole default suite in one pytest run fails with
`import file mismatch` (measured: `pytest cam_live/tests --collect-only` → 1389 collected, 1 error).
`--import-mode=importlib` is not an option: the integration and runtime tests import their sibling helpers
(`int_helpers`, `rt_helpers`) through the prepended test directory.

**Proposed diff.** `git mv cam_live/tests/predict/test_providers.py cam_live/tests/predict/test_predict_providers.py`
(or the credit one to `test_credit_providers.py`).

**Workaround.** The whole-suite JUnit (`docs/evidence/test_results/INT2_full.xml`) is produced by two pytest runs —
the suite with `--deselect`-free `--ignore=cam_live/tests/predict/test_providers.py`, then that file alone — merged
into one file (`docs/evidence/test_results/merge_junit.py` is not needed: the merge is a `<testsuites>` wrapper).
