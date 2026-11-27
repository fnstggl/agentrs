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

## INT2-6 (reward, `reward/bootstrap.py` / `reward/integrity.bound_reward`, owner C): the verified term saturates

**What / why (measured, `docs/evidence/validation/diagnosis/`, LabWorld F1, debug scale).** With the verified
progress of each decision now available (INT2-1 workaround: the decision's own attributed, signed outcomes), the
bootstrap reward of a decision that brought 1 adopter and of one that brought 42 is the same, 1.0:
`verified = 2.0 * log1p(v)` is 1.39 for v = 1 and `bound_reward(..., 1.0)` clips the sum to [-1, 1]. Returns are then
standardised within the batch, so the inner policy update cannot tell a good intervention from a poor one — only
"some adopters" from "none".

**Proposed diff.** Scale before bounding, e.g. `verified = w * tanh(log1p(v) / s)` with `s` the running scale of
`log1p(v)` over the lifetime (like `scale_normalise`), or bound with `bound * tanh(total / bound)` and a larger
`bound` for the verified component.

## INT2-7 (reward, `reward/metalearner.MetaLearner.policy_update` / `MetaConfig.inner_lr`, owner C): the live
policy update moves the policy too little to change behaviour

**What / why (measured).** `policy_update` applies K=2 SGD steps with `inner_lr = 1e-2` (the meta-gradient's inner
step) to the policy head + trunk. Over a 16-decision lifetime (5 updates) the heads' weight delta is 0.004–0.011 per
update and the selection probabilities stay within 0.062–0.073 for every candidate of a 16-candidate set (uniform:
0.0625): the policy that selects is, to two decimals, uniform after 128 decisions, whatever the reward says.
**Proposed.** Separate the step size of the *live* policy update from the differentiated inner step (or scale the
inner step to a target KL change of the policy per update, a trust region), so the live policy can become decisive
within the experience an objective provides; the meta-gradient keeps differentiating the update actually applied.
**Workaround.** None in INT2's files (the inner step is C's mechanism); the ablation with `optim.inner_lr = 0.3` is
reported in `docs/INTEGRATION.md` §9.

## INT2-8 (reward, `reward/metalearner.MetaLearner.state_dict` / `load_state_dict`, owner LRN): the support
batches of the retained update events are not checkpointed, so a restored agent meta-learns on different data

**What (measured).** `load_state_dict` restores every `UpdateEvent` with `ib=None`; `_steps_of` then rebuilds the
support batch from the store at the next meta-update. The rebuilt batch is not the one the live update used: in a
LabWorld run (debug core, 12 decisions, 4 events) the rebuilt batch differs from the cached one in `b`, `e_a`
(encoder features under the since-updated weights), `signals`, `r_boot` and `norm_state` for every event except the
newest (`docs/evidence/validation/diagnosis/inner_batch_rebuild.py`, output `inner_batch_rebuild_20261127.txt`). The continuing process differentiates
the update it applied, the restored one a different one: after the first meta-update following a load the reward
version differs, and `tests/integration/test_agent_recovery.py::test_save_fresh_process_load_gives_identical_decisions`
fails whenever that difference moves a later choice (seen as a different randomized-experiment arm at the 4th
decision after the load; the test passes in runs where it does not).
**Proposed.** Persist the support batch tensors of the retained events (`InnerBatch`, bounded by `max_events`) in
`state_dict`, or snapshot at event time exactly the inputs `build_batch` reads (features, signals, bootstrap reward,
normaliser state) — the meta-gradient should differentiate the update actually applied, so rebuilding in both
processes is not equivalent.
**Workaround.** None in INT2's files; the test is left failing when it diverges (not marked expected-to-fail).

## Requests from LRN (details and diffs: `docs/change_requests/LRN.md`)

* **LRN-1** `AgentSettings.selection_rule` default `"thompson_policy"` (already accepted by `from_cfg`; rule `policy`
  with `value_weight` stays selectable as the ablation). No other wiring: the selection calibration is shared through
  the policy head and checkpointed in `meta.state_dict()`; the trust-region policy step (INT2-7), matured replay
  (INT2-1 consequence), unsaturated bootstrap verified term + goal term (INT2-6) and the moving bootstrap mix are
  inside `reward/`.
* **LRN-3** (not blocking) record `reward_signals.epistemic_after` and `learning_progress` so the bootstrap's
  exploration / learning-progress terms have inputs in runtime runs.
* **INT2-8 done by LRN** (5ad3777): events that will still be differentiated are checkpointed with their support
  batch; experiment assignment no longer depends on the random experiment id. Both recovery tests pass.
* **LRN-6** a refused promotion's rollback also reverts psi and the policy head (not judged by the promotion gates):
  4-10 rollbacks per 8-lifetime debug run, 3 in the real-model pathway run, where it undid the applied meta-step
  before the next live update. Proposed diff (keep `reward` params + policy head across the rollback) in LRN.md.

### Applied by INT2 (round 2)

* **LRN-1** applied: default `selection_rule = "thompson_policy"`; `tests/integration/test_int2_selection.py` runs the
  `policy` + value-prior ablation explicitly.
* **LRN-6** applied with one difference: R_psi (its whole module state: parameters, normalisers, lifetime context)
  and the meta optimizer are kept across a rollback; the policy head is kept **unless** the refusal names its
  retention metric `decision_agreement` — that gate (frozen action distributions on earlier objectives' retention
  sets, `CamLiveAgent.retention_metrics`) does judge the policy, so there a rollback is the forgetting it guards
  against. `promotions[-1]["kept"]` records what stayed; test
  `tests/integration/test_int2_checkpoints.py::test_promotion_needs_independent_evidence_and_a_refusal_rolls_back`.
* **LRN-3** not applied this round (not blocking).
