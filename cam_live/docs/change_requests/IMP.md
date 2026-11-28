# Change request IMP -> ST (agent.py): imagination in the decision (amendment 2026-11-28 item 2)

Owner of the file: ST. Requested by IMP. Status: **applied by ST in `dde7d2d`** (together with PS-1, RW-1/2,
SG-1..3; ST tested it on a clean snapshot of HEAD 00f8913 + agent.py: `test_imp_agent` among 33 passed). Before that,
IMP tested it on a patched copy (see below).

## What

`CamLiveAgent.evaluate` already calls `LatentPlanner.plan` and records three imagined numbers that no rule reads.
After IMP (`predict/planner.py`, `predict/policy.py`, `predict/predictor.py`, `predict/imagination.py`,
`predict/world.py`, `predict/heads.py`; commits `bb24f96` .. `00f8913`, see docs/IMP.md §6):

1. `plan(...)` gets the predictor of the same decision (`predictor=self.predictor`, `evaluations=evals`): the planner
   reuses its belief / latents / world view and values imagined continuations in the same goal units as the real
   evaluation (`Predictor.goal_values`), with the candidates' feasibility and costs.
2. Every candidate's `extras` receive the planner's numbers (`ImaginedEvaluation.extras()`: `im_cont`, `imc_<p>`,
   `im_horizon`, `im_info_value`, `im_next`, ... plus the three keys recorded so far). Rule `thompson_policy`
   (`DecisionMaker.thompson_policy`) reads them from the evaluations it is given: **no change to `select()`**.
3. The planner state in checkpoints is `self.planner.state()` (the controller's keys + the imagination
   calibration) and is restored with `self.planner.load_state(...)`, which also reads the old bare controller state
   of the live run's checkpoints (calibration then starts from its prior; `planner.restore_info` says so).
4. Comment of `AgentSettings.planner` updated (it was "recorded; no rule selects on them").

The ablation switch is in the predict section (`predict.imagination_select = false`: weight 0, the planner still runs
and the rule still records both distributions); no agent setting is added.

## Why

Amendment item 2: imagined consequences must enter actual decisions, gated by a weight learned from verified outcomes.
Without (1)-(2) the rule never sees imagined values; without (3) the imagination calibration (learned from verified
multi-step windows) is lost at every restart.

## Exact diff (against agent.py as of 2026-11-28 ~09:40 UTC; applies with `patch -p3` from the repository root's
`cam_live/src/cam_live` or `git apply`)

```diff
--- a/cam_live/src/cam_live/agent.py
+++ b/cam_live/src/cam_live/agent.py
@@ -143,7 +143,9 @@
                                              # INT3A); > 0 bounds the list (ablation only, recorded per decision)
     proposer_fit_prompt: bool = True         # shrink the observation digest (never tools / objective) so the
                                              # prompt fits the backbone's prompt window, which keeps the END
-    planner: bool = True                     # imagined values of candidates (recorded; no rule selects on them)
+    planner: bool = True                     # multi-step latent imagination of every candidate (predict.planner);
+                                             # rule thompson_policy weights it by a calibration on verified
+                                             # multi-step outcomes (predict.imagination_select = false: weight 0)
     goal_metric_query: bool = True           # ask the verifier for the objective metric after every action
     skills: bool = True                      # validated skills and skill induction from verified-successful runs
     induce_min_steps: int = 2
@@ -1882,11 +1884,10 @@
             try:
                 imagined = self.planner.plan(items, self._goal_text(), cands, now=float(self.clock.now()),
                                              deadline=self._deadline(), objective_scale=self._scale(),
-                                             horizon=allowed, generator=self.gen)
+                                             horizon=allowed, generator=self.gen, predictor=self.predictor,
+                                             evaluations=evals)
                 for ev, im in zip(evals, imagined):
-                    ev.extras.update({"imagined_value_mean": float(im.value_mean),
-                                      "imagined_value_std": float(im.value_std),
-                                      "imagined_horizon_steps": float(im.horizon_steps)})
+                    ev.extras.update(im.extras())
             except Exception as e:  # noqa: BLE001
                 self.last["planner_error"] = f"{type(e).__name__}: {e}"
         self.pending_eval = {"decision_id": did, "refs": refs, "imagined": imagined is not None,
@@ -2492,7 +2493,7 @@
                                 auth_config_hash=auth_config_hash(self.task) if self.task is not None else "",
                                 reward_version=self.meta.reward.version,
                                 extra={"agent": self._json_state(), "scaler": self.heads.scaler.state(),
-                                       "planner": self.planner.controller.state()})
+                                       "planner": self.planner.state()})
         state.aux_state = dict(state.aux_state or {})
         state.aux_state["agent"] = {"meta": self.meta.state_dict(), "meta_batches": self._meta_batches(),
                                     "predictor": self._predictor_state(),
@@ -2531,7 +2532,7 @@
         ex = st.extra or {}
         self.heads.scaler.load_state(ex.get("scaler", {}))
         if ex.get("planner"):
-            self.planner.controller.load_state(ex["planner"])
+            self.planner.load_state(ex["planner"])
         self._load_json_state(ex.get("agent", {}))
         if current is not None and self.restore_report.get("auth_config_changed"):
             self.counters["auth_config_changed_on_load"] += 1
```

## How it was tested

`$SCRATCH/IMP/sync.sh` copies `src/` and applies this diff; with `PYTHONPATH=<copy>`:

* `tests/predict/test_imp_agent.py::test_agent_decides_with_imagination_and_learns_its_weight_from_verified_windows`
  (skipped in the worktree until the diff is applied; it passes on the patched copy — run listed in docs/IMP.md),
* the paired ablation `python -m cam_live.predict.imp_eval` (debug scale, T1, 6 seeds x 2 arms) and the full-size runs
  in docs/IMP.md all ran on the patched copy.

The rest of the test suite does not depend on it (the old keys `imagined_value_mean / _std / _horizon_steps` are
still written).
