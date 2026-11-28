# Change requests of PS (persistent internal representations, amendment item 3)

## PS-1 — agent.py: thread the persistent core state through the agent (for ST)

**What.** Attach a `cam_live.nn.persist.PersistentStateChain` to the model in `CamLiveAgent.__init__`, read the
current window with the carried state inside the decision-time calls (`_belief`, `update_beliefs`' `belief_value`,
`evaluate`'s `evaluate_candidates` and `planner.plan`), commit every decision to the chain in `record_result`,
save / restore the chain's bookkeeping in the agent checkpoint, report it in `status()`, close it in `close()`.

**Why.** KF-E04 / MODEL_ARCHITECTURE §11.1: every core call passes `state=None`; the core's persistent slots are
never carried, so `m_t` (spec 2.1) holds only the per-kind fast posteriors and the experiential backbone has no
recurrent or persistent latent state across decisions (spec 1.3). `nn/persist.py` (commit e8af02c) defines the
carried state and how it is trained; the model resolves `core_forward(tokens, None)` through the attached chain, so
predict/, credit/, reward/ and learning_api.py need **no** change: stored records re-encoded by the learner get
their own decision's carried state (context window -> its position, post window -> the next position), decision-time
calls inside a `live()` block get the state of the next decision. Without the chain (or `persistent_state.enabled:
false`) the agent computes exactly what it computed before.

**Compatibility.** Checkpoints written before PS load unchanged: `load()` passes `None` to
`PersistentStateChain.load_state_dict`, which sets the new retention parameter `exp.core.persist_rho` to 0 (the
restored model then computes bit-identically what it computed before, tested) and reports
`restore_report["persistent_state"] = {"legacy_checkpoint": true, "rho_initialised": 0.0, ...}`; learning opens rho
where carrying the state lowers the losses. The chain's anchors live durably in `<run_dir>/learner/core_state.sqlite`
(12 KB per decision at the 50m preset), the checkpoint records only its length and last decision id.

**Configuration.** New optional section `persistent_state` (`enabled` true, `unroll` 2, `anchor_cache` 2048,
`legacy_rho` 0.0); `nn` section and `arch_signature` unchanged.

**Tested** on a patched copy of the tree at e8af02c (this diff applied to agent.py, nothing else), PYTHONPATH pointing
at it: `docs/PS.md` §3 (runtime path, checkpoint across rebuilds, existing agent / runtime test directories).

**Exact diff** (against `agent.py` at 87eab34 = 257267c; it applies cleanly there with `patch -p1`). Against the
working copy of 09:20 UTC every hunk applied except the `status()` one (ST had changed that dict); there the line
to add is the one `"persistent_state": ...` entry.

```diff
--- a/cam_live/src/cam_live/agent.py
+++ b/cam_live/src/cam_live/agent.py
@@ -28,6 +28,7 @@
 from __future__ import annotations
 
 import collections
+import contextlib
 import copy
 import dataclasses
 import hashlib
@@ -58,6 +59,7 @@
 from .credit.select import EstimatorSelector
 from .learning_api import DecisionBatch
 from .nn.model import build_model
+from .nn.persist import PersistentStateChain, PersistSettings
 from .predict.candidates import BackboneProposer
 from .predict.heads import attach_heads, canonical_query
 from .predict.imagination import is_imagined
@@ -702,6 +704,14 @@
         self.heads = attach_heads(self.model)
         # ---- durable experience, registries ----
         self.store = SqliteExperienceStore(os.path.join(self.dir, RunFiles.STORE), seed=cfg.seed, durable=s.durable)
+        # ---- persistent latent state of the experiential core, carried across decisions (PS, KF-E04) ----
+        pss = PersistSettings.from_cfg(cfg)
+        self.persist: Optional[PersistentStateChain] = None
+        if pss.enabled:
+            self.persist = PersistentStateChain(self.model, self.store, pss,
+                                                path=os.path.join(self.dir, "core_state.sqlite"))
+        elif getattr(self.model, "persistent", None) is not None:
+            self.model.persistent = None                  # a shared model must not keep another agent's chain
         self.predictions = PredictionRegistry(os.path.join(self.dir, "prediction_registry.jsonl"), self.clock)
         self.experiments = ExperimentRegistry(os.path.join(self.dir, "experiment_registry.jsonl"), self.predictions,
                                               self.clock, seed=cfg.seed)
@@ -863,9 +873,15 @@
     def _items(self, refs: Optional[Sequence[str]] = None) -> List[ExperienceItem]:
         return self.store.items(list(refs if refs is not None else self._window()))
 
+    def _live(self) -> Any:
+        """Decision-time block: the current context window is read with the carried core state of the next
+        decision (``nn.persist``); stored records are re-encoded with their own decision's state."""
+        return self.persist.live() if self.persist is not None else contextlib.nullcontext()
+
     @torch.no_grad()
     def _belief(self, items: Sequence[ExperienceItem]) -> torch.Tensor:
-        return self.model.core_forward(self.model.tokenize([list(items)]), None).b
+        with self._live():
+            return self.model.core_forward(self.model.tokenize([list(items)]), None).b
 
     def _goal_text(self) -> str:
         return (self.task.objective if self.task is not None else "") or ""
@@ -1261,8 +1277,9 @@
                 self.fast_records.append(rec)
         items = self._items()
         if self.task is not None and self.task.objective.strip():
-            bv = self.predictor.belief_value(items, self._goal_text(), deadline=self._deadline(),
-                                             objective_scale=self._scale())
+            with self._live():
+                bv = self.predictor.belief_value(items, self._goal_text(), deadline=self._deadline(),
+                                                 objective_scale=self._scale())
             out.update(p_success=bv["p_success"], b_norm=bv["b_norm"])
         return out
 
@@ -1728,16 +1745,18 @@
         items = self._items(refs)
         did = self._decision_id()
         objective = self.task.objective_id if (self.task is not None and self.task.objective.strip()) else ""
-        evals = self.predictor.evaluate_candidates(
-            items, self._goal_text(), cands, queries=self._queries_for, decision_id=did, objective_id=objective,
-            deadline=self._deadline(), objective_scale=self._scale(), compute_voi=True, generator=self.gen)
+        with self._live():
+            evals = self.predictor.evaluate_candidates(
+                items, self._goal_text(), cands, queries=self._queries_for, decision_id=did, objective_id=objective,
+                deadline=self._deadline(), objective_scale=self._scale(), compute_voi=True, generator=self.gen)
         imagined = None
         allowed = self.planner.controller.allowed() if self.settings.planner else 0
         if allowed >= 1:                                   # only where multi-step accuracy has been measured
             try:
-                imagined = self.planner.plan(items, self._goal_text(), cands, now=float(self.clock.now()),
-                                             deadline=self._deadline(), objective_scale=self._scale(),
-                                             horizon=allowed, generator=self.gen)
+                with self._live():
+                    imagined = self.planner.plan(items, self._goal_text(), cands, now=float(self.clock.now()),
+                                                 deadline=self._deadline(), objective_scale=self._scale(),
+                                                 horizon=allowed, generator=self.gen)
                 for ev, im in zip(evals, imagined):
                     ev.extras.update({"imagined_value_mean": float(im.value_mean),
                                       "imagined_value_std": float(im.value_std),
@@ -1924,6 +1943,9 @@
             existing.outcomes = list(existing.outcomes) + [o for o in outcomes if o.query_id not in held]
             existing.duration_s = max(0.0, float(result.ended - result.started)) if result.ended else 0.0
             self.store.update(existing)
+        if self.persist is not None:                       # PS: this decision becomes the next position of the chain
+            self.persist.commit(decision.decision_id, context_refs=refs, post_refs=post,
+                                objective_id=decision.objective_id, ts=float(decision.ts))
         if immediate:
             self.store.mark_for_retraining([exp_id], "outcome_matured", outcome_key=immediate[0].dedup_key,
                                            now=float(self.clock.now()))
@@ -2344,7 +2366,8 @@
                                     "predictor": self._predictor_state(),
                                     "goal_free": self.goal_free.state_dict(),
                                     "goal_free_fast": self.goal_free.fast.state_dict(),
-                                    "py_rng": _rng_state(self.rng), "proposer_rng": _rng_state(self.proposer.rng)}
+                                    "py_rng": _rng_state(self.rng), "proposer_rng": _rng_state(self.proposer.rng),
+                                    "persistent_state": self.persist.state_dict() if self.persist is not None else None}
         info = save_checkpoint_at(path, state, notes=f"agent decisions={self.n_decisions}", clock=self.clock)
         self.checkpoint_id = info.checkpoint_id
         return info.checkpoint_id
@@ -2360,6 +2383,9 @@
         # (counter + restore report) and the current task is kept; an agent restored before it has a task compares
         # when the runtime sets one (set_objective).
         self.restore_report = dict(rep or {})
+        if self.persist is not None:                       # PS: None = checkpoint written before PS (rho closed)
+            self.restore_report["persistent_state"] = self.persist.load_state_dict(
+                ((st.aux_state or {}).get("agent") or {}).get("persistent_state"))
         self._checkpoint_auth_hash = "" if current is not None else str(getattr(st, "auth_config_hash", "") or "")
         aux = (st.aux_state or {}).get("agent", {})
         self.meta.load_state_dict(aux["meta"])
@@ -2550,7 +2576,8 @@
                                     if self.ckpts is not None and self.ckpts.promoted() is not None else ""),
             "last_promotion": self.promotions[-1] if self.promotions else None,
             "skills_known": len(self.skill_catalog), "surface": (self.surface or {}).get("url", ""),
-            "proposal": (self.last.get("proposal") or {}).get("stats", {})})
+            "proposal": (self.last.get("proposal") or {}).get("stats", {}),
+            "persistent_state": self.persist.status() if self.persist is not None else {"enabled": False}})
 
     # ============================================================================================ F-1 hooks
     def drain_experiments(self) -> List[ExperimentRecord]:
@@ -2589,6 +2616,8 @@
             return self.heads.policy(z, mask, b).probs[0].detach().cpu().double()
 
     def close(self) -> None:
+        if self.persist is not None:
+            self.persist.close()
         try:
             self.store.close()
         except Exception:  # noqa: BLE001
```
