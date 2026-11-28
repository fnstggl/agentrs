# Change request SG (amendment items 6, 7) — agent.py hooks for ST

Owner of the change: ST (agent.py). Requested by SG. Diff against `agent.py` of HEAD `d1d06dc` / `8d82f91`
(identical file); every anchor line below also exists unchanged in ST's working copy of 2026-11-28 09:35 UTC.
Tested on a patched copy (`git archive HEAD` + this diff) in SG's scratch directory with
`PYTHONPATH=<copy>/cam_live/src` (results in `docs/SG.md` §5). Nothing else in agent.py needs to change for SG: the
skill executor's new behaviour (re-executable steps, learned applicability, composition, `skill.discover`, refinement)
lives in `harness/skills.py`, the learned transfer estimate in `reward/selfgoals.py`.

## SG-1 composition: `skill.run` stays procedural

`_NOT_PROCEDURAL` excluded every `skill.` operation from the verified-progress traces offered to `skill.induce`, so a
sequence that ran a skill (e.g. navigate, then `skill.run`) could never become a composed skill (spec 12A.9 "select
and compose procedures"; amendment item 6 "acquired skills can be composed into more sophisticated actions"). The skill
library's own management operations stay excluded. `harness/skills.py` now allows exactly `skill.run` inside a skill
(depth limit 3, a skill cannot contain itself, only validated skills, scopes = union of the sub-skills' scopes).

## SG-2 transfer evidence for goal-free mode

`GoalFreeMode`'s expected transferable competence was the constant 1.0 (AE-04 partial, KF-E21). It is now learned
(`reward.selfgoals.TransferEstimator`) from outcomes of later *independently assigned* objectives: each is scored by
the goal-free posterior and by the reference prior that posterior started from, on the features the predictor computed
for the executed action. The agent has to hand those outcomes over (`_transfer_evidence`, only when the current
objective is an assigned one, never during a goal-free period); the features come from the predictor's cached
commitment of the prediction (`Predictor._cache`, alias-resolved) — no extra model call. Counter `transfer_evidence`.

## SG-3 restore: the reference prior of the transfer evidence

After the goal-free fast state is restored, `GoalFreeMode.after_fast_restore()` restores the reference prior (new
checkpoints) or takes the restored state as the reference (checkpoints written before SG: earlier goal-free learning is
then not credited; `goal_free.restore_notes` says so). The live planning run's checkpoints load unchanged (the new
`state_dict` keys are optional).

## Exact diff

```diff
--- agent_head.py	2026-11-28 09:34:31.155369360 +0000
+++ patched/cam_live/src/cam_live/agent.py	2026-11-28 09:34:31.255367296 +0000
@@ -211,7 +211,8 @@
 PROV_SKILL = "skill_library"
 PROV_EXPERIENCE = "experience_trace"         # arguments taken from the agent's own verified-successful executions
 #: operations that never belong to a reusable procedure (bookkeeping, waiting, the skill library itself)
-_NOT_PROCEDURAL = ("wait", "internal.", "skill.", "program")
+_NOT_PROCEDURAL = ("wait", "internal.", "skill.list", "skill.describe", "skill.induce", "skill.discover", "program")
+#: (SG) ``skill.run`` stays procedural: a sequence that ran a skill can become a composed skill (spec 12A.9)
 _NONTEXT_INPUTS = ("checkbox", "radio", "submit", "button", "hidden", "file", "image", "reset", "range", "color")
 
 
@@ -1997,6 +1998,8 @@
 
     def _goal_free_outcome(self, o: OutcomeObservation, owner: Dict[str, str], q: Optional[OutcomeQuery]) -> None:
         aid = owner.get("action_id", "")
+        if q is not None and aid not in self.active_goals and not self._goal_free_active():
+            self._transfer_evidence(o, owner, q)
         if q is None or aid not in self.active_goals:
             return
         rec = self.store.get(owner.get("exp_id", ""))
@@ -2004,6 +2007,29 @@
         self._goal_outcome(aid, canonical_query(q, [aid]).metric, float(o.value), cost,
                            censored=o.status == OutcomeStatus.CENSORED)
 
+    def _transfer_evidence(self, o: OutcomeObservation, owner: Dict[str, str], q: OutcomeQuery) -> None:
+        """(SG) An outcome of an independently assigned objective is evidence about what goal-free learning was worth:
+        goal-free mode scores it with its posterior and with the reference prior that posterior started from, on the
+        features the predictor computed for the executed action (its learned transfer estimate, spec 11.4)."""
+        fn = getattr(self.goal_free, "observe_assigned", None)
+        if not callable(fn) or o.status != OutcomeStatus.OBSERVED or o.value is None:
+            return
+        cache = getattr(self.predictor, "_cache", None) or {}
+        qid = (getattr(self.predictor, "_alias", None) or {}).get(o.query_id, o.query_id)
+        c = cache.get(qid)
+        rec = self.store.get(owner.get("exp_id", "")) if owner.get("exp_id") else None
+        if c is None or rec is None or rec.decision is None:
+            return
+        key = self.goal_free.key_of(rec.decision.chosen, canonical_query(q, [owner.get("action_id", "")]))
+        try:
+            gain = fn(key, c.phi, q.kind, float(o.value), float(q.exposure or 1.0),
+                      objective_id=getattr(self.task, "objective_id", ""))
+        except Exception as e:  # noqa: BLE001 — transfer evidence is an aid; its absence is reported
+            self.last["transfer_error"] = f"{type(e).__name__}: {e}"
+            return
+        if gain is not None:
+            self.counters["transfer_evidence"] += 1
+
     # ============================================================================================ KF-E01
     def _credit_for(self, rec: ExperienceRecord) -> Optional[Tuple[float, str, bool]]:
         """The stored retrospective credit the policy update's goal term uses for one decision (spec 9.2: "use
@@ -2370,6 +2396,10 @@
             self.goal_free.load_state_dict(aux["goal_free"])
         if aux.get("goal_free_fast"):
             self.goal_free.fast.load_state_dict(aux["goal_free_fast"])
+        if aux.get("goal_free") or aux.get("goal_free_fast"):
+            fn = getattr(self.goal_free, "after_fast_restore", None)     # (SG) reference prior of transfer evidence
+            if callable(fn):
+                fn()
         if aux.get("py_rng"):
             self.rng.setstate(_rng_from(aux["py_rng"]))
         if aux.get("proposer_rng"):
```
