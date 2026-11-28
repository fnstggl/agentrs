# Change requests of RW (amendment item 1, genuinely learned rewards) — for ST (owner of `agent.py`)

RW's mechanism needs **no** agent change to work: `MetaLearner`'s API is unchanged, its new defaults
(`promotion_rule="validity"`, `estimator="auto"`, `utility="direct"`, level-change utilities) are active for every agent
built from the configuration, and the evidence is fed from `MetaLearner.internal_rewards` / `meta_update`, which the
agent already calls (steps 13 and 15). The two hooks below only make the new state *visible* to the operator and to
the restore report (amendment: "new parameters initialise fresh and are reported; nothing silently dropped").
Both were tested on a patched copy (`PYTHONPATH` pointing at it; see the end).

## RW-1 report state that an older checkpoint did not contain (restore)

What: after `self.meta.load_state_dict(aux["meta"])`, copy `MetaLearner.restored_fresh` (the meta-learner's parts
absent from the checkpoint and initialised fresh — for checkpoints written before RW: the prospective validity
evidence) into the restore report.

Why: CAM's live planning run keeps writing checkpoints with the pre-RW code; when it is restored by the new code,
the validity evidence starts empty. That must be reported, not silent.

```diff
--- a/cam_live/src/cam_live/agent.py
+++ b/cam_live/src/cam_live/agent.py
@@ def load(self, path: str) -> None:  (the line after "aux = (st.aux_state or {}).get("agent", {})")
         self.meta.load_state_dict(aux["meta"])
+        if getattr(self.meta, "restored_fresh", None):          # RW-1: parts absent from an older checkpoint
+            self.restore_report["meta_restored_fresh"] = list(self.meta.restored_fresh)
         self._load_meta_batches(aux.get("meta_batches", {}))
```

## RW-2 the learned reward's evidence in `status()` (dashboard / state.json)

What: one more key in `CamLiveAgent.status()`, next to `bootstrap_mix`.

Why: the operator sees `bootstrap_mix` but not *why* it is where it is (how many independent verified pairs scored,
the prospective agreement of each reward with the verified outcomes, the e-values of the current decision).

```diff
--- a/cam_live/src/cam_live/agent.py
+++ b/cam_live/src/cam_live/agent.py
@@ def status(self) -> Dict[str, Any]:
             "bootstrap_mix": self.meta.mixer.mix, "learner": self.learner.status(), "meta": dict(self.meta.stats),
+            "reward_validity": {k: v for k, v in self.meta.validity.summary().items() if k != "by_mix"}
+            if hasattr(self.meta, "validity") else None,
             "meta_events": len(self.meta.events), "ledger_entries": len(self.meta.ledger),
```

## Test on a patched copy

`tests/reward/test_reward_rw.py::test_checkpoint_without_validity_state_loads_and_reports_it_fresh` pins the
meta-learner side (`restored_fresh == ["validity (not in the checkpoint)"]`). The agent side was checked on a copy of
the tree (HEAD dc1753c + RW's `reward/`) with both hunks applied: `docs/evidence/RW/check_hooks.py` builds a debug
agent, saves a checkpoint without the validity state (as the pre-RW code wrote it), loads it into a fresh agent and
checks the restore report and `status()`; output `docs/evidence/RW/check_hooks.log`, unified diff
`docs/evidence/RW/rw_agent_hooks.diff`.
