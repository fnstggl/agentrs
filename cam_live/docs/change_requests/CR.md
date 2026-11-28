# Change requests of CR (retrospective credit assignment, amendment item 4)

Owner of agent.py: ST (integrator). CR's code is in `src/cam_live/credit/**`; nothing below is applied by CR.

## CR-1 (src/cam_live/agent.py): the learner's blocked split by credit segment

**What.** One line in `CamLiveAgent.__init__` (learner construction): the default learner split of the agent becomes
`"segment"` instead of `"record"`.

```diff
         lsec = cfg.section("credit").get("learner", {})
-        lover = {} if "val_split" in lsec else {"val_split": "record"}   # a single live objective must train
+        # a single live objective must train, and its held-out blocks validate the credit estimators (CR-1):
+        # every 1/val_fraction-th segment of credit.segments' top-level decisions is held out as a whole
+        lover = {} if "val_split" in lsec else {"val_split": "segment"}
```

**Why.** With `"record"` every 7th record is held out, so every credit segment of the live objective contains
records the learner trains on: no randomised decision inside the objective is ever a held-out unit and the
estimator selection cannot run inside one long objective (the agent's evaluation log says so at every update).
With `"segment"` (credit/learner.py `_segment_held_out`, credit/segments.py) the held-out fraction is unchanged
(`val_fraction`, 0.15) but held out in blocks of `CreditAssigner.units.decisions` (16) top-level decisions, so the
randomised decisions of a held-out block are units whose own outcomes no learner and no estimator trained on
(blocked cross-validation for temporally dependent data). Those units feed the pooled, estimand-matched selection
rule (credit/select.py `pooled_unit_score`, `EstimatorSelector._select_pooled`) without any other agent change.

**Compatibility (live run).** Records stored before the learner first uses the segment split keep the record split
(`ContinualLearner.segment_split_from_seq`, persisted in `learner.state_dict()`), and their segments count as
trained on (`trained_units`), so nothing the live run trained on can become held-out evidence. A checkpoint written
before CR loads (no `trained_units` / `segment_split_from_seq`: both derived as above;
`tests/credit/test_cr.py::test_switching_to_the_segment_split_keeps_older_records_on_the_record_split`).

**Cost.** Short lifetimes (eval runs of 16-20 decisions) lose INT3A-5's guarantee "an objective with >= 7 records
has a held-out record": with blocks of 16 decisions an objective needs >= 7 blocks for that guarantee; across
lifetimes the held-out fraction stays 0.15. A run that needs the old behaviour sets
`credit.learner.val_split: record` in its configuration (the line above keeps a configured value).

**Tested** on a copy of the working tree with the line applied through the configuration
(`credit.learner.val_split: segment`, which the agent already honours): `credit/cr_agent_eval.py` runs (docs/CR.md).

## CR-2 (optional, src/cam_live/agent.py): segment length from the configuration

```diff
         self.assigner = CreditAssigner(self.model, self.store, rudder=self.rudder, cocoa=self.cocoa,
-                                       selector=self.selector, beta=s.credit_beta, clock=self.clock)
+                                       selector=self.selector, beta=s.credit_beta, clock=self.clock,
+                                       segment_decisions=int(cfg.section("credit").get("segment_decisions", 16)))
```

**Why.** `segment_decisions` bounds the horizon of learned credit inside an open objective (outcomes measured by
queries of later segments are not part of an earlier segment's return until the objective closes). The default 16
is used without this line; the line only makes it configurable per task.
