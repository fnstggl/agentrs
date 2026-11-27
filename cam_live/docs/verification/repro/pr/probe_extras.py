"""Smaller claims: action quality vs luck (SE-05), the goal-free budget (AE-04), the new contract flag
OutcomeObservation.imagined in C's ledger screen (MR-09, PI-10).
"""
import dataclasses
import sys

import torch

import _common as C
from cam_live.contracts import Cost, OutcomeObservation, OutcomeStatus
from cam_live.predict.evaluator import action_quality, outcome_decomposition
from cam_live.reward.integrity import OutcomeLedger
from cam_live.reward.selfgoals import GoalFreeBudget, LearningGoalCandidate, SelfGoalSelector

# ------------------------------------------------------------------ SE-05: quality + luck = surprise; lucky success
q = torch.tensor([[0.2, 0.6, 0.4]], dtype=torch.float64)
p = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
aq = action_quality(q, p, torch.tensor([0]))
base = float((p * q).sum())
C.check("action quality = q[chosen] - E_pi q, regret = max q - q[chosen]",
        abs(float(aq["quality"]) - (0.2 - base)) < 1e-15 and abs(float(aq["regret"]) - 0.4) < 1e-15)
dec = outcome_decomposition(torch.tensor([1.0], dtype=torch.float64), aq["q_chosen"], aq["baseline"])
C.check("a lucky success of a poor decision: quality < 0, luck > 0, quality + luck = surprise",
        float(dec["quality"]) < 0 < float(dec["luck"]) and abs(float(dec["quality"] + dec["luck"] - dec["surprise"])) < 1e-15)

# ------------------------------------------------------------------ AE-04: the goal-free budget
cands = [LearningGoalCandidate(key=f"k{i}", description=f"learn about channel {chr(97 + i)}", epistemic=1.0,
                               achievable_progress=0.5, est_cost=Cost(seconds=10.0)) for i in range(5)]
sel = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=2))
got = sel.select(cands, k=5)
C.check("one select(k=5) call never returns more goals than the decision budget (2)", len(got) <= 2,
        f"returned {len(got)} goals (can_afford checks each candidate alone, nothing is reserved)")
sel2 = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=2))
n = 0
for _ in range(6):                                   # goals still pending: outcomes not yet observed
    n += len(sel2.select(cands, k=1))
C.check("repeated selection while earlier goals are pending stays within the decision budget (2)", n <= 2,
        f"{n} goals selected; the budget is charged only in record_result / resolve")

# ------------------------------------------------------------------ the imagined flag of the contract
ok = OutcomeObservation("q1", OutcomeStatus.OBSERVED, 5.0, observed_at=10.0, verified=True, source="labworld",
                        dedup_key="d1")
if hasattr(ok, "imagined"):
    im = dataclasses.replace(ok, dedup_key="d2")
    im.imagined = True
    L = OutcomeLedger(lambda v: False, check_measurement=lambda o: True)
    r = L.admit_measurement(im)
    C.check("an OutcomeObservation flagged imagined=True is refused by the outcome ledger", not r.accepted,
            f"accepted={r.accepted} (screen_observation reads only its imagined= argument)")
else:
    C.info("contracts.OutcomeObservation", "has no imagined field in this source tree")
sys.exit(C.finish())
