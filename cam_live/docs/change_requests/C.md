# Change requests — Workstream C

## C-1 `contracts.OutcomeObservation`: verifier measurements cannot be authenticated

**What.** `OutcomeVerifier.measure()` returns `OutcomeObservation`s with `verified=True`, but the record has no
signature and the verifier protocol has no check for it. `verified` is a plain bool the learner process could set.
**Why.** Spec 7.8 / architecture section 1: only signed outcomes may supervise the outer objective.
**Proposed diff.**
```diff
 class OutcomeObservation(Record):
     ...
     dedup_key: str = ""              # two observations with the same key are the same event
+    signature: str = ""              # verifier signature over the record with signature="" (measure() only)

 class OutcomeVerifier(Protocol):
     def check_signature(self, outcome: VerifiedOutcome) -> bool: ...
+    def check_measurement(self, obs: OutcomeObservation) -> bool: ...
```
**Workaround in place.** `reward.integrity.OutcomeLedger(check_signature, check_measurement=None)`: without an
injected `check_measurement` every measurement is rejected (`no_measurement_authentication`); objective outcomes
(`VerifiedOutcome`) work today.

## C-2 `contracts.TaskSpec`: no origin of the objective

**What / why.** Self-generated goals must never supervise the outer objective (spec 7.8, 11.4). Nothing on a
`TaskSpec` or `VerifiedOutcome` says whether the objective was assigned by the operator or chosen by CAM.
**Proposed diff.**
```diff
 class TaskSpec(Record):
     ...
     mode: str = "staging"            # "diagnostic" | "staging" | "live"
+    origin: str = Trust.OPERATOR     # Trust.OPERATOR | Trust.SELF (goal-free mode); set by the runtime only
```
**Workaround.** `OutcomeLedger.admit_verified(..., objective_origin=...)` takes it as an argument; the caller
must pass `Trust.SELF` for goal-free goals.

## C-3 `contracts.MetaUpdateRecord`: no place for the ledger head

**What / why.** Tamper evidence needs the hash-chain head of the outcome ledger at the time of each meta-update;
`diagnostics` is `Dict[str, float]` and cannot hold it.
**Proposed diff.**
```diff
 class MetaUpdateRecord(Record):
     ...
     skipped_reason: str = ""
+    ledger_head: str = ""            # reward.integrity.OutcomeLedger.head when the update was computed
+    window_pair: str = ""            # reward.windows.WindowPair.pair_id
```
**Workaround.** Stage 2 can write `query_window = f"{window_id}@{ledger.head}"`.

## C-4 `contracts.ExperienceRecord`: reward inputs are not persisted

**What / why.** Recomputing R_psi for a stored decision (stale `reward_version`, meta windows) needs the signal
vector and the lifetime step it was computed at; the record has no field for them.
**Proposed diff.**
```diff
 class ExperienceRecord(Record):
     ...
     bootstrap_reward: Optional[float] = None
+    reward_inputs: Dict[str, Any] = field(default_factory=dict)   # RewardSignals.to_dict() + {"life_step": int}
```
**Workaround.** `decision.diagnostics["reward_signals"]` (it is `Dict[str, Any]`).

## C-5 `contracts.ExperienceStore`: no way to list the records of a lifetime in order (stage 2)

**What / why.** `MetaLearner` needs the decisions taken after a policy update, in order. The protocol has `get`,
`sample` and `window` only (`MemoryStore.all()` and `SqliteExperienceStore.iter_records()` exist but are not in the
contract).
**Proposed diff.**
```diff
 class ExperienceStore(Protocol):
     def window(self, window_id: str) -> List[ExperienceRecord]: ...
+    def iter_records(self, *, since_seq: int = 0, objective_id: str = "") -> Iterator[ExperienceRecord]: ...
```
**Workaround in place.** `MetaLearner.observe()` / `internal_rewards()` register every decision once and the
meta-learner keeps the ordered ids itself (they are part of `state_dict()`).

## C-6 `nn.api.LatentModelAPI`: the decision trunk is not addressable as a module (stage 2)

**What / why.** The differentiable inner update is taken over the policy head *and the decision trunk* through
`torch.func.functional_call`, which needs the trunk module and its named parameters. The API only has
`decision_latent(b, e_a, e_g)`.
**Proposed diff.**
```diff
 class LatentModelAPI(nn.Module, abc.ABC):
+    @abc.abstractmethod
+    def decision_trunk(self) -> nn.Module:
+        """The module behind ``decision_latent`` (forward(b, e_a, e_g) -> z), for functional inner updates."""
```
**Workaround in place.** `reward.inner.AdaptableSubset` reads `model.exp.trunk` (Workstream A's attribute) and
falls back to adapting the policy head alone when it is absent.

## C-7 `contracts.ExperienceRecord`: no lifetime index (stage 2)

**What / why.** Intrinsic returns and the lifetime context of R_psi run along one lifetime and must not cross into
another (multi-lifetime diagnostics; a restarted live run). Nothing on a record says which lifetime it belongs to.
**Proposed diff.**
```diff
 class ExperienceRecord(Record):
     seq: int = 0                     # monotone position in the lifetime
+    lifetime: int = 0                # index of the lifetime (0 for the single live lifetime)
```
**Workaround in place.** `MetaLearner.new_lifetime()` + its own `exp_id -> lifetime` map.

## C-8 Integration: who applies the policy-improvement update (to master / Workstream D, stage 2)

**What / why.** The meta-gradient is exact for the update `theta <- theta - alpha * grad L_inner` (plain SGD,
per-group clip in the graph). If the continual learner applies `PolicyImprovementLoss` with AdamW together with
other losses, the live update is not the differentiated one; the meta-update is then a *shadow* update evaluated by
importance weighting (supported, but weaker evidence).
**Proposed integration.** Either the agent calls `MetaLearner.policy_update(records)` for the policy head + trunk
on the frequent schedule and excludes those parameters from the learner's policy-improvement step, or the learner
calls `MetaLearner.note_policy_update(records)` immediately before it steps. No contract change is needed.
**Workaround in place.** Both entry points exist and are tested.

## C-9 Engine fix inside this workstream (information only, stage 2)

`reward.meta.policy_gradient_inner_loss` used `v.detach()` as the advantage baseline. For K >= 2 this removes the
path theta_k(psi) -> V -> advantage from the unroll (bias measured on the real model: 1.0 % at K = 2, 3.3 % at K = 4, docs/evidence/C/diagnostics.json;
stage-1 finite-difference tests did not combine a value function with K >= 2). Fixed with `theta_sg` (see
docs/C/API.md); regression test `test_value_baseline_must_not_be_plainly_detached_for_K_above_one`.

## C-10 Integration (`agent.py`, M-int): register the task's origin with the meta-learner (G-pr D3)

**What / why.** After the G-pr fixes the outcome ledger admits a signed objective outcome only if the objective's
*recorded* origin is the operator (or the verifier); an objective that was never registered is refused
(`objective_origin_unknown`, fail closed) and the caller's `objective_origin` argument can only exclude. The agent
currently passes `objective_origin=self.task.origin` to `admit_objective` but never registers the task, so every
objective outcome it offers is now refused for the meta ledger (the evaluator, the store's objective record and the
predictor are not affected).
**Proposed diff** (wherever the runtime assigns a task to the agent, e.g. `set_task` / the start of a lifetime, and
after `load()` for the task in force):
```diff
         self.task = task
+        self.meta.register_objective(task)          # TaskSpec.origin decides whether its outcome may supervise psi
```
and in `_verified` (optional, equivalent):
```diff
-        adm = self.meta.admit_objective(v, objective_origin=origin)
+        adm = self.meta.admit_objective(v, objective_origin=origin, task=self.task)
```
(`task=` raises `ValueError` if `task.objective_id != v.objective_id`; pass it only for the current task.)
`MetaLearner.state_dict()` now carries the recorded origins (`"objective_origins"`), so a restart keeps them.
**Workaround.** None inside C: refusing unregistered objectives is the fix.

## C-11 Integration (`agent.py`, M-int): give back the budget of goal-free goals that are not executed (G-pr D4)

**What / why.** A goal selected by `GoalFreeMode.propose` now *reserves* one decision and its estimated cost until
`resolve` (or the new `GoalFreeMode.abandon`) is called, so pending goals count against the goal-free budget. In
`record_result` the agent drops the proposed goals that were not chosen
(`self.active_goals = {k: v ... if k == a.action_id}`) without resolving them: their reservations stay held and,
after about `goal_free_decisions / goal_free_k` decisions, goal-free mode selects nothing (fail closed).
**Proposed diff** (`agent.py`, `record_result`):
```diff
-        self.active_goals = {k: v for k, v in self.active_goals.items() if k == a.action_id}
+        for k, v in list(self.active_goals.items()):
+            if k != a.action_id:
+                self.goal_free.abandon(v)            # not executed: release its reservation, nothing charged
+                del self.active_goals[k]
```
Also abandon the active goals when an action is refused before execution, and after `load()` for goals that were
pending at the checkpoint and will not be resolved (`GoalFreeMode.abandon(goal)`; reservations are checkpointed in
`SelfGoalSelector.state_dict()` as `reserved` / `n_reserved`).

## C-12 `contracts.VerifiedOutcome` / verifiers: sign the objective's origin (G-pr D3, defence in depth)

**What / why.** The recorded origin (C-10) is registered by the learner process from the `TaskSpec` the runtime
gave it. The verifier knows which task it was asked to verify and signs the outcome; if it also signed the task's
origin, a self-generated goal could not be passed off as an operator objective even by code inside the learner.
**Proposed diff.**
```diff
 class VerifiedOutcome(Record):
     ...
     signature: str = ""
+    origin: str = ""                 # TaskSpec.origin of the verified task, covered by the signature ("" = not stated)
```
`runtime.verifier.SignedVerifier.verify(task)` / `diagnostics.world.LabVerifier.verify(task)`: set
`origin=task.origin` before signing. `OutcomeLedger.admit_verified` would then refuse when a stated `origin` is
not OPERATOR / VERIFIER (in addition to the recorded origin). Not needed for the fix; no workaround required.
