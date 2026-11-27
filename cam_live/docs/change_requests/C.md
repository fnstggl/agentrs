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
