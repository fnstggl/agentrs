# Change requests — Workstream C (stage 1)

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
