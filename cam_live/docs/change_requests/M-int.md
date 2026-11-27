# Change requests from the integration engineer (M-int)

Each has a workaround in M-int's own files; none blocks the integrated agent's tests.

## M-int-1 (runtime, `runtime/loop_service.py`): pass the verifier to the agent factory

**What / why.** `CamLiveAgent` checks the verifier's signature itself before a `VerifiedOutcome` may supervise the
evaluator, the store's objective record or the meta-learner (it does not rely on the runtime having checked it).
`run_service` builds the verifier but calls the factory with `cfg, run_dir, clock, task` only, so an agent started
by `cam-live run` has no verifier to check against and refuses every verified outcome for supervision (fail closed,
reported in `reconcile`'s result and `status()["verifier_attached"]`).

**Proposed diff.**
```diff
-    agent = call_factory(load_factory(agent_factory), cfg=cfg, run_dir=run_dir, clock=clock, task=task)
+    agent = call_factory(load_factory(agent_factory), cfg=cfg, run_dir=run_dir, clock=clock, task=task,
+                         verifier=verifier)
```
`call_factory` only passes the keywords a factory accepts, so `rt_lab_agent.build_agent` is unaffected.

**Workaround.** `CamLiveAgent.attach_verifier(verifier)`; the integration tests and `cam_live.validation` call it.

## M-int-2 (runtime, `runtime/cli.py`): options of `cam-live validate --mode staging`

**What / why.** `cmd_validate` calls `run_staging(cfg)`. The staging validation needs a choice of the real model
(GPU, default) or the tiny CPU configuration, the number of decisions and where the evidence goes.
`cam_live.validation.run_staging(cfg, *, out_dir=None, real_model=True, n_decisions=..., seed=0)` defaults
`out_dir` to `<CAM_LIVE_HOME>/validation`, so the current call works; the options are not reachable from the CLI.

**Proposed diff** (argument parser of `validate` and `cmd_validate`):
```diff
+    p.add_argument("--cpu", action="store_true", help="tiny backbone + debug core on the CPU (plumbing check)")
+    p.add_argument("--decisions", type=int, default=0)
+    p.add_argument("--out", default="")
 ...
-    report = run_staging(cfg)
+    kw = {"real_model": not a.cpu, "out_dir": a.out or None}
+    if a.decisions:
+        kw["n_decisions"] = a.decisions
+    report = run_staging(cfg, **kw)
```
`run_staging` returns a report with `"ok"` (and `"passed"`) so the exit code logic stays as it is.

## M-int-3 (runtime, `runtime/verifier.py`): sign verifier measurements (contract 1.1, C-1 / F-5)

**What / why.** `OutcomeObservation.signature` and `OutcomeVerifier.check_measurement` now exist in the contract.
`SignedVerifier._measure` returns unsigned observations and has no `check_measurement`, so verifier measurements
of a real run can never supervise the meta-objective (only signed objective outcomes can).

**Proposed diff** (`SignedVerifier`):
```diff
+    def _sign_obs(self, o: OutcomeObservation) -> str:
+        d = o.to_dict(); d.pop("signature", None)
+        return hmac.new(self._key, _canon(d), hashlib.sha256).hexdigest()
+
+    def check_measurement(self, obs: OutcomeObservation) -> bool:
+        return bool(obs.signature) and obs.verified and hmac.compare_digest(obs.signature, self._sign_obs(obs))
 ...
             out.append(OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(vals[q.metric]), ...))
+            out[-1].signature = self._sign_obs(out[-1])
```
and `check_measurement` in `verifier_service.RemoteVerifier` / the service's method table.

**Workaround.** The agent calls `check_measurement` through `getattr`; without it measurements are kept for the
world model and the evaluator but never admitted to the meta ledger. `diagnostics.world.LabVerifier` signs its
measurements already.

## M-int-4 (neural architecture, `nn/model.py`): override `LatentModelAPI.decision_trunk()` (C-6)

**What.** `return self.exp.trunk` in `CamLiveModel`. The base method (contract 1.1) returns `None`;
`reward.inner.AdaptableSubset` still reads `model.exp.trunk` directly, so nothing breaks meanwhile.

## M-int-5 (authorization, `harness/auth/vault.py`): `ExecutorSecrets.has(name)` (ES-2)

**What.** The contract's `SecretHandle` now documents an optional `has(name) -> bool` (whether a non-revoked
secret of that name exists, releasing nothing). The diff is the one in `docs/change_requests/ES.md` (ES-2).

## M-int-6 (runtime, `runtime/loop.py`): do not consume the grant before `env.execute`

**What / why.** `LiveRuntime` calls `policy.check_grant(action, auth)` (which *consumes* the grant of
`harness.auth.PolicyService`, EA-2) before `env.execute(action, auth)`; `harness.env.ComputerEnvironment` then checks
the same grant again and refuses the action: measured in the staging validation, every browser / API action came back
`denied / missing_permission: the grant does not verify for this action`. LabWorld does not check grants, which is why
the LabWorld tests do not see it. The same holds for `_recover_inflight` and the approval path.

**Proposed diff** (every runtime-side call):
```diff
-        if auth.verdict != AuthVerdict.ALLOW or not self.policy.check_grant(action, auth):
+        if auth.verdict != AuthVerdict.ALLOW or not _look(self.policy, action, auth):
```
with
```python
def _look(policy, action, auth) -> bool:
    """Verify a grant without using it up: the executor consumes it immediately before execution."""
    try:
        return bool(policy.check_grant(action, auth, consume=False))
    except TypeError:                       # a PolicyEngine without the keyword (single-use not implemented)
        return bool(policy.check_grant(action, auth))
```

**Workaround.** `cam_live.validation.LookOnlyGrants` (the runtime is handed a view of the service whose
`check_grant` only looks; the environment holds the real service).
