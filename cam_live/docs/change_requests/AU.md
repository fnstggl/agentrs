# Change requests of workstream AU (operational autonomy, amendment 2026-11-28)

## AU-1 — `harness/jobs.py`: a background job's primitive carries the hosts it declared (EC-owned file)

**What.** `sandbox.job_start` with `network='scoped'` is refused by the real authorization service before the job
starts: the job-level primitive that `execute_job_tool` sends to `ExecContext.check` lists `network` but not the
`hosts` the action declared, and the policy denies network for generated code that names no hosts
(`P-CODE-NETWORK`, "network access requested without naming the hosts").

**Why.** Background jobs are the bounded way to run long installs, builds and tests (spec 12A.11). AU made
`hosts` an argument of `sandbox.python` / `sandbox.shell` / `sandbox.job_start` (`harness/sandbox.py` schema), put
the hosts on the primitive of a foreground run, added the policy's per-connection rule (`T-CODE-CONNECT`,
`declared_hosts`) and keeps a job's grant open for its connections within the grant's execution window
(`PolicyService.record`). The only missing piece is this one line in `jobs.py`.

**Exact diff** (`cam_live/src/cam_live/harness/jobs.py`, function `execute_job_tool`):

```diff
     op = c.PrimitiveOp(op=name, modality=executor.modality, started=t0,
-                       arguments={k: a[k] for k in ("job_id", "purpose", "kind", "deadline_s", "network") if k in a})
+                       arguments={k: a[k] for k in ("job_id", "purpose", "kind", "deadline_s", "network", "hosts")
+                                  if k in a})
```

**Tested on a patched copy** (PYTHONPATH pointing at a copy of `src` with only this change), against the real
`PolicyService` on a live task with a standing record and a local package index:
`docs/evidence/AU/job_probe.py`, output `docs/evidence/AU/job_probe_output.txt`:

* unpatched: `job_start denied network access requested without naming the hosts`, the index received nothing;
* patched: `job_start ok`, the job `succeeded` with exit code 0, stdout `200`, the index received
  `/simple/autinypkg/`.

Until it is applied: foreground `sandbox.python` / `sandbox.shell` runs with `network='scoped'` work (privileged
tests `tests/harness/autonomy/test_au_sandbox_packages.py`); background jobs with network stay refused (fail-closed).
Limitation that remains after it is applied: a job's connections are allowed only within the grant's execution
window (`grants.exec_window_s` 900 s + slack); a longer job loses network after that.

## No change request to `agent.py`

AU needs no agent-side hook: the standing profile, retained approvals, scoped credentials and package installation
act in the authorization service, the sandbox and the operator tools. The learner sees the new `hosts` argument
and the two new request kinds (`account`, `verification`) through `discover_tools()` like every other tool
description.
