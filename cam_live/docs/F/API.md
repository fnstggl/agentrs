# Workstream F — runtime: public API

All under `cam_live.runtime`. Tests: `cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/runtime -q --rootdir=cam_live`
(separated-uid test: add `-m privileged`, needs root).

## Loop
```python
loop.LiveRuntime(agent: AgentAPI, env: EnvironmentAdapter, policy: PolicyEngine, verifier: OutcomeVerifier,
                 task: TaskSpec, run_dir: str, clock: Clock, cfg=None,          # CamLiveConfig (section "runtime") | RuntimeConfig | None
                 *, wake: threading.Event | None = None, actor_id: str = "")
  .start() -> None                      # fresh start, or restore from <run_dir>/state.json + wal.jsonl
  .step() -> dict                       # one pass over the 17 steps (may sleep at step 17)
  .run(max_iterations=None, block=None) -> str   # running|paused|stopped|completed|failed|exhausted|idle
  .request_stop(reason="process signal"); .status_report() -> dict; .close()
  .hooks["after_cycle"]: list[Callable[[LiveRuntime], None]]
loop.RuntimeConfig(candidates_k=16, learn_budget_s=30, meta_budget_s=30, checkpoint_every_s=0, keep_checkpoints=5,
                   verify_every_s=0, evidence_poll_s=0, approval_poll_s=0, control_poll_s=1, min_idle_sleep_s=1,
                   min_cycle_interval_s=0, pending_retry_s=300, error_backoff_s=30, max_consecutive_errors=5,
                   journal_every=200, journal_min_interval_s=600, fsync=True, record_at_policy=False,
                   sleeper="auto", redaction={})
```
What the runtime expects of an agent beyond `contracts.AgentAPI` (all optional, duck-typed):
`drain_experiments() -> list[ExperimentRecord]` (new / updated experiment records; called after evaluate and after
reconcile), `explain_obstacle(ObstacleRecord) -> list[{"hypothesis","p"}]`. After `load()` the runtime may call
`record_result(decision, result)` for decisions journalled after that checkpoint, and re-delivers evidence read
after the committed cursor. A `wait` action (modality `internal`, operation ending in `wait`, argument `seconds`)
is slept by the runtime, never sent to an executor. `task.measurement.verify_interval_s` /
`evidence_poll_interval_s` override `verify_every_s` / `evidence_poll_s`.
In-flight recovery re-issues an action only if `env.get_capabilities()` has `features.at_most_once_by_action_id`
(or `idempotent_actions`); otherwise the result is reported as `cancelled / unknown` and nothing is re-executed.

Run directory (besides `contracts.RunFiles`): `task.json`, `task.yaml` (operator copy), `wal.jsonl`, `runtime.pid`,
`run.json`, `preflight.json`, `exports/`, `runtime_logs.json` (only if the JSONL logs had to move to `learner/`).

## Logs, journal
```python
logs.RunLog(run_dir, clock, *, run_id="", provenance="diagnostic", redactor=None, source="runtime", durable=True)
  .append(name, type_, data, *, consequential=False, trust="") -> rid
  .event/.observation/.decision/.predictions/.resolution/.experiment/.experiment_outcome/.learning_update/.meta_update/.obstacle
logs.Redactor(salt=b"", *, secret_values=(), secret_patterns=None, pii_patterns=None, member_keys=..., secret_keys=...,
              permission_keys=None, data_permissions=None);  Redactor.from_config(section, salt, data_permissions)
logs.RunReader(run_dir).records(name, types=None, tail=None) / .state() / .obstacles() / .path(name)
logs.atomic_write_json(path, obj, mode=0o644, durable=True); logs.iter_jsonl(path)
journal.generate_journal(run_dir, state=None) -> str; journal.write_journal(run_dir, state=None) -> path
journal.prediction_stats(resolutions) -> dict; journal.behavioural_change(update_dict) -> bool
```
Envelope of every log line: `{rid, seq, ts, type, run_id, provenance, source, data[, trust]}`; `seq` is global across files.

## Tasks
```python
tasks.load_task(path, *, for_run=True) -> TaskSpec        # raises tasks.TaskValidationError(.problems)
tasks.task_from_dict(raw, *, for_run=True); tasks.validate_task(raw, *, for_run=True) -> list[str]
tasks.strategy_findings(raw) -> list[str]; tasks.scan_file(path) -> list[str]; tasks.unset_paths(raw) -> list[str]
tasks.live_requirements(raw) -> list[{kind,item,detail}]; tasks.check_runnable(task) -> list[str]
tasks.autonomous_task(objective_id, limits, *, interfaces=None, mode="staging", family="autonomous") -> TaskSpec
```
Unset marker in YAML: `REQUIRED`. Extra keys used inside TaskSpec dicts: `resources.authorized_tools`,
`resources.publishing.{authorized,approval_required,destinations}`, `success.{verifier,metric,target,confidence,
exclusions,evidence_source}`, `measurement.{verify_interval_s,evidence_poll_interval_s,deadline_ts,window_start_ts}`.

## Verifier
```python
verifier.MemberRecord(id, is_bot=False, joined_at=0.0, pending=False, invite_code="", inviter_id="", left_at=None,
                      account_created_at=0.0, flags=[], link_key="")
verifier.MembershipSource (Protocol): name: str; live: bool; list_members() -> list[MemberRecord]
        optional: participation() -> dict[str,int]; feedback() -> list[{"member_id","consented"}]; resource_usage() -> dict[str,float]
verifier.MetricSource (Protocol): name; live; read() -> list[Quantity]
verifier.MembershipVerifier(source, clock, state_dir, *, log_path="", pinned_task=None, cost_source=None)   # OutcomeVerifier
verifier.MetricThresholdVerifier(source, clock, state_dir, *, log_path="", pinned_task=None)                # OutcomeVerifier
verifier.build_verifier(task, state_dir, clock, source, *, log_path="", pin=True, cost_source=None)
verifier.source_from_spec("jsonl:<path>" | "json:<path>" | "<module>:<factory>", task, run_dir)   # factory(task, run_dir)
verifier.HashChainLog.verify_chain(path, expected_head=None) -> (ok, n, problem)
verifier_service.RemoteVerifier(socket_path, timeout=120.0)      # verify / measure / check_signature / info
verifier_service.serve(run_dir, socket_path, task, source, *, clock=None, allowed_uids=None) -> RpcServer
python -m cam_live.runtime.verifier_service --run-dir R --socket R/sock/verify.sock [--task T] [--source SPEC]
```
A live Discord run needs a `"<module>:<factory>"` returning a `MembershipSource` with `live = True` for the
authorized guild (workstream ES); it goes into `success.evidence_source` of the task file.

## Control, scheduler, obstacles, multi-actor
```python
control.OperatorControl(run_dir, policy=None, by="operator", clock=None)
  .pause(reason="") .resume() .stop(reason="", emergency=False) .approve(id, note="") .reject(id, note="")
  .assign_objective(task_path) .export_checkpoint(dest="") .wake() .result(cmd_id) .wait_result(cmd_id, timeout_s=30)
control.ControlChannel(run_dir, clock=None).send(cmd, by="operator", wake=True, **args) -> command id
scheduler.PendingOutcomes  .add(query, *, action_id="", decision_id="", experiment_id="") .due(now) .resolve(obs) .next_ready()
scheduler.make_sleeper(clock, wake=None, poll_s=1.0) -> SimSleeper | WallSleeper
obstacles.ObstacleTracker(log, clock)  .open(action, result, agent=None) .on_decision(d) .on_result(a, r) .on_learning(updates)
multi.SharedLearner(learner: CentralLearner, clock=None, learn_budget_s=10.0)  .submit(actor_id, records) .step() .pull(have_version) .stats()
multi.CentralLearner (Protocol): add_experience(actor_id, records) -> int; learn(budget_s) -> list[LearningUpdateRecord]; weights() -> (weights, version)
multi.ActorAgent (Protocol, on top of AgentAPI): export_experience() -> list; weights_version() -> int; set_weights(weights, version)
multi.MultiActorRuntime(actors: list[multi.Actor(actor_id, runtime)], learner)  .step() .run(max_rounds=1000) .status()
multi.serve_learner(shared, socket_path, allowed_uids=None); multi.RemoteSharedLearner(socket_path)
```

## Dashboard, doctor, preflight, CLI, learner process
```python
dashboard.create_app(run_dir, token, policy=None, tasks_dir=None, operator="operator") -> FastAPI
dashboard.serve(run_dir, token, host="127.0.0.1", port=8787, policy=None, tasks_dir=None)   # loopback only
dashboard.load_or_create_token(directory) -> str
doctor.run_doctor(cfg=None, config_path="") -> {"sections", "summary"}; doctor.render(report) -> str
preflight.preflight(task_path, cfg=None, *, policy=None, env=None, verifier_info=None, with_doctor=True) -> report
cli.main(argv=None) -> int            # cam-live; `python -m cam_live.runtime.cli`
loop_service.run_service(run_dir, task, cfg, *, agent_factory, inproc, max_iterations=None, block=None) -> str
python -m cam_live.runtime.loop_service --run-dir R [--config C] [--task T] [--agent-factory M:F] [--inproc] [--wait-for FILE]
```
Agent factory: `cam_live.agent:build_agent`, called with those of `cfg, run_dir, clock, task` it accepts.
Config section `runtime` also reads: `agent_factory`, `executors` (list of `module:factory` for in-process
environments / inspect-harness), `browser`, `inproc_factory` (`module:factory(task, cfg, run_dir) -> (env, policy,
verifier, clock)`).

## Changes with the G-rt fixes (2026-11-27; `docs/F/FIXES.md`)

* `runtime.tasks`: `learner_view(task) -> TaskSpec` (what `agent.set_objective` receives), `limit_problems(task) ->
  {field: reason}`, `objective_flags(text)`, `objective_confirmation(text) -> "sha256:<hex>"`, `has_discord(task_or_raw)`,
  `evidence_source(task_or_raw)`, `DEFAULT_DISCORD_EVIDENCE_SOURCE`, `schema_problems(raw)`, `load_task_text(text, path)`;
  task files follow a closed schema (`SCHEMA`), anchors / aliases / duplicate keys are refused.
* `runtime.control`: `OperatorKey(run_dir)` (`.ensure()`, `.sign(cmd)`), `trusted_public_key(run_dir)`,
  `ControlChannel.authenticate(cmd) -> (signed, why)`, `ControlChannel.send(..., signer=None)`, `PRIVILEGED`;
  `OperatorControl` signs every command, `OperatorControl.reset_stop()`, `resume(wake=True)`.
* `runtime.private`: `PrivateBox(run_dir).seal(obj) / .open(text)` (AES-256-GCM, key `<run_dir>/learner/runtime.key`).
* `runtime.logs`: `Redactor(..., keep_ids=())`, `.set_keep_ids(ids)`, `Redactor.from_config(..., keep_ids=())`;
  `iter_since(path, seq)` (records with seq >= seq, read backwards).
* `runtime.scheduler`: `PendingItem.origin` ("environment" | "learner"), `PendingOutcomes.add(..., origin=)`, `.has(id)`,
  `.count_origin(origin)`.
* `runtime.loop.LiveRuntime`: `state.json` version 2 = redacted fields + `sealed`; journal lines
  `{"g", "t", "kind", "action_id", "x"}` with `t` in begin (`kind` decided / execute / wait) / end / outcome; new events
  `beliefs_updated`, `prediction_fields_cleared`, `learner_queries_capped`, `approval_not_restorable`; `run_resumed`
  carries `decisions_closed_unhandled`, `replayed_outcomes`, `redelivered_outcomes`; outcome records carry
  `query_origin`; experiment records carry `learner_reported_actual` and `assessed_by`.
* `runtime.verifier`: `SignedVerifier.check_measurement(obs)`, measurements signed; `norm_id`;
  `runtime.verifier_service`: `check_measurement` over VerifyRPC (`RemoteVerifier.check_measurement`).
* `runtime.preflight`: `preflight(..., run_dir="")`, `missing_integrations(task, tools)`, `discord_items(task, raw,
  policy, env, run_dir)`, `DISCORD_CHECK_TOOLS`.
* `runtime.dashboard`: `create_app(run_dir, token, policy=None, tasks_dir=None, operator="operator",
  operator_token=None)`, `serve(..., operator_token=None)`, `load_or_create_operator_token(directory)`; routes
  `/app.js`, `/app.css`, `/api/operator`, `POST /api/control/reset-stop`; headers `X-CAM-Operator`, `X-CSRF-Token`;
  cookie `cam_live_csrf`.
* `runtime.cli`: `load_hook(spec, what)`, `ComponentMissing`, `ComponentBroken`, `AGENT_HOOK`, `VALIDATION_HOOK`,
  `TRANSFER_HOOK`, `COMPLETION_HOOK`; `cam-live dashboard --view-only`; `cam-live verify-completion <checker options>`;
  `cam-live evaluate-transfer --checkpoint P <transfer options>`; `run.json` records `authorization` and
  `verifier_source`.
