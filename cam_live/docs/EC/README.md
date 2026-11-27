# Workstream EC — sandbox, background jobs, visual computer use, grounding models

Files: `src/cam_live/harness/{sandbox.py, jobs.py, visual.py, grounding/}`; tests in `tests/harness/compute/`.
Both executors implement `contracts.ModalityExecutor` and take everything else from `contracts.ExecContext`.

## Wiring into the computer environment

```python
from cam_live.harness.sandbox import SandboxExecutor, SandboxConfig
from cam_live.harness.visual import VisualExecutor, VisualConfig

code = SandboxExecutor(workspace=f"{run_dir}/sandbox/ws", cfg=SandboxConfig())      # modality "code"
visual = VisualExecutor(screen_factory=lambda name, action, ctx:                     # modality "visual"
                        browser.screen(session=name, ctx=ctx, action=action))
```

* The workspace directory must lie below directories the sandbox uid can traverse (`o+x`); otherwise the
  executor reports itself unavailable with the reason. `SandboxExecutor.capabilities()` says which isolation
  mode is active and what is and is not kernel-enforced; `tools()` marks every tool `available=False` with the
  reason when the host cannot isolate.
* Background jobs: `code.jobs(ctx)` returns the `JobManager` bound to `ctx.run_dir` (records under
  `<run_dir>/jobs/<job_id>/record.json`, events in `<run_dir>/jobs/events.jsonl` and to `ctx.obs_sink`).
  `JobManager.events(cursor)` lets the environment merge finished jobs into `poll_events`. Opening a manager on a
  run directory after a crash reports the jobs that were live as `interrupted`.
* Visual actions always name the observation they were decided on: take `visual.screenshot`, use the returned
  `obs_id` as `observation_id`; every action returns the next observation. `VisualExecutor.image(obs_id)` gives the
  PNG bytes to perception code in the same process; `Observation.refs["image"]` is the file under
  `<run_dir>/blobs/visual/`.
* Grounders are created on first use by `grounding.load_grounder(name)` (a separate process with a clean
  environment per model). `VisualConfig.grounder_order` is the engineering default (`goclick-base`, then
  `mai-ui-2b`); the modality selector may pass `"grounder"` in the action arguments instead.

## Supplying a different screen or grounder

A screen is any object with `screenshot() -> PNG bytes`, `size()`, `click(x, y)`, `double_click(x, y)`,
`drag(x1, y1, x2, y2)`, `scroll(dx, dy)`, `type(text)`, `key(combo)` (`visual.ScreenProvider`); optional `url()`
and `screenshot_full_page()` are used when present. A grounder is any `contracts.VisualGrounder`; pass it in
`VisualExecutor(grounders={"name": obj})`. Its answer is read as `x`, `y`, `confidence` and nothing else.

## Evidence

* `docs/evidence/EC/grounding_bench.json` (+ `_rows.json`): `python -m cam_live.harness.grounding.bench`
* `docs/EC/MODEL_REVIEW.md`: licences, pinned revisions, what was read, how each model is loaded.
* `docs/evidence/test_results/EC*.xml`.
