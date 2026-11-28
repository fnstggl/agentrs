# INT3B — final runtime-side integration (write-up)

Scope: `src/cam_live/runtime/**`, `src/cam_live/validation.py`, `tasks/staging/`, `tools/planning_rehearsal.py`,
`tests/runtime/**`, `tests/integration/**` (not `test_int3a_*`), the pytest configuration. Commits `cam live [INT3B]`.
Nothing in this work writes a niche, strategy, plan, example, hint or growth tactic anywhere; the staging web serves
neutral generic pages; CAM's own texts are counted and, where quoted, quoted verbatim.

## 1. Outcomes pending when an objective ends (INTEGRATION.md §8)

**Defect (measured before the fix).** The drain (S-1) covered only an exhausted decision budget. When an objective
ended by a signed verdict (achieved / failed) or by another limit, the runtime finished at once and its pending
outcome queue was never measured: LabWorld, target 3, 2-day attribution windows, seeds 3-8 — 2 to 7 outcomes of the
objective's own decisions left pending at every verified success (`tests/runtime/test_int3b_settle.py` on the code of
HEAD b663d5e: `assert 3 == 0` pending, no outcome after the success).

**Fix (`runtime/loop.py`).** An ended objective is *settled*: `RuntimeConfig.settle_after_finish_s` (default 7 days,
never past the task deadline) — the outcomes that can mature in that window are measured, reconciled and learned
from (steps 2-5, 13-15), nothing is proposed or executed, approvals are not processed; then the objective is closed
with the status it ended with (an exhausted objective that the verifier signs as achieved while settling closes as
completed). Outcomes that cannot mature in the window, and everything pending at an operator stop, are *carried*:
they stay in the committed, persisted pending-outcome queue (`state.json`), and `objective_finished` / `halted`
record `pending_outcomes_carried`. The settle state is in the sealed state and survives a restart. `0` turns it off.
Events: `settling`, `objective_finished.settle {measured, cycles, ...}`.

Tests (`tests/runtime/test_int3b_settle.py`, 6; all fail on the earlier code): outcomes pending at a verified success
are measured (by the verifier, signed) before the objective closes, reach `agent.reconcile`, resolve their
predictions, with no decision after the verdict; the window is bounded and the rest is carried in `state.json`;
never past the deadline; another limit (external actions) settles too; a stop carries; settling resumes after a
restart. Test-infrastructure fix: `rt_helpers.LabHost` now exposes `v_check_measurement` (without it every signed
measurement over RPC was an error, which a settling run turned into a pause).

## 2. Shared GPU, ephemeral storage, debug validation (master items C, A, B)

**C — the GPU lease (`runtime/gpu_cycle.py`, `runtime/loop_service.py`).** With a pretrained backbone on CUDA
(`runtime.gpu_lock: auto`, now also written in `configs/cam_live.yaml` with `gpu_hold_s: 840`, `gpu_yield_s: 15`)
the learner process leases the machine's shared lock (`config.GPU_LOCK_PATH`, the file `nn.gpu.gpu_lock` uses): taken
before the agent is built; before every cycle with work (a new `before_cycle` hook of `LiveRuntime`) it is re-taken
if not held, and if the next cycle would exceed the hold (held time + the last cycle's duration > `gpu_hold_s`) the
runtime commits its checkpoint, releases, yields `gpu_yield_s`, and waits for the lock again — same process, no exit,
no manual resume; the model stays resident in GPU memory (its cache is emptied). The runtime's sleeper releases the
lock before every sleep longer than 5 s (waiting for outcomes, paused, finished). Waiting is a *queued* blocking
`flock` (a helper thread), so a tenant that releases and re-polls at once cannot keep it; the main loop stays
responsive: operator commands are applied while waiting, a stop or pause ends the wait. While waiting the activity is
`waiting_for_gpu_lock`; `state.json["gpu"]` (held, held_for_s, waiting_for_s, acquisitions, releases, yields,
waited_s_total, longest_hold_s) is shown by `cam-live status` and the dashboard; every release / acquisition is a
`gpu_lock` event. Waiting is not charged to `limits.compute_s` (only learner calls are). The first version of this
work (a hold that exited and needed `cam-live resume`) was replaced by this one.
Tests (`tests/runtime/test_int3b_ops.py`): cycling with a fake tenant that takes the lock during a yield (the
runtime waits >= 1 s, keeps deciding, compute not charged), a stop ends a wait and the wait is visible in
`state.json`, a sleep releases the lock, a queued waiter gets its turn from a tenant that re-polls immediately, the
run service uses the lease with the real configuration by default.

**A — `--accept-ephemeral-storage`.** The doctor's `persistent_storage` check is unchanged and still blocks a live
preflight. `cam-live run --accept-ephemeral-storage` (also `cam-live preflight`, `cam-live resume`, and the internal
`_supervise`, to which `run` passes it) is the operator's explicit acknowledgement: effective only in mode live and
only for that check (`preflight.ACCEPTABLE_RISKS`); the item moves from `live.missing` to `live.accepted_risks` with
its finding; `run.json["accepted_risks"]` records the check, the finding, who accepted it (command line) and when;
`cam-live status` and the dashboard show it as ACCEPTED RISK. `resume` passes it only when given again on the resume
command line. Tests: blocks without the flag (synthetic doctor finding, and the real doctor of this machine where it
fails), accepted with it, nothing else acceptable, no effect in staging, recorded in run.json / preflight.json and
passed to the supervisor. `cam-live plan-report --out F` writes to any path the operator gives (directories created).

**B — debug validation.** `cam-live validate --mode staging --debug` passed the operator's real configuration to
`run_staging(real_model=False)`, which used it as given (Qwen3-4B on CUDA, no lock). `staging_config(..., False)` now
always builds the debug configuration (tiny backbone, debug core, CPU; only the state root is kept), and
`run_staging` holds the GPU lock whenever the configuration it will build puts a pretrained backbone on CUDA,
whatever `real_model` says. Tests in `tests/runtime/test_int3b_ops.py`.

## 3. Planning-phase rehearsal in staging (OP-11 prerequisites)

**Task** `tasks/staging/discord_100_planning_staging.yaml`: the operator's objective verbatim (same
`objective_confirmation`; the loader accepts it, `strategy_findings` = []), same success definition and exclusions,
interfaces, data permissions, stop conditions and approval rules as `tasks/discord_100_planning.yaml`; mode staging;
Discord = `harness/staging_discord` (a blank guild: one default channel, the bot, no members) on 127.0.0.1:18471; the
public web = a local server of neutral generic pages without any topic (`tools/planning_rehearsal.py`, index / three
test pages / a search form that finds nothing / a form) on 127.0.0.1:18472; the rehearsal's authorization policy is
the run policy with loopback removed from the staging mode's owned / pre-authorized destinations, so the web is
non-owned (navigation allowed, input queued for approval) and Discord writes and publications need approval;
`max_decisions` 30 (live: 120). Operator steps (`planning_rehearsal.py prepare`): a staging token in the run's vault
(bound to the staging address), the two local services registered, the configuration. Nothing was approved,
rejected or answered: approvals and operator requests were left as CAM's requests (the phase authorizes no execution).

**Configuration.** `configs/cam_live.yaml` (Qwen3-4B frozen bf16 + LoRA + bridge + 50m core), agent defaults
(backbone proposals on, `proposer_k` 8, rule `thompson_policy`), `cam-live run --inproc`, GPU lease 840 s.

**Segment A — default configuration** (`docs/evidence/INT3B/rehearsal_segmentA_*`; 12 decisions, 00:03-00:08 UTC,
stopped by the operator). The proposer prompt (every permitted tool listed, INT3A S-6b) was 4112 tokens with the
observation digest already shrunk to 94 characters, against the backbone's 2048-token prompt window
(`nn.max_prompt_tokens`): over the window in **12 of 12** decisions, the backbone wrote no parseable action
(0 valid), every chosen action came from the schema walker, CAM wrote no note and no operator request. **Finding:**
with the default window open-ended generation does not work on the planning task (change request to the
configuration owner / INT3A, `docs/change_requests/INT3B.md` INT3B-4).

**Segment B — `nn.max_prompt_tokens: 6144`** (fresh run directory; `docs/evidence/INT3B/rehearsal_segmentB_*`;
30 decisions = the decision budget, 00:11-00:48 UTC; prompts 4369-5303 tokens, never over the window). Measured
(`planning_rehearsal.py check`, `cam-live plan-report`):

* CAM received the objective verbatim: identical in `run_started`, in the learner view (`task.json`) and in all 30
  decision records.
* Backbone proposals: valid in 27 of 30 decisions (96 valid actions); **8 of the 30 chosen actions were written by
  the backbone** (6 `internal.request_operator`, `discord.list_events`, `discord.list_channels`), 22 by the schema
  walker.
* **Operator requests: 5 recorded** (kinds resource 2, capability 2, permission 1), 4 with a justification, 1 with an
  empty justification; a sixth identical request was returned as already open. All open (unanswered). Their texts
  are in the plan report verbatim (e.g. one requests "Member List Permission" — the task's `member_list: false`).
* **Notes: 0** — CAM never chose `internal.note`; so no plan, hypotheses or lessons were written in this rehearsal.
* Predictions: 2380 registered before their outcomes (16 candidates x queries per decision), 95 resolved;
  experiment registry: 6 records.
* **Every Discord write was queued and nothing executed without approval**: 3 `discord.modify_guild` queued for
  approval (pending), 1 `discord.listen_events` denied (data permission `member_list` is false), 0 executed without
  an approval; the staging Discord received 65 requests, **0 of them writes** (POST/PATCH/PUT/DELETE), counted at the
  service itself. Web: 3 `http.request` writes to the non-owned web denied, 4 `http.get` denied as repeats of a
  denied action.
* **Discord observations entered the experience store**: 4 observations with provenance `discord:staging` logged and
  all 4 found as items of the learner's store (by observation id); 8 experience records of Discord actions.
* **Learning updates logged**: 43 (27 fast, 15 frequent, 1 slow adapter), 16 with non-zero weight deltas; 31
  meta-update calls, 0 applied (28 insufficient independent outcomes, 3 no policy update yet) — expected: no verdict
  can be reached in the planning phase.
* GPU lease (in the process, no restart): holds of 775 s, **1234 s** and 171 s, waits 14 s, 85 s, 119 s. The 1234 s
  hold overran the 840 s bound by one cycle: the bound is checked between cycles, and one cycle took ~13 minutes
  while this machine was at load 68 (partly my own CPU jobs). A hold can exceed `gpu_hold_s` by one cycle.
* Speed: median 22 s between decisions, longest 788 s; the learner was CPU-bound in long cycles (GPU 0-25 % busy).

What this is not: no strategy was proposed by me or anyone; CAM's requests are its own; the rehearsal shows the
plumbing (objective -> CAM -> requests / queued actions / observations / learning, all recorded), not that CAM can plan
the community: in 30 decisions it wrote no plan or hypothesis note.

## 4. `cam-live plan-report`

`runtime/plan_report.py`, `cam-live plan-report [--run-dir R] [--out F] [--json]`: an export, not a summary. The
objective as CAM received it (with every record that delivered it and whether it is identical), CAM's notes (text as
saved by the notebook, joined to the decision that wrote it; grouped only by CAM's own tags), the predictions it
registered (with resolutions), every revision of its experiment registry, its operator requests (title,
description, justification, status, the operator's answer), every chosen action (executed / queued / refused, with
the runtime's result), approvals, and learning updates (counts by timescale, weight-delta norms per group, meta-updates
applied, reward versions). Provenance on every item: decision id, record id, time. The only additions are counts,
identifiers and times; the run's own redaction applies (the source is the run's records). Tests: verbatim texts
(newlines, quotes, unicode, pipes) and provenance on a run written by the runtime's own writers and the internal
executor's notebook / request file; output anywhere the operator says, Markdown or JSON.

## 5. Staging validation (LX-11, CM-04)

Changes (`validation.py`): the computer phase gets a compute limit (`compute_limit_s`, default 8 s per decision of
its budget: about half of the 16 s per decision measured on the real configuration in staging_20261127T223510Z) that
ends it before its decision budget — a resource limit legitimately reached first, checkpointed by the runtime;
`resource_limit_exercised` holds for any enforced limit except the decision budget (an action refused for it, or the
objective ended by it); LabWorld lifetimes drain (S-1) and settle, so each lifetime ends with the verifier's signed
terminal verdict (an independent verified group for the meta-learner) without one more decision;
`meta_min_independent` is a parameter (default 4, the earlier value); `cam-live validate` passes `--budget-s`,
`--meta-min-independent`, `--computer-share`; the lock wait is the queued lease (the 1800 s polling timeout of
`nn.gpu.gpu_lock` expired once behind another tenant).

**Run 1** (`docs/evidence/validation/staging_20261128T004747Z.json`; `cam-live validate --mode staging --real-model
--decisions 40 --budget-s 840 --meta-min-independent 3 --computer-share 0.2`, code fc96aeb; lock 546 s, peak
18.5 GB): **13 of 16 checks hold, not passed.** Now holding: `resource_limit_exercised` (computer phase ended by
`limit compute_s exhausted`, 71 s of 64 s at decision 4 of 8). Holding as before: computer-environment decisions, no
grant refusals, recovery (interruption + restart from disk in lifetime 2), logging, 4 verified lifetimes (3 closed:
failed 5/6, achieved 32/30, achieved 13/10; one still pending at its step budget), predictions before outcomes (167
resolved), internal rewards, experiential (51 non-zero) and adapter (13) weight changes, 52 delayed-outcome links,
behaviour change on the held-out context (TV 0.60) explained by the weights (TV 0.0 after restoring them).
Failing: `approvals_exercised` — none of the 4 computer decisions chose an action that needed approval (0 requested;
the compute limit shortened the phase); `meta_update_applied` / `meta_step_changes_policy` — 0 applied: 34 skips for
fewer than 3 independent groups, 5 effective-sample-size-too-small, 2 no policy update yet.

**Run 2** (`docs/evidence/validation/staging_20261128T005844Z.json`; the same entry point called with more
independent lifetimes: `run_staging(real_model=True, n_decisions=46, budget_s=840, computer_share=0.22,
lab_seeds=(101..106), lab_target=(6, 30, 10, 40, 8, 20), meta_min_independent=3, compute_limit_s=140)`; worktree at
3e29595 = INT3B fc96aeb + INT3A's later agent commits; lock 809 s, peak 18.7 GB, wall 892 s): **15 of 16 checks
hold, not passed.** Now holding as well: `meta_update_applied` — 2 reward meta-updates applied with non-zero
meta-gradients (norms 40.7 and 15.4, importance-weighted, 3 independent groups each; reward-parameter deltas 0.097
and 0.062; 50 skips for insufficient groups, 5 ESS too small) — and `meta_step_changes_policy` — the inner update
recomputed with psi after vs before the step changes the policy on the support decisions (mean TV 0.22 / 0.011, max
probability difference 0.24 / 0.019). 6 lifetimes, 5 closed by signed verdicts (2 achieved, 3 failed), one stopped
by the budget. Still failing: `approvals_exercised` — one approval was requested and approved by the validation's
operator, but the computer phase's compute limit ended the objective before the runtime processed it (an approved
action is executed only while no limit is exhausted), so nothing was executed after an approval; no rejection was
resolved by the operator. Configuration used for the meta checks, documented: `meta_min_independent_outcomes` 3 (the
live default is 8; 4 in the earlier validations) and six lifetimes of 6 decisions, each drained to a signed verdict.

## 6. Whole suite in one process (INT2-5)

`cam_live/pyproject.toml`: `addopts = "--import-mode=importlib"` (module names from the path: the two
`test_providers.py` no longer clash); `tests/conftest.py` appends the helper directories (`credit`, `runtime`,
`integration`, `nn`, `harness/env`, `harness/auth`, `harness/skills`) to `sys.path` and makes `tests/credit/conftest.py`
the module `conftest` at import, at the collection of every credit file and before every credit test (pytest itself
imports a conftest whose directory is on `sys.path` as top-level `conftest`: the first whole-suite run had 5 credit
failures `cannot import name 'build_record' from 'conftest' (tests/harness/skills/conftest.py)`). No file was renamed.
`pytest cam_live/tests --collect-only`: 1501 collected, 0 errors (before: 1492 collected, 1 error).

First whole-suite run (scratch, before the conftest fix): 1414 passed, 93 skipped, 9 failed in 2328 s — 5 the conftest
clash (fixed), 2 INT2 tests outdated by INT3A's parser fix of INT2-9 and by the operator's filled planning task
(updated), `test_int3a_lrn6` (INT3A's; the first promotion of its two-lifetime scenario was refused for want of a
held-out validation metric — the same scenario failed once and passed twice in `test_int2_checkpoints` with the
settle phase on, and passed with it off: trajectory-dependent, see §8), and `harness/compute/test_ec_jobs` (a job's
streamed output read as `'phase one'` vs `'phase one\n'` under load; not mine). Final run: see §9.

## 7. Claims (status F.json, M-int.json)

* **LX-07** experiment logging — implemented: experiments.jsonl from the integrated agent's `drain_experiments`
  (6 registry records in rehearsal B), learner-reported actuals kept apart (D10 test), plan report exports them.
* **LX-17** secondary metrics separate — implemented (verifier test); in staging only resource use and time exist.
* **LX-19** after verified success: stop with the learner state checkpointed (settle, forced commit) or the next
  queued objective — implemented, LabWorld tests only.
* **LX-20** durable, timestamped, provenance, redacted — implemented (tests; rehearsal token scan 0 matches; loopback
  IPv4 over-redacted).
* **CL-06** restart — runtime side: settle state restored, kill -9 tests, validation recovery check; G-cl's real-model
  embedding-cache difference not addressed (partial stays as D / M-int had it).
* **OP-05** Discord observations into the experience pipeline — partial: shown in staging (4 observations as store
  items, 8 records of Discord actions, learned from in the same run's frequent updates); nothing live.
* **OP-11** planning phase — partial: prerequisites rehearsed in staging (objective verbatim, requests, queued writes,
  learning); the live planning run was not started (not mine to start; needs the operator).
* **LX-11** staging validation — partial (run 2: 15 of 16 checks; approvals not exercised); **CM-04** — implemented.

## 8. Not done / limits

* **No live run.** The planning phase itself (`tasks/discord_100_planning.yaml`, live) was not started; nothing
  contacted discord.com. The rehearsal is staging only.
* **CAM wrote no notes** in 30 decisions of segment B (no plan, hypotheses or lessons); its operator requests are its
  only planning output. Whether more decisions or another configuration change this is not measured.
* **Open-ended generation needs a wider prompt window** than the default on the planning task (INT3B-4; INT3A has
  since shortened tool descriptions when the tool list alone exceeds the window — not re-measured by me).
* **Decision speed** on the real configuration: median 22 s, but single cycles took up to 13 minutes under load (the
  learner was CPU-bound; GPU 0-25 % busy). A GPU hold can overrun `gpu_hold_s` by one such cycle (measured 1234 s).
* **Staging validation not passed**: 15 of 16 checks in run 2; `approvals_exercised` fails (the one approved action
  came after the compute limit ended the objective — an approved action still waiting when an objective ends is left
  in state `approved`, never executed nor reported back to the learner: a small runtime gap, not fixed). The
  meta-update checks hold with a staging-scale threshold (3 groups) and six lifetimes; with 4 lifetimes (run 1) they
  did not.
* **Settle and promotions**: with the settle phase on, lifetimes run more learning cycles after their verdict; the
  two-lifetime promotion scenario of `test_int2_checkpoints` / `test_int3a_lrn6` failed once each (first promotion
  refused: "no independent validation metric supplied") and passed on repeats; a repeat study was started and stopped
  (CPU contention with the rehearsal), so the failure rate with and without settle is not measured.
* `harness/compute/test_ec_jobs::test_status_is_observable_while_running_and_cancel_stops_the_job` failed once under
  load (not my code).
