# Change requests from INT3B (final runtime-side integration)

Each has a workaround or is informational; none blocks the runtime.

## INT3B-1 (eval, `eval/oe_runner.py`, `eval/runner.py`, owner S): use the in-process GPU lease

**What / why (measured 2026-11-27 23:20-00:00 UTC).** `python -m cam_live.eval.oe_runner sequence ...` held the shared
GPU lock file continuously for over 70 minutes (`/proc/locks`, one FLOCK entry of the same pid), taking one hold per
objective and re-acquiring with non-blocking polls immediately after each release. Another tenant that polls the same
lock (`nn.gpu.gpu_lock`, every 5 s) practically never gets a turn between two holds. The INT3B rehearsal learner and
the staging validation waited behind it.
**Proposed.** Take the lock through `cam_live.runtime.gpu_cycle.GpuLease` (a queued blocking `flock` in a helper
thread, so a waiter is woken by the kernel at the release), or yield for `runtime.gpu_yield_s` (15 s) after each
release before re-acquiring.

## INT3B-2 (tests, owners B and D): the INT2-5 renames are no longer needed

`pytest cam_live/tests` now collects and runs in ONE process: `cam_live/pyproject.toml` sets
`addopts = "--import-mode=importlib"` (module names from the path: the two `test_providers.py` no longer clash) and
`tests/conftest.py` appends the directories whose sibling helpers tests import (`rt_helpers`, `int_helpers`,
`ea_helpers`, `eb_support`, `es_support`, `e_fix_support`, `nn_realistic_inputs`, `int2_factory`) and makes
`from conftest import build_record` always mean `tests/credit/conftest.py`. If owners prefer explicit names, the
renames of INT2-5 still apply and the alias in the root conftest can then be dropped.

## INT3B-3 (agent, `agent.py`, owner INT3A): nothing in the store marks an outcome that arrived after the verdict

Outcomes measured while an ended objective settles (`runtime.loop`, `settle_after_finish_s`) reach
`agent.reconcile` in the same lifetime (no `set_objective` in between), after the signed terminal verdict was
reconciled. The agent links them and resolves their predictions like any other; the meta-learner's view of "outcomes
after the verdict" is LRN's/INT3A's to decide. Informational: no change is required for correctness; a field on the
record (`OutcomeObservation.source` already says `verifier`) is enough to tell them apart in an analysis.

## INT3B-4 (configuration `configs/cam_live.yaml` section `nn`, owner master; agent prompt INT3A): the proposer prompt does not fit the default window on the planning task

**Measured (rehearsal segment A, real configuration).** With every permitted tool listed (S-6b), the planning task's
proposer prompt is 4112 tokens even after the observation digest is shrunk to 94 characters; the backbone keeps
`nn.max_prompt_tokens` = 2048 tokens (the END of the prompt: instructions, objective and most tools are cut). 12 of 12
decisions were over the window and the backbone produced 0 valid actions: no note, no operator request could be
written. With `nn: {max_prompt_tokens: 6144}` (segment B) the prompts were 4369-5303 tokens and 27 of 30 decisions had
valid backbone proposals (96 actions).
**Proposed.** `nn: {max_prompt_tokens: 6144}` in `configs/cam_live.yaml` for the planning run (Qwen3-4B's context is
32k; nvidia-smi sampled the segment-B learner at ~10 GB GPU memory early in the run — no peak was recorded), or a compact tool list for large tool sets.

## INT3B-5 (tests, `tests/eval/test_meta_integrity.py`, owner S): a tolerance that depends on the process's torch thread count

In the whole suite run in ONE process (INT3B final run, 2026-11-28) `test_meta_checks_on_a_saved_run` failed with
`1.10e-07 <= 1e-05 * 0.00902` (magnitude-only reward change vs update norm); alone it passes (98 s). The module sets
`torch.set_num_threads(2)` at import, which other modules of the same process change (the last setter wins), and the
bound (1e-5 of the update norm) is tighter than the reduction-order noise of a different thread count.
**Proposed.** Set the thread count inside the test (or a fixture) rather than at import, or bound by an absolute
floating-point floor (e.g. `max(1e-5 * norm, 1e-6)`).

## INT3B-6 (authorization service, `harness/auth/policy.py`, owner H): an approval's grant reserves, and the identical re-authorization reserves again (G2I-15)

**Measured (G2-integration run B, `p08_approval_trace.py`).** `resolve_approval(..., approve=True)` ends in
`authorize`, which issues a grant and reserves the action's budget (`grants.reserve_json`, counted by
`limits.check` as "incl. reserved"). The runtime, told by the operator's notice that the approval was resolved,
asked `authorize` again for the identical action: a second grant, a second reservation, and with
`max_external_actions` 2 (one used by the harness's navigation) the service refused it — `used 2 incl. reserved,
requested 1, limit 2` — so the approved action was never executed. The first grant stays reserved until its TTL.
**Done in the runtime (INT3B, owned files).** `OperatorControl` (the path of `cam-live approve` / the dashboard)
relays the approval's grant with its signed `approval_notice`; the runtime checks it with the service
(`check_grant(..., consume=False)`) and executes the action with it, without a second `authorize`
(`test_the_operators_approval_grant_is_used_so_the_budget_is_not_reserved_twice`). The run log names the relayed
grant without copying its token.
**Still open in the service.** An operator who approves with another tool that does not relay the grant (or a
runtime that never received the notice) still re-authorizes and reserves twice. **Proposed.** In `_authorize`, when
an APPROVED approval of the same binding already has an `issued`, unused grant, close (or return) that grant instead
of issuing and reserving a second one; or release the unused grant's reservation when a newer grant for the same
approval is issued.

## INT3B-7 (validation, informational): what the default staging validation can and cannot guarantee

`approvals_exercised` needs the learner to choose, in the approvals objective, at least two interactions with the
staging site that need the operator's approval (the first is approved and must execute, the second is rejected).
The validation never steers that choice (operator rule: CAM authors its own plan), so a run in which the learner
makes fewer than two such requests fails the check honestly. The time box (G2I-11) cannot pre-empt a learner call
once started: the first call of a kind, or one much slower than the five before it, can end after the deadline.
