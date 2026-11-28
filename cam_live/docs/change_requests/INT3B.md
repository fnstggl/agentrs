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
