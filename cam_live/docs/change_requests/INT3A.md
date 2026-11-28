# Change requests and status notes of INT3A (final decision-side integration fixes)

Each request has the reason, the exact diff and what INT3A did meanwhile in its own files.

## INT3A-1 (tests/integration/test_int2_queries.py, owner INT3B): the INT2-9 test pins the parser defect

**What / why.** INT2-9 is now fixed where it belongs (`predict/candidates.py::_build`: a parsed program gets the
modality of the environment's `program` tool, internal when none is described). The INT2 test
`test_backbone_program_takes_the_modality_of_the_program_tool` asserts the *defect* first
(`assert progs and progs[0].modality == "api"  # what the parser gives it`, line 71) and then that the agent's
workaround corrects it, so it fails after the fix (measured: 1 failure in the integration directory, nothing else).
**Proposed diff.**
```diff
-    assert progs and progs[0].modality == "api"             # what the parser gives it
+    assert progs and progs[0].modality == "internal"        # INT2-9 fixed in the parser (the program tool's modality)
+    import dataclasses
+    defective = dataclasses.replace(progs[0], modality="api")  # the form the parser produced before INT2-9
     task = TaskSpec(...)
 ...
-    verdict, why = judged(progs[0], "auth1")
+    verdict, why = judged(defective, "auth1")
     assert verdict == "deny" and "T-PROGRAM" in why
 ...
-        (a,) = agent._feasible(progs, tools, clock.now(), {})
+        (a,) = agent._feasible([defective], tools, clock.now(), {})   # the agent still repairs such an action
```
**Meanwhile.** The agent's workaround in `CamLiveAgent._feasible` is kept (behaviour identical; it is now a no-op for
parser-built programs and still repairs a composition from any other source). The regression test of the fix is
`tests/predict/test_int3a_program_modality.py` (fails on the old parser).

## INT3A-2 (eval/openended.py `OpenEndedComputer.open`, eval/oe_runner.py; owner of eval/): refuse to run an
objective whose permitted tools are unavailable

**What / why (measured, the S-6b root cause).** The sandbox is available only when its workspace can be traversed by
the sandbox uid (`harness/sandbox.py`: "workspace ... is below a directory the sandbox uid 996 cannot traverse").
S-6's real-model probe and sequence run were run from a frozen copy under the session scratch directory
(`/tmp/claude-0/...`, mode 0700): every `sandbox.*` tool was `available=False`, the agent (correctly) shows only
available tools, so the backbone was shown no way to write or run code in a code objective and asked the operator
for "the capability to write solution.py" — which was the true state of that computer. Nothing reported it: the run
looked like an agent failure. The evaluation currently running (`CAM_LIVE_EVAL_HOME=/workspace/camlive/runs/...`)
is traversable and not affected; any run rooted under a 0700 directory is.
**Proposed diff** (`OpenEndedComputer.open`, after the executors are registered):
```diff
+        allowed = set(task.interfaces or [])
+        missing = {t.name: t.unavailable_reason for t in self.env.discover_tools()
+                   if not t.available and t.modality in allowed and not t.name.startswith(UNUSED_PREFIXES)}
+        if missing:
+            raise RuntimeError("tools the task permits are unavailable: " +
+                               "; ".join(f"{k}: {v}" for k, v in sorted(missing.items())))
```
(`cmd_transfer` already does `os.chmod(work, 0o755)` for its temporary roots; `cmd_sequence` should check the same.)
**Meanwhile.** The agent records, per decision, the permitted tools the environment reports unavailable with the
reason (`decision.diagnostics["generation"]["tools_unavailable"]`, counter
`decisions_with_unavailable_permitted_tools`); `python -m cam_live.eval.proposal_probe` stops with that list unless
`--allow-unavailable`.

## INT3A-3 (tests/integration/test_int2_prompts.py, owner INT3B): optional extension, no failure

The prompt the agent sends still decomposes into the constant instruction text + the objective verbatim + the tool
descriptors + the observation digest (the test passes). New content inside the digest (S-6b): the task's success
definition as one line `[task] success definition: <the task's success dict, JSON>` and one line per executed
action `[result] <operation> -> <status> (<failure class>): <the environment's detail>`. Suggested addition to the
test: assert the success line is present and decodes to `task.success`.

## Status notes (requests of other workstreams handled in INT3A's files)

* **INT2-9** — done: `predict/candidates.py::_build` (program = the program tool's modality; steps keep theirs and are
  bounded by the tools given / the agent's interface check). Test `tests/predict/test_int3a_program_modality.py`.
* **LRN-3** — done: `reward_signals.epistemic_after` (Predictor's goal-value epistemic std on the decision-time head
  view after the fast layer absorbed every goal-relevant outcome of the decision) and `reward_signals.learning_progress`
  (world-model NLL on the decision's matured outcomes before minus after the next frequent update) are written as
  added annotation keys; the assembler reads `reward_signals.<name>` keys. Test
  `tests/integration/test_int3a_bootstrap_inputs.py`.
* **LRN-6** — confirmed done (INT2, 07be9ba): a refused promotion keeps R_psi (whole module state) and the meta
  optimizer, and the policy head unless `decision_agreement` is a reason of the refusal. INT3A's regression test
  `tests/integration/test_int3a_lrn6.py` changes psi, the meta optimizer state and the policy head after the last
  promotion and fails on 78ad939 (psi restored by the rollback); passes now.
* **KF-E01 / KF-E02 / KF-E05** (documentation audit) — see `docs/INT3A.md`; tests `tests/integration/test_int3a_audit.py`.
* `tests/integration/test_int2_planning.py::test_planning_task_file` fails at HEAD independently of INT3A: the task
  file was filled with the operator's values by M (d3ce519), so the "unset items" problem it expects is gone
  (owner of the test: INT3B).

## INT3A-4 (configs/cam_live.yaml or nn/presets.py `NNConfig.max_prompt_tokens`, owner A / M): a prompt window that fits the authorized action space

**What / why (measured).** The backbone keeps the END of a prompt longer than `nn.max_prompt_tokens` (2048), so the
instruction block, the objective and the tool list are cut first. INT3B's planning-phase rehearsal measured 4112
prompt tokens after the agent's digest fit (`docs/evidence/INT3B/rehearsal_summary.json`, segment A: 12 of 12
decisions over the window, no backbone proposal); with `nn.max_prompt_tokens: 6144` (segment B) 4369-4757 tokens
fitted and 8 backbone-written actions were chosen. The open-ended families (14-16 tools) are close to 2048 too.
**Proposed diff** (`configs/cam_live.yaml`, section `nn`): `max_prompt_tokens: 6144` (Qwen3-4B's context is 32k;
generation cost grows with the prompt: measure before / after on the real configuration).
**Meanwhile (agent).** `_fitted_digest` shortens the observation digest, then the tool descriptions (100 characters,
then none) — never the tool list, the objective or the instructions — and records `prompt_fit` (tokens, window,
`over_window`) per decision and the counter `decisions_prompt_over_window`.
