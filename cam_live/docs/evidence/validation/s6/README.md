# S-6: backbone-written actions on an open-ended software objective (real configuration, 2026-11-27)

`probe_sw_20261127.json` — `python -m cam_live.eval.proposal_probe --family SW --problem rle --rounds 4` (S's probe:
calls `agent.propose` as the runtime does), run from a frozen copy (`git archive`) of commit d55986d, Qwen3-4B +
50m core under the shared GPU lock, 19–29 s per round.

* 16 actions written in 4 rounds, **16 parsed as valid candidates**, 0 rejected; 6 needed the program form lifted
  (`{"operation": "program", "arguments": {"steps": [...]}}`), nothing else was normalised. Prompt 3615 characters.
* What was written: `program`s of `internal.inspect` steps, `internal.note`s restating the task, one `wait`, and in
  every round an `internal.request_operator` asking for the capability to write `solution.py`. No proposal wrote or
  ran code (`sandbox.write_file` / `sandbox.python` are among the task's 16 tools; the probe does not record which 12
  tools each round's prompt showed).
* So: proposals are now executable; the software objective was not attempted by any backbone proposal here.

`oe_seq_20261127.json` — `python -m cam_live.eval.oe_runner sequence --condition E --seed 0 --plan SW:rle,MECH:7
--decisions 16` (S's runner: the integrated agent through the runtime, one objective after another), same snapshot
(d55986d), real configuration, each objective stopped by the runner's lock-hold budget (~790 s; 110–150 s per
decision on the shared machine), 3341 s in total:

| objective | decisions | executed | chosen candidates written by the backbone | outcome |
|---|---|---|---|---|
| SW rle (write `solve` in `solution.py`) | 7 | 7 / 7 | 3 / 7 (`wait`, `internal.inspect` x2) | `solution.py` never written: every verifier check "missing"; not attempted |
| MECH 7 (measure an unknown function over HTTP) | 6 | 5 / 6 | 5 / 6 (`http.get` measurements at chosen points, notes of the values, one `program`) | measurements taken; objective not reached in 6 decisions; the `program` was refused (INT2-9, fixed in the agent after this run) |

Valid-and-executed: 12 of 13 chosen actions executed; of the 8 chosen backbone actions 7 executed. The software
objective was not attempted on the real configuration in this run.
