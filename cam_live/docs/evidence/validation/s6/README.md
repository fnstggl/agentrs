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
