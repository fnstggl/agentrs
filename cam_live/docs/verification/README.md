# Independent verification (workstream G)

Verifiers did not write the code they check. Each lens reads the operator specification, the architecture
contract and the claims in `docs/status/<workstream>.json`, then inspects the actual code, runs the tests itself
and writes its own probes, with the aim of **refuting** each claim.

* `G_<lens>.md` — what was inspected and executed, defects with file:line and a reproduction command.
* `verdicts_<lens>.json` — `{requirement id: {"status": "verified" | "partial" | "failed" | "unverified",
  "by": "G-<lens>", "notes": "...", "evidence": ["command or file", ...]}}`.
* `repro/<lens>/` — scripts that reproduce a defect or demonstrate a property (run from `/workspace/camlive`).
* `verdicts.json` — merged by `tools/merge_verdicts.py` (worst status wins when two lenses judge one requirement);
  read by `python -m cam_live.verify_completion --merge-status`.

A verifier never edits source or other workstreams' tests; defects are reported, the owning side fixes them, and
the requirement is re-verified.
