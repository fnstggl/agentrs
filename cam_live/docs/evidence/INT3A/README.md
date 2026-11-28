# INT3A evidence (2026-11-27/28)

Write-up: `docs/INT3A.md`. Real configuration = Qwen3-4B + LoRA + bridge + 50m core, under the shared GPU lock.

* `promotion_power.json` — item 6: power of the bootstrap-mix promotion test (one decision on n held-out groups) for
  the spread measured in `docs/evidence/C/real_pathway_20261127T200546Z.json`; groups needed for 80 % power.
* `probe_before_untraversable.json`, `probe_before_traversable.json`, `probe_after.json` — S-6b:
  `python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4` with the code of b663d5e (run root below a
  directory the sandbox uid cannot traverse, as S-6's run was; and a traversable root) and with this round's code
  (traversable root). Every prompt sent, the tools it listed, the raw backbone text, parsed / rejected actions.
* `profile_before.json`, `profile_after.json` — item 2: decisions of the real configuration through the runtime
  (SW:rle), every agent / component / environment / runtime method timed (CUDA synchronised), per runtime step.
  `profile_decision.py` is the script (run from a frozen copy of the code; `--voi-replay` recomputes the first VOI
  of the before-run on its captured inputs with the after-code).
