# INT3A evidence (2026-11-27/28)

Write-up: `docs/INT3A.md`. Real configuration = Qwen3-4B + LoRA + bridge + 50m core, under the shared GPU lock.

* `promotion_power.json` — item 6: power of the bootstrap-mix promotion test (one decision on n held-out groups) for
  the spread measured in `docs/evidence/C/real_pathway_20261127T200546Z.json`; groups needed for 80 % power.
* `probe_before_untraversable.json`, `probe_before_traversable.json`, `probe_after.json` — S-6b:
  `python -m cam_live.eval.proposal_probe --plan SW:rle,MECH:7 --rounds 4` with the code of b663d5e (run root below a
  directory the sandbox uid cannot traverse, as S-6's run was; and a traversable root) and with this round's code
  (traversable root). Every prompt sent, the tools it listed, the raw backbone text, parsed / rejected actions.
* `runtime_sw_rle_profile.json` — the agent of this round (snapshot of 8cebdb0's code, old VOI path) through the real
  runtime on SW:rle (`LiveRuntime`, the open-ended runner's `RuntimeConfig`): every chosen action, the verifier's
  score history, and the time of every agent / component / environment / runtime method per runtime step (CUDA
  synchronised; inclusive times). The objective was verified achieved after 2 decisions.
* `profile_mech_before.json`, `profile_mech_after.json` — item 2: MECH:7 through the runtime, the same code except the
  VOI change (before / after), in one lock hold; `voi_replay` in the after-file recomputes the first VOI call of the SW
  run on its captured inputs with the after-code (old vs new numbers on the same inputs).
* `profile_decision.py` — the script of both (run from frozen copies of the code under `runs/INT3A/`).
* `<INT3A scratch>` in the files = the session scratch directory `/tmp/claude-0/.../scratchpad/INT3A` (below a 0700
  directory: the reason the sandbox was unavailable in `probe_before_untraversable.json`).
