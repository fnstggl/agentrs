# Evidence of RW (amendment item 1, genuinely learned rewards) — see `docs/RW.md`

Scripts (run from `/workspace/camlive`; `PYTHONPATH=cam_live/src`, or a copy of the tree as named in `docs/RW.md`):

| file | what |
|---|---|
| `probe_long.py` -> `probe_before.json` | the audit run on the pre-RW reward code: one 40-decision LabWorld objective through the integrated agent and runtime (debug), every reward, every unit's verified progress, every meta-update skip |
| `rw_meta_diag.py` -> `diag_*.json` | one long LabWorld objective per seed with `MetaLearner` acting itself (debug, controlled configuration): arms pooled / query x direct / return_to_go x inner half-life, frozen ablation |
| `summarise_diag.py`, `table_diag.py` -> `diag_summary*.json` | per seed: mix changes, agreement, divergence from the frozen arm, ground truth of the chosen actions after the change |
| `rw_credit_probe.py` -> `credit_probe_*.json` | which support rows the outer gradient asks R_psi to reward (per-row derivative of the first-order surrogate) |
| `promotion_power_sim.py` -> `promotion_power.json` | pairs needed by the validity rule's own test before it lowers the mix |
| `rw_pathway.py` -> `real_*.json`, `dbg_*.json` | the integrated agent (`CamLiveAgent` + `LiveRuntime`) in one long LabWorld staging objective; `--debug` = CPU plumbing check; without it the real configuration under the shared GPU lock; arms learned / frozen |
| `check_hooks.py`, `check_hooks.log`, `rw_agent_hooks.diff` | change requests RW-1 / RW-2 tested on a patched copy |
