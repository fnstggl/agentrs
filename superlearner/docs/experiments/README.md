# Measured results (raw)

| File | What |
|---|---|
| `exp_b_multi.json` | Experiment B: before/after RL, three seeds, two algorithms, per-task outcomes |
| `exp_final.json`, `final_tasks.json` | the sealed final list and its one evaluation |
| `baselines_exp_b_fewdemo_sil_s0.json`, `baselines_memory_on_sil_s0.json` | Part 5 comparison; memory ablation |
| `exp_a_real_actions.json` | Experiment A, with the action sequence of every episode |
| `exp_c_*.json`, `exp_d_retention.json`, `exp_d_clone.json` | transfer; retention (RL and supervised variants) |
| `exp_worldmodel_*.json`, `exp_selfplay*.json` | world model; several learners in one world (two worlds) |
| `exp_e_meta.json` | summary of the meta-search: every screened candidate, the inert-change probe, selection analysis |
| `meta_search4/` | the search's ledger, configuration, provenance, reports, and the code diff of every candidate |
| `curves/` | in-training evaluation records and configurations of the training runs; sealed-list records of the 13-arm learner |
| `NOTES.md` | chronological lab notes, including superseded attempts, the outage and the recovery |
| `render.py` | prints the tables of `../RESULTS.md` from these files (`python render.py --fill ../RESULTS.md`) |

Run directories with checkpoints, trajectories and candidate source trees are under `runs/` and are not committed.
