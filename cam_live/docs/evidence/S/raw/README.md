# Raw evaluation results preserved before disk cleanup (2026-11-27 ~18:00 UTC)

`eval_results_20261127.tar.xz` — every result JSON, log and evidence file of `runs/cam_live/eval/` as it stood when
the 300 GB disk filled (~10:16 UTC): `v2/results/{real,main_cpu,frozen_cpu,integrity,meta_main_cpu,scaling,
transfer_cpu}`, `v2/evidence`, `v2/logs`, and the pilot `results/`, `logs/`. Status of these runs:

* `results/` (pilot, seeds 900–901, snapshot a8444e6): **superseded / invalid** — the code predates the objective
  registration fix (C-10, 3597af2), so full-CAM conditions could not apply reward meta-updates; used only to set q.
* `v2/` (snapshot 00cb5a4, includes C-10): interrupted when the disk filled; partial; not analysed. In the real-model
  E run seed 0 the reward-parameter group hash is identical before and after 2 lifetimes (no applied meta-update).

Checkpoints of these runs were pruned with `cam_live/tools/prune_runs.py` (plan in `prune_report_20261127.json`):
pre/post weights and weights-only exports kept, per-lifetime runtime checkpoints and optimizer states removed.
The per-run JSONL records (decisions, predictions, learning/meta updates, experience stores) remain on the instance
under `runs/cam_live/eval/` and are not in git.
