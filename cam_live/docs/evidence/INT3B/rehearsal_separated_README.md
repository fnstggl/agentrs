# Separated staging rehearsal of the planning phase (master, 2026-11-28 06:05-06:25 UTC)

First run of the planning task's code path under the **separated deployment** with the **real configuration**
(Qwen3-4B + 50m, open-ended generation on): `cam-live run --task tasks/staging/discord_100_planning_staging.yaml`
without `--inproc` (supervisor root; services under camauth / camverify / cambrowser / camsbx; learner camlearn),
staging Discord and neutral local web from `tools/planning_rehearsal.py serve`. Code: HEAD 59807fc (with the GPU-lock
read-only fallback; before it the learner could not open the root:root 0664 machine lock).

* The learner (camlearn) waited 849 s in the shared lock's queue, took it, loaded Qwen3-4B from the shared HF cache
  (one unreadable tree-cache metadata file, ignored by transformers) and ran 8+ decisions (~20 s each).
* Discord writes (`discord.modify_guild`, `discord.create_channel`) were queued for approval; the staging Discord
  received 0 write requests. Browser navigations to the neutral web executed. No notes and no operator requests by
  CAM in these 8 decisions.
* `cam-live plan-report --out /root/camlive_reports/...` exported the report (copy: `rehearsal_separated_plan_report_20261128.md`).
* `cam-live stop`: stopped, learner state checkpointed.
* Artefact of the rehearsal tool (not of the operator's launch path): `planning_rehearsal.py prepare` registers the
  staging services through an in-process PolicyService as root, so `<run>/auth/state.sqlite`, `audit.head`,
  `keys/grant.key`, `keys/audit.key` were root-owned; the separated auth service (camauth) could not open them until
  the master chowned `<run>/auth` to camauth. `tools/launch_live.sh` creates none of these files.

Check output: `rehearsal_separated_20261128.json`.
