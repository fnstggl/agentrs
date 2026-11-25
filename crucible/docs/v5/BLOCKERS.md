# Crucible 5.0 blockers and risks

A blocker is something the project cannot resolve itself. A requirement marked BLOCKED in the manifest must cite an
entry here by id. Risks are listed so they are watched; they are not blockers.

## Open blockers

None recorded yet.

## Risks being managed

* **R-1 Disk.** 26 GB free on a 300 GB overlay that is 92% full (another worktree holds 165 GB of checkpoints). Datasets,
  encoder weights, worktrees and long-run logs must fit. Budgets in INTERFACE_CONTRACTS section 0.
* **R-2 Persistence.** `/workspace` is not a persistent volume. Pushing to GitHub needs a credential the user provides
  for each push; until then the only durable copy of new work is local. Local bundles are written to `/workspace/backups`.
* **R-3 Shared compute.** One 11.5-CPU quota and one GPU shared with other tenants; schedule through slots.
* **R-4 Dataset access.** KuaiRand (Zenodo), MIND (Azure blob) and Criteo (go.criteo.net) host the calibration data;
  first probes returned 200 for Zenodo and Hugging Face, and 400/404 for the bare MIND and Criteo hostnames (file URLs
  still to be tried). If a dataset is unreachable, an alternative legitimate dataset is used and recorded.
* **R-5 Scale of the assignment.** The specification is far larger than one integration cycle; the manifest and the
  checker keep the status honest (IN_PROGRESS) until every requirement is verified.
